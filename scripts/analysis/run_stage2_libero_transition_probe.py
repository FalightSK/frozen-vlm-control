"""Record per-step object and goal-predicate evidence on selected LIBERO starts.

Diagnostic only: loads existing epoch-120 heads, does not train, and checks each
rerun against the corresponding locked final-test record. Outputs should be
treated as selected examples, not failure-rate estimates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TASK_STARTS = {6: {3: 13, 4: 11, 5: 15}, 9: {3: 28, 4: 11, 5: 21}}
STUDY = "libero_progression_120_v1"
PLAN = "c6854e3ff020"


def stage_snapshot(wrapper, step: int, action: np.ndarray) -> dict:
    domain = wrapper.env
    goals = domain.parsed_problem["goal_state"]
    objects = domain.object_states_dict
    state = goals[0] if len(goals) == 1 else None
    result = {"step": step, "gripper_action": float(action[-1]),
              "goal_predicates": [bool(domain._eval_predicate(p)) for p in goals]}
    if state is None or len(state) != 3:
        return result
    predicate, object_name, target_name = state
    obj_state, target_state = objects[object_name], objects[target_name]
    obj = obj_state.get_geom_state()
    target = target_state.get_geom_state()
    obj_pos = np.asarray(obj["pos"], dtype=float)
    target_pos = np.asarray(target["pos"], dtype=float)
    delta = obj_pos - target_pos
    sim = domain.sim
    obj_body = domain.obj_body_id.get(object_name)
    target_parent = getattr(target_state, "parent_name", target_name)
    target_body = domain.obj_body_id.get(target_parent)
    contacts = []
    def is_object_body(body_id):
        if obj_body is None:
            return False
        while body_id > 0:
            if body_id == obj_body:
                return True
            body_id = int(sim.model.body_parentid[body_id])
        return body_id == obj_body
    for contact in sim.data.contact[:sim.data.ncon]:
        geom_ids = (int(contact.geom1), int(contact.geom2))
        geom_names = [sim.model.geom(g).name for g in geom_ids]
        body_ids = [int(sim.model.geom_bodyid[g]) for g in geom_ids]
        if any(is_object_body(b) for b in body_ids):
            contacts.append(geom_names)
    target_contact = None
    region_inside = None
    if getattr(target_state, "object_state_type", None) == "site":
        site = domain.object_sites_dict[target_name]
        site_pos = np.asarray(sim.data.get_site_xpos(target_name), dtype=float)
        site_mat = np.asarray(sim.data.get_site_xmat(target_name), dtype=float).reshape(3, 3)
        region_inside = bool(site.under(site_pos, site_mat, obj_pos))
        parent = domain.get_object(target_parent)
        target_object = domain.get_object(object_name)
        target_contact = bool(domain.check_contact(parent, target_object))
        delta = site_mat.T @ delta
    elif predicate == "On":
        target_contact = bool(obj_state.check_contact(target_state))
    result.update({"predicate": state, "object_position": obj_pos.tolist(),
                   "target_position": target_pos.tolist(), "target_relative_position": delta.tolist(),
                   "target_distance": float(np.linalg.norm(delta)),
                   "target_xy_distance": float(np.linalg.norm(delta[:2])),
                   "target_contact": target_contact, "inside_target_region": region_inside,
                   "object_contact_pairs": contacts})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite probe output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    partial = args.output_dir / "stage2_transition_probe.jsonl"

    from configs.registry import get_config
    from envs.libero_env import LiberoAgent, build_conditions, load_init_states, make_env, run_episode
    from experiment_protocol import checkpoint_config, file_digest, sampling_seed, seed_everything
    from libero.libero import get_libero_path
    from libero.libero import benchmark
    from scripts.evaluate import deterministic_inference
    from scripts.eval_libero import load_policy

    deterministic_inference()
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    rows = []
    for exp, backbone in (("exp05", "groot"), ("exp06", "qwen3vl")):
        for seed in (3, 4, 5):
            rel = f"{STUDY}/{PLAN}/libero/{exp}/seed_{seed:03d}"
            result_path = ROOT / "asset/durable/libero/evaluation_results" / rel / "final_test.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            checkpoint_path = ROOT / "asset/durable/libero/evaluation_checkpoints" / rel / "final_test.pt"
            checkpoint_sha = file_digest(checkpoint_path)
            if checkpoint_sha != result["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            cfg = checkpoint_config(get_config("libero", exp), checkpoint, "libero", verify_live_encoder=True)
            vlm, head = load_policy(cfg, checkpoint_path, args.device)
            agent = LiberoAgent(vlm, head, cfg, args.device)
            original = {(r["task"], r["initial_state_id"]): r for r in result["results"]
                        if r["condition"] == "canonical"}
            for task_id, starts in TASK_STARTS.items():
                start_id = starts[seed]
                expected = original[(task_id, start_id)]
                if expected["initial_state_sha256"] == "":
                    raise ValueError("Missing original initial state hash")
                task = suite.get_task(task_id)
                bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                env = make_env(str(bddl), cfg)
                stage_trace = []
                try:
                    init_states = load_init_states(task)
                    policy_seed = sampling_seed("libero", seed, task_id, start_id)
                    seed_everything(policy_seed)
                    def capture(step, wrapped, action, done):
                        stage_trace.append(stage_snapshot(wrapped, step, action))
                    outcome = run_episode(env, agent, cfg, task.language,
                        seed=start_id, record=True, init_state=init_states[start_id],
                        trace_callback=capture)
                finally:
                    env.close()
                success = bool(outcome["success"])
                if (outcome["initial_state_sha256"] != expected["initial_state_sha256"]
                        or policy_seed != expected["policy_sampling_seed"]):
                    raise ValueError(f"Rerun start or policy seed mismatch: seed={seed}, task={task_id}, "
                        f"start={start_id}, state={outcome['initial_state_sha256']==expected['initial_state_sha256']}, "
                        f"sampling_seed={policy_seed==expected['policy_sampling_seed']}")
                import imageio.v2 as imageio
                video = args.output_dir / f"task{task_id}_seed{seed}_{backbone}_start{start_id}.mp4"
                imageio.mimwrite(video, outcome.pop("frames"), fps=cfg.video_fps)
                rows.append({"dataset": "libero", "task": task_id, "instruction": task.language,
                    "training_seed": seed, "backbone": backbone, "initial_state_id": start_id,
                    "policy_sampling_seed": policy_seed, "success": success,
                    "original_success": bool(expected["success"]),
                    "initial_observation_matches": outcome["initial_observation_sha256"] == expected["initial_observation_sha256"],
                    "original_steps": expected["steps"], "rerun_steps": outcome["steps"],
                    "video": str(video), "trajectory": stage_trace})
                with partial.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rows[-1], separators=(",", ":")) + "\n")
                print(f"{backbone} seed {seed} task {task_id} start {start_id}: "
                      f"success={success} (original {expected['success']}), steps={outcome['steps']} "
                      f"(original {expected['steps']}), initial image match="
                      f"{outcome['initial_observation_sha256'] == expected['initial_observation_sha256']}", flush=True)
            del agent, head, vlm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    out = args.output_dir / "stage2_transition_probe.json"
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} verified instrumented episodes to {out}")


if __name__ == "__main__":
    main()
