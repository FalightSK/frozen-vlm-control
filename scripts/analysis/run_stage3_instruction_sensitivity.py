"""Test whether the LIBERO policies pursue a changed, valid goal instruction.

Uses the existing epoch-120 heads and held-out LIBERO fixed starts. On each
start, the requested instruction is swapped between the cheese-in-bowl and
wine-on-rack tasks. The scorer checks the goal named by the instruction while
also recording whether the original scene goal was reached first. No training.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

if sys.platform == "win32" and os.environ.get("MUJOCO_GL", "").lower() != "wgl":
    os.environ["MUJOCO_GL"] = "wgl"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

STUDY = "libero_progression_120_v1"
PLAN = "c6854e3ff020"
EXPERIMENTS = {"groot": "exp05", "qwen3vl": "exp06"}
GOALS = {
    6: ["on", "cream_cheese_1", "akita_black_bowl_1"],
    9: ["on", "wine_bottle_1", "wine_rack_1_top_region"],
}
SWAP = {6: 9, 9: 6}


def read_completed(path: Path) -> tuple[list[dict], bool]:
    rows, damaged = [], False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.endswith("\n"):
                    damaged = True
                    break
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    damaged = True
                    break
    except (EOFError, OSError):
        damaged = True
    return rows, damaged


def write_progress_checkpoint(output_dir: Path, rows: list[dict], seeds: list[int],
                              tasks: list[int], starts: list[int], planned: int,
                              raw_path: Path, include_sha: bool = True) -> None:
    completed = {(r["training_seed"], r["backbone"], r["task"], r["initial_state_id"])
                 for r in rows}
    groups, remaining = [], []
    for seed in seeds:
        for backbone in EXPERIMENTS:
            for task in tasks:
                done = sorted(s for se, bb, ti, s in completed
                              if (se, bb, ti) == (seed, backbone, task))
                todo = sorted(set(starts) - set(done))
                groups.append({"training_seed": seed, "backbone": backbone,
                               "task": task, "completed_start_ids": done})
                if todo:
                    remaining.append({"training_seed": seed, "backbone": backbone,
                                      "task": task, "missing_start_ids": todo})
    checkpoint = {
        "study": STUDY, "plan": PLAN,
        "condition": "swapped; cheese-bowl <-> wine-rack instructions",
        "planned_episodes": planned, "completed_episodes": len(completed),
        "checkpoint_file": raw_path.name,
        "checkpoint_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest() if include_sha else None,
        "completed_groups": groups, "remaining": remaining,
    }
    tmp = output_dir / "progress_checkpoint.tmp.json"
    tmp.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, output_dir / "progress_checkpoint.json")


def goal_snapshot(domain, predicate: list[str]) -> dict:
    object_name, target_name = predicate[1:]
    object_state = domain.object_states_dict[object_name]
    target_state = domain.object_states_dict[target_name]
    object_pos = np.asarray(object_state.get_geom_state()["pos"], dtype=float)
    target_pos = np.asarray(target_state.get_geom_state()["pos"], dtype=float)
    delta = object_pos - target_pos
    inside = None
    contact = None
    if getattr(target_state, "object_state_type", None) == "site":
        site = domain.object_sites_dict[target_name]
        target_pos = np.asarray(domain.sim.data.get_site_xpos(target_name), dtype=float)
        site_mat = np.asarray(domain.sim.data.get_site_xmat(target_name), dtype=float).reshape(3, 3)
        delta = site_mat.T @ (object_pos - target_pos)
        inside = bool(site.under(target_pos, site_mat, object_pos))
        parent = domain.get_object(getattr(target_state, "parent_name", target_name))
        obj = domain.get_object(object_name)
        contact = bool(domain.check_contact(parent, obj))
    else:
        contact = bool(object_state.check_contact(target_state))
    return {
        "predicate": predicate,
        "met": bool(domain._eval_predicate(predicate)),
        "object_position": object_pos.tolist(),
        "target_position": target_pos.tolist(),
        "relative_position": delta.tolist(),
        "distance": float(np.linalg.norm(delta)),
        "xy_distance": float(np.linalg.norm(delta[:2])),
        "target_contact": contact,
        "inside_target_region": inside,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 4, 5])
    ap.add_argument("--tasks", nargs="+", type=int, default=[6, 9])
    ap.add_argument("--start-ids", nargs="+", type=int, default=list(range(10, 50)))
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run in the existing output directory")
    args = ap.parse_args()

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite Stage 3 output: {args.output_dir}")
    if args.resume and not args.output_dir.is_dir():
        raise FileNotFoundError(f"No Stage 3 output directory to resume: {args.output_dir}")
    if set(args.tasks) - set(GOALS):
        raise ValueError("Stage 3 task IDs must be the cheese/bowl task 6 and/or wine/rack task 9")
    if not args.start_ids or any(i < 10 or i >= 50 for i in args.start_ids):
        raise ValueError("Use held-out fixed start IDs 10 through 49")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    from configs.registry import get_config
    from envs.libero_env import LiberoAgent, load_init_states, make_env, run_episode
    from experiment_protocol import checkpoint_config, file_digest, sampling_seed, seed_everything
    from libero.libero import benchmark, get_libero_path
    from scripts.eval_libero import load_policy
    from scripts.evaluate import deterministic_inference

    deterministic_inference()
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    tasks = {task_id: suite.get_task(task_id) for task_id in set(args.tasks) | {SWAP[i] for i in args.tasks}}
    raw_path = args.output_dir / "instruction_swap_trajectories.jsonl.gz"
    expected_total = len(args.seeds) * len(EXPERIMENTS) * len(args.tasks) * len(args.start_ids)
    progress_path = args.output_dir / "progress_checkpoint.json"
    if args.resume:
        if not raw_path.exists() or not progress_path.exists():
            raise FileNotFoundError("Resume requires the trajectory file and progress_checkpoint.json")
        prior = json.loads(progress_path.read_text(encoding="utf-8"))
        if (prior.get("study") != STUDY or prior.get("plan") != PLAN
                or prior.get("planned_episodes") != expected_total):
            raise ValueError("Resume settings do not match the checkpointed Stage 3 run")
        rows, damaged = read_completed(raw_path)
        if damaged:
            backup = args.output_dir / f"instruction_swap_trajectories.interrupted-{int(time.time())}.gz"
            shutil.copy2(raw_path, backup)
            recovered = args.output_dir / "instruction_swap_trajectories.recovered.tmp.gz"
            with gzip.open(recovered, "wt", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
            os.replace(recovered, raw_path)
        keys = [(r["training_seed"], r["backbone"], r["task"], r["initial_state_id"])
                for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate completed episodes in Stage 3 checkpoint")
        if any(r["condition"] != "swapped" or not r["initial_state_matches_canonical"]
               or not r["initial_observation_matches_canonical"] for r in rows):
            raise ValueError("Stage 3 checkpoint contains a mismatched or unsupported record")
        completed_keys = set(keys)
        print(f"Resuming with {len(rows)}/{expected_total} completed episodes", flush=True)
    else:
        args.output_dir.mkdir(parents=True)
        rows = []
        completed_keys = set()
    started = time.time()
    completed = len(rows)
    if args.resume:
        write_progress_checkpoint(args.output_dir, rows, args.seeds, args.tasks,
                                  args.start_ids, expected_total, raw_path)

    with gzip.open(raw_path, "at" if args.resume else "wt", encoding="utf-8") as raw:
        write_progress_checkpoint(args.output_dir, rows, args.seeds, args.tasks,
                                  args.start_ids, expected_total, raw_path, include_sha=False)
        for seed in args.seeds:
            for backbone, exp in EXPERIMENTS.items():
                result_rel = f"{STUDY}/{PLAN}/libero/{exp}/seed_{seed:03d}"
                result_path = ROOT / "asset/durable/libero/evaluation_results" / result_rel / "final_test.json"
                checkpoint_path = ROOT / "asset/durable/libero/evaluation_checkpoints" / result_rel / "final_test.pt"
                final = json.loads(result_path.read_text(encoding="utf-8"))
                checkpoint_sha = file_digest(checkpoint_path)
                if checkpoint_sha != final["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                cfg = checkpoint_config(get_config("libero", exp), checkpoint, "libero",
                                        verify_live_encoder=True)
                vlm, head = load_policy(cfg, checkpoint_path, args.device)
                agent = LiberoAgent(vlm, head, cfg, args.device)
                canonical = {(r["task"], r["initial_state_id"]): r for r in final["results"]
                             if r["condition"] == "canonical"}

                for task_id in args.tasks:
                    task = tasks[task_id]
                    alternative = tasks[SWAP[task_id]]
                    alt_goal = GOALS[SWAP[task_id]]
                    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                    init_states = load_init_states(task)
                    env = make_env(str(bddl), cfg)
                    try:
                        for start_id in args.start_ids:
                            if (seed, backbone, task_id, start_id) in completed_keys:
                                continue
                            expected = canonical[(task_id, start_id)]
                            policy_seed = sampling_seed("libero", seed, task_id, start_id)
                            if policy_seed != expected["policy_sampling_seed"]:
                                raise ValueError(f"Policy seed mismatch at seed={seed}, task={task_id}, start={start_id}")
                            condition = "swapped"
                            requested_instruction = alternative.language
                            requested_goal = alt_goal
                            trace: list[dict] = []

                            def capture(step, wrapped, action, done):
                                domain = wrapped.env
                                current = {
                                    "step": step,
                                    "action": np.asarray(action, dtype=float).tolist(),
                                    "environment_goal_done": bool(done),
                                    "canonical": goal_snapshot(domain, GOALS[task_id]),
                                    "requested": goal_snapshot(domain, requested_goal),
                                }
                                trace.append(current)

                            seed_everything(policy_seed)
                            outcome = run_episode(
                                env, agent, cfg, requested_instruction,
                                seed=int(expected.get("environment_seed", start_id)),
                                init_state=init_states[start_id],
                                trace_callback=capture,
                                success_predicate=lambda wrapped, goal=requested_goal:
                                    bool(wrapped.env._eval_predicate(goal)),
                            )
                            start_match = outcome["initial_state_sha256"] == expected["initial_state_sha256"]
                            image_match = outcome["initial_observation_sha256"] == expected["initial_observation_sha256"]
                            if not start_match or not image_match:
                                raise ValueError(
                                    f"Matched-start mismatch at seed={seed}, task={task_id}, start={start_id}: "
                                    f"state={start_match}, image={image_match}")
                            row = {
                                "training_seed": seed,
                                "backbone": backbone,
                                "checkpoint_sha256": checkpoint_sha,
                                "cache_id": final.get("cache_id"),
                                "task": task_id,
                                "canonical_instruction": task.language,
                                "requested_instruction": requested_instruction,
                                "condition": condition,
                                "canonical_goal": GOALS[task_id],
                                "requested_goal": requested_goal,
                                "initial_state_id": start_id,
                                "policy_sampling_seed": policy_seed,
                                "environment_seed": int(expected.get("environment_seed", start_id)),
                                "initial_state_sha256": outcome["initial_state_sha256"],
                                "initial_observation_sha256": outcome["initial_observation_sha256"],
                                "initial_state_matches_canonical": start_match,
                                "initial_observation_matches_canonical": image_match,
                                "canonical_test_success": bool(expected["success"]),
                                "requested_goal_success": bool(outcome["success"]),
                                "canonical_goal_success_under_swap": bool(outcome["environment_success"]),
                                "steps": outcome["steps"],
                                "trajectory": trace,
                            }
                            rows.append(row)
                            completed_keys.add((seed, backbone, task_id, start_id))
                            raw.write(json.dumps(row, separators=(",", ":")) + "\n")
                            raw.flush()
                            completed += 1
                            write_progress_checkpoint(args.output_dir, rows, args.seeds,
                                                      args.tasks, args.start_ids,
                                                      expected_total, raw_path, include_sha=False)
                            if completed % 10 == 0 or completed == expected_total:
                                elapsed = time.time() - started
                                print(f"{completed}/{expected_total}: {backbone} seed={seed} "
                                      f"task={task_id} start={start_id} "
                                      f"swapped_goal={row['requested_goal_success']} "
                                      f"original_goal={row['canonical_goal_success_under_swap']} "
                                      f"{elapsed:.0f}s elapsed", flush=True)
                    finally:
                        env.close()

                del agent, head, vlm
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if len(rows) != expected_total:
        raise RuntimeError(f"Stage 3 ended with {len(rows)}/{expected_total} episodes")
    write_progress_checkpoint(args.output_dir, rows, args.seeds, args.tasks,
                              args.start_ids, expected_total, raw_path)
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["task"], row["training_seed"], row["backbone"]), []).append(row)
    summary = {
        "protocol": "Same LIBERO held-out reset; swap cheese/bowl and wine/rack canonical instructions; score requested alternate BDDL predicate; stop on requested or original goal.",
        "no_training": True,
        "tasks": args.tasks,
        "start_ids": args.start_ids,
        "heads": len(args.seeds) * len(EXPERIMENTS),
        "episodes": len(rows),
        "trajectory_file": raw_path.name,
        "elapsed_seconds": time.time() - started,
        "by_task_seed_backbone": [
            {
                "task": key[0], "training_seed": key[1], "backbone": key[2],
                "n": len(group),
                "canonical_success_n": sum(r["canonical_test_success"] for r in group),
                "swapped_instruction_requested_goal_success_n": sum(r["requested_goal_success"] for r in group),
                "swapped_instruction_original_goal_success_n": sum(r["canonical_goal_success_under_swap"] for r in group),
                "neither_goal_success_n": sum(not r["requested_goal_success"] and not r["canonical_goal_success_under_swap"] for r in group),
            }
            for key, group in sorted(groups.items())
        ],
        "limits": [
            "The behavioral swap tests two familiar goals in the same LIBERO-Goal scene, not every language capability.",
            "Training-seed count remains three; reset-level results are not independent training replicates.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} matched instruction-swap episodes to {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
