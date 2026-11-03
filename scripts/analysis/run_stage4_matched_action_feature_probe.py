"""Probe layer-16 features and first actions on matched LIBERO starts.

For the saved Stage 3 cheese/bowl and wine/rack starts, this script recreates
the exact settled observation, checks its fingerprints, and queries each
epoch-120 head once under the canonical and familiar swapped instruction.
It runs inference only: no policy is stepped after its first predicted action.

Run with the project's existing ``vla_libero`` interpreter from the project
root. Outputs are written to ignored durable storage and docs/analysis/stage4.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

if sys.platform == "win32" and os.environ.get("MUJOCO_GL", "").lower() != "wgl":
    os.environ["MUJOCO_GL"] = "wgl"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
STUDY = "libero_progression_120_v1"
PLAN = "c6854e3ff020"
EXP = {"groot": "exp05", "qwen3vl": "exp06"}
LABEL = {"groot": "GR00T", "qwen3vl": "Qwen"}
TASKS = {6: 9, 9: 6}
SEEDS = (3, 4, 5)
STARTS = tuple(range(10, 50))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_gzip_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pool_features(tokens: torch.Tensor, img_mask: torch.Tensor,
                  valid_length: int) -> tuple[list[float], list[float]]:
    row = tokens[0].float()
    image = img_mask[0].bool()
    valid = torch.arange(len(image), device=image.device) < valid_length
    text = valid & ~image
    if not bool(image.any()) or not bool(text.any()):
        raise ValueError("Expected non-empty image and text token groups")
    image_mean = row[image].mean(0)
    text_mean = row[text].mean(0)
    return image_mean.cpu().tolist(), text_mean.cpu().tolist()


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA on paired observations; invariant to orthogonal feature rotations."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    kx, ky = x @ x.T, y @ y.T
    denom = np.linalg.norm(kx) * np.linalg.norm(ky)
    return float(np.sum(kx * ky) / denom) if denom else float("nan")


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    return float(1.0 - np.dot(x, y) / (nx * ny)) if nx and ny else float("nan")


def bootstrap_mean_ci(values: np.ndarray, seed: int = 1847,
                      reps: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(temp, path)


def analyze(rows: list[dict], output: Path) -> dict:
    stage3_path = ROOT / "asset/durable/libero/instruction_sensitivity/stage3_cross_goal_v1/instruction_swap_trajectories.jsonl.gz"
    stage3 = read_gzip_jsonl(stage3_path)
    archived_actions = {(r["training_seed"], r["backbone"], r["task"], r["initial_state_id"]):
                        np.asarray(r["trajectory"][0]["action"], dtype=float) for r in stage3}
    for r in rows:
        # LiberoAgent clips the decoder's first action before env.step(). Store
        # and compare the action actually applied to the simulator.
        r["action_canonical"] = np.clip(np.asarray(r["action_canonical"], dtype=float), -1, 1).tolist()
        r["action_swapped"] = np.clip(np.asarray(r["action_swapped"], dtype=float), -1, 1).tolist()
        key = (r["training_seed"], r["backbone"], r["task"], r["initial_state_id"])
        r["swapped_action_reproduced"] = bool(np.allclose(
            r["action_swapped"], archived_actions[key], atol=1e-5, rtol=0.0))
    # One observation per task/start/backbone; frozen features do not vary by
    # training seed, so the duplicate copies are checked then deduplicated.
    feat_index = {}
    for r in rows:
        key = (r["backbone"], r["task"], r["initial_state_id"])
        value = (r["initial_observation_sha256"], r["features"])
        if key in feat_index and feat_index[key] != value:
            # JSON float string round trips are deterministic, so this catches
            # a changed model input or inconsistent frozen-backbone encoding.
            raise ValueError(f"Frozen feature repeat mismatch at {key}")
        feat_index[key] = value

    action_rows, head_rows, feature_rows = [], [], []
    heads = sorted({(r["task"], r["training_seed"], r["backbone"]) for r in rows})
    for task, seed, backbone in heads:
        group = [r for r in rows if (r["task"], r["training_seed"], r["backbone"])
                 == (task, seed, backbone)]
        group.sort(key=lambda r: r["initial_state_id"])
        diffs = np.asarray([np.asarray(r["action_swapped"][:6], dtype=float)
                            - np.asarray(r["action_canonical"][:6], dtype=float)
                            for r in group], dtype=float)
        grip_diffs = np.asarray([r["action_swapped"][6] - r["action_canonical"][6]
                                 for r in group], dtype=float)
        motion_norm = np.linalg.norm(diffs, axis=1)
        action_rows.append({
            "task": task, "training_seed": seed, "backbone": backbone,
            "n_matched_starts": len(group),
            "canonical_successes": sum(r["canonical_success"] for r in group),
            "swapped_goal_successes": sum(r["swapped_goal_success"] for r in group),
            "mean_instruction_induced_motion_delta_l2": float(motion_norm.mean()),
            "median_instruction_induced_motion_delta_l2": float(np.median(motion_norm)),
            "mean_absolute_gripper_delta": float(np.abs(grip_diffs).mean()),
            "mean_text_feature_cosine_distance": float(np.mean(
                [r["text_feature_cosine_distance"] for r in group])),
            "mean_image_feature_cosine_distance": float(np.mean(
                [r["image_feature_cosine_distance"] for r in group])),
            "correlation_text_feature_shift_vs_motion_delta": float(np.corrcoef(
                [r["text_feature_cosine_distance"] for r in group], motion_norm)[0, 1]),
            "correlation_image_feature_shift_vs_motion_delta": float(np.corrcoef(
                [r["image_feature_cosine_distance"] for r in group], motion_norm)[0, 1]),
            "swapped_action_reproduced_within_1e-5": sum(r["swapped_action_reproduced"] for r in group),
        })
        for dim, name in enumerate(("dx", "dy", "dz", "droll", "dpitch", "dyaw")):
            lo, hi = bootstrap_mean_ci(diffs[:, dim], seed=1000 + task * 10 + seed + dim)
            action_rows.append({
                "task": task, "training_seed": seed, "backbone": backbone,
                "n_matched_starts": len(group), "action_component": name,
                "mean_swapped_minus_canonical": float(diffs[:, dim].mean()),
                "conditional_start_bootstrap95": [lo, hi],
            })
        lo, hi = bootstrap_mean_ci(grip_diffs, seed=2000 + task * 10 + seed)
        action_rows.append({
            "task": task, "training_seed": seed, "backbone": backbone,
            "n_matched_starts": len(group), "action_component": "gripper",
            "mean_swapped_minus_canonical": float(grip_diffs.mean()),
            "conditional_start_bootstrap95": [lo, hi],
        })
        for r in group:
            outcome = ("both" if r["canonical_success"] and r["swapped_goal_success"] else
                       "canonical_only" if r["canonical_success"] else
                       "swapped_only" if r["swapped_goal_success"] else "neither")
            head_rows.append({"task": task, "training_seed": seed,
                              "backbone": backbone, "start_id": r["initial_state_id"],
                              "outcome_transition": outcome,
                              "first_action_motion_delta_l2": float(np.linalg.norm(
                                  np.asarray(r["action_swapped"][:6], dtype=float) -
                                  np.asarray(r["action_canonical"][:6], dtype=float))),
                              "first_action_gripper_delta": float(
                                  r["action_swapped"][6] - r["action_canonical"][6]),
                              "text_feature_cosine_distance": r["text_feature_cosine_distance"],
                              "image_feature_cosine_distance": r["image_feature_cosine_distance"]})

    reciprocity_rows = []
    for seed in SEEDS:
        for backbone in EXP:
            effects = {}
            for task in (6, 9):
                group = [r for r in rows if r["task"] == task and r["training_seed"] == seed
                         and r["backbone"] == backbone]
                effects[task] = np.mean([
                    np.asarray(r["action_swapped"][:6], dtype=float)
                    - np.asarray(r["action_canonical"][:6], dtype=float)
                    for r in group], axis=0)
            # Task 6 changes cheese -> wine; task 9 changes wine -> cheese.
            # Their scene cohorts differ, so this is descriptive direction
            # agreement, not a matched-scene causal estimate.
            cosine = float(np.dot(effects[6], -effects[9]) /
                           (np.linalg.norm(effects[6]) * np.linalg.norm(effects[9])))
            reciprocity_rows.append({
                "training_seed": seed, "backbone": backbone,
                "cosine_task6_vs_reverse_task9": cosine,
                "relative_vector_residual": float(np.linalg.norm(effects[6] + effects[9]) /
                                                   np.linalg.norm(effects[6])),
                "task6_mean_motion_delta": effects[6].tolist(),
                "task9_mean_motion_delta": effects[9].tolist(),
                "n_starts_per_task": 40,
            })

    for task in (6, 9):
        for modality in ("image", "text"):
            groot = np.asarray([feat_index[("groot", task, start)][1]["swapped"][f"{modality}_mean"]
                                for start in STARTS])
            qwen = np.asarray([feat_index[("qwen3vl", task, start)][1]["swapped"][f"{modality}_mean"]
                               for start in STARTS])
            feature_rows.append({"task": task, "modality": modality,
                                 "n_matched_inputs": len(STARTS),
                                 "cross_backbone_linear_CKA": linear_cka(groot, qwen),
                                 "interpretation": "similarity of sample-to-sample representational geometry; not feature-space alignment"})

    output.mkdir(parents=True, exist_ok=True)
    def write_csv(path: Path, data: list[dict]) -> None:
        if not data:
            return
        fields = list(dict.fromkeys(k for row in data for k in row))
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    write_csv(output / "stage4_instruction_action_effects.csv", action_rows)
    write_csv(output / "stage4_action_by_outcome.csv", head_rows)
    write_csv(output / "stage4_feature_geometry.csv", feature_rows)
    write_csv(output / "stage4_swap_reciprocity.csv", reciprocity_rows)

    # Bootstrap intervals are plotted per training head to keep all 18 action
    # component comparisons visible and avoid pooling training seeds.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), sharey="row")
    dims = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
    colors = {"groot": "#2475ad", "qwen3vl": "#d87923"}
    for ti, task in enumerate((6, 9)):
        for si, seed in enumerate(SEEDS):
            ax = axes[ti, si]
            for backbone in ("groot", "qwen3vl"):
                means, lows, highs = [], [], []
                for dim in dims:
                    record = next(r for r in action_rows if r.get("task") == task
                                  and r.get("training_seed") == seed and r.get("backbone") == backbone
                                  and r.get("action_component") == dim)
                    means.append(record["mean_swapped_minus_canonical"])
                    lows.append(means[-1] - record["conditional_start_bootstrap95"][0])
                    highs.append(record["conditional_start_bootstrap95"][1] - means[-1])
                x = np.arange(len(dims)) + (-.08 if backbone == "groot" else .08)
                ax.errorbar(x, means, yerr=np.asarray([lows, highs]), fmt="o",
                            color=colors[backbone], capsize=2, ms=4,
                            label=LABEL[backbone] if (ti, si) == (0, 0) else None)
            ax.axhline(0, color="#596579", lw=.8)
            ax.set_xticks(np.arange(len(dims)), dims, rotation=25)
            ax.set_title(f"Task {task} · seed {seed}")
            ax.grid(axis="y", alpha=.25)
            if si == 0:
                ax.set_ylabel("First action: swapped − canonical")
    fig.legend(loc="upper center", bbox_to_anchor=(.5, .985), ncol=2, frameon=False)
    fig.suptitle("Instruction changes the first action on identical LIBERO starts", y=1.045)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(output / "stage4_first_action_instruction_delta.png", dpi=190,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    order = [(seed, backbone) for seed in SEEDS for backbone in ("groot", "qwen3vl")]
    for i, (seed, backbone) in enumerate(order):
        row = next(r for r in reciprocity_rows if r["training_seed"] == seed and r["backbone"] == backbone)
        ax.scatter(i, row["cosine_task6_vs_reverse_task9"], s=65,
                   color=colors[backbone], zorder=3)
        ax.vlines(i, 0, row["cosine_task6_vs_reverse_task9"], color=colors[backbone], alpha=.35)
        ax.text(i, row["cosine_task6_vs_reverse_task9"] - .035,
                f"resid {row['relative_vector_residual']:.2f}", ha="center", fontsize=8)
    ax.axhline(1, color="#596579", linestyle="--", linewidth=1)
    ax.set_ylim(.85, 1.04)
    ax.set_xticks(range(len(order)), [f"{LABEL[b]} · seed {s}" for s, b in order])
    ax.set_ylabel("Cosine: task 6 change vs reverse task 9 change")
    ax.set_title("The same instruction contrast shifts first actions in opposite directions",
                 fontweight="bold")
    ax.grid(axis="y", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.01, .01, "Task 6 changes cheese → wine; task 9 changes wine → cheese. Different scene cohorts; descriptive only.",
             color="#4e5863", fontsize=8.5)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(output / "stage4_swapped_instruction_reciprocity.png", dpi=190,
                bbox_inches="tight")
    plt.close(fig)

    summary = {
            "scope": "One first-action query per instruction and matched start; no post-action simulator steps.",
        "n_records": len(rows), "unique_scene_prompts_per_backbone": 160,
        "matched_starts_per_head_and_task": 40,
        "all_initial_state_and_observation_hashes_verified": True,
        "all_policy_sampling_seeds_matched_to_stage3": True,
        "swapped_first_action_reproduction": {
            "verified": sum(r["swapped_action_reproduced"] for r in rows),
            "total": len(rows), "tolerance": 1e-5,
        },
        "frozen_feature_comparison": "Pooled layer-16 image and non-image token means on the exact held-out initial observations; cross-backbone comparison uses linear CKA and does not assert coordinate alignment.",
        "head_level_action_effects": action_rows,
        "instruction_swap_reciprocity": reciprocity_rows,
        "feature_geometry": feature_rows,
        "action_note": "Recorded action vectors are clipped to [-1, 1], matching LiberoAgent.act before env.step().",
        "limitations": [
            "An initial action is not the full closed-loop policy mechanism; later states diverge.",
            "Canonical and swapped goal tasks have different target objects; instruction-induced action changes are task-conditioned changes, not a language-only causal decomposition.",
            "Feature/action associations are descriptive and cannot isolate the causal contribution of the frozen backbone from independently trained heads.",
            "Conditional bootstrap intervals resample matched starts within each fixed head; they are not uncertainty over new training seeds.",
        ],
    }
    (output / "stage4_probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-raw", type=Path,
                    default=ROOT / "asset/durable/libero/features_actions/stage4_matched_start_probe_v1/stage4_probe_rows.jsonl.gz")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "docs/analysis/stage4")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
    if args.output_raw.exists():
        raise FileExistsError(f"Refusing to overwrite probe data: {args.output_raw}")

    from configs.registry import get_config
    from envs.libero_env import (CAMERA, CAMERA_WRIST, _array_sha256,
                                _mujoco_state_sha256, build_state,
                                load_init_states, make_env, resize_frame)
    from experiment_protocol import checkpoint_config, file_digest, sampling_seed, seed_everything
    from libero.libero import benchmark, get_libero_path
    from scripts.eval_libero import load_policy
    from scripts.evaluate import deterministic_inference
    from models.vla_train import VLATrainModel

    deterministic_inference()
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    tasks = {task: suite.get_task(task) for task in (6, 9)}
    swap_task = TASKS
    archive_path = ROOT / "asset/durable/libero/instruction_sensitivity/stage3_cross_goal_v1/instruction_swap_trajectories.jsonl.gz"
    stage3 = read_gzip_jsonl(archive_path)
    stage3_index = {(r["training_seed"], r["backbone"], r["task"], r["initial_state_id"]): r
                    for r in stage3}
    if len(stage3) != 480 or len(stage3_index) != 480:
        raise ValueError("Expected the complete, duplicate-free 480-row Stage 3 archive")
    final_index = {}
    for backbone, exp in EXP.items():
        for seed in SEEDS:
            p = (ROOT / "asset/durable/libero/evaluation_results" / STUDY / PLAN /
                 "libero" / exp / f"seed_{seed:03d}" / "final_test.json")
            rec = json.loads(p.read_text(encoding="utf-8"))
            final_index[(seed, backbone)] = {(r["task"], r["initial_state_id"]): r
                                             for r in rec["results"] if r["condition"] == "canonical"}

    # Recreate each fixed initial observation once. All six heads and both
    # prompt conditions must match the archived state/image fingerprints.
    observations = {}
    for task_id, alternate_task_id in swap_task.items():
        task, alternate = tasks[task_id], tasks[alternate_task_id]
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_states = load_init_states(task)
        cfg = get_config("libero", "exp05")
        env = make_env(str(bddl), cfg)
        try:
            for start_id in STARTS:
                canonical_record = final_index[(3, "groot")][(task_id, start_id)]
                env.seed(int(canonical_record["environment_seed"]))
                obs = env.reset()
                obs = env.set_init_state(init_states[start_id])
                settle = np.zeros(cfg.action_dim, dtype=np.float32)
                settle[-1] = -1.0
                for _ in range(5):
                    obs, _reward, _done, _info = env.step(settle)
                state_sha = _mujoco_state_sha256(env.sim.data)
                image_sha = _array_sha256(np.asarray(obs[CAMERA]))
                for seed in SEEDS:
                    for backbone in EXP:
                        row = stage3_index[(seed, backbone, task_id, start_id)]
                        canonical = final_index[(seed, backbone)][(task_id, start_id)]
                        if state_sha != row["initial_state_sha256"] or image_sha != row["initial_observation_sha256"]:
                            raise ValueError(f"Recreated observation differs from Stage 3: {task_id}/{start_id}/{seed}/{backbone}")
                        if state_sha != canonical["initial_state_sha256"] or image_sha != canonical["initial_observation_sha256"]:
                            raise ValueError(f"Recreated observation differs from final test: {task_id}/{start_id}/{seed}/{backbone}")
                        if int(canonical_record["environment_seed"]) != int(canonical["environment_seed"]):
                            raise ValueError("Matched environment seed differs across head records")
                main_image = resize_frame(np.asarray(obs[CAMERA])[::-1], cfg.lt_img_w, cfg.lt_img_h)
                wrist_image = resize_frame(np.asarray(obs[CAMERA_WRIST])[::-1], cfg.lt_img_w, cfg.lt_img_h)
                observations[(task_id, start_id)] = {
                    "views": [main_image, wrist_image],
                    "state": build_state(obs).astype(np.float32),
                    "state_sha": state_sha, "image_sha": image_sha,
                    "canonical_instruction": task.language,
                    "swapped_instruction": alternate.language,
                    "canonical_success": {},
                    "swapped_success": {},
                }
                for seed in SEEDS:
                    for backbone in EXP:
                        orig = final_index[(seed, backbone)][(task_id, start_id)]
                        swap = stage3_index[(seed, backbone, task_id, start_id)]
                        observations[(task_id, start_id)]["canonical_success"][(seed, backbone)] = bool(orig["success"])
                        observations[(task_id, start_id)]["swapped_success"][(seed, backbone)] = bool(swap["requested_goal_success"])
        finally:
            env.close()
        print(f"Verified 40 matched initial observations for task {task_id}", flush=True)

    rows = []
    for backbone, exp in EXP.items():
        print(f"Loading {LABEL[backbone]} VLM and cached head inputs", flush=True)
        first_seed = SEEDS[0]
        result_rel = f"{STUDY}/{PLAN}/libero/{exp}/seed_{first_seed:03d}"
        ckpt_path = ROOT / "asset/durable/libero/evaluation_checkpoints" / result_rel / "final_test.pt"
        result_path = ROOT / "asset/durable/libero/evaluation_results" / result_rel / "final_test.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if file_digest(ckpt_path) != result["checkpoint_sha256"]:
            raise ValueError(f"Selected checkpoint hash mismatch: {ckpt_path}")
        base_cfg = get_config("libero", exp)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = checkpoint_config(base_cfg, checkpoint, "libero", verify_live_encoder=True)
        vlm, head = load_policy(cfg, ckpt_path, args.device)

        encoded = {}
        with torch.no_grad():
            for (task_id, start_id), obs in sorted(observations.items()):
                inputs = vlm.build_vlm_inputs(
                    [obs["views"], obs["views"]],
                    [obs["canonical_instruction"], obs["swapped_instruction"]],
                    args.device,
                )
                valid_lengths = inputs["attention_mask"].sum(dim=1).tolist()
                tokens, img_mask = vlm.encode_vlm(inputs)
                if tokens.ndim == 4:
                    raise ValueError("Expected a single requested layer for this probe")
                enc = []
                for prompt_i, condition in enumerate(("canonical", "swapped")):
                    token = tokens[prompt_i:prompt_i + 1]
                    mask = img_mask[prompt_i:prompt_i + 1]
                    image_mean, text_mean = pool_features(token, mask, int(valid_lengths[prompt_i]))
                    enc.append({"tokens": token[0].detach().cpu(), "img_mask": mask[0].detach().cpu(),
                                "valid_length": int(valid_lengths[prompt_i]),
                                "image_mean": image_mean, "text_mean": text_mean,
                                "condition": condition})
                encoded[(task_id, start_id)] = enc

        # Reuse the frozen VLM encodings for the three independent trained
        # heads. Only the trainable head checkpoint changes across seeds.
        for seed in SEEDS:
            if seed != first_seed:
                seed_rel = f"{STUDY}/{PLAN}/libero/{exp}/seed_{seed:03d}"
                seed_ckpt = ROOT / "asset/durable/libero/evaluation_checkpoints" / seed_rel / "final_test.pt"
                seed_result = json.loads((ROOT / "asset/durable/libero/evaluation_results" / seed_rel / "final_test.json").read_text(encoding="utf-8"))
                if file_digest(seed_ckpt) != seed_result["checkpoint_sha256"]:
                    raise ValueError(f"Selected checkpoint hash mismatch: {seed_ckpt}")
                ck = torch.load(seed_ckpt, map_location="cpu", weights_only=False)
                saved_state = ck.get("model", ck.get("state_dict", ck))
                missing, unexpected = head.load_state_dict(saved_state, strict=False)
                hard = [k for k in missing if "rotary" not in k and "inv_freq" not in k]
                if hard or unexpected:
                    raise ValueError(f"Head checkpoint mismatch: missing={hard[:3]}, unexpected={unexpected[:3]}")
                del ck, saved_state
            print(f"Querying {LABEL[backbone]} head seed {seed}", flush=True)
            with torch.no_grad():
                for (task_id, start_id), obs in sorted(observations.items()):
                    stage3_row = stage3_index[(seed, backbone, task_id, start_id)]
                    sample_seed = int(stage3_row["policy_sampling_seed"])
                    pair = encoded[(task_id, start_id)]
                    action_by_condition = {}
                    for enc in pair:
                        token = enc["tokens"].unsqueeze(0).to(args.device)
                        mask = enc["img_mask"].unsqueeze(0).to(args.device)
                        state_t = torch.from_numpy(obs["state"]).unsqueeze(0).to(args.device)
                        seed_everything(sample_seed)
                        sampled = head.sample(token, state_t, num_steps=cfg.num_flow_steps, img_mask=mask)
                        chunk = sampled.view(cfg.action_horizon, cfg.action_dim).cpu().float().numpy()
                        action_by_condition[enc["condition"]] = np.clip(chunk[0], -1.0, 1.0).tolist()
                    archived = np.asarray(stage3_row["trajectory"][0]["action"], dtype=float)
                    swapped_match = bool(np.allclose(action_by_condition["swapped"], archived,
                                                     atol=1e-5, rtol=0.0))
                    img_dist = cosine_distance(np.asarray(pair[0]["image_mean"]),
                                               np.asarray(pair[1]["image_mean"]))
                    txt_dist = cosine_distance(np.asarray(pair[0]["text_mean"]),
                                               np.asarray(pair[1]["text_mean"]))
                    rows.append({
                        "training_seed": seed, "backbone": backbone, "task": task_id,
                        "initial_state_id": start_id, "policy_sampling_seed": sample_seed,
                        "initial_state_sha256": obs["state_sha"],
                        "initial_observation_sha256": obs["image_sha"],
                        "canonical_instruction": obs["canonical_instruction"],
                        "swapped_instruction": obs["swapped_instruction"],
                        "canonical_success": obs["canonical_success"][(seed, backbone)],
                        "swapped_goal_success": obs["swapped_success"][(seed, backbone)],
                        "action_canonical": action_by_condition["canonical"],
                        "action_swapped": action_by_condition["swapped"],
                        "swapped_action_reproduced": swapped_match,
                        "text_feature_cosine_distance": txt_dist,
                        "image_feature_cosine_distance": img_dist,
                        "features": {condition: {"image_mean": item["image_mean"],
                                                  "text_mean": item["text_mean"],
                                                  "valid_tokens": item["valid_length"],
                                                  "image_tokens": int(item["img_mask"].sum())}
                                     for condition, item in zip(("canonical", "swapped"), pair)},
                    })
            print(f"  Completed {LABEL[backbone]} seed {seed}: {len(observations)} matched starts", flush=True)

        del head, vlm, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(rows) != 480:
        raise RuntimeError(f"Expected 480 matched head/start rows, found {len(rows)}")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    summary = analyze(rows, output_dir)
    save_rows(args.output_raw, rows)
    summary["raw_probe_path"] = args.output_raw.relative_to(ROOT).as_posix() if args.output_raw.is_relative_to(ROOT) else str(args.output_raw)
    summary["raw_probe_sha256"] = sha256(args.output_raw)
    (output_dir / "stage4_probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote 480 matched probes to {args.output_raw}", flush=True)
    print(f"Wrote Stage 4 summary, plots, and tables to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
