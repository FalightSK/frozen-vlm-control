"""Plot held-out backbone comparisons from the two progression studies.

Run from the project root: python scripts/analysis/plot_stage1_results.py
The ignored asset/durable final-test records and asset/runs study statuses must
be present locally. Outputs go to docs/analysis/stage1/ by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (3, 4, 5)
STUDIES = {
    "aloha": ("aloha_progression_250_v1", "415a08a2c6f2", 250, 200, "ALOHA Transfer Cube"),
    "libero": ("libero_progression_120_v1", "c6854e3ff020", 120, 400, "LIBERO-Goal"),
}
COLORS = {"GR00T": "#2475ad", "Qwen": "#d87923"}


def outcome_map(record):
    dataset = record["dataset"]
    kinds = ({"gym_aloha_mujoco_state_v1"} if dataset == "aloha" else
             {"libero_mujoco_state_after_init_and_settle_v1", "libero_mujoco_state_after_init_and_settle5_v1"})
    episodes = record["episodes"] if dataset == "aloha" else record["results"]
    outcomes = {}
    for row in episodes:
        if dataset == "libero" and row["condition"] != "canonical":
            continue
        key = (int(row["task"]) if dataset == "libero" else 0, int(row["initial_state_id"]))
        success = row["is_success"] if dataset == "aloha" else row["success"]
        digest = row.get("initial_state_sha256", "")
        if (key in outcomes or not isinstance(success, bool) or row.get("initial_state_kind") not in kinds
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower())
                or row.get("policy_sampling_seed") is None):
            raise ValueError(f"Invalid or duplicate episode {key}")
        outcomes[key] = (success, row["environment_seed"], row["policy_sampling_seed"],
                         digest.lower(), row["initial_state_kind"])
    return outcomes


def compare_records(left, right):
    for key in ("dataset", "protocol_id", "training_seed", "feature_mode"):
        if left.get(key) is None or left.get(key) != right.get(key):
            raise ValueError(f"Result pair has missing or mismatched {key}")
    if left["feature_mode"] != "cached":
        raise ValueError("Expected frozen cached features")
    if not all(r.get("checkpoint_sha256") and r.get("run_signature") and r.get("cache_id")
               for r in (left, right)) or left["checkpoint_sha256"] == right["checkpoint_sha256"]:
        raise ValueError("Result pair lacks distinct checkpoint, run, or cache provenance")
    allowed = {"output_dir", "embeddings_cache", "model_path", "device", "embed_batch_size", "vlm_backbone"}
    configs = [{k: v for k, v in r["configuration"].items() if k not in allowed} for r in (left, right)]
    if configs[0] != configs[1]:
        raise ValueError("Paired policies differ in settings beyond the backbone")
    a, b = outcome_map(left), outcome_map(right)
    if a.keys() != b.keys() or any(a[key][1:] != b[key][1:] for key in a):
        raise ValueError("Paired starts, state hashes, or policy-sampling seeds do not match")
    keys = sorted(a)
    g = np.array([a[key][0] for key in keys], dtype=bool)
    q = np.array([b[key][0] for key in keys], dtype=bool)
    delta = g.astype(float) - q.astype(float)
    groups = [np.array([i for i, key in enumerate(keys) if key[0] == task])
              for task in sorted({key[0] for key in keys})]
    if len({len(group) for group in groups}) != 1:
        raise ValueError("Task-stratified bootstrap requires equal start counts per task")
    rng = np.random.default_rng(20261018)
    boot = np.zeros(20000)
    for group in groups:
        for start in range(0, len(boot), 1000):
            indices = rng.choice(group, size=(min(1000, len(boot) - start), len(group)))
            boot[start:start + len(indices)] += delta[indices].mean(axis=1) / len(groups)
    return {"dataset": left["dataset"], "training_seed": left["training_seed"], "n": len(keys),
            "protocol_id": left["protocol_id"], "both_succeed": int((g & q).sum()),
            "both_fail": int((~g & ~q).sum()), "only_left": int((g & ~q).sum()),
            "only_right": int((~g & q).sum()), "left_success_rate": float(g.mean()),
            "right_success_rate": float(q.mean()), "difference_pp": float(100 * delta.mean()),
            "difference_bootstrap95_pp": (100 * np.quantile(boot, [0.025, 0.975])).tolist()}


def read_results(dataset, study, plan, epoch, expected_n):
    status_path = ROOT / "asset" / "runs" / study / "study_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    selected = {(r["dataset"], r["experiment"], r["training_seed"]): r
                for r in status["completed_policies"]}
    results, cohort = [], None
    for seed in SEEDS:
        pair, sources = [], []
        for exp, backbone in (("exp05", "groot"), ("exp06", "qwen3vl")):
            path = (ROOT / "asset" / "durable" / dataset / "evaluation_results" / study / plan
                    / dataset / exp / f"seed_{seed:03d}" / "final_test.json")
            raw = path.read_bytes()
            record = json.loads(raw)
            identity = (record.get("dataset"), record.get("experiment"), record.get("training_seed"),
                        record.get("checkpoint_epoch"), record.get("study_id"), record.get("purpose"))
            if identity != (dataset, exp, seed, epoch, study, "final_test") or record.get("is_final") is not True:
                raise ValueError(f"Unexpected final-test identity: {path}")
            if record["configuration"].get("vlm_backbone") != backbone:
                raise ValueError(f"Unexpected backbone in {path}")
            entry = selected[(dataset, exp, seed)]
            if (entry["selected_epoch"] != epoch or
                    entry["selected_checkpoint_sha256"] != record["checkpoint_sha256"]):
                raise ValueError(f"Selected checkpoint metadata differs for {path}")
            mapped = outcome_map(record)
            signature = tuple((key, value[1], value[3], value[4]) for key, value in sorted(mapped.items()))
            if cohort is None:
                cohort = signature
            if signature != cohort:
                raise ValueError(f"Reset cohort differs across policies: {path}")
            if len(mapped) != expected_n or not math.isclose(
                    sum(value[0] for value in mapped.values()) / expected_n,
                    record["success_rate"], abs_tol=1e-12):
                raise ValueError(f"Unexpected episode count or success rate in {path}")
            if dataset == "libero" and any(sum(key[0] == task for key in mapped) != 40 for task in range(10)):
                raise ValueError(f"LIBERO requires 40 starts for each task: {path}")
            pair.append(record)
            sources.append({"path": path.relative_to(ROOT).as_posix(),
                            "result_sha256": hashlib.sha256(raw).hexdigest(),
                            "recorded_checkpoint_sha256": record["checkpoint_sha256"],
                            "cache_id": record["cache_id"]})
        comparison = compare_records(*pair)
        comparison["sources"] = sources
        results.append(comparison)
    return results


def success_plot(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), sharey=True)
    for ax, (dataset, (_, _, epoch, n, label)) in zip(axes, STUDIES.items()):
        rows = data[dataset]
        x = np.arange(len(SEEDS))
        for i, row in enumerate(rows):
            g, q = 100 * row["left_success_rate"], 100 * row["right_success_rate"]
            ax.plot([x[i] - 0.06, x[i] + 0.06], [g, q], color="#abb4bf", lw=1.4, zorder=1)
            for dx, value, name, offset in ((-0.06, g, "GR00T", 8), (0.06, q, "Qwen", -16)):
                ax.scatter(x[i] + dx, value, color=COLORS[name], s=64, zorder=3)
                ax.annotate(f"{value:.1f}%", (x[i] + dx, value), xytext=(0, offset),
                            textcoords="offset points", ha="center", fontsize=9, color=COLORS[name])
        ax.set_title(f"{label} · epoch {epoch}\n{n} matched starts per head", fontsize=11, pad=12)
        ax.set_xticks(x, [f"Seed {seed}" for seed in SEEDS])
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color="#dce1e7", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out success (%)")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color, label=name)
               for name, color in COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Final policy success by independent head-training seed", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, "Each point is one trained head; lines connect the two heads with the same seed number.",
             fontsize=9, color="#4e5863")
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    fig.savefig(output / "success_by_training_seed.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def difference_plot(data, output):
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 6.3), sharex=True)
    for ax, (dataset, (_, _, epoch, n, label)) in zip(axes, STUDIES.items()):
        rows = data[dataset]
        mean = statistics.mean(row["difference_pp"] for row in rows)
        for i, row in enumerate(rows):
            point = row["difference_pp"]
            lower, upper = row["difference_bootstrap95_pp"]
            ax.errorbar(point, i, xerr=[[point - lower], [upper - point]], fmt="o", color="#2475ad",
                        capsize=4, markersize=6, linewidth=1.5)
            ax.text(1.015, i, f"{point:+.2f} pp", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=9)
        ax.axvline(0, color="#58616b", linewidth=1)
        ax.set_yticks(range(3), [f"Seed {seed}" for seed in SEEDS])
        ax.set_ylim(2.5, -0.5)
        ax.set_xlim(-15, 45)
        ax.set_title(f"{label} · epoch {epoch} · {n} starts per head   |   3-seed mean: {mean:+.1f} pp",
                     fontsize=10.5, loc="left", pad=9)
        ax.grid(axis="x", color="#dce1e7", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xlabel("GR00T minus Qwen success (percentage points); positive favors GR00T")
    fig.suptitle("Backbone difference for each trained head pair", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01,
             "Lines: 95% paired-start bootstrap intervals conditional on each head pair; LIBERO resampled within tasks.\n"
             "These intervals do not measure uncertainty across independently trained heads (3 seed pairs).",
             fontsize=8.5, color="#4e5863")
    fig.tight_layout(rect=(0, 0.10, 0.91, 0.93), h_pad=2)
    fig.savefig(output / "backbone_difference_by_training_seed.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "analysis" / "stage1")
    args = parser.parse_args()
    data = {name: read_results(name, study, plan, epoch, count)
            for name, (study, plan, epoch, count, _) in STUDIES.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    success_plot(data, args.output_dir)
    difference_plot(data, args.output_dir)
    report = {
        "scope": "Fixed-checkpoint progression final tests; not the earlier manuscript evidence",
        "bootstrap": "95% paired-start percentile intervals, conditional on each trained pair; LIBERO resampled within each task",
        "uncertainty_review": {
            "resampling_unit": "Matched reset ID; the two backbone outcomes are resampled together. LIBERO resamples separately within each of ten tasks.",
            "scope": "Intervals describe variation over the evaluated reset cohort and its paired policy-sampling draws conditional on these fixed trained heads. They do not estimate new training-seed variation.",
            "training_seed_reporting": "Three independent head-training seeds are the reviewer-accepted replication level; report each seed and the direction count.",
            "multiplicity": "No confirmatory p-values or multiplicity adjustment were used. The per-seed intervals are descriptive; task-level patterns remain exploratory.",
        },
        "checkpoint_note": "The provenance audit recomputes SHA-256 for all 12 selected checkpoint files and matches each result, locked job, and status record.",
        "status_reconciliation": "Both status files report training_complete_evaluation_pending, completed_policy_count=5, and six pending final tests. All six final records per dataset exist and pass the integrity audit; the operational status files were left unchanged.",
        "training_seed_summary": {},
        "datasets": data,
    }
    for dataset, rows in data.items():
        diffs = [float(r["difference_pp"]) for r in rows]
        report["training_seed_summary"][dataset] = {
            "training_seeds": [int(r["training_seed"]) for r in rows],
            "per_seed_groot_success_percent": [round(100 * r["left_success_rate"], 2) for r in rows],
            "per_seed_qwen_success_percent": [round(100 * r["right_success_rate"], 2) for r in rows],
            "per_seed_groot_minus_qwen_pp": [round(d, 2) for d in diffs],
            "mean_paired_difference_pp": float(statistics.mean(diffs)),
            "sample_sd_of_seed_differences_pp": float(statistics.stdev(diffs)),
            "difference_range_pp": [float(min(diffs)), float(max(diffs))],
            "seeds_favoring_groot": sum(d > 0 for d in diffs),
            "seeds_favoring_qwen": sum(d < 0 for d in diffs),
            "tied_seeds": sum(d == 0 for d in diffs),
        }
    (args.output_dir / "stage1_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote stage 1 figures and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
