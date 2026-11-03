"""Plot matched canonical and counterfactual-goal results from Stage 3."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = "libero_progression_120_v1"
PLAN = "c6854e3ff020"
EXPERIMENTS = {"groot": "exp05", "qwen3vl": "exp06"}
BACKBONE_LABELS = {"groot": "GR00T", "qwen3vl": "Qwen"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "docs/analysis/stage3")
    args = ap.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    raw_path = run_dir / "instruction_swap_trajectories.jsonl.gz"
    rows = []
    with gzip.open(raw_path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"No completed episode records in {raw_path}")

    canonical = {}
    for seed in sorted({r["training_seed"] for r in rows}):
        for backbone, exp in EXPERIMENTS.items():
            path = (ROOT / "asset/durable/libero/evaluation_results" / STUDY / PLAN /
                    "libero" / exp / f"seed_{seed:03d}" / "final_test.json")
            result = json.loads(path.read_text(encoding="utf-8"))
            canonical[(seed, backbone)] = {
                (r["task"], r["initial_state_id"]): r
                for r in result["results"] if r["condition"] == "canonical"
            }

    groups = {}
    transitions = {}
    for row in rows:
        key = (row["task"], row["training_seed"], row["backbone"])
        group = groups.setdefault(key, {
            "n": 0, "canonical": 0, "swapped": 0,
            "original_under_swap": 0, "neither_under_swap": 0,
            "both": 0, "canonical_only": 0, "swapped_only": 0, "neither": 0,
        })
        base = canonical[(row["training_seed"], row["backbone"])][
            (row["task"], row["initial_state_id"])
        ]
        if (not row["initial_state_matches_canonical"]
                or not row["initial_observation_matches_canonical"]):
            raise ValueError(f"Matched-start check failed: {key}, start={row['initial_state_id']}")
        group["n"] += 1
        group["canonical"] += int(base["success"])
        group["swapped"] += int(row["requested_goal_success"])
        group["original_under_swap"] += int(row["canonical_goal_success_under_swap"])
        group["neither_under_swap"] += int(
            not row["requested_goal_success"] and not row["canonical_goal_success_under_swap"])
        transition = transitions.setdefault(key, {"both": 0, "canonical_only": 0,
                                                   "swapped_only": 0, "neither": 0})
        if base["success"] and row["requested_goal_success"]:
            transition["both"] += 1
        elif base["success"]:
            transition["canonical_only"] += 1
        elif row["requested_goal_success"]:
            transition["swapped_only"] += 1
        else:
            transition["neither"] += 1
    for key, group in groups.items():
        group.update(transitions[key])

    swapped_by_head = {}
    for row in rows:
        swapped_by_head.setdefault((row["task"], row["training_seed"], row["backbone"]), {})[
            row["initial_state_id"]
        ] = bool(row["requested_goal_success"])
    matched_comparisons = []
    for task in sorted({r["task"] for r in rows}):
        for seed in sorted({r["training_seed"] for r in rows}):
            groot = swapped_by_head[(task, seed, "groot")]
            qwen = swapped_by_head[(task, seed, "qwen3vl")]
            starts = sorted(set(groot) & set(qwen))
            both = sum(groot[s] and qwen[s] for s in starts)
            groot_only = sum(groot[s] and not qwen[s] for s in starts)
            qwen_only = sum(qwen[s] and not groot[s] for s in starts)
            neither = sum(not groot[s] and not qwen[s] for s in starts)
            matched_comparisons.append({
                "task": task, "training_seed": seed, "n_matched_starts": len(starts),
                "both_reach_requested_goal": both,
                "groot_only": groot_only, "qwen_only": qwen_only,
                "neither": neither,
                "groot_minus_qwen_pp": 100.0 * (groot_only - qwen_only) / len(starts),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "scope": "Matched held-out starts, canonical outcomes reused from final tests; swapped requested-goal outcomes from Stage 3 reruns.",
        "n_records": len(rows),
        "all_matched_initial_states_and_images": True,
        "groups": [
            {"task": key[0], "training_seed": key[1], "backbone": key[2], **value,
             "canonical_rate": value["canonical"] / value["n"],
             "swapped_requested_goal_rate": value["swapped"] / value["n"],
             "original_goal_rate_under_swap": value["original_under_swap"] / value["n"]}
            for key, value in sorted(groups.items())
        ],
        "matched_backbone_swapped_goal_comparisons": matched_comparisons,
    }
    summary_path = args.output_dir / "stage3_instruction_sensitivity.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_path = args.output_dir / "stage3_instruction_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(summary["groups"][0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["groups"])

    matched_csv = args.output_dir / "stage3_matched_backbone_swapped_goal.csv"
    with matched_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(matched_comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(matched_comparisons)

    tasks = sorted({r["task"] for r in rows})
    seeds = sorted({r["training_seed"] for r in rows})
    fig, axes = plt.subplots(len(tasks), len(seeds), figsize=(12.4, 7.0),
                             sharey=True, squeeze=False)
    colors = {"canonical": "#596579", "swapped": "#2A9D8F"}
    for ti, task in enumerate(tasks):
        for si, seed in enumerate(seeds):
            ax = axes[ti, si]
            selected = {k[2]: g for k, g in groups.items() if k[:2] == (task, seed)}
            xs = np.arange(2)
            width = 0.34
            for bi, backbone in enumerate(("groot", "qwen3vl")):
                g = selected[backbone]
                x = xs[bi]
                for offset, field, color, label in (
                    (-width / 2, "canonical", colors["canonical"], "Canonical instruction"),
                    (width / 2, "swapped", colors["swapped"], "Swapped goal reached"),
                ):
                    n = g["n"]
                    count = g[field]
                    ax.bar(x + offset, count / n, width, color=color,
                           label=label if (ti, si, bi) == (0, 0, 0) else None)
                    ax.text(x + offset, count / n + 0.025, f"{count}/{n}",
                            ha="center", va="bottom", fontsize=8)
            ax.set_xticks(xs, ["GR00T", "Qwen"])
            ax.set_ylim(0, 1.12)
            ax.set_title(f"Task {task} · seed {seed}", fontsize=10)
            ax.grid(axis="y", alpha=0.25)
            if si == 0:
                ax.set_ylabel("Requested goal success")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=2, frameon=False)
    fig.suptitle("Canonical versus swapped instruction on matched LIBERO starts", y=1.07, fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output_dir / "stage3_canonical_vs_swapped_success.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(tasks), len(seeds), figsize=(12.4, 7.0),
                             sharey=True, squeeze=False)
    outcome_fields = (
        ("swapped", "Requested alternate goal", "#2A9D8F"),
        ("original_under_swap", "Original goal first", "#596579"),
        ("neither_under_swap", "Neither within horizon", "#D97757"),
    )
    for ti, task in enumerate(tasks):
        for si, seed in enumerate(seeds):
            ax = axes[ti, si]
            selected = {k[2]: g for k, g in groups.items() if k[:2] == (task, seed)}
            bottom = np.zeros(2)
            for field, label, color in outcome_fields:
                vals = [selected[b][field] / selected[b]["n"] for b in ("groot", "qwen3vl")]
                ax.bar([0, 1], vals, bottom=bottom, color=color, label=label)
                for x, val, base, backbone in zip([0, 1], vals, bottom, ("groot", "qwen3vl")):
                    count = selected[backbone][field]
                    if count >= 3:
                        ax.text(x, base + val / 2, str(count), ha="center", va="center",
                                color="white", fontsize=8, fontweight="bold")
                bottom += np.asarray(vals)
            ax.set_xticks([0, 1], ["GR00T", "Qwen"])
            ax.set_ylim(0, 1)
            ax.set_title(f"Task {task} · seed {seed}", fontsize=10)
            ax.grid(axis="y", alpha=0.25)
            if si == 0:
                ax.set_ylabel("Outcome under swapped instruction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015),
               ncol=3, frameon=False)
    fig.suptitle("Which goal did the policy reach after the instruction swap?", y=1.07, fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output_dir / "stage3_swapped_goal_outcomes.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(tasks), figsize=(8.5, 4.0), sharey=True, squeeze=False)
    for i, task in enumerate(tasks):
        ax = axes[0, i]
        points = [r for r in matched_comparisons if r["task"] == task]
        values = [r["groot_minus_qwen_pp"] for r in points]
        colors = ["#2A9D8F" if x > 0 else "#D97757" if x < 0 else "#596579" for x in values]
        ax.axhline(0, color="#333333", linewidth=1)
        ax.scatter([r["training_seed"] for r in points], values, s=85, c=colors, zorder=3)
        for row, value in zip(points, values):
            ax.annotate(f"{value:+.1f}", (row["training_seed"], value),
                        xytext=(0, 8 if value >= 0 else -15), textcoords="offset points",
                        ha="center", fontsize=9)
        ax.set_ylim(-22, 32)
        ax.set_xticks(seeds, [str(s) for s in seeds])
        ax.set_title(f"Task {task}")
        ax.set_xlabel("Training seed")
        ax.grid(axis="y", alpha=0.25)
        if i == 0:
            ax.set_ylabel("GR00T − Qwen success (percentage points)")
    fig.suptitle("Matched-start backbone difference under the swapped goal", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output_dir / "stage3_matched_backbone_difference.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {summary_path}, {csv_path}, {matched_csv}, and Stage 3 plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
