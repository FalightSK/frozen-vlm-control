"""Summarize the 12 saved, outcome-stratified LIBERO transition probes.

The input contains selected reruns of existing epoch-120 checkpoints. These
examples help explain trajectories but are not a representative sample and
must not be used to estimate failure-stage prevalence.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COLORS = {"groot": "#2475ad", "qwen3vl": "#d87923"}
LABELS = {"groot": "GR00T", "qwen3vl": "Qwen"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=ROOT / "asset/durable/libero/diagnostics/stage2_transition_probe_v5/stage2_transition_probe.json")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "docs/analysis/stage2")
    args = ap.parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if len(records) != 12:
        raise ValueError(f"Expected 12 selected transition episodes, found {len(records)}")
    if any(not r.get("initial_observation_matches") or r.get("success") != r.get("original_success")
           for r in records):
        raise ValueError("Diagnostic rerun is inconsistent with the original saved result")

    rows = []
    for record in records:
        trace = record["trajectory"]
        d = np.asarray([step["target_distance"] for step in trace], dtype=float)
        xy = np.asarray([step["target_xy_distance"] for step in trace], dtype=float)
        goal_steps = [int(step["step"]) for step in trace if any(step["goal_predicates"])]
        contact_steps = [int(step["step"]) for step in trace if step["target_contact"]]
        inside_steps = [int(step["step"]) for step in trace if step["inside_target_region"]]
        rows.append({
            "task": record["task"], "training_seed": record["training_seed"],
            "backbone": LABELS[record["backbone"]], "start_id": record["initial_state_id"],
            "success": record["success"], "steps": record["rerun_steps"],
            "step_count_matches_original": record["rerun_steps"] == record["original_steps"],
            "initial_target_distance": float(d[0]), "minimum_target_distance": float(d.min()),
            "final_target_distance": float(d[-1]), "minimum_target_distance_step": int(np.argmin(d) + 1),
            "initial_xy_distance": float(xy[0]), "minimum_xy_distance": float(xy.min()),
            "final_xy_distance": float(xy[-1]),
            "first_goal_predicate_step": min(goal_steps) if goal_steps else None,
            "first_target_contact_step": min(contact_steps) if contact_steps else None,
            "first_inside_target_region_step": min(inside_steps) if inside_steps else None,
            "initial_observation_matches_original": record["initial_observation_matches"],
        })

    output.mkdir(parents=True, exist_ok=True)
    with (output / "libero_selected_transition_probe_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4), sharex=True)
    for ri, task in enumerate((6, 9)):
        for ci, seed in enumerate((3, 4, 5)):
            ax = axes[ri, ci]
            pair = [r for r in records if r["task"] == task and r["training_seed"] == seed]
            if len(pair) != 2 or len({r["initial_state_id"] for r in pair}) != 1:
                raise ValueError(f"Expected one matched GR00T/Qwen pair for task {task}, seed {seed}")
            for record in sorted(pair, key=lambda r: r["backbone"]):
                trace = record["trajectory"]
                xs = [step["step"] for step in trace]
                ys = [step["target_distance"] for step in trace]
                backbone = record["backbone"]
                ax.plot(xs, ys, color=COLORS[backbone],
                        linestyle="-" if backbone == "groot" else "--", linewidth=1.8,
                        label=LABELS[backbone] if (ri, ci) == (0, 0) else None)
                goal_steps = [step["step"] for step in trace if any(step["goal_predicates"])]
                if goal_steps:
                    step_index = goal_steps[0] - 1
                    ax.scatter(goal_steps[0], ys[step_index], color=COLORS[backbone],
                               marker="*", s=85, edgecolor="white", linewidth=.5, zorder=4)
                else:
                    ax.scatter(xs[-1], ys[-1], color=COLORS[backbone], marker="x", s=45,
                               linewidth=1.5, zorder=4)
            ax.set_title(f"Task {task} · seed {seed} · start {pair[0]['initial_state_id']}")
            ax.grid(alpha=.23)
            ax.spines[["top", "right"]].set_visible(False)
            if ci == 0:
                ax.set_ylabel("Object-to-target distance (m)")
            if ri == 1:
                ax.set_xlabel("Environment step")
    fig.legend(loc="upper center", bbox_to_anchor=(.5, .985), ncol=2, frameon=False)
    fig.suptitle("Selected LIBERO transitions: the winning policy reaches the goal predicate",
                 y=1.03, fontweight="bold")
    fig.text(.01, .005,
             "Selected discordant examples, not a failure-rate sample. Star = BDDL goal first reached; × = horizon endpoint. "
             "Distance alone does not encode all goal geometry/contact requirements.",
             color="#4e5863", fontsize=8.5)
    fig.tight_layout(rect=(0, .035, 1, .93))
    fig.savefig(output / "libero_selected_transition_probe.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_selected_episodes": len(records),
        "selection": "Task 6 GR00T-only successes (one per seed) and task 9 Qwen-only successes (one per seed), with their matched failures.",
        "all_initial_observations_and_outcomes_reproduced": True,
        "rerun_step_counts_matching_original": sum(r["step_count_matches_original"] for r in rows),
        "step_count_note": "Some successful reruns terminate a few steps earlier or later despite identical start hashes and the same binary outcome; use canonical final-test records for episode counts and steps.",
        "records": rows,
        "scope_limit": "This selected sample explains examples only; it does not estimate how often any transition or failure stage occurs in canonical LIBERO final tests.",
    }
    (output / "libero_selected_transition_probe_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote selected LIBERO transition table and plot to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
