"""Reproduce stage 2 descriptive plots from existing progression final records.

Run: python scripts/analysis/plot_stage2_results.py
Requires ignored durable result records, ALOHA trajectories and study statuses.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

from plot_stage1_results import ROOT, SEEDS, STUDIES, read_results

OUT = ROOT / "docs" / "analysis" / "stage2"

def load_policies(dataset):
    study, plan, epoch, n, _ = STUDIES[dataset]
    read_results(dataset, study, plan, epoch, n)  # Use stage 1 matching and provenance checks.
    policies = {}
    for seed in SEEDS:
        for exp in ("exp05", "exp06"):
            path = (ROOT / "asset/durable" / dataset / "evaluation_results" / study / plan
                    / dataset / exp / f"seed_{seed:03d}" / "final_test.json")
            policies[(seed, exp)] = json.loads(path.read_text(encoding="utf-8"))
    return policies


def audit_aloha_trajectory(record):
    info = record["trajectory"]
    path = ROOT / "asset/durable/aloha" / info["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != info["sha256"] or path.stat().st_size != info["size_bytes"]:
        raise ValueError(f"Trajectory size or hash mismatch: {path}")
    rows = record["episodes"]
    stats = Counter()
    with np.load(path, allow_pickle=False) as trace:
        if int(trace["n_episodes"][0]) != len(rows) or not np.array_equal(
                trace["initial_state_ids"], [row["initial_state_id"] for row in rows]):
            raise ValueError(f"Trajectory episode IDs mismatch: {path}")
        for i, row in enumerate(rows):
            rewards = trace[f"ep{i:04d}_reward"]
            if (len(rewards) != row["steps"] or not len(rewards)
                    or int(rewards.max()) != row["max_reward"]
                    or bool(rewards.max() >= 4) != row["is_success"]
                    or not np.isin(rewards, [0, 1, 2, 3, 4]).all()):
                raise ValueError(f"Trajectory reward mismatch: {path}, episode {i}")
            stats["episodes"] += 1
            stats["success_without_observed_score2"] += int(row["is_success"] and not (rewards == 2).any())
            stats["score2_observed"] += int((rewards == 2).any())
            stats["score2_without_observed_score1"] += int((rewards == 2).any() and not (rewards == 1).any())
            stats["score3_observed"] += int((rewards == 3).any())
    return {"path": str(path), "sha256": digest, "reward_checks": dict(stats)}


def aloha_summary(policies):
    summary = []
    audits = []
    for seed in SEEDS:
        g, q = (policies[(seed, exp)] for exp in ("exp05", "exp06"))
        audits.extend({"seed": seed, "experiment": r["experiment"], **audit_aloha_trajectory(r)} for r in (g, q))
        by_start = [{r["initial_state_id"]: r for r in p["episodes"]} for p in (g, q)]
        wins = {"groot_only": [], "qwen_only": [], "both_succeed": 0, "both_fail": 0}
        for key in sorted(by_start[0]):
            gr, qr = (m[key] for m in by_start)
            if gr["is_success"] and not qr["is_success"]:
                wins["groot_only"].append(qr["max_reward"])
            elif qr["is_success"] and not gr["is_success"]:
                wins["qwen_only"].append(gr["max_reward"])
            elif gr["is_success"]:
                wins["both_succeed"] += 1
            else:
                wins["both_fail"] += 1
        summary.append({"seed": seed, "both_succeed": wins["both_succeed"], "both_fail": wins["both_fail"],
                        "groot_only": len(wins["groot_only"]), "qwen_only": len(wins["qwen_only"]),
                        "qwen_failure_max_reward_when_groot_only": [wins["groot_only"].count(i) for i in range(4)],
                        "groot_failure_max_reward_when_qwen_only": [wins["qwen_only"].count(i) for i in range(4)],
                        "net_advantage_by_failed_policy_max_reward":
                            [wins["groot_only"].count(i) - wins["qwen_only"].count(i) for i in range(4)]})
    return summary, audits


def libero_summary(policies):
    tasks, diagnostic = [], []
    for seed in SEEDS:
        for exp in ("exp05", "exp06"):
            rows = [r for r in policies[(seed, exp)]["results"] if r["condition"] == "canonical"]
            failed = [r for r in rows if not r["success"]]
            diagnostic.append({"seed": seed, "experiment": exp, "failed_episodes": len(failed),
                "failure_termination_reasons": dict(Counter(r["termination_reason"] for r in failed)),
                "failed_with_zero_gripper_command_sign_changes": sum(r["gripper_changes"] == 0 for r in failed)})
    for task in range(10):
        entries = []
        for seed in SEEDS:
            g, q = (policies[(seed, exp)] for exp in ("exp05", "exp06"))
            gr = {r["initial_state_id"]: r for r in g["results"] if r["condition"] == "canonical" and r["task"] == task}
            qr = {r["initial_state_id"]: r for r in q["results"] if r["condition"] == "canonical" and r["task"] == task}
            if len(gr) != 40 or gr.keys() != qr.keys():
                raise ValueError(f"Unmatched LIBERO task {task}, seed {seed}")
            gs, qs = sum(r["success"] for r in gr.values()), sum(r["success"] for r in qr.values())
            entries.append({"seed": seed, "groot_success": gs, "qwen_success": qs,
                "difference_pp": 2.5 * (gs - qs),
                "groot_only": sum(gr[k]["success"] and not qr[k]["success"] for k in gr),
                "qwen_only": sum(qr[k]["success"] and not gr[k]["success"] for k in gr)})
        tasks.append({"task": task, "instruction": next(iter(gr.values()))["task_language"],
                      "by_seed": entries, "mean_difference_pp": sum(e["difference_pp"] for e in entries) / 3})
    return tasks, diagnostic


def plot_libero(tasks):
    matrix = np.array([[entry["difference_pp"] for entry in task["by_seed"]] +
                      [task["mean_difference_pp"]] for task in tasks])
    cmap = LinearSegmentedColormap.from_list("qwen_to_groot", ["#e28b4b", "#f4f5f7", "#347eae"])
    fig, ax = plt.subplots(figsize=(11.6, 8))
    image = ax.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30), aspect="auto")
    ax.set_xticks(range(4), ["Seed 3", "Seed 4", "Seed 5", "3-seed mean"])
    ax.set_yticks(range(10), [f"Task {t['task']}: {t['instruction']}" for t in tasks], fontsize=9)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row, task in enumerate(tasks):
        for col in range(4):
            text = f"{matrix[row, col]:+.1f} pp"
            if col < 3:
                entry = task["by_seed"][col]
                text += f"\nG {entry['groot_success']} / Q {entry['qwen_success']}"
            ax.text(col, row, text, ha="center", va="center", fontsize=8.5, color="#18212a")
    ax.set_title("LIBERO-Goal: task-level success differences by training seed", loc="left", fontsize=12,
                 fontweight="bold", pad=12)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("GR00T minus Qwen success (pp): orange favors Qwen, blue favors GR00T")
    fig.text(0.01, 0.01, "G and Q are successes out of 40 matched starts per task and seed; 2.5 pp = one episode.\n"
             "The three-seed mean is descriptive; task-level comparisons have no multiplicity correction.",
             fontsize=8.5, color="#4e5863")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "libero_task_seed_outcomes.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_aloha_outcomes(rows):
    fig, ax = plt.subplots(figsize=(10, 4.7))
    categories = (("both_succeed", "Both succeed", "#829f91"),
                  ("groot_only", "Only GR00T succeeds", "#2475ad"),
                  ("qwen_only", "Only Qwen succeeds", "#d87923"),
                  ("both_fail", "Both fail", "#c4cbd2"))
    for i, row in enumerate(rows):
        left = 0
        for key, label, color in categories:
            count = row[key]
            ax.barh(i, count, left=left, height=0.6, color=color, edgecolor="white",
                    label=label if i == 0 else None)
            if count:
                ax.text(left + count / 2, i, str(count), ha="center", va="center", fontsize=10)
            left += count
        if left != 200:
            raise ValueError(f"Expected 200 matched ALOHA starts for seed {row['seed']}")
    ax.invert_yaxis()
    ax.set_yticks(range(len(rows)), [f"Seed {row['seed']}" for row in rows])
    ax.set_xlim(0, 200)
    ax.set_xlabel("Number of matched starts (200 per seed pair)")
    fig.suptitle("ALOHA: outcomes on the same starts", x=0.13, ha="left", fontweight="bold")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False, fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(0.01, 0.01, "Each bar compares one GR00T and one Qwen head on the same starts.\n"
             "Only-GR00T and only-Qwen counts identify where the difference comes from; they are not failure causes.",
             fontsize=8.5, color="#4e5863")
    fig.tight_layout(rect=(0, 0.12, 1, 0.75))
    fig.savefig(OUT / "aloha_matched_outcomes.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def aloha_reward_evidence(policies):
    records = []
    for seed in SEEDS:
        groot = policies[(seed, "exp05")]
        qwen = policies[(seed, "exp06")]
        result_rows = [{r["initial_state_id"]: r for r in p["episodes"]}
                       for p in (groot, qwen)]
        trace_paths = [ROOT / "asset/durable/aloha" / p["trajectory"]["path"]
                       for p in (groot, qwen)]
        with np.load(trace_paths[0], allow_pickle=False) as gt, np.load(trace_paths[1], allow_pickle=False) as qt:
            if not np.array_equal(gt["initial_state_ids"], qt["initial_state_ids"]):
                raise ValueError(f"Unmatched ALOHA trace start IDs for seed {seed}")
            if len(gt["initial_state_ids"]) != 200:
                raise ValueError(f"Expected 200 ALOHA traces per head for seed {seed}")
            for i, start_id in enumerate(gt["initial_state_ids"].tolist()):
                scores = []
                for trace in (gt, qt):
                    reward = trace[f"ep{i:04d}_reward"]
                    scores.append(int(reward.max()))
                outcomes = tuple(bool(m[start_id]["is_success"]) for m in result_rows)
                if tuple(score == 4 for score in scores) != outcomes:
                    raise ValueError(f"Reward max and success disagree at seed {seed}, start {start_id}")
                records.append({"training_seed": seed, "initial_state_id": start_id,
                                "groot_success": outcomes[0], "qwen_success": outcomes[1],
                                "groot_max_reward": scores[0], "qwen_max_reward": scores[1],
                                "winner": ("GR00T" if outcomes == (True, False) else
                                           "Qwen" if outcomes == (False, True) else
                                           "both" if outcomes == (True, True) else "neither")})
    return records


def plot_aloha_reward_maxima(records):
    labels = ["0: no scored finger contact", "1: right finger contact",
              "2: right finger; cube off table", "3: left finger contact; table state unspecified",
              "4: left finger; cube off table (success)"]
    colors = ["#c4cbd2", "#e6b04d", "#4f8bb6", "#a879bc", "#3a9d7d"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = 0
    positions, ylabels = [], []
    for seed in SEEDS:
        for backbone, key in (("GR00T", "groot_max_reward"), ("Qwen", "qwen_max_reward")):
            subset = [r[key] for r in records if r["training_seed"] == seed]
            counts = [subset.count(i) for i in range(5)]
            left = 0
            for score, count in enumerate(counts):
                ax.barh(y, count, left=left, height=0.68, color=colors[score], edgecolor="white",
                        label=labels[score] if y == 0 else None)
                if count >= 8:
                    ax.text(left + count / 2, y, str(count), ha="center", va="center", fontsize=8)
                left += count
            if left != 200:
                raise ValueError(f"Expected 200 reward traces, got {left}")
            positions.append(y); ylabels.append(f"Seed {seed} · {backbone}"); y += 1
        y += 0.2
    ax.invert_yaxis()
    ax.set_yticks(positions, ylabels)
    ax.set_xlim(0, 200)
    ax.set_xlabel("Episodes out of 200")
    fig.suptitle("ALOHA: highest recorded reward in each episode", x=0.12, ha="left", fontweight="bold")
    fig.legend(loc="upper center", bbox_to_anchor=(0.52, 0.91), ncol=2, frameon=False, fontsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(0.01, 0.01, "Bars classify each episode by its maximum reward; categories are not a required time sequence.\n"
             "Reward checks specific finger contacts and cube-table contact; they do not describe every physical contact.",
             fontsize=8.3, color="#4e5863")
    fig.tight_layout(rect=(0, 0.12, 1, 0.75))
    fig.savefig(OUT / "aloha_reward_maximum_by_seed.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def aloha_reward_events(policies):
    rows = []
    for seed in SEEDS:
        for exp, backbone in (("exp05", "GR00T"), ("exp06", "Qwen")):
            policy = policies[(seed, exp)]
            episodes = policy["episodes"]
            path = ROOT / "asset/durable/aloha" / policy["trajectory"]["path"]
            counts = {"right_finger_contact": 0, "right_finger_cube_off_table": 0,
                      "successful_handoff": 0, "score2_episodes": 0,
                      "score2_then_score4": 0, "success_without_prior_score2": 0,
                      "first_right_contact_steps": [], "score2_to_score4_steps": []}
            with np.load(path, allow_pickle=False) as trace:
                if not np.array_equal(trace["initial_state_ids"],
                                      [episode["initial_state_id"] for episode in episodes]):
                    raise ValueError(f"Trace order mismatch for seed {seed} {backbone}")
                for i, episode in enumerate(episodes):
                    reward = trace[f"ep{i:04d}_reward"]
                    contact = np.flatnonzero(np.isin(reward, [1, 2]))
                    counts["right_finger_contact"] += int(len(contact) > 0)
                    counts["right_finger_cube_off_table"] += int((reward == 2).any())
                    success = bool((reward == 4).any())
                    if success != bool(episode["is_success"]):
                        raise ValueError(f"Reward 4 and outcome disagree for seed {seed} {backbone}")
                    counts["successful_handoff"] += int(success)
                    score2 = np.flatnonzero(reward == 2)
                    score4 = np.flatnonzero(reward == 4)
                    counts["score2_episodes"] += int(len(score2) > 0)
                    if len(score4):
                        if not len(score2) or int(score2[0]) >= int(score4[0]):
                            counts["success_without_prior_score2"] += 1
                        else:
                            counts["score2_then_score4"] += 1
                            counts["score2_to_score4_steps"].append(
                                int(score4[0]) - int(score2[0]))
                    if len(contact):
                        counts["first_right_contact_steps"].append(int(contact[0]) + 1)
            rows.append({"training_seed": seed, "backbone": backbone, "episodes": len(episodes),
                         "right_finger_contact": counts["right_finger_contact"],
                         "right_finger_cube_off_table": counts["right_finger_cube_off_table"],
                         "successful_handoff": counts["successful_handoff"],
                         "score2_episodes": counts["score2_episodes"],
                         "score2_then_score4": counts["score2_then_score4"],
                         "share_score2_episodes_later_reaching_score4": (
                             counts["score2_then_score4"] / counts["score2_episodes"]
                             if counts["score2_episodes"] else None),
                         "success_without_prior_score2": counts["success_without_prior_score2"],
                         "median_steps_from_first_score2_to_first_score4": (
                             float(np.median(counts["score2_to_score4_steps"]))
                             if counts["score2_to_score4_steps"] else None),
                         "median_first_right_contact_step": float(np.median(counts["first_right_contact_steps"]))
                         if counts["first_right_contact_steps"] else None})
    return rows


def plot_aloha_reward_events(rows):
    categories = (("right_finger_contact", "Right-finger\ncontact"),
                  ("right_finger_cube_off_table", "Right-finger contact;\ncube off table"),
                  ("successful_handoff", "Left-finger contact;\ncube off table (success)"))
    colors = {"GR00T": "#2475ad", "Qwen": "#d87923"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    x = np.arange(len(categories)); width = 0.34
    for ax, seed in zip(axes, SEEDS):
        for offset, backbone in ((-width/2, "GR00T"), (width/2, "Qwen")):
            row = next(r for r in rows if r["training_seed"] == seed and r["backbone"] == backbone)
            values = [100 * row[key] / row["episodes"] for key, _ in categories]
            bars = ax.bar(x + offset, values, width, color=colors[backbone],
                          label=backbone if seed == SEEDS[0] else None)
            ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=2, fontsize=8)
        ax.set_title(f"Training seed {seed}")
        ax.set_xticks(x, [label for _, label in categories], fontsize=8)
        ax.set_ylim(0, 115)
        ax.grid(axis="y", color="#dce1e7", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Episodes with event observed at least once (%)")
    fig.suptitle("ALOHA: how often each scored contact condition occurs", fontsize=13, fontweight="bold")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=2, frameon=False)
    fig.text(0.01, 0.01, "Each percentage is out of 200 episodes. Events are counted from the full per-step reward trace,\n"
             "not inferred from the maximum-reward category; they are not assumed to form a fixed sequence.",
             fontsize=8.5, color="#4e5863")
    fig.tight_layout(rect=(0, 0.12, 1, 0.78))
    fig.savefig(OUT / "aloha_reward_event_rates_by_seed.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        for x, backbone, color in ((0, "GR00T", colors["GR00T"]),
                                   (1, "Qwen", colors["Qwen"])):
            row = next(r for r in rows if r["training_seed"] == seed and r["backbone"] == backbone)
            value = 100 * row["share_score2_episodes_later_reaching_score4"]
            ax.bar(x, value, color=color, width=.58)
            ax.text(x, value + 2, f"{value:.1f}%\n{row['score2_then_score4']}/{row['score2_episodes']}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks([0, 1], ["GR00T", "Qwen"])
        ax.set_title(f"Training seed {seed}")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#dce1e7", linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Episodes with score 2 that later reach score 4 (%)")
    fig.suptitle("ALOHA: completion after the right-finger/off-table condition",
                 fontsize=12.5, fontweight="bold")
    fig.text(.01, .01, "All recorded score-4 episodes followed score 2 in these traces. This is a simulator predicate transition, not a physical grasp diagnosis.",
             color="#4e5863", fontsize=8.2)
    fig.tight_layout(rect=(0, .06, 1, .92))
    fig.savefig(OUT / "aloha_handoff_conditional_by_seed.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_aloha_reward_event_table(rows):
    with (OUT / "aloha_reward_event_rates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def write_aloha_discordant_table(records):
    rows = []
    for seed in SEEDS:
        for winner, success_key, failed_reward_key in (
                ("GR00T only", "groot_success", "qwen_max_reward"),
                ("Qwen only", "qwen_success", "groot_max_reward")):
            subset = [r for r in records if r["training_seed"] == seed and
                      ((winner == "GR00T only" and r["winner"] == "GR00T") or
                       (winner == "Qwen only" and r["winner"] == "Qwen"))]
            counts = [sum(r[failed_reward_key] == score for r in subset) for score in range(4)]
            rows.append({"training_seed": seed, "only_successful_backbone": winner.split()[0],
                         "discordant_starts": len(subset), **{f"failed_policy_max_reward_{i}": v
                         for i, v in enumerate(counts)}})
    with (OUT / "aloha_discordant_reward_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    discordant = [r for r in records if r["winner"] in ("GR00T", "Qwen")]
    with (OUT / "aloha_discordant_start_rewards.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=discordant[0].keys())
        writer.writeheader(); writer.writerows(discordant)


def write_libero_task_table(tasks):
    with (OUT / "libero_task_seed_counts.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["task", "instruction", "training_seed", "groot_successes", "qwen_successes",
                  "groot_only", "qwen_only", "difference_pp"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            for row in task["by_seed"]:
                writer.writerow({"task": task["task"], "instruction": task["instruction"],
                                 "training_seed": row["seed"], "groot_successes": row["groot_success"],
                                 "qwen_successes": row["qwen_success"], "groot_only": row["groot_only"],
                                 "qwen_only": row["qwen_only"], "difference_pp": row["difference_pp"]})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    aloha_policies = load_policies("aloha")
    aloha, audits = aloha_summary(aloha_policies)
    reward_records = aloha_reward_evidence(aloha_policies)
    reward_events = aloha_reward_events(aloha_policies)
    libero, diagnostics = libero_summary(load_policies("libero"))
    plot_aloha_outcomes(aloha)
    plot_aloha_reward_maxima(reward_records)
    plot_aloha_reward_events(reward_events)
    write_aloha_discordant_table(reward_records)
    write_aloha_reward_event_table(reward_events)
    plot_libero(libero)
    write_libero_task_table(libero)
    report = {"scope": "Progression held-out tests: ALOHA epoch 250; LIBERO epoch 120",
              "aloha": aloha, "aloha_trajectory_audit": audits,
              "aloha_reward_maximum_by_seed": {str(seed): {
                  backbone: [sum(r["training_seed"] == seed and r[key] == i for r in reward_records)
                             for i in range(5)]
                  for backbone, key in (("GR00T", "groot_max_reward"), ("Qwen", "qwen_max_reward"))}
                  for seed in SEEDS},
              "aloha_reward_event_rates": reward_events,
              "aloha_conditional_handoff_plot": "aloha_handoff_conditional_by_seed.png",
              "aloha_discordant_reward_table": "aloha_discordant_reward_counts.csv",
              "aloha_discordant_start_records": "aloha_discordant_start_rewards.csv",
              "libero_tasks": libero, "libero_failure_record_diagnostics": diagnostics,
              "limitations": ["Three independent training-seed pairs; LIBERO has 40 starts per task and pair",
                              "Task comparisons are exploratory and not adjusted for multiple comparisons",
                              "Reward categories and selected videos do not establish failure causes or their prevalence"]}
    (OUT / "stage2_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote stage 2 figures and summary to {OUT}")


if __name__ == "__main__":
    main()
