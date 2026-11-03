"""Compare offline validation loss with development rollout checkpoint ranking.

Run from the project root with the existing ``vla`` environment. This reads
training histories and saved development probes; it does not load models or
run training/rollouts.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
STUDIES = {
    "aloha": ("aloha_progression_250_v1", "415a08a2c6f2",
              {"exp05": "groot_transfer", "exp06": "qwen3vl_transfer"}),
    "libero": ("libero_progression_120_v1", "c6854e3ff020",
               {"exp05": "groot_2view", "exp06": "qwen3vl_2view"}),
}
SEEDS = (3, 4, 5)


def load_head(dataset: str, study: str, plan: str, exp: str,
              folder: str, seed: int) -> tuple[list[dict], list[dict]]:
    run = ROOT / "asset" / "runs" / study / folder / f"seed_{seed:03d}"
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()
               if line.strip()]
    loss_by_epoch = {int(r["epoch"]): r for r in history
                     if r.get("validation_loss") is not None}
    result_dir = (ROOT / "asset" / "durable" / dataset / "evaluation_results"
                  / study / plan / dataset / exp / f"seed_{seed:03d}")
    records = list(result_dir.glob("progress_epoch_*.json"))
    records += list(result_dir.glob("selection_epoch_*.json"))
    success_by_epoch = {}
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("purpose") not in ("progress_probe", "development_selection"):
            continue
        epoch = int(record["checkpoint_epoch"])
        if epoch in success_by_epoch:
            raise ValueError(f"Duplicate rollout probe at {dataset}/{exp}/seed{seed}/epoch{epoch}")
        episodes = (int(record["n_episodes"]) if dataset == "aloha"
                    else len(record.get("results", [])))
        success_by_epoch[epoch] = (float(record["success_rate"]), episodes,
                                   str(record.get("purpose")))
    epochs = sorted(set(loss_by_epoch) & set(success_by_epoch))
    if len(epochs) < 5:
        raise ValueError(f"Too few aligned anchors for {dataset}/{exp}/seed{seed}: {epochs}")
    rows = [{"dataset": dataset, "experiment": exp,
             "backbone": "GR00T" if exp == "exp05" else "Qwen",
             "training_seed": seed, "epoch": epoch,
             "train_loss": float(loss_by_epoch[epoch]["train_loss"]),
             "validation_loss": float(loss_by_epoch[epoch]["validation_loss"]),
             "development_success": success_by_epoch[epoch][0],
             "development_episodes": success_by_epoch[epoch][1],
             "probe_purpose": success_by_epoch[epoch][2]}
            for epoch in epochs]
    rho = float(spearmanr([r["validation_loss"] for r in rows],
                          [r["development_success"] for r in rows]).statistic)
    min_loss = min(rows, key=lambda r: (r["validation_loss"], r["epoch"]))
    max_success = max(rows, key=lambda r: (r["development_success"], -r["epoch"]))
    summary = {
        "dataset": dataset, "experiment": exp, "backbone": rows[0]["backbone"],
        "training_seed": seed, "n_anchors": len(rows), "epochs": epochs,
        "development_episodes_per_anchor": sorted({r["development_episodes"] for r in rows}),
        "probe_purposes": sorted({r["probe_purpose"] for r in rows}),
        "spearman_validation_loss_vs_development_success": rho,
        "minimum_validation_loss_epoch": min_loss["epoch"],
        "development_success_at_minimum_loss": min_loss["development_success"],
        "best_development_epoch_earliest_tie": max_success["epoch"],
        "best_development_success": max_success["development_success"],
        "success_regret_of_minimum_loss_checkpoint_pp": 100 * (
            max_success["development_success"] - min_loss["development_success"]),
        "training_loss_first_last": [rows[0]["train_loss"], rows[-1]["train_loss"]],
        "validation_loss_first_last": [rows[0]["validation_loss"], rows[-1]["validation_loss"]],
    }
    return rows, summary


def main() -> None:
    output = ROOT / "docs" / "analysis" / "stage5"
    output.mkdir(parents=True, exist_ok=True)
    all_rows, summaries = [], []
    for dataset, (study, plan, folders) in STUDIES.items():
        for exp, folder in folders.items():
            for seed in SEEDS:
                rows, summary = load_head(dataset, study, plan, exp, folder, seed)
                all_rows.extend(rows)
                summaries.append(summary)

    with (output / "checkpoint_anchor_alignment.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    (output / "checkpoint_alignment_summary.json").write_text(
        json.dumps({"scope": "Development checkpoint probes only; no held-out final-test data used.",
                    "rho_note": "Spearman correlation between validation loss and rollout success within each trained head; anchors are not independent replicates.",
                    "heads": summaries,
                    "median_spearman_by_dataset": {
                        ds: statistics.median(s["spearman_validation_loss_vs_development_success"]
                                              for s in summaries if s["dataset"] == ds)
                        for ds in STUDIES},
                    "median_loss_checkpoint_regret_pp_by_dataset": {
                        ds: statistics.median(s["success_regret_of_minimum_loss_checkpoint_pp"]
                                              for s in summaries if s["dataset"] == ds)
                        for ds in STUDIES}}, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    colors = {"GR00T": "#2475ad", "Qwen": "#d87923"}
    for ax, ds, title in zip(axes, ("aloha", "libero"), ("ALOHA", "LIBERO-Goal")):
        subset = [s for s in summaries if s["dataset"] == ds]
        x = np.arange(len(subset))
        for i, s in enumerate(subset):
            color = colors[s["backbone"]]
            a, b = 100 * s["development_success_at_minimum_loss"], 100 * s["best_development_success"]
            ax.plot([i, i], [a, b], color="#b8c0c8", lw=2, zorder=1)
            ax.scatter(i - 0.06, a, color=color, marker="o", s=48, zorder=2)
            ax.scatter(i + 0.06, b, color=color, marker="D", s=42, zorder=2)
            ax.text(i, max(a, b) + 3, f"+{b-a:.0f} pp", ha="center", fontsize=8)
        ax.set_title(title)
        ax.set_xticks(x, [f"{s['backbone'][0]}{s['training_seed']}" for s in subset])
        ax.set_ylim(0, 100); ax.grid(axis="y", color="#dce1e7", linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False); ax.set_axisbelow(True)
    axes[0].set_ylabel("Development rollout success (%)")
    fig.suptitle("Checkpoint chosen by minimum validation loss vs best development success", fontweight="bold")
    fig.text(.5, .015, "Circle: success at lowest-loss anchor. Diamond: highest development success (earliest tie).",
             ha="center", color="#4e5863", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .92))
    fig.savefig(output / "loss_checkpoint_success_regret.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    labels, vals, cols = [], [], []
    for ds in ("aloha", "libero"):
        for s in summaries:
            if s["dataset"] == ds:
                labels.append(f"{ds.title()} {s['backbone'][0]} seed {s['training_seed']}")
                vals.append(s["spearman_validation_loss_vs_development_success"])
                cols.append(colors[s["backbone"]])
    y = np.arange(len(vals))
    ax.axvline(0, color="#58616b", lw=1)
    ax.scatter(vals, y, c=cols, s=58)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman ρ: validation loss vs development success")
    ax.set_title("Within-head checkpoint ranking agreement", fontweight="bold")
    ax.grid(axis="x", color="#dce1e7", linewidth=.8); ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(.01, .01, "Positive ρ means higher loss tends to coincide with higher rollout success at these anchors.",
             color="#4e5863", fontsize=8.5)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(output / "loss_success_rank_correlation.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote Stage 5 alignment plots and tables to {output}")


if __name__ == "__main__":
    main()
