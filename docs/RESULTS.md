# frozen-vlm-control: experiment results

## Primary held-out comparison

Both endpoints are frozen at language layer 16. Heads are independently trained with seeds 3, 4, and 5 and matched in architecture within each benchmark. ALOHA final tests use epoch 250 and 200 starts per head; LIBERO final tests use epoch 120 and 400 starts per head. Development probes and selected diagnostics are separate.

| Benchmark | GR00T seeds 3 / 4 / 5 | Qwen seeds 3 / 4 / 5 | Means: GR00T / Qwen |
|---|---|---|---|
| ALOHA | 49.5 / 42.5 / 64.5% | 34.5 / 18.0 / 33.5% | 52.2 / 28.7% |
| LIBERO-Goal | 89.0 / 88.8 / 91.8% | 89.8 / 91.8 / 88.0% | 89.8 / 89.8% |

The three independent training heads are the replication unit. Multiple rollouts or checkpoints from a head are dependent training observations. The paired-start bootstrap intervals describe reset-cohort uncertainty conditional on the trained heads. Equal LIBERO means are not an equivalence test.

Sources: [seed summary](analysis/stage1/stage1_results.json), [checkpoint/record integrity audit](analysis/provenance/progression_integrity_audit.json), and [per-task counts](analysis/stage2/libero_task_seed_counts.csv).

## Behavioral findings

- **ALOHA stage completion:** among episodes reaching reward 2, the shares later reaching reward 4 are GR00T/Qwen 60.7/42.3%, 55.6/25.4%, and 69.0/47.5% for seeds 3–5. Reward stages are simulator predicates.
- **ALOHA timing:** on matched starts where both reach reward 2 but only GR00T succeeds, median paired Qwen-minus-GR00T delays are 34, 61, and 124 steps (35, 39, and 39 starts). Two selected Qwen timeouts succeed at steps 489 and 465 after extension beyond 400 steps. These samples do not estimate recovery prevalence or establish a causal timing effect.
- **LIBERO familiar goals:** the two-scene comparisons favor Qwen on wine-to-rack and GR00T or a tie on cheese-to-bowl. Task 6's original goal is cheese-to-bowl; its swapped goal is wine-to-rack. Task 9 has the reverse mapping. Object and destination change together.
- **Action and feature probes:** matched observations can produce different gripper predictions, with differences varying by seed and state. Alternative predictions were not executed. Selected image masks affect features and actions but do not explain the backbone success gap.

The [manuscript](../manuscript/main.tex) reports the selected LIBERO repeat audit separately; its 72 diagnostic episodes do not change the locked final tests.

## Offline loss

Validation flow loss reaches its minimum at epoch 25 for all twelve heads, while development success often improves later. Loss alone did not reliably rank checkpoint success in these runs. See [anchor data](analysis/stage5/checkpoint_anchor_alignment.csv). These curves do not prove overfitting or establish held-out test performance at every anchor.

## Earlier evidence

The earlier epoch-100 ALOHA comparison averages 49.3% versus 39.8%, with one seed pair favoring Qwen. The earlier LIBERO seed-3 comparison favors Qwen 91.5% versus 84.25% with unequal development-selected budgets. Neither is pooled with the progression campaign. The older component-ablation narrative is retained as a [historical record](archive/historical_results.md), not as the current paper's conclusion.
