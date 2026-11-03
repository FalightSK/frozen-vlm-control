# Local asset layout

`asset/` contains local experiment data and generated files. These folders are
ignored by Git, so keep the study manifests and recovery instructions in the
tracked project files.

| Folder | Role | Keep/cleanup rule |
|---|---|---|
| `asset/cache/precomputed/<dataset>/<experiment>/` | Published frozen VLM features and verification manifests. `aloha/` and `libero/` each contain `exp05` (GR00T) and `exp06` (Qwen); all seeds reuse the same features. | Keep the published caches while a run may use them. The legacy cache producer ID remains in each manifest for provenance. |
| `asset/data/` | Local demonstrations. `aloha_sim_transfer_cube_human/` and `aloha_sim_insertion_human/` are the two ALOHA datasets; `libero/libero_goal/` holds LIBERO Goal episodes; `pusht/` supports the PushT experiments. Dataset `data/`, `meta/`, and `videos/` folders contain the named data, statistics/manifests, and source videos. | Keep while a run or recovery may need the source demonstrations. |
| `asset/models/` | Pinned pretrained backbone snapshots. `groot_n17_3b/` and `qwen3_vl_2b/` support the current comparison; `cosmos_reason2_2b/` supplies GR00T tokenizer assets; `pi05_base/`, `paligemma_3b_pt_224/`, `smolvla_base/`, and `smolvlm2_500m/` support prior experiments. | Keep snapshots and their small Hugging Face download metadata for offline resume. |
| `asset/protocols/<dataset>/` | Hash-named dataset split and evaluation-cohort manifests, separated by dataset. | Treat manifests as immutable inputs; keep them with their runs. |
| `asset/runs/<study-id>/` | Training outputs arranged by backbone and `seed_NNN`: focused and midpoint weights, recovery checkpoints, histories, logs, and summaries. `aloha_progression_250_v1/` is the completed six-policy training progression. | Preserve the full progression and recovery files for every policy. Delete a run only when a complete replacement and its evaluations are verified. |
| `asset/durable/<dataset>/` | Durable launcher status/logs; `controller_state/` stores selection and stopping state; `evaluation_tasks/`, `evaluation_status/`, and `evaluation_results/` are the queue, outcomes, and attempt state; `evaluation_checkpoints/` and `evaluation_trajectories/` hold inputs and captured rollouts; `success_studies/<study-id>/` holds cache reports, manifests, and recovery revisions. | Keep completed records and recovery revisions. This is the authoritative local evaluation and resume record. |
| `asset/eval_worker_local/<task-hash>/` and `asset/evaluation_scratch/<task-hash>/` | Temporary evaluator work files and per-attempt logs; the former is legacy scratch. | A task's scratch is removable only after its durable status is `complete` and its durable result exists. |

The completed ALOHA progression is `asset/runs/aloha_progression_250_v1/`.
The paused LIBERO campaign stores its verified cache recovery copy, manifests,
and launcher records at `asset/durable/libero/`. Its working cache is at
`asset/cache/precomputed/libero/`. New evaluations use
`asset/evaluation_scratch/<task-hash>/`.

For new studies, use `<dataset>_progression_<budget>_v1` as the study ID, for
example `libero_progression_120_v1`. Cache producer IDs such as
`runpod_success_stopped_v6_libero` remain in manifests and plans as provenance
identifiers, not folder names.
