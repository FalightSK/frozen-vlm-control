# Manuscript

`main.tex` is the current IEEE-format draft with a six-page compile guard. It compares the frozen
Qwen3-VL-2B and GR00T N1.7 vision--language endpoints using newly trained
action heads. The main results section reports the separate, three-seed
progression campaign (ALOHA at epoch 250 and LIBERO-Goal at epoch 120), then
keeps the earlier fixed-epoch ALOHA and single-seed LIBERO results clearly
separate.

The progression records and selected checkpoint hashes are summarized in
`../docs/analysis/stage1/stage1_results.json` and checked by
`../docs/analysis/provenance/progression_integrity_audit.json`. The audit
passes for all six final records per benchmark. The two operational
`study_status.json` files are stale and still say final evaluation is pending;
use the audit and result records for the completed campaign status.

The manuscript figures are the static PDFs in `figures/`. The two progression
figures were generated from the audited seed-level results and the LIBERO
scene-by-goal crossover records. Their temporary builder is in the ignored
`../asset/tmp_analysis/manuscript_update_v1/` directory. The older
`plot_figures.py` rebuilds the historical figures from its external results
bundle; it does not rebuild the new progression figures.

The revised story uses four vector figures ending in `_review.pdf`: ALOHA
progress, LIBERO goal comparison, paired instruction outcomes, and checkpoint
progression. Their current builder is
`../asset/tmp_analysis/visual_review_v1/paper_figures.py`; it reads saved
analysis tables and performs no training or rollouts. The complete ten-task,
per-seed final-test table appears in the manuscript and as counts in
`../docs/analysis/stage2/libero_all_task_success_by_seed.md`. The builder and
durable inputs are ignored local assets and must be available to regenerate
the figures; the PDFs used by `main.tex` are in the manuscript project.

The results show a consistent observed GR00T advantage on ALOHA across the
three trained seed pairs. On aggregate LIBERO-Goal success, the three-seed
means are equal and seed directions are mixed. The familiar LIBERO scene-by-goal
analysis exposes a repeatable task-specific crossover: Qwen leads on wine to
rack, while GR00T leads or ties on cheese to bowl. These findings compare two
released endpoints from one lineage; they do not isolate one pretraining stage
or establish that differences between benchmarks are caused by language,
camera count, or embodiment.

No training or simulator rollouts were run while preparing this manuscript
update. Compile `main.tex` with a standard LaTeX installation before
submission; the draft checks its six-page limit during compilation.
