# frozen-vlm-control: project guide

The project name is **frozen-vlm-control**. Recorded paths containing `action-omni` refer to historical or existing checkout locations. The manuscript retains its fixed paper title.

Run commands from the repository root. This is a research repository with flat
Python packages: source stays in `models/`, `data/`, `envs/`, and `configs/`.
The existing import paths are also used in experiment configurations and saved
checkpoints, so no `src/` migration or additional packaging layer is needed.

## Where things belong

| Location | Contents |
|---|---|
| `models/` | Runtime models, action heads, and the shared frozen-backbone loader. |
| `data/` | Dataset readers and embedding-cache readers/writers. |
| `envs/` | Simulator wrappers and policy agents. |
| `configs/` | Shared configuration and registered benchmark experiments. |
| `experiment_protocol.py` | Shared split, seed, provenance, and checkpoint checks. |
| `scripts/` | Precompute, train, evaluate, and protocol-preparation commands. |
| `scripts/analysis/`, `scripts/eval/`, `scripts/validation/` | Diagnostics, evaluation studies, and harness checks. |
| `scripts/data/`, `scripts/plots/`, `scripts/media/`, `scripts/report/` | Data preparation and output generators. |
| `scripts/archive/` | Superseded exploratory scripts, retained as historical source. |
| `tests/` | Synthetic protocol, statistics, import, and command regression checks. |
| `docs/` | Research notes, camera-ready workflow, validation, and review tracking. |
| `manuscript/` | LaTeX source, final PDF, curated figure data, and plots. |
| External report output | Generated reports, kept outside the source repository. |
| `asset/` | Local datasets, pretrained weights, feature caches, protocol manifests, run outputs, and durable evaluation/recovery state. See the [asset layout](ASSET_LAYOUT.md). |
| `asset/cache/`, `asset/data/`, `asset/models/` | Reusable frozen features, local datasets, and pretrained backbone snapshots. |
| `asset/protocols/`, `asset/runs/`, `asset/durable/` | Immutable episode-split manifests, experiment outputs, and durable evaluation/recovery records. |
| External evidence archive | Numerical diagnostics from the earlier camera-ready check. |
| `baselines/` | Separate baseline reproduction instructions and scripts. |
| `third_party/` | Ignored local upstream checkouts. |

## Current experiment

The completed campaign compares frozen layer-16 GR00T and Qwen features using three independently trained heads per backbone per benchmark. Final checkpoints are ALOHA epoch 250 and LIBERO epoch 120. See the [project overview](../README.md), [objective](OBJECTIVE.md), [results](RESULTS.md), and [manuscript](../manuscript/main.tex).

The integrity audit and final records establish completion; operational study-status files are stale. Historical RunPod notices in the continuity guide do not describe current process state. Local planning files are ignored and are not prerequisites for understanding the public experiment.

## Setup and checks

Use the existing `vla` environment for model/data tools and synthetic checks.
The LIBERO simulator uses the separate `vla_libero` environment; its setup is
documented in the [baseline guide](../baselines/smolvla_libero/README.md).
`requirements.txt` lists the core dependencies, including HDF5 storage, paired
statistics, and plotting. Simulator stacks remain separate dependencies.

```powershell
python -m unittest discover -s tests -v
python scripts/train.py --help
python scripts/precompute.py --help
python scripts/evaluate.py --help
```

The tests use temporary synthetic data and mocked backbone loading. They do not
train policies, encode the real dataset, or run performance rollouts. The
[camera-ready workflow](CAMERA_READY.md) contains actual preparation commands
and explicitly deferred experiment commands. Its bounded CUDA observation uses
`scripts/validation/observe_training.py` and actual optimizer updates on repeated
real-data features, separately from the synthetic suite. See the [validation record](camera_ready_validation.md)
for remaining real-model and simulator checks.

## Documents and compatibility

| Previous path | Current path |
|---|---|
| `OBJECTIVE.md` | [docs/OBJECTIVE.md](OBJECTIVE.md) |
| `RESULTS.md` | [docs/RESULTS.md](RESULTS.md) |
| `REFERENCES.md` | [docs/REFERENCES.md](REFERENCES.md) |
| `PAPER_OUTLINE.md` | [manuscript/outline.md](../manuscript/outline.md) |
| `scripts/CAMERA_READY.md` | [docs/CAMERA_READY.md](CAMERA_READY.md) |
| `report/camera_ready_validation.md` | [docs/camera_ready_validation.md](camera_ready_validation.md) |
| `report/reviewer_response_matrix.md` | [docs/reviewer_response_matrix.md](reviewer_response_matrix.md) |

The objective and results describe the current progression campaign. Older component-study notes are preserved under `docs/archive/`. The reviewer matrix and camera-ready workflow are historical records; the manuscript and integrity audit define the current evidence.

The root `train.py`, `precompute_embeddings.py`, and `inference.py` now forward
to their `scripts/` counterparts rather than maintaining separate pipelines.
Legacy `--exp 2` maps to `--dataset pusht --exp exp02a`; other integer
experiments were already unregistered. Remaining options go to the canonical
parser, so unsupported options fail instead of silently being ignored. The
trainer's resume behavior and evaluator's metadata checks are now shared too.
Existing public `train`/`inference` imports and `config_loader.get_config` remain.

Both live policies and latent studies use `models/backbones.py`. The old
`scripts/analysis/latent_compare/backbones.py` is an import-only compatibility
shim for the existing study commands. Encoder provenance now hashes the canonical
loader path; caches/checkpoints with older encoder fingerprints are not silently
accepted by the opt-in protocol. Historical artifacts have not been rewritten.
