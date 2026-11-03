"""Keep two durable resume points and policy-only midpoints for a local study."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.checkpoint_io import atomic_copy_checkpoint, prune_recovery_revisions
from scripts.run_log import replace_with_retry


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_midpoint(run: Path, durable_run: Path, checkpoint: dict) -> Path:
    epoch = int(checkpoint["epoch"])
    path = run / "checkpoints" / f"midpoint_epoch_{epoch:06d}.pt"
    durable_path = durable_run / "checkpoints" / path.name
    if not path.exists():
        omit = {"optim", "scheduler", "rng_state", "train_generator", "val_generator",
                "best_val_loss", "no_improve", "next_recovery_step",
                "success_stopping_state", "recovery_revision"}
        policy = {key: value for key, value in checkpoint.items() if key not in omit}
        temporary = path.with_name(path.name + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy, temporary)
        replace_with_retry(temporary, path)
    atomic_copy_checkpoint(path, durable_path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("aloha", "libero"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--durable-root", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--minimum-epochs", type=int, required=True)
    parser.add_argument("--evaluation-every-epochs", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.evaluation_every_epochs < 2 or args.poll_seconds < 1:
        parser.error("evaluation cadence must be at least 2 and poll interval at least 1 second")

    run_root = args.run_root.resolve()
    durable_study = args.durable_root.resolve() / "success_studies" / args.study_id
    study_status = durable_study / "study_status.json"
    log_path = args.durable_root.resolve() / "checkpoint_retention.log"
    seen_last = {}

    policy_path = durable_study / "checkpoint_retention_policy.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy = {
        "schema": 1,
        "resume_checkpoints": "last.pt and previous.pt locally; newest two verified recovery revisions durably",
        "focus_checkpoints": "immutable selection_epoch checkpoints used for rollout evaluation and selected.pt",
        "midpoint_checkpoints": "policy-only snapshot halfway between successive evaluation epochs",
        "evaluation_every_epochs": args.evaluation_every_epochs,
        "midpoint_offset_epochs": args.evaluation_every_epochs // 2,
        "updated_utc": stamp(),
    }
    temporary_policy = policy_path.with_name(policy_path.name + ".tmp")
    temporary_policy.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    replace_with_retry(temporary_policy, policy_path)

    def record(event: str, **fields) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, "study_id": args.study_id,
                                    "dataset": args.dataset, "time_utc": stamp(),
                                    **fields}, sort_keys=True) + "\n")

    while True:
        # New runs put the dataset in the study-root name, while older active
        # runs keep the dataset as a directory below the root.
        progress_paths = set(run_root.glob("*/seed_*/progress.json"))
        progress_paths.update((run_root / args.dataset).glob("*/seed_*/progress.json"))
        for progress_path in sorted(progress_paths):
            run = progress_path.parent
            relative = run.relative_to(run_root)
            durable_run = durable_study / args.dataset / relative
            recovery_dir = durable_run / "checkpoints"
            removed = prune_recovery_revisions(recovery_dir, keep=2)
            if removed:
                record("old_recovery_pruned", policy=str(relative),
                       removed=[path.name for path in removed], retained=2)

            last_path = run / "checkpoints" / "last.pt"
            if not last_path.is_file():
                continue
            try:
                stat = last_path.stat()
                signature = (stat.st_size, stat.st_mtime_ns)
                if seen_last.get(last_path) == signature:
                    continue
                seen_last[last_path] = signature
                checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
                epoch = int(checkpoint.get("epoch", -1))
                offset = (epoch - args.minimum_epochs) % args.evaluation_every_epochs
                if (epoch >= args.minimum_epochs and offset == args.evaluation_every_epochs // 2
                        and not (run / "checkpoints" / f"midpoint_epoch_{epoch:06d}.pt").exists()):
                    saved = save_midpoint(run, durable_run, checkpoint)
                    record("policy_midpoint_saved", policy=str(relative), epoch=epoch,
                           checkpoint=str(saved.relative_to(run_root)),
                           bytes=saved.stat().st_size)
            except (OSError, RuntimeError, ValueError, KeyError, EOFError) as exc:
                record("checkpoint_inspection_retry", policy=str(relative),
                       error=f"{type(exc).__name__}: {exc}")

        try:
            state = json.loads(study_status.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        if state.get("status") in {"training_complete_evaluation_pending", "failed", "interrupted"}:
            record("retention_worker_finished", status=state.get("status"))
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
