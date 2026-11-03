"""Audit final result, checkpoint, protocol, and paired-cache provenance.

This is a read-only audit of the saved ALOHA and LIBERO progression campaigns.
It rehashes selected checkpoint files, compares their metadata with final
records and study status, verifies cache-manifest links to locked jobs, and
checks that paired backbone caches contain the same training examples.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDIES = {
    "aloha": ("aloha_progression_250_v1", "415a08a2c6f2", 250),
    "libero": ("libero_progression_120_v1", "c6854e3ff020", 120),
}
BACKBONES = {"exp05": "groot", "exp06": "qwen3vl"}
PAIR_FIELDS = ("episode_ids", "states", "actions", "validation_mask", "img_masks")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrays_equal_chunked(left, right, rows: int = 4096) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.ndim == 0:
        return bool(np.array_equal(left[()], right[()]))
    for start in range(0, left.shape[0], rows):
        stop = min(start + rows, left.shape[0])
        if not np.array_equal(left[start:stop], right[start:stop]):
            return False
    return True


def main() -> int:
    audit = {"scope": "Read-only integrity audit; no training or simulator rollouts.",
             "studies": {}, "all_checks_pass": True}
    for dataset, (study, plan_prefix, epoch) in STUDIES.items():
        locked_path = ROOT / "asset/durable" / dataset / "success_studies" / study / "locked_success_study.json"
        locked = json.loads(locked_path.read_text(encoding="utf-8"))
        plan_id = locked["plan_id"]
        plan = locked["plan"]
        status_path = ROOT / "asset/runs" / study / "study_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        completed = {r["policy_id"]: r for r in status.get("completed_policies", [])}
        jobs = {(j["experiment"], int(j["training_seed"])): j for j in locked["jobs"]}
        policy_records = []
        cache_manifest_records = {}

        for exp, backbone in BACKBONES.items():
            manifest_path = ROOT / "asset/cache/precomputed" / dataset / exp / "vlm_embeddings.h5.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cache_path = ROOT / "asset/cache/precomputed" / dataset / exp / "vlm_embeddings.h5"
            with h5py.File(cache_path, "r") as cache:
                attrs = {k: (v.item() if isinstance(v, np.generic) else v)
                         for k, v in cache.attrs.items() if k != "source_assets"}
                attrs_config = json.loads(attrs["cache_config"])
                cache_id_match = attrs.get("cache_id") == manifest.get("cache_id")
                protocol_match = attrs.get("protocol_id") == manifest.get("protocol_id")
                cache_manifest_records[exp] = {
                    "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
                    "cache_path": cache_path.relative_to(ROOT).as_posix(),
                    "manifest_cache_id": manifest.get("cache_id"),
                    "h5_cache_id": attrs.get("cache_id"),
                    "protocol_id": manifest.get("protocol_id"),
                    "backbone": manifest.get("backbone"),
                    "manifest_size_bytes": manifest.get("size_bytes"),
                    "actual_size_bytes": cache_path.stat().st_size,
                    "manifest_size_matches": manifest.get("size_bytes") == cache_path.stat().st_size,
                    "h5_cache_id_matches_manifest": cache_id_match,
                    "h5_protocol_matches_manifest": protocol_match,
                    "encoder_code_id": manifest.get("encoder_code_id"),
                    "source_asset_count": len(manifest.get("source_assets", {})),
                    "h5_config": {k: attrs_config.get(k) for k in
                                  ("vlm_backbone", "vlm_extract_layers", "vlm_hidden_size", "n_views", "action_dim", "state_dim")},
                }
                cache_manifest_records[exp]["checks_pass"] = all((
                    manifest.get("dataset") == dataset,
                    manifest.get("backbone") == backbone,
                    cache_manifest_records[exp]["manifest_size_matches"],
                    cache_id_match, protocol_match,
                    manifest.get("protocol_id") == jobs[(exp, 3)]["protocol_id"],
                    manifest.get("cache_id") == jobs[(exp, 3)]["cache_id"],
                    attrs_config.get("vlm_backbone") == backbone,
                    attrs_config.get("vlm_extract_layers") == [16],
                ))

            for seed in (3, 4, 5):
                job = jobs[(exp, seed)]
                result_path = (ROOT / "asset/durable" / dataset / "evaluation_results" /
                               study / plan_prefix / dataset / exp / f"seed_{seed:03d}" / "final_test.json")
                record = json.loads(result_path.read_text(encoding="utf-8"))
                ckpt_path = (ROOT / "asset/durable" / dataset / "evaluation_checkpoints" /
                             study / plan_prefix / dataset / exp / f"seed_{seed:03d}" / "final_test.pt")
                actual_ckpt_sha = sha256(ckpt_path)
                status_entry = completed.get(f"{dataset}/{exp}/seed_{seed:03d}")
                if status_entry is None:
                    status_entry = next((r for r in status.get("completed_policies", [])
                                         if r.get("training_seed") == seed and r.get("experiment") == exp), None)
                allowed = {"output_dir", "embeddings_cache", "model_path", "device", "embed_batch_size", "vlm_backbone"}
                record_config = {k: v for k, v in record.get("configuration", {}).items() if k not in allowed}
                paired_path = (ROOT / "asset/durable" / dataset / "evaluation_results" /
                               study / plan_prefix / dataset /
                               ("exp06" if exp == "exp05" else "exp05") / f"seed_{seed:03d}" / "final_test.json")
                paired = json.loads(paired_path.read_text(encoding="utf-8"))
                paired_config = {k: v for k, v in paired.get("configuration", {}).items() if k not in allowed}
                checks = {
                    "plan_id_matches": record.get("plan_id") == plan_id,
                    "protocol_id_matches_locked_job": record.get("protocol_id") == job["protocol_id"],
                    "cache_id_matches_locked_job": record.get("cache_id") == job["cache_id"],
                    "run_signature_matches_locked_job": record.get("run_signature") == job["run_signature"],
                    "result_identity_matches": (record.get("dataset") == dataset and record.get("backbone") == backbone
                                                 and int(record.get("training_seed")) == seed),
                    "final_epoch_matches": record.get("checkpoint_epoch") == epoch,
                    "result_checkpoint_hash_matches_actual_file": record.get("checkpoint_sha256") == actual_ckpt_sha,
                    "selected_checkpoint_hash_matches_actual_file": record.get("selected_checkpoint_sha256") == actual_ckpt_sha,
                    "selected_checkpoint_hash_matches_status": bool(status_entry) and status_entry.get("selected_checkpoint_sha256") == actual_ckpt_sha,
                    "selected_epoch_matches_status": bool(status_entry) and status_entry.get("selected_epoch") == epoch,
                    "head_config_matches_paired_backbone": record_config == paired_config,
                }
                if dataset == "aloha":
                    n_episode = len(record.get("episodes", []))
                    checks["episode_count_is_200"] = n_episode == 200
                else:
                    n_episode = len(record.get("results", []))
                    checks["episode_count_is_400"] = n_episode == 400
                policy_records.append({
                    "experiment": exp, "backbone": backbone, "training_seed": seed,
                    "result_path": result_path.relative_to(ROOT).as_posix(),
                    "checkpoint_path": ckpt_path.relative_to(ROOT).as_posix(),
                    "checkpoint_sha256_recomputed": actual_ckpt_sha,
                    "result_sha256": sha256(result_path),
                    "episode_count": n_episode,
                    "checks": checks,
                    "checks_pass": all(checks.values()),
                })

        # Check paired caches contain the same row ordering and labels/actions.
        pair_checks = {}
        left_path = ROOT / "asset/cache/precomputed" / dataset / "exp05" / "vlm_embeddings.h5"
        right_path = ROOT / "asset/cache/precomputed" / dataset / "exp06" / "vlm_embeddings.h5"
        with h5py.File(left_path, "r") as left, h5py.File(right_path, "r") as right:
            pair_checks["same_sample_count"] = left["episode_ids"].shape == right["episode_ids"].shape
            pair_checks["same_sequence_shape"] = left["embeddings"].shape == right["embeddings"].shape
            for name in PAIR_FIELDS:
                pair_checks[f"{name}_identical"] = arrays_equal_chunked(left[name], right[name])
            left_ids, right_ids = left["episode_ids"][:], right["episode_ids"][:]
            pair_checks["episode_ids_same_order"] = bool(np.array_equal(left_ids, right_ids))
            pair_checks["validation_mask_same_count"] = int(left["validation_mask"][:].sum()) == int(right["validation_mask"][:].sum())

        pending = status.get("pending_final_tests", [])
        status_stale = (status.get("status") == "training_complete_evaluation_pending"
                        and len(policy_records) == 6 and all(r["checks_pass"] for r in policy_records))
        study_result = {
            "study_id": study, "locked_plan_id": plan_id,
            "locked_protocol_ids": sorted({j["protocol_id"] for j in jobs.values()}),
            "locked_checkpoint_rule": plan.get("checkpoint_rule"),
            "study_status_file": status_path.relative_to(ROOT).as_posix(),
            "status_value": status.get("status"),
            "status_completed_policy_count": status.get("completed_policy_count"),
            "status_pending_final_test_records": len(pending),
            "status_is_stale_vs_verified_files": status_stale,
            "final_records": policy_records,
            "cache_manifests": cache_manifest_records,
            "paired_cache_input_checks": pair_checks,
            "checks_pass": (all(r["checks_pass"] for r in policy_records)
                            and all(r["checks_pass"] for r in cache_manifest_records.values())
                            and all(pair_checks.values())),
        }
        audit["studies"][dataset] = study_result
        audit["all_checks_pass"] &= study_result["checks_pass"]

    output = ROOT / "docs/analysis/provenance"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "progression_integrity_audit.json"
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}; all checks pass={audit['all_checks_pass']}")
    for dataset, result in audit["studies"].items():
        print(f"{dataset}: 6/6 final records, actual checkpoint SHA-256s recomputed; "
              f"status={result['status_value']}, stale={result['status_is_stale_vs_verified_files']}; "
              f"paired inputs identical={all(result['paired_cache_input_checks'].values())}")
    return 0 if audit["all_checks_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
