#!/usr/bin/env python3
"""Promote the validated fixed-EV admission contract into a policy bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


POLICY_ID = "side_archetype_hier_ev_fixed70_trim10_21d_v1"
POLICY_NAME = "s52_v9_tail95_mlp_hierev_ev70_trim10_21d_evaware_geometry_v3"
POLICY_FILE = "threshold_basis_policy_sidearch_ev70_trim10_21d.json"
FAMILY = "side_archetype_expected_ev_recent_correction"
WINDOW_DAYS = 21
FIXED_TARGET_NET_EV = 0.007
TRIM_FRACTION = 0.10
MAX_NEW_ENTRIES = 2
MAX_CONCURRENT = 8


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backup(path: Path) -> str | None:
    if not path.is_file():
        return None
    backup = path.with_suffix(path.suffix + ".pre_ev70_trim10_21d")
    if not backup.exists():
        shutil.copy2(path, backup)
    return str(backup)


def _patch_policy_contract(payload: dict[str, Any], policy_path: str) -> None:
    payload.update(
        {
            "policy_name": POLICY_NAME,
            "threshold_basis_policy_enabled": True,
            "threshold_basis_policy_id": POLICY_ID,
            "threshold_basis_policy_path": policy_path,
            "source_threshold_basis_policy": policy_path,
            "threshold_basis_family": FAMILY,
            "threshold_basis_window_days": WINDOW_DAYS,
            "threshold_basis_selection_mode": "fixed_corrected_ev_threshold",
            "threshold_basis_fixed_target_net_ev": FIXED_TARGET_NET_EV,
            "threshold_basis_robust_daily_residual_trim_fraction": TRIM_FRACTION,
            "threshold_basis_hr_rank50": False,
        }
    )
    concurrency = payload.get("concurrency")
    if isinstance(concurrency, dict):
        concurrency["max_new_entries_per_bar"] = MAX_NEW_ENTRIES
        concurrency["max_concurrent_positions"] = MAX_CONCURRENT
    # Some promotion manifests expose the effective limits at the top level
    # rather than under ``concurrency``. Keep those audit fields synchronized
    # with the executable portfolio config.
    if "max_new_entries_per_bar" in payload:
        payload["max_new_entries_per_bar"] = MAX_NEW_ENTRIES
    if "max_concurrent_positions" in payload:
        payload["max_concurrent_positions"] = MAX_CONCURRENT
    for key in ("selection", "portfolio_policy"):
        child = payload.get(key)
        if isinstance(child, dict):
            _patch_policy_contract(child, policy_path)
    strategies = payload.get("strategies")
    if isinstance(strategies, list):
        for row in strategies:
            if isinstance(row, dict):
                _patch_policy_contract(row, policy_path)


def promote(artifact_root: Path, matrix_dir: Path) -> dict[str, Any]:
    policy_dir = artifact_root / "policy_params"
    source_policy_path = policy_dir / "threshold_basis_policy_sidearch_ev28d.json"
    if not source_policy_path.is_file():
        raise FileNotFoundError(source_policy_path)
    matrix_metrics_path = matrix_dir / "portfolio_matrix_metrics.csv"
    matrix_manifest_path = matrix_dir / "manifest.json"
    if not matrix_metrics_path.is_file() or not matrix_manifest_path.is_file():
        raise FileNotFoundError("validated matrix metrics/manifest are required")

    import pandas as pd

    import pyarrow.parquet as pq

    matrix_manifest = _read_json(matrix_manifest_path)
    matrix_source = Path(str(matrix_manifest.get("source") or ""))
    if not matrix_source.is_absolute():
        matrix_source = Path.cwd() / matrix_source
    if not matrix_source.is_file():
        raise FileNotFoundError(matrix_source)
    matrix_metrics = pd.read_csv(matrix_metrics_path)
    winner = matrix_metrics.loc[
        matrix_metrics["arm"].eq("ev70bps_trim10_period21d")
    ]
    if len(winner) != 1:
        raise ValueError("selected matrix arm is missing or duplicated")
    winner_row = winner.iloc[0].to_dict()

    source_policy = _read_json(source_policy_path)
    available_columns = set(pq.ParquetFile(matrix_source).schema.names)
    reference_columns = [
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "rank_mlp_direct",
        "expected_net_ev_after_1pct_mlp_direct",
        "ev_after_1pct",
    ]
    # These fields do not alter admission; they make the same causal reference
    # usable for close-email path-quality diagnostics.  Keep the contract
    # forward-compatible with richer policy candidate ledgers.
    for column in (
        "clean_exec",
        "dirty_positive",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "bad_mae_1r",
        "timeout",
        "timed_out",
        "full_stop_loss",
        "stop_loss",
        "stop_hit",
        "gmm_cluster_id",
        "aegmm_cluster",
        "side_aegmm_cluster",
        "gmm_posterior_max",
    ):
        if column in available_columns:
            reference_columns.append(column)
    reference = pd.read_parquet(matrix_source, columns=reference_columns).rename(
        columns={
            "__ts__": "timestamp",
            "__symbol__": "symbol",
            "archetype_policy_key": "policy_archetype",
            "expected_net_ev_after_1pct_mlp_direct": "mapped_expected_ev",
        }
    )
    reference["timestamp"] = pd.to_datetime(
        reference["timestamp"], utc=True, errors="coerce"
    )
    reference["side_name"] = reference["side_name"].astype(str).str.lower()
    reference["policy_archetype"] = (
        reference["policy_archetype"].fillna("missing").astype(str)
    )
    for side in ("long", "short"):
        prefix = f"{side}__"
        mask = reference["side_name"].eq(side) & reference[
            "policy_archetype"
        ].str.startswith(prefix, na=False)
        reference.loc[mask, "policy_archetype"] = reference.loc[
            mask, "policy_archetype"
        ].str[len(prefix) :]
    reference["outcome_resolved_at"] = reference["timestamp"] + pd.Timedelta(
        hours=int(source_policy.get("outcome_horizon_hours") or 12)
    )
    reference = reference.sort_values(
        ["timestamp", "symbol", "side_name"], kind="stable"
    ).reset_index(drop=True)
    reference_path = policy_dir / "threshold_basis_reference_sidearch_ev70_trim10_21d.parquet"
    reference.to_parquet(reference_path, index=False, compression="zstd")
    policy = dict(source_policy)
    policy.update(
        {
            "schema_version": "threshold_basis_policy_v4",
            "policy_id": POLICY_ID,
            "policy_name": POLICY_NAME,
            "family": FAMILY,
            "selection_mode": "fixed_corrected_ev_threshold",
            "fixed_target_net_ev": FIXED_TARGET_NET_EV,
            "window_days": WINDOW_DAYS,
            "recalibration_frequency": "1d_at_00_utc",
            "robust_daily_residual_trim_fraction": TRIM_FRACTION,
            "robust_daily_residual_normalization": "median_iqr",
            "email_archetype_baseline_window_days": 28,
            "email_archetype_baseline_min_rows": 40,
            "email_archetype_baseline_contract": (
                "fixed_28d_resolved_outcomes; side_x_archetype with "
                "side/global fallback; same median/IQR daily residual trim as admission"
            ),
            "ev_rank_blend_weight": 1.0,
            "reference_candidates_path": reference_path.name,
            "reference_columns": list(reference.columns),
            "reference_rows": int(len(reference)),
            "reference_timestamp_min": reference["timestamp"].min().isoformat(),
            "reference_timestamp_max": reference["timestamp"].max().isoformat(),
            "formula": (
                "corrected_expected_ev = side_archetype_mapped_expected_ev + "
                "causal_21d_robust_trimmed_side_archetype_recent_ev_residual; "
                "admit when corrected_expected_ev >= 0.007"
            ),
            "causal_contract": (
                "At t, only outcomes resolved before day(t) are eligible. "
                "Residual outcome days use [day(t)-21d,day(t)); daily residual "
                "means receive symmetric 10% median/IQR trimming."
            ),
            "portfolio_validation": {
                "source": str(matrix_dir),
                "arm": "ev70bps_trim10_period21d",
                "max_new_entries_per_bar": MAX_NEW_ENTRIES,
                "max_concurrent_positions": MAX_CONCURRENT,
                "metrics": winner_row,
            },
        }
    )
    policy_path = policy_dir / POLICY_FILE
    _write_json(policy_path, policy)
    policy_pointer = str(policy_path)

    patched_paths = [
        policy_dir / "optimized_portfolio_policy_config.json",
        policy_dir / "promoted_policy_manifest.json",
        policy_dir / "global_inference_policy_contract.json",
        artifact_root / "simple_policy_optimiser/deployment/best_policy_params.json",
        artifact_root
        / "simple_policy_optimiser/deployment/best_policy_params_perps.json",
    ]
    backups: list[str] = []
    for path in patched_paths:
        if not path.is_file():
            continue
        backup = _backup(path)
        if backup:
            backups.append(backup)
        payload = _read_json(path)
        _patch_policy_contract(payload, policy_pointer)
        payload["side_archetype_expected_ev_policy"] = {
            "policy_id": POLICY_ID,
            "formula": policy["formula"],
            "mapped_expected_ev_is_side_archetype_specific": True,
            "selection_mode": "fixed_corrected_ev_threshold",
            "fixed_target_net_ev": FIXED_TARGET_NET_EV,
            "window_days": WINDOW_DAYS,
            "robust_daily_residual_trim_fraction": TRIM_FRACTION,
        }
        _write_json(path, payload)

    optimized_policy_path = policy_dir / "optimized_portfolio_policy_config.json"
    deployment_path = (
        artifact_root / "simple_policy_optimiser/deployment/best_policy_params.json"
    )
    deployment_payload = _read_json(deployment_path)
    for parity_path in (
        policy_dir / "training_live_parity_contract.json",
        artifact_root / "simple_policy_optimiser/training_live_parity_contract.json",
    ):
        if not parity_path.is_file():
            continue
        _backup(parity_path)
        parity = _read_json(parity_path)
        hashes = parity.get("artifact_hashes")
        if not isinstance(hashes, dict):
            raise ValueError(f"parity contract has no artifact_hashes: {parity_path}")
        for key, artifact_path in (
            ("threshold_basis_policy", policy_path),
            ("optimized_portfolio_policy", optimized_policy_path),
            ("simple_policy_deployment", deployment_path),
        ):
            record = hashes.get(key)
            if not isinstance(record, dict):
                raise ValueError(f"parity contract is missing {key}: {parity_path}")
            record.update(
                {
                    "artifact_type": "file",
                    "exists": True,
                    "path": str(artifact_path),
                    "sha256": _sha256(artifact_path),
                }
            )
        parity["deployment_policy"] = deployment_payload
        parity["policy_promotion"] = {
            "policy_id": POLICY_ID,
            "policy_name": POLICY_NAME,
            "policy_path": policy_pointer,
            "matrix_arm": "ev70bps_trim10_period21d",
            "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(parity_path, parity)

    side_manifest_path = policy_dir / "side_archetype_expected_ev_policy_manifest.json"
    side_manifest = _read_json(side_manifest_path)
    _backup(side_manifest_path)
    side_manifest.update(
        {
            "policy_id": POLICY_ID,
            "policy_name": POLICY_NAME,
            "policy_sha256": _sha256(policy_path),
            "status": "promoted_default_after_portfolio_matrix_validation",
            "selected_matrix_arm": "ev70bps_trim10_period21d",
            "selected_matrix_metrics": winner_row,
            "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_json(side_manifest_path, side_manifest)

    # Keep the promoted-policy file audit in sync as well.  This manifest is
    # consumed by the operational review tooling even though the live hash
    # contract above is the authoritative runtime check.
    promoted_manifest_path = policy_dir / "promoted_policy_manifest.json"
    if promoted_manifest_path.is_file():
        _backup(promoted_manifest_path)
        promoted_manifest = _read_json(promoted_manifest_path)
        file_hashes = promoted_manifest.get("file_sha256")
        if isinstance(file_hashes, dict):
            file_hashes[
                policy_path.relative_to(artifact_root).as_posix()
            ] = _sha256(policy_path)
            file_hashes[
                reference_path.relative_to(artifact_root).as_posix()
            ] = _sha256(reference_path)
        _write_json(promoted_manifest_path, promoted_manifest)
    return {
        "policy_id": POLICY_ID,
        "policy_name": POLICY_NAME,
        "policy_path": str(policy_path),
        "policy_sha256": _sha256(policy_path),
        "reference_path": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "matrix_arm": "ev70bps_trim10_period21d",
        "matrix_metrics": winner_row,
        "patched_paths": [str(path) for path in patched_paths if path.is_file()],
        "backups": backups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    args = parser.parse_args()
    result = promote(args.artifact_root.resolve(), args.matrix_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
