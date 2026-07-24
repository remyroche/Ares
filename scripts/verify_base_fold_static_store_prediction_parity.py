#!/usr/bin/env python3
"""Verify saved base-fold predictions from the canonical static feature store."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd

from scripts.run_label_quality_proxy_diagnostics import _load_feature_store_columns
from scripts.verify_base_frozen_aegmm_static_store_parity import (
    KEY_COLUMNS,
    _normalize_side,
    _read_sidecar_sample,
    _sidecar_output_features,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _sample_oos_rows(path: Path, samples_per_fold: int) -> pd.DataFrame:
    escaped = str(path.resolve()).replace("'", "''")
    take = max(1, int(samples_per_fold))
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone='UTC'")
        rows = connection.execute(
            f"""
            WITH ordered AS (
                SELECT
                    CAST(__ts__ AS TIMESTAMPTZ) AS __ts__,
                    CAST(__symbol__ AS VARCHAR) AS __symbol__,
                    side,
                    CAST(score AS DOUBLE) AS expected_score,
                    CAST(oos_fold AS VARCHAR) AS oos_fold,
                    row_number() OVER (
                        PARTITION BY oos_fold ORDER BY __ts__, __symbol__, side
                    ) AS row_number_in_fold,
                    count(*) OVER (PARTITION BY oos_fold) AS rows_in_fold
                FROM read_parquet('{escaped}')
            )
            SELECT * EXCLUDE (row_number_in_fold, rows_in_fold)
            FROM ordered
            WHERE rows_in_fold <= {take}
               OR floor((row_number_in_fold - 1) * {take} / rows_in_fold)
                  < floor(row_number_in_fold * {take} / rows_in_fold)
            ORDER BY __ts__, __symbol__, side
            """
        ).fetchdf()
    finally:
        connection.close()
    rows["__ts__"] = pd.to_datetime(rows["__ts__"], utc=True, errors="coerce")
    rows["__symbol__"] = rows["__symbol__"].astype(str)
    rows["side"] = _normalize_side(rows["side"])
    if rows[list(KEY_COLUMNS)].isna().any().any() or rows.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("sampled OOS identities are invalid or duplicated")
    return rows.reset_index(drop=True)


def _load_columns(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = [str(value) for value in payload.get("feature_names", [])]
    if not names:
        raise ValueError(f"no feature names in {path}")
    return names


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _imputation_contract_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {
            "schema": payload.get("schema"),
            "strategy": payload.get("strategy"),
            "feature_names": payload.get("feature_names"),
            "fill_values": payload.get("fill_values"),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _feature_order_hash(columns: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(columns, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _load_training_imputation(fold_dir: Path, columns: list[str]) -> np.ndarray:
    """Load the mandatory, ordered train-only fill values for one model."""

    manifest_path = fold_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"fold {fold_dir.name} is missing model manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configured_path = str(manifest.get("imputation_path") or "")
    local_path = fold_dir / "imputation.json"
    imputation_path = (
        local_path
        if local_path.is_file()
        else Path(configured_path) if configured_path else local_path
    )
    if not imputation_path.is_file():
        raise ValueError(
            f"fold {fold_dir.name} is missing required train-median imputation artifact"
        )
    expected_sha256 = str(manifest.get("imputation_sha256") or "")
    actual_sha256 = _sha256_file(imputation_path)
    if not expected_sha256 or expected_sha256 != actual_sha256:
        raise ValueError(
            f"fold {fold_dir.name} imputation artifact hash does not match manifest"
        )
    payload = json.loads(imputation_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "s60_base_train_median_imputation_v1":
        raise ValueError(f"fold {fold_dir.name} has unsupported imputation schema")
    names = [str(value) for value in payload.get("feature_names", [])]
    if names != columns:
        raise ValueError(
            f"fold {fold_dir.name} imputation feature order does not match columns.json"
        )
    if str(payload.get("feature_order_hash") or "") != _feature_order_hash(columns):
        raise ValueError(f"fold {fold_dir.name} imputation feature-order hash is invalid")
    fill_values = np.asarray(payload.get("fill_values", []), dtype=np.float32)
    if fill_values.shape != (len(columns),) or not np.isfinite(fill_values).all():
        raise ValueError(f"fold {fold_dir.name} imputation fill values are invalid")
    expected_contract_hash = str(manifest.get("imputation_contract_hash") or "")
    actual_contract_hash = _imputation_contract_hash(payload)
    if not expected_contract_hash or expected_contract_hash != actual_contract_hash:
        raise ValueError(
            f"fold {fold_dir.name} imputation contract hash does not match manifest"
        )
    return fill_values


def _required_input_complete(values: np.ndarray) -> np.ndarray:
    """Return rows eligible for scoring under the required-input contract."""

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("model feature matrix must be two-dimensional")
    return np.isfinite(matrix).all(axis=1)


def _predict_serialized_lgbm(model: Any, values: np.ndarray) -> np.ndarray:
    """Score through the native booster to avoid sklearn-version coupling."""
    booster = getattr(model, "booster_", None)
    if booster is None:
        booster = getattr(model, "_Booster", None)
    if booster is None:
        return np.asarray(model.predict(values), dtype=np.float64)
    best_iteration = getattr(model, "best_iteration_", None)
    num_iteration = int(best_iteration) if best_iteration not in (None, 0, -1) else None
    return np.asarray(
        booster.predict(values, num_iteration=num_iteration), dtype=np.float64
    )


def verify_base_fold_prediction_parity(
    *,
    report_dir: Path,
    feature_store: Path,
    sidecar: Path,
    samples_per_fold: int = 16,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    report_dir = Path(report_dir)
    oos_path = report_dir / "best_oos_scored_ledger.parquet"
    models_dir = report_dir / "models"
    sampled = _sample_oos_rows(oos_path, samples_per_fold)
    sidecar_outputs, sidecar_contract = _sidecar_output_features(sidecar)
    sidecar_output_set = set(sidecar_outputs)

    fold_reports: list[dict[str, Any]] = []
    row_reports: list[pd.DataFrame] = []
    for fold, keys in sampled.groupby("oos_fold", sort=True):
        keys = keys.reset_index(drop=True)
        fold_dir = models_dir / str(fold)
        columns = _load_columns(fold_dir / "columns.json")
        raw_features = [
            name for name in columns if name != "side" and name not in sidecar_output_set
        ]
        generated_features = [name for name in columns if name in sidecar_output_set]
        raw, loader = _load_feature_store_columns(
            keys.loc[:, list(KEY_COLUMNS)],
            feature_dir=feature_store,
            selected_features=raw_features,
            min_feature_finite_frac=0.0,
        )
        missing_raw = sorted(set(raw_features) - set(raw.columns))
        if missing_raw:
            raise ValueError(f"fold {fold} missing static features: {missing_raw}")
        generated = _read_sidecar_sample(
            sidecar,
            keys.loc[:, list(KEY_COLUMNS)],
            generated_features,
        )
        matrix = raw.copy()
        if generated_features:
            generated = keys.loc[:, list(KEY_COLUMNS)].merge(
                generated,
                on=list(KEY_COLUMNS),
                how="left",
                validate="one_to_one",
            )
            for name in generated_features:
                matrix[name] = pd.to_numeric(generated[name], errors="coerce").to_numpy(
                    dtype=np.float32
                )
        if "side" in columns:
            matrix["side"] = keys["side"].to_numpy(dtype=np.float32)
        matrix = matrix.reindex(columns=columns)
        if list(matrix.columns) != columns:
            raise ValueError(f"fold {fold} model feature order could not be reconstructed")

        # New runs persist the fitted imputation transform for provenance. Older
        # runs may predate that artifact; it is never applied here because only
        # complete rows are eligible for strict replay scoring.
        try:
            _load_training_imputation(fold_dir, columns)
            imputation_contract_status = "present_verified_not_applied"
        except ValueError as exc:
            imputation_contract_status = "legacy_missing_not_required_for_complete_rows"
            imputation_contract_detail = str(exc)
        else:
            imputation_contract_detail = ""
        model = joblib.load(fold_dir / "base_model.joblib")
        # Training fold payloads are persisted as clipped float16. Reproduce
        # that numerical boundary for complete, contract-eligible rows only.
        values = matrix.to_numpy(dtype=np.float32, copy=True)
        complete = _required_input_complete(values)
        predicted = np.full(len(values), np.nan, dtype=np.float64)
        if complete.any():
            complete_values = values[complete]
            complete_values = np.clip(
                complete_values, np.finfo(np.float16).min, np.finfo(np.float16).max
            ).astype(np.float16).astype(np.float32)
            predicted[complete] = _predict_serialized_lgbm(model, complete_values)
        expected = pd.to_numeric(keys["expected_score"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        abs_diff = np.full(len(values), np.nan, dtype=np.float64)
        rel_diff = np.full(len(values), np.nan, dtype=np.float64)
        abs_diff[complete] = np.abs(predicted[complete] - expected[complete])
        scale = np.maximum(np.abs(expected[complete]), 1e-12)
        rel_diff[complete] = abs_diff[complete] / scale
        detail = keys.loc[:, [*KEY_COLUMNS, "oos_fold"]].copy()
        detail["expected_score"] = expected
        detail["replayed_score"] = predicted
        detail["abs_diff"] = abs_diff
        detail["rel_diff"] = rel_diff
        detail["required_input_complete"] = complete
        detail["verification_status"] = np.where(
            complete,
            "eligible_complete",
            "historically_scored_incomplete_contract_violation",
        )
        row_reports.append(detail)
        complete_abs_diff = abs_diff[complete]
        complete_rel_diff = rel_diff[complete]
        incomplete_rows = int((~complete).sum())
        fold_reports.append(
            {
                "fold": str(fold),
                "rows": int(len(keys)),
                "eligible_complete_rows": int(complete.sum()),
                "historically_scored_incomplete_rows": incomplete_rows,
                "required_input_completeness": float(complete.mean()),
                "features": int(len(columns)),
                "raw_features": int(len(raw_features)),
                "frozen_ae_gmm_features": int(len(generated_features)),
                "max_abs_diff": (
                    float(np.max(complete_abs_diff))
                    if len(complete_abs_diff)
                    else None
                ),
                "max_rel_diff": (
                    float(np.max(complete_rel_diff))
                    if len(complete_rel_diff)
                    else None
                ),
                "within_tolerance": bool(
                    incomplete_rows == 0
                    and len(complete_rel_diff) > 0
                    and np.all(complete_rel_diff <= float(tolerance))
                ),
                "imputation_contract_status": imputation_contract_status,
                "imputation_contract_detail": imputation_contract_detail,
                "static_loader": loader,
            }
        )
    details = pd.concat(row_reports, ignore_index=True)
    eligible = details["required_input_complete"].to_numpy(dtype=bool)
    eligible_abs_diff = details.loc[eligible, "abs_diff"].to_numpy(dtype=np.float64)
    eligible_rel_diff = details.loc[eligible, "rel_diff"].to_numpy(dtype=np.float64)
    report = {
        "schema": "base_fold_static_store_prediction_parity_v2",
        "report_dir": str(report_dir),
        "oos_ledger": str(oos_path),
        "feature_store": str(feature_store),
        "frozen_ae_gmm_sidecar": str(sidecar),
        "samples_per_fold": int(samples_per_fold),
        "sample_rows": int(len(details)),
        "folds": int(details["oos_fold"].nunique()),
        "eligible_complete_rows": int(eligible.sum()),
        "historically_scored_incomplete_rows": int((~eligible).sum()),
        "required_input_completeness": float(eligible.mean()),
        "incomplete_row_policy": "contract_violation_exclude_or_retrain_never_impute_for_eligibility",
        "relative_tolerance": float(tolerance),
        "max_abs_diff": float(np.max(eligible_abs_diff)) if len(eligible_abs_diff) else None,
        "max_rel_diff": float(np.max(eligible_rel_diff)) if len(eligible_rel_diff) else None,
        "pass": bool(all(row["within_tolerance"] for row in fold_reports)),
        "fold_reports": fold_reports,
        "sidecar_contract": sidecar_contract,
        "row_details": details,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--samples-per-fold", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_base_fold_prediction_parity(
        report_dir=args.report_dir,
        feature_store=args.feature_store,
        sidecar=args.sidecar,
        samples_per_fold=args.samples_per_fold,
        tolerance=args.tolerance,
    )
    details = report.pop("row_details")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output.with_suffix(".rows.csv"), index=False)
    args.output.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "pass", "folds", "sample_rows", "max_abs_diff", "max_rel_diff"
    )}, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
