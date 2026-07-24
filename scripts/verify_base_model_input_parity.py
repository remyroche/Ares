#!/usr/bin/env python3
"""Verify persisted base OOS model-input hashes against the canonical store.

This is deliberately stricter than score replay.  It rebuilds the exact model
matrix for every requested persisted OOS row from the read-only static store
and the frozen AE/GMM output sidecar, restores the saved fold train-only
imputation, and crosses the saved float16 -> float32 numeric boundary.  The
result is compared with the row hashes and deterministic B/M/E feature anchors
persisted beside the OOS scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import duckdb

from extreme_price_movements.feature_transform_contract import (
    apply_model_input_numeric_contract,
    build_model_input_numeric_contract,
    compute_contract_hash,
    model_matrix_hash,
    ordered_names_hash,
)
from scripts.run_label_quality_proxy_diagnostics import _load_feature_store_columns
from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (
    MODEL_INPUT_PARITY_SCHEMA,
    _feature_contract_hash,
    _model_input_row_hashes,
)
from scripts.verify_base_fold_static_store_prediction_parity import (
    _feature_order_hash,
    _imputation_contract_hash,
    _load_training_imputation,
)
from scripts.verify_base_frozen_aegmm_static_store_parity import (
    KEY_COLUMNS,
    _normalize_side,
    _read_sidecar_sample,
    _sidecar_output_features,
)


HASH_MODES = ("anchors", "sample", "all")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _safe_fold_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def _feature_order_digest(columns: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps([str(name) for name in columns], separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact is not a JSON object: {path}")
    return payload


def _load_feature_contracts(columns_path: Path) -> dict[str, list[str]]:
    payload = _read_json(columns_path)
    by_side = payload.get("feature_names_by_side")
    if isinstance(by_side, dict) and by_side:
        contracts = {
            str(side): [str(name) for name in names]
            for side, names in by_side.items()
            if isinstance(names, list) and names
        }
        if not contracts:
            raise ValueError(f"side-local feature contract is empty: {columns_path}")
        return contracts
    names = [str(name) for name in payload.get("feature_names", [])]
    if not names:
        raise ValueError(f"feature contract is empty: {columns_path}")
    return {"shared": names}


def _find_model_fold_dir(models_dir: Path, fold: str) -> Path:
    direct = models_dir / _safe_fold_name(fold)
    if (direct / "columns.json").is_file() and (direct / "manifest.json").is_file():
        return direct
    matches: list[Path] = []
    for manifest_path in models_dir.glob("*/manifest.json"):
        payload = _read_json(manifest_path)
        if str(payload.get("fold")) == str(fold):
            matches.append(manifest_path.parent)
    if len(matches) != 1:
        raise ValueError(
            f"unable to resolve exactly one model directory for fold {fold!r}: {matches}"
        )
    return matches[0]


def _normalize_hash_keys(frame: pd.DataFrame, *, artifact: str) -> pd.DataFrame:
    required = [*KEY_COLUMNS, "model_side", "feature_contract_hash", "numeric_contract_hash"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"{artifact} is missing required columns: {missing}")
    out = frame.copy()
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
    out["__symbol__"] = out["__symbol__"].astype(str)
    out["side"] = _normalize_side(out["side"])
    out["model_side"] = out["model_side"].astype(str)
    if out.loc[:, [*KEY_COLUMNS, "model_side"]].isna().any(axis=None):
        raise ValueError(f"{artifact} contains invalid model-input keys")
    if out.duplicated([*KEY_COLUMNS, "model_side"], keep=False).any():
        raise ValueError(f"{artifact} contains duplicate model-input keys")
    return out


def _deterministic_sample_positions(row_count: int, max_rows: int) -> np.ndarray:
    if row_count <= 0:
        return np.empty(0, dtype=np.int64)
    take = min(int(row_count), max(1, int(max_rows)))
    return np.unique(np.floor(np.linspace(0, row_count - 1, take)).astype(np.int64))


def _sample_hash_rows(hashes: pd.DataFrame, *, mode: str, sample_rows: int) -> pd.DataFrame:
    if mode not in HASH_MODES:
        raise ValueError(f"unknown hash mode {mode!r}; expected one of {HASH_MODES}")
    if mode == "all":
        return hashes.copy()
    positions: list[np.ndarray] = []
    for _side, group in hashes.groupby("model_side", sort=True):
        if mode == "anchors":
            positions.append(np.empty(0, dtype=np.int64))
        else:
            positions.append(group.index.to_numpy()[
                _deterministic_sample_positions(len(group), sample_rows)
            ])
    selected = np.concatenate(positions) if positions else np.empty(0, dtype=np.int64)
    return hashes.loc[selected].copy()


def _read_hash_summary(path: Path) -> pd.DataFrame:
    """Read only group-level contract metadata from a large row-hash sidecar."""

    escaped = str(path.resolve()).replace("'", "''")
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone='UTC'")
        summary = connection.execute(
            f"""
            SELECT
                CAST(model_side AS VARCHAR) AS model_side,
                count(*) AS rows,
                count(DISTINCT feature_contract_hash) AS feature_contracts,
                count(DISTINCT numeric_contract_hash) AS numeric_contracts,
                min(CAST(feature_contract_hash AS VARCHAR)) AS feature_contract_hash,
                min(CAST(numeric_contract_hash AS VARCHAR)) AS numeric_contract_hash
            FROM read_parquet('{escaped}')
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchdf()
    finally:
        connection.close()
    if summary.empty:
        raise ValueError(f"row-hash artifact is empty: {path}")
    return summary


def _read_hash_rows(
    path: Path, *, mode: str, sample_rows: int
) -> pd.DataFrame:
    """Read all hashes only for full verification; otherwise take B/M/E rows."""

    if mode not in HASH_MODES:
        raise ValueError(f"unknown hash mode {mode!r}; expected one of {HASH_MODES}")
    if mode == "all":
        return _normalize_hash_keys(pd.read_parquet(path), artifact=str(path))
    if mode == "anchors":
        # The anchor table itself drives reconstruction in this mode.
        return pd.DataFrame()
    escaped = str(path.resolve()).replace("'", "''")
    take = max(1, int(sample_rows))
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone='UTC'")
        rows = connection.execute(
            f"""
            WITH ordered AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY model_side ORDER BY __ts__, __symbol__, side
                    ) AS row_number_in_side,
                    count(*) OVER (PARTITION BY model_side) AS rows_in_side
                FROM read_parquet('{escaped}')
            )
            SELECT * EXCLUDE (row_number_in_side, rows_in_side)
            FROM ordered
            WHERE rows_in_side <= {take}
               OR floor((row_number_in_side - 1) * {take} / rows_in_side)
                  < floor(row_number_in_side * {take} / rows_in_side)
            ORDER BY model_side, __ts__, __symbol__, side
            """
        ).fetchdf()
    finally:
        connection.close()
    return _normalize_hash_keys(rows, artifact=str(path))


def _assert_anchor_keys_exist(hashes_path: Path, anchors: pd.DataFrame) -> None:
    """Validate anchors against the complete hash table without loading it."""

    if anchors.empty:
        raise ValueError("anchor artifact has no rows")
    escaped = str(hashes_path.resolve()).replace("'", "''")
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone='UTC'")
        connection.register("anchors", anchors.loc[:, [*KEY_COLUMNS, "model_side"]])
        missing = connection.execute(
            f"""
            SELECT a.*
            FROM anchors AS a
            LEFT JOIN read_parquet('{escaped}') AS h
              ON CAST(h.__ts__ AS TIMESTAMPTZ) = a.__ts__
             AND CAST(h.__symbol__ AS VARCHAR) = a.__symbol__
             AND CAST(h.side AS SMALLINT) = a.side
             AND CAST(h.model_side AS VARCHAR) = a.model_side
            WHERE h.model_side IS NULL
            LIMIT 1
            """
        ).fetchdf()
    finally:
        connection.close()
    if not missing.empty:
        raise ValueError("anchors contain keys absent from row hashes")


def _validate_numeric_contract(
    *,
    expected: dict[str, Any],
    columns: list[str],
    artifact: str,
) -> None:
    expected_hash = str(expected.get("contract_hash") or "")
    if not expected_hash or expected_hash != compute_contract_hash(expected):
        raise ValueError(f"{artifact} numeric contract hash is invalid")
    if str(expected.get("feature_names_hash") or "") != ordered_names_hash(columns):
        raise ValueError(f"{artifact} numeric contract feature order mismatch")
    canonical = build_model_input_numeric_contract(columns).asdict()
    for name in ("schema_version", "name", "source_dtype", "storage_dtype", "prediction_dtype", "clip_abs", "require_finite", "feature_names_hash"):
        if expected.get(name) != canonical.get(name):
            raise ValueError(
                f"{artifact} numeric contract differs for {name}: "
                f"expected={expected.get(name)!r} canonical={canonical.get(name)!r}"
            )


def _restore_matrix(
    *,
    keys: pd.DataFrame,
    raw: pd.DataFrame,
    generated: pd.DataFrame,
    columns: list[str],
    sidecar_output_set: set[str],
    fill_values: np.ndarray,
) -> pd.DataFrame:
    if len(fill_values) != len(columns):
        raise ValueError("saved imputation values do not match feature contract")
    matrix = pd.DataFrame(index=keys.index, columns=columns, dtype=np.float32)
    raw_columns = [name for name in columns if name != "side" and name not in sidecar_output_set]
    generated_columns = [name for name in columns if name in sidecar_output_set]
    missing_raw = sorted(set(raw_columns).difference(raw.columns))
    missing_generated = sorted(set(generated_columns).difference(generated.columns))
    if missing_raw or missing_generated:
        raise ValueError(
            "reconstructed inputs are missing selected features: "
            f"static={missing_raw[:20]} frozen_ae_gmm={missing_generated[:20]}"
        )
    for name in raw_columns:
        matrix[name] = pd.to_numeric(raw[name], errors="coerce").to_numpy(dtype=np.float32)
    for name in generated_columns:
        matrix[name] = pd.to_numeric(generated[name], errors="coerce").to_numpy(dtype=np.float32)
    if "side" in columns:
        matrix["side"] = keys["side"].to_numpy(dtype=np.float32)
    values = matrix.to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.broadcast_to(fill_values, values.shape)[missing]
    if not np.isfinite(values).all():
        raise ValueError("train-only imputation did not produce finite model inputs")
    return pd.DataFrame(values, index=keys.index, columns=columns)


def _load_reconstructed_inputs(
    *,
    keys: pd.DataFrame,
    feature_store: Path,
    sidecar: Path,
    raw_features: list[str],
    generated_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw, loader_report = _load_feature_store_columns(
        keys.loc[:, list(KEY_COLUMNS)],
        feature_dir=feature_store,
        selected_features=raw_features,
        min_feature_finite_frac=0.0,
    )
    if len(raw) != len(keys):
        raise ValueError("canonical static store returned a misaligned model-input matrix")
    generated = pd.DataFrame(index=keys.index)
    if generated_features:
        selected = _read_sidecar_sample(
            sidecar, keys.loc[:, list(KEY_COLUMNS)], generated_features
        )
        if selected.duplicated(list(KEY_COLUMNS), keep=False).any():
            raise ValueError("frozen AE/GMM sidecar has duplicate model-input keys")
        generated = keys.loc[:, list(KEY_COLUMNS)].merge(
            selected,
            on=list(KEY_COLUMNS),
            how="left",
            validate="one_to_one",
        )
        generated.index = keys.index
    return raw, generated, loader_report


def _verify_anchor_values(
    *,
    anchors: pd.DataFrame,
    reconstructed: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[bool, float, str | None]:
    if anchors.empty:
        return False, float("nan"), "anchor artifact has no rows"
    missing_features = [name for name in feature_columns if name not in anchors.columns]
    if missing_features:
        return False, float("nan"), f"anchor artifact missing features: {missing_features[:20]}"
    expected = anchors.loc[:, feature_columns].to_numpy(dtype=np.float32)
    actual = reconstructed.loc[:, feature_columns].to_numpy(dtype=np.float32)
    equal = np.array_equal(expected, actual, equal_nan=False)
    max_abs = float(np.max(np.abs(expected - actual))) if len(expected) else 0.0
    if equal:
        return True, max_abs, None
    positions = np.argwhere(expected != actual)
    row, col = positions[0]
    return False, max_abs, (
        f"anchor mismatch row={int(row)} feature={feature_columns[int(col)]} "
        f"expected={float(expected[row, col])} actual={float(actual[row, col])}"
    )


def _contracts_for_parity(
    *,
    parity_manifest: dict[str, Any],
    model_contracts: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    scope = str(parity_manifest.get("model_side_scope") or "")
    expected_scope = "shared" if set(model_contracts) == {"shared"} else "per_side"
    if scope != expected_scope:
        raise ValueError(
            f"model side scope mismatch: parity={scope!r} model={expected_scope!r}"
        )
    contracts = parity_manifest.get("contracts_by_model_side")
    if not isinstance(contracts, dict) or set(contracts) != set(model_contracts):
        raise ValueError("parity manifest model-side contracts do not match columns.json")
    for model_side, columns in model_contracts.items():
        contract = contracts[model_side]
        if [str(name) for name in contract.get("feature_names", [])] != columns:
            raise ValueError(f"{model_side} parity feature order does not match columns.json")
        if str(contract.get("feature_contract_hash") or "") != _feature_contract_hash(columns):
            raise ValueError(f"{model_side} parity feature hash mismatch")
        _validate_numeric_contract(
            expected=dict(contract.get("numeric_contract") or {}),
            columns=columns,
            artifact=f"{model_side} parity manifest",
        )
    return {str(key): dict(value) for key, value in contracts.items()}


def verify_base_model_input_parity(
    *,
    report_dir: Path,
    feature_store: Path,
    sidecar: Path,
    hash_mode: str = "sample",
    sample_rows_per_fold: int = 256,
) -> dict[str, Any]:
    """Verify every anchor and sampled or all persisted OOS input hashes."""

    report_dir = Path(report_dir)
    parity_root = report_dir / "model_input_parity"
    models_dir = report_dir / "models"
    if not parity_root.is_dir():
        raise ValueError(f"model input parity root is missing: {parity_root}")
    sidecar_outputs, sidecar_contract = _sidecar_output_features(Path(sidecar))
    sidecar_set = set(sidecar_outputs)
    fold_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for parity_path in sorted(parity_root.glob("*/manifest.json")):
        parity_manifest = _read_json(parity_path)
        fold = str(parity_manifest.get("fold") or "")
        if parity_manifest.get("schema") != MODEL_INPUT_PARITY_SCHEMA or not fold:
            raise ValueError(f"invalid parity manifest: {parity_path}")
        hashes_path = Path(str(parity_manifest.get("row_hashes_path") or ""))
        anchors_path = Path(str(parity_manifest.get("anchors_path") or ""))
        if not hashes_path.is_file() or not anchors_path.is_file():
            raise ValueError(f"fold {fold} is missing persisted row hash or anchor parquet")
        hash_summary = _read_hash_summary(hashes_path)
        hashes = _read_hash_rows(
            hashes_path, mode=hash_mode, sample_rows=sample_rows_per_fold
        )
        if not hashes.empty and "model_input_row_hash" not in hashes.columns:
            raise ValueError(f"row-hash artifact is missing model_input_row_hash: {hashes_path}")
        anchors = _normalize_hash_keys(pd.read_parquet(anchors_path), artifact=str(anchors_path))
        fold_dir = _find_model_fold_dir(models_dir, fold)
        model_contracts = _load_feature_contracts(fold_dir / "columns.json")
        contracts = _contracts_for_parity(
            parity_manifest=parity_manifest, model_contracts=model_contracts
        )
        for row in hash_summary.itertuples(index=False):
            model_side = str(row.model_side)
            if model_side not in model_contracts:
                raise ValueError(f"fold {fold} row hashes reference unknown model side {model_side!r}")
            if int(row.feature_contracts) != 1 or int(row.numeric_contracts) != 1:
                raise ValueError(f"fold {fold} {model_side} row hashes contain mixed contracts")
            expected = contracts[model_side]
            if str(row.feature_contract_hash) != expected["feature_contract_hash"]:
                raise ValueError(f"fold {fold} {model_side} row-hash feature contract mismatch")
            numeric = dict(expected["numeric_contract"])
            if str(row.numeric_contract_hash) != numeric["contract_hash"]:
                raise ValueError(f"fold {fold} {model_side} row-hash numeric contract mismatch")
        if anchors.empty:
            raise ValueError(f"fold {fold} has no deterministic anchors")
        _assert_anchor_keys_exist(hashes_path, anchors)

        selected_hashes = hashes if not hashes.empty else anchors.iloc[:0].copy()
        work_hashes = pd.concat(
            [
                selected_hashes,
                anchors.loc[:, selected_hashes.columns.intersection(anchors.columns)],
            ],
            ignore_index=True,
        )
        work_hashes = work_hashes.drop_duplicates([*KEY_COLUMNS, "model_side"])
        keys = work_hashes.loc[:, list(KEY_COLUMNS)].drop_duplicates().reset_index(drop=True)
        all_columns = list(dict.fromkeys(name for names in model_contracts.values() for name in names))
        raw_features = [name for name in all_columns if name != "side" and name not in sidecar_set]
        generated_features = [name for name in all_columns if name in sidecar_set]
        raw, generated, loader_report = _load_reconstructed_inputs(
            keys=keys,
            feature_store=Path(feature_store),
            sidecar=Path(sidecar),
            raw_features=raw_features,
            generated_features=generated_features,
        )
        keyed_raw = keys.loc[:, list(KEY_COLUMNS)].copy()
        for name in raw.columns:
            keyed_raw[name] = raw[name].to_numpy(copy=False)
        keyed_generated = generated
        fold_ok = True
        side_reports: list[dict[str, Any]] = []
        for model_side, expected_contract in contracts.items():
            expected_rows = work_hashes.loc[work_hashes["model_side"].eq(model_side)].copy()
            if expected_rows.empty:
                continue
            expected_rows = expected_rows.merge(
                keys.assign(_key_order=np.arange(len(keys), dtype=np.int64)),
                on=list(KEY_COLUMNS),
                how="left",
                validate="many_to_one",
            ).sort_values("_key_order", kind="stable").drop(columns="_key_order")
            selected_keys = expected_rows.loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
            raw_aligned = selected_keys.merge(
                keyed_raw, on=list(KEY_COLUMNS), how="left", validate="one_to_one"
            )
            generated_aligned = selected_keys.merge(
                keyed_generated, on=list(KEY_COLUMNS), how="left", validate="one_to_one"
            )
            columns = model_contracts[model_side]
            fills = _load_training_imputation(fold_dir, columns)
            restored = _restore_matrix(
                keys=selected_keys,
                raw=raw_aligned,
                generated=generated_aligned,
                columns=columns,
                sidecar_output_set=sidecar_set,
                fill_values=fills,
            )
            numeric_contract = dict(expected_contract["numeric_contract"])
            scored = apply_model_input_numeric_contract(restored, numeric_contract)
            actual_hashes = _model_input_row_hashes(scored)
            expected_hashes = expected_rows.get("model_input_row_hash")
            hash_checked = (
                int(expected_hashes.notna().sum())
                if isinstance(expected_hashes, pd.Series)
                else 0
            )
            hash_match = True
            if hash_checked:
                hash_match = np.array_equal(
                    np.asarray(actual_hashes, dtype=object), expected_hashes.to_numpy(dtype=object)
                )
            anchor_rows = anchors.loc[anchors["model_side"].eq(model_side)].copy()
            anchor_match = True
            anchor_max_abs = 0.0
            anchor_detail: str | None = None
            if not anchor_rows.empty:
                anchor_rows = anchor_rows.merge(
                    selected_keys.assign(_anchor_order=np.arange(len(selected_keys), dtype=np.int64)),
                    on=list(KEY_COLUMNS), how="left", validate="one_to_one"
                ).sort_values("_anchor_order", kind="stable")
                anchor_input = anchor_rows.loc[:, columns].reset_index(drop=True)
                actual_anchor = scored.iloc[anchor_rows["_anchor_order"].to_numpy(dtype=np.int64)].reset_index(drop=True)
                anchor_match, anchor_max_abs, anchor_detail = _verify_anchor_values(
                    anchors=anchor_input,
                    reconstructed=actual_anchor,
                    feature_columns=columns,
                )
            if hash_mode == "all":
                full_rows = hashes.loc[hashes["model_side"].eq(model_side)].copy()
                # ``work_hashes`` is already all rows in this mode and persists the
                # scorer's original row order, so the full matrix hash is comparable.
                if len(full_rows) != len(scored):
                    raise ValueError(f"fold {fold} {model_side} full hash row count mismatch")
                actual_matrix_hash = model_matrix_hash(scored, row_ids=selected_keys)
                matrix_match = actual_matrix_hash == str(expected_contract.get("matrix_hash") or "")
            else:
                actual_matrix_hash = None
                matrix_match = None
            side_ok = bool(hash_match and anchor_match and (matrix_match is not False))
            fold_ok = fold_ok and side_ok
            side_reports.append(
                {
                    "model_side": model_side,
                    "hash_rows_checked": hash_checked,
                    "hash_match": bool(hash_match),
                    "anchor_rows_checked": int(len(anchor_rows)),
                    "anchors_match_exactly": bool(anchor_match),
                    "anchor_max_abs_diff": anchor_max_abs,
                    "anchor_detail": anchor_detail,
                    "matrix_hash_checked": bool(hash_mode == "all"),
                    "matrix_hash_match": matrix_match,
                    "actual_matrix_hash": actual_matrix_hash,
                    "expected_matrix_hash": expected_contract.get("matrix_hash"),
                }
            )
        fold_reports.append(
            {
                "fold": fold,
                "pass": fold_ok,
                "hash_mode": hash_mode,
                "row_hash_rows": int(hash_summary["rows"].sum()),
                "row_hash_rows_reconstructed": int(len(hashes)),
                "anchors": int(len(anchors)),
                "loader": loader_report,
                "model_sides": side_reports,
            }
        )
    if not fold_reports:
        raise ValueError(f"no model input parity manifests found under {parity_root}")
    return {
        "schema": "base_model_input_parity_verifier_v1",
        "report_dir": str(report_dir),
        "feature_store": str(feature_store),
        "frozen_ae_gmm_sidecar": str(sidecar),
        "hash_mode": hash_mode,
        "sample_rows_per_fold": int(sample_rows_per_fold),
        "sidecar_contract": sidecar_contract,
        "pass": bool(all(item["pass"] for item in fold_reports) and not errors),
        "fold_reports": fold_reports,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--hash-mode", choices=HASH_MODES, default="sample")
    parser.add_argument("--sample-rows-per-fold", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_base_model_input_parity(
        report_dir=args.report_dir,
        feature_store=args.feature_store,
        sidecar=args.sidecar,
        hash_mode=args.hash_mode,
        sample_rows_per_fold=args.sample_rows_per_fold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pass": report["pass"], "folds": len(report["fold_reports"])}, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
