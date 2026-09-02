#!/usr/bin/env python3
"""Audit strict base-feature completeness on the persisted top-30 meta handoff.

The audit is read-only.  It reconstructs the base model's saved ordered feature
contract from every fold ``columns.json``, reads raw features through the shared
static-store loader, and reads selected frozen AE/GMM outputs through the keyed
sidecar reader.  Candidate rows are scanned one UTC month at a time so neither
the handoff nor the wide feature matrix is materialized in full.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import AE_GMM_FEATURE_COLUMNS
from scripts.run_label_quality_proxy_diagnostics import _load_feature_store_columns
from scripts.verify_base_frozen_aegmm_static_store_parity import (
    KEY_COLUMNS,
    _normalize_side,
    _read_sidecar_sample,
    _sidecar_output_features,
)


CONTRACT_LIST_KEYS = (
    "feature_names",
    "selected_features",
    "lgbm_selected_model_features",
    "feat_cols",
)
HANDOFF_REQUIRED_KEYS = ("__ts__", "__symbol__", "selected_top30")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _feature_list_from_columns_payload(payload: dict[str, Any], path: Path) -> list[str]:
    for key in CONTRACT_LIST_KEYS:
        values = payload.get(key)
        if isinstance(values, list) and values:
            features = [str(value) for value in values]
            if len(features) != len(set(features)):
                duplicates = sorted(name for name, count in Counter(features).items() if count > 1)
                raise ValueError(f"feature contract has duplicate columns in {path}: {duplicates[:12]}")
            return features
    raise ValueError(f"no supported ordered feature list in {path}")


def load_base_selected_feature_contract(
    report_dir: Path, *, expected_feature_count: int = 150
) -> tuple[list[str], list[Path]]:
    """Return the one ordered contract shared by all saved base fold artifacts."""

    model_root = Path(report_dir) / "models"
    files = sorted(model_root.glob("**/columns.json"))
    if not files:
        raise FileNotFoundError(f"No saved base columns.json files under {model_root}")
    contracts: list[tuple[Path, list[str]]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid columns.json payload in {path}")
        contracts.append((path, _feature_list_from_columns_payload(payload, path)))
    selected = contracts[0][1]
    mismatched = [str(path) for path, features in contracts if features != selected]
    if mismatched:
        raise ValueError(
            "saved base feature contracts differ in ordered content: "
            f"reference={contracts[0][0]} mismatched={mismatched[:12]}"
        )
    if len(selected) != int(expected_feature_count):
        raise ValueError(
            f"expected exactly {expected_feature_count} base selected features, found {len(selected)}"
        )
    return selected, [path for path, _features in contracts]


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _parquet_schema_names(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(map(str, pq.read_schema(path).names))
    except Exception:
        return list(map(str, pd.read_parquet(path).columns))


def _handoff_side_column(schema: Sequence[str]) -> str:
    for name in ("side", "side_name", "__side__"):
        if name in schema:
            return name
    raise ValueError("candidate handoff is missing a side or side_name identity column")


def _validate_handoff_schema(path: Path) -> tuple[list[str], str]:
    schema = _parquet_schema_names(path)
    missing = sorted(set(HANDOFF_REQUIRED_KEYS).difference(schema))
    if missing:
        raise ValueError(f"candidate handoff is missing required columns: {missing}")
    return schema, _handoff_side_column(schema)


def _handoff_months(path: Path, *, side_column: str, side_filter: str) -> list[pd.Timestamp]:
    escaped = str(path.resolve()).replace("'", "''")
    side_expr = _quote_identifier(side_column)
    predicate = 'TRY_CAST("selected_top30" AS BOOLEAN) IS TRUE'
    if side_filter != "all":
        target = -1 if side_filter == "short" else 1
        predicate += (
            " AND CASE "
            f"WHEN lower(trim(CAST({side_expr} AS VARCHAR))) = 'short' "
            f"OR TRY_CAST({side_expr} AS DOUBLE) < 0 THEN -1 "
            f"WHEN lower(trim(CAST({side_expr} AS VARCHAR))) = 'long' "
            f"OR TRY_CAST({side_expr} AS DOUBLE) > 0 THEN 1 ELSE 0 END = {target}"
        )
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone='UTC'")
        rows = connection.execute(
            f"""
            SELECT DISTINCT date_trunc('month', CAST("__ts__" AS TIMESTAMPTZ)) AS month
            FROM read_parquet('{escaped}')
            WHERE {predicate}
            ORDER BY month
            """
        ).fetchdf()
    finally:
        connection.close()
    months = pd.to_datetime(rows.get("month"), utc=True, errors="coerce").dropna().tolist()
    if not months:
        raise ValueError("candidate handoff has no selected_top30 rows for the requested side")
    return [pd.Timestamp(month) for month in months]


def _read_handoff_month(
    path: Path,
    *,
    month: pd.Timestamp,
    side_column: str,
    side_filter: str,
) -> pd.DataFrame:
    """Read only the persisted top-30 identities for one UTC month."""

    escaped = str(path.resolve()).replace("'", "''")
    side_expr = _quote_identifier(side_column)
    month_start = pd.Timestamp(month).tz_convert("UTC").strftime("%Y-%m-%d")
    predicate = (
        'TRY_CAST("selected_top30" AS BOOLEAN) IS TRUE '
        "AND date_trunc('month', CAST(\"__ts__\" AS TIMESTAMPTZ)) "
        f"= TIMESTAMPTZ '{month_start} 00:00:00+00'"
    )
    if side_filter != "all":
        target = -1 if side_filter == "short" else 1
        predicate += (
            " AND CASE "
            f"WHEN lower(trim(CAST({side_expr} AS VARCHAR))) = 'short' "
            f"OR TRY_CAST({side_expr} AS DOUBLE) < 0 THEN -1 "
            f"WHEN lower(trim(CAST({side_expr} AS VARCHAR))) = 'long' "
            f"OR TRY_CAST({side_expr} AS DOUBLE) > 0 THEN 1 ELSE 0 END = {target}"
        )
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone='UTC'")
        out = connection.execute(
            f"""
            SELECT
                CAST("__ts__" AS TIMESTAMPTZ) AS "__ts__",
                CAST("__symbol__" AS VARCHAR) AS "__symbol__",
                {side_expr} AS "__handoff_side__"
            FROM read_parquet('{escaped}')
            WHERE {predicate}
            ORDER BY "__ts__", "__symbol__", "__handoff_side__"
            """
        ).fetchdf()
    finally:
        connection.close()
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
    out["__symbol__"] = out["__symbol__"].astype(str)
    out["side"] = _normalize_side(out.pop("__handoff_side__"))
    if out["__ts__"].isna().any() or out["__symbol__"].str.len().eq(0).any() or out["side"].eq(0).any():
        raise ValueError(f"invalid UTC timestamp, symbol, or side identity in handoff month {month_start}")
    if out.duplicated(list(KEY_COLUMNS), keep=False).any():
        raise ValueError(f"duplicate candidate handoff identities in UTC month {month_start}")
    return out.reset_index(drop=True)


def _feature_sources(
    selected_features: Sequence[str], sidecar_outputs: Sequence[str]
) -> tuple[list[str], list[str]]:
    sidecar_set = set(map(str, sidecar_outputs))
    ae_gmm_set = set(AE_GMM_FEATURE_COLUMNS)
    undeclared = [
        name
        for name in selected_features
        if name in ae_gmm_set and name not in sidecar_set
    ]
    if undeclared:
        raise ValueError(
            "selected AE/GMM outputs missing from frozen sidecar contract: "
            f"{undeclared[:12]}"
        )
    raw = [name for name in selected_features if name != "side" and name not in sidecar_set]
    generated = [name for name in selected_features if name in sidecar_set]
    return raw, generated


def reconstruct_candidate_matrix(
    keys: pd.DataFrame,
    *,
    feature_store: Path,
    sidecar: Path,
    selected_features: Sequence[str],
    sidecar_outputs: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild one selected-candidate matrix without eligibility imputation."""

    if keys.duplicated(list(KEY_COLUMNS), keep=False).any():
        raise ValueError("candidate identities must be unique before feature reconstruction")
    raw_features, generated_features = _feature_sources(selected_features, sidecar_outputs)
    raw, static_loader = _load_feature_store_columns(
        keys.loc[:, list(KEY_COLUMNS)],
        feature_dir=Path(feature_store),
        selected_features=raw_features,
        min_feature_finite_frac=0.0,
    )
    missing_raw = sorted(set(raw_features).difference(raw.columns))
    if missing_raw:
        raise ValueError(f"canonical static store omitted selected raw features: {missing_raw[:12]}")
    matrix = raw.reindex(index=keys.index, columns=raw_features).copy()
    if generated_features:
        sidecar_rows = _read_sidecar_sample(sidecar, keys, generated_features)
        if sidecar_rows.duplicated(list(KEY_COLUMNS), keep=False).any():
            raise ValueError("frozen AE/GMM sidecar has duplicate matching identities")
        joined = keys.loc[:, list(KEY_COLUMNS)].merge(
            sidecar_rows,
            on=list(KEY_COLUMNS),
            how="left",
            validate="one_to_one",
            sort=False,
        )
        for name in generated_features:
            matrix[name] = joined[name].to_numpy(copy=False)
    if "side" in selected_features:
        matrix["side"] = keys["side"].to_numpy(dtype=np.float32, copy=False)
    matrix = matrix.reindex(columns=list(selected_features))
    if list(matrix.columns) != list(selected_features):
        raise RuntimeError("saved selected feature order could not be reconstructed")
    for name in matrix.columns:
        matrix[name] = pd.to_numeric(matrix[name], errors="coerce")
    return matrix, {
        "static_loader": static_loader,
        "raw_feature_count": int(len(raw_features)),
        "frozen_ae_gmm_feature_count": int(len(generated_features)),
    }


def summarize_completeness(
    keys: pd.DataFrame, matrix: pd.DataFrame, selected_features: Sequence[str]
) -> tuple[dict[str, int | float], Counter[str], pd.DataFrame]:
    """Return strict joint completeness and per-feature non-finite counts."""

    if len(keys) != len(matrix):
        raise ValueError("candidate keys and reconstructed matrix have different row counts")
    values = matrix.reindex(columns=list(selected_features)).to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(values)
    complete = finite.all(axis=1)
    missing = Counter(
        {
            name: int((~finite[:, index]).sum())
            for index, name in enumerate(selected_features)
        }
    )
    details = keys.loc[:, list(KEY_COLUMNS)].copy()
    details["required_input_complete"] = complete
    summary: dict[str, int | float] = {
        "rows": int(len(details)),
        "complete_rows": int(complete.sum()),
        "incomplete_rows": int((~complete).sum()),
        "joint_complete_fraction": float(complete.mean()) if len(complete) else 0.0,
    }
    return summary, missing, details


def _combine_group_summaries(records: Iterable[dict[str, Any]], columns: Sequence[str]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(list(records))
    if frame.empty:
        return []
    grouped = frame.groupby(list(columns), dropna=False, sort=True)[
        ["rows", "complete_rows", "incomplete_rows"]
    ].sum().reset_index()
    grouped["joint_complete_fraction"] = np.where(
        grouped["rows"] > 0,
        grouped["complete_rows"] / grouped["rows"],
        0.0,
    )
    return grouped.to_dict(orient="records")


def audit_base_candidate_feature_completeness(
    *,
    report_dir: Path,
    feature_store: Path,
    sidecar: Path,
    candidate_handoff: Path,
    side: str = "all",
    expected_feature_count: int = 150,
    top_missing_limit: int = 25,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run the bounded strict-completeness audit and return JSON/CSV payloads."""

    side = str(side).strip().lower()
    if side not in {"all", "long", "short"}:
        raise ValueError("side must be one of: all, long, short")
    selected_features, contract_paths = load_base_selected_feature_contract(
        Path(report_dir), expected_feature_count=expected_feature_count
    )
    _schema, side_column = _validate_handoff_schema(Path(candidate_handoff))
    sidecar_outputs, sidecar_contract = _sidecar_output_features(Path(sidecar))
    raw_features, generated_features = _feature_sources(selected_features, sidecar_outputs)
    months = _handoff_months(
        Path(candidate_handoff), side_column=side_column, side_filter=side
    )

    missing_counts: Counter[str] = Counter({name: 0 for name in selected_features})
    chunk_records: list[dict[str, Any]] = []
    loader_records: list[dict[str, Any]] = []
    total_rows = 0
    total_complete = 0
    for month in months:
        keys = _read_handoff_month(
            Path(candidate_handoff),
            month=month,
            side_column=side_column,
            side_filter=side,
        )
        matrix, reconstruction = reconstruct_candidate_matrix(
            keys,
            feature_store=Path(feature_store),
            sidecar=Path(sidecar),
            selected_features=selected_features,
            sidecar_outputs=sidecar_outputs,
        )
        summary, month_missing, details = summarize_completeness(keys, matrix, selected_features)
        missing_counts.update(month_missing)
        total_rows += int(summary["rows"])
        total_complete += int(summary["complete_rows"])
        details["month"] = pd.Timestamp(month).strftime("%Y-%m")
        details["side_name"] = np.where(details["side"].to_numpy(dtype=np.int8) < 0, "short", "long")
        for (month_name, side_name), group in details.groupby(["month", "side_name"], sort=True):
            chunk_records.append(
                {
                    "month": str(month_name),
                    "side": str(side_name),
                    "rows": int(len(group)),
                    "complete_rows": int(group["required_input_complete"].sum()),
                    "incomplete_rows": int((~group["required_input_complete"]).sum()),
                }
            )
        loader_records.append(
            {
                "month": pd.Timestamp(month).strftime("%Y-%m"),
                "rows": int(len(keys)),
                **reconstruction,
            }
        )

    feature_rows = pd.DataFrame(
        {
            "feature": selected_features,
            "source": [
                "side_numeric" if name == "side" else "frozen_ae_gmm_sidecar" if name in generated_features else "canonical_static_store"
                for name in selected_features
            ],
            "missing_rows": [int(missing_counts[name]) for name in selected_features],
        }
    )
    feature_rows["finite_rows"] = int(total_rows) - feature_rows["missing_rows"]
    feature_rows["missing_fraction"] = np.where(
        total_rows > 0, feature_rows["missing_rows"] / float(total_rows), 0.0
    )
    feature_rows = feature_rows.sort_values(
        ["missing_rows", "feature"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    by_month_side = _combine_group_summaries(chunk_records, ["month", "side"])
    by_month = _combine_group_summaries(chunk_records, ["month"])
    by_side = _combine_group_summaries(chunk_records, ["side"])
    overall = {
        "rows": int(total_rows),
        "complete_rows": int(total_complete),
        "incomplete_rows": int(total_rows - total_complete),
        "joint_complete_fraction": float(total_complete / total_rows) if total_rows else 0.0,
    }
    report = {
        "schema": "base_candidate_feature_completeness_audit_v1",
        "pass": bool(total_rows > 0 and total_complete == total_rows),
        "incomplete_row_policy": "contract_violation_never_impute_for_eligibility",
        "inputs": {
            "base_report_dir": str(Path(report_dir)),
            "feature_store": str(Path(feature_store)),
            "frozen_ae_gmm_sidecar": str(Path(sidecar)),
            "candidate_handoff": str(Path(candidate_handoff)),
            "side_filter": side,
        },
        "contract": {
            "selected_feature_count": int(len(selected_features)),
            "selected_feature_order": selected_features,
            "columns_json_paths": [str(path) for path in contract_paths],
            "raw_feature_count": int(len(raw_features)),
            "frozen_ae_gmm_feature_count": int(len(generated_features)),
            "side_numeric_included": bool("side" in selected_features),
        },
        "handoff": {
            "selection_predicate": "selected_top30 IS TRUE (persisted handoff; no reranking)",
            "side_identity_column": side_column,
            "utc_month_chunks": [pd.Timestamp(month).strftime("%Y-%m") for month in months],
        },
        "overall": overall,
        "by_month": by_month,
        "by_side": by_side,
        "by_month_side": by_month_side,
        "per_feature_missing": feature_rows.to_dict(orient="records"),
        "top_missing_features": feature_rows.head(max(0, int(top_missing_limit))).to_dict(orient="records"),
        "static_loader_chunks": loader_records,
        "sidecar_contract": sidecar_contract,
    }
    return report, feature_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--candidate-handoff", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--side", choices=("all", "long", "short"), default="all")
    parser.add_argument("--top-missing-limit", type=int, default=25)
    args = parser.parse_args()
    report, feature_rows = audit_base_candidate_feature_completeness(
        report_dir=args.report_dir,
        feature_store=args.feature_store,
        sidecar=args.sidecar,
        candidate_handoff=args.candidate_handoff,
        side=args.side,
        top_missing_limit=args.top_missing_limit,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    feature_rows.to_csv(args.output_csv, index=False)
    print(json.dumps({"pass": report["pass"], **report["overall"]}, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
