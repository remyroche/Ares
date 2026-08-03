"""Materialize a versioned canonical exact-H12 target pack.

The upstream exact-H12 pack is immutable and predates the roadmap's explicit
support-label metadata suffixes.  This module preserves its values and policy
identity, adds the vectorized metadata projection, extends the label dictionary
and support report, and writes a new research-only pack with hashes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.target_feature_execution_alignment import (
    materialize_supportive_metadata,
    materialize_alias,
    sha256,
)


SCHEMA = "root_cause_exact_h12_execution_target_pack_v2_canonical_support_metadata"
METADATA_COLUMNS = tuple(
    f"{head}{suffix}"
    for head in (
        "peak_mfe_atr_12h", "time_to_first_meaningful_mfe_hours_12h",
        "mae_before_meaningful_mfe_atr_12h", "bars_before_price_stops_decreasing_12h",
        "future_slope_atr_per_hour_12h",
    )
    for suffix in ("__valid", "__condition_met", "__censored", "__support_count")
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metadata_dictionary() -> pd.DataFrame:
    rows = []
    descriptions = {
        "__valid": "whether the supportive target is observed on a complete path",
        "__condition_met": "whether the target's conditioning event is observed by H12",
        "__censored": "whether the conditioning event was not observed by H12",
        "__support_count": "row-level indicator contributing to the conditional support count",
    }
    for label in METADATA_COLUMNS:
        suffix = next(suffix for suffix in descriptions if label.endswith(suffix))
        unit = "count" if suffix == "__support_count" else "indicator"
        rows.append({
            "surface": "supportive",
            "label_name": label,
            "role": "supportive_metadata",
            "label_kind": "hard_binary" if suffix != "__support_count" else "continuous",
            "unit": unit,
            "condition": "meaningful MFE reached by H12" if suffix in {"__condition_met", "__censored"} else "unconditional",
            "availability": "decision_ts + 12h only",
            "model_input_allowed": False,
            "description": descriptions[suffix],
            "path_semantics": "historical_exact_1m_unadjusted_decision_path_v1",
        })
    return pd.DataFrame(rows)


def augment_support_report(source: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Append monthly/side support diagnostics for the new metadata labels."""

    decision = pd.to_datetime(labels["decision_ts"], utc=True)
    work = labels.copy()
    work["month"] = decision.dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    for label in METADATA_COLUMNS:
        grouped = work.groupby(["month", "side"], sort=True, dropna=False)[label]
        summary = grouped.agg(rows="size", non_null_rows="count", mean="mean", std="std")
        quantiles = grouped.quantile([0.05, 0.50, 0.95]).unstack(-1)
        for (month, side), values in summary.iterrows():
            rows.append({
                "surface": "supportive_metadata",
                "month": str(month),
                "side": str(side),
                "label_name": label,
                "rows": int(values["rows"]),
                "non_null_rows": int(values["non_null_rows"]),
                "mean": float(values["mean"]),
                "std": float(values["std"]) if pd.notna(values["std"]) else 0.0,
                "p05": float(quantiles.loc[(month, side), 0.05]),
                "p50": float(quantiles.loc[(month, side), 0.50]),
                "p95": float(quantiles.loc[(month, side), 0.95]),
            })
    additions = pd.DataFrame(rows, columns=source.columns)
    if source.empty:
        return additions
    return pd.concat([source, additions], ignore_index=True)


def materialize_canonical_target_pack(source_dir: Path, output_dir: Path, supportive_canonical_path: Path | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = source_dir.resolve()
    if output_dir.resolve() == source_dir:
        raise ValueError("canonical target pack must be a new versioned directory")
    if supportive_canonical_path is None:
        supportive_canonical_path = source_dir.parent / "target_alignment" / "alignment_audit_20260801_v2" / "supportive_labels_canonical.parquet"
    supportive_canonical_path = supportive_canonical_path.resolve()
    for name in ("execution_target_contract.json", "primary_labels.parquet"):
        materialize_alias(source_dir / name, output_dir / name)
    # Preserve the complete supportive surface (including competing-risk
    # fields) and add the explicit roadmap metadata.  Older canonical
    # projections contained only the five headline heads, which silently
    # removed the fields needed by T1/T3/T4.
    supportive = pd.read_parquet(supportive_canonical_path)
    metadata_source = supportive_canonical_path.parent / "supportive_label_metadata.parquet"
    if metadata_source.exists():
        metadata = pd.read_parquet(metadata_source)
    else:
        metadata_tmp = output_dir / ".supportive_label_metadata.derived.parquet"
        materialize_supportive_metadata(supportive, metadata_tmp)
        metadata = pd.read_parquet(metadata_tmp)
        metadata_tmp.unlink()
    metadata_columns = [column for column in metadata.columns if column != "candidate_id"]
    if supportive["candidate_id"].duplicated().any() or metadata["candidate_id"].duplicated().any():
        raise ValueError("canonical supportive surface requires unique candidate_id")
    overlap = sorted(set(metadata_columns).intersection(supportive.columns))
    if overlap:
        # Existing explicit columns are accepted only when the values are
        # identical; silently replacing them would weaken provenance.
        joined = supportive.merge(metadata[["candidate_id", *overlap]], on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "__metadata"))
        for column in overlap:
            if not joined[column].equals(joined[f"{column}__metadata"]):
                raise ValueError(f"supportive metadata disagrees with source column: {column}")
        metadata = metadata.drop(columns=overlap)
    supportive = supportive.merge(metadata, on="candidate_id", how="inner", validate="one_to_one")
    if len(supportive) != len(metadata):
        raise ValueError("canonical supportive metadata join changed the row population")
    supportive.to_parquet(output_dir / "supportive_labels.parquet", index=False, compression="zstd")
    metadata_only = supportive[["candidate_id", *METADATA_COLUMNS]]
    metadata_only.to_parquet(output_dir / "supportive_label_metadata.parquet", index=False, compression="zstd")

    source_dictionary = pd.read_parquet(source_dir / "label_dictionary.parquet")
    dictionary = pd.concat([source_dictionary, _metadata_dictionary()], ignore_index=True)
    if dictionary["label_name"].duplicated().any():
        raise ValueError("canonical label dictionary contains duplicate labels")
    dictionary.to_parquet(output_dir / "label_dictionary.parquet", index=False, compression="zstd")

    labels = pd.read_parquet(output_dir / "supportive_labels.parquet")
    support_report = pd.read_parquet(source_dir / "support_report.parquet")
    support_report = augment_support_report(support_report, labels)
    support_report.to_parquet(output_dir / "support_report.parquet", index=False, compression="zstd")

    # Bind the roadmap's row-level canonical field view when it is available.
    contract_view = supportive_canonical_path.parent / "candidate_target_contract.parquet"
    if contract_view.exists():
        materialize_alias(contract_view, output_dir / "candidate_target_contract.parquet")
    else:
        primary = pd.read_parquet(output_dir / "primary_labels.parquet")
        contract = pd.DataFrame({
            "candidate_id": primary["candidate_id"],
            "symbol": primary["symbol"],
            "side": primary["side"],
            "decision_ts": primary["decision_ts"],
            "entry_ts": primary["entry_ts"],
            "entry_price": primary["execution_entry_price"],
            "horizon_end_ts": primary["label_end_ts"],
            "label_available_ts": primary["label_available_ts"],
            "row_cost_bps": primary["execution_exact_h12_cost_bps"],
            "policy_geometry_id": primary["execution_geometry_id"],
            "path_source": "historical_exact_1m_unadjusted_decision_path_v1",
            "path_complete": True,
            "execution_policy_id": primary["execution_policy_id"],
            "cost_model_id": primary["cost_model_id"],
        })
        contract.to_parquet(output_dir / "candidate_target_contract.parquet", index=False, compression="zstd")

    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schema": SCHEMA,
        "status": "MATERIALIZED_RESEARCH_ONLY_TARGETS_NOT_MODEL_INPUTS",
        "promotion_eligible": False,
        "source_pack": str(source_dir),
        "source_pack_sha256": sha256(source_dir / "manifest.json"),
        "supportive_canonical_source": str(supportive_canonical_path),
        "supportive_canonical_source_sha256": sha256(supportive_canonical_path),
        "source_schema": source_manifest.get("schema"),
        "rows": int(len(labels)),
        "supportive_metadata_columns": list(METADATA_COLUMNS),
        "assertions": [
            "canonical supportive values are preserved from the immutable exact-H12 pack",
            "explicit support metadata is vectorized and unavailable until label_end_ts",
            "all primary and supportive labels remain forbidden model inputs",
            "primary policy, cost, geometry and exact net accounting remain unchanged",
        ],
        "outputs": {},
    }
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["outputs"][path.name] = sha256(path)
    write_json(output_dir / "manifest.json", manifest)
    manifest["outputs"]["manifest.json"] = sha256(output_dir / "manifest.json")
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest
