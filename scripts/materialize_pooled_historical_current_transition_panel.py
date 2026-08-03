#!/usr/bin/env python3
"""Materialize the strict 90-field pooled historical/current H12 panel.

The historical 2022--2023 source is a diagnostic, non-OOF backcast.  The
existing 2025/current sources retain their original source and mapping
provenance.  All sources share exactly the semantic 90-field decision-time
geometry and exact H12 global-book before/after labels.  Targets are rebuilt
uniformly after pooling so active/onset/recovery/reversal semantics and their
availability lineage cannot drift by source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from scripts.materialize_cross_era_global_book_transition_research_panel import (
        ADVERSE_SENSITIVITY_BPS,
        _add_persistent_adverse_labels,
    )
    from scripts.materialize_historical_current_common_transition_geometry import CANONICAL_FEATURES
    from scripts.run_pooled_transition_lifecycle_diagnostic import derive_lifecycle_targets, lifecycle_target_names
except ModuleNotFoundError:
    from materialize_cross_era_global_book_transition_research_panel import ADVERSE_SENSITIVITY_BPS, _add_persistent_adverse_labels
    from materialize_historical_current_common_transition_geometry import CANONICAL_FEATURES
    from run_pooled_transition_lifecycle_diagnostic import derive_lifecycle_targets, lifecycle_target_names


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORICAL_LABELS = ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_global_book_transition_labels_20260730_v1"
DEFAULT_CURRENT_PANEL = ROOT / "data_perp/artifacts/cross_era_global_book_transition_research_panel_20260730_v4"
DEFAULT_COMMON_GEOMETRY = ROOT / "data_perp/artifacts/historical_current_common_transition_geometry_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/pooled_historical_current_transition_panel_20260730_v1"
SCHEMA = "pooled_historical_current_transition_panel_v1"
HORIZON_HOURS = 12
BOOK_FRACTION = 0.10
HISTORICAL_SOURCE = "historical_backcast_2022_2023_non_oof"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _manifest(root: Path) -> tuple[dict[str, Any], Path]:
    path = root / "manifest.json"
    sidecar = root / "manifest.sha256"
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"incomplete immutable source: {root}")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(path):
        raise ValueError(f"manifest checksum fails: {root}")
    return json.loads(path.read_text(encoding="utf-8")), path


def _verify_hash(path: Path, expected: str | None, *, source: str) -> None:
    if not expected or sha256(path) != expected:
        raise ValueError(f"{source} does not hash-bind {path.name}")


def _filter_complete_h12(labels: pd.DataFrame) -> pd.DataFrame:
    required = {
        "cohort_anchor_utc", "horizon_hours", "book_fraction",
        "before_global_hour_complete_flag", "after_global_hour_complete_flag",
        "before_selected_candidate_support", "after_selected_candidate_support",
        "before_target_available_utc", "after_target_available_utc",
        "outcome_only_not_model_feature", "delta_direct_mean_net",
        "delta_mean_conversion_residual", "after_mean_conversion_residual",
        "delta_positive_net_contribution", "delta_loss_net_contribution",
    }
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"global-book labels lack required H12 fields: {missing}")
    work = labels.copy()
    work["cohort_anchor_utc"] = pd.to_datetime(work["cohort_anchor_utc"], utc=True, errors="raise")
    selected = (
        work["horizon_hours"].eq(HORIZON_HOURS)
        & np.isclose(work["book_fraction"], BOOK_FRACTION)
        & work["before_global_hour_complete_flag"].astype(bool)
        & work["after_global_hour_complete_flag"].astype(bool)
        & work["before_selected_candidate_support"].gt(0)
        & work["after_selected_candidate_support"].gt(0)
    )
    result = work.loc[selected].copy()
    if result.empty or not result["outcome_only_not_model_feature"].astype(bool).all():
        raise ValueError("no complete outcome-guarded H12 top10 labels remain")
    identity = ["cohort_anchor_utc", "horizon_hours", "book_fraction"]
    if "source_family" in result:
        identity.append("source_family")
    if result.duplicated(identity).any():
        raise ValueError("source has duplicate H12 global-book anchors")
    result["signal_context_utc"] = result["cohort_anchor_utc"] - pd.Timedelta(hours=1)
    return result


def _base_targets(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    # Remove source-local derived targets and rebuild every target uniformly.
    work = work.drop(columns=[column for column in work.columns if column.startswith("target__")], errors="ignore")
    work["target__net_crosses_below_zero"] = (
        work["before_direct_mean_net"].gt(0.0) & work["after_direct_mean_net"].le(0.0)
    ).astype(float)
    work["target__opportunity_collapse_10pp"] = work["delta_opportunity_probability_0bps"].le(-0.10).astype(float)
    work["target__loss_expansion_25bps"] = work["delta_loss_net_contribution"].ge(0.0025).astype(float)
    work["target__soft_net_deterioration_25bps"] = np.clip(-work["delta_direct_mean_net"] / 0.0025, 0.0, 1.0)
    work["target__soft_opportunity_collapse_10pp"] = np.clip(-work["delta_opportunity_probability_0bps"] / 0.10, 0.0, 1.0)
    work["target__soft_loss_expansion_25bps"] = np.clip(work["delta_loss_net_contribution"] / 0.0025, 0.0, 1.0)
    work["target__soft_conversion_deterioration_25bps"] = np.clip(-work["delta_mean_conversion_residual"] / 0.0025, 0.0, 1.0)
    work["target__adverse_transition_any"] = work[["target__net_crosses_below_zero", "target__opportunity_collapse_10pp", "target__loss_expansion_25bps"]].max(axis=1)
    work = _add_persistent_adverse_labels(work)
    work = derive_lifecycle_targets(work)
    return work


def build_panel(
    historical_labels: pd.DataFrame,
    current_panel: pd.DataFrame,
    historical_geometry: pd.DataFrame,
    current_geometry: pd.DataFrame,
) -> pd.DataFrame:
    history = _filter_complete_h12(historical_labels)
    current = _filter_complete_h12(current_panel)
    historical_geometry = historical_geometry.copy()
    current_geometry = current_geometry.copy()
    # Each published historical-hourly row is itself an exact recovered
    # geometry row.  Candidate-level availability lives in the companion
    # projection, while this hourly table needs no redundant flag.
    if "common_transition_context_available" not in historical_geometry:
        historical_geometry["common_transition_context_available"] = True

    # Use only the shared outcome-label surface before source metadata is added.
    label_columns = list(historical_labels.columns)
    missing_current = sorted(set(label_columns).difference(current.columns))
    if missing_current:
        raise ValueError(f"current v4 source lacks canonical global-book label fields: {missing_current}")
    history = history.loc[:, [*label_columns, "signal_context_utc"]]
    current = current.loc[:, [*label_columns, "signal_context_utc", "source_family", "economics_tier", "policy_cost_contract", "path_frequency", "promotion_use", "mapping_provenance_role", "provenance_oof_share", "provenance_forward_oos_share"]]
    history["source_family"] = HISTORICAL_SOURCE
    history["economics_tier"] = "exact_1m_frozen_current_spread_counterfactual"
    history["policy_cost_contract"] = "historical_backcast_counterfactual_non_execution_parity"
    history["path_frequency"] = "1m"
    history["promotion_use"] = "diagnostic_only_non_oof"
    history["mapping_provenance_role"] = "historical_non_oof_backcast"
    history["provenance_oof_share"] = 0.0
    history["provenance_forward_oos_share"] = 0.0

    feature_columns = list(CANONICAL_FEATURES)
    for geometry, timestamp, name in (
        (historical_geometry, "signal_context_utc", "historical"),
        (current_geometry, "signal_context_utc", "current"),
    ):
        missing = sorted({timestamp, "common_transition_context_available", *feature_columns}.difference(geometry.columns))
        if missing:
            raise ValueError(f"{name} common geometry lacks: {missing}")
        if geometry[timestamp].duplicated().any():
            raise ValueError(f"{name} common geometry timestamp is duplicated")

    history = history.merge(historical_geometry, on="signal_context_utc", how="left", validate="many_to_one")
    current = current.merge(current_geometry, on="signal_context_utc", how="left", validate="many_to_one")
    panel = pd.concat([history, current], ignore_index=True, sort=False)
    panel["context_available"] = panel["common_transition_context_available"].fillna(False).astype(bool)
    panel.loc[~panel["context_available"], feature_columns] = np.nan
    if not panel.loc[panel["source_family"].eq(HISTORICAL_SOURCE), "context_available"].any():
        raise ValueError("historical source has no exact common geometry")
    if not panel.loc[panel["source_family"].eq("current_exact_spread_mayjul2026"), "context_available"].any():
        raise ValueError("current source has no exact common geometry")
    panel = _base_targets(panel)

    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    block_number = ((panel["cohort_anchor_utc"] - epoch) // pd.Timedelta(days=7)).astype(int)
    panel["cv_block_start_utc"] = epoch + pd.to_timedelta(block_number * 7, unit="D")
    panel["cv_group_id"] = "utc7d_" + block_number.astype(str)
    panel["nonoverlap_anchor_flag"] = (
        panel["cohort_anchor_utc"].astype("int64") // 3_600_000_000_000 % (2 * HORIZON_HOURS)
    ).eq(0)
    panel["source_domain"] = np.select(
        [panel["source_family"].eq(HISTORICAL_SOURCE), panel["source_family"].eq("current_exact_spread_mayjul2026")],
        ["historical_non_oof_backcast", "current_2026"],
        default="historical_2025_strict_oof",
    )
    panel["target_available_utc"] = pd.to_datetime(panel["target__adverse_onset_within_3h_available_utc"], utc=True)
    illegal = [column for column in feature_columns if any(token in column.lower() for token in ("target", "future", "outcome", "execution", "mfe", "mae", "exit", "realized"))]
    if illegal:
        raise ValueError(f"prohibited field entered strict common geometry: {illegal}")
    return panel.sort_values(["cohort_anchor_utc", "source_family"], kind="stable").reset_index(drop=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    historical_root, current_root, geometry_root = Path(args.historical_labels), Path(args.current_panel), Path(args.common_geometry)
    historical_manifest, historical_manifest_path = _manifest(historical_root)
    current_manifest, current_manifest_path = _manifest(current_root)
    geometry_manifest, geometry_manifest_path = _manifest(geometry_root)
    historical_path = historical_root / "global_book_transition_labels.parquet"
    current_path = current_root / "transition_research_panel.parquet"
    historical_geometry_path = geometry_root / "historical_hourly_state_geometry.parquet"
    current_geometry_path = geometry_root / "current_v4_semantic_context.parquet"
    _verify_hash(historical_path, historical_manifest.get("outputs_sha256", {}).get(historical_path.name), source="historical labels")
    _verify_hash(current_path, current_manifest.get("outputs", {}).get("panel", {}).get("sha256"), source="current panel")
    _verify_hash(historical_geometry_path, geometry_manifest.get("outputs", {}).get("historical_hourly_state_geometry", {}).get("sha256"), source="common geometry")
    _verify_hash(current_geometry_path, geometry_manifest.get("outputs", {}).get("current_v4_semantic_context", {}).get("sha256"), source="common geometry")
    panel = build_panel(
        pd.read_parquet(historical_path), pd.read_parquet(current_path),
        pd.read_parquet(historical_geometry_path), pd.read_parquet(current_geometry_path),
    )
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    panel_path, catalog_path, coverage_path = temporary / "transition_panel.parquet", temporary / "field_catalog.csv", temporary / "coverage.csv"
    panel.to_parquet(panel_path, index=False, compression="zstd")
    feature_columns = list(CANONICAL_FEATURES)
    catalog = pd.DataFrame({"column": panel.columns, "role": ["decision_time_feature" if column in feature_columns else "target_availability" if column.startswith("target__") and column.endswith("_available_utc") else "target" if column.startswith("target__") or column.startswith(("before_", "after_", "delta_")) else "metadata" for column in panel.columns]})
    catalog.to_csv(catalog_path, index=False)
    coverage = panel.groupby(["source_family", "mapping_provenance_role"], sort=True).agg(rows=("cohort_anchor_utc", "size"), context_rows=("context_available", "sum"), first_anchor=("cohort_anchor_utc", "min"), last_anchor=("cohort_anchor_utc", "max"), groups=("cv_group_id", "nunique"), active_positives=("target__active_adverse", "sum"), onset_positives=("target__adverse_onset_within_3h", "sum"), recovery_positives=("target__lifecycle_recovery_within_3h", "sum"), reversal_positives=("target__lifecycle_reversal_after_recovery_within_3h", "sum")).reset_index()
    coverage.to_csv(coverage_path, index=False)
    target_columns = [column for column in panel.columns if column.startswith("target__")]
    manifest = {
        "schema": SCHEMA, "status": "MATERIALIZED_STRICT_COMMON_H12_SOURCE_SEPARATED_PANEL",
        "rows": int(len(panel)), "feature_count": len(feature_columns), "target_count": len(target_columns),
        "feature_columns": feature_columns, "target_columns": target_columns,
        "source_rows": {str(key): int(value) for key, value in panel["source_family"].value_counts().items()},
        "contracts": {
            "geometry": "exact 90-field semantic common geometry only; signal context is anchor-1h; exact 1/3/12h lags; no asof/resample/fill",
            "labels": "exact H12 one-pooled-global-top10 before [s-12h,s) / after [s,s+12h) labels; no timestamp/side/asset quotas",
            "targets": "50/75/100-bps active/onset families plus exact-availability lifecycle onset/recovery/reversal rebuilt uniformly after pooling",
            "historical_provenance": "2022-23 is a frozen counterfactual non-OOF backcast, diagnostic-only, non-promotable and not execution-parity evidence",
            "current_provenance": "current rows retain strict_oof versus frozen_forward_oos metadata; classifier training must exclude the latter",
            "source_flags": "source_family/source_domain/economics/provenance are metadata for weighting and reporting only, never model features",
            "cross_validation": "seven-day groups are materialized for shuffled grouped CV with 36h purge; no walk-forward requirement",
        },
        "sources": {
            "historical_labels": {"path": str(historical_path), "sha256": sha256(historical_path), "manifest_sha256": sha256(historical_manifest_path)},
            "current_panel": {"path": str(current_path), "sha256": sha256(current_path), "manifest_sha256": sha256(current_manifest_path)},
            "common_geometry": {"manifest_sha256": sha256(geometry_manifest_path), "historical_sha256": sha256(historical_geometry_path), "current_sha256": sha256(current_geometry_path)},
        },
        "outputs_sha256": {panel_path.name: sha256(panel_path), catalog_path.name: sha256(catalog_path), coverage_path.name: sha256(coverage_path)},
        "promotion_eligible": False,
    }
    _write_json(temporary / "manifest.json", manifest)
    (temporary / "manifest.sha256").write_text(f"{sha256(temporary / 'manifest.json')}  manifest.json\n")
    os.replace(temporary, output)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-labels", type=Path, default=DEFAULT_HISTORICAL_LABELS)
    parser.add_argument("--current-panel", type=Path, default=DEFAULT_CURRENT_PANEL)
    parser.add_argument("--common-geometry", type=Path, default=DEFAULT_COMMON_GEOMETRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(_safe(run(parse_args())), indent=2, sort_keys=True))
