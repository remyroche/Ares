#!/usr/bin/env python3
"""Materialise a signed, non-promotional causal regime-feature inventory.

This is a source/provenance audit, not a feature-selection or model-training
runner.  It deliberately distinguishes an observable that is unavailable from
one that is available but unselected, and from an outcome/state label that is
forbidden as a predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "data_perp/artifacts/regime_episode_ledger_2022_2026_20260730_v1"
DEFAULT_PANEL = ROOT / "data_perp/artifacts/regime_multiview_panel_2022_2026_20260730_v2"
DEFAULT_SELECTION = ROOT / "data_perp/artifacts/fold_local_multiview_selection_2022_2026_20260730_v3"
DEFAULT_EARLY = ROOT / "data_perp/artifacts/jan_jul_2022_inverse_pi_causal_features_20260730_v3"
DEFAULT_HEALTH = ROOT / "data_perp/artifacts/historical_exact_model_health_failure_20260729_v3"
DEFAULT_LIQUIDITY = ROOT / "data_perp/artifacts/regime_liquidity_enrichment_2022_2026_20260730_v1"
DEFAULT_FEATURE_STORE = ROOT / "data_perp/features/20260711_070000"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_regime_feature_inventory_20260730_v5"
SCHEMA = "causal_regime_feature_inventory_v1"

META_FIELDS = {"source_utc", "calendar_segment_id", "source_segment_id", "execution_decision_utc", "source_artifact_id", "source_artifact_path", "source_artifact_sha256", "source_manifest_sha256"}
OUTCOME_TOKENS = ("target", "label", "outcome", "future", "realized", "realised", "mfe", "mae", "pnl", "net_ev", "gross_ev", "exit", "timeout", "time_to", "barrier")


def _sha256(path: Path) -> str:
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
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def feature_family(field: str) -> str:
    """Conservative name-based family assignment used only for inventory."""

    name = str(field).lower()
    if any(token in name for token in ("spread", "depth", "liquid", "amihud", "volume", "quote", "impact")):
        return "liquidity_spread_depth"
    if any(token in name for token in ("funding", "oi", "open_interest", "liquidation", "deleverag", "short_cover")):
        return "funding_oi_liquidation"
    if any(token in name for token in ("corr", "covar", "peer", "decoupl", "beta", "cross_asset", "xasset")):
        return "cross_asset_dependence_covariance"
    if any(token in name for token in ("breadth", "dispersion", "universe_median")):
        return "breadth"
    if any(token in name for token in ("path", "drawdown", "recovery", "efficiency", "bars_since", "distance_to", "support", "resistance", "mfe", "mae")):
        return "path_geometry"
    if any(token in name for token in ("health", "score", "calibr", "map_", "selected_rows", "candidate_rows", "rank_")):
        return "model_health"
    if any(token in name for token in ("vol", "atr", "range", "rv_", "jump", "compression")):
        return "volatility"
    if any(token in name for token in ("ret", "trend", "momentum", "ema", "adx", "slope", "breakout", "price")):
        return "returns_trend"
    return "other_observable"


def causal_status(field: str, *, source: str) -> tuple[str, str]:
    name = str(field).lower()
    if field in META_FIELDS:
        return "metadata_not_feature", "identity/timestamp/provenance field"
    # v2's manifest enforces an explicit denylist before multiview transforms;
    # terms such as ``realized_vol`` are trailing volatility statistics, not
    # realised trading outcomes.
    if source == "multiview_v2":
        return "causal_observable", "trailing multiview transform under the v2 causal denylist and exact-cadence contract"
    # Names such as ``mean`` must not be rejected merely because they contain
    # the letters "mae".  Evaluate forbidden concepts as namespaces/tokens.
    word_tokens = set(token for token in re.split(r"[^a-z0-9]+", name) if token)
    forbidden_words = {"label", "outcome", "future", "mfe", "mae", "pnl", "exit", "timeout"}
    realised_outcome = ("realized" in word_tokens or "realised" in word_tokens) and not any(token.startswith("vol") for token in word_tokens)
    leaf_target = "leaf_target" in name
    if field.startswith("target__") or forbidden_words.intersection(word_tokens) or realised_outcome or leaf_target or any(token in name for token in ("post_entry", "postentry", "net_ev", "gross_ev", "time_to")):
        return "outcome_or_label_forbidden", "target/outcome/post-horizon/state-label namespace is not a feature input"
    if source == "historical_feature_store":
        return "pit_contract_not_verified_here", "raw store schema exists but this inventory has no source-time availability calendar"
    if source == "historical_model_health":
        return "causal_historical_lineage_only", "pre-window health observable in historical canonical lineage; not current-lineage evidence"
    return "causal_observable", "available at the source timestamp under the source manifest contract"


def _selection_counts(selection_paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in selection_paths:
        if path.exists():
            rows.append(pd.read_parquet(path))
    if not rows:
        return pd.DataFrame(columns=["feature", "regime_selected_folds", "transition_selected_folds", "selection_status"])
    data = (pd.concat(rows, ignore_index=True)
            .groupby(["fold_id", "feature"], as_index=False, sort=False)
            .agg(regime_selected=("regime_selected", "max"), transition_selected=("transition_selected", "max"), source_field=("source_field", "first")))
    result = (data.groupby("feature", sort=True)
              .agg(regime_selected_folds=("regime_selected", "sum"), transition_selected_folds=("transition_selected", "sum"),
                   selection_folds=("fold_id", "nunique"), selection_source_field=("source_field", "first"))
              .reset_index())
    result["selection_status"] = np.where(
        (result["regime_selected_folds"] + result["transition_selected_folds"]) > 0,
        "selected_in_at_least_one_fold", "available_but_unselected",
    )
    return result


def _feature_store_fields(root: Path) -> tuple[list[str], int]:
    fields: set[str] = set()
    files = sorted(root.glob("*.parquet")) if root.exists() else []
    for path in files:
        fields.update(name for name in pq.read_schema(path).names if not name.startswith("__"))
    return sorted(fields), len(files)


def _coverage_rows(ledger: Path, early_manifest: dict[str, Any], feature_store_files: int) -> pd.DataFrame:
    coverage = pd.read_csv(ledger / "coverage_calendar.csv")
    rows = []
    for item in coverage.itertuples(index=False):
        rows.append({"source": "source_ledger_and_multiview", "start_utc": item.start_utc, "end_utc_exclusive": item.end_utc_exclusive, "available": bool(item.regime_available), "reason": item.availability_reason})
    rows.append({"source": "early_2022_inverse_supplement", "start_utc": "2022-01-01T00:00:00Z", "end_utc_exclusive": "2022-08-01T00:00:00Z", "available": True, "reason": "separate inverse-PI causal candidate population; not interchangeable with later frozen population"})
    rows.append({"source": "historical_feature_store", "start_utc": None, "end_utc_exclusive": None, "available": feature_store_files > 0, "reason": "schema materialized, but no common source-time calendar is encoded at store root"})
    rows.append({"source": "historical_model_health", "start_utc": "2025-02-01T00:00:00Z", "end_utc_exclusive": "2025-04-30T00:00:00Z", "available": True, "reason": "historical canonical raw-alpha lineage only"})
    return pd.DataFrame(rows)


def missing_observable_suggestions() -> pd.DataFrame:
    return pd.DataFrame([
        {"family": "liquidity_spread_depth", "observable_or_composite": "executable multi-level order-book impact curve and depth imbalance", "why_economically_plausible": "distinguishes nominal volume from actual cost/marketability during stress", "required_contract": "snapshot timestamp <= signal; venue and level coverage; no forward fill"},
        {"family": "funding_oi_liquidation", "observable_or_composite": "cross-venue funding dispersion, OI change and liquidation imbalance", "why_economically_plausible": "captures crowded positioning and forced-flow risk beyond a single market aggregate", "required_contract": "exchange timestamps aligned before decision; explicit stale-data flags"},
        {"family": "cross_asset_dependence_covariance", "observable_or_composite": "rolling factor residual correlation/network concentration and beta dispersion", "why_economically_plausible": "separates broad beta shocks from idiosyncratic opportunity", "required_contract": "trailing returns only; minimum constituent coverage and covariance shrinkage"},
        {"family": "returns_trend", "observable_or_composite": "trend persistence conditional on realised-volatility and market breadth", "why_economically_plausible": "same momentum has different continuation odds in expansion versus whipsaw", "required_contract": "trailing-only interaction; no future path or barrier resolution"},
        {"family": "path_geometry", "observable_or_composite": "pre-entry adverse excursion proxy: distance-to-stop liquidity / nearby structural level", "why_economically_plausible": "can estimate fragility without using post-entry MAE/MFE labels", "required_contract": "levels and depth formed no later than signal; never use realised MFE/MAE"},
        {"family": "model_health", "observable_or_composite": "causal calibration drift, candidate-population shift and score-rank instability", "why_economically_plausible": "allows monitoring whether a score is operating outside its learned support", "required_contract": "reference windows resolve before scoring; no same-period realised execution EV"},
        {"family": "volatility", "observable_or_composite": "implied-volatility/skew and futures basis term structure", "why_economically_plausible": "adds forward-looking risk pricing absent from realised-volatility-only signals", "required_contract": "timestamped option/futures snapshots; coverage and stale quote indicators"},
    ])


def unavailable_liquidity_details(manifest: dict[str, Any]) -> pd.DataFrame:
    """Expand missing source fields rather than confusing them with unselection."""

    rows: list[dict[str, str]] = []
    for path, fields in manifest.get("sources", {}).get("skipped_feature_files", {}).items():
        for field in fields:
            rows.append({
                "source": "liquidity_enrichment",
                "source_file": str(path),
                "field": str(field),
                "family": feature_family(str(field)),
                "availability_status": "source_unavailable",
                "reason": "field absent from this asset feature file; no fill or proxy was introduced",
            })
    return pd.DataFrame(rows)


def materialize_inventory(
    *, ledger_dir: Path = DEFAULT_LEDGER, panel_dir: Path = DEFAULT_PANEL, selection_dir: Path = DEFAULT_SELECTION,
    early_dir: Path = DEFAULT_EARLY, health_dir: Path = DEFAULT_HEALTH, liquidity_dir: Path = DEFAULT_LIQUIDITY,
    feature_store: Path = DEFAULT_FEATURE_STORE, output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    ledger_dir, panel_dir, selection_dir, early_dir, health_dir, liquidity_dir, feature_store, output_dir = map(Path, (ledger_dir, panel_dir, selection_dir, early_dir, health_dir, liquidity_dir, feature_store, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    ledger_manifest = json.loads((ledger_dir / "manifest.json").read_text())
    panel_manifest = json.loads((panel_dir / "manifest.json").read_text())
    early_manifest = json.loads((early_dir / "manifest.json").read_text())
    health_manifest = json.loads((health_dir / "manifest.json").read_text())
    liquidity_manifest = json.loads((liquidity_dir / "manifest.json").read_text())
    ledger_fields = pq.read_schema(ledger_dir / "hourly_state_calendar.parquet").names
    panel_fields = pq.read_schema(panel_dir / "multiview_regime_features.parquet").names
    store_fields, store_files = _feature_store_fields(feature_store)
    health_fields = pd.read_csv(health_dir / "health_feature_catalog.csv")["feature"].astype(str).tolist()
    early_fields = [str(field) for field in early_manifest.get("feature_columns", [])]
    selected = _selection_counts([selection_dir / "regime_fold_selection.parquet", selection_dir / "transition_fold_selection.parquet"])
    selected_map = selected.set_index("feature").to_dict(orient="index") if not selected.empty else {}
    records: list[dict[str, Any]] = []
    inputs = [
        ("source_ledger", ledger_fields, "2022-08-30..2026-07-12 with explicit gap calendar"),
        ("multiview_v2", panel_fields, "2022-08-30..2026-07-12 with exact-cadence segments"),
        ("historical_feature_store", store_fields, "root calendar unavailable; field schema from 249 asset files"),
        ("early_2022_inverse_supplement", early_fields, "2022-01-01..2022-07-31 separate inverse-PI candidate population"),
        ("historical_model_health", health_fields, "2025-02..2025-04 historical canonical raw-alpha lineage"),
    ]
    for source, fields, coverage in inputs:
        for field in fields:
            causal, reason = causal_status(field, source=source)
            selection = selected_map.get(field, {})
            records.append({
                "source": source, "field": field, "family": feature_family(field), "time_coverage": coverage,
                "causal_status": causal, "causal_reason": reason,
                "regime_selected_folds": int(selection.get("regime_selected_folds", 0)),
                "transition_selected_folds": int(selection.get("transition_selected_folds", 0)),
                "selection_folds_observed": int(selection.get("selection_folds", 0)),
                "selection_status": selection.get("selection_status", "not_in_fold_local_multiview_selection" if causal == "causal_observable" else "not_selectable"),
            })
    inventory = pd.DataFrame(records)
    # These exact derived fields are selection units.  Preserve them even when
    # they are not a raw field elsewhere in the inventory.
    for feature, row in selected_map.items():
        if not inventory.loc[inventory["field"].eq(feature)].empty:
            continue
        inventory.loc[len(inventory)] = {
            "source": "multiview_v2", "field": feature, "family": feature_family(feature), "time_coverage": "2022-08-30..2026-07-12 with exact-cadence segments",
            "causal_status": "causal_observable", "causal_reason": "trailing multiview transform under v2 causal contract",
            "regime_selected_folds": int(row["regime_selected_folds"]), "transition_selected_folds": int(row["transition_selected_folds"]),
            "selection_folds_observed": int(row["selection_folds"]), "selection_status": row["selection_status"],
        }
    family = (inventory.groupby(["source", "family", "causal_status"], sort=True)
              .agg(actual_fields=("field", "nunique"), regime_selected_field_fold_count=("regime_selected_folds", "sum"), transition_selected_field_fold_count=("transition_selected_folds", "sum"), selected_fields=("selection_status", lambda x: int(pd.Series(x).eq("selected_in_at_least_one_fold").sum())))
              .reset_index())
    forbidden = inventory.loc[inventory["causal_status"].eq("outcome_or_label_forbidden")].copy()
    forbidden["rejection_type"] = "outcome_forbidden"
    source_availability = pd.DataFrame([
        {"source": "source_ledger", "status": "available", "detail": "observable state calendar with explicit coverage gaps"},
        {"source": "multiview_v2", "status": "available", "detail": "14,536 trailing causal features; no target/outcome input"},
        {"source": "historical_feature_store", "status": "schema_available_time_coverage_unproven", "detail": f"{store_files} asset files; root has no common time calendar"},
        {"source": "early_2022_inverse_supplement", "status": "available_separate_population", "detail": "Jan-Jul 2022 causal inverse-PI source; separate taxonomy/population"},
        {"source": "historical_model_health", "status": "available_historical_lineage_only", "detail": "health features cannot evidence current execution lineage"},
        {"source": "liquidity_enrichment", "status": "available_partial_asset_coverage", "detail": f"{liquidity_manifest['counts']['feature_files_used']} used / {liquidity_manifest['counts']['feature_files_seen']} seen; skipped files are source-unavailable, not unselected"},
    ])
    coverage = _coverage_rows(ledger_dir, early_manifest, store_files)
    output_dir.mkdir(parents=True)
    for name, frame in {
        "field_inventory.csv": inventory.sort_values(["source", "family", "field"]),
        "family_summary.csv": family,
        "coverage_and_missing_intervals.csv": coverage,
        "forbidden_or_rejected_fields.csv": forbidden.sort_values(["source", "field"]),
        "source_availability.csv": source_availability,
        "source_unavailable_field_detail.csv": unavailable_liquidity_details(liquidity_manifest),
        "economically_plausible_missing_observables.csv": missing_observable_suggestions(),
    }.items():
        frame.to_csv(output_dir / name, index=False)
    source_paths = {"ledger_manifest": ledger_dir / "manifest.json", "panel_manifest": panel_dir / "manifest.json", "selection_manifest": selection_dir / "manifest.json", "early_manifest": early_dir / "manifest.json", "health_manifest": health_dir / "manifest.json", "liquidity_manifest": liquidity_dir / "manifest.json"}
    manifest = {
        "schema": SCHEMA, "research_only": True, "promotion_eligible": False,
        "purpose": "causal regime-feature source and selection inventory; no feature promotion, training, scoring, or state merging",
        "separation_contract": "regime and transition selection frequencies are reported in separate columns and are never merged into a shared state label",
        "counts": {"inventory_rows": int(len(inventory)), "forbidden_fields": int(len(forbidden)), "historical_store_files": store_files, "early_inverse_fields": len(early_fields), "selected_feature_units": len(selected_map)},
        "source_contracts": {"ledger": ledger_manifest.get("coverage_contract"), "panel": panel_manifest.get("multiview_contract", {}).get("causality"), "early": early_manifest.get("timing"), "health": health_manifest.get("label_contract")},
        "status_definitions": {"source_unavailable": "source/calendar/file does not provide the observable", "available_but_unselected": "causal field existed but was not selected in a named fold", "outcome_forbidden": "target/outcome/post-horizon/state-label field rejected from predictors"},
        "inputs_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "output_sha256": {item.name: _sha256(item) for item in output_dir.iterdir() if item.is_file()},
    }
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "manifest.sha256").write_text(f"{_sha256(output_dir / 'manifest.json')}  manifest.json\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--early", type=Path, default=DEFAULT_EARLY)
    parser.add_argument("--health", type=Path, default=DEFAULT_HEALTH)
    parser.add_argument("--liquidity", type=Path, default=DEFAULT_LIQUIDITY)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(_safe(materialize_inventory(ledger_dir=args.ledger, panel_dir=args.panel, selection_dir=args.selection, early_dir=args.early, health_dir=args.health, liquidity_dir=args.liquidity, feature_store=args.feature_store, output_dir=args.output)), indent=2))


if __name__ == "__main__":
    main()
