#!/usr/bin/env python3
"""Materialise causal Stage-C OHLCV continuation features on frozen v11 IDs.

This script intentionally blocks new OI/funding groups: the archived sources
do not carry an observed/publication timestamp, so pretending their hourly
values were known at a decision would violate the Stage-C contract.  The
resulting immutable population is suitable for C0--C8 comparisons; unavailable
groups are explicitly recorded rather than silently omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.continuation_features import (
    CONTINUATION_FEATURE_GROUPS,
    CONTINUATION_SIDE_PRICE_FEATURE_KEYS,
    CONTINUATION_SIDE_VOLATILITY_FEATURE_KEYS,
    materialize_ohlcv_continuation_features,
    side_normalize_continuation_features,
)
from scripts.materialize_historical_exact_h12_alignment_sidecar import COST_MODEL_ID, EXECUTION_POLICY_ID, TARGET_ID

PANEL = ROOT / "data_perp/artifacts/long_exact_h12_raw_base_panel_20260730_v2/raw_base_panel.parquet"
ALIGNMENT = ROOT / "data_perp/artifacts/historical_exact_h12_alignment_sidecar_research_only_20260731_v1/alignment_sidecar.parquet"
PERSISTENCE = ROOT / "data_perp/artifacts/historical_exact_h12_postcost_persistence_labels_20260731_v1/postcost_persistence_labels.parquet"
OHLCV_ROOT = ROOT / "data_perp/exchanges/krakenfutures/ohlcv"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/stage_c_continuation_feature_panel_20260731_v2"
E15_FEATURE_MANIFEST = ROOT / "data_perp/artifacts/exact_h12_target_purity_ablation_20260731_v11/selected_execution_features.json"
OI_FUNDING_BLOCK_REASON = (
    "rejected: archived hourly OI/funding sidecars have only a nominal ts/index, "
    "not semantically declared source_event_ts, source_observed_ts, available_ts, "
    "ingested_ts, provider/product/revision or payload lineage; their ingestion "
    "path applies unbounded ffill (data_store.py:3042-3043), so an index timestamp "
    "or assumed delay cannot prove point-in-time availability or bounded staleness"
)

# Exact inherited candidates are recorded even where the Stage-C definition is
# deliberately different (for example a shorter trailing horizon).  This makes
# the add-on/redundancy decision auditable rather than relying on a name scan.
REUSE_CANDIDATES: dict[str, tuple[str | None, str, str]] = {
    "cont_efficiency_12h": ("efficiency_ratio_20;path_efficiency_24", "different_horizon_redundant_candidate", "12h log-return path efficiency is retained only as a compact shorter-horizon check"),
    "cont_volume_z_48h": ("volume_zscore_48h", "different_definition_redundant_candidate", "same 48h concept; Stage-C uses raw completed-bar z-score"),
    "cont_volume_price_corr_12h": ("volume_price_corr_ts_resid", "different_transform_redundant_candidate", "raw 12h correlation versus inherited time-series residual"),
    "cont_asset_minus_market_ret_4h": ("asset_minus_universe_median_ret_4h;symbol_minus_mkt_ret_4h", "different_definition_redundant_candidate", "eligible-universe median residual is recomputed directly"),
    "cont_market_ret_breadth_4h": ("market_breadth_4h", "different_definition_redundant_candidate", "fraction positive in the timestamp-eligible Stage-C universe"),
    "cont_market_ret_dispersion_4h": ("market_dispersion_4h", "different_definition_redundant_candidate", "timestamp-eligible raw-return dispersion"),
    "cont_range_expansion_12h": ("range_expansion_ratio", "different_horizon_redundant_candidate", "current bar range divided by trailing 12h range"),
    "cont_rv_12h": ("rv_24h;mkt_rv_4h", "different_horizon_redundant_candidate", "asset realised volatility, not the inherited 24h/market field"),
    "cont_vol_of_vol_12h": ("volatility_of_volatility_48", "different_horizon_redundant_candidate", "12h trailing return-absolute-deviation volatility"),
}


def _feature_details(name: str, group: str) -> tuple[str, str, int, str, str, str]:
    """Return exact-enough per-field lineage, not a group-wide placeholder."""
    details = {
        "cont_ret_1h": ("log(close_t / close_t-1h)", "1h", 2, "log return", "unbounded", "none"),
        "cont_ret_4h": ("log(close_t / close_t-4h)", "4h", 5, "log return", "unbounded", "none"),
        "cont_ret_12h": ("log(close_t / close_t-12h)", "12h", 13, "log return", "unbounded", "none"),
        "cont_return_acceleration_1h_4h": ("ret_1h - ret_4h / 4", "4h", 5, "log-return acceleration", "unbounded", "side counterpart"),
        "cont_efficiency_12h": ("abs(ret_12h) / sum(abs(ret_1h), trailing 12h)", "12h", 13, "ratio", "[0,1]", "none"),
        "cont_directional_consistency_12h": ("abs(2 * fraction(ret_1h > 0) - 1)", "12h", 12, "ratio", "[0,1]", "none"),
        "cont_slope_12h": ("trailing OLS slope of log close against bar position", "12h", 12, "log-price/bar", "unbounded", "side counterpart"),
        "cont_slope_r2_12h": ("trailing OLS R² of log close", "12h", 12, "ratio", "[0,1]", "none"),
        "cont_distance_from_high_12h": ("(close - trailing max(high)) / close", "12h", 12, "ratio", "[-inf,0]", "side interpreted"),
        "cont_distance_from_low_12h": ("(close - trailing min(low)) / close", "12h", 12, "ratio", "[0,inf]", "side interpreted"),
        "cont_high_recency_12h": ("bars since a trailing-12h high", "12h", 12, "bars", "[0,11]", "side interpreted"),
        "cont_low_recency_12h": ("bars since a trailing-12h low", "12h", 12, "bars", "[0,11]", "side interpreted"),
        "cont_direction_changes_12h": ("sum of sign(ret_1h) changes", "12h", 12, "count", "[0,11]", "none"),
        "cont_close_location_ohlcv_proxy": ("(close-low)/(high-low)", "bar", 1, "ratio", "[0,1]", "side counterpart via wick"),
        "cont_side_wick_imbalance_raw": ("(lower_wick-upper_wick)/(high-low)", "bar", 1, "ratio", "[-1,1]", "side counterpart"),
        "cont_range_expansion_12h": ("current(high-low) / trailing mean(high-low)", "12h", 3, "ratio", "[0,inf]", "none"),
        "cont_failed_up_breakout_count_12h": ("sum(high > prior trailing high and close < prior high)", "12h", 12, "count", "[0,12]", "long counterpart"),
        "cont_failed_down_breakout_count_12h": ("sum(low < prior trailing low and close > prior low)", "12h", 12, "count", "[0,12]", "short counterpart"),
        "cont_up_breakout_rejection_12h": ("failed up breakout and close location < .35", "12h", 12, "indicator", "{0,1}", "long counterpart"),
        "cont_down_breakout_rejection_12h": ("failed down breakout and close location > .65", "12h", 12, "indicator", "{0,1}", "short counterpart"),
        "cont_volume_z_48h": ("(volume-trailing mean)/trailing std", "48h", 3, "z-score", "unbounded", "none"),
        "cont_volume_persistence_12h": ("trailing mean(volume,12h)/trailing mean(volume,48h)", "48h", 3, "ratio", "[0,inf]", "none"),
        "cont_signed_volume_proxy_12h": ("sum(sign(close-open)*volume)", "12h", 12, "OHLCV proxy", "unbounded", "side interpreted"),
        "cont_range_to_volume_ohlcv_proxy": ("((high-low)/close)/volume", "bar", 1, "OHLCV proxy", "[0,inf]", "none"),
        "cont_high_volume_low_return_churn_12h": ("mean(max(volume_z,0)*(1-abs(close/open-1)))", "12h", 12, "OHLCV proxy", "[0,inf]", "none"),
        "cont_volume_price_corr_12h": ("trailing corr(volume, ret_1h)", "12h", 6, "correlation", "[-1,1]", "none"),
        "cont_volume_concentration_4h_12h": ("sum(volume,4h)/sum(volume,12h)", "12h", 3, "ratio", "[0,1]", "none"),
        "cont_volume_shock_age_hrs": ("bars since volume > mean_48h + 2*std_48h", "48h", 3, "hours", "[0,inf]", "none"),
        "cont_volume_shock_decay_12h": ("exp(-volume_shock_age/12)", "48h", 3, "ratio", "(0,1]", "none"),
        "cont_rv_12h": ("sqrt(mean(ret_1h²))", "12h", 3, "log-return volatility", "[0,inf]", "none"),
        "cont_downside_rv_12h": ("sqrt(mean(min(ret_1h,0)²))", "12h", 3, "log-return volatility", "[0,inf]", "side-adverse counterpart"),
        "cont_vol_ratio_4h_12h": ("rv_4h / rv_12h", "12h", 3, "ratio", "[0,inf]", "none"),
        "cont_vol_of_vol_12h": ("std(abs(ret_1h))", "12h", 3, "volatility", "[0,inf]", "none"),
        "cont_range_z_48h": ("(range-mean_48h)/std_48h", "48h", 3, "z-score", "unbounded", "none"),
        "cont_squared_return_autocorr_12h": ("trailing corr(ret_1h², lag(ret_1h²))", "12h", 6, "correlation", "[-1,1]", "none"),
        "cont_atr_slope_12h": ("mean_range_12h - trailing mean(mean_range_12h)", "24h", 12, "price range", "unbounded", "none"),
        "cont_atr_acceleration_12h": ("atr_slope_12h - lag_4h(atr_slope_12h)", "28h", 16, "price range", "unbounded", "none"),
        "cont_vol_shock_age_hrs": ("bars since range_z_48h > 2", "48h", 3, "hours", "[0,inf]", "none"),
        "cont_vol_shock_decay_12h": ("exp(-vol_shock_age/12)", "48h", 3, "ratio", "(0,1]", "none"),
        "cont_vol_climax_persistence_12h": ("mean(range_z_48h > 1.5)", "48h", 3, "ratio", "[0,1]", "none"),
    }
    if name in details:
        return details[name]
    if name.startswith("side_cont_"):
        return (f"candidate side-sign applied to {name.removeprefix('side_')}", "same as raw counterpart", 1, "side-normalised feature", "varies", "candidate side sign")
    if name.startswith("cont_cs_") or name.startswith("cont_market_") or name.startswith("cont_asset_"):
        return (f"timestamp-eligible cross-sectional {name.removeprefix('cont_')}", "timestamp snapshot", 1, "rank/ratio/return", "varies", "side interpreted where directional")
    if name.startswith("cont_") and "_x_" in name:
        return (f"predeclared product of named causal components: {name}", "component lookbacks", 1, "composite", "varies", "no fitted weights")
    return (f"deterministic causal transformation named {name}", "encoded by field name", 1, "float32", "varies", "none")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def _source_files(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    directory = OHLCV_ROOT / f"symbol={symbol}"
    if not directory.exists():
        return []
    files: list[Path] = []
    for path in directory.rglob("*.parquet"):
        years = [part.removeprefix("year=") for part in path.parts if part.startswith("year=")]
        if years and start.year <= int(years[-1]) <= end.year:
            files.append(path)
    return sorted(files)


def _read_bars(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only source bars potentially needed, retaining source identity."""
    bars, lineage = [], []
    for symbol in symbols:
        files = _source_files(symbol, start, end)
        if not files:
            lineage.append({"symbol": symbol, "source": "kraken_linear_perpetual_ohlcv", "available": False, "reason": "source_directory_absent"})
            continue
        parts = []
        for path in files:
            source = pd.read_parquet(path, columns=["ts", "open", "high", "low", "close", "volume"])
            source["ts"] = pd.to_datetime(source.ts, utc=True, errors="raise")
            source = source.loc[source.ts.ge(start) & source.ts.le(end)].copy()
            if len(source):
                parts.append(source)
            lineage.append({"symbol": symbol, "source": "kraken_linear_perpetual_ohlcv", "source_path": str(path), "available": bool(len(source)), "source_frequency": "1h", "source_publication_delay": "bar close / decision uses feature_cutoff_ts", "point_in_time_safe": True, "live_reproducible": True, "proxy_or_factual": "factual_ohlcv"})
        if parts:
            joined = pd.concat(parts, ignore_index=True).drop_duplicates("ts", keep="last")
            joined["symbol"] = symbol
            bars.append(joined)
    return (pd.concat(bars, ignore_index=True) if bars else pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume"]), pd.DataFrame(lineage))


def _lineage_records(*, inherited_columns: set[str], e15_features: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Complete Stage-0 lineage/reuse map; values never imply model admission."""
    rows: list[dict[str, Any]] = []
    f0 = sorted({name for values in e15_features.values() for name in values})
    for name in f0:
        rows.append({
            "feature_name": name, "feature_group": "F0_existing_E15_control", "source": "frozen v11 E15 selected_execution_features.json",
            "source_frequency": "candidate decision row", "lookback": "frozen inherited contract", "minimum_observations": None,
            "formula": "exact persisted E15 per-side control feature", "units_range": "inherited", "side_normalisation": "as frozen by v11",
            "feature_available_ts": "frozen v11 decision-time contract", "source_publication_delay": "inherited contract",
            "missingness_rule": "frozen v11 transform", "staleness_rule": "frozen v11 transform", "point_in_time_safe": True,
            "live_reproducible": True, "proxy_or_factual": "inherited", "reuse_status": "existing_control",
            "existing_380_field": name if name in inherited_columns else None,
            "lineage_note": "F0 is persisted, not recomputed, so the comparison control is byte-stable.",
        })
    materialized_groups = {
        **CONTINUATION_FEATURE_GROUPS,
        "F1_price_continuation_exhaustion": [*CONTINUATION_FEATURE_GROUPS["F1_price_continuation_exhaustion"], *CONTINUATION_SIDE_PRICE_FEATURE_KEYS],
        "F3_volatility_transition": [*CONTINUATION_FEATURE_GROUPS["F3_volatility_transition"], *CONTINUATION_SIDE_VOLATILITY_FEATURE_KEYS],
    }
    for group, names in materialized_groups.items():
        for name in names:
            formula, lookback, minimum_observations, units_range, expected_range, side_normalisation = _feature_details(name, group)
            existing, reuse_status, lineage_note = REUSE_CANDIDATES.get(name, (None, "new_compact_transform", "no exact inherited 380-field definition; Stage-C compact addition"))
            rows.append({
                "feature_name": name, "feature_group": group, "source": "Kraken linear perpetual completed OHLCV",
                "source_frequency": "1h", "lookback": lookback, "minimum_observations": minimum_observations,
                "formula": formula, "units_range": f"{units_range}; expected {expected_range}",
                "side_normalisation": side_normalisation, "feature_available_ts": "completed feature_cutoff bar close",
                "source_publication_delay": "bar close", "missingness_rule": "leave missing; no backfill", "staleness_rule": "exact source bar at cutoff",
                "point_in_time_safe": True, "live_reproducible": True, "proxy_or_factual": "ohlcv_proxy" if "proxy" in name else "factual_ohlcv_transform",
                "reuse_status": reuse_status, "existing_380_field": existing,
                "lineage_note": lineage_note,
            })
    for group in ("F4_oi_dynamics", "F5_funding_crowding"):
        rows.append({
            "feature_name": f"{group}__BLOCKED", "feature_group": group, "source": "archived Kraken hourly sidecar",
            "source_frequency": "nominal hourly; availability unproven", "lookback": None, "minimum_observations": None,
            "formula": "not materialised", "units_range": None, "side_normalisation": "not applicable", "feature_available_ts": None,
            "source_publication_delay": "unproven; no inferred lag accepted", "missingness_rule": "rejected", "staleness_rule": "rejected",
            "point_in_time_safe": False, "live_reproducible": False, "proxy_or_factual": "blocked_unverified_source_timing",
            "reuse_status": "rejected_source_timing", "existing_380_field": None, "lineage_note": OI_FUNDING_BLOCK_REASON,
        })
    rows.append({
        "feature_name": "F7_causal_regime_transition__BLOCKED", "feature_group": "F7_causal_regime_transition", "source": "existing regime/transition outputs",
        "source_frequency": "varies", "lookback": None, "minimum_observations": None, "formula": "not materialised", "units_range": None,
        "side_normalisation": "not applicable", "feature_available_ts": None, "source_publication_delay": "unproven for raw panel fields",
        "missingness_rule": "rejected", "staleness_rule": "requires candidate OOF/prequential provenance", "point_in_time_safe": False,
        "live_reproducible": False, "proxy_or_factual": "blocked_unproven_oof", "reuse_status": "rejected_unproven_oof",
        "existing_380_field": None, "lineage_note": "Only an explicit candidate-ID OOF/prequential sidecar with fold/train-end/available timestamps and hashes may populate F7.",
    })
    return rows


def run(*, output: Path, smoke: bool = False) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    alignment = pd.read_parquet(ALIGNMENT)
    labels = pd.read_parquet(PERSISTENCE)
    panel = pd.read_parquet(PANEL, columns=["candidate_id"])
    inherited_columns = set(pq.read_schema(PANEL).names)
    e15_features = json.loads(E15_FEATURE_MANIFEST.read_text(encoding="utf-8"))
    e15_by_side = {side: list(e15_features[side]) for side in ("long", "short")}
    required = {"candidate_id", "symbol", "side", "decision_ts", "feature_cutoff_ts", "label_end_ts", "label_available_ts", "target_id", "execution_policy_id", "cost_model_id", "exact_h12_net_bps"}
    if required.difference(alignment.columns) or len(alignment) != len(panel):
        raise ValueError("frozen alignment/panel contract is incomplete")
    if alignment.target_id.nunique() != 1 or alignment.target_id.iloc[0] != TARGET_ID or alignment.execution_policy_id.iloc[0] != EXECUTION_POLICY_ID or alignment.cost_model_id.iloc[0] != COST_MODEL_ID:
        raise ValueError("frozen target/cost/policy IDs are incompatible")
    candidates = alignment.merge(panel, on="candidate_id", how="inner", validate="one_to_one")
    labels = labels.loc[:, [
        "candidate_id", "postcost_h0_clear_first", "postcost_h0_persistence_target_valid",
        "postcost_h0_retained_net", "postcost_h25_clear_first",
        "postcost_h25_persistence_target_valid", "postcost_h25_retained_net",
    ]]
    candidates = candidates.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    candidates["decision_ts"] = pd.to_datetime(candidates.decision_ts, utc=True, errors="raise")
    candidates["feature_cutoff_ts"] = pd.to_datetime(candidates.feature_cutoff_ts, utc=True, errors="raise")
    candidates["label_available_ts"] = pd.to_datetime(candidates.label_available_ts, utc=True, errors="raise")
    candidates = candidates.sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
    if smoke:
        candidates = candidates.loc[candidates.decision_ts.dt.month.isin([4, 5])].groupby([candidates.decision_ts.dt.to_period("M"), "side"], group_keys=False).head(800).reset_index(drop=True)
    # The exact source symbol is the frozen linear PF symbol.  Add 48 hours of
    # warm-up, yet only join a feature with an exact source bar at feature cutoff.
    start = candidates.feature_cutoff_ts.min() - pd.Timedelta(hours=48)
    end = candidates.feature_cutoff_ts.max()
    # Frozen candidate symbols use CCXT slash notation; the authoritative
    # Kraken archive uses the same linear contract with ``/`` encoded as ``_``.
    candidates["source_symbol"] = candidates.symbol.str.replace("/", "_", regex=False)
    bars, source_ledger = _read_bars(sorted(candidates.source_symbol.unique()), start, end)
    features = materialize_ohlcv_continuation_features(bars)
    eligible_universe = (features.loc[:, ["ts", "symbol"]].drop_duplicates().groupby("ts", sort=True)["symbol"]
                         .agg(lambda values: hashlib.sha256("\n".join(sorted(values.astype(str))).encode("utf-8")).hexdigest())
                         .rename("eligible_symbol_sha256").reset_index())
    eligible_universe["eligible_universe_size"] = features.groupby("ts", observed=True)["symbol"].nunique().reindex(eligible_universe.ts).to_numpy()
    features = features.rename(columns={"symbol": "source_symbol", "ts": "feature_source_ts"})
    # Side fields must be calculated over the completed-bar history, before
    # candidate-side filtering.  Rolling over candidate rows would turn a 12h
    # window into an irregular sequence and could mix side histories.
    side_input_columns = [
        "source_symbol", "feature_source_ts", "cont_ret_1h", "cont_ret_4h", "cont_ret_12h",
        "cont_return_acceleration_1h_4h", "cont_slope_12h", "cont_asset_minus_market_ret_4h",
        "cont_side_wick_imbalance_raw", "cont_up_breakout_rejection_12h", "cont_down_breakout_rejection_12h",
    ]
    side_feature_parts = []
    for candidate_side in ("long", "short"):
        side_panel = side_normalize_continuation_features(
            features.loc[:, side_input_columns].assign(side=candidate_side),
            (candidate_side,),
            side_column="side",
        )
        side_feature_parts.append(side_panel.loc[:, ["source_symbol", "feature_source_ts", "side", *CONTINUATION_SIDE_PRICE_FEATURE_KEYS, *CONTINUATION_SIDE_VOLATILITY_FEATURE_KEYS]])
    side_features = pd.concat(side_feature_parts, ignore_index=True)
    joined = candidates.merge(features, left_on=["source_symbol", "feature_cutoff_ts"], right_on=["source_symbol", "feature_source_ts"], how="left", validate="many_to_one")
    joined = joined.merge(side_features, on=["source_symbol", "feature_source_ts", "side"], how="left", validate="many_to_one")
    group_feature_sets = {
        "F1_price_continuation_exhaustion": [*CONTINUATION_FEATURE_GROUPS["F1_price_continuation_exhaustion"], *CONTINUATION_SIDE_PRICE_FEATURE_KEYS],
        "F2_volume_liquidity_proxies": CONTINUATION_FEATURE_GROUPS["F2_volume_liquidity_proxies"],
        "F3_volatility_transition": [*CONTINUATION_FEATURE_GROUPS["F3_volatility_transition"], *CONTINUATION_SIDE_VOLATILITY_FEATURE_KEYS],
        "F6_cross_sectional_confirmation": CONTINUATION_FEATURE_GROUPS["F6_cross_sectional_confirmation"],
        "F8_predeclared_composites": CONTINUATION_FEATURE_GROUPS["F8_predeclared_composites"],
    }
    engineered = [name for names in group_feature_sets.values() for name in names]
    engineered = list(dict.fromkeys(engineered))
    joined["feature_available_ts"] = joined["feature_cutoff_ts"]
    joined["retain_h0_given_clear__condition_met"] = joined.postcost_h0_clear_first.astype(bool)
    joined["retain_h0_given_clear__valid"] = joined.postcost_h0_persistence_target_valid.astype(bool)
    joined["retain_h0_given_clear"] = np.where(joined["retain_h0_given_clear__valid"], joined.postcost_h0_retained_net.astype(float), np.nan)
    joined["retain_h0_given_clear__support_side"] = np.where(joined["retain_h0_given_clear__valid"], joined.side, None)
    joined["retain_h0_given_clear__support_month"] = np.where(joined["retain_h0_given_clear__valid"], joined.decision_ts.dt.strftime("%Y-%m"), None)
    joined["retain_h25_given_clear"] = np.where(
        joined.postcost_h25_persistence_target_valid.astype(bool),
        joined.postcost_h25_retained_net.astype(float),
        np.nan,
    )
    joined["continuous_net_given_clear"] = np.where(joined.postcost_h0_clear_first.astype(bool), joined.exact_h12_net_bps.astype(float), np.nan)
    if not joined.feature_available_ts.le(joined.decision_ts).all():
        raise AssertionError("a Stage-C feature is later than its decision timestamp")
    group_validity = pd.DataFrame({"candidate_id": joined.candidate_id})
    group_exclusion_parts: list[pd.DataFrame] = []
    for group, fields in group_feature_sets.items():
        valid = joined[fields].notna().all(axis=1)
        group_validity[group] = valid.to_numpy(bool)
        invalid = joined.loc[~valid, ["candidate_id", "side", "source_symbol", "decision_ts"]].copy()
        invalid["month"] = invalid.decision_ts.dt.strftime("%Y-%m")
        invalid["feature_group"] = group
        invalid["reason"] = "one_or_more_required_group_features_missing"
        group_exclusion_parts.append(invalid.drop(columns="decision_ts"))
    available = group_validity.drop(columns="candidate_id").all(axis=1)
    joined["stage_c_compatible"] = available
    group_exclusions = pd.concat(group_exclusion_parts, ignore_index=True)
    exclusion = group_exclusions.groupby(["feature_group", "month", "side", "source_symbol", "reason"], dropna=False).size().rename("excluded_rows").reset_index()
    # Persist only the immutable identity/label contract and Stage-C fields.
    # The frozen 380-column matrix is rejoined by candidate_id by the runner;
    # retaining it here would duplicate a large panel and needlessly inflate
    # the materialisation peak.
    essential = [
        "candidate_id", "source_symbol", "side", "decision_ts", "feature_cutoff_ts", "label_end_ts",
        "label_available_ts", "target_id", "execution_policy_id", "cost_model_id", "feature_available_ts", "retain_h0_given_clear",
        "retain_h0_given_clear__valid", "retain_h0_given_clear__condition_met",
        "retain_h0_given_clear__support_side", "retain_h0_given_clear__support_month",
        "retain_h25_given_clear", "continuous_net_given_clear",
        *engineered,
    ]
    population = joined.loc[available, essential].copy().sort_values(["decision_ts", "candidate_id"], kind="stable")
    compatible_ids = population.candidate_id.astype(str).tolist()
    compatible_id_sha256 = hashlib.sha256("\n".join(compatible_ids).encode("utf-8")).hexdigest()
    group_hashes = {
        group: hashlib.sha256("\n".join(joined.loc[group_validity[group], "candidate_id"].astype(str).sort_values()).encode("utf-8")).hexdigest()
        for group in group_feature_sets
    }
    coverage_parts = []
    grouping = [population.decision_ts.dt.strftime("%Y-%m").rename("month"), population.side, population.source_symbol]
    for name in engineered:
        part = population.groupby(grouping, dropna=False)[name].agg(rows="size", non_missing="count").reset_index()
        part["feature_name"] = name
        part["missing_rate"] = 1.0 - part.non_missing / part.rows.clip(lower=1)
        coverage_parts.append(part)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
    try:
        population.to_parquet(stage / "stage_c_candidate_population.parquet", index=False, compression="zstd")
        population.loc[:, ["candidate_id", "side", "decision_ts", "feature_cutoff_ts", "feature_available_ts", *engineered]].to_parquet(stage / "stage_c_features.parquet", index=False, compression="zstd")
        pd.DataFrame(_lineage_records(inherited_columns=inherited_columns, e15_features=e15_by_side)).to_parquet(stage / "feature_source_lineage.parquet", index=False, compression="zstd")
        coverage.to_parquet(stage / "feature_coverage_by_month_side_symbol.parquet", index=False, compression="zstd")
        exclusion.to_parquet(stage / "feature_exclusions_by_month_side_symbol.parquet", index=False, compression="zstd")
        group_validity.to_parquet(stage / "feature_group_validity.parquet", index=False, compression="zstd")
        pd.DataFrame({"candidate_id": compatible_ids}).to_parquet(stage / "stage_c_compatible_candidate_ids.parquet", index=False, compression="zstd")
        source_ledger.to_parquet(stage / "ohlcv_source_ledger.parquet", index=False, compression="zstd")
        report = ["# Stage-C feature availability", "", f"Compatible immutable candidate rows: **{len(population):,}** / {len(joined):,}.", f"Immutable compatible candidate-ID SHA256: `{compatible_id_sha256}`.", f"OHLCV source range: {bars.ts.min()} to {bars.ts.max()} UTC across {bars.symbol.nunique():,} eligible linear-perpetual symbols.", f"Timestamp universe: exact completed OHLCV rows from the frozen candidate-symbol universe; membership hashes are in `eligible_universe_membership.parquet`.", "", "- F1/F2/F3/F6/F8 are trailing vectorised OHLCV transforms; all liquidity-like fields are explicitly `*_proxy`.", f"- F4 OI and F5 funding are blocked: {OI_FUNDING_BLOCK_REASON}", "- F7 is blocked: learned transition values require an explicit strict OOF/prequential candidate sidecar, not fuzzy raw-panel columns.", "- No inverse PI rows are accepted. No OI/funding value, payment, or next-settlement field is used.", "- A feature joins only to an exact completed source bar at `feature_cutoff_ts`; it is therefore available by the decision timestamp.", "", "`feature_source_lineage.parquet` is the reuse map and full feature dictionary. `feature_coverage_by_month_side_symbol.parquet` reports missingness per feature; `feature_exclusions_by_month_side_symbol.parquet` reports every group exclusion by month, side, symbol, and reason. Every comparison must use the hashed compatible candidate population."]
        (stage / "feature_availability_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        _write_json(stage / "retention_feature_groups.json", {**group_feature_sets, "F4_oi_dynamics": [], "F5_funding_crowding": [], "F7_causal_regime_transition": [], "blocked": {"F4": OI_FUNDING_BLOCK_REASON, "F5": OI_FUNDING_BLOCK_REASON, "F7": "no candidate-level strict OOF/prequential provenance sidecar"}, "compatibility": {"definition": "all fields in every enabled F1/F2/F3/F6/F8 group non-missing", "common_candidate_id_sha256": compatible_id_sha256, "group_candidate_id_sha256": group_hashes}})
        dictionary = {row["feature_name"]: row for row in _lineage_records(inherited_columns=inherited_columns, e15_features=e15_by_side)}
        _write_json(stage / "retention_feature_dictionary.json", dictionary)
        _write_json(stage / "frozen_e15_retention_control.json", {"feature_lists": e15_by_side, "sha256": _sha256(E15_FEATURE_MANIFEST), "source": str(E15_FEATURE_MANIFEST)})
        eligible_universe.to_parquet(stage / "eligible_universe_membership.parquet", index=False, compression="zstd")
        source_paths = sorted(set(source_ledger.get("source_path", pd.Series(dtype=str)).dropna().astype(str)))
        manifest = {"schema": "stage_c_continuation_feature_panel_v3", "status": "MATERIALIZED_RESEARCH_ONLY", "contract": {"target_id": TARGET_ID, "execution_policy_id": EXECUTION_POLICY_ID, "cost_model_id": COST_MODEL_ID, "f0_e15_feature_hash": _sha256(E15_FEATURE_MANIFEST), "h12_label_endpoint": "label_end_ts persisted", "purge_embargo_prerequisite": "12h H12 label horizon; Stage-1 folds must train only before validation_start - 12h and record the rule"}, "rows": {"frozen": len(joined), "compatible": len(population), "clear_first_compatible": int(population.retain_h0_given_clear__valid.sum())}, "compatible_population": {"candidate_id_sha256": compatible_id_sha256, "candidate_id_count": len(compatible_ids), "definition": "complete values in every enabled F1/F2/F3/F6/F8 group", "group_candidate_id_sha256": group_hashes}, "blocked_groups": {"F4": OI_FUNDING_BLOCK_REASON, "F5": OI_FUNDING_BLOCK_REASON, "F7": "learned OOF lineage not yet verified"}, "source_contract": {"ohlcv": "exact bar-cutoff join; completed bar close", "oi_funding": "rejected; no availability contract", "product_mapping": "candidate SYMBOL/USD:USD -> Kraken linear source SYMBOL_USD:USD", "eligible_universe": "timestamp-present completed OHLCV rows within frozen candidate-symbol universe"}, "source_bounds": {"start_utc": str(start), "end_utc": str(end), "source_files": len(source_paths), "eligible_symbol_count": int(bars.symbol.nunique()), "eligible_snapshot_count": int(len(eligible_universe))}, "source_file_sha256": {path: _sha256(Path(path)) for path in source_paths}, "code_sha256": {"continuation_features.py": _sha256(ROOT / "extreme_price_movements/continuation_features.py"), "materializer": _sha256(Path(__file__))}, "inputs": {str(PANEL): _sha256(PANEL), str(ALIGNMENT): _sha256(ALIGNMENT), str(PERSISTENCE): _sha256(PERSISTENCE), str(E15_FEATURE_MANIFEST): _sha256(E15_FEATURE_MANIFEST)}}
        _write_json(stage / "run_manifest.json", manifest)
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(output=args.output, smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()
