#!/usr/bin/env python3
"""Materialise the causal Stage-D A0--A9 action feature substrate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_d_action_features import (
    A1_FEATURES, A2_FEATURES, A3_FEATURES, A4_FEATURES, A5_FEATURES, A9_FEATURES,
    add_a9_composites, batch_path_to_clear_features, build_market_context_snapshots, join_market_context_features,
)
from scripts.materialize_stage_c_continuation_feature_panel import _read_bars

ART = ROOT / "data_perp/artifacts"
COUNTERFACTUALS = ART / "stage_d_action_counterfactuals_20260731_v2/stage_d_action_counterfactuals.parquet"
ALIGNMENT = ART / "historical_exact_h12_alignment_sidecar_research_only_20260731_v1/alignment_sidecar.parquet"
RAW_PANEL = ART / "long_exact_h12_raw_base_panel_20260730_v2/raw_base_panel.parquet"
SELECTED = ART / "exact_h12_target_purity_ablation_20260731_v11/selected_execution_features.json"
PATHS = (
    ART / "failure_2022_2023_pf_exact1m_paths_20260730_v1/paths.parquet",
    ART / "failure_2024_exact1m_paths_20260730_v2/paths.parquet",
)
OI_FUNDING_AUDIT = ART / "stage_d_oi_funding_lineage_audit_20260731_v4/run_manifest.json"
DEFAULT_OUTPUT = ART / "stage_d_action_features_20260731_v4"

FORBIDDEN_FUTURE = re.compile(r"(?:future|label|target|net_continue|delta_continue|continue_better|net_exit_now|exit_reason|mfe_12h|mae_12h)", re.I)
REJECTED_SOURCE = re.compile(r"(?:^oi_|_oi_|funding|^ob_|order_book|liquidat|aggressor)", re.I)
TRANSITIVE_REJECTED_CONTROLS = {
    "mkt_flush_exhaustion_score": "features_oi.py composite uses OI changes/recovery",
    "mkt_leverage_rebuild_score": "features_oi.py composite uses OI and funding changes",
    "unwind": "perp_features.py crowding composite uses OI, funding and basis",
    "unwind_score": "perp_features.py alias of rejected unwind composite",
    "xasset_mkt_spread_bps": "config.py family cross_asset_orderbook; pipeline_steps.py derives from order-book spread",
}
FROZEN_GEOMETRY_REUSE = {"estimated_spread_bps", "entry_half_spread_bps", "barrier_pct", "entry_price_log"}
A0_ACTION_STATE_FIELDS = {
    "time_to_clear_minutes", "gross_return_at_action_bps", "estimated_net_if_exit_now_bps",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def candidate_hash(ids: pd.Series) -> str:
    return hashlib.sha256("\n".join(ids.astype(str)).encode()).hexdigest()


def candidate_set_hash(ids: pd.Series) -> str:
    return hashlib.sha256("\n".join(sorted(ids.astype(str))).encode()).hexdigest()


def admissible_entry_controls(selected: dict[str, list[str]], panel_columns: set[str]) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    admitted, rejected = {}, []
    for side, fields in selected.items():
        admitted[side] = []
        for name in fields:
            reason = None
            if name not in panel_columns:
                reason = "missing_from_frozen_panel"
            elif FORBIDDEN_FUTURE.search(name):
                reason = "future_or_target_semantics_rejected"
            elif REJECTED_SOURCE.search(name):
                reason = "source_lineage_rejected_OI_funding_or_order_book"
            elif name in TRANSITIVE_REJECTED_CONTROLS:
                reason = f"transitive_source_lineage_rejected: {TRANSITIVE_REJECTED_CONTROLS[name]}"
            if reason:
                rejected.append({"feature_name": name, "side": side, "disposition": "REJECTED_LINEAGE", "reason": reason})
            else:
                admitted[side].append(name)
    return admitted, rejected


def entry_control_dependency_ledger(selected: dict[str, list[str]], admitted: dict[str, list[str]], rejected: list[dict[str, str]]) -> pd.DataFrame:
    rejected_map = {(row["side"], row["feature_name"]): row["reason"] for row in rejected}
    rows = []
    for side, names in selected.items():
        for name in names:
            reason = rejected_map.get((side, name))
            if name in FROZEN_GEOMETRY_REUSE:
                family = "frozen_alignment_geometry"
                disposition = "REUSED_FROZEN_GEOMETRY"
                reason = "persisted in frozen alignment; entry_price_log deterministically recomputed from executable entry"
            elif reason:
                family = "rejected_transitive_or_direct_source"
                disposition = "REJECTED_LINEAGE"
            elif name in {"mark_perp_dislocation", "mark_vs_perp_bps"}:
                family, disposition = "causal_mark_and_perpetual_price", "ADMITTED_CAUSAL_ENTRY_STATIC"
            elif "liquidity_ratio" in name:
                family, disposition = "OHLCV_volume_liquidity_ratio", "ADMITTED_CAUSAL_ENTRY_STATIC"
            else:
                family, disposition = "OHLCV_price_volume_or_calendar_transform", "ADMITTED_CAUSAL_ENTRY_STATIC"
            rows.append({"side": side, "feature_name": name, "dependency_family": family, "disposition": disposition, "dependency_evidence": reason or "persisted frozen entry-time value; no OI/funding/order-book dependency found in repository family/definition audit"})
    return pd.DataFrame(rows)


FORMULAS = {
    "side_long": "1[side=long]", "time_to_clear_minutes": "first_clear_bar_index + 1",
    "gross_return_at_action_bps": "side * (completed_clear_bar_close / executable_entry - 1) * 10000",
    "estimated_net_if_exit_now_bps": "gross_return_at_action_bps - known_row_cost_bps; diagnostic close estimate, not action fill",
    "known_row_cost_bps": "frozen total row cost", "barrier_pct": "frozen entry-known barrier fraction",
    "estimated_spread_bps": "frozen entry-known spread estimate", "entry_half_spread_bps": "frozen entry half spread",
    "exit_half_spread_bps": "frozen entry-known exit half-spread estimate",
    "entry_price_log": "natural log(executable entry price)",
    "completed_bars_to_clear": "count(completed 1m bars entry..clear inclusive)",
    "mfe_observed_bps": "max(side-favourable bar extreme / executable_entry - 1) * 10000 through clear",
    "mae_observed_bps": "min(side-aligned adverse bar extreme / executable_entry - 1) * 10000 through clear",
    "mfe_to_mae_ratio": "mfe_observed_bps / max(abs(mae_observed_bps),1e-12)",
    "mae_before_clear_bps": "mae_observed_bps through completed clear bar",
    "distance_to_observed_mfe_bps": "mfe_observed_bps - side_return_at_clear_bps",
    "giveback_from_observed_mfe_bps": "mfe_observed_bps - side_return_at_clear_bps",
    "fraction_observed_mfe_surrendered": "giveback_from_observed_mfe_bps / max(abs(mfe_observed_bps),1e-12)",
    "time_since_observed_mfe_minutes": "completed_bar_count-1-argmax(side-favourable bar extreme)",
    "time_since_observed_mae_minutes": "completed_bar_count-1-argmin(side-adverse bar extreme)",
    "path_efficiency": "abs(side return at clear) / sum(abs(side log 1m returns))",
    "sum_absolute_returns_bps": "sum(abs(side log 1m return bps))",
    "directional_consistency": "mean(side log 1m return >= 0)", "direction_changes": "count(adjacent sign changes in side log 1m returns)",
    "max_counter_direction_move_bps": "maximum cumulative consecutive negative side-return magnitude",
    "max_same_direction_continuation_bps": "maximum cumulative consecutive non-negative side return",
    "return_slope_bps_per_bar": "OLS slope(side close return bps ~ completed bar index)",
    "return_slope_r2": "OLS R2(side close return bps ~ completed bar index)",
    "short_vs_full_path_slope": "OLS slope(last min(5,n) bars) / abs(full-path OLS slope)",
    "return_acceleration_into_clear": "last-min(5,n) OLS slope - preceding-path OLS slope",
    "clear_single_jump_fraction": "max(non-negative side 1m return) / max(side return at clear,1e-12)",
    "latest_candle_body_to_range": "abs(close-open)/(high-low) on completed clear bar",
    "latest_close_location": "(close-low)/(high-low) on completed clear bar",
    "latest_side_aligned_wick_rejection": "upper wick/range for long; lower wick/range for short on clear bar",
    "rolling_side_wick_imbalance": "mean((side-supporting wick-side-rejecting wick)/range) entry..clear",
    "failed_breakout_count": "count(new side extreme followed by close back inside prior side extreme)",
    "breakout_rejection_intensity": "sum(side-rejecting wick/range * failed-breakout indicator)",
    "distance_from_recent_extreme_bps": "observed MFE bps - side close return bps at clear",
    "recency_of_recent_extreme_minutes": "completed_bar_count-1-index of path side extreme",
    "new_side_extremes_since_entry": "count updates to cumulative side high/low including first bar",
    "range_expansion_into_clear": "mean range last min(5,n) bars / mean range first floor(n/3) bars",
    "range_compression_before_clear": "mean range bars n-10..n-3 / mean range first floor(n/3) bars",
    "compression_to_expansion_transition": "mean last-min(5,n) range / pre-clear mean range",
    "post_impulse_rejection": "clear-bar side-rejecting wick fraction * abs(clear-bar return) / absolute path return",
    "fraction_clear_move_latest_bar": "max(clear-bar side return,0) / max(side return at clear,1e-12)",
    "jump_concentration": "max(abs(1m side return)) / sum(abs(1m side returns))",
    "volume_since_entry": "sum(1m volume entry..clear)",
    "volume_acceleration": "mean volume last min(5,n) / mean volume first floor(n/3) - 1",
    "volume_persistence": "mean volume last min(5,n) / full-path mean volume",
    "volume_z_at_clear": "(clear-bar volume-full-path mean volume)/full-path std volume",
    "latest_bars_volume_fraction": "sum volume last min(5,n) / full-path volume",
    "signed_volume_proxy": "side * sign(close-open) * volume on completed clear bar",
    "cumulative_signed_volume_proxy": "sum(side * sign(close-open) * volume) entry..clear",
    "obv_change_proxy": "same causal OHLCV proxy as cumulative_signed_volume_proxy",
    "obv_slope_proxy": "OLS slope of cumulative side-signed-volume proxy",
    "price_volume_correlation": "Pearson correlation(side close-return bps, volume) entry..clear",
    "return_volume_correlation": "Pearson correlation(side 1m log-return bps, volume) entry..clear",
    "volume_weighted_path_efficiency": "path_efficiency * volume-weighted mean(non-negative side return) / mean(abs(side return))",
    "volume_confirmed_continuation": "path_efficiency * max(volume_z_at_clear,0)",
    "high_volume_low_efficiency_churn": "max(volume_z_at_clear,0) * (1-clip(path_efficiency,0,1))",
    "volume_climax": "max(volume_z_at_clear,0)",
    "volume_shock_age_minutes": "bars since last volume > path mean + 2*path std; n if absent",
    "volume_shock_decay": "exp(-volume_shock_age_minutes/n)",
    "range_per_unit_volume_proxy": "sum(high-low)/sum(volume)",
    "absolute_return_per_unit_volume_proxy": "sum(abs(side 1m return bps))/sum(volume)",
    "realised_volatility": "sqrt(mean(side 1m log-return bps squared))",
    "side_adverse_semivolatility": "sqrt(mean(min(side 1m return bps,0)^2))",
    "short_full_volatility_ratio": "RV last min(5,n) bars / full-path RV",
    "volatility_of_volatility": "std(rolling std(side 1m return bps), window=min(5,n), min_periods=2)",
    "atr_change_since_entry": "mean range last min(5,n) / mean range first floor(n/3) - 1",
    "range_expansion_ratio": "clear-bar range / median path bar range",
    "squared_return_autocorrelation": "corr(squared side returns t, t-1)",
    "jump_frequency": "mean(abs(side return)>3*median(abs(side return)))",
    "extreme_bar_frequency": "mean(range > path mean range + 2*path std range)",
    "volatility_shock_magnitude": "max((clear range-path mean range)/path std range,0)",
    "time_since_volatility_shock_minutes": "bars since last range > path mean + 2*path std; n if absent",
    "volatility_shock_decay": "exp(-time_since_volatility_shock_minutes/n)",
    "return_per_unit_volatility": "side return at clear bps / realised_volatility",
    "path_efficiency_conditional_on_volatility": "path_efficiency / realised_volatility",
    "market_return_since_entry": "timestamp-universe median h-hour log return at latest completed hourly bar; h=hours between completed entry/action cutoffs clipped 1..12",
    "market_recent_action_return": "timestamp-universe median 1h log return",
    "side_aligned_breadth": "fraction positive h-hour returns for long; one minus that fraction for short",
    "return_breadth": "fraction timestamp-eligible symbols with positive h-hour return",
    "return_dispersion": "cross-sectional std of h-hour log returns",
    "volatility_dispersion": "cross-sectional std of symbol trailing-24h realised volatility",
    "volume_breadth": "fraction timestamp-eligible symbols with volume/24h-mean above timestamp median",
    "candidate_cross_sectional_return_rank": "percentile rank of candidate symbol h-hour return in timestamp universe",
    "candidate_volume_rank": "percentile rank of candidate volume/24h-mean in timestamp universe",
    "candidate_residual_return_vs_market": "candidate h-hour log return - market_return_since_entry",
    "market_beta": "trailing 24-observation covariance(asset 1h return, timestamp median 1h return)/market variance; min 8",
    "asset_move_vs_market_move": "candidate h-hour log return / abs(market h-hour median return)",
    "breadth_confirmation": "signed side confirmation times side_aligned_breadth",
    "isolated_move_indicator": "1[abs(candidate residual return)>return_dispersion]",
    "leader_laggard_status": "2*candidate_cross_sectional_return_rank-1",
    "change_in_breadth_since_entry": "h-hour breadth at action completed cutoff - 1h breadth at entry completed cutoff",
    "change_in_dispersion_since_entry": "h-hour dispersion at action completed cutoff - 1h dispersion at entry completed cutoff",
    "eligible_universe_size": "count distinct symbols with completed OHLCV at market_feature_cutoff_ts",
    "path_efficiency_x_volume_persistence": "path_efficiency * volume_persistence",
    "path_efficiency_x_breadth_confirmation": "path_efficiency * breadth_confirmation",
    "return_acceleration_x_wick_rejection": "return_acceleration_into_clear * latest_side_aligned_wick_rejection",
    "volume_climax_x_low_path_efficiency": "volume_climax * (1-clip(path_efficiency,0,1))",
    "volatility_climax_x_wick_rejection": "volatility_shock_magnitude * latest_side_aligned_wick_rejection",
    "isolated_move_x_volume_climax": "isolated_move_indicator * volume_climax",
    "giveback_x_time_since_mfe": "giveback_from_observed_mfe_bps * time_since_observed_mfe_minutes",
    "time_to_clear_x_path_efficiency": "time_to_clear_minutes * path_efficiency",
}


def _lineage(name: str, group: str, *, status: str = "ADMITTED_CAUSAL", note: str = "") -> dict[str, Any]:
    is_a0_action_state = group == "A0" and name in A0_ACTION_STATE_FIELDS
    source = "exact completed 1m OHLCV path entry..action_decision"
    if group == "A0": source = "frozen entry-time panel or frozen action geometry"
    if is_a0_action_state:
        source = "exact completed 1m OHLC action path entry..action_decision plus frozen entry geometry"
    if group == "A5": source = "exact action-timestamp eligible candidate-symbol cross-section"
    if group in {"A6", "A7", "A8"}: source = "conditional source not admitted"
    is_path = group in {"A1", "A2", "A3", "A4", "A9"}
    units = "dimensionless_ratio"
    if name.endswith("_bps") or "bps_" in name: units = "bps"
    elif name.endswith("_minutes") or "time_since" in name or name == "completed_bars_to_clear": units = "minutes_or_bars"
    elif "volume" in name and not any(token in name for token in ("rank", "breadth", "persistence", "z_", "fraction", "per_unit")): units = "source_volume_units_or_proxy"
    elif name in {"side_long", "failed_breakout_count", "new_side_extremes_since_entry", "direction_changes", "eligible_universe_size"}: units = "indicator_or_count"
    exact_formula = FORMULAS.get(name, "persisted frozen entry-control value; exact transform defined by frozen feature_set_id")
    bounded_01 = name in {
        "path_efficiency", "directional_consistency", "return_slope_r2", "latest_candle_body_to_range",
        "latest_close_location", "latest_side_aligned_wick_rejection", "latest_bars_volume_fraction",
        "volume_shock_decay", "volatility_shock_decay", "jump_concentration", "jump_frequency",
        "extreme_bar_frequency", "return_breadth", "volume_breadth", "candidate_cross_sectional_return_rank",
        "candidate_volume_rank", "isolated_move_indicator",
    }
    expected_range = "[0,1]" if bounded_01 else "finite_unbounded"
    if name in {"latest_side_aligned_wick_rejection", "latest_close_location", "latest_candle_body_to_range", "jump_concentration"}: expected_range = "[0,1] under valid OHLCV"
    if name == "side_long": expected_range = "{0,1}"
    if name in {"direction_changes", "failed_breakout_count", "new_side_extremes_since_entry", "eligible_universe_size", "completed_bars_to_clear"}: expected_range = "nonnegative_integer"
    if "slope_r2" in name: expected_range = "[0,1]"
    min_obs = 1
    if is_path: min_obs = 2
    if any(token in name for token in ("correlation", "autocorrelation")): min_obs = 3
    if name == "market_beta": min_obs = 8
    if group == "A5" and name not in {"eligible_universe_size"}: min_obs = max(min_obs, 2)
    availability = "action_decision_ts"
    if group == "A0": availability = "entry_ts (persisted frozen row)"
    if is_a0_action_state:
        availability = "action_decision_ts"
    elif group == "A5": availability = "market_source_bar_open_ts + 1h, required <= action_decision_ts"
    lookback = "entry_ts..action_decision_ts" if is_path or is_a0_action_state else ("1h..24h trailing at hourly as-of cutoff" if group == "A5" else "frozen entry-time")
    return {
        "feature_name": name, "feature_group": group, "disposition": status, "source": source,
        "feature_available_ts": availability,
        "path_stop_rule": "inclusive through first_clear_bar_index; future suffix never decoded" if is_path or is_a0_action_state else None,
        "formula": exact_formula, "lookback_window": lookback,
        "minimum_observations": min_obs, "units": units, "expected_range": expected_range,
        "side_normalization": "long=+1 short=-1 where formula declares side; otherwise invariant",
        "missingness_rule": "leave missing; never backfill from future", "staleness_rule": "exact path prefix through action decision" if is_path or is_a0_action_state else ("hourly approximation; latest bar-open whose +1h close is <= action decision; maximum staleness <1h after availability" if group == "A5" else "frozen persisted entry row"),
        "point_in_time_safe": status == "ADMITTED_CAUSAL", "live_reproducible": status == "ADMITTED_CAUSAL",
        "proxy_or_factual": "ohlcv_proxy" if "proxy" in name else "causal_transform",
        "note": note,
    }


def run(*, output: Path, smoke_rows: int | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    counter = pd.read_parquet(COUNTERFACTUALS)
    alignment = pd.read_parquet(ALIGNMENT, columns=[
        "candidate_id", "symbol", "side", "barrier_pct", "estimated_spread_bps",
        "entry_half_spread_bps", "policy_archetype",
        "execution_geometry_key", "feature_set_id",
    ])
    if counter.candidate_id.duplicated().any() or alignment.candidate_id.duplicated().any():
        raise ValueError("candidate identity is not unique")
    rows = counter.merge(alignment, on=["candidate_id", "side"], how="left", validate="one_to_one")
    if rows.symbol.isna().any():
        raise ValueError("alignment coverage incomplete")
    rows["source_symbol"] = rows.symbol.str.replace("/", "_", regex=False)
    rows = rows.sort_values(["action_decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
    if smoke_rows:
        rows = rows.head(smoke_rows).copy()
    wanted = set(rows.candidate_id.astype(str))

    selected = json.loads(SELECTED.read_text())
    panel_schema = set(pq.read_schema(RAW_PANEL).names)
    admitted_control, rejected_control = admissible_entry_controls(selected, panel_schema)
    rejected_control = [row for row in rejected_control if row["feature_name"] not in FROZEN_GEOMETRY_REUSE]
    reserved_action_names = set(A1_FEATURES + A2_FEATURES + A3_FEATURES + A4_FEATURES + A5_FEATURES + A9_FEATURES)
    for side in ("long", "short"):
        collisions = [name for name in admitted_control[side] if name in reserved_action_names]
        admitted_control[side] = [name for name in admitted_control[side] if name not in reserved_action_names]
        rejected_control.extend({"feature_name": name, "side": side, "disposition": "REJECTED_NAME_COLLISION", "reason": "action-time feature of same name is recomputed from causal path"} for name in collisions)
    control_features = sorted(set(admitted_control["long"] + admitted_control["short"]))
    dependency_ledger = entry_control_dependency_ledger(selected, admitted_control, rejected_control)
    panel = pd.read_parquet(RAW_PANEL, columns=["candidate_id", *control_features])
    panel = panel.loc[panel.candidate_id.astype(str).isin(wanted)].drop_duplicates("candidate_id")
    rows = rows.merge(panel, on="candidate_id", how="left", validate="one_to_one")

    seen: set[str] = set()
    source_hashes: dict[str, str] = {}
    fd, path_temp_name = tempfile.mkstemp(prefix=".stage_d_path_features.", suffix=".parquet", dir=output.parent)
    os.close(fd)
    path_temp = Path(path_temp_name)
    path_writer: pq.ParquetWriter | None = None
    max_batch_rows = 0; max_batch_width = 0; max_numeric_working_bytes = 0
    try:
        for path_file in PATHS:
            source_hashes[str(path_file)] = sha256(path_file)
            parquet = pq.ParquetFile(path_file)
            for batch in parquet.iter_batches(batch_size=256, columns=["candidate_id", "execution_future_path"]):
                paths = batch.to_pandas()
                paths = paths.loc[paths.candidate_id.astype(str).isin(wanted)]
                if paths.empty:
                    continue
                joined = paths.merge(rows[["candidate_id", "side", "first_clear_bar_index", "entry_executable_price"]], on="candidate_id", validate="one_to_one")
                ids = joined.candidate_id.astype(str).tolist()
                if len(set(ids)) != len(ids) or seen.intersection(ids):
                    raise ValueError("duplicate path identity")
                feature_batch = batch_path_to_clear_features(
                    joined.execution_future_path.tolist(), stop_indices=joined.first_clear_bar_index.to_numpy(int),
                    sides=joined.side.astype(str).to_numpy(), entry_prices=joined.entry_executable_price.to_numpy(float),
                )
                feature_batch.insert(0, "candidate_id", ids)
                table = pa.Table.from_pandas(feature_batch, preserve_index=False)
                if path_writer is None: path_writer = pq.ParquetWriter(path_temp, table.schema, compression="zstd")
                path_writer.write_table(table if table.schema == path_writer.schema else table.cast(path_writer.schema))
                seen.update(ids)
                width = int(joined.first_clear_bar_index.max()) + 1
                max_batch_rows = max(max_batch_rows, len(joined)); max_batch_width = max(max_batch_width, width)
                # Five OHLC/timestamp matrices plus main derived matrices; conservative upper bound.
                max_numeric_working_bytes = max(max_numeric_working_bytes, int(len(joined) * width * 8 * 24))
            print(f"[stage-d-features] {len(seen):,}/{len(wanted):,} paths", flush=True)
        if path_writer is None: raise ValueError("no path features materialized")
        path_writer.close(); path_writer = None
        if seen != wanted: raise ValueError(f"path coverage incomplete: {len(wanted - seen)} missing")
        path_frame = pd.read_parquet(path_temp)
        rows = rows.merge(path_frame, on="candidate_id", how="left", validate="one_to_one")
    finally:
        if path_writer is not None: path_writer.close()
        path_temp.unlink(missing_ok=True)
    rows["path_observed_through_bar_open_ts"] = pd.to_datetime(rows.pop("_path_last_bar_open_ns").astype("int64"), utc=True)
    if not (rows.path_observed_through_bar_open_ts + pd.Timedelta(minutes=1)).eq(rows.action_decision_ts).all():
        raise AssertionError("completed clear-bar prefix does not end exactly at action decision")
    rows["time_to_clear_minutes"] = rows.first_clear_bar_index.astype(float) + 1.0
    rows["gross_return_at_action_bps"] = rows.side_return_since_entry_bps
    rows["estimated_net_if_exit_now_bps"] = rows.gross_return_at_action_bps - rows.known_row_cost_bps
    rows["side_long"] = rows.side.eq("long").astype(float)
    rows["entry_price_log"] = np.log(rows.entry_executable_price.astype(float))
    # String policy geometry is retained as identity but not silently ordinal encoded.
    a0_geometry = ["side_long", "time_to_clear_minutes", "gross_return_at_action_bps", "estimated_net_if_exit_now_bps", "known_row_cost_bps", "barrier_pct", "estimated_spread_bps", "entry_half_spread_bps", "exit_half_spread_bps", "entry_price_log"]
    from extreme_price_movements.stage_d_action_features import latest_completed_hour_open
    action_market_cutoffs = latest_completed_hour_open(rows.action_decision_ts)
    entry_market_cutoffs = latest_completed_hour_open(rows.entry_ts)
    market_cutoffs = pd.concat([action_market_cutoffs, entry_market_cutoffs], ignore_index=True)
    market_start = entry_market_cutoffs.min() - pd.Timedelta(hours=24)
    market_end = action_market_cutoffs.max()
    market_bars, market_source_ledger = _read_bars(sorted(rows.source_symbol.unique()), market_start, market_end)
    snapshots, membership = build_market_context_snapshots(market_bars, market_cutoffs)
    rows = join_market_context_features(rows, snapshots, membership)
    a3_available = bool(rows[A3_FEATURES].notna().all(axis=1).all())
    volume_dependent_a9 = {"path_efficiency_x_volume_persistence", "volume_climax_x_low_path_efficiency", "isolated_move_x_volume_climax"}
    if a3_available:
        rows = add_a9_composites(rows)
        supported_a9 = list(A9_FEATURES)
    else:
        # Only compose fields whose components passed causal source lineage.
        rows["path_efficiency_x_breadth_confirmation"] = rows.path_efficiency * rows.breadth_confirmation
        rows["return_acceleration_x_wick_rejection"] = rows.return_acceleration_into_clear * rows.latest_side_aligned_wick_rejection
        rows["volatility_climax_x_wick_rejection"] = rows.volatility_shock_magnitude * rows.latest_side_aligned_wick_rejection
        rows["giveback_x_time_since_mfe"] = rows.giveback_from_observed_mfe_bps * rows.time_since_observed_mfe_minutes
        rows["time_to_clear_x_path_efficiency"] = rows.time_to_clear_minutes * rows.path_efficiency
        supported_a9 = [name for name in A9_FEATURES if name not in volume_dependent_a9]
    rows["feature_available_ts"] = rows.action_decision_ts
    if not rows.feature_available_ts.le(rows.action_decision_ts).all():
        raise AssertionError("feature availability violation")

    unsupported_a5: list[str] = []
    supported_a5 = list(A5_FEATURES)
    groups = {
        "A0_minimal_action_state_control": [*a0_geometry, *control_features],
        "A1_path_geometry_to_clear": A1_FEATURES,
        "A2_candle_rejection_structure": A2_FEATURES,
        "A3_volume_confirmation_to_clear": A3_FEATURES if a3_available else [],
        "A4_volatility_instability_to_clear": A4_FEATURES,
        "A5_market_cross_sectional_confirmation": supported_a5,
        "A6_open_interest_path": [], "A7_funding_path_crowding": [], "A8_regime_transition_context": [],
        "A9_compact_composites": supported_a9,
    }
    feature_columns = list(dict.fromkeys(name for values in groups.values() for name in values))
    forbidden_materialized = [name for name in feature_columns if FORBIDDEN_FUTURE.search(name)]
    if forbidden_materialized:
        raise AssertionError(f"future/target feature leaked: {forbidden_materialized}")
    identity = ["candidate_id", "source_symbol", "side", "entry_ts", "first_clear_ts", "action_decision_ts", "action_execution_ts", "horizon_end_ts", "label_available_ts", "execution_policy_id", "cost_model_id", "path_source_id", "path_observed_through_bar_open_ts", "market_source_bar_open_ts", "market_feature_available_ts", "market_entry_source_bar_open_ts", "feature_available_ts", "eligible_universe_membership_sha256"]
    feature_panel = rows[[*identity, *feature_columns]].copy()
    if feature_panel.candidate_id.duplicated().any() or candidate_hash(feature_panel.candidate_id) != candidate_hash(rows.candidate_id):
        raise AssertionError("feature arm identity changed")

    lineage: list[dict[str, Any]] = []
    for group, names in groups.items():
        for name in names:
            lineage.append(_lineage(name, group.split("_", 1)[0]))
    for rejected in rejected_control:
        lineage.append(_lineage(rejected["feature_name"], "A0", status="REJECTED_LINEAGE", note=f"{rejected['side']}: {rejected['reason']}"))
    if not a3_available:
        for name in A3_FEATURES:
            lineage.append(_lineage(name, "A3", status="REJECTED_SOURCE_UNAVAILABLE", note="sealed exact 1m path JSON contains OHLC timestamps but no volume; no exact aligned immutable 1m volume source is proven"))
        for name in sorted(volume_dependent_a9):
            lineage.append(_lineage(name, "A9", status="NOT_RUN_BLOCKED_COMPONENT", note="A3 volume component unavailable"))
    lineage.extend([
        _lineage("A6__REJECTED_LINEAGE", "A6", status="REJECTED_LINEAGE", note="OI audit rejected all sources"),
        _lineage("A7__REJECTED_LINEAGE", "A7", status="REJECTED_LINEAGE", note="funding audit rejected all sources"),
        _lineage("A8__REJECTED_OOF_LINEAGE", "A8", status="REJECTED_OOF_LINEAGE", note="no strict action-level OOF/prequential sidecar proven"),
        _lineage("OI_expansion_x_path_efficiency__NOT_RUN", "A9", status="NOT_RUN_BLOCKED_COMPONENT", note="A6 rejected"),
        _lineage("OI_expansion_x_funding_crowding__NOT_RUN", "A9", status="NOT_RUN_BLOCKED_COMPONENT", note="A6/A7 rejected"),
    ])
    lineage_frame = pd.DataFrame(lineage)

    coverage_parts = []
    month = rows.action_decision_ts.dt.strftime("%Y-%m")
    for name in feature_columns:
        part = rows.groupby([month.rename("month"), "side"], dropna=False)[name].agg(rows="size", non_missing="count").reset_index()
        part["feature_name"] = name
        part["missing_rate"] = 1 - part.non_missing / part.rows.clip(lower=1)
        coverage_parts.append(part)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    dictionary = {row["feature_name"]: row for row in lineage}
    group_payload = {
        **groups,
        "dispositions": {"A3": "ADMITTED_CAUSAL" if a3_available else "REJECTED_SOURCE_UNAVAILABLE", "A6": "REJECTED_LINEAGE", "A7": "REJECTED_LINEAGE", "A8": "REJECTED_OOF_LINEAGE"},
        "common_population": {"rows": len(feature_panel), "ordered_candidate_id_sha256": candidate_hash(feature_panel.candidate_id), "candidate_id_set_sha256": candidate_set_hash(feature_panel.candidate_id)},
        "A5_unmaterialized": unsupported_a5,
    }

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        feature_panel.to_parquet(stage / "stage_d_action_features.parquet", index=False, compression="zstd")
        lineage_frame.to_parquet(stage / "stage_d_action_feature_lineage.parquet", index=False, compression="zstd")
        coverage.to_parquet(stage / "stage_d_action_coverage_report.parquet", index=False, compression="zstd")
        membership.to_parquet(stage / "stage_d_eligible_universe_membership.parquet", index=False, compression="zstd")
        market_source_ledger.to_parquet(stage / "stage_d_market_ohlcv_source_ledger.parquet", index=False, compression="zstd")
        pd.DataFrame(rejected_control).to_parquet(stage / "stage_d_rejected_entry_controls.parquet", index=False, compression="zstd")
        dependency_ledger.to_parquet(stage / "stage_d_entry_control_dependency_ledger.parquet", index=False, compression="zstd")
        write_json(stage / "stage_d_action_feature_dictionary.json", dictionary)
        write_json(stage / "stage_d_action_feature_groups.json", group_payload)
        report = [
            "# Stage-D causal action feature audit", "",
            f"- Materialized {len(feature_panel):,} rows on the exact D0 ordered population (`{candidate_hash(feature_panel.candidate_id)}`).",
            "- Every path transform decodes only bars 0..first_clear_bar_index inclusive; no future suffix is made available to the generator.",
            "- A5 uses all frozen-universe symbols with a completed synchronized OHLCV row at the hourly as-of cutoff; action outcomes do not define membership.",
            "- A6 and A7 are REJECTED_LINEAGE. A8 is REJECTED_OOF_LINEAGE.",
            f"- A3 disposition: {'ADMITTED_CAUSAL' if a3_available else 'REJECTED_SOURCE_UNAVAILABLE: exact one-minute paths have no volume and no aligned immutable replacement is proven'}.",
            "- Inherited OI, funding, order-book, target, and future fields are rejected rather than reused through A0.",
            "- Market beta is a trailing 24-observation asset/median-market beta. Breadth/dispersion change compares the duration bucket with its causal preceding bucket.",
            "- No labels, future MFE/MAE, action execution fill, learned output, threshold, entry rule, or portfolio rule is present.", "",
        ]
        (stage / "stage_d_action_feature_audit.md").write_text("\n".join(report))
        output_names = [p.name for p in stage.iterdir()]
        market_source_paths = sorted(set(market_source_ledger.get("source_path", pd.Series(dtype=str)).dropna().astype(str)))
        market_source_hashes = {path: sha256(Path(path)) for path in market_source_paths}
        manifest = {
            "schema": "stage_d_action_feature_panel_v1", "status": "MATERIALIZED_CAUSAL_FEATURES_ONLY",
            "rows": len(feature_panel), "features": len(feature_columns), "ordered_candidate_id_sha256": candidate_hash(feature_panel.candidate_id), "candidate_id_set_sha256": candidate_set_hash(feature_panel.candidate_id),
            "conditional_groups": {"A3": "ADMITTED_CAUSAL" if a3_available else "REJECTED_SOURCE_UNAVAILABLE", "A6": "REJECTED_LINEAGE", "A7": "REJECTED_LINEAGE", "A8": "REJECTED_OOF_LINEAGE"},
            "inputs": {str(COUNTERFACTUALS): sha256(COUNTERFACTUALS), str(ALIGNMENT): sha256(ALIGNMENT), str(RAW_PANEL): sha256(RAW_PANEL), str(SELECTED): sha256(SELECTED), **source_hashes, **market_source_hashes, str(OI_FUNDING_AUDIT): sha256(OI_FUNDING_AUDIT)},
            "code": {str(Path(__file__).resolve()): sha256(Path(__file__)), str(ROOT / "extreme_price_movements/stage_d_action_features.py"): sha256(ROOT / "extreme_price_movements/stage_d_action_features.py")},
            "compute_contract": {"path_batch_rows_max": max_batch_rows, "path_width_max": max_batch_width, "estimated_numeric_working_bytes_upper_bound": max_numeric_working_bytes, "json_decode": "per-payload unavoidable parsing only", "numeric_generation": "NumPy vectorized across bounded batches; fixed time-axis loops only", "intermediate": "streamed temporary Parquet; no global Python record list"},
            "outputs_sha256": {name: sha256(stage / name) for name in output_names},
        }
        write_json(stage / "run_manifest.json", manifest)
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-rows", type=int)
    args = parser.parse_args()
    print(json.dumps(run(output=args.output, smoke_rows=args.smoke_rows), indent=2))


if __name__ == "__main__":
    main()
