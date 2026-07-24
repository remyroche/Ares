#!/usr/bin/env python3
"""Replay the July winner/deployed comparison after an entry-minute volume filter.

This is an ex-post liquidity diagnostic: a candidate is eligible only when the
1m candle stamped at its entry timestamp has finite, strictly positive volume.
The filter is applied before the shared 8-open/2-new capacity selector.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import PartitionedOHLCVStore  # noqa: E402
from extreme_price_movements.simple_policy_1m_ablation import evaluate_results  # noqa: E402
from extreme_price_movements.simple_policy_1m_constrained import (  # noqa: E402
    FAMILY_TRAILING_ONLY,
    ConstrainedReplaySpec,
)
from scripts.report_simple_policy_1m_winner_daily_july import (  # noqa: E402
    OLD_ATR,
    OLD_CACHE,
    OLD_CANDIDATES,
    PARAMS,
    PARENT,
    POSTERIOR,
    RICH,
    STORE,
    _prediction_candidates,
)
from scripts.report_simple_policy_1m_winner_forward_july import (  # noqa: E402
    CHAMPION,
    _forward_context,
)
from scripts.run_simple_policy_1m_capital_ablation import (  # noqa: E402
    _load_deployed_side_params,
    _load_or_build_path_cache,
)
from scripts.run_simple_policy_1m_constrained_search import (  # noqa: E402
    ExperimentData,
    _indices_between,
)
from scripts.run_simple_policy_1m_contextual_ablation import (  # noqa: E402
    _bayesian_sizes,
    _load_atr,
    _load_context,
)


FORWARD_DIR = CHAMPION / "forward_replay_jul11_17_v1"
DAILY_DIR = CHAMPION / "daily_replay_july01_17_v1"
VOLUME_CACHE = CHAMPION / "reverse_dca_executable_portfolio_v2/volume_cache/volume.f32"


def _combine(parts: list[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def _spread_only_rows(rows: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    out = rows.copy()
    for column in ("spread_cost_bps", "exit_spread_cost_bps"):
        out[column] = pd.to_numeric(out[column], errors="coerce") * multiplier
    return out


def _period_metrics(
    rows: pd.DataFrame,
    outputs: Mapping[str, np.ndarray],
    multipliers: np.ndarray,
    selected: np.ndarray,
    *,
    policy: str,
    cost_model: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    ts = pd.to_datetime(rows["timestamp"], utc=True)
    july = ts.ge(pd.Timestamp("2026-07-01", tz="UTC"))
    rank = pd.to_numeric(rows["rank_pct"], errors="coerce").fillna(0.9).to_numpy(float)
    base = 0.075 + 0.075 * np.power(np.clip(rank, 0.0, 1.0), 1.1)
    size = base * multipliers
    net = np.asarray(outputs["net_return"], dtype=float)
    pnl = net * size
    exit_bars = np.asarray(outputs["exit_bars"], dtype=np.int32)
    chosen = selected & july.to_numpy()

    weekly_records: list[dict[str, Any]] = []
    week = ts.dt.tz_localize(None).dt.to_period("W-SUN").astype(str)
    for label in sorted(week[july].unique()):
        local = chosen & week.eq(label).to_numpy()
        if not local.any():
            continue
        weekly_records.append(
            {
                "policy": policy,
                "cost_model": cost_model,
                "week": label,
                "trades": int(local.sum()),
                "net_ev_per_trade": float(np.mean(net[local])),
                "net_pnl_bankroll": float(np.sum(pnl[local])),
                "hit_rate": float(np.mean(net[local] > 0.0)),
                "mean_holding_hours": float(np.mean(exit_bars[local] + 1) / 60.0),
                "gross_notional_exposure": float(np.sum(size[local])),
            }
        )
    weekly = pd.DataFrame(weekly_records)
    ordered = np.flatnonzero(chosen)
    equity = np.cumsum(pnl[ordered])
    peak = np.maximum.accumulate(np.r_[0.0, equity])[-len(equity) :] if len(equity) else np.array([])
    drawdown = equity - peak if len(equity) else np.array([0.0])
    global_record = {
        "policy": policy,
        "cost_model": cost_model,
        "trades": int(chosen.sum()),
        "net_ev_per_trade": float(np.mean(net[chosen])) if chosen.any() else np.nan,
        "net_pnl_bankroll": float(np.sum(pnl[chosen])),
        "hit_rate": float(np.mean(net[chosen] > 0.0)) if chosen.any() else np.nan,
        "max_drawdown": float(np.min(drawdown)),
        "worst_week": float(weekly["net_pnl_bankroll"].min()) if len(weekly) else np.nan,
        "mean_holding_hours": float(np.mean(exit_bars[chosen] + 1) / 60.0) if chosen.any() else np.nan,
        "gross_notional_exposure": float(np.sum(size[chosen])),
    }
    return global_record, weekly


def _entry_minute_volume(rows: pd.DataFrame, store_root: Path) -> np.ndarray:
    """Load exact event-minute volume; absent/non-finite bars remain ineligible."""
    out = np.full(len(rows), np.nan, dtype=np.float64)
    store = PartitionedOHLCVStore(str(store_root), timeframe="1m")
    for symbol, group in rows.groupby("symbol", sort=True):
        timestamps = pd.to_datetime(group["timestamp"], utc=True)
        bars = store.load(
            str(symbol),
            columns=["ts", "volume"],
            start_ts=timestamps.min(),
            end_ts=timestamps.max() + pd.Timedelta(minutes=1),
        )
        if bars is None or bars.empty or not isinstance(bars.index, pd.DatetimeIndex):
            continue
        bars = bars[~bars.index.duplicated(keep="last")].sort_index()
        index = bars.index.tz_localize("UTC") if bars.index.tz is None else bars.index.tz_convert("UTC")
        volume = pd.Series(pd.to_numeric(bars["volume"], errors="coerce").to_numpy(), index=index)
        out[group.index.to_numpy(np.int64)] = volume.reindex(timestamps).to_numpy(np.float64)
    return out


def _select_after_filter(
    rows: pd.DataFrame,
    outputs: Mapping[str, np.ndarray],
    eligible: np.ndarray,
) -> np.ndarray:
    positions = np.flatnonzero(eligible)
    local = rows.iloc[positions].reset_index(drop=True)
    filler = np.full(len(positions), np.nan)
    _, chosen = evaluate_results(
        local,
        np.asarray(outputs["exit_bars"])[positions],
        np.asarray(outputs["gross_return"])[positions],
        np.asarray(outputs["net_return"])[positions],
        np.asarray(outputs["reason"])[positions],
        filler,
        filler,
        bar_minutes=1,
        apply_capacity=True,
    )
    selected = np.zeros(len(rows), dtype=bool)
    selected[positions[chosen]] = True
    return selected


def _daily(
    rows: pd.DataFrame,
    outputs: Mapping[str, np.ndarray],
    multipliers: np.ndarray,
    selected: np.ndarray,
    *,
    policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(rows["timestamp"], utc=True)
    in_july = ts.ge(pd.Timestamp("2026-07-01", tz="UTC"))
    days = ts.dt.floor("D")
    rank = pd.to_numeric(rows["rank_pct"], errors="coerce").fillna(0.9).to_numpy(float)
    base = 0.075 + 0.075 * np.power(np.clip(rank, 0.0, 1.0), 1.1)
    pnl = np.asarray(outputs["net_return"], dtype=float) * base * multipliers
    records: list[dict[str, Any]] = []
    for day in pd.date_range("2026-07-01", "2026-07-17", freq="D", tz="UTC"):
        local = in_july.to_numpy() & days.eq(day).to_numpy()
        chosen = local & selected
        records.append(
            {
                "day": day,
                "policy": policy,
                "candidates": int(local.sum()),
                "nonzero_volume_candidates": int(
                    (local & (pd.to_numeric(rows["entry_minute_volume"], errors="coerce").to_numpy() > 0.0)).sum()
                ),
                "trades": int(chosen.sum()),
                "net_ev_per_trade": float(np.mean(np.asarray(outputs["net_return"])[chosen]))
                if chosen.any()
                else np.nan,
                "net_pnl_bankroll": float(np.sum(pnl[chosen])),
                "hit_rate": float(np.mean(np.asarray(outputs["net_return"])[chosen] > 0.0))
                if chosen.any()
                else np.nan,
            }
        )
    chosen = selected & in_july.to_numpy()
    ledger = rows.loc[chosen, ["timestamp", "symbol", "side_name", "policy_archetype", "rank_pct"]].copy()
    idx = np.flatnonzero(chosen)
    ledger["policy"] = policy
    ledger["entry_minute_volume"] = rows.loc[chosen, "entry_minute_volume"].to_numpy()
    ledger["exit_bars"] = np.asarray(outputs["exit_bars"])[idx]
    ledger["exit_reason_code"] = np.asarray(outputs["reason"])[idx]
    ledger["gross_return"] = np.asarray(outputs["gross_return"])[idx]
    ledger["net_return"] = np.asarray(outputs["net_return"])[idx]
    ledger["base_size"] = base[idx]
    ledger["size_multiplier"] = multipliers[idx]
    ledger["net_pnl_bankroll"] = pnl[idx]
    return pd.DataFrame(records), ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CHAMPION / "daily_replay_july01_17_nonzero_entry_volume_v1",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    deployed, _ = _load_deployed_side_params(PARENT)
    spec = ConstrainedReplaySpec()
    params = json.loads(PARAMS.read_text())
    geometry = params["fold_3"]["full_train_parent"]
    sizing = params["fold_3"]["sizing"]

    old_rows = pd.read_parquet(OLD_CANDIDATES)
    old_rows["timestamp"] = pd.to_datetime(old_rows["timestamp"], utc=True)
    old_rows = old_rows.sort_values(
        ["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort"
    ).reset_index(drop=True)
    old_context, _, context_audit = _load_context(old_rows, RICH, POSTERIOR)
    old_atr = _load_atr(old_rows, OLD_ATR)
    oo, oh, ol, oc, ov, old_path_manifest = _load_or_build_path_cache(
        old_rows, store_root=STORE, cache_dir=OLD_CACHE, spec=spec, rebuild=False
    )
    old_data = ExperimentData(old_rows, oo, oh, ol, oc, ov, old_atr, spec, deployed)
    train_idx = _indices_between(old_data, "2026-05-01", "2026-06-14")
    train_outputs = old_data.simulate(train_idx, geometry, FAMILY_TRAILING_ONLY)
    old_ts = pd.to_datetime(old_rows["timestamp"], utc=True)
    old_report_idx = np.flatnonzero(
        old_ts.ge(pd.Timestamp("2026-06-29", tz="UTC")).to_numpy()
        & old_ts.lt(pd.Timestamp("2026-07-11", tz="UTC")).to_numpy()
        & old_data.valid
    )

    forward = pd.read_parquet(FORWARD_DIR / "forward_candidates_jul11_16.parquet")
    forward["timestamp"] = pd.to_datetime(forward["timestamp"], utc=True)
    forward = forward.sort_values(
        ["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort"
    ).reset_index(drop=True)
    forward_context, forward_context_audit = _forward_context(forward)
    forward_atr = _load_atr(forward, FORWARD_DIR / "causal_entry_atr_audit.parquet")
    fo, fh, fl, fc, fv, forward_path_manifest = _load_or_build_path_cache(
        forward, store_root=STORE, cache_dir=FORWARD_DIR / "path_cache", spec=spec, rebuild=False
    )
    forward_data = ExperimentData(forward, fo, fh, fl, fc, fv, forward_atr, spec, deployed)

    july17 = pd.read_parquet(DAILY_DIR / "july17_partial_candidates.parquet")
    july17["timestamp"] = pd.to_datetime(july17["timestamp"], utc=True)
    july17 = july17.sort_values(
        ["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort"
    ).reset_index(drop=True)
    spread_reference = pd.read_parquet(
        "data_perp/reports/july_01_16_current_policy_metrics_20260717/"
        "current_policy_candidates_through_july16.parquet"
    )
    rebuilt17, july17_context = _prediction_candidates(
        DAILY_DIR / "jul17_prediction_ledger.parquet",
        pd.Timestamp("2026-07-17 08:00", tz="UTC"),
        spread_reference,
    )
    keys = ["timestamp", "symbol", "side", "rank_pct"]
    if not july17[keys].reset_index(drop=True).equals(rebuilt17[keys].reset_index(drop=True)):
        raise RuntimeError("Saved July 17 candidates do not align with reconstructed sizing context")
    july17_atr = _load_atr(july17, DAILY_DIR / "july17_causal_entry_atr_audit.parquet")
    jo, jh, jl, jc, jv, july17_path_manifest = _load_or_build_path_cache(
        july17,
        store_root=STORE,
        cache_dir=DAILY_DIR / "july17_path_cache",
        spec=spec,
        rebuild=False,
    )
    july17_data = ExperimentData(july17, jo, jh, jl, jc, jv, july17_atr, spec, deployed)

    report_rows = pd.concat([old_rows.iloc[old_report_idx], forward, july17], ignore_index=True)
    if not report_rows["timestamp"].is_monotonic_increasing:
        raise RuntimeError("Combined candidate stream is not chronological")
    report_outputs = _combine(
        [
            old_data.simulate(old_report_idx, geometry, FAMILY_TRAILING_ONLY),
            forward_data.simulate(np.arange(len(forward)), geometry, FAMILY_TRAILING_ONLY),
            july17_data.simulate(np.arange(len(july17)), geometry, FAMILY_TRAILING_ONLY),
        ]
    )
    deployed_outputs = _combine(
        [
            old_data.simulate_deployed(old_report_idx),
            forward_data.simulate_deployed(np.arange(len(forward))),
            july17_data.simulate_deployed(np.arange(len(july17))),
        ]
    )
    spread_only_spec = replace(spec, fee_per_side=0.0)
    old_spread_only = ExperimentData(
        _spread_only_rows(old_rows), oo, oh, ol, oc, ov, old_atr, spread_only_spec, deployed
    )
    forward_spread_only = ExperimentData(
        _spread_only_rows(forward), fo, fh, fl, fc, fv, forward_atr, spread_only_spec, deployed
    )
    july17_spread_only = ExperimentData(
        _spread_only_rows(july17), jo, jh, jl, jc, jv, july17_atr, spread_only_spec, deployed
    )
    spread_only_outputs = _combine(
        [
            old_spread_only.simulate(old_report_idx, geometry, FAMILY_TRAILING_ONLY),
            forward_spread_only.simulate(
                np.arange(len(forward)), geometry, FAMILY_TRAILING_ONLY
            ),
            july17_spread_only.simulate(
                np.arange(len(july17)), geometry, FAMILY_TRAILING_ONLY
            ),
        ]
    )
    deployed_spread_only_outputs = _combine(
        [
            old_spread_only.simulate_deployed(old_report_idx),
            forward_spread_only.simulate_deployed(np.arange(len(forward))),
            july17_spread_only.simulate_deployed(np.arange(len(july17))),
        ]
    )

    combined_rows = pd.concat([old_rows, forward, july17], ignore_index=True)
    combined_context = pd.concat([old_context, forward_context, july17_context], ignore_index=True)
    sizing_data = SimpleNamespace(
        rows=combined_rows,
        side=pd.to_numeric(combined_rows["side"], errors="coerce").to_numpy(float),
        rank=pd.to_numeric(combined_rows["rank_pct"], errors="coerce").to_numpy(float),
    )
    report_combined_idx = np.concatenate(
        [
            old_report_idx,
            np.arange(len(old_rows), len(old_rows) + len(forward), dtype=np.int64),
            np.arange(len(old_rows) + len(forward), len(combined_rows), dtype=np.int64),
        ]
    )
    all_sizes, sizing_state = _bayesian_sizes(
        sizing_data,
        train_idx,
        report_combined_idx,
        train_outputs,
        combined_context,
        strength=float(sizing["strength"]),
        ood_weight=float(sizing["ood_weight"]),
    )
    winner_sizes = all_sizes[report_combined_idx]

    cached_volume = np.memmap(
        VOLUME_CACHE, mode="r", dtype="float32", shape=(len(old_rows), spec.path_len)
    )
    old_entry_volume = np.asarray(cached_volume[old_report_idx, 0], dtype=np.float64)
    forward_entry_volume = _entry_minute_volume(forward, STORE)
    july17_entry_volume = _entry_minute_volume(july17, STORE)
    entry_volume = np.concatenate([old_entry_volume, forward_entry_volume, july17_entry_volume])
    report_rows["entry_minute_volume"] = entry_volume
    eligible = np.isfinite(entry_volume) & (entry_volume > 0.0)
    if not np.isfinite(entry_volume).all():
        raise RuntimeError("Entry-minute volume lookup is incomplete")

    winner_selected = _select_after_filter(report_rows, report_outputs, eligible)
    deployed_selected = _select_after_filter(report_rows, deployed_outputs, eligible)
    spread_only_selected = _select_after_filter(report_rows, spread_only_outputs, eligible)
    deployed_spread_only_selected = _select_after_filter(
        report_rows, deployed_spread_only_outputs, eligible
    )
    winner_daily, winner_ledger = _daily(
        report_rows,
        report_outputs,
        winner_sizes,
        winner_selected,
        policy="joint_trailing_plus_bayesian_raw",
    )
    deployed_daily, deployed_ledger = _daily(
        report_rows,
        deployed_outputs,
        np.ones(len(report_rows)),
        deployed_selected,
        policy="current_deployed_reference",
    )
    _, spread_only_ledger = _daily(
        report_rows,
        spread_only_outputs,
        winner_sizes,
        spread_only_selected,
        policy="joint_trailing_plus_bayesian_raw__spread_1p5_no_fee",
    )
    _, deployed_spread_only_ledger = _daily(
        report_rows,
        deployed_spread_only_outputs,
        np.ones(len(report_rows)),
        deployed_spread_only_selected,
        policy="current_deployed_reference__spread_1p5_no_fee",
    )
    long = pd.concat([winner_daily, deployed_daily], ignore_index=True)
    winner = winner_daily.set_index("day")
    reference = deployed_daily.set_index("day")
    table = winner[["candidates", "trades", "net_ev_per_trade", "net_pnl_bankroll", "hit_rate"]].rename(
        columns={"net_pnl_bankroll": "winner_pnl"}
    )
    table["deployed_trades"] = reference["trades"]
    table["deployed_net_ev_per_trade"] = reference["net_ev_per_trade"]
    table["deployed_pnl"] = reference["net_pnl_bankroll"]
    table["delta_pnl"] = table["winner_pnl"] - table["deployed_pnl"]
    day_volume = report_rows.assign(day=pd.to_datetime(report_rows["timestamp"], utc=True).dt.floor("D"))
    volume_stats = day_volume.loc[day_volume["day"].ge(pd.Timestamp("2026-07-01", tz="UTC"))].groupby("day").agg(
        nonzero_volume_candidates=("entry_minute_volume", lambda x: int(np.sum(np.asarray(x) > 0.0))),
        zero_volume_candidates=("entry_minute_volume", lambda x: int(np.sum(np.asarray(x) == 0.0))),
    )
    table = table.join(volume_stats)
    table["nonzero_candidate_rate"] = table["nonzero_volume_candidates"] / table["candidates"]
    table["status"] = "complete"
    table.loc[pd.Timestamp("2026-07-17", tz="UTC"), "status"] = (
        "partial: entries before 2026-07-17T08:00:00Z"
    )
    table = table.reset_index()

    table.to_csv(args.output_dir / "daily_comparison.csv", index=False)
    long.to_csv(args.output_dir / "daily_metrics_long.csv", index=False)
    report_rows.loc[
        pd.to_datetime(report_rows["timestamp"], utc=True).ge(pd.Timestamp("2026-07-01", tz="UTC"))
    ].to_parquet(args.output_dir / "candidate_liquidity_audit.parquet", index=False)
    pd.concat([winner_ledger, deployed_ledger], ignore_index=True).to_parquet(
        args.output_dir / "selected_trade_ledger.parquet", index=False
    )
    pd.concat(
        [spread_only_ledger, deployed_spread_only_ledger], ignore_index=True
    ).to_parquet(args.output_dir / "cost_ablation_selected_trade_ledger.parquet", index=False)

    metric_specs = (
        (
            report_outputs,
            winner_sizes,
            winner_selected,
            "joint_trailing_plus_bayesian_raw",
            "baseline_1pct_fee_plus_spread",
        ),
        (
            spread_only_outputs,
            winner_sizes,
            spread_only_selected,
            "joint_trailing_plus_bayesian_raw",
            "spread_1p5_no_fee",
        ),
        (
            deployed_outputs,
            np.ones(len(report_rows)),
            deployed_selected,
            "current_deployed_reference",
            "baseline_1pct_fee_plus_spread",
        ),
        (
            deployed_spread_only_outputs,
            np.ones(len(report_rows)),
            deployed_spread_only_selected,
            "current_deployed_reference",
            "spread_1p5_no_fee",
        ),
    )
    global_records: list[dict[str, Any]] = []
    weekly_parts: list[pd.DataFrame] = []
    baseline_selection = {
        "joint_trailing_plus_bayesian_raw": winner_selected,
        "current_deployed_reference": deployed_selected,
    }
    baseline_outputs = {
        "joint_trailing_plus_bayesian_raw": report_outputs,
        "current_deployed_reference": deployed_outputs,
    }
    july_mask = pd.to_datetime(report_rows["timestamp"], utc=True).ge(
        pd.Timestamp("2026-07-01", tz="UTC")
    ).to_numpy()
    spread_observed = np.isfinite(
        pd.to_numeric(report_rows["spread_cost_bps"], errors="coerce").to_numpy(float)
    )
    for outputs, multipliers, selected, policy, cost_model in metric_specs:
        global_record, weekly_metrics = _period_metrics(
            report_rows,
            outputs,
            multipliers,
            selected,
            policy=policy,
            cost_model=cost_model,
        )
        baseline_selected = baseline_selection[policy]
        union = selected | baseline_selected
        global_record["selected_overlap_count"] = int(
            np.sum(selected & baseline_selected & july_mask)
        )
        global_record["selected_jaccard_vs_baseline"] = float(
            np.sum(selected & baseline_selected & july_mask)
            / max(np.sum(union & july_mask), 1)
        )
        global_record["selected_missing_spread_rows"] = int(
            np.sum(selected & july_mask & ~spread_observed)
        )
        global_record["eligible_exit_bar_change_rate_vs_baseline"] = float(
            np.mean(
                np.asarray(outputs["exit_bars"])[eligible & july_mask]
                != np.asarray(baseline_outputs[policy]["exit_bars"])[eligible & july_mask]
            )
        )
        global_records.append(global_record)
        weekly_parts.append(weekly_metrics)
    global_metrics = pd.DataFrame(global_records)
    weekly_metrics = pd.concat(weekly_parts, ignore_index=True)
    for metrics, keys in (
        (global_metrics, ["policy"]),
        (weekly_metrics, ["policy", "week"]),
    ):
        baseline = metrics.loc[
            metrics["cost_model"].eq("baseline_1pct_fee_plus_spread"),
            keys + ["net_pnl_bankroll", "net_ev_per_trade", "trades"],
        ].rename(
            columns={
                "net_pnl_bankroll": "baseline_net_pnl_bankroll",
                "net_ev_per_trade": "baseline_net_ev_per_trade",
                "trades": "baseline_trades",
            }
        )
        merged = metrics.merge(baseline, on=keys, how="left", validate="many_to_one")
        merged["delta_net_pnl_vs_baseline"] = (
            merged["net_pnl_bankroll"] - merged["baseline_net_pnl_bankroll"]
        )
        merged["delta_net_ev_vs_baseline"] = (
            merged["net_ev_per_trade"] - merged["baseline_net_ev_per_trade"]
        )
        if metrics is global_metrics:
            global_metrics = merged
        else:
            weekly_metrics = merged
    global_metrics.to_csv(args.output_dir / "cost_ablation_global_metrics.csv", index=False)
    weekly_metrics.to_csv(args.output_dir / "cost_ablation_weekly_metrics.csv", index=False)
    geometry_artifact = {
        "selection_contract": "frozen; no geometry or sizing re-optimization under alternate costs",
        "winner_family": "trailing_only_total_mfe",
        "winner_geometry_by_side": geometry,
        "winner_sizing": sizing,
        "deployed_geometry_by_side": deployed,
    }
    (args.output_dir / "selected_geometries.json").write_text(
        json.dumps(geometry_artifact, indent=2, default=str)
    )
    manifest = {
        "status": "complete_with_partial_july17",
        "filter": "finite entry-minute 1m candle volume > 0, applied before capacity admission",
        "causality_warning": (
            "ex-post diagnostic: final volume for candle t is unavailable at the open of t; "
            "do not use as a live gate without lagging to t-1 or defining an intrabar execution rule"
        ),
        "portfolio": "same rank order, duplicate-symbol rule, maximum 8 open and 2 new per timestamp",
        "costs": "same 1% round trip plus policy spread, applied once",
        "cost_ablation": {
            "baseline": "0.5% fee per side plus one half-spread at entry and exit",
            "alternate": "zero fees; entry and exit half-spreads each multiplied by 1.5",
            "sizing": "frozen baseline raw-Bayesian sizing state",
            "geometry": "frozen for both cost arms; no cost-conditioned re-optimization",
        },
        "july17_entry_cutoff_exclusive_utc": "2026-07-17T08:00:00Z",
        "warmup_start_utc": str(report_rows["timestamp"].min()),
        "candidate_rows_including_warmup": int(len(report_rows)),
        "eligible_rows_including_warmup": int(eligible.sum()),
        "july_candidate_rows": int(pd.to_datetime(report_rows["timestamp"], utc=True).ge(pd.Timestamp("2026-07-01", tz="UTC")).sum()),
        "sizing_state": sizing_state,
        "context_audit": context_audit,
        "forward_context_audit": forward_context_audit,
        "path_manifests": [old_path_manifest, forward_path_manifest, july17_path_manifest],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(table.to_string(index=False))
    print("\nCOST ABLATION GLOBAL\n", global_metrics.to_string(index=False))
    print("\nCOST ABLATION WEEKLY\n", weekly_metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
