#!/usr/bin/env python3
"""Nested executable ATR pyramiding and production-constrained portfolio search."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import PartitionedOHLCVStore
from extreme_price_movements.simple_policy_1m_ablation import capacity_select
from extreme_price_movements.simple_policy_1m_constrained import ConstrainedReplaySpec
from extreme_price_movements.simple_policy_1m_executable_pyramiding import (
    simulate_executable_pyramiding,
)
from extreme_price_movements.simple_policy_1m_pyramiding_portfolio import (
    allocate_pyramiding_portfolio,
)
from scripts.report_simple_policy_1m_winner_forward_july import BASE, CHAMPION
from scripts.run_simple_policy_1m_capital_ablation import (
    FOLDS,
    _load_deployed_side_params,
    _load_or_build_path_cache,
)
from scripts.run_simple_policy_1m_constrained_search import (
    INNER_FOLDS,
    ExperimentData,
    _indices_between,
)
from scripts.run_simple_policy_1m_contextual_ablation import (
    _bayesian_sizes,
    _load_atr,
    _load_context,
)

X_GRID = (1, 2, 3, 4, 5, 6, 8)
Y_GRID = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00)
MIN_BARS = 5
MIN_GAP_BPS = 50.0
WALLET_CAP = 0.75
POSITION_CAP = 0.15
MAX_OPEN = 8
MAX_NEW = 2
MAX_DCA_PER_MINUTE = 2
MIN_ORDER = 0.001


def _load_or_build_volume_cache(
    rows: pd.DataFrame,
    *,
    store_root: Path,
    cache_dir: Path,
    path_len: int,
) -> tuple[np.memmap, np.memmap, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    volume_path = cache_dir / "volume.f32"
    previous_path = cache_dir / "entry_previous_minute_volume.f32"
    manifest_path = cache_dir / "volume_manifest.json"
    shape = (len(rows), path_len)
    if volume_path.exists() and previous_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("shape") == [len(rows), path_len]:
            return (
                np.memmap(volume_path, mode="r", dtype="float32", shape=shape),
                np.memmap(previous_path, mode="r", dtype="float32", shape=(len(rows),)),
                manifest,
            )

    volume = np.memmap(volume_path, mode="w+", dtype="float32", shape=shape)
    previous = np.memmap(previous_path, mode="w+", dtype="float32", shape=(len(rows),))
    volume[:] = np.nan
    previous[:] = np.nan
    store = PartitionedOHLCVStore(str(store_root), timeframe="1m")
    symbols = rows["symbol"].nunique()
    for number, (symbol, group) in enumerate(rows.groupby("symbol", sort=True), start=1):
        timestamps = pd.to_datetime(group["timestamp"], utc=True)
        bars = store.load(
            str(symbol),
            columns=["ts", "volume"],
            start_ts=timestamps.min() - pd.Timedelta(minutes=1),
            end_ts=timestamps.max() + pd.Timedelta(minutes=path_len),
        )
        if bars is None or bars.empty or not isinstance(bars.index, pd.DatetimeIndex):
            print(f"[volume-cache {number}/{symbols}] {symbol} empty", flush=True)
            continue
        bars = bars[~bars.index.duplicated(keep="last")].sort_index()
        index = bars.index.tz_localize("UTC") if bars.index.tz is None else bars.index.tz_convert("UTC")
        values = pd.to_numeric(bars["volume"], errors="coerce").to_numpy(np.float32)
        filled = 0
        for row_i, timestamp in zip(group.index.to_numpy(np.int64), timestamps):
            position = int(index.searchsorted(timestamp))
            if position <= 0 or position + path_len > len(index):
                continue
            expected = pd.date_range(timestamp, periods=path_len, freq="1min", tz="UTC")
            if not index[position : position + path_len].equals(expected):
                continue
            if index[position - 1] != timestamp - pd.Timedelta(minutes=1):
                continue
            volume[row_i, :] = values[position : position + path_len]
            previous[row_i] = values[position - 1]
            filled += 1
        print(f"[volume-cache {number}/{symbols}] {symbol} rows={filled}/{len(group)}", flush=True)
    volume.flush()
    previous.flush()
    manifest = {
        "shape": [len(rows), path_len],
        "source": str(store_root),
        "entry_liquidity": "strictly positive volume in completed minute t-1",
        "add_liquidity": "strictly positive volume in completed trigger minute",
        "finite_path_rows": int(np.isfinite(volume).all(axis=1).sum()),
        "positive_previous_entry_rows": int((previous > 0.0).sum()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return volume, previous, manifest


def _global_exit_arrays(n: int, idx: np.ndarray, outputs: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    bars = np.full(n, -1, dtype=np.int32)
    prices = np.full(n, np.nan, dtype=np.float64)
    bars[idx] = np.asarray(outputs["exit_bars"], dtype=np.int32)
    prices[idx] = np.asarray(outputs["exit_price"], dtype=np.float64)
    return bars, prices


def _schedule(
    data: ExperimentData,
    volume: np.ndarray,
    idx: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    y: float,
) -> dict[str, Any]:
    exit_bars, exit_prices = _global_exit_arrays(len(data.rows), idx, outputs)
    return simulate_executable_pyramiding(
        idx,
        data.open0,
        data.close,
        volume,
        data.side,
        data.atr_frac,
        data.entry_spread,
        2.0 * data.entry_spread,
        exit_bars,
        exit_prices,
        np.ones(8),
        y,
        data.spec.fee_per_side,
        minimum_bars_between_fills=MIN_BARS,
        minimum_gap_bps=MIN_GAP_BPS,
    )


def _weights(x: int, *, total: float = 1.0, initial: float | None = None, ratio: float = 1.0) -> np.ndarray:
    if x == 1:
        return np.asarray([total], dtype=np.float64)
    if initial is None:
        return np.full(x, total / x, dtype=np.float64)
    initial = min(max(float(initial), 1e-9), float(total))
    remaining = max(float(total) - initial, 0.0)
    powers = np.power(max(float(ratio), 1e-9), np.arange(x - 1, dtype=np.float64))
    tail = remaining * powers / max(float(powers.sum()), 1e-12)
    return np.r_[initial, tail]


def _position_metrics(
    rows: pd.DataFrame,
    exit_bars: np.ndarray,
    selected: np.ndarray,
    net_pnl: np.ndarray,
    gross_pnl: np.ndarray,
    allocated: np.ndarray,
    filled_tranches: np.ndarray,
    *,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    chosen = np.flatnonzero(selected)
    if not len(chosen):
        return {"n_trades": 0, "net_pnl_bankroll": 0.0, "objective": -1e9}
    entry_ts = pd.to_datetime(rows.iloc[chosen]["timestamp"], utc=True).reset_index(drop=True)
    exit_ts = entry_ts + pd.to_timedelta(exit_bars[chosen] + 1, unit="m")
    order = np.argsort(exit_ts.astype("int64").to_numpy(), kind="mergesort")
    pnl = net_pnl[chosen]
    gross = gross_pnl[chosen]
    equity = np.cumsum(pnl[order])
    peak = np.maximum.accumulate(np.r_[0.0, equity])[-len(equity) :]
    drawdown = equity - peak
    week = exit_ts.dt.tz_localize(None).dt.to_period("W").astype(str)
    month = exit_ts.dt.strftime("%Y-%m")
    weekly = pd.Series(pnl).groupby(week).sum().to_numpy(float)
    monthly = pd.Series(pnl).groupby(month).sum().to_numpy(float)
    mean, std, worst = float(weekly.mean()), float(weekly.std()), float(weekly.min())
    max_dd = float(drawdown.min())
    per_trade_return = np.divide(pnl, allocated[chosen], out=np.zeros(len(chosen)), where=allocated[chosen] > 0)
    result = {
        "n_trades": int(len(chosen)),
        "trades_per_day": float(len(chosen) / max((entry_ts.max() - entry_ts.min()).total_seconds() / 86400.0 + 1.0, 1.0)),
        "gross_pnl_bankroll": float(gross.sum()),
        "fee_pnl_bankroll": float((gross - pnl).sum()),
        "net_pnl_bankroll": float(pnl.sum()),
        "mean_net_return_on_allocated": float(per_trade_return.mean()),
        "hit_rate": float((pnl > 0.0).mean()),
        "worst_week": worst,
        "worst_month": float(monthly.min()),
        "weekly_mean": mean,
        "weekly_std": std,
        "positive_week_fraction": float((weekly > 0).mean()),
        "max_drawdown": max_dd,
        "objective": float(mean - 0.5 * std + 0.25 * worst - 0.10 * abs(max_dd)),
        "mean_allocated": float(allocated[chosen].mean()),
        "total_allocated": float(allocated[chosen].sum()),
        "mean_filled_tranches": float(filled_tranches[chosen].mean()),
        "fraction_above_original_target": 0.0,
    }
    if extras:
        result.update(extras)
    return result


def _slot_only_metrics(
    data: ExperimentData,
    idx: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    schedule: Mapping[str, np.ndarray],
    sizes: np.ndarray,
    previous_volume: np.ndarray,
    x: int,
    y: float,
) -> dict[str, Any]:
    rows = data.rows.iloc[idx].reset_index(drop=True)
    eligible = np.isfinite(previous_volume[idx]) & (previous_volume[idx] > 0.0)
    eligible_local = np.flatnonzero(eligible)
    selected = np.zeros(len(idx), dtype=bool)
    if len(eligible_local):
        cap = capacity_select(
            pd.to_datetime(rows.iloc[eligible_local]["timestamp"], utc=True).astype("int64").to_numpy(np.int64),
            pd.Categorical(rows.iloc[eligible_local]["symbol"].astype(str)).codes.astype(np.int32),
            np.asarray(outputs["exit_bars"], dtype=np.int32)[eligible_local],
            1,
        )
        selected[eligible_local] = cap
    rank = pd.to_numeric(rows["rank_pct"], errors="coerce").fillna(0.9).to_numpy(float)
    target = (0.075 + 0.075 * np.power(np.clip(rank, 0, 1), 1.1)) * sizes[idx]
    raw_net = np.nan_to_num(schedule["tranche_net_return"][:, :x], nan=0.0)
    raw_gross = np.nan_to_num(schedule["tranche_gross_return"][:, :x], nan=0.0)
    weights = _weights(x)
    net_pnl = target * (raw_net @ weights)
    gross_pnl = target * (raw_gross @ weights)
    filled = np.isfinite(schedule["tranche_net_return"][:, :x])
    allocated = target * (filled @ weights)
    metrics = _position_metrics(
        rows,
        np.asarray(outputs["exit_bars"], dtype=np.int32),
        selected,
        net_pnl,
        gross_pnl,
        allocated,
        filled.sum(axis=1),
        extras={
            "entry_liquidity_reject_rate": float((~eligible).mean()),
            "add_zero_volume_rejections": int(np.asarray(schedule["zero_volume_rejections"])[selected].sum()),
            "add_time_gate_rejections": int(np.asarray(schedule["time_gate_rejections"])[selected].sum()),
            "spread_floor_binding_rate": float(
                np.mean(
                    np.maximum(2.0 * data.entry_spread[idx][selected], MIN_GAP_BPS)
                    >= 10_000.0 * float(y) * data.atr_frac[idx][selected]
                )
            )
            if selected.any()
            else 0.0,
        },
    )
    metrics["fraction_above_original_target"] = float(np.mean(allocated[selected] > target[selected] + 1e-12)) if selected.any() else 0.0
    return metrics


def _portfolio_metrics(
    data: ExperimentData,
    idx: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    schedule: Mapping[str, np.ndarray],
    sizes: np.ndarray,
    previous_volume: np.ndarray,
    weights: np.ndarray,
    dca_bonus: float,
) -> dict[str, Any]:
    rows = data.rows.iloc[idx].reset_index(drop=True)
    ts_min = (pd.to_datetime(rows["timestamp"], utc=True).astype("int64").to_numpy(np.int64) // 60_000_000_000).astype(np.int64)
    symbols = pd.Categorical(rows["symbol"].astype(str)).codes.astype(np.int32)
    rank = pd.to_numeric(rows["rank_pct"], errors="coerce").fillna(0.9).to_numpy(float)
    target = (0.075 + 0.075 * np.power(np.clip(rank, 0, 1), 1.1)) * sizes[idx]
    eligible = np.isfinite(previous_volume[idx]) & (previous_volume[idx] > 0.0)
    result = allocate_pyramiding_portfolio(
        ts_min,
        symbols,
        rank,
        np.asarray(outputs["exit_bars"], dtype=np.int32),
        np.asarray(schedule["fill_bars"], dtype=np.int32),
        np.asarray(schedule["tranche_gross_return"], dtype=np.float64),
        np.asarray(schedule["tranche_net_return"], dtype=np.float64),
        target,
        eligible,
        np.asarray(weights, dtype=np.float64),
        wallet_cap=WALLET_CAP,
        position_cap=POSITION_CAP,
        max_open=MAX_OPEN,
        max_new_per_minute=MAX_NEW,
        max_dca_per_minute=MAX_DCA_PER_MINUTE,
        dca_priority_bonus=float(dca_bonus),
        minimum_order=MIN_ORDER,
    )
    selected, net_pnl, gross_pnl, allocated, initial_allocated, filled = result[:6]
    diagnostics = result[-1]
    metrics = _position_metrics(
        rows,
        np.asarray(outputs["exit_bars"], dtype=np.int32),
        selected,
        net_pnl,
        gross_pnl,
        allocated,
        filled,
        extras={
            "entry_liquidity_rejects": int(np.asarray(result[6]).sum()),
            "entry_slot_rejects": int(np.asarray(result[7]).sum()),
            "entry_book_rejects": int(np.asarray(result[8]).sum()),
            "dca_book_rejects": int(np.asarray(result[9]).sum()),
            "dca_order_cap_rejects": int(np.asarray(result[10]).sum()),
            "peak_book": float(diagnostics[0]),
            "mean_book_time": float(diagnostics[1]),
            "book_cap_minute_fraction": float(diagnostics[2]),
            "turnover": float(diagnostics[3]),
            "mean_initial_allocated": float(initial_allocated[selected].mean()) if selected.any() else 0.0,
        },
    )
    metrics["fraction_above_original_target"] = float(np.mean(allocated[selected] > target[selected] + 1e-12)) if selected.any() else 0.0
    return metrics


def _best(frame: pd.DataFrame) -> dict[str, Any]:
    return frame.sort_values(
        ["objective", "worst_week", "max_drawdown", "turnover" if "turnover" in frame else "n_trades"],
        ascending=[False, False, False, True if "turnover" in frame else False],
        kind="mergesort",
    ).iloc[0].to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CHAMPION / "reverse_dca_executable_portfolio_v2",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    candidates = BASE / "execution_candidates_may_july_v1/simple_policy_candidates_with_archetypes.parquet"
    rich = BASE / "admission_may_july_oos_v1/admitted_oos_rows_execution_ledger.parquet"
    posterior = BASE / "complete_parent_state_july_v1/complete_oos_residual_event_states.parquet"
    parent = BASE / "simple_policy_mayjune_fit_july_holdout_v1/side_parent_policy_summary.csv"
    nested = json.loads((CHAMPION / "evidence/nested_params.json").read_text())
    rows = pd.read_parquet(candidates)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows = rows.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    context, _, context_audit = _load_context(rows, rich, posterior)
    atr = _load_atr(rows, CHAMPION / "replay/causal_entry_atr_audit.parquet")
    deployed, _ = _load_deployed_side_params(parent)
    spec = ConstrainedReplaySpec()
    open0, high, low, close, valid, path_manifest = _load_or_build_path_cache(
        rows,
        store_root=Path("data_perp/exchanges/krakenfutures/execution_1m"),
        cache_dir=CHAMPION / "replay/path_cache",
        spec=spec,
        rebuild=False,
    )
    volume, previous_volume, volume_manifest = _load_or_build_volume_cache(
        rows,
        store_root=Path("data_perp/exchanges/krakenfutures/execution_1m"),
        cache_dir=args.output_dir / "volume_cache",
        path_len=spec.path_len,
    )
    data = ExperimentData(rows, open0, high, low, close, valid, atr, spec, deployed)

    fold_records: list[dict[str, Any]] = []
    search_records: list[dict[str, Any]] = []
    local_records: list[dict[str, Any]] = []
    choices: dict[str, Any] = {}
    for fold in FOLDS:
        name = fold["fold"]
        inner = INNER_FOLDS[name]
        search_idx = _indices_between(data, fold["train_start"], inner["search_end"])
        inner_idx = _indices_between(data, inner["inner_start"], inner["inner_end"])
        train_idx = _indices_between(data, fold["train_start"], fold["train_end"])
        outer_idx = _indices_between(data, fold["validation_start"], fold["validation_end"])
        sizing = nested[name]["sizing"]
        search_params = nested[name]["search_parent"]
        outer_params = nested[name]["full_train_parent"]
        fit_search = data.simulate(search_idx, search_params, 0)
        fit_outer = data.simulate(train_idx, outer_params, 0)
        sizes_inner, _ = _bayesian_sizes(
            data, search_idx, inner_idx, fit_search, context,
            strength=float(sizing["strength"]), ood_weight=float(sizing["ood_weight"]),
        )
        sizes_outer, sizing_state = _bayesian_sizes(
            data, train_idx, outer_idx, fit_outer, context,
            strength=float(sizing["strength"]), ood_weight=float(sizing["ood_weight"]),
        )
        inner_outputs = data.simulate(inner_idx, search_params, 0)
        outer_outputs = data.simulate(outer_idx, outer_params, 0)
        inner_schedules = {y: _schedule(data, volume, inner_idx, inner_outputs, y) for y in Y_GRID}

        group1_rows = []
        for y in Y_GRID:
            for x in X_GRID:
                metrics = _slot_only_metrics(
                    data, inner_idx, inner_outputs, inner_schedules[y], sizes_inner, previous_volume, x, y
                )
                record = {"fold": name, "stage": "group1_inner", "x": x, "y_atr": y, **metrics}
                group1_rows.append(record)
                search_records.append(record)
        group1_frame = pd.DataFrame(group1_rows)
        group1_best = _best(group1_frame)
        top_pairs = (
            group1_frame.loc[group1_frame["x"].gt(1)]
            .sort_values(["objective", "worst_week", "max_drawdown"], ascending=[False, False, False])
            .drop_duplicates(["x", "y_atr"])
            .head(3)[["x", "y_atr"]]
            .to_dict("records")
        )

        # Group 2: production-constrained, staged joint search over unequal
        # weights, optional over-target exposure, and DCA/new-entry priority.
        portfolio_search = []
        for pair in top_pairs:
            x, y = int(pair["x"]), float(pair["y_atr"])
            schedule = inner_schedules[y]
            for total in (0.75, 1.0, 1.15, 1.30, 1.50, 2.0):
                for initial in (0.25, 0.50, 0.75, 1.0):
                    if initial > total:
                        continue
                    for ratio in (0.50, 1.0, 2.0):
                        weights = _weights(x, total=total, initial=initial, ratio=ratio)
                        for bonus in (-0.025, 0.0, 0.025):
                            metrics = _portfolio_metrics(
                                data, inner_idx, inner_outputs, schedule, sizes_inner,
                                previous_volume, weights, bonus,
                            )
                            record = {
                                "fold": name, "stage": "group2_inner", "x": x,
                                "y_atr": y, "total_target_multiple": total,
                                "initial_target_multiple": initial, "tail_ratio": ratio,
                                "dca_priority_bonus": bonus, **metrics,
                            }
                            portfolio_search.append(record)
                            search_records.append(record)
        portfolio_frame = pd.DataFrame(portfolio_search)
        capped_best = _best(portfolio_frame.loc[portfolio_frame["total_target_multiple"].le(1.0)])
        over_best = _best(portfolio_frame.loc[portfolio_frame["total_target_multiple"].gt(1.0)])

        group1_local = group1_frame.loc[
            group1_frame["x"].between(max(int(group1_best["x"]) - 1, 1), int(group1_best["x"]) + 1)
            & group1_frame["y_atr"].between(max(float(group1_best["y_atr"]) - 0.5, 0.0), float(group1_best["y_atr"]) + 0.5)
        ]
        local_records.append({
            "fold": name,
            "policy": "group1_executable_equal",
            "neighbors": int(len(group1_local)),
            "median_objective": float(group1_local["objective"].median()),
            "minimum_objective": float(group1_local["objective"].min()),
            "selected_objective": float(group1_best["objective"]),
        })
        for label, choice in (("group2_unequal_capped", capped_best), ("group2_unequal_over_target", over_best)):
            local = portfolio_frame.loc[
                portfolio_frame["x"].eq(int(choice["x"]))
                & portfolio_frame["y_atr"].between(max(float(choice["y_atr"]) - 0.75, 0.0), float(choice["y_atr"]) + 0.75)
                & portfolio_frame["total_target_multiple"].between(max(float(choice["total_target_multiple"]) - 0.35, 0.0), float(choice["total_target_multiple"]) + 0.35)
                & portfolio_frame["initial_target_multiple"].between(max(float(choice["initial_target_multiple"]) - 0.25, 0.0), float(choice["initial_target_multiple"]) + 0.25)
                & portfolio_frame["tail_ratio"].between(float(choice["tail_ratio"]) / 2.0, float(choice["tail_ratio"]) * 2.0)
                & portfolio_frame["dca_priority_bonus"].between(float(choice["dca_priority_bonus"]) - 0.025, float(choice["dca_priority_bonus"]) + 0.025)
            ]
            local_records.append({
                "fold": name,
                "policy": label,
                "neighbors": int(len(local)),
                "median_objective": float(local["objective"].median()),
                "minimum_objective": float(local["objective"].min()),
                "selected_objective": float(choice["objective"]),
            })

        outer_schedule_cache: dict[float, dict[str, Any]] = {}
        def outer_schedule(y: float) -> dict[str, Any]:
            if y not in outer_schedule_cache:
                outer_schedule_cache[y] = _schedule(data, volume, outer_idx, outer_outputs, y)
            return outer_schedule_cache[y]

        baseline_schedule = outer_schedule(float(Y_GRID[0]))
        baseline_slot = _slot_only_metrics(
            data, outer_idx, outer_outputs, baseline_schedule, sizes_outer, previous_volume, 1, 0.0
        )
        fold_records.append({"fold": name, "policy": "strict_liquidity_slot_baseline", "x": 1, "y_atr": 0.0, **baseline_slot})

        gx, gy = int(group1_best["x"]), float(group1_best["y_atr"])
        group1_outer = _slot_only_metrics(
            data, outer_idx, outer_outputs, outer_schedule(gy), sizes_outer, previous_volume, gx, gy
        )
        fold_records.append({"fold": name, "policy": "group1_executable_equal", "x": gx, "y_atr": gy, **group1_outer})

        forced_x, forced_y = int(top_pairs[0]["x"]), float(top_pairs[0]["y_atr"])
        forced_outer = _slot_only_metrics(
            data,
            outer_idx,
            outer_outputs,
            outer_schedule(forced_y),
            sizes_outer,
            previous_volume,
            forced_x,
            forced_y,
        )
        fold_records.append({
            "fold": name,
            "policy": "group1_best_forced_dca",
            "x": forced_x,
            "y_atr": forced_y,
            **forced_outer,
        })

        global_baseline = _portfolio_metrics(
            data, outer_idx, outer_outputs, baseline_schedule, sizes_outer,
            previous_volume, np.asarray([1.0]), 0.0,
        )
        fold_records.append({"fold": name, "policy": "production_constrained_x1", "x": 1, "y_atr": 0.0, **global_baseline})

        equal_global = _portfolio_metrics(
            data, outer_idx, outer_outputs, outer_schedule(gy), sizes_outer,
            previous_volume, _weights(gx), 0.0,
        )
        fold_records.append({"fold": name, "policy": "group1_equal_under_global", "x": gx, "y_atr": gy, **equal_global})

        forced_equal_global = _portfolio_metrics(
            data,
            outer_idx,
            outer_outputs,
            outer_schedule(forced_y),
            sizes_outer,
            previous_volume,
            _weights(forced_x),
            0.0,
        )
        fold_records.append({
            "fold": name,
            "policy": "group1_forced_dca_under_global",
            "x": forced_x,
            "y_atr": forced_y,
            **forced_equal_global,
        })

        selected_configs = (("group2_unequal_capped", capped_best), ("group2_unequal_over_target", over_best))
        for policy, choice in selected_configs:
            x, y = int(choice["x"]), float(choice["y_atr"])
            weights = _weights(
                x,
                total=float(choice["total_target_multiple"]),
                initial=float(choice["initial_target_multiple"]),
                ratio=float(choice["tail_ratio"]),
            )
            metrics = _portfolio_metrics(
                data, outer_idx, outer_outputs, outer_schedule(y), sizes_outer,
                previous_volume, weights, float(choice["dca_priority_bonus"]),
            )
            fold_records.append({
                "fold": name, "policy": policy, "x": x, "y_atr": y,
                "total_target_multiple": float(choice["total_target_multiple"]),
                "initial_target_multiple": float(choice["initial_target_multiple"]),
                "tail_ratio": float(choice["tail_ratio"]),
                "dca_priority_bonus": float(choice["dca_priority_bonus"]), **metrics,
            })

        choices[name] = {
            "group1": {"x": gx, "y_atr": gy, "inner_objective": group1_best["objective"]},
            "group1_best_forced_dca": {"x": forced_x, "y_atr": forced_y},
            "group2_capped": {k: capped_best[k] for k in ("x", "y_atr", "total_target_multiple", "initial_target_multiple", "tail_ratio", "dca_priority_bonus", "objective")},
            "group2_over_target": {k: over_best[k] for k in ("x", "y_atr", "total_target_multiple", "initial_target_multiple", "tail_ratio", "dca_priority_bonus", "objective")},
            "sizing_state": sizing_state,
        }
        print(f"{name}: group1 x={gx} y={gy}; capped={choices[name]['group2_capped']}; over={choices[name]['group2_over_target']}", flush=True)

    fold_metrics = pd.DataFrame(fold_records)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    search = pd.DataFrame(search_records)
    search.to_parquet(args.output_dir / "inner_search.parquet", index=False)
    pd.DataFrame(local_records).to_csv(args.output_dir / "local_robustness.csv", index=False)
    summary = (
        fold_metrics.groupby("policy", sort=False)
        .agg(
            folds=("fold", "count"), total_trades=("n_trades", "sum"),
            total_net_pnl=("net_pnl_bankroll", "sum"), mean_net_pnl=("net_pnl_bankroll", "mean"),
            worst_fold_pnl=("net_pnl_bankroll", "min"), worst_week=("worst_week", "min"),
            worst_drawdown=("max_drawdown", "min"), mean_net_return=("mean_net_return_on_allocated", "mean"),
            mean_hit_rate=("hit_rate", "mean"), mean_allocated=("mean_allocated", "mean"),
            mean_filled_tranches=("mean_filled_tranches", "mean"),
            mean_fraction_above_target=("fraction_above_original_target", "mean"),
        )
        .reset_index()
    )
    global_base = summary.loc[summary["policy"].eq("production_constrained_x1"), "total_net_pnl"].iloc[0]
    summary["delta_total_pnl_vs_production_constrained_x1"] = summary["total_net_pnl"] - global_base
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    manifest = {
        "status": "complete",
        "experiment": "executable ATR favorable pyramiding with initial-anchor frozen exits and production portfolio constraints",
        "evidence": "nested policy-validation OOS; repeated May-June policy research, not untouched confirmation",
        "foundation": "joint trailing total-MFE plus raw Bayesian sizing; ATR spacing; initial-entry exit anchor",
        "execution": {
            "minimum_minutes_between_fills": MIN_BARS,
            "trigger": "completed 1m favorable close",
            "add_volume": "positive trigger-minute volume required",
            "initial_liquidity": "positive completed t-1 minute volume required",
            "fill": "trigger-minute close plus side-correct entry half-spread",
            "minimum_gap": "max(y * causal entry-frozen ATR, point-in-time full spread, 50 bps)",
            "one_add_per_bar": True,
            "no_add_on_exit_bar": True,
            "blocked_add": "cancel remaining chain; never backfill stale levels",
            "known_limitation": "positive prior-minute volume is a causal liquidity screen, not order-book fill proof",
        },
        "portfolio": {
            "wallet_cap": WALLET_CAP, "position_cap": POSITION_CAP,
            "max_open": MAX_OPEN, "max_new_per_minute": MAX_NEW,
            "max_dca_per_minute": MAX_DCA_PER_MINUTE, "same_symbol": 1,
            "event_order": "frozen exits release first; DCA and new entries share a rank auction",
        },
        "search": {
            "x": list(X_GRID), "y_atr": list(Y_GRID),
            "group2_stage1_top_pairs": 3,
            "total_target_multiple": [0.75, 1.0, 1.15, 1.30, 1.50, 2.0],
            "initial_target_multiple": [0.25, 0.50, 0.75, 1.0],
            "tail_geometric_ratio": [0.50, 1.0, 2.0],
            "dca_priority_bonus": [-0.025, 0.0, 0.025],
            "selection": "inner validation only; outer fold fixed",
            "evaluated_rows": int(len(search)),
        },
        "folds": FOLDS, "inner_folds": INNER_FOLDS, "choices": choices,
        "cost": "0.5% entry plus 0.5% exit fee per filled tranche once; side-correct spreads",
        "path_manifest": path_manifest, "volume_manifest": volume_manifest,
        "context_audit": context_audit, "elapsed_seconds": time.monotonic() - started,
        "outputs": ["summary.csv", "fold_metrics.csv", "inner_search.parquet", "local_robustness.csv"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
