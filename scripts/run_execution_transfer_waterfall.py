#!/usr/bin/env python3
"""Decompose label-to-execution transfer on one immutable trade population.

No model is fitted. Every stage uses the exact same signal rows and changes one
path, entry, geometry, spread, fee, or policy assumption at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import PartitionedOHLCVStore  # noqa: E402
from extreme_price_movements.simple_policy_1m_constrained import (  # noqa: E402
    ConstrainedReplaySpec,
    FAMILY_TRAILING_ONLY,
)
from scripts.replay_current_policy_july_1m import _side_params  # noqa: E402
from scripts.run_simple_policy_1m_capital_ablation import (  # noqa: E402
    _load_deployed_side_params,
)
from scripts.run_simple_policy_1m_constrained_search import (  # noqa: E402
    ExperimentData,
    _causal_entry_atr,
)
from scripts.run_label_first_touch_capture_proxy import (  # noqa: E402
    _first_touch_capture_outcome,
)
from scripts.run_label_widestop_capture_proxy import CaptureArm  # noqa: E402


DEFAULT_SELECTED = Path(
    "data_perp/reports/may_july_failure_diagnosis_20260722_v1/"
    "full_hybrid_long_short_diagnosis/normalized_selected_ledger.parquet"
)
DEFAULT_LABELS = Path(
    "data_perp/artifacts/"
    "20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
)
DEFAULT_STORE = Path("data_perp/exchanges/krakenfutures/execution_1m")
DEFAULT_POLICY = Path(
    "data_perp/artifacts/"
    "s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2/"
    "simple_policy_optimiser/deployment/best_policy_params.json"
)
DEFAULT_PARENT_SUMMARY = Path(
    "data_perp/reports/meta_v9_recovery_20260717/"
    "residual_state_mda95_hier_newaegmm_downstream_retrain_v1/"
    "simple_policy_mayjune_fit_july_holdout_v1/side_parent_policy_summary.csv"
)
DEFAULT_OUT = Path(
    "data_perp/reports/may_july_failure_diagnosis_20260722_v1/"
    "execution_transfer_waterfall_v1"
)

REASON_TIMEOUT = 0
REASON_STOP = 1
REASON_TRAIL = 2

STAGE_COMPARATOR = {
    "recomputed_15m_signal_close": "stored_label",
    "label_1m_signal_open": "recomputed_15m_signal_close",
    "label_1m_signal_close_frozen_anchor": "label_1m_signal_open",
    "label_1m_signal_close_reanchored": "recomputed_15m_signal_close",
    "label_1m_delay_1m_reanchored": "label_1m_signal_close_reanchored",
    "label_1m_delay_2m_reanchored": "label_1m_signal_close_reanchored",
    "label_1m_delay_5m_frozen_anchor": "label_1m_signal_close_frozen_anchor",
    "label_1m_delay_5m_reanchored": "label_1m_signal_close_reanchored",
    "label_1m_delay_5m_reanchored_8h": "label_1m_delay_5m_reanchored",
    "label_1m_delay_10m_reanchored": "label_1m_signal_close_reanchored",
    "label_1m_delay_15m_reanchored": "label_1m_signal_close_reanchored",
    "label_1m_delay_5m_plus_spread": "label_1m_delay_5m_reanchored",
    "label_1m_delay_5m_plus_spread_fee": "label_1m_delay_5m_plus_spread",
    "optimized_policy_delay_5m_raw": "label_1m_delay_5m_reanchored_8h",
    "optimized_policy_delay_5m_posthoc_spread": "optimized_policy_delay_5m_raw",
    "optimized_policy_delay_5m_posthoc_spread_fee": "optimized_policy_delay_5m_posthoc_spread",
    "optimized_policy_delay_5m_spread_aware": "optimized_policy_delay_5m_posthoc_spread",
    "optimized_policy_delay_5m_spread_aware_fee": "optimized_policy_delay_5m_spread_aware",
}


def _read_labels(labels_dir: Path) -> pd.DataFrame:
    columns = [
        "__ts__", "__symbol__", "side_name", "candidate_id",
        "__barrier_pct__", "__first_touch_capture_net__",
        "__first_touch_round_trip_cost__", "__first_touch_hit__",
        "__first_touch_stop__", "__first_touch_timeout__",
        "__first_touch_full_path_mfe_norm__",
        "__first_touch_full_path_mae_norm__", "__first_touch_bar__",
        "__first_touch_mfe_norm__", "__first_touch_mae_norm__",
        "__archetype_policy_tp_r__", "__archetype_policy_sl_r__",
        "__archetype_policy_trail_r__",
        "__archetype_policy_max_bars_to_mfe__",
        "__archetype_label_family__",
    ]
    parts: list[pd.DataFrame] = []
    for month in ("05", "06", "07"):
        for side in ("long", "short"):
            path = labels_dir / f"train_global_{side}_5_2026_{month}.parquet"
            part = pd.read_parquet(path, columns=columns)
            parts.append(part)
    labels = pd.concat(parts, ignore_index=True, copy=False)
    labels["timestamp"] = pd.to_datetime(labels.pop("__ts__"), utc=True)
    labels["symbol"] = labels.pop("__symbol__").astype(str)
    labels["side_name"] = labels["side_name"].astype(str).str.lower()
    return labels


def _load_population(selected_path: Path, labels_dir: Path) -> pd.DataFrame:
    selected = pd.read_parquet(selected_path)
    selected["timestamp"] = pd.to_datetime(selected["timestamp"], utc=True)
    selected["side_name"] = selected["side"].astype(str).str.lower()
    labels = _read_labels(labels_dir)
    rows = selected.merge(
        labels,
        on=["timestamp", "symbol", "side_name"],
        how="inner",
        validate="one_to_one",
    )
    if len(rows) != len(selected):
        raise RuntimeError(
            f"Label join attrition is forbidden: selected={len(selected)} joined={len(rows)}"
        )
    rows = rows.sort_values(["timestamp", "symbol", "side_name"], kind="stable")
    rows = rows.reset_index(drop=True)
    rows["side_sign"] = np.where(rows["side_name"].eq("short"), -1.0, 1.0)
    rows["month"] = rows["timestamp"].dt.strftime("%Y-%m")
    rows["archetype"] = rows["archetype"].fillna(
        rows["__archetype_label_family__"]
    ).astype(str)
    return rows


def _build_multistage_paths(
    rows: pd.DataFrame,
    *,
    store_root: Path,
    offsets_minutes: dict[str, int],
    horizon_minutes: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]:
    n = len(rows)
    shape = (n, int(horizon_minutes))
    paths = {
        name: (
            np.full(shape, np.nan, dtype=np.float32),
            np.full(shape, np.nan, dtype=np.float32),
            np.full(shape, np.nan, dtype=np.float32),
            np.full(shape, np.nan, dtype=np.float32),
        )
        for name in offsets_minutes
    }
    store = PartitionedOHLCVStore(str(store_root), timeframe="1m")
    signal = pd.to_datetime(rows["timestamp"], utc=True)
    total_symbols = int(rows["symbol"].nunique())
    for number, (symbol, group) in enumerate(rows.groupby("symbol", sort=True), start=1):
        group_signal = signal.loc[group.index]
        start = group_signal.min() + pd.Timedelta(minutes=min(offsets_minutes.values()))
        end = (
            group_signal.max()
            + pd.Timedelta(minutes=max(offsets_minutes.values()) + horizon_minutes)
        )
        bars = store.load(
            str(symbol),
            columns=["ts", "open", "high", "low", "close"],
            start_ts=start,
            end_ts=end,
        )
        if bars is None or bars.empty or not isinstance(bars.index, pd.DatetimeIndex):
            continue
        bars = bars[~bars.index.duplicated(keep="last")].sort_index()
        idx = bars.index.tz_localize("UTC") if bars.index.tz is None else bars.index.tz_convert("UTC")
        values = {
            col: pd.to_numeric(bars[col], errors="coerce").to_numpy(dtype=np.float32)
            for col in ("open", "high", "low", "close")
        }
        for row_i, signal_ts in zip(group.index.to_numpy(dtype=np.int64), group_signal):
            for name, offset in offsets_minutes.items():
                entry_ts = signal_ts + pd.Timedelta(minutes=int(offset))
                pos = int(idx.searchsorted(entry_ts))
                if pos + horizon_minutes > len(idx):
                    continue
                actual = idx[pos : pos + horizon_minutes]
                expected = pd.date_range(entry_ts, periods=horizon_minutes, freq="1min", tz="UTC")
                if not actual.equals(expected):
                    continue
                opens, high, low, close = paths[name]
                opens[row_i, :] = values["open"][pos : pos + horizon_minutes]
                high[row_i, :] = values["high"][pos : pos + horizon_minutes]
                low[row_i, :] = values["low"][pos : pos + horizon_minutes]
                close[row_i, :] = values["close"][pos : pos + horizon_minutes]
        if number == 1 or number % 25 == 0 or number == total_symbols:
            print(f"[waterfall-paths] {number}/{total_symbols} {symbol}", flush=True)
    coverage: dict[str, float] = {}
    for name, (opens, high, low, close) in paths.items():
        valid = (
            np.isfinite(opens).all(axis=1)
            & np.isfinite(high).all(axis=1)
            & np.isfinite(low).all(axis=1)
            & np.isfinite(close).all(axis=1)
        )
        coverage[name] = float(valid.mean())
    return paths, {"coverage": coverage, "offsets_minutes": offsets_minutes}


def _aggregate_15m(
    path: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    horizon_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    opens, high, low, close = path
    bars = int(horizon_minutes // 15)
    upto = bars * 15
    high15 = np.nanmax(high[:, :upto].reshape(len(high), bars, 15), axis=2).astype(np.float32)
    low15 = np.nanmin(low[:, :upto].reshape(len(low), bars, 15), axis=2).astype(np.float32)
    close15 = close[:, :upto].reshape(len(close), bars, 15)[:, :, -1].astype(np.float32)
    open15 = opens[:, :upto].reshape(len(opens), bars, 15)[:, :, 0].astype(np.float32)
    return open15, high15, low15, close15


def _simulate_canonical_label_geometry(
    rows: pd.DataFrame,
    path: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    bar_minutes: int,
) -> tuple[np.ndarray, ...]:
    """Resolve row-local geometry with the canonical trailing-label engine."""

    n = len(rows)
    gross = np.full(n, np.nan, dtype=np.float64)
    exit_price = np.full(n, np.nan, dtype=np.float64)
    mfe = np.full(n, np.nan, dtype=np.float64)
    mae = np.full(n, np.nan, dtype=np.float64)
    holding = np.full(n, -1, dtype=np.int32)
    reason = np.full(n, REASON_TIMEOUT, dtype=np.int8)
    valid = np.zeros(n, dtype=bool)
    geometry_cols = [
        "side_name", "__archetype_policy_tp_r__", "__archetype_policy_sl_r__",
        "__archetype_policy_trail_r__", "__archetype_policy_max_bars_to_mfe__",
    ]
    for key, index in rows.groupby(geometry_cols, observed=True, dropna=False).groups.items():
        positions = np.asarray(list(index), dtype=np.int64)
        side_name, tp_r, sl_r, trail_r, max_bars = key
        local_paths = tuple(values[positions] for values in path)
        arm = CaptureArm(
            name="transfer_waterfall",
            tp_r=float(tp_r),
            sl_r=float(sl_r),
            trail_r=float(trail_r),
            max_bars_to_mfe=float(max_bars) * (15.0 / float(bar_minutes)),
            max_barrier=1.0,
        )
        capture = _first_touch_capture_outcome(
            rows.iloc[positions],
            local_paths,
            arm,
            side_name=str(side_name),
            outcome_mode="trailing_profit",
            round_trip_cost=0.0,
            executable_cost_floor=0.0,
        )
        local_gross = pd.to_numeric(
            capture["capture_gross"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        entry = local_paths[0][:, 0].astype(np.float64)
        sign = -1.0 if str(side_name).lower() == "short" else 1.0
        barrier = pd.to_numeric(
            rows.iloc[positions]["__barrier_pct__"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        gross[positions] = local_gross
        exit_price[positions] = entry * (1.0 + sign * local_gross)
        mfe[positions] = pd.to_numeric(
            capture["first_touch_mfe_norm"], errors="coerce"
        ).to_numpy(dtype=np.float64) * barrier
        mae[positions] = pd.to_numeric(
            capture["first_touch_mae_norm"], errors="coerce"
        ).to_numpy(dtype=np.float64) * barrier
        holding[positions] = pd.to_numeric(
            capture["first_touch_bar"], errors="coerce"
        ).fillna(-1).to_numpy(dtype=np.int32)
        is_hit = pd.to_numeric(capture["capture_hit"], errors="coerce").fillna(0).to_numpy() > 0.5
        is_stop = pd.to_numeric(capture["capture_stop"], errors="coerce").fillna(0).to_numpy() > 0.5
        reason[positions] = np.where(
            is_hit, REASON_TRAIL, np.where(is_stop, REASON_STOP, REASON_TIMEOUT)
        )
        valid[positions] = pd.to_numeric(
            capture["capture_valid_path"], errors="coerce"
        ).fillna(0).to_numpy() > 0.5
    return gross, exit_price, mfe, mae, holding, reason, valid


@njit(cache=True, parallel=True)
def _simulate_label_geometry(
    execution_entry: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    side: np.ndarray,
    barrier_pct: np.ndarray,
    tp_r: np.ndarray,
    sl_r: np.ndarray,
    trail_r: np.ndarray,
    max_activation_bars: np.ndarray,
    anchor_entry: np.ndarray,
) -> tuple[np.ndarray, ...]:
    n = len(execution_entry)
    gross = np.full(n, np.nan, dtype=np.float64)
    exit_price = np.full(n, np.nan, dtype=np.float64)
    mfe = np.full(n, np.nan, dtype=np.float64)
    mae = np.full(n, np.nan, dtype=np.float64)
    holding = np.full(n, -1, dtype=np.int32)
    reason = np.full(n, REASON_TIMEOUT, dtype=np.int8)
    valid = np.zeros(n, dtype=np.bool_)
    for i in prange(n):
        entry = float(execution_entry[i])
        anchor = float(anchor_entry[i])
        if not (np.isfinite(entry) and entry > 0.0 and np.isfinite(anchor) and anchor > 0.0):
            continue
        sign = 1.0 if side[i] >= 0.0 else -1.0
        barrier_abs = anchor * max(float(barrier_pct[i]), 1e-8)
        activation = anchor + sign * max(float(tp_r[i]), 0.0) * barrier_abs
        stop_level = anchor - sign * max(float(sl_r[i]), 0.0) * barrier_abs
        trail_abs = max(float(trail_r[i]), 1e-8) * barrier_abs
        max_bar = max(int(max_activation_bars[i]), 1)
        best_fav_px = entry
        max_fav = 0.0
        max_adv = 0.0
        activated = False
        completed = False
        for j in range(high.shape[1]):
            hi = float(high[i, j])
            lo = float(low[i, j])
            op = entry if j == 0 else float(close[i, j - 1])
            cl = float(close[i, j])
            if not (np.isfinite(hi) and np.isfinite(lo) and np.isfinite(cl)):
                break
            fav = max(hi - entry, 0.0) if sign > 0.0 else max(entry - lo, 0.0)
            adv = max(entry - lo, 0.0) if sign > 0.0 else max(hi - entry, 0.0)
            max_fav = max(max_fav, fav)
            max_adv = max(max_adv, adv)
            if activated:
                candidate = best_fav_px - sign * trail_abs
                trail_hit = lo <= candidate if sign > 0.0 else hi >= candidate
                stop_hit = lo <= stop_level if sign > 0.0 else hi >= stop_level
                if trail_hit or stop_hit:
                    chosen = candidate
                    chosen_reason = REASON_TRAIL
                    if stop_hit and (
                        not trail_hit or abs(stop_level - op) <= abs(candidate - op)
                    ):
                        chosen = stop_level
                        chosen_reason = REASON_STOP
                    exit_price[i] = chosen
                    reason[i] = chosen_reason
                    holding[i] = j + 1
                    completed = True
                    break
            else:
                activation_hit = (hi >= activation if sign > 0.0 else lo <= activation) and (j + 1 <= max_bar)
                stop_hit = lo <= stop_level if sign > 0.0 else hi >= stop_level
                if activation_hit and stop_hit:
                    if abs(stop_level - op) <= abs(activation - op):
                        exit_price[i] = stop_level
                        reason[i] = REASON_STOP
                        holding[i] = j + 1
                        completed = True
                        break
                    activated = True
                elif stop_hit:
                    exit_price[i] = stop_level
                    reason[i] = REASON_STOP
                    holding[i] = j + 1
                    completed = True
                    break
                elif activation_hit:
                    activated = True
            if activated:
                best_fav_px = max(best_fav_px, hi) if sign > 0.0 else min(best_fav_px, lo)
        if not completed:
            last = -1
            for j in range(close.shape[1] - 1, -1, -1):
                if np.isfinite(close[i, j]):
                    last = j
                    break
            if last < 0:
                continue
            exit_price[i] = close[i, last]
            holding[i] = last + 1
            reason[i] = REASON_TIMEOUT
        gross[i] = sign * (exit_price[i] / entry - 1.0)
        mfe[i] = max_fav / entry
        mae[i] = max_adv / entry
        valid[i] = True
    return gross, exit_price, mfe, mae, holding, reason, valid


def _stage_frame(
    rows: pd.DataFrame,
    *,
    stage: str,
    gross_raw: np.ndarray,
    exit_raw: np.ndarray,
    entry_raw: np.ndarray,
    mfe: np.ndarray,
    mae: np.ndarray,
    holding: np.ndarray,
    reason: np.ndarray,
    valid: np.ndarray,
    apply_spread: bool,
    fee_round_trip: float,
    bar_minutes: int,
) -> pd.DataFrame:
    side = rows["side_sign"].to_numpy(dtype=np.float64)
    half_spread = rows["p90_spread_bps"].fillna(0.0).to_numpy(dtype=np.float64) / 20_000.0
    if apply_spread:
        entry_exec = entry_raw * (1.0 + side * half_spread)
        exit_exec = exit_raw * (1.0 - side * half_spread)
        gross = side * (exit_exec / np.maximum(entry_exec, 1e-12) - 1.0)
    else:
        gross = gross_raw.copy()
    fee_side = float(fee_round_trip) / 2.0
    net = gross - fee_side - fee_side * (1.0 + gross)
    out = rows[["timestamp", "symbol", "side_name", "archetype", "rank_score"]].copy()
    out["stage"] = stage
    out["gross_return"] = gross
    out["net_return"] = net
    barrier = pd.to_numeric(rows["__barrier_pct__"], errors="coerce").to_numpy(dtype=np.float64)
    out["mfe"] = mfe / np.maximum(barrier, 1e-8)
    out["mae"] = mae / np.maximum(barrier, 1e-8)
    out["holding_minutes"] = holding * int(bar_minutes)
    out["reason"] = reason
    out["valid"] = valid & np.isfinite(net)
    out["target"] = reason == REASON_TRAIL
    out["stop"] = reason == REASON_STOP
    out["timeout"] = reason == REASON_TIMEOUT
    out["spread_applied"] = bool(apply_spread)
    out["round_trip_fee"] = float(fee_round_trip)
    return out


def _stored_stage(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows[["timestamp", "symbol", "side_name", "archetype", "rank_score"]].copy()
    cost = pd.to_numeric(rows["__first_touch_round_trip_cost__"], errors="coerce")
    net = pd.to_numeric(rows["__first_touch_capture_net__"], errors="coerce")
    out["stage"] = "stored_label"
    out["gross_return"] = net + cost
    out["net_return"] = net
    out["mfe"] = pd.to_numeric(rows["__first_touch_mfe_norm__"], errors="coerce")
    out["mae"] = pd.to_numeric(rows["__first_touch_mae_norm__"], errors="coerce")
    out["holding_minutes"] = pd.to_numeric(rows["__first_touch_bar__"], errors="coerce") * 15.0
    out["target"] = pd.to_numeric(rows["__first_touch_hit__"], errors="coerce").fillna(0).gt(0.5)
    out["stop"] = pd.to_numeric(rows["__first_touch_stop__"], errors="coerce").fillna(0).gt(0.5)
    out["timeout"] = pd.to_numeric(rows["__first_touch_timeout__"], errors="coerce").fillna(0).gt(0.5)
    out["reason"] = np.select([out["stop"], out["target"]], [REASON_STOP, REASON_TRAIL], default=REASON_TIMEOUT)
    out["valid"] = np.isfinite(out["net_return"])
    out["spread_applied"] = False
    out["round_trip_fee"] = cost
    return out


def _metrics(frame: pd.DataFrame, stage_order: dict[str, int]) -> pd.DataFrame:
    valid = frame.loc[frame["valid"]].copy()
    valid["month"] = valid["timestamp"].dt.strftime("%Y-%m")
    # The selected population and rank are frozen across stages.  Define each
    # cumulative tail once from the common-row population, globally rather than
    # re-ranking within a month/archetype slice.
    reference_stage = min(stage_order, key=stage_order.get)
    reference = valid.loc[valid["stage"].eq(reference_stage)].copy()
    reference = reference.sort_values(
        ["rank_score", "timestamp", "symbol", "side_name"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    key_cols = ["timestamp", "symbol", "side_name"]
    tail_memberships: list[pd.DataFrame] = []
    for name, fraction in (("top01", 0.01), ("top02", 0.02), ("top05", 0.05), ("top10", 0.10), ("top20", 0.20)):
        count = max(1, int(np.ceil(float(fraction) * len(reference))))
        member = reference.iloc[:count][key_cols].copy()
        member["score_tail"] = name
        tail_memberships.append(member)
    tails = pd.concat(tail_memberships, ignore_index=True, copy=False)
    tail_valid = valid.merge(tails, on=key_cols, how="inner", validate="many_to_many")
    group_specs = [
        ("overall", []), ("side", ["side_name"]), ("month", ["month"]),
        ("side_month", ["side_name", "month"]),
        ("archetype", ["side_name", "archetype"]),
        ("side_month_archetype", ["side_name", "month", "archetype"]),
    ]
    tail_group_specs = [
        ("score_tail", ["score_tail"]),
        ("side_score_tail", ["side_name", "score_tail"]),
        ("month_score_tail", ["month", "score_tail"]),
        ("archetype_score_tail", ["side_name", "archetype", "score_tail"]),
    ]
    records: list[dict[str, Any]] = []
    for scope, cols in [*group_specs, *tail_group_specs]:
        source = valid if scope not in {item[0] for item in tail_group_specs} else tail_valid
        iterator = [((), source)] if not cols else source.groupby(cols, observed=True, dropna=False)
        for key, group in iterator:
            keys = key if isinstance(key, tuple) else (key,)
            base = {col: value for col, value in zip(cols, keys)}
            for stage, part in group.groupby("stage", observed=True):
                records.append({
                    "scope": scope, **base, "stage": stage,
                    "stage_order": int(stage_order[stage]), "trade_count": int(len(part)),
                    "gross_ev_per_trade": float(part["gross_return"].mean()),
                    "net_ev_per_trade": float(part["net_return"].mean()),
                    "win_rate": float(part["net_return"].gt(0).mean()),
                    "mean_mfe": float(part["mfe"].mean()), "mean_mae": float(part["mae"].mean()),
                    "target_rate": float(part["target"].mean()),
                    "stop_rate": float(part["stop"].mean()),
                    "timeout_rate": float(part["timeout"].mean()),
                    "mean_holding_minutes": float(part["holding_minutes"].mean()),
                    "gross_to_net_delta": float((part["gross_return"] - part["net_return"]).mean()),
                    "flips_vs_previous": int(part["outcome_flip_vs_previous"].sum()),
                    "flips_vs_stored": int(part["outcome_flip_vs_stored"].sum()),
                })
    result = pd.DataFrame(records).sort_values(["scope", "stage_order"], kind="stable")
    identity = [
        column for column in ("scope", "side_name", "month", "archetype", "score_tail")
        if column in result.columns
    ]
    result["delta_net_ev_vs_previous"] = result.groupby(identity, dropna=False, observed=True)[
        "net_ev_per_trade"
    ].diff()
    stored = result.loc[result["stage"].eq(reference_stage), [*identity, "net_ev_per_trade"]].rename(
        columns={"net_ev_per_trade": "stored_net_ev_per_trade"}
    )
    result = result.merge(stored, on=identity, how="left", validate="many_to_one")
    result["delta_net_ev_vs_stored"] = result["net_ev_per_trade"] - result["stored_net_ev_per_trade"]
    result["comparison_stage"] = result["stage"].map(STAGE_COMPARATOR)
    comparator = result.loc[:, [
        *identity, "stage", "gross_ev_per_trade", "net_ev_per_trade"
    ]].rename(columns={
        "stage": "comparison_stage",
        "gross_ev_per_trade": "comparison_gross_ev_per_trade",
        "net_ev_per_trade": "comparison_net_ev_per_trade",
    })
    result = result.merge(
        comparator,
        on=[*identity, "comparison_stage"],
        how="left",
        validate="many_to_one",
    )
    result["delta_gross_ev_vs_comparator"] = (
        result["gross_ev_per_trade"] - result["comparison_gross_ev_per_trade"]
    )
    result["delta_net_ev_vs_comparator"] = (
        result["net_ev_per_trade"] - result["comparison_net_ev_per_trade"]
    )
    return result.sort_values(["scope", "stage_order"], kind="stable")


def _write_report(summary: pd.DataFrame, manifest: dict[str, Any], output: Path) -> None:
    overall = summary.loc[summary["scope"].eq("overall")].set_index("stage")
    chain = [
        "stored_label",
        "recomputed_15m_signal_close",
        "label_1m_signal_close_reanchored",
        "label_1m_delay_5m_reanchored",
        "label_1m_delay_5m_reanchored_8h",
        "optimized_policy_delay_5m_raw",
        "optimized_policy_delay_5m_posthoc_spread",
        "optimized_policy_delay_5m_spread_aware",
        "optimized_policy_delay_5m_spread_aware_fee",
    ]
    lines = [
        "# Execution Transfer Waterfall",
        "",
        "## Contract",
        "",
        f"- Fixed selected population: `{manifest['input_rows']:,}` rows.",
        f"- Complete identical-row intersection: `{manifest['common_rows']:,}` rows.",
        "- Stored and recomputed labels begin at `signal + 1h`.",
        "- Label horizon: 96 x 15-minute bars (24h); policy horizon: 8h.",
        f"- Costs: p90 spread once, then `{100.0 * manifest['fee_round_trip']:.2f}%` round-trip fee once.",
        "- No model, threshold, geometry, or policy parameter was fitted in this diagnostic.",
        "",
        "## Overall Waterfall",
        "",
        "| Stage | Gross EV/trade | Net EV/trade | Win rate | Stop-like | Timeout | Delta vs comparator |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in chain:
        row = overall.loc[stage]
        delta = row["delta_gross_ev_vs_comparator"]
        if stage.endswith("_fee"):
            delta = row["delta_net_ev_vs_comparator"]
        delta_text = "n/a" if not np.isfinite(delta) else f"{100.0 * delta:+.4f} pp"
        lines.append(
            f"| `{stage}` | {100.0 * row['gross_ev_per_trade']:+.4f}% | "
            f"{100.0 * row['net_ev_per_trade']:+.4f}% | {100.0 * row['win_rate']:.2f}% | "
            f"{100.0 * row['stop_rate']:.2f}% | {100.0 * row['timeout_rate']:.2f}% | {delta_text} |"
        )
    final_stage = "optimized_policy_delay_5m_spread_aware_fee"
    lines += [
        "",
        "## Attribution",
        "",
        "- Stored-label parity is assessed gross-to-gross because stored labels embed a 1% cost.",
        "- `posthoc_spread` is pure execution-price drag with raw policy exits held fixed.",
        "- `spread_aware` reruns policy triggers/exits with spread-aware state; its delta versus posthoc spread is the policy interaction.",
        "- Optimized-policy `stop_rate` includes full stops plus capital/adverse exits and is therefore a stop-like risk-exit rate.",
        "",
        "## Final Stage By Month And Side",
        "",
        "| Month | Side | Rows | Net EV/trade | Win rate | Stop-like | Timeout |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    sliced = summary.loc[
        summary["scope"].eq("side_month") & summary["stage"].eq(final_stage)
    ].sort_values(["month", "side_name"])
    for _, row in sliced.iterrows():
        lines.append(
            f"| {row['month']} | {row['side_name']} | {int(row['trade_count']):,} | "
            f"{100.0 * row['net_ev_per_trade']:+.4f}% | {100.0 * row['win_rate']:.2f}% | "
            f"{100.0 * row['stop_rate']:.2f}% | {100.0 * row['timeout_rate']:.2f}% |"
        )
    lines += [
        "",
        "## Score Tails",
        "",
        "Global cumulative top-1/2/5/10/20 tails are fixed from the same parent rank and are not recomputed by month or archetype. See `transfer_waterfall_metrics.csv` for every stage and slice.",
        "",
        "## Interpretation Boundary",
        "",
        "This is an identical-row notional-return diagnostic. It is not a portfolio replay and does not report bankroll PnL, concurrency, sizing, or capacity effects.",
    ]
    (output / "TRANSFER_WATERFALL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--parent-summary", type=Path, default=DEFAULT_PARENT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--label-horizon-minutes", type=int, default=1440)
    parser.add_argument("--policy-horizon-minutes", type=int, default=480)
    parser.add_argument("--fee-round-trip", type=float, default=0.003)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_population(args.selected, args.labels_dir)
    offsets = {
        "signal_open": 0, "signal_close": 60, "delay_1m": 61,
        "delay_2m": 62, "delay_5m": 65, "delay_10m": 70, "delay_15m": 75,
    }
    paths, path_manifest = _build_multistage_paths(
        rows, store_root=args.store, offsets_minutes=offsets,
        horizon_minutes=max(
            int(args.label_horizon_minutes), int(args.policy_horizon_minutes)
        ),
    )
    side = rows["side_sign"].to_numpy(dtype=np.float64)
    barrier = pd.to_numeric(rows["__barrier_pct__"], errors="coerce").to_numpy(dtype=np.float64)
    tp_r = pd.to_numeric(rows["__archetype_policy_tp_r__"], errors="coerce").fillna(0.5).to_numpy(dtype=np.float64)
    sl_r = pd.to_numeric(rows["__archetype_policy_sl_r__"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    trail_r = pd.to_numeric(rows["__archetype_policy_trail_r__"], errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
    max_bars = pd.to_numeric(rows["__archetype_policy_max_bars_to_mfe__"], errors="coerce").fillna(24).to_numpy(dtype=np.float64)
    signal_entry = paths["signal_open"][0][:, 0].astype(np.float64)
    stage_frames = [_stored_stage(rows)]

    label_specs = [
        ("recomputed_15m_signal_close", "signal_close", "reanchor", False, 0.0),
        ("label_1m_signal_open", "signal_open", "reanchor", False, 0.0),
        ("label_1m_signal_close_frozen_anchor", "signal_close", "frozen", False, 0.0),
        ("label_1m_signal_close_reanchored", "signal_close", "reanchor", False, 0.0),
        ("label_1m_delay_1m_reanchored", "delay_1m", "reanchor", False, 0.0),
        ("label_1m_delay_2m_reanchored", "delay_2m", "reanchor", False, 0.0),
        ("label_1m_delay_5m_frozen_anchor", "delay_5m", "frozen", False, 0.0),
        ("label_1m_delay_5m_reanchored", "delay_5m", "reanchor", False, 0.0),
        ("label_1m_delay_5m_reanchored_8h", "delay_5m", "reanchor", False, 0.0),
        ("label_1m_delay_10m_reanchored", "delay_10m", "reanchor", False, 0.0),
        ("label_1m_delay_15m_reanchored", "delay_15m", "reanchor", False, 0.0),
        ("label_1m_delay_5m_plus_spread", "delay_5m", "reanchor", True, 0.0),
        ("label_1m_delay_5m_plus_spread_fee", "delay_5m", "reanchor", True, float(args.fee_round_trip)),
    ]
    for stage, path_name, anchor_mode, apply_spread, fee in label_specs:
        path = paths[path_name]
        if stage == "recomputed_15m_signal_close":
            path = _aggregate_15m(
                path, horizon_minutes=int(args.label_horizon_minutes)
            )
            bar_minutes = 15
        else:
            stage_horizon = (
                int(args.policy_horizon_minutes)
                if stage.endswith("_8h")
                else int(args.label_horizon_minutes)
            )
            path = (
                path[0][:, :stage_horizon],
                path[1][:, :stage_horizon],
                path[2][:, :stage_horizon],
                path[3][:, :stage_horizon],
            )
            bar_minutes = 1
        execution_entry = path[0][:, 0].astype(np.float64)
        if anchor_mode == "reanchor":
            result = _simulate_canonical_label_geometry(
                rows, path, bar_minutes=bar_minutes
            )
        else:
            local_max_bars = max_bars * (15.0 / float(bar_minutes))
            result = _simulate_label_geometry(
                execution_entry, path[1], path[2], path[3], side,
                barrier, tp_r, sl_r, trail_r, local_max_bars, signal_entry,
            )
        gross, exit_px, mfe, mae, holding, reason, valid = result
        stage_frames.append(_stage_frame(
            rows, stage=stage, gross_raw=gross, exit_raw=exit_px,
            entry_raw=execution_entry, mfe=mfe, mae=mae,
            holding=holding, reason=reason, valid=valid,
            apply_spread=apply_spread, fee_round_trip=fee,
            bar_minutes=bar_minutes,
        ))

    # Final policy-geometry bridge on the same delayed 5-minute paths.
    policy_payload = json.loads(args.policy.read_text(encoding="utf-8"))
    side_params = _side_params(policy_payload)
    deployed, _ = _load_deployed_side_params(args.parent_summary)
    policy_rows = rows.copy()
    policy_rows["timestamp"] = rows["timestamp"] + pd.Timedelta(hours=1)
    policy_rows["side"] = side
    policy_rows["rank_pct"] = rows["rank_score"]
    policy_rows["spread_cost_bps"] = rows["p90_spread_bps"] / 2.0
    policy_rows["exit_spread_cost_bps"] = rows["p90_spread_bps"] / 2.0
    atr, atr_audit, atr_manifest = _causal_entry_atr(
        policy_rows, store_root=args.store, deployed_by_side=deployed,
        parent_summary=args.parent_summary, warmup_hours=48,
    )
    delayed_full = paths["delay_5m"]
    delayed = tuple(
        values[:, : int(args.policy_horizon_minutes)] for values in delayed_full
    )
    valid = (
        np.isfinite(delayed[0]).all(axis=1) & np.isfinite(delayed[1]).all(axis=1)
        & np.isfinite(delayed[2]).all(axis=1) & np.isfinite(delayed[3]).all(axis=1)
        & np.isfinite(atr)
    )
    def _simulate_policy(*, spread_aware: bool, fee: float) -> dict[str, np.ndarray]:
        local_rows = policy_rows.copy()
        if not spread_aware:
            local_rows["spread_cost_bps"] = 0.0
            local_rows["exit_spread_cost_bps"] = 0.0
        spec = ConstrainedReplaySpec(
            horizon_minutes=int(args.policy_horizon_minutes),
            fee_per_side=float(fee) / 2.0,
        )
        data = ExperimentData(
            local_rows, delayed[0][:, 0], delayed[1], delayed[2], delayed[3],
            valid, atr, spec, deployed,
        )
        idx = np.flatnonzero(data.valid)
        output = data.simulate(idx, side_params, FAMILY_TRAILING_ONLY)
        full = {key: np.full(len(rows), np.nan) for key in ("gross_return", "net_return", "mfe", "mae", "exit_price")}
        full["holding"] = np.full(len(rows), -1, dtype=np.int32)
        full["reason"] = np.full(len(rows), REASON_TIMEOUT, dtype=np.int8)
        full["valid"] = np.zeros(len(rows), dtype=bool)
        for key in ("gross_return", "net_return", "mfe", "mae", "exit_price"):
            full[key][idx] = output[key]
        full["holding"][idx] = output["exit_bars"] + 1
        # Constrained reasons: timeout=0, full stop=1, trailing=3. Collapse
        # capital/adverse exits into stop-like risk exits for this bridge table.
        full["reason"][idx] = np.where(output["reason"] == 3, REASON_TRAIL, np.where(output["reason"] == 0, REASON_TIMEOUT, REASON_STOP))
        full["valid"][idx] = np.isfinite(output["net_return"])
        return full

    raw_full = _simulate_policy(spread_aware=False, fee=0.0)
    for stage, apply_spread, fee in (
        ("optimized_policy_delay_5m_raw", False, 0.0),
        ("optimized_policy_delay_5m_posthoc_spread", True, 0.0),
        (
            "optimized_policy_delay_5m_posthoc_spread_fee",
            True,
            float(args.fee_round_trip),
        ),
    ):
        stage_frames.append(_stage_frame(
            rows, stage=stage, gross_raw=raw_full["gross_return"],
            exit_raw=raw_full["exit_price"],
            entry_raw=delayed[0][:, 0].astype(np.float64),
            mfe=raw_full["mfe"], mae=raw_full["mae"], holding=raw_full["holding"],
            reason=raw_full["reason"], valid=raw_full["valid"],
            apply_spread=apply_spread, fee_round_trip=fee, bar_minutes=1,
        ))

    for stage, fee in (
        ("optimized_policy_delay_5m_spread_aware", 0.0),
        ("optimized_policy_delay_5m_spread_aware_fee", float(args.fee_round_trip)),
    ):
        full = _simulate_policy(spread_aware=True, fee=fee)
        stage_frames.append(_stage_frame(
            rows, stage=stage, gross_raw=full["gross_return"],
            exit_raw=full["exit_price"], entry_raw=delayed[0][:, 0].astype(np.float64),
            mfe=full["mfe"], mae=full["mae"], holding=full["holding"],
            reason=full["reason"], valid=full["valid"], apply_spread=False,
            fee_round_trip=0.0, bar_minutes=1,
        ).assign(
            gross_return=full["gross_return"], net_return=full["net_return"],
            spread_applied=True, round_trip_fee=fee,
        ))

    ledger = pd.concat(stage_frames, ignore_index=True, copy=False)
    stage_names = [frame["stage"].iloc[0] for frame in stage_frames]
    stage_order = {name: i for i, name in enumerate(stage_names)}
    common_keys = None
    for stage, part in ledger.loc[ledger["valid"]].groupby("stage", observed=True):
        keys = set(zip(part["timestamp"], part["symbol"], part["side_name"]))
        common_keys = keys if common_keys is None else common_keys & keys
    common = pd.MultiIndex.from_tuples(sorted(common_keys or []), names=["timestamp", "symbol", "side_name"])
    ledger_index = pd.MultiIndex.from_frame(ledger[["timestamp", "symbol", "side_name"]])
    ledger = ledger.loc[ledger_index.isin(common)].copy()
    ledger["stage_order"] = ledger["stage"].map(stage_order).astype(np.int16)
    ledger = ledger.sort_values(["timestamp", "symbol", "side_name", "stage_order"], kind="stable")

    # Outcome flips use the immediately preceding stage on the common rows.
    ledger["positive"] = ledger["net_return"] > 0.0
    ledger["previous_positive"] = ledger.groupby(["timestamp", "symbol", "side_name"], observed=True)["positive"].shift()
    ledger["outcome_flip_vs_previous"] = ledger["previous_positive"].notna() & ledger["positive"].ne(ledger["previous_positive"])
    stored_positive = ledger.loc[ledger["stage"].eq("stored_label"), ["timestamp", "symbol", "side_name", "positive"]].rename(columns={"positive": "stored_positive"})
    ledger = ledger.merge(stored_positive, on=["timestamp", "symbol", "side_name"], how="left", validate="many_to_one")
    ledger["outcome_flip_vs_stored"] = ledger["positive"].ne(ledger["stored_positive"])

    summary = _metrics(ledger, stage_order)
    ledger.to_parquet(args.output_dir / "identical_row_stage_ledger.parquet", index=False)
    summary.to_csv(args.output_dir / "transfer_waterfall_metrics.csv", index=False)
    atr_audit.to_parquet(args.output_dir / "causal_atr_audit.parquet", index=False)
    manifest = {
        "selected_source": str(args.selected), "labels_dir": str(args.labels_dir),
        "one_minute_store": str(args.store), "policy": str(args.policy),
        "input_rows": int(len(rows)), "common_rows": int(len(common)),
        "sides": rows["side_name"].value_counts().to_dict(),
        "stages": stage_names, "path_manifest": path_manifest,
        "atr_manifest": atr_manifest, "fee_round_trip": float(args.fee_round_trip),
        "spread_contract": "full p90 spread split equally between entry and exit",
        "cost_contract": "spread and fee are introduced in distinct stages and each charged once",
        "selection_contract": "frozen selected population; no stage-specific reselection",
        "label_horizon_minutes": int(args.label_horizon_minutes),
        "policy_horizon_minutes": int(args.policy_horizon_minutes),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    _write_report(summary, manifest, args.output_dir)
    print(json.dumps(manifest, indent=2, default=str))
    print(summary.loc[summary["scope"].eq("overall")].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
