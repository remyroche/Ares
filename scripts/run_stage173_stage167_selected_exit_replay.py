#!/usr/bin/env python3
"""Replay alternative exits for Stage167 selected rows on executable candles.

Stage172 showed that the Stage167 label artifact's net-return columns collapse
to first-touch capture on selected rows. This script re-fetches the same delayed
15m policy paths used by the first-touch materializer and computes actual
exit-policy outcomes on the Stage167 selected ledger.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import simple_policy_optimiser as spo  # noqa: E402
from scripts.run_first_touch_label_training_smoke import _table  # noqa: E402
from scripts.run_label_first_touch_capture_proxy import _fetch_policy_paths  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import ROUND_TRIP_COST, _json_safe  # noqa: E402


DEFAULT_LEDGER_CSV = Path(
    "data_perp/reports/stage167_full_path_tail_feature_gap_v1/"
    "stage167_full_path_tail_selected_ledger.csv"
)
DEFAULT_LABELS_PATH = Path("data_perp/artifacts/20260703_190000_clean_first_touch_tail_veto_stage167_labels/labels")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/stage173_stage167_selected_exit_replay_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_HOLD_BARS = (2, 4, 6, 8, 12, 24, 48, 96)
DEFAULT_MAX_BARRIER = 0.03


@dataclass(frozen=True)
class TrailSpec:
    name: str
    hold_bars: int
    activation_mult: float
    min_activation_mult: float
    giveback_frac: float
    decay_half_life_bars: float
    decay_start_bars: int


def _safe_numeric(values: Any, *, index: pd.Index | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    if values is None:
        return pd.Series(np.nan, index=index)
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def _safe_mean(values: Any) -> float:
    series = _safe_numeric(values).dropna()
    return float(series.mean()) if len(series) else float("nan")


def _safe_sum(values: Any) -> float:
    series = _safe_numeric(values).dropna()
    return float(series.sum()) if len(series) else 0.0


def _safe_quantile(values: Any, q: float) -> float:
    series = _safe_numeric(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(series.quantile(float(q))) if len(series) else float("nan")


def _parse_csv(value: str | list[str] | tuple[str, ...], default: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value).strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_int_csv(value: str | list[int] | tuple[int, ...], default: tuple[int, ...] = ()) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    text = str(value).strip()
    if not text:
        return list(default)
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _load_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"__ts__", "__symbol__", "period", "first_touch_net", "barrier", "side"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame = frame.copy()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], errors="coerce")
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    frame["period"] = frame["period"].astype(str)
    if frame["__ts__"].isna().any():
        raise ValueError(f"{path} contains non-parseable __ts__ values")
    dupes = int(frame.duplicated(["__ts__", "__symbol__"]).sum())
    if dupes:
        raise ValueError(f"{path} contains duplicate __ts__/__symbol__ keys: {dupes}")
    return frame.sort_values(["__ts__", "__symbol__"], kind="mergesort").reset_index(drop=True)


def _label_parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if files:
            return files
    raise FileNotFoundError(f"No parquet label files found at {path}")


def _load_label_subset(path: Path) -> pd.DataFrame:
    requested = [
        "__ts__",
        "__symbol__",
        "__barrier_pct__",
        "__first_touch_effective_tp_abs__",
        "__first_touch_effective_sl_abs__",
        "__first_touch_capture_net__",
        "__first_touch_hit__",
        "__first_touch_stop__",
        "__first_touch_timeout__",
        "__first_touch_bar__",
        "__first_touch_full_path_mae_to_sl__",
        "__first_touch_full_path_mfe_to_tp__",
        "__u_policy_net__",
        "__r_policy_net__",
        "__y_ret__",
    ]
    parts: list[pd.DataFrame] = []
    for file in _label_parquet_files(path):
        columns = pd.read_parquet(file, columns=None).columns
        keep = [col for col in requested if col in columns]
        missing_keys = sorted({"__ts__", "__symbol__"}.difference(keep))
        if missing_keys:
            raise ValueError(f"{file} is missing label key columns: {missing_keys}")
        parts.append(pd.read_parquet(file, columns=keep))
    frame = pd.concat(parts, ignore_index=True)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], errors="coerce")
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    dupes = int(frame.duplicated(["__ts__", "__symbol__"]).sum())
    if dupes:
        raise ValueError(f"{path} contains duplicate __ts__/__symbol__ keys: {dupes}")
    return frame.sort_values(["__ts__", "__symbol__"], kind="mergesort").reset_index(drop=True)


def _join_inputs(ledger: pd.DataFrame, labels: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    joined = ledger.merge(labels, on=["__ts__", "__symbol__"], how="left", validate="one_to_one")
    if "__barrier_pct__" not in joined.columns:
        joined["__barrier_pct__"] = pd.to_numeric(joined["barrier"], errors="coerce")
    missing = int(joined["__barrier_pct__"].isna().sum())
    if missing:
        raise ValueError(f"Missing joined label barrier rows for selected ledger keys: {missing}")
    joined["month"] = joined["__ts__"].dt.to_period("M").astype(str)
    joined["week"] = joined["__ts__"].dt.to_period("W-SUN").astype(str)
    if months:
        joined = joined[joined["month"].isin(months)].copy()
    if joined.empty:
        raise ValueError("No selected rows remain after month filtering")
    return joined.reset_index(drop=True)


def _side_array(frame: pd.DataFrame) -> np.ndarray:
    side = _safe_numeric(frame.get("side"), index=frame.index).fillna(1.0).to_numpy(dtype=np.float64)
    side = np.where(side < 0.0, -1.0, 1.0)
    return side


def _fetch_paths(
    frame: pd.DataFrame,
    *,
    labels_path: Path,
    data_root: Path,
    market_mode: str,
    exchange: str,
    path_len: int,
    apply_delayed_entry: bool,
) -> tuple[pd.DataFrame, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    side_values = set(np.unique(_side_array(frame)).tolist())
    if side_values == {-1.0}:
        side_name = "short"
    elif side_values == {1.0}:
        side_name = "long"
    else:
        raise ValueError("Stage173 currently expects one side per selected ledger; got mixed sides")
    rows, paths, stats = _fetch_policy_paths(
        frame,
        labels_path=labels_path,
        side=side_name,
        data_root=data_root,
        market_mode=market_mode,
        exchange=exchange,
        path_len=int(path_len),
        apply_delayed_entry=bool(apply_delayed_entry),
    )
    return rows, paths, stats


def _entry_and_path_returns(
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    side: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    opens, highs, lows, closes = paths
    entry = opens[:, 0].astype(np.float64, copy=False)
    denom = np.maximum(entry[:, None], 1e-12)
    fav = np.where(
        side[:, None] >= 0.0,
        (highs.astype(np.float64) - entry[:, None]) / denom,
        (entry[:, None] - lows.astype(np.float64)) / denom,
    )
    adv = np.where(
        side[:, None] >= 0.0,
        (entry[:, None] - lows.astype(np.float64)) / denom,
        (highs.astype(np.float64) - entry[:, None]) / denom,
    )
    close_ret = side[:, None] * (closes.astype(np.float64) / denom - 1.0)
    fav = np.where(np.isfinite(fav), np.maximum(fav, 0.0), np.nan)
    adv = np.where(np.isfinite(adv), np.maximum(adv, 0.0), np.nan)
    close_ret = np.where(np.isfinite(close_ret), close_ret, np.nan)
    finite = spo._policy_path_finite_mask(paths) & np.isfinite(entry) & (entry > 0.0)
    return entry, fav, adv, close_ret, closes.astype(np.float64), finite


def _same_bar_decision(
    *,
    bar_open: np.ndarray,
    entry: np.ndarray,
    side: np.ndarray,
    tp_abs: np.ndarray,
    sl_abs: np.ndarray,
    tp_hit: np.ndarray,
    sl_hit: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(bar_open), dtype=np.int8)
    result[tp_hit & ~sl_hit] = 1
    result[sl_hit & ~tp_hit] = -1
    both = tp_hit & sl_hit
    if np.any(both):
        tp_px = entry[both] * (1.0 + side[both] * tp_abs[both])
        sl_px = entry[both] * (1.0 - side[both] * sl_abs[both])
        tp_dist = np.abs(tp_px - bar_open[both])
        sl_dist = np.abs(sl_px - bar_open[both])
        both_result = np.where(tp_dist < sl_dist, 1, -1).astype(np.int8)
        result[np.flatnonzero(both)] = both_result
    return result


def _finalize_policy_frame(
    *,
    base: pd.DataFrame,
    policy_name: str,
    policy_family: str,
    net: np.ndarray,
    gross: np.ndarray,
    exit_bars: np.ndarray,
    exit_reason: np.ndarray,
    finite_path: np.ndarray,
    fav: np.ndarray,
    adv: np.ndarray,
    tp_abs: np.ndarray,
    sl_abs: np.ndarray,
) -> pd.DataFrame:
    n = len(base)
    safe_exit = np.clip(np.nan_to_num(exit_bars, nan=1.0).astype(int), 1, fav.shape[1])
    max_fav = np.full(n, np.nan, dtype=np.float64)
    max_adv = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not finite_path[i]:
            continue
        end = int(safe_exit[i])
        max_fav[i] = float(np.nanmax(fav[i, :end]))
        max_adv[i] = float(np.nanmax(adv[i, :end]))
    peak_giveback = np.maximum(max_fav - np.maximum(gross, -1e9), 0.0)
    out = pd.DataFrame(
        {
            "policy": str(policy_name),
            "policy_family": str(policy_family),
            "__ts__": base["__ts__"].to_numpy(copy=False),
            "__symbol__": base["__symbol__"].astype(str).to_numpy(copy=False),
            "month": base["month"].astype(str).to_numpy(copy=False),
            "week": base["week"].astype(str).to_numpy(copy=False),
            "side": _side_array(base),
            "barrier_pct": _safe_numeric(base["__barrier_pct__"]).to_numpy(dtype=np.float64),
            "tp_abs": tp_abs.astype(np.float64, copy=False),
            "sl_abs": sl_abs.astype(np.float64, copy=False),
            "finite_path": finite_path.astype(float),
            "net_return": np.where(finite_path, net, np.nan),
            "gross_return": np.where(finite_path, gross, np.nan),
            "round_trip_cost": float(ROUND_TRIP_COST),
            "exit_bars": np.where(finite_path, exit_bars, np.nan),
            "exit_hours": np.where(finite_path, exit_bars.astype(np.float64) * 0.25, np.nan),
            "exit_reason": np.where(finite_path, exit_reason, "missing_path"),
            "mfe_to_tp_until_exit": max_fav / np.maximum(tp_abs, 1e-12),
            "mae_to_sl_until_exit": max_adv / np.maximum(sl_abs, 1e-12),
            "max_favorable_return_until_exit": max_fav,
            "max_adverse_return_until_exit": max_adv,
            "peak_giveback_return": np.where(finite_path, peak_giveback, np.nan),
            "peak_giveback_to_tp": peak_giveback / np.maximum(tp_abs, 1e-12),
            "label_first_touch_net": _safe_numeric(base.get("first_touch_net"), index=base.index).to_numpy(
                dtype=np.float64
            ),
            "label_first_touch_bar": _safe_numeric(
                base.get("__first_touch_bar__"),
                index=base.index,
            ).to_numpy(dtype=np.float64),
            "label_full_path_mae_to_sl": _safe_numeric(
                base.get("__first_touch_full_path_mae_to_sl__"),
                index=base.index,
            ).to_numpy(dtype=np.float64),
        }
    )
    return out


def _fixed_hold_policy(
    base: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    hold_bars: int,
    round_trip_cost: float,
) -> pd.DataFrame:
    opens, _highs, _lows, _closes = paths
    side = _side_array(base)
    _entry, fav, adv, close_ret, _close_px, finite = _entry_and_path_returns(paths, side)
    idx = min(max(1, int(hold_bars)), close_ret.shape[1]) - 1
    gross = close_ret[:, idx]
    net = gross - float(round_trip_cost)
    exit_bars = np.full(len(base), idx + 1, dtype=np.float64)
    reason = np.full(len(base), f"fixed_hold_{idx + 1}", dtype=object)
    tp_abs = _safe_numeric(base["__first_touch_effective_tp_abs__"]).to_numpy(dtype=np.float64)
    sl_abs = _safe_numeric(base["__first_touch_effective_sl_abs__"]).to_numpy(dtype=np.float64)
    return _finalize_policy_frame(
        base=base,
        policy_name=f"fixed_hold_{idx + 1}",
        policy_family="fixed_hold",
        net=net,
        gross=gross,
        exit_bars=exit_bars,
        exit_reason=reason,
        finite_path=finite & np.isfinite(opens[:, 0]) & np.isfinite(gross),
        fav=fav,
        adv=adv,
        tp_abs=tp_abs,
        sl_abs=sl_abs,
    )


def _tp_sl_hold_policy(
    base: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    hold_bars: int,
    max_tp_bars: int,
    round_trip_cost: float,
) -> pd.DataFrame:
    opens, highs, lows, _closes = paths
    side = _side_array(base)
    entry, fav, adv, close_ret, _close_px, finite = _entry_and_path_returns(paths, side)
    n = len(base)
    max_len = close_ret.shape[1]
    hold = min(max(1, int(hold_bars)), max_len)
    tp_abs = _safe_numeric(base["__first_touch_effective_tp_abs__"]).to_numpy(dtype=np.float64)
    sl_abs = _safe_numeric(base["__first_touch_effective_sl_abs__"]).to_numpy(dtype=np.float64)
    net = np.full(n, np.nan, dtype=np.float64)
    gross = np.full(n, np.nan, dtype=np.float64)
    exit_bars = np.full(n, float(hold), dtype=np.float64)
    reason = np.full(n, f"timeout_close_{hold}", dtype=object)
    active = finite.copy()
    for j in range(hold):
        active_idx = np.flatnonzero(active)
        if not len(active_idx):
            break
        hi = highs[active_idx, j].astype(np.float64, copy=False)
        lo = lows[active_idx, j].astype(np.float64, copy=False)
        op = opens[active_idx, j].astype(np.float64, copy=False)
        ent = entry[active_idx]
        s = side[active_idx]
        tp = tp_abs[active_idx]
        sl = sl_abs[active_idx]
        tp_hit = np.where(s >= 0.0, hi >= ent * (1.0 + tp), lo <= ent * (1.0 - tp))
        tp_hit &= (j + 1) <= int(max_tp_bars)
        sl_hit = np.where(s >= 0.0, lo <= ent * (1.0 - sl), hi >= ent * (1.0 + sl))
        decision = _same_bar_decision(
            bar_open=op,
            entry=ent,
            side=s,
            tp_abs=tp,
            sl_abs=sl,
            tp_hit=tp_hit,
            sl_hit=sl_hit,
        )
        decided = active_idx[decision != 0]
        if not len(decided):
            continue
        hit = decided[decision[decision != 0] > 0]
        stopped = decided[decision[decision != 0] < 0]
        if len(hit):
            gross[hit] = tp_abs[hit]
            net[hit] = tp_abs[hit] - float(round_trip_cost)
            reason[hit] = "tp_first_touch"
            exit_bars[hit] = float(j + 1)
        if len(stopped):
            gross[stopped] = -sl_abs[stopped]
            net[stopped] = -sl_abs[stopped] - float(round_trip_cost)
            reason[stopped] = "sl_first_touch"
            exit_bars[stopped] = float(j + 1)
        active[decided] = False
    if np.any(active):
        idx = hold - 1
        gross[active] = close_ret[active, idx]
        net[active] = gross[active] - float(round_trip_cost)
    policy_name = f"tp_sl_hold_{hold}_tpmax_{int(max_tp_bars)}"
    return _finalize_policy_frame(
        base=base,
        policy_name=policy_name,
        policy_family="tp_sl_time_stop",
        net=net,
        gross=gross,
        exit_bars=exit_bars,
        exit_reason=reason,
        finite_path=finite & np.isfinite(net),
        fav=fav,
        adv=adv,
        tp_abs=tp_abs,
        sl_abs=sl_abs,
    )


def _activation_mult_at_bar(spec: TrailSpec, bar_num: int) -> float:
    if spec.decay_half_life_bars <= 0.0 or spec.min_activation_mult >= spec.activation_mult:
        return float(spec.activation_mult)
    decay_bars = max(0.0, float(bar_num - int(spec.decay_start_bars)))
    if decay_bars <= 0.0:
        return float(spec.activation_mult)
    decay = 0.5 ** (decay_bars / float(spec.decay_half_life_bars))
    return float(spec.min_activation_mult + (spec.activation_mult - spec.min_activation_mult) * decay)


def _trailing_policy(
    base: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    spec: TrailSpec,
    round_trip_cost: float,
) -> pd.DataFrame:
    opens, highs, lows, _closes = paths
    side = _side_array(base)
    entry, fav, adv, close_ret, _close_px, finite = _entry_and_path_returns(paths, side)
    n = len(base)
    max_len = close_ret.shape[1]
    hold = min(max(1, int(spec.hold_bars)), max_len)
    barrier = _safe_numeric(base["__barrier_pct__"]).to_numpy(dtype=np.float64)
    tp_abs = _safe_numeric(base["__first_touch_effective_tp_abs__"]).to_numpy(dtype=np.float64)
    sl_abs = _safe_numeric(base["__first_touch_effective_sl_abs__"]).to_numpy(dtype=np.float64)
    net = np.full(n, np.nan, dtype=np.float64)
    gross = np.full(n, np.nan, dtype=np.float64)
    exit_bars = np.full(n, float(hold), dtype=np.float64)
    reason = np.full(n, f"timeout_close_{hold}", dtype=object)
    active = finite.copy()
    max_fav_prev = np.zeros(n, dtype=np.float64)

    for j in range(hold):
        active_idx = np.flatnonzero(active)
        if not len(active_idx):
            break
        hi = highs[active_idx, j].astype(np.float64, copy=False)
        lo = lows[active_idx, j].astype(np.float64, copy=False)
        ent = entry[active_idx]
        s = side[active_idx]
        sl = sl_abs[active_idx]

        sl_hit = np.where(s >= 0.0, lo <= ent * (1.0 - sl), hi >= ent * (1.0 + sl))
        stopped = active_idx[sl_hit]
        if len(stopped):
            gross[stopped] = -sl_abs[stopped]
            net[stopped] = -sl_abs[stopped] - float(round_trip_cost)
            exit_bars[stopped] = float(j + 1)
            reason[stopped] = "full_sl"
            active[stopped] = False

        active_idx = np.flatnonzero(active)
        if not len(active_idx):
            break
        hi = highs[active_idx, j].astype(np.float64, copy=False)
        lo = lows[active_idx, j].astype(np.float64, copy=False)
        ent = entry[active_idx]
        s = side[active_idx]
        activation_ret = barrier[active_idx] * _activation_mult_at_bar(spec, j + 1)
        trail_active = max_fav_prev[active_idx] >= activation_ret
        trail_ret = max_fav_prev[active_idx] * (1.0 - float(spec.giveback_frac))
        trail_ret = np.maximum(trail_ret, 0.0)
        trail_hit = np.where(s >= 0.0, lo <= ent * (1.0 + trail_ret), hi >= ent * (1.0 - trail_ret))
        trail_hit &= trail_active
        trailed = active_idx[trail_hit]
        if len(trailed):
            gross[trailed] = trail_ret[trail_hit]
            net[trailed] = gross[trailed] - float(round_trip_cost)
            exit_bars[trailed] = float(j + 1)
            reason[trailed] = "trailing"
            active[trailed] = False

        still = np.flatnonzero(active)
        if len(still):
            # Match simple_policy semantics: this bar's favorable excursion can
            # only promote the trailing level for later bars.
            max_fav_prev[still] = np.maximum(max_fav_prev[still], fav[still, j])

    if np.any(active):
        idx = hold - 1
        gross[active] = close_ret[active, idx]
        net[active] = gross[active] - float(round_trip_cost)
    return _finalize_policy_frame(
        base=base,
        policy_name=spec.name,
        policy_family="trailing",
        net=net,
        gross=gross,
        exit_bars=exit_bars,
        exit_reason=reason,
        finite_path=finite & np.isfinite(net),
        fav=fav,
        adv=adv,
        tp_abs=tp_abs,
        sl_abs=sl_abs,
    )


def _label_policy_frame(base: pd.DataFrame) -> pd.DataFrame:
    net = _safe_numeric(base.get("first_touch_net"), index=base.index).to_numpy(dtype=np.float64)
    gross = net + float(ROUND_TRIP_COST)
    exit_bars = _safe_numeric(base.get("__first_touch_bar__"), index=base.index).fillna(96.0).to_numpy(dtype=np.float64)
    reason = np.full(len(base), "label_first_touch", dtype=object)
    hit = _safe_numeric(base.get("__first_touch_hit__"), index=base.index).fillna(0.0).to_numpy(dtype=np.float64) > 0.5
    stop = _safe_numeric(base.get("__first_touch_stop__"), index=base.index).fillna(0.0).to_numpy(dtype=np.float64) > 0.5
    timeout = _safe_numeric(base.get("__first_touch_timeout__"), index=base.index).fillna(0.0).to_numpy(dtype=np.float64) > 0.5
    reason[hit] = "tp_first_touch"
    reason[stop] = "sl_first_touch"
    reason[timeout] = "timeout_close_96"
    tp_abs = _safe_numeric(base["__first_touch_effective_tp_abs__"]).to_numpy(dtype=np.float64)
    sl_abs = _safe_numeric(base["__first_touch_effective_sl_abs__"]).to_numpy(dtype=np.float64)
    out = pd.DataFrame(
        {
            "policy": "label_first_touch_96",
            "policy_family": "label",
            "__ts__": base["__ts__"].to_numpy(copy=False),
            "__symbol__": base["__symbol__"].astype(str).to_numpy(copy=False),
            "month": base["month"].astype(str).to_numpy(copy=False),
            "week": base["week"].astype(str).to_numpy(copy=False),
            "side": _side_array(base),
            "barrier_pct": _safe_numeric(base["__barrier_pct__"]).to_numpy(dtype=np.float64),
            "tp_abs": tp_abs,
            "sl_abs": sl_abs,
            "finite_path": np.ones(len(base), dtype=float),
            "net_return": net,
            "gross_return": gross,
            "round_trip_cost": float(ROUND_TRIP_COST),
            "exit_bars": exit_bars,
            "exit_hours": exit_bars * 0.25,
            "exit_reason": reason,
            "mfe_to_tp_until_exit": _safe_numeric(
                base.get("__first_touch_full_path_mfe_to_tp__"),
                index=base.index,
            ).to_numpy(dtype=np.float64),
            "mae_to_sl_until_exit": _safe_numeric(
                base.get("__first_touch_full_path_mae_to_sl__"),
                index=base.index,
            ).to_numpy(dtype=np.float64),
            "max_favorable_return_until_exit": np.nan,
            "max_adverse_return_until_exit": np.nan,
            "peak_giveback_return": np.nan,
            "peak_giveback_to_tp": np.nan,
            "label_first_touch_net": net,
            "label_first_touch_bar": exit_bars,
            "label_full_path_mae_to_sl": _safe_numeric(
                base.get("__first_touch_full_path_mae_to_sl__"),
                index=base.index,
            ).to_numpy(dtype=np.float64),
        }
    )
    return out


def _build_policy_rows(
    base: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    hold_bars: list[int],
    round_trip_cost: float,
    max_barrier: float,
    include_contract_variants: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = [_label_policy_frame(base)]
    for bars in hold_bars:
        frames.append(_fixed_hold_policy(base, paths, hold_bars=bars, round_trip_cost=round_trip_cost))
    for bars in hold_bars:
        frames.append(
            _tp_sl_hold_policy(
                base,
                paths,
                hold_bars=bars,
                max_tp_bars=6,
                round_trip_cost=round_trip_cost,
            )
        )
    trail_specs = [
        TrailSpec("trail_static_act075_gb35_hold24", 24, 0.75, 0.75, 0.35, 0.0, 0),
        TrailSpec("trail_static_act075_gb50_hold24", 24, 0.75, 0.75, 0.50, 0.0, 0),
        TrailSpec("trail_static_act075_gb35_hold48", 48, 0.75, 0.75, 0.35, 0.0, 0),
        TrailSpec("trail_static_act075_gb35_hold96", 96, 0.75, 0.75, 0.35, 0.0, 0),
        TrailSpec("trail_decay_act075_min040_gb35_hold24", 24, 0.75, 0.40, 0.35, 4.0, 4),
        TrailSpec("trail_decay_act075_min040_gb35_hold48", 48, 0.75, 0.40, 0.35, 4.0, 4),
        TrailSpec("trail_decay_act075_min040_gb35_hold96", 96, 0.75, 0.40, 0.35, 4.0, 4),
    ]
    for spec in trail_specs:
        frames.append(_trailing_policy(base, paths, spec=spec, round_trip_cost=round_trip_cost))
    policy_rows = pd.concat(frames, ignore_index=True)
    if not include_contract_variants:
        return policy_rows
    replay_rows = policy_rows[policy_rows["policy_family"] != "label"].copy()
    if replay_rows.empty:
        return policy_rows
    ineligible = _safe_numeric(replay_rows["barrier_pct"]) > float(max_barrier)
    replay_rows["policy"] = "contract_" + replay_rows["policy"].astype(str)
    replay_rows["policy_family"] = "contract_" + replay_rows["policy_family"].astype(str)
    replay_rows.loc[ineligible, "net_return"] = -float(round_trip_cost)
    replay_rows.loc[ineligible, "gross_return"] = 0.0
    replay_rows.loc[ineligible, "exit_bars"] = 0.0
    replay_rows.loc[ineligible, "exit_hours"] = 0.0
    replay_rows.loc[ineligible, "exit_reason"] = "ineligible_barrier"
    replay_rows.loc[ineligible, "mae_to_sl_until_exit"] = np.nan
    replay_rows.loc[ineligible, "mfe_to_tp_until_exit"] = np.nan
    replay_rows.loc[ineligible, "max_favorable_return_until_exit"] = np.nan
    replay_rows.loc[ineligible, "max_adverse_return_until_exit"] = np.nan
    replay_rows.loc[ineligible, "peak_giveback_return"] = np.nan
    replay_rows.loc[ineligible, "peak_giveback_to_tp"] = np.nan
    return pd.concat([policy_rows, replay_rows], ignore_index=True)


def _exit_reason_rates(group: pd.DataFrame) -> dict[str, float]:
    reason = group["exit_reason"].astype(str)
    return {
        "tp_rate": float(reason.eq("tp_first_touch").mean()) if len(reason) else float("nan"),
        "sl_rate": float((reason.eq("sl_first_touch") | reason.eq("full_sl")).mean()) if len(reason) else float("nan"),
        "trail_rate": float(reason.eq("trailing").mean()) if len(reason) else float("nan"),
        "timeout_rate": float(reason.str.startswith("timeout").mean()) if len(reason) else float("nan"),
        "fixed_exit_rate": float(reason.str.startswith("fixed_hold").mean()) if len(reason) else float("nan"),
        "ineligible_barrier_rate": float(reason.eq("ineligible_barrier").mean()) if len(reason) else float("nan"),
    }


def _summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for values, group in frame.groupby(keys, observed=True, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        net = _safe_numeric(group["net_return"])
        row = {key: str(value) for key, value in zip(keys, values)}
        row.update(
            {
                "rows": int(len(group)),
                "finite_rows": int(_safe_numeric(group["finite_path"]).ge(0.5).sum()),
                "sum_net": _safe_sum(net),
                "mean_net": _safe_mean(net),
                "median_net": _safe_quantile(net, 0.50),
                "q10_net": _safe_quantile(net, 0.10),
                "win_rate": _safe_mean(net > 0.0),
                "exit_bars_p50": _safe_quantile(group["exit_bars"], 0.50),
                "exit_bars_p90": _safe_quantile(group["exit_bars"], 0.90),
                "exit_hours_p90": _safe_quantile(group["exit_hours"], 0.90),
                "mae_to_sl_p90": _safe_quantile(group["mae_to_sl_until_exit"], 0.90),
                "mfe_to_tp_p90": _safe_quantile(group["mfe_to_tp_until_exit"], 0.90),
                "peak_giveback_to_tp_p90": _safe_quantile(group["peak_giveback_to_tp"], 0.90),
            }
        )
        row.update(_exit_reason_rates(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_with_week_stability(policy_rows: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    agg = _summarize(policy_rows, ["policy", "policy_family"])
    week_stats: list[dict[str, Any]] = []
    for policy, group in weekly.groupby("policy", observed=True, sort=True):
        net = _safe_numeric(group["sum_net"])
        week_stats.append(
            {
                "policy": str(policy),
                "weeks": int(len(group)),
                "positive_weeks": int((net > 0.0).sum()),
                "positive_week_rate": _safe_mean(net > 0.0),
                "worst_week_sum_net": float(net.min()) if len(net.dropna()) else float("nan"),
            }
        )
    stable = pd.DataFrame(week_stats)
    if stable.empty:
        return agg
    return agg.merge(stable, on="policy", how="left")


def _write_markdown(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    aggregate: pd.DataFrame,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    alignment: pd.DataFrame,
) -> Path:
    path = output_dir / "stage173_stage167_selected_exit_replay.md"
    agg_cols = [
        "policy",
        "policy_family",
        "rows",
        "sum_net",
        "mean_net",
        "q10_net",
        "win_rate",
        "positive_week_rate",
        "worst_week_sum_net",
        "exit_bars_p50",
        "exit_bars_p90",
        "mae_to_sl_p90",
        "peak_giveback_to_tp_p90",
        "tp_rate",
        "sl_rate",
        "trail_rate",
        "timeout_rate",
    ]
    month_cols = [
        "month",
        "policy",
        "sum_net",
        "mean_net",
        "win_rate",
        "exit_bars_p50",
        "exit_bars_p90",
        "mae_to_sl_p90",
        "peak_giveback_to_tp_p90",
        "tp_rate",
        "sl_rate",
        "trail_rate",
        "timeout_rate",
    ]
    week_cols = [
        "week",
        "policy",
        "rows",
        "sum_net",
        "mean_net",
        "win_rate",
        "exit_bars_p90",
        "mae_to_sl_p90",
    ]
    focus_policies = [
        "label_first_touch_96",
        "tp_sl_hold_6_tpmax_6",
        "tp_sl_hold_12_tpmax_6",
        "tp_sl_hold_24_tpmax_6",
        "fixed_hold_6",
        "fixed_hold_12",
        "trail_static_act075_gb35_hold24",
        "trail_decay_act075_min040_gb35_hold24",
        "contract_tp_sl_hold_6_tpmax_6",
        "contract_tp_sl_hold_12_tpmax_6",
        "contract_tp_sl_hold_24_tpmax_6",
        "contract_trail_static_act075_gb35_hold24",
        "contract_trail_decay_act075_min040_gb35_hold24",
    ]
    focus_monthly = monthly[monthly["policy"].isin(focus_policies)].copy()
    focus_weekly = weekly[weekly["policy"].isin(focus_policies[:4])].copy()
    lines = [
        "# Stage173 Stage167 Selected Exit Replay",
        "",
        "Scope: actual delayed 15m candle-path replay on the same Stage167 selected rows. This is not a new model and not a portfolio concurrency simulation.",
        "",
        f"Selected ledger: `{manifest['ledger_csv']}`",
        f"Labels: `{manifest['labels_path']}`",
        f"Data root: `{manifest['data_root']}`",
        f"Market mode: `{manifest['market_mode']}`",
        f"Exchange: `{manifest['exchange']}`",
        f"Path length: `{manifest['path_len']}` 15m bars",
        f"Delayed entry: `{manifest['apply_delayed_entry']}`",
        f"Round-trip cost: `{manifest['round_trip_cost']:.6f}`",
        f"Contract max barrier: `{manifest['max_barrier']:.4f}`",
        f"Contract variants included: `{manifest['include_contract_variants']}`",
        f"Finite path coverage: `{manifest['path_fetch']['finite_path_coverage']:.2%}`",
        "",
        "Convention: fixed-hold and TP/SL hold policies close on the delayed-entry 15m path. `tp_sl_hold_N_tpmax_6` allows TP only through bar 6, keeps the original SL, and closes at bar N if no touch. Trailing variants use prior-bar favorable excursion for trail activation, then close on SL, trail, or time stop.",
        "",
        "## Aggregate Policies",
        "",
        _table(aggregate.sort_values(["sum_net", "mean_net"], ascending=[False, False]), agg_cols, limit=80),
        "",
        "## Focus Monthly Policies",
        "",
        _table(focus_monthly, month_cols, limit=160),
        "",
        "## Focus Weekly Policies",
        "",
        _table(focus_weekly, week_cols, limit=240),
        "",
        "## Replay Alignment",
        "",
        _table(alignment, list(alignment.columns), limit=40),
        "",
        "## Outputs",
        "",
    ]
    for key, value in manifest["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_replay(
    *,
    ledger_csv: Path,
    labels_path: Path,
    output_dir: Path,
    data_root: Path,
    market_mode: str,
    exchange: str,
    months: list[str],
    hold_bars: list[int],
    path_len: int,
    apply_delayed_entry: bool,
    round_trip_cost: float,
    max_barrier: float,
    include_contract_variants: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = _load_ledger(ledger_csv)
    labels = _load_label_subset(labels_path)
    selected = _join_inputs(ledger, labels, months)
    _rows_exec, paths, path_stats = _fetch_paths(
        selected,
        labels_path=labels_path,
        data_root=data_root,
        market_mode=market_mode,
        exchange=exchange,
        path_len=path_len,
        apply_delayed_entry=apply_delayed_entry,
    )
    policy_rows = _build_policy_rows(
        selected,
        paths,
        hold_bars=hold_bars,
        round_trip_cost=round_trip_cost,
        max_barrier=max_barrier,
        include_contract_variants=include_contract_variants,
    )
    monthly = _summarize(policy_rows, ["month", "policy", "policy_family"])
    weekly = _summarize(policy_rows, ["week", "policy", "policy_family"])
    aggregate = _aggregate_with_week_stability(policy_rows, weekly)

    replay_policy_name = (
        f"contract_tp_sl_hold_{int(path_len)}_tpmax_6"
        if include_contract_variants
        else f"tp_sl_hold_{int(path_len)}_tpmax_6"
    )
    replay96 = policy_rows[policy_rows["policy"] == replay_policy_name].copy()
    label = policy_rows[policy_rows["policy"] == "label_first_touch_96"].copy()
    alignment = pd.DataFrame()
    if not replay96.empty and len(replay96) == len(label):
        diff = replay96["net_return"].reset_index(drop=True) - label["net_return"].reset_index(drop=True)
        bar_diff = replay96["exit_bars"].reset_index(drop=True) - label["exit_bars"].reset_index(drop=True)
        alignment = pd.DataFrame(
            [
                {
                    "comparison": "replay_tp_sl_hold_path_len_vs_label_first_touch",
                    "replay_policy": replay_policy_name,
                    "rows": int(len(diff)),
                    "max_abs_net_diff": float(diff.abs().max()),
                    "mean_abs_net_diff": float(diff.abs().mean()),
                    "max_abs_exit_bar_diff": float(bar_diff.abs().max()),
                    "mean_abs_exit_bar_diff": float(bar_diff.abs().mean()),
                }
            ]
        )

    paths_out = {
        "selected": output_dir / "stage173_selected_rows.csv",
        "policy_rows": output_dir / "stage173_policy_rows.csv",
        "aggregate": output_dir / "stage173_policy_aggregate.csv",
        "monthly": output_dir / "stage173_policy_monthly.csv",
        "weekly": output_dir / "stage173_policy_weekly.csv",
        "alignment": output_dir / "stage173_replay_alignment.csv",
        "manifest": output_dir / "manifest.json",
    }
    selected.to_csv(paths_out["selected"], index=False)
    policy_rows.to_csv(paths_out["policy_rows"], index=False)
    aggregate.to_csv(paths_out["aggregate"], index=False)
    monthly.to_csv(paths_out["monthly"], index=False)
    weekly.to_csv(paths_out["weekly"], index=False)
    alignment.to_csv(paths_out["alignment"], index=False)

    manifest = {
        "scope": "stage173_stage167_selected_exit_replay",
        "ledger_csv": str(ledger_csv),
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "data_root": str(data_root),
        "market_mode": str(market_mode),
        "exchange": str(exchange),
        "months": list(months),
        "rows": int(len(selected)),
        "hold_bars": [int(v) for v in hold_bars],
        "path_len": int(path_len),
        "apply_delayed_entry": bool(apply_delayed_entry),
        "round_trip_cost": float(round_trip_cost),
        "max_barrier": float(max_barrier),
        "include_contract_variants": bool(include_contract_variants),
        "contract_ineligible_rows": int((_safe_numeric(selected["__barrier_pct__"]) > float(max_barrier)).sum()),
        "path_fetch": path_stats,
        "outputs": {key: str(value) for key, value in paths_out.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        manifest=manifest,
        aggregate=aggregate,
        monthly=monthly,
        weekly=weekly,
        alignment=alignment,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths_out["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-csv", type=Path, default=DEFAULT_LEDGER_CSV)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--exchange", default="krakenfutures")
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--hold-bars", default=",".join(str(v) for v in DEFAULT_HOLD_BARS))
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--apply-delayed-entry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--round-trip-cost", type=float, default=ROUND_TRIP_COST)
    parser.add_argument("--max-barrier", type=float, default=DEFAULT_MAX_BARRIER)
    parser.add_argument("--include-contract-variants", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_replay(
        ledger_csv=args.ledger_csv,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        data_root=args.data_root,
        market_mode=args.market_mode,
        exchange=args.exchange,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        hold_bars=_parse_int_csv(args.hold_bars, DEFAULT_HOLD_BARS),
        path_len=args.path_len,
        apply_delayed_entry=bool(args.apply_delayed_entry),
        round_trip_cost=float(args.round_trip_cost),
        max_barrier=float(args.max_barrier),
        include_contract_variants=bool(args.include_contract_variants),
    )
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
