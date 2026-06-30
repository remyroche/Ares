#!/usr/bin/env python3
"""Materialize pre-head symbol-guard candidate ledgers for dynamic HR ablations.

The guards are intentionally applied before per-head dynamic threshold replay:
hard-guard variants remove candidate rows, while the soft variant lowers the
rank/score columns so the downstream dynamic threshold path sees a per-symbol
threshold raise without changing the existing replay code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration dependency
    njit = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_dynamic_hr_surprise_threshold import (  # noqa: E402
    _infer_head,
    _spread_net_return_from_columns,
)


NS_PER_HOUR = 60 * 60 * 1_000_000_000
NS_PER_DAY = 24 * NS_PER_HOUR


VARIANTS = (
    "A0_current",
    "A1_loss_cooldown_3of4_24h",
    "A3_symbol_z_guard_7d_m5_zneg15",
    "A4_soft_raise_loss2_zneg125",
    "A7_shared_symbol_side_hybrid",
    "A7_shared_symbol_hybrid",
)


@dataclass(frozen=True)
class GuardConfig:
    variant: str
    scope: str
    mode: str
    loss_window: int = 4
    hard_loss_threshold: int = 3
    soft_loss_threshold: int = 2
    cooldown_hours: float = 24.0
    lookback_days: float = 7.0
    z_min_count: int = 5
    hard_z_threshold: float = -1.50
    severe_z_threshold: float = -2.00
    soft_z_threshold: float = -1.25
    soft_penalty: float = 0.05
    severe_penalty: float = 0.10


def _variant_config(name: str, args: argparse.Namespace) -> GuardConfig:
    common = dict(
        loss_window=int(args.loss_window),
        hard_loss_threshold=int(args.loss_threshold),
        soft_loss_threshold=int(args.soft_loss_threshold),
        cooldown_hours=float(args.cooldown_hours),
        lookback_days=float(args.lookback_days),
        z_min_count=int(args.z_min_count),
        hard_z_threshold=float(args.z_threshold),
        severe_z_threshold=float(args.severe_z_threshold),
        soft_z_threshold=float(args.soft_z_threshold),
        soft_penalty=float(args.soft_penalty),
        severe_penalty=float(args.severe_penalty),
    )
    if name == "A0_current":
        return GuardConfig(name, scope="none", mode="none", **common)
    if name == "A1_loss_cooldown_3of4_24h":
        return GuardConfig(name, scope="head_symbol_side", mode="loss_cooldown", **common)
    if name == "A3_symbol_z_guard_7d_m5_zneg15":
        return GuardConfig(name, scope="head_symbol_side", mode="z_guard", **common)
    if name == "A4_soft_raise_loss2_zneg125":
        return GuardConfig(name, scope="head_symbol_side", mode="soft_raise", **common)
    if name == "A7_shared_symbol_side_hybrid":
        return GuardConfig(name, scope="symbol_side", mode="hybrid_hard", **common)
    if name == "A7_shared_symbol_hybrid":
        return GuardConfig(name, scope="symbol", mode="hybrid_hard", **common)
    raise ValueError(f"Unknown variant: {name}")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    return value


def _scope_columns(scope: str) -> list[str]:
    if scope == "head_symbol_side":
        return ["head", "symbol", "side"]
    if scope == "symbol_side":
        return ["symbol", "side"]
    if scope == "symbol":
        return ["symbol"]
    if scope == "none":
        return []
    raise ValueError(f"Unknown guard scope: {scope}")


def _read_candidates(path: Path, *, return_col: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    if "timestamp" not in frame.columns:
        raise ValueError("Candidate ledger must include timestamp")
    if "head" not in frame.columns:
        if "strategy_id" not in frame.columns:
            raise ValueError("Candidate ledger must include head or strategy_id")
        frame["head"] = frame["strategy_id"].map(_infer_head)
    for col in ("symbol", "side"):
        if col not in frame.columns:
            raise ValueError(f"Candidate ledger must include {col!r} for symbol guards")
    for col in ("normalized_rank_score", "policy_rank_pct", "calibrated_score"):
        if col not in frame.columns:
            raise ValueError(f"Candidate ledger must include {col!r}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "head", "symbol", "side"]).copy()
    frame["head"] = frame["head"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["side"] = frame["side"].astype(str)
    frame["__row_pos"] = np.arange(len(frame), dtype=np.int64)
    frame["__rank_for_guard"] = pd.to_numeric(frame["policy_rank_pct"], errors="coerce")
    frame["__p_hit_for_guard"] = pd.to_numeric(frame["calibrated_score"], errors="coerce").clip(1e-6, 1.0 - 1e-6)
    spread_net, spread_diag = _spread_net_return_from_columns(frame, return_col=return_col)
    frame["__guard_net_return"] = spread_net
    frame["__guard_hit"] = frame["__guard_net_return"].gt(0.0).astype(float)
    frame.attrs["spread_return_diagnostics"] = spread_diag
    return frame.sort_values(["timestamp", "head", "symbol", "side"]).reset_index(drop=True)


def _loss_state(history: pd.DataFrame, key_cols: list[str], cfg: GuardConfig) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=[*key_cols, "losses_last_n", "recent_n", "last_seen_ts"])
    ordered = history.sort_values([*key_cols, "timestamp"])
    recent = ordered.groupby(key_cols, sort=False).tail(int(cfg.loss_window)).copy()
    recent["loss"] = recent["__guard_net_return"].le(0.0).astype(int)
    out = (
        recent.groupby(key_cols, sort=False)
        .agg(
            losses_last_n=("loss", "sum"),
            recent_n=("loss", "size"),
            last_seen_ts=("timestamp", "max"),
        )
        .reset_index()
    )
    return out


def _z_state(history: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=[*key_cols, "z_count", "z_num", "z_var", "z_symbol"])
    work = history.copy()
    hit = pd.to_numeric(work["__guard_hit"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    p_hit = pd.to_numeric(work["__p_hit_for_guard"], errors="coerce").fillna(0.5).clip(1e-6, 1.0 - 1e-6).to_numpy(dtype=float)
    work["z_num_component"] = hit - p_hit
    work["z_var_component"] = p_hit * (1.0 - p_hit)
    grouped = (
        work.groupby(key_cols, sort=False)
        .agg(
            z_count=("z_num_component", "size"),
            z_num=("z_num_component", "sum"),
            z_var=("z_var_component", "sum"),
        )
        .reset_index()
    )
    grouped["z_symbol"] = grouped["z_num"] / np.sqrt(grouped["z_var"].clip(lower=1e-12))
    grouped["z_symbol"] = grouped["z_symbol"].replace([np.inf, -np.inf], np.nan)
    return grouped


if njit is not None:

    @njit(cache=True)
    def _compute_guard_state_arrays_numba(
        ts_ns: np.ndarray,
        key_codes: np.ndarray,
        ranks: np.ndarray,
        net_returns: np.ndarray,
        p_hits: np.ndarray,
        day_starts_ns: np.ndarray,
        day_row_starts: np.ndarray,
        day_row_ends: np.ndarray,
        n_keys: int,
        loss_window: int,
        hard_loss_threshold: int,
        soft_loss_threshold: int,
        z_min_count: int,
        hard_z_threshold: float,
        severe_z_threshold: float,
        soft_z_threshold: float,
        top_rank_floor: float,
        cooldown_ns: int,
        lookback_ns: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_rows = ts_ns.shape[0]
        losses_out = np.zeros(n_rows, dtype=np.int64)
        recent_out = np.zeros(n_rows, dtype=np.int64)
        z_count_out = np.zeros(n_rows, dtype=np.int64)
        z_out = np.empty(n_rows, dtype=np.float64)
        z_out[:] = np.nan
        hard_loss_out = np.zeros(n_rows, dtype=np.bool_)
        soft_loss_out = np.zeros(n_rows, dtype=np.bool_)
        hard_z_out = np.zeros(n_rows, dtype=np.bool_)
        severe_z_out = np.zeros(n_rows, dtype=np.bool_)
        soft_z_out = np.zeros(n_rows, dtype=np.bool_)

        loss_ring = np.zeros((n_keys, loss_window), dtype=np.int64)
        loss_pos = np.zeros(n_keys, dtype=np.int64)
        recent_n = np.zeros(n_keys, dtype=np.int64)
        loss_sum = np.zeros(n_keys, dtype=np.int64)
        last_seen_ns = np.empty(n_keys, dtype=np.int64)
        last_seen_ns[:] = -9223372036854775807

        z_count = np.zeros(n_keys, dtype=np.int64)
        z_num = np.zeros(n_keys, dtype=np.float64)
        z_var = np.zeros(n_keys, dtype=np.float64)

        add_idx = 0
        remove_idx = 0
        n_days = day_starts_ns.shape[0]

        for day_i in range(n_days):
            day_start = day_starts_ns[day_i]
            while add_idx < n_rows and ts_ns[add_idx] < day_start:
                if ranks[add_idx] >= top_rank_floor:
                    key = key_codes[add_idx]
                    loss_value = 1 if net_returns[add_idx] <= 0.0 else 0
                    pos = loss_pos[key]
                    if recent_n[key] < loss_window:
                        recent_n[key] += 1
                    else:
                        loss_sum[key] -= loss_ring[key, pos]
                    loss_ring[key, pos] = loss_value
                    loss_sum[key] += loss_value
                    loss_pos[key] = (pos + 1) % loss_window
                    last_seen_ns[key] = ts_ns[add_idx]

                    p_hit = p_hits[add_idx]
                    if not np.isfinite(p_hit):
                        p_hit = 0.5
                    if p_hit < 1e-6:
                        p_hit = 1e-6
                    elif p_hit > 1.0 - 1e-6:
                        p_hit = 1.0 - 1e-6
                    hit_value = 1.0 if net_returns[add_idx] > 0.0 else 0.0
                    z_count[key] += 1
                    z_num[key] += hit_value - p_hit
                    z_var[key] += p_hit * (1.0 - p_hit)
                add_idx += 1

            cutoff = day_start - lookback_ns
            while remove_idx < add_idx and ts_ns[remove_idx] < cutoff:
                if ranks[remove_idx] >= top_rank_floor:
                    key = key_codes[remove_idx]
                    p_hit = p_hits[remove_idx]
                    if not np.isfinite(p_hit):
                        p_hit = 0.5
                    if p_hit < 1e-6:
                        p_hit = 1e-6
                    elif p_hit > 1.0 - 1e-6:
                        p_hit = 1.0 - 1e-6
                    hit_value = 1.0 if net_returns[remove_idx] > 0.0 else 0.0
                    z_count[key] -= 1
                    z_num[key] -= hit_value - p_hit
                    z_var[key] -= p_hit * (1.0 - p_hit)
                    if z_count[key] < 0:
                        z_count[key] = 0
                        z_num[key] = 0.0
                        z_var[key] = 0.0
                remove_idx += 1

            start = day_row_starts[day_i]
            end = day_row_ends[day_i]
            for row_i in range(start, end):
                key = key_codes[row_i]
                losses = loss_sum[key]
                recent = recent_n[key]
                zc = z_count[key]
                z_value = np.nan
                if z_var[key] > 1e-12:
                    z_value = z_num[key] / np.sqrt(z_var[key])

                losses_out[row_i] = losses
                recent_out[row_i] = recent
                z_count_out[row_i] = zc
                z_out[row_i] = z_value

                cooldown_active = False
                if last_seen_ns[key] > -9223372036854775800:
                    cooldown_active = day_start - last_seen_ns[key] <= cooldown_ns

                hard_loss_out[row_i] = losses >= hard_loss_threshold and cooldown_active
                soft_loss_out[row_i] = losses >= soft_loss_threshold and cooldown_active
                if zc >= z_min_count and np.isfinite(z_value):
                    hard_z_out[row_i] = z_value <= hard_z_threshold
                    severe_z_out[row_i] = z_value <= severe_z_threshold
                    soft_z_out[row_i] = z_value <= soft_z_threshold

        return (
            losses_out,
            recent_out,
            z_count_out,
            z_out,
            hard_loss_out,
            soft_loss_out,
            hard_z_out,
            severe_z_out,
            soft_z_out,
        )


def _factorize_scope(frame: pd.DataFrame, key_cols: list[str]) -> np.ndarray:
    if len(key_cols) == 1:
        codes, _ = pd.factorize(frame[key_cols[0]].astype(str), sort=True)
        return codes.astype(np.int64, copy=False)
    key_series = frame[key_cols].astype(str).agg("\x1f".join, axis=1)
    codes, _ = pd.factorize(key_series, sort=True)
    return codes.astype(np.int64, copy=False)


def _guard_decisions_fast(
    frame: pd.DataFrame,
    cfg: GuardConfig,
    *,
    top_rank_floor: float,
) -> pd.DataFrame:
    if njit is None:
        return _guard_decisions_pandas(frame, cfg, top_rank_floor=top_rank_floor)
    key_cols = _scope_columns(cfg.scope)
    if not key_cols:
        out = frame[["__row_pos"]].copy()
        out["prehead_symbol_guard_blocked"] = False
        out["prehead_symbol_guard_penalty"] = 0.0
        out["prehead_symbol_guard_reason"] = "none"
        out["prehead_symbol_guard_losses_last_n"] = 0
        out["prehead_symbol_guard_recent_n"] = 0
        out["prehead_symbol_guard_z_count"] = 0
        out["prehead_symbol_guard_z"] = np.nan
        out["prehead_symbol_guard_variant"] = cfg.variant
        out["prehead_symbol_guard_scope"] = cfg.scope
        out["prehead_symbol_guard_mode"] = cfg.mode
        return out

    work = frame.sort_values(["timestamp", "head", "symbol", "side"]).reset_index(drop=True).copy()
    ts_ns = work["timestamp"].astype("int64").to_numpy(dtype=np.int64, copy=False)
    key_codes = _factorize_scope(work, key_cols)
    n_keys = int(key_codes.max()) + 1 if len(key_codes) else 0
    ranks = pd.to_numeric(work["__rank_for_guard"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    net_returns = pd.to_numeric(work["__guard_net_return"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    p_hits = (
        pd.to_numeric(work["__p_hit_for_guard"], errors="coerce")
        .fillna(0.5)
        .clip(1e-6, 1.0 - 1e-6)
        .to_numpy(dtype=np.float64, copy=False)
    )
    day_floor_ns = work["timestamp"].dt.floor("D").astype("int64").to_numpy(dtype=np.int64, copy=False)
    day_starts_ns, day_row_starts = np.unique(day_floor_ns, return_index=True)
    day_row_ends = np.r_[day_row_starts[1:], np.array([len(work)], dtype=np.int64)].astype(np.int64, copy=False)

    (
        losses,
        recent_n,
        z_count,
        z_symbol,
        hard_loss,
        soft_loss,
        hard_z,
        severe_z,
        soft_z,
    ) = _compute_guard_state_arrays_numba(
        ts_ns,
        key_codes,
        ranks,
        net_returns,
        p_hits,
        day_starts_ns.astype(np.int64, copy=False),
        day_row_starts.astype(np.int64, copy=False),
        day_row_ends.astype(np.int64, copy=False),
        n_keys,
        int(cfg.loss_window),
        int(cfg.hard_loss_threshold),
        int(cfg.soft_loss_threshold),
        int(cfg.z_min_count),
        float(cfg.hard_z_threshold),
        float(cfg.severe_z_threshold),
        float(cfg.soft_z_threshold),
        float(top_rank_floor),
        int(float(cfg.cooldown_hours) * NS_PER_HOUR),
        int(float(cfg.lookback_days) * NS_PER_DAY),
    )

    if cfg.mode == "loss_cooldown":
        blocked = hard_loss
        penalty = np.zeros(len(work), dtype=float)
    elif cfg.mode == "z_guard":
        blocked = z_count >= int(cfg.z_min_count)
        blocked &= z_symbol <= float(cfg.hard_z_threshold)
        penalty = np.zeros(len(work), dtype=float)
    elif cfg.mode == "hybrid_hard":
        blocked = hard_loss | ((z_count >= int(cfg.z_min_count)) & (z_symbol <= float(cfg.hard_z_threshold)))
        penalty = np.zeros(len(work), dtype=float)
    elif cfg.mode == "soft_raise":
        blocked = np.zeros(len(work), dtype=bool)
        severe = hard_loss | ((z_count >= int(cfg.z_min_count)) & (z_symbol <= float(cfg.severe_z_threshold)))
        soft = soft_loss | ((z_count >= int(cfg.z_min_count)) & (z_symbol <= float(cfg.soft_z_threshold)))
        penalty = np.where(severe, float(cfg.severe_penalty), np.where(soft, float(cfg.soft_penalty), 0.0))
    else:
        raise ValueError(f"Unsupported guard mode: {cfg.mode}")

    effective_hard_z = (z_count >= int(cfg.z_min_count)) & (z_symbol <= float(cfg.hard_z_threshold))
    effective_severe_z = (z_count >= int(cfg.z_min_count)) & (z_symbol <= float(cfg.severe_z_threshold))
    reason = np.full(len(work), "", dtype=object)
    reason = np.where(hard_loss, "loss_cooldown", reason)
    reason = np.where(effective_hard_z, np.where(reason == "", "symbol_z", reason + "+symbol_z"), reason)
    if cfg.mode == "soft_raise":
        reason = np.where((penalty > 0.0) & (reason == ""), "soft_symbol_weakness", reason)
        reason = np.where((penalty > 0.0) & hard_loss, "severe_loss_cooldown", reason)
        reason = np.where(
            (penalty > 0.0) & effective_severe_z,
            np.where(reason == "", "severe_symbol_z", reason + "+severe_symbol_z"),
            reason,
        )
    reason = np.where((~blocked) & (penalty <= 0.0) & (reason == ""), "pass", reason)

    out = pd.DataFrame(
        {
            "__row_pos": work["__row_pos"].to_numpy(dtype=np.int64, copy=False),
            "prehead_symbol_guard_blocked": blocked.astype(bool),
            "prehead_symbol_guard_penalty": penalty.astype(float),
            "prehead_symbol_guard_reason": reason,
            "prehead_symbol_guard_losses_last_n": losses.astype(int),
            "prehead_symbol_guard_recent_n": recent_n.astype(int),
            "prehead_symbol_guard_z_count": z_count.astype(int),
            "prehead_symbol_guard_z": z_symbol.astype(float),
            "prehead_symbol_guard_variant": cfg.variant,
            "prehead_symbol_guard_scope": cfg.scope,
            "prehead_symbol_guard_mode": cfg.mode,
        }
    )
    return out


def _decision_for_day(
    frame: pd.DataFrame,
    day_start: pd.Timestamp,
    cfg: GuardConfig,
    *,
    top_rank_floor: float,
) -> pd.DataFrame:
    day_end = day_start + pd.Timedelta(days=1)
    day_rows = frame.loc[frame["timestamp"].ge(day_start) & frame["timestamp"].lt(day_end)].copy()
    if day_rows.empty:
        return day_rows[["__row_pos"]].assign(
            prehead_symbol_guard_blocked=False,
            prehead_symbol_guard_penalty=0.0,
            prehead_symbol_guard_reason="",
        )
    key_cols = _scope_columns(cfg.scope)
    if not key_cols:
        return day_rows[["__row_pos"]].assign(
            prehead_symbol_guard_blocked=False,
            prehead_symbol_guard_penalty=0.0,
            prehead_symbol_guard_reason="none",
            prehead_symbol_guard_losses_last_n=0,
            prehead_symbol_guard_recent_n=0,
            prehead_symbol_guard_z_count=0,
            prehead_symbol_guard_z=np.nan,
        )

    eligible_history = frame.loc[
        frame["timestamp"].lt(day_start)
        & frame["__rank_for_guard"].ge(float(top_rank_floor))
    ].copy()
    loss = _loss_state(eligible_history, key_cols, cfg)
    lookback_start = day_start - pd.Timedelta(days=float(cfg.lookback_days))
    zhist = eligible_history.loc[eligible_history["timestamp"].ge(lookback_start)].copy()
    zstate = _z_state(zhist, key_cols)
    state = loss.merge(zstate, on=key_cols, how="outer", sort=False) if not loss.empty or not zstate.empty else pd.DataFrame(columns=key_cols)
    if state.empty:
        out = day_rows[["__row_pos"]].copy()
        out["prehead_symbol_guard_blocked"] = False
        out["prehead_symbol_guard_penalty"] = 0.0
        out["prehead_symbol_guard_reason"] = "no_prior_state"
        out["prehead_symbol_guard_losses_last_n"] = 0
        out["prehead_symbol_guard_recent_n"] = 0
        out["prehead_symbol_guard_z_count"] = 0
        out["prehead_symbol_guard_z"] = np.nan
        return out

    merged = day_rows[["__row_pos", *key_cols]].merge(state, on=key_cols, how="left", sort=False)
    losses = pd.to_numeric(merged.get("losses_last_n", 0), errors="coerce").fillna(0).astype(int)
    recent_n = pd.to_numeric(merged.get("recent_n", 0), errors="coerce").fillna(0).astype(int)
    z_count = pd.to_numeric(merged.get("z_count", 0), errors="coerce").fillna(0).astype(int)
    z_symbol = pd.to_numeric(merged.get("z_symbol", np.nan), errors="coerce")
    last_seen = pd.to_datetime(merged.get("last_seen_ts"), utc=True, errors="coerce")
    cooldown_active = (
        last_seen.notna()
        & ((day_start - last_seen) <= pd.Timedelta(hours=float(cfg.cooldown_hours)))
    )
    hard_loss = losses.ge(int(cfg.hard_loss_threshold)) & cooldown_active
    soft_loss = losses.ge(int(cfg.soft_loss_threshold)) & cooldown_active
    hard_z = z_count.ge(int(cfg.z_min_count)) & z_symbol.le(float(cfg.hard_z_threshold))
    severe_z = z_count.ge(int(cfg.z_min_count)) & z_symbol.le(float(cfg.severe_z_threshold))
    soft_z = z_count.ge(int(cfg.z_min_count)) & z_symbol.le(float(cfg.soft_z_threshold))

    if cfg.mode == "loss_cooldown":
        blocked = hard_loss
        penalty = np.zeros(len(merged), dtype=float)
    elif cfg.mode == "z_guard":
        blocked = hard_z
        penalty = np.zeros(len(merged), dtype=float)
    elif cfg.mode == "hybrid_hard":
        blocked = hard_loss | hard_z
        penalty = np.zeros(len(merged), dtype=float)
    elif cfg.mode == "soft_raise":
        blocked = pd.Series(False, index=merged.index)
        severe = hard_loss | severe_z
        soft = soft_loss | soft_z
        penalty = np.where(severe, float(cfg.severe_penalty), np.where(soft, float(cfg.soft_penalty), 0.0))
    else:
        raise ValueError(f"Unsupported guard mode: {cfg.mode}")

    reason = np.full(len(merged), "", dtype=object)
    reason = np.where(hard_loss, "loss_cooldown", reason)
    reason = np.where(hard_z, np.where(reason == "", "symbol_z", reason + "+symbol_z"), reason)
    if cfg.mode == "soft_raise":
        reason = np.where((penalty > 0.0) & (reason == ""), "soft_symbol_weakness", reason)
        reason = np.where((penalty > 0.0) & hard_loss, "severe_loss_cooldown", reason)
        reason = np.where((penalty > 0.0) & severe_z, np.where(reason == "", "severe_symbol_z", reason + "+severe_symbol_z"), reason)
    reason = np.where((~np.asarray(blocked, dtype=bool)) & (penalty <= 0.0) & (reason == ""), "pass", reason)

    out = merged[["__row_pos"]].copy()
    out["prehead_symbol_guard_blocked"] = np.asarray(blocked, dtype=bool)
    out["prehead_symbol_guard_penalty"] = penalty.astype(float)
    out["prehead_symbol_guard_reason"] = reason
    out["prehead_symbol_guard_losses_last_n"] = losses.to_numpy(dtype=int)
    out["prehead_symbol_guard_recent_n"] = recent_n.to_numpy(dtype=int)
    out["prehead_symbol_guard_z_count"] = z_count.to_numpy(dtype=int)
    out["prehead_symbol_guard_z"] = z_symbol.to_numpy(dtype=float)
    return out


def _guard_decisions_pandas(
    frame: pd.DataFrame,
    cfg: GuardConfig,
    *,
    top_rank_floor: float,
) -> pd.DataFrame:
    days = pd.date_range(
        pd.Timestamp(frame["timestamp"].min()).floor("D"),
        pd.Timestamp(frame["timestamp"].max()).ceil("D"),
        freq="D",
        tz="UTC",
    )
    parts: list[pd.DataFrame] = []
    for day in days:
        part = _decision_for_day(frame, pd.Timestamp(day), cfg, top_rank_floor=top_rank_floor)
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["__row_pos"])
    decisions = pd.concat(parts, ignore_index=True)
    decisions["prehead_symbol_guard_variant"] = cfg.variant
    decisions["prehead_symbol_guard_scope"] = cfg.scope
    decisions["prehead_symbol_guard_mode"] = cfg.mode
    return decisions


def _guard_decisions(
    frame: pd.DataFrame,
    cfg: GuardConfig,
    *,
    top_rank_floor: float,
    engine: str,
) -> pd.DataFrame:
    if engine == "pandas":
        return _guard_decisions_pandas(frame, cfg, top_rank_floor=top_rank_floor)
    if engine == "fast":
        if njit is None:
            raise RuntimeError("Numba is not installed; use --engine pandas")
        return _guard_decisions_fast(frame, cfg, top_rank_floor=top_rank_floor)
    if engine == "auto":
        return _guard_decisions_fast(frame, cfg, top_rank_floor=top_rank_floor)
    raise ValueError(f"Unknown engine: {engine}")


def _apply_blacklist_breadth_veto(
    frame: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    max_asset_fraction: float,
) -> pd.DataFrame:
    if decisions.empty:
        return decisions
    out = decisions.copy()
    out["prehead_symbol_guard_basket_asset_count"] = int(frame["symbol"].nunique())
    out["prehead_symbol_guard_blacklist_asset_count"] = 0
    out["prehead_symbol_guard_blacklist_asset_fraction"] = 0.0
    out["prehead_symbol_guard_market_wide_veto"] = False
    if max_asset_fraction <= 0.0 or max_asset_fraction >= 1.0:
        return out
    if "prehead_symbol_guard_blocked" not in out.columns:
        return out

    blocked = out["prehead_symbol_guard_blocked"].astype(bool)
    if not blocked.any():
        return out

    meta = frame[["__row_pos", "timestamp", "symbol"]].copy()
    meta["prehead_symbol_guard_day"] = meta["timestamp"].dt.floor("D")
    joined = out[["__row_pos", "prehead_symbol_guard_blocked"]].merge(
        meta,
        on="__row_pos",
        how="left",
        sort=False,
    )
    basket_asset_count = max(int(frame["symbol"].nunique()), 1)
    blocked_daily = (
        joined.loc[joined["prehead_symbol_guard_blocked"].astype(bool)]
        .groupby("prehead_symbol_guard_day", sort=True)
        .agg(
            prehead_symbol_guard_blacklist_asset_count=("symbol", "nunique"),
            prehead_symbol_guard_blacklist_row_count=("__row_pos", "size"),
        )
    )
    if blocked_daily.empty:
        return out
    blocked_daily["prehead_symbol_guard_blacklist_asset_fraction"] = (
        blocked_daily["prehead_symbol_guard_blacklist_asset_count"].astype(float) / float(basket_asset_count)
    )
    day_lookup = joined.set_index("__row_pos")["prehead_symbol_guard_day"]
    asset_count_lookup = blocked_daily["prehead_symbol_guard_blacklist_asset_count"]
    asset_fraction_lookup = blocked_daily["prehead_symbol_guard_blacklist_asset_fraction"]
    row_days = out["__row_pos"].map(day_lookup)
    out["prehead_symbol_guard_blacklist_asset_count"] = row_days.map(asset_count_lookup).fillna(0).astype(int)
    out["prehead_symbol_guard_blacklist_asset_fraction"] = row_days.map(asset_fraction_lookup).fillna(0.0).astype(float)

    veto_days = set(
        blocked_daily.index[
            blocked_daily["prehead_symbol_guard_blacklist_asset_fraction"].gt(float(max_asset_fraction))
        ]
    )
    if not veto_days:
        return out
    veto_mask = blocked & row_days.isin(veto_days).to_numpy(dtype=bool)
    if not veto_mask.any():
        return out
    out.loc[veto_mask, "prehead_symbol_guard_market_wide_veto"] = True
    out.loc[veto_mask, "prehead_symbol_guard_blocked"] = False
    existing_reason = out.loc[veto_mask, "prehead_symbol_guard_reason"].astype(str)
    out.loc[veto_mask, "prehead_symbol_guard_reason"] = np.where(
        existing_reason.eq("") | existing_reason.eq("pass"),
        "market_wide_blacklist_veto",
        existing_reason + "+market_wide_blacklist_veto",
    )
    return out


def _peer_columns_for_scope(scope: str) -> list[str]:
    if scope == "head_symbol_side":
        return ["prehead_symbol_guard_day", "head", "side"]
    if scope == "symbol_side":
        return ["prehead_symbol_guard_day", "side"]
    if scope == "symbol":
        return ["prehead_symbol_guard_day"]
    return ["prehead_symbol_guard_day"]


def _apply_relative_symbol_weakness_filter(
    frame: pd.DataFrame,
    decisions: pd.DataFrame,
    cfg: GuardConfig,
    *,
    min_peer_symbols: int,
    z_peer_quantile: float,
    z_margin: float,
    loss_peer_quantile: float,
    loss_margin: float,
) -> pd.DataFrame:
    if decisions.empty:
        return decisions
    out = decisions.copy()
    out["prehead_symbol_guard_relative_weakness_pass"] = True
    out["prehead_symbol_guard_relative_peer_count"] = 0
    out["prehead_symbol_guard_relative_z_rank_pct"] = np.nan
    out["prehead_symbol_guard_relative_z_edge"] = np.nan
    out["prehead_symbol_guard_relative_loss_rank_pct"] = np.nan
    out["prehead_symbol_guard_relative_loss_edge"] = np.nan
    out["prehead_symbol_guard_relative_peer_veto"] = False

    risk = out.get("prehead_symbol_guard_blocked", False)
    risk = np.asarray(risk, dtype=bool) | pd.to_numeric(
        out.get("prehead_symbol_guard_penalty", 0.0),
        errors="coerce",
    ).fillna(0.0).gt(0.0).to_numpy(dtype=bool)
    if not risk.any():
        return out

    meta = frame[["__row_pos", "timestamp", "head", "symbol", "side"]].copy()
    meta["prehead_symbol_guard_day"] = meta["timestamp"].dt.floor("D")
    diagnostic = out.merge(meta, on="__row_pos", how="left", sort=False)
    peer_cols = _peer_columns_for_scope(cfg.scope)
    symbol_state = (
        diagnostic.groupby([*peer_cols, "symbol"], sort=False)
        .agg(
            prehead_symbol_guard_losses_last_n=("prehead_symbol_guard_losses_last_n", "max"),
            prehead_symbol_guard_recent_n=("prehead_symbol_guard_recent_n", "max"),
            prehead_symbol_guard_z_count=("prehead_symbol_guard_z_count", "max"),
            prehead_symbol_guard_z=("prehead_symbol_guard_z", "mean"),
        )
        .reset_index()
    )
    if symbol_state.empty:
        return out

    grouped = symbol_state.groupby(peer_cols, sort=False)
    peer_count = grouped["symbol"].transform("nunique").astype(int)
    z = pd.to_numeric(symbol_state["prehead_symbol_guard_z"], errors="coerce")
    z_count = pd.to_numeric(symbol_state["prehead_symbol_guard_z_count"], errors="coerce").fillna(0).astype(int)
    losses = pd.to_numeric(symbol_state["prehead_symbol_guard_losses_last_n"], errors="coerce").fillna(0.0)

    z_median = grouped["prehead_symbol_guard_z"].transform("median")
    z_rank_pct = grouped["prehead_symbol_guard_z"].rank(method="average", pct=True, ascending=True)
    z_edge = z_median - z
    loss_median = grouped["prehead_symbol_guard_losses_last_n"].transform("median")
    loss_rank_pct = grouped["prehead_symbol_guard_losses_last_n"].rank(method="average", pct=True, ascending=True)
    loss_edge = losses - loss_median

    enough_peers = peer_count.ge(int(min_peer_symbols))
    z_relative = (
        enough_peers
        & z_count.ge(int(cfg.z_min_count))
        & z_rank_pct.le(float(z_peer_quantile))
        & z_edge.ge(float(z_margin))
    )
    loss_relative = (
        enough_peers
        & loss_rank_pct.ge(float(loss_peer_quantile))
        & loss_edge.ge(float(loss_margin))
    )

    if cfg.mode == "loss_cooldown":
        relative_pass = loss_relative
    elif cfg.mode == "z_guard":
        relative_pass = z_relative
    elif cfg.mode == "soft_raise":
        relative_pass = loss_relative | z_relative
    elif cfg.mode == "hybrid_hard":
        relative_pass = loss_relative | z_relative
    else:
        relative_pass = pd.Series(True, index=symbol_state.index)

    symbol_state["prehead_symbol_guard_relative_weakness_pass"] = relative_pass.to_numpy(dtype=bool)
    symbol_state["prehead_symbol_guard_relative_peer_count"] = peer_count.to_numpy(dtype=int)
    symbol_state["prehead_symbol_guard_relative_z_rank_pct"] = z_rank_pct.to_numpy(dtype=float)
    symbol_state["prehead_symbol_guard_relative_z_edge"] = z_edge.to_numpy(dtype=float)
    symbol_state["prehead_symbol_guard_relative_loss_rank_pct"] = loss_rank_pct.to_numpy(dtype=float)
    symbol_state["prehead_symbol_guard_relative_loss_edge"] = loss_edge.to_numpy(dtype=float)

    relative_cols = [
        *peer_cols,
        "symbol",
        "prehead_symbol_guard_relative_weakness_pass",
        "prehead_symbol_guard_relative_peer_count",
        "prehead_symbol_guard_relative_z_rank_pct",
        "prehead_symbol_guard_relative_z_edge",
        "prehead_symbol_guard_relative_loss_rank_pct",
        "prehead_symbol_guard_relative_loss_edge",
    ]
    keyed = diagnostic[["__row_pos", *peer_cols, "symbol"]].merge(
        symbol_state[relative_cols],
        on=[*peer_cols, "symbol"],
        how="left",
        sort=False,
    )
    keyed = keyed.drop_duplicates("__row_pos", keep="last").set_index("__row_pos")
    for col in relative_cols:
        if col in peer_cols or col == "symbol":
            continue
        out[col] = out["__row_pos"].map(keyed[col])
    out["prehead_symbol_guard_relative_weakness_pass"] = (
        out["prehead_symbol_guard_relative_weakness_pass"].fillna(False).astype(bool)
    )
    veto = risk & ~out["prehead_symbol_guard_relative_weakness_pass"].to_numpy(dtype=bool)
    if not veto.any():
        return out
    out.loc[veto, "prehead_symbol_guard_relative_peer_veto"] = True
    if "prehead_symbol_guard_blocked" in out.columns:
        out.loc[veto, "prehead_symbol_guard_blocked"] = False
    if "prehead_symbol_guard_penalty" in out.columns:
        out.loc[veto, "prehead_symbol_guard_penalty"] = 0.0
    existing_reason = out.loc[veto, "prehead_symbol_guard_reason"].astype(str)
    out.loc[veto, "prehead_symbol_guard_reason"] = np.where(
        existing_reason.eq("") | existing_reason.eq("pass"),
        "relative_peer_veto",
        existing_reason + "+relative_peer_veto",
    )
    return out


def _apply_variant(frame: pd.DataFrame, cfg: GuardConfig, decisions: pd.DataFrame) -> pd.DataFrame:
    work = frame.merge(decisions, on="__row_pos", how="left", sort=False)
    defaults = {
        "prehead_symbol_guard_blocked": False,
        "prehead_symbol_guard_penalty": 0.0,
        "prehead_symbol_guard_reason": "no_decision",
        "prehead_symbol_guard_losses_last_n": 0,
        "prehead_symbol_guard_recent_n": 0,
        "prehead_symbol_guard_z_count": 0,
        "prehead_symbol_guard_z": np.nan,
        "prehead_symbol_guard_variant": cfg.variant,
        "prehead_symbol_guard_scope": cfg.scope,
        "prehead_symbol_guard_mode": cfg.mode,
    }
    for col, value in defaults.items():
        if col not in work.columns:
            work[col] = value
        else:
            work[col] = work[col].fillna(value)

    if cfg.mode in {"loss_cooldown", "z_guard", "hybrid_hard"}:
        work = work.loc[~work["prehead_symbol_guard_blocked"].astype(bool)].copy()
    elif cfg.mode == "soft_raise":
        penalty = pd.to_numeric(work["prehead_symbol_guard_penalty"], errors="coerce").fillna(0.0).clip(lower=0.0)
        for col in ("normalized_rank_score", "policy_rank_pct", "rank_pct", "strategy_rank_pct"):
            if col in work.columns:
                original_col = f"prehead_symbol_guard_original_{col}"
                if original_col not in work.columns:
                    work[original_col] = pd.to_numeric(work[col], errors="coerce")
                work[col] = (pd.to_numeric(work[col], errors="coerce") - penalty).clip(0.0, 1.0)
    drop_cols = [c for c in work.columns if c.startswith("__")]
    return work.drop(columns=drop_cols, errors="ignore")


def _summarize_variant(original: pd.DataFrame, variant: pd.DataFrame, decisions: pd.DataFrame, cfg: GuardConfig) -> dict[str, Any]:
    blocked = decisions["prehead_symbol_guard_blocked"].astype(bool) if "prehead_symbol_guard_blocked" in decisions else pd.Series(False)
    penalty = pd.to_numeric(decisions.get("prehead_symbol_guard_penalty", 0.0), errors="coerce").fillna(0.0)
    market_veto = (
        decisions["prehead_symbol_guard_market_wide_veto"].astype(bool)
        if "prehead_symbol_guard_market_wide_veto" in decisions
        else pd.Series(False)
    )
    relative_veto = (
        decisions["prehead_symbol_guard_relative_peer_veto"].astype(bool)
        if "prehead_symbol_guard_relative_peer_veto" in decisions
        else pd.Series(False)
    )
    blacklist_fraction = pd.to_numeric(
        decisions.get("prehead_symbol_guard_blacklist_asset_fraction", 0.0),
        errors="coerce",
    ).fillna(0.0)
    return {
        "variant": cfg.variant,
        "scope": cfg.scope,
        "mode": cfg.mode,
        "input_rows": int(len(original)),
        "output_rows": int(len(variant)),
        "removed_rows": int(len(original) - len(variant)),
        "blocked_rows": int(blocked.sum()) if len(blocked) else 0,
        "penalized_rows": int(penalty.gt(0.0).sum()) if len(penalty) else 0,
        "mean_penalty": float(penalty.loc[penalty.gt(0.0)].mean()) if penalty.gt(0.0).any() else 0.0,
        "max_penalty": float(penalty.max()) if len(penalty) else 0.0,
        "market_wide_veto_rows": int(market_veto.sum()) if len(market_veto) else 0,
        "relative_peer_veto_rows": int(relative_veto.sum()) if len(relative_veto) else 0,
        "max_blacklist_asset_fraction": float(blacklist_fraction.max()) if len(blacklist_fraction) else 0.0,
        "basket_asset_count": int(original["symbol"].nunique()) if "symbol" in original.columns else 0,
    }


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False)


def _iter_variants(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(VARIANTS)
    out = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(out) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; valid={VARIANTS}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--return-col", default="net_return")
    parser.add_argument("--top-rank-floor", type=float, default=0.70)
    parser.add_argument("--loss-window", type=int, default=4)
    parser.add_argument("--loss-threshold", type=int, default=3)
    parser.add_argument("--soft-loss-threshold", type=int, default=2)
    parser.add_argument("--cooldown-hours", type=float, default=24.0)
    parser.add_argument("--lookback-days", type=float, default=7.0)
    parser.add_argument("--z-min-count", type=int, default=5)
    parser.add_argument("--z-threshold", type=float, default=-1.50)
    parser.add_argument("--severe-z-threshold", type=float, default=-2.00)
    parser.add_argument("--soft-z-threshold", type=float, default=-1.25)
    parser.add_argument("--soft-penalty", type=float, default=0.05)
    parser.add_argument("--severe-penalty", type=float, default=0.10)
    parser.add_argument(
        "--max-blacklisted-asset-fraction",
        type=float,
        default=0.10,
        help=(
            "Day-level breadth veto for hard symbol blacklists. If more than this share "
            "of the full symbol basket would be blacklisted, treat it as market-wide "
            "and do not remove those rows. Set >=1 to disable."
        ),
    )
    parser.add_argument(
        "--require-relative-symbol-weakness",
        action="store_true",
        help=(
            "Apply hard removals or soft penalties only when the symbol is weak "
            "relative to its same-day peer basket, not just absolutely weak."
        ),
    )
    parser.add_argument("--relative-peer-min-symbols", type=int, default=20)
    parser.add_argument("--relative-z-peer-quantile", type=float, default=0.25)
    parser.add_argument("--relative-z-margin", type=float, default=0.50)
    parser.add_argument("--relative-loss-peer-quantile", type=float, default=0.75)
    parser.add_argument("--relative-loss-margin", type=float, default=1.0)
    parser.add_argument(
        "--engine",
        choices=("auto", "fast", "pandas"),
        default="auto",
        help="Guard-state engine. auto/fast use a Numba single-pass state machine when available.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = _iter_variants(args.variants)
    frame = _read_candidates(Path(args.candidates), return_col=str(args.return_col))

    summaries: list[dict[str, Any]] = []
    for name in variants:
        cfg = _variant_config(name, args)
        decisions = _guard_decisions(
            frame,
            cfg,
            top_rank_floor=float(args.top_rank_floor),
            engine=str(args.engine),
        )
        if bool(args.require_relative_symbol_weakness):
            decisions = _apply_relative_symbol_weakness_filter(
                frame,
                decisions,
                cfg,
                min_peer_symbols=int(args.relative_peer_min_symbols),
                z_peer_quantile=float(args.relative_z_peer_quantile),
                z_margin=float(args.relative_z_margin),
                loss_peer_quantile=float(args.relative_loss_peer_quantile),
                loss_margin=float(args.relative_loss_margin),
            )
        decisions = _apply_blacklist_breadth_veto(
            frame,
            decisions,
            max_asset_fraction=float(args.max_blacklisted_asset_fraction),
        )
        variant_frame = _apply_variant(frame, cfg, decisions)
        variant_path = output_dir / name / "simple_policy_candidates_broad.parquet"
        decision_path = output_dir / name / "prehead_symbol_guard_decisions.parquet"
        summary_path = output_dir / name / "prehead_symbol_guard_summary.parquet"
        _write_frame(variant_path, variant_frame)
        _write_frame(decision_path, decisions.drop(columns=[c for c in decisions.columns if c.startswith("__")], errors="ignore"))
        summary = _summarize_variant(frame, variant_frame, decisions, cfg)
        summary["candidate_path"] = str(variant_path)
        summaries.append(summary)

        if not decisions.empty:
            diagnostics = frame[["__row_pos", "timestamp", "head", "symbol", "side", "__guard_net_return", "__rank_for_guard"]].merge(
                decisions,
                on="__row_pos",
                how="inner",
                sort=False,
            )
            diagnostics["week_start"] = diagnostics["timestamp"].dt.floor("D") - pd.to_timedelta(
                diagnostics["timestamp"].dt.weekday,
                unit="D",
            )
            by_symbol = (
                diagnostics.loc[
                    diagnostics["prehead_symbol_guard_blocked"].astype(bool)
                    | pd.to_numeric(diagnostics["prehead_symbol_guard_penalty"], errors="coerce").fillna(0.0).gt(0.0)
                ]
                .groupby(["week_start", "head", "symbol", "side", "prehead_symbol_guard_reason"], sort=True)
                .agg(
                    rows=("__guard_net_return", "size"),
                    net_return=("__guard_net_return", "sum"),
                    avg_net_return=("__guard_net_return", "mean"),
                    mean_rank=("__rank_for_guard", "mean"),
                    mean_penalty=("prehead_symbol_guard_penalty", "mean"),
                    mean_z=("prehead_symbol_guard_z", "mean"),
                    max_losses_last_n=("prehead_symbol_guard_losses_last_n", "max"),
                )
                .reset_index()
            )
        else:
            by_symbol = pd.DataFrame()
        _write_frame(summary_path, by_symbol)
        print(json.dumps(summary, default=_json_default, sort_keys=True))

    manifest = {
        "candidate_path": str(Path(args.candidates)),
        "variants": variants,
        "top_rank_floor": float(args.top_rank_floor),
        "engine": str(args.engine),
        "numba_available": bool(njit is not None),
        "max_blacklisted_asset_fraction": float(args.max_blacklisted_asset_fraction),
        "require_relative_symbol_weakness": bool(args.require_relative_symbol_weakness),
        "relative_peer_min_symbols": int(args.relative_peer_min_symbols),
        "relative_z_peer_quantile": float(args.relative_z_peer_quantile),
        "relative_z_margin": float(args.relative_z_margin),
        "relative_loss_peer_quantile": float(args.relative_loss_peer_quantile),
        "relative_loss_margin": float(args.relative_loss_margin),
        "spread_return_diagnostics": frame.attrs.get("spread_return_diagnostics", {}),
        "configs": [asdict(_variant_config(name, args)) for name in variants],
        "summaries": summaries,
    }
    (output_dir / "prehead_symbol_guard_ablation_manifest.json").write_text(
        json.dumps(_json_default(manifest), indent=2),
        encoding="utf-8",
    )
    _write_frame(output_dir / "prehead_symbol_guard_ablation_summary.parquet", pd.DataFrame(summaries))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
