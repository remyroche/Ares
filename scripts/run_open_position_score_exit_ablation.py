#!/usr/bin/env python3
"""Fixed-entry exit ablation for refreshed open-position score policy ideas.

This intentionally does not re-rank or add trades. It replays the accepted T1
trade set through alternative exit geometry so the effect is attributable to
exit logic only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.simple_policy_optimiser import (
    DEFAULT_POLICY_PER_SIDE_COST_PCT,
    _apply_delayed_entry_execution_model,
    _fetch_policy_paths,
    _make_policy_replay_store,
    _policy_path_finite_mask,
    simulate_and_score,
)


@dataclass(frozen=True)
class ReplayContext:
    rows: pd.DataFrame
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    output_dir: Path
    saved_reference_summary: dict[str, Any] | None = None
    baseline_net_returns: np.ndarray | None = None
    baseline_position_sizes: np.ndarray | None = None
    replay_reference_net_returns: np.ndarray | None = None
    replay_reference_exit_bars: np.ndarray | None = None
    replay_reference_exit_reasons: np.ndarray | None = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _rank_tightened_rows(rows: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    strength = _safe_float(params.get("rank_tighten_strength", 0.0))
    if strength <= 0.0:
        return rows
    rank = pd.to_numeric(rows["rank_pct"], errors="coerce").fillna(0.5).to_numpy(
        dtype=np.float64
    )
    center = float(np.clip(_safe_float(params.get("rank_tighten_center", 0.90)), 0.55, 0.999))
    floor = float(np.clip(_safe_float(params.get("rank_tighten_floor", 0.65)), 0.05, 1.0))
    power = float(np.clip(_safe_float(params.get("rank_tighten_power", 1.0)), 0.25, 4.0))
    denom = max(center - 0.50, 1e-6)
    weakness = np.clip((center - rank) / denom, 0.0, 1.0) ** power
    multiplier = np.clip(1.0 - strength * weakness, floor, 1.0)
    out = rows.copy()
    for col in ("barrier_pct", "barrier_frac"):
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=np.float64)
            out[col] = (vals * multiplier).astype(np.float32)
    out["entry_rank_barrier_multiplier"] = multiplier.astype(np.float32)
    return out


def _selected_trade_returns(
    metrics: dict[str, Any],
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    selected_mask = np.asarray(metrics.get("selected_mask"), dtype=bool)
    if selected_mask.size != len(rows):
        selected_mask = np.ones(len(rows), dtype=bool)
    selected_rows = rows.iloc[np.flatnonzero(selected_mask)].copy()
    raw = np.asarray(metrics.get("raw_gains", []), dtype=np.float64)
    gross = np.asarray(metrics.get("gross_gains", []), dtype=np.float64)
    sizes = np.asarray(metrics.get("sizes", []), dtype=np.float64)
    valid = np.isfinite(sizes) & (np.abs(sizes) > 1e-12)
    if raw.size != selected_rows.shape[0] or gross.size != selected_rows.shape[0] or sizes.size != selected_rows.shape[0]:
        raise RuntimeError(
            "Replay metric length mismatch: "
            f"rows={selected_rows.shape[0]} raw={raw.size} gross={gross.size} sizes={sizes.size}"
        )
    net_return = np.zeros_like(raw, dtype=np.float64)
    gross_return = np.zeros_like(gross, dtype=np.float64)
    net_return[valid] = raw[valid] / sizes[valid]
    gross_return[valid] = gross[valid] / sizes[valid]
    exit_bars = np.asarray(metrics.get("exit_bars", []), dtype=np.float64)
    if exit_bars.size != selected_rows.shape[0]:
        exit_bars = np.full(selected_rows.shape[0], np.nan, dtype=np.float64)
    return selected_rows, net_return, gross_return, exit_bars


def _exit_rate(exit_reason: np.ndarray, token: str) -> float:
    if exit_reason.size == 0:
        return 0.0
    reason = pd.Series(exit_reason.astype(str))
    return float(reason.str.contains(token, regex=False).mean())


def evaluate_arm(
    name: str,
    ctx: ReplayContext,
    params: dict[str, Any],
    *,
    overlay_on_t1: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = _rank_tightened_rows(ctx.rows, params)
    sim_params = {k: v for k, v in params.items() if not k.startswith("rank_tighten_")}
    metrics = simulate_and_score(
        rows,
        *ctx.paths,
        cost_pct=DEFAULT_POLICY_PER_SIDE_COST_PCT,
        size_power=float(sim_params.pop("size_power", 1.0)),
        max_concurrent_trades=1_000_000,
        max_concurrent_per_asset=1_000_000,
        market_mode="perps",
        **sim_params,
    )
    selected_rows, sim_net_return, sim_gross_return, sim_exit_bars = _selected_trade_returns(metrics, rows)
    position_size = pd.to_numeric(
        selected_rows["accepted_position_size"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=np.float64)
    t1_net_return = pd.to_numeric(
        selected_rows["accepted_net_return"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=np.float64)
    t1_gross_return = pd.to_numeric(
        selected_rows["accepted_gross_return"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    t1_gross_return = np.nan_to_num(t1_gross_return, nan=t1_net_return)
    t1_holding_bars = pd.to_numeric(
        selected_rows.get("holding_bars", pd.Series(np.inf, index=selected_rows.index)),
        errors="coerce",
    ).fillna(np.inf).to_numpy(dtype=np.float64)
    exit_reason = np.asarray(metrics.get("exit_reason", []), dtype=object)
    if exit_reason.size != selected_rows.shape[0]:
        exit_reason = np.repeat("", selected_rows.shape[0])
    sim_exit_reason = exit_reason.astype(str).copy()
    overlay_applied = np.zeros(len(selected_rows), dtype=bool)
    if overlay_on_t1:
        # Overlay semantics: T1 remains the base policy. A candidate exit layer
        # can only replace T1 if it changes the no-op replay and triggers no
        # later than the materialized T1 exit. Otherwise, the exact saved T1
        # outcome is retained.
        changed_vs_replay_reference = np.ones(len(selected_rows), dtype=bool)
        if (
            ctx.replay_reference_net_returns is not None
            and ctx.replay_reference_exit_bars is not None
            and len(ctx.replay_reference_net_returns) == len(selected_rows)
            and len(ctx.replay_reference_exit_bars) == len(selected_rows)
        ):
            changed_vs_replay_reference = (
                np.abs(sim_net_return - ctx.replay_reference_net_returns) > 1e-9
            ) | (
                np.nan_to_num(sim_exit_bars, nan=-1.0)
                != np.nan_to_num(ctx.replay_reference_exit_bars, nan=-1.0)
            )
            if (
                ctx.replay_reference_exit_reasons is not None
                and len(ctx.replay_reference_exit_reasons) == len(selected_rows)
            ):
                changed_vs_replay_reference |= (
                    sim_exit_reason.astype(str)
                    != ctx.replay_reference_exit_reasons.astype(str)
                )
        overlay_applied = (
            changed_vs_replay_reference
            & np.isfinite(sim_exit_bars)
            & (sim_exit_bars <= t1_holding_bars)
        )
        net_return = np.where(overlay_applied, sim_net_return, t1_net_return)
        gross_return = np.where(overlay_applied, sim_gross_return, t1_gross_return)
    else:
        net_return = sim_net_return
        gross_return = sim_gross_return
    net_pnl_by_trade = net_return * position_size
    gross_pnl_by_trade = gross_return * position_size
    cost_pnl_by_trade = gross_pnl_by_trade - net_pnl_by_trade
    if overlay_on_t1:
        t1_reason = selected_rows["accepted_exit_reason"].astype(str).to_numpy(dtype=object)
        exit_reason = np.where(
            overlay_applied,
            np.char.add("overlay_", exit_reason.astype(str)),
            np.char.add("t1_", t1_reason.astype(str)),
        )

    per_trade = selected_rows[
        [
            "timestamp",
            "symbol",
            "strategy_id",
            "head",
            "accepted_position_size",
            "rank_pct",
            "barrier_pct",
            "candidate_index",
        ]
    ].copy()
    per_trade["arm"] = name
    per_trade["t1_net_return"] = t1_net_return.astype(np.float32)
    per_trade["t1_gross_return"] = t1_gross_return.astype(np.float32)
    per_trade["t1_holding_bars"] = t1_holding_bars.astype(np.float32)
    per_trade["sim_net_return"] = sim_net_return.astype(np.float32)
    per_trade["sim_gross_return"] = sim_gross_return.astype(np.float32)
    per_trade["sim_exit_bars"] = sim_exit_bars.astype(np.float32)
    per_trade["sim_exit_reason"] = sim_exit_reason
    per_trade["overlay_applied"] = overlay_applied
    per_trade["net_return_replay"] = net_return.astype(np.float32)
    per_trade["gross_return_replay"] = gross_return.astype(np.float32)
    per_trade["net_pnl_replay"] = net_pnl_by_trade.astype(np.float32)
    per_trade["gross_pnl_replay"] = gross_pnl_by_trade.astype(np.float32)
    per_trade["cost_pnl_replay"] = cost_pnl_by_trade.astype(np.float32)
    per_trade["exit_reason_replay"] = exit_reason

    baseline_net = ctx.baseline_net_returns
    baseline_size = ctx.baseline_position_sizes
    loss_avoided = winner_sacrificed = loser_loss_worsened = defensive_success = net_pnl_delta = np.nan
    if overlay_on_t1:
        baseline_net = t1_net_return
        baseline_size = position_size
    if baseline_net is not None and baseline_size is not None and len(baseline_net) == len(net_return):
        baseline_pnl = baseline_net * baseline_size
        arm_pnl = net_return * baseline_size
        pnl_delta = arm_pnl - baseline_pnl
        loss_avoided = float(np.sum(np.maximum(0.0, arm_pnl - baseline_pnl)[baseline_pnl < 0.0]))
        loser_loss_worsened = float(
            np.sum(np.maximum(0.0, baseline_pnl - arm_pnl)[baseline_pnl < 0.0])
        )
        winner_sacrificed = float(
            np.sum(np.maximum(0.0, baseline_pnl - arm_pnl)[baseline_pnl > 0.0])
        )
        defensive_success = loss_avoided - winner_sacrificed - loser_loss_worsened
        net_pnl_delta = float(np.sum(pnl_delta))

    timestamps = pd.to_datetime(per_trade["timestamp"], utc=True, errors="coerce")
    day_pnl = per_trade.assign(day=timestamps.dt.floor("D")).groupby("day")[
        "net_pnl_replay"
    ].sum()
    head_pnl = per_trade.groupby("head")["net_pnl_replay"].sum().to_dict()
    summary = {
        "arm": name,
        "trade_count": int(len(per_trade)),
        "net_pnl": float(np.sum(net_pnl_by_trade)),
        "gross_pnl": float(np.sum(gross_pnl_by_trade)),
        "cost_pnl": float(np.sum(cost_pnl_by_trade)),
        "mean_net_return": float(np.mean(net_return)) if len(net_return) else 0.0,
        "median_net_return": float(np.median(net_return)) if len(net_return) else 0.0,
        "win_rate": float(np.mean(net_return > 0.0)) if len(net_return) else 0.0,
        "q05_net_return": float(np.quantile(net_return, 0.05)) if len(net_return) else 0.0,
        "q15_net_return": float(np.quantile(net_return, 0.15)) if len(net_return) else 0.0,
        "worst_day_net_pnl": float(day_pnl.min()) if len(day_pnl) else 0.0,
        "best_day_net_pnl": float(day_pnl.max()) if len(day_pnl) else 0.0,
        "full_sl_rate": _exit_rate(exit_reason, "full_sl"),
        "timeout_rate": _exit_rate(exit_reason, "timeout"),
        "trailing_rate": _exit_rate(exit_reason, "trailing"),
        "adverse_exit_rate": _exit_rate(exit_reason, "adverse_exit"),
        "hard_tp_rate": _exit_rate(exit_reason, "hard_tp"),
        "capital_protect_rate": _exit_rate(exit_reason, "capital_protect"),
        "overlay_applied_rate": float(np.mean(overlay_applied)) if len(overlay_applied) else 0.0,
        "loss_avoided_vs_baseline": loss_avoided,
        "loser_loss_worsened_vs_baseline": loser_loss_worsened,
        "winner_pnl_sacrificed_vs_baseline": winner_sacrificed,
        "defensive_success_vs_baseline": defensive_success,
        "net_pnl_delta_vs_baseline": net_pnl_delta,
        "head_net_pnl_json": json.dumps({str(k): float(v) for k, v in head_pnl.items()}, sort_keys=True),
        "params_json": json.dumps(params, sort_keys=True),
    }
    return summary, per_trade


def _load_fixed_entry_rows(artifact_dir: Path) -> pd.DataFrame:
    accepted_path = artifact_dir / "simple_policy_optimiser" / "accepted_trades.parquet"
    broad_path = artifact_dir / "simple_policy_optimiser" / "simple_policy_candidates_broad.parquet"
    if not accepted_path.exists():
        raise FileNotFoundError(accepted_path)
    if not broad_path.exists():
        raise FileNotFoundError(broad_path)
    accepted = pd.read_parquet(accepted_path).reset_index(drop=True)
    broad = pd.read_parquet(broad_path).reset_index(drop=True)
    idx = pd.to_numeric(accepted["candidate_index"], errors="coerce").astype("int64")
    rows = broad.iloc[idx.to_numpy()].reset_index(drop=True).copy()
    rows["candidate_index"] = idx.to_numpy()
    rows["accepted_position_size"] = pd.to_numeric(
        accepted["position_size"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=np.float64)
    rows["accepted_net_pnl"] = pd.to_numeric(
        accepted["net_pnl"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=np.float64)
    rows["accepted_net_return"] = pd.to_numeric(
        accepted["net_return"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=np.float64)
    rows["accepted_gross_return"] = pd.to_numeric(
        accepted.get("gross_return", pd.Series(np.nan, index=accepted.index)),
        errors="coerce",
    ).fillna(np.nan).to_numpy(dtype=np.float64)
    rows["accepted_exit_reason"] = accepted["simple_policy_exit_reason"].astype(str).to_numpy()
    if "head" not in rows.columns:
        rows["head"] = accepted["head"].astype(str).to_numpy()
    else:
        rows["head"] = rows["head"].fillna(accepted["head"]).astype(str)
    if "timestamp" not in rows.columns:
        rows["timestamp"] = accepted["timestamp"]
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    side_text = rows["side"].astype(str).str.lower() if "side" in rows.columns else pd.Series("long", index=rows.index)
    rows["side_text"] = side_text
    rows["side"] = np.where(side_text.eq("short"), -1.0, 1.0).astype(np.float32)
    for col in ("rank_pct", "barrier_pct"):
        if col not in rows.columns:
            raise KeyError(f"Missing required column {col}")
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    required = ["timestamp", "symbol", "strategy_id", "rank_pct", "barrier_pct"]
    rows = rows.dropna(subset=required).reset_index(drop=True)
    return rows


def _load_fixed_entry_rows_from_accepted_report(path: Path) -> pd.DataFrame:
    accepted = pd.read_parquet(path).reset_index(drop=True)
    if "candidate_timestamp" in accepted.columns:
        candidate_cols = {
            col: col.removeprefix("candidate_")
            for col in accepted.columns
            if col.startswith("candidate_")
        }
        rows = accepted[list(candidate_cols)].rename(columns=candidate_cols).copy()
    else:
        rows = accepted.copy()
    if "candidate_index" in accepted.columns:
        rows["candidate_index"] = pd.to_numeric(
            accepted["candidate_index"], errors="coerce"
        ).fillna(-1).astype("int64")
    else:
        rows["candidate_index"] = np.arange(len(accepted), dtype=np.int64)
    size_source = "position_size" if "position_size" in accepted.columns else "accepted_position_size"
    rows["accepted_position_size"] = pd.to_numeric(
        accepted[size_source], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=np.float64)
    net_source = "net_return" if "net_return" in accepted.columns else "candidate_net_return"
    gross_source = "gross_return" if "gross_return" in accepted.columns else "candidate_gross_return"
    reason_source = (
        "simple_policy_exit_reason"
        if "simple_policy_exit_reason" in accepted.columns
        else "candidate_simple_policy_exit_reason"
    )
    if net_source in accepted.columns:
        rows["accepted_net_return"] = pd.to_numeric(
            accepted[net_source], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float64)
    elif "net_return" in rows.columns:
        rows["accepted_net_return"] = pd.to_numeric(
            rows["net_return"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float64)
    else:
        raise KeyError("Missing accepted net return in accepted report")
    if gross_source in accepted.columns:
        rows["accepted_gross_return"] = pd.to_numeric(
            accepted[gross_source], errors="coerce"
        ).to_numpy(dtype=np.float64)
    elif "gross_return" in rows.columns:
        rows["accepted_gross_return"] = pd.to_numeric(
            rows["gross_return"], errors="coerce"
        ).to_numpy(dtype=np.float64)
    else:
        rows["accepted_gross_return"] = np.full(len(rows), np.nan, dtype=np.float64)
    if reason_source in accepted.columns:
        rows["accepted_exit_reason"] = accepted[reason_source].astype(str).to_numpy()
    elif "simple_policy_exit_reason" in rows.columns:
        rows["accepted_exit_reason"] = rows["simple_policy_exit_reason"].astype(str).to_numpy()
    else:
        rows["accepted_exit_reason"] = np.repeat("unknown", len(rows))
    if "head" not in rows.columns and "head" in accepted.columns:
        rows["head"] = accepted["head"].astype(str).to_numpy()
    if "head" not in rows.columns:
        raise KeyError("Missing head in accepted report")
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    side_text = (
        rows["side"].astype(str).str.lower()
        if "side" in rows.columns
        else pd.Series("long", index=rows.index)
    )
    rows["side_text"] = side_text
    rows["side"] = np.where(side_text.eq("short"), -1.0, 1.0).astype(np.float32)
    for col in ("rank_pct", "barrier_pct"):
        if col not in rows.columns:
            raise KeyError(f"Missing required column {col}")
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    required = ["timestamp", "symbol", "strategy_id", "rank_pct", "barrier_pct"]
    rows = rows.dropna(subset=required).reset_index(drop=True)
    return rows


def _saved_reference_summary(rows: pd.DataFrame) -> dict[str, Any]:
    net_return = pd.to_numeric(rows["accepted_net_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    gross_return = pd.to_numeric(rows["accepted_gross_return"], errors="coerce").to_numpy(dtype=np.float64)
    position_size = pd.to_numeric(rows["accepted_position_size"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    net_pnl = net_return * position_size
    if np.isfinite(gross_return).any():
        gross_pnl = np.nan_to_num(gross_return, nan=net_return) * position_size
    else:
        gross_pnl = net_pnl.copy()
    exit_reason = rows["accepted_exit_reason"].astype(str)
    exit_counts = exit_reason.value_counts(normalize=True).to_dict()
    timestamps = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    day_pnl = pd.DataFrame({"day": timestamps.dt.floor("D"), "net_pnl": net_pnl}).groupby("day")["net_pnl"].sum()
    head_pnl = (
        pd.DataFrame({"head": rows["head"].astype(str), "net_pnl": net_pnl})
        .groupby("head")["net_pnl"]
        .sum()
        .to_dict()
    )
    return {
        "arm": "saved_T1_reference",
        "trade_count": int(len(rows)),
        "net_pnl": float(np.sum(net_pnl)),
        "gross_pnl": float(np.sum(gross_pnl)),
        "cost_pnl": float(np.sum(gross_pnl - net_pnl)),
        "mean_net_return": float(np.mean(net_return)) if len(net_return) else 0.0,
        "median_net_return": float(np.median(net_return)) if len(net_return) else 0.0,
        "win_rate": float(np.mean(net_return > 0.0)) if len(net_return) else 0.0,
        "q05_net_return": float(np.quantile(net_return, 0.05)) if len(net_return) else 0.0,
        "q15_net_return": float(np.quantile(net_return, 0.15)) if len(net_return) else 0.0,
        "worst_day_net_pnl": float(day_pnl.min()) if len(day_pnl) else 0.0,
        "best_day_net_pnl": float(day_pnl.max()) if len(day_pnl) else 0.0,
        "full_sl_rate": float(exit_counts.get("full_sl", 0.0)),
        "timeout_rate": float(exit_counts.get("timeout", 0.0)),
        "trailing_rate": float(exit_counts.get("trailing", 0.0)),
        "adverse_exit_rate": float(exit_counts.get("adverse_exit", 0.0)),
        "hard_tp_rate": float(exit_counts.get("hard_tp", 0.0)),
        "capital_protect_rate": float(exit_counts.get("capital_protect", 0.0)),
        "loss_avoided_vs_baseline": np.nan,
        "loser_loss_worsened_vs_baseline": np.nan,
        "winner_pnl_sacrificed_vs_baseline": np.nan,
        "defensive_success_vs_baseline": np.nan,
        "net_pnl_delta_vs_baseline": np.nan,
        "overlay_applied_rate": 0.0,
        "head_net_pnl_json": json.dumps({str(k): float(v) for k, v in head_pnl.items()}, sort_keys=True),
        "params_json": json.dumps({"source": "accepted_trades.parquet"}, sort_keys=True),
    }


def _build_context(
    *,
    artifact_dir: Path,
    output_dir: Path,
    data_root: str,
    market_mode: str,
    max_rows: int,
) -> ReplayContext:
    rows = _load_fixed_entry_rows(artifact_dir)
    if max_rows and max_rows > 0:
        rows = rows.head(max_rows).copy()
    ds = _make_policy_replay_store(data_root, market_mode)
    paths = _fetch_policy_paths(rows, ds)
    rows, paths = _apply_delayed_entry_execution_model(
        rows,
        paths,
        data_root=data_root,
        market_mode=market_mode,
    )
    finite_mask = _policy_path_finite_mask(paths)
    if finite_mask.size != len(rows):
        raise RuntimeError(
            f"Path coverage mismatch: rows={len(rows)} finite_mask={finite_mask.size}"
        )
    if not bool(finite_mask.all()):
        rows = rows.iloc[np.flatnonzero(finite_mask)].reset_index(drop=True)
        paths = tuple(arr[finite_mask] for arr in paths)  # type: ignore[assignment]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = rows.reset_index(drop=True)
    return ReplayContext(
        rows=rows,
        paths=paths,
        output_dir=output_dir,
        saved_reference_summary=_saved_reference_summary(rows),
    )


def _build_context_from_accepted_report(
    *,
    accepted_report_path: Path,
    output_dir: Path,
    data_root: str,
    market_mode: str,
    max_rows: int,
) -> ReplayContext:
    rows = _load_fixed_entry_rows_from_accepted_report(accepted_report_path)
    if max_rows and max_rows > 0:
        rows = rows.head(max_rows).copy()
    ds = _make_policy_replay_store(data_root, market_mode)
    paths = _fetch_policy_paths(rows, ds)
    rows, paths = _apply_delayed_entry_execution_model(
        rows,
        paths,
        data_root=data_root,
        market_mode=market_mode,
    )
    finite_mask = _policy_path_finite_mask(paths)
    if finite_mask.size != len(rows):
        raise RuntimeError(
            f"Path coverage mismatch: rows={len(rows)} finite_mask={finite_mask.size}"
        )
    if not bool(finite_mask.all()):
        rows = rows.iloc[np.flatnonzero(finite_mask)].reset_index(drop=True)
        paths = tuple(arr[finite_mask] for arr in paths)  # type: ignore[assignment]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = rows.reset_index(drop=True)
    return ReplayContext(
        rows=rows,
        paths=paths,
        output_dir=output_dir,
        saved_reference_summary=_saved_reference_summary(rows),
    )


def _combine_contexts(contexts: list[ReplayContext], output_dir: Path) -> ReplayContext:
    contexts = [ctx for ctx in contexts if len(ctx.rows) > 0]
    if not contexts:
        raise ValueError("No non-empty contexts to combine.")
    rows = pd.concat([ctx.rows for ctx in contexts], ignore_index=True)
    paths = tuple(
        np.concatenate([ctx.paths[i] for ctx in contexts], axis=0)
        for i in range(len(contexts[0].paths))
    )
    return ReplayContext(
        rows=rows.reset_index(drop=True),
        paths=paths,  # type: ignore[arg-type]
        output_dir=output_dir,
        saved_reference_summary=_saved_reference_summary(rows),
    )


def _build_multi_context(
    *,
    artifact_dirs: list[Path],
    accepted_report_paths: list[Path],
    output_dir: Path,
    data_root: str,
    market_mode: str,
    max_rows: int,
) -> ReplayContext:
    contexts = [
        _build_context(
            artifact_dir=artifact_dir,
            output_dir=output_dir,
            data_root=data_root,
            market_mode=market_mode,
            max_rows=max_rows,
        )
        for artifact_dir in artifact_dirs
    ]
    contexts.extend(
        _build_context_from_accepted_report(
            accepted_report_path=accepted_report_path,
            output_dir=output_dir,
            data_root=data_root,
            market_mode=market_mode,
            max_rows=max_rows,
        )
        for accepted_report_path in accepted_report_paths
    )
    return _combine_contexts(contexts, output_dir)


def _subset_context(ctx: ReplayContext, mask: np.ndarray | pd.Series) -> ReplayContext:
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.size != len(ctx.rows):
        raise ValueError(f"Subset mask length mismatch: {mask_arr.size} != {len(ctx.rows)}")
    idx = np.flatnonzero(mask_arr)
    replay_ref_net = (
        ctx.replay_reference_net_returns[idx]
        if ctx.replay_reference_net_returns is not None
        and len(ctx.replay_reference_net_returns) == len(ctx.rows)
        else None
    )
    replay_ref_bars = (
        ctx.replay_reference_exit_bars[idx]
        if ctx.replay_reference_exit_bars is not None
        and len(ctx.replay_reference_exit_bars) == len(ctx.rows)
        else None
    )
    replay_ref_reasons = (
        ctx.replay_reference_exit_reasons[idx]
        if ctx.replay_reference_exit_reasons is not None
        and len(ctx.replay_reference_exit_reasons) == len(ctx.rows)
        else None
    )
    baseline_net = (
        ctx.baseline_net_returns[idx]
        if ctx.baseline_net_returns is not None
        and len(ctx.baseline_net_returns) == len(ctx.rows)
        else None
    )
    baseline_sizes = (
        ctx.baseline_position_sizes[idx]
        if ctx.baseline_position_sizes is not None
        and len(ctx.baseline_position_sizes) == len(ctx.rows)
        else None
    )
    rows = ctx.rows.iloc[idx].reset_index(drop=True)
    return ReplayContext(
        rows=rows,
        paths=tuple(arr[idx] for arr in ctx.paths),  # type: ignore[arg-type]
        output_dir=ctx.output_dir,
        saved_reference_summary=_saved_reference_summary(rows),
        baseline_net_returns=baseline_net,
        baseline_position_sizes=baseline_sizes,
        replay_reference_net_returns=replay_ref_net,
        replay_reference_exit_bars=replay_ref_bars,
        replay_reference_exit_reasons=replay_ref_reasons,
    )


def _summary_from_per_trade(
    name: str,
    per_trade: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    if per_trade.empty:
        return {
            "arm": name,
            "trade_count": 0,
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "cost_pnl": 0.0,
            "mean_net_return": 0.0,
            "median_net_return": 0.0,
            "win_rate": 0.0,
            "q05_net_return": 0.0,
            "q15_net_return": 0.0,
            "worst_day_net_pnl": 0.0,
            "best_day_net_pnl": 0.0,
            "full_sl_rate": 0.0,
            "timeout_rate": 0.0,
            "trailing_rate": 0.0,
            "adverse_exit_rate": 0.0,
            "hard_tp_rate": 0.0,
            "capital_protect_rate": 0.0,
            "overlay_applied_rate": 0.0,
            "loss_avoided_vs_baseline": 0.0,
            "loser_loss_worsened_vs_baseline": 0.0,
            "winner_pnl_sacrificed_vs_baseline": 0.0,
            "defensive_success_vs_baseline": 0.0,
            "net_pnl_delta_vs_baseline": 0.0,
            "head_net_pnl_json": "{}",
            "params_json": json.dumps(params, sort_keys=True),
        }
    net_return = per_trade["net_return_replay"].to_numpy(dtype=np.float64)
    gross_return = per_trade["gross_return_replay"].to_numpy(dtype=np.float64)
    net_pnl = per_trade["net_pnl_replay"].to_numpy(dtype=np.float64)
    gross_pnl = per_trade["gross_pnl_replay"].to_numpy(dtype=np.float64)
    baseline_net = per_trade["t1_net_return"].to_numpy(dtype=np.float64)
    size = per_trade["accepted_position_size"].to_numpy(dtype=np.float64)
    baseline_pnl = baseline_net * size
    pnl_delta = net_pnl - baseline_pnl
    timestamps = pd.to_datetime(per_trade["timestamp"], utc=True, errors="coerce")
    day_pnl = per_trade.assign(day=timestamps.dt.floor("D")).groupby("day")[
        "net_pnl_replay"
    ].sum()
    head_pnl = per_trade.groupby("head")["net_pnl_replay"].sum().to_dict()
    exit_reason = per_trade["exit_reason_replay"].astype(str).to_numpy(dtype=object)
    return {
        "arm": name,
        "trade_count": int(len(per_trade)),
        "net_pnl": float(np.sum(net_pnl)),
        "gross_pnl": float(np.sum(gross_pnl)),
        "cost_pnl": float(np.sum(gross_pnl - net_pnl)),
        "mean_net_return": float(np.mean(net_return)),
        "median_net_return": float(np.median(net_return)),
        "win_rate": float(np.mean(net_return > 0.0)),
        "q05_net_return": float(np.quantile(net_return, 0.05)),
        "q15_net_return": float(np.quantile(net_return, 0.15)),
        "worst_day_net_pnl": float(day_pnl.min()) if len(day_pnl) else 0.0,
        "best_day_net_pnl": float(day_pnl.max()) if len(day_pnl) else 0.0,
        "full_sl_rate": _exit_rate(exit_reason, "full_sl"),
        "timeout_rate": _exit_rate(exit_reason, "timeout"),
        "trailing_rate": _exit_rate(exit_reason, "trailing"),
        "adverse_exit_rate": _exit_rate(exit_reason, "adverse_exit"),
        "hard_tp_rate": _exit_rate(exit_reason, "hard_tp"),
        "capital_protect_rate": _exit_rate(exit_reason, "capital_protect"),
        "overlay_applied_rate": float(per_trade["overlay_applied"].mean()),
        "loss_avoided_vs_baseline": float(
            np.sum(np.maximum(0.0, pnl_delta)[baseline_pnl < 0.0])
        ),
        "loser_loss_worsened_vs_baseline": float(
            np.sum(np.maximum(0.0, -pnl_delta)[baseline_pnl < 0.0])
        ),
        "winner_pnl_sacrificed_vs_baseline": float(
            np.sum(np.maximum(0.0, -pnl_delta)[baseline_pnl > 0.0])
        ),
        "defensive_success_vs_baseline": float(np.sum(pnl_delta)),
        "net_pnl_delta_vs_baseline": float(np.sum(pnl_delta)),
        "head_net_pnl_json": json.dumps(
            {str(k): float(v) for k, v in head_pnl.items()}, sort_keys=True
        ),
        "params_json": json.dumps(params, sort_keys=True),
    }


def evaluate_head_params_arm(
    name: str,
    ctx: ReplayContext,
    head_params: dict[str, dict[str, Any]],
    *,
    overlay_on_t1: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for head in sorted(ctx.rows["head"].astype(str).unique()):
        mask = ctx.rows["head"].astype(str).eq(head).to_numpy()
        if not bool(mask.any()):
            continue
        subctx = _subset_context(ctx, mask)
        _, trades = evaluate_arm(
            f"{name}__{head}",
            subctx,
            dict(head_params.get(head, {})),
            overlay_on_t1=overlay_on_t1,
        )
        frames.append(trades)
    all_trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not all_trades.empty:
        all_trades["arm"] = name
    return _summary_from_per_trade(name, all_trades, {"per_head": head_params}), all_trades


def _static_arms(*, overlay_on_t1: bool = False) -> list[tuple[str, dict[str, Any]]]:
    arms = [
        ("baseline_replay", {}),
        ("tight_sl_0p80", {"sl_mult": 0.80}),
        ("trailing_fast_0p70", {"trailing_activation_mult": 0.70}),
        ("hard_tp_1pct", {"hard_tp_abs_pct": 0.010}),
        (
            "adverse_exit_mild",
            {
                "adverse_exit_enabled": True,
                "adverse_exit_min_mae_atr": 0.55,
                "adverse_exit_min_speed": 0.20,
                "adverse_exit_theta_quantile": 0.65,
            },
        ),
        (
            "entry_rank_tighten_mild",
            {
                "rank_tighten_strength": 0.30,
                "rank_tighten_center": 0.90,
                "rank_tighten_floor": 0.70,
                "rank_tighten_power": 1.0,
            },
        ),
        (
            "entry_rank_tighten_strong",
            {
                "rank_tighten_strength": 0.50,
                "rank_tighten_center": 0.95,
                "rank_tighten_floor": 0.55,
                "rank_tighten_power": 1.0,
            },
        ),
        (
            "exit_pressure_duration",
            {
                "exit_pressure_enabled": True,
                "exit_pressure_beta": 0.40,
                "exit_pressure_kappa": 0.30,
                "exit_pressure_min_multiplier": 0.70,
                "target_holding_hours": 8.0,
                "redeploy_scale_bps": 120.0,
            },
        ),
    ]
    if not overlay_on_t1:
        # Widening the original T1 stop is a replacement-policy experiment,
        # not a protective overlay on top of T1.
        arms.insert(2, ("wide_sl_1p20", {"sl_mult": 1.20}))
    if overlay_on_t1:
        arms = [
            (f"t1_plus_{name}", params)
            for name, params in arms
            if name != "baseline_replay"
        ]
    return arms


def _suggest_overlay_params(
    trial: Any,
    *,
    overlay_on_t1: bool,
    safe_space: bool = True,
) -> dict[str, Any]:
    if overlay_on_t1 and safe_space:
        params: dict[str, Any] = {
            "sl_mult": trial.suggest_float("sl_mult", 0.75, 1.00),
            "trailing_activation_mult": trial.suggest_float(
                "trailing_activation_mult", 0.95, 1.15
            ),
            "trailing_power": trial.suggest_float("trailing_power", 0.80, 2.50),
            "giveback_beta": trial.suggest_float("giveback_beta", 0.25, 0.85),
            "hard_tp_abs_pct": 0.0,
            "rank_tighten_strength": trial.suggest_float(
                "rank_tighten_strength", 0.0, 0.35
            ),
            "rank_tighten_center": trial.suggest_float(
                "rank_tighten_center", 0.75, 0.90
            ),
            "rank_tighten_floor": trial.suggest_float(
                "rank_tighten_floor", 0.70, 0.95
            ),
            "rank_tighten_power": trial.suggest_float(
                "rank_tighten_power", 0.50, 2.50
            ),
            "exit_pressure_enabled": False,
            "adverse_exit_enabled": False,
        }
        return params

    sl_low, sl_high = (0.55, 1.00) if overlay_on_t1 else (0.60, 1.35)
    trail_low, trail_high = (0.40, 1.00) if overlay_on_t1 else (0.50, 1.50)
    params = {
        "sl_mult": trial.suggest_float("sl_mult", sl_low, sl_high),
        "trailing_activation_mult": trial.suggest_float(
            "trailing_activation_mult", trail_low, trail_high
        ),
        "trailing_power": trial.suggest_float("trailing_power", 0.80, 2.50),
        "giveback_beta": trial.suggest_float("giveback_beta", 0.25, 0.85),
        "hard_tp_abs_pct": trial.suggest_categorical(
            "hard_tp_abs_pct",
            [0.0, 0.005, 0.010]
            if overlay_on_t1
            else [0.0, 0.005, 0.010, 0.015, 0.020],
        ),
        "rank_tighten_strength": trial.suggest_float("rank_tighten_strength", 0.0, 0.65),
        "rank_tighten_center": trial.suggest_float("rank_tighten_center", 0.75, 0.98),
        "rank_tighten_floor": trial.suggest_float("rank_tighten_floor", 0.45, 0.95),
        "rank_tighten_power": trial.suggest_float("rank_tighten_power", 0.50, 2.50),
        "exit_pressure_enabled": trial.suggest_categorical(
            "exit_pressure_enabled", [False, True]
        ),
    }
    if params["exit_pressure_enabled"]:
        params.update(
            {
                "exit_pressure_beta": trial.suggest_float("exit_pressure_beta", 0.05, 0.80),
                "exit_pressure_kappa": trial.suggest_float("exit_pressure_kappa", 0.0, 0.80),
                "exit_pressure_min_multiplier": trial.suggest_float(
                    "exit_pressure_min_multiplier", 0.45, 0.95
                ),
                "target_holding_hours": trial.suggest_float("target_holding_hours", 4.0, 18.0),
                "redeploy_scale_bps": trial.suggest_float("redeploy_scale_bps", 40.0, 220.0),
            }
        )
    adverse_enabled = trial.suggest_categorical("adverse_exit_enabled", [False, True])
    params["adverse_exit_enabled"] = adverse_enabled
    if adverse_enabled:
        params.update(
            {
                "adverse_exit_min_mae_atr": trial.suggest_float(
                    "adverse_exit_min_mae_atr", 0.25, 1.40
                ),
                "adverse_exit_min_speed": trial.suggest_float(
                    "adverse_exit_min_speed", 0.08, 0.90
                ),
                "adverse_exit_theta_quantile": trial.suggest_float(
                    "adverse_exit_theta_quantile", 0.55, 0.90
                ),
                "adverse_exit_max_mfe_atr": trial.suggest_float(
                    "adverse_exit_max_mfe_atr", 0.10, 0.80
                ),
            }
        )
    return params


def _objective_window_deltas(per_trade: pd.DataFrame, window_hours: int) -> np.ndarray:
    if per_trade.empty:
        return np.zeros(0, dtype=np.float64)
    timestamps = pd.to_datetime(per_trade["timestamp"], utc=True, errors="coerce")
    size = per_trade["accepted_position_size"].to_numpy(dtype=np.float64)
    baseline = per_trade["t1_net_return"].to_numpy(dtype=np.float64) * size
    delta = per_trade["net_pnl_replay"].to_numpy(dtype=np.float64) - baseline
    freq = f"{max(int(window_hours), 1)}h"
    grouped = pd.DataFrame({"bucket": timestamps.dt.floor(freq), "delta": delta}).groupby(
        "bucket"
    )["delta"].sum()
    return grouped.to_numpy(dtype=np.float64)


def _day_pnl_quantile(
    timestamps: pd.Series,
    pnl: np.ndarray,
    *,
    quantile: float,
) -> float:
    if len(pnl) == 0:
        return 0.0
    work = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True, errors="coerce"),
            "pnl": np.asarray(pnl, dtype=np.float64),
        }
    ).dropna(subset=["timestamp"])
    if work.empty:
        return 0.0
    work["day"] = work["timestamp"].dt.floor("D")
    day_pnl = work.groupby("day", sort=True)["pnl"].sum().to_numpy(dtype=np.float64)
    return float(np.nanquantile(day_pnl, float(quantile))) if day_pnl.size else 0.0


def _robust_quantile_components(per_trade: pd.DataFrame) -> dict[str, float]:
    if per_trade.empty:
        return {
            "day_q05_delta": 0.0,
            "day_q15_delta": 0.0,
            "robust_quantile_delta": 0.0,
        }
    timestamps = per_trade["timestamp"]
    size = per_trade["accepted_position_size"].to_numpy(dtype=np.float64)
    baseline_pnl = per_trade["t1_net_return"].to_numpy(dtype=np.float64) * size
    overlay_pnl = per_trade["net_pnl_replay"].to_numpy(dtype=np.float64)

    def _delta_day(q: float) -> float:
        return _day_pnl_quantile(timestamps, overlay_pnl, quantile=q) - _day_pnl_quantile(
            timestamps, baseline_pnl, quantile=q
        )

    out = {
        "day_q05_delta": float(_delta_day(0.05)),
        "day_q15_delta": float(_delta_day(0.15)),
    }
    out["robust_quantile_delta"] = float(
        out["day_q05_delta"]
        + out["day_q15_delta"]
    )
    return out


def _objective_value(
    summary: dict[str, Any],
    per_trade: pd.DataFrame,
    baseline_summary: dict[str, Any],
    *,
    overlay_rate_cap: float,
    objective_window_hours: int,
) -> tuple[float, dict[str, float]]:
    baseline_net = float(baseline_summary.get("net_pnl", 0.0))
    baseline_win = float(baseline_summary.get("win_rate", 0.0))
    baseline_full_sl = float(baseline_summary.get("full_sl_rate", 0.0))
    delta_pnl = _safe_float(summary.get("net_pnl_delta_vs_baseline"), np.nan)
    if not np.isfinite(delta_pnl):
        delta_pnl = float(summary.get("net_pnl", 0.0)) - baseline_net
    window_deltas = _objective_window_deltas(per_trade, objective_window_hours)
    median_window_delta = float(np.median(window_deltas)) if window_deltas.size else delta_pnl
    q25_window_delta = float(np.quantile(window_deltas, 0.25)) if window_deltas.size else delta_pnl
    robust = _robust_quantile_components(per_trade)
    overlay_excess = max(0.0, float(summary.get("overlay_applied_rate", 0.0)) - overlay_rate_cap)
    full_sl_increase = max(0.0, float(summary.get("full_sl_rate", 0.0)) - baseline_full_sl)
    hit_rate_drop = max(0.0, baseline_win - float(summary.get("win_rate", 0.0)))
    loser_worsened = max(
        0.0, _safe_float(summary.get("loser_loss_worsened_vs_baseline"), 0.0)
    )
    # PnL is still dominant, but the penalties stop the optimizer from buying
    # one-window PnL by modifying most trades or converting normal exits into
    # premature full-stop exits.
    objective = (
        delta_pnl
        + 0.10 * robust["robust_quantile_delta"]
        + 0.10 * median_window_delta
        + 0.05 * q25_window_delta
        - 225.0 * overlay_excess
        - 200.0 * full_sl_increase
        - 150.0 * hit_rate_drop
        - 1.25 * loser_worsened
    )
    components = {
        "delta_pnl": float(delta_pnl),
        "median_window_delta": float(median_window_delta),
        "q25_window_delta": float(q25_window_delta),
        **{k: float(v) for k, v in robust.items()},
        "overlay_excess": float(overlay_excess),
        "full_sl_increase": float(full_sl_increase),
        "hit_rate_drop": float(hit_rate_drop),
        "loser_worsened": float(loser_worsened),
    }
    return float(objective), components


def _run_optuna(
    ctx: ReplayContext,
    baseline_summary: dict[str, Any],
    trials: int,
    seed: int,
    *,
    overlay_on_t1: bool = False,
    safe_space: bool = True,
    overlay_rate_cap: float = 0.35,
    objective_window_hours: int = 24,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    import optuna

    records: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_overlay_params(
            trial, overlay_on_t1=overlay_on_t1, safe_space=safe_space
        )
        summary, trades = evaluate_arm(
            f"optuna_trial_{trial.number}",
            ctx,
            params,
            overlay_on_t1=overlay_on_t1,
        )
        utility, components = _objective_value(
            summary,
            trades,
            baseline_summary,
            overlay_rate_cap=overlay_rate_cap,
            objective_window_hours=objective_window_hours,
        )
        record = dict(summary)
        record["trial"] = int(trial.number)
        record["objective_value"] = float(utility)
        record.update({f"objective_{k}": v for k, v in components.items()})
        records.append(record)
        return float(utility)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=int(trials), show_progress_bar=False)
    best_summary, best_trades = evaluate_arm(
        "t1_plus_optuna_best" if overlay_on_t1 else "optuna_best",
        ctx,
        dict(study.best_trial.params),
        overlay_on_t1=overlay_on_t1,
    )
    best_summary["objective_value"] = float(study.best_value)
    trials_df = pd.DataFrame(records).sort_values("objective_value", ascending=False)
    return best_summary, best_trades, trials_df


def _run_per_head_optuna(
    ctx: ReplayContext,
    trials: int,
    seed: int,
    *,
    overlay_on_t1: bool = False,
    safe_space: bool = True,
    overlay_rate_cap: float = 0.35,
    objective_window_hours: int = 24,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    best_params_by_head: dict[str, dict[str, Any]] = {}
    trial_frames: list[pd.DataFrame] = []
    for offset, head in enumerate(sorted(ctx.rows["head"].astype(str).unique())):
        mask = ctx.rows["head"].astype(str).eq(head).to_numpy()
        if not bool(mask.any()):
            continue
        subctx = _subset_context(ctx, mask)
        if subctx.saved_reference_summary is None:
            continue
        best_summary, _, trials_df = _run_optuna(
            subctx,
            dict(subctx.saved_reference_summary),
            trials=int(trials),
            seed=int(seed) + 997 * offset,
            overlay_on_t1=overlay_on_t1,
            safe_space=safe_space,
            overlay_rate_cap=overlay_rate_cap,
            objective_window_hours=objective_window_hours,
        )
        params = json.loads(str(best_summary.get("params_json", "{}")))
        best_params_by_head[head] = dict(params)
        if trials_df is not None and not trials_df.empty:
            trials_df = trials_df.copy()
            trials_df.insert(0, "head", head)
            trial_frames.append(trials_df)
    combined_summary, combined_trades = evaluate_head_params_arm(
        "t1_plus_per_head_optuna_best" if overlay_on_t1 else "per_head_optuna_best",
        ctx,
        best_params_by_head,
        overlay_on_t1=overlay_on_t1,
    )
    baseline_summary = ctx.saved_reference_summary or {}
    objective, components = _objective_value(
        combined_summary,
        combined_trades,
        dict(baseline_summary),
        overlay_rate_cap=overlay_rate_cap,
        objective_window_hours=objective_window_hours,
    )
    combined_summary["objective_value"] = float(objective)
    combined_summary.update({f"objective_{k}": v for k, v in components.items()})
    trials_out = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
    return combined_summary, combined_trades, trials_out, best_params_by_head


def _load_fixed_params(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.fixed_params_json and args.fixed_params_file:
        raise ValueError("Use only one of --fixed-params-json or --fixed-params-file.")
    if args.fixed_params_file:
        return json.loads(Path(args.fixed_params_file).read_text(encoding="utf-8"))
    if args.fixed_params_json:
        return json.loads(str(args.fixed_params_json))
    return None


def _write_report(
    output_dir: Path,
    *,
    artifact_dir: Path,
    ctx: ReplayContext,
    summary: pd.DataFrame,
    trials_df: pd.DataFrame | None,
    overlay_on_t1: bool = False,
) -> None:
    period_start = pd.to_datetime(ctx.rows["timestamp"], utc=True).min()
    period_end = pd.to_datetime(ctx.rows["timestamp"], utc=True).max()
    lines = [
        "# Open-Position Score Exit Ablation",
        "",
        f"- Source artifact: `{artifact_dir}`",
        f"- Period: `{period_start}` to `{period_end}`",
        f"- Fixed entries replayed: `{len(ctx.rows)}`",
        f"- Costs: `{DEFAULT_POLICY_PER_SIDE_COST_PCT:.6f}` per side, `{DEFAULT_POLICY_PER_SIDE_COST_PCT * 2.0:.6f}` round trip",
        "- Scope: fixed T1 entries only; no portfolio re-ranking, no new entries, no capacity changes.",
        "- Current limitation: no repeated post-entry refreshed score path exists in this historical ledger; rank-based arms use entry-rank proxies.",
        (
            "- Overlay mode: enabled. `saved_T1_reference` is the baseline; an overlay can only replace a T1 exit when it triggers no later than the materialized T1 exit."
            if overlay_on_t1
            else "- Overlay mode: disabled. `baseline_replay` is the reconstructed simulator baseline used for internally comparable exit deltas."
        ),
        "",
        "## Arm Summary",
        "",
        summary[
            [
                "arm",
                "trade_count",
                "net_pnl",
                "gross_pnl",
                "cost_pnl",
                "win_rate",
                "full_sl_rate",
                "timeout_rate",
                "trailing_rate",
                "adverse_exit_rate",
                "hard_tp_rate",
                "overlay_applied_rate",
                "net_pnl_delta_vs_baseline",
                "loss_avoided_vs_baseline",
                "loser_loss_worsened_vs_baseline",
                "winner_pnl_sacrificed_vs_baseline",
                "defensive_success_vs_baseline",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    if trials_df is not None and not trials_df.empty:
        lines.extend(
            [
                "## Top Optuna Trials",
                "",
                trials_df.head(10)[
                    [
                        "trial",
                        "objective_value",
                        "net_pnl",
                        "win_rate",
                        "full_sl_rate",
                        "q05_net_return",
                        "q15_net_return",
                        "defensive_success_vs_baseline",
                        "objective_delta_pnl",
                        "objective_robust_quantile_delta",
                        "objective_day_q05_delta",
                        "objective_overlay_excess",
                        "objective_full_sl_increase",
                        "objective_hit_rate_drop",
                        "objective_loser_worsened",
                        "params_json",
                    ]
                ].to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        default="data_perp/artifacts/reliability_blend_T1_global_rank_static_baseline_active_20260627",
    )
    parser.add_argument(
        "--extra-artifact-dir",
        action="append",
        default=[],
        help="Additional fixed-entry artifact directory to include in the HPO/evaluation context.",
    )
    parser.add_argument(
        "--accepted-report-path",
        action="append",
        default=[],
        help="Accepted-trade parquet report with embedded candidate_* columns to include in the HPO/evaluation context.",
    )
    parser.add_argument(
        "--output-dir",
        default="data_perp/reports/open_position_score_exit_ablation_20260630_t1_jun15_22",
    )
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--overlay-on-t1",
        action="store_true",
        help="Apply exit changes as protective overlays on the saved T1 exits.",
    )
    parser.add_argument(
        "--fixed-params-json",
        default="",
        help="JSON object of frozen overlay/replay parameters to evaluate without re-optimizing.",
    )
    parser.add_argument(
        "--fixed-params-file",
        default="",
        help="Path to a JSON object of frozen overlay/replay parameters to evaluate without re-optimizing.",
    )
    parser.add_argument(
        "--fixed-arm-name",
        default="fixed_params_overlay",
        help="Arm name used when --fixed-params-json or --fixed-params-file is provided.",
    )
    parser.add_argument(
        "--skip-static-arms",
        action="store_true",
        help="Only write the saved reference and fixed/Optuna arms; skip built-in static ablation arms.",
    )
    parser.add_argument(
        "--per-head-optuna",
        action="store_true",
        help="Run one Optuna study per head and combine the head-specific best overlays.",
    )
    parser.add_argument(
        "--unsafe-search-space",
        action="store_true",
        help="Use the original broad search space instead of the safer overlay bounds.",
    )
    parser.add_argument(
        "--overlay-rate-cap",
        type=float,
        default=0.35,
        help="Penalty threshold for overlay_applied_rate in the Optuna objective.",
    )
    parser.add_argument(
        "--objective-window-hours",
        type=int,
        default=24,
        help="Window size used for median/q25 PnL-delta terms in the Optuna objective.",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dirs = [artifact_dir] + [Path(p) for p in args.extra_artifact_dir]
    accepted_report_paths = [Path(p) for p in args.accepted_report_path]
    output_dir = Path(args.output_dir)
    if len(artifact_dirs) == 1 and not accepted_report_paths:
        ctx = _build_context(
            artifact_dir=artifact_dir,
            output_dir=output_dir,
            data_root=str(args.data_root),
            market_mode=str(args.market_mode),
            max_rows=int(args.max_rows),
        )
    else:
        ctx = _build_multi_context(
            artifact_dirs=artifact_dirs,
            accepted_report_paths=accepted_report_paths,
            output_dir=output_dir,
            data_root=str(args.data_root),
            market_mode=str(args.market_mode),
            max_rows=int(args.max_rows),
        )

    summaries: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    baseline_summary: dict[str, Any] | None = None
    baseline_returns: np.ndarray | None = None
    baseline_sizes: np.ndarray | None = None

    if ctx.saved_reference_summary is not None:
        summaries.append(dict(ctx.saved_reference_summary))
        if args.overlay_on_t1:
            baseline_summary = dict(ctx.saved_reference_summary)
    if args.overlay_on_t1:
        _, replay_reference_trades = evaluate_arm(
            "baseline_replay_reference",
            ctx,
            {},
            overlay_on_t1=False,
        )
        ctx = ReplayContext(
            rows=ctx.rows,
            paths=ctx.paths,
            output_dir=ctx.output_dir,
            saved_reference_summary=ctx.saved_reference_summary,
            replay_reference_net_returns=replay_reference_trades[
                "net_return_replay"
            ].to_numpy(dtype=np.float64),
            replay_reference_exit_bars=replay_reference_trades[
                "sim_exit_bars"
            ].to_numpy(dtype=np.float64),
            replay_reference_exit_reasons=replay_reference_trades[
                "sim_exit_reason"
            ].astype(str).to_numpy(),
        )

    if not args.skip_static_arms:
        for name, params in _static_arms(overlay_on_t1=bool(args.overlay_on_t1)):
            summary, trades = evaluate_arm(
                name,
                ctx,
                params,
                overlay_on_t1=bool(args.overlay_on_t1),
            )
            if name == "baseline_replay" and not args.overlay_on_t1:
                baseline_summary = summary
                baseline_returns = trades["net_return_replay"].to_numpy(dtype=np.float64)
                baseline_sizes = trades["accepted_position_size"].to_numpy(dtype=np.float64)
                ctx = ReplayContext(
                    rows=ctx.rows,
                    paths=ctx.paths,
                    output_dir=ctx.output_dir,
                    saved_reference_summary=ctx.saved_reference_summary,
                    baseline_net_returns=baseline_returns,
                    baseline_position_sizes=baseline_sizes,
                    replay_reference_net_returns=ctx.replay_reference_net_returns,
                    replay_reference_exit_bars=ctx.replay_reference_exit_bars,
                    replay_reference_exit_reasons=ctx.replay_reference_exit_reasons,
                )
                summary, trades = evaluate_arm(name, ctx, params)
            summaries.append(summary)
            trade_frames.append(trades)

    fixed_params = _load_fixed_params(args)
    if fixed_params is not None:
        fixed_arm_name = str(args.fixed_arm_name)
        if args.overlay_on_t1 and not fixed_arm_name.startswith("t1_plus_"):
            fixed_arm_name = f"t1_plus_{fixed_arm_name}"
        if isinstance(fixed_params.get("per_head"), dict):
            summary, trades = evaluate_head_params_arm(
                fixed_arm_name,
                ctx,
                {
                    str(head): dict(params)
                    for head, params in fixed_params["per_head"].items()
                },
                overlay_on_t1=bool(args.overlay_on_t1),
            )
        else:
            summary, trades = evaluate_arm(
                fixed_arm_name,
                ctx,
                fixed_params,
                overlay_on_t1=bool(args.overlay_on_t1),
            )
        summaries.append(summary)
        trade_frames.append(trades)

    trials_df: pd.DataFrame | None = None
    best_head_params: dict[str, dict[str, Any]] | None = None
    if args.trials and args.trials > 0:
        if baseline_summary is None:
            raise RuntimeError("Baseline summary missing before Optuna.")
        if args.per_head_optuna:
            best_summary, best_trades, trials_df, best_head_params = _run_per_head_optuna(
                ctx,
                trials=int(args.trials),
                seed=int(args.seed),
                overlay_on_t1=bool(args.overlay_on_t1),
                safe_space=not bool(args.unsafe_search_space),
                overlay_rate_cap=float(args.overlay_rate_cap),
                objective_window_hours=int(args.objective_window_hours),
            )
        else:
            best_summary, best_trades, trials_df = _run_optuna(
                ctx,
                baseline_summary,
                trials=int(args.trials),
                seed=int(args.seed),
                overlay_on_t1=bool(args.overlay_on_t1),
                safe_space=not bool(args.unsafe_search_space),
                overlay_rate_cap=float(args.overlay_rate_cap),
                objective_window_hours=int(args.objective_window_hours),
            )
        summaries.append(best_summary)
        trade_frames.append(best_trades)

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values("net_pnl", ascending=False).reset_index(drop=True)
    all_trades = pd.concat(trade_frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    all_trades.to_parquet(output_dir / "per_trade_results.parquet", index=False)
    if trials_df is not None:
        trials_df.to_csv(output_dir / "optuna_trials.csv", index=False)
    if best_head_params is not None:
        (output_dir / "per_head_best_params.json").write_text(
            json.dumps(best_head_params, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    _write_report(
        output_dir,
        artifact_dir=artifact_dir,
        ctx=ctx,
        summary=summary_df,
        trials_df=trials_df,
        overlay_on_t1=bool(args.overlay_on_t1),
    )
    print(f"Wrote {output_dir / 'report.md'}")
    print(summary_df[["arm", "trade_count", "net_pnl", "win_rate", "full_sl_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
