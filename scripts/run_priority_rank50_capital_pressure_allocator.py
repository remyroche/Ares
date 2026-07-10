#!/usr/bin/env python3
"""Wire the capital-pressure allocator onto the priority_rank50_focus policy.

This runner rebuilds the promoted hit-surprise priority-rank portfolio
candidate stream from replay-ready S52 candidates, then applies the existing
pressure allocator from ``run_portfolio_marginal_utility_ablation.py``.

Default contract:
- baseline candidate stream: top10 + priority_rank50_focus archetype nudges
- pressure params: optimised on May 2026, evaluated on June 2026
- replay config: saved ``optimized_portfolio_policy_config.json``
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    fit_hierarchical_ev_curves,
    load_portfolio_policy_params,
    normalise_candidate_table,
    replay_candidates,
)
from scripts.ablate_s52_archetype_hit_surprise_thresholds import (  # noqa: E402
    TOP_THRESHOLDS,
    _apply_portfolio_hr_adjustments,
    _load_parent_summary,
    _path_take,
    _portfolio_candidate_table,
    _simulate_selected_rows,
)
from scripts.backtest_exit_tightening_redeploy import (  # noqa: E402
    DEFAULT_OHLCV_ROOT,
    MAX_PATH_MINUTES,
    ExitTighteningConfig,
    _build_paths,
    _edge_and_cost_columns,
    _make_exit_adjust_callback,
    _timestamp_edge_table,
)
from scripts.ablate_simple_policy_exit_geometry import (  # noqa: E402
    _load_bundles,
    _prepare_rows,
)
from scripts.run_global_portfolio_period_multiplier import (  # noqa: E402
    _add_open_position_concentration_features,
    _add_portfolio_state_features,
    _json_safe,
    _timestamp_feature_fill_values,
    _timestamp_features,
)
from scripts.run_portfolio_marginal_utility_ablation import (  # noqa: E402
    _apply_pressure_reallocation,
    _optimise_reallocation_params,
    _replay_pressure_reallocation,
)
from scripts.run_s52_side_archetype_simple_policy_optimiser import (  # noqa: E402
    _params_from_parent_summary_row,
)
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    calculate_advanced_metrics,
    simulate_and_score,
)


DEFAULT_ARTIFACT_DIR = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_ae3000_"
    "nocrossfit_k34567_payload300k_20260708_may_june_hr_priority_rank50_focus"
)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _quality_by_archetype(policy: Mapping[str, Any]) -> dict[str, float]:
    rows = policy.get("archetype_adjustments", [])
    if not isinstance(rows, list):
        return {}
    out: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        archetype = str(row.get("policy_archetype", "missing"))
        out[archetype] = _safe_float(row.get("quality_adjustment"), 0.0)
    return out


def _simulate_selected_rows_with_paths(
    rows: pd.DataFrame,
    paths: tuple[np.ndarray, ...],
    *,
    rank_threshold: float,
    params: Mapping[str, Any],
    size_power: float,
    cost_pct: float,
) -> tuple[pd.DataFrame, dict[str, Any], tuple[np.ndarray, ...]]:
    """Replay a strategy slice and return selected rows plus aligned paths."""
    if rows.empty or "rank_pct" not in rows.columns:
        empty = rows.iloc[0:0].copy()
        return empty, {}, _path_take(paths, np.array([], dtype=np.int64))
    rank = pd.to_numeric(rows["rank_pct"], errors="coerce").to_numpy(dtype=np.float64)
    idx = np.flatnonzero(np.isfinite(rank) & (rank >= float(rank_threshold)))
    if idx.size == 0:
        empty = rows.iloc[0:0].copy()
        return empty, {}, _path_take(paths, np.array([], dtype=np.int64))
    sub = rows.iloc[idx].copy().reset_index(drop=True)
    sub_paths = _path_take(paths, idx)
    metrics = simulate_and_score(
        sub,
        *sub_paths,
        cost_pct=float(cost_pct),
        size_power=float(size_power),
        **dict(params),
    )
    adv = calculate_advanced_metrics(
        sub,
        metrics.get("raw_gains", np.array([], dtype=np.float32)),
        metrics.get("sizes", np.array([], dtype=np.float32)),
        metrics.get("selected_mask"),
        metrics.get("gross_gains"),
        metrics.get("exit_reason"),
        metrics.get("exit_bars"),
    )
    selected_mask = np.asarray(metrics.get("selected_mask", np.zeros(len(sub), dtype=bool)), dtype=bool)
    selected_idx = np.flatnonzero(selected_mask)
    selected = sub.loc[selected_mask].copy().reset_index(drop=True)
    selected_paths = _path_take(sub_paths, selected_idx)
    raw = np.asarray(metrics.get("raw_gains", np.array([], dtype=np.float32)), dtype=np.float64)
    gross = np.asarray(metrics.get("gross_gains", np.array([], dtype=np.float32)), dtype=np.float64)
    sizes = np.asarray(metrics.get("sizes", np.array([], dtype=np.float32)), dtype=np.float64)
    exit_reason = np.asarray(metrics.get("exit_reason", np.array([], dtype=object))).astype(str)
    exit_bars = np.asarray(metrics.get("exit_bars", np.array([], dtype=np.int16)), dtype=np.float64)
    n = min(len(selected), len(raw), len(gross), len(sizes), len(exit_reason), len(exit_bars), len(selected_idx))
    selected = selected.iloc[:n].copy()
    selected_paths = _path_take(selected_paths, np.arange(n, dtype=np.int64))
    if n:
        selected["net_gain"] = raw[:n]
        selected["gross_gain"] = gross[:n]
        selected["position_size"] = sizes[:n]
        with np.errstate(divide="ignore", invalid="ignore"):
            selected["ret_net_notional"] = selected["net_gain"].to_numpy(dtype=np.float64) / np.maximum(sizes[:n], 1e-12)
            selected["ret_gross_notional"] = selected["gross_gain"].to_numpy(dtype=np.float64) / np.maximum(sizes[:n], 1e-12)
        selected["exit_reason"] = exit_reason[:n]
        selected["exit_bars"] = np.maximum(exit_bars[:n], 1.0)
        selected["expected_probability"] = selected.get("calibrated_score", selected.get("rank_pct", 0.5))
        entry = selected_paths[3][:, 0].astype(np.float64, copy=False) if len(selected_paths) >= 4 else np.full(n, np.nan)
        selected["entry_price_real"] = np.where(np.isfinite(entry) & (entry > 0.0), entry, np.nan)
    return selected, adv, selected_paths


def _build_priority_rank50_candidates(
    *,
    artifact_dir: Path,
    min_ts: pd.Timestamp,
    max_ts: pd.Timestamp,
    data_root: str,
    market_mode: str,
    path_len: int,
) -> pd.DataFrame:
    manifest = _load_json(artifact_dir / "manifest.json")
    policy = _load_json(artifact_dir / "policy_params" / "hit_surprise_archetype_portfolio_policy.json")
    source_candidates = Path(manifest["candidates"])
    parent_summary = _load_parent_summary(Path(manifest["parent_policy_summary"]))
    rows = _prepare_rows(source_candidates, min_rank=float(TOP_THRESHOLDS["top10"]))
    ts = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    rows = rows.loc[ts.ge(min_ts) & ts.lt(max_ts)].copy()
    if rows.empty:
        raise ValueError(f"No top10 rows in {min_ts} -> {max_ts}")
    bundles = _load_bundles(
        rows,
        data_root=str(data_root),
        market_mode=str(market_mode),
        path_len=int(path_len),
        min_rows_per_strategy=5,
    )
    quality = _quality_by_archetype(policy)
    frames: list[pd.DataFrame] = []
    for bundle in bundles:
        strategy_id = str(bundle.strategy_id)
        if strategy_id not in parent_summary:
            continue
        params, size_power = _params_from_parent_summary_row(parent_summary[strategy_id])
        selected, _metrics = _simulate_selected_rows(
            bundle.rows,
            bundle.paths,
            rank_threshold=float(TOP_THRESHOLDS["top10"]),
            params=params,
            size_power=size_power,
            cost_pct=0.005,
        )
        if selected.empty:
            continue
        selected = selected.copy()
        selected["mode"] = "hit_surprise_priority_rank_50"
        selected["top_slice"] = "top10"
        selected["base_rank_threshold"] = float(TOP_THRESHOLDS["top10"])
        selected["half_life_days"] = float(policy.get("selection", {}).get("half_life_days", 14.0))
        selected["alpha"] = float(policy.get("selection", {}).get("alpha", 0.25))
        selected["max_adjust"] = float(policy.get("selection", {}).get("max_adjust", 0.05))
        selected = _apply_portfolio_hr_adjustments(
            selected,
            mode="hit_surprise_priority_rank_50",
            quality_by_archetype=quality,
            max_adjust=float(policy.get("selection", {}).get("max_adjust", 0.05)),
        )
        frames.append(selected)
    if not frames:
        raise ValueError("No replayable priority-rank selected rows were generated")
    return _portfolio_candidate_table(pd.concat(frames, ignore_index=True))


def _build_priority_rank50_candidates_with_context(
    *,
    artifact_dir: Path,
    min_ts: pd.Timestamp,
    max_ts: pd.Timestamp,
    data_root: str,
    market_mode: str,
    path_len: int,
) -> tuple[pd.DataFrame, tuple[np.ndarray, ...]]:
    """Build the same promoted candidate stream plus 15m path context.

    The returned paths are aligned to the returned candidate frame through
    ``candidate_uid`` after the portfolio candidate table's sort/drop phase.
    They are only used as a construction aid; the advanced trial rebuilds 1m
    paths before replay because the exit-tightening simulator is minute-level.
    """
    manifest = _load_json(artifact_dir / "manifest.json")
    policy = _load_json(artifact_dir / "policy_params" / "hit_surprise_archetype_portfolio_policy.json")
    source_candidates = Path(manifest["candidates"])
    parent_summary = _load_parent_summary(Path(manifest["parent_policy_summary"]))
    rows = _prepare_rows(source_candidates, min_rank=float(TOP_THRESHOLDS["top10"]))
    ts = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    rows = rows.loc[ts.ge(min_ts) & ts.lt(max_ts)].copy()
    if rows.empty:
        raise ValueError(f"No top10 rows in {min_ts} -> {max_ts}")
    bundles = _load_bundles(
        rows,
        data_root=str(data_root),
        market_mode=str(market_mode),
        path_len=int(path_len),
        min_rows_per_strategy=5,
    )
    quality = _quality_by_archetype(policy)
    frames: list[pd.DataFrame] = []
    path_chunks: list[tuple[np.ndarray, ...]] = []
    next_uid = 0
    for bundle in bundles:
        strategy_id = str(bundle.strategy_id)
        if strategy_id not in parent_summary:
            continue
        params, size_power = _params_from_parent_summary_row(parent_summary[strategy_id])
        selected, _metrics, selected_paths = _simulate_selected_rows_with_paths(
            bundle.rows,
            bundle.paths,
            rank_threshold=float(TOP_THRESHOLDS["top10"]),
            params=params,
            size_power=size_power,
            cost_pct=0.005,
        )
        if selected.empty:
            continue
        selected = selected.copy()
        n = len(selected)
        selected["candidate_uid"] = np.arange(next_uid, next_uid + n, dtype=np.int64)
        next_uid += n
        selected["mode"] = "hit_surprise_priority_rank_50"
        selected["top_slice"] = "top10"
        selected["base_rank_threshold"] = float(TOP_THRESHOLDS["top10"])
        selected["half_life_days"] = float(policy.get("selection", {}).get("half_life_days", 14.0))
        selected["alpha"] = float(policy.get("selection", {}).get("alpha", 0.25))
        selected["max_adjust"] = float(policy.get("selection", {}).get("max_adjust", 0.05))
        selected = _apply_portfolio_hr_adjustments(
            selected,
            mode="hit_surprise_priority_rank_50",
            quality_by_archetype=quality,
            max_adjust=float(policy.get("selection", {}).get("max_adjust", 0.05)),
        )
        frames.append(selected)
        path_chunks.append(selected_paths)
    if not frames:
        raise ValueError("No replayable priority-rank selected rows were generated")
    selected_all = pd.concat(frames, ignore_index=True, copy=False)
    all_paths = tuple(np.concatenate([chunk[i] for chunk in path_chunks], axis=0) for i in range(len(path_chunks[0])))
    candidates = _portfolio_candidate_table(selected_all)
    if "candidate_uid" not in candidates.columns:
        raise RuntimeError("candidate_uid was dropped from the portfolio candidate table")
    uid = pd.to_numeric(candidates["candidate_uid"], errors="coerce").astype("Int64")
    keep = uid.notna()
    candidates = candidates.loc[keep].copy().reset_index(drop=True)
    uid_arr = uid.loc[keep].astype(int).to_numpy(dtype=np.int64)
    aligned_paths = _path_take(all_paths, uid_arr)
    if "entry_price_real" in candidates.columns:
        real_entry = pd.to_numeric(candidates["entry_price_real"], errors="coerce")
        candidates["entry_price"] = real_entry.where(real_entry.gt(0.0), candidates["entry_price"])
        side = candidates["side"].astype(str).str.lower()
        gross = pd.to_numeric(candidates["gross_return"], errors="coerce").fillna(0.0)
        entry = pd.to_numeric(candidates["entry_price"], errors="coerce").fillna(1.0)
        candidates["exit_price"] = np.where(
            side.str.contains("short|-1", regex=True),
            entry / np.maximum(1.0 + gross, 1e-9),
            entry * (1.0 + gross),
        )
    return candidates, aligned_paths


def _accepted_with_candidates(candidates: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame()
    idx = pd.to_numeric(accepted["candidate_index"], errors="coerce").astype("Int64")
    accepted = accepted.loc[idx.notna()].copy().reset_index(drop=True)
    # replay_candidates() normalises and sorts the candidate table internally.
    # candidate_index refers to that canonical replay order, not to the raw
    # input parquet order.
    cand = normalise_candidate_table(candidates).iloc[
        idx.loc[idx.notna()].astype(int).to_numpy()
    ].reset_index(drop=True)
    useful = [
        "timestamp",
        "symbol",
        "side",
        "strategy_id",
        "policy_archetype",
        "normalized_rank_score",
        "base_strategy_threshold",
        "net_return",
        "gross_return",
        "simple_policy_exit_reason",
        "exit_timestamp",
        "portfolio_priority_multiplier",
        "portfolio_rank_adjustment",
        "portfolio_priority_adjustment",
        "portfolio_size_multiplier",
        "portfolio_reallocation_pressure",
        "portfolio_reallocation_top_score",
        "portfolio_reallocation_weak_score",
    ]
    for col in useful:
        if col in cand.columns:
            accepted[col] = cand[col].to_numpy()
    if "position_exit_timestamp" in accepted.columns:
        accepted["exit_timestamp"] = pd.to_datetime(
            accepted["position_exit_timestamp"],
            utc=True,
            errors="coerce",
        ).where(
            pd.to_datetime(accepted["position_exit_timestamp"], utc=True, errors="coerce").notna(),
            accepted.get("exit_timestamp"),
        )
    if "position_net_return" in accepted.columns:
        adjusted_net = pd.to_numeric(accepted["position_net_return"], errors="coerce")
        accepted["net_return"] = adjusted_net.where(adjusted_net.notna(), accepted["net_return"])
    if "position_gross_return" in accepted.columns:
        adjusted_gross = pd.to_numeric(accepted["position_gross_return"], errors="coerce")
        accepted["gross_return"] = adjusted_gross.where(adjusted_gross.notna(), accepted["gross_return"])
    if "position_exit_reason" in accepted.columns:
        adjusted_reason = accepted["position_exit_reason"].astype(str)
        accepted["simple_policy_exit_reason"] = adjusted_reason.where(
            adjusted_reason.str.len() > 0,
            accepted.get("simple_policy_exit_reason", ""),
        )
    accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    if "exit_timestamp" in accepted.columns:
        accepted["exit_timestamp"] = pd.to_datetime(accepted["exit_timestamp"], utc=True, errors="coerce")
        bad_exit = accepted["exit_timestamp"].notna() & accepted["timestamp"].notna() & (
            accepted["exit_timestamp"] < accepted["timestamp"]
        )
        if bool(bad_exit.any()):
            sample = accepted.loc[
                bad_exit,
                ["timestamp", "exit_timestamp", "symbol", "side", "candidate_index"],
            ].head(5)
            raise ValueError(
                "accepted replay rows contain exits before entry timestamps; "
                f"sample={sample.to_dict(orient='records')}"
            )
    accepted["position_size"] = pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
    accepted["net_return"] = pd.to_numeric(accepted["net_return"], errors="coerce").fillna(0.0)
    accepted["gross_return"] = pd.to_numeric(accepted["gross_return"], errors="coerce").fillna(0.0)
    accepted["net_pnl"] = accepted["position_size"] * accepted["net_return"]
    accepted["gross_pnl"] = accepted["position_size"] * accepted["gross_return"]
    accepted["cost_pnl"] = accepted["gross_pnl"] - accepted["net_pnl"]
    accepted["month"] = accepted["timestamp"].dt.to_period("M").astype(str)
    accepted["week_start"] = accepted["timestamp"].dt.to_period("W-MON").apply(lambda p: p.start_time).astype(str)
    accepted["side_name"] = np.where(accepted["side"].astype(str).str.lower().str.contains("short|-1"), "short", "long")
    reason = accepted.get("simple_policy_exit_reason", pd.Series("", index=accepted.index)).astype(str).str.lower()
    accepted["full_sl"] = reason.isin(["sl", "full_sl", "stop", "stop_loss"])
    accepted["timeout"] = reason.str.contains("timeout", regex=False)
    accepted["ev_redeployment_early_exit"] = reason.str.contains("ev_redeployment", regex=False)
    accepted["advanced_ev_redeployment_early_exit"] = (
        reason.str.contains("tightened_", regex=False) | reason.str.contains("force_exit", regex=False)
    )
    return accepted


def _replay_baseline(candidates: pd.DataFrame, params: Any, ev_curve: dict[str, Any], *, market_mode: str):
    decisions, equity, metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = _accepted_with_candidates(candidates, decisions)
    return decisions, equity, metrics, accepted


def _make_ev_redeployment_callback(
    *,
    max_concurrent_positions: int,
    pressure_start: float = 0.65,
    remaining_capital_frac: float = 0.25,
    min_future_rank: float = 0.955,
    rank_margin: float = 0.025,
    lookahead_hours: float = 12.0,
    min_hold_bars: int = 1,
    churn_penalty_bps: float = 100.0,
):
    """Return an accepted-position callback that schedules early EV redeployment.

    The replay engine only allows entry-time exit adjustment. This callback
    approximates live rotation by shortening the accepted position to the first
    later timestamp where a stronger OOS candidate is available while capital is
    pressured. It uses only candidate scores/timestamps, not realized future PnL.
    """

    def _callback(
        idx: int,
        ts: pd.Timestamp,
        state: Any,
        cache: Any,
        position_size: float,
        capital_limit: float,
        remaining_capital: float,
        group_idx: np.ndarray,
    ) -> dict[str, Any] | None:
        current_open = int(len(getattr(state, "open_positions", []))) + 1
        occupancy = current_open / max(int(max_concurrent_positions), 1)
        remaining_after = max(float(remaining_capital) - float(position_size), 0.0)
        remaining_frac = remaining_after / max(float(capital_limit), 1e-12)
        pressure = max(
            (occupancy - float(pressure_start)) / max(1.0 - float(pressure_start), 1e-6),
            0.0,
        )
        if pressure <= 0.0 and remaining_frac > float(remaining_capital_frac):
            return None

        original_exit = pd.Timestamp(cache.exit_timestamp[idx])
        if pd.isna(original_exit) or original_exit <= ts:
            return None
        min_exit = ts + pd.Timedelta(minutes=15 * max(int(min_hold_bars), 1))
        max_exit = min(original_exit, ts + pd.Timedelta(hours=float(lookahead_hours)))
        if max_exit <= min_exit:
            return None

        frame_ts = pd.to_datetime(cache.frame["timestamp"], utc=True, errors="coerce")
        current_score = float(
            np.clip(
                cache.rank_score[idx] + cache.portfolio_rank_adjustment[idx],
                0.0,
                1.0,
            )
        )
        effective_rank = np.clip(
            cache.rank_score + cache.portfolio_rank_adjustment,
            0.0,
            1.0,
        )
        mask = (
            frame_ts.gt(min_exit)
            & frame_ts.le(max_exit)
            & (effective_rank >= max(float(min_future_rank), current_score + float(rank_margin)))
            & (effective_rank >= cache.base_threshold)
        )
        if not bool(mask.any()):
            return None
        future_idx = np.flatnonzero(mask.to_numpy())
        if future_idx.size == 0:
            return None
        future_ts = frame_ts.iloc[future_idx].to_numpy()
        order = np.lexsort((-effective_rank[future_idx], future_ts))
        replacement_idx = int(future_idx[order[0]])
        replacement_ts = pd.Timestamp(frame_ts.iloc[replacement_idx])
        if replacement_ts <= min_exit or replacement_ts >= original_exit:
            return None

        total_seconds = max(float((original_exit - ts).total_seconds()), 1.0)
        elapsed_seconds = max(float((replacement_ts - ts).total_seconds()), 0.0)
        progress = float(np.clip(elapsed_seconds / total_seconds, 0.0, 1.0))
        original_gross = float(cache.gross_return[idx])
        original_net = float(cache.net_return[idx])
        fee_return = max(float(original_gross - original_net), 0.0)
        gross_return = float(original_gross * progress)
        net_return = float(gross_return - fee_return - float(churn_penalty_bps) / 10_000.0)
        return {
            "exit_timestamp": replacement_ts,
            "gross_return": gross_return,
            "net_return": net_return,
            "exit_reason": "ev_redeployment_early_exit",
        }

    return _callback


def _replay_with_ev_redeployment(
    candidates: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    market_mode: str,
    redeploy_params: Mapping[str, Any],
):
    decisions, equity, metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
        accepted_position_callback=_make_ev_redeployment_callback(
            max_concurrent_positions=int(params.max_concurrent_positions),
            pressure_start=float(redeploy_params.get("pressure_start", 0.65)),
            remaining_capital_frac=float(redeploy_params.get("remaining_capital_frac", 0.25)),
            min_future_rank=float(redeploy_params.get("min_future_rank", 0.955)),
            rank_margin=float(redeploy_params.get("rank_margin", 0.025)),
            lookahead_hours=float(redeploy_params.get("lookahead_hours", 12.0)),
            min_hold_bars=int(redeploy_params.get("min_hold_bars", 1)),
            churn_penalty_bps=float(redeploy_params.get("churn_penalty_bps", 100.0)),
        ),
    )
    accepted = _accepted_with_candidates(candidates, decisions)
    return decisions, equity, metrics, accepted


def _advanced_exit_config(profile: str) -> ExitTighteningConfig:
    if str(profile) == "aggressive":
        return ExitTighteningConfig(
            config_id="priority_rank50_advanced_ev_redeploy_aggressive_trial_v1",
            candidate_edge_quantile=0.65,
            pressure_mode="count",
            pressure_mid=0.10,
            pressure_power=1.0,
            churn_penalty_bps=0.0,
            exit_hysteresis_bps=5.0,
            base_stop_loss_bps=80.0,
            min_stop_loss_bps=5.0,
            base_trailing_gap_bps=70.0,
            min_trailing_gap_bps=5.0,
            base_tp_remaining_bps=120.0,
            min_tp_remaining_bps=10.0,
            pressure_use_mode="convex",
        )
    return ExitTighteningConfig(
        config_id="priority_rank50_advanced_ev_redeploy_balanced_trial_v1",
        candidate_edge_quantile=0.75,
        pressure_mode="count",
        pressure_mid=0.50,
        pressure_power=1.5,
        churn_penalty_bps=5.0,
        exit_hysteresis_bps=25.0,
        base_stop_loss_bps=80.0,
        min_stop_loss_bps=25.0,
        base_trailing_gap_bps=70.0,
        min_trailing_gap_bps=20.0,
        base_tp_remaining_bps=120.0,
        min_tp_remaining_bps=35.0,
        pressure_use_mode="hierarchical",
    )


def _side_codes(frame: pd.DataFrame) -> np.ndarray:
    side = frame.get("side", pd.Series("long", index=frame.index)).astype(str).str.lower()
    return np.where(side.str.contains("short|-1", regex=True), -1, 1).astype(np.int8)


def _advanced_exit_arrays(
    candidates: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    params: Any,
) -> dict[str, np.ndarray]:
    edge_frame = candidates.copy()
    # The candidate returns already use the promoted 1% round-trip cost. Keep the
    # advanced callback on the same cost definition instead of adding spread a
    # second time during the trial.
    if "barrier_pct" not in edge_frame.columns:
        edge_frame["barrier_pct"] = 0.005
    if "policy_effective_barrier_pct" not in edge_frame.columns:
        edge_frame["policy_effective_barrier_pct"] = edge_frame["barrier_pct"]
    edge_frame["fees_bps"] = 100.0
    edge_frame["expected_friction_bps"] = 100.0
    edge_frame["expected_spread_bps"] = 0.0
    edge_frame = _edge_and_cost_columns(edge_frame, spread_floor_bps=0.0, execution_gap_bps=0.0)
    edge_table = _timestamp_edge_table(edge_frame, global_floor=0.90)
    work = edge_frame.merge(edge_table, on="timestamp", how="left")
    entry_from_path = paths[3][:, 0].astype(np.float64, copy=False)
    entry = pd.to_numeric(work.get("entry_price"), errors="coerce").to_numpy(dtype=np.float64)
    entry = np.where(np.isfinite(entry) & (entry > 0.0), entry, entry_from_path)
    entry = np.where(np.isfinite(entry) & (entry > 0.0), entry, 1.0)
    hold_minutes = (
        pd.to_numeric(work.get("holding_bars"), errors="coerce").fillna(1.0).clip(lower=1.0).to_numpy(dtype=np.float64)
        * 15.0
    )
    cand_edges: dict[str, np.ndarray] = {}
    for q in (65, 75, 85):
        cand_edges[f"candidate_edge_p{q}_bps"] = (
            pd.to_numeric(work.get(f"candidate_edge_p{q}_bps"), errors="coerce")
            .fillna(pd.to_numeric(work.get("_edge_gross_bps"), errors="coerce"))
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
    return {
        "side": _side_codes(work),
        "entry_timestamp_ns": pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
        .astype("int64")
        .to_numpy(dtype=np.int64),
        "entry_price": entry.astype(np.float32),
        "hold_minutes": np.clip(np.ceil(hold_minutes), 1, MAX_PATH_MINUTES).astype(np.int16),
        "cost_bps": np.full(len(work), 100.0, dtype=np.float32),
        "edge_net_bps": pd.to_numeric(work["_edge_net_bps"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        "candidate_edge_p65_bps": cand_edges["candidate_edge_p65_bps"],
        "candidate_edge_p75_bps": cand_edges["candidate_edge_p75_bps"],
        "candidate_edge_p85_bps": cand_edges["candidate_edge_p85_bps"],
        "candidate_friction_bps": (
            pd.to_numeric(work.get("candidate_fees_and_slippage_p75_bps"), errors="coerce")
            .fillna(100.0)
            .to_numpy(dtype=np.float32)
        ),
    }


def _replay_with_advanced_ev_redeployment(
    candidates: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    market_mode: str,
    ohlcv_root: Path,
    cfg: ExitTighteningConfig,
):
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame()
    work = normalise_candidate_table(candidates.reset_index(drop=True))
    opens, highs, lows, closes, coverage = _build_paths(
        work,
        ohlcv_root=Path(ohlcv_root),
        max_path_minutes=MAX_PATH_MINUTES,
    )
    paths = (opens, highs, lows, closes)
    arrays = _advanced_exit_arrays(work, paths, params=params)
    decisions, equity, metrics = replay_candidates(
        work,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
        accepted_position_callback=_make_exit_adjust_callback(cfg, arrays, paths, params=params),
    )
    accepted = _accepted_with_candidates(work, decisions)
    accepted["advanced_path_coverage_bars"] = np.nan
    if not accepted.empty and "candidate_index" in decisions.columns:
        idx = pd.to_numeric(decisions.loc[decisions["accepted"].astype(bool), "candidate_index"], errors="coerce")
        idx = idx.dropna().astype(int).to_numpy()
        if idx.size:
            accepted["advanced_path_coverage_bars"] = coverage[idx[: len(accepted)]]
    return decisions, equity, metrics, accepted


def _timestamp_pressure_features(candidates: pd.DataFrame, equity: pd.DataFrame, accepted: pd.DataFrame) -> pd.DataFrame:
    features = _timestamp_features(candidates, max_cols=48)
    fill_values = _timestamp_feature_fill_values(features)
    features = _timestamp_features(candidates, max_cols=48, fill_values=fill_values)
    features = _add_portfolio_state_features(features, equity)
    features = _add_open_position_concentration_features(features, accepted)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _summary_row(arm: str, metrics: Mapping[str, Any], accepted: pd.DataFrame) -> dict[str, Any]:
    days = 0.0
    if not accepted.empty:
        ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
        days = max((ts.max() - ts.min()).total_seconds() / 86400.0, 1.0)
    return {
        "arm": arm,
        "trade_count": int(metrics.get("trade_count", len(accepted)) or len(accepted)),
        "trades_per_day": float(len(accepted) / days) if days else 0.0,
        "net_pnl": _safe_float(metrics.get("net_pnl"), 0.0),
        "gross_pnl": _safe_float(metrics.get("gross_pnl"), 0.0),
        "cost_pnl": _safe_float(metrics.get("cost_pnl"), _safe_float(metrics.get("gross_pnl"), 0.0) - _safe_float(metrics.get("net_pnl"), 0.0)),
        "notional_turnover": _safe_float(metrics.get("notional_turnover"), 0.0),
        "mean_net_return_per_trade": _safe_float(accepted["net_return"].mean(), 0.0) if not accepted.empty else 0.0,
        "notional_weighted_net_return": (
            _safe_float(accepted["net_pnl"].sum() / max(accepted["position_size"].sum(), 1e-12), 0.0)
            if not accepted.empty
            else 0.0
        ),
        "full_sl_rate": _safe_float(accepted["full_sl"].mean(), 0.0) if not accepted.empty else 0.0,
        "timeout_rate": _safe_float(accepted["timeout"].mean(), 0.0) if not accepted.empty else 0.0,
        "ev_redeployment_exit_rate": (
            _safe_float(accepted["ev_redeployment_early_exit"].mean(), 0.0)
            if not accepted.empty and "ev_redeployment_early_exit" in accepted.columns
            else 0.0
        ),
        "advanced_ev_exit_rate": (
            _safe_float(accepted["advanced_ev_redeployment_early_exit"].mean(), 0.0)
            if not accepted.empty and "advanced_ev_redeployment_early_exit" in accepted.columns
            else 0.0
        ),
        "avg_open_positions": _safe_float(metrics.get("avg_open_positions"), 0.0),
        "max_drawdown": _safe_float(metrics.get("max_drawdown"), 0.0),
        "worst_24h_net_pnl": _safe_float(metrics.get("worst_24h_net_pnl"), 0.0),
    }


def _group_metrics(accepted: pd.DataFrame, group_cols: list[str], arm: str) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame(columns=["arm", *group_cols])
    rows: list[dict[str, Any]] = []
    for key, group in accepted.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        position = pd.to_numeric(group["position_size"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "arm": arm,
                **{col: value for col, value in zip(group_cols, key)},
                "trade_count": int(len(group)),
                "net_pnl": float(group["net_pnl"].sum()),
                "gross_pnl": float(group["gross_pnl"].sum()),
                "notional_weighted_net_return": float(group["net_pnl"].sum() / max(position.sum(), 1e-12)),
                "mean_net_return_per_trade": float(pd.to_numeric(group["net_return"], errors="coerce").mean()),
                "full_sl_rate": float(group["full_sl"].mean()),
                "timeout_rate": float(group["timeout"].mean()),
                "ev_redeployment_exit_rate": float(
                    group.get("ev_redeployment_early_exit", pd.Series(False, index=group.index)).mean()
                ),
                "advanced_ev_exit_rate": float(
                    group.get("advanced_ev_redeployment_early_exit", pd.Series(False, index=group.index)).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _delta_table(summary: pd.DataFrame, baseline_arm: str = "baseline") -> pd.DataFrame:
    base = summary.loc[summary["arm"].eq(baseline_arm)]
    if base.empty:
        return summary.copy()
    base_row = base.iloc[0]
    out = summary.copy()
    for col in [
        "trade_count",
        "net_pnl",
        "notional_weighted_net_return",
        "mean_net_return_per_trade",
        "full_sl_rate",
        "timeout_rate",
        "max_drawdown",
    ]:
        if col in out.columns:
            out[f"delta_{col}_vs_baseline"] = pd.to_numeric(out[col], errors="coerce") - float(base_row[col])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--candidate-cache", type=Path, default=None)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--market-mode", choices=["spot", "perps"], default="perps")
    parser.add_argument("--ohlcv-root", type=Path, default=DEFAULT_OHLCV_ROOT)
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--start", default="2026-05-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--train-end", default="2026-06-01")
    parser.add_argument("--optuna-trials", type=int, default=64)
    parser.add_argument("--optuna-validation-hours", type=int, default=168)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--constrained", action="store_true", default=True)
    parser.add_argument("--enable-ev-redeployment", action="store_true")
    parser.add_argument("--redeploy-pressure-start", type=float, default=0.65)
    parser.add_argument("--redeploy-remaining-capital-frac", type=float, default=0.25)
    parser.add_argument("--redeploy-min-future-rank", type=float, default=0.955)
    parser.add_argument("--redeploy-rank-margin", type=float, default=0.025)
    parser.add_argument("--redeploy-lookahead-hours", type=float, default=12.0)
    parser.add_argument("--redeploy-min-hold-bars", type=int, default=1)
    parser.add_argument("--redeploy-churn-penalty-bps", type=float, default=100.0)
    parser.add_argument("--enable-advanced-ev-redeployment", action="store_true")
    parser.add_argument("--advanced-exit-profile", choices=["balanced", "aggressive"], default="balanced")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    out_dir = args.output_dir or (artifact_dir / "capital_pressure_allocator_priority_rank50")
    out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    candidate_cache = args.candidate_cache or (out_dir / "priority_rank50_replay_candidates.parquet")
    if candidate_cache.exists():
        candidates = pd.read_parquet(candidate_cache)
        candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    elif bool(args.enable_advanced_ev_redeployment):
        candidates, _construction_paths = _build_priority_rank50_candidates_with_context(
            artifact_dir=artifact_dir,
            min_ts=start,
            max_ts=end,
            data_root=str(args.data_root),
            market_mode=str(args.market_mode),
            path_len=int(args.path_len),
        )
        candidates.to_parquet(candidate_cache, index=False)
    else:
        candidates = _build_priority_rank50_candidates(
            artifact_dir=artifact_dir,
            min_ts=start,
            max_ts=end,
            data_root=str(args.data_root),
            market_mode=str(args.market_mode),
            path_len=int(args.path_len),
        )
        candidates.to_parquet(candidate_cache, index=False)
    if candidate_cache.resolve() != (out_dir / "priority_rank50_replay_candidates.parquet").resolve():
        candidates.to_parquet(out_dir / "priority_rank50_replay_candidates.parquet", index=False)

    params = load_portfolio_policy_params(artifact_dir / "policy_params" / "optimized_portfolio_policy_config.json")
    ev_curve = fit_hierarchical_ev_curves(candidates)
    base_decisions, base_equity, base_metrics, base_accepted = _replay_baseline(
        candidates, params, ev_curve, market_mode=str(args.market_mode)
    )
    features = _timestamp_pressure_features(candidates, base_equity, base_accepted)

    ts = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    train_candidates = candidates.loc[ts < train_end].copy()
    train_features = features.loc[pd.to_datetime(features["timestamp"], utc=True, errors="coerce").lt(train_end)].copy()
    pressure_params, optuna_trials = _optimise_reallocation_params(
        train_candidates,
        train_features,
        params,
        ev_curve,
        market_mode=str(args.market_mode),
        validation_hours=int(args.optuna_validation_hours),
        n_trials=int(args.optuna_trials),
        seed=int(args.seed),
        constrained=bool(args.constrained),
    )

    pressure_candidates = _apply_pressure_reallocation(
        candidates,
        features,
        params_dict=pressure_params,
        max_entries_per_bar=int(params.max_new_entries_per_bar),
    )
    pressure_decisions, pressure_equity, pressure_metrics = replay_candidates(
        pressure_candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=str(args.market_mode),
    )
    pressure_accepted = _accepted_with_candidates(pressure_candidates, pressure_decisions)
    redeploy_params = {
        "pressure_start": float(args.redeploy_pressure_start),
        "remaining_capital_frac": float(args.redeploy_remaining_capital_frac),
        "min_future_rank": float(args.redeploy_min_future_rank),
        "rank_margin": float(args.redeploy_rank_margin),
        "lookahead_hours": float(args.redeploy_lookahead_hours),
        "min_hold_bars": int(args.redeploy_min_hold_bars),
        "churn_penalty_bps": float(args.redeploy_churn_penalty_bps),
    }
    if bool(args.enable_ev_redeployment):
        redeploy_decisions, redeploy_equity, redeploy_metrics, redeploy_accepted = _replay_with_ev_redeployment(
            pressure_candidates,
            params,
            ev_curve,
            market_mode=str(args.market_mode),
            redeploy_params=redeploy_params,
        )
    else:
        redeploy_decisions = pd.DataFrame()
        redeploy_equity = pd.DataFrame()
        redeploy_metrics = {}
        redeploy_accepted = pd.DataFrame()

    advanced_cfg = _advanced_exit_config(str(args.advanced_exit_profile))
    if bool(args.enable_advanced_ev_redeployment):
        (
            advanced_decisions,
            advanced_equity,
            advanced_metrics,
            advanced_accepted,
        ) = _replay_with_advanced_ev_redeployment(
            pressure_candidates,
            params,
            ev_curve,
            market_mode=str(args.market_mode),
            ohlcv_root=Path(args.ohlcv_root),
            cfg=advanced_cfg,
        )
    else:
        advanced_decisions = pd.DataFrame()
        advanced_equity = pd.DataFrame()
        advanced_metrics = {}
        advanced_accepted = pd.DataFrame()

    oos_mask = ts >= train_end
    oos_candidates = candidates.loc[oos_mask].copy()
    oos_features = features.loc[pd.to_datetime(features["timestamp"], utc=True, errors="coerce").ge(train_end)].copy()
    oos_base_decisions, oos_base_equity, oos_base_metrics, oos_base_accepted = _replay_baseline(
        oos_candidates, params, ev_curve, market_mode=str(args.market_mode)
    )
    oos_pressure_candidates = _apply_pressure_reallocation(
        oos_candidates,
        oos_features,
        params_dict=pressure_params,
        max_entries_per_bar=int(params.max_new_entries_per_bar),
    )
    oos_pressure_decisions, _oos_pressure_equity, oos_pressure_metrics = replay_candidates(
        oos_pressure_candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=str(args.market_mode),
    )
    oos_pressure_accepted = _accepted_with_candidates(oos_pressure_candidates, oos_pressure_decisions)
    if bool(args.enable_ev_redeployment):
        (
            oos_redeploy_decisions,
            _oos_redeploy_equity,
            oos_redeploy_metrics,
            oos_redeploy_accepted,
        ) = _replay_with_ev_redeployment(
            oos_pressure_candidates,
            params,
            ev_curve,
            market_mode=str(args.market_mode),
            redeploy_params=redeploy_params,
        )
    else:
        oos_redeploy_decisions = pd.DataFrame()
        oos_redeploy_metrics = {}
        oos_redeploy_accepted = pd.DataFrame()
    if bool(args.enable_advanced_ev_redeployment):
        (
            oos_advanced_decisions,
            _oos_advanced_equity,
            oos_advanced_metrics,
            oos_advanced_accepted,
        ) = _replay_with_advanced_ev_redeployment(
            oos_pressure_candidates,
            params,
            ev_curve,
            market_mode=str(args.market_mode),
            ohlcv_root=Path(args.ohlcv_root),
            cfg=advanced_cfg,
        )
    else:
        oos_advanced_decisions = pd.DataFrame()
        oos_advanced_metrics = {}
        oos_advanced_accepted = pd.DataFrame()

    summary_rows = [
        _summary_row("baseline", base_metrics, base_accepted),
        _summary_row("capital_pressure", pressure_metrics, pressure_accepted),
        _summary_row("oos_june_baseline", oos_base_metrics, oos_base_accepted),
        _summary_row("oos_june_capital_pressure", oos_pressure_metrics, oos_pressure_accepted),
    ]
    if bool(args.enable_ev_redeployment):
        summary_rows.extend(
            [
                _summary_row("capital_pressure_ev_redeployment", redeploy_metrics, redeploy_accepted),
                _summary_row("oos_june_capital_pressure_ev_redeployment", oos_redeploy_metrics, oos_redeploy_accepted),
            ]
        )
    if bool(args.enable_advanced_ev_redeployment):
        summary_rows.extend(
            [
                _summary_row(
                    "capital_pressure_advanced_ev_redeployment",
                    advanced_metrics,
                    advanced_accepted,
                ),
                _summary_row(
                    "oos_june_capital_pressure_advanced_ev_redeployment",
                    oos_advanced_metrics,
                    oos_advanced_accepted,
                ),
            ]
        )
    summary = pd.DataFrame(summary_rows)
    summary = _delta_table(summary)
    summary.to_csv(out_dir / "capital_pressure_summary.csv", index=False)
    if not optuna_trials.empty:
        optuna_trials.to_csv(out_dir / "capital_pressure_optuna_trials.csv", index=False)

    pressure_candidates.to_parquet(out_dir / "capital_pressure_replay_candidates.parquet", index=False)
    base_decisions.to_parquet(out_dir / "baseline_decisions.parquet", index=False)
    pressure_decisions.to_parquet(out_dir / "capital_pressure_decisions.parquet", index=False)
    base_accepted.to_parquet(out_dir / "baseline_accepted.parquet", index=False)
    pressure_accepted.to_parquet(out_dir / "capital_pressure_accepted.parquet", index=False)
    oos_base_accepted.to_parquet(out_dir / "oos_june_baseline_accepted.parquet", index=False)
    oos_pressure_accepted.to_parquet(out_dir / "oos_june_capital_pressure_accepted.parquet", index=False)
    if bool(args.enable_ev_redeployment):
        redeploy_decisions.to_parquet(out_dir / "capital_pressure_ev_redeployment_decisions.parquet", index=False)
        redeploy_accepted.to_parquet(out_dir / "capital_pressure_ev_redeployment_accepted.parquet", index=False)
        oos_redeploy_decisions.to_parquet(out_dir / "oos_june_capital_pressure_ev_redeployment_decisions.parquet", index=False)
        oos_redeploy_accepted.to_parquet(out_dir / "oos_june_capital_pressure_ev_redeployment_accepted.parquet", index=False)
    if bool(args.enable_advanced_ev_redeployment):
        advanced_decisions.to_parquet(out_dir / "capital_pressure_advanced_ev_redeployment_decisions.parquet", index=False)
        advanced_accepted.to_parquet(out_dir / "capital_pressure_advanced_ev_redeployment_accepted.parquet", index=False)
        oos_advanced_decisions.to_parquet(
            out_dir / "oos_june_capital_pressure_advanced_ev_redeployment_decisions.parquet",
            index=False,
        )
        oos_advanced_accepted.to_parquet(
            out_dir / "oos_june_capital_pressure_advanced_ev_redeployment_accepted.parquet",
            index=False,
        )

    detail_frames: list[pd.DataFrame] = []
    for arm, frame in (
        ("baseline", base_accepted),
        ("capital_pressure", pressure_accepted),
        ("oos_june_baseline", oos_base_accepted),
        ("oos_june_capital_pressure", oos_pressure_accepted),
    ):
        for group_cols, name in (
            (["month"], "by_month"),
            (["week_start"], "by_week"),
            (["side_name"], "by_side"),
            (["policy_archetype"], "by_archetype"),
            (["month", "side_name", "policy_archetype"], "by_month_side_archetype"),
            (["week_start", "side_name", "policy_archetype"], "by_week_side_archetype"),
        ):
            metrics = _group_metrics(frame, group_cols, arm)
            metrics.insert(1, "breakdown", name)
            detail_frames.append(metrics)
    if bool(args.enable_ev_redeployment):
        for arm, frame in (
            ("capital_pressure_ev_redeployment", redeploy_accepted),
            ("oos_june_capital_pressure_ev_redeployment", oos_redeploy_accepted),
        ):
            for group_cols, name in (
                (["month"], "by_month"),
                (["week_start"], "by_week"),
                (["side_name"], "by_side"),
                (["policy_archetype"], "by_archetype"),
                (["month", "side_name", "policy_archetype"], "by_month_side_archetype"),
                (["week_start", "side_name", "policy_archetype"], "by_week_side_archetype"),
            ):
                metrics = _group_metrics(frame, group_cols, arm)
                metrics.insert(1, "breakdown", name)
                detail_frames.append(metrics)
    if bool(args.enable_advanced_ev_redeployment):
        for arm, frame in (
            ("capital_pressure_advanced_ev_redeployment", advanced_accepted),
            ("oos_june_capital_pressure_advanced_ev_redeployment", oos_advanced_accepted),
        ):
            for group_cols, name in (
                (["month"], "by_month"),
                (["week_start"], "by_week"),
                (["side_name"], "by_side"),
                (["policy_archetype"], "by_archetype"),
                (["month", "side_name", "policy_archetype"], "by_month_side_archetype"),
                (["week_start", "side_name", "policy_archetype"], "by_week_side_archetype"),
            ):
                metrics = _group_metrics(frame, group_cols, arm)
                metrics.insert(1, "breakdown", name)
                detail_frames.append(metrics)
    pd.concat(detail_frames, ignore_index=True).to_csv(out_dir / "capital_pressure_breakdowns.csv", index=False)

    (out_dir / "capital_pressure_params.json").write_text(
        json.dumps(_json_safe(pressure_params), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if bool(args.enable_ev_redeployment):
        (out_dir / "capital_pressure_ev_redeployment_params.json").write_text(
            json.dumps(_json_safe(redeploy_params), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if bool(args.enable_advanced_ev_redeployment):
        (out_dir / "capital_pressure_advanced_ev_redeployment_params.json").write_text(
            json.dumps(_json_safe(asdict(advanced_cfg)), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "candidate_source": str(_load_json(artifact_dir / "manifest.json").get("candidates")),
        "policy_config": str(artifact_dir / "policy_params" / "optimized_portfolio_policy_config.json"),
        "policy_config_params": asdict(params),
        "start": str(start),
        "end": str(end),
        "train_end": str(train_end),
        "top_slice": "top10",
        "hit_surprise_mode": "hit_surprise_priority_rank_50",
        "optuna_trials": int(args.optuna_trials),
        "optuna_validation_hours": int(args.optuna_validation_hours),
        "constrained_pressure_space": bool(args.constrained),
        "ev_redeployment_enabled": bool(args.enable_ev_redeployment),
        "ev_redeployment_params": redeploy_params if bool(args.enable_ev_redeployment) else None,
        "advanced_ev_redeployment_enabled": bool(args.enable_advanced_ev_redeployment),
        "advanced_exit_profile": str(args.advanced_exit_profile),
        "advanced_ev_redeployment_params": asdict(advanced_cfg) if bool(args.enable_advanced_ev_redeployment) else None,
        "outputs": {
            "summary": str(out_dir / "capital_pressure_summary.csv"),
            "breakdowns": str(out_dir / "capital_pressure_breakdowns.csv"),
            "params": str(out_dir / "capital_pressure_params.json"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
