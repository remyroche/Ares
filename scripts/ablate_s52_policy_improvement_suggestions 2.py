#!/usr/bin/env python3
"""Replay-only ablations for S52 portfolio-policy improvement ideas.

The script consumes the materialized hit-surprise selected rows from the current
policy artifact and runs one arm at a time through the same global portfolio
replay. It deliberately does not retrain base/meta or alter exit geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    replay_candidates,
)
from scripts.ablate_s52_archetype_hit_surprise_thresholds import (  # noqa: E402
    _portfolio_candidate_table,
)


DEFAULT_REPORT_DIR = ROOT / (
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_"
    "ae3000_nocrossfit_k34567_payload300k_20260708_hr_threshold_modulation_top15_top5_"
    "protected_regime_rank_retained50"
)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _params(
    *,
    allocation_alpha: float = 0.30,
    occupancy_alpha: float = 0.30,
    max_concurrent_positions: int = 6,
) -> PortfolioPolicyParams:
    return PortfolioPolicyParams(
        max_concurrent_positions=int(max_concurrent_positions),
        max_concurrent_per_side=None,
        max_concurrent_per_strategy=None,
        max_new_entries_per_bar=2,
        max_total_wallet_allocation_pct=0.75,
        global_threshold_floor=0.0,
        occupancy_threshold_alpha=float(occupancy_alpha),
        occupancy_threshold_power=1.5,
        allocation_threshold_alpha=float(allocation_alpha),
        allocation_threshold_power=1.0,
        rank_size_power=1.5,
        rank_multiplier_min=0.5,
        rank_multiplier_max=1.5,
        min_position_size=1.0,
    )


def _load_best_selected(report_dir: Path) -> pd.DataFrame:
    path = report_dir / "hit_surprise_preportfolio_selected.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = pd.read_parquet(path)
    rows = rows[
        rows["top_slice"].astype(str).eq("top10")
        & rows["mode"].astype(str).eq("hit_surprise_priority_rank_50")
        & pd.to_numeric(rows["half_life_days"], errors="coerce").eq(3.0)
        & pd.to_numeric(rows["alpha"], errors="coerce").eq(0.25)
        & pd.to_numeric(rows["max_adjust"], errors="coerce").eq(0.05)
    ].copy()
    if rows.empty:
        raise RuntimeError("No current best top10 hit-surprise selected rows found.")
    return rows


def _candidate_table(rows: pd.DataFrame) -> pd.DataFrame:
    candidates = _portfolio_candidate_table(rows)
    if candidates.empty:
        return candidates
    candidates["timestamp"] = pd.to_datetime(
        candidates["timestamp"], utc=True, errors="coerce"
    )
    return candidates.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _arm_current_best(rows: pd.DataFrame) -> tuple[pd.DataFrame, PortfolioPolicyParams, str]:
    return _candidate_table(rows), _params(), "current best policy"


def _arm_underinvested_quality_relax(rows: pd.DataFrame) -> tuple[pd.DataFrame, PortfolioPolicyParams, str]:
    """Relax positive-quality archetypes toward top15, offset by stronger capital pressure."""
    work = rows.copy()
    q_unit = pd.to_numeric(work.get("hr_quality_unit"), errors="coerce").fillna(0.0)
    positive = q_unit.clip(lower=0.0, upper=1.0)
    base = pd.to_numeric(work.get("base_rank_threshold"), errors="coerce").fillna(0.90)
    # Positive recent quality can widen top10 toward top15; negative quality keeps top10.
    relaxed = (base - 0.05 * positive).clip(lower=0.85, upper=0.95)
    work["applied_rank_threshold"] = relaxed
    candidates = _candidate_table(work)
    return (
        candidates,
        _params(allocation_alpha=0.45, occupancy_alpha=0.40),
        "positive HR cells can relax toward top15; higher allocation/occupancy pressure tightens when invested",
    )


def _arm_ev_after_spread_priority(rows: pd.DataFrame) -> tuple[pd.DataFrame, PortfolioPolicyParams, str]:
    """Penalize portfolio priority by row-level spread/friction cost instead of fixed friction."""
    candidates = _candidate_table(rows)
    if candidates.empty:
        return candidates, _params(), "empty"
    spread = pd.to_numeric(candidates.get("spread_cost_bps"), errors="coerce")
    expected = pd.to_numeric(candidates.get("expected_spread_bps"), errors="coerce")
    friction = spread.where(spread.notna(), expected).fillna(0.0).clip(lower=0.0)
    # Keep the 1% round-trip baseline visible, but add row-level spread cost to
    # auction priority so high-friction candidates need more rank surplus.
    candidates["expected_friction_bps"] = 100.0 + friction.astype(float)
    return candidates, _params(), "auction priority uses 1% baseline plus row-level spread-cost bps"


def _arm_current_best_strong_capital_pressure(rows: pd.DataFrame) -> tuple[pd.DataFrame, PortfolioPolicyParams, str]:
    return (
        _candidate_table(rows),
        _params(allocation_alpha=0.55, occupancy_alpha=0.55),
        "current best rank/HR logic with stronger wallet-allocation and occupancy threshold pressure",
    )


def _arm_current_best_light_capital_pressure(rows: pd.DataFrame) -> tuple[pd.DataFrame, PortfolioPolicyParams, str]:
    return (
        _candidate_table(rows),
        _params(allocation_alpha=0.15, occupancy_alpha=0.20),
        "current best rank/HR logic with lighter wallet-allocation and occupancy threshold pressure",
    )


def _attach_recent_path_surprise(rows: pd.DataFrame, *, half_life_days: float = 7.0) -> pd.DataFrame:
    work = rows.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    if "policy_archetype" not in work.columns:
        work["policy_archetype"] = "missing"
    work["recent_ev_mean_prior"] = 0.0
    work["recent_full_sl_rate_prior"] = 0.0
    work["recent_surprise_n_eff_prior"] = 0.0
    for _, idx in work.sort_values("timestamp").groupby("policy_archetype", sort=False).groups.items():
        idx_list = list(idx)
        ev_mean = 0.0
        sl_mean = 0.0
        n_eff = 0.0
        last_ts: pd.Timestamp | None = None
        for row_idx in idx_list:
            ts = work.at[row_idx, "timestamp"]
            if last_ts is not None and pd.notna(ts) and pd.notna(last_ts):
                dt_days = max(0.0, float((ts - last_ts).total_seconds()) / 86400.0)
                decay = 0.5 ** (dt_days / max(float(half_life_days), 1e-6))
                n_eff *= decay
            work.at[row_idx, "recent_ev_mean_prior"] = ev_mean if n_eff >= 8.0 else 0.0
            work.at[row_idx, "recent_full_sl_rate_prior"] = sl_mean if n_eff >= 8.0 else 0.0
            work.at[row_idx, "recent_surprise_n_eff_prior"] = n_eff
            ret = _safe_float(work.at[row_idx, "ret_net_notional"], np.nan)
            full_sl = 1.0 if str(work.at[row_idx, "exit_reason"]) == "full_sl" else 0.0
            if np.isfinite(ret):
                new_n = n_eff + 1.0
                ev_mean = (ev_mean * n_eff + float(ret)) / new_n
                sl_mean = (sl_mean * n_eff + float(full_sl)) / new_n
                n_eff = new_n
                last_ts = ts
    return work


def _arm_recent_ev_surprise_priority(rows: pd.DataFrame) -> tuple[pd.DataFrame, PortfolioPolicyParams, str]:
    work = _attach_recent_path_surprise(rows, half_life_days=7.0)
    ev = pd.to_numeric(work["recent_ev_mean_prior"], errors="coerce").fillna(0.0)
    support = (pd.to_numeric(work["recent_surprise_n_eff_prior"], errors="coerce").fillna(0.0) / 30.0).clip(0.0, 1.0)
    q = (ev / 0.01).clip(-1.0, 1.0) * 0.05 * support
    q_unit = (q / 0.05).clip(-1.0, 1.0)
    base_rank_adj = pd.to_numeric(work.get("portfolio_rank_adjustment"), errors="coerce").fillna(0.0)
    base_priority = pd.to_numeric(work.get("portfolio_priority_multiplier"), errors="coerce").fillna(1.0)
    work["portfolio_rank_adjustment"] = (base_rank_adj + 0.5 * q).clip(-0.05, 0.05)
    work["portfolio_priority_multiplier"] = (base_priority * (1.0 + 0.35 * q_unit)).clip(0.50, 1.50)
    return _candidate_table(work), _params(), "online 7d archetype EV surprise adjusts rank and auction priority"


def _arm_recent_ev_stop_surprise(rows: pd.DataFrame) -> tuple[pd.DataFrame, PortfolioPolicyParams, str]:
    work = _attach_recent_path_surprise(rows, half_life_days=7.0)
    ev = pd.to_numeric(work["recent_ev_mean_prior"], errors="coerce").fillna(0.0)
    full_sl = pd.to_numeric(work["recent_full_sl_rate_prior"], errors="coerce").fillna(0.0)
    support = (pd.to_numeric(work["recent_surprise_n_eff_prior"], errors="coerce").fillna(0.0) / 30.0).clip(0.0, 1.0)
    q_ev = (ev / 0.01).clip(-1.0, 1.0) * 0.04 * support
    sl_penalty = ((full_sl - 0.04) / 0.08).clip(0.0, 1.0) * support
    base_rank_adj = pd.to_numeric(work.get("portfolio_rank_adjustment"), errors="coerce").fillna(0.0)
    base_priority = pd.to_numeric(work.get("portfolio_priority_multiplier"), errors="coerce").fillna(1.0)
    if "portfolio_size_multiplier" in work.columns:
        base_size = pd.to_numeric(work["portfolio_size_multiplier"], errors="coerce").fillna(1.0)
    else:
        base_size = pd.Series(1.0, index=work.index)
    work["portfolio_rank_adjustment"] = (base_rank_adj + 0.5 * q_ev - 0.025 * sl_penalty).clip(-0.05, 0.05)
    work["portfolio_priority_multiplier"] = (base_priority * (1.0 + 0.30 * (q_ev / 0.04).clip(-1.0, 1.0) - 0.35 * sl_penalty)).clip(0.40, 1.50)
    work["portfolio_size_multiplier"] = (base_size * (1.0 - 0.50 * sl_penalty)).clip(0.35, 1.50)
    return _candidate_table(work), _params(), "online 7d EV surprise plus full-stop surprise adjusts rank, priority, and size"


def _arm_recent_ev_quality_balanced(rows: pd.DataFrame) -> tuple[pd.DataFrame, PortfolioPolicyParams, str]:
    work = _attach_recent_path_surprise(rows, half_life_days=7.0)
    ev = pd.to_numeric(work["recent_ev_mean_prior"], errors="coerce").fillna(0.0)
    full_sl = pd.to_numeric(work["recent_full_sl_rate_prior"], errors="coerce").fillna(0.0)
    support = (pd.to_numeric(work["recent_surprise_n_eff_prior"], errors="coerce").fillna(0.0) / 40.0).clip(0.0, 1.0)
    q_ev = (ev / 0.0125).clip(-1.0, 1.0) * 0.03 * support
    sl_penalty = ((full_sl - 0.035) / 0.10).clip(0.0, 1.0) * support
    base_rank_adj = pd.to_numeric(work.get("portfolio_rank_adjustment"), errors="coerce").fillna(0.0)
    base_priority = pd.to_numeric(work.get("portfolio_priority_multiplier"), errors="coerce").fillna(1.0)
    if "portfolio_size_multiplier" in work.columns:
        base_size = pd.to_numeric(work["portfolio_size_multiplier"], errors="coerce").fillna(1.0)
    else:
        base_size = pd.Series(1.0, index=work.index)
    work["portfolio_rank_adjustment"] = (base_rank_adj + 0.35 * q_ev - 0.018 * sl_penalty).clip(-0.04, 0.04)
    work["portfolio_priority_multiplier"] = (base_priority * (1.0 + 0.18 * (q_ev / 0.03).clip(-1.0, 1.0) - 0.30 * sl_penalty)).clip(0.50, 1.35)
    work["portfolio_size_multiplier"] = (base_size * (1.0 - 0.35 * sl_penalty)).clip(0.50, 1.25)
    return (
        _candidate_table(work),
        _params(allocation_alpha=0.45, occupancy_alpha=0.45),
        "balanced online 7d EV signal with full-stop penalty and stronger portfolio pressure",
    )


def _recent_ev_quality_template(
    rows: pd.DataFrame,
    *,
    support_divisor: float,
    ev_scale: float,
    ev_rank_weight: float,
    priority_boost: float,
    stop_threshold: float,
    stop_scale: float,
    stop_rank_penalty: float,
    stop_priority_penalty: float,
    stop_size_penalty: float,
    allocation_alpha: float,
    occupancy_alpha: float,
    label: str,
) -> tuple[pd.DataFrame, PortfolioPolicyParams, str]:
    work = _attach_recent_path_surprise(rows, half_life_days=7.0)
    ev = pd.to_numeric(work["recent_ev_mean_prior"], errors="coerce").fillna(0.0)
    full_sl = pd.to_numeric(work["recent_full_sl_rate_prior"], errors="coerce").fillna(0.0)
    support = (pd.to_numeric(work["recent_surprise_n_eff_prior"], errors="coerce").fillna(0.0) / float(support_divisor)).clip(0.0, 1.0)
    q_ev = (ev / float(ev_scale)).clip(-1.0, 1.0) * 0.03 * support
    sl_penalty = ((full_sl - float(stop_threshold)) / float(stop_scale)).clip(0.0, 1.0) * support
    base_rank_adj = pd.to_numeric(work.get("portfolio_rank_adjustment"), errors="coerce").fillna(0.0)
    base_priority = pd.to_numeric(work.get("portfolio_priority_multiplier"), errors="coerce").fillna(1.0)
    if "portfolio_size_multiplier" in work.columns:
        base_size = pd.to_numeric(work["portfolio_size_multiplier"], errors="coerce").fillna(1.0)
    else:
        base_size = pd.Series(1.0, index=work.index)
    work["portfolio_rank_adjustment"] = (base_rank_adj + float(ev_rank_weight) * q_ev - float(stop_rank_penalty) * sl_penalty).clip(-0.045, 0.045)
    work["portfolio_priority_multiplier"] = (base_priority * (1.0 + float(priority_boost) * (q_ev / 0.03).clip(-1.0, 1.0) - float(stop_priority_penalty) * sl_penalty)).clip(0.45, 1.45)
    work["portfolio_size_multiplier"] = (base_size * (1.0 - float(stop_size_penalty) * sl_penalty)).clip(0.45, 1.35)
    return (
        _candidate_table(work),
        _params(allocation_alpha=allocation_alpha, occupancy_alpha=occupancy_alpha),
        label,
    )


def _arm_recent_ev_quality_mid(rows: pd.DataFrame) -> tuple[pd.DataFrame, PortfolioPolicyParams, str]:
    return _recent_ev_quality_template(
        rows,
        support_divisor=35.0,
        ev_scale=0.012,
        ev_rank_weight=0.42,
        priority_boost=0.24,
        stop_threshold=0.04,
        stop_scale=0.11,
        stop_rank_penalty=0.014,
        stop_priority_penalty=0.24,
        stop_size_penalty=0.25,
        allocation_alpha=0.40,
        occupancy_alpha=0.40,
        label="midpoint volume/quality balance: EV boost with moderate full-stop penalty and portfolio pressure",
    )


def _arm_recent_ev_quality_volume_tilt(rows: pd.DataFrame) -> tuple[pd.DataFrame, PortfolioPolicyParams, str]:
    return _recent_ev_quality_template(
        rows,
        support_divisor=32.0,
        ev_scale=0.011,
        ev_rank_weight=0.50,
        priority_boost=0.30,
        stop_threshold=0.05,
        stop_scale=0.13,
        stop_rank_penalty=0.008,
        stop_priority_penalty=0.15,
        stop_size_penalty=0.15,
        allocation_alpha=0.35,
        occupancy_alpha=0.35,
        label="volume-tilted balance: stronger recent EV boost with mild path-quality penalty",
    )


def _arm_recent_ev_quality_quality_tilt(rows: pd.DataFrame) -> tuple[pd.DataFrame, PortfolioPolicyParams, str]:
    return _recent_ev_quality_template(
        rows,
        support_divisor=45.0,
        ev_scale=0.014,
        ev_rank_weight=0.30,
        priority_boost=0.16,
        stop_threshold=0.035,
        stop_scale=0.09,
        stop_rank_penalty=0.022,
        stop_priority_penalty=0.36,
        stop_size_penalty=0.45,
        allocation_alpha=0.50,
        occupancy_alpha=0.50,
        label="quality-tilted balance: conservative EV boost with stronger full-stop penalty and portfolio pressure",
    )


def _arm_recent_ev_volume_quality_light(rows: pd.DataFrame) -> tuple[pd.DataFrame, PortfolioPolicyParams, str]:
    work = _attach_recent_path_surprise(rows, half_life_days=7.0)
    ev = pd.to_numeric(work["recent_ev_mean_prior"], errors="coerce").fillna(0.0)
    full_sl = pd.to_numeric(work["recent_full_sl_rate_prior"], errors="coerce").fillna(0.0)
    support = (pd.to_numeric(work["recent_surprise_n_eff_prior"], errors="coerce").fillna(0.0) / 35.0).clip(0.0, 1.0)
    q_ev = (ev / 0.0125).clip(-1.0, 1.0) * 0.025 * support
    sl_penalty = ((full_sl - 0.045) / 0.12).clip(0.0, 1.0) * support
    base_rank_adj = pd.to_numeric(work.get("portfolio_rank_adjustment"), errors="coerce").fillna(0.0)
    base_priority = pd.to_numeric(work.get("portfolio_priority_multiplier"), errors="coerce").fillna(1.0)
    work["portfolio_rank_adjustment"] = (base_rank_adj + 0.45 * q_ev - 0.010 * sl_penalty).clip(-0.045, 0.045)
    work["portfolio_priority_multiplier"] = (base_priority * (1.0 + 0.25 * (q_ev / 0.025).clip(-1.0, 1.0) - 0.18 * sl_penalty)).clip(0.55, 1.45)
    return (
        _candidate_table(work),
        _params(allocation_alpha=0.35, occupancy_alpha=0.35),
        "lighter volume/quality blend: recent EV boosts priority/rank, full-stop surprise mildly penalizes",
    )


def _arm_recent_ev_threshold_relax(rows: pd.DataFrame) -> tuple[pd.DataFrame, PortfolioPolicyParams, str]:
    work = _attach_recent_path_surprise(rows, half_life_days=7.0)
    ev = pd.to_numeric(work["recent_ev_mean_prior"], errors="coerce").fillna(0.0)
    full_sl = pd.to_numeric(work["recent_full_sl_rate_prior"], errors="coerce").fillna(0.0)
    support = (pd.to_numeric(work["recent_surprise_n_eff_prior"], errors="coerce").fillna(0.0) / 35.0).clip(0.0, 1.0)
    base = pd.to_numeric(work.get("base_rank_threshold"), errors="coerce").fillna(0.90)
    positive_ev_relax = (ev / 0.0125).clip(0.0, 1.0) * 0.035 * support
    stop_tighten = ((full_sl - 0.04) / 0.10).clip(0.0, 1.0) * 0.035 * support
    work["applied_rank_threshold"] = (base - positive_ev_relax + stop_tighten).clip(0.865, 0.95)
    return (
        _candidate_table(work),
        _params(allocation_alpha=0.45, occupancy_alpha=0.45),
        "recent EV can relax threshold toward top13.5 while recent full-stop surprise tightens; stronger portfolio pressure controls capital use",
    )


def _arm_archetype_support_shrunk_hr(rows: pd.DataFrame) -> tuple[pd.DataFrame, PortfolioPolicyParams, str]:
    """Shrink existing hit-rate modulation toward neutral when archetype support is thin."""
    work = _attach_recent_path_surprise(rows, half_life_days=7.0)
    support = (pd.to_numeric(work["recent_surprise_n_eff_prior"], errors="coerce").fillna(0.0) / 50.0).clip(0.0, 1.0)
    base_rank_adj = pd.to_numeric(work.get("portfolio_rank_adjustment"), errors="coerce").fillna(0.0)
    base_priority = pd.to_numeric(work.get("portfolio_priority_multiplier"), errors="coerce").fillna(1.0)
    if "portfolio_size_multiplier" in work.columns:
        base_size = pd.to_numeric(work["portfolio_size_multiplier"], errors="coerce").fillna(1.0)
    else:
        base_size = pd.Series(1.0, index=work.index)
    if "base_rank_threshold" in work.columns and "applied_rank_threshold" in work.columns:
        base_threshold = pd.to_numeric(work["base_rank_threshold"], errors="coerce").fillna(0.90)
        applied_threshold = pd.to_numeric(work["applied_rank_threshold"], errors="coerce").fillna(base_threshold)
        work["applied_rank_threshold"] = base_threshold + support * (applied_threshold - base_threshold)
    work["portfolio_rank_adjustment"] = base_rank_adj * support
    work["portfolio_priority_multiplier"] = 1.0 + support * (base_priority - 1.0)
    work["portfolio_size_multiplier"] = 1.0 + support * (base_size - 1.0)
    return (
        _candidate_table(work),
        _params(),
        "shrink existing archetype hit-rate rank, priority, size, and threshold modulation toward neutral when recent archetype support is weak",
    )


ARMS: dict[str, Callable[[pd.DataFrame], tuple[pd.DataFrame, PortfolioPolicyParams, str]]] = {
    "current_best": _arm_current_best,
    "underinvested_quality_relax": _arm_underinvested_quality_relax,
    "ev_after_spread_priority": _arm_ev_after_spread_priority,
    "current_best_strong_capital_pressure": _arm_current_best_strong_capital_pressure,
    "current_best_light_capital_pressure": _arm_current_best_light_capital_pressure,
    "recent_ev_surprise_priority": _arm_recent_ev_surprise_priority,
    "recent_ev_stop_surprise": _arm_recent_ev_stop_surprise,
    "recent_ev_quality_balanced": _arm_recent_ev_quality_balanced,
    "recent_ev_quality_mid": _arm_recent_ev_quality_mid,
    "recent_ev_quality_volume_tilt": _arm_recent_ev_quality_volume_tilt,
    "recent_ev_quality_quality_tilt": _arm_recent_ev_quality_quality_tilt,
    "recent_ev_volume_quality_light": _arm_recent_ev_volume_quality_light,
    "recent_ev_threshold_relax": _arm_recent_ev_threshold_relax,
    "archetype_support_shrunk_hr": _arm_archetype_support_shrunk_hr,
}


def _metrics_from_decisions(
    decisions: pd.DataFrame,
    metrics: dict[str, Any],
    *,
    arm: str,
    description: str,
) -> dict[str, Any]:
    accepted = decisions[decisions["accepted"]].copy()
    return {
        "arm": arm,
        "description": description,
        "trade_count": int(metrics.get("trade_count", 0)),
        "trades_per_day": _safe_float(metrics.get("trades_per_day")),
        "net_pnl": _safe_float(metrics.get("net_pnl")),
        "gross_pnl": _safe_float(metrics.get("gross_pnl")),
        "compounded_return": _safe_float(metrics.get("compounded_return")),
        "max_drawdown": _safe_float(metrics.get("max_drawdown")),
        "worst_week": _safe_float(metrics.get("worst_week")),
        "notional_weighted_net_return": _safe_float(metrics.get("notional_weighted_net_return")),
        "mean_net_return_per_trade": _safe_float(metrics.get("mean_net_return_per_trade")),
        "full_sl_rate": _safe_float(metrics.get("full_sl_rate")),
        "timeout_rate": _safe_float(metrics.get("timeout_rate")),
        "avg_open_positions": _safe_float(metrics.get("avg_open_positions")),
        "position_utilization": _safe_float(metrics.get("position_utilization")),
        "accepted_rows": int(len(accepted)),
    }


def _weekly(decisions: pd.DataFrame, *, arm: str) -> pd.DataFrame:
    accepted = decisions[decisions["accepted"]].copy()
    if accepted.empty:
        return pd.DataFrame()
    ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["week_start"] = (ts.dt.floor("D") - pd.to_timedelta(ts.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
    accepted["month"] = ts.dt.strftime("%Y-%m")
    accepted = accepted[accepted["month"].isin(["2026-05", "2026-06"])].copy()
    if accepted.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (month, week), group in accepted.groupby(["month", "week_start"], sort=True):
        days = max(1.0, group["timestamp"].dt.floor("D").nunique())
        rows.append(
            {
                "arm": arm,
                "month": month,
                "week_start": week,
                "trades": int(len(group)),
                "trades_per_day": float(len(group) / days),
                "avg_net_ev_per_trade": float(pd.to_numeric(group["position_net_return"], errors="coerce").mean()),
                "net_pnl": float(
                    (
                        pd.to_numeric(group["position_net_return"], errors="coerce")
                        * pd.to_numeric(group["position_size"], errors="coerce")
                    ).sum()
                ),
                "full_sl_rate": float(group["position_exit_reason"].astype(str).eq("full_sl").mean()),
                "timeout_rate": float(group["position_exit_reason"].astype(str).eq("timeout").mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--arms",
        default=(
            "current_best,underinvested_quality_relax,ev_after_spread_priority,"
            "current_best_strong_capital_pressure,current_best_light_capital_pressure,"
            "recent_ev_surprise_priority,recent_ev_stop_surprise,recent_ev_quality_balanced,"
            "recent_ev_quality_mid,recent_ev_quality_volume_tilt,"
            "recent_ev_quality_quality_tilt,recent_ev_volume_quality_light,"
            "recent_ev_threshold_relax,archetype_support_shrunk_hr"
        ),
        help=f"Comma-separated arms. Available: {','.join(sorted(ARMS))}",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (args.report_dir / "suggestion_ablation_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_best_selected(args.report_dir)
    arm_names = [x.strip() for x in str(args.arms).split(",") if x.strip()]
    metrics_rows: list[dict[str, Any]] = []
    weekly_frames: list[pd.DataFrame] = []
    manifest = {
        "generated_by": "ablate_s52_policy_improvement_suggestions",
        "source_report_dir": str(args.report_dir),
        "source_rows": int(len(rows)),
        "arms": arm_names,
        "contract": "replay-only; fixed base/meta predictions and fixed exit geometry; May-June policy-tuned sample",
    }
    for arm in arm_names:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; available={sorted(ARMS)}")
        candidates, params, description = ARMS[arm](rows)
        ev_curve = fit_hierarchical_ev_curves(candidates)
        decisions, _equity, metrics = replay_candidates(
            candidates,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode="perps",
        )
        decisions.to_parquet(out_dir / f"{arm}_decisions.parquet", index=False)
        metrics_rows.append(
            _metrics_from_decisions(decisions, metrics, arm=arm, description=description)
        )
        weekly_frames.append(_weekly(decisions, arm=arm))

    metrics_df = pd.DataFrame(metrics_rows)
    if "current_best" in metrics_df["arm"].values:
        baseline = metrics_df.loc[metrics_df["arm"].eq("current_best")].iloc[0]
        for col in [
            "trade_count",
            "trades_per_day",
            "net_pnl",
            "worst_week",
            "mean_net_return_per_trade",
            "full_sl_rate",
            "timeout_rate",
        ]:
            metrics_df[f"delta_{col}_vs_current_best"] = metrics_df[col] - baseline[col]
    weekly_df = pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame()
    metrics_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    weekly_df.to_csv(out_dir / "weekly_metrics.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved {out_dir}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
