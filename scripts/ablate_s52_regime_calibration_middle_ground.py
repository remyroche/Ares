#!/usr/bin/env python3
"""Replay-only middle-ground ablations for regime calibration protection.

The current policy protects the top10 admission floor from
``per_regime_archetype_calibration_v1`` because the unprotected calibration
removed too many rows. This script tests softer alternatives while keeping the
same base/meta predictions, hit-surprise policy, exit geometry, and portfolio
replay.
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

TOP10_FLOOR = 0.90


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _params() -> PortfolioPolicyParams:
    return PortfolioPolicyParams(
        max_concurrent_positions=6,
        max_concurrent_per_side=None,
        max_concurrent_per_strategy=None,
        max_new_entries_per_bar=2,
        max_total_wallet_allocation_pct=0.75,
        global_threshold_floor=0.0,
        occupancy_threshold_alpha=0.30,
        occupancy_threshold_power=1.5,
        allocation_threshold_alpha=0.30,
        allocation_threshold_power=1.0,
        rank_size_power=1.5,
        rank_multiplier_min=0.5,
        rank_multiplier_max=1.5,
        min_position_size=1.0,
    )


def _load_current_rows(report_dir: Path) -> pd.DataFrame:
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
        raise RuntimeError("No current top10 hit-surprise rows found.")
    return rows.reset_index(drop=True)


def _effect_parts(rows: pd.DataFrame, *, clip: float) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    raw = pd.to_numeric(rows["rank_pct_raw"], errors="coerce").fillna(
        pd.to_numeric(rows["rank_pct"], errors="coerce")
    )
    protected = pd.to_numeric(rows["rank_pct"], errors="coerce").fillna(raw)
    unprotected = pd.to_numeric(rows["rank_pct_regime_ev_unprotected"], errors="coerce").fillna(raw)
    adverse = (raw - unprotected).clip(lower=0.0)
    unit = (adverse / max(float(clip), 1e-9)).clip(lower=0.0, upper=1.0)
    return raw, protected, unprotected, unit


def _candidate_table(rows: pd.DataFrame) -> pd.DataFrame:
    candidates = _portfolio_candidate_table(rows)
    if candidates.empty:
        return candidates
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    return candidates.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _arm_baseline(rows: pd.DataFrame) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    return rows.copy(), "current protected top10 floor", {}


def _soft_priority(rows: pd.DataFrame, *, clip: float, strength: float, power: float) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    work = rows.copy()
    _raw, _protected, _unprotected, unit = _effect_parts(work, clip=clip)
    penalty = float(strength) * np.power(unit, float(power))
    base_priority = pd.to_numeric(work.get("portfolio_priority_multiplier"), errors="coerce").fillna(1.0)
    base_size = (
        pd.to_numeric(work["portfolio_size_multiplier"], errors="coerce").fillna(1.0)
        if "portfolio_size_multiplier" in work.columns
        else pd.Series(1.0, index=work.index)
    )
    work["portfolio_priority_multiplier"] = (base_priority * (1.0 - penalty)).clip(0.35, 1.50)
    # Keep size impact weaker than priority; this is a soft allocation test, not a gate.
    work["portfolio_size_multiplier"] = (base_size * (1.0 - 0.50 * penalty)).clip(0.50, 1.50)
    return (
        work,
        "soft priority/size penalty from adverse unprotected regime-calibration effect",
        {"clip": clip, "strength": strength, "power": power},
    )


def _floor_weaken(rows: pd.DataFrame, *, clip: float, strength: float, power: float) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    work = rows.copy()
    _raw, protected, unprotected, unit = _effect_parts(work, clip=clip)
    weaken = (float(strength) * np.power(unit, float(power))).clip(lower=0.0, upper=1.0)
    # Continuous weakening: move protected rank toward unprotected rank, but do
    # not add any discrete regime threshold.
    work["rank_pct"] = (protected - weaken * (protected - unprotected).clip(lower=0.0)).clip(0.0, 1.0)
    return (
        work,
        "continuous floor weakening toward unprotected regime-calibrated rank",
        {"clip": clip, "strength": strength, "power": power},
    )


def _direct_clip(rows: pd.DataFrame, *, clip: float, multiplier: float) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    work = rows.copy()
    raw, _protected, unprotected, _unit = _effect_parts(work, clip=clip)
    effect = (raw - unprotected).clip(lower=0.0, upper=float(clip))
    work["rank_pct"] = (raw - float(multiplier) * effect).clip(0.0, 1.0)
    return (
        work,
        "direct clipped multiplier on per_regime_archetype_calibration_v1 adverse rank effect",
        {"clip": clip, "multiplier": multiplier},
    )


def _surplus_capped(rows: pd.DataFrame, *, clip: float, multiplier: float, surplus_cap: float) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    work = rows.copy()
    raw, _protected, unprotected, _unit = _effect_parts(work, clip=clip)
    effect = (raw - unprotected).clip(lower=0.0, upper=float(clip)) * float(multiplier)
    surplus = (raw - TOP10_FLOOR).clip(lower=0.0)
    penalty = np.minimum(effect, surplus * float(surplus_cap))
    work["rank_pct"] = (raw - penalty).clip(0.0, 1.0)
    return (
        work,
        "surplus-capped direct regime effect; high raw-rank rows retain more protection",
        {"clip": clip, "multiplier": multiplier, "surplus_cap": surplus_cap},
    )


def _build_arms() -> dict[str, Callable[[pd.DataFrame], tuple[pd.DataFrame, str, dict[str, Any]]]]:
    arms: dict[str, Callable[[pd.DataFrame], tuple[pd.DataFrame, str, dict[str, Any]]]] = {
        "a0_baseline": _arm_baseline,
    }
    for clip in (0.05, 0.10, 0.15):
        for strength in (0.20, 0.35, 0.50):
            arms[f"a1_soft_priority_linear_c{int(clip*100):02d}_s{int(strength*100):02d}"] = (
                lambda rows, clip=clip, strength=strength: _soft_priority(
                    rows, clip=clip, strength=strength, power=1.0
                )
            )
            arms[f"a1_soft_priority_convex_c{int(clip*100):02d}_s{int(strength*100):02d}"] = (
                lambda rows, clip=clip, strength=strength: _soft_priority(
                    rows, clip=clip, strength=strength, power=2.0
                )
            )
    for clip in (0.05, 0.10, 0.15):
        for strength in (0.25, 0.50, 0.75):
            arms[f"a2_floor_linear_c{int(clip*100):02d}_s{int(strength*100):02d}"] = (
                lambda rows, clip=clip, strength=strength: _floor_weaken(
                    rows, clip=clip, strength=strength, power=1.0
                )
            )
            arms[f"a2_floor_convex_c{int(clip*100):02d}_s{int(strength*100):02d}"] = (
                lambda rows, clip=clip, strength=strength: _floor_weaken(
                    rows, clip=clip, strength=strength, power=2.0
                )
            )
    for clip in (0.03, 0.05, 0.08, 0.12):
        for multiplier in (0.25, 0.50, 0.75):
            arms[f"a3_direct_clip_c{int(clip*100):02d}_m{int(multiplier*100):02d}"] = (
                lambda rows, clip=clip, multiplier=multiplier: _direct_clip(
                    rows, clip=clip, multiplier=multiplier
                )
            )
    for clip in (0.05, 0.08, 0.12):
        for multiplier in (0.50, 0.75, 1.00):
            for surplus_cap in (0.50, 0.75, 1.00, 1.25):
                arms[
                    f"a3_surplus_c{int(clip*100):02d}_m{int(multiplier*100):03d}_cap{int(surplus_cap*100):03d}"
                ] = (
                    lambda rows, clip=clip, multiplier=multiplier, surplus_cap=surplus_cap: _surplus_capped(
                        rows,
                        clip=clip,
                        multiplier=multiplier,
                        surplus_cap=surplus_cap,
                    )
                )
    return arms


def _summary(decisions: pd.DataFrame, metrics: dict[str, Any], *, arm: str, description: str, params: dict[str, Any]) -> dict[str, Any]:
    accepted = decisions[decisions["accepted"]].copy()
    ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce") if not accepted.empty else pd.Series(dtype="datetime64[ns, UTC]")
    accepted["week_start"] = (
        ts.dt.floor("D") - pd.to_timedelta((ts.dt.weekday - 1) % 7, unit="D")
    ).dt.strftime("%Y-%m-%d") if not accepted.empty else []
    weak = accepted[accepted["week_start"].isin(["2026-06-16", "2026-06-23"])] if not accepted.empty else accepted
    net = pd.to_numeric(accepted.get("position_net_return"), errors="coerce") if not accepted.empty else pd.Series(dtype=float)
    weak_net = pd.to_numeric(weak.get("position_net_return"), errors="coerce") if not weak.empty else pd.Series(dtype=float)
    return {
        "arm": arm,
        "description": description,
        **{f"param_{k}": v for k, v in params.items()},
        "trade_count": int(metrics.get("trade_count", 0)),
        "trades_per_day": _safe_float(metrics.get("trades_per_day")),
        "net_pnl": _safe_float(metrics.get("net_pnl")),
        "gross_pnl": _safe_float(metrics.get("gross_pnl")),
        "max_drawdown": _safe_float(metrics.get("max_drawdown")),
        "worst_week": _safe_float(metrics.get("worst_week")),
        "mean_net_return_per_trade": _safe_float(metrics.get("mean_net_return_per_trade")),
        "notional_weighted_net_return": _safe_float(metrics.get("notional_weighted_net_return")),
        "full_sl_rate": _safe_float(metrics.get("full_sl_rate")),
        "timeout_rate": _safe_float(metrics.get("timeout_rate")),
        "accepted_rows": int(len(accepted)),
        "weak_week_trades": int(len(weak)),
        "weak_week_mean_net_return": float(weak_net.mean()) if len(weak_net) else np.nan,
        "weak_week_sum_net_return": float(weak_net.sum()) if len(weak_net) else np.nan,
        "weak_week_full_sl_rate": float(weak["position_exit_reason"].astype(str).eq("full_sl").mean()) if not weak.empty else np.nan,
        "weak_week_timeout_rate": float(weak["position_exit_reason"].astype(str).eq("timeout").mean()) if not weak.empty else np.nan,
        "accepted_mean_net_return_check": float(net.mean()) if len(net) else np.nan,
    }


def _weekly(decisions: pd.DataFrame, *, arm: str) -> pd.DataFrame:
    accepted = decisions[decisions["accepted"]].copy()
    if accepted.empty:
        return pd.DataFrame()
    ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["week_start"] = (
        ts.dt.floor("D") - pd.to_timedelta((ts.dt.weekday - 1) % 7, unit="D")
    ).dt.strftime("%Y-%m-%d")
    accepted["month"] = ts.dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    for (month, week), group in accepted.groupby(["month", "week_start"], sort=True):
        days = max(1.0, group["timestamp"].dt.floor("D").nunique())
        returns = pd.to_numeric(group["position_net_return"], errors="coerce")
        sizes = pd.to_numeric(group["position_size"], errors="coerce")
        rows.append(
            {
                "arm": arm,
                "month": month,
                "week_start": week,
                "trades": int(len(group)),
                "trades_per_day": float(len(group) / days),
                "avg_net_ev_per_trade": float(returns.mean()),
                "net_pnl": float((returns * sizes).sum()),
                "full_sl_rate": float(group["position_exit_reason"].astype(str).eq("full_sl").mean()),
                "timeout_rate": float(group["position_exit_reason"].astype(str).eq("timeout").mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--arms", default="", help="Optional comma-separated subset of arms.")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.report_dir / "regime_calibration_middle_ground_ablation_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_current_rows(args.report_dir)
    all_arms = _build_arms()
    arm_names = [name.strip() for name in args.arms.split(",") if name.strip()] or list(all_arms)
    summaries: list[dict[str, Any]] = []
    weekly_frames: list[pd.DataFrame] = []
    manifest = {
        "generated_by": "ablate_s52_regime_calibration_middle_ground",
        "source_report_dir": str(args.report_dir),
        "source_rows": int(len(rows)),
        "contract": "replay-only; fixed base/meta predictions, hit-surprise policy, exit geometry, and portfolio replay",
        "effect_definition": "adverse_effect = max(0, rank_pct_raw - rank_pct_regime_ev_unprotected)",
        "arms": arm_names,
    }
    for arm in arm_names:
        if arm not in all_arms:
            raise ValueError(f"Unknown arm {arm!r}")
        arm_rows, description, params = all_arms[arm](rows)
        candidates = _candidate_table(arm_rows)
        ev_curve = fit_hierarchical_ev_curves(candidates)
        decisions, _equity, metrics = replay_candidates(
            candidates,
            _params(),
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode="perps",
        )
        decisions.to_parquet(out_dir / f"{arm}_decisions.parquet", index=False)
        summaries.append(_summary(decisions, metrics, arm=arm, description=description, params=params))
        weekly_frames.append(_weekly(decisions, arm=arm))
    summary = pd.DataFrame(summaries)
    if "a0_baseline" in summary["arm"].values:
        base = summary.loc[summary["arm"].eq("a0_baseline")].iloc[0]
        for col in (
            "trade_count",
            "net_pnl",
            "worst_week",
            "mean_net_return_per_trade",
            "full_sl_rate",
            "timeout_rate",
            "weak_week_mean_net_return",
            "weak_week_sum_net_return",
            "weak_week_full_sl_rate",
        ):
            summary[f"delta_{col}_vs_baseline"] = summary[col] - base[col]
    weekly = pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame()
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    weekly.to_csv(out_dir / "weekly_metrics.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved {out_dir}")
    sort_cols = ["net_pnl", "mean_net_return_per_trade"]
    print(summary.sort_values(sort_cols, ascending=False).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
