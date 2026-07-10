#!/usr/bin/env python3
"""Replay-only ablations for regime-calibration rank modulation.

This script compares middle-ground ways to use the existing
``per_regime_archetype_calibration_v1`` effect without letting it freely remove
all protected top10 rows. It keeps base/meta predictions and execution geometry
fixed, then changes only portfolio priority or the rank score consumed by the
portfolio replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class Arm:
    name: str
    family: str
    description: str
    priority_alpha: float = 0.0
    priority_shape: str = "linear"
    score_penalty_max: float = 0.0
    score_penalty_shape: str = "linear"
    direct_effect_multiplier: float | None = None
    direct_effect_clip: float = 0.05
    priority_floor: float = 0.35


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
        raise RuntimeError("No current default top10 hit-surprise rows found.")
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    return rows.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _effect_frame(rows: pd.DataFrame) -> pd.DataFrame:
    raw = pd.to_numeric(rows.get("rank_pct_raw"), errors="coerce")
    unprotected = pd.to_numeric(rows.get("rank_pct_regime_ev_unprotected"), errors="coerce")
    current = pd.to_numeric(rows.get("rank_pct"), errors="coerce")
    raw = raw.fillna(current).fillna(0.90).astype(float)
    unprotected = unprotected.fillna(current).fillna(raw).astype(float)
    current = current.fillna(raw).astype(float)
    full_effect = (unprotected - raw).astype(float)
    protected_effect = (current - raw).astype(float)
    badness = (raw - unprotected).clip(lower=0.0)
    scale = float(badness.quantile(0.95))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 0.10
    bad_unit = (badness / scale).clip(0.0, 1.0)
    return pd.DataFrame(
        {
            "raw": raw,
            "unprotected": unprotected,
            "current": current,
            "full_effect": full_effect,
            "protected_effect": protected_effect,
            "badness": badness,
            "bad_unit": bad_unit,
        },
        index=rows.index,
    )


def _shape(values: pd.Series, shape: str) -> pd.Series:
    x = values.clip(0.0, 1.0)
    if shape == "convex":
        return x * x
    if shape == "sqrt":
        return np.sqrt(x)
    return x


def _apply_arm(rows: pd.DataFrame, arm: Arm) -> pd.DataFrame:
    work = rows.copy()
    effect = _effect_frame(work)
    if arm.priority_alpha > 0.0:
        shaped = _shape(effect["bad_unit"], arm.priority_shape)
        base_priority = pd.to_numeric(
            work.get("portfolio_priority_multiplier"), errors="coerce"
        ).fillna(1.0)
        multiplier = (1.0 - float(arm.priority_alpha) * shaped).clip(
            lower=float(arm.priority_floor),
            upper=1.25,
        )
        work["portfolio_priority_multiplier"] = (base_priority * multiplier).clip(0.20, 1.50)
    if arm.score_penalty_max > 0.0:
        shaped = _shape(effect["bad_unit"], arm.score_penalty_shape)
        score = effect["current"] - float(arm.score_penalty_max) * shaped
        work["rank_pct"] = score.clip(0.0, 1.0)
        work["rank_score_source"] = f"{arm.name}:current_minus_shaped_badness"
    if arm.direct_effect_multiplier is not None:
        clipped_effect = effect["full_effect"].clip(
            lower=-float(arm.direct_effect_clip),
            upper=float(arm.direct_effect_clip),
        )
        # Re-apply a controlled fraction of the original regime calibration
        # directly to the raw rank. This avoids a hard floor rule and lets the
        # replay decide whether marginal rows still clear top10.
        score = effect["raw"] + float(arm.direct_effect_multiplier) * clipped_effect
        work["rank_pct"] = score.clip(0.0, 1.0)
        work["rank_score_source"] = f"{arm.name}:raw_plus_clipped_regime_effect"
    work["regime_rank_ablation_arm"] = arm.name
    work["regime_rank_badness"] = effect["badness"].astype(np.float32)
    work["regime_rank_bad_unit"] = effect["bad_unit"].astype(np.float32)
    work["regime_rank_full_effect"] = effect["full_effect"].astype(np.float32)
    return work


def _arms() -> list[Arm]:
    return [
        Arm("baseline_current_default", "baseline", "Current protected top10 rank and current hit-surprise portfolio priority."),
        Arm("soft_priority_linear_a25", "soft_priority", "Priority-only linear penalty from regime down-rank badness.", priority_alpha=0.25, priority_shape="linear"),
        Arm("soft_priority_linear_a50", "soft_priority", "Priority-only stronger linear penalty from regime down-rank badness.", priority_alpha=0.50, priority_shape="linear"),
        Arm("soft_priority_convex_a15", "soft_priority", "Very mild priority-only convex penalty.", priority_alpha=0.15, priority_shape="convex"),
        Arm("soft_priority_convex_a25", "soft_priority", "Mild priority-only convex penalty.", priority_alpha=0.25, priority_shape="convex"),
        Arm("soft_priority_convex_a50", "soft_priority", "Priority-only convex penalty, mild for small effects and stronger for large effects.", priority_alpha=0.50, priority_shape="convex"),
        Arm("soft_priority_convex_a85", "soft_priority", "Priority-only stronger convex penalty.", priority_alpha=0.85, priority_shape="convex"),
        Arm("score_penalty_linear_clip0025", "score_modulation", "Linear rank-score penalty capped at 0.25 percentage points.", score_penalty_max=0.0025, score_penalty_shape="linear"),
        Arm("score_penalty_linear_clip005", "score_modulation", "Linear rank-score penalty capped at 0.5 percentage points.", score_penalty_max=0.005, score_penalty_shape="linear"),
        Arm("score_penalty_linear_clip0075", "score_modulation", "Linear rank-score penalty capped at 0.75 percentage points.", score_penalty_max=0.0075, score_penalty_shape="linear"),
        Arm("score_penalty_linear_clip010", "score_modulation", "Linear rank-score penalty capped at 1 percentage point.", score_penalty_max=0.010, score_penalty_shape="linear"),
        Arm("score_penalty_linear_clip020", "score_modulation", "Linear rank-score penalty capped at 2 percentage points.", score_penalty_max=0.020, score_penalty_shape="linear"),
        Arm("score_penalty_linear_clip035", "score_modulation", "Linear rank-score penalty capped at 3.5 percentage points.", score_penalty_max=0.035, score_penalty_shape="linear"),
        Arm("score_penalty_convex_clip005", "score_modulation", "Convex rank-score penalty capped at 0.5 percentage points.", score_penalty_max=0.005, score_penalty_shape="convex"),
        Arm("score_penalty_convex_clip010", "score_modulation", "Convex rank-score penalty capped at 1 percentage point.", score_penalty_max=0.010, score_penalty_shape="convex"),
        Arm("score_penalty_convex_clip020", "score_modulation", "Convex rank-score penalty capped at 2 percentage points.", score_penalty_max=0.020, score_penalty_shape="convex"),
        Arm("score_penalty_convex_clip035", "score_modulation", "Convex rank-score penalty capped at 3.5 percentage points.", score_penalty_max=0.035, score_penalty_shape="convex"),
        Arm("direct_effect_m010_clip010", "direct_effect", "Use 10% of regime effect clipped to 1 point.", direct_effect_multiplier=0.10, direct_effect_clip=0.010),
        Arm("direct_effect_m015_clip015", "direct_effect", "Use 15% of regime effect clipped to 1.5 points.", direct_effect_multiplier=0.15, direct_effect_clip=0.015),
        Arm("direct_effect_m025_clip015", "direct_effect", "Use 25% of regime effect clipped to 1.5 points.", direct_effect_multiplier=0.25, direct_effect_clip=0.015),
        Arm("direct_effect_m025_clip025", "direct_effect", "Use 25% of regime effect clipped to 2.5 points.", direct_effect_multiplier=0.25, direct_effect_clip=0.025),
        Arm("direct_effect_m050_clip025", "direct_effect", "Use 50% of regime effect clipped to 2.5 points.", direct_effect_multiplier=0.50, direct_effect_clip=0.025),
        Arm("direct_effect_m050_clip050", "direct_effect", "Use 50% of regime effect clipped to 5 points.", direct_effect_multiplier=0.50, direct_effect_clip=0.050),
        Arm("direct_effect_m075_clip050", "direct_effect", "Use 75% of regime effect clipped to 5 points.", direct_effect_multiplier=0.75, direct_effect_clip=0.050),
        Arm("direct_effect_m050_clip075", "direct_effect", "Use 50% of regime effect clipped to 7.5 points.", direct_effect_multiplier=0.50, direct_effect_clip=0.075),
        Arm("direct_effect_m075_clip075", "direct_effect", "Use 75% of regime effect clipped to 7.5 points.", direct_effect_multiplier=0.75, direct_effect_clip=0.075),
    ]


def _metrics_from_decisions(decisions: pd.DataFrame, metrics: dict[str, Any], *, arm: Arm) -> dict[str, Any]:
    accepted = decisions[decisions["accepted"]].copy()
    return {
        "arm": arm.name,
        "family": arm.family,
        "description": arm.description,
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


def _weekly(decisions: pd.DataFrame, *, arm: Arm) -> pd.DataFrame:
    accepted = decisions[decisions["accepted"]].copy()
    if accepted.empty:
        return pd.DataFrame()
    ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["week_start"] = (
        ts.dt.floor("D") - pd.to_timedelta((ts.dt.weekday - 1) % 7, unit="D")
    ).dt.strftime("%Y-%m-%d")
    accepted["month"] = ts.dt.strftime("%Y-%m")
    accepted = accepted[accepted["month"].isin(["2026-05", "2026-06"])].copy()
    rows: list[dict[str, Any]] = []
    for (month, week), group in accepted.groupby(["month", "week_start"], sort=True):
        days = max(1.0, group["timestamp"].dt.floor("D").nunique())
        ret = pd.to_numeric(group["position_net_return"], errors="coerce")
        size = pd.to_numeric(group["position_size"], errors="coerce")
        rows.append(
            {
                "arm": arm.name,
                "family": arm.family,
                "month": month,
                "week_start": week,
                "trades": int(len(group)),
                "trades_per_day": float(len(group) / days),
                "avg_net_ev_per_trade": float(ret.mean()),
                "sum_net_ev_notional": float(ret.sum()),
                "net_pnl": float((ret * size).sum()),
                "full_sl_rate": float(group["position_exit_reason"].astype(str).eq("full_sl").mean()),
                "timeout_rate": float(group["position_exit_reason"].astype(str).eq("timeout").mean()),
            }
        )
    return pd.DataFrame(rows)


def _add_deltas(summary: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if "baseline_current_default" in set(out["arm"]):
        base = out.loc[out["arm"].eq("baseline_current_default")].iloc[0]
        for col in [
            "trade_count",
            "trades_per_day",
            "net_pnl",
            "worst_week",
            "mean_net_return_per_trade",
            "full_sl_rate",
            "timeout_rate",
        ]:
            out[f"delta_{col}_vs_baseline"] = out[col] - base[col]
    if not weekly.empty:
        stability_rows: list[dict[str, Any]] = []
        for arm, group in weekly.groupby("arm", sort=False):
            pnl = pd.to_numeric(group["net_pnl"], errors="coerce")
            ev = pd.to_numeric(group["avg_net_ev_per_trade"], errors="coerce")
            weak = group[group["week_start"].isin(["2026-06-16", "2026-06-23"])]
            stability_rows.append(
                {
                    "arm": arm,
                    "stable_net_score": float(pnl.mean() - 0.5 * pnl.std(ddof=0) + 0.25 * pnl.min()),
                    "stable_ev_score": float(ev.mean() - 0.5 * ev.std(ddof=0) + 0.25 * ev.min()),
                    "june_weak_net_pnl": float(pd.to_numeric(weak["net_pnl"], errors="coerce").sum()),
                    "june_weak_avg_ev": float(
                        np.average(
                            pd.to_numeric(weak["avg_net_ev_per_trade"], errors="coerce"),
                            weights=pd.to_numeric(weak["trades"], errors="coerce"),
                        )
                    )
                    if int(pd.to_numeric(weak["trades"], errors="coerce").sum()) > 0
                    else np.nan,
                    "june_weak_trades": int(pd.to_numeric(weak["trades"], errors="coerce").sum()),
                }
            )
        st = pd.DataFrame(stability_rows)
        out = out.merge(st, on="arm", how="left")
        if "baseline_current_default" in set(out["arm"]):
            base = out.loc[out["arm"].eq("baseline_current_default")].iloc[0]
            for col in ["stable_net_score", "stable_ev_score", "june_weak_net_pnl", "june_weak_avg_ev", "june_weak_trades"]:
                out[f"delta_{col}_vs_baseline"] = out[col] - base[col]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    rows = _load_current_rows(args.report_dir)
    out_dir = args.out_dir or (args.report_dir / "regime_rank_modulation_ablation_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    weekly_frames: list[pd.DataFrame] = []
    arms = _arms()
    for arm in arms:
        arm_rows = _apply_arm(rows, arm)
        candidates = _portfolio_candidate_table(arm_rows)
        ev_curve = fit_hierarchical_ev_curves(candidates)
        decisions, _equity, metrics = replay_candidates(
            candidates,
            _params(),
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode="perps",
        )
        decisions.to_parquet(out_dir / f"{arm.name}_decisions.parquet", index=False)
        metric_rows.append(_metrics_from_decisions(decisions, metrics, arm=arm))
        weekly_frames.append(_weekly(decisions, arm=arm))

    summary = pd.DataFrame(metric_rows)
    weekly = pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame()
    summary = _add_deltas(summary, weekly)
    summary = summary.sort_values(["net_pnl", "mean_net_return_per_trade"], ascending=False)
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    weekly.to_csv(out_dir / "weekly_metrics.csv", index=False)
    manifest = {
        "generated_by": "ablate_s52_regime_calibration_rank_modulation",
        "source_report_dir": str(args.report_dir),
        "source_rows": int(len(rows)),
        "policy_contract": "replay-only; fixed base/meta predictions and exit geometry; May-June optimisation sample",
        "calibration_policy_id": "per_regime_archetype_calibration_v1",
        "arms": [arm.__dict__ for arm in arms],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved {out_dir}")
    cols = [
        "arm",
        "family",
        "trade_count",
        "net_pnl",
        "mean_net_return_per_trade",
        "worst_week",
        "full_sl_rate",
        "timeout_rate",
        "june_weak_net_pnl",
        "june_weak_avg_ev",
        "delta_net_pnl_vs_baseline",
        "delta_mean_net_return_per_trade_vs_baseline",
        "delta_june_weak_net_pnl_vs_baseline",
    ]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
