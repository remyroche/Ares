#!/usr/bin/env python3
"""Build a consolidated evidence matrix for contextual TP/SL A/B candidates.

The matrix intentionally separates:

* long-window development replay evidence;
* portfolio-wide frozen/post-freeze readiness;
* optional eligible-head frozen replay evidence.

This keeps promising historical A/B variants visible without confusing them
with deployable candidates when live-like accepted-trade evidence is sparse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEV_DASHBOARD = ROOT / "data_perp/reports/contextual_tp_sl_ablation_dashboard_v1_20260701"
DEFAULT_WORKFLOW = ROOT / "data_perp/reports/contextual_tp_sl_ablation_workflow_v15_eligible_head_gate_20260701"
DEFAULT_SCORECARD_DIR = ROOT / "data_perp/reports/contextual_tp_sl_reliability_feature_scorecard_v1_20260701"
DEFAULT_FAMILY_EFFECT_DIR = ROOT / "data_perp/reports/contextual_tp_sl_diagnostic_family_effect_summary_20260701"
DEFAULT_OUT = ROOT / "data_perp/reports/contextual_tp_sl_consolidated_evidence_matrix_v1_20260701"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _canonical_head_from_series(head: pd.Series, strategy_id: pd.Series) -> pd.Series:
    raw_head = head.fillna("").astype(str).str.strip()
    raw_strategy = strategy_id.fillna("").astype(str)
    out = pd.Series("unknown", index=head.index, dtype=object)
    lower_head = raw_head.str.lower()
    lower_strategy = raw_strategy.str.lower()

    def assign(mask: pd.Series, value: str) -> None:
        out.loc[mask & out.eq("unknown")] = value

    assign(lower_head.str.contains("long_bars", regex=False), "long_bars")
    assign(lower_head.str.contains("long_dist", regex=False), "long_dist")
    assign(lower_head.str.contains("short_asset", regex=False), "short_asset")
    assign(lower_head.str.contains("short_boll", regex=False), "short_bollinger")
    assign(lower_strategy.str.contains("long_bars", regex=False), "long_bars")
    assign(lower_strategy.str.contains("long_dist", regex=False), "long_dist")
    assign(lower_strategy.str.contains("short_asset", regex=False), "short_asset")
    assign(lower_strategy.str.contains("short_boll", regex=False), "short_bollinger")
    return out


def _best_dev_rows(dashboard: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if dashboard.empty:
        return pd.DataFrame()
    frame = dashboard.copy()
    if "variant" not in frame.columns:
        return pd.DataFrame()
    frame = frame[~frame["variant"].astype(str).eq("baseline")].copy()
    if frame.empty:
        return pd.DataFrame()
    for col in (
        "delta_vs_baseline_net_pnl",
        "delta_vs_baseline_full_sl_rate",
        "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
        "positive_net_month_share",
        "positive_net_head_share",
        "jaccard_vs_baseline",
    ):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.sort_values(
        [
            "development_candidate_pass",
            "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
            "delta_vs_baseline_net_pnl",
        ],
        ascending=[False, False, False],
    )
    keep = [
        "variant",
        "role",
        "delta_vs_baseline_net_pnl",
        "delta_vs_baseline_full_sl_rate",
        "delta_vs_baseline_timeout_rate",
        "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
        "avg_week_pnl",
        "weekly_q05_pnl",
        "weekly_q10_pnl",
        "weekly_q20_pnl",
        "weekly_q35_pnl",
        "delta_vs_baseline_avg_week_pnl",
        "delta_vs_baseline_weekly_q05_pnl",
        "delta_vs_baseline_weekly_q10_pnl",
        "delta_vs_baseline_weekly_q20_pnl",
        "delta_vs_baseline_weekly_q35_pnl",
        "daily_q20_pnl",
        "daily_q35_pnl",
        "delta_vs_baseline_daily_q20_pnl",
        "delta_vs_baseline_daily_q35_pnl",
        "positive_net_month_share",
        "positive_net_head_share",
        "jaccard_vs_baseline",
        "entrants_vs_baseline",
        "removed_vs_baseline",
        "development_candidate_pass",
        "deployment_candidate_pass",
        "candidate_status",
    ]
    return frame[[col for col in keep if col in frame.columns]].head(top_n).copy()


def _weekly_tail_tradeoff(best_dev: pd.DataFrame) -> pd.DataFrame:
    if best_dev.empty:
        return pd.DataFrame()
    keep = [
        "variant",
        "role",
        "delta_vs_baseline_net_pnl",
        "delta_vs_baseline_full_sl_rate",
        "delta_vs_baseline_avg_week_pnl",
        "delta_vs_baseline_weekly_q05_pnl",
        "delta_vs_baseline_weekly_q10_pnl",
        "delta_vs_baseline_weekly_q20_pnl",
        "delta_vs_baseline_weekly_q35_pnl",
        "delta_vs_baseline_daily_q20_pnl",
        "delta_vs_baseline_daily_q35_pnl",
    ]
    out = best_dev[[col for col in keep if col in best_dev.columns]].copy()
    if out.empty:
        return out
    for col in out.columns:
        if col.startswith("delta_"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["tail_tradeoff_note"] = np.select(
        [
            out.get("delta_vs_baseline_weekly_q05_pnl", pd.Series(0.0, index=out.index)).lt(0.0)
            & out.get("delta_vs_baseline_net_pnl", pd.Series(0.0, index=out.index)).gt(0.0),
            out.get("delta_vs_baseline_weekly_q05_pnl", pd.Series(0.0, index=out.index)).ge(0.0)
            & out.get("delta_vs_baseline_full_sl_rate", pd.Series(0.0, index=out.index)).le(0.0),
        ],
        [
            "higher_pnl_but_weaker_weekly_q05",
            "pnl_and_tail_aligned",
        ],
        default="mixed_tail_tradeoff",
    )
    return out


def _scale01(values: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if not higher_is_better:
        numeric = -numeric
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series(0.0, index=values.index)
    lo = float(finite.min())
    hi = float(finite.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=values.index)
    return ((numeric - lo) / (hi - lo)).fillna(0.0).clip(0.0, 1.0)


def _risk_profile_candidates(dashboard: pd.DataFrame) -> pd.DataFrame:
    if dashboard.empty or "variant" not in dashboard.columns:
        return pd.DataFrame()
    frame = dashboard[~dashboard["variant"].astype(str).eq("baseline")].copy()
    if frame.empty:
        return pd.DataFrame()
    metric_cols = [
        "delta_vs_baseline_net_pnl",
        "delta_vs_baseline_full_sl_rate",
        "delta_vs_baseline_weekly_q05_pnl",
        "delta_vs_baseline_weekly_q10_pnl",
        "delta_vs_baseline_weekly_q20_pnl",
        "delta_vs_baseline_weekly_q35_pnl",
        "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
        "positive_net_month_share",
        "positive_net_head_share",
        "jaccard_vs_baseline",
    ]
    for col in metric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["pnl_score"] = _scale01(frame.get("delta_vs_baseline_net_pnl", pd.Series(index=frame.index)))
    frame["full_sl_score"] = _scale01(
        frame.get("delta_vs_baseline_full_sl_rate", pd.Series(index=frame.index)),
        higher_is_better=False,
    )
    tail_cols = [
        col
        for col in (
            "delta_vs_baseline_weekly_q05_pnl",
            "delta_vs_baseline_weekly_q10_pnl",
            "delta_vs_baseline_weekly_q20_pnl",
            "delta_vs_baseline_weekly_q35_pnl",
        )
        if col in frame.columns
    ]
    if tail_cols:
        tail_scores = pd.concat([_scale01(frame[col]) for col in tail_cols], axis=1)
        frame["weekly_tail_score"] = tail_scores.mean(axis=1)
    else:
        frame["weekly_tail_score"] = 0.0
    consistency_cols = [
        col for col in ("positive_net_month_share", "positive_net_head_share") if col in frame.columns
    ]
    if consistency_cols:
        frame["consistency_score"] = frame[consistency_cols].mean(axis=1).fillna(0.0).clip(0.0, 1.0)
    else:
        frame["consistency_score"] = 0.0
    frame["low_churn_score"] = pd.to_numeric(
        frame.get("jaccard_vs_baseline", pd.Series(0.0, index=frame.index)), errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)

    profiles = {
        "high_return": {
            "pnl_score": 0.60,
            "weekly_tail_score": 0.20,
            "full_sl_score": 0.10,
            "consistency_score": 0.10,
            "low_churn_score": 0.00,
        },
        "balanced_pnl_tail": {
            "pnl_score": 0.35,
            "weekly_tail_score": 0.30,
            "full_sl_score": 0.20,
            "consistency_score": 0.15,
            "low_churn_score": 0.00,
        },
        "conservative_tail": {
            "pnl_score": 0.20,
            "weekly_tail_score": 0.35,
            "full_sl_score": 0.30,
            "consistency_score": 0.10,
            "low_churn_score": 0.05,
        },
        "low_churn_tail": {
            "pnl_score": 0.20,
            "weekly_tail_score": 0.15,
            "full_sl_score": 0.15,
            "consistency_score": 0.00,
            "low_churn_score": 0.50,
        },
    }

    rows: list[dict[str, Any]] = []
    for profile, weights in profiles.items():
        scores = pd.Series(0.0, index=frame.index)
        for col, weight in weights.items():
            scores = scores + frame[col].fillna(0.0) * float(weight)
        ranked = frame.assign(risk_profile=profile, risk_profile_score=scores).sort_values(
            ["risk_profile_score", "delta_vs_baseline_net_pnl"],
            ascending=[False, False],
        )
        for rank, (_, row) in enumerate(ranked.head(3).iterrows(), start=1):
            rows.append(
                {
                    "risk_profile": profile,
                    "risk_profile_rank": rank,
                    "variant": str(row.get("variant") or ""),
                    "role": str(row.get("role") or ""),
                    "risk_profile_score": row.get("risk_profile_score"),
                    "pnl_score": row.get("pnl_score"),
                    "weekly_tail_score": row.get("weekly_tail_score"),
                    "full_sl_score": row.get("full_sl_score"),
                    "consistency_score": row.get("consistency_score"),
                    "low_churn_score": row.get("low_churn_score"),
                    "delta_vs_baseline_net_pnl": row.get("delta_vs_baseline_net_pnl"),
                    "delta_vs_baseline_full_sl_rate": row.get("delta_vs_baseline_full_sl_rate"),
                    "delta_vs_baseline_weekly_q05_pnl": row.get("delta_vs_baseline_weekly_q05_pnl"),
                    "delta_vs_baseline_weekly_q10_pnl": row.get("delta_vs_baseline_weekly_q10_pnl"),
                    "delta_vs_baseline_weekly_q20_pnl": row.get("delta_vs_baseline_weekly_q20_pnl"),
                    "delta_vs_baseline_weekly_q35_pnl": row.get("delta_vs_baseline_weekly_q35_pnl"),
                    "jaccard_vs_baseline": row.get("jaccard_vs_baseline"),
                    "development_candidate_pass": row.get("development_candidate_pass"),
                    "candidate_status": row.get("candidate_status"),
                }
            )
    return pd.DataFrame(rows)


def _guardrail_matrix(dashboard: pd.DataFrame) -> pd.DataFrame:
    if dashboard.empty or "variant" not in dashboard.columns:
        return pd.DataFrame()
    frame = dashboard[~dashboard["variant"].astype(str).eq("baseline")].copy()
    if frame.empty:
        return pd.DataFrame()
    metric_cols = [
        "delta_vs_baseline_net_pnl",
        "delta_vs_baseline_full_sl_rate",
        "delta_vs_baseline_timeout_rate",
        "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
        "delta_vs_baseline_weekly_q05_pnl",
        "delta_vs_baseline_weekly_q10_pnl",
        "delta_vs_baseline_weekly_q20_pnl",
        "delta_vs_baseline_weekly_q35_pnl",
        "positive_net_month_share",
        "positive_net_head_share",
        "jaccard_vs_baseline",
    ]
    for col in metric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    tail_cols = [
        col
        for col in (
            "delta_vs_baseline_weekly_q05_pnl",
            "delta_vs_baseline_weekly_q10_pnl",
            "delta_vs_baseline_weekly_q20_pnl",
            "delta_vs_baseline_weekly_q35_pnl",
        )
        if col in frame.columns
    ]
    if tail_cols:
        tails = frame[tail_cols].apply(pd.to_numeric, errors="coerce")
        frame["tail_positive_count"] = tails.ge(0.0).sum(axis=1)
        frame["tail_tested_count"] = tails.notna().sum(axis=1)
        frame["worst_weekly_tail_delta"] = tails.min(axis=1)
        frame["tail_shortfall_count"] = tails.lt(0.0).sum(axis=1)
        frame["tail_shortfall_fields"] = tails.lt(0.0).apply(
            lambda row: ",".join(col.replace("delta_vs_baseline_", "") for col, bad in row.items() if bool(bad)),
            axis=1,
        )
    else:
        frame["tail_positive_count"] = 0
        frame["tail_tested_count"] = 0
        frame["worst_weekly_tail_delta"] = np.nan
        frame["tail_shortfall_count"] = 0
        frame["tail_shortfall_fields"] = ""

    frame["pnl_positive"] = frame.get("delta_vs_baseline_net_pnl", pd.Series(0.0, index=frame.index)).gt(0.0)
    frame["objective_positive"] = frame.get(
        "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
        pd.Series(0.0, index=frame.index),
    ).gt(0.0)
    frame["full_sl_nonworse"] = frame.get(
        "delta_vs_baseline_full_sl_rate",
        pd.Series(0.0, index=frame.index),
    ).le(0.0)
    frame["all_weekly_tail_nonworse"] = frame["tail_positive_count"].eq(frame["tail_tested_count"])
    frame["strict_tail_pass"] = (
        frame["pnl_positive"]
        & frame["objective_positive"]
        & frame["full_sl_nonworse"]
        & frame["all_weekly_tail_nonworse"]
    )
    frame["pragmatic_tail_pass"] = (
        frame["pnl_positive"]
        & frame["objective_positive"]
        & frame["full_sl_nonworse"]
        & frame.get("delta_vs_baseline_weekly_q05_pnl", pd.Series(0.0, index=frame.index)).ge(0.0)
        & frame.get("delta_vs_baseline_weekly_q20_pnl", pd.Series(0.0, index=frame.index)).ge(0.0)
    )
    frame["guardrail_label"] = np.select(
        [
            frame["strict_tail_pass"],
            frame["pragmatic_tail_pass"] & frame["tail_shortfall_count"].gt(0),
            frame["pnl_positive"] & frame["objective_positive"],
        ],
        [
            "strict_pnl_tail_pass",
            "pnl_tail_candidate_with_tail_warning",
            "pnl_candidate_tail_incomplete",
        ],
        default="reject_or_diagnostic",
    )
    frame["guardrail_rank_score"] = (
        frame.get("delta_vs_baseline_net_pnl", pd.Series(0.0, index=frame.index)).fillna(0.0)
        + 1000.0
        * frame.get("delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20", pd.Series(0.0, index=frame.index)).fillna(0.0)
        - 250.0 * frame["tail_shortfall_count"].fillna(0.0)
    )
    keep = [
        "variant",
        "role",
        "guardrail_label",
        "strict_tail_pass",
        "pragmatic_tail_pass",
        "tail_positive_count",
        "tail_tested_count",
        "tail_shortfall_count",
        "tail_shortfall_fields",
        "worst_weekly_tail_delta",
        "guardrail_rank_score",
        "delta_vs_baseline_net_pnl",
        "delta_vs_baseline_full_sl_rate",
        "delta_vs_baseline_timeout_rate",
        "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
        "delta_vs_baseline_weekly_q05_pnl",
        "delta_vs_baseline_weekly_q10_pnl",
        "delta_vs_baseline_weekly_q20_pnl",
        "delta_vs_baseline_weekly_q35_pnl",
        "positive_net_month_share",
        "positive_net_head_share",
        "jaccard_vs_baseline",
        "candidate_status",
    ]
    return frame[[col for col in keep if col in frame.columns]].sort_values(
        ["strict_tail_pass", "pragmatic_tail_pass", "guardrail_rank_score"],
        ascending=[False, False, False],
    )


def _long_period_adequacy_matrix(dashboard: pd.DataFrame) -> pd.DataFrame:
    if dashboard.empty or "variant" not in dashboard.columns:
        return pd.DataFrame()
    frame = dashboard[~dashboard["variant"].astype(str).eq("baseline")].copy()
    if frame.empty:
        return pd.DataFrame()
    metric_cols = [
        "trade_count",
        "months",
        "heads",
        "positive_net_month_share",
        "positive_net_head_share",
        "min_month_delta_net_pnl",
        "min_head_delta_net_pnl",
        "delta_vs_baseline_net_pnl",
        "delta_vs_baseline_hit_rate",
        "delta_vs_baseline_full_sl_rate",
        "delta_vs_baseline_timeout_rate",
        "delta_vs_baseline_weekly_q05_pnl",
        "delta_vs_baseline_weekly_q10_pnl",
        "delta_vs_baseline_weekly_q20_pnl",
        "delta_vs_baseline_weekly_q35_pnl",
        "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
        "jaccard_vs_baseline",
    ]
    for col in metric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["long_period_months_pass"] = frame.get("months", pd.Series(0, index=frame.index)).ge(5)
    frame["long_period_heads_pass"] = frame.get("heads", pd.Series(0, index=frame.index)).ge(4)
    frame["long_period_trades_pass"] = frame.get("trade_count", pd.Series(0, index=frame.index)).ge(5000)
    frame["month_consistency_pass"] = (
        frame.get("positive_net_month_share", pd.Series(0.0, index=frame.index)).ge(1.0)
        & frame.get("min_month_delta_net_pnl", pd.Series(0.0, index=frame.index)).gt(0.0)
    )
    frame["head_consistency_pass"] = (
        frame.get("positive_net_head_share", pd.Series(0.0, index=frame.index)).ge(1.0)
        & frame.get("min_head_delta_net_pnl", pd.Series(0.0, index=frame.index)).gt(0.0)
    )
    frame["pnl_objective_pass"] = (
        frame.get("delta_vs_baseline_net_pnl", pd.Series(0.0, index=frame.index)).gt(0.0)
        & frame.get(
            "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
            pd.Series(0.0, index=frame.index),
        ).gt(0.0)
    )
    frame["full_sl_pass"] = frame.get("delta_vs_baseline_full_sl_rate", pd.Series(0.0, index=frame.index)).le(0.0)
    tail_cols = [
        col
        for col in (
            "delta_vs_baseline_weekly_q05_pnl",
            "delta_vs_baseline_weekly_q10_pnl",
            "delta_vs_baseline_weekly_q20_pnl",
            "delta_vs_baseline_weekly_q35_pnl",
        )
        if col in frame.columns
    ]
    if tail_cols:
        tails = frame[tail_cols].apply(pd.to_numeric, errors="coerce")
        frame["tail_nonnegative_count"] = tails.ge(0.0).sum(axis=1)
        frame["tail_tested_count"] = tails.notna().sum(axis=1)
        frame["worst_weekly_tail_delta"] = tails.min(axis=1)
        frame["all_weekly_tail_pass"] = frame["tail_nonnegative_count"].eq(frame["tail_tested_count"])
        frame["pragmatic_weekly_tail_pass"] = (
            frame.get("delta_vs_baseline_weekly_q05_pnl", pd.Series(np.nan, index=frame.index)).ge(0.0)
            & frame.get("delta_vs_baseline_weekly_q10_pnl", pd.Series(np.nan, index=frame.index)).notna()
            & frame.get("delta_vs_baseline_weekly_q20_pnl", pd.Series(np.nan, index=frame.index)).ge(0.0)
            & frame.get("delta_vs_baseline_weekly_q35_pnl", pd.Series(np.nan, index=frame.index)).ge(0.0)
        )
    else:
        frame["tail_nonnegative_count"] = 0
        frame["tail_tested_count"] = 0
        frame["worst_weekly_tail_delta"] = np.nan
        frame["all_weekly_tail_pass"] = False
        frame["pragmatic_weekly_tail_pass"] = False

    required = [
        "long_period_months_pass",
        "long_period_heads_pass",
        "long_period_trades_pass",
        "month_consistency_pass",
        "head_consistency_pass",
        "pnl_objective_pass",
        "full_sl_pass",
    ]
    frame["long_period_core_pass"] = frame[required].all(axis=1)
    frame["long_period_strict_tail_pass"] = frame["long_period_core_pass"] & frame["all_weekly_tail_pass"]
    frame["long_period_pragmatic_tail_pass"] = frame["long_period_core_pass"] & frame["pragmatic_weekly_tail_pass"]
    frame["long_period_score"] = (
        frame.get("delta_vs_baseline_net_pnl", pd.Series(0.0, index=frame.index)).fillna(0.0)
        + 1000.0
        * frame.get(
            "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
            pd.Series(0.0, index=frame.index),
        ).fillna(0.0)
        + 250.0 * frame["tail_nonnegative_count"].fillna(0.0)
        - 1000.0 * frame.get("delta_vs_baseline_full_sl_rate", pd.Series(0.0, index=frame.index)).fillna(0.0)
    )
    frame["long_period_decision"] = np.select(
        [
            frame["long_period_strict_tail_pass"],
            frame["long_period_pragmatic_tail_pass"],
            frame["long_period_core_pass"],
            frame["pnl_objective_pass"],
        ],
        [
            "strict_pnl_tail_long_period_research_champion",
            "pragmatic_long_period_research_candidate_q10_warning",
            "long_period_research_champion_tail_warning",
            "diagnostic_candidate_needs_consistency",
        ],
        default="reject_or_diagnostic",
    )
    keep = [
        "variant",
        "role",
        "long_period_decision",
        "long_period_score",
        "long_period_core_pass",
        "long_period_strict_tail_pass",
        "long_period_pragmatic_tail_pass",
        "months",
        "heads",
        "trade_count",
        "positive_net_month_share",
        "positive_net_head_share",
        "min_month_delta_net_pnl",
        "min_head_delta_net_pnl",
        "delta_vs_baseline_net_pnl",
        "delta_vs_baseline_hit_rate",
        "delta_vs_baseline_full_sl_rate",
        "delta_vs_baseline_timeout_rate",
        "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
        "tail_nonnegative_count",
        "tail_tested_count",
        "worst_weekly_tail_delta",
        "delta_vs_baseline_weekly_q05_pnl",
        "delta_vs_baseline_weekly_q10_pnl",
        "delta_vs_baseline_weekly_q20_pnl",
        "delta_vs_baseline_weekly_q35_pnl",
        "jaccard_vs_baseline",
        "candidate_status",
    ]
    return frame[[col for col in keep if col in frame.columns]].sort_values(
        ["long_period_strict_tail_pass", "long_period_pragmatic_tail_pass", "long_period_core_pass", "long_period_score"],
        ascending=[False, False, False, False],
    )


def _tail_repair_frontier_from_existing_grids(report_root: Path, *, max_rows_per_source: int = 8) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidate_files = [
        report_root / "contextual_tp_sl_current_candidate_promotion_table_v1_20260701/candidate_promotion_summary.csv",
        report_root / "contextual_tp_sl_ab_objective_tradeoff_20260630/ab_tail_tradeoff_summary.csv",
    ]
    candidate_files.extend(sorted(report_root.glob("contextual_tp_sl_combo_sweep*/head_arm_combination_summary.csv")))

    for path in candidate_files:
        if not path.exists():
            continue
        frame = _read_csv(path)
        if frame.empty:
            continue
        id_col = "variant" if "variant" in frame.columns else ("combo_id" if "combo_id" in frame.columns else "arm")
        if id_col not in frame.columns or "net_pnl" not in frame.columns:
            continue
        if not any(col in frame.columns for col in ("weekly_q10_pnl", "delta_vs_baseline_weekly_q10_pnl")):
            continue
        work = frame.copy()
        for col in (
            "net_pnl",
            "trade_count",
            "full_sl_rate",
            "timeout_rate",
            "max_drawdown",
            "objective",
            "balanced_score",
            "avg_week_pnl",
            "objective_avgweek_0p7dayq35_0p3dayq20",
            "weekly_q05_pnl",
            "weekly_q10_pnl",
            "weekly_q20_pnl",
            "weekly_q35_pnl",
            "daily_q10_pnl",
            "daily_q20_pnl",
            "daily_q35_pnl",
            "delta_vs_baseline_net_pnl",
            "delta_vs_baseline_full_sl_rate",
            "delta_vs_baseline_weekly_q05_pnl",
            "delta_vs_baseline_weekly_q10_pnl",
            "delta_vs_baseline_weekly_q20_pnl",
            "delta_vs_baseline_weekly_q35_pnl",
        ):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
        if "variant" in work.columns:
            work = work[~work["variant"].astype(str).str.lower().eq("baseline")].copy()
        if work.empty:
            continue

        if "delta_vs_baseline_weekly_q10_pnl" in work.columns:
            q10 = work["delta_vs_baseline_weekly_q10_pnl"]
            q05 = work.get("delta_vs_baseline_weekly_q05_pnl", pd.Series(np.nan, index=work.index))
            q20 = work.get("delta_vs_baseline_weekly_q20_pnl", pd.Series(np.nan, index=work.index))
            q35 = work.get("delta_vs_baseline_weekly_q35_pnl", pd.Series(np.nan, index=work.index))
            pnl = work.get("delta_vs_baseline_net_pnl", pd.Series(np.nan, index=work.index))
            evidence_basis = "delta_vs_baseline"
        else:
            q10 = work.get("weekly_q10_pnl", pd.Series(np.nan, index=work.index))
            q05 = work.get("weekly_q05_pnl", pd.Series(np.nan, index=work.index))
            q20 = work.get("weekly_q20_pnl", pd.Series(np.nan, index=work.index))
            q35 = work.get("weekly_q35_pnl", pd.Series(np.nan, index=work.index))
            pnl = work.get("net_pnl", pd.Series(np.nan, index=work.index))
            evidence_basis = "absolute_not_baseline_aligned"
        weekly_tail = pd.concat([q05, q10, q20, q35], axis=1)
        work["_tail_tested_count"] = weekly_tail.notna().sum(axis=1)
        work["_tail_nonnegative_count"] = weekly_tail.ge(0.0).sum(axis=1)
        work["_strict_weekly_tail_positive"] = work["_tail_tested_count"].gt(0) & work[
            "_tail_nonnegative_count"
        ].eq(work["_tail_tested_count"])
        work["_q10_positive"] = q10.ge(0.0)
        work["_frontier_score"] = (
            pnl.fillna(0.0)
            + 500.0 * q10.fillna(0.0)
            + 100.0 * q05.fillna(0.0)
            + 100.0 * q20.fillna(0.0)
            - 100000.0 * work.get("full_sl_rate", pd.Series(0.0, index=work.index)).fillna(0.0)
        )
        selected = pd.concat(
            [
                work.loc[work["_q10_positive"]].sort_values(
                    ["_strict_weekly_tail_positive", "_frontier_score", "net_pnl"],
                    ascending=[False, False, False],
                ).head(max_rows_per_source),
                work.sort_values(["_frontier_score", "net_pnl"], ascending=[False, False]).head(3),
            ],
            ignore_index=False,
        ).drop_duplicates()
        for _, row in selected.iterrows():
            rows.append(
                {
                    "source_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                    "source_name": path.parent.name,
                    "evidence_basis": evidence_basis,
                    "candidate_id": str(row.get(id_col) or ""),
                    "candidate_start": row.get("candidate_start", ""),
                    "candidate_end": row.get("candidate_end", ""),
                    "trade_count": row.get("trade_count"),
                    "net_pnl": row.get("net_pnl"),
                    "delta_vs_baseline_net_pnl": row.get("delta_vs_baseline_net_pnl"),
                    "full_sl_rate": row.get("full_sl_rate"),
                    "delta_vs_baseline_full_sl_rate": row.get("delta_vs_baseline_full_sl_rate"),
                    "timeout_rate": row.get("timeout_rate"),
                    "max_drawdown": row.get("max_drawdown"),
                    "objective": row.get("objective"),
                    "balanced_score": row.get("balanced_score"),
                    "avg_week_pnl": row.get("avg_week_pnl"),
                    "objective_avgweek_0p7dayq35_0p3dayq20": row.get(
                        "objective_avgweek_0p7dayq35_0p3dayq20"
                    ),
                    "weekly_q05_pnl": row.get("weekly_q05_pnl"),
                    "weekly_q10_pnl": row.get("weekly_q10_pnl"),
                    "weekly_q20_pnl": row.get("weekly_q20_pnl"),
                    "weekly_q35_pnl": row.get("weekly_q35_pnl"),
                    "delta_vs_baseline_weekly_q05_pnl": row.get("delta_vs_baseline_weekly_q05_pnl"),
                    "delta_vs_baseline_weekly_q10_pnl": row.get("delta_vs_baseline_weekly_q10_pnl"),
                    "delta_vs_baseline_weekly_q20_pnl": row.get("delta_vs_baseline_weekly_q20_pnl"),
                    "delta_vs_baseline_weekly_q35_pnl": row.get("delta_vs_baseline_weekly_q35_pnl"),
                    "daily_q10_pnl": row.get("daily_q10_pnl"),
                    "daily_q20_pnl": row.get("daily_q20_pnl"),
                    "daily_q35_pnl": row.get("daily_q35_pnl"),
                    "q10_positive": bool(row.get("_q10_positive")),
                    "strict_weekly_tail_positive": bool(row.get("_strict_weekly_tail_positive")),
                    "tail_nonnegative_count": int(row.get("_tail_nonnegative_count") or 0),
                    "tail_tested_count": int(row.get("_tail_tested_count") or 0),
                    "frontier_score": row.get("_frontier_score"),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    dedupe_cols = [
        col
        for col in (
            "evidence_basis",
            "candidate_id",
            "candidate_start",
            "candidate_end",
            "net_pnl",
            "weekly_q10_pnl",
            "delta_vs_baseline_weekly_q10_pnl",
        )
        if col in out.columns
    ]
    if dedupe_cols:
        if "objective_avgweek_0p7dayq35_0p3dayq20" in out.columns:
            out["_has_requested_objective"] = pd.to_numeric(
                out["objective_avgweek_0p7dayq35_0p3dayq20"],
                errors="coerce",
            ).notna()
            out = out.sort_values(
                ["_has_requested_objective", "frontier_score", "net_pnl"],
                ascending=[False, False, False],
                na_position="last",
            )
        out = out.drop_duplicates(subset=dedupe_cols).copy()
        out = out.drop(columns=["_has_requested_objective"], errors="ignore")
    for col in out.columns:
        if col not in {"source_path", "source_name", "evidence_basis", "candidate_id", "candidate_start", "candidate_end"}:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().any() or out[col].isna().all():
                out[col] = converted
    return out.sort_values(
        ["strict_weekly_tail_positive", "q10_positive", "frontier_score", "net_pnl"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def _tail_frontier_rerun_shortlist(
    frontier: pd.DataFrame,
    report_root: Path,
    output_dir: Path,
    *,
    top_n: int = 8,
) -> pd.DataFrame:
    if frontier.empty:
        return pd.DataFrame()
    work = frontier.loc[
        frontier.get("strict_weekly_tail_positive", pd.Series(False, index=frontier.index)).fillna(False)
        & frontier["evidence_basis"].astype(str).eq("absolute_not_baseline_aligned")
    ].copy()
    if work.empty:
        return pd.DataFrame()
    for col in ("frontier_score", "net_pnl", "weekly_q10_pnl", "full_sl_rate"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.sort_values(
        ["frontier_score", "net_pnl", "weekly_q10_pnl"],
        ascending=[False, False, False],
        na_position="last",
    ).head(top_n)

    rows: list[dict[str, Any]] = []
    combo_dir = output_dir / "tail_frontier_combo_files"
    combo_dir.mkdir(parents=True, exist_ok=True)
    for source_name, group in work.groupby("source_name", sort=False):
        source_report_dir = report_root / str(source_name)
        manifest = _read_json(source_report_dir / "head_arm_combination_summary.json")
        source_dir = str(manifest.get("source_dir") or "")
        combo_file = combo_dir / f"{source_name}_combo_ids.csv"
        pd.DataFrame({"combo_id": group["candidate_id"].dropna().astype(str).drop_duplicates()}).to_csv(
            combo_file,
            index=False,
        )
        suggested_out_dir = output_dir / f"tail_frontier_subset_replay_{source_name}"
        combo_file_display = _display_path(combo_file)
        suggested_out_dir_display = _display_path(suggested_out_dir)
        rerun_command = (
            "python3 scripts/sweep_contextual_tp_sl_arm_combinations.py "
            f"--source-dir {source_dir} "
            f"--out-dir {suggested_out_dir_display} "
            "--market-mode perps "
            f"--combo-file {combo_file_display}"
        )
        for _, row in group.iterrows():
            rows.append(
                {
                    "rerun_priority": int(len(rows) + 1),
                    "source_name": source_name,
                    "source_dir": source_dir,
                    "candidate_id": row.get("candidate_id"),
                    "combo_file": combo_file_display,
                    "suggested_out_dir": suggested_out_dir_display,
                    "rerun_command": rerun_command,
                    "candidate_start": row.get("candidate_start"),
                    "candidate_end": row.get("candidate_end"),
                    "trade_count": row.get("trade_count"),
                    "net_pnl": row.get("net_pnl"),
                    "weekly_q05_pnl": row.get("weekly_q05_pnl"),
                    "weekly_q10_pnl": row.get("weekly_q10_pnl"),
                    "weekly_q20_pnl": row.get("weekly_q20_pnl"),
                    "weekly_q35_pnl": row.get("weekly_q35_pnl"),
                    "daily_q10_pnl": row.get("daily_q10_pnl"),
                    "full_sl_rate": row.get("full_sl_rate"),
                    "timeout_rate": row.get("timeout_rate"),
                    "max_drawdown": row.get("max_drawdown"),
                    "frontier_score": row.get("frontier_score"),
                    "strict_weekly_tail_positive": row.get("strict_weekly_tail_positive"),
                    "evidence_note": "subset replay lead; rerun under matched source and compare against source static/baseline before promotion",
                }
            )
    return pd.DataFrame(rows)


def _consistency_breakdowns(dev_dashboard_dir: Path, best_dev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if best_dev.empty or "variant" not in best_dev.columns:
        return pd.DataFrame(), pd.DataFrame()
    variants = set(best_dev["variant"].astype(str))
    monthly = _read_csv(dev_dashboard_dir / "candidate_monthly_consistency.csv")
    head = _read_csv(dev_dashboard_dir / "candidate_head_consistency.csv")
    if not monthly.empty and "variant" in monthly.columns:
        monthly = monthly[monthly["variant"].astype(str).isin(variants)].copy()
        for col in monthly.columns:
            if col != "variant":
                monthly[col] = pd.to_numeric(monthly[col], errors="coerce")
        monthly = monthly.sort_values(
            ["positive_net_month_share", "min_month_delta_net_pnl", "mean_month_delta_net_pnl"],
            ascending=[False, False, False],
        )
    if not head.empty and "variant" in head.columns:
        head = head[head["variant"].astype(str).isin(variants)].copy()
        for col in head.columns:
            if col != "variant":
                head[col] = pd.to_numeric(head[col], errors="coerce")
        head = head.sort_values(
            ["positive_net_head_share", "min_head_delta_net_pnl", "mean_head_delta_net_pnl"],
            ascending=[False, False, False],
        )
    return monthly, head


def _readiness_summary(workflow_dir: Path) -> dict[str, Any]:
    readiness = _read_json(workflow_dir / "readiness/latest_flat_frozen_gate_readiness.json")
    source = readiness.get("selected_source") or readiness.get("nearest_source") or {}
    req = readiness.get("requirements") or {}
    return {
        "ready_sources": int(readiness.get("ready_sources") or 0),
        "ran_gate": bool(readiness.get("ran_gate")),
        "rejection_reasons": str(source.get("rejection_reasons") or ""),
        "post_cutoff_rows": int(source.get("post_cutoff_rows") or 0),
        "post_cutoff_rows_required": int(req.get("min_post_cutoff_rows") or 0),
        "post_cutoff_timestamps": int(source.get("post_cutoff_timestamps") or 0),
        "post_cutoff_timestamps_required": int(req.get("min_post_cutoff_timestamps") or 0),
        "policy_action_rows": int(source.get("policy_action_rows_estimate") or 0),
        "policy_action_rows_required": int(req.get("min_policy_action_rows") or 0),
        "policy_outcome_rows": int(source.get("policy_outcome_rows_estimate") or 0),
        "policy_outcome_rows_required": int(req.get("min_policy_outcome_rows") or 0),
        "policy_action_head_counts": str(source.get("policy_action_head_counts") or "{}"),
        "policy_outcome_head_counts": str(source.get("policy_outcome_head_counts") or "{}"),
        "low_required_head_counts": str(source.get("policy_outcome_low_required_head_counts") or "{}"),
        "uncertainty_finite_row_rate": float(source.get("uncertainty_finite_row_rate") or 0.0),
        "drift_finite_row_rate": float(source.get("drift_finite_row_rate") or 0.0),
        "ood_finite_row_rate": float(source.get("ood_finite_row_rate") or 0.0),
        "recent_hit_rate_surprise_finite_row_rate": float(
            source.get("recent_hit_rate_surprise_finite_row_rate") or 0.0
        ),
    }


def _evidence_gap_frames(workflow_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    readiness = _read_json(workflow_dir / "readiness/latest_flat_frozen_gate_readiness.json")
    source = readiness.get("selected_source") or readiness.get("nearest_source") or {}
    req = readiness.get("requirements") or {}
    metric_specs = [
        ("post_cutoff_rows", "post_cutoff_rows", "min_post_cutoff_rows"),
        ("post_cutoff_timestamps", "post_cutoff_timestamps", "min_post_cutoff_timestamps"),
        ("post_cutoff_active_heads", "post_cutoff_active_heads", "min_post_cutoff_active_heads"),
        ("policy_action_rows", "policy_action_rows_estimate", "min_policy_action_rows"),
        ("policy_action_timestamps", "policy_action_timestamps_estimate", "min_policy_action_timestamps"),
        ("policy_outcome_rows", "policy_outcome_rows_estimate", "min_policy_outcome_rows"),
        ("policy_outcome_timestamps", "policy_outcome_timestamps_estimate", "min_policy_outcome_timestamps"),
    ]
    gap_rows: list[dict[str, Any]] = []
    for gate, observed_key, required_key in metric_specs:
        observed = int(source.get(observed_key) or 0)
        required = int(req.get(required_key) or 0)
        gap_rows.append(
            {
                "gate": gate,
                "observed": observed,
                "required": required,
                "deficit": max(required - observed, 0),
                "pass": observed >= required,
            }
        )

    action_counts = _json_dict(source.get("policy_action_head_counts"))
    outcome_counts = _json_dict(source.get("policy_outcome_head_counts"))
    required_heads = set(req.get("required_policy_outcome_head") or [])
    min_action_outcomes = int(req.get("min_policy_outcome_rows_per_action_head") or 0)
    min_required_outcomes = int(req.get("min_policy_outcome_rows_per_required_head") or 0)
    heads = sorted(set(action_counts) | set(outcome_counts) | required_heads)
    head_rows: list[dict[str, Any]] = []
    for head in heads:
        action_rows = int(action_counts.get(head) or 0)
        matured_rows = int(outcome_counts.get(head) or 0)
        required_matured = 0
        if head in required_heads:
            required_matured = max(required_matured, min_required_outcomes)
        if action_rows > 0:
            required_matured = max(required_matured, min_action_outcomes)
        deficit = max(required_matured - matured_rows, 0)
        if deficit <= 0:
            status = "ready"
        elif action_rows <= 0:
            status = "needs_policy_action_and_matured_outcomes"
        else:
            status = "needs_matured_outcomes"
        head_rows.append(
            {
                "head": head,
                "policy_action_rows": action_rows,
                "matured_outcome_rows": matured_rows,
                "required_matured_outcomes": required_matured,
                "matured_outcome_deficit": deficit,
                "required_head": head in required_heads,
                "status": status,
            }
        )
    gap = pd.DataFrame(gap_rows)
    head_gap = pd.DataFrame(head_rows)
    if not head_gap.empty:
        head_gap = head_gap.sort_values(
            ["matured_outcome_deficit", "required_head", "policy_action_rows"],
            ascending=[False, False, True],
        )
    return gap, head_gap


def _head_action_opportunity_frame(workflow_dir: Path) -> pd.DataFrame:
    readiness = _read_json(workflow_dir / "readiness/latest_flat_frozen_gate_readiness.json")
    source = readiness.get("selected_source") or readiness.get("nearest_source") or {}
    path_value = source.get("path")
    if not path_value:
        return pd.DataFrame()
    ledger_path = Path(path_value)
    if not ledger_path.is_absolute():
        ledger_path = ROOT / ledger_path
    if not ledger_path.exists():
        return pd.DataFrame()
    cutoff = pd.Timestamp(readiness.get("cutoff") or "1970-01-01T00:00:00Z")
    requested_columns = [
        "timestamp",
        "head",
        "strategy_id",
        "auction_rank_score",
        "adjusted_rank_score",
        "rank_pct",
        "portfolio_decision",
        "was_traded",
        "net_return",
        "live_replay_net_return",
    ]
    try:
        frame = pd.read_parquet(ledger_path, columns=requested_columns)
    except Exception:
        try:
            frame = pd.read_parquet(ledger_path)
        except Exception:
            return pd.DataFrame()
    if frame.empty or "timestamp" not in frame.columns:
        return pd.DataFrame()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame[frame["timestamp"].ge(cutoff)].copy()
    if frame.empty:
        return pd.DataFrame()
    if "head" not in frame.columns:
        frame["head"] = ""
    if "strategy_id" not in frame.columns:
        frame["strategy_id"] = ""
    frame["canonical_head"] = _canonical_head_from_series(frame["head"], frame["strategy_id"])

    def numeric_column(name: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype="float64")
        return pd.to_numeric(frame[name], errors="coerce")

    auction = numeric_column("auction_rank_score")
    adjusted = numeric_column("adjusted_rank_score")
    rank_pct = numeric_column("rank_pct")
    net_return = numeric_column("net_return")
    live_return = numeric_column("live_replay_net_return")
    decision = frame.get("portfolio_decision", pd.Series("", index=frame.index)).fillna("").astype(str)
    was_traded = frame.get("was_traded", pd.Series(False, index=frame.index)).astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    decision_traded = decision.str.lower().eq("traded")
    finite_outcome = net_return.notna() | live_return.notna()
    rows: list[dict[str, Any]] = []
    for head, group in frame.groupby("canonical_head", dropna=False):
        idx = group.index
        decisions = decision.loc[idx].replace("", "missing").value_counts(dropna=False).to_dict()
        finite_auction = int(auction.loc[idx].notna().sum())
        finite_adjusted = int(adjusted.loc[idx].notna().sum())
        traded_rows = int((was_traded.loc[idx] | decision_traded.loc[idx]).sum())
        rank_rejected = int(decision.loc[idx].str.lower().eq("rank_rejected").sum())
        portfolio_rejected = int(decision.loc[idx].str.lower().eq("portfolio_rejected").sum())
        missing_decision = int(decision.loc[idx].eq("").sum())
        matured_rows = int(finite_outcome.loc[idx].sum())
        if traded_rows > 0:
            blocker = "has_policy_actions"
        elif finite_auction <= 0 and finite_adjusted <= 0:
            blocker = "missing_live_rank_reference_or_not_scored"
        elif rank_rejected > 0 and portfolio_rejected <= 0:
            blocker = "rank_rejected_below_policy_threshold"
        elif portfolio_rejected > 0:
            blocker = "portfolio_capacity_or_conflict_rejected"
        else:
            blocker = "no_policy_action_observed"
        rows.append(
            {
                "head": str(head),
                "candidate_rows": int(len(group)),
                "timestamps": int(group["timestamp"].nunique()),
                "finite_auction_rank_rows": finite_auction,
                "finite_adjusted_rank_rows": finite_adjusted,
                "rank_pct_rows": int(rank_pct.loc[idx].notna().sum()),
                "policy_action_rows": traded_rows,
                "rank_rejected_rows": rank_rejected,
                "portfolio_rejected_rows": portfolio_rejected,
                "missing_decision_rows": missing_decision,
                "candidate_outcome_rows": matured_rows,
                "auction_rank_max": float(auction.loc[idx].max()) if auction.loc[idx].notna().any() else np.nan,
                "adjusted_rank_max": float(adjusted.loc[idx].max()) if adjusted.loc[idx].notna().any() else np.nan,
                "rank_pct_max": float(rank_pct.loc[idx].max()) if rank_pct.loc[idx].notna().any() else np.nan,
                "decision_counts": json.dumps({str(k): int(v) for k, v in decisions.items()}, sort_keys=True),
                "action_blocker_hint": blocker,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["policy_action_rows", "candidate_rows"],
        ascending=[True, False],
    )


def _live_health_summary(workflow_dir: Path) -> dict[str, Any]:
    manifest = _read_json(workflow_dir / "live_outcomes/live_prediction_ledger_outcome_manifest.json")
    return {
        "prediction_rows": int(manifest.get("prediction_rows") or 0),
        "trade_log_rows": int(manifest.get("trade_log_rows") or 0),
        "traded_rows": int(manifest.get("traded_rows") or 0),
        "realized_traded_rows": int(manifest.get("realized_traded_rows") or 0),
        "unresolved_traded_rows": int(manifest.get("unresolved_traded_rows") or 0),
        "prediction_timestamp_max": str(manifest.get("timestamp_max") or ""),
        "trade_log_timestamp_max": str(manifest.get("trade_log_timestamp_max") or ""),
        "prediction_to_trade_log_lag_minutes": float(
            manifest.get("prediction_to_trade_log_lag_minutes") or 0.0
        ),
        "prediction_ledger_stale_vs_trade_log": bool(
            manifest.get("prediction_ledger_stale_vs_trade_log")
        ),
    }


def _eligible_gate_summary(workflow_dir: Path) -> pd.DataFrame:
    path = workflow_dir / "eligible_head_gate/frozen_dual_scoring_gate_summary.csv"
    frame = _read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    keep = [
        "bundle",
        "tested_feature_families",
        "baseline_trade_count",
        "best_delta_pnl_variant",
        "best_delta_net_pnl",
        "best_delta_full_sl_rate",
        "max_adjusted_rows",
        "max_adjusted_share",
        "min_accepted_jaccard",
        "total_entrants",
        "total_removed",
        "max_adjusted_acceptance_changed",
        "promotion_ready",
        "failed_checks",
    ]
    return frame[[col for col in keep if col in frame.columns]].copy()


def _family_attribution_from_scorecards(scorecard_dir: Path, family_effect_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    promotion = _read_csv(scorecard_dir / "promotion_scorecard.csv")
    if not promotion.empty and {"variant", "family"}.issubset(promotion.columns):
        frame = promotion[~promotion["variant"].astype(str).eq("baseline")].copy()
        for col in (
            "delta_vs_baseline_net_pnl",
            "delta_vs_baseline_full_sl_rate",
            "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
            "weekly_q05_pnl",
            "weekly_q10_pnl",
            "weekly_q20_pnl",
            "scorecard_score",
        ):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if "scorecard_score" in frame.columns:
            frame = frame.sort_values("scorecard_score", ascending=False)
        for _, row in frame.head(5).iterrows():
            rows.append(
                {
                    "source": "promotion_scorecard",
                    "family": str(row.get("family") or ""),
                    "best_variant": str(row.get("variant") or ""),
                    "best_head": "",
                    "delta_net_pnl": row.get("delta_vs_baseline_net_pnl"),
                    "delta_objective": row.get(
                        "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20"
                    ),
                    "delta_tail_objective": np.nan,
                    "delta_full_sl_rate": row.get("delta_vs_baseline_full_sl_rate"),
                    "delta_q20_pnl": row.get("weekly_q20_pnl"),
                    "delta_q35_pnl": np.nan,
                    "positive_head_count": np.nan,
                    "tested_head_count": np.nan,
                    "readout": "long_window_candidate",
                }
            )

    expanding = _read_csv(scorecard_dir / "expanding_family_scorecard.csv")
    if not expanding.empty and {"family", "variant"}.issubset(expanding.columns):
        frame = expanding.copy()
        for col in (
            "delta_net_pnl",
            "delta_objective_week",
            "delta_full_sl_rate",
            "delta_q20_week_net_pnl",
            "delta_q35_week_net_pnl",
            "scorecard_score",
        ):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if "scorecard_score" in frame.columns:
            idx = frame.groupby("family")["scorecard_score"].idxmax()
        else:
            idx = frame.groupby("family")["delta_net_pnl"].idxmax()
        for _, row in frame.loc[idx].sort_values("scorecard_score", ascending=False).iterrows():
            rows.append(
                {
                    "source": "expanding_family_scorecard",
                    "family": str(row.get("family") or ""),
                    "best_variant": str(row.get("variant") or ""),
                    "best_head": "",
                    "delta_net_pnl": row.get("delta_net_pnl"),
                    "delta_objective": row.get("delta_objective_week"),
                    "delta_tail_objective": np.nan,
                    "delta_full_sl_rate": row.get("delta_full_sl_rate"),
                    "delta_q20_pnl": row.get("delta_q20_week_net_pnl"),
                    "delta_q35_pnl": row.get("delta_q35_week_net_pnl"),
                    "positive_head_count": np.nan,
                    "tested_head_count": np.nan,
                    "readout": "expanding_walk_forward",
                }
            )

    by_head = _read_csv(family_effect_dir / "best_by_head_family.csv")
    if not by_head.empty and {"head", "diagnostic_family", "label"}.issubset(by_head.columns):
        frame = by_head.copy()
        for col in ("delta_tail_objective", "delta_q20_day_net_pnl", "delta_q35_day_net_pnl"):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        for family, group in frame.groupby("diagnostic_family", dropna=False):
            best_idx = group["delta_tail_objective"].idxmax()
            best = group.loc[best_idx]
            rows.append(
                {
                    "source": "head_family_tail_effect",
                    "family": str(family or ""),
                    "best_variant": str(best.get("label") or ""),
                    "best_head": str(best.get("head") or ""),
                    "delta_net_pnl": np.nan,
                    "delta_objective": np.nan,
                    "delta_tail_objective": best.get("delta_tail_objective"),
                    "delta_full_sl_rate": np.nan,
                    "delta_q20_pnl": best.get("delta_q20_day_net_pnl"),
                    "delta_q35_pnl": best.get("delta_q35_day_net_pnl"),
                    "positive_head_count": int((group["delta_tail_objective"] > 0).sum()),
                    "tested_head_count": int(group["head"].nunique()),
                    "readout": "per_head_standalone_tail",
                }
            )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    numeric_cols = [
        "delta_net_pnl",
        "delta_objective",
        "delta_tail_objective",
        "delta_full_sl_rate",
        "delta_q20_pnl",
        "delta_q35_pnl",
        "positive_head_count",
        "tested_head_count",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    source_rank = {
        "promotion_scorecard": 0,
        "expanding_family_scorecard": 1,
        "head_family_tail_effect": 2,
    }
    out["_source_rank"] = out["source"].map(source_rank).fillna(99)
    out = out.sort_values(
        ["_source_rank", "delta_net_pnl", "delta_tail_objective", "delta_objective"],
        ascending=[True, False, False, False],
        na_position="last",
    )
    return out.drop(columns=["_source_rank"]).reset_index(drop=True)


REQUESTED_RELIABILITY_FAMILIES = {
    "drift": ("drift",),
    "recent_hit_rate_surprise": ("recent_hr", "recent_hit_rate", "recent_perf", "hr_surprise"),
    "ood": ("ood",),
    "uncertainty": ("uncertainty",),
}


def _family_name_contains(family: Any, needles: tuple[str, ...]) -> bool:
    text = str(family or "").lower()
    return any(needle in text for needle in needles)


def _requested_reliability_family_verdict(
    family_attribution: pd.DataFrame,
    readiness: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tested_families: set[str] = set()
    if not family_attribution.empty and "family" in family_attribution.columns:
        tested_families = {str(value).lower() for value in family_attribution["family"].dropna().astype(str)}

    for family, needles in REQUESTED_RELIABILITY_FAMILIES.items():
        matches = (
            family_attribution[
                family_attribution["family"].map(lambda value: _family_name_contains(value, needles))
            ].copy()
            if not family_attribution.empty and "family" in family_attribution.columns
            else pd.DataFrame()
        )
        finite_row_rate = float(readiness.get(f"{family}_finite_row_rate") or 0.0)
        columns_present_raw = readiness.get(f"{family}_columns_present")
        columns_required_raw = readiness.get(f"{family}_columns_required")
        columns_present = (
            int(columns_present_raw) if columns_present_raw is not None and str(columns_present_raw) != "" else np.nan
        )
        columns_required = (
            int(columns_required_raw) if columns_required_raw is not None and str(columns_required_raw) != "" else np.nan
        )
        if not matches.empty:
            for col in (
                "delta_net_pnl",
                "delta_objective",
                "delta_tail_objective",
                "delta_full_sl_rate",
                "delta_q20_pnl",
                "delta_q35_pnl",
                "positive_head_count",
                "tested_head_count",
            ):
                if col in matches.columns:
                    matches[col] = pd.to_numeric(matches[col], errors="coerce")
        best_long_pnl = (
            float(matches["delta_net_pnl"].max(skipna=True))
            if not matches.empty and "delta_net_pnl" in matches.columns and matches["delta_net_pnl"].notna().any()
            else np.nan
        )
        best_tail_objective = (
            float(matches["delta_tail_objective"].max(skipna=True))
            if not matches.empty
            and "delta_tail_objective" in matches.columns
            and matches["delta_tail_objective"].notna().any()
            else np.nan
        )
        best_q20 = (
            float(matches["delta_q20_pnl"].max(skipna=True))
            if not matches.empty and "delta_q20_pnl" in matches.columns and matches["delta_q20_pnl"].notna().any()
            else np.nan
        )
        positive_head_count = (
            int(matches["positive_head_count"].max(skipna=True))
            if not matches.empty
            and "positive_head_count" in matches.columns
            and matches["positive_head_count"].notna().any()
            else 0
        )
        tested_head_count = (
            int(matches["tested_head_count"].max(skipna=True))
            if not matches.empty
            and "tested_head_count" in matches.columns
            and matches["tested_head_count"].notna().any()
            else 0
        )
        exact_or_composite_tested = any(_family_name_contains(value, needles) for value in tested_families)
        helped_long_window = bool(np.isfinite(best_long_pnl) and best_long_pnl > 0.0)
        helped_tail = bool(
            (np.isfinite(best_tail_objective) and best_tail_objective > 0.0)
            or (np.isfinite(best_q20) and best_q20 > 0.0)
            or positive_head_count > 0
        )
        missing_contract = (
            np.isfinite(columns_required)
            and np.isfinite(columns_present)
            and columns_required > 0
            and columns_present <= 0
            and finite_row_rate <= 0.0
        )
        if missing_contract:
            verdict = "missing_from_candidate_contract"
        elif finite_row_rate < 0.25:
            verdict = "present_but_low_finite_coverage"
        elif helped_long_window or helped_tail:
            verdict = "helpful_in_tests"
        elif exact_or_composite_tested:
            verdict = "tested_no_clear_lift"
        else:
            verdict = "present_not_yet_tested"
        rows.append(
            {
                "family": family,
                "columns_present": columns_present,
                "columns_required": columns_required,
                "finite_row_rate": finite_row_rate,
                "tested_in_scorecards": bool(exact_or_composite_tested),
                "best_long_window_delta_net_pnl": best_long_pnl,
                "best_tail_objective_delta": best_tail_objective,
                "best_q20_delta_pnl": best_q20,
                "positive_head_count": positive_head_count,
                "tested_head_count": tested_head_count,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def _family_flags(family: Any) -> dict[str, bool]:
    text = str(family or "").lower()
    return {
        "has_recent_hit_rate_surprise": any(
            token in text for token in REQUESTED_RELIABILITY_FAMILIES["recent_hit_rate_surprise"]
        ),
        "has_drift": _family_name_contains(text, REQUESTED_RELIABILITY_FAMILIES["drift"]),
        "has_ood": _family_name_contains(text, REQUESTED_RELIABILITY_FAMILIES["ood"]),
        "has_uncertainty": _family_name_contains(text, REQUESTED_RELIABILITY_FAMILIES["uncertainty"]),
        "has_weekly_head_gate": "weekly_head_gate" in text,
    }


def _scorecard_rows_for_marginal_ablation(scorecard_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    promotion = _read_csv(scorecard_dir / "promotion_scorecard.csv")
    if not promotion.empty and {"variant", "family"}.issubset(promotion.columns):
        frame = promotion.copy()
        for _, row in frame.iterrows():
            rows.append(
                {
                    "source_table": "promotion_scorecard",
                    "evidence_family": "promotion_scorecard",
                    "variant": str(row.get("variant") or ""),
                    "family": str(row.get("family") or ""),
                    "delta_net_pnl": row.get("delta_vs_baseline_net_pnl"),
                    "delta_objective": row.get(
                        "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20"
                    ),
                    "delta_full_sl_rate": row.get("delta_vs_baseline_full_sl_rate"),
                    "delta_q20_pnl": row.get("weekly_q20_pnl"),
                    "delta_q35_pnl": np.nan,
                    "scorecard_score": row.get("scorecard_score"),
                }
            )

    expanding = _read_csv(scorecard_dir / "expanding_family_scorecard.csv")
    if not expanding.empty and {"variant", "family"}.issubset(expanding.columns):
        for _, row in expanding.iterrows():
            rows.append(
                {
                    "source_table": "expanding_family_scorecard",
                    "evidence_family": str(row.get("evidence_family") or "expanding_family_scorecard"),
                    "variant": str(row.get("variant") or ""),
                    "family": str(row.get("family") or ""),
                    "delta_net_pnl": row.get("delta_net_pnl"),
                    "delta_objective": row.get("delta_objective_week"),
                    "delta_full_sl_rate": row.get("delta_full_sl_rate"),
                    "delta_q20_pnl": row.get("delta_q20_week_net_pnl"),
                    "delta_q35_pnl": row.get("delta_q35_week_net_pnl"),
                    "scorecard_score": row.get("scorecard_score"),
                }
            )

    tailgrid = _read_csv(scorecard_dir / "tailgrid_recent_hr_scorecard.csv")
    if not tailgrid.empty and {"variant", "family"}.issubset(tailgrid.columns):
        for _, row in tailgrid.iterrows():
            rows.append(
                {
                    "source_table": "tailgrid_recent_hr_scorecard",
                    "evidence_family": str(row.get("evidence_family") or "tailgrid_recent_hr_scorecard"),
                    "variant": str(row.get("variant") or ""),
                    "family": str(row.get("family") or ""),
                    "delta_net_pnl": row.get("delta_net_pnl"),
                    "delta_objective": row.get("tail_objective_delta"),
                    "delta_full_sl_rate": row.get("delta_full_sl_rate"),
                    "delta_q20_pnl": row.get("delta_q20"),
                    "delta_q35_pnl": row.get("delta_q35"),
                    "scorecard_score": row.get("scorecard_score"),
                }
            )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    for col in (
        "delta_net_pnl",
        "delta_objective",
        "delta_full_sl_rate",
        "delta_q20_pnl",
        "delta_q35_pnl",
        "scorecard_score",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    flag_frame = pd.DataFrame([_family_flags(value) for value in out["family"]], index=out.index)
    out = pd.concat([out, flag_frame], axis=1)
    out["is_baseline"] = out["family"].astype(str).str.lower().eq("baseline_or_other") | out[
        "variant"
    ].astype(str).str.lower().eq("baseline")
    baseline_delta_cols = [
        "delta_net_pnl",
        "delta_objective",
        "delta_full_sl_rate",
        "delta_q20_pnl",
        "delta_q35_pnl",
        "scorecard_score",
    ]
    for col in baseline_delta_cols:
        out.loc[out["is_baseline"] & out[col].isna(), col] = 0.0
    return out


def _marginal_family_ablation_from_scorecards(scorecard_dir: Path) -> pd.DataFrame:
    frame = _scorecard_rows_for_marginal_ablation(scorecard_dir)
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    flag_cols = list(next(iter([_family_flags("")])).keys())
    metric_cols = [
        "delta_net_pnl",
        "delta_objective",
        "delta_full_sl_rate",
        "delta_q20_pnl",
        "delta_q35_pnl",
        "scorecard_score",
    ]

    def best_candidate(candidates: pd.DataFrame) -> pd.Series | None:
        if candidates.empty:
            return None
        sort_cols = [col for col in ("scorecard_score", "delta_net_pnl", "delta_objective") if col in candidates.columns]
        return candidates.sort_values(sort_cols, ascending=False, na_position="last").iloc[0]

    for _, row in frame.loc[~frame["is_baseline"]].iterrows():
        for family, flag_col in (
            ("recent_hit_rate_surprise", "has_recent_hit_rate_surprise"),
            ("drift", "has_drift"),
            ("ood", "has_ood"),
            ("uncertainty", "has_uncertainty"),
        ):
            if not bool(row.get(flag_col)):
                continue
            same_scope = frame[
                frame["source_table"].eq(row["source_table"])
                & frame["evidence_family"].eq(row["evidence_family"])
            ]
            target_flags = {col: bool(row.get(col)) for col in flag_cols}
            target_flags[flag_col] = False
            exact = same_scope.copy()
            for col, expected in target_flags.items():
                exact = exact[exact[col].eq(expected)]
            base = best_candidate(exact)
            comparison_type = "marginal_vs_without_family"
            if base is None:
                baseline = same_scope.loc[same_scope["is_baseline"]]
                base = best_candidate(baseline)
                comparison_type = "vs_baseline"
            if base is None:
                continue

            out_row: dict[str, Any] = {
                "family": family,
                "comparison_type": comparison_type,
                "source_table": row["source_table"],
                "evidence_family": row["evidence_family"],
                "variant": row["variant"],
                "variant_family": row["family"],
                "baseline_variant": base["variant"],
                "baseline_family": base["family"],
            }
            for col in metric_cols:
                variant_val = row.get(col)
                baseline_val = base.get(col)
                out_row[f"variant_{col}"] = variant_val
                out_row[f"baseline_{col}"] = baseline_val
                out_row[f"marginal_{col}"] = (
                    float(variant_val) - float(baseline_val)
                    if pd.notna(variant_val) and pd.notna(baseline_val)
                    else np.nan
                )
            rows.append(out_row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    metric_suffixes = (
        "delta_net_pnl",
        "delta_objective",
        "delta_full_sl_rate",
        "delta_q20_pnl",
        "delta_q35_pnl",
        "scorecard_score",
    )
    for col in out.columns:
        if col.startswith(("variant_", "baseline_", "marginal_")) and col.endswith(metric_suffixes):
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_values(
        ["family", "comparison_type", "marginal_scorecard_score", "marginal_delta_net_pnl"],
        ascending=[True, True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return out


def _deployment_verdict(readiness: dict[str, Any], eligible: pd.DataFrame) -> tuple[str, str]:
    if readiness.get("ready_sources", 0) > 0:
        return "portfolio_gate_ready", "Portfolio-wide frozen gate has enough evidence."
    if not eligible.empty and bool(eligible.get("promotion_ready", pd.Series([False])).astype(bool).any()):
        return "eligible_head_only_candidate", "Eligible-head gate passed, but portfolio-wide gate is still not ready."
    if not eligible.empty:
        accepted = int(pd.to_numeric(eligible.get("baseline_trade_count", 0), errors="coerce").max())
        return (
            "research_only_waiting_evidence",
            f"Eligible-head replay ran but accepted-trade evidence remains sparse: baseline accepted={accepted}.",
        )
    return "research_only_waiting_evidence", "No frozen replay has enough current evidence for deployment."


def build_matrix(
    dev_dashboard_dir: Path,
    workflow_dir: Path,
    output_dir: Path,
    *,
    scorecard_dir: Path,
    family_effect_dir: Path,
    top_n: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_dashboard = _read_csv(dev_dashboard_dir / "candidate_deployment_dashboard.csv")
    best_dev = _best_dev_rows(dev_dashboard, top_n=top_n)
    weekly_tail = _weekly_tail_tradeoff(best_dev)
    risk_profiles = _risk_profile_candidates(dev_dashboard)
    guardrails = _guardrail_matrix(dev_dashboard)
    long_period_adequacy = _long_period_adequacy_matrix(dev_dashboard)
    tail_repair_frontier = _tail_repair_frontier_from_existing_grids(ROOT / "data_perp/reports")
    tail_frontier_rerun_shortlist = _tail_frontier_rerun_shortlist(
        tail_repair_frontier,
        ROOT / "data_perp/reports",
        output_dir,
    )
    monthly_consistency, head_consistency = _consistency_breakdowns(dev_dashboard_dir, best_dev)
    family_attribution = _family_attribution_from_scorecards(scorecard_dir, family_effect_dir)
    readiness = _readiness_summary(workflow_dir)
    requested_family_verdict = _requested_reliability_family_verdict(family_attribution, readiness)
    marginal_family_ablation = _marginal_family_ablation_from_scorecards(scorecard_dir)
    evidence_gap, head_evidence_gap = _evidence_gap_frames(workflow_dir)
    head_action_opportunity = _head_action_opportunity_frame(workflow_dir)
    live_health = _live_health_summary(workflow_dir)
    eligible = _eligible_gate_summary(workflow_dir)
    status, reason = _deployment_verdict(readiness, eligible)

    best_dev.to_csv(output_dir / "long_window_development_candidates.csv", index=False)
    weekly_tail.to_csv(output_dir / "long_window_weekly_tail_tradeoff.csv", index=False)
    risk_profiles.to_csv(output_dir / "long_window_risk_profile_candidates.csv", index=False)
    guardrails.to_csv(output_dir / "long_window_guardrail_matrix.csv", index=False)
    long_period_adequacy.to_csv(output_dir / "long_period_adequacy_matrix.csv", index=False)
    tail_repair_frontier.to_csv(output_dir / "tail_repair_frontier_candidates.csv", index=False)
    tail_frontier_rerun_shortlist.to_csv(output_dir / "tail_frontier_rerun_shortlist.csv", index=False)
    monthly_consistency.to_csv(output_dir / "long_window_monthly_consistency.csv", index=False)
    head_consistency.to_csv(output_dir / "long_window_head_consistency.csv", index=False)
    family_attribution.to_csv(output_dir / "reliability_family_attribution.csv", index=False)
    requested_family_verdict.to_csv(output_dir / "requested_reliability_family_verdict.csv", index=False)
    marginal_family_ablation.to_csv(output_dir / "requested_reliability_family_marginal_ablation.csv", index=False)
    evidence_gap.to_csv(output_dir / "post_freeze_evidence_gap.csv", index=False)
    head_evidence_gap.to_csv(output_dir / "post_freeze_head_evidence_gap.csv", index=False)
    head_action_opportunity.to_csv(output_dir / "post_freeze_head_action_opportunity.csv", index=False)
    eligible.to_csv(output_dir / "eligible_head_frozen_gate_summary.csv", index=False)
    readiness_frame = pd.DataFrame([readiness])
    live_frame = pd.DataFrame([live_health])
    readiness_frame.to_csv(output_dir / "portfolio_frozen_readiness_summary.csv", index=False)
    live_frame.to_csv(output_dir / "live_evidence_health_summary.csv", index=False)

    manifest = {
        "generated_by": Path(__file__).name,
        "development_dashboard_dir": str(dev_dashboard_dir),
        "workflow_dir": str(workflow_dir),
        "scorecard_dir": str(scorecard_dir),
        "family_effect_dir": str(family_effect_dir),
        "output_dir": str(output_dir),
        "deployment_status": status,
        "deployment_reason": reason,
        "top_development_rows": int(len(best_dev)),
        "weekly_tail_tradeoff_rows": int(len(weekly_tail)),
        "risk_profile_candidate_rows": int(len(risk_profiles)),
        "guardrail_matrix_rows": int(len(guardrails)),
        "long_period_adequacy_rows": int(len(long_period_adequacy)),
        "tail_repair_frontier_rows": int(len(tail_repair_frontier)),
        "tail_frontier_rerun_shortlist_rows": int(len(tail_frontier_rerun_shortlist)),
        "tail_repair_strict_weekly_tail_rows": int(
            tail_repair_frontier.get("strict_weekly_tail_positive", pd.Series(dtype=bool)).fillna(False).sum()
        )
        if not tail_repair_frontier.empty
        else 0,
        "tail_repair_q10_positive_rows": int(
            tail_repair_frontier.get("q10_positive", pd.Series(dtype=bool)).fillna(False).sum()
        )
        if not tail_repair_frontier.empty
        else 0,
        "strict_tail_long_period_candidate_rows": int(
            long_period_adequacy.get("long_period_strict_tail_pass", pd.Series(dtype=bool)).fillna(False).sum()
        )
        if not long_period_adequacy.empty
        else 0,
        "pragmatic_tail_long_period_candidate_rows": int(
            long_period_adequacy.get("long_period_pragmatic_tail_pass", pd.Series(dtype=bool)).fillna(False).sum()
        )
        if not long_period_adequacy.empty
        else 0,
        "core_long_period_candidate_rows": int(
            long_period_adequacy.get("long_period_core_pass", pd.Series(dtype=bool)).fillna(False).sum()
        )
        if not long_period_adequacy.empty
        else 0,
        "monthly_consistency_rows": int(len(monthly_consistency)),
        "head_consistency_rows": int(len(head_consistency)),
        "evidence_gap_rows": int(len(evidence_gap)),
        "head_evidence_gap_rows": int(len(head_evidence_gap)),
        "head_action_opportunity_rows": int(len(head_action_opportunity)),
        "total_gate_deficit": int(evidence_gap.get("deficit", pd.Series(dtype=int)).sum())
        if not evidence_gap.empty
        else 0,
        "total_head_matured_outcome_deficit": int(
            head_evidence_gap.get("matured_outcome_deficit", pd.Series(dtype=int)).sum()
        )
        if not head_evidence_gap.empty
        else 0,
        "strict_tail_pass_rows": int(guardrails.get("strict_tail_pass", pd.Series(dtype=bool)).fillna(False).sum())
        if not guardrails.empty
        else 0,
        "pragmatic_tail_pass_rows": int(
            guardrails.get("pragmatic_tail_pass", pd.Series(dtype=bool)).fillna(False).sum()
        )
        if not guardrails.empty
        else 0,
        "reliability_family_attribution_rows": int(len(family_attribution)),
        "requested_reliability_family_verdict_rows": int(len(requested_family_verdict)),
        "requested_reliability_family_marginal_ablation_rows": int(len(marginal_family_ablation)),
        "eligible_gate_rows": int(len(eligible)),
        "portfolio_readiness": readiness,
        "live_evidence_health": live_health,
    }
    (output_dir / "contextual_tp_sl_evidence_matrix_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Contextual TP/SL Consolidated Evidence Matrix",
        "",
        f"Deployment status: `{status}`",
        "",
        reason,
        "",
        "## Long-Window Development Candidates",
        "",
        best_dev.to_markdown(index=False) if not best_dev.empty else "_No development candidates._",
        "",
        "## Weekly Tail Tradeoff",
        "",
        weekly_tail.to_markdown(index=False) if not weekly_tail.empty else "_No weekly tail tradeoff rows._",
        "",
        "## Risk-Profile Candidate Selection",
        "",
        risk_profiles.to_markdown(index=False)
        if not risk_profiles.empty
        else "_No risk-profile candidate rows._",
        "",
        "## Strict Guardrail Matrix",
        "",
        guardrails.to_markdown(index=False) if not guardrails.empty else "_No guardrail rows._",
        "",
        "## Long-Period Adequacy And Champion Decision",
        "",
        long_period_adequacy.to_markdown(index=False)
        if not long_period_adequacy.empty
        else "_No long-period adequacy rows._",
        "",
        "## Tail-Repair Frontier Candidates",
        "",
        tail_repair_frontier.to_markdown(index=False)
        if not tail_repair_frontier.empty
        else "_No tail-repair frontier rows._",
        "",
        "## Tail-Frontier Rerun Shortlist",
        "",
        tail_frontier_rerun_shortlist.to_markdown(index=False)
        if not tail_frontier_rerun_shortlist.empty
        else "_No tail-frontier rerun shortlist rows._",
        "",
        "## Long-Window Monthly Consistency",
        "",
        monthly_consistency.to_markdown(index=False)
        if not monthly_consistency.empty
        else "_No monthly consistency rows._",
        "",
        "## Long-Window Head Consistency",
        "",
        head_consistency.to_markdown(index=False)
        if not head_consistency.empty
        else "_No head consistency rows._",
        "",
        "## Reliability Family Attribution",
        "",
        family_attribution.to_markdown(index=False)
        if not family_attribution.empty
        else "_No reliability family attribution rows._",
        "",
        "## Requested Reliability Family Verdict",
        "",
        requested_family_verdict.to_markdown(index=False)
        if not requested_family_verdict.empty
        else "_No requested reliability family verdict rows._",
        "",
        "## Requested Reliability Family Marginal Ablation",
        "",
        marginal_family_ablation.to_markdown(index=False)
        if not marginal_family_ablation.empty
        else "_No requested reliability family marginal ablation rows._",
        "",
        "## Portfolio Frozen Readiness",
        "",
        readiness_frame.to_markdown(index=False),
        "",
        "## Post-Freeze Evidence Gap",
        "",
        evidence_gap.to_markdown(index=False) if not evidence_gap.empty else "_No evidence gap rows._",
        "",
        "## Post-Freeze Head Evidence Gap",
        "",
        head_evidence_gap.to_markdown(index=False)
        if not head_evidence_gap.empty
        else "_No head evidence gap rows._",
        "",
        "## Post-Freeze Head Action Opportunity",
        "",
        head_action_opportunity.to_markdown(index=False)
        if not head_action_opportunity.empty
        else "_No head action opportunity rows._",
        "",
        "## Eligible-Head Frozen Gate",
        "",
        eligible.to_markdown(index=False) if not eligible.empty else "_Eligible-head gate was not run._",
        "",
        "## Live Evidence Health",
        "",
        live_frame.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Development metrics are long-window replay evidence, not deployable proof.",
        "- Portfolio-wide promotion remains blocked until post-freeze policy-action and matured-outcome evidence is sufficient.",
        "- Eligible-head evidence is a smoke test when one head has no action evidence; it cannot promote a portfolio-wide policy.",
        "- Current family attribution favors recent hit-rate surprise plus drift in long-window candidates.",
        "- Standalone OOD and uncertainty are wired into the tests but need stronger binding evidence before promotion.",
        "- Strict guardrails make tail shortfalls explicit instead of allowing normalized scores to hide them.",
        "- Long-period adequacy requires 5 months, 4 heads, at least 5k trades, positive monthly/head consistency, positive objective, and non-worse full-SL rate.",
        "- Long-period strict tail pass requires all weekly tail buckets to improve; pragmatic pass allows Q10 weakness only when Q5/Q20/Q35 improve.",
        "- Tail-repair frontier rows mine existing wider grids for Q10-positive alternatives; rows marked `absolute_not_baseline_aligned` are next-ablation leads, not promotion evidence.",
        "- Tail-frontier rerun shortlist files use `sweep_contextual_tp_sl_arm_combinations.py --combo-file` to replay only selected combinations.",
        "- Monthly/head consistency tables show whether the long-window lift is broad or concentrated.",
        "- Evidence-gap tables state exactly what live-like proof is still missing before promotion.",
        "- Head action-opportunity diagnostics separate absent candidates from missing ranks, rank rejection, and portfolio rejection.",
        "- The requested-family verdict table explicitly audits drift, recent hit-rate surprise, OOD, and uncertainty.",
        "- The marginal-ablation table separates foundational benefit from incremental lift over the closest simpler family.",
    ]
    (output_dir / "contextual_tp_sl_evidence_matrix.md").write_text("\n".join(lines) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-dashboard-dir", type=Path, default=DEFAULT_DEV_DASHBOARD)
    parser.add_argument("--workflow-dir", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--scorecard-dir", type=Path, default=DEFAULT_SCORECARD_DIR)
    parser.add_argument("--family-effect-dir", type=Path, default=DEFAULT_FAMILY_EFFECT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()
    build_matrix(
        args.dev_dashboard_dir,
        args.workflow_dir,
        args.output_dir,
        scorecard_dir=args.scorecard_dir,
        family_effect_dir=args.family_effect_dir,
        top_n=int(args.top_n),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
