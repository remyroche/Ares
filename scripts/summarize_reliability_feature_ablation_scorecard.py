#!/usr/bin/env python3
"""Consolidate contextual TP/SL reliability feature ablation evidence.

The inputs are already-generated replay artifacts.  This script does not rerun
portfolio replay; it creates one scorecard that compares drift, recent
hit-rate/performance surprise, OOD, and uncertainty branches under the current
long-window evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_OUT = ROOT / "data_perp/reports/contextual_tp_sl_reliability_feature_scorecard_v1_20260701"


INPUTS = {
    "promotion": ROOT
    / "data_perp/reports/contextual_tp_sl_current_candidate_promotion_table_v1_20260701/candidate_promotion_summary.csv",
    "promotion_monthly": ROOT
    / "data_perp/reports/contextual_tp_sl_current_candidate_promotion_table_v1_20260701/candidate_monthly_metrics.csv",
    "promotion_head": ROOT
    / "data_perp/reports/contextual_tp_sl_current_candidate_promotion_table_v1_20260701/candidate_per_head_metrics.csv",
    "shortlist": ROOT
    / "data_perp/reports/contextual_tp_sl_candidate_shortlist_v1_20260701/candidate_readiness_matrix.csv",
    "feature_family": ROOT
    / "data_perp/reports/contextual_tp_sl_feature_family_readout_v1_20260701/feature_family_ablation_readout.csv",
    "tailgrid": ROOT
    / "data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_tail_balance_tailgrid_v4_20260701/portfolio_summary_tailgrid.csv",
    "tailgrid_tail": ROOT
    / "data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_tail_balance_tailgrid_v4_20260701/tail_quantile_summary.csv",
    "headscope": ROOT
    / "data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_tail_balance_headscope_v5_20260701/portfolio_summary_headscope.csv",
    "headscope_tail": ROOT
    / "data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_tail_balance_headscope_v5_20260701/tail_quantile_summary.csv",
    "expanding_ood_uncertainty": ROOT
    / "data_perp/reports/contextual_tp_sl_recent_drift_ood_uncertainty_expanding_v1_20260701/combo_expanding_summary.csv",
    "headscoped_ood": ROOT
    / "data_perp/reports/contextual_tp_sl_headscoped_ood_expanding_v1_20260701/combo_expanding_summary.csv",
    "row_guard": ROOT
    / "data_perp/reports/contextual_tp_sl_candidate_shortlist_v1_20260701/full_guarded_promotion_metrics.csv",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(val):
        return ""
    return f"{val:,.{digits}f}"


def _fmt_table(frame: pd.DataFrame, cols: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    data = frame.loc[:, [c for c in cols if c in frame.columns]].copy()
    if data.empty:
        return "_No matching columns._"
    return data.to_markdown(index=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _family_for_variant(variant: str) -> str:
    text = str(variant).lower()
    families: list[str] = []
    if "recent_hr" in text or "recent_perf" in text or "recent_drift" in text or text.startswith("rh_"):
        families.append("recent_hr")
    if "drift" in text:
        families.append("drift")
    if "ood" in text:
        families.append("OOD")
    if "uncertainty" in text:
        families.append("uncertainty")
    if "gate" in text:
        families.append("weekly_head_gate")
    if not families:
        return "baseline_or_other"
    return "+".join(families)


def _score_candidate(row: pd.Series) -> float:
    pnl = float(row.get("delta_net_pnl", row.get("delta_vs_baseline_net_pnl", 0.0)) or 0.0)
    objective = float(
        row.get(
            "delta_objective_week",
            row.get("delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20", 0.0),
        )
        or 0.0
    )
    full_sl = float(row.get("delta_full_sl_rate", row.get("delta_vs_baseline_full_sl_rate", 0.0)) or 0.0)
    q10 = float(row.get("delta_q10", row.get("delta_q10_week_net_pnl", 0.0)) or 0.0)
    return pnl + 2.0 * objective + 50000.0 * max(0.0, -full_sl) + 0.25 * q10


def _promotion_rows(promotion: pd.DataFrame) -> pd.DataFrame:
    if promotion.empty:
        return pd.DataFrame()
    out = promotion.copy()
    out["family"] = out["variant"].map(_family_for_variant)
    out["scorecard_score"] = out.apply(_score_candidate, axis=1)
    keep = [
        "variant",
        "role",
        "family",
        "net_pnl",
        "delta_vs_baseline_net_pnl",
        "hit_rate",
        "full_sl_rate",
        "delta_vs_baseline_full_sl_rate",
        "weekly_q05_pnl",
        "weekly_q10_pnl",
        "weekly_q20_pnl",
        "objective_avgweek_0p7dayq35_0p3dayq20",
        "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
        "jaccard_vs_baseline",
        "scorecard_score",
    ]
    return out[[c for c in keep if c in out.columns]].sort_values("scorecard_score", ascending=False)


def _tailgrid_rows(name: str, portfolio: pd.DataFrame, tail: pd.DataFrame) -> pd.DataFrame:
    if portfolio.empty:
        return pd.DataFrame()
    out = portfolio.copy()
    if not tail.empty and "variant" in tail.columns:
        if "granularity" in tail.columns:
            weekly = tail[tail["granularity"].astype(str).eq("weekly")].copy()
            if not weekly.empty:
                tail = weekly
        tail_cols = [
            "variant",
            "q05",
            "q10",
            "q20",
            "q35",
            "delta_q05",
            "delta_q10",
            "delta_q20",
            "delta_q35",
            "tail_objective_delta",
        ]
        out = out.merge(tail[[c for c in tail_cols if c in tail.columns]], on="variant", how="left")
    out["evidence_family"] = name
    out["family"] = out["variant"].map(_family_for_variant)
    out["scorecard_score"] = out.apply(_score_candidate, axis=1)
    keep = [
        "evidence_family",
        "variant",
        "family",
        "net_pnl",
        "delta_net_pnl",
        "trade_count",
        "full_sl_rate",
        "delta_full_sl_rate",
        "timeout_rate",
        "delta_timeout_rate",
        "max_drawdown",
        "q05",
        "q10",
        "q20",
        "q35",
        "delta_q05",
        "delta_q10",
        "delta_q20",
        "delta_q35",
        "tail_objective_delta",
        "scorecard_score",
    ]
    return out[[c for c in keep if c in out.columns]].sort_values("scorecard_score", ascending=False)


def _expanding_rows(label: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out["evidence_family"] = label
    out["family"] = out["variant"].map(_family_for_variant)
    out["scorecard_score"] = out.apply(_score_candidate, axis=1)
    keep = [
        "evidence_family",
        "variant",
        "family",
        "delta_net_pnl",
        "delta_objective_week",
        "delta_hit_rate",
        "delta_full_sl_rate",
        "delta_timeout_rate",
        "delta_q20_week_net_pnl",
        "delta_q35_week_net_pnl",
        "delta_worst_week_net_pnl",
        "penalized_rows",
        "penalized_share_month_mean",
        "scorecard_score",
    ]
    return out[[c for c in keep if c in out.columns]].sort_values("scorecard_score", ascending=False)


def _family_readout(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    numeric = [c for c in out.columns if out[c].dtype.kind in "if"]
    sort_col = "delta_objective" if "delta_objective" in out.columns else numeric[0] if numeric else out.columns[0]
    return out.sort_values(sort_col, ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {name: _read_csv(path) for name, path in INPUTS.items()}

    promotion = _promotion_rows(inputs["promotion"])
    tailgrid = _tailgrid_rows("tailgrid_v4_recent_hr", inputs["tailgrid"], inputs["tailgrid_tail"])
    headscope = _tailgrid_rows("headscope_v5_recent_hr", inputs["headscope"], inputs["headscope_tail"])
    expanding = pd.concat(
        [
            _expanding_rows("expanding_recent_drift_ood_uncertainty", inputs["expanding_ood_uncertainty"]),
            _expanding_rows("expanding_headscoped_ood", inputs["headscoped_ood"]),
        ],
        ignore_index=True,
    ).sort_values("scorecard_score", ascending=False)
    feature_family = _family_readout(inputs["feature_family"])

    promotion.to_csv(args.output_dir / "promotion_scorecard.csv", index=False)
    tailgrid.to_csv(args.output_dir / "tailgrid_recent_hr_scorecard.csv", index=False)
    headscope.to_csv(args.output_dir / "headscope_recent_hr_scorecard.csv", index=False)
    expanding.to_csv(args.output_dir / "expanding_family_scorecard.csv", index=False)
    feature_family.to_csv(args.output_dir / "feature_family_readout.csv", index=False)

    best_promotion = promotion.iloc[0].to_dict() if not promotion.empty else {}
    best_tailgrid = tailgrid[tailgrid["variant"].ne("baseline")].iloc[0].to_dict() if not tailgrid.empty else {}
    best_expanding = expanding.iloc[0].to_dict() if not expanding.empty else {}
    report = [
        "# Reliability Feature Ablation Scorecard",
        "",
        "This is a consolidation of already-generated long-window replay artifacts. It does not rerun portfolio replay.",
        "",
        "## Scope",
        "",
        "- Main guarded replay period: `2026-02-01` to `2026-06-27 12:00 UTC`.",
        "- Expanding monthly evidence spans the broader materialized history used by the source artifacts.",
        "- Costs are included in replay summaries.",
        "- Evidence remains proxy/guarded replay unless explicitly marked prospective.",
        "",
        "## Lead Decisions",
        "",
        f"- Best current promotion-table candidate: `{best_promotion.get('variant', 'n/a')}`.",
        f"- Best v4/v5 recent-HR replay candidate: `{best_tailgrid.get('variant', 'n/a')}`.",
        f"- Best expanding diagnostic-family candidate: `{best_expanding.get('variant', 'n/a')}`.",
        "",
        "Current interpretation:",
        "",
        "- Recent hit-rate / recent-performance surprise is the strongest active reliability family.",
        "- Drift helps only when blended or scoped; broad drift alone is too unstable.",
        "- OOD can help in narrow/head-scoped forms, but it is not yet a robust default.",
        "- Uncertainty has useful pockets but current thresholds often do not bind or do not beat recent-HR/drift blends.",
        "- The open requirement is still prospective/frozen dual scoring with enough accepted-trade changes.",
        "",
        "## Promotion Candidates",
        "",
        _fmt_table(
            promotion,
            [
                "variant",
                "role",
                "family",
                "delta_vs_baseline_net_pnl",
                "delta_vs_baseline_full_sl_rate",
                "weekly_q05_pnl",
                "weekly_q10_pnl",
                "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
                "jaccard_vs_baseline",
            ],
        ),
        "",
        "## Recent-HR Tailgrid v4",
        "",
        _fmt_table(
            tailgrid,
            [
                "variant",
                "family",
                "delta_net_pnl",
                "delta_full_sl_rate",
                "delta_timeout_rate",
                "delta_q05",
                "delta_q10",
                "delta_q20",
                "tail_objective_delta",
            ],
        ),
        "",
        "## Headscope v5",
        "",
        _fmt_table(
            headscope,
            [
                "variant",
                "family",
                "delta_net_pnl",
                "delta_full_sl_rate",
                "delta_timeout_rate",
                "delta_q05",
                "delta_q10",
                "delta_q20",
                "tail_objective_delta",
            ],
        ),
        "",
        "## Expanding Diagnostic Families",
        "",
        _fmt_table(
            expanding,
            [
                "evidence_family",
                "variant",
                "family",
                "delta_net_pnl",
                "delta_objective_week",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_q20_week_net_pnl",
                "delta_q35_week_net_pnl",
            ],
        ),
        "",
        "## Feature-Family Readout",
        "",
        _fmt_table(feature_family.head(40), list(feature_family.columns[:10])),
        "",
        "## Next Gate",
        "",
        "Freeze the lead balanced candidate plus one high-PnL challenger and one conservative/tail challenger, then continue dual scoring until accepted-trade membership changes on enough post-freeze rows. Do not promote from this scorecard alone.",
    ]
    (args.output_dir / "reliability_feature_ablation_scorecard.md").write_text("\n".join(report) + "\n")

    manifest = {
        "generated_by": Path(__file__).name,
        "inputs": {name: str(path) for name, path in INPUTS.items()},
        "outputs": [
            "promotion_scorecard.csv",
            "tailgrid_recent_hr_scorecard.csv",
            "headscope_recent_hr_scorecard.csv",
            "expanding_family_scorecard.csv",
            "feature_family_readout.csv",
            "reliability_feature_ablation_scorecard.md",
        ],
        "best": {
            "promotion": best_promotion,
            "tailgrid": best_tailgrid,
            "expanding": best_expanding,
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
