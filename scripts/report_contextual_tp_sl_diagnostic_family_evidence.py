#!/usr/bin/env python3
"""Report diagnostic-family evidence for contextual TP/SL candidates.

This is a lightweight consolidation script. It does not rerun portfolio replay;
it reads already-materialized contextual TP/SL walk-forward outputs and the
source-readiness scan, then writes a family-level audit focused on PnL/tail
trade-offs and forward-test blockers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


FAMILY_BY_LABEL = {
    "shortasset_uncertainty_only": "uncertainty",
    "shortasset_drift_only": "drift",
    "shortasset_ood_only": "ood",
    "longbars_uncertainty_only": "uncertainty",
    "longbars_drift_only": "drift",
    "longbars_ood_only": "ood",
    "longbars_weekgate_only": "recent_hr_surprise",
    "combined": "combined",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    if df.empty:
        return "_No rows._"
    cols = [col for col in columns if col in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals: List[str] = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def _family_summary(root: Path) -> pd.DataFrame:
    daily_weekly = _read_csv(root / "daily_weekly_gate" / "daily_weekly_objective_summary.csv")
    promotion = _read_csv(root / "promotion_gate" / "promotion_gate_summary.csv")
    if daily_weekly.empty:
        return pd.DataFrame()
    keep_cols = [
        "label",
        "diagnostic_family",
        "daily_weekly_objective",
        "avg_week_delta_net_pnl",
        "q35_day_delta_net_pnl",
        "q20_day_delta_net_pnl",
        "sum_delta_net_pnl",
        "positive_week_count",
        "positive_week_share",
        "positive_day_count",
        "mean_day_full_sl_delta",
        "mean_week_full_sl_delta",
        "mean_day_hit_rate_delta",
        "june_net_delta",
        "june_full_sl_delta",
        "sum_delta_trades",
    ]
    summary = daily_weekly[[col for col in keep_cols if col in daily_weekly.columns]].copy()
    summary["diagnostic_family"] = summary["label"].map(FAMILY_BY_LABEL).fillna(summary.get("diagnostic_family", "unknown"))
    if not promotion.empty:
        extra = promotion[
            [
                col
                for col in [
                    "label",
                    "monthly_objective",
                    "positive_month_count",
                    "mean_month_full_sl_delta",
                    "mean_month_drawdown_delta",
                    "min_monthly_drawdown_delta",
                    "monthly_objective_rank",
                ]
                if col in promotion.columns
            ]
        ]
        summary = summary.merge(extra, on="label", how="left")
    return summary.sort_values(["daily_weekly_objective", "sum_delta_net_pnl"], ascending=False)


def _monthly_summary(root: Path) -> pd.DataFrame:
    monthly = _read_csv(root / "monthly_walkforward_global.csv")
    if monthly.empty:
        return pd.DataFrame()
    monthly = monthly[monthly["label"].ne("wf_recent")].copy()
    monthly["diagnostic_family"] = monthly["label"].map(FAMILY_BY_LABEL).fillna("unknown")
    rows: List[Dict[str, Any]] = []
    for (label, family), group in monthly.groupby(["label", "diagnostic_family"], sort=False):
        rows.append(
            {
                "label": label,
                "diagnostic_family": family,
                "months": int(group["month"].nunique()),
                "sum_delta_net_pnl": float(group["delta_net_pnl"].sum()),
                "positive_month_count": int((pd.to_numeric(group["delta_net_pnl"], errors="coerce") > 0).sum()),
                "min_month_delta_net_pnl": float(pd.to_numeric(group["delta_net_pnl"], errors="coerce").min()),
                "apr_jun_delta_net_pnl": float(
                    pd.to_numeric(group.loc[group["month"].astype(str).isin(["2026-04", "2026-05", "2026-06"]), "delta_net_pnl"], errors="coerce").sum()
                ),
                "june_delta_net_pnl": float(
                    pd.to_numeric(group.loc[group["month"].astype(str).eq("2026-06"), "delta_net_pnl"], errors="coerce").sum()
                ),
                "mean_delta_full_sl_rate": float(pd.to_numeric(group["delta_full_sl_rate"], errors="coerce").mean()),
                "mean_delta_trade_count": float(pd.to_numeric(group["delta_trade_count"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["apr_jun_delta_net_pnl", "sum_delta_net_pnl"], ascending=False)


def _head_contributions(root: Path) -> pd.DataFrame:
    head = _read_csv(root / "monthly_walkforward_head.csv")
    if head.empty:
        return pd.DataFrame()
    head = head[head["label"].ne("wf_recent")].copy()
    head["diagnostic_family"] = head["label"].map(FAMILY_BY_LABEL).fillna("unknown")
    rows: List[Dict[str, Any]] = []
    for (label, family, strategy_head), group in head.groupby(["label", "diagnostic_family", "head"], sort=False):
        pnl = pd.to_numeric(group["delta_net_pnl"], errors="coerce")
        rows.append(
            {
                "label": label,
                "diagnostic_family": family,
                "head": strategy_head,
                "sum_delta_net_pnl": float(pnl.sum()),
                "positive_month_count": int((pnl > 0).sum()),
                "mean_delta_hit_rate": float(pd.to_numeric(group["delta_hit_rate"], errors="coerce").mean()),
                "mean_delta_full_sl_rate": float(pd.to_numeric(group["delta_full_sl_rate"], errors="coerce").mean()),
                "sum_delta_trades": int(pd.to_numeric(group["delta_trades"], errors="coerce").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["label", "sum_delta_net_pnl"], ascending=[True, False])


def _source_readiness(scan_dir: Path) -> pd.DataFrame:
    scan_csv = scan_dir / "contextual_tp_sl_candidate_source_scan.csv"
    if not scan_csv.exists():
        return pd.DataFrame()
    frame = pd.read_csv(scan_csv)
    keep = [
        "source_dir",
        "candidate_end",
        "post_cutoff_rows",
        "post_cutoff_timestamps",
        "post_cutoff_active_heads",
        "has_required_diagnostic_groups",
        "missing_required_diagnostic_groups",
        "diagnostic_group_coverage",
        "usable_post_cutoff",
    ]
    return frame[[col for col in keep if col in frame.columns]].head(8)


def _decision_rows(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for label in ["longbars_drift_only", "longbars_weekgate_only", "longbars_uncertainty_only"]:
        match = summary[summary["label"].eq(label)]
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        if label == "longbars_drift_only":
            role = "risk_controlled_default"
            rationale = "Best recurrence/tail compromise; positive weeks and months are strongest."
        elif label == "longbars_weekgate_only":
            role = "high_upside_challenger"
            rationale = "Best raw PnL and recent-HR-surprise evidence; weaker recurrence."
        else:
            role = "pnl_tail_challenger"
            rationale = "Strong PnL with full-SL improvement; less stable than drift."
        rows.append(
            {
                "label": label,
                "role": role,
                "diagnostic_family": row.get("diagnostic_family"),
                "sum_delta_net_pnl": row.get("sum_delta_net_pnl"),
                "positive_week_count": row.get("positive_week_count"),
                "q20_day_delta_net_pnl": row.get("q20_day_delta_net_pnl"),
                "mean_day_full_sl_delta": row.get("mean_day_full_sl_delta"),
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-scan-dir", default="")
    args = parser.parse_args()

    root = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_dir = Path(args.source_scan_dir) if args.source_scan_dir else root / "fixed_validation_pack" / "broad_source_scan_diagnostic_groups"

    family = _family_summary(root)
    monthly = _monthly_summary(root)
    heads = _head_contributions(root)
    readiness = _source_readiness(scan_dir)
    decisions = _decision_rows(family)

    family.to_csv(out_dir / "diagnostic_family_long_window_summary.csv", index=False)
    monthly.to_csv(out_dir / "diagnostic_family_monthly_summary.csv", index=False)
    heads.to_csv(out_dir / "diagnostic_family_head_contributions.csv", index=False)
    readiness.to_csv(out_dir / "diagnostic_family_forward_readiness.csv", index=False)
    decisions.to_csv(out_dir / "diagnostic_family_carry_forward_decisions.csv", index=False)

    manifest = {
        "generated_by": "report_contextual_tp_sl_diagnostic_family_evidence",
        "root_dir": str(root),
        "source_scan_dir": str(scan_dir),
        "outputs": [
            "diagnostic_family_evidence_report.md",
            "diagnostic_family_long_window_summary.csv",
            "diagnostic_family_monthly_summary.csv",
            "diagnostic_family_head_contributions.csv",
            "diagnostic_family_forward_readiness.csv",
            "diagnostic_family_carry_forward_decisions.csv",
        ],
    }
    (out_dir / "diagnostic_family_evidence_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    md = [
        "# Contextual TP/SL Diagnostic Family Evidence",
        "",
        "Scope: January-June 2026 development walk-forward replay. This report consolidates existing artifacts only; it does not rerun replay or claim untouched OOS.",
        "",
        "Baseline: `wf_recent`. Objective used by the latest gate: `avg_week_pnl + 0.7 * q35_day_pnl + 0.3 * q20_day_pnl`.",
        "",
        "## Long-Window A/B Summary",
        "",
        _markdown_table(
            family,
            [
                "label",
                "diagnostic_family",
                "daily_weekly_objective",
                "sum_delta_net_pnl",
                "positive_week_count",
                "positive_week_share",
                "q35_day_delta_net_pnl",
                "q20_day_delta_net_pnl",
                "mean_day_full_sl_delta",
                "june_net_delta",
                "positive_month_count",
            ],
        ),
        "",
        "## Monthly Stability",
        "",
        _markdown_table(
            monthly,
            [
                "label",
                "diagnostic_family",
                "months",
                "sum_delta_net_pnl",
                "positive_month_count",
                "min_month_delta_net_pnl",
                "apr_jun_delta_net_pnl",
                "june_delta_net_pnl",
                "mean_delta_full_sl_rate",
                "mean_delta_trade_count",
            ],
        ),
        "",
        "## Per-Head Contributions",
        "",
        _markdown_table(
            heads,
            [
                "label",
                "diagnostic_family",
                "head",
                "sum_delta_net_pnl",
                "positive_month_count",
                "mean_delta_hit_rate",
                "mean_delta_full_sl_rate",
                "sum_delta_trades",
            ],
        ),
        "",
        "## Carry-Forward Decision",
        "",
        _markdown_table(
            decisions,
            [
                "label",
                "role",
                "diagnostic_family",
                "sum_delta_net_pnl",
                "positive_week_count",
                "q20_day_delta_net_pnl",
                "mean_day_full_sl_delta",
                "rationale",
            ],
        ),
        "",
        "## Forward Readiness",
        "",
        _markdown_table(
            readiness,
            [
                "source_dir",
                "candidate_end",
                "post_cutoff_rows",
                "post_cutoff_timestamps",
                "post_cutoff_active_heads",
                "has_required_diagnostic_groups",
                "missing_required_diagnostic_groups",
                "diagnostic_group_coverage",
                "usable_post_cutoff",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- Recent hit-rate surprise is the strongest raw-PnL diagnostic, via `longbars_weekgate_only`, but it has weaker recurrence and is missing from the closest forward candidate tables.",
        "- Drift is the most stable diagnostic family and remains the risk-controlled default/fallback.",
        "- Uncertainty is a valid challenger with strong PnL and full-SL improvement, but it has worse daily-tail quantiles than drift.",
        "- OOD is not globally reliable in the current evidence; it is useful to keep in tests, but not as a standalone default.",
        "- The next replay should require enough post-cutoff rows and should materialize recent-HR-surprise/performance columns if we want the full four-family test to remain comparable.",
        "",
    ]
    (out_dir / "diagnostic_family_evidence_report.md").write_text("\n".join(md), encoding="utf-8")
    print(out_dir / "diagnostic_family_evidence_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
