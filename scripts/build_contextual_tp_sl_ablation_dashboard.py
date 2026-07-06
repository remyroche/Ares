#!/usr/bin/env python3
"""Build a contextual TP/SL A/B dashboard from current evidence artifacts.

This consolidates long-window development replay metrics with frozen
dual-scoring readiness.  It deliberately separates a candidate that looks good
in development replay from one that is deployable under the frozen gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMOTION_DIR = ROOT / "data_perp/reports/contextual_tp_sl_current_candidate_promotion_table_v1_20260701"
DEFAULT_SCORECARD_DIR = ROOT / "data_perp/reports/contextual_tp_sl_reliability_feature_scorecard_v1_20260701"
DEFAULT_READINESS_DIR = ROOT / "data_perp/reports/contextual_tp_sl_cumulative_flat_gate_readiness_v1_20260701"
DEFAULT_LEDGER_DIR = ROOT / "data_perp/reports/contextual_tp_sl_cumulative_flat_gate_ledger_v1_20260701"
DEFAULT_OUT = ROOT / "data_perp/reports/contextual_tp_sl_ablation_dashboard_v1_20260701"


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


def _fmt_table(frame: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[c for c in cols if c in frame.columns]].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    return view.to_markdown(index=False)


def _monthly_consistency(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty or "baseline" not in set(monthly.get("variant", [])):
        return pd.DataFrame()
    baseline = monthly[monthly["variant"].eq("baseline")].copy()
    base_cols = {
        "net_pnl": "baseline_net_pnl",
        "hit_rate": "baseline_hit_rate",
        "full_sl_rate": "baseline_full_sl_rate",
        "timeout_rate": "baseline_timeout_rate",
    }
    baseline = baseline.rename(columns=base_cols)
    merged = monthly[~monthly["variant"].eq("baseline")].merge(
        baseline[["month", *base_cols.values()]],
        on="month",
        how="left",
    )
    if merged.empty:
        return pd.DataFrame()
    merged["delta_month_net_pnl"] = merged["net_pnl"] - merged["baseline_net_pnl"]
    merged["delta_month_hit_rate"] = merged["hit_rate"] - merged["baseline_hit_rate"]
    merged["delta_month_full_sl_rate"] = merged["full_sl_rate"] - merged["baseline_full_sl_rate"]
    merged["delta_month_timeout_rate"] = merged["timeout_rate"] - merged["baseline_timeout_rate"]
    rows = []
    for variant, group in merged.groupby("variant", sort=False):
        rows.append(
            {
                "variant": variant,
                "months": int(group["month"].nunique()),
                "positive_net_months": int(group["delta_month_net_pnl"].gt(0.0).sum()),
                "positive_net_month_share": float(group["delta_month_net_pnl"].gt(0.0).mean()),
                "mean_month_delta_net_pnl": float(group["delta_month_net_pnl"].mean()),
                "min_month_delta_net_pnl": float(group["delta_month_net_pnl"].min()),
                "mean_month_delta_hit_rate": float(group["delta_month_hit_rate"].mean()),
                "mean_month_delta_full_sl_rate": float(group["delta_month_full_sl_rate"].mean()),
                "mean_month_delta_timeout_rate": float(group["delta_month_timeout_rate"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _head_consistency(heads: pd.DataFrame) -> pd.DataFrame:
    if heads.empty or "baseline" not in set(heads.get("variant", [])):
        return pd.DataFrame()
    baseline = heads[heads["variant"].eq("baseline")].rename(
        columns={
            "net_pnl": "baseline_net_pnl",
            "hit_rate": "baseline_hit_rate",
            "full_sl_rate": "baseline_full_sl_rate",
            "timeout_rate": "baseline_timeout_rate",
        }
    )
    merged = heads[~heads["variant"].eq("baseline")].merge(
        baseline[["head", "baseline_net_pnl", "baseline_hit_rate", "baseline_full_sl_rate", "baseline_timeout_rate"]],
        on="head",
        how="left",
    )
    if merged.empty:
        return pd.DataFrame()
    merged["delta_head_net_pnl"] = merged["net_pnl"] - merged["baseline_net_pnl"]
    merged["delta_head_hit_rate"] = merged["hit_rate"] - merged["baseline_hit_rate"]
    merged["delta_head_full_sl_rate"] = merged["full_sl_rate"] - merged["baseline_full_sl_rate"]
    rows = []
    for variant, group in merged.groupby("variant", sort=False):
        rows.append(
            {
                "variant": variant,
                "heads": int(group["head"].nunique()),
                "positive_net_heads": int(group["delta_head_net_pnl"].gt(0.0).sum()),
                "positive_net_head_share": float(group["delta_head_net_pnl"].gt(0.0).mean()),
                "mean_head_delta_net_pnl": float(group["delta_head_net_pnl"].mean()),
                "min_head_delta_net_pnl": float(group["delta_head_net_pnl"].min()),
                "mean_head_delta_hit_rate": float(group["delta_head_hit_rate"].mean()),
                "mean_head_delta_full_sl_rate": float(group["delta_head_full_sl_rate"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _candidate_dashboard(
    promotion: pd.DataFrame,
    monthly_consistency: pd.DataFrame,
    head_consistency: pd.DataFrame,
    readiness: dict[str, Any],
    ledger_manifest: dict[str, Any],
) -> pd.DataFrame:
    if promotion.empty:
        return pd.DataFrame()
    out = promotion.copy()
    out = out.merge(monthly_consistency, on="variant", how="left")
    out = out.merge(head_consistency, on="variant", how="left")
    req = readiness.get("requirements") or {}
    source = readiness.get("selected_source") or readiness.get("nearest_source") or {}
    ledger_post = ledger_manifest.get("post_cutoff") or {}
    ready_sources = int(readiness.get("ready_sources") or 0)
    out["frozen_gate_ready"] = bool(ready_sources > 0)
    post_cutoff_rows = int(source.get("post_cutoff_rows") or ledger_post.get("post_cutoff_rows") or 0)
    post_cutoff_timestamps = int(
        source.get("post_cutoff_timestamps") or ledger_post.get("post_cutoff_timestamps") or 0
    )
    post_cutoff_active_heads = int(
        source.get("post_cutoff_active_heads") or ledger_post.get("post_cutoff_active_heads") or 0
    )
    out["post_cutoff_rows"] = post_cutoff_rows
    out["post_cutoff_timestamps"] = post_cutoff_timestamps
    out["post_cutoff_active_heads"] = post_cutoff_active_heads
    out["policy_action_rows_estimate"] = int(source.get("policy_action_rows_estimate") or 0)
    out["policy_action_timestamps_estimate"] = int(source.get("policy_action_timestamps_estimate") or 0)
    out["policy_action_estimate_source"] = str(source.get("policy_action_estimate_source") or "")
    out["policy_outcome_rows_estimate"] = int(source.get("policy_outcome_rows_estimate") or 0)
    out["policy_outcome_timestamps_estimate"] = int(source.get("policy_outcome_timestamps_estimate") or 0)
    out["policy_outcome_estimate_source"] = str(source.get("policy_outcome_estimate_source") or "")
    out["post_cutoff_rows_needed"] = max(0, int(req.get("min_post_cutoff_rows") or 0) - post_cutoff_rows)
    out["post_cutoff_timestamps_needed"] = max(
        0,
        int(req.get("min_post_cutoff_timestamps") or 0) - post_cutoff_timestamps,
    )
    out["policy_action_rows_needed"] = max(
        0,
        int(req.get("min_policy_action_rows") or 0) - int(out["policy_action_rows_estimate"].iloc[0]),
    )
    out["policy_action_timestamps_needed"] = max(
        0,
        int(req.get("min_policy_action_timestamps") or 0) - int(out["policy_action_timestamps_estimate"].iloc[0]),
    )
    out["policy_outcome_rows_needed"] = max(
        0,
        int(req.get("min_policy_outcome_rows") or 0) - int(out["policy_outcome_rows_estimate"].iloc[0]),
    )
    out["policy_outcome_timestamps_needed"] = max(
        0,
        int(req.get("min_policy_outcome_timestamps") or 0) - int(out["policy_outcome_timestamps_estimate"].iloc[0]),
    )
    out["dev_pnl_pass"] = pd.to_numeric(out.get("delta_vs_baseline_net_pnl", 0.0), errors="coerce").gt(0.0)
    out["dev_objective_pass"] = pd.to_numeric(
        out.get("delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20", 0.0),
        errors="coerce",
    ).gt(0.0)
    out["dev_tail_pass"] = pd.to_numeric(out.get("delta_vs_baseline_full_sl_rate", 0.0), errors="coerce").le(0.0)
    out["monthly_consistency_pass"] = pd.to_numeric(
        out.get("positive_net_month_share", np.nan),
        errors="coerce",
    ).ge(0.60)
    out["development_candidate_pass"] = (
        out["dev_pnl_pass"] & out["dev_objective_pass"] & out["dev_tail_pass"] & out["monthly_consistency_pass"]
    )
    out["deployment_candidate_pass"] = out["development_candidate_pass"] & out["frozen_gate_ready"]
    out["candidate_status"] = np.where(
        out["deployment_candidate_pass"],
        "deployment_candidate",
        np.where(out["development_candidate_pass"], "research_candidate_waiting_frozen_evidence", "rejected_or_diagnostic"),
    )
    sort_cols = [
        "deployment_candidate_pass",
        "development_candidate_pass",
        "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
        "delta_vs_baseline_net_pnl",
    ]
    return out.sort_values(sort_cols, ascending=[False, False, False, False])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion-dir", type=Path, default=DEFAULT_PROMOTION_DIR)
    parser.add_argument("--scorecard-dir", type=Path, default=DEFAULT_SCORECARD_DIR)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    promotion = _read_csv(args.promotion_dir / "candidate_promotion_summary.csv")
    monthly = _read_csv(args.promotion_dir / "candidate_monthly_metrics.csv")
    heads = _read_csv(args.promotion_dir / "candidate_per_head_metrics.csv")
    scorecard = _read_csv(args.scorecard_dir / "promotion_scorecard.csv")
    tailgrid = _read_csv(args.scorecard_dir / "tailgrid_recent_hr_scorecard.csv")
    expanding = _read_csv(args.scorecard_dir / "expanding_family_scorecard.csv")
    readiness = _read_json(args.readiness_dir / "latest_flat_frozen_gate_readiness.json")
    ledger_manifest = _read_json(args.ledger_dir / "cumulative_flat_ledger_manifest.json")

    monthly_out = _monthly_consistency(monthly)
    head_out = _head_consistency(heads)
    dashboard = _candidate_dashboard(promotion, monthly_out, head_out, readiness, ledger_manifest)

    dashboard.to_csv(args.output_dir / "candidate_deployment_dashboard.csv", index=False)
    monthly_out.to_csv(args.output_dir / "candidate_monthly_consistency.csv", index=False)
    head_out.to_csv(args.output_dir / "candidate_head_consistency.csv", index=False)

    ready_sources = int(readiness.get("ready_sources") or 0)
    post = ledger_manifest.get("post_cutoff") or {}
    req = readiness.get("requirements") or {}
    source = readiness.get("selected_source") or readiness.get("nearest_source") or {}
    lines = [
        "# Contextual TP/SL A/B Promotion Dashboard",
        "",
        "This dashboard separates long-window development replay evidence from frozen dual-scoring deployment evidence.",
        "",
        "## Frozen Evidence Gate",
        "",
        f"- Ready sources: `{ready_sources}`",
        f"- Gate ran: `{bool(readiness.get('ran_gate'))}`",
        f"- Post-cutoff rows: `{post.get('post_cutoff_rows', 0)}` / `{req.get('min_post_cutoff_rows', 0)}`",
        f"- Post-cutoff timestamps: `{post.get('post_cutoff_timestamps', 0)}` / `{req.get('min_post_cutoff_timestamps', 0)}`",
        f"- Post-cutoff active heads: `{post.get('post_cutoff_active_heads', 0)}` / `{req.get('min_post_cutoff_active_heads', 0)}`",
        f"- Estimated policy-action rows: `{source.get('policy_action_rows_estimate', 0)}` / `{req.get('min_policy_action_rows', 0)}`",
        f"- Estimated policy-action timestamps: `{source.get('policy_action_timestamps_estimate', 0)}` / `{req.get('min_policy_action_timestamps', 0)}`",
        f"- Policy-action estimate source: `{source.get('policy_action_estimate_source', '')}`",
        f"- Estimated matured policy-outcome rows: `{source.get('policy_outcome_rows_estimate', 0)}` / `{req.get('min_policy_outcome_rows', 0)}`",
        f"- Estimated matured policy-outcome timestamps: `{source.get('policy_outcome_timestamps_estimate', 0)}` / `{req.get('min_policy_outcome_timestamps', 0)}`",
        f"- Policy-outcome estimate source: `{source.get('policy_outcome_estimate_source', '')}`",
        "",
        "## Candidate Status",
        "",
        _fmt_table(
            dashboard,
            [
                "variant",
                "role",
                "candidate_status",
                "delta_vs_baseline_net_pnl",
                "delta_vs_baseline_full_sl_rate",
                "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
                "positive_net_month_share",
                "positive_net_head_share",
                "jaccard_vs_baseline",
                "post_cutoff_rows_needed",
                "post_cutoff_timestamps_needed",
                "policy_action_rows_needed",
                "policy_action_timestamps_needed",
                "policy_outcome_rows_needed",
                "policy_outcome_timestamps_needed",
            ],
        ),
        "",
        "## Monthly Consistency",
        "",
        _fmt_table(
            monthly_out,
            [
                "variant",
                "months",
                "positive_net_month_share",
                "mean_month_delta_net_pnl",
                "min_month_delta_net_pnl",
                "mean_month_delta_full_sl_rate",
                "mean_month_delta_timeout_rate",
            ],
        ),
        "",
        "## Head Consistency",
        "",
        _fmt_table(
            head_out,
            [
                "variant",
                "heads",
                "positive_net_head_share",
                "mean_head_delta_net_pnl",
                "min_head_delta_net_pnl",
                "mean_head_delta_full_sl_rate",
            ],
        ),
        "",
        "## Reliability-Family Evidence",
        "",
        "Promotion scorecard:",
        "",
        _fmt_table(
            scorecard,
            [
                "variant",
                "family",
                "delta_vs_baseline_net_pnl",
                "delta_vs_baseline_full_sl_rate",
                "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
                "jaccard_vs_baseline",
            ],
        ),
        "",
        "Tailgrid:",
        "",
        _fmt_table(
            tailgrid,
            [
                "variant",
                "family",
                "delta_net_pnl",
                "delta_full_sl_rate",
                "delta_q20",
                "tail_objective_delta",
            ],
            max_rows=10,
        ),
        "",
        "Expanding diagnostics:",
        "",
        _fmt_table(
            expanding,
            [
                "variant",
                "family",
                "delta_net_pnl",
                "delta_objective_week",
                "delta_full_sl_rate",
                "delta_q20_week_net_pnl",
                "delta_q35_week_net_pnl",
            ],
            max_rows=10,
        ),
        "",
        "## Decision",
        "",
        "- Development replay has research candidates with positive PnL/objective/tail metrics.",
        "- No candidate is deployment-ready until the frozen evidence gate has enough post-cutoff rows and timestamps.",
        "- The next operational step is to append future candidate ledgers into the cumulative flat ledger and rerun the readiness gate.",
    ]
    (args.output_dir / "contextual_tp_sl_ablation_dashboard.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "generated_by": Path(__file__).name,
        "inputs": {
            "promotion_dir": str(args.promotion_dir),
            "scorecard_dir": str(args.scorecard_dir),
            "readiness_dir": str(args.readiness_dir),
            "ledger_dir": str(args.ledger_dir),
        },
        "outputs": [
            "candidate_deployment_dashboard.csv",
            "candidate_monthly_consistency.csv",
            "candidate_head_consistency.csv",
            "contextual_tp_sl_ablation_dashboard.md",
        ],
        "ready_sources": ready_sources,
        "development_pass_count": int(dashboard.get("development_candidate_pass", pd.Series(dtype=bool)).sum())
        if not dashboard.empty
        else 0,
        "deployment_pass_count": int(dashboard.get("deployment_candidate_pass", pd.Series(dtype=bool)).sum())
        if not dashboard.empty
        else 0,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
