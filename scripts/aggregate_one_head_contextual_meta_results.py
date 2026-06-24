#!/usr/bin/env python3
"""Aggregate one-head contextual meta ablation outputs across heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_one_head_contextual_meta_ablation import _promotion_table


DEFAULT_INPUT_DIRS = (
    "data_perp/reports/one_head_contextual_meta_ablation_short_asset_full_20260622",
    "data_perp/reports/one_head_contextual_meta_ablation_remaining_heads_full_20260622",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _promotion_for_dir(
    path: Path,
    winner_epsilon: float,
    lower_tail_tolerance: float,
    directional_hr10_tolerance: float,
    directional_normal_tolerance: float,
) -> pd.DataFrame:
    existing = _read_csv(path / "one_head_contextual_meta_promotion_table.csv")
    if not existing.empty:
        existing["source_dir"] = str(path)
        return existing
    summary = _read_csv(path / "one_head_contextual_meta_ablation_summary.csv")
    period = _read_csv(path / "one_head_contextual_meta_period_conflict_diagnostics.csv")
    ci = _read_csv(path / "one_head_contextual_meta_episode_block_confidence_intervals.csv")
    gradient = _read_csv(path / "one_head_contextual_meta_gradient_conflict_diagnostics.csv")
    oracle = _read_csv(path / "one_head_contextual_meta_oracle_specialist_leave_one.csv")
    directional = _read_csv(path / "one_head_contextual_meta_directional_metrics.csv")
    directional_episode_ci = _read_csv(path / "one_head_contextual_meta_directional_episode_confidence_intervals.csv")
    promotion = _promotion_table(
        summary,
        period,
        ci,
        gradient,
        oracle,
        directional,
        directional_episode_ci,
        winner_epsilon=winner_epsilon,
        lower_tail_tolerance=lower_tail_tolerance,
        directional_hr10_tolerance=directional_hr10_tolerance,
        directional_normal_tolerance=directional_normal_tolerance,
    )
    if not promotion.empty:
        promotion["source_dir"] = str(path)
    return promotion


def _audit_for_dir(path: Path) -> dict[str, Any]:
    audit_path = path / "one_head_contextual_meta_ablation_requirement_audit.json"
    if not audit_path.exists():
        return {"source_dir": str(path), "status": "missing"}
    data = json.loads(audit_path.read_text())
    failed = [
        item.get("requirement")
        for item in data.get("items", [])
        if str(item.get("status", "")).lower() != "passed"
    ]
    return {
        "source_dir": str(path),
        "status": data.get("status", ""),
        "failed_requirements": failed,
        "outcomes": data.get("outcomes", {}),
    }


def _decision_rows(promotion: pd.DataFrame) -> pd.DataFrame:
    if promotion.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for head, group in promotion.groupby("head", sort=True):
        candidates = group.loc[group["arm"].astype(str).ne("A_current_meta_model")].copy()
        for col in (
            "directional_episode_median_delta_timestamp_weighted_hr_top30",
            "delta_timestamp_weighted_hr_top30",
            "delta_ndcg_top30",
            "delta_timestamp_weighted_hr_top20",
            "delta_timestamp_weighted_hr_top10",
            "top30_delta_log_loss_on_selected",
            "delta_log_loss_improvement",
        ):
            candidates[col] = pd.to_numeric(candidates.get(col), errors="coerce")
        candidates["promotion_candidate"] = candidates.get("promotion_candidate", False).astype(bool)
        sort_cols = [
            "directional_episode_median_delta_timestamp_weighted_hr_top30",
            "delta_timestamp_weighted_hr_top30",
            "delta_ndcg_top30",
            "delta_timestamp_weighted_hr_top20",
            "delta_timestamp_weighted_hr_top10",
            "top30_delta_log_loss_on_selected",
            "delta_log_loss_improvement",
        ]
        best_directional = candidates.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(1)
        best_candidate = candidates.loc[candidates["promotion_candidate"]].sort_values(
            sort_cols, ascending=[False] * len(sort_cols)
        ).head(1)
        best_directional_row = best_directional.iloc[0].to_dict() if not best_directional.empty else {}
        best_candidate_row = best_candidate.iloc[0].to_dict() if not best_candidate.empty else {}
        oracle_gap = pd.to_numeric(
            candidates.get("oracle_specialist_gap_log_loss", pd.Series(np.nan, index=candidates.index)),
            errors="coerce",
        )
        oracle_beats = bool((oracle_gap > 0.0).fillna(False).any())
        if best_candidate_row:
            decision = "contextual_single_head_candidate"
            selected = best_candidate_row
        elif best_directional_row and float(best_directional_row.get("delta_timestamp_weighted_hr_top30", np.nan)) > 0:
            decision = "directional_improves_but_recurrence_gate_fails"
            selected = best_directional_row
        else:
            decision = "keep_baseline_for_now"
            selected = best_directional_row
        if oracle_beats:
            architecture_note = "oracle_specialist_gap_positive_for_some_arm"
        else:
            architecture_note = "pooled_contextual_model_sufficient_vs_oracle"
        rows.append(
            {
                "head": head,
                "decision": decision,
                "selected_arm": selected.get("arm", ""),
                "selected_timestamp_weighted_hr_top30": selected.get("timestamp_weighted_hr_top30", np.nan),
                "selected_delta_timestamp_weighted_hr_top30": selected.get("delta_timestamp_weighted_hr_top30", np.nan),
                "selected_directional_episode_median_delta_hr_top30": selected.get(
                    "directional_episode_median_delta_timestamp_weighted_hr_top30", np.nan
                ),
                "selected_directional_episode_positive_rate_hr_top30": selected.get(
                    "directional_episode_positive_rate_delta_timestamp_weighted_hr_top30", np.nan
                ),
                "selected_ndcg_top30": selected.get("ndcg_top30", np.nan),
                "selected_delta_ndcg_top30": selected.get("delta_ndcg_top30", np.nan),
                "selected_delta_hr_top10": selected.get("delta_timestamp_weighted_hr_top10", np.nan),
                "selected_delta_hr_top20": selected.get("delta_timestamp_weighted_hr_top20", np.nan),
                "selected_worst_week_hr_top30": selected.get("worst_week_hr_top30", np.nan),
                "selected_q10_week_hr_top30": selected.get("q10_week_hr_top30", np.nan),
                "selected_net_correct_trades_gained": selected.get("net_correct_trades_gained", np.nan),
                "selected_delta_log_loss_improvement": selected.get("delta_log_loss_improvement", np.nan),
                "selected_delta_brier_improvement": selected.get("delta_brier_improvement", np.nan),
                "selected_top10_delta_mean_return": selected.get("top10_delta_mean_return", np.nan),
                "selected_top10_delta_winner_magnitude": selected.get("top10_delta_winner_magnitude", np.nan),
                "selected_top10_delta_lower_tail_return": selected.get("top10_delta_lower_tail_return", np.nan),
                "selected_episode_median_delta_log_loss": selected.get(
                    "episode_median_delta_log_loss_improvement", np.nan
                ),
                "selected_episode_positive_rate_log_loss": selected.get(
                    "episode_positive_rate_delta_log_loss_improvement", np.nan
                ),
                "selected_gradient_conflict_weighted": selected.get("gradient_conflict_weighted", np.nan),
                "selected_gradient_conflict_high_row_fraction": selected.get(
                    "gradient_conflict_high_row_fraction", np.nan
                ),
                "oracle_specialist_beats_any_pooled_arm": oracle_beats,
                "architecture_note": architecture_note,
            }
        )
    return pd.DataFrame(rows)


def _write_report(out_dir: Path, promotion: pd.DataFrame, decisions: pd.DataFrame, audits: list[dict[str, Any]]) -> None:
    lines = [
        "# All-Head Contextual Meta Ablation Aggregation",
        "",
        "Aggregates the verified one-head contextual meta experiments without changing labels or model outputs.",
        "",
        "## Source Audits",
        "",
    ]
    audit_rows = [
        {
            "source_dir": item["source_dir"],
            "status": item["status"],
            "failed_requirements": ",".join(item.get("failed_requirements", [])),
        }
        for item in audits
    ]
    lines.append(pd.DataFrame(audit_rows).to_markdown(index=False))
    lines.append("")
    if not decisions.empty:
        lines.append("## Architecture Decisions")
        lines.append("")
        lines.append(decisions.to_markdown(index=False, floatfmt=".6f"))
        lines.append("")
    if not promotion.empty:
        view_cols = [
            "head",
            "arm",
            "timestamp_weighted_hr_top30",
            "delta_timestamp_weighted_hr_top30",
            "directional_episode_median_delta_timestamp_weighted_hr_top30",
            "directional_episode_positive_rate_delta_timestamp_weighted_hr_top30",
            "ndcg_top30",
            "delta_ndcg_top30",
            "delta_timestamp_weighted_hr_top10",
            "delta_timestamp_weighted_hr_top20",
            "normal_period_delta_hr_top30",
            "bad_period_delta_hr_top30",
            "top30_delta_log_loss_on_selected",
            "top10_delta_mean_return",
            "top10_delta_winner_magnitude",
            "top10_delta_lower_tail_return",
            "passes_directional_pooled_constraints",
            "passes_directional_episode_constraints",
            "gradient_conflict_weighted",
            "gradient_conflict_high_row_fraction",
            "oracle_specialist_gap_log_loss",
            "promotion_candidate",
        ]
        lines.append("## Promotion Table")
        lines.append("")
        lines.append(promotion[[c for c in view_cols if c in promotion.columns]].to_markdown(index=False, floatfmt=".6f"))
        lines.append("")
    (out_dir / "all_head_contextual_meta_aggregation_report.md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = [Path(p) for p in args.input_dirs]
    promotions = [
        _promotion_for_dir(
            path,
            winner_epsilon=float(args.winner_epsilon),
            lower_tail_tolerance=float(args.lower_tail_tolerance),
            directional_hr10_tolerance=float(args.directional_hr10_tolerance),
            directional_normal_tolerance=float(args.directional_normal_tolerance),
        )
        for path in inputs
    ]
    promotion = pd.concat([df for df in promotions if not df.empty], axis=0, ignore_index=True)
    if not promotion.empty:
        for col in (
            "directional_episode_median_delta_timestamp_weighted_hr_top30",
            "delta_timestamp_weighted_hr_top30",
            "delta_ndcg_top30",
            "delta_timestamp_weighted_hr_top20",
            "delta_timestamp_weighted_hr_top10",
            "top30_delta_log_loss_on_selected",
            "delta_log_loss_improvement",
        ):
            promotion[col] = pd.to_numeric(promotion.get(col), errors="coerce")
        promotion = promotion.sort_values(
            [
                "head",
                "promotion_candidate",
                "directional_episode_median_delta_timestamp_weighted_hr_top30",
                "delta_timestamp_weighted_hr_top30",
                "delta_ndcg_top30",
                "delta_timestamp_weighted_hr_top20",
                "delta_timestamp_weighted_hr_top10",
            ],
            ascending=[True, False, False, False, False, False, False],
        )
    decisions = _decision_rows(promotion)
    audits = [_audit_for_dir(path) for path in inputs]
    promotion.to_csv(out_dir / "all_head_contextual_meta_promotion_table.csv", index=False)
    decisions.to_csv(out_dir / "all_head_contextual_meta_decisions.csv", index=False)
    (out_dir / "all_head_contextual_meta_aggregation_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "inputs": [str(p) for p in inputs],
                "promotion_rows": int(len(promotion)),
                "decision_rows": int(len(decisions)),
                "audits": audits,
                "winner_epsilon": float(args.winner_epsilon),
                "lower_tail_tolerance": float(args.lower_tail_tolerance),
                "directional_hr10_tolerance": float(args.directional_hr10_tolerance),
                "directional_normal_tolerance": float(args.directional_normal_tolerance),
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )
    _write_report(out_dir, promotion, decisions, audits)
    print(f"[aggregate_one_head_context] wrote results to {out_dir}", flush=True)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", default=list(DEFAULT_INPUT_DIRS))
    parser.add_argument("--output-dir", default="data_perp/reports/one_head_contextual_meta_all_heads_20260622")
    parser.add_argument("--winner-epsilon", type=float, default=0.0005)
    parser.add_argument("--lower-tail-tolerance", type=float, default=0.0010)
    parser.add_argument("--directional-hr10-tolerance", type=float, default=0.001)
    parser.add_argument("--directional-normal-tolerance", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
