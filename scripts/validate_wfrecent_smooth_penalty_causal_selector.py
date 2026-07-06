#!/usr/bin/env python3
"""Causal monthly selector for wf_recent smooth-penalty diagnostic rules.

The family-grid holdouts showed that OOD, uncertainty, and recent-HR surprise
can all help, but different windows prefer different families. This script
tests whether a small rule selector can choose among those rules using only
pre-holdout data, then apply the selected rule to the next chronological month.

This is a development validation, not a production promotion gate.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from scripts.freeze_apply_wfrecent_smooth_penalty_bundle import _sha256_file  # noqa: E402
from scripts.replay_wfrecent_smooth_rank_penalty_fixed import _per_head_table  # noqa: E402
from scripts.validate_wfrecent_row_guard_walkforward import (  # noqa: E402
    _apply_risk_scores,
    _fit_percentile_reference,
    _fmt_table,
    _head_name,
    _json_safe,
    _period_tables,
    _summary,
)
from scripts.validate_wfrecent_smooth_penalty_frozen_holdout import (  # noqa: E402
    _apply_rule,
    _load_candidates,
    _rules_for_mode,
)


def _month_starts(first: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    starts: list[pd.Timestamp] = []
    cur = first
    while cur < end:
        starts.append(cur)
        cur = pd.Timestamp(cur + pd.offsets.MonthBegin(1))
    return starts


def _add_rule_metadata(row: dict[str, Any], rule: Any) -> dict[str, Any]:
    row.update(
        {
            "score_name": rule.score_name,
            "scope": rule.scope,
            "risk_quantile": float(rule.risk_quantile),
            "min_rank_pct": float(rule.min_rank_pct),
            "max_penalty": float(rule.max_penalty),
            "power": float(rule.power),
        }
    )
    return row


def _score_rule_on_split(
    *,
    fit_raw: pd.DataFrame,
    target_raw: pd.DataFrame,
    rule_name: str,
    rule: Any,
    q35_weight: float,
    q20_weight: float,
    full_sl_penalty: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    refs = _fit_percentile_reference(fit_raw)
    fit_scored = _apply_risk_scores(fit_raw, refs)
    target_scored = _apply_risk_scores(target_raw, refs)
    ev_curve = fit_hierarchical_ev_curves(fit_scored)
    params = PortfolioPolicyParams(global_threshold_floor=0.0)

    base_decisions, _base_eq, base_metrics = replay_candidates(
        target_scored,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode="perps",
    )
    _base_daily, base_weekly = _period_tables(base_decisions)
    base_summary = _summary("baseline", base_decisions, base_weekly, base_metrics, q35_weight, q20_weight)

    adjusted, audit = _apply_rule(target_scored, fit_scored, rule)
    decisions, _eq, metrics = replay_candidates(
        adjusted,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode="perps",
    )
    _daily, weekly = _period_tables(decisions)
    cur_summary = _summary(rule_name, decisions, weekly, metrics, q35_weight, q20_weight)

    row: dict[str, Any] = {"variant": rule_name}
    for key in (
        "net_pnl",
        "gross_pnl",
        "trade_count",
        "hit_rate",
        "full_sl_rate",
        "timeout_rate",
        "max_drawdown",
        "objective_week",
        "q20_week_net_pnl",
        "q35_week_net_pnl",
        "worst_week_net_pnl",
        "positive_weeks",
    ):
        row[f"baseline_{key}"] = base_summary[key]
        row[f"challenger_{key}"] = cur_summary[key]
        row[f"delta_{key}"] = float(cur_summary[key] - base_summary[key])
    row["selection_score"] = float(
        row["delta_objective_week"] - float(full_sl_penalty) * max(float(row["delta_full_sl_rate"]), 0.0)
    )
    row.update(audit)
    _add_rule_metadata(row, rule)
    return row, base_decisions, decisions


def _score_baseline_on_split(
    *,
    fit_raw: pd.DataFrame,
    target_raw: pd.DataFrame,
    q35_weight: float,
    q20_weight: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    refs = _fit_percentile_reference(fit_raw)
    fit_scored = _apply_risk_scores(fit_raw, refs)
    target_scored = _apply_risk_scores(target_raw, refs)
    ev_curve = fit_hierarchical_ev_curves(fit_scored)
    params = PortfolioPolicyParams(global_threshold_floor=0.0)
    decisions, _eq, metrics = replay_candidates(
        target_scored,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode="perps",
    )
    _daily, weekly = _period_tables(decisions)
    return _summary("baseline", decisions, weekly, metrics, q35_weight, q20_weight), decisions


def _empty_month_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "holdout_start",
            "holdout_end",
            "selection_start",
            "selection_end",
            "variant",
            "selection_score",
            "delta_net_pnl",
            "delta_objective_week",
            "delta_q35_week_net_pnl",
            "delta_worst_week_net_pnl",
            "delta_hit_rate",
            "delta_full_sl_rate",
            "delta_trade_count",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_materialized_wf_recent_q35w07_q20w03_6mo_20260701/combo_candidates.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_causal_selector_20260701"),
    )
    parser.add_argument("--first-holdout", default="2026-04-01T00:00:00+00:00")
    parser.add_argument("--end", default="2026-06-27T00:00:00+00:00")
    parser.add_argument("--selection-months", type=int, default=2)
    parser.add_argument("--rule-mode", default="default_plus_diagnostic_family_grid")
    parser.add_argument(
        "--variants",
        default="",
        help="Optional comma-separated variant labels to evaluate after building the rule mode.",
    )
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    parser.add_argument("--full-sl-penalty", type=float, default=2500.0)
    parser.add_argument("--top-selection-rows", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = _load_candidates(args.candidates)
    first_holdout = pd.Timestamp(args.first_holdout, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    rules = _rules_for_mode(str(args.rule_mode))
    requested_variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    if requested_variants:
        missing = [item for item in requested_variants if item not in rules]
        if missing:
            raise ValueError(f"Requested variants are not available in {args.rule_mode}: {missing}")
        rules = {item: rules[item] for item in requested_variants}

    selected_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    per_head_rows: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []

    for holdout_start in _month_starts(first_holdout, end):
        holdout_end = min(pd.Timestamp(holdout_start + pd.offsets.MonthBegin(1)), end)
        selection_start = pd.Timestamp(holdout_start - pd.DateOffset(months=int(args.selection_months)))
        selection_end = holdout_start

        fit_for_selection = candidates[candidates["timestamp"].lt(selection_start)].copy().reset_index(drop=True)
        selection = candidates[
            candidates["timestamp"].ge(selection_start) & candidates["timestamp"].lt(selection_end)
        ].copy().reset_index(drop=True)
        fit_for_holdout = candidates[candidates["timestamp"].lt(holdout_start)].copy().reset_index(drop=True)
        holdout = candidates[
            candidates["timestamp"].ge(holdout_start) & candidates["timestamp"].lt(holdout_end)
        ].copy().reset_index(drop=True)
        if fit_for_selection.empty or selection.empty or fit_for_holdout.empty or holdout.empty:
            continue

        month_selection_rows: list[dict[str, Any]] = []
        for variant, rule in rules.items():
            row, _base_decisions, _rule_decisions = _score_rule_on_split(
                fit_raw=fit_for_selection,
                target_raw=selection,
                rule_name=variant,
                rule=rule,
                q35_weight=float(args.q35_weight),
                q20_weight=float(args.q20_weight),
                full_sl_penalty=float(args.full_sl_penalty),
            )
            row.update(
                {
                    "holdout_start": holdout_start.isoformat(),
                    "holdout_end": holdout_end.isoformat(),
                    "selection_start": selection_start.isoformat(),
                    "selection_end": selection_end.isoformat(),
                    "fit_rows": int(len(fit_for_selection)),
                    "selection_rows": int(len(selection)),
                }
            )
            month_selection_rows.append(row)
        ranked = sorted(month_selection_rows, key=lambda r: (float(r["selection_score"]), float(r["delta_net_pnl"])), reverse=True)
        selection_rows.extend(ranked[: max(1, int(args.top_selection_rows))])
        chosen = ranked[0]
        chosen_rule = rules[str(chosen["variant"])]

        holdout_row, base_decisions, challenger_decisions = _score_rule_on_split(
            fit_raw=fit_for_holdout,
            target_raw=holdout,
            rule_name=str(chosen["variant"]),
            rule=chosen_rule,
            q35_weight=float(args.q35_weight),
            q20_weight=float(args.q20_weight),
            full_sl_penalty=float(args.full_sl_penalty),
        )
        holdout_row.update(
            {
                "holdout_start": holdout_start.isoformat(),
                "holdout_end": holdout_end.isoformat(),
                "selection_start": selection_start.isoformat(),
                "selection_end": selection_end.isoformat(),
                "fit_rows": int(len(fit_for_holdout)),
                "holdout_rows": int(len(holdout)),
                "selected_by": "max_selection_score_then_delta_net_pnl",
                "selection_variant": chosen["variant"],
                "selection_score": float(chosen["selection_score"]),
                "selection_delta_net_pnl": float(chosen["delta_net_pnl"]),
                "selection_delta_objective_week": float(chosen["delta_objective_week"]),
                "selection_delta_full_sl_rate": float(chosen["delta_full_sl_rate"]),
            }
        )
        selected_rows.append(holdout_row)

        ph = _per_head_table(base_decisions, challenger_decisions)
        ph["holdout_start"] = holdout_start.isoformat()
        ph["holdout_end"] = holdout_end.isoformat()
        ph["variant"] = str(chosen["variant"])
        per_head_rows.append(ph)
        dec = challenger_decisions.copy()
        dec["holdout_start"] = holdout_start.isoformat()
        dec["holdout_end"] = holdout_end.isoformat()
        dec["variant"] = str(chosen["variant"])
        decision_frames.append(dec)

    selected = pd.DataFrame(selected_rows) if selected_rows else _empty_month_frame()
    selection_audit = pd.DataFrame(selection_rows) if selection_rows else _empty_month_frame()
    per_head = pd.concat(per_head_rows, ignore_index=True) if per_head_rows else pd.DataFrame()
    decisions = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()

    selected.to_csv(args.output_dir / "causal_selector_monthly.csv", index=False)
    selection_audit.to_csv(args.output_dir / "causal_selector_selection_audit.csv", index=False)
    per_head.to_csv(args.output_dir / "causal_selector_per_head.csv", index=False)
    if not decisions.empty:
        decisions.to_parquet(args.output_dir / "causal_selector_challenger_decisions.parquet", index=False)

    totals: dict[str, Any] = {}
    if not selected.empty:
        for key in (
            "delta_net_pnl",
            "delta_gross_pnl",
            "delta_trade_count",
            "delta_objective_week",
            "delta_q20_week_net_pnl",
            "delta_q35_week_net_pnl",
            "delta_worst_week_net_pnl",
        ):
            totals[f"sum_{key}"] = float(pd.to_numeric(selected[key], errors="coerce").sum())
        for key in ("delta_hit_rate", "delta_full_sl_rate", "delta_timeout_rate"):
            totals[f"mean_{key}"] = float(pd.to_numeric(selected[key], errors="coerce").mean())
        totals["positive_delta_net_pnl_share"] = float((pd.to_numeric(selected["delta_net_pnl"], errors="coerce") > 0.0).mean())
        totals["positive_delta_objective_share"] = float((pd.to_numeric(selected["delta_objective_week"], errors="coerce") > 0.0).mean())
        totals["months"] = int(len(selected))
    summary = pd.DataFrame([totals]) if totals else pd.DataFrame()
    summary.to_csv(args.output_dir / "causal_selector_summary.csv", index=False)

    manifest = {
        "generated_by": "validate_wfrecent_smooth_penalty_causal_selector",
        "candidate_source": str(args.candidates),
        "candidate_source_sha256": _sha256_file(args.candidates),
        "first_holdout": first_holdout.isoformat(),
        "end": end.isoformat(),
        "selection_months": int(args.selection_months),
        "rule_mode": str(args.rule_mode),
        "variants": requested_variants,
        "q35_weight": float(args.q35_weight),
        "q20_weight": float(args.q20_weight),
        "full_sl_penalty": float(args.full_sl_penalty),
        "rules": {name: rule.__dict__ for name, rule in rules.items()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")

    lines = [
        "# wf_recent Smooth-Penalty Causal Selector",
        "",
        "For each monthly holdout, this selects one smooth-penalty rule using only a trailing pre-holdout selection window, then refits references on all pre-holdout rows and replays the selected rule on the holdout month.",
        "",
        f"Rule mode: `{args.rule_mode}`",
        f"Selection months: `{int(args.selection_months)}`",
        f"Full-SL penalty in selection score: `{float(args.full_sl_penalty):.3f}`",
        "",
        "## Summary",
        "",
        _fmt_table(
            summary,
            [
                "months",
                "sum_delta_net_pnl",
                "sum_delta_objective_week",
                "sum_delta_q35_week_net_pnl",
                "sum_delta_worst_week_net_pnl",
                "mean_delta_hit_rate",
                "mean_delta_full_sl_rate",
                "sum_delta_trade_count",
                "positive_delta_net_pnl_share",
                "positive_delta_objective_share",
            ],
        ),
        "",
        "## Monthly Choices And Holdout Results",
        "",
        _fmt_table(
            selected,
            [
                "holdout_start",
                "selection_variant",
                "score_name",
                "scope",
                "selection_score",
                "selection_delta_net_pnl",
                "selection_delta_objective_week",
                "delta_net_pnl",
                "delta_objective_week",
                "delta_q35_week_net_pnl",
                "delta_worst_week_net_pnl",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_trade_count",
            ],
        ),
        "",
        "## Per-Head Holdout Deltas",
        "",
        _fmt_table(
            per_head,
            [
                "holdout_start",
                "variant",
                "head",
                "delta_net_pnl",
                "delta_gross_pnl",
                "delta_trades",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_timeout_rate",
            ],
        ),
    ]
    (args.output_dir / "causal_selector_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
