#!/usr/bin/env python3
"""Build a consolidated review of the market-state / T1 workstream.

The report is intentionally read-only.  It aggregates already-produced
artifacts so the current state of the experiment can be inspected without
rerunning memory-heavy replay or training jobs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OPERATIONAL_STATUS = Path(
    "data_perp/reports/market_state_operational_status_20260627_v15_pruned_no_backfill_audited/"
    "market_state_operational_status.json"
)
DEFAULT_JUNE_DESIGN = Path("data_perp/reports/short_asset_short_boll_june_design_20260625_rerun")
DEFAULT_RANK_COMPARISON = Path(
    "data_perp/reports/t1_rank_contract_comparison_20260626_timestamp_vs_global_strict_rankref_v1"
)
DEFAULT_PREJUNE_WALKFORWARD = Path(
    "data_perp/reports/t1_rank_contract_walkforward_20260626_prejune_timestamp_vs_global_v4_timestamp_utility"
)
DEFAULT_BACKEND_COMPARISON = Path(
    "data_perp/reports/market_state_backend_comparison_20260626_t1_lgbm_vs_xgb_maturity_contract_v2"
)
DEFAULT_SHADOW_MONITOR = Path(
    "data_perp/reports/market_state_shadow_controller_monitor_t1_lgbm_maturity_shadow_s2_20260626_v2_all_current"
)
DEFAULT_INSTABILITY = Path(
    "data_perp/reports/performance_regime_instability_explanation_20260628_badregime_v2"
)
DEFAULT_SHALLOW_PILOT = Path(
    "data_perp/reports/performance_regime_instability_explanation_20260628_shallow12_24_short_asset_pilot_v2"
)
DEFAULT_SAFE_ALLHEAD = Path(
    "data_perp/reports/performance_regime_instability_explanation_20260628_shallow12_24_allheads_v1"
)
DEFAULT_CONTRACT_AUDIT = Path(
    "data_perp/reports/market_state_goal_review_20260628_v3/globalrank_no_backfill_pruned_contract_audit.json"
)
DEFAULT_STATE_HEAD_PRUNING_AUDIT = Path(
    "data_perp/reports/market_state_state_head_pruning_audit_globalrank_no_backfill_pruned_20260628_v1"
)
DEFAULT_STRATEGY_RESPONSE_QUALITY = Path(
    "data_perp/reports/market_state_strategy_response_quality_globalrank_no_backfill_pruned_20260628_v1"
)
DEFAULT_PROMOTION_GATE_AUDIT = Path(
    "data_perp/reports/market_state_controller_promotion_gate_audit_globalrank_no_backfill_pruned_20260628_v2"
)
DEFAULT_PLAN_COMPLETION_AUDIT = Path(
    "data_perp/reports/market_state_plan_completion_audit_globalrank_no_backfill_pruned_20260628_v4"
)
DEFAULT_DIRECT_SUPPRESSION_ACTIONABILITY_AUDIT = Path(
    "data_perp/reports/market_state_direct_suppression_actionability_audit_globalrank_no_backfill_20260628_v1"
)
DEFAULT_NEXT_NO_BACKFILL_READINESS = Path(
    "data_perp/reports/market_state_next_no_backfill_shadow_window_readiness_globalrank_20260628_v6"
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fh:
        raw = json.load(fh)
    return raw if isinstance(raw, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _md_table(frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> list[str]:
    if frame.empty:
        return ["_No artifact rows found._"]
    out = frame.copy()
    if columns is not None:
        out = out.loc[:, [col for col in columns if col in out.columns]]
    if max_rows is not None:
        out = out.head(max_rows)
    if out.empty:
        return ["_No requested columns found._"]
    headers = list(out.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(_fmt(row.get(col)) for col in headers) + " |")
    return lines


def _arm_row(frame: pd.DataFrame, arm: str) -> pd.Series:
    if frame.empty or "arm" not in frame.columns:
        return pd.Series(dtype=object)
    rows = frame.loc[frame["arm"].astype(str).eq(arm)]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def _delta_row(base: pd.Series, challenger: pd.Series) -> dict[str, Any]:
    metrics = ["trade_count", "win_rate", "net_pnl", "gross_pnl", "cost_pnl", "full_sl_rate", "timeout_rate"]
    out: dict[str, Any] = {"comparison": "challenger_minus_base"}
    for metric in metrics:
        if metric in base.index and metric in challenger.index:
            out[f"delta_{metric}"] = pd.to_numeric(pd.Series([challenger[metric]]), errors="coerce").iloc[0] - pd.to_numeric(
                pd.Series([base[metric]]), errors="coerce"
            ).iloc[0]
    return out


def build_review(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    status = _read_json(args.operational_status)
    june = _read_csv(args.june_design / "portfolio_summary.csv")
    june_by_head = _read_csv(args.june_design / "portfolio_summary_by_head.csv")
    short_boll_validation = _read_csv(args.june_design / "short_boll_validation.csv")
    rank_summary = _read_csv(args.rank_comparison / "rank_contract_summary.csv")
    rank_by_head = _read_csv(args.rank_comparison / "rank_contract_by_head.csv")
    rank_delta = _read_csv(args.rank_comparison / "rank_contract_delta.csv")
    walkforward_agg = _read_csv(args.prejune_walkforward / "rank_contract_walkforward_aggregate.csv")
    backend = _read_csv(args.backend_comparison / "controller_selection_comparison.csv")
    prospective = _read_csv(args.backend_comparison / "prospective_increment_comparison.csv")
    shadow_summary = _read_json(args.shadow_monitor / "shadow_controller_monitor_summary.json")
    shadow_by_head = _read_csv(args.shadow_monitor / "shadow_controller_monitor_by_head.csv")
    instability = _read_csv(args.instability_report / "instability_explanation_by_strategy.csv")
    top_features = _read_csv(args.instability_report / "instability_top_leaf_features.csv")
    top_interactions = _read_csv(args.instability_report / "instability_top_interactions.csv")
    shallow_pilot = _read_csv(args.shallow_pilot_report / "instability_explanation_by_strategy.csv")
    shallow_pilot_features = _read_csv(args.shallow_pilot_report / "instability_top_leaf_features.csv")
    safe_allhead = _read_csv(args.safe_allhead_report / "instability_explanation_by_strategy.csv")
    safe_allhead_coverage = _read_csv(args.safe_allhead_report / "instability_feature_family_coverage.csv")
    safe_allhead_features = _read_csv(args.safe_allhead_report / "instability_top_leaf_features.csv")
    contract_audit = _read_json(args.contract_audit)
    promotion_gate = _read_json(args.promotion_gate_audit / "market_state_controller_promotion_gate_audit.json")
    suppression_actionability = _read_json(
        args.direct_suppression_actionability_audit / "direct_suppression_actionability_audit.json"
    )
    next_readiness = _read_json(
        args.next_no_backfill_readiness / "next_no_backfill_shadow_window_readiness.json"
    )
    plan_completion = _read_json(args.plan_completion_audit / "market_state_plan_completion_audit.json")
    pruning_audit = _read_json(args.state_head_pruning_audit / "market_state_state_head_pruning_audit.json")
    pruning_table = _read_csv(args.state_head_pruning_audit / "market_state_state_head_pruning_audit.csv")
    response_quality = _read_json(args.strategy_response_quality / "market_state_strategy_response_quality_gate.json")
    response_quality_by_arm = _read_csv(
        args.strategy_response_quality / "market_state_strategy_response_quality_by_arm.csv"
    )
    response_quality_by_head = _read_csv(
        args.strategy_response_quality / "market_state_strategy_response_quality_by_head.csv"
    )

    d0 = _arm_row(june, "D0_A0_anchor_only")
    d2 = _arm_row(june, "D2_A0_plus_short_boll_timestamp_rank")
    d2_delta = _delta_row(d0, d2) if not d0.empty and not d2.empty else {}

    backend_view = backend.copy()
    if not backend_view.empty:
        backend_view = backend_view.loc[
            ~backend_view.get("is_baseline", pd.Series(False, index=backend_view.index)).fillna(False).astype(bool)
        ]
        backend_view = backend_view[
            [
                col
                for col in [
                    "backend",
                    "arm",
                    "folds",
                    "median_delta_net_pnl",
                    "q25_delta_net_pnl",
                    "positive_delta_share",
                    "mean_delta_net_pnl",
                    "realized_defensive_success",
                    "passed_selection_gates",
                    "selection_fail_reasons",
                ]
                if col in backend_view.columns
            ]
        ]

    short_boll_all = (
        short_boll_validation.loc[short_boll_validation.get("slice", "").astype(str).eq("all")]
        if not short_boll_validation.empty and "slice" in short_boll_validation.columns
        else pd.DataFrame()
    )

    lines: list[str] = [
        "# Market-State / T1 Goal Review",
        "",
        "This report aggregates existing artifacts only; it does not retrain models or replay portfolios.",
        "",
        "## Active Operational Contract",
        "",
    ]
    active_rows = pd.DataFrame(
        [
            {
                "stack": status.get("active_stack_name"),
                "score": status.get("active_score_column"),
                "policy": status.get("active_policy_variant"),
                "rank_contract": status.get("active_rank_contract"),
                "rank_scope": status.get("active_rank_scope"),
                "active_heads": ",".join(map(str, status.get("active_heads", []))),
                "disabled_heads": ",".join(map(str, status.get("disabled_heads", []))),
                "qfail_active": status.get("qfail_active"),
                "threshold_controller_active": status.get("threshold_controller_active"),
                "market_state_shadow_only": status.get("market_state_shadow_logging_only"),
            }
        ]
    )
    lines.extend(_md_table(active_rows))
    lines.extend(
        [
            "",
            f"- Experiment attribution note: {status.get('experiment_attribution_note', '')}",
            "",
            "## June T1 Attribution",
            "",
            "The historical June design artifact explains why short_boll repair mattered. It remains separate from the active global-rank contract unless explicitly selected by a later promotion gate.",
            "",
        ]
    )
    lines.extend(
        _md_table(
            june,
            columns=["arm", "trade_count", "win_rate", "net_pnl", "gross_pnl", "cost_pnl", "full_sl_rate", "timeout_rate"],
            max_rows=8,
        )
    )
    if d2_delta:
        lines.extend(["", "### D2 Minus D0", ""])
        lines.extend(_md_table(pd.DataFrame([d2_delta])))
    lines.extend(["", "### D2 By Head", ""])
    lines.extend(
        _md_table(
            june_by_head.loc[june_by_head.get("arm", "").astype(str).eq("D2_A0_plus_short_boll_timestamp_rank")]
            if not june_by_head.empty
            else june_by_head,
            columns=["head", "trade_count", "win_rate", "net_pnl", "gross_pnl", "cost_pnl", "full_sl_rate", "timeout_rate"],
        )
    )
    lines.extend(["", "### Short-Boll Eligibility Evidence", ""])
    lines.extend(
        _md_table(
            short_boll_all,
            columns=[
                "contract",
                "rows",
                "timestamps",
                "rank_min",
                "rank_median",
                "rank_max",
                "rank_ge_070",
                "win_rate",
                "mean_net",
                "sum_net",
                "q05_net",
            ],
        )
    )

    lines.extend(["", "## Rank-Contract Evidence", ""])
    lines.extend(
        _md_table(
            rank_summary,
            columns=[
                "contract_name",
                "trade_count",
                "net_pnl",
                "gross_pnl",
                "cost_pnl",
                "full_sl_rate",
                "timeout_rate",
                "worst_24h_net_pnl",
            ],
        )
    )
    if not rank_delta.empty:
        lines.extend(["", "### Global Strict Rank Minus Timestamp Rank", ""])
        lines.extend(_md_table(rank_delta))
    lines.extend(["", "### Rank Contract By Head", ""])
    lines.extend(
        _md_table(
            rank_by_head,
            columns=["contract_name", "head", "trade_count", "win_rate", "net_pnl", "full_sl_rate", "timeout_rate"],
            max_rows=8,
        )
    )
    lines.extend(["", "### Pre-June Walk-Forward", ""])
    lines.extend(_md_table(walkforward_agg))

    lines.extend(["", "## Global-Rank Controller Contract Audit", ""])
    if contract_audit:
        audit_rows = pd.DataFrame(
            [
                {
                    "artifact": contract_audit.get("artifact_dir"),
                    "kind": contract_audit.get("artifact_kind"),
                    "scope": contract_audit.get("audit_scope"),
                    "expected_rank_contract": contract_audit.get("expected_rank_contract"),
                    "passed": contract_audit.get("passed"),
                    "completion_grade_passed": contract_audit.get("completion_grade_passed"),
                    "failure_count": len(contract_audit.get("failures") or []),
                }
            ]
        )
        lines.extend(_md_table(audit_rows))
    else:
        lines.append("_No contract audit artifact found._")

    lines.extend(["", "## Controller Promotion Gate", ""])
    if promotion_gate:
        attribution = promotion_gate.get("action_attribution_gate") or {}
        promotion_rows = pd.DataFrame(
            [
                {
                    "passed": promotion_gate.get("passed"),
                    "promotion_gate_passed": promotion_gate.get("promotion_gate_passed"),
                    "action_attribution_gate_passed": attribution.get("passed"),
                    "controller_promotion_ready": promotion_gate.get("controller_promotion_ready"),
                    "controller_should_remain_disabled": promotion_gate.get("controller_should_remain_disabled"),
                    "expected_selected_arm": promotion_gate.get("expected_selected_arm"),
                    "attribution_failures": ";".join(map(str, attribution.get("failures") or [])),
                }
            ]
        )
        lines.extend(_md_table(promotion_rows))
        best = promotion_gate.get("best_raw_candidate") or {}
        best_rows = pd.DataFrame(
            [
                {
                    "arm": best.get("arm"),
                    "median_delta_net_pnl": best.get("median_delta_net_pnl"),
                    "q25_delta_net_pnl": best.get("q25_delta_net_pnl"),
                    "positive_delta_share": best.get("positive_delta_share"),
                    "realized_defensive_success": best.get("realized_defensive_success"),
                    "positive_suppression_fold_share": best.get("positive_suppression_fold_share"),
                    "freed_capacity_entrant_count": best.get("freed_capacity_entrant_count"),
                    "freed_capacity_net_action_pnl_delta": best.get("freed_capacity_net_action_pnl_delta"),
                    "direct_suppression_value_share": best.get("direct_suppression_value_share"),
                    "replacement_dependent_lift": best.get("replacement_dependent_lift"),
                }
            ]
        )
        lines.extend(["", "Best raw candidate attribution:", ""])
        lines.extend(_md_table(best_rows))
    else:
        lines.append("_No controller promotion-gate audit found._")

    lines.extend(["", "## Direct Suppression Actionability", ""])
    if suppression_actionability:
        actionability_rows = pd.DataFrame(
            [
                {
                    "selected_arm": suppression_actionability.get("selected_arm"),
                    "selection_reason": suppression_actionability.get("selection_reason"),
                    "dominant_blocker": suppression_actionability.get("dominant_blocker"),
                    "oof_auc": suppression_actionability.get("oof_probability_auc"),
                    "oof_ap": suppression_actionability.get("oof_average_precision"),
                    "oof_utility_spearman": suppression_actionability.get("oof_utility_spearman"),
                    "policy_grid_rows": suppression_actionability.get("policy_grid_rows"),
                    "passing_policy_rows": suppression_actionability.get("passing_policy_rows"),
                    "positive_suppression_policy_rows": suppression_actionability.get(
                        "positive_suppression_policy_rows"
                    ),
                    "recurrent_support_policy_rows": suppression_actionability.get(
                        "recurrent_support_policy_rows"
                    ),
                    "max_suppressed_rows": suppression_actionability.get("max_suppressed_rows"),
                    "max_suppressed_folds": suppression_actionability.get("max_suppressed_folds"),
                    "max_recurrent_defensive_success": suppression_actionability.get(
                        "max_recurrent_defensive_success"
                    ),
                    "max_recurrent_positive_fold_share": suppression_actionability.get(
                        "max_recurrent_positive_fold_share"
                    ),
                }
            ]
        )
        lines.extend(_md_table(actionability_rows))
        lines.extend(["", f"Interpretation: {suppression_actionability.get('interpretation')}"])
    else:
        lines.append("_No direct-suppression actionability audit found._")

    lines.extend(["", "## Next Shadow Window Readiness", ""])
    if next_readiness:
        readiness_rows = pd.DataFrame(
            [
                {
                    "status": next_readiness.get("status"),
                    "scoreable_min_window_now": next_readiness.get("scoreable_min_window_now"),
                    "scoreable_full_window_now": next_readiness.get("scoreable_full_window_now"),
                    "next_window_start": next_readiness.get("next_window_start"),
                    "minimum_window_end": next_readiness.get("minimum_window_end"),
                    "target_window_end": next_readiness.get("target_window_end"),
                    "mature_timestamp_count_available": next_readiness.get(
                        "mature_timestamp_count_available"
                    ),
                    "min_timestamp_count": next_readiness.get("min_timestamp_count"),
                    "minimum_feature_coverage_ready": next_readiness.get(
                        "minimum_window_feature_coverage_ready"
                    ),
                    "full_feature_coverage_ready": next_readiness.get(
                        "full_window_feature_coverage_ready"
                    ),
                    "minimum_low_coverage_hours": (
                        (next_readiness.get("minimum_window_feature_coverage") or {}).get(
                            "low_coverage_timestamp_count"
                        )
                    ),
                    "full_low_coverage_hours": (
                        (next_readiness.get("full_window_feature_coverage") or {}).get(
                            "low_coverage_timestamp_count"
                        )
                    ),
                    "missing_feature_hours_for_min_window": next_readiness.get(
                        "missing_feature_hours_for_min_window"
                    ),
                    "missing_feature_hours_for_full_window": next_readiness.get(
                        "missing_feature_hours_for_full_window"
                    ),
                    "coverage_repair_action": next_readiness.get("coverage_repair_action"),
                    "next_action": next_readiness.get("next_action"),
                }
            ]
        )
        lines.extend(_md_table(readiness_rows))
        min_cov = next_readiness.get("minimum_window_feature_coverage") or {}
        full_cov = next_readiness.get("full_window_feature_coverage") or {}
        lines.extend(
            [
                "",
                "Coverage gap classification:",
                "",
            ]
        )
        lines.extend(
            _md_table(
                pd.DataFrame(
                    [
                        {
                            "window": "minimum",
                            "low_gap_counts": min_cov.get("low_coverage_gap_type_counts"),
                            "blocking_gap_counts": min_cov.get(
                                "blocking_low_coverage_gap_type_counts"
                            ),
                            "blocking_timestamps_sample": ",".join(
                                map(str, min_cov.get("blocking_low_coverage_timestamps_sample") or [])
                            ),
                        },
                        {
                            "window": "full",
                            "low_gap_counts": full_cov.get("low_coverage_gap_type_counts"),
                            "blocking_gap_counts": full_cov.get(
                                "blocking_low_coverage_gap_type_counts"
                            ),
                            "blocking_timestamps_sample": ",".join(
                                map(str, full_cov.get("blocking_low_coverage_timestamps_sample") or [])
                            ),
                        },
                    ]
                )
            )
        )
        lines.extend(["", f"Interpretation: {next_readiness.get('interpretation')}"])
        if next_readiness.get("failures"):
            lines.extend(["", f"Failures: `{';'.join(map(str, next_readiness.get('failures') or []))}`"])
    else:
        lines.append("_No next no-backfill shadow-window readiness audit found._")

    lines.extend(["", "## Plan Completion Audit", ""])
    if plan_completion:
        status_counts = plan_completion.get("status_counts") or {}
        completion_rows = pd.DataFrame(
            [
                {
                    "passed_structural_audit": plan_completion.get("passed_structural_audit"),
                    "hard_failure_count": plan_completion.get("hard_failure_count"),
                    "complete": status_counts.get("complete"),
                    "gate_blocked": status_counts.get("gate_blocked"),
                    "partial": status_counts.get("partial"),
                    "shadow_only": status_counts.get("shadow_only"),
                    "controller_promotion_ready": plan_completion.get("controller_promotion_ready"),
                    "action_attribution_gate_passed": plan_completion.get(
                        "controller_action_attribution_gate_passed"
                    ),
                    "expected_rank_contract": plan_completion.get("expected_rank_contract"),
                }
            ]
        )
        lines.extend(_md_table(completion_rows))
        blockers = pd.DataFrame(plan_completion.get("gate_blocked_requirements") or [])
        if not blockers.empty:
            lines.extend(["", "Gate-blocked plan requirements:", ""])
            lines.extend(
                _md_table(
                    blockers,
                    columns=[
                        "requirement_id",
                        "section",
                        "requirement",
                        "status",
                        "notes",
                    ],
                    max_rows=10,
                )
            )
        shadow_only = pd.DataFrame(plan_completion.get("shadow_only_requirements") or [])
        if not shadow_only.empty:
            lines.extend(["", "Shadow-only plan requirements:", ""])
            lines.extend(
                _md_table(
                    shadow_only,
                    columns=[
                        "requirement_id",
                        "section",
                        "requirement",
                        "status",
                        "notes",
                    ],
                    max_rows=10,
                )
            )
    else:
        lines.append("_No plan-completion audit found._")

    lines.extend(["", "## Strategy Response And State-Head Gates", ""])
    if response_quality:
        response_rows = pd.DataFrame(
            [
                {
                    "passed": response_quality.get("passed"),
                    "quality_gate_passed": response_quality.get("quality_gate_passed"),
                    "controller_activation_allowed": response_quality.get("controller_activation_allowed"),
                    "quality_passing_arms": ",".join(map(str, response_quality.get("quality_passing_arms", []))),
                    "quality_passing_heads": ",".join(map(str, response_quality.get("quality_passing_heads", []))),
                    "response_rows": response_quality.get("response_rows"),
                    "duplicate_rows": response_quality.get("duplicate_rows"),
                }
            ]
        )
        lines.extend(_md_table(response_rows))
    else:
        lines.append("_No strategy-response quality audit found._")
    lines.extend(["", "### Response Quality By Arm", ""])
    lines.extend(
        _md_table(
            response_quality_by_arm,
            columns=[
                "arm",
                "heads",
                "passed_heads",
                "rows_total",
                "median_utility_spearman",
                "min_q25_utility_spearman",
                "median_utility_decile_spread",
                "min_q25_utility_decile_spread",
                "mean_state_ood_share",
                "all_heads_passed_response_quality",
            ],
        )
    )
    lines.extend(["", "### Response Quality By Head", ""])
    lines.extend(
        _md_table(
            response_quality_by_head,
            columns=[
                "arm",
                "head",
                "rows_total",
                "timestamp_count_total",
                "mean_state_ood_share",
                "median_utility_spearman",
                "q25_utility_spearman",
                "median_utility_decile_spread",
                "q25_utility_decile_spread",
                "median_full_sl_calibration_error",
                "median_timeout_calibration_error",
                "response_quality_passed",
            ],
        )
    )
    lines.extend(["", "### State-Head Pruning Audit", ""])
    if pruning_audit:
        pruning_rows = pd.DataFrame(
            [
                {
                    "passed": pruning_audit.get("passed"),
                    "registry_rows": pruning_audit.get("registry_rows"),
                    "active_candidate_count": pruning_audit.get("active_candidate_count"),
                    "disabled_candidate_count": pruning_audit.get("disabled_candidate_count"),
                    "shadow_count": pruning_audit.get("shadow_count"),
                    "disable_reasons": ";".join(
                        f"{key}:{value}"
                        for key, value in sorted((pruning_audit.get("disable_reason_counts") or {}).items())
                    ),
                }
            ]
        )
        lines.extend(_md_table(pruning_rows))
    else:
        lines.append("_No state-head pruning audit found._")
    lines.extend(["", "Worst state heads by leave-one-out q25:", ""])
    lines.extend(
        _md_table(
            pruning_table.sort_values(
                ["loo_q25_increment_net_pnl", "loo_median_increment_net_pnl"],
                ascending=[True, True],
            )
            if not pruning_table.empty
            else pruning_table,
            columns=[
                "state_head",
                "recommended_status",
                "activation_disable_reason",
                "loo_median_increment_net_pnl",
                "loo_q25_increment_net_pnl",
                "loo_positive_increment_share",
                "loo_state_head_defensive_success",
                "loo_state_head_loss_avoided",
                "loo_state_head_winner_pnl_sacrificed",
                "max_abs_spearman_corr",
                "redundant_with",
            ],
            max_rows=12,
        )
    )

    lines.extend(["", "## LGBM vs XGB Market-State Controller Evidence", ""])
    lines.extend(_md_table(backend_view, max_rows=12))
    lines.extend(["", "### Prospective Forecast Increment", ""])
    lines.extend(_md_table(prospective))

    lines.extend(["", "## Shadow Controller Monitor", ""])
    shadow_rows = pd.DataFrame(
        [
            {
                "bundle_count": shadow_summary.get("bundle_count"),
                "promotion_ready": shadow_summary.get("promotion_ready"),
                "defensive_positive_bundle_share": shadow_summary.get("defensive_positive_bundle_share"),
                "total_replay_trades": shadow_summary.get("total_replay_trades"),
                "total_replay_net_pnl": shadow_summary.get("total_replay_net_pnl"),
                "total_shadow_suppressed_candidates": shadow_summary.get("total_shadow_suppressed_candidates"),
                "loss_avoided": shadow_summary.get("total_shadow_loss_avoided"),
                "winner_pnl_sacrificed": shadow_summary.get("total_shadow_winner_pnl_sacrificed"),
                "defensive_success": shadow_summary.get("total_shadow_realized_defensive_success"),
                "failures": ",".join(map(str, shadow_summary.get("shadow_promotion_failures", []))),
            }
        ]
    )
    lines.extend(_md_table(shadow_rows))
    lines.extend(["", "### Shadow Monitor By Head", ""])
    lines.extend(
        _md_table(
            shadow_by_head,
            columns=[
                "head",
                "trade_count",
                "win_rate",
                "net_pnl",
                "full_sl_rate",
                "shadow_suppressed_candidates",
                "shadow_loss_avoided",
                "shadow_winner_pnl_sacrificed",
                "shadow_realized_defensive_success",
            ],
            max_rows=12,
        )
    )

    lines.extend(["", "## Historical Instability Explanation", ""])
    lines.extend(
        _md_table(
            instability,
            columns=[
                "strategy",
                "timestamp_count",
                "mean_strategy_performance",
                "negative_performance_share",
                "bad_label_ge_075_share",
                "explained_instability_share",
                "unexplained_instability_share",
                "first_stage_mean_oof_brier",
                "first_stage_median_prediction_std",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Explained instability share is clipped OOF weighted R2 of the timestamp-level bad-performance model. It is a learnability diagnostic, not executable PnL.",
            "",
            "### Shallow 12/24h Tail-Focused Pilot",
            "",
            "This pilot used short 12/24h bad-regime windows, shallow first-stage trees, no feedback pass, and capped archetype experts. It is a runtime-safe diagnostic, not a promoted controller.",
            "",
        ]
    )
    lines.extend(
        _md_table(
            shallow_pilot,
            columns=[
                "strategy",
                "timestamp_count",
                "bad_label_mean",
                "bad_label_ge_075_share",
                "composite_bad_pressure_share",
                "explained_instability_share",
                "unexplained_instability_share",
                "first_stage_mean_oof_brier",
                "first_stage_median_prediction_std",
            ],
        )
    )
    lines.extend(["", "Top pilot leaf features:", ""])
    lines.extend(
        _md_table(
            shallow_pilot_features,
            columns=["strategy", "feature", "weighted_score", "leaf_count", "fold_count"],
            max_rows=10,
        )
    )
    lines.extend(
        [
            "",
            "### Safe All-Head 12/24h Tail-Focused Run",
            "",
            "This run used the same memory-safe 12/24h tail-focused settings across all four heads. It is still diagnostic: it measures how much recent poor-performance structure is learnable from timestamp-level latent/context features.",
            "",
        ]
    )
    lines.extend(
        _md_table(
            safe_allhead,
            columns=[
                "strategy",
                "timestamp_count",
                "bad_label_mean",
                "bad_label_ge_075_share",
                "composite_bad_pressure_share",
                "explained_instability_share",
                "unexplained_instability_share",
                "first_stage_mean_oof_brier",
                "first_stage_median_prediction_std",
            ],
        )
    )
    lines.extend(["", "Safe all-head feature coverage:", ""])
    lines.extend(
        _md_table(
            safe_allhead_coverage,
            columns=[
                "strategy",
                "family",
                "requested_feature_count",
                "available_feature_count",
                "missing_feature_count",
                "missing_share",
                "fold_count",
            ],
        )
    )
    lines.extend(["", "Top safe all-head leaf features:", ""])
    safe_feature_rows = []
    if not safe_allhead_features.empty:
        for strategy, frame in safe_allhead_features.groupby("strategy", sort=True):
            safe_feature_rows.append(frame.head(5))
    lines.extend(
        _md_table(
            pd.concat(safe_feature_rows, ignore_index=True) if safe_feature_rows else pd.DataFrame(),
            columns=["strategy", "feature", "weighted_score", "leaf_count", "fold_count"],
            max_rows=24,
        )
    )
    lines.extend(
        [
            "",
            "### Top Leaf Features By Strategy",
            "",
        ]
    )
    feature_rows = []
    if not top_features.empty:
        for strategy, frame in top_features.groupby("strategy", sort=True):
            feature_rows.append(frame.head(5))
    lines.extend(
        _md_table(
            pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame(),
            columns=["strategy", "feature", "weighted_score", "leaf_count", "fold_count"],
            max_rows=40,
        )
    )
    lines.extend(["", "### Top Leaf-Guided Interactions By Strategy", ""])
    interaction_rows = []
    if not top_interactions.empty:
        for strategy, frame in top_interactions.groupby("strategy", sort=True):
            interaction_rows.append(frame.head(5))
    lines.extend(
        _md_table(
            pd.concat(interaction_rows, ignore_index=True) if interaction_rows else pd.DataFrame(),
            columns=["strategy", "kind", "features", "candidate_score", "fold_count", "source_leaf_count"],
            max_rows=40,
        )
    )

    lines.extend(
        [
            "",
            "## Current Interpretation",
            "",
            "- The strongest measured portfolio improvement came from repaired short_boll eligibility/ranking, not from executing the market-state controller.",
            "- LGBM and XGB state models contain useful state information, but the executable suppression overlay has not passed recurrence/defensive-success gates.",
            "- The strategy-response audit passes for both active heads across S1, S2, and S7; residual utility is learnable after conditioning on rank and state.",
            "- The state-head pruning audit leaves zero active candidates: all heads fail the leave-one-out action gate, and most direct actions sacrifice more winners than losses avoided.",
            "- The current bottleneck is therefore the controller action mapping and promotion objective, not the existence of state signal.",
            "- The next no-backfill shadow window is not currently scoreable if the readiness audit reports sparse mature feature coverage; collect or generate the missing feature history before running another shadow score.",
            "- Per-head instability is only partly explained by the current state pack; short_asset is most explainable, while long_bars remains almost unexplained.",
            "- The next controller experiment should remain per-strategy, rank-conditioned, and penalty-only, but should optimize direct suppression/frontier utility rather than full-replay gains that can be driven by replacement/backfill effects.",
            "- The T1 rank-contract line must remain explicit: historical timestamp-rank evidence and active global-over-time contract are not interchangeable.",
            "",
            "## Safe Next Run",
            "",
            "For additional instability diagnostics, use the existing runner with shallow first-stage trees and shorter bad-regime windows, ideally one head at a time if memory pressure returns:",
            "",
            "```bash",
            "PYTHONPATH=. python3 scripts/run_performance_market_state_modulator.py \\",
            "  --input <candidate_timestamp_panel.parquet> \\",
            "  --output-dir data_perp/reports/<new_run_id> \\",
            "  --pipeline-scope per_head \\",
            "  --strategies short_asset,short_boll,long_bars,long_dist \\",
            "  --rolling-bad-regime-window-hours 12,24 \\",
            "  --first-stage-max-depth 3 \\",
            "  --first-stage-num-leaves 8 \\",
            "  --first-stage-min-child-samples-fraction 0.025 \\",
            "  --stage-gate-profile lenient",
            "```",
            "",
            "For the next controller ablation, keep the active global-rank T1 contract fixed, keep q-fail/controller execution disabled, and rerun only a direct-suppression/frontier objective variant. Do not promote any controller until the pruning audit yields at least one state head with positive leave-one-out and defensive-action gates.",
        ]
    )

    summary = {
        "active_contract": active_rows.iloc[0].to_dict(),
        "june_d2_minus_d0": d2_delta,
        "shadow_controller": shadow_rows.iloc[0].to_dict(),
        "instability_by_strategy": instability.to_dict(orient="records") if not instability.empty else [],
        "source_paths": {
            "operational_status": args.operational_status,
            "june_design": args.june_design,
            "rank_comparison": args.rank_comparison,
            "prejune_walkforward": args.prejune_walkforward,
            "backend_comparison": args.backend_comparison,
            "shadow_monitor": args.shadow_monitor,
            "instability_report": args.instability_report,
            "shallow_pilot_report": args.shallow_pilot_report,
            "safe_allhead_report": args.safe_allhead_report,
            "contract_audit": args.contract_audit,
            "promotion_gate_audit": args.promotion_gate_audit,
            "direct_suppression_actionability_audit": args.direct_suppression_actionability_audit,
            "next_no_backfill_readiness": args.next_no_backfill_readiness,
            "plan_completion_audit": args.plan_completion_audit,
            "state_head_pruning_audit": args.state_head_pruning_audit,
            "strategy_response_quality": args.strategy_response_quality,
        },
    }
    return lines, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operational-status", type=Path, default=DEFAULT_OPERATIONAL_STATUS)
    parser.add_argument("--june-design", type=Path, default=DEFAULT_JUNE_DESIGN)
    parser.add_argument("--rank-comparison", type=Path, default=DEFAULT_RANK_COMPARISON)
    parser.add_argument("--prejune-walkforward", type=Path, default=DEFAULT_PREJUNE_WALKFORWARD)
    parser.add_argument("--backend-comparison", type=Path, default=DEFAULT_BACKEND_COMPARISON)
    parser.add_argument("--shadow-monitor", type=Path, default=DEFAULT_SHADOW_MONITOR)
    parser.add_argument("--instability-report", type=Path, default=DEFAULT_INSTABILITY)
    parser.add_argument("--shallow-pilot-report", type=Path, default=DEFAULT_SHALLOW_PILOT)
    parser.add_argument("--safe-allhead-report", type=Path, default=DEFAULT_SAFE_ALLHEAD)
    parser.add_argument("--contract-audit", type=Path, default=DEFAULT_CONTRACT_AUDIT)
    parser.add_argument("--promotion-gate-audit", type=Path, default=DEFAULT_PROMOTION_GATE_AUDIT)
    parser.add_argument(
        "--direct-suppression-actionability-audit",
        type=Path,
        default=DEFAULT_DIRECT_SUPPRESSION_ACTIONABILITY_AUDIT,
    )
    parser.add_argument(
        "--next-no-backfill-readiness",
        type=Path,
        default=DEFAULT_NEXT_NO_BACKFILL_READINESS,
    )
    parser.add_argument("--plan-completion-audit", type=Path, default=DEFAULT_PLAN_COMPLETION_AUDIT)
    parser.add_argument("--state-head-pruning-audit", type=Path, default=DEFAULT_STATE_HEAD_PRUNING_AUDIT)
    parser.add_argument("--strategy-response-quality", type=Path, default=DEFAULT_STRATEGY_RESPONSE_QUALITY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_perp/reports/market_state_goal_review_20260628_v1"),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    lines, summary = build_review(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "market_state_goal_review.md").write_text("\n".join(lines) + "\n")
    (args.output_dir / "market_state_goal_review_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(_json_safe({"output_dir": args.output_dir, "summary": summary}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
