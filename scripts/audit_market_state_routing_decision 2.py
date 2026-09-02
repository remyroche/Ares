#!/usr/bin/env python3
"""Summarize market-state routing evidence and active execution status.

This audit intentionally separates three ideas that were easy to confound:

* T1 static routing baseline;
* penalty-only threshold controller;
* head-priority / auction-order modulation.

It does not train models or replay trades.  It reads already generated
walk-forward and June attribution artifacts, then emits a small decision record
that can be checked before materializing production/shadow bundles.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_THRESHOLD_DIR = Path(
    "data_perp/reports/market_state_threshold_controller_walkforward_20260626_t1_lgbm_maturity_contract_v1"
)
DEFAULT_HEAD_PRIORITY_DIR = Path(
    "data_perp/reports/market_state_priority_shadow_windows_timestamprank_safegrid_lgbm_broad_parityfixed_20260626_v1"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_routing_decision_audit_20260626")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_priority_gate_audit(audit_dir: Path | None) -> dict[str, Any] | None:
    if audit_dir is None:
        return None
    candidates = [
        audit_dir / "market_state_head_priority_promotion_gate_audit.json",
        audit_dir / "market_state_priority_shadow_promotion_gate.json",
        audit_dir / "head_priority_window_transfer_audit.json",
        audit_dir / "promotion_audit" / "market_state_priority_shadow_promotion_gate.json",
    ]
    for path in candidates:
        payload = _read_json(path)
        if payload:
            if path.name == "head_priority_window_transfer_audit.json":
                gate = payload.get("promotion_gate")
                if isinstance(gate, dict):
                    return gate
            return payload
    return None


def _row_by_arm(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    if frame.empty or "arm" not in frame.columns:
        return {}
    rows = frame.loc[frame["arm"].astype(str).eq(str(arm))]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _float(row: dict[str, Any], key: str, default: float = np.nan) -> float:
    value = pd.to_numeric(pd.Series([row.get(key, default)]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else float(default)


def threshold_controller_decision(threshold_dir: Path) -> dict[str, Any]:
    promotion = _read_json(threshold_dir / "market_state_controller_promotion_gate_audit.json")
    selected = _read_json(threshold_dir / "walkforward_selected_controller_candidate.json")
    selection = _read_csv(threshold_dir / "walkforward_controller_candidate_selection.csv")
    aggregate = _read_csv(threshold_dir / "walkforward_aggregate_delta.csv")
    activation = _read_csv(threshold_dir / "market_state_activation_registry.csv")

    best_raw = dict(promotion.get("best_raw_candidate") or {})
    selected_arm = selected.get("selected_arm")
    should_remain_disabled = bool(promotion.get("controller_should_remain_disabled", True))
    passing = int(promotion.get("passing_candidate_count", 0) or 0)

    return {
        "component": "penalty_only_threshold_controller",
        "artifact_dir": str(threshold_dir),
        "active_status": "disabled_noop" if should_remain_disabled or not selected_arm else "candidate_selected",
        "selected_arm": selected_arm,
        "selection_reason": selected.get("reason"),
        "passing_candidate_count": passing,
        "controller_should_remain_disabled": should_remain_disabled,
        "best_raw_candidate": {
            "arm": best_raw.get("arm"),
            "median_delta_net_pnl": _float(best_raw, "median_delta_net_pnl"),
            "q25_delta_net_pnl": _float(best_raw, "q25_delta_net_pnl"),
            "positive_delta_share": _float(best_raw, "positive_delta_share"),
            "realized_defensive_success": _float(best_raw, "realized_defensive_success"),
            "post_selection_realized_defensive_success": _float(
                best_raw,
                "post_selection_realized_defensive_success",
            ),
            "post_selection_positive_suppression_fold_share": _float(
                best_raw,
                "post_selection_positive_suppression_fold_share",
            ),
            "freed_capacity_entrant_count": _float(best_raw, "freed_capacity_entrant_count"),
            "freed_capacity_net_replacement_pnl": _float(
                best_raw,
                "freed_capacity_net_replacement_pnl",
            ),
            "freed_capacity_net_action_pnl_delta": _float(
                best_raw,
                "freed_capacity_net_action_pnl_delta",
            ),
            "positive_freed_capacity_fold_share": _float(
                best_raw,
                "positive_freed_capacity_fold_share",
            ),
            "post_selection_freed_capacity_net_replacement_pnl": _float(
                best_raw,
                "post_selection_freed_capacity_net_replacement_pnl",
            ),
            "post_selection_freed_capacity_net_action_pnl_delta": _float(
                best_raw,
                "post_selection_freed_capacity_net_action_pnl_delta",
            ),
            "fail_reasons": best_raw.get("recomputed_fail_reasons"),
        },
        "aggregate_rows": int(len(aggregate)),
        "candidate_rows": int(len(selection)),
        "failure_reason_counts": promotion.get("failure_reason_counts", {}),
        "state_head_activation": state_head_activation_summary(
            activation,
            controller_enabled=bool(selected_arm) and not should_remain_disabled,
        ),
    }


def state_head_activation_summary(
    activation: pd.DataFrame,
    *,
    controller_enabled: bool,
) -> dict[str, Any]:
    if activation.empty:
        return {
            "available": False,
            "controller_enabled": bool(controller_enabled),
            "executable_state_heads": [],
            "shadow_state_heads": [],
            "disabled_state_heads": [],
            "status_counts": {},
            "by_level": {},
            "by_component_group": {},
        }
    work = activation.copy()
    status = work.get("recommended_status", pd.Series("", index=work.index)).astype(str)
    state_head = work.get("state_head", pd.Series("", index=work.index)).astype(str)
    active_candidates = sorted(state_head.loc[status.eq("active_candidate")].dropna().unique().tolist())
    shadows = sorted(state_head.loc[status.eq("shadow")].dropna().unique().tolist())
    disabled = sorted(state_head.loc[status.eq("disabled_candidate")].dropna().unique().tolist())
    executable = active_candidates if bool(controller_enabled) else []
    shadow_state_heads = sorted(set(active_candidates + shadows)) if not controller_enabled else shadows

    by_level: dict[str, dict[str, int]] = {}
    if "state_level" in work.columns:
        for level, group in work.groupby(work["state_level"].astype(str), sort=True):
            counts = group.get("recommended_status", pd.Series("", index=group.index)).astype(str).value_counts()
            by_level[str(level)] = {str(k): int(v) for k, v in counts.items()}
    by_component: dict[str, dict[str, int]] = {}
    if "component_group" in work.columns:
        for group_name, group in work.groupby(work["component_group"].astype(str), sort=True):
            counts = group.get("recommended_status", pd.Series("", index=group.index)).astype(str).value_counts()
            by_component[str(group_name)] = {str(k): int(v) for k, v in counts.items()}

    ranked_cols = [
        "state_head",
        "state_level",
        "component_group",
        "recommended_status",
        "loo_median_increment_net_pnl",
        "loo_q25_increment_net_pnl",
        "loo_positive_increment_share",
        "activation_disable_reason",
    ]
    ranked = work[[c for c in ranked_cols if c in work.columns]].copy()
    if "loo_median_increment_net_pnl" in ranked.columns:
        ranked["_sort"] = pd.to_numeric(ranked["loo_median_increment_net_pnl"], errors="coerce")
        ranked = ranked.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    return {
        "available": True,
        "controller_enabled": bool(controller_enabled),
        "executable_state_heads": executable,
        "shadow_state_heads": shadow_state_heads,
        "disabled_state_heads": disabled,
        "active_candidate_state_heads": active_candidates,
        "status_counts": {str(k): int(v) for k, v in status.value_counts().items()},
        "by_level": by_level,
        "by_component_group": by_component,
        "top_state_heads_by_leave_one_out": ranked.head(8).to_dict("records"),
    }


def head_priority_decision(priority_dir: Path) -> dict[str, Any]:
    shadow_summary_path = priority_dir / "promotion_audit" / "market_state_priority_shadow_window_summary.csv"
    if shadow_summary_path.exists():
        return shadow_window_priority_decision(priority_dir)

    summary = _read_csv(priority_dir / "head_priority_learning_replay_summary.csv")
    by_head = _read_csv(priority_dir / "head_priority_learning_by_head.csv")
    diagnostics = _read_csv(priority_dir / "head_priority_learning_model_diagnostics.csv")
    selection = _read_csv(priority_dir / "head_priority_config_selection.csv")
    overlap = _read_csv(priority_dir / "head_priority_learning_accepted_overlap.csv")

    baseline_arm = "P0_static_priority"
    candidate_rows = summary.loc[~summary.get("arm", pd.Series(dtype=str)).astype(str).eq(baseline_arm)]
    candidate_arm = str(candidate_rows.iloc[0]["arm"]) if not candidate_rows.empty else None
    baseline = _row_by_arm(summary, baseline_arm)
    candidate = _row_by_arm(summary, candidate_arm) if candidate_arm else {}
    base_net = _float(baseline, "net_pnl", 0.0)
    cand_net = _float(candidate, "net_pnl", np.nan)
    base_full_sl = _float(baseline, "full_sl_rate", np.nan)
    cand_full_sl = _float(candidate, "full_sl_rate", np.nan)
    net_delta = cand_net - base_net if np.isfinite(cand_net) else float("nan")
    full_sl_delta = cand_full_sl - base_full_sl if np.isfinite(cand_full_sl) else float("nan")

    diag = diagnostics.iloc[0].to_dict() if not diagnostics.empty else {}
    pass_count = int(selection.get("selection_gate_passed", pd.Series(dtype=bool)).astype(bool).sum()) if not selection.empty else 0
    overlap_row = _row_by_arm(overlap, candidate_arm) if candidate_arm else {}
    active_status = "shadow_rejected"
    if (
        pass_count > 0
        and np.isfinite(net_delta)
        and net_delta > 0.0
        and np.isfinite(full_sl_delta)
        and full_sl_delta <= 0.0
    ):
        active_status = "shadow_candidate_needs_later_oos"

    return {
        "component": "head_priority_auction_modulation",
        "artifact_dir": str(priority_dir),
        "active_status": active_status,
        "baseline_arm": baseline_arm,
        "candidate_arm": candidate_arm,
        "net_pnl_delta_vs_static": net_delta,
        "full_sl_delta_vs_static": full_sl_delta,
        "trade_count_delta_vs_static": int(_float(candidate, "trade_count", 0.0) - _float(baseline, "trade_count", 0.0)),
        "accepted_jaccard_vs_static": _float(overlap_row, "jaccard_vs_baseline"),
        "selection_gate_pass_count": pass_count,
        "selected_config": {
            "backend": diag.get("backend"),
            "min_rank": _float(diag, "config_min_rank"),
            "frontier_gamma": _float(diag, "config_frontier_gamma"),
            "frontier_bandwidth": _float(diag, "config_frontier_bandwidth"),
            "sl_penalty": _float(diag, "config_sl_penalty"),
            "selection_objective": _float(diag, "selection_objective"),
            "fold_mean_spearman": _float(diag, "fold_mean_spearman"),
            "fold_mean_directional_accuracy": _float(diag, "fold_mean_directional_accuracy"),
            "fold_incremental_objective": _float(diag, "fold_incremental_objective"),
            "trailing_validation_spearman": _float(diag, "validation_spearman"),
            "trailing_validation_directional_accuracy": _float(diag, "validation_directional_accuracy"),
        },
        "by_head_rows": by_head.to_dict("records"),
    }


def shadow_window_priority_decision(priority_dir: Path) -> dict[str, Any]:
    """Summarize a multi-window priority-shadow aggregate.

    These artifacts are produced by `run_market_state_priority_shadow_windows.py`.
    They are the current authority for priority modulation because each window
    can be checked against the materialized T1 static baseline before applying a
    learned schedule.
    """

    summary = _read_csv(priority_dir / "promotion_audit" / "market_state_priority_shadow_window_summary.csv")
    by_head = _read_csv(priority_dir / "promotion_audit" / "market_state_priority_shadow_by_head.csv")
    manifest = _read_json(priority_dir / "manifest.json")
    gate = load_priority_gate_audit(priority_dir) or {}
    if summary.empty:
        return {
            "component": "head_priority_auction_modulation",
            "artifact_dir": str(priority_dir),
            "active_status": "shadow_rejected",
            "baseline_arm": "P0_static_priority",
            "candidate_arm": None,
            "net_pnl_delta_vs_static": float("nan"),
            "full_sl_delta_vs_static": float("nan"),
            "trade_count_delta_vs_static": 0,
            "accepted_jaccard_vs_static": float("nan"),
            "selection_gate_pass_count": 0,
            "selected_config": {},
            "window_summary": [],
            "by_head_rows": [],
            "source_format": "shadow_window_aggregate",
        }

    work = summary.copy()
    if "gate_passed" in work.columns:
        gate_pass_count = int(work["gate_passed"].astype(bool).sum())
    else:
        gate_pass_count = 0
    arm_counts = work.get("arm", pd.Series(dtype=str)).astype(str).value_counts()
    candidate_arm = str(arm_counts.index[0]) if not arm_counts.empty else None
    first = work.iloc[0].to_dict()

    gate_status = priority_gate_shadow_decision(gate)
    should_remain_shadow = bool(gate_status.get("should_remain_shadow", True))
    active_status = "shadow_rejected" if should_remain_shadow else "shadow_candidate_needs_later_oos"

    full_sl_delta = pd.to_numeric(work.get("delta_full_sl_rate"), errors="coerce")
    timeout_delta = pd.to_numeric(work.get("delta_timeout_rate"), errors="coerce")
    accepted_jaccard = pd.to_numeric(work.get("accepted_jaccard"), errors="coerce")
    net_delta = pd.to_numeric(work.get("delta_net_pnl"), errors="coerce")
    contract = manifest.get("contract") if isinstance(manifest.get("contract"), dict) else {}
    parity = manifest.get("static_baseline_parity") if isinstance(manifest.get("static_baseline_parity"), dict) else {}
    base_trade_count = 0.0
    for window in manifest.get("windows") or []:
        if not isinstance(window, dict):
            continue
        observed = ((window.get("static_baseline_parity") or {}).get("observed") or {})
        value = _float(observed, "trade_count", np.nan)
        if np.isfinite(value):
            base_trade_count += value
    candidate_trade_count = pd.to_numeric(work.get("trade_count"), errors="coerce").sum()
    trade_count_delta = (
        int(round(float(candidate_trade_count - base_trade_count)))
        if np.isfinite(candidate_trade_count) and np.isfinite(base_trade_count)
        else 0
    )

    return {
        "component": "head_priority_auction_modulation",
        "artifact_dir": str(priority_dir),
        "source_format": "shadow_window_aggregate",
        "active_status": active_status,
        "baseline_arm": "P0_static_priority",
        "candidate_arm": candidate_arm,
        "net_pnl_delta_vs_static": float(net_delta.median()) if not net_delta.dropna().empty else float("nan"),
        "full_sl_delta_vs_static": float(full_sl_delta.max()) if not full_sl_delta.dropna().empty else float("nan"),
        "timeout_delta_vs_static": float(timeout_delta.max()) if not timeout_delta.dropna().empty else float("nan"),
        "trade_count_delta_vs_static": trade_count_delta,
        "candidate_trade_count_sum": int(round(float(candidate_trade_count))) if np.isfinite(candidate_trade_count) else None,
        "base_trade_count_sum": int(round(float(base_trade_count))) if np.isfinite(base_trade_count) else None,
        "accepted_jaccard_vs_static": float(accepted_jaccard.min()) if not accepted_jaccard.dropna().empty else float("nan"),
        "selection_gate_pass_count": gate_pass_count,
        "selected_config": {
            "rank_contract_preserved": contract.get("rank_contract_preserved"),
            "candidate_rank_contract_names": contract.get("candidate_rank_contract_names"),
            "baseline_source": contract.get("active_baseline"),
            "static_baseline_parity_promotion_grade": parity.get("promotion_grade"),
            "static_baseline_parity_checked_windows": parity.get("checked_windows"),
            "static_baseline_parity_passed_windows": parity.get("passed_windows"),
        },
        "window_count": int(len(work)),
        "positive_delta_window_share": float((net_delta > 0.0).mean()) if len(net_delta) else float("nan"),
        "min_accepted_jaccard": float(accepted_jaccard.min()) if not accepted_jaccard.dropna().empty else float("nan"),
        "max_full_sl_delta": float(full_sl_delta.max()) if not full_sl_delta.dropna().empty else float("nan"),
        "max_timeout_delta": float(timeout_delta.max()) if not timeout_delta.dropna().empty else float("nan"),
        "window_summary": work.to_dict("records"),
        "by_head_rows": by_head.to_dict("records"),
    }


def priority_gate_shadow_decision(priority_gate: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve shadow status from legacy and opportunity-routing gate payloads."""
    if not priority_gate:
        return {
            "available": False,
            "should_remain_shadow": None,
            "gate_family": "missing",
            "gate_passed": None,
            "failures": [],
        }
    if "opportunity_should_remain_shadow" in priority_gate:
        opportunity = priority_gate.get("opportunity_routing_gate") or {}
        return {
            "available": True,
            "should_remain_shadow": bool(priority_gate.get("opportunity_should_remain_shadow", True)),
            "gate_family": "opportunity_routing",
            "gate_passed": bool(priority_gate.get("opportunity_routing_passed", False)),
            "failures": opportunity.get("failures") or [],
            "opportunity_routing_gate": opportunity,
            "defensive_suppression_gate": priority_gate.get("defensive_suppression_gate") or {},
        }
    if "priority_should_remain_shadow" in priority_gate:
        return {
            "available": True,
            "should_remain_shadow": bool(priority_gate.get("priority_should_remain_shadow", True)),
            "gate_family": "legacy_priority",
            "gate_passed": not bool(priority_gate.get("priority_should_remain_shadow", True)),
            "failures": (priority_gate.get("best_raw_candidate") or {}).get("fail_reasons"),
        }
    if "passed" in priority_gate:
        return {
            "available": True,
            "should_remain_shadow": not bool(priority_gate.get("passed")),
            "gate_family": "legacy_window_summary",
            "gate_passed": bool(priority_gate.get("passed")),
            "failures": priority_gate.get("failures") or [],
        }
    return {
        "available": True,
        "should_remain_shadow": True,
        "gate_family": "unknown",
        "gate_passed": False,
        "failures": ["unrecognized_priority_gate_payload"],
    }


def combined_decision(
    threshold: dict[str, Any],
    priority: dict[str, Any],
    priority_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    threshold_disabled = threshold.get("active_status") == "disabled_noop"
    priority_rejected = str(priority.get("active_status", "")).startswith("shadow_rejected")
    priority_gate_status = priority_gate_shadow_decision(priority_gate)
    if priority_gate:
        if bool(priority_gate_status.get("should_remain_shadow", True)):
            priority_rejected = True
    # The market-state rollout contract requires threshold-only control to pass
    # before any head-priority / auction-order modulation can become executable.
    # Even a passing opportunity-routing gate is therefore shadow-only while the
    # penalty-only threshold controller remains disabled.
    priority_blocked_by_threshold_controller = (
        threshold_disabled
        and bool(priority_gate_status.get("gate_passed", False))
        and not bool(priority_gate_status.get("should_remain_shadow", True))
    )
    if priority_blocked_by_threshold_controller:
        priority_rejected = True
    priority_status = "shadow_only_rejected" if priority_rejected else priority.get("active_status")
    if priority_blocked_by_threshold_controller:
        priority_status = "shadow_blocked_until_threshold_controller_promoted"
    active_stack = {
        "baseline": "T1_repaired_static_baseline",
        "active_heads": ["short_asset", "short_boll"],
        "disabled_heads": ["long_bars", "long_dist"],
        "q_fail": "disabled",
        "head_health": "disabled",
        "threshold_controller": "disabled_noop" if threshold_disabled else "candidate_selected",
        "head_priority_modulation": priority_status,
        "market_state_logging": "enabled_shadow",
        "executable_market_state_heads": threshold.get("state_head_activation", {}).get(
            "executable_state_heads",
            [],
        ),
        "shadow_market_state_heads": threshold.get("state_head_activation", {}).get(
            "shadow_state_heads",
            [],
        ),
    }
    recommendation = (
        "keep_static_T1_active_log_market_state_shadow_only"
        if threshold_disabled and priority_rejected
        else "review_non_static_market_state_candidate_before_any_promotion"
    )
    return {
        "decision_version": "market_state_routing_decision_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "recommendation": recommendation,
        "production_active_stack": active_stack,
        "promotion_allowed": False if recommendation.startswith("keep_static") else None,
        "priority_blocked_by_threshold_controller": bool(priority_blocked_by_threshold_controller),
        "required_next_validation": [
            "later_untouched_matured_period",
            "paired_against_exact_T1_static_baseline",
            "fixed_rank_contracts_and_candidate_universe",
            "threshold_controller_defensive_success_positive_without_winner_sacrifice_dominating",
            "head_priority_opportunity_routing_recurrent_positive_replacement_value",
            "threshold_only_controller_promotion_before_priority_modulation",
            "no_auction_priority_modulation_until_threshold_only_passes",
        ],
        "threshold_controller": threshold,
        "head_priority_modulation": priority,
        "head_priority_promotion_gate": priority_gate or {},
        "head_priority_gate_status": priority_gate_status,
    }


def render_markdown(report: dict[str, Any]) -> str:
    threshold = report["threshold_controller"]
    priority = report["head_priority_modulation"]
    priority_gate = report.get("head_priority_promotion_gate") or {}
    priority_gate_status = report.get("head_priority_gate_status") or {}
    lines = [
        "# Market-State Routing Decision Audit",
        "",
        f"Recommendation: `{report['recommendation']}`",
        "",
        "## Active Stack",
        "",
    ]
    for key, value in report["production_active_stack"].items():
        lines.append(f"- `{key}`: `{value}`")
    best = threshold["best_raw_candidate"]
    state_activation = threshold.get("state_head_activation", {})
    lines.extend(
        [
            "",
            "## Threshold Controller",
            "",
            f"- Status: `{threshold['active_status']}`",
            f"- Selected arm: `{threshold['selected_arm']}`",
            f"- Best raw arm: `{best['arm']}`",
            f"- Median delta net PnL: `{best['median_delta_net_pnl']:.6f}`",
            f"- Q25 delta net PnL: `{best['q25_delta_net_pnl']:.6f}`",
            f"- Defensive success: `{best['realized_defensive_success']:.6f}`",
            f"- Post-selection defensive success: `{best['post_selection_realized_defensive_success']:.6f}`",
            f"- Freed-capacity entrants: `{best.get('freed_capacity_entrant_count'):.6f}`",
            f"- Freed-capacity replacement PnL: `{best.get('freed_capacity_net_replacement_pnl'):.6f}`",
            f"- Freed-capacity action PnL delta: `{best.get('freed_capacity_net_action_pnl_delta'):.6f}`",
            f"- Post-selection freed-capacity replacement PnL: `{best.get('post_selection_freed_capacity_net_replacement_pnl'):.6f}`",
            f"- Post-selection freed-capacity action PnL delta: `{best.get('post_selection_freed_capacity_net_action_pnl_delta'):.6f}`",
            f"- Fail reasons: `{best['fail_reasons']}`",
            "",
            "## State-Head Activation",
            "",
            f"- Executable state heads: `{state_activation.get('executable_state_heads', [])}`",
            f"- Shadow state heads: `{state_activation.get('shadow_state_heads', [])}`",
            f"- Disabled state-head count: `{len(state_activation.get('disabled_state_heads', []))}`",
            f"- Status counts: `{state_activation.get('status_counts', {})}`",
            "",
            "## Head-Priority Modulation",
            "",
            f"- Status: `{priority['active_status']}`",
            f"- Candidate arm: `{priority['candidate_arm']}`",
            f"- Net PnL delta vs static: `{priority['net_pnl_delta_vs_static']:.6f}`",
            f"- Full-SL delta vs static: `{priority['full_sl_delta_vs_static']:.6f}`",
            f"- Trade-count delta vs static: `{priority['trade_count_delta_vs_static']}`",
            f"- Accepted Jaccard vs static: `{priority['accepted_jaccard_vs_static']:.6f}`",
            "",
            "## Head-Priority Promotion Gate",
            "",
        ]
    )
    if priority_gate:
        best_gate = priority_gate.get("best_raw_candidate") or {}
        opportunity = priority_gate_status.get("opportunity_routing_gate") or priority_gate.get("opportunity_routing_gate") or {}
        defensive = priority_gate_status.get("defensive_suppression_gate") or priority_gate.get("defensive_suppression_gate") or {}
        lines.extend(
            [
                f"- Gate family: `{priority_gate_status.get('gate_family')}`",
                f"- Gate passed: `{priority_gate_status.get('gate_passed')}`",
                f"- Should remain shadow: `{priority_gate_status.get('should_remain_shadow')}`",
                f"- Passing candidate count: `{priority_gate.get('passing_candidate_count')}`",
                f"- Best raw arm: `{best_gate.get('arm')}`",
                f"- Best raw fail reasons: `{best_gate.get('fail_reasons')}`",
                f"- Best raw net action PnL delta: `{_float(best_gate, 'net_action_pnl_delta'):.6f}`",
                f"- Best raw net replacement PnL: `{_float(best_gate, 'net_replacement_pnl'):.6f}`",
                f"- Opportunity action windows: `{opportunity.get('action_window_count')}`",
                f"- Opportunity positive action windows: `{opportunity.get('positive_action_window_count')}`",
                f"- Opportunity median replacement PnL: `{_float(opportunity, 'median_replacement_pnl_action_windows'):.6f}`",
                f"- Opportunity median net action PnL delta: `{_float(opportunity, 'median_net_action_pnl_delta_action_windows'):.6f}`",
                f"- Opportunity failures: `{opportunity.get('failures')}`",
                f"- Defensive-suppression passed: `{defensive.get('passed')}`",
                "",
            ]
        )
    else:
        lines.extend(["_No head-priority promotion gate audit supplied._", ""])
    lines.extend(
        [
            "## Next Validation",
            "",
        ]
    )
    for item in report["required_next_validation"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-dir", type=Path, default=DEFAULT_THRESHOLD_DIR)
    parser.add_argument("--head-priority-dir", type=Path, default=DEFAULT_HEAD_PRIORITY_DIR)
    parser.add_argument("--head-priority-gate-audit-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    threshold = threshold_controller_decision(args.threshold_dir)
    priority = head_priority_decision(args.head_priority_dir)
    priority_gate = load_priority_gate_audit(args.head_priority_gate_audit_dir)
    report = combined_decision(threshold, priority, priority_gate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "market_state_routing_decision_audit.json"
    md_path = args.output_dir / "market_state_routing_decision_audit.md"
    json_path.write_text(json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(_json_safe({"output_dir": str(args.output_dir), "recommendation": report["recommendation"]}), indent=2))


if __name__ == "__main__":
    main()
