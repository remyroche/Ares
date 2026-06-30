#!/usr/bin/env python3
"""Summarize how market state should modulate portfolio head priority.

The report is intentionally diagnostic.  It does not train models, replay
trades, or change the active production stack.  It combines the current rank
starvation evidence, LGBM/XGB priority shadow gates, rank-scope router results,
and operational status into a single decision artifact.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_STARVATION_DIR = Path(
    "data_perp/reports/market_state_rank_priority_starvation_audit_20260626_refresh_v1"
)
DEFAULT_LGBM_GATE = Path(
    "data_perp/reports/market_state_priority_shadow_windows_timestamprank_safegrid_lgbm_broad_parityfixed_20260626_v1"
    "/promotion_audit/market_state_priority_shadow_promotion_gate.json"
)
DEFAULT_XGB_GATE = Path(
    "data_perp/reports/market_state_priority_shadow_windows_timestamprank_safegrid_xgb_broad_parityfixed_20260626_v1"
    "/promotion_audit/market_state_priority_shadow_promotion_gate.json"
)
DEFAULT_RANK_ROUTER = Path(
    "data_perp/reports/market_state_rank_scope_blend_validation_20260626_jun15_24_v1"
    "/rank_scope_blend_validation_summary.json"
)
DEFAULT_OPERATIONAL_STATUS = Path(
    "data_perp/reports/market_state_operational_status_20260626_v1"
    "/market_state_operational_status.json"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_head_priority_assessment_20260626_v1")


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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_json_any(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _first_row(frame: pd.DataFrame, **equals: str) -> dict[str, Any]:
    if frame.empty:
        return {}
    work = frame
    for col, value in equals.items():
        if col not in work.columns:
            return {}
        work = work.loc[work[col].astype(str).eq(str(value))]
    return work.iloc[0].to_dict() if not work.empty else {}


def _priority_replay(priority: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for row in priority:
        if str(row.get("label")) == str(label):
            return dict(row)
    return {}


def build_assessment(
    *,
    starvation_dir: Path,
    lgbm_gate_path: Path,
    xgb_gate_path: Path,
    rank_router_path: Path,
    operational_status_path: Path,
) -> dict[str, Any]:
    starvation = _load_csv(starvation_dir / "rank_priority_starvation_by_head.csv")
    delta = _load_csv(starvation_dir / "rank_priority_starvation_delta.csv")
    priority_payload = _load_json_any(starvation_dir / "rank_priority_learned_priority_replays.json")
    if isinstance(priority_payload, list):
        priority_replays = priority_payload
    else:
        priority_replays = []
    lgbm_gate = _load_json(lgbm_gate_path)
    xgb_gate = _load_json(xgb_gate_path)
    rank_router = _load_json(rank_router_path)
    operational_status = _load_json(operational_status_path)

    short_boll_delta = _first_row(delta, head="short_boll")
    timestamp_short_boll = _first_row(
        starvation,
        contract_name="timestamp_rank_t1",
        head="short_boll",
    )
    global_short_boll = _first_row(
        starvation,
        contract_name="global_rank_challenger",
        head="short_boll",
    )
    global_priority = _priority_replay(priority_replays, "global_rank_priority")
    timestamp_priority = _priority_replay(priority_replays, "timestamp_rank_priority")
    router_rollup = dict(rank_router.get("rollup") or {})

    delta_deployable = int(float(short_boll_delta.get("delta_deployable_rows", 0) or 0))
    delta_accepted = int(float(short_boll_delta.get("delta_accepted_rows", 0) or 0))
    global_priority_jaccard = float(global_priority.get("accepted_jaccard", np.nan))
    global_priority_delta = float(global_priority.get("delta_net_pnl", np.nan))
    global_starvation_confirmed = bool(
        delta_deployable < 0
        and delta_accepted < 0
        and np.isfinite(global_priority_jaccard)
        and global_priority_jaccard >= 0.999
        and np.isfinite(global_priority_delta)
        and abs(global_priority_delta) <= 1e-9
    )

    lgbm_opportunity = dict(lgbm_gate.get("opportunity_routing_gate") or {})
    xgb_opportunity = dict(xgb_gate.get("opportunity_routing_gate") or {})
    router_gate_passed = bool(router_rollup.get("shadow_promotion_gate_passed") is True)
    later_positive_share = float(router_rollup.get("later_blend_positive_delta_share", np.nan))
    later_median_delta = float(router_rollup.get("later_blend_median_delta_net_pnl", np.nan))

    pure_auction_priority_sufficient = not global_starvation_confirmed
    if global_starvation_confirmed:
        recommended_modulation = (
            "state_conditioned_head_prior_before_eligibility_then_global_auction"
        )
        rank_reference_track = (
            "candidate_eligibility_component_of_head_priority_modulation"
        )
        next_layer = "shadow_pre_filter_head_priority_rank_modulation"
    else:
        recommended_modulation = "bounded_auction_priority_shadow_only_after_threshold_controller"
        rank_reference_track = "no_starvation_blocker_detected"
        next_layer = "bounded_auction_priority_shadow_replay"

    promotion_allowed = bool(
        operational_status.get("production_ready") is True
        and not global_starvation_confirmed
        and bool(lgbm_gate.get("passed") is True or xgb_gate.get("passed") is True or router_gate_passed)
    )
    if global_starvation_confirmed:
        promotion_reason = (
            "short_boll_global_rank_starvation_requires_pre_filter_head_prior_not_post_filter_auction_tilt"
        )
    elif promotion_allowed:
        promotion_reason = "at_least_one_priority_modulator_passed_current_gates"
    else:
        promotion_reason = "market_state_priority_modulation_remains_shadow_only"

    return {
        "generated_by": "report_market_state_head_priority_assessment",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_stack": {
            "name": operational_status.get("active_stack_name"),
            "rank_contract": operational_status.get("active_rank_contract"),
            "rank_scope": operational_status.get("active_rank_scope"),
            "active_heads": operational_status.get("active_heads"),
            "disabled_heads": operational_status.get("disabled_heads"),
            "threshold_controller_active": operational_status.get("threshold_controller_active"),
            "priority_modulation_active": operational_status.get("priority_modulation_active"),
            "production_ready": operational_status.get("production_ready"),
        },
        "short_boll_starvation": {
            "global_starvation_confirmed": global_starvation_confirmed,
            "timestamp_deployable_rows": int(float(timestamp_short_boll.get("deployable_rows", 0) or 0)),
            "global_deployable_rows": int(float(global_short_boll.get("deployable_rows", 0) or 0)),
            "delta_deployable_rows_global_minus_timestamp": delta_deployable,
            "timestamp_accepted_rows": int(float(timestamp_short_boll.get("accepted_rows", 0) or 0)),
            "global_accepted_rows": int(float(global_short_boll.get("accepted_rows", 0) or 0)),
            "delta_accepted_rows_global_minus_timestamp": delta_accepted,
            "timestamp_accepted_mean_net_return": float(timestamp_short_boll.get("accepted_mean_net_return", np.nan)),
            "global_accepted_mean_net_return": float(global_short_boll.get("accepted_mean_net_return", np.nan)),
            "global_priority_jaccard": global_priority_jaccard,
            "global_priority_delta_net_pnl": global_priority_delta,
        },
        "learned_priority_evidence": {
            "timestamp_rank_priority_delta_net_pnl": float(timestamp_priority.get("delta_net_pnl", np.nan)),
            "timestamp_rank_priority_accepted_jaccard": float(timestamp_priority.get("accepted_jaccard", np.nan)),
            "global_rank_priority_delta_net_pnl": global_priority_delta,
            "global_rank_priority_accepted_jaccard": global_priority_jaccard,
            "lgbm_gate_passed": bool(lgbm_gate.get("passed") is True),
            "lgbm_failures": lgbm_gate.get("failures", []),
            "lgbm_opportunity_positive_delta_window_share": lgbm_opportunity.get("positive_delta_window_share"),
            "lgbm_opportunity_median_delta_net_pnl": lgbm_opportunity.get("median_delta_net_pnl"),
            "xgb_gate_passed": bool(xgb_gate.get("passed") is True),
            "xgb_failures": xgb_gate.get("failures", []),
            "xgb_opportunity_positive_delta_window_share": xgb_opportunity.get("positive_delta_window_share"),
            "xgb_opportunity_median_delta_net_pnl": xgb_opportunity.get("median_delta_net_pnl"),
        },
        "rank_scope_router_evidence": {
            "gate_passed": router_gate_passed,
            "later_window_count": router_rollup.get("later_window_count"),
            "later_blend_positive_delta_share": later_positive_share,
            "later_blend_median_delta_net_pnl": later_median_delta,
            "later_blend_q25_delta_net_pnl": router_rollup.get("later_blend_q25_delta_net_pnl"),
            "max_blend_delta_full_sl_rate": router_rollup.get("max_blend_delta_full_sl_rate"),
            "max_blend_delta_timeout_rate": router_rollup.get("max_blend_delta_timeout_rate"),
            "failures": router_rollup.get("shadow_promotion_failures", []),
        },
        "decision": {
            "pure_auction_priority_sufficient": pure_auction_priority_sufficient,
            "recommended_modulation": recommended_modulation,
            "next_executable_market_state_layer": next_layer,
            "rank_reference_track": rank_reference_track,
            "priority_modulation_status": "shadow_only",
            "production_promotion_allowed": promotion_allowed,
            "promotion_reason": promotion_reason,
            "implementation_order": [
                "keep_exact_T1_as_frozen_static_baseline",
                "score_market_wide_state_axes_and_ood_drift",
                "fit_lgbm_xgb_head_timestamp_frontier_utility_models",
                "convert_state_response_into_bounded_head_prior_before_candidate_eligibility",
                "apply_existing_global_auction_after_state_adjusted_head_eligibility",
                "separate_candidate_admission_effect_from_auction_priority_effect",
                "keep_threshold_only_controller_as_defensive_secondary_track",
                "promote_only_after_later_window_and_replay_parity_gates_pass",
            ],
            "required_next_validation": [
                "lgbm_xgb_pre_filter_head_prior_replay_on_identical_candidate_universe",
                "rank_admission_and_auction_priority_attribution_reported_separately",
                "strict_global_over_time_short_boll_rank_reference_vs_exact_T1",
                "later_matured_window_shadow_validation_before_activation",
                "per_head_contribution_and_short_asset_flow_preservation_without_head_specific_reward_terms",
            ],
        },
        "inputs": {
            "starvation_dir": str(starvation_dir),
            "lgbm_gate_path": str(lgbm_gate_path),
            "xgb_gate_path": str(xgb_gate_path),
            "rank_router_path": str(rank_router_path),
            "operational_status_path": str(operational_status_path),
        },
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "n/a"
    return f"{f:.{digits}f}"


def render_report(payload: dict[str, Any]) -> str:
    active = dict(payload["active_stack"])
    starv = dict(payload["short_boll_starvation"])
    priority = dict(payload["learned_priority_evidence"])
    router = dict(payload["rank_scope_router_evidence"])
    decision = dict(payload["decision"])
    lines = [
        "# Market-State Head-Priority Assessment",
        "",
        "This report answers whether market state should modulate global portfolio priority, and at which layer.",
        "It is a diagnostic artifact only; it does not change the active stack.",
        "",
        "## Active Stack",
        "",
        f"- Stack: `{active.get('name')}`",
        f"- Rank contract: `{active.get('rank_contract')}` / `{active.get('rank_scope')}`",
        f"- Active heads: `{active.get('active_heads')}`",
        f"- Disabled heads: `{active.get('disabled_heads')}`",
        f"- Production ready for market-state modulation: `{active.get('production_ready')}`",
        "",
        "## Short-Boll Starvation",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| timestamp deployable rows | {starv.get('timestamp_deployable_rows')} |",
        f"| global deployable rows | {starv.get('global_deployable_rows')} |",
        f"| global minus timestamp deployable rows | {starv.get('delta_deployable_rows_global_minus_timestamp')} |",
        f"| timestamp accepted rows | {starv.get('timestamp_accepted_rows')} |",
        f"| global accepted rows | {starv.get('global_accepted_rows')} |",
        f"| global minus timestamp accepted rows | {starv.get('delta_accepted_rows_global_minus_timestamp')} |",
        f"| global priority replay Jaccard | {_fmt(starv.get('global_priority_jaccard'))} |",
        f"| global priority replay delta net PnL | {_fmt(starv.get('global_priority_delta_net_pnl'))} |",
        "",
        "Interpretation: pure auction-priority modulation is insufficient when the global-rank contract removes the relevant short_boll candidates before the auction can select them.",
        "",
        "## LGBM/XGB Priority Evidence",
        "",
        "| backend/path | passed | delta net PnL | positive-window share | accepted Jaccard / failures |",
        "|---|---:|---:|---:|---|",
        f"| timestamp-rank LGBM priority replay | n/a | {_fmt(priority.get('timestamp_rank_priority_delta_net_pnl'))} | n/a | Jaccard {_fmt(priority.get('timestamp_rank_priority_accepted_jaccard'))} |",
        f"| global-rank LGBM priority replay | n/a | {_fmt(priority.get('global_rank_priority_delta_net_pnl'))} | n/a | Jaccard {_fmt(priority.get('global_rank_priority_accepted_jaccard'))} |",
        f"| LGBM shadow gate | {priority.get('lgbm_gate_passed')} | {_fmt(priority.get('lgbm_opportunity_median_delta_net_pnl'))} | {_fmt(priority.get('lgbm_opportunity_positive_delta_window_share'))} | {priority.get('lgbm_failures')} |",
        f"| XGB shadow gate | {priority.get('xgb_gate_passed')} | {_fmt(priority.get('xgb_opportunity_median_delta_net_pnl'))} | {_fmt(priority.get('xgb_opportunity_positive_delta_window_share'))} | {priority.get('xgb_failures')} |",
        "",
        "## Rank-Scope Router Evidence",
        "",
        f"- Gate passed: `{router.get('gate_passed')}`",
        f"- Later windows: `{router.get('later_window_count')}`",
        f"- Later median delta net PnL: `{_fmt(router.get('later_blend_median_delta_net_pnl'))}`",
        f"- Later positive delta share: `{_fmt(router.get('later_blend_positive_delta_share'))}`",
        f"- Failures: `{router.get('failures')}`",
        "",
        "## Decision",
        "",
        f"- Pure auction priority sufficient: `{decision.get('pure_auction_priority_sufficient')}`",
        f"- Recommended modulation: `{decision.get('recommended_modulation')}`",
        f"- Next executable market-state layer: `{decision.get('next_executable_market_state_layer')}`",
        f"- Rank-reference track: `{decision.get('rank_reference_track')}`",
        f"- Priority modulation status: `{decision.get('priority_modulation_status')}`",
        f"- Promotion allowed now: `{decision.get('production_promotion_allowed')}`",
        f"- Reason: `{decision.get('promotion_reason')}`",
        "",
        "The next market-state priority assessment should not be a pure post-filter auction tweak. The June evidence shows short_boll can be removed before the auction sees it, so the research path is a shadow pre-filter head-prior layer: score market-wide state with LGBM/XGB, translate it into bounded head-specific eligibility/rank-prior adjustments, then run the existing global auction. Threshold-only control remains a separate defensive track.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starvation-dir", type=Path, default=DEFAULT_STARVATION_DIR)
    parser.add_argument("--lgbm-gate", type=Path, default=DEFAULT_LGBM_GATE)
    parser.add_argument("--xgb-gate", type=Path, default=DEFAULT_XGB_GATE)
    parser.add_argument("--rank-router", type=Path, default=DEFAULT_RANK_ROUTER)
    parser.add_argument("--operational-status", type=Path, default=DEFAULT_OPERATIONAL_STATUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_assessment(
        starvation_dir=args.starvation_dir,
        lgbm_gate_path=args.lgbm_gate,
        xgb_gate_path=args.xgb_gate,
        rank_router_path=args.rank_router,
        operational_status_path=args.operational_status,
    )
    report = render_report(payload)
    (args.output_dir / "market_state_head_priority_assessment.json").write_text(
        json.dumps(_json_safe(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "market_state_head_priority_assessment.md").write_text(
        report,
        encoding="utf-8",
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "output_dir": str(args.output_dir),
                    "global_starvation_confirmed": payload["short_boll_starvation"][
                        "global_starvation_confirmed"
                    ],
                    "recommended_modulation": payload["decision"]["recommended_modulation"],
                    "promotion_allowed": payload["decision"]["production_promotion_allowed"],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
