from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.report_market_state_head_priority_assessment import build_assessment, render_report


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_assessment_flags_global_rank_starvation_before_auction_priority(tmp_path: Path) -> None:
    starvation_dir = tmp_path / "starvation"
    starvation_dir.mkdir()
    pd.DataFrame(
        [
            {
                "contract_name": "timestamp_rank_t1",
                "head": "short_boll",
                "deployable_rows": 100,
                "accepted_rows": 20,
                "accepted_mean_net_return": 0.01,
            },
            {
                "contract_name": "global_rank_challenger",
                "head": "short_boll",
                "deployable_rows": 25,
                "accepted_rows": 2,
                "accepted_mean_net_return": 0.02,
            },
        ]
    ).to_csv(starvation_dir / "rank_priority_starvation_by_head.csv", index=False)
    pd.DataFrame(
        [
            {
                "head": "short_boll",
                "delta_deployable_rows": -75,
                "delta_accepted_rows": -18,
            }
        ]
    ).to_csv(starvation_dir / "rank_priority_starvation_delta.csv", index=False)
    _write_json(
        starvation_dir / "rank_priority_learned_priority_replays.json",
        [
            {
                "label": "global_rank_priority",
                "accepted_jaccard": 1.0,
                "delta_net_pnl": 0.0,
            },
            {
                "label": "timestamp_rank_priority",
                "accepted_jaccard": 0.96,
                "delta_net_pnl": 4.0,
            },
        ],
    )
    lgbm_gate = tmp_path / "lgbm.json"
    xgb_gate = tmp_path / "xgb.json"
    _write_json(
        lgbm_gate,
        {
            "passed": False,
            "failures": ["accepted_jaccard_below_required_95pct"],
            "opportunity_routing_gate": {
                "median_delta_net_pnl": 10.0,
                "positive_delta_window_share": 0.67,
            },
        },
    )
    _write_json(
        xgb_gate,
        {
            "passed": False,
            "failures": ["full_sl_worsened_in_a_window"],
            "opportunity_routing_gate": {
                "median_delta_net_pnl": 5.0,
                "positive_delta_window_share": 0.67,
            },
        },
    )
    rank_router = tmp_path / "rank_router.json"
    _write_json(
        rank_router,
        {
            "rollup": {
                "shadow_promotion_gate_passed": False,
                "later_window_count": 2,
                "later_blend_positive_delta_share": 0.0,
                "later_blend_median_delta_net_pnl": -1.0,
                "shadow_promotion_failures": ["later_median_delta_not_positive"],
            }
        },
    )
    status = tmp_path / "status.json"
    _write_json(
        status,
        {
            "active_stack_name": "T1_repaired_static_baseline",
            "active_rank_contract": "short_boll_timestamp_rank",
            "active_rank_scope": "within_timestamp",
            "active_heads": ["short_asset", "short_boll"],
            "disabled_heads": ["long_bars", "long_dist"],
            "threshold_controller_active": False,
            "priority_modulation_active": False,
            "production_ready": False,
        },
    )

    payload = build_assessment(
        starvation_dir=starvation_dir,
        lgbm_gate_path=lgbm_gate,
        xgb_gate_path=xgb_gate,
        rank_router_path=rank_router,
        operational_status_path=status,
    )

    assert payload["short_boll_starvation"]["global_starvation_confirmed"] is True
    assert payload["decision"]["pure_auction_priority_sufficient"] is False
    assert (
        payload["decision"]["recommended_modulation"]
        == "state_conditioned_head_prior_before_eligibility_then_global_auction"
    )
    assert (
        payload["decision"]["next_executable_market_state_layer"]
        == "shadow_pre_filter_head_priority_rank_modulation"
    )
    assert (
        payload["decision"]["rank_reference_track"]
        == "candidate_eligibility_component_of_head_priority_modulation"
    )
    assert payload["decision"]["priority_modulation_status"] == "shadow_only"
    assert payload["decision"]["production_promotion_allowed"] is False
    report = render_report(payload)
    assert "pre-filter head-prior" in report.lower()
    assert "threshold-only" in report.lower()


def test_assessment_allows_pure_auction_priority_when_no_global_starvation(tmp_path: Path) -> None:
    starvation_dir = tmp_path / "starvation"
    starvation_dir.mkdir()
    pd.DataFrame(
        [
            {
                "contract_name": "timestamp_rank_t1",
                "head": "short_boll",
                "deployable_rows": 100,
                "accepted_rows": 20,
            },
            {
                "contract_name": "global_rank_challenger",
                "head": "short_boll",
                "deployable_rows": 105,
                "accepted_rows": 22,
            },
        ]
    ).to_csv(starvation_dir / "rank_priority_starvation_by_head.csv", index=False)
    pd.DataFrame(
        [
            {
                "head": "short_boll",
                "delta_deployable_rows": 5,
                "delta_accepted_rows": 2,
            }
        ]
    ).to_csv(starvation_dir / "rank_priority_starvation_delta.csv", index=False)
    _write_json(
        starvation_dir / "rank_priority_learned_priority_replays.json",
        [{"label": "global_rank_priority", "accepted_jaccard": 0.9, "delta_net_pnl": 2.0}],
    )
    lgbm_gate = tmp_path / "lgbm.json"
    xgb_gate = tmp_path / "xgb.json"
    rank_router = tmp_path / "rank_router.json"
    status = tmp_path / "status.json"
    _write_json(lgbm_gate, {"passed": True})
    _write_json(xgb_gate, {"passed": False})
    _write_json(rank_router, {"rollup": {"shadow_promotion_gate_passed": False}})
    _write_json(status, {"production_ready": True})

    payload = build_assessment(
        starvation_dir=starvation_dir,
        lgbm_gate_path=lgbm_gate,
        xgb_gate_path=xgb_gate,
        rank_router_path=rank_router,
        operational_status_path=status,
    )

    assert payload["short_boll_starvation"]["global_starvation_confirmed"] is False
    assert payload["decision"]["pure_auction_priority_sufficient"] is True
    assert (
        payload["decision"]["recommended_modulation"]
        == "bounded_auction_priority_shadow_only_after_threshold_controller"
    )
    assert payload["decision"]["next_executable_market_state_layer"] == "bounded_auction_priority_shadow_replay"
    assert payload["decision"]["production_promotion_allowed"] is True
