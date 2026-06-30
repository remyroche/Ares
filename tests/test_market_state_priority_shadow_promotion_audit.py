from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.audit_market_state_priority_shadow_promotion import (
    head_mix_metrics,
    load_window,
    opportunity_routing_gate,
    promotion_gate,
    resolve_gate_tolerances,
    resolve_arm_selector,
    select_recurrent_challenger_selector,
    selected_challenger_selector,
)


def test_head_mix_metrics_measure_generic_concentration_change() -> None:
    by_head = pd.DataFrame(
        {
            "window_label": ["w1", "w1", "w2", "w2"],
            "head": ["short_asset", "short_boll", "short_asset", "short_boll"],
            "baseline_trade_count": [9, 1, 4, 4],
            "shadow_trade_count": [6, 4, 8, 0],
        }
    )

    metrics = head_mix_metrics(by_head).set_index("window_label")

    assert metrics.loc["w1", "baseline_active_head_count"] == 2
    assert metrics.loc["w1", "shadow_active_head_count"] == 2
    assert metrics.loc["w1", "baseline_dominant_head_share"] == 0.9
    assert metrics.loc["w1", "shadow_dominant_head_share"] == 0.6
    assert round(metrics.loc["w1", "head_trade_share_l1_delta"], 6) == 0.3
    assert metrics.loc["w2", "starved_head_count"] == 1
    assert metrics.loc["w2", "shadow_active_head_count"] == 1


def test_opportunity_gate_can_optionally_reject_head_monoculture() -> None:
    summary = pd.DataFrame(
        {
            "delta_net_pnl": [12.0, 4.0, 0.5],
            "accepted_jaccard": [0.95, 0.93, 0.98],
            "coverage": [1.0, 1.0, 1.0],
            "entrants": [2, 1, 1],
            "removed": [2, 1, 1],
            "delta_full_sl_rate": [-0.01, 0.0, -0.02],
            "delta_timeout_rate": [0.0, -0.01, 0.0],
            "net_replacement_pnl": [6.0, 2.0, 1.0],
            "net_action_pnl_delta": [12.0, 4.0, 0.5],
            "defensive_success": [-4.0, -2.0, -1.0],
            "shadow_active_head_count": [2, 1, 2],
            "shadow_dominant_head_share": [0.70, 1.00, 0.60],
            "head_trade_share_l1_delta": [0.20, 0.50, 0.10],
        }
    )

    relaxed = opportunity_routing_gate(summary)
    strict = opportunity_routing_gate(
        summary,
        min_shadow_active_head_count=2,
        max_shadow_dominant_head_share=0.95,
        max_head_trade_share_l1_delta=0.40,
    )

    assert relaxed["passed"] is True
    assert strict["passed"] is False
    assert "shadow_active_head_count_below_gate" in strict["failures"]
    assert "shadow_dominant_head_share_above_gate" in strict["failures"]
    assert "head_trade_share_l1_delta_above_gate" in strict["failures"]
    assert strict["min_shadow_active_head_count"] == 1
    assert strict["max_shadow_dominant_head_share"] == 1.0


def _write_cap_metrics(root: Path, rows: list[dict[str, object]]) -> None:
    root.mkdir()
    pd.DataFrame(rows).to_csv(root / "head_priority_cap_sweep_metrics.csv", index=False)


def test_recurrent_selector_prefers_cap_with_repeated_accepted_set_action(tmp_path: Path) -> None:
    dirs = [tmp_path / f"w{i}" for i in range(3)]
    for i, root in enumerate(dirs):
        _write_cap_metrics(
            root,
            [
                {
                    "arm": "L1_lgbm_learned_priority_cap_0p3",
                    "delta_net_pnl": [6.0, 4.0, 2.0][i],
                    "accepted_jaccard": 0.97,
                    "coverage": 1.0,
                    "entrants": 1,
                    "removed": 1,
                    "delta_full_sl_rate": 0.0,
                    "delta_timeout_rate": 0.0,
                    "net_replacement_pnl": 2.0,
                    "net_action_pnl_delta": 2.0,
                    "max_adjustment": 0.3,
                    "min_abs_z": 0.0,
                },
                {
                    "arm": "L1_lgbm_learned_priority_cap_0p6",
                    "delta_net_pnl": [20.0, 0.0, 0.0][i],
                    "accepted_jaccard": 0.99,
                    "coverage": 1.0,
                    "entrants": [3, 0, 0][i],
                    "removed": [1, 0, 0][i],
                    "delta_full_sl_rate": 0.0,
                    "delta_timeout_rate": 0.0,
                    "net_replacement_pnl": [12.0, 0.0, 0.0][i],
                    "net_action_pnl_delta": [20.0, 0.0, 0.0][i],
                    "max_adjustment": 0.6,
                    "min_abs_z": 0.0,
                },
            ],
        )

    selected = select_recurrent_challenger_selector(dirs)

    assert selected["selected"] is True
    assert selected["arm_selector"] == "cap_0p3"
    chosen = selected["selected_row"]
    assert chosen["action_window_count"] == 3
    assert chosen["positive_action_window_count"] == 3
    rejected = {
        row["arm_selector"]: row["fail_reasons"]
        for row in selected["candidates"]
        if row["arm_selector"] == "cap_0p6"
    }
    assert "fewer_than_required_action_windows" in rejected["cap_0p6"]


def test_recurrent_selector_refuses_one_window_development_artifact(tmp_path: Path) -> None:
    dirs = [tmp_path / f"w{i}" for i in range(3)]
    for i, root in enumerate(dirs):
        _write_cap_metrics(
            root,
            [
                {
                    "arm": "L1_lgbm_learned_priority_cap_0p3",
                    "delta_net_pnl": [25.0, 0.0, 0.0][i],
                    "accepted_jaccard": 0.97,
                    "coverage": 1.0,
                    "entrants": [4, 0, 0][i],
                    "removed": [1, 0, 0][i],
                    "delta_full_sl_rate": 0.0,
                    "delta_timeout_rate": 0.0,
                    "net_replacement_pnl": [20.0, 0.0, 0.0][i],
                    "net_action_pnl_delta": [25.0, 0.0, 0.0][i],
                    "max_adjustment": 0.3,
                    "min_abs_z": 0.0,
                }
            ],
        )

    selected = select_recurrent_challenger_selector(dirs)

    assert selected["selected"] is False
    assert selected["reason"] == "no_recurrent_gate_passing_arm"
    assert selected["best_candidate"]["arm_selector"] == "cap_0p3"
    assert "fewer_than_required_action_windows" in selected["best_candidate"]["fail_reasons"]


def test_shadow_priority_gate_rejects_single_action_window_with_neutral_windows() -> None:
    summary = pd.DataFrame(
        {
            "delta_net_pnl": [22.0, 0.0, 0.0],
            "accepted_jaccard": [0.96, 1.0, 1.0],
            "coverage": [1.0, 1.0, 1.0],
            "entrants": [2, 0, 0],
            "removed": [2, 0, 0],
            "delta_full_sl_rate": [-0.01, 0.0, 0.0],
            "delta_timeout_rate": [0.0, 0.0, 0.0],
            "defensive_success": [8.0, 0.0, 0.0],
        }
    )

    gate = promotion_gate(summary)

    assert gate["passed"] is False
    assert gate["window_count"] == 3
    assert gate["action_window_count"] == 1
    assert gate["positive_action_window_count"] == 1
    assert "fewer_than_2_positive_action_windows" in gate["failures"]
    assert "median_delta_net_pnl_not_positive" in gate["failures"]


def test_shadow_priority_gate_accepts_repeated_bounded_positive_action() -> None:
    summary = pd.DataFrame(
        {
            "delta_net_pnl": [12.0, 4.0, 0.5],
            "accepted_jaccard": [0.95, 0.93, 0.98],
            "coverage": [1.0, 1.0, 1.0],
            "entrants": [2, 1, 1],
            "removed": [2, 1, 1],
            "delta_full_sl_rate": [-0.01, 0.0, -0.02],
            "delta_timeout_rate": [0.0, -0.01, 0.0],
            "defensive_success": [8.0, 3.0, 1.0],
        }
    )

    gate = promotion_gate(summary)

    assert gate["passed"] is True
    assert gate["failures"] == []


def test_opportunity_gate_accepts_replacement_value_without_defensive_success() -> None:
    summary = pd.DataFrame(
        {
            "delta_net_pnl": [12.0, 4.0, 0.5],
            "accepted_jaccard": [0.95, 0.93, 0.98],
            "coverage": [1.0, 1.0, 1.0],
            "entrants": [2, 1, 1],
            "removed": [2, 1, 1],
            "delta_full_sl_rate": [-0.01, 0.0, -0.02],
            "delta_timeout_rate": [0.0, -0.01, 0.0],
            "net_replacement_pnl": [6.0, 2.0, 1.0],
            "net_action_pnl_delta": [12.0, 4.0, 0.5],
            "defensive_success": [-4.0, -2.0, -1.0],
        }
    )

    defensive = promotion_gate(summary)
    opportunity = opportunity_routing_gate(summary)

    assert defensive["passed"] is False
    assert "action_window_defensive_success_not_positive" in defensive["failures"]
    assert opportunity["passed"] is True
    assert opportunity["failures"] == []


def test_opportunity_gate_rejects_single_june_like_action_window() -> None:
    summary = pd.DataFrame(
        {
            "delta_net_pnl": [22.0, 0.0, 0.0],
            "accepted_jaccard": [0.96, 1.0, 1.0],
            "coverage": [1.0, 1.0, 1.0],
            "entrants": [2, 0, 0],
            "removed": [2, 0, 0],
            "delta_full_sl_rate": [-0.01, 0.0, 0.0],
            "delta_timeout_rate": [0.0, 0.0, 0.0],
            "net_replacement_pnl": [17.0, 0.0, 0.0],
            "net_action_pnl_delta": [22.0, 0.0, 0.0],
            "defensive_success": [-1.0, 0.0, 0.0],
        }
    )

    gate = opportunity_routing_gate(summary)

    assert gate["passed"] is False
    assert "fewer_than_2_positive_action_windows" in gate["failures"]


def test_opportunity_gate_uses_explicit_safety_tolerances() -> None:
    summary = pd.DataFrame(
        {
            "delta_net_pnl": [12.0, 4.0, 0.5],
            "accepted_jaccard": [0.955, 0.96, 0.98],
            "coverage": [1.0, 1.0, 1.0],
            "entrants": [2, 1, 1],
            "removed": [2, 1, 1],
            "delta_full_sl_rate": [0.003, 0.0, 0.0],
            "delta_timeout_rate": [0.0, -0.01, 0.0],
            "net_replacement_pnl": [6.0, 2.0, 1.0],
            "net_action_pnl_delta": [12.0, 4.0, 0.5],
            "defensive_success": [-4.0, -2.0, -1.0],
        }
    )

    strict = opportunity_routing_gate(summary)
    tolerant = opportunity_routing_gate(
        summary,
        min_accepted_jaccard=0.95,
        max_full_sl_delta=0.005,
        max_timeout_delta=0.0,
    )

    assert strict["passed"] is False
    assert "full_sl_worsened_in_a_window" in strict["failures"]
    assert tolerant["passed"] is True
    assert tolerant["allowed_max_full_sl_delta"] == 0.005


def test_resolve_gate_tolerances_reads_cap_sweep_manifest(tmp_path: Path) -> None:
    cap_dir = tmp_path / "cap"
    cap_dir.mkdir()
    (cap_dir / "manifest.json").write_text(
        json.dumps(
            {
                "params": {
                    "selection_min_accepted_jaccard": 0.95,
                    "selection_max_full_sl_delta": 0.005,
                    "selection_max_timeout_delta": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )

    tolerances = resolve_gate_tolerances([cap_dir])

    assert tolerances == {
        "min_accepted_jaccard": 0.95,
        "max_full_sl_delta": 0.005,
        "max_timeout_delta": 0.0,
    }


def test_selected_challenger_selector_uses_portable_cap_suffix(tmp_path: Path) -> None:
    cap_dir = tmp_path / "cap"
    cap_dir.mkdir()
    (cap_dir / "selected_shadow_challenger.json").write_text(
        json.dumps(
            {
                "selected": True,
                "arm": "L0_selected_lgbm_priority_cap_0p15_zge_0p5",
            }
        ),
        encoding="utf-8",
    )

    assert selected_challenger_selector(cap_dir) == "cap_0p15_zge_0p5"


def test_resolve_arm_selector_falls_back_without_selected_challenger(tmp_path: Path) -> None:
    cap_dir = tmp_path / "cap"
    cap_dir.mkdir()

    selector, source = resolve_arm_selector(
        [cap_dir],
        arm_contains="cap_0p10",
        use_selected_challenger=True,
    )

    assert selector == "cap_0p10"
    assert source == "fallback_explicit_arm_contains"


def test_load_window_by_head_uses_exact_selected_arm(tmp_path: Path) -> None:
    cap_dir = tmp_path / "cap"
    cap_dir.mkdir()
    (cap_dir / "manifest.json").write_text(json.dumps({"inputs": {}}), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "arm": "L1_cap_0p3",
                "max_adjustment": 0.3,
                "min_abs_z": 0.0,
                "delta_net_pnl": 1.0,
            },
            {
                "arm": "L1_cap_0p3_zge_0p25",
                "max_adjustment": 0.3,
                "min_abs_z": 0.25,
                "delta_net_pnl": 1.0,
            },
        ]
    ).to_csv(cap_dir / "head_priority_cap_sweep_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"arm": "P0_static_priority", "head": "short_asset", "net_pnl": 10.0, "trade_count": 2},
            {"arm": "P0_static_priority", "head": "short_boll", "net_pnl": 20.0, "trade_count": 3},
            {"arm": "L1_cap_0p3", "head": "short_asset", "net_pnl": 11.0, "trade_count": 2},
            {"arm": "L1_cap_0p3", "head": "short_boll", "net_pnl": 22.0, "trade_count": 4},
            {"arm": "L1_cap_0p3_zge_0p25", "head": "short_asset", "net_pnl": 99.0, "trade_count": 9},
            {"arm": "L1_cap_0p3_zge_0p25", "head": "short_boll", "net_pnl": 99.0, "trade_count": 9},
        ]
    ).to_csv(cap_dir / "head_priority_cap_sweep_by_head.csv", index=False)

    _row, by_head = load_window(cap_dir, arm_contains="cap_0p3")

    assert len(by_head) == 2
    assert set(by_head["shadow_arm"]) == {"L1_cap_0p3"}
    assert set(by_head["shadow_net_pnl"]) == {11.0, 22.0}
