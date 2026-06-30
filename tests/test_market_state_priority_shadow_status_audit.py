from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.audit_market_state_priority_shadow_status import audit_priority_shadow_status


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_cap_sweep(root: Path, *, selected: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    contract = {
        "changes_scores_or_ranks": False,
        "changes_thresholds": False,
        "changes_position_sizing": False,
        "changes_auction_ordering": True,
        "qfail_active": False,
        "head_health_active": False,
        "market_state_threshold_controller_active": False,
        "priority_adjustment_column": "portfolio_priority_adjustment",
    }
    _write_json(root / "manifest.json", {"contract": contract})
    row = {
        "arm": "L0_selected_lgbm_priority_cap_0p15_zge_0p5",
        "max_adjustment": 0.15,
        "min_abs_z": 0.5,
        "active_schedule_share": 0.34,
        "coverage": 1.0,
        "trade_count": 117,
        "net_pnl": 330.0,
        "delta_net_pnl": 22.0,
        "full_sl_rate": 0.42,
        "delta_full_sl_rate": -0.01,
        "timeout_rate": 0.09,
        "delta_timeout_rate": 0.0,
        "accepted_jaccard": 0.96,
        "entrants": 2,
        "removed": 2,
        "net_replacement_pnl": 17.0,
        "net_action_pnl_delta": 22.0,
        "defensive_success": 8.0,
        "gate_passed": True,
    }
    pd.DataFrame([row]).to_csv(root / "head_priority_cap_sweep_metrics.csv", index=False)
    _write_json(
        root / "selected_shadow_challenger.json",
        {
            "selected": selected,
            "arm": row["arm"] if selected else None,
            "selected_row": row if selected else {},
        },
    )


def _write_promotion(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "manifest.json",
        {
            "params": {
                "resolved_arm_contains": "cap_0p15_zge_0p5",
                "arm_selector_source": "selected_shadow_challenger:/tmp/cap",
            }
        },
    )
    _write_json(
        root / "market_state_priority_shadow_promotion_gate.json",
        {
            "passed": False,
            "failures": [
                "median_delta_net_pnl_not_positive",
                "positive_delta_window_share_below_50pct",
                "fewer_than_2_action_windows",
                "fewer_than_2_positive_action_windows",
            ],
            "window_count": 3,
            "action_window_count": 1,
            "positive_action_window_count": 1,
            "median_delta_net_pnl": 0.0,
            "q25_delta_net_pnl": 0.0,
            "positive_delta_window_share": 1 / 3,
            "nonnegative_delta_window_share": 1.0,
            "min_accepted_jaccard": 0.96,
            "min_coverage": 1.0,
            "max_full_sl_delta": 0.0,
            "max_timeout_delta": 0.0,
        },
    )
    pd.DataFrame(
        {
            "window_label": ["a", "b", "c"],
            "coverage": [1.0, 1.0, 1.0],
            "delta_full_sl_rate": [-0.01, 0.0, 0.0],
            "delta_timeout_rate": [0.0, 0.0, 0.0],
            "delta_net_pnl": [22.0, 0.0, 0.0],
            "accepted_jaccard": [0.96, 1.0, 1.0],
            "entrants": [2, 0, 0],
            "removed": [2, 0, 0],
            "defensive_success": [8.0, 0.0, 0.0],
        }
    ).to_csv(root / "market_state_priority_shadow_window_summary.csv", index=False)


def test_priority_shadow_status_accepts_contract_clean_non_promoted_shadow(tmp_path: Path) -> None:
    cap = tmp_path / "cap"
    promo = tmp_path / "promo"
    out = tmp_path / "out"
    _write_cap_sweep(cap)
    _write_promotion(promo)

    summary = audit_priority_shadow_status(
        cap_sweep_dir=cap,
        promotion_audit_dir=promo,
        output_dir=out,
    )

    assert summary["passed"] is True
    assert summary["operational_status"] == "shadow_only"
    assert summary["selected_selector"] == "cap_0p15_zge_0p5"
    assert (out / "market_state_priority_shadow_status_evidence.csv").exists()
    assert (out / "market_state_priority_shadow_status_report.md").exists()


def test_priority_shadow_status_flags_missing_selected_challenger(tmp_path: Path) -> None:
    cap = tmp_path / "cap"
    promo = tmp_path / "promo"
    out = tmp_path / "out"
    _write_cap_sweep(cap, selected=False)
    _write_promotion(promo)

    summary = audit_priority_shadow_status(
        cap_sweep_dir=cap,
        promotion_audit_dir=promo,
        output_dir=out,
    )

    assert summary["passed"] is False
    assert "P1" in summary["missing"]
