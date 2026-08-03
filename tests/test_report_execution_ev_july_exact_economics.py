from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.report_execution_ev_july_exact_economics import (
    build_portfolio_candidates,
    cohort_metrics,
    portfolio_replays,
    score_diagnostics,
    top_k_tie_sensitivity,
)
from scripts.score_execution_ev_forward_population import apply_global_admission


def _population(rows: int = 10) -> pd.DataFrame:
    decision = pd.Timestamp("2026-07-20 01:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "candidate_id": [f"c{i:02d}" for i in range(rows)],
            "__ts__": [decision - pd.Timedelta(hours=1)] * rows,
            "__symbol__": [f"S{i:02d}" for i in range(rows)],
            "side_name": ["long", "short"] * (rows // 2),
            "execution_decision_utc": [
                decision + pd.Timedelta(hours=i) for i in range(rows)
            ],
            "mapped_execution_ev": [0.02, 0.01, 0.01] + [-0.01] * (rows - 3),
            "execution_net_ev_12h": np.linspace(-0.03, 0.03, rows),
            "execution_gross_ev_12h": np.linspace(-0.02, 0.04, rows),
            "execution_cost_return": [0.01] * rows,
            "execution_mfe_return_12h": np.linspace(0.0, 0.08, rows),
            "execution_mae_return_12h": np.linspace(-0.05, 0.0, rows),
            "execution_exit_hour": [2.0] * rows,
            "execution_label_end_utc": [
                decision + pd.Timedelta(hours=i + 12) for i in range(rows)
            ],
            "execution_entry_price": [100.0] * rows,
            "execution_exit_price": [101.0] * rows,
            "execution_exit_reason": ["timeout"] * rows,
            "policy_archetype": ["side_parent"] * rows,
            "base_oof_score": np.arange(rows, dtype=float),
            "base_alpha_ev": np.arange(rows, dtype=float),
            "existing_alpha_ev": np.arange(rows, dtype=float),
            "residual_delta_ev": np.arange(rows, dtype=float),
            "final_direct_net_raw": np.arange(rows, dtype=float),
            "final_capture_probability": np.linspace(0.1, 0.9, rows),
            "oof_clean_favorable_probability": np.linspace(0.1, 0.9, rows),
            "pred_peak_MFE_12h_ATR": np.arange(rows, dtype=float),
            "catboost_p_0": [0.1] * rows,
            "catboost_p_1": [0.1] * rows,
            "catboost_p_2": [0.2] * rows,
            "catboost_p_3": [0.2] * rows,
            "catboost_p_4": [0.1] * rows,
            "catboost_p_5": [0.1] * rows,
            "catboost_p_6": [0.2] * rows,
        }
    )
    frame = apply_global_admission(frame, top_k_fraction=0.20)
    frame["utc_date"] = frame["execution_decision_utc"].dt.strftime("%Y-%m-%d")
    frame["positive_net"] = frame["execution_net_ev_12h"] > 0
    frame["negative_net"] = frame["execution_net_ev_12h"] < 0
    frame["catboost_usable_path_probability"] = frame[
        ["catboost_p_2", "catboost_p_3", "catboost_p_4", "catboost_p_5"]
    ].sum(axis=1)
    frame["catboost_adverse_path_probability"] = frame[
        ["catboost_p_0", "catboost_p_1", "catboost_p_6"]
    ].sum(axis=1)
    return frame


def _policy(path: Path, *, count_cap: bool = False) -> Path:
    payload = {
        "portfolio_policy": {
            "schema_version": "portfolio_policy_v2",
            "portfolio_policy_version": "global_auction_v1",
            "enforce_position_count_cap": count_cap,
            "concurrency": {
                "max_concurrent_positions": 2,
                "max_concurrent_per_side": 1,
                "max_concurrent_per_symbol": 1,
                "max_new_entries_per_bar": 2,
            },
            "allocation": {"max_total_wallet_allocation_pct": 0.7},
            "selection": {
                "global_threshold_floor": 0.0,
                "occupancy_threshold_alpha": 0.0,
            },
            "risk": {"cooldown_hours_after_loss": 0.0},
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_global_cutoff_tie_is_reported_without_changing_positive_admission() -> None:
    frame = _population()
    frame = apply_global_admission(
        frame.drop(
            columns=[
                "global_top10_capacity_member",
                "globally_admitted_floor_0bps",
                "globally_admitted_floor_25bps",
                "globally_admitted_floor_50bps",
                "globally_admitted",
                "global_rank",
            ]
        ),
        top_k_fraction=0.40,
    )
    report = top_k_tie_sensitivity(frame, top_k_fraction=0.40)
    assert report["selected_global_top_k_rows"] == 4
    assert report["strictly_above_cutoff_rows"] == 3
    assert report["tied_at_cutoff_rows"] == 7
    assert report["selected_from_cutoff_tie_rows"] == 1
    assert report["positive_floor_unaffected_by_cutoff_tie"]


def test_portfolio_candidates_use_actual_exit_and_do_not_add_friction() -> None:
    frame = _population()
    candidates = build_portfolio_candidates(frame, "global_top10_capacity_member")
    expected = candidates["timestamp"] + pd.Timedelta(hours=2)
    assert candidates["exit_timestamp"].equals(expected)
    assert candidates["expected_friction_bps"].eq(0.0).all()
    joined = candidates.set_index("candidate_id").join(
        frame.set_index("candidate_id")["execution_net_ev_12h"]
    )
    assert np.allclose(joined["net_return"], joined["execution_net_ev_12h"])


def test_report_tables_cover_heads_cohorts_and_constraint_replays(
    tmp_path: Path,
) -> None:
    frame = _population()
    cohorts = cohort_metrics(frame)
    diagnostics = score_diagnostics(frame)
    summary, decisions, _, side, contract = portfolio_replays(
        frame, policy_path=_policy(tmp_path / "policy.json"), initial_wallet=1_000.0
    )
    assert {
        "full_population",
        "global_top10",
        "admitted_gt_0bps",
        "admitted_gt_25bps",
        "admitted_gt_50bps",
    }.issubset(cohorts["cohort"])
    assert {
        "base_score",
        "residual_enhanced_alpha",
        "direct_execution_ev",
        "capture_probability",
        "peak_mfe_aux",
    }.issubset(diagnostics["head"])
    assert set(summary["replay_arm"]) == {
        "config_faithful",
        "explicit_count_cap_2",
    }
    assert decisions.loc[decisions["accepted"], "position_net_return"].notna().all()
    assert not side.empty
    assert contract["contract"]["max_concurrent_per_symbol"] == 1
