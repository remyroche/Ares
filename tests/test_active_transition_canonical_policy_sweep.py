from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGING = Path(__file__).resolve().parents[1] / "scripts"
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from run_active_transition_canonical_policy_sweep import (  # noqa: E402
    _active_validation_metadata,
    _monthly_accepted_metrics,
    _stable_top_k,
    replacement_attribution,
    select_arm,
    to_replay_candidates,
)


def _cohort() -> pd.DataFrame:
    decision = pd.date_range("2025-02-01 01:00", periods=5, freq="h", tz="UTC")
    gross = np.array([0.03, 0.02, 0.015, 0.01, 0.005])
    return pd.DataFrame(
        {
            "__ts__": decision - pd.Timedelta(hours=1),
            "__symbol__": list("ABCDE"),
            "side_name": ["long", "long", "short", "short", "long"],
            "candidate_id": list("abcde"),
            "execution_decision_utc": decision,
            "execution_label_end_utc": decision + pd.Timedelta(hours=12),
            "execution_gross_ev_12h": gross,
            "execution_cost_return": 0.01,
            "execution_net_ev_12h": gross - 0.01,
            "execution_exit_minute": [60, 120, 180, 240, 300],
            "execution_exit_class": [
                "trailing",
                "timeout",
                "full_stop",
                "adverse_exit",
                "timeout",
            ],
            "score_raw": [0.9, 0.8, 0.7, 0.6, 0.5],
            "mapped_direct_net": [-0.01, -0.02, -0.03, -0.04, -0.05],
            "active_transition_probability_oof": [1.0, 0.0, 0.0, 0.0, 0.0],
            "expost_transition_active": [1, 0, 0, 0, 0],
            "transition_event_id": ["event", None, None, None, None],
        }
    )


def test_stable_top_k_uses_candidate_id_for_ties() -> None:
    frame = _cohort()
    frame["tie"] = 1.0
    selected = _stable_top_k(frame, score_column="tie", count=2)
    assert selected["candidate_id"].tolist() == ["a", "b"]


def test_sign_safe_trust_discount_penalizes_negative_risky_score() -> None:
    frame = _cohort()
    baseline = _stable_top_k(frame, score_column="mapped_direct_net", count=2)
    selected = select_arm(
        frame,
        score_column="mapped_direct_net",
        baseline_ids=set(baseline["candidate_id"]),
        baseline_count=2,
        score_scale=0.03,
        policy="trust_discount",
        value=1.0,
    )
    assert set(selected["candidate_id"]) == {"b", "c"}
    risky = frame.loc[frame["candidate_id"].eq("a"), "mapped_direct_net"].iloc[0]
    assert risky == -0.01


def test_threshold_increase_only_removes_frozen_baseline_rows() -> None:
    frame = _cohort()
    baseline = _stable_top_k(frame, score_column="score_raw", count=2)
    selected = select_arm(
        frame,
        score_column="score_raw",
        baseline_ids=set(baseline["candidate_id"]),
        baseline_count=2,
        score_scale=0.2,
        policy="threshold_increase",
        value=1.0,
    )
    assert selected["candidate_id"].tolist() == ["b"]


def test_exposure_reduction_preserves_book_and_scales_active_row() -> None:
    frame = _cohort()
    baseline = _stable_top_k(frame, score_column="score_raw", count=2)
    selected = select_arm(
        frame,
        score_column="score_raw",
        baseline_ids=set(baseline["candidate_id"]),
        baseline_count=2,
        score_scale=0.2,
        policy="exposure_reduction",
        value=0.5,
    ).set_index("candidate_id")
    assert set(selected.index) == {"a", "b"}
    assert np.isclose(selected.loc["a", "portfolio_size_multiplier"], 0.5)
    assert np.isclose(selected.loc["b", "portfolio_size_multiplier"], 1.0)


def test_replay_translation_uses_exact_cost_and_exit_minute() -> None:
    frame = _cohort().iloc[:2].copy()
    frame["policy_score"] = frame["score_raw"]
    frame["policy_global_rank_pct"] = [1.0, 0.5]
    frame["portfolio_size_multiplier"] = 1.0
    replay = to_replay_candidates(frame)
    expected_exit = pd.to_datetime(
        frame["execution_decision_utc"], utc=True
    ) + pd.to_timedelta(frame["execution_exit_minute"], unit="m")
    assert pd.to_datetime(replay["exit_timestamp"], utc=True).tolist() == expected_exit.tolist()
    assert np.allclose(replay["gross_return"], frame["execution_gross_ev_12h"])
    assert np.allclose(replay["net_return"], frame["execution_net_ev_12h"])
    assert np.allclose(replay["fees_bps"], 100.0)
    assert np.allclose(replay["expected_friction_bps"], 0.0)


def test_replacement_attribution_separates_kept_removed_and_added() -> None:
    frame = _cohort()
    metrics, relation = replacement_attribution(
        frame,
        baseline_ids={"a", "b"},
        selected_ids={"b", "c"},
    )
    by_id = relation.set_index("candidate_id")["selection_relation"]
    assert by_id.to_dict() == {
        "a": "removed",
        "b": "kept",
        "c": "newly_added",
    }
    assert metrics["kept_rows"] == 1
    assert metrics["removed_rows"] == 1
    assert metrics["newly_added_rows"] == 1
    assert np.isclose(metrics["replacement_sum_net_delta"], -0.015)


def test_chronological_validation_contract_does_not_claim_grouped_oof() -> None:
    metadata = _active_validation_metadata(
        "chronological_label_oos_pooled_geometry"
    )
    assert "CHRONOLOGICAL_LABEL_OOS" in metadata["status"]
    assert "pooled" in metadata["blocker"]
    assert "lambda grid" in metadata["blocker"]


def test_prior_frozen_contract_removes_same_cohort_grid_blocker() -> None:
    metadata = _active_validation_metadata(
        "chronological_label_oos_pooled_geometry", "prior_frozen"
    )
    assert "prior cohort" in metadata["blocker"]
    assert "same cohort" not in metadata["blocker"]


def test_monthly_metrics_preserve_active_and_outside_pnl() -> None:
    accepted = pd.DataFrame(
        {
            "month": ["2025-02", "2025-02"],
            "position_net_return": [0.10, -0.05],
            "position_size": [100.0, 200.0],
            "expost_transition_active": [1, 0],
            "transition_event_id": ["event", None],
        }
    )
    row = _monthly_accepted_metrics(accepted)[0]
    assert row["trades"] == 2
    assert row["active_trades"] == 1
    assert row["active_transition_events"] == 1
    assert np.isclose(row["net_pnl"], 0.0)
    assert np.isclose(row["active_net_pnl"], 10.0)
    assert np.isclose(row["outside_net_pnl"], -10.0)
