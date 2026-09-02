"""Focused correctness tests for the research-only matched exit attribution."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_strict_r3_exact_1m_rich_matched_attribution.py"
SPEC = importlib.util.spec_from_file_location("matched_attribution", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _routed() -> pd.DataFrame:
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    return pd.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "timestamp": [timestamp, timestamp, timestamp],
        "symbol": ["A/USD:USD", "B/USD:USD", "C/USD:USD"],
        "side_name": ["long", "long", "long"],
        "entry_ts": [timestamp, timestamp, timestamp],
        "priority_bps": [130.0, 90.0, 60.0],
        "bcf_mc1_expected_bps": [130.0, 90.0, 60.0],
        "current_v5_mc1_expected_bps": [80.0, 70.0, 40.0],
    })


def _outcomes() -> pd.DataFrame:
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    return pd.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "decision_timestamp": [timestamp, timestamp, timestamp],
        "entry_timestamp": [timestamp, timestamp, timestamp],
        "entry_price": [100.0, 100.0, np.nan],
        "exit_timestamp": [timestamp + pd.Timedelta(minutes=3), timestamp + pd.Timedelta(minutes=5), pd.NaT],
        "exit_price": [102.0, 101.0, np.nan],
        "gross_bps": [200.0, 100.0, np.nan],
        "net_bps": [100.0, 0.0, np.nan],
        "exit_reason": ["trailing", "timeout_h12", MODULE.INVALID_OUTCOME_REASON],
        "outcome_available": [True, True, False],
        "outcome_invalid_reason": ["", "", "incomplete_h12_minute_path"],
        "outcome_source": ["synthetic", "synthetic", "synthetic"],
    })


def test_label_complete_filter_happens_after_target_free_route_without_capacity_reservation() -> None:
    routed = _routed()
    candidates, population = MODULE._portfolio_candidates(
        routed, _outcomes(), arm="synthetic_exact_1m",
    )
    # The coverage receipt preserves all predeclared routed identities.
    assert population["candidate_id"].tolist() == ["a", "b", "c"]
    assert population["outcome_available"].tolist() == [True, True, False]
    # Only after that audit does evaluation remove the incomplete outcome.
    assert candidates["candidate_id"].tolist() == ["a", "b"]
    assert np.isclose(candidates.loc[candidates.candidate_id.eq("a"), "portfolio_priority_adjustment"].iloc[0], 130.0)
    assert np.isclose(candidates.loc[candidates.candidate_id.eq("b"), "portfolio_priority_adjustment"].iloc[0], 90.0)
    assert candidates["simple_policy_exit_reason"].tolist() == ["trailing", "timeout_h12"]


def test_priority_is_only_bcf_mc1_with_no_timestamp_local_rank_rule() -> None:
    candidates, _ = MODULE._portfolio_candidates(_routed(), _outcomes(), arm="synthetic_exact_1m")
    # Portfolio ordering uses the explicit BCF-MC1 adjustment (130 > 90).
    # The fixed canonical margin-slot portfolio deliberately receives a
    # constant pass-through rank, rather than a new timestamp-local rule.
    by_id = candidates.set_index("candidate_id")
    assert by_id.loc["a", "portfolio_priority_adjustment"] > by_id.loc["b", "portfolio_priority_adjustment"]
    assert by_id.loc["a", "normalized_rank_score"] == 1.0
    assert by_id.loc["b", "normalized_rank_score"] == 1.0


def test_attribution_labels_separate_legacy_policy_from_clean_bar_resolution() -> None:
    summary = pd.DataFrame({
        "arm": [
            "parent_proxy_15m_decision",
            "frozen_rich_15m_aggregated_decision",
            "exact_1m_rich_v1_decision",
        ],
        "label_complete_candidates_after_route": [2, 2, 2],
        "portfolio_accepted_trades": [2, 2, 2],
        "net_ev_bps_per_trade": [0.0, 10.0, 20.0],
        "net_sum_bps": [0.0, 20.0, 40.0],
        "portfolio_net_pnl_quote": [0.0, 0.1, 0.2],
        "portfolio_final_wallet": [1.0, 1.1, 1.2],
        "portfolio_max_drawdown": [-0.2, -0.1, -0.05],
        "portfolio_sortino": [0.0, 0.5, 1.0],
        "portfolio_worst_week_return": [-0.1, 0.0, 0.1],
    })
    overall, _, _ = MODULE._comparison_deltas(summary, pd.DataFrame(), pd.DataFrame())
    names = set(overall["comparison"].astype(str))
    assert "legacy_policy_and_resolution_delta_exact1m_decision_minus_parent15m" in names
    assert "clean_resolution_delta_exact1m_decision_minus_frozen_rich_15m_aggregated_decision" in names
    clean = overall.loc[
        overall["comparison"].eq(
            "clean_resolution_delta_exact1m_decision_minus_frozen_rich_15m_aggregated_decision"
        )
    ].iloc[0]
    assert clean["delta_net_ev_bps_per_trade"] == 10.0
