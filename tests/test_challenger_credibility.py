from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.challenger_credibility import (
    PosteriorConfig,
    bayesian_bootstrap_contract_probability,
    consecutive_event_blocks,
    daily_decision_deltas,
    hierarchical_student_t_posterior,
    leave_group_out,
)
from scripts.run_short_default_conditional_mechanism_discrimination import _event_balanced_weights


def _rows() -> pd.DataFrame:
    repeats = 16
    return pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-04-01", periods=4 * repeats, freq="12h", tz="UTC"),
            "parent_rank": [0.91, 0.89, 0.91, 0.89] * repeats,
            "challenger_rank": [0.91, 0.91, 0.91, 0.91] * repeats,
            "ev_after_1pct": [0.01, 0.02, 0.01, -0.01] * repeats,
            "clean_exec": [1, 1, 1, 0] * repeats,
            "adverse_calendar_cell": [0, 1, 1, 0] * repeats,
        }
    )


def test_event_blocks_split_on_non_adverse_days() -> None:
    days = pd.Series(pd.date_range("2026-04-01", periods=5, freq="D", tz="UTC"))
    values = consecutive_event_blocks(days, pd.Series([1, 1, 0, 1, 1]))
    assert values.tolist() == ["event_001", "event_001", "normal", "event_002", "event_002"]


def test_daily_deltas_keep_day_as_evidence_unit() -> None:
    daily = daily_decision_deltas(_rows(), parent_rank="parent_rank", challenger_rank="challenger_rank")
    assert len(daily) == 32
    assert daily["delta_total_ev"].gt(0).any()
    assert daily["event_block"].str.startswith(("event_", "normal")).all()


def test_posterior_and_bootstrap_contract_are_reproducible() -> None:
    daily = daily_decision_deltas(_rows(), parent_rank="parent_rank", challenger_rank="challenger_rank")
    posterior = hierarchical_student_t_posterior(
        daily, config=PosteriorConfig(draws=120, burn_in=80, thin=4, seed=3)
    )
    assert len(posterior) == 30
    assert np.isfinite(posterior["mu"]).all()
    result = bayesian_bootstrap_contract_probability(daily, draws=300, seed=3)
    assert 0.0 <= result["joint_pass_probability"] <= 1.0


def test_leave_group_out_reports_influence() -> None:
    daily = daily_decision_deltas(_rows(), parent_rank="parent_rank", challenger_rank="challenger_rank")
    report = leave_group_out(daily, "month")
    assert len(report) == 2
    assert "influence_share" in report


def test_event_balanced_weights_limit_long_adverse_episode_influence() -> None:
    frame = _rows().copy()
    frame["bad_residual_event_target"] = np.repeat([1, 0, 0, 0], len(frame) // 4)
    weights = _event_balanced_weights(frame)
    adverse = frame["bad_residual_event_target"].to_numpy(bool)
    assert np.isfinite(weights).all()
    assert float(weights[adverse].sum()) < float(weights[~adverse].sum()) * 1.1
