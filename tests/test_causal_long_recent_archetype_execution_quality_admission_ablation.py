from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_causal_long_recent_archetype_execution_quality_admission_ablation import (
    Arm,
    add_causal_recent_quality,
    apply_arm,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_timestamp": pd.to_datetime(
                ["2026-04-01T09:00:00Z", "2026-04-01T10:00:00Z", "2026-04-01T10:01:00Z"],
                utc=True,
            ),
            "exit_timestamp": pd.to_datetime(
                ["2026-04-01T10:00:00Z", "2026-04-01T11:00:00Z", "2026-04-01T12:00:00Z"],
                utc=True,
            ),
            "side_name": ["long", "long", "long"],
            "policy_archetype": ["a", "a", "a"],
            "net_return_notional": [0.02, -0.02, 0.00],
        }
    )


def test_equal_exit_timestamp_is_not_resolved_at_decision() -> None:
    out = add_causal_recent_quality(_rows(), window_days=7, prior_support=2.0)

    assert out.loc[1, "recent_parent_support"] == 0
    assert out.loc[2, "recent_parent_support"] == 1
    assert out.loc[2, "recent_parent_ev"] == 0.02


def test_future_exit_outcome_cannot_change_earlier_decision_feature() -> None:
    baseline = add_causal_recent_quality(_rows(), window_days=7, prior_support=2.0)
    changed = _rows()
    changed.loc[1, "net_return_notional"] = 0.99  # exits after the final decision
    compared = add_causal_recent_quality(changed, window_days=7, prior_support=2.0)

    columns = [
        "recent_parent_support",
        "recent_local_support",
        "recent_parent_ev",
        "recent_parent_hit_rate",
        "recent_shrunk_ev",
        "recent_shrunk_hit_rate",
        "recent_quality_score",
    ]
    pd.testing.assert_series_equal(baseline.loc[2, columns], compared.loc[2, columns])


def test_rank_and_threshold_nudges_follow_causal_quality_sign() -> None:
    candidates = pd.DataFrame(
        {
            "recent_quality_score": [-1.0, 1.0],
            "base_strategy_threshold": [0.94, 0.94],
        }
    )

    out = apply_arm(candidates, Arm("both", 7, "rank_threshold", 0.02))

    np.testing.assert_allclose(out["portfolio_rank_adjustment"], [-0.02, 0.02])
    np.testing.assert_allclose(out["base_strategy_threshold"], [0.96, 0.92])
