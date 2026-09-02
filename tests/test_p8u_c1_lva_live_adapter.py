from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.inference.p8u_c1_lva_live_adapter import require_coordinates


def _scores() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["one", "two"],
        "__decision_ts__": pd.to_datetime(["2026-07-01T00:00:00Z"] * 2),
        "__symbol__": ["A/USD:USD", "B/USD:USD"],
        "side_name": ["long", "long"],
        "final_score": [0.4, 0.8],
        "base_rank42": [0.3, 0.9],
        "conditional_consensus_rank": [0.2, 0.7],
        "upstream": [0.275, 0.85],
        "ordinary_shadow_consensus_rank": [0.3, 0.9],
        "correctness_rank": [0.9, 0.8],
    })


def test_retains_each_independently_generated_coordinate_family() -> None:
    current = require_coordinates(_scores(), family="Current")
    bcf = require_coordinates(
        _scores().assign(
            final_score=[0.3, 0.9],
            conditional_consensus_rank=[0.3, 0.9],
            correctness_rank=[0.5, 0.5],
        ),
        family="BCF",
    )
    assert current["conditional_consensus_rank"].tolist() == [0.2, 0.7]
    assert bcf["final_score"].tolist() == [0.3, 0.9]
    assert bcf["correctness_rank"].tolist() == [0.5, 0.5]


def test_rejects_outcome_columns() -> None:
    with pytest.raises(ValueError, match="outcome-derived"):
        require_coordinates(_scores().assign(policy_net_bps=[1.0, 2.0]), family="Current")
