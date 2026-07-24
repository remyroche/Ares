from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import compare_packb_outer_oof_feature_routes as gate


def _frame(prediction: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "side_name": ["long", "long"],
            "__ts__": pd.to_datetime(
                ["2026-04-01T00:00:00Z", "2026-04-01T00:00:00Z"], utc=True
            ),
            "__symbol__": ["BTC", "ETH"],
            "outer_fold": ["outer_1", "outer_1"],
            "prediction": prediction,
            gate.TARGET_COLUMN: [1.0, 0.0],
            gate.WEIGHT_COLUMN: [1.0, 1.0],
            gate.ECONOMIC_COLUMN: [0.02, -0.01],
        }
    )


def test_exact_pair_binds_costs_and_identity() -> None:
    paired = gate._exact_pair(_frame([0.9, 0.1]), _frame([0.8, 0.2]))
    assert paired["candidate_id"].tolist() == ["a", "b"]
    changed = _frame([0.8, 0.2])
    changed.loc[0, gate.ECONOMIC_COLUMN] = 0.03
    with pytest.raises(gate.PackBOuterRouteGateError, match="net__"):
        gate._exact_pair(_frame([0.9, 0.1]), changed)


def test_winner_requires_all_four_core_metrics_to_be_higher() -> None:
    left = {
        "objective": 0.1,
        "weighted_rank_ic": 0.1,
        "top10_net_return_lift": 0.1,
        "relative_rmse_gain": 0.1,
    }
    right = {name: value + 0.01 for name, value in left.items()}
    assert gate._winner(left, right) == "right"
    right["weighted_rank_ic"] = np.float64(0.05)
    assert gate._winner(left, right) == "mixed_requires_review"
