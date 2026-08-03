import json

import numpy as np
import pandas as pd

from scripts.run_residual_selected_exit_opportunity_counterfactual import (
    decode_fixed_12h,
    global_book_weights,
    metric_row,
)


def test_fractional_global_book_weights_preserve_exact_capacity():
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d", "e"],
            "score": [3.0, 2.0, 2.0, 2.0, 1.0],
        }
    )
    weights, meta = global_book_weights(frame, "score", 0.40)
    assert meta["selected_rows"] == 2
    assert np.isclose(weights.sum(), 2.0)
    assert weights.iloc[0] == 1.0
    assert np.allclose(weights.iloc[1:4], 1.0 / 3.0)


def test_fixed_12h_return_is_side_signed():
    start = pd.Timestamp("2025-03-01T00:00:00Z").value
    timestamp = start + np.arange(720, dtype=np.int64) * 60_000_000_000
    payload = json.dumps(
        {
            "timestamp": timestamp.tolist(),
            "close": np.linspace(100.0, 110.0, 720).tolist(),
        }
    )
    long_return, first, last = decode_fixed_12h(payload, 100.0, "long")
    short_return, _, _ = decode_fixed_12h(payload, 100.0, "short")
    assert np.isclose(long_return, 0.10)
    assert np.isclose(short_return, -0.10)
    assert first == pd.Timestamp("2025-03-01T00:00:00Z")
    assert last == first + pd.Timedelta(minutes=719)


def test_metric_row_reconciles_oracle_regret_and_exit_contributions():
    frame = pd.DataFrame(
        {
            "w": [1.0, 1.0],
            "deployed_gross": [0.01, -0.02],
            "deployed_net": [0.0, -0.03],
            "cost": [0.01, 0.01],
            "oracle_mfe_gross": [0.04, 0.01],
            "oracle_mfe_net": [0.03, 0.0],
            "pre_exit_mfe_gross": [0.02, 0.0],
            "pre_exit_mfe_net": [0.01, -0.01],
            "fixed_12h_gross": [0.03, -0.01],
            "fixed_12h_net": [0.02, -0.02],
            "oracle_regret": [0.03, 0.03],
            "fixed_12h_delta_vs_deployed": [0.02, 0.01],
            "pre_exit_uncaptured_net_opportunity": [0.01, 0.0],
            "opportunity_0bps": [1, 0],
            "opportunity_25bps": [1, 0],
            "opportunity_50bps": [1, 0],
            "deployed_positive": [0, 0],
            "fixed_12h_positive": [1, 0],
            "full_stop": [0, 1],
            "timeout": [1, 0],
            "capture_ratio": [0.5, np.nan],
            "economic_capture_ratio": [0.4, np.nan],
        }
    )
    row = metric_row(
        frame,
        month="2025-03",
        fraction=0.10,
        scope="global",
        weight="w",
        selection={},
    )
    assert np.isclose(row["oracle_regret_bps"], 300.0)
    assert np.isclose(row["full_stop_oracle_regret_contribution_bps"], 150.0)
    assert np.isclose(row["timeout_oracle_regret_contribution_bps"], 150.0)
    assert np.isclose(row["capture_ratio"], 0.5)
    assert np.isclose(row["capture_ratio_expected_support"], 1.0)
