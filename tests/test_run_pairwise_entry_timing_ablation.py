from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_pairwise_entry_timing_ablation import (
    _Constant,
    _fit_pairwise,
    _predict_pairwise,
)


def test_pairwise_heads_reconstruct_signed_expected_delta() -> None:
    x = pd.DataFrame({"x": np.arange(40, dtype=float)})
    utility = np.r_[np.full(20, 0.01), np.full(20, -0.01)]
    labels = pd.DataFrame(
        {
            "action_realized_utility": utility,
            "enter_now_net_ev": np.zeros(40),
            "fill_indicator": np.ones(40),
            "missed_opportunity_ev": np.zeros(40),
        }
    )
    models = _fit_pairwise(x, labels, temperature_bps=25.0, seed=7)
    prediction = _predict_pairwise(models, x)
    assert len(prediction["expected_delta_bps"]) == 40
    assert np.isfinite(prediction["expected_delta_bps"]).all()
    assert (prediction["fill_probability"] == 1.0).all()


def test_constant_classifier_has_binary_probability_shape() -> None:
    probability = _Constant(0.25).predict_proba(pd.DataFrame({"x": [1, 2]}))
    assert probability.shape == (2, 2)
    np.testing.assert_allclose(probability[:, 1], 0.25)
