from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_canonical_conversion_shared_horizon_ablation import (
    _equal_horizon_weights,
    _training_scale,
)


def test_training_scale_is_robust_and_positive() -> None:
    center, scale = _training_scale(pd.Series([-1.0, 0.0, 1.0, 100.0]))
    assert center == 0.5
    assert np.isfinite(scale)
    assert scale > 0.0


def test_equal_horizon_weights_equalize_total_loss_mass() -> None:
    horizons = pd.Series([3, 3, 3, 12])
    weights = _equal_horizon_weights(horizons)
    assert np.isclose(weights[horizons.eq(3)].sum(), 2.0)
    assert np.isclose(weights[horizons.eq(12)].sum(), 2.0)
    assert np.isclose(weights.mean(), 1.0)
