from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.bootstrap_cross_era_transition_classifier import (
    paired_block_bootstrap,
)


def test_paired_block_bootstrap_detects_better_predictions() -> None:
    rows = []
    rng = np.random.default_rng(17)
    for group in range(10):
        for row in range(20):
            target = float((group + row) % 4 == 0)
            model = 0.75 if target else 0.10
            rows.append(
                {
                    "cv_group_id": f"g{group}",
                    "target": target,
                    "model_prediction": model + rng.normal(0, 0.01),
                    "prior_prediction": 0.25,
                }
            )
    result = paired_block_bootstrap(
        pd.DataFrame(rows), draws=200, random_state=7
    )
    assert result["delta_brier"] < 0
    assert result["delta_average_precision"] > 0
    assert result["delta_brier_ci_high"] < 0
    assert result["bootstrap_probability_brier_improves"] == 1.0
