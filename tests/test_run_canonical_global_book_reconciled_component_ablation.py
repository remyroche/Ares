from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_canonical_global_book_reconciled_component_ablation import (
    _causal_persistence,
    _component_sum,
)


def test_causal_persistence_uses_only_targets_available_before_anchor() -> None:
    base = pd.Timestamp("2025-01-01T00:00:00Z")
    population = pd.DataFrame(
        {
            "cohort_anchor_utc": pd.to_datetime(
                [base, base + pd.Timedelta(hours=1)], utc=True
            ),
            "label_available_utc": pd.to_datetime(
                [
                    base + pd.Timedelta(hours=2),
                    base + pd.Timedelta(hours=4),
                ],
                utc=True,
            ),
            "target_delta": [1.0, 2.0],
            "target_valid": [True, True],
        }
    )
    evaluation = pd.DataFrame(
        {
            "cohort_anchor_utc": pd.to_datetime(
                [
                    base + pd.Timedelta(hours=2),
                    base + pd.Timedelta(hours=3),
                    base + pd.Timedelta(hours=5),
                ],
                utc=True,
            )
        }
    )
    prediction = _causal_persistence(population, evaluation)
    assert np.isnan(prediction[0])
    assert prediction[1] == 1.0
    assert prediction[2] == 2.0


def test_component_sum_reconciles_values_but_not_sign_probabilities() -> None:
    anchor = pd.Timestamp("2025-01-01T00:00:00Z")
    rows = []
    for index, band in enumerate(("B1", "B2", "B3", "B4"), start=1):
        rows.append(
            {
                "cohort_anchor_utc": anchor,
                "horizon_hours": 12,
                "book_fraction": 0.1,
                "fold_id": 0,
                "validation_start_utc": anchor,
                "validation_end_utc": anchor + pd.Timedelta(days=14),
                "target_delta": float(index),
                "target_valid": True,
                "delta_prediction": float(index) / 2,
                "zero_delta_prediction": 0.0,
                "constant_delta_prediction": 0.1,
                "causal_persistence_prediction": 0.2,
                "delta_direct_mean_net": 11.0,
                "delta_mapped_score_mean": 1.0,
                "sign_probability": 0.9,
                "constant_sign_probability": 0.5,
                "label_available_utc": anchor,
                "after_target_available_utc": anchor,
                "training_rows": 500,
                "training_max_after_target_available_utc": anchor,
                "model_name": f"component_{band}",
            }
        )
    result = _component_sum(pd.DataFrame(rows)).iloc[0]
    assert result["target_delta"] == 10.0
    assert result["delta_prediction"] == 5.0
    assert np.isnan(result["sign_probability"])
    assert np.isnan(result["constant_sign_probability"])
