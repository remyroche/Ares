import pandas as pd
import pytest

from src.training.steps.pre_training.standardized_labeling_interface import (
    assert_labels_sigma_scaled,
)


def test_assert_labels_sigma_scaled_single_observation_does_not_raise():
    labels = pd.DataFrame(
        {
            "target_primary": [0.5],
        }
    )

    # Should not raise even though only a single non-null observation is present.
    assert_labels_sigma_scaled(labels)


def test_assert_labels_sigma_scaled_variance_out_of_bounds_raises():
    labels = pd.DataFrame(
        {
            "target_primary": [0.0, 3.0],
        }
    )

    with pytest.raises(ValueError):
        assert_labels_sigma_scaled(labels, tolerance=0.35)
