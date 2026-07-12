from __future__ import annotations

import numpy as np

from scripts.report_meta_residual_interaction_validation import _design


def test_interaction_design_contains_local_slopes() -> None:
    numeric = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    group = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    additive = _design(numeric, group, interaction=False)
    interaction = _design(numeric, group, interaction=True)
    assert additive.shape == (2, 4)
    assert interaction.shape == (2, 8)
    np.testing.assert_array_equal(interaction[0, -4:], [1.0, 2.0, 0.0, 0.0])
    np.testing.assert_array_equal(interaction[1, -4:], [0.0, 0.0, 3.0, 4.0])
