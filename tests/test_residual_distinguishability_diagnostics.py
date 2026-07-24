from __future__ import annotations

import numpy as np

from scripts.run_residual_distinguishability_diagnostics import (
    _binary_entropy,
    _difficulty_label,
)


def test_binary_entropy_is_highest_at_half_probability() -> None:
    entropy = _binary_entropy(np.array([0.01, 0.50, 0.99], dtype=np.float32))
    assert entropy[1] > entropy[0]
    assert entropy[1] > entropy[2]


def test_difficulty_label_separates_easy_hard_and_ambiguous_states() -> None:
    labels = _difficulty_label(
        np.array([0.1, 0.8, 0.8, 0.6], dtype=np.float32),
        np.array([0.1, 0.2, 0.9, 0.6], dtype=np.float32),
        np.array([0.1, 0.95, 0.5, 0.6], dtype=np.float32),
    )
    assert labels.tolist() == ["easy", "hard", "ambiguous", "medium"]
