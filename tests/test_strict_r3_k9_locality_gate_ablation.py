from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_strict_r3_k9_locality_gate_ablation import _gate, _quantiles


def test_locality_thresholds_are_fit_only_on_training_rows() -> None:
    train = pd.DataFrame({
        "k9_top2_margin": [0.01, 0.02, 0.03, 0.04],
        "cluster_recent_7d_support": [10.0, 20.0, 30.0, 40.0],
    })
    held = pd.DataFrame({
        "k9_top2_margin": [0.01, 1.0],
        "cluster_recent_7d_support": [10.0, 1_000_000.0],
    })
    q = _quantiles(train)
    assert q["margin_q95"] < 0.05
    assert q["support_q90"] < np.log1p(100.0)
    gate = _gate(held, q, "history_hard_m90_s75")
    assert gate.tolist() == [0.0, 1.0]


def test_soft_locality_gate_is_bounded_and_candidate_specific() -> None:
    train = pd.DataFrame({
        "k9_top2_margin": np.linspace(0.0, 0.10, 101),
        "cluster_recent_7d_support": np.linspace(1.0, 101.0, 101),
    })
    held = pd.DataFrame({
        "k9_top2_margin": [0.01, 0.09],
        "cluster_recent_7d_support": [2.0, 100.0],
    })
    gate = _gate(held, _quantiles(train), "history_soft_m75_95_x_s50_90")
    assert np.all((gate >= 0.0) & (gate <= 1.0))
    assert gate[1] > gate[0]
