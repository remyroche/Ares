from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import run_top30_aware_contextual_meta_training_ablation as mod


def test_timestamp_reference_rank_is_timestamp_local() -> None:
    timestamps = pd.Series(["2026-05-25 00:00Z"] * 3 + ["2026-05-25 01:00Z"] * 3)
    score = np.array([0.1, 0.3, 0.2, 0.9, 0.7, 0.8], dtype=np.float32)

    ranks = mod._timestamp_reference_rank(timestamps, score)

    assert ranks.tolist() == pytest.approx([0.0, 1.0, 0.5, 1.0, 0.0, 0.5])


def test_top30_weights_keep_equal_timestamp_mass_and_emphasize_swaps() -> None:
    timestamps = pd.Series(["2026-05-25 00:00Z"] * 4 + ["2026-05-25 01:00Z"] * 4)
    y = np.array([0, 1, 1, 0, 1, 0, 1, 0], dtype=np.int8)
    # Timestamp-local ranks put row 0 and row 5 into false-positive top30,
    # and row 2 and row 6 just below the cutoff as missed positives.
    ref_score = np.array([0.95, 0.10, 0.60, 0.20, 0.10, 0.95, 0.60, 0.20], dtype=np.float32)
    spec = mod.WeightSpec(
        "J3_test",
        "test",
        timestamp_weight=True,
        tail_weight=True,
        swap_weight=True,
        alpha=1.0,
        beta=2.0,
        gamma=2.0,
        tau=0.05,
    )

    weight, diag = mod._build_top30_weights(timestamps=timestamps, y=y, ref_score=ref_score, spec=spec)

    assert weight is not None
    mass = pd.Series(weight).groupby(pd.to_datetime(timestamps, utc=True)).sum()
    assert mass.iloc[0] == pytest.approx(mass.iloc[1], abs=1e-6)
    assert weight.mean() == pytest.approx(1.0, abs=1e-6)
    assert diag["swap_false_positive_top30_rows"] == 2
    assert diag["swap_missed_positive_boundary_rows"] == 2
    assert diag["timestamp_mass_cv"] <= 1e-6
    assert weight[0] > weight[1]
    assert weight[2] > weight[3]


def test_unweighted_spec_returns_none() -> None:
    spec = mod.WeightSpec("J0", "current", timestamp_weight=False)
    weight, diag = mod._build_top30_weights(
        timestamps=pd.Series(["2026-05-25 00:00Z", "2026-05-25 00:00Z"]),
        y=np.array([0, 1], dtype=np.int8),
        ref_score=np.array([0.2, 0.8], dtype=np.float32),
        spec=spec,
    )

    assert weight is None
    assert diag["weight_mode"] == "current_unweighted"


def test_j3_grid_cap_is_exact() -> None:
    args = type(
        "Args",
        (),
        {
            "alpha_grid": [0.5, 1.0],
            "beta_grid": [1.0, 2.0],
            "gamma_grid": [1.0, 2.0],
            "tau_grid": [0.03, 0.05],
            "max_j2_configs": 2,
            "max_j3_configs": 1,
        },
    )()

    specs = mod._specs_from_args(args)

    assert sum(spec.arm.startswith("J3_") for spec in specs) == 1
