from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_adjacent_july_state_adapter_ablation import (
    BASE_SCORE,
    DECISION,
    RESOLUTION,
    SIDE,
    TARGET,
    _fit_adapter,
    evaluate_predictions,
    specialist_eligibility,
)


def test_adapter_has_exact_zero_fallback_without_prior_rows() -> None:
    evaluation = pd.DataFrame(
        {
            SIDE: ["long", "short"],
            "feature": [1.0, 2.0],
        }
    )
    delta, report = _fit_adapter(
        evaluation.iloc[0:0],
        evaluation,
        ["feature"],
        min_rows=10,
        iterations=5,
        seed=1,
        n_jobs=1,
    )
    np.testing.assert_allclose(delta, 0.0)
    assert report["long"]["status"] == "zero_fallback"
    assert report["short"]["status"] == "zero_fallback"


def test_specialist_requires_two_stable_recurring_blocks() -> None:
    rows = []
    for block, means in (
        ("july_01_05", {0: -0.004, 1: 0.004}),
        ("july_06_12", {0: -0.003, 1: 0.005}),
    ):
        for side in ("long", "short"):
            for state, residual in means.items():
                for index in range(110):
                    rows.append(
                        {
                            SIDE: side,
                            "july_block": block,
                            "causal_regime_state": state,
                            TARGET: residual,
                            BASE_SCORE: 0.0,
                            "candidate_id": f"{block}-{side}-{state}-{index}",
                        }
                    )
    result = specialist_eligibility(pd.DataFrame(rows))
    assert result["eligible"] is True
    assert result["decision"] == "test_state_specialist"


def test_specialist_rejects_unstable_state_mapping() -> None:
    rows = []
    for block, means in (
        ("july_01_05", {0: -0.004, 1: 0.004}),
        ("july_06_12", {0: 0.004, 1: -0.004}),
    ):
        for side in ("long", "short"):
            for state, residual in means.items():
                for index in range(110):
                    rows.append(
                        {
                            SIDE: side,
                            "july_block": block,
                            "causal_regime_state": state,
                            TARGET: residual,
                            BASE_SCORE: 0.0,
                            "candidate_id": f"{block}-{side}-{state}-{index}",
                        }
                    )
    result = specialist_eligibility(pd.DataFrame(rows))
    assert result["eligible"] is False
    assert result["decision"].startswith("zero_fallback")


def test_pooled_global_selection_is_not_a_timestamp_quota() -> None:
    timestamp = pd.Timestamp("2026-07-01T00:00:00Z")
    frame = pd.DataFrame(
        {
            "arm": ["baseline"] * 20,
            "july_block": ["july_01_05"] * 10 + ["july_13_19"] * 10,
            SIDE: ["long", "short"] * 10,
            DECISION: [timestamp] * 20,
            RESOLUTION: [timestamp + pd.Timedelta(hours=12)] * 20,
            TARGET: np.linspace(-0.01, 0.01, 20),
            "score": np.arange(20, dtype=float),
        }
    )
    weekly, pooled = evaluate_predictions(frame, top_k_fraction=0.10)
    assert weekly["selected_rows"].sum() == 2
    all_row = pooled.loc[
        pooled["segment"].eq("all_july") & pooled["scope"].eq("pooled")
    ].iloc[0]
    assert all_row["selected_rows"] == 2
    latest = pooled.loc[
        pooled["segment"].eq("latest_july_block")
        & pooled["scope"].eq("pooled")
    ].iloc[0]
    assert latest["selected_rows"] == 2
