from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_historical_execution_ev_opportunity_payoff_trust_ablation import (
    exit_mixture_from_components,
    fit_hierarchical_ev_calibration,
    planned_fit_count,
    stable_top_k_mask,
    strict_chronological_folds,
    validate_canonical_exit_labels,
    validate_feature_columns,
)


def test_strict_folds_use_only_labels_resolved_before_validation_start() -> None:
    decision = pd.date_range("2025-03-01", periods=10, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "__ts__": decision,
            "execution_label_end_utc": decision + pd.Timedelta(hours=12),
        }
    )
    # This otherwise historical row resolves exactly at the first validation
    # boundary and must therefore not enter that fold's training set.
    frame.loc[1, "execution_label_end_utc"] = pd.Timestamp(
        "2025-03-05", tz="UTC"
    )
    windows = [
        (
            pd.Timestamp("2025-03-05", tz="UTC"),
            pd.Timestamp("2025-03-07", tz="UTC"),
        ),
        (
            pd.Timestamp("2025-03-07", tz="UTC"),
            pd.Timestamp("2025-03-09", tz="UTC"),
        ),
    ]

    folds = strict_chronological_folds(
        frame,
        windows,
        decision_col="__ts__",
        resolution_col="execution_label_end_utc",
    )

    assert len(folds) == 2
    for fold, (validation_start, validation_end) in zip(folds, windows):
        train = frame.iloc[np.asarray(fold["train_positions"], dtype=int)]
        validation = frame.iloc[
            np.asarray(fold["validation_positions"], dtype=int)
        ]
        assert (train["__ts__"] < validation_start).all()
        assert (train["execution_label_end_utc"] < validation_start).all()
        assert validation["__ts__"].ge(validation_start).all()
        assert validation["__ts__"].lt(validation_end).all()
    assert 1 not in set(folds[0]["train_positions"])


def test_stable_top_k_breaks_score_ties_by_candidate_id() -> None:
    score = np.array([0.1, 0.9, 0.9, 0.2])
    candidate_id = np.array(["z", "b", "a", "x"], dtype=object)

    selected = stable_top_k_mask(score, candidate_id, k=2)

    assert np.asarray(selected, dtype=bool).tolist() == [
        False,
        True,
        True,
        False,
    ]


def test_exit_mixture_preserves_signed_conditional_payoffs() -> None:
    mixture = exit_mixture_from_components(
        probabilities=np.array(
            [
                [0.50, 0.25, 0.25, 0.00],
                [0.00, 0.00, 0.00, 1.00],
            ]
        ),
        conditional_payoffs=np.array(
            [
                [0.10, -0.20, 0.04, -0.30],
                [0.10, 0.02, -0.10, -0.40],
            ]
        ),
    )

    np.testing.assert_allclose(mixture, [0.01, -0.40])


def test_canonical_exit_labels_require_exactly_one_matching_class_flag() -> None:
    frame = pd.DataFrame(
        {
            "execution_exit_class": [
                "trailing",
                "timeout",
                "full_stop",
                "adverse_exit",
            ],
            "exit_is_trailing": [True, False, False, False],
            "exit_is_timeout": [False, True, False, False],
            "exit_is_full_stop": [False, False, True, False],
            "exit_is_adverse_exit": [False, False, False, True],
        }
    )
    validate_canonical_exit_labels(frame)
    invalid = frame.copy()
    invalid.loc[0, "exit_is_timeout"] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_canonical_exit_labels(invalid)


def test_hierarchical_calibration_ignores_evaluation_outcomes() -> None:
    reference = pd.DataFrame(
        {
            "side_name": ["long", "short"] * 6,
            "score": np.linspace(-2.0, 2.0, 12),
            "execution_net_ev_12h": np.linspace(-0.02, 0.02, 12),
        }
    )
    evaluation = pd.DataFrame(
        {
            "side_name": ["long", "short", "long", "short"],
            "score": [-1.5, -0.5, 0.5, 1.5],
        }
    )
    poisoned = evaluation.assign(
        execution_net_ev_12h=[1_000_000.0, -1_000_000.0] * 2
    )

    first, _ = fit_hierarchical_ev_calibration(
        reference,
        evaluation,
        score_column="score",
        min_rows=2,
        side_shrinkage=2.0,
    )
    second, _ = fit_hierarchical_ev_calibration(
        reference,
        poisoned,
        score_column="score",
        min_rows=2,
        side_shrinkage=2.0,
    )

    np.testing.assert_allclose(
        np.asarray(first, dtype=float),
        np.asarray(second, dtype=float),
    )
    assert np.isfinite(np.asarray(first, dtype=float)).all()


@pytest.mark.parametrize(
    "forbidden",
    [
        "execution_net_ev_12h",
        "execution_gross_ev_12h",
        "execution_exit_class",
        "__future_slope_atr_per_hour_12h__",
    ],
)
def test_forbidden_outcome_and_post_entry_features_are_rejected(
    forbidden: str,
) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_feature_columns(["base_oof_score", forbidden])


def test_fit_count_math_matches_bounded_two_fold_two_side_contract() -> None:
    assert planned_fit_count(n_folds=2, n_sides=2) == 100
