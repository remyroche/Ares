from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.support_aware_pairwise_objective import (
    SupportAwarePairwiseConfig,
    SupportAwarePairwiseObjectiveError,
    build_support_aware_pairwise_objective,
)


def _frame() -> pd.DataFrame:
    # Each query contains deliberately mixed support/base-order cases.  The
    # labels are resolved training labels, not candidate inference fields.
    return pd.DataFrame(
        {
            "query_id": ["q1"] * 5 + ["q2"] * 4,
            "atr_residual": [-1.2, -0.4, 0.2, 1.0, 1.8, -1.1, -0.2, 0.8, 1.7],
            "candidate_residual_bps": [-150, -55, 35, 115, 245, -130, -20, 95, 230],
            "atr_residual_grade": [0, 1, 2, 3, 4, 0, 2, 3, 4],
            "support_h12": [False, True, False, True, True, False, True, False, True],
            # q1's largest label is deliberately under-ranked by the incumbent.
            "prequential_base_expected_net_bps": [80, 60, 40, 20, 0, 20, 10, 0, -10],
        }
    )


def _config(**updates: object) -> SupportAwarePairwiseConfig:
    defaults: dict[str, object] = {
        "max_pairs_per_query": 16,
        "exhaustive_pair_limit": 100,
        "loose_atr_separation": 0.20,
        "loose_bps_separation": 20.0,
        "strict_atr_separation": 0.75,
        "strict_bps_separation": 75.0,
        "random_state": 19,
    }
    defaults.update(updates)
    return SupportAwarePairwiseConfig(**defaults)


def test_pair_ledger_is_deterministic_and_query_normalised() -> None:
    first = build_support_aware_pairwise_objective(_frame(), config=_config())
    second = build_support_aware_pairwise_objective(_frame(), config=_config())

    np.testing.assert_array_equal(first.winner_rows, second.winner_rows)
    np.testing.assert_array_equal(first.loser_rows, second.loser_rows)
    np.testing.assert_allclose(first.pair_weights, second.pair_weights)
    assert first.pair_count == first.audit.selected_pairs
    assert first.audit.selected_pairs_by_query == second.audit.selected_pairs_by_query
    for _, group in first.pair_frame().groupby("query_code", sort=False):
        assert group["pair_weight"].sum() == pytest.approx(1.0)
        assert len(group) <= first.config.max_pairs_per_query


def test_pairs_have_agreeing_residual_direction_and_declared_separation() -> None:
    frame = _frame()
    objective = build_support_aware_pairwise_objective(frame, config=_config())
    pairs = objective.pair_frame()
    atr = frame["atr_residual"].to_numpy()
    bps = frame["candidate_residual_bps"].to_numpy()
    grade = frame["atr_residual_grade"].to_numpy()

    for pair in pairs.itertuples(index=False):
        winner, loser = int(pair.winner_row), int(pair.loser_row)
        assert atr[winner] > atr[loser]
        assert bps[winner] > bps[loser]
        assert atr[winner] - atr[loser] >= objective.config.loose_atr_separation
        assert bps[winner] - bps[loser] >= objective.config.loose_bps_separation
        if pair.is_strict:
            assert atr[winner] - atr[loser] >= objective.config.strict_atr_separation
            assert bps[winner] - bps[loser] >= objective.config.strict_bps_separation
            assert abs(grade[winner] - grade[loser]) >= objective.config.strict_grade_separation


def test_support_and_incumbent_misorder_are_recorded_without_changing_pair_labels() -> None:
    objective = build_support_aware_pairwise_objective(_frame(), config=_config())
    pairs = objective.pair_frame()
    assert set(pairs["support_class"]).issubset(
        {"both", "winner_only", "loser_only", "neither"}
    )
    assert pairs["incumbent_misordered"].any()

    only_misordered = build_support_aware_pairwise_objective(
        _frame(), config=_config(require_incumbent_misorder=True)
    )
    assert only_misordered.pair_count > 0
    assert only_misordered.incumbent_misordered.all()
    # Labels still decide the pair direction; the incumbent is only a pair
    # selection/weighting control and never reverses winner/loser semantics.
    assert np.all(
        _frame().loc[only_misordered.winner_rows, "atr_residual"].to_numpy()
        > _frame().loc[only_misordered.loser_rows, "atr_residual"].to_numpy()
    )


def test_custom_logistic_gradient_promotes_winner_and_penalises_loser() -> None:
    objective = build_support_aware_pairwise_objective(
        _frame().iloc[:2].copy(),
        config=_config(loose_atr_separation=0.1, loose_bps_separation=1.0),
    )
    assert objective.pair_count == 1
    gradient, hessian = objective(np.zeros(objective.row_count), train_data=None)
    winner, loser = int(objective.winner_rows[0]), int(objective.loser_rows[0])
    assert gradient[winner] < 0.0
    assert gradient[loser] > 0.0
    assert hessian[winner] > objective.config.min_hessian
    assert hessian[loser] > objective.config.min_hessian


def test_invalid_rows_are_excluded_from_pair_construction() -> None:
    frame = _frame()
    frame.loc[0, "candidate_residual_bps"] = np.nan
    objective = build_support_aware_pairwise_objective(frame, config=_config())
    assert objective.audit.invalid_rows == 1
    assert 0 not in objective.winner_rows
    assert 0 not in objective.loser_rows


def test_missing_training_columns_fail_loudly() -> None:
    with pytest.raises(SupportAwarePairwiseObjectiveError, match="missing required training column"):
        build_support_aware_pairwise_objective(_frame().drop(columns="support_h12"))


def test_lightgbm_train_accepts_the_frozen_closure_when_available() -> None:
    lgb = pytest.importorskip("lightgbm")
    frame = _frame()
    objective = build_support_aware_pairwise_objective(frame, config=_config())
    dataset = lgb.Dataset(
        np.arange(len(frame) * 2, dtype=np.float32).reshape(len(frame), 2),
        # LightGBM requires a label even though the custom objective does not
        # consume it after construction.
        label=np.zeros(len(frame), dtype=np.float32),
        free_raw_data=False,
    )
    model = lgb.train(
        {
            "objective": objective.lightgbm_objective(),
            "metric": "None",
            "verbosity": -1,
            "num_leaves": 3,
            "learning_rate": 0.1,
            "min_data_in_leaf": 1,
            "min_data_in_bin": 1,
            "feature_pre_filter": False,
        },
        dataset,
        num_boost_round=3,
    )
    prediction = model.predict(np.zeros((len(frame), 2), dtype=np.float32))
    assert len(prediction) == len(frame)
