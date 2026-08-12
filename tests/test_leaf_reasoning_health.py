from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.performance_regimes.leaf_reasoning_health import (
    LeafReasoningHealthColumns,
    LeafReasoningHealthConfig,
    analyze_leaf_reasoning_health,
)


def _rows(
    *,
    leaf: str,
    fold: int,
    activation: float,
    prediction: list[float],
    label: list[float],
    net_bps: list[float],
    side: str = "long",
) -> pd.DataFrame:
    count = len(prediction)
    return pd.DataFrame(
        {
            "candidate_id": [f"{leaf}-{fold}-{index}" for index in range(count)],
            "__ts__": pd.date_range("2024-01-01", periods=count, freq="12h", tz="UTC"),
            "head": "residual",
            "side_name": side,
            "fold": fold,
            "leaf_token": leaf,
            "activation": activation,
            "prediction": prediction,
            "label": label,
            "net_bps": net_bps,
            "is_strict_oof": True,
        }
    )


def _config() -> LeafReasoningHealthConfig:
    return LeafReasoningHealthConfig(
        minimum_rows=4,
        minimum_active_rows=4,
        minimum_active_periods=3,
        minimum_active_months=1,
        minimum_score_rows=4,
        minimum_economic_rows=4,
        maximum_normalized_calibration_bias=0.25,
    )


def test_fold_local_leaf_health_reports_support_calibration_economics_and_months() -> None:
    healthy = _rows(
        leaf="tree_7_leaf_2", fold=2, activation=.8,
        prediction=[.1, .3, .6, .9], label=[.1, .3, .6, .9], net_bps=[10., 20., 30., 40.],
    )
    # The same raw token belongs to a different tree fit.  It must remain a
    # distinct audit row rather than being treated as a recurring leaf.
    adverse_other_fold = _rows(
        leaf="tree_7_leaf_2", fold=3, activation=.8,
        prediction=[.1, .3, .6, .9], label=[.1, .3, .6, .9], net_bps=[-10., -20., -30., -40.],
    )
    low_activation = _rows(
        leaf="tree_1_leaf_1", fold=2, activation=.2,
        prediction=[.1, .3, .6, .9], label=[.1, .3, .6, .9], net_bps=[20., 20., 20., 20.],
    )
    result = analyze_leaf_reasoning_health(
        pd.concat([healthy, adverse_other_fold, low_activation], ignore_index=True),
        columns=LeafReasoningHealthColumns(strict_oof="is_strict_oof"),
        config=_config(),
    )

    health = result.leaf_health.set_index(["fold", "leaf_token"])
    first = health.loc[(2, "tree_7_leaf_2")]
    assert first["within_fold_health"] == "HEALTHY"
    assert first["row_support"] == 4
    assert first["active_rows"] == 4
    assert first["active_period_support"] == 4
    assert first["active_month_support"] == 1
    assert first["active_prediction_mean"] == pytest.approx(.475)
    assert first["active_economic_mean"] == pytest.approx(25.0)
    assert first["calibration_mae"] == pytest.approx(0.0)
    assert first["prediction_label_pearson"] == pytest.approx(1.0)
    assert health.loc[(3, "tree_7_leaf_2"), "within_fold_health"] == "ECONOMICALLY_ADVERSE"
    assert health.loc[(2, "tree_1_leaf_1"), "within_fold_health"] == "LOW_ACTIVATION_OR_ACTIVE_SUPPORT"
    assert set(result.period_health["fold"]) == {2, 3}
    assert set(result.month_health["month"]) == {"2024-01"}


def test_health_classifies_inversion_and_calibration_without_fitting() -> None:
    inverted = _rows(
        leaf="inverted", fold=2, activation=.9,
        prediction=[.9, .6, .3, .1], label=[.1, .3, .6, .9], net_bps=[5., 10., 15., 20.],
    )
    biased = _rows(
        leaf="biased", fold=2, activation=.9,
        prediction=[.6, .8, 1.0, 1.2], label=[.1, .3, .6, .9], net_bps=[5., 10., 15., 20.],
    )
    health = analyze_leaf_reasoning_health(
        pd.concat([inverted, biased], ignore_index=True), config=_config()
    ).leaf_health.set_index("leaf_token")
    assert health.loc["inverted", "within_fold_health"] == "PREDICTION_INVERTED"
    assert health.loc["biased", "within_fold_health"] == "CALIBRATION_BIAS"
    assert health.loc["biased", "calibration_signed_error"] > 0.0


def test_chunked_and_single_frame_paths_have_identical_fold_local_statistics() -> None:
    frame = pd.concat(
        [
            _rows(
                leaf="stable", fold=2, activation=.8,
                prediction=[.1, .3, .6, .9], label=[.1, .3, .6, .9], net_bps=[10., 20., 30., 40.],
            ),
            _rows(
                leaf="stable", fold=3, activation=.8,
                prediction=[.1, .3, .6, .9], label=[.1, .3, .6, .9], net_bps=[10., 20., 30., 40.],
            ),
        ],
        ignore_index=True,
    )
    expected = analyze_leaf_reasoning_health(frame, config=_config()).leaf_health.sort_values(["fold", "leaf_token"]).reset_index(drop=True)
    observed = analyze_leaf_reasoning_health([frame.iloc[:4].copy(), frame.iloc[4:].copy()], config=_config()).leaf_health.sort_values(["fold", "leaf_token"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, observed)


def test_strict_oof_marker_and_duplicate_leaf_assignment_are_fail_closed() -> None:
    frame = _rows(
        leaf="leaf", fold=2, activation=.8,
        prediction=[.1, .3, .6, .9], label=[.1, .3, .6, .9], net_bps=[10., 20., 30., 40.],
    )
    bad_oof = frame.copy()
    bad_oof.loc[0, "is_strict_oof"] = False
    with pytest.raises(ValueError, match="strict-OOF"):
        analyze_leaf_reasoning_health(
            bad_oof, columns=LeafReasoningHealthColumns(strict_oof="is_strict_oof"), config=_config()
        )
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate candidate"):
        analyze_leaf_reasoning_health(duplicated, config=_config())
