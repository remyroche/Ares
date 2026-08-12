from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.base_population_drift import (
    BasePopulationDriftError,
    feature_distribution_drift,
    held_out_adversarial_separability,
    population_composition,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"),
        "signal": np.linspace(0.0, 10.0, 100),
        "constant": 1.0,
        "missing": [1.0] * 90 + [np.nan] * 10,
    })
    current = pd.DataFrame({
        "ts": pd.date_range("2024-02-01", periods=100, freq="h", tz="UTC"),
        "signal": np.linspace(4.0, 14.0, 100),
        "constant": 1.0,
        "missing": [1.0] * 60 + [np.nan] * 40,
    })
    return reference, current


def test_feature_distribution_is_train_referenced_and_reports_missing_constant_and_support() -> None:
    reference, current = _frames()
    report = feature_distribution_drift(
        reference, current, feature_names=["signal", "constant", "missing"], timestamp_column="ts"
    ).set_index("feature")
    assert report.loc["signal", "psi"] > 0.1
    assert report.loc["signal", "wasserstein"] == pytest.approx(4.0)
    assert report.loc["signal", "upper_extrapolation_rate"] > 0.3
    assert report.loc["constant", "is_reference_constant"]
    assert report.loc["constant", "psi"] == pytest.approx(0.0)
    assert report.loc["missing", "coverage_delta"] == pytest.approx(-0.3)


def test_feature_distribution_rejects_contemporaneous_or_future_reference() -> None:
    reference, current = _frames()
    reference.loc[0, "ts"] = current.ts.iloc[0]
    with pytest.raises(BasePopulationDriftError, match="strictly precede"):
        feature_distribution_drift(reference, current, feature_names=["signal"], timestamp_column="ts")


def test_population_composition_reports_asset_concentration_validity_classes_and_causal_numeric_fields() -> None:
    frame = pd.DataFrame({
        "month": ["2024-01", "2024-01", "2024-01", "2024-02"],
        "side": ["long", "long", "short", "long"],
        "asset": ["BTC", "BTC", "ETH", "SOL"],
        "valid": [True, False, True, True],
        "r3": ["clear", "weak", "adverse", "clear"],
        "atr_cost": [1.2, 1.4, 0.8, 1.1],
    })
    result = population_composition(
        frame, month_column="month", side_column="side", asset_column="asset",
        label_valid_column="valid", class_column="r3", numeric_columns=["atr_cost"],
    )
    jan_long = result[(result.month == "2024-01") & (result.side == "long")].iloc[0]
    assert jan_long.candidate_rows == 2
    assert jan_long.active_assets == 1
    assert jan_long.asset_hhi == pytest.approx(1.0)
    assert jan_long.label_valid_rate == pytest.approx(0.5)
    assert jan_long["class_share__clear"] == pytest.approx(0.5)
    assert jan_long["atr_cost__median"] == pytest.approx(1.3)


def test_adversarial_separability_is_held_out_deterministic_and_outcome_free() -> None:
    reference, current = _frames()
    # An outcome-like field is intentionally absent from the function call;
    # changing it cannot affect an input-only population diagnostic.
    reference["outcome"] = 0.0
    current["outcome"] = 1.0
    first = held_out_adversarial_separability(
        reference, current, feature_names=["signal", "constant", "missing"], timestamp_column="ts", random_state=7
    )
    current["outcome"] = -999.0
    second = held_out_adversarial_separability(
        reference, current, feature_names=["signal", "constant", "missing"], timestamp_column="ts", random_state=7
    )
    # Held-out rather than in-sample AUC is deliberately conservative on this
    # small overlapping synthetic shift, but it should still be informative.
    assert first.held_out_auc > 0.7
    assert first.held_out_auc == pytest.approx(second.held_out_auc)
    assert first.held_out_rows == 50
    assert first.feature_contributions.iloc[0].feature == "signal"


def test_adversarial_separability_rejects_future_reference() -> None:
    reference, current = _frames()
    reference.loc[0, "ts"] = current.ts.iloc[0]
    with pytest.raises(BasePopulationDriftError, match="strictly precede"):
        held_out_adversarial_separability(reference, current, feature_names=["signal"], timestamp_column="ts")
