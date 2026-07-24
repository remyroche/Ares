from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.diagnostics.distribution_drift import (
    benjamini_hochberg_qvalues,
    categorical_distribution_shift,
    feature_distribution_drift,
    may_june_july_feature_drift,
    nearest_neighbor_losing_trade_diagnostic,
    numeric_distribution_shift,
)


def test_numeric_shift_reports_reference_decile_psi_and_distribution_metrics() -> None:
    reference = np.linspace(0.0, 9.0, 100)
    comparison = reference + 4.0

    result = numeric_distribution_shift(reference, comparison)

    assert result["ks_statistic"] > 0.4
    assert result["ks_pvalue"] < 1e-6
    assert result["wasserstein"] == pytest.approx(4.0)
    assert result["wasserstein_normalized"] == pytest.approx(4.0 / 4.5)
    assert result["mean_shift"] == pytest.approx(4.0)
    assert result["variance_shift"] == pytest.approx(0.0)
    assert result["quantile_p50_shift"] == pytest.approx(4.0)
    assert result["psi_bin_count"] == 10
    assert result["psi"] > 1.0
    assert numeric_distribution_shift(reference, reference)["psi"] == pytest.approx(0.0)


def test_categorical_metrics_and_bh_qvalues_are_correct() -> None:
    categorical = categorical_distribution_shift(
        ["a"] * 7 + ["b"] * 3,
        ["a"] * 3 + ["b"] * 7,
    )
    qvalues = benjamini_hochberg_qvalues([0.01, 0.04, 0.03, np.nan])

    assert categorical["total_variation"] == pytest.approx(0.4)
    assert 0.0 < categorical["jensen_shannon_base2"] < 1.0
    np.testing.assert_allclose(qvalues[:3], [0.03, 0.04, 0.04])
    assert np.isnan(qvalues[3])


def test_feature_distribution_drift_applies_bh_only_to_numeric_ks_tests() -> None:
    reference = pd.DataFrame(
        {"x": np.arange(20, dtype=float), "y": np.arange(20, dtype=float), "cat": ["a"] * 20}
    )
    comparison = pd.DataFrame(
        {"x": np.arange(20, dtype=float) + 10.0, "y": np.arange(20, dtype=float), "cat": ["b"] * 20}
    )

    report = feature_distribution_drift(
        reference,
        comparison,
        numeric_features=["x", "y"],
        categorical_features=["cat"],
    ).set_index("feature")

    assert report.loc["x", "ks_qvalue"] < 0.05
    assert report.loc["y", "ks_qvalue"] == pytest.approx(1.0)
    assert np.isnan(report.loc["cat", "ks_qvalue"])
    assert report.loc["cat", "total_variation"] == pytest.approx(1.0)


def test_may_june_july_drift_includes_all_pairs_and_worst_day_repeat() -> None:
    rows = []
    for month, values, categories, pnl in (
        (5, [0.0, 1.0, 2.0, 3.0], ["a", "a", "b", "b"], [-1.0, -1.0, 1.0, 1.0]),
        (6, [3.0, 4.0, 5.0, 6.0], ["b", "b", "b", "a"], [-2.0, -2.0, 1.0, 1.0]),
        (7, [6.0, 7.0, 8.0, 9.0], ["c", "c", "b", "b"], [-3.0, -3.0, 1.0, 1.0]),
    ):
        for index, (value, category, row_pnl) in enumerate(zip(values, categories, pnl)):
            rows.append(
                {
                    "timestamp": pd.Timestamp(year=2026, month=month, day=1 + index // 2, tz="UTC"),
                    "feature": value,
                    "category": category,
                    "pnl": row_pnl,
                }
            )
    report = may_june_july_feature_drift(
        pd.DataFrame(rows),
        timestamp_column="timestamp",
        numeric_features=["feature"],
        categorical_features=["category"],
        include_worst_day=True,
        outcome_column="pnl",
    )

    pairs = set(zip(report["reference_month"], report["comparison_month"]))
    assert pairs == {("2026-05", "2026-06"), ("2026-06", "2026-07"), ("2026-05", "2026-07")}
    assert set(report["scope"]) == {"all_rows", "worst_day_only"}
    worst_numeric = report.loc[
        (report["feature"] == "feature") & (report["scope"] == "worst_day_only")
    ]
    assert worst_numeric["n_reference"].eq(2).all()
    assert worst_numeric["n_comparison"].eq(2).all()


def test_nearest_neighbor_loss_diagnostic_uses_reference_scaling_and_exclusions() -> None:
    diagnostic = nearest_neighbor_losing_trade_diagnostic(
        comparison_features=np.array([[0.0, 0.0], [1.0, 1.0]]),
        reference_features=np.array(
            [[0.0, 0.0], [0.05, 0.0], [0.2, 0.0], [1.0, 1.0]]
        ),
        reference_is_loss=[False, True, True, False],
        comparison_is_loss=[True, False],
        reference_month=["July", "July", "July", "June"],
        comparison_month=["July", "July"],
        reference_episode=["episode-a", "episode-a", "episode-a", "other"],
        comparison_episode=["episode-a", "other"],
        reference_timestamps=pd.to_datetime(
            ["2026-07-10 00:00", "2026-07-10 00:30", "2026-07-10 02:00", "2026-07-12 00:00"],
            utc=True,
        ),
        comparison_timestamps=pd.to_datetime(["2026-07-10 00:00", "2026-07-12 00:00"], utc=True),
        reference_ids=["self", "near", "kept-loss", "kept-win"],
        comparison_ids=["self", "other"],
        k=2,
        near_time_window="1h",
    )

    assert diagnostic.summary["n_losing_queries"] == 1
    assert diagnostic.summary["k_requested"] == 2
    row = diagnostic.neighbors.iloc[0]
    assert row["comparison_row"] == 0
    assert row["neighbor_count"] == 2
    assert row["neighbor_loss_fraction"] == pytest.approx(0.5)
    assert row["neighbor_same_month_fraction"] == pytest.approx(0.5)
    assert row["neighbor_same_episode_fraction"] == pytest.approx(0.5)
    assert row["min_neighbor_distance"] > 0.0


def test_nearest_neighbor_diagnostic_rejects_k_above_twenty() -> None:
    with pytest.raises(ValueError, match="between 1 and 20"):
        nearest_neighbor_losing_trade_diagnostic(
            comparison_features=np.array([[0.0]]),
            reference_features=np.array([[0.0]]),
            reference_is_loss=[True],
            k=21,
        )
