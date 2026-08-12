from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from extreme_price_movements.market_spine_cluster_covariance import (
    MarketSpineClusterCovarianceConfig,
    _blockwise_consensus_coassociation,
    aggregate_hourly_market_spine,
    fit_market_spine_cluster_model,
    transform_market_spine_cluster_covariance,
)


def test_hourly_market_spine_uses_robust_candidate_aggregates() -> None:
    timestamp = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    candidates = pd.DataFrame(
        {
            "timestamp": [timestamp, timestamp + pd.Timedelta(minutes=10), timestamp + pd.Timedelta(minutes=40)],
            "move": [-2.0, 0.0, 10.0],
        }
    )

    spine = aggregate_hourly_market_spine(candidates, ["move"], timestamp_col="timestamp")

    row = spine.iloc[0]
    assert row["mspine__move__median"] == 0.0
    assert row["mspine__move__p10"] == -1.6
    assert row["mspine__move__p90"] == 8.0
    assert row["mspine__move__iqr"] == 6.0
    assert row["mspine__move__dispersion"] == 6.0 / 1.349
    assert row["mspine__move__breadth"] == 1.0 / 3.0


def test_frozen_training_clusters_and_transform_are_future_safe() -> None:
    spine = _market_spine()
    config = MarketSpineClusterCovarianceConfig(
        level_window_hours=96,
        final_normalization_window_hours=96,
        innovation_ewma_span_hours=12,
        cluster_block_hours=60 * 24,
        min_block_observations=72,
        cluster_count=2,
        min_covariance_observations=12,
    )
    training_end = spine.index[1500]
    model = fit_market_spine_cluster_model(spine, training_end, config)

    changed_future = spine.copy()
    changed_future.loc[changed_future.index > training_end, "mspine__alpha__median"] *= -100.0
    changed_model = fit_market_spine_cluster_model(changed_future, training_end, config)

    assert model.memberships == changed_model.memberships
    assert model.orientations == changed_model.orientations
    assert model.weights == changed_model.weights

    baseline = transform_market_spine_cluster_covariance(spine, model)
    future_changed = spine.copy()
    causal_cutoff = spine.index[1200]
    future_changed.loc[future_changed.index > causal_cutoff, "mspine__beta__median"] += 1_000.0
    perturbed = transform_market_spine_cluster_covariance(future_changed, model)

    assert_frame_equal(
        baseline.features.loc[:causal_cutoff],
        perturbed.features.loc[:causal_cutoff],
        check_exact=True,
    )
    assert all(name.startswith("mspine_factor__cluster_") for name in baseline.factors.columns)


def test_covariance_panel_contains_global_and_factor_breaks() -> None:
    spine = _market_spine()
    config = MarketSpineClusterCovarianceConfig(
        level_window_hours=96,
        final_normalization_window_hours=96,
        innovation_ewma_span_hours=12,
        cluster_block_hours=60 * 24,
        min_block_observations=72,
        cluster_count=2,
        min_covariance_observations=12,
    )
    model = fit_market_spine_cluster_model(spine, spine.index[1500], config)
    result = transform_market_spine_cluster_covariance(spine, model)

    expected = {
        "mspine_cov__global__scale_ratio__12h_vs_720h",
        "mspine_cov__global__corr_frobenius__24h_vs_720h",
        "mspine_cov__global__coherence_drop__72h_vs_720h",
        "mspine_cov__global__pc1_evr_drop__12h_vs_720h",
        "mspine_cov__global__loading_angle__24h_vs_720h",
        "mspine_cov__global__effective_rank_drop__72h_vs_720h",
    }
    assert expected.issubset(result.raw_features.columns)
    assert any("__global_corr_break__" in col for col in result.raw_features.columns)
    assert any("__internal_corr_frobenius__" in col for col in result.raw_features.columns)
    assert any("__internal_loading_angle__" in col for col in result.raw_features.columns)
    assert result.features.iloc[-1].notna().any()


def test_consensus_uses_block_cluster_coassociation_not_average_correlation() -> None:
    """A pair that clusters in one block only does not pass a 75% consensus gate."""

    n_block = 60 * 24
    rng = np.random.default_rng(7)
    first = np.arange(n_block, dtype=float)
    second = np.arange(n_block, dtype=float)
    innovations = pd.DataFrame(
        {
            "a": np.r_[first, second],
            "b": np.r_[first + rng.normal(0.0, 0.01, n_block), rng.normal(0.0, 1.0, n_block)],
            "c": np.r_[rng.normal(0.0, 1.0, n_block), second + rng.normal(0.0, 0.01, n_block)],
            "d": rng.normal(0.0, 1.0, 2 * n_block),
        }
    )
    config = MarketSpineClusterCovarianceConfig(
        cluster_count=2,
        consensus_coassociation_threshold=0.75,
        min_block_observations=72,
    )

    coassociation = _blockwise_consensus_coassociation(innovations, config)

    assert coassociation[0, 1] == 0.5
    assert coassociation[0, 2] == 0.5


def _market_spine(n: int = 1_900) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    time = np.arange(n, dtype=float)
    base_a = np.sin(time / 17.0) + 0.2 * np.sin(time / 5.0)
    base_b = np.cos(time / 23.0)
    return pd.DataFrame(
        {
            "mspine__alpha__median": base_a + 0.03 * np.sin(time / 3.0),
            "mspine__beta__median": base_a * 0.9 + 0.04 * np.cos(time / 4.0),
            "mspine__gamma__median": -base_b + 0.03 * np.sin(time / 7.0),
            "mspine__delta__median": base_b + 0.04 * np.cos(time / 6.0),
        },
        index=index,
    )
