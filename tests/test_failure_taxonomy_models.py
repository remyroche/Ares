from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.unsupervised_regime_learning.failure_taxonomy_models import (
    DiagonalStudentTMixture,
    FailureTaxonomyModelConfig,
    _fit_representation_seed_stability,
    _learned_representation_embeddings,
    failure_taxonomy_nonredundancy,
    failure_taxonomy_temporal_stability,
    fit_failure_taxonomy_models,
    fit_frozen_consensus_taxonomy,
)


def test_student_t_mixture_separates_heavy_tailed_groups() -> None:
    rng = np.random.default_rng(17)
    matrix = np.vstack(
        [
            rng.standard_t(4, size=(40, 3)) - 3.0,
            rng.standard_t(4, size=(40, 3)) + 3.0,
        ]
    )
    labels = DiagonalStudentTMixture(2, random_state=17).fit(matrix).predict(matrix)
    assert max(np.mean(labels[:40] == 0), np.mean(labels[:40] == 1)) > 0.85
    assert adjusted_group_separation(labels) > 0.80


def adjusted_group_separation(labels: np.ndarray) -> float:
    first = int(np.bincount(labels[:40]).argmax())
    second = int(np.bincount(labels[40:]).argmax())
    return float((np.mean(labels[:40] == first) + np.mean(labels[40:] == second)) / 2.0)


def test_failure_taxonomy_selects_a_stable_local_model() -> None:
    rng = np.random.default_rng(29)
    frame = pd.DataFrame(
        {
            "side_name": ["long"] * 30,
            "archetype_policy_key": ["trend"] * 30,
            "event_block": [f"event_{idx:03d}" for idx in range(30)],
            "event_start": pd.date_range("2025-01-01", periods=30, freq="7D", tz="UTC"),
            "event_end": pd.date_range("2025-01-01", periods=30, freq="7D", tz="UTC"),
            "family__liquidation__active_mean_z": np.r_[
                rng.normal(-2, 0.2, 15), rng.normal(2, 0.2, 15)
            ],
            "family__recovery__onset_abs_delta": np.r_[
                rng.normal(0.2, 0.1, 15), rng.normal(2.0, 0.1, 15)
            ],
        }
    )
    assignments, diagnostics = fit_failure_taxonomy_models(
        frame,
        config=FailureTaxonomyModelConfig(
            latent_dims=(2,),
            cluster_counts=(2, 3),
            methods=("pca_student_t", "pca_gmm"),
            stability_seeds=(17, 29),
        ),
    )
    assert len(assignments) == len(frame)
    assert assignments["cluster_id"].nunique() >= 2
    assert diagnostics["seed_ari"].max() > 0.7


def test_failure_taxonomy_emits_multiview_pairing_and_stability_diagnostics() -> None:
    rng = np.random.default_rng(71)
    count = 48
    error_state = np.repeat([0, 1], count // 2)
    market_state = np.tile(np.repeat([0, 1], count // 4), 2)
    frame = pd.DataFrame(
        {
            "side_name": ["short"] * count,
            "archetype_policy_key": ["breakout"] * count,
            "event_block": [f"event_{idx:03d}" for idx in range(count)],
            "event_start": pd.date_range(
                "2025-01-01", periods=count, freq="7D", tz="UTC"
            ),
            "event_end": pd.date_range(
                "2025-01-01", periods=count, freq="7D", tz="UTC"
            ),
            # Explicit error shape columns must never be mixed into state PCA.
            "family__error__signed_residual_mean": error_state * 5.0
            + rng.normal(0, 0.10, count),
            "family__error__bad_mae_rate": error_state * 0.8
            + rng.normal(0, 0.02, count),
            "expost__peak_signed_residual": error_state * 4.0
            + rng.normal(0, 0.10, count),
            # Observable market-state columns form a different partition.
            "family__market__oi_flush": market_state * 4.0 + rng.normal(0, 0.10, count),
            "family__market__breadth_recovery": market_state * -3.0
            + rng.normal(0, 0.10, count),
        }
    )
    assignments, diagnostics = fit_failure_taxonomy_models(
        frame,
        config=FailureTaxonomyModelConfig(
            latent_dims=(2,),
            cluster_counts=(2,),
            methods=("pca_gmm",),
            stability_seeds=(17, 29),
            episode_bootstrap_repeats=2,
        ),
    )
    required_assignment_columns = {
        "error_cluster_id",
        "error_cluster_posterior_max",
        "market_state_cluster_id",
        "market_state_cluster_posterior_max",
        "consensus_pair_id",
        "consensus_pair_posterior",
        "winner_selection_objective",
    }
    assert required_assignment_columns.issubset(assignments.columns)
    assert assignments["error_cluster_id"].nunique() == 2
    assert assignments["market_state_cluster_id"].nunique() == 2
    winner = diagnostics.loc[diagnostics["is_winner"]].iloc[0]
    assert winner["error_view_feature_count"] == 3
    assert winner["market_state_view_feature_count"] == 2
    assert winner["error_seed_ari"] > 0.90
    assert winner["market_state_seed_ari"] > 0.90
    assert -1.0 <= winner["error_episode_bootstrap_ari"] <= 1.0
    assert -1.0 <= winner["market_state_episode_bootstrap_ari"] <= 1.0
    assert 0.0 <= winner["error_seed_posterior_js"] <= 1.0
    assert 0.0 <= winner["market_state_seed_posterior_js"] <= 1.0
    assert winner["consensus_pair_count"] >= 2


def test_failure_taxonomy_keeps_single_view_behavior_without_error_columns() -> None:
    rng = np.random.default_rng(73)
    frame = pd.DataFrame(
        {
            "side_name": ["long"] * 24,
            "archetype_policy_key": ["continuation"] * 24,
            "event_block": [f"event_{idx:03d}" for idx in range(24)],
            "event_start": pd.date_range("2025-01-01", periods=24, freq="7D", tz="UTC"),
            "event_end": pd.date_range("2025-01-01", periods=24, freq="7D", tz="UTC"),
            "family__market__trend": np.r_[
                rng.normal(-2, 0.1, 12), rng.normal(2, 0.1, 12)
            ],
            "family__market__volatility": np.r_[
                rng.normal(-1, 0.1, 12), rng.normal(1, 0.1, 12)
            ],
        }
    )
    assignments, diagnostics = fit_failure_taxonomy_models(
        frame,
        config=FailureTaxonomyModelConfig(
            latent_dims=(2,),
            cluster_counts=(2,),
            methods=("pca_gmm",),
            stability_seeds=(17, 29),
        ),
    )
    assert len(assignments) == len(frame)
    assert "error_cluster_id" not in assignments
    assert (
        diagnostics.loc[diagnostics["is_winner"], "error_view_feature_count"]
        .eq(0)
        .all()
    )
    assert diagnostics.loc[diagnostics["is_winner"], "is_winner"].all()


def test_learned_episode_representations_refit_across_stability_seeds() -> None:
    rng = np.random.default_rng(101)
    matrix = np.vstack(
        [
            rng.normal(-1.5, 0.2, size=(16, 12)),
            rng.normal(1.5, 0.2, size=(16, 12)),
        ]
    ).astype(np.float32)
    config = FailureTaxonomyModelConfig(
        latent_dims=(2, 8),
        cluster_counts=(2,),
        stability_seeds=(17, 29),
        learned_encoder_epochs=2,
        episode_bootstrap_repeats=0,
    )
    for method, latent_dim in (("small_dae_gmm", 8), ("vicreg_gmm", 2)):
        embeddings = _learned_representation_embeddings(
            matrix,
            method=method,
            latent_dim=latent_dim,
            config=config,
        )
        assert set(embeddings) == {17, 29}
        assert all(len(value) == len(matrix) for value in embeddings.values())
        fitted = _fit_representation_seed_stability(
            embeddings,
            method=method,
            clusters=2,
            config=config,
        )
        assert fitted is not None
        labels, probabilities, seed_ari, posterior_js = fitted
        assert len(labels) == len(matrix)
        assert probabilities.shape == (len(matrix), 2)
        assert -1.0 <= seed_ari <= 1.0
        assert 0.0 <= posterior_js <= 1.0


def test_nonredundancy_flags_single_month_or_severity_modes() -> None:
    starts = pd.to_datetime(
        [
            "2025-01-01",
            "2025-01-05",
            "2025-01-09",
            "2025-01-13",
            "2025-02-01",
            "2025-03-01",
        ],
        utc=True,
    )
    episodes = pd.DataFrame(
        {
            "side_name": "long",
            "archetype_policy_key": "trend",
            "event_start": starts,
            "event_end": starts,
            "calendar_mean_ev": [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0],
        }
    )
    assignments = episodes.reset_index().rename(columns={"index": "source_index"})[
        [
            "source_index",
            "side_name",
            "archetype_policy_key",
            "event_start",
            "event_end",
        ]
    ]
    assignments["method"] = "pca_gmm"
    assignments["latent_dim"] = 2
    assignments["clusters"] = 2
    assignments["cluster_id"] = [0, 0, 0, 0, 1, 1]

    report = failure_taxonomy_nonredundancy(episodes, assignments)

    concentrated = report.loc[report["cluster_id"].eq(0)].iloc[0]
    assert concentrated["active_months"] == 1
    assert concentrated["month_max_fraction"] == 1.0
    assert concentrated["calendar_redundancy_warning"]


def test_temporal_stability_reports_recurring_monthly_modes() -> None:
    starts = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-16",
            "2025-02-02",
            "2025-02-16",
            "2025-03-02",
            "2025-03-16",
            "2025-04-02",
            "2025-04-16",
        ],
        utc=True,
    )
    episodes = pd.DataFrame(
        {
            "side_name": "short",
            "archetype_policy_key": "breakout",
            "event_start": starts,
            "event_end": starts,
            "calendar_mean_ev": [-0.02, -0.01] * 4,
            "family__error_shape__active__signed_residual": [-1.0, 1.0] * 4,
        }
    )
    assignments = episodes.reset_index().rename(columns={"index": "source_index"})[
        [
            "source_index",
            "side_name",
            "archetype_policy_key",
            "event_start",
            "event_end",
        ]
    ]
    assignments["method"] = "pca_gmm"
    assignments["latent_dim"] = 2
    assignments["clusters"] = 2
    assignments["cluster_id"] = [0, 1] * 4

    report = failure_taxonomy_temporal_stability(episodes, assignments)

    assert len(report) == 2
    assert report["active_months"].eq(4).all()
    assert report["month_presence_rate"].eq(1.0).all()
    assert report["monthly_support_std"].eq(0.0).all()
    assert not report["temporal_stability_warning"].any()


def test_frozen_consensus_taxonomy_ignores_post_reference_changes() -> None:
    rng = np.random.default_rng(303)
    count = 48
    starts = pd.date_range("2024-01-01", periods=count, freq="14D", tz="UTC")
    frame = pd.DataFrame(
        {
            "side_name": "short",
            "archetype_policy_key": "breakout",
            "event_block": [f"event_{idx:03d}" for idx in range(count)],
            "event_start": starts,
            "event_end": starts,
            "family__error_vector__active__signed_residual": np.r_[
                rng.normal(-2.0, 0.1, count // 2),
                rng.normal(2.0, 0.1, count // 2),
            ],
            "family__market__oi_flush": np.tile([-2.0, 2.0], count // 2)
            + rng.normal(0.0, 0.1, count),
        }
    )
    cutoff = starts[30]
    config = FailureTaxonomyModelConfig(
        latent_dims=(2,),
        cluster_counts=(2,),
        methods=("pca_gmm",),
        stability_seeds=(17, 29),
        episode_bootstrap_repeats=1,
        min_cluster_episodes=3,
    )

    first, diagnostics, state = fit_frozen_consensus_taxonomy(
        frame, reference_end=cutoff, config=config
    )
    changed = frame.copy()
    changed.loc[changed["event_end"].ge(cutoff), "family__market__oi_flush"] += 100.0
    second, _, changed_state = fit_frozen_consensus_taxonomy(
        changed, reference_end=cutoff, config=config
    )

    assert len(first) == len(frame)
    assert not diagnostics.empty
    assert state["reference_end"] == cutoff.isoformat()
    assert state["reference_label_availability_horizon_days"] == 15
    assert state["groups"].keys() == changed_state["groups"].keys()
    key = next(iter(state["groups"]))
    assert state["groups"][key]["median"] == changed_state["groups"][key]["median"]
    assert state["groups"][key]["centroids"] == changed_state["groups"][key]["centroids"]
    reference_mask = first["event_end"].lt(cutoff)
    expected_reference = first["event_end"].add(pd.Timedelta(days=15)).lt(cutoff)
    assert first["assignment_is_reference"].equals(expected_reference)
    pd.testing.assert_series_equal(
        first.loc[reference_mask, "cluster_id"].reset_index(drop=True),
        second.loc[reference_mask, "cluster_id"].reset_index(drop=True),
    )
