from __future__ import annotations

import json

import numpy as np
from sklearn.mixture import GaussianMixture

from extreme_price_movements.representation_proxy_metrics import (
    DEFAULT_GMM_PANEL,
    GmmPanelSpec,
    align_diagonal_gmm_components,
    apply_ood_calibration,
    bounded_diagonal_gmm_overlap,
    diagonal_gmm_statistics,
    entropy_distribution_diagnostics,
    evaluate_ood_proxy,
    evaluate_representation_proxies,
    fit_common_gmm_panel,
    fit_ood_calibration,
    heldout_nll_degradation,
    occupancy_excess_instability,
    perturbation_consistency,
    refine_diagonal_gmm,
    refinement_promotion_diagnostics,
    reorder_posteriors_to_reference,
)


def _latent(seed: int = 4, n_rows: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack(
        (
            rng.normal(loc=(-2.0, 0.0), scale=0.35, size=(n_rows // 2, 2)),
            rng.normal(loc=(2.0, 0.5), scale=0.45, size=(n_rows - n_rows // 2, 2)),
        )
    ).astype(np.float32)


def _model(z: np.ndarray) -> GaussianMixture:
    return GaussianMixture(
        n_components=2,
        covariance_type="diag",
        reg_covar=0.003,
        random_state=9,
        n_init=1,
    ).fit(z)


def test_cached_diagonal_statistics_match_sklearn_and_float_dtypes() -> None:
    z = _latent()
    model = _model(z)

    stats32 = diagonal_gmm_statistics(z, model, dtype=np.float32)
    stats64 = diagonal_gmm_statistics(z, model, dtype=np.float64)
    stats_chunked = diagonal_gmm_statistics(z, model, dtype=np.float64, batch_rows=17)

    np.testing.assert_allclose(stats32["posteriors"], model.predict_proba(z), atol=2e-6)
    np.testing.assert_allclose(stats64["posteriors"], stats_chunked["posteriors"], atol=1e-12)
    np.testing.assert_allclose(stats64["ood_score"], stats_chunked["ood_score"], atol=1e-12)
    assert stats32["posteriors"].dtype == np.float32
    assert stats64["posteriors"].dtype == np.float64
    assert np.all(np.isfinite(stats32["ood_score"]))


def test_component_alignment_reorders_permuted_posteriors() -> None:
    z = _latent()
    model = _model(z)
    original = {"means": model.means_, "covariances": model.covariances_, "weights": model.weights_}
    permuted = {
        "means": model.means_[::-1],
        "covariances": model.covariances_[::-1],
        "weights": model.weights_[::-1],
    }
    alignment = align_diagonal_gmm_components(original, permuted)
    candidate_posteriors = model.predict_proba(z)[:, ::-1]

    restored = reorder_posteriors_to_reference(candidate_posteriors, alignment, dtype=np.float64)

    assert alignment.mean_cost < 1e-10
    np.testing.assert_allclose(restored, model.predict_proba(z), atol=1e-10)


def test_entropy_and_occupancy_cover_symbol_calendar_and_regime_strata() -> None:
    base = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype=np.float32)
    shifted = np.asarray([[0.7, 0.3], [0.7, 0.3], [0.4, 0.6], [0.3, 0.7]], dtype=np.float32)
    entropy = entropy_distribution_diagnostics(base, bins=4)
    rows = occupancy_excess_instability(
        {"seed_1": base, "resample_2": shifted},
        expected_weights=np.asarray([0.5, 0.5]),
        strata={"symbol": ["A", "A", "B", "B"], "calendar": ["weekday", "weekend", "weekday", "weekend"], "regime": [0, 0, 1, 1]},
    )

    assert entropy["n_components"] == 2
    assert sum(entropy["histogram_counts"]) == 4
    assert {(row["stratum"], row["level"]) for row in rows} >= {
        ("overall", "all"),
        ("symbol", "A"),
        ("calendar", "weekday"),
        ("regime", "1"),
    }
    assert max(row["max_component_occupancy_std"] for row in rows) > 0.0


def test_perturbation_and_three_tier_ood_are_consistent_for_identical_cache() -> None:
    z = _latent()
    model = _model(z)
    stats = diagonal_gmm_statistics(z, model, dtype=np.float64)
    consistency = perturbation_consistency(
        z,
        z.copy(),
        stats["posteriors"],
        stats["posteriors"].copy(),
        stats["ood_score"],
        stats["ood_score"].copy(),
    )
    calibration = fit_ood_calibration(stats["ood_score"])
    applied = apply_ood_calibration(stats["ood_score"], calibration)

    assert consistency["latent_cosine_mean"] == 1.0
    assert consistency["posterior_tv_mean"] == 0.0
    assert consistency["ood_rank_correlation"] == 1.0
    assert len(applied["tiers"]) == len(z)
    assert applied["tier_names"] == ["in_distribution", "elevated", "extreme"]


def test_standard_panel_and_lambda_zero_refinement_are_serialization_friendly() -> None:
    z = _latent(n_rows=120)
    model = _model(z)
    fits = fit_common_gmm_panel(z, seeds=(2,), max_iter=30)
    untouched = refine_diagonal_gmm(z, model, overlap_lambda=0.0)

    assert [(fit.spec.n_components, fit.spec.covariance_type, fit.spec.reg_covar) for fit in fits] == list(DEFAULT_GMM_PANEL)
    assert untouched["refined"] is False
    np.testing.assert_allclose(untouched["state"]["means"], model.means_)
    np.testing.assert_allclose(untouched["state"]["covariances"], model.covariances_)
    np.testing.assert_allclose(untouched["state"]["weights"], model.weights_)
    json.dumps(untouched)


def test_common_panel_retries_higher_regularization_for_collapsed_latent() -> None:
    latent = np.zeros((32, 2), dtype=np.float32)
    latent[:2, 0] = np.asarray([-1.0, 1.0], dtype=np.float32)
    failures: list[dict[str, object]] = []
    fits = fit_common_gmm_panel(
        latent,
        specs=(GmmPanelSpec(4, "diag", 0.0),),
        retry_reg_covars=(0.01,),
        failure_records=failures,
    )
    assert len(fits) == 1
    assert fits[0].spec.reg_covar == 0.01
    assert failures[0]["status"] == "recovered_with_higher_reg_covar"


def test_proxy_result_is_outcome_free_and_json_serializable() -> None:
    z = _latent()
    model = _model(z)
    stats = diagonal_gmm_statistics(z, model)
    result = evaluate_representation_proxies(
        stats["posteriors"],
        posterior_runs={"seed_1": stats["posteriors"], "perturbation": stats["posteriors"]},
        expected_weights=model.weights_,
        strata={"symbol": np.where(np.arange(len(z)) % 2, "B", "A")},
        latent_reference=z,
        latent_perturbed=z,
        posterior_perturbed=stats["posteriors"],
        ood_reference=stats["ood_score"],
        ood_perturbed=stats["ood_score"],
    )

    payload = result.to_dict()
    assert payload["manifest"]["outcome_free"] is True
    assert payload["perturbation"]["hard_assignment_agreement"] == 1.0
    json.dumps(payload)


def test_ood_proxy_reports_severity_rank_separation_and_untouched_fpr() -> None:
    clean = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    result = evaluate_ood_proxy(
        clean,
        clean + 0.5,
        clean + 2.0,
        clean + 3.0,
        calibration_scores=np.asarray([0.7, 0.8, 0.9, 1.0, 1.1]),
        aligned_clean_scores={
            "mild_synthetic": clean,
            "structural_synthetic": clean,
            "natural_temporal": None,
        },
    )

    assert result["outcome_free"] is True
    assert result["monotonicity"]["mean_non_decreasing"] is True
    assert result["rank_consistency"]["mild_synthetic"]["rank_correlation"] == 1.0
    assert result["rank_consistency"]["natural_temporal"]["aligned"] is False
    assert result["clean_corrupted_separation"]["structural_synthetic"]["probability_corrupted_gt_clean"] == 1.0
    assert result["untouched_later_false_positive_rate"]["elevated_or_extreme"] == 0.0
    json.dumps(result)


def test_refinement_promotion_reports_bounded_overlap_and_heldout_nll() -> None:
    overlapping = {
        "means": [[-0.1, 0.0], [0.1, 0.0]],
        "covariances": [[1.0, 1.0], [1.0, 1.0]],
        "weights": [0.5, 0.5],
    }
    separated = {
        "means": [[-2.0, 0.0], [2.0, 0.0]],
        "covariances": [[0.5, 0.5], [0.5, 0.5]],
        "weights": [0.5, 0.5],
    }
    heldout = _latent(n_rows=120)
    before = bounded_diagonal_gmm_overlap(overlapping)
    after = bounded_diagonal_gmm_overlap(separated)
    nll = heldout_nll_degradation(heldout, overlapping, separated, max_degradation=0.0)
    promotion = refinement_promotion_diagnostics(
        heldout,
        overlapping,
        separated,
        max_heldout_nll_degradation=0.0,
    )

    assert 0.0 <= after["bounded_overlap"] < before["bounded_overlap"] <= 1.0
    assert nll["mean_nll_degradation"] < 0.0
    assert nll["passes_max_degradation"] is True
    assert promotion["overlap_not_increased"] is True
    assert promotion["promotion_eligible"] is True
    json.dumps(promotion)
