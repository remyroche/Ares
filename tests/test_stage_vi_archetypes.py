from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_vi_archetypes import (
    CAUSAL_PREFIX,
    PATH_PREFIX,
    ArchetypeConfig,
    ArchetypeView,
    ArchetypeWeightConfig,
    ArchetypeDecisionConfig,
    CompactMultiViewState,
    archetype_alignment_switching,
    align_archetype_catalogues,
    archetype_economic_separation,
    archetype_membership_validation,
    archetype_fold_stability,
    archetype_sample_weights,
    fit_side_local_archetypes,
    materialize_archetype_decision_matrix,
    materialize_multiview_composite_objective,
    remove_current_archetype_columns,
    run_matched_incremental_archetype_comparison,
    stage_vi_ablation_grid,
    stage_vi_path_ablation_grid,
    strict_oof_archetype_features,
)


def _frame(rows: int = 720) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    decision = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    side = np.where(np.arange(rows) % 2, "long", "short")
    mode = (np.arange(rows) // 24) % 3
    setup = (mode * 2.0 + rng.normal(0, 0.15, rows)).astype(np.float32)
    regime = (np.where(side == "long", 0.5, -0.5) + mode + rng.normal(0, 0.2, rows)).astype(np.float32)
    trust = (setup + rng.normal(0, 0.3, rows)).astype(np.float32)
    net = (45.0 + mode * 40.0 + rng.normal(0, 4.0, rows)).astype(np.float32)
    # Path coordinates are clearly different but retain a soft boundary.
    mfe = (mode * 1.2 + rng.normal(0, 0.12, rows)).astype(np.float32)
    mae = (-mode * 0.7 + rng.normal(0, 0.12, rows)).astype(np.float32)
    return pd.DataFrame(
        {
            "decision_ts": decision,
            "label_available_ts": decision + pd.Timedelta(hours=13),
            "side_name": side,
            "symbol": np.where(np.arange(rows) % 3, "BTC", "ETH"),
            "exact_net_bps": net,
            "setup": setup,
            "regime": regime,
            "trust": trust,
            "path_mfe": mfe,
            "path_mae": mae,
            "path_certainty": np.where(mode == 0, 0.6, 1.0).astype(np.float32),
            "economy_bucket": pd.Series(mode).astype(str),
            "event_upper": (mode == 2).astype(np.float32),
        }
    )


def _causal_config() -> ArchetypeConfig:
    return ArchetypeConfig(
        view=ArchetypeView("CF4", ("setup", "regime", "trust"), "causal"),
        components=3,
        min_side_rows=60,
        min_component_rows=10,
        random_state=4,
    )


def _path_config() -> ArchetypeConfig:
    return ArchetypeConfig(
        view=ArchetypeView("PF4", ("path_mfe", "path_mae"), "path"),
        components=3,
        min_side_rows=60,
        min_component_rows=10,
        random_state=5,
    )


def test_causal_archetypes_are_side_local_soft_and_inference_safe() -> None:
    frame = _frame()
    state = fit_side_local_archetypes(frame.iloc[:500], config=_causal_config())
    safe = frame.iloc[500:].loc[:, ["side_name", "setup", "regime", "trust"]]
    out = state.transform(safe)
    probabilities = out[[f"{CAUSAL_PREFIX}prob__{i}" for i in range(3)]]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert out[f"{CAUSAL_PREFIX}available"].eq(1.0).all()
    assert state.manifest()["side_local_construction"] is True
    assert state.manifest()["local_trading_experts"] is False


def test_path_archetypes_are_diagnostic_until_causal_recogniser() -> None:
    frame = _frame()
    state = fit_side_local_archetypes(
        frame.iloc[:500], config=_path_config(), causal_recogniser_columns=("setup", "regime", "trust"),
    )
    with pytest.raises(ValueError, match="realised path"):
        state.transform(frame.iloc[500:])
    safe = frame.iloc[500:].loc[:, ["side_name", "setup", "regime", "trust"]]
    out = state.transform(safe)
    probs = out[[f"{PATH_PREFIX}prob__{i}" for i in range(3)]]
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert state.manifest()["path_memberships_diagnostic_until_strict_oof_recogniser"] is True


def test_strict_oof_path_recogniser_uses_prior_resolved_positive_rows() -> None:
    frame = _frame()
    result = strict_oof_archetype_features(
        frame,
        config=_path_config(),
        causal_recogniser_columns=("setup", "regime", "trust"),
        folds=3,
    )
    scored = result.fold_audit.loc[result.fold_audit.status.eq("scored")]
    assert not scored.empty
    assert (pd.to_datetime(scored.train_max_label_available_ts, utc=True) < pd.to_datetime(scored.valid_start, utc=True)).all()
    assert result.manifest["strict_oof"] is True
    assert result.manifest["hard_routing"] is False
    assert result.features[f"{PATH_PREFIX}available"].eq(1.0).any()
    assert result.features[f"{PATH_PREFIX}prob__unknown"].gt(0.5).any()
    assert result.diagnostic_truth_memberships.notna().any().any()
    assert scored["membership_log_loss"].notna().all()
    assert {
        "mean_calibration_intercept", "mean_calibration_slope", "mean_calibration_ece", "mean_membership_correlation",
        "mean_top_decile_enrichment", "economic_confusion_cost_bps",
    }.issubset(scored.columns)


def test_causal_view_refuses_outcome_fields() -> None:
    frame = _frame()
    bad = ArchetypeConfig(
        view=ArchetypeView("bad", ("setup", "path_mfe"), "causal"), components=3,
        min_side_rows=60, min_component_rows=10,
    )
    with pytest.raises(ValueError, match="outcome/path"):
        fit_side_local_archetypes(frame, config=bad)


def test_weight_contracts_and_grid_are_bounded_and_explicit() -> None:
    frame = _frame()
    cfg = ArchetypeConfig(
        view=ArchetypeView("PF0", ("path_mfe", "path_mae"), "path"), components=3,
        min_side_rows=60, min_component_rows=10,
        weights=ArchetypeWeightConfig(mode="economic_diversity", economic_bucket_col="economy_bucket"),
    )
    weights = archetype_sample_weights(frame, cfg)
    assert np.isfinite(weights).all()
    assert ((weights >= 0.25) & (weights <= 4.0)).all()
    grid = stage_vi_ablation_grid(
        [ArchetypeView("CF1", ("setup", "regime"), "causal")],
        methods=("kmeans",), components=(3, 4), weight_modes=("uniform",),
        min_side_rows=60, min_component_rows=10,
    )
    assert [item.components for item in grid] == [3, 4]


def test_catalogue_alignment_and_economic_separation_are_evaluation_only() -> None:
    reference = pd.DataFrame({"side": ["long", "long"], "rank": [0, 1], "centroid": [[0.0, 0.0], [2.0, 2.0]]})
    candidate = pd.DataFrame({"side": ["long", "long"], "rank": [7, 8], "centroid": [[2.1, 1.9], [0.1, 0.0]]})
    aligned = align_archetype_catalogues(reference, candidate)
    assert set(aligned.candidate_rank) == {7, 8}
    frame = _frame(90)
    memberships = np.zeros((len(frame), 3), dtype=np.float32)
    memberships[np.arange(len(frame)), np.arange(len(frame)) % 3] = 1.0
    report = archetype_economic_separation(
        frame, memberships, outcome_columns=("exact_net_bps", "path_mfe", "event_upper"),
    )
    assert {"exact_net_bps__mean", "path_mfe__q90", "transition_rate", "top_symbol_share"}.issubset(report.columns)
    assert "month" in report
    assert not report.empty


def test_compact_multiview_and_fold_stability_are_train_only_and_aligned() -> None:
    frame = _frame(120)
    state = CompactMultiViewState({"setup": ("setup", "trust"), "regime": ("regime",)}, dimensions_per_view=2).fit(frame.iloc[:80])
    embedded = state.transform(frame.iloc[80:])
    assert embedded.shape == (40, 3)
    assert state.manifest()["nonlinear_embedding"] is False
    catalogue = pd.DataFrame(
        {
            "fold": [0, 0, 1, 1], "side": ["long"] * 4, "rank": [0, 1, 0, 1],
            "centroid": [[0.0, 0.0], [2.0, 2.0], [0.1, 0.0], [2.1, 2.0]],
        }
    )
    stability = archetype_fold_stability(catalogue)
    assert stability.loc[0, "matched_components"] == 2
    assert stability.loc[0, "mean_centroid_distance"] < 0.2


def test_current_archetype_outputs_can_be_cleanly_excluded() -> None:
    fields = ["setup", "gmm_posterior_0", "meta_conversion_arch_prob__0", "dae_density", "trust"]
    assert remove_current_archetype_columns(fields) == ["setup", "trust"]
    frame = _frame()
    legacy = ArchetypeConfig(
        view=ArchetypeView("legacy", ("setup", "gmm_posterior_0"), "causal"), components=3,
        min_side_rows=60, min_component_rows=10,
    )
    frame["gmm_posterior_0"] = 0.5
    with pytest.raises(ValueError, match="legacy AE/GMM"):
        fit_side_local_archetypes(frame, config=legacy)


def test_path_grid_includes_k8_aw4_aw5_and_bounded_pca_ae_gmm() -> None:
    grid = stage_vi_path_ablation_grid(
        [ArchetypeView("PF4", ("path_mfe", "path_mae"), "path")],
        methods=("gmm_pca_diag", "ae_gmm_diag"), components=(3, 8),
        path_certainty_col="path_certainty", economic_bucket_col="economy_bucket",
        min_side_rows=160, min_component_rows=10,
    )
    assert {item.components for item in grid} == {3, 8}
    assert {item.weights.mode for item in grid} == {
        "uniform", "time_balanced", "symbol_balanced", "path_certainty", "economic_diversity",
    }
    frame = _frame(360)
    for mode in ("path_certainty", "economic_diversity"):
        weighted = next(item for item in grid if item.weights.mode == mode and item.components == 3)
        values = archetype_sample_weights(frame, weighted)
        assert np.isfinite(values).all() and (values > 0).all()
    config = ArchetypeConfig(
        view=ArchetypeView("PF4", ("path_mfe", "path_mae"), "path"),
        method="ae_gmm_diag", components=3, min_side_rows=60, min_component_rows=10,
        embedding_dimensions=2, ae_hidden_units=4, ae_max_iter=50, random_state=17,
    )
    first = fit_side_local_archetypes(frame.iloc[:280], config=config, causal_recogniser_columns=("setup", "regime", "trust"))
    second = fit_side_local_archetypes(frame.iloc[:280], config=config, causal_recogniser_columns=("setup", "regime", "trust"))
    safe = frame.iloc[280:].loc[:, ["side_name", "setup", "regime", "trust"]]
    np.testing.assert_allclose(first.transform(safe), second.transform(safe), atol=1e-6)
    assert first.manifest()["discovery_embedding"] == "small_ae"
    assert first.manifest()["ae_bounded_deterministic"] is True
    pca = ArchetypeConfig(
        view=config.view, method="gmm_pca_full", components=3, min_side_rows=60, min_component_rows=10,
        embedding_dimensions=2, random_state=17,
    )
    assert fit_side_local_archetypes(frame.iloc[:280], config=pca, causal_recogniser_columns=("setup", "regime", "trust")).manifest()["discovery_embedding"] == "pca"


def test_path_recogniser_validation_reports_required_forecast_metrics() -> None:
    truth = np.array([[.9, .1], [.8, .2], [.1, .9], [.2, .8]], dtype=float)
    predicted = np.array([[.8, .2], [.7, .3], [.2, .8], [.1, .9]], dtype=float)
    report = archetype_membership_validation(
        predicted, truth, prior_cluster_economic_bps=(75.0, -125.0),
        realised_net_bps=np.array([60.0, 50.0, -100.0, -120.0]),
    )
    assert {
        "membership_log_loss", "membership_brier", "membership_rps", "mean_calibration_slope", "mean_calibration_ece",
        "mean_membership_correlation", "mean_top_decile_enrichment", "economic_confusion_cost_bps",
    }.issubset(report.summary.columns)
    assert {"calibration_slope", "membership_correlation", "top_decile_enrichment"}.issubset(report.per_membership.columns)
    assert report.summary.loc[0, "economic_confusion_cost_bps"] > 0


def test_semantic_alignment_switching_and_matched_incremental_decisions() -> None:
    reference = pd.DataFrame({"side": ["long", "long"], "rank": [0, 1], "centroid": [[0.0], [2.0]]})
    candidate = pd.DataFrame({"side": ["long", "long"], "rank": [7, 8], "centroid": [[0.1], [2.1]]})
    ref_econ = pd.DataFrame({"side": ["long", "long"], "rank": [0, 1], "net": [0.0, 100.0]})
    cand_econ = pd.DataFrame({"side": ["long", "long"], "rank": [7, 8], "net": [100.0, 0.0]})
    switching = archetype_alignment_switching(reference, candidate, ref_econ, cand_econ, economic_columns=("net",))
    assert switching.semantic_alignment_switch.all()
    stability_catalog = pd.concat([
        reference.assign(fold=0), candidate.rename(columns={"rank": "rank"}).assign(fold=1),
    ], ignore_index=True)
    stability_economics = pd.concat([
        ref_econ.assign(fold=0), cand_econ.assign(fold=1),
    ], ignore_index=True)
    stability = archetype_fold_stability(
        stability_catalog, economic_catalog=stability_economics, economic_columns=("net",),
    )
    assert stability.loc[0, "semantic_alignment_switch_rate"] == 1.0
    ledger = pd.DataFrame({
        "candidate_id": range(10), "symbol": ["BTC"] * 10,
        "decision_ts": pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC"),
        "side_name": ["long"] * 10, "net": [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35],
        "control": np.arange(10), "base": np.arange(10), "meta": np.arange(10),
        "both": np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8]),
    })
    matched = run_matched_incremental_archetype_comparison(
        ledger, arm_score_columns={"control": "control", "base": "base", "meta": "meta", "both": "both"},
        net_bps_col="net", top_fractions=(.1, .2),
    )
    assert set(matched.arm) == {"control", "base", "meta", "both"}
    assert matched.global_ranking.all() and matched.matched_candidate_population.all()
    evidence = pd.DataFrame({
        "path_separation": [.8, .8], "economic_separation": [.7, .01],
        "causal_predictability": [.5, .5], "temporal_stability": [.9, .9], "concentration": [.2, .2],
        "base_incremental_bps": [1.0, 1.0], "meta_incremental_bps": [0.0, 0.0],
        "hard_label_value": [0.0, 0.0], "soft_membership_value": [1.0, 1.0],
    })
    objective = materialize_multiview_composite_objective(evidence)
    decision = materialize_archetype_decision_matrix(objective, config=ArchetypeDecisionConfig())
    assert decision.loc[0, "disposition"] == "Retained base context"
    assert decision.loc[1, "disposition"] == "Reject"
