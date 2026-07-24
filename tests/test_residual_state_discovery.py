from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.global_residual_latent_state import (
    CausalCategoricalStateHMM,
    GlobalGMMStateModel,
    GlobalResidualSignatureEncoder,
    GMMGridConfig,
    HMMGridConfig,
    ResidualAEConfig,
    ResidualAwareAutoencoder,
    ResidualEncoderConfig,
    SideArchetypeStatePriors,
    StateVectorConfig,
    _archetype_surprise_persistence_signature,
    _normalized_binned_mutual_information,
    add_causal_phase_state_features,
    add_temporal_state_features,
    archetype_state_token,
    build_global_residual_signature,
    build_side_timestamp_states,
    latent_geometry_diagnostics,
    prepare_archetype_state_partition,
    select_partition_state_features,
)
from extreme_price_movements.residual_state_discovery import (
    ReliabilityEventConfig,
    audit_feature_concepts,
    benjamini_hochberg,
    discover_reliability_events,
    feature_quality_metrics,
    matched_control_feature_diagnostics,
)


def _reliability_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(52)
    for day in range(36):
        timestamp = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=day)
        for side in ("long", "short"):
            for archetype in ("mixed", "breakout"):
                for asset in range(12):
                    probability = 0.65
                    hit = float(rng.random() < probability)
                    ev = 0.004 if hit else -0.005
                    bad_mae = float(not hit)
                    timeout = 0.0
                    if side == "long" and archetype == "mixed" and day in (30, 31):
                        hit = 0.0
                        ev = -0.012
                        bad_mae = 1.0
                    if side == "short" and archetype == "breakout" and day == 32:
                        hit = 1.0
                        ev = -0.009
                    rows.append(
                        {
                            "__ts__": timestamp + pd.Timedelta(minutes=asset),
                            "__symbol__": f"S{asset}",
                            "side_name": side,
                            "archetype_policy_key": archetype,
                            "hit_probability": probability,
                            "clean_exec": hit,
                            "ev_after_1pct": ev,
                            "full_path_bad_mae_1r": bad_mae,
                            "timeout": timeout,
                        }
                    )
    return pd.DataFrame(rows)


def test_archetype_fast_persistence_uses_two_consecutive_days() -> None:
    selected = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-04"], utc=True
            ),
            "side_name": ["long"] * 3,
            "archetype_policy_key": ["mixed"] * 3,
            "_signature_signed_surprise": [0.5, 0.4, 0.9],
        }
    )
    result, _ = _archetype_surprise_persistence_signature(
        selected,
        timestamp_col="__ts__",
        side_col="side_name",
        archetype_col="archetype_policy_key",
    )
    column = "target_signature_arch__long_mixed_positive_persistence_2d"
    by_day = result.set_index("__signature_day__")[column]
    assert by_day.loc[pd.Timestamp("2026-01-02", tz="UTC")] == 0.2
    assert np.isnan(by_day.loc[pd.Timestamp("2026-01-04", tz="UTC")])


def test_archetype_primary_persistence_uses_shifted_prior_week() -> None:
    selected = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC"),
            "side_name": ["short"] * 8,
            "archetype_policy_key": ["continuation"] * 8,
            "_signature_signed_surprise": [0.5] * 7 + [0.8],
        }
    )
    result, _ = _archetype_surprise_persistence_signature(
        selected,
        timestamp_col="__ts__",
        side_col="side_name",
        archetype_col="archetype_policy_key",
    )
    column = "target_signature_arch__short_continuation_positive_persistence_prev7d"
    by_day = result.set_index("__signature_day__")[column]
    assert np.isclose(by_day.loc[pd.Timestamp("2026-01-08", tz="UTC")], 0.4)


def test_weekly_persistence_history_is_isolated_by_archetype() -> None:
    days = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    selected = pd.DataFrame(
        {
            "__ts__": list(days) + list(days),
            "side_name": ["long"] * 16,
            "archetype_policy_key": ["mixed"] * 8 + ["breakout"] * 8,
            "_signature_signed_surprise": [0.5] * 8 + [-0.5] * 8,
        }
    )
    result, _ = _archetype_surprise_persistence_signature(
        selected,
        timestamp_col="__ts__",
        side_col="side_name",
        archetype_col="archetype_policy_key",
    )
    last = result.set_index("__signature_day__").loc[days[-1]]
    assert np.isclose(
        last["target_signature_arch__long_mixed_positive_persistence_prev7d"],
        0.25,
    )
    assert np.isclose(
        last["target_signature_arch__long_breakout_negative_persistence_prev7d"],
        0.25,
    )


def test_benjamini_hochberg_is_monotone_in_p_order() -> None:
    p = pd.Series([0.04, 0.001, 0.02, np.nan, 0.2])
    result = benjamini_hochberg(p)
    valid = result.loc[p.notna()].assign(p=p[p.notna()]).sort_values("p")
    assert valid["bh_q"].is_monotonic_increasing
    assert bool(result.loc[1, "bh_reject"])


def test_event_catalogue_detects_persistent_loss_and_payoff_disagreement() -> None:
    result = discover_reliability_events(
        _reliability_rows(),
        ReliabilityEventConfig(causal_min_days=12, bootstrap_draws=40),
    )
    membership = result.event_membership
    assert not result.events.empty
    assert membership["evidence_type"].str.contains("persistent_loss").any()
    assert (
        membership["evidence_type"]
        .str.contains("calibration_economics_disagreement")
        .any()
    )
    assert result.events["discovery_eligible"].any()
    assert set(result.sensitivity["z_threshold"]) == {1.96, 2.33, 2.58}


def test_event_catalogue_keeps_same_day_archetype_failures_separate() -> None:
    rows = _reliability_rows()
    day = pd.Timestamp("2026-01-31", tz="UTC")
    mask = rows["__ts__"].dt.floor("D").eq(day)
    mask &= rows["side_name"].eq("long")
    mask &= rows["archetype_policy_key"].isin(["mixed", "breakout"])
    rows.loc[mask, "clean_exec"] = 0.0
    rows.loc[mask, "ev_after_1pct"] = -0.015
    rows.loc[mask, "full_path_bad_mae_1r"] = 1.0
    result = discover_reliability_events(
        rows,
        ReliabilityEventConfig(causal_min_days=12, bootstrap_draws=40),
    )
    membership = result.event_membership.copy()
    membership["day"] = pd.to_datetime(membership["day"], utc=True)
    local = membership.loc[
        membership["day"].eq(day)
        & membership["side_name"].eq("long")
        & membership["archetype_policy_key"].isin(["mixed", "breakout"])
    ]
    assert set(local["archetype_policy_key"]) == {"mixed", "breakout"}
    ids = local.groupby("archetype_policy_key", observed=True)["event_id"].first()
    assert ids.nunique() == 2


def test_phase_features_and_partition_relevance_are_train_local() -> None:
    timestamps = pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC")
    phase = np.linspace(0.0, 1.0, len(timestamps), dtype=np.float32)
    states = pd.DataFrame(
        {
            "__ts__": timestamps,
            "side_name": ["short"] * len(timestamps),
            "universe__median__mkt_median_oi_chg_4h_rz": -phase * 2.0,
            "universe__median__mkt_pct_oi_chg_4h_rz_lt_minus1": phase * 0.5,
            "universe__median__mkt_pct_price_down_oi_down_4h": phase * 0.4,
            "universe__median__market_breadth_chg_1h": -phase * 0.2,
            "universe__median__mkt_systemic_deleveraging_score": phase,
            "universe__median__mkt_flush_exhaustion_score": phase * 0.8,
            "universe__median__mkt_oi_flush_breadth_recovery_4h": phase * 0.15,
            "universe__median__market_breadth_recovery_from_6h_min": phase * 0.2,
            "universe__median__mkt_pct_price_up_oi_down_1h": phase * 0.3,
            "universe__median__asset_flush_exhaustion_score": phase * 0.7,
            "target_negative_surprise": phase,
            "target_negative_ev": phase,
            "target_positive_surprise": 1.0 - phase,
            "target_mean_ev": 1.0 - phase,
        }
    )
    enriched, manifest = add_causal_phase_state_features(states)
    assert set(manifest["features"]).issubset(enriched.columns)
    assert enriched["state_phase__liquidation_onset"].notna().all()
    selected, relevance = select_partition_state_features(
        enriched,
        [
            "universe__median__mkt_median_oi_chg_4h_rz",
            "state_phase__liquidation_onset",
            "state_phase__flush_exhaustion",
        ],
        max_features=2,
    )
    assert "state_phase__liquidation_onset" in selected
    assert not relevance.empty
    assert relevance["phase_feature"].any()


def test_feature_audit_distinguishes_exact_proxy_and_missing() -> None:
    audit = audit_feature_concepts(
        {
            "oi_drawdown_from_peak_24h",
            "mkt_pct_price_up_oi_down_1h",
        },
        configured_columns={"funding_sign_persistence_24h"},
    ).set_index("concept")
    assert audit.loc["oi_drawdown_from_peak_24h", "status"] == "present_exactly"
    assert audit.loc["price_up_oi_down_1h", "status"] == "present_as_close_proxy"
    assert audit.loc["funding_sign_persistence", "status"] == "unreliable_coverage"
    assert audit.loc["return_dispersion_change", "status"] == "missing"


def test_matched_control_feature_diagnostics_find_shift() -> None:
    rng = np.random.default_rng(7)
    n = 240
    event = np.zeros(n, dtype=bool)
    event[:80] = True
    frame = pd.DataFrame(
        {
            "is_event": event,
            "market_return_bucket": rng.integers(0, 4, n),
            "volatility": rng.normal(size=n),
            "candidate_signal": rng.normal(size=n) + event.astype(float) * 1.5,
            "noise": rng.normal(size=n),
        }
    )
    diagnostics, matched = matched_control_feature_diagnostics(
        frame,
        ["candidate_signal", "noise"],
        ["market_return_bucket", "volatility"],
    )
    by_feature = diagnostics.set_index("feature")
    assert not matched.empty
    assert by_feature.loc["candidate_signal", "univariate_event_auc"] > 0.70
    assert abs(by_feature.loc["candidate_signal", "standardized_mean_difference"]) > 0.8


def test_feature_quality_metrics_accepts_boolean_validity_flags() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-06-01", periods=120, freq="h", tz="UTC"),
            "__symbol__": ["S0"] * 120,
            "validity_flag": np.arange(120) % 3 != 0,
        }
    )
    quality = feature_quality_metrics(frame, ["validity_flag"])
    assert len(quality) == 1
    assert quality.loc[0, "q01"] == 0.0
    assert quality.loc[0, "q99"] == 1.0


def test_side_timestamp_state_is_shared_input_for_archetype_partition_models() -> None:
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    for hour in range(96):
        timestamp = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=hour)
        for side in ("long", "short"):
            for asset in range(8):
                shock = np.sin(hour / 9.0) + rng.normal(scale=0.1)
                selected = asset < 2
                rows.append(
                    {
                        "__ts__": timestamp,
                        "__symbol__": f"S{asset}",
                        "side_name": side,
                        "archetype_policy_key": "mixed" if asset % 2 else "breakout",
                        "selected_for_monitor": selected,
                        "score_meta_base_soft_label": 0.5 + 0.1 * shock,
                        "hit_probability": 0.6,
                        "clean_exec": float(shock > -0.5) if selected else np.nan,
                        "ev_after_1pct": (0.01 if shock > 0 else -0.008)
                        if selected
                        else np.nan,
                        "full_path_bad_mae_1r": float(shock < 0)
                        if selected
                        else np.nan,
                        "timeout": 0.0 if selected else np.nan,
                        "mkt_systemic_deleveraging_score": shock,
                        "asset_minus_mkt_oi_chg_4h_rz": shock + asset / 20,
                    }
                )
    candidates = pd.DataFrame(rows)
    states, features, manifest = build_side_timestamp_states(
        candidates,
        ["mkt_systemic_deleveraging_score"],
        ["asset_minus_mkt_oi_chg_4h_rz"],
        StateVectorConfig(min_feature_coverage=0.1),
    )
    assert len(states) == 96 * 2
    assert states.groupby(["__ts__", "side_name"]).size().max() == 1
    assert "selected_minus_universe__asset_minus_mkt_oi_chg_4h_rz" in features
    assert not any(name.startswith("target_") for name in features)
    assert manifest["schema"] == "global_side_timestamp_market_state_v1"
    assert archetype_state_token("long", "mixed") != archetype_state_token(
        "long", "breakout"
    )

    long = prepare_archetype_state_partition(
        states,
        side="long",
        archetype="mixed",
    ).reset_index(drop=True)
    ae = ResidualAwareAutoencoder(
        ResidualAEConfig(
            latent_dim=4, hidden_dim=12, epochs=8, patience=3, batch_size=64
        )
    ).fit(long, features)
    oos_without_outcomes = long.drop(
        columns=[name for name in long if name.startswith("target_")], errors="ignore"
    )
    latent = ae.transform(oos_without_outcomes)
    assert latent.filter(regex="^global_state_latent_").shape[1] == 4
    assert np.isfinite(latent["global_state_reconstruction_error"]).all()
    gmm = GlobalGMMStateModel(
        GMMGridConfig(
            components=(2, 3), covariance_types=("diag",), reg_covars=(1e-3,), n_init=1
        )
    ).fit(latent, long, long["__ts__"])
    state = gmm.transform(latent)
    temporal = add_temporal_state_features(state, long["__ts__"])
    assert state.filter(regex="^global_state_posterior_[0-9]+$").shape[1] in {2, 3}
    assert np.isfinite(state["global_state_posterior_max"]).all()
    assert (state["global_state_posterior_margin"] >= 0.0).all()
    assert (state["global_state_effective_components"] >= 1.0).all()
    assert "global_state_latent_speed" in temporal
    wrong_partition = long.copy()
    wrong_partition["archetype_policy_key"] = "breakout"
    with pytest.raises(ValueError, match="partition mismatch"):
        ae.transform(wrong_partition)


def test_global_signature_and_hybrid_encoder_are_outcome_free_at_transform() -> None:
    rows = _reliability_rows()
    rows["selected_for_monitor"] = (
        rows.groupby(["__ts__", "side_name"])["hit_probability"].rank(pct=True).ge(0.70)
    )
    rows["dirty_positive"] = (
        rows["ev_after_1pct"].gt(0.0) & rows["full_path_bad_mae_1r"].gt(0.5)
    ).astype(np.float32)
    rows["mkt_systemic_deleveraging_score"] = np.sin(np.arange(len(rows)) / 11.0)
    rows["asset_minus_mkt_oi_chg_4h_rz"] = np.cos(np.arange(len(rows)) / 7.0)
    rows["score_meta_base_soft_label"] = rows["hit_probability"].astype(np.float32)

    signature, manifest = build_global_residual_signature(rows)
    assert manifest["schema"] == "global_residual_signature_v1"
    assert any(name.startswith("target_signature_global_") for name in signature)
    assert any(name.startswith("target_signature_long_") for name in signature)
    assert any(name.endswith("negative_persistence_2d") for name in signature)
    assert any(name.endswith("negative_persistence_prev7d") for name in signature)

    states, features, state_manifest = build_side_timestamp_states(
        rows,
        ["mkt_systemic_deleveraging_score"],
        ["asset_minus_mkt_oi_chg_4h_rz"],
        StateVectorConfig(min_feature_coverage=0.1),
    )
    assert state_manifest["global_residual_signature"]["target_columns"]
    local_mixed = prepare_archetype_state_partition(
        states,
        side="long",
        archetype="mixed",
    )
    assert "target_signature_arch__long_mixed_mean_ev" in local_mixed
    assert "target_signature_arch__long_breakout_mean_ev" not in local_mixed
    assert np.allclose(
        local_mixed["target_mean_ev"].fillna(0.0),
        local_mixed["target_signature_arch__long_mixed_mean_ev"].fillna(0.0),
    )
    train = local_mixed.reset_index(drop=True)
    encoder = GlobalResidualSignatureEncoder(
        ResidualEncoderConfig(
            encoder_kind="hybrid_mlp",
            latent_dim=4,
            hidden_dims=(16, 8),
            epochs=8,
            patience=3,
            batch_size=64,
        )
    ).fit(train, features)
    assert not any(
        name.startswith(("target_signature_short_", "target_signature_arch__short_"))
        for name in encoder.target_columns
    )
    assert all(
        name.startswith("target_signature_arch__long_mixed_")
        for name in encoder.target_columns
    )
    oos = train.drop(
        columns=[name for name in train if name.startswith("target_")], errors="ignore"
    )
    transformed = encoder.transform(oos)
    assert transformed.filter(regex=r"^global_state_latent_").shape[1] == 4
    assert transformed.filter(regex=r"^global_state_pred_signature_").shape[1] > 0
    assert np.isfinite(transformed["global_state_input_novelty"]).all()
    assert "recent model outcomes" in encoder.manifest()["inference_contract"]
    assert encoder.manifest()["partition"]["token"] == "long_mixed"
    assert encoder.manifest()["training_report"]["mean_sample_weight"] >= 1.0
    assert encoder.manifest()["training_report"]["latent_covariance_weight"] > 0.0
    wrong_partition = oos.copy()
    wrong_partition["archetype_policy_key"] = "breakout"
    with pytest.raises(ValueError, match="partition mismatch"):
        encoder.transform(wrong_partition)


def test_variational_encoder_is_deterministic_oos_and_gmm_ready() -> None:
    rows = _reliability_rows()
    rows["selected_for_monitor"] = True
    rows["score_meta_base_soft_label"] = rows["hit_probability"].astype(np.float32)
    rows["mkt_systemic_deleveraging_score"] = np.sin(np.arange(len(rows)) / 9.0)
    states, features, _ = build_side_timestamp_states(
        rows,
        ["mkt_systemic_deleveraging_score"],
        [],
        StateVectorConfig(min_feature_coverage=0.1),
    )
    local = prepare_archetype_state_partition(
        states, side="long", archetype="mixed"
    ).reset_index(drop=True)
    encoder = GlobalResidualSignatureEncoder(
        ResidualEncoderConfig(
            encoder_kind="variational_ae",
            latent_dim=3,
            hidden_dims=(12, 6),
            epochs=5,
            patience=2,
            batch_size=64,
            kl_warmup_epochs=2,
        )
    ).fit(local, features)
    oos = local.drop(
        columns=[name for name in local if name.startswith("target_")],
        errors="ignore",
    )
    first = encoder.transform(oos)
    second = encoder.transform(oos)
    np.testing.assert_allclose(first, second, atol=1e-7)
    geometry = latent_geometry_diagnostics(
        first.filter(regex=r"^global_state_latent_").to_numpy()
    )
    assert geometry["finite_row_share"] == pytest.approx(1.0)
    assert geometry["dimensions"] == 3
    assert encoder.manifest()["training_report"]["effective_kl_weight"] > 0.0


def test_side_archetype_state_priors_transform_without_outcomes() -> None:
    frame = _reliability_rows().copy()
    frame["posterior_0"] = np.linspace(0.1, 0.9, len(frame), dtype=np.float32)
    frame["posterior_1"] = 1.0 - frame["posterior_0"]
    frame["dirty_positive"] = (
        frame["ev_after_1pct"].gt(0.0) & frame["full_path_bad_mae_1r"].gt(0.5)
    ).astype(np.float32)
    priors = SideArchetypeStatePriors().fit(frame, ["posterior_0", "posterior_1"])
    oos = frame.drop(
        columns=[
            "ev_after_1pct",
            "clean_exec",
            "dirty_positive",
            "full_path_bad_mae_1r",
            "timeout",
            "hit_probability",
        ],
        errors="ignore",
    )
    transformed = priors.transform(oos)
    assert transformed.filter(like="global_state_expected_arch_").shape[1] >= 6
    assert np.isfinite(transformed.to_numpy(dtype=float)).all()


def test_partition_local_state_priors_do_not_fallback_across_archetypes() -> None:
    frame = _reliability_rows().copy()
    frame["posterior_0"] = np.linspace(0.1, 0.9, len(frame), dtype=np.float32)
    frame["posterior_1"] = 1.0 - frame["posterior_0"]
    frame["dirty_positive"] = (
        frame["ev_after_1pct"].gt(0.0) & frame["full_path_bad_mae_1r"].gt(0.5)
    ).astype(np.float32)
    priors = SideArchetypeStatePriors(
        partition_local=True,
        strict_unknown=True,
    ).fit(frame, ["posterior_0", "posterior_1"])
    known = frame.iloc[:4].drop(
        columns=list(SideArchetypeStatePriors.target_sources.values()), errors="ignore"
    )
    assert np.isfinite(priors.transform(known).to_numpy(dtype=float)).all()
    unknown = known.copy()
    unknown["archetype_policy_key"] = "unseen"
    with pytest.raises(ValueError, match="No frozen state prior"):
        priors.transform(unknown)


def test_causal_hmm_is_outcome_free_prefix_stable_and_serializable(tmp_path) -> None:
    rng = np.random.default_rng(607)
    timestamps = pd.Series(pd.date_range("2025-01-01", periods=360, freq="h", tz="UTC"))
    state_ids = np.repeat(np.array([0, 1, 2, 1], dtype=np.int16), 90)
    state_ids = np.where(
        rng.random(len(state_ids)) < 0.04, (state_ids + 1) % 3, state_ids
    )
    states = pd.DataFrame({"global_state_id": state_ids})
    targets = pd.DataFrame(
        {
            "side_name": "long",
            "archetype_policy_key": "mixed",
            "target_signed_surprise": np.where(state_ids == 2, -0.4, 0.2),
            "target_positive_surprise": np.where(state_ids == 0, 0.8, 0.2),
            "target_negative_surprise": np.where(state_ids == 2, 0.7, 0.1),
            "target_negative_ev": np.where(state_ids == 2, 0.02, 0.002),
            "target_payoff_asymmetry": np.where(state_ids == 0, 0.01, -0.004),
        }
    )
    model = CausalCategoricalStateHMM(
        HMMGridConfig(hidden_states=(2, 3), n_iter=40, random_state=607)
    ).fit(states, targets, timestamps)
    full = model.transform(states, timestamps, continue_from_train=False)
    prefix = model.transform(
        states.iloc[:180], timestamps.iloc[:180], continue_from_train=False
    )
    posterior_columns = full.filter(regex=r"^global_hmm_posterior_[0-9]+$").columns
    np.testing.assert_allclose(
        full.loc[:179, posterior_columns], prefix[posterior_columns], atol=1e-6
    )
    np.testing.assert_allclose(full[posterior_columns].sum(axis=1), 1.0, atol=1e-6)
    assert "global_hmm_expected_negative_ev" in full

    artifact = tmp_path / "causal_hmm.joblib"
    joblib.dump(model, artifact)
    restored = joblib.load(artifact)
    restored_output = restored.transform(states, timestamps, continue_from_train=False)
    np.testing.assert_allclose(
        restored_output[posterior_columns], full[posterior_columns], atol=1e-7
    )


def test_binned_mi_screen_finds_stable_nonlinear_archetype_signal() -> None:
    rng = np.random.default_rng(812)
    rows = 1_200
    nonlinear = rng.uniform(-2.0, 2.0, rows).astype(np.float32)
    economic_state = (nonlinear**2 + rng.normal(0.0, 0.08, rows)).astype(np.float32)
    noise = rng.normal(size=rows).astype(np.float32)
    frame = pd.DataFrame(
        {
            "nonlinear_state": nonlinear,
            "noise": noise,
            "target_negative_surprise": economic_state,
            "target_negative_ev": economic_state,
            "target_bad_mae_rate": economic_state,
            "target_timeout_rate": economic_state,
            "target_positive_surprise": -economic_state,
            "target_mean_ev": -economic_state,
            "target_payoff_asymmetry": -economic_state,
        }
    )
    selected, diagnostic = select_partition_state_features(
        frame,
        ["nonlinear_state", "noise"],
        max_features=1,
        max_rows=rows,
    )
    by_feature = diagnostic.set_index("feature")
    assert abs(np.corrcoef(nonlinear, economic_state)[0, 1]) < 0.10
    assert _normalized_binned_mutual_information(nonlinear, economic_state) > 0.50
    assert (
        by_feature.loc["nonlinear_state", "mi_relevance"]
        > by_feature.loc["noise", "mi_relevance"]
    )
    assert selected == ["nonlinear_state"]
