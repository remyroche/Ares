from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.causal_market_regime_systems import (
    CausalContinuousContextConfig,
    CausalRelationshipBreakConfig,
    CausalMarketRegimeSystems,
    LATENT_GEOMETRY_SYSTEM_NAMES,
    MarketRegimeSystemConfig,
    PRIMARY_SEMANTIC_STATE_NAMES,
    _merge_low_support_component,
    build_causal_continuous_context_features,
    build_causal_relationship_break_features,
    continuous_context_feature_names,
    fit_causal_market_geometry_systems,
    latent_geometry_output_feature_names,
    relationship_break_feature_names,
)


def _panel(rows: int = 240) -> pd.DataFrame:
    x = np.arange(rows, dtype=np.float32)
    return pd.DataFrame(
        {
            "source_utc": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "trend_strength": np.sin(x / 12.0),
            "atr_percentile": 0.5 + 0.25 * np.cos(x / 17.0),
            "market_breadth_4h": np.sin(x / 19.0),
            "avg_pair_corr_24h": 0.4 + 0.2 * np.cos(x / 9.0),
            "funding_z": np.sin(x / 23.0),
            "oi_value_1d_log_chg": np.cos(x / 15.0),
            "ob_spread_bps_z_24h": np.sin(x / 11.0),
            "volume_zscore_48h": np.cos(x / 13.0),
        }
    )


def test_frozen_multigeometry_outputs_complete_causal_contract() -> None:
    train, evaluate = _panel().iloc[:180], _panel().iloc[180:]
    features = [column for column in train if column != "source_utc"]
    systems = CausalMarketRegimeSystems.fit(
        train,
        features,
        config=MarketRegimeSystemConfig(max_train_rows=180, max_proxy_rows=120, max_iter=40),
    )
    # A materializer can continue an adjacent fold from frozen train history;
    # use a gap here to assert the explicit reset semantics.
    evaluate = evaluate.copy()
    evaluate["source_utc"] += pd.Timedelta(hours=10)
    out = systems.transform(evaluate)

    primary_probs = [f"market_regime__state_p_{index}" for index in range(5)]
    assert set(primary_probs).issubset(out)
    assert np.allclose(out[primary_probs].sum(axis=1), 1.0, atol=1e-6)
    phase = [f"market_regime__phase_p_{name}" for name in ("stable", "onset", "active", "settling")]
    assert np.allclose(out[phase].sum(axis=1), 1.0, atol=1e-6)
    assert out.loc[out.index[0], "market_regime__state_age_hours"] == 0.0
    assert {"market_regime__entropy", "market_regime__top2_margin", "market_regime__state_switch_probability", "market_regime__ood_distance_percentile", "market_regime__input_coverage", "market_regime__phase_entropy", "market_regime__phase_top2_margin", "market_regime__assigned_centroid_distance", "market_regime__within_state_radius_percentile", "market_regime__state_boundary_margin", "market_regime__centroid_distance_velocity"}.issubset(out)
    assert out["market_regime__within_state_radius_percentile"].between(0.0, 1.0).all()
    for name in ("trend_volatility", "breadth_dependence", "leverage_flow", "liquidity"):
        prefix = f"geometry_regime__{name}__"
        assert any(column.startswith(prefix + "state_p_") for column in out)
        assert {prefix + "entropy", prefix + "top2_margin", prefix + "state_age_hours", prefix + "state_switch_probability", prefix + "ood_distance_percentile", prefix + "input_coverage"}.issubset(out)
    assert out.dtypes.eq(np.dtype("float32")).all()
    assert systems.models["primary"].state_count == 5
    assert systems.models["primary"].diagnostics["selected_k"] == 5
    semantic = [f"regime_p_{name}" for name in PRIMARY_SEMANTIC_STATE_NAMES]
    assert set(semantic).issubset(out)
    assert np.allclose(out[semantic].sum(axis=1), 1.0, atol=1e-6)
    mapping = systems.models["primary"].diagnostics["semantic_ontology"]
    assert mapping["status"] == "train_only_centroid_prototype_assignment"
    assert set(mapping["semantic_to_frozen_component"]) == set(PRIMARY_SEMANTIC_STATE_NAMES)
    for semantic_name, component in mapping["semantic_to_frozen_component"].items():
        np.testing.assert_allclose(
            out[f"regime_p_{semantic_name}"],
            out[f"market_regime__state_p_{component}"],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            out[f"market_regime__{semantic_name}_probability"],
            out[f"regime_p_{semantic_name}"],
            atol=1e-7,
        )
    assert {"regime_entropy", "regime_top2_margin", "state_age", "state_switch_probability", "market_regime__direction_score", "market_regime__direction_positive_probability", "market_direction_sign"}.issubset(out)
    assert out["market_direction_sign"].isin((-1.0, 0.0, 1.0)).all()
    stickiness = systems.models["primary"].diagnostics["stickiness_selection"]
    assert stickiness
    assert {"persistent_state_gate_passed", "median_dwell_hours", "temporal_switch_rate"}.issubset(stickiness[0])


def test_geometry_only_systems_are_disjoint_from_primary_and_have_stable_padded_schema() -> None:
    train, evaluate = _panel().iloc[:180], _panel().iloc[180:]
    features = [column for column in train if column != "source_utc"]
    systems = fit_causal_market_geometry_systems(
        train,
        features,
        config=MarketRegimeSystemConfig(max_train_rows=180, max_proxy_rows=100, max_iter=35),
    )
    assert tuple(systems.models) == LATENT_GEOMETRY_SYSTEM_NAMES
    assert "primary" not in systems.models
    views = [set(systems.feature_views[name]) for name in LATENT_GEOMETRY_SYSTEM_NAMES]
    assert all(left.isdisjoint(right) for index, left in enumerate(views) for right in views[index + 1:])
    output = systems.transform(evaluate)
    assert not any(column.startswith("market_regime__") for column in output)
    for system in LATENT_GEOMETRY_SYSTEM_NAMES:
        posterior = [name for name in output if name.startswith(f"geometry_regime__{system}__state_p_")]
        assert posterior
        assert np.allclose(output[posterior].sum(axis=1), 1.0, atol=1e-6)
    padded = latent_geometry_output_feature_names(include_memberships=True)
    invariants = latent_geometry_output_feature_names(include_memberships=False)
    assert set(invariants).issubset(padded)
    assert all("__state_p_" not in name for name in invariants)


def test_geometry_config_is_meta_candidate_only_and_memberships_are_gated() -> None:
    from extreme_price_movements.config import CFG
    from extreme_price_movements.stage_i_feature_selection import resolve_stage_i_feature_universe

    base = resolve_stage_i_feature_universe(
        CFG, layer="base", side="long", head="R3_economic_simplex_b25",
    )
    meta = resolve_stage_i_feature_universe(
        CFG, layer="meta", side="long", head="shared_exact_net_residual",
    )
    invariant = set(CFG["CAUSAL_LATENT_GEOMETRY_META_CANDIDATE_FEATURE_KEYS"])
    memberships = set(CFG["CAUSAL_LATENT_GEOMETRY_MEMBERSHIP_CANDIDATE_FEATURE_KEYS"])
    assert invariant
    assert memberships
    assert not any(name.startswith("geometry_regime__") for name in base)
    # Structural eligibility alone does not promote a latent bundle.  The
    # matched economic/IC portability gate must pass first.
    assert invariant.isdisjoint(meta)
    assert memberships.isdisjoint(meta)
    assert all("__state_p_" not in name for name in invariant)
    assert all("__state_p_" in name for name in memberships)
    gate = CFG["CAUSAL_LATENT_GEOMETRY_MEMBERSHIP_PROMOTION_GATE"]
    assert gate["status"] == "candidate_only_not_promoted"
    assert gate["forbidden_layers"] == ["base"]


def test_future_rows_cannot_change_prior_states_and_outcomes_are_rejected() -> None:
    train, evaluate = _panel().iloc[:180], _panel().iloc[180:]
    features = [column for column in train if column != "source_utc"]
    systems = CausalMarketRegimeSystems.fit(
        train,
        features,
        config=MarketRegimeSystemConfig(max_train_rows=180, max_proxy_rows=100, max_iter=35),
    )
    baseline = systems.transform(evaluate)
    changed = evaluate.copy()
    changed.loc[changed.index[20]:, features] *= -50.0
    mutated = systems.transform(changed)
    pd.testing.assert_frame_equal(baseline.iloc[:20], mutated.iloc[:20])

    bad = train.assign(target__future_net_ev=1.0)
    with pytest.raises(ValueError, match="label/outcome"):
        CausalMarketRegimeSystems.fit(bad, [*features, "target__future_net_ev"])


def test_primary_k_override_and_postfit_low_support_merge_preserve_simplex() -> None:
    train = _panel()
    features = [column for column in train if column != "source_utc"]
    systems = CausalMarketRegimeSystems.fit(
        train,
        features,
        config=MarketRegimeSystemConfig(primary_state_count=3, max_train_rows=220, max_proxy_rows=120, max_iter=35),
    )
    assert systems.models["primary"].state_count == 3
    values = np.asarray([[0.97, 0.02, 0.01], [0.80, 0.15, 0.05]], dtype=np.float32)
    collapsed, source, target, occupancy = _merge_low_support_component(
        values,
        np.asarray([[0.0], [1.0], [3.0]], dtype=np.float32),
        minimum_occupancy=0.10,
    )
    assert occupancy.shape == (3,)
    assert source == 2 and target == 1
    assert collapsed.shape == (2, 2)
    assert np.allclose(collapsed.sum(axis=1), 1.0, atol=1e-6)


def test_continuous_context_is_strict_prequential_and_excludes_memberships() -> None:
    rows = 160
    source = np.arange(rows, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "source_utc": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "observable": source,
        }
    )
    config = CausalContinuousContextConfig(
        rank_windows_days=(2, 4), recent_median_days=2,
        min_reference_rows=24,
    )
    contract = {"trend": "observable"}
    baseline = build_causal_continuous_context_features(frame, contract, config=config)
    assert tuple(baseline.columns) == continuous_context_feature_names(contract, config=config)
    assert baseline.dtypes.eq(np.dtype("float32")).all()
    # A strictly increasing source is above every prior observation.  The
    # rolling rank therefore tends to one, while the current row itself never
    # changes any earlier context value.
    assert baseline.loc[100, "continuous_regime__trend__rank_2d"] == pytest.approx(1.0)
    assert baseline.loc[100, "continuous_regime__trend__change_4h"] == pytest.approx(4.0)
    assert baseline.loc[100, "continuous_regime__trend__change_24h"] == pytest.approx(24.0)
    assert baseline.loc[100, "continuous_regime__trend__distance_recent_median_2d"] > 20.0

    changed = frame.copy()
    changed.loc[120:, "observable"] = -10_000.0
    mutated = build_causal_continuous_context_features(changed, contract, config=config)
    pd.testing.assert_frame_equal(baseline.iloc[:120], mutated.iloc[:120])

    bad = frame.assign(market_regime__state_p_0=0.5)
    with pytest.raises(ValueError, match="membership"):
        build_causal_continuous_context_features(
            bad,
            {"bad": "market_regime__state_p_0"},
            config=config,
        )


def test_relationship_breaks_are_strict_prequential_and_compact() -> None:
    rows = 160
    trend = np.arange(rows, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "source_utc": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "trend_raw": trend,
            "breadth_raw": 2.0 * trend + 3.0,
        }
    )
    contract = {"trend_quality": "trend_raw", "breadth": "breadth_raw"}
    config = CausalRelationshipBreakConfig(windows_days=(2, 4), min_reference_rows=24)
    baseline = build_causal_relationship_break_features(frame, contract, config=config)
    assert tuple(baseline.columns) == relationship_break_feature_names(contract, config=config)
    assert baseline.dtypes.eq(np.dtype("float32")).all()
    assert (
        "continuous_regime__relationship_break__volatility_liquidity__residual_abs_2d"
        not in baseline
    )
    signed_2d = "continuous_regime__relationship_break__trend_breadth__residual_signed_2d"
    absolute_2d = "continuous_regime__relationship_break__trend_breadth__residual_abs_2d"
    assert baseline.loc[100, signed_2d] == pytest.approx(0.0, abs=1e-5)
    assert baseline.loc[100, absolute_2d] == pytest.approx(0.0, abs=1e-5)

    # A current break is evaluated against the preceding fitted relationship;
    # it is not diluted by itself or any subsequent observation.
    broken = frame.copy()
    broken.loc[110, "breadth_raw"] += 10.0
    observed = build_causal_relationship_break_features(broken, contract, config=config)
    assert observed.loc[110, signed_2d] == pytest.approx(10.0, abs=1e-4)
    assert observed.loc[110, absolute_2d] == pytest.approx(10.0, abs=1e-4)

    # Future observations cannot alter any already-issued feature value.
    future_changed = frame.copy()
    future_changed.loc[120:, "breadth_raw"] = -10_000.0
    mutated = build_causal_relationship_break_features(future_changed, contract, config=config)
    pd.testing.assert_frame_equal(baseline.iloc[:120], mutated.iloc[:120])

    with pytest.raises(ValueError, match="membership"):
        build_causal_relationship_break_features(
            frame.assign(market_regime__state_p_0=0.5),
            {"trend_quality": "trend_raw", "breadth": "market_regime__state_p_0"},
            config=config,
        )


def test_full_observable_contract_exposes_all_six_predeclared_relationships() -> None:
    names = relationship_break_feature_names()
    assert any("__volatility_liquidity__" in name for name in names)
    assert any("__isolation_dependence__" in name for name in names)
