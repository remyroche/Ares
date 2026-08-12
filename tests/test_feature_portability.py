from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.feature_portability import (
    FeaturePortabilityError,
    FeatureRole,
    FeatureSemanticRole,
    PortabilityPolicy,
    assign_feature_dispositions,
    causal_rolling_portability_transform_batches,
    causal_rolling_portability_transforms,
    classify_feature_roles,
    feature_portability_diagnostics,
    estimate_causal_rolling_transform_memory,
)
from extreme_price_movements.feature_family_registry import FeatureFamily
from extreme_price_movements.feature_transforms import CausalFeatureTransformer
from extreme_price_movements.features import (
    _kalman_local_level_df,
    add_cross_sectional_peer_context_features,
    add_time_series_percentile_features,
    compute_features_hourly,
    compute_market_features,
)
from extreme_price_movements.features_oi import compute_oi_features


def test_full_graph_portability_repair_includes_explicit_contract_keys(monkeypatch) -> None:
    """A repair must not silently omit a frozen artifact-only output.

    The normal no-request path builds its graph from the current config union.
    A portability migration may intentionally include a selected feature that
    is absent from that union, so it needs an explicit additive closure.
    """
    import extreme_price_movements.features as features_module
    from extreme_price_movements.config import CFG

    captured: dict[str, set[str]] = {}

    def _capture(_panel, _gates, _cfg, requested_feature_keys=None):
        captured["requested"] = set(requested_feature_keys or [])
        return "captured"

    monkeypatch.setattr(features_module, "_compute_features_impl", _capture)
    cfg = dict(CFG)
    cfg["feature_portability_repair_keys"] = ["btc_over_eth_dominance_roc"]

    assert compute_features_hourly({}, pd.DataFrame(), cfg) == "captured"
    assert "btc_over_eth_dominance_roc" in captured["requested"]


def test_selected_runtime_backfill_uses_narrow_causal_closure_and_persists_outputs_only(
    monkeypatch,
) -> None:
    """Selected residual/composite repairs need parents, not the full graph."""
    from types import SimpleNamespace

    import extreme_price_movements.pipeline_steps as pipeline_steps

    selected = ["ret4h_bench_resid", "q_iqr__ret1h"]
    captured: dict[str, list[str]] = {}
    index = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    columns = ["BTC/USD:USD"]

    monkeypatch.setattr(pipeline_steps, "_ensure_feature_runtime_support", lambda: None)

    def _fake_static(*_args, requested_feature_keys=None, **_kwargs):
        captured["requested"] = list(requested_feature_keys or [])
        return SimpleNamespace(
            features={
                key: pd.DataFrame(1.0, index=index, columns=columns, dtype=np.float32)
                for key in captured["requested"]
            },
            index=index,
            columns=columns,
        )

    monkeypatch.setattr(pipeline_steps, "compute_static_features", _fake_static)
    monkeypatch.setattr(
        pipeline_steps, "_inject_orderbook_summary_features", lambda feats, *_args: feats
    )
    monkeypatch.setattr(
        pipeline_steps, "_inject_orderbook_wall_features", lambda feats, *_args, **_kwargs: feats
    )

    def _add_residual(feats, *_args):
        feats["ret4h_bench_resid"] = pd.DataFrame(
            2.0, index=index, columns=columns, dtype=np.float32
        )

    monkeypatch.setattr(pipeline_steps, "add_residual_features", _add_residual)
    monkeypatch.setattr(
        pipeline_steps.epm_features,
        "_add_regime_panel_composite_features",
        lambda *_args, **_kwargs: set(),
    )

    result, _, _ = pipeline_steps._compute_features_hourly_runtime(
        {}, pd.DataFrame(index=index), {}, {}, requested_feature_keys=selected
    )

    requested = set(captured["requested"])
    assert set(selected).issubset(requested)
    assert set(pipeline_steps.RESIDUAL_PARENT_FEATURE_KEYS).issubset(requested)
    assert "ret1h" in requested  # q_iqr__ret1h static-generator parent.
    assert "unrelated_feature" not in requested
    assert set(result) == set(selected)


def test_fail_closed_selected_repair_relaxes_compute_policy_for_barred_parent(
    monkeypatch,
) -> None:
    """A barred prerequisite must not abort a selected fail-closed migration."""
    from types import SimpleNamespace

    import extreme_price_movements.pipeline_steps as pipeline_steps

    selected = ["ret4h_bench_resid"]
    index = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    columns = ["BTC/USD:USD"]
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(pipeline_steps, "_ensure_feature_runtime_support", lambda: None)
    monkeypatch.setattr(
        pipeline_steps, "RESIDUAL_PARENT_FEATURE_KEYS", frozenset({"barred_parent"})
    )
    monkeypatch.setattr(
        pipeline_steps,
        "_is_feature_allowed_by_runtime_portability_policy",
        lambda name, _cfg: name != "barred_parent",
    )

    def _fake_static(*_args, requested_feature_keys=None, **_kwargs):
        captured["requested"] = list(requested_feature_keys or [])
        return SimpleNamespace(
            features={
                key: pd.DataFrame(1.0, index=index, columns=columns, dtype=np.float32)
                for key in captured["requested"]
            },
            index=index,
            columns=columns,
        )

    monkeypatch.setattr(pipeline_steps, "compute_static_features", _fake_static)
    monkeypatch.setattr(
        pipeline_steps, "_inject_orderbook_summary_features", lambda feats, *_args: feats
    )
    monkeypatch.setattr(
        pipeline_steps, "_inject_orderbook_wall_features", lambda feats, *_args, **_kwargs: feats
    )
    monkeypatch.setattr(
        pipeline_steps, "add_residual_features", lambda feats, *_args: None
    )
    monkeypatch.setattr(
        pipeline_steps.epm_features,
        "_add_regime_panel_composite_features",
        lambda *_args, **_kwargs: set(),
    )

    result, _, _ = pipeline_steps._compute_features_hourly_runtime(
        {},
        pd.DataFrame(index=index),
        {
            "feature_portability_mode": "cross_asset_portable",
            "feature_portability_strict": False,
            "feature_portability_repair_keys": selected,
            "feature_portability_selected_dependency_closure": True,
        },
        {},
        requested_feature_keys=selected,
    )

    assert "barred_parent" in captured["requested"]
    assert set(result) == set(selected)


def test_role_inventory_is_safe_by_default_and_respects_overrides() -> None:
    roles = classify_feature_roles(
        [
            "ret_1h_atr",
            "close_price",
            "symbol",
            "market_regime_state_id",
            "exact_net_bps",
            "decision_timestamp",
            "unclassified_signal",
            "venue_raw_value",
        ],
        overrides={"venue_raw_value": FeatureRole.PORTABLE},
    ).set_index("feature")
    assert roles.loc["ret_1h_atr", "role"] == FeatureSemanticRole.RELATIVE_LEVEL
    assert roles.loc["close_price", "role"] == FeatureSemanticRole.LEVEL
    assert roles.loc["symbol", "portability_role"] == FeatureRole.IDENTITY
    assert roles.loc["market_regime_state_id", "portability_role"] == FeatureRole.FOLD_LOCAL_STATE
    assert roles.loc["exact_net_bps", "portability_role"] == FeatureRole.OUTCOME_DERIVED
    assert roles.loc["decision_timestamp", "portability_role"] == FeatureRole.CONTROL
    assert roles.loc["unclassified_signal", "portability_role"] == FeatureRole.UNKNOWN
    assert roles.loc["venue_raw_value", "portability_role"] == FeatureRole.PORTABLE


def test_selected_causal_feature_names_are_not_misclassified_as_identity_or_outcome() -> None:
    """Selected ``asset_*``/``barrier_*`` names are entry-time primitives."""
    roles = classify_feature_roles(
        [
            "asset_minus_mkt_price_recovery_fraction_24h",
            "asset_mkt_exhaustion_phase_divergence",
            "up_barrier_pressure_daily_donchian",
            "dist_ema_fast_ts_resid",
        ]
    ).set_index("feature")
    assert roles.loc["asset_minus_mkt_price_recovery_fraction_24h", "portability_role"] == FeatureRole.PORTABLE
    assert roles.loc["asset_mkt_exhaustion_phase_divergence", "portability_role"] == FeatureRole.PORTABLE
    assert roles.loc["up_barrier_pressure_daily_donchian", "portability_role"] == FeatureRole.PORTABLE
    assert roles.loc["dist_ema_fast_ts_resid", "portability_role"] == FeatureRole.PORTABLE


def test_stage_a_semantic_role_taxonomy_is_complete() -> None:
    roles = classify_feature_roles(
        [
            "close_price",
            "ret_1h",
            "oi_change_4h",
            "return_velocity_3h",
            "relationship_break",
            "setup_alignment",
            "recent_trust_score",
            "base_model_score",
        ]
    ).set_index("feature")
    assert roles.loc["close_price", "role"] == FeatureSemanticRole.LEVEL
    assert roles.loc["ret_1h", "role"] == FeatureSemanticRole.RELATIVE_LEVEL
    assert roles.loc["oi_change_4h", "role"] == FeatureSemanticRole.CHANGE
    assert roles.loc["return_velocity_3h", "role"] == FeatureSemanticRole.ACCELERATION
    assert roles.loc["relationship_break", "role"] == FeatureSemanticRole.RELATIONSHIP_BREAK
    assert roles.loc["setup_alignment", "role"] == FeatureSemanticRole.SETUP_ALIGNMENT
    assert roles.loc["recent_trust_score", "role"] == FeatureSemanticRole.SUPPORT_OR_TRUST
    assert roles.loc["base_model_score", "role"] == FeatureSemanticRole.MODEL_OUTPUT


def test_causal_rolling_transforms_are_grouped_order_stable_and_have_no_lookahead() -> None:
    timestamp = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "ts": [timestamp[1], timestamp[0], timestamp[2], timestamp[0], timestamp[1]],
            "symbol": ["A", "A", "A", "B", "B"],
            "raw_price": [20.0, 10.0, 30.0, 100.0, 80.0],
        },
        index=[9, 3, 8, 4, 1],
    )
    first = causal_rolling_portability_transforms(
        frame,
        feature_names=["raw_price"],
        timestamp_column="ts",
        group_columns=["symbol"],
        rank_windows=[2],
        robust_z_windows=[3],
        change_periods=[1],
        minimum_periods=2,
    )
    modified = frame.copy()
    # Altering the last A observation cannot change preceding A rows.  It also
    # cannot affect B because transforms are explicitly per symbol.
    modified.loc[8, "raw_price"] = 3_000.0
    second = causal_rolling_portability_transforms(
        modified,
        feature_names=["raw_price"],
        timestamp_column="ts",
        group_columns=["symbol"],
        rank_windows=[2],
        robust_z_windows=[3],
        change_periods=[1],
        minimum_periods=2,
    )
    assert first.index.equals(frame.index)
    assert first.loc[[9, 3, 4, 1]].equals(second.loc[[9, 3, 4, 1]])
    # A at its second observation: values [10, 20], rank=1 and delta=10.
    assert first.loc[9, "raw_price__causal_rank_w2"] == pytest.approx(1.0)
    assert first.loc[9, "raw_price__causal_delta_p1"] == pytest.approx(10.0)
    # B is isolated, so its second value is compared only with 100, not A.
    assert first.loc[1, "raw_price__causal_delta_p1"] == pytest.approx(-20.0)


def test_batched_causal_transform_matches_materialized_contract_and_has_explicit_budget() -> None:
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC"),
            "asset": ["A"] * 4 + ["B"] * 4,
            "x": np.arange(8, dtype=float),
            "y": np.arange(8, dtype=float) * -2.0,
        }
    )
    batches = list(
        causal_rolling_portability_transform_batches(
            frame, feature_names=["x", "y"], timestamp_column="ts", group_columns=["asset"],
            rank_windows=[2], robust_z_windows=[3], change_periods=[1],
            minimum_periods=2, feature_batch_size=1,
        )
    )
    assert len(batches) == 2
    materialized = causal_rolling_portability_transforms(
        frame, feature_names=["x", "y"], timestamp_column="ts", group_columns=["asset"],
        rank_windows=[2], robust_z_windows=[3], change_periods=[1], minimum_periods=2,
    )
    pd.testing.assert_frame_equal(pd.concat(batches, axis=1), materialized)
    estimate = estimate_causal_rolling_transform_memory(
        rows=1_000_000, source_features=32, rank_windows=[90, 180],
        robust_z_windows=[90, 180], change_periods=[4, 24], feature_batch_size=1,
    )
    assert estimate.generated_columns_per_source == 8
    assert estimate.materialized_output_bytes == 1_024_000_000
    assert estimate.peak_batch_working_bytes < 384 * 1024 * 1024
    compact = estimate_causal_rolling_transform_memory(
        rows=1_000_000, source_features=32, rank_windows=[90, 180],
        robust_z_windows=[], change_periods=[4, 24], include_relative_change=False,
        feature_batch_size=1,
    )
    assert compact.generated_columns_per_source == 4
    assert compact.materialized_output_bytes == 512_000_000
    safe_batches = list(
        causal_rolling_portability_transform_batches(
            frame, feature_names=["x"], timestamp_column="ts", group_columns=["asset"],
            rank_windows=[2], robust_z_windows=[], change_periods=[1],
            include_relative_change=False, minimum_periods=2,
        )
    )
    assert set(safe_batches[0]) == {"x__causal_rank_w2", "x__causal_delta_p1"}
    with pytest.raises(FeaturePortabilityError, match="exceeds memory budget"):
        list(
            causal_rolling_portability_transform_batches(
                frame, feature_names=["x"], timestamp_column="ts", rank_windows=[2],
                robust_z_windows=[2], change_periods=[1], minimum_periods=2,
                max_batch_working_bytes=1,
            )
        )


def test_per_era_diagnostics_report_coverage_drift_and_effects_by_stratum() -> None:
    rows = []
    for side in ("long", "short"):
        for era, offset in (("early", 0.0), ("late", 5.0)):
            for value in range(10):
                rows.append(
                    {
                        "side": side,
                        "era": era,
                        "portable_ret": float(value + offset),
                        "target": float(value if side == "long" else -value),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.loc[(frame["side"] == "short") & (frame["era"] == "late") & (frame.index % 3 == 0), "portable_ret"] = np.nan
    audit = feature_portability_diagnostics(
        frame,
        feature_names=["portable_ret"],
        era_column="era",
        target_column="target",
        strata_columns=["side"],
        reference_era="early",
    ).set_index(["feature", "side", "era"])
    assert audit.loc[("portable_ret", "long", "early"), "coverage"] == pytest.approx(1.0)
    assert audit.loc[("portable_ret", "short", "late"), "coverage"] < 1.0
    assert audit.loc[("portable_ret", "long", "late"), "robust_median_shift"] > 0.0
    assert audit.loc[("portable_ret", "long", "late"), "effect_spearman"] > 0.99
    assert audit.loc[("portable_ret", "short", "late"), "effect_spearman"] < -0.99
    assert audit.loc[("portable_ret", "long", "late"), "psi"] > 0.0


def test_dispositions_distinguish_portability_support_effect_and_unsafe_roles() -> None:
    diagnostics = pd.DataFrame(
        [
            {"feature": "ret_1h", "coverage": 1.0, "finite_rows": 200, "unique_values": 20, "robust_median_shift": 0.2, "psi": 0.01, "effect_support": 200, "effect_spearman": 0.10},
            {"feature": "ret_1h", "coverage": 1.0, "finite_rows": 200, "unique_values": 20, "robust_median_shift": 0.3, "psi": 0.02, "effect_support": 200, "effect_spearman": 0.12},
            {"feature": "close_price", "coverage": 1.0, "finite_rows": 200, "unique_values": 20, "robust_median_shift": 0.0, "psi": 0.0, "effect_support": 0, "effect_spearman": np.nan},
            {"feature": "exact_net_bps", "coverage": 1.0, "finite_rows": 200, "unique_values": 20, "robust_median_shift": 0.0, "psi": 0.0, "effect_support": 0, "effect_spearman": np.nan},
            {"feature": "portable_unstable", "coverage": 1.0, "finite_rows": 200, "unique_values": 20, "robust_median_shift": 0.0, "psi": 0.0, "effect_support": 200, "effect_spearman": -0.4},
            {"feature": "portable_unstable", "coverage": 1.0, "finite_rows": 200, "unique_values": 20, "robust_median_shift": 0.0, "psi": 0.0, "effect_support": 200, "effect_spearman": 0.4},
        ]
    )
    roles = classify_feature_roles(diagnostics["feature"].drop_duplicates())
    roles.loc[roles["feature"].eq("portable_unstable"), "role"] = FeatureSemanticRole.RELATIVE_LEVEL
    roles.loc[roles["feature"].eq("portable_unstable"), "portability_role"] = FeatureRole.PORTABLE
    result = assign_feature_dispositions(
        diagnostics,
        roles=roles,
        policy=PortabilityPolicy(min_finite_support=100, min_effect_support=100),
    ).set_index("feature")
    assert result.loc["ret_1h", "disposition"] == "KEEP_PORTABLE"
    assert result.loc["close_price", "disposition"] == "TRANSFORM_CAUSALLY"
    assert result.loc["exact_net_bps", "disposition"] == "EXCLUDE_OUTCOME_DERIVED"
    assert result.loc["portable_unstable", "disposition"] == "REVIEW_EFFECT_INSTABILITY"


def test_invalid_roll_window_and_missing_reference_era_fail_closed() -> None:
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"),
            "x": [1.0, 2.0, 3.0],
            "era": ["a", "a", "b"],
        }
    )
    with pytest.raises(FeaturePortabilityError, match="positive"):
        causal_rolling_portability_transforms(
            frame, feature_names=["x"], timestamp_column="ts", rank_windows=[0]
        )
    with pytest.raises(FeaturePortabilityError, match="reference_era"):
        feature_portability_diagnostics(
            frame, feature_names=["x"], era_column="era", reference_era="missing"
        )


def test_causal_normalizers_preserve_real_zero_and_never_turn_missing_into_neutral() -> None:
    """A warm-up/missing observation must not be indistinguishable from zero.

    This is the contract used by the repaired liquidity, impact, residual and
    movement families: a real zero is a finite observation and may receive a
    normalized value once its trailing support exists; a missing source stays
    unavailable rather than being silently zero-imputed.
    """
    ts = pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "ts": ts,
            "symbol": ["A"] * len(ts),
            # The zero at position four is a genuine neutral observation.
            "impact_log": [-2.0, -1.0, 0.5, 1.0, 0.0, np.nan, 2.0, 3.0],
            "residual_vol_scaled": [-1.0, -0.5, 0.0, 0.5, 0.0, np.nan, 1.0, 1.5],
        }
    )
    result = causal_rolling_portability_transforms(
        frame,
        feature_names=["impact_log", "residual_vol_scaled"],
        timestamp_column="ts",
        group_columns=["symbol"],
        rank_windows=[4],
        robust_z_windows=[4],
        change_periods=[],
        minimum_periods=3,
    )

    # No fabricated neutral feature during the insufficient-history prefix.
    assert result.loc[0, "impact_log__causal_rank_w4"] != 0.0
    assert np.isnan(result.loc[0, "impact_log__causal_rank_w4"])
    # The actual zero is observed and ranks in a window with adequate support.
    assert np.isfinite(result.loc[4, "impact_log__causal_rank_w4"])
    assert np.isfinite(result.loc[4, "residual_vol_scaled__causal_rank_w4"])
    # A missing source row remains missing in every generated representation;
    # it cannot masquerade as a zero-impact / zero-residual observation.
    for column in result:
        assert np.isnan(result.loc[5, column]), column


def test_causal_impact_and_residual_normalizers_are_invariant_to_future_scale_shock() -> None:
    """Log-impact and volatility-scaled residual normalizers are P.I.T.

    A large liquidity/return shock after the decision bar must not revise the
    normalized values available for an earlier decision.  This also covers
    the clipped, high-magnitude tail where accidental full-sample fitting is
    easiest to miss in ordinary value tests.
    """
    ts = pd.date_range("2025-02-01", periods=12, freq="h", tz="UTC")
    base = pd.DataFrame(
        {
            "ts": ts,
            "symbol": ["A"] * len(ts),
            "log_impact": np.linspace(-6.0, -2.0, len(ts)),
            "residual_over_vol": np.linspace(-1.5, 1.5, len(ts)),
        }
    )
    shocked = base.copy()
    shocked.loc[9:, "log_impact"] = [15.0, 25.0, 35.0]
    shocked.loc[9:, "residual_over_vol"] = [-100.0, 100.0, -100.0]
    kwargs = dict(
        feature_names=["log_impact", "residual_over_vol"],
        timestamp_column="ts",
        group_columns=["symbol"],
        rank_windows=[6],
        robust_z_windows=[6],
        change_periods=[],
        minimum_periods=4,
    )
    before = causal_rolling_portability_transforms(base, **kwargs)
    after = causal_rolling_portability_transforms(shocked, **kwargs)
    pd.testing.assert_frame_equal(before.iloc[:9], after.iloc[:9])


def test_causal_robust_z_requires_non_degenerate_scale_instead_of_emitting_zero() -> None:
    """A constant history has no standardized signal, not a neutral signal."""
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2025-03-01", periods=6, freq="h", tz="UTC"),
            "symbol": ["A"] * 6,
            "raw_spread_bps": [5.0] * 6,
        }
    )
    result = causal_rolling_portability_transforms(
        frame,
        feature_names=["raw_spread_bps"],
        timestamp_column="ts",
        group_columns=["symbol"],
        rank_windows=[],
        robust_z_windows=[4],
        change_periods=[],
        minimum_periods=3,
    )
    assert result["raw_spread_bps__causal_robust_z_w4"].isna().all()


def test_generic_causal_transform_rejects_warmup_and_flat_zscore_as_neutral() -> None:
    """The generic accelerated path must honour the same missingness contract."""
    transformer = CausalFeatureTransformer(roll_window=4, enable_cache=False)
    raw = np.array([[1.0], [1.0], [1.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    out = transformer._apply_transform_numpy(
        raw.copy(), family=FeatureFamily.RISK_NORMALIZED_CONTINUOUS
    )
    # Three observations are not a valid four-bar normalisation; the fourth is
    # still flat. Neither may be fabricated as a neutral z-score.
    assert np.isnan(out[:4]).all()
    assert np.isfinite(out[4:]).all()


def test_kalman_state_is_invariant_to_appended_future_rows() -> None:
    """The local-level state may use only its declared bootstrap and past."""
    index = pd.date_range("2025-01-01", periods=20, freq="h", tz="UTC")
    baseline = pd.DataFrame({"x": np.linspace(1.0, 3.0, len(index))}, index=index)
    extended = pd.concat(
        [
            baseline,
            pd.DataFrame(
                {"x": [10_000.0, -10_000.0, 20_000.0]},
                index=pd.date_range(index[-1] + pd.Timedelta(hours=1), periods=3, freq="h", tz="UTC"),
            ),
        ]
    )
    before, _, _, _ = _kalman_local_level_df(
        baseline, lambda_qr=0.05, bootstrap_rows=8
    )
    after, _, _, _ = _kalman_local_level_df(
        extended, lambda_qr=0.05, bootstrap_rows=8
    )
    pd.testing.assert_frame_equal(before, after.iloc[: len(before)])


def test_market_return_is_price_scale_invariant_and_not_a_price_level() -> None:
    """Market return names must represent supported aggregated log returns."""
    index = pd.date_range("2025-01-01", periods=40, freq="h", tz="UTC")
    symbols = [f"A{i}" for i in range(5)]
    base = np.exp(np.linspace(0.0, 0.2, len(index))).astype(np.float32)
    close = pd.DataFrame(
        {symbol: base * (1.0 + 0.01 * i) for i, symbol in enumerate(symbols)},
        index=index,
    )
    panel = {
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": pd.DataFrame(100.0, index=index, columns=symbols),
    }
    rescaled = dict(panel)
    rescaled["close"] = close * 10_000.0
    rescaled["high"] = panel["high"] * 10_000.0
    rescaled["low"] = panel["low"] * 10_000.0
    first = compute_market_features(panel, symbols, trend_sma_hours=12)
    second = compute_market_features(rescaled, symbols, trend_sma_hours=12)
    pd.testing.assert_series_equal(
        first["mkt_ret1h"], second["mkt_ret1h"], check_exact=False, rtol=1e-3, atol=1e-7
    )
    assert first["mkt_ret1h"].iloc[1] != pytest.approx(close[symbols].iloc[1].mean())


def test_cross_sectional_and_time_percentiles_fail_closed_without_support() -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC")
    sparse = pd.DataFrame({"A": np.arange(8, dtype=np.float32)}, index=index)
    peer = add_cross_sectional_peer_context_features({"ret6h": sparse}, min_group_size=5)
    assert peer["cs_rank_ret6h"].isna().all().all()
    assert peer["cs_rz_ret6h"].isna().all().all()
    ts = add_time_series_percentile_features(
        {"ret1h": sparse}, lookback=8, min_history_fraction=0.75
    )
    assert ts["ts_pct_ret1h"].iloc[:5].isna().all().all()


def test_missing_funding_is_not_encoded_as_a_neutral_oi_state() -> None:
    """Funding-dependent composites fail closed when the feed is absent."""
    index = pd.date_range("2025-01-01", periods=24 * 10, freq="h", tz="UTC")
    columns = ["AAA/USD:USD", "BBB/USD:USD"]
    phase = np.linspace(0.0, 1.0, len(index), dtype=np.float32)[:, None]
    price = pd.DataFrame(
        100.0 * np.exp(0.02 * phase) * np.array([[1.0, 1.1]], dtype=np.float32),
        index=index,
        columns=columns,
    )
    oi = pd.DataFrame(
        10_000.0 * (1.0 + 0.01 * np.repeat(phase, len(columns), axis=1)),
        index=index,
        columns=columns,
    )
    qv = pd.DataFrame(50_000.0, index=index, columns=columns)
    output = compute_oi_features(
        oi_native=oi,
        price=price,
        quote_volume=qv,
        funding_rate=None,
        bars_per_day=24,
    )
    # This state has an explicit funding component.  A missing funding panel
    # must not appear as a finite zero / neutral liquidation signal.
    assert output["asset_liquidation_phase_score"].isna().all().all()
    assert output["asset_short_covering_score"].isna().all().all()
