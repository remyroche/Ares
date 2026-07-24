from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.features_negative_residuals import (
    NEGATIVE_RESIDUAL_CAUSAL_WINDOW_HOURS,
    NEGATIVE_RESIDUAL_COMPOSITE_FEATURE_KEYS,
    NEGATIVE_RESIDUAL_FEATURE_SCHEMA_VERSION,
    NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
    NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS,
    add_residual_state_target_composites,
    add_negative_residual_features,
    expand_negative_residual_feature_dependencies,
    negative_residual_feature_contract,
    residual_state_target_feature_names,
    _symbol,
)
from extreme_price_movements.market_regime_change_contract import (
    MARKET_REGIME_CHANGE_FEATURE_KEYS,
)


def _panels(rows: int = 900) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    columns = pd.Index(
        [
            "BTC/USD:USD",
            "ETH/USD:USD",
            "SOL/USD:USD",
            "XRP/USD:USD",
            "ADA/USD:USD",
        ]
    )
    rng = np.random.default_rng(7)

    def panel(scale: float = 1.0) -> pd.DataFrame:
        return pd.DataFrame(
            rng.normal(0.0, scale, (rows, len(columns))).astype(np.float32),
            index=index,
            columns=columns,
        )

    result = {name: panel() for name in expand_negative_residual_feature_dependencies(NEGATIVE_RESIDUAL_META_FEATURE_KEYS) if name not in NEGATIVE_RESIDUAL_META_FEATURE_KEYS}
    result["ret4h"] = panel(0.02)
    result["ret_resid_btc_4h"] = panel(0.015)
    result["corr_btc_24h"] = panel(0.25).clip(-1, 1)
    result["corr_eth_24h"] = panel(0.25).clip(-1, 1)
    result["oi_value_z_30d"] = panel(1.0)
    result["oi_value_1d_log_chg"] = panel(0.03)
    result["pct_assets_new_low_24h"] = panel(0.1).abs().clip(0, 1)
    result["mkt_pct_oi_chg_4h_rz_lt_minus1"] = panel(0.1).abs().clip(0, 1)
    return result


def test_negative_residual_features_are_meta_only() -> None:
    import extreme_price_movements.config as config

    keys = set(NEGATIVE_RESIDUAL_META_FEATURE_KEYS)
    assert keys <= set(config.MODEL_REGIME_CONTEXT_META_FEATURE_KEYS)
    assert keys <= set(config.CFG["META_CROSS_SECTIONAL_REGIME_KEYS"])
    assert config.CFG["NEGATIVE_RESIDUAL_META_FEATURE_KEYS"] == list(
        NEGATIVE_RESIDUAL_META_FEATURE_KEYS
    )
    assert not keys.intersection(config.MODEL_DIRECT_BASE_FEATURE_KEYS)
    assert config.CFG["MARKET_REGIME_CHANGE"]["feature_keys"] == list(
        MARKET_REGIME_CHANGE_FEATURE_KEYS
    )
    assert config.CFG["NEGATIVE_RESIDUAL_FEATURE_SCHEMA_VERSION"] == (
        NEGATIVE_RESIDUAL_FEATURE_SCHEMA_VERSION
    )
    assert all(
        config.CFG["feature_required_lookback_hours_by_feature"][key]
        == NEGATIVE_RESIDUAL_CAUSAL_WINDOW_HOURS
        for key in keys
    )


def test_feature_contract_is_stable_and_excludes_forbidden_sources() -> None:
    first = negative_residual_feature_contract()
    second = negative_residual_feature_contract()
    assert first == second
    assert first["contract_hash"].startswith("sha256:")
    assert set(first["forbidden_sources"]) >= {"future_path", "outcomes", "spread"}
    assert set(
        first["primitive_features"]
        + first["composite_features"]
        + first["temporal_mechanism_features"]
        + first["market_regime_change"]["features"]
    ) == set(NEGATIVE_RESIDUAL_META_FEATURE_KEYS)


def test_dependency_expansion_includes_oi_funding_and_ohlcv() -> None:
    expanded = expand_negative_residual_feature_dependencies(
        ["short_breakout_exhaustion"]
    )
    assert "ret4h" in expanded
    assert "mkt_median_oi_chg_4h_rz" in expanded
    assert "funding_1d_chg_ts_resid" in expanded
    assert "corr_eth_24h" in expanded


def test_benchmark_symbol_prefers_usd_settled_perpetual() -> None:
    columns = pd.Index(["BTC/USD:BTC", "BTC/USD:USD", "ETH/USD:ETH", "ETH/USD:USD"])
    assert _symbol(columns, "BTC") == "BTC/USD:USD"
    assert _symbol(columns, "ETH") == "ETH/USD:USD"


def test_generation_is_finite_after_warmup_and_causal() -> None:
    first = _panels()
    second = {name: value.copy() for name, value in first.items()}
    cutoff = 760
    for value in second.values():
        value.iloc[cutoff:] = value.iloc[cutoff:] * np.float32(100.0) + np.float32(50.0)

    generated_first = add_negative_residual_features(first, cfg={"feature_bars_per_hour": 1})
    generated_second = add_negative_residual_features(second, cfg={"feature_bars_per_hour": 1})
    assert generated_first == set(NEGATIVE_RESIDUAL_META_FEATURE_KEYS)
    for name in NEGATIVE_RESIDUAL_META_FEATURE_KEYS:
        left = first[name]
        right = second[name]
        np.testing.assert_allclose(
            left.iloc[:cutoff].to_numpy(),
            right.iloc[:cutoff].to_numpy(),
            equal_nan=True,
        )
        assert np.isfinite(left.iloc[24 * 7 :].to_numpy()).any(), name
        assert all(dtype == np.dtype("float32") for dtype in left.dtypes)
        if name in NEGATIVE_RESIDUAL_COMPOSITE_FEATURE_KEYS:
            assert np.nanmax(np.abs(left.to_numpy())) <= 6.0


def test_market_regime_change_operators_are_causal_and_wired() -> None:
    features = _panels(rows=1_000)
    generated = add_negative_residual_features(
        features,
        requested_feature_keys=MARKET_REGIME_CHANGE_FEATURE_KEYS,
        cfg={"feature_bars_per_hour": 1},
    )
    assert generated == set(MARKET_REGIME_CHANGE_FEATURE_KEYS)
    delta = features["mkt_regime_change__negative_breadth__delta_1h"].iloc[:, 0]
    level_features = _panels(rows=1_000)
    add_negative_residual_features(
        level_features,
        requested_feature_keys=["negative_breadth_pct"],
        cfg={"feature_bars_per_hour": 1},
    )
    level = level_features["negative_breadth_pct"].iloc[:, 0]
    np.testing.assert_allclose(
        delta.to_numpy(),
        level.diff().to_numpy(),
        equal_nan=True,
    )
    assert all(
        all(dtype == np.dtype("float32") for dtype in features[name].dtypes)
        for name in MARKET_REGIME_CHANGE_FEATURE_KEYS
    )


def test_temporal_mechanisms_are_registered_meta_context() -> None:
    import extreme_price_movements.config as config

    keys = set(NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS)
    assert keys <= set(NEGATIVE_RESIDUAL_META_FEATURE_KEYS)
    assert keys <= set(config.MODEL_REGIME_CONTEXT_META_FEATURE_KEYS)
    assert keys <= set(config.CFG["META_CROSS_SECTIONAL_REGIME_KEYS"])
    assert not keys.intersection(config.MODEL_DIRECT_BASE_FEATURE_KEYS)
    assert config.CFG["NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS"] == list(
        NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS
    )


def test_requested_subset_and_minimum_universe_coverage() -> None:
    features = _panels()
    generated = add_negative_residual_features(
        features,
        requested_feature_keys=["short_breakout_exhaustion"],
        cfg={"feature_bars_per_hour": 1},
    )
    assert generated == {"short_breakout_exhaustion"}
    assert "negative_breadth_pct" not in features

    sparse = _panels()
    for frame in sparse.values():
        frame.iloc[:, 1:] = np.nan
    generated = add_negative_residual_features(
        sparse,
        requested_feature_keys=["negative_breadth_pct", "breadth_dispersion"],
        cfg={"feature_bars_per_hour": 1},
    )
    assert generated == {"negative_breadth_pct", "breadth_dispersion"}
    assert sparse["negative_breadth_pct"].isna().all().all()
    assert sparse["breadth_dispersion"].isna().all().all()


def test_residual_state_composites_are_outcome_only_targets() -> None:
    ts = pd.date_range("2026-07-05", periods=4, freq="6h", tz="UTC")
    frame = pd.DataFrame(
        {
            "__ts__": ts,
            "side_name": ["short"] * 4,
            "archetype_policy_key": ["short_breakout"] * 4,
            "resid_event_timestamp_neutral_surprise": [0.5, 0.7, -0.2, 0.3],
            "resid_event_ev_timestamp_neutral_surprise": [-0.4, -0.8, -0.1, 0.2],
            "resid_event_daily_neutral_z": [0.6, 0.8, -0.2, 0.3],
            "resid_event_daily_ev_neutral_z": [-0.5, -0.9, -0.1, 0.2],
            "resid_event_persistence_strength": [1.0, 2.0, 0.2, 0.1],
            "resid_event_negative_large": [0, 0, 1, 0],
            "resid_event_top10_population": [1, 1, 1, 1],
            "resid_event_large_event_strength": [1.2, 1.5, 0.3, 0.1],
        }
    )
    result = add_residual_state_target_composites(frame)
    names = residual_state_target_feature_names()
    assert set(names).issubset(result.columns)
    assert not set(names).intersection(NEGATIVE_RESIDUAL_META_FEATURE_KEYS)
    assert result.loc[
        1, "resid_target_side_archetype_bullish_tape_adverse_ev_6h"
    ] > 0.0
    assert result.loc[
        1, "resid_target_side_archetype_timestamp_ev_sign_disagreement_6h"
    ] > 0.0
    assert result[names].dtypes.eq(np.dtype("float32")).all()
