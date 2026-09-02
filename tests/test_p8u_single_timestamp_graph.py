from __future__ import annotations

import json

import numpy as np
import pandas as pd

from extreme_price_movements.feature_transforms import CausalFeatureTransformer
from extreme_price_movements.inference.causal_feature_output_state import (
    CausalFeatureOutputState,
)
from extreme_price_movements import features as feature_engine
from extreme_price_movements import features_residual as residual_feature_engine
from extreme_price_movements.features import (
    _transform_close_fixed_ffd,
    add_regime_gates,
    compute_market_features,
)
from extreme_price_movements import fast_funcs as ff
from extreme_price_movements.features_oi import rolling_robust_zscore_by_symbol
from extreme_price_movements.inference.p8u_single_timestamp_graph import (
    P8UMarketRegimeSnapshotState,
    P8UFixedFfdRollingState,
    P8ULiquiditySafetySnapshotNode,
    P8URouterAtrIdentityState,
    P8UOneTimestampExecutor,
    P8UPriceMemoryCausalState,
    P8UReturnVolatilityState,
    P8UCrossAssetRouterState,
    P8URangeVolatilityState,
    P8UVolatilityOfVolatilityState,
    P8UHourOfDayRelativeVolumeState,
    P8USeasonalityStrengthState,
    P8URangePerVolumeState,
    P8UPriceRvRobustZState,
    P8ULiquidityPeerResidualState,
    P8UOrderbookDepthPortabilityState,
    P8URawOhlcRollingState,
    P8USingleTimestampCoverageError,
)
from extreme_price_movements.inference.orderbook_feature_state import OrderbookFeatureState
from extreme_price_movements.inference.live_zscore_state import RawRollingFeatureState


def _panel(rows: int = 220, symbols: int = 12) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(1729)
    index = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    columns = [f"S{value:02d}/USD:USD" for value in range(symbols)]
    returns = rng.normal(0.0, 0.006, size=(rows, symbols)).astype(np.float32)
    close = (100.0 * np.exp(np.cumsum(returns, axis=0))).astype(np.float32)
    high = (close * (1.0 + rng.uniform(0.0001, 0.009, size=close.shape))).astype(np.float32)
    low = (close * (1.0 - rng.uniform(0.0001, 0.009, size=close.shape))).astype(np.float32)
    volume = rng.lognormal(mean=5.0, sigma=0.5, size=close.shape).astype(np.float32)
    quote = (close * volume).astype(np.float32)
    # Exercise row-wise quote-volume fallback rather than a panel-wide branch.
    quote[25::17, 1] = np.nan
    return {
        "open": pd.DataFrame(np.vstack((close[0], close[:-1])), index=index, columns=columns),
        "close": pd.DataFrame(close, index=index, columns=columns),
        "high": pd.DataFrame(high, index=index, columns=columns),
        "low": pd.DataFrame(low, index=index, columns=columns),
        "volume": pd.DataFrame(volume, index=index, columns=columns),
        "quote_volume": pd.DataFrame(quote, index=index, columns=columns),
        "funding_rate": pd.DataFrame(
            rng.normal(0.0, 0.0001, size=close.shape).astype(np.float32),
            index=index,
            columns=columns,
        ),
        "mark_price": pd.DataFrame(
            (close * (1.0 + rng.normal(0.0, 0.0002, size=close.shape))).astype(np.float32),
            index=index,
            columns=columns,
        ),
        "orderbook_best_bid": pd.DataFrame(close * np.float32(0.9995), index=index, columns=columns),
        "orderbook_best_ask": pd.DataFrame(close * np.float32(1.0005), index=index, columns=columns),
        "orderbook_mid": pd.DataFrame(close, index=index, columns=columns),
        "orderbook_bid_qty_1": pd.DataFrame(
            rng.lognormal(mean=2.0, sigma=0.2, size=close.shape).astype(np.float32),
            index=index,
            columns=columns,
        ),
        "orderbook_ask_qty_1": pd.DataFrame(
            rng.lognormal(mean=2.0, sigma=0.2, size=close.shape).astype(np.float32),
            index=index,
            columns=columns,
        ),
        "orderbook_cum_bid_qty_l20": pd.DataFrame(
            rng.lognormal(mean=4.0, sigma=0.2, size=close.shape).astype(np.float32),
            index=index,
            columns=columns,
        ),
        "orderbook_cum_ask_qty_l20": pd.DataFrame(
            rng.lognormal(mean=4.0, sigma=0.2, size=close.shape).astype(np.float32),
            index=index,
            columns=columns,
        ),
        "orderbook_mean_trade_qty_1h": pd.DataFrame(
            rng.lognormal(mean=1.0, sigma=0.2, size=close.shape).astype(np.float32),
            index=index,
            columns=columns,
        ),
    }


def test_market_regime_snapshot_matches_canonical_batch_graph() -> None:
    panel = _panel()
    symbols = list(panel["close"].columns)
    batch = compute_market_features(panel, symbols)
    expected = add_regime_gates(batch, gate_vol_lookback_hours=24 * 7, gate_trend_thr=0.0)
    state = P8UMarketRegimeSnapshotState(symbols=symbols, market_basket=symbols)
    output: dict[str, list[np.ndarray]] = {name: [] for name in state.OUTPUTS}
    for timestamp in panel["close"].index:
        row = state.update(
            {name: frame.loc[timestamp].to_numpy(np.float32) for name, frame in panel.items()},
            timestamp=timestamp,
        )
        for name in state.OUTPUTS:
            output[name].append(row[name][0])
    for name in state.OUTPUTS:
        actual = np.asarray(output[name], dtype=np.float32)
        reference = expected[name].to_numpy(np.float32)
        # Output is intentionally broadcast to every frozen-universe symbol.
        np.testing.assert_allclose(
            actual,
            reference,
            rtol=2e-5,
            atol=2e-6,
            equal_nan=True,
            err_msg=name,
        )


def test_market_regime_bootstrap_matches_chronological_updates() -> None:
    panel = _panel(rows=40)
    symbols = list(panel["close"].columns)
    direct = P8UMarketRegimeSnapshotState(symbols=symbols, market_basket=symbols)
    sequential = P8UMarketRegimeSnapshotState(symbols=symbols, market_basket=symbols)
    bootstrapped = direct.bootstrap(panel)
    for timestamp in panel["close"].index:
        sequential.update(
            {name: frame.loc[timestamp].to_numpy(np.float32) for name, frame in panel.items()},
            timestamp=timestamp,
        )
    assert direct.last_timestamp == sequential.last_timestamp
    for name in direct.OUTPUTS:
        assert bootstrapped[name].shape == (len(panel["close"]), len(symbols))


def test_market_regime_snapshot_rejects_gap_and_restores_atomically(tmp_path) -> None:
    panel = _panel(rows=4)
    symbols = list(panel["close"].columns)
    state = P8UMarketRegimeSnapshotState(symbols=symbols, market_basket=symbols)
    index = panel["close"].index
    for timestamp in index[:3]:
        state.update(
            {name: frame.loc[timestamp].to_numpy(np.float32) for name, frame in panel.items()},
            timestamp=timestamp,
        )
    path = tmp_path / "market_state.npz"
    state.save(path)
    restored = P8UMarketRegimeSnapshotState.load(path)
    assert restored is not None
    assert restored.last_timestamp == state.last_timestamp
    with np.testing.assert_raises_regex(ValueError, "exactly one hour"):
        restored.update(
            {name: frame.loc[index[-1]].to_numpy(np.float32) for name, frame in panel.items()},
            timestamp=index[-1] + pd.Timedelta(hours=1),
        )


def test_raw_ohlc_range_state_matches_rolling_source_definition(tmp_path) -> None:
    panel = _panel(rows=220)
    symbols = list(panel["close"].columns)
    state = P8URawOhlcRollingState(symbols=symbols)
    actual = {name: [] for name in state.OUTPUTS}
    for timestamp in panel["close"].index:
        current = state.update(
            {name: panel[name].loc[timestamp].to_numpy(np.float32) for name in state.SOURCE_FIELDS},
            timestamp=timestamp,
        )
        for name in state.OUTPUTS:
            actual[name].append(current[name])
    high_24 = panel["high"].rolling(24, min_periods=1).max()
    low_24 = panel["low"].rolling(24, min_periods=1).min()
    expected = {
        "range_24h_pct": ((high_24 - low_24) / panel["close"]),
        "dist_rolling_7d_high": ((panel["high"].rolling(168, min_periods=1).max() - panel["close"]) / panel["close"]),
        "dist_prior_day_high": ((high_24.shift(1) - panel["close"]) / panel["close"]),
        "dist_prior_day_low": ((panel["close"] - low_24.shift(1)) / panel["close"]),
    }
    log_close = np.log(panel["close"].to_numpy(np.float32))
    for window in (10, 16, 24):
        ker = np.full_like(log_close, np.nan, dtype=np.float32)
        for row in range(window, len(log_close)):
            tail = log_close[row - window : row + 1]
            direction = np.abs(tail[-1] - tail[0])
            volatility = np.nansum(np.abs(np.diff(tail, axis=0)), axis=0)
            ker[row] = np.where(
                volatility > 1e-9,
                direction / volatility,
                np.where(direction == 0.0, 1.0, 0.0),
            )
        expected[f"ker_{window}"] = pd.DataFrame(ker, index=panel["close"].index, columns=symbols)
    for name, reference in expected.items():
        np.testing.assert_allclose(
            np.asarray(actual[name]), reference.to_numpy(np.float32), rtol=2e-5, atol=2e-6, equal_nan=True
        )
    path = tmp_path / "raw_ohlc.npz"
    state.save(path)
    restored = P8URawOhlcRollingState.load(path)
    assert restored is not None and restored.last_timestamp == state.last_timestamp
    with np.testing.assert_raises_regex(ValueError, "exactly one hour"):
        restored.update(
            {name: panel[name].loc[panel["close"].index[-1]].to_numpy(np.float32) for name in state.SOURCE_FIELDS},
            timestamp=panel["close"].index[-1] + pd.Timedelta(hours=2),
        )


def test_liquidity_safety_snapshot_matches_declared_wilder_atr_formula() -> None:
    node = P8ULiquiditySafetySnapshotNode(symbols=("A/USD:USD", "B/USD:USD"))
    actual = node.update(
        {
            "close": np.asarray([100.0, 200.0], dtype=np.float32),
            "mark_price": np.asarray([101.0, 199.0], dtype=np.float32),
        },
        raw_atr_pct=np.asarray([0.01, 0.002], dtype=np.float32),
    )["liq_stop_safety_short_atr"]
    expected = np.asarray(
        [
            ((1.0 / 3.0 - 0.005 - 0.003) - 0.01 - 3.0 * 0.01) / 0.01,
            ((1.0 / 3.0 - 0.005 - 0.003) - (-0.005) - 3.0 * 0.003) / 0.002,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, np.clip(expected, -100.0, 100.0), rtol=2e-6, atol=2e-6)


def test_router_atr_identity_state_matches_canonical_rank_primitive(tmp_path) -> None:
    symbols = ("A/USD:USD", "B/USD:USD")
    index = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    raw_atr = np.asarray(
        [[float(row + 1), float(12 - row)] for row in range(len(index))], dtype=np.float32
    )
    state = P8URouterAtrIdentityState(symbols=symbols)
    history = state.bootstrap(raw_atr, index=index)["asset_atr_level"]
    control = RawRollingFeatureState(
        op="rank_pct", name="p8u_asset_atr_level", symbols=symbols, window=24 * 60
    )
    expected = np.asarray(
        [control.update(raw_atr[row], timestamp=index[row]) for row in range(len(index))],
        dtype=np.float32,
    )
    np.testing.assert_allclose(history, expected, rtol=2e-6, atol=2e-6)
    state.save(tmp_path / "atr_identity")
    restored = P8URouterAtrIdentityState.load(tmp_path / "atr_identity")
    assert restored is not None and restored.last_timestamp == state.last_timestamp


def test_fixed_ffd_state_matches_canonical_fixed_width_transform(tmp_path) -> None:
    # d=.4 needs 1,458 source rows under the frozen threshold.  The test
    # deliberately crosses that boundary rather than comparing a warm-up-only
    # stream where both implementations are merely unavailable.
    panel = _panel(rows=1_500, symbols=3)
    symbols = list(panel["close"].columns)
    state = P8UFixedFfdRollingState(symbols=symbols)
    actual = {name: [] for name in state.OUTPUTS}
    for timestamp in panel["close"].index:
        current = state.update(
            {"close": panel["close"].loc[timestamp].to_numpy(np.float32)},
            timestamp=timestamp,
        )
        for name in state.OUTPUTS:
            actual[name].append(current[name])
    expected: dict[str, np.ndarray] = {}
    for d_value in state.D_VALUES:
        tag = f"{int(round(d_value * 10)):02d}"
        transformed = _transform_close_fixed_ffd(
            panel["close"], d=d_value, _label=f"test_d{tag}", already_logged=False
        )
        diff = transformed.diff(1)
        for window in state.RV_WINDOWS:
            expected[f"ffd_rv_{window}h_{tag}"] = ff.apply_to_frame(
                diff, ff._numba_rolling_std_nan_safe, window
            ).to_numpy(np.float32)
    for name in state.OUTPUTS:
        np.testing.assert_allclose(
            np.asarray(actual[name]), expected[name], rtol=2e-5, atol=2e-6, equal_nan=True,
            err_msg=name,
        )
    state.save(tmp_path / "fixed_ffd")
    restored = P8UFixedFfdRollingState.load(tmp_path / "fixed_ffd")
    assert restored is not None and restored.last_timestamp == state.last_timestamp


def test_return_volatility_state_matches_canonical_ffd_impulse_features(tmp_path) -> None:
    panel = _panel(rows=1_500, symbols=3)
    symbols = list(panel["close"].columns)
    ffd_state = P8UFixedFfdRollingState(symbols=symbols)
    state = P8UReturnVolatilityState(symbols=symbols)
    actual = {name: [] for name in state.OUTPUTS}
    for timestamp in panel["close"].index:
        ffd = ffd_state.update(
            {"close": panel["close"].loc[timestamp].to_numpy(np.float32)}, timestamp=timestamp
        )
        current = state.update(ffd, timestamp=timestamp)
        for name in state.OUTPUTS:
            actual[name].append(current[name])

    ffd06 = _transform_close_fixed_ffd(
        panel["close"], d=0.6, _label="test_return_d06", already_logged=False
    )
    ret = ffd06.diff(1).astype(np.float32)
    rv48 = ff.apply_to_frame(ret, ff._numba_rolling_std_nan_safe, 48)
    rv96 = ff.apply_to_frame(ret, ff._numba_rolling_std_nan_safe, 96)
    rv120 = ff.apply_to_frame(ret, ff._numba_rolling_std_nan_safe, 120)
    hr48 = ret.abs().shift(1).rolling(48, min_periods=1).median()
    raw = {
        "cvar_5pct": ff.numba_rolling_quantile(ret, 48, 0.05).fillna(0.0),
        "realized_volatility_24h": rv96,
        "rv_48h": np.log((rv48 + 1e-12) / (rv120 + 1e-12)),
        "upside_semivariance_8": ff.apply_to_frame(
            ret.clip(lower=0.0) ** 2, ff._numba_rolling_mean_nan_safe, 8
        ),
        "upside_semivariance_24": ff.apply_to_frame(
            ret.clip(lower=0.0) ** 2, ff._numba_rolling_mean_nan_safe, 24
        ),
        "t_be_proxy": 0.0035 / (hr48 + 1e-12),
        "t_pl_proxy": 0.0050 / (hr48 + 1e-12),
    }
    transformer = CausalFeatureTransformer(enable_cache=False)
    for name in state.OUTPUTS:
        expected = (
            transformer.transform(raw[name].to_numpy(np.float32), name=name)
            if name in (*state.TRANSFORMED_OUTPUTS, "t_be_proxy", "t_pl_proxy")
            else raw[name].to_numpy(np.float32)
        )
        np.testing.assert_allclose(
            np.asarray(actual[name]), expected, rtol=2e-5, atol=2e-6, equal_nan=True,
            err_msg=name,
        )
    state.save(tmp_path / "return_volatility")
    restored = P8UReturnVolatilityState.load(tmp_path / "return_volatility")
    assert restored is not None and restored.last_timestamp == state.last_timestamp


def test_cross_asset_router_state_matches_canonical_ffd_impulse_features(tmp_path) -> None:
    rng = np.random.default_rng(913)
    index = pd.date_range("2026-01-01", periods=1_500, freq="h", tz="UTC")
    columns = ("BTC/USD:USD", "ETH/USD:USD", "A/USD:USD", "B/USD:USD")
    close = pd.DataFrame(
        np.float32(100.0) * np.exp(np.cumsum(rng.normal(0.0, 0.004, (len(index), len(columns))), axis=0)),
        index=index,
        columns=columns,
        dtype=np.float32,
    )
    ffd06 = _transform_close_fixed_ffd(close, 0.6, 1e-5)
    ret = ffd06.diff(1).astype(np.float32)
    rv24 = ff.apply_to_frame(ret, ff._numba_rolling_std_nan_safe, 24)
    btc = ret["BTC/USD:USD"]
    eth = ret["ETH/USD:USD"]
    btc_var = btc.rolling(24, min_periods=8).var().replace(0.0, np.nan)
    eth_var = eth.rolling(24, min_periods=8).var().replace(0.0, np.nan)
    btc_cov, _ = feature_engine._rolling_cov_corr_with_series_frames(ret, btc, window=24, min_periods=8)
    eth_cov, _ = feature_engine._rolling_cov_corr_with_series_frames(ret, eth, window=24, min_periods=8)
    raw = {
        "beta_btc_24h": btc_cov.div(btc_var + 1e-12, axis=0).clip(-5.0, 5.0),
        "beta_eth_24h": eth_cov.div(eth_var + 1e-12, axis=0).clip(-5.0, 5.0),
        "rv_rel_universe": rv24.div(
            pd.Series(np.nanmedian(rv24.to_numpy(np.float32), axis=1), index=index) + 1e-12,
            axis=0,
        ),
    }
    state = P8UCrossAssetRouterState(symbols=columns)
    actual = state.bootstrap(
        {state.PARENT_INPUT: ret.to_numpy(np.float32)}, index=index
    )
    transformer = CausalFeatureTransformer(enable_cache=False)
    for name in state.OUTPUTS:
        expected = transformer.transform(raw[name].to_numpy(np.float32), name=name)
        np.testing.assert_allclose(
            actual[name], expected, rtol=2e-5, atol=2e-6, equal_nan=True, err_msg=name
        )
    state.save(tmp_path / "cross_asset_router")
    restored = P8UCrossAssetRouterState.load(tmp_path / "cross_asset_router")
    assert restored is not None and restored.last_timestamp == state.last_timestamp


def test_range_volatility_state_matches_canonical_feature_transform(tmp_path) -> None:
    panel = _panel(rows=1_000, symbols=3)
    high = panel["high"]
    low = panel["low"]
    range_ln = (
        ff.numba_ewma(np.log(high).astype(np.float32), 2.0 / 6.0, False)
        - ff.numba_ewma(np.log(low).astype(np.float32), 2.0 / 6.0, False)
    ).astype(np.float32)
    raw = ff.apply_to_frame(range_ln, ff._numba_rolling_std_nan_safe, 24).fillna(0.0)
    expected = CausalFeatureTransformer(enable_cache=False).transform(
        raw.to_numpy(np.float32), name="range_volatility"
    )
    state = P8URangeVolatilityState(symbols=tuple(high.columns))
    actual = state.bootstrap({"high": high, "low": low})["range_volatility"]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6, equal_nan=True)
    state.save(tmp_path / "range_volatility")
    restored = P8URangeVolatilityState.load(tmp_path / "range_volatility")
    assert restored is not None and restored.last_timestamp == state.last_timestamp


def test_orderbook_state_treats_moving_zero_volume_as_unavailable() -> None:
    symbols = ("A/USD:USD",)
    state = OrderbookFeatureState(symbols=symbols)
    base = {
        "best_bid": np.asarray([99.0], dtype=np.float32),
        "best_ask": np.asarray([101.0], dtype=np.float32),
        "mid": np.asarray([100.0], dtype=np.float32),
        "bid_qty_1": np.asarray([1.0], dtype=np.float32),
        "ask_qty_1": np.asarray([1.0], dtype=np.float32),
        "cum_bid_qty_l20": np.asarray([10.0], dtype=np.float32),
        "cum_ask_qty_l20": np.asarray([10.0], dtype=np.float32),
        "mean_trade_qty_1h": np.asarray([1.0], dtype=np.float32),
        "close": np.asarray([100.0], dtype=np.float32),
    }
    # A moving candle with zero reported volume cannot contribute a synthetic
    # zero to the causal 24h denominator.
    state.update(
        {**base, "open": np.asarray([99.0], dtype=np.float32), "volume": np.asarray([0.0], dtype=np.float32)},
        timestamp="2026-01-01T00:00:00Z",
    )
    assert int(state._qv_count[0]) == 0
    # A genuinely flat no-trade candle remains a valid zero observation.
    state.update(
        {**base, "open": np.asarray([100.0], dtype=np.float32), "volume": np.asarray([0.0], dtype=np.float32)},
        timestamp="2026-01-01T01:00:00Z",
    )
    assert int(state._qv_count[0]) == 1


def test_one_timestamp_executor_never_publishes_partial_contract(tmp_path) -> None:
    panel = _panel(rows=3)
    symbols = list(panel["close"].columns)
    executor = P8UOneTimestampExecutor(root=tmp_path, symbols=symbols, market_basket=symbols)
    timestamp = panel["close"].index[0]
    snapshot = {name: frame.loc[timestamp].to_numpy(np.float32) for name, frame in panel.items()}
    with np.testing.assert_raises(P8USingleTimestampCoverageError):
        executor.advance(
            timestamp=timestamp,
            snapshot=snapshot,
            required_features=("mkt_rv", "not_a_direct_node"),
        )
    assert not executor.ledger_path.exists()
    assert not (tmp_path / "commits").exists()

    output = executor.advance(
        timestamp=timestamp,
        snapshot=snapshot,
        required_features=("mkt_rv", "mkt_rv_pct", "prior_volatility"),
    )
    assert set(output) == {
        *P8UMarketRegimeSnapshotState.OUTPUTS,
        "prior_volatility",
        "asset_atr_level",
        "fund_rate",
        "hour_of_week_sin",
        "mark_perp_dislocation",
        "liq_stop_safety_short_atr",
        "range_24h_pct",
        "dist_rolling_7d_high",
        "dist_prior_day_high",
        "dist_prior_day_low",
        "ker_10",
        "ker_16",
        "ker_24",
        "ffd_rv_2h_04",
        "ffd_rv_6h_04",
        "ffd_rv_24h_04",
        "ffd_rv_2h_06",
        "ffd_rv_6h_06",
        "ffd_rv_24h_06",
        "cvar_5pct",
        "realized_volatility_24h",
        "rv_48h",
        "upside_semivariance_8",
        "upside_semivariance_24",
        "t_be_proxy",
        "t_pl_proxy",
        *P8UCrossAssetRouterState.OUTPUTS,
        *P8URangeVolatilityState.OUTPUTS,
        *P8UVolatilityOfVolatilityState.OUTPUTS,
        *P8UHourOfDayRelativeVolumeState.OUTPUTS,
        *P8USeasonalityStrengthState.OUTPUTS,
        *P8URangePerVolumeState.OUTPUTS,
        *P8UPriceRvRobustZState.OUTPUTS,
        *P8ULiquidityPeerResidualState.OUTPUTS,
        *P8UOneTimestampExecutor.ORDERBOOK_DEPTH_PORTABILITY_OUTPUTS,
    }
    assert executor.ledger_path.exists()
    receipt = next((tmp_path / "commits").rglob("receipt.json"))
    assert "pass_partial_direct_graph" in receipt.read_text()

    # The successor must load and advance both atomic node states, rather than
    # restarting price-memory history from the second source row.
    timestamp_2 = panel["close"].index[1]
    output_2 = executor.advance(
        timestamp=timestamp_2,
        snapshot={name: frame.loc[timestamp_2].to_numpy(np.float32) for name, frame in panel.items()},
        required_features=("mkt_rv", "prior_volatility"),
    )
    assert "prior_volatility" in output_2


def test_one_timestamp_executor_bootstrap_then_advances_saved_nodes(tmp_path) -> None:
    panel = _panel(rows=40)
    symbols = list(panel["close"].columns)
    executor = P8UOneTimestampExecutor(root=tmp_path, symbols=symbols, market_basket=symbols)
    latest = executor.bootstrap(panel)
    assert "prior_volatility" in latest
    ledger = json.loads(executor.ledger_path.read_text())
    assert ledger["active_commit"] == "commits/bootstrap"
    assert (tmp_path / "commits" / "bootstrap" / "state" / "price_memory" / "metadata.json").is_file()

    next_timestamp = panel["close"].index[-1] + pd.Timedelta(hours=1)
    next_snapshot = {
        name: frame.iloc[-1].to_numpy(np.float32, copy=True)
        for name, frame in panel.items()
    }
    output = executor.advance(
        timestamp=next_timestamp,
        snapshot=next_snapshot,
        required_features=("prior_volatility", "mkt_rv"),
    )
    assert set((tmp_path / "commits").iterdir()) == {
        tmp_path / "commits" / "bootstrap",
        tmp_path / "commits" / "20260102T160000Z",
    }
    assert "prior_volatility" in output
    assert "fund_rate" in output
    assert "hour_of_week_sin" in output
    assert "mark_perp_dislocation" in output
    assert "range_24h_pct" in output
    # The raw book recurrence advances and persists.  Its selected P8U final
    # transform is only promoted after a source-aligned reference-parity audit.
    assert (tmp_path / "commits" / "20260102T160000Z" / "state" / "orderbook_feature_state.npz").is_file()


def test_price_memory_causal_state_is_serializable_and_chronological(tmp_path) -> None:
    panel = _panel(rows=64)
    symbols = list(panel["close"].columns)
    direct = P8UPriceMemoryCausalState(
        symbols=symbols,
        transform_keys=("prior_volatility", "bars_to_resistance_daily_donchian"),
    )
    bootstrap = direct.bootstrap(panel)
    assert direct.last_timestamp == panel["close"].index[-1].isoformat()
    assert bootstrap["feature__prior_volatility"].shape == (64, len(symbols))
    state_root = tmp_path / "price_memory"
    direct.save(state_root)
    restored = P8UPriceMemoryCausalState.load(state_root)
    assert restored is not None
    assert restored.last_timestamp == direct.last_timestamp

    # A synthetic next bar checks append-state recovery without letting the
    # direct node reseed from raw history.
    timestamp = panel["close"].index[-1] + pd.Timedelta(hours=1)
    snapshot = {
        name: frame.iloc[-1].to_numpy(np.float32, copy=True)
        for name, frame in panel.items()
    }
    emitted = restored.update(snapshot, timestamp=timestamp)
    assert set(P8UPriceMemoryCausalState.RAW_OUTPUTS).issubset(
        {key.removeprefix("raw__") for key in emitted if key.startswith("raw__")}
    )
    assert "feature__prior_volatility" in emitted
    with np.testing.assert_raises_regex(ValueError, "exactly one hour"):
        restored.update(snapshot, timestamp=timestamp + pd.Timedelta(hours=2))


def test_vov_state_matches_canonical_bounded_operators_and_restores(tmp_path) -> None:
    """VOV must retain only its direct FFD-parent sufficient state."""

    rng = np.random.default_rng(531)
    index = pd.date_range("2026-01-01", periods=126, freq="h", tz="UTC")
    symbols = ("A/USD:USD", "B/USD:USD", "C/USD:USD")
    ret = rng.normal(0.0, 0.012, size=(len(index), len(symbols))).astype(np.float32)
    # Exercise the canonical mixed partial-window and unavailable-value paths.
    ret[:4, 1] = np.nan
    ret[69, 2] = np.nan
    state = P8UVolatilityOfVolatilityState(symbols=symbols)
    emitted = {name: [] for name in state.OUTPUTS}
    for row, stamp in enumerate(index):
        current = state.update(
            {state.PARENT_INPUT: ret[row]}, timestamp=stamp
        )
        for name in state.OUTPUTS:
            emitted[name].append(current[name])

    vov_fast = ff.numba_rolling_std(ret, 20).astype(np.float32)
    expected_iqr = (
        ff.numba_rolling_quantile(vov_fast, 20, 0.75)
        - ff.numba_rolling_quantile(vov_fast, 20, 0.25)
    ).astype(np.float32)
    expected_mad = ff.numba_rolling_mad(
        pd.DataFrame(vov_fast, index=index, columns=symbols), 60
    ).to_numpy(np.float32)
    clip = np.float32(CausalFeatureTransformer(enable_cache=False).sigma_k)
    expected = {
        "vov_iqr_20": np.where(
            np.isfinite(expected_iqr), np.clip(expected_iqr, -clip, clip), np.nan
        ).astype(np.float32),
        "vov_mad_60": np.where(
            np.isfinite(expected_mad), np.clip(expected_mad, -clip, clip), np.nan
        ).astype(np.float32),
    }
    for name in state.OUTPUTS:
        np.testing.assert_allclose(
            np.asarray(emitted[name], dtype=np.float32),
            expected[name],
            rtol=2e-6,
            atol=2e-6,
            equal_nan=True,
            err_msg=name,
        )

    state.save(tmp_path / "vov")
    restored = P8UVolatilityOfVolatilityState.load(tmp_path / "vov")
    assert restored is not None and restored.last_timestamp == state.last_timestamp
    next_value = rng.normal(0.0, 0.012, size=len(symbols)).astype(np.float32)
    direct = restored.update(
        {restored.PARENT_INPUT: next_value}, timestamp=index[-1] + pd.Timedelta(hours=1)
    )
    full = np.vstack((ret, next_value[None, :]))
    full_fast = ff.numba_rolling_std(full, 20).astype(np.float32)
    full_iqr = (
        ff.numba_rolling_quantile(full_fast, 20, 0.75)
        - ff.numba_rolling_quantile(full_fast, 20, 0.25)
    ).astype(np.float32)[-1]
    full_mad = ff.numba_rolling_mad(
        pd.DataFrame(
            full_fast,
            index=index.append(pd.DatetimeIndex([index[-1] + pd.Timedelta(hours=1)])),
            columns=symbols,
        ),
        60,
    ).to_numpy(np.float32)[-1]
    np.testing.assert_allclose(
        direct["vov_iqr_20"],
        np.where(np.isfinite(full_iqr), np.clip(full_iqr, -clip, clip), np.nan),
        rtol=2e-6,
        atol=2e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        direct["vov_mad_60"],
        np.where(np.isfinite(full_mad), np.clip(full_mad, -clip, clip), np.nan),
        rtol=2e-6,
        atol=2e-6,
        equal_nan=True,
    )


def test_hour_of_day_relative_volume_state_matches_canonical_grouped_mean_and_restores(tmp_path) -> None:
    """The 336-observation HOD denominator must not be approximated by hours."""

    rng = np.random.default_rng(711)
    index = pd.date_range("2026-01-01", periods=820, freq="h", tz="UTC")
    symbols = ("A/USD:USD", "B/USD:USD", "C/USD:USD")
    volume = rng.lognormal(mean=5.0, sigma=0.7, size=(len(index), len(symbols))).astype(np.float32)
    close = rng.lognormal(mean=4.0, sigma=0.1, size=(len(index), len(symbols))).astype(np.float32)
    opening = np.vstack((close[0], close[:-1])).astype(np.float32)
    # A missing value and a moving-candle zero both remain unavailable; a
    # flat zero is a genuine no-trade input to the canonical transform.
    volume[93, 1] = np.nan
    volume[177, 2] = np.float32(0.0)
    volume[211, 0] = np.float32(0.0)
    opening[211, 0] = close[211, 0]
    state = P8UHourOfDayRelativeVolumeState(symbols=symbols)
    emitted = []
    for row, stamp in enumerate(index):
        emitted.append(
            state.update(
                {"open": opening[row], "close": close[row], "volume": volume[row]}, timestamp=stamp
            )["rvol_hod_base"]
        )
    cleaned = feature_engine._backfill_short_volume_gaps(
        pd.DataFrame(volume, index=index, columns=symbols),
        pd.DataFrame(opening, index=index, columns=symbols),
        pd.DataFrame(close, index=index, columns=symbols),
    )
    transformed_volume = ff.numba_ewma(np.log1p(cleaned), 2.0 / 6.0, False)
    grouped = ff.numba_grouped_rolling_mean(
        transformed_volume,
        pd.Series(index.hour, index=index),
        14 * 24,
    ).to_numpy(np.float32)
    raw = (transformed_volume.to_numpy(np.float32) / (grouped + np.float32(1e-12))).astype(np.float32)
    expected = CausalFeatureTransformer(enable_cache=False).transform(
        raw.copy(), name="rvol_hod_base"
    )
    np.testing.assert_allclose(
        np.asarray(emitted, dtype=np.float32),
        expected,
        rtol=2e-5,
        atol=2e-6,
        equal_nan=True,
    )

    state.save(tmp_path / "rvol")
    restored = P8UHourOfDayRelativeVolumeState.load(tmp_path / "rvol")
    assert restored is not None and restored.last_timestamp == state.last_timestamp
    next_value = rng.lognormal(mean=5.0, sigma=0.7, size=len(symbols)).astype(np.float32)
    next_close = rng.lognormal(mean=4.0, sigma=0.1, size=len(symbols)).astype(np.float32)
    direct = restored.update(
        {"open": close[-1], "close": next_close, "volume": next_value},
        timestamp=index[-1] + pd.Timedelta(hours=1),
    )["rvol_hod_base"]
    extension_index = index.append(pd.DatetimeIndex([index[-1] + pd.Timedelta(hours=1)]))
    full_volume = np.vstack((volume, next_value[None, :])).astype(np.float32)
    full_open = np.vstack((opening, close[-1][None, :])).astype(np.float32)
    full_close = np.vstack((close, next_close[None, :])).astype(np.float32)
    full_clean = feature_engine._backfill_short_volume_gaps(
        pd.DataFrame(full_volume, index=extension_index, columns=symbols),
        pd.DataFrame(full_open, index=extension_index, columns=symbols),
        pd.DataFrame(full_close, index=extension_index, columns=symbols),
    )
    full_transformed = ff.numba_ewma(np.log1p(full_clean), 2.0 / 6.0, False)
    full_grouped = ff.numba_grouped_rolling_mean(
        full_transformed,
        pd.Series(extension_index.hour, index=extension_index),
        14 * 24,
    ).to_numpy(np.float32)
    full_raw = (
        full_transformed.to_numpy(np.float32) / (full_grouped + np.float32(1e-12))
    ).astype(np.float32)
    full_expected = CausalFeatureTransformer(enable_cache=False).transform(
        full_raw, name="rvol_hod_base"
    )[-1]
    np.testing.assert_allclose(direct, full_expected, rtol=2e-5, atol=2e-6, equal_nan=True)


def test_seasonality_state_matches_canonical_ret1h_mean_and_restores(tmp_path) -> None:
    rng = np.random.default_rng(251)
    index = pd.date_range("2026-01-01", periods=96, freq="h", tz="UTC")
    symbols = ("A/USD:USD", "B/USD:USD")
    ret = rng.normal(0.0, 0.01, size=(len(index), len(symbols))).astype(np.float32)
    ret[:2, 1] = np.nan
    state = P8USeasonalityStrengthState(symbols=symbols)
    emitted = np.asarray(
        [
            state.update({state.PARENT_INPUT: ret[row]}, timestamp=stamp)["seasonality_strength"]
            for row, stamp in enumerate(index)
        ],
        dtype=np.float32,
    )
    expected = np.abs(
        ret - ff.numba_rolling_mean(ret, 24).astype(np.float32)
    ).astype(np.float32)
    clip = np.float32(CausalFeatureTransformer(enable_cache=False).sigma_k)
    expected = np.where(
        np.isfinite(expected), np.clip(expected, -clip, clip), np.nan
    ).astype(np.float32)
    np.testing.assert_allclose(emitted, expected, rtol=2e-6, atol=2e-6, equal_nan=True)

    state.save(tmp_path / "seasonality")
    restored = P8USeasonalityStrengthState.load(tmp_path / "seasonality")
    assert restored is not None and restored.last_timestamp == state.last_timestamp
    extension = rng.normal(0.0, 0.01, size=len(symbols)).astype(np.float32)
    direct = restored.update(
        {restored.PARENT_INPUT: extension}, timestamp=index[-1] + pd.Timedelta(hours=1)
    )["seasonality_strength"]
    full = np.vstack((ret, extension[None, :]))
    full_expected = np.abs(
        full - ff.numba_rolling_mean(full, 24).astype(np.float32)
    ).astype(np.float32)[-1]
    full_expected = np.where(
        np.isfinite(full_expected), np.clip(full_expected, -clip, clip), np.nan
    ).astype(np.float32)
    np.testing.assert_allclose(direct, full_expected, rtol=2e-6, atol=2e-6, equal_nan=True)


def test_range_per_volume_state_matches_canonical_parents_and_restores(tmp_path) -> None:
    panel = _panel(rows=800, symbols=3)
    symbols = list(panel["close"].columns)
    index = panel["close"].index
    # Exercise row-local quote-notional fallback without future repair.
    panel["quote_volume"].iloc[427, 1] = np.nan
    state = P8URangePerVolumeState(symbols=symbols)
    emitted = np.asarray(
        [
            state.update(
                {name: panel[name].loc[stamp].to_numpy(np.float32) for name in state.SOURCE_FIELDS},
                timestamp=stamp,
            )["range_per_volume"]
            for stamp in index
        ],
        dtype=np.float32,
    )
    opening = panel["open"].astype(np.float32)
    closing = panel["close"].astype(np.float32)
    clean_volume = feature_engine._backfill_short_volume_gaps(
        panel["volume"].astype(np.float32), opening, closing
    )
    quote_source = panel["quote_volume"].astype(np.float32).where(
        lambda frame: frame > 0.0
    )
    fallback = (closing * clean_volume).where(lambda frame: frame > 0.0)
    quote = quote_source.where(quote_source.notna(), fallback).astype(np.float32)
    high = ff.numba_ewma(np.log(panel["high"]), 2.0 / 6.0, False)
    low = ff.numba_ewma(np.log(panel["low"]), 2.0 / 6.0, False)
    denominator = np.log1p(quote.clip(lower=0.0)).replace(0.0, np.nan)
    raw = ((high - low).abs() / (denominator + 1e-9)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0).astype(np.float32)
    expected = CausalFeatureTransformer(enable_cache=False).transform(
        raw.to_numpy(np.float32), name="range_per_volume"
    )
    np.testing.assert_allclose(emitted, expected, rtol=2e-5, atol=2e-6, equal_nan=True)

    state.save(tmp_path / "range_per_volume")
    restored = P8URangePerVolumeState.load(tmp_path / "range_per_volume")
    assert restored is not None and restored.last_timestamp == state.last_timestamp
    timestamp = index[-1] + pd.Timedelta(hours=1)
    snapshot = {name: panel[name].iloc[-1].to_numpy(np.float32, copy=True) for name in state.SOURCE_FIELDS}
    direct = restored.update(snapshot, timestamp=timestamp)["range_per_volume"]
    full = {name: pd.concat((panel[name], panel[name].iloc[[-1]].set_axis([timestamp]))) for name in state.SOURCE_FIELDS}
    full_clean = feature_engine._backfill_short_volume_gaps(full["volume"], full["open"], full["close"])
    full_quote_source = full["quote_volume"].where(lambda frame: frame > 0.0)
    full_fallback = (full["close"] * full_clean).where(lambda frame: frame > 0.0)
    full_quote = full_quote_source.where(full_quote_source.notna(), full_fallback).astype(np.float32)
    full_h = ff.numba_ewma(np.log(full["high"]), 2.0 / 6.0, False)
    full_l = ff.numba_ewma(np.log(full["low"]), 2.0 / 6.0, False)
    full_denom = np.log1p(full_quote.clip(lower=0.0)).replace(0.0, np.nan)
    full_raw = ((full_h - full_l).abs() / (full_denom + 1e-9)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0).astype(np.float32)
    full_expected = CausalFeatureTransformer(enable_cache=False).transform(
        full_raw.to_numpy(np.float32), name="range_per_volume"
    )[-1]
    np.testing.assert_allclose(direct, full_expected, rtol=2e-5, atol=2e-6, equal_nan=True)


def test_price_rv_robust_z_state_matches_canonical_price_path_and_restores(tmp_path) -> None:
    """The selected expensive 7/15-day price-RV fields need no panel replay."""

    panel = _panel(rows=1_120, symbols=3)
    index = panel["close"].index
    symbols = list(panel["close"].columns)
    # Exercise the exact positive-price ffill boundary, including an expired
    # 24-hour gap.  A valid close resumes the canonical price path causally.
    panel["close"].iloc[143:151, 1] = np.nan
    panel["close"].iloc[420:446, 2] = np.float32(0.0)
    state = P8UPriceRvRobustZState(symbols=symbols)
    emitted_rows = {name: [] for name in state.OUTPUTS}
    for row, stamp in enumerate(index):
        current = state.update(
            {"close": panel["close"].iloc[row].to_numpy(np.float32, copy=False)},
            timestamp=stamp,
        )
        for name in state.OUTPUTS:
            emitted_rows[name].append(current[name])
    emitted = {
        name: np.asarray(rows, dtype=np.float32)
        for name, rows in emitted_rows.items()
    }
    price = (
        panel["close"].replace([np.inf, -np.inf], np.nan).where(lambda frame: frame > 0.0)
        .ffill(limit=24)
        .astype(np.float32)
    )
    ret = np.log(price.clip(lower=1e-12)).astype(np.float32).diff(1).astype(np.float32)
    for name, window in state._SPECS:
        rv = ret.rolling(window, min_periods=window // 6).std(ddof=0)
        expected = rolling_robust_zscore_by_symbol(
            np.log(rv.clip(lower=1e-12)), 24 * 30, min_periods=24 * 7
        ).clip(-10, 10).to_numpy(np.float32)
        np.testing.assert_allclose(
            emitted[name], expected, rtol=3e-5, atol=3e-6, equal_nan=True, err_msg=name
        )

    state.save(tmp_path / "price_rv")
    restored = P8UPriceRvRobustZState.load(tmp_path / "price_rv")
    assert restored is not None and restored.last_timestamp == state.last_timestamp
    timestamp = index[-1] + pd.Timedelta(hours=1)
    extension = panel["close"].iloc[-1].to_numpy(np.float32, copy=True)
    direct = restored.update({"close": extension}, timestamp=timestamp)
    full_price = pd.concat(
        (panel["close"], panel["close"].iloc[[-1]].set_axis([timestamp]))
    )
    full_clean = (
        full_price.replace([np.inf, -np.inf], np.nan).where(lambda frame: frame > 0.0)
        .ffill(limit=24)
        .astype(np.float32)
    )
    full_ret = np.log(full_clean.clip(lower=1e-12)).astype(np.float32).diff(1).astype(np.float32)
    for name, window in state._SPECS:
        full_rv = full_ret.rolling(window, min_periods=window // 6).std(ddof=0)
        expected = rolling_robust_zscore_by_symbol(
            np.log(full_rv.clip(lower=1e-12)), 24 * 30, min_periods=24 * 7
        ).clip(-10, 10).iloc[-1].to_numpy(np.float32)
        np.testing.assert_allclose(direct[name], expected, rtol=3e-5, atol=3e-6, equal_nan=True)


def test_liquidity_peer_residual_state_matches_canonical_path_and_restores(tmp_path) -> None:
    """The 720h lagged liquidity baseline must never require a panel replay."""

    panel = _panel(rows=860, symbols=7)
    index = panel["close"].index
    symbols = list(panel["close"].columns)
    # Exercise both retained no-trade zeros and unavailable moving-zero rows.
    panel["volume"].iloc[341, 1] = 0.0
    panel["open"].iloc[341, 1] = panel["close"].iloc[341, 1]
    panel["volume"].iloc[452, 2] = 0.0
    panel["open"].iloc[452, 2] = panel["close"].iloc[452, 2] * np.float32(1.01)
    state = P8ULiquidityPeerResidualState(symbols=symbols)
    emitted = np.asarray(
        [
            state.update(
                {name: panel[name].loc[stamp].to_numpy(np.float32, copy=False) for name in state.SOURCE_FIELDS},
                timestamp=stamp,
            )["liquidity_ratio_peer_resid"]
            for stamp in index
        ],
        dtype=np.float32,
    )
    clean = feature_engine._backfill_short_volume_gaps(
        panel["volume"], panel["open"], panel["close"]
    )
    transformed = ff.numba_ewma(np.log1p(clean).astype(np.float32), 2.0 / 6.0, False)
    baseline = transformed.rolling(24 * 30).mean().shift(1)
    # Pandas preserves float64 division here because of the scalar epsilon;
    # the peer median/MAD sees that precision before its final float32 cast.
    ratio = transformed / (baseline + 1e-12)
    expected = residual_feature_engine._peer_resid(ratio).to_numpy(np.float32)
    np.testing.assert_allclose(emitted, expected, rtol=3e-5, atol=3e-6, equal_nan=True)

    state.save(tmp_path / "liquidity_peer")
    restored = P8ULiquidityPeerResidualState.load(tmp_path / "liquidity_peer")
    assert restored is not None and restored.last_timestamp == state.last_timestamp
    timestamp = index[-1] + pd.Timedelta(hours=1)
    snapshot = {name: panel[name].iloc[-1].to_numpy(np.float32, copy=True) for name in state.SOURCE_FIELDS}
    direct = restored.update(snapshot, timestamp=timestamp)["liquidity_ratio_peer_resid"]
    full = {
        name: pd.concat((panel[name], panel[name].iloc[[-1]].set_axis([timestamp])))
        for name in state.SOURCE_FIELDS
    }
    full_clean = feature_engine._backfill_short_volume_gaps(full["volume"], full["open"], full["close"])
    full_transformed = ff.numba_ewma(np.log1p(full_clean).astype(np.float32), 2.0 / 6.0, False)
    full_ratio = full_transformed / (full_transformed.rolling(24 * 30).mean().shift(1) + 1e-12)
    expected_next = residual_feature_engine._peer_resid(full_ratio).iloc[-1].to_numpy(np.float32)
    np.testing.assert_allclose(direct, expected_next, rtol=3e-5, atol=3e-6, equal_nan=True)


def test_orderbook_depth_portability_matches_canonical_p8u_path_and_restores(tmp_path) -> None:
    """Depth uses raw close×volume, then the exact causal portability repair."""

    panel = _panel(rows=810, symbols=3)
    index = panel["close"].index
    symbols = list(panel["close"].columns)
    raw_state = OrderbookFeatureState(symbols=symbols, quote_volume_mode="close_volume_raw")
    raw = np.asarray(
        [
            raw_state.update(
                {
                    "best_bid": panel["orderbook_best_bid"].iloc[row].to_numpy(np.float32, copy=False),
                    "best_ask": panel["orderbook_best_ask"].iloc[row].to_numpy(np.float32, copy=False),
                    "mid": panel["orderbook_mid"].iloc[row].to_numpy(np.float32, copy=False),
                    "bid_qty_1": panel["orderbook_bid_qty_1"].iloc[row].to_numpy(np.float32, copy=False),
                    "ask_qty_1": panel["orderbook_ask_qty_1"].iloc[row].to_numpy(np.float32, copy=False),
                    "cum_bid_qty_l20": panel["orderbook_cum_bid_qty_l20"].iloc[row].to_numpy(np.float32, copy=False),
                    "cum_ask_qty_l20": panel["orderbook_cum_ask_qty_l20"].iloc[row].to_numpy(np.float32, copy=False),
                    "mean_trade_qty_1h": panel["orderbook_mean_trade_qty_1h"].iloc[row].to_numpy(np.float32, copy=False),
                    "open": panel["open"].iloc[row].to_numpy(np.float32, copy=False),
                    "close": panel["close"].iloc[row].to_numpy(np.float32, copy=False),
                    "volume": panel["volume"].iloc[row].to_numpy(np.float32, copy=False),
                },
                timestamp=stamp,
            )["ob_depth_l20_to_qv_24h"]
            for row, stamp in enumerate(index)
        ],
        dtype=np.float32,
    )
    state = P8UOrderbookDepthPortabilityState(symbols=symbols)
    emitted = np.asarray(
        [state.update(raw[row], timestamp=stamp)["ob_depth_l20_to_qv_24h"] for row, stamp in enumerate(index)],
        dtype=np.float32,
    )

    qv = (panel["close"] * panel["volume"]).rolling(24, min_periods=6).sum().shift(1)
    delayed_mid = panel["orderbook_mid"].ffill().shift(1)
    delayed_bid = panel["orderbook_cum_bid_qty_l20"].ffill().shift(1)
    delayed_ask = panel["orderbook_cum_ask_qty_l20"].ffill().shift(1)
    canonical_raw = ((delayed_mid * (delayed_bid + delayed_ask)) / (qv + 1e-12)).clip(0.0, 100.0)
    log_raw = np.log1p(canonical_raw.where(canonical_raw >= 0.0)).astype(np.float32)
    raw_z = ff.numba_rolling_robust_zscore(log_raw.to_numpy(np.float32), 24 * 30)
    support = log_raw.notna().rolling(24 * 30).sum()
    dispersion = pd.DataFrame(
        ff.numba_rolling_std(log_raw.to_numpy(np.float32), 24 * 30),
        index=index,
        columns=symbols,
    )
    expected = pd.DataFrame(raw_z, index=index, columns=symbols).where(
        log_raw.notna()
        & (support >= 180)
        & (support >= 720)
        & np.isfinite(dispersion)
        & (dispersion > 1e-8)
    ).clip(-8.0, 8.0).to_numpy(np.float32)
    np.testing.assert_allclose(emitted, expected, rtol=3e-5, atol=3e-6, equal_nan=True)

    state.save(tmp_path / "orderbook_depth_portability")
    restored = P8UOrderbookDepthPortabilityState.load(tmp_path / "orderbook_depth_portability")
    assert restored is not None and restored.last_timestamp == state.last_timestamp
    extension = raw[-1].copy()
    direct = restored.update(extension, timestamp=index[-1] + pd.Timedelta(hours=1))
    full_raw = np.vstack((raw, extension[None, :]))
    full_log = np.log1p(full_raw).astype(np.float32)
    full_z = ff.numba_rolling_robust_zscore(full_log, 24 * 30)[-1]
    full_support = np.isfinite(full_log[-720:]).sum(axis=0)
    full_dispersion = ff.numba_rolling_std(full_log, 24 * 30)[-1]
    expected_next = np.where(
        np.isfinite(full_log[-1])
        & (full_support >= 180)
        & (full_support >= 720)
        & np.isfinite(full_dispersion)
        & (full_dispersion > 1e-8),
        np.clip(full_z, -8.0, 8.0),
        np.nan,
    ).astype(np.float32)
    np.testing.assert_allclose(direct["ob_depth_l20_to_qv_24h"], expected_next, rtol=3e-5, atol=3e-6, equal_nan=True)


def test_direct_causal_output_bootstrap_preserves_full_batch_anchor() -> None:
    """The first append after bootstrap must equal a full batch replay."""

    rng = np.random.default_rng(91)
    index = pd.date_range("2026-01-01", periods=800, freq="h", tz="UTC")
    symbols = ("A/USD:USD", "B/USD:USD", "C/USD:USD")
    history = rng.lognormal(mean=-4.0, sigma=0.5, size=(len(index), len(symbols))).astype(np.float32)
    state = CausalFeatureOutputState(feature_keys=("prior_volatility",), symbols=symbols)
    emitted = state.bootstrap({"prior_volatility": history}, index=index)["prior_volatility"]
    reference = CausalFeatureTransformer(enable_cache=False).transform(
        history.copy(), name="prior_volatility"
    )
    np.testing.assert_allclose(emitted, reference, rtol=2e-5, atol=2e-6, equal_nan=True)

    extension = rng.lognormal(mean=-4.0, sigma=0.5, size=len(symbols)).astype(np.float32)
    direct = state.update({"prior_volatility": extension}, timestamp=index[-1] + pd.Timedelta(hours=1))["prior_volatility"]
    full = np.vstack((history, extension[None, :]))
    expected = CausalFeatureTransformer(enable_cache=False).transform(
        full, name="prior_volatility"
    )[-1]
    np.testing.assert_allclose(direct, expected, rtol=2e-5, atol=2e-6, equal_nan=True)
