import numpy as np
import pandas as pd

import extreme_price_movements.fast_funcs as ff
from extreme_price_movements.data_store import _write_feature_delta_part, load_features_selected
from extreme_price_movements.inference import feature_generator as fg


def _reference_selector_features(panel, symbols):
    close = panel["close"].loc[:, symbols].astype(np.float32)
    high = panel["high"].loc[:, symbols].astype(np.float32)
    low = panel["low"].loc[:, symbols].astype(np.float32)
    volume = panel["volume"].loc[:, symbols].astype(np.float32)
    feats = {}

    ret24h = close / close.shift(24) - 1.0
    ret12h = close / close.shift(12) - 1.0
    ret6h = close / close.shift(6) - 1.0
    ret1h = close / close.shift(1) - 1.0
    feats["ret24h"] = ret24h.astype(np.float32)
    feats["ret12h"] = ret12h.astype(np.float32)
    feats["ret6h"] = ret6h.astype(np.float32)
    feats["ret1h"] = ret1h.astype(np.float32)

    h_12 = fg._mask_rolling_min_periods(ff.numba_rolling_max(high, 12), high, 12, 12)
    l_12 = fg._mask_rolling_min_periods(ff.numba_rolling_min(low, 12), low, 12, 12)
    feats["range_12h_pct"] = ((h_12 - l_12) / (close + 1e-12)).astype(np.float32)

    h_24 = ff.numba_rolling_max(high, 24)
    l_24 = ff.numba_rolling_min(low, 24)
    feats["range_24h_pct"] = ((h_24 - l_24) / (close + 1e-12)).astype(np.float32)
    feats["dist_prior_day_low"] = (
        ((close - l_24.shift(1)) / (close + 1e-12)).fillna(0.0).astype(np.float32)
    )

    prev_close = close.shift(1)
    tr = np.maximum(
        np.maximum((high - low).abs(), (high - prev_close).abs()),
        (low - prev_close).abs(),
    )
    atr = ff.numba_rolling_mean(tr.astype(np.float32), 14).replace(0.0, np.nan)
    ema_fast = close.ewm(span=10, adjust=False, min_periods=1).mean()
    feats["dist_ema_fast"] = (
        ((close - ema_fast) / (atr + 1e-12))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )

    delta = close.diff()
    gain = ff.numba_rolling_mean(delta.clip(lower=0.0).astype(np.float32), 14)
    loss = ff.numba_rolling_mean((-delta.clip(upper=0.0)).astype(np.float32), 14)
    rsi = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-12)))
    feats["rsi_slope"] = rsi.diff(3).fillna(0.0).astype(np.float32)

    vwap_48 = ff.numba_rolling_vwap(close, volume, 48)
    session_stdev_48 = ff.numba_rolling_std(close, 48)
    feats["loc_vwap_dev_z_48"] = (
        ((close - vwap_48) / (np.maximum(session_stdev_48, atr * 0.5) + 1e-12))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )

    rv_24h = fg._mask_rolling_min_periods(
        ff.numba_rolling_std(ret1h.astype(np.float32), 24), ret1h, 24, 24
    )
    rv_24h_mean = fg._mask_rolling_min_periods(
        ff.numba_rolling_mean(rv_24h.astype(np.float32), 24 * 90),
        rv_24h,
        24 * 90,
        100,
    )
    rv_24h_std = fg._mask_rolling_min_periods(
        ff.numba_rolling_std(rv_24h.astype(np.float32), 24 * 90),
        rv_24h,
        24 * 90,
        100,
    )
    feats["volatility_zscore"] = (
        (rv_24h - rv_24h_mean) / (rv_24h_std + 1e-12)
    ).astype(np.float32)

    sum_abs_ret = fg._mask_rolling_min_periods(
        ff.numba_rolling_sum(ret1h.abs().astype(np.float32), 24), ret1h, 24, 24
    )
    high_low_range = (
        fg._mask_rolling_min_periods(ff.numba_rolling_max(high, 24), high, 24, 24)
        - fg._mask_rolling_min_periods(ff.numba_rolling_min(low, 24), low, 24, 24)
    )
    chop_score = sum_abs_ret / (np.log(high_low_range + 1e-12) + 1e-12)
    feats["chop_score"] = (1 - np.clip(chop_score / 50, 0, 1)).astype(np.float32)
    feats["mkt_rv_24h"] = rv_24h.mean(axis=1).astype(np.float32)
    return feats


def test_compute_selector_features_array_fastpath_matches_pandas_reference():
    idx = pd.date_range("2026-01-01", periods=180, freq="h", tz="UTC")
    symbols = ["AAA/USD:USD", "BBB/USD:USD", "CCC/USD:USD"]
    base = np.linspace(10.0, 15.0, len(idx), dtype=np.float32)[:, None]
    offsets = np.array([0.0, 2.0, 5.0], dtype=np.float32)[None, :]
    wave = np.sin(np.arange(len(idx), dtype=np.float32)[:, None] / 7.0) * 0.2
    close = base + offsets + wave
    high = close + 0.2
    low = close - 0.15
    volume = 1000.0 + np.arange(len(idx), dtype=np.float32)[:, None] * 3.0
    volume = volume + np.array([10.0, 20.0, 30.0], dtype=np.float32)[None, :]
    panel = {
        "close": pd.DataFrame(close, index=idx, columns=symbols),
        "high": pd.DataFrame(high, index=idx, columns=symbols),
        "low": pd.DataFrame(low, index=idx, columns=symbols),
        "volume": pd.DataFrame(volume, index=idx, columns=symbols),
    }

    actual = fg.compute_selector_features(panel, symbols)
    expected = _reference_selector_features(panel, symbols)

    assert set(actual) == set(expected)
    for key, exp in expected.items():
        act = actual[key]
        np.testing.assert_allclose(
            np.asarray(act, dtype=np.float32),
            np.asarray(exp, dtype=np.float32),
            rtol=1e-4,
            atol=1e-4,
            equal_nan=True,
            err_msg=key,
        )


def test_latest_feature_matrix_uses_lazy_latest_values_without_materializing():
    idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")

    class Lazy:
        def keys(self):
            return ["ret24h"]

        def has_raw_key(self, key):
            return key == "ret24h"

        def latest_values_at(self, key, symbols, ts, *, stale_sensitive=False):
            assert key == "ret24h"
            assert pd.Timestamp(ts) == idx[-1]
            return pd.Series([0.1, 0.2], index=pd.Index(symbols, name="symbol"))

        def get(self, *_args, **_kwargs):
            raise AssertionError("latest path should not assemble full frames")

    matrix = fg._latest_feature_matrix(
        Lazy(),
        ["AAA/USD:USD", "BBB/USD:USD"],
        idx[-1],
        {"ret24h"},
    )
    assert matrix.loc["AAA/USD:USD", "ret24h"] == np.float32(0.1)
    assert matrix.loc["BBB/USD:USD", "ret24h"] == np.float32(0.2)


def test_feature_history_matrix_append_latest_fastpath():
    idx = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    feats = {
        "ret24h": pd.DataFrame(
            {"AAA/USD:USD": [0.1, 0.2], "BBB/USD:USD": [0.3, 0.4]},
            index=idx,
            dtype=np.float32,
        )
    }
    matrix = fg._feature_history_matrix(
        feats,
        symbols=["AAA/USD:USD", "BBB/USD:USD"],
        required_feature_keys={"ret24h"},
        start_ts=idx[-1],
        end_ts=idx[-1],
    )
    assert list(matrix.columns) == ["ret24h"]
    assert list(matrix.index.get_level_values("timestamp").unique()) == [idx[-1]]
    assert matrix.loc[(idx[-1], "AAA/USD:USD"), "ret24h"] == np.float32(0.2)
    assert matrix.loc[(idx[-1], "BBB/USD:USD"), "ret24h"] == np.float32(0.4)


def test_selected_feature_latest_matrix_cache_roundtrip(tmp_path):
    idx = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    symbols = ["AAA/USD:USD", "BBB/USD:USD"]
    feats = {
        "ret24h": pd.DataFrame(
            {"AAA/USD:USD": [0.1, 0.2], "BBB/USD:USD": [0.3, 0.4]},
            index=idx,
            dtype=np.float32,
        ),
        "range_24h_pct": pd.DataFrame(
            {"AAA/USD:USD": [1.1, 1.2], "BBB/USD:USD": [1.3, 1.4]},
            index=idx,
            dtype=np.float32,
        ),
    }
    fg._write_selected_feature_latest_matrix_cache(
        cache_root=str(tmp_path),
        source_run_id="20260101_000000",
        source_root=str(tmp_path / "source"),
        symbols=symbols,
        feature_keys=set(feats),
        end_ts=idx[-1],
        feats=feats,
    )
    loaded = fg._load_selected_feature_latest_matrix_cache(
        cache_root=str(tmp_path),
        source_run_id="20260101_000000",
        source_root=str(tmp_path / "source"),
        symbols=symbols,
        feature_keys=set(feats),
        end_ts=idx[-1],
    )
    assert set(loaded) == set(feats)
    assert loaded["ret24h"].loc[idx[-1], "AAA/USD:USD"] == np.float32(0.2)
    assert loaded["range_24h_pct"].loc[idx[-1], "BBB/USD:USD"] == np.float32(1.4)


def test_load_features_selected_parallel_latest_path(tmp_path, monkeypatch):
    run_ts = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    in_dir = tmp_path / "features" / run_ts.strftime("%Y%m%d_%H%M%S")
    in_dir.mkdir(parents=True)
    idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC", name="ts")
    for sym, offset in [("AAA/USD:USD", 0.0), ("BBB/USD:USD", 10.0)]:
        safe = sym.replace("/", "_")
        df = pd.DataFrame(
            {
                "ret24h": np.array([0.1, 0.2, 0.3], dtype=np.float32) + offset,
                "unused": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "__symbol__": sym,
            },
            index=idx,
        )
        df.to_parquet(in_dir / f"symbol={safe}.parquet")

    monkeypatch.setenv("EPM_FEATURE_SELECTED_LOAD_WORKERS", "2")
    monkeypatch.setenv("EPM_FEATURE_SELECTED_LOAD_PARALLEL", "1")
    loaded = load_features_selected(
        run_ts,
        str(tmp_path),
        feature_keys=["ret24h"],
        symbols=["AAA/USD:USD", "BBB/USD:USD"],
        start_ts=idx[-1],
        end_ts=idx[-1] + pd.Timedelta(microseconds=1),
    )
    values = loaded.latest_values_at(
        "ret24h",
        ["AAA/USD:USD", "BBB/USD:USD"],
        idx[-1],
    )
    assert values.loc["AAA/USD:USD"] == np.float32(0.3)
    assert values.loc["BBB/USD:USD"] == np.float32(10.3)


def test_load_features_selected_reads_duckdb_delta_timestamp(tmp_path, monkeypatch):
    run_ts = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    in_dir = tmp_path / "features" / run_ts.strftime("%Y%m%d_%H%M%S")
    in_dir.mkdir(parents=True)
    sym = "AAA/USD:USD"
    safe = sym.replace("/", "_")
    idx = pd.DatetimeIndex([pd.Timestamp("2026-01-01 00:00:00", tz="UTC")], name="ts")
    parquet_path = in_dir / f"symbol={safe}.parquet"
    pd.DataFrame(
        {
            "ret24h": np.array([0.1], dtype=np.float32),
            "__symbol__": sym,
        },
        index=idx,
    ).to_parquet(parquet_path)

    delta_ts = pd.Timestamp("2026-01-01 01:00:00", tz="UTC")
    delta = pd.DataFrame(
        {
            "ret24h": np.array([0.2], dtype=np.float32),
            "__symbol__": sym,
        },
        index=pd.DatetimeIndex([delta_ts], name="ts"),
    )
    monkeypatch.setenv("EPM_FEATURE_DELTA_DUCKDB", "1")
    assert _write_feature_delta_part(str(parquet_path), sym, delta) == 1

    loaded = load_features_selected(
        run_ts,
        str(tmp_path),
        feature_keys=["ret24h"],
        symbols=[sym],
        start_ts=delta_ts,
        end_ts=delta_ts + pd.Timedelta(microseconds=1),
    )
    values = loaded.latest_values_at("ret24h", [sym], delta_ts)
    assert values.loc[sym] == np.float32(0.2)
