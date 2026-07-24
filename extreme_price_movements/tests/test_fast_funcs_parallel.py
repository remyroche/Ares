import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
import extreme_price_movements.fast_funcs as ff
from extreme_price_movements.inference.live_zscore_state import (
    RawRollingFeatureState,
    RawRollingStateContainer,
    RawRollingStateContainerBusy,
    raw_rolling_state_container_path,
)

def test_numba_rolling_mean_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)

    # Numba implementation behaves like min_periods=1
    expected = df.rolling(10, min_periods=1).mean().astype(np.float32)
    actual = ff.numba_rolling_mean(df, 10)

    pd.testing.assert_frame_equal(expected, actual, atol=1e-5)

def test_numba_rolling_std_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)

    # Numba implementation behaves like min_periods=2
    expected = df.rolling(10, min_periods=2).std().astype(np.float32)

    # Pandas puts NaN for min_periods < 2. Numba puts 0.0?
    # Let's check _numba_rolling_std_nan_safe.
    # It inits with np.full(..., np.nan).
    # if count > 1: output logic.
    # So it stays NaN if count <= 1.

    actual = ff.numba_rolling_std(df, 10)

    pd.testing.assert_frame_equal(expected, actual, atol=1e-5)

def test_numba_ewma_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)
    alpha = 0.5

    # Expected: Pandas ewm
    # adjust=False matches our default for most cases in features.py
    # ignore_na=True matches _numba_ewma_nan_safe behavior (skipping NaNs but not propagating unless all are NaN)
    # Actually _numba_ewma_nan_safe logic:
    # if np.isnan(val): out[i] = out[i-1] (carry forward)
    # Pandas ignore_na=True does weighted calc ignoring NaNs.
    # Pandas default ignore_na=False propagates NaNs?

    # Let's look at _numba_ewma_nan_safe again.
    # if np.isnan(val): out[i] = out[i-1]
    # else: update with val.

    # This is equivalent to "holding" the previous value during NaNs.
    # Pandas 'ignore_na=True' calculates weights assuming relative positions shift.
    # Our simple implementation just fills forward the output.
    # For clean data (no NaNs), it matches pandas.

    # We test with clean data first.
    expected = df.ewm(alpha=alpha, adjust=False).mean().astype(np.float32)
    actual = ff.numba_ewma(df, alpha, adjust=False)

    pd.testing.assert_frame_equal(expected, actual, atol=1e-5)

def test_numba_rolling_sum_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)

    # min_periods=1
    expected = df.rolling(10, min_periods=1).sum().astype(np.float32)
    actual = ff.numba_rolling_sum(df, 10)

    pd.testing.assert_frame_equal(expected, actual, atol=1e-4)


def test_raw_rolling_feature_state_append_matches_vectorized():
    np.random.seed(7)
    data = np.random.randn(32, 4).astype(np.float32)
    data[5, 1] = np.nan
    data[17, 2] = np.nan
    index = pd.date_range("2026-01-01", periods=data.shape[0], freq="h", tz="UTC")
    columns = [f"S{i}" for i in range(data.shape[1])]
    frame = pd.DataFrame(data, index=index, columns=columns)
    window = 8
    seed_rows = 24

    expected = {
        "sum": ff.numba_rolling_sum(frame, window),
        "mean": ff.numba_rolling_mean(frame, window),
        "std": ff.apply_to_frame(frame, ff._numba_rolling_std_nan_safe, window),
        "max": ff.numba_rolling_max(frame, window),
        "min": ff.numba_rolling_min(frame, window),
    }
    for op in ("sum", "mean", "std", "max", "min"):
        state = RawRollingFeatureState(
            op=op,
            name="x",
            symbols=columns,
            window=window,
        )
        state.seed_from_frame(data[:seed_rows], index[:seed_rows])
        out = []
        for pos in range(seed_rows, data.shape[0]):
            out.append(state.update(data[pos], timestamp=index[pos].isoformat()))
        actual_tail = pd.DataFrame(
            np.asarray(out, dtype=np.float32),
            index=index[seed_rows:],
            columns=columns,
        )
        pd.testing.assert_frame_equal(
            expected[op].iloc[seed_rows:].astype(np.float32),
            actual_tail.astype(np.float32),
            atol=1e-4,
        )


def test_raw_rolling_state_container_preserves_namespaced_state(tmp_path):
    index = pd.date_range("2026-07-01", periods=6, freq="h", tz="UTC")
    values = np.arange(12, dtype=np.float32).reshape(6, 2)
    symbols = ["AAA/USD:USD", "BBB/USD:USD"]
    state = RawRollingFeatureState(
        op="mean",
        name="ret1h",
        symbols=symbols,
        window=4,
    )
    state.seed_from_frame(values, index)
    legacy_root = tmp_path / "raw_rolling_state.npz"
    container_path = raw_rolling_state_container_path(legacy_root)

    container = RawRollingStateContainer.open(container_path)
    container.put("workset_a", state)
    assert container.flush() == 1
    container.close()

    reopened = RawRollingStateContainer.open(container_path)
    loaded = reopened.get(
        "workset_a",
        op="mean",
        name="ret1h",
        symbols=symbols,
        window=4,
    )
    incompatible = reopened.get(
        "workset_a",
        op="mean",
        name="ret1h",
        symbols=symbols,
        window=8,
    )
    reopened.close()

    assert loaded is not None
    assert incompatible is None
    assert loaded.last_timestamp == state.last_timestamp
    np.testing.assert_array_equal(loaded.buffer, state.buffer)
    np.testing.assert_array_equal(loaded.valid, state.valid)
    np.testing.assert_array_equal(loaded.ptr, state.ptr)
    np.testing.assert_array_equal(loaded.count, state.count)
    np.testing.assert_array_equal(loaded.sum, state.sum)
    np.testing.assert_array_equal(loaded.sum_sq, state.sum_sq)


def test_raw_rolling_state_container_aborts_unflushed_updates(tmp_path):
    state = RawRollingFeatureState(
        op="sum",
        name="ret1h",
        symbols=["AAA/USD:USD"],
        window=4,
    )
    state.seed_from_frame(
        np.asarray([[1.0], [2.0]], dtype=np.float32),
        pd.date_range("2026-07-01", periods=2, freq="h", tz="UTC"),
    )
    path = raw_rolling_state_container_path(tmp_path / "raw_rolling_state.npz")
    container = RawRollingStateContainer.open(path)
    container.put("unflushed", state)
    container.abort()

    reopened = RawRollingStateContainer.open(path)
    assert (
        reopened.get(
            "unflushed",
            op="sum",
            name="ret1h",
            symbols=["AAA/USD:USD"],
            window=4,
        )
        is None
    )
    reopened.close()


def test_raw_rolling_state_container_holds_exclusive_snapshot_lock(tmp_path):
    path = raw_rolling_state_container_path(tmp_path / "raw_rolling_state.npz")
    owner = RawRollingStateContainer.open(path, lock_timeout_seconds=0.0)
    with pytest.raises(RawRollingStateContainerBusy):
        RawRollingStateContainer.open(path, lock_timeout_seconds=0.0)
    owner.abort()

    reopened = RawRollingStateContainer.open(path, lock_timeout_seconds=0.0)
    reopened.close()


def test_raw_rolling_state_container_quarantines_corrupt_database(tmp_path):
    path = raw_rolling_state_container_path(tmp_path / "raw_rolling_state.npz")
    path.write_text("not a sqlite database", encoding="utf-8")

    container = RawRollingStateContainer.open(path, lock_timeout_seconds=0.0)
    assert list(tmp_path.glob("raw_rolling_state.container.sqlite.corrupt.*"))
    state = RawRollingFeatureState(
        op="sum",
        name="ret1h",
        symbols=["AAA/USD:USD"],
        window=4,
    )
    state.seed_from_frame(
        np.asarray([[1.0]], dtype=np.float32),
        pd.date_range("2026-07-01", periods=1, freq="h", tz="UTC"),
    )
    container.put("recovered", state)
    assert container.flush() == 1
    container.close()

    reopened = RawRollingStateContainer.open(path, lock_timeout_seconds=0.0)
    assert (
        reopened.get(
            "recovered",
            op="sum",
            name="ret1h",
            symbols=["AAA/USD:USD"],
            window=4,
        )
        is not None
    )
    reopened.close()


def test_raw_rolling_state_container_rejects_tampered_entry_contract(tmp_path):
    path = raw_rolling_state_container_path(tmp_path / "raw_rolling_state.npz")
    state = RawRollingFeatureState(
        op="mean",
        name="ret1h",
        symbols=["AAA/USD:USD"],
        window=4,
    )
    state.seed_from_frame(
        np.asarray([[1.0]], dtype=np.float32),
        pd.date_range("2026-07-01", periods=1, freq="h", tz="UTC"),
    )
    container = RawRollingStateContainer.open(path)
    container.put("expected_key", state)
    container.flush()
    container.close()

    connection = sqlite3.connect(path)
    metadata_raw = connection.execute(
        "SELECT metadata_json FROM raw_rolling_states WHERE state_key = ?",
        ("expected_key",),
    ).fetchone()[0]
    metadata = json.loads(metadata_raw)
    metadata["container_state_key"] = "wrong_key"
    connection.execute(
        "UPDATE raw_rolling_states SET metadata_json = ? WHERE state_key = ?",
        (json.dumps(metadata), "expected_key"),
    )
    connection.commit()
    connection.close()

    reopened = RawRollingStateContainer.open(path)
    assert (
        reopened.get(
            "expected_key",
            op="mean",
            name="ret1h",
            symbols=["AAA/USD:USD"],
            window=4,
        )
        is None
    )
    reopened.close()


def test_numba_rolling_max_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)

    # min_periods=1
    expected = df.rolling(10, min_periods=1).max().astype(np.float32)
    actual = ff.numba_rolling_max(df, 10)

    pd.testing.assert_frame_equal(expected, actual, atol=1e-5)

def test_numba_rolling_min_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)

    # min_periods=1
    expected = df.rolling(10, min_periods=1).min().astype(np.float32)
    actual = ff.numba_rolling_min(df, 10)

    pd.testing.assert_frame_equal(expected, actual, atol=1e-5)

def test_numba_rolling_median_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)
    window = 10

    # _numba_rolling_median outputs 0.0 for the first window-1 elements
    # Pandas outputs NaN.
    # We'll replace Pandas NaNs with 0.0 to match our legacy implementation behavior
    expected = df.rolling(window).median().fillna(0.0).astype(np.float32)

    actual = ff.numba_rolling_median(df, window)

    pd.testing.assert_frame_equal(expected, actual, atol=1e-5)

def test_numba_pct_change_correctness():
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    df = pd.DataFrame(data)

    expected = df.pct_change(1).astype(np.float32)
    actual = ff.numba_pct_change(df, 1)

    # Pandas fills first row with NaN.
    # _numba_pct_change also fills with NaN.

    pd.testing.assert_frame_equal(expected, actual, atol=1e-5)
