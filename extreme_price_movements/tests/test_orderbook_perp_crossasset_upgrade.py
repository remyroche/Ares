import numpy as np
import pandas as pd

from extreme_price_movements.features import compute_orderbook_snapshot_features
from extreme_price_movements.config import CFG


def _close_panel():
    idx = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
    cols = ["BTC/USDT", "ETH/USDT"]
    close = pd.DataFrame([[100, 50], [101, 51], [102, 52], [103, 53], [104, 54], [105, 55]], index=idx, columns=cols, dtype=float)
    vol = pd.DataFrame(1000.0, index=idx, columns=cols)
    atr = pd.DataFrame(0.01, index=idx, columns=cols)
    return idx, cols, close, vol, atr


def test_orderbook_shapes_and_nonzero_with_valid_l2():
    idx, cols, close, vol, atr = _close_panel()
    rows = []
    for ts in idx:
        for sym, px in [("BTC/USDT", 100.0), ("ETH/USDT", 50.0)]:
            for lvl in range(1, 21):
                rows.append((ts, sym, "bid", lvl, px * (1 - lvl * 0.0001), 10 + lvl))
                rows.append((ts, sym, "ask", lvl, px * (1 + lvl * 0.0001), 9 + lvl))
    ob = pd.DataFrame(rows, columns=["timestamp", "symbol", "side", "level", "price", "qty"])
    feats = compute_orderbook_snapshot_features(ob, close, vol, atr, CFG, shift_bars=1)
    assert all(df.shape == close.shape for df in feats.values())
    assert all(df.index.equals(close.index) for df in feats.values())
    assert all(df.columns.equals(close.columns) for df in feats.values())
    assert float(feats["ob_l10_imbalance"].abs().sum().sum()) > 0.0


def test_orderbook_missing_sets_flags_not_fake_signal():
    _, _, close, vol, atr = _close_panel()
    feats = compute_orderbook_snapshot_features(None, close, vol, atr, CFG, shift_bars=1)
    assert (feats["ob_available"] == 0).all().all()
    assert (feats["ob_stale_flag"] == 1).all().all()


def test_orderbook_causal_future_change_does_not_touch_past():
    idx, _, close, vol, atr = _close_panel()
    rows = []
    for ts in idx:
        for lvl in range(1, 21):
            rows.append((ts, "BTC/USDT", "bid", lvl, 100 * (1 - lvl * 0.0001), 10 + lvl))
            rows.append((ts, "BTC/USDT", "ask", lvl, 100 * (1 + lvl * 0.0001), 10 + lvl))
    ob = pd.DataFrame(rows, columns=["timestamp", "symbol", "side", "level", "price", "qty"])
    base = compute_orderbook_snapshot_features(ob, close, vol, atr, CFG, shift_bars=1)
    ob2 = ob.copy()
    ob2.loc[ob2["timestamp"] == idx[-1], "qty"] *= 100.0
    mod = compute_orderbook_snapshot_features(ob2, close, vol, atr, CFG, shift_bars=1)
    t = idx[-2]
    assert np.isclose(base["ob_l10_imbalance"].loc[t, "BTC/USDT"], mod["ob_l10_imbalance"].loc[t, "BTC/USDT"], equal_nan=True)


def test_snapshot_impacts_t_plus_1_only():
    idx, _, close, vol, atr = _close_panel()
    ob = pd.DataFrame(
        [
            (idx[2], "BTC/USDT", "bid", 1, 100.0, 10.0),
            (idx[2], "BTC/USDT", "ask", 1, 100.2, 10.0),
        ],
        columns=["timestamp", "symbol", "side", "level", "price", "qty"],
    )
    feats = compute_orderbook_snapshot_features(ob, close, vol, atr, CFG, shift_bars=1)
    assert feats["ob_available"].loc[idx[2], "BTC/USDT"] == 0.0
    assert feats["ob_available"].loc[idx[3], "BTC/USDT"] == 1.0
    assert feats["ob_available"].loc[idx[4], "BTC/USDT"] == 1.0


def test_stale_masks_dependent_features():
    idx, _, close, vol, atr = _close_panel()
    ob = pd.DataFrame(
        [
            (idx[0], "BTC/USDT", "bid", 1, 100.0, 10.0),
            (idx[0], "BTC/USDT", "ask", 1, 100.2, 10.0),
        ],
        columns=["timestamp", "symbol", "side", "level", "price", "qty"],
    )
    cfg = dict(CFG)
    cfg["orderbook_stale_hours"] = 0.0
    feats = compute_orderbook_snapshot_features(ob, close, vol, atr, cfg, shift_bars=1)
    assert feats["ob_available"].loc[idx[2], "BTC/USDT"] == 0.0
    assert feats["ob_spread_bps"].loc[idx[2], "BTC/USDT"] == 0.0


def test_bps_buckets_differ_and_wall_found_flag():
    idx, _, close, vol, atr = _close_panel()
    rows = []
    for lvl in range(1, 21):
        rows.append((idx[1], "BTC/USDT", "bid", lvl, 100.0 * (1 - lvl * 0.0005), 1000.0 if lvl == 10 else 1.0))
        rows.append((idx[1], "BTC/USDT", "ask", lvl, 100.0 * (1 + lvl * 0.0005), 1000.0 if lvl == 11 else 1.0))
    ob = pd.DataFrame(rows, columns=["timestamp", "symbol", "side", "level", "price", "qty"])
    feats = compute_orderbook_snapshot_features(ob, close, vol, atr, CFG, shift_bars=1)
    t = idx[2]
    assert feats["ob_bid_depth_5bps"].loc[t, "BTC/USDT"] != feats["ob_bid_depth_100bps"].loc[t, "BTC/USDT"]
    assert feats["ob_bid_wall_found"].loc[t, "BTC/USDT"] in (0.0, 1.0)


def test_tz_naive_index_is_normalized():
    idx, _, close, vol, atr = _close_panel()
    close_naive = close.copy()
    close_naive.index = close_naive.index.tz_convert(None)
    vol_naive = vol.copy()
    vol_naive.index = close_naive.index
    atr_naive = atr.copy()
    atr_naive.index = close_naive.index
    ob = pd.DataFrame(
        [(idx[0], "BTC/USDT", "bid", 1, 100.0, 10.0), (idx[0], "BTC/USDT", "ask", 1, 100.1, 10.0)],
        columns=["timestamp", "symbol", "side", "level", "price", "qty"],
    )
    feats = compute_orderbook_snapshot_features(ob, close_naive, vol_naive, atr_naive, CFG, shift_bars=1)
    assert str(feats["ob_available"].index.tz) == "UTC"
    assert feats["ob_nearest_bid_wall_to_qv24"].notna().all().all()


def test_no_qualifying_wall_sets_flag_zero():
    idx, _, close, vol, atr = _close_panel()
    rows = []
    for lvl in range(1, 21):
        rows.append((idx[1], "BTC/USDT", "bid", lvl, 100.0 * (1 - lvl * 0.0001), 1.0))
        rows.append((idx[1], "BTC/USDT", "ask", lvl, 100.0 * (1 + lvl * 0.0001), 1.0))
    ob = pd.DataFrame(rows, columns=["timestamp", "symbol", "side", "level", "price", "qty"])
    cfg = dict(CFG)
    cfg["orderbook_wall_qty_mult"] = 10.0
    feats = compute_orderbook_snapshot_features(ob, close, vol, atr, cfg, shift_bars=1)
    assert feats["ob_bid_wall_found"].loc[idx[2], "BTC/USDT"] == 0.0
    assert feats["ob_ask_wall_found"].loc[idx[2], "BTC/USDT"] == 0.0
