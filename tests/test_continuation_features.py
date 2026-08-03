import numpy as np
import pandas as pd

from extreme_price_movements.continuation_features import materialize_ohlcv_continuation_features
from scripts.run_exact_h12_target_purity_ablation import _hierarchical_persistence_expected_net


def _bars() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=64, freq="h", tz="UTC")
    rows = []
    for symbol, offset in (("A_USD:USD", 0.0), ("B_USD:USD", 0.2)):
        for i, stamp in enumerate(ts):
            close = 100.0 + offset + i * 0.1
            rows.append({"ts": stamp, "symbol": symbol, "open": close - 0.05, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 1000.0 + i})
    return pd.DataFrame(rows)


def test_rolling_features_use_trailing_data_only():
    baseline = materialize_ohlcv_continuation_features(_bars())
    changed = _bars()
    changed.loc[(changed.symbol.eq("A_USD:USD")) & (changed.ts.gt(pd.Timestamp("2024-01-03 00:00", tz="UTC"))), "close"] *= 5.0
    candidate = materialize_ohlcv_continuation_features(changed)
    cutoff = pd.Timestamp("2024-01-03 00:00", tz="UTC")
    cols = [name for name in baseline if name.startswith("cont_")]
    before = baseline.loc[(baseline.symbol.eq("A_USD:USD")) & (baseline.ts.le(cutoff)), cols]
    after = candidate.loc[(candidate.symbol.eq("A_USD:USD")) & (candidate.ts.le(cutoff)), cols]
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))


def test_cross_sectional_features_use_timestamp_eligible_universe():
    bars = _bars()
    full = materialize_ohlcv_continuation_features(bars)
    reduced = materialize_ohlcv_continuation_features(bars.loc[~((bars.symbol.eq("B_USD:USD")) & (bars.ts.eq(bars.ts.max())))])
    last = bars.ts.max()
    assert int(full.loc[full.ts.eq(last), "cont_cs_universe_size"].iloc[0]) == 2
    assert int(reduced.loc[(reduced.ts.eq(last)) & (reduced.symbol.eq("A_USD:USD")), "cont_cs_universe_size"].iloc[0]) == 1


def test_ohlcv_proxy_names_cannot_masquerade_as_factual_l2():
    generated = materialize_ohlcv_continuation_features(_bars()).columns
    forbidden = ("orderbook", "depth", "aggressor", "liquidation", "spread")
    assert all(not any(token in name.lower() for token in forbidden) or name.endswith(("_proxy", "_estimator", "_ohlcv_proxy")) for name in generated)


def test_stage_b_retention_matrix_changes_only_retention_head_features():
    n = 300
    rng = np.random.default_rng(11)
    train = pd.DataFrame({"exact_h12_net_bps": rng.normal(0, 100, n), "postcost_h0_four_state": np.resize(["clear_then_retained", "clear_then_giveback", "adverse_first_or_conflict", "timeout"], n)})
    test = train.iloc[:50].copy()
    base_train = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
    base_test = base_train.iloc[:50].copy()
    retain_train = pd.DataFrame({"only_retention": rng.normal(size=n)})
    retain_test = retain_train.iloc[:50].copy()
    first = _hierarchical_persistence_expected_net(train, test, base_train, base_test, retention_x_train=retain_train, retention_x_test=retain_test, seed=7, trees=12, token="h0", return_components=True)
    second = _hierarchical_persistence_expected_net(train, test, base_train, base_test, retention_x_train=retain_train.assign(only_retention=lambda x: -x.only_retention), retention_x_test=retain_test.assign(only_retention=lambda x: -x.only_retention), seed=7, trees=12, token="h0", return_components=True)
    np.testing.assert_allclose(first["p_clear_cost_before_adverse"], second["p_clear_cost_before_adverse"])
    np.testing.assert_allclose(first["p_adverse_given_not_clear"], second["p_adverse_given_not_clear"])
