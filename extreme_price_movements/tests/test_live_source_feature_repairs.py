import numpy as np
import pandas as pd

import extreme_price_movements.fast_funcs as ff
import extreme_price_movements.inference.feature_generator as live_fg
from extreme_price_movements.features_residual import add_residual_features
from extreme_price_movements.inference.data_fetcher import get_panel_from_dict
from extreme_price_movements.inference.feature_generator import (
    _backfill_missing_requested_keys,
    _materialize_live_perp_contract_aliases,
    _source_derived_unusable_requested_keys,
)
from extreme_price_movements.inference.run_inference import (
    _filter_strategy_masks_by_finite_model_contract,
)
from extreme_price_movements.perp_features import compute_features as compute_perp_features


def test_get_panel_from_dict_synthesizes_quote_volume_from_close_and_volume():
    idx = pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
    ohlcv = {
        "BTC/USD:USD": pd.DataFrame(
            {
                "open": [9.0, 10.0],
                "high": [11.0, 12.0],
                "low": [8.0, 9.0],
                "close": [10.0, 11.0],
                "volume": [2.0, 0.0],
            },
            index=idx,
        )
    }

    panel = get_panel_from_dict(ohlcv)

    assert "quote_volume" in panel
    assert panel["quote_volume"].loc[idx[0], "BTC/USD:USD"] == np.float32(20.0)
    assert panel["quote_volume"].loc[idx[1], "BTC/USD:USD"] == np.float32(0.0)


def test_materialize_live_perp_contract_aliases_match_training_perp_features():
    idx = pd.date_range("2026-01-01", periods=400, freq="1h", tz="UTC")
    cols = ["BTC/USD:USD", "ETH/USD:USD"]
    x = np.linspace(0.0, 20.0, len(idx), dtype=np.float32)
    close = pd.DataFrame(
        {
            cols[0]: 100.0 + 0.05 * np.sin(x),
            cols[1]: 80.0 + 0.04 * np.cos(x),
        },
        index=idx,
    ).astype(np.float32)
    spot = close / pd.DataFrame(
        {
            cols[0]: 1.0 + 0.001 * np.sin(x),
            cols[1]: 1.0 + 0.0015 * np.cos(x),
        },
        index=idx,
    ).astype(np.float32)
    funding = pd.DataFrame(
        {
            cols[0]: 0.0001 * np.sin(x / 3.0),
            cols[1]: 0.00012 * np.cos(x / 4.0),
        },
        index=idx,
    ).astype(np.float32)
    oi = pd.DataFrame(
        {
            cols[0]: 100000.0 + 1000.0 * np.sin(x / 2.0),
            cols[1]: 90000.0 + 800.0 * np.cos(x / 2.5),
        },
        index=idx,
    ).astype(np.float32)
    volume = pd.DataFrame(
        {
            cols[0]: 25.0 + 2.0 * np.sin(x),
            cols[1]: 20.0 + 1.5 * np.cos(x),
        },
        index=idx,
    ).astype(np.float32)
    quote_volume = (close * volume).astype(np.float32)
    feats = {
        "basis_mom_2h": pd.DataFrame(np.nan, index=idx, columns=cols, dtype=np.float32),
    }
    panel = {
        "close": close,
        "index_price": spot,
        "funding_rate": funding,
        "open_interest": oi,
        "volume": volume,
        "quote_volume": quote_volume,
    }
    required = {
        "basis",
        "basis_mom_2h",
        "basis_mom_8h",
        "basis_mom_w",
        "basis_stretch",
        "basis_vol",
        "unwind_score",
    }

    out = _materialize_live_perp_contract_aliases(panel, cols, feats, required)

    for key in required:
        assert key in out
        latest = out[key].iloc[-1].to_numpy(dtype=np.float64, copy=False)
        assert np.isfinite(latest).all(), key

    for sym in cols:
        expected = compute_perp_features(
            pd.DataFrame(
                {
                    "funding_rate": funding[sym],
                    "open_interest": oi[sym],
                    "open_interest_quote": oi[sym] * close[sym],
                    "perp_price": close[sym],
                    "spot_price": spot[sym],
                    "volume": volume[sym],
                    "quote_volume": quote_volume[sym],
                    "close": close[sym],
                },
                index=idx,
            )
        )
        for key in required:
            actual_latest = float(out[key].loc[idx[-1], sym])
            expected_latest = float(expected.loc[idx[-1], key])
            assert np.isclose(actual_latest, expected_latest, rtol=1e-6, atol=1e-7), key


def test_materialize_live_perp_aliases_use_sidecar_history_when_ohlcv_tail_short():
    idx = pd.date_range("2026-01-01", periods=900, freq="1h", tz="UTC")
    cols = ["BTC/USD:USD", "ETH/USD:USD"]
    x = np.linspace(0.0, 40.0, len(idx), dtype=np.float32)
    close_full = pd.DataFrame(
        {
            cols[0]: 100.0 + 0.08 * np.sin(x),
            cols[1]: 80.0 + 0.06 * np.cos(x / 2.0),
        },
        index=idx,
    ).astype(np.float32)
    spot_full = close_full / pd.DataFrame(
        {
            cols[0]: 1.0 + 0.0010 * np.sin(x / 2.0),
            cols[1]: 1.0 + 0.0012 * np.cos(x / 3.0),
        },
        index=idx,
    ).astype(np.float32)
    tail_mask = pd.DataFrame(False, index=idx, columns=cols)
    tail_mask.iloc[-60:] = True
    close_tail = close_full.where(tail_mask).astype(np.float32)
    spot_tail = spot_full.where(tail_mask).astype(np.float32)
    mark = (close_full * 1.0002).astype(np.float32)
    index_price = (spot_full * 0.9999).astype(np.float32)
    funding = pd.DataFrame(
        {
            cols[0]: 0.00010 * np.sin(x / 3.0),
            cols[1]: 0.00012 * np.cos(x / 4.0),
        },
        index=idx,
    ).astype(np.float32)
    oi = pd.DataFrame(
        {
            cols[0]: 100000.0 + 1000.0 * np.sin(x / 2.0),
            cols[1]: 90000.0 + 800.0 * np.cos(x / 2.5),
        },
        index=idx,
    ).astype(np.float32)
    volume = pd.DataFrame(
        {
            cols[0]: 25.0 + 2.0 * np.sin(x),
            cols[1]: 20.0 + 1.5 * np.cos(x),
        },
        index=idx,
    ).astype(np.float32)
    quote_volume = (mark * volume).astype(np.float32)
    sparse_existing = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=np.float32)
    sparse_existing.iloc[-1, 0] = 0.0
    panel = {
        "close": close_tail,
        "spot_close": spot_tail,
        "mark_price": mark,
        "index_price": index_price,
        "funding_rate": funding,
        "open_interest": oi,
        "volume": volume,
        "quote_volume": quote_volume,
    }
    feats = {"basis_mom_2h": sparse_existing}
    required = {
        "basis",
        "basis_frac_rank_30d",
        "basis_mom_2h",
        "basis_mom_8h",
        "basis_mom_w",
        "basis_stretch",
        "unwind_score",
    }

    out = _materialize_live_perp_contract_aliases(panel, cols, feats, required)

    perp_price = close_tail.combine_first(mark).astype(np.float32)
    reference = spot_tail.combine_first(index_price).astype(np.float32)
    for key in required:
        assert key in out
        assert np.isfinite(out[key].iloc[-1].to_numpy(dtype=np.float64)).all(), key

    for sym in cols:
        expected = compute_perp_features(
            pd.DataFrame(
                {
                    "funding_rate": funding[sym],
                    "open_interest": oi[sym],
                    "open_interest_quote": oi[sym] * perp_price[sym],
                    "perp_price": perp_price[sym],
                    "spot_price": reference[sym],
                    "volume": volume[sym],
                    "quote_volume": quote_volume[sym],
                    "close": perp_price[sym],
                    "mark_price": mark[sym],
                },
                index=idx,
            )
        )
        for key in required:
            actual_latest = float(out[key].loc[idx[-1], sym])
            expected_latest = float(expected.loc[idx[-1], key])
            assert np.isclose(actual_latest, expected_latest, rtol=1e-6, atol=1e-7), key


def test_sparse_source_derived_latest_frame_is_marked_unusable():
    idx = pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
    cols = ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"]
    sparse = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=np.float32)
    sparse.loc[idx[-1], cols[0]] = np.float32(0.1)

    unusable = _source_derived_unusable_requested_keys(
        {"basis_mom_2h": sparse},
        {"basis_mom_2h"},
        cols,
        end_ts=idx[-1],
    )

    assert unusable == {"basis_mom_2h"}


def test_sparse_train_tolerated_source_keys_are_not_repairable_unusable():
    idx = pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
    cols = ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"]
    sparse = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=np.float32)
    sparse.loc[idx[-1], cols[0]] = np.float32(0.1)
    feats = {"dist_vwap_atr": sparse, "basis_mom_2h": sparse.copy()}

    full = _source_derived_unusable_requested_keys(
        feats,
        {"dist_vwap_atr", "basis_mom_2h"},
        cols,
        end_ts=idx[-1],
    )
    repairable = _source_derived_unusable_requested_keys(
        feats,
        {"dist_vwap_atr", "basis_mom_2h"},
        cols,
        end_ts=idx[-1],
        repairable_only=True,
    )

    assert full == {"basis_mom_2h", "dist_vwap_atr"}
    assert repairable == {"basis_mom_2h"}


def test_source_backfill_materializes_sparse_contract_without_shared_recompute(monkeypatch):
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    cols = ["BTC/USD:USD", "ETH/USD:USD"]
    close = pd.DataFrame(100.0, index=idx, columns=cols, dtype=np.float32)

    def fail_compute_features_hourly(*args, **kwargs):
        raise AssertionError("source-derived sparse keys should not run shared compute")

    monkeypatch.setattr(live_fg, "compute_features_hourly", fail_compute_features_hourly)

    out = _backfill_missing_requested_keys(
        {"close": close},
        cols,
        {},
        {},
        {"dist_vwap_atr", "oi_rel_vol_4h"},
    )

    assert set(out) == {"dist_vwap_atr", "oi_rel_vol_4h"}
    for key in out:
        assert out[key].index.equals(idx)
        assert list(out[key].columns) == cols
        assert np.isnan(out[key].to_numpy(dtype=np.float32)).all()


def test_finite_contract_gate_allows_train_tolerated_nonfinite_but_blocks_other():
    idx = pd.date_range("2026-01-01", periods=1, freq="1h", tz="UTC")
    feats = {
        "dist_vwap_atr": pd.DataFrame(
            [[np.nan, 1.0, 1.0]],
            index=idx,
            columns=["A/USD:USD", "B/USD:USD", "C/USD:USD"],
            dtype=np.float32,
        ),
        "volume_entropy_24": pd.DataFrame(
            [[1.0, np.nan, 1.0]],
            index=idx,
            columns=["A/USD:USD", "B/USD:USD", "C/USD:USD"],
            dtype=np.float32,
        ),
        "ret1h_G_VOL_0": pd.DataFrame(
            [[np.nan, np.nan, np.nan]],
            index=idx,
            columns=["A/USD:USD", "B/USD:USD", "C/USD:USD"],
            dtype=np.float32,
        ),
        "ret1h_G_VOL_1": pd.DataFrame(
            [[np.nan, np.nan, np.nan]],
            index=idx,
            columns=["A/USD:USD", "B/USD:USD", "C/USD:USD"],
            dtype=np.float32,
        ),
        "ret4h": pd.DataFrame(
            [[1.0, 1.0, np.nan]],
            index=idx,
            columns=["A/USD:USD", "B/USD:USD", "C/USD:USD"],
            dtype=np.float32,
        ),
    }
    masks = {"strategy": ["A/USD:USD", "B/USD:USD", "C/USD:USD"]}
    contracts = {
        "strategy": [
            "dist_vwap_atr",
            "volume_entropy_24",
            "ret1h_G_VOL_0",
            "ret1h_G_VOL_1",
            "ret4h",
        ]
    }

    filtered, diagnostics = _filter_strategy_masks_by_finite_model_contract(
        feats,
        masks,
        contracts,
        latest_ts=idx[-1],
        cfg={
            "strict_feature_parity": False,
            "live_model_contract_allow_train_tolerated_nonfinite": True,
        },
    )

    assert filtered["strategy"] == ["A/USD:USD", "B/USD:USD"]
    assert diagnostics["strategy"]["top_blocking_nonfinite_features"] == [
        {"feature": "ret4h", "rows": 1, "pct": 33.33}
    ]
    allowed = {
        item["feature"]
        for item in diagnostics["strategy"]["top_allowed_nonfinite_features"]
    }
    assert allowed == {
        "dist_vwap_atr",
        "volume_entropy_24",
        "ret1h_G_VOL_0",
        "ret1h_G_VOL_1",
    }


def test_finite_contract_gate_strict_parity_rejects_tolerated_nan():
    idx = pd.date_range("2026-01-01", periods=1, freq="1h", tz="UTC")
    feats = {
        "volume_entropy_24": pd.DataFrame(
            [[np.nan]], index=idx, columns=["A/USD:USD"], dtype=np.float32
        )
    }

    filtered, diagnostics = _filter_strategy_masks_by_finite_model_contract(
        feats,
        {"strategy": ["A/USD:USD"]},
        {"strategy": ["volume_entropy_24"]},
        latest_ts=idx[-1],
        cfg={
            "strict_feature_parity": True,
            "live_model_contract_allow_train_tolerated_nonfinite": True,
        },
    )

    assert filtered["strategy"] == []
    assert diagnostics["strategy"]["rejected"] == 1
    assert diagnostics["strategy"]["top_allowed_nonfinite_features"] == []


def test_materialize_live_residual_and_premium_aliases_match_training_helpers():
    idx = pd.date_range("2026-01-01", periods=400, freq="1h", tz="UTC")
    cols = ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"]
    x = np.linspace(0.0, 30.0, len(idx), dtype=np.float32)
    squeeze = pd.DataFrame(
        {
            cols[0]: 0.40 + 0.10 * np.sin(x),
            cols[1]: 0.45 + 0.12 * np.cos(x / 2.0),
            cols[2]: 0.35 + 0.08 * np.sin(x / 3.0),
        },
        index=idx,
    ).astype(np.float32)
    premium = pd.DataFrame(
        {
            cols[0]: 0.001 * np.sin(x / 4.0),
            cols[1]: 0.0012 * np.cos(x / 5.0),
            cols[2]: 0.0008 * np.sin(x / 6.0),
        },
        index=idx,
    ).astype(np.float32)
    feats = {
        "squeeze_prob": squeeze,
        "premium_proxy": premium,
        "squeeze_prob_mkt_resid": pd.DataFrame(np.nan, index=idx, columns=cols),
        "premium_expansion_speed_5h": pd.DataFrame(np.nan, index=idx, columns=cols),
    }
    panel = {"close": pd.DataFrame(100.0, index=idx, columns=cols)}
    required = {"squeeze_prob_mkt_resid", "premium_expansion_speed_5h"}

    out = _materialize_live_perp_contract_aliases(
        panel,
        cols,
        feats,
        required,
        cfg={"market_basket": cols},
    )

    expected_residual_inputs = {"squeeze_prob": squeeze.copy()}
    add_residual_features(expected_residual_inputs, None, {"market_basket": cols})
    expected_premium = ff.numba_rolling_zscore_fused(
        premium.diff(5).astype(np.float32),
        24 * 14,
    ).clip(-6.0, 6.0)

    for sym in cols:
        assert np.isclose(
            float(out["squeeze_prob_mkt_resid"].loc[idx[-1], sym]),
            float(expected_residual_inputs["squeeze_prob_mkt_resid"].loc[idx[-1], sym]),
            rtol=1e-6,
            atol=1e-7,
        )
        assert np.isclose(
            float(out["premium_expansion_speed_5h"].loc[idx[-1], sym]),
            float(expected_premium.loc[idx[-1], sym]),
            rtol=1e-6,
            atol=1e-7,
        )


def test_materialize_live_squeeze_residual_computes_hidden_base_dependency():
    idx = pd.date_range("2026-01-01", periods=900, freq="1h", tz="UTC")
    cols = ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"]
    x = np.linspace(0.0, 30.0, len(idx), dtype=np.float32)
    close = pd.DataFrame(
        {
            cols[0]: 100.0 + 0.08 * np.sin(x),
            cols[1]: 85.0 + 0.06 * np.cos(x / 2.0),
            cols[2]: 55.0 + 0.05 * np.sin(x / 3.0),
        },
        index=idx,
    ).astype(np.float32)
    spot = close / pd.DataFrame(
        {
            cols[0]: 1.0 + 0.001 * np.sin(x / 2.0),
            cols[1]: 1.0 + 0.0012 * np.cos(x / 3.0),
            cols[2]: 1.0 + 0.0008 * np.sin(x / 4.0),
        },
        index=idx,
    ).astype(np.float32)
    funding = pd.DataFrame(
        {
            cols[0]: 0.00010 * np.sin(x / 3.0),
            cols[1]: 0.00012 * np.cos(x / 4.0),
            cols[2]: 0.00008 * np.sin(x / 5.0),
        },
        index=idx,
    ).astype(np.float32)
    oi = pd.DataFrame(
        {
            cols[0]: 100000.0 + 1000.0 * np.sin(x / 2.0),
            cols[1]: 90000.0 + 800.0 * np.cos(x / 2.5),
            cols[2]: 75000.0 + 700.0 * np.sin(x / 3.0),
        },
        index=idx,
    ).astype(np.float32)
    volume = pd.DataFrame(
        {
            cols[0]: 25.0 + 2.0 * np.sin(x),
            cols[1]: 20.0 + 1.5 * np.cos(x),
            cols[2]: 18.0 + 1.2 * np.sin(x / 2.0),
        },
        index=idx,
    ).astype(np.float32)
    quote_volume = (close * volume).astype(np.float32)
    panel = {
        "close": close,
        "index_price": spot,
        "funding_rate": funding,
        "open_interest": oi,
        "volume": volume,
        "quote_volume": quote_volume,
    }
    feats = {
        "squeeze_prob_mkt_resid": pd.DataFrame(np.nan, index=idx, columns=cols),
    }

    out = _materialize_live_perp_contract_aliases(
        panel,
        cols,
        feats,
        {"squeeze_prob_mkt_resid"},
        cfg={"market_basket": cols},
    )

    expected_squeeze = {}
    for sym in cols:
        expected = compute_perp_features(
            pd.DataFrame(
                {
                    "funding_rate": funding[sym],
                    "open_interest": oi[sym],
                    "open_interest_quote": oi[sym] * close[sym],
                    "perp_price": close[sym],
                    "spot_price": spot[sym],
                    "volume": volume[sym],
                    "quote_volume": quote_volume[sym],
                    "close": close[sym],
                },
                index=idx,
            )
        )
        expected_squeeze[sym] = expected["squeeze_prob"]
    expected_residual_inputs = {
        "squeeze_prob": pd.DataFrame(expected_squeeze).astype(np.float32)
    }
    add_residual_features(expected_residual_inputs, None, {"market_basket": cols})

    assert "squeeze_prob" in out
    assert "squeeze_prob_mkt_resid" in out
    for sym in cols:
        actual_latest = float(out["squeeze_prob_mkt_resid"].loc[idx[-1], sym])
        expected_latest = float(
            expected_residual_inputs["squeeze_prob_mkt_resid"].loc[idx[-1], sym]
        )
        assert np.isfinite(actual_latest), sym
        assert np.isclose(actual_latest, expected_latest, rtol=1e-6, atol=1e-7)


def test_materialize_live_perp_aliases_fill_sparse_index_reference_with_spot():
    idx = pd.date_range("2026-01-01", periods=900, freq="1h", tz="UTC")
    cols = ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"]
    x = np.linspace(0.0, 35.0, len(idx), dtype=np.float32)
    close = pd.DataFrame(
        {
            cols[0]: 100.0 + 0.10 * np.sin(x),
            cols[1]: 80.0 + 0.08 * np.cos(x / 2.0),
            cols[2]: 50.0 + 0.07 * np.sin(x / 3.0),
        },
        index=idx,
    ).astype(np.float32)
    spot = close / pd.DataFrame(
        {
            cols[0]: 1.0 + 0.0010 * np.sin(x / 2.0),
            cols[1]: 1.0 + 0.0012 * np.cos(x / 3.0),
            cols[2]: 1.0 + 0.0009 * np.sin(x / 4.0),
        },
        index=idx,
    ).astype(np.float32)
    sparse_mask = pd.DataFrame(False, index=idx, columns=cols)
    sparse_mask.iloc[-120:] = True
    sparse_index = (spot * 1.0001).where(sparse_mask)
    funding = pd.DataFrame(
        {
            cols[0]: 0.00010 * np.sin(x / 3.0),
            cols[1]: 0.00012 * np.cos(x / 4.0),
            cols[2]: 0.00008 * np.sin(x / 5.0),
        },
        index=idx,
    ).astype(np.float32)
    oi = pd.DataFrame(
        {
            cols[0]: 100000.0 + 1000.0 * np.sin(x / 2.0),
            cols[1]: 90000.0 + 800.0 * np.cos(x / 2.5),
            cols[2]: 75000.0 + 700.0 * np.sin(x / 3.0),
        },
        index=idx,
    ).astype(np.float32)
    volume = pd.DataFrame(
        {
            cols[0]: 25.0 + 2.0 * np.sin(x),
            cols[1]: 20.0 + 1.5 * np.cos(x),
            cols[2]: 18.0 + 1.2 * np.sin(x / 2.0),
        },
        index=idx,
    ).astype(np.float32)
    quote_volume = (close * volume).astype(np.float32)
    panel = {
        "close": close,
        "index_price": sparse_index.astype(np.float32),
        "spot_close": spot,
        "funding_rate": funding,
        "open_interest": oi,
        "volume": volume,
        "quote_volume": quote_volume,
    }

    out = _materialize_live_perp_contract_aliases(
        panel,
        cols,
        {},
        {"basis_frac_rank_30d", "squeeze_prob_mkt_resid"},
        cfg={"market_basket": cols},
    )

    expected_squeeze = {}
    reference = spot.combine_first(sparse_index).astype(np.float32)
    for sym in cols:
        expected = compute_perp_features(
            pd.DataFrame(
                {
                    "funding_rate": funding[sym],
                    "open_interest": oi[sym],
                    "open_interest_quote": oi[sym] * close[sym],
                    "perp_price": close[sym],
                    "spot_price": reference[sym],
                    "volume": volume[sym],
                    "quote_volume": quote_volume[sym],
                    "close": close[sym],
                },
                index=idx,
            )
        )
        expected_squeeze[sym] = expected["squeeze_prob"]
        assert np.isfinite(float(out["basis_frac_rank_30d"].loc[idx[-1], sym]))
    expected_residual_inputs = {
        "squeeze_prob": pd.DataFrame(expected_squeeze).astype(np.float32)
    }
    add_residual_features(expected_residual_inputs, None, {"market_basket": cols})

    for sym in cols:
        actual_latest = float(out["squeeze_prob_mkt_resid"].loc[idx[-1], sym])
        expected_latest = float(
            expected_residual_inputs["squeeze_prob_mkt_resid"].loc[idx[-1], sym]
        )
        assert np.isfinite(actual_latest), sym
        assert np.isclose(actual_latest, expected_latest, rtol=1e-6, atol=1e-7)
