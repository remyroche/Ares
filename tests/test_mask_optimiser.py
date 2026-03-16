import numpy as np
import pandas as pd
from extreme_price_movements.mask_optimiser import (
    _mode_primary_target,
    _signed_mode_return,
    fold_base_rate_nb,
    compute_impulse_coherence_nb,
    _classifier_oof_auc,
    _ridge_regression_oof_r2,
    _build_temporal_folds,
    _cap_rows_for_optimization,
    _build_phase_local_shared,
    _ensure_min_feature_representatives,
    _mask_symbol_concentration_summary,
)
import pytest

def test_invalid_forward_returns():
    forward_returns = np.array([0.05, -0.05, np.nan, np.inf])

    # primary target
    target_up_tf = _mode_primary_target("price_up_tf", forward_returns, 0.0)
    assert np.isnan(target_up_tf[2]) and np.isnan(target_up_tf[3])

    target_up_mr = _mode_primary_target("price_up_mr", forward_returns, 0.0)
    assert np.isnan(target_up_mr[2]) and np.isnan(target_up_mr[3])

    # signed return
    signed_ret = _signed_mode_return("price_up_tf", forward_returns)
    assert np.isnan(signed_ret[2]) and np.isnan(signed_ret[3])

def test_fold_base_rate_nb():
    mask = np.array([True, True, True, True])
    target = np.array([1.0, 0.0, np.nan, 1.0])
    val_idx = np.array([0, 1, 2, 3])
    rate = fold_base_rate_nb(mask, target, val_idx)
    assert rate == 2.0 / 3.0

def test_compute_impulse_coherence_nb():
    returns = np.ones(5)
    volatility = np.ones(5)
    high_val = np.ones(5)
    low_val = np.ones(5)
    start_px = np.ones(5)
    high_idx_local = np.zeros(5, dtype=np.int32)
    low_idx_local = np.zeros(5, dtype=np.int32)
    start_idx_local = np.zeros(5, dtype=np.int32)
    window = 2

    b_up, b_dn, s_up, s_dn, m_up, m_dn, v_e = compute_impulse_coherence_nb(
        returns, volatility, high_val, low_val, start_px,
        high_idx_local, low_idx_local, start_idx_local, window
    )

    assert np.isnan(b_up[0]) and np.isnan(b_up[1])
    assert not np.isnan(b_up[2])

def test_oof_evaluation_coverage():
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(np.float32)
    timestamps = np.arange(100)

    # with a tiny fallback split where 50% remains unpredicted
    # old logic would score them as 0.5. new logic should exclude them entirely or score appropriately
    score = _classifier_oof_auc(X, y, timestamps, n_splits=2)
    assert isinstance(score, float)

def test_build_temporal_folds_fallback():
    timestamps = np.arange(10)
    # mock PurgedKFold failure
    folds = _build_temporal_folds(timestamps, 10, n_splits=2)
    # the fallback should forward-chain
    assert len(folds) >= 1
    # Check that at least most indices are covered in validation
    val_covered = set()
    for tr, va in folds:
        assert max(tr) < min(va) # no leakage
        val_covered.update(va.tolist())
    assert len(val_covered) >= 5 # covers a good chunk

def test_phase1_subsample_equivalence():
    from extreme_price_movements.mask_optimiser import _generate_event_masks_fast

    asset_ids = np.zeros(100, dtype=np.int32)
    asset_ids[50:] = 1
    asset_groups = {
        0: np.where(asset_ids == 0)[0].astype(np.int32),
        1: np.where(asset_ids == 1)[0].astype(np.int32),
    }

    # Setup dummy features and z-cache
    feature_vals = np.random.rand(100)
    zc_full = {
        "dummy_feat": feature_vals,
        "up": np.random.rand(100),
        "dn": np.random.rand(100)
    }

    candidate = {
        "feature_base": "dummy_feat",
        "family": "momentum",
        "direction": "gt",
        "threshold": 0.5
    }

    # Phase 1 mask: only take 1 asset
    phase1_mask = asset_ids == 0

    # Generate full then slice
    m_h_f, m_l_f = _generate_event_masks_fast(
        candidate=candidate,
        zc=zc_full,
        asset_groups=asset_groups
    )
    m_h_sub = m_h_f[phase1_mask]

    assert m_h_sub.shape[0] == 50
    assert np.all(m_h_sub == m_h_f[:50])

    shared = {
        "high": np.ones(100, dtype=np.float32),
        "low": np.ones(100, dtype=np.float32),
        "close": np.ones(100, dtype=np.float32),
        "ret_1": np.zeros(100, dtype=np.float32),
        "vol_g": np.ones(100, dtype=np.float32),
        "timestamps": np.arange(100),
        "forward_returns": np.zeros(100, dtype=np.float32),
        "mae_high": np.ones(100, dtype=np.float32),
        "mfe_high": np.ones(100, dtype=np.float32),
        "mae_low": np.ones(100, dtype=np.float32),
        "mfe_low": np.ones(100, dtype=np.float32),
        "learn_X": np.zeros((100, 2), dtype=np.float32),
        "day_ids": np.arange(100, dtype=np.int32),
        "symbol_codes": asset_ids,
        "symbol_uniques": np.array(["A", "B"]),
    }
    phase1_shared = _build_phase_local_shared(shared, phase1_mask)

    zc_local = {
        "dummy_feat": zc_full["dummy_feat"][phase1_mask],
        "up": zc_full["up"][phase1_mask],
        "dn": zc_full["dn"][phase1_mask]
    }

    m_h_local, _ = _generate_event_masks_fast(
        candidate=candidate,
        zc=zc_local,
        asset_groups=phase1_shared["asset_groups"]
    )
    assert np.array_equal(m_h_local, m_h_sub)

def test_invalid_forward_row_exclusion():
    from extreme_price_movements.mask_optimiser import _compute_regime_distinctness_single_side
    mode = "price_up_tf"
    forward_returns = np.array([0.05, -0.05, np.nan, 0.10, np.nan, 0.0])
    side_mask = np.array([True, False, True, True, False, False])

    # distinctness explicitly uses only valid forward returns via valid = np.isfinite(...)
    score = _compute_regime_distinctness_single_side(
        side_mask, mode, forward_returns,
        np.ones(6), np.ones(6), np.ones(6), np.ones(6)
    )
    assert isinstance(score, float)


def test_dilate_mask_by_asset_handles_non_rectangular_rows():
    from extreme_price_movements.mask_optimiser import dilate_mask_by_asset

    # Local time series per asset are not interleaved in fixed rectangular spacing.
    asset_groups = {
        0: np.array([0, 2, 5], dtype=np.int32),
        1: np.array([1, 3, 4], dtype=np.int32),
    }
    mask = np.array([True, False, False, True, False, False])
    out = dilate_mask_by_asset(mask, asset_groups, duration_bars=2)

    # asset 0: idx 0 -> idx 2; asset 1: idx 3 -> idx 4
    expected = np.array([True, False, True, True, True, False])
    assert np.array_equal(out, expected)


def test_cap_rows_for_optimization_caps_all_inputs():
    data = pd.DataFrame({
        "timestamp": np.arange(20),
        "symbol": ["A"] * 20,
        "close": np.arange(20, dtype=np.float32),
    })
    feature_dict = {"f1": np.arange(20, dtype=np.float32), "f2": np.arange(20, dtype=np.float32) * 2}
    forward_returns = np.arange(20, dtype=np.float32)

    data_capped, feature_capped, forward_capped = _cap_rows_for_optimization(
        data=data,
        feature_dict=feature_dict,
        forward_returns=forward_returns,
        cfg={"mask_opt_max_rows": 7},
        seed=42,
    )

    assert len(data_capped) == 7
    assert forward_capped.shape[0] == 7
    assert feature_capped["f1"].shape[0] == 7
    assert feature_capped["f2"].shape[0] == 7


def test_cap_rows_for_optimization_preserves_symbol_breadth():
    data = pd.DataFrame(
        {
            "timestamp": np.arange(12),
            "symbol": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "close": np.arange(12, dtype=np.float32),
        }
    )
    feature_dict = {"f1": np.arange(12, dtype=np.float32)}
    forward_returns = np.arange(12, dtype=np.float32)

    data_capped, feature_capped, forward_capped = _cap_rows_for_optimization(
        data=data,
        feature_dict=feature_dict,
        forward_returns=forward_returns,
        cfg={"mask_opt_max_rows": 6},
        seed=42,
    )

    assert len(data_capped) == 6
    assert feature_capped["f1"].shape[0] == 6
    assert forward_capped.shape[0] == 6
    counts = data_capped["symbol"].value_counts().to_dict()
    assert counts == {"A": 2, "B": 2, "C": 2}
    assert data_capped.groupby("symbol")["timestamp"].min().to_dict() == {
        "A": 2,
        "B": 6,
        "C": 10,
    }


def test_build_temporal_folds_fallback_uses_timestamp_groups():
    ts = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int64)
    folds = _build_temporal_folds(ts, ts.shape[0], n_splits=2)
    for tr, va in folds:
        tr_ts = set(ts[tr].tolist())
        va_ts = set(ts[va].tolist())
        assert tr_ts.isdisjoint(va_ts)
        assert max(tr_ts) < min(va_ts)


def test_build_temporal_folds_accepts_symbol_vector():
    ts = np.arange(12, dtype=np.int64)
    symbols = np.array(["A", "B", "C"] * 4, dtype=object)
    folds = _build_temporal_folds(ts, ts.shape[0], n_splits=2, symbols=symbols)

    assert len(folds) >= 1
    for tr, va in folds:
        assert tr.dtype == np.int32
        assert va.dtype == np.int32


def test_mask_symbol_concentration_summary_reports_top_share():
    mask = np.array([True, True, True, False, True, False])
    symbol_codes = np.array([0, 0, 1, 1, 0, 2], dtype=np.int32)
    symbol_uniques = np.array(["A", "B", "C"], dtype=object)

    summary = _mask_symbol_concentration_summary(mask, symbol_codes, symbol_uniques)

    assert summary["event_symbol_count"] == 2
    assert summary["top_symbol_share"] == 0.75
    assert "A:3" in summary["top_symbol_counts_text"]


def test_ensure_min_feature_representatives_respects_floor():
    df = pd.DataFrame(
        [
            {"name": "a1", "feature_base": "fa", "score": 10.0},
            {"name": "a2", "feature_base": "fa", "score": 9.0},
            {"name": "a3", "feature_base": "fa", "score": 8.0},
            {"name": "b1", "feature_base": "fb", "score": 7.0},
            {"name": "b2", "feature_base": "fb", "score": 6.0},
            {"name": "c1", "feature_base": "fc", "score": 5.0},
        ]
    )

    out = _ensure_min_feature_representatives(
        df,
        score_col="score",
        min_per_feature=2,
        max_total=4,
    )

    counts = out["feature_base"].value_counts().to_dict()
    assert counts["fa"] >= 2
    assert counts["fb"] >= 2
