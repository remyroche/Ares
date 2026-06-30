import argparse

import numpy as np
import pandas as pd

from scripts import run_reliability_blend_optuna as rb


def test_period_spectral_features_are_fold_local_and_timestamp_level():
    train_idx = pd.date_range("2026-05-01", periods=40, freq="h", tz="UTC")
    valid_idx = pd.date_range("2026-05-03", periods=4, freq="h", tz="UTC")
    x = np.linspace(-1.0, 1.0, len(train_idx), dtype=np.float32)
    train = pd.DataFrame(
        {
            "fs__mkt_ret_eq_1h__mean": x,
            "fs__market_breadth_1h__mean": 0.5 * x,
            "fs__rv_24h__mean": np.sin(np.arange(len(train_idx)) / 5.0).astype(np.float32),
            "fs__oi_pressure__mean": np.cos(np.arange(len(train_idx)) / 6.0).astype(np.float32),
        },
        index=train_idx,
    )
    valid = pd.DataFrame(
        {
            "fs__mkt_ret_eq_1h__mean": [2.0, 2.2, 2.4, 2.6],
            "fs__market_breadth_1h__mean": [1.0, 1.1, 1.2, 1.3],
            "fs__rv_24h__mean": [0.1, 0.2, 0.3, 0.4],
            "fs__oi_pressure__mean": [0.4, 0.3, 0.2, 0.1],
        },
        index=valid_idx,
    )
    args = argparse.Namespace(
        period_spectral_features=True,
        period_spectral_lookback=24,
        period_spectral_min_periods=12,
        period_spectral_top_k=3,
        period_spectral_max_features=8,
        period_spectral_shrinkage=0.10,
    )

    train_out, valid_out, diag = rb._append_fold_spectral_position_features(train, valid, args=args)

    spectral_cols = [c for c in train_out.columns if c.startswith("state_spectral_")]
    assert "state_spectral_eig_lambda1_share" in spectral_cols
    assert "state_spectral_pc1_z" in spectral_cols
    assert "state_spectral_top3_mahalanobis" in spectral_cols
    assert diag["spectral_feature_count"] == len(spectral_cols)
    assert diag["spectral_source_feature_count"] >= 2
    assert train_out.index.equals(train_idx)
    assert valid_out.index.equals(valid_idx)
    assert list(valid_out.columns) == list(train_out.columns)
    assert np.isfinite(valid_out[spectral_cols].to_numpy(dtype=np.float32)).all()

