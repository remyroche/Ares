import numpy as np
import pandas as pd

from extreme_price_movements.meta_training.recent_effectiveness_features import (
    add_recent_meta_self_features,
    add_recent_prediction_disagreement_features,
)


def _recent_frame(n=8):
    ts = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "label_available_ts": ts - pd.Timedelta(hours=1),
            "p_meta": np.linspace(0.2, 0.8, n),
            "meta_score": np.linspace(0.1, 0.9, n),
            "y_true": ([0, 0, 1, 1, 0, 1, 1, 1] * ((n // 8) + 1))[:n],
            "y_ret_net": np.linspace(-0.02, 0.03, n),
            "side": ["long"] * n,
            "horizon": [4] * n,
            "bucket": ["b"] * n,
            "symbol": ["BTC/USDT"] * n,
            "base_prob_mr": np.linspace(0.25, 0.75, n),
            "base_prob_tf": np.linspace(0.30, 0.70, n),
        }
    )


def test_recent_meta_self_features_include_brier():
    out = add_recent_meta_self_features(
        _recent_frame(8),
        windows=("3D",),
        min_samples=2,
        min_top_samples=1,
        standardize=False,
    )
    assert "recent_meta_global_brier_3d" in out.columns
    assert out["recent_meta_global_brier_3d"].notna().any()


def test_recent_prediction_disagreement_features_for_3_7_15_days():
    out = add_recent_prediction_disagreement_features(
        _recent_frame(20),
        windows=("3D", "7D", "15D"),
        min_samples=2,
        standardize=False,
    )
    for suffix in ("3d", "7d", "15d"):
        assert f"recent_meta_brier_{suffix}" in out.columns
        assert f"recent_base_meta_disagreement_sub_mean_{suffix}" in out.columns
        assert f"recent_base_meta_disagreement_ratio_mean_{suffix}" in out.columns
        assert f"recent_base_internal_disagreement_std_mean_{suffix}" in out.columns
    assert out["recent_prediction_disagreement_available_3d"].max() == 1
