import numpy as np

from extreme_price_movements import lgbm_pipeline as lp


def test_vol_normalized_tpsl_precision_metrics_use_requested_top_fracs():
    pred = np.arange(10, dtype=np.float64)
    vol = np.full(10, 0.01, dtype=np.float64)
    mfe = np.zeros(10, dtype=np.float64)
    mae = np.full(10, 0.03, dtype=np.float64)
    mfe[7:] = 0.04
    mae[7:] = 0.005

    metrics = lp._vol_normalized_tp_sl_precision_metrics(
        pred,
        {"mfe": mfe, "mae": mae, "barrier_pct": vol},
    )

    assert metrics["vol_norm_tpsl_metrics_available"] == 1.0
    assert metrics["baseline_tp3_sl2_vol_norm"] == 0.3
    assert metrics["precision_at_30_tp3_sl2_vol_norm"] == 1.0
    assert metrics["precision_at_20_tp3_sl2_vol_norm"] == 1.0
    assert metrics["precision_at_10_tp3_sl2_vol_norm"] == 1.0
    assert metrics["precision_at_30_tp2_sl1_vol_norm"] == 1.0
    assert metrics["precision_at_20_tp2_sl1_vol_norm"] == 1.0
    assert metrics["precision_at_10_tp2_sl1_vol_norm"] == 1.0


def test_vol_normalized_tpsl_precision_metrics_use_hit_order_when_available():
    pred = np.arange(10, dtype=np.float64)
    vol = np.full(10, 0.01, dtype=np.float64)
    mfe = np.zeros(10, dtype=np.float64)
    mae = np.zeros(10, dtype=np.float64)
    mfe[-1] = 0.04
    mae[-1] = 0.03

    no_timing = lp._vol_normalized_tp_sl_precision_metrics(
        pred,
        {"mfe": mfe, "mae": mae, "barrier_pct": vol},
    )
    with_timing = lp._vol_normalized_tp_sl_precision_metrics(
        pred,
        {
            "mfe": mfe,
            "mae": mae,
            "barrier_pct": vol,
            "tau_tp": np.r_[np.full(9, np.nan), 2.0],
            "tau_sl": np.r_[np.full(9, np.nan), 5.0],
        },
    )

    assert no_timing["precision_at_10_tp3_sl2_vol_norm"] == 0.0
    assert with_timing["precision_at_10_tp3_sl2_vol_norm"] == 1.0
