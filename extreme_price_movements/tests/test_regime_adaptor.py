from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.regime_adaptor import (
    apply_regime_adaptor,
    fit_regime_adaptor,
    load_regime_adaptor,
    save_regime_adaptor_outputs,
)


def test_regime_adaptor_training_live_parity(tmp_path):
    n = 240
    rng = np.random.RandomState(7)
    ts = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    symbols = np.where(np.arange(n) % 2 == 0, "AAA_USDT", "BBB_USDT")
    frame = pd.DataFrame(
        {
            "rv_24h": rng.lognormal(-4.0, 0.3, n).astype(np.float32),
            "ret1h": rng.normal(0.0, 0.01, n).astype(np.float32),
            "rv_6h": rng.lognormal(-4.2, 0.25, n).astype(np.float32),
            "adx_14": rng.uniform(5, 35, n).astype(np.float32),
            "trend_regime": rng.normal(0.0, 1.0, n).astype(np.float32),
            "dist_ema_fast": rng.normal(0.0, 1.0, n).astype(np.float32),
            "dist_ema_slow": rng.normal(0.0, 1.0, n).astype(np.float32),
            "loc_vwap_dev_z_24": rng.normal(0.0, 1.0, n).astype(np.float32),
            "dist_prior_day_low": rng.normal(0.0, 1.0, n).astype(np.float32),
            "dist_prior_day_high": rng.normal(0.0, 1.0, n).astype(np.float32),
            "rvol_z": rng.normal(0.0, 1.0, n).astype(np.float32),
            "spectral_entropy_ret_24": rng.uniform(0, 1, n).astype(np.float32),
            "volume": rng.lognormal(10.0, 0.4, n).astype(np.float32),
            "atr_pct": rng.lognormal(-4.0, 0.2, n).astype(np.float32),
        }
    )
    pred = pd.Series(rng.normal(0.0, 1.0, n)).rank(pct=True).to_numpy()
    returns = (0.002 * (pred - 0.5) + rng.normal(0.0, 0.01, n)).astype(np.float32)

    fit = fit_regime_adaptor(
        frame,
        pred,
        returns,
        ts,
        symbols,
        strategy_id="long_test_strategy",
        model_name="unit",
    )
    artifact_path = save_regime_adaptor_outputs(
        str(tmp_path), "run", "long_test_strategy", fit
    )
    artifact = load_regime_adaptor(artifact_path)

    training_apply = apply_regime_adaptor(frame, pred, artifact, ts, symbols)
    live_apply = apply_regime_adaptor(frame.copy(), pred.copy(), artifact, ts, symbols)

    assert np.allclose(
        training_apply["regime_weight"],
        live_apply["regime_weight"],
        atol=1e-10,
        rtol=0.0,
    )
    assert np.array_equal(training_apply["eligible"], live_apply["eligible"])
