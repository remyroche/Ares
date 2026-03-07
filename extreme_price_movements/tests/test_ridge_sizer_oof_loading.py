import numpy as np
import pandas as pd

import extreme_price_movements.config as epm_config
from extreme_price_movements.run_ridge_sizer import load_meta_oof_predictions


def test_run_ridge_sizer_loader_backfills_legacy_nan_meta_oof(tmp_path):
    epm_config.load_config = lambda: {}
    run_id = "20260214_190000"
    meta_oof_dir = tmp_path / "artifacts" / run_id / "meta_oof"
    meta_oof_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "oof_pred": [np.nan, 0.3, 0.7, np.nan],
            "index": [0, 1, 2, 3],
            "is_long": [1, 1, 1, 1],
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="1h"),
            "symbol": ["BTC/USDT"] * 4,
            "return": [0.01, -0.02, 0.03, 0.04],
            "oof_u_hat": [np.nan, 0.1, 0.2, np.nan],
            "oof_log_mae_q70_hat": [np.nan, 0.4, 0.5, np.nan],
            "oof_log_mfe_hat": [np.nan, 0.6, 0.8, np.nan],
            "mae_ret": [0.01, 0.02, 0.03, 0.04],
            "mfe_ret": [0.02, 0.03, 0.04, 0.05],
            "u_policy_net": [0.1, 0.2, 0.3, 0.4],
            "exit_code": [0, 1, 0, 2],
        }
    )
    df.to_parquet(meta_oof_dir / "meta_oof_long_mr_reg.parquet", index=False)

    out = load_meta_oof_predictions(str(tmp_path), run_id)
    loaded = out["long_mr"]

    assert "reg" in loaded.columns
    assert np.isfinite(loaded["reg"].values).all()
    assert np.isfinite(loaded["reg_mean"].values).all()
    assert np.isfinite(loaded["oof_u_hat"].values).all()
    assert np.isfinite(loaded["oof_log_mae_q70_hat"].values).all()
    assert np.isfinite(loaded["oof_log_mfe_hat"].values).all()
