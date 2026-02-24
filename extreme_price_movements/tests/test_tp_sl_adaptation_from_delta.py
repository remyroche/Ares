import numpy as np
import pandas as pd

from extreme_price_movements.tpsl_optimiser import load_step_module


m05 = load_step_module("05_entry_offset_opt.py")


def test_tp_sl_adaptation_uses_delta_and_clips_stop_factor():
    trades = pd.DataFrame({"entry_price": [100.0], "trail_mult": [0.25]})
    pol = pd.DataFrame(
        {
            "u_hat": [1.0],
            "u_hat_z": [1.0],
            "mae_hat": [0.5],
            "mae_hat_z": [1.0],
            "mfe_hat": [0.8],
            "mfe_hat_z": [0.5],
            "dur_hat": [2.0],
            "dur_hat_z": [0.25],
            "signal_px": [100.0],
            "entry_px_fill": [99.0],
            "delta_atr_star": [2.0],
            "delta_price_star": [1.0],
            "p_fill_star": [0.5],
            "eu_star": [0.2],
            "place_order": [True],
            "atr_policy": [0.02],
        }
    )

    out = m05.apply_effective_policy_params(trades, pol)
    assert "sl_distance_atr_eff" in out.columns
    assert "tp_distance_atr_eff" in out.columns
    assert float(out.loc[0, "stop_factor_eff"]) == 0.5  # clipped by formula
    assert float(out.loc[0, "tp_distance_atr_eff"]) >= 0.0
