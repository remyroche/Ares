import numpy as np
import pandas as pd

from extreme_price_movements.tpsl_optimiser import load_step_module


m05 = load_step_module("05_entry_offset_opt.py")


def test_entry_policy_eu_argmax_and_place_order_cutoff():
    cfg = m05.EntryOffsetConfig(c_atr=0.1, delta_atr_max=2.0, delta_atr_step=0.5)
    feats = pd.DataFrame(
        {
            "u_hat": [1.0, -1.0],
            "u_hat_z": [1.5, -1.5],
            "mae_hat": [0.1, 0.5],
            "mae_hat_z": [0.2, 1.5],
            "mfe_hat": [0.2, 0.1],
            "mfe_hat_z": [0.1, -0.2],
            "dur_hat": [1.0, 1.0],
            "dur_hat_z": [0.0, 0.0],
            "signal_px": [100.0, 100.0],
            "atr_policy": [0.02, 0.02],
        }
    )
    model = {"alpha0": 0.4, "alpha_u": 0.25, "alpha_mae": 0.25, "beta_delta": 0.4}
    out = m05.choose_entry_offsets(feats, model, cfg)

    assert "delta_atr_star" in out.columns
    assert "eu_star" in out.columns
    assert out.loc[0, "eu_star"] > out.loc[1, "eu_star"]
    assert bool(out.loc[0, "place_order"]) is True
    assert float(out.loc[0, "delta_atr_star"]) >= 0.0
    assert float(out.loc[0, "delta_atr_star"]) <= cfg.delta_atr_max
