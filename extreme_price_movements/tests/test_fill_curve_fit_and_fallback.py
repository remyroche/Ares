import numpy as np
import pandas as pd

from extreme_price_movements.tpsl_optimiser import load_step_module


m05 = load_step_module("05_entry_offset_opt.py")


def _mk_trades(n=120, variable_delta=True):
    rng = np.random.default_rng(42)
    signal = np.full(n, 100.0)
    if variable_delta:
        entry = signal * (1.0 - rng.uniform(0.0, 0.02, size=n))
    else:
        entry = signal * 0.99
    filled = rng.uniform(0, 1, size=n) > 0.35
    return pd.DataFrame(
        {
            "entry_price": entry,
            "signal_px": signal,
            "filled_via_limit": filled,
            "reason": np.where(filled, "trailing_stop", "limit_not_filled"),
            "score": rng.normal(0.0, 1.0, size=n),
            "mae_pct": rng.uniform(0.0, 0.03, size=n),
            "mfe_pct": rng.uniform(0.0, 0.05, size=n),
            "duration": rng.integers(1, 20, size=n),
            "atr": np.full(n, 0.02),
        }
    )


def test_fit_fill_model_fits_on_variable_delta():
    df = _mk_trades(variable_delta=True)
    feats = m05.build_policy_features(df)
    model, meta = m05.fit_fill_model(df, feats)
    assert meta["source_quality"] == "fitted"
    assert model["beta_delta"] > 0.0


def test_fit_fill_model_fallback_on_constant_delta():
    df = _mk_trades(variable_delta=False)
    feats = m05.build_policy_features(df)
    model, meta = m05.fit_fill_model(df, feats)
    assert meta["source_quality"].startswith("fallback")
    assert np.isfinite(model["alpha0"])
