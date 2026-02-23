import numpy as np
import pandas as pd

from extreme_price_movements.policy_ml import policy_rollout_engine, policy_rollout_ml


def _make_ohlc(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    ret = rng.normal(0.0, 0.003, size=n)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    span = np.abs(rng.normal(0.002, 0.001, size=n))
    high = np.maximum(open_, close) * (1.0 + span)
    low = np.minimum(open_, close) * (1.0 - span)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_policy_rollout_ml_matches_engine_codepath_random_samples():
    ohlc = _make_ohlc()
    atr = pd.Series(0.02, index=ohlc.index)
    policy_params = {
        "tp_mult": 1.0,
        "sl_mult": 0.5,
        "trail_mult": 0.25,
        "vol_lo": 0.03,
        "vol_hi": 0.06,
        "vol_z_max": 3.0,
        "fee_bps": 0.0,
        "be_threshold_pct": 0.005,
        "be_buffer_pct": 0.0,
        "profit_lock_pct": 0.015,
        "profit_lock_amount": 0.003,
        "giveback_pct": 0.005,
        "max_loss_pct": 0.03,
        "kill_a": 0.002,
        "kill_b": 1.5,
        "kill_c": 0.005,
        "kill_min_bars": 2,
        "use_limit_orders": False,
        "use_exit_limits": False,
    }

    rng = np.random.default_rng(42)
    starts = rng.integers(low=1, high=len(ohlc) - 50, size=100)

    for t0 in starts:
        for direction in (1, -1):
            ml = policy_rollout_ml(
                ohlc=ohlc,
                atr_pct=atr,
                t0=int(t0),
                direction=direction,
                policy_params=policy_params,
                max_hold_hours=24,
            )
            eng = policy_rollout_engine(
                ohlc=ohlc,
                atr_pct=atr,
                t0=int(t0),
                direction=direction,
                policy_params=policy_params,
                max_hold_hours=24,
            )

            assert ml.exit_code == eng.exit_code
            assert abs(ml.r_policy - eng.r_policy) <= 1e-12
            assert abs(ml.mae - eng.mae) <= 1e-12
            assert abs(ml.mfe - eng.mfe) <= 1e-12
            assert ml.bars_held == eng.bars_held
