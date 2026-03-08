import pandas as pd

from extreme_price_movements.policy_ml import policy_rollout_ml


def _make_trending_ohlc(direction: int, n: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    step = 0.01 if direction > 0 else -0.01
    close = [100.0]
    for _ in range(1, n):
        close.append(close[-1] * (1.0 + step))
    close = pd.Series(close, index=idx, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.002
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.998
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_policy_rollout_ml_uses_sizer_aligned_tp_sl_semantics():
    policy_params = {
        "policy_label_sl_atr_mult": 1.2,
        "policy_label_tp_sl_ratio": 2.0,
        "policy_label_trailing_pct": 0.35,
        "policy_label_max_hold_hours": 24,
    }
    for direction in (1, -1):
        ohlc = _make_trending_ohlc(direction)
        atr = pd.Series(0.02, index=ohlc.index)
        out = policy_rollout_ml(
            ohlc=ohlc,
            atr_pct=atr,
            t0=5,
            direction=direction,
            policy_params=policy_params,
            max_hold_hours=24,
        )

        assert out.exit_code == 2
        assert out.r_policy > 0.0
        assert out.bars_held >= 1
        assert out.mfe >= 0.0
        assert out.mae >= 0.0
