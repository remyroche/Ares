import pandas as pd

from extreme_price_movements.inference_backtest import (
    InferenceBacktestConfig,
    run_inference_backtest,
)
from extreme_price_movements.periods_symbols_management import SlicePlannerConfig


def test_inference_backtest_smoke_runs_and_returns_metrics():
    idx = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC")
    close = pd.DataFrame(
        {"BTC/USDT": [100 + i * 0.1 for i in range(len(idx))]}, index=idx
    )
    high = close + 0.2
    low = close - 0.2
    panel = {
        "close": close,
        "high": high,
        "low": low,
        "open": close,
        "volume": close * 10.0,
    }

    feats = {
        "ret12h": close.pct_change(12).fillna(0.0),
        "atr_pct": ((high - low) / close).fillna(0.0),
    }

    t0 = idx[24]
    t1 = idx[30]
    trades = pd.DataFrame(
        {
            "event_id": ["e1"],
            "symbol": ["BTC/USDT"],
            "t0": [t0],
            "t1": [t1],
            "entry_price": [float(close.loc[t0, "BTC/USDT"])],
            "trading_score_oof": [0.9],
            "limit_offset_oof": [0.0],
            "regime_kind": ["tf"],
            "strategy": ["long_mr"],
            "side": ["long"],
        }
    )

    mask_params_by_mode = {
        "price_up_tf": {
            "family": "abs_move_threshold",
            "param": 0.0,
            "z_hours": 1.0,
            "duration_hours": 1.0,
        },
        "price_up_mr": {
            "family": "abs_move_threshold",
            "param": 999.0,
            "z_hours": 1.0,
            "duration_hours": 1.0,
        },
        "price_down_tf": {
            "family": "abs_move_threshold",
            "param": 999.0,
            "z_hours": 1.0,
            "duration_hours": 1.0,
        },
        "price_down_mr": {
            "family": "abs_move_threshold",
            "param": 999.0,
            "z_hours": 1.0,
            "duration_hours": 1.0,
        },
    }

    res = run_inference_backtest(
        trades=trades,
        panel=panel,
        feats=feats,
        mask_params_by_mode=mask_params_by_mode,
        strategy_exit_params={"long_mr": {"sl_mult": 1.0, "trail_mult": 0.2}},
        config=InferenceBacktestConfig(top_fracs=(1.0,)),
        planner_cfg=SlicePlannerConfig.fast_defaults(),
        use_portfolio_manager=False,
        use_strategy_acceptance=False,
        use_calibration_filter=False,
    )

    assert res.get("n_unseen", 0) >= 0
    assert "metrics" in res
