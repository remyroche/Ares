import pandas as pd

from extreme_price_movements.inference import run_inference as ri
from extreme_price_movements.portfolio_manager import PortfolioManager


class _DummyOrchestrator:
    def run_full_chain(self, symbol, side, features, panel=None):
        return {
            "symbol": symbol,
            "side": side,
            "action": "enter",
            "position_size": 9000.0,
            "meta_pred": 0.9,
            "strategy_id": f"{side}_mr",
        }


class _DummyExecutor:
    mode = "shadow"

    def __init__(self):
        self.calls = []

    def get_cooldown_hours(self, bucket_key):
        return 0.0

    def get_active_positions(self):
        return {}

    def execute_trade(self, symbol, side, size, price=None, bucket_key=None):
        self.calls.append(
            {"symbol": symbol, "side": side, "size": size, "bucket_key": bucket_key}
        )
        return {"status": "recorded", "success": True}


class _DummyLogger:
    def __init__(self):
        self.entries = []

    def get_last_trade_timestamp(self, symbol):
        return None

    def log_entry(self, **kwargs):
        self.entries.append(kwargs)


def test_run_inference_step_applies_strategy_rank_and_portfolio_caps(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=10, freq="1h", tz="UTC")
    close = pd.DataFrame({"BTC/USDT": [100.0] * len(idx)}, index=idx)
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {
        "ret12h": close.pct_change().fillna(0.0),
        "ret24h": close.pct_change().fillna(0.0),
        "range_12h_pct": close * 0.0,
        "volatility_zscore": close * 0.0,
    }

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (["BTC/USDT"], []))
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": [1.0] * len(candidates)}, index=candidates
        ),
    )

    orchestrator = _DummyOrchestrator()
    executor = _DummyExecutor()
    logger = _DummyLogger()
    portfolio_mgr = PortfolioManager(portfolio_value=10000.0)

    calibration_data = {
        "long_mr": {
            "p75_threshold": 0.6,
            "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
        }
    }

    results = ri.run_inference_step(
        orchestrator=orchestrator,
        panel=panel,
        feats=feats,
        thresholds={
            "extreme_pct": None,
            "min_move_12h_pct": None,
            "min_range_pct": None,
            "min_vol_zscore": None,
            "metric": "ret12h",
        },
        executor=executor,
        logger=logger,
        accepted_strategies={"long_mr"},
        calibration_data=calibration_data,
        portfolio_mgr=portfolio_mgr,
        initial_rank_threshold=0.5,
    )

    assert len(results["trades"]) == 1
    assert executor.calls, "expected trade execution call"
    # Must cap to portfolio constraints: <= 30% of 10k and <= 5000, so <= 3000
    assert executor.calls[0]["size"] <= 3000.0 + 1e-9


def test_run_inference_step_blocks_non_accepted_strategy(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    close = pd.DataFrame({"BTC/USDT": [100.0, 100.0, 100.0]}, index=idx)
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (["BTC/USDT"], []))
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": [1.0] * len(candidates)}, index=candidates
        ),
    )

    orchestrator = _DummyOrchestrator()
    executor = _DummyExecutor()
    logger = _DummyLogger()

    results = ri.run_inference_step(
        orchestrator=orchestrator,
        panel=panel,
        feats=feats,
        thresholds={
            "extreme_pct": None,
            "min_move_12h_pct": None,
            "min_range_pct": None,
            "min_vol_zscore": None,
            "metric": "ret12h",
        },
        executor=executor,
        logger=logger,
        accepted_strategies={"short_mr"},
        calibration_data={},
    )

    assert not executor.calls
    assert results["trades"] == []
