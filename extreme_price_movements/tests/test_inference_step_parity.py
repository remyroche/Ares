import pandas as pd

from extreme_price_movements.inference import run_inference as ri
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
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


def test_model_orchestrator_uses_runtime_model_bundle_when_full_state_is_partial():
    full_state = {
        "bundle": {
            "alpha_models": {"short_old_strategy": {"model": object(), "feat_cols": []}}
        }
    }
    runtime_cfg = {
        "model_bundle": {
            "alpha_models": {
                "short_selected_strategy": {"model": object(), "feat_cols": []}
            }
        }
    }

    orchestrator = ModelOrchestrator(full_state, runtime_cfg)

    assert "short_selected_strategy" in orchestrator.alpha_by_strategy
    assert "short_old_strategy" not in orchestrator.alpha_by_strategy


def test_model_orchestrator_calls_ridge_sizer_with_named_dataframe():
    class _Sizer:
        model_names_ = ["meta_pred", "calibrated_reg_pred"]

        def __init__(self):
            self.seen_columns = None

        def predict(self, frame):
            self.seen_columns = list(frame.columns)
            return [0.12]

    sizer = _Sizer()
    orchestrator = ModelOrchestrator({"bundle": {}, "ridge_sizer": sizer}, {})
    features = pd.DataFrame({"meta_pred": [0.8]}, index=["BTC/USDT"])

    position_size, _ = orchestrator.compute_ridge_position_size(
        features, side="long", kind="long_mr"
    )

    assert position_size.iloc[0] == 0.12
    assert "calibrated_reg_pred" in sizer.seen_columns


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


def test_portfolio_manager_hard_gates_require_manual_reset():
    portfolio_mgr = PortfolioManager(portfolio_value=10000.0)
    now = pd.Timestamp("2026-03-01 00:00", tz="UTC")
    for i in range(5):
        symbol = f"LOSS{i}/USDT"
        portfolio_mgr.record_position_open(
            symbol=symbol,
            side="long",
            strategy_id="long_mr",
            position_size=100.0,
            entry_price=100.0,
            entry_time=now + pd.Timedelta(minutes=i),
        )
        portfolio_mgr.record_position_close(
            symbol=symbol,
            exit_price=99.0,
            exit_time=now + pd.Timedelta(minutes=i, seconds=1),
            exit_reason="test_loss",
        )

    allowed, info = portfolio_mgr.can_enter_position(
        symbol="NEXT/USDT",
        side="long",
        strategy_id="long_mr",
        confidence_score=1.0,
        initial_threshold=0.5,
        current_time=now + pd.Timedelta(minutes=10),
        requested_position_size=100.0,
    )
    assert not allowed
    assert info["hard_limits"]["manual_reset_required"]
    assert "consecutive_losing_trades" in info["hard_limits"]["hard_limit_reason"]

    portfolio_mgr.manual_reset_hard_limits()
    allowed_after_reset, _ = portfolio_mgr.can_enter_position(
        symbol="NEXT/USDT",
        side="long",
        strategy_id="long_mr",
        confidence_score=1.0,
        initial_threshold=0.5,
        current_time=now + pd.Timedelta(minutes=11),
        requested_position_size=100.0,
    )
    assert allowed_after_reset


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


def test_run_inference_step_blocks_policy_excluded_asset(monkeypatch):
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

    executor = _DummyExecutor()
    results = ri.run_inference_step(
        orchestrator=_DummyOrchestrator(),
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=_DummyLogger(),
        accepted_strategies={"long_mr"},
        strategy_asset_exclusions={"long_mr": {"BTC/USDT"}},
    )

    assert not executor.calls
    assert results["trades"] == []


def test_run_inference_step_gates_meta_to_top_quartile_base_preds(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    symbols = ["A/USDT", "B/USDT", "C/USDT", "D/USDT"]
    close = pd.DataFrame(
        {symbol: [100.0, 100.0, 100.0] for symbol in symbols},
        index=idx,
    )
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (symbols, []))
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": list(range(len(candidates)))}, index=candidates
        ),
    )

    class _GatedOrchestrator:
        def __init__(self):
            self.full_chain_symbols = []

        def available_strategies(self, side, accepted=None):
            return ["long_mr"]

        def predict_alpha(self, features, side, kind):
            return pd.Series(
                [0.1, 0.9, 0.2, 0.3],
                index=["A/USDT", "B/USDT", "C/USDT", "D/USDT"],
            )

        def run_full_chain(self, symbol, side, features, panel=None, kind=None):
            self.full_chain_symbols.append(symbol)
            return {
                "symbol": symbol,
                "side": side,
                "action": "enter",
                "position_size": 100.0,
                "meta_pred": 0.9,
                "strategy_id": "long_mr",
            }

    orchestrator = _GatedOrchestrator()
    executor = _DummyExecutor()
    logger = _DummyLogger()

    results = ri.run_inference_step(
        orchestrator=orchestrator,
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=logger,
        accepted_strategies={"long_mr"},
        calibration_data={},
    )

    assert orchestrator.full_chain_symbols == ["B/USDT"]
    assert [call["symbol"] for call in executor.calls] == ["B/USDT"]
    assert results["trades"][0]["symbol"] == "B/USDT"


def test_trade_execution_health_records_rejections_and_api_failures():
    portfolio_mgr = PortfolioManager(max_consecutive_order_rejections=5)

    ri._record_trade_execution_health(
        portfolio_mgr,
        {
            "success": False,
            "error_category": "duplicate_client_order_id",
            "error": "Duplicate clientOrderId was sent",
        },
    )
    assert portfolio_mgr.consecutive_order_rejections == 1
    assert portfolio_mgr.order_rejection_backoff_until is not None

    ri._record_trade_execution_health(
        portfolio_mgr,
        {
            "success": False,
            "error_category": "network_timeout",
            "error": "network timeout while sending order",
        },
    )
    assert len(portfolio_mgr.failed_api_events) == 1
