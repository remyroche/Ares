import json

import pandas as pd
import pytest

from extreme_price_movements.inference import run_inference as ri
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.inference.safety_switches import StrategyKillSwitch
from extreme_price_movements.portfolio_manager import PortfolioManager


def test_load_normalized_threshold_map_prefers_policy_deployment_rank(tmp_path):
    run_id = "run"
    sizer_dir = tmp_path / "artifacts" / run_id / "simple_position_sizer"
    sizer_dir.mkdir(parents=True)
    (sizer_dir / "normalized_strategy_thresholds.json").write_text(
        json.dumps(
            {
                "threshold_space": "rank_percentile",
                "strategies": {
                    "mr": {
                        "normalized_threshold": 0.59,
                        "viability_margin": 0.01,
                    }
                },
            }
        )
    )
    policy_dir = tmp_path / "artifacts" / run_id / "policy_params"
    policy_dir.mkdir(parents=True)
    (policy_dir / "strategy_for_inference.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "long_mr",
                        "canonical_strategy_id": "mr",
                        "strategy_for_inference": "long_mr",
                        "side": "long",
                        "selected": True,
                        "deployment_rank_threshold": 0.73,
                        "avg_trades_per_day_at_top_1pct": 9.0,
                        "avg_holding_time_hours": 8.0,
                    }
                ]
            }
        )
    )

    thresholds = ri._load_normalized_threshold_map(str(tmp_path), run_id)

    assert thresholds["long_mr"]["threshold_space"] == "rank_percentile"
    assert thresholds["long_mr"]["normalized_threshold"] == 0.73
    assert (
        thresholds["long_mr"]["threshold_scope"] == "per_strategy_prediction_rank_only"
    )
    assert thresholds["mr"]["normalized_threshold"] == 0.73


def test_load_lgbm_strategy_masks_prefers_embedded_strategy_contract(tmp_path):
    run_id = "run"
    policy_dir = tmp_path / "artifacts" / run_id / "policy_params"
    policy_dir.mkdir(parents=True)
    (policy_dir / "strategy_for_inference.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "long_mr",
                        "canonical_strategy_id": "mr",
                        "strategy_for_inference": "long_mr",
                        "side": "long",
                        "selected": True,
                        "lgbm_regime_mask": {
                            "strategy_id": "registry_mr",
                            "trade_side": "long",
                            "base_event_trigger": "price_up|dist_ema_fast|bucket_hi",
                            "mask_params": {
                                "canonical_key": "price_up|dist_ema_fast|bucket_hi",
                                "threshold": 0.2,
                            },
                            "source_target": "price_up",
                            "source_horizon": 12,
                        },
                    }
                ]
            }
        )
    )

    masks = ri._load_lgbm_strategy_mask_rows(str(tmp_path), run_id)

    assert set(masks) >= {"long_mr", "mr"}
    assert masks["long_mr"]["strategy_id"] == "long_mr"
    assert masks["long_mr"]["base_event_trigger"] == "price_up|dist_ema_fast|bucket_hi"
    assert (
        masks["long_mr"]["mask_params"]["canonical_key"]
        == "price_up|dist_ema_fast|bucket_hi"
    )
    assert masks["mr"]["strategy_id"] == "long_mr"


def test_load_lgbm_strategy_masks_fallback_filters_to_selected_strategies(
    tmp_path, monkeypatch
):
    run_id = "run"
    policy_dir = tmp_path / "artifacts" / run_id / "policy_params"
    policy_dir.mkdir(parents=True)
    (policy_dir / "strategy_for_inference.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "long_selected",
                        "canonical_strategy_id": "selected",
                        "strategy_for_inference": "long_selected",
                        "side": "long",
                        "selected": True,
                        "lgbm_regime_mask": {},
                    }
                ]
            }
        )
    )

    from extreme_price_movements.offline_optimisers import params_store

    monkeypatch.setattr(
        params_store,
        "load_inference_candidate_mask_params_per_bucket",
        lambda **_: [
            {
                "strategy_id": "selected",
                "trade_side": "long",
                "base_event_trigger": "selected_rule",
                "mask_params": {"canonical_key": "selected_rule"},
            },
            {
                "strategy_id": "unselected",
                "trade_side": "short",
                "base_event_trigger": "unselected_rule",
                "mask_params": {"canonical_key": "unselected_rule"},
            },
        ],
    )

    masks = ri._load_lgbm_strategy_mask_rows(str(tmp_path), run_id)

    assert {row["strategy_id"] for row in masks.values()} == {"selected"}
    assert set(masks) >= {"selected", "long_selected"}
    assert "unselected" not in masks


def test_lgbm_strategy_mask_coverage_fails_closed_for_missing_selected_strategy():
    with pytest.raises(RuntimeError, match="regime masks missing"):
        ri._validate_lgbm_strategy_mask_coverage(
            {
                "long_present": {
                    "strategy_id": "present",
                    "base_event_trigger": "(*)|(x>0)|(*)",
                }
            },
            {"long_missing"},
        )


def test_lgbm_strategy_mask_coverage_accepts_side_alias_for_selected_strategy():
    ri._validate_lgbm_strategy_mask_coverage(
        {
            "long_selected": {
                "strategy_id": "selected",
                "base_event_trigger": "(*)|(x>0)|(*)",
            }
        },
        {"long_selected"},
    )


def test_lgbm_strategy_mask_coverage_fails_closed_for_missing_trigger():
    with pytest.raises(RuntimeError, match="missing base_event_trigger"):
        ri._validate_lgbm_strategy_mask_coverage(
            {"long_selected": {"strategy_id": "selected"}},
            {"long_selected"},
        )


def test_strategy_asset_exclusion_matches_across_usdt_usdc_quote():
    assert ri._is_symbol_blocked_for_strategy(
        "BTC/USDC",
        "long_mr",
        {"long_mr": {"BTC/USDT"}},
    )


class _DummyOrchestrator:
    bucket_params = {}
    alpha_by_strategy = {
        "long_mr": {"feat_cols": ["dummy"]},
        "short_mr": {"feat_cols": ["dummy"]},
    }

    def _align_alpha_feature_contract(self, features, feat_cols):
        return features.reindex(columns=feat_cols)

    def predict_alpha(self, features, side, kind):
        return pd.Series(0.9, index=features.index, dtype=float)

    def predict_meta(self, features, side, kind):
        return pd.Series(0.9, index=features.index, dtype=float)

    def run_full_chain(self, symbol, side, features, panel=None):
        return {
            "symbol": symbol,
            "side": side,
            "action": "enter",
            "position_size": 9000.0,
            "meta_pred": 0.9,
            "strategy_id": f"{side}_mr",
        }


class _NoEntryOrchestrator:
    bucket_params = {}
    alpha_by_strategy = {"long_mr": {"feat_cols": ["dummy"]}}

    def _align_alpha_feature_contract(self, features, feat_cols):
        return features.reindex(columns=feat_cols)

    def predict_alpha(self, features, side, kind):
        return pd.Series(0.9, index=features.index, dtype=float)

    def predict_meta(self, features, side, kind):
        return pd.Series(0.9, index=features.index, dtype=float)

    def run_full_chain(self, symbol, side, features, panel=None):
        return {
            "symbol": symbol,
            "side": side,
            "action": "no_entry",
            "reason": "entry_policy_rejected",
            "position_size": 0.3,
            "meta_pred": 0.9,
            "strategy_id": f"{side}_mr",
        }


class _DummyExecutor:
    mode = "shadow"

    def __init__(self):
        self.calls = []
        self.config = {"allow_live_batch_rank_fallback_for_debug": True}

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
    # Rank sizing reserves capacity per slot so early positions cannot consume
    # the whole safe book.
    assert executor.calls[0]["size"] <= 1500.0 + 1e-9


def test_run_inference_step_global_auction_executes_best_cross_side_first(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    close = pd.DataFrame(
        {"LONG/USDT": [100.0] * len(idx), "SHORT/USDT": [100.0] * len(idx)},
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

    monkeypatch.setattr(
        ri, "select_candidates", lambda **kwargs: (["LONG/USDT"], ["SHORT/USDT"])
    )
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": [1.0] * len(candidates)}, index=candidates
        ),
    )

    def _gate(decision, **kwargs):
        score = 0.99 if decision["side"] == "short" else 0.80
        decision["policy_rank_pct"] = score
        decision["sizer_rank_percentile"] = score
        decision["threshold_score"] = score
        decision["normalized_rank_score"] = score
        chain = dict(decision.get("chain_results") or {})
        chain["policy_rank_pct"] = score
        chain["sizer_rank_percentile"] = score
        chain["normalized_rank_score"] = score
        decision["chain_results"] = chain
        return True, None

    monkeypatch.setattr(ri, "apply_policy_rank_percentile_gate", _gate)

    executor = _DummyExecutor()
    results = ri.run_inference_step(
        orchestrator=_DummyOrchestrator(),
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=_DummyLogger(),
        accepted_strategies={"long_mr", "short_mr"},
        calibration_data={
            "long_mr": {
                "p75_threshold": 0.5,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            },
            "short_mr": {
                "p75_threshold": 0.5,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            },
        },
        portfolio_mgr=PortfolioManager(portfolio_value=10000.0, max_positions=1),
        initial_rank_threshold=0.5,
        max_entries_total=1,
    )

    assert [call["symbol"] for call in executor.calls] == ["SHORT/USDT"]
    assert results["trades"][0]["side"] == "short"


def test_run_inference_step_rejects_portfolio_strategy_contract_mismatch(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    close = pd.DataFrame({"LONG/USDT": [100.0] * len(idx)}, index=idx)
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (["LONG/USDT"], []))

    with pytest.raises(ValueError, match="Portfolio strategy contract mismatch"):
        ri.run_inference_step(
            orchestrator=_DummyOrchestrator(),
            panel=panel,
            feats=feats,
            thresholds={"metric": "ret12h"},
            executor=_DummyExecutor(),
            logger=_DummyLogger(),
            accepted_strategies={"long_mr"},
            portfolio_policy=ri.PortfolioPolicyConfig(strategy_ids=("short_mr",)),
        )


def test_portfolio_contract_strategy_filter_overrides_manifest_filter():
    policy = ri.PortfolioPolicyConfig(strategy_ids=("long_contract", "short_contract"))

    selected = ri._resolve_portfolio_contract_strategy_filter(
        policy,
        {"long_manifest"},
    )

    assert selected == {"long_contract", "short_contract"}


def test_portfolio_contract_strategy_filter_can_use_cores_without_manifest():
    policy = ri.PortfolioPolicyConfig(strategy_cores=("core_a", "core_b"))

    selected = ri._resolve_portfolio_contract_strategy_filter(policy, set())

    assert selected == {"core_a", "core_b"}


def test_run_inference_step_global_auction_keeps_ticker_liquidity_gate(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    close = pd.DataFrame({"WIDE/USDT": [100.0] * len(idx)}, index=idx)
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (["WIDE/USDT"], []))
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": [1.0] * len(candidates)}, index=candidates
        ),
    )

    class _WideSpreadExchange:
        markets = {"WIDE/USDT": {"limits": {"cost": {"min": 1.0}}}}

        def fetch_ticker(self, symbol):
            return {"bid": 100.0, "ask": 103.0, "last": 101.5}

        def fetch_order_book(self, symbol):
            return {"asks": [[103.0, 100.0]], "bids": [[100.0, 100.0]]}

        def market(self, symbol):
            return self.markets[symbol]

    executor = _DummyExecutor()
    executor.exchange = _WideSpreadExchange()
    executor.config = {"allow_live_batch_rank_fallback_for_debug": True}

    results = ri.run_inference_step(
        orchestrator=_DummyOrchestrator(),
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=_DummyLogger(),
        accepted_strategies={"long_mr"},
        calibration_data={
            "long_mr": {
                "p75_threshold": 0.5,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            }
        },
        portfolio_mgr=PortfolioManager(portfolio_value=10000.0),
        initial_rank_threshold=0.5,
    )

    assert executor.calls == []
    assert results["trades"] == []
    assert results["side_metrics"]["long"]["non_fatal_issues"] >= 1


def test_run_inference_step_blocks_strategy_kill_switch(monkeypatch, tmp_path):
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
    switch = StrategyKillSwitch(
        tmp_path / "strategy_kill_switches.json",
        observe_only=False,
    )
    switch.set_state("long_mr", active=True, reason="weak_hit_rate")
    executor = _DummyExecutor()

    results = ri.run_inference_step(
        orchestrator=_DummyOrchestrator(),
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=_DummyLogger(),
        accepted_strategies={"long_mr"},
        calibration_data={
            "long_mr": {
                "p75_threshold": 0.6,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            }
        },
        portfolio_mgr=PortfolioManager(portfolio_value=10000.0),
        initial_rank_threshold=0.5,
        strategy_kill_switch=switch,
    )

    assert results["trades"] == []
    assert executor.calls == []


def test_run_inference_step_allows_unblocked_strategy_kill_switch(
    monkeypatch, tmp_path
):
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
    switch = StrategyKillSwitch(
        tmp_path / "strategy_kill_switches.json",
        observe_only=False,
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
        calibration_data={
            "long_mr": {
                "p75_threshold": 0.6,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            }
        },
        portfolio_mgr=PortfolioManager(portfolio_value=10000.0),
        initial_rank_threshold=0.5,
        strategy_kill_switch=switch,
    )

    assert len(results["trades"]) == 1
    assert executor.calls, "expected unblocked strategy to continue to execution"


def test_run_inference_step_sizes_from_calibrated_meta_policy_power(monkeypatch):
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

    class _PolicySizingOrchestrator(_DummyOrchestrator):
        bucket_params = {
            "long_mr": {
                "best_size_power": 2.0,
                "asset_metrics": [
                    {
                        "symbol": "BTC/USDT",
                        "asset_decision": "down_weight",
                        "asset_weight_multiplier": 0.5,
                    }
                ],
            }
        }

        def run_full_chain(self, symbol, side, features, panel=None, kind=None):
            out = super().run_full_chain(symbol, side, features, panel=panel)
            out["position_size"] = 9999.0
            out["orchestrator_position_size"] = 9999.0
            return out

    executor = _DummyExecutor()
    portfolio_mgr = PortfolioManager(portfolio_value=10000.0)
    portfolio_policy = ri.PortfolioPolicyConfig()

    results = ri.run_inference_step(
        orchestrator=_PolicySizingOrchestrator(),
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=_DummyLogger(),
        accepted_strategies={"long_mr"},
        calibration_data={
            "long_mr": {
                "p75_threshold": 0.5,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            }
        },
        portfolio_mgr=portfolio_mgr,
        portfolio_policy=portfolio_policy,
    )

    assert len(results["trades"]) == 1
    # With PortfolioManager active, live rank-sizing supersedes legacy
    # calibrated policy sizing and symbol-underperformance downweights. A
    # single-candidate auction rank is 1.0, so the default global-auction policy
    # fills one reserved slot, capped by max_position_wallet_pct.
    expected_slot_size = (
        10000.0
        * portfolio_policy.max_total_wallet_allocation_pct
        / float(
            portfolio_policy.reserved_position_slots
            or portfolio_policy.max_concurrent_positions
        )
    )
    expected_size = min(
        expected_slot_size,
        10000.0 * portfolio_policy.max_position_wallet_pct,
        portfolio_policy.max_position_quote_notional,
    )
    assert abs(executor.calls[0]["size"] - expected_size) < 1e-9


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


def test_run_inference_step_can_force_shadow_entry_for_integration(monkeypatch):
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
    executor.config["force_shadow_entry_for_integration"] = True
    logger = _DummyLogger()

    results = ri.run_inference_step(
        orchestrator=_NoEntryOrchestrator(),
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=logger,
        accepted_strategies={"long_mr"},
        calibration_data={
            "long_mr": {
                "p75_threshold": 0.5,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            }
        },
    )

    assert len(results["trades"]) == 1
    assert executor.calls


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


def test_run_inference_step_ranks_after_policy_asset_exclusions(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    symbols = ["BLOCKED/USDT", "OK/USDT"]
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

    class _ExclusionAwareOrchestrator:
        def __init__(self):
            self.full_chain_symbols = []

        def available_strategies(self, side, accepted=None):
            return ["long_mr"]

        def predict_alpha(self, features, side, kind):
            return pd.Series(
                [0.99 if idx == "BLOCKED/USDT" else 0.5 for idx in features.index],
                index=features.index,
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

    orchestrator = _ExclusionAwareOrchestrator()
    executor = _DummyExecutor()
    results = ri.run_inference_step(
        orchestrator=orchestrator,
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=_DummyLogger(),
        accepted_strategies={"long_mr"},
        strategy_asset_exclusions={"long_mr": {"BLOCKED/USDT"}},
    )

    assert orchestrator.full_chain_symbols == ["OK/USDT"]
    assert [call["symbol"] for call in executor.calls] == ["OK/USDT"]
    assert results["trades"][0]["symbol"] == "OK/USDT"


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
