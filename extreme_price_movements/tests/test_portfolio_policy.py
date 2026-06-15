import json

import pytest

from extreme_price_movements.inference.portfolio_policy import (
    PortfolioPolicyConfig,
    compute_rank_based_position_size,
    load_portfolio_policy_config,
    validate_portfolio_strategy_contract,
)
from extreme_price_movements.inference.training_live_parity_contract import (
    load_training_live_parity_contract,
    validate_training_live_parity_contract,
)
from extreme_price_movements.portfolio_manager import PortfolioManager


def test_portfolio_policy_defaults_resolve_to_8_and_dynamic_75pct_caps():
    policy = PortfolioPolicyConfig()
    assert policy.max_concurrent_positions == 8
    assert policy.max_concurrent_per_side is None
    assert policy.max_concurrent_per_strategy is None
    assert policy.resolved_max_concurrent_per_side() == 8
    assert policy.resolved_max_concurrent_per_strategy() == 6
    assert policy.max_total_wallet_allocation_pct == 0.75
    assert policy.max_available_wallet_position_pct == 0.50
    assert policy.book_notional_multiplier == 1.0
    assert policy.leverage_wallet_multiplier == 1.0
    assert policy.min_margin_level_after_entry == 2.5
    assert policy.live_test_min_quote_notional == 5.0
    assert policy.live_test_quote_notional == 10.0


def test_portfolio_policy_loads_artifact_before_runtime(tmp_path):
    root = tmp_path / "data"
    path = root / "artifacts" / "RID" / "policy_params"
    path.mkdir(parents=True)
    (path / "optimized_portfolio_policy_config.json").write_text(
        json.dumps({"max_concurrent_positions": 12, "initial_rank_threshold": 0.93})
    )
    policy = load_portfolio_policy_config(
        data_root=str(root),
        run_id="RID",
        runtime_cfg={"max_concurrent_positions": 4},
    )
    assert policy.max_concurrent_positions == 12
    assert policy.initial_rank_threshold == 0.93


def test_portfolio_policy_honors_policy_artifact_root_override(tmp_path, monkeypatch):
    root = tmp_path / "data"
    active = root / "artifacts" / "RID" / "policy_params"
    active.mkdir(parents=True)
    (active / "optimized_portfolio_policy_config.json").write_text(
        json.dumps(
            {
                "max_concurrent_positions": 12,
                "strategy_contract": {"strategy_ids": ["long_stale"]},
            }
        )
    )
    override = tmp_path / "policy_override" / "policy_params"
    override.mkdir(parents=True)
    (override / "optimized_portfolio_policy_config.json").write_text(
        json.dumps(
            {
                "max_concurrent_positions": 4,
                "strategy_contract": {"strategy_ids": ["long_current"]},
            }
        )
    )

    monkeypatch.setenv("EPM_INFERENCE_POLICY_ARTIFACT_ROOT", str(override.parent))
    policy = load_portfolio_policy_config(data_root=str(root), run_id="RID")

    assert policy.max_concurrent_positions == 4
    assert policy.strategy_ids == ("long_current",)


def test_portfolio_policy_uses_override_replay_config_when_policy_params_missing(
    tmp_path, monkeypatch
):
    root = tmp_path / "data"
    active = root / "artifacts" / "RID" / "policy_params"
    active.mkdir(parents=True)
    (active / "optimized_portfolio_policy_config.json").write_text(
        json.dumps({"strategy_contract": {"strategy_ids": ["long_stale"]}})
    )
    override = tmp_path / "policy_override"
    replay = override / "portfolio_policy_replay"
    replay.mkdir(parents=True)
    (replay / "optimized_portfolio_policy_config.json").write_text(
        json.dumps({"strategy_contract": {"strategy_ids": ["long_current"]}})
    )

    monkeypatch.setenv("EPM_INFERENCE_POLICY_ARTIFACT_ROOT", str(override))
    policy = load_portfolio_policy_config(data_root=str(root), run_id="RID")

    assert policy.strategy_ids == ("long_current",)


def test_portfolio_policy_loads_nested_artifact_sections_and_aliases(tmp_path):
    root = tmp_path / "data"
    path = root / "artifacts" / "RID" / "policy_params"
    path.mkdir(parents=True)
    (path / "optimized_portfolio_policy_config.json").write_text(
        json.dumps(
            {
                "rank_sizing": {"rank_multiplier_min": 0.7},
                "selection": {
                    "occupancy_threshold_alpha": 0.3,
                    "occupancy_threshold_power": 1.5,
                    "threshold_viability_margin": 0.02,
                },
                "concurrency": {
                    "max_new_entries_per_bar": 2,
                    "max_concurrent_per_symbol": 1,
                },
                "liquidity": {"max_orderbook_slippage_bps": 40.0},
                "symbol_underperformance_gates_enabled": True,
            }
        )
    )
    policy = load_portfolio_policy_config(data_root=str(root), run_id="RID")
    assert policy.rank_multiplier_min == 0.7
    assert policy.occupancy_threshold_alpha == 0.3
    assert policy.occupancy_threshold_power == 1.5
    assert policy.threshold_viability_margin == 0.02
    assert policy.max_new_entries_per_bar == 2
    assert policy.max_concurrent_per_symbol == 1
    assert policy.max_orderbook_slippage_bps == 40.0
    assert policy.enable_symbol_underperformance_gates is True


def test_portfolio_policy_loads_and_enforces_strategy_contract(tmp_path):
    root = tmp_path / "data"
    path = root / "artifacts" / "RID" / "policy_params"
    path.mkdir(parents=True)
    (path / "optimized_portfolio_policy_config.json").write_text(
        json.dumps(
            {
                "strategy_contract": {
                    "strategy_ids": ["long_alpha", "short_beta"],
                    "strategy_cores": ["alpha", "beta"],
                }
            }
        )
    )

    policy = load_portfolio_policy_config(data_root=str(root), run_id="RID")

    assert policy.strategy_ids == ("long_alpha", "short_beta")
    assert policy.strategy_cores == ("alpha", "beta")
    assert validate_portfolio_strategy_contract(policy, ["short_beta", "long_alpha"])
    with pytest.raises(ValueError, match="Portfolio strategy contract mismatch"):
        validate_portfolio_strategy_contract(policy, ["long_alpha"])


def test_portfolio_policy_does_not_fallback_to_legacy_flat_artifact(tmp_path):
    root = tmp_path / "data"
    path = root / "artifacts" / "RID" / "policy_params"
    path.mkdir(parents=True)
    (path / "portfolio_policy_config.json").write_text(
        json.dumps({"max_concurrent_positions": 12, "initial_rank_threshold": 0.93})
    )

    policy = load_portfolio_policy_config(data_root=str(root), run_id="RID")

    assert policy.max_concurrent_positions == 8
    assert policy.initial_rank_threshold == 0.90


def test_portfolio_policy_can_require_optimized_artifact(tmp_path):
    root = tmp_path / "data"
    (root / "artifacts" / "RID" / "policy_params").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Optimized portfolio policy artifact"):
        load_portfolio_policy_config(
            data_root=str(root),
            run_id="RID",
            require_artifact=True,
        )


def test_training_live_parity_contract_can_be_required_and_validated(tmp_path):
    root = tmp_path / "data"
    path = root / "artifacts" / "RID" / "policy_params"
    path.mkdir(parents=True)
    (path / "training_live_parity_contract.json").write_text(
        json.dumps(
            {
                "schema_version": "training_live_parity_contract_v1",
                "strategy_contract": {
                    "strategy_ids": ["long_alpha", "short_beta"],
                },
            }
        )
    )

    contract = load_training_live_parity_contract(
        data_root=str(root),
        run_id="RID",
        require=True,
    )

    assert contract["_contract_sha256"]
    assert validate_training_live_parity_contract(
        contract,
        active_strategy_ids=["short_beta", "long_alpha"],
    )
    with pytest.raises(ValueError, match="Training-live parity strategy contract"):
        validate_training_live_parity_contract(contract, active_strategy_ids=["long_alpha"])


def test_portfolio_manager_from_policy_config_enforces_caps():
    policy = PortfolioPolicyConfig()
    mgr = PortfolioManager.from_policy_config(policy, portfolio_value=10000.0)
    assert mgr.max_positions == 8
    assert mgr.max_same_side == 8
    assert mgr.max_same_strategy == 6
    assert mgr.max_portfolio_pct == 0.75
    assert mgr.max_position_usdt == 5000.0


def test_rank_based_position_size_caps_and_live_test_override():
    policy = PortfolioPolicyConfig()
    prod = compute_rank_based_position_size(
        wallet_value=100000.0,
        open_notional=0.0,
        adjusted_rank_score=1.0,
        final_threshold=0.90,
        policy=policy,
        live_test_mode=False,
    )
    assert prod["size_after_liquidity"] == 5000.0
    assert prod["book_notional_multiplier"] == 1.0
    assert prod["leverage_wallet_multiplier"] == 1.0
    live_test = compute_rank_based_position_size(
        wallet_value=100000.0,
        open_notional=0.0,
        adjusted_rank_score=1.0,
        final_threshold=0.90,
        policy=policy,
        live_test_mode=True,
    )
    assert live_test["size_after_liquidity"] == 10.0


def test_rank_based_position_size_live_test_minimum_for_positive_size():
    policy = PortfolioPolicyConfig()
    small = compute_rank_based_position_size(
        wallet_value=100.0,
        open_notional=0.0,
        adjusted_rank_score=0.91,
        final_threshold=0.90,
        policy=policy,
        liquidity_capacity_weight=0.25,
        live_test_mode=True,
    )
    assert small["size_after_liquidity"] == 0.0


def test_rank_based_position_size_uses_available_wallet_cap():
    policy = PortfolioPolicyConfig(
        max_position_wallet_pct=1.0,
        max_position_quote_notional=1_000_000.0,
        max_total_wallet_allocation_pct=1.0,
    )
    high_rank = compute_rank_based_position_size(
        wallet_value=10000.0,
        open_notional=6000.0,
        adjusted_rank_score=1.0,
        final_threshold=0.90,
        policy=policy,
        live_test_mode=False,
    )
    assert high_rank["open_equity_allocation"] == 6000.0
    assert high_rank["available_wallet"] == 4000.0
    assert high_rank["available_wallet_position_cap"] == 2000.0
    assert high_rank["size_after_liquidity"] == 2000.0


def test_rank_based_position_size_rank_scales_under_position_cap():
    policy = PortfolioPolicyConfig()
    low_rank = compute_rank_based_position_size(
        wallet_value=10000.0,
        open_notional=0.0,
        adjusted_rank_score=0.90,
        final_threshold=0.90,
        policy=policy,
        live_test_mode=False,
    )
    high_rank = compute_rank_based_position_size(
        wallet_value=10000.0,
        open_notional=0.0,
        adjusted_rank_score=1.0,
        final_threshold=0.90,
        policy=policy,
        live_test_mode=False,
    )
    assert high_rank["rank_scaled_cap"] == 1500.0
    assert high_rank["size_after_liquidity"] == 1500.0
    assert low_rank["size_after_liquidity"] == 750.0
    assert low_rank["size_after_liquidity"] < high_rank["size_after_liquidity"]


def test_perps_rank_sizing_uses_same_dynamic_leverage_in_live_and_live_test():
    policy = PortfolioPolicyConfig()
    prod = compute_rank_based_position_size(
        wallet_value=100.0,
        open_notional=0.0,
        adjusted_rank_score=0.95,
        final_threshold=0.70,
        policy=policy,
        liquidity_capacity_weight=1.0,
        live_test_mode=False,
        market_mode="perps",
        available_wallet_value=100.0,
        stop_loss_pct=0.01,
        rank_number=1,
        rank_x=5,
        orderbook_capacity_quote=1_000.0,
    )
    live_test = compute_rank_based_position_size(
        wallet_value=100.0,
        open_notional=0.0,
        adjusted_rank_score=0.95,
        final_threshold=0.70,
        policy=policy,
        liquidity_capacity_weight=1.0,
        live_test_mode=True,
        market_mode="perps",
        available_wallet_value=100.0,
        stop_loss_pct=0.01,
        rank_number=1,
        rank_x=5,
        orderbook_capacity_quote=1_000.0,
    )

    assert prod["perp_rank_leverage"] == 25.0
    assert prod["perp_effective_leverage"] == 25.0
    assert live_test["perp_rank_leverage"] == prod["perp_rank_leverage"]
    assert live_test["perp_effective_leverage"] == prod["perp_effective_leverage"]
    assert live_test["size_after_liquidity"] == policy.live_test_quote_notional
    assert prod["size_after_liquidity"] > live_test["size_after_liquidity"]


def test_capacity_api_reports_remaining_slots_and_notional():
    mgr = PortfolioManager(portfolio_value=10000.0)
    mgr.record_position_open(
        "BTC/USDC",
        "long",
        "s1",
        position_size=1000.0,
        entry_price=100.0,
    )
    cap = mgr.get_portfolio_capacity(side="long", strategy_id="s1")
    assert cap["open_positions"] == 1
    assert cap["remaining_position_slots"] == 9
    assert cap["remaining_side_slots"] == 5
    assert cap["remaining_strategy_slots"] == 5
    assert cap["remaining_total_notional"] == 6500.0
