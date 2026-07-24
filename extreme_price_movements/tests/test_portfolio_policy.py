import json

import pytest

from extreme_price_movements.inference.portfolio_policy import (
    PortfolioPolicyConfig,
    compute_rank_based_position_size,
    load_portfolio_policy_config,
    validate_portfolio_strategy_contract,
)
from extreme_price_movements.inference.dynamic_hr_surprise_threshold import (
    DynamicHrHeadState,
    apply_archetype_hit_surprise_threshold,
    apply_dynamic_hr_surprise_threshold,
    dynamic_hr_state_payload_from_daily_params,
    load_archetype_hit_surprise_policy,
    load_dynamic_hr_surprise_state,
    patch_portfolio_policy_payload_with_dynamic_hr_surprise,
    validate_dynamic_hr_replay_gate,
)
from extreme_price_movements.inference.training_live_parity_contract import (
    load_training_live_parity_contract,
    validate_training_live_parity_contract,
)
from extreme_price_movements.portfolio_manager import PortfolioManager


def test_portfolio_policy_defaults_use_pre_leverage_70pct_capacity():
    policy = PortfolioPolicyConfig()
    assert policy.schema_version == "portfolio_policy_v2"
    assert policy.capacity_mode == "pre_leverage_wallet"
    assert policy.enforce_position_count_cap is False
    assert policy.max_concurrent_positions == 64
    assert policy.max_concurrent_per_side is None
    assert policy.max_concurrent_per_strategy is None
    assert policy.resolved_max_concurrent_per_side() == 64
    assert policy.resolved_max_concurrent_per_strategy() == 64
    assert policy.max_total_wallet_allocation_pct == 0.70
    assert policy.max_available_wallet_position_pct == 0.50
    assert policy.book_notional_multiplier == 1.0
    assert policy.leverage_wallet_multiplier == 1.0
    assert policy.min_margin_level_after_entry == 2.5
    assert policy.min_entry_quote_notional == 3.0
    assert policy.perp_default_leverage == 10.0
    assert policy.live_test_min_quote_notional == 5.0
    assert policy.live_test_quote_notional == 10.0


def test_portfolio_policy_loads_dynamic_hr_surprise_selection_config(tmp_path):
    root = tmp_path / "data"
    path = root / "artifacts" / "RID" / "policy_params"
    path.mkdir(parents=True)
    (path / "optimized_portfolio_policy_config.json").write_text(
        json.dumps(
            {
                "selection": {
                    "dynamic_hr_surprise_enabled": True,
                    "dynamic_hr_surprise_artifact_path": "state.json",
                    "dynamic_hr_surprise_use_deployed_floor": False,
                    "dynamic_hr_surprise_fallback_to_deployed": False,
                    "dynamic_hr_surprise_stale_fallback_to_deployed": True,
                    "dynamic_hr_surprise_max_state_age_days": 3.5,
                    "dynamic_hr_surprise_lower_bound": -0.25,
                    "dynamic_hr_surprise_upper_bound": 1.25,
                }
            }
        )
    )

    policy = load_portfolio_policy_config(data_root=str(root), run_id="RID")

    assert policy.dynamic_hr_surprise_enabled is True
    assert policy.dynamic_hr_surprise_artifact_path == "state.json"
    assert policy.dynamic_hr_surprise_use_deployed_floor is False
    assert policy.dynamic_hr_surprise_fallback_to_deployed is False
    assert policy.dynamic_hr_surprise_stale_fallback_to_deployed is True
    assert policy.dynamic_hr_surprise_max_state_age_days == pytest.approx(3.5)
    assert policy.dynamic_hr_surprise_lower_bound == -0.25
    assert policy.dynamic_hr_surprise_upper_bound == 1.25


def test_portfolio_policy_loads_archetype_hit_surprise_config(tmp_path):
    root = tmp_path / "data"
    path = root / "artifacts" / "RID" / "policy_params"
    path.mkdir(parents=True)
    (path / "optimized_portfolio_policy_config.json").write_text(
        json.dumps(
            {
                "selection": {
                    "archetype_hit_surprise_enabled": True,
                    "archetype_hit_surprise_policy_path": "policy.json",
                    "archetype_hit_surprise_mode": "hit_surprise_priority_rank_50",
                },
                "concurrency": {
                    "max_concurrent_positions": 8,
                    "max_concurrent_per_side": None,
                },
            }
        )
    )

    policy = load_portfolio_policy_config(data_root=str(root), run_id="RID")

    assert policy.archetype_hit_surprise_enabled is True
    assert policy.archetype_hit_surprise_policy_path == "policy.json"
    assert policy.archetype_hit_surprise_mode == "hit_surprise_priority_rank_50"
    assert policy.max_concurrent_per_side is None


def test_archetype_hit_surprise_applies_side_archetype_threshold(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "selection": {"hit_surprise_threshold_enabled": True},
                "archetype_thresholds": [
                    {
                        "side": "short",
                        "strategy_id": "short_s52_meta_threshold_handoff",
                        "policy_archetype": "short__short_mixed_clean_path",
                        "base_rank_threshold": 0.90,
                        "adjusted_rank_threshold": 0.995,
                    }
                ],
            }
        )
    )
    policy = load_archetype_hit_surprise_policy(path)

    result = apply_archetype_hit_surprise_threshold(
        strategy_id="short_s52_meta_threshold_handoff",
        side="short",
        deployed_threshold=0.90,
        policy=policy,
        policy_archetype="short_mixed_clean_path",
        enabled=True,
    )

    assert result.threshold == pytest.approx(0.995)
    assert result.applied is True
    assert result.matched_key == "short__short_mixed_clean_path"


def test_archetype_hit_surprise_applies_priority_rank_adjustment(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "selection": {
                    "hit_surprise_mode": "hit_surprise_priority_rank_50",
                    "hit_surprise_threshold_enabled": False,
                },
                "archetype_adjustments": [
                    {
                        "side": "long",
                        "strategy_id": "long_s52_meta_threshold_handoff",
                        "policy_archetype": "long__long_mixed_wideslow_tentative",
                        "base_rank_threshold": 0.90,
                        "quality_adjustment": 0.04,
                        "portfolio_priority_multiplier": 1.40,
                        "portfolio_priority_adjustment": 0.0,
                        "portfolio_rank_adjustment": 0.02,
                        "actual_hit_rate": 0.92,
                        "expected_hit_rate": 0.84,
                    }
                ],
            }
        )
    )
    policy = load_archetype_hit_surprise_policy(path)

    result = apply_archetype_hit_surprise_threshold(
        strategy_id="long_s52_meta_threshold_handoff",
        side="long",
        deployed_threshold=0.90,
        policy=policy,
        policy_archetype="long_mixed_wideslow_tentative",
        enabled=True,
    )

    assert result.threshold == pytest.approx(0.90)
    assert result.applied is True
    assert result.mode == "priority_rank_50"
    assert result.priority_multiplier == pytest.approx(1.40)
    assert result.priority_adjustment == pytest.approx(0.0)
    assert result.rank_adjustment == pytest.approx(0.02)
    assert result.quality_adjustment == pytest.approx(0.04)
    assert result.actual_hit_rate == pytest.approx(0.92)


def test_archetype_hit_surprise_matches_side_prefixed_policy_strategy_from_core(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "selection": {
                    "hit_surprise_mode": "hit_surprise_priority_rank_50",
                    "hit_surprise_threshold_enabled": False,
                },
                "archetype_adjustments": [
                    {
                        "side": "short",
                        "strategy_id": "short_s52_meta_threshold_handoff",
                        "policy_archetype": "short__short_mixed_clean_path",
                        "portfolio_priority_multiplier": 1.25,
                        "portfolio_rank_adjustment": 0.015,
                    }
                ],
            }
        )
    )
    policy = load_archetype_hit_surprise_policy(path)

    result = apply_archetype_hit_surprise_threshold(
        strategy_id="s52_meta_threshold_handoff",
        side="short",
        deployed_threshold=0.90,
        policy=policy,
        policy_archetype="short__short_mixed_clean_path",
        enabled=True,
    )

    assert result.reason == "applied_priority_rank"
    assert result.matched_key == "short__short_mixed_clean_path"
    assert result.priority_multiplier == pytest.approx(1.25)
    assert result.rank_adjustment == pytest.approx(0.015)


def test_archetype_hit_surprise_applies_threshold_priority_rank_adjustment(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "selection": {
                    "hit_surprise_mode": "hit_surprise_threshold_priority_rank_50",
                    "hit_surprise_threshold_enabled": True,
                },
                "archetype_adjustments": [
                    {
                        "side": "short",
                        "strategy_id": "short_s52_meta_threshold_handoff",
                        "policy_archetype": "short__short_breakout_precision",
                        "base_rank_threshold": 0.90,
                        "adjusted_rank_threshold": 0.85,
                        "threshold_delta": -0.05,
                        "quality_adjustment": 0.05,
                        "portfolio_priority_multiplier": 1.50,
                        "portfolio_priority_adjustment": 0.0,
                        "portfolio_rank_adjustment": 0.025,
                        "actual_hit_rate": 0.93,
                        "expected_hit_rate": 0.82,
                    }
                ],
            }
        )
    )
    policy = load_archetype_hit_surprise_policy(path)

    result = apply_archetype_hit_surprise_threshold(
        strategy_id="short_s52_meta_threshold_handoff",
        side="short",
        deployed_threshold=0.90,
        policy=policy,
        policy_archetype="short_breakout_precision",
        enabled=True,
    )

    assert result.threshold == pytest.approx(0.85)
    assert result.threshold_delta == pytest.approx(-0.05)
    assert result.applied is True
    assert result.mode == "threshold_priority_rank_50"
    assert result.reason == "applied_threshold_priority_rank"
    assert result.priority_multiplier == pytest.approx(1.50)
    assert result.rank_adjustment == pytest.approx(0.025)
    assert result.quality_adjustment == pytest.approx(0.05)


def test_dynamic_hr_surprise_rejected_head_falls_back_to_deployed():
    result = apply_dynamic_hr_surprise_threshold(
        strategy_id="short_asset_alpha",
        deployed_threshold=0.94,
        enabled=True,
        state={
            "short_asset": DynamicHrHeadState(
                head="short_asset",
                guarded_y=1.50,
                dynamic_rejected=True,
                fallback_to_deployed=True,
                reason="robust<=deployed",
            )
        },
    )

    assert result.threshold == 0.94
    assert result.applied is False
    assert result.reason == "robust<=deployed"


def test_dynamic_hr_surprise_can_raise_threshold_above_one():
    result = apply_dynamic_hr_surprise_threshold(
        strategy_id="long_bars_alpha",
        deployed_threshold=0.70,
        enabled=True,
        state={
            "long_bars": DynamicHrHeadState(
                head="long_bars",
                guarded_y=0.90,
                w_lower=0.05,
                w_raise=0.40,
                z_eff=-1.0,
            )
        },
    )

    assert result.threshold == 1.30
    assert result.applied is True


def test_dynamic_hr_surprise_stale_state_falls_back_to_deployed():
    result = apply_dynamic_hr_surprise_threshold(
        strategy_id="short_asset_alpha",
        deployed_threshold=0.70,
        enabled=True,
        max_state_age_days=2.0,
        now="2026-06-28T00:00:00Z",
        state={
            "short_asset": DynamicHrHeadState(
                head="short_asset",
                guarded_y=0.10,
                w_lower=0.10,
                w_raise=0.40,
                z_eff=3.0,
                as_of="2026-06-25T00:00:00Z",
            )
        },
    )

    assert result.threshold == 0.70
    assert result.applied is False
    assert result.reason == "stale_head_state"
    assert result.state_age_days == pytest.approx(3.0)


def test_dynamic_hr_surprise_payload_from_daily_params_keeps_t16_semantics():
    import pandas as pd

    params = pd.DataFrame(
        [
            {
                "day_start": "2026-06-24T00:00:00Z",
                "day_end": "2026-06-25T00:00:00Z",
                "head": "short_asset",
                "guarded_y": 0.67,
                "w_lower": 0.108,
                "w_raise": 0.164,
                "deployed_fixed_threshold": 0.702,
                "recent_validation_guarded": False,
            },
            {
                "day_start": "2026-06-25T00:00:00Z",
                "day_end": "2026-06-25T23:59:59Z",
                "head": "short_asset",
                "guarded_y": 0.71,
                "w_lower": 0.111,
                "w_raise": 0.222,
                "deployed_fixed_threshold": 0.702,
                "recent_validation_guarded": True,
            },
        ]
    )

    payload = dynamic_hr_state_payload_from_daily_params(params)
    head = payload["heads"]["short_asset"]

    assert head["guarded_y"] == pytest.approx(0.71)
    assert head["w_lower"] == pytest.approx(0.111)
    assert head["w_raise"] == pytest.approx(0.222)
    assert head["deployed_threshold"] == pytest.approx(0.702)
    assert head["recent_validation_guarded"] is True


def test_dynamic_hr_surprise_promotion_gate_requires_non_degrading_replay():
    import pandas as pd

    summary = pd.DataFrame(
        [
            {
                "policy": "fixed_deployed_thresholds",
                "total_net_pnl": 1.0,
                "objective": 0.0,
                "q05_rolling_week_pnl": -1.0,
                "q15_rolling_week_pnl": -0.5,
            },
            {
                "policy": "calendar_dynamic_hr_surprise",
                "total_net_pnl": 2.0,
                "objective": 0.5,
                "q05_rolling_week_pnl": -0.2,
                "q15_rolling_week_pnl": -0.1,
            },
        ]
    )

    gate = validate_dynamic_hr_replay_gate(summary)

    assert gate["accepted"] is True
    assert gate["deltas"]["total_net_pnl_delta"] == pytest.approx(1.0)


def test_patch_portfolio_policy_payload_enables_no_floor_t16():
    payload = {"selection": {"global_threshold_floor": 0.60}}

    patched = patch_portfolio_policy_payload_with_dynamic_hr_surprise(
        payload,
        artifact_path="policy_params/dynamic_hr_surprise_t16_state.json",
        enabled=True,
    )

    assert patched["selection"]["global_threshold_floor"] == 0.60
    assert patched["selection"]["dynamic_hr_surprise_enabled"] is True
    assert patched["selection"]["dynamic_hr_surprise_use_deployed_floor"] is False
    assert patched["selection"]["dynamic_hr_surprise_fallback_to_deployed"] is False
    assert (
        patched["selection"]["dynamic_hr_surprise_stale_fallback_to_deployed"] is True
    )


def test_dynamic_hr_surprise_loads_latest_table_row(tmp_path):
    path = tmp_path / "dynamic_hr.csv"
    path.write_text(
        "\n".join(
            [
                "day_start,head,guarded_y,w_lower,w_raise,z_eff,dynamic_rejected,fallback_to_deployed",
                "2026-06-25,long_bars,0.80,0.01,0.20,-1.0,False,False",
                "2026-06-26,long_bars,0.90,0.02,0.30,-2.0,False,False",
            ]
        )
    )

    state = load_dynamic_hr_surprise_state(path)

    assert state["long_bars"].guarded_y == 0.90
    assert state["long_bars"].w_raise == 0.30
    assert state["long_bars"].z_eff == -2.0


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
                    "allocation_threshold_alpha": 0.4,
                    "allocation_threshold_power": 2.0,
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
    assert policy.allocation_threshold_alpha == 0.4
    assert policy.allocation_threshold_power == 2.0
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

    assert policy.max_concurrent_positions == 64
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
        validate_training_live_parity_contract(
            contract, active_strategy_ids=["long_alpha"]
        )


def test_portfolio_manager_from_policy_config_enforces_caps():
    policy = PortfolioPolicyConfig()
    mgr = PortfolioManager.from_policy_config(policy, portfolio_value=10000.0)
    assert mgr.max_positions == 64
    assert mgr.enforce_position_count_cap is False
    assert mgr.max_same_side == 64
    assert mgr.max_same_strategy == 64
    assert mgr.max_portfolio_pct == 0.70
    assert mgr.max_position_usdt == 5000.0


def test_portfolio_manager_preserves_above_one_deactivation_threshold():
    mgr = PortfolioManager(portfolio_value=10000.0)

    assert mgr.calculate_dynamic_threshold(1.25) == 1.25


def test_portfolio_manager_dynamic_threshold_uses_allocated_share():
    mgr = PortfolioManager(
        max_positions=10,
        portfolio_value=10000.0,
        occupancy_threshold_alpha=0.0,
        allocation_threshold_alpha=1.0,
        allocation_threshold_power=1.0,
    )
    mgr.positions["BTC/USD"] = type(
        "OpenPosition",
        (),
        {"is_open": True, "position_size": 7000.0},
    )()

    assert mgr.calculate_dynamic_threshold(0.60) == pytest.approx(0.99)


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


def test_rank_based_position_size_canonical_policy_has_two_x_range():
    policy = PortfolioPolicyConfig(
        rank_multiplier_min=0.75,
        rank_multiplier_max=1.50,
    )
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

    assert high_rank["size_after_liquidity"] / low_rank[
        "size_after_liquidity"
    ] == pytest.approx(2.0)


def test_perps_rank_sizing_uses_same_default_leverage_in_live_and_live_test():
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

    assert prod["perp_rank_leverage"] == 10.0
    assert prod["perp_default_leverage"] == 10.0
    assert prod["perp_effective_leverage"] == 10.0
    assert live_test["perp_rank_leverage"] == prod["perp_rank_leverage"]
    assert live_test["perp_effective_leverage"] == prod["perp_effective_leverage"]
    assert live_test["size_after_liquidity"] == policy.live_test_quote_notional
    assert prod["size_after_liquidity"] > live_test["size_after_liquidity"]


def test_perps_rank_sizing_scales_with_adjusted_rank_above_threshold():
    policy = PortfolioPolicyConfig(max_position_wallet_pct=1.0)
    low_rank = compute_rank_based_position_size(
        wallet_value=100.0,
        open_notional=0.0,
        adjusted_rank_score=0.90,
        final_threshold=0.90,
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
    high_rank = compute_rank_based_position_size(
        wallet_value=100.0,
        open_notional=0.0,
        adjusted_rank_score=1.0,
        final_threshold=0.90,
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

    assert low_rank["rank_excess"] == pytest.approx(0.0)
    assert high_rank["rank_excess"] == pytest.approx(1.0)
    assert low_rank["size_after_liquidity"] < high_rank["size_after_liquidity"]
    assert low_rank["perp_rank_slot_fraction"] < high_rank["perp_rank_slot_fraction"]


def test_perps_rank_sizing_caps_request_to_remaining_total_notional():
    policy = PortfolioPolicyConfig(max_position_wallet_pct=1.0)
    sizing = compute_rank_based_position_size(
        wallet_value=100.0,
        open_notional=58.0,
        adjusted_rank_score=0.99,
        final_threshold=0.70,
        policy=policy,
        liquidity_capacity_weight=1.0,
        live_test_mode=False,
        market_mode="perps",
        available_wallet_value=100.0,
        remaining_total_notional=17.0,
        stop_loss_pct=0.01,
        rank_number=1,
        rank_x=5,
        orderbook_capacity_quote=1_000.0,
    )

    assert sizing["perp_rank_leverage"] == 10.0
    assert sizing["perp_dynamic_rank_notional_cap"] == 1000.0
    assert sizing["max_total_notional"] == 75.0
    assert sizing["remaining_total_notional"] == 17.0
    assert sizing["size_before_liquidity"] == 17.0
    assert sizing["size_after_liquidity"] == 17.0


def test_perps_rank_sizing_applies_wallet_cap_before_leverage():
    policy = PortfolioPolicyConfig(
        max_total_wallet_allocation_pct=1.0,
        max_position_wallet_pct=0.01,
        max_position_quote_notional=1_000_000.0,
        perp_default_leverage=10.0,
    )
    sizing = compute_rank_based_position_size(
        wallet_value=100.0,
        open_notional=0.0,
        adjusted_rank_score=0.99,
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

    assert sizing["perp_effective_leverage"] == 10.0
    assert sizing["configured_book_notional"] == 1000.0
    assert sizing["max_position_wallet_allocation"] == 1.0
    assert sizing["max_position_notional"] == 10.0
    assert sizing["size_before_liquidity"] == 10.0
    assert sizing["size_after_liquidity"] == 10.0


def test_perps_default_leverage_is_capped_by_stop_loss_risk():
    policy = PortfolioPolicyConfig(perp_default_leverage=10.0)
    sizing = compute_rank_based_position_size(
        wallet_value=100.0,
        open_notional=0.0,
        adjusted_rank_score=0.99,
        final_threshold=0.70,
        policy=policy,
        liquidity_capacity_weight=1.0,
        live_test_mode=False,
        market_mode="perps",
        available_wallet_value=100.0,
        stop_loss_pct=0.10,
        rank_number=1,
        rank_x=5,
        orderbook_capacity_quote=1_000.0,
    )

    assert sizing["perp_rank_leverage"] == 10.0
    expected_liquidation_cap = 1.0 / (0.10 + 0.01 + 0.05 + 0.005)
    assert sizing["perp_legacy_risk_cap_leverage"] == pytest.approx(100.0 / 15.0)
    assert sizing["perp_liquidation_risk_cap_leverage"] == pytest.approx(
        expected_liquidation_cap
    )
    assert sizing["perp_risk_cap_leverage"] == pytest.approx(expected_liquidation_cap)
    assert sizing["perp_effective_leverage"] == pytest.approx(expected_liquidation_cap)


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
    assert cap["remaining_position_slots"] == 63
    assert cap["remaining_side_slots"] == 63
    assert cap["remaining_strategy_slots"] == 63
    assert cap["remaining_total_notional"] == 6000.0


def test_marked_notional_wallet_cap_tracks_marks_and_ignores_leverage():
    mgr = PortfolioManager(
        portfolio_value=10_000.0,
        max_portfolio_pct=0.80,
        max_position_pct=1.0,
        max_position_usdt=None,
        leverage_wallet_multiplier=10.0,
    )
    mgr.record_position_open(
        "BTC/USD", "long", "global", position_size=4_000.0, entry_price=100.0
    )
    assert mgr.update_position_mark("BTC/USD", mark_price=110.0)
    cap = mgr.get_portfolio_capacity(side="long", strategy_id="global")

    assert cap["open_marked_notional"] == pytest.approx(4_400.0)
    assert cap["max_total_notional"] == pytest.approx(8_000.0)
    assert cap["remaining_total_notional"] == pytest.approx(3_600.0)
    assert cap["wallet_investment_utilization"] == pytest.approx(0.55)


def test_more_than_eight_positions_are_allowed_below_marked_notional_cap():
    mgr = PortfolioManager(
        portfolio_value=10_000.0,
        max_portfolio_pct=0.80,
        max_position_pct=1.0,
        max_position_usdt=None,
        max_positions=64,
        enforce_position_count_cap=False,
        max_same_side=64,
        max_same_strategy=64,
    )
    for idx in range(9):
        mgr.record_position_open(
            f"ASSET{idx}/USD",
            "long",
            "global",
            position_size=500.0,
            entry_price=100.0,
        )

    allowed, info = mgr.can_enter_position(
        symbol="ASSET9/USD",
        side="long",
        strategy_id="global",
        rank_score=0.99,
        initial_threshold=0.90,
        requested_position_size=500.0,
    )
    assert allowed is True
    assert info["position_size_cap"] == pytest.approx(500.0)


def test_pending_notional_is_included_and_final_candidate_is_clipped():
    mgr = PortfolioManager(
        portfolio_value=10_000.0,
        max_portfolio_pct=0.80,
        max_position_pct=1.0,
        max_position_usdt=None,
    )
    mgr.record_position_open(
        "BTC/USD", "long", "global", position_size=7_000.0, entry_price=100.0
    )
    mgr.reserve_pending_notional("pending-1", 800.0)

    cap = mgr.get_portfolio_capacity(side="long", strategy_id="global")
    assert cap["pending_reserved_notional"] == pytest.approx(800.0)
    assert cap["remaining_total_notional"] == pytest.approx(200.0)
    assert mgr.calculate_position_size_cap(500.0) == pytest.approx(200.0)
