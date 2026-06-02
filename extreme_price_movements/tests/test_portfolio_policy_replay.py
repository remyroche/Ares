import json

import pandas as pd

from extreme_price_movements.portfolio_policy_replay import (
    DEFAULT_OFFLINE_PRICE_GAP_BPS,
    PortfolioPolicyParams,
    load_portfolio_policy_params,
    normalise_candidate_table,
    portfolio_policy_params_from_live_config,
    replay_candidates,
    run_portfolio_policy_replay,
)
from extreme_price_movements.inference.portfolio_policy import (
    load_portfolio_policy_config,
)


def _candidate(
    ts: str,
    symbol: str,
    side: str,
    strategy_id: str,
    rank: float,
    *,
    threshold: float = 0.60,
    net_return: float = 0.01,
    holding_bars: int = 4,
) -> dict:
    timestamp = pd.Timestamp(ts, tz="UTC")
    entry = 100.0
    gross_return = net_return + 0.001
    signed_exit = 1.0 + (gross_return if side == "long" else -gross_return)
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "side": side,
        "strategy_id": strategy_id,
        "normalized_rank_score": rank,
        "base_strategy_threshold": threshold,
        "calibrated_score": rank,
        "entry_price": entry,
        "exit_timestamp": timestamp + pd.Timedelta(minutes=15 * holding_bars),
        "exit_price": entry * signed_exit,
        "net_return": net_return,
        "gross_return": gross_return,
        "fees_bps": 10.0,
        "slippage_bps": 0.0,
        "holding_bars": holding_bars,
        "simple_policy_exit_reason": "trailing" if net_return > 0 else "full_sl",
        "price_gap_bps": 0.0,
        "expected_friction_bps": 10.0,
        "liquidity_capacity_weight": 1.0,
        "market_mode": "spot",
    }


def test_global_auction_uses_one_priority_queue_without_long_first_bias():
    rows = pd.DataFrame(
        [
            _candidate("2026-01-01 00:00", "BTC/USD", "long", "L_a", 0.80),
            _candidate("2026-01-01 00:00", "ETH/USD", "short", "S_a", 0.95),
        ]
    )
    params = PortfolioPolicyParams(
        max_concurrent_positions=1,
        max_concurrent_per_side=None,
        max_concurrent_per_strategy=None,
        max_new_entries_per_bar=1,
        global_threshold_floor=0.50,
        threshold_viability_margin=0.0,
        min_position_size=0.01,
    )

    global_decisions, _, _ = replay_candidates(rows, params, mode="global_auction")
    baseline_decisions, _, _ = replay_candidates(rows, params, mode="live_baseline")

    assert global_decisions.loc[global_decisions["accepted"], "symbol"].tolist() == [
        "ETH/USD"
    ]
    assert baseline_decisions.loc[
        baseline_decisions["accepted"], "symbol"
    ].tolist() == ["BTC/USD"]


def test_dynamic_threshold_increases_with_occupancy():
    params = PortfolioPolicyParams(
        max_concurrent_positions=2,
        global_threshold_floor=0.50,
        occupancy_threshold_alpha=0.50,
        occupancy_threshold_power=1.0,
    )
    empty_decisions, _, _ = replay_candidates(
        pd.DataFrame(
            [
                _candidate(
                    "2026-01-01 00:00",
                    "BTC/USD",
                    "long",
                    "L_a",
                    0.90,
                    threshold=0.50,
                )
            ]
        ),
        params,
    )
    assert empty_decisions["dynamic_threshold"].iloc[0] == 0.50

    rows = pd.DataFrame(
        [
            _candidate(
                "2026-01-01 00:00", "BTC/USD", "long", "L_a", 0.95, holding_bars=8
            ),
            _candidate(
                "2026-01-01 00:15", "ETH/USD", "long", "L_a", 0.70, holding_bars=8
            ),
        ]
    )
    decisions, _, _ = replay_candidates(rows, params, mode="global_auction")
    second = decisions[decisions["symbol"] == "ETH/USD"].iloc[0]
    assert second["dynamic_threshold"] > 0.50


def test_replay_applies_loss_cooldown_before_reentry():
    rows = pd.DataFrame(
        [
            _candidate(
                "2026-01-01 00:00",
                "BTC/USD",
                "long",
                "L_a",
                0.95,
                net_return=-0.02,
                holding_bars=1,
            ),
            _candidate("2026-01-01 00:30", "BTC/USD", "long", "L_a", 0.96),
        ]
    )
    params = PortfolioPolicyParams(
        max_concurrent_positions=2,
        max_concurrent_per_side=None,
        max_concurrent_per_strategy=None,
        global_threshold_floor=0.50,
        threshold_viability_margin=0.0,
        min_position_size=0.01,
        cooldown_hours_after_loss=24.0,
    )

    decisions, _, _ = replay_candidates(rows, params, mode="global_auction")

    assert bool(decisions.iloc[0]["accepted"]) is True
    assert decisions.iloc[1]["rejection_reason"] == "symbol_in_cooldown"


def test_replay_defaults_missing_offline_price_gap_to_50_bps():
    row = _candidate("2026-01-01 00:00", "BTC/USD", "long", "L_a", 0.95)
    row.pop("price_gap_bps")

    normalised = normalise_candidate_table(pd.DataFrame([row]))

    assert normalised["price_gap_bps"].iloc[0] == DEFAULT_OFFLINE_PRICE_GAP_BPS


def test_portfolio_policy_params_round_trip_live_config(tmp_path):
    params = PortfolioPolicyParams(
        max_concurrent_positions=7,
        max_concurrent_per_side=None,
        max_concurrent_per_strategy=3,
        max_new_entries_per_bar=2,
        max_total_wallet_allocation_pct=0.8,
        global_threshold_floor=0.45,
        threshold_viability_margin=0.05,
        occupancy_threshold_alpha=0.1,
        occupancy_threshold_power=2.0,
        rank_size_power=2.2,
        rank_multiplier_min=0.5,
        rank_multiplier_max=1.0,
        max_signal_gap_bps=150.0,
        strategy_ids=("long_a", "short_b"),
        strategy_cores=("a", "b"),
    )
    payload = params.to_live_config()
    loaded = portfolio_policy_params_from_live_config(payload)

    assert loaded == params

    path = tmp_path / "optimized_portfolio_policy_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_portfolio_policy_params(path) == params


def test_run_portfolio_policy_replay_writes_contract_and_live_loader_reads_it(tmp_path):
    data_root = tmp_path / "data"
    run_id = "test-run"
    run_root = data_root / "artifacts" / run_id
    candidate_dir = run_root / "simple_policy_optimiser"
    candidate_dir.mkdir(parents=True)
    candidate_path = candidate_dir / "simple_policy_candidates.parquet"
    rows = pd.DataFrame(
        [
            _candidate("2026-01-01 00:00", "BTC/USD", "long", "L_a", 0.95),
            _candidate("2026-01-01 01:00", "ETH/USD", "short", "S_a", 0.94),
            _candidate("2026-01-02 00:00", "SOL/USD", "long", "L_a", 0.93),
            _candidate("2026-01-03 00:00", "XRP/USD", "short", "S_a", 0.92),
        ]
    )
    rows.to_parquet(candidate_path, index=False)

    report = run_portfolio_policy_replay(
        data_root=str(data_root),
        run_id=run_id,
        candidate_path=candidate_path,
        max_evaluations=3,
    )

    assert report["generated_by"] == "portfolio_policy_replay"
    assert report["baseline_diagnostics"]["position_count_path_error"] is not None
    assert report["baseline_diagnostics"]["notional_path_error"] is not None
    assert report["baseline_diagnostics"]["wallet_path_error"] is not None
    assert (
        run_root / "portfolio_policy_replay" / "per_candidate_replay_decisions.parquet"
    ).exists()
    config_path = run_root / "policy_params" / "optimized_portfolio_policy_config.json"
    assert config_path.exists()
    payload = json.loads(config_path.read_text())
    assert payload["portfolio_policy_version"] == "global_auction_v1"
    assert payload["strategy_contract"]["strategy_ids"]
    assert set(payload["strategy_contract"]["strategy_ids"]) == set(
        pd.read_parquet(candidate_path)["strategy_id"].astype(str).unique()
    )
    parity_contract = (
        run_root / "policy_params" / "training_live_parity_contract.json"
    )
    assert parity_contract.exists()
    parity_payload = json.loads(parity_contract.read_text())
    assert parity_payload["schema_version"] == "training_live_parity_contract_v1"
    assert set(parity_payload["strategy_contract"]["strategy_ids"]) == set(
        payload["strategy_contract"]["strategy_ids"]
    )

    loaded = load_portfolio_policy_config(data_root=str(data_root), run_id=run_id)
    assert (
        loaded.max_concurrent_positions
        == payload["concurrency"]["max_concurrent_positions"]
    )
    assert (
        loaded.initial_rank_threshold == payload["selection"]["global_threshold_floor"]
    )


def test_run_portfolio_policy_replay_can_use_fixed_policy_config_and_ev_source(tmp_path):
    data_root = tmp_path / "data"
    run_id = "test-run"
    run_root = data_root / "artifacts" / run_id
    candidate_dir = run_root / "simple_policy_optimiser"
    candidate_dir.mkdir(parents=True)
    holdout_candidate_path = candidate_dir / "holdout_candidates.parquet"
    ev_candidate_path = candidate_dir / "policy_window_candidates.parquet"
    rows = pd.DataFrame(
        [
            _candidate("2026-02-01 00:00", "BTC/USD", "long", "L_a", 0.95),
            _candidate("2026-02-01 00:00", "ETH/USD", "short", "S_a", 0.94),
        ]
    )
    rows.to_parquet(holdout_candidate_path, index=False)
    rows.assign(net_return=[0.10, 0.20], gross_return=[0.11, 0.21]).to_parquet(
        ev_candidate_path,
        index=False,
    )
    fixed = PortfolioPolicyParams(
        max_concurrent_positions=1,
        max_concurrent_per_side=None,
        max_concurrent_per_strategy=None,
        max_new_entries_per_bar=1,
        global_threshold_floor=0.50,
        threshold_viability_margin=0.0,
        min_position_size=0.01,
        strategy_ids=("L_a", "S_a"),
        strategy_cores=("a",),
    )
    fixed_config_path = run_root / "policy_params" / "optimized_portfolio_policy_config.json"
    fixed_config_path.parent.mkdir(parents=True)
    fixed_config_path.write_text(json.dumps(fixed.to_live_config()), encoding="utf-8")

    report = run_portfolio_policy_replay(
        data_root=str(data_root),
        run_id=run_id,
        candidate_path=holdout_candidate_path,
        output_dir=run_root / "holdout_replay",
        fixed_policy_config_path=fixed_config_path,
        ev_curve_candidate_path=ev_candidate_path,
    )

    assert report["policy_replay_mode"] == "fixed_policy_config"
    assert report["walk_forward"]["selection_reason"] == "fixed_policy_config_no_optimisation"
    assert report["ev_curve_candidate_path"] == str(ev_candidate_path)
    assert report["optimized_params"]["concurrency"]["max_concurrent_positions"] == 1
