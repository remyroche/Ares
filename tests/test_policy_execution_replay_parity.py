from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.threshold_basis_policy import (
    apply_threshold_basis_policy_to_decisions,
)
from extreme_price_movements.inference.simple_policy_stop import (
    SIMPLE_POLICY_GENERATOR,
    SIMPLE_POLICY_SCHEMA,
    compute_initial_simple_policy_stop_decision,
    compute_simple_policy_stop_decision,
)
from extreme_price_movements.simple_policy_optimiser import simulate_and_score
from scripts.materialize_canonical_exit_policy_replay import (
    _apply_policy_spread_to_returns,
)
from scripts.run_policy_execution_replay_parity import (
    audit_policy_execution_contract,
    compare_close_rows,
    compare_policy_rows,
    compare_portfolio_rows,
    cost_reconciliation,
    summarize_fixed_ev_policy_rows,
)


def test_live_stop_uses_and_reports_spread_protected_capital_lock(
    tmp_path: Path,
) -> None:
    params_source = (
        "artifacts/test-run/simple_policy_optimiser/deployment/"
        "best_policy_params.json"
    )
    artifact_path = tmp_path / params_source
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("{}", encoding="utf-8")
    params = {
        "params_source": params_source,
        "params_hash": hashlib.sha256(artifact_path.read_bytes()).hexdigest()[:16],
        "_loaded_from_simple_policy_artifact": True,
        "_artifact_path": str(artifact_path),
        "generated_by": SIMPLE_POLICY_GENERATOR,
        "schema": SIMPLE_POLICY_SCHEMA,
        "strategy_id": "long_policy_parity",
        "barrier_frac": 0.02,
        "sl_mult": 2.0,
        "trailing_activation_mult": 10.0,
        "trailing_power": 1.5,
        "trailing_squash_divisor": 2.0,
        "giveback_beta": 0.5,
        "atr_power": 1.0,
        "atr_multiplier": 1.0,
        "hard_tp_abs_pct": 0.0,
        "capital_protect_mfe_mult": 0.5,
        "capital_protect_regression_frac": 0.45,
        "capital_protect_lock_frac": 0.0,
        "capital_protect_min_lock_bps": 0.0,
        "capital_protect_spread_lock_mult": 1.5,
    }
    armed = compute_simple_policy_stop_decision(
        side="long",
        state={
            "entry_price": 100.0,
            "peak_price": 103.0,
            "mfe": 0.03,
            "mae": 0.0,
            "stop_price": 96.0,
            "strategy_id": "long_policy_parity",
            "barrier_frac": 0.02,
            "expected_spread_bps": 100.0,
        },
        latest_market_state={},
        policy_params=params,
        require_metadata=True,
    )

    assert armed.reason == "capital_preservation_armed"
    assert armed.capital_protect_armed
    assert not armed.should_replace

    decision = compute_simple_policy_stop_decision(
        side="long",
        state={
            "entry_price": 100.0,
            "peak_price": 103.0,
            "mfe": 0.03,
            "mae": 0.0,
            "stop_price": 96.0,
            "strategy_id": "long_policy_parity",
            "barrier_frac": 0.02,
            "expected_spread_bps": 100.0,
            "capital_protect_armed": True,
        },
        latest_market_state={},
        policy_params=params,
        require_metadata=True,
    )

    assert decision.reason == "capital_preservation"
    assert decision.stop_price == pytest.approx(101.5)
    assert decision.capital_protect_spread_lock_mult == pytest.approx(1.5)


def test_live_monitor_does_not_rescale_effective_entry_barrier(
    tmp_path: Path,
) -> None:
    params_source = (
        "artifacts/test-run/simple_policy_optimiser/deployment/"
        "best_policy_params.json"
    )
    artifact_path = tmp_path / params_source
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("{}", encoding="utf-8")
    params = {
        "params_source": params_source,
        "params_hash": hashlib.sha256(artifact_path.read_bytes()).hexdigest()[:16],
        "_loaded_from_simple_policy_artifact": True,
        "_artifact_path": str(artifact_path),
        "generated_by": SIMPLE_POLICY_GENERATOR,
        "schema": SIMPLE_POLICY_SCHEMA,
        "strategy_id": "long_policy_parity",
        "barrier_frac": 0.02,
        "median_barrier_frac": 0.01,
        "sl_mult": 2.0,
        "trailing_activation_mult": 10.0,
        "trailing_power": 1.5,
        "trailing_squash_divisor": 2.0,
        "giveback_beta": 0.5,
        "atr_power": 1.0,
        "atr_multiplier": 0.5,
        "hard_tp_abs_pct": 0.0,
        "capital_protect_mfe_mult": 1.0,
        "capital_protect_regression_frac": 0.45,
    }
    initial = compute_initial_simple_policy_stop_decision(
        entry_price=100.0,
        policy_params=params,
        side="long",
        strategy_id="long_policy_parity",
        require_metadata=True,
    )
    assert initial.barrier_frac == pytest.approx(0.01)

    monitored = compute_simple_policy_stop_decision(
        side="long",
        state={
            "entry_price": 100.0,
            "peak_price": 100.5,
            "mfe": 0.005,
            "mae": 0.0,
            "stop_price": 98.0,
            "strategy_id": "long_policy_parity",
            "barrier_frac": initial.barrier_frac,
            "barrier_frac_is_effective": True,
        },
        latest_market_state={},
        policy_params=params,
        require_metadata=True,
    )
    assert monitored.barrier_frac == pytest.approx(initial.barrier_frac)
    assert monitored.reason != "capital_preservation"


def test_policy_prefers_canonical_archetype_key_and_rejects_ev_sentinel(
    tmp_path: Path,
) -> None:
    reference = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-06-01", periods=60, freq="6h", tz="UTC"
            ),
            "side_name": ["long"] * 60,
            "policy_archetype": [
                "long_dirtyavoid_sparse_questionable"
            ] * 60,
            "archetype_policy_key": [""] * 60,
            "__archetype_policy_key__": ["long__long_mixed"] * 40
            + ["long__long_other"] * 20,
            "mapped_expected_ev": [0.005] * 60,
            "ev_after_1pct": [0.005] * 60,
        }
    )
    reference_path = tmp_path / "threshold_reference.parquet"
    reference.to_parquet(reference_path, index=False)
    policy = {
        "enabled": True,
        "policy_id": "sentinel_key_parity",
        "family": "side_archetype_expected_ev_recent_correction",
        "selection_mode": "fixed_corrected_ev_threshold",
        "fixed_target_net_ev": 0.007,
        "window_days": 21,
        "min_reference_rows": 5,
        "side_support_target": 1,
        "local_support_target": 1,
        "recent_ev_correction_cap": 0.1,
        "mapped_expected_ev_col": (
            "expected_net_ev_after_1pct_side_archetype"
        ),
        "reference_mapped_expected_ev_col": "mapped_expected_ev",
        "return_col": "ev_after_1pct",
        "reference_candidates_path": str(reference_path),
    }
    decisions = [
        {
            "timestamp": "2026-06-16T00:00:00Z",
            "side_name": "long",
            "policy_archetype": (
                "long__long_dirtyavoid_sparse_questionable"
            ),
            "expected_net_ev_after_1pct_side_archetype": 0.008,
            "chain_results": {
                "archetype_policy_key": "long__long_mixed",
            },
        },
        {
            "timestamp": "2026-06-16T00:00:00Z",
            "side_name": "long",
            "policy_archetype": (
                "long__long_dirtyavoid_sparse_questionable"
            ),
            "expected_net_ev_after_1pct_side_archetype": -1.0,
            "chain_results": {
                "archetype_policy_key": "long__long_mixed",
            },
        },
    ]

    apply_threshold_basis_policy_to_decisions(decisions, policy=policy)

    valid, sentinel = decisions
    assert valid["threshold_basis_policy_archetype"] == "long_mixed"
    assert valid["threshold_basis_ev_target_local_support"] == 40
    assert valid["threshold_basis_expected_ev_correction_scope"] == (
        "side_x_archetype"
    )
    assert valid["threshold_basis_selected"] is True
    assert sentinel["threshold_basis_selected"] is False
    assert sentinel["threshold_basis_reason"] == (
        "invalid_mapped_expected_ev_sentinel"
    )
    assert sentinel["threshold_basis_invalid_mapped_expected_ev_sentinel"] is True
    assert sentinel["threshold_basis_mapped_expected_ev_valid"] is False
    assert np.isnan(
        sentinel["threshold_basis_mapped_expected_ev_side_archetype"]
    )
    assert np.isnan(sentinel["threshold_basis_corrected_expected_ev"])
    assert sentinel["threshold_basis_side_archetype_recent_ev_correction"] == 0.0


def _policy_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z"], utc=True
            ),
            "__symbol__": ["A/USD:USD", "B/USD:USD"],
            "side_name": ["long", "short"],
            "archetype_policy_key": ["long_mixed", "short_default"],
            "historical_rank": [0.91, 0.92],
            "rank_mlp_direct": [0.93, 0.94],
            "expected_net_ev_after_1pct_mlp_direct": [0.008, 0.009],
            "expected_ev_rank_score": [0.95, 0.96],
            "threshold_basis_corrected_expected_ev": [0.0075, 0.0085],
            "threshold_basis_rank_score": [0.97, 0.98],
            "threshold_basis_selected": [True, True],
        }
    )


def test_policy_comparison_is_exact_across_all_policy_layers() -> None:
    reference = _policy_rows()
    replay = reference.rename(
        columns={
            "rank_mlp_direct": "score_regime_calibrated",
            "expected_net_ev_after_1pct_mlp_direct": "expected_net_ev_after_1pct",
        }
    )
    _, summaries = compare_policy_rows(reference, replay)
    assert summaries
    assert all(summary["pass"] for summary in summaries)


def test_policy_comparison_localises_first_admission_mismatch() -> None:
    reference = _policy_rows()
    replay = reference.rename(
        columns={
            "rank_mlp_direct": "score_regime_calibrated",
            "expected_net_ev_after_1pct_mlp_direct": "expected_net_ev_after_1pct",
        }
    )
    replay.loc[1, "threshold_basis_selected"] = False
    _, summaries = compare_policy_rows(reference, replay)
    admission = next(row for row in summaries if row["layer"] == "admission_decision")
    assert admission["pass"] is False
    assert admission["mismatch_count"] == 1


def test_timeout_uses_last_close_and_costs_once() -> None:
    rows = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-07-01T00:00:00Z")],
            "symbol": ["A/USD:USD"],
            "side": np.asarray([1.0], dtype=np.float32),
            "rank_pct": np.asarray([1.0], dtype=np.float32),
            "barrier_pct": np.asarray([0.05], dtype=np.float32),
            "expected_half_spread_bps": np.asarray([10.0], dtype=np.float32),
            "exit_quote_half_spread_bps": np.asarray([10.0], dtype=np.float32),
        }
    )
    paths = np.full((1, 4), 100.0, dtype=np.float32)
    result = simulate_and_score(
        rows,
        paths,
        paths,
        paths,
        paths,
        cost_pct=0.005,
        sl_mult=2.0,
        trailing_activation_mult=2.0,
        max_concurrent_trades=1,
        max_concurrent_per_asset=1,
        market_mode="perps",
    )
    assert result["exit_reason"] == ["timeout"]
    assert int(result["exit_bars"][0]) == 3
    gross = float(result["gross_returns"][0])
    fee = float(result["fee_returns"][0])
    net = float(result["net_returns"][0])
    assert gross < 0.0  # Flat mid-price still pays executable spread once.
    assert np.isclose(net, gross - fee, atol=2e-7)

    materialized = pd.DataFrame(
        {
            "gross_return": [gross],
            "net_return": [net],
            "spread_cost_bps": [10.0],
            "exit_spread_cost_bps": [10.0],
        }
    )
    audited = _apply_policy_spread_to_returns(materialized)
    assert np.isclose(audited.loc[0, "gross_return"], gross)
    assert np.isclose(audited.loc[0, "net_return"], net)
    assert bool(audited.loc[0, "policy_spread_embedded_in_executable_prices"])
    assert not bool(audited.loc[0, "policy_spread_applied_to_returns"])


def test_cost_reconciliation_rejects_legacy_double_spread_marker() -> None:
    current = pd.DataFrame(
        {
            "gross_return": [0.02],
            "fee_return": [0.0101],
            "net_return": [0.0099],
            "spread_cost_bps": [5.0],
            "exit_spread_cost_bps": [5.0],
            "policy_spread_applied_to_returns": [False],
        }
    )
    assert cost_reconciliation(current)["pass"] is True
    current["policy_spread_applied_to_returns"] = True
    audit = cost_reconciliation(current)
    assert audit["pass"] is False
    assert audit["legacy_double_spread_rows"] == 1
    assert audit["legacy_double_spread_mean_bps"] == 10.0


def test_cost_reconciliation_rejects_unknown_legacy_spread_provenance() -> None:
    rows = pd.DataFrame(
        {
            "gross_return": [0.02],
            "fee_return": [0.01],
            "net_return": [0.01],
        }
    )
    audit = cost_reconciliation(rows)
    assert audit["pass"] is False
    assert audit["spread_provenance"] == "legacy_unverifiable"


def test_portfolio_and_close_comparisons_are_row_level() -> None:
    common = {
        "timestamp": [pd.Timestamp("2026-07-01T00:00:00Z")],
        "symbol": ["A/USD:USD"],
        "side_name": ["long"],
        "policy_archetype": ["long_mixed"],
    }
    portfolio = pd.DataFrame(
        {
            **common,
            "accepted": [True],
            "rejection_reason": ["accepted"],
            "position_size": [100.0],
            "position_net_return": [0.01],
            "position_gross_return": [0.02],
            "position_exit_reason": ["trailing"],
        }
    )
    detail, summary = compare_portfolio_rows(portfolio, portfolio.copy())
    assert len(detail) == 1
    assert summary["pass"] is True

    close = pd.DataFrame(
        {
            **common,
            "entry_price": [100.0],
            "exit_price": [102.0],
            "exit_timestamp": [pd.Timestamp("2026-07-01T01:00:00Z")],
            "holding_bars": [4],
            "gross_return": [0.02],
            "fee_return": [0.0101],
            "net_return": [0.0099],
            "policy_size_multiplier": [1.0],
            "simple_policy_exit_reason": ["trailing"],
            "execution_policy_key": ["long_mixed"],
        }
    )
    close_replay = close.copy()
    close_replay.loc[0, "simple_policy_exit_reason"] = "timeout"
    close_detail, close_summary = compare_close_rows(close, close_replay)
    assert len(close_detail) == 1
    assert close_summary["pass"] is False
    assert close_summary["mismatch_count"] == 1
    assert close_summary["close_reason_mismatch_count"] == 1
    assert close_summary["numeric_metrics"]["net_return"]["mismatch_count"] == 0


def test_close_comparison_reports_missing_positions() -> None:
    reference = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z"], utc=True
            ),
            "symbol": ["A/USD:USD", "B/USD:USD"],
            "side_name": ["long", "short"],
            "policy_archetype": ["long_mixed", "short_default"],
            "net_return": [0.01, 0.02],
            "simple_policy_exit_reason": ["trailing", "timeout"],
        }
    )
    replay = reference.iloc[:1].copy()
    _, summary = compare_close_rows(reference, replay)
    assert summary["matched_positions"] == 1
    assert summary["reference_only_positions"] == 1
    assert summary["replay_only_positions"] == 0
    assert summary["pass"] is False


def test_contract_audit_checks_fixed_ev_portfolio_and_local_geometry(tmp_path) -> None:
    admission = tmp_path / "admission.json"
    portfolio = tmp_path / "portfolio.json"
    exit_dir = tmp_path / "exit"
    exit_dir.mkdir()
    admission.write_text(
        """{
          "policy_id": "side_archetype_hier_ev_fixed70_trim10_21d_v1",
          "family": "side_archetype_expected_ev_recent_correction",
          "selection_mode": "fixed_corrected_ev_threshold",
          "window_days": 21,
          "robust_daily_residual_trim_fraction": 0.10,
          "fixed_target_net_ev": 0.007,
          "outcome_horizon_hours": 12
        }"""
    )
    portfolio.write_text(
        """{
          "portfolio_policy_version": "global_auction_v1",
          "regime_ev_calibration_policy_id":
            "meta_residual_v9_tail95_market_state_mlp_hier_ev_v1",
          "concurrency": {
            "max_new_entries_per_bar": 2,
            "max_concurrent_positions": 8
          }
        }"""
    )
    pd.DataFrame({"side": ["long", "short"]}).to_csv(
        exit_dir / "side_parent_policy_summary.csv", index=False
    )
    pd.DataFrame(
        {
            "side": ["long", "short"],
            "policy_archetype": ["long_mixed", "short_default"],
        }
    ).to_csv(exit_dir / "side_archetype_policy_summary.csv", index=False)
    audit = audit_policy_execution_contract(
        admission_policy_path=admission,
        portfolio_config_path=portfolio,
        exit_policy_dir=exit_dir,
    )
    assert audit["pass"] is True
    assert audit["local_geometry_rows"] == 2


def test_fixed_ev_matrix_comparison_is_row_level() -> None:
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z"], utc=True
            ),
            "symbol": ["A/USD:USD", "B/USD:USD"],
            "side_name": ["long", "short"],
            "policy_archetype": ["long_mixed", "short_default"],
            "v9_tail95_rank": [0.9, 0.8],
            "mlp_rank": [0.91, 0.81],
            "hierarchical_expected_ev": [0.008, 0.006],
            "matrix_corrected_ev": [0.009, 0.0065],
            "replay_corrected_ev": [0.009, 0.0065],
            "matrix_admitted": [True, False],
            "replay_admitted": [True, False],
        }
    )
    detail, summaries = summarize_fixed_ev_policy_rows(rows)
    assert len(detail) == 2
    assert all(summary["pass"] for summary in summaries)
    admission = next(row for row in summaries if row["layer"] == "admission_decision")
    assert admission["reference_selected"] == 1
    assert admission["replay_selected"] == 1
