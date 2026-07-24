import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.execution_fill_model import (
    stop_exit_fill_price,
    stop_exit_fill_price_array,
)
from extreme_price_movements.inference.execution_reconciliation import (
    _filter_table_since,
    _replay_active_threshold_policy,
    build_ledger_replay_field_coverage,
    build_live_decision_replay_reconciliation,
    build_shadow_trade_reconciliation,
    build_spread_slippage_reconciliation,
    execution_parity_audit_status,
    _logged_meta_prediction,
)
from extreme_price_movements.inference.threshold_basis_policy import (
    apply_threshold_basis_policy_to_decisions,
)
from extreme_price_movements.portfolio_policy_replay import (
    PortfolioPolicyParams,
    dynamic_threshold_for_count,
)


class _DummyMetaModel:
    feature_columns = ["base_score", "drift_context"]

    def predict(self, X):
        return X["drift_context"].to_numpy(dtype=float)


def test_active_threshold_replay_uses_reconstructed_postprocessor_outputs(tmp_path):
    reference = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=48, freq="6h", tz="UTC"),
            "outcome_resolved_at": pd.date_range(
                "2026-06-01 12:00", periods=48, freq="6h", tz="UTC"
            ),
            "side_name": ["long"] * 48,
            "policy_archetype": ["long_mixed"] * 48,
            "mapped_expected_ev": [0.006] * 48,
            "ev_after_1pct": [0.006] * 48,
            "rank_mlp_direct": np.linspace(0.1, 0.9, 48),
        }
    )
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    policy = {
        "enabled": True,
        "policy_id": "active_ev70_test",
        "family": "side_archetype_expected_ev_recent_correction",
        "selection_mode": "fixed_corrected_ev_threshold",
        "fixed_target_net_ev": 0.007,
        "window_days": 21,
        "min_reference_rows": 10,
        "local_support_target": 1,
        "side_support_target": 1,
        "recent_ev_correction_cap": 0.03,
        "mapped_expected_ev_col": "expected_net_ev_after_1pct_side_archetype",
        "reference_mapped_expected_ev_col": "mapped_expected_ev",
        "rank_blend_parent_col": "v9_tail95_predecessor_rank",
        "live_score_col": "expected_ev_rank_score",
        "return_col": "ev_after_1pct",
        "reference_candidates_path": str(reference_path),
        "recalibration_frequency": "1d_at_00_utc",
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    source_decisions = []
    for idx, ev in enumerate((0.008, 0.006)):
        source_decisions.append(
            {
                "signal_bar_ts": pd.Timestamp("2026-06-14 00:00", tz="UTC"),
                "symbol": f"A{idx}/USD:USD",
                "side": "long",
                "side_name": "long",
                "strategy_id": "long_test",
                "policy_archetype": "long__long_mixed",
                "archetype_policy_key": "long__long_mixed",
                "policy_rank_pct": 0.9,
                "expected_ev_rank_score": 0.8 - idx * 0.1,
                "expected_net_ev_after_1pct_side_archetype": ev,
                "expected_net_ev_after_1pct": ev,
                "v9_tail95_predecessor_rank": 0.75 - idx * 0.1,
            }
        )
    apply_threshold_basis_policy_to_decisions(source_decisions, policy=policy)

    ledger_rows = []
    report_rows = []
    for idx, decision in enumerate(source_decisions):
        ledger_row = dict(decision)
        ledger_row["_ledger_row_id"] = idx
        ledger_rows.append(ledger_row)
        report_row = {
            "_ledger_row_id": idx,
            "replay_status": "ok",
            "replay_expected_ev_rank_score": decision["expected_ev_rank_score"],
            "replay_expected_net_ev_after_1pct": decision[
                "expected_net_ev_after_1pct"
            ],
            "replay_v9_tail95_predecessor_rank": decision[
                "v9_tail95_predecessor_rank"
            ],
        }
        for field in (
            "threshold_basis_selected",
            "threshold_basis_rank_score",
            "threshold_basis_corrected_expected_ev",
            "threshold_basis_corrected_expected_ev_rank",
            "threshold_basis_side_archetype_recent_ev_correction",
            "threshold_basis_ev_target_local_support",
            "threshold_basis_reference_asof",
            "threshold_basis_reason",
        ):
            report_row[f"stored_{field}"] = decision.get(field)
        report_rows.append(report_row)

    _, summary = _replay_active_threshold_policy(
        pd.DataFrame(report_rows),
        pd.DataFrame(ledger_rows),
        policy_path=policy_path,
        tolerance=1e-12,
    )

    assert summary["pass"] is True
    assert summary["selected_rows"] == 1
    assert summary["mismatch_rows"] == 0


def test_threshold_policy_attaches_robust_28d_archetype_baseline(tmp_path):
    timestamps = pd.date_range("2026-05-10", periods=15, freq="D", tz="UTC")
    reference = pd.DataFrame(
        {
            "timestamp": timestamps,
            "outcome_resolved_at": timestamps + pd.Timedelta(hours=12),
            "side_name": ["long"] * len(timestamps),
            "policy_archetype": ["compression_release"] * len(timestamps),
            "rank_mlp_direct": np.linspace(0.1, 0.9, len(timestamps)),
            "mapped_expected_ev": [0.007] * len(timestamps),
            # One negative daily residual must be trimmed with the same daily
            # median/IQR procedure used by the admission correction.
            "ev_after_1pct": [-0.30] + [0.012] * (len(timestamps) - 1),
            "clean_exec": [0] + [1] * (len(timestamps) - 1),
            "dirty_positive": [1] + [0] * (len(timestamps) - 1),
            "full_path_bad_mae_1r": [1] + [0] * (len(timestamps) - 1),
            "timeout": [1] + [0] * (len(timestamps) - 1),
            "first_touch_mae_to_sl": [1.0] + [0.25] * (len(timestamps) - 1),
            "gmm_cluster_id": [2] * len(timestamps),
        }
    )
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    policy = {
        "enabled": True,
        "policy_id": "email_baseline_test",
        "family": "side_archetype_expected_ev_recent_correction",
        "selection_mode": "fixed_corrected_ev_threshold",
        "fixed_target_net_ev": 0.007,
        "window_days": 21,
        "email_archetype_baseline_window_days": 28,
        "email_archetype_baseline_min_rows": 4,
        "min_reference_rows": 4,
        "local_support_target": 1,
        "side_support_target": 1,
        "recent_ev_correction_cap": 0.03,
        "mapped_expected_ev_col": "expected_net_ev_after_1pct_side_archetype",
        "reference_mapped_expected_ev_col": "mapped_expected_ev",
        "rank_blend_parent_col": "v9_tail95_predecessor_rank",
        "live_score_col": "expected_ev_rank_score",
        "return_col": "ev_after_1pct",
        "reference_candidates_path": str(reference_path),
        "recalibration_frequency": "1d_at_00_utc",
        "robust_daily_residual_trim_fraction": 0.10,
    }
    decision = {
        "signal_bar_ts": pd.Timestamp("2026-05-26 06:00", tz="UTC"),
        "symbol": "TEST/USD:USD",
        "side": "long",
        "side_name": "long",
        "strategy_id": "long_test",
        "policy_archetype": "long__compression_release",
        "archetype_policy_key": "long__compression_release",
        "policy_rank_pct": 0.92,
        "expected_ev_rank_score": 0.94,
        "expected_net_ev_after_1pct_side_archetype": 0.009,
        "expected_net_ev_after_1pct": 0.009,
        "v9_tail95_predecessor_rank": 0.82,
        "chain_results": {"gmm_cluster_id": 2.0},
    }

    apply_threshold_basis_policy_to_decisions([decision], policy=policy)

    assert decision["threshold_basis_archetype_baseline_window_days"] == 28
    assert decision["threshold_basis_archetype_baseline_scope"] == "side_x_archetype"
    assert decision["threshold_basis_archetype_baseline_trimmed_days"] >= 1
    assert decision["threshold_basis_archetype_baseline_ev_mean"] > 0.0
    assert decision["threshold_basis_archetype_baseline_clean_rate"] == pytest.approx(1.0)
    assert decision["threshold_basis_archetype_baseline_bad_mae_rate"] == pytest.approx(0.0)
    assert decision["threshold_basis_archetype_baseline_timeout_rate"] == pytest.approx(0.0)
    assert decision[
        "threshold_basis_archetype_baseline_successful_trade_mae_to_sl_mean"
    ] == pytest.approx(0.25)
    assert decision[
        "threshold_basis_archetype_baseline_successful_trade_mae_to_sl_support"
    ] == 14
    assert 1 <= decision["threshold_basis_archetype_baseline_mapped_ev_decile"] <= 10
    assert np.isfinite(
        decision[
            "threshold_basis_archetype_baseline_mapped_ev_decile_calibration_residual"
        ]
    )
    assert decision["threshold_basis_archetype_baseline_gmm_state_support"] >= 12


def test_threshold_policy_email_baseline_compares_recent_and_historical_rates(tmp_path):
    timestamps = pd.date_range("2026-03-01", periods=84, freq="D", tz="UTC")
    reference = pd.DataFrame(
        {
            "timestamp": timestamps,
            "outcome_resolved_at": timestamps + pd.Timedelta(hours=12),
            "side_name": ["long"] * len(timestamps),
            "policy_archetype": ["long__compression_release"] * len(timestamps),
            "rank_mlp_direct": np.linspace(0.1, 0.9, len(timestamps)),
            "mapped_expected_ev": [0.007] * len(timestamps),
            # The current 28d reference is clean; the older reference is not.
            "ev_after_1pct": [-0.01] * 57 + [0.01] * 27,
        }
    )
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    policy = {
        "enabled": True,
        "policy_id": "email_baseline_history_test",
        "family": "side_archetype_expected_ev_recent_correction",
        "selection_mode": "fixed_corrected_ev_threshold",
        "fixed_target_net_ev": 0.007,
        "window_days": 21,
        "email_archetype_baseline_window_days": 28,
        "email_archetype_baseline_min_rows": 4,
        "min_reference_rows": 4,
        "local_support_target": 1,
        "side_support_target": 1,
        "recent_ev_correction_cap": 0.03,
        "mapped_expected_ev_col": "expected_net_ev_after_1pct_side_archetype",
        "reference_mapped_expected_ev_col": "mapped_expected_ev",
        "rank_blend_parent_col": "v9_tail95_predecessor_rank",
        "live_score_col": "expected_ev_rank_score",
        "return_col": "ev_after_1pct",
        "reference_candidates_path": str(reference_path),
        "recalibration_frequency": "1d_at_00_utc",
        "robust_daily_residual_trim_fraction": 0.10,
    }
    decision = {
        "signal_bar_ts": pd.Timestamp("2026-05-25 06:00", tz="UTC"),
        "symbol": "TEST/USD:USD",
        "side": "long",
        "side_name": "long",
        "strategy_id": "long_test",
        "policy_archetype": "long__compression_release",
        "archetype_policy_key": "long__compression_release",
        "policy_rank_pct": 0.92,
        "expected_ev_rank_score": 0.94,
        "expected_net_ev_after_1pct_side_archetype": 0.009,
        "expected_net_ev_after_1pct": 0.009,
        "v9_tail95_predecessor_rank": 0.82,
    }

    apply_threshold_basis_policy_to_decisions([decision], policy=policy)

    assert decision["threshold_basis_archetype_baseline_positive_ev_rate"] == pytest.approx(1.0)
    assert decision[
        "threshold_basis_archetype_baseline_historical_positive_ev_rate"
    ] == pytest.approx(0.0)
    assert decision[
        "threshold_basis_archetype_baseline_recent_vs_historical_positive_ev_rate"
    ] == pytest.approx(1.0)


def test_threshold_policy_email_diagnostics_do_not_change_admission(tmp_path):
    """Extra outcome fields are email-only and cannot move a policy decision."""

    timestamps = pd.date_range("2026-06-01", periods=72, freq="12h", tz="UTC")
    core = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["X/USD:USD"] * len(timestamps),
            "side_name": ["long"] * len(timestamps),
            "policy_archetype": ["default"] * len(timestamps),
            "rank_mlp_direct": np.linspace(0.1, 0.9, len(timestamps)),
            "mapped_expected_ev": np.linspace(0.001, 0.014, len(timestamps)),
            "ev_after_1pct": np.linspace(0.002, 0.015, len(timestamps)),
            "outcome_resolved_at": timestamps + pd.Timedelta(hours=12),
        }
    )
    enriched = core.assign(
        clean_exec=1.0,
        dirty_positive=0.0,
        full_path_bad_mae_1r=0.0,
        timeout=0.0,
        first_touch_mae_to_sl=0.25,
    )
    core_path = tmp_path / "core.parquet"
    enriched_path = tmp_path / "enriched.parquet"
    core.to_parquet(core_path, index=False)
    enriched.to_parquet(enriched_path, index=False)
    base_policy = {
        "enabled": True,
        "policy_id": "fixed-ev-test",
        "family": "side_archetype_expected_ev_recent_correction",
        "selection_mode": "fixed_corrected_ev_threshold",
        "fixed_target_net_ev": 0.007,
        "window_days": 21,
        "min_reference_rows": 4,
        "reference_mapped_expected_ev_col": "mapped_expected_ev",
        "return_col": "ev_after_1pct",
        "robust_daily_residual_trim_fraction": 0.1,
        "outcome_horizon_hours": 12,
        "reference_candidates_path": str(core_path),
        "reference_columns": list(core.columns),
    }
    enriched_policy = dict(base_policy)
    enriched_policy.update(
        {
            "reference_candidates_path": str(enriched_path),
            "reference_columns": list(enriched.columns),
            "email_archetype_baseline_window_days": 28,
        }
    )
    source = {
        "timestamp": "2026-07-08T00:00:00+00:00",
        "symbol": "Y/USD:USD",
        "side_name": "long",
        "policy_archetype": "long__default",
        "expected_net_ev_after_1pct_side_archetype": 0.010,
        "v9_tail95_predecessor_rank": 0.85,
    }
    core_decision = dict(source)
    enriched_decision = dict(source)
    apply_threshold_basis_policy_to_decisions([core_decision], policy=base_policy)
    apply_threshold_basis_policy_to_decisions(
        [enriched_decision], policy=enriched_policy
    )
    for key in (
        "threshold_basis_selected",
        "threshold_basis_rank_score",
        "threshold_basis_dynamic_ev_target",
        "threshold_basis_dynamic_score_threshold",
        "threshold_basis_corrected_expected_ev",
        "threshold_basis_side_archetype_recent_ev_correction",
    ):
        assert enriched_decision[key] == core_decision[key]
    assert enriched_decision["threshold_basis_archetype_baseline_clean_rate"] == 1.0


class _DummyAlignedMetaModel(_DummyMetaModel):
    s52_meta_score_alignment_ = {
        "enabled": True,
        "mode": "affine",
        "slope": 0.5,
        "intercept": 0.1,
    }


def test_filter_table_since_uses_decision_time_for_live_ledgers():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-07-14T15:00:00Z", "2026-07-14T15:00:00Z"], utc=True
            ),
            "decision_ts": pd.to_datetime(
                ["2026-07-14T15:55:00Z", "2026-07-14T16:05:00Z"], utc=True
            ),
            "symbol": ["OLD/USD:USD", "NEW/USD:USD"],
        }
    )

    filtered = _filter_table_since(frame, "2026-07-14T16:00:00Z")

    assert filtered["symbol"].tolist() == ["NEW/USD:USD"]


def test_filter_table_since_prefers_utc_lifecycle_event_times():
    frame = pd.DataFrame(
        {
            # These legacy display timestamps are CEST-naive and must not be
            # interpreted as canonical UTC session evidence.
            "timestamp": [
                "2026-07-17T21:08:46.130613",
                "2026-07-17T22:19:46.199673",
                "2026-07-17T22:30:00+00:00",
            ],
            "lifecycle_event": ["entry_placed", "exit_filled", "entry_placed"],
            "action": ["enter", "exit", "enter"],
            "entry_time": [
                "2026-07-17T19:08:40.591000+00:00",
                "2026-07-17T19:08:40.591000+00:00",
                None,
            ],
            "exit_time": [None, "2026-07-17T20:19:44.329288+00:00", None],
            "symbol": ["OLD_ENTRY", "NEW_EXIT", "AWARE_FALLBACK"],
        }
    )

    filtered = _filter_table_since(frame, "2026-07-17T19:52:00Z")

    assert filtered["symbol"].tolist() == ["NEW_EXIT", "AWARE_FALLBACK"]


def test_filter_table_since_drops_lifecycle_rows_without_canonical_time():
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-07-17T22:30:00", "not-a-time"],
            "lifecycle_event": ["entry_placed", "entry_placed"],
            "action": ["enter", "enter"],
            "entry_time": [None, None],
            "symbol": ["NAIVE", "INVALID"],
        }
    )

    filtered = _filter_table_since(frame, "2026-07-17T19:52:00Z")

    assert filtered.empty


def test_filter_table_since_returns_empty_when_no_timestamp_contract_exists():
    frame = pd.DataFrame({"symbol": ["BTC/USD:USD"]})

    filtered = _filter_table_since(frame, "2026-07-14T16:00:00Z")

    assert filtered.empty


class _DummyOrchestrator:
    meta_models = {"long_demo_tbm_clf": _DummyMetaModel()}


class _DummyAlignedOrchestrator:
    meta_models = {"long_demo_tbm_clf": _DummyAlignedMetaModel()}


def test_logged_meta_prediction_scores_final_logged_matrix_directly():
    meta_features = pd.DataFrame(
        [{"base_score": 0.42, "drift_context": 0.73}],
        index=["BTC/USD:USD"],
    )

    pred, source = _logged_meta_prediction(
        _DummyOrchestrator(),
        meta_features,
        side="long",
        strategy_id="demo",
        meta_model_key="long_demo_tbm_clf",
    )

    assert pred == pytest.approx(0.73)
    assert source == "logged_final_meta_input"


def test_logged_meta_prediction_uses_training_neutral_fill_for_nonfinite_values():
    meta_features = pd.DataFrame(
        [{"base_score": 0.42, "drift_context": np.nan}],
        index=["BTC/USD:USD"],
    )

    pred, source = _logged_meta_prediction(
        _DummyOrchestrator(),
        meta_features,
        side="long",
        strategy_id="demo",
        meta_model_key="long_demo_tbm_clf",
    )

    assert pred == pytest.approx(0.0)
    assert source == "logged_final_meta_input"


def test_logged_meta_prediction_requires_complete_logged_contract():
    meta_features = pd.DataFrame([{"base_score": 0.42}], index=["BTC/USD:USD"])

    pred, source = _logged_meta_prediction(
        _DummyOrchestrator(),
        meta_features,
        side="long",
        strategy_id="demo",
        meta_model_key="long_demo_tbm_clf",
    )

    assert np.isnan(pred)
    assert source == "incomplete_logged_meta_features:1"


def test_spread_slippage_reconciliation_compares_policy_proxy_to_live():
    ledger = pd.DataFrame(
        [
            {
                "signal_bar_ts": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "symbol": "BTC/USD:USD",
                "side": "long",
                "strategy_id": "long_a",
                "portfolio_decision": "traded",
                "was_traded": True,
                "entry_slippage_proxy_bps": 4.0,
                "expected_fill_slippage_bps": 6.0,
                "ticker_spread_bps": 8.0,
                "expected_fill_price": 101.0,
                "realized_entry_price": 101.1,
                "theoretical_entry_price": 100.0,
            }
        ]
    )

    rows, summary = build_spread_slippage_reconciliation(ledger)

    assert rows["expected_policy_slippage_bps"].iloc[0] == 4.0
    assert rows["live_total_entry_friction_bps"].iloc[0] == 10.0
    assert rows["policy_vs_live_slippage_delta_bps"].iloc[0] == 2.0
    assert summary["traded_rows"] == 1


def test_decision_replay_ignores_live_hard_veto_rows_for_auction_capacity(tmp_path):
    config_path = tmp_path / "optimized_portfolio_policy_config.json"
    params = PortfolioPolicyParams(
        max_concurrent_positions=4,
        max_concurrent_per_side=None,
        max_concurrent_per_strategy=None,
        max_new_entries_per_bar=2,
        global_threshold_floor=0.50,
        threshold_viability_margin=0.0,
        min_position_size=0.01,
    )
    config_path.write_text(json.dumps(params.to_live_config()), encoding="utf-8")
    ledger = pd.DataFrame(
        [
            {
                "signal_bar_ts": pd.Timestamp.now(tz="UTC").floor("min"),
                "symbol": "BTC/USD:USD",
                "side": "long",
                "strategy_id": "long_a",
                "threshold_rank_score": 0.90,
                "initial_rank_threshold": 0.50,
                "theoretical_entry_price": 100.0,
                "portfolio_decision": "portfolio_rejected",
                "portfolio_reject_reason": "global_auction_capacity:global_entry_cap_reached",
                "was_traded": False,
            }
        ]
    )

    rows, summary = build_live_decision_replay_reconciliation(
        ledger,
        portfolio_policy_config_path=config_path,
    )

    assert summary["replay_accepted"] == 0
    assert summary["live_traded"] == 0
    assert rows["replay_live_gap_class"].iloc[0] == "match"
    assert (
        rows["replay_live_gap_explanation"].iloc[0]
        == "live_reject:global_auction_capacity:global_entry_cap_reached"
    )


def test_decision_replay_uses_persisted_state_and_quality_skips(tmp_path):
    config_path = tmp_path / "optimized_portfolio_policy_config.json"
    params = PortfolioPolicyParams(
        max_concurrent_positions=8,
        max_concurrent_per_side=None,
        max_concurrent_per_strategy=None,
        max_new_entries_per_bar=2,
        global_threshold_floor=0.0,
        occupancy_threshold_alpha=0.3,
        occupancy_threshold_power=1.5,
        allocation_threshold_alpha=0.3,
        allocation_threshold_power=1.0,
        threshold_viability_margin=0.0,
        min_position_size=0.01,
    )
    config_path.write_text(json.dumps(params.to_live_config()), encoding="utf-8")
    ts = pd.Timestamp("2026-07-17 19:00", tz="UTC")

    def snapshot(open_positions, open_notional, wallet=40.0):
        return json.dumps(
            {
                "capacity": {
                    "open_positions": open_positions,
                    "open_notional": open_notional,
                    "wallet_value": wallet,
                }
            }
        )

    threshold_open4 = dynamic_threshold_for_count(
        0.90, 4, params, allocation_share=27.2 / 40.0
    )
    threshold_open5 = dynamic_threshold_for_count(
        0.90, 5, params, allocation_share=37.4 / 40.0
    )
    ledger = pd.DataFrame(
        [
            {
                "signal_bar_ts": ts,
                "decision_ts": ts + pd.Timedelta(seconds=1),
                "symbol": "BAD/USD:USD",
                "side": "long",
                "strategy_id": "long_a",
                "auction_rank_number": 1,
                "portfolio_gate_rank_score": np.nan,
                "portfolio_gate_initial_threshold": np.nan,
                "portfolio_gate_final_threshold": np.nan,
                "portfolio_state_snapshot_json": snapshot(4, 27.2),
                "raw_signal_close_unreliable": True,
                "raw_signal_close_unreliable_reason": "zero_volume_raw_close",
                "portfolio_decision": "portfolio_rejected",
                "portfolio_reject_reason": (
                    "global_auction_data_quality:unreliable_raw_signal_close:"
                    "zero_volume_raw_close"
                ),
                "was_traded": False,
            },
            {
                "signal_bar_ts": ts,
                "decision_ts": ts + pd.Timedelta(seconds=2),
                "symbol": "CAKE/USD:USD",
                "side": "long",
                "strategy_id": "long_a",
                "auction_rank_number": 2,
                "portfolio_gate_rank_score": 0.96,
                "portfolio_gate_initial_threshold": 0.90,
                "portfolio_gate_final_threshold": threshold_open4,
                "portfolio_state_snapshot_json": snapshot(4, 27.2),
                "raw_signal_close_unreliable": False,
                "portfolio_decision": "traded",
                "was_traded": True,
            },
            {
                "signal_bar_ts": ts,
                "decision_ts": ts + pd.Timedelta(seconds=3),
                "symbol": "LDO/USD:USD",
                "side": "short",
                "strategy_id": "short_a",
                "auction_rank_number": 3,
                "portfolio_gate_rank_score": 0.94,
                "portfolio_gate_initial_threshold": 0.90,
                "portfolio_gate_final_threshold": threshold_open5,
                "portfolio_state_snapshot_json": snapshot(5, 37.4),
                "portfolio_decision": "portfolio_rejected",
                "portfolio_reject_reason": (
                    "global_auction_portfolio_pre_liquidity:"
                    "rank_below_dynamic_threshold"
                ),
                "was_traded": False,
            },
        ]
    )

    rows, summary = build_live_decision_replay_reconciliation(
        ledger,
        portfolio_policy_config_path=config_path,
    )

    assert summary["replay_mode"] == "persisted_auction_state"
    assert summary["live_traded"] == 1
    assert summary["replay_accepted"] == 1
    assert summary["decision_mismatches"] == 0
    assert summary["recomputed_threshold_max_abs_delta"] == pytest.approx(0.0)
    accepted = rows.set_index("symbol")["replay_accepted"].to_dict()
    assert accepted == {
        "BAD/USD:USD": False,
        "CAKE/USD:USD": True,
        "LDO/USD:USD": False,
    }


def test_decision_replay_keeps_stateful_mode_when_entry_cap_is_zero(tmp_path):
    config_path = tmp_path / "optimized_portfolio_policy_config.json"
    params = PortfolioPolicyParams(
        max_concurrent_positions=8,
        max_new_entries_per_bar=2,
        min_position_size=0.01,
    )
    config_path.write_text(json.dumps(params.to_live_config()), encoding="utf-8")
    ledger = pd.DataFrame(
        [
            {
                "signal_bar_ts": pd.Timestamp("2026-07-17 19:00", tz="UTC"),
                "symbol": "FULL/USD:USD",
                "side": "long",
                "strategy_id": "long_a",
                "auction_rank_number": 1,
                "auction_entry_cap": 0,
                "portfolio_gate_rank_score": np.nan,
                "portfolio_gate_initial_threshold": np.nan,
                "portfolio_gate_final_threshold": np.nan,
                "portfolio_state_snapshot_json": json.dumps(
                    {
                        "capacity": {
                            "open_positions": 8,
                            "open_notional": 40.0,
                            "wallet_value": 40.0,
                        }
                    }
                ),
                "portfolio_decision": "portfolio_rejected",
                "portfolio_reject_reason": (
                    "global_auction_capacity:global_entry_cap_reached"
                ),
                "was_traded": False,
            }
        ]
    )

    rows, summary = build_live_decision_replay_reconciliation(
        ledger,
        portfolio_policy_config_path=config_path,
    )

    assert summary["replay_mode"] == "persisted_auction_state"
    assert summary["decision_mismatches"] == 0
    assert not bool(rows.loc[0, "replay_accepted"])


def test_ledger_replay_field_coverage_flags_missing_traded_entry_fields():
    ledger = pd.DataFrame(
        [
            {
                "signal_bar_ts": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "symbol": "BTC/USD:USD",
                "side": "long",
                "strategy_id": "long_a",
                "portfolio_decision": "traded",
                "base_model_features_json": '["x"]',
                "base_model_feature_values_json": '{"x": 1.0}',
                "base_pred": 0.61,
                "meta_pred": 0.62,
                "calibrated_score": 0.62,
                "policy_rank_pct": 0.9,
                "auction_rank_pct": 0.9,
                "threshold_rank_score": 0.9,
                "threshold_rank_score_source": "policy",
                "passed_rank_gate": True,
                "decision_ts": pd.Timestamp("2026-01-01 00:05", tz="UTC"),
                "signal_bar_close_ts": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "policy_entry_price": 100.0,
                "expected_fill_price": 100.2,
                # Deliberately omit realized_entry_price and order_id.
                "signal_to_entry_seconds": 300.0,
                "decision_to_entry_seconds": 3.0,
                "hourly_close_to_latest_decision_price_bps": 1.0,
                "decision_price_to_fill_bps": 2.0,
                "ticker_spread_bps": 8.0,
                "expected_fill_slippage_bps": 3.0,
                "expected_total_entry_friction_bps": 7.0,
                "fee_bps": 7.0,
                "ev_haircut_bps": 0.0,
                "position_id": "p1",
                "was_traded": True,
            }
        ]
    )

    rows, summary = build_ledger_replay_field_coverage(ledger)

    assert summary["live_traded_rows"] == 1
    assert summary["failed_traded_field_checks"] >= 2
    missing = rows.loc[rows["missing_rows"].gt(0), "accepted_alternatives"].tolist()
    assert "realized_entry_price|entry_price_actual" in missing
    assert "order_id" in missing
    assert summary["exact_portfolio_state_replayable_rows"] == 0


def test_ledger_replay_field_coverage_accepts_portfolio_state_snapshot():
    ledger = pd.DataFrame(
        [
            {
                "signal_bar_ts": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "symbol": "BTC/USD:USD",
                "side": "long",
                "strategy_id": "long_a",
                "portfolio_decision": "portfolio_rejected",
                "base_model_features_json": '["x"]',
                "base_model_feature_values_json": '{"x": 1.0}',
                "base_pred": 0.61,
                "meta_pred": 0.62,
                "calibrated_score": 0.62,
                "policy_rank_pct": 0.9,
                "auction_rank_pct": 0.9,
                "threshold_rank_score": 0.9,
                "threshold_rank_score_source": "policy",
                "passed_rank_gate": True,
                "portfolio_state_snapshot_json": '{"positions":[],"cooldowns":{}}',
                "portfolio_state_snapshot_hash": "abc123",
                "wallet_before": 10000.0,
                "open_positions_before": 0,
                "cooldowns_before_json": "{}",
                "portfolio_priority": 0.5,
            }
        ]
    )

    rows, summary = build_ledger_replay_field_coverage(ledger)

    assert summary["exact_portfolio_state_replayable_rows"] == 1
    state_rows = rows.loc[rows["field_group"].eq("exact_portfolio_state_replay")]
    assert state_rows["missing_rows"].sum() == 0


def test_shadow_trade_reconciliation_reports_exit_parity_pass():
    trade_log = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "symbol": "BIO/USD:USD",
                "side": "short",
                "strategy_id": "short_a",
                "action": "exit",
                "status": "closed",
                "exit_price": 0.0312,
                "shadow_exit_price": 0.0312,
                "shadow_entry_gap_bps": 0.0,
                "stop_price": 0.0318,
                "shadow_latest_stop_price": 0.0318,
                "shadow_status": "shadow_exit_triggered",
            }
        ]
    )

    rows, summary = build_shadow_trade_reconciliation(trade_log, tolerance_bps=1.0)

    assert len(rows) == 1
    assert summary["exit_execution_parity_status"] == "pass"


def test_shadow_trade_reconciliation_reports_trigger_slippage_separately():
    trade_log = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "symbol": "BIO/USD:USD",
                "side": "long",
                "strategy_id": "long_a",
                "action": "exit",
                "status": "closed",
                "realized_exit_price": 90.0,
                "shadow_exit_price": 90.0,
                "shadow_exit_price_source": "observed_exchange_stop_fill",
                "shadow_theoretical_exit_price": 100.0,
                "shadow_entry_gap_bps": 0.0,
                "stop_price": 100.0,
                "shadow_latest_stop_price": 100.0,
                "shadow_status": "shadow_exit_triggered",
            }
        ]
    )

    rows, summary = build_shadow_trade_reconciliation(trade_log, tolerance_bps=1.0)

    assert len(rows) == 1
    assert summary["exit_execution_parity_status"] == "pass"
    assert rows.iloc[0]["live_vs_shadow_exit_gap_bps"] == pytest.approx(0.0)
    assert rows.iloc[0]["shadow_trigger_vs_live_exit_gap_bps"] == pytest.approx(-1000.0)
    assert rows.iloc[0]["shadow_exit_price_source"] == "observed_exchange_stop_fill"
    assert rows.iloc[0]["shadow_theoretical_exit_price"] == pytest.approx(100.0)
    assert summary["shadow_trigger_vs_live_exit_gap_bps"]["mean"] == pytest.approx(-1000.0)


def test_shadow_trade_reconciliation_reports_current_run_scope():
    trade_log = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "run_id": "old_run",
                "trade_id": "old_run:pos1|exit",
                "position_id": "old_run:pos1",
                "symbol": "OLD/USD:USD",
                "side": "long",
                "strategy_id": "long_a",
                "action": "exit",
                "status": "closed",
                "realized_exit_price": 90.0,
                "shadow_exit_price": 100.0,
                "shadow_entry_gap_bps": 0.0,
                "stop_price": 99.0,
                "shadow_latest_stop_price": 99.0,
                "shadow_status": "shadow_exit_triggered",
            },
            {
                "timestamp": pd.Timestamp("2026-01-01 01:00", tz="UTC"),
                "run_id": "current_run",
                "trade_id": "current_run:pos2|exit",
                "position_id": "current_run:pos2",
                "symbol": "CUR/USD:USD",
                "side": "long",
                "strategy_id": "long_a",
                "action": "exit",
                "status": "closed",
                "realized_exit_price": 100.1,
                "shadow_exit_price": 100.0,
                "shadow_entry_gap_bps": 0.0,
                "stop_price": 99.0,
                "shadow_latest_stop_price": 99.0,
                "shadow_status": "shadow_exit_triggered",
            },
        ]
    )

    _, summary = build_shadow_trade_reconciliation(
        trade_log,
        tolerance_bps=50.0,
        run_id="current_run",
    )

    assert summary["exit_execution_parity_status"] == "fail"
    assert summary["exit_gap_mismatch_rows"] == 1
    assert summary["current_run"]["exit_execution_parity_status"] == "pass"
    assert summary["current_run"]["exit_gap_mismatch_rows"] == 0
    assert summary["current_run"]["closed_shadow_rows"] == 1
    assert summary["closed_shadow_rows"] == 2


def test_shadow_trade_reconciliation_reports_open_positions_pending():
    trade_log = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "symbol": "PNUT/USD:USD",
                "side": "short",
                "strategy_id": "short_a",
                "action": "entry",
                "status": "open",
                "shadow_entry_gap_bps": 0.0,
                "stop_price": 0.0427,
                "shadow_latest_stop_price": 0.0427,
                "shadow_status": "open",
            }
        ]
    )

    _, summary = build_shadow_trade_reconciliation(trade_log, tolerance_bps=1.0)

    assert summary["closed_shadow_rows"] == 0
    assert summary["open_shadow_rows"] == 1
    assert summary["exit_execution_parity_status"] == "pending_open_positions"


@pytest.mark.parametrize(
    ("raw_status", "closed_shadow_rows", "expected_status"),
    [
        ("pass", 1, "pass"),
        ("fail", 1, "fail"),
        ("pass", 0, "pending"),
        ("pending_no_rows", 0, "pending"),
        ("pending_no_shadow_rows", 0, "pending"),
        ("pending_open_positions", 0, "pending"),
        ("pending_no_closed_rows", 0, "pending"),
    ],
)
def test_execution_parity_audit_status_preserves_missing_exit_evidence(
    raw_status, closed_shadow_rows, expected_status
):
    audit = execution_parity_audit_status(
        {
            "current_run": {
                "exit_execution_parity_status": raw_status,
                "shadow_rows": 1,
                "closed_shadow_rows": closed_shadow_rows,
                "open_shadow_rows": 1 - closed_shadow_rows,
                "exit_gap_mismatch_rows": 0,
            }
        }
    )

    assert audit["status"] == expected_status
    assert audit["reason"] == raw_status
    assert audit["scope"] == "current_run"


def test_stop_exit_fill_model_scalar_and_array_match_long_short():
    long_hit, long_px = stop_exit_fill_price(
        side="long",
        stop_px=100.0,
        candle_high=101.0,
        candle_low=99.5,
        base_gap_bps=15.0,
        alpha_through=0.05,
        max_gap_bps=75.0,
    )
    short_hit, short_px = stop_exit_fill_price(
        side="short",
        stop_px=100.0,
        candle_high=100.5,
        candle_low=99.0,
        base_gap_bps=15.0,
        alpha_through=0.05,
        max_gap_bps=75.0,
    )

    hit, px = stop_exit_fill_price_array(
        side=np.array([1.0, -1.0], dtype=np.float32),
        stop_px=np.array([100.0, 100.0], dtype=np.float32),
        candle_high=np.array([101.0, 100.5], dtype=np.float32),
        candle_low=np.array([99.5, 99.0], dtype=np.float32),
        base_gap_bps=15.0,
        alpha_through=0.05,
        max_gap_bps=75.0,
    )

    assert long_hit is True
    assert short_hit is True
    assert hit.tolist() == [True, True]
    assert px[0] == pytest.approx(long_px, rel=1e-6)
    assert px[1] == pytest.approx(short_px, rel=1e-6)
