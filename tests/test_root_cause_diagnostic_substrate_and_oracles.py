"""Correctness tests for the Stage 0/1 root-cause diagnostic substrate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_root_cause_diagnostic_substrate import ContractError, build_substrate
from scripts.run_root_cause_oracle_ladder import run as run_oracle_ladder


def _timestamps() -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    decision = pd.Timestamp("2024-01-01T01:00:00Z")
    return decision - pd.Timedelta(hours=1), decision, decision + pd.Timedelta(hours=12)


def _write_stage0_inputs(tmp_path: Path, symbol: str = "BTC/USD:USD") -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cutoff, decision, end = _timestamps()
    ids = ["a", "b"]
    alignment = pd.DataFrame({
        "candidate_id": ids, "symbol": [symbol, symbol], "side": ["long", "short"],
        "decision_ts": [decision, decision], "feature_cutoff_ts": [cutoff, cutoff], "entry_ts": [decision, decision],
        "label_end_ts": [end, end], "label_available_ts": [end, end], "execution_policy_id": ["p", "p"],
        "cost_model_id": ["c", "c"], "policy_archetype": ["r", "r"], "execution_geometry_key": ["g", "g"],
        "execution_geometry_source": ["frozen", "frozen"], "exact_h12_gross_bps": [150.0, -50.0],
        "row_cost_bps": [100.75, 99.75], "exact_h12_net_bps": [49.25, -149.75], "estimated_spread_bps": [3.0, 3.0],
        "entry_half_spread_bps": [1.5, 1.5], "exit_half_spread_bps": [1.5, 1.5], "exit_reason": ["h12", "h12"], "exit_hour": [12, 12],
    })
    events = pd.DataFrame({
        "candidate_id": ids, "postcost_h0_event": ["clear_cost_first", "timeout"], "postcost_h0_favorable_minute": [10, np.nan],
        "postcost_h0_adverse_minute": [np.nan, np.nan], "postcost_h0_resolved_minute": [10, np.nan],
        "postcost_h25_event": ["clear_cost_first", "timeout"], "postcost_h25_favorable_minute": [12, np.nan],
        "postcost_h25_adverse_minute": [np.nan, np.nan], "postcost_h25_resolved_minute": [12, np.nan], "fixed_cost_bps": [0.0, 0.0],
    })
    persistence = pd.DataFrame({
        "candidate_id": ids, "postcost_h0_four_state": ["clear_then_retained", "timeout"],
        "postcost_h0_retained_net": [True, False], "postcost_h0_giveback_after_clear": [False, False],
    })
    counter = pd.DataFrame({
        "candidate_id": ["a"], "action_decision_ts": [decision + pd.Timedelta(minutes=10)],
        "action_execution_ts": [decision + pd.Timedelta(minutes=11)], "net_continue_gross_bps": [150.0],
        "net_exit_now_gross_bps": [40.0], "delta_continue_bps": [110.0],
    })
    oof = pd.DataFrame({
        "candidate_id": ids, "__ts__": [cutoff, cutoff], "side_name": ["long", "short"],
        "score_base_alpha": [0.1, -0.1], "score_residual_alpha": [0.2, -0.2],
        "score_base_expected_ev": [10.0, -10.0], "score_residual_expected_ev": [20.0, -20.0],
        "score_residual_delta_alpha": [0.05, -0.05], "stack_lineage": ["frozen_pf_2022aug_2024"] * 2,
        "residual_fold": [0, 0], "residual_is_oof": [True, True],
    })
    paths = {name: tmp_path / f"{name}.parquet" for name in ("alignment", "events", "persistence", "counter", "oof")}
    alignment.to_parquet(paths["alignment"]); events.to_parquet(paths["events"]); persistence.to_parquet(paths["persistence"])
    counter.to_parquet(paths["counter"]); oof.to_parquet(paths["oof"])
    paths["policy"] = tmp_path / "policy.json"
    paths["policy"].write_text(json.dumps({"strategies": [{"selected": True, "cost_pct_per_side": 0.005}]}))
    return paths


def test_stage0_uses_clean_action_arithmetic_and_fails_closed_on_product_mixing(tmp_path: Path) -> None:
    paths = _write_stage0_inputs(tmp_path)
    ledger, reconciliation, _ = build_substrate(
        alignment_path=paths["alignment"], events_path=paths["events"], persistence_path=paths["persistence"],
        counterfactual_path=paths["counter"], oof_path=paths["oof"], policy_path=paths["policy"],
    )
    action = ledger.loc[ledger.candidate_id.eq("a")].iloc[0]
    assert action.action_delta_clean_bps == pytest.approx(110.0)
    assert action.action_continue_clean_value_bps == pytest.approx(50.0)
    assert action.action_exit_clean_value_bps == pytest.approx(-60.0)
    assert "known_row_cost_bps" not in ledger.columns
    assert set(ledger.contract_family) == {"PF_USD_LINEAR_PERPETUAL"}
    assert set(ledger.settlement_currency) == {"USD"}
    assert reconciliation.max_abs_fee_reconciliation_bps.max() == pytest.approx(0.0)
    # The OOF timestamp is the causal feature cutoff, not the later entry.
    assert ledger.score_ts.equals(ledger.feature_cutoff_ts)
    assert {"score_base_expected_ev", "score_residual_expected_ev"}.issubset(ledger.columns)

    bad_paths = _write_stage0_inputs(tmp_path / "bad", symbol="BTC/USD:BTC")
    with pytest.raises(ContractError, match="mixed or non-PF"):
        build_substrate(
            alignment_path=bad_paths["alignment"], events_path=bad_paths["events"], persistence_path=bad_paths["persistence"],
            counterfactual_path=bad_paths["counter"], oof_path=bad_paths["oof"], policy_path=bad_paths["policy"],
        )


def _oracle_ledger() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
    gross = np.arange(-100.0, 100.0, 10.0)
    return pd.DataFrame({
        "candidate_id": [f"c{i:02d}" for i in range(len(timestamps))], "side": np.where(np.arange(len(timestamps)) % 2, "short", "long"),
        "symbol": "BTC/USD:USD", "decision_ts": timestamps, "feature_cutoff_ts": timestamps - pd.Timedelta(hours=1),
        "gross_h12_bps": gross, "net_h12_bps": gross - 100.0,
        "postcost_h0_event": np.where(gross > 0, "clear_cost_first", "timeout"),
        "postcost_h25_event": np.where(gross > 20, "clear_cost_first", "timeout"),
        "postcost_h0_four_state": np.where(gross > 40, "clear_then_retained", np.where(gross > 0, "clear_then_giveback", "timeout")),
        "policy_archetype": "r", "action_continue_execution_adjusted_gross_bps": np.where(gross > 0, gross, np.nan),
        "action_exit_execution_adjusted_gross_bps": np.where(gross > 0, gross + 5.0, np.nan),
        "score_base_alpha": gross / 100.0, "score_residual_alpha": gross / 200.0, "score_residual_delta_alpha": gross / 400.0,
        "residual_is_oof": True,
    })


def test_oracle_ladder_is_global_reports_net_and_preserves_unavailability(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.parquet"
    manifest_path = tmp_path / "substrate.json"
    output = tmp_path / "oracle"
    _oracle_ledger().to_parquet(ledger_path)
    manifest_path.write_text("{}")
    result = run_oracle_ladder(ledger_path, manifest_path, output)
    rows = pd.read_parquet(output / "oracle_ladder_results.parquet")
    o2 = rows.loc[(rows.oracle == "O2_realised_net_h12") & (rows.top_fraction == 0.2) & (rows.slice_kind == "pooled")].iloc[0]
    assert o2.selection_scope == "GLOBAL_TOP_K"
    assert o2.mean_net_bps == pytest.approx(result["phase1_decision"]["o2_net_bps"])
    assert o2.mean_net_bps < o2.mean_evaluation_gross_bps
    o4 = rows.loc[(rows.oracle == "O4_hindsight_permitted_action") & (rows.top_fraction == 0.2) & (rows.slice_kind == "pooled")].iloc[0]
    assert o4.net_status.startswith("NOT_AVAILABLE")
    sensitivity = pd.read_parquet(output / "target_sensitivity_results.parquet")
    assert sensitivity.loc[sensitivity.sensitivity.eq("entry_delay_1m"), "status"].iloc[0].startswith("NOT_AVAILABLE")
    assert sensitivity.loc[sensitivity.sensitivity.eq("fixed_cost_hurdle_0bps_vs_fixed_cost_hurdle_25bps"), "event_label_agreement"].iloc[0] < 1.0
