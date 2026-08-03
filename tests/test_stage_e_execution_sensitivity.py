from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_stage_e_execution_sensitivity import (
    DEFAULT_OUTPUT, ESTIMATOR_STRESSES, LATENCIES, SLIPPAGES,
    adverse_exit_fill_proxy, load_frozen_decisions, replay_scenario,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = DEFAULT_OUTPUT / "stage_e_execution_sensitivity.parquet"
ROW_AUDIT = DEFAULT_OUTPUT / "stage_e_execution_sensitivity_row_audit.parquet"
MANIFEST = DEFAULT_OUTPUT / "run_manifest.json"


def test_latency_replay_keeps_model_decisions_frozen() -> None:
    frozen = load_frozen_decisions().set_index("candidate_id")
    rows = pd.read_parquet(ROW_AUDIT).set_index("candidate_id")
    assert rows.index.isin(frozen.index).all()
    assert np.array_equal(rows.action, frozen.loc[rows.index, "action"])
    manifest = json.loads(MANIFEST.read_text())
    contract = manifest["decision_contract"]
    assert contract == {
        "source": "canonical Stage-D v9 final_oos selected-margin rows",
        "margin_bps": 0.0, "model_refit": False,
        "predictions_recomputed": False, "actions_recomputed": False,
    }


def test_slippage_replay_applies_incremental_cost_once() -> None:
    rows = pd.read_parquet(ROW_AUDIT)
    assert np.allclose(rows.replayed_exit_gross_bps - rows.replayed_exit_cost_bps, rows.replayed_exit_net_bps)
    results = pd.read_parquet(RESULTS)
    grid = results[(results.population == "fixed_common_support") & (results.slice == "all") & (results.latency_minutes == 0) & (results.exit_estimator_stress_bps == 0)].sort_values("added_exit_slippage_bps")
    base = grid.iloc[0]
    # Only EXIT_NOW rows bear the extra cost; frozen continue decisions do not.
    expected = -(grid.added_exit_slippage_bps - base.added_exit_slippage_bps) * base.exit_rate
    actual = grid.uplift_vs_always_continue_bps - base.uplift_vs_always_continue_bps
    assert np.allclose(actual, expected, atol=1e-9)


def test_execution_sensitivity_grid_is_complete_and_on_identical_rows() -> None:
    results = pd.read_parquet(RESULTS)
    grid = results[(results.population == "fixed_common_support") & (results.slice == "all")]
    assert len(grid) == len(LATENCIES) * len(SLIPPAGES) * len(ESTIMATOR_STRESSES)
    assert grid.candidate_id_sha256.nunique() == 1
    assert grid.rows.nunique() == 1
    assert set(grid.latency_minutes) == set(LATENCIES)
    assert set(grid.added_exit_slippage_bps) == set(SLIPPAGES)
    assert set(grid.exit_estimator_stress_bps) == set(ESTIMATOR_STRESSES)


def test_unperturbed_full_population_reproduces_v9() -> None:
    frozen = load_frozen_decisions()
    result = pd.read_parquet(RESULTS)
    full = result[result.population.eq("canonical_full_population")].iloc[0]
    assert full.rows == len(frozen)
    assert full.policy_net_bps == np.mean(frozen.policy_net_bps)
    assert full.uplift_vs_always_continue_bps == np.mean(frozen.incremental_vs_always_continue_bps)


def test_barrier_ambiguity_slices_are_reported_on_frozen_grid() -> None:
    results = pd.read_parquet(RESULTS)
    subset = results[results.population.eq("fixed_common_support")]
    assert {"large_clear_jump", "clear_adverse_geometry_close", "next_fill_materially_differs"}.issubset(set(subset.slice))
    assert set(subset.loc[subset.slice.ne("all"), "slice_value"]) == {"TRUE", "FALSE"}


def test_manifest_binds_v9_and_never_refits() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["schema"] == "stage_e4_frozen_decision_execution_sensitivity_v1"
    assert manifest["decision_contract"]["model_refit"] is False
    assert manifest["decision_contract"]["actions_recomputed"] is False
    assert "stage_d_compact_action_model_20260731_v9" in next(k for k in manifest["inputs"] if k.endswith("stage_d_action_policy_replay.parquet"))


def test_maximum_positive_latency_slippage_is_derived_from_zero_stress_grid() -> None:
    manifest = json.loads(MANIFEST.read_text())
    result = pd.read_parquet(RESULTS)
    grid = result[(result.population == "fixed_common_support") & (result.slice == "all") & (result.exit_estimator_stress_bps == 0) & (result.uplift_vs_always_continue_bps > 0)].sort_values(["latency_minutes", "added_exit_slippage_bps"], ascending=[False, False])
    expected = None if grid.empty else grid.iloc[0]
    actual = manifest["maximum_positive_combination"]
    if expected is None:
        assert actual is None
    else:
        assert actual["latency_minutes"] == expected.latency_minutes
        assert actual["added_exit_slippage_bps"] == expected.added_exit_slippage_bps
        assert actual["uplift_vs_always_continue_bps"] == expected.uplift_vs_always_continue_bps
