from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import audit_stage_e_a0_causal_sufficiency as e1


def test_estimated_exit_net_uses_decision_time_price_only() -> None:
    ts = np.asarray([1, 2, 3], dtype=np.int64)
    close = np.asarray([100.0, 101.0, 102.0])
    got = e1.reconstruct_prefix_a0(
        side="long", stop_index=2, entry_price=100.0,
        prefix_timestamp=ts, prefix_close=close,
    )
    assert np.isclose(got["gross_return_at_action_bps"], 200.0)
    assert "estimated_net_if_exit_now_bps" not in got
    assert "known_row_cost_bps" in e1.CAUSALLY_UNRECONSTRUCTABLE
    assert got["path_observed_through_bar_open_ns"] == 3


def test_estimated_exit_net_does_not_equal_realised_next_fill_by_construction() -> None:
    signature = inspect.signature(e1.reconstruct_prefix_a0)
    assert not ({"action_exit_raw_open", "action_exit_executable_price", "net_exit_now_bps"} & set(signature.parameters))
    got = e1.reconstruct_prefix_a0(
        side="long", stop_index=0, entry_price=100.0,
        prefix_timestamp=np.asarray([1]), prefix_close=np.asarray([102.0]),
    )
    assert "estimated_net_if_exit_now_bps" not in got
    assert "future exit path" in e1.CAUSALLY_UNRECONSTRUCTABLE["known_row_cost_bps"]


def test_no_target_column_used_to_reconstruct_a0() -> None:
    assert not e1.TARGET_OR_FUTURE_COLUMNS.intersection(e1.RECON_COUNTER_COLUMNS)
    tree = ast.parse(inspect.getsource(e1.reconstruct_prefix_a0))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not e1.TARGET_OR_FUTURE_COLUMNS.intersection(names | attrs)


def test_a0_inventory_covers_all_61_fields_and_selected_folds() -> None:
    frame = e1.build_inventory()
    assert len(frame) == frame.feature_name.nunique() == 61
    assert set(frame.category).issubset({
        "action_state", "entry_static", "cost", "policy_geometry",
        "base_or_upstream_output", "symbol_or_side_identity", "other",
    })
    assert frame.selected_fold_count.between(0, 8).all()
    assert {"gross_return_at_action_bps", "estimated_net_if_exit_now_bps", "time_to_clear_minutes"}.issubset(set(frame.feature_name))


def test_a0_independent_recomputation_matches_sealed_features() -> None:
    output = e1.DEFAULT_OUTPUT
    if not output.exists():
        return
    result = json.loads((output / "stage_e_independent_feature_recomputation.json").read_text())
    assert result["passed"] is False
    assert "known_row_cost_bps" in result["selected_failures"]
    assert "estimated_net_if_exit_now_bps" in result["causal_defects"]
    assert result["population_rows"] == 108139
    assert result["rows"] == e1.PREFIX_AUDIT_ROWS


def test_e1_artifact_proves_estimate_differs_from_realised_fill() -> None:
    output = e1.DEFAULT_OUTPUT
    if not output.exists():
        return
    metrics = pd.read_parquet(output / "stage_e_a0_exit_value_diagnostics.parquet")
    row = metrics.query("latency_minutes_beyond_canonical == 0 and dimension == 'overall'").iloc[0]
    # Once the selected outcome-derived cost proves the causal failure, the
    # expensive path comparison is deliberately bounded to the sealed prefix
    # sample.  Population coverage is asserted separately in the recomputation
    # manifest.
    assert row.rows == e1.PREFIX_AUDIT_ROWS
    assert row.exact_equal_fraction < 1.0
    assert np.isfinite(row.mae_bps)


def test_stage_e1_is_bound_to_canonical_stage_d_inputs() -> None:
    assert e1.COUNTER_ROOT.name.endswith("_v2")
    assert e1.FEATURE_ROOT.name.endswith("_v5")
    assert e1.MODEL_ROOT.name.endswith("_v9")
    assert e1.MODEL_REPRO_ROOT.name.endswith("_v10")


def test_prefix_decode_drops_future_suffix() -> None:
    raw = json.dumps({"timestamp": [1, 2, 3, 4], "close": [10.0, 11.0, 999.0, 888.0]})
    ts, close = e1._decode_prefix(raw, 1)
    assert ts.tolist() == [1, 2]
    assert close.tolist() == [10.0, 11.0]
