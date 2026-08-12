"""Focused contract tests for the frozen-ten-head C3 full-stack adapter."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ten_head_c3_full_stack_replay.py"
SPEC = importlib.util.spec_from_file_location("ten_head_c3_full_stack", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_frozen_seed_offsets_reproduce_separate_development_and_final_loops():
    head = "cap40_ordinary"
    base = MODULE.ten._head_seed(head)
    assert MODULE._head_seed_for_month(head, "2025-05", 99) == base
    assert MODULE._head_seed_for_month(head, "2025-08", 99) == base
    assert MODULE._head_seed_for_month(head, "2025-07", 99) == base + 2
    assert MODULE._head_seed_for_month(head, "2025-10", 99) == base + 2
    assert MODULE._head_seed_for_month(head, "2025-04", 1) == base + 101


def test_global_tail_selection_precedes_outcome_coverage():
    # Two highest-scored candidates are selected globally. Only one has an
    # executable outcome, which must lower coverage rather than change which
    # candidates entered the tail.
    rows = 400
    frame = pd.DataFrame({
        "__ts__": pd.to_datetime(["2025-08-01"] * rows, utc=True),
        "final_score": np.linspace(1.0, 0.0, rows),
        "policy_path_valid": [True, False] + [True] * (rows - 2),
        "policy_gross_bps": [300.0, np.nan] + [100.0] * (rows - 2),
        "policy_net_bps": [200.0, np.nan] + [0.0] * (rows - 2),
    })
    metrics, _ = MODULE._tail_metrics(
        frame, arm="test", stage="final_score",
        start=pd.Timestamp("2025-08-01", tz="UTC"),
        end=pd.Timestamp("2025-09-01", tz="UTC"),
    )
    top_half_percent = metrics.loc[metrics["tail"].eq(.005)].iloc[0]
    assert top_half_percent["selected_score_rows"] == 2
    assert top_half_percent["valid_outcomes"] == 1
    assert top_half_percent["outcome_coverage"] == .5
    assert top_half_percent["net_bps_per_trade"] == 200.0


def test_c3_overlay_input_contract_contains_no_label_or_outcome_field():
    state = [
        "k09__cluster_00__membership", "k9_entropy", "leaf_support_effective",
        "k9_model_ood_marginal",
    ]
    fields = MODULE._overlay_fields(
        ["base_score", "base_anchor_bps", "base_rank", "consensus_rank", "final_score"],
        state, include_k9_soft_memberships=True,
    )
    forbidden = ("policy_net", "policy_gross", "h12", "future", "outcome", "label")
    assert all(not any(token in field.lower() for token in forbidden) for field in fields)
    assert "k09__cluster_00__membership" in fields
