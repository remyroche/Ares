"""Regression tests for the strict-R3 lock-step source/label handoff."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "strict_r3_lockstep_walkforward",
    ROOT / "scripts" / "run_strict_r3_canonical_lockstep_walkforward.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _policy_outcomes() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["known"],
        "__decision_ts__": [pd.Timestamp("2025-01-01T00:00:00Z")],
        "policy_path_valid": [True],
        "policy_gross_bps": [180.0],
        "policy_net_bps": [80.0],
        "policy_exit_bar_15m": [12],
        "policy_exit_reason": ["TRAILING"],
        "policy_entry_price": [100.0],
        "policy_exit_price": [101.8],
        "policy_label_available_ts": [pd.Timestamp("2025-01-01T12:00:00Z")],
        "policy_outcome_source": ["exact"],
        "policy_cost_bps": [100.0],
    })


def test_canonical_policy_supervision_invalidates_legacy_rows(tmp_path: Path) -> None:
    outcome_path = tmp_path / "outcomes.parquet"
    _policy_outcomes().to_parquet(outcome_path, index=False)
    ledger = pd.DataFrame({
        "candidate_id": ["known", "legacy"],
        "__decision_ts__": [
            pd.Timestamp("2025-01-01T00:00:00Z"),
            pd.Timestamp("2024-12-31T00:00:00Z"),
        ],
        "policy_path_valid": [True, True],
        "policy_net_bps": [999.0, 999.0],
    })
    result, audit = RUNNER._attach_canonical_policy_supervision(
        ledger, outcome_path, end=pd.Timestamp("2025-02-01T00:00:00Z"),
    )
    known = result.loc[result["candidate_id"].eq("known")].iloc[0]
    legacy = result.loc[result["candidate_id"].eq("legacy")].iloc[0]
    assert known["policy_net_bps"] == pytest.approx(80.0)
    assert bool(known["policy_path_valid"])
    assert np.isnan(legacy["policy_net_bps"])
    assert not bool(legacy["policy_path_valid"])
    assert audit["canonical_policy_identity_rows"] == 1
    assert audit["legacy_policy_rows_invalidated"] == 1


def test_source_aligned_ledger_manifest_is_required(tmp_path: Path) -> None:
    ledger_path = tmp_path / "prequential_stack_ledger.parquet"
    ledger_path.touch()
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "source_panel_sha256": "expected",
        "reference_window_days": 28,
    }))
    assert RUNNER._require_source_aligned_ledger(ledger_path, "expected")["source_panel_sha256"] == "expected"
    with pytest.raises(ValueError, match="not generated from this target-free source"):
        RUNNER._require_source_aligned_ledger(ledger_path, "other")


def test_canonical_runner_rejects_legacy_42_day_prequential_ledger(tmp_path: Path) -> None:
    ledger_path = tmp_path / "prequential_stack_ledger.parquet"
    ledger_path.touch()
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "source_panel_sha256": "expected",
        "reference_window_days": 42,
    }))
    with pytest.raises(ValueError, match="reference_window_days=28"):
        RUNNER._require_source_aligned_ledger(ledger_path, "expected")


def test_lockstep_manifest_declares_physical_reference_window() -> None:
    source = (ROOT / "scripts" / "run_strict_r3_canonical_lockstep_walkforward.py").read_text()
    assert RUNNER.REFERENCE_DAYS == 28
    assert '"reference_window_days": REFERENCE_DAYS' in source


def test_upstream_handoff_requires_complete_identity_coverage() -> None:
    working = pd.DataFrame({"candidate_id": ["one"]}).set_index("candidate_id", drop=False)
    score = pd.DataFrame({"candidate_id": ["one", "missing"]})
    with pytest.raises(ValueError, match="does not cover every scored target-free candidate"):
        RUNNER._apply_upstream_scores(working, score)


def test_prequential_ledger_projection_keeps_the_complete_live_contract() -> None:
    base_fields = [f"base_{index:03d}" for index in range(120)]
    columns = RUNNER._prequential_ledger_columns(base_fields)
    required = {
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "r3_class", "r3_label_available_ts", "policy_net_bps",
        "policy_label_available_ts", "h12_label_valid",
        "h12_label_available_ts", "h12_tp6_sl4_net_bps",
        "prequential_base_score", "prequential_base_rank42",
        "prequential_base_anchor_bps", "prequential_consensus_rank",
        "prequential_residual_rank", "prequential_upstream",
        "stack_is_prequential",
    }
    assert required.issubset(columns)
    assert set(base_fields).issubset(columns)
    assert "policy_outcome_source" not in columns
    assert len(columns) == len(set(columns))


def test_compact_prequential_projection_defers_only_frozen_base_fields() -> None:
    base_fields = [f"base_{index:03d}" for index in range(120)]
    compact = RUNNER._prequential_ledger_columns(base_fields, include_base_fields=False)
    assert not set(base_fields).intersection(compact)
    assert {
        "candidate_id", "__decision_ts__", "r3_class", "policy_net_bps",
        "prequential_base_rank42", "stack_is_prequential",
    }.issubset(compact)
