from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE = Path(__file__).resolve().parents[1] / "scripts/prepare_causal_sr_c1_direct_target_free_panels.py"
SPEC = importlib.util.spec_from_file_location("prepare_panels", MODULE)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_slice_is_target_free_and_interval_bounded() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "__decision_ts__": ["2026-08-01T00:00Z", "2026-08-01T01:00Z", "2026-08-01T02:00Z"],
        "__symbol__": ["A", "B", "C"], "side_name": ["long"] * 3,
        "bcf_mc1_expected_bps": [51.0, 52.0, 53.0],
        "current_mc1_expected_bps": [51.0, 52.0, 53.0],
        "auction_priority_bps": [51.0, 52.0, 53.0],
    })
    got = module.slice_target_free_panel(
        frame,
        start=pd.Timestamp("2026-08-01T01:00Z"),
        end=pd.Timestamp("2026-08-01T02:00Z"),
    )
    assert got["candidate_id"].tolist() == ["b"]


def test_slice_rejects_outcome_column() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["a"], "__decision_ts__": ["2026-08-01T01:00Z"],
        "__symbol__": ["A"], "side_name": ["long"],
        "bcf_mc1_expected_bps": [51.0], "current_mc1_expected_bps": [51.0],
        "auction_priority_bps": [51.0], "policy_net_bps": [99.0],
    })
    try:
        module.slice_target_free_panel(
            frame, start=pd.Timestamp("2026-08-01T00:00Z"), end=pd.Timestamp("2026-08-02T00:00Z"),
        )
    except AssertionError as exc:
        assert "outcome" in str(exc)
    else:
        raise AssertionError("expected target-free guard")
