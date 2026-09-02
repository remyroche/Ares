"""Regression coverage for the strict short MC1/BCF OOS driver."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_strict_r3_short_p0_static_mc1_oof.py"
    spec = importlib.util.spec_from_file_location("short_static_mc1_oof", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["old", "same_boundary", "held", "later"],
        "__decision_ts__": pd.to_datetime([
            "2024-12-30T00:00:00Z", "2024-12-31T00:00:00Z",
            "2025-01-01T03:00:00Z", "2025-01-02T00:00:00Z",
        ], utc=True),
        "policy_label_available_ts": pd.to_datetime([
            "2024-12-31T23:59:59Z", "2025-01-01T00:00:00Z",
            "2025-01-01T15:00:00Z", "2025-01-02T12:00:00Z",
        ], utc=True),
        "policy_path_valid": [True, True, True, True],
        "policy_net_bps": [80.0, -20.0, 50.0, 10.0],
        "final_score": [0.8, 0.7, 0.9, 0.6],
        "side_name": ["short"] * 4,
    })


def test_short_mapper_driver_passes_only_strictly_prior_resolved_history():
    module = _module()
    observed: list[set[str]] = []

    def score(current: pd.DataFrame, history: pd.DataFrame, decision: pd.Timestamp) -> pd.DataFrame:
        assert decision == pd.Timestamp("2025-01-01T00:00:00Z")
        observed.append(set(history["candidate_id"]))
        return pd.DataFrame({"candidate_id": current["candidate_id"]})

    output = module._daily_score(
        _rows(),
        start=pd.Timestamp("2025-01-01T00:00:00Z"),
        end=pd.Timestamp("2025-01-02T00:00:00Z"),
        score=score,
        family="test",
    )
    assert output["candidate_id"].tolist() == ["held"]
    assert observed == [{"old"}]


def test_dual_metrics_never_count_one_family_only_admission():
    module = _module()
    frame = pd.DataFrame({
        "candidate_id": ["both", "current_only", "bcf_only"],
        "__decision_ts__": pd.to_datetime(["2025-01-01T00:00:00Z"] * 3, utc=True),
        "policy_path_valid": [True, True, True],
        "policy_net_bps": [100.0, 200.0, 300.0],
        "mc1_d2_expected_net_bps": [60.0, 60.0, 20.0],
        "bcf_mc1_expected_net_bps": [60.0, 20.0, 60.0],
    })
    metrics = module._dual_metric_rows(frame)
    pooled_50 = metrics.loc[
        metrics["month"].eq("pooled") & metrics["threshold_bps"].eq(50.0)
    ].iloc[0]
    assert pooled_50["trades"] == 1
    assert np.isclose(pooled_50["net_bps_per_trade"], 100.0)
