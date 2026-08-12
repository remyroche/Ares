from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "report_strict_r3_self_distillation_funnel.py"
SPEC = importlib.util.spec_from_file_location("strict_r3_distillation_report", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_global_tail_selection_is_not_per_timestamp() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": [str(index) for index in range(100)],
            "final_score": np.arange(100, dtype=float),
            "__decision_ts__": pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC"),
            "policy_net_bps": np.arange(100, dtype=float),
        }
    )
    selected = MODULE._selected(frame, 0.02)
    assert selected["candidate_id"].tolist() == ["99", "98"]


def test_paired_day_bootstrap_detects_uniform_positive_delta() -> None:
    timestamps = pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC")
    control = pd.DataFrame({"__decision_ts__": timestamps, "policy_net_bps": 10.0})
    challenger = pd.DataFrame({"__decision_ts__": timestamps, "policy_net_bps": 20.0})
    result = MODULE._paired_day_bootstrap(control, challenger, draws=500, seed=7)
    assert result["delta_mean_bps"] == 10.0
    assert result["delta_ci025_bps"] == 10.0
    assert result["probability_delta_positive"] == 1.0


def test_valid_outcome_filter_does_not_convert_missing_path_to_failure() -> None:
    frame = pd.DataFrame(
        {
            "policy_path_valid": [True, False, True],
            "policy_net_bps": [25.0, -500.0, np.nan],
        }
    )
    valid = MODULE._valid_outcomes(frame)
    assert valid.index.tolist() == [0]
    assert valid["policy_net_bps"].gt(0).mean() == 1.0
