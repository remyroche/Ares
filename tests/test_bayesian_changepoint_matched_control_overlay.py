import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_bayesian_changepoint_matched_control_overlay.py"
SPEC = importlib.util.spec_from_file_location("bocpd_matched_overlay", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_threshold_accepts_train_only_separation() -> None:
    positives = pd.DataFrame({"synchronized_break_score": np.repeat(6.0, 16)})
    controls = pd.DataFrame({"synchronized_break_score": np.linspace(0.0, 2.0, 64)})
    result = MODULE._select_threshold(positives, controls)
    assert result["status"] == "accepted"
    assert result["train_lift"] >= 1.5
    assert result["train_matched_fpr"] <= 0.15


def test_threshold_rejects_indistinguishable_scores() -> None:
    scores = np.linspace(0.0, 2.0, 64)
    positives = pd.DataFrame({"synchronized_break_score": scores[:16]})
    controls = pd.DataFrame({"synchronized_break_score": scores})
    result = MODULE._select_threshold(positives, controls)
    assert result["status"] == "no_train_only_discriminative_threshold"


def test_matched_controls_use_observable_context_and_hour() -> None:
    panel = pd.DataFrame({
        "__ts__": pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC"),
        "target": [1, 1, 0, 0, 0, 0, 0, 0],
        "candidate_rows": [10, 20, 10, 20, 10, 20, 10, 20],
        "parent_rank_mean": [0.92, 0.95, 0.92, 0.95, 0.92, 0.95, 0.92, 0.95],
        "hour": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    positive, controls = MODULE._matched_controls(panel, controls_per_positive=2)
    assert len(positive) == 2
    assert len(controls) == 4
    assert controls["target"].eq(0).all()
    assert controls["match_distance"].ge(0).all()
