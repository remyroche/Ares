from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_july_local_transition_diagnosis.py"
SPEC = importlib.util.spec_from_file_location("july_local_transition_diagnosis", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _frame(rows: int = 100) -> pd.DataFrame:
    anchor = pd.date_range("2026-07-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame({
        "cohort_anchor_utc": anchor, "target": np.arange(rows) % 4 == 0,
        "utc_week": MODULE._week(pd.DataFrame({"cohort_anchor_utc": anchor})),
        "mapping_provenance_role": "strict_oof",
    })


def test_tie_aware_control_reports_expected_prevalence_lift_one() -> None:
    frame = _frame()
    result = MODULE._top_tie_metrics(frame, np.full(len(frame), .5))
    assert result["cutoff_tie_rows"] == len(frame)
    assert result["expected_tie_lift"] == pytest.approx(1.0)
    assert result["best_tie_lift"] >= 1.0 >= result["worst_tie_lift"]


def test_valid_rows_requires_target_specific_availability_and_strict_population() -> None:
    frame = _frame(4)
    frame["target__active_adverse"] = [0., 1., 0., 1.]
    frame["target__active_adverse_available_utc"] = [pd.Timestamp("2026-07-02", tz="UTC"), pd.NaT, pd.Timestamp("2026-07-02", tz="UTC"), pd.Timestamp("2026-07-02", tz="UTC")]
    frame.loc[3, "mapping_provenance_role"] = "frozen_forward_oos"
    strict = MODULE.valid_rows(frame, "target__active_adverse", population="strict_oof_only")
    combined = MODULE.valid_rows(frame, "target__active_adverse", population="strict_oof_plus_frozen_forward_diagnostic")
    assert len(strict) == 2
    assert len(combined) == 3


def test_metric_row_has_calibration_false_alert_and_tie_outputs() -> None:
    frame = _frame(40)
    frame["prediction"] = np.linspace(.1, .9, len(frame))
    result = MODULE.metric_row(frame, target="target", population="test", protocol="oof", model="model")
    assert {"brier", "ece5", "false_alerts_per_selected_day", "expected_tie_lift", "worst_tie_lift"}.issubset(result)
