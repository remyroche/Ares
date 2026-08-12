from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_market_spine_covariance_meta.py"
_SPEC = importlib.util.spec_from_file_location("run_market_spine_covariance_meta", _PATH)
assert _SPEC and _SPEC.loader
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


def test_declared_spine_is_21_target_free_causal_fields() -> None:
    assert len(runner.SOURCE_SPINE_FIELDS) == 21
    assert len(set(runner.SOURCE_SPINE_FIELDS)) == 21
    assert set(runner.SOURCE_SPINE_FIELDS) == set(runner.CONTEXT_FIELDS) | set(runner.SOFT_STATE_FIELDS)
    assert not set(runner.SOURCE_SPINE_FIELDS) & {"p_adverse", "p_weak", "p_clear", "prequential_base_expected_net_bps"}
    assert runner._assert_feature_contract(runner.SOURCE_SPINE_FIELDS) == runner.SOURCE_SPINE_FIELDS


def test_direct_r3_base_outputs_are_legal_meta_inputs_but_not_residual_targets() -> None:
    assert runner.BASE_OUTPUT_FIELDS == (
        "p_clear", "p_adverse", "p_weak", "prequential_base_expected_net_bps",
        "base_score_clear_minus_half_adverse",
    )
    assert runner._assert_feature_contract(runner.BASE_OUTPUT_FIELDS) == runner.BASE_OUTPUT_FIELDS
    with np.testing.assert_raises(ValueError):
        runner._assert_feature_contract(("realized_residual_bps",))


def test_residual_grade_uses_declared_bps_boundaries() -> None:
    result = runner.residual_grade(np.array([-151., -150., -149., -50., -49., 50., 51., 150., 151.]))
    assert result.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4]


def test_fold_cluster_cutoff_precedes_test_and_labels_are_strict() -> None:
    fold = runner.FOLDS[0]
    assert fold.calibration_end < runner._utc(fold.test_start)
    rows = pd.DataFrame({
        "__ts__": pd.to_datetime(["2023-07-31 10:00Z", "2023-08-01 10:00Z", "2023-09-01 10:00Z"], utc=True),
    })
    rows["label_available_ts"] = rows["__ts__"] + runner.LABEL_DELAY
    calibration_start, test_start = runner._utc(fold.calibration_start), runner._utc(fold.test_start)
    train = rows.loc[rows["__ts__"].lt(calibration_start) & rows["label_available_ts"].lt(calibration_start)]
    calibration = rows.loc[rows["__ts__"].between(calibration_start, test_start, inclusive="left") & rows["label_available_ts"].lt(test_start)]
    assert train.iloc[0]["label_available_ts"] < calibration_start
    assert calibration.iloc[0]["__ts__"] == pd.Timestamp("2023-08-01 10:00", tz="UTC")


def test_long_history_transport_keeps_the_elapsed_gap_and_extends_training() -> None:
    standard = runner.FOLDS[-1]
    extended = runner.LONG_HISTORY_FOLDS[-1]
    assert standard.test_start == extended.test_start
    assert standard.calibration_start == extended.calibration_start
    assert extended.train_start == "2023-07-01"
    assert runner._utc(extended.train_start) < runner._utc(standard.train_start)
