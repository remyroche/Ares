import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "banded_catboost", ROOT / "scripts" / "run_strict_r3_p8u_meta_banded_catboost_mapper_v1.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_bcf_bands_are_disjoint_and_below_contract_is_unavailable():
    values = pd.Series([29.9, 30.0, 49.9, 50.0, 74.9, 75.0, 99.9, 100.0, 149.9, 150.0, 1000.0])
    assert MODULE._band(values).tolist() == [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_model_target_is_clipped_before_fit():
    assert np.clip(np.array([-900.0, -300.0, 600.0, 1000.0]), MODULE.TARGET_LOW_BPS, MODULE.TARGET_HIGH_BPS).tolist() == [-300.0, -300.0, 600.0, 600.0]


def test_month_parser_rejects_unordered_folds():
    try:
        MODULE._months("2026-02,2026-01,2026-03,2026-04,2026-05")
    except ValueError:
        pass
    else:
        raise AssertionError("unordered monthly folds must fail")
