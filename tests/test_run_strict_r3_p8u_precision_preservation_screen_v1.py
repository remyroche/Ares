from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_p8u_precision_preservation_screen_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p8u_precision_preservation_screen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_screen_requires_fixed_policy_ordinal_control() -> None:
    with pytest.raises(ValueError):
        MODULE._parse_arms("raw_bps__balanced_quantile6")


def test_policy_floor_geometry_is_economically_ordered() -> None:
    values = pd.Series([-10.0, 0.0, 50.0, 51.0, 100.0, 151.0, 251.0, 401.0])
    labels = MODULE._policy_floor_labels(values, 50.0)
    assert labels.tolist() == [0, 0, 0, 1, 1, 3, 4, 5]


def test_policy_ordinal_uses_policy_bps_not_prebinned_grade() -> None:
    frame = pd.DataFrame({
        "policy_ordinal_grade": [0, 1, 2, 3, 4, 5],
        "policy_net_bps": [-10.0, 25.0, 75.0, 150.0, 250.0, 500.0],
    })
    labels, receipt = MODULE._labels(
        frame,
        MODULE.Arm("policy_ordinal", "t0_policy_ordinal", "fixed_0_50_100_200_400"),
    )
    assert labels.tolist() == [0, 1, 2, 3, 4, 5]
    assert receipt["edges_bps"] == [0.0, 50.0, 100.0, 200.0, 400.0]


def test_continuous_geometries_are_train_only_and_bounded() -> None:
    values = pd.Series(np.linspace(-100.0, 300.0, 2_000))
    for geometry in ("balanced_quantile6", "tail_quantile6", "equal_width6"):
        labels, receipt = MODULE._continuous_labels(values, geometry)
        assert labels.min() == 0 and labels.max() == 5
        assert receipt["training_only"] is True


def test_held_months_must_span_multiple_years() -> None:
    with pytest.raises(ValueError):
        MODULE._parse_months("2026-01,2026-02,2026-03,2026-04,2026-05")
