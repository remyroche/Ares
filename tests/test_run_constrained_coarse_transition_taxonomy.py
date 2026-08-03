from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "coarse_taxonomy", ROOT / "scripts/run_constrained_coarse_transition_taxonomy.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_profile_uses_fixed_sparse_scale_floor() -> None:
    field = "sequence__breadth_dispersion__mean_168h"
    spec = pd.DataFrame(
        [{"group": "breadth", "phase": "precondition_168h", "field": field}]
    )
    # Tiny but non-zero IQR must not explode into an artificial prototype axis.
    frame = pd.DataFrame({field: [0.0, 0.001, 0.002, 0.003]})
    profile = MODULE.Profile(spec).fit(frame)
    assert profile.scale[0] == 1.0
    transformed = profile.transform(frame)
    assert transformed.shape == (4, len(MODULE.GROUPS) * len(MODULE.PHASES))
    assert np.isfinite(transformed).all()


def test_hungarian_alignment_returns_all_coarse_slots() -> None:
    left = np.array([[1.0, 0.0], [0.0, 1.0]])
    right = np.array([[0.0, 1.0], [1.0, 0.0]])
    mean, minimum, pairs = MODULE.align(left, right)
    assert mean == 1.0
    assert minimum == 1.0
    assert len(__import__("json").loads(pairs)) == 2
