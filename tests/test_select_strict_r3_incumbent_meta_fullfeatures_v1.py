from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "select_strict_r3_incumbent_meta_fullfeatures_v1.py"
SPEC = importlib.util.spec_from_file_location("incumbent_meta_fullfeatures", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_over_screen_direction_matches_inverted_head_authority() -> None:
    arm = MODULE.meta_grid.Arm("over", "over", "atr", "timestamp", 1.25)
    labels = np.array([1, 0, -1], dtype=np.int32)
    residual = np.array([-100.0, 10.0, 0.0])
    result = MODULE._screen_direction(arm, labels, residual)
    # A high final over-head rank denotes less adverse risk because the model
    # prediction is inverted after fitting; the screen must use that direction.
    assert result[0] == -1.0
    assert result[1] == 0.0
    assert np.isnan(result[2])


def test_redundancy_veto_keeps_best_representative() -> None:
    summary = pd.DataFrame({
        "feature": ["best", "duplicate", "independent"],
        "family": ["a", "a", "b"],
        "screen_score": [.9, .8, .7],
    })
    values = pd.DataFrame({
        "best": np.arange(1_000, dtype=float),
        "duplicate": np.arange(1_000, dtype=float) * 2.0,
        "independent": np.tile([0.0, 1.0], 500),
    })
    result = MODULE._redundancy_veto(summary, values, ceiling=.98, keep_limit=3).set_index("feature")
    assert bool(result.loc["best", "kept_after_redundancy"])
    assert not bool(result.loc["duplicate", "kept_after_redundancy"])
    assert result.loc["duplicate", "redundancy_representative"] == "best"
    assert bool(result.loc["independent", "kept_after_redundancy"])


def test_contracts_are_explicit_and_limited_to_veto_survivors(tmp_path: Path) -> None:
    features = [f"f_{index:03d}" for index in range(130)]
    ranked = pd.DataFrame({
        "feature": features,
        "final_selection_score": np.linspace(1.0, 0.0, len(features)),
    })
    veto = pd.DataFrame({
        "feature": features,
        "kept_after_redundancy": [True] * 120 + [False] * 10,
    })
    arm = MODULE.meta_grid.Arm("magnitude", "magnitude", "sqrt_atr", "base_band_block28", classes=5)
    paths = MODULE._contracts(out=tmp_path, arm=arm, ranked=ranked, redundancy=veto)
    assert {"f120", "f90", "f70", "f50", "f35"}.issubset(paths)
    payload = __import__("json").loads(Path(paths["f70"]).read_text())
    assert payload["feature_count"] == 70
    assert len(payload["features"]) == 70
    assert set(payload["features"]).issubset(set(features[:120]))
