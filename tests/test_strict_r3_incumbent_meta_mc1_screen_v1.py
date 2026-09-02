from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_incumbent_meta_mc1_screen_v1.py"
SPEC = importlib.util.spec_from_file_location("incumbent_meta_mc1_screen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_score_bands_are_timestamp_local() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "__decision_ts__": pd.to_datetime([
            "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z", "2026-01-02T00:00:00Z",
        ]),
        "final_score": [.9, .1, .9, .1],
    })
    bands = MODULE._rank_bands(frame)
    assert bands.tolist() == [2, 7, 2, 7]


def test_robust_mean_trims_only_when_support_allows() -> None:
    assert MODULE._robust_mean([1.0, 2.0, 3.0], trim=.20) == 2.0
    assert MODULE._robust_mean([0.0, 0.0, 100.0, 100.0, 1_000.0], trim=.20) == 200.0 / 3.0
