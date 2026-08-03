from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_febmar_true_signal_density_overlap.py"
SPEC = importlib.util.spec_from_file_location("true_signal_density_overlap", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _frame(month: str, scores: list[float]) -> pd.DataFrame:
    rows = len(scores); hours = np.repeat(pd.Timestamp(f"{month}-01T00:00:00Z"), rows)
    return pd.DataFrame({"candidate_id": [f"{month}-{idx}" for idx in range(rows)], "base_oof_score": scores, "__ts__": hours, "__symbol__": ["BTC", "BTC", "ETH", "ETH", "SOL", "SOL"][:rows], "side_name": ["long", "short"] * (rows // 2)})


def test_frozen_definition_is_unchanged_when_march_scores_change() -> None:
    february = _frame("2025-02", [0., 1., 2., 3., 4., 5.])
    first = MODULE.freeze_february_score_definition(february)
    _ = _frame("2025-03", [100., 101., 102., 103., 104., 105.])
    second = MODULE.freeze_february_score_definition(february)
    assert first == second
    assert first["near_cutoff_band_width"] > 0


def test_signal_density_uses_score_counts_not_eligible_asset_count_as_feature() -> None:
    frame = _frame("2025-02", [0., 1., 2., 3., 4., 5.])
    frozen = MODULE.freeze_february_score_definition(frame)
    joined, hourly = MODULE.materialize_signal_density(frame, frozen)
    assert hourly.iloc[0].eligible_asset_count == 3
    assert hourly.iloc[0].eligible_rows == 6
    assert "eligible_asset_count" not in {"above_threshold_count", "above_threshold_fraction", "above_threshold_long_short_imbalance", "near_cutoff_fraction"}
    assert joined.above_threshold_fraction.between(0, 1).all()
