"""Focused contract tests for the offline rich-extension HPO funnel."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_strict_r3_exact_1m_rich_extensions_hpo.py"
SPEC = importlib.util.spec_from_file_location("rich_extensions_hpo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _paths() -> object:
    # Archive identity order deliberately differs from chronological order.
    rows = pd.DataFrame({
        "candidate_id": ["late", "early"],
        "timestamp": pd.to_datetime(["2025-02-02T00:00:00Z", "2025-02-01T00:00:00Z"]),
        "entry_ts": pd.to_datetime(["2025-02-02T00:00:00Z", "2025-02-01T00:00:00Z"]),
    })
    return MODULE.ExactPaths(
        rows=rows,
        entry=np.asarray([2.0, 1.0], dtype=np.float32),
        atr=np.asarray([0.2, 0.1], dtype=np.float32),
        high=np.asarray([[2.1] * 720, [1.1] * 720], dtype=np.float32),
        low=np.asarray([[1.9] * 720, [0.9] * 720], dtype=np.float32),
        close=np.asarray([[2.0] * 720, [1.0] * 720], dtype=np.float32),
        manifest={}, audit={},
    )


def test_resort_reorders_rows_and_all_path_matrices_together() -> None:
    result = MODULE._resort(_paths())
    assert result.rows["candidate_id"].tolist() == ["early", "late"]
    assert result.entry.tolist() == [1.0, 2.0]
    assert np.allclose(result.high[0], 1.1)
    assert np.allclose(result.high[1], 2.1)


def test_extension_funnel_is_predeclared_and_keeps_default_control() -> None:
    base = MODULE.RichExitExtensions()
    for stage in ("soft_trailing", "no_progress", "local_peak_velocity", "smooth_protection"):
        options = MODULE._stage_options(stage, base)
        assert options
        assert options[0] == base
        assert len({MODULE._extension_key(value) for value in options}) == len(options)
        for value in options:
            value.validate()
    no_progress = MODULE._stage_options("no_progress", base)
    assert any(
        value.no_progress_origin == "mae"
        and value.no_progress_start_minutes in {45, 60, 90}
        and value.no_progress_required_mfe_atr > 0.0
        for value in no_progress
    )
    assert any(
        value.no_progress_origin == "mae"
        and value.no_progress_min_mfe_slope_atr_per_hour > 0.0
        for value in no_progress
    )
    trailing = MODULE._stage_options("soft_trailing", base)
    assert any(
        value.giveback_confirmation_window_minutes > 1
        and value.trailing_ratchet_step_atr > 0.0
        for value in trailing
    )
    assert any(
        value.trail_hysteresis_atr > 0.0
        and value.trailing_ratchet_step_atr > 0.0
        for value in trailing
    )


def test_protocol_separates_2025_selection_from_2026_frozen_evaluation() -> None:
    assert MODULE.TUNE_START.year == 2025
    assert MODULE.TUNE_END.year == 2025
    assert MODULE.SELECT_START.year == 2025
    assert MODULE.SELECT_END.year == 2026
    assert MODULE.FROZEN_START.year == 2026
    assert MODULE.SELECT_END == MODULE.FROZEN_START


def test_optional_acceleration_sample_keeps_complete_candidate_competition_per_timestamp() -> None:
    rows = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "timestamp": pd.to_datetime([
            "2025-03-01T00:00:00Z", "2025-03-01T00:00:00Z",
            "2025-03-01T01:00:00Z", "2025-03-01T01:00:00Z",
        ]),
        "entry_ts": pd.to_datetime([
            "2025-03-01T00:00:00Z", "2025-03-01T00:00:00Z",
            "2025-03-01T01:00:00Z", "2025-03-01T01:00:00Z",
        ]),
    })
    paths = MODULE.ExactPaths(
        rows=rows, entry=np.ones(4, dtype=np.float32), atr=np.ones(4, dtype=np.float32),
        high=np.ones((4, 720), dtype=np.float32), low=np.ones((4, 720), dtype=np.float32),
        close=np.ones((4, 720), dtype=np.float32), manifest={}, audit={},
    )
    sampled = MODULE._stable_month_sample(paths, rows_per_month=1, seed=1729)
    assert len(sampled.rows) == 2  # one complete timestamp cohort, never one candidate
    assert sampled.rows["timestamp"].nunique() == 1
