from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_r3_global_tail_lambdarank_ablation import _load_frame


def test_runner_reports_missing_p90_spread_column_at_preflight(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    part = source / "part.parquet"
    pd.DataFrame({
        "candidate_id": ["candidate"],
        "__ts__": [pd.Timestamp("2025-01-01T00:00:00Z")],
        "side_name": ["long"],
        "label_available_ts": [pd.Timestamp("2025-01-01T13:00:00Z")],
        "label_valid": [True],
        "exact_net_bps": [1.0],
        "robust_clear_event_b25": [1.0],
        "lower_touch_minute": [-1.0],
        "tail_target_t1_valid": [True],
        "tail_target_net_grade_0_5": [1],
        "tail_target_t2_valid": [True],
        "tail_target_atr_grade_0_5": [1],
        "tail_target_t3_valid": [True],
        "tail_target_t3_first_touch_grade_0_4": [1],
    }).to_parquet(part, index=False)
    (source / "manifest.json").write_text(json.dumps({"status": "complete", "parts": [part.name]}))

    with pytest.raises(ValueError, match="p90_spread_bps"):
        _load_frame(source)


def test_t1_t2_preflight_does_not_require_t3_path_target(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    part = source / "part.parquet"
    pd.DataFrame({
        "candidate_id": ["candidate"],
        "__ts__": [pd.Timestamp("2025-01-01T00:00:00Z")],
        "side_name": ["long"],
        "label_available_ts": [pd.Timestamp("2025-01-01T13:00:00Z")],
        "label_valid": [True],
        "p90_spread_bps": [10.0],
        "exact_net_bps": [1.0],
        "robust_clear_event_b25": [1.0],
        "lower_touch_minute": [-1.0],
        "tail_target_t1_valid": [True],
        "tail_target_net_grade_0_5": [1],
        "tail_target_t2_valid": [True],
        "tail_target_atr_grade_0_5": [1],
    }).to_parquet(part, index=False)
    (source / "manifest.json").write_text(json.dumps({"status": "complete", "parts": [part.name]}))

    frame, _ = _load_frame(source, selected_targets=("t1_exact_net", "t2_atr_net"))
    assert len(frame) == 1
