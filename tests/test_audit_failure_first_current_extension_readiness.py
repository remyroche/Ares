from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.audit_failure_first_current_extension_readiness import run


def test_readiness_audit_rejects_upstream_only_later_source(
    tmp_path: Path,
) -> None:
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    criteria = {
        "observed_calendar_days": {
            "observed": 74,
            "required": 180,
            "pass": False,
        },
        "failure_episodes": {
            "observed": 2,
            "required": 40,
            "pass": False,
        },
        "complete_window_episodes": {
            "observed": 1,
            "required": 40,
            "pass": False,
        },
        "failure_bins": {
            "observed": 3,
            "required": 40,
            "pass": False,
        },
    }
    (pipeline / "sufficiency_gate.json").write_text(
        json.dumps(
            {
                "status": "INSUFFICIENT_SUPPORT",
                "taxonomy_training_allowed": False,
                "criteria": criteria,
            }
        )
    )
    (pipeline / "manifest.json").write_text(
        json.dumps(
            {"score_valid_flag": "failure_first_score_is_strict_model_oos"}
        )
    )
    history_time = pd.date_range(
        "2026-07-18", periods=2, freq="h", tz="UTC"
    )
    history = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "execution_decision_utc": history_time,
        }
    )
    state = pd.DataFrame({"candidate_id": ["a", "b", "later"]})
    upstream = pd.DataFrame(
        {
            "candidate_id": ["later"],
            "__ts__": [history_time.max()],
            "base_score": [0.5],
        }
    )
    history_path = tmp_path / "history.parquet"
    state_path = tmp_path / "state.parquet"
    upstream_path = tmp_path / "upstream.parquet"
    output = tmp_path / "output"
    history.to_parquet(history_path, index=False)
    state.to_parquet(state_path, index=False)
    upstream.to_parquet(upstream_path, index=False)
    result = run(
        argparse.Namespace(
            current_pipeline=pipeline,
            current_history=history_path,
            state_source=state_path,
            candidate=[upstream_path],
            output_dir=output,
        )
    )
    assert result["status"] == "WAITING_FOR_NEW_SAME_MODEL_HISTORY"
    assert result["ready_extension_sources"] == 0
    assert result["minimum_remaining_deficits"][
        "additional_observed_days"
    ] == 106
