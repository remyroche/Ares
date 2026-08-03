from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

STAGING = Path(__file__).resolve().parents[1] / "scripts"
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from run_model_failure_risk_policy_sweep import (  # noqa: E402
    _condition_rows,
    _failure_metric_keys,
    attach_failure_risk,
)


def test_failure_risk_join_keeps_semantics_distinct_from_transition() -> None:
    candidates = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2025-01-01"], utc=True),
            "candidate_id": ["candidate"],
        }
    )
    predictions = pd.DataFrame(
        {
            "source_utc": pd.to_datetime(["2025-01-01"], utc=True),
            "prediction__broad__market_plus_health": [0.8],
            "target__economic_failure_broad_active": [1],
            "target__economic_failure_broad_event_id": ["failure"],
        }
    )
    joined = attach_failure_risk(
        candidates,
        predictions,
        failure_label="broad",
        feature_set="market_plus_health",
    )
    assert joined.loc[0, "failure_probability_oof"] == 0.8
    assert joined.loc[0, "expost_failure_active"] == 1
    assert joined.loc[0, "failure_event_id"] == "failure"


def test_condition_output_renames_internal_aliases() -> None:
    accepted = pd.DataFrame(
        {
            "expost_transition_active": [1],
            "active_transition_probability_oof": [0.8],
            "position_net_return": [0.1],
            "position_size": [100.0],
            "transition_event_id": ["failure"],
        }
    )
    rows = _condition_rows(
        accepted,
        score_stream="score_raw",
        arm="baseline_0p0000",
        policy="baseline",
        value=0.0,
    )
    conditions = {row["condition"] for row in rows}
    assert "true_economic_failure" in conditions
    assert "predicted_failure_ge_0p5" in conditions
    assert "true_active_transition" not in conditions


def test_failure_summary_metric_names_do_not_leak_transition_semantics() -> None:
    renamed = _failure_metric_keys(
        {
            "selected_active_rows": 3,
            "removed_active_mean_return_bps": -25.0,
            "ordinary_metric": 1,
        }
    )
    assert renamed["selected_failure_rows"] == 3
    assert renamed["removed_failure_mean_return_bps"] == -25.0
    assert renamed["ordinary_metric"] == 1
    assert not any("active_" in key for key in renamed)
