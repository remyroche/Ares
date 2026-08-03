from __future__ import annotations

import pandas as pd
import pytest

from scripts.materialize_historical_exact_h12_alignment_sidecar import (
    COST_MODEL_ID,
    EXECUTION_POLICY_ID,
    TARGET_ID,
    validate_alignment,
)


def _sidecar() -> pd.DataFrame:
    decision = pd.Timestamp("2024-04-01T01:00:00Z")
    return pd.DataFrame({
        "candidate_id": ["c1"], "side": ["long"],
        "feature_cutoff_ts": [decision - pd.Timedelta(hours=1)],
        "decision_ts": [decision], "entry_ts": [decision],
        "label_end_ts": [decision + pd.Timedelta(hours=12)],
        "label_available_ts": [decision + pd.Timedelta(hours=12)],
        "target_id": [TARGET_ID], "execution_policy_id": [EXECUTION_POLICY_ID],
        "replay_execution_policy_id": [EXECUTION_POLICY_ID],
        "cost_model_id": [COST_MODEL_ID], "feature_set_id": ["raw_380_test"],
        "execution_geometry_id": ["geometry"], "source_row_number": [1],
        "source_shard_sha256": ["hash"], "exact_h12_gross_bps": [150.0],
        "row_cost_bps": [100.0], "exact_h12_net_bps": [50.0],
    })


def test_alignment_accepts_exact_h12_cost_and_policy_contract() -> None:
    validate_alignment(_sidecar(), feature_set_id="raw_380_test")


def test_alignment_rejects_double_cost_or_late_feature() -> None:
    bad = _sidecar()
    bad.loc[0, "exact_h12_net_bps"] = 40.0
    with pytest.raises(ValueError, match="gross minus row cost"):
        validate_alignment(bad, feature_set_id="raw_380_test")
    bad = _sidecar()
    bad.loc[0, "feature_cutoff_ts"] = bad.loc[0, "decision_ts"] + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="feature cutoff"):
        validate_alignment(bad, feature_set_id="raw_380_test")


def test_alignment_rejects_mismatched_replay_policy() -> None:
    bad = _sidecar()
    bad.loc[0, "replay_execution_policy_id"] = "other"
    with pytest.raises(ValueError, match="policy"):
        validate_alignment(bad, feature_set_id="raw_380_test")
