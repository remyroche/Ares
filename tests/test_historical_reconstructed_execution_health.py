from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGING = Path(__file__).resolve().parents[1] / "scripts"
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from materialize_historical_reconstructed_execution_health import (  # noqa: E402
    COMMON_HEALTH_COLUMNS,
    NONCOMPARABLE_HEALTH_FIELDS,
    _catboost_entropy,
)


def test_six_class_entropy_is_exact_and_identity_preserving() -> None:
    probability = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    frame = pd.DataFrame(
        {
            "candidate_id": ["candidate"],
            "side_name": ["long"],
            "__symbol__": ["BTC"],
            "__ts__": pd.to_datetime(["2025-03-01"], utc=True),
            "prob_immediate_adverse_path": [probability[0]],
            "prob_fast_realization_winner": [probability[1]],
            "prob_late_breakout": [probability[2]],
            "prob_slow_grinder": [probability[3]],
            "prob_mfe_reversal_or_timeout": [probability[4]],
            "prob_dead_timeout": [probability[5]],
        }
    )
    result = _catboost_entropy(frame)
    assert result.loc[0, "candidate_id"] == "candidate"
    assert np.isclose(result.loc[0, "catboost_entropy"], np.log(2.0))


def test_common_catalog_excludes_noncomparable_current_only_fields() -> None:
    assert "health__alpha_uncertainty_mean" in NONCOMPARABLE_HEALTH_FIELDS
    assert "health__catboost_entropy_mean" in NONCOMPARABLE_HEALTH_FIELDS
    assert not set(COMMON_HEALTH_COLUMNS).intersection(
        NONCOMPARABLE_HEALTH_FIELDS
    )
    assert len(COMMON_HEALTH_COLUMNS) == 27
