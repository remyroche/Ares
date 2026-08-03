from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.materialize_canonical_economic_conversion_contribution_labels import (
    _window_contributions,
)


def test_raw_contributions_reconcile_direct_mean_and_soft_rate() -> None:
    anchor = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = pd.DataFrame(
        {
            "execution_net_ev_12h": [0.03, -0.02, 0.0, 0.01],
            "execution_label_end_utc": [
                anchor + pd.Timedelta(hours=value) for value in (12, 13, 14, 15)
            ],
        }
    )
    result = _window_contributions(rows)
    assert result["candidate_support"] == 4
    assert result["net_positive_support"] == 2
    assert result["net_positive_rate"] == 0.5
    assert result["positive_net_contribution"] == 0.01
    assert result["loss_net_contribution"] == 0.005
    assert np.isclose(result["direct_mean_net"], 0.005)
    assert result["raw_components_reconcile_direct_mean_flag"]
    assert result["target_available_utc"] == anchor + pd.Timedelta(hours=16)


def test_empty_window_stays_missing_instead_of_becoming_zero() -> None:
    result = _window_contributions(
        pd.DataFrame(columns=["execution_net_ev_12h", "execution_label_end_utc"])
    )
    assert result["candidate_support"] == 0
    assert np.isnan(result["positive_net_contribution"])
    assert np.isnan(result["loss_net_contribution"])
    assert np.isnan(result["net_positive_rate"])
    assert not result["raw_components_reconcile_direct_mean_flag"]
