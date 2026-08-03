from __future__ import annotations

import pandas as pd

from scripts.summarize_canonical_conversion_transition_workstream import (
    _soft_label_equivalence,
)


def test_soft_label_equivalence_is_checked_only_on_complete_windows() -> None:
    keys = {
        "cohort_anchor_utc": pd.to_datetime(
            ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"]
        ),
        "side_name": ["long", "long"],
        "frozen_base_score_decile": [0, 0],
        "horizon_hours": [12, 12],
    }
    base = pd.DataFrame(
        {
            **keys,
            "before_global_hour_complete_flag": [True, False],
            "after_global_hour_complete_flag": [True, True],
            "delta_opportunity_probability_0bps": [0.2, 0.4],
        }
    )
    contribution = pd.DataFrame(
        {**keys, "delta_net_positive_rate": [0.2, 99.0]}
    )
    result = _soft_label_equivalence(base, contribution)
    assert result.loc[0, "complete_finite_rows"] == 1
    assert result.loc[0, "max_absolute_difference"] == 0.0
    assert bool(result.loc[0, "exact_equal"])
