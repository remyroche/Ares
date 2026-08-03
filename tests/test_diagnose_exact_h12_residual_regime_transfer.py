from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.diagnose_exact_h12_residual_regime_transfer import (
    _build_hour_side_panel,
    _trajectory_neutral_fill,
)


def test_trajectory_neutral_fill_preserves_availability() -> None:
    frame = pd.DataFrame(
        {
            "trajectory_available": [True, False, True],
            "trajectory_transition_probability": [0.8, np.nan, np.nan],
            "probability_entropy": [0.2, np.nan, 0.3],
            "top2_margin": [0.6, np.nan, 0.4],
        }
    )
    result = _trajectory_neutral_fill(frame)
    assert result["trajectory_available_numeric"].tolist() == [1.0, 0.0, 0.0]
    assert result["trajectory_transition_probability"].tolist() == [0.8, 0.5, 0.5]
    assert result.loc[1, "trajectory_probability_entropy"] == math.log(2.0)
    assert result.loc[2, "trajectory_top2_margin"] == 0.0


def test_hour_side_contributions_reconcile_monthly_global_book_delta() -> None:
    timestamps = pd.to_datetime(
        ["2026-05-01T00:00:00Z", "2026-05-01T01:00:00Z"]
    )
    candidate = pd.DataFrame(
        {
            "__ts__": np.repeat(timestamps, 2),
            **{
                field: np.repeat(float(index + 1), 4)
                for index, field in enumerate(
                    (
                        "bocpd__change_probability_mean",
                        "bocpd__change_probability_max",
                        "bocpd__run_length_mean",
                        "bocpd__run_length_q05",
                        "bocpd__run_length_entropy",
                        "bocpd__signal_count",
                        "bocpd__state_age_hours",
                        "bocpd__is_persistent_24h",
                        "bocpd__is_persistent_72h",
                        "lgbm_transition_probability",
                        "lgbm_entropy",
                        "lgbm_margin",
                        "bocpd_stable_vs_transition_probability",
                        "bocpd_onset_h1_probability",
                        "bocpd_onset_h3_probability",
                        "bocpd_onset_h6_probability",
                        "bocpd_onset_h12_probability",
                        "trajectory_transition_probability",
                        "trajectory_probability_entropy",
                        "trajectory_top2_margin",
                        "trajectory_available_numeric",
                    )
                )
            },
        }
    )
    rows = []
    for score, values in {
        "base_ev_exact_h12": [0.01, -0.01],
        "residual_exact_h12": [0.02, 0.00],
        "direct_q25_exact_h12": [-0.01, 0.01],
    }.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "candidate_id": f"{score}-{index}",
                    "candidate_month": "2026-05",
                    "score_name": score,
                    "__ts__": timestamps[index],
                    "side_name": "long",
                    "execution_net_ev_12h": value,
                }
            )
    books = pd.DataFrame(rows)
    panel = _build_hour_side_panel(candidate, books)
    long_rows = panel.loc[panel["side_name"].eq("long")]
    assert np.isclose(long_rows["residual_minus_base_bps"].sum(), 100.0)
    assert np.isclose(long_rows["direct_minus_residual_bps"].sum(), -100.0)
    assert panel.loc[panel["side_name"].eq("short"), "residual_minus_base_bps"].eq(0).all()
