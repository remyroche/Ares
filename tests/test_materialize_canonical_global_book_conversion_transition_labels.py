from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.materialize_canonical_global_book_conversion_transition_labels import (
    GLOBAL_EV_BANDS,
    _normalise_mapping,
    _stable_select,
    _window_book_metrics,
    add_causal_global_mapped_ev_coordinates,
)


def _population() -> pd.DataFrame:
    anchor = pd.Timestamp("2025-03-01T00:00:00Z")
    return pd.DataFrame(
        {
            "candidate_id": ["b", "a", "c", "d"],
            "__symbol__": ["B", "A", "C", "D"],
            "side_name": ["long", "short", "long", "short"],
            "mapped_direct_net": [0.02, 0.02, 0.01, -0.01],
            "map_reference_rows": [1000] * 4,
            "map_side_reference_rows": [500] * 4,
            "map_cell_reference_rows": [100] * 4,
            "execution_gross_ev_12h": [0.04, 0.03, 0.02, 0.00],
            "execution_cost_return": [0.01] * 4,
            "execution_net_ev_12h": [0.03, 0.02, 0.01, -0.01],
            "opportunity_gross_above_cost_0bps": [True, True, True, False],
            "opportunity_gross_above_cost_25bps": [True, True, True, False],
            "execution_exit_class": [
                "trailing",
                "timeout",
                "full_stop",
                "adverse_exit",
            ],
            "execution_label_end_utc": [
                anchor + pd.Timedelta(hours=value) for value in (12, 13, 14, 15)
            ],
        }
    )


def test_global_selection_uses_score_then_candidate_id_without_side_quota() -> None:
    selected = _stable_select(
        _population(), score_column="mapped_direct_net", fraction=0.5
    )
    assert selected["candidate_id"].tolist() == ["a", "b"]
    assert set(selected["side_name"]) == {"long", "short"}


def test_window_book_metrics_reconcile_and_record_cutoff_plateau() -> None:
    end = pd.Timestamp("2025-03-01T12:00:00Z")
    result = _window_book_metrics(_population(), fraction=0.5, window_end=end)
    assert result["selected_candidate_support"] == 2
    assert result["mapped_score_cutoff"] == 0.02
    assert result["cutoff_plateau_population_rows"] == 2
    assert result["cutoff_plateau_selected_rows"] == 2
    assert np.isclose(
        result["positive_net_contribution"]
        - result["loss_net_contribution"],
        result["direct_mean_net"],
    )
    assert result["raw_components_reconcile_direct_mean_flag"]
    assert result["mapped_plus_residual_reconciles_direct_mean_flag"]
    assert np.isclose(
        result["mapped_score_mean"] + result["mean_conversion_residual"],
        result["direct_mean_net"],
    )
    assert result["target_available_utc"] == pd.Timestamp(
        "2025-03-01T14:00:00Z"
    )


def test_causal_global_coordinates_use_prior_days_and_keep_ties_together() -> None:
    first = _population().copy()
    first["execution_decision_utc"] = pd.Timestamp("2025-03-01T12:00:00Z")
    second = _population().copy()
    second["candidate_id"] = ["f", "e", "g", "h"]
    second["execution_decision_utc"] = pd.Timestamp("2025-03-02T12:00:00Z")
    coordinates, audit = add_causal_global_mapped_ev_coordinates(
        pd.concat([first, second], ignore_index=True),
        window_days=21,
        minimum_reference_rows=4,
    )
    unavailable = coordinates["execution_decision_utc"].dt.day.eq(1)
    assert coordinates.loc[
        unavailable, "causal_global_mapped_ev_band"
    ].eq("UNAVAILABLE").all()
    available = coordinates.loc[~unavailable]
    assert available["causal_global_mapped_ev_band"].isin(GLOBAL_EV_BANDS).all()
    tied = available.loc[available["mapped_direct_net"].eq(0.02)]
    assert tied["causal_global_mapped_ev_percentile"].nunique() == 1
    assert tied["causal_global_mapped_ev_band"].nunique() == 1
    assert audit.loc[audit["coordinate_available"], "reference_rows"].iloc[0] == 4


def test_float32_accounting_roundoff_is_accepted() -> None:
    frame = _population().copy()
    frame["execution_decision_utc"] = pd.Timestamp("2025-03-01T12:00:00Z")
    frame["candidate_month"] = "2025-03"
    frame["mapped_eligible"] = True
    frame["opportunity_gross_above_cost_0bps"] = frame[
        "opportunity_gross_above_cost_0bps"
    ].astype(float)
    frame["opportunity_gross_above_cost_25bps"] = frame[
        "opportunity_gross_above_cost_25bps"
    ].astype(float)
    for column in (
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
    ):
        frame[column] = frame[column].astype("float32")
    # Emulate a one-ULP independent arithmetic path.
    frame.loc[0, "execution_net_ev_12h"] = np.nextafter(
        frame.loc[0, "execution_net_ev_12h"], np.float32(np.inf)
    )
    eligible, audit = _normalise_mapping(frame)
    assert len(eligible) == len(frame)
    assert audit["warmup_unmapped_rows"] == 0
