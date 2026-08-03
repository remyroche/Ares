from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.diagnose_within_july_opportunity_capture import (
    compare_selections,
    economic_components,
    selection_metrics,
)


def _frame() -> pd.DataFrame:
    rows = []
    for index in range(20):
        mfe = 0.03 + index * 0.001
        gross = mfe - (0.004 if index >= 18 else 0.02)
        cost = 0.01
        rows.append(
            {
                "__ts__": pd.Timestamp("2026-07-08", tz="UTC")
                + pd.Timedelta(hours=index),
                "__symbol__": f"S{index}",
                "side_name": "long" if index % 2 else "short",
                "candidate_id": str(index),
                "prediction": float(index),
                "existing_alpha_ev": float(19 - index),
                "execution_gross_ev_12h": gross,
                "execution_cost_return": cost,
                "execution_net_ev_12h": gross - cost,
                "execution_mfe_return_12h": mfe,
                "execution_mae_return_12h": 0.01,
                "execution_exit_hour": 4.0,
                "favorable_first": int(index >= 10),
                "adverse_first": int(index < 10),
                "timeout": 0,
            }
        )
    return pd.DataFrame(rows)


def test_components_reconcile_net_from_opportunity_regret_and_cost() -> None:
    components = economic_components(_frame())
    reconstructed = (
        components["mean_path_mfe_bps"]
        - components["mean_mfe_to_gross_gap_bps"]
        - components["mean_cost_bps"]
    )
    assert np.isclose(reconstructed, components["mean_net_bps"])


def test_selection_comparison_uses_one_pooled_top_decile() -> None:
    frame = _frame()
    model_metrics, model_selected = selection_metrics(
        frame,
        score_name="within_july_model",
        score_column="prediction",
        evaluation="aggregate",
        scope="pooled_global",
    )
    _, alpha_selected = selection_metrics(
        frame,
        score_name="frozen_alpha",
        score_column="existing_alpha_ev",
        evaluation="aggregate",
        scope="pooled_global",
    )
    comparison, replacements = compare_selections(
        model_selected,
        alpha_selected,
        evaluation="aggregate",
        scope="pooled_global",
    )
    assert model_metrics["rows"] == 2
    assert comparison["added_rows"] == 2
    assert comparison["dropped_rows"] == 2
    assert abs(comparison["reconciliation_error_bps"]) < 1e-10
    assert {row["replacement_role"] for row in replacements} == {
        "added",
        "dropped",
    }
