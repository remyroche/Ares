from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.diagnose_july_alpha_economics_conversion import (
    _daily_metrics,
    _top_with_tie_bounds,
)


def _frame() -> pd.DataFrame:
    rows = []
    for day in ("2026-07-20", "2026-07-21"):
        for index in range(20):
            rows.append(
                {
                    "utc_day": day,
                    "candidate_id": f"{day}-{index:02d}",
                    "side_name": "long" if index % 2 == 0 else "short",
                    "base_oof_score": float(index),
                    "existing_alpha_ev": float(index) / 10.0,
                    # Large equal mapped level deliberately crosses top-10 cutoff.
                    "mapped_execution_ev": 1.0 if index == 0 else 0.0,
                    "execution_mfe_return_12h": float(index) / 100.0,
                    "execution_gross_ev_12h": float(index - 8) / 10_000.0,
                    "execution_cost_return": 0.0001,
                    "execution_net_ev_12h": float(index - 9) / 10_000.0,
                    "execution_exit_reason": "target" if index % 3 else "stop",
                }
            )
    return pd.DataFrame(rows)


def test_daily_metrics_report_base_and_residual_conversion_components() -> None:
    result = _daily_metrics(_frame())
    row = result.loc[
        result["utc_day"].eq("2026-07-20")
        & result["scope"].eq("pooled_global")
        & result["score"].eq("base_oof_score")
    ].iloc[0]
    assert row["rows"] == 20
    assert set(row["rank_ic"]) == {"mfe", "gross", "net", "cost"}
    assert 0.0 <= row["opportunity_incidence"] <= 1.0
    assert set(row["exit_mixture"]) == {"target", "stop"}


def test_tie_diagnostic_reports_deterministic_bounds_and_bootstrap() -> None:
    result = _top_with_tie_bounds(_frame(), fraction=0.10, bootstrap_draws=50)
    overall = result.loc[result["scope"].eq("all_july")].iloc[0]
    assert overall["top_k"] == 4
    assert overall["cutoff_tie_rows"] > overall["slots_from_cutoff_tie"]
    assert bool(overall["arbitrary_candidate_id_tie_break"])
    assert overall["best_tie_selected_net_bps"] >= overall["worst_tie_selected_net_bps"]
    assert overall["tie_bootstrap_net_bps_p05"] <= overall["tie_bootstrap_net_bps_p95"]
    assert np.isfinite(overall["tie_selection_sensitivity_bps"])
