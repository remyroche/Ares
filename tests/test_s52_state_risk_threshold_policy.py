from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_s52_state_risk_threshold_policy import select_policy_rows  # noqa: E402


def _row(selected_col: str, threshold: float, fit_evw: float, holdout_evw: float, fit_rows: int) -> dict:
    return {
        "overlay_action": "risk_filter",
        "selected_col": selected_col,
        "combine_mode": "mean_top3",
        "risk_threshold": threshold,
        "fit_all_sides_adjusted_ev_weighted_first_touch_precision": fit_evw,
        "fit_all_sides_baseline_ev_weighted_first_touch_precision": 0.50,
        "fit_all_sides_adjusted_mean_u": -0.001,
        "fit_long_adjusted_ev_weighted_first_touch_precision": fit_evw,
        "fit_long_adjusted_mae_before_mfe_1r_rate": 0.20,
        "fit_long_adjusted_mean_underwater_bars_before_mfe": 8.0,
        "fit_long_adjusted_selected_rows": fit_rows,
        "fit_long_baseline_selected_rows": 100,
        "holdout_all_sides_adjusted_ev_weighted_first_touch_precision": holdout_evw,
        "holdout_all_sides_baseline_ev_weighted_first_touch_precision": 0.50,
        "holdout_all_sides_adjusted_mean_u": -0.001,
        "holdout_long_adjusted_ev_weighted_first_touch_precision": holdout_evw,
        "holdout_long_adjusted_mae_before_mfe_1r_rate": 0.20,
        "holdout_long_adjusted_mean_underwater_bars_before_mfe": 8.0,
        "holdout_long_adjusted_selected_rows": fit_rows,
        "holdout_long_baseline_selected_rows": 100,
    }


def test_select_policy_rows_uses_fit_objective_not_holdout() -> None:
    summary = pd.DataFrame(
        [
            _row("selected_top10", 0.25, fit_evw=0.60, holdout_evw=0.99, fit_rows=80),
            _row("selected_top10", 0.50, fit_evw=0.80, holdout_evw=0.40, fit_rows=80),
        ]
    )

    selected = select_policy_rows(
        summary,
        retention_floors=[0.5],
        fit_mean_u_floor=-0.01,
        min_fit_long_evw=0.0,
        max_fit_long_mae_before_mfe=1.0,
        min_fit_long_rows=10,
        retention_penalty=0.08,
        mean_u_weight=2.0,
    )

    assert selected["risk_threshold"].tolist() == [0.50]


def test_select_policy_rows_respects_retention_floor_and_min_rows() -> None:
    summary = pd.DataFrame(
        [
            _row("selected_top20", 0.25, fit_evw=0.90, holdout_evw=0.70, fit_rows=20),
            _row("selected_top20", 0.75, fit_evw=0.70, holdout_evw=0.60, fit_rows=75),
        ]
    )

    selected = select_policy_rows(
        summary,
        retention_floors=[0.5],
        fit_mean_u_floor=-0.01,
        min_fit_long_evw=0.0,
        max_fit_long_mae_before_mfe=1.0,
        min_fit_long_rows=50,
        retention_penalty=0.08,
        mean_u_weight=2.0,
    )

    assert selected["risk_threshold"].tolist() == [0.75]


def test_select_policy_rows_applies_fit_quality_constraints() -> None:
    weak = _row("selected_top30", 0.25, fit_evw=0.40, holdout_evw=0.90, fit_rows=80)
    strong = _row("selected_top30", 0.75, fit_evw=0.70, holdout_evw=0.60, fit_rows=80)
    summary = pd.DataFrame([weak, strong])

    selected = select_policy_rows(
        summary,
        retention_floors=[0.5],
        fit_mean_u_floor=-0.01,
        min_fit_long_evw=0.60,
        max_fit_long_mae_before_mfe=0.25,
        min_fit_long_rows=50,
        retention_penalty=0.08,
        mean_u_weight=2.0,
    )

    assert selected["risk_threshold"].tolist() == [0.75]
