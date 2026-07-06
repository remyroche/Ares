from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import dataclass

import pandas as pd

from scripts.run_s52_bidirectional_geometry_sweep import (
    _precision_geometry_score,
    _select_proxy_shortlist,
    _summarize_slice,
)


@dataclass(frozen=True)
class _Arm:
    name: str


def test_precision_geometry_score_uses_ev_weighted_precision_not_diagnostics():
    clean_row = {
        "fit_gross_ev_weighted_first_touch_precision": 0.55,
        "holdout_gross_ev_weighted_first_touch_precision": 0.60,
        "min_side_holdout_ev_weighted_precision": 0.50,
        "holdout_precision": 0.90,
        "holdout_stop_rate": 0.05,
        "holdout_timeout_rate": 0.02,
        "holdout_mean_capture_net": 0.10,
    }
    dirty_same_precision = {
        **clean_row,
        "holdout_precision": 0.20,
        "holdout_stop_rate": 0.95,
        "holdout_timeout_rate": 0.80,
        "holdout_mean_capture_net": -0.50,
    }

    assert _precision_geometry_score(clean_row, include_side_floor=True) == _precision_geometry_score(
        dirty_same_precision,
        include_side_floor=True,
    )


def test_precision_geometry_score_prefers_ev_weighted_clean_first_touch_precision():
    lower_ev_higher_raw = {
        "fit_gross_ev_weighted_first_touch_precision": 0.45,
        "holdout_gross_ev_weighted_first_touch_precision": 0.50,
        "min_side_holdout_ev_weighted_precision": 0.45,
        "holdout_precision": 0.95,
    }
    higher_ev_lower_raw = {
        "fit_gross_ev_weighted_first_touch_precision": 0.55,
        "holdout_gross_ev_weighted_first_touch_precision": 0.60,
        "min_side_holdout_ev_weighted_precision": 0.55,
        "holdout_precision": 0.50,
    }

    assert _precision_geometry_score(higher_ev_lower_raw, include_side_floor=True) > _precision_geometry_score(
        lower_ev_higher_raw,
        include_side_floor=True,
    )


def test_proxy_shortlist_uses_fit_metrics_not_holdout_lure():
    arms = [_Arm("fit_good"), _Arm("holdout_lure")]
    coarse = pd.DataFrame(
        [
            {
                "arm": "fit_good",
                "fit_gross_ev_weighted_first_touch_precision": 0.70,
                "fit_mean_capture_net": 0.002,
                "fit_first_touch_bad_mae_1r_rate": 0.12,
                "fit_mae_1r_before_mfe_1r_rate": 0.20,
                "fit_mfe_1r_before_mae_1r_rate": 0.80,
                "fit_mean_underwater_bars_before_mfe_1r": 8.0,
                "fit_mean_underwater_fraction_before_mfe_1r": 0.30,
                "fit_timeout_rate": 0.02,
                "holdout_gross_ev_weighted_first_touch_precision": 0.30,
                "path_quality_geometry_score": 0.10,
                "precision_geometry_score": 0.10,
            },
            {
                "arm": "holdout_lure",
                "fit_gross_ev_weighted_first_touch_precision": 0.30,
                "fit_mean_capture_net": -0.004,
                "fit_first_touch_bad_mae_1r_rate": 0.35,
                "fit_mae_1r_before_mfe_1r_rate": 0.45,
                "fit_mfe_1r_before_mae_1r_rate": 0.55,
                "fit_mean_underwater_bars_before_mfe_1r": 18.0,
                "fit_mean_underwater_fraction_before_mfe_1r": 0.60,
                "fit_timeout_rate": 0.02,
                "holdout_gross_ev_weighted_first_touch_precision": 0.95,
                "path_quality_geometry_score": 5.00,
                "precision_geometry_score": 5.00,
            },
        ]
    )

    selected = _select_proxy_shortlist(arms, coarse, shortlist_size=1)

    assert [arm.name for arm in selected] == ["fit_good"]


def test_summarize_slice_reports_cost_coverage_ratio():
    frame = pd.DataFrame(
        {
            "period": ["2026-06", "2026-06"],
            "selected_rows": [10, 30],
            "capture_gross_mean": [0.006, 0.010],
            "capture_net_mean": [-0.004, 0.000],
            "ev_weighted_first_touch_precision": [0.60, 0.70],
            "gross_hit_value_mean": [0.006, 0.010],
            "gross_stop_value_mean": [0.004, 0.002],
            "gross_timeout_value_mean": [0.0, 0.0],
        }
    )

    summary = _summarize_slice("holdout", frame)

    assert round(summary["holdout_mean_capture_gross"], 6) == 0.009
    assert round(summary["holdout_implied_round_trip_cost"], 6) == 0.01
    assert round(summary["holdout_cost_coverage_ratio"], 6) == 0.9
