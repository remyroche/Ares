from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_s52_geometry_path_quality_selection import (  # noqa: E402
    select_path_quality_rows,
)


def _row(
    arm: str,
    *,
    top_frac: float = 0.20,
    fit_evw: float = 0.70,
    holdout_evw: float = 0.60,
    fit_net: float = 0.001,
    fit_bad: float = 0.40,
    fit_p90: float = 4.0,
    fit_first_touch_p90: float = 1.5,
    fit_mae_before: float = 0.30,
    fit_rows: int = 1000,
    fit_side_rows: int = 200,
) -> dict[str, object]:
    return {
        "source": "sweep",
        "arm": arm,
        "selection_mode": "global",
        "top_frac": top_frac,
        "regime_family": "all",
        "fit_selected_rows": fit_rows,
        "fit_min_side_selected_rows": fit_side_rows,
        "fit_gross_ev_weighted_first_touch_precision": fit_evw,
        "fit_min_side_gross_ev_weighted_first_touch_precision": fit_evw - 0.05,
        "fit_mean_capture_net": fit_net,
        "fit_first_touch_bad_mae_1r_rate": 0.20,
        "fit_selected_path_bad_mae_1r_rate": fit_bad,
        "fit_selected_path_p90_mae_norm": fit_p90,
        "fit_first_touch_p90_mae_norm": fit_first_touch_p90,
        "fit_mae_1r_before_mfe_1r_rate": fit_mae_before,
        "fit_mfe_1r_before_mae_1r_rate": 1.0 - fit_mae_before,
        "fit_mean_max_adverse_before_mfe_1r": 1.20,
        "fit_mean_underwater_bars_before_mfe_1r": 8.0,
        "fit_mean_underwater_fraction_before_mfe_1r": 0.30,
        "fit_timeout_rate": 0.04,
        "holdout_selected_rows": fit_rows,
        "holdout_min_side_selected_rows": fit_side_rows,
        "holdout_gross_ev_weighted_first_touch_precision": holdout_evw,
        "holdout_min_side_gross_ev_weighted_first_touch_precision": holdout_evw - 0.05,
        "holdout_mean_capture_net": fit_net,
        "holdout_first_touch_bad_mae_1r_rate": 0.20,
        "holdout_selected_path_bad_mae_1r_rate": fit_bad,
        "holdout_selected_path_p90_mae_norm": fit_p90,
        "holdout_first_touch_p90_mae_norm": fit_first_touch_p90,
        "holdout_mae_1r_before_mfe_1r_rate": fit_mae_before,
        "holdout_mfe_1r_before_mae_1r_rate": 1.0 - fit_mae_before,
        "holdout_mean_max_adverse_before_mfe_1r": 1.20,
        "holdout_mean_underwater_bars_before_mfe_1r": 8.0,
        "holdout_mean_underwater_fraction_before_mfe_1r": 0.30,
        "holdout_timeout_rate": 0.04,
    }


def test_path_quality_selection_uses_fit_not_holdout_lure() -> None:
    candidates = pd.DataFrame(
        [
            _row("fit_best", fit_evw=0.72, holdout_evw=0.50),
            _row("holdout_lure", fit_evw=0.45, holdout_evw=0.95),
        ]
    )

    selected = select_path_quality_rows(
        candidates,
        top_fracs=[0.20],
        min_fit_rows=100,
        min_fit_side_rows=50,
    )

    assert selected.iloc[0]["arm"] == "fit_best"


def test_path_quality_selection_treats_post_exit_full_path_pain_as_warning() -> None:
    candidates = pd.DataFrame(
        [
            _row("dirty_high_evw", fit_evw=0.80, fit_bad=0.72, fit_p90=10.0),
            _row("cleaner_lower_evw", fit_evw=0.66, fit_bad=0.45, fit_p90=4.0),
        ]
    )

    selected = select_path_quality_rows(
        candidates,
        top_fracs=[0.20],
        min_fit_rows=100,
        min_fit_side_rows=50,
    )

    assert selected.iloc[0]["arm"] == "dirty_high_evw"
    assert selected.iloc[0]["selection_reason"] in {
        "fit_strict_path_bar_best",
        "fit_relative_path_bar_best",
    }


def test_path_quality_selection_penalizes_dirty_first_touch_order() -> None:
    candidates = pd.DataFrame(
        [
            _row("dirty_first_touch_high_evw", fit_evw=0.80, fit_mae_before=0.55),
            _row("cleaner_lower_evw", fit_evw=0.66, fit_mae_before=0.25),
        ]
    )

    selected = select_path_quality_rows(
        candidates,
        top_fracs=[0.20],
        min_fit_rows=100,
        min_fit_side_rows=50,
    )

    assert selected.iloc[0]["arm"] == "cleaner_lower_evw"


def test_path_quality_selection_falls_back_when_no_path_bar_passes() -> None:
    candidates = pd.DataFrame(
        [
            _row("least_bad", fit_evw=0.50, fit_bad=0.80, fit_p90=14.0, fit_mae_before=0.55),
            _row("worse", fit_evw=0.40, fit_bad=0.85, fit_p90=16.0, fit_mae_before=0.65),
        ]
    )

    selected = select_path_quality_rows(
        candidates,
        top_fracs=[0.20],
        min_fit_rows=100,
        min_fit_side_rows=50,
    )

    assert selected.iloc[0]["arm"] == "least_bad"
    assert selected.iloc[0]["selection_reason"] == "fallback_fit_path_quality_score"
