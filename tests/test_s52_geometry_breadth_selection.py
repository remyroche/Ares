from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_s52_geometry_breadth_selection import select_breadth_rows  # noqa: E402


def _row(
    arm: str,
    *,
    top_frac: float = 0.2,
    fit_evw: float,
    holdout_evw: float,
    fit_net: float = 0.0,
    fit_bad_mae: float = 0.20,
    fit_mae_before: float = 0.20,
    fit_underwater: float = 8.0,
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
        "holdout_gross_ev_weighted_first_touch_precision": holdout_evw,
        "fit_mean_capture_net": fit_net,
        "fit_first_touch_bad_mae_1r_rate": fit_bad_mae,
        "fit_mae_1r_before_mfe_1r_rate": fit_mae_before,
        "fit_mfe_1r_before_mae_1r_rate": 1.0 - fit_mae_before,
        "fit_mean_underwater_bars_before_mfe_1r": fit_underwater,
        "fit_mean_underwater_fraction_before_mfe_1r": 0.30,
        "fit_timeout_rate": 0.02,
    }


def test_breadth_selector_uses_fit_not_holdout() -> None:
    candidates = pd.DataFrame(
        [
            _row("fit_best", fit_evw=0.62, holdout_evw=0.50),
            _row("holdout_lure", fit_evw=0.42, holdout_evw=0.90),
        ]
    )

    selected = select_breadth_rows(
        candidates,
        top_fracs=[0.20],
        min_fit_selected_rows=100,
        min_fit_side_rows=50,
        min_fit_evw=0.35,
        max_fit_bad_mae=0.35,
        max_fit_mae_before=0.40,
        max_fit_underwater=16.0,
    )

    assert selected.iloc[0]["arm"] == "fit_best"


def test_breadth_selector_respects_fit_path_constraints() -> None:
    candidates = pd.DataFrame(
        [
            _row("dirty_high_evw", fit_evw=0.80, holdout_evw=0.80, fit_bad_mae=0.60),
            _row("clean_lower_evw", fit_evw=0.62, holdout_evw=0.60, fit_bad_mae=0.20),
        ]
    )

    selected = select_breadth_rows(
        candidates,
        top_fracs=[0.20],
        min_fit_selected_rows=100,
        min_fit_side_rows=50,
        min_fit_evw=0.35,
        max_fit_bad_mae=0.35,
        max_fit_mae_before=0.40,
        max_fit_underwater=16.0,
    )

    assert selected.iloc[0]["arm"] == "clean_lower_evw"
    assert selected.iloc[0]["selection_reason"] == "eligible_fit_best"


def test_breadth_selector_returns_fallback_when_no_eligible() -> None:
    candidates = pd.DataFrame(
        [
            _row("least_bad", fit_evw=0.34, holdout_evw=0.90),
            _row("worse_fit", fit_evw=0.20, holdout_evw=0.95),
        ]
    )

    selected = select_breadth_rows(
        candidates,
        top_fracs=[0.20],
        min_fit_selected_rows=100,
        min_fit_side_rows=50,
        min_fit_evw=0.60,
        max_fit_bad_mae=0.10,
        max_fit_mae_before=0.10,
        max_fit_underwater=4.0,
    )

    assert selected.iloc[0]["arm"] == "least_bad"
    assert selected.iloc[0]["selection_reason"] == "fallback_no_eligible"


def test_breadth_selector_can_require_positive_fit_net_after_cost() -> None:
    candidates = pd.DataFrame(
        [
            _row("negative_net_high_evw", fit_evw=0.80, holdout_evw=0.80, fit_net=-0.003),
            _row("positive_net_lower_evw", fit_evw=0.62, holdout_evw=0.60, fit_net=0.001),
        ]
    )

    selected = select_breadth_rows(
        candidates,
        top_fracs=[0.20],
        min_fit_selected_rows=100,
        min_fit_side_rows=50,
        min_fit_evw=0.35,
        min_fit_net=0.0,
        max_fit_bad_mae=0.35,
        max_fit_mae_before=0.40,
        max_fit_underwater=16.0,
    )

    assert selected.iloc[0]["arm"] == "positive_net_lower_evw"
    assert selected.iloc[0]["selection_reason"] == "eligible_fit_best"
