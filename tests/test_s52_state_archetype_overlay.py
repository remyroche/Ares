from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_s52_state_archetype_overlay import (  # noqa: E402
    build_archetype_candidates,
    select_archetype_overlay_rows,
)


def _candidate(
    feature: str,
    *,
    selected_col: str = "selected_top10",
    side: str = "long",
    fit_evw: float,
    holdout_evw: float,
    fit_net: float = -0.001,
    fit_rows: int = 100,
    fit_bad_mae: float = 0.20,
    fit_mae_before: float = 0.20,
    fit_adverse: float = 1.0,
    fit_underwater: float = 8.0,
) -> dict[str, object]:
    return {
        "selected_col": selected_col,
        "state_feature": feature,
        "bucket": "state",
        "side": side,
        "fit_selected_rows": fit_rows,
        "fit_ev_weighted_first_touch_precision": fit_evw,
        "holdout_ev_weighted_first_touch_precision": holdout_evw,
        "fit_mean_first_touch_net": fit_net,
        "holdout_mean_first_touch_net": -0.001,
        "fit_baseline_mean_first_touch_net": -0.002,
        "fit_first_touch_bad_mae_1r_rate": fit_bad_mae,
        "fit_mae_before_mfe_1r_rate": fit_mae_before,
        "fit_mean_max_adverse_before_mfe_1r": fit_adverse,
        "fit_mean_underwater_bars_before_mfe_1r": fit_underwater,
        "fit_timeout_rate": 0.02,
        "holdout_first_touch_bad_mae_1r_rate": fit_bad_mae,
        "holdout_mae_before_mfe_1r_rate": fit_mae_before,
        "holdout_mean_max_adverse_before_mfe_1r": fit_adverse,
        "holdout_mean_underwater_bars_before_mfe_1r": fit_underwater,
        "holdout_timeout_rate": 0.02,
    }


def test_archetype_overlay_selector_uses_fit_not_holdout() -> None:
    candidates = pd.DataFrame(
        [
            _candidate("gmm_cluster_id", fit_evw=0.78, holdout_evw=0.40),
            _candidate("holdout_lure", fit_evw=0.50, holdout_evw=0.99),
        ]
    )

    selected = select_archetype_overlay_rows(
        candidates,
        selected_cols=["selected_top10"],
        sides=["long"],
        top_n_per_group=1,
        min_fit_selected_rows=50,
        min_fit_evw=0.40,
        min_fit_net=-0.01,
        max_fit_first_touch_bad_mae=0.40,
        max_fit_mae_before=0.40,
        max_fit_adverse=2.0,
        max_fit_underwater=12.0,
        max_fit_timeout=0.12,
    )

    assert selected.iloc[0]["state_feature"] == "gmm_cluster_id"


def test_archetype_overlay_selector_respects_fit_constraints() -> None:
    candidates = pd.DataFrame(
        [
            _candidate("dirty_high_evw", fit_evw=0.90, holdout_evw=0.90, fit_bad_mae=0.60),
            _candidate("clean_lower_evw", fit_evw=0.70, holdout_evw=0.70, fit_bad_mae=0.18),
        ]
    )

    selected = select_archetype_overlay_rows(
        candidates,
        selected_cols=["selected_top10"],
        sides=["long"],
        top_n_per_group=1,
        min_fit_selected_rows=50,
        min_fit_evw=0.40,
        min_fit_net=-0.01,
        max_fit_first_touch_bad_mae=0.40,
        max_fit_mae_before=0.40,
        max_fit_adverse=2.0,
        max_fit_underwater=12.0,
        max_fit_timeout=0.12,
    )

    assert selected.iloc[0]["state_feature"] == "clean_lower_evw"
    assert selected.iloc[0]["selection_reason"] == "eligible_fit_best"


def test_archetype_overlay_selector_returns_fallback_when_no_eligible() -> None:
    candidates = pd.DataFrame(
        [
            _candidate("least_bad", fit_evw=0.30, holdout_evw=0.90),
            _candidate("worse_fit", fit_evw=0.20, holdout_evw=0.99),
        ]
    )

    selected = select_archetype_overlay_rows(
        candidates,
        selected_cols=["selected_top10"],
        sides=["long"],
        top_n_per_group=1,
        min_fit_selected_rows=50,
        min_fit_evw=0.80,
        min_fit_net=0.0,
        max_fit_first_touch_bad_mae=0.10,
        max_fit_mae_before=0.10,
        max_fit_adverse=0.5,
        max_fit_underwater=4.0,
        max_fit_timeout=0.01,
    )

    assert selected.iloc[0]["state_feature"] == "least_bad"
    assert selected.iloc[0]["selection_reason"] == "fallback_no_eligible"


def test_build_archetype_candidates_fits_buckets_on_fit_and_reports_holdout() -> None:
    fit_months = ["2026-04"] * 10 + ["2026-05"] * 10
    fit_values = [float(i) for i in range(20)]
    holdout_values = [999.0, -100.0, 2.5, 0.5]
    ledger = pd.DataFrame(
        {
            "variant": ["v"] * 24,
            "month": fit_months + ["2026-06"] * 4,
            "side_name": ["long"] * 24,
            "selected_top10": [True] * 24,
            "score": [1.0 - i * 0.01 for i in range(24)],
            "first_touch_net": [0.01 if i % 3 else -0.002 for i in range(20)] + [0.02, -0.01, 0.015, -0.002],
            "first_pass_good": [0 if i % 3 == 0 else 1 for i in range(20)] + [1, 0, 1, 0],
            "first_pass_bad": [1 if i % 3 == 0 else 0 for i in range(20)] + [0, 1, 0, 1],
            "first_touch_mae_norm": [1.2 if i % 3 == 0 else 0.4 for i in range(20)] + [0.3, 1.4, 0.2, 1.2],
            "mfe_1r_before_mae_1r": [0 if i % 3 == 0 else 1 for i in range(20)] + [1, 0, 1, 0],
            "mae_1r_before_mfe_1r": [1 if i % 3 == 0 else 0 for i in range(20)] + [0, 1, 0, 1],
            "max_adverse_before_mfe_1r": [2.0 if i % 3 == 0 else 0.4 for i in range(20)] + [0.2, 2.2, 0.4, 2.0],
            "underwater_bars_before_mfe_1r": [20 if i % 3 == 0 else 3 for i in range(20)] + [1, 22, 3, 18],
            "underwater_fraction_before_mfe_1r": [0.8 if i % 3 == 0 else 0.2 for i in range(20)] + [0.1, 0.8, 0.2, 0.7],
            "is_timeout": [0] * 24,
            "gmm_mahal_1": fit_values + holdout_values,
        }
    )

    candidates = build_archetype_candidates(
        ledger,
        selected_cols=["selected_top10"],
        fit_months=["2026-04", "2026-05"],
        holdout_month="2026-06",
        round_trip_cost=0.01,
        q=2,
    )

    assert not candidates.empty
    assert set(candidates["state_feature"]) == {"gmm_mahal_1"}
    assert candidates["fit_selected_rows"].sum() == 20
    assert candidates["holdout_selected_rows"].sum() == 4
