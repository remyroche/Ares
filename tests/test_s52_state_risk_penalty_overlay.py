from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s52_state_risk_penalty_overlay import (  # noqa: E402
    _apply_bucket_spec,
    _bucket_risk,
    _fit_bucket_spec,
    _filter_selected_by_risk,
    _select_budgeted,
)


def test_fit_bucket_spec_uses_fit_distribution_and_applies_holdout_outliers() -> None:
    fit = pd.Series(range(100), name="gmm_mahal_1")
    spec = _fit_bucket_spec(fit, q=5)
    buckets = _apply_bucket_spec(pd.Series([-100, 0, 50, 99, 999], name="gmm_mahal_1"), spec)

    assert spec["kind"] == "continuous"
    assert len(set(buckets.astype(str))) >= 3
    assert str(buckets.iloc[0]).startswith("(-inf")
    assert "inf" in str(buckets.iloc[-1])


def test_bucket_risk_increases_for_worse_path_and_lower_ev() -> None:
    baseline = {
        "mae_before_mfe_1r_rate": 0.20,
        "mean_max_adverse_before_mfe_1r": 1.0,
        "mean_underwater_bars_before_mfe": 8.0,
        "ev_weighted_first_touch_precision": 0.70,
        "mean_u": -0.001,
    }
    clean = {
        "mae_before_mfe_1r_rate": 0.15,
        "mean_max_adverse_before_mfe_1r": 0.8,
        "mean_underwater_bars_before_mfe": 7.0,
        "ev_weighted_first_touch_precision": 0.75,
        "mean_u": 0.0,
    }
    dirty = {
        "mae_before_mfe_1r_rate": 0.35,
        "mean_max_adverse_before_mfe_1r": 1.8,
        "mean_underwater_bars_before_mfe": 14.0,
        "ev_weighted_first_touch_precision": 0.55,
        "mean_u": -0.004,
    }

    clean_risk = _bucket_risk(
        clean,
        baseline,
        mae_weight=1.0,
        adverse_weight=0.75,
        underwater_weight=0.50,
        ev_weight=1.0,
        mean_u_weight=8.0,
    )
    dirty_risk = _bucket_risk(
        dirty,
        baseline,
        mae_weight=1.0,
        adverse_weight=0.75,
        underwater_weight=0.50,
        ev_weight=1.0,
        mean_u_weight=8.0,
    )

    assert clean_risk == 0.0
    assert dirty_risk > 0.0


def test_select_budgeted_preserves_side_month_budget_and_untargeted_side() -> None:
    ledger = pd.DataFrame(
        {
            "month": ["2026-06"] * 6,
            "side_name": ["long", "long", "long", "short", "short", "short"],
            "selected_top10": [True, True, False, True, False, False],
            "adjusted_score": [0.1, 0.2, 0.9, 0.1, 0.9, 0.8],
        }
    )

    selected = _select_budgeted(
        ledger,
        selected_col="selected_top10",
        adjusted_score_col="adjusted_score",
        sides={"long"},
        group_cols=["month", "side_name"],
    )

    assert selected.iloc[:3].sum() == 2
    assert selected.iloc[2]
    assert selected.iloc[3:].tolist() == [True, False, False]


def test_filter_selected_by_risk_only_abstains_target_side() -> None:
    ledger = pd.DataFrame(
        {
            "side_name": ["long", "long", "short", "short"],
            "selected_top10": [True, True, True, True],
        }
    )
    risk = pd.Series([0.1, 0.8, 0.9, 0.0])

    selected = _filter_selected_by_risk(
        ledger,
        selected_col="selected_top10",
        risk=risk,
        threshold=0.5,
        sides={"long"},
    )

    assert selected.tolist() == [True, False, True, True]
