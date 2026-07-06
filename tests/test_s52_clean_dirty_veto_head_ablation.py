from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s52_clean_dirty_veto_head_ablation import (  # noqa: E402
    _combine_scores,
    _parse_alpha_grid,
    _rank_pct,
    _select_calibrated_alpha,
    _veto_population_and_label,
)


def test_rank_pct_handles_nan_and_preserves_order() -> None:
    rank = _rank_pct(pd.Series([10.0, float("nan"), 30.0, 20.0]))

    assert rank.iloc[2] > rank.iloc[3] > rank.iloc[0] > rank.iloc[1]
    assert rank.iloc[1] == 0.0


def test_veto_policy_demotes_low_clean_probability() -> None:
    base = pd.Series([1.0, 2.0, 3.0, 4.0])
    clean = pd.Series([0.9, 0.8, 0.4, 0.3])

    combined = _combine_scores(base, clean, policy="veto_60")

    assert combined.iloc[1] > combined.iloc[2]
    assert combined.iloc[0] > combined.iloc[3]


def test_multiplicative_policy_requires_opportunity_and_clean_score() -> None:
    base = pd.Series([1.0, 4.0, 3.0])
    clean = pd.Series([0.95, 0.10, 0.80])

    combined = _combine_scores(base, clean, policy="multiplicative")

    assert combined.iloc[2] > combined.iloc[1]
    assert combined.iloc[2] > combined.iloc[0]


def test_conservative_agreement_keeps_base_order_when_clean_scores_are_close() -> None:
    base = pd.Series([1.0, 2.0, 3.0])
    clean = pd.Series([0.55, 0.56, 0.57])

    combined = _combine_scores(base, clean, policy="agreement_10")

    assert combined.iloc[2] > combined.iloc[1] > combined.iloc[0]


def test_explicit_agreement_alpha_policy_uses_requested_weight() -> None:
    base = pd.Series([1.0, 2.0, 3.0])
    clean = pd.Series([1.0, 0.0, 0.0])

    low_alpha = _combine_scores(base, clean, policy="agreement_alpha_0.10")
    high_alpha = _combine_scores(base, clean, policy="agreement_alpha_0.90")

    assert low_alpha.iloc[2] > low_alpha.iloc[0]
    assert high_alpha.iloc[0] > high_alpha.iloc[2]


def test_calibrated_agreement_requires_alpha() -> None:
    with pytest.raises(ValueError):
        _combine_scores(pd.Series([1.0]), pd.Series([1.0]), policy="calibrated_agreement")


def test_parse_alpha_grid_clips_sorts_and_deduplicates() -> None:
    assert _parse_alpha_grid("0.2,1.5,0.2,-1") == [0.0, 0.2, 1.0]


def test_select_calibrated_alpha_uses_objective() -> None:
    alpha_rows = {
        0.1: [
            {"top10_mean_ev": -0.01, "top20_mean_ev": -0.01, "top30_mean_ev": -0.01},
            {"top10_mean_ev": -0.02, "top20_mean_ev": -0.01, "top30_mean_ev": -0.01},
        ],
        0.5: [
            {"top10_mean_ev": 0.01, "top20_mean_ev": 0.01, "top30_mean_ev": 0.00},
            {"top10_mean_ev": 0.02, "top20_mean_ev": 0.01, "top30_mean_ev": 0.00},
        ],
    }

    selected = _select_calibrated_alpha(alpha_rows, objective_mode="pnl_only", default_alpha=0.2)

    assert selected["alpha"] == 0.5
    assert selected["calibration_folds"] == 2


def test_unknown_policy_raises() -> None:
    with pytest.raises(ValueError):
        _combine_scores(pd.Series([1.0]), pd.Series([1.0]), policy="not_a_policy")


def test_materialized_s52_veto_label_uses_first_touch_path_cleanliness() -> None:
    frame = pd.DataFrame(
        {
            "__first_touch_target_soft__": [0.9, 0.8, 0.1],
            "__first_touch_hit__": [1.0, 1.0, 0.0],
            "__first_touch_stop__": [0.0, 0.0, 1.0],
            "__first_touch_timeout__": [0.0, 0.0, 0.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "u_policy_net": [0.01, 0.02, -0.01],
            "first_touch_available": [1.0, 1.0, 1.0],
            "first_touch_mae_norm": [0.25, 1.25, 0.30],
            "mfe_1r_before_mae_1r": [1.0, 0.0, 0.0],
            "mae_1r_before_mfe_1r": [0.0, 1.0, 0.0],
            "max_adverse_before_mfe_1r": [0.50, 2.00, 0.20],
            "underwater_bars_before_mfe_1r": [4.0, 18.0, 2.0],
            "underwater_fraction_before_mfe_1r": [0.20, 0.60, 0.10],
        }
    )

    population, target, diag = _veto_population_and_label(
        frame,
        metrics,
        label_variant="s52_materialized_path_clean",
    )

    assert population.tolist() == [True, True, True]
    assert target.tolist() == [1, 0, 0]
    assert diag["clean_rows"] == 1
