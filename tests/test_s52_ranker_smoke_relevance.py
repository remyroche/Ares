from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s52_ranker_smoke import _materialized_soft_label, _rank_relevance
from scripts.run_s52_ranker_smoke import _ranker_model_params, _ranker_sample_weight
from scripts.run_s52_clean_gross_ranker_hpo import _topk_gate_penalty


def test_fullpath_rank_relevance_demotes_dirty_after_first_touch() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-06-01 00:00:00", "2026-06-01 00:00:00"]),
        }
    )
    label = pd.DataFrame(
        {
            "target_soft": [0.8, 0.8],
            "first_pass_good": [1.0, 1.0],
            "first_pass_bad": [0.0, 0.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "side": [1.0, 1.0],
            "mae_1r_before_mfe_1r": [0.0, 0.0],
            "first_touch_mae_norm": [0.25, 0.25],
            "first_touch_full_path_mae_norm": [0.60, 3.00],
            "first_touch_timeout": [0.0, 0.0],
        }
    )

    first_touch_rel = _rank_relevance(
        label,
        metrics,
        frame,
        group_mode="timestamp",
        relevance_mode="first_touch",
    )
    fullpath_rel = _rank_relevance(
        label,
        metrics,
        frame,
        group_mode="timestamp",
        relevance_mode="fullpath",
    )

    assert first_touch_rel.tolist() == [3, 3]
    assert fullpath_rel[0] > fullpath_rel[1]


def test_rank_relevance_demotes_slow_underwater_path_order() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-06-01 00:00:00", "2026-06-01 00:00:00"]),
        }
    )
    label = pd.DataFrame(
        {
            "target_soft": [0.8, 0.8],
            "first_pass_good": [1.0, 1.0],
            "first_pass_bad": [0.0, 0.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "side": [1.0, 1.0],
            "mae_1r_before_mfe_1r": [0.0, 0.0],
            "first_touch_mae_norm": [0.25, 0.25],
            "first_touch_full_path_mae_norm": [0.60, 0.60],
            "first_touch_timeout": [0.0, 0.0],
            "max_adverse_before_mfe_1r": [0.50, 1.80],
            "underwater_bars_before_mfe_1r": [4.0, 18.0],
            "underwater_fraction_before_mfe_1r": [0.20, 0.55],
        }
    )

    relevance = _rank_relevance(
        label,
        metrics,
        frame,
        group_mode="timestamp",
        relevance_mode="first_touch",
    )

    assert relevance[0] > relevance[1]


def test_evpath_rank_relevance_prefers_larger_clean_gross_capture() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-06-01 00:00:00", "2026-06-01 00:00:00"]),
        }
    )
    label = pd.DataFrame(
        {
            "target_soft": [0.8, 0.8],
            "first_pass_good": [1.0, 1.0],
            "first_pass_bad": [0.0, 0.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "side": [1.0, 1.0],
            "first_touch_net": [-0.006, 0.004],
            "mae_1r_before_mfe_1r": [0.0, 0.0],
            "first_touch_mae_norm": [0.25, 0.25],
            "max_adverse_before_mfe_1r": [0.50, 0.50],
            "underwater_bars_before_mfe_1r": [4.0, 4.0],
            "underwater_fraction_before_mfe_1r": [0.20, 0.20],
            "first_touch_timeout": [0.0, 0.0],
        }
    )

    first_touch_rel = _rank_relevance(
        label,
        metrics,
        frame,
        group_mode="timestamp",
        relevance_mode="first_touch",
        round_trip_cost=0.01,
    )
    evpath_rel = _rank_relevance(
        label,
        metrics,
        frame,
        group_mode="timestamp",
        relevance_mode="evpath",
        round_trip_cost=0.01,
    )

    assert first_touch_rel.tolist() == [3, 3]
    assert evpath_rel[1] > evpath_rel[0]


def test_cleangross_rank_relevance_prefers_clean_gross_tiers() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                ]
            ),
        }
    )
    label = pd.DataFrame(
        {
            "target_soft": [0.8, 0.8, 0.8],
            "first_pass_good": [1.0, 1.0, 1.0],
            "first_pass_bad": [0.0, 0.0, 0.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "side": [1.0, 1.0, 1.0],
            "first_touch_net": [-0.005, 0.004, 0.020],
            "mae_1r_before_mfe_1r": [0.0, 0.0, 1.0],
            "first_touch_mae_norm": [0.25, 0.25, 1.25],
            "max_adverse_before_mfe_1r": [0.50, 0.50, 2.00],
            "underwater_bars_before_mfe_1r": [4.0, 4.0, 18.0],
            "underwater_fraction_before_mfe_1r": [0.20, 0.20, 0.55],
            "first_touch_timeout": [0.0, 0.0, 0.0],
        }
    )

    relevance = _rank_relevance(
        label,
        metrics,
        frame,
        group_mode="timestamp",
        relevance_mode="cleangross",
        round_trip_cost=0.01,
    )

    assert relevance[1] > relevance[0]
    assert relevance[1] > relevance[2]


def test_ordered_clean_ev_relevance_bottoms_dirty_high_gross_paths() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                ]
            ),
        }
    )
    label = pd.DataFrame(
        {
            "target_soft": [0.7, 0.9, 0.8, 0.6],
            "target_hard": [1.0, 0.0, 1.0, 0.0],
            "dirty_positive": [0.0, 1.0, 0.0, 0.0],
            "first_pass_good": [1.0, 1.0, 1.0, 0.0],
            "first_pass_bad": [0.0, 1.0, 0.0, 0.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "side": [1.0, 1.0, 1.0, 1.0],
            "first_touch_net": [0.002, 0.050, 0.010, -0.002],
            "mae_1r_before_mfe_1r": [0.0, 1.0, 0.0, 0.0],
            "first_touch_mae_norm": [0.30, 1.50, 0.25, 0.20],
            "max_adverse_before_mfe_1r": [0.70, 2.50, 0.40, 0.20],
            "underwater_bars_before_mfe_1r": [5.0, 20.0, 3.0, 2.0],
            "underwater_fraction_before_mfe_1r": [0.25, 0.60, 0.15, 0.10],
            "first_touch_timeout": [0.0, 0.0, 0.0, 0.0],
        }
    )

    relevance = _rank_relevance(
        label,
        metrics,
        frame,
        group_mode="timestamp",
        relevance_mode="ordered_clean_ev",
        round_trip_cost=0.01,
    )

    assert relevance[2] > relevance[0]
    assert relevance[1] == 0
    assert relevance[3] == 0


def test_soft_ordered_ev_relevance_preserves_near_clean_breadth() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                ]
            ),
        }
    )
    label = pd.DataFrame(
        {
            "target_soft": [0.70, 0.90, 0.75, 0.40],
            "target_hard": [1.0, 0.0, 0.0, 0.0],
            "dirty_positive": [0.0, 1.0, 0.0, 0.0],
            "first_pass_good": [1.0, 1.0, 1.0, 0.0],
            "first_pass_bad": [0.0, 1.0, 0.0, 0.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "side": [1.0, 1.0, 1.0, 1.0],
            "first_touch_net": [0.006, 0.050, 0.004, -0.002],
            "mae_1r_before_mfe_1r": [0.0, 1.0, 0.0, 0.0],
            "first_touch_mae_norm": [0.30, 1.50, 0.40, 0.20],
            "max_adverse_before_mfe_1r": [0.70, 2.50, 1.35, 0.20],
            "underwater_bars_before_mfe_1r": [5.0, 20.0, 12.0, 2.0],
            "underwater_fraction_before_mfe_1r": [0.25, 0.60, 0.50, 0.10],
            "first_touch_timeout": [0.0, 0.0, 0.0, 0.0],
        }
    )

    relevance = _rank_relevance(
        label,
        metrics,
        frame,
        group_mode="timestamp",
        relevance_mode="soft_ordered_ev",
        round_trip_cost=0.01,
    )

    assert relevance[0] > relevance[1]
    assert relevance[2] > relevance[1]
    assert relevance[2] > 0
    assert relevance[1] <= relevance[3]


def test_exec_ordered_ev_relevance_requires_positive_net_clean_path() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                ]
            ),
        }
    )
    label = pd.DataFrame(
        {
            "target_soft": [0.70, 0.90, 0.80, 0.65],
            "target_hard": [1.0, 0.0, 1.0, 0.0],
            "dirty_positive": [0.0, 1.0, 0.0, 0.0],
            "first_pass_good": [1.0, 1.0, 1.0, 1.0],
            "first_pass_bad": [0.0, 1.0, 0.0, 0.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "side": [1.0, 1.0, 1.0, 1.0],
            "first_touch_net": [-0.001, 0.050, 0.004, 0.002],
            "mae_1r_before_mfe_1r": [0.0, 1.0, 0.0, 0.0],
            "first_touch_mae_norm": [0.20, 1.50, 0.25, 0.25],
            "max_adverse_before_mfe_1r": [0.40, 2.50, 0.40, 1.55],
            "underwater_bars_before_mfe_1r": [3.0, 20.0, 3.0, 12.0],
            "underwater_fraction_before_mfe_1r": [0.15, 0.60, 0.15, 0.50],
            "first_touch_timeout": [0.0, 0.0, 0.0, 0.0],
        }
    )

    relevance = _rank_relevance(
        label,
        metrics,
        frame,
        group_mode="timestamp",
        relevance_mode="exec_ordered_ev",
        round_trip_cost=0.01,
    )

    assert relevance[2] > relevance[0]
    assert relevance[2] > relevance[1]
    assert relevance[2] > relevance[3]
    assert relevance[0] == 0
    assert relevance[1] == 0


def test_soft_exec_ordered_ev_prefers_positive_net_without_promoting_dirty_path() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                ]
            ),
        }
    )
    label = pd.DataFrame(
        {
            "target_soft": [0.70, 0.70, 0.95, 0.65],
            "target_hard": [1.0, 1.0, 0.0, 0.0],
            "dirty_positive": [0.0, 0.0, 1.0, 0.0],
            "first_pass_good": [1.0, 1.0, 1.0, 1.0],
            "first_pass_bad": [0.0, 0.0, 1.0, 0.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "side": [1.0, 1.0, 1.0, 1.0],
            "first_touch_net": [-0.001, 0.004, 0.050, 0.002],
            "mae_1r_before_mfe_1r": [0.0, 0.0, 1.0, 0.0],
            "first_touch_mae_norm": [0.25, 0.25, 1.50, 0.30],
            "max_adverse_before_mfe_1r": [0.40, 0.40, 2.50, 1.35],
            "underwater_bars_before_mfe_1r": [3.0, 3.0, 20.0, 12.0],
            "underwater_fraction_before_mfe_1r": [0.15, 0.15, 0.60, 0.50],
            "first_touch_timeout": [0.0, 0.0, 0.0, 0.0],
        }
    )

    relevance = _rank_relevance(
        label,
        metrics,
        frame,
        group_mode="timestamp",
        relevance_mode="soft_exec_ordered_ev",
        round_trip_cost=0.01,
    )

    assert relevance[1] > relevance[0]
    assert relevance[1] > relevance[2]
    assert relevance[3] > relevance[2]
    assert relevance[3] > 0


def test_soft_breadth_ordered_ev_demotes_high_net_underwater_path() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                ]
            ),
        }
    )
    label = pd.DataFrame(
        {
            "target_soft": [0.70, 0.95, 0.65, 0.40],
            "target_hard": [1.0, 0.0, 0.0, 0.0],
            "dirty_positive": [0.0, 1.0, 0.0, 0.0],
            "first_pass_good": [1.0, 1.0, 1.0, 0.0],
            "first_pass_bad": [0.0, 1.0, 0.0, 0.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "side": [1.0, 1.0, 1.0, 1.0],
            "first_touch_net": [0.004, 0.050, 0.002, -0.002],
            "mae_1r_before_mfe_1r": [0.0, 1.0, 0.0, 0.0],
            "first_touch_mae_norm": [0.25, 1.50, 0.25, 0.20],
            "max_adverse_before_mfe_1r": [0.45, 2.50, 1.15, 0.20],
            "underwater_bars_before_mfe_1r": [3.0, 20.0, 9.0, 2.0],
            "underwater_fraction_before_mfe_1r": [0.15, 0.60, 0.40, 0.10],
            "first_touch_timeout": [0.0, 0.0, 0.0, 0.0],
        }
    )

    relevance = _rank_relevance(
        label,
        metrics,
        frame,
        group_mode="timestamp",
        relevance_mode="soft_breadth_ordered_ev",
        round_trip_cost=0.01,
    )

    assert relevance[0] > relevance[1]
    assert relevance[2] > relevance[1]
    assert relevance[2] > 0


def test_firstpass_exec_ev_uses_absolute_tiers_for_dirty_local_groups() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                ]
            ),
        }
    )
    label = pd.DataFrame(
        {
            "target_soft": [0.70, 0.95, 0.65, 0.90],
            "dirty_positive": [0.0, 1.0, 0.0, 1.0],
            "first_pass_good": [1.0, 1.0, 1.0, 1.0],
            "first_pass_bad": [0.0, 0.0, 0.0, 1.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "side": [1.0, 1.0, 1.0, 1.0],
            "first_touch_net": [0.004, 0.050, -0.001, 0.060],
            "mae_1r_before_mfe_1r": [0.0, 1.0, 0.0, 1.0],
            "first_touch_mae_norm": [0.25, 1.50, 0.25, 1.70],
            "first_touch_mfe_norm": [1.20, 4.00, 1.20, 5.00],
            "max_adverse_before_mfe_1r": [0.45, 2.50, 0.90, 3.00],
            "underwater_bars_before_mfe_1r": [3.0, 20.0, 7.0, 24.0],
            "underwater_fraction_before_mfe_1r": [0.15, 0.60, 0.33, 0.75],
            "first_touch_timeout": [0.0, 0.0, 0.0, 0.0],
        }
    )

    relevance = _rank_relevance(
        label,
        metrics,
        frame,
        group_mode="timestamp",
        relevance_mode="firstpass_exec_ev",
        round_trip_cost=0.01,
    )

    assert relevance[0] >= 3
    assert relevance[2] <= 1
    assert relevance[1] == 0
    assert relevance[3] == 0


def test_ev_preserving_ordered_relevance_keeps_ev_positive_clean_paths_on_top() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                ]
            ),
        }
    )
    label = pd.DataFrame(
        {
            "target_soft": [0.75, 0.95, 0.70, 0.70],
            "target_hard": [1.0, 0.0, 1.0, 0.0],
            "dirty_positive": [0.0, 1.0, 0.0, 0.0],
            "first_pass_good": [1.0, 1.0, 1.0, 1.0],
            "first_pass_bad": [0.0, 1.0, 0.0, 0.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "side": [1.0, 1.0, 1.0, 1.0],
            "first_touch_net": [0.004, 0.050, -0.001, 0.002],
            "mae_1r_before_mfe_1r": [0.0, 1.0, 0.0, 0.0],
            "first_touch_mae_norm": [0.25, 1.50, 0.25, 0.25],
            "max_adverse_before_mfe_1r": [0.45, 2.50, 0.45, 1.45],
            "underwater_bars_before_mfe_1r": [3.0, 20.0, 3.0, 11.0],
            "underwater_fraction_before_mfe_1r": [0.15, 0.60, 0.15, 0.49],
            "first_touch_timeout": [0.0, 0.0, 0.0, 0.0],
        }
    )

    relevance = _rank_relevance(
        label,
        metrics,
        frame,
        group_mode="timestamp",
        relevance_mode="ev_preserving_ordered",
        round_trip_cost=0.01,
    )

    assert relevance[0] > relevance[2]
    assert relevance[0] > relevance[1]
    assert relevance[0] > relevance[3]
    assert relevance[1] == 0
    assert relevance[2] <= 1


def test_pathnet_guarded_relevance_requires_net_and_bounded_full_path() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                ]
            ),
        }
    )
    label = pd.DataFrame(
        {
            "target_soft": [0.80, 0.95, 0.75, 0.70],
            "dirty_positive": [0.0, 0.0, 0.0, 0.0],
            "first_pass_good": [1.0, 1.0, 1.0, 1.0],
            "first_pass_bad": [0.0, 0.0, 0.0, 0.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "side": [1.0, 1.0, -1.0, -1.0],
            "first_touch_net": [0.004, 0.030, -0.001, 0.006],
            "mae_1r_before_mfe_1r": [0.0, 0.0, 0.0, 0.0],
            "first_touch_mae_norm": [0.25, 0.25, 0.25, 0.25],
            "first_touch_full_path_mae_norm": [0.80, 3.20, 0.70, 1.20],
            "max_adverse_before_mfe_1r": [0.45, 0.45, 0.40, 0.70],
            "underwater_bars_before_mfe_1r": [3.0, 3.0, 3.0, 5.0],
            "underwater_fraction_before_mfe_1r": [0.15, 0.15, 0.15, 0.20],
            "first_touch_timeout": [0.0, 0.0, 0.0, 0.0],
        }
    )

    relevance = _rank_relevance(
        label,
        metrics,
        frame,
        group_mode="timestamp_side",
        relevance_mode="pathnet_guarded",
        round_trip_cost=0.01,
    )

    assert relevance[0] > relevance[1]
    assert relevance[0] >= 3
    assert relevance[1] == 0
    assert relevance[2] == 0
    assert relevance[3] >= 2


def test_materialized_soft_label_uses_artifact_target_and_path_cleanliness() -> None:
    frame = pd.DataFrame(
        {
            "__first_touch_target_soft__": [0.9, 0.8],
            "__first_touch_hit__": [1.0, 1.0],
            "__first_touch_stop__": [0.0, 0.0],
            "__first_touch_timeout__": [0.0, 0.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "u_policy_net": [0.01, 0.01],
            "first_touch_available": [1.0, 1.0],
            "first_touch_mae_norm": [0.25, 0.25],
            "mfe_1r_before_mae_1r": [1.0, 1.0],
            "mae_1r_before_mfe_1r": [0.0, 0.0],
            "max_adverse_before_mfe_1r": [0.5, 1.8],
            "underwater_bars_before_mfe_1r": [4.0, 18.0],
            "underwater_fraction_before_mfe_1r": [0.2, 0.55],
        }
    )

    label = _materialized_soft_label(frame, metrics)

    assert label["target_soft"].tolist() == pytest.approx([0.9, 0.8])
    assert label["target_hard"].tolist() == [1, 0]
    assert label["first_pass_good"].tolist() == [1, 1]
    assert label["first_pass_bad"].tolist() == [0, 1]


def test_ranker_model_params_accept_bounded_overrides() -> None:
    params = _ranker_model_params(
        seed=7,
        ranker_params={"n_estimators": 260, "learning_rate": 0.025, "reg_lambda": 6.0},
    )

    assert params["objective"] == "lambdarank"
    assert params["random_state"] == 7
    assert params["n_estimators"] == 260
    assert params["learning_rate"] == pytest.approx(0.025)
    assert params["reg_lambda"] == pytest.approx(6.0)


def test_hpo_gate_shortfall_penalty_rewards_gate_closeness() -> None:
    weak = {
        "mean_top10_ev_weighted_first_touch_precision": 0.62,
        "mean_top20_ev_weighted_first_touch_precision": 0.50,
        "mean_top30_ev_weighted_first_touch_precision": 0.40,
        "mean_top10_first_touch_bad_mae_1r_rate": 0.20,
        "mean_top10_mean_underwater_bars_before_mfe": 12.0,
        "mean_top10_mean_ev": -0.02,
    }
    stronger = {
        "mean_top10_ev_weighted_first_touch_precision": 0.70,
        "mean_top20_ev_weighted_first_touch_precision": 0.60,
        "mean_top30_ev_weighted_first_touch_precision": 0.50,
        "mean_top10_first_touch_bad_mae_1r_rate": 0.12,
        "mean_top10_mean_underwater_bars_before_mfe": 9.0,
        "mean_top10_mean_ev": -0.005,
    }

    assert _topk_gate_penalty(stronger) < _topk_gate_penalty(weak)


def test_long_clean_dirty_sample_weight_emphasizes_long_ambiguity() -> None:
    metrics = pd.DataFrame(
        {
            "side": [1.0, 1.0, -1.0],
            "u_policy_net": [0.02, 0.02, 0.02],
            "mae_norm": [0.25, 1.50, 1.50],
            "is_timeout": [0.0, 0.0, 0.0],
            "mae_1r_before_mfe_1r": [0.0, 1.0, 1.0],
            "first_touch_mae_norm": [0.25, 1.25, 1.25],
            "max_adverse_before_mfe_1r": [0.50, 2.25, 2.25],
            "underwater_bars_before_mfe_1r": [4.0, 18.0, 18.0],
        }
    )
    label = pd.DataFrame(
        {
            "target_hard": [1.0, 0.0, 0.0],
            "dirty_positive": [0.0, 1.0, 1.0],
            "positive_u": [1.0, 1.0, 1.0],
            "first_pass_good": [1.0, 1.0, 1.0],
            "first_pass_bad": [0.0, 1.0, 1.0],
        }
    )

    weights = _ranker_sample_weight(metrics, label, round_trip_cost=0.01, mode="long_clean_dirty")

    assert float(weights.iloc[1]) > float(weights.iloc[2])
    assert float(weights.mean()) == pytest.approx(1.0)
