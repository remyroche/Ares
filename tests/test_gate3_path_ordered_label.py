from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_gate3_side_soft_label_hpo import _family_config, _make_side_soft_label, _sample_weight
from scripts.run_label_quality_proxy_diagnostics import _path_metrics


def test_s52_demotes_adverse_first_positive_path() -> None:
    frame = pd.DataFrame(
        {
            "__barrier_pct__": [0.01, 0.01],
            "__mfe_ret__": [0.015, 0.020],
            "__mae_ret__": [0.003, 0.012],
            "__bars_to_mfe__": [2, 8],
            "__bars_policy__": [4, 10],
            "__y_ret__": [0.010, 0.010],
            "__y_bin__": [1, 1],
            "__u_policy_net__": [0.012, 0.011],
            "__is_timeout__": [0, 0],
            "side": [1, 1],
            "__bars_to_mfe_125r__": [2, 6],
            "__bars_to_mfe_1r__": [1, 5],
            "__bars_to_mae_075r__": [-1, 2],
            "__bars_to_mae_1r__": [-1, 3],
            "__max_adverse_before_mfe_1r__": [0.2, 1.2],
            "__underwater_bars_before_mfe_1r__": [1, 4],
        }
    )

    metrics = _path_metrics(frame)
    labels = _make_side_soft_label(
        metrics,
        _family_config("long_fast_short_controlled"),
        round_trip_cost=0.010,
        path_order_mode="s52_first_touch",
        target_utility_mode="geometry_only",
    )

    assert labels.loc[0, "target_hard"] == 1
    assert labels.loc[0, "first_pass_good"] == 1
    assert labels.loc[0, "first_pass_bad"] == 0
    assert labels.loc[1, "target_hard"] == 0
    assert labels.loc[1, "first_pass_good"] == 0
    assert labels.loc[1, "first_pass_bad"] == 1
    assert labels.loc[1, "target_soft"] < labels.loc[0, "target_soft"]


def test_s52_uses_actual_underwater_bars_over_proxy() -> None:
    frame = pd.DataFrame(
        {
            "__barrier_pct__": [0.01, 0.01],
            "__mfe_ret__": [0.018, 0.018],
            "__mae_ret__": [0.002, 0.002],
            "__bars_to_mfe__": [2, 2],
            "__bars_policy__": [4, 4],
            "__y_ret__": [0.012, 0.012],
            "__y_bin__": [1, 1],
            "__u_policy_net__": [0.012, 0.012],
            "__is_timeout__": [0, 0],
            "side": [1, 1],
            "__bars_to_mfe_125r__": [2, 2],
            "__bars_to_mfe_1r__": [1, 1],
            "__bars_to_mae_075r__": [-1, -1],
            "__bars_to_mae_1r__": [-1, -1],
            "__max_adverse_before_mfe_1r__": [0.2, 0.2],
            "__underwater_bars_before_mfe_1r__": [1, 20],
        }
    )

    metrics = _path_metrics(frame)
    metrics["underwater_bars_before_mfe_proxy"] = [1.0, 1.0]
    labels = _make_side_soft_label(
        metrics,
        _family_config("long_fast_short_controlled"),
        round_trip_cost=0.010,
        path_order_mode="s52_first_touch",
        target_utility_mode="geometry_only",
    )

    cap = _family_config("long_fast_short_controlled").long.ordered_dirty_cap
    assert labels.loc[0, "target_hard"] == 1
    assert labels.loc[1, "target_hard"] == 0
    assert labels.loc[1, "target_soft"] <= cap
    assert labels.loc[1, "target_soft"] < labels.loc[0, "target_soft"]


def test_s52_first_touch_net_mode_rejects_clean_path_with_negative_executable_ev() -> None:
    frame = pd.DataFrame(
        {
            "__barrier_pct__": [0.01, 0.01],
            "__mfe_ret__": [0.018, 0.018],
            "__mae_ret__": [0.002, 0.002],
            "__bars_to_mfe__": [2, 2],
            "__bars_policy__": [4, 4],
            "__y_ret__": [0.012, 0.012],
            "__y_bin__": [1, 1],
            "__u_policy_net__": [0.020, 0.020],
            "__stage167_first_touch_net__": [-0.001, 0.004],
            "__first_touch_hit__": [1.0, 1.0],
            "__is_timeout__": [0, 0],
            "side": [1, 1],
            "__bars_to_mfe_125r__": [2, 2],
            "__bars_to_mfe_1r__": [1, 1],
            "__bars_to_mae_075r__": [-1, -1],
            "__bars_to_mae_1r__": [-1, -1],
            "__max_adverse_before_mfe_1r__": [0.2, 0.2],
            "__underwater_bars_before_mfe_1r__": [1, 1],
        }
    )

    metrics = _path_metrics(frame)
    config = _family_config("long_fast_short_controlled")
    labels = _make_side_soft_label(
        metrics,
        config,
        round_trip_cost=0.010,
        path_order_mode="s52_first_touch",
        target_utility_mode="first_touch_net",
    )

    assert labels.loc[0, "target_hard"] == 0
    assert labels.loc[0, "positive_u"] == 0
    assert labels.loc[0, "target_soft"] <= config.long.ordered_dirty_cap
    assert labels.loc[1, "target_hard"] == 1
    assert labels.loc[1, "positive_u"] == 1
    assert labels.loc[1, "target_soft"] > labels.loc[0, "target_soft"]


def test_first_touch_net_mode_weights_first_touch_edge_not_stale_raw_utility() -> None:
    frame = pd.DataFrame(
        {
            "__barrier_pct__": [0.01, 0.01],
            "__mfe_ret__": [0.018, 0.018],
            "__mae_ret__": [0.002, 0.002],
            "__bars_to_mfe__": [2, 2],
            "__bars_policy__": [4, 4],
            "__y_ret__": [0.012, 0.012],
            "__y_bin__": [1, 1],
            "__u_policy_net__": [0.030, -0.001],
            "__stage167_first_touch_net__": [-0.004, 0.006],
            "__first_touch_hit__": [0.0, 1.0],
            "__is_timeout__": [0, 0],
            "side": [1, 1],
        }
    )
    metrics = _path_metrics(frame)
    label = pd.DataFrame(
        {
            "target_hard": [0.0, 0.0],
            "dirty_positive": [1.0, 1.0],
        }
    )

    first_touch_weights = _sample_weight(
        metrics,
        label,
        round_trip_cost=0.010,
        target_utility_mode="first_touch_net",
    )
    raw_weights = _sample_weight(
        metrics,
        label,
        round_trip_cost=0.010,
        target_utility_mode="raw_positive",
    )

    assert first_touch_weights.iloc[1] > first_touch_weights.iloc[0]
    assert raw_weights.iloc[0] > raw_weights.iloc[1]
