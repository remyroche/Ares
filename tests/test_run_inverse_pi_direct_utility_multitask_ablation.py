from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_inverse_pi_direct_utility_multitask_ablation import (
    SCORE_FEATURES,
    add_bounded_interactions,
    experiment_arms,
    feature_arms,
    fold_ids,
    stable_global_top_mask,
    task_arms,
    validate_feature_names,
)


def _features() -> tuple[str, ...]:
    asset = tuple(f"asset_{index}" for index in range(29))
    market = tuple(f"market_{index}" for index in range(15))
    bases = (
        "market_median_rv_24h",
        "market_dispersion_1h",
        "market_negative_breadth_4h",
        "market_average_pair_corr_24h",
        "btc_minus_alt_median_ret_24h",
    )
    suffixes = (
        "__delta_1h",
        "__delta_6h",
        "__acceleration_1h",
        "__cumulative_change_24h",
        "__z_72h",
    )
    transition = tuple(
        f"transition_raw__{base}{suffix}" for base in bases for suffix in suffixes
    )
    return (*asset, *market, *transition)


def test_feature_arms_are_fixed_and_interactions_are_bounded() -> None:
    arms = feature_arms(_features())
    assert len(arms["market"]) == 45
    assert len(arms["transition_context"]) == 26
    assert len(arms["market_transition"]) == 70
    interactions = [
        name
        for name in arms["market_transition_interactions"]
        if name.startswith("interaction__")
    ]
    assert len(interactions) == 5
    frame = pd.DataFrame(
        {
            "base_score": [2.0],
            **{
                name.removeprefix("interaction__base_score__x__"): [3.0]
                for name in interactions
            },
        }
    )
    enriched = add_bounded_interactions(frame, interactions)
    assert enriched.loc[0, interactions].eq(6.0).all()


def test_forbidden_outcomes_never_enter_feature_contract() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_feature_names(["market_level", "future_slope_atr_per_hour"])


def test_calendar_blocks_cover_each_month_once_and_allow_reverse_time() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [f"2022-{month:02d}-15T00:00:00Z" for month in range(1, 8)],
                utc=True,
            )
        }
    )
    assert fold_ids(frame).tolist() == [0, 1, 1, 2, 3, 3, 4]


def test_global_top_selection_crosses_timestamps_and_sides() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "__ts__": pd.to_datetime(
                ["2022-01-01", "2022-01-01", "2022-01-02", "2022-01-02"],
                utc=True,
            ),
            "side_name": ["long", "short", "long", "short"],
        }
    )
    mask = stable_global_top_mask(frame, [0.1, 0.9, 0.8, 0.2], 0.5)
    assert frame.loc[mask, "candidate_id"].tolist() == ["b", "c"]


def test_experiment_matrix_keeps_direct_head_as_only_score() -> None:
    features = feature_arms(_features())
    arms = experiment_arms(features)
    assert len(arms) == 6
    tasks = task_arms()
    assert tasks["direct_only"][0].name == "direct_net"
    assert tasks["economic_multitask"][0].name == "direct_net"
    assert tasks["economic_path_multitask"][0].weight == 4.0
    assert (
        sum(task.weight for task in tasks["economic_path_multitask"][1:])
        < tasks["economic_path_multitask"][0].weight
    )
