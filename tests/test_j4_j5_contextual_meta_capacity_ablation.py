from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import run_j4_j5_contextual_meta_capacity_ablation as mod


def test_capacity_ladder_is_bounded_and_contains_regimes() -> None:
    specs = mod._capacity_ladder(5)

    assert len(specs) == 5
    assert {spec.regime for spec in specs} >= {"conservative", "moderate"}
    assert all(spec.num_leaves <= 2 ** spec.max_depth for spec in specs)
    assert all(spec.min_data_in_leaf >= 900 for spec in specs)


def test_trial_table_enforces_directional_and_leaf_constraints() -> None:
    summary = pd.DataFrame(
        {
            "head": ["short_asset"],
            "arm": ["J4_test_seed29"],
            "distillation_variant": ["j4_hard_label_capacity"],
            "config_id": ["test"],
            "seed": [29],
            "regime": ["moderate"],
        }
    )
    directional = pd.DataFrame(
        {
            "head": ["short_asset"],
            "arm": ["J4_test_seed29"],
            "distillation_variant": ["j4_hard_label_capacity"],
            "delta_timestamp_weighted_hr_top30": [0.01],
            "delta_timestamp_weighted_hr_top10": [-0.0005],
            "delta_timestamp_weighted_hr_top20": [0.0],
            "normal_period_delta_hr_top30": [0.0],
            "delta_ndcg_top30": [0.002],
            "net_correct_trades_gained": [5],
        }
    )
    directional_episode = pd.DataFrame(
        {
            "head": ["short_asset", "short_asset", "short_asset", "short_asset"],
            "arm": ["J4_test_seed29"] * 4,
            "distillation_variant": ["j4_hard_label_capacity"] * 4,
            "period_type": ["bad_episode"] * 4,
            "delta_timestamp_weighted_hr_top30": [0.02, 0.01, 0.03, -0.01],
        }
    )
    folds = pd.DataFrame(
        {
            "head": ["short_asset"],
            "arm": ["J4_test_seed29"],
            "distillation_variant": ["j4_hard_label_capacity"],
            "leaf_count_min": [100],
            "leaf_count_q10": [150],
            "context_split_count": [3],
            "context_split_share": [0.2],
            "context_gain_share": [0.1],
        }
    )

    out = mod._trial_table(summary, directional, directional_episode, folds, hr_tolerance=0.001, leaf_floor=50)

    row = out.iloc[0]
    assert bool(row["passes_hard_constraints"])
    assert bool(row["passes_episode_recurrence"])
    assert bool(row["trial_promoted"])
    assert row["episode_positive_rate_delta_hr30"] == 0.75


def test_config_table_requires_seed_pass_rate() -> None:
    trials = pd.DataFrame(
        {
            "head": ["short_asset"] * 3,
            "config_id": ["cfg"] * 3,
            "regime": ["moderate"] * 3,
            "seed": [29, 31, 37],
            "trial_promoted": [True, True, False],
            "episode_selection_delta_hr30": [0.02, 0.01, -0.01],
            "episode_q25_selection_delta_hr30": [0.01, 0.0, -0.02],
            "delta_timestamp_weighted_hr_top30": [0.01, 0.02, -0.01],
            "delta_ndcg_top30": [0.002, 0.001, -0.001],
            "net_correct_trades_gained": [5, 4, -2],
            "delta_timestamp_weighted_hr_top10": [0.0, 0.0, -0.002],
            "delta_timestamp_weighted_hr_top20": [0.0, 0.0, -0.002],
            "leaf_count_min": [100, 100, 100],
            "context_split_share": [0.1, 0.2, 0.3],
            "context_gain_share": [0.1, 0.1, 0.2],
        }
    )

    out = mod._config_table(trials, min_seed_pass_rate=2 / 3)

    row = out.iloc[0]
    assert row["seed_count"] == 3
    assert row["seed_pass_rate"] == 2 / 3
    assert bool(row["config_promoted"])


def test_leaf_split_diagnostics_handles_leaf_only_tree() -> None:
    class Booster:
        def dump_model(self):
            return {"tree_info": [{"tree_structure": {"leaf_value": 0.1}}]}

    out = mod._leaf_split_diagnostics(Booster(), ["a"], {"a"})

    assert out["split_count"] == 0
    assert out["context_split_count"] == 0
    assert np.isnan(out["leaf_count_min"])
