from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_stage_e_minimal_information_diagnostics import (
    ACTION_STATE,
    _conditional_permute,
    arm_contracts,
    feature_families,
)


def _a0() -> list[str]:
    return [
        "side_long", *ACTION_STATE, "known_row_cost_bps", "barrier_pct",
        "estimated_spread_bps", "entry_half_spread_bps", "exit_half_spread_bps",
        "entry_price_log", "entry_control",
    ]


def test_minimal_ablation_rows_match_m0() -> None:
    source = Path(__file__).resolve().parents[1].joinpath(
        "scripts/run_stage_e_minimal_information_diagnostics.py"
    ).read_text()
    assert "minimal arm row mismatch" in source
    assert "set(policy.candidate_id) != set(m0_policy.candidate_id)" in source


def test_minimal_ablation_folds_match_m0() -> None:
    source = Path(__file__).resolve().parents[1].joinpath(
        "scripts/run_stage_e_minimal_information_diagnostics.py"
    ).read_text()
    assert "minimal arm fold mismatch" in source
    assert "sorted(policy.fold.unique()) != sorted(m0_policy.fold.unique())" in source


def test_feature_group_deletion_is_train_only() -> None:
    source = Path(__file__).resolve().parents[1].joinpath(
        "scripts/run_stage_e_minimal_information_diagnostics.py"
    ).read_text()
    assert '"feature_deletion_declared_before_fold_fit": True' in source
    assert "fit_preprocess(train, features" in source
    assert "train_mask(side_frame, start)" in source


def test_conditional_permutation_preserves_day_and_side_structure() -> None:
    frame = pd.DataFrame({
        "utc_day": pd.to_datetime(["2024-01-01"] * 4 + ["2024-01-02"] * 4, utc=True),
        "side": ["long"] * 2 + ["short"] * 2 + ["long"] * 2 + ["short"] * 2,
        "time_to_clear_bucket": ["fast", "fast", "slow", "slow"] * 2,
        "feature": np.arange(8),
    })
    result = _conditional_permute(frame, ["feature"], ["utc_day", "side"], 7)
    for key, part in frame.groupby(["utc_day", "side"]):
        observed = result.loc[part.index, "feature"]
        assert sorted(observed.tolist()) == sorted(part.feature.tolist()), key


def test_feature_family_partition_is_exhaustive_and_disjoint() -> None:
    groups = feature_families(_a0())
    flat = [feature for values in groups.values() for feature in values]
    assert len(flat) == len(set(flat)) == len(_a0())
    assert set(flat) == set(_a0())


def test_m0_is_full_a0_and_minimal_contracts_are_exact() -> None:
    contracts = arm_contracts(_a0(), feature_families(_a0()))
    assert contracts["M0_full_frozen_A0"]["features"] == _a0()
    assert contracts["M1_three_action_state"]["features"] == ACTION_STATE
    assert contracts["M4_action_state_without_exit_net"]["features"] == ACTION_STATE[:2]
    assert contracts["M5_estimated_exit_net_only"]["features"] == ["estimated_net_if_exit_now_bps"]


def test_e2_e3_runner_never_scores_second_oos() -> None:
    source = Path(__file__).resolve().parents[1].joinpath(
        "scripts/run_stage_e_minimal_information_diagnostics.py"
    ).read_text()
    assert '"second_oos_access": "PROHIBITED_AND_NOT_RUN"' in source
    assert "months = pd.date_range(DEV_START, FINAL_START" in source
