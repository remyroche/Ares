from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.replay_market_state_priority_capacity_reallocation import (
    _read_cap_sweep_inputs,
    apply_capacity_reallocation,
)


def test_apply_capacity_reallocation_shifts_caps_by_signed_priority_adjustment() -> None:
    candidates = pd.DataFrame(
        {
            "portfolio_priority_adjustment": [0.10, -0.20, 0.0],
            "head": ["short_boll", "short_asset", "short_boll"],
        }
    )

    out = apply_capacity_reallocation(
        candidates,
        base_strategy_bar_cap=2,
        base_strategy_concurrent_cap=6,
        bar_uplift=1,
        concurrent_uplift=2,
        reduce_disfavored=True,
    )

    assert out["portfolio_max_new_entries_per_strategy_per_bar"].tolist()[:2] == [3.0, 1.0]
    assert np.isnan(out["portfolio_max_new_entries_per_strategy_per_bar"].iloc[2])
    assert out["portfolio_max_concurrent_per_strategy"].tolist()[:2] == [8.0, 4.0]
    assert np.isnan(out["portfolio_max_concurrent_per_strategy"].iloc[2])


def test_apply_capacity_reallocation_can_leave_disfavored_caps_unchanged() -> None:
    candidates = pd.DataFrame({"portfolio_priority_adjustment": [0.10, -0.20]})

    out = apply_capacity_reallocation(
        candidates,
        base_strategy_bar_cap=2,
        base_strategy_concurrent_cap=6,
        bar_uplift=1,
        concurrent_uplift=1,
        reduce_disfavored=False,
    )

    assert out["portfolio_max_new_entries_per_strategy_per_bar"].tolist()[0] == 3.0
    assert np.isnan(out["portfolio_max_new_entries_per_strategy_per_bar"].iloc[1])
    assert out["portfolio_max_concurrent_per_strategy"].tolist()[0] == 7.0
    assert np.isnan(out["portfolio_max_concurrent_per_strategy"].iloc[1])


def test_apply_capacity_reallocation_requires_priority_adjustment() -> None:
    with pytest.raises(ValueError, match="portfolio_priority_adjustment"):
        apply_capacity_reallocation(
            pd.DataFrame({"x": [1]}),
            base_strategy_bar_cap=2,
            base_strategy_concurrent_cap=6,
            bar_uplift=1,
            concurrent_uplift=1,
        )


def test_read_cap_sweep_inputs_can_use_selected_challenger(tmp_path: Path) -> None:
    cap_dir = tmp_path / "cap"
    cap_dir.mkdir()
    default_arm = "L1_lgbm_learned_priority_cap_0p10_zge_0p5"
    selected_arm = "L1_lgbm_learned_priority_cap_0p15_zge_0p5"
    train_path = tmp_path / "train.parquet"
    policy_manifest_path = tmp_path / "policy_manifest.json"
    pd.DataFrame({"timestamp": [pd.Timestamp("2026-06-20T00:00:00Z")]}).to_parquet(
        train_path,
        index=False,
    )
    policy_manifest_path.write_text(json.dumps({"run_id": "test"}), encoding="utf-8")
    (cap_dir / "manifest.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "train_deployable_candidates": str(train_path),
                    "policy_manifest": str(policy_manifest_path),
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"arm": [default_arm, selected_arm]}).to_csv(
        cap_dir / "head_priority_cap_sweep_metrics.csv",
        index=False,
    )
    (cap_dir / "selected_shadow_challenger.json").write_text(
        json.dumps({"selected": True, "arm": selected_arm}),
        encoding="utf-8",
    )
    for arm in [default_arm, selected_arm]:
        pd.DataFrame({"timestamp": [pd.Timestamp("2026-06-20T00:00:00Z")]}).to_parquet(
            cap_dir / f"{arm}_candidates.parquet",
            index=False,
        )

    inputs = _read_cap_sweep_inputs(
        cap_dir,
        "cap_0p10",
        use_selected_challenger=True,
    )

    assert inputs["arm"] == selected_arm
    assert inputs["resolved_arm_contains"] == "cap_0p15_zge_0p5"
    assert inputs["arm_selector_source"].startswith("selected_shadow_challenger:")
