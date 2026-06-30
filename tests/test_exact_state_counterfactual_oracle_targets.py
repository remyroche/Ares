from pathlib import Path

import pandas as pd

from scripts.run_exact_state_counterfactual_oracle import (
    _load_target_actions,
    _target_action_values,
    _target_strategy_ids_for_timestamp,
)


def test_target_actions_limit_strategies_and_actions(tmp_path: Path) -> None:
    ts = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
    path = tmp_path / "targets.csv"
    pd.DataFrame(
        {
            "timestamp": [ts, ts, ts + pd.Timedelta(hours=1)],
            "strategy_id": ["short_asset_a", "short_asset_a", "short_asset_b"],
            "action_family": ["size", "size", "threshold"],
            "action_value": [0.0, 0.5, 0.02],
        }
    ).to_csv(path, index=False)

    targets = _load_target_actions(path)

    assert _target_strategy_ids_for_timestamp(targets, ts, ["fallback"]) == ["short_asset_a"]
    assert _target_strategy_ids_for_timestamp(targets, ts + pd.Timedelta(hours=2), ["fallback"]) == []
    assert _target_action_values(
        targets,
        timestamp=ts,
        strategy_id="short_asset_a",
        action_family="size",
        default=(0.0, 0.5, 0.75, 1.0),
    ) == [0.0, 0.5]
    assert _target_action_values(
        targets,
        timestamp=ts,
        strategy_id="short_asset_a",
        action_family="threshold",
        default=(0.0, 0.02),
    ) == []


def test_target_actions_with_missing_action_value_use_family_defaults(tmp_path: Path) -> None:
    ts = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
    path = tmp_path / "targets.csv"
    pd.DataFrame(
        {
            "timestamp": [ts],
            "strategy_id": ["short_asset_a"],
            "action_family": ["size"],
        }
    ).to_csv(path, index=False)

    targets = _load_target_actions(path)

    assert _target_action_values(
        targets,
        timestamp=ts,
        strategy_id="short_asset_a",
        action_family="size",
        default=(0.0, 0.5, 0.75, 1.0),
    ) == [0.0, 0.5, 0.75, 1.0]
