from pathlib import Path

import pandas as pd
import pytest

from scripts.build_c3el_rule_oracle_targets import build_targets


def test_rule_targets_exclude_multiple_existing_label_files_and_cap_by_day(tmp_path: Path) -> None:
    ts = pd.Timestamp("2026-06-15 00:00:00", tz="UTC")
    tagged = tmp_path / "tagged.csv"
    existing_a = tmp_path / "existing_a.csv"
    existing_b = tmp_path / "existing_b.csv"
    out_dir = tmp_path / "out"
    rows = []
    for i in range(6):
        rows.append(
            {
                "timestamp": ts + pd.Timedelta(hours=i),
                "strategy_id": "short_asset_alpha",
                "action_family": "size",
                "action_value": 0.0,
                "p_intervene": 0.90 + i * 0.01,
                "pred_action_delta_J": 320.0 + i,
                "monitor_condition_count": i % 4,
                "rule_p80_d320_cooldown_lte_38_5": True,
            }
        )
    pd.DataFrame(rows).to_csv(tagged, index=False)
    pd.DataFrame(
        {
            "timestamp": [ts + pd.Timedelta(hours=5)],
            "strategy_id": ["short_asset_alpha"],
            "action_family": ["size"],
            "action_value": [0.0],
        }
    ).to_csv(existing_a, index=False)
    pd.DataFrame(
        {
            "timestamp": [ts + pd.Timedelta(hours=4)],
            "strategy_id": ["short_asset_alpha"],
            "action_family": ["size"],
            "action_value": [0.0],
        }
    ).to_csv(existing_b, index=False)

    manifest = build_targets(
        tagged_path=tagged,
        existing_label_paths=[existing_a, existing_b],
        out_dir=out_dir,
        max_targets=3,
        max_per_day=2,
    )

    targets = pd.read_csv(out_dir / "target_actions.csv")
    assert manifest["rule_candidate_rows_before_existing_exclusion"] == 6
    assert manifest["candidate_pool_rows"] == 4
    assert manifest["target_rows"] == 2
    assert len(targets) == 2
    assert set(pd.to_datetime(targets["timestamp"], utc=True)) == {
        ts + pd.Timedelta(hours=2),
        ts + pd.Timedelta(hours=3),
    }


def test_rule_targets_missing_rule_column_raises(tmp_path: Path) -> None:
    tagged = tmp_path / "tagged.csv"
    pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-06-15", tz="UTC")],
            "strategy_id": ["short_asset_alpha"],
            "p_intervene": [0.9],
            "pred_action_delta_J": [400.0],
        }
    ).to_csv(tagged, index=False)

    with pytest.raises(ValueError, match="missing requested rule column"):
        build_targets(
            tagged_path=tagged,
            existing_label_paths=[],
            out_dir=tmp_path / "out",
            rule="rule_not_present",
        )
