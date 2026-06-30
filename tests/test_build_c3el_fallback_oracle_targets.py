from pathlib import Path

import pandas as pd

from scripts.build_c3el_fallback_oracle_targets import build_targets


def test_build_targets_excludes_existing_labels_and_caps_by_day(tmp_path: Path) -> None:
    ts = pd.Timestamp("2026-06-15 00:00:00", tz="UTC")
    strategy = "short_asset_alpha"
    scores = tmp_path / "scores.csv"
    features = tmp_path / "features.parquet"
    existing = tmp_path / "existing.csv"
    out_dir = tmp_path / "out"

    pd.DataFrame(
        {
            "timestamp": [ts + pd.Timedelta(hours=i) for i in range(5)],
            "strategy_id": [strategy] * 5,
            "head": ["short_asset"] * 5,
            "action_family": ["size"] * 5,
            "action_value": [0.0] * 5,
            "p_intervene": [0.95, 0.94, 0.93, 0.20, 0.92],
            "pred_action_delta_J": [500.0, 490.0, 480.0, 1000.0, 470.0],
            "selected_multiplier": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    ).to_csv(scores, index=False)
    pd.DataFrame(
        {
            "timestamp": [ts + pd.Timedelta(hours=i) for i in range(5)],
            "strategy_id": [strategy] * 5,
            "multiplier": [0.0] * 5,
            "cooldown_hours_mean": [20.0] * 5,
        }
    ).to_parquet(features, index=False)
    pd.DataFrame(
        {
            "timestamp": [ts + pd.Timedelta(hours=1)],
            "strategy_id": [strategy],
            "action_family": ["size"],
            "action_value": [0.0],
        }
    ).to_csv(existing, index=False)

    build_targets(
        scores_path=scores,
        action_features_path=features,
        existing_labels_path=existing,
        out_dir=out_dir,
        head="short_asset",
        action_value=0.0,
        min_p_intervene=0.80,
        min_pred_delta_j=320.0,
        max_selected_multiplier=0.5,
        quality_rule=None,
        max_targets=10,
        max_per_day=2,
    )

    targets = pd.read_csv(out_dir / "target_actions.csv")
    assert len(targets) == 2
    assert set(targets["timestamp"]) == {
        str(ts),
        str(ts + pd.Timedelta(hours=2)),
    }
    assert set(targets["action_family"]) == {"size"}
    assert set(targets["action_value"]) == {0.0}


def test_build_targets_applies_quality_rule(tmp_path: Path) -> None:
    ts = pd.Timestamp("2026-06-15 00:00:00", tz="UTC")
    scores = tmp_path / "scores.csv"
    features = tmp_path / "features.parquet"
    out_dir = tmp_path / "out"

    pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1)],
            "strategy_id": ["short_asset_alpha", "short_asset_alpha"],
            "head": ["short_asset", "short_asset"],
            "p_intervene": [0.9, 0.9],
            "pred_action_delta_J": [400.0, 390.0],
            "selected_multiplier": [0.0, 0.0],
        }
    ).to_csv(scores, index=False)
    pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1)],
            "strategy_id": ["short_asset_alpha", "short_asset_alpha"],
            "multiplier": [0.0, 0.0],
            "cooldown_hours_mean": [10.0, 15.0],
        }
    ).to_parquet(features, index=False)

    build_targets(
        scores_path=scores,
        action_features_path=features,
        existing_labels_path=None,
        out_dir=out_dir,
        head="short_asset",
        action_value=0.0,
        min_p_intervene=0.80,
        min_pred_delta_j=320.0,
        max_selected_multiplier=0.5,
        quality_rule="cooldown_hours_mean >= 12.0",
        max_targets=10,
        max_per_day=10,
    )

    targets = pd.read_csv(out_dir / "target_actions.csv")
    assert len(targets) == 1
    assert targets.iloc[0]["timestamp"] == str(ts + pd.Timedelta(hours=1))
    assert targets.iloc[0]["quality_rule"] == "cooldown_hours_mean >= 12.0"
