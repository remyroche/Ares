import pandas as pd
import pytest

from scripts.build_size_action_exact_panel_from_oracle import build_exact_panel


def test_build_exact_panel_joins_features_to_size_labels(tmp_path) -> None:
    features = pd.DataFrame(
        {
            "timestamp": ["2026-06-26T08:00:00+00:00", "2026-06-26T08:00:00+00:00"],
            "strategy_id": ["short_asset_a", "short_asset_a"],
            "multiplier": [0.0, 1.0],
            "affected_notional": [100.0, 100.0],
            "strategy_candidate_count": [3, 3],
            "action_binds": [False, False],
        }
    )
    labels = pd.DataFrame(
        {
            "timestamp": [
                "2026-06-26T08:00:00+00:00",
                "2026-06-26T08:00:00+00:00",
                "2026-06-26T08:00:00+00:00",
            ],
            "strategy_id": ["short_asset_a", "short_asset_a", "short_asset_a"],
            "action_family": ["size", "size", "threshold"],
            "action_value": [0.0, 1.0, 0.0],
            "is_baseline_action": ["False", "True", "True"],
            "action_binds": ["True", "False", "False"],
            "delta_immediate_J": [5.0, 0.0, 0.0],
            "delta_full_J": [10.0, 0.0, 0.0],
            "delta_full_net_pnl": [11.0, 0.0, 0.0],
            "delta_full_cost_pnl": [-1.0, 0.0, 0.0],
            "delta_full_turnover": [-50.0, 0.0, 0.0],
        }
    )
    feature_path = tmp_path / "features.parquet"
    label_path = tmp_path / "labels.csv"
    features.to_parquet(feature_path, index=False)
    labels.to_csv(label_path, index=False)

    panel, audit = build_exact_panel(action_features=feature_path, oracle_labels=label_path)

    assert len(panel) == 2
    assert audit["label_rows_family"] == 2
    assert audit["stale_feature_label_columns_dropped"] == ["action_binds"]
    assert audit["positive_delta_full_rows"] == 1
    first = panel.loc[panel["multiplier"].eq(0.0)].iloc[0]
    assert first["delta_full_J"] == pytest.approx(10.0)
    assert first["delta_full_J_per_notional"] == pytest.approx(0.10)
    assert bool(first["action_binds"])
    second = panel.loc[panel["multiplier"].eq(1.0)].iloc[0]
    assert not bool(second["action_binds"])
    assert bool(second["is_baseline_action"])


def test_build_exact_panel_rejects_unmatched_feature_rows(tmp_path) -> None:
    features = pd.DataFrame(
        {
            "timestamp": ["2026-06-26T08:00:00+00:00"],
            "strategy_id": ["short_asset_a"],
            "multiplier": [0.5],
        }
    )
    labels = pd.DataFrame(
        {
            "timestamp": ["2026-06-26T08:00:00+00:00"],
            "strategy_id": ["short_asset_a"],
            "action_family": ["size"],
            "action_value": [1.0],
            "delta_immediate_J": [0.0],
            "delta_full_J": [0.0],
        }
    )
    feature_path = tmp_path / "features.parquet"
    label_path = tmp_path / "labels.csv"
    features.to_parquet(feature_path, index=False)
    labels.to_csv(label_path, index=False)

    with pytest.raises(ValueError, match="not fully matched"):
        build_exact_panel(action_features=feature_path, oracle_labels=label_path)


def test_build_exact_panel_can_filter_features_to_label_keys(tmp_path) -> None:
    features = pd.DataFrame(
        {
            "timestamp": ["2026-06-26T08:00:00+00:00", "2026-06-26T09:00:00+00:00"],
            "strategy_id": ["short_asset_a", "short_asset_a"],
            "multiplier": [0.0, 0.0],
            "affected_notional": [100.0, 200.0],
        }
    )
    labels = pd.DataFrame(
        {
            "timestamp": ["2026-06-26T08:00:00+00:00"],
            "strategy_id": ["short_asset_a"],
            "action_family": ["size"],
            "action_value": [0.0],
            "delta_immediate_J": [5.0],
            "delta_full_J": [10.0],
        }
    )
    feature_path = tmp_path / "features.parquet"
    label_path = tmp_path / "labels.csv"
    features.to_parquet(feature_path, index=False)
    labels.to_csv(label_path, index=False)

    panel, audit = build_exact_panel(
        action_features=feature_path,
        oracle_labels=label_path,
        feature_key_mode="labels",
    )

    assert len(panel) == 1
    assert audit["feature_rows_before_filter"] == 2
    assert audit["feature_rows"] == 1
    assert audit["feature_key_mode"] == "labels"
    assert panel.iloc[0]["timestamp"] == pd.Timestamp("2026-06-26T08:00:00+00:00")


def test_build_exact_panel_rejects_duplicate_oracle_keys(tmp_path) -> None:
    features = pd.DataFrame(
        {
            "timestamp": ["2026-06-26T08:00:00+00:00"],
            "strategy_id": ["short_asset_a"],
            "multiplier": [0.5],
        }
    )
    labels = pd.DataFrame(
        {
            "timestamp": ["2026-06-26T08:00:00+00:00", "2026-06-26T08:00:00+00:00"],
            "strategy_id": ["short_asset_a", "short_asset_a"],
            "action_family": ["size", "size"],
            "action_value": [0.5, 0.5],
            "delta_immediate_J": [1.0, 2.0],
            "delta_full_J": [1.0, 2.0],
        }
    )
    feature_path = tmp_path / "features.parquet"
    label_path = tmp_path / "labels.csv"
    features.to_parquet(feature_path, index=False)
    labels.to_csv(label_path, index=False)

    with pytest.raises(ValueError, match="duplicate size-action keys"):
        build_exact_panel(action_features=feature_path, oracle_labels=label_path)
