from pathlib import Path

import pandas as pd

from scripts.run_c3el_exact_accept_gate import _feature_columns, _join_frames, _normalise_action_features, _normalise_labels, run_accept_gate


def test_accept_gate_excludes_outcome_columns() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=8, freq="h", tz="UTC"),
            "strategy_id": ["short_asset_a"] * 8,
            "action_value": [0.0] * 8,
            "deployable_state": list(range(8)),
            "delta_full_J": [100, -10, 80, -5, 90, -20, 70, -1],
            "future_return": [1, 2, 3, 4, 5, 6, 7, 8],
            "p_intervene": [0.1, 0.2, 0.9, 0.8, 0.7, 0.4, 0.3, 0.6],
            "exact_positive_e50": [True, False, True, False, True, False, True, False],
        }
    )

    cols = _feature_columns(frame, max_features=10)

    assert "deployable_state" in cols
    assert "p_intervene" in cols
    assert "future_return" not in cols
    assert "delta_full_J" not in cols


def test_accept_gate_defaults_to_noop_when_training_gate_invalid(tmp_path: Path) -> None:
    ts = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
    labels = tmp_path / "labels.csv"
    features = tmp_path / "features.parquet"
    out_dir = tmp_path / "out"
    rows = []
    feature_rows = []
    for i in range(12):
        t = ts + pd.Timedelta(days=i)
        rows.append(
            {
                "timestamp": t,
                "strategy_id": "short_asset_a",
                "action_family": "size",
                "action_value": 0.0,
                "delta_full_J": -10.0,
                "delta_immediate_J": -5.0,
            }
        )
        feature_rows.append(
            {
                "timestamp": t,
                "strategy_id": "short_asset_a",
                "multiplier": 0.0,
                "deployable_state": float(i),
            }
        )
    pd.DataFrame(rows).to_csv(labels, index=False)
    pd.DataFrame(feature_rows).to_parquet(features, index=False)

    summary = run_accept_gate(
        label_paths=[labels],
        action_features_path=features,
        score_paths=[],
        out_dir=out_dir,
        default_action_value=0.0,
        max_features=4,
        thresholds=[0.5],
        min_keep=2,
        min_precision=0.6,
        seed=1,
    )

    loo = pd.read_csv(out_dir / "leave_one_day_gate_validation.csv")
    assert summary["loo"]["total_keep_count"] == 0
    assert int(loo["keep_count"].sum()) == 0


def test_accept_gate_feature_match_marker_does_not_depend_on_label_multiplier(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    features = tmp_path / "features.parquet"
    ts = pd.Timestamp("2026-06-01", tz="UTC")
    pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1)],
            "strategy_id": ["short_asset_a", "short_asset_a"],
            "action_family": ["size", "size"],
            "action_value": [0.0, 0.0],
            "multiplier": [0.0, None],
            "delta_full_J": [100.0, -50.0],
        }
    ).to_csv(labels, index=False)
    pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1)],
            "strategy_id": ["short_asset_a", "short_asset_a"],
            "multiplier": [0.0, 0.0],
            "deployable_state": [1.0, 2.0],
        }
    ).to_parquet(features, index=False)

    joined = _join_frames(
        _normalise_labels([labels], default_action_value=0.0),
        _normalise_action_features(features, default_action_value=0.0),
        pd.DataFrame(),
    )

    assert joined["feature_row_matched"].sum() == 2
