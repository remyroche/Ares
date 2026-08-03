from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from scripts.run_meaningful_mfe_exact_grid_reset import (
    TRANSFER_SPECS,
    JULY_DIAGNOSTIC_END,
    JULY_START,
    _base_masks,
    _trial_score,
    derive_targets,
    july_grouped_day_folds,
    load_panel,
    stable_top,
)


def test_any_touch_is_atr_normalized_and_distinct_from_clean_first() -> None:
    frame = pd.DataFrame(
        {
            "peak_mfe_atr": [2.0, 1.4, 2.0],
            "oof_entry_atr_fraction": [0.01, 0.01, 0.01],
            "upper_return": [0.02, 0.015, 0.02],
            "favorable_first": [1.0, 0.0, 0.0],
            "execution_net_ev_12h": [0.01, -0.01, 0.02],
            "soft_label": [0.9, 0.2, 0.5],
        }
    )

    result = derive_targets(frame)

    assert result["any_touch"].tolist() == [1, 0, 1]
    assert result["clean_first"].tolist() == [1, 0, 0]
    assert result["positive_net"].tolist() == [1, 0, 1]


def test_clean_first_without_touch_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "peak_mfe_atr": [1.0],
            "oof_entry_atr_fraction": [0.01],
            "upper_return": [0.02],
            "favorable_first": [1.0],
            "execution_net_ev_12h": [0.01],
            "soft_label": [0.9],
        }
    )

    try:
        derive_targets(frame)
    except ValueError as error:
        assert "cannot occur without" in str(error)
    else:
        raise AssertionError("incoherent clean-first row was accepted")


def test_stable_top_is_input_order_invariant_with_full_identity_tiebreak() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": [f"candidate-{index:02d}" for index in range(20)],
            "__ts__": pd.date_range("2026-07-01", periods=20, freq="h", tz="UTC"),
            "__symbol__": ["A"] * 20,
            "side_name": ["long"] * 20,
            "score": np.ones(20),
        }
    )

    first = stable_top(frame.sample(frac=1.0, random_state=1), "score")
    second = stable_top(frame.sample(frac=1.0, random_state=2), "score")

    assert first["candidate_id"].tolist() == ["candidate-00", "candidate-01"]
    assert first["candidate_id"].tolist() == second["candidate_id"].tolist()


def test_july_grouped_day_folds_are_exact_contiguous_partition_with_embargo() -> None:
    timestamps = pd.date_range(
        JULY_START,
        JULY_DIAGNOSTIC_END,
        freq="6h",
        inclusive="left",
    )
    panel = pd.DataFrame({"__ts__": timestamps})

    folds = july_grouped_day_folds(panel)

    assert len(folds) == 5
    validation_positions: list[int] = []
    for _, train, validation, validation_days in folds:
        validation_positions.extend(validation.tolist())
        assert len(validation_days) == 2
        first, second = map(pd.Timestamp, validation_days)
        assert second - first == pd.Timedelta(days=1)
        train_ts = panel.iloc[train]["__ts__"]
        for day_string in validation_days:
            day = pd.Timestamp(day_string, tz="UTC")
            assert not (
                train_ts.ge(day - pd.Timedelta(hours=12))
                & train_ts.lt(day + pd.Timedelta(days=1, hours=12))
            ).any()
    assert sorted(validation_positions) == list(range(len(panel)))


def test_forward_transfer_requires_explicit_decision_time_purge() -> None:
    spec = TRANSFER_SPECS[0]
    panel = pd.DataFrame(
        {
            "__ts__": [
                spec.evaluation_start - pd.Timedelta(hours=13),
                spec.evaluation_start - pd.Timedelta(hours=11),
                spec.evaluation_start,
            ],
            "execution_decision_utc": [
                spec.evaluation_start - pd.Timedelta(hours=13),
                spec.evaluation_start - pd.Timedelta(hours=11),
                spec.evaluation_start,
            ],
            # The second row is deliberately given an impossible early
            # resolution; the independent decision purge must still reject it.
            "label_resolution_utc": [
                spec.evaluation_start - pd.Timedelta(hours=1),
                spec.evaluation_start - pd.Timedelta(hours=1),
                spec.evaluation_start + pd.Timedelta(hours=12),
            ],
        }
    )

    train, evaluation = _base_masks(panel, spec)

    assert train.tolist() == [0]
    assert evaluation.tolist() == [2]


def test_conditional_head_objective_can_exclude_uncomposed_economics() -> None:
    metrics = {
        "auc": 0.7,
        "pr_auc": 0.5,
        "prevalence": 0.4,
        "brier": 0.2,
        "log_loss": 0.6,
    }

    assert _trial_score(metrics, None) == _trial_score(metrics, None)
    assert _trial_score(metrics, None) != _trial_score(metrics, 500.0)


def test_signed_panel_proves_grid_economics_anchor_and_lineage() -> None:
    feature_path = Path(
        "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/"
        "capture_feature_universe.parquet"
    )
    feature_manifest = feature_path.with_name("manifest.json")
    grid_path = Path(
        "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/"
        "meaningful_mfe_label_grid.parquet"
    )
    grid_manifest = grid_path.with_name("manifest.json")
    panel, matrix, features, lineage = load_panel(
        feature_path,
        feature_manifest,
        grid_path,
        grid_manifest,
    )

    assert len(panel) == len(matrix) == 134_889
    assert len(features) == lineage["raw_feature_count"] == 249
    assert lineage["features"]["sha256"].startswith("2c360b70")
    assert (
        panel["label_resolution_utc"]
        == panel["execution_decision_utc"] + pd.Timedelta(hours=12)
    ).all()

    u2_panel, _, _, u2_lineage = load_panel(
        feature_path,
        feature_manifest,
        grid_path,
        grid_manifest,
        grid_name="h12_u2p0atr",
    )
    assert u2_lineage["labels"]["grid_name"] == "h12_u2p0atr"
    assert u2_panel["any_touch"].mean() < panel["any_touch"].mean()
