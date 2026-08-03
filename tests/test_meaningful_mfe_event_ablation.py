from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.meaningful_mfe_event_ablation import (
    TripleBarrierSoftLabel,
    atr_soft_triple_barrier_labels,
    competing_risk_targets,
    event_quality_scores,
    expanding_resolved_month_folds,
    first_21d_admission,
)


def _paths() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__path_auxiliary_atr_fraction__": [0.01, 0.01, 0.01, 0.005],
            "__peak_mfe_atr_clip_8__": [2.0, 1.4, 1.2, 2.5],
            "__mae_before_meaningful_mfe_atr_12h__": [0.2, 1.1, 0.3, 0.2],
            "__time_to_first_meaningful_mfe_hours_12h__": [2.0, 5.0, 12.0, 12.0],
            "__meaningful_mfe_reached_12h__": [1, 1, 0, 0],
            "__path_auxiliary_target_valid__": [1, 1, 1, 1],
        }
    )


def test_soft_triple_barrier_orders_clean_hit_timeout_and_adverse() -> None:
    labels = atr_soft_triple_barrier_labels(_paths())
    assert labels.loc[0, "tb_outcome"] == "favorable_first"
    assert labels.loc[1, "tb_outcome"] == "adverse_first_or_conflict"
    assert labels.loc[2, "tb_outcome"] == "timeout"
    assert labels.loc[0, "tb_soft_label"] > labels.loc[2, "tb_soft_label"]
    assert labels.loc[2, "tb_soft_label"] > labels.loc[1, "tb_soft_label"]
    assert labels.loc[0, "tb_hard_label"] == 1.0
    assert labels.loc[1, "tb_hard_label"] == 0.0


def test_return_floor_is_converted_to_atr_units() -> None:
    labels = atr_soft_triple_barrier_labels(_paths())
    assert labels.loc[0, "tb_upper_atr"] == 1.5
    assert labels.loc[3, "tb_upper_atr"] == 3.0
    assert labels.loc[3, "tb_outcome"] == "timeout"


def test_lower_barrier_geometry_changes_conservative_first_touch() -> None:
    loose = atr_soft_triple_barrier_labels(
        _paths(), TripleBarrierSoftLabel(lower_atr=1.5)
    )
    strict = atr_soft_triple_barrier_labels(
        _paths(), TripleBarrierSoftLabel(lower_atr=0.5)
    )
    assert loose.loc[1, "tb_outcome"] == "favorable_first"
    assert strict.loc[1, "tb_outcome"] == "adverse_first_or_conflict"


def test_expanding_folds_require_resolved_training_labels() -> None:
    timestamps = pd.date_range("2026-04-01", "2026-07-10 23:00", freq="h", tz="UTC")
    resolved = timestamps + pd.Timedelta(hours=12)
    folds = expanding_resolved_month_folds(timestamps, resolved)
    assert [fold["month"] for fold in folds] == [
        "2026-05",
        "2026-06",
        "2026-07",
    ]
    for fold in folds:
        train = np.asarray(fold["train_indices"])
        valid = np.asarray(fold["validation_indices"])
        assert resolved[train].max() < fold["validation_start"]
        assert timestamps[valid].min() == fold["validation_start"]


def test_competing_risks_preserve_ambiguous_adverse_rows() -> None:
    paths = _paths()
    labels = atr_soft_triple_barrier_labels(paths)
    labels["meaningful_mfe_reached"] = paths[
        "__meaningful_mfe_reached_12h__"
    ]
    risks = competing_risk_targets(labels)
    assert risks.loc[0, "risk_class"] == 2
    assert risks.loc[1, "risk_class"] == 1
    assert risks.loc[2, "risk_class"] == 0
    assert risks.loc[1, "order_ambiguous"]
    assert np.isnan(risks.loc[1, "conditional_quality"])
    assert 0.0 <= risks.loc[0, "conditional_quality"] <= 1.0


def test_event_quality_composition_keeps_probability_as_gate() -> None:
    scores = event_quality_scores([0.8, 0.4], [0.5, 1.0], quality_floor=0.25)
    assert np.allclose(scores["probability_x_quality"], [0.5, 0.4])
    assert np.allclose(scores["probability_gated_quality"], [0.625, 0.4])


def test_first_21d_admission_never_scores_calibration_rows() -> None:
    timestamps = pd.date_range("2026-05-01", periods=30, freq="D", tz="UTC")
    score = np.linspace(0.0, 1.0, len(timestamps))
    realized = score - 0.5
    result = first_21d_admission(timestamps, score, realized)
    evaluate = np.asarray(result["evaluation_mask"])
    admitted = np.asarray(result["admitted_mask"])
    assert result["fit_rows"] == 21
    assert not evaluate[:21].any()
    assert evaluate[21:].all()
    assert not admitted[:21].any()
