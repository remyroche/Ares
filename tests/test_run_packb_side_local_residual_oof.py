from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_packb_side_local_residual_oof import (
    ECONOMIC_COLUMN,
    FOLDS,
    WEIGHT_COLUMN,
    _bounded_position_indices,
    _development_split,
    _metrics,
    _promotion_gate,
)


def _metric(
    *,
    objective: float,
    rank_ic: float = 0.10,
    top10_lift: float = 0.002,
    rmse_gain: float = 0.05,
) -> dict[str, float]:
    return {
        "objective": objective,
        "weighted_rank_ic": rank_ic,
        "top10_net_return_lift": top10_lift,
        "relative_rmse_gain": rmse_gain,
    }


def test_development_split_purges_labels_resolving_at_validation_start() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-04-20T00:00:00Z",
                    "2026-04-21T00:00:00Z",
                    "2026-04-22T00:00:00Z",
                ],
                utc=True,
            ),
            "__label_resolution_ts__": pd.to_datetime(
                [
                    "2026-04-21T23:00:00Z",
                    "2026-04-22T00:00:00Z",
                    "2026-04-23T00:00:00Z",
                ],
                utc=True,
            ),
        }
    )
    repeated = pd.concat([frame] * 5000, ignore_index=True)

    train, validation = _development_split(repeated)

    assert train.sum() == 5000
    assert validation.sum() == 5000
    assert not (train & validation).any()


def test_bounded_positions_preserve_original_identity_after_sampler_reset() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["late", "early", "middle"],
            "__ts__": pd.to_datetime(
                [
                    "2026-04-03T00:00:00Z",
                    "2026-04-01T00:00:00Z",
                    "2026-04-02T00:00:00Z",
                ],
                utc=True,
            ),
            "__symbol__": ["C", "A", "B"],
        },
        index=[91, 17, 44],
    )

    positions = _bounded_position_indices(frame, max_rows=2, name="test")

    assert positions.tolist() == [1, 0]
    assert frame.iloc[positions]["candidate_id"].tolist() == ["early", "late"]


def test_metrics_pass_economic_weight_and_net_return_in_correct_order() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-05-01T00:00:00Z"] * 4, utc=True),
            "__symbol__": ["A", "B", "C", "D"],
            ECONOMIC_COLUMN: [0.04, 0.03, -0.02, -0.03],
            WEIGHT_COLUMN: [10.0, 1.0, 1.0, 1.0],
        }
    )

    result = _metrics(np.asarray([4.0, 3.0, 2.0, 1.0]), frame)

    assert result["weighted_rank_ic"] > 0.99
    assert result["top10_mean_net_return"] == 0.04
    assert result["overall_mean_net_return"] == pytest.approx(0.005)


def test_promotion_gate_requires_uplift_stability_and_no_quality_collapse() -> None:
    base = _metric(objective=0.10)
    improving_folds = [
        {
            "base_metrics": _metric(objective=0.10),
            "residual_metrics": _metric(objective=value),
        }
        for value in (0.12, 0.11, 0.09)
    ]

    passed, checks = _promotion_gate(
        base=base,
        residual=_metric(objective=0.12),
        folds=improving_folds,
    )

    assert passed
    assert all(checks.values())

    collapsed = list(improving_folds)
    collapsed[-1] = {
        "base_metrics": _metric(objective=0.10),
        "residual_metrics": _metric(objective=0.05),
    }
    passed, checks = _promotion_gate(
        base=base,
        residual=_metric(objective=0.12),
        folds=collapsed,
    )
    assert not passed
    assert not checks["no_fold_objective_collapse"]


def test_oof_calendar_starts_after_april_development_month() -> None:
    assert [fold[0] for fold in FOLDS] == [
        "residual_1_20260501",
        "residual_2_20260601",
        "residual_3_20260701",
    ]
    assert all(pd.Timestamp(start).month >= 5 for _, start, _ in FOLDS)
