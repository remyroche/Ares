from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

STAGING = Path(__file__).resolve().parents[1] / "scripts"
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from run_regime_transition_active_head_chronological_oos import (  # noqa: E402
    active_operating_curve,
    chronological_month_folds,
    conservative_label_available_utc,
    feature_columns,
)


def _frame() -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-01", "2024-03-31 23:00", freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "source_utc": timestamps,
            "execution_decision_utc": timestamps + pd.Timedelta(hours=1),
            "segment_id": 0,
            "feature": 1.0,
            "target__transition_active": 0,
            "target__available_utc": pd.Series(
                pd.NaT, index=range(len(timestamps)), dtype="datetime64[ns, UTC]"
            ),
        }
    )


def test_conservative_availability_uses_floor_and_declared_maximum() -> None:
    frame = _frame().iloc[:2].copy()
    frame.loc[frame.index[0], "target__available_utc"] = (
        frame.iloc[0]["source_utc"] + pd.Timedelta(hours=25)
    )
    available = conservative_label_available_utc(frame)
    assert available.iloc[0] == frame.iloc[0]["source_utc"] + pd.Timedelta(hours=25)
    assert available.iloc[1] == frame.iloc[1]["source_utc"] + pd.Timedelta(hours=12)


def test_month_fold_is_expanding_and_strictly_label_purged() -> None:
    frame = _frame()
    folds = chronological_month_folds(
        frame,
        first_evaluation_month="2024-01-01",
        last_evaluation_month="2024-03-01",
        minimum_train_months=12,
    )
    assert len(folds) == 3
    previous_train_rows = 0
    available = conservative_label_available_utc(frame)
    for start, train, evaluation in folds:
        assert len(train) > previous_train_rows
        assert available.iloc[train].max() < start
        source = frame.iloc[evaluation]["source_utc"]
        assert source.min() == start
        assert source.max() < start + pd.offsets.MonthBegin(1)
        previous_train_rows = len(train)


def test_feature_contract_excludes_targets_and_identifiers() -> None:
    frame = _frame()
    frame["another_numeric"] = 2.0
    assert feature_columns(frame) == ["feature", "another_numeric"]


def test_active_event_recall_does_not_use_high_approach_score() -> None:
    oos = pd.DataFrame(
        {
            "source_utc": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
            "target__event_id": ["event", "event", None, None],
            "target__transition_active": [0, 1, 0, 0],
            "prediction": [0.9, 0.1, 0.0, 0.0],
        }
    )
    operating = active_operating_curve(oos, (0.5,))
    assert operating.loc[0, "event_count"] == 1
    assert operating.loc[0, "event_recall"] == 0.0
