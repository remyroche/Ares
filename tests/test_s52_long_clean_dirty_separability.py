from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_s52_long_clean_dirty_separability import (  # noqa: E402
    _candidate_labels,
    _safe_auc,
    _top_metrics,
)
from scripts.run_s52_long_clean_aux_overlay import _long_clean_gross_target  # noqa: E402


def test_candidate_labels_separate_long_clean_from_dirty_positive() -> None:
    frame = pd.DataFrame(
        {
            "__first_touch_target_soft__": [0.9, 0.2, 0.9],
            "__first_touch_hit__": [1.0, 1.0, 1.0],
            "__first_touch_stop__": [0.0, 0.0, 0.0],
            "__first_touch_timeout__": [0.0, 0.0, 0.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "side": [1.0, 1.0, -1.0],
            "u_policy_net": [0.01, 0.01, 0.01],
            "first_touch_net": [0.004, 0.004, 0.004],
            "first_touch_available": [1.0, 1.0, 1.0],
            "first_touch_mae_norm": [0.25, 1.25, 0.25],
            "mfe_1r_before_mae_1r": [1.0, 1.0, 1.0],
            "mae_1r_before_mfe_1r": [0.0, 1.0, 0.0],
            "max_adverse_before_mfe_1r": [0.50, 2.00, 0.50],
            "underwater_bars_before_mfe_1r": [4.0, 18.0, 4.0],
            "underwater_fraction_before_mfe_1r": [0.2, 0.6, 0.2],
        }
    )

    labels = _candidate_labels(frame, metrics, round_trip_cost=0.01, side="long")

    assert labels["candidate"].tolist() == [True, True, False]
    assert labels["clean"].tolist() == [True, False, False]
    assert labels["dirty"].tolist() == [False, True, False]


def test_top_metrics_reports_clean_precision_and_dirty_rate() -> None:
    labels = pd.DataFrame(
        {
            "clean": [True, False, True, False],
            "dirty": [False, True, False, True],
            "first_good": [True, True, True, False],
            "first_bad": [False, True, False, True],
            "gross": [0.02, 0.01, 0.02, -0.01],
            "net": [0.01, 0.0, 0.01, -0.02],
        }
    )
    metrics = pd.DataFrame(
        {
            "first_touch_mae_norm": [0.2, 1.2, 0.3, 1.5],
            "mae_1r_before_mfe_1r": [0.0, 1.0, 0.0, 1.0],
            "max_adverse_before_mfe_1r": [0.5, 2.0, 0.6, 2.5],
            "underwater_bars_before_mfe_1r": [3.0, 18.0, 4.0, 20.0],
        }
    )

    out = _top_metrics([0.9, 0.8, 0.2, 0.1], labels, metrics)

    assert out["top10_clean_precision"] == pytest.approx(1.0)
    assert out["top30_dirty_rate"] == pytest.approx(0.5)
    assert _safe_auc([1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.75)


def test_long_clean_gross_target_rewards_clean_high_gross_only() -> None:
    labels = pd.DataFrame(
        {
            "clean": [True, True, False],
            "gross": [0.015, 0.004, 0.020],
        }
    )
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                    "2026-06-01 00:00:00",
                ]
            )
        }
    )

    target = _long_clean_gross_target(labels, frame, round_trip_cost=0.01, mode="clean_gross_rank")
    margin_target = _long_clean_gross_target(labels, frame, round_trip_cost=0.01, mode="clean_gross_margin")

    assert float(target.iloc[0]) > float(target.iloc[1])
    assert float(target.iloc[2]) == pytest.approx(0.0)
    assert float(margin_target.iloc[0]) > float(margin_target.iloc[1])
    assert float(margin_target.iloc[2]) == pytest.approx(0.0)
