from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_first_touch_capture_proxy import _first_touch_capture_outcome
from scripts.run_label_widestop_capture_proxy import CaptureArm, _selection_metrics


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"__barrier_pct__": [0.01]})


def test_first_touch_mae_is_truncated_at_tp_decision_bar() -> None:
    arm = CaptureArm("test", tp_r=1.0, sl_r=1.0, max_bars_to_mfe=10.0, max_barrier=0.03)
    paths = (
        np.array([[100.0, 101.0, 100.0]], dtype=float),
        np.array([[101.2, 101.5, 101.0]], dtype=float),
        np.array([[99.5, 95.0, 90.0]], dtype=float),
        np.array([[101.0, 100.0, 100.0]], dtype=float),
    )

    out = _first_touch_capture_outcome(_frame(), paths, arm, side_name="long")

    assert out.loc[0, "capture_hit"] == 1.0
    assert out.loc[0, "capture_stop"] == 0.0
    assert out.loc[0, "first_touch_bar"] == 1.0
    assert out.loc[0, "mae_to_sl"] == pytest.approx(0.5)
    assert out.loc[0, "mfe_to_tp"] == pytest.approx(1.2)
    assert out.loc[0, "full_path_mae_to_sl"] == pytest.approx(10.0)


def test_first_touch_mfe_is_truncated_at_sl_decision_bar() -> None:
    arm = CaptureArm("test", tp_r=1.0, sl_r=1.0, max_bars_to_mfe=10.0, max_barrier=0.03)
    paths = (
        np.array([[100.0, 100.0, 100.0]], dtype=float),
        np.array([[100.2, 105.0, 106.0]], dtype=float),
        np.array([[98.8, 98.0, 97.0]], dtype=float),
        np.array([[99.0, 103.0, 104.0]], dtype=float),
    )

    out = _first_touch_capture_outcome(_frame(), paths, arm, side_name="long")

    assert out.loc[0, "capture_hit"] == 0.0
    assert out.loc[0, "capture_stop"] == 1.0
    assert out.loc[0, "first_touch_bar"] == 1.0
    assert out.loc[0, "mae_to_sl"] == pytest.approx(1.2)
    assert out.loc[0, "mfe_to_tp"] == pytest.approx(0.2)
    assert out.loc[0, "full_path_mfe_to_tp"] == pytest.approx(6.0)


def test_first_touch_short_uses_mirrored_path_geometry() -> None:
    arm = CaptureArm("test", tp_r=1.0, sl_r=1.0, max_bars_to_mfe=10.0, max_barrier=0.03)
    paths = (
        np.array([[100.0, 100.0, 100.0]], dtype=float),
        np.array([[100.2, 100.5, 106.0]], dtype=float),
        np.array([[98.8, 98.0, 97.0]], dtype=float),
        np.array([[99.0, 98.0, 97.0]], dtype=float),
    )

    out = _first_touch_capture_outcome(_frame(), paths, arm, side_name="short")

    assert out.loc[0, "capture_hit"] == 1.0
    assert out.loc[0, "capture_stop"] == 0.0
    assert out.loc[0, "first_touch_bar"] == 1.0
    assert out.loc[0, "mae_to_sl"] == pytest.approx(0.2)
    assert out.loc[0, "mfe_to_tp"] == pytest.approx(1.2)
    assert out.loc[0, "full_path_mae_to_sl"] == pytest.approx(6.0)


def test_path_ordered_soft_target_demotes_adverse_first_hit() -> None:
    arm = CaptureArm("test", tp_r=1.0, sl_r=2.0, max_bars_to_mfe=10.0, max_barrier=0.03)
    frame = pd.DataFrame({"__barrier_pct__": [0.01, 0.01]})
    paths = (
        np.array([[100.0, 101.0, 101.0], [100.0, 99.0, 101.0]], dtype=float),
        np.array([[101.2, 101.4, 101.5], [100.2, 101.2, 101.4]], dtype=float),
        np.array([[99.8, 100.8, 100.9], [98.8, 98.8, 100.8]], dtype=float),
        np.array([[101.0, 101.2, 101.2], [99.0, 101.0, 101.2]], dtype=float),
    )

    out = _first_touch_capture_outcome(frame, paths, arm, side_name="long", round_trip_cost=0.0)

    assert out.loc[0, "capture_hit"] == 1.0
    assert out.loc[1, "capture_hit"] == 1.0
    assert out.loc[0, "mfe_1r_before_mae_1r"] == 1.0
    assert out.loc[1, "mae_1r_before_mfe_1r"] == 1.0
    assert out.loc[0, "target_soft"] > 0.65
    assert out.loc[1, "target_soft"] <= 0.10


def test_trailing_profit_activates_then_exits_on_trailing_stop() -> None:
    arm = CaptureArm("trail", tp_r=1.0, sl_r=1.0, max_bars_to_mfe=10.0, max_barrier=0.03, trail_r=0.50)
    paths = (
        np.array([[100.0, 101.0, 100.8]], dtype=float),
        np.array([[101.2, 101.4, 101.0]], dtype=float),
        np.array([[99.8, 100.6, 100.0]], dtype=float),
        np.array([[101.0, 100.8, 100.2]], dtype=float),
    )

    out = _first_touch_capture_outcome(_frame(), paths, arm, side_name="long", outcome_mode="trailing_profit")

    assert out.loc[0, "trailing_activated"] == 1.0
    assert out.loc[0, "trailing_activation_bar"] == 1.0
    assert out.loc[0, "capture_hit"] == 1.0
    assert out.loc[0, "capture_stop"] == 0.0
    assert out.loc[0, "first_touch_bar"] == 2.0
    assert out.loc[0, "effective_trail_abs"] == pytest.approx(0.005)
    assert out.loc[0, "capture_net"] == pytest.approx(0.004)


def test_selection_metrics_keep_first_touch_and_full_path_mae_separate() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-06-01 00:00:00", "2026-06-01 00:15:00"]),
            "__symbol__": ["BTC", "ETH"],
        }
    )
    metrics = pd.DataFrame(
        {
            "u_policy_net": [0.01, 0.02],
            "ret_net": [0.01, 0.02],
            "mae_norm": [2.0, 3.0],
            "mfe_norm": [4.0, 5.0],
            "bars_to_mfe": [3.0, 4.0],
            "barrier": [0.01, 0.01],
            "is_timeout": [0.0, 0.0],
        }
    )
    target = pd.DataFrame(
        {
            "capture_net": [0.01, 0.02],
            "round_trip_cost": [0.001, 0.001],
            "capture_hit": [1.0, 1.0],
            "capture_stop": [0.0, 0.0],
            "capture_timeout": [0.0, 0.0],
            "capture_eligible": [1.0, 1.0],
            "target_soft": [0.9, 0.8],
            "target_hard": [1.0, 1.0],
            "first_touch_mae_norm": [0.2, 0.3],
            "first_touch_mfe_norm": [1.2, 1.4],
            "mae_to_sl": [0.4, 0.6],
            "full_path_mae_norm": [2.0, 3.0],
            "full_path_mae_to_sl": [4.0, 5.0],
            "effective_sl_abs": [0.005, 0.005],
            "mfe_to_tp": [1.2, 1.4],
            "mfe_1r_before_mae_1r": [1.0, 0.0],
            "mae_1r_before_mfe_1r": [0.0, 1.0],
            "max_adverse_before_mfe_1r": [0.4, 1.8],
            "underwater_bars_before_mfe_1r": [2.0, 12.0],
            "underwater_fraction_before_mfe_1r": [0.2, 0.6],
            "area_underwater_before_mfe_1r": [0.5, 8.0],
        }
    )

    row = _selection_metrics(
        frame=frame,
        metrics=metrics,
        target=target,
        score=pd.Series([0.9, 0.8]),
        arm="test",
        period="2026-06",
        top_frac=1.0,
        selection_mode="global",
    )

    assert row["selected_path_bad_mae_1r_rate"] == pytest.approx(1.0)
    assert row["first_touch_bad_mae_1r_rate"] == pytest.approx(0.0)
    assert row["target_full_path_bad_mae_1r_rate"] == pytest.approx(1.0)
    assert row["bad_mae_1r_rate"] == pytest.approx(row["selected_path_bad_mae_1r_rate"])
    assert row["mfe_1r_before_mae_1r_rate"] == pytest.approx(0.5)
    assert row["mae_1r_before_mfe_1r_rate"] == pytest.approx(0.5)
    assert row["mean_max_adverse_before_mfe_1r"] == pytest.approx(1.1)
    assert row["mean_underwater_bars_before_mfe_1r"] == pytest.approx(7.0)
    assert row["mean_underwater_fraction_before_mfe_1r"] == pytest.approx(0.4)
