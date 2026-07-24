from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_first_touch_capture_labels import _parse_side_arm_specs
from scripts.materialize_archetype_conditioned_trailing_labels import (
    _copy_capture_columns,
)


def test_parse_side_arm_specs_supports_side_specific_geometries() -> None:
    arms = _parse_side_arm_specs(
        "long:ft_long:0.75:0.5:16:0.05;short:ft_short:1.0:0.5:16:0.03:0.4"
    )

    assert set(arms) == {"long", "short"}
    assert arms["long"].name == "ft_long"
    assert arms["long"].tp_r == pytest.approx(0.75)
    assert arms["long"].sl_r == pytest.approx(0.5)
    assert arms["long"].max_bars_to_mfe == pytest.approx(16.0)
    assert arms["long"].max_barrier == pytest.approx(0.05)
    assert arms["long"].trail_r == pytest.approx(0.5)
    assert arms["short"].name == "ft_short"
    assert arms["short"].tp_r == pytest.approx(1.0)
    assert arms["short"].trail_r == pytest.approx(0.4)


def test_parse_side_arm_specs_rejects_unknown_side() -> None:
    with pytest.raises(ValueError, match="Invalid side"):
        _parse_side_arm_specs("flat:ft_flat:0.75:0.5:16:0.05")


def test_causal_materializer_overwrites_legacy_path_support_aliases() -> None:
    out = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01 00:00:00"], utc=True),
            "__symbol__": ["BTC/USD:USD"],
            "__barrier_pct__": [0.02],
            "__mfe_ret__": [0.90],
            "__mae_ret__": [0.00],
            "__bars_to_mfe__": [5.0],
            "__quality__": [0.01],
            "__w__": [9.0],
        }
    )
    capture = pd.DataFrame(
        {
            "capture_net": [0.01],
            "capture_hit": [1.0],
            "capture_stop": [0.0],
            "capture_timeout": [0.0],
            "capture_eligible": [1.0],
            "capture_valid_path": [1.0],
            "target_soft": [0.75],
            "effective_tp_abs": [0.02],
            "effective_sl_abs": [0.03],
            "first_touch_bar": [3.0],
            "same_bar_both_hit": [0.0],
            "full_path_mfe_norm": [1.50],
            "full_path_mae_norm": [0.25],
            "bars_to_mfe_1r": [2.0],
            "bars_to_mae_1r": [np.nan],
        }
    )

    corrected, _ = _copy_capture_columns(
        out=out,
        capture=capture,
        side="long",
        timeframe="1h",
        round_trip_cost=0.01,
        policy_label_center=0.0,
        policy_label_temperature=0.01,
    )

    assert corrected.loc[0, "__source__mfe_ret__"] == pytest.approx(0.90)
    assert corrected.loc[0, "__mfe_ret__"] == pytest.approx(0.03)
    assert corrected.loc[0, "__mae_ret__"] == pytest.approx(0.005)
    assert corrected.loc[0, "__bars_to_mfe__"] == pytest.approx(2.0)
    assert corrected.loc[0, "__quality__"] == pytest.approx(0.75)
    assert corrected.loc[0, "__w__"] == pytest.approx(1.0)
