from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_source_utility_risk_gate_candidate_weekly import (
    _aggregate_weekly as _candidate_aggregate_weekly,
)
from scripts.report_source_utility_risk_gate_candidate_weekly import (
    _weekly_summary as _candidate_weekly_summary,
)
from scripts.run_source_utility_path_risk_dual_head_diagnostic import (
    _selected_frame as _path_risk_selected_frame,
)
from scripts.run_source_utility_path_risk_dual_head_diagnostic import (
    _weekly_summary as _path_risk_weekly_summary,
)
from scripts.run_source_utility_path_timeout_risk_diagnostic import (
    _selected_frame as _path_timeout_selected_frame,
)
from scripts.run_source_utility_path_timeout_risk_diagnostic import (
    _selected_group_stats,
)


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-04-01", periods=4, freq="h", tz="UTC"),
            "__symbol__": ["BTC", "ETH", "BTC", "SOL"],
            "side": [1, -1, 1, -1],
            "side_name": ["long", "short", "long", "short"],
            "timeframe": ["1h", "1h", "1h", "1h"],
            "candidate_id": ["btc-l-1", "eth-s-1", "btc-l-2", "sol-s-1"],
            "primary_source_tag": ["risk", "risk", "compression", "compression"],
        }
    )


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "u_policy_net": [0.02, 0.01, -0.01, 0.03],
            "mae_norm": [0.2, 0.4, 1.2, 0.3],
            "barrier": [0.01, 0.015, 0.03, 0.012],
            "is_timeout": [False, False, True, False],
            "bars_policy": [3.0, 4.0, 24.0, 5.0],
            "side": [1, -1, 1, -1],
        }
    )


def _target(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"target_soft": values, "target_hard": [float(v >= 0.5) for v in values]})


def test_path_risk_selected_frame_and_weekly_summary_preserve_side_contract() -> None:
    frame = _base_frame()
    metrics = _metrics()
    selected = _path_risk_selected_frame(
        frame=frame,
        metrics=metrics,
        utility_target=_target([0.8, 0.7, 0.1, 0.9]),
        risk_target=_target([0.1, 0.2, 0.9, 0.2]),
        utility_pred=pd.Series([0.8, 0.7, 0.1, 0.9]),
        risk_pred=pd.Series([0.1, 0.2, 0.9, 0.2]),
        selected_idx=np.array([0, 1, 3]),
        context={
            "candidate": "c1",
            "period": "2026-04",
            "label": "utility_linear_source_q80_v1",
            "risk_target": "bad_mae_risk_v1",
            "feature_set": "base_plus_source",
            "source_bucket": "all_rows",
            "causal_gate": "no_gate",
            "selection": "utility_minus_risk",
            "top_frac": 0.5,
        },
    )

    for col in ("side", "side_name", "timeframe", "candidate_id"):
        assert col in selected.columns
    assert selected["side"].astype(int).tolist() == [1, -1, -1]

    weekly = _path_risk_weekly_summary(selected)
    assert "side_top_share" in weekly.columns
    assert float(weekly["side_top_share"].iloc[0]) == 2 / 3
    assert float(weekly["short_share"].iloc[0]) == 2 / 3


def test_path_timeout_selected_frame_and_group_stats_preserve_side_contract() -> None:
    frame = _base_frame()
    metrics = _metrics()
    selected = _path_timeout_selected_frame(
        frame=frame,
        metrics=metrics,
        utility_target=_target([0.8, 0.7, 0.1, 0.9]),
        risk_targets={"timeout_risk_v1": _target([0.1, 0.2, 0.9, 0.2])},
        utility_pred=pd.Series([0.8, 0.7, 0.1, 0.9]),
        risk_preds={"timeout_risk_v1": pd.Series([0.1, 0.2, 0.9, 0.2])},
        final_score=pd.Series([0.7, 0.6, -0.8, 0.8]),
        selected_idx=np.array([0, 1, 3]),
        context={
            "candidate": "c2",
            "period": "2026-04",
            "label": "utility_linear_source_q80_v1",
            "risk_heads": "timeout_risk_v1",
            "feature_set": "base_plus_source",
            "source_bucket": "all_rows",
            "causal_gate": "no_gate",
            "selection": "utility_minus_timeout_0p50",
            "top_frac": 0.5,
        },
    )

    for col in ("side", "side_name", "timeframe", "candidate_id"):
        assert col in selected.columns
    stats = _selected_group_stats(selected)
    assert stats["overall_side_top_share"] == 2 / 3


def test_candidate_weekly_summary_reports_side_concentration() -> None:
    selected = _base_frame().iloc[[0, 1, 3]].copy()
    selected.insert(0, "candidate", "weekly_c")
    selected["period"] = "2026-04"
    selected["week_start"] = "2026-03-30"
    selected["label"] = "utility_linear_source_q80_v1"
    selected["feature_set"] = "base_plus_source"
    selected["source_bucket"] = "risk_adjusted_capture_candidate"
    selected["risk_gate"] = "low_barrier_pressure_q50"
    selected["top_frac"] = 0.1
    selected["selection_mode"] = "gate_relative"
    selected["u_policy_net"] = [0.02, 0.01, 0.03]
    selected["mae_norm"] = [0.2, 0.4, 0.3]
    selected["barrier"] = [0.01, 0.015, 0.012]
    selected["is_timeout"] = [False, False, False]

    weekly = _candidate_weekly_summary(selected)
    aggregate = _candidate_aggregate_weekly(weekly)

    assert "side_top_share" in weekly.columns
    assert float(weekly["side_top_share"].iloc[0]) == 2 / 3
    assert "max_side_top_share" in aggregate.columns
    assert float(aggregate["max_side_top_share"].iloc[0]) == 2 / 3
