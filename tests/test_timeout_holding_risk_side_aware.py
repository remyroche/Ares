from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_quality_label_walkforward_ablation import _load_joined_frame
from scripts.run_timeout_holding_risk_label_diagnostic import (
    TargetSpec,
    _build_target,
    _selected_rows,
)


def test_side_aware_join_preserves_long_short_label_rows(tmp_path: Path) -> None:
    ts = pd.Timestamp("2026-04-01 12:00:00", tz="UTC")
    quality = pd.DataFrame(
        {
            "__ts__": [ts],
            "__symbol__": ["BTC"],
            "primary_source_tag": ["risk_adjusted_capture_candidate"],
            "trend_path_score": [0.7],
        }
    )
    labels = pd.DataFrame(
        {
            "__ts__": [ts, ts],
            "__symbol__": ["BTC", "BTC"],
            "side": [1, -1],
            "side_name": ["long", "short"],
            "timeframe": ["1h", "1h"],
            "candidate_id": [
                "BTC|2026-04-01T12:00:00Z|1h|long",
                "BTC|2026-04-01T12:00:00Z|1h|short",
            ],
            "__u_policy_net__": [0.01, -0.02],
            "__is_timeout__": [0.0, 1.0],
            "__bars_policy__": [3.0, 24.0],
            "__barrier_pct__": [0.01, 0.01],
            "__mae_ret__": [0.002, 0.012],
            "__mfe_ret__": [0.010, 0.001],
            "__y_ret__": [0.011, -0.013],
        }
    )
    quality_path = tmp_path / "quality.parquet"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    quality.to_parquet(quality_path, index=False)
    labels.to_parquet(labels_dir / "part.parquet", index=False)

    joined, report = _load_joined_frame(quality_labels_path=quality_path, labels_path=labels_dir)

    assert len(joined) == 2
    assert report["join_mode"] == "timestamp_symbol_broadcast_label_side"
    assert report["merge_validate"] == "one_to_many"
    assert report["label_duplicate_timestamp_symbol_rows"] == 1
    assert set(joined["side"].astype(int)) == {-1, 1}
    assert set(joined["candidate_id"].astype(str)) == set(labels["candidate_id"])


def test_timeout_low_progress_target_uses_realized_outcomes_only() -> None:
    metrics = pd.DataFrame(
        {
            "bars_policy": [2.0, 8.0, 24.0, 4.0],
            "bars_to_mfe": [1.0, 8.0, 24.0, 2.0],
            "is_timeout": [False, False, True, False],
            "u_policy_net": [0.02, -0.01, -0.03, 0.01],
            "mfe_norm": [1.2, 0.2, 0.1, 0.8],
        }
    )
    target, weights, report = _build_target(
        metrics=metrics,
        train_mask=pd.Series([True, True, True, False]),
        spec=TargetSpec("timeout_or_low_progress_v1", "timeout_or_low_progress"),
    )

    assert target["target_hard"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert target.loc[2, "target_bucket"] == "timeout"
    assert target.loc[1, "target_bucket"] == "low_progress"
    assert float(weights.min()) > 0.0
    assert report["label"] == "timeout_or_low_progress_v1"


def test_selected_rows_preserve_side_contract_columns() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-05-01", periods=3, freq="h", tz="UTC"),
            "__symbol__": ["BTC", "ETH", "BTC"],
            "side": [1, -1, 1],
            "side_name": ["long", "short", "long"],
            "timeframe": ["1h", "1h", "1h"],
            "candidate_id": ["btc-long", "eth-short", "btc-long-2"],
            "primary_source_tag": ["a", "b", "a"],
        }
    )
    metrics = pd.DataFrame(
        {
            "u_policy_net": [0.01, -0.01, 0.02],
            "mae_norm": [0.2, 1.4, 0.3],
            "barrier": [0.01, 0.03, 0.02],
            "is_timeout": [False, True, False],
            "bars_policy": [3.0, 24.0, 4.0],
            "side": [1, -1, 1],
        }
    )
    target = pd.DataFrame(
        {
            "target_soft": [0.1, 1.0, 0.2],
            "target_hard": [0.0, 1.0, 0.0],
            "target_bucket": ["ok", "timeout", "ok"],
        }
    )

    selected = _selected_rows(
        frame=frame,
        metrics=metrics,
        target=target,
        score=pd.Series([0.1, 0.9, 0.2]),
        selected_idx=np.array([1, 2]),
        context={"period": "2026-05", "label": "timeout_event_v1"},
    )

    for col in ("side", "side_name", "timeframe", "candidate_id"):
        assert col in selected.columns
    assert selected["side"].astype(int).tolist() == [-1, 1]
    assert selected["side_name"].tolist() == ["short", "long"]
