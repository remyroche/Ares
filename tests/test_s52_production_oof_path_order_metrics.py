from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_s52_production_oof_path_order_metrics import build_report


def test_production_oof_report_uses_gross_ev_weighted_clean_precision(tmp_path: Path) -> None:
    data_root = tmp_path / "data_perp"
    artifact_run_id = "base_run"
    label_run_id = "label_run"
    oof_dir = data_root / "artifacts" / artifact_run_id / "oof"
    labels_dir = data_root / "artifacts" / label_run_id / "labels"
    oof_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    timestamps = pd.date_range("2026-06-01", periods=4, freq="h", tz="UTC")
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTC", "ETH", "SOL", "XRP"],
            "oof_prob": [0.9, 0.8, 0.1, 0.0],
        }
    ).to_parquet(oof_dir / "oof_global_short_H5.parquet", index=False)

    labels = pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": ["BTC", "ETH", "SOL", "XRP"],
            "__u_policy_net__": [0.01, -0.03, 0.02, -0.01],
            "__first_touch_capture_net__": [0.01, -0.03, 0.02, -0.01],
            "__first_touch_round_trip_cost__": [0.01, 0.01, 0.01, 0.01],
            "__first_touch_hit__": [1, 0, 1, 0],
            "__first_touch_stop__": [0, 1, 0, 1],
            "__first_touch_timeout__": [0, 0, 0, 0],
            "__first_touch_valid_path__": [1, 1, 1, 1],
            "__first_touch_same_bar_both__": [0, 0, 0, 0],
            "__first_touch_mae_norm__": [0.2, 1.5, 0.1, 2.0],
            "__first_touch_mfe_norm__": [1.2, 0.3, 1.1, 0.2],
            "__mfe_1r_before_mae_1r__": [1, 0, 1, 0],
            "__mae_1r_before_mfe_1r__": [0, 1, 0, 1],
        }
    )
    labels.to_parquet(labels_dir / "train_short_s52_5.parquet", index=False)

    result = build_report(
        artifact_run_id=artifact_run_id,
        label_run_id=label_run_id,
        data_root=data_root,
        output_dir=tmp_path / "report",
        top_fracs=(0.50,),
    )
    row = result["summary"].iloc[0]
    # Top 50% selects BTC and ETH. Raw clean hit precision is 1/2, but gross
    # EV-weighted precision is 0.02 / (0.02 + abs(-0.02)) = 0.50.
    assert row["raw_clean_first_touch_precision"] == 0.5
    assert np.isclose(row["gross_ev_weighted_clean_first_touch_precision"], 0.5)
    assert row["first_touch_bad_mae_1r_rate"] == 0.5
    assert result["manifest"]["status"] == "fail_missing_side_oof"


def test_production_oof_report_falls_back_to_lgbm_reference_predictions(tmp_path: Path) -> None:
    data_root = tmp_path / "data_perp"
    artifact_run_id = "base_run"
    label_run_id = "label_run"
    labels_dir = data_root / "artifacts" / label_run_id / "labels"
    labels_dir.mkdir(parents=True)
    artifact_dir = data_root / "artifacts" / artifact_run_id

    for side in ("long", "short"):
        timestamps = pd.date_range("2026-06-01", periods=2, freq="h", tz="UTC")
        row_dir = artifact_dir / "row_universe"
        row_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "dataset": [f"train_global_{side}_5"] * 2,
                "timestamp": timestamps,
                "symbol": ["BTC", "ETH"],
            }
        ).to_parquet(row_dir / f"train_global_{side}_5.parquet", index=False)
        ref_dir = artifact_dir / "lgbm_reference" / "base" / f"global_{side}" / "train_time_provenance"
        ref_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "row_index": [0, 1],
                "oof_prediction": [0.9, np.nan],
                "oof_raw_margin": [1.0, np.nan],
            }
        ).to_parquet(ref_dir / "predictions.parquet", index=False)
        pd.DataFrame(
            {
                "__ts__": timestamps,
                "__symbol__": ["BTC", "ETH"],
                "__u_policy_net__": [0.01, -0.01],
                "__first_touch_capture_net__": [0.01, -0.01],
                "__first_touch_round_trip_cost__": [0.01, 0.01],
                "__first_touch_hit__": [1, 0],
                "__first_touch_stop__": [0, 1],
                "__first_touch_timeout__": [0, 0],
                "__first_touch_valid_path__": [1, 1],
                "__first_touch_same_bar_both__": [0, 0],
                "__first_touch_mae_norm__": [0.2, 2.0],
                "__mfe_1r_before_mae_1r__": [1, 0],
                "__mae_1r_before_mfe_1r__": [0, 1],
            }
        ).to_parquet(labels_dir / f"train_{side}_s52_5.parquet", index=False)

    result = build_report(
        artifact_run_id=artifact_run_id,
        label_run_id=label_run_id,
        data_root=data_root,
        output_dir=tmp_path / "report",
        top_fracs=(1.0,),
    )

    assert set(result["heads"]["source_kind"]) == {"lgbm_reference_predictions"}
    assert result["heads"]["oof_rows"].tolist() == [1, 1]
    assert set(result["manifest"]["sides_seen"]) == {"long", "short"}
    assert result["summary"]["gross_ev_weighted_clean_first_touch_precision"].tolist() == [1.0, 1.0]
