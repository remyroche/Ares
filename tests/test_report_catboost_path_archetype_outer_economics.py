from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "report_catboost_path_archetype_outer_economics.py"
)
_SPEC = importlib.util.spec_from_file_location("outer_economic_report", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_outer_economic_report_reconstructs_fold_train_rows(tmp_path: Path) -> None:
    rows = 40
    timestamps = pd.date_range("2026-04-30T05:00:00Z", periods=rows, freq="h")
    labels = np.tile(
        ["fast_clean_winner", "dead_timeout"],
        rows // 2,
    )
    candidate_ids = [f"candidate-{index}" for index in range(rows)]
    frame = pd.DataFrame(
        {
            "candidate_id": candidate_ids,
            "__ts__": timestamps,
            "__label_end_ts__": timestamps + pd.Timedelta(minutes=30),
            "side_name": "long",
            "path_arch_complete_24h": 1,
            "path_shape_archetype": labels,
            "path_arch_final_return_net_1pct": np.where(
                labels == "fast_clean_winner", 0.02, -0.02
            ),
            "path_arch_peak_mfe_atr": np.where(labels == "fast_clean_winner", 2.0, 0.2),
            "path_arch_mae_12h_r": np.where(labels == "fast_clean_winner", 0.2, 2.0),
            "path_arch_mae_before_meaningful_mfe_r": np.where(
                labels == "fast_clean_winner", 0.1, 1.5
            ),
            "path_arch_stop_before_meaningful_mfe": (labels == "dead_timeout").astype(
                float
            ),
            "path_arch_reaches_meaningful_mfe": (labels == "fast_clean_winner").astype(
                float
            ),
            "path_arch_time_to_first_meaningful_mfe_h": np.where(
                labels == "fast_clean_winner", 2.0, np.nan
            ),
            "path_arch_peak_retention_ratio": np.where(
                labels == "fast_clean_winner", 0.8, 0.1
            ),
            "path_arch_time_to_trailing_h": np.where(
                labels == "fast_clean_winner", 4.0, np.nan
            ),
            "path_arch_mfe_to_activation_distance": np.where(
                labels == "fast_clean_winner", 2.0, 0.2
            ),
        }
    )
    labels_path = tmp_path / "labels.parquet"
    frame.to_parquet(labels_path, index=False)
    validation_start = timestamps[20]
    oof = (
        frame.iloc[20:]
        .loc[:, ["candidate_id", "__ts__", "__label_end_ts__", "side_name"]]
        .copy()
    )
    oof["oof_fold_id"] = 0
    oof["validation_start"] = validation_start
    favourable = labels[20:] == "fast_clean_winner"
    oof["probability__fast_realization_winner"] = np.where(favourable, 0.8, 0.2)
    oof["probability__dead_timeout"] = np.where(favourable, 0.2, 0.8)
    oof_path = tmp_path / "oof.parquet"
    oof.to_parquet(oof_path, index=False)
    training_report = {
        "oof_diagnostics": {
            "class_names": ["fast_realization_winner", "dead_timeout"],
            "fold_fit_reports": [
                {"fold_id": 0, "train_rows": 19, "validation_rows": 20}
            ],
        },
        "class_balance": {
            "selection_provenance": {
                "structural_fingerprint": "structural",
                "feature_fingerprint": "feature",
                "geometry_fingerprint": "geometry",
            },
            "selected_arm_oof_guard": {},
        },
    }
    training_report_path = tmp_path / "training_report.json"
    training_report_path.write_text(json.dumps(training_report), encoding="utf-8")
    output_path = tmp_path / "report.json"

    result = _MODULE.run_report(
        oof_path=oof_path,
        labels_path=labels_path,
        training_report_path=training_report_path,
        output_path=output_path,
        embargo_hours=1.0,
    )

    assert result["rows"] == {"complete_side_labels": 40, "outer_oof": 20}
    assert result["folds"] == [{"fold_id": 0, "train_rows": 19, "validation_rows": 20}]
    aggregate = result["scoring"]["per_arm"]["uniform"]["aggregate"]
    assert aggregate["ml"]["logloss"] < 0.3
    assert aggregate["economic"]["top_tail"]["realised_net_ev"] > 0.0
    assert result["claim"].startswith("Every class-outcome prior")
    assert output_path.is_file()
