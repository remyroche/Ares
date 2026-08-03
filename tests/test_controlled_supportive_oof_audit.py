from __future__ import annotations

import json

import pandas as pd

from scripts.run_controlled_supportive_oof_audit import _attach_oof_lineage


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["b0", "b1", "m0", "m1"],
            "oof_fold": ["base_train", "base_train", "meta_train", "meta_train"],
            "fold_order": [0, 0, 1, 1],
            "__ts__": pd.to_datetime([
                "2024-01-01 00:00Z", "2024-01-01 01:00Z",
                "2024-02-01 00:00Z", "2024-02-01 01:00Z",
            ], utc=True),
            "__decision_ts__": pd.to_datetime([
                "2024-01-01 01:00Z", "2024-01-01 02:00Z",
                "2024-02-01 01:00Z", "2024-02-01 02:00Z",
            ], utc=True),
            "__label_available_at__": pd.to_datetime([
                "2024-01-01 13:00Z", "2024-01-01 14:00Z",
                "2024-02-01 13:00Z", "2024-02-01 14:00Z",
            ], utc=True),
        }
    )


def test_support_oof_lineage_is_strict_and_hash_bound() -> None:
    frame = _frame()
    support = frame.iloc[2:].copy()
    for name in ("clean_opportunity", "peak_mfe_atr", "time_to_meaningful_mfe_hours", "mae_before_meaningful_mfe_atr", "future_slope_atr_per_hour"):
        support[f"support_oof__{name}"] = 0.5
    output = _attach_oof_lineage(
        frame,
        support,
        fold_column="oof_fold",
        features_sha256="f" * 64,
        semantic_contract_sha256="s" * 64,
    )
    assert output.is_oof.all()
    assert output.prediction_fold_id.tolist() == ["meta_train", "meta_train"]
    assert (output.prediction_fit_end_ts < output.__decision_ts__).all()
    assert (output.prediction_generated_ts <= output.__decision_ts__).all()
    assert output.semantic_target_contract_sha256.eq("s" * 64).all()
    lineage = json.loads(output.support_head_lineage.iloc[0])
    assert lineage["schema"] == "supportive_head_lineage_v2"
    assert set(lineage["heads"]) == {
        "clean_opportunity", "peak_mfe_atr", "time_to_meaningful_mfe_hours",
        "mae_before_meaningful_mfe_atr", "future_slope_atr_per_hour",
    }
