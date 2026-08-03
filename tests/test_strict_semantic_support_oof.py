from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_strict_semantic_support_oof import _join_labels
from extreme_price_movements import strict_semantic_support_oof as module


def test_semantic_oof_emits_only_scored_folds_and_lineage(monkeypatch) -> None:
    ledger = pd.read_parquet(
        "data_perp/artifacts/controlled_target_supportive_prepared_ledger_20260801_v1/prepared_target_supportive_ledger.parquet",
        columns=["candidate_id", "__decision_ts__", "__label_available_at__", "__ts__", "oof_fold", "fold_order", "AE_reconstruction_error", "atr_percentile"],
    )
    # Keep a small but chronological sample from every protocol fold.
    sample = pd.concat([
        ledger[ledger.oof_fold.eq("base_train")].head(12),
        ledger[ledger.oof_fold.eq("meta_train")].head(12),
        ledger[ledger.oof_fold.eq("meta_oos")].head(12),
    ], ignore_index=True)
    labels = pd.read_parquet(
        "data_perp/artifacts/root_cause_supportive_target_semantics_20260801_v1/supportive_target_semantics.parquet"
    )
    merged = _join_labels(sample, labels)

    def fake_fit(train_x, train_y, test_x, *, kind):
        return np.full(len(test_x), float(np.nanmean(train_y)))

    monkeypatch.setattr(module, "_fit_predict", fake_fit)
    result = module.generate_strict_semantic_oof(
        merged,
        feature_columns=["AE_reconstruction_error", "atr_percentile"],
        fold_column="oof_fold",
        semantic_contract_sha256="a" * 64,
    )
    output = result.predictions
    assert output.oof_fold.ne("base_train").all()
    assert output.is_oof.all()
    assert (output.prediction_fit_end_ts < output.__decision_ts__).all()
    assert (output.prediction_generated_ts <= output.__decision_ts__).all()
    assert output.semantic_target_contract_sha256.eq("a" * 64).all()
    assert any(column.startswith("semantic_oof__") for column in output.columns)
