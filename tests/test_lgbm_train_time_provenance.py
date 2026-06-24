import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.lgbm_pipeline import (
    LGBMStabilityModel,
    LGBM_META_LEAF_LITE_FEATURE_NAMES,
    LGBM_META_SCORE_PATH_FEATURE_NAMES,
    _save_lgbm_train_time_provenance,
)


class _TinyLeafModel:
    n_estimators_ = 2

    def predict(self, X, pred_leaf=False):
        n = len(X)
        if pred_leaf:
            return np.column_stack(
                [
                    np.arange(n, dtype=np.int32) % 3,
                    (np.arange(n, dtype=np.int32) + 1) % 5,
                ]
            )
        return np.linspace(0.2, 0.8, n, dtype=np.float32)


def _load_frame(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    return pd.read_parquet(path)


def test_lgbm_train_time_provenance_writes_core_artifacts(tmp_path):
    n = 6
    X = pd.DataFrame(
        {
            "feature_a": np.linspace(0.0, 1.0, n, dtype=np.float32),
            "feature_b": np.linspace(1.0, 2.0, n, dtype=np.float32),
        }
    )
    model = LGBMStabilityModel(mode="classifier")
    model.models = [_TinyLeafModel()]
    model.selected_features = list(X.columns)
    model.input_feature_names = list(X.columns)
    model.best_params = {"max_depth": 3, "num_leaves": 7}
    model.oof_probs = np.linspace(0.1, 0.9, n, dtype=np.float32)
    meta = pd.DataFrame(index=np.arange(n))
    for col in LGBM_META_SCORE_PATH_FEATURE_NAMES[:3]:
        meta[col] = np.linspace(0.0, 1.0, n, dtype=np.float32)
    for col in LGBM_META_LEAF_LITE_FEATURE_NAMES[:3]:
        meta[col] = np.linspace(1.0, 2.0, n, dtype=np.float32)
    model.meta_oof_features = meta

    _save_lgbm_train_time_provenance(
        model,
        tmp_path,
        X_selected=X,
        y_metric=np.array([0, 1, 0, 1, 0, 1], dtype=np.float32),
        returns=np.linspace(-0.1, 0.1, n, dtype=np.float32),
        timestamps=pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
        assets=np.array(["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"], dtype=object),
        sample_weight=np.ones(n, dtype=np.float32),
        final_weights=np.linspace(0.8, 1.2, n, dtype=np.float32),
        stage_indices={
            "lgbm_select": np.array([0, 1, 2], dtype=np.int32),
            "hpo": np.array([1, 3, 5], dtype=np.int32),
            "fit_oof": np.arange(n, dtype=np.int32),
        },
        fit_idx=np.arange(n, dtype=np.int32),
        hpo_idx=np.array([1, 3, 5], dtype=np.int32),
        oof_fold_ids=np.array([1, 1, 2, 2, 3, 3], dtype=np.int16),
        objective_mode="train_meta",
        mode="classifier",
        cfg={
            "lgbm_train_provenance_enabled": True,
            "lgbm_train_provenance_save_matrix": True,
            "lgbm_train_provenance_max_matrix_cells": 100,
            "lgbm_train_provenance_save_leaf_ids": True,
            "lgbm_train_provenance_max_leaf_cells": 100,
            "lgbm_train_provenance_save_leaf_hashes": True,
            "lgbm_train_provenance_max_leaf_hash_cells": 100,
        },
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    artifacts = manifest["artifacts"]
    assert manifest["layer"] == "meta"
    assert manifest["row_count"] == n
    assert manifest["selected_features"] == ["feature_a", "feature_b"]
    assert manifest["selected_matrix_saved"] is True
    assert manifest["leaf_ids_saved"] is True
    assert manifest["leaf_hashes_saved"] is True
    assert len(manifest["model_hash"]) == 64

    row_refs = _load_frame(artifacts["row_references"])
    assert row_refs["oof_fold_id"].tolist() == [1, 1, 2, 2, 3, 3]
    assert row_refs["is_stage_hpo"].tolist() == [False, True, False, True, False, True]

    preds = _load_frame(artifacts["predictions"])
    assert {"oof_prediction", "oof_raw_margin", "oof_calibrated_probability"}.issubset(
        preds.columns
    )
    assert Path(artifacts["leaf_ids"]).exists()
    leaf_npz = np.load(artifacts["leaf_ids"])
    assert leaf_npz["leaf_ids"].shape == (n, 2)
