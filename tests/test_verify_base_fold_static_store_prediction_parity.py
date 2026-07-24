from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_materialized_trailing_label_topk_lgbm_hpo as runner
import scripts.verify_base_fold_static_store_prediction_parity as verifier
from scripts.verify_base_fold_static_store_prediction_parity import (
    _load_training_imputation,
    _predict_serialized_lgbm,
    _required_input_complete,
    verify_base_fold_prediction_parity,
)


class _Booster:
    def __init__(self) -> None:
        self.num_iteration = "unset"

    def predict(self, values: np.ndarray, *, num_iteration: int | None = None) -> np.ndarray:
        self.num_iteration = num_iteration
        return values[:, 0] + values[:, 1]


class _SklearnWrapper:
    def __init__(self) -> None:
        self.booster_ = _Booster()
        self.best_iteration_ = 7

    def predict(self, values: np.ndarray) -> np.ndarray:
        raise AssertionError("native booster should be used")


class _ArtifactModel:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.zeros(len(values), dtype=np.float32)


def test_predict_serialized_lgbm_uses_native_booster() -> None:
    model = _SklearnWrapper()
    values = np.asarray([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32)

    actual = _predict_serialized_lgbm(model, values)

    np.testing.assert_allclose(actual, np.asarray([3.0, 12.0]))
    assert model.booster_.num_iteration == 7


def test_train_median_imputation_is_required_but_does_not_make_sparse_rows_eligible(
    tmp_path: Path,
) -> None:
    fold_dir = tmp_path / "2025-07"
    fold_dir.mkdir()
    payload = {
        "schema": "s60_base_train_median_imputation_v1",
        "strategy": "per_feature_train_median_then_zero_if_all_missing",
        "feature_names": ["sparse", "dense"],
        "feature_order_hash": verifier._feature_order_hash(["sparse", "dense"]),
        "fill_values": [12.3456, -2.0],
    }
    payload["imputation_contract_hash"] = verifier._imputation_contract_hash(payload)
    artifact = fold_dir / "imputation.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    (fold_dir / "manifest.json").write_text(
        json.dumps(
            {
                "imputation_path": str(artifact),
                "imputation_sha256": verifier._sha256_file(artifact),
                "imputation_contract_hash": payload["imputation_contract_hash"],
            }
        ),
        encoding="utf-8",
    )

    fills = _load_training_imputation(fold_dir, ["sparse", "dense"])
    np.testing.assert_allclose(fills, np.asarray([12.3456, -2.0], dtype=np.float32))
    eligible = _required_input_complete(
        np.asarray([[np.nan, 5.0], [np.inf, -3.0], [12.3456, 2.0]], dtype=np.float32)
    )

    assert eligible.tolist() == [False, False, True]


def test_train_median_imputation_rejects_missing_artifact(tmp_path: Path) -> None:
    fold_dir = tmp_path / "2025-07"
    fold_dir.mkdir()
    (fold_dir / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required train-median"):
        _load_training_imputation(fold_dir, ["sparse"])


def test_saved_fold_model_persists_ordered_train_median_artifact(tmp_path: Path) -> None:
    x_train = np.asarray(
        [[1.0, np.nan, 7.0], [3.0, np.nan, 9.0], [5.0, np.nan, 11.0]],
        dtype=np.float32,
    )
    result = runner._save_base_fold_model(
        model_dir=tmp_path / "models",
        fold={"fold": "2025-07", "month": "2025-07"},
        model=_ArtifactModel(),
        feature_names=["sparse", "all_missing", "dense"],
        x_train=pd.DataFrame(
            x_train, columns=["sparse", "all_missing", "dense"]
        ),
        imputation_fill_values={"sparse": 2.0, "all_missing": 0.0, "dense": 8.0},
        params={"target_mode": "target_soft", "weight_arm": "W0_base"},
        trial_number=1,
        seed=7,
        train_rows_available=3,
        train_rows_fit=3,
        valid_rows=2,
    )

    artifact = Path(result["imputation_path"])
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["feature_names"] == ["sparse", "all_missing", "dense"]
    assert payload["fill_values"] == [2.0, 0.0, 8.0]
    assert payload["fit_scope"] == "fold_train_rows_before_fit_cap"
    assert payload["feature_order_hash"] == runner._feature_contract_hash(
        payload["feature_names"]
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["imputation_sha256"] == runner._sha256_file(artifact)
    assert manifest["imputation_contract_hash"] == payload["imputation_contract_hash"]


def test_verifier_reports_historically_scored_incomplete_rows_as_contract_violations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_dir = tmp_path / "report"
    saved = runner._save_base_fold_model(
        model_dir=report_dir / "models",
        fold={"fold": "2025-07", "month": "2025-07"},
        model=_ArtifactModel(),
        feature_names=["sparse"],
        x_train=pd.DataFrame({"sparse": [1.0, 2.0, 3.0]}),
        imputation_fill_values={"sparse": 2.0},
        params={"target_mode": "target_soft", "weight_arm": "W0_base"},
        trial_number=1,
        seed=7,
        train_rows_available=3,
        train_rows_fit=3,
        valid_rows=2,
    )
    assert Path(saved["model_dir"]).name == "2025-07"
    keys = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2025-07-19T00:00:00Z", "2025-07-19T01:00:00Z"], utc=True
            ),
            "__symbol__": ["SPK", "SPK"],
            "side": [1, 1],
            "expected_score": [0.0, 0.0],
            "oos_fold": ["2025-07", "2025-07"],
        }
    )
    monkeypatch.setattr(verifier, "_sample_oos_rows", lambda *_args, **_kwargs: keys)
    monkeypatch.setattr(verifier, "_sidecar_output_features", lambda *_args: ([], {}))
    monkeypatch.setattr(
        verifier,
        "_load_feature_store_columns",
        lambda *_args, **_kwargs: (
            pd.DataFrame({"sparse": [1.0, np.nan]}, dtype=np.float32),
            "test_loader",
        ),
    )
    monkeypatch.setattr(
        verifier,
        "_read_sidecar_sample",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )

    result = verify_base_fold_prediction_parity(
        report_dir=report_dir,
        feature_store=tmp_path / "feature-store",
        sidecar=tmp_path / "sidecar.parquet",
    )

    assert result["eligible_complete_rows"] == 1
    assert result["historically_scored_incomplete_rows"] == 1
    assert result["pass"] is False
    assert result["row_details"]["verification_status"].tolist() == [
        "eligible_complete",
        "historically_scored_incomplete_contract_violation",
    ]
    assert np.isnan(result["row_details"].loc[1, "replayed_score"])
