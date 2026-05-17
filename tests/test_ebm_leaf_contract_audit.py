from __future__ import annotations

import json
import importlib.util
import sys
import types
import importlib.machinery
from pathlib import Path
import pickle
import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.ebm_on_lgbm import (
    EBMOnLGBMModel,
    _predict_raw_ebm,
    iter_ebm_models,
    summarize_ebm_leaf_contract,
)

_optuna = types.ModuleType("optuna")
_optuna.__spec__ = importlib.machinery.ModuleSpec("optuna", loader=None)
_optuna_pruners = types.ModuleType("optuna.pruners")
_optuna_pruners.__spec__ = importlib.machinery.ModuleSpec("optuna.pruners", loader=None)
_optuna_samplers = types.ModuleType("optuna.samplers")
_optuna_samplers.__spec__ = importlib.machinery.ModuleSpec(
    "optuna.samplers", loader=None
)
_optuna_pruners.MedianPruner = object
_optuna_pruners.SuccessiveHalvingPruner = object
_optuna_samplers.TPESampler = object
_optuna.pruners = _optuna_pruners
_optuna.samplers = _optuna_samplers
sys.modules.setdefault("optuna", _optuna)
sys.modules.setdefault("optuna.pruners", _optuna_pruners)
sys.modules.setdefault("optuna.samplers", _optuna_samplers)
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.model_loader import load_meta_models_from_pickle

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "audit_ebm_leaf_contract.py"
)
_SPEC = importlib.util.spec_from_file_location("audit_ebm_leaf_contract", _SCRIPT_PATH)
audit_ebm_leaf_contract = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(audit_ebm_leaf_contract)


def _model_with_leaf_contract() -> EBMOnLGBMModel:
    model = EBMOnLGBMModel(mode="classifier")
    model.raw_selected_features = ["raw_a", "raw_b"]
    model.tree_feature_names = ["lgbm_model0_tree0_leaf0_soft"]
    model.selected_features = ["raw_a", "raw_b", "lgbm_model0_tree0_leaf0_soft"]
    model.tree_feature_scales = np.ones(2, dtype=np.float32)
    return model


def test_iter_ebm_models_finds_direct_and_weak_wrapper_paths() -> None:
    direct = EBMOnLGBMModel()
    wrapped_model = EBMOnLGBMModel()
    WeakResidualMetaRegressor = type("WeakResidualMetaRegressor", (), {})
    weak_wrapper = WeakResidualMetaRegressor()
    weak_wrapper.ebm_model = wrapped_model

    found = list(iter_ebm_models({"direct": direct, "nested": [weak_wrapper]}))

    assert [(path, model) for path, model in found] == [
        ("state['direct']", direct),
        ("state['nested'][0].ebm_model", wrapped_model),
    ]


def test_summarize_ebm_leaf_contract_counts_and_hashes() -> None:
    model = _model_with_leaf_contract()
    summary = summarize_ebm_leaf_contract("state['m']", model)

    assert summary["model_path"] == "state['m']"
    assert summary["selected_features_n"] == 3
    assert summary["raw_selected_features_n"] == 2
    assert summary["tree_feature_names_n"] == 1
    assert summary["selected_lgbm_features_n"] == 1
    assert summary["selected_raw_features_n"] == 2
    assert len(summary["selected_feature_hash"]) == 16


def test_audit_model_detects_raw_zero_fill_and_missing_tree_regeneration(
    tmp_path: Path,
) -> None:
    model = _model_with_leaf_contract()

    row = audit_ebm_leaf_contract.audit_model(
        "state['m']",
        model,
        tmp_path,
        max_rows=3,
    )

    assert row["raw_missing_probe_silent_zero_fill"] is True
    assert row["tree_features_missing_count"] == 1
    assert row["selected_missing_before_zero_fill_count"] == 1
    assert row["leaf_transform_state_missing"] is True
    assert row["status"] == "fail"
    assert "selected_tree_features_not_regenerated" in row["fail_reasons"]


def test_predict_meta_rejects_missing_ebm_raw_contract() -> None:
    model = EBMOnLGBMModel(mode="classifier")
    model.raw_selected_features = ["raw_a", "raw_b"]
    model.selected_features = ["raw_a", "raw_b"]
    orchestrator = ModelOrchestrator({"meta_models": {"mr": model}})

    preds = orchestrator.predict_meta(
        pd.DataFrame({"raw_a": [1.0], "other": [2.0]}), "long", "mr"
    )

    assert preds.empty
    assert (
        orchestrator.get_last_results()["meta_contract_error"]["reason"]
        == "missing_ebm_feature_contract"
    )
    assert orchestrator.get_last_results()["meta_contract_error"][
        "missing_raw_features_sample"
    ] == ["raw_b"]


def test_frame_uses_explicit_fn_mapping_not_live_column_order() -> None:
    model = EBMOnLGBMModel(mode="classifier")
    model.raw_selected_features = ["f0", "f1"]
    model.selected_features = ["f0", "f1"]
    model.positional_feature_mapping = {"f0": "real_a", "f1": "real_b"}

    live = pd.DataFrame(
        {
            "real_b": [10.0, 20.0],
            "unrelated_first": [999.0, 888.0],
            "real_a": [1.0, 2.0],
        }
    )
    shuffled = live[["unrelated_first", "real_a", "real_b"]]

    frame = model._frame(live)
    shuffled_frame = model._frame(shuffled)

    expected = pd.DataFrame({"f0": [1.0, 2.0], "f1": [10.0, 20.0]})
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), expected)
    pd.testing.assert_frame_equal(shuffled_frame.reset_index(drop=True), expected)


def test_predict_uses_raw_oof_contract_batch_invariant_when_calibrator_collapses() -> None:
    class RawProbabilityModel:
        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            p = frame["x"].to_numpy(dtype=np.float64)
            return np.column_stack([1.0 - p, p])

    class ConstantPostProcessor:
        def predict(self, raw: np.ndarray) -> np.ndarray:
            return np.full(len(raw), 0.5, dtype=np.float32)

    model = EBMOnLGBMModel(mode="classifier")
    model.selected_features = ["x"]
    model.models = [RawProbabilityModel()]
    model.postprocessors = [ConstantPostProcessor()]
    model.oof_probs_raw_ebm = np.array([0.1, 0.4, 0.8], dtype=np.float32)
    # Persisted OOF final scores equal raw scores: raw EBM probabilities are the
    # deployment ranking contract, even if the stored postprocessor collapses.
    model.oof_probs = model.oof_probs_raw_ebm.copy()

    batch = pd.DataFrame({"x": [0.2, 0.7]})
    batch_pred = model.predict(batch)
    single_pred = model.predict(batch.iloc[[1]])

    assert np.allclose(batch_pred, np.array([0.2, 0.7], dtype=np.float32))
    assert np.isclose(single_pred[0], batch_pred[1])


def test_predict_raw_uses_fitted_feature_types_not_live_batch_cardinality() -> None:
    class ContinuousContractModel:
        feature_types_in_ = ["continuous", "continuous"]

        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            p = 0.25 + 0.2 * frame["x"].to_numpy(dtype=np.float64)
            p += 0.05 * frame["flag_like_continuous"].to_numpy(dtype=np.float64)
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            return np.column_stack([1.0 - p, p])

    model = ContinuousContractModel()
    batch = pd.DataFrame(
        {
            "x": [0.20, 0.70],
            "flag_like_continuous": [1.0, 0.3],
        }
    )

    batch_pred = _predict_raw_ebm(model, batch, "classifier")
    single_pred = _predict_raw_ebm(model, batch.iloc[[0]], "classifier")

    assert np.isclose(single_pred[0], batch_pred[0])


def test_model_derived_meta_features_materialized_from_base_prediction() -> None:
    model = EBMOnLGBMModel(mode="classifier")
    model.feature_columns = [
        "strategy_score",
        "pred_H5",
        "pred_logit_H5",
        "base_model_margin",
        "recent_hit_rate_24h",
        "base_prob_x_raw_a",
    ]
    model.raw_selected_features = list(model.feature_columns)
    model.selected_features = list(model.feature_columns)
    orchestrator = ModelOrchestrator({"meta_models": {"long_demo": model}})
    features = pd.DataFrame({"strategy_score": [0.8], "raw_a": [2.0]})

    materialized = orchestrator._materialize_meta_model_derived_features(
        features,
        model,
        side="long",
        kind="strategy_score",
    )

    assert materialized["pred_H5"].iloc[0] == 0.8
    assert np.isclose(materialized["pred_logit_H5"].iloc[0], np.log(0.8 / 0.2))
    assert np.isclose(materialized["base_model_margin"].iloc[0], 0.3)
    assert materialized["recent_hit_rate_24h"].iloc[0] == 0.0
    assert materialized["base_prob_x_raw_a"].iloc[0] == 1.6


def test_model_state_meta_loader_attaches_feature_contract(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "run_a"
    models_dir = run_dir / "models"
    meta_dir = run_dir / "meta_oof"
    models_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    trained_state_path = models_dir / "trained_state.pkl"
    with trained_state_path.open("wb") as f:
        pickle.dump({"bundle": {"meta_models": {}}}, f)

    model = EBMOnLGBMModel(mode="classifier")
    model.raw_selected_features = ["f0", "f1"]
    model.selected_features = ["f0", "f1"]
    joblib.dump(
        {"bundle": {"meta_models": {"long_demo_clf": model}}},
        models_dir / "model_state_meta.pkl",
    )
    (meta_dir / "meta_feature_contract.json").write_text(
        json.dumps(
            {
                "meta_models": {
                    "long_demo_clf": {
                        "feature_columns": ["real_a", "real_b"],
                        "positional_feature_mapping": {
                            "f0": "real_a",
                            "f1": "real_b",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_meta_models_from_pickle(str(trained_state_path))
    loaded_model = loaded["long_demo_clf"]

    assert loaded_model.feature_columns == ["real_a", "real_b"]
    assert loaded_model.positional_feature_mapping == {
        "f0": "real_a",
        "f1": "real_b",
    }
    frame = loaded_model._frame(pd.DataFrame({"real_b": [2.0], "real_a": [1.0]}))
    assert frame.iloc[0].to_dict() == {"f0": 1.0, "f1": 2.0}
