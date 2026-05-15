from __future__ import annotations

import importlib.util
import sys
import types
import importlib.machinery
from pathlib import Path
import numpy as np
import pandas as pd

from extreme_price_movements.ebm_on_lgbm import (
    EBMOnLGBMModel,
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
