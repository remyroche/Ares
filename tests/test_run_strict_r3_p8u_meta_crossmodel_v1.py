from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_p8u_meta_crossmodel_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("strict_r3_p8u_meta_crossmodel", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_every_cross_model_family_scores_tiny_exact_timestamp_queries() -> None:
    timestamps = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    rng = np.random.default_rng(1729)
    x_train = rng.normal(size=(len(timestamps) * 5, 4)).astype(np.float32)
    labels = np.tile(np.array([0, 0, 1, 1, 1], dtype=np.int32), len(timestamps))
    held = rng.normal(size=(9, 4)).astype(np.float32)
    group = [5] * len(timestamps)
    qid = np.repeat(np.arange(len(timestamps), dtype=np.int64), 5)
    for family in MODULE.MODEL_FAMILIES:
        values = MODULE._fit_predict(
            family=family,
            train_x=x_train,
            labels=labels,
            group=group,
            qid=qid,
            held_x=held,
            sample_weight=None,
            label_gain=[0.0, 1.0],
            seed=1729,
        )
        assert values.shape == (len(held),)
        assert np.isfinite(values).all()


def test_query_ids_are_exact_timestamp_and_opaque() -> None:
    frame = pd.DataFrame({"__rank_query_id__": ["q_0100", "q_0000", "q_0100"]})
    assert MODULE._qid(frame).tolist() == [1, 0, 1]


def test_weighted_yetirank_is_rejected_before_fitting() -> None:
    with pytest.raises(AssertionError, match="YetiRank does not support"):
        MODULE._validate_weight_compatibility(
            sample_weight_profile={"equal_timestamp": True},
            model_candidates=[{"model_family": "catboost_yetirank"}],
        )
    MODULE._validate_weight_compatibility(
        sample_weight_profile=None,
        model_candidates=[{"model_family": "catboost_yetirank"}],
    )


def test_final_lgbm_hpo_is_restricted_to_a_real_lgbm_candidate_bank() -> None:
    candidates = [{"model_family": "lgbm_xendcg"} for _ in range(4)]
    MODULE._validate_final_lgbm_hpo(enabled=True, model_candidates=candidates)
    with pytest.raises(AssertionError, match="only lgbm_xendcg"):
        MODULE._validate_final_lgbm_hpo(
            enabled=True,
            model_candidates=[*candidates[:3], {"model_family": "catboost_queryrmse"}],
        )
