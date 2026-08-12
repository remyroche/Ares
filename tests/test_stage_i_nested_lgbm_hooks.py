from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_nested_feature_challenger import NestedFeatureSet
from extreme_price_movements.stage_i_nested_lgbm_hooks import FixedLGBMContract, fixed_lgbm_base_predictor, fixed_lgbm_meta_predictor, fold_local_meta_feature_selector, require_side_meta_params, resolve_side_meta_context_universe
from extreme_price_movements.stage_i_nested_stack_execution import GuardedMetaArmSpec


def test_fold_local_meta_selector_keeps_direct_trust_and_never_uses_converted_base_score() -> None:
    n = 80
    rng = np.random.default_rng(4)
    mandatory = ("r3_p_adverse", "r3_p_weak", "r3_p_clear", "r3_opportunity_score", "base_r3_max_probability", "base_r3_top2_margin", "base_r3_entropy")
    frame = pd.DataFrame({name: rng.normal(size=n) for name in mandatory})
    frame["regime_signal"] = np.linspace(-1.0, 1.0, n)
    frame["regime_duplicate"] = frame["regime_signal"]
    frame["constant"] = 1.0
    frame["sparse"] = np.where(np.arange(n) < 10, 1.0, np.nan)
    # This name is intentionally absent from the declared list: conversion is
    # prohibited before the final post-OOF pooled mapping stage.
    frame["prequential_base_expected_net_bps"] = rng.normal(size=n)
    selector = fold_local_meta_feature_selector(FixedLGBMContract(base_params={}, meta_params={}, meta_feature_cap=9))
    selected, audit = selector(frame, (frame.regime_signal > 0).to_numpy(float), (*mandatory, "regime_signal", "regime_duplicate", "constant", "sparse"), mandatory, GuardedMetaArmSpec("reliable", "reliability"))
    assert set(mandatory).issubset(selected)
    assert "prequential_base_expected_net_bps" not in selected
    assert len(selected) <= 9
    assert "constant" not in selected and "sparse" not in selected
    assert len({"regime_signal", "regime_duplicate"}.intersection(selected)) == 1
    assert len({"regime_signal", "regime_duplicate"}.intersection(audit["spearman_pruned_features"])) == 1
    assert audit["schema"] == "stage_i_nested_meta_fold_selector_v1"


def test_fixed_factories_fit_r3_and_target_aligned_binary_models() -> None:
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(8)
    train = pd.DataFrame({"x": rng.normal(size=45), "z": rng.normal(size=45)})
    valid = pd.DataFrame({"x": rng.normal(size=9), "z": rng.normal(size=9)})
    contract = FixedLGBMContract(
        base_params={"n_estimators": 3, "min_child_samples": 2, "random_state": 3},
        meta_params={"n_estimators": 3, "min_child_samples": 2, "random_state": 3},
    )
    feature_set = NestedFeatureSet("long", "top20", 20, ("x", "z"), (), {"x": 1, "z": 2}, {"x": "x", "z": "z"}, {"x": 1, "z": 1}, {"selected_automatic_sparse": 2}, "source")
    probability = fixed_lgbm_base_predictor(contract)(train, np.tile([0, 1, 2], 15), valid, feature_set)
    assert probability.shape == (len(valid), 3)
    assert np.allclose(probability.sum(axis=1), 1.0)
    prediction = fixed_lgbm_meta_predictor(contract)(train, np.tile([0, 1], 23)[:len(train)], np.ones(len(train)), valid, GuardedMetaArmSpec("reliable", "reliability"))
    assert prediction.shape == (len(valid),)
    assert np.all((prediction >= 0) & (prediction <= 1))


def test_side_meta_parameter_contract_rejects_missing_and_extra_sides() -> None:
    params = require_side_meta_params({"long": {"n_estimators": 3}, "short": {"n_estimators": 5}})
    assert params["long"]["n_estimators"] == 3
    with pytest.raises(Exception, match="missing=.*short"):
        require_side_meta_params({"long": {"n_estimators": 3}})
    with pytest.raises(Exception, match="extra=.*other"):
        require_side_meta_params({"long": {"n_estimators": 3}, "short": {"n_estimators": 5}, "other": {}})


def test_declared_meta_context_is_side_local_and_records_universe_provenance() -> None:
    cfg = {
        "meta_long_feature_keys": ["long_only"], "meta_short_feature_keys": ["short_only"],
        "meta_shared_feature_keys": ["shared"], "meta_product_feature_keys": ["product"],
        "STAGE_I_M6_SHARED_UNION_META_FEATURE_KEYS": ["m6"],
    }
    columns = ["long_only", "short_only", "shared", "product", "m6"]
    long, long_audit = resolve_side_meta_context_universe(cfg, side="long", available_columns=columns, direct_columns=())
    short, short_audit = resolve_side_meta_context_universe(cfg, side="short", available_columns=columns, direct_columns=())
    assert "long_only" in long and "short_only" not in long
    assert "short_only" in short and "long_only" not in short
    assert long_audit["side_specific_key_present"] is True
    assert short_audit["side_specific_key_present"] is True
    assert long_audit["declared_universe_sha256"] != short_audit["declared_universe_sha256"]


def test_declared_meta_context_supports_shared_pool_only_with_side_local_lineage() -> None:
    cfg = {
        "meta_shared_feature_keys": ["shared"], "meta_product_feature_keys": ["product"],
        "STAGE_I_M6_SHARED_UNION_META_FEATURE_KEYS": ["m6"],
    }
    long, long_audit = resolve_side_meta_context_universe(cfg, side="long", available_columns=["shared", "product", "m6"], direct_columns=())
    short, short_audit = resolve_side_meta_context_universe(cfg, side="short", available_columns=["shared", "product", "m6"], direct_columns=())
    assert long == short == ("shared", "product", "m6")
    assert long_audit["side_specific_key_present"] is False
    assert short_audit["side_specific_key_present"] is False
    # The same layer pool is resolved and selected independently under a
    # side-labelled provenance hash; no shared fitted selector state exists.
    assert long_audit["side"] == "long" and short_audit["side"] == "short"
    assert long_audit["declared_universe_sha256"] != short_audit["declared_universe_sha256"]
