from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
from lightgbm import LGBMClassifier

from scripts import run_feature_leaf_reasoning_portability as portability_runner
from extreme_price_movements.tp6_portability_data import TP6PortabilityContract

from extreme_price_movements.strict_oof_base_reasoning import (
    StrictOOFBaseReasoningConfig,
    StrictOOFBaseReasoningError,
    _banded_structural_rule_path,
    build_strict_oof_multiclass_contribution_cache,
    _fit_rule_threshold_bands,
    _rule_hash,
    materialize_strict_oof_base_reasoning,
)


def _models_and_matrices():
    rng = np.random.default_rng(7)
    train = pd.DataFrame(rng.normal(size=(72, 3)), columns=["trend", "flow", "risk"])
    evaluate = pd.DataFrame(rng.normal(size=(18, 3)), columns=train.columns)
    target = ((train["trend"] + 0.5 * train["flow"]) > 0.0).astype(int).to_numpy()
    models = []
    for seed in (11, 23):
        model = LGBMClassifier(
            n_estimators=12,
            learning_rate=0.12,
            num_leaves=5,
            min_child_samples=4,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(train, target)
        models.append(model)
    train_ts = pd.date_range("2024-01-01", periods=len(train), freq="h", tz="UTC")
    eval_ts = pd.date_range("2024-02-01", periods=len(evaluate), freq="h", tz="UTC")
    identity = pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(len(evaluate))],
            "__ts__": eval_ts,
            "side_name": "long",
        }
    )
    return models, train, evaluate, target, train_ts, eval_ts, identity


def test_reasoning_accepts_only_a_certified_f4_compact_manifest(tmp_path) -> None:
    selected = tmp_path / "f4_selected_feature_contract.json"
    selected.write_text(json.dumps({"representation": "F4_compact_top01"}), encoding="utf-8")
    (tmp_path / "f4_run_manifest.json").write_text(json.dumps({
        "status": "F4_FEATURE_CONTRACT_SELECTED",
        "compact_manifest_status": "F4_TRANSPORT_SELECTED_COMPACT_FEATURE_MANIFEST",
    }), encoding="utf-8")
    manifest_path = tmp_path / "portable_feature_manifest.json"
    payload = {
        "status": "F4_TRANSPORT_SELECTED_COMPACT_FEATURE_MANIFEST",
        "selected_representation": "F4_compact_top01",
        "base_control_verified": True,
        "full_f3_control_verified": True,
        "full_f3_control_eligible": False,
        "full_f3_control_status": "FULL_F3_DIAGNOSTIC_INELIGIBLE_COVERAGE_NOT_A_PROMOTION_GATE",
        "meta_control_gate": "PENDING",
        "feature_contract": {"long": ["long_x"], "short": ["short_x"]},
        "selection_artifact": {
            "path": selected.name,
            "sha256": portability_runner._sha256_file(selected),
        },
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert portability_runner._load_certified_f4_feature_contract(manifest_path) == {
        "long": ["long_x"], "short": ["short_x"],
    }
    payload["full_f3_control_verified"] = False
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(portability_runner.PortabilityRunError, match="full-F3"):
        portability_runner._load_certified_f4_feature_contract(manifest_path)


def _multiclass_models_and_matrices():
    rng = np.random.default_rng(17)
    train = pd.DataFrame(rng.normal(size=(108, 3)), columns=["trend", "flow", "risk"])
    evaluate = pd.DataFrame(rng.normal(size=(18, 3)), columns=train.columns)
    latent = train["trend"] + 0.35 * train["flow"] - 0.2 * train["risk"]
    target = np.select([latent < -0.35, latent > 0.35], [0, 2], default=1).astype(int)
    models = []
    for seed in (31, 47):
        model = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=12,
            learning_rate=0.12,
            num_leaves=5,
            min_child_samples=4,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(train, target)
        models.append(model)
    train_ts = pd.date_range("2024-01-01", periods=len(train), freq="h", tz="UTC")
    eval_ts = pd.date_range("2024-02-01", periods=len(evaluate), freq="h", tz="UTC")
    identity = pd.DataFrame(
        {
            "candidate_id": [f"m{i}" for i in range(len(evaluate))],
            "__ts__": eval_ts,
            "side_name": "long",
        }
    )
    return models, train, evaluate, target, train_ts, eval_ts, identity


def test_materialises_strict_oof_g1_g2_g3_and_artifacts(tmp_path):
    models, train, evaluate, target, train_ts, eval_ts, identity = _models_and_matrices()
    labels = pd.DataFrame({"exact_net_bps": np.linspace(-100.0, 100.0, len(evaluate))})
    result = materialize_strict_oof_base_reasoning(
        models,
        train,
        evaluate,
        head_name="r3_tp6_sl4",
        side_name="long",
        fold_id="fold_02",
        train_timestamps=train_ts,
        eval_timestamps=eval_ts,
        eval_identity=identity,
        train_targets=target,
        eval_labels=labels,
        artifact_dir=tmp_path / "fold_02",
        config=StrictOOFBaseReasoningConfig(max_trees_per_model=12, contribution_components=4),
    )

    assert result.artifact_dir == tmp_path / "fold_02"
    assert any(name.startswith("base_reasoning__g1_leaf_train_frequency") for name in result.features)
    assert any(name.startswith("base_reasoning__g2_path_depth") for name in result.features)
    assert any(name.startswith("base_reasoning__g3_contribution_svd") for name in result.features)
    assert not any(name.startswith("label__") for name in result.features)
    assert result.labels is not None
    assert result.labels["label__exact_net_bps"].equals(labels["exact_net_bps"])
    assert {
        "train_leaf_count", "train_leaf_frequency", "train_target_mean",
        "tree_leaf_value", "ensemble_tree_contribution",
    }.issubset(
        result.leaf_rule_catalog.columns
    )
    assert result.leaf_rule_catalog["train_leaf_count"].gt(0.0).all()
    assert np.isfinite(result.leaf_rule_catalog["ensemble_tree_contribution"].to_numpy(float)).all()
    assert result.features["side_name"].eq("long").all()
    assert result.features["head_name"].eq("r3_tp6_sl4").all()
    assert np.isfinite(
        result.features.filter(like="base_reasoning__").to_numpy(dtype=float)
    ).all()

    expected = {
        "base_reasoning_features.parquet",
        "base_reasoning_predictions.parquet",
        "base_reasoning_labels.parquet",
        "leaf_assignments.parquet",
        "leaf_rule_catalog.parquet",
        "contribution_bundle.parquet",
        "base_reasoning_manifest.json",
    }
    assert expected.issubset({path.name for path in result.artifact_dir.iterdir()})
    manifest = json.loads((result.artifact_dir / "base_reasoning_manifest.json").read_text())
    assert manifest["status"] == "MATERIALIZED_STRICT_OOF"
    assert manifest["contract"]["latent_regime_inputs"] is False


def test_leaf_assignments_are_opaque_local_tokens_not_raw_ids():
    models, train, evaluate, target, train_ts, eval_ts, identity = _models_and_matrices()
    result = materialize_strict_oof_base_reasoning(
        models,
        train,
        evaluate,
        head_name="p_clear",
        side_name="long",
        fold_id="2",
        train_timestamps=train_ts,
        eval_timestamps=eval_ts,
        eval_identity=identity,
        train_targets=target,
        config=StrictOOFBaseReasoningConfig(max_trees_per_model=12),
    )
    assignment_columns = [name for name in result.leaf_assignments if name.startswith("leaf_assignment__")]
    assert assignment_columns
    assert all(pd.api.types.is_unsigned_integer_dtype(result.leaf_assignments[name]) for name in assignment_columns)
    assert "leaf_index" not in result.leaf_rule_catalog.columns
    assert {
        "leaf_token",
        "rule_signature",
        "rule_raw_signature",
        "rule_structural_path_json",
        "rule_threshold_band_json",
        "rule_path_json",
        "model_slot",
        "tree_index",
        "tree_leaf_value",
        "ensemble_tree_contribution",
    }.issubset(result.leaf_rule_catalog.columns)
    # Identical numeric LightGBM leaf indices are intentionally scoped with a
    # model hash/slot before persistence, preventing cross-model alignment.
    first = [name for name in assignment_columns if "model_00" in name][0]
    second = [name for name in assignment_columns if "model_01" in name][0]
    assert not result.leaf_assignments[first].equals(result.leaf_assignments[second])


def test_g2_signature_uses_train_only_threshold_bands_not_raw_thresholds():
    train = pd.DataFrame({"trend": np.linspace(-1.0, 1.0, 101, dtype=np.float64)})
    bands = _fit_rule_threshold_bands(
        train,
        ["trend"],
        band_count=10,
        min_train_rows=32,
    )
    early_threshold = [
        {
            "feature": "trend",
            "decision_type": "<=",
            "threshold": "0.11",
            "threshold_kind": "numeric",
            "branch": "right",
        }
    ]
    refit_threshold = [
        {
            "feature": "trend",
            "decision_type": "<=",
            "threshold": "0.19",
            "threshold_kind": "numeric",
            "branch": "right",
        }
    ]
    early_structural = _banded_structural_rule_path(
        early_threshold,
        threshold_bands=bands,
    )
    refit_structural = _banded_structural_rule_path(
        refit_threshold,
        threshold_bands=bands,
    )
    assert _rule_hash(early_threshold) != _rule_hash(refit_threshold)
    assert _rule_hash(early_structural) == _rule_hash(refit_structural)
    assert early_structural[0]["threshold_band_state"] == "numeric_quantile"
    assert early_structural[0]["threshold_band_index"] == refit_structural[0]["threshold_band_index"]


def test_rejects_non_strict_time_boundary_and_cross_side_identity():
    models, train, evaluate, target, train_ts, eval_ts, identity = _models_and_matrices()
    with pytest.raises(StrictOOFBaseReasoningError, match="strict OOF violation"):
        materialize_strict_oof_base_reasoning(
            models,
            train,
            evaluate,
            head_name="p_clear",
            side_name="long",
            fold_id="2",
            train_timestamps=train_ts,
            eval_timestamps=pd.date_range("2024-01-02", periods=len(evaluate), freq="h", tz="UTC"),
            eval_identity=identity.assign(__ts__=pd.date_range("2024-01-02", periods=len(evaluate), freq="h", tz="UTC")),
            train_targets=target,
        )
    bad_identity = identity.copy()
    bad_identity.loc[0, "side_name"] = "short"
    with pytest.raises(StrictOOFBaseReasoningError, match="crosses sides"):
        materialize_strict_oof_base_reasoning(
            models,
            train,
            evaluate,
            head_name="p_clear",
            side_name="long",
            fold_id="2",
            train_timestamps=train_ts,
            eval_timestamps=eval_ts,
            eval_identity=bad_identity,
            train_targets=target,
        )


def test_multiclass_requires_and_materialises_the_explicit_semantic_head(tmp_path):
    models, train, evaluate, target, train_ts, eval_ts, identity = _multiclass_models_and_matrices()
    with pytest.raises(StrictOOFBaseReasoningError, match="multiclass.*explicit class_index"):
        materialize_strict_oof_base_reasoning(
            models,
            train,
            evaluate,
            head_name="p_clear",
            side_name="long",
            fold_id="fold_03",
            train_timestamps=train_ts,
            eval_timestamps=eval_ts,
            eval_identity=identity,
            train_targets=target,
        )
    result = materialize_strict_oof_base_reasoning(
        models,
        train,
        evaluate,
        head_name="p_clear",
        side_name="long",
        fold_id="fold_03",
        train_timestamps=train_ts,
        eval_timestamps=eval_ts,
        eval_identity=identity,
        train_targets=target,
        head_class_map={"p_clear": 2, "p_adverse": 0, "p_weak": 1},
        artifact_dir=tmp_path / "multiclass",
        config=StrictOOFBaseReasoningConfig(
            max_trees_per_model=12,
            contribution_components=4,
            contribution_batch_rows=9,
        ),
    )
    expected = np.mean(
        np.vstack([model.predict_proba(evaluate)[:, 2] for model in models]), axis=0
    )
    assert np.allclose(result.predictions["base_model_prediction"], expected)
    assert result.predictions["class_index"].eq(2).all()
    assert result.manifest["provenance"]["class_index"] == 2
    assert result.manifest["provenance"]["class_selection_source"] == "head_class_map"
    assert result.manifest["provenance"]["model_class_counts"] == [3, 3]
    assert "explicitly selected LightGBM pred_contrib class" in result.manifest["contract"]["contribution_bundle"]

    adverse = materialize_strict_oof_base_reasoning(
        models,
        train,
        evaluate,
        head_name="p_adverse",
        side_name="long",
        fold_id="fold_03",
        train_timestamps=train_ts,
        eval_timestamps=eval_ts,
        eval_identity=identity,
        train_targets=target,
        class_index=0,
        config=StrictOOFBaseReasoningConfig(
            max_trees_per_model=12,
            contribution_components=4,
            contribution_batch_rows=9,
        ),
    )
    weak = materialize_strict_oof_base_reasoning(
        models,
        train,
        evaluate,
        head_name="p_weak",
        side_name="long",
        fold_id="fold_03",
        train_timestamps=train_ts,
        eval_timestamps=eval_ts,
        eval_identity=identity,
        train_targets=target,
        head_class_map={"p_clear": 2, "p_adverse": 0, "p_weak": 1},
        config=StrictOOFBaseReasoningConfig(
            max_trees_per_model=12,
            contribution_components=4,
            contribution_batch_rows=9,
        ),
    )
    for output, class_i, source in (
        (adverse, 0, "explicit_class_index"),
        (weak, 1, "head_class_map"),
    ):
        expected = np.mean(
            np.vstack([model.predict_proba(evaluate)[:, class_i] for model in models]),
            axis=0,
        )
        assert np.allclose(output.predictions["base_model_prediction"], expected)
        assert output.predictions["class_index"].eq(class_i).all()
        assert output.manifest["provenance"]["class_index"] == class_i
        assert output.manifest["provenance"]["class_selection_source"] == source
    # The G3 path is selected with the same head index, not merely the score:
    # the clear and adverse additive bundles cannot be identical here.
    assert not np.allclose(
        result.contribution_bundle.filter(like="base_reasoning__g3_").to_numpy(),
        adverse.contribution_bundle.filter(like="base_reasoning__g3_").to_numpy(),
    )


def test_multiclass_leaf_rules_use_only_the_selected_head_tree_stride():
    models, train, evaluate, target, train_ts, eval_ts, identity = _multiclass_models_and_matrices()
    config = StrictOOFBaseReasoningConfig(
        max_trees_per_model=3,
        contribution_components=4,
    )
    clear = materialize_strict_oof_base_reasoning(
        models,
        train,
        evaluate,
        head_name="p_clear",
        side_name="long",
        fold_id="fold_04",
        train_timestamps=train_ts,
        eval_timestamps=eval_ts,
        eval_identity=identity,
        train_targets=target,
        class_index=2,
        config=config,
    )
    adverse = materialize_strict_oof_base_reasoning(
        models,
        train,
        evaluate,
        head_name="p_adverse",
        side_name="long",
        fold_id="fold_04",
        train_timestamps=train_ts,
        eval_timestamps=eval_ts,
        eval_identity=identity,
        train_targets=target,
        class_index=0,
        config=config,
    )
    for output, expected_indices in ((clear, [2, 5, 8]), (adverse, [0, 3, 6])):
        selections = output.manifest["provenance"]["head_tree_selection"]
        assert len(selections) == len(models)
        assert all(item["trees_per_iteration"] == 3 for item in selections)
        assert all(item["selected_model_tree_indices"] == expected_indices for item in selections)
        for _, catalog in output.leaf_rule_catalog.groupby("model_slot", sort=True):
            assert sorted(catalog["tree_index"].unique().tolist()) == expected_indices
            assert sorted(catalog["head_tree_slot"].unique().tolist()) == [0, 1, 2]
        assignment_columns = [
            name
            for name in output.leaf_assignments
            if name.startswith("leaf_assignment__")
        ]
        assert len(assignment_columns) == len(models) * 3
        assert all("_head_tree_" in name for name in assignment_columns)


def test_shared_multiclass_contribution_cache_matches_uncached_g3_without_leaf_leakage():
    """All semantic heads must retain their exact G3 class slice semantics."""

    models, train, evaluate, target, train_ts, eval_ts, identity = _multiclass_models_and_matrices()
    config = StrictOOFBaseReasoningConfig(
        max_trees_per_model=12,
        contribution_components=4,
        contribution_batch_rows=9,
    )
    cache = build_strict_oof_multiclass_contribution_cache(
        models,
        train,
        evaluate,
        batch_rows=config.contribution_batch_rows,
        max_cache_bytes=64 * 1024 * 1024,
    )
    # The cache is additive contribution data only.  It has no leaf assignment
    # payload, and the matrices are immutable after construction.
    assert not any("leaf" in name.lower() for name in vars(cache))
    assert not cache.train_contributions.flags.writeable
    assert not cache.eval_contributions.flags.writeable
    for head, class_index in (("p_adverse", 0), ("p_weak", 1), ("p_clear", 2)):
        uncached = materialize_strict_oof_base_reasoning(
            models,
            train,
            evaluate,
            head_name=head,
            side_name="long",
            fold_id="fold_cache",
            train_timestamps=train_ts,
            eval_timestamps=eval_ts,
            eval_identity=identity,
            train_targets=target,
            class_index=class_index,
            config=config,
        )
        cached = materialize_strict_oof_base_reasoning(
            models,
            train,
            evaluate,
            head_name=head,
            side_name="long",
            fold_id="fold_cache",
            train_timestamps=train_ts,
            eval_timestamps=eval_ts,
            eval_identity=identity,
            train_targets=target,
            class_index=class_index,
            contribution_cache=cache,
            config=config,
        )
        g3 = [name for name in uncached.contribution_bundle if name.startswith("base_reasoning__g3_")]
        assert g3
        np.testing.assert_allclose(
            cached.contribution_bundle.loc[:, g3].to_numpy(dtype=np.float32),
            uncached.contribution_bundle.loc[:, g3].to_numpy(dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )
        assert not any("leaf" in name.lower() for name in cached.contribution_bundle)
        assert cached.manifest["provenance"]["contribution_cache"] == {
            "mode": "shared_multiclass_per_fold",
            "class_count": 3,
            "retained_bytes": cache.retained_bytes,
            "selected_class_slice": class_index,
            "contains_raw_leaf_identifiers": False,
        }


def test_spill_multiclass_contribution_cache_matches_memory_and_is_released(tmp_path):
    """A RAM-capacity miss must not force three later-fold contribution passes."""

    models, train, evaluate, target, train_ts, eval_ts, identity = _multiclass_models_and_matrices()
    config = StrictOOFBaseReasoningConfig(
        max_trees_per_model=12,
        contribution_components=4,
        contribution_batch_rows=9,
    )
    memory_cache = build_strict_oof_multiclass_contribution_cache(
        models,
        train,
        evaluate,
        batch_rows=config.contribution_batch_rows,
        max_cache_bytes=64 * 1024 * 1024,
    )
    spill_root = tmp_path / "spill"
    spill_cache = build_strict_oof_multiclass_contribution_cache(
        models,
        train,
        evaluate,
        batch_rows=config.contribution_batch_rows,
        max_cache_bytes=1,
        spill_directory=spill_root,
        max_spill_bytes=64 * 1024 * 1024,
        spill_max_working_bytes=4 * 1024,
    )
    assert spill_cache.storage_mode == "disk_mmap"
    assert spill_cache.backing_directory is not None
    assert spill_cache.backing_directory.exists()
    assert isinstance(spill_cache.train_contributions, np.memmap)
    assert not spill_cache.train_contributions.flags.writeable
    assert not spill_cache.eval_contributions.flags.writeable
    try:
        for head, class_index in (("p_adverse", 0), ("p_weak", 1), ("p_clear", 2)):
            memory = materialize_strict_oof_base_reasoning(
                models,
                train,
                evaluate,
                head_name=head,
                side_name="long",
                fold_id="fold_spill",
                train_timestamps=train_ts,
                eval_timestamps=eval_ts,
                eval_identity=identity,
                train_targets=target,
                class_index=class_index,
                contribution_cache=memory_cache,
                config=config,
            )
            spill = materialize_strict_oof_base_reasoning(
                models,
                train,
                evaluate,
                head_name=head,
                side_name="long",
                fold_id="fold_spill",
                train_timestamps=train_ts,
                eval_timestamps=eval_ts,
                eval_identity=identity,
                train_targets=target,
                class_index=class_index,
                contribution_cache=spill_cache,
                config=config,
            )
            g3 = [name for name in memory.contribution_bundle if name.startswith("base_reasoning__g3_")]
            np.testing.assert_allclose(
                spill.contribution_bundle.loc[:, g3].to_numpy(dtype=np.float32),
                memory.contribution_bundle.loc[:, g3].to_numpy(dtype=np.float32),
                rtol=0.0,
                atol=0.0,
            )
            provenance = spill.manifest["provenance"]["contribution_cache"]
            assert provenance["storage_mode"] == "disk_mmap"
            assert not any("leaf" in name.lower() for name in spill.contribution_bundle)
    finally:
        spill_cache.release()
        memory_cache.release()
    assert spill_cache.closed
    assert spill_cache.backing_directory is not None
    assert not spill_cache.backing_directory.exists()


def test_reasoning_runner_reuses_one_bounded_multiclass_cache_for_all_heads(tmp_path, monkeypatch):
    """The strict runner must opt in without weakening per-head selection."""

    models, train_matrix, eval_matrix, target, train_ts, eval_ts, _ = _multiclass_models_and_matrices()
    model = models[0]
    train = train_matrix.copy()
    evaluate = eval_matrix.copy()
    train["candidate_id"] = [f"train_{index}" for index in range(len(train))]
    evaluate["candidate_id"] = [f"eval_{index}" for index in range(len(evaluate))]
    train["decision_ts"] = train_ts
    evaluate["decision_ts"] = eval_ts
    train["label_available_ts"] = train_ts + pd.Timedelta(hours=13)
    evaluate["label_available_ts"] = eval_ts + pd.Timedelta(hours=13)
    train["side_name"] = "long"
    evaluate["side_name"] = "long"
    train["r3_class"] = target
    evaluate["r3_class"] = 1
    train["gross_bps"] = 5.0
    evaluate["gross_bps"] = 5.0
    train["net_bps"] = -95.0
    evaluate["net_bps"] = -95.0

    # Force the runner past its RAM cache bound.  It must make one all-class
    # contribution pass for train and one for evaluation, then reuse the
    # bounded temporary cache for all three semantic heads.
    calls: list[int] = []
    original_predict = model.predict

    def tracked_predict(*args, **kwargs):
        if kwargs.get("pred_contrib"):
            calls.append(len(args[0]))
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(model, "predict", tracked_predict)
    monkeypatch.setattr(portability_runner, "STRICT_REASONING_MULTICLASS_CACHE_MAX_BYTES", 1)
    monkeypatch.setattr(
        portability_runner,
        "STRICT_REASONING_MULTICLASS_SPILL_MAX_BYTES",
        64 * 1024 * 1024,
    )
    monkeypatch.setattr(
        portability_runner,
        "STRICT_REASONING_MULTICLASS_SPILL_MAX_WORKING_BYTES",
        64 * 1024,
    )

    portability_runner._reasoning_for_fold(
        model=model,
        train=train,
        evaluate=evaluate,
        columns=list(train_matrix.columns),
        side="long",
        fold_id="runner_cache",
        probabilities=np.asarray(model.predict_proba(eval_matrix), dtype=np.float32),
        destination=tmp_path / "reasoning",
        return_payload=False,
    )
    manifests = sorted((tmp_path / "reasoning").glob("p_*/base_reasoning_manifest.json"))
    assert len(manifests) == 3
    for path in manifests:
        manifest = json.loads(path.read_text())
        cache_provenance = manifest["provenance"]["contribution_cache"]
        assert cache_provenance["mode"] == "shared_multiclass_per_fold"
        assert cache_provenance["storage_mode"] == "disk_mmap"
        assert cache_provenance["contains_raw_leaf_identifiers"] is False
        bundle = pd.read_parquet(path.parent / "contribution_bundle.parquet")
        assert not any("leaf" in name.lower() for name in bundle)
    assert calls == [len(train), len(evaluate)]
    assert not list(tmp_path.glob(".strict_oof_contrib_*"))


def _runner_resume_inputs():
    models, train_matrix, eval_matrix, target, train_ts, eval_ts, _ = _multiclass_models_and_matrices()
    model = models[0]
    train = train_matrix.copy()
    evaluate = eval_matrix.copy()
    train["candidate_id"] = [f"train_{index}" for index in range(len(train))]
    evaluate["candidate_id"] = [f"eval_{index}" for index in range(len(evaluate))]
    train["decision_ts"] = train_ts
    evaluate["decision_ts"] = eval_ts
    train["label_available_ts"] = train_ts + pd.Timedelta(hours=13)
    evaluate["label_available_ts"] = eval_ts + pd.Timedelta(hours=13)
    train["side_name"] = "long"
    evaluate["side_name"] = "long"
    train["r3_class"] = target
    evaluate["r3_class"] = 1
    train["gross_bps"] = 5.0
    evaluate["gross_bps"] = 5.0
    train["net_bps"] = -95.0
    evaluate["net_bps"] = -95.0
    return model, train, evaluate, list(train_matrix.columns), np.asarray(model.predict_proba(eval_matrix), dtype=np.float32)


def test_reasoning_runner_resume_reuses_only_exact_complete_heads(tmp_path, monkeypatch):
    model, train, evaluate, columns, probabilities = _runner_resume_inputs()
    destination = tmp_path / "reasoning"
    kwargs = dict(
        model=model,
        train=train,
        evaluate=evaluate,
        columns=columns,
        side="long",
        fold_id="resume_fold",
        probabilities=probabilities,
        destination=destination,
        return_payload=False,
    )
    portability_runner._reasoning_for_fold(**kwargs)

    def should_not_build_cache(*args, **kwargs):
        raise AssertionError("a fully verified resumed fold must not recompute pred_contrib")

    monkeypatch.setattr(
        portability_runner,
        "build_strict_oof_multiclass_contribution_cache",
        should_not_build_cache,
    )
    portability_runner._reasoning_for_fold(**kwargs, resume_existing=True)
    assert len(list(destination.glob("p_*/base_reasoning_manifest.json"))) == 3


def test_reasoning_runner_resume_rejects_changed_labels_even_with_same_base_model(tmp_path):
    model, train, evaluate, columns, probabilities = _runner_resume_inputs()
    destination = tmp_path / "reasoning"
    kwargs = dict(
        model=model,
        train=train,
        evaluate=evaluate,
        columns=columns,
        side="long",
        fold_id="resume_label_guard",
        probabilities=probabilities,
        destination=destination,
        return_payload=False,
    )
    portability_runner._reasoning_for_fold(**kwargs)
    changed = evaluate.copy()
    changed.loc[changed.index[0], "net_bps"] = -94.0
    with pytest.raises(portability_runner.PortabilityRunError, match="label provenance mismatch"):
        portability_runner._reasoning_for_fold(**{**kwargs, "evaluate": changed}, resume_existing=True)


def test_reasoning_runner_resume_materializes_only_a_missing_head(tmp_path, monkeypatch):
    model, train, evaluate, columns, probabilities = _runner_resume_inputs()
    destination = tmp_path / "reasoning"
    kwargs = dict(
        model=model,
        train=train,
        evaluate=evaluate,
        columns=columns,
        side="long",
        fold_id="resume_missing_head",
        probabilities=probabilities,
        destination=destination,
        return_payload=False,
    )
    portability_runner._reasoning_for_fold(**kwargs)
    removed = destination / "p_weak"
    original_adverse_hash = portability_runner._sha256_file(
        destination / "p_adverse" / "base_reasoning_features.parquet"
    )
    shutil.rmtree(removed)

    materialized: list[str] = []
    original_materialize = portability_runner.materialize_strict_oof_base_reasoning

    def track_materialized(*args, **inner_kwargs):
        materialized.append(str(inner_kwargs["head_name"]))
        return original_materialize(*args, **inner_kwargs)

    monkeypatch.setattr(portability_runner, "materialize_strict_oof_base_reasoning", track_materialized)
    portability_runner._reasoning_for_fold(**kwargs, resume_existing=True)
    assert materialized == ["p_weak"]
    assert (removed / "base_reasoning_manifest.json").exists()
    assert portability_runner._sha256_file(
        destination / "p_adverse" / "base_reasoning_features.parquet"
    ) == original_adverse_hash


def test_prepare_destination_resume_is_explicit_and_running_only(tmp_path):
    destination = tmp_path / "interrupted"
    destination.mkdir()
    (destination / "run_manifest.json").write_text(json.dumps({
        "schema": portability_runner.SCHEMA,
        "status": "RUNNING",
    }), encoding="utf-8")
    assert portability_runner._prepare_destination(destination, resume=True) is True
    with pytest.raises(FileExistsError):
        portability_runner._prepare_destination(destination)
    (destination / "run_manifest.json").write_text(json.dumps({
        "schema": portability_runner.SCHEMA,
        "status": "STRICT_OOF_BASE_REASONING_COMPLETED",
    }), encoding="utf-8")
    with pytest.raises(portability_runner.PortabilityRunError, match="immutable"):
        portability_runner._prepare_destination(destination, resume=True)


def test_resume_source_contract_uses_the_same_json_safe_representation_as_manifest():
    payload = portability_runner._source_contract_payload(TP6PortabilityContract())
    assert all(isinstance(value, str) for key, value in payload.items() if key in {
        "panel", "winner", "robust", "feature_manifest",
    })
    assert payload == json.loads(json.dumps(TP6PortabilityContract().__dict__, default=str, sort_keys=True))
