from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements import catboost_archetype_classifier as archetype
from extreme_price_movements.features_gmm_ae import AE_GMM_FEATURE_COLUMNS
from extreme_price_movements.path_archetype_labels import (
    PATH_ARCHETYPE_RULE_VERSION,
    deterministic_combined_path_archetype,
    deterministic_path_archetype,
)

RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_catboost_path_archetype_classifier.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_catboost_path_archetype_classifier", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_hpo_sample_time_spreads_each_side_class_stratum_in_chronological_regions() -> None:
    rows = 1_200
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
            "__label_end_ts__": pd.date_range(
                "2026-01-01", periods=rows, freq="h", tz="UTC"
            )
            + pd.Timedelta(hours=4),
            "side": np.where(np.arange(rows) % 2, "long", "short"),
        }
    )
    labels = pd.Series(np.where(np.arange(rows) % 3, "winner", "timeout"))
    positions, contract = runner._stratified_hpo_sample(
        frame,
        labels,
        sample_rows=240,
        validation_folds=3,
        timestamp_column="__ts__",
        label_end_column="__label_end_ts__",
        side_column="side",
    )
    assert len(positions) == 240
    assert contract["version"] == runner.HPO_SAMPLING_CONTRACT_VERSION
    assert [region["sample_rows"] for region in contract["regions"]] == [60] * 4
    assert all(np.all(np.diff(chunk) > 0) for chunk in np.array_split(positions, 4))
    assert any(np.diff(positions) > 1)
    for region in contract["regions"]:
        support = region["sample_support"]["side_class_support"]
        assert {(row["side"], row["class"]) for row in support} == {
            ("long", "timeout"),
            ("long", "winner"),
            ("short", "timeout"),
            ("short", "winner"),
        }


def test_hpo_sample_rejects_validation_class_without_prior_sampled_train_support() -> None:
    rows = 300
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
            "__label_end_ts__": pd.date_range(
                "2026-01-01", periods=rows, freq="h", tz="UTC"
            )
            + pd.Timedelta(hours=2),
            "side": "long",
        }
    )
    labels = pd.Series(np.where(np.arange(rows) < 100, "old", "new"))
    positions, _contract = runner._stratified_hpo_sample(
        frame,
        labels,
        sample_rows=90,
        validation_folds=2,
        timestamp_column="__ts__",
        label_end_column="__label_end_ts__",
        side_column="side",
    )
    config = archetype.PathArchetypeConfig(
        oof_folds=2,
        embargo=pd.Timedelta(hours=2),
    )
    with pytest.raises(ValueError, match="validation classes absent from prior sampled"):
        runner._validate_hpo_sample_class_support(
            frame.iloc[positions],
            labels.iloc[positions],
            timestamp_column="__ts__",
            label_end_column="__label_end_ts__",
            side_column="side",
            config=config,
        )


class _PickleModel:
    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return np.full((len(values), 4), 0.25)


def _frame(rows: int = 300) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    paths = [[-1.0, -0.6], [-0.2, 0.1], [0.5, 1.0], [1.1, 1.6]] * (rows // 4)
    summaries = archetype.summarize_future_paths(
        pd.DataFrame({"future_path": paths}), "future_path"
    )
    frame = pd.DataFrame(
        {
            "__ts__": ts,
            "__label_end_ts__": ts + pd.Timedelta(hours=4),
            "__symbol__": "BTC/USD",
            "side": "long",
            "side_name": "long",
            "candidate_id": [f"BTC/USD|{value.isoformat()}|1h|long" for value in ts],
            "base_x": np.arange(rows, dtype=float),
            "meta_x": np.arange(rows, dtype=float) % 7,
        }
    )
    frame = pd.concat([frame, summaries], axis=1)
    frame["path_archetype"] = summaries.apply(
        deterministic_combined_path_archetype, axis=1
    )
    frame["path_shape_archetype"] = summaries.apply(
        deterministic_path_archetype, axis=1
    )
    frame["path_archetype_rule_version"] = PATH_ARCHETYPE_RULE_VERSION
    return frame


def _config() -> dict[str, object]:
    return {
        "base_shared_feature_keys": ["base_x"],
        "base_long_feature_keys": [],
        "base_short_feature_keys": [],
        "meta_shared_feature_keys": ["meta_x"],
        "META_BASE_PERFORMANCE_FEATURE_KEYS": [],
        "MODEL_DERIVED_META_PERFORMANCE_FEATURE_KEYS": [],
    }


def _write_frozen_ae_gmm_sidecar(
    path: Path,
    frame: pd.DataFrame,
    *,
    shuffle: bool = False,
) -> pd.DataFrame:
    values = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(frame["__ts__"], utc=True),
            "__symbol__": frame["__symbol__"].astype(str),
            "side": np.int8(1),
            **{
                column: np.arange(len(frame), dtype=np.float32) + index
                for index, column in enumerate(AE_GMM_FEATURE_COLUMNS)
            },
        }
    )
    if shuffle:
        values = values.sample(frac=1.0, random_state=7).reset_index(drop=True)
    values.to_parquet(path, index=False)
    return values


@pytest.mark.parametrize("failure", ["missing", "duplicate", "nonfinite"])
def test_frozen_ae_gmm_sidecar_rejects_missing_duplicate_and_nonfinite_rows(
    tmp_path: Path,
    failure: str,
) -> None:
    frame = _frame(40)
    sidecar = tmp_path / "frozen.parquet"
    values = _write_frozen_ae_gmm_sidecar(sidecar, frame)
    if failure == "missing":
        values.iloc[:-1].to_parquet(sidecar, index=False)
        message = "does not cover every label key"
    elif failure == "duplicate":
        pd.concat([values, values.iloc[[0]]], ignore_index=True).to_parquet(
            sidecar, index=False
        )
        message = "duplicate timestamp/symbol/side"
    else:
        values.loc[0, AE_GMM_FEATURE_COLUMNS[0]] = np.nan
        values.to_parquet(sidecar, index=False)
        message = "non-finite generated outputs"
    with pytest.raises(ValueError, match=message):
        runner._validate_frozen_ae_gmm_sidecar(
            frame,
            sidecar_path=sidecar,
            manifest_path=None,
            timestamp_column="__ts__",
            side_column="side",
        )


def test_frozen_ae_gmm_sidecar_load_preserves_full_label_order_and_alignment(
    tmp_path: Path,
) -> None:
    frame = _frame(40)
    sidecar = tmp_path / "frozen.parquet"
    _write_frozen_ae_gmm_sidecar(sidecar, frame, shuffle=True)
    contract = runner._validate_frozen_ae_gmm_sidecar(
        frame,
        sidecar_path=sidecar,
        manifest_path=None,
        timestamp_column="__ts__",
        side_column="side",
    )
    loaded = runner._load_frozen_ae_gmm_matrix(
        frame,
        AE_GMM_FEATURE_COLUMNS[:2],
        sidecar_contract=contract,
        timestamp_column="__ts__",
        side_column="side",
    )
    assert loaded.index.equals(frame.index)
    np.testing.assert_allclose(loaded[AE_GMM_FEATURE_COLUMNS[0]], np.arange(len(frame)))
    np.testing.assert_allclose(
        loaded[AE_GMM_FEATURE_COLUMNS[1]], np.arange(len(frame)) + 1
    )


def test_smoke_runner_freezes_train_only_labels_and_persists_required_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {"selector_calls": 0, "hpo_calls": 0}

    def fake_selector(
        features: pd.DataFrame, *args: object, **kwargs: object
    ) -> archetype.FastSelectorResult:
        captured["selector_calls"] = int(captured["selector_calls"]) + 1
        captured["selector_labels"] = set(pd.Series(args[0]).astype(str))
        return archetype.FastSelectorResult(
            selected_features=("base_x", "meta_x"),
            candidate_features=("base_x", "meta_x"),
            mandatory_features=("base_x",),
            availability={"base_x": 1.0, "meta_x": 1.0},
            scores=pd.DataFrame({"mi": [1.0, 0.5]}, index=["base_x", "meta_x"]),
            correlation_clusters=(("base_x",), ("meta_x",)),
            proxy_backend="test_proxy",
        )

    def fake_hpo(*args: object, **kwargs: object) -> SimpleNamespace:
        captured["hpo_calls"] = int(captured["hpo_calls"]) + 1
        captured["hpo_trials"] = kwargs["n_trials"]
        return SimpleNamespace(
            best_params={"iterations": 4},
            report=lambda: {"best_params": {"iterations": 4}},
        )

    def fake_oof(
        features: pd.DataFrame, target: pd.Series, *args: object, **kwargs: object
    ) -> archetype.OOFPathArchetypeResult:
        classes = np.asarray(sorted(pd.unique(target.astype(str))))
        n = len(features)
        probabilities = np.full((n, len(classes)), 1.0 / len(classes))
        fold_ids = np.where(np.arange(n) < n // 4, -1, np.arange(n) % 2)
        return archetype.OOFPathArchetypeResult(
            probabilities=probabilities,
            fold_ids=fold_ids,
            folds=[],
            models=[],
            classes=classes,
            feature_columns=tuple(features.columns),
            diagnostics={"logloss": 1.0},
        )

    def fake_permutation(
        *args: object, **kwargs: object
    ) -> tuple[list[str], pd.DataFrame]:
        captured["stages"] = kwargs["stages"]
        return ["base_x"], pd.DataFrame(
            [{
                "stage": 32,
                "feature": "base_x",
                "selected": True,
                "stage_acceleration_algorithm_version": "test-v1",
                "stage_input_feature_count": 2,
                "stage_keep_count": 1,
                "stage_full_mda_candidate_count": 2,
                "stage_screened_out_count": 0,
                "stage_screening_used": False,
                "stage_fit_calls": 0,
                "stage_permutation_predict_calls": 1,
                "stage_total_seconds": 0.25,
            }]
        )

    def fake_final(
        *args: object, **kwargs: object
    ) -> archetype.PathArchetypeClassifier:
        return archetype.PathArchetypeClassifier(
            ("base_x",), ("a", "b", "c", "d"), _PickleModel()
        )

    monkeypatch.setattr(archetype, "fast_select_preentry_features", fake_selector)
    monkeypatch.setattr(archetype, "optimize_purged_catboost_hpo", fake_hpo)
    monkeypatch.setattr(archetype, "fit_purged_chronological_oof_catboost", fake_oof)
    monkeypatch.setattr(archetype, "staged_permutation_selection", fake_permutation)
    monkeypatch.setattr(runner, "_fit_final_classifier", fake_final)

    output = tmp_path / "artifacts"
    frame = _frame()
    manifest = runner.run_pipeline(
        frame,
        output,
        discovery_end="2026-01-05T00:00:00Z",
        config_mapping=_config(),
        mandatory_features=["base_x"],
        hpo_trials=9,
        max_rows=900,
        smoke=True,
    )

    assert manifest["rows"] == runner.SMOKE_MAX_ROWS
    assert manifest["discovery_rows"] == 92
    assert manifest["future_training_taxonomy"]["ordered_classes"] == list(
        runner.MERGED_PATH_ARCHETYPE_CLASSES
    )
    assert captured["hpo_trials"] == 1
    assert captured["stages"] == runner.SMOKE_PERMUTATION_STAGES
    assert captured["selector_calls"] == 1
    assert "fast_realization_winner" in captured["selector_labels"]
    assert not set(runner.LEGACY_FAST_CLASSES).intersection(captured["selector_labels"])
    assert captured["hpo_calls"] == 1
    assert not (output / "path_archetype_discovery.joblib").exists()
    assert (output / "path_archetype_classifier.joblib").exists()
    assert (output / "feature_selection_manifest.json").exists()
    assert (output / "hpo_manifest.json").exists()
    oof = pd.read_parquet(output / "oof_probabilities.parquet")
    assert "path_archetype_raw" in oof
    assert "predicted_path_archetype" in oof
    assert "probability_entropy" in oof
    assert {
        "max_probability",
        "normalized_entropy",
        "top2_probability_margin",
        "adverse_probability_mass",
        "favorable_probability_mass",
    }.issubset(oof.columns)
    assert {
        f"probability__{label}" for label in runner.MERGED_PATH_ARCHETYPE_CLASSES
    }.issubset(oof.columns)
    assert oof["probability_entropy"].notna().all()
    assert np.allclose(
        oof.loc[:, [f"probability__{label}" for label in runner.MERGED_PATH_ARCHETYPE_CLASSES]].sum(axis=1),
        1.0,
    )
    role_manifest = json.loads(
        (output / "oof_probabilities.role_manifest.json").read_text()
    )
    assert role_manifest["future_training_taxonomy"]["probability_contract"] == (
        runner.FUTURE_TRAINING_TAXONOMY_CONTRACT["probability_contract"]
    )
    assert oof["candidate_id"].notna().all()
    assert {
        "available_at",
        "validation_start",
        "latest_train_decision_ts",
        "train_decision_cutoff",
        "label_resolution_available_at",
    }.issubset(oof.columns)
    assert (oof["label_resolution_available_at"] <= oof["train_decision_cutoff"]).all()
    assert (oof["train_decision_cutoff"] < oof["validation_start"]).all()
    role_manifest = json.loads(
        (output / "oof_probabilities.role_manifest.json").read_text()
    )
    assert role_manifest["prediction_role"] == "path_archetype_oof"
    assert role_manifest["source_artifact_sha256"] == runner._sha256_file(
        output / "oof_probabilities.parquet"
    )
    assert role_manifest["prediction_role_manifest_sha256"] == runner._signed_manifest_hash(
        role_manifest
    )
    feature_manifest = json.loads(
        (output / "feature_selection_manifest.json").read_text()
    )
    assert feature_manifest["final_selected_features"] == ["base_x"]
    assert feature_manifest["permutation_stage_metrics"] == [{
        "stage": 32,
        "stage_acceleration_algorithm_version": "test-v1",
        "stage_input_feature_count": 2,
        "stage_keep_count": 1,
        "stage_full_mda_candidate_count": 2,
        "stage_screened_out_count": 0,
        "stage_screening_used": False,
        "stage_fit_calls": 0,
        "stage_permutation_predict_calls": 1,
        "stage_total_seconds": 0.25,
    }]
    assert feature_manifest["permutation_acceleration_contract"]["algorithm_version"] == (
        archetype.STAGED_PERMUTATION_ACCELERATION_VERSION
    )
    assert "path_arch" not in " ".join(feature_manifest["configured_universe"])
    hpo_manifest = json.loads((output / "hpo_manifest.json").read_text())
    assert hpo_manifest["feature_contract_frozen_before_hpo"] is True
    assert hpo_manifest["hpo_features"] == ["base_x"]
    assert hpo_manifest["no_improvement_patience_trials"] == 30
    resource = hpo_manifest["catboost_resource_contract"]
    assert resource["requested_thread_count"] == 4
    assert resource["effective_thread_count"] <= resource["requested_thread_count"]
    selection_contract = json.loads(
        (output / runner.FEATURE_SELECTION_HPO_CONTRACT_FILENAME).read_text()
    )
    assert (
        selection_contract["fingerprint_inputs"]["selection_hpo_settings"]
        ["catboost_resource_contract"]
        == resource
    )
    assert (
        selection_contract["fingerprint_inputs"]["selection_hpo_settings"]
        ["permutation_execution_contract"]["algorithm_version"]
        == archetype.STAGED_PERMUTATION_ACCELERATION_VERSION
    )
    assert selection_contract["fingerprint_inputs"]["selection_hpo_settings"][
        "hpo_sampling_contract"
    ]["version"] == runner.HPO_SAMPLING_CONTRACT_VERSION
    assert hpo_manifest["hpo_sampling_contract"]["purged_fold_support"]
    assert manifest["training_phase_order"] == [
        "fast_feature_selection",
        "permutation_feature_selection",
        "hpo_on_frozen_selected_features",
        "final_oof_and_refit",
    ]
    reused = runner.run_pipeline(
        frame,
        tmp_path / "reused_artifacts",
        discovery_end="2026-01-05T00:00:00Z",
        config_mapping=_config(),
        mandatory_features=["base_x"],
        hpo_trials=9,
        max_rows=900,
        smoke=True,
    )
    assert captured["selector_calls"] == 1
    assert captured["hpo_calls"] == 1
    assert reused["feature_selection_hpo_reuse"]["reused"] is True


def test_interrupted_hpo_resumes_exact_pre_hpo_selection_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = {"selector": 0, "permutation": 0, "hpo": 0}

    def fake_selector(*args: object, **kwargs: object) -> archetype.FastSelectorResult:
        calls["selector"] += 1
        return archetype.FastSelectorResult(
            selected_features=("base_x", "meta_x"),
            candidate_features=("base_x", "meta_x"),
            mandatory_features=("base_x",),
            availability={"base_x": 1.0, "meta_x": 1.0},
            scores=pd.DataFrame({"mi": [1.0, 0.5]}, index=["base_x", "meta_x"]),
            correlation_clusters=(("base_x",), ("meta_x",)),
            proxy_backend="test_proxy",
        )

    def fake_oof(features: pd.DataFrame, target: pd.Series, *args: object, **kwargs: object) -> archetype.OOFPathArchetypeResult:
        classes = np.asarray(sorted(pd.unique(target.astype(str))))
        return archetype.OOFPathArchetypeResult(
            probabilities=np.full((len(features), len(classes)), 1.0 / len(classes)),
            fold_ids=np.where(np.arange(len(features)) < len(features) // 4, -1, 0),
            folds=[], models=[], classes=classes, feature_columns=tuple(features.columns),
            diagnostics={"logloss": 1.0},
        )

    def fake_permutation(*args: object, **kwargs: object) -> tuple[list[str], pd.DataFrame]:
        calls["permutation"] += 1
        return ["base_x"], pd.DataFrame([{"stage": 32, "feature": "base_x", "selected": True}])

    def fake_hpo(*args: object, **kwargs: object) -> SimpleNamespace:
        calls["hpo"] += 1
        if calls["hpo"] == 1:
            raise RuntimeError("simulated HPO interruption")
        return SimpleNamespace(
            best_params={"iterations": 4}, report=lambda: {"best_params": {"iterations": 4}}
        )

    monkeypatch.setattr(archetype, "fast_select_preentry_features", fake_selector)
    monkeypatch.setattr(archetype, "fit_purged_chronological_oof_catboost", fake_oof)
    monkeypatch.setattr(archetype, "staged_permutation_selection", fake_permutation)
    monkeypatch.setattr(archetype, "optimize_purged_catboost_hpo", fake_hpo)
    monkeypatch.setattr(
        runner,
        "_fit_final_classifier",
        lambda *args, **kwargs: archetype.PathArchetypeClassifier(("base_x",), ("a", "b", "c", "d"), _PickleModel()),
    )

    output = tmp_path / "interrupted"
    kwargs = {
        "discovery_end": "2026-01-05T00:00:00Z",
        "config_mapping": _config(),
        "mandatory_features": ["base_x"],
        "hpo_trials": 1,
        "smoke": True,
    }
    with pytest.raises(RuntimeError, match="simulated HPO interruption"):
        runner.run_pipeline(_frame(), output, **kwargs)
    checkpoint = json.loads((output / "feature_selection_checkpoint.json").read_text())
    assert checkpoint["status"] == "feature_selection_complete"
    assert checkpoint["selected_features"] == ["base_x"]
    assert "catboost_resource_contract" in checkpoint
    assert {path.name for path in output.iterdir()} == {
        "feature_selection_checkpoint.json",
        runner.MDA_PROGRESS_FILENAME,
    }

    manifest = runner.run_pipeline(_frame(), output, **kwargs)
    assert calls == {"selector": 1, "permutation": 1, "hpo": 2}
    assert manifest["feature_selection_hpo_reuse"]["mode"] == "interrupted_hpo_resume"


def test_selection_and_hpo_proxies_do_not_leak_into_final_model_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    captured: dict[str, object] = {"oof_params": []}

    def fake_selector(*args: object, **kwargs: object) -> archetype.FastSelectorResult:
        return archetype.FastSelectorResult(
            selected_features=("base_x",), candidate_features=("base_x",),
            mandatory_features=("base_x",), availability={"base_x": 1.0},
            scores=pd.DataFrame({"mi": [1.0]}, index=["base_x"]),
            correlation_clusters=(("base_x",),), proxy_backend="test_proxy",
        )

    def fake_oof(
        features: pd.DataFrame, target: pd.Series, *args: object, **kwargs: object
    ) -> archetype.OOFPathArchetypeResult:
        captured["oof_params"].append(dict(kwargs.get("params") or {}))
        classes = np.asarray(sorted(pd.unique(target.astype(str))))
        return archetype.OOFPathArchetypeResult(
            probabilities=np.full((len(features), len(classes)), 1.0 / len(classes)),
            fold_ids=np.where(np.arange(len(features)) < len(features) // 4, -1, 0),
            folds=[], models=[], classes=classes, feature_columns=tuple(features.columns),
            diagnostics={"logloss": 1.0},
        )

    def fake_permutation(*args: object, **kwargs: object) -> tuple[list[str], pd.DataFrame]:
        return ["base_x"], pd.DataFrame([{"stage": 150, "feature": "base_x", "selected": True}])

    def fake_hpo(*args: object, **kwargs: object) -> SimpleNamespace:
        captured["hpo_iterations"] = kwargs["search_iterations"]
        captured["hpo_od_wait"] = kwargs["search_od_wait"]
        captured["hpo_storage"] = kwargs["storage"]
        captured["hpo_progress_path"] = kwargs["progress_path"]
        return SimpleNamespace(
            best_params={"iterations": 400, "od_wait": 40, "depth": 6},
            report=lambda: {"best_params": {"iterations": 400, "od_wait": 40}},
        )

    monkeypatch.setattr(archetype, "fast_select_preentry_features", fake_selector)
    monkeypatch.setattr(archetype, "fit_purged_chronological_oof_catboost", fake_oof)
    monkeypatch.setattr(archetype, "staged_permutation_selection", fake_permutation)
    monkeypatch.setattr(archetype, "optimize_purged_catboost_hpo", fake_hpo)
    monkeypatch.setattr(
        runner,
        "_fit_final_classifier",
        lambda *args, **kwargs: archetype.PathArchetypeClassifier(
            ("base_x",), ("a", "b", "c", "d"), _PickleModel()
        ),
    )

    output = tmp_path / "proxy"
    runner.run_pipeline(
        _frame(), output, discovery_end="2026-01-05T00:00:00Z",
        config_mapping=_config(), mandatory_features=["base_x"], hpo_trials=20,
        hpo_rows=8_000, hpo_folds=2, hpo_iterations=400, hpo_od_wait=40,
        selection_iterations=500, selection_od_wait=50,
    )

    assert captured["oof_params"][0] == {"iterations": 500, "od_wait": 50}
    assert captured["oof_params"][-1]["iterations"] == 3_000
    assert captured["oof_params"][-1]["od_wait"] == 150
    assert captured["hpo_iterations"] == 400
    assert captured["hpo_od_wait"] == 40
    assert str(captured["hpo_storage"]).endswith("/hpo_study.sqlite3")
    assert captured["hpo_progress_path"] == output / runner.HPO_PROGRESS_FILENAME
    report = json.loads((output / "training_report.json").read_text())
    assert report["effective_model_params"]["iterations"] == 3_000
    assert report["effective_model_params"]["od_wait"] == 150


def test_interrupted_mda_resumes_only_remaining_exact_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls = {"selector": 0, "mda": 0}

    def fake_selector(*args: object, **kwargs: object) -> archetype.FastSelectorResult:
        calls["selector"] += 1
        return archetype.FastSelectorResult(
            selected_features=("base_x",), candidate_features=("base_x",),
            mandatory_features=("base_x",), availability={"base_x": 1.0},
            scores=pd.DataFrame({"mi": [1.0]}, index=["base_x"]),
            correlation_clusters=(("base_x",),), proxy_backend="test_proxy",
        )

    def fake_oof(
        features: pd.DataFrame, target: pd.Series, *args: object, **kwargs: object
    ) -> archetype.OOFPathArchetypeResult:
        classes = np.asarray(sorted(pd.unique(target.astype(str))))
        return archetype.OOFPathArchetypeResult(
            probabilities=np.full((len(features), len(classes)), 1.0 / len(classes)),
            fold_ids=np.where(np.arange(len(features)) < len(features) // 4, -1, 0),
            folds=[], models=[], classes=classes, feature_columns=tuple(features.columns),
            diagnostics={"logloss": 1.0},
        )

    def fake_mda(*args: object, **kwargs: object) -> tuple[list[str], pd.DataFrame]:
        calls["mda"] += 1
        completed = kwargs["completed_stages"]
        if not completed:
            kwargs["stage_callback"]({
                "stage_index": 0,
                "stage": runner.SMOKE_PERMUTATION_STAGES[0],
                "input_features": ["base_x"],
                "selected_features": ["base_x"],
                "records": [{
                    "stage": runner.SMOKE_PERMUTATION_STAGES[0],
                    "feature": "base_x", "selected": True,
                }],
                "stage_total_seconds": 0.25,
            })
            raise RuntimeError("simulated MDA interruption")
        assert completed[0]["selected_features"] == ["base_x"]
        return ["base_x"], pd.DataFrame(completed[0]["records"])

    monkeypatch.setattr(archetype, "fast_select_preentry_features", fake_selector)
    monkeypatch.setattr(archetype, "fit_purged_chronological_oof_catboost", fake_oof)
    monkeypatch.setattr(archetype, "staged_permutation_selection", fake_mda)
    monkeypatch.setattr(
        archetype,
        "optimize_purged_catboost_hpo",
        lambda *args, **kwargs: SimpleNamespace(
            best_params={"depth": 6}, report=lambda: {"best_params": {"depth": 6}}
        ),
    )
    monkeypatch.setattr(
        runner,
        "_fit_final_classifier",
        lambda *args, **kwargs: archetype.PathArchetypeClassifier(
            ("base_x",), ("a", "b", "c", "d"), _PickleModel()
        ),
    )

    output = tmp_path / "mda_resume"
    kwargs = {
        "discovery_end": "2026-01-05T00:00:00Z",
        "config_mapping": _config(),
        "mandatory_features": ["base_x"],
        "hpo_trials": 1,
        "smoke": True,
    }
    with pytest.raises(RuntimeError, match="simulated MDA interruption"):
        runner.run_pipeline(_frame(), output, **kwargs)
    progress = json.loads((output / runner.MDA_PROGRESS_FILENAME).read_text())
    assert progress["status"] == "mda_running"
    assert len(progress["completed_stages"]) == 1
    assert not (output / "feature_selection_checkpoint.json").exists()

    manifest = runner.run_pipeline(_frame(), output, **kwargs)
    assert calls == {"selector": 1, "mda": 2}
    assert manifest["feature_selection_hpo_reuse"]["mode"] == "interrupted_mda_resume"


def test_intermediate_selection_checkpoint_requires_exact_current_contract(tmp_path: Path) -> None:
    path = tmp_path / "feature_selection_checkpoint.json"
    path.write_text(json.dumps({
        "schema": runner.RUNNER_SCHEMA,
        "status": "feature_selection_complete",
        "fingerprint": "stale",
        "selected_features": ["base_x"],
        "selection": {},
        "permutation": [],
    }))
    with pytest.raises(ValueError, match="does not match the current exact contract"):
        runner._read_resumable_feature_selection_checkpoint(path, "current")


def test_runner_rejects_realized_path_summary_in_configured_feature_universe(
    tmp_path: Path,
) -> None:
    config = _config()
    config["meta_shared_feature_keys"] = ["meta_x", "path_arch_efficiency"]
    with pytest.raises(ValueError, match="non-pre-entry"):
        runner.run_pipeline(
            _frame(40),
            tmp_path / "reject",
            discovery_end="2026-01-02T00:00:00Z",
            config_mapping=config,
            hpo_trials=1,
        )


def test_runner_rejects_array_only_future_path_semantics(tmp_path: Path) -> None:
    frame = _frame(40)
    path_columns = [
        column for column in frame.columns if column.startswith("path_arch_")
    ]
    frame = frame.drop(columns=path_columns)
    frame["future_path"] = [[0.1, 0.2, 0.3, 0.4]] * len(frame)
    with pytest.raises(ValueError, match="array-only future paths"):
        runner.run_pipeline(
            frame,
            tmp_path / "reject_array_path",
            discovery_end="2026-01-02T00:00:00Z",
            future_path_column="future_path",
            config_mapping=_config(),
            hpo_trials=1,
        )


def test_mandatory_feature_file_accepts_json_and_line_lists(tmp_path: Path) -> None:
    json_path = tmp_path / "mandatory.json"
    json_path.write_text('{"mandatory_features": ["base_x", "meta_x", "base_x"]}')
    text_path = tmp_path / "mandatory.txt"
    text_path.write_text("# comment\nbase_x\nmeta_x\n")
    assert runner._read_optional_list(json_path) == ["base_x", "meta_x"]
    assert runner._read_optional_list(text_path) == ["base_x", "meta_x"]


def _write_completed_selection_hpo_contract(
    directory: Path,
    *,
    fingerprint: str = "matching-fingerprint",
    selected_features: list[str] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / runner.FEATURE_SELECTION_HPO_CONTRACT_FILENAME
    path.write_text(
        json.dumps(
            {
                "schema": runner.FEATURE_SELECTION_HPO_CONTRACT_SCHEMA,
                "status": "feature_selection_hpo_complete",
                "fingerprint": fingerprint,
                "selected_features": selected_features or ["base_x"],
                "effective_model_params": {"iterations": 3000, "od_wait": 150},
                "selection": {"proxy_backend": "test"},
                "permutation": [],
                "hpo": {"best_params": {"iterations": 4}},
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_legacy_selection_hpo_run(
    directory: Path,
    *,
    candidate_hash: str = "candidate-hash",
    selected_features: list[str] | None = None,
) -> Path:
    selected = selected_features or ["base_x"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "feature_selection_checkpoint.json").write_text(
        json.dumps(
            {
                "schema": runner.RUNNER_SCHEMA,
                "status": "feature_selection_complete",
                "selected_features": selected,
                "selection": {"proxy_backend": "legacy-test"},
                "permutation": [{"feature": selected[0], "selected": True}],
            }
        ),
        encoding="utf-8",
    )
    (directory / "hpo_checkpoint.json").write_text(
        json.dumps(
            {
                "schema": runner.RUNNER_SCHEMA,
                "status": "hpo_complete",
                "selected_features": selected,
                "effective_model_params": {"iterations": 3000, "od_wait": 150},
                "hpo": {"best_params": {"depth": 6}},
            }
        ),
        encoding="utf-8",
    )
    (directory / "run_manifest.json").write_text(
        json.dumps({"candidate_identity_sha256": candidate_hash}),
        encoding="utf-8",
    )
    return directory


def _expected_selection_hpo_contract(
    *, candidate_hash: str = "candidate-hash"
) -> dict[str, object]:
    return {
        "schema": runner.FEATURE_SELECTION_HPO_CONTRACT_SCHEMA,
        "fingerprint": "strict-fingerprint",
        "fingerprint_inputs": {"candidate_identity_sha256": candidate_hash},
    }


def test_feature_selection_hpo_exact_registry_match_is_reused(tmp_path: Path) -> None:
    checkpoint = _write_completed_selection_hpo_contract(tmp_path / "prior")
    contract, provenance = runner._find_reusable_feature_selection_hpo_contract(
        {"fingerprint": "matching-fingerprint"},
        output_dir=tmp_path / "new_run",
        checkpoint_path=None,
        registry_root=tmp_path,
        force=False,
    )
    assert contract is not None
    assert provenance == {
        "mode": "automatic_registry",
        "reused": True,
        "registry_root": str(tmp_path),
        "path": str(checkpoint),
        "candidates": 1,
        "mismatched_candidates": 0,
    }


def test_selection_only_registry_reuses_completed_mda_after_hpo_contract_changes(
    tmp_path: Path,
) -> None:
    prior = tmp_path / "prior"
    prior.mkdir()
    fingerprint_inputs = {
        "candidate_identity_sha256": "candidate-hash",
        "selection_hpo_settings": {
            "selector_sample_rows": 45_000,
            "hpo_rows": 8_000,
        },
    }
    selection_fingerprint = runner._selection_only_fingerprint(fingerprint_inputs)
    (prior / runner.FEATURE_SELECTION_HPO_CONTRACT_FILENAME).write_text(
        json.dumps(
            {
                "schema": runner.FEATURE_SELECTION_HPO_CONTRACT_SCHEMA,
                "status": "feature_selection_hpo_complete",
                "fingerprint": "stale-hpo-fingerprint",
                "fingerprint_inputs": fingerprint_inputs,
                "selected_features": ["base_x"],
                "effective_model_params": {"iterations": 3000},
            }
        ),
        encoding="utf-8",
    )
    (prior / "feature_selection_checkpoint.json").write_text(
        json.dumps(
            {
                "schema": runner.RUNNER_SCHEMA,
                "status": "feature_selection_complete",
                "fingerprint": "old-combined-fingerprint",
                "selected_features": ["base_x"],
                "selection": {"proxy_backend": "test"},
                "permutation": [],
            }
        ),
        encoding="utf-8",
    )
    checkpoint, provenance = runner._find_reusable_feature_selection_checkpoint(
        selection_fingerprint,
        output_dir=tmp_path / "new",
        checkpoint_path=None,
        registry_root=tmp_path,
        force=False,
    )
    assert checkpoint is not None
    assert checkpoint["selected_features"] == ["base_x"]
    assert provenance["selection_only"] is True
    assert provenance["path"] == str(
        prior / runner.FEATURE_SELECTION_HPO_CONTRACT_FILENAME
    )


def test_explicit_legacy_run_is_adopted_into_current_strict_contract(
    tmp_path: Path,
) -> None:
    legacy_dir = _write_legacy_selection_hpo_run(tmp_path / "legacy")
    contract, provenance = runner._find_reusable_feature_selection_hpo_contract(
        _expected_selection_hpo_contract(),
        output_dir=tmp_path / "new_run",
        checkpoint_path=legacy_dir,
        registry_root=None,
        force=False,
    )
    assert contract is not None
    assert contract["fingerprint"] == "strict-fingerprint"
    assert contract["selected_features"] == ["base_x"]
    assert contract["effective_model_params"] == {
        "iterations": 3000,
        "od_wait": 150,
    }
    assert provenance["mode"] == "explicit_legacy_adoption"
    assert provenance["reused"] is True
    assert set(provenance["legacy_artifact_sha256"]) == {
        "feature_selection_checkpoint",
        "hpo_checkpoint",
        "run_manifest",
    }


def test_explicit_legacy_adoption_rejects_candidate_mismatch(tmp_path: Path) -> None:
    legacy_dir = _write_legacy_selection_hpo_run(
        tmp_path / "legacy", candidate_hash="different-candidate"
    )
    with pytest.raises(ValueError, match="candidate_identity_sha256 does not match"):
        runner._find_reusable_feature_selection_hpo_contract(
            _expected_selection_hpo_contract(),
            output_dir=tmp_path / "new_run",
            checkpoint_path=legacy_dir,
            registry_root=None,
            force=False,
        )


def test_registry_does_not_auto_adopt_legacy_checkpoints(tmp_path: Path) -> None:
    _write_legacy_selection_hpo_run(tmp_path / "legacy")
    contract, provenance = runner._find_reusable_feature_selection_hpo_contract(
        _expected_selection_hpo_contract(),
        output_dir=tmp_path / "new_run",
        checkpoint_path=None,
        registry_root=tmp_path,
        force=False,
    )
    assert contract is None
    assert provenance["candidates"] == 0
    assert provenance["mismatched_candidates"] == 0


def test_explicit_legacy_adoption_rejects_unavailable_selected_feature(
    tmp_path: Path,
) -> None:
    legacy_dir = _write_legacy_selection_hpo_run(
        tmp_path / "legacy", selected_features=["missing_x"]
    )
    contract, _ = runner._find_reusable_feature_selection_hpo_contract(
        _expected_selection_hpo_contract(),
        output_dir=tmp_path / "new_run",
        checkpoint_path=legacy_dir,
        registry_root=None,
        force=False,
    )
    assert contract is not None
    with pytest.raises(ValueError, match="selected features unavailable"):
        runner._validate_reused_selected_features(
            contract["selected_features"], ["base_x"]
        )


def test_feature_selection_hpo_mismatch_reruns_automatically_and_errors_explicitly(
    tmp_path: Path,
) -> None:
    checkpoint = _write_completed_selection_hpo_contract(
        tmp_path / "prior", fingerprint="old-fingerprint"
    )
    contract, provenance = runner._find_reusable_feature_selection_hpo_contract(
        {"fingerprint": "new-fingerprint"},
        output_dir=tmp_path / "new_run",
        checkpoint_path=None,
        registry_root=tmp_path,
        force=False,
    )
    assert contract is None
    assert provenance["reused"] is False
    assert provenance["mismatched_candidates"] == 1
    with pytest.raises(ValueError, match="fingerprint does not match"):
        runner._find_reusable_feature_selection_hpo_contract(
            {"fingerprint": "new-fingerprint"},
            output_dir=tmp_path / "new_run",
            checkpoint_path=checkpoint,
            registry_root=None,
            force=False,
        )


def test_feature_selection_hpo_force_skips_an_exact_checkpoint(tmp_path: Path) -> None:
    _write_completed_selection_hpo_contract(tmp_path / "prior")
    contract, provenance = runner._find_reusable_feature_selection_hpo_contract(
        {"fingerprint": "matching-fingerprint"},
        output_dir=tmp_path / "new_run",
        checkpoint_path=None,
        registry_root=tmp_path,
        force=True,
    )
    assert contract is None
    assert provenance == {"mode": "forced_rerun", "reused": False}


def test_reused_feature_selection_hpo_rejects_unavailable_selected_feature() -> None:
    with pytest.raises(ValueError, match="selected features unavailable"):
        runner._validate_reused_selected_features(["base_x", "missing_x"], ["base_x"])


def test_canonical_feature_dir_reads_bme_sample_then_selected_full_population(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[tuple[int, tuple[str, ...]]]] = {
        "selector": [],
        "hpo": [],
        "oof": [],
        "static": [],
    }

    class _StaticFeatures:
        def symbol_frame(self, symbol: str, keys: tuple[str, ...]) -> pd.DataFrame:
            assert symbol == "BTC/USD"
            index = pd.date_range("2026-01-01", periods=3000, freq="h", tz="UTC")
            return pd.DataFrame(
                {
                    "base_x": np.arange(3000, dtype=np.float32),
                    "meta_x": np.arange(3000, dtype=np.float32) % 7,
                    "unused_x": np.ones(3000, dtype=np.float32),
                },
                index=index,
            ).reindex(columns=list(keys))

    def fake_static_reader(**kwargs: object) -> _StaticFeatures:
        captured["static"].append(
            (
                len(kwargs["feature_keys"]),  # type: ignore[index]
                tuple(kwargs["feature_keys"]),  # type: ignore[index]
            )
        )
        return _StaticFeatures()

    def fake_selector(
        features: pd.DataFrame, *args: object, **kwargs: object
    ) -> archetype.FastSelectorResult:
        captured["selector"].append((len(features), tuple(features.columns)))
        return archetype.FastSelectorResult(
            selected_features=("base_x", "meta_x"),
            candidate_features=("base_x", "meta_x"),
            mandatory_features=(),
            availability={"base_x": 1.0, "meta_x": 1.0},
            scores=pd.DataFrame({"mi": [1.0, 0.5]}, index=["base_x", "meta_x"]),
            correlation_clusters=(("base_x",), ("meta_x",)),
            proxy_backend="test_proxy",
        )

    def fake_hpo(
        features: pd.DataFrame, *args: object, **kwargs: object
    ) -> SimpleNamespace:
        captured["hpo"].append((len(features), tuple(features.columns)))
        return SimpleNamespace(
            best_params={"iterations": 4},
            report=lambda: {"best_params": {"iterations": 4}},
        )

    def fake_oof(
        features: pd.DataFrame, target: pd.Series, *args: object, **kwargs: object
    ) -> archetype.OOFPathArchetypeResult:
        captured["oof"].append((len(features), tuple(features.columns)))
        classes = np.asarray(sorted(pd.unique(target.astype(str))))
        return archetype.OOFPathArchetypeResult(
            probabilities=np.full((len(features), len(classes)), 1.0 / len(classes)),
            fold_ids=np.where(
                np.arange(len(features)) < 75, -1, np.arange(len(features)) % 2
            ),
            folds=[],
            models=[],
            classes=classes,
            feature_columns=tuple(features.columns),
            diagnostics={"logloss": 1.0},
        )

    def fake_permutation(
        *args: object, **kwargs: object
    ) -> tuple[list[str], pd.DataFrame]:
        return ["base_x"], pd.DataFrame(
            [{"stage": 32, "feature": "base_x", "selected": True}]
        )

    def fake_final(
        *args: object, **kwargs: object
    ) -> archetype.PathArchetypeClassifier:
        return archetype.PathArchetypeClassifier(
            ("base_x",), ("a", "b", "c", "d"), _PickleModel()
        )

    monkeypatch.setattr(runner, "read_static_features", fake_static_reader)
    monkeypatch.setattr(
        runner,
        "_feature_store_schemas",
        lambda _feature_dir: (
            {"base_x", "meta_x", "unused_x"},
            {"symbol=BTC_USD.parquet": {"base_x", "meta_x", "unused_x"}},
        ),
    )
    monkeypatch.setattr(archetype, "fast_select_preentry_features", fake_selector)
    monkeypatch.setattr(archetype, "optimize_purged_catboost_hpo", fake_hpo)
    monkeypatch.setattr(archetype, "fit_purged_chronological_oof_catboost", fake_oof)
    monkeypatch.setattr(archetype, "staged_permutation_selection", fake_permutation)
    monkeypatch.setattr(runner, "_fit_final_classifier", fake_final)

    narrow = _frame(3000).drop(columns=["base_x", "meta_x"])
    narrow["path_arch_complete_24h"] = np.int8(1)
    input_path = tmp_path / "path_labels.parquet"
    narrow.to_parquet(input_path, index=False)
    feature_dir = tmp_path / "data" / "features" / "20260102_000000"
    feature_dir.mkdir(parents=True)
    output = tmp_path / "artifacts"
    config = _config()
    config["base_shared_feature_keys"] = ["base_x", "unused_x"]
    manifest = runner.run_pipeline(
        input_path,
        output,
        discovery_end="2026-01-05T00:00:00Z",
        feature_dir=feature_dir,
        config_mapping=config,
        hpo_trials=1,
        selection_rows=600,
    )

    assert captured["selector"] == [(600, ("base_x", "unused_x", "meta_x"))]
    # HPO must see only the contract frozen by the final permutation stage.
    assert captured["hpo"] == [(3000, ("base_x",))]
    assert captured["oof"] == [
        (600, ("base_x", "meta_x")),
        (3000, ("base_x",)),
    ]
    assert captured["static"] == [
        (3, ("base_x", "unused_x", "meta_x")),
        (3, ("base_x", "unused_x", "meta_x")),
        (3, ("base_x", "unused_x", "meta_x")),
        (1, ("base_x",)),
        (1, ("base_x",)),
    ]
    availability = json.loads(
        (output / "feature_availability_manifest.json").read_text(encoding="utf-8")
    )
    assert availability["configured_features_absent_from_schema"] == []
    assert availability["selection_sample_contract"].startswith("deterministic")
    assert (
        manifest["artifacts"]["feature_availability"]
        == "feature_availability_manifest.json"
    )


def test_canonical_runner_lets_frozen_aegmm_compete_without_forcing_them(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[tuple[int, tuple[str, ...]]]] = {
        "selector": [],
        "hpo": [],
        "oof": [],
        "static": [],
    }
    captured_mandatory: list[tuple[str, ...]] = []

    class _StaticFeatures:
        def symbol_frame(self, _symbol: str, keys: tuple[str, ...]) -> pd.DataFrame:
            index = pd.date_range("2026-01-01", periods=300, freq="h", tz="UTC")
            return pd.DataFrame(
                {
                    "base_x": np.arange(300, dtype=np.float32),
                    "meta_x": np.arange(300, dtype=np.float32),
                },
                index=index,
            ).reindex(columns=list(keys))

    def fake_static_reader(**kwargs: object) -> _StaticFeatures:
        captured["static"].append(
            (len(kwargs["feature_keys"]), tuple(kwargs["feature_keys"]))
        )  # type: ignore[index]
        return _StaticFeatures()

    def fake_selector(
        features: pd.DataFrame, *args: object, **kwargs: object
    ) -> archetype.FastSelectorResult:
        captured["selector"].append((len(features), tuple(features.columns)))
        captured_mandatory.append(tuple(kwargs["mandatory_features"]))
        return archetype.FastSelectorResult(
            selected_features=("base_x", *AE_GMM_FEATURE_COLUMNS),
            candidate_features=("base_x", *AE_GMM_FEATURE_COLUMNS),
            mandatory_features=tuple(AE_GMM_FEATURE_COLUMNS),
            availability={
                "base_x": 1.0,
                **{name: 1.0 for name in AE_GMM_FEATURE_COLUMNS},
            },
            scores=pd.DataFrame({"mi": 1.0}, index=["base_x", *AE_GMM_FEATURE_COLUMNS]),
            correlation_clusters=(
                ("base_x",),
                *[(name,) for name in AE_GMM_FEATURE_COLUMNS],
            ),
            proxy_backend="test_proxy",
        )

    def fake_hpo(
        features: pd.DataFrame, *args: object, **kwargs: object
    ) -> SimpleNamespace:
        captured["hpo"].append((len(features), tuple(features.columns)))
        return SimpleNamespace(
            best_params={"iterations": 4},
            report=lambda: {"best_params": {"iterations": 4}},
        )

    def fake_oof(
        features: pd.DataFrame, target: pd.Series, *args: object, **kwargs: object
    ) -> archetype.OOFPathArchetypeResult:
        captured["oof"].append((len(features), tuple(features.columns)))
        classes = np.asarray(sorted(pd.unique(target.astype(str))))
        return archetype.OOFPathArchetypeResult(
            probabilities=np.full((len(features), len(classes)), 1.0 / len(classes)),
            fold_ids=np.where(
                np.arange(len(features)) < 75, -1, np.arange(len(features)) % 2
            ),
            folds=[],
            models=[],
            classes=classes,
            feature_columns=tuple(features.columns),
            diagnostics={"logloss": 1.0},
        )

    def fake_permutation(
        *args: object, **kwargs: object
    ) -> tuple[list[str], pd.DataFrame]:
        return list(AE_GMM_FEATURE_COLUMNS), pd.DataFrame(
            [
                {"stage": 32, "feature": name, "selected": True}
                for name in AE_GMM_FEATURE_COLUMNS
            ]
        )

    def fake_final(
        *args: object, **kwargs: object
    ) -> archetype.PathArchetypeClassifier:
        return archetype.PathArchetypeClassifier(
            tuple(AE_GMM_FEATURE_COLUMNS), ("a", "b", "c", "d"), _PickleModel()
        )

    monkeypatch.setattr(runner, "read_static_features", fake_static_reader)
    monkeypatch.setattr(
        runner,
        "_feature_store_schemas",
        lambda _path: (
            {"base_x", "meta_x"},
            {"symbol=BTC.parquet": {"base_x", "meta_x"}},
        ),
    )
    monkeypatch.setattr(archetype, "fast_select_preentry_features", fake_selector)
    monkeypatch.setattr(archetype, "optimize_purged_catboost_hpo", fake_hpo)
    monkeypatch.setattr(archetype, "fit_purged_chronological_oof_catboost", fake_oof)
    monkeypatch.setattr(archetype, "staged_permutation_selection", fake_permutation)
    monkeypatch.setattr(runner, "_fit_final_classifier", fake_final)

    frame = _frame(300)
    input_path = tmp_path / "path_labels.parquet"
    frame.drop(columns=["base_x", "meta_x"]).to_parquet(input_path, index=False)
    sidecar = tmp_path / "frozen.parquet"
    _write_frozen_ae_gmm_sidecar(sidecar, frame, shuffle=True)
    feature_dir = tmp_path / "data" / "features" / "20260102_000000"
    feature_dir.mkdir(parents=True)
    output = tmp_path / "artifacts"
    manifest = runner.run_pipeline(
        input_path,
        output,
        discovery_end="2026-01-05T00:00:00Z",
        feature_dir=feature_dir,
        frozen_ae_gmm_sidecar=sidecar,
        config_mapping=_config(),
        hpo_trials=1,
    )

    assert captured["selector"] == [
        (300, ("base_x", "meta_x", *AE_GMM_FEATURE_COLUMNS))
    ]
    assert captured_mandatory == [()]
    # The staged selector removes base_x, so HPO is restricted to the frozen
    # AE/GMM-only final contract rather than the wider preselection contract.
    assert captured["hpo"] == [(300, tuple(AE_GMM_FEATURE_COLUMNS))]
    assert captured["oof"] == [
        (300, ("base_x", *AE_GMM_FEATURE_COLUMNS)),
        (300, tuple(AE_GMM_FEATURE_COLUMNS)),
    ]
    assert captured["static"] == [(2, ("base_x", "meta_x"))]
    availability = json.loads(
        (output / "feature_availability_manifest.json").read_text()
    )
    frozen_contract = availability["frozen_ae_gmm_sidecar"]
    assert frozen_contract["matched_rows"] == 300
    assert frozen_contract["missing_rows"] == 0
    assert manifest["rows"] == 300


def test_sparse_class_consolidation_preserves_shape_and_uses_nearest_strength() -> None:
    labels = pd.Series(
        [
            "fast_clean_impulse__below_150atr",
            *(["fast_clean_impulse__atr200_300"] * 4),
            *(["fast_clean_impulse__atr500_plus"] * 3),
        ],
        dtype="string",
    )
    effective, report = runner._consolidate_sparse_supervised_classes(
        labels,
        pd.Series(True, index=labels.index),
        min_class_rows=3,
    )
    assert effective.iloc[0] == "fast_clean_impulse__atr200_300"
    row = report.loc[
        report["raw_path_archetype"].eq("fast_clean_impulse__below_150atr")
    ].iloc[0]
    assert bool(row["was_consolidated"])
    assert row["effective_discovery_rows"] == 4


def test_parquet_input_requires_feature_dir(tmp_path: Path) -> None:
    input_path = tmp_path / "path_labels.parquet"
    _frame(40).to_parquet(input_path, index=False)
    with pytest.raises(ValueError, match="requires --feature-dir"):
        runner.run_pipeline(
            input_path,
            tmp_path / "artifacts",
            discovery_end="2026-01-02T00:00:00Z",
            config_mapping=_config(),
            hpo_trials=1,
        )


def test_runner_rejects_missing_or_stale_deterministic_path_target(
    tmp_path: Path,
) -> None:
    missing = _frame(40).drop(columns=["path_archetype", "path_archetype_rule_version"])
    with pytest.raises(ValueError, match="requires the frozen deterministic"):
        runner.run_pipeline(
            missing,
            tmp_path / "missing",
            discovery_end="2026-01-02T00:00:00Z",
            config_mapping=_config(),
            hpo_trials=1,
        )

    stale = _frame(40)
    stale["path_archetype_rule_version"] = "economic_path_v2"
    with pytest.raises(ValueError, match="stale path_archetype_rule_version"):
        runner.run_pipeline(
            stale,
            tmp_path / "stale",
            discovery_end="2026-01-02T00:00:00Z",
            config_mapping=_config(),
            hpo_trials=1,
        )
