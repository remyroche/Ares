from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements import packb_side_local_fs_hpo_stage as stage
from extreme_price_movements import packb_side_stage_manifest as manifest


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


SOURCE_REVISION = "a" * 40
CALENDAR_SHA256 = _sha("locked-calendar")
BASE_SOURCE_HASHES = {
    "dec09_decisions_sha256": _sha("dec09"),
    "canonical_shard_inventory_sha256": _sha("inventory"),
    "causal_audit_sha256": _sha("audit"),
    "population_preflight_sha256": _sha("preflight"),
    "feature_store_inventory_sha256": _sha("store"),
    "feature_store_inventory_evidence_sha256": _sha("store-evidence"),
}


class _RecordingGuard:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def preflight(self, name: str) -> None:
        self.calls.append(("preflight", name))

    def checkpoint(self, name: str) -> None:
        self.calls.append(("checkpoint", name))


def _row(candidate_id: str, signal: str) -> dict[str, object]:
    timestamp = pd.Timestamp(signal)
    decision = timestamp + pd.Timedelta(hours=1)
    return {
        "candidate_id": candidate_id,
        "side_name": "long",
        "__ts__": timestamp,
        "__decision_ts__": decision,
        "__label_resolution_ts__": decision + pd.Timedelta(hours=24),
        "__symbol__": "BTCUSDT",
    }


def _frame(prefix: str, signals: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [_row(f"{prefix}-{index}", signal) for index, signal in enumerate(signals)]
    )


def _write(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_parquet(path, index=False)
    return path


def _feature_provenance(features: list[str]) -> dict[str, dict[str, str]]:
    return {
        feature: {
            "causal_definition_sha256": _sha(f"{feature}:causal"),
            "inference_availability_sha256": _sha(f"{feature}:inference"),
            "units_contract_sha256": _sha(f"{feature}:units"),
        }
        for feature in features
    }


def _inputs(
    tmp_path: Path, *, invalid_fs_validation: bool = False
) -> dict[str, object]:
    fs_train = _frame(
        "fs-train",
        ["2025-10-20T00:00:00Z", "2025-10-21T00:00:00Z", "2025-10-22T00:00:00Z"],
    )
    fs_valid = _frame(
        "fs-valid",
        ["2025-12-01T00:00:00Z"]
        if invalid_fs_validation
        else [
            "2025-11-10T00:00:00Z",
            "2025-11-11T00:00:00Z",
            "2025-11-12T00:00:00Z",
        ],
    )
    folds_data = [
        (
            _frame("hpo1-train", ["2025-10-20T00:00:00Z", "2025-10-21T00:00:00Z"]),
            _frame("hpo1-valid", ["2025-12-10T00:00:00Z", "2025-12-11T00:00:00Z"]),
        ),
        (
            _frame("hpo2-train", ["2025-12-20T00:00:00Z", "2025-12-21T00:00:00Z"]),
            _frame("hpo2-valid", ["2026-01-10T00:00:00Z", "2026-01-11T00:00:00Z"]),
        ),
        (
            _frame("hpo3-train", ["2026-01-20T00:00:00Z", "2026-01-21T00:00:00Z"]),
            _frame("hpo3-valid", ["2026-02-10T00:00:00Z", "2026-02-11T00:00:00Z"]),
        ),
    ]
    all_frames = [fs_train, fs_valid, *(item for pair in folds_data for item in pair)]
    population = pd.concat(all_frames, ignore_index=True).drop_duplicates(
        subset=list(stage.REQUIRED_LEDGER_COLUMNS)
    )
    population_path = _write(tmp_path / "population.parquet", population)
    hpo_folds = []
    for index, (train, valid) in enumerate(folds_data, start=1):
        hpo_folds.append(
            stage.HPOFoldLedger(
                name=f"hpo_{index}",
                train_ledger=train,
                train_ledger_path=_write(
                    tmp_path / f"hpo_{index}_train.parquet", train
                ),
                valid_ledger=valid,
                valid_ledger_path=_write(
                    tmp_path / f"hpo_{index}_valid.parquet", valid
                ),
            )
        )
    return {
        "fs_train": fs_train,
        "fs_train_path": _write(tmp_path / "fs_train.parquet", fs_train),
        "fs_valid": fs_valid,
        "fs_valid_path": _write(tmp_path / "fs_valid.parquet", fs_valid),
        "hpo_folds": hpo_folds,
        "population_path": population_path,
        "source_hashes": {
            **BASE_SOURCE_HASHES,
            "authorized_population_ledger_sha256": manifest.sha256_file(
                population_path
            ),
        },
    }


def _run(
    tmp_path: Path,
    *,
    inputs: dict[str, object] | None = None,
    selection_result: dict[str, object] | None = None,
    loader=None,
    caps: bool = False,
    published_output_dir: Path | None = None,
) -> tuple[dict[str, object], dict[str, object], _RecordingGuard]:
    active = _inputs(tmp_path) if inputs is None else inputs
    seen: dict[str, object] = {"loader": [], "hpo": [], "target": [], "weight": []}
    guard = _RecordingGuard()

    def default_loader(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        seen["loader"].append((list(rows["candidate_id"]), list(columns)))
        values = np.arange(len(rows), dtype=np.float32)
        return pd.DataFrame(
            {column: values + position for position, column in enumerate(columns)}
        )

    def target(rows: pd.DataFrame) -> pd.Series:
        seen["target"].append(list(rows["candidate_id"]))
        return pd.Series(np.arange(len(rows), dtype=np.int64) % 2)

    def weight(rows: pd.DataFrame, target_values: pd.Series) -> pd.Series:
        seen["weight"].append((list(rows["candidate_id"]), list(target_values)))
        return pd.Series(np.ones(len(rows), dtype=np.float64))

    def selector(value: stage.FeatureSelectionInput) -> dict[str, object]:
        seen["selector"] = value
        return selection_result or {
            "side": "long",
            "selected_features": ["observable_a"],
            "selection_scope": "side_local",
            "fallback_used": False,
            "selection_methods": ["univariate_prescreen", "relief", "mda"],
            "search_breadth": 7,
        }

    def evaluator(value: stage.HPOFoldInput) -> dict[str, object]:
        seen["hpo"].append(
            (value.fold_name, value.trial.trial_id, len(value.train.ledger))
        )
        return {
            "objective": float(value.trial.params["depth"]),
            "fold": value.fold_name,
        }

    def choose(results: tuple[stage.HPOTrialEvaluation, ...]) -> dict[str, object]:
        seen["selection_results"] = results
        return {
            "side": "long",
            "selected_trial_id": "deep",
            "selected_params": {"depth": 5},
            "selection_scope": "side_local",
            "fallback_used": False,
            "selection_metric": "mean_side_local_validation_objective",
        }

    report = stage.fit_side_local_fs_hpo_stages(
        side="long",
        fs_train_ledger=active["fs_train"],
        fs_train_ledger_path=active["fs_train_path"],
        fs_valid_ledger=active["fs_valid"],
        fs_valid_ledger_path=active["fs_valid_path"],
        hpo_folds=active["hpo_folds"],
        authorized_population_ledger_path=active["population_path"],
        feature_loader=default_loader if loader is None else loader,
        target_loader=target,
        weight_loader=weight,
        candidate_features=["observable_a", "observable_b"],
        feature_provenance=_feature_provenance(["observable_a", "observable_b"]),
        feature_selection_callback=selector,
        hpo_trials=(
            stage.HPOTrial("shallow", {"depth": 3}),
            stage.HPOTrial("deep", {"depth": 5}),
        ),
        hpo_trial_evaluator=evaluator,
        hpo_selection_callback=choose,
        output_dir=tmp_path / "out",
        published_output_dir=published_output_dir,
        source_hashes=active["source_hashes"],
        source_revision=SOURCE_REVISION,
        fixed_calendar_sha256=CALENDAR_SHA256,
        extra_provenance_hashes={
            "raw_universe_sha256": _sha("raw-universe"),
            "feature_loader_contract_sha256": _sha("loader-contract"),
        },
        fs_train_max_rows=2 if caps else stage.DEFAULT_FS_TRAIN_MAX_ROWS,
        fs_valid_max_rows=2 if caps else stage.DEFAULT_FS_VALID_MAX_ROWS,
        hpo_train_max_rows=1 if caps else stage.DEFAULT_HPO_TRAIN_MAX_ROWS,
        hpo_valid_max_rows=1 if caps else stage.DEFAULT_HPO_VALID_MAX_ROWS,
        resource_guard=guard,
    )
    return report, seen, guard


def test_returned_paths_survive_outer_atomic_publication(
    tmp_path: Path,
) -> None:
    published = tmp_path / "published"
    report, _seen, _guard = _run(
        tmp_path,
        caps=True,
        published_output_dir=published,
    )
    (tmp_path / "out").rename(published)

    assert Path(str(report["summary_path"])).is_file()
    for stage_name in ("feature_selection", "hpo"):
        for key, value in report[stage_name].items():
            if key.endswith("_path"):
                assert Path(str(value)).is_file()
                assert str(value).startswith(str(published))


def test_runs_side_local_fixed_calendar_stages_with_real_sampled_evidence(
    tmp_path: Path,
) -> None:
    report, seen, guard = _run(tmp_path, caps=True)

    assert report["status"] == "FROZEN_SIDE_LOCAL_FEATURE_SELECTION_AND_HPO"
    assert report["selected_features"] == ["observable_a"]
    assert report["selected_params"] == {"depth": 5}
    # Two November raw loads plus two loads for each of three folds: no reload
    # for every HPO trial.
    assert len(seen["loader"]) == 8
    assert len(seen["hpo"]) == 6
    assert {fold for fold, _trial, _rows in seen["hpo"]} == {"hpo_1", "hpo_2", "hpo_3"}
    assert any("before_hpo_trial_evaluation" in value for _kind, value in guard.calls)
    assert any("before_persist" in value for _kind, value in guard.calls)

    fs_manifest = Path(str(report["feature_selection"]["manifest_path"]))
    hpo_manifest = Path(str(report["hpo"]["manifest_path"]))
    fs_evidence = manifest.validate_side_stage_manifest(
        fs_manifest,
        expected_side="long",
        expected_stage="feature_selection",
    )
    hpo_evidence = manifest.validate_side_stage_manifest(
        hpo_manifest,
        expected_side="long",
        expected_stage="hpo",
    )
    assert fs_evidence["candidate_stream"]["count"] == 4
    assert hpo_evidence["candidate_stream"]["count"] == 3
    fs_config = json.loads(
        Path(str(report["feature_selection"]["config_path"])).read_text(
            encoding="utf-8"
        )
    )
    assert fs_config["details"]["sample_caps"]["train_max_rows"] == 2
    assert fs_config["extra_provenance_hashes"]["raw_universe_sha256"] == _sha(
        "raw-universe"
    )
    assert fs_config["details"]["coverage"]["policy"]["global_fallback"] == "FORBIDDEN"
    assert set(fs_config["details"]["dataset_sha256"]) == {"train", "validation"}
    assert len(fs_config["details"]["dataset_sha256"]["train"]) == 64
    hpo_config = json.loads(
        Path(str(report["hpo"]["config_path"])).read_text(encoding="utf-8")
    )
    assert set(hpo_config["details"]["dataset_sha256_by_fold"]) == {
        "hpo_1",
        "hpo_2",
        "hpo_3",
    }


def test_rejects_calendar_before_feature_target_or_weight_callbacks(
    tmp_path: Path,
) -> None:
    active = _inputs(tmp_path, invalid_fs_validation=True)
    calls: list[str] = []

    with pytest.raises(
        stage.PackBSideLocalFSHPOStageError, match="locked validation calendar"
    ):
        stage.fit_side_local_fs_hpo_stages(
            side="long",
            fs_train_ledger=active["fs_train"],
            fs_train_ledger_path=active["fs_train_path"],
            fs_valid_ledger=active["fs_valid"],
            fs_valid_ledger_path=active["fs_valid_path"],
            hpo_folds=active["hpo_folds"],
            authorized_population_ledger_path=active["population_path"],
            feature_loader=lambda *_args: calls.append("feature"),
            target_loader=lambda *_args: calls.append("target"),
            weight_loader=lambda *_args: calls.append("weight"),
            candidate_features=["observable_a"],
            feature_provenance=_feature_provenance(["observable_a"]),
            feature_selection_callback=lambda *_args: {},
            hpo_trials=(
                stage.HPOTrial("one", {"depth": 1}),
                stage.HPOTrial("two", {"depth": 2}),
            ),
            hpo_trial_evaluator=lambda *_args: 0.0,
            hpo_selection_callback=lambda *_args: {},
            output_dir=tmp_path / "out",
            source_hashes=active["source_hashes"],
            source_revision=SOURCE_REVISION,
            fixed_calendar_sha256=CALENDAR_SHA256,
            resource_guard=_RecordingGuard(),
        )
    assert calls == []


def test_requires_side_local_mda_not_univariate_or_default_fallback(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        stage.PackBSideLocalFSHPOStageError, match="include side-local MDA"
    ):
        _run(
            tmp_path,
            selection_result={
                "side": "long",
                "selected_features": ["observable_a"],
                "selection_scope": "side_local",
                "fallback_used": False,
                "selection_methods": ["univariate"],
                "search_breadth": 1,
            },
        )
    assert not (tmp_path / "out").exists()


def test_rejects_selected_feature_below_98_percent_in_a_side_hpo_window(
    tmp_path: Path,
) -> None:
    def sparse_loader(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        result = pd.DataFrame(
            {column: np.ones(len(rows), dtype=np.float32) for column in columns}
        )
        if columns == ["observable_a"] and any(
            candidate.startswith("hpo2") for candidate in rows["candidate_id"]
        ):
            result.loc[0, "observable_a"] = np.nan
        return result

    with pytest.raises(
        stage.PackBSideLocalFSHPOStageError, match="coverage is below 98%"
    ):
        _run(tmp_path, loader=sparse_loader)


def test_rejects_callback_mutation_before_another_hpo_trial(
    tmp_path: Path,
) -> None:
    active = _inputs(tmp_path)

    def feature_loader(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {column: np.arange(len(rows), dtype=np.float32) for column in columns}
        )

    def selector(_value: stage.FeatureSelectionInput) -> dict[str, object]:
        return {
            "side": "long",
            "selected_features": ["observable_a"],
            "selection_scope": "side_local",
            "fallback_used": False,
            "selection_methods": ["mda"],
            "search_breadth": 2,
        }

    def mutating_evaluator(value: stage.HPOFoldInput) -> float:
        value.train.features.iloc[0, 0] = 999.0
        return 0.0

    with pytest.raises(
        stage.PackBSideLocalFSHPOStageError,
        match="mutated its read-only side-local dataset inputs",
    ):
        stage.fit_side_local_fs_hpo_stages(
            side="long",
            fs_train_ledger=active["fs_train"],
            fs_train_ledger_path=active["fs_train_path"],
            fs_valid_ledger=active["fs_valid"],
            fs_valid_ledger_path=active["fs_valid_path"],
            hpo_folds=active["hpo_folds"],
            authorized_population_ledger_path=active["population_path"],
            feature_loader=feature_loader,
            target_loader=lambda rows: pd.Series(np.zeros(len(rows), dtype=np.int64)),
            weight_loader=lambda rows, _target: pd.Series(
                np.ones(len(rows), dtype=np.float64)
            ),
            candidate_features=["observable_a", "observable_b"],
            feature_provenance=_feature_provenance(["observable_a", "observable_b"]),
            feature_selection_callback=selector,
            hpo_trials=(
                stage.HPOTrial("one", {"depth": 1}),
                stage.HPOTrial("two", {"depth": 2}),
            ),
            hpo_trial_evaluator=mutating_evaluator,
            hpo_selection_callback=lambda _results: {},
            output_dir=tmp_path / "out",
            source_hashes=active["source_hashes"],
            source_revision=SOURCE_REVISION,
            fixed_calendar_sha256=CALENDAR_SHA256,
            resource_guard=_RecordingGuard(),
        )
    assert not (tmp_path / "out").exists()
