from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import refit_package_side_specific_meta_champion as runner
from scripts.run_s52_train_meta_regime_handoff_smoke import (
    BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN,
    BASE_TARGET_CONTRACT_HASH_COLUMN,
    HANDOFF_RANK_SCOPE_COLUMN,
    _contract_hash,
)


class _DeterministicModel:
    def __init__(self, side: str) -> None:
        self.side = side

    def predict(self, matrix: pd.DataFrame) -> np.ndarray:
        return matrix.iloc[:, 0].to_numpy(dtype=np.float32) + (
            0.1 if self.side == "long" else -0.1
        )


def _base_contract() -> dict[str, object]:
    target = {
        "schema": "base_soft_label_contract_v1",
        "target_column": "target_soft",
        "target_mode": "first_touch_target_soft",
    }
    weights = {
        "schema": "target_strength_weight_v1",
        "spec": {"exponent": 1.5, "weight_range_ratio": 4.0},
    }
    return {
        "candidate_handoff_rank_scope": "timestamp_side",
        "base_target_contract": target,
        BASE_TARGET_CONTRACT_HASH_COLUMN: _contract_hash(target),
        "base_sample_weight_spec": weights,
        BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN: _contract_hash(weights),
        "validation_status": "strict_pass",
    }


def _write_frozen_columns_contract(
    path: Path,
    *,
    numeric: list[str],
    categorical: list[str],
) -> None:
    entries = [
        {
            "feature": feature,
            "source_column": feature,
            "source_type": "materialized_numeric_or_boolean",
        }
        for feature in numeric
    ]
    entries.extend(
        {
            "feature": f"{source}_example",
            "source_column": source,
            "source_type": "categorical_one_hot",
        }
        for source in categorical
    )
    path.write_text(
        json.dumps(
            {
                "input_feature_contract": {"entries": entries},
                "preprocessing_state": {
                    "numeric_columns": numeric,
                    "categorical_source_columns": categorical,
                },
            }
        ),
        encoding="utf-8",
    )


def _frozen_preprocessing(*, numeric: list[str], categorical: list[str]) -> dict[str, list[str]]:
    return {
        "numeric_columns": numeric,
        "categorical_source_columns": categorical,
    }


def _write_smoke_fixture(tmp_path: Path) -> tuple[Path, Path, Path, pd.DataFrame]:
    handoff = tmp_path / "train_meta_regime_handoff.parquet"
    ledger = tmp_path / "s52_trailing_regime_scored_ledger.parquet"
    handoff.touch()
    ledger.touch()
    inherited = _base_contract()
    (tmp_path / "train_meta_regime_handoff_contract.json").write_text(
        json.dumps({"inherited_base_contract": inherited}), encoding="utf-8"
    )
    rows: list[dict[str, object]] = []
    for side_index, side in enumerate(("long", "short")):
        for index in range(55):
            rows.append(
                {
                    "__ts__": pd.Timestamp("2026-06-01", tz="UTC")
                    + pd.Timedelta(hours=index + side_index * 100),
                    "__symbol__": f"{side[:1].upper()}{index:03d}",
                    "side_name": side,
                    "score": float(index) / 100.0,
                    "target_soft": 0.2 + float(index % 7) / 10.0,
                    "selected_top30": True,
                    HANDOFF_RANK_SCOPE_COLUMN: "timestamp_side",
                    BASE_TARGET_CONTRACT_HASH_COLUMN: inherited[
                        BASE_TARGET_CONTRACT_HASH_COLUMN
                    ],
                    BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN: inherited[
                        BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN
                    ],
                }
            )
    frame = pd.DataFrame(rows)
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    columns_path = smoke / "columns.json"
    _write_frozen_columns_contract(columns_path, numeric=["score"], categorical=[])
    manifest = {
        "generated_by": "run_s52_train_meta_regime_handoff_smoke",
        "frontier": "top30",
        "strict_handoff_contract": True,
        "meta_head_mode": "single_base_soft_label",
        "side_specific_feature_contract_enabled": True,
        "selected_features_by_side": {"long": ["score"], "short": ["score"]},
        "classifier_params": {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 5,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        "inherited_base_handoff_contract": inherited,
        "handoff_path": str(handoff),
        "ledger_path": str(ledger),
        "saved_fold_models": [{"columns_path": str(columns_path)}],
    }
    (smoke / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return smoke, handoff, ledger, frame


def test_final_refit_packages_separate_side_models_with_non_oos_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke, _handoff, _ledger, frame = _write_smoke_fixture(tmp_path)
    monkeypatch.setattr(runner, "_projected_handoff_columns_for_selected", lambda *_a: ["score"])
    monkeypatch.setattr(runner, "_load_joined_frame", lambda *_a, **_k: frame.copy())
    captured: list[dict[str, object]] = []

    def fake_fit(matrix, target, train, seed, **kwargs):
        captured.append(
            {
                "side": train["side_name"].iloc[0],
                "rows": len(train),
                "features": list(matrix.columns),
                "strict": kwargs["strict_handoff_contract"],
                "weight_spec": kwargs["target_strength_weight_spec"],
            }
        )
        kwargs["weight_diagnostics_out"].update({"schema": "captured"})
        return _DeterministicModel(str(train["side_name"].iloc[0]))

    monkeypatch.setattr(runner, "_fit_base_soft_label_model", fake_fit)
    output = tmp_path / "package"
    manifest = runner.run_final_refit(smoke_result=smoke, output_dir=output, seed=17)

    assert [item["side"] for item in captured] == ["long", "short"]
    assert [item["rows"] for item in captured] == [55, 55]
    assert all(item["features"] == ["score"] for item in captured)
    assert all(item["strict"] is True for item in captured)
    assert manifest["excluded_from_oos_metrics"] is True
    assert manifest["all_permitted_rows_refit"] is True
    assert manifest["permitted_row_accounting"]["permitted_rows"] == 110
    assert manifest["score_reference_and_side_comparability"][
        "raw_scores_directly_comparable_across_sides"
    ] is False
    for side in ("long", "short"):
        metadata = json.loads((output / side / "metadata.json").read_text())
        features = json.loads((output / side / "features.json").read_text())
        assert (output / side / "base_soft_label.joblib").is_file()
        assert metadata["leakage_contract"]["excluded_from_oos_metrics"] is True
        assert metadata["score_reference"]["oos"] is False
        assert len(metadata["model_sha256"]) == 64
        assert len(metadata["features_sha256"]) == 64
        assert metadata["target_strength_weight_diagnostics"] == {"schema": "captured"}
        assert metadata["inherited_base_contract"][BASE_TARGET_CONTRACT_HASH_COLUMN] == _base_contract()[BASE_TARGET_CONTRACT_HASH_COLUMN]
        assert metadata["inherited_base_contract"][BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN] == _base_contract()[BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN]
        assert features["feature_contract_hash"] == runner._feature_contract_hash(["score"])


def test_final_refit_fails_closed_when_row_base_hash_is_mixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke, _handoff, _ledger, frame = _write_smoke_fixture(tmp_path)
    frame.loc[0, BASE_TARGET_CONTRACT_HASH_COLUMN] = "mismatched"
    monkeypatch.setattr(runner, "_projected_handoff_columns_for_selected", lambda *_a: ["score"])
    monkeypatch.setattr(runner, "_load_joined_frame", lambda *_a, **_k: frame.copy())

    with pytest.raises(ValueError, match="missing or mixed base contract hashes"):
        runner.run_final_refit(smoke_result=smoke, output_dir=tmp_path / "package")


def test_final_refit_rejects_non_side_specific_or_non_strict_smoke_manifest(
    tmp_path: Path
) -> None:
    smoke, _handoff, _ledger, _frame = _write_smoke_fixture(tmp_path)
    manifest_path = smoke / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["side_specific_feature_contract_enabled"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="side-specific META feature contract"):
        runner.run_final_refit(smoke_result=smoke, output_dir=tmp_path / "package")


def test_final_refit_rejects_stale_smoke_base_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    smoke, _handoff, _ledger, frame = _write_smoke_fixture(tmp_path)
    manifest_path = smoke / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inherited_base_handoff_contract"][BASE_TARGET_CONTRACT_HASH_COLUMN] = "stale"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runner, "_projected_handoff_columns_for_selected", lambda *_a: ["score"])
    monkeypatch.setattr(runner, "_load_joined_frame", lambda *_a, **_k: frame.copy())

    with pytest.raises(ValueError, match="does not match the strict handoff sidecar"):
        runner.run_final_refit(smoke_result=smoke, output_dir=tmp_path / "package")


def test_final_refit_persists_train_derived_selected_ood_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke, _handoff, _ledger, frame = _write_smoke_fixture(tmp_path)
    frame["f1"] = np.linspace(-1.0, 1.0, len(frame), dtype=np.float32)
    frame["f2"] = np.linspace(2.0, -2.0, len(frame), dtype=np.float32)
    manifest_path = smoke / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selected_features_by_side"] = {
        "long": ["score", "f1", "f2", "meta_sel_ood_abs_z_mean"],
        "short": ["score", "f1", "f2", "meta_sel_ood_abs_z_max"],
    }
    columns_path = smoke / "columns.json"
    _write_frozen_columns_contract(
        columns_path,
        numeric=["score", "f1", "f2"],
        categorical=[],
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "_projected_handoff_columns_for_selected",
        lambda *_a: ["score", "f1", "f2"],
    )
    monkeypatch.setattr(runner, "_load_joined_frame", lambda *_a, **_k: frame.copy())
    monkeypatch.setattr(
        runner,
        "_fit_base_soft_label_model",
        lambda matrix, _target, train, _seed, **kwargs: (
            kwargs["weight_diagnostics_out"].update({"schema": "captured"})
            or _DeterministicModel(str(train["side_name"].iloc[0]))
        ),
    )

    output = tmp_path / "package"
    runner.run_final_refit(smoke_result=smoke, output_dir=output, seed=17)

    for side, selected_ood in (
        ("long", "meta_sel_ood_abs_z_mean"),
        ("short", "meta_sel_ood_abs_z_max"),
    ):
        model = runner.joblib.load(output / side / "base_soft_label.joblib")
        reference = model.s52_meta_ood_reference_
        assert reference["enabled"] is True
        assert reference["feature_names"] == ["score", "f1", "f2"]
        features = json.loads((output / side / "features.json").read_text())
        assert features["preprocessing"]["post_selection_ood_outputs"] == [
            "meta_sel_ood_abs_z_mean",
            "meta_sel_ood_abs_z_max",
        ]
        assert selected_ood in features["feature_names"]


def test_final_matrix_builds_shared_priors_and_frozen_ood_before_side_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "side_name": ["long", "short", "long", "short"],
            "f1": [0.0, 1.0, 2.0, 3.0],
            "f2": [1.0, 2.0, 3.0, 4.0],
            "f3": [4.0, 3.0, 2.0, 1.0],
        }
    )
    calls: list[int] = []

    def passthrough_prior(train, valid, **_kwargs):
        calls.append(len(train))
        return train.copy(), valid.copy()

    def passthrough_reliability(train, valid):
        calls.append(len(train))
        return train.copy(), valid.copy()

    def fake_make_xy(train, valid, *, selected_features, preprocessing_state_out, **_kwargs):
        preprocessing_state_out["schema"] = "test"
        return (
            train.reindex(columns=selected_features).copy(),
            valid.reindex(columns=selected_features).copy(),
            list(selected_features),
        )

    monkeypatch.setattr(runner, "_add_fold_base_prior_features", passthrough_prior)
    monkeypatch.setattr(runner, "_add_fold_reliability_features", passthrough_reliability)
    monkeypatch.setattr(runner, "_make_xy", fake_make_xy)
    selected = {
        "long": ["f1", "f2", "f3", "meta_sel_ood_abs_z_mean"],
        "short": ["f1", "f2", "f3", "meta_sel_ood_abs_z_max"],
    }

    matrix, preprocessing, reference = runner._final_training_matrix(
        frame,
        selected,
        frozen_preprocessing=_frozen_preprocessing(
            numeric=["f1", "f2", "f3"], categorical=[]
        ),
    )

    assert calls == [4, 4]
    assert list(matrix.columns) == [
        "f1",
        "f2",
        "f3",
        "meta_sel_ood_abs_z_mean",
        "meta_sel_ood_abs_z_max",
    ]
    assert reference["enabled"] is True
    assert reference["feature_names"] == ["f1", "f2", "f3"]
    assert preprocessing["post_selection_ood_outputs"] == [
        "meta_sel_ood_abs_z_mean",
        "meta_sel_ood_abs_z_max",
    ]
    assert np.isfinite(matrix.to_numpy(dtype=np.float32)).all()
    assert not np.allclose(matrix["meta_sel_ood_abs_z_mean"].to_numpy(), 0.0)


def test_final_matrix_fails_closed_when_selected_ood_cannot_be_reproduced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "side_name": ["long", "short"],
            "f1": [0.0, 1.0],
            "f2": [1.0, 0.0],
        }
    )
    monkeypatch.setattr(
        runner,
        "_add_fold_base_prior_features",
        lambda train, valid, **_kwargs: (train.copy(), valid.copy()),
    )
    monkeypatch.setattr(
        runner,
        "_add_fold_reliability_features",
        lambda train, valid: (train.copy(), valid.copy()),
    )

    with pytest.raises(ValueError, match="cannot reproduce selected post-selection OOD"):
        runner._final_training_matrix(
            frame,
            {
                "long": ["f1", "f2", "meta_sel_ood_abs_z_mean"],
                "short": ["f1", "f2", "meta_sel_ood_abs_z_mean"],
            },
            frozen_preprocessing=_frozen_preprocessing(
                numeric=["f1", "f2"], categorical=[]
            ),
        )


def test_final_matrix_uses_frozen_categorical_sources_not_prefix_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "side_name": ["long", "short"],
            "base_margin_to_cutoff": ["low", "high"],
            "base_margin_to_cutoff_z": [0.25, -0.50],
        }
    )
    monkeypatch.setattr(
        runner,
        "_add_fold_base_prior_features",
        lambda train, valid, **_kwargs: (train.copy(), valid.copy()),
    )
    monkeypatch.setattr(
        runner,
        "_add_fold_reliability_features",
        lambda train, valid: (train.copy(), valid.copy()),
    )
    selected = {
        "long": [
            "base_margin_to_cutoff_z",
            "base_margin_to_cutoff_low",
        ],
        "short": [
            "base_margin_to_cutoff_z",
            "base_margin_to_cutoff_high",
        ],
    }

    matrix, preprocessing, _ = runner._final_training_matrix(
        frame,
        selected,
        frozen_preprocessing=_frozen_preprocessing(
            numeric=["base_margin_to_cutoff_z"],
            categorical=["base_margin_to_cutoff"],
        ),
    )

    assert list(matrix.columns) == [
        "base_margin_to_cutoff_z",
        "base_margin_to_cutoff_low",
        "base_margin_to_cutoff_high",
    ]
    assert matrix.columns.is_unique
    assert preprocessing["numeric_columns"] == ["base_margin_to_cutoff_z"]
    assert preprocessing["categorical_source_columns"] == ["base_margin_to_cutoff"]
