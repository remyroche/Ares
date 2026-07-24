from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_execution_ev_catboost_refinement_oof",
    ROOT / "scripts" / "materialize_execution_ev_catboost_refinement_oof.py",
)
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)

JOINER_SPEC = importlib.util.spec_from_file_location(
    "materialize_execution_ev_joined_handoff",
    ROOT / "scripts" / "materialize_execution_ev_joined_handoff.py",
)
assert JOINER_SPEC and JOINER_SPEC.loader
joiner = importlib.util.module_from_spec(JOINER_SPEC)
sys.modules[JOINER_SPEC.name] = joiner
JOINER_SPEC.loader.exec_module(joiner)

CLASSES = adapter.RAW_CLASSES


def _signed(payload: dict[str, object]) -> dict[str, object]:
    payload.pop("prediction_role_manifest_sha256", None)
    payload["prediction_role_manifest_sha256"] = adapter._canonical_json_hash(payload)
    return payload


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    timestamps = pd.to_datetime(["2026-03-01T00:00:00Z", "2026-03-01T01:00:00Z"])
    probabilities = np.full((2, len(CLASSES)), 0.05)
    probabilities[0, 0] = 0.70
    probabilities[1, 4] = 0.70
    entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
    ordered = np.sort(probabilities, axis=1)
    class_index = {name: index for index, name in enumerate(CLASSES)}
    predictions = pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD"],
            "side": ["1", "-1"],
            "candidate_id": ["oos-0", "oos-1"],
            "config_id": adapter.RETAINED_GEOMETRY_ID,
            "fold_id": [0, 1],
            "available_at": timestamps,
            "validation_start": timestamps,
            "train_cutoff_utc": pd.to_datetime(["2026-02-28T23:00:00Z"] * 2),
            "train_decision_cutoff": pd.to_datetime(["2026-02-28T23:00:00Z"] * 2),
            "label_resolution_available_at": pd.to_datetime(["2026-02-28T22:00:00Z"] * 2),
            "oos_start_utc": timestamps,
            "oos_end_utc": pd.to_datetime(["2026-07-01T00:00:00Z"] * 2),
            "true_dynamic_label": ["dead_timeout", "slow_grinder"],
            "realized_net_ev_after_1pct_return": [-0.01, 0.02],
            "predicted_class": [CLASSES[0], CLASSES[4]],
            "probability_vector": probabilities.tolist(),
            "probability_entropy": entropy,
            "raw_max_probability": probabilities.max(axis=1),
            "raw_normalized_entropy": entropy / np.log(len(CLASSES)),
            "raw_top1_top2_probability_margin": ordered[:, -1] - ordered[:, -2],
            "raw_adverse_probability_mass": probabilities[
                :, [class_index[name] for name in adapter.ADVERSE_CLASSES]
            ].sum(axis=1),
            "raw_favorable_probability_mass": probabilities[
                :, [class_index[name] for name in adapter.FAVORABLE_CLASSES]
            ].sum(axis=1),
            "calibrated_predicted_class": [CLASSES[1], CLASSES[3]],
            **{
                f"probability_{name}": probabilities[:, index]
                for index, name in enumerate(CLASSES)
            },
            **{
                f"calibrated_probability_{name}": probabilities[:, index]
                for index, name in enumerate(CLASSES)
            },
        }
    )
    paths = {
        "predictions": tmp_path / "raw_oos_predictions.parquet",
        "source_role_manifest": tmp_path / "raw_oos_predictions.role_manifest.json",
        "source_manifest": tmp_path / "exact_geometry_oos_predictions_manifest.json",
        "output": tmp_path / "adapted.parquet",
        "manifest": tmp_path / "adapted.manifest.json",
    }
    predictions.to_parquet(paths["predictions"], index=False)
    source_role = _signed(
        {
            "schema": "path_archetype_oof_prediction_role_v1",
            "prediction_role": adapter.RAW_SOURCE_ROLE,
            "source_artifact_sha256": adapter._sha256(paths["predictions"]),
            "class_order": list(CLASSES),
            "prediction_columns": {
                f"probability_{name}": {
                    "role": adapter.RAW_SOURCE_PROBABILITY_ROLE,
                    "target": False,
                }
                for name in CLASSES
            },
        }
    )
    paths["source_role_manifest"].write_text(
        json.dumps(source_role, indent=2, sort_keys=True), encoding="utf-8"
    )
    source_manifest = {
        "schema": adapter.EXACT_GEOMETRY_SCHEMA,
        "config_id": adapter.RETAINED_GEOMETRY_ID,
        "hard_label_target": adapter.RAW_HARD_LABEL_TARGET,
        "class_order": list(CLASSES),
        "class_merge": adapter.FAST_WINNER_CLASS_MERGE,
        "sample_weight_contract": adapter.RAW_SAMPLE_WEIGHT_CONTRACT,
        "probability_output": adapter.RAW_PROBABILITY_OUTPUT,
        "calibration_required_for_raw_output": False,
        "centroid_required_for_raw_output": False,
        "prediction_sha256": adapter._sha256(paths["predictions"]),
        "prediction_role_manifest": paths["source_role_manifest"].name,
        "prediction_role_manifest_sha256": adapter._sha256(
            paths["source_role_manifest"]
        ),
    }
    paths["source_manifest"].write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return paths


def _run(paths: dict[str, Path]) -> dict[str, Path]:
    return adapter.run(
        paths["predictions"],
        paths["source_manifest"],
        paths["output"],
        paths["manifest"],
    )


def test_emits_raw_merged_seven_class_oof_contract_and_signed_manifest(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    result = _run(paths)
    output = pd.read_parquet(result["oof"])
    manifest = json.loads(result["manifest"].read_text(encoding="utf-8"))

    probability_columns = [f"probability__{name}" for name in CLASSES]
    assert output.columns.tolist() == [
        "__ts__",
        "__symbol__",
        "side_name",
        "candidate_id",
        *probability_columns,
        "probability_entropy",
        "max_probability",
        "normalized_entropy",
        "top2_probability_margin",
        "adverse_probability_mass",
        "favorable_probability_mass",
        "predicted_path_archetype",
        "oof_fold",
        "available_at",
        "validation_start",
        "train_decision_cutoff",
        "label_resolution_available_at",
    ]
    assert output["candidate_id"].tolist() == ["oos-0", "oos-1"]
    assert output["side_name"].tolist() == ["long", "short"]
    assert output["oof_fold"].tolist() == ["0", "1"]
    assert output["predicted_path_archetype"].tolist() == [CLASSES[0], CLASSES[4]]
    np.testing.assert_allclose(output[probability_columns].sum(axis=1), 1.0)
    np.testing.assert_allclose(output["max_probability"], output[probability_columns].max(axis=1))
    np.testing.assert_allclose(
        output["normalized_entropy"], output["probability_entropy"] / np.log(7)
    )
    np.testing.assert_allclose(
        output["top2_probability_margin"],
        np.diff(np.sort(output[probability_columns].to_numpy(), axis=1)[:, -2:]).ravel(),
    )
    np.testing.assert_allclose(
        output["adverse_probability_mass"],
        output[[f"probability__{name}" for name in adapter.ADVERSE_CLASSES]].sum(axis=1),
    )
    np.testing.assert_allclose(
        output["favorable_probability_mass"],
        output[[f"probability__{name}" for name in adapter.FAVORABLE_CLASSES]].sum(axis=1),
    )
    assert not any("calibrated" in name for name in output.columns)
    assert "true_dynamic_label" not in output
    assert not any(column.startswith("realized_") for column in output)

    assert manifest["prediction_role"] == "path_archetype_oof"
    assert manifest["path_shape_types"] == list(CLASSES)
    assert manifest["class_names"] == list(CLASSES)
    assert manifest["class_order_sha256"] == adapter._class_order_sha256(CLASSES)
    assert manifest["raw_probability_contract"] == {
        "source_prediction_role": adapter.RAW_SOURCE_ROLE,
        "retained_geometry_id": adapter.RETAINED_GEOMETRY_ID,
        "hard_label_target": adapter.RAW_HARD_LABEL_TARGET,
        "class_merge": adapter.FAST_WINNER_CLASS_MERGE,
        "sample_weight_contract": adapter.RAW_SAMPLE_WEIGHT_CONTRACT,
        "probability_output": adapter.RAW_PROBABILITY_OUTPUT,
        "calibration_required_for_raw_output": False,
        "centroid_required_for_raw_output": False,
    }
    assert set(manifest["prediction_columns"]) == {
        *probability_columns,
        "probability_entropy",
        *adapter.RAW_DERIVED_COLUMNS,
        "predicted_path_archetype",
    }
    assert all(
        declaration["target"] is False
        for declaration in manifest["prediction_columns"].values()
    )
    assert manifest["source_artifact_sha256"] == adapter._sha256(paths["output"])
    assert manifest["raw_probability_derivations"]["adverse_probability_mass"] == {
        "classes": list(adapter.ADVERSE_CLASSES),
        "formula": "sum(raw probability over adverse classes)",
    }
    assert manifest["raw_probability_derivations"]["favorable_probability_mass"] == {
        "classes": list(adapter.FAVORABLE_CLASSES),
        "formula": "sum(raw probability over favorable classes)",
    }
    assert manifest["raw_probability_derivations"]["neutral_classes"] == list(
        adapter.NEUTRAL_CLASSES
    )
    assert manifest["target_columns_emitted"] == []
    assert manifest["prediction_role_manifest_sha256"] == adapter._canonical_json_hash(
        manifest, excluded=("prediction_role_manifest_sha256",)
    )
    downstream_manifest = joiner._load_signed_prediction_manifest(
        joiner.SourceSpec(
            name="catboost",
            path=result["oof"],
            timestamp_col="__ts__",
            symbol_col="__symbol__",
            side_col="side_name",
            candidate_id_col="candidate_id",
            available_at_col="available_at",
            fold_col="oof_fold",
            validation_start_col="validation_start",
            train_decision_cutoff_col="train_decision_cutoff",
            label_resolution_available_at_col="label_resolution_available_at",
            manifest_path=result["manifest"],
            prediction_role="path_archetype_oof",
            output_columns={},
            require_oof_fold=True,
        )
    )
    assert downstream_manifest["class_contract"]["class_order"] == list(CLASSES)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("class_order", ["fast_clean_winner", *CLASSES[1:]], "class_order"),
        ("class_merge", {}, "class_merge"),
        ("sample_weight_contract", "economic_weights_v1", "sample_weight_contract"),
        ("probability_output", "calibrated_probabilities", "probability_output"),
        ("calibration_required_for_raw_output", True, "calibration_required"),
    ],
)
def test_rejects_noncanonical_raw_source_contract(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    paths = _artifacts(tmp_path)
    manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    manifest[field] = value
    paths["source_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _run(paths)


def test_rejects_calibrated_or_refinement_source_role(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    role = json.loads(paths["source_role_manifest"].read_text(encoding="utf-8"))
    role["prediction_role"] = "path_archetype_oof"
    _signed(role)
    paths["source_role_manifest"].write_text(json.dumps(role), encoding="utf-8")
    manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    manifest["prediction_role_manifest_sha256"] = adapter._sha256(
        paths["source_role_manifest"]
    )
    paths["source_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="prediction_role"):
        _run(paths)


@pytest.mark.parametrize("failure", ["normalization", "vector", "entropy", "argmax"])
def test_rejects_invalid_raw_probability_contract(
    tmp_path: Path, failure: str
) -> None:
    paths = _artifacts(tmp_path)
    predictions = pd.read_parquet(paths["predictions"])
    if failure == "normalization":
        predictions.loc[0, f"probability_{CLASSES[0]}"] = 0.60
        match = "not normalized"
    elif failure == "vector":
        predictions.at[0, "probability_vector"] = [1.0] + [0.0] * (len(CLASSES) - 1)
        match = "does not match"
    elif failure == "entropy":
        predictions.loc[0, "probability_entropy"] += 0.01
        match = "entropy"
    else:
        predictions.loc[0, "predicted_class"] = CLASSES[1]
        match = "argmax"
    predictions.to_parquet(paths["predictions"], index=False)
    with pytest.raises(ValueError, match="bind the OOS parquet"):
        _run(paths)

    # Rebind the test fixture to exercise the row-level raw contract.
    role = json.loads(paths["source_role_manifest"].read_text(encoding="utf-8"))
    role["source_artifact_sha256"] = adapter._sha256(paths["predictions"])
    _signed(role)
    paths["source_role_manifest"].write_text(json.dumps(role), encoding="utf-8")
    manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    manifest["prediction_sha256"] = adapter._sha256(paths["predictions"])
    manifest["prediction_role_manifest_sha256"] = adapter._sha256(
        paths["source_role_manifest"]
    )
    paths["source_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _run(paths)


@pytest.mark.parametrize("source_column", adapter.RAW_DERIVED_COLUMNS.values())
def test_rejects_raw_probability_summary_mismatch(
    tmp_path: Path, source_column: str
) -> None:
    paths = _artifacts(tmp_path)
    predictions = pd.read_parquet(paths["predictions"])
    predictions.loc[0, source_column] += 0.01
    predictions.to_parquet(paths["predictions"], index=False)
    role = json.loads(paths["source_role_manifest"].read_text(encoding="utf-8"))
    role["source_artifact_sha256"] = adapter._sha256(paths["predictions"])
    _signed(role)
    paths["source_role_manifest"].write_text(json.dumps(role), encoding="utf-8")
    manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    manifest["prediction_sha256"] = adapter._sha256(paths["predictions"])
    manifest["prediction_role_manifest_sha256"] = adapter._sha256(
        paths["source_role_manifest"]
    )
    paths["source_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    output_column = next(
        key for key, value in adapter.RAW_DERIVED_COLUMNS.items() if value == source_column
    )
    with pytest.raises(ValueError, match=output_column):
        _run(paths)


def test_rejects_noncausal_raw_provenance(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    predictions = pd.read_parquet(paths["predictions"])
    predictions.loc[0, "train_decision_cutoff"] = pd.Timestamp("2026-03-01T00:00:00Z")
    predictions.to_parquet(paths["predictions"], index=False)
    role = json.loads(paths["source_role_manifest"].read_text(encoding="utf-8"))
    role["source_artifact_sha256"] = adapter._sha256(paths["predictions"])
    _signed(role)
    paths["source_role_manifest"].write_text(json.dumps(role), encoding="utf-8")
    manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    manifest["prediction_sha256"] = adapter._sha256(paths["predictions"])
    manifest["prediction_role_manifest_sha256"] = adapter._sha256(
        paths["source_role_manifest"]
    )
    paths["source_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="strict OOS provenance"):
        _run(paths)
