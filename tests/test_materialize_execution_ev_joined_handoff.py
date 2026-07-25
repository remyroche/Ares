from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_execution_ev_joined_handoff",
    ROOT / "scripts" / "materialize_execution_ev_joined_handoff.py",
)
assert SPEC and SPEC.loader
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_execution_ev_meta_for_materializer_test",
    ROOT / "scripts" / "run_execution_ev_meta.py",
)
assert RUNNER_SPEC and RUNNER_SPEC.loader
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)


def _keys(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-07-01", periods=rows, freq="h", tz="UTC"),
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD"] * (rows // 2),
            "side_name": ["LONG", "short"] * (rows // 2),
            "candidate_id": [f"candidate-{index}" for index in range(rows)],
        }
    )


def _inputs(
    tmp_path: Path,
    *,
    class_order: tuple[str, ...] = materializer.MERGED_PATH_ARCHETYPE_CLASSES,
    manifest_class_order: bool = False,
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    keys = _keys()
    alpha = keys.assign(
        existing_alpha_ev=np.linspace(0.01, 0.04, len(keys)),
        base_prediction_uncertainty=0.2,
        meta_leaf_support_log1p=2.0,
        base_archetype_label__archetype_label_family__trend__abc123=np.array(
            [1.0, 0.0, 1.0, 0.0], dtype=np.float32
        ),
        oof_fold=[0, 0, 1, 1],
    )
    time = keys.assign(pred_time_to_first_meaningful_mfe_12h=4.0, oof_fold=[0, 0, 1, 1])
    peak = keys.assign(pred_peak_mfe_12h_atr=0.5, oof_fold=[0, 0, 1, 1])
    mae = keys.assign(prediction=0.4, oof_fold=[0, 0, 1, 1])
    turn = keys.assign(prediction=2.0, oof_fold=[0, 0, 1, 1])
    slope = keys.assign(prediction=0.8, oof_fold=[0, 0, 1, 1])
    class_count = len(class_order)
    probabilities = np.full((len(keys), class_count), 0.3 / (class_count - 1))
    winning_classes = np.arange(len(keys)) % class_count
    probabilities[np.arange(len(keys)), winning_classes] = 0.7
    entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
    ordered_probabilities = np.sort(probabilities, axis=1)
    class_index = {name: index for index, name in enumerate(class_order)}
    catboost = keys.assign(
        **{
            f"probability__{shape}": probabilities[:, index]
            for index, shape in enumerate(class_order)
        },
        probability_entropy=entropy,
        max_probability=probabilities.max(axis=1),
        normalized_entropy=entropy / np.log(class_count),
        top2_probability_margin=(
            ordered_probabilities[:, -1] - ordered_probabilities[:, -2]
        ),
        adverse_probability_mass=probabilities[
            :,
            [class_index[name] for name in materializer.CATBOOST_ADVERSE_CLASSES],
        ].sum(axis=1),
        favorable_probability_mass=probabilities[
            :,
            [class_index[name] for name in materializer.CATBOOST_FAVORABLE_CLASSES],
        ].sum(axis=1),
        predicted_path_archetype=[class_order[index] for index in winning_classes],
        oof_fold_id=[0, 0, 1, 1],
    )
    gross = np.linspace(-0.007, 0.023, len(keys))
    labels = keys.assign(
        __decision_ts__=keys["__ts__"] + pd.Timedelta(hours=1),
        execution_gross_ev_12h=gross,
        execution_cost_return=0.003,
        execution_net_ev_12h=gross - 0.003,
        execution_label_end_utc=keys["__ts__"] + pd.Timedelta(hours=13),
        execution_exit_reason=["timeout", "full_stop", "trailing", "timeout"],
        execution_exit_hour=[12, 3, 5, 12],
        execution_mfe_return_12h=[0.01, 0.002, 0.04, 0.01],
        execution_mae_return_12h=[-0.005, -0.03, -0.004, -0.008],
        execution_label_available_at=keys["__ts__"] + pd.Timedelta(hours=13),
    )
    oof_evidence = {
        "available_at": keys["__ts__"],
        "validation_start": keys["__ts__"],
        "train_decision_cutoff": keys["__ts__"] - pd.Timedelta(hours=1),
        "label_resolution_available_at": keys["__ts__"] - pd.Timedelta(hours=2),
    }
    for frame in (alpha, time, peak, mae, turn, slope, catboost):
        for column, values in oof_evidence.items():
            frame[column] = values
    frames = {
        "alpha": alpha,
        "time": time,
        "peak": peak,
        "mae": mae,
        "turn": turn,
        "slope": slope,
        "catboost": catboost,
        "labels": labels,
    }
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    roles = {
        "alpha": "alpha_ev_oof",
        "time": "time_to_mfe_oof",
        "peak": "peak_mfe_oof",
        "mae": "mae_before_mfe_oof",
        "turn": "adverse_turn_oof",
        "slope": "path_slope_oof",
        "catboost": "path_archetype_oof",
        "labels": "execution_ev_12h_labels",
    }
    for name, role in roles.items():
        payload: dict[str, object] = {
            "prediction_role": role,
            "source_artifact_sha256": materializer._sha256(paths[name]),
        }
        if name == "alpha":
            payload["prediction_columns"] = {
                "existing_alpha_ev": {
                    "role": "pre_entry_alpha_ev_oof_prediction",
                    "target": False,
                }
            }
            payload["alpha_cost_basis"] = {
                "deducted_cost_return": 0.01,
                "cost_unit": "return",
                "target_semantics": "residual_net_ev_after_1pct",
                "source_manifest": {
                    "path": "legacy_residual_manifest.json",
                    "sha256": "a" * 64,
                },
                "source_manifest_evidence": [
                    {
                        "field": "residual_expert_target",
                        "value": "ev_after_1pct - train-only expected EV",
                    }
                ],
            }
        if name == "catboost":
            payload["prediction_columns"] = {
                f"probability__{shape}": {
                    "role": "pre_entry_path_archetype_oof_prediction",
                    "target": False,
                }
                for shape in class_order
            }
            payload["prediction_columns"].update(
                {
                    column: {
                        "role": "pre_entry_path_archetype_oof_raw_probability_summary",
                        "target": False,
                    }
                    for column in (
                        "probability_entropy",
                        "max_probability",
                        "normalized_entropy",
                        "top2_probability_margin",
                        "adverse_probability_mass",
                        "favorable_probability_mass",
                        "predicted_path_archetype",
                    )
                }
            )
            if manifest_class_order:
                payload["path_shape_types"] = list(class_order)
        payload["prediction_role_manifest_sha256"] = materializer._canonical_json_hash(
            payload
        )
        manifest = tmp_path / f"{name}.manifest.json"
        manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        paths[f"{name}_manifest"] = manifest
    return paths


def _args(
    tmp_path: Path, paths: dict[str, Path], **overrides: object
) -> SimpleNamespace:
    values: dict[str, object] = {
        "alpha": paths["alpha"],
        "time_oof": paths["time"],
        "peak_oof": paths["peak"],
        "mae_oof": paths["mae"],
        "turn_oof": paths["turn"],
        "slope_oof": paths["slope"],
        "catboost_oof": paths["catboost"],
        "execution_labels": paths["labels"],
        "output": tmp_path / "joined.parquet",
        "provenance_json": tmp_path / "joined.json",
        "timestamp_col": "__ts__",
        "symbol_col": "__symbol__",
        "side_col": "side_name",
        "candidate_id_col": "candidate_id",
        "alpha_timestamp_col": None,
        "alpha_symbol_col": None,
        "alpha_side_col": None,
        "alpha_candidate_id_col": None,
        "time_timestamp_col": None,
        "time_symbol_col": None,
        "time_side_col": None,
        "time_candidate_id_col": None,
        "peak_timestamp_col": None,
        "peak_symbol_col": None,
        "peak_side_col": None,
        "peak_candidate_id_col": None,
        "mae_timestamp_col": None,
        "mae_symbol_col": None,
        "mae_side_col": None,
        "mae_candidate_id_col": None,
        "turn_timestamp_col": None,
        "turn_symbol_col": None,
        "turn_side_col": None,
        "turn_candidate_id_col": None,
        "slope_timestamp_col": None,
        "slope_symbol_col": None,
        "slope_side_col": None,
        "slope_candidate_id_col": None,
        "catboost_timestamp_col": None,
        "catboost_symbol_col": None,
        "catboost_side_col": None,
        "catboost_candidate_id_col": None,
        "labels_timestamp_col": None,
        "labels_symbol_col": None,
        "labels_side_col": None,
        "labels_candidate_id_col": None,
        "alpha_ev_col": "existing_alpha_ev",
        "alpha_uncertainty_col": "base_prediction_uncertainty",
        "alpha_leaf_support_col": "meta_leaf_support_log1p",
        "alpha_fold_col": "oof_fold",
        "alpha_available_at_col": None,
        "time_prediction_col": "pred_time_to_first_meaningful_mfe_12h",
        "time_fold_col": "oof_fold",
        "time_available_at_col": "available_at",
        "peak_prediction_col": "pred_peak_mfe_12h_atr",
        "peak_fold_col": "oof_fold",
        "peak_available_at_col": "available_at",
        "mae_prediction_col": "prediction",
        "mae_fold_col": "oof_fold",
        "mae_available_at_col": "available_at",
        "turn_prediction_col": "prediction",
        "turn_fold_col": "oof_fold",
        "turn_available_at_col": "available_at",
        "slope_prediction_col": "prediction",
        "slope_fold_col": "oof_fold",
        "slope_available_at_col": "available_at",
        "catboost_prob_cols": [
            f"probability__{shape}"
            for shape in materializer.MERGED_PATH_ARCHETYPE_CLASSES
        ],
        "catboost_entropy_col": "probability_entropy",
        "catboost_max_probability_col": "max_probability",
        "catboost_normalized_entropy_col": "normalized_entropy",
        "catboost_top2_margin_col": "top2_probability_margin",
        "catboost_adverse_mass_col": "adverse_probability_mass",
        "catboost_favorable_mass_col": "favorable_probability_mass",
        "catboost_archetype_col": "predicted_path_archetype",
        "catboost_fold_col": "oof_fold_id",
        "catboost_available_at_col": "available_at",
        "execution_decision_ts_col": "__decision_ts__",
        "execution_gross_ev_col": "execution_gross_ev_12h",
        "execution_cost_return_col": "execution_cost_return",
        "execution_net_ev_col": "execution_net_ev_12h",
        "execution_label_end_col": "execution_label_end_utc",
        "execution_exit_reason_col": "execution_exit_reason",
        "execution_exit_hour_col": "execution_exit_hour",
        "execution_mfe_col": "execution_mfe_return_12h",
        "execution_mae_col": "execution_mae_return_12h",
        "alpha_source_cost_return": 0.01,
    }
    for source in ("alpha", "time", "peak", "mae", "turn", "slope", "catboost"):
        values[f"{source}_manifest"] = paths[f"{source}_manifest"]
        values[f"{source}_validation_start_col"] = "validation_start"
        values[f"{source}_train_decision_cutoff_col"] = "train_decision_cutoff"
        values[f"{source}_label_resolution_available_at_col"] = (
            "label_resolution_available_at"
        )
    values.update(
        {
            "alpha_available_at_col": "available_at",
            "labels_manifest": paths["labels_manifest"],
            "labels_available_at_col": "execution_label_available_at",
            "labels_label_resolution_available_at_col": "execution_label_available_at",
        }
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _resign_manifest(paths: dict[str, Path], source: str) -> None:
    payload = json.loads(paths[f"{source}_manifest"].read_text(encoding="utf-8"))
    payload["source_artifact_sha256"] = materializer._sha256(paths[source])
    payload["prediction_role_manifest_sha256"] = materializer._canonical_json_hash(
        payload, excluded=("prediction_role_manifest_sha256",)
    )
    paths[f"{source}_manifest"].write_text(json.dumps(payload), encoding="utf-8")


def _add_signed_timing_cdf_vector(paths: dict[str, Path]) -> None:
    time = pd.read_parquet(paths["time"])
    for hours, values in {
        2: [0.1, 0.2, 0.3, 0.4],
        4: [0.2, 0.3, 0.4, 0.5],
        8: [0.4, 0.5, 0.6, 0.7],
        12: [0.5, 0.6, 0.7, 0.8],
    }.items():
        time[f"prediction_p_hit_by_{hours}h"] = values
    time.to_parquet(paths["time"], index=False)
    manifest = json.loads(paths["time_manifest"].read_text(encoding="utf-8"))
    manifest["prediction_columns"] = {
        "prediction": {
            "role": "pre_entry_auxiliary_oof_prediction",
            "target": False,
            "head": "timing",
            "source_prediction_column": "pred_time_to_first_meaningful_mfe_12h",
        },
        **{
            f"prediction_p_hit_by_{hours}h": {
                "role": "pre_entry_auxiliary_timing_cdf_probability_oof",
                "target": False,
                "head": "timing",
                "source_prediction_column": f"pred_p_hit_by_{hours}h",
            }
            for hours in (2, 4, 8, 12)
        },
    }
    paths["time_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _resign_manifest(paths, "time")


def test_materializes_runner_compatible_exact_oof_handoff(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    result = materializer.run(_args(tmp_path, paths))
    handoff = pd.read_parquet(result["handoff"])
    provenance = json.loads(result["provenance"].read_text())
    assert handoff["side_name"].tolist() == ["long", "short", "long", "short"]
    assert set(materializer.BASE_JOIN_KEYS).issubset(handoff.columns)
    assert {
        "candidate_id",
        "execution_decision_utc",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_label_end_utc",
        "alpha_oof_fold",
        "time_to_mfe_oof_fold",
        "peak_mfe_oof_fold",
        "mae_before_mfe_oof_fold",
        "adverse_turn_oof_fold",
        "path_slope_oof_fold",
        "catboost_oof_fold",
        "pred_mae_before_meaningful_mfe_atr",
        "pred_bars_before_price_stops_decreasing",
        "pred_favorable_path_slope_atr_per_hour",
        "catboost_max_probability",
        "catboost_normalized_entropy",
        "catboost_top2_probability_margin",
        "catboost_adverse_probability_mass",
        "catboost_favorable_probability_mass",
    }.issubset(handoff.columns)
    assert provenance["schema"] == "execution_ev_joined_handoff_v2"
    assert provenance["handoff"]["join_mode"] == "exact_inner_one_to_one"
    assert provenance["handoff"]["join_keys"] == [
        "__ts__",
        "__symbol__",
        "side_name",
        "candidate_id",
    ]
    assert all(
        "sha256" in source
        for source in provenance["handoff"]["source_artifacts"].values()
    )
    assert all(
        source["signed_prediction_role_manifest"][
            "signed_prediction_role_manifest_sha256"
        ]
        for source in provenance["handoff"]["source_artifacts"].values()
    )
    assert provenance["features"]["catboost_archetype"]["model_input"] is False
    assert (
        provenance["features"]["pred_time_to_first_meaningful_MFE"]["model_input"]
        is True
    )
    assert (
        provenance["features"]["pred_mae_before_meaningful_mfe_atr"]["model_input"]
        is False
    )
    assert (
        provenance["features"]["pred_bars_before_price_stops_decreasing"]["model_input"]
        is False
    )
    assert (
        provenance["features"]["pred_favorable_path_slope_atr_per_hour"]["model_input"]
        is False
    )
    alignment = provenance["handoff"]["population_alignment"]
    assert alignment["mode"] == "explicit_common_identity_intersection_one_to_one"
    assert alignment["common_rows"] == len(handoff)
    assert len(alignment["common_identity_sha256"]) == 64
    assert provenance["handoff"]["cost_basis"]["source_alpha_cost_return"] == 0.01
    probability_columns = [
        column for column in handoff if column.startswith("catboost_p_")
    ]
    assert len(probability_columns) == len(materializer.MERGED_PATH_ARCHETYPE_CLASSES)
    np.testing.assert_allclose(handoff[probability_columns].sum(axis=1), 1.0)
    np.testing.assert_allclose(
        handoff["existing_alpha_ev"],
        handoff["existing_alpha_ev_source_basis"]
        + 0.01
        - handoff["execution_cost_return"],
    )
    feature_provenance, payload = runner._load_provenance(result["provenance"])
    runner._validate_handoff(
        handoff,
        provenance=feature_provenance,
        provenance_payload=payload,
        id_columns=runner.DEFAULT_ID_COLUMNS,
        timestamp_col="__ts__",
        side_col="side_name",
        archetype_col="catboost_archetype",
        label_end_time_col="execution_label_end_utc",
        target_horizon_hours=12.0,
        max_span_days=31.0,
    )


def test_materializes_signed_24h_deployed_policy_horizon(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    labels = pd.read_parquet(paths["labels"])
    labels["execution_label_end_utc"] = labels["__decision_ts__"] + pd.Timedelta(
        minutes=1440
    )
    labels["execution_label_available_at"] = labels["execution_label_end_utc"]
    labels.to_parquet(paths["labels"], index=False)
    manifest = json.loads(paths["labels_manifest"].read_text(encoding="utf-8"))
    manifest["schema"] = "execution_ev_deployed_policy_1m_labels_v1"
    manifest["exit_policy_contract"] = {
        "replay_timeframe": "1m",
        "horizon_minutes": 1440,
        "geometry_scope": "side_x_policy_archetype_with_side_parent_fallback",
        "policy_pathway_id": "joint_trailing_total_mfe_raw_bayesian_v1",
        "simulator": (
            "extreme_price_movements.simple_policy_optimiser.simulate_and_score"
        ),
    }
    paths["labels_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _resign_manifest(paths, "labels")

    result = materializer.run(_args(tmp_path, paths))
    provenance = json.loads(result["provenance"].read_text())
    target = provenance["targets"]["execution_net_ev_12h"]
    assert target["horizon_hours"] == 24.0
    assert target["exit_policy_contract"]["horizon_minutes"] == 1440
    assert target["exit_policy_contract"]["replay_timeframe"] == "1m"


def test_accepts_oof_prediction_available_at_execution_decision(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    alpha = pd.read_parquet(paths["alpha"])
    alpha["available_at"] = pd.to_datetime(alpha["__ts__"], utc=True) + pd.Timedelta(
        hours=1
    )
    alpha.to_parquet(paths["alpha"], index=False)
    _resign_manifest(paths, "alpha")

    result = materializer.run(_args(tmp_path, paths))
    handoff = pd.read_parquet(result["handoff"])
    assert (
        pd.to_datetime(handoff["alpha_available_at"], utc=True)
        == pd.to_datetime(handoff["execution_decision_utc"], utc=True)
    ).all()


def test_ingests_complete_signed_timing_cdf_vector_as_model_inputs(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    _add_signed_timing_cdf_vector(paths)

    result = materializer.run(_args(tmp_path, paths))
    handoff = pd.read_parquet(result["handoff"])
    provenance = json.loads(result["provenance"].read_text(encoding="utf-8"))
    expected_columns = list(materializer.TIMING_CDF_JOINED_FEATURE_COLUMNS.values())

    assert set(expected_columns).issubset(handoff.columns)
    assert (
        provenance["handoff"]["source_artifacts"]["time_to_mfe"]["timing_cdf_vector"][
            "status"
        ]
        == "signed_complete_timing_oof_vector"
    )
    for column in expected_columns:
        feature = provenance["features"][column]
        assert feature["model_input"] is True
        assert feature["oof_or_frozen"] is True
        assert feature["pre_entry"] is True
    assert (
        provenance["features"]["pred_time_to_first_meaningful_MFE"]["model_input"]
        is False
    )

    feature_provenance, payload = runner._load_provenance(result["provenance"])
    assert set(expected_columns).issubset(feature_provenance)
    runner._validate_handoff(
        handoff,
        provenance=feature_provenance,
        provenance_payload=payload,
        id_columns=runner.DEFAULT_ID_COLUMNS,
        timestamp_col="__ts__",
        side_col="side_name",
        archetype_col="catboost_archetype",
        label_end_time_col="execution_label_end_utc",
        target_horizon_hours=12.0,
        max_span_days=31.0,
    )


def test_rejects_partial_or_target_bearing_signed_timing_cdf_vector(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path / "missing-column")
    _add_signed_timing_cdf_vector(paths)
    time = pd.read_parquet(paths["time"]).drop(columns="prediction_p_hit_by_8h")
    time.to_parquet(paths["time"], index=False)
    _resign_manifest(paths, "time")
    with pytest.raises(
        ValueError, match="missing required columns: prediction_p_hit_by_8h"
    ):
        materializer.run(_args(tmp_path / "missing-column", paths))

    paths = _inputs(tmp_path / "target-bearing")
    _add_signed_timing_cdf_vector(paths)
    manifest = json.loads(paths["time_manifest"].read_text(encoding="utf-8"))
    manifest["prediction_columns"]["prediction_p_hit_by_4h"]["target"] = True
    paths["time_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _resign_manifest(paths, "time")
    with pytest.raises(ValueError, match="not a target-free timing OOF prediction"):
        materializer.run(_args(tmp_path / "target-bearing", paths))


def test_materializer_records_signed_merged_seven_class_contract(
    tmp_path: Path,
) -> None:
    merged = (
        "immediate_adverse_path",
        "early_mfe_full_reversal",
        "fast_realization_winner",
        "late_breakout",
        "slow_grinder",
        "noisy_timeout_usable_mfe",
        "dead_timeout",
    )
    paths = _inputs(tmp_path, class_order=merged)
    canonical = merged
    result = materializer.run(
        _args(
            tmp_path,
            paths,
            catboost_prob_cols=[f"probability__{shape}" for shape in canonical],
        )
    )
    provenance = json.loads(result["provenance"].read_text())
    contract = provenance["handoff"]["catboost_class_contract"]

    assert contract["class_order"] == list(canonical)
    assert contract["class_order_sha256"] == materializer.catboost_class_order_sha256(
        canonical
    )
    assert contract["source"] == "merged_path_archetype_classes_from_probability_names"
    for name, feature in provenance["features"].items():
        if name.startswith("catboost_p_") or name in {
            "catboost_entropy",
            "catboost_max_probability",
            "catboost_normalized_entropy",
            "catboost_top2_probability_margin",
            "catboost_adverse_probability_mass",
            "catboost_favorable_probability_mass",
            "catboost_archetype",
        }:
            assert feature["class_order"] == list(canonical)
            assert feature["class_order_sha256"] == contract["class_order_sha256"]


def test_materializer_preserves_explicit_signed_merged_class_order(
    tmp_path: Path,
) -> None:
    merged = (
        "dead_timeout",
        "noisy_timeout_usable_mfe",
        "slow_grinder",
        "late_breakout",
        "fast_realization_winner",
        "early_mfe_full_reversal",
        "immediate_adverse_path",
    )
    paths = _inputs(tmp_path, class_order=merged, manifest_class_order=True)
    result = materializer.run(
        _args(
            tmp_path,
            paths,
            catboost_prob_cols=[f"probability__{shape}" for shape in merged],
        )
    )
    contract = json.loads(result["provenance"].read_text())["handoff"][
        "catboost_class_contract"
    ]

    assert contract["class_order"] == list(merged)
    assert contract["source"] == "path_shape_types"


def test_materializer_rejects_probability_columns_out_of_signed_class_order(
    tmp_path: Path,
) -> None:
    merged = (
        "immediate_adverse_path",
        "early_mfe_full_reversal",
        "fast_realization_winner",
        "late_breakout",
        "slow_grinder",
        "noisy_timeout_usable_mfe",
        "dead_timeout",
    )
    paths = _inputs(tmp_path, class_order=merged)
    with pytest.raises(ValueError, match="match the signed manifest class order"):
        materializer.run(
            _args(
                tmp_path,
                paths,
                catboost_prob_cols=[
                    f"probability__{shape}" for shape in reversed(merged)
                ],
            )
        )


def test_joiner_defaults_match_the_canonical_auxiliary_adapter_schema() -> None:
    parser = materializer._parser()
    args = parser.parse_args(
        [
            "--alpha",
            "alpha.parquet",
            "--time-oof",
            "time.parquet",
            "--peak-oof",
            "peak.parquet",
            "--mae-oof",
            "mae.parquet",
            "--turn-oof",
            "turn.parquet",
            "--slope-oof",
            "slope.parquet",
            "--catboost-oof",
            "catboost.parquet",
            "--execution-labels",
            "labels.parquet",
            "--output",
            "joined.parquet",
            "--alpha-manifest",
            "alpha.json",
            "--time-manifest",
            "time.json",
            "--peak-manifest",
            "peak.json",
            "--mae-manifest",
            "mae.json",
            "--turn-manifest",
            "turn.json",
            "--slope-manifest",
            "slope.json",
            "--catboost-manifest",
            "catboost.json",
            "--labels-manifest",
            "labels.json",
        ]
    )
    assert {
        args.time_prediction_col,
        args.peak_prediction_col,
        args.mae_prediction_col,
        args.turn_prediction_col,
        args.slope_prediction_col,
    } == {"prediction"}
    assert args.catboost_prob_cols == [
        f"probability__{shape}" for shape in materializer.MERGED_PATH_ARCHETYPE_CLASSES
    ]
    assert {
        args.catboost_max_probability_col,
        args.catboost_normalized_entropy_col,
        args.catboost_top2_margin_col,
        args.catboost_adverse_mass_col,
        args.catboost_favorable_mass_col,
    } == {
        "max_probability",
        "normalized_entropy",
        "top2_probability_margin",
        "adverse_probability_mass",
        "favorable_probability_mass",
    }


def test_rejects_catboost_probability_summary_drift(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    catboost = pd.read_parquet(paths["catboost"])
    catboost.loc[0, "adverse_probability_mass"] += 0.01
    catboost.to_parquet(paths["catboost"], index=False)
    _resign_manifest(paths, "catboost")

    with pytest.raises(
        ValueError,
        match="catboost_adverse_probability_mass does not match",
    ):
        materializer.run(_args(tmp_path, paths))


def test_rejects_duplicate_identity_and_missing_oof_fold(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    duplicate = pd.read_parquet(paths["time"])
    duplicate = pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True)
    duplicate.to_parquet(paths["time"], index=False)
    with pytest.raises(ValueError, match="duplicate rows"):
        materializer.run(_args(tmp_path, paths))

    paths = _inputs(tmp_path / "missing-fold")
    missing = pd.read_parquet(paths["peak"])
    missing.loc[0, "oof_fold"] = np.nan
    missing.to_parquet(paths["peak"], index=False)
    with pytest.raises(ValueError, match="missing or invalid required OOF fold IDs"):
        materializer.run(_args(tmp_path / "missing-fold", paths))


def test_rejects_target_feature_and_empty_common_intersection(tmp_path: Path) -> None:
    paths = _inputs(tmp_path / "target")
    target_time = pd.read_parquet(paths["time"]).rename(
        columns={
            "pred_time_to_first_meaningful_mfe_12h": "realized_time_to_first_meaningful_mfe"
        }
    )
    target_time.to_parquet(paths["time"], index=False)
    with pytest.raises(ValueError, match="target leakage"):
        materializer.run(
            _args(
                tmp_path / "target",
                paths,
                time_prediction_col="realized_time_to_first_meaningful_mfe",
            )
        )

    paths = _inputs(tmp_path / "disjoint")
    disjoint = pd.read_parquet(paths["catboost"])
    disjoint["__ts__"] = disjoint["__ts__"] + pd.Timedelta(days=30)
    disjoint.to_parquet(paths["catboost"], index=False)
    _resign_manifest(paths, "catboost")
    with pytest.raises(
        ValueError, match="common OOF identity intersection is too small"
    ):
        materializer.run(_args(tmp_path / "disjoint", paths))


def test_rejects_missing_candidate_id_and_invalid_upstream_timing_evidence(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path / "missing-candidate")
    time = pd.read_parquet(paths["time"]).drop(columns="candidate_id")
    time.to_parquet(paths["time"], index=False)
    with pytest.raises(
        ValueError, match="strict joined execution-EV provenance is unavailable"
    ):
        materializer.run(_args(tmp_path / "missing-candidate", paths))

    paths = _inputs(tmp_path / "partial-coverage")
    slope = pd.read_parquet(paths["slope"]).iloc[:-1]
    slope.to_parquet(paths["slope"], index=False)
    _resign_manifest(paths, "slope")
    result = materializer.run(_args(tmp_path / "partial-coverage", paths))
    handoff = pd.read_parquet(result["handoff"])
    provenance = json.loads(result["provenance"].read_text(encoding="utf-8"))
    assert len(handoff) == 3
    slope_alignment = provenance["handoff"]["population_alignment"]["sources"][
        "path_slope"
    ]
    assert slope_alignment["input_rows"] == 3
    assert slope_alignment["retained_common_rows"] == 3
    assert slope_alignment["dropped_not_in_common_rows"] == 0
    assert (
        provenance["handoff"]["population_alignment"]["sources"]["alpha"][
            "dropped_not_in_common_rows"
        ]
        == 1
    )

    paths = _inputs(tmp_path / "late-resolution")
    peak = pd.read_parquet(paths["peak"])
    peak["label_resolution_available_at"] = peak[
        "train_decision_cutoff"
    ] + pd.Timedelta(hours=1)
    peak.to_parquet(paths["peak"], index=False)
    with pytest.raises(ValueError, match="training labels must resolve before"):
        materializer.run(_args(tmp_path / "late-resolution", paths))

    paths = _inputs(tmp_path / "late-cutoff")
    mae = pd.read_parquet(paths["mae"])
    mae["train_decision_cutoff"] = mae["__ts__"]
    mae.to_parquet(paths["mae"], index=False)
    with pytest.raises(
        ValueError,
        match="train decision cutoff must be strictly before validation start",
    ):
        materializer.run(_args(tmp_path / "late-cutoff", paths))


def test_rejects_execution_target_alias_and_unsigned_alpha_role_manifest(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path / "target-alias")
    alpha = pd.read_parquet(paths["alpha"])
    alpha["execution_net_ev_12h"] = alpha["existing_alpha_ev"]
    alpha.to_parquet(paths["alpha"], index=False)
    with pytest.raises(ValueError, match="target leakage"):
        materializer.run(
            _args(
                tmp_path / "target-alias",
                paths,
                alpha_ev_col="execution_net_ev_12h",
            )
        )

    paths = _inputs(tmp_path / "unsigned-alpha")
    manifest = json.loads(paths["alpha_manifest"].read_text(encoding="utf-8"))
    manifest["prediction_columns"]["existing_alpha_ev"]["target"] = True
    paths["alpha_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        ValueError, match="signed prediction-role manifest hash does not verify"
    ):
        materializer.run(_args(tmp_path / "unsigned-alpha", paths))


def test_rejects_missing_alpha_cost_proof_and_mismatched_override(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path / "missing-cost-proof")
    manifest = json.loads(paths["alpha_manifest"].read_text(encoding="utf-8"))
    manifest.pop("alpha_cost_basis")
    manifest["prediction_role_manifest_sha256"] = materializer._canonical_json_hash(
        manifest, excluded=("prediction_role_manifest_sha256",)
    )
    paths["alpha_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="requires an explicit alpha_cost_basis"):
        materializer.run(_args(tmp_path / "missing-cost-proof", paths))

    paths = _inputs(tmp_path / "mismatched-override")
    with pytest.raises(ValueError, match="override must exactly match"):
        materializer.run(
            _args(
                tmp_path / "mismatched-override", paths, alpha_source_cost_return=0.003
            )
        )


def test_rejects_small_common_intersection(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    for source in ("time", "peak", "mae", "turn", "slope", "catboost", "labels"):
        frame = pd.read_parquet(paths[source]).iloc[:1]
        frame.to_parquet(paths[source], index=False)
        _resign_manifest(paths, source)
    with pytest.raises(ValueError, match="retained_rows=1, min_common_rows=2"):
        materializer.run(_args(tmp_path, paths))
