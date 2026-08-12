from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.leaf_reasoning_final_oos import DEVELOPMENT_CUTOFF, FinalOOSReplayContract
from extreme_price_movements.leaf_reasoning_finalizer import (
    DEVELOPMENT_SELECTION_SCHEMA,
    DEVELOPMENT_SELECTION_STATUS,
    DevelopmentFinalizationSelection,
    F0_REPRESENTATION,
    LeafReasoningFinalizerError,
    finalize_leaf_reasoning_final_oos,
    read_pre_cutoff_parquet,
)


def _digest(path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path, payload: dict) -> dict[str, str]:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return {"path": path.name, "sha256": _digest(path), "fit_end_utc": "2024-10-31T23:00:00Z"}


def _selection_file(tmp_path):
    base_features = {"long": ["base_long"], "short": ["base_short"]}
    meta_features = {
        "long": ["p_adverse", "p_weak", "p_clear", "base_expected_bps", "ctx"],
        "short": ["p_adverse", "p_weak", "p_clear", "base_expected_bps", "ctx"],
    }
    base_decision = _write_json(tmp_path / "base_selection.json", {
        "development_only": True,
        "final_november_oos_consumed": False,
        "development_evaluation_end_utc": DEVELOPMENT_CUTOFF.isoformat(),
        "winner": F0_REPRESENTATION,
    })
    selected = _write_json(tmp_path / "development_selection.json", {
        "development_only": True,
        "final_november_oos_consumed": False,
        "development_evaluation_end_utc": DEVELOPMENT_CUTOFF.isoformat(),
        "selected_arm": "C5",
        "successor": "S1",
    })
    groups = _write_json(tmp_path / "feature_groups.json", {
        "development_only": True,
        "final_november_oos_consumed": False,
        "development_evaluation_end_utc": DEVELOPMENT_CUTOFF.isoformat(),
        "selected_arm": "C5",
        "selected_meta_features_by_side": meta_features,
    })
    taxonomy = _write_json(tmp_path / "taxonomy.json", {
        "development_only": True,
        "final_november_oos_consumed": False,
        "development_evaluation_end_utc": DEVELOPMENT_CUTOFF.isoformat(),
        "linkage": "average",
        "threshold_by_arm": {"C1": .60, "C2": .70, "C3": .80, "C4": .90},
        "cluster_ids_by_arm": {
            "C1": ["one", "two"], "C2": ["one"], "C3": ["one"], "C4": ["one"],
            "C5": ["one", "two"], "C6": ["one"],
        },
        "c5_source_arm": "C1", "c6_source_arm": "C5",
        "top_decile_coverage_target": .95,
        "top_decile_coverage_by_arm": {"C5": .96},
        "portable_top_decile_coverage_by_arm": {"C5": .96},
        "c6_best_cross_era_score": 1.0,
        "c6_best_cross_era_standard_error": .10,
        "c6_compact_cross_era_score": .95,
    })
    successor = _write_json(tmp_path / "successor.json", {
        "development_only": True,
        "final_november_oos_consumed": False,
        "development_evaluation_end_utc": DEVELOPMENT_CUTOFF.isoformat(),
        "successor": "S1", "terminal_decision": "COMPACT_REASONING_GENERATION_SELECTED",
    })
    meta_spec = _write_json(tmp_path / "meta_spec.json", {
        "development_only": True,
        "final_november_oos_consumed": False,
        "development_evaluation_end_utc": DEVELOPMENT_CUTOFF.isoformat(),
        "family": "lightgbm_lgbmregressor", "contract_id": "test_frozen_huber",
        "params": {
            "objective": "huber", "n_estimators": 4, "learning_rate": .1,
            "num_leaves": 5, "min_child_samples": 1, "random_state": 19,
            "n_jobs": 1, "verbosity": -1,
        },
    })
    state = _write_json(tmp_path / "causal_state.json", {
        "development_only": True, "final_november_oos_consumed": False,
        "fit_end_utc": "2024-10-31T23:00:00Z",
    })
    source = {
        "schema": DEVELOPMENT_SELECTION_SCHEMA,
        "status": DEVELOPMENT_SELECTION_STATUS,
        "immutable_output": True,
        "final_november_oos_consumed": False,
        "development_evaluation_end_utc": DEVELOPMENT_CUTOFF.isoformat(),
        "base": {
            "representation": F0_REPRESENTATION,
            "feature_columns_by_side": base_features,
            "final_seed_by_side": {"long": 101, "short": 202},
            "selection_artifact": base_decision,
        },
        "development_selection": {
            "selected_arm": "C5", "successor": "S1",
            "selected_meta_features_by_side": meta_features,
            "development_transports": [
                "transport_a_2023q4_to_2024h1", "transport_b_2024h1_to_2024h2_to_date",
            ],
            "selection_artifact": selected,
            "feature_group_artifact": groups,
            "taxonomy_artifact": taxonomy,
            "successor_decision_artifact": successor,
            "frozen_meta_model_spec_artifact": meta_spec,
            "development_evaluation_end_utc": DEVELOPMENT_CUTOFF.isoformat(),
            "final_november_oos_consumed": False,
        },
        "causal_state_artifacts": [state],
    }
    path = tmp_path / "sealed_development_selection.json"
    path.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
    return path


def _training_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    decision = pd.date_range("2024-10-20", periods=18, freq="h", tz="UTC")
    rows = []
    for side_number, side in enumerate(("long", "short")):
        for index, timestamp in enumerate(decision):
            klass = index % 3
            net = [-180.0, 10.0, 210.0][klass] + side_number * 5.0
            rows.append({
                "candidate_id": f"{side}-{index}", "side_name": side,
                "decision_ts": timestamp, "label_available_ts": timestamp + pd.Timedelta(hours=13),
                "gross_bps": net + 100.0, "net_bps": net, "r3_class": klass,
                "robust_clear_event_b0": int(klass == 2),
                "robust_clear_event_b25": int(klass == 2),
                "robust_clear_event_b50": int(klass == 2),
                "base_long": float(index + side_number), "base_short": float(index - side_number),
            })
    base = pd.DataFrame(rows)
    meta = base.loc[:, ["candidate_id", "side_name", "decision_ts", "label_available_ts", "net_bps"]].rename(columns={"net_bps": "realized_net_bps"})
    meta["p_adverse"] = np.where(meta.index % 3 == 0, .8, .1)
    meta["p_weak"] = .1
    meta["p_clear"] = 1.0 - meta["p_adverse"] - meta["p_weak"]
    meta["base_expected_bps"] = np.where(meta.index % 3 == 2, 150.0, -50.0)
    meta["ctx"] = (np.arange(len(meta), dtype=float) % 5) / 4.0
    meta["base_same_side_strict_oof"] = True
    meta["base_oof_fit_end_ts"] = meta["decision_ts"] - pd.Timedelta(hours=14)
    meta["base_oof_generated_ts"] = meta["decision_ts"]
    return base, meta


def test_finalizer_seals_native_side_models_maps_and_replay_contract(tmp_path) -> None:
    lgb = pytest.importorskip("lightgbm")
    selection = DevelopmentFinalizationSelection.from_json_path(_selection_file(tmp_path))
    base, meta = _training_frames()
    result = finalize_leaf_reasoning_final_oos(selection, base, meta, output_dir=tmp_path / "sealed")

    assert result.output_dir.is_dir()
    assert (result.output_dir / "models" / "base_long.txt").is_file()
    assert (result.output_dir / "models" / "meta_short.txt").is_file()
    assert (result.output_dir / "value_maps" / "base_class_value_map_long.json").is_file()
    assert (result.output_dir / "causal_feature_contract.json").is_file()
    contract = FinalOOSReplayContract.from_json_path(result.frozen_contract_path)
    assert contract.sha256 == result.contract_sha256
    assert contract.finalization_provenance is not None
    assert contract.finalization_provenance["development_cutoff_utc"] == DEVELOPMENT_CUTOFF.isoformat()
    assert contract.scoring_by_side["long"].base_model.path.suffix == ".txt"
    assert lgb.Booster(model_file=str(contract.scoring_by_side["long"].base_model.path)).feature_name() == ["base_long"]
    manifest = json.loads((result.output_dir / "run_manifest.json").read_text())
    assert manifest["final_november_oos_consumed"] is False
    assert manifest["no_final_oos_labels_or_features_read_for_selection_or_fitting"] is True


def test_finalizer_rejects_november_rows_before_any_model_fit(tmp_path) -> None:
    selection = DevelopmentFinalizationSelection.from_json_path(_selection_file(tmp_path))
    base, meta = _training_frames()
    late = base.iloc[[0]].copy()
    late["decision_ts"] = DEVELOPMENT_CUTOFF
    late["label_available_ts"] = DEVELOPMENT_CUTOFF + pd.Timedelta(hours=13)
    with pytest.raises(LeafReasoningFinalizerError, match="untouched November cutoff"):
        finalize_leaf_reasoning_final_oos(selection, pd.concat([base, late], ignore_index=True), meta, output_dir=tmp_path / "late")
    assert not (tmp_path / "late").exists()


def test_finalizer_requires_strict_same_side_base_oof_for_meta_training(tmp_path) -> None:
    selection = DevelopmentFinalizationSelection.from_json_path(_selection_file(tmp_path))
    base, meta = _training_frames()
    meta.loc[0, "base_same_side_strict_oof"] = False
    with pytest.raises(LeafReasoningFinalizerError, match="same-side strict OOF"):
        finalize_leaf_reasoning_final_oos(selection, base, meta, output_dir=tmp_path / "non_oof")


def test_cli_reader_predicate_projects_only_pre_cutoff_resolved_rows(tmp_path) -> None:
    base, _ = _training_frames()
    late = base.iloc[[0]].copy()
    late["candidate_id"] = "november-row"
    late["decision_ts"] = DEVELOPMENT_CUTOFF
    late["label_available_ts"] = DEVELOPMENT_CUTOFF + pd.Timedelta(hours=13)
    source = tmp_path / "base_source.parquet"
    pd.concat([base, late], ignore_index=True).to_parquet(source, index=False)
    fields = ["candidate_id", "side_name", "decision_ts", "label_available_ts", "net_bps"]
    loaded = read_pre_cutoff_parquet(source, columns=fields)
    assert len(loaded) == len(base)
    assert loaded["decision_ts"].lt(DEVELOPMENT_CUTOFF).all()
    assert loaded["label_available_ts"].lt(DEVELOPMENT_CUTOFF).all()
