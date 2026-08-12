from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.leaf_reasoning_final_oos import (
    FINAL_OOS_START,
    FinalOOSReplayContract,
    FinalOOSReplayError,
    SCHEMA,
    run_leaf_reasoning_final_oos_replay,
)


def _digest(path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path, value: dict) -> dict[str, str]:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": path.name, "sha256": _digest(path), "fit_end_utc": "2024-10-31T23:00:00Z"}


def _write_model(path) -> dict[str, str]:
    path.write_text("frozen native lightgbm text placeholder", encoding="utf-8")
    return {"path": path.name, "sha256": _digest(path), "fit_end_utc": "2024-10-31T23:00:00Z"}


def _contract_payload(tmp_path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    features = {
        "long": ["p_adverse", "p_weak", "p_clear", "base_expected_bps", "ctx"],
        "short": ["p_adverse", "p_weak", "p_clear", "base_expected_bps", "ctx"],
    }
    selection = _write_json(tmp_path / "selection.json", {
        "development_only": True, "final_november_oos_consumed": False,
        "development_evaluation_end_utc": "2024-11-01T00:00:00Z",
        "selected_arm": "C5", "successor": "S1",
    })
    groups = _write_json(tmp_path / "groups.json", {
        "development_only": True, "final_november_oos_consumed": False,
        "development_evaluation_end_utc": "2024-11-01T00:00:00Z",
        "selected_arm": "C5",
        "selected_meta_features_by_side": features,
    })
    taxonomy = _write_json(tmp_path / "taxonomy.json", {
        "development_only": True, "final_november_oos_consumed": False,
        "development_evaluation_end_utc": "2024-11-01T00:00:00Z",
        "linkage": "average", "threshold_by_arm": {"C1": .60, "C2": .70, "C3": .80, "C4": .90},
        "cluster_ids_by_arm": {
            "C1": ["c1_a", "c1_b"], "C2": ["c2_a"], "C3": ["c3_a"], "C4": ["c4_a"],
            "C5": ["c1_a", "c1_b"], "C6": ["c1_a"],
        },
        "c5_source_arm": "C1", "c6_source_arm": "C5", "top_decile_coverage_target": .95,
        "top_decile_coverage_by_arm": {"C5": .96}, "portable_top_decile_coverage_by_arm": {"C5": .96},
        "c6_best_cross_era_score": 10.0, "c6_best_cross_era_standard_error": 1.0,
        "c6_compact_cross_era_score": 9.5,
    })
    successor = _write_json(tmp_path / "successor.json", {
        "development_only": True, "final_november_oos_consumed": False,
        "development_evaluation_end_utc": "2024-11-01T00:00:00Z", "successor": "S1",
        "terminal_decision": "COMPACT_REASONING_GENERATION_SELECTED",
    })
    model_spec = _write_json(tmp_path / "meta_spec.json", {
        "development_only": True, "final_november_oos_consumed": False,
        "development_evaluation_end_utc": "2024-11-01T00:00:00Z",
        "family": "lightgbm_lgbmregressor", "contract_id": "frozen_test_huber_v1",
        "params": {"objective": "huber", "n_estimators": 10},
    })
    state = _write_json(tmp_path / "causal_state.json", {
        "development_only": True, "final_november_oos_consumed": False,
    })
    base_long = _write_model(tmp_path / "base_long.txt")
    base_short = _write_model(tmp_path / "base_short.txt")
    meta_long = _write_model(tmp_path / "meta_long.txt")
    meta_short = _write_model(tmp_path / "meta_short.txt")
    map_long = _write_json(tmp_path / "map_long.json", {
        "development_only": True, "final_november_oos_consumed": False, "side_name": "long",
        "fit_end_utc": "2024-10-31T23:00:00Z", "class_expected_net_bps": {"adverse": -200.0, "weak": 0.0, "clear": 200.0},
    })
    map_short = _write_json(tmp_path / "map_short.json", {
        "development_only": True, "final_november_oos_consumed": False, "side_name": "short",
        "fit_end_utc": "2024-10-31T23:00:00Z", "class_expected_net_bps": {"adverse": -200.0, "weak": 0.0, "clear": 200.0},
    })
    return {
        "schema": SCHEMA,
        "status": "DEVELOPMENT_SELECTED_FROZEN_FINAL_OOS_CONTRACT",
        "final_november_oos_consumed": False,
        "development_selection": {
            "selected_arm": "C5", "successor": "S1", "selected_meta_features_by_side": features,
            "development_transports": ["transport_a_2023q4_to_2024h1", "transport_b_2024h1_to_2024h2_to_date"],
            "selection_artifact": selection, "feature_group_artifact": groups,
            "taxonomy_artifact": taxonomy, "successor_decision_artifact": successor,
            "frozen_meta_model_spec_artifact": model_spec,
            "development_evaluation_end_utc": "2024-11-01T00:00:00Z", "final_november_oos_consumed": False,
        },
        "scoring": {
            "long": {"base_model": base_long, "base_feature_columns": ["base_x"], "base_value_map": map_long, "meta_model": meta_long, "meta_feature_columns": features["long"]},
            "short": {"base_model": base_short, "base_feature_columns": ["base_x"], "base_value_map": map_short, "meta_model": meta_short, "meta_feature_columns": features["short"]},
        },
        "causal_state_artifacts": [state],
    }


def _panel() -> pd.DataFrame:
    decision = pd.date_range(FINAL_OOS_START, periods=4, freq="h", tz="UTC")
    side = ["long", "short", "long", "short"]
    gross = np.array([230.0, 90.0, 180.0, 250.0])
    return pd.DataFrame({
        "candidate_id": ["same", "same", "third", "fourth"], "side_name": side,
        "decision_ts": decision, "entry_ts": decision + pd.Timedelta(hours=1),
        "label_available_ts": decision + pd.Timedelta(hours=13),
        "feature_available_ts": decision, "causal_state_available_ts": decision,
        "gross_bps": gross, "net_bps": gross - 100.0,
        "base_x": np.arange(4, dtype=np.float32), "ctx": np.array([.0, 1.0, 2.0, 7.0], dtype=np.float32),
    })


class _FakeBase:
    def __init__(self, side: str) -> None:
        self.side = side

    def num_feature(self) -> int:
        return 1

    def feature_name(self) -> list[str]:
        return ["base_x"]

    def predict(self, data):
        rows = len(data)
        p = np.array([.1, .2, .7] if self.side == "long" else [.3, .2, .5], dtype=float)
        return np.tile(p, (rows, 1))


class _FakeMeta:
    def num_feature(self) -> int:
        return 5

    def feature_name(self) -> list[str]:
        return ["p_adverse", "p_weak", "p_clear", "base_expected_bps", "ctx"]

    def predict(self, data):
        return data["ctx"].to_numpy(float) * 100.0


def _loader(path, role: str, side: str):
    return _FakeBase(side) if role == "base" else _FakeMeta()


def test_replay_requires_hash_bound_selection_group_taxonomy_and_successor(tmp_path) -> None:
    payload = _contract_payload(tmp_path)
    payload["development_selection"].pop("taxonomy_artifact")
    with pytest.raises(FinalOOSReplayError, match="taxonomy_artifact"):
        FinalOOSReplayContract.from_dict(payload, root=tmp_path)

    payload = _contract_payload(tmp_path / "other")
    # The group decision must bind exactly the actual runtime side-local lists.
    group_path = tmp_path / "other" / "groups.json"
    group = json.loads(group_path.read_text())
    group["selected_meta_features_by_side"]["long"] = ["p_adverse"]
    group_path.write_text(json.dumps(group), encoding="utf-8")
    payload["development_selection"]["feature_group_artifact"]["sha256"] = _digest(group_path)
    with pytest.raises(FinalOOSReplayError, match="feature group artifact differs"):
        FinalOOSReplayContract.from_dict(payload, root=tmp_path / "other")


def test_replay_rejects_any_frozen_artifact_fit_at_or_after_november(tmp_path) -> None:
    payload = _contract_payload(tmp_path)
    payload["scoring"]["long"]["base_model"]["fit_end_utc"] = "2024-11-01T00:00:00Z"
    with pytest.raises(FinalOOSReplayError, match="strictly before final November"):
        FinalOOSReplayContract.from_dict(payload, root=tmp_path)


def test_replay_requires_the_development_frozen_huber_meta_spec(tmp_path) -> None:
    payload = _contract_payload(tmp_path)
    spec_path = tmp_path / "meta_spec.json"
    spec = json.loads(spec_path.read_text())
    spec["params"]["objective"] = "regression"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    payload["development_selection"]["frozen_meta_model_spec_artifact"]["sha256"] = _digest(spec_path)
    with pytest.raises(FinalOOSReplayError, match="Huber"):
        FinalOOSReplayContract.from_dict(payload, root=tmp_path)


def test_replay_never_accepts_raw_leaf_identifiers_as_final_meta_inputs(tmp_path) -> None:
    payload = _contract_payload(tmp_path)
    for side in ("long", "short"):
        payload["development_selection"]["selected_meta_features_by_side"][side].append("raw_leaf_token")
        if "raw_leaf_token" not in payload["scoring"][side]["meta_feature_columns"]:
            payload["scoring"][side]["meta_feature_columns"].append("raw_leaf_token")
    group_path = tmp_path / "groups.json"
    group = json.loads(group_path.read_text())
    for side in ("long", "short"):
        group["selected_meta_features_by_side"][side].append("raw_leaf_token")
    group_path.write_text(json.dumps(group), encoding="utf-8")
    payload["development_selection"]["feature_group_artifact"]["sha256"] = _digest(group_path)
    with pytest.raises(FinalOOSReplayError, match="raw leaf"):
        FinalOOSReplayContract.from_dict(payload, root=tmp_path)


def test_replay_scores_frozen_models_and_ranks_once_pooled_cross_side(tmp_path) -> None:
    contract = FinalOOSReplayContract.from_dict(_contract_payload(tmp_path), root=tmp_path)
    result = run_leaf_reasoning_final_oos_replay(
        contract, _panel(), output_dir=tmp_path / "out", consumption_registry=tmp_path / "one_time.json", model_loader=_loader,
    )
    assert len(result.scored_predictions) == 4
    assert np.allclose(result.scored_predictions["common_bps_score"], result.scored_predictions["base_expected_bps"] + result.scored_predictions["predicted_residual_bps"])
    top = pd.read_parquet(result.output_dir / "final_oos_selected_candidates.parquet")
    top_one = top.loc[top.top_fraction.eq(.01)].iloc[0]
    assert top_one.side_name == "short"
    assert top_one.candidate_id == "fourth"
    metrics = pd.read_parquet(result.output_dir / "final_oos_global_topk_metrics.parquet")
    assert set(metrics["scope"]).issuperset({"global", "side", "month", "side_month"})
    assert set(metrics["ranking_basis"]) == {"global_common_bps_score"}
    manifest = json.loads((result.output_dir / "run_manifest.json").read_text())
    assert manifest["final_november_oos_consumed"] is True
    assert manifest["no_final_oos_selection_hpo_or_refit_tuning"] is True
    with pytest.raises(FinalOOSReplayError, match="already reserved/consumed"):
        run_leaf_reasoning_final_oos_replay(
            contract, _panel(), output_dir=tmp_path / "second", consumption_registry=tmp_path / "one_time.json", model_loader=_loader,
        )


def test_replay_fails_before_scoring_on_late_state_or_wrong_entry_and_score_smuggling(tmp_path) -> None:
    contract = FinalOOSReplayContract.from_dict(_contract_payload(tmp_path), root=tmp_path)
    panel = _panel()
    panel.loc[0, "causal_state_available_ts"] = panel.loc[0, "decision_ts"] + pd.Timedelta(seconds=1)
    with pytest.raises(FinalOOSReplayError, match="availability"):
        run_leaf_reasoning_final_oos_replay(
            contract, panel, output_dir=tmp_path / "late", consumption_registry=tmp_path / "late.registry", model_loader=_loader,
        )
    panel = _panel()
    panel["common_bps_score"] = 999.0
    with pytest.raises(FinalOOSReplayError, match="smuggle"):
        run_leaf_reasoning_final_oos_replay(
            contract, panel, output_dir=tmp_path / "smuggled", consumption_registry=tmp_path / "smuggled.registry", model_loader=_loader,
        )
    panel = _panel()
    panel.loc[0, "entry_ts"] = panel.loc[0, "decision_ts"]
    with pytest.raises(FinalOOSReplayError, match="next hourly open"):
        run_leaf_reasoning_final_oos_replay(
            contract, panel, output_dir=tmp_path / "entry", consumption_registry=tmp_path / "entry.registry", model_loader=_loader,
        )
