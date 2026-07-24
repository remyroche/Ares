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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_entry_timing_meta import (
    EntryTimingFeatureProvenance,
    validate_entry_timing_feature_contract,
)
from extreme_price_movements.path_archetype_labels import PATH_SHAPE_TYPES

SPEC = importlib.util.spec_from_file_location(
    "materialize_execution_entry_timing_handoff",
    ROOT / "scripts" / "materialize_execution_entry_timing_handoff.py",
)
assert SPEC and SPEC.loader
materializer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materializer)


def _write_signed(path: Path, payload: dict[str, object]) -> Path:
    payload["prediction_role_manifest_sha256"] = materializer._canonical_manifest_hash(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _keys(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-07-01", periods=rows, freq="h", tz="UTC"),
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD"] * (rows // 2),
            "side_name": ["long", "short"] * (rows // 2),
            "candidate_id": [f"candidate-{index}" for index in range(rows)],
        }
    )


def _path(timestamp: pd.Timestamp) -> list[dict[str, object]]:
    return [
        {
            "timestamp": timestamp + pd.Timedelta(minutes=index),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
        }
        for index in range(12 * 60)
    ]


def _inputs(tmp_path: Path) -> dict[str, Path]:
    keys = _keys()
    probability = np.full((len(keys), len(PATH_SHAPE_TYPES)), 0.3 / (len(PATH_SHAPE_TYPES) - 1))
    probability[np.arange(len(keys)), np.arange(len(keys)) % len(PATH_SHAPE_TYPES)] = 0.7
    handoff = keys.copy()
    handoff["__decision_ts__"] = handoff["__ts__"] + pd.Timedelta(hours=1)
    handoff["execution_label_end_utc"] = handoff["__ts__"] + pd.Timedelta(hours=13)
    handoff["existing_alpha_ev"] = np.linspace(0.01, 0.04, len(keys))
    handoff["catboost_entropy"] = -np.sum(probability * np.log(probability), axis=1)
    handoff["catboost_archetype"] = [PATH_SHAPE_TYPES[index] for index in range(len(keys))]
    for index in range(len(PATH_SHAPE_TYPES)):
        handoff[f"catboost_p_{index}"] = probability[:, index]
    handoff["base_archetype_label__trend"] = np.array([1.0, 0.0, 1.0, 0.0])
    for index, column in enumerate(materializer._AUXILIARY_COLUMNS):
        handoff[column] = 0.1 + index
    for source in materializer._HANDOFF_OOF_SOURCES:
        handoff[f"{source}_oof_fold"] = "fold-0"
        handoff[f"{source}_train_decision_cutoff"] = handoff["__ts__"] - pd.Timedelta(hours=1)
        handoff[f"{source}_available_at"] = handoff["__ts__"]
    handoff_path = tmp_path / "joined.parquet"
    handoff.to_parquet(handoff_path, index=False)
    joined_provenance = tmp_path / "joined.provenance.json"
    joined_provenance.write_text(
        json.dumps(
            {
                "schema": materializer.HANDOFF_SCHEMA,
                "handoff": {"join_mode": "exact_inner_one_to_one", "join_keys": list(materializer.JOIN_KEYS)},
            }
        ),
        encoding="utf-8",
    )

    oof = keys.copy()
    oof["direct__all_features"] = 0.02
    oof["residual__all_features"] = -0.001
    oof["direct__all_features__is_oof"] = True
    oof["residual__all_features__is_oof"] = True
    oof["execution_ev_oof_fold"] = "fold-0"
    oof["execution_ev_oof_train_decision_cutoff_utc"] = keys["__ts__"] - pd.Timedelta(hours=1)
    oof["execution_ev_oof_available_at"] = keys["__ts__"]
    oof_path = tmp_path / "execution_ev_oof.parquet"
    oof.to_parquet(oof_path, index=False)
    runner_manifest = tmp_path / "execution_ev_runner.json"
    runner_manifest.write_text(
        json.dumps(
            {
                "schema": "execution_ev_meta_runner_v1",
                "status": "completed",
                "input": {"sha256": materializer._sha256(handoff_path)},
                "provenance": {"sha256": materializer._sha256(joined_provenance)},
                "oof_ledger": oof_path.name,
            }
        ),
        encoding="utf-8",
    )

    ev_map = keys.copy()
    ev_map["mapped_execution_ev"] = 0.018
    ev_map["mapped_execution_ev__is_oof"] = True
    ev_map["execution_ev_map_oof_fold"] = "map-fold-0"
    ev_map["execution_ev_map_train_decision_cutoff_utc"] = keys["__ts__"] - pd.Timedelta(hours=1)
    ev_map["execution_ev_map_available_at"] = keys["__ts__"]
    map_path = tmp_path / "execution_ev_map.parquet"
    ev_map.to_parquet(map_path, index=False)
    map_manifest = _write_signed(
        tmp_path / "execution_ev_map.manifest.json",
        {
            "prediction_role": materializer.EV_MAP_ROLE,
            "source_artifact_sha256": materializer._sha256(map_path),
            "oof_only": True,
            "prediction_scope": "oof",
        },
    )

    target_manifest = _write_signed(
        tmp_path / "execution_target.manifest.json",
        {
            "schema": "execution_ev_12h_hourly_policy_labels_v2",
            "prediction_role": "execution_ev_12h_labels",
            "timing": {
                "signal_timestamp": "__ts__",
                "first_path_timestamp": "__decision_ts__",
                "horizon_hours": 12,
            },
        },
    )
    timing = keys.copy()
    timing["execution_future_path"] = [_path(timestamp + pd.Timedelta(hours=1)) for timestamp in keys["__ts__"]]
    timing["atr_1h"] = 1.0
    timing["fee"] = 0.001
    timing["entry_spread"] = 10.0
    timing["exit_spread"] = 10.0
    timing_path = tmp_path / "timing_labels.parquet"
    timing.to_parquet(timing_path, index=False)
    timing_manifest = _write_signed(
        tmp_path / "timing_labels.manifest.json",
        {
            "schema": materializer.TIMING_PATH_SCHEMA,
            "prediction_role": materializer.TIMING_PATH_ROLE,
            "source_artifact_sha256": materializer._sha256(timing_path),
            "execution_ev_target_manifest_sha256": materializer._sha256(target_manifest),
            "execution_ev_target_signed_manifest_sha256": json.loads(target_manifest.read_text())["prediction_role_manifest_sha256"],
            "cost_accounting": "fee_once_entry_spread_once_exit_spread_once",
        },
    )
    return {
        "joined_handoff": handoff_path,
        "joined_handoff_provenance": joined_provenance,
        "execution_ev_oof": oof_path,
        "execution_ev_runner_manifest": runner_manifest,
        "execution_ev_map_oof": map_path,
        "execution_ev_map_manifest": map_manifest,
        "timing_labels": timing_path,
        "timing_labels_manifest": timing_manifest,
        "execution_ev_target_manifest": target_manifest,
    }


def _args(tmp_path: Path, paths: dict[str, Path]) -> SimpleNamespace:
    return SimpleNamespace(**paths, output=tmp_path / "timing_handoff.parquet", provenance_json=tmp_path / "timing_handoff.provenance.json")


def _resign_map(paths: dict[str, Path]) -> None:
    manifest = json.loads(paths["execution_ev_map_manifest"].read_text())
    manifest["source_artifact_sha256"] = materializer._sha256(paths["execution_ev_map_oof"])
    _write_signed(paths["execution_ev_map_manifest"], {key: value for key, value in manifest.items() if key != "prediction_role_manifest_sha256"})


def test_materializes_strict_post_execution_ev_timing_handoff(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    result = materializer.run(_args(tmp_path, paths))
    output = pd.read_parquet(result["handoff"])
    provenance = json.loads(result["provenance"].read_text())

    assert {"frozen_execution_ev", "frozen_ev_map", "frozen_alpha", "frozen_residual", "frozen_aux_time", "frozen_aux_peak", "frozen_aux_mae", "frozen_aux_turn", "frozen_aux_slope", "frozen_entropy", "catboost_archetype", "execution_future_path", "fee", "entry_spread", "exit_spread"}.issubset(output.columns)
    assert {f"frozen_p_{index}" for index in range(len(PATH_SHAPE_TYPES))}.issubset(output.columns)
    assert output["side_name"].tolist() == ["long", "short", "long", "short"]
    assert output["frozen_side_is_long"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert provenance["schema"] == materializer.SCHEMA
    assert provenance["handoff"]["join_mode"] == "exact_inner_one_to_one"

    specs = {name: EntryTimingFeatureProvenance(**record) for name, record in provenance["features"].items()}
    names, protected = validate_entry_timing_feature_contract(output, specs)
    assert protected == "frozen_execution_ev"
    assert "frozen_ev_map" in names


def test_rejects_exact_key_mismatch(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    ev_map = pd.read_parquet(paths["execution_ev_map_oof"])
    ev_map.loc[0, "candidate_id"] = "wrong-candidate"
    ev_map.to_parquet(paths["execution_ev_map_oof"], index=False)
    _resign_map(paths)

    with pytest.raises(ValueError, match="exact candidate identity coverage mismatch"):
        materializer.run(_args(tmp_path, paths))


def test_rejects_non_oof_execution_ev_prediction(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    oof = pd.read_parquet(paths["execution_ev_oof"])
    oof.loc[0, "direct__all_features__is_oof"] = False
    oof.to_parquet(paths["execution_ev_oof"], index=False)

    with pytest.raises(ValueError, match="in-sample/final-refit predictions are rejected"):
        materializer.run(_args(tmp_path, paths))


def test_rejects_missing_execution_ev_map(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    ev_map = pd.read_parquet(paths["execution_ev_map_oof"]).drop(columns="mapped_execution_ev")
    ev_map.to_parquet(paths["execution_ev_map_oof"], index=False)
    _resign_map(paths)

    with pytest.raises(ValueError, match="missing required columns: mapped_execution_ev"):
        materializer.run(_args(tmp_path, paths))
