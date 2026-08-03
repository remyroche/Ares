from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements.stage_iii_artifacts import (
    StageIIIArtifactError,
    StageIIIReproducibilityManifest,
    publish_stage_iii_compact_bundle,
)
from extreme_price_movements.stage_iii_reporting import (
    StageIIIReportTables,
)


def _hash(seed: str) -> str:
    return sha256(seed.encode()).hexdigest()


def _manifest() -> StageIIIReproducibilityManifest:
    return StageIIIReproducibilityManifest(
        run_id="stage3_test", dataset_id="dataset_v1", dataset_sha256=_hash("dataset"),
        label_manifest_id="tp6_sl4_h12", label_manifest_sha256=_hash("labels"),
        feature_contract_sha256=_hash("features"),
        input_lineage_contract_sha256=_hash("lineage"), code_revision="deadbeef",
        split_definition={"kind": "expanding_environment"},
        model_configuration={"routing": "one_shared_model"}, random_seeds=(17,),
    )


@dataclass
class _Arm:
    arm: str
    round_name: str
    oof_predictions: pd.DataFrame
    metrics: pd.DataFrame
    fold_audit: pd.DataFrame
    calibration_audit: pd.DataFrame
    model_feature_names: tuple[str, ...] = ("meta_context", "meta_trust")


@dataclass
class _Result:
    schema: str
    winner: _Arm
    arms: tuple[_Arm, ...]
    arm_summary: pd.DataFrame
    transport_matrix: pd.DataFrame
    advancement_gates: dict[str, object]
    round_winners: dict[str, str]


def _result() -> _Result:
    predictions = pd.DataFrame({
        "candidate_id": ["c1", "c2"], "symbol": ["BTC", "ETH"],
        "decision_ts": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        "side_name": ["long", "short"], "score_bps": [10.0, 20.0],
        "exact_net_bps": [5.0, 30.0],
    })
    arm = _Arm(
        arm="E2", round_name="E_calibration", oof_predictions=predictions,
        metrics=pd.DataFrame({"scope": ["pooled"], "net_bps": [17.5]}),
        fold_audit=pd.DataFrame({"fold": [1]}),
        calibration_audit=pd.DataFrame({"anchor": ["day"]}),
    )
    return _Result(
        schema="runner_v1", winner=arm, arms=(arm,),
        arm_summary=pd.DataFrame({"arm": ["E2"]}),
        transport_matrix=pd.DataFrame({"train": ["era1"], "test": ["era2"]}),
        advancement_gates={"advances": False}, round_winners={"E": "E2"},
    )


def test_compact_publication_is_atomic_hashed_and_storage_bounded(tmp_path: Path) -> None:
    reports = StageIIIReportTables(
        schema="stage_iii_pooled_global_reporting_v1",
        tail_summary=pd.DataFrame({"layer": ["meta"]}),
        selected_attribution=pd.DataFrame({"scope": ["week_side"]}),
        residual_diagnostics=pd.DataFrame({"mean": [1.0]}),
        time_concentration=pd.DataFrame({"hhi": [0.5]}),
        hit_surprise=pd.DataFrame({"horizon": ["3d"]}),
    )
    output = publish_stage_iii_compact_bundle(
        _result(), tmp_path / "bundle", reproducibility=_manifest(),
        winner_prediction_columns=(
            "candidate_id", "symbol", "decision_ts", "side_name", "score_bps", "exact_net_bps",
        ),
        report_tables=reports,
    )
    manifest = json.loads((output / "run_manifest.json").read_text())
    checksums = json.loads((output / "checksums.json").read_text())
    assert manifest["storage_policy"] == "winner_rows_plus_all_arm_compact_metrics"
    assert manifest["evaluation_status"] == "DEVELOPMENT_STRICT_OOF_NOT_FINAL_TEST"
    assert "winner_oof_predictions.parquet" in checksums
    assert set(manifest["report_tables"]) == {
        "tail_summary", "selected_attribution", "residual_diagnostics",
        "time_concentration", "hit_surprise",
    }
    assert "selected_attribution.parquet" in checksums
    assert "feature_lists.json" in checksums
    assert "all_arm_feature_contracts.parquet" in checksums
    assert json.loads((output / "feature_lists.json").read_text()) == {
        "meta_residual": ["meta_context", "meta_trust"],
    }
    assert manifest["feature_list_layers"] == ["meta_residual"]
    all_contracts = pd.read_parquet(output / "all_arm_feature_contracts.parquet")
    assert all_contracts.feature_names_json.tolist() == ['["meta_context","meta_trust"]']
    assert not any(path.name.startswith("arm_oof_") for path in output.iterdir())
    with pytest.raises(StageIIIArtifactError, match="already exists"):
        publish_stage_iii_compact_bundle(
            _result(), output, reproducibility=_manifest(),
            winner_prediction_columns=("candidate_id", "symbol", "decision_ts", "side_name"),
        )


def test_publication_requires_identity_and_nonplaceholder_hashes(tmp_path: Path) -> None:
    with pytest.raises(StageIIIArtifactError, match="identity"):
        publish_stage_iii_compact_bundle(
            _result(), tmp_path / "missing_identity", reproducibility=_manifest(),
            winner_prediction_columns=("candidate_id", "decision_ts", "side_name"),
        )


def test_publication_requires_exact_nonduplicated_feature_lists(tmp_path: Path) -> None:
    with pytest.raises(StageIIIArtifactError, match="duplicates"):
        publish_stage_iii_compact_bundle(
            _result(), tmp_path / "duplicate_features", reproducibility=_manifest(),
            winner_prediction_columns=("candidate_id", "symbol", "decision_ts", "side_name"),
            feature_lists={"base": ("f1", "f1")},
        )
    output = publish_stage_iii_compact_bundle(
        _result(), tmp_path / "all_layers", reproducibility=_manifest(),
        winner_prediction_columns=("candidate_id", "symbol", "decision_ts", "side_name"),
        feature_lists={"base": ("base_a", "base_b"), "meta": ("meta_a",)},
    )
    assert json.loads((output / "feature_lists.json").read_text()) == {
        "base": ["base_a", "base_b"],
        "meta": ["meta_a"],
    }
    invalid = StageIIIReproducibilityManifest(
        **{**_manifest().__dict__, "dataset_sha256": "0" * 64}
    )
    with pytest.raises(StageIIIArtifactError, match="dataset_sha256"):
        publish_stage_iii_compact_bundle(
            _result(), tmp_path / "invalid_hash", reproducibility=invalid,
            winner_prediction_columns=("candidate_id", "symbol", "decision_ts", "side_name"),
        )
