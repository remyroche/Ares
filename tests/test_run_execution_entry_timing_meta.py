from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_execution_entry_timing_meta.py"
_SPEC = importlib.util.spec_from_file_location("run_execution_entry_timing_meta", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)

_HELPER = Path(__file__).resolve().with_name("test_execution_entry_timing_meta.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("entry_timing_test_helpers", _HELPER)
assert _HELPER_SPEC is not None and _HELPER_SPEC.loader is not None
helpers = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(helpers)


def _target_manifest(path: Path) -> Path:
    geometry = helpers._geometry()
    payload = {
        "schema": "execution_ev_12h_hourly_policy_labels_v2",
        "prediction_role": "execution_ev_12h_labels",
        "timing": {
            "signal_timestamp": "__ts__",
            "decision_delay_hours": 1,
            "first_path_timestamp": "__decision_ts__",
            "horizon_hours": 12,
            "label_end": "__decision_ts__ + 12h",
        },
        "policy": {
            "sha256": "policy-manifest-sha256",
            "long_geometry": geometry,
            "short_geometry": geometry,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["prediction_role_manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _args(
    input_path: Path, provenance_path: Path, target_manifest_path: Path, output_dir: Path
) -> SimpleNamespace:
    return SimpleNamespace(
        input=input_path,
        provenance_json=provenance_path,
        execution_ev_target_manifest=target_manifest_path,
        output_dir=output_dir,
        timestamp_col="__decision_ts__",
        side_col="side_name",
        archetype_col="catboost_archetype",
        label_end_time_col="execution_label_end_utc",
        path_col="execution_future_path",
        atr_col="atr_1h",
        decision_price_col=None,
        cost_return_col=None,
        fee_return_col="fee",
        entry_spread_bps_col="entry_spread",
        exit_spread_bps_col="exit_spread",
        allow_action_invariant_all_in_cost=False,
        horizon_hours=12.0,
        wait_minutes=(2,),
        adverse_offset_atr=(0.5,),
        n_splits=2,
        min_train_rows=4,
        purge_hours=1.0,
        embargo_hours=1.0,
        n_estimators=16,
        early_stopping_rounds=4,
        hpo_trials=0,
        decision_hpo_trials=0,
        n_jobs=1,
        dry_run=True,
    )


def test_runner_dry_run_writes_strict_manifest(tmp_path) -> None:
    frame, provenance = helpers._strict_frame(8, with_path=True, path_minutes=720)
    input_path = tmp_path / "handoff.parquet"
    provenance_path = tmp_path / "provenance.json"
    target_manifest_path = _target_manifest(tmp_path / "execution_ev_target.manifest.json")
    frame.to_parquet(input_path, index=False)
    provenance_path.write_text(
        json.dumps({"features": {name: vars(spec) for name, spec in provenance.items()}}),
        encoding="utf-8",
    )
    paths = runner.run(
        _args(input_path, provenance_path, target_manifest_path, tmp_path / "output")
    )
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["counterfactual_rows"] == len(frame) * 3
    assert manifest["protected_execution_ev_feature"] == "frozen_execution_ev"
    assert manifest["execution_ev_target"]["policy_manifest_sha256"] == "policy-manifest-sha256"
    assert manifest["target_spec"]["long_policy_geometry"]["sl_mult"] == 3.0
    assert "train-OOF isotonic" in manifest["leakage_contract"]
