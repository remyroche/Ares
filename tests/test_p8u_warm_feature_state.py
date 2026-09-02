from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements.inference.p8u_warm_feature_state import (
    P8UWarmFeatureConfig,
    P8UWarmFeatureRequest,
    P8U_REQUIRED_STATE_KINDS,
    assert_next_hour,
    audit_feature_parity,
    feature_union_sha256,
)
from scripts.run_strict_r3_p8u_warm_feature_worker import _worker_command


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_bundle(root: Path, plan_sha: str, *, matching: bool = True) -> Path:
    bundle = root / "state_bundle"
    bundle.mkdir()
    inventory = pd.DataFrame({
        "kind": sorted(P8U_REQUIRED_STATE_KINDS),
        "last_timestamp": ["2026-08-01T00:00:00Z"] * len(P8U_REQUIRED_STATE_KINDS),
    })
    inventory.to_parquet(bundle / "operator_state_inventory.parquet", index=False)
    manifest = {
        "schema": "strict_r3_causal_feature_state_bundle_v2",
        "feature_contract_sha256": plan_sha if matching else "incorrect",
        "scope": f"p8u-long-router50-2-{plan_sha[:16]}",
        "latest_state_timestamp": "2026-08-01T00:00:00Z",
    }
    (bundle / "state_bundle_manifest.json").write_text(json.dumps(manifest))
    return bundle


def _config(root: Path, plan: Path, bundle: Path) -> Path:
    canonical = root / "canonical.json"
    canonical.write_text("{}")
    config = root / "config.json"
    union = feature_union_sha256(["f_router", "f_under"])
    config.write_text(json.dumps({
        "schema": "strict_r3_p8u_warm_feature_worker_config_v1",
        "side": "long",
        "exchange_io": False,
        "order_submission": False,
        "state_name": "p8u-test-state",
        "state_contract_id": f"p8u-long-router50-2-{union[:16]}",
        "work_root": "worker",
        "canonical_contract_path": "canonical.json",
        "canonical_contract_sha256": _sha(canonical),
        "feature_plan_path": str(plan.relative_to(root)),
        "feature_plan_file_sha256": _sha(plan),
        "feature_union_sha256": union,
        "feature_count": 2,
        "stateful_tail_hours": 72,
        "required_state_kinds": sorted(P8U_REQUIRED_STATE_KINDS),
        "initial_state_bundle": str(bundle.relative_to(root)),
        "initial_state_bundle_manifest_sha256": _sha(bundle / "state_bundle_manifest.json"),
        "parity_required": True,
    }))
    return config


def _feature_frame(values: tuple[float, float]) -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["A|2026-08-01T01:00:00Z|long"],
        "__decision_ts__": [pd.Timestamp("2026-08-01T01:00:00Z")],
        "side_name": ["long"],
        "f_router": [values[0]],
        "f_under": [values[1]],
    })


def test_same_plan_state_bundle_is_required_and_request_is_exact_hour(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"full_union": ["f_router", "f_under"]}))
    bundle = _state_bundle(tmp_path, feature_union_sha256(["f_router", "f_under"]))
    config = P8UWarmFeatureConfig.load(_config(tmp_path, plan, bundle), root=tmp_path)
    assert config.require_state_bundle() == bundle

    candidates = tmp_path / "candidates.parquet"
    frame = _feature_frame((1.0, 2.0)).assign(
        __ts__=pd.Timestamp("2026-08-01T01:00:00Z"), __symbol__="A/USD:USD"
    )
    frame.to_parquet(candidates, index=False)
    panel = tmp_path / "panel.joblib"
    panel.write_bytes(b"state")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps({
        "schema": "strict_r3_p8u_warm_feature_request_v1",
        "signal_ts": "2026-08-01T01:00:00Z",
        "candidates": "candidates.parquet",
        "panel_state": "panel.joblib",
        "outcome_columns_consumed": [],
    }))
    request = P8UWarmFeatureRequest.load(request_path, root=tmp_path)
    assert request.validate_candidate_timestamp() == 1
    assert_next_hour(
        request.signal_ts,
        ledger=None,
        bundle_latest_timestamp="2026-08-01T00:00:00Z",
    )
    with pytest.raises(ValueError, match="exactly one hourly"):
        assert_next_hour(
            pd.Timestamp("2026-08-01T02:00:00Z"),
            ledger=None,
            bundle_latest_timestamp="2026-08-01T00:00:00Z",
        )


def test_mismatched_legacy_state_bundle_is_refused(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"full_union": ["f_router", "f_under"]}))
    bundle = _state_bundle(tmp_path, feature_union_sha256(["f_router", "f_under"]), matching=False)
    config = P8UWarmFeatureConfig.load(_config(tmp_path, plan, bundle), root=tmp_path)
    with pytest.raises(ValueError, match="not bootstrapped"):
        config.require_state_bundle()


def test_worker_restores_by_stable_union_not_generated_plan_file_hash(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"full_union": ["f_router", "f_under"]}))
    bundle = _state_bundle(tmp_path, feature_union_sha256(["f_router", "f_under"]))
    config = P8UWarmFeatureConfig.load(_config(tmp_path, plan, bundle), root=tmp_path)
    request = P8UWarmFeatureRequest(
        path=tmp_path / "request.json",
        signal_ts=pd.Timestamp("2026-08-01T01:00:00Z"),
        candidates=tmp_path / "candidates.parquet",
        panel_state=tmp_path / "panel.joblib",
        reference_features=None,
    )
    command = _worker_command(
        config=config,
        request=request,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        initial_bundle=bundle,
    )
    value = command[command.index("--expected-state-contract-hash") + 1]
    assert value == config.feature_union_sha256
    assert value != _sha(plan)


def test_full_union_parity_is_reported_per_field_and_fails_on_delta(tmp_path: Path) -> None:
    incremental = tmp_path / "incremental.parquet"
    reference = tmp_path / "reference.parquet"
    _feature_frame((1.0, 2.0)).to_parquet(incremental, index=False)
    _feature_frame((1.0, 2.0)).to_parquet(reference, index=False)
    summary = audit_feature_parity(
        incremental_features=incremental,
        reference_features=reference,
        required_features=("f_router", "f_under"),
        out_dir=tmp_path / "pass",
    )
    assert summary["status"] == "pass"
    fields = pd.read_parquet(tmp_path / "pass/feature_parity_by_field.parquet")
    assert fields.set_index("feature").loc["f_router", "status"] == "pass"

    _feature_frame((1.0, 2.5)).to_parquet(reference, index=False)
    summary = audit_feature_parity(
        incremental_features=incremental,
        reference_features=reference,
        required_features=("f_router", "f_under"),
        out_dir=tmp_path / "fail",
    )
    assert summary["status"] == "fail"
    assert summary["failing_fields"] == ["f_under"]
