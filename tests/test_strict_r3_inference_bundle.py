from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pandas as pd

from extreme_price_movements.strict_r3_inference_bundle import (
    StrictR3InferenceBundle,
    validate_live_feature_frame,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> Path:
    files = {}
    for name in (
        "conversion/four_week_conversion_bundle.joblib", "upstream/monthly_upstream_bundle.joblib",
        "geometry/frozen_geometry_k9.joblib", "feature.json", "universe.json", "reference_candidates",
        "reference_features", "ledger", "index", "bridge", "exit.json", "portfolio.json",
    ):
        p = tmp_path / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(name); files[name] = p
    conversion_manifest = tmp_path / "conversion/run_manifest.json"
    conversion_manifest.write_text(json.dumps({
        "cutoff": "2026-07-16T00:00:00Z", "end_exclusive": "2026-08-13T00:00:00Z",
        "geometry_refit_cadence": "never", "bundle_sha256": _sha(files["conversion/four_week_conversion_bundle.joblib"]),
        "geometry_bundle_sha256": "geometry-semantic",
    }))
    upstream_manifest = tmp_path / "upstream/run_manifest.json"
    upstream_manifest.write_text(json.dumps({
        "cutoff": "2026-07-16T00:00:00Z", "end_exclusive": "2026-08-13T00:00:00Z",
        "bundle_sha256": _sha(files["upstream/monthly_upstream_bundle.joblib"]),
    }))
    geometry_manifest = tmp_path / "geometry/run_manifest.json"; geometry_manifest.write_text("geometry manifest")
    paths = {
        "conversion_bundle_dir": "conversion", "upstream_bundle_dir": "upstream",
        "frozen_geometry_bundle": "geometry/frozen_geometry_k9.joblib", "feature_contract": "feature.json",
        "frozen_universe_manifest": "universe.json", "same_model_reference_candidates": "reference_candidates",
        "same_model_reference_features": "reference_features", "resolved_score_label_ledger": "ledger",
        "immediate_calibration_index": "index", "ev_bridge_bundle": "bridge", "exit_policy": "exit.json",
        "portfolio_policy": "portfolio.json",
    }
    hash_paths = {
        "conversion_bundle": files["conversion/four_week_conversion_bundle.joblib"], "conversion_manifest": conversion_manifest,
        "upstream_bundle": files["upstream/monthly_upstream_bundle.joblib"], "upstream_manifest": upstream_manifest,
        "frozen_geometry_bundle": files["geometry/frozen_geometry_k9.joblib"], "frozen_geometry_manifest": geometry_manifest,
        "feature_contract": files["feature.json"], "frozen_universe_manifest": files["universe.json"],
        "same_model_reference_candidates": files["reference_candidates"], "same_model_reference_features": files["reference_features"],
        "resolved_score_label_ledger": files["ledger"], "immediate_calibration_index": files["index"],
        "ev_bridge_bundle": files["bridge"], "exit_policy": files["exit.json"], "portfolio_policy": files["portfolio.json"],
    }
    payload = {
        "schema": "strict_r3_inference_bundle_v1", "scope": "long_only_shadow",
        "activation_ts": "2026-07-16T00:00:00Z", "end_exclusive_ts": "2026-08-13T00:00:00Z",
        "outside_window": "fail_closed", "paths": paths,
        "sha256": {name: _sha(path) for name, path in hash_paths.items()},
        "producer": {"conversion_bundle_sha256": _sha(files["conversion/four_week_conversion_bundle.joblib"]),
                     "upstream_bundle_sha256": _sha(files["upstream/monthly_upstream_bundle.joblib"]),
                     "geometry_bundle_sha256": "geometry-semantic"},
        "runtime": {"mode": "shadow-only", "exchange_io": False, "order_submission": False},
    }
    path = tmp_path / "bundle.json"; path.write_text(json.dumps(payload)); return path


def test_sealed_bundle_validates_inside_window(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    audit = StrictR3InferenceBundle.load(path, root=tmp_path).validate(decision_ts="2026-08-01T00:00:00Z")
    assert audit["hashes_verified"] == 15
    assert audit["mode"] == "shadow-only"


def test_sealed_bundle_fails_closed_after_expiry(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="outside sealed producer window"):
        StrictR3InferenceBundle.load(path, root=tmp_path).validate(decision_ts="2026-08-13T00:00:00Z")


def test_sealed_bundle_rejects_hash_drift(tmp_path: Path) -> None:
    path = _fixture(tmp_path); (tmp_path / "portfolio.json").write_text("drift")
    with pytest.raises(ValueError, match="hash mismatch"):
        StrictR3InferenceBundle.load(path, root=tmp_path).validate(decision_ts="2026-08-01T00:00:00Z")


def test_schema_v2_requires_complete_universe_and_current_spread_contract(
    tmp_path: Path,
) -> None:
    path = _fixture(tmp_path)
    payload = json.loads(path.read_text())
    payload["schema"] = "strict_r3_inference_bundle_v2"
    payload["runtime"].update({
        "feature_history_start": "2026-02-01T00:00:00Z",
        "candidate_feature_population": (
            "complete_frozen_universe_before_current_spread_filter"
        ),
        "current_spread_gate": "official_kraken_signal_hour_bid_ask_bps_le_100",
    })
    path.write_text(json.dumps(payload))
    audit = StrictR3InferenceBundle.load(path, root=tmp_path).validate(
        decision_ts="2026-08-01T00:00:00Z",
    )
    assert audit["schema"] == "strict_r3_inference_bundle_v2"

    payload["runtime"].pop("current_spread_gate")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="current_spread_gate"):
        StrictR3InferenceBundle.load(path, root=tmp_path)


def test_live_feature_frame_enforces_sealed_coverage_gates() -> None:
    requirements = {
        "minimum_row_feature_fraction": 0.5,
        "minimum_cycle_complete_fraction": 0.5,
        "minimum_per_field_finite_fraction": 0.5,
    }
    audit = validate_live_feature_frame(
        pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, None]}),
        fields=["a", "b"], requirements=requirements,
    )
    assert audit["minimum_per_field_finite_fraction"] == 0.5
    with pytest.raises(ValueError, match="feature parity gate"):
        validate_live_feature_frame(
            pd.DataFrame({"a": [1.0, 2.0], "b": [None, None]}),
            fields=["a", "b"], requirements=requirements,
        )


def test_schema_v4_requires_nine_month_posterior_contract(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    payload = json.loads(path.read_text())
    payload.update({
        "schema": "strict_r3_inference_bundle_v4_28d_r5_9m_posterior",
        "reference_window_days": 28,
        "admission_contract": (
            "strict_r3_cell_day_residual_trust_posterior_28d_challenger_v1"
        ),
        "trust_overlay_contract": (
            "strict_r3_cell_day_residual_trust_model_r5_9m_v1"
        ),
    })
    payload["runtime"].update({
        "feature_history_start": "2026-02-01T00:00:00Z",
        "candidate_feature_population": (
            "complete_frozen_universe_before_current_spread_filter"
        ),
        "current_spread_gate": "official_kraken_signal_hour_bid_ask_bps_le_100",
    })
    path.write_text(json.dumps(payload))
    bundle = StrictR3InferenceBundle.load(path, root=tmp_path)
    assert bundle.payload["schema"].endswith("r5_9m_posterior")

    payload["trust_overlay_contract"] = "strict_r3_cell_day_residual_trust_overlay_v1"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="canonical R5 model"):
        StrictR3InferenceBundle.load(path, root=tmp_path)
