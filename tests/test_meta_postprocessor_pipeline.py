from __future__ import annotations

import json

from extreme_price_movements.meta_postprocessor_pipeline import (
    ARTIFACT_NAME,
    POLICY_ID,
    meta_postprocessor_enabled,
    run_scoped_postprocessor_artifact,
    validate_meta_postprocessor_bundle,
)


def _bundle(tmp_path):
    bundle = tmp_path / "artifacts" / "run1" / "meta_postprocessors"
    models = bundle / "policy_models"
    models.mkdir(parents=True)
    (models / "local.joblib").write_bytes(b"placeholder")
    (bundle / "manifest.json").write_text("{}\n")
    artifact = {
        "policy_id": POLICY_ID,
        "predecessor": (
            "meta_residual_extreme_local_champion_overlay_ooftrain_tieaware_"
            "downonly_20260712_v9::forced_local_tail_0.950"
        ),
        "blacklisted_side_archetypes": [
            "long||long_dirtyavoid_sparse_questionable"
        ],
        "strict_required_features": True,
        "effects": [{"model_path": "policy_models/local.joblib"}],
        "expected_ev_mapping": {
            "schema": "hierarchical_monotonic_expected_ev_v1",
            "local": {"short||breakout": {}},
        },
    }
    (bundle / ARTIFACT_NAME).write_text(json.dumps(artifact) + "\n")
    return bundle


def test_run_scoped_bundle_validation_and_resolution(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    result = validate_meta_postprocessor_bundle(bundle)
    assert result["policy_id"] == POLICY_ID
    assert result["model_count"] == 1
    assert run_scoped_postprocessor_artifact(tmp_path, "run1") == (
        bundle / ARTIFACT_NAME
    )


def test_meta_postprocessor_is_explicitly_enabled(monkeypatch) -> None:
    assert not meta_postprocessor_enabled({})
    assert meta_postprocessor_enabled({"meta_postprocessor_enabled": True})
    monkeypatch.setenv("EPM_META_POSTPROCESSOR_ENABLED", "0")
    assert not meta_postprocessor_enabled({"meta_postprocessor_enabled": True})
