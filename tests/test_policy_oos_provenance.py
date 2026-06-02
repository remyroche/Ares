import json
from pathlib import Path

from extreme_price_movements.policy_oos_provenance import (
    provenance_manifest_path,
    sha256_file,
    validate_policy_oos_source_artifacts,
)


def _source_validation():
    return {
        "policy_optimiser_fit_end": "2026-01-19T06:00:00+00:00",
        "policy_optimiser_predict_start": "2026-01-22T10:00:00+00:00",
        "policy_optimiser_predict_end": "2026-05-22T00:00:00+00:00",
        "oos_policy_slice_verified": True,
    }


def _touch_artifacts(run_root: Path) -> tuple[Path, Path]:
    base = run_root / "base_models_intermediate.pkl"
    meta = run_root / "models" / "model_state_meta.pkl"
    meta.parent.mkdir(parents=True)
    base.write_bytes(b"base")
    meta.write_bytes(b"meta")
    return base, meta


def _write_manifest(path: Path, *, role: str, slice_hash: str, final_fit: bool = False):
    provenance_manifest_path(path).write_text(
        json.dumps(
            {
                "source_model_fit_start": "2023-03-12T23:00:00+00:00",
                "source_model_fit_end": "2026-01-19T06:00:00+00:00",
                "source_slice_role": role,
                "generated_from_final_fit_bundle": final_fit,
                "slice_plan_sha256": slice_hash,
                "feature_contract_hash": f"{role}:features",
            }
        ),
        encoding="utf-8",
    )


def test_policy_oos_preflight_requires_artifact_manifests(tmp_path):
    run_root = tmp_path / "artifacts" / "run"
    run_root.mkdir(parents=True)
    slice_plan = run_root / "slices" / "slice_plan.json"
    slice_plan.parent.mkdir()
    slice_plan.write_text("{}", encoding="utf-8")
    _touch_artifacts(run_root)

    report = validate_policy_oos_source_artifacts(
        run_root=run_root,
        slice_plan_path=slice_plan,
        source_validation=_source_validation(),
    )

    assert report["valid"] is False
    assert "base_artifact_not_policy_oos_safe" in report["errors"]
    assert "meta_artifact_not_policy_oos_safe" in report["errors"]
    assert report["artifacts"]["base"]["errors"] == [
        "missing_artifact_provenance_manifest"
    ]


def test_policy_oos_preflight_accepts_pre_policy_base_and_meta(tmp_path):
    run_root = tmp_path / "artifacts" / "run"
    run_root.mkdir(parents=True)
    slice_plan = run_root / "slices" / "slice_plan.json"
    slice_plan.parent.mkdir()
    slice_plan.write_text('{"version": 1}', encoding="utf-8")
    base, meta = _touch_artifacts(run_root)
    slice_hash = sha256_file(slice_plan)
    _write_manifest(base, role="base_model_fit", slice_hash=slice_hash)
    _write_manifest(meta, role="meta_model_fit", slice_hash=slice_hash)

    report = validate_policy_oos_source_artifacts(
        run_root=run_root,
        slice_plan_path=slice_plan,
        source_validation=_source_validation(),
    )

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["source_model_fit_end"] == "2026-01-19T06:00:00+00:00"


def test_policy_oos_preflight_rejects_final_fit_or_slice_mismatch(tmp_path):
    run_root = tmp_path / "artifacts" / "run"
    run_root.mkdir(parents=True)
    slice_plan = run_root / "slices" / "slice_plan.json"
    slice_plan.parent.mkdir()
    slice_plan.write_text('{"version": 1}', encoding="utf-8")
    base, meta = _touch_artifacts(run_root)
    slice_hash = sha256_file(slice_plan)
    _write_manifest(base, role="full_inference_fit", slice_hash=slice_hash, final_fit=True)
    _write_manifest(meta, role="meta_model_fit", slice_hash="wrong")

    report = validate_policy_oos_source_artifacts(
        run_root=run_root,
        slice_plan_path=slice_plan,
        source_validation=_source_validation(),
    )

    assert report["valid"] is False
    assert "generated_from_final_fit_bundle" in report["artifacts"]["base"]["errors"]
    assert "unexpected_source_slice_role" in report["artifacts"]["base"]["errors"]
    assert "slice_plan_sha256_mismatch" in report["artifacts"]["meta"]["errors"]
