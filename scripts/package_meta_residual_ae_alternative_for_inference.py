#!/usr/bin/env python3
"""Package the promoted residual-aware AE/GMM meta alternative for inference."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.alternative_meta_residual_bundle import (
    AlternativeMetaResidualBundle,  # noqa: E402
)
from extreme_price_movements.features_gmm_ae import (
    save_ae_gmm_state_artifact,  # noqa: E402
)
from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    ResidualArchetypeConfig,
    ResidualArchetypeRecognizer,
)
from scripts.package_meta_residual_alternative_for_inference import (  # noqa: E402
    DEPLOYED_AE_GMM,
    TRAINING_AE_GMM,
)
from scripts.run_meta_residual_ae_representation_ablation import (  # noqa: E402
    ARM,
    _candidate_features,
)
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_OUT_DIR,
    _add_reference_fold_features,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = DEFAULT_OUT_DIR
    package_dir = root / "inference_bundle_residual_ae_gmm"
    package_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(root / "cache" / "compact_reference_with_lifecycle.parquet")
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    july_start = pd.Timestamp("2026-07-01", tz="UTC")
    train = data[data["__ts__"].lt(july_start)].copy()
    valid = data[data["__ts__"].ge(july_start)].copy()
    candidates = _candidate_features(data, root)
    recognizer = ResidualArchetypeRecognizer(
        ResidualArchetypeConfig(
            use_residual_ae_gmm=True,
            ae_gmm_max_rows=5_000,
            ae_gmm_max_iter=80,
            random_state=20260711,
        ),
        candidates,
    ).fit(train)
    recognizer_path = package_dir / "residual_ae_gmm_recognizer.joblib"
    joblib.dump(recognizer, recognizer_path, compress=3)
    recognizer.catalog_.to_csv(
        package_dir / "residual_archetype_catalog.csv", index=False
    )
    residual_state_paths = save_ae_gmm_state_artifact(
        recognizer.ae_gmm_state,
        package_dir / "residual_ae_gmm_state.pkl",
        manifest_path=package_dir / "residual_ae_gmm_state_manifest.json",
        extra_manifest={
            "role": "alternative_meta_residual_archetype_recognizer",
            "fit_through": str(recognizer.train_end_),
            "inference_transform_only": True,
        },
    )

    lifecycle_artifact = joblib.load(
        root
        / "lifecycle_only_final_through_june"
        / "latest_oos_fold_model_july_oos_parity.joblib"
    )
    overlay = joblib.load(root / ARM / "residual_overlay_state.joblib")
    overlay_selection = json.loads((root / ARM / "overlay_selection.json").read_text())
    arm_manifest = json.loads((root / ARM / "manifest.json").read_text())
    calibrator = joblib.load(root / ARM / "hit_calibrator.joblib")
    historical_rank_path = (
        package_dir / "alternative_meta_historical_rank_reference.joblib"
    )
    historical_rank_reference = (
        joblib.load(historical_rank_path) if historical_rank_path.exists() else None
    )
    deployed_hash = _sha256(DEPLOYED_AE_GMM)
    training_hash = _sha256(TRAINING_AE_GMM)
    if deployed_hash != training_hash:
        raise RuntimeError("Existing frozen AE/GMM state mismatch")
    shutil.copy2(DEPLOYED_AE_GMM, package_dir / "base_ae_gmm_state.pkl")
    bundle = AlternativeMetaResidualBundle(
        lifecycle_model=lifecycle_artifact["model"],
        selected_features=list(lifecycle_artifact["selected_features"]),
        raw_selected_features=list(lifecycle_artifact["raw_selected_features"]),
        feature_medians=dict(lifecycle_artifact["feature_medians"]),
        ood_state=dict(lifecycle_artifact["ood_state"]),
        residual_recognizer=recognizer,
        overlay_state=overlay,
        hit_calibrator=calibrator,
        historical_rank_reference=historical_rank_reference,
        fit_through=str(lifecycle_artifact["fit_through"]),
        frozen_ae_gmm_sha256=deployed_hash,
        metadata={
            "role": "promoted_alternative_meta_residual_aware_ae_gmm",
            "representation_seed": 20260711,
            "current_meta_model_overwritten": False,
            "base_model_retrained": False,
            "formal_oos_months": ["2026-04", "2026-05", "2026-06"],
            "overlay_selection": overlay_selection,
            "hit_calibration": arm_manifest["hit_calibration"],
        },
    )
    bundle_path = package_dir / "alternative_meta_residual_ae_gmm_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    restored = joblib.load(bundle_path)

    train_enriched, valid_enriched = _add_reference_fold_features(train, valid)
    del train_enriched
    predicted = bundle.predict(valid_enriched)
    restored_predicted = restored.predict(valid_enriched)
    max_diff = (
        float(
            np.max(
                np.abs(
                    predicted.to_numpy(dtype=np.float32)
                    - restored_predicted.to_numpy(dtype=np.float32)
                )
            )
        )
        if len(predicted)
        else float("nan")
    )
    preview = valid_enriched[
        ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
    ].copy()
    preview[predicted.columns] = predicted.to_numpy(dtype=np.float32)
    preview.to_parquet(
        package_dir / "july_oos_inference_preview.parquet",
        index=False,
        compression="zstd",
    )
    manifest = bundle.manifest()
    manifest.update(
        {
            "bundle_path": str(bundle_path),
            "recognizer_path": str(recognizer_path),
            "recognizer_catalog_path": str(
                package_dir / "residual_archetype_catalog.csv"
            ),
            "residual_ae_gmm_state_artifact": residual_state_paths,
            "base_ae_gmm_training_sha256": training_hash,
            "base_ae_gmm_deployed_sha256": deployed_hash,
            "base_ae_gmm_hash_match": training_hash == deployed_hash,
            "july_parity_rows": int(len(predicted)),
            "bundle_roundtrip_max_abs_diff": max_diff,
            "inference_parity_pass": bool(
                len(predicted) > 0
                and max_diff <= 1e-7
                and training_hash == deployed_hash
                and bool(recognizer.ae_gmm_state.get("enabled", False))
            ),
        }
    )
    (package_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "bundle": str(bundle_path),
                "fit_through": bundle.fit_through,
                "july_parity_rows": len(predicted),
                "max_abs_diff": max_diff,
                "base_ae_gmm_hash_match": training_hash == deployed_hash,
                "residual_ae_gmm_enabled": bool(
                    recognizer.ae_gmm_state.get("enabled", False)
                ),
                "historical_rank_reference_path": (
                    str(historical_rank_path)
                    if historical_rank_reference is not None
                    else None
                ),
                "historical_rank_embedded": historical_rank_reference is not None,
                "inference_parity_pass": manifest["inference_parity_pass"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
