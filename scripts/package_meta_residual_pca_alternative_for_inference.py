#!/usr/bin/env python3
"""Package the corrected PCA residual-meta alternative for frozen inference."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.alternative_meta_residual_bundle import (  # noqa: E402
    AlternativeMetaResidualBundle,
    _apply_residual_representation,
)
from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    ResidualArchetypeConfig,
    ResidualArchetypeRecognizer,
)
from scripts.package_meta_residual_alternative_for_inference import (  # noqa: E402
    DEPLOYED_AE_GMM,
    TRAINING_AE_GMM,
)
from scripts.run_meta_residual_ae_representation_ablation import (
    _candidate_features,  # noqa: E402
)
from scripts.run_meta_residual_pca_representation_ablation import (  # noqa: E402
    _fit_pca,
    _transform_pca,
)
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_OUT_DIR,
    _add_reference_fold_features,
)

ARM = "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_globaloverlay"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    root = DEFAULT_OUT_DIR
    package_dir = root / "inference_bundle_residual_pca8_globaloverlay"
    package_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(root / "cache" / "compact_reference_with_lifecycle.parquet")
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    july_start = pd.Timestamp("2026-07-01", tz="UTC")
    train = data[data["__ts__"].lt(july_start)].copy()
    valid = data[data["__ts__"].ge(july_start)].copy()
    candidates = _candidate_features(data, root)
    pca_inputs = candidates[: min(80, len(candidates))]
    pca_state = _fit_pca(
        train,
        pca_inputs,
        seed=20260711,
        requested_components=8,
        scaled_clip=8.0,
    )
    train_pca = _transform_pca(train, pca_state)
    valid_pca = _transform_pca(valid, pca_state)
    output_columns = train_pca.columns.astype(str).tolist()
    for name in output_columns:
        train[name] = train_pca[name].to_numpy(dtype=np.float32, copy=False)
        valid[name] = valid_pca[name].to_numpy(dtype=np.float32, copy=False)
    representation_state = dict(pca_state)
    representation_state.update(
        {
            "kind": "robust_pca",
            "output_columns": output_columns,
            "fit_through": str(train["__ts__"].max()),
        }
    )
    representation_path = package_dir / "residual_pca8_clip8_state.joblib"
    joblib.dump(representation_state, representation_path, compress=3)

    recognizer = ResidualArchetypeRecognizer(
        ResidualArchetypeConfig(
            use_residual_ae_gmm=False,
            random_state=20260711,
        ),
        list(dict.fromkeys([*candidates, *output_columns])),
    ).fit(train)
    recognizer_path = package_dir / "residual_pca8_recognizer.joblib"
    joblib.dump(recognizer, recognizer_path, compress=3)
    recognizer.catalog_.to_csv(
        package_dir / "residual_archetype_catalog.csv", index=False
    )

    lifecycle_artifact = joblib.load(
        root
        / "lifecycle_only_final_through_june"
        / "latest_oos_fold_model_july_oos_parity.joblib"
    )
    overlay = joblib.load(root / ARM / "residual_overlay_state.joblib")
    calibrator = joblib.load(root / ARM / "hit_calibrator.joblib")
    historical_path = (
        root / f"historical_rank_oos_{ARM}" / "historical_rank_reference.joblib"
    )
    historical_rank = joblib.load(historical_path)
    if not DEPLOYED_AE_GMM.exists() or not TRAINING_AE_GMM.exists():
        raise FileNotFoundError(
            "Required frozen base AE/GMM parity artifacts are missing"
        )
    deployed_hash = _sha256(DEPLOYED_AE_GMM)
    training_hash = _sha256(TRAINING_AE_GMM)
    if deployed_hash != training_hash:
        raise RuntimeError("Existing frozen base AE/GMM state mismatch")
    base_ae_path = package_dir / "base_ae_gmm_state.pkl"
    shutil.copy2(DEPLOYED_AE_GMM, base_ae_path)

    bundle = AlternativeMetaResidualBundle(
        lifecycle_model=lifecycle_artifact["model"],
        selected_features=list(lifecycle_artifact["selected_features"]),
        raw_selected_features=list(lifecycle_artifact["raw_selected_features"]),
        feature_medians=dict(lifecycle_artifact["feature_medians"]),
        ood_state=dict(lifecycle_artifact["ood_state"]),
        residual_recognizer=recognizer,
        overlay_state=overlay,
        residual_representation_state=representation_state,
        hit_calibrator=calibrator,
        historical_rank_reference=historical_rank,
        fit_through=str(lifecycle_artifact["fit_through"]),
        frozen_ae_gmm_sha256=deployed_hash,
        metadata={
            "role": "alternative_meta_residual_pca8_clip8_globaloverlay",
            "representation_seed": 20260711,
            "current_meta_model_overwritten": False,
            "base_model_retrained": False,
            "formal_oos_months": ["2026-04", "2026-05", "2026-06"],
            "representation_selection": "three_seed_causal_historical_rank_and_placebo_suite",
            "local_overlay_normalization": False,
        },
    )
    bundle_path = package_dir / "alternative_meta_residual_pca8_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    restored = joblib.load(bundle_path)

    _train_enriched, valid_enriched = _add_reference_fold_features(
        data[data["__ts__"].lt(july_start)].copy(),
        data[data["__ts__"].ge(july_start)].copy(),
    )
    predicted = bundle.predict(valid_enriched)
    restored_predicted = restored.predict(valid_enriched)
    roundtrip_diff = (
        float(
            np.nanmax(
                np.abs(
                    predicted.to_numpy(dtype=np.float32)
                    - restored_predicted.to_numpy(dtype=np.float32)
                )
            )
        )
        if len(predicted)
        else np.nan
    )
    direct_representation = _transform_pca(valid_enriched, pca_state)
    bundled_representation = _apply_residual_representation(
        valid_enriched,
        representation_state,
    ).reindex(columns=output_columns)
    representation_diff = (
        float(
            np.nanmax(
                np.abs(
                    direct_representation.to_numpy(dtype=np.float32)
                    - bundled_representation.to_numpy(dtype=np.float32)
                )
            )
        )
        if len(valid_enriched)
        else np.nan
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
            "representation_path": str(representation_path),
            "historical_rank_reference_path": str(historical_path),
            "base_ae_gmm_path": str(base_ae_path),
            "base_ae_gmm_training_sha256": training_hash,
            "base_ae_gmm_deployed_sha256": deployed_hash,
            "base_ae_gmm_hash_match": training_hash == deployed_hash,
            "pca_effective_rank": float(pca_state["effective_rank"]),
            "pca_explained_variance_sum": float(
                sum(pca_state["explained_variance_ratio"])
            ),
            "july_parity_rows": int(len(predicted)),
            "bundle_roundtrip_max_abs_diff": roundtrip_diff,
            "representation_transform_max_abs_diff": representation_diff,
            "historical_rank_embedded": True,
            "inference_parity_pass": bool(
                len(predicted) > 0
                and roundtrip_diff <= 1e-7
                and representation_diff <= 1e-7
                and training_hash == deployed_hash
            ),
        }
    )
    (package_dir / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_safe(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
