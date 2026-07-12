#!/usr/bin/env python3
"""Fit the alternative meta model through June and package frozen inference state."""

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

from extreme_price_movements.alternative_meta_residual_bundle import (
    AlternativeMetaResidualBundle,  # noqa: E402
)
from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    residual_feature_names,
    strip_outcomes_for_oos,
)
from extreme_price_movements.meta_residual_overlay import (
    ResidualOverlayState,  # noqa: E402
)
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_OUT_DIR,
    _add_reference_fold_features,
    train_arm_oos,
)

DEPLOYED_AE_GMM = Path(
    "data_perp/artifacts/s59_s52_frozen_native_shadow_20260709/ae_gmm_state/ae_gmm_state.pkl"
)
TRAINING_AE_GMM = Path(
    "data_perp/reports/s59_h5_2025start_monthly_v6_15mchart_base_frozenfs_fixedparams_"
    "uncapped_trainthroughjun_scorejul_20260708/ae_gmm_states/"
    "2026-07-01_2026-07-16__global_state.pkl"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    root = DEFAULT_OUT_DIR
    package_dir = root / "inference_bundle"
    package_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(root / "cache" / "compact_reference_with_lifecycle.parquet")
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    residual = pd.read_parquet(root / "cache" / "residual_walkforward_raw.parquet")
    selected_rows = pd.read_csv(root / "lifecycle_only" / "selected_features.csv")
    selected_mask = selected_rows["selected"].fillna(False).astype(bool)
    selected = selected_rows.loc[selected_mask, "feature"].astype(str).tolist()
    dataset_manifest = json.loads((root / "dataset_manifest.json").read_text())
    params = dict(dataset_manifest["reference_model_params"])

    july_predictions, _ = train_arm_oos(
        arm="lifecycle_only_final_through_june",
        data=data,
        selected_features=selected,
        params=params,
        output_dir=root,
        seed=20260711,
        eval_months=("2026-07",),
        artifact_tag="july_oos_parity",
    )
    model_path = (
        root
        / "lifecycle_only_final_through_june"
        / "latest_oos_fold_model_july_oos_parity.joblib"
    )
    model_artifact = joblib.load(model_path)
    recognizer_path = (
        root / "states" / "residual_walkforward_raw_latest_recognizer.joblib"
    )
    overlay_path = (
        root / "lifecycle_residual_local_overlay" / "residual_overlay_state.joblib"
    )
    recognizer = joblib.load(recognizer_path)
    overlay: ResidualOverlayState = joblib.load(overlay_path)

    if not DEPLOYED_AE_GMM.exists() or not TRAINING_AE_GMM.exists():
        raise FileNotFoundError("Required frozen AE/GMM parity artifacts are missing")
    deployed_hash = _sha256(DEPLOYED_AE_GMM)
    training_hash = _sha256(TRAINING_AE_GMM)
    if deployed_hash != training_hash:
        raise RuntimeError(
            "Deployed AE/GMM state differs from the state used for July scoring"
        )
    packaged_ae_path = package_dir / "ae_gmm_state.pkl"
    shutil.copy2(DEPLOYED_AE_GMM, packaged_ae_path)

    bundle = AlternativeMetaResidualBundle(
        lifecycle_model=model_artifact["model"],
        selected_features=list(model_artifact["selected_features"]),
        raw_selected_features=list(model_artifact["raw_selected_features"]),
        feature_medians=dict(model_artifact["feature_medians"]),
        ood_state=dict(model_artifact["ood_state"]),
        residual_recognizer=recognizer,
        overlay_state=overlay,
        hit_calibrator=model_artifact.get("hit_calibrator"),
        fit_through=str(model_artifact.get("fit_through")),
        frozen_ae_gmm_sha256=deployed_hash,
        metadata={
            "role": "alternative_meta_model_not_current_production",
            "feature_selection": "frozen lifecycle_only 58-feature contract selected before April",
            "model_params": params,
            "base_model_retrained": False,
            "current_meta_model_overwritten": False,
        },
    )
    bundle_path = package_dir / "alternative_meta_residual_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)

    july_start = pd.Timestamp("2026-07-01", tz="UTC")
    train = data[data["__ts__"].lt(july_start)].copy()
    valid = data[data["__ts__"].ge(july_start)].copy()
    train, valid = _add_reference_fold_features(train, valid)
    scored = bundle.predict(valid)
    reference = july_predictions.set_index(
        ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
    )
    valid_keys = valid[
        ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
    ].copy()
    valid_keys.index = scored.index
    parity = valid_keys.join(scored).set_index(
        ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
    )
    aligned = parity.join(
        reference[["score_alternative"]].rename(
            columns={"score_alternative": "score_reference"}
        ),
        how="inner",
    )
    lifecycle_diff = np.abs(
        aligned["score_lifecycle_only"] - aligned["score_reference"]
    )

    cached_residual = residual[
        residual["calendar_month"].astype(str).eq("2026-07")
    ].copy()
    cached_residual = cached_residual.drop_duplicates(
        ["__ts__", "__symbol__", "side_name", "archetype_policy_key"], keep="last"
    )
    generated_residual = recognizer.transform_oos(strip_outcomes_for_oos(valid))
    generated = valid_keys.copy()
    generated[residual_feature_names(include_ae_gmm=False)] = (
        generated_residual.reindex(
            columns=residual_feature_names(include_ae_gmm=False), fill_value=0.0
        ).to_numpy(dtype=np.float32)
    )
    cached_cols = [
        name
        for name in residual_feature_names(include_ae_gmm=False)
        if name in cached_residual.columns
    ]
    residual_cmp = generated.merge(
        cached_residual[
            ["__ts__", "__symbol__", "side_name", "archetype_policy_key"] + cached_cols
        ],
        on=["__ts__", "__symbol__", "side_name", "archetype_policy_key"],
        how="inner",
        suffixes=("_generated", "_cached"),
        validate="one_to_one",
    )
    residual_max_diff = 0.0
    for name in cached_cols:
        diff = np.abs(
            pd.to_numeric(residual_cmp[f"{name}_generated"], errors="coerce").to_numpy(
                dtype=np.float32
            )
            - pd.to_numeric(residual_cmp[f"{name}_cached"], errors="coerce").to_numpy(
                dtype=np.float32
            )
        )
        if len(diff):
            residual_max_diff = max(residual_max_diff, float(np.nanmax(diff)))

    preview = valid_keys.join(scored)
    preview.to_parquet(
        package_dir / "july_oos_inference_preview.parquet",
        index=False,
        compression="zstd",
    )
    manifest = bundle.manifest()
    manifest.update(
        {
            "bundle_path": str(bundle_path),
            "packaged_ae_gmm_path": str(packaged_ae_path),
            "training_ae_gmm_path": str(TRAINING_AE_GMM),
            "deployed_ae_gmm_path": str(DEPLOYED_AE_GMM),
            "training_ae_gmm_sha256": training_hash,
            "deployed_ae_gmm_sha256": deployed_hash,
            "ae_gmm_hash_match": training_hash == deployed_hash,
            "july_parity_rows": int(len(aligned)),
            "july_lifecycle_score_max_abs_diff": float(lifecycle_diff.max())
            if len(lifecycle_diff)
            else None,
            "july_residual_feature_rows": int(len(residual_cmp)),
            "july_residual_feature_max_abs_diff": residual_max_diff,
            "inference_parity_pass": bool(
                len(aligned) > 0
                and float(lifecycle_diff.max()) <= 1e-7
                and len(residual_cmp) > 0
                and residual_max_diff <= 1e-7
                and training_hash == deployed_hash
            ),
        }
    )
    (package_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
