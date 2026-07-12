#!/usr/bin/env python3
"""Package the sparse market-shock residual-meta alternative for inference."""

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

from extreme_price_movements.meta_historical_rank import (
    HistoricalScoreRankReference,  # noqa: E402
)
from extreme_price_movements.meta_residual_archetypes import (
    strip_outcomes_for_oos,  # noqa: E402
)
from extreme_price_movements.meta_residual_shock_overlay import (
    ResidualShockOverlayState,  # noqa: E402
)
from scripts.run_meta_residual_sparse_shock_composite import (  # noqa: E402
    ARM,
    COMPONENTS,
    _fit_state,
    _rank,
    _reconstruct_march_champion,
)
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_OUT_DIR,
    _add_reference_fold_features,
    _fit_platt,
)


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
    source_bundle_dir = root / "inference_bundle_residual_pca8_globaloverlay"
    package_dir = root / "inference_bundle_residual_pca8_globaloverlay_shock"
    package_dir.mkdir(parents=True, exist_ok=True)
    source_bundle_path = (
        source_bundle_dir / "alternative_meta_residual_pca8_bundle.joblib"
    )
    bundle = joblib.load(source_bundle_path)
    shock_manifest = json.loads((root / ARM / "manifest.json").read_text())
    selected_parameters = dict(shock_manifest["selected_side_parameters"])

    columns = [
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "score_meta_base_soft_label",
        "clean_exec",
        "ev_after_1pct",
        *COMPONENTS,
    ]
    data = pd.read_parquet(
        root / "cache" / "compact_reference_with_lifecycle.parquet", columns=columns
    )
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    data = data.sort_values("__ts__", kind="stable").reset_index(drop=True)
    july_start = pd.Timestamp("2026-07-01", tz="UTC")
    train = data.loc[data["__ts__"].lt(july_start)].copy()
    train["training_rank_pct"] = _rank(train)
    raw_score = pd.to_numeric(
        train["score_meta_base_soft_label"], errors="coerce"
    ).fillna(0.5)
    clean = pd.to_numeric(train["clean_exec"], errors="coerce").fillna(0.0)
    train["risk_target"] = (raw_score - clean).clip(lower=0.0).astype(np.float32)
    fitted = _fit_state(train)
    shock_state = ResidualShockOverlayState(
        references={
            name: np.asarray(values, dtype=np.float32)
            for name, values in fitted.references.items()
        },
        archetype_multipliers=dict(fitted.archetype_multipliers),
        train_end=str(fitted.train_end),
    )
    state_path = package_dir / "shock_overlay_state.joblib"
    joblib.dump(shock_state, state_path, compress=3)

    output = pd.read_parquet(root / ARM / "oos_predictions_historical_rank.parquet")
    output["__ts__"] = pd.to_datetime(output["__ts__"], utc=True, errors="coerce")
    march = _reconstruct_march_champion(root)
    march["__ts__"] = pd.to_datetime(march["__ts__"], utc=True, errors="coerce")
    march_train = data.loc[
        data["__ts__"].ge(pd.Timestamp("2026-03-01", tz="UTC"))
        & data["__ts__"].lt(pd.Timestamp("2026-04-01", tz="UTC")),
        ["__ts__", "__symbol__", "side_name", "archetype_policy_key", *COMPONENTS],
    ]
    march_fit_train = train.loc[
        train["__ts__"].lt(pd.Timestamp("2026-03-01", tz="UTC"))
    ]
    march_fitted = _fit_state(march_fit_train)
    march_shock = march_train[
        ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
    ].copy()
    march_shock["shock_raw"] = march_fitted.transform_raw(march_train)
    march_shock["shock_local"] = march_fitted.transform(march_train)
    keys = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
    march = march.merge(march_shock, on=keys, how="left", validate="one_to_one")
    march_adjusted = march["score_champion"].to_numpy(dtype=np.float32, copy=True)
    for side, params in selected_parameters.items():
        mask = march["side_name"].eq(side).to_numpy()
        source = march.loc[
            mask, "shock_raw" if params["variant"] == "raw" else "shock_local"
        ]
        threshold = float(params["threshold"])
        intensity = np.clip(
            (source.to_numpy(dtype=np.float32) - threshold)
            / max(1.0 - threshold, 1e-3),
            0.0,
            1.0,
        )
        march_adjusted[mask] -= np.float32(params["alpha"]) * intensity.astype(
            np.float32
        )
    march_rank = pd.DataFrame(
        {
            "__ts__": march["__ts__"],
            "side_name": march["side_name"],
            "score_shock_adjusted": np.clip(march_adjusted, 0.0, 1.0),
        }
    )
    calibrator = _fit_platt(march_rank["score_shock_adjusted"], march["clean_exec"])
    oos_rank = output[["__ts__", "side_name", "score_adjusted"]].rename(
        columns={"score_adjusted": "score_shock_adjusted"}
    )
    rank_reference = HistoricalScoreRankReference(
        score_col="score_shock_adjusted", side_col="side_name"
    ).fit(pd.concat([march_rank, oos_rank], ignore_index=True))
    rank_path = package_dir / "historical_rank_reference.joblib"
    joblib.dump(rank_reference, rank_path, compress=3)

    bundle.shock_overlay_state = shock_state
    bundle.shock_side_parameters = selected_parameters
    bundle.hit_calibrator = calibrator
    bundle.historical_rank_reference = rank_reference
    bundle.metadata = {
        **dict(getattr(bundle, "metadata", {})),
        "role": "alternative_meta_residual_pca8_globaloverlay_sparse_shock",
        "shock_selection_months": shock_manifest["selection_months"],
        "shock_selection_objective": shock_manifest["selection_objective"],
        "current_meta_model_overwritten": False,
    }
    bundle_path = package_dir / "alternative_meta_residual_pca8_shock_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    restored = joblib.load(bundle_path)

    full = pd.read_parquet(root / "cache" / "compact_reference_with_lifecycle.parquet")
    full["__ts__"] = pd.to_datetime(full["__ts__"], utc=True, errors="coerce")
    _train_enriched, valid_enriched = _add_reference_fold_features(
        full[full["__ts__"].lt(july_start)].copy(),
        full[full["__ts__"].ge(july_start)].copy(),
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
    safe_valid = strip_outcomes_for_oos(valid_enriched)
    direct_adjusted, direct_raw, direct_local = shock_state.adjust_scores(
        safe_valid,
        predicted["score_residual_overlay"].to_numpy(dtype=np.float32),
        selected_parameters,
    )
    shock_diff = (
        float(
            np.nanmax(
                np.abs(
                    direct_adjusted
                    - predicted["score_shock_adjusted"].to_numpy(dtype=np.float32)
                )
            )
        )
        if len(predicted)
        else np.nan
    )
    raw_diff = (
        float(
            np.nanmax(
                np.abs(
                    direct_raw
                    - predicted["shock_composite_raw"].to_numpy(dtype=np.float32)
                )
            )
        )
        if len(predicted)
        else np.nan
    )
    local_diff = (
        float(
            np.nanmax(
                np.abs(
                    direct_local
                    - predicted["shock_composite_local"].to_numpy(dtype=np.float32)
                )
            )
        )
        if len(predicted)
        else np.nan
    )
    preview = valid_enriched[keys].copy()
    preview[predicted.columns] = predicted.to_numpy(dtype=np.float32)
    preview.to_parquet(
        package_dir / "july_oos_inference_preview.parquet",
        index=False,
        compression="zstd",
    )

    source_ae = source_bundle_dir / "base_ae_gmm_state.pkl"
    target_ae = package_dir / "base_ae_gmm_state.pkl"
    shutil.copy2(source_ae, target_ae)
    source_hash = _sha256(source_ae)
    target_hash = _sha256(target_ae)
    manifest = bundle.manifest()
    manifest.update(
        {
            "bundle_path": str(bundle_path),
            "shock_state_path": str(state_path),
            "historical_rank_reference_path": str(rank_path),
            "base_ae_gmm_path": str(target_ae),
            "base_ae_gmm_source_sha256": source_hash,
            "base_ae_gmm_packaged_sha256": target_hash,
            "base_ae_gmm_hash_match": source_hash == target_hash,
            "july_parity_rows": int(len(predicted)),
            "bundle_roundtrip_max_abs_diff": roundtrip_diff,
            "shock_adjustment_max_abs_diff": shock_diff,
            "shock_raw_max_abs_diff": raw_diff,
            "shock_local_max_abs_diff": local_diff,
            "inference_parity_pass": bool(
                len(predicted) > 0
                and roundtrip_diff <= 1e-7
                and shock_diff <= 1e-7
                and raw_diff <= 1e-7
                and local_diff <= 1e-7
                and source_hash == target_hash
            ),
        }
    )
    (package_dir / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2), encoding="utf-8"
    )
    print(json.dumps(_safe(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
