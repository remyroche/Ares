#!/usr/bin/env python3
"""Compare a train-only PCA residual representation with the AE/GMM arm."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    OUTCOME_COLUMNS,
    REFERENCE_DERIVED_COLUMNS,
    ResidualArchetypeConfig,
    ResidualArchetypeRecognizer,
    _time_spread_indices,
    strip_outcomes_for_oos,
)
from scripts.run_meta_residual_ae_representation_ablation import (  # noqa: E402
    ARM,
    EVAL_MONTHS,
    _candidate_features,
    _fit_overlay,
)
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_OUT_DIR,
    KEY_COLUMNS,
    _calibrate,
    _fit_platt,
    _merge_residual_features,
    _selection_mask,
    metrics_by_scope,
    surprise_calendar,
)

BASE_TAG = "pca_baseline"
PCA_COLUMNS = tuple(f"meta_resid_pca_{idx:02d}" for idx in range(16))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.write_text(json.dumps(safe(payload), indent=2), encoding="utf-8")


def _fit_pca(
    frame: pd.DataFrame,
    columns: list[str],
    seed: int,
    requested_components: int = 16,
    scaled_clip: float | None = None,
) -> dict[str, Any]:
    positions = _time_spread_indices(len(frame), 5_000)
    sample = (
        frame.iloc[positions]
        .reindex(columns=columns)
        .apply(pd.to_numeric, errors="coerce")
    )
    values = sample.to_numpy(dtype=np.float32)
    medians = np.nanmedian(values, axis=0).astype(np.float32)
    medians = np.nan_to_num(medians, nan=0.0)
    values = np.where(np.isfinite(values), values, medians)
    low = np.percentile(values, 0.5, axis=0).astype(np.float32)
    high = np.percentile(values, 99.5, axis=0).astype(np.float32)
    values = np.clip(values, low, high)
    scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(values)
    scaled = scaler.transform(values).astype(np.float32)
    if scaled_clip is not None and float(scaled_clip) > 0.0:
        scaled = np.clip(scaled, -float(scaled_clip), float(scaled_clip)).astype(
            np.float32,
            copy=False,
        )
    components = min(
        max(int(requested_components), 1), scaled.shape[1], max(1, scaled.shape[0] - 1)
    )
    pca = PCA(n_components=components, random_state=seed).fit(scaled)
    return {
        "columns": columns,
        "medians": medians,
        "low": low,
        "high": high,
        "scaler": scaler,
        "pca": pca,
        "sample_rows": int(len(positions)),
        "scaled_clip": None if scaled_clip is None else float(scaled_clip),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(
            float
        ).tolist(),
        "effective_rank": float(
            np.exp(
                -np.sum(
                    (
                        pca.explained_variance_ratio_
                        / pca.explained_variance_ratio_.sum()
                    )
                    * np.log(
                        np.maximum(
                            pca.explained_variance_ratio_
                            / pca.explained_variance_ratio_.sum(),
                            1e-12,
                        )
                    )
                )
            )
        ),
    }


def _transform_pca(
    frame: pd.DataFrame, state: dict[str, Any], batch_rows: int = 100_000
) -> pd.DataFrame:
    output = np.empty((len(frame), len(state["pca"].components_)), dtype=np.float32)
    columns = state["columns"]
    medians = state["medians"]
    for start in range(0, len(frame), batch_rows):
        stop = min(start + batch_rows, len(frame))
        values = (
            frame.iloc[start:stop]
            .reindex(columns=columns)
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        values = np.where(np.isfinite(values), values, medians)
        values = np.clip(values, state["low"], state["high"])
        values = state["scaler"].transform(values).astype(np.float32)
        scaled_clip = state.get("scaled_clip")
        if scaled_clip is not None and float(scaled_clip) > 0.0:
            values = np.clip(values, -float(scaled_clip), float(scaled_clip)).astype(
                np.float32,
                copy=False,
            )
        output[start:stop] = state["pca"].transform(values).astype(np.float32)
    names = list(PCA_COLUMNS[: output.shape[1]])
    return pd.DataFrame(output, index=frame.index, columns=names, dtype=np.float32)


def _safe(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[
            name
            for name in OUTCOME_COLUMNS | REFERENCE_DERIVED_COLUMNS
            if name in frame.columns
        ],
        errors="ignore",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument(
        "--scaled-clip",
        type=float,
        default=0.0,
        help="Symmetric clip after robust scaling; zero preserves the legacy baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_offset = int(args.seed_offset)
    components = max(int(args.components), 1)
    scaled_clip = float(args.scaled_clip)
    clip_tag = "" if scaled_clip <= 0.0 else f"_clip{scaled_clip:g}"
    component_tag = (
        BASE_TAG
        if components == 16 and not clip_tag
        else f"pca{components}{clip_tag}_baseline"
    )
    tag = component_tag if seed_offset == 0 else f"{component_tag}_seed{seed_offset}"
    pca_arm = f"{ARM}_{tag}"
    root = DEFAULT_OUT_DIR
    arm_dir = root / pca_arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    cache = root / "cache" / f"residual_walkforward_ae_gmm_eval_mar_jun_{tag}.parquet"
    catalog_path = root / f"residual_walkforward_ae_gmm_eval_mar_jun_{tag}_catalog.csv"
    data = pd.read_parquet(root / "cache" / "compact_reference_with_lifecycle.parquet")
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    candidates = _candidate_features(data, root)
    pca_inputs = candidates[: min(80, len(candidates))]
    generated_frames: list[pd.DataFrame] = []
    catalogs: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    latest_pca = None
    latest_recognizer = None
    for fold_idx, month in enumerate(EVAL_MONTHS):
        start = pd.Timestamp(pd.Period(month).start_time, tz="UTC")
        end = pd.Timestamp((pd.Period(month) + 1).start_time, tz="UTC")
        train = data[data["__ts__"].lt(start)].copy()
        valid = data[data["__ts__"].ge(start) & data["__ts__"].lt(end)].copy()
        pca_state = _fit_pca(
            train,
            pca_inputs,
            20260711 + seed_offset + fold_idx * 101,
            requested_components=components,
            scaled_clip=scaled_clip,
        )
        train_pca = _transform_pca(train, pca_state)
        valid_pca = _transform_pca(valid, pca_state)
        for name in train_pca.columns:
            train[name] = train_pca[name].to_numpy(dtype=np.float32, copy=False)
            valid[name] = valid_pca[name].to_numpy(dtype=np.float32, copy=False)
        recognizer = ResidualArchetypeRecognizer(
            ResidualArchetypeConfig(
                use_residual_ae_gmm=False,
                random_state=20260711 + seed_offset + fold_idx * 101,
            ),
            list(dict.fromkeys([*candidates, *train_pca.columns.tolist()])),
        ).fit(train)
        generated = recognizer.transform_oos(strip_outcomes_for_oos(valid))
        keys = (
            valid[[name for name in KEY_COLUMNS if name in valid.columns]]
            .copy()
            .reset_index(drop=True)
        )
        keys["calendar_month"] = month
        generated_frames.append(
            pd.concat([keys, generated.reset_index(drop=True)], axis=1)
        )
        if not recognizer.catalog_.empty:
            catalog = recognizer.catalog_.copy()
            catalog["oos_month"] = month
            catalogs.append(catalog)
        folds.append(
            {
                "month": month,
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "pca_sample_rows": pca_state["sample_rows"],
                "pca_effective_rank": pca_state["effective_rank"],
                "pca_explained_variance_sum": float(
                    sum(pca_state["explained_variance_ratio"])
                ),
                **recognizer.manifest(),
            }
        )
        latest_pca = pca_state
        latest_recognizer = recognizer
        print(json.dumps({"event": "pca_fold_complete", "month": month}), flush=True)
        del train, valid, train_pca, valid_pca, generated, recognizer
        gc.collect()
    generated_all = pd.concat(generated_frames, ignore_index=True)
    generated_all.to_parquet(cache, index=False, compression="zstd")
    catalog_all = pd.concat(catalogs, ignore_index=True) if catalogs else pd.DataFrame()
    catalog_all.to_csv(catalog_path, index=False)
    joblib.dump(
        {"pca_state": latest_pca, "recognizer": latest_recognizer},
        root / "states" / f"residual_pca_eval_latest_{tag}.joblib",
        compress=3,
    )

    burnin = pd.read_parquet(
        root / "lifecycle_only_burnin" / "oos_predictions_march_burnin.parquet"
    )
    burnin = _merge_residual_features(burnin, generated_all)
    burnin["score_lifecycle_only"] = pd.to_numeric(
        burnin["score_alternative"], errors="coerce"
    ).astype(np.float32)
    oos = pd.read_parquet(
        root / "lifecycle_only" / "oos_predictions_apr_may_jun.parquet"
    )
    oos = _merge_residual_features(oos, generated_all)
    oos["score_lifecycle_only"] = pd.to_numeric(
        oos["score_alternative"], errors="coerce"
    ).astype(np.float32)
    overlay, search, selection = _fit_overlay(burnin)
    search.to_csv(arm_dir / "burnin_overlay_search.csv", index=False)
    _write_json(arm_dir / "overlay_selection.json", selection)
    joblib.dump(overlay, arm_dir / "residual_overlay_state.joblib")
    oos["score_alternative"] = overlay.transform(
        _safe(oos),
        oos["score_lifecycle_only"].fillna(0.5).to_numpy(dtype=np.float32),
    )
    burnin["score_pca_overlay"] = overlay.transform(
        _safe(burnin),
        burnin["score_lifecycle_only"].fillna(0.5).to_numpy(dtype=np.float32),
    )
    calibration_mask = _selection_mask(
        burnin,
        "score_pca_overlay",
        0.10,
        ["calendar_month", "side_name"],
    )
    calibrator = _fit_platt(
        burnin.loc[calibration_mask, "score_pca_overlay"],
        burnin.loc[calibration_mask, "clean_exec"],
    )
    joblib.dump(calibrator, arm_dir / "hit_calibrator.joblib")
    oos["hit_prob_alternative"] = _calibrate(calibrator, oos["score_alternative"])
    oos.to_parquet(arm_dir / "oos_predictions.parquet", index=False, compression="zstd")
    metrics = metrics_by_scope(oos, pca_arm)
    calendar, autocorr, comparison = surprise_calendar(oos, pca_arm)
    metrics.to_csv(arm_dir / "metrics_by_scope.csv", index=False)
    calendar.to_csv(arm_dir / "hit_surprise_calendar.csv", index=False)
    autocorr.to_csv(arm_dir / "hit_surprise_autocorrelation.csv", index=False)
    comparison.to_csv(arm_dir / "high_surprise_period_comparison.csv", index=False)
    top10 = metrics[
        metrics["scope"].eq("overall")
        & metrics["fraction"].eq(0.10)
        & metrics["selector"].eq(pca_arm)
    ].iloc[0]
    manifest = {
        "schema": "meta_residual_pca_representation_ablation_v1",
        "arm": pca_arm,
        "seed_offset": seed_offset,
        "requested_pca_components": components,
        "scaled_clip": scaled_clip,
        "folds": folds,
        "overlay": overlay.manifest(),
        "overlay_selection": selection,
        "top10_ev": float(top10["mean_ev_after_1pct"]),
        "top10_clean": float(top10["clean_exec_precision"]),
        "top10_full_bad_mae": float(top10["full_path_bad_mae_rate"]),
        "mean_abs_surprise_autocorr_lag1": float(
            autocorr[autocorr["selector"].eq(pca_arm)]["surprise_autocorr_lag1"]
            .abs()
            .mean()
        ),
        "current_model_overwritten": False,
    }
    _write_json(arm_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
