#!/usr/bin/env python3
"""Test residual-aware AE/GMM context against the raw residual recognizer."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    OUTCOME_COLUMNS,
    REFERENCE_DERIVED_COLUMNS,
    ResidualArchetypeConfig,
    ResidualArchetypeRecognizer,
    strip_outcomes_for_oos,
)
from extreme_price_movements.meta_residual_overlay import (
    ResidualOverlayState,  # noqa: E402
)
from scripts.run_meta_residual_overlay_ablation import _objective  # noqa: E402
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    AE_GMM_HINTS,
    DEFAULT_OUT_DIR,
    KEY_COLUMNS,
    _calibrate,
    _fit_platt,
    _merge_residual_features,
    _selection_mask,
    metrics_by_scope,
    surprise_calendar,
)

ARM = "lifecycle_residual_aware_ae_gmm_overlay"
EVAL_MONTHS = ("2026-03", "2026-04", "2026-05", "2026-06")
DEFAULT_SIMPLIFICATION_TOLERANCE = 0.002


def _safe(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[
            name
            for name in OUTCOME_COLUMNS | REFERENCE_DERIVED_COLUMNS
            if name in frame.columns
        ],
        errors="ignore",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def _candidate_features(data: pd.DataFrame, root: Path) -> list[str]:
    dataset_manifest = json.loads((root / "dataset_manifest.json").read_text())
    reference = [
        str(name)
        for name in dataset_manifest["reference_selected_features"]
        if str(name) in data.columns
    ]
    lifecycle = [
        str(name)
        for name in CFG.get("CRASH_LIFECYCLE_NEW_FEATURE_KEYS", [])
        if str(name) in data.columns
    ]
    existing = [
        str(name)
        for name in data.columns
        if any(token.lower() in str(name).lower() for token in AE_GMM_HINTS)
        and name not in OUTCOME_COLUMNS
    ]
    broad_numeric = [
        str(name)
        for name in data.columns
        if name not in OUTCOME_COLUMNS | REFERENCE_DERIVED_COLUMNS
        and pd.api.types.is_numeric_dtype(data[name])
    ]
    return list(dict.fromkeys(reference + lifecycle + existing + broad_numeric))


def _build_features(
    data: pd.DataFrame,
    root: Path,
    *,
    seed_offset: int = 0,
    tag: str = "",
    force: bool = False,
    fit_local_models: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], Any]:
    suffix = f"_{tag}" if tag else ""
    cache = root / "cache" / f"residual_walkforward_ae_gmm_eval_mar_jun{suffix}.parquet"
    catalog_path = (
        root / f"residual_walkforward_ae_gmm_eval_mar_jun{suffix}_catalog.csv"
    )
    manifest_path = (
        root
        / "cache"
        / f"residual_walkforward_ae_gmm_eval_mar_jun{suffix}.manifest.json"
    )
    if cache.exists() and manifest_path.exists() and not force:
        catalog = pd.read_csv(catalog_path) if catalog_path.exists() else pd.DataFrame()
        return (
            pd.read_parquet(cache),
            catalog,
            json.loads(manifest_path.read_text())["folds"],
            None,
        )
    candidates = _candidate_features(data, root)
    frames: list[pd.DataFrame] = []
    catalogs: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    latest = None
    for fold_idx, month in enumerate(EVAL_MONTHS):
        start = pd.Timestamp(pd.Period(month).start_time, tz="UTC")
        end = pd.Timestamp((pd.Period(month) + 1).start_time, tz="UTC")
        train = data[data["__ts__"].lt(start)]
        valid = data[data["__ts__"].ge(start) & data["__ts__"].lt(end)]
        cfg = ResidualArchetypeConfig(
            score_col=(
                "score_regime_calibrated"
                if "score_regime_calibrated" in train.columns
                else "score_meta_base_soft_label"
            ),
            rank_scope="global",
            use_residual_ae_gmm=True,
            ae_gmm_max_rows=5_000,
            ae_gmm_max_iter=80,
            fit_local_models=fit_local_models,
            allow_side_fallback=False,
            random_state=20260711 + int(seed_offset) + fold_idx * 101,
        )
        recognizer = ResidualArchetypeRecognizer(cfg, candidates).fit(train)
        generated = recognizer.transform_oos(strip_outcomes_for_oos(valid))
        keys = (
            valid[[name for name in KEY_COLUMNS if name in valid.columns]]
            .copy()
            .reset_index(drop=True)
        )
        keys["calendar_month"] = month
        frames.append(pd.concat([keys, generated.reset_index(drop=True)], axis=1))
        if not recognizer.catalog_.empty:
            catalog = recognizer.catalog_.copy()
            catalog["oos_month"] = month
            catalog["fit_through"] = str(start - pd.Timedelta(nanoseconds=1))
            catalogs.append(catalog)
        fold = {
            "month": month,
            "train_rows": len(train),
            "valid_rows": len(valid),
            **recognizer.manifest(),
        }
        folds.append(fold)
        latest = recognizer
        print(
            json.dumps(
                {
                    "event": "residual_ae_fold_complete",
                    "month": month,
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                }
            ),
            flush=True,
        )
        del train, valid, generated, recognizer
        gc.collect()
    output = pd.concat(frames, ignore_index=True)
    catalog_all = pd.concat(catalogs, ignore_index=True) if catalogs else pd.DataFrame()
    output.to_parquet(cache, index=False, compression="zstd")
    catalog_all.to_csv(catalog_path, index=False)
    _write_json(
        manifest_path,
        {
            "schema": "residual_aware_ae_gmm_walkforward_eval_v1",
            "folds": folds,
            "candidate_features": candidates,
            "leakage_contract": "Each AE/GMM/recognizer is fitted only on rows before its OOS month.",
        },
    )
    if latest is not None:
        joblib.dump(
            latest,
            root / "states" / f"residual_ae_gmm_eval_latest_recognizer{suffix}.joblib",
        )
    return output, catalog_all, folds, latest


def _fit_overlay(
    burnin: pd.DataFrame,
    *,
    simplification_tolerance: float = DEFAULT_SIMPLIFICATION_TOLERANCE,
) -> tuple[ResidualOverlayState, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    normalizer = ResidualOverlayState().fit_normalization(burnin)
    for hit_alpha in (0.0, 0.05, 0.10, 0.20, 0.30):
        for dirty_lambda in (0.0, 0.025, 0.05, 0.10):
            for local_hit_alpha in (0.0, 0.005, 0.01, 0.02):
                state = ResidualOverlayState(
                    hit_alpha=hit_alpha,
                    dirty_lambda=dirty_lambda,
                    local_hit_alpha=local_hit_alpha,
                    group_stats=dict(normalizer.group_stats),
                    side_stats=dict(normalizer.side_stats),
                    global_stats=normalizer.global_stats,
                    calibration_start=normalizer.calibration_start,
                    calibration_end=normalizer.calibration_end,
                )
                burnin["score_ae_overlay"] = state.transform(
                    _safe(burnin),
                    pd.to_numeric(burnin["score_lifecycle_only"], errors="coerce")
                    .fillna(0.5)
                    .to_numpy(dtype=np.float32),
                )
                rows.append(
                    {
                        "hit_alpha": hit_alpha,
                        "dirty_lambda": dirty_lambda,
                        "local_hit_alpha": local_hit_alpha,
                        **_objective(burnin, "score_ae_overlay"),
                    }
                )
    search = pd.DataFrame(rows).sort_values("objective", ascending=False, kind="stable")
    best_objective = float(search["objective"].max())
    tolerance = max(0.0, float(simplification_tolerance))
    eligible = search[search["objective"].ge(best_objective - tolerance)].copy()
    eligible["complexity"] = (
        eligible["hit_alpha"].abs()
        + eligible["dirty_lambda"].abs()
        + eligible["local_hit_alpha"].abs()
    )
    best = eligible.sort_values(
        ["complexity", "objective"], ascending=[True, False], kind="stable"
    ).iloc[0]
    state = ResidualOverlayState(
        hit_alpha=float(best["hit_alpha"]),
        dirty_lambda=float(best["dirty_lambda"]),
        local_hit_alpha=float(best["local_hit_alpha"]),
        group_stats=dict(normalizer.group_stats),
        side_stats=dict(normalizer.side_stats),
        global_stats=normalizer.global_stats,
        calibration_start=normalizer.calibration_start,
        calibration_end=normalizer.calibration_end,
    )
    selection = {
        "rule": "minimum_l1_overlay_complexity_within_absolute_burnin_objective_tolerance",
        "objective_best": best_objective,
        "absolute_tolerance": tolerance,
        "objective_selected": float(best["objective"]),
        "eligible_rows": int(len(eligible)),
        "selected_hit_alpha": float(best["hit_alpha"]),
        "selected_dirty_lambda": float(best["dirty_lambda"]),
        "selected_local_hit_alpha": float(best["local_hit_alpha"]),
        "rationale": (
            "Prefer the least complex burn-in-competitive overlay. Local side-by-archetype "
            "normalization is retained only when its incremental burn-in objective exceeds "
            "the declared tolerance."
        ),
    }
    return state, search, selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--disable-local-models",
        action="store_true",
        help="Fit only side-level residual recognizers; retain the same overlay search.",
    )
    parser.add_argument(
        "--simplification-tolerance",
        type=float,
        default=DEFAULT_SIMPLIFICATION_TOLERANCE,
        help="Absolute burn-in objective tolerance used to choose the least complex overlay.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = DEFAULT_OUT_DIR
    arm = ARM if not args.tag else f"{ARM}_{args.tag}"
    arm_dir = root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(root / "cache" / "compact_reference_with_lifecycle.parquet")
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    generated, catalog, folds, latest = _build_features(
        data,
        root,
        seed_offset=int(args.seed_offset),
        tag=str(args.tag),
        force=bool(args.force),
        fit_local_models=not bool(args.disable_local_models),
    )
    burnin = pd.read_parquet(
        root / "lifecycle_only_burnin" / "oos_predictions_march_burnin.parquet"
    )
    burnin = _merge_residual_features(burnin, generated)
    burnin["score_lifecycle_only"] = pd.to_numeric(
        burnin["score_alternative"], errors="coerce"
    ).astype(np.float32)
    oos = pd.read_parquet(
        root / "lifecycle_only" / "oos_predictions_apr_may_jun.parquet"
    )
    oos = _merge_residual_features(oos, generated)
    oos["score_lifecycle_only"] = pd.to_numeric(
        oos["score_alternative"], errors="coerce"
    ).astype(np.float32)
    state, search, selection = _fit_overlay(
        burnin,
        simplification_tolerance=float(args.simplification_tolerance),
    )
    search.to_csv(arm_dir / "burnin_overlay_search.csv", index=False)
    _write_json(arm_dir / "overlay_selection.json", selection)
    joblib.dump(state, arm_dir / "residual_overlay_state.joblib")
    oos["score_alternative"] = state.transform(
        _safe(oos),
        oos["score_lifecycle_only"].fillna(0.5).to_numpy(dtype=np.float32),
    )
    burnin["score_ae_overlay"] = state.transform(
        _safe(burnin),
        burnin["score_lifecycle_only"].fillna(0.5).to_numpy(dtype=np.float32),
    )
    calibration_mask = _selection_mask(
        burnin,
        "score_ae_overlay",
        0.10,
        ["calendar_month", "side_name"],
    )
    calibrator = _fit_platt(
        burnin.loc[calibration_mask, "score_ae_overlay"],
        burnin.loc[calibration_mask, "clean_exec"],
    )
    joblib.dump(calibrator, arm_dir / "hit_calibrator.joblib")
    oos["hit_prob_alternative"] = _calibrate(calibrator, oos["score_alternative"])
    oos.to_parquet(arm_dir / "oos_predictions.parquet", index=False, compression="zstd")
    metrics = metrics_by_scope(oos, arm)
    calendar, autocorr, comparison = surprise_calendar(oos, arm)
    metrics.to_csv(arm_dir / "metrics_by_scope.csv", index=False)
    calendar.to_csv(arm_dir / "hit_surprise_calendar.csv", index=False)
    autocorr.to_csv(arm_dir / "hit_surprise_autocorrelation.csv", index=False)
    comparison.to_csv(arm_dir / "high_surprise_period_comparison.csv", index=False)
    summary = metrics[
        metrics["scope"].eq("overall")
        & metrics["fraction"].eq(0.10)
        & metrics["selector"].eq(arm)
    ].iloc[0]
    ac = autocorr[autocorr["selector"].eq(arm)]["surprise_autocorr_lag1"].abs().mean()
    manifest = {
        "schema": "meta_residual_aware_ae_gmm_ablation_v1",
        "arm": arm,
        "seed_offset": int(args.seed_offset),
        "folds": folds,
        "overlay": state.manifest(),
        "overlay_selection": selection,
        "hit_calibration": {
            "method": "platt_logistic",
            "population": "top10_by_calendar_month_and_side",
            "fit_period": "2026-03_burnin",
            "rows": int(calibration_mask.sum()),
            "changes_ranking": False,
        },
        "top10_ev": float(summary["mean_ev_after_1pct"]),
        "top10_clean": float(summary["clean_exec_precision"]),
        "top10_full_bad_mae": float(summary["full_path_bad_mae_rate"]),
        "mean_abs_surprise_autocorr_lag1": float(ac),
        "latest_recognizer_materialized": latest is not None,
        "fit_local_models": not bool(args.disable_local_models),
        "catalog_rows": int(len(catalog)),
        "current_model_overwritten": False,
    }
    _write_json(arm_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
