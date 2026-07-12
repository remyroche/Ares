#!/usr/bin/env python3
"""Build walk-forward surprise heads and test them as residual-meta support."""

from __future__ import annotations

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

from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    OUTCOME_COLUMNS,
    REFERENCE_DERIVED_COLUMNS,
    ResidualArchetypeConfig,
    strip_outcomes_for_oos,
)
from extreme_price_movements.meta_residual_overlay import (
    ResidualOverlayState,  # noqa: E402
)
from extreme_price_movements.meta_residual_surprise_heads import (  # noqa: E402
    ResidualSurpriseHeadState,
)
from scripts.run_meta_residual_ae_representation_ablation import (
    _candidate_features,  # noqa: E402
)
from scripts.run_meta_residual_overlay_ablation import _objective  # noqa: E402
from scripts.run_meta_residual_pca_representation_ablation import (  # noqa: E402
    _fit_pca,
    _transform_pca,
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

ARM = "lifecycle_residual_pca8_globaloverlay_surprise_head"
FORMAL_EVAL_MONTHS = ("2026-03", "2026-04", "2026-05", "2026-06")
SEMANTIC_CACHE = "residual_walkforward_ae_gmm_eval_mar_jun_pca8_clip8_baseline.parquet"
HEAD_CACHE = "residual_walkforward_surprise_head_pca8_clip8.parquet"
COMBINED_CACHE = "residual_walkforward_pca8_semantic_surprise_head.parquet"


def _burnin_autocorrelation(frame: pd.DataFrame, score_col: str) -> dict[str, float]:
    mask = _selection_mask(frame, score_col, 0.10, ["calendar_month", "side_name"])
    selected = frame.loc[mask].copy()
    calibrator = _fit_platt(selected[score_col], selected["clean_exec"])
    probability = _calibrate(calibrator, selected[score_col])
    signed = (
        pd.to_numeric(selected["clean_exec"], errors="coerce").to_numpy(
            dtype=np.float32
        )
        - probability
    )
    selected["date"] = pd.to_datetime(selected["__ts__"], utc=True).dt.floor("D")
    selected["signed_surprise"] = signed
    selected["negative_surprise"] = np.maximum(-signed, 0.0)
    selected["positive_surprise"] = np.maximum(signed, 0.0)
    values = {name: [] for name in ("signed", "negative", "positive")}
    for _, group in selected.groupby(["side_name", "archetype_policy_key"], sort=False):
        daily = group.groupby("date", sort=True)[
            ["signed_surprise", "negative_surprise", "positive_surprise"]
        ].mean()
        for name, column in (
            ("signed", "signed_surprise"),
            ("negative", "negative_surprise"),
            ("positive", "positive_surprise"),
        ):
            value = float(daily[column].autocorr(1)) if len(daily) >= 3 else np.nan
            if np.isfinite(value):
                values[name].append(abs(value))
    return {
        f"burnin_{name}_surprise_abs_ac_lag1": (
            float(np.mean(series)) if series else np.nan
        )
        for name, series in values.items()
    }


def _safe(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[
            name
            for name in OUTCOME_COLUMNS | REFERENCE_DERIVED_COLUMNS
            if name in frame.columns
        ],
        errors="ignore",
    )


def _walkforward_heads(
    data: pd.DataFrame, root: Path
) -> tuple[pd.DataFrame, Any, list[dict[str, Any]]]:
    cache = root / "cache" / HEAD_CACHE
    state_path = root / "states" / "residual_surprise_head_pca8_latest.joblib"
    manifest_path = cache.with_suffix(".manifest.json")
    if cache.exists() and state_path.exists() and manifest_path.exists():
        cached_manifest = json.loads(manifest_path.read_text())
        cached_months = [
            str(row.get("month")) for row in cached_manifest.get("folds", [])
        ]
        if cached_months and min(cached_months) <= "2025-06":
            return (
                pd.read_parquet(cache),
                joblib.load(state_path),
                cached_manifest["folds"],
            )
    candidates = _candidate_features(data, root)
    pca_inputs = candidates[: min(80, len(candidates))]
    frames: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    latest = None
    score_rows = data[
        pd.to_numeric(data["score_meta_base_soft_label"], errors="coerce").notna()
    ]
    available_months = sorted(
        score_rows["__ts__"].dt.to_period("M").astype(str).unique().tolist()
    )
    eval_months = available_months[2:]
    for fold_idx, month in enumerate(eval_months):
        start = pd.Timestamp(pd.Period(month).start_time, tz="UTC")
        end = pd.Timestamp((pd.Period(month) + 1).start_time, tz="UTC")
        train = data[data["__ts__"].lt(start)].copy()
        valid = data[data["__ts__"].ge(start) & data["__ts__"].lt(end)].copy()
        if len(train) < 5_000 or len(valid) < 100:
            continue
        pca = _fit_pca(
            train,
            pca_inputs,
            seed=20260711 + fold_idx * 101,
            requested_components=8,
            scaled_clip=8.0,
        )
        train_pca = _transform_pca(train, pca)
        valid_pca = _transform_pca(valid, pca)
        for name in train_pca.columns:
            train[name] = train_pca[name].to_numpy(dtype=np.float32, copy=False)
            valid[name] = valid_pca[name].to_numpy(dtype=np.float32, copy=False)
        head = ResidualSurpriseHeadState(
            candidate_features=list(
                dict.fromkeys([*candidates, *train_pca.columns.astype(str).tolist()])
            ),
            config=ResidualArchetypeConfig(random_state=20260711 + fold_idx * 101),
        ).fit(train)
        generated = head.transform(strip_outcomes_for_oos(valid))
        keys = (
            valid[[name for name in KEY_COLUMNS if name in valid.columns]]
            .copy()
            .reset_index(drop=True)
        )
        keys["calendar_month"] = month
        frames.append(pd.concat([keys, generated.reset_index(drop=True)], axis=1))
        folds.append(
            {
                "month": month,
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "pca_effective_rank": float(pca["effective_rank"]),
                "surprise_head": head.manifest(),
            }
        )
        latest = {"head": head, "pca": pca}
        print(
            json.dumps({"event": "surprise_head_fold_complete", "month": month}),
            flush=True,
        )
        del train, valid, train_pca, valid_pca, generated, head
        gc.collect()
    output = pd.concat(frames, ignore_index=True)
    output.to_parquet(cache, index=False, compression="zstd")
    joblib.dump(latest, state_path, compress=3)
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "meta_residual_surprise_head_walkforward_v1",
                "folds": folds,
                "leakage_contract": "Every PCA and surprise head fits only rows before its OOS month.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output, latest, folds


def main() -> None:
    root = DEFAULT_OUT_DIR
    arm_dir = root / ARM
    arm_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(root / "cache" / "compact_reference_with_lifecycle.parquet")
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    semantic = pd.read_parquet(root / "cache" / SEMANTIC_CACHE)
    head_features, latest, folds = _walkforward_heads(data, root)
    generated = _merge_residual_features(semantic, head_features)
    generated.to_parquet(
        root / "cache" / COMBINED_CACHE,
        index=False,
        compression="zstd",
    )
    burnin = pd.read_parquet(
        root / "lifecycle_only_burnin" / "oos_predictions_march_burnin.parquet"
    )
    burnin = _merge_residual_features(burnin, generated)
    burnin["calendar_month"] = "2026-03"
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
    feature_pairs = {
        "semantic_hit_x_negative_head": (
            "meta_resid_arch_expected_hit_surprise",
            "meta_resid_negative_tail_probability",
        ),
        "signed_head_x_negative_head": (
            "meta_resid_signed_surprise_prediction",
            "meta_resid_negative_tail_probability",
        ),
        "net_head_x_negative_head": (
            "meta_resid_surprise_head_net_probability",
            "meta_resid_negative_tail_probability",
        ),
    }
    rows: list[dict[str, Any]] = []
    states: dict[tuple[str, float, float], ResidualOverlayState] = {}
    for pair_name, (hit_feature, dirty_feature) in feature_pairs.items():
        normalizer = ResidualOverlayState(
            hit_feature=hit_feature,
            dirty_feature=dirty_feature,
        ).fit_normalization(burnin)
        for hit_alpha in (0.0, 0.025, 0.05, 0.10, 0.20, 0.30):
            for dirty_lambda in (0.0, 0.025, 0.05, 0.10, 0.20, 0.30):
                state = ResidualOverlayState(
                    hit_alpha=hit_alpha,
                    dirty_lambda=dirty_lambda,
                    local_hit_alpha=0.0,
                    local_dirty_lambda=0.0,
                    hit_feature=hit_feature,
                    dirty_feature=dirty_feature,
                    group_stats=dict(normalizer.group_stats),
                    side_stats=dict(normalizer.side_stats),
                    global_stats=normalizer.global_stats,
                    calibration_start=normalizer.calibration_start,
                    calibration_end=normalizer.calibration_end,
                )
                score_col = "__score_surprise_head"
                burnin[score_col] = state.transform(
                    _safe(burnin),
                    burnin["score_lifecycle_only"]
                    .fillna(0.5)
                    .to_numpy(dtype=np.float32),
                )
                metrics = _objective(burnin, score_col)
                autocorr = _burnin_autocorrelation(burnin, score_col)
                calendar_penalty = (
                    0.15 * autocorr["burnin_signed_surprise_abs_ac_lag1"]
                    + 0.10 * autocorr["burnin_negative_surprise_abs_ac_lag1"]
                    + 0.05 * autocorr["burnin_positive_surprise_abs_ac_lag1"]
                )
                rows.append(
                    {
                        "pair": pair_name,
                        "hit_feature": hit_feature,
                        "dirty_feature": dirty_feature,
                        "hit_alpha": hit_alpha,
                        "dirty_lambda": dirty_lambda,
                        **metrics,
                        **autocorr,
                        "calendar_penalty": calendar_penalty,
                        "objective_with_calendar": float(
                            metrics["objective"] - calendar_penalty
                        ),
                    }
                )
                states[(pair_name, hit_alpha, dirty_lambda)] = state
    search = pd.DataFrame(rows).sort_values(
        "objective_with_calendar", ascending=False, kind="stable"
    )
    best_objective = float(search["objective_with_calendar"].max())
    eligible = search[
        search["objective_with_calendar"].ge(best_objective - 0.002)
    ].copy()
    eligible["complexity"] = (
        eligible["hit_alpha"].abs() + eligible["dirty_lambda"].abs()
    )
    best = eligible.sort_values(
        ["complexity", "objective_with_calendar"], ascending=[True, False]
    ).iloc[0]
    state = states[
        (str(best["pair"]), float(best["hit_alpha"]), float(best["dirty_lambda"]))
    ]
    search.to_csv(arm_dir / "burnin_overlay_search.csv", index=False)
    joblib.dump(state, arm_dir / "residual_overlay_state.joblib", compress=3)
    oos["score_alternative"] = state.transform(
        _safe(oos),
        oos["score_lifecycle_only"].fillna(0.5).to_numpy(dtype=np.float32),
    )
    burnin["score_selected"] = state.transform(
        _safe(burnin),
        burnin["score_lifecycle_only"].fillna(0.5).to_numpy(dtype=np.float32),
    )
    calibration_mask = _selection_mask(
        burnin,
        "score_selected",
        0.10,
        ["calendar_month", "side_name"],
    )
    calibrator = _fit_platt(
        burnin.loc[calibration_mask, "score_selected"],
        burnin.loc[calibration_mask, "clean_exec"],
    )
    joblib.dump(calibrator, arm_dir / "hit_calibrator.joblib", compress=3)
    oos["hit_prob_alternative"] = _calibrate(calibrator, oos["score_alternative"])
    oos.to_parquet(arm_dir / "oos_predictions.parquet", index=False, compression="zstd")
    metrics = metrics_by_scope(oos, ARM)
    calendar, autocorr, comparison = surprise_calendar(oos, ARM)
    metrics.to_csv(arm_dir / "metrics_by_scope.csv", index=False)
    calendar.to_csv(arm_dir / "hit_surprise_calendar.csv", index=False)
    autocorr.to_csv(arm_dir / "hit_surprise_autocorrelation.csv", index=False)
    comparison.to_csv(arm_dir / "high_surprise_period_comparison.csv", index=False)
    manifest = {
        "schema": "meta_residual_surprise_head_ablation_v1",
        "arm": ARM,
        "walkforward_folds": folds,
        "selected_pair": str(best["pair"]),
        "selected_hit_feature": state.hit_feature,
        "selected_dirty_feature": state.dirty_feature,
        "selected_hit_alpha": float(state.hit_alpha),
        "selected_dirty_lambda": float(state.dirty_lambda),
        "burnin_objective_best": best_objective,
        "burnin_objective_selected": float(best["objective_with_calendar"]),
        "burnin_economic_objective_selected": float(best["objective"]),
        "burnin_calendar_penalty_selected": float(best["calendar_penalty"]),
        "latest_surprise_head_materialized": latest is not None,
        "current_meta_model_overwritten": False,
        "leakage_contract": (
            "Surprise heads and PCA are monthly walk-forward; coefficients are selected on March "
            "burn-in only; April-June remain untouched."
        ),
    }
    (arm_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
