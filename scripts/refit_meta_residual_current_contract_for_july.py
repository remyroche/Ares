#!/usr/bin/env python3
"""Refit the frozen residual-meta architecture on the current handoff contract.

Feature selection, LightGBM parameters, model structure, and policy parameters
come from the promoted residual bundle. Only model states are refit on rows
strictly before July so the extended July pool can be scored OOS without mixing
in the older handoff's incompatible transformed feature scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.alternative_meta_residual_bundle import (
    AlternativeMetaResidualBundle,
)
from extreme_price_movements.meta_historical_rank import HistoricalScoreRankReference
from extreme_price_movements.meta_residual_archetypes import (
    ResidualArchetypeConfig,
    ResidualArchetypeRecognizer,
    strip_outcomes_for_oos,
)
from extreme_price_movements.meta_residual_overlay import ResidualOverlayState
from extreme_price_movements.meta_residual_shock_overlay import (
    ResidualShockOverlayState,
)
from scripts.run_meta_residual_pca_representation_ablation import (
    _fit_pca,
    _transform_pca,
)
from scripts.run_meta_residual_sparse_shock_composite import (
    COMPONENTS,
    _fit_state,
    _rank,
)
from scripts.run_s52_train_meta_regime_handoff_smoke import (
    _base_soft_label_target,
    _fit_base_soft_label_model,
    _predict,
)
from scripts.run_train_meta_residual_archetype_enhancement import (
    _apply_ood_state,
    _fit_ood_state,
    _fit_platt,
    _matrix_fit_transform,
)
from scripts.score_compare_meta_residual_july_oos import (
    DEFAULT_AEGMM,
    DEFAULT_BUNDLE,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_HANDOFF,
    DEFAULT_OLD_PREDICTIONS,
    _append_store_features,
    _merge_frozen_aegmm,
    _merge_old_predictions,
    _read_july_handoff,
)

DEFAULT_OUTPUT = Path(
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "july_current_contract_refit"
)
DEFAULT_OLD_TRAINING_PREDICTIONS = Path(
    "data_perp/reports/s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_"
    "largestfold_oos15_ae3000_nocrossfit_k34567_payload300k_20260706/"
    "train_meta_regime_handoff_singlehead_base_soft_lgbmpipeline_auto_hpo150_oos15_"
    "top30_hpo45k_20260706_v5/best_full_oos_fixedfs_streamed_v1/"
    "s52_train_meta_regime_handoff_smoke_predictions.parquet"
)
DEFAULT_TRAINING_LABEL_HANDOFF = Path(
    "data_perp/reports/s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_"
    "largestfold_oos15_ae3000_nocrossfit_k34567_payload300k_20260706/"
    "s52_trailing_regime_meta_handoff_top30_allsafe_aegmm_fixedtargets_oos15_20260706/"
    "s52_trailing_regime_scored_ledger.parquet"
)
TARGET_COLUMNS = [
    "__first_touch_target_soft__",
    "target_soft",
    "clean_exec",
    "dirty_positive",
    "full_path_bad_mae_1r",
    "first_touch_bad_mae_1r",
    "timeout",
    "ev_after_1pct",
    "exec_margin",
]


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--old-predictions", type=Path, default=DEFAULT_OLD_PREDICTIONS)
    parser.add_argument(
        "--old-training-predictions",
        type=Path,
        default=DEFAULT_OLD_TRAINING_PREDICTIONS,
    )
    parser.add_argument(
        "--training-label-handoff",
        type=Path,
        default=DEFAULT_TRAINING_LABEL_HANDOFF,
    )
    parser.add_argument("--aegmm", type=Path, default=DEFAULT_AEGMM)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source: AlternativeMetaResidualBundle = joblib.load(args.source_bundle)
    start = pd.Timestamp("2026-05-01", tz="UTC")
    split = pd.Timestamp("2026-07-01", tz="UTC")
    end = pd.Timestamp("2026-07-09", tz="UTC")
    pca_columns = list(source.residual_representation_state["columns"])
    required = list(
        dict.fromkeys(
            source.required_input_features()
            + source.raw_selected_features
            + source.residual_recognizer.candidate_features
            + pca_columns
            + TARGET_COLUMNS
            + list(COMPONENTS)
        )
    )
    frame = _read_july_handoff(args.handoff, required, start, end)
    train_frame = _merge_old_predictions(
        frame.loc[frame["__ts__"].lt(split)].copy(),
        args.old_training_predictions,
        start,
        split,
    )
    valid_frame = _merge_old_predictions(
        frame.loc[frame["__ts__"].ge(split)].copy(),
        args.old_predictions,
        split,
        end,
    )
    frame = pd.concat([train_frame, valid_frame], ignore_index=True, sort=False)
    label_rows = pd.read_parquet(
        args.training_label_handoff,
        columns=["__ts__", "__symbol__", "side_name", "__first_touch_target_soft__"],
    )
    label_rows["__ts__"] = pd.to_datetime(
        label_rows["__ts__"], utc=True, errors="coerce"
    )
    label_rows = label_rows.loc[
        label_rows["__ts__"].ge(start) & label_rows["__ts__"].lt(split)
    ].drop_duplicates(["__ts__", "__symbol__", "side_name"], keep="last")
    frame = frame.merge(
        label_rows,
        on=["__ts__", "__symbol__", "side_name"],
        how="left",
        validate="one_to_one",
        suffixes=("", "__label"),
    )
    if "__first_touch_target_soft____label" in frame.columns:
        frame["__first_touch_target_soft__"] = frame.get(
            "__first_touch_target_soft__",
            pd.Series(np.nan, index=frame.index),
        ).fillna(frame["__first_touch_target_soft____label"])
        frame = frame.drop(columns=["__first_touch_target_soft____label"])
    frame = _merge_frozen_aegmm(frame, args.aegmm, start, end)
    frame, coverage = _append_store_features(frame, args.feature_root, required)
    frame = frame.sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="stable"
    ).reset_index(drop=True)
    train = frame.loc[frame["__ts__"].lt(split)].copy()
    valid = frame.loc[frame["__ts__"].ge(split)].copy()
    if len(train) < 50_000 or len(valid) < 5_000:
        raise ValueError(
            f"Insufficient current-contract rows: train={len(train)} valid={len(valid)}"
        )

    y_train, target_name = _base_soft_label_target(train)
    valid_target = y_train.notna()
    train = train.loc[valid_target].copy()
    y_train = y_train.loc[valid_target]
    x_train, x_valid, medians = _matrix_fit_transform(
        train, valid, source.raw_selected_features
    )
    ood_state = _fit_ood_state(x_train, source.raw_selected_features)
    x_train = _apply_ood_state(x_train, ood_state).reindex(
        columns=source.selected_features, fill_value=0.0
    )
    x_valid = _apply_ood_state(x_valid, ood_state).reindex(
        columns=source.selected_features, fill_value=0.0
    )
    params = source.lifecycle_model.get_params()
    lifecycle_model = _fit_base_soft_label_model(
        x_train,
        y_train,
        train,
        20260711,
        lgbm_params=params,
    )
    lifecycle_train = _predict(lifecycle_model, x_train, classifier=False)

    pca_state = _fit_pca(
        train,
        pca_columns,
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

    recognizer = ResidualArchetypeRecognizer(
        ResidualArchetypeConfig(use_residual_ae_gmm=False, random_state=20260711),
        list(
            dict.fromkeys(
                source.residual_recognizer.candidate_features + output_columns
            )
        ),
    ).fit(train)
    train_safe = strip_outcomes_for_oos(train)
    train_residual = recognizer.transform_oos(train_safe).set_axis(train.index)
    overlay_fit = pd.concat([train_safe, train_residual], axis=1, copy=False)
    overlay = ResidualOverlayState(
        hit_alpha=float(source.overlay_state.hit_alpha),
        dirty_lambda=float(source.overlay_state.dirty_lambda),
        local_hit_alpha=float(source.overlay_state.local_hit_alpha),
        local_dirty_lambda=float(source.overlay_state.local_dirty_lambda),
    ).fit_normalization(overlay_fit)
    overlay_train = overlay.transform(overlay_fit, lifecycle_train)

    shock_train = train.copy(deep=False)
    shock_train["training_rank_pct"] = _rank(shock_train)
    raw_score = pd.to_numeric(
        shock_train["score_meta_base_soft_label"], errors="coerce"
    ).fillna(0.5)
    clean = pd.to_numeric(shock_train["clean_exec"], errors="coerce").fillna(0.0)
    shock_train["risk_target"] = (raw_score - clean).clip(lower=0.0).astype(np.float32)
    fitted_shock = _fit_state(shock_train)
    shock_state = ResidualShockOverlayState(
        references={
            name: np.asarray(values, dtype=np.float32)
            for name, values in fitted_shock.references.items()
        },
        archetype_multipliers=dict(fitted_shock.archetype_multipliers),
        train_end=str(fitted_shock.train_end),
    )
    adjusted_train, _, _ = shock_state.adjust_scores(
        train_safe, overlay_train, source.shock_side_parameters
    )
    hit_calibrator = _fit_platt(
        pd.Series(adjusted_train), train["clean_exec"].reset_index(drop=True)
    )
    rank_train = pd.DataFrame(
        {
            "__ts__": train["__ts__"],
            "side_name": train["side_name"],
            "score_shock_adjusted": adjusted_train,
        }
    )
    rank_reference = HistoricalScoreRankReference(
        score_col="score_shock_adjusted", side_col="side_name"
    ).fit(rank_train)

    bundle = AlternativeMetaResidualBundle(
        lifecycle_model=lifecycle_model,
        selected_features=list(source.selected_features),
        raw_selected_features=list(source.raw_selected_features),
        feature_medians=medians,
        ood_state=ood_state,
        residual_recognizer=recognizer,
        overlay_state=overlay,
        residual_representation_state=representation_state,
        hit_calibrator=hit_calibrator,
        historical_rank_reference=rank_reference,
        shock_overlay_state=shock_state,
        shock_side_parameters=dict(source.shock_side_parameters),
        fit_through=str(train["__ts__"].max()),
        frozen_ae_gmm_sha256=source.frozen_ae_gmm_sha256,
        metadata={
            **dict(source.metadata),
            "role": "current_handoff_contract_refit_for_july_oos",
            "feature_selection_frozen": True,
            "hpo_params_frozen": True,
            "policy_frozen": True,
        },
    )
    bundle_path = args.output_dir / "alternative_meta_residual_current_contract.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    predicted = bundle.predict(valid)
    scored = valid[
        ["__ts__", "__symbol__", "side_name", "archetype_policy_key", *TARGET_COLUMNS]
    ].copy()
    scored[predicted.columns] = predicted.to_numpy(dtype=np.float32, copy=False)
    scored.to_parquet(
        args.output_dir / "july_oos_predictions.parquet",
        index=False,
        compression="zstd",
    )

    historical = train[
        [
            "__ts__",
            "__symbol__",
            "side_name",
            "archetype_policy_key",
            "score_meta_base_soft_label",
        ]
    ].copy()
    historical["score_current_reference"] = historical["score_meta_base_soft_label"]
    historical["score_adjusted"] = adjusted_train
    historical.to_parquet(
        args.output_dir / "train_reference_scores.parquet",
        index=False,
        compression="zstd",
    )
    manifest = {
        "schema": "meta_residual_current_contract_july_oos_v1",
        "source_bundle": str(args.source_bundle),
        "bundle": str(bundle_path),
        "train_start": train["__ts__"].min(),
        "train_end": train["__ts__"].max(),
        "train_rows": int(len(train)),
        "july_rows": int(len(valid)),
        "july_start": valid["__ts__"].min(),
        "july_end": valid["__ts__"].max(),
        "target": target_name,
        "selected_features": list(source.selected_features),
        "lgbm_params": params,
        "required_input_feature_count": len(required),
        "store_coverage": coverage,
        "leakage_contract": (
            "All fitted states use May-June rows only. July outcomes are excluded from every fit and are retained solely for reporting. "
            "Feature selection, HPO parameters, architecture, overlay coefficients, shock thresholds, and downstream policy are frozen."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2), encoding="utf-8"
    )
    print(json.dumps(_safe(manifest), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
