#!/usr/bin/env python3
"""Ablate a config-routed base -> residual CatBoost event architecture.

The base classifier receives only the resolved base feature families from
``config.py``. A residual regressor receives only resolved meta feature
families plus cross-fitted base probabilities. Residual shrinkage is selected
on the purged April holdout. May, June, and partial July remain expanding OOF.
The strongest v2 conditional-quality prediction is composed only after the
event probability has been generated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import config as model_config  # noqa: E402
from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.exact_policy_soft_binary_ablation import (  # noqa: E402
    economic_metrics,
)
from extreme_price_movements.meaningful_mfe_event_ablation import (  # noqa: E402
    TripleBarrierSoftLabel,
    atr_soft_triple_barrier_labels,
    event_quality_scores,
    expanding_resolved_month_folds,
    first_21d_admission,
)
from scripts.run_meaningful_mfe_catboost_v2_ablation import (  # noqa: E402
    DEFAULT_CONTEXT,
    DEFAULT_EXACT_POLICY,
    DEFAULT_FEATURE_DIR,
    DEFAULT_LABELS,
    DEFAULT_LABEL_RESOLUTION_COLUMN,
    IDENTITY,
    SIDES,
    _classification_metrics,
    _fit_catboost,
    _predict,
    _rolling_platt,
    _safe,
    _univariate_prescreen,
)
from scripts.run_path_auxiliary_lgbm_models import (  # noqa: E402
    _file_sha256,
    _join_archetype_context,
    _load_labels,
    _load_static_features,
    _overlay_handoff_model_features,
)

SCHEMA = "meaningful_mfe_base_residual_catboost_ablation_v1"
DEFAULT_V2 = (
    ROOT
    / "data_perp/artifacts/meaningful_mfe_catboost_v2_ablation_20260725_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/meaningful_mfe_base_residual_catboost_ablation_20260725_v1"
)
BASE_GROUPS = (
    "base_shared_feature_keys",
    "base_long_feature_keys",
    "base_short_feature_keys",
)
META_GROUPS = (
    "meta_shared_feature_keys",
    "meta_product_feature_keys",
    "meta_reg_feature_keys",
    "meta_clf_feature_keys",
)


def _expand_config_features(values: Sequence[str]) -> list[str]:
    """Resolve config feature-family aliases recursively and deterministically."""

    resolved: list[str] = []
    visiting: set[str] = set()

    def visit(value: str) -> None:
        key = str(value)
        if key == "FEATURE_SELECTION_KEYS":
            # This routing helper points back to both base and meta groups. It
            # is not itself a model feature and must not collapse the split.
            return
        if key in visiting:
            return
        nested = CFG.get(key, getattr(model_config, key, None))
        is_alias = isinstance(nested, (list, tuple))
        if is_alias:
            visiting.add(key)
            for item in nested:
                visit(str(item))
            visiting.remove(key)
        else:
            resolved.append(key)

    for item in values:
        visit(str(item))
    return list(dict.fromkeys(resolved))


def _configured_features_by_side() -> tuple[dict[str, list[str]], list[str]]:
    shared_base = _expand_config_features(CFG.get("base_shared_feature_keys", []))
    base_by_side = {
        "long": list(
            dict.fromkeys(
                [
                    *shared_base,
                    *_expand_config_features(CFG.get("base_long_feature_keys", [])),
                ]
            )
        ),
        "short": list(
            dict.fromkeys(
                [
                    *shared_base,
                    *_expand_config_features(CFG.get("base_short_feature_keys", [])),
                ]
            )
        ),
    }
    meta = _expand_config_features(
        [
            *CFG.get("meta_shared_feature_keys", []),
            *CFG.get("meta_product_feature_keys", []),
            *CFG.get("meta_reg_feature_keys", []),
            *CFG.get("meta_clf_feature_keys", []),
        ]
    )
    return base_by_side, meta


def _load_config_matrix(
    labels: pd.DataFrame,
    *,
    context_path: Path,
    feature_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], list[str], dict[str, Any]]:
    labels, context_report = _join_archetype_context(
        labels, context_path, labels_are_canonical_top40=False
    )
    base_by_side, meta = _configured_features_by_side()
    requested = list(
        dict.fromkeys(
            [
                *base_by_side["long"],
                *base_by_side["short"],
                *meta,
            ]
        )
    )
    matrix, report = _load_static_features(
        labels,
        feature_dir=feature_dir,
        requested_features=requested,
        read_cache=None,
    )
    matrix, report = _overlay_handoff_model_features(
        matrix,
        labels,
        requested_features=requested,
        static_report=report,
        handoff_feature_columns=context_report["handoff_model_feature_columns"],
    )
    unavailable = [column for column in requested if matrix[column].isna().all()]
    matrix = matrix.drop(columns=unavailable).astype(np.float32)
    for side in SIDES:
        base_by_side[side] = [
            column for column in base_by_side[side] if column in matrix
        ]
    meta = [column for column in meta if column in matrix]
    if min(map(len, base_by_side.values())) < 20 or len(meta) < 20:
        raise ValueError(
            "config-routed base/meta feature support is insufficient: "
            f"base={dict(map(lambda item: (item[0], len(item[1])), base_by_side.items()))}, "
            f"meta={len(meta)}"
        )
    return labels, matrix, base_by_side, meta, {
        "requested_base_by_side": _configured_features_by_side()[0],
        "requested_meta": _configured_features_by_side()[1],
        "available_base_by_side": base_by_side,
        "available_meta": meta,
        "unavailable": unavailable,
        "load_report": report,
        "source_groups": {
            "base": list(BASE_GROUPS),
            "meta": list(META_GROUPS),
        },
    }


def _crossfit_base(
    X: pd.DataFrame,
    target: np.ndarray,
    timestamps: pd.Series,
    resolved: pd.Series,
    params: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate expanding base OOF predictions inside one outer train window."""

    ts = pd.to_datetime(timestamps, utc=True)
    unique = np.sort(ts.unique())
    boundaries = np.unique(
        np.quantile(np.arange(len(unique)), [0.25, 0.50, 0.75]).astype(int)
    )
    prediction = np.full(len(X), np.nan, dtype=np.float64)
    reports = []
    for fold, start_position in enumerate(boundaries):
        start = pd.Timestamp(unique[start_position])
        stop = (
            pd.Timestamp(unique[boundaries[fold + 1]])
            if fold + 1 < len(boundaries)
            else pd.Timestamp(unique[-1]) + pd.Timedelta(nanoseconds=1)
        )
        train = np.flatnonzero(
            ts.lt(start).to_numpy() & resolved.lt(start).to_numpy()
        )
        valid = np.flatnonzero(
            ts.ge(start).to_numpy() & ts.lt(stop).to_numpy()
        )
        if len(train) < 2_000 or len(valid) < 500:
            continue
        model = _fit_catboost(
            "binary",
            X.iloc[train],
            target[train],
            params,
            seed=seed + fold,
        )
        prediction[valid] = _predict(model, "binary", X.iloc[valid])
        reports.append(
            {
                "fold": int(fold),
                "train_rows": int(len(train)),
                "validation_rows": int(len(valid)),
                "validation_start": start,
                "validation_end": stop,
                "training_label_resolved_max": resolved.iloc[train].max(),
            }
        )
    return prediction, reports


def _fit_base_residual(
    base_X: pd.DataFrame,
    meta_X: pd.DataFrame,
    target: np.ndarray,
    frame: pd.DataFrame,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    base_params: Mapping[str, Any],
    residual_params: Mapping[str, Any],
    shrinkage: float,
    seed: int,
    ensemble_seeds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    local_train_frame = frame.iloc[train].reset_index(drop=True)
    base_oof, crossfit_report = _crossfit_base(
        base_X.iloc[train].reset_index(drop=True),
        target[train],
        local_train_frame["__ts__"],
        local_train_frame[DEFAULT_LABEL_RESOLUTION_COLUMN],
        base_params,
        seed=seed + 10_000,
    )
    residual_rows = np.flatnonzero(np.isfinite(base_oof))
    if len(residual_rows) < 2_000:
        raise ValueError("base cross-fit produced insufficient residual rows")
    residual_train_X = meta_X.iloc[train].iloc[residual_rows].copy()
    residual_train_X["__base_oof_probability__"] = base_oof[residual_rows]
    residual_target = target[train][residual_rows] - base_oof[residual_rows]
    residual_model = _fit_catboost(
        "quality",
        residual_train_X,
        residual_target,
        residual_params,
        seed=seed + 20_000,
    )
    base_predictions = []
    for seed_index in range(ensemble_seeds):
        base_model = _fit_catboost(
            "binary",
            base_X.iloc[train],
            target[train],
            base_params,
            seed=seed + seed_index,
        )
        base_predictions.append(
            _predict(base_model, "binary", base_X.iloc[validation])
        )
    base_prediction = np.mean(np.vstack(base_predictions), axis=0)
    residual_valid_X = meta_X.iloc[validation].copy()
    residual_valid_X["__base_oof_probability__"] = base_prediction
    residual_prediction = np.asarray(
        residual_model.predict(residual_valid_X), dtype=np.float64
    )
    combined = np.clip(
        base_prediction + float(shrinkage) * residual_prediction,
        1e-6,
        1.0 - 1e-6,
    )
    return (
        combined,
        base_prediction,
        residual_prediction,
        {
            "crossfit": crossfit_report,
            "residual_rows": int(len(residual_rows)),
            "residual_target_mean": float(np.mean(residual_target)),
            "residual_target_std": float(np.std(residual_target)),
            "base_prediction_mean": float(np.mean(base_prediction)),
            "residual_prediction_mean": float(np.mean(residual_prediction)),
            "shrinkage": float(shrinkage),
        },
    )


def _april_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    start = pd.Timestamp("2026-04-22T00:00:00Z")
    stop = pd.Timestamp("2026-05-01T00:00:00Z")
    train = np.flatnonzero(
        frame["__ts__"].lt(start).to_numpy()
        & frame[DEFAULT_LABEL_RESOLUTION_COLUMN].lt(start).to_numpy()
    )
    valid = np.flatnonzero(
        frame["__ts__"].ge(start).to_numpy()
        & frame["__ts__"].lt(stop).to_numpy()
    )
    return train, valid


def run(
    *,
    labels_path: Path = DEFAULT_LABELS,
    context_path: Path = DEFAULT_CONTEXT,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    exact_policy_path: Path = DEFAULT_EXACT_POLICY,
    v2_dir: Path = DEFAULT_V2,
    output_dir: Path = DEFAULT_OUTPUT,
    seed: int = 20260725,
    training_target: str = "clean",
) -> dict[str, Any]:
    if training_target not in {"clean", "literal"}:
        raise ValueError("training_target must be clean or literal")
    output_dir.mkdir(parents=True, exist_ok=True)
    v2_summary = json.loads((v2_dir / "summary.json").read_text())
    v2_predictions = pd.read_parquet(v2_dir / "oof_predictions.parquet")
    labels, label_report = _load_labels(
        labels_path,
        label_resolution_column=DEFAULT_LABEL_RESOLUTION_COLUMN,
    )
    if "__path_auxiliary_atr_fraction__" not in labels:
        atr = pd.read_parquet(
            labels_path,
            columns=[
                "__ts__",
                "__symbol__",
                "side",
                "candidate_id",
                "__path_auxiliary_atr_fraction__",
            ],
        )
        atr["__ts__"] = pd.to_datetime(atr["__ts__"], utc=True, errors="raise")
        labels = labels.merge(
            atr,
            on=["__ts__", "__symbol__", "side", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
    labels["side_name"] = labels["side"].astype(str)
    labels, matrix, base_by_side, meta_features, feature_report = _load_config_matrix(
        labels,
        context_path=context_path,
        feature_dir=feature_dir,
    )
    labels["side_name"] = labels["side"].astype(str)
    valid = labels["__path_auxiliary_target_valid__"].eq(1).to_numpy()
    labels = labels.loc[valid].reset_index(drop=True)
    matrix = matrix.loc[valid].reset_index(drop=True)
    baseline = atr_soft_triple_barrier_labels(labels, TripleBarrierSoftLabel())
    clean_hard = baseline["tb_hard_label"].to_numpy(np.float32)
    meaningful = labels["__meaningful_mfe_reached_12h__"].to_numpy(np.float32)
    hard = clean_hard if training_target == "clean" else meaningful
    soft = baseline["tb_soft_label"].to_numpy(np.float32)
    objective_soft = soft if training_target == "clean" else hard

    selected_shrinkage: dict[str, float] = {}
    selected_features: dict[str, dict[str, list[str]]] = {}
    april_predictions: dict[str, np.ndarray] = {}
    april_base_predictions: dict[str, np.ndarray] = {}
    hpo: dict[str, Any] = {}
    for side_index, side in enumerate(SIDES):
        positions = np.flatnonzero(labels["side_name"].eq(side).to_numpy())
        local_frame = labels.iloc[positions].reset_index(drop=True)
        train, validation = _april_split(local_frame)
        base_params = v2_summary["winners"][side]["hard"]["geometry"]
        residual_params = v2_summary["winners"][side]["soft"]["geometry"]
        base_selected = _univariate_prescreen(
            matrix.iloc[positions].iloc[train][base_by_side[side]],
            hard[positions][train],
            mandatory=(),
            limit=80,
        )
        selector_base_oof, _ = _crossfit_base(
            matrix.iloc[positions]
            .iloc[train][base_selected]
            .reset_index(drop=True),
            hard[positions][train],
            local_frame.iloc[train]["__ts__"].reset_index(drop=True),
            local_frame.iloc[train][DEFAULT_LABEL_RESOLUTION_COLUMN].reset_index(
                drop=True
            ),
            base_params,
            seed=seed + 50_000 + 1000 * side_index,
        )
        selector_rows = np.flatnonzero(np.isfinite(selector_base_oof))
        selector_residual = (
            hard[positions][train][selector_rows]
            - selector_base_oof[selector_rows]
        )
        meta_selected = _univariate_prescreen(
            matrix.iloc[positions]
            .iloc[train]
            .iloc[selector_rows][meta_features],
            selector_residual,
            mandatory=(),
            limit=80,
        )
        selected_features[side] = {
            "base": base_selected,
            "meta": meta_selected,
        }
        raw, base_only, residual_component, architecture = _fit_base_residual(
            matrix.iloc[positions][base_selected].reset_index(drop=True),
            matrix.iloc[positions][meta_selected].reset_index(drop=True),
            hard[positions],
            local_frame,
            train,
            validation,
            base_params=base_params,
            residual_params=residual_params,
            shrinkage=1.0,
            seed=seed + 100_000 * side_index,
            ensemble_seeds=1,
        )
        trials = []
        trial_predictions = []
        for shrinkage in (0.0, 0.25, 0.50, 0.75, 1.0):
            prediction = np.clip(
                base_only + shrinkage * residual_component,
                1e-6,
                1.0 - 1e-6,
            )
            metrics = _classification_metrics(
                hard[positions][validation],
                objective_soft[positions][validation],
                prediction,
            )
            objective = (
                metrics["log_loss_hard"]
                + metrics["brier_hard"]
                - 0.20 * metrics["roc_auc"]
                - 0.05 * metrics["top10_precision"]
            )
            trials.append(
                {
                    "shrinkage": shrinkage,
                    "objective": float(objective),
                    **metrics,
                }
            )
            trial_predictions.append(prediction)
        winner_index = min(
            range(len(trials)),
            key=lambda index: (trials[index]["objective"], trials[index]["shrinkage"]),
        )
        selected_shrinkage[side] = float(trials[winner_index]["shrinkage"])
        april_predictions[side] = trial_predictions[winner_index]
        april_base_predictions[side] = base_only
        hpo[side] = {
            "trials": trials,
            "winner": trials[winner_index],
            "architecture": architecture,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
        }
        print(
            f"selected residual side={side} shrinkage={selected_shrinkage[side]:.2f}",
            flush=True,
        )

    base_only_oof = np.full(len(labels), np.nan, dtype=np.float64)
    base_residual = np.full(len(labels), np.nan, dtype=np.float64)
    fold_reports = []
    for side_index, side in enumerate(SIDES):
        positions = np.flatnonzero(labels["side_name"].eq(side).to_numpy())
        local_frame = labels.iloc[positions].reset_index(drop=True)
        local_base = matrix.iloc[positions][
            selected_features[side]["base"]
        ].reset_index(drop=True)
        local_meta = matrix.iloc[positions][
            selected_features[side]["meta"]
        ].reset_index(drop=True)
        for fold in expanding_resolved_month_folds(
            local_frame["__ts__"],
            local_frame[DEFAULT_LABEL_RESOLUTION_COLUMN],
        ):
            train = np.asarray(fold["train_indices"], dtype=np.int64)
            validation = np.asarray(fold["validation_indices"], dtype=np.int64)
            prediction, base_prediction, _, architecture = _fit_base_residual(
                local_base,
                local_meta,
                hard[positions],
                local_frame,
                train,
                validation,
                base_params=v2_summary["winners"][side]["hard"]["geometry"],
                residual_params=v2_summary["winners"][side]["soft"]["geometry"],
                shrinkage=selected_shrinkage[side],
                seed=seed
                + 500_000
                + 100_000 * side_index
                + 1000 * int(fold["fold"]),
                ensemble_seeds=3,
            )
            base_only_oof[positions[validation]] = base_prediction
            base_residual[positions[validation]] = prediction
            fold_reports.append(
                {
                    "side": side,
                    "month": fold["month"],
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                    "training_label_resolved_max": fold[
                        "training_label_resolved_max"
                    ],
                    "architecture": architecture,
                }
            )
            print(
                f"base-residual oof side={side} month={fold['month']} "
                f"rows={len(validation)}",
                flush=True,
            )

    base_calibrated = np.full(len(labels), np.nan, dtype=np.float64)
    calibrated = np.full(len(labels), np.nan, dtype=np.float64)
    calibration_reports = []
    for side in SIDES:
        positions = np.flatnonzero(labels["side_name"].eq(side).to_numpy())
        _, april_valid = _april_split(labels.iloc[positions].reset_index(drop=True))
        local_calibrated, local_reports = _rolling_platt(
            base_residual,
            hard,
            labels,
            side=side,
            april_prediction=april_predictions[side],
            april_hard=hard[positions][april_valid],
        )
        finite = np.isfinite(local_calibrated)
        calibrated[finite] = local_calibrated[finite]
        calibration_reports.extend(local_reports)
        local_base_calibrated, local_base_reports = _rolling_platt(
            base_only_oof,
            hard,
            labels,
            side=side,
            april_prediction=april_base_predictions[side],
            april_hard=hard[positions][april_valid],
        )
        base_finite = np.isfinite(local_base_calibrated)
        base_calibrated[base_finite] = local_base_calibrated[base_finite]
        calibration_reports.extend(
            [
                {"architecture": "base_only", **report}
                for report in local_base_reports
            ]
        )

    aligned_v2 = labels.loc[:, list(IDENTITY)].merge(
        v2_predictions.loc[
            :,
            [
                *IDENTITY,
                "catboost_hard_ensemble_platt",
                "catboost_conditional_quality",
                "catboost_probability_x_quality",
                "catboost_probability_gated_quality",
            ],
        ],
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    quality = aligned_v2["catboost_conditional_quality"].to_numpy(np.float64)
    composed = event_quality_scores(calibrated, quality)
    scores = {
        "config_base_only_raw": base_only_oof,
        "config_base_only_platt": base_calibrated,
        "config_base_residual_raw": base_residual,
        "config_base_residual_platt": calibrated,
        "config_base_residual_probability_x_quality": composed[
            "probability_x_quality"
        ],
        "config_base_residual_probability_gated_quality": composed[
            "probability_gated_quality"
        ],
        "v2_hard_ensemble_platt": aligned_v2[
            "catboost_hard_ensemble_platt"
        ].to_numpy(np.float64),
        "v2_probability_x_quality": aligned_v2[
            "catboost_probability_x_quality"
        ].to_numpy(np.float64),
        "v2_probability_gated_quality": aligned_v2[
            "catboost_probability_gated_quality"
        ].to_numpy(np.float64),
    }
    predictions = labels.loc[:, list(IDENTITY)].copy()
    predictions["tb_hard_label"] = clean_hard
    predictions["tb_soft_label"] = soft
    predictions["meaningful_mfe_reached"] = labels[
        "__meaningful_mfe_reached_12h__"
    ].to_numpy(np.float32)
    for name, score in scores.items():
        predictions[name] = score

    exact = pd.read_parquet(exact_policy_path)
    paired = exact.merge(
        predictions,
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    reports = {}
    meaningful = predictions["meaningful_mfe_reached"].to_numpy(np.float32)
    for name, score in scores.items():
        finite = np.isfinite(score)
        paired_score = paired[name].to_numpy(np.float64)
        paired_finite = np.isfinite(paired_score)
        admission = first_21d_admission(
            paired["__ts__"],
            paired_score,
            pd.to_numeric(
                paired["execution_net_ev_12h"], errors="coerce"
            ).to_numpy(np.float64),
        )
        evaluation = np.asarray(admission["evaluation_mask"], dtype=bool)
        admitted = np.asarray(admission["admitted_mask"], dtype=bool)
        reports[name] = {
            "oof_rows": int(finite.sum()),
            "clean_event": _classification_metrics(
                clean_hard[finite], soft[finite], score[finite]
            ),
            "literal_event": _classification_metrics(
                meaningful[finite], meaningful[finite], score[finite]
            ),
            "exact_policy": economic_metrics(
                paired.loc[paired_finite].reset_index(drop=True),
                paired_score[paired_finite],
            ),
            "post_21d_admission": {
                "contract": {
                    key: value
                    for key, value in admission.items()
                    if key
                    not in {
                        "evaluation_mask",
                        "admitted_mask",
                        "calibrated_expected_net_return",
                    }
                },
                "raw_after_fit_window": economic_metrics(
                    paired.loc[evaluation].reset_index(drop=True),
                    paired_score[evaluation],
                ),
                "admitted": economic_metrics(
                    paired,
                    paired_score,
                    admitted=admitted,
                ),
            },
            "exact_policy_by_side_month": [
                {
                    "side": side,
                    "month": month,
                    **economic_metrics(
                        group.reset_index(drop=True),
                        group[name].to_numpy(np.float64),
                    ),
                }
                for (side, month), group in paired.loc[paired_finite].groupby(
                    [
                        "side_name",
                        paired.loc[paired_finite, "__ts__"].dt.strftime("%Y-%m"),
                    ]
                )
            ],
        }

    output_predictions = output_dir / "oof_predictions.parquet"
    predictions.to_parquet(output_predictions, index=False)
    output_paired = output_dir / "exact_policy_paired.parquet"
    paired.to_parquet(output_paired, index=False)
    summary = {
        "schema": SCHEMA,
        "training_target": training_target,
        "status": "research_oof_not_untouched_final_test",
        "architecture": (
            "config base features -> CatBoost clean-event probability -> "
            "cross-fitted probability residual on config meta features + "
            "base OOF probability -> rolling Platt -> v2 conditional quality"
        ),
        "chronology": {
            "residual_training": (
                "base predictions are expanding cross-fit within each outer "
                "training window; residual rows never use in-sample base scores"
            ),
            "selection": (
                "residual shrinkage selected on April 22-30 only after every "
                "training label resolves before April 22"
            ),
            "outer_oof": (
                "expanding May, June, July; every outer and inner training "
                "label resolves before its validation boundary"
            ),
            "calibration": (
                "May uses April held-out; June adds May OOF; July adds June OOF"
            ),
            "final_test_disclosure": (
                "July 1-10 was previously inspected and is not untouched"
            ),
        },
        "feature_contract": {
            **feature_report,
            "selected_by_side": selected_features,
            "selection_contract": (
                "side-local top-80 univariate screen inside config-routed "
                "base and meta pools; meta screen targets cross-fitted base residual"
            ),
        },
        "selected_shrinkage_by_side": selected_shrinkage,
        "hpo": hpo,
        "folds": fold_reports,
        "calibration": calibration_reports,
        "reports": reports,
        "rows": {
            "valid_labels": int(len(labels)),
            "outer_oof": int(np.isfinite(base_residual).sum()),
            "exact_policy_paired": int(len(paired)),
        },
        "sources": {
            "labels": {
                "path": str(labels_path),
                "sha256": _file_sha256(labels_path),
                "report": label_report,
            },
            "v2_summary": {
                "path": str(v2_dir / "summary.json"),
                "sha256": _file_sha256(v2_dir / "summary.json"),
            },
            "v2_predictions": {
                "path": str(v2_dir / "oof_predictions.parquet"),
                "sha256": _file_sha256(v2_dir / "oof_predictions.parquet"),
            },
            "exact_policy": {
                "path": str(exact_policy_path),
                "sha256": _file_sha256(exact_policy_path),
            },
        },
        "outputs": {
            "predictions": str(output_predictions),
            "exact_policy_paired": str(output_paired),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--context-path", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--exact-policy-path", type=Path, default=DEFAULT_EXACT_POLICY)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--training-target", choices=("clean", "literal"), default="clean"
    )
    args = parser.parse_args(argv)
    summary = run(**vars(args))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "rows": summary["rows"],
                "selected_shrinkage_by_side": summary[
                    "selected_shrinkage_by_side"
                ],
                "output": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
