#!/usr/bin/env python3
"""Run side-local meaningful-MFE event-classifier ablations.

Model settings are selected only on a purged April split.  The selected
settings are frozen and refitted in expanding May/June/July folds whose
training labels resolve strictly before each validation month.  The existing
meaningful-event feature-selection contracts remain frozen so this experiment
isolates classifier family and label geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_policy_soft_binary_ablation import (  # noqa: E402
    economic_metrics,
)
from extreme_price_movements.meaningful_mfe_event_ablation import (  # noqa: E402
    TripleBarrierSoftLabel,
    atr_soft_triple_barrier_labels,
    expanding_resolved_month_folds,
)
from extreme_price_movements.path_auxiliary_lgbm import (  # noqa: E402
    fit_base_archetype_label_feature_contract,
    transform_base_archetype_label_features,
)
from scripts.run_path_auxiliary_lgbm_models import (  # noqa: E402
    ARCHETYPE_COLUMNS,
    DEFAULT_LABEL_RESOLUTION_COLUMN,
    MANDATORY_HANDOFF_MODEL_FEATURES,
    REPRESENTATION_AVAILABLE_FEATURE,
    _complete_archetype_source,
    _file_sha256,
    _join_archetype_context,
    _load_labels,
    _load_static_features,
    _overlay_handoff_model_features,
)

SCHEMA = "meaningful_mfe_event_classifier_ablation_v1"
DEFAULT_LABELS = (
    ROOT
    / "data_perp/artifacts/packb_path_auxiliary_targets_20260725_v1_31_8/targets.parquet"
)
DEFAULT_CONTEXT = (
    ROOT
    / "data_perp/artifacts/packb_downstream_context_20260725_v2_31_8_frozen_ae_gmm/context.parquet"
)
DEFAULT_FEATURE_DIR = ROOT / "data_perp/features/20260711_070000"
DEFAULT_SELECTION = (
    ROOT
    / "data_perp/artifacts/packb_path_auxiliary_role_bundles_20260725_v1_31_8/shared/selection_contracts.joblib"
)
DEFAULT_INCUMBENT = (
    ROOT
    / "data_perp/artifacts/packb_path_auxiliary_role_bundles_20260725_v1_31_8/peak_mfe_12h_atr/oof_bundle.parquet"
)
DEFAULT_EXACT_POLICY = (
    ROOT
    / "data_perp/artifacts/execution_ev_joined_handoff_policy_labels_20260725_v2/joined.parquet"
)
DEFAULT_OUTPUT = (
    ROOT / "data_perp/artifacts/meaningful_mfe_event_classifier_ablation_20260725_v1"
)
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
SIDES = ("long", "short")
MODEL_GRIDS: Mapping[str, tuple[Mapping[str, Any], ...]] = {
    "logistic": (
        {"C": 0.05},
        {"C": 0.5},
    ),
    "lightgbm": (
        {
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 250,
            "reg_lambda": 8.0,
        },
        {
            "num_leaves": 31,
            "max_depth": 7,
            "min_child_samples": 150,
            "reg_lambda": 12.0,
        },
    ),
    "catboost": (
        {"depth": 5, "l2_leaf_reg": 8.0},
        {"depth": 7, "l2_leaf_reg": 12.0},
    ),
}


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit_model(
    family: str,
    params: Mapping[str, Any],
    X: pd.DataFrame,
    target: np.ndarray,
    *,
    seed: int,
) -> Any:
    if family == "logistic":
        # Duplicated pseudo-observations make the weighted Bernoulli loss
        # exactly equal to soft-label cross entropy.
        augmented_X = pd.concat([X, X], ignore_index=True)
        augmented_y = np.r_[np.ones(len(X)), np.zeros(len(X))]
        weights = np.r_[target, 1.0 - target]
        keep = weights > 1e-8
        pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=float(params["C"]),
                max_iter=250,
                solver="lbfgs",
                random_state=int(seed),
            ),
        )
        pipeline.fit(
            augmented_X.iloc[np.flatnonzero(keep)],
            augmented_y[keep],
            logisticregression__sample_weight=weights[keep],
        )
        return pipeline
    if family == "lightgbm":
        model = lgb.LGBMRegressor(
            objective="cross_entropy",
            n_estimators=320,
            learning_rate=0.035,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            n_jobs=6,
            verbosity=-1,
            random_state=int(seed),
            **dict(params),
        )
        model.fit(X, target)
        return model
    if family == "catboost":
        model = CatBoostClassifier(
            loss_function="CrossEntropy",
            iterations=300,
            learning_rate=0.04,
            random_seed=int(seed),
            thread_count=6,
            verbose=False,
            allow_writing_files=False,
            **dict(params),
        )
        model.fit(X, target)
        return model
    raise ValueError(f"unknown classifier family: {family}")


def _predict(model: Any, family: str, X: pd.DataFrame) -> np.ndarray:
    if family in {"logistic", "catboost"}:
        prediction = model.predict_proba(X)[:, 1]
    else:
        prediction = model.predict(X)
    return np.clip(np.asarray(prediction, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def _classification_metrics(
    hard: np.ndarray,
    soft: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 1e-6, 1 - 1e-6)
    hard = np.asarray(hard, dtype=np.float64)
    soft = np.asarray(soft, dtype=np.float64)
    order = np.argsort(-prediction, kind="stable")
    n10 = max(1, int(np.ceil(len(order) * 0.10)))
    selected = order[:n10]
    bins = pd.qcut(prediction, 10, labels=False, duplicates="drop")
    calibration = (
        pd.DataFrame({"bin": bins, "prediction": prediction, "hard": hard})
        .groupby("bin", sort=True)
        .agg(
            rows=("hard", "size"),
            prediction=("prediction", "mean"),
            observed=("hard", "mean"),
        )
        .reset_index()
    )
    return {
        "rows": int(len(hard)),
        "prevalence": float(np.mean(hard)),
        "roc_auc": float(roc_auc_score(hard, prediction)),
        "average_precision": float(average_precision_score(hard, prediction)),
        "brier_hard": float(brier_score_loss(hard, prediction)),
        "brier_soft": float(np.mean((prediction - soft) ** 2)),
        "log_loss_hard": float(log_loss(hard, prediction, labels=[0.0, 1.0])),
        "spearman_soft": float(spearmanr(prediction, soft).statistic),
        "top10_rows": int(n10),
        "top10_precision": float(np.mean(hard[selected])),
        "top10_recall": float(np.sum(hard[selected]) / max(np.sum(hard), 1.0)),
        "ece": float(
            np.average(
                np.abs(calibration["prediction"] - calibration["observed"]),
                weights=calibration["rows"],
            )
        ),
        "calibration_bins": calibration.to_dict(orient="records"),
    }


def _inner_april_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    validation_start = pd.Timestamp("2026-04-22T00:00:00Z")
    validation_end = pd.Timestamp("2026-05-01T00:00:00Z")
    train = np.flatnonzero(
        frame["__ts__"].lt(validation_start).to_numpy()
        & frame[DEFAULT_LABEL_RESOLUTION_COLUMN].lt(validation_start).to_numpy()
    )
    valid = np.flatnonzero(
        frame["__ts__"].ge(validation_start).to_numpy()
        & frame["__ts__"].lt(validation_end).to_numpy()
    )
    if len(train) < 2_000 or len(valid) < 500:
        raise ValueError("April HPO split has insufficient support")
    return train, valid


def _hpo_family(
    family: str,
    X: pd.DataFrame,
    hard: np.ndarray,
    soft: np.ndarray,
    frame: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, Any]:
    train, valid = _inner_april_split(frame)
    trials = []
    for index, params in enumerate(MODEL_GRIDS[family]):
        model = _fit_model(
            family, params, X.iloc[train], soft[train], seed=seed + index
        )
        prediction = _predict(model, family, X.iloc[valid])
        metrics = _classification_metrics(hard[valid], soft[valid], prediction)
        objective = (
            metrics["log_loss_hard"] + metrics["brier_soft"] - 0.10 * metrics["roc_auc"]
        )
        trials.append(
            {"params": dict(params), "objective": float(objective), **metrics}
        )
    winner = min(trials, key=lambda row: (row["objective"], str(row["params"])))
    return {
        "selection_period": "train resolved before 2026-04-22; validate 2026-04-22..2026-04-30",
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "trials": trials,
        "winner": winner,
    }


def _load_feature_matrix(
    labels: pd.DataFrame,
    *,
    context_path: Path,
    feature_dir: Path,
    selection_path: Path,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    labels, context_report = _join_archetype_context(
        labels, context_path, labels_are_canonical_top40=False
    )
    selection_payload = joblib.load(selection_path)
    selection = selection_payload["selection_contracts"]["meaningful_mfe_event"]
    features_by_side = {
        side: list(map(str, selection["selected_features_by_side"][side]))
        for side in SIDES
    }
    complete_archetypes = [
        column
        for column in ARCHETYPE_COLUMNS
        if _complete_archetype_source(labels, column)
    ]
    if not complete_archetypes:
        raise ValueError("no complete base-archetype source for event ablation")
    reference = labels.loc[
        labels["__ts__"].lt(pd.Timestamp("2026-05-01T00:00:00Z"))
        & labels[DEFAULT_LABEL_RESOLUTION_COLUMN].lt(
            pd.Timestamp("2026-05-01T00:00:00Z")
        )
    ].reset_index(drop=True)
    archetype_contract = fit_base_archetype_label_feature_contract(
        reference,
        source_columns=complete_archetypes,
        canonical_source=complete_archetypes[0],
    )
    archetype = transform_base_archetype_label_features(labels, archetype_contract)
    union = list(
        dict.fromkeys([feature for side in SIDES for feature in features_by_side[side]])
    )
    matrix, report = _load_static_features(
        labels,
        feature_dir=feature_dir,
        requested_features=union,
        read_cache=None,
    )
    matrix, report = _overlay_handoff_model_features(
        matrix,
        labels,
        requested_features=union,
        static_report=report,
        handoff_feature_columns=context_report["handoff_model_feature_columns"],
    )
    archetype_columns = [column for column in union if column in archetype]
    if archetype_columns:
        matrix.loc[:, archetype_columns] = archetype.loc[:, archetype_columns].to_numpy(
            dtype=np.float32, copy=False
        )
    matrix = matrix.reindex(columns=union).astype(np.float32)
    missing = [column for column in union if matrix[column].isna().all()]
    if missing:
        raise ValueError("selected event features unavailable: " + ", ".join(missing))
    return (
        labels,
        features_by_side,
        {
            "feature_load": report,
            "archetype_contract": archetype_contract,
            "selection_sha256": _sha256(selection_path),
            "selected_features_by_side": features_by_side,
            "matrix": matrix,
        },
    )


def run(
    *,
    labels_path: Path = DEFAULT_LABELS,
    context_path: Path = DEFAULT_CONTEXT,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    selection_path: Path = DEFAULT_SELECTION,
    incumbent_path: Path = DEFAULT_INCUMBENT,
    exact_policy_path: Path = DEFAULT_EXACT_POLICY,
    output_dir: Path = DEFAULT_OUTPUT,
    seed: int = 20260725,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
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
        atr["__symbol__"] = atr["__symbol__"].astype(str)
        atr["side"] = atr["side"].astype(str)
        atr["candidate_id"] = atr["candidate_id"].astype(str)
        labels = labels.merge(
            atr,
            on=["__ts__", "__symbol__", "side", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
        if labels["__path_auxiliary_atr_fraction__"].isna().any():
            raise ValueError("ATR fraction exact-identity join is incomplete")
    labels["side_name"] = labels["side"].astype(str)
    labels, features_by_side, feature_payload = _load_feature_matrix(
        labels,
        context_path=context_path,
        feature_dir=feature_dir,
        selection_path=selection_path,
    )
    print(
        f"loaded selected feature matrix rows={len(labels)} "
        f"columns={feature_payload['feature_load']['available_features']}",
        flush=True,
    )
    labels["side_name"] = labels["side"].astype(str)
    matrix = feature_payload.pop("matrix")
    valid = labels["__path_auxiliary_target_valid__"].eq(1).to_numpy()
    labels = labels.loc[valid].reset_index(drop=True)
    matrix = matrix.loc[valid].reset_index(drop=True)
    baseline_contract = TripleBarrierSoftLabel()
    contracts = {
        "soft_tb_lower_0p5": TripleBarrierSoftLabel(lower_atr=0.5),
        "soft_tb_baseline_1p0": baseline_contract,
        "soft_tb_lower_1p5": TripleBarrierSoftLabel(lower_atr=1.5),
        "soft_tb_no_time_bonus": TripleBarrierSoftLabel(use_time_bonus=False),
    }
    target_frames = {
        name: atr_soft_triple_barrier_labels(labels, contract)
        for name, contract in contracts.items()
    }
    baseline = target_frames["soft_tb_baseline_1p0"]
    hard_meaningful = labels["__meaningful_mfe_reached_12h__"].to_numpy(
        dtype=np.float32
    )
    incumbent = pd.read_parquet(incumbent_path)
    incumbent = incumbent.rename(columns={"side": "side_name"})
    incumbent = labels.loc[:, list(IDENTITY)].merge(
        incumbent.loc[
            :,
            [
                *IDENTITY,
                "pred_p_meaningful_mfe_12h",
                "oof_fold",
                "train_decision_cutoff",
            ],
        ],
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    folds = expanding_resolved_month_folds(
        labels["__ts__"], labels[DEFAULT_LABEL_RESOLUTION_COLUMN]
    )
    prediction_columns: dict[str, np.ndarray] = {}
    reports: dict[str, Any] = {}
    hpo: dict[str, Any] = {}
    selected_family_by_side: dict[str, str] = {}
    selected_params_by_side: dict[str, dict[str, Any]] = {}

    for side_index, side in enumerate(SIDES):
        side_mask = labels["side_name"].eq(side).to_numpy()
        side_positions = np.flatnonzero(side_mask)
        local_frame = labels.iloc[side_positions].reset_index(drop=True)
        local_X = matrix.iloc[side_positions][features_by_side[side]].reset_index(
            drop=True
        )
        local_hard = baseline.iloc[side_positions]["tb_hard_label"].to_numpy(np.float32)
        local_soft = baseline.iloc[side_positions]["tb_soft_label"].to_numpy(np.float32)
        hpo[side] = {}
        for family_index, family in enumerate(MODEL_GRIDS):
            hpo[side][family] = _hpo_family(
                family,
                local_X,
                local_hard,
                local_soft,
                local_frame,
                seed=seed + 1000 * side_index + 100 * family_index,
            )
            print(
                f"hpo side={side} family={family} "
                f"objective={hpo[side][family]['winner']['objective']:.6f}",
                flush=True,
            )
        family_winner = min(
            MODEL_GRIDS,
            key=lambda family: (
                hpo[side][family]["winner"]["objective"],
                family,
            ),
        )
        selected_family_by_side[side] = family_winner
        selected_params_by_side[side] = dict(
            hpo[side][family_winner]["winner"]["params"]
        )

    fitted_arms: dict[str, dict[str, Any]] = {
        f"{family}_soft_tb_baseline": {
            "family_by_side": {side: family for side in SIDES},
            "params_by_side": {
                side: hpo[side][family]["winner"]["params"] for side in SIDES
            },
            "target": "soft_tb_baseline_1p0",
        }
        for family in MODEL_GRIDS
    }
    fitted_arms.update(
        {
            "winner_hard_meaningful": {
                "family_by_side": selected_family_by_side,
                "params_by_side": selected_params_by_side,
                "target": "hard_meaningful",
            },
            "winner_hard_tb": {
                "family_by_side": selected_family_by_side,
                "params_by_side": selected_params_by_side,
                "target": "hard_tb",
            },
            "winner_soft_tb_lower_0p5": {
                "family_by_side": selected_family_by_side,
                "params_by_side": selected_params_by_side,
                "target": "soft_tb_lower_0p5",
            },
            "winner_soft_tb_lower_1p5": {
                "family_by_side": selected_family_by_side,
                "params_by_side": selected_params_by_side,
                "target": "soft_tb_lower_1p5",
            },
            "winner_soft_tb_no_time_bonus": {
                "family_by_side": selected_family_by_side,
                "params_by_side": selected_params_by_side,
                "target": "soft_tb_no_time_bonus",
            },
        }
    )

    for arm_index, (arm, spec) in enumerate(fitted_arms.items()):
        prediction = np.full(len(labels), np.nan, dtype=np.float32)
        fold_id = np.full(len(labels), -1, dtype=np.int16)
        fold_reports = []
        for side_index, side in enumerate(SIDES):
            side_positions = np.flatnonzero(labels["side_name"].eq(side).to_numpy())
            local_frame = labels.iloc[side_positions].reset_index(drop=True)
            local_X = matrix.iloc[side_positions][features_by_side[side]].reset_index(
                drop=True
            )
            local_folds = expanding_resolved_month_folds(
                local_frame["__ts__"],
                local_frame[DEFAULT_LABEL_RESOLUTION_COLUMN],
            )
            if spec["target"] == "hard_meaningful":
                target = hard_meaningful[side_positions]
            elif spec["target"] == "hard_tb":
                target = baseline.iloc[side_positions]["tb_hard_label"].to_numpy(
                    np.float32
                )
            else:
                target = (
                    target_frames[spec["target"]]
                    .iloc[side_positions]["tb_soft_label"]
                    .to_numpy(np.float32)
                )
            for fold in local_folds:
                train = np.asarray(fold["train_indices"], dtype=np.int64)
                validation = np.asarray(fold["validation_indices"], dtype=np.int64)
                family = spec["family_by_side"][side]
                model = _fit_model(
                    family,
                    spec["params_by_side"][side],
                    local_X.iloc[train],
                    target[train],
                    seed=seed
                    + 10000 * arm_index
                    + 1000 * side_index
                    + int(fold["fold"]),
                )
                local_prediction = _predict(model, family, local_X.iloc[validation])
                global_positions = side_positions[validation]
                prediction[global_positions] = local_prediction.astype(np.float32)
                fold_id[global_positions] = int(fold["fold"])
                fold_reports.append(
                    {
                        "side": side,
                        "month": fold["month"],
                        "family": family,
                        "params": spec["params_by_side"][side],
                        "train_rows": int(len(train)),
                        "validation_rows": int(len(validation)),
                        "training_label_resolved_max": fold[
                            "training_label_resolved_max"
                        ],
                        "metrics": _classification_metrics(
                            baseline.iloc[global_positions]["tb_hard_label"].to_numpy(
                                np.float32
                            ),
                            baseline.iloc[global_positions]["tb_soft_label"].to_numpy(
                                np.float32
                            ),
                            local_prediction,
                        ),
                        "meaningful_event_metrics": _classification_metrics(
                            hard_meaningful[global_positions],
                            hard_meaningful[global_positions],
                            local_prediction,
                        ),
                    }
                )
                print(
                    f"oof arm={arm} side={side} month={fold['month']} "
                    f"rows={len(validation)}",
                    flush=True,
                )
        prediction_columns[arm] = prediction
        prediction_columns[f"{arm}__oof_fold"] = fold_id
        oof = np.isfinite(prediction)
        reports[arm] = {
            "contract": spec,
            "oof_rows": int(oof.sum()),
            "aggregate": _classification_metrics(
                baseline.loc[oof, "tb_hard_label"].to_numpy(np.float32),
                baseline.loc[oof, "tb_soft_label"].to_numpy(np.float32),
                prediction[oof],
            ),
            "meaningful_event_aggregate": _classification_metrics(
                hard_meaningful[oof],
                hard_meaningful[oof],
                prediction[oof],
            ),
            "folds": fold_reports,
        }

    incumbent_prediction = pd.to_numeric(
        incumbent["pred_p_meaningful_mfe_12h"], errors="coerce"
    ).to_numpy(np.float64)
    incumbent_oof = np.isfinite(incumbent_prediction)
    prediction_columns["incumbent_hard_meaningful"] = incumbent_prediction.astype(
        np.float32
    )
    reports["incumbent_hard_meaningful"] = {
        "contract": "frozen canonical meaningful-MFE event model",
        "oof_rows": int(incumbent_oof.sum()),
        "aggregate": _classification_metrics(
            baseline.loc[incumbent_oof, "tb_hard_label"].to_numpy(np.float32),
            baseline.loc[incumbent_oof, "tb_soft_label"].to_numpy(np.float32),
            incumbent_prediction[incumbent_oof],
        ),
        "meaningful_event_aggregate": _classification_metrics(
            hard_meaningful[incumbent_oof],
            hard_meaningful[incumbent_oof],
            incumbent_prediction[incumbent_oof],
        ),
    }

    predictions = labels.loc[:, list(IDENTITY)].copy()
    predictions["tb_hard_label"] = baseline["tb_hard_label"].to_numpy(np.float32)
    predictions["tb_soft_label"] = baseline["tb_soft_label"].to_numpy(np.float32)
    predictions["meaningful_mfe_reached"] = hard_meaningful
    for column, values in prediction_columns.items():
        predictions[column] = values

    exact = pd.read_parquet(exact_policy_path)
    paired = exact.merge(
        predictions,
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    for arm, report in reports.items():
        score = pd.to_numeric(paired[arm], errors="coerce").to_numpy(np.float64)
        finite = np.isfinite(score)
        report["exact_policy"] = economic_metrics(
            paired.loc[finite].reset_index(drop=True), score[finite]
        )
        report["exact_policy_by_side_month"] = []
        for (side, month), group in paired.loc[finite].groupby(
            ["side_name", paired.loc[finite, "__ts__"].dt.strftime("%Y-%m")]
        ):
            group_score = pd.to_numeric(group[arm], errors="coerce").to_numpy(
                np.float64
            )
            report["exact_policy_by_side_month"].append(
                {
                    "side": side,
                    "month": month,
                    **economic_metrics(group.reset_index(drop=True), group_score),
                }
            )

    output_predictions = output_dir / "oof_predictions.parquet"
    predictions.to_parquet(output_predictions, index=False)
    output_paired = output_dir / "exact_policy_paired.parquet"
    paired.to_parquet(output_paired, index=False)
    summary = {
        "schema": SCHEMA,
        "status": "research_oof_not_untouched_final_test",
        "label_contracts": {
            name: vars(contract) for name, contract in contracts.items()
        },
        "baseline_outcome_counts": baseline["tb_outcome"].value_counts().to_dict(),
        "chronology": {
            "hpo": "April only: train labels resolved before 2026-04-22; validate April 22-30",
            "outer_oof": "expanding May, June, July; training label resolution strictly before validation start",
            "final_test_disclosure": "July 1-10 has been inspected in prior research and is not untouched",
        },
        "feature_contract": {
            **feature_payload,
            "feature_selection_reused": True,
            "reason": "isolate model and label ablation; winner requires task-specific reselection before promotion",
            "mandatory_handoff_features": list(MANDATORY_HANDOFF_MODEL_FEATURES),
            "representation_feature": REPRESENTATION_AVAILABLE_FEATURE,
        },
        "hpo": hpo,
        "selected_family_by_side": selected_family_by_side,
        "reports": reports,
        "rows": {
            "valid_labels": int(len(labels)),
            "oof_prediction_rows": int(
                np.isfinite(
                    prediction_columns[
                        "lightgbm_soft_tb_baseline"
                        if "lightgbm_soft_tb_baseline" in prediction_columns
                        else next(iter(prediction_columns))
                    ]
                ).sum()
            ),
            "exact_policy_paired_rows": int(len(paired)),
        },
        "sources": {
            "labels": {
                "path": str(labels_path),
                "sha256": _file_sha256(labels_path),
                "report": label_report,
            },
            "context": {"path": str(context_path), "sha256": _sha256(context_path)},
            "selection": {
                "path": str(selection_path),
                "sha256": _sha256(selection_path),
            },
            "incumbent": {
                "path": str(incumbent_path),
                "sha256": _sha256(incumbent_path),
            },
            "exact_policy": {
                "path": str(exact_policy_path),
                "sha256": _sha256(exact_policy_path),
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
    parser.add_argument("--selection-path", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--incumbent-path", type=Path, default=DEFAULT_INCUMBENT)
    parser.add_argument("--exact-policy-path", type=Path, default=DEFAULT_EXACT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args(argv)
    summary = run(**vars(args))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "rows": summary["rows"],
                "selected_family_by_side": summary["selected_family_by_side"],
                "output": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
