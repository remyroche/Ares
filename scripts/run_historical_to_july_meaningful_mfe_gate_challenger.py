#!/usr/bin/env python3
"""Train historical meaningful-MFE gates and evaluate once on July 20-23.

All feature selection, model/parameter selection, calibration, admission
thresholds, and adverse-risk weights are selected from side-local temporal OOF
predictions whose labels resolve before 2026-07-20.  Current July outcomes are
loaded only after the frozen state has been built.  This is research-only
evidence and cannot promote a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_july_exact_preentry_heads import (  # noqa: E402
    IDENTITY,
    MIN_USABLE_MFE_ATR,
    MIN_USABLE_MFE_RETURN,
    prepare_joined,
    sha256,
    validate_manifest_hash,
)


SCHEMA = "historical_to_july_meaningful_mfe_gate_challenger_v1"
SIDES = ("long", "short")
HISTORY_CUTOFF = pd.Timestamp("2026-07-20T00:00:00Z")
RAW_PREFIX = "capture_candidate__"
FORBIDDEN_MODEL_INPUT_FRAGMENTS = (
    "execution_",
    "label",
    "target",
    "outcome",
    "mfe",
    "mae",
    "favorable",
    "adverse",
    "timeout",
)
OUTER_WINDOWS = (
    ("2026-06-01T00:00:00Z", "2026-06-15T00:00:00Z"),
    ("2026-06-15T00:00:00Z", "2026-07-01T00:00:00Z"),
    ("2026-07-01T00:00:00Z", "2026-07-20T00:00:00Z"),
)
FEATURE_COUNTS = (32, 64)
MODEL_GRIDS: Mapping[str, tuple[Mapping[str, Any], ...]] = {
    "logistic": ({"C": 0.10}, {"C": 1.0}),
    "lightgbm": (
        {"num_leaves": 15, "max_depth": 5, "min_child_samples": 250, "reg_lambda": 8.0},
        {"num_leaves": 31, "max_depth": 7, "min_child_samples": 150, "reg_lambda": 12.0},
    ),
    "catboost": (
        {"depth": 5, "l2_leaf_reg": 8.0},
        {"depth": 7, "l2_leaf_reg": 12.0},
    ),
}
ARM_SPECS: Mapping[str, Mapping[str, str]] = {
    "logistic_hard_meaningful": {
        "family": "logistic",
        "fit_target": "hard_meaningful",
        "calibration_target": "hard_meaningful",
    },
    "logistic_soft_triple_barrier": {
        "family": "logistic",
        "fit_target": "soft_triple_barrier",
        "calibration_target": "hard_meaningful",
    },
    "lightgbm_hard_meaningful": {
        "family": "lightgbm",
        "fit_target": "hard_meaningful",
        "calibration_target": "hard_meaningful",
    },
    "lightgbm_soft_triple_barrier": {
        "family": "lightgbm",
        "fit_target": "soft_triple_barrier",
        "calibration_target": "hard_meaningful",
    },
    "catboost_hard_meaningful": {
        "family": "catboost",
        "fit_target": "hard_meaningful",
        "calibration_target": "hard_meaningful",
    },
    "catboost_soft_triple_barrier": {
        "family": "catboost",
        "fit_target": "soft_triple_barrier",
        "calibration_target": "hard_meaningful",
    },
    "catboost_hard_clean_first": {
        "family": "catboost",
        "fit_target": "hard_clean_first",
        "calibration_target": "hard_meaningful",
    },
}
ADVERSE_SPEC: Mapping[str, str] = {
    "family": "catboost",
    "fit_target": "adverse_1atr_reached",
    "calibration_target": "adverse_1atr_reached",
}


@dataclass(frozen=True)
class TemporalFold:
    name: str
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    train: np.ndarray
    validation: np.ndarray


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _identity_set(frame: pd.DataFrame) -> set[tuple[Any, ...]]:
    return set(map(tuple, frame.loc[:, IDENTITY].itertuples(index=False, name=None)))


def _require_unique(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} identity columns missing: {missing}")
    work = frame.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    work["execution_decision_utc"] = pd.to_datetime(
        work["execution_decision_utc"], utc=True, errors="raise"
    )
    if work.duplicated(list(IDENTITY)).any():
        raise ValueError(f"{name} contains duplicate identities")
    return work


def historical_folds(frame: pd.DataFrame) -> list[TemporalFold]:
    signal = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    resolved = pd.to_datetime(frame["label_resolution_utc"], utc=True, errors="raise")
    folds: list[TemporalFold] = []
    for index, (start_value, end_value) in enumerate(OUTER_WINDOWS):
        start, end = pd.Timestamp(start_value), pd.Timestamp(end_value)
        train = np.flatnonzero((signal < start).to_numpy() & (resolved < start).to_numpy())
        validation = np.flatnonzero(
            (signal >= start).to_numpy() & (signal < end).to_numpy()
        )
        if len(train) < 5_000 or len(validation) < 1_000:
            raise ValueError(f"historical fold {index} has insufficient support")
        if not bool((resolved.iloc[train] < start).all()):
            raise AssertionError("training label resolution is not strictly before validation")
        folds.append(
            TemporalFold(
                name=f"fold_{index}",
                validation_start=start,
                validation_end=end,
                train=train,
                validation=validation,
            )
        )
    return folds


def load_historical(
    feature_path: Path,
    feature_manifest_path: Path,
    grid_path: Path,
    grid_manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if feature_manifest.get("schema") != "exact_policy_capture_causal_feature_universe_v1":
        raise ValueError("unexpected historical feature-universe schema")
    if sha256(feature_path) != feature_manifest["outputs"]["universe"]["sha256"]:
        raise ValueError("historical feature-universe hash mismatch")
    if feature_manifest.get("contract", {}).get("point_in_time") != (
        "immutable causal feature-store value at candidate signal __ts__"
    ):
        raise ValueError("historical features are not proven point-in-time")
    universe_manifest_path = Path(
        feature_manifest["outputs"]["feature_manifest"]["path"]
    )
    universe_manifest = json.loads(universe_manifest_path.read_text(encoding="utf-8"))
    eligible = list(map(str, universe_manifest["eligible_full_period_feature_columns"]))
    if not eligible or any(not feature.startswith(RAW_PREFIX) for feature in eligible):
        raise ValueError("historical eligible feature contract is malformed")
    forbidden = [
        feature
        for feature in eligible
        if any(fragment in feature.lower() for fragment in FORBIDDEN_MODEL_INPUT_FRAGMENTS)
    ]
    if forbidden:
        raise ValueError(f"outcome-like historical model features forbidden: {forbidden}")

    grid_manifest = json.loads(grid_manifest_path.read_text(encoding="utf-8"))
    if grid_manifest.get("schema") != "materialize_meaningful_mfe_label_grid_v1":
        raise ValueError("unexpected meaningful-MFE grid schema")
    if sha256(grid_path) != grid_manifest["outputs"]["labels"]["sha256"]:
        raise ValueError("meaningful-MFE label-grid hash mismatch")
    specs = {
        str(record["name"]): record for record in grid_manifest.get("specs", [])
    }
    expected = specs.get("h12_u1p5atr")
    if not expected or expected != {
        "horizon_hours": 12,
        "lower_atr": 1.0,
        "name": "h12_u1p5atr",
        "round_trip_cost": 0.01,
        "temperature": 0.35,
        "upper_atr": 1.5,
        "upper_return_floor": 0.015,
    }:
        raise ValueError("canonical h12_u1p5atr label geometry changed")

    history = _require_unique(pd.read_parquet(feature_path), "historical features")
    grid = pd.read_parquet(grid_path)
    grid = grid.loc[grid["grid_name"].astype(str).eq("h12_u1p5atr")].copy()
    grid = _require_unique(grid, "historical h12 grid")
    if _identity_set(history) != _identity_set(grid):
        raise ValueError("historical feature/grid identity join is incomplete")
    columns = [
        *IDENTITY,
        "label_resolution_utc",
        "soft_label",
        "favorable_first",
        "adverse_first",
        "timeout",
        "upper_return",
        "peak_mfe_atr",
    ]
    history = history.merge(
        grid.loc[:, columns],
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    history["label_resolution_utc"] = pd.to_datetime(
        history["label_resolution_utc"], utc=True, errors="raise"
    )
    history["execution_label_end_utc"] = pd.to_datetime(
        history["execution_label_end_utc"], utc=True, errors="raise"
    )
    history = history.loc[
        history["label_resolution_utc"].lt(HISTORY_CUTOFF)
        & history["execution_label_end_utc"].lt(HISTORY_CUTOFF)
    ].reset_index(drop=True)
    if history.empty or history["label_resolution_utc"].max() >= HISTORY_CUTOFF:
        raise ValueError("historical cutoff filter failed")
    atr = pd.to_numeric(history["oof_entry_atr_fraction"], errors="raise").to_numpy(float)
    mfe = pd.to_numeric(history["execution_mfe_return_12h"], errors="raise").to_numpy(float)
    mae = pd.to_numeric(history["execution_mae_return_12h"], errors="raise").to_numpy(float)
    threshold = np.maximum(MIN_USABLE_MFE_ATR * atr, MIN_USABLE_MFE_RETURN)
    history["hard_meaningful"] = (mfe >= threshold).astype(np.int8)
    history["hard_clean_first"] = history["favorable_first"].astype(np.int8)
    history["soft_triple_barrier"] = np.clip(
        pd.to_numeric(history["soft_label"], errors="raise"), 0.0, 1.0
    )
    history["adverse_1atr_reached"] = (mae >= atr).astype(np.int8)
    incumbent_p = np.clip(
        pd.to_numeric(history["oof_clean_favorable_probability"], errors="coerce").to_numpy(float),
        1e-8,
        1.0,
    )
    history["frozen_conditional_peak_atr"] = np.divide(
        pd.to_numeric(history["pred_peak_MFE_12h_ATR"], errors="coerce").to_numpy(float),
        incumbent_p,
    )
    feature_matrix = history.loc[:, eligible].copy()
    feature_matrix.columns = [column.removeprefix(RAW_PREFIX) for column in eligible]
    raw_features = list(feature_matrix.columns)
    history = pd.concat(
        [history.drop(columns=eligible), feature_matrix.astype(np.float32)], axis=1
    )
    if set(history["side_name"].astype(str)) != set(SIDES):
        raise ValueError("historical data must contain both sides")
    report = {
        "features": {
            "path": str(feature_path),
            "sha256": sha256(feature_path),
            "manifest_path": str(feature_manifest_path),
            "manifest_sha256": sha256(feature_manifest_path),
            "universe_manifest_path": str(universe_manifest_path),
            "universe_manifest_sha256": sha256(universe_manifest_path),
        },
        "labels": {
            "path": str(grid_path),
            "sha256": sha256(grid_path),
            "manifest_path": str(grid_manifest_path),
            "manifest_sha256": sha256(grid_manifest_path),
        },
        "rows_after_strict_cutoff": int(len(history)),
        "label_resolution_max": history["label_resolution_utc"].max(),
        "raw_features": len(raw_features),
    }
    return history, history.loc[:, raw_features].copy(), raw_features, report


def load_current_features(
    packb_path: Path,
    raw_features: Sequence[str],
) -> pd.DataFrame:
    packb = _require_unique(pd.read_parquet(packb_path), "current Pack-B")
    missing = sorted(set(raw_features).difference(packb.columns))
    if missing:
        raise ValueError(f"current Pack-B lacks historical raw features: {missing}")
    matrix = packb.loc[:, list(raw_features)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(matrix.to_numpy(float)).all():
        raise ValueError("current Pack-B raw challenger features are not finite")
    return packb.loc[:, list(IDENTITY)].join(matrix.astype(np.float32))


def _spearman(feature: pd.Series, target: np.ndarray) -> float:
    values = pd.to_numeric(feature, errors="coerce").to_numpy(float)
    finite = np.isfinite(values) & np.isfinite(target)
    if finite.sum() < 100 or np.unique(values[finite]).size < 2:
        return 0.0
    result = spearmanr(values[finite], target[finite]).statistic
    return float(result) if np.isfinite(result) else 0.0


def select_features_nested(
    matrix: pd.DataFrame,
    target: np.ndarray,
    positions: np.ndarray,
    count: int,
) -> tuple[list[str], pd.DataFrame]:
    train = matrix.iloc[positions]
    y = np.asarray(target, dtype=float)[positions]
    midpoint = len(train) // 2
    rows: list[dict[str, Any]] = []
    for feature in matrix.columns:
        values = pd.to_numeric(train[feature], errors="coerce")
        coverage = float(values.notna().mean())
        variance = float(values.var()) if coverage else 0.0
        early = _spearman(values.iloc[:midpoint], y[:midpoint])
        late = _spearman(values.iloc[midpoint:], y[midpoint:])
        full = _spearman(values, y)
        stable = (
            min(abs(early), abs(late))
            if early * late > 0.0
            else 0.10 * abs(full)
        )
        rows.append(
            {
                "feature": feature,
                "coverage": coverage,
                "variance": variance,
                "early_ic": early,
                "late_ic": late,
                "full_ic": full,
                "stable_score": stable,
            }
        )
    screen = pd.DataFrame(rows)
    screen = screen.loc[
        screen["coverage"].ge(0.99)
        & screen["variance"].gt(1e-12)
    ].sort_values(
        ["stable_score", "feature"], ascending=[False, True], kind="stable"
    )
    selected: list[str] = []
    for feature in screen["feature"]:
        if len(selected) >= int(count):
            break
        candidate = pd.to_numeric(train[feature], errors="coerce")
        if any(
            abs(candidate.corr(pd.to_numeric(train[prior], errors="coerce"))) >= 0.95
            for prior in selected
        ):
            continue
        selected.append(str(feature))
    if len(selected) < min(8, count):
        raise ValueError("nested feature selector returned insufficient features")
    screen["selected"] = screen["feature"].isin(selected)
    return selected, screen


def _balanced_weights(target: np.ndarray) -> np.ndarray:
    y = np.asarray(target, dtype=float)
    hard = (y >= 0.5).astype(int)
    positive = max(int(hard.sum()), 1)
    negative = max(int((hard == 0).sum()), 1)
    return np.where(
        hard == 1,
        len(hard) / (2.0 * positive),
        len(hard) / (2.0 * negative),
    )


def fit_model(
    family: str,
    params: Mapping[str, Any],
    matrix: pd.DataFrame,
    target: np.ndarray,
    *,
    soft: bool,
    seed: int,
) -> Any:
    y = np.asarray(target, dtype=float)
    if family == "logistic":
        pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=float(params["C"]),
                max_iter=300,
                solver="lbfgs",
                random_state=int(seed),
            ),
        )
        if soft:
            doubled = pd.concat([matrix, matrix], ignore_index=True)
            labels = np.r_[np.ones(len(y)), np.zeros(len(y))]
            weights = np.r_[y, 1.0 - y]
            keep = weights > 1e-8
            pipeline.fit(
                doubled.iloc[np.flatnonzero(keep)],
                labels[keep],
                logisticregression__sample_weight=weights[keep],
            )
        else:
            pipeline.fit(
                matrix,
                y.astype(int),
                logisticregression__sample_weight=_balanced_weights(y),
            )
        return pipeline
    if family == "lightgbm":
        common = dict(
            n_estimators=240,
            learning_rate=0.04,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            n_jobs=4,
            verbosity=-1,
            random_state=int(seed),
            **dict(params),
        )
        if soft:
            model = lgb.LGBMRegressor(objective="cross_entropy", **common)
            model.fit(matrix, y)
        else:
            model = lgb.LGBMClassifier(objective="binary", **common)
            model.fit(matrix, y.astype(int), sample_weight=_balanced_weights(y))
        return model
    if family == "catboost":
        model = CatBoostClassifier(
            loss_function="CrossEntropy" if soft else "Logloss",
            iterations=240,
            learning_rate=0.04,
            random_seed=int(seed),
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
            **dict(params),
        )
        model.fit(
            matrix,
            y if soft else y.astype(int),
            sample_weight=None if soft else _balanced_weights(y),
        )
        return model
    raise ValueError(f"unknown model family: {family}")


def predict_model(model: Any, family: str, matrix: pd.DataFrame) -> np.ndarray:
    prediction = (
        model.predict_proba(matrix)[:, 1]
        if hasattr(model, "predict_proba")
        else model.predict(matrix)
    )
    return np.clip(np.asarray(prediction, dtype=float), 1e-6, 1.0 - 1e-6)


def classification_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.clip(np.asarray(prediction, dtype=float), 1e-6, 1.0 - 1e-6)
    result = {
        "rows": int(len(y)),
        "prevalence": float(y.mean()),
        "mean_probability": float(p.mean()),
        "auc": float("nan"),
        "pr_auc": float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10": float("nan"),
    }
    bins = np.minimum((p * 10).astype(int), 9)
    result["ece_10"] = float(
        sum(
            (bins == index).mean()
            * abs(float(p[bins == index].mean()) - float(y[bins == index].mean()))
            for index in np.unique(bins)
        )
    )
    if np.unique(y).size == 2:
        result["auc"] = float(roc_auc_score(y, p))
        result["pr_auc"] = float(average_precision_score(y, p))
    return result


def _trial_objective(metrics: Mapping[str, float]) -> float:
    return float(
        metrics["log_loss"]
        + metrics["brier"]
        - 0.10 * metrics["auc"]
        - 0.05 * metrics["pr_auc"]
    )


def run_oof_trial(
    local: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    family: str,
    fit_target: str,
    calibration_target: str,
    params: Mapping[str, Any],
    feature_count: int,
    seed: int,
    selection_cache: dict[tuple[str, int, str], tuple[list[str], pd.DataFrame]]
    | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    folds = historical_folds(local)
    prediction = np.full(len(local), np.nan)
    reports: list[dict[str, Any]] = []
    fit_values = local[fit_target].to_numpy(float)
    calibration_values = local[calibration_target].to_numpy(int)
    for fold_index, fold in enumerate(folds):
        cache_key = (fit_target, int(feature_count), fold.name)
        cached = selection_cache.get(cache_key) if selection_cache is not None else None
        if cached is None:
            features, screen = select_features_nested(
                matrix, fit_values, fold.train, feature_count
            )
            if selection_cache is not None:
                selection_cache[cache_key] = (features, screen)
        else:
            features, screen = cached
        model = fit_model(
            family,
            params,
            matrix.iloc[fold.train].loc[:, features],
            fit_values[fold.train],
            soft=fit_target == "soft_triple_barrier",
            seed=seed + fold_index,
        )
        fold_prediction = predict_model(
            model, family, matrix.iloc[fold.validation].loc[:, features]
        )
        prediction[fold.validation] = fold_prediction
        reports.append(
            {
                "fold": fold.name,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "train_rows": int(len(fold.train)),
                "validation_rows": int(len(fold.validation)),
                "training_label_resolution_max": local.iloc[fold.train][
                    "label_resolution_utc"
                ].max(),
                "selected_features": features,
                "selected_feature_count": len(features),
                "screen_top": screen.head(20).to_dict("records"),
                "metrics": classification_metrics(
                    calibration_values[fold.validation], fold_prediction
                ),
            }
        )
    return prediction, reports


def fit_calibrator(
    method: str, prediction: np.ndarray, target: np.ndarray
) -> Any:
    p = np.clip(np.asarray(prediction, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(target, dtype=int)
    if method == "identity":
        return None
    if method == "sigmoid":
        model = LogisticRegression(C=1.0, max_iter=300)
        model.fit(np.log(p / (1.0 - p)).reshape(-1, 1), y)
        return model
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(p, y)
        return model
    raise ValueError(f"unknown calibration method: {method}")


def apply_calibrator(method: str, model: Any, prediction: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prediction, dtype=float), 1e-6, 1.0 - 1e-6)
    if method == "identity":
        return p
    if method == "sigmoid":
        return model.predict_proba(np.log(p / (1.0 - p)).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return np.asarray(model.predict(p), dtype=float)
    raise ValueError(f"unknown calibration method: {method}")


def select_calibration(
    frame: pd.DataFrame,
    raw_prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[str, Any, list[dict[str, Any]], np.ndarray]:
    finite = np.isfinite(raw_prediction)
    indices = np.flatnonzero(finite)
    ordering = indices[
        np.argsort(
            pd.to_datetime(
                frame.iloc[indices]["__ts__"], utc=True, errors="raise"
            ).astype("int64").to_numpy(),
            kind="stable",
        )
    ]
    split = max(1, int(len(ordering) * 0.67))
    fit_indices, validation_indices = ordering[:split], ordering[split:]
    if len(validation_indices) < 500:
        raise ValueError("calibration validation support is insufficient")
    trials: list[dict[str, Any]] = []
    for method in ("identity", "sigmoid", "isotonic"):
        model = fit_calibrator(method, raw_prediction[fit_indices], target[fit_indices])
        calibrated = apply_calibrator(method, model, raw_prediction[validation_indices])
        metrics = classification_metrics(target[validation_indices], calibrated)
        trials.append({"method": method, **metrics})
    winner = min(trials, key=lambda row: (row["brier"], row["log_loss"], row["method"]))
    method = str(winner["method"])
    final_model = fit_calibrator(method, raw_prediction[indices], target[indices])
    calibrated_oof = np.full(len(frame), np.nan)
    calibrated_oof[indices] = apply_calibrator(
        method, final_model, raw_prediction[indices]
    )
    return method, final_model, trials, calibrated_oof


def _top10_net(frame: pd.DataFrame, score: np.ndarray) -> float:
    valid = np.isfinite(score)
    local = frame.loc[valid].copy()
    local["_score"] = np.asarray(score)[valid]
    count = max(1, int(math.ceil(len(local) * 0.10)))
    selected = local.sort_values(
        ["_score", "candidate_id"], ascending=[False, True], kind="stable"
    ).iloc[:count]
    return float(selected["execution_net_ev_12h"].mean() * 1e4)


def select_admission_threshold(
    frame: pd.DataFrame, prediction: np.ndarray
) -> dict[str, Any]:
    finite = np.isfinite(prediction)
    local = frame.loc[finite].copy()
    local["_prediction"] = np.asarray(prediction)[finite]
    trials: list[dict[str, Any]] = []
    for quantile in (0.50, 0.60, 0.70, 0.80, 0.90):
        threshold = float(local["_prediction"].quantile(quantile))
        selected = local.loc[local["_prediction"].ge(threshold)]
        trials.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                "rows": int(len(selected)),
                "coverage": float(len(selected) / len(local)),
                "net_ev_bps": float(selected["execution_net_ev_12h"].mean() * 1e4),
                "positive_net_precision": float(
                    (selected["execution_net_ev_12h"] > 0.0).mean()
                ),
            }
        )
    winner = max(
        trials,
        key=lambda row: (
            row["net_ev_bps"],
            row["positive_net_precision"],
            -row["coverage"],
        ),
    )
    return {"winner": winner, "trials": trials}


def freeze_historical_state(
    history: pd.DataFrame,
    matrix: pd.DataFrame,
    raw_features: Sequence[str],
    output_dir: Path,
    *,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    states: dict[str, Any] = {}
    selection_caches: dict[
        str, dict[tuple[str, int, str], tuple[list[str], pd.DataFrame]]
    ] = {side: {} for side in SIDES}
    oof = history.loc[
        :,
        [
            *IDENTITY,
            "execution_net_ev_12h",
            "hard_meaningful",
            "hard_clean_first",
            "adverse_1atr_reached",
            "frozen_conditional_peak_atr",
        ],
    ].copy()
    fit_specs = {**ARM_SPECS, "catboost_adverse_1atr_gate": ADVERSE_SPEC}
    for arm_index, (arm, spec) in enumerate(fit_specs.items()):
        states[arm] = {}
        oof[f"{arm}__probability"] = np.nan
        for side_index, side in enumerate(SIDES):
            positions = np.flatnonzero(history["side_name"].astype(str).eq(side).to_numpy())
            local = history.iloc[positions].reset_index(drop=True)
            local_matrix = matrix.iloc[positions].reset_index(drop=True)
            trials: list[dict[str, Any]] = []
            trial_predictions: list[np.ndarray] = []
            for feature_count in FEATURE_COUNTS:
                for param_index, params in enumerate(MODEL_GRIDS[spec["family"]]):
                    prediction, fold_reports = run_oof_trial(
                        local,
                        local_matrix,
                        family=spec["family"],
                        fit_target=spec["fit_target"],
                        calibration_target=spec["calibration_target"],
                        params=params,
                        feature_count=feature_count,
                        seed=seed
                        + 100_000 * arm_index
                        + 10_000 * side_index
                        + 100 * feature_count
                        + param_index,
                        selection_cache=selection_caches[side],
                    )
                    finite = np.isfinite(prediction)
                    metrics = classification_metrics(
                        local.loc[finite, spec["calibration_target"]].to_numpy(int),
                        prediction[finite],
                    )
                    trials.append(
                        {
                            "feature_count": feature_count,
                            "params": dict(params),
                            "objective": _trial_objective(metrics),
                            "metrics": metrics,
                            "folds": fold_reports,
                        }
                    )
                    trial_predictions.append(prediction)
            winner_index = min(
                range(len(trials)),
                key=lambda index: (
                    trials[index]["objective"],
                    trials[index]["feature_count"],
                    str(trials[index]["params"]),
                ),
            )
            winner = trials[winner_index]
            raw_oof = trial_predictions[winner_index]
            method, calibrator, calibration_trials, calibrated_oof = select_calibration(
                local,
                raw_oof,
                local[spec["calibration_target"]].to_numpy(int),
            )
            fit_target = local[spec["fit_target"]].to_numpy(float)
            final_features, final_screen = select_features_nested(
                local_matrix,
                fit_target,
                np.arange(len(local)),
                int(winner["feature_count"]),
            )
            final_model = fit_model(
                spec["family"],
                winner["params"],
                local_matrix.loc[:, final_features],
                fit_target,
                soft=spec["fit_target"] == "soft_triple_barrier",
                seed=seed + 1_000_000 + arm_index * 100 + side_index,
            )
            admission = (
                select_admission_threshold(local, calibrated_oof)
                if arm != "catboost_adverse_1atr_gate"
                else None
            )
            model_path = output_dir / "models" / f"{arm}__{side}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "schema": SCHEMA,
                    "arm": arm,
                    "side": side,
                    "family": spec["family"],
                    "fit_target": spec["fit_target"],
                    "calibration_target": spec["calibration_target"],
                    "features": final_features,
                    "model": final_model,
                    "calibration_method": method,
                    "calibrator": calibrator,
                    "history_cutoff": HISTORY_CUTOFF,
                },
                model_path,
            )
            global_positions = positions[np.flatnonzero(np.isfinite(calibrated_oof))]
            oof.loc[global_positions, f"{arm}__probability"] = calibrated_oof[
                np.isfinite(calibrated_oof)
            ]
            states[arm][side] = {
                "family": spec["family"],
                "fit_target": spec["fit_target"],
                "calibration_target": spec["calibration_target"],
                "winner": winner,
                "calibration_method": method,
                "calibration_trials": calibration_trials,
                "calibrated_oof_metrics": classification_metrics(
                    local.loc[
                        np.isfinite(calibrated_oof), spec["calibration_target"]
                    ].to_numpy(int),
                    calibrated_oof[np.isfinite(calibrated_oof)],
                ),
                "final_features": final_features,
                "final_feature_screen_top": final_screen.head(30).to_dict("records"),
                "model": {
                    "path": str(model_path),
                    "sha256": sha256(model_path),
                },
                "admission": admission,
            }
            print(
                f"frozen arm={arm} side={side} family={spec['family']} "
                f"features={len(final_features)} calibration={method}",
                flush=True,
            )

    adverse = oof["catboost_adverse_1atr_gate__probability"].to_numpy(float)
    risk_weight_trials = []
    incidence_arms = list(ARM_SPECS)
    for arm in incidence_arms:
        probability = oof[f"{arm}__probability"].to_numpy(float)
        magnitude = pd.to_numeric(
            oof["frozen_conditional_peak_atr"], errors="coerce"
        ).to_numpy(float)
        for weight in (0.0, 0.5, 1.0, 2.0):
            score = probability * magnitude * np.power(
                np.clip(1.0 - adverse, 1e-3, 1.0), weight
            )
            risk_weight_trials.append(
                {
                    "arm": arm,
                    "adverse_weight": weight,
                    "historical_oof_top10_net_bps": _top10_net(oof, score),
                }
            )
    risk_winner = max(
        risk_weight_trials,
        key=lambda row: (
            row["historical_oof_top10_net_bps"],
            -row["adverse_weight"],
            row["arm"],
        ),
    )
    classification_rows = []
    for arm in incidence_arms:
        for side in SIDES:
            local = oof.loc[oof["side_name"].eq(side)]
            prediction = local[f"{arm}__probability"].to_numpy(float)
            finite = np.isfinite(prediction)
            classification_rows.append(
                {
                    "arm": arm,
                    "side_name": side,
                    **classification_metrics(
                        local.loc[finite, "hard_meaningful"].to_numpy(int),
                        prediction[finite],
                    ),
                    "historical_oof_top10_net_bps": _top10_net(
                        local, prediction
                    ),
                }
            )
    summary = {
        "schema": SCHEMA,
        "selection_status": "frozen_before_current_july_outcomes_loaded",
        "history_cutoff": HISTORY_CUTOFF,
        "rows": int(len(history)),
        "label_resolution_max": history["label_resolution_utc"].max(),
        "feature_count": len(raw_features),
        "arms": states,
        "risk_weight_trials": risk_weight_trials,
        "risk_winner": risk_winner,
        "historical_oof_classification": classification_rows,
    }
    return summary, oof


def score_frozen_models(
    state: Mapping[str, Any],
    current_features: pd.DataFrame,
) -> pd.DataFrame:
    output = current_features.loc[:, list(IDENTITY)].copy()
    for arm in (*ARM_SPECS, "catboost_adverse_1atr_gate"):
        output[f"{arm}__probability"] = np.nan
        for side in SIDES:
            record = state["arms"][arm][side]["model"]
            path = Path(record["path"])
            if sha256(path) != record["sha256"]:
                raise ValueError(f"frozen challenger model hash mismatch: {arm}/{side}")
            bundle = joblib.load(path)
            positions = np.flatnonzero(output["side_name"].astype(str).eq(side).to_numpy())
            matrix = current_features.iloc[positions].loc[:, bundle["features"]]
            raw = predict_model(bundle["model"], bundle["family"], matrix)
            calibrated = apply_calibrator(
                bundle["calibration_method"], bundle["calibrator"], raw
            )
            output.loc[positions, f"{arm}__probability"] = calibrated
    probability_columns = [
        column for column in output if column.endswith("__probability")
    ]
    if not np.isfinite(output[probability_columns].to_numpy(float)).all():
        raise ValueError("frozen challenger produced non-finite current probabilities")
    return output


def _evaluate_selection(
    frame: pd.DataFrame,
    *,
    arm: str,
    score_column: str,
    scope: str,
    admitted_column: str | None = None,
) -> dict[str, Any]:
    local = frame if scope == "pooled" else frame.loc[frame["side_name"].eq(scope)]
    if admitted_column:
        local = local.loc[local[admitted_column]]
    if local.empty:
        return {
            "arm": arm,
            "score_column": score_column,
            "scope": scope,
            "rows": 0,
            "top10_rows": 0,
        }
    ordered = local.sort_values(
        [score_column, "candidate_id"], ascending=[False, True], kind="stable"
    )
    count = max(1, int(math.ceil(len(ordered) * 0.10)))
    top = ordered.iloc[:count]
    y = local["meaningful_mfe_reached"].to_numpy(int)
    p = np.clip(local[score_column].rank(pct=True).to_numpy(float), 0.0, 1.0)
    return {
        "arm": arm,
        "score_column": score_column,
        "scope": scope,
        "rows": int(len(local)),
        "top10_rows": int(count),
        "top10_net_ev_bps": float(top["execution_net_ev_12h"].mean() * 1e4),
        "top10_positive_net_precision": float(
            (top["execution_net_ev_12h"] > 0.0).mean()
        ),
        "top10_meaningful_mfe_precision": float(top["meaningful_mfe_reached"].mean()),
        "top10_adverse_1atr_fraction": float(top["adverse_1atr_reached"].mean()),
        "rank_ic_net": float(
            spearmanr(local[score_column], local["execution_net_ev_12h"]).statistic
        ),
        "auc_meaningful_rank_score": float(roc_auc_score(y, p)),
    }


def evaluate_current(
    current: pd.DataFrame,
    state: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    score_columns: dict[str, str] = {
        "baseline_frozen_clean_probability": "oof_clean_favorable_probability",
        "baseline_frozen_unconditional_peak": "pred_peak_MFE_12h_ATR",
        "direct_execution_ev_comparator": "final_direct_net_raw",
        "mapped_execution_ev_comparator": "mapped_execution_ev",
    }
    adverse = current["catboost_adverse_1atr_gate__probability"].to_numpy(float)
    for arm in ARM_SPECS:
        probability_column = f"{arm}__probability"
        score_columns[f"{arm}__incidence_probability"] = probability_column
        unconditional_column = f"{arm}__unconditional_peak"
        current[unconditional_column] = (
            current[probability_column] * current["pred_peak_mfe_if_hit_atr"]
        )
        score_columns[f"{arm}__probability_x_conditional_peak"] = unconditional_column
        weight = next(
            row["adverse_weight"]
            for row in state["risk_weight_trials"]
            if row["arm"] == arm
            and row["historical_oof_top10_net_bps"]
            == max(
                candidate["historical_oof_top10_net_bps"]
                for candidate in state["risk_weight_trials"]
                if candidate["arm"] == arm
            )
        )
        risk_column = f"{arm}__risk_adjusted_peak"
        current[risk_column] = current[unconditional_column] * np.power(
            np.clip(1.0 - adverse, 1e-3, 1.0), float(weight)
        )
        score_columns[f"{arm}__risk_adjusted_probability_x_peak"] = risk_column
        admission_column = f"{arm}__historical_admission"
        current[admission_column] = False
        for side in SIDES:
            threshold = state["arms"][arm][side]["admission"]["winner"]["threshold"]
            mask = current["side_name"].eq(side)
            current.loc[mask, admission_column] = current.loc[
                mask, probability_column
            ].ge(threshold)

    probability_rows: list[dict[str, Any]] = []
    for arm in ("baseline_frozen_clean_probability", *ARM_SPECS):
        column = (
            "oof_clean_favorable_probability"
            if arm == "baseline_frozen_clean_probability"
            else f"{arm}__probability"
        )
        for scope in ("pooled", *SIDES):
            local = current if scope == "pooled" else current.loc[current["side_name"].eq(scope)]
            metrics = classification_metrics(
                local["meaningful_mfe_reached"].to_numpy(int),
                local[column].to_numpy(float),
            )
            probability_rows.append(
                {"arm": arm, "scope": scope, **metrics}
            )
    adverse_rows: list[dict[str, Any]] = []
    for scope in ("pooled", *SIDES):
        local = current if scope == "pooled" else current.loc[current["side_name"].eq(scope)]
        adverse_rows.append(
            {
                "scope": scope,
                **classification_metrics(
                    local["adverse_1atr_reached"].to_numpy(int),
                    local["catboost_adverse_1atr_gate__probability"].to_numpy(float),
                ),
            }
        )

    economics_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    for arm, column in score_columns.items():
        for scope in ("pooled", *SIDES):
            economics_rows.append(
                _evaluate_selection(current, arm=arm, score_column=column, scope=scope)
            )
        for day, group in current.groupby("audit_day_utc", sort=True):
            daily_rows.append(
                {
                    "day_utc": day,
                    **_evaluate_selection(
                        group, arm=arm, score_column=column, scope="pooled"
                    ),
                }
            )
    admission_rows: list[dict[str, Any]] = []
    for arm in ARM_SPECS:
        admitted_column = f"{arm}__historical_admission"
        for scope in ("pooled", *SIDES):
            local = current if scope == "pooled" else current.loc[current["side_name"].eq(scope)]
            admitted = local.loc[local[admitted_column]]
            row = _evaluate_selection(
                current,
                arm=f"{arm}__direct_incidence_admission",
                score_column=f"{arm}__probability",
                scope=scope,
                admitted_column=admitted_column,
            )
            row.update(
                {
                    "admitted_rows": int(len(admitted)),
                    "admitted_coverage": float(len(admitted) / len(local)),
                    "admitted_net_ev_bps": (
                        float(admitted["execution_net_ev_12h"].mean() * 1e4)
                        if len(admitted)
                        else np.nan
                    ),
                    "admitted_positive_net_precision": (
                        float((admitted["execution_net_ev_12h"] > 0.0).mean())
                        if len(admitted)
                        else np.nan
                    ),
                }
            )
            admission_rows.append(row)
    return {
        "probability_metrics": pd.DataFrame(probability_rows),
        "adverse_metrics": pd.DataFrame(adverse_rows),
        "economics": pd.DataFrame(economics_rows),
        "daily_economics": pd.DataFrame(daily_rows),
        "admission_economics": pd.DataFrame(admission_rows),
        "current_predictions": current,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    history, historical_matrix, raw_features, history_lineage = load_historical(
        args.historical_features,
        args.historical_feature_manifest,
        args.historical_grid,
        args.historical_grid_manifest,
    )
    # This state is fully frozen before current labels or outcomes are loaded.
    state, historical_oof = freeze_historical_state(
        history,
        historical_matrix,
        raw_features,
        args.output_dir,
        seed=args.seed,
    )
    state["history_lineage"] = history_lineage
    frozen_state_path = args.output_dir / "frozen_before_current_evaluation.json"
    _write_json(frozen_state_path, state)
    frozen_state_sha = sha256(frozen_state_path)

    current_bindings = {
        "packb": validate_manifest_hash(
            args.current_packb,
            args.current_packb_manifest,
            ("output", "sha256"),
            expected_schema="packb_final_refits_forward_v1",
        ),
        "preentry": validate_manifest_hash(
            args.current_preentry,
            args.current_preentry_manifest,
            ("output", "sha256"),
            expected_schema="execution_ev_forward_preentry_v1",
        ),
        "scored": validate_manifest_hash(
            args.current_scored,
            args.current_scored_manifest,
            ("outputs", "scored_population", "sha256"),
            expected_schema="execution_ev_retrospective_scored_population_v1",
        ),
        "labels": validate_manifest_hash(
            args.current_labels,
            args.current_labels_manifest,
            ("output", "sha256"),
            expected_schema="execution_ev_deployed_policy_1m_labels_v1",
        ),
        "geometry": validate_manifest_hash(
            args.current_geometry,
            args.current_geometry_manifest,
            ("outputs", "path_targets", "sha256"),
            expected_schema="execution_ev_retrospective_causal_geometry_v1",
        ),
    }
    packb = pd.read_parquet(args.current_packb)
    current_features = load_current_features(args.current_packb, raw_features)
    current_scores = score_frozen_models(state, current_features)
    current_exact = prepare_joined(
        packb,
        pd.read_parquet(args.current_preentry),
        pd.read_parquet(args.current_scored),
        pd.read_parquet(args.current_labels),
        pd.read_parquet(args.current_geometry),
    )
    current = current_exact.merge(
        current_scores, on=list(IDENTITY), how="inner", validate="one_to_one"
    )
    if len(current) != len(current_exact):
        raise ValueError("current challenger score join is incomplete")
    current["adverse_1atr_reached"] = (
        current["execution_mae_return_12h"]
        >= current["__path_auxiliary_atr_fraction__"]
    ).astype(np.int8)
    results = evaluate_current(current, state)
    historical_oof_path = args.output_dir / "historical_oof_predictions.parquet"
    historical_oof.to_parquet(historical_oof_path, index=False)
    outputs: dict[str, Any] = {
        "historical_oof_predictions": {
            "path": str(historical_oof_path),
            "rows": int(len(historical_oof)),
            "sha256": sha256(historical_oof_path),
        }
    }
    for name, table in results.items():
        extension = "parquet" if name == "current_predictions" else "csv"
        path = args.output_dir / f"{name}.{extension}"
        if extension == "parquet":
            table.to_parquet(path, index=False)
        else:
            table.to_csv(path, index=False)
        outputs[name] = {
            "path": str(path),
            "rows": int(len(table)),
            "sha256": sha256(path),
        }
    report = {
        "schema": SCHEMA,
        "status": "completed_research_only_untouched_current_evaluation",
        "promotion_eligible": False,
        "chronology": {
            "history_cutoff_exclusive": HISTORY_CUTOFF,
            "historical_label_resolution_max": history["label_resolution_utc"].max(),
            "selection": (
                "nested side-local historical temporal OOF only; final selection, "
                "calibration, admission thresholds and risk weights persisted before "
                "current outcomes were loaded"
            ),
            "current_evaluation": "one evaluation on exact July20-23 12h labels",
        },
        "target_contract": {
            "hard_meaningful": "exact 1m MFE >= max(1.5 * decision ATR, 1.5%)",
            "soft_triple_barrier": (
                "ATR-normalized h12_u1p5atr soft label; upper=max(1.5 ATR,1.5%), "
                "lower=1 ATR, same-hour conflict adverse, temperature .35"
            ),
            "hard_clean_first": "hourly favorable upper barrier before 1 ATR adverse barrier",
            "adverse_risk": "exact whole-horizon MAE >= 1 decision-time ATR",
            "current_clean_first_limit": (
                "current exact artifact lacks first-touch ordering; no clean-first "
                "class accuracy is claimed"
            ),
        },
        "frozen_state": {
            "path": str(frozen_state_path),
            "sha256_before_current_outcomes_loaded": frozen_state_sha,
        },
        "history_lineage": history_lineage,
        "current_lineage": current_bindings,
        "outputs": outputs,
    }
    report_path = args.output_dir / "report.json"
    _write_json(report_path, report)
    manifest = {
        "schema": SCHEMA,
        "status": report["status"],
        "promotion_eligible": False,
        "models_refit_on_current": False,
        "current_outcomes_used_for_selection": False,
        "frozen_state_sha256": frozen_state_sha,
        "report": {"path": str(report_path), "sha256": sha256(report_path)},
        "outputs": outputs,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return report


def parser() -> argparse.ArgumentParser:
    current = Path(
        "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2"
    )
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--historical-features",
        type=Path,
        default=Path(
            "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/"
            "capture_feature_universe.parquet"
        ),
    )
    value.add_argument(
        "--historical-feature-manifest",
        type=Path,
        default=Path(
            "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/"
            "manifest.json"
        ),
    )
    value.add_argument(
        "--historical-grid",
        type=Path,
        default=Path(
            "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/"
            "meaningful_mfe_label_grid.parquet"
        ),
    )
    value.add_argument(
        "--historical-grid-manifest",
        type=Path,
        default=Path(
            "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/"
            "manifest.json"
        ),
    )
    value.add_argument("--current-packb", type=Path, default=current / "packb/packb_forward_context.parquet")
    value.add_argument("--current-packb-manifest", type=Path, default=current / "packb/manifest.json")
    value.add_argument("--current-preentry", type=Path, default=current / "preentry/preentry.parquet")
    value.add_argument("--current-preentry-manifest", type=Path, default=current / "preentry/manifest.json")
    value.add_argument("--current-scored", type=Path, default=current / "scored/scored_population.parquet")
    value.add_argument("--current-scored-manifest", type=Path, default=current / "scored/manifest.json")
    value.add_argument("--current-labels", type=Path, default=current / "labels_12h/execution_ev_policy_labels.parquet")
    value.add_argument("--current-labels-manifest", type=Path, default=current / "labels_12h/manifest.json")
    value.add_argument("--current-geometry", type=Path, default=current / "geometry/path_targets.parquet")
    value.add_argument("--current-geometry-manifest", type=Path, default=current / "geometry/manifest.json")
    value.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data_perp/artifacts/historical_to_july_meaningful_mfe_gate_challenger_"
            "20260730_v2"
        ),
    )
    value.add_argument("--seed", type=int, default=20260730)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
