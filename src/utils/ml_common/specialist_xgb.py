from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from src.utils.purged_kfold import PurgedKFoldTime
from src.utils.tprint import tprint_info

DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "learning_rate": 0.05,
    "max_depth": 4,
    "n_estimators": 300,
    "subsample": 0.6,
    "colsample_bytree": 0.7,
    "gamma": 4,
    "min_child_weight": 20,
    "reg_alpha": 4.0,
    "reg_lambda": 1.25,
    "tree_method": "hist",
    "max_delta_step": 5.0,
    "early_stopping_rounds": 40,
    "n_jobs": -1,
}


@dataclass
class XGBTrainingResult:
    model: xgb.XGBClassifier
    params: Dict[str, Any]
    oof_predictions: pd.Series
    metrics: Dict[str, float]


def _compute_scale_pos_weight(y: pd.Series) -> float:
    y_clean = pd.Series(y).dropna()
    pos = float((y_clean == 1).sum())
    neg = float((y_clean == 0).sum())
    if pos <= 0 or neg <= 0:
        return 1.0
    return max(1.0, neg / pos)


def _build_params(y: pd.Series, params_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = DEFAULT_XGB_PARAMS.copy()
    params["scale_pos_weight"] = _compute_scale_pos_weight(y)
    if params_override:
        params.update(params_override)
    return params


def _determine_monotonic_constraints(X: pd.DataFrame, y: pd.Series, threshold: float = 0.03) -> Tuple[Optional[Dict[str, int]], List[str]]:
    tprint_info("   [XGB] Determining Monotonic Constraints via Ridge...")
    model = Ridge(alpha=1.0)
    X_clean = X.fillna(0.0)
    model.fit(X_clean, y)

    constraints = {}
    strong_features = []

    for feat, coef in zip(X.columns, model.coef_):
        if abs(coef) > threshold:
            strong_features.append(feat)
            constraints[feat] = 1 if coef > 0 else -1
        else:
            constraints[feat] = 0

    tprint_info(f"   [XGB] Found {len(strong_features)} strong monotonic features.")
    return constraints, strong_features

def _fit_single_model(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    eval_set: Optional[list[tuple[pd.DataFrame, pd.Series]]] = None,
    params_override: Optional[Dict[str, Any]] = None,
    monotonic_constraints: Optional[Dict[str, int]] = None
) -> xgb.XGBClassifier:

    params = _build_params(y, params_override)
    early_stopping_rounds = params.pop("early_stopping_rounds", 40)

    if monotonic_constraints:
        params["monotone_constraints"] = tuple(monotonic_constraints.get(c, 0) for c in X.columns)

    model = xgb.XGBClassifier(**params)

    fit_kwargs: Dict[str, Any] = {"verbose": False}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    if eval_set:
        fit_kwargs["eval_set"] = eval_set
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

    model.fit(X, y, **fit_kwargs)
    return model


PURGE_MINUTES = 45
EMBARGO_MINUTES = 15


def train_specialist_xgb_with_oof(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    n_splits: int = 5,
    params_override: Optional[Dict[str, Any]] = None,
    apply_monotonic_constraints: bool = True
) -> XGBTrainingResult:
    tprint_info(f"   [XGB] Training with OOF (Splits: {n_splits})...")
    X = X.copy()
    y = y.astype(float).copy()

    if sample_weight is not None:
        sample_weight = sample_weight.astype(float)

    mono_constraints = None
    if apply_monotonic_constraints:
        mono_constraints, strong_feats = _determine_monotonic_constraints(X, y)

    splitter = PurgedKFoldTime(
        n_splits=n_splits,
        purge=pd.Timedelta(minutes=PURGE_MINUTES),
        embargo=pd.Timedelta(minutes=EMBARGO_MINUTES),
    )
    oof_probs = pd.Series(np.nan, index=X.index, dtype=float)
    last_model: Optional[xgb.XGBClassifier] = None

    for train_idx, val_idx in splitter.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train = sample_weight.iloc[train_idx] if sample_weight is not None else None

        model = _fit_single_model(
            X_train.fillna(0.0),
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val.fillna(0.0), y_val)],
            params_override=params_override,
            monotonic_constraints=mono_constraints
        )
        fold_probs = model.predict_proba(X_val.fillna(0.0))[:, 1]
        oof_probs.iloc[val_idx] = fold_probs
        last_model = model

    tprint_info("   [XGB] Training Final Model on Full Data...")
    final_model = _fit_single_model(
        X.fillna(0.0),
        y,
        sample_weight=sample_weight,
        eval_set=None,
        params_override=params_override,
        monotonic_constraints=mono_constraints
    )

    metrics: Dict[str, float] = {
        "n_features": float(X.shape[1]),
        "n_samples": float(len(X)),
    }

    valid_mask = oof_probs.notna()
    if valid_mask.sum() > 10:
        y_valid = y.loc[valid_mask]
        probs_valid = oof_probs.loc[valid_mask]
        try:
            metrics["auc"] = float(roc_auc_score(y_valid, probs_valid))
        except Exception:
            metrics["auc"] = 0.5
        try:
            metrics["aucpr"] = float(average_precision_score(y_valid, probs_valid))
        except Exception:
            metrics["aucpr"] = 0.0
        try:
            mi_score = mutual_info_regression(
                probs_valid.values.reshape(-1, 1),
                y_valid.values,
            )[0]
            metrics["mi_score"] = float(mi_score)
        except Exception:
            metrics["mi_score"] = 0.0
    else:
        metrics["auc"] = 0.5
        metrics["aucpr"] = 0.0
        metrics["mi_score"] = 0.0

    params_snapshot = getattr(last_model, "get_params", lambda: DEFAULT_XGB_PARAMS.copy())()

    return XGBTrainingResult(
        model=final_model,
        params=params_snapshot,
        oof_predictions=oof_probs,
        metrics=metrics,
    )
