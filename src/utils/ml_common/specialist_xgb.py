from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from src.utils.purged_kfold import PurgedKFoldTime
from src.utils.tprint import tprint_info

@dataclass
class SpecialistTrainingResult:
    model: ExtraTreesClassifier
    params: Dict[str, Any]
    oof_predictions: pd.Series
    metrics: Dict[str, float]


def _fit_single_model(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    params_override: Optional[Dict[str, Any]] = None,
) -> Optional[ExtraTreesClassifier]:

    if len(np.unique(y)) < 2:
        tprint_info(f"   [ExtraTrees] Single class detected in training data ({np.unique(y)}). Skipping fit.")
        return None

    # User-specified parameters for ExtraTrees
    n_features = X.shape[1]
    max_features = int(np.log2(n_features)) if n_features > 1 else 1
    
    import multiprocessing
    n_cpus = multiprocessing.cpu_count()
    default_n_jobs = min(n_cpus, 4) if n_cpus > 4 else max(1, n_cpus - 1)

    params = {
        "n_estimators": 1000,
        "max_features": max_features,
        "min_samples_leaf": 0.02,
        "max_depth": None, # controlled by min_samples_leaf
        "class_weight": "balanced_subsample",
        "criterion": "entropy",
        "n_jobs": default_n_jobs,
        "random_state": 42
    }
    
    if params_override:
        params.update(params_override)

    model = ExtraTreesClassifier(**params)

    # Note: ExtraTrees doesn't use eval_set or early_stopping_rounds in fit()
    model.fit(X, y, sample_weight=sample_weight)
    return model


PURGE_MINUTES = 45
EMBARGO_MINUTES = 15


def train_specialist_model_with_oof(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    n_splits: int = 5,
    params_override: Optional[Dict[str, Any]] = None,
) -> SpecialistTrainingResult:
    tprint_info(f"   [ExtraTrees] Training with OOF (Splits: {n_splits})...")
    
    if X.empty or len(y) == 0:
        raise ValueError("Cannot train model on empty dataset")

    X = X.copy()
    y = y.astype(float).copy()

    if sample_weight is not None:
        sample_weight = sample_weight.astype(float)
        # Ensure no NaN values in sample weights
        if sample_weight.isna().any():
            tprint_warning(f"   [ExtraTrees] Sample weights contain NaN values, replacing with 1.0")
            sample_weight = sample_weight.fillna(1.0)
        # Ensure no NaN values in sample weights
        if sample_weight.isna().any():
            tprint_warning(f"   [ExtraTrees] Sample weights contain NaN values, replacing with 1.0")
            sample_weight = sample_weight.fillna(1.0)

    # Adjust n_splits if n_samples is too small
    n_samples = len(X)
    if n_samples < n_splits * 2:
        tprint_info(f"   [ExtraTrees] Sample size {n_samples} too small for {n_splits} splits. Reducing n_splits.")
        n_splits = max(2, n_samples // 2)
        
    if n_samples < 2:
        raise ValueError(f"Cannot train model with only {n_samples} samples")

    splitter = PurgedKFoldTime(
        n_splits=n_splits,
        purge=pd.Timedelta(minutes=PURGE_MINUTES),
        embargo=pd.Timedelta(minutes=EMBARGO_MINUTES),
    )
    oof_probs = pd.Series(np.nan, index=X.index, dtype=float)
    last_model: Optional[ExtraTreesClassifier] = None

    for train_idx, val_idx in splitter.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train = sample_weight.iloc[train_idx] if sample_weight is not None else None

        model = _fit_single_model(
            X_train.fillna(0.0),
            y_train,
            sample_weight=w_train,
            params_override=params_override,
        )
        if model is not None:
            fold_probs = model.predict_proba(X_val.fillna(0.0))[:, 1]
            oof_probs.iloc[val_idx] = fold_probs
            last_model = model
        else:
            val_val = float(y_train.iloc[0])
            oof_probs.iloc[val_idx] = val_val

    tprint_info("   [ExtraTrees] Training Final Model on Full Data...")
    final_model = _fit_single_model(
        X.fillna(0.0),
        y,
        sample_weight=sample_weight,
        params_override=params_override,
    )
    
    metrics: Dict[str, float] = {
        "n_features": float(X.shape[1]),
        "n_samples": float(len(X)),
    }

    if final_model is None:
        tprint_info("   [ExtraTrees] Could not train final model due to single class data.")
        return SpecialistTrainingResult(
            model=None,
            params={},
            oof_predictions=oof_probs,
            metrics={**metrics, "auc": 0.5, "aucpr": 0.0, "mi_score": 0.0}
        )

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

    params_snapshot = final_model.get_params()

    return SpecialistTrainingResult(
        model=final_model,
        params=params_snapshot,
        oof_predictions=oof_probs,
        metrics=metrics,
    )
