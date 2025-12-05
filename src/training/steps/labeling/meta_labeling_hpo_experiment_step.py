"""Meta-Labeling HPO Experiment Step.

This offline step performs hierarchical hyperparameter optimization over
labeling-specific parameters (triple-barrier / TPSL, horizon, and target
clipping) using the HierarchicalParameterOptimizer.

It is intentionally decoupled from standard training runs. Invoke it
explicitly via the launcher with an appropriate config. A simple config
flag `enable_labeling_hpo` can be used to disable the optimization and
exit early if desired.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Optional
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
import lightgbm as lgb

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_success
from src.utils.ml_common.get_specialist_models_outputs import get_specialist_models_outputs

# Reuse core labeling utilities from the production meta-labeling step
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    compute_realized_returns,
    kalman_smooth_labels,
    fit_probability_to_return_mapping,
    translate_to_targets_with_isotonic,
    generate_primary_signals,
    DEFAULT_TRANSACTION_COST,
    ECON_MIN_RETURN_MULTIPLE,
    build_meta_features_for_model,
    compute_label_entropy_score,
    generate_diagnostics_report,
    compute_vol_scaled_returns_for_events,
    create_quantile_labels_from_vol_scaled_returns,
    attach_rolling_hmm_regimes_to_market_data,
    create_regime_aware_quantile_labels_from_vol_scaled_returns,
)

from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    create_param_group,
    OptimizationStage,
)
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    OptimizationConfig,
)
from src.utils.ml_common.optimization.pareto import (
    Solution,
    ParetoFront,
    compute_pareto_front,
    select_knee_point,
)


logger = system_logger.getChild("MetaLabelingHPOExperiment")

# Optional diagnostics for the recommended configuration can be useful but are
# not required for the HPO step to function. They have occasionally triggered
# pandas categorical setitem issues in some environments. To keep the HPO step
# robust, we disable these diagnostics by default and gate them behind this
# constant, which can be flipped to True if deeper investigation is needed.
GENERATE_RECOMMENDED_DIAGNOSTICS: bool = False

# Toggle for underfit diagnostics - computes learning curves, feature importance
# concentration, and probe vs deeper model comparisons. Adds computational cost.
ENABLE_UNDERFIT_DIAGNOSTICS: bool = True

# Multi-stage HPO configuration defaults
DEFAULT_STAGE_CONFIG = [
    {
        "name": "Stage 1 (Screening)",
        "complexity": "fast",
        "n_trials": 100,
        "top_k_to_pass": 30,  # Pass top 30 configurations to next stage
        "model_params": {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.1,
            "cv_splits": 3,
        },
    },
    {
        "name": "Stage 2 (Refinement)",
        "complexity": "medium",
        "n_trials": 50,
        "top_k_to_pass": 10,  # Pass top 10 to final stage
        "model_params": {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.05,
            "cv_splits": 4,
        },
    },
    {
        "name": "Stage 3 (Production Proxy)",
        "complexity": "strong",
        "n_trials": 20,  # Fewer trials, expensive model
        "top_k_to_pass": 1,
        "model_params": {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.01,
            "cv_splits": 5,
        },
    },
    {
        "name": "Stage 4 (Labeling Refinement)",
        "complexity": "strong",
        "n_trials": 30,
        "top_k_to_pass": 1,
        "model_params": {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.01,
            "cv_splits": 5,
        },
    },
]


def compute_learnability_with_calibration(
    X: pd.DataFrame,
    y: pd.Series,
    realized_returns: pd.Series,
    model_complexity: str = "fast",
    cv_splits: int = 3,
    time_aware_cv: bool = True,
    use_ensemble: bool = False,
    signal_strength_scale_max: float = 1.5,
) -> Tuple[float, float, np.ndarray, Optional[IsotonicRegression]]:
    """Compute learnability score with isotonic calibration for accurate P&L estimation.

    Unlike the basic compute_learnability_score, this function:
    1. Uses model complexity levels (fast/medium/strong)
    2. Returns calibrated probabilities via isotonic regression
    3. Supports ensemble models for strong complexity

    Args:
        X: Feature matrix
        y: Binary labels
        realized_returns: Realized returns for isotonic calibration
        model_complexity: "fast", "medium", or "strong"
        cv_splits: Number of CV splits
        time_aware_cv: Use TimeSeriesSplit instead of KFold
        use_ensemble: Whether to use ensemble of models (for strong complexity)

    Returns:
        Tuple of (learnability_score, mean_auc, calibrated_probabilities, isotonic_regressor)
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score

    # Remove NaN labels
    valid_mask = ~y.isna()
    X_num = X.select_dtypes(include=[np.number]) if isinstance(X, pd.DataFrame) else X
    if isinstance(X_num, pd.DataFrame) and X_num.empty:
        return 0.0, 0.5, np.array([]), None

    X_clean = X_num[valid_mask].fillna(0)
    y_clean = y[valid_mask]
    returns_clean = realized_returns[valid_mask]

    if len(y_clean) < 50:
        return 0.0, 0.5, np.array([]), None

    if len(y_clean.unique()) < 2:
        return 0.0, 0.5, np.array([]), None

    # Select model based on complexity
    if model_complexity == "fast":
        models = [lgb.LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            max_depth=3,
            n_estimators=50,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_samples=20,
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            n_jobs=-1,
            verbose=-1,
            random_state=42
        )]

    elif model_complexity == "medium":
        models = [lgb.LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            max_depth=5,
            n_estimators=150,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=15,
            reg_alpha=0.05,
            reg_lambda=0.05,
            n_jobs=-1,
            verbose=-1,
            random_state=42
        )]

    else:  # strong
        models = [
            lgb.LGBMClassifier(
                boosting_type='gbdt',
                objective='binary',
                max_depth=6,
                n_estimators=300,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=10,
                reg_alpha=0.01,
                reg_lambda=0.01,
                n_jobs=-1,
                verbose=-1,
                random_state=42
            )
        ]

        # Add XGBoost and RF for ensemble if available and requested
        if use_ensemble:
            if XGBOOST_AVAILABLE:
                models.append(xgb.XGBClassifier(
                    max_depth=6,
                    n_estimators=500,
                    learning_rate=0.01,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.01,
                    reg_lambda=0.01,
                    n_jobs=-1,
                    verbosity=0,
                    random_state=42
                ))

            models.append(RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=42
            ))

    # Time-aware CV
    if time_aware_cv:
        cv = TimeSeriesSplit(n_splits=cv_splits)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # Cost/return-aware sample weights with slight positive class bias (1.2x)
    returns_array = returns_clean.fillna(0.0).to_numpy(dtype=float)
    y_array = y_clean.to_numpy(dtype=float)

    sample_weights = np.ones_like(returns_array, dtype=float)

    # Conservative positive class up-weighting
    pos_mask = (y_array == 1.0)
    sample_weights[pos_mask] *= 1.2

    # Return-based weighting for label=1: linear in realized return, clipped
    try:
        finite_returns = returns_clean.replace([np.inf, -np.inf], np.nan).dropna().values
        if finite_returns.size >= 50:
            ret_clip = float(np.nanpercentile(np.abs(finite_returns), 95))
            ret_clip = max(ret_clip, 1e-4)
        else:
            ret_clip = 0.02
    except Exception:
        ret_clip = 0.02

    if ret_clip > 0:
        ret_for_weight = np.clip(np.maximum(returns_array, 0.0), 0.0, ret_clip)
        weight_factor = 1.0 + (ret_for_weight / ret_clip)
        sample_weights[pos_mask] *= weight_factor[pos_mask]

    # Optional: scale positive-class weights by signal strength so that
    # high-confidence signal configurations receive slightly higher weight
    # in the learnability scorer. We use the "signal_strength_all" meta-feature
    # when present in X_clean, normalised and clipped for robustness. The
    # overall strength of this effect is controlled by ``signal_strength_scale_max``.
    signal_strength = None
    if isinstance(X_clean, pd.DataFrame) and "signal_strength_all" in X_clean.columns:
        try:
            s = X_clean.loc[valid_mask, "signal_strength_all"].to_numpy(dtype=float)
            s = np.abs(s)
            # Robust scaling: use 90th percentile to avoid extreme values
            if np.isfinite(s).any():
                s_clean = s[np.isfinite(s)]
                if s_clean.size >= 10:
                    s_clip = float(np.nanpercentile(s_clean, 90))
                    s_clip = max(s_clip, 1e-6)
                else:
                    s_clip = float(np.nanmax(s_clean)) if s_clean.size > 0 else 1.0
                if s_clip <= 0:
                    s_clip = 1.0
                strength_norm = np.clip(s / s_clip, 0.0, 1.0)
                # Map to [1.0, signal_strength_scale_max] so HPO can tune the
                # influence of signal strength on sample weighting.
                scale_max = max(1.0, float(signal_strength_scale_max))
                scale_range = max(0.0, scale_max - 1.0)
                signal_weight = 1.0 + scale_range * strength_norm
                sample_weights[pos_mask] *= signal_weight[pos_mask]
        except Exception:
            # If anything goes wrong, fall back to return-only weighting.
            pass

    # Normalize weights for numerical stability
    mean_w = float(sample_weights.mean()) if sample_weights.size > 0 else 1.0
    if mean_w > 0:
        sample_weights = sample_weights / mean_w

    try:
        # Collect probability predictions and AUC scores from all models
        all_full_probs = []
        all_aucs = []

        # Store OOF predictions for each model to build ensemble OOF
        # This allows accurate OOF-based calibration
        all_model_oof_preds: List[np.ndarray] = []
        oof_indices: np.ndarray = np.array([], dtype=int)

        for i, model in enumerate(models):
            # Manual time-series CV with sample weights
            fold_aucs = []

            # Temporary storage for this model's OOF predictions across folds
            model_oof_parts: List[np.ndarray] = []
            current_oof_indices: List[np.ndarray] = []

            for train_idx, test_idx in cv.split(X_clean):
                X_train_cv = X_clean.iloc[train_idx]
                y_train_cv = y_clean.iloc[train_idx]
                X_test_cv = X_clean.iloc[test_idx]
                y_test_cv = y_clean.iloc[test_idx]

                w_train_cv = sample_weights[train_idx]

                try:
                    model.fit(X_train_cv, y_train_cv, sample_weight=w_train_cv)
                except TypeError:
                    # Some models may not support sample_weight
                    model.fit(X_train_cv, y_train_cv)

                y_proba_cv = model.predict_proba(X_test_cv)[:, 1]

                # Store OOF predictions
                model_oof_parts.append(y_proba_cv)
                if i == 0:
                    current_oof_indices.append(test_idx)

                try:
                    fold_auc = roc_auc_score(y_test_cv, y_proba_cv)
                    fold_aucs.append(fold_auc)
                except Exception:
                    pass

            if fold_aucs:
                mean_auc = float(np.mean(fold_aucs))
            else:
                mean_auc = 0.5

            # Combine OOF parts for this model
            if model_oof_parts:
                all_model_oof_preds.append(np.concatenate(model_oof_parts))

            if i == 0 and current_oof_indices:
                oof_indices = np.concatenate(current_oof_indices)

            # Fit on full cleaned data with weights for final probabilities
            try:
                model.fit(X_clean, y_clean, sample_weight=sample_weights)
            except TypeError:
                model.fit(X_clean, y_clean)

            full_probs = model.predict_proba(X_clean)[:, 1]

            all_full_probs.append(full_probs)
            all_aucs.append(mean_auc)

        # Ensemble: average probabilities (with signal disagreement awareness)
        if len(models) > 1:
            full_probs_array = np.array(all_full_probs)

            # Calculate disagreement (std across models)
            disagreement = np.std(full_probs_array, axis=0)

            # Average probabilities
            avg_probs = np.mean(full_probs_array, axis=0)

            # Penalize high-disagreement predictions slightly
            # (reduce confidence when models disagree)
            confidence_penalty = 1.0 - (disagreement * 0.5)  # Max 50% penalty
            final_probs = avg_probs * confidence_penalty + (1 - confidence_penalty) * 0.5
            final_probs = np.clip(final_probs, 0.0, 1.0)

            mean_auc = np.mean(all_aucs)
        else:
            final_probs = all_full_probs[0]
            mean_auc = all_aucs[0]

        std_auc = np.std(all_aucs) if len(all_aucs) > 1 else 0.0

        # Apply isotonic calibration to probabilities using OOF predictions
        iso_reg = None
        calibrated_probs = final_probs

        if model_complexity in ["medium", "strong"] and all_model_oof_preds and len(oof_indices) > 50:
            try:
                # Build Ensemble OOF predictions
                oof_probs_array = np.array(all_model_oof_preds)

                if len(models) > 1:
                    avg_oof_probs = np.mean(oof_probs_array, axis=0)
                    disagreement_oof = np.std(oof_probs_array, axis=0)
                    confidence_penalty_oof = 1.0 - (disagreement_oof * 0.5)
                    final_oof_probs = avg_oof_probs * confidence_penalty_oof + (1 - confidence_penalty_oof) * 0.5
                    final_oof_probs = np.clip(final_oof_probs, 0.0, 1.0)
                else:
                    final_oof_probs = oof_probs_array[0]

                # Get OOF returns for calibration
                oof_returns = returns_clean.iloc[oof_indices].values

                iso_reg = IsotonicRegression(out_of_bounds='clip')

                # Fit on valid OOF samples
                valid_for_iso = np.isfinite(oof_returns) & np.isfinite(final_oof_probs)
                if np.sum(valid_for_iso) > 50:
                    iso_reg.fit(final_oof_probs[valid_for_iso], oof_returns[valid_for_iso])
                    # Calibrate full model probabilities using OOF-learned mapping
                    calibrated_probs = iso_reg.predict(final_probs)
                else:
                    # Fallback to in-sample if OOF not valid
                    valid_in_sample = np.isfinite(returns_clean.values) & np.isfinite(final_probs)
                    if np.sum(valid_in_sample) > 50:
                         iso_reg.fit(final_probs[valid_in_sample], returns_clean.values[valid_in_sample])
                         calibrated_probs = iso_reg.predict(final_probs)
            except Exception:
                calibrated_probs = final_probs
                iso_reg = None
        else:
             # Fallback calibration using in-sample data if OOF unavailable (e.g. data too small for CV)
             if model_complexity in ["medium", "strong"]:
                 try:
                     iso_reg = IsotonicRegression(out_of_bounds='clip')
                     valid_for_iso = np.isfinite(returns_clean.values) & np.isfinite(final_probs)
                     if np.sum(valid_for_iso) > 50:
                        iso_reg.fit(final_probs[valid_for_iso], returns_clean.values[valid_for_iso])
                        calibrated_probs = iso_reg.predict(final_probs)
                 except Exception:
                     pass

        # Learnability score: penalize instability
        learnability = mean_auc - (0.5 * std_auc)

        return learnability, mean_auc, calibrated_probs, iso_reg

    except Exception as e:
        tprint(f"⚠️ Calibrated learnability scoring failed: {e}", "WARNING")
        return 0.0, 0.5, np.array([]), None


def compute_underfit_diagnostics(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 3,
    time_aware_cv: bool = True,
) -> Dict[str, Any]:
    """Compute underfit diagnostics to assess room for model improvement.

    Computes:
    1. Learning curves at different data fractions
    2. Learning curves at different depths
    3. Feature importance concentration
    4. Probe vs deeper model AUC comparison

    Args:
        X: Feature matrix
        y: Binary labels
        cv_splits: Number of CV splits
        time_aware_cv: Use TimeSeriesSplit

    Returns:
        Dictionary with diagnostic metrics
    """
    from sklearn.model_selection import cross_val_score, learning_curve

    diagnostics = {
        "learning_curve_fractions": {},
        "learning_curve_depths": {},
        "feature_importance_concentration": None,
        "feature_group_importance": None,
        "top_feature_importances": [],
        "probe_vs_deep_auc_diff": None,
        "is_underfit": False,
        "underfit_indicators": [],
    }

    # Remove NaN labels
    valid_mask = ~y.isna()
    X_num = X.select_dtypes(include=[np.number]) if isinstance(X, pd.DataFrame) else X
    if isinstance(X_num, pd.DataFrame) and X_num.empty:
        return diagnostics

    X_clean = X_num[valid_mask].fillna(0)
    y_clean = y[valid_mask]

    if len(y_clean) < 100 or len(y_clean.unique()) < 2:
        return diagnostics

    # Time-aware CV
    if time_aware_cv:
        cv = TimeSeriesSplit(n_splits=cv_splits)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    try:
        # 1. Learning curves with data fractions (20%, 40%, 60%, 80%, 100%)
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        fraction_aucs = []

        probe_model = lgb.LGBMClassifier(
            max_depth=3, n_estimators=50, learning_rate=0.1,
            n_jobs=-1, verbose=-1, random_state=42
        )

        for frac in fractions:
            n_samples = int(len(X_clean) * frac)
            if n_samples < 50:
                continue

            X_frac = X_clean.iloc[:n_samples]
            y_frac = y_clean.iloc[:n_samples]

            if len(y_frac.unique()) < 2:
                continue

            try:
                scores = cross_val_score(
                    probe_model, X_frac, y_frac,
                    cv=min(cv_splits, len(y_frac) // 20),
                    scoring='roc_auc', n_jobs=-1
                )
                diagnostics["learning_curve_fractions"][frac] = float(scores.mean())
                fraction_aucs.append(scores.mean())
            except Exception:
                pass

        # Check if AUC keeps rising without plateau (underfit indicator)
        if len(fraction_aucs) >= 3:
            # If last improvement > 2%, likely underfit
            if fraction_aucs[-1] - fraction_aucs[-2] > 0.02:
                diagnostics["is_underfit"] = True
                diagnostics["underfit_indicators"].append("AUC still rising with more data")

        # 2. Learning curves with different depths (3, 5, 7)
        depths = [3, 5, 7]
        depth_aucs = []

        for depth in depths:
            model = lgb.LGBMClassifier(
                max_depth=depth, n_estimators=100, learning_rate=0.05,
                n_jobs=-1, verbose=-1, random_state=42
            )

            try:
                scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring='roc_auc', n_jobs=-1)
                auc = float(scores.mean())
                diagnostics["learning_curve_depths"][depth] = auc
                depth_aucs.append(auc)
            except Exception:
                pass

        # 3. Probe vs deeper model comparison
        if len(depth_aucs) >= 2:
            probe_auc = depth_aucs[0]  # depth=3
            best_deep_auc = max(depth_aucs[1:])  # best of deeper models

            auc_diff = best_deep_auc - probe_auc
            diagnostics["probe_vs_deep_auc_diff"] = float(auc_diff)

            # If deeper model improves >5%, probe is underfit
            if auc_diff > 0.05:
                diagnostics["is_underfit"] = True
                diagnostics["underfit_indicators"].append(f"Deeper model +{auc_diff:.1%} AUC")

        # 4. Feature importance concentration (top 5 features)
        try:
            final_model = lgb.LGBMClassifier(
                max_depth=5, n_estimators=100, learning_rate=0.05,
                n_jobs=-1, verbose=-1, random_state=42
            )
            final_model.fit(X_clean, y_clean)

            importances = final_model.feature_importances_
            sorted_imp = np.sort(importances)[::-1]
            total_imp = importances.sum()
            feature_names = list(X_clean.columns) if isinstance(X_clean, pd.DataFrame) else None

            if total_imp > 0:
                top_5_concentration = sorted_imp[:5].sum() / total_imp
                diagnostics["feature_importance_concentration"] = float(top_5_concentration)

                if feature_names is not None and len(feature_names) == len(importances):
                    group_totals = {"volatility": 0.0, "signal": 0.0, "other": 0.0}
                    top_features: list[dict[str, Any]] = []
                    idx_sorted = np.argsort(importances)[::-1]
                    top_k = min(20, len(idx_sorted))
                    for idx in idx_sorted[:top_k]:
                        name = str(feature_names[idx])
                        share = float(importances[idx]) / float(total_imp)
                        lname = name.lower()
                        if any(tok in lname for tok in ("vol", "atr", "std", "var", "range")):
                            group = "volatility"
                        elif any(tok in lname for tok in ("signal", "alpha", "entry", "meta", "prob")):
                            group = "signal"
                        else:
                            group = "other"
                        group_totals[group] += share
                        top_features.append({"name": name, "importance": share})
                    diagnostics["feature_group_importance"] = group_totals
                    diagnostics["top_feature_importances"] = top_features

                # If top 5 features dominate 80%+, model is only scratching surface
                if top_5_concentration > 0.8:
                    diagnostics["is_underfit"] = True
                    diagnostics["underfit_indicators"].append(
                        f"Top 5 features = {top_5_concentration:.1%} importance"
                    )
        except Exception:
            pass

    except Exception as e:
        tprint(f"⚠️ Underfit diagnostics failed: {e}", "WARNING")

    return diagnostics


def compute_filtering_inflation_diagnostics(
    X: pd.DataFrame,
    y_full: pd.Series,
    y_filtered: pd.Series,
    realized_returns: pd.Series,
    volatility: pd.Series,
    probabilities: np.ndarray,
    cv_splits: int = 3,
    time_aware_cv: bool = True,
) -> Dict[str, Any]:
    """Compute diagnostics to detect AUC inflation from aggressive filtering.

    Tests:
    1. AUC on full vs filtered labels - if filtered AUC >> full AUC, filtering inflates
    2. AUC by return magnitude bucket - if AUC high only in large-move bins, labels dominated
    3. Class overlap metrics - how separable are retained vs discarded events
    4. Precision@K drop - if precision collapses as K increases, model only good on easy cases

    Args:
        X: Feature matrix (aligned to full label space)
        y_full: Labels before filtering (pre-quantile, using economic floor only)
        y_filtered: Labels after all filtering (quantile-based, final)
        realized_returns: Realized returns per event
        volatility: Volatility series for normalization
        probabilities: Model predicted probabilities (aligned to filtered labels)
        cv_splits: Number of CV splits for AUC computation
        time_aware_cv: Use TimeSeriesSplit

    Returns:
        Dictionary with diagnostic metrics
    """
    from sklearn.metrics import roc_auc_score, precision_score
    from sklearn.model_selection import TimeSeriesSplit, cross_val_predict

    diagnostics = {
        # 1. Full vs Filtered AUC comparison
        "auc_full": None,
        "auc_filtered": None,
        "auc_inflation": None,  # filtered - full
        "filtering_is_major_contributor": False,  # True if inflation > 0.08
        
        # 2. AUC by return magnitude bucket
        "auc_by_return_bucket": {},
        "auc_dominated_by_large_moves": False,
        
        # 3. Retention metrics
        "n_full_events": 0,
        "n_filtered_events": 0,
        "retention_rate": 0.0,
        
        # 4. Precision@K analysis
        "precision_at_k": {},
        "precision_collapse_detected": False,
    }

    try:
        # Align indices
        common_idx = y_full.index.intersection(y_filtered.index)
        if len(common_idx) < 50:
            return diagnostics

        # ===== 1. AUC on Full vs Filtered Labels =====
        full_mask = ~y_full.isna()
        filtered_mask = ~y_filtered.isna()
        
        n_full = int(full_mask.sum())
        n_filtered = int(filtered_mask.sum())
        diagnostics["n_full_events"] = n_full
        diagnostics["n_filtered_events"] = n_filtered
        diagnostics["retention_rate"] = n_filtered / max(n_full, 1)

        # Compute AUC on filtered labels (what we normally report)
        if n_filtered >= 50 and len(np.unique(y_filtered[filtered_mask])) >= 2:
            if probabilities is not None and len(probabilities) == len(y_filtered):
                try:
                    probs_filtered = probabilities[filtered_mask.values]
                    y_filt_vals = y_filtered[filtered_mask].values
                    diagnostics["auc_filtered"] = float(roc_auc_score(y_filt_vals, probs_filtered))
                except Exception:
                    pass

        # Compute AUC on full labels (before quantile filtering)
        # Need to retrain/predict on full label space
        if n_full >= 50 and len(np.unique(y_full[full_mask])) >= 2:
            try:
                X_num = X.select_dtypes(include=[np.number])
                X_full = X_num[full_mask].fillna(0)
                y_full_vals = y_full[full_mask]
                
                if len(y_full_vals) >= 50 and len(y_full_vals.unique()) >= 2:
                    # Quick probe model for full-label AUC
                    probe = lgb.LGBMClassifier(
                        max_depth=3, n_estimators=50, learning_rate=0.1,
                        n_jobs=-1, verbose=-1, random_state=42
                    )
                    
                    if time_aware_cv:
                        cv = TimeSeriesSplit(n_splits=min(cv_splits, len(y_full_vals) // 20))
                    else:
                        from sklearn.model_selection import KFold
                        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
                    
                    probs_full = cross_val_predict(probe, X_full, y_full_vals, cv=cv, method='predict_proba')[:, 1]
                    diagnostics["auc_full"] = float(roc_auc_score(y_full_vals, probs_full))
            except Exception:
                pass

        # Compute inflation
        if diagnostics["auc_filtered"] is not None and diagnostics["auc_full"] is not None:
            diagnostics["auc_inflation"] = diagnostics["auc_filtered"] - diagnostics["auc_full"]
            diagnostics["filtering_is_major_contributor"] = diagnostics["auc_inflation"] > 0.08

        # ===== 2. AUC by Return Magnitude Bucket =====
        try:
            vol_aligned = volatility.reindex(realized_returns.index).fillna(volatility.median())
            vol_scaled = realized_returns / (vol_aligned.abs() + 1e-8)
            abs_vol_scaled = vol_scaled.abs()
            
            # Define buckets by sigma
            buckets = [
                ("0-0.1σ", 0.0, 0.1),
                ("0.1-0.3σ", 0.1, 0.3),
                ("0.3-0.8σ", 0.3, 0.8),
                ("0.8-1.5σ", 0.8, 1.5),
                ("1.5σ+", 1.5, float('inf')),
            ]
            
            bucket_aucs = {}
            small_move_aucs = []
            large_move_aucs = []
            
            for bucket_name, low, high in buckets:
                bucket_mask = (abs_vol_scaled >= low) & (abs_vol_scaled < high) & filtered_mask
                n_bucket = int(bucket_mask.sum())
                
                if n_bucket >= 30:
                    try:
                        y_bucket = y_filtered[bucket_mask]
                        if len(y_bucket.unique()) >= 2 and probabilities is not None:
                            probs_bucket = probabilities[bucket_mask.values]
                            auc_bucket = float(roc_auc_score(y_bucket.values, probs_bucket))
                            bucket_aucs[bucket_name] = {"auc": auc_bucket, "n": n_bucket}
                            
                            # Track for dominance check
                            if high <= 0.3:
                                small_move_aucs.append(auc_bucket)
                            elif low >= 0.8:
                                large_move_aucs.append(auc_bucket)
                    except Exception:
                        bucket_aucs[bucket_name] = {"auc": None, "n": n_bucket}
                else:
                    bucket_aucs[bucket_name] = {"auc": None, "n": n_bucket}
            
            diagnostics["auc_by_return_bucket"] = bucket_aucs
            
            # Check if dominated by large moves
            if small_move_aucs and large_move_aucs:
                avg_small = np.mean(small_move_aucs)
                avg_large = np.mean(large_move_aucs)
                # Dominated if small AUC is 0.52-0.56 and large is 0.65+
                if avg_small < 0.58 and avg_large > 0.63:
                    diagnostics["auc_dominated_by_large_moves"] = True
                    
        except Exception:
            pass

        # ===== 4. Precision@K Drop Analysis =====
        try:
            if probabilities is not None and n_filtered >= 50:
                y_filt = y_filtered[filtered_mask].values
                probs_filt = probabilities[filtered_mask.values]
                
                # Sort by probability descending
                sorted_idx = np.argsort(probs_filt)[::-1]
                y_sorted = y_filt[sorted_idx]
                
                # Compute precision at different K percentiles
                precision_at_k = {}
                for k_pct in [10, 20, 30, 50, 70]:
                    k = max(1, int(len(y_sorted) * k_pct / 100))
                    precision_k = float(y_sorted[:k].mean())
                    precision_at_k[f"top_{k_pct}%"] = precision_k
                
                diagnostics["precision_at_k"] = precision_at_k
                
                # Check for collapse: top 10% precision is good but top 50% collapses
                if "top_10%" in precision_at_k and "top_50%" in precision_at_k:
                    if precision_at_k["top_10%"] > 0.65 and precision_at_k["top_50%"] < 0.55:
                        diagnostics["precision_collapse_detected"] = True
                        
        except Exception:
            pass

    except Exception as e:
        tprint(f"⚠️ Filtering inflation diagnostics failed: {e}", "WARNING")

    return diagnostics


def compute_calibration_diagnostics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    realized_returns: np.ndarray,
    transaction_cost: float = 0.003,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute calibration diagnostics for the meta-model.

    Includes:
    - Brier score
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Reliability diagram data
    - Precision per probability bin (especially ≥0.6, ≥0.7, ≥0.8)
    - Expected net P&L by probability bucket

    Args:
        y_true: True binary labels
        probabilities: Model predicted probabilities
        realized_returns: Realized returns for P&L calculation
        transaction_cost: Transaction cost per trade
        n_bins: Number of bins for calibration

    Returns:
        Dictionary with calibration metrics
    """
    diagnostics = {
        "brier_score": None,
        "ece": None,  # Expected Calibration Error
        "mce": None,  # Maximum Calibration Error
        "reliability_diagram": [],  # List of (mean_pred, mean_actual, count) per bin
        "precision_per_bin": {},
        "fpr_per_bin": {},
        "expected_pnl_per_bin": {},
        "is_well_calibrated": False,
    }

    try:
        # Remove NaN
        valid_mask = ~(np.isnan(y_true) | np.isnan(probabilities))
        y = y_true[valid_mask]
        probs = probabilities[valid_mask]
        returns = realized_returns[valid_mask] if realized_returns is not None else None
        
        if len(y) < 50:
            return diagnostics

        # ===== Brier Score =====
        diagnostics["brier_score"] = float(np.mean((probs - y) ** 2))

        # ===== Calibration Error (ECE, MCE) and Reliability Diagram =====
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        reliability_data = []
        
        for i in range(n_bins):
            bin_mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if i == n_bins - 1:  # Include 1.0 in last bin
                bin_mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
            
            n_in_bin = int(bin_mask.sum())
            if n_in_bin > 0:
                mean_pred = float(probs[bin_mask].mean())
                mean_actual = float(y[bin_mask].mean())
                calibration_error = abs(mean_pred - mean_actual)
                
                ece += (n_in_bin / len(y)) * calibration_error
                mce = max(mce, calibration_error)
                
                reliability_data.append({
                    "bin_start": float(bin_edges[i]),
                    "bin_end": float(bin_edges[i + 1]),
                    "mean_predicted": mean_pred,
                    "mean_actual": mean_actual,
                    "count": n_in_bin,
                    "calibration_error": calibration_error,
                })

        diagnostics["ece"] = float(ece)
        diagnostics["mce"] = float(mce)
        diagnostics["reliability_diagram"] = reliability_data
        diagnostics["is_well_calibrated"] = ece < 0.05

        # ===== Precision Per Bin (especially high-probability regions) =====
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        for thresh in thresholds:
            mask = probs >= thresh
            n_above = int(mask.sum())
            if n_above >= 10:
                precision = float(y[mask].mean())
                fpr = float((1 - y[mask]).mean())  # False positive rate
                diagnostics["precision_per_bin"][f"≥{thresh}"] = {
                    "precision": precision,
                    "count": n_above,
                }
                diagnostics["fpr_per_bin"][f"≥{thresh}"] = {
                    "fpr": fpr,
                    "count": n_above,
                }

        # ===== Expected Net P&L by Probability Bucket =====
        if returns is not None:
            pnl_bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
            for low, high in pnl_bins:
                mask = (probs >= low) & (probs < high)
                n_in_bin = int(mask.sum())
                if n_in_bin >= 10:
                    bin_returns = returns[mask]
                    # Net P&L after costs (assuming we trade all events in bin)
                    net_pnl = float(bin_returns.mean() - transaction_cost)
                    pnl_per_trade = net_pnl
                    expected_total = net_pnl * n_in_bin
                    
                    diagnostics["expected_pnl_per_bin"][f"{low:.1f}-{high:.1f}"] = {
                        "mean_return": float(bin_returns.mean()),
                        "net_pnl_per_trade": pnl_per_trade,
                        "n_trades": n_in_bin,
                        "expected_total_pnl": expected_total,
                        "is_profitable": pnl_per_trade > 0,
                    }

    except Exception as e:
        tprint(f"⚠️ Calibration diagnostics failed: {e}", "WARNING")

    return diagnostics


def compute_robustness_diagnostics(
    X: pd.DataFrame,
    y: pd.Series,
    realized_returns: pd.Series,
    regimes: Optional[pd.Series] = None,
    volatility: Optional[pd.Series] = None,
    n_folds: int = 5,
    transaction_cost: float = 0.003,
) -> Dict[str, Any]:
    """Compute robustness diagnostics across time and regimes.

    Tests:
    - Per-fold metrics (AUC, ECE, precision@threshold, net P&L)
    - Worst-fold performance and CV dispersion
    - Performance vs volatility regimes (low, median, high)
    - Per-regime breakdown if regimes provided

    Args:
        X: Feature matrix
        y: Binary labels
        realized_returns: Realized returns
        regimes: Optional regime labels
        volatility: Optional volatility series for regime splitting
        n_folds: Number of CV folds
        transaction_cost: Transaction cost

    Returns:
        Dictionary with robustness metrics
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score

    diagnostics = {
        "per_fold_metrics": [],
        "worst_fold_auc": None,
        "best_fold_auc": None,
        "auc_cv_std": None,
        "auc_cv_coefficient_of_variation": None,
        "per_volatility_regime": {},
        "per_regime_metrics": {},
        "is_robust": False,
    }

    try:
        # Clean data
        valid_mask = ~y.isna()
        X_num = X.select_dtypes(include=[np.number])
        X_clean = X_num[valid_mask].fillna(0)
        y_clean = y[valid_mask]
        returns_clean = realized_returns[valid_mask] if realized_returns is not None else None

        if len(y_clean) < 100 or len(y_clean.unique()) < 2:
            return diagnostics

        # ===== Per-Fold Metrics =====
        cv = TimeSeriesSplit(n_splits=n_folds)
        fold_aucs = []
        fold_metrics = []

        model = lgb.LGBMClassifier(
            max_depth=5, n_estimators=100, learning_rate=0.05,
            n_jobs=-1, verbose=-1, random_state=42
        )

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_clean)):
            try:
                X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
                y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
                
                if len(y_train.unique()) < 2 or len(y_test.unique()) < 2:
                    continue
                
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)[:, 1]
                
                # AUC
                auc = float(roc_auc_score(y_test, probs))
                fold_aucs.append(auc)
                
                # ECE
                ece = 0.0
                bin_edges = np.linspace(0, 1, 11)
                for i in range(10):
                    mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
                    if mask.sum() > 0:
                        ece += (mask.sum() / len(probs)) * abs(probs[mask].mean() - y_test.iloc[mask.values].mean())
                
                # Precision at 0.6
                precision_06 = float(y_test[probs >= 0.6].mean()) if (probs >= 0.6).sum() > 0 else None
                
                # Net P&L
                net_pnl = None
                if returns_clean is not None:
                    returns_test = returns_clean.iloc[test_idx]
                    trade_mask = probs >= 0.5
                    if trade_mask.sum() > 0:
                        net_pnl = float(returns_test[trade_mask].mean() - transaction_cost)
                
                fold_metrics.append({
                    "fold": fold_idx,
                    "auc": auc,
                    "ece": float(ece),
                    "precision_at_0.6": precision_06,
                    "net_pnl_per_trade": net_pnl,
                    "n_test": len(y_test),
                })
                
            except Exception:
                continue

        diagnostics["per_fold_metrics"] = fold_metrics
        
        if fold_aucs:
            diagnostics["worst_fold_auc"] = float(min(fold_aucs))
            diagnostics["best_fold_auc"] = float(max(fold_aucs))
            diagnostics["auc_cv_std"] = float(np.std(fold_aucs))
            mean_auc = float(np.mean(fold_aucs))
            diagnostics["auc_cv_coefficient_of_variation"] = float(np.std(fold_aucs) / mean_auc) if mean_auc > 0 else None
            
            # Robust if CV std < 0.05 and worst fold > 0.52
            diagnostics["is_robust"] = (diagnostics["auc_cv_std"] < 0.05 and diagnostics["worst_fold_auc"] > 0.52)

        # ===== Per Volatility Regime =====
        if volatility is not None:
            try:
                vol_aligned = volatility.reindex(y_clean.index)
                vol_clean = vol_aligned[~vol_aligned.isna()]
                
                if len(vol_clean) >= 50:
                    vol_low_thresh = vol_clean.quantile(0.33)
                    vol_high_thresh = vol_clean.quantile(0.67)
                    
                    vol_regimes = {
                        "low_vol": vol_aligned <= vol_low_thresh,
                        "medium_vol": (vol_aligned > vol_low_thresh) & (vol_aligned <= vol_high_thresh),
                        "high_vol": vol_aligned > vol_high_thresh,
                    }
                    
                    for regime_name, regime_mask in vol_regimes.items():
                        regime_mask = regime_mask.reindex(y_clean.index, fill_value=False)
                        n_regime = int(regime_mask.sum())
                        
                        if n_regime >= 30 and len(y_clean[regime_mask].unique()) >= 2:
                            try:
                                # Train on all, evaluate on regime
                                model.fit(X_clean, y_clean)
                                probs_regime = model.predict_proba(X_clean[regime_mask])[:, 1]
                                auc_regime = float(roc_auc_score(y_clean[regime_mask], probs_regime))
                                
                                diagnostics["per_volatility_regime"][regime_name] = {
                                    "auc": auc_regime,
                                    "n_events": n_regime,
                                }
                            except Exception:
                                diagnostics["per_volatility_regime"][regime_name] = {"auc": None, "n_events": n_regime}
                        else:
                            diagnostics["per_volatility_regime"][regime_name] = {"auc": None, "n_events": n_regime}
                            
            except Exception:
                pass

        # ===== Per Regime (if provided) =====
        if regimes is not None:
            try:
                regimes_aligned = regimes.reindex(y_clean.index)
                unique_regimes = pd.unique(regimes_aligned.dropna())
                
                for reg_val in unique_regimes:
                    regime_mask = regimes_aligned == reg_val
                    n_regime = int(regime_mask.sum())
                    
                    if n_regime >= 30 and len(y_clean[regime_mask].unique()) >= 2:
                        try:
                            model.fit(X_clean, y_clean)
                            probs_regime = model.predict_proba(X_clean[regime_mask])[:, 1]
                            auc_regime = float(roc_auc_score(y_clean[regime_mask], probs_regime))
                            
                            # Net P&L in regime
                            net_pnl = None
                            if returns_clean is not None:
                                returns_regime = returns_clean[regime_mask]
                                trade_mask = probs_regime >= 0.5
                                if trade_mask.sum() > 0:
                                    net_pnl = float(returns_regime[trade_mask].mean() - transaction_cost)
                            
                            diagnostics["per_regime_metrics"][str(reg_val)] = {
                                "auc": auc_regime,
                                "n_events": n_regime,
                                "net_pnl_per_trade": net_pnl,
                            }
                        except Exception:
                            diagnostics["per_regime_metrics"][str(reg_val)] = {"auc": None, "n_events": n_regime}
                    else:
                        diagnostics["per_regime_metrics"][str(reg_val)] = {"auc": None, "n_events": n_regime}
                        
            except Exception:
                pass

    except Exception as e:
        tprint(f"⚠️ Robustness diagnostics failed: {e}", "WARNING")

    return diagnostics


def compute_class_overlap_features(
    X: pd.DataFrame,
    retained_mask: pd.Series,
    top_k_features: int = 10,
) -> Dict[str, Any]:
    """Compute class overlap visualization data for retained vs discarded events.

    Args:
        X: Feature matrix
        retained_mask: Boolean mask - True for retained events
        top_k_features: Number of top features to analyze

    Returns:
        Dictionary with overlap metrics and distribution data
    """
    diagnostics = {
        "feature_distributions": {},
        "overlap_scores": {},
        "retained_cluster_tightness": None,
        "easy_problem_detected": False,
    }

    try:
        X_num = X.select_dtypes(include=[np.number]).fillna(0)
        
        if X_num.empty or len(retained_mask) != len(X_num):
            return diagnostics

        retained = X_num[retained_mask]
        discarded = X_num[~retained_mask]
        
        if len(retained) < 20 or len(discarded) < 20:
            return diagnostics

        # Compute feature importance to select top features
        from sklearn.ensemble import RandomForestClassifier
        try:
            rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1)
            rf.fit(X_num, retained_mask.astype(int))
            importances = rf.feature_importances_
            top_idx = np.argsort(importances)[::-1][:top_k_features]
            top_features = [X_num.columns[i] for i in top_idx]
        except Exception:
            top_features = list(X_num.columns[:top_k_features])

        # For each top feature, compute distribution stats
        overlap_scores = []
        for feat in top_features:
            try:
                ret_vals = retained[feat].values
                disc_vals = discarded[feat].values
                
                # Compute overlap using Bhattacharyya coefficient approximation
                ret_mean, ret_std = ret_vals.mean(), ret_vals.std() + 1e-8
                disc_mean, disc_std = disc_vals.mean(), disc_vals.std() + 1e-8
                
                # Bhattacharyya distance (lower = more overlap)
                bd = 0.25 * np.log(0.25 * ((ret_std/disc_std)**2 + (disc_std/ret_std)**2 + 2)) + \
                     0.25 * ((ret_mean - disc_mean)**2 / (ret_std**2 + disc_std**2))
                
                overlap_score = np.exp(-bd)  # Convert to 0-1, higher = more overlap
                overlap_scores.append(overlap_score)
                
                diagnostics["feature_distributions"][feat] = {
                    "retained_mean": float(ret_mean),
                    "retained_std": float(ret_std),
                    "discarded_mean": float(disc_mean),
                    "discarded_std": float(disc_std),
                    "overlap_score": float(overlap_score),
                }
                diagnostics["overlap_scores"][feat] = float(overlap_score)
                
            except Exception:
                continue

        # Overall cluster tightness (variance ratio)
        try:
            ret_var = retained.var().mean()
            disc_var = discarded.var().mean()
            full_var = X_num.var().mean()
            
            # If retained events have much lower variance, they're clustered
            diagnostics["retained_cluster_tightness"] = float(ret_var / (full_var + 1e-8))
            
            # Easy problem if: low overlap AND tight clustering
            avg_overlap = np.mean(overlap_scores) if overlap_scores else 1.0
            if avg_overlap < 0.4 and diagnostics["retained_cluster_tightness"] < 0.7:
                diagnostics["easy_problem_detected"] = True
                
        except Exception:
            pass

    except Exception as e:
        tprint(f"⚠️ Class overlap diagnostics failed: {e}", "WARNING")

    return diagnostics


def shrink_search_space(
    original_space: Dict[str, Any],
    previous_results: List[Dict[str, Any]],
    top_k: int = 20,
) -> Dict[str, Any]:
    """Narrow the search space around the top-K best performing parameters.

    Args:
        original_space: Original parameter search space
        previous_results: List of results from previous stage (must have 'params' and 'combined' keys)
        top_k: Number of top candidates to consider

    Returns:
        Narrowed search space
    """
    if not previous_results:
        return original_space.copy()

    # Sort by objective score (prefer edge, fallback to combined) and take top-K
    sorted_results = sorted(previous_results, key=lambda x: x.get('edge', x.get('combined', 0)), reverse=True)
    best_candidates = sorted_results[:top_k]

    if not best_candidates:
        return original_space.copy()

    new_space = {}

    for param_name, config in original_space.items():
        # Extract values for this param from best candidates
        values = []
        for c in best_candidates:
            if 'params' in c and param_name in c['params']:
                values.append(c['params'][param_name])

        if not values:
            new_space[param_name] = config.copy()
            continue

        param_type = config.get('type', 'float')

        if param_type in ['float', 'int']:
            min_val = min(values)
            max_val = max(values)

            # Add 10% buffer so we don't over-collapse
            spread = max_val - min_val
            if spread == 0:
                spread = (config.get('high', 1) - config.get('low', 0)) * 0.1

            buffer = spread * 0.1
            new_low = max(config.get('low', 0), min_val - buffer)
            new_high = min(config.get('high', 1), max_val + buffer)

            # Ensure we don't collapse completely
            if new_high <= new_low:
                new_low = config.get('low', 0)
                new_high = config.get('high', 1)

            new_config = config.copy()
            new_config['low'] = new_low if param_type == 'float' else int(new_low)
            new_config['high'] = new_high if param_type == 'float' else int(max(new_high, new_low + 1))
            new_space[param_name] = new_config

        else:
            # For categorical params, keep all or filter to frequent ones
            new_space[param_name] = config.copy()

    return new_space


def compute_realistic_pnl_edge(
    mean_return_positive: float,
    mean_auc: float,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    n_trades: int | None = None,
    reference_trades: float | None = None,
    days_span: float | None = None,
    target_trades_per_day: float = 2.0,
) -> float:
    """Compute realistic P&L edge using the capture ratio formula.

    Edge_base = (Mean_Return_Label1 - Cost) × max(0, 2×AUC - 1)

    When ``n_trades`` and ``reference_trades`` are provided, apply a
    Sharpe-like scaling so that configurations with more high-quality
    trades are rewarded, while extremely sparse or excessively dense
    configurations do not dominate purely by count:

        Edge = Edge_base × sqrt(min(n_trades, 4×reference_trades) / reference_trades)

    ENHANCED: When days_span is provided, add an explicit density bonus
    for configurations achieving the target trades/day range (1-3/day),
    to help HPO escape local minima of very sparse configurations.

    Args:
        mean_return_positive: Mean return of positive-labeled events
        mean_auc: Cross-validated AUC of the model
        transaction_cost: Transaction cost per trade
        n_trades: Number of labeled events/trades used for this edge
        reference_trades: Reference trade count for scaling (e.g. days_span × target_trades_per_day)
        days_span: Number of days in the dataset (for density bonus)
        target_trades_per_day: Target trades/day for density bonus calculation

    Returns:
        Realistic P&L edge score
    """
    # Capture ratio: how much of theoretical profit we actually capture
    # Clamped to [0, 1] to prevent negative edge from AUC < 0.5
    capture_ratio = max(0.0, (2 * mean_auc) - 1)

    # Net profitability after costs
    net_profit = mean_return_positive - transaction_cost

    # Base edge = net profit × capture ratio
    edge = net_profit * capture_ratio

    # Optional Sharpe-like scaling by number of trades when a
    # reasonable reference count is provided. This nudges the
    # optimizer towards configurations that generate a healthy number
    # of good trades, without letting sheer trade count dominate.
    if (
        n_trades is not None
        and reference_trades is not None
        and reference_trades > 0
        and n_trades > 0
    ):
        # Cap effective trade count at 4× reference to avoid runaway
        # scaling for extremely dense configurations.
        effective_trades = min(float(n_trades), float(reference_trades) * 4.0)
        trade_factor = float(np.sqrt(effective_trades / float(reference_trades)))
        edge *= trade_factor

    # NEW: Explicit density bonus to encourage configurations with
    # healthy trade counts in the target range (1-3 trades/day).
    # This helps HPO escape local minima of very sparse configs.
    if days_span is not None and days_span > 0 and n_trades is not None and n_trades > 0:
        trades_per_day = float(n_trades) / float(days_span)
        
        # Density bonus: reward configurations in the sweet spot (1-3 trades/day)
        # Peak bonus at target_trades_per_day, with gentle falloff outside range
        if trades_per_day >= 0.5 and trades_per_day <= 5.0:
            # Gaussian-like bonus centered at target
            distance_from_target = abs(trades_per_day - target_trades_per_day)
            # Max 20% bonus at target, decaying with distance
            density_bonus = 1.0 + 0.2 * max(0.0, 1.0 - distance_from_target / 2.0)
            edge *= density_bonus
        elif trades_per_day < 0.5:
            # Strong penalty for very sparse configs (< 0.5/day)
            sparsity_penalty = 0.5 + 0.5 * (trades_per_day / 0.5)  # Range [0.5, 1.0]
            edge *= sparsity_penalty

    return edge


class MetaLabelingHPOExperimentStep(BaseStep):
    """Offline HPO step to optimize labeling parameters.

    This step does *not* run as part of standard training. It must be
    invoked explicitly (e.g. via the launcher). It reuses the existing
    labeling utilities but searches over a small parameter space:

    - Event definition / TPSL (profit threshold, stop ratio, horizon 2–12,
      min spacing)
    - Target transformation (symmetric probability clipping and
      symmetric quantile clipping of target magnitudes)

    The objective focuses on label quality and economic separation
    between positive/negative labels using realized returns.
    """

    def __init__(self, step_name: str = "meta_labeling_hpo_experiment") -> None:
        super().__init__(step_name, use_versioned_artifacts=False)
        self.logger = logger

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run hierarchical HPO over labeling parameters.

        Config keys (non-exhaustive):
        - symbol, exchange, timeframe: market context
        - enable_labeling_hpo: if False, step exits early
        - execution_mode: 'full'/'light'/'blank' for data loading scope
        """
        if not config.get("enable_labeling_hpo", True):
            tprint("ℹ️ Labeling HPO disabled via config.enable_labeling_hpo", "INFO")
            return {"success": True, "metrics": {}, "artifacts": {}, "skipped": True}

        symbol = config.get("symbol", "ETHUSDT")
        exchange = config.get("exchange", "binance")
        timeframe = config.get("timeframe", "15m")

        tprint_info(
            f"🚀 Starting Meta-Labeling HPO experiment for {symbol}/{exchange} [{timeframe}]"
        )

        # ------------------------------------------------------------------
        # 1) Load market data once and generate primary signals
        # ------------------------------------------------------------------
        pipeline_state: Dict[str, Any] = {}
        market_data, source = self.load_market_data_or_fail(
            config,
            pipeline_state,
            allow_config_override=True,
            skip_artifacts=True,
        )

        if not isinstance(market_data, pd.DataFrame) or market_data.empty:
            msg = "❌ No market data available for labeling HPO"
            tprint(msg, "ERROR")
            return {"success": False, "error": msg, "metrics": {}, "artifacts": {}}

        # Honour centralized lookback_days from the launcher for full/blank modes
        # by trimming to the last N days on the DateTimeIndex. Light-mode behavior
        # (shorter windows) is handled via BaseStep._apply_light_mode_filter.
        try:
            exec_mode = str(config.get("execution_mode", "full")).lower()
            lookback_days = int(config.get("lookback_days", 0) or 0)
            if (
                lookback_days > 0
                and exec_mode in {"full", "blank"}
                and isinstance(market_data.index, pd.DatetimeIndex)
            ):
                end_ts = market_data.index.max()
                start_ts = end_ts - pd.Timedelta(days=lookback_days)
                orig_rows = len(market_data)
                try:
                    orig_span_days = max(
                        1,
                        (market_data.index.max() - market_data.index.min()).days,
                    )
                except Exception:
                    orig_span_days = -1

                mask = market_data.index >= start_ts
                if int(mask.sum()) > 0:
                    market_data = market_data.loc[mask].copy()
                    try:
                        new_span_days = max(
                            1,
                            (market_data.index.max() - market_data.index.min()).days,
                        )
                    except Exception:
                        new_span_days = -1
                    tprint_info(
                        f"⏱️ HPO lookback alignment: mode={exec_mode}, "
                        f"requested={lookback_days}d, span {orig_span_days}d → {new_span_days}d, "
                        f"rows {orig_rows}→{len(market_data)}",
                    )
                else:
                    tprint_warning(
                        f"⚠️ HPO lookback alignment requested {lookback_days}d but no data "
                        f"falls in that window; keeping original dataset (rows={orig_rows})",
                    )
        except Exception as lb_exc:
            tprint_warning(
                f"⚠️ Failed to apply lookback_days to HPO market_data; proceeding with raw window: {lb_exc}",
            )

        # Attach rolling HMM regimes (typically 1h) to the market_data frame so that
        # regime-aware features and thresholds can be evaluated during HPO.
        try:
            regime_cfg = dict(config)
            if "regime_timeframe" not in regime_cfg:
                regime_cfg["regime_timeframe"] = "1h"
            market_data = attach_rolling_hmm_regimes_to_market_data(
                self,
                market_data,
                regime_cfg,
            )
        except Exception as e_reg:
            tprint_warning(f"⚠️ Failed to attach rolling HMM regimes to market_data for HPO: {e_reg}")

        # Attach specialist liquidity regime probabilities as additional regime features
        try:
            tprint_info("💧 Attempting to attach liquidity regime probabilities for HPO via specialist loader...")

            config_for_specialists = dict(config)
            config_for_specialists.setdefault("use_canonical_specialist_scalars", True)

            specialist_df = get_specialist_models_outputs(
                artifact_router=self.artifact_router,
                training_index=market_data.index,
                config=config_for_specialists,
                logger=self.logger,
                strict=False,
            )

            if specialist_df is not None and not specialist_df.empty:
                prob_cols = [
                    c for c in specialist_df.columns
                    if c.startswith('liquidity_regime_') and 'prob_' in c
                ]

                if prob_cols:
                    liquidity_features = specialist_df[prob_cols].reindex(market_data.index, method='ffill')
                    tprint_info(f"   ↪ Selected {len(prob_cols)} liquidity regime probability columns: {prob_cols}")

                    for col in liquidity_features.columns:
                        market_data[f'liquidity_{col}'] = liquidity_features[col]

                    tprint_success(f"✅ Added {len(prob_cols)} liquidity regime probability features to market_data")
                else:
                    tprint_warning("⚠️ No liquidity regime probability columns found in specialist outputs for HPO")
            else:
                tprint_warning("⚠️ No specialist liquidity regime outputs found for HPO")

        except Exception as e_liquidity:
            tprint_warning(f"⚠️ Failed to attach specialist liquidity regime probabilities for HPO: {e_liquidity}")

        tprint_info(f"📊 Loaded market data from: {source} | rows={len(market_data)}")

        # Generate primary consensus signals using the production helper
        tprint_info("⚙️ Generating primary signals for HPO labeling runs…")
        primary_signals = generate_primary_signals(market_data.copy())

        # Precompute volatility for Kalman smoothing and label normalization
        # IMPORTANT: NO FUTURE LEAKAGE - using backward-looking rolling window only
        # At time T, volatility_1d[T] uses returns from T-96 to T-1 (past data only)
        # This ensures labels at time T do not peek at future volatility
        log_ret = np.log(market_data["close"]).diff()
        volatility_1d = log_ret.rolling(96).std()
        
        # Verify no NaN at start could cause issues (first 96 bars will have partial window)
        n_valid_vol = int((~volatility_1d.isna()).sum())
        tprint_info(f"📊 Volatility computed: {n_valid_vol}/{len(volatility_1d)} valid values (backward-looking, no future leakage)")

        # Precompute span in days once (used by density penalty in objective)
        try:
            days_span = max(
                1,
                (market_data.index.max() - market_data.index.min()).days,
            )
        except Exception:
            days_span = 1

        stage1_enable_subsample = bool(config.get("stage1_enable_subsample", True))
        lookback_days_cfg = int(config.get("lookback_days", 0) or 0)
        if lookback_days_cfg > 0:
            default_stage1_window = min(days_span, lookback_days_cfg)
        else:
            default_stage1_window = days_span
        stage1_subsample_window_days = int(
            config.get("stage1_subsample_window_days", default_stage1_window)
        )

        # Precompute ATR for trailing profit simulation
        # Using 14-period True Range as standard
        high_prices = market_data["high"] if "high" in market_data.columns else market_data["close"]
        low_prices = market_data["low"] if "low" in market_data.columns else market_data["close"]
        close_prices = market_data["close"]

        tr1 = high_prices - low_prices
        tr2 = (high_prices - close_prices.shift(1)).abs()
        tr3 = (low_prices - close_prices.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Using 14-period rolling mean for ATR
        atr_series = true_range.rolling(window=14, min_periods=1).mean()

        stage1_market_data = market_data
        stage1_primary_signals = primary_signals
        stage1_volatility_1d = volatility_1d
        stage1_atr_series = atr_series
        stage1_days_span = days_span

        if stage1_enable_subsample:
            try:
                start_ts = market_data.index.min()
                end_ts = start_ts + pd.Timedelta(days=stage1_subsample_window_days)
                mask = (market_data.index >= start_ts) & (market_data.index <= end_ts)
                if int(mask.sum()) > 0:
                    stage1_market_data = market_data.loc[mask].copy()
                    stage1_primary_signals = primary_signals.loc[mask].copy()
                    stage1_volatility_1d = volatility_1d.loc[mask].copy()
                    stage1_atr_series = atr_series.loc[mask].copy()
                    try:
                        stage1_days_span = max(
                            1,
                            (stage1_market_data.index.max() - stage1_market_data.index.min()).days,
                        )
                    except Exception:
                        stage1_days_span = days_span
                else:
                    stage1_enable_subsample = False
            except Exception:
                stage1_enable_subsample = False
                stage1_market_data = market_data
                stage1_primary_signals = primary_signals
                stage1_volatility_1d = volatility_1d
                stage1_atr_series = atr_series
                stage1_days_span = days_span

        # Build simple arrays for the optimizer API (they are not used in
        # the objective itself but provide shapes/logging)
        X_dummy = market_data[["close"]].dropna().values.astype("float32")
        y_dummy = np.zeros(len(X_dummy), dtype="float32")

        # ------------------------------------------------------------------
        # 2) Define parameter groups for hierarchical HPO
        # ------------------------------------------------------------------
        param_groups = [
            # Group 1: Signal Structure (3 params)
            create_param_group(
                name="signal_structure",
                params={
                    "cusum_threshold": {
                        "type": "float",
                        "low": 0.008,
                        "high": 0.025,
                    },
                    "target_signal_density": {
                        "type": "float",
                        "low": 3.0,
                        "high": 10.0,
                    },
                    "min_event_spacing": {
                        "type": "int",
                        "low": 1,
                        "high": 3,
                    },
                },
                priority=1,
                description="Signal generation and event spacing",
            ),
            # Group 2: Event Geometry (4 params) - Depends on Signal Structure
            create_param_group(
                name="event_geometry",
                params={
                    "horizon_bars": {
                        "type": "int",
                        "low": 8,
                        "high": 56,
                        "step": 2,
                    },
                    "profit_thr_base": {
                        "type": "float",
                        "low": 0.008,
                        "high": 0.022,
                    },
                    "stop_to_profit_ratio": {
                        "type": "float",
                        "low": 0.3,
                        "high": 0.67,
                    },
                    "trail_distance": {
                        "type": "float",
                        "low": 0.6,
                        "high": 1.2,
                    },
                },
                priority=2,
                depends_on=["signal_structure"],
                description="Triple-barrier shape and trailing stop logic",
            ),
            # Group 3: Volatility Adaptation (5 params) - Depends on Event Geometry
            create_param_group(
                name="volatility_adaptation",
                params={
                    "vol_baseline_window": {
                        "type": "int",
                        "low": 48,
                        "high": 192,
                    },
                    "profit_mult_min": {
                        "type": "float",
                        "low": 0.7,
                        "high": 1.0,
                    },
                    "profit_mult_max": {
                        "type": "float",
                        "low": 1.0,
                        "high": 2.0,
                    },
                    "stop_mult_min": {
                        "type": "float",
                        "low": 0.5,
                        "high": 1.0,
                    },
                    "stop_mult_max": {
                        "type": "float",
                        "low": 1.0,
                        "high": 1.5,
                    },
                },
                priority=3,
                depends_on=["event_geometry"],
                description="Volatility adaptation baseline and multipliers",
            ),
            # Group 4: Label Definition (3 params) - Depends on Volatility Adaptation
            create_param_group(
                name="label_definition",
                params={
                    "label_low_q": {
                        "type": "float",
                        "low": 0.10,
                        "high": 0.50,
                    },
                    "label_high_q": {
                        "type": "float",
                        "low": 0.50,
                        "high": 0.90,
                    },
                    "econ_min_return_multiple": {
                        "type": "float",
                        "low": 1.0,
                        "high": 2.5,
                    },
                },
                priority=4,
                depends_on=["volatility_adaptation"],
                description="Quantile-based label definition",
            ),
            # Group 5: Target Engineering (5 params) - Depends on Label Definition
            create_param_group(
                name="target_engineering",
                params={
                    "iso_min_prob": {
                        "type": "float",
                        "low": 0.05,
                        "high": 0.15,
                    },
                    "target_clip_high_q": {
                        "type": "float",
                        "low": 0.90,
                        "high": 0.98,
                    },
                    "signal_strength_scale_max": {
                        "type": "float",
                        "low": 1.2,
                        "high": 2.0,
                    },
                    "r_multiple_pos_threshold": {
                        "type": "float",
                        "low": 0.3,
                        "high": 1.0,
                    },
                    "transaction_cost_mult": {
                        "type": "float",
                        "low": 0.5,
                        "high": 1.5,
                    },
                },
                priority=5,
                depends_on=["label_definition"],
                description="Target transformation and trade filters",
            ),
            # Group 6: Smoothing (2 params) - Independent/Late stage
            create_param_group(
                name="kalman_smoothing",
                params={
                    "kalman_Q": {
                        "type": "float",
                        "low": 1e-5,
                        "high": 1e-3,
                        "log": True,
                    },
                    "kalman_R": {
                        "type": "float",
                        "low": 1e-3,
                        "high": 0.1,
                        "log": True,
                    },
                },
                priority=6,
                depends_on=["event_geometry"],
                description="Kalman smoothing noise parameters",
            ),
        ]

        warm_start_best_params: Dict[str, Any] = {}
        warm_start_candidates_df: Optional[pd.DataFrame] = None
        outcomes_dir = Path("outcomes")
        try:
            json_pattern = f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_*.json"
            json_paths = sorted(outcomes_dir.glob(json_pattern))
            if json_paths:
                latest_json = json_paths[-1]
                with open(latest_json, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        best_params_data = data.get("best_params", {})
                        if isinstance(best_params_data, dict):
                            warm_start_best_params = best_params_data
        except Exception:
            warm_start_best_params = {}
        try:
            csv_pattern = f"meta_labeling_hpo_candidate_pool_{symbol}_{timeframe}_*.csv"
            csv_paths = sorted(outcomes_dir.glob(csv_pattern))
            if csv_paths:
                latest_csv = csv_paths[-1]
                warm_start_candidates_df = pd.read_csv(latest_csv)
        except Exception:
            warm_start_candidates_df = None

        calibrated_horizon: Optional[int] = None
        if stage1_enable_subsample:
            def _evaluate_horizon_candidate(h: int) -> Dict[str, float]:
                (
                    realized_returns_h, 
                    binary_labels_h, 
                    exit_reasons_h, 
                    event_durations_h, 
                    mfe_h, 
                    mae_h,
                    _binary_labels_long_h,  # Not used in HPO scoring
                    _binary_labels_short_h,  # Not used in HPO scoring
                ) = compute_realized_returns(
                    stage1_market_data,
                    stage1_primary_signals,
                    profit_threshold=float(warm_start_best_params.get("profit_thr_base", 0.012)),
                    stop_threshold=float(warm_start_best_params.get("profit_thr_base", 0.012)) * float(warm_start_best_params.get("stop_to_profit_ratio", 0.5)),
                    horizon=int(h),
                    transaction_cost=DEFAULT_TRANSACTION_COST,
                    min_event_spacing=int(warm_start_best_params.get("min_event_spacing", 2)),
                    atr_series=stage1_atr_series,
                    trail_distance_atr_mult=float(warm_start_best_params.get("trail_distance", 0.0)),
                )
                labeled_mask_h = ~binary_labels_h.isna()
                n_events_h = int(labeled_mask_h.sum())
                if n_events_h <= 0 or stage1_days_span <= 0:
                    return {
                        "trades_per_day": 0.0,
                        "risk_reward": 0.0,
                        "profit_potential": 0.0,
                    }
                trades_per_day_h = n_events_h / float(stage1_days_span)
                returns_labeled_h = realized_returns_h[labeled_mask_h].dropna()
                if len(returns_labeled_h) == 0:
                    return {
                        "trades_per_day": trades_per_day_h,
                        "risk_reward": 0.0,
                        "profit_potential": 0.0,
                    }
                labels_labeled_h = binary_labels_h[labeled_mask_h]
                r_pos_h = returns_labeled_h[labels_labeled_h == 1]
                r_neg_h = returns_labeled_h[labels_labeled_h == 0]
                mean_pos_h = float(r_pos_h.mean()) if len(r_pos_h) > 0 else 0.0
                mean_neg_h = float(r_neg_h.mean()) if len(r_neg_h) > 0 else 0.0
                mean_loss_h = abs(mean_neg_h) if mean_neg_h < 0 else (abs(float(r_neg_h.mean())) if len(r_neg_h) > 0 else 0.0)
                mean_win_h = mean_pos_h
                risk_reward_h = mean_win_h / (mean_loss_h + 1e-8) if mean_loss_h > 0 else 0.0
                mean_return_h = float(returns_labeled_h.mean()) if len(returns_labeled_h) > 0 else 0.0
                profit_potential_h = trades_per_day_h * mean_return_h
                return {
                    "trades_per_day": trades_per_day_h,
                    "risk_reward": risk_reward_h,
                    "profit_potential": profit_potential_h,
                }
            try:
                event_group = param_groups[0]
                horizon_spec = event_group.params.get("horizon_bars", {})
                h_low = int(horizon_spec.get("low", 8))
                h_high = int(horizon_spec.get("high", 56))
                h_step = int(horizon_spec.get("step", 2)) or 1
                candidate_horizons = list(range(h_low, h_high + 1, h_step))
                best_h = None
                best_potential = float("-inf")
                best_trades = 0.0
                best_rr = 0.0
                for h in candidate_horizons:
                    metrics_h = _evaluate_horizon_candidate(h)
                    trades_h = metrics_h["trades_per_day"]
                    rr_h = metrics_h["risk_reward"]
                    # RELAXED: Allow wider trade density range (0.2 - 6.0/day)
                    # to avoid prematurely filtering out potentially good horizons
                    if trades_h < 0.2 or trades_h > 6.0:
                        continue
                    # RELAXED: Lower R/R threshold (1.0 instead of 1.2)
                    # since HPO will refine the TPSL params later
                    if rr_h < 1.0:
                        continue
                    if metrics_h["profit_potential"] > best_potential:
                        best_potential = metrics_h["profit_potential"]
                        best_h = h
                        best_trades = trades_h
                        best_rr = rr_h
                if best_h is None and candidate_horizons:
                    best_h = int(np.median(candidate_horizons))
                calibrated_horizon = int(best_h) if best_h is not None else None
                if calibrated_horizon is not None:
                    tprint_info(
                        f"📏 Using calibrated horizon_bars={calibrated_horizon} on subsample (trades_per_day≈{best_trades:.3f}, rr≈{best_rr:.3f})"
                    )
            except Exception:
                calibrated_horizon = None
        else:
            calibrated_horizon = None

        # Storage for candidate label configurations
        candidate_pool: List[Dict[str, Any]] = []

        # Lightweight debug sampling for objective diagnostics
        debug_sample_limit = 50
        debug_sample_count = 0

        # ------------------------------------------------------------------
        # 3) Define objective function for labeling quality (with learnability)
        # ------------------------------------------------------------------

        def labeling_objective(
            params: Dict[str, Any],
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
            model: Any | None = None,
            cv_folds: int = 1,
            scoring_metric: str = "custom_balanced_score",
            model_complexity: str = "fast",  # NEW: Model complexity level
            use_ensemble: bool = False,  # NEW: Use ensemble for strong complexity
            compute_diagnostics: bool = False,  # NEW: Whether to compute underfit diagnostics
            **kwargs: Any,
        ) -> Dict[str, float]:
            """Evaluate one labeling configuration with multi-objective scoring.

            This function:
            - Recomputes realized returns & binary labels with candidate TPSL parameters
            - Smooths labels via Kalman filter
            - Creates meta-features for learnability assessment
            - Computes learnability score with isotonic calibration (cross-validated AUC)
            - Computes realistic P&L edge metric
            - Applies regularization checks (temporal stability, regime consistency)
            - Optionally computes underfit diagnostics
            - Returns dict of objectives for Pareto frontier

            Args:
                model_complexity: "fast", "medium", or "strong" - controls model capacity
                use_ensemble: Whether to use model ensemble (for strong complexity)
                compute_diagnostics: Whether to compute underfit diagnostics

            Returns:
                Dict with keys: 'learnability', 'profitability', 'combined', 'edge'
            """
            # Determine CV splits based on model complexity
            cv_splits_map = {"fast": 3, "medium": 4, "strong": 5}
            cv_splits = cv_splits_map.get(model_complexity, 3)

            try:
                nonlocal debug_sample_count
                # Enforce profit >= 1.5x stop constraint. During stages that do not
                # actively optimize profit_thr_base/stop_to_profit_ratio, fall back
                # to conservative defaults.
                profit_thr_base = float(params.get("profit_thr_base", 0.012))
                stop_ratio = float(params.get("stop_to_profit_ratio", 0.5))
                trail_dist = float(params.get("trail_distance", 0.0))

                # CONSTRAINT: Ensure profit is at least 1.5x stop
                stop_thr_base = max(0.0005, profit_thr_base * stop_ratio)
                if profit_thr_base < 1.5 * stop_thr_base:
                    tprint_warning(f"⚠️ Config rejected: profit {profit_thr_base:.4f} < 1.5x stop {stop_thr_base:.4f}")
                    return {
                        'learnability': 0.0,
                        'profitability': -1e9,
                        'edge': -1e9,
                        'combined': -1e9,
                    }

                # Extract parameters
                horizon = int(params["horizon_bars"])
                min_spacing = int(params["min_event_spacing"])

                kalman_Q = float(params.get("kalman_Q", 1e-4))
                kalman_R = float(params.get("kalman_R", 0.01))
                vol_baseline_window = int(params.get("vol_baseline_window", 96))
                profit_mult_min = float(params.get("profit_mult_min", 0.5))
                profit_mult_max = float(params.get("profit_mult_max", 2.0))
                stop_mult_min = float(params.get("stop_mult_min", 0.5))
                stop_mult_max = float(params.get("stop_mult_max", 2.0))

                # Enforce horizon is in 8-56 range with steps of 2
                horizon = max(8, min(56, horizon))
                if horizon % 2 != 0:
                    horizon = (horizon // 2) * 2  # Round down to even
                min_spacing = max(1, min(16, min_spacing))
                vol_baseline_window = max(8, min(512, vol_baseline_window))

                if profit_mult_min > profit_mult_max:
                    profit_mult_min, profit_mult_max = profit_mult_max, profit_mult_min
                if stop_mult_min > stop_mult_max:
                    stop_mult_min, stop_mult_max = stop_mult_max, stop_mult_min

                # Hard RR constraint: even in the worst-case (smallest profit, largest stop),
                # require a minimum RR ~1.4 (≈1.25 net after fees).
                worst_rr = (profit_thr_base * profit_mult_min) / max(stop_thr_base * stop_mult_max, 1e-8)
                if worst_rr < 1.4:
                    return {
                        'learnability': 0.0,
                        'profitability': -1e9,
                        'edge': -1e9,
                        'combined': -1e9,
                    }

                # Use safe defaults when target_transform params are not part of the current group
                iso_min_prob = float(params.get("iso_min_prob", 0.05))
                # Allow slightly stronger clipping on both tails; keep symmetric band.
                iso_min_prob = max(0.05, min(0.15, iso_min_prob))
                iso_max_prob = 1.0 - iso_min_prob
                iso_max_prob = max(0.85, min(1.0, iso_max_prob))

                q_high = float(params.get("target_clip_high_q", 0.95))
                q_high = max(0.90, min(0.98, q_high))
                q_low = max(0.0, min(0.5, 1.0 - q_high))

                # Economic floor multiplier for vol-scaled labels and isotonic mapping
                econ_min_mult = float(params.get("econ_min_return_multiple", ECON_MIN_RETURN_MULTIPLE))
                if not np.isfinite(econ_min_mult) or econ_min_mult <= 0:
                    econ_min_mult = float(ECON_MIN_RETURN_MULTIPLE)

                # Label quantile thresholds (regime-aware when regimes are present).
                label_low_q = float(params.get("label_low_q", 0.30))
                label_high_q = float(params.get("label_high_q", 0.70))
                # Guard-rail: ensure a proper ordering and keep them away from extremes.
                # WIDENED bounds to allow denser label configurations
                label_low_q = max(0.10, min(0.50, label_low_q))
                label_high_q = max(0.50, min(0.90, label_high_q))
                if label_high_q <= label_low_q:
                    label_low_q, label_high_q = 0.30, 0.70

                # NEW: R-multiple threshold for labeling - controls trade velocity filter
                r_multiple_threshold = float(params.get("r_multiple_pos_threshold", 0.7))
                r_multiple_threshold = max(0.3, min(1.0, r_multiple_threshold))

                # NEW: Transaction cost multiplier - allows HPO to explore cost sensitivity
                tx_cost_mult = float(params.get("transaction_cost_mult", 1.0))
                tx_cost_mult = max(0.5, min(1.5, tx_cost_mult))
                effective_tx_cost = DEFAULT_TRANSACTION_COST * tx_cost_mult

                # NEW: Signal generation parameters
                cusum_threshold = float(params.get("cusum_threshold", 0.015))
                cusum_threshold = max(0.008, min(0.025, cusum_threshold))
                target_signal_density = float(params.get("target_signal_density", 4.0))
                target_signal_density = max(3.0, min(10.0, target_signal_density))

                # --- Recompute realized returns ---
                # NO FUTURE LEAKAGE in volatility-based thresholds:
                # - volatility_1d is backward-looking (rolling 96-bar std)
                # - vol_baseline is backward-looking (rolling mean of past volatility)
                # - vol_factor at time T uses only volatility from T-vol_baseline_window to T
                vol_baseline = volatility_1d.rolling(vol_baseline_window).mean()
                vol_factor = volatility_1d / (vol_baseline + 1e-8)

                high_prices = market_data["high"] if "high" in market_data.columns else market_data["close"]
                low_prices = market_data["low"] if "low" in market_data.columns else market_data["close"]
                close_prices = market_data["close"]

                tr1 = high_prices - low_prices
                tr2 = (high_prices - close_prices.shift(1)).abs()
                tr3 = (low_prices - close_prices.shift(1)).abs()
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

                trend_atr_window = int(config.get("trend_strength_atr_window", 14))
                atr_series = true_range.rolling(window=trend_atr_window, min_periods=1).mean()

                trend_delta_lookback = int(config.get("trend_strength_delta_lookback", 4))
                price_delta = close_prices.diff(trend_delta_lookback).abs()

                trend_strength = (price_delta / (atr_series + 1e-8)).replace([np.inf, -np.inf], np.nan)
                trend_strength = trend_strength.clip(
                    lower=0.0,
                    upper=float(config.get("trend_strength_clip", 5.0)),
                ).fillna(0.0)

                trend_alpha = float(config.get("trend_strength_alpha_profit", 0.5))
                trend_beta = float(config.get("trend_strength_beta_stop", 0.5))

                profit_factor = 1.0 + trend_alpha * trend_strength
                stop_factor = 1.0 + trend_beta * trend_strength

                adaptive_profit = profit_thr_base * vol_factor * profit_factor
                adaptive_stop = stop_thr_base * vol_factor * stop_factor
                adaptive_profit = adaptive_profit.clip(
                    lower=profit_thr_base * profit_mult_min,
                    upper=profit_thr_base * profit_mult_max,
                )
                adaptive_stop = adaptive_stop.clip(
                    lower=stop_thr_base * stop_mult_min,
                    upper=stop_thr_base * stop_mult_max,
                )

                # NEW: Regenerate primary signals if cusum_threshold differs from default
                # This allows HPO to explore different signal densities
                signals_to_use = primary_signals
                default_cusum = 0.015
                if abs(cusum_threshold - default_cusum) > 0.001:
                    try:
                        signals_to_use = generate_primary_signals(
                            market_data.copy(),
                            cusum_threshold=cusum_threshold,
                            target_trades_per_day=target_signal_density,
                        )
                    except Exception:
                        signals_to_use = primary_signals  # Fallback if regeneration fails

                (
                    realized_returns,
                    binary_labels,
                    exit_reasons,
                    event_durations,
                    mfe_series,
                    mae_series,
                    _binary_labels_long,  # Not used in HPO objective function
                    _binary_labels_short,  # Not used in HPO objective function
                ) = compute_realized_returns(
                    market_data,
                    signals_to_use,
                    profit_threshold=adaptive_profit,
                    stop_threshold=adaptive_stop,
                    horizon=horizon,
                    transaction_cost=effective_tx_cost,  # Use HPO-tunable tx cost
                    min_event_spacing=min_spacing,
                    atr_series=atr_series,
                    trail_distance_atr_mult=trail_dist,
                )

                # Basic diagnostics on raw realized returns and labels before
                # any vol-scaling or quantile-based relabeling.
                n_raw_events = len(realized_returns)
                n_raw_labeled = int((~binary_labels.isna()).sum())
                if debug_sample_count < debug_sample_limit:
                    tprint_info(
                        f"[HPO_DEBUG_LABELS] raw_events={n_raw_events}, raw_labeled={n_raw_labeled}, "
                        f"profit_thr_base={profit_thr_base:.6f}, stop_thr_base={stop_thr_base:.6f}, "
                        f"econ_min_mult={econ_min_mult:.3f}, label_low_q={label_low_q:.3f}, label_high_q={label_high_q:.3f}",
                    )

                # Replace legacy R-multiple based labels with quantile-based labels
                # derived from volatility-scaled realized returns, to improve label
                # balance and economic relevance in HPO scoring.
                #
                # NO FUTURE VOLATILITY LEAKAGE:
                # - volatility_1d at time T uses only data from T-96 to T-1 (backward-looking)
                # - realized_returns at time T uses future prices (expected for labeling)
                # - vol_scaled = realized_returns / past_volatility (no future vol leakage)
                #
                # NOTE: Quantile thresholds are computed from ALL training data, which is
                # acceptable for HPO/training. In production, use expanding/rolling quantiles.
                vol_scaled_returns = compute_vol_scaled_returns_for_events(
                    realized_returns=realized_returns,
                    volatility=volatility_1d,
                    econ_min_return_multiple=econ_min_mult,
                )

                n_vol_scaled_events = int(vol_scaled_returns.dropna().size)
                if debug_sample_count < debug_sample_limit:
                    tprint_info(
                        f"[HPO_DEBUG_LABELS] vol_scaled_events={n_vol_scaled_events}",
                    )

                # Decide whether to use regime-aware quantiles based on the
                # attached HMM regimes (typically 1h) on market_data.
                regimes_for_labeling = None
                if config.get("enable_regime_aware_quantiles", True) and "hmm_regime_label_1h" in market_data.columns:
                    regimes_for_labeling = market_data["hmm_regime_label_1h"]

                def _make_quantile_labels(vol_scaled_series: pd.Series) -> pd.Series:
                    """Helper to create (regime-aware) quantile labels from a score series."""
                    if regimes_for_labeling is not None:
                        return create_regime_aware_quantile_labels_from_vol_scaled_returns(
                            vol_scaled=vol_scaled_series,
                            regimes=regimes_for_labeling,
                            low_q=label_low_q,
                            high_q=label_high_q,
                        )
                    return create_quantile_labels_from_vol_scaled_returns(
                        vol_scaled=vol_scaled_series,
                        low_q=label_low_q,
                        high_q=label_high_q,
                    )

                # Primary quantile labels on vol-scaled returns with the
                # HPO-chosen economic floor.
                quantile_labels = _make_quantile_labels(vol_scaled_returns)
                binary_labels = quantile_labels

                n_quantile_non_nan = int((~quantile_labels.isna()).sum())
                unique_quantile_vals: list[int] = []
                if n_quantile_non_nan > 0:
                    try:
                        unique_quantile_vals = sorted(
                            pd.unique(quantile_labels.dropna().astype(int)).tolist()
                        )
                    except Exception:
                        unique_quantile_vals = []
                if debug_sample_count < debug_sample_limit:
                    tprint_info(
                        f"[HPO_DEBUG_LABELS] quantile_labels_non_nan={n_quantile_non_nan}, "
                        f"unique_labels={unique_quantile_vals}",
                    )

                # Guard: configurations that produce no labeled events at all are
                # rejected outright; sparse but non-zero densities are handled via
                # softer density penalties further down.
                labeled_mask = ~binary_labels.isna()
                n_events = int(labeled_mask.sum())
                events_per_day = n_events / max(days_span, 1)
                if n_events == 0:
                    if debug_sample_count < debug_sample_limit:
                        tprint_warning(
                            f"[HPO_DEBUG_LABELS] rejecting config with zero labeled events: "
                            f"raw_labeled={n_raw_labeled}, vol_non_nan={n_vol_scaled_events}, "
                            f"quantile_non_nan={n_quantile_non_nan}",
                        )
                    tprint_warning(
                        f"⚠️ HPO config produced zero labeled events (n={n_events}, {events_per_day:.3f} events/day), rejecting",
                    )
                    return -1e9

                # Lightweight density diagnostics for the first few configs:
                # report days_span and events/day so we can tune label density
                # targets and event filters more precisely.
                if debug_sample_count < debug_sample_limit:
                    tprint_info(
                        f"[HPO_DEBUG_DENSITY] days_span={days_span}, n_events={n_events}, "
                        f"events_per_day={events_per_day:.3f}",
                    )

                # --- Time-to-Outcome (TTO) metrics for constraints & penalties ---
                mean_tto = float("nan")
                timeout_rate = float("nan")
                tto_penalty = 0.0
                try:
                    if horizon > 0 and isinstance(event_durations, pd.Series):
                        event_mask_tto = labeled_mask & ~event_durations.isna()
                        if event_mask_tto.any():
                            tto_series = (event_durations[event_mask_tto] / float(horizon)).replace([np.inf, -np.inf], np.nan)
                            if len(tto_series) > 0:
                                mean_tto = float(tto_series.mean())

                        if exit_reasons is not None and isinstance(exit_reasons, pd.Series):
                            exit_events = exit_reasons[event_mask_tto]
                            timeout_rate = float((exit_events == 2).mean())

                    # Hard constraint on mean TTO to avoid pathologically slow exits
                    tto_hard_max = float(config.get("tto_max", 0.6))
                    if np.isfinite(mean_tto) and mean_tto > tto_hard_max:
                        tprint_warning(
                            f"⚠️ HPO config rejected due to high mean TTO={mean_tto:.3f} (> {tto_hard_max:.2f})"
                        )
                        return {
                            'learnability': 0.0,
                            'profitability': -1e9,
                            'edge': -1e9,
                            'combined': -1e9,
                        }

                    # Soft TTO penalty (secondary importance): gently discourage
                    # configurations with mean TTO above a target.
                    if np.isfinite(mean_tto):
                        tto_target = float(config.get("tto_target", 0.4))
                        tto_penalty_weight = float(config.get("tto_penalty_weight", 20.0))
                        tto_excess = max(0.0, mean_tto - tto_target)
                        tto_penalty = tto_excess * tto_penalty_weight
                except Exception:
                    mean_tto = float("nan")
                    timeout_rate = float("nan")
                    tto_penalty = 0.0

                # --- Kalman smoothing for meta probability proxy ---
                smoothed_labels, _ = kalman_smooth_labels(
                    binary_labels,
                    Q=kalman_Q,
                    R=kalman_R,
                    volatility=volatility_1d,
                )

                # Probabilities limited to [0, 1]
                prob_series = smoothed_labels.clip(0.0, 1.0)

                # Symmetric clipping before isotonic regression
                prob_clipped = prob_series.clip(iso_min_prob, iso_max_prob)

                # Fit probability→expected-return mapping on labeled events
                iso_reg = fit_probability_to_return_mapping(
                    probabilities=prob_clipped.values,
                    realized_returns=realized_returns.values,
                    method="isotonic",
                    econ_min_return_multiple=econ_min_mult,
                )

                # Translate to long/short targets using existing helper
                target_long, target_short = translate_to_targets_with_isotonic(
                    realized_returns=realized_returns,
                    probabilities=prob_clipped.values,
                    signals=primary_signals,
                    iso_regressor=iso_reg,
                )

                # Construct a unified target magnitude (as in diagnostics)
                target_mag = pd.Series(0.0, index=market_data.index)
                long_mask_meta = (target_long > 0).reindex(target_mag.index, fill_value=False)
                short_mask_meta = (target_short > 0).reindex(target_mag.index, fill_value=False)
                target_mag[long_mask_meta] = target_long.reindex(target_mag.index)[long_mask_meta]
                target_mag[short_mask_meta] = target_short.reindex(target_mag.index)[short_mask_meta]

                # Quantile clipping of non-zero targets (symmetric tails) and write
                # clipped values back into target_mag so downstream diagnostics see
                # the effect of winsorisation.
                target_nz = target_mag[target_mag > 0]
                if len(target_nz) >= 100:
                    low_val = target_nz.quantile(q_low)
                    high_val = target_nz.quantile(q_high)
                    if low_val < high_val:
                        target_nz_clipped = target_nz.clip(low_val, high_val)
                        target_mag.loc[target_nz_clipped.index] = target_nz_clipped

                # ===== REGRESSION TARGETS (EVENT-ONLY, DIAGNOSTIC) =====
                # Build an event-only, signed regression target from the
                # isotonic-based long/short targets. This is used only for
                # diagnostics and does not directly affect the main edge
                # metric, but provides a clearer view of payoff learnability
                # for regressors.
                try:
                    # Event mask and direction
                    event_mask_reg = labeled_mask.reindex(realized_returns.index, fill_value=False)
                    consensus_dir = primary_signals.get("consensus")
                    if isinstance(consensus_dir, pd.Series):
                        consensus_dir = consensus_dir.reindex(realized_returns.index)

                    if isinstance(consensus_dir, pd.Series):
                        long_event_mask = event_mask_reg & (consensus_dir > 0)
                        short_event_mask = event_mask_reg & (consensus_dir < 0)
                    else:
                        long_event_mask = event_mask_reg
                        short_event_mask = event_mask_reg & False

                    # Signed regression base target: long>0, short<0
                    y_reg_base = pd.Series(np.nan, index=realized_returns.index)
                    if target_long is not None:
                        tl = target_long.reindex(realized_returns.index)
                        y_reg_base.loc[long_event_mask] = tl.loc[long_event_mask]
                    if target_short is not None:
                        ts = target_short.reindex(realized_returns.index)
                        y_reg_base.loc[short_event_mask] = -ts.loc[short_event_mask]

                    # Drop NaNs to derive clipping/scaling statistics
                    y_ev = y_reg_base[event_mask_reg].dropna()

                    if len(y_ev) >= 50:
                        reg_cfg = config.get("regression_targets", {}) or {}
                        clip_low_q = float(reg_cfg.get("clip_low_q", 0.01))
                        clip_high_q = float(reg_cfg.get("clip_high_q", 0.99))

                        clip_low_q = max(0.0, min(0.2, clip_low_q))
                        clip_high_q = max(0.8, min(1.0, clip_high_q))
                        if clip_low_q >= clip_high_q:
                            clip_low_q, clip_high_q = 0.01, 0.99

                        lo = float(y_ev.quantile(clip_low_q))
                        hi = float(y_ev.quantile(clip_high_q))

                        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                            y_reg_base = y_reg_base.clip(lo, hi)
                            y_ev = y_reg_base[event_mask_reg].dropna()

                        # Robust scaling
                        med = float(y_ev.median())
                        q25 = float(y_ev.quantile(0.25))
                        q75 = float(y_ev.quantile(0.75))
                        iqr = q75 - q25
                        scale = iqr if iqr > 1e-8 else float(y_ev.mad()) if hasattr(y_ev, "mad") else 0.0

                        if scale > 0:
                            y_reg_scaled = (y_reg_base - med) / scale
                        else:
                            y_reg_scaled = y_reg_base * 0.0

                        # Store compact diagnostics in candidate_config later
                        # via local variables captured in this scope.
                        reg_target_stats = {
                            "n_events_reg": int(y_ev.shape[0]),
                            "mean_raw": float(y_ev.mean()),
                            "std_raw": float(y_ev.std()),
                            "min_raw": float(y_ev.min()),
                            "max_raw": float(y_ev.max()),
                            "median_raw": float(med),
                        }
                    else:
                        y_reg_scaled = y_reg_base
                        reg_target_stats = None
                except Exception:
                    y_reg_scaled = None
                    reg_target_stats = None

                # ===== LEARNABILITY ASSESSMENT WITH CALIBRATION =====
                # Create meta-features for this labeling configuration using the
                # same pipeline as the production meta-labeling step.
                meta_feature_cfg = config.get("meta_feature_engineering", {})
                volume_available = "volume" in market_data.columns

                meta_features, meta_features_model_processed, selected_feature_names, sample_weights = build_meta_features_for_model(
                    market_data=market_data,
                    primary_signals=primary_signals,
                    realized_returns=realized_returns,
                    binary_labels=binary_labels,
                    event_durations=event_durations,
                    mfe_series=mfe_series,
                    mae_series=mae_series,
                    adaptive_stop_threshold=adaptive_stop,
                    horizon=horizon,
                    volume_available=volume_available,
                    meta_feature_cfg=meta_feature_cfg,
                )

                # Use the fully processed feature matrix (winsorisation, robust
                # scaling, selection) for learnability and diagnostics, to
                # match the production training path.
                X_for_learnability = meta_features_model_processed

                # Compute learnability score with isotonic calibration. Allow HPO to
                # tune the strength of signal-strength-based weighting.
                signal_strength_scale_max = float(params.get("signal_strength_scale_max", 1.5))
                if not np.isfinite(signal_strength_scale_max) or signal_strength_scale_max < 1.0:
                    signal_strength_scale_max = 1.5

                learnability_score, mean_auc, calibrated_probs, iso_reg_probe = compute_learnability_with_calibration(
                    X=X_for_learnability,
                    y=binary_labels,
                    realized_returns=realized_returns,
                    model_complexity=model_complexity,
                    cv_splits=cv_splits,
                    time_aware_cv=True,
                    use_ensemble=use_ensemble,
                    signal_strength_scale_max=signal_strength_scale_max,
                )

                # ===== AUC RANGE-BASED SCORING =====
                # Target AUC range: 0.55 - 0.67 (prop-shop acceptable range)
                # 
                # AUC INTERPRETATION GUIDELINES:
                # < 0.54:  Too noisy - model cannot distinguish signal from noise
                # 0.54 - 0.62: Good edge, acceptable by prop shops
                # 0.60 - 0.67: Excellent, but check for leakage or horizon bias
                # > 0.70:  Suspicious - likely data leakage or look-ahead bias
                #
                # Within target range: higher is better
                # Outside target range: penalize and shift weight to edge/pnl/trades
                
                auc_in_target_range = (mean_auc >= 0.55) and (mean_auc <= 0.67)
                auc_penalty = 0.0
                auc_weight_multiplier = 1.0  # Will reduce AUC influence when outside range
                
                if mean_auc < 0.54:
                    # Too noisy - heavy penalty, strong shift to edge metrics
                    auc_penalty = (0.54 - mean_auc) * 15.0
                    auc_weight_multiplier = 0.3  # AUC has less sway, edge/pnl matter more
                    if debug_sample_count < debug_sample_limit:
                        tprint_warning(f"[AUC] {mean_auc:.3f} < 0.54: Too noisy, model cannot distinguish signal")
                elif mean_auc < 0.55:
                    # Slightly below target - mild penalty
                    auc_penalty = (0.55 - mean_auc) * 8.0
                    auc_weight_multiplier = 0.7
                elif mean_auc > 0.70:
                    # Suspicious - likely leakage, heavy penalty
                    auc_penalty = (mean_auc - 0.70) * 20.0
                    auc_weight_multiplier = 0.2  # Strongly discount AUC contribution
                    tprint_warning(
                        f"[AUC] {mean_auc:.3f} > 0.70: SUSPICIOUS - check for data leakage, "
                        f"look-ahead bias, or horizon issues"
                    )
                elif mean_auc > 0.67:
                    # Above excellent range - moderate penalty, check for issues
                    auc_penalty = (mean_auc - 0.67) * 10.0
                    auc_weight_multiplier = 0.5
                    if debug_sample_count < debug_sample_limit:
                        tprint_warning(f"[AUC] {mean_auc:.3f} > 0.67: Excellent but verify no leakage/horizon bias")
                else:
                    # In target range [0.55, 0.67]: reward higher AUC within range
                    # Bonus peaks at 0.62 (center of "good edge" zone)
                    distance_from_sweet_spot = abs(mean_auc - 0.62)
                    auc_penalty = -max(0, 0.05 - distance_from_sweet_spot) * 5.0  # Negative = bonus
                    auc_weight_multiplier = 1.0
                
                learnability_score -= auc_penalty

                # Compute label entropy/balance score
                balance_score = compute_label_entropy_score(binary_labels)

                # NOTE: Comprehensive diagnostics (filtering, calibration, robustness, overlap)
                # are computed ONLY for the best config at the end of HPO, not during
                # the search loop. This improves HPO performance significantly.
                # See post-HPO diagnostics section below.

                # ===== ECONOMIC PROFITABILITY =====
                # Compute economic separation metrics on labeled events
                returns_labeled = realized_returns[labeled_mask]
                labels_labeled = binary_labels[labeled_mask]

                # Mean return for label=1 vs label=0
                r_pos = returns_labeled[labels_labeled == 1]
                r_neg = returns_labeled[labels_labeled == 0]

                mean_pos = float(r_pos.mean()) if len(r_pos) > 0 else 0.0
                mean_neg = float(r_neg.mean()) if len(r_neg) > 0 else 0.0
                sep = mean_pos - mean_neg

                # Simple Sharpe for label=1 trades
                std_pos = float(r_pos.std()) if len(r_pos) > 1 else 0.0
                sharpe_pos = mean_pos / (std_pos + 1e-8) if std_pos > 0 else 0.0

                # Use effective transaction cost (HPO-tunable via tx_cost_mult)
                tx = float(effective_tx_cost)

                # Targeted debug logging for a small sample of trials
                if debug_sample_count < debug_sample_limit:
                    tprint_info(
                        f"[HPO_DEBUG] n_events={n_events}, events_per_day={events_per_day:.3f}, "
                        f"mean_pos={mean_pos:.6f}, tx={tx:.6f}, above_tx={mean_pos > tx}",
                    )
                    debug_sample_count += 1

                # Hard economic gate: positive bucket must beat transaction cost
                if mean_pos <= tx:
                    return {
                        'learnability': float(learnability_score),
                        'profitability': -1e9,
                        'edge': -1e9,
                        'combined': -1e9,
                    }

                # Penalize configurations dominated by economically trivial events
                returns_labeled_nonnull = returns_labeled.dropna()
                if len(returns_labeled_nonnull) > 0:
                    small_band = tx
                    frac_small = float((returns_labeled_nonnull.abs() < small_band).mean())
                else:
                    frac_small = 1.0

                # ===== PRE- VS POST-FILTER DIAGNOSTICS (RETENTION & SNR) =====
                try:
                    pre_mask = ~realized_returns.isna()
                    n_pre_total = int(pre_mask.sum())

                    if n_pre_total > 0:
                        pre_returns = realized_returns[pre_mask]
                        raw_label_pre = (pre_returns > tx).astype(int)

                        n_pre_pos = int((raw_label_pre == 1).sum())
                        n_pre_neg = int((raw_label_pre == 0).sum())

                        n_post_total = int(n_events)
                        n_post_pos = int((labels_labeled == 1).sum())
                        n_post_neg = int((labels_labeled == 0).sum())

                        retention_total = n_post_total / max(n_pre_total, 1)
                        retention_pos = n_post_pos / max(n_pre_pos, 1) if n_pre_pos > 0 else 0.0
                        retention_neg = n_post_neg / max(n_pre_neg, 1) if n_pre_neg > 0 else 0.0

                        pre_pos_ret = pre_returns[raw_label_pre == 1]
                        pre_neg_ret = pre_returns[raw_label_pre == 0]

                        def _safe_stats(x: pd.Series) -> tuple[float, float]:
                            return (
                                float(x.mean()) if len(x) > 0 else 0.0,
                                float(x.std() if len(x) > 1 else 0.0),
                            )

                        pre_pos_mean, pre_pos_std = _safe_stats(pre_pos_ret)
                        pre_neg_mean, pre_neg_std = _safe_stats(pre_neg_ret)
                        post_pos_mean, post_pos_std = _safe_stats(r_pos)
                        post_neg_mean, post_neg_std = _safe_stats(r_neg)

                        def _cohens_d(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> float:
                            if n1 <= 1 or n2 <= 1:
                                return float('nan')
                            pooled = ((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / max(n1 + n2 - 2, 1)
                            if pooled <= 0:
                                return float('nan')
                            return (m1 - m2) / np.sqrt(pooled)

                        d_pre = _cohens_d(
                            pre_pos_mean,
                            pre_pos_std,
                            max(len(pre_pos_ret), 1),
                            pre_neg_mean,
                            pre_neg_std,
                            max(len(pre_neg_ret), 1),
                        )
                        d_post = _cohens_d(
                            post_pos_mean,
                            post_pos_std,
                            max(len(r_pos), 1),
                            post_neg_mean,
                            post_neg_std,
                            max(len(r_neg), 1),
                        )

                        snr_pre = pre_pos_mean / (pre_pos_std + 1e-8) if pre_pos_std > 0 else 0.0
                        snr_post = post_pos_mean / (post_pos_std + 1e-8) if post_pos_std > 0 else 0.0
                    else:
                        n_pre_total = 0
                        retention_total = 0.0
                        retention_pos = 0.0
                        retention_neg = 0.0
                        d_pre = float('nan')
                        d_post = float('nan')
                        snr_pre = 0.0
                        snr_post = 0.0
                except Exception:
                    n_pre_total = 0
                    retention_total = 0.0
                    retention_pos = 0.0
                    retention_neg = 0.0
                    d_pre = float('nan')
                    d_post = float('nan')
                    snr_pre = 0.0
                    snr_post = 0.0

                # Event density penalty: prefer ~0.5–4 trades/day (centered near ~2)
                # RELAXED: Wider acceptable range and gentler penalties to allow
                # HPO to explore more configurations before harsh filtering
                trades_per_day = n_events / days_span
                penalty_density = 0.0
                if trades_per_day < 0.3:
                    # Moderate penalty for very sparse regimes (was 0.5 threshold)
                    penalty_density += (0.3 - trades_per_day) * 8.0
                elif trades_per_day > 4.0:
                    # Gentler penalty for active regimes (was 3.0 threshold, 5.0 weight)
                    penalty_density += (trades_per_day - 4.0) * 3.0

                penalty_noise = frac_small * 10.0

                # Top-bucket economics (top 10% by smoothed probability) to capture
                # how good the very best signals are.
                top_bucket_mean = 0.0
                top_bucket_sharpe = 0.0
                try:
                    prob_array = prob_clipped.values.astype(float)
                    if np.isfinite(prob_array).any():
                        q90 = np.nanquantile(prob_array, 0.9)
                        top_mask = (prob_array >= q90) & labeled_mask.to_numpy()
                        top_returns = realized_returns[top_mask]
                        top_returns = top_returns.dropna()
                        if len(top_returns) >= 20:
                            top_bucket_mean = float(top_returns.mean())
                            top_std = float(top_returns.std())
                            top_bucket_sharpe = top_bucket_mean / (top_std + 1e-8) if top_std > 0 else 0.0
                except Exception:
                    top_bucket_mean = 0.0
                    top_bucket_sharpe = 0.0

                # Profitability score: emphasize separation and Sharpe, subtract penalties,
                # and reward strong top-bucket performance. TTO penalty is secondary but
                # present so that extremely slow configurations are disfavoured.
                profitability_score = (
                    sep * 100.0
                    + sharpe_pos * 10.0
                    + top_bucket_sharpe * 15.0
                    + top_bucket_mean * 1000.0
                    - penalty_density
                    - penalty_noise
                    - tto_penalty
                )

                # Extra penalty when label balance is extreme (balance_score == 0)
                if balance_score <= 0.0:
                    learnability_score -= 0.5

                # Simple power heuristic: required samples for ~80% power based on post-filter effect size
                try:
                    if np.isfinite(d_post) and d_post != 0.0:
                        n_required_80 = 16.0 / (d_post ** 2)
                    else:
                        n_required_80 = float('inf')
                except Exception:
                    n_required_80 = float('inf')

                # ===== REGULARIZATION CHECKS =====
                # Temporal stability check (rolling window AUC variance)
                auc_variance = 0.0
                try:
                    window_size = max(100, n_events // 5)
                    n_windows = min(5, n_events // window_size)

                    if n_windows >= 2:
                        window_aucs = []
                        for w in range(n_windows):
                            start_idx = w * window_size
                            end_idx = min((w + 1) * window_size, n_events)
                            window_labels = labels_labeled.iloc[start_idx:end_idx]

                            if len(window_labels.unique()) >= 2 and len(window_labels) >= 20:
                                # Compute simple correlation as AUC proxy
                                window_returns = returns_labeled.iloc[start_idx:end_idx]
                                try:
                                    window_auc = abs(window_labels.corr(window_returns))
                                    window_aucs.append(window_auc if not np.isnan(window_auc) else 0.5)
                                except:
                                    window_aucs.append(0.5)

                        if len(window_aucs) >= 2:
                            auc_variance = float(np.var(window_aucs))
                            # Penalize high variance (instability across time)
                            temporal_stability_penalty = auc_variance * 10.0
                            profitability_score -= temporal_stability_penalty
                except Exception:
                    pass  # Skip if temporal check fails

                # Reference trade count for Sharpe-like scaling of edge. We
                # center this around a target trades/day consistent with the
                # density band above so that edge mildly rewards configurations
                # that achieve a healthy number of good trades.
                target_trades_per_day = float(config.get("edge_target_trades_per_day", 2.0))
                reference_trades = max(1.0, float(days_span) * target_trades_per_day)

                # ===== REALISTIC P&L EDGE METRIC =====
                # Edge = (Mean Return - Cost) × max(0, 2×AUC - 1)
                # This penalizes "profitable but unlearnable" strategies more realistically
                # ENHANCED: Pass days_span for density bonus to encourage healthy trade counts
                edge_score = compute_realistic_pnl_edge(
                    mean_return_positive=mean_pos,
                    mean_auc=mean_auc,
                    transaction_cost=tx,
                    n_trades=n_events,
                    reference_trades=reference_trades,
                    days_span=float(days_span),
                    target_trades_per_day=target_signal_density,
                )
                # Tie temporal instability to edge: softly down-weight edge when
                # rolling-window AUC variance is high.
                if auc_variance > 0.0:
                    # For typical auc_variance in [0, ~0.02], this yields a modest
                    # 0–20% down-weighting for unstable configurations.
                    instability = min(1.0, auc_variance / 0.02)
                    edge_score *= (1.0 - 0.3 * instability)

                # Scale edge for combined metric (multiply by 1000 to make comparable)
                edge_scaled = edge_score * 1000.0

                # Additional hard penalties for pathological configurations (no positive
                # or negative bucket), while still keeping them in the candidate pool
                # for diagnostics.
                if len(r_pos) == 0 or len(r_neg) == 0:
                    profitability_score = -1e9
                    edge_score = 0.0
                    edge_scaled = 0.0
                    learnability_score -= 0.5

                # ===== COMBINED OBJECTIVE WITH RETENTION REGULARIZATION =====
                # Simplified formula: objective = AUC_adjusted - α * (1 / retention_rate)
                # 
                # This penalizes configurations that achieve high AUC through aggressive
                # filtering (low retention) rather than genuine signal quality.
                #
                # α is small (0.01-0.05) to allow retention penalty without dominating
                # Retention rate = n_filtered_events / n_raw_events (candidate survival rate)
                
                # Retention regularization parameter (configurable)
                alpha_retention = float(config.get("retention_regularization_alpha", 0.03))
                alpha_retention = max(0.01, min(0.10, alpha_retention))  # Clamp to [0.01, 0.10]
                
                # Compute retention penalty
                # retention_total is already computed above as n_post_total / n_pre_total
                if retention_total > 0:
                    retention_penalty = alpha_retention * (1.0 / retention_total)
                else:
                    retention_penalty = alpha_retention * 100.0  # Severe penalty for 0 retention
                
                # Cap retention penalty to avoid dominating the objective
                retention_penalty = min(retention_penalty, 0.5)
                
                # ===== UNIFIED AUC ADJUSTMENT =====
                # Combine AUC range penalties directly into adjusted AUC instead of
                # having separate auc_penalty and learnability_bonus terms.
                # This simplifies the objective and avoids double-counting.
                
                auc_range_adjustment = 0.0
                if mean_auc < 0.54:
                    # Too noisy: heavy penalty incorporated into AUC
                    auc_range_adjustment = -(0.54 - mean_auc) * 0.5
                elif mean_auc > 0.70:
                    # Suspicious (likely leakage): heavy penalty
                    auc_range_adjustment = -(mean_auc - 0.70) * 1.0
                elif mean_auc > 0.67:
                    # Above excellent range: moderate penalty
                    auc_range_adjustment = -(mean_auc - 0.67) * 0.3
                elif auc_in_target_range:
                    # In target range [0.55, 0.67]: small bonus for 0.60-0.62 sweet spot
                    if mean_auc >= 0.58 and mean_auc <= 0.64:
                        auc_range_adjustment = 0.02  # Small bonus for ideal range
                
                # Final retention-adjusted AUC (unified primary signal)
                auc_retention_adjusted = mean_auc + auc_range_adjustment - retention_penalty
                
                # Trade density bonus: stronger when AUC is outside target range
                # This gives edge/pnl/trades more influence when AUC is penalized
                edge_weight = 1.0 + (1.0 - auc_weight_multiplier) * 0.5  # Up to 1.5x edge weight
                density_weight = 1.0 + (1.0 - auc_weight_multiplier) * 1.0  # Up to 2x density weight
                
                # Trades/day bonus for being in sweet spot (1-3 trades/day)
                trades_bonus = 0.0
                if trades_per_day >= 0.8 and trades_per_day <= 4.0:
                    # Peak bonus at 2 trades/day
                    trades_bonus = max(0, 1.0 - abs(trades_per_day - 2.0) / 2.0) * 50.0
                
                # ===== SIMPLIFIED COMBINED OBJECTIVE =====
                # Primary components:
                # 1. edge_scaled: Captures profitability AND learnability (via capture ratio)
                # 2. auc_retention_adjusted: AUC adjusted for retention penalty & range
                # 3. trades_bonus: Reward healthy trade density
                # 4. Penalties for extreme density and slow exits
                combined_score = (
                    edge_scaled * edge_weight
                    + (auc_retention_adjusted * 100.0)  # Unified AUC signal
                    + (trades_bonus * density_weight)
                    - (penalty_density * 0.1)
                    - (tto_penalty * 0.1)
                )

                # Store candidate configuration for later persistence
                candidate_config = {
                    'params': params.copy(),
                    'learnability': float(learnability_score),
                    'mean_auc': float(mean_auc),
                    'profitability': float(profitability_score),
                    'edge': float(edge_score),
                    'edge_scaled': float(edge_scaled),
                    'combined': float(combined_score),
                    'mean_pos': float(mean_pos),
                    'mean_neg': float(mean_neg),
                    'sharpe_pos': float(sharpe_pos),
                    'n_events': int(n_events),
                    'n_raw_events': int(n_raw_events),
                    'n_vol_scaled_events': int(n_vol_scaled_events),
                    'balance_score': float(balance_score),
                    'trades_per_day': float(trades_per_day),
                    'mean_tto': float(mean_tto) if np.isfinite(mean_tto) else float('nan'),
                    'timeout_rate': float(timeout_rate) if np.isfinite(timeout_rate) else float('nan'),
                    'tto_penalty': float(tto_penalty),
                    'n_pre_events': int(n_pre_total),
                    'retention_total': float(retention_total),
                    'retention_pos': float(retention_pos),
                    'retention_neg': float(retention_neg),
                    'snr_pre': float(snr_pre),
                    'snr_post': float(snr_post),
                    'effect_size_pre': float(d_pre) if np.isfinite(d_pre) else 0.0,
                    'effect_size_post': float(d_post) if np.isfinite(d_post) else 0.0,
                    'n_required_80pct_power': float(n_required_80),
                    'model_complexity': model_complexity,
                    # NEW: Track trade count control parameters
                    'cusum_threshold': float(cusum_threshold),
                    'target_signal_density': float(target_signal_density),
                    'r_multiple_threshold': float(r_multiple_threshold),
                    'effective_tx_cost': float(effective_tx_cost),
                    'label_low_q': float(label_low_q),
                    'label_high_q': float(label_high_q),
                    # AUC range tracking (unified into auc_retention_adjusted)
                    'auc_in_target_range': bool(auc_in_target_range),
                    'auc_range_adjustment': float(auc_range_adjustment),
                    'auc_weight_multiplier': float(auc_weight_multiplier),
                    'auc_interpretation': (
                        'too_noisy' if mean_auc < 0.54 else
                        'good_edge' if mean_auc <= 0.62 else
                        'excellent_check_leakage' if mean_auc <= 0.67 else
                        'suspicious_leakage' if mean_auc > 0.70 else
                        'above_target'
                    ),
                    # Retention regularization (unified objective)
                    'retention_penalty': float(retention_penalty),
                    'auc_retention_adjusted': float(auc_retention_adjusted),
                    'alpha_retention': float(alpha_retention),
                }

                # Optional per-regime breakdown using attached HMM regimes, if available.
                per_regime_metrics: Dict[str, Any] = {}
                try:
                    if "hmm_regime_label_1h" in market_data.columns:
                        regimes_all = market_data["hmm_regime_label_1h"]
                        regimes_events = regimes_all[labeled_mask]

                        # Align calibrated probabilities with labeled events
                        try:
                            probs_series_full = pd.Series(calibrated_probs, index=binary_labels.index)
                            probs_events = probs_series_full[labeled_mask]
                        except Exception:
                            probs_events = None

                        from sklearn.metrics import roc_auc_score as _roc_auc_score_reg

                        unique_regs = pd.unique(regimes_events.dropna())
                        for reg_val in unique_regs:
                            try:
                                reg_mask = regimes_events == reg_val
                                n_reg = int(reg_mask.sum())
                                if n_reg < 20:
                                    continue

                                returns_reg = returns_labeled[reg_mask]
                                labels_reg = labels_labeled[reg_mask]

                                r_pos_reg = returns_reg[labels_reg == 1]
                                r_neg_reg = returns_reg[labels_reg == 0]

                                mean_pos_reg = float(r_pos_reg.mean()) if len(r_pos_reg) > 0 else 0.0
                                mean_neg_reg = float(r_neg_reg.mean()) if len(r_neg_reg) > 0 else 0.0

                                # Realized AUC within this regime (diagnostic only)
                                auc_reg_local = float("nan")
                                if probs_events is not None:
                                    try:
                                        probs_reg = probs_events[reg_mask]
                                        if len(labels_reg.unique()) >= 2:
                                            auc_reg_local = float(_roc_auc_score_reg(labels_reg, probs_reg))
                                    except Exception:
                                        auc_reg_local = float("nan")

                                # For edge, use the same cross-validated mean_auc and
                                # Sharpe-like trade-count scaling as the global metric,
                                # so that regime-level edges are directly comparable to
                                # the reported best_edge.
                                auc_for_edge = float(mean_auc)

                                edge_reg = compute_realistic_pnl_edge(
                                    mean_return_positive=mean_pos_reg,
                                    mean_auc=auc_for_edge,
                                    transaction_cost=tx,
                                    n_trades=n_reg,
                                    reference_trades=reference_trades,
                                )

                                trades_per_day_reg = float(n_reg) / max(days_span, 1)

                                per_regime_metrics[str(reg_val)] = {
                                    'n_events': n_reg,
                                    'trades_per_day': trades_per_day_reg,
                                    'mean_pos': mean_pos_reg,
                                    'mean_neg': mean_neg_reg,
                                    # Expose the aligned AUC used for edge, and keep the
                                    # local realized AUC as an auxiliary diagnostic field.
                                    'auc': auc_for_edge,
                                    'auc_local': auc_reg_local,
                                    'edge': edge_reg,
                                }
                            except Exception:
                                continue
                except Exception:
                    per_regime_metrics = {}

                if per_regime_metrics:
                    candidate_config['per_regime_metrics'] = per_regime_metrics

                # NOTE: Comprehensive diagnostics are computed post-HPO for best config only.
                # See compute_best_config_diagnostics() call after HPO loop.

                # Attach regression target diagnostics, if available, so that
                # downstream analysis can assess how well continuous payoffs
                # behaved for this labeling configuration.
                if reg_target_stats is not None:
                    candidate_config['regression_target_stats'] = reg_target_stats

                candidate_pool.append(candidate_config)

                return {
                    'learnability': float(learnability_score),
                    'profitability': float(profitability_score),
                    'edge': float(edge_score),
                    'combined': float(combined_score)
                }

            except Exception as exc:  # Defensive: never crash HPO on one config
                tprint_warning(f"⚠️ Labeling objective failed: {exc}")
                import traceback
                traceback.print_exc()
                return {'learnability': 0.0, 'profitability': -1e9, 'edge': -1e9, 'combined': -1e9}

        # ------------------------------------------------------------------
        # 4) Multi-objective wrapper for single-objective optimizers
        # ------------------------------------------------------------------
        def create_scalar_objective_wrapper(
            model_complexity: str = "fast",
            use_ensemble: bool = False,
            compute_diagnostics: bool = False,
        ) -> callable:
            """Create a wrapper with specific complexity settings.

            BayesianTPEOptimizer always calls the objective as objective(params), so
            we close over X_dummy / y_dummy and complexity settings.
            """
            def wrapper(params: Dict[str, Any]) -> float:
                result = labeling_objective(
                    params, X_dummy, y_dummy,
                    model_complexity=model_complexity,
                    use_ensemble=use_ensemble,
                    compute_diagnostics=compute_diagnostics,
                )
                if isinstance(result, dict):
                    return float(result.get('edge', result.get('combined', 0.0)))
                return float(result)
            return wrapper

        # ------------------------------------------------------------------
        # 5) Convert param_groups to search space
        # ------------------------------------------------------------------
        tprint_info("🚀 Using Multi-Stage Bayesian TPE optimization")

        # Convert param_groups (ParameterGroup instances) to a flat Optuna-style search space
        initial_search_space: Dict[str, Dict[str, Any]] = {}
        for group in param_groups:
            for param_name, param_spec in group.params.items():
                initial_search_space[param_name] = param_spec

        if warm_start_candidates_df is not None and not warm_start_candidates_df.empty:
            try:
                shrinkable_params = [
                    "profit_thr_base",
                    "stop_to_profit_ratio",
                    "horizon_bars",
                    "min_event_spacing",
                    "vol_baseline_window",
                    "profit_mult_min",
                    "profit_mult_max",
                    "stop_mult_min",
                    "stop_mult_max",
                    "iso_min_prob",
                    "target_clip_high_q",
                    "econ_min_return_multiple",
                    "label_low_q",
                    "label_high_q",
                    "signal_strength_scale_max",
                    # NEW parameters for trade count control
                    "cusum_threshold",
                    "target_signal_density",
                    "r_multiple_pos_threshold",
                    "transaction_cost_mult",
                ]
                for p in shrinkable_params:
                    if p not in initial_search_space:
                        continue
                    if p not in warm_start_candidates_df.columns:
                        continue
                    spec = initial_search_space.get(p, {})
                    if not isinstance(spec, dict):
                        continue
                    ptype = spec.get("type", "float")
                    if ptype not in ["float", "int"]:
                        continue
                    series = warm_start_candidates_df[p].dropna()
                    if series.empty or len(series) < 20:
                        continue
                    try:
                        q_low = float(series.quantile(0.10))
                        q_high = float(series.quantile(0.90))
                    except Exception:
                        continue
                    low = spec.get("low")
                    high = spec.get("high")
                    if low is None or high is None:
                        continue
                    new_low = max(float(low), q_low)
                    new_high = min(float(high), q_high)
                    if new_high <= new_low:
                        continue
                    new_spec = spec.copy()
                    if ptype == "int":
                        new_spec["low"] = int(new_low)
                        new_spec["high"] = int(max(new_high, new_low + 1))
                    else:
                        new_spec["low"] = float(new_low)
                        new_spec["high"] = float(new_high)
                    initial_search_space[p] = new_spec
                tprint_info("📌 Warm-start: narrowed search space around previous candidate quantiles")
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 6) Multi-Stage HPO Execution Loop
        # ------------------------------------------------------------------
        # Parameters optimized per stage:
        # Stage 1: horizon_bars, min_event_spacing, target_transform (iso/clipping,
        #          econ_min_return_multiple, label quantiles, signal strength
        #          scaling) + kalman_Q, kalman_R, vol_baseline_window,
        #          profit_mult_min/max, stop_mult_min/max
        # Stage 2: kalman_Q, kalman_R, vol_baseline_window, profit_mult_min/max,
        #          stop_mult_min/max
        # Stage 3: All parameters (profit_thr_base, stop_to_profit_ratio,
        #          iso_min_prob, target_transform refinements, etc.)
        # Stage 4: Filtering parameters only (label_low_q, label_high_q, econ_min_return_multiple, etc.)
        #          with fixed structural parameters.
        #
        # The multi-stage process progressively increases model complexity to find
        # configurations that are both profitable AND learnable by production models.

        # Define which parameters to optimize at each stage
        if calibrated_horizon is not None:
            stage_1_params = [
                'min_event_spacing',
                'cusum_threshold', 'target_signal_density',
                'profit_thr_base', 'stop_to_profit_ratio', 'trail_distance',
                'iso_min_prob', 'target_clip_high_q',
                'econ_min_return_multiple', 'label_low_q', 'label_high_q',
                'signal_strength_scale_max',
                'r_multiple_pos_threshold', 'transaction_cost_mult',
                'kalman_Q', 'kalman_R', 'vol_baseline_window',
                'profit_mult_min', 'profit_mult_max', 'stop_mult_min', 'stop_mult_max',
            ]
            stage_2_params = [
                'kalman_Q', 'kalman_R', 'vol_baseline_window',
                'profit_mult_min', 'profit_mult_max', 'stop_mult_min', 'stop_mult_max'
            ]
        else:
            stage_1_params = [
                'horizon_bars', 'min_event_spacing',
                'cusum_threshold', 'target_signal_density',
                'profit_thr_base', 'stop_to_profit_ratio', 'trail_distance',
                'iso_min_prob', 'target_clip_high_q',
                'econ_min_return_multiple', 'label_low_q', 'label_high_q',
                'signal_strength_scale_max',
                'r_multiple_pos_threshold', 'transaction_cost_mult',
                'kalman_Q', 'kalman_R', 'vol_baseline_window',
                'profit_mult_min', 'profit_mult_max', 'stop_mult_min', 'stop_mult_max',
            ]
            stage_2_params = [
                'kalman_Q', 'kalman_R', 'vol_baseline_window',
                'profit_mult_min', 'profit_mult_max', 'stop_mult_min', 'stop_mult_max'
            ]

        # Stage 4 specific params (filtering/labeling only)
        # ENHANCED: Include all trade count-affecting parameters for final refinement
        stage_4_params = [
            'label_low_q', 'label_high_q',
            'econ_min_return_multiple',
            'iso_min_prob', 'target_clip_high_q',
            'signal_strength_scale_max',
            'r_multiple_pos_threshold',  # NEW: R-multiple threshold
            'transaction_cost_mult',  # NEW: transaction cost sensitivity
            'cusum_threshold', 'target_signal_density',  # NEW: signal density
        ]

        # Stage 3 uses all parameters (optionally treating horizon_bars as fixed when calibrated)

        stages = DEFAULT_STAGE_CONFIG
        stage_results: List[Dict[str, Any]] = []
        all_trials_count = 0
        best_overall_score = float('-inf')
        best_overall_params = {}

        # Track best params from each stage to use as defaults for fixed params
        accumulated_best_params: Dict[str, Any] = {}
        if calibrated_horizon is not None:
            accumulated_best_params["horizon_bars"] = int(calibrated_horizon)

        for stage_idx, stage in enumerate(stages):
            tprint_info(f"🚀 Starting {stage['name']} with complexity={stage['complexity']}...")

            # Determine which parameters to optimize in this stage
            if stage_idx == 0:  # Stage 1
                active_params = stage_1_params
            elif stage_idx == 1:  # Stage 2
                active_params = stage_2_params
            elif stage_idx == 2:  # Stage 3 - all parameters
                if calibrated_horizon is not None:
                    active_params = [
                        k for k in initial_search_space.keys()
                        if k != 'horizon_bars'
                    ]
                else:
                    active_params = list(initial_search_space.keys())
            else:  # Stage 4 - Labeling Refinement
                active_params = stage_4_params

            # Create stage-specific search space
            current_search_space = {
                k: v for k, v in initial_search_space.items()
                if k in active_params
            }

            tprint_info(f"   Optimizing parameters: {list(current_search_space.keys())}")

            # Create objective wrapper that merges optimized params with fixed params from previous stages
            def create_stage_objective_wrapper(
                model_complexity: str,
                use_ensemble: bool,
                compute_diagnostics: bool,
                fixed_params: Dict[str, Any],
                use_stage1_subsample: bool,
            ) -> callable:
                """Create a wrapper that injects fixed params from previous stages."""
                def wrapper(params: Dict[str, Any]) -> float:
                    nonlocal market_data, primary_signals, volatility_1d, days_span, atr_series
                    if use_stage1_subsample and model_complexity == "fast" and stage1_enable_subsample:
                        md_backup = market_data
                        ps_backup = primary_signals
                        vol_backup = volatility_1d
                        atr_backup = atr_series
                        days_backup = days_span
                        try:
                            market_data = stage1_market_data
                            primary_signals = stage1_primary_signals
                            volatility_1d = stage1_volatility_1d
                            atr_series = stage1_atr_series
                            days_span = stage1_days_span
                            full_params = {**fixed_params, **params}
                            result = labeling_objective(
                                full_params, X_dummy, y_dummy,
                                model_complexity=model_complexity,
                                use_ensemble=use_ensemble,
                                compute_diagnostics=compute_diagnostics,
                            )
                        finally:
                            market_data = md_backup
                            primary_signals = ps_backup
                            volatility_1d = vol_backup
                            atr_series = atr_backup
                            days_span = days_backup
                    else:
                        full_params = {**fixed_params, **params}
                        result = labeling_objective(
                            full_params, X_dummy, y_dummy,
                            model_complexity=model_complexity,
                            use_ensemble=use_ensemble,
                            compute_diagnostics=compute_diagnostics,
                        )
                    if isinstance(result, dict):
                        return float(result.get('edge', result.get('combined', 0.0)))
                    return float(result)
                return wrapper

            if stage_idx in (0, 1):
                stage_param_groups: list[list[str]] = []
                if stage_idx == 0:
                    # Stage 1 (fast model, subsampled data):
                    # Group A – event shape / density + signal generation
                    if calibrated_horizon is not None:
                        stage_param_groups.append(['min_event_spacing', 'cusum_threshold', 'target_signal_density', 'trail_distance'])
                    else:
                        stage_param_groups.append(['horizon_bars', 'min_event_spacing', 'cusum_threshold', 'target_signal_density', 'trail_distance'])
                    # Group B – TPSL geometry
                    stage_param_groups.append(['profit_mult_min', 'profit_mult_max', 'stop_mult_min', 'stop_mult_max'])
                    # Group C – smoothing
                    stage_param_groups.append(['kalman_Q', 'kalman_R'])
                    # Group D – target transform (clipping, econ floor, label quantiles,
                    # and signal-strength weighting strength) + trade filters
                    stage_param_groups.append([
                        'iso_min_prob', 'target_clip_high_q',
                        'econ_min_return_multiple', 'label_low_q', 'label_high_q',
                        'signal_strength_scale_max',
                        'r_multiple_pos_threshold', 'transaction_cost_mult',  # NEW
                    ])
                else:
                    # Stage 2 (medium model, full data): smoothing + TPSL/vol
                    stage_param_groups.append(['kalman_Q', 'kalman_R'])
                    stage_param_groups.append(['vol_baseline_window', 'profit_mult_min', 'profit_mult_max', 'stop_mult_min', 'stop_mult_max'])

                stage_best_score = float('-inf')
                stage_best_params: Dict[str, Any] = {}
                stage_trials_total = 0

                n_groups = max(1, len(stage_param_groups))

                for group_idx, group_params in enumerate(stage_param_groups):
                    group_search_space = {
                        k: v for k, v in current_search_space.items()
                        if k in group_params
                    }
                    if not group_search_space:
                        continue

                    # Fixed params: best from previous stages and previous groups in this stage
                    group_fixed_params = {
                        k: v for k, v in accumulated_best_params.items()
                        if k not in group_search_space
                    }

                    group_objective = create_stage_objective_wrapper(
                        model_complexity=stage["complexity"],
                        use_ensemble=(stage["complexity"] == "strong"),
                        compute_diagnostics=(stage_idx == len(stages) - 1 and group_idx == len(stage_param_groups) - 1),
                        fixed_params=group_fixed_params,
                        use_stage1_subsample=(stage_idx == 0),
                    )

                    group_n_trials = max(5, stage["n_trials"] // n_groups)

                    group_config = OptimizationConfig(
                        n_trials=group_n_trials,
                        execution_mode=config.get("execution_mode", "full"),
                        direction='maximize',
                        # For small groups, use direct TPE (no internal grid) for speed
                        enable_staged_optimization=False,
                        coarse_grid_trials=0,
                        fine_grid_trials=0,
                        tpe_trials=group_n_trials,
                        enable_hardware_optimization=False,
                        enable_vectorbt_optimization=False,
                        early_stopping_patience=max(5, group_n_trials // 3),
                        early_stopping_threshold=None,
                        seed=42 + stage_idx * 10 + group_idx,
                    )

                    group_optimizer = BayesianTPEOptimizer(config=group_config)

                    tprint_info(
                        f"   → Group {group_idx + 1}/{len(stage_param_groups)} params={list(group_search_space.keys())}, "
                        f"trials={group_n_trials}"
                    )

                    try:
                        group_result = group_optimizer.optimize(
                            objective=group_objective,
                            search_space=group_search_space,
                        )
                        group_best_params = group_result.get('best_params', {})
                        group_best_score = group_result.get('best_value', 0.0)
                        group_trials = int(group_result.get('n_trials', group_n_trials))

                        stage_trials_total += group_trials

                        if group_best_score > stage_best_score:
                            stage_best_score = group_best_score

                        # Accumulate params from this group
                        accumulated_best_params.update(group_best_params)
                        stage_best_params.update(group_best_params)

                        tprint_success(
                            f"   ✅ Group {group_idx + 1}/{len(stage_param_groups)} complete: "
                            f"best_score={group_best_score:.4f}, trials={group_trials}"
                        )

                    except Exception as group_exc:
                        tprint_warning(
                            f"   ⚠️ Stage {stage['name']} group {group_idx + 1} failed: {group_exc}"
                        )
                        import traceback
                        traceback.print_exc()
                        continue

                if stage_best_score == float('-inf'):
                    # No successful groups; fall back to next stage
                    continue

                all_trials_count += stage_trials_total

                if stage_best_score > best_overall_score:
                    best_overall_score = stage_best_score
                    best_overall_params = stage_best_params.copy()

                stage_results.append({
                    'stage': stage['name'],
                    'complexity': stage['complexity'],
                    'best_score': stage_best_score,
                    'best_params': stage_best_params,
                    'trials': stage_trials_total,
                })

                # For Stage 2 (index 1), shrink Stage 3 search space using medium-model candidates
                if stage_idx == 1:
                    stage_candidates = [
                        c for c in candidate_pool
                        if c.get('model_complexity') == stage['complexity']
                    ]
                    if stage_candidates:
                        try:
                            initial_search_space = shrink_search_space(
                                original_space=initial_search_space,
                                previous_results=stage_candidates,
                                top_k=stage['top_k_to_pass'],
                            )
                            tprint_info(
                                f"   📉 Narrowed Stage 3 search space based on "
                                f"Top {min(len(stage_candidates), stage['top_k_to_pass'])} candidates"
                            )
                        except Exception:
                            pass

                tprint_success(
                    f"   ✅ {stage['name']} complete (hierarchical groups): "
                    f"best_score={stage_best_score:.4f}, trials={stage_trials_total}"
                )

                # Done with this stage; move to next global stage
                continue

            # Get fixed params: best values from previous stages for params not being optimized
            fixed_params = {
                k: v for k, v in accumulated_best_params.items()
                if k not in current_search_space
            }

            stage_objective = create_stage_objective_wrapper(
                model_complexity=stage["complexity"],
                use_ensemble=(stage["complexity"] == "strong"),
                compute_diagnostics=(stage_idx == len(stages) - 1),
                fixed_params=fixed_params,
                use_stage1_subsample=(stage_idx == 0),
            )

            # Configure Bayesian optimizer for this stage
            bayesian_config = OptimizationConfig(
                n_trials=stage["n_trials"],
                execution_mode=config.get("execution_mode", "full"),
                direction='maximize',
                # Staged optimization settings (use TPE directly for speed)
                enable_staged_optimization=(stage_idx == 0),  # Only use grid in first stage
                coarse_grid_trials=min(15, stage["n_trials"] // 5) if stage_idx == 0 else 0,
                fine_grid_trials=min(10, stage["n_trials"] // 10) if stage_idx == 0 else 0,
                tpe_trials=stage["n_trials"] - (25 if stage_idx == 0 else 0),
                # Disable hardware/VectorBT-specific acceleration
                enable_hardware_optimization=False,
                enable_vectorbt_optimization=False,
                # Early stopping per trial
                early_stopping_patience=max(5, stage["n_trials"] // 5),
                early_stopping_threshold=None,
                # Reproducibility
                seed=42 + stage_idx,
            )

            optimizer = BayesianTPEOptimizer(config=bayesian_config)

            # Run optimization for this stage
            tprint_info(f"   Running {stage['n_trials']} trials with {stage['complexity']} model...")

            try:
                result = optimizer.optimize(
                    objective=stage_objective,
                    search_space=current_search_space,
                )

                stage_best_params = result.get('best_params', {})
                stage_best_score = result.get('best_value', 0.0)
                stage_trials = result.get('total_trials', stage["n_trials"])

                all_trials_count += stage_trials

                # Track best overall
                if stage_best_score > best_overall_score:
                    best_overall_score = stage_best_score
                    best_overall_params = stage_best_params.copy()

                # Accumulate best params from this stage for use in subsequent stages
                accumulated_best_params.update(stage_best_params)

                tprint_success(
                    f"   ✅ {stage['name']} complete: "
                    f"best_score={stage_best_score:.4f}, trials={stage_trials}"
                )

                # Store stage results
                stage_results.append({
                    'stage': stage['name'],
                    'complexity': stage['complexity'],
                    'best_score': stage_best_score,
                    'best_params': stage_best_params,
                    'trials': stage_trials,
                })

                # Get candidates from this stage
                stage_candidates = [
                    c for c in candidate_pool
                    if c.get('model_complexity') == stage['complexity']
                ]

                # Identify best candidate from this stage for explicit reporting
                if stage_candidates:
                    best_cand = max(stage_candidates, key=lambda x: x.get('edge', x.get('combined', 0)))
                    tprint_info(
                        f"   🏆 Best candidate {stage['name']}: "
                        f"Edge={best_cand.get('edge', 0):.4f}, "
                        f"AUC={best_cand.get('mean_auc', 0):.3f}, "
                        f"Trades/Day={best_cand.get('trades_per_day', 0):.2f}, "
                        f"Raw={best_cand.get('n_raw_events', 0)} → Vol={best_cand.get('n_vol_scaled_events', 0)} → Final={best_cand.get('n_events', 0)}"
                    )

                # For 4-stage setup:
                # - After Stage 2 (index 1): Shrink space for Stage 3 (handled above in group block or here if not group)
                # - After Stage 3 (index 2): No shrinking needed for Stage 4 because Stage 4 uses a specific subset
                #   of parameters (labeling only), and structural params are fixed via accumulated_best_params.
                if stage_idx == 1 and stage_candidates:
                    try:
                        initial_search_space = shrink_search_space(
                            original_space=initial_search_space,
                            previous_results=stage_candidates,
                            top_k=stage['top_k_to_pass'],
                        )
                        tprint_info(
                            f"   📉 Narrowed Stage 3 search space based on "
                            f"Top {min(len(stage_candidates), stage['top_k_to_pass'])} candidates"
                        )
                    except Exception:
                        pass

                # Per-stage early stopping: if no improvement in this stage, move to next
                # (This is handled by the optimizer's early_stopping_patience setting)

            except Exception as stage_exc:
                tprint_warning(f"   ⚠️ Stage {stage['name']} failed: {stage_exc}")
                import traceback
                traceback.print_exc()
                # Continue to next stage or use best so far
                continue

        # Final best from all stages
        best_params = best_overall_params
        best_score = best_overall_score

        tprint_info(f"   Total trials across all stages: {all_trials_count}")

        tprint_success(f"✅ Labeling HPO completed. Best edge={best_score:.6f}")
        tprint_info(f"Best parameters: {best_params}")

        # Extract best candidate's detailed metrics for summary
        best_candidate_metrics = {}
        try:
            if candidate_pool:
                for cand in candidate_pool:
                    cand_params = cand.get("params", {})
                    if cand_params == best_params:
                        best_candidate_metrics = {
                            "mean_auc": float(cand.get("mean_auc", 0.5)),
                            "trades_per_day": float(cand.get("trades_per_day", 0.0)),
                            "learnability": float(cand.get("learnability", 0.0)),
                            "profitability": float(cand.get("profitability", 0.0)),
                            "sharpe_pos": float(cand.get("sharpe_pos", 0.0)),
                            "balance_score": float(cand.get("balance_score", 0.0)),
                            "n_events": int(cand.get("n_events", 0)),
                        }
                        break
        except Exception as metric_exc:
            tprint_warning(f"⚠️ Failed to extract best candidate metrics: {metric_exc}")
            best_candidate_metrics = {}

        # ------------------------------------------------------------------
        # 7) COMPREHENSIVE DIAGNOSTICS FOR BEST CONFIG
        # ------------------------------------------------------------------
        # These diagnostics are computed ONLY for the best config, not during HPO
        tprint_info("🔍 Computing comprehensive diagnostics for best configuration...")
        
        best_config_diagnostics = {}
        try:
            # Re-run labeling with best params to get intermediate data
            diag_params = best_params.copy()
            
            # Extract parameters
            profit_thr_base = float(diag_params.get("profit_thr_base", 0.012))
            stop_ratio = float(diag_params.get("stop_to_profit_ratio", 0.5))
            stop_thr_base = max(0.0005, profit_thr_base * stop_ratio)
            horizon = int(diag_params.get("horizon_bars", 24))
            min_spacing = int(diag_params.get("min_event_spacing", 4))
            econ_min_mult = float(diag_params.get("econ_min_return_multiple", 2.0))
            label_low_q = float(diag_params.get("label_low_q", 0.30))
            label_high_q = float(diag_params.get("label_high_q", 0.70))
            tx_cost_mult = float(diag_params.get("transaction_cost_mult", 1.0))
            effective_tx_cost = DEFAULT_TRANSACTION_COST * tx_cost_mult
            vol_baseline_window = int(diag_params.get("vol_baseline_window", 96))
            profit_mult_min = float(diag_params.get("profit_mult_min", 0.5))
            profit_mult_max = float(diag_params.get("profit_mult_max", 2.0))
            stop_mult_min = float(diag_params.get("stop_mult_min", 0.5))
            stop_mult_max = float(diag_params.get("stop_mult_max", 2.0))
            
            # Extract trailing distance if available
            trail_dist_diag = float(diag_params.get("trail_distance", 0.0))

            # Recompute ATR for diagnostics
            high_prices = market_data["high"] if "high" in market_data.columns else market_data["close"]
            low_prices = market_data["low"] if "low" in market_data.columns else market_data["close"]
            close_prices = market_data["close"]
            tr1 = high_prices - low_prices
            tr2 = (high_prices - close_prices.shift(1)).abs()
            tr3 = (low_prices - close_prices.shift(1)).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_series_diag = true_range.rolling(window=14, min_periods=1).mean()

            # Compute adaptive thresholds
            vol_baseline = volatility_1d.rolling(vol_baseline_window).mean()
            vol_factor = volatility_1d / (vol_baseline + 1e-8)
            
            adaptive_profit = profit_thr_base * vol_factor
            adaptive_stop = stop_thr_base * vol_factor
            adaptive_profit = adaptive_profit.clip(
                lower=profit_thr_base * profit_mult_min,
                upper=profit_thr_base * profit_mult_max,
            )
            adaptive_stop = adaptive_stop.clip(
                lower=stop_thr_base * stop_mult_min,
                upper=stop_thr_base * stop_mult_max,
            )
            
            # Compute realized returns
            (
                realized_returns_diag,
                binary_labels_raw,
                exit_reasons_diag,
                event_durations_diag,
                mfe_diag,
                mae_diag,
                _, _
            ) = compute_realized_returns(
                market_data,
                primary_signals,
                profit_threshold=adaptive_profit,
                stop_threshold=adaptive_stop,
                horizon=horizon,
                transaction_cost=effective_tx_cost,
                min_event_spacing=min_spacing,
                atr_series=atr_series_diag,
                trail_distance_atr_mult=trail_dist_diag,
            )
            
            # Vol-scaled returns and quantile labels
            vol_scaled_diag = compute_vol_scaled_returns_for_events(
                realized_returns=realized_returns_diag,
                volatility=volatility_1d,
                econ_min_return_multiple=econ_min_mult,
            )
            
            regimes_diag = market_data.get("hmm_regime_label_1h")
            if regimes_diag is not None:
                quantile_labels_diag = create_regime_aware_quantile_labels_from_vol_scaled_returns(
                    vol_scaled=vol_scaled_diag,
                    regimes=regimes_diag,
                    low_q=label_low_q,
                    high_q=label_high_q,
                )
            else:
                quantile_labels_diag = create_quantile_labels_from_vol_scaled_returns(
                    vol_scaled=vol_scaled_diag,
                    low_q=label_low_q,
                    high_q=label_high_q,
                )
            
            labeled_mask_diag = ~quantile_labels_diag.isna()
            
            # Build meta-features
            meta_feature_cfg = config.get("meta_feature_engineering", {})
            volume_available = "volume" in market_data.columns
            
            meta_features_diag, meta_features_processed, _, _ = build_meta_features_for_model(
                market_data=market_data,
                primary_signals=primary_signals,
                realized_returns=realized_returns_diag,
                binary_labels=quantile_labels_diag,
                event_durations=event_durations_diag,
                mfe_series=mfe_diag,
                mae_series=mae_diag,
                adaptive_stop_threshold=adaptive_stop,
                horizon=horizon,
                volume_available=volume_available,
                meta_feature_cfg=meta_feature_cfg,
            )
            
            # Compute calibrated probabilities
            _, mean_auc_diag, calibrated_probs_diag, _ = compute_learnability_with_calibration(
                X=meta_features_processed,
                y=quantile_labels_diag,
                realized_returns=realized_returns_diag,
                model_complexity="strong",
                cv_splits=5,
                time_aware_cv=True,
                use_ensemble=False,
            )
            
            # Create "full" labels (before quantile filtering)
            econ_floor = econ_min_mult * effective_tx_cost
            y_full_diag = pd.Series(np.nan, index=realized_returns_diag.index)
            full_mask = ~realized_returns_diag.isna() & (realized_returns_diag.abs() >= econ_floor)
            y_full_diag[full_mask & (realized_returns_diag > 0)] = 1.0
            y_full_diag[full_mask & (realized_returns_diag <= 0)] = 0.0
            
            # 1. Filtering inflation diagnostics
            tprint_info("  → Computing filtering inflation diagnostics...")
            try:
                filtering_diag = compute_filtering_inflation_diagnostics(
                    X=meta_features_processed,
                    y_full=y_full_diag,
                    y_filtered=quantile_labels_diag,
                    realized_returns=realized_returns_diag,
                    volatility=volatility_1d,
                    probabilities=calibrated_probs_diag,
                    cv_splits=5,
                    time_aware_cv=True,
                )
                best_config_diagnostics['filtering_diagnostics'] = filtering_diag
                
                if filtering_diag.get('filtering_is_major_contributor'):
                    tprint_warning("  ⚠️ WARNING: Filtering is a major contributor to AUC inflation")
                if filtering_diag.get('auc_dominated_by_large_moves'):
                    tprint_warning("  ⚠️ WARNING: AUC dominated by large-move events")
                if filtering_diag.get('precision_collapse_detected'):
                    tprint_warning("  ⚠️ WARNING: Precision collapse detected (model only good on easy cases)")
            except Exception as e:
                tprint_warning(f"  ⚠️ Filtering diagnostics failed: {e}")
            
            # 2. Calibration diagnostics
            tprint_info("  → Computing calibration diagnostics...")
            try:
                y_calib = quantile_labels_diag[labeled_mask_diag].values
                probs_calib = calibrated_probs_diag[labeled_mask_diag.values] if len(calibrated_probs_diag) == len(quantile_labels_diag) else calibrated_probs_diag
                returns_calib = realized_returns_diag[labeled_mask_diag].values
                
                calib_diag = compute_calibration_diagnostics(
                    y_true=y_calib,
                    probabilities=probs_calib,
                    realized_returns=returns_calib,
                    transaction_cost=effective_tx_cost,
                    n_bins=10,
                )
                best_config_diagnostics['calibration_diagnostics'] = calib_diag
                
                if calib_diag.get('brier_score') is not None:
                    tprint_info(f"  → Brier Score: {calib_diag['brier_score']:.4f}")
                if calib_diag.get('ece') is not None:
                    tprint_info(f"  → ECE: {calib_diag['ece']:.4f}")
                if not calib_diag.get('is_well_calibrated', True):
                    tprint_warning("  ⚠️ WARNING: Model is miscalibrated (ECE > 0.05)")
            except Exception as e:
                tprint_warning(f"  ⚠️ Calibration diagnostics failed: {e}")
            
            # 3. Robustness diagnostics
            tprint_info("  → Computing robustness diagnostics...")
            try:
                robust_diag = compute_robustness_diagnostics(
                    X=meta_features_processed,
                    y=quantile_labels_diag,
                    realized_returns=realized_returns_diag,
                    regimes=regimes_diag,
                    volatility=volatility_1d,
                    n_folds=5,
                    transaction_cost=effective_tx_cost,
                )
                best_config_diagnostics['robustness_diagnostics'] = robust_diag
                
                if robust_diag.get('worst_fold_auc') is not None:
                    tprint_info(f"  → Worst-fold AUC: {robust_diag['worst_fold_auc']:.3f}")
                if robust_diag.get('auc_cv_std') is not None:
                    tprint_info(f"  → AUC CV Std: {robust_diag['auc_cv_std']:.4f}")
                if not robust_diag.get('is_robust', True):
                    tprint_warning("  ⚠️ WARNING: Model not robust (high CV variance or poor worst-fold)")
            except Exception as e:
                tprint_warning(f"  ⚠️ Robustness diagnostics failed: {e}")
            
            # 4. Class overlap diagnostics
            tprint_info("  → Computing class overlap diagnostics...")
            try:
                overlap_diag = compute_class_overlap_features(
                    X=meta_features_processed,
                    retained_mask=labeled_mask_diag,
                    top_k_features=10,
                )
                best_config_diagnostics['class_overlap_diagnostics'] = overlap_diag
                
                if overlap_diag.get('easy_problem_detected'):
                    tprint_warning("  ⚠️ WARNING: Easy problem detected (retained events form tight cluster)")
            except Exception as e:
                tprint_warning(f"  ⚠️ Class overlap diagnostics failed: {e}")
            
            tprint_success("✅ Comprehensive diagnostics completed for best configuration")
            
        except Exception as diag_exc:
            tprint_warning(f"⚠️ Failed to compute comprehensive diagnostics: {diag_exc}")
            import traceback
            traceback.print_exc()

        # ------------------------------------------------------------------
        # 8) (Disabled) Pareto frontier and knee-point logic
        # ------------------------------------------------------------------
        pareto_solutions: list[Solution] = []
        pareto_front: list[Solution] = []
        knee_solution = None
        knee_params = best_params

        try:
            if candidate_pool:
                auc_values = [float(c.get("mean_auc", np.nan)) for c in candidate_pool]
                auc_values_clean = [v for v in auc_values if np.isfinite(v)]
                auc_median = float(np.median(auc_values_clean)) if auc_values_clean else 0.6

                k_auc = 10.0

                for cand in candidate_pool:
                    edge_val = float(cand.get("edge", 0.0))
                    mean_auc_raw = float(cand.get("mean_auc", 0.0))
                    auc_centered = mean_auc_raw - auc_median
                    if np.isfinite(auc_centered):
                        smooth_auc = float(1.0 / (1.0 + np.exp(-k_auc * auc_centered)))
                    else:
                        smooth_auc = 0.5

                    metrics = {
                        "edge": edge_val,
                        "mean_auc_smooth": smooth_auc,
                        "mean_auc_raw": mean_auc_raw,
                        "learnability": float(cand.get("learnability", 0.0)),
                        "profitability": float(cand.get("profitability", 0.0)),
                        "sharpe_pos": float(cand.get("sharpe_pos", 0.0)),
                    }
                    params_for_sol = cand.get("params") or {}
                    pareto_solutions.append(Solution(metrics=metrics, params=params_for_sol))

                objectives = {"edge": "max", "mean_auc_smooth": "max"}
                pareto_front = compute_pareto_front(pareto_solutions, objectives, use_gpu=True, use_vectorbt=True)
                knee_solution = select_knee_point(pareto_front, objectives)
                if knee_solution and knee_solution.params:
                    tmp_params = best_params.copy()
                    tmp_params.update(knee_solution.params)
                    knee_params = tmp_params
        except Exception as pareto_exc:
            tprint_warning(f"⚠️ Pareto frontier construction failed: {pareto_exc}")
            pareto_solutions = []
            pareto_front = []
            knee_solution = None
            knee_params = best_params

        # Compact run summary for quick log scanning
        try:
            round_results = getattr(optimizer, "round_results", [])
            n_rounds = len(round_results) if isinstance(round_results, list) else None
            total_trials = sum(r.get("trials", 0) for r in round_results) if isinstance(round_results, list) else None
        except Exception:
            n_rounds = None
            total_trials = None

        # Build comprehensive summary with key quality metrics
        mean_auc = best_candidate_metrics.get("mean_auc", 0.5)
        trades_per_day = best_candidate_metrics.get("trades_per_day", 0.0)
        learnability = best_candidate_metrics.get("learnability", 0.0)
        profitability = best_candidate_metrics.get("profitability", 0.0)
        sharpe_pos = best_candidate_metrics.get("sharpe_pos", 0.0)
        n_events = best_candidate_metrics.get("n_events", 0)

        tprint_info(
            "HPO summary → "
            f"symbol={symbol}, timeframe={timeframe}, "
            f"best_score(edge)={best_score:.6f}, "
            f"mean_auc={mean_auc:.4f}, "
            f"trades_per_day={trades_per_day:.3f}, "
            f"learnability={learnability:.4f}, "
            f"profitability={profitability:.4f}, "
            f"sharpe_pos={sharpe_pos:.4f}, "
            f"n_events={n_events}, "
            f"rounds={n_rounds}, trials={total_trials}",
        )

        # Persist best parameters and candidate pool to outcomes/
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # ===== OPTIONAL: Generate diagnostics for recommended configuration =====
        diagnostics_path: str | None = None
        try:
            # Prefer knee point if available, otherwise fall back to best_params
            diag_params: Dict[str, Any] = knee_params if knee_solution else best_params
            # Only run diagnostics when explicitly enabled via the module-level
            # constant to avoid categorical setitem issues in some environments.
            if diag_params and GENERATE_RECOMMENDED_DIAGNOSTICS:
                tprint_info("📊 Generating meta-labeling diagnostics for recommended configuration...")

                # Reconstruct labeling parameters (consistent with labeling_objective)
                profit_thr_base = float(diag_params["profit_thr_base"])
                stop_ratio = float(diag_params["stop_to_profit_ratio"])
                stop_thr_base = max(0.0005, profit_thr_base * stop_ratio)

                horizon = int(diag_params["horizon_bars"])
                # Use safer get() with default 2 if key missing (e.g. older artifact)
                min_spacing = int(diag_params.get("min_event_spacing", 2))

                kalman_Q = float(diag_params.get("kalman_Q", 1e-4))
                kalman_R = float(diag_params.get("kalman_R", 0.01))
                vol_baseline_window = int(diag_params.get("vol_baseline_window", 96))
                profit_mult_min = float(diag_params.get("profit_mult_min", 0.5))
                profit_mult_max = float(diag_params.get("profit_mult_max", 2.0))
                stop_mult_min = float(diag_params.get("stop_mult_min", 0.5))
                stop_mult_max = float(diag_params.get("stop_mult_max", 2.0))

                # Apply same constraints as HPO objective
                horizon = max(8, min(28, horizon))
                if horizon % 2 != 0:
                    horizon = (horizon // 2) * 2
                min_spacing = max(1, min(16, min_spacing))
                vol_baseline_window = max(8, min(512, vol_baseline_window))

                if profit_mult_min > profit_mult_max:
                    profit_mult_min, profit_mult_max = profit_mult_max, profit_mult_min
                if stop_mult_min > stop_mult_max:
                    stop_mult_min, stop_mult_max = stop_mult_max, stop_mult_min

                iso_min_prob = float(diag_params.get("iso_min_prob", 0.05))
                iso_min_prob = max(0.05, min(0.15, iso_min_prob))
                iso_max_prob = 1.0 - iso_min_prob
                iso_max_prob = max(0.85, min(1.0, iso_max_prob))

                q_high = float(diag_params.get("target_clip_high_q", 0.95))
                q_high = max(0.90, min(0.98, q_high))
                q_low = max(0.0, min(0.5, 1.0 - q_high))

                econ_min_mult = float(diag_params.get("econ_min_return_multiple", ECON_MIN_RETURN_MULTIPLE))
                if not np.isfinite(econ_min_mult) or econ_min_mult <= 0:
                    econ_min_mult = float(ECON_MIN_RETURN_MULTIPLE)

                # Recompute adaptive profit/stop thresholds
                vol_baseline = volatility_1d.rolling(vol_baseline_window).mean()
                vol_factor = volatility_1d / (vol_baseline + 1e-8)

                high_prices = market_data["high"] if "high" in market_data.columns else market_data["close"]
                low_prices = market_data["low"] if "low" in market_data.columns else market_data["close"]
                close_prices = market_data["close"]

                tr1 = high_prices - low_prices
                tr2 = (high_prices - close_prices.shift(1)).abs()
                tr3 = (low_prices - close_prices.shift(1)).abs()
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

                trend_atr_window = int(config.get("trend_strength_atr_window", 14))
                atr_series = true_range.rolling(window=trend_atr_window, min_periods=1).mean()

                trend_delta_lookback = int(config.get("trend_strength_delta_lookback", 4))
                price_delta = close_prices.diff(trend_delta_lookback).abs()

                trend_strength = (price_delta / (atr_series + 1e-8)).replace([np.inf, -np.inf], np.nan)
                trend_strength = trend_strength.clip(
                    lower=0.0,
                    upper=float(config.get("trend_strength_clip", 5.0)),
                ).fillna(0.0)

                trend_alpha = float(config.get("trend_strength_alpha_profit", 0.5))
                trend_beta = float(config.get("trend_strength_beta_stop", 0.5))

                profit_factor = 1.0 + trend_alpha * trend_strength
                stop_factor = 1.0 + trend_beta * trend_strength

                adaptive_profit = profit_thr_base * vol_factor * profit_factor
                adaptive_stop = stop_thr_base * vol_factor * stop_factor
                adaptive_profit = adaptive_profit.clip(
                    lower=profit_thr_base * profit_mult_min,
                    upper=profit_thr_base * profit_mult_max,
                )
                adaptive_stop = adaptive_stop.clip(
                    lower=stop_thr_base * stop_mult_min,
                    upper=stop_thr_base * stop_mult_max,
                )

                (
                    realized_returns,
                    binary_labels,
                    exit_reasons,
                    event_durations,
                    mfe_series_diag,
                    mae_series_diag,
                    _binary_labels_long_diag,  # Not used in diagnostics
                    _binary_labels_short_diag,  # Not used in diagnostics
                ) = compute_realized_returns(
                    market_data,
                    primary_signals,
                    profit_threshold=adaptive_profit,
                    stop_threshold=adaptive_stop,
                    horizon=horizon,
                    transaction_cost=DEFAULT_TRANSACTION_COST,
                    min_event_spacing=min_spacing,
                )

                # Guard: if too few events, skip diagnostics
                labeled_mask = ~binary_labels.isna()
                if int(labeled_mask.sum()) < 100:
                    tprint_warning(
                        "⚠️ Recommended config produced too few events for diagnostics; skipping report generation",
                        "WARNING",
                    )
                else:
                    # Kalman smoothing
                    smoothed_labels, _ = kalman_smooth_labels(
                        binary_labels,
                        Q=kalman_Q,
                        R=kalman_R,
                        volatility=volatility_1d,
                    )

                    prob_series = smoothed_labels.clip(0.0, 1.0)
                    prob_clipped = prob_series.clip(iso_min_prob, iso_max_prob)

                    # Fit probability→expected-return mapping
                    iso_reg = fit_probability_to_return_mapping(
                        probabilities=prob_clipped.values,
                        realized_returns=realized_returns.values,
                        method="isotonic",
                        econ_min_return_multiple=econ_min_mult,
                    )

                    # Translate to long/short targets
                    target_long, target_short = translate_to_targets_with_isotonic(
                        realized_returns=realized_returns,
                        probabilities=prob_clipped.values,
                        signals=primary_signals,
                        iso_regressor=iso_reg,
                    )

                    # Build labeled_data in the same spirit as production step
                    labeled_data = market_data.copy()

                    # Drop any existing derived columns that might carry categorical dtypes
                    # so that we can safely assign fresh non-categorical Series
                    derived_cols = [
                        "log_ret",
                        "volatility_1d",
                        "realized_return",
                        "binary_label",
                        "smoothed_label",
                        "meta_probability",
                        "exit_reason",
                        "event_duration_bars",
                        "target_long",
                        "target_short",
                        "primary_signal",
                    ]
                    labeled_data = labeled_data.drop(columns=[c for c in derived_cols if c in labeled_data.columns], errors="ignore")

                    log_ret = np.log(market_data["close"]).diff()
                    labeled_data["log_ret"] = log_ret
                    labeled_data["volatility_1d"] = volatility_1d
                    labeled_data["realized_return"] = realized_returns
                    labeled_data["binary_label"] = binary_labels
                    labeled_data["smoothed_label"] = smoothed_labels
                    labeled_data["meta_probability"] = prob_clipped.values
                    labeled_data["exit_reason"] = exit_reasons
                    labeled_data["event_duration_bars"] = event_durations
                    labeled_data["target_long"] = target_long
                    labeled_data["target_short"] = target_short
                    labeled_data["primary_signal"] = primary_signals["consensus"]

                    # Meta-features for diagnostics (same helper as production)
                    meta_feature_cfg = config.get("meta_feature_engineering", {})
                    volume_available = "volume" in market_data.columns

                    meta_features_diag, meta_features_model_diag, _, _ = build_meta_features_for_model(
                        market_data=market_data,
                        primary_signals=primary_signals,
                        realized_returns=realized_returns,
                        binary_labels=binary_labels,
                        event_durations=event_durations,
                        mfe_series=mfe_series_diag,
                        mae_series=mae_series_diag,
                        adaptive_stop_threshold=adaptive_stop,
                        horizon=horizon,
                        volume_available=volume_available,
                        meta_feature_cfg=meta_feature_cfg,
                    )

                    # Simple RF meta-model for feature importances
                    X_diag = meta_features_model_diag[labeled_mask].fillna(0)
                    y_diag = binary_labels[labeled_mask]
                    final_model = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=6,
                        min_samples_leaf=20,
                        n_jobs=-1,
                        random_state=42,
                    )
                    if len(y_diag.unique()) >= 2 and len(y_diag) >= 50:
                        final_model.fit(X_diag, y_diag)
                    else:
                        # Fallback: still fit to avoid attribute errors in diagnostics
                        final_model.fit(X_diag, y_diag)

                    # Slightly enriched config for diagnostics
                    diag_config = dict(config)
                    diag_config["horizon"] = horizon
                    diag_config["profit_thr_base"] = profit_thr_base
                    diag_config["stop_thr_base"] = stop_thr_base

                    # Sanitize any categorical columns to avoid setitem/category issues
                    labeled_data_for_diag = labeled_data.copy()
                    cat_cols = labeled_data_for_diag.select_dtypes(include=["category"]).columns
                    if len(cat_cols) > 0:
                        for col in cat_cols:
                            labeled_data_for_diag[col] = labeled_data_for_diag[col].astype(object)

                    # Also ensure core Series are not categorical
                    binary_labels_diag = pd.Series(
                        binary_labels.astype(float).values,
                        index=binary_labels.index,
                    )
                    exit_reasons_diag = None
                    event_durations_diag = None
                    if exit_reasons is not None:
                        exit_reasons_diag = pd.Series(
                            exit_reasons.astype(object).values,
                            index=exit_reasons.index,
                        )
                    if event_durations is not None:
                        event_durations_diag = pd.Series(
                            event_durations.astype(float).values,
                            index=event_durations.index,
                        )

                    diagnostics_path_obj = generate_diagnostics_report(
                        labeled_data=labeled_data_for_diag,
                        meta_features=meta_features_diag,
                        binary_labels=binary_labels_diag,
                        realized_returns=realized_returns,
                        smoothed_labels=smoothed_labels,
                        probabilities=prob_clipped.values,
                        final_model=final_model,
                        config=diag_config,
                        output_dir=outcomes_dir,
                        exit_reasons=exit_reasons_diag,
                        event_durations=event_durations_diag,
                        mfe_series=mfe_series_diag,
                        mae_series=mae_series_diag,
                        target_long=target_long,
                        target_short=target_short,
                    )
                    diagnostics_path = str(diagnostics_path_obj)

                    tprint_success(
                        f"📊 Saved diagnostics for recommended labeling configuration to {diagnostics_path}",
                    )
        except Exception as diag_exc:
            tprint_warning(f"⚠️ Failed to generate diagnostics for recommended configuration: {diag_exc}")

        # ===== SAVE BEST PARAMS JSON =====
        json_name = f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_{timestamp}.json"
        json_path = outcomes_dir / json_name

        try:
            # Get best edge from the best candidate
            best_candidate_edge = 0.0
            if candidate_pool:
                sorted_candidates = sorted(candidate_pool, key=lambda x: x.get('edge', x.get('combined', 0)), reverse=True)
                if sorted_candidates:
                    best_candidate_edge = sorted_candidates[0].get('edge', 0.0)

            # Build output dict with diagnostics
            output_dict = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "best_score": best_score,
                "best_edge": best_candidate_edge,
                "best_params": best_params,
                "knee_params": knee_params,
                "pareto_front_size": len(pareto_front),
                "total_trials": all_trials_count,
                "stage_results": [
                    {
                        "stage": sr["stage"],
                        "complexity": sr["complexity"],
                        "best_score": sr["best_score"],
                        "trials": sr["trials"],
                    }
                    for sr in stage_results
                ],
            }
            
            # Add comprehensive diagnostics for best config
            if best_config_diagnostics:
                output_dict["best_config_diagnostics"] = best_config_diagnostics
                
                # Add summary warnings as top-level fields
                filtering_diag = best_config_diagnostics.get('filtering_diagnostics', {})
                calib_diag = best_config_diagnostics.get('calibration_diagnostics', {})
                robust_diag = best_config_diagnostics.get('robustness_diagnostics', {})
                overlap_diag = best_config_diagnostics.get('class_overlap_diagnostics', {})
                
                output_dict["diagnostics_summary"] = {
                    "filtering_is_major_contributor": filtering_diag.get('filtering_is_major_contributor', False),
                    "auc_dominated_by_large_moves": filtering_diag.get('auc_dominated_by_large_moves', False),
                    "precision_collapse_detected": filtering_diag.get('precision_collapse_detected', False),
                    "is_well_calibrated": calib_diag.get('is_well_calibrated', True),
                    "brier_score": calib_diag.get('brier_score'),
                    "ece": calib_diag.get('ece'),
                    "mce": calib_diag.get('mce'),
                    "is_robust": robust_diag.get('is_robust', True),
                    "worst_fold_auc": robust_diag.get('worst_fold_auc'),
                    "auc_cv_std": robust_diag.get('auc_cv_std'),
                    "easy_problem_detected": overlap_diag.get('easy_problem_detected', False),
                }
            
            with open(json_path, "w") as f:
                json.dump(output_dict, f, indent=2, default=str)
            tprint_success(f"💾 Saved best labeling HPO params to {json_path}")
        except Exception as save_exc:
            tprint_warning(f"⚠️ Failed to save best_params JSON: {save_exc}")
            json_path = None

        # ===== SAVE CANDIDATE POOL CSV =====
        csv_name = f"meta_labeling_hpo_candidate_pool_{symbol}_{timeframe}_{timestamp}.csv"
        csv_path = outcomes_dir / csv_name

        try:
            if not candidate_pool:
                tprint_warning("⚠️ Candidate pool is empty; skipping candidate CSV export")
                csv_path = None
            else:
                candidate_df = pd.DataFrame(candidate_pool)

                if 'params' in candidate_df.columns:
                    params_df = pd.json_normalize(candidate_df['params'])
                    candidate_df = candidate_df.drop(columns=['params'])
                    candidate_df = pd.concat([candidate_df, params_df], axis=1)

                if 'edge' in candidate_df.columns:
                    candidate_df = candidate_df.sort_values('edge', ascending=False)
                elif 'combined' in candidate_df.columns:
                    candidate_df = candidate_df.sort_values('combined', ascending=False)

                candidate_df.to_csv(csv_path, index=False, float_format='%.6f')
                tprint_success(f"💾 Saved {len(candidate_pool)} candidate configs to {csv_path}")
        except Exception as csv_exc:
            tprint_warning(f"⚠️ Failed to save candidate pool CSV: {csv_exc}")
            csv_path = None

        # ===== SAVE PARETO FRONTIER CSV =====
        pareto_csv_name = f"meta_labeling_hpo_pareto_front_{symbol}_{timeframe}_{timestamp}.csv"
        pareto_csv_path = outcomes_dir / pareto_csv_name

        try:
            if not pareto_front:
                pareto_csv_path = None
            else:
                pareto_data: list[dict[str, Any]] = []
                for sol in pareto_front:
                    row = dict(sol.metrics)
                    if sol.params:
                        row.update(sol.params)
                    pareto_data.append(row)

                pareto_df = pd.DataFrame(pareto_data)
                if 'edge' in pareto_df.columns:
                    pareto_df = pareto_df.sort_values('edge', ascending=False)
                elif 'combined' in pareto_df.columns:
                    pareto_df = pareto_df.sort_values('combined', ascending=False)

                pareto_df.to_csv(pareto_csv_path, index=False, float_format='%.6f')
                tprint_success(f"💾 Saved {len(pareto_front)} Pareto solutions to {pareto_csv_path}")
        except Exception as pareto_exc:
            tprint_warning(f"⚠️ Failed to save Pareto frontier CSV: {pareto_exc}")
            pareto_csv_path = None

        # ===== SAVE COMPREHENSIVE MARKDOWN REPORT =====
        md_name = f"meta_labeling_hpo_report_{symbol}_{timeframe}_{timestamp}.md"
        md_path = outcomes_dir / md_name

        try:
            with open(md_path, "w") as f:
                f.write(f"# Meta-Labeling HPO Report\n\n")
                f.write(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n")
                f.write(f"**Symbol:** {symbol} | **Exchange:** {exchange} | **Timeframe:** {timeframe}\n\n")
                f.write(f"---\n\n")

                # Summary
                f.write(f"## Summary\n\n")
                f.write(f"- **Total Configurations Evaluated:** {len(candidate_pool)}\n")
                f.write(f"- **Total Trials:** {all_trials_count}\n")
                f.write(f"- **Best Edge:** {best_score:.6f}\n")
                f.write(f"- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration\n\n")

                # Multi-Stage Results
                f.write(f"## Multi-Stage HPO Results\n\n")
                f.write(f"| Stage | Complexity | Best Score | Trials |\n")
                f.write(f"|-------|------------|------------|--------|\n")
                for sr in stage_results:
                    f.write(f"| {sr['stage']} | {sr['complexity']} | {sr['best_score']:.4f} | {sr['trials']} |\n")
                f.write(f"\n")

                # Best Parameters
                f.write(f"## Best Parameters (Highest Edge)\n\n")
                f.write(f"```json\n")
                f.write(json.dumps(best_params, indent=2))
                f.write(f"\n```\n\n")

                # Per-regime metrics for best-edge configuration, if available
                best_regime_metrics = None
                try:
                    if candidate_pool:
                        best_candidate = max(
                            candidate_pool,
                            key=lambda x: x.get('edge', x.get('combined', 0)),
                        )
                        best_regime_metrics = best_candidate.get('per_regime_metrics')
                except Exception:
                    best_regime_metrics = None

                if best_regime_metrics:
                    f.write(f"## Per-Regime Metrics (Best Edge Configuration)\n\n")
                    f.write(f"| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |\n")
                    f.write(f"|--------|----------|----------------|----------|----------|------|-----|\n")
                    for reg_key, m in best_regime_metrics.items():
                        try:
                            f.write(
                                f"| {reg_key} | {int(m.get('n_events', 0))} | "
                                f"{float(m.get('trades_per_day', 0.0)):.3f} | "
                                f"{float(m.get('mean_pos', 0.0)):.5f} | "
                                f"{float(m.get('mean_neg', 0.0)):.5f} | "
                                f"{float(m.get('edge', 0.0)):.6f} | "
                                f"{float(m.get('auc', 0.5)):.3f} |\n"
                            )
                        except Exception:
                            continue
                    f.write("\n")

                # Knee Point Parameters
                if knee_solution:
                    f.write(f"## Recommended Parameters (Pareto Knee Point)\n\n")
                    f.write(f"Balanced trade-off between learnability and profitability:\n\n")
                    f.write(f"- **Learnability:** {knee_solution.metrics['learnability']:.4f}\n")
                    f.write(f"- **Profitability:** {knee_solution.metrics['profitability']:.4f}\n")
                    f.write(f"- **Mean AUC:** {knee_solution.metrics.get('mean_auc_raw', knee_solution.metrics.get('mean_auc_smooth', 0)):.4f}\n")
                    f.write(f"- **Sharpe (Winners):** {knee_solution.metrics.get('sharpe_pos', 0):.4f}\n\n")
                    f.write(f"```json\n")
                    f.write(json.dumps(knee_params, indent=2))
                    f.write(f"\n```\n\n")

                # Underfit Diagnostics (if available)
                final_stage_candidates = [
                    c for c in candidate_pool
                    if c.get('model_complexity') == 'strong' and c.get('underfit_diagnostics')
                ]
                if final_stage_candidates:
                    f.write(f"## Underfit Diagnostics (Final Stage)\n\n")

                    # Get diagnostics from best strong candidate
                    best_strong = sorted(final_stage_candidates, key=lambda x: x['combined'], reverse=True)[0]
                    diag = best_strong.get('underfit_diagnostics', {})

                    if diag.get('is_underfit'):
                        f.write(f"**⚠️ Underfit Detected:** The model shows signs of underfitting.\n\n")
                        f.write(f"Indicators:\n")
                        for indicator in diag.get('underfit_indicators', []):
                            f.write(f"- {indicator}\n")
                        f.write(f"\n")
                    else:
                        f.write(f"**✅ No Significant Underfit:** The model appears well-fitted.\n\n")

                    # Learning curves by data fraction
                    if diag.get('learning_curve_fractions'):
                        f.write(f"### Learning Curves (Data Fractions)\n\n")
                        f.write(f"| Data Fraction | AUC |\n")
                        f.write(f"|---------------|-----|\n")
                        for frac, auc in sorted(diag['learning_curve_fractions'].items()):
                            f.write(f"| {frac:.0%} | {auc:.4f} |\n")
                        f.write(f"\n")

                    # Learning curves by depth
                    if diag.get('learning_curve_depths'):
                        f.write(f"### Learning Curves (Model Depths)\n\n")
                        f.write(f"| Depth | AUC |\n")
                        f.write(f"|-------|-----|\n")
                        for depth, auc in sorted(diag['learning_curve_depths'].items()):
                            f.write(f"| {depth} | {auc:.4f} |\n")
                        f.write(f"\n")

                    # Feature importance concentration
                    if diag.get('feature_importance_concentration') is not None:
                        f.write(f"**Feature Importance Concentration (Top 5):** {diag['feature_importance_concentration']:.1%}\n\n")

                    if diag.get('feature_group_importance'):
                        groups = diag['feature_group_importance']
                        f.write("**Feature Group Importance:**\n\n")
                        for g, share in groups.items():
                            f.write(f"- {g}: {share:.1%}\n")
                        f.write("\n")

                    if diag.get('top_feature_importances'):
                        f.write("**Top Features by Importance:**\n\n")
                        for item in diag['top_feature_importances'][:10]:
                            try:
                                name = item.get('name', '')
                                imp = float(item.get('importance', 0.0))
                                f.write(f"- {name}: {imp:.2%}\n")
                            except Exception:
                                continue
                        f.write("\n")

                    # Probe vs deep AUC diff
                    if diag.get('probe_vs_deep_auc_diff') is not None:
                        f.write(f"**Probe vs Deep Model AUC Improvement:** {diag['probe_vs_deep_auc_diff']:.1%}\n\n")

                # Regularization Checks Summary
                f.write(f"## Regularization & Scoring\n\n")
                f.write(f"### Realistic P&L Edge Metric\n\n")
                f.write(f"The primary scoring metric is the **Realistic P&L Edge**:\n\n")
                f.write(f"```\n")
                f.write(f"Edge = (Mean_Return_Label1 - Transaction_Cost) × max(0, 2×AUC - 1)\n")
                f.write(f"```\n\n")
                f.write(f"This metric penalizes 'profitable but unlearnable' strategies more realistically:\n")
                f.write(f"- If AUC = 0.5 (random), Edge = 0 regardless of profitability\n")
                f.write(f"- If AUC = 1.0 (perfect), you capture full mean return minus cost\n\n")
                f.write(f"### Regularization Checks\n\n")
                f.write(f"All configurations were evaluated with:\n\n")
                f.write(f"1. **Isotonic Calibration:** Probabilities calibrated to align with real expected returns\n")
                f.write(f"2. **Temporal Stability:** Rolling window AUC variance penalty\n")
                f.write(f"3. **Learnability Threshold:** Mean AUC < 0.7 heavily penalized\n")
                f.write(f"4. **Profit/Stop Constraint:** Profit threshold must be ≥ 1.5× stop threshold\n")
                f.write(f"5. **Label Balance:** Entropy-based balance scoring\n")
                f.write(f"6. **Early Stopping:** Per-trial and global early stopping to prevent overfitting\n\n")

                # Artifacts
                f.write(f"## Artifacts\n\n")
                f.write(f"- **Best Params JSON:** `{json_path.name if json_path else 'N/A'}`\n")
                f.write(f"- **Candidate Pool CSV:** `{csv_path.name if csv_path else 'N/A'}`\n")
                f.write(f"- **Pareto Frontier CSV:** `{pareto_csv_path.name if pareto_csv_path else 'N/A'}`\n")
                f.write(f"- **This Report:** `{md_name}`\n\n")

            tprint_success(f"📄 Saved comprehensive report to {md_path}")
        except Exception as md_exc:
            tprint_warning(f"⚠️ Failed to save markdown report: {md_exc}")
            md_path = None

        # Persist per-round HPO metrics to CSV for analysis
        csv_path = None
        try:
            round_results = getattr(optimizer, "round_results", [])
            if isinstance(round_results, list) and round_results:
                rows: list[dict[str, Any]] = []
                for rr in round_results:
                    rows.append(
                        {
                            "round": rr.get("round"),
                            "best_score": rr.get("best_score"),
                            "improvement": rr.get("improvement"),
                            "time_seconds": rr.get("time"),
                            "trials": rr.get("trials"),
                        }
                    )

                df_rounds = pd.DataFrame(rows)
                csv_name = (
                    f"meta_labeling_hpo_round_metrics_{symbol}_{timeframe}_{timestamp}.csv"
                )
                csv_path = outcomes_dir / csv_name
                df_rounds.to_csv(csv_path, index=False)
                tprint_success(f"💾 Saved HPO round metrics to {csv_path}")
        except Exception as csv_exc:
            tprint_warning(f"⚠️ Failed to save HPO round metrics CSV: {csv_exc}")

        # Compute best edge from candidate pool
        best_edge = 0.0
        if candidate_pool:
            best_candidate = max(candidate_pool, key=lambda x: x.get('edge', x.get('combined', 0)))
            best_edge = best_candidate.get('edge', 0.0)

        metrics: Dict[str, Any] = {
            "best_score": best_score,
            "best_edge": best_edge,
            "best_params": best_params,
            "best_params_json": str(json_path) if json_path is not None else None,
            "round_metrics_csv": str(csv_path) if csv_path is not None else None,
            "recommended_diagnostics_path": diagnostics_path,
            "total_trials": all_trials_count,
            "stage_results": stage_results,
            "pareto_frontier_size": len(pareto_front),
            "candidate_pool_size": len(candidate_pool),
        }
        
        # Add diagnostics summary to metrics
        if best_config_diagnostics:
            metrics["best_config_diagnostics"] = best_config_diagnostics

        artifacts: Dict[str, Any] = {}
        if json_path is not None:
            artifacts["best_params_json"] = str(json_path)
        if csv_path is not None:
            artifacts["round_metrics_csv"] = str(csv_path)
        if diagnostics_path is not None:
            artifacts["recommended_diagnostics_path"] = diagnostics_path

        return {
            "success": True,
            "metrics": metrics,
            "artifacts": artifacts,
        }


def register_meta_labeling_hpo_experiment_step() -> None:
    """Register the meta-labeling HPO experiment step in the registry."""
    from src.training.steps.base_step import step_registry

    step_registry.register("meta_labeling_hpo_experiment", MetaLabelingHPOExperimentStep)
    tprint("✅ Meta-labeling HPO experiment step registered", "SUCCESS")


# Auto-register when module is imported
register_meta_labeling_hpo_experiment_step()
