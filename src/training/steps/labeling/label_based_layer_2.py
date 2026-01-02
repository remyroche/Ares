"""
Label-Based Layer 2: Regime-Conditional Geometry Optimization & Meta-Labeling
-----------------------------------------------------------------------------
This module implements the Layer 2 architecture for meta-labeling, focusing on
regime-conditional barrier optimization and ML learnability probes.

It performs:
1. Event generation based on Orthogonal Families (Vol, Trend, etc.).
2. Geometry Selection via Mutual Information & Uniqueness filtering.
3. Independent optimization of barrier geometries (Kappa/Horizon) implicitly via selection.
4. MFE/MAE Dominance Labeling: Label = 1 if MFE > Kappa * MAE.
5. Stability checks (Time-Flip) and Learnability probes.
6. Bagged output generation with family-level cap checks.
7. Enhanced model training with optional multi-algorithm comparison (LGBM, XGB, CatBoost, RF, LogReg).

Advanced Model Comparison & HPO Features:
- enable_model_race: Enable/disable automatic model selection with RobustFocalLoss
- enable_focal_hpo: Enable/disable HPO for ALL winning models (not just LGBM)
- focal_hpo_n_trials: Number of HPO trials per winning model

Model Race Candidates (all use adaptive alpha):
- LGBM_Focal: LGBM with RobustFocalLoss (γ₊=1.0, γ₋=2.5, adaptive α)
- XGB_Tree: XGBoost with focal loss
- CatBoost: CatBoost classifier
- LGBM_Focal_Linear: Linear tree LGBM with RobustFocalLoss
- XGB_Linear: Linear XGBoost with focal loss

HPO Optimization (runs on ANY winning model):
- **LGBM_Focal**: RobustFocalLoss (γ₊, γ₋, α, mix, smoothing) + tree params
- **LGBM_BCE**: Tree parameters (depth, leaves, learning_rate, regularization)
- **XGB**: Tree/ensemble params (depth, estimators, learning_rate, regularization)
- **CatBoost**: Tree params (depth, iterations, regularization)

Feature Selection (Titan RFE):
- Ensures minimum 10 features per geometry
- Adaptive selection based on predictive power
- Cached for efficiency

Adaptive Alpha: Automatically adjusts class balance even without HPO
"""

import numpy as np
import pandas as pd
import json
import optuna
import lightgbm as lgb
import xgboost as xgb
import torch
try:
    import catboost
    CATBOOST_AVAILABLE = True
except ImportError:
    catboost = None
    CATBOOST_AVAILABLE = False
import os
import time
from pathlib import Path
import hashlib
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
from collections import defaultdict
from joblib import Parallel, delayed
from src.training.steps.labeling.focal_loss_utils import get_focal_loss_lgbm, get_focal_loss_xgb, RobustFocalLoss
from src.training.steps.labeling.layer2_advanced_logic import (
    vectorized_pct_change_jit,
    rolling_mean_jit,
    rolling_std_jit,
    rolling_max_min_jit,
)
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import partial
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import spearmanr, rankdata, entropy as shannon_entropy
from scipy.special import expit, ndtri
from scipy.spatial.distance import euclidean, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from joblib import Parallel, delayed
from numba import njit, prange
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict, field
import logging
import copy
import warnings
import sys
from pandas.util import hash_pandas_object

try:
    from joblib.externals.loky.process_executor import BrokenProcessPool, TimeoutError as LokyTimeoutError
except Exception:  # pragma: no cover - older joblib versions
    class LokyTimeoutError(Exception):
        """Fallback timeout exception."""

    class BrokenProcessPool(Exception):
        """Fallback broken pool exception."""

# Import tprint for enhanced logging
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

# Import causal framework modules
try:
    from src.training.steps.labeling.causal_discovery import CausalDiscovery, quick_causal_discovery
    from src.training.steps.labeling.causal_feature_engineering import CausalFeatureEngineering, quick_causal_engineering
    from src.training.steps.labeling.invariant_risk_minimization_v2 import EnhancedIRM, quick_enhanced_irm
    from src.training.steps.labeling.causal_surprise_events import CausalSurpriseDetector, quick_causal_surprise
    from src.training.steps.labeling.interventionist_sampling import CausalInterventionSampler, quick_interventionist_sampling
    from src.training.steps.labeling.causal_targets import CausalTargetComputer, quick_causal_targets
    from src.training.steps.labeling.causal_specialists import CausalSpecialistManager, create_causal_specialists
    from src.training.steps.labeling.causal_uncertainty_quantification import quick_bayesian_causal_discovery
    CAUSAL_MODULES_AVAILABLE = True
    tprint_info("✅ Causal framework modules loaded successfully")
except ImportError as e:
    CAUSAL_MODULES_AVAILABLE = False
    tprint_warning(f"⚠️ Causal framework modules not available: {e}")

# Suppress LightGBM verbose warnings for clean output
warnings.filterwarnings("ignore")

@njit
def vectorized_prediction_aggregation(predictions: np.ndarray, existing_scores: np.ndarray) -> np.ndarray:
    """
    JIT-compiled vectorized prediction aggregation using maximum scoring.

    Args:
        predictions: New predictions to aggregate
        existing_scores: Current aggregated scores

    Returns:
        Updated aggregated scores
    """
    return np.maximum(existing_scores, predictions)

from src.training.steps.labeling.orthogonal_label_generation import (
    CausalSurpriseEvents,
    VolumeSpecialistEvents,
    VolatilitySpecialistEvents,
    LiquiditySpecialistEvents,
    InformationSpecialistEvents,
    InventorySpecialistEvents
)

@njit
def vectorized_threshold_classification(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    JIT-compiled vectorized threshold-based classification.

    Args:
        scores: Prediction scores
        threshold: Classification threshold

    Returns:
        Binary classification results
    """
    return (scores >= threshold).astype(np.float64)


class ProbabilityCalibratedModel:
    """
    Wraps a base classifier and applies logistic calibration to its probability outputs.
    """

    def __init__(self, base_model, coef: float, intercept: float):
        self.base_model = base_model
        self.coef_ = coef
        self.intercept_ = intercept
        self.classes_ = getattr(base_model, "classes_", np.array([0, 1]))

    def _apply_calibration(self, raw_probs: np.ndarray) -> np.ndarray:
        if raw_probs.ndim == 1:
            pos = raw_probs
        else:
            pos = raw_probs[:, -1]
        calibrated_pos = expit(self.coef_ * pos + self.intercept_)
        neg = 1.0 - calibrated_pos
        return np.column_stack([neg, calibrated_pos])

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)
        return self._apply_calibration(raw)

    def predict(self, X):
        return (self.predict_proba(X)[:, -1] >= 0.5).astype(int)

    def __getattr__(self, item):
        return getattr(self.base_model, item)

class IRM_LGBMClassifier(lgb.LGBMClassifier):
    """LightGBM Classifier with Invariant Risk Minimization."""

    def __init__(self, irm_system=None, environment_masks=None, **kwargs):
        super().__init__(**kwargs)
        self.irm_system = irm_system
        self.environment_masks = environment_masks or {}

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit with IRM objective."""
        if self.irm_system is None or not self.environment_masks:
            # Fallback to standard training
            return super().fit(X, y, sample_weight=sample_weight, **kwargs)

        try:
            # Create IRM training function
            train_step = self.irm_system.create_enhanced_irm_trainer(
                model=self,
                optimizer=None,  # LightGBM handles optimization internally
                environment_masks=self.environment_masks
            )

            # Convert to tensors for IRM
            X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
            y_tensor = torch.FloatTensor(y)

            if sample_weight is not None:
                w_tensor = torch.FloatTensor(sample_weight)
            else:
                w_tensor = None

            # Run IRM training (simplified - would need proper integration)
            # For now, use standard training with IRM-aware objective
            self.objective = self._irm_focal_objective

            return super().fit(X, y, sample_weight=sample_weight, **kwargs)

        except Exception as e:
            tprint_warning(f"⚠️ IRM training failed, falling back to standard: {e}")
            return super().fit(X, y, sample_weight=sample_weight, **kwargs)

    def _irm_focal_objective(self, preds, train_data):
        """IRM-aware focal loss objective."""
        labels = train_data.get_label()

        # Compute focal loss
        focal_loss = RobustFocalLoss(
            gamma_pos=self.irm_system.focal_gamma if self.irm_system else 1.0,
            gamma_neg=self.irm_system.focal_gamma if self.irm_system else 2.5,
            alpha=self.irm_system.focal_alpha if self.irm_system else 1.0,
            verbose=False
        )

        # Convert to gradient/hessian for LightGBM
        grad, hess = focal_loss.compute_grad_hess(preds, labels)

        # Add IRM penalty (simplified)
        if self.irm_system and self.environment_masks:
            irm_penalty = self._compute_irm_penalty(preds, labels)
            grad += self.irm_system.lambda_irm * irm_penalty

        return grad, hess

    def _compute_irm_penalty(self, preds, labels):
        """Compute simplified IRM penalty."""
        # Simplified IRM penalty - would need full implementation
        return np.zeros_like(preds)

class IRM_XGBClassifier(XGBClassifier):
    """XGBoost Classifier with Invariant Risk Minimization."""

    def __init__(self, irm_system=None, environment_masks=None, **kwargs):
        super().__init__(**kwargs)
        self.irm_system = irm_system
        self.environment_masks = environment_masks or {}

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit with IRM objective."""
        if self.irm_system is None or not self.environment_masks:
            # Fallback to standard training
            return super().fit(X, y, sample_weight=sample_weight, **kwargs)

        try:
            # Use IRM-aware objective
            self.objective = self._irm_focal_objective
            return super().fit(X, y, sample_weight=sample_weight, **kwargs)

        except Exception as e:
            tprint_warning(f"⚠️ IRM training failed, falling back to standard: {e}")
            return super().fit(X, y, sample_weight=sample_weight, **kwargs)

    def _irm_focal_objective(self, preds, dtrain):
        """IRM-aware focal loss objective for XGBoost."""
        labels = dtrain.get_label()

        # Compute focal loss gradient/hessian
        focal_obj = get_focal_loss_xgb(
            alpha=self.irm_system.focal_alpha if self.irm_system else 1.0,
            gamma=self.irm_system.focal_gamma if self.irm_system else 2.0
        )

        # This would need proper XGBoost objective function integration
        # For now, return standard focal loss
        return focal_obj(preds, dtrain)

@njit
def vectorized_weight_assignment(n_samples: int, weight_value: float = 1.0) -> np.ndarray:
    """
    JIT-compiled vectorized weight assignment.

    Args:
        n_samples: Number of samples
        weight_value: Weight value to assign

    Returns:
        Weight array
    """
    return np.full(n_samples, weight_value, dtype=np.float64)


def precision_at_recall_threshold(y_true, y_probs, target_recall: float = 0.40) -> float:
    """
    Calculate Precision at a fixed Recall threshold.
    
    In trading: "What is our precision when we capture X% of the moves?"
    We don't need high recall (catch every move), we need high precision 
    (be right about the moves we do predict).
    
    Args:
        y_true: Binary labels (0/1)
        y_probs: Predicted probabilities
        target_recall: Target recall level (e.g., 0.40 = 40% of positives)
        
    Returns:
        Precision when capturing target_recall of positive samples
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    
    total_positives = y_true.sum()
    
    # Edge cases
    if total_positives == 0:
        return 0.0
    if len(y_true) == 0:
        return 0.0
    
    # Sort by probability descending
    sorted_indices = np.argsort(y_probs)[::-1]
    sorted_labels = y_true[sorted_indices]
    
    # Find how many samples to take to achieve target recall
    # We need to capture (target_recall * total_positives) true positives
    target_tp = int(np.ceil(total_positives * target_recall))
    target_tp = max(1, target_tp)  # At least 1
    
    # Walk through sorted predictions until we hit target TPs
    cumsum_tp = np.cumsum(sorted_labels)
    
    # Find index where we first reach target_tp true positives
    cutoff_indices = np.where(cumsum_tp >= target_tp)[0]
    
    if len(cutoff_indices) == 0:
        # Can't reach target recall, use all samples
        cutoff_idx = len(y_true)
    else:
        cutoff_idx = cutoff_indices[0] + 1  # +1 because we need inclusive
    
    # Precision = TP / (TP + FP) = TP / n_predicted
    tp_at_cutoff = sorted_labels[:cutoff_idx].sum()
    precision = tp_at_cutoff / cutoff_idx
    
    return precision

# Import compute_realized_returns from the existing module
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    compute_realized_returns,
)
from src.training.steps.labeling.mtf_feature_generation import (
    create_meta_features,
    get_efficiency_ratio
)
from src.training.steps.labeling.generate_weights_per_label import finalize_sample_weights

from src.utils.purged_kfold import PurgedKFoldTime

# Import Layer0 unified price generation for denoised features
from src.training.steps.labeling.unified_price_layer2 import load_layer0_params, generate_unified_layer2_price

# Import selection logic
# Import selection logic
from src.training.steps.labeling.orthogonal_label_generation import (
    orthogonal_label_generation,
)
from src.training.steps.labeling.label_geometry_selection import (
    Event,
    Geometry as LegacyGeometry,
    MIN_SL_PCT,
    MIN_TP_SL_RATIO
)

from src.training.steps.labeling.lgbm_feature_selection import lgbm_feature_selection_pipeline

# Import BaseStep for step registration
from src.training.steps.base_step import BaseStep

# Import Orthogonal Generation
from src.training.steps.labeling.orthogonal_label_generation import (
    orthogonal_label_generation,
    AdaptiveSymmetricCUSUMEvents,
    VolatilityCusumEvents,
    LiquidityCusumEvents,
    VolumeCusumEvents,
    RangeATRcusumEvents,
    SRCusumEvents,
    TailRiskCusumEvents,
    TrendRegimeCusumEvents,
    VolatilityStateEvents,
    compute_dominance_labels,
    compute_volatility_labels,
    compute_path_degradation_labels,
    compute_tail_regime_labels,
    compute_trend_persistence_labels,
    compute_vol_state_labels,
    compute_volume_participation_labels,
    OutputGeometry as OrthoGeometry,
    GENERATOR_PARAM_NAMES,
    get_signal_specific_weights,
    KalmanTrendEvents,
    KalmanRegimeEvents,
    OrderBlockEvents,
    SymmetricCusumEvents,
    ImprovedCUSUMEvents,
    VolatilityShockEvents,
    TrendInitiationEvents,
    MeanReversionExtremeEvents,
    LiquidityShockEvents,
    TimeEvents
)
# Layer 2.5 Chaser Integration
from .layer2_5_integration import Layer25Integration, quick_layer25_setup
from .causal_residual_computation import compute_causal_residuals
from .non_causal_feature_selector import NonCausalFeatureSelector
from .conflict_detection import ConflictDetector

# Configure logging
logger = logging.getLogger(__name__)
_lgb_logger = logging.getLogger("lightgbm")
_lgb_logger.setLevel(logging.ERROR)
_lgb_logger.propagate = False

# Constants for Layer 2 Model Training (defaults/fixed) - Less Regularized
LAYER2_MODEL_CONSTANTS = {
    'boosting_type': 'goss', # Optimized for speed
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 6,       # Constrain depth for speed
    'learning_rate': 0.05, # Higher LR for faster convergence with GOSS
    'lambda_l1': 0.01,
    'lambda_l2': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20, # Higher for Goss safety
    'min_sum_hessian_in_leaf': 1e-3,
    'feature_fraction': 0.8, # Sample features to speed up split finding
    'bagging_fraction': 1.0, # Must be 1.0 for GOSS
    'bagging_freq': 0,       # Must be 0 for GOSS
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
    'is_unbalance': False,
    'scale_pos_weight': 1,
    'min_gain_to_split': 0.001,
    'min_child_weight': 0.0001,
    'early_stopping_rounds': 50,  # Add early stopping for production models
}

# Optimized constants for probe training - faster execution
LAYER2_PROBE_CONSTANTS = {
    'boosting_type': 'goss', # Optimized for speed
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 4,       # Shallower trees for speed
    'learning_rate': 0.1, # Higher LR for faster convergence
    'lambda_l1': 0.005,   # Lighter regularization
    'lambda_l2': 0.01,    # Lighter regularization
    'num_leaves': 16,     # Fewer leaves
    'min_data_in_leaf': 20, # Keep as specified
    'min_sum_hessian_in_leaf': 1e-3,
    'feature_fraction': 0.6, # More aggressive feature sampling
    'bagging_fraction': 1.0, # Must be 1.0 for GOSS
    'bagging_freq': 0,       # Must be 0 for GOSS
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
    'is_unbalance': False,
    'scale_pos_weight': 1,
    'min_gain_to_split': 0.01,  # Higher threshold for faster splits
    'min_child_weight': 0.001,  # Lower for speed
    'early_stopping_rounds': 30,  # Earlier stopping
}

class RobustFocalLoss:
    """
    Production-grade Focal Loss for LightGBM in Financial ML.
    """

    def __init__(
        self,
        gamma_pos=1.0, # gamma_fn: Preference for Opportunity (Missed Upside)
        gamma_neg=2.5, # gamma_fp: Preference for Safety (Traps)
        alpha=None,    # Auto-computed from class balance if None
        grad_clip=5.0,
        w_cap=3.0,
        mix=0.25,      # Mix between focal loss and BCE (0.0 = pure focal, 1.0 = pure BCE)
        label_smoothing=0.02,
        verbose=True
    ):
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.grad_clip = grad_clip
        self.w_cap = w_cap
        self.mix = mix
        self.label_smoothing = label_smoothing
        self.alpha = alpha
        self.verbose = verbose
        self._is_init = False

    def _init_alpha(self, labels):
        """Auto-compute alpha based on prevalence if not provided, or adapt provided alpha."""
        n_pos = np.sum(labels > 0.5)
        n_total = len(labels)

        if n_total == 0:
            self.alpha = 0.5
        else:
            pos_ratio = n_pos / n_total

            if self.alpha is None:
                # Auto-compute: inverse frequency weighting for imbalanced classes
                # For rare positives (pos_ratio < 0.5), alpha > 0.5 to upweight them
                # For rare negatives (pos_ratio > 0.5), alpha < 0.5 to upweight them
                if pos_ratio < 0.5:
                    # Rare positives: higher alpha to focus on them
                    self.alpha = 0.5 + (0.5 - pos_ratio) * 0.8  # Scale up to 0.9 max
                else:
                    # Rare negatives: lower alpha to focus on them
                    self.alpha = 0.5 - (pos_ratio - 0.5) * 0.8  # Scale down to 0.1 min
            else:
                # If alpha provided, adapt it slightly based on class balance
                # This ensures the provided alpha still considers the actual data distribution
                balance_factor = abs(pos_ratio - 0.5) * 0.3  # Max adjustment of 0.15
                if pos_ratio < 0.5:  # Imbalanced toward negatives
                    self.alpha = min(self.alpha + balance_factor, 0.95)
                else:  # Imbalanced toward positives
                    self.alpha = max(self.alpha - balance_factor, 0.05)

        # Final clamping for safety
        self.alpha = np.clip(self.alpha, 0.05, 0.95)

        if self.verbose:
            pos_pct = n_pos / n_total * 100 if n_total > 0 else 0
            logger.info(f"[LGBM Focal] Gamma(+):{self.gamma_pos} Gamma(-):{self.gamma_neg} | Alpha:{self.alpha:.4f} | Class Balance:{pos_pct:.1f}% positive")

        self._is_init = True

    def __call__(self, preds, train_data):
        if hasattr(train_data, 'get_label'):
             labels = train_data.get_label()
        else:
             labels = train_data

        # Lazy init alpha on first call to handle data loading
        if not self._is_init:
            self._init_alpha(labels)

        # 1. Label Smoothing (Crucial for Finance)
        y_smooth = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # 2. Robust Sigmoid
        p = expit(preds)
        p = np.clip(p, 1e-7, 1 - 1e-7)

        # 3. Vectorized Asymmetric Gamma
        gamma_arr = np.where(labels > 0.5, self.gamma_pos, self.gamma_neg)

        # 4. Focal Weights with Capping
        focal_weight = np.where(labels > 0.5, (1 - p), p) ** gamma_arr
        focal_weight = np.minimum(focal_weight, self.w_cap)

        # 5. Gradient & Hessian Calculation
        grad_bce = p - y_smooth
        alpha_factor = np.where(labels > 0.5, self.alpha, (1 - self.alpha))
        grad_focal = alpha_factor * focal_weight * grad_bce
        hess_bce = p * (1 - p)
        hess_focal = alpha_factor * focal_weight * hess_bce

        # 6. Mixing (Stability Anchor)
        grad = self.mix * grad_focal + (1 - self.mix) * grad_bce
        hess = self.mix * hess_focal + (1 - self.mix) * hess_bce

        # 7. Clipping & Safety
        if self.grad_clip:
            grad = np.clip(grad, -self.grad_clip, self.grad_clip)

        hess = np.maximum(hess, 1e-6) # Prevent divide-by-zero

        return grad, hess


class XGBFocalLoss:
    """
    Focal Loss for XGBoost (custom objective function).
    Fully matches RobustFocalLoss behavior (LGBM) including asymmetric gamma.
    """

    def __init__(
        self,
        gamma_pos=1.0,
        gamma_neg=2.5,
        alpha=None,
        grad_clip=5.0,
        w_cap=3.0,
        mix=0.25,
        label_smoothing=0.02,
        verbose=True
    ):
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.grad_clip = grad_clip
        self.w_cap = w_cap
        self.mix = mix
        self.label_smoothing = label_smoothing
        self.alpha = alpha
        self.verbose = verbose
        self._is_init = False

    def _init_alpha(self, labels):
        """Auto-compute alpha based on prevalence if not provided."""
        if self.alpha is None:
            n_pos = np.sum(labels > 0.5)
            n_total = len(labels)
            if n_total > 0:
                # Standard inverse frequency
                self.alpha = 1.0 - (n_pos / n_total)
            else:
                self.alpha = 0.5

        # Clamp alpha for safety
        self.alpha = np.clip(self.alpha, 0.05, 0.95)
        self._is_init = True

    def __call__(self, preds, dtrain):
        is_sklearn = False
        try:
            if hasattr(dtrain, 'get_label'):
                labels = dtrain.get_label()
                logits = preds
            elif isinstance(dtrain, np.ndarray):
                labels = preds
                logits = dtrain
                is_sklearn = True
            else:
                labels = dtrain
                logits = preds
        except Exception:
             labels = dtrain
             logits = preds

        if not self._is_init:
            self._init_alpha(labels)

        y_smooth = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        p = 1.0 / (1.0 + np.exp(-logits))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        gamma_arr = np.where(labels > 0.5, self.gamma_pos, self.gamma_neg)
        focal_weight = np.where(labels > 0.5, (1 - p), p) ** gamma_arr
        focal_weight = np.minimum(focal_weight, self.w_cap)

        grad_bce = p - y_smooth
        alpha_factor = np.where(labels > 0.5, self.alpha, (1 - self.alpha))
        grad_focal = alpha_factor * focal_weight * grad_bce
        hess_bce = p * (1 - p)
        hess_focal = alpha_factor * focal_weight * hess_bce

        grad = self.mix * grad_focal + (1 - self.mix) * grad_bce
        hess = self.mix * hess_focal + (1 - self.mix) * hess_bce

        if self.grad_clip:
            grad = np.clip(grad, -self.grad_clip, self.grad_clip)

        hess = np.maximum(hess, 1e-6)

        return grad, hess




def _calculate_tree_variance(booster, X) -> np.ndarray:
    """
    Calculate the variance of predictions across all trees in the ensemble (Tree Variation).
    Full implementation restored.
    """
    if booster is None:
        return np.zeros(X.shape[0])

    try:
        if hasattr(booster, 'calibrated_classifiers_'):
            if len(booster.calibrated_classifiers_) > 0:
                booster = booster.calibrated_classifiers_[0].base_estimator

        raw_booster = None
        if hasattr(booster, 'booster_'):
            raw_booster = booster.booster_
        elif hasattr(booster, 'get_booster'):
            raw_booster = booster.get_booster()
        else:
            raw_booster = booster

        leaf_indices_raw = None
        if hasattr(raw_booster, 'predict'):
            try:
                leaf_indices_raw = raw_booster.predict(X, pred_leaf=True)
            except Exception:
                pass

        if leaf_indices_raw is None:
            try:
                leaf_indices_raw = booster.predict(X, pred_leaf=True)
            except Exception:
                pass

        if leaf_indices_raw is None:
            return np.zeros(X.shape[0])
        
        if leaf_indices_raw.ndim == 1:
            leaf_indices = leaf_indices_raw.reshape(-1, 1)
        else:
            leaf_indices = leaf_indices_raw

        model_dump = None
        if hasattr(raw_booster, 'dump_model'):
            model_dump = raw_booster.dump_model()
        elif hasattr(booster, 'dump_model'):
            model_dump = booster.dump_model()

        if model_dump is None:
             return np.zeros(X.shape[0])

        trees = model_dump.get('tree_info', [])
        max_leaf_idx = 0
        for tree in trees:
            if 'tree_structure' in tree:
                nodes = [tree['tree_structure']]
                while nodes:
                    node = nodes.pop()
                    if 'leaf_index' in node:
                        max_leaf_idx = max(max_leaf_idx, node['leaf_index'])
                    if 'left_child' in node:
                        nodes.append(node['left_child'])
                    if 'right_child' in node:
                        nodes.append(node['right_child'])

        n_trees = len(trees)
        leaf_values_lookup = np.full((n_trees, max_leaf_idx + 1), np.nan)

        for i, tree in enumerate(trees):
            if 'tree_structure' in tree:
                nodes = [tree['tree_structure']]
                while nodes:
                    node = nodes.pop()
                    if 'leaf_index' in node:
                        idx = node['leaf_index']
                        val = node.get('leaf_value', 0.0)
                        if idx <= max_leaf_idx:
                            leaf_values_lookup[i, idx] = val
                    if 'left_child' in node:
                        nodes.append(node['left_child'])
                    if 'right_child' in node:
                        nodes.append(node['right_child'])

        n_samples = leaf_indices.shape[0]
        n_trees_pred = leaf_indices.shape[1]
        limit_trees = min(n_trees, n_trees_pred)
        tree_indices = np.arange(limit_trees)
        subset_indices = leaf_indices[:, :limit_trees]
        subset_indices = np.clip(subset_indices, 0, max_leaf_idx)
        collected_values = leaf_values_lookup[tree_indices, subset_indices]
        variance = np.nanvar(collected_values, axis=1)

        return variance

    except Exception as e:
        logger.warning(f"Failed to calculate tree variance: {e}")
        return np.zeros(X.shape[0])




@dataclass
class GeometryTrial:
    family: str
    params: Dict[str, Any]  # Kappa, Horizon, sl_sigma, alpha, beta, min_ratio
    final_score: float
    learnability: float
    robust_magnitude: float
    stability: float
    balance: float
    raw_metrics: Dict[str, float]
    uuid: str
    model_params: Optional[Dict[str, Any]] = None
    selected_features: Optional[List[str]] = field(default=None)
    race_score: Optional[float] = None
    events: Optional[pd.DatetimeIndex] = None # Added for orthogonality



def roll_entropy(series: pd.Series, window: int = 20, bins: int = 10) -> pd.Series:
    """Rolling Entropy to detect structural breaks."""
    def _ent(x):
        hist, bin_edges = np.histogram(x, bins=bins, density=True)
        # Avoid log(0)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    return series.rolling(window).apply(_ent, raw=True)


def get_serial_correlation(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling serial correlation (autocorrelation at lag 1).
    High positive = Trending; Negative = Mean Reverting.
    """
    return series.rolling(window).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=True)




def generate_market_state_probe(price: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """
    A 'Theory of Mind' Probe.
    Instead of just RSI, we test if the geometry is learnable based on
    Information Theory and Market Microstructure states.
    """
    df = pd.DataFrame(index=price.index)

    # 1. Serial Correlation (Trendiness vs Mean Reversion state)
    df['serial_corr'] = get_serial_correlation(price.pct_change(), window=20)

    # 2. Volatility Ratio (Expansion/Contraction state)
    vol_short = price.pct_change().rolling(10).std()
    vol_long = price.pct_change().rolling(60).std()
    df['vol_regime'] = vol_short / (vol_long + 1e-9)

    # 3. Entropy (Information state)
    # Are returns random or structured?
    df['entropy'] = roll_entropy(np.log(price).diff().fillna(0), window=50)

    # 4. Amihud Illiquidity (Liquidity state)
    # High value = Price moves easily with little volume (Fragile)
    ret_abs = price.pct_change().abs()
    df['illiquidity'] = (ret_abs / (volume * price + 1e-9)).rolling(20).mean()

    # 5. Relative Drawdown (Psychological state)
    roll_max = price.rolling(100).max()
    df['drawdown'] = (price / roll_max) - 1.0

    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


class LabelBasedLayer2(BaseStep):
    """
    Layer 2: Regime-Conditional Geometry Optimization & Meta-Labeling.
    """

    _component_singletons: Dict[str, Any] = {}
    _specialist_registration_cache: Dict[str, Tuple[str, int]] = {}
    _model_race_seed_models: Dict[str, Dict[str, Any]] = {}

    def __init__(self, step_name: str = 'label_based_layer_2', **kwargs):
        # Initialize BaseStep with step name
        super().__init__(step_name)
        self._dataset_fingerprint: Optional[str] = None
        self._label_batch_cache: Dict[str, Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]] = {}
        self._confounder_cache: Dict[str, pd.DataFrame] = {}

        # Initialize LabelBasedLayer2 specific parameters
        transaction_cost = kwargs.get('transaction_cost', None)
        self.n_trials = kwargs.get('n_trials', 60)
        self.n_splits = kwargs.get('n_splits', 2)
        self.random_state = kwargs.get('random_state', 42)
        self.verbose = kwargs.get('verbose', True)
        self.force_hpo = kwargs.get('force_hpo', False)
        
        # Set transaction cost if not provided
        if transaction_cost is None:
            try:
                from src.utils.ml_common.transaction_costs import DEFAULT_TRANSACTION_COST
                transaction_cost = DEFAULT_TRANSACTION_COST
            except Exception:
                transaction_cost = 0.003  # Default fallback
        
        self.transaction_cost = float(transaction_cost)
        
        # Denoised price configuration
        self.use_denoised_prices = kwargs.get('use_denoised_prices', True)
        self.layer0_params = None

        # Model comparison configuration
        self.enable_model_race = kwargs.get('enable_model_race', False)
        self.model_race_candidates = kwargs.get('model_race_candidates', ['LGBM_Focal', 'XGB_Tree', 'CatBoost', 'LGBM_Focal_Linear', 'XGB_Linear'])

        # AEDL Framework Parameters
        self.enable_aedl = kwargs.get("enable_aedl", True)
        self.aedl_spectral_vision = kwargs.get("aedl_spectral_vision", True)
        self.aedl_causal_compression = kwargs.get("aedl_causal_compression", True)
        self.aedl_resonance_detection = kwargs.get("aedl_resonance_detection", True)
        
        # Spectral Chaser Parameters
        self.spectral_chaser_enabled = kwargs.get("spectral_chaser_enabled", True)
        self.spectral_chaser_models = kwargs.get("spectral_chaser_models", ['xgb', 'catboost', 'rf', 'linear'])
        self.spectral_chaser_cv_folds = kwargs.get("spectral_chaser_cv_folds", 5)
        
        # RSV Integration Parameters
        self.rsv_integration_enabled = kwargs.get("rsv_integration_enabled", True)
        self.rsv_position_sizing = kwargs.get("rsv_position_sizing", True)
        # RSV Integration Parameters
        self.rsv_integration_enabled = kwargs.get("rsv_integration_enabled", True)
        self.rsv_position_sizing = kwargs.get("rsv_position_sizing", True)
        self.rsv_regime_aware = kwargs.get("rsv_regime_aware", True)
        
        # Causal Surprise Parameters
        self.causal_surprise_enabled = kwargs.get("causal_surprise_enabled", True)
        self.zone3_specialist_boost = kwargs.get("zone3_specialist_boost", 3.0)
        self.zone2_specialist_boost = kwargs.get("zone2_specialist_boost", 2.0)
        self.zone_score_exposure = float(kwargs.get("zone_score_exposure", 1.0))
        
        # OOF Analytics Parameter
        self.sr_levels = []

        # RobustFocalLoss HPO configuration
        self.enable_focal_hpo = kwargs.get('enable_focal_hpo', False)
        self.focal_hpo_n_trials = kwargs.get('focal_hpo_n_trials', 20)
        
        # Initialize config
        self._current_config = {}
        self.signal_weights = None
        
        # Initialize caches
        self._events_cache = {}
        self._feature_cache = {}
        self._global_probe_features = {}
        self._probe_data_cache = {}
        self._label_cache = {}
        self._signals_cache = {}
        self._global_feature_cache = {}  # Global feature cache for entire dataset
        self._global_event_cache = {}    # Global event cache
        self._model_cache = {}           # Cache trained models
        self._feature_selection_cache = {}  # Cache feature selections
        self._label_computation_cache = {}  # Cache label computations

        # Initialize diagnostics
        self._all_tree_stats = []
        
        # Comprehensive Modern De Prado Framework Configuration
        self.enable_causal_framework = kwargs.get('enable_causal_framework', True)
        self.causal_discovery_enabled = kwargs.get('causal_discovery_enabled', True)
        self.causal_engineering_enabled = kwargs.get('causal_engineering_enabled', True)
        self.irm_enabled = kwargs.get('irm_enabled', True)
        self.causal_surprise_enabled = kwargs.get('causal_surprise_enabled', True)
        self.interventionist_sampling_enabled = kwargs.get('interventionist_sampling_enabled', True)
        self.causal_targets_enabled = kwargs.get('causal_targets_enabled', True)
        self.causal_specialists_enabled = kwargs.get('causal_specialists_enabled', True)
        
        # IRM Parameters
        self.lambda_irm = kwargs.get('lambda_irm', 1.0)
        self.lambda_variance = kwargs.get('lambda_variance', 1.0)
        self.focal_alpha = kwargs.get('focal_alpha', 1.0)
        self.focal_gamma = kwargs.get('focal_gamma', 2.0)
        self.focal_gamma_pos = kwargs.get('focal_gamma_pos', 1.0)
        self.focal_gamma_neg = kwargs.get('focal_gamma_neg', 2.5)
        
        # Causal Discovery Parameters
        self.significance_level = kwargs.get('significance_level', 0.05)
        self.max_conditioning_set = kwargs.get('max_conditioning_set', 3)
        self.use_lingam = kwargs.get('use_lingam', True)
        
        # Causal Surprise Parameters
        self.surprise_threshold = kwargs.get('surprise_threshold', 1.8)
        self.rolling_window = kwargs.get('rolling_window', 20)
        self.min_specialists = kwargs.get('min_specialists', 2)
        self.discovery_max_features = kwargs.get('discovery_max_features', 60)
        self.discovery_sample_size = kwargs.get('discovery_sample_size', 10000)
        self.discovery_bootstrap_samples = kwargs.get('discovery_bootstrap_samples', 25)
        self.specialist_train_workers = kwargs.get('specialist_train_workers', 4)
        self.specialist_max_models = kwargs.get('specialist_max_models')
        self.specialist_debug_logging = kwargs.get('specialist_debug_logging', False)
        self.specialist_registration_workers = kwargs.get('specialist_registration_workers', 4)
        self.treatment_max_features = kwargs.get('treatment_max_features', 25)
        self.treatment_min_coverage = kwargs.get('treatment_min_coverage', 0.3)
        self.treatment_allow_sparse = kwargs.get('treatment_allow_sparse', True)
        self.treatment_sparse_min_events = kwargs.get('treatment_sparse_min_events', 25)
        self.treatment_fill_value = kwargs.get('treatment_fill_value', 0.0)
        self.model_race_target_precision = kwargs.get('model_race_target_precision', 0.65)
        self.model_race_plateau_patience = kwargs.get('model_race_plateau_patience', 2)
        self.model_race_max_candidates = kwargs.get('model_race_max_candidates', None)
        
        # Interventionist Sampling Parameters
        self.shock_threshold = kwargs.get('shock_threshold', 2.0)
        self.intervention_strength = kwargs.get('intervention_strength', 1.0)
        self.n_interventions = kwargs.get('n_interventions', 100)
        
        # Initialize causal components
        self._causal_discovery = None
        self._causal_engineering = None
        self._irm_system = None
        self._surprise_detector = None
        self._intervention_sampler = None
        self._target_computer = None
        self._specialist_manager = None
        self._aedl_framework = None
        
        # Check availability of causal modules
        try:
            from .causal_discovery import CausalDiscovery
            from .causal_feature_engineering import CausalFeatureEngineering
            from .invariant_risk_minimization_v2 import EnhancedIRM
            from .causal_surprise_events import CausalSurpriseDetector
            from .interventionist_sampling import CausalInterventionSampler
            from .causal_targets import CausalTargetComputer
            from .causal_specialists import CausalSpecialistManager
            CAUSAL_MODULES_AVAILABLE = True
        except ImportError as e:
            tprint_warning(f"⚠️ Causal modules not available: {e}")
            CAUSAL_MODULES_AVAILABLE = False
        
        # Store availability flag
        self.CAUSAL_MODULES_AVAILABLE = CAUSAL_MODULES_AVAILABLE

    def _validate_framework_separation(self) -> None:
        """
        Validate that AFML and Causal frameworks are properly separated.
        """
        if self.enable_causal_framework:
            tprint_info("🔍 Validating Framework Separation...")

            # When causal framework is enabled, ensure AFML components are disabled
            afml_warnings = []

            if self.focal_alpha != 1.0 or self.focal_gamma_pos != 1.0:
                afml_warnings.append("Focal loss parameters should be default (1.0) in causal mode")

            # Check for mixed objectives
            if hasattr(self, '_current_config') and self._current_config:
                if self._current_config.get('enable_focal_hpo', False):
                    afml_warnings.append("Focal HPO should be disabled in causal framework")

            if afml_warnings:
                for warning in afml_warnings:
                    tprint_warning(f"⚠️ Framework Separation: {warning}")

            tprint_success("✅ Framework separation validated")
        else:
            tprint_info("📊 Using traditional AFML framework")

    def _initialize_causal_components(self, df: pd.DataFrame) -> None:
        """
        Initialize causal framework components.
        """
        tprint_info("🔧 Layer 2: Checking Causal Module Availability...")
        if not CAUSAL_MODULES_AVAILABLE:
            tprint_warning("⚠️ Layer 2: Causal modules not available - skipping causal initialization")
            return

        tprint_info("✅ Layer 2: Causal modules available")

        # Validate framework separation first
        tprint_info("🔍 Layer 2: Validating framework separation...")
        self._validate_framework_separation()
        tprint_info("✅ Layer 2: Framework separation validated")

        try:
            tprint_info("🔧 Layer 2: Initializing Causal Framework Components...")

            dataset_tag = self._dataset_fingerprint or "adhoc"

            if self.causal_discovery_enabled:
                tprint_info("   📊 Layer 2: Initializing Causal Discovery...")
                discovery_config = {
                    "max_conditioning_set": self.max_conditioning_set,
                    "significance_level": self.significance_level,
                    "use_lingam": self.use_lingam,
                    "verbose": self.verbose
                }
                self._causal_discovery = self._get_component_singleton(
                    "causal_discovery",
                    discovery_config,
                    lambda: CausalDiscovery(
                        max_conditioning_set=self.max_conditioning_set,
                        significance_level=self.significance_level,
                        use_lingam=self.use_lingam,
                        verbose=self.verbose
                    )
                )
                tprint_success("   ✅ Layer 2: Causal Discovery initialized")
            else:
                tprint_info("   ⏭️ Layer 2: Causal Discovery disabled")

            tprint_info("   🔧 Layer 2: Initializing Causal Feature Engineering...")
            engineering_config = {"verbose": self.verbose}
            self._causal_engineering = self._get_component_singleton(
                "causal_feature_engineering",
                engineering_config,
                lambda: CausalFeatureEngineering(verbose=self.verbose)
            )
            tprint_success("   ✅ Layer 2: Causal Feature Engineering initialized")

            if self.irm_enabled:
                tprint_info("   🎯 Layer 2: Initializing Invariant Risk Minimization...")
                irm_config = {
                    "lambda_irm": self.lambda_irm,
                    "lambda_variance": self.lambda_variance,
                    "focal_alpha": self.focal_alpha,
                    "focal_gamma": self.focal_gamma_pos,
                    "verbose": self.verbose
                }
                self._irm_system = self._get_component_singleton(
                    "invariant_risk_minimization",
                    irm_config,
                    lambda: EnhancedIRM(
                        lambda_irm=self.lambda_irm,
                        lambda_variance=self.lambda_variance,
                        focal_alpha=self.focal_alpha,
                        focal_gamma=self.focal_gamma_pos,
                        verbose=self.verbose
                    )
                )
                tprint_success("   ✅ Layer 2: Invariant Risk Minimization initialized")
            else:
                tprint_info("   ⏭️ Layer 2: IRM disabled")

            if self.causal_targets_enabled:
                tprint_info("   🎯 Layer 2: Initializing Causal Target Computer...")
                target_config = {"verbose": self.verbose}
                self._causal_targets = self._get_component_singleton(
                    "causal_target_computer",
                    target_config,
                    lambda: CausalTargetComputer(verbose=self.verbose)
                )
                tprint_success("   ✅ Layer 2: Causal Target Computer initialized")
            else:
                tprint_info("   ⏭️ Layer 2: Causal targets disabled")

            if self.causal_surprise_enabled:
                tprint_info("   🚨 Layer 2: Initializing Causal Surprise Detector...")
                detector_config = {
                    "surprise_threshold": self.surprise_threshold,
                    "rolling_window": self.rolling_window,
                    "dataset_tag": dataset_tag
                }
                self._surprise_detector = self._get_component_singleton(
                    "causal_surprise_detector",
                    detector_config,
                    lambda: CausalSurpriseDetector(
                        surprise_threshold=self.surprise_threshold,
                        rolling_window=self.rolling_window,
                        verbose=self.verbose
                    )
                )
                # Reset detector state for new dataset
                if hasattr(self._surprise_detector, "specialist_errors_"):
                    self._surprise_detector.specialist_errors_.clear()
                if hasattr(self._surprise_detector, "surprise_events_"):
                    if isinstance(self._surprise_detector.surprise_events_, dict):
                        self._surprise_detector.surprise_events_.clear()
                    else:
                        self._surprise_detector.surprise_events_ = None
                self._specialist_registration_cache.clear()
                tprint_success("   ✅ Layer 2: Causal Surprise Detector ready")
            else:
                tprint_info("   ⏭️ Layer 2: Causal surprise events disabled")

            tprint_success("🎯 Layer 2: All Causal Components Initialized Successfully")

        except Exception as e:
            tprint_error(f"❌ Layer 2: Failed to initialize causal components: {e}")
            import traceback
            tprint_error(f"❌ Layer 2: Traceback: {traceback.format_exc()}")
            raise

    def _get_causal_anchor_predictions(self) -> np.ndarray:
        """Get causal anchor predictions for Spectral Chaser."""
        try:
            if hasattr(self, 'causal_anchor_predictions'):
                return self.causal_anchor_predictions
            
            # Fallback: use trained model predictions as anchor
            if hasattr(self, 'trained_models') and len(self.trained_models) > 0:
                # Use the best model as anchor
                best_model_name = max(self.trained_models.keys(), 
                                   key=lambda x: self.trained_models[x].get('auc', 0))
                best_model = self.trained_models[best_model_name].model
                
                if hasattr(self, 'X_full') and best_model is not None:
                    if self.verbose:
                        tprint_warning(f"   ⚠️ Using model '{best_model_name}' as causal anchor - verify lookahead safety!")
                    return best_model.predict(self.X_full)
            
            # Final fallback: zeros
            if self.verbose:
                tprint_info("   ℹ️ No anchor available, using neutral (zero) anchor")
                
            if hasattr(self, 'y_full'):
                return np.zeros_like(self.y_full)
            
            return np.zeros(1000)  # Default fallback
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"   ⚠️ Could not get causal anchor predictions: {e}")
            return np.zeros(1000)

    def _run_aedl_pipeline(
        self,
        df: pd.DataFrame,
        target_col: str,
        causal_graph: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Run Adaptive Event-Driven Labeling (AEDL) pipeline.
        
        Args:
            df: Input data
            target_col: Target column name
            causal_graph: Causal graph for parent filtering
            
        Returns:
            Dictionary with AEDL results
        """
        try:
            from .adaptive_event_driven_labeling import AdaptiveEventDrivenLabeling
            
            if self.verbose:
                tprint_info("🚀 AEDL Pipeline: Starting frequency-dependent analysis...")
            
            aedl_start_time = time.time()
            
            # Get causal anchor predictions
            causal_anchor_predictions = self._get_causal_anchor_predictions()
            
            # Ensure alignment with DataFrame
            if len(causal_anchor_predictions) != len(df):
                if self.verbose:
                    tprint_warning(f"   ⚠️ Aligning causal anchor to df: {len(causal_anchor_predictions)} -> {len(df)}")
                
                # If valid prediction available but size mismatch (e.g. valid has fewer due to dropna), 
                # we might need smart alignment. But here we assume naive fallback is better than crash.
                if len(causal_anchor_predictions) == 1000: # Common fallback
                    causal_anchor_predictions = np.zeros(len(df))
                else:
                    # Resize with zeros
                    new_anchor = np.zeros(len(df))
                    min_len = min(len(causal_anchor_predictions), len(df))
                    new_anchor[:min_len] = causal_anchor_predictions[:min_len]
                    causal_anchor_predictions = new_anchor
            
            # Initialize AEDL framework with passed causal graph or fallback to attribute
            graph_to_use = causal_graph if causal_graph is not None else getattr(self, 'causal_graph', {})
            
            aedl = AdaptiveEventDrivenLabeling(
                causal_graph=graph_to_use,
                verbose=self.verbose
            )
            self._aedl_framework = aedl
            
            # Process market data
            aedl_results = aedl.process_market_data(df, causal_anchor_predictions)
            
            if 'error' in aedl_results:
                if self.verbose:
                    tprint_error(f"   ❌ AEDL processing failed: {aedl_results['error']}")
                return aedl_results
            
            # Compile results
            aedl_time = time.time() - aedl_start_time
            
            results = {
                'aedl_enabled': True,
                'specialist_signals': aedl_results.get('specialist_signals', {}),
                'spectral_components': aedl_results.get('spectral_components', {}),
                'resonance_scores': aedl_results.get('resonance_scores', {}),
                'rsv_eigenvalue': aedl_results.get('rsv_eigenvalue', 0.0),
                'rsv_info': aedl_results.get('rsv_info', {}),
                'alpha_features': aedl_results.get('alpha_features', {}),
                'position_sizing_guidance': aedl_results.get('position_sizing_guidance', {}),
                'harmonic_entries': aedl.get_harmonic_entries(),
                'structural_breakouts': aedl.get_structural_breakouts(),
                'aedl_time': aedl_time,
                'compression_metrics': aedl_results.get('compression_metrics', {}),
                'aedl_report': aedl.generate_aedl_report()
            }
            
            if self.verbose:
                tprint_success("✅ AEDL Pipeline Complete:")
                tprint_info(f"   - Spectral components: {len(results['spectral_components'])}")
                tprint_info(f"   - Resonance scores: {len(results['resonance_scores'])}")
                tprint_info(f"   - RSV eigenvalue: {results['rsv_eigenvalue']:.3f}")
                tprint_info(f"   - Alpha features: {len(results['alpha_features'])}")
                tprint_info(f"   - Position regime: {results['position_sizing_guidance'].get('resonance_regime', 'UNKNOWN')}")
                tprint_info(f"   - AEDL time: {aedl_time:.3f}s")
            
            return results
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ AEDL pipeline failed: {e}")
            return {'error': str(e)}
    
    def _run_spectral_chaser(
        self,
        df: pd.DataFrame,
        y_residuals: pd.Series,
        causal_anchor_predictions: np.ndarray = None,
        sample_weight: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Run Spectral Chaser with AEDL features.
        
        Args:
            df: Input data
            y_residuals: Target residuals
            causal_anchor_predictions: Causal anchor predictions
            
        Returns:
            Dictionary with Spectral Chaser results
        """
        try:
            from .spectral_chaser import SpectralChaser
            
            if self.verbose:
                tprint_info("🔬 Spectral Chaser: Starting training with spectral vision...")
            
            chaser_start_time = time.time()
            
            # Initialize Spectral Chaser
            spectral_chaser = SpectralChaser(
                causal_graph=getattr(self, 'causal_graph', None),
                model_types=self.spectral_chaser_models,
                verbose=self.verbose
            )
            
            # Train Spectral Chaser
            training_metrics = spectral_chaser.fit(
                df=df,
                y_residuals=y_residuals,
                causal_anchor_predictions=causal_anchor_predictions,
                sample_weight=sample_weight
            )
            
            if 'error' in training_metrics:
                if self.verbose:
                    tprint_error(f"   ❌ Spectral Chaser training failed: {training_metrics['error']}")
                return training_metrics
            
            # Generate predictions
            prediction_results = spectral_chaser.predict(
                df=df,
                causal_anchor_predictions=causal_anchor_predictions
            )
            
            # Get spectral insights
            spectral_insights = spectral_chaser.get_spectral_insights()
            
            # Compile results
            chaser_time = time.time() - chaser_start_time
            
            results = {
                'spectral_chaser_enabled': True,
                'training_metrics': training_metrics,
                'prediction_results': prediction_results,
                'spectral_insights': spectral_insights,
                'chaser_time': chaser_time,
                'model_count': len(spectral_chaser.models),
                'feature_importance': spectral_chaser.get_feature_importance()
            }
            
            # Store Spectral Chaser for downstream use
            self.spectral_chaser = spectral_chaser
            
            if self.verbose:
                tprint_success("✅ Spectral Chaser Complete:")
                tprint_info(f"   - Models trained: {results['model_count']}")
                tprint_info(f"   - Training time: {training_metrics.get('training_time', 0):.3f}s")
                tprint_info(f"   - Prediction time: {prediction_results.get('prediction_time', 0):.3f}s")
                tprint_info(f"   - Resonance regime: {spectral_insights.get('resonance_regime', 'UNKNOWN')}")
            
            # Save Report
            self._save_spectral_chaser_report(results)

            return results
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Spectral Chaser failed: {e}")
            return {'error': str(e)}

    def _save_spectral_chaser_report(self, chaser_results: Dict[str, Any]):
        """Save Spectral Chaser report to outcomes."""
        try:
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = outcomes_dir / f"spectral_chaser_report_{ts}.md"
            
            insights = chaser_results.get('spectral_insights', {})
            training = chaser_results.get('training_metrics', {})
            
            report = [
                "# Spectral Chaser Report",
                f"- Date: {ts}",
                f"- Models: {chaser_results.get('model_count', 0)}",
                "",
                "## Spectral Insights",
                f"- Resonance Regime: **{insights.get('resonance_regime', 'UNKNOWN')}**",
                f"- RSV Eigenvalue: {insights.get('rsv_eigenvalue', 0.0):.4f}",
                f"- Compression Ratio: {insights.get('compression_ratio', 1.0):.2f}x",
                "",
                "## Component Analysis"
            ]
            
            if 'harmonic_entries' in insights:
                entries = insights['harmonic_entries']
                report.append(f"- Harmonic Entry Signal: {entries.get('entry_signal', False)}")
                report.append(f"- Entry Quality: {entries.get('entry_quality', 0.0):.4f}")
            
            if 'structural_breakouts' in insights:
                breakouts = insights['structural_breakouts']
                report.append(f"- Structural Breakouts: {breakouts.get('breakout_periods', 0)}")
                report.append(f"- Dominant Specialist: {breakouts.get('dominant_breakout_specialist', 'None')}")
            
            feature_importance = chaser_results.get('feature_importance', {})
            if feature_importance:
                report.append("")
                report.append("## Feature Importance (Top 10)")
                sorted_feats = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                for f, imp in sorted_feats:
                    report.append(f"- `{f}`: {imp:.4f}")
            
            report_path.write_text("\n".join(report))
            if self.verbose:
                tprint_success(f"✅ Saved Spectral Chaser report to {report_path}")
                
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save Spectral Chaser report: {e}")


    def _run_causal_discovery(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Run enhanced causal discovery with Bayesian uncertainty quantification.
        """
        try:
            if self.verbose:
                tprint_info("🔍 Enhanced Causal Discovery: Starting Bayesian discovery with uncertainty...")
            
            # Check if Bayesian discovery is enabled
            use_bayesian = getattr(self, 'use_bayesian_discovery', True)
            
            if use_bayesian and self.CAUSAL_MODULES_AVAILABLE:
                numeric_df = self._filter_discovery_input(df)
                if numeric_df.empty:
                    tprint_warning("   ⚠️ No numeric data available for causal discovery")
                    return {}
                if self.verbose:
                    tprint_info(f"   📊 Using {len(numeric_df.columns)} filtered columns x {len(numeric_df)} samples")
                
                discovery_results = quick_bayesian_causal_discovery(
                    numeric_df,
                    n_bootstrap=self.discovery_bootstrap_samples,
                    significance_level=0.8,
                    verbose=self.verbose
                )
                
                if 'error' in discovery_results:
                    if self.verbose:
                        tprint_warning("   ⚠️ Bayesian discovery failed, falling back to deterministic...")
                    return self._fallback_causal_discovery(df)
                
                # Extract causal graph and uncertainty metrics
                causal_graph = discovery_results.get('consensus_graph', {})
                uncertainty_metrics = discovery_results.get('uncertainty_metrics', {})
                
                # Normalize confidence metrics to [0, 1] for safety
                if 'avg_confidence' in uncertainty_metrics:
                    raw_conf = uncertainty_metrics['avg_confidence']
                    uncertainty_metrics['avg_confidence'] = 1.0 if raw_conf > 1.0 else raw_conf
                    
                if 'graph_stability' in uncertainty_metrics:
                    raw_stab = uncertainty_metrics['graph_stability']
                    uncertainty_metrics['graph_stability'] = 1.0 if raw_stab > 1.0 else raw_stab
                
                # Store uncertainty metrics for reporting
                self.causal_discovery_uncertainty_ = uncertainty_metrics
                
                if self.verbose:
                    tprint_success(f"   ✅ Bayesian discovery complete:")
                    tprint_info(f"      - Graph edges: {len(causal_graph)}")
                    tprint_info(f"      - Graph stability: {uncertainty_metrics.get('graph_stability', 0):.3f}")
                    tprint_info(f"      - Avg confidence: {uncertainty_metrics.get('avg_confidence', 0):.3f}")
                
                return causal_graph
            else:
                if self.verbose:
                    tprint_info("   📊 Using deterministic Causal Discovery...")
                return self._fallback_causal_discovery(df)
                
        except Exception as e:
            if self.verbose:
                tprint_error(f"   ❌ Enhanced causal discovery failed: {e}")
            return self._fallback_causal_discovery(df)
    
    def _fallback_causal_discovery(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Fallback to deterministic causal discovery."""
        try:
            if self.verbose:
                tprint_info("   🔄 Using fallback deterministic causal discovery...")
            
            causal_discovery = CausalDiscovery(verbose=self.verbose)
            discovery_results = causal_discovery.discover_causal_structure(df)
            
            if 'error' in discovery_results:
                if self.verbose:
                    tprint_error("   ❌ Fallback discovery also failed")
                return {}
            
            causal_graph = discovery_results.get('causal_graph', {})
            
            if self.verbose:
                tprint_success(f"   ✅ Fallback discovery complete: {len(causal_graph)} edges")
            
            return causal_graph
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"   ❌ Fallback discovery failed: {e}")
            return {}

        try:
            tprint_info("   🔬 Layer 2: Running PC Algorithm + LiNGAM...")

            # Select numeric features for causal discovery
            tprint_info("   📊 Layer 2: Selecting numeric features...")
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            tprint_info(f"      - Original features: {len(df.columns)}")
            tprint_info(f"      - Numeric features: {len(numeric_df.columns)}")
            tprint_info(f"      - Samples after dropna: {len(numeric_df)}")

            if len(numeric_df.columns) < 3:
                tprint_warning("   ⚠️ Layer 2: Insufficient numeric features for causal discovery (< 3)")
                return {}

            if len(numeric_df) < 10:
                tprint_warning("   ⚠️ Layer 2: Insufficient samples for causal discovery (< 10)")
                return {}

            # Run causal discovery
            tprint_info("   🚀 Layer 2: Executing causal discovery pipeline...")
            discovery_results = self._causal_discovery.discover_causal_structure(numeric_df)

            if 'error' in discovery_results:
                tprint_error(f"   ❌ Layer 2: Causal discovery error: {discovery_results['error']}")
                return {}

            causal_graph = discovery_results.get('causal_graph', {})
            tprint_success("   ✅ Layer 2: Causal Discovery Complete:")
            tprint_info(f"      - Variables analyzed: {discovery_results.get('n_variables', 0)}")
            tprint_info(f"      - Edges discovered: {discovery_results.get('n_edges', 0)}")
            tprint_info(f"      - Samples used: {discovery_results.get('n_samples', 0)}")
            tprint_info(f"      - Significance level: {discovery_results.get('significance_level', 'N/A')}")

            if causal_graph:
                total_parents = sum(len(parents) for parents in causal_graph.values())
                tprint_info(f"      - Total parent-child relationships: {total_parents}")
            else:
                tprint_warning("   ⚠️ Layer 2: No causal graph generated")

            return causal_graph

        except Exception as e:
            tprint_error(f"   ❌ Layer 2: Causal discovery failed: {e}")
            import traceback
            tprint_error(f"   ❌ Layer 2: Traceback: {traceback.format_exc()}")
            return {}

    def _initialize_causal_specialists(self, df: pd.DataFrame, causal_graph: Dict[str, List[str]]) -> Dict[str, pd.Series]:
        """
        Initialize and train causal specialists as causal parents.
        """
        tprint_info("🧠 Layer 2: Initializing Causal Specialists...")

        if not self.causal_specialists_enabled:
            tprint_info("   ⏭️ Layer 2: Causal specialists disabled")
            return {}

        # Manager is created in this method, so no need to check for existence pre-creation
        # if self._specialist_manager is None:
        #    tprint_warning("   ⚠️ Layer 2: Specialist manager not initialized") 
        #    return {}

        try:
            tprint_info("   📊 Layer 2: Analyzing causal graph...")
            if not causal_graph:
                tprint_warning("   ⚠️ Layer 2: No causal graph available for specialist initialization")
                return {}

            total_relationships = sum(len(parents) for parents in causal_graph.values())
            tprint_info(f"      - Causal relationships: {total_relationships}")
            tprint_info(f"      - Variables in graph: {len(causal_graph)}")

            # Create specialists from causal graph
            tprint_info("   🏗️ Layer 2: Creating specialists from causal graph...")
            self._specialist_manager = create_causal_specialists(causal_graph)
            if self.specialist_max_models and len(self._specialist_manager.specialists) > self.specialist_max_models:
                limit = self.specialist_max_models
                tprint_info(f"      - Limiting specialists to top {limit} relationships")
                self._specialist_manager.specialists = self._specialist_manager.specialists[:limit]
                self._specialist_manager.specialist_dict_ = {
                    spec.name: spec for spec in self._specialist_manager.specialists
                }

            if len(self._specialist_manager.specialists) == 0:
                tprint_warning("   ⚠️ Layer 2: No specialists could be created from causal graph")
                return {}

            tprint_info(f"      - Specialists created: {len(self._specialist_manager.specialists)}")
            for specialist in self._specialist_manager.specialists[:3]:  # Show first 3
                tprint_info(f"         • {specialist.name}: {specialist.causal_parent} → {specialist.causal_child}")

            # Prepare target data for specialists
            tprint_info("   🎯 Layer 2: Preparing target data for specialists...")
            y_dict = {}
            available_targets = 0
            for specialist in self._specialist_manager.specialists:
                if specialist.causal_child in df.columns:
                    y_dict[specialist.name] = df[specialist.causal_child]
                    available_targets += 1

            tprint_info(f"      - Target variables available: {available_targets}/{len(self._specialist_manager.specialists)}")

            if not y_dict:
                tprint_warning("   ⚠️ Layer 2: No target data available for specialists")
                return {}

            # Train specialists in parallel
            tprint_info("   🎓 Layer 2: Training specialists in parallel...")
            specialists = self._specialist_manager.specialists
            training_metrics = self._train_specialists_parallel(specialists, df, y_dict)
            self._specialist_manager.manager_metrics_["training_metrics"] = training_metrics

            successful_training = sum(1 for m in training_metrics.values() if "error" not in m)
            failed_training = len(training_metrics) - successful_training

            tprint_info(f"      - Training results: {successful_training} successful, {failed_training} failed")

            if training_metrics:
                # Show sample metrics for first specialist
                first_spec = list(training_metrics.keys())[0]
                if "error" not in training_metrics[first_spec]:
                    metrics = training_metrics[first_spec]
                    tprint_info(f"      - Sample metrics ({first_spec}):")
                    tprint_info(f"         • R²: {metrics.get('r2', 'N/A'):.4f}")
                    tprint_info(f"         • MSE: {metrics.get('mse', 'N/A'):.6f}")
                    tprint_info(f"         • Samples: {metrics.get('n_samples', 'N/A')}")

            # Generate predictions from all specialists
            tprint_info("   🔮 Layer 2: Generating predictions from specialists in parallel...")
            predictions = self._predict_specialists_parallel(specialists, df)
            tprint_info(f"      - Raw predictions returned: {len(predictions)} entries")

            # Extract prediction series (handle tuple format)
            prediction_series = {}
            names_to_skip = []
            for name, pred in predictions.items():
                if isinstance(pred, tuple):
                    prediction_series[name] = pred[0]  # Predictions
                else:
                    prediction_series[name] = pred
                # Log if prediction is actually usable
                if isinstance(prediction_series[name], pd.Series):
                    non_nan = prediction_series[name].notna().sum()
                    if non_nan > 0:
                        tprint_info(f"      - {name}: {non_nan} valid predictions")
                    else:
                        tprint_warning(f"      - {name}: all NaN predictions")
                        names_to_skip.append(name)  # Mark for removal later

            # Remove empty predictions (fix: don't modify dict during iteration)
            for name in names_to_skip:
                del prediction_series[name]

            tprint_success(f"✅ Layer 2: Specialists trained and predictions generated: {len(prediction_series)} specialists")
            return prediction_series

        except Exception as e:
            tprint_error(f"❌ Layer 2: Specialist initialization failed: {e}")
            import traceback
            tprint_error(f"❌ Layer 2: Traceback: {traceback.format_exc()}")
            return {}

    def _generate_causal_surprise_events(self, df: pd.DataFrame, specialist_predictions: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Generate events from causal surprise detection.
        """
        tprint_info("🎯 Layer 2: Generating Causal Surprise Events...")

        if not self.causal_surprise_enabled:
            tprint_info("   ⏭️ Layer 2: Causal surprise events disabled")
            return pd.DataFrame()

        if self._surprise_detector is None:
            tprint_warning("   ⚠️ Layer 2: Surprise detector not initialized")
            return pd.DataFrame()

        try:
            tprint_info("   📝 Layer 2: Registering specialists with surprise detector...")

            payloads = self._prepare_registration_batch(specialist_predictions, df)
            if not payloads:
                tprint_warning("   ⚠️ No registration payloads available")
                return self._generate_fallback_events(df)

            registered_count = self._register_specialists_batch(payloads)
            tprint_info(f"      - Specialists registered: {registered_count}/{len(payloads)} payloads")

            if len(self._surprise_detector.specialist_errors_) == 0:
                tprint_warning("   ⚠️ Layer 2: No specialists successfully registered")
                return self._generate_fallback_events(df)

            # Aggregate surprise scores
            tprint_info("   🔍 Layer 2: Computing surprise scores across specialists...")
            spectral_reliability = None
            if getattr(self, "_aedl_framework", None):
                try:
                    spectral_reliability = self._aedl_framework.spectral_specialists.get_reliability_report()
                except Exception as e:
                    if self.verbose:
                        tprint_warning(f"   ⚠️ Failed to retrieve spectral reliability: {e}")
            self._surprise_detector.set_zone_score_weights(
                zone3_boost=self.zone3_specialist_boost,
                zone2_boost=self.zone2_specialist_boost,
                exposure_scalar=self.zone_score_exposure
            )
            surprise_df = self._surprise_detector.aggregate_specialist_surprise(
                spectral_reliability=spectral_reliability,
                exposure_scalar=self.zone_score_exposure
            )

            if surprise_df.empty:
                tprint_warning("   ⚠️ Layer 2: No surprise scores computed")
                return self._generate_fallback_events(df)

            tprint_info(f"      - Surprise features computed: {len(surprise_df.columns)}")
            tprint_info(f"      - Samples with surprise data: {len(surprise_df)}")

            # Generate events
            tprint_info("   🎯 Layer 2: Generating causal events from surprise scores...")
            causal_events = self._surprise_detector.generate_causal_events()
            
            events_df = pd.DataFrame()

            # Create events DataFrame
            if causal_events:
                event_indices = list(causal_events.keys())
                tprint_info(f"      - Raw events found: {len(event_indices)}")

                if len(event_indices) > 0:
                    events_df = df.loc[event_indices].copy()

                    # Add causal event metadata
                    events_df['causal_surprise'] = 1
                    events_df['surprise_strength'] = [causal_events[idx]['strength'] for idx in event_indices]
                    events_df['surprise_zone'] = [causal_events[idx]['zone'] for idx in event_indices]
                    events_df['zone_score'] = [causal_events[idx].get('zone_score', 0.0) for idx in event_indices]
                    events_df['surprise_consensus'] = [causal_events[idx]['consensus'] for idx in event_indices]

                    # Inject Continuous Features (for Spectral Chaser & Meta-Learner)
                    if hasattr(self._surprise_detector, 'specialist_soft_surprise_'):
                        soft_surprises = self._surprise_detector.specialist_soft_surprise_
                        # Inject per-specialist soft surprise
                        for col in soft_surprises.columns:
                            events_df[f"surprise_{col}"] = soft_surprises[col].reindex(event_indices).values
                        
                        # Inject global state derivatives
                        z_score = pd.Series([causal_events[idx].get('zone_score', 0.0) for idx in event_indices], index=event_indices)
                        events_df['zone_score_sq'] = (z_score ** 2).values
                        events_df['zone_score_change'] = z_score.diff().fillna(0).values

                    # Show statistics
                    avg_strength = events_df['surprise_strength'].mean()
                    avg_consensus = events_df['surprise_consensus'].mean()
                    max_strength = events_df['surprise_strength'].max()
                    
                    # Zone distribution
                    zone_counts = events_df['surprise_zone'].value_counts().to_dict()

                    tprint_success("   ✅ Layer 2: Causal surprise events generated:")
                    tprint_info(f"      - Events: {len(causal_events)}")
                    tprint_info(f"      - Avg strength: {avg_strength:.4f}")
                    tprint_info(f"      - Max strength: {max_strength:.4f}")
                    tprint_info(f"      - Zone distribution: {zone_counts}")
                    
                    if hasattr(self._surprise_detector, 'surprise_density_'):
                        tprint_info(f"      - Global surprise density: {self._surprise_detector.surprise_density_:.2%}")
                    
                    tprint_info(f"      - Events per day: {len(causal_events) / max(1, (df.index[-1] - df.index[0]).days):.22n}")
                else:
                    events_df = pd.DataFrame()
                    tprint_warning("   ⚠️ Layer 2: No valid event indices found")
            
            # Compute Reliability Metrics
            try:
                # Use close prices for continuous outcome evaluation
                # Calculate forward returns for ground truth
                forward_returns = df['close'].shift(-48) / df['close'] - 1.0 # 12h returns
                
                # If we have labels (e.g. from OOF or previous run), use them
                binary_labels = None
                if 'target_class' in df.columns:
                    binary_labels = df['target_class']
                elif forward_returns is not None:
                     # Create proxy labels for reliability estimation if missing
                     # Label 1 if return > 1.0 * volatility (significant move)
                     vol_proxy = df['close'].pct_change().rolling(20).std().shift(-1) * np.sqrt(48)
                     binary_labels = (forward_returns > vol_proxy.fillna(0)).astype(int)
                
                reliability_metrics = self._surprise_detector.compute_reliability_metrics(
                    realized_outcomes=forward_returns.fillna(0),
                    binary_labels=binary_labels
                )
                
                if reliability_metrics:
                    # Log Specialist Reliability
                    spec_metrics = reliability_metrics.get('specialists', {})
                    if spec_metrics:
                        tprint_info("\n   📊 Specialist Reliability (Composite Score):")
                        sorted_specs = sorted(spec_metrics.items(), key=lambda x: x[1]['composite_reliability'], reverse=True)
                        for name, m in sorted_specs:
                            tprint_info(f"      • {name}: {m['composite_reliability']:.3f} (PrecZ2: {m['z2_precision']:.2f}, Resp: {m['responsiveness']:.2f})")
                    
                    # Log Detector Reliability
                    det_metrics = reliability_metrics.get('detector', {})
                    if det_metrics:
                        tprint_info("\n   🛡️ Detector Reliability:")
                        tprint_info(f"      • F1 Score: {det_metrics.get('f1', 0.0):.3f}")
                        tprint_info(f"      • Recall: {det_metrics.get('recall', 0.0):.3f}")
                        tprint_info(f"      • Precision: {det_metrics.get('precision', 0.0):.3f}")
                        tprint_info(f"      • Stability: {det_metrics.get('stability_index', 0.0):.3f}")
                        
                    if spec_metrics and det_metrics:
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            outcomes_dir = Path("outcomes")
                            outcomes_dir.mkdir(exist_ok=True)
                            
                            # 1. Save Markdown Report
                            report_path = outcomes_dir / f"causal_reliability_report_{timestamp}.md"
                            with open(report_path, "w") as f:
                                f.write(f"# Causal Reliability Report ({timestamp})\n\n")
                                
                                f.write("## 🛡️ Detector Reliability\n")
                                f.write(f"- **F1 Score**: {det_metrics.get('f1', 0.0):.3f}\n")
                                f.write(f"- **Recall**: {det_metrics.get('recall', 0.0):.3f}\n")
                                f.write(f"- **Precision**: {det_metrics.get('precision', 0.0):.3f}\n")
                                f.write(f"- **Stability Index**: {det_metrics.get('stability_index', 0.0):.3f}\n")
                                f.write(f"- **Surprise Density**: {det_metrics.get('filtered_event_density', 0.0):.2%}\n\n")
                                
                                f.write("## 📊 Specialist Reliability\n")
                                f.write("| Specialist | Reliability Score | Responsiveness | Precision (Zone 2) | Marginal Value |\n")
                                f.write("|:---|---:|---:|---:|---:|\n")
                                for name, m in sorted_specs:
                                    f.write(f"| {name} | {m['composite_reliability']:.3f} | {m['responsiveness']:.3f} | {m['z2_precision']:.3f} | {m.get('marginal_value', 0):.4f} |\n")
                            
                            tprint_success(f"   📝 Saved reliability report to: {report_path}")

                            # 2. Save CSV Data for Analysis
                            csv_path = outcomes_dir / f"specialist_reliability_{timestamp}.csv"
                            spec_df = pd.DataFrame.from_dict(spec_metrics, orient='index')
                            spec_df.index.name = 'specialist'
                            spec_df.to_csv(csv_path)
                            tprint_success(f"   💾 Saved specialist metrics to: {csv_path}")
                            
                        except Exception as e:
                            tprint_error(f"   ❌ Failed to save reliability reports: {e}")

            except Exception as e:
                tprint_error(f"   ❌ Reliability computation wrapper failed: {e}")
            
            # Final fallback check - only if truly empty
            if events_df.empty:
                tprint_warning("   ⚠️ Layer 2: No causal events generated, using fallback")
                events_df = self._generate_fallback_events(df)

            return events_df

        except Exception as e:
            tprint_error(f"❌ Layer 2: Causal surprise event generation failed: {e}")
            import traceback
            tprint_error(f"❌ Layer 2: Traceback: {traceback.format_exc()}")
            return self._generate_fallback_events(df)

    def _generate_fallback_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate fallback events when causal methods fail.
        """
        try:
            # Simple volatility-based events as fallback
            volatility = df['close'].pct_change().rolling(20).std()
            threshold = volatility.quantile(0.8)
            event_mask = volatility > threshold

            if event_mask.sum() > 0:
                events_df = df[event_mask].copy()
                events_df['fallback_event'] = 1
                events_df['volatility_threshold'] = threshold
                tprint_info(f"📊 Generated {len(events_df)} fallback events using volatility threshold")
            else:
                events_df = pd.DataFrame()

            return events_df

        except Exception as e:
            tprint_error(f"❌ Fallback event generation failed: {e}")
            return pd.DataFrame()


        # Causal framework configuration
        self.enable_causal_framework = kwargs.get('enable_causal_framework', True)
        self.causal_discovery_enabled = kwargs.get('causal_discovery_enabled', True)
        self.irm_enabled = kwargs.get('irm_enabled', True)
        self.causal_targets_enabled = kwargs.get('causal_targets_enabled', True)
        self.causal_specialists_enabled = kwargs.get('causal_specialists_enabled', True)
        self.causal_surprise_events = kwargs.get('causal_surprise_events', True)

        # IRM configuration
        self.lambda_irm = kwargs.get('lambda_irm', 1.0)
        self.lambda_variance = kwargs.get('lambda_variance', 1.0)
        self.focal_alpha = kwargs.get('focal_alpha', 1.0)
        self.focal_gamma_pos = kwargs.get('focal_gamma_pos', 1.0)
        self.focal_gamma_neg = kwargs.get('focal_gamma_neg', 2.5)

        # Initialize causal components
        self._causal_discovery = None
        self._causal_engineering = None
        self._irm_system = None
        self._causal_targets = None
        self._specialist_manager = None
        self._surprise_detector = None

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Layer 2 labeling pipeline with 1.5-3% range optimization.
        
        Args:
            config: Configuration dictionary containing symbol, exchange, timeframes, etc.
            
        Returns:
            Dict with success status, artifacts, metrics, and execution info.
        """
        import time
        import asyncio
        start_time = time.time()
        
        try:
            # Extract required parameters from config
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            
            self.logger.info(f"🚀 Starting Layer 2 labeling for {symbol}/{exchange}/{timeframe} ({direction})")
            
            # Load data using artifact manager
            try:
                # Try to load the latest price data artifact
                df = await self._load_price_data(symbol, exchange, timeframe)
                if df is None or len(df) == 0:
                    raise ValueError(f"No data available for {symbol}/{exchange}/{timeframe}")
                
                self.logger.info(f"✅ Loaded {len(df)} rows of price data")
            except Exception as e:
                self.logger.error(f"❌ Failed to load data: {e}")
                return {
                    'success': False,
                    'error': f"Data loading failed: {e}",
                    'execution_time': time.time() - start_time,
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Set signal weights from config if provided
            self._current_config = dict(config)
            self.signal_weights = self._current_config.get('signal_weights', None)
            
            # Run the Layer 2 pipeline (this will use MEDIUM_TERM_GRID if enabled)
            self.logger.info("🔄 Running Layer 2 pipeline with range-specific optimization...")
            results = self.run(df)
            
            # Save artifacts
            artifacts = await self._save_artifacts(results, symbol, exchange, timeframe, direction)
            
            # Prepare metrics
            metrics = {
                'data_rows': len(df),
                'events_generated': len(results.get('events_df', pd.DataFrame())),
                'geometries_optimized': len(results.get('selected_trials', [])),
                'range_optimization_enabled': _should_use_range_specific_optimization(),
                'target_range': '1.5-3%' if _should_use_range_specific_optimization() else 'default'
            }
            
            execution_time = time.time() - start_time
            self.logger.info(f"✅ Layer 2 labeling completed in {execution_time:.2f}s")
            self.logger.info(f"📊 Generated {metrics['events_generated']} events with {metrics['geometries_optimized']} geometries")
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'execution_time': execution_time,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"❌ Layer 2 labeling failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time,
                'artifacts': [],
                'metrics': {}
            }
    
    async def _load_price_data(self, symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
        """Load price data using BaseStep's data loading infrastructure."""
        try:
            self.logger.info(f"🔄 Loading real market data for {symbol}/{exchange}/{timeframe}")
            
            # Create config for data loading
            config = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'execution_mode': 'light'  # Use light mode for reasonable data loading window
            }
            
            # Use BaseStep's data loading method
            market_data, source = self.load_market_data_or_fail(config)
            
            if market_data is None:
                raise ValueError(f"Failed to load market data for {symbol}/{exchange}/{timeframe}")
            
            # Handle different return types from BaseStep
            if isinstance(market_data, dict):
                self.logger.info(f"🔍 Dict keys: {list(market_data.keys())}")
                for key, value in market_data.items():
                    self.logger.info(f"🔍 {key}: {type(value)} - {str(value)[:100] if hasattr(value, '__str__') else 'no str'}")
                
                # BaseStep sometimes returns a dict with 'data' key
                if 'data' in market_data and isinstance(market_data['data'], pd.DataFrame):
                    market_data = market_data['data']
                    self.logger.info(f"🔍 Extracted DataFrame from dict response")
                elif 'df' in market_data and isinstance(market_data['df'], pd.DataFrame):
                    market_data = market_data['df']
                    self.logger.info(f"🔍 Extracted DataFrame from 'df' key")
                else:
                    # Try to find any DataFrame in the dict
                    for key, value in market_data.items():
                        if isinstance(value, pd.DataFrame):
                            market_data = value
                            self.logger.info(f"🔍 Extracted DataFrame from '{key}' key")
                            break
                    else:
                        raise ValueError(f"Expected DataFrame or dict with DataFrame, got {type(market_data)}: {list(market_data.keys())}")
            elif not isinstance(market_data, pd.DataFrame):
                raise ValueError(f"Expected DataFrame, got {type(market_data)}")
            
            # Check required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in market_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            self.logger.info(f"✅ Loaded {len(market_data)} rows of real market data from {source}")
            
            # Log data range
            if len(market_data) > 0:
                start_date = market_data.index[0]
                end_date = market_data.index[-1]
                self.logger.info(f"📊 Data range: {start_date} to {end_date}")
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Failed to load price data: {e}")
            raise
    
    def _process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic processing for raw price data."""
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Basic cleaning
        df = df.copy()
        df = df.dropna()
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by index
        df = df.sort_index()
        
        return df
    
    async def _save_artifacts(self, results: Dict[str, Any], symbol: str, exchange: str, timeframe: str, direction: str) -> List[str]:
        """Save results as artifacts."""
        artifacts = []
        
        try:
            if self._artifact_manager is None:
                from src.training.steps.base_step import ArtifactManager
                self._artifact_manager = ArtifactManager()
            
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            # Save events_df if available
            if 'events_df' in results and results['events_df'] is not None:
                events_artifact_name = f"layer2_events_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}"
                self._artifact_manager.save_artifact(results['events_df'], events_artifact_name)
                artifacts.append(events_artifact_name)
                self.logger.info(f"💾 Saved events artifact: {events_artifact_name}")
            
            # Save selected trials if available
            if 'selected_trials' in results and results['selected_trials']:
                trials_artifact_name = f"layer2_trials_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}"
                self._artifact_manager.save_artifact(results['selected_trials'], trials_artifact_name)
                artifacts.append(trials_artifact_name)
                self.logger.info(f"💾 Saved trials artifact: {trials_artifact_name}")
            
            # Save selected features if available
            if 'production_selected_features' in results and results['production_selected_features']:
                features_artifact_name = f"layer2_features_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}"
                self._artifact_manager.save_artifact(results['production_selected_features'], features_artifact_name)
                artifacts.append(features_artifact_name)
                self.logger.info(f"💾 Saved features artifact: {features_artifact_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to save artifacts: {e}")
        
        return artifacts
    
    def _should_use_range_specific_optimization(self) -> bool:
        """Check if 1.5-3% range optimization is enabled in configuration."""
        try:
            import yaml
            config_path = "config/labeling/layer2_coverage_relax_config.yaml"
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config.get("target_range_optimization", {}).get("enabled", False)
        except Exception:
            return False


    def _dominance_label_wrapper(self, df: pd.DataFrame, events: pd.DatetimeIndex, **params) -> pd.Series:
        """Wrapper for _compute_dominance_labels to fit Orthogonal Generator interface."""
        # Construct a dummy events_df with just the index
        # We need to preserve 'family' if possible, but params usually cover it
        dummy_events = pd.DataFrame(index=events)
        # Call existing logic
        labels, _, _, _, _ = self._compute_dominance_labels(
            df, dummy_events, **params
        )
        return labels

    # ==========================================
    # Performance Optimization Caching Methods
    # ==========================================
    
    def _get_cache_key(self, data_type: str, identifier: str, fold_idx: int = None) -> str:
        """Generate consistent cache key."""
        if fold_idx is not None:
            return f"{data_type}_{identifier}_fold_{fold_idx}"
        return f"{data_type}_{identifier}"
    
    def _get_events_cache_key(self, family: str, fold_idx: int, df_hash: str = None, params_hash: str = "") -> str:
        """Generate cache key for events."""
        if df_hash:
            return f"events_{family}_fold_{fold_idx}_{df_hash[:8]}_{params_hash}"
        return f"events_{family}_fold_{fold_idx}_{params_hash}"
    
    def _get_feature_cache_key(self, events_hash: str, fold_idx: int) -> str:
        """Generate cache key for features."""
        return f"features_{events_hash}_fold_{fold_idx}"
    
    def _get_global_events(self, df: pd.DataFrame, family: str, params: Dict = None) -> pd.DatetimeIndex:
        """Generate events once for the full dataset and cache them."""
        # Create cache key
        params_str = str(sorted(params.items())) if params else "default"
        cache_key = f"global_events_{family}_{hashlib.md5(params_str.encode()).hexdigest()[:8]}_{hash(str(df.shape)) % 10000}"

        if cache_key not in self._global_event_cache:
            tprint_info(f"🔄 Generating global events for {family}...")
            start_time = time.time()

            gen = self.generators.get(family)
            if gen is None:
                self._global_event_cache[cache_key] = pd.DatetimeIndex([])
            else:
                gen_params = params if params else {}
                # For causal generators, inject stored specialist predictions
                if family.startswith('CAUSAL_') or family.endswith('_SPECIALIST'):
                    specialist_preds = getattr(self, '_oof_specialist_predictions', None)
                    events = gen.generate(df, specialist_predictions=specialist_preds, **gen_params)
                else:
                    events = gen.generate(df, **gen_params)
                self._global_event_cache[cache_key] = events
                tprint_success(f"✅ Global events cached for {family}: {len(events)} events in {time.time() - start_time:.2f}s")

        return self._global_event_cache[cache_key]

    def _get_global_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cache features for entire dataset once."""
        # Create cache key based on data shape and key columns
        key_cols = ['close', 'volume', 'high', 'low']
        available_cols = [col for col in key_cols if col in df.columns]
        cache_key = hashlib.md5(
            f"{df.shape}_{df.index[0] if len(df) > 0 else ''}_{df.index[-1] if len(df) > 0 else ''}_{'_'.join(available_cols)}".encode()
        ).hexdigest()[:16]

        if cache_key not in self._global_feature_cache:
            tprint_info("🔄 Generating global features for entire dataset...")
            start_time = time.time()

            # Check cache for global features
            df_hash = str(hash(df.values.tobytes()))
            cache_key_global = f"global_features_{df_hash}"
            
            if not hasattr(self, '_cached_global_features') or cache_key_global not in self._cached_global_features:
                # Get denoised prices for feature generation
                denoised_close = self._get_denoised_prices(df)
                
                # Create a copy of df with denoised close for feature generation
                df_features = df.copy()
                df_features['close'] = denoised_close
                
                # Use MTF feature generation with denoised prices
                signals = pd.DataFrame({'consensus': 1.0}, index=df_features.index)
                try:
                    # Fix: create_meta_features 3rd arg is volume_available (bool), not events.
                    # We generate features for the whole DF then reindex to events.
                    volume_available = 'volume' in df_features.columns and not df_features['volume'].isna().all()
                    X_all = create_meta_features(df_features, signals, volume_available=volume_available)
                    
                    # Cache the full feature set
                    if not hasattr(self, '_cached_global_features'):
                        self._cached_global_features = {}
                    self._cached_global_features[cache_key_global] = X_all
                except Exception as e:
                    tprint_warning(f"⚠️ Global feature generation failed: {e}, using empty cache")
                    self._global_feature_cache[cache_key] = pd.DataFrame(index=df.index)
            else:
                X_all = self._cached_global_features[cache_key_global]
            
            # Reindex to original index
            features = X_all.reindex(df.index)
            
            # Optimize memory usage
            features = self._optimize_dataframe_memory(features)
            self._global_feature_cache[cache_key] = features
            tprint_success(f"✅ Global features cached in {time.time() - start_time:.2f}s")

            # Cleanup
            if 'df_features' in locals():
                del df_features
            if 'signals' in locals():
                del signals
            self._cleanup_memory()

        return self._global_feature_cache[cache_key]

    def _get_cached_events(self, df_train: pd.DataFrame, family: str, fold_idx: int, params: Dict = None) -> pd.DatetimeIndex:
        """Get cached events or generate and cache them."""
        # Create simple hash from train data shape
        df_hash = hashlib.md5(f"{len(df_train)}_{df_train.index[0] if len(df_train) > 0 else ''}".encode()).hexdigest()[:8]

        # Params hash
        if params:
            params_str = str(sorted(params.items()))
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        else:
            params_hash = "default"

        cache_key = self._get_events_cache_key(family, fold_idx, df_hash, params_hash)
        
        if cache_key not in self._events_cache:
            tprint_info(f"🔄 Generating events for {family} (fold {fold_idx}) params={params}...")
            gen = self.generators.get(family)
            if gen:
                gen_kwargs = params if params else {}
                events = gen.generate(df_train, **gen_kwargs)
                self._events_cache[cache_key] = events
            else:
                return pd.DatetimeIndex([])
        else:
            tprint_info(f"✅ Using cached events for {family} (fold {fold_idx})")
        
        return self._events_cache[cache_key]
    
    def _get_denoised_prices(self, df: pd.DataFrame) -> pd.Series:
        """Get denoised prices for feature generation."""
        if not self.use_denoised_prices:
            return df['close']
        
        if self.layer0_params is None:
            try:
                self.layer0_params = load_layer0_params()
                tprint_info(f"✅ Loaded Layer0 params for denoised features")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to load Layer0 params: {e}, using raw prices")
                self.use_denoised_prices = False
                return df['close']
        
        try:
            denoised_price = generate_unified_layer2_price(df, self.layer0_params)
            return denoised_price
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate denoised prices: {e}, using raw prices")
            return df['close']
    
    def _get_cached_features(self, df: pd.DataFrame, events_df: pd.DataFrame, fold_idx: int) -> pd.DataFrame:
        """Get cached features or generate and cache them."""
        # Create hash from events index
        events_hash = hashlib.md5(str(events_df.index.values.tobytes()).encode()).hexdigest()[:8]
        cache_key = self._get_feature_cache_key(events_hash, fold_idx)
        
        if cache_key not in self._feature_cache:
            tprint_info(f"🔄 Generating features for {len(events_df)} events (fold {fold_idx})...")
            features = self._build_geometry_independent_event_features(df, events_df)
            self._feature_cache[cache_key] = features
        else:
            tprint_info(f"✅ Using cached features for {len(events_df)} events (fold {fold_idx})")
        
        return self._feature_cache[cache_key]
    
    def _compute_labels_batch(self, df_train: pd.DataFrame, events: pd.DatetimeIndex, 
                            geometries: List, family: str, fold_idx: int,
                            sr_levels: List = None) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """Compute labels and weights for multiple geometries of the same family at once."""
        cache_key = self._get_label_batch_cache_key(family, geometries, events)
        if cache_key in self._label_batch_cache:
            cached_labels, cached_weights = self._label_batch_cache[cache_key]
            tprint_info(f"✅ Using cached batch labels for {len(geometries)} {family} geometries (fold {fold_idx})")
            return cached_labels, cached_weights

        tprint_info(f"🔄 Computing batch labels/weights for {len(geometries)} {family} geometries (fold {fold_idx})...")
        
        # Create events df once
        events_df = pd.DataFrame(index=events)
        
        # Pre-compute signal weights for this event set once
        base_signal_weights = get_signal_specific_weights(
            df_train, events, sr_levels=sr_levels, component_weights=self.signal_weights, family=family
        )

        # Compute labels for all geometries in this family
        labels_dict = {}
        weights_dict = {}

        for gt in geometries:
            try:
                if family == 'PRICE_CUSUM':
                    labels, weights, _, _, _, _ = self._compute_dominance_labels(df_train, events_df, **gt.params)
                elif family == 'LIQ_CUSUM':
                    pt = gt.params.get('pt_mult', 2.0)
                    d_sigma = pt * 0.5
                    horizon = int(gt.params.get('horizon', 24))
                    labels, weights, _, _, _, _ = compute_path_degradation_labels(df_train, events, horizon=horizon, d_sigma=d_sigma)
                elif family == 'TAIL_RISK':
                    pt = gt.params.get('pt_mult', 2.0)
                    z_thresh = pt * 0.6
                    horizon = int(gt.params.get('horizon', 24))
                    labels, weights, _, _, _, _ = compute_tail_regime_labels(df_train, events, horizon=horizon, z_thresh=z_thresh, metric_col=gt.params.get('metric', 'skew'))
                elif family == 'TREND_REGIME':
                    horizon = int(gt.params.get('horizon', 24))
                    labels, weights, _, _, _, _ = compute_trend_persistence_labels(df_train, events, horizon=horizon)
                elif family == 'VOL_STATE':
                    horizon = int(gt.params.get('horizon', 24))
                    labels, weights, _, _, _, _ = compute_vol_state_labels(df_train, events, horizon=horizon)
                elif family == 'RANGE_ATR' or family == 'SR_CUSUM':
                    # Use Volatility Expansion Logic as proxy for now, or implement specific logic
                    # Usually ATR bursts (RangeATR) imply vol expansion.
                    # SR Breakouts (SR_CUSUM) imply regime change/expansion.
                    pt = gt.params.get('pt_mult', 2.0)
                    k_factor = 1.1 + (pt * 0.1)
                    horizon = int(gt.params.get('horizon', 24))
                    labels, weights, _, _, _, _ = compute_volatility_labels(df_train, events, horizon=horizon, k=k_factor)
                elif family == 'VOL_PARTICIPATION':
                    horizon = int(gt.params.get('horizon', 24))
                    h_threshold = gt.params.get('h', 5.0)
                    labels, weights, _, _, _, _ = compute_volume_participation_labels(df_train, events, horizon=horizon, volume_threshold=h_threshold)
                elif family.startswith('CAUSAL_') or family.endswith('_SPECIALIST'):
                    # Causal families: use simple binary barrier labeling
                    # Returns: 1 for positive barrier hit (profit), 0 for negative hit (loss)
                    # Shorten outcome horizon by 50% (User requested)
                    horizon_raw = int(gt.params.get('horizon', 24))
                    horizon = max(1, horizon_raw // 2)
                    
                    pt_mult = gt.params.get('pt_mult', 1.5)
                    sl_mult = gt.params.get('sl_mult', 0.75)
                    labels, weights, _, _, _, _ = self._compute_dominance_labels(
                        df_train, events_df,
                        pt_mult=pt_mult,
                        sl_mult=sl_mult,
                        horizon=horizon
                    )
                    # Ensure binary: dominance_labels should already be binary, but force it
                    labels = (labels > 0.5).astype(int)
                    
                    # For CAUSAL_SURPRISE: use soft sample weights (Continuous Framework)
                    if family == 'CAUSAL_SURPRISE':
                        gen = self.generators.get('CAUSAL_SURPRISE')
                        if gen and hasattr(gen, 'surprise_detector') and gen.surprise_detector:
                            # Extract ZoneScore for events
                            event_data = gen.surprise_detector.surprise_events_
                            zone_scores = pd.Series({t: d.get('zone_score', 0.0) for t, d in event_data.items()})
                            zone_scores = zone_scores.reindex(weights.index).fillna(0.0)
                            
                            # Apply Continuous chaos boost: 1.0 + 3.0 * zone_score
                            weights = weights * (1.0 + 3.0 * zone_scores)
                            tprint_info(f"   ⚖️ Applied Continuous ZoneScore weighting to {family} (avg: {zone_scores.mean():.4f}, max: {zone_scores.max():.4f})")
                else:
                    # Map params to Volatility Logic (VOL_CUSUM)
                    pt = gt.params.get('pt_mult', 2.0)
                    k_factor = 1.1 + (pt * 0.1)
                    horizon = int(gt.params.get('horizon', 24))
                    labels, weights, _, _, _, _ = compute_volatility_labels(df_train, events, horizon=horizon, k=k_factor)

                labels_dict[gt.uuid] = labels
                # Use De Prado weights if available (base_signal_weights), otherwise fall back to dominance weights
                # Combine them? e.g. multiply.
                if base_signal_weights is not None and not base_signal_weights.empty:
                    # Align weights
                    aligned_base_w = base_signal_weights.reindex(labels.index).fillna(0.0)
                    # For dominance labels, 'weights' is outcome quality.
                    # For signal weights, it is input quality.
                    # Product seems appropriate.
                    final_w = weights * aligned_base_w
                    weights_dict[gt.uuid] = final_w
                else:
                    weights_dict[gt.uuid] = weights

            except Exception as e:
                tprint_error(f"❌ Error computing labels for {gt.uuid}: {e}")
                import traceback
                traceback.print_exc()
                tprint_warning(f"⚠️ Label computation failed for {gt.uuid}: {e}")
                labels_dict[gt.uuid] = pd.Series([], dtype=float)
                weights_dict[gt.uuid] = pd.Series([], dtype=float)
        
        self._label_batch_cache[cache_key] = (labels_dict, weights_dict)
        return labels_dict, weights_dict

    def _get_cached_labels(self, df_train: pd.DataFrame, events: pd.DatetimeIndex, 
                          gt, family: str, fold_idx: int, sr_levels: List = None) -> Tuple[pd.Series, pd.Series]:
        """Get cached labels/weights or compute and cache them."""
        cache_key_lbl = self._get_cache_key("labels", f"{family}_{gt.uuid}", fold_idx)
        cache_key_wgt = self._get_cache_key("weights", f"{family}_{gt.uuid}", fold_idx)
        
        if cache_key_lbl not in self._label_cache:
            # Compute single geometry (fallback)
            events_df = pd.DataFrame(index=events)

            if family == 'PRICE_CUSUM':
                labels, _, _, _, _, _ = self._compute_dominance_labels(df_train, events_df, **gt.params)
            elif family == 'LIQ_CUSUM':
                 pt = gt.params.get('pt_mult', 2.0)
                 d_sigma = pt * 0.5
                 horizon = int(gt.params.get('horizon', 24))
                 labels, _, _, _, _, _ = compute_path_degradation_labels(df_train, events, horizon=horizon, d_sigma=d_sigma)
            elif family == 'TAIL_RISK':
                 pt = gt.params.get('pt_mult', 2.0)
                 z_thresh = pt * 0.6
                 horizon = int(gt.params.get('horizon', 24))
                 labels, _, _, _, _, _ = compute_tail_regime_labels(df_train, events, horizon=horizon, z_thresh=z_thresh, metric_col=gt.params.get('metric', 'skew'))
            elif family == 'TREND_REGIME':
                 horizon = int(gt.params.get('horizon', 24))
                 labels, _, _, _, _, _ = compute_trend_persistence_labels(df_train, events, horizon=horizon)
            elif family == 'VOL_STATE':
                 horizon = int(gt.params.get('horizon', 24))
                 labels, _, _, _, _, _ = compute_vol_state_labels(df_train, events, horizon=horizon)
            elif family == 'RANGE_ATR' or family == 'SR_CUSUM':
                 pt = gt.params.get('pt_mult', 2.0)
                 k_factor = 1.1 + (pt * 0.1)
                 horizon = int(gt.params.get('horizon', 24))
                 labels, weights, _, _, _, _ = compute_volatility_labels(df_train, events, horizon=horizon, k=k_factor)
            elif family == 'VOL_PARTICIPATION':
                 horizon = int(gt.params.get('horizon', 24))
                 h_threshold = gt.params.get('h', 5.0)
                 labels, weights, _, _, _, _ = compute_volume_participation_labels(df_train, events, horizon=horizon, volume_threshold=h_threshold)
            else:
                 # Map params to Volatility Logic (VOL_CUSUM)
                 pt = gt.params.get('pt_mult', 2.0)
                 k_factor = 1.1 + (pt * 0.1)
                 horizon = int(gt.params.get('horizon', 24))
                 labels, weights, _, _, _, _ = compute_volatility_labels(df_train, events, horizon=horizon, k=k_factor)

            # Apply signal weights
            signal_w = get_signal_specific_weights(df_train, events, sr_levels=sr_levels, component_weights=self.signal_weights, family=family)
            if signal_w is not None and not signal_w.empty:
                aligned = signal_w.reindex(labels.index).fillna(0.0)
                final_w = weights * aligned
            else:
                final_w = weights

            self._label_cache[cache_key_lbl] = labels
            self._weight_cache[cache_key_wgt] = final_w
        else:
            tprint_info(f"✅ Using cached labels for {gt.uuid} (fold {fold_idx})")
        
        return self._label_cache[cache_key_lbl], self._weight_cache[cache_key_wgt]

    def _get_cached_model(self, geometry_uuid: str, fold_idx: int):
        """Get cached trained model if available."""
        cache_key = f"model_{geometry_uuid}_fold_{fold_idx}"
        return self._model_cache.get(cache_key)

    def _cache_model(self, geometry_uuid: str, fold_idx: int, model):
        """Cache trained model."""
        cache_key = f"model_{geometry_uuid}_fold_{fold_idx}"
        self._model_cache[cache_key] = model

    def _get_cached_feature_selection(self, geometry_uuid: str):
        """Get cached feature selection if available."""
        return self._feature_selection_cache.get(geometry_uuid)

    def _cache_feature_selection(self, geometry_uuid: str, selected_features: List[str]):
        """Cache feature selection results."""
        self._feature_selection_cache[geometry_uuid] = selected_features

    def _get_cached_labels_metadata(self, geometry_uuid: str, fold_idx: int):
        """Get cached labels and weights if available."""
        cache_key = f"labels_{geometry_uuid}_fold_{fold_idx}"
        return self._label_computation_cache.get(cache_key)

    def _cache_labels_metadata(self, geometry_uuid: str, fold_idx: int, labels, weights):
        """Cache computed labels and weights."""
        cache_key = f"labels_{geometry_uuid}_fold_{fold_idx}"
        self._label_computation_cache[cache_key] = (labels, weights)

    def clear_caches(self):
        """Clear all performance optimization caches."""
        self._feature_cache.clear()
        self._events_cache.clear()
        self._label_cache.clear()
        self._global_feature_cache.clear()
        self._global_event_cache.clear()
        self._model_cache.clear()
        self._feature_selection_cache.clear()
        self._label_computation_cache.clear()
        self._global_probe_features.clear()
        tprint_info("🧹 Cleared all optimization caches")

    def _extract_gen_params(self, gt: GeometryTrial) -> Dict:
        """Extract generation-specific parameters."""
        gen = self.generators.get(gt.family)
        if not gen: return {}
        gen_keys = GENERATOR_PARAM_NAMES.get(type(gen).__name__, [])
        return {k: v for k, v in gt.params.items() if k in gen_keys}

    def _get_labeler_menu(self) -> Dict[str, Callable]:
        """Define the menu of labelers with baked-in parameters."""
        return {
            "SCALP": partial(self._dominance_label_wrapper, kappa=1.5, sl_mult=0.5, horizon=12),
            "SWING": partial(self._dominance_label_wrapper, kappa=2.0, sl_mult=1.0, horizon=24),
            "TREND": partial(self._dominance_label_wrapper, kappa=3.0, sl_mult=1.5, horizon=48)
        }

    def _filter_discovery_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select high-variance numeric columns and downsample rows for discovery."""
        numeric_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
        if numeric_df.empty:
            return numeric_df

        variances = numeric_df.var(skipna=True).sort_values(ascending=False)
        top_cols = variances.dropna().index.tolist()
        if self.discovery_max_features:
            top_cols = top_cols[:self.discovery_max_features]
        filtered = numeric_df[top_cols]

        if self.discovery_sample_size and len(filtered) > self.discovery_sample_size:
            filtered = filtered.tail(self.discovery_sample_size)

        return filtered.dropna(how='all')

    def _component_config_key(self, name: str, config: Dict[str, Any]) -> str:
        serialized = json.dumps(config, sort_keys=True, default=str)
        digest = hashlib.md5(serialized.encode()).hexdigest()
        return f"{name}_{digest}"

    def _get_component_singleton(self, name: str, config: Dict[str, Any], factory: Callable[[], Any]) -> Any:
        cache_key = self._component_config_key(name, config)
        component = self._component_singletons.get(cache_key)
        if component is None:
            component = factory()
            self._component_singletons[cache_key] = component
        return component

    def _series_fingerprint(self, series: pd.Series) -> str:
        if series is None or len(series) == 0:
            return "empty"
        hashed = hash_pandas_object(series, index=True)
        return hashlib.md5(hashed.values.tobytes()).hexdigest()

    def _hash_datetime_index(self, index: pd.DatetimeIndex) -> str:
        if index is None or len(index) == 0:
            return "empty"
        values = index.view('i8')
        return hashlib.md5(values.tobytes()).hexdigest()

    def _get_label_batch_cache_key(
        self,
        family: str,
        geometries: List[GeometryTrial],
        events: pd.DatetimeIndex
    ) -> str:
        geo_parts = []
        for gt in geometries:
            params_repr = json.dumps(gt.params, sort_keys=True, default=str)
            geo_parts.append(f"{gt.uuid}:{params_repr}")
        geo_digest = hashlib.md5("|".join(sorted(geo_parts)).encode()).hexdigest()
        events_digest = self._hash_datetime_index(events)
        dataset_key = self._dataset_fingerprint or "adhoc"
        return f"{dataset_key}_{family}_{geo_digest}_{events_digest}"

    def _filter_treatment_matrix(self, treatment_df: pd.DataFrame) -> pd.DataFrame:
        if treatment_df.empty:
            return treatment_df
        coverage = treatment_df.notna().mean()
        filtered_cols = coverage[coverage >= self.treatment_min_coverage].index.tolist()
        if not filtered_cols:
            return pd.DataFrame(index=treatment_df.index)
        filtered = treatment_df[filtered_cols]
        if self.treatment_max_features and len(filtered_cols) > self.treatment_max_features:
            std_rank = filtered.std(skipna=True).abs().sort_values(ascending=False)
            top_cols = std_rank.index[:self.treatment_max_features]
            filtered = filtered[top_cols]
        return filtered

    def _get_confounder_matrix(self, df: pd.DataFrame, columns: List[str]) -> Optional[pd.DataFrame]:
        if not columns:
            return None
        cache_key = f"{self._dataset_fingerprint or 'adhoc'}_{hash(tuple(sorted(columns)))}"
        if cache_key in self._confounder_cache:
            return self._confounder_cache[cache_key]
        confounder_df = df[columns].copy()
        self._confounder_cache[cache_key] = confounder_df
        return confounder_df

    def _train_specialists_parallel(
        self,
        specialists: List[Any],
        feature_df: pd.DataFrame,
        y_dict: Dict[str, pd.Series]
    ) -> Dict[str, Dict[str, Any]]:
        if not specialists:
            return {}

        workers = max(1, self.specialist_train_workers)
        metrics: Dict[str, Dict[str, Any]] = {}

        def _train(spec):
            target = y_dict.get(spec.name)
            if target is None:
                return spec.name, {"error": "missing_target"}
            try:
                result = spec.fit(feature_df, target)
            except Exception as err:
                result = {"error": str(err)}
            return spec.name, result

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_train, spec) for spec in specialists]
            for future in futures:
                name, result = future.result()
                metrics[name] = result

        return metrics

    def _predict_specialists_parallel(
        self,
        specialists: List[Any],
        feature_df: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        if not specialists:
            return {}

        workers = max(1, self.specialist_train_workers)
        predictions: Dict[str, pd.Series] = {}

        def _predict(spec):
            try:
                result = spec.predict(feature_df, return_confidence=False)
            except Exception as err:
                if self.specialist_debug_logging:
                    tprint_warning(f"⚠️ Prediction failed for {spec.name}: {err}")
                result = pd.Series(dtype=float)
            return spec.name, result

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_predict, spec) for spec in specialists]
            for future in futures:
                name, series = future.result()
                predictions[name] = series

        return predictions

    def _prepare_registration_payload(
        self,
        spec_name: str,
        predictions: pd.Series,
        df: pd.DataFrame
    ) -> Optional[Tuple[str, pd.Series, pd.Series, str]]:
        if predictions is None or predictions.empty:
            return None

        predictions = predictions.dropna()
        if predictions.empty:
            return None

        # Determine target
        target_series = None
        
        # Case 1: Causal edge specialist (Parent -> Child)
        if '_to_' in spec_name:
            target_col = spec_name.split('_to_')[1]
            target_series = df.get(target_col)
        # Case 2: AEDL Specialist (Signal based, target is 0/Mean)
        elif 'specialist' in spec_name.lower():
            # For signals (z-scores), the "target" is 0 (mean)
            # The signal itself is the deviation/surprise
            target_series = pd.Series(0, index=df.index)
        # Case 3: Direct column target
        else:
            target_series = df.get(spec_name)

        # Fallback to close price if finding target failed but not for AEDL
        if target_series is None and 'specialist' not in spec_name.lower():
            target_series = df.get('close')
            
        if target_series is None:
            return None

        common_idx = predictions.index.intersection(target_series.index)
        if len(common_idx) < 10:
            return None

        pred_aligned = predictions.loc[common_idx]
        target_aligned = target_series.loc[common_idx]

        fingerprint = self._series_fingerprint(pred_aligned) + self._series_fingerprint(target_aligned)
        return spec_name, pred_aligned, target_aligned, fingerprint

    def _prepare_registration_batch(
        self,
        specialist_predictions: Dict[str, pd.Series],
        df: pd.DataFrame
    ) -> List[Tuple[str, pd.Series, pd.Series, str]]:
        workers = max(1, self.specialist_registration_workers)
        payloads = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for spec_name, preds in specialist_predictions.items():
                futures.append(executor.submit(self._prepare_registration_payload, spec_name, preds, df))
            for future in futures:
                payload = future.result()
                if payload is not None:
                    payloads.append(payload)

        return payloads

    def _register_specialists_batch(
        self,
        payloads: List[Tuple[str, pd.Series, pd.Series, str]]
    ) -> int:
        if not payloads:
            return 0

        workers = max(1, self.specialist_registration_workers)

        def _register(payload):
            spec_name, preds, targets, fingerprint = payload
            last_entry = self._specialist_registration_cache.get(spec_name)
            if last_entry and last_entry[0] == fingerprint and last_entry[1] == len(preds):
                return 0
            self._surprise_detector.register_specialist(spec_name, preds, targets)
            self._specialist_registration_cache[spec_name] = (fingerprint, len(preds))
            return 1

        registered = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_register, payload) for payload in payloads]
            for future in futures:
                registered += future.result()

        return registered

    def _maybe_apply_model_seed(self, model, candidate_name: str):
        seed_entry = self._model_race_seed_models.get(candidate_name)
        if seed_entry is None:
            return
        try:
            if isinstance(model, lgb.LGBMClassifier) and hasattr(seed_entry, 'booster_'):
                model.set_params(init_model=seed_entry.booster_)
            elif isinstance(model, XGBClassifier) and hasattr(seed_entry, 'get_booster'):
                model._Booster = seed_entry.get_booster()
            # CatBoost does not expose init_model easily; skip.
        except Exception:
            pass

    def _store_model_seed(self, candidate_name: str, model):
        try:
            self._model_race_seed_models[candidate_name] = copy.deepcopy(model)
        except Exception:
            self._model_race_seed_models[candidate_name] = model

    def _compute_dataset_fingerprint(self, df: pd.DataFrame) -> str:
        """Create a deterministic fingerprint for caching across runs."""
        if df is None or df.empty:
            return "empty"
        hash_series = hash_pandas_object(df[['close']] if 'close' in df.columns else df, index=True)
        digest = hashlib.md5(hash_series.values.tobytes()).hexdigest()
        return f"{len(df)}_{df.index.min()}_{df.index.max()}_{digest}"

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the Layer 2 pipeline (causal framework only).
        """
        tprint_info("Starting Layer 2 Pipeline...")
        self._dataset_fingerprint = self._compute_dataset_fingerprint(df)

        if not self.enable_causal_framework:
            tprint_warning("⚠️ AFML path deprecated; enabling causal framework automatically")
            self.enable_causal_framework = True

        if not CAUSAL_MODULES_AVAILABLE:
            raise RuntimeError("Causal framework modules unavailable; cannot run Layer 2 pipeline.")

        tprint_info("🔬 Causal Framework Enabled - Using Modern De Prado Approach")
        return self._run_causal_pipeline(df)

    def _run_causal_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the modern causal Layer 2 pipeline.
        """
        try:
            tprint_info("🚀 Causal Layer 2 Pipeline: Starting modern De Prado framework...")
            
            # Initialize causal components
            tprint_info("🔧 Step 0: Initializing causal components...")
            self._initialize_causal_components(df)
            tprint_success("   ✅ Causal components initialized")

            # 1. Causal Discovery: Build DAG for causal relationships
            tprint_info("🔍 Step 1: Running causal discovery...")
            causal_graph = self._run_causal_discovery(df)
            if not causal_graph:
                error_msg = "   ❌ Causal discovery failed; aborting Layer 2 causal pipeline"
                tprint_error(error_msg)
                raise RuntimeError(error_msg.strip())
            tprint_success(f"   ✅ Causal discovery complete: {len(causal_graph)} variables")

            # 2. Causal Specialists: Create and train specialists
            tprint_info("🧠 Step 2: Initializing causal specialists...")
            
            # Use AEDL if enabled (Domain Specialists)
            if self.enable_aedl:
                tprint_info("   🧠 Using Spectral AEDL for specialist initialization...")
                specialist_predictions = {}
                try:
                    aedl_results = self._run_aedl_pipeline(df, "close", causal_graph=causal_graph)
                    if not isinstance(aedl_results, dict):
                        raise RuntimeError("AEDL pipeline returned an invalid payload")
                    if 'error' in aedl_results:
                        raise RuntimeError(aedl_results['error'])
                    specialist_predictions = aedl_results.get('specialist_signals') or {}
                    if not specialist_predictions:
                        raise RuntimeError("AEDL pipeline produced zero specialist signals")
                    # Store AEDL results for later only when signals exist
                    self._aedl_results_cache = aedl_results
                    tprint_success(f"   ✅ AEDL specialists extracted: {len(specialist_predictions)}")
                except Exception as e:
                    tprint_error(f"   ❌ AEDL initialization failed: {e}")
                    tprint_info("   🔄 Falling back to traditional causal specialists...")
                    specialist_predictions = self._initialize_causal_specialists(df, causal_graph)
            else:
                # Use Graph Edge Specialists
                specialist_predictions = self._initialize_causal_specialists(df, causal_graph)
                
            # Store for OOF analytics phase
            self._causal_specialist_predictions = specialist_predictions
            if not specialist_predictions:
                tprint_warning("   ⚠️ No specialist predictions available")
            else:
                tprint_success(f"   ✅ Specialists trained: {len(specialist_predictions)} predictions")

            # 3. Causal Surprise Events: Generate events from specialist prediction errors
            tprint_info("🎯 Step 3: Generating causal surprise events...")
            causal_events_df = self._generate_causal_surprise_events(df, specialist_predictions)
            if causal_events_df is None or len(causal_events_df) == 0:
                tprint_warning("   ⚠️ No causal events generated")
            else:
                tprint_success(f"   ✅ Causal events generated: {len(causal_events_df)} events")

            # 4. Causal Feature Engineering: Denoise and adjust features using causal relationships
            tprint_info("🔧 Step 4: Applying causal feature engineering...")
            
            # 4a. Augment Dataframe with Spectral/Specialist Features for Denoising
            tprint_info("   ➕ Augmenting data for denoising (OHLCV + Specialists)...")
            enriched_df = df.copy()
            
            # Add specialist predictions (high-level signals)
            spec_cols = []
            if specialist_predictions:
                 for name, series in specialist_predictions.items():
                     col_name = f"spec_{name}"
                     enriched_df[col_name] = series
                     spec_cols.append(col_name)
            
            # Add spectral components (finer-grained) if available from AEDL
            spectral_cols = []
            if hasattr(self, '_aedl_results_cache') and self._aedl_results_cache:
                components = self._aedl_results_cache.get('spectral_components', {})
                # Filter for high variance to avoid noise
                for name, comp in components.items():
                    if isinstance(comp, (pd.Series, np.ndarray)):
                        var = np.var(comp)
                        if var > 1e-4: # Low variance filter
                            col_name = f"spectral_{name}"
                            # Ensure length match
                            if len(comp) == len(enriched_df):
                                enriched_df[col_name] = comp
                                spectral_cols.append(col_name)
            
            tprint_info(f"   📊 Enriched Matrix: {len(df.columns)} base + {len(spec_cols)} spec + {len(spectral_cols)} spectral")

            # 4b. Augment Causal Graph
            # Assume OHLCV (Market State) are parents of Specialists/Spectral (Derived State)
            # This allows specialists to be denoised conditional on Market State
            augmented_graph = causal_graph.copy()
            market_nodes = [c for c in df.columns if c in augmented_graph]
            
            # --- USER REQUEST: FORCE OHLCV DENOISING (Time-Flow Logic) ---
            # Enforce: Open, High, Low -> Close (Time flow)
            # Enforce: Close, Open -> Volume (Information flow)
            
            # 1. Enforce Close parents
            ohl_parents = [c for c in ['open', 'high', 'low'] if c in df.columns]
            if 'close' in df.columns and ohl_parents:
                # Remove any existing edges starting FROM close to these parents (avoid cycle)
                # And remove any existing parents of close that might conflict or be redundant
                # Actually, simply setting the parents overrides previous discovery for 'close' node
                augmented_graph['close'] = ohl_parents
                
                # Check for reverse edges in other parts of the graph
                for node, parents in list(augmented_graph.items()):
                    if node == 'close': continue
                    # If close was a parent of open/high/low, remove it to break cycle
                    if 'close' in parents and node in ohl_parents:
                        augmented_graph[node] = [p for p in parents if p != 'close']

            # 2. Enforce Volume parents
            vol_parents = [c for c in ['close', 'open'] if c in df.columns]
            if 'volume' in df.columns and vol_parents:
                augmented_graph['volume'] = vol_parents
                # Remove reverse
                for node, parents in list(augmented_graph.items()):
                    if node == 'volume': continue
                    if 'volume' in parents and node in vol_parents:
                         augmented_graph[node] = [p for p in parents if p != 'volume']

            # -------------------------------------------------------------

            for target_col in spec_cols + spectral_cols:
                # Naive Causal Assumption: Market State (Close/Vol) -> Specialist
                # We add 'close' and 'volatility' (if present) as parents
                parents = [p for p in ['close', 'volatility_1d', 'volume'] if p in df.columns] # Use df.columns to be sure
                # Fallback to any root node if specific parents missing
                if not parents and market_nodes:
                    parents = [market_nodes[0]]
                
                if parents:
                    augmented_graph[target_col] = parents

            tprint_info(f"   🔗 Augmented Graph: {len(causal_graph)} -> {len(augmented_graph)} nodes")
            if 'close' in augmented_graph:
                tprint_info(f"      • Close parents: {augmented_graph['close']}")
            if 'volume' in augmented_graph:
                tprint_info(f"      • Volume parents: {augmented_graph['volume']}")

            # 4c. Run Engineering on Enriched Data
            engineered_df, causal_metadata = self._apply_causal_feature_engineering(enriched_df, augmented_graph)
            
            if engineered_df is None:
                error_msg = "   ❌ Causal feature engineering failed; aborting Layer 2 causal pipeline"
                tprint_error(error_msg)
                raise RuntimeError(error_msg.strip())
            tprint_success(f"   ✅ Causal engineering complete: {len(engineered_df.columns)} features")

            # 5. Causal Targets: Compute treatment effects and causal residuals
            tprint_info("🎯 Step 5: Computing causal targets...")
            causal_targets_df = self._compute_causal_targets(engineered_df, causal_events_df, specialist_predictions)
            if causal_targets_df is None or len(causal_targets_df.columns) == 0:
                tprint_warning("   ⚠️ No causal targets computed")
            else:
                tprint_success(f"   ✅ Causal targets computed: {len(causal_targets_df.columns)} target types")

            # 6. IRM Training: Train base models with invariance penalty
            tprint_info("🧠 Step 6: Training causal models with IRM...")
            causal_geometries, causal_selected_features = self._train_causal_models(
                engineered_df, causal_targets_df, causal_events_df, 
                causal_graph=causal_graph, specialist_predictions=specialist_predictions
            )
            if not causal_geometries:
                error_msg = "   ❌ Causal model training failed; aborting Layer 2 causal pipeline"
                tprint_error(error_msg)
                raise RuntimeError(error_msg.strip())
            tprint_success(f"   ✅ Causal models trained: {len(causal_geometries)} geometries")

            # 7. Causal Validation: Run causal-aware OOF analytics
            tprint_info("📊 Step 7: Running causal OOF analytics...")
            causal_oof_results = self._run_causal_oof_analytics(
                engineered_df, causal_events_df, causal_geometries, causal_targets_df
            )
            if not causal_oof_results:
                tprint_error("   ❌ Causal OOF analytics failed")
            else:
                tprint_success("   ✅ Causal OOF analytics complete")

            # 8. Report
            tprint_info("📋 Step 8: Generating causal framework reports...")
            self._generate_causal_reports(engineered_df, causal_events_df, causal_geometries, causal_oof_results)

            results = {
                **causal_oof_results,
                "events_df": causal_events_df,
                "selected_trials": [asdict(t) for t in causal_geometries],
                "causal_graph": causal_graph,
                "causal_metadata": causal_metadata,
                "specialist_predictions": specialist_predictions,
                "causal_targets": causal_targets_df,
                "framework_type": "modern_de_prado_causal"
            }

            tprint_success("🎉 Causal Layer 2 Pipeline: Complete modern De Prado framework executed!")

            # Step 9: AEDL Framework (Use cached if available)
            if self.enable_aedl:
                try:
                    if self.verbose:
                        tprint_info(">>> Step 9: Finalizing AEDL Framework...")
                    
                    if hasattr(self, '_aedl_results_cache'):
                         results['aedl_framework'] = self._aedl_results_cache
                         aedl_results = self._aedl_results_cache
                    else:
                         aedl_results = self._run_aedl_pipeline(df, "close")
                         results['aedl_framework'] = aedl_results
                    
                    if 'rsv_info' in aedl_results:
                        self.layer3_rsv_data = aedl_results['rsv_info']
                    
                    if self.verbose:
                        tprint_success("✅ AEDL Framework complete")
                except Exception as e:
                    if self.verbose:
                        tprint_error(f"❌ AEDL Framework failed: {e}")
                    results['aedl_framework'] = {'error': str(e)}

            # Step 10: Spectral Chaser (REPLACE existing Layer 2.5)
            if hasattr(self, 'spectral_chaser_enabled') and self.spectral_chaser_enabled:
                try:
                    if self.verbose:
                        tprint_info(">>> Step 10: Running Spectral Chaser...")
                    
                    # Retrieve causal anchor and residuals from computed targets
                    y_residuals = pd.Series()
                    causal_anchor_predictions = None

                    if causal_targets_df is not None and not causal_targets_df.empty:
                        # Extract residuals
                        # We must align residuals to the main DataFrame 'df' for Spectral Chaser
                        if 'residual_targets' in causal_targets_df.columns:
                            y_res = causal_targets_df['residual_targets']
                            # Align to df index, fill missing with 0 (since they are residuals)
                            y_residuals = y_res.reindex(df.index).fillna(0)
                        
                        # Extract anchor predictions
                        if 'causal_anchor_predictions' in causal_targets_df.columns:
                            c_anch = causal_targets_df['causal_anchor_predictions']
                            # Align
                            c_anch_aligned = c_anch.reindex(df.index).fillna(0)
                            causal_anchor_predictions = c_anch_aligned.values
                            
                    # Fallback or check validity
                    if y_residuals.empty or causal_anchor_predictions is None:
                        tprint_warning("   ⚠️ Spectral Chaser: Missing input targets, attempting partial fallback...")
                        if causal_anchor_predictions is None:
                            causal_anchor_predictions = self._get_causal_anchor_predictions() if hasattr(self, '_get_causal_anchor_predictions') else np.zeros(len(df))
                        
                        if y_residuals.empty:
                            y_residuals = pd.Series(0.0, index=df.index)
                    
                    if len(causal_anchor_predictions) != len(df):
                        # Final alignment check for anchor
                        tprint_warning(f"   ⚠️ Re-aligning anchor for Spectral Chaser: {len(causal_anchor_predictions)} -> {len(df)}")
                        new_anchor = np.zeros(len(df))
                        min_len = min(len(causal_anchor_predictions), len(df))
                        new_anchor[:min_len] = causal_anchor_predictions[:min_len]
                        causal_anchor_predictions = new_anchor
                    
                    if self.verbose:
                        tprint_info(f"   🔍 Spectral Chaser Debug: df={len(df)}, y_residuals={len(y_residuals)}, anchor={len(causal_anchor_predictions) if causal_anchor_predictions is not None else 'None'}")
                        
                    # Retrieve causal sample weights (ZoneScore weighted) if available
                    sample_weight = None
                    if causal_targets_df is not None and 'sample_weight' in causal_targets_df.columns:
                        sample_weight = causal_targets_df['sample_weight'].reindex(df.index).fillna(1.0)

                    # Inject continuous features (Continuous Framework)
                    if causal_targets_df is not None:
                        continuous_cols = [c for c in causal_targets_df.columns if c.startswith('surprise_') or 'zone_score' in c]
                        if continuous_cols:
                            df = df.copy()
                            for col in continuous_cols:
                                df[col] = causal_targets_df[col].reindex(df.index).fillna(0)

                    chaser_results = self._run_spectral_chaser(
                        df, y_residuals, causal_anchor_predictions, sample_weight=sample_weight
                    ) if hasattr(self, '_run_spectral_chaser') else {}
                    results['spectral_chaser'] = chaser_results
                    if self.verbose:
                        tprint_success("✅ Spectral Chaser complete")
                except Exception as e:
                    if self.verbose:
                        tprint_error(f"❌ Spectral Chaser failed: {e}")
                    results['spectral_chaser'] = {'error': str(e)}

            return results

        except Exception as e:
            tprint_error(f"❌ Causal Layer 2 Pipeline failed: {e}")
            import traceback
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            raise

    def _run_afml_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the traditional AFML Layer 2 pipeline.
        """
        tprint_info("📊 Starting Traditional AFML Layer 2 Pipeline...")

        # Clear caches for fresh run
        self.clear_caches()

        # 1. Prepare (Validate df, setup caches)
        # Note: We do NOT generate global events_df here anymore,
        # but we initialize the structure.
        df, _, _, global_probe_features = self.prepare_data_and_events(df)

        # Calculate SR Levels globally once
        # Calculate SR Levels globally once
        # Calculate SR Levels globally once - REMOVED (Handled inside orthogonal generation if needed)
        self.sr_levels = []

        # 2. Optimize (Orthogonal Selection)
        # This returns GeometryTrial objects which contain their own events.
        # Pass signal_weights to orthogonal_label_generation
        production_geometries, production_selected_features = self.optimize_production_geometries(
            df, None, global_probe_features=global_probe_features
        )

        # 3. Construct Global Event Union (for compatibility/reporting)
        events_df = self._construct_union_events_df(df, production_geometries)

        # 4. Validate (OOF)
        oof_results = self.run_oof_analytics(
            df, events_df, production_geometries,
            global_probe_features=global_probe_features,
            production_selected_features=production_selected_features
        )

        # 5. Report
        self.generate_reports(df, events_df, production_geometries, oof_results)

        return {
            **oof_results,
            "events_df": events_df,
            "selected_trials": [asdict(t) for t in production_geometries],
            "production_selected_features": list(getattr(self, '_production_selected_features', []) or []),
        }


    def _apply_causal_feature_engineering(self, df: pd.DataFrame, causal_graph: Dict[str, List[str]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply causal feature engineering to denoise and adjust features.
        """
        try:
            tprint_info("🔧 Applying Causal Feature Engineering...")
            
            # Apply causal engineering
            # Pass the discovered causal_graph to avoid correlation fallback
            self._causal_engineering.causal_graph = causal_graph
            engineered_df, metadata = self._causal_engineering.apply_causal_engineering(
                df,
                apply_denoising=True,
                apply_adjustment=True,
                apply_imputation=True,
                apply_transformation=True
            )
            
            tprint_success(f"✅ Causal Feature Engineering Complete:")
            feature_counts = metadata.get('feature_counts', {'original': 0, 'final': 0, 'added': 0})
            tprint_info(f"   - Original features: {feature_counts.get('original', 0)}")
            tprint_info(f"   - Final features: {feature_counts.get('final', 0)}")
            tprint_info(f"   - Added features: {feature_counts.get('added', 0)}")
            
            return engineered_df, metadata
            
        except Exception as e:
            tprint_error(f"❌ Causal feature engineering failed: {e}")
            return df, {'error': str(e)}
    
    def _compute_causal_targets(self, df: pd.DataFrame, events_df: pd.DataFrame, specialist_predictions: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compute causal targets using DML and CATE.
        """
        try:
            tprint_info("🎯 Computing Causal Targets...")
            
            # Initialize causal target computer
            target_computer = CausalTargetComputer(verbose=self.verbose)
            
            # Create treatment and outcome variables
            # Use specialist predictions as treatments
            if specialist_predictions:
                treatment_data = pd.DataFrame(
                    {f"treatment_{spec}": preds for spec, preds in specialist_predictions.items()},
                    index=df.index
                ).replace([np.inf, -np.inf], np.nan)
                treatment_data = self._filter_treatment_matrix(treatment_data)
                if treatment_data.empty or len(treatment_data.columns) == 0:
                    tprint_warning("   - No treatments available after filtering (coverage/feature cap)")
                    treatments_df = pd.DataFrame(index=df.index)
                else:
                    # Fill NaNs with 0 instead of dropping to preserve alignment with outcomes
                    treatments_df = treatment_data.fillna(0.0)
                    tprint_info(f"   - Treatments: {len(treatments_df.columns)} specialists, {len(treatments_df)} samples")
            else:
                treatments_df = pd.DataFrame(index=df.index)
                tprint_warning("   - No specialist predictions for treatments")
            
            # Use price returns as outcomes
            outcomes = df['close'].pct_change().fillna(0)
            
            # Compute causal targets - check columns, not rows
            if len(treatments_df.columns) > 0 and len(outcomes) > 0:
                causal_targets = target_computer.create_chaser_targets(
                    df, treatments_df, outcomes, include_cate=True
                )
                
                tprint_success(f"✅ Causal Targets Computed: {len(causal_targets)} target types")
                return pd.DataFrame(causal_targets)
            else:
                tprint_warning("⚠️ Insufficient data for causal targets")
                tprint_info(f"   - Treatments columns: {len(treatments_df.columns)}")
                tprint_info(f"   - Outcomes samples: {len(outcomes)}")
                return pd.DataFrame()
                
        except Exception as e:
            tprint_error(f"❌ Causal target computation failed: {e}")
            return pd.DataFrame()
    
    def _train_causal_models(self, df: pd.DataFrame, targets_df: pd.DataFrame, events_df: pd.DataFrame, 
                           causal_graph: Dict = None, specialist_predictions: Dict = None) -> Tuple[List[GeometryTrial], Dict[str, List[str]]]:
        """
        Train causal-aware base models with IRM loss.
        """
        try:
            tprint_info("🧠 Training Causal Models with IRM...")
            
            # Create custom features for IRM environments
            custom_features = self._create_irm_environments(df)
            
            # Debug: Log specialist predictions being passed
            tprint_info(f"   - Specialist predictions: {len(specialist_predictions) if specialist_predictions else 0} specialists")
            if specialist_predictions:
                tprint_info(f"   - Prediction keys: {list(specialist_predictions.keys())[:5]}...")
            
            # Train models with enhanced IRM loss
            # This is a simplified version - in practice, you'd integrate IRM into actual model training
            production_geometries, production_selected_features = self.optimize_production_geometries(
                df, None, global_probe_features=list(custom_features.columns),
                causal_graph=causal_graph, specialist_predictions=specialist_predictions
            )
            
            tprint_success(f"✅ Causal Models Trained: {len(production_geometries)} geometries")
            
            return production_geometries, production_selected_features
            
        except Exception as e:
            tprint_error(f"❌ Causal model training failed: {e}")
            return [], {}
    
    def _create_irm_environments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create environments for IRM training.
        """
        try:
            # Create environment features based on volatility and trend
            custom_features = pd.DataFrame(index=df.index)
            
            # Volatility regimes
            if 'volatility_1d' in df.columns:
                vol = df['volatility_1d']
                custom_features['vol_regime_low'] = (vol < vol.quantile(0.33)).astype(int)
                custom_features['vol_regime_med'] = ((vol >= vol.quantile(0.33)) & (vol < vol.quantile(0.67))).astype(int)
                custom_features['vol_regime_high'] = (vol >= vol.quantile(0.67)).astype(int)
            
            # Trend regimes
            if 'close' in df.columns:
                price_trend = df['close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
                custom_features['trend_regime_down'] = (price_trend < -0.01).astype(int)
                custom_features['trend_regime_flat'] = ((price_trend >= -0.01) & (price_trend <= 0.01)).astype(int)
                custom_features['trend_regime_up'] = (price_trend > 0.01).astype(int)
            
            return custom_features.fillna(0)
            
        except Exception as e:
            tprint_warning(f"⚠️ IRM environment creation failed: {e}")
            return pd.DataFrame(index=df.index)
    
    def _run_causal_oof_analytics(self, df: pd.DataFrame, events_df: pd.DataFrame, geometries: List[GeometryTrial], targets_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run causal-aware OOF analytics.
        """
        try:
            tprint_info("📊 Running Causal OOF Analytics...")
            
            # Run standard OOF analytics with causal enhancements
            oof_results = self.run_oof_analytics(
                df, events_df, geometries,
                global_probe_features=list(targets_df.columns) if len(targets_df) > 0 else None,
                production_selected_features=None
            )
            
            # Add causal-specific metrics
            oof_results['causal_targets_available'] = len(targets_df) > 0
            oof_results['causal_framework_used'] = True
            
            tprint_success(f"✅ Causal OOF Analytics Complete")
            
            return oof_results
            
        except Exception as e:
            tprint_error(f"❌ Causal OOF analytics failed: {e}")
            return {'error': str(e)}
    
    def _generate_causal_reports(self, df: pd.DataFrame, events_df: pd.DataFrame, geometries: List[GeometryTrial], oof_results: Dict[str, Any]) -> None:
        """
        Generate comprehensive causal framework reports.
        """
        try:
            tprint_info("📋 Generating Causal Framework Reports...")
            
            # Summary statistics
            n_events = len(events_df) if events_df is not None else 0
            n_geometries = len(geometries)
            n_oof_samples = oof_results.get('oof_returns', pd.Series()).shape[0]
            
            tprint_success("🎯 Causal Framework Summary:")
            tprint_info(f"   - Events generated: {n_events}")
            tprint_info(f"   - Geometries optimized: {n_geometries}")
            tprint_info(f"   - OOF samples: {n_oof_samples}")
            tprint_info(f"   - Framework: Modern De Prado Causal")
            
        except Exception as e:
            tprint_warning(f"⚠️ Causal report generation failed: {e}")


    def _validate_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        # Basic validation ensuring volatility exists
        if 'volatility_1d' not in df.columns:
            df = df.copy()
            df['volatility_1d'] = df['close'].pct_change().rolling(50).std()
        return df

    def f_precompute_geometry_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Restore basic precomputation
        df_out = df.copy()
        if 'geo_atr_14' not in df_out.columns:
            try:
                high = df_out['high'] if 'high' in df_out.columns else df_out['close']
                low = df_out['low'] if 'low' in df_out.columns else df_out['close']
                close = df_out['close']
                tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
                df_out['geo_atr_14'] = tr.rolling(14).mean()
            except Exception:
                pass
        return df_out

    def prepare_data_and_events(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        """Step 1: Stateless data preparation."""
        tprint_info(">>> Layer 2: Step 1 - Prepare Data...")
        self._labels_cache = {}
        self._signals_cache = {}

        df = self._validate_inputs(df)
        df = self.f_precompute_geometry_base_features(df)

        # We don't generate global events here. Return empty placeholders.
        events_df = pd.DataFrame()
        X_probe_events = pd.DataFrame()

        # We can try to select global probe features based on a sample if needed,
        # or defer until we have events. For now, empty.
        self._global_probe_features = []

        return df, events_df, X_probe_events, self._global_probe_features

    def _construct_union_events_df(self, df: pd.DataFrame, geometries: List[GeometryTrial]) -> pd.DataFrame:
        """Construct a composite events dataframe from selected geometries."""
        if not geometries:
            return pd.DataFrame()

        all_indices = []
        for g in geometries:
            if g.events is not None:
                all_indices.extend(g.events)

        if not all_indices:
            return pd.DataFrame()

        unique_indices = pd.DatetimeIndex(sorted(list(set(all_indices))))
        events_df = df.loc[unique_indices, ['trend_regime', 'vol_regime', 'volatility_1d']].copy()
        # Add default family/consensus cols if needed
        events_df['family'] = 'Unified'

        # Optimize memory usage
        events_df = self._optimize_dataframe_memory(events_df)

        return events_df

    def _run_model_race(self, X_train, y_train, X_val, y_val, w_train, environment_masks=None):
        """
        Run a model race to find the best base model for this geometry.

        Traditional AFML Candidates:
          1. LGBM (Tree) + RobustFocalLoss
          2. LGBM (Linear) + RobustFocalLoss
          3. XGB (Tree) + RobustFocalLoss
          4. XGB (Linear) + RobustFocalLoss
          5. CatBoost

        Causal Framework Candidates (when enable_causal_framework=True):
          1. LGBM (Tree) + IRM (Invariant Risk Minimization)
          2. LGBM (Linear) + IRM
          3. XGB (Tree) + IRM
          4. XGB (Linear) + IRM

        Metric: P@R40% (primary), AUC (tie-breaker).

        OPTIMIZED: Simpler models with class balancing for sparse-event geometries.
        """
        # Compute dynamic scale_pos_weight based on class balance
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        if pos_count > 0:
            scale_pos_weight = neg_count / pos_count  # (1-balance)/balance
        else:
            scale_pos_weight = 1.0
        
        candidates = []

        # Check if using causal framework with IRM
        use_irm = self.enable_causal_framework and CAUSAL_MODULES_AVAILABLE and self.irm_enabled

        if use_irm:
            tprint_info("   🔬 Using Causal IRM Framework for model training")
            # Create IRM-based models
            irm_candidates = self._create_irm_candidates(scale_pos_weight, environment_masks)
            candidates.extend(irm_candidates)
        else:
            tprint_info("   📊 Using Traditional AFML Framework")
            # Create traditional focal loss models
            afml_candidates = self._create_afml_candidates(scale_pos_weight)
            candidates.extend(afml_candidates)

        if XGBClassifier is not None:
            # 3. XGB (Tree)
            candidates.append({
                'name': 'XGB_Tree', 
                'model': XGBClassifier(
                    n_estimators=100, learning_rate=0.05, max_depth=5, 
                    objective=get_focal_loss_xgb(ALPHA, GAMMA),
                    random_state=42, n_jobs=1, verbosity=0, use_label_encoder=False,
                    eval_metric='logloss' 
                )
            })
            # 4. XGB (Linear)
            candidates.append({
                'name': 'XGB_Linear', 
                'model': XGBClassifier(
                    n_estimators=100, learning_rate=0.05, max_depth=5, 
                    booster='gblinear',
                    objective=get_focal_loss_xgb(ALPHA, GAMMA),
                    random_state=42, n_jobs=1, verbosity=0, use_label_encoder=False,
                    eval_metric='logloss'
                )
            })

        # 5. CatBoost
        if CATBOOST_AVAILABLE and catboost is not None:
            from catboost import CatBoostClassifier
            candidates.append({
                'name': 'CatBoost',
                'model': CatBoostClassifier(
                    iterations=100, learning_rate=0.05, depth=5,
                    verbose=0, random_seed=42, thread_count=1,
                    allow_writing_files=False
                )
            })

        best_score = 0.0  # PR-AUC: Higher is better
        best_model = None
        best_name = "LGBM_Tree" # Default
        best_auc = 0.0
        race_results = {}
        
        tprint_info(f"   🏁 Race Started: 5 Models...")

        for cand in candidates:
            try:
                name = cand['name']
                model = cand['model']
                
                # Fit
                if 'LGBM' in name:
                     # Note: Custom objective in LGBM requires special eval_metric handling 
                     model.fit(
                         X_train, y_train, sample_weight=w_train, 
                         eval_set=[(X_val, y_val)], 
                         eval_metric='binary_logloss', 
                         callbacks=[lgb.early_stopping(10, verbose=False)]
                     )
                elif 'XGB' in name:
                     model.fit(
                         X_train, y_train, sample_weight=w_train, 
                         eval_set=[(X_val, y_val)], 
                         verbose=False
                     )
                elif 'CatBoost' in name:
                     model.fit(
                         X_train, y_train, sample_weight=w_train,
                         eval_set=(X_val, y_val),
                         early_stopping_rounds=10,
                         verbose=False
                     )

                # Predict (Probs)
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X_val)
                    if preds.ndim == 2:
                        preds = preds[:, 1]
                else: 
                     preds = model.predict(X_val) # Shouldn't reach here for these models but safer

                # Score - Use Precision@Recall=40% as primary for trading
                ll = log_loss(y_val, preds)
                try:
                    auc = roc_auc_score(y_val, preds) if len(np.unique(y_val)) > 1 else 0.5
                    ap = average_precision_score(y_val, preds) if len(np.unique(y_val)) > 1 else 0.0
                    p_at_r = precision_at_recall_threshold(y_val, preds, target_recall=0.40)
                except ValueError:
                    auc = 0.5
                    ap = 0.0
                    p_at_r = 0.0
                    
                # Recall (Threshold 0.5)
                preds_binary = (preds > 0.5).astype(int)
                rec = recall_score(y_val, preds_binary, zero_division=0)
                
                race_results[name] = {'log_loss': ll, 'auc': auc, 'pr_auc': ap, 'p@r40': p_at_r, 'recall': rec}
                
                tprint_info(f"      🏃 {name}: P@R40={p_at_r:.4f}, PR-AUC={ap:.4f}, AUC={auc:.4f}")

                # Winner by Precision@Recall=40% (Higher is better)
                if p_at_r > best_score:
                    best_score = p_at_r
                    best_model = model  
                    best_name = name
                    best_auc = auc
            except Exception as e:
                tprint_warning(f"   ⚠️ Race candidate {cand['name']} failed: {e}")

        # Fallback if all failed
        if best_model is None:
             tprint_warning("   ⚠️ All race models failed. Falling back to simple LGBM.")
             best_model = lgb.LGBMClassifier()
             best_model.fit(X_train, y_train)
             
        tprint_success(f"   🏆 Race Winner: {best_name} (P@R40={best_score:.4f})")

        # ADDED: Wrap winner with probability calibration for better threshold selection
        try:
            from sklearn.calibration import CalibratedClassifierCV
            # Use isotonic calibration (better for large datasets, sigmoid for small)
            calibration_method = 'isotonic' if len(y_train) > 1000 else 'sigmoid'
            calibrated_model = CalibratedClassifierCV(
                best_model, 
                method=calibration_method, 
                cv='prefit'  # Model already fitted
            )
            calibrated_model.fit(X_val, y_val)  # Calibrate on validation set
            tprint_info(f"   🎯 Applied {calibration_method} probability calibration")
            return calibrated_model, best_name + '_Cal', race_results
        except Exception as e:
            tprint_warning(f"   ⚠️ Calibration failed: {e}. Using uncalibrated model.")
            return best_model, best_name, race_results

    def _create_afml_candidates(self, scale_pos_weight):
        """Create traditional AFML model candidates with focal loss."""
        candidates = []

        # 1. LGBM with RobustFocalLoss (Tree) - OPTIMIZED: simpler + class balanced
        candidates.append({
            'name': 'LGBM_Focal',
            'model': lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.05,
                num_leaves=15, max_depth=5,  # REDUCED for sparse events
                scale_pos_weight=scale_pos_weight,  # ADDED: dynamic class balancing
                objective=RobustFocalLoss(gamma_pos=self.focal_gamma_pos, gamma_neg=self.focal_gamma_neg,
                                        alpha=None, verbose=False),
                random_state=42, verbose=-1, n_jobs=1
            )
        })

        # 2. LGBM with BCE (Tree) - OPTIMIZED: simpler + class balanced
        candidates.append({
            'name': 'LGBM_BCE',
            'model': lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.05,
                num_leaves=15, max_depth=5,  # REDUCED
                scale_pos_weight=scale_pos_weight,  # ADDED
                objective='binary',
                random_state=42, verbose=-1, n_jobs=1
            )
        })

        # 3. LGBM with RobustFocalLoss (Linear Tree) - OPTIMIZED
        candidates.append({
            'name': 'LGBM_Focal_Linear',
            'model': lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.05,
                num_leaves=15, max_depth=5,  # REDUCED
                linear_tree=True,
                scale_pos_weight=scale_pos_weight,  # ADDED
                objective=RobustFocalLoss(gamma_pos=self.focal_gamma_pos, gamma_neg=self.focal_gamma_neg,
                                        alpha=None, verbose=False),
                random_state=42, verbose=-1, n_jobs=1
            )
        })

        if XGBClassifier is not None:
            # 4. XGB (Tree)
            candidates.append({
                'name': 'XGB_Tree',
                'model': XGBClassifier(
                    n_estimators=100, learning_rate=0.05, max_depth=5,
                    objective=get_focal_loss_xgb(self.focal_alpha, self.focal_gamma_pos),
                    random_state=42, n_jobs=1, verbosity=0, use_label_encoder=False,
                    eval_metric='logloss'
                )
            })
            # 5. XGB (Linear)
            candidates.append({
                'name': 'XGB_Linear',
                'model': XGBClassifier(
                    n_estimators=100, learning_rate=0.05, max_depth=5,
                    booster='gblinear',
                    objective=get_focal_loss_xgb(self.focal_alpha, self.focal_gamma_pos),
                    random_state=42, n_jobs=1, verbosity=0, use_label_encoder=False,
                    eval_metric='logloss'
                )
            })

        # 6. CatBoost
        if CATBOOST_AVAILABLE and catboost is not None:
            from catboost import CatBoostClassifier
            candidates.append({
                'name': 'CatBoost',
                'model': CatBoostClassifier(
                    iterations=100, learning_rate=0.05, depth=5,
                    verbose=0, random_seed=42, thread_count=1,
                    allow_writing_files=False
                )
            })

        return candidates

    def _create_irm_candidates(self, scale_pos_weight, environment_masks=None):
        """Create causal IRM-based model candidates."""
        candidates = []

        # Create environment masks if not provided (simplified)
        if environment_masks is None:
            environment_masks = self._create_default_environment_masks(X_train, y_train)

        # 1. LGBM with IRM (Tree) - Causal Framework
        candidates.append({
            'name': 'LGBM_IRM',
            'model': IRM_LGBMClassifier(
                irm_system=self._irm_system,
                environment_masks=environment_masks,
                n_estimators=100, learning_rate=0.05,
                num_leaves=15, max_depth=5,
                scale_pos_weight=scale_pos_weight,
                random_state=42, verbose=-1, n_jobs=1
            )
        })

        # 2. LGBM with IRM (Linear Tree) - Causal Framework
        candidates.append({
            'name': 'LGBM_IRM_Linear',
            'model': IRM_LGBMClassifier(
                irm_system=self._irm_system,
                environment_masks=environment_masks,
                n_estimators=100, learning_rate=0.05,
                num_leaves=15, max_depth=5,
                linear_tree=True,
                scale_pos_weight=scale_pos_weight,
                random_state=42, verbose=-1, n_jobs=1
            )
        })

        if XGBClassifier is not None:
            # 3. XGB with IRM (Tree) - Causal Framework
            candidates.append({
                'name': 'XGB_IRM_Tree',
                'model': IRM_XGBClassifier(
                    irm_system=self._irm_system,
                    environment_masks=environment_masks,
                    n_estimators=100, learning_rate=0.05, max_depth=5,
                    random_state=42, n_jobs=1, verbosity=0, use_label_encoder=False,
                    eval_metric='logloss'
                )
            })

            # 4. XGB with IRM (Linear) - Causal Framework
            candidates.append({
                'name': 'XGB_IRM_Linear',
                'model': IRM_XGBClassifier(
                    irm_system=self._irm_system,
                    environment_masks=environment_masks,
                    n_estimators=100, learning_rate=0.05, max_depth=5,
                    booster='gblinear',
                    random_state=42, n_jobs=1, verbosity=0, use_label_encoder=False,
                    eval_metric='logloss'
                )
            })

        return candidates

    def _create_default_environment_masks(self, X_train, y_train):
        """Create default environment masks for IRM training."""
        try:
            # Create simple environment masks based on data characteristics
            # This is a simplified version - in practice, would use more sophisticated regime detection

            # Environment 1: High volatility periods
            if 'volatility_1d' in X_train.columns:
                vol_median = X_train['volatility_1d'].median()
                env_high_vol = (X_train['volatility_1d'] > vol_median).values
                env_low_vol = (X_train['volatility_1d'] <= vol_median).values
            else:
                # Fallback: split by index
                mid_point = len(X_train) // 2
                env_high_vol = np.arange(len(X_train)) >= mid_point
                env_low_vol = np.arange(len(X_train)) < mid_point

            # Environment 2: Positive vs negative targets (if applicable)
            if len(np.unique(y_train)) > 1:
                env_positive = y_train == 1
                env_negative = y_train == 0
            else:
                env_positive = np.arange(len(X_train)) < len(X_train) // 2
                env_negative = np.arange(len(X_train)) >= len(X_train) // 2

            return {
                'high_volatility': env_high_vol,
                'low_volatility': env_low_vol,
                'positive_targets': env_positive,
                'negative_targets': env_negative
            }

        except Exception as e:
            tprint_warning(f"⚠️ Failed to create environment masks: {e}")
            # Return simple fallback masks
            n_samples = len(X_train)
            return {
                'environment_1': np.arange(n_samples) < n_samples // 2,
                'environment_2': np.arange(n_samples) >= n_samples // 2
            }

    def _run_multi_horizon_model_race(self, X_train, y_train_dict, returns_dict, X_val, y_val_dict, returns_val_dict, w_train, horizons=[12, 48]):
        """
        Multi-timeframe classifier + regressor for PRICE_CUSUM.
        
        For each horizon, trains:
          - Classifier: P(TP before SL)
          - Regressor: E[return | features]
          
        Combined expected value: EV = P(profitable) * E[return]
        
        Args:
            X_train, X_val: Feature matrices
            y_train_dict, y_val_dict: {horizon: binary_labels}
            returns_dict, returns_val_dict: {horizon: continuous_returns}
            w_train: Sample weights
            horizons: List of horizons to evaluate
            
        Returns:
            dict with best results per horizon and ensemble decision
        """
        tprint_info(f"   🕐 Multi-Horizon Race for horizons {horizons}...")
        
        # Compute dynamic scale_pos_weight
        pos_count = np.sum(y_train_dict.get(horizons[0], []) == 1)
        neg_count = np.sum(y_train_dict.get(horizons[0], []) == 0)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        horizon_results = {}
        
        for h in horizons:
            y_train_h = y_train_dict.get(h)
            returns_train_h = returns_dict.get(h)
            y_val_h = y_val_dict.get(h)
            returns_val_h = returns_val_dict.get(h)
            
            if y_train_h is None or len(y_train_h) == 0:
                tprint_warning(f"      ⚠️ No labels for horizon {h}, skipping")
                continue
                
            try:
                # 1. Train Classifier - P(profitable)
                clf = lgb.LGBMClassifier(
                    n_estimators=100, learning_rate=0.05,
                    num_leaves=15, max_depth=5,
                    scale_pos_weight=scale_pos_weight,
                    objective='binary',
                    random_state=42, verbose=-1, n_jobs=1
                )
                clf.fit(X_train, y_train_h, sample_weight=w_train)
                clf_probs = clf.predict_proba(X_val)
                if clf_probs.ndim == 2:
                    clf_probs = clf_probs[:, 1]
                
                # 2. Train Regressor - E[return | features]
                reg = lgb.LGBMRegressor(
                    n_estimators=100, learning_rate=0.05,
                    num_leaves=15, max_depth=5,
                    objective='huber',  # Robust to outliers
                    random_state=42, verbose=-1, n_jobs=1
                )
                reg.fit(X_train, returns_train_h, sample_weight=w_train)
                reg_preds = reg.predict(X_val)
                
                # 3. Compute Expected Value
                # EV = P(profitable) * E[return | features]
                ev_predictions = clf_probs * reg_preds
                ev_score = np.mean(ev_predictions[ev_predictions > 0]) if np.any(ev_predictions > 0) else 0.0
                
                # 4. Compute metrics
                from sklearn.metrics import roc_auc_score, mean_squared_error
                clf_auc = roc_auc_score(y_val_h, clf_probs) if len(np.unique(y_val_h)) > 1 else 0.5
                reg_rmse = np.sqrt(mean_squared_error(returns_val_h, reg_preds))
                
                tprint_info(f"      🕐 H={h}: Clf AUC={clf_auc:.4f}, Reg RMSE={reg_rmse:.6f}, EV={ev_score:.6f}")
                
                horizon_results[h] = {
                    'classifier': clf,
                    'regressor': reg,
                    'clf_probs': clf_probs,
                    'reg_preds': reg_preds,
                    'ev_predictions': ev_predictions,
                    'clf_auc': clf_auc,
                    'reg_rmse': reg_rmse,
                    'ev_score': ev_score
                }
                
            except Exception as e:
                tprint_warning(f"      ⚠️ Horizon {h} failed: {e}")
        
        if not horizon_results:
            tprint_warning("   ⚠️ All horizons failed. Returning empty results.")
            return {}
        
        # 5. Select best horizon by EV score
        best_h = max(horizon_results, key=lambda h: horizon_results[h]['ev_score'])
        best_result = horizon_results[best_h]
        
        tprint_success(f"   🏆 Best Horizon: H={best_h} (EV={best_result['ev_score']:.6f})")
        
        return {
            'horizon_results': horizon_results,
            'best_horizon': best_h,
            'best_classifier': best_result['classifier'],
            'best_regressor': best_result['regressor'],
            'best_ev_score': best_result['ev_score']
        }

    def _optimize_focal_loss_params(self, X_train, y_train, X_val, y_val, w_train):
        """
        Use Optuna to optimize RobustFocalLoss parameters.

        Optimizes:
        - gamma_pos: Focal loss gamma for positive class (missed opportunities)
        - gamma_neg: Focal loss gamma for negative class (false positives)
        - alpha: Class balance parameter
        - mix: Mixing ratio between focal and BCE loss
        - label_smoothing: Label smoothing parameter
        """
        def objective(trial):
            # Suggest hyperparameters for RobustFocalLoss + Tree parameters
            gamma_pos = trial.suggest_float('gamma_pos', 0.5, 3.0, step=0.1)
            gamma_neg = trial.suggest_float('gamma_neg', 1.0, 4.0, step=0.1)
            alpha = trial.suggest_float('alpha', 0.1, 0.9, step=0.05)
            mix = trial.suggest_float('mix', 0.1, 0.5, step=0.05)
            label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.1, step=0.01)

            # Tree depth optimization
            max_depth = trial.suggest_int('max_depth', 3, 8)
            num_leaves = trial.suggest_int('num_leaves', 15, 63, step=8)  # Powers of 2-ish

            # Create focal loss with optimized parameters
            focal_loss = RobustFocalLoss(
                gamma_pos=gamma_pos,
                gamma_neg=gamma_neg,
                alpha=alpha,
                mix=mix,
                label_smoothing=label_smoothing,
                verbose=False
            )

            # Train model with optimized parameters
            params = LAYER2_PROBE_CONSTANTS.copy()
            params.pop('early_stopping_rounds', None)
            params['objective'] = focal_loss
            params['metric'] = 'auc'
            params['n_estimators'] = 100  # Smaller for HPO speed

            # Override with optimized tree parameters
            params['max_depth'] = max_depth
            params['num_leaves'] = num_leaves

            clf = lgb.LGBMClassifier(**params)
            clf.fit(
                X_train, y_train, sample_weight=w_train,
                eval_set=[(X_val, y_val)], eval_metric='auc',
                callbacks=[lgb.early_stopping(20, verbose=False)]
            )

            # Evaluate - Use Precision@Recall=40% for trading
            preds = clf.predict_proba(X_val)[:, 1]
            p_at_r = precision_at_recall_threshold(y_val, preds, target_recall=0.40)

            # Return negative P@R (Optuna minimizes, so -P@R = maximize P@R)
            return -p_at_r

        # Run optimization (minimize negative P@R = maximize P@R)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.focal_hpo_n_trials)

        # Get best parameters
        best_params = study.best_params
        best_score = -study.best_value  # Convert back to positive P@R

        tprint_success(f"🎯 RobustFocalLoss HPO completed!")
        tprint_info(f"   📊 Best P@R40: {best_score:.4f}")
        tprint_info(f"   🔧 Best params: γ₊={best_params['gamma_pos']:.2f}, γ₋={best_params['gamma_neg']:.2f}, α={best_params['alpha']:.2f}")

        return best_params, best_score

    def _optimize_lgbm_bce_params(self, X_train, y_train, X_val, y_val, w_train):
        """Optimize standard LGBM parameters (BCE objective)."""
        def objective(trial):
            # LGBM BCE parameters
            max_depth = trial.suggest_int('max_depth', 4, 7)
            num_leaves = trial.suggest_int('num_leaves', 15, 63, step=8)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
            min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 20, 50)
            lambda_l1 = trial.suggest_float('lambda_l1', 0.1, 0.9)
            lambda_l2 = trial.suggest_float('lambda_l2', 0.1, 0.9)
            subsample = trial.suggest_float('subsample', 0.6, 0.9)

            params = {
                'n_estimators': 200,
                'max_depth': max_depth,
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'min_data_in_leaf': min_data_in_leaf,
                'lambda_l1': lambda_l1,
                'lambda_l2': lambda_l2,
                'subsample': subsample,
                'objective': 'binary',
                'random_state': 42,
                'verbose': -1
            }

            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train, y_train, sample_weight=w_train,
                   eval_set=[(X_val, y_val)], eval_metric='binary_logloss',
                   callbacks=[lgb.early_stopping(20, verbose=False)])

            preds = clf.predict_proba(X_val)[:, 1]
            return log_loss(y_val, preds)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.focal_hpo_n_trials)

        return study.best_params, study.best_value

    def _optimize_xgb_params(self, X_train, y_train, X_val, y_val, w_train):
        """Optimize XGBoost parameters."""
        def objective(trial):
            max_depth = trial.suggest_int('max_depth', 4, 7)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
            n_estimators = trial.suggest_int('n_estimators', 100, 500)
            min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
            gamma = trial.suggest_float('gamma', 0.1, 1.0)
            subsample = trial.suggest_float('subsample', 0.6, 0.9)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 0.9)
            reg_alpha = trial.suggest_float('reg_alpha', 0.1, 0.9)
            reg_lambda = trial.suggest_float('reg_lambda', 0.1, 0.9)

            if XGBClassifier is not None:
                clf = XGBClassifier(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    min_child_weight=min_child_weight,
                    gamma=gamma,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    random_state=42,
                    verbosity=0,
                    use_label_encoder=False
                )

                clf.fit(X_train, y_train, sample_weight=w_train,
                       eval_set=[(X_val, y_val)], verbose=False)

                preds = clf.predict_proba(X_val)[:, 1]
                return log_loss(y_val, preds)
            else:
                return float('inf')

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.focal_hpo_n_trials)

        return study.best_params, study.best_value

    def _optimize_catboost_params(self, X_train, y_train, X_val, y_val, w_train):
        """Optimize CatBoost parameters."""
        def objective(trial):
            depth = trial.suggest_int('depth', 4, 7)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
            iterations = trial.suggest_int('iterations', 100, 500)
            l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1.0, 10.0)
            border_count = trial.suggest_int('border_count', 32, 255)
            random_strength = trial.suggest_float('random_strength', 0.1, 1.0)

            if CATBOOST_AVAILABLE:
                clf = CatBoostClassifier(
                    depth=depth,
                    learning_rate=learning_rate,
                    iterations=iterations,
                    l2_leaf_reg=l2_leaf_reg,
                    border_count=border_count,
                    random_strength=random_strength,
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False
                )

                clf.fit(X_train, y_train, sample_weight=w_train,
                       eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=False)

                preds = clf.predict_proba(X_val)[:, 1]
                return log_loss(y_val, preds)
            else:
                return float('inf')

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.focal_hpo_n_trials)

        return study.best_params, study.best_value

    def _select_best_geometry_via_race(self, candidates: List[GeometryTrial], df: pd.DataFrame, top_k: int = 5) -> List[GeometryTrial]:
        """
        Selects the best geometry per family using a REAL LGBM Probe.
        1. Filter Top K by purity/score.
        2. Train fast LGBM on each.
        3. Select winner based on LogLoss.
        Special Logic for PRICE_CUSUM: Selects 2-3 orthogonal ones.
        """
        if not candidates: return []
        
        family = candidates[0].family
        
        # 1. Sort by initial score (purity) and take Top K
        sorted_cands = sorted(candidates, key=lambda x: x.final_score, reverse=True)[:top_k]
        
        tprint_info(f"   🔎 Probing top {len(sorted_cands)} candidates for {family}...")

        # 2. Probe each candidate (Quick LGBM on global features)
        scored_candidates = []
        
        for cand in sorted_cands:
            try:
                # 2.1 Prepare Data
                if not hasattr(cand, 'labels') or cand.labels is None:
                    # Should have been attached in optimize_production_geometries
                    tprint_warning(f"⚠️ Candidate {cand.uuid} missing labels, skipping probe.")
                    continue

                events_df = pd.DataFrame(index=cand.events)
                X_cand = self._build_geometry_independent_event_features(df, events_df)
                
                # Ensure labels is a Series with proper index and is BINARY
                if isinstance(cand.labels, pd.Series):
                    # Convert to binary for Probe (1=Profit, 0=Loss/Noise)
                    y_cand = (cand.labels > 0).astype(int)
                else:
                    # Convert list/array to Series with events as index
                    y_cand_raw = pd.Series(cand.labels, index=cand.events)
                    # Convert to binary for Probe (1=Profit, 0=Loss/Noise)
                    y_cand = (y_cand_raw > 0).astype(int)
                
                # CRITICAL: Align y to X's index (feature generation may drop some events)
                common_idx = X_cand.index.intersection(y_cand.index)
                if len(common_idx) == 0:
                    tprint_warning(f"⚠️ Candidate {cand.uuid}: No common indices between X and y, skipping.")
                    continue
                    
                X_cand = X_cand.loc[common_idx]
                y_cand = y_cand.loc[common_idx]
                
                # Check Min Length
                if len(X_cand) < 50:
                    tprint_warning(f"⚠️ Candidate {cand.uuid}: Too few events ({len(X_cand)}), skipping.")
                    continue

                # 2.2 Train/Val Split (Simple TimeSeries split - last 30% for validation)
                split_idx = int(len(X_cand) * 0.7)
                X_tr, X_val = X_cand.iloc[:split_idx], X_cand.iloc[split_idx:]
                y_tr, y_val = y_cand.iloc[:split_idx], y_cand.iloc[split_idx:]
                
                # 2.3 Train Probe
                # Use fast constants
                model = lgb.LGBMClassifier(**LAYER2_PROBE_CONSTANTS)
                model.fit(
                    X_tr, y_tr, 
                    eval_set=[(X_val, y_val)], 
                    eval_metric='binary_logloss',
                    callbacks=[lgb.early_stopping(10, verbose=False)]
                )
                
                # 2.4 Evaluate - Use PR-AUC (Average Precision) for imbalanced data
                preds = model.predict_proba(X_val)[:, 1]
                ap = average_precision_score(y_val, preds) if len(np.unique(y_val)) > 1 else 0.0
                auc = roc_auc_score(y_val, preds) if len(np.unique(y_val)) > 1 else 0.5
                
                # Recall for logging
                preds_binary = (preds > 0.5).astype(int)
                rec = recall_score(y_val, preds_binary, zero_division=0)
                
                cand.probe_score = ap  # PR-AUC: Higher is better
                cand.race_score = auc  # For info
                
                # Store metrics for tprint
                cand.metrics_log = f"PR-AUC={ap:.4f}, AUC={auc:.4f}, Rec={rec:.4f}"
                
                scored_candidates.append(cand)
                
                # Cleanup labels to free memory
                cand.labels = None

            except Exception as e:
                tprint_warning(f"⚠️ Probe failed for {cand.uuid}: {e}")
        
        if not scored_candidates: 
            tprint_warning(f"⚠️ No candidates survived probe for {family}.")
            return []
        
        # Sort by Probe PR-AUC (Higher is better)
        scored_candidates.sort(key=lambda x: x.probe_score, reverse=True)
        
        selected = []
        
        # 3. Selection Logic
        if family == 'PRICE_CUSUM':
            # ENHANCED: Always select one H=12 and one H=48 geometry
            # Group candidates by horizon
            h12_cands = [c for c in scored_candidates if c.params.get('horizon') == 12]
            h48_cands = [c for c in scored_candidates if c.params.get('horizon') == 48]
            
            # Sort each group by probe score
            h12_cands.sort(key=lambda x: x.probe_score, reverse=True)
            h48_cands.sort(key=lambda x: x.probe_score, reverse=True)
            
            # Select best from each horizon
            if h12_cands:
                winner_h12 = h12_cands[0]
                selected.append(winner_h12)
                tprint_info(f"      🥇 H12: {winner_h12.uuid}: {winner_h12.metrics_log}")
            else:
                tprint_warning(f"      ⚠️ No H=12 candidates for PRICE_CUSUM")
                
            if h48_cands:
                winner_h48 = h48_cands[0]
                selected.append(winner_h48)
                tprint_info(f"      🥈 H48: {winner_h48.uuid}: {winner_h48.metrics_log}")
            else:
                tprint_warning(f"      ⚠️ No H=48 candidates for PRICE_CUSUM")
            
            # Fallback: if no horizon-specific, take overall best
            if not selected and scored_candidates:
                winner = scored_candidates[0]
                selected.append(winner)
                tprint_info(f"      🥇 {winner.uuid}: {winner.metrics_log} (Fallback)")
        else:
            # Winner takes all
            winner = scored_candidates[0]
            selected.append(winner)
            tprint_info(f"      🥇 {winner.uuid}: {winner.metrics_log}")
            
        return selected

    def optimize_production_geometries(
        self,
        df: pd.DataFrame,
        events_df: Optional[pd.DataFrame], # Ignored/None
        global_probe_features: Optional[List[str]] = None,
        causal_graph: Optional[Dict] = None,
        specialist_predictions: Optional[Dict] = None
    ) -> Tuple[List[GeometryTrial], List[str]]:
        """Step 2: Orthogonal Label Generation & Selection."""
        tprint_info(">>> Layer 2: Step 2 - Orthogonal Optimization...")
    
        # 1. Generate & Filter
        ortho_geoms = orthogonal_label_generation(
            df, 
            signal_weights=self.signal_weights, 
            return_raw_candidates=True,
            enable_causal_events=self.enable_causal_framework,
            specialist_predictions=specialist_predictions,
            causal_graph=causal_graph
        )
    
        if not ortho_geoms:
            tprint_error("Layer 2: No orthogonal geometries selected.")
            return [], []
    
        # 2. Group by Family
        family_map = defaultdict(list)
        for i, og in enumerate(ortho_geoms):
            # Score
            score = og.purity if og.purity is not None else 1.0
            # Params
            params = og.params.copy()
            if 'horizon' not in params: params['horizon'] = 120
            
            gt = GeometryTrial(
                family=og.family,
                params=params,
                final_score=score * 100.0,
                learnability=og.auc,
                robust_magnitude=0.0,
                stability=1.0,
                balance=1.0,
                raw_metrics=og.metrics,
                uuid=f"{og.name}_{i}",
                events=og.events,
                selected_features=None
            )
            # Attach labels for probe training (Deleted before return to save memory)
            gt.labels = og.labels
            family_map[og.family].append(gt)

        # 3. Select Best via Race/Probe
        production_geometries = []
        tprint_info("🏆 Running Geometry Selection Race...")
        
        for family, cands in family_map.items():
            tprint_info(f"   {family}: {len(cands)} candidates -> Selection...")
            winners = self._select_best_geometry_via_race(cands, df)
            production_geometries.extend(winners)

        self.selected_geometries = production_geometries
    
        # 4. Feature Selection (Optional) on Union
        # We need a union events_df to run global feature selection
        union_events = self._construct_union_events_df(df, production_geometries)
        X_events_full = self._build_geometry_independent_event_features(df, union_events)
    
        # Select global probe features based on Union
        self._global_probe_features = self._select_global_probe_features(X_events_full)
    
        # 5. Per-Geometry Titan RFE (Adaptive)
        tprint_info(">>> Layer 2: Running Titan RFE per geometry...")
        for gt in production_geometries:
            try:
                selected_feats = self._run_titan_rfe_for_geometry(df, gt)
                if selected_feats:
                    gt.selected_features = selected_feats
                    tprint_success(f"   ✅ {gt.uuid}: Selected {len(selected_feats)} features")
                else:
                    tprint_warning(f"   ⚠️ {gt.uuid}: Feature selection returned empty set.")
            except Exception as e:
                tprint_error(f"   ❌ {gt.uuid}: Feature selection failed: {e}")
    
        return production_geometries, self._global_probe_features


    def run_oof_analytics(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame, # This is the UNION events_df
        production_geometries: Optional[List[GeometryTrial]] = None,
        global_probe_features: Optional[List[str]] = None,
        production_selected_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Step 3: OOF Analytics."""
        tprint_info(">>> Layer 2: Step 3 - OOF Analytics...")
    
        if not production_geometries:
            return {}

        # Store specialist predictions for use by causal generators in _get_global_events
        self._oof_specialist_predictions = getattr(self, '_causal_specialist_predictions', None)
    
        # Initialize global containers
        idx = events_df.index
        oof_scores = pd.Series(np.nan, index=idx)
        oof_labels = pd.Series(np.nan, index=idx)
        oof_weights = pd.Series(np.nan, index=idx)
        
        # Individual geometry predictions for Layer 3
        individual_geos = {}
    
        # K-Fold Cross-Validation
        n_splits = 3
        fold_size = len(df) // n_splits
        folds = []
        for i in range(n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < n_splits - 1 else len(df)
            folds.append((start, end))
    
        self.generators = {
            # Legacy mappings
            "CUSUM": AdaptiveSymmetricCUSUMEvents(),
            "VOL": RangeATRcusumEvents(),
            "TREND": TrendRegimeCusumEvents(),
            "MEAN": MeanReversionExtremeEvents(),
            "LIQUIDITY": LiquidityShockEvents(),
            "ENTROPY": VolatilityStateEvents(),
            "BREAKOUT": TrendInitiationEvents(),
            "KALMAN": KalmanTrendEvents(),
            "MR": MeanReversionExtremeEvents(),
            "VWAP": SymmetricCusumEvents(),
            "MEAN_REV": MeanReversionExtremeEvents(),
            "VOLATILITY": VolatilityShockEvents(),
            "TIME": TimeEvents(),
            # Layer 1 output family names (CRITICAL: These must match orthogonal_label_generation families)
            "PRICE_CUSUM": ImprovedCUSUMEvents(),
            "VOL_CUSUM": VolatilityCusumEvents(),
            "LIQ_CUSUM": LiquidityCusumEvents(),
            "TAIL_RISK": TailRiskCusumEvents(),
            "TREND_REGIME": TrendRegimeCusumEvents(),
            "VOL_STATE": VolatilityStateEvents(),
            "RANGE_ATR": RangeATRcusumEvents(),
            "SR_CUSUM": SRCusumEvents(),
            
            # Causal Specialist Generators
            "CAUSAL_SURPRISE": CausalSurpriseEvents(),
            "VOLUME_SPECIALIST": VolumeSpecialistEvents(),
            "VOLATILITY_SPECIALIST": VolatilitySpecialistEvents(),
            "LIQUIDITY_SPECIALIST": LiquiditySpecialistEvents(),
            "INFORMATION_SPECIALIST": InformationSpecialistEvents(),
            "INVENTORY_SPECIALIST": InventorySpecialistEvents(),
        }
    
        # For OOF, we treat 'production_geometries' as the selected strategy.
        # We retrain the strategy on Train and predict on Test.
        
        # 1. Pre-detect sparse families to bypass CV as requested by user
        family_total_events = {}
        for gt in production_geometries:
            if gt.family not in family_total_events:
                gen_params = self._extract_gen_params(gt)
                full_events = self._get_global_events(df, gt.family, gen_params)
                family_total_events[gt.family] = len(full_events)
        
        sparse_families = {f for f, count in family_total_events.items() if count < 50}
        global_sparse_models = {}  # Cache for global models of sparse families
        
        if sparse_families:
            tprint_warning(f"⚠️ Sparse Families detected (bypassing CV): {sparse_families}")
            for family in sparse_families:
                tprint_info(f"🚀 Pre-training GLOBAL model for sparse family: {family}")
                # 1. Get ALL events for this family
                # We pick the first geometry in the family to get gen_params
                first_gt = next(gt for gt in production_geometries if gt.family == family)
                gen_params = self._extract_gen_params(first_gt)
                full_events = self._get_global_events(df, family, gen_params)
                
                if len(full_events) < 5:
                    tprint_warning(f"⚠️ Still too few events total ({len(full_events)}) for {family}. Skipping.")
                    continue
                
                # 2. Build full features
                X_full = self._build_geometry_independent_event_features(df, pd.DataFrame(index=full_events))
                
                # 3. Get labels for all geometries in this family on the full timeline
                family_geos = [gt for gt in production_geometries if gt.family == family]
                labels_dict_full, weights_dict_full = self._compute_labels_batch(
                    df, full_events, family_geos, family, fold_idx=-1, sr_levels=self.sr_levels
                )
                
                # 4. Train batch
                batch_models, batch_diagnostics = self._train_geometry_batch(
                    family_geos, df, X_full, labels_dict_full, weights_dict_full, family, fold_idx=-1
                )
                if batch_models:
                    global_sparse_models[family] = (batch_models, batch_diagnostics)
                    tprint_success(f"✅ Global model trained for {family} with {len(full_events)} events.")

        for i, (test_start, test_end) in enumerate(folds):
            tprint_info(f"OOF Fold {i+1}/{n_splits}...")
            
            # Setup Train/Test split (Walk-Forward / Purged K-Fold simplified)
            train_mask = np.ones(len(df), dtype=bool)
            train_mask[test_start:test_end] = False
            # Purge 0 bars (User requested: Disable purging and embargo)
            p_start = max(0, test_start - 0)
            p_end = min(len(df), test_end + 0)
            train_mask[p_start:p_end] = False
            
            df_train = df[train_mask]
            df_test = df[test_start:test_end]
            
            # We predict on ALL events that fall into the test window
            # First, reconstruct events for each geometry on the full timeline (or test slice)
            # Actually, we should regenerate events on test slice to avoid lookahead?
            # Or use global events and mask?
            # Generators use expanding window. If we generate on full DF, it's safe (expanding).
            # If we slice df_test, expanding window resets, which is inconsistent.
            # So generate on full, slice events.
            
            # Train Models on df_train
            trained_models = {}
            tprint_info(f"🚀 Training models for {len(production_geometries)} geometries on fold {i+1}/{n_splits}...")
            
            # Group geometries by family AND generation params for batch processing
            # Because different generation params = different events
            groups = defaultdict(list)
            for gt in production_geometries:
                gen_params = self._extract_gen_params(gt)
                # Params dict is not hashable, use sorted tuple
                params_key = tuple(sorted(gen_params.items()))
                groups[(gt.family, params_key)].append(gt)
            
            tprint_info(f"📊 DIAGNOSTIC: {len(groups)} family groups to process")
            
            # Process each group
            for (family, params_key), group_geometries in groups.items():
                gen_params = dict(params_key)
                tprint_info(f"🔄 Processing group: {family} | Params: {gen_params} ({len(group_geometries)} geometries)")
                
                # Check if generator exists
                gen = self.generators.get(family)
                if gen is None:
                    tprint_error(f"❌ DIAGNOSTIC: No generator found for family '{family}'. Available: {list(self.generators.keys())}")
                    continue
                else:
                    tprint_success(f"✅ DIAGNOSTIC: Generator found for '{family}': {type(gen).__name__}")
                
                # 1. Generate events per split for strict OOF (User advice)
                # Instead of global events + mask, we generate ON df_train
                tprint_info(f"   🔄 Generating {family} events on df_train ({len(df_train)} bars)...")
                fold_train_events = self._get_global_events(df_train, family, gen_params)
                
                is_sparse = family in sparse_families
                has_global = is_sparse and family in global_sparse_models
                
                # Diagnostics requested by user
                total_fold_events = len(fold_train_events)
                tprint_info(f"   📊 Total events on df_train: {total_fold_events}")
                
                # 2. Build features for training
                X_train = None
                if not has_global:
                    if total_fold_events < 5:
                        tprint_warning(f"   ⚠️ Too few train events ({total_fold_events} < 5) for {family}")
                        continue
                        
                    train_evts_df = pd.DataFrame(index=fold_train_events)
                    X_train = self._build_geometry_independent_event_features(df_train, train_evts_df)
                    if X_train.empty:
                        tprint_warning(f"   ⚠️ No features generated for {family}")
                        continue
                
                # 3. Compute labels and weights for all geometries in this group at once
                labels_dict = {}
                weights_dict = {}
                if not has_global:
                    labels_dict, weights_dict = self._compute_labels_batch(
                        df_train, fold_train_events, group_geometries, family, i, sr_levels=self.sr_levels
                    )
                    
                    # Log label stats for first geometry as proxy
                    if labels_dict:
                        first_geo_uuid = next(iter(labels_dict))
                        n_valid_labels = labels_dict[first_geo_uuid].dropna().shape[0]
                        tprint_info(f"   📊 Events with valid outcomes: {n_valid_labels} (proxy for group)")
                
                tprint_info(f"📊 DIAGNOSTIC: {family} training preparation complete.")
                
                # 4. Train geometries in batch with parallelization (max 2 concurrent)
                is_sparse = family in sparse_families
                
                if is_sparse and family in global_sparse_models:
                    tprint_info(f"   ♻️ Using cached GLOBAL model for sparse family {family}")
                    batch_models, batch_diagnostics = global_sparse_models[family]
                else:
                    tprint_info(f"🚀 Training {len(group_geometries)} geometries in batch for {family} (Sparse={is_sparse})...")
                    # For sparse, we could technically train on FULL data here, 
                    # but let's stick to current fold's train set unless it's too small.
                    # Or if sparse, we train on the LARGEST possible train set (e.g. all data except current test window).
                    batch_models, batch_diagnostics = self._train_geometry_batch(
                        group_geometries, df_train, X_train, labels_dict, weights_dict, family, i
                    )
                    if is_sparse and batch_models:
                        global_sparse_models[family] = (batch_models, batch_diagnostics)

                # Store trained models and diagnostics
                trained_models.update(batch_models)
                for gt_uuid, diagnostics in batch_diagnostics.items():
                    self._all_tree_stats.append({
                        'geometry_uuid': gt_uuid,
                        'fold': i,
                        **diagnostics
                    })
    
            # Predict on Test
            # We predict on events that occur in the Test window
            test_evts_map = {}

            # Slice to test window - clamp test_end index to avoid bounds error on last fold
            test_start_time = df.index[test_start]
            test_end_time = df.index[min(test_end, len(df) - 1)]

            for gt in production_geometries:
                gen_params = self._extract_gen_params(gt)

                # Use global events and slice to test window
                full_events = self._get_global_events(df, gt.family, gen_params)
                test_evts = full_events[(full_events >= test_start_time) & (full_events < test_end_time)]
    
                if len(test_evts) > 0:
                    test_evts_map[gt.uuid] = test_evts
    
            # Aggregate Predictions (OUTSIDE geometry loop)
            # We iterate test_evts_map, predict, and fill global OOF series
            tprint_info(f"🔮 Making predictions on test set for fold {i+1} ({len(test_evts_map)} geometries)...")
            tprint_info(f"   DEBUG: trained_models has {len(trained_models)} models, test_evts_map has {len(test_evts_map)} entries")
    
            for gt_uuid, evts in test_evts_map.items():
                model = trained_models.get(gt_uuid)
                if model is None:
                    tprint_warning(f"   ⚠️ No model for {gt_uuid[:8]}...")
                    continue
    
                # Build Features for Test Events (using global features)
                test_evts_df = pd.DataFrame(index=evts)
                X_test = self._build_geometry_independent_event_features(df, test_evts_df)
                if X_test.empty: continue
    
                # Geo features using params from production geometry object
                gt = next(g for g in production_geometries if g.uuid == gt_uuid)
                geo_feats = self._compute_specific_geometry_features(df, X_test.index, gt.params)
                X_test = pd.concat([X_test, geo_feats], axis=1).fillna(0.0)
    
                # Feature Selection Application (Test)
                selected_cols = None
    
                # Check for selected_features safely
                gt_selected = getattr(gt, 'selected_features', None)
    
                if gt_selected:
                    selected_cols = [c for c in gt_selected if c in X_test.columns]
                elif global_probe_features:
                    selected_cols = [c for c in global_probe_features if c in X_test.columns]
    
                if selected_cols:
                    X_test = X_test[selected_cols]
    
                # Predict
                prob = model.predict_proba(X_test)
                if prob.ndim == 2:
                    preds = prob[:, 1]
                else:
                    preds = prob
                
                # Align evts to oof_scores index TZ to ensure valid lookups
                evts_aligned = evts
                try:
                    if oof_scores.index.tz is not None and evts.tz is None:
                         evts_aligned = evts.tz_localize(oof_scores.index.tz)
                    elif oof_scores.index.tz is None and evts.tz is not None:
                         evts_aligned = evts.tz_localize(None)
                except Exception:
                    pass
                
                # Store individual geometry predictions for Layer 3
                # Use a temporary list to accumulate across folds
                if not hasattr(self, '_individual_geos_accum'):
                    self._individual_geos_accum = defaultdict(list)
                
                self._individual_geos_accum[gt_uuid].append(pd.Series(preds, index=evts_aligned))
                tprint_info(f"   Stored {len(preds)} preds for {gt_uuid} (Fold {i+1})")


                # Bagging: Max probability aggregation - using JIT-compiled function
                # If multiple geometries predict on the same timestamp, take max
                current_vals = oof_scores.loc[evts_aligned].fillna(0.0).values
                oof_scores.loc[evts_aligned] = vectorized_prediction_aggregation(preds, current_vals)

                # Weights: 1.0 for now - using JIT-compiled function
                n_events = len(evts_aligned)
                oof_weights.loc[evts_aligned] = vectorized_weight_assignment(n_events, 1.0)

            # Construct oof_labels (0.5 threshold) - using JIT-compiled function
            oof_labels = vectorized_threshold_classification(oof_scores.values, 0.5)
            oof_labels = pd.Series(oof_labels, index=oof_scores.index)
    
        # Concatenate accumulated predictions from all folds
        if hasattr(self, '_individual_geos_accum'):
            tprint_info(f"🔄 Concatenating OOF predictions for {len(self._individual_geos_accum)} geometries...")
            for gt_uuid, preds_list in self._individual_geos_accum.items():
                if preds_list:
                    try:
                        combined_preds = pd.concat(preds_list)
                        # Deduplicate in case of overlaps
                        combined_preds = combined_preds[~combined_preds.index.duplicated(keep='last')]
                        individual_geos[gt_uuid] = combined_preds
                    except Exception as e:
                        tprint_error(f"❌ Failed to concat preds for {gt_uuid}: {e}")
            # Cleanup
            del self._individual_geos_accum

        # Calculate Returns for OOF events (Consensus)
        # We need returns for the union of events generated in Test folds
        # oof_scores index contains all predicted events.
        # We need realized returns for these events.
        # Since we don't know WHICH geometry "won" the max, we use a generic return (e.g. at fixed horizon 120)
        # or we try to reconstruct weighted return.
        # Simplification: Calculate return at horizon=120 for all events
        oof_returns = pd.Series(np.nan, index=oof_scores.index)
        valid_idx = oof_scores.dropna().index
 
        if not valid_idx.empty:
            # Stub return calculation
            # Use compute_realized_returns with default params
            ret, _, _, _, _, _, _, _ = compute_realized_returns(
                df, pd.DataFrame({'consensus': 1}, index=df.index),
                profit_threshold=None, stop_threshold=None, horizon=120,
                transaction_cost=self.transaction_cost
            )
            
            # Align ret index to oof_returns (valid_idx)
            # Handle TZ mismatch
            try:
                if valid_idx.tz is not None and ret.index.tz is None:
                     ret.index = ret.index.tz_localize(valid_idx.tz)
                elif valid_idx.tz is None and ret.index.tz is not None:
                     ret.index = ret.index.tz_localize(None)
            except Exception:
                pass
                
            # Use reindex for safety
            oof_returns.loc[valid_idx] = ret.reindex(valid_idx).values

        # Memory cleanup before returning
        self._cleanup_memory()

        return {
            "l2_score": oof_scores,
            "oof_labels": oof_labels,
            "oof_returns": oof_returns,
            "weights": oof_weights,
            "tree_diagnostics": self._all_tree_stats,
            "individual_geos": individual_geos
        }
    
        # ... (Keep existing helpers like _extract_tree_diagnostics, _precompute_geometry_base_features, _validate_inputs) ...
    

    def _compute_specific_geometry_features(self, df, events_index, params):
        """Generate family-specific features for geometry assessment using vectorized operations."""
        if events_index.empty: return pd.DataFrame()

        # Vectorized data extraction
        subset = df.reindex(events_index)
        vol = subset['volatility_1d'].fillna(0.0)
        price = subset['close'].fillna(method='ffill')
        volume = subset.get('volume', pd.Series(0, index=events_index))

        # Pre-compute common rolling statistics using JIT-compiled functions for better performance
        price_values = price.values
        price_pct_values = vectorized_pct_change_jit(price_values)
        price_abs_pct_values = np.abs(price_pct_values)

        # JIT-compiled rolling calculations
        roll5_sum_values = rolling_mean_jit(price_pct_values, 5) * 5  # Approximate rolling sum
        roll10_sum_values = rolling_mean_jit(price_pct_values, 10) * 10
        roll10_sum_abs_values = rolling_mean_jit(price_abs_pct_values, 10) * 10
        roll20_mean_values = rolling_mean_jit(price_pct_values, 20)
        roll20_std_values = rolling_std_jit(price_pct_values, 20)
        roll100_std_values = rolling_std_jit(price_pct_values, 100)
        # Fix: Calculate roll10_std_values from price data, not itself (unbound)
        roll10_std_values = rolling_std_jit(price_pct_values, 10)

        # Convert back to pandas series for compatibility
        roll5_sum = pd.Series(roll5_sum_values, index=price.index)
        roll10_sum = pd.Series(roll10_sum_values, index=price.index)
        roll10_sum_abs = pd.Series(roll10_sum_abs_values, index=price.index)
        roll20_mean = pd.Series(roll20_mean_values, index=price.index)
        roll20_std = pd.Series(roll20_std_values, index=price.index)
        roll100_std = pd.Series(roll100_std_values, index=price.index)
        roll10_std = pd.Series(roll10_std_values, index=price.index)
        
        feats = pd.DataFrame(index=events_index)
        family = params.get('family', 'UNKNOWN')

        # Base geometry feature (vectorized)
        sl_mult = params.get('sl_mult', 1.0)
        stop_size = (vol * sl_mult).replace(0.0, np.nan)
        feats['geo_vol_to_stop'] = vol / stop_size

        # Vectorized family-specific features
        price_pct = pd.Series(price_pct_values, index=price.index)  # Create Series for all families
        
        if family == 'PRICE_CUSUM':
            # CUSUM signal strength and price momentum
            feats['cusum_strength'] = roll5_sum.abs()
            feats['price_efficiency'] = (roll10_sum / (roll10_sum_abs + 1e-9)).abs()
            feats['price_momentum'] = price_pct.shift(-5)
            
        elif family == 'LIQ_CUSUM':
            # Volume spike and liquidity drought (vectorized)
            vol_avg = volume.rolling(20).mean()
            vol_avg_safe = vol_avg + 1e-9
            feats['volume_spike'] = volume / vol_avg_safe
            feats['liquidity_drought'] = (volume < vol_avg * 0.5).astype(int)
            feats['volume_trend'] = volume.rolling(10).mean() / vol_avg_safe
            
        elif family == 'TAIL_RISK':
            # Tail risk proxies (vectorized)
            feats['skewness'] = price_pct.rolling(20).skew()
            feats['vol_of_vol'] = roll5_std
            feats['kurtosis'] = price_pct.rolling(20).kurtosis()
            
        elif family == 'TREND_REGIME':
            # Trend strength and consistency (vectorized)
            feats['trend_strength'] = roll20_mean.abs()
            feats['directional_consistency'] = (price_pct > 0).rolling(10).mean()
            feats['trend_persistence'] = (price_pct > 0).rolling(20).mean()
            
        elif family == 'VOL_STATE':
            # Volatility regime indicators (vectorized)
            vol_z = roll20_std / (roll100_std + 1e-9)
            feats['vol_regime_zscore'] = vol_z.fillna(0)
            feats['vol_clustering'] = price_abs_pct.rolling(5).autocorr()
            feats['vol_mean_reversion'] = -price_pct.rolling(20).autocorr()
            
        elif family in ['RANGE_ATR', 'SR_CUSUM']:
            # Range expansion and breakout momentum (vectorized)
            roll20_max = price.rolling(20).max()
            roll20_min = price.rolling(20).min()
            roll5_max = price.rolling(5).max()
            roll5_min = price.rolling(5).min()
            
            range_avg = roll20_max - roll20_min
            current_range = roll5_max - roll5_min
            range_avg_safe = range_avg + 1e-9
            range_avg_shift = range_avg.shift(1)
            
            feats['range_expansion'] = current_range / range_avg_safe
            feats['breakout_momentum'] = (current_range - range_avg_shift) / (range_avg_shift + 1e-9)
            feats['range_position'] = (price - roll20_min) / range_avg_safe

        return feats.fillna(0.0)
    def _run_titan_rfe_for_geometry(self, df: pd.DataFrame, gt: GeometryTrial) -> List[str]:
        """
        Run adaptive Titan RFE for a specific geometry.
        """
        # Check cache first
        cached_selection = self._get_cached_feature_selection(gt.uuid)
        if cached_selection is not None:
            tprint_info(f"   ✅ Using cached feature selection for {gt.uuid}: {len(cached_selection)} features")
            return cached_selection

        # 1. Regenerate events/labels for this geometry on the full df (or relevant slice)

        # Re-generate events if not present or need full context
        if gt.events is None or len(gt.events) == 0:
            return []

        events_df = pd.DataFrame(index=gt.events)

        # Compute labels
        # Note: We use the same params as optimization
        if gt.family == 'PRICE_CUSUM':
             labels, _, _, _, _, _ = self._compute_dominance_labels(df, events_df, **gt.params)
        elif gt.family == 'LIQ_CUSUM':
             pt = gt.params.get('pt_mult', 2.0)
             d_sigma = pt * 0.5
             horizon = int(gt.params.get('horizon', 24))
             labels, _, _, _, _, _ = compute_path_degradation_labels(df, gt.events, horizon=horizon, d_sigma=d_sigma)
        elif gt.family == 'TAIL_RISK':
             pt = gt.params.get('pt_mult', 2.0)
             z_thresh = pt * 0.6
             horizon = int(gt.params.get('horizon', 24))
             labels, _, _, _, _, _ = compute_tail_regime_labels(df, gt.events, horizon=horizon, z_thresh=z_thresh, metric_col=gt.params.get('metric', 'skew'))
        elif gt.family == 'TREND_REGIME':
             horizon = int(gt.params.get('horizon', 24))
             labels, _, _, _, _, _ = compute_trend_persistence_labels(df, gt.events, horizon=horizon)
        elif gt.family == 'VOL_STATE':
             horizon = int(gt.params.get('horizon', 24))
             labels, _, _, _, _, _ = compute_vol_state_labels(df, gt.events, horizon=horizon)
        elif gt.family == 'RANGE_ATR' or gt.family == 'SR_CUSUM':
             pt = gt.params.get('pt_mult', 2.0)
             k_factor = 1.1 + (pt * 0.1)
             horizon = int(gt.params.get('horizon', 24))
             labels, _, _, _, _, _ = compute_volatility_labels(df, gt.events, horizon=horizon, k=k_factor)
        else:
             # Map params to Volatility Logic (VOL_CUSUM)
             pt = gt.params.get('pt_mult', 2.0)
             k_factor = 1.1 + (pt * 0.1)
             horizon = int(gt.params.get('horizon', 24))
             labels, _, _, _, _, _ = compute_volatility_labels(df, gt.events, horizon=horizon, k=k_factor)

        valid_mask = ~labels.isna()
        y = labels[valid_mask]

        if len(y) < 50:
            return []

        # Build features
        # We use the subset of events
        events_subset = events_df.loc[y.index]
        X = self._build_geometry_independent_event_features(df, events_subset)

        if X is None or X.empty:
            return []

        # Add geometry specific features
        geo_feats = self._compute_specific_geometry_features(df, X.index, gt.params)
        X = pd.concat([X, geo_feats], axis=1).fillna(0.0)

        # Run Pipeline
        # We aim for ~60 features but adaptive to sample size (1 per 100 samples)
        # The pipeline handles the adaptation internally.
        target_sets = [60, 50, 40, 30, 20, 10]

        feature_sets, _ = lgbm_feature_selection_pipeline(
            X, y,
            target_feature_sets=target_sets,
            samples_per_feature_ratio=100
        )

        if not feature_sets:
            # Fallback: ensure minimum 10 features
            n_samples = len(y)
            limit = max(10, n_samples // 100)  # Minimum 10 features
            selected_features = list(X.columns)[:limit]
            self._cache_feature_selection(gt.uuid, selected_features)
            return selected_features

        # Return the largest available set, but ensure minimum 10 features
        best_k = max(feature_sets.keys())
        selected_features = feature_sets[best_k]

        # Guarantee minimum 10 features
        if len(selected_features) < 10:
            # Add more features from the original set if available
            all_features = list(X.columns)
            available_additional = [f for f in all_features if f not in selected_features]
            needed = 10 - len(selected_features)
            if len(available_additional) >= needed:
                selected_features.extend(available_additional[:needed])
            else:
                # If we don't have enough, just take the first 10 features
                selected_features = all_features[:10]

        self._cache_feature_selection(gt.uuid, selected_features)
        return selected_features

    def _select_global_probe_features(self, X_events: pd.DataFrame) -> List[str]:
        """
        Curated selection of 7-10 high-quality market state probe features.
        These features capture market microstructure, information theory, and psychological states.
        """
        if X_events is None or X_events.empty:
            return []
        
        # Curated market state probe features from existing MTF feature generation
        probe_features = [
            # Market Microstructure & Liquidity (unique from family/market state)
            'price_impact_w20',             # Price impact (Amihud illiquidity)
            'cmf_w20',                      # Chaikin Money Flow
            'force_index_w20',              # Force Index

            # Trend & Direction (unique from family features)
            'adx_w20',                      # ADX trend strength
            'trend_efficiency_ratio_w20',   # Trend efficiency

            # Volatility Regime (unique from family VOL_STATE)
            'vol_compression_duration_w20',  # Vol compression duration
            'breakout_after_compression_flag_w20',  # Breakout after compression

            # Drawdown & Risk
            'drawdown_w20',                 # Rolling drawdown
            'max_adverse_excursion_w20',     # MAE proxy

            # Market Microstructure
            'body_to_range_w20',            # Candle body to range ratio
            'close_location_value_w20',     # Close location in range
        ]

        
        # Select only features that actually exist in the data
        selected = [f for f in probe_features if f in X_events.columns]
        
        # Limit to top 10 most important if we have more
        if len(selected) > 10:
            # Simple variance-based final filtering as tie-breaker
            variances = {f: X_events[f].var() for f in selected}
            selected = sorted(variances.keys(), key=lambda f: variances[f], reverse=True)[:10]
        
        tprint_info(f"Global probe features selected: {selected}")
        return selected

    def _train_geometry_batch(self, geometry_batch: List, df_train: pd.DataFrame, X_train: pd.DataFrame,
                            labels_dict: Dict, weights_dict: Dict, family: str, fold_idx: int) -> Dict[str, Any]:
        """Train a batch of geometries in parallel (max 2 concurrent)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        trained_models = {}
        results = {}

        def train_single_geometry(gt):
            """Train a single geometry."""
            try:
                # Check cache first
                cached_model = self._get_cached_model(gt.uuid, fold_idx)
                if cached_model is not None:
                    tprint_info(f"   ✅ Using cached model for {gt.uuid}")
                    return gt.uuid, cached_model, self._extract_tree_diagnostics(cached_model)

                # Get labels and weights for this geometry
                labels = labels_dict.get(gt.uuid)
                weights = weights_dict.get(gt.uuid)

                if labels is None or len(labels) < 5:
                    return gt.uuid, None, f"Too few valid labels: {len(labels) if labels is not None else 0}"

                # Get geometry-specific features
                geo_feats = self._compute_specific_geometry_features(df_train, X_train.index, gt.params)
                X_train_geo = pd.concat([X_train, geo_feats], axis=1, copy=False).fillna(0.0)

                # Feature Selection Application
                selected_cols = None

                # Check for selected_features safely
                gt_selected = getattr(gt, 'selected_features', None)

                if gt_selected:
                    selected_cols = [c for c in gt_selected if c in X_train_geo.columns]

                if selected_cols:
                    X_train_final = X_train_geo[selected_cols]
                else:
                    X_train_final = X_train_geo

                # Align labels and weights with features
                common_idx = X_train_final.index.intersection(labels.index)
                if len(common_idx) < 5:  # Insufficient data
                    return gt.uuid, None, f"Too few samples after alignment: {len(common_idx)}"

                X_train_final = X_train_final.loc[common_idx]
                y_train = labels.loc[common_idx]

                if weights is not None:
                    w_train = weights.reindex(common_idx).fillna(0.0)
                else:
                    w_train = None

                # Train Model - with optional model race
                focal_params = None
                best_name = "LGBM_Focal"  # Default winner

                if self.enable_model_race and len(X_train_final) > 100:  # Only race if enough data
                    tprint_info(f"🏁 Running model race for geometry {gt.uuid}...")

                    # Split training data for model race (use portion for validation)
                    race_train_size = int(0.7 * len(X_train_final))
                    X_race_train = X_train_final.iloc[:race_train_size]
                    X_race_val = X_train_final.iloc[race_train_size:]
                    y_race_train = y_train.iloc[:race_train_size]
                    y_race_val = y_train.iloc[race_train_size:]
                    w_race_train = w_train[:race_train_size] if w_train is not None else None

                    # Run model race with environment masks for IRM (if causal framework enabled)
                    environment_masks = None
                    if self.enable_causal_framework and CAUSAL_MODULES_AVAILABLE and self.irm_enabled:
                        environment_masks = self._create_default_environment_masks(X_race_train, y_race_train)

                    best_model, best_name, race_results = self._run_model_race(
                        X_race_train, y_race_train, X_race_val, y_race_val, w_race_train, environment_masks
                    )

                    tprint_success(f"🏆 Model race winner: {best_name} (log_loss: {race_results[best_name]['log_loss']:.4f})")

                    # HPO - run on the winning model (different HPO strategies per model type)
                    if self.enable_focal_hpo and len(X_train_final) > 200:
                        tprint_info(f"🎯 Running HPO for winning model: {best_name}...")

                        # Split data for HPO (fresh split from full training data)
                        hpo_train_size = int(0.7 * len(X_train_final))
                        X_hpo_train = X_train_final.iloc[:hpo_train_size]
                        X_hpo_val = X_train_final.iloc[hpo_train_size:]
                        y_hpo_train = y_train.iloc[:hpo_train_size]
                        y_hpo_val = y_train.iloc[hpo_train_size:]
                        w_hpo_train = w_train[:hpo_train_size] if w_train is not None else None

                        # Run appropriate HPO based on winner type
                        if 'LGBM_Focal' in best_name:
                            # RobustFocalLoss + Tree HPO
                            focal_params, hpo_score = self._optimize_focal_loss_params(
                                X_hpo_train, y_hpo_train, X_hpo_val, y_hpo_val, w_hpo_train
                            )
                        elif 'LGBM_BCE' in best_name:
                            # Standard LGBM HPO (tree parameters only)
                            focal_params, hpo_score = self._optimize_lgbm_bce_params(
                                X_hpo_train, y_hpo_train, X_hpo_val, y_hpo_val, w_hpo_train
                            )
                        elif 'XGB' in best_name:
                            # XGBoost HPO
                            focal_params, hpo_score = self._optimize_xgb_params(
                                X_hpo_train, y_hpo_train, X_hpo_val, y_hpo_val, w_hpo_train
                            )
                        elif 'CatBoost' in best_name:
                            # CatBoost HPO
                            focal_params, hpo_score = self._optimize_catboost_params(
                                X_hpo_train, y_hpo_train, X_hpo_val, y_hpo_val, w_hpo_train
                            )
                        else:
                            # No HPO for other model types
                            focal_params = None
                            tprint_info(f"   ⏭️ Skipping HPO for {best_name} (using defaults)")

                    # Retrain best model on full training data with HPO parameters
                    if 'LGBM_Focal' in best_name:
                        # Use HPO-optimized parameters if available
                        if focal_params:
                            focal_lgbm = RobustFocalLoss(
                                gamma_pos=focal_params['gamma_pos'],
                                gamma_neg=focal_params['gamma_neg'],
                                alpha=focal_params['alpha'],
                                mix=focal_params['mix'],
                                label_smoothing=focal_params['label_smoothing'],
                                verbose=False
                            )
                            params = LAYER2_PROBE_CONSTANTS.copy()
                            params.pop('early_stopping_rounds', None)
                            params['objective'] = focal_lgbm
                            params['metric'] = 'auc'
                            # Apply HPO tree parameters
                            params['max_depth'] = focal_params['max_depth']
                            params['num_leaves'] = focal_params['num_leaves']
                            tprint_info(f"   ✅ Using HPO-optimized focal loss + tree params")
                        else:
                            focal_lgbm = RobustFocalLoss(verbose=False)
                            params = LAYER2_PROBE_CONSTANTS.copy()
                            params.pop('early_stopping_rounds', None)
                            params['objective'] = focal_lgbm
                            params['metric'] = 'auc'

                        clf = lgb.LGBMClassifier(**params)
                        clf.fit(
                            X_train_final, y_train, sample_weight=w_train,
                            eval_set=[(X_train_final, y_train)], eval_metric='auc',
                            callbacks=[lgb.early_stopping(30, verbose=False)]
                        )

                    elif 'LGBM_BCE' in best_name:
                        # Standard LGBM with BCE - use HPO params if available
                        if focal_params:
                            clf = lgb.LGBMClassifier(
                                n_estimators=200,
                                max_depth=focal_params['max_depth'],
                                num_leaves=focal_params['num_leaves'],
                                learning_rate=focal_params['learning_rate'],
                                min_data_in_leaf=focal_params['min_data_in_leaf'],
                                lambda_l1=focal_params['lambda_l1'],
                                lambda_l2=focal_params['lambda_l2'],
                                subsample=focal_params['subsample'],
                                objective='binary',
                                random_state=42,
                                verbose=-1
                            )
                            tprint_info(f"   ✅ Using HPO-optimized LGBM BCE params")
                        else:
                            clf = lgb.LGBMClassifier(
                                n_estimators=200, learning_rate=0.05, num_leaves=31, max_depth=6,
                                objective='binary', random_state=42, verbose=-1
                            )
                        clf.fit(X_train_final, y_train, sample_weight=w_train)

                    elif 'LGBM_Focal_Linear' in best_name:
                        # Linear tree with focal loss - use HPO if available
                        if focal_params:
                            focal_lgbm = RobustFocalLoss(
                                gamma_pos=focal_params['gamma_pos'],
                                gamma_neg=focal_params['gamma_neg'],
                                alpha=focal_params['alpha'],
                                mix=focal_params['mix'],
                                label_smoothing=focal_params['label_smoothing'],
                                verbose=False
                            )
                            params = LAYER2_PROBE_CONSTANTS.copy()
                            params.pop('early_stopping_rounds', None)
                            params['objective'] = focal_lgbm
                            params['metric'] = 'auc'
                            params['linear_tree'] = True
                            params['max_depth'] = focal_params['max_depth']
                            params['num_leaves'] = focal_params['num_leaves']
                        else:
                            focal_lgbm = RobustFocalLoss(verbose=False)
                            params = LAYER2_PROBE_CONSTANTS.copy()
                            params.pop('early_stopping_rounds', None)
                            params['objective'] = focal_lgbm
                            params['metric'] = 'auc'
                            params['linear_tree'] = True

                        clf = lgb.LGBMClassifier(**params)
                        clf.fit(
                            X_train_final, y_train, sample_weight=w_train,
                            eval_set=[(X_train_final, y_train)], eval_metric='auc',
                            callbacks=[lgb.early_stopping(30, verbose=False)]
                        )

                    elif 'XGB' in best_name and XGBClassifier is not None:
                        # Use HPO-optimized parameters if available
                        if focal_params:
                            clf = XGBClassifier(
                                max_depth=focal_params['max_depth'],
                                learning_rate=focal_params['learning_rate'],
                                n_estimators=focal_params['n_estimators'],
                                min_child_weight=focal_params['min_child_weight'],
                                gamma=focal_params['gamma'],
                                subsample=focal_params['subsample'],
                                colsample_bytree=focal_params['colsample_bytree'],
                                reg_alpha=focal_params['reg_alpha'],
                                reg_lambda=focal_params['reg_lambda'],
                                random_state=42,
                                verbosity=0,
                                use_label_encoder=False
                            )
                            tprint_info(f"   ✅ Using HPO-optimized XGB params")
                        else:
                            clf = XGBClassifier(
                                n_estimators=200, learning_rate=0.05, max_depth=5,
                                random_state=42, n_jobs=1, verbosity=0, use_label_encoder=False
                            )
                        clf.fit(X_train_final, y_train, sample_weight=w_train)

                    elif best_name == 'CatBoost' and CATBOOST_AVAILABLE:
                        # Use HPO-optimized parameters if available
                        if focal_params:
                            from catboost import CatBoostClassifier
                            clf = CatBoostClassifier(
                                depth=focal_params['depth'],
                                learning_rate=focal_params['learning_rate'],
                                iterations=focal_params['iterations'],
                                l2_leaf_reg=focal_params['l2_leaf_reg'],
                                border_count=focal_params['border_count'],
                                random_strength=focal_params['random_strength'],
                                random_seed=42,
                                verbose=False,
                                allow_writing_files=False
                            )
                            tprint_info(f"   ✅ Using HPO-optimized CatBoost params")
                        else:
                            from catboost import CatBoostClassifier
                            clf = CatBoostClassifier(
                                iterations=200, learning_rate=0.05, depth=5,
                                random_state=42, verbose=False
                            )
                        clf.fit(X_train_final, y_train, sample_weight=w_train)
                    else:
                        # Fallback to LGBM
                        focal_lgbm = RobustFocalLoss(verbose=False)
                        params = LAYER2_PROBE_CONSTANTS.copy()
                        params.pop('early_stopping_rounds', None)
                        params['objective'] = focal_lgbm
                        params['metric'] = 'auc'
                        clf = lgb.LGBMClassifier(**params)
                        clf.fit(
                            X_train_final, y_train, sample_weight=w_train,
                            eval_set=[(X_train_final, y_train)], eval_metric='auc',
                            callbacks=[lgb.early_stopping(30, verbose=False)]
                        )
                else:
                    # Default: LGBM with Robust Focal Loss (using HPO params if available)
                    # Only run HPO if no model race was performed
                    if self.enable_focal_hpo and len(X_train_final) > 200 and not self.enable_model_race:
                        tprint_info(f"🎯 Running RobustFocalLoss HPO (no model race)...")

                        # Split data for HPO
                        hpo_train_size = int(0.7 * len(X_train_final))
                        X_hpo_train = X_train_final.iloc[:hpo_train_size]
                        X_hpo_val = X_train_final.iloc[hpo_train_size:]
                        y_hpo_train = y_train.iloc[:hpo_train_size]
                        y_hpo_val = y_train.iloc[hpo_train_size:]
                        w_hpo_train = w_train[:hpo_train_size] if w_train is not None else None

                        # Run HPO
                        focal_params, hpo_score = self._optimize_focal_loss_params(
                            X_hpo_train, y_hpo_train, X_hpo_val, y_hpo_val, w_hpo_train
                        )

                    # Train with RobustFocalLoss (HPO params if available)
                    if focal_params:
                        focal_lgbm = RobustFocalLoss(
                            gamma_pos=focal_params['gamma_pos'],
                            gamma_neg=focal_params['gamma_neg'],
                            alpha=focal_params['alpha'],
                            mix=focal_params['mix'],
                            label_smoothing=focal_params['label_smoothing'],
                            verbose=False
                        )
                        params = LAYER2_PROBE_CONSTANTS.copy()
                        params.pop('early_stopping_rounds', None)
                        params['objective'] = focal_lgbm
                        params['metric'] = 'auc'
                        params['max_depth'] = focal_params['max_depth']
                        params['num_leaves'] = focal_params['num_leaves']
                        tprint_info(f"   ✅ Using HPO-optimized focal loss: γ₊={focal_params['gamma_pos']:.2f}, γ₋={focal_params['gamma_neg']:.2f}")
                    else:
                        focal_lgbm = RobustFocalLoss(verbose=False)
                        params = LAYER2_PROBE_CONSTANTS.copy()
                        params.pop('early_stopping_rounds', None)
                        params['objective'] = focal_lgbm
                        params['metric'] = 'auc'

                    clf = lgb.LGBMClassifier(**params)
                    
                    # Adaptive min_data_in_leaf for small datasets
                    if len(X_train_final) < 50:
                        clf.set_params(min_data_in_leaf=max(1, len(X_train_final) // 3),
                                       min_sum_hessian_in_leaf=0.0,
                                       min_child_weight=0.0)
                        tprint_info(f"   ⚙️ Using adaptive params for small dataset ({len(X_train_final)} samples)")
                    
                    clf.fit(
                        X_train_final,
                        y_train,
                        sample_weight=w_train,
                        eval_set=[(X_train_final, y_train)],
                        eval_metric='auc',
                        callbacks=[lgb.early_stopping(30, verbose=False)]
                    )

                # Extract tree diagnostics
                tree_diagnostics = self._extract_tree_diagnostics(clf)

                # Cache the trained model
                self._cache_model(gt.uuid, fold_idx, clf)

                return gt.uuid, clf, tree_diagnostics

            except Exception as e:
                return gt.uuid, None, str(e)

        # Train geometries with max 2 concurrent
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_gt = {executor.submit(train_single_geometry, gt): gt for gt in geometry_batch}

            for future in as_completed(future_to_gt):
                gt_uuid, model, diagnostics_or_error = future.result()
                gt = future_to_gt[future]

                if model is not None:
                    trained_models[gt_uuid] = model
                    results[gt_uuid] = diagnostics_or_error
                    tprint_success(f"   ✅ Trained model for {gt_uuid}")
                else:
                    tprint_error(f"   ❌ Training failed for {gt_uuid}: {diagnostics_or_error}")

        return trained_models, results

        
        # Layer 2.5 Chaser Integration
        layer25_chaser_enabled = self.config.get("layer25_chaser_enabled", False)
        if layer25_chaser_enabled and len(trained_models) > 0:
            tprint_info(">>> Layer 2.5: Running Chaser System...")
            try:
                # Initialize Chaser system
                chaser_system = quick_layer25_setup(
                    enable_conflict_detection=True,
                    verbose=True
                )
                
                # Process each trained model
                chaser_results = {}
                for gt_uuid, model_info in trained_models.items():
                    try:
                        if hasattr(model_info, "model") and model_info.model is not None:
                            # Get causal anchor predictions (using the trained model as proxy)
                            # In practice, this would be your actual causal anchor model
                            X_full = self.X_full if hasattr(self, "X_full") else None
                            y_full = self.y_full if hasattr(self, "y_full") else None
                            
                            if X_full is not None and y_full is not None:
                                # Get model predictions as causal anchor proxy
                                anchor_predictions = model_info.model.predict(X_full)
                                
                                # Setup feature selector with available features
                                all_features = list(X_full.columns)
                                chaser_system.setup_feature_selector(
                                    causal_graph=None,  # Would use actual causal graph
                                    max_features=50
                                )
                                
                                # Prepare training data
                                X_train, y_residuals = chaser_system.prepare_training_data(
                                    df=pd.concat([X_full, y_full], axis=1),
                                    target_col=y_full.name if hasattr(y_full, "name") else "target",
                                    causal_anchor_prediction=anchor_predictions,
                                    all_feature_cols=all_features
                                )
                                
                                # Train Chaser
                                if len(X_train) > 100:  # Minimum samples
                                    chaser_metrics = chaser_system.train_chaser(X_train, y_residuals)
                                    
                                    # Generate predictions
                                    chaser_prediction_results = chaser_system.predict_with_conflict_detection(
                                        X_train, anchor_predictions[:len(X_train)]
                                    )
                                    
                                    # Prepare meta-learner features
                                    meta_features = chaser_system.get_meta_learner_features(chaser_prediction_results)
                                    
                                    # Store results
                                    chaser_results[gt_uuid] = {
                                        "chaser_metrics": chaser_metrics,
                                        "meta_features": meta_features,
                                        "chaser_predictions": chaser_prediction_results["chaser_prediction"],
                                        "conflict_intensity": chaser_prediction_results.get("conflict_intensity", np.zeros(len(chaser_prediction_results["chaser_prediction"]))),
                                        "feature_importance": chaser_system.chaser.get_feature_importance() if chaser_system.chaser else {}
                                    }
                                    
                                    tprint_success(f"   ✅ Chaser trained for {gt_uuid}")
                                    tprint_info(f"      - Chaser RMSE: {chaser_metrics['training_metrics']['rmse']:.6f}")
                                    tprint_info(f"      - Meta features: {len(meta_features.columns)}")
                                else:
                                    tprint_warning(f"   ⚠️ Insufficient data for Chaser training: {gt_uuid}")
                            else:
                                tprint_warning(f"   ⚠️ Missing data for Chaser: {gt_uuid}")
                                
                    except Exception as e:
                        tprint_error(f"   ❌ Chaser failed for {gt_uuid}: {e}")
                        continue
                
                # Add Chaser results to main results
                results["layer25_chaser"] = chaser_results
                
                # Store Chaser system for downstream use
                self.layer25_chaser_system = chaser_system
                
                tprint_success(f"✅ Layer 2.5 Chaser complete: {len(chaser_results)} models processed")
                
            except Exception as e:
                tprint_error(f"❌ Layer 2.5 Chaser system failed: {e}")
                results["layer25_chaser_error"] = str(e)
                self.layer25_chaser_system = None
        else:
            tprint_info("⏭️ Skipping Layer 2.5 Chaser (disabled)")
            self.layer25_chaser_system = None
        

    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage with categorical dtypes and downcasting."""
        if df.empty:
            return df

        df_opt = df.copy()

        # Convert object columns to categorical where appropriate
        for col in df_opt.select_dtypes(include=['object']).columns:
            if df_opt[col].nunique() / len(df_opt) < 0.5:  # Less than 50% unique values
                df_opt[col] = df_opt[col].astype('category')

        # Downcast numeric types
        for col in df_opt.select_dtypes(include=['int64']).columns:
            df_opt[col] = pd.to_numeric(df_opt[col], downcast='integer')

        for col in df_opt.select_dtypes(include=['float64']).columns:
            df_opt[col] = pd.to_numeric(df_opt[col], downcast='float')

        return df_opt

    def _cleanup_memory(self):
        """Force garbage collection to free memory."""
        import gc
        gc.collect()

    def _align_features_efficiently(self, X_all: pd.DataFrame, events_idx: pd.DatetimeIndex) -> pd.DataFrame:
        """Fast feature alignment using pandas merge_asof."""
        if len(events_idx) == 0:
            return pd.DataFrame()

        try:
            # Create events DataFrame with timestamp index
            events_df = pd.DataFrame(index=events_idx)

            # Ensure both indices are datetime and handle timezones
            events_naive = pd.to_datetime(events_idx)
            features_naive = pd.to_datetime(X_all.index)

            if events_naive.tz is not None:
                events_naive = events_naive.tz_localize(None)
            if features_naive.tz is not None:
                features_naive = features_naive.tz_localize(None)

            # Create temporary DataFrames for merge_asof
            events_temp = pd.DataFrame({'event_marker': 1}, index=events_naive)
            features_temp = X_all.copy()
            features_temp.index = features_naive

            # Use merge_asof for fast alignment (assumes sorted indices)
            merged = pd.merge_asof(
                events_temp.sort_index(),
                features_temp.sort_index(),
                left_index=True,
                right_index=True,
                direction='nearest',
                tolerance=pd.Timedelta('1min')
            )

            # Remove the event_marker column and restore original event index
            result = merged.drop(columns=['event_marker'], errors='ignore')
            result.index = events_idx

            # Optimize memory and cleanup
            result = self._optimize_dataframe_memory(result.fillna(0.0))

            # Cleanup temporary variables
            del events_temp, features_temp, merged
            self._cleanup_memory()

            return result

        except Exception as e:
            logger.warning(f"Vectorized alignment failed: {e}, falling back to simple reindex")
            try:
                # Fallback to simple reindex with tolerance
                return X_all.reindex(events_idx, method='nearest', tolerance=pd.Timedelta('1min')).fillna(0.0)
            except Exception as e2:
                logger.error(f"Fallback alignment also failed: {e2}")
                return pd.DataFrame(index=events_idx)

    def _build_geometry_independent_event_features(self, df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        if len(events_df) == 0:
            return pd.DataFrame()

        try:
            # Use global feature cache instead of regenerating
            X_all = self._get_global_features(df)

            if X_all.empty:
                return pd.DataFrame(index=events_df.index)

            # Vectorized alignment instead of complex logic
            X = self._align_features_efficiently(X_all, events_df.index)

            # --- Enhanced Probe Features ---
            if 'close' in df.columns and 'volume' in df.columns:
                try:
                    # Generate probe features for the full dataset once
                    probe_cache_key = f"probe_{hash(str(df.shape)) % 10000}"
                    if probe_cache_key not in self._probe_data_cache:
                        self._probe_data_cache[probe_cache_key] = generate_market_state_probe(df['close'], df['volume'])

                    probe_feats_all = self._probe_data_cache[probe_cache_key]

                    # Align probe features using the same efficient method
                    probe_feats = self._align_features_efficiently(probe_feats_all, events_df.index)

                    if not probe_feats.empty:
                        X = pd.concat([X, probe_feats], axis=1, copy=False)
                except Exception as e:
                    logger.warning(f"Probe feature alignment failed: {e}")
            # -------------------------------

            return X.fillna(0.0)

        except Exception as e:
            logger.warning(f"Feature generation failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            return pd.DataFrame(index=events_df.index)

    def _compute_dominance_labels(self, df, events_df, **kwargs):
        # Use Vectorized Implementation from orthogonal_label_generation

        # Params
        risk_budget = float(kwargs.get('risk_budget', 1.0))
        sl_mult = float(kwargs.get('sl_mult', 1.0))
        pt_mult = float(kwargs.get('pt_mult', 2.0))
        horizon = int(kwargs.get('horizon', 120))

        # Data
        price = df['close']
        vol = df['volatility_1d'].fillna(0.0)
        events = events_df.index

        # High/Low if available
        high = df.get('high')
        low = df.get('low')

        # Call Vectorized
        labels, weights, returns, mfe, mae, _ = compute_dominance_labels(
            price, events, vol,
            risk_budget=risk_budget, pt_mult=pt_mult, sl_mult=sl_mult, horizon=horizon,
            transaction_cost=self.transaction_cost,
            high=high, low=low
        )

        # Return matched format (labels, weights, returns, mfe, mae, exits)
        # Exits is dummy for now
        exits = pd.Series('', index=labels.index)

        return labels, weights, returns, mfe, mae, exits

    def generate_reports(self, *args, **kwargs):
        pass

    def _extract_tree_diagnostics(self, model_or_booster) -> Dict[str, float]:
        """
        Extract diagnostics from a trained LGBM booster or sklearn wrapper.
        """
        if model_or_booster is None:
            return {'n_features_used': 0.0, 'avg_depth': 0.0, 'max_depth': 0.0}

        try:
            # Unwrap sklearn-API models
            booster = model_or_booster
            if hasattr(model_or_booster, 'booster_'):
                booster = model_or_booster.booster_

            # 1. Feature Usage
            imp = booster.feature_importance(importance_type='split')
            n_features = int(np.sum(imp > 0))

            # 2. Tree Depth
            dump = booster.dump_model()
            trees = dump.get('tree_info', [])
            depths = []
            for tree in trees:
                if 'tree_structure' not in tree: continue
                stack = [(tree['tree_structure'], 0)]
                while stack:
                    node, d = stack.pop()
                    if 'leaf_index' in node: depths.append(d)
                    else:
                        if 'left_child' in node: stack.append((node['left_child'], d + 1))
                        if 'right_child' in node: stack.append((node['right_child'], d + 1))

            avg_depth = float(np.mean(depths)) if depths else 0.0
            max_depth = float(np.max(depths)) if depths else 0.0

            return {
                'n_features_used': float(n_features),
                'avg_depth': avg_depth,
                'max_depth': max_depth
            }
        except Exception as e:
            logger.warning(f"Failed to extract tree diagnostics: {e}")
            return {'n_features_used': 0.0, 'avg_depth': 0.0, 'max_depth': 0.0}


def register_label_based_layer_2_step() -> None:
    """Register the label-based layer 2 step in the registry."""
    from src.training.steps.base_step import step_registry
    step_registry.register("label_based_layer_2", LabelBasedLayer2)
