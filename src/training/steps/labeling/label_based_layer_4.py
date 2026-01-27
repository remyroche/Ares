"""Layer 4 — Triple Barrier Trailing Profit & Sizing with Entropy Bars.

Layer2 is about learnability, layer3 about relation to target (IC, calibration),
layer4 is about position sizing. I want to trade it with a triple barrier method
that includes trailing profit.

This module implements:
1.  Triple Barrier Trailing Logic (Exit Strategy).
2.  Inverse Volatility Sizing (Position Sizing).
3.  Integration with Layer 5 via `layer4_prob` proxy generation.
4.  Entropy Bars integration for improved information-based sampling.

REFACTORED: Now uses Layer 3 probability outputs (12/48 horizons).
Enhanced with entropy bars for better market microstructure analysis.
Sizing uses calibrated probability consensus.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from datetime import datetime
import json
import os
import hashlib

from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from scipy.stats import spearmanr, norm
import statsmodels.api as sm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.base import clone
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import itertools

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import RobustScaler

from src.utils.huber_regressor_for_trees import prepare_huber_teacher_outputs
from src.training.steps.labeling.layer4_dual_chaser import (
    StructuralRegimeGMM,
    train_dual_chaser_audit,
    generate_dual_chaser_features,
    tune_gate_params_grid
)
from src.training.steps.labeling.layer2_5_prediction_averaging import average_layer25_predictions
from src.utils.layer4_optimized import (
    compute_financial_weights_numba,
    extract_prob_features_numba,
    compute_prob_stats_numba,
    rolling_sadf_score_numba,
    rolling_cusum_scores_numba,
    compute_proxy_entropy_numba
)
from src.training.steps.labeling.feature_engineering_utils import apply_layer2_price_processing
from src.training.steps.labeling.layer4_checkpoint_manager import get_layer4_checkpoint_manager

# Import entropy bars functionality
try:
    from src.utils.entropy_bars import (
        fetch_1min_data_for_entropy_bars,
        generate_entropy_bars_from_ohlcv,
        calculate_specialized_entropy_features
    )
    ENTROPY_BARS_AVAILABLE = True
except ImportError as e:
    ENTROPY_BARS_AVAILABLE = False
    print(f"⚠️ Entropy bars not available in Layer 4: {e}")

# Configuration Constants
STOP_LOSS_FLOOR = 0.004  # 0.3% Fees + 0.1% Spread Buffer
TARGET_VOLATILITY = 0.01  # 1% target volatility for sizing
VOLATILITY_SAFETY_FLOOR = 1e-4  # Prevent division by zero
HOME_RUN_MULTIPLIER = 3.0  # Multiplier for home run detection
WEIGHT_CLIP_MIN = 0.5  # Minimum weight clip
WEIGHT_CLIP_MAX = 2.0  # Maximum weight clip

def downcast_float(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 columns to float32 to save memory and speed up processing."""
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        # Check for infs before casting (only if necessary to avoid copy if not needed)
        # Using a more efficient check
        vals = df[float_cols].values
        if np.isinf(vals).any():
             df[float_cols] = df[float_cols].replace([np.inf, -np.inf], np.nan)

        # Use simple astype
        df[float_cols] = df[float_cols].astype(np.float32)
    return df

class SimpleMultiModelRiskEngine:
    """
    Simple Multi-Model Risk Engine updated for Layer 3 probability inputs with Entropy Bars.

    Consumes:
    - Layer 3 probability outputs (12/48 horizons, ensemble variants)
    - Disagreement/dispersion features across horizons
    - Entropy bar features for market microstructure analysis

    Updated to use Huber Regressor for Teacher logic, pruning, and constraints.
    """
    
    def __init__(self, 
                 n_estimators: int = 1000, 
                 max_features: str = 'log2',
                 consensus_weights: Optional[Dict[str, float]] = None):
        
        # Default weights if not provided
        self.consensus_weights = consensus_weights or {
            'extratrees': 0.20,  # Reduced from 0.25
            'lgbm': 0.20,        # Reduced from 0.25
            'xgboost': 0.20,     # Reduced from 0.25
            'catboost': 0.20,    # Reduced from 0.25
            'ridge_alpha1': 0.0667,  # New: 1/15
            'ridge_alpha5': 0.0667,  # New: 1/15
            'ridge_alpha10': 0.0666  # New: 1/15 (rounded)
        }
        
        # Models configuration
        self.extratrees = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        )
        
        # LGBM params (implicit defaults + warm start handling in train)
        self.lgbm_params = {
            'n_estimators': 1000,
            'n_jobs': 2,
            'random_state': 42,
            'verbosity': -1,
            'reg_alpha': 5.0,
            'reg_lambda': 50.0
        }
        self.lgbm_model = None

        # XGB params
        self.xgb_params = {
            'n_estimators': 4,
            'num_parallel_tree': 250,
            'linear_tree': True,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bynode': 0.65,
            'tree_method': 'hist',
            'reg_lambda': 5,
            'reg_alpha': 0.5,
            'gamma': 0.2,
            'colsample_bytree': 0.8,
            'learning_rate': 0.2,
            'min_child_weight': 25,
            'n_jobs': 2,
            'random_state': 42
        }
        self.xgb_model = None

        # CatBoost params
        self.catboost_params = {
            'n_estimators': 1000,
            'random_state': 42,
            'thread_count': -1,
            'subsample': 0.6,
            'colsample_bylevel': 0.5,
            'leaf_estimation_iterations': 10,
            'l2_leaf_reg': 20, # overriding 7 with 20 per instructions
            'random_strength': 5,
            'bootstrap_type': 'MVS',
            'allow_writing_files': False,
            'verbose': 0
        }
        self.catboost_model = None
        
        # Ridge models with different alphas
        self.ridge_alpha1 = Ridge(alpha=1.0, random_state=42)
        self.ridge_alpha5 = Ridge(alpha=5.0, random_state=42)
        self.ridge_alpha10 = Ridge(alpha=10.0, random_state=42)
        
        self.calibrators = {
            'extratrees': IsotonicRegression(out_of_bounds='clip'),
            'lgbm': IsotonicRegression(out_of_bounds='clip'),
            'xgboost': IsotonicRegression(out_of_bounds='clip'),
            'catboost': IsotonicRegression(out_of_bounds='clip'),
            'ridge_alpha1': IsotonicRegression(out_of_bounds='clip'),
            'ridge_alpha5': IsotonicRegression(out_of_bounds='clip'),
            'ridge_alpha10': IsotonicRegression(out_of_bounds='clip')
        }
        
        self.consensus_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.feature_names = None
        self.selected_features = None
        self.huber_model = None
        self.huber_scaler = None
        self.huber_feature_columns = None
        self.is_fitted = False
        
        # Dynamic consensus attributes
        self.selected_models = None
        self.optimized_weights = None
        self.model_selection_results = None

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the risk engine for checkpointing.
        """
        return {
            'extratrees': self.extratrees,
            'lgbm_model': self.lgbm_model,
            'xgb_model': self.xgb_model,
            'catboost_model': self.catboost_model,
            'ridge_alpha1': self.ridge_alpha1,
            'ridge_alpha5': self.ridge_alpha5,
            'ridge_alpha10': self.ridge_alpha10,
            'calibrators': self.calibrators,
            'consensus_calibrator': self.consensus_calibrator,
            'selected_features': self.selected_features,
            'huber_model': self.huber_model,
            'huber_scaler': self.huber_scaler,
            'huber_feature_columns': self.huber_feature_columns,
            'is_fitted': self.is_fitted,
            'selected_models': self.selected_models,
            'optimized_weights': self.optimized_weights,
            'model_selection_results': self.model_selection_results,
            'gate_params': getattr(self, 'gate_params', None),
            'stable_chaser': getattr(self, 'stable_chaser', None),
            'aggressive_chaser': getattr(self, 'aggressive_chaser', None),
            'dual_chaser_scaler': getattr(self, 'dual_chaser_scaler', None),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load state from a checkpoint.
        """
        self.extratrees = state.get('extratrees', self.extratrees)
        self.lgbm_model = state.get('lgbm_model', self.lgbm_model)
        self.xgb_model = state.get('xgb_model', self.xgb_model)
        self.catboost_model = state.get('catboost_model', self.catboost_model)
        self.ridge_alpha1 = state.get('ridge_alpha1', self.ridge_alpha1)
        self.ridge_alpha5 = state.get('ridge_alpha5', self.ridge_alpha5)
        self.ridge_alpha10 = state.get('ridge_alpha10', self.ridge_alpha10)
        self.calibrators = state.get('calibrators', self.calibrators)
        self.consensus_calibrator = state.get('consensus_calibrator', self.consensus_calibrator)
        self.selected_features = state.get('selected_features', self.selected_features)
        self.huber_model = state.get('huber_model', self.huber_model)
        self.huber_scaler = state.get('huber_scaler', self.huber_scaler)
        self.huber_feature_columns = state.get('huber_feature_columns', self.huber_feature_columns)
        self.is_fitted = state.get('is_fitted', self.is_fitted)
        self.selected_models = state.get('selected_models', self.selected_models)
        self.optimized_weights = state.get('optimized_weights', self.optimized_weights)
        self.model_selection_results = state.get('model_selection_results', self.model_selection_results)

        if 'gate_params' in state:
            self.gate_params = state['gate_params']
        if 'stable_chaser' in state:
            self.stable_chaser = state['stable_chaser']
        if 'aggressive_chaser' in state:
            self.aggressive_chaser = state['aggressive_chaser']
        if 'dual_chaser_scaler' in state:
            self.dual_chaser_scaler = state['dual_chaser_scaler']
    
    def _compute_financial_weights(self, abs_returns: pd.Series, volatility: pd.Series) -> pd.Series:
        # Use Numba-optimized implementation
        weights_array = compute_financial_weights_numba(
            abs_returns.values.astype(np.float64),
            volatility.values.astype(np.float64)
        )
        return pd.Series(weights_array, index=abs_returns.index)

    def _calculate_learnability_weights(self, X, residuals, env_indices=None):
        """
        Scout Pass: Uses a smaller forest to determine sample weights
        based on prediction consensus (inverse of variance).
        If env_indices is provided, performs estimation per regime.
        """
        # Ensure X is numpy
        X_np = X.values if hasattr(X, "values") else X
        res_np = residuals.values if hasattr(residuals, "values") else residuals

        weights = np.zeros(len(res_np))

        # Helper to get fresh scout
        def get_scout():
             return ExtraTreesRegressor(n_estimators=100, max_depth=4, bootstrap=True, n_jobs=-1, random_state=42)

        if env_indices is not None:
            # Per-regime
            unique_regimes = np.unique(env_indices)
            tprint_info(f"⚖️ Learnability Scout: Training per-regime ({len(unique_regimes)} regimes)...")

            for regime in unique_regimes:
                if regime == -1: continue # Skip noise label if any
                mask = (env_indices == regime)
                if np.sum(mask) < 20:
                    # Fallback for tiny regimes: use global mean weight later (zeros)
                    continue

                scout = get_scout()
                X_sub = X_np[mask]
                y_sub = res_np[mask]

                scout.fit(X_sub, y_sub)

                # Get variance on subset
                tree_preds = np.array([tree.predict(X_sub) for tree in scout.estimators_])
                variance = np.var(tree_preds, axis=0)

                # Local weights
                w_local = 1.0 / (1.0 + variance)
                weights[mask] = w_local

            # Handle unassigned (zeros) with mean of assigned
            if np.any(weights == 0):
                mean_w = np.mean(weights[weights > 0]) if np.any(weights > 0) else 1.0
                weights[weights == 0] = mean_w

        else:
            # Global
            scout = get_scout()
            scout.fit(X_np, res_np)
            tree_preds = np.array([tree.predict(X_np) for tree in scout.estimators_])
            variance = np.var(tree_preds, axis=0)
            weights = 1.0 / (1.0 + variance)

        # Normalize (0 to 1)
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
             weights = (weights - w_min) / (w_max - w_min + 1e-9)
        else:
             weights = np.ones_like(weights)

        return weights
    
    def _extract_layer3_prob_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives features from Layer 3 probability outputs:
        1. Raw probabilities per horizon/model
        2. Confidence (distance from 0.5)
        3. Horizon disagreement (12 vs 48)
        4. Cross-model dispersion (mean/std/range)
        """
        feats = pd.DataFrame(index=df.index)

        base_prob_cols = []
        if 'meta_prob' in df.columns:
            base_prob_cols.append('meta_prob')
        if 'meta_prob_48' in df.columns:
            base_prob_cols.append('meta_prob_48')

        extra_prob_cols = [c for c in df.columns if c.startswith('meta_prob_') and c not in base_prob_cols]
        prob_cols = base_prob_cols + extra_prob_cols

        if not prob_cols:
            return feats

        # Vectorized feature extraction using Numba
        # Optimization: Avoid astype copy if already float32
        probs_vals = df[prob_cols].values
        if probs_vals.dtype == np.float32:
            probs_array = probs_vals
        else:
            probs_array = probs_vals.astype(np.float32)

        logits, confidences = extract_prob_features_numba(probs_array)

        for idx, col in enumerate(prob_cols):
            feats[f'{col}_raw'] = probs_array[:, idx]
            feats[f'{col}_logit'] = logits[:, idx]
            feats[f'{col}_confidence'] = confidences[:, idx]

        if 'meta_prob' in df.columns and 'meta_prob_48' in df.columns:
            # Vectorized horizon features
            p12 = df['meta_prob'].values
            p48 = df['meta_prob_48'].values
            diff = p12 - p48
            feats['horizon_disagreement'] = diff
            feats['horizon_agreement'] = 1.0 - np.abs(diff)

        # Numba-optimized statistics
        means, stds, mins, maxs, ranges = compute_prob_stats_numba(probs_array)
        feats['prob_mean'] = means
        feats['prob_std'] = stds
        feats['prob_min'] = mins
        feats['prob_max'] = maxs
        feats['prob_range'] = ranges

        return feats.fillna(0.0)

    def _extract_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract entropy bar features for enhanced market microstructure analysis.
        """
        feats = pd.DataFrame(index=df.index)
        
        # Entropy bar specific features
        entropy_feature_cols = [
            'staleness_seconds', 'staleness_minutes', 'drift_proxy', 
            'lz_complexity', 'trend_conviction_index', 'staleness_adjusted_drift',
            'entropy_ma', 'entropy_std', 'entropy_zscore', 'proxy_entropy'
        ]
        
        for col in entropy_feature_cols:
            if col in df.columns:
                feats[col] = df[col]
        
        # Entropy OHLCV features
        entropy_ohlcv_cols = ['entropy_close', 'entropy_volume']
        for col in entropy_ohlcv_cols:
            if col in df.columns:
                feats[col] = df[col]
        
        return feats.fillna(0)

    def _analyze_model_correlations(self, base_predictions: Dict[str, np.ndarray], 
                                   y_true: pd.Series, abs_returns: pd.Series) -> Dict[str, Any]:
        """
        Analyze correlations between model predictions and select top 4 based on:
        1. Low pairwise correlation (diversity)
        2. High PnL performance
        """
        model_names = list(base_predictions.keys())
        n_models = len(model_names)
        
        # Calculate correlation matrix
        corr_matrix = np.zeros((n_models, n_models))
        for i, j in itertools.combinations(range(n_models), 2):
            corr, _ = spearmanr(base_predictions[model_names[i]], base_predictions[model_names[j]])
            corr_matrix[i, j] = abs(corr)
            corr_matrix[j, i] = abs(corr)
        
        # Calculate PnL metrics for each model
        pnl_metrics = {}
        for name in model_names:
            preds = base_predictions[name]
            # Simple PnL: direction * return
            direction = np.sign(preds - 0.5)
            pnl = direction * y_true.values
            # Sharpe-like metric
            sharpe = np.mean(pnl) / (np.std(pnl) + 1e-9)
            pnl_metrics[name] = sharpe
        
        # Model selection algorithm
        selected_models = []
        remaining_models = model_names.copy()
        
        # Select first model (highest PnL)
        first_model = max(remaining_models, key=lambda x: pnl_metrics[x])
        selected_models.append(first_model)
        remaining_models.remove(first_model)
        
        # Select remaining 3 models
        while len(selected_models) < 4 and remaining_models:
            best_score = -np.inf
            best_model = None
            
            for candidate in remaining_models:
                # Calculate average correlation with selected models
                candidate_idx = model_names.index(candidate)
                selected_indices = [model_names.index(m) for m in selected_models]
                avg_corr = np.mean([corr_matrix[candidate_idx, sel_idx] for sel_idx in selected_indices])
                
                # Diversity-adjusted score (lower correlation = higher score)
                diversity_bonus = 1.0 - avg_corr
                combined_score = pnl_metrics[candidate] * diversity_bonus
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_model = candidate
            
            if best_model:
                selected_models.append(best_model)
                remaining_models.remove(best_model)
            else:
                break
        
        # Optimize weights for selected models
        optimized_weights = self._optimize_consensus_weights(
            selected_models, base_predictions, y_true, abs_returns
        )
        
        return {
            'selected_models': selected_models,
            'optimized_weights': optimized_weights,
            'correlation_matrix': corr_matrix,
            'pnl_metrics': pnl_metrics,
            'all_models': model_names
        }

    def _optimize_consensus_weights(self, selected_models: List[str], 
                                   base_predictions: Dict[str, np.ndarray],
                                   y_true: pd.Series, abs_returns: pd.Series) -> Dict[str, float]:
        """
        Optimize consensus weights to maximize PnL Sharpe ratio.
        """
        def objective(weights):
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Calculate weighted consensus
            consensus = np.zeros(len(y_true))
            for i, model in enumerate(selected_models):
                consensus += weights[i] * base_predictions[model]
            
            # Calculate PnL
            direction = np.sign(consensus - 0.5)
            pnl = direction * y_true.values
            
            # Negative Sharpe (for minimization)
            sharpe = np.mean(pnl) / (np.std(pnl) + 1e-9)
            return -sharpe
        
        # Initial guess (equal weights)
        n_models = len(selected_models)
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: weights >= 0, sum(weights) = 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        result = minimize(objective, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimized_weights = result.x / np.sum(result.x)
        else:
            # Fallback to equal weights
            optimized_weights = initial_weights
        
        return dict(zip(selected_models, optimized_weights))

    def train(self, df: pd.DataFrame, market_features: pd.DataFrame,
              y_true: pd.Series, abs_returns: pd.Series) -> Dict[str, Any]:
        """
        Train risk engine using Layer 3 probability features, entropy features, and market context.
        Uses Huber Regressor for Teacher output generation (constraints, pruning, warm start).
        """
        tprint_info("🚀 Training Layer 4 Risk Engine (Layer 3 Probabilities + Entropy Bars)...")

        prob_feats = self._extract_layer3_prob_features(df)
        entropy_feats = self._extract_entropy_features(df) if ENTROPY_BARS_AVAILABLE else pd.DataFrame(index=df.index)
        
        # Combine all features
        X_full = pd.concat([prob_feats, entropy_feats, market_features], axis=1).fillna(0)

        # --- Layer 2.5 Chaser Integration ---
        # Average available chaser predictions (if any) and add to features
        try:
            chaser_avg_feats = average_layer25_predictions(df)
            X_full = pd.concat([X_full, chaser_avg_feats], axis=1)
            tprint_info(f"✅ Added {chaser_avg_feats.shape[1]} Layer 2.5 Chaser average features")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to add Layer 2.5 Chaser features: {e}")

        # --- Dual Chaser Audit & Feature Generation ---
        # 1. Fit GMM to identify regimes (if data available)
        env_indices = None
        try:
            gmm = StructuralRegimeGMM(n_regimes=4)
            gmm_indices, _ = gmm.fit_predict(df)

            if gmm_indices is not None and len(gmm_indices) >= 2:
                env_indices = gmm_indices # Capture for learnability scout
                tprint_info("🏃 Training Dual Chaser Audit (IRM vs Ridge) with OOF...")

                # Filter out performance features for Chasers to avoid leakage
                chaser_cols = [c for c in X_full.columns if not c.startswith('perf_') and not c.startswith('meta_')]
                X_for_chaser = X_full[chaser_cols]

                # 2. Prepare Scaled Features for Chasers
                self.dual_chaser_scaler = RobustScaler()
                X_chaser_scaled_np = self.dual_chaser_scaler.fit_transform(X_for_chaser)
                # Ensure DataFrame for feature selection inside training
                X_chaser_scaled = pd.DataFrame(
                    X_chaser_scaled_np,
                    index=X_for_chaser.index,
                    columns=X_for_chaser.columns
                )

                # Construct simple CV splits for OOF generation
                n_samples = len(X_full)
                n_splits = 5
                fold_size = n_samples // n_splits
                cv_splits = []
                for k in range(n_splits):
                    val_start = k * fold_size
                    val_end = (k + 1) * fold_size if k < n_splits - 1 else n_samples
                    val_idx = np.arange(val_start, val_end)
                    train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n_samples)])
                    cv_splits.append((train_idx, val_idx))

                # 3. Train Chasers & Get OOF
                self.stable_chaser, self.aggressive_chaser, oof_stable, oof_agg = train_dual_chaser_audit(
                    X_chaser_scaled,
                    y_true.values,
                    env_indices, # Use captured indices
                    cv_splits=cv_splits
                )

                # 4. Generate Dual Chaser Features using OOF predictions
                if oof_stable is not None and oof_agg is not None:
                    oof_stable.index = X_full.index
                    oof_agg.index = X_full.index

                    # Prepare trade_returns proxy for meta-health features
                    trade_ret_proxy = (2 * y_true - 1) * abs_returns

                    # Initial feature generation with default params
                    temp_feats = generate_dual_chaser_features(
                        df=df,
                        p_stable=oof_stable,
                        p_agg=oof_agg,
                        trade_returns=trade_ret_proxy
                    )

                    # 4b. Tune Gate Parameters
                    tprint_info("🔧 Tuning Orchestrator Gate Parameters...")

                    # Split feats into core and structural for tuning
                    # Columns are known from build_* functions in layer4_dual_chaser
                    core_cols = ["cos_sim", "raw_direction", "consensus_strength_alt"]
                    struct_cols = ["ker_fast", "ker_slow", "liquidity_score", "anchor_extreme"]

                    # Ensure structural columns are present (some might be missing if dependencies failed)
                    # We rebuild structural features temporarily to ensure alignment for tuning if needed,
                    # but temp_feats already contains everything combined.

                    # Check if columns exist
                    cols_present = [c for c in core_cols + struct_cols if c in temp_feats.columns]
                    if len(cols_present) == len(core_cols) + len(struct_cols):

                        best_gate_params, report = tune_gate_params_grid(
                            core=temp_feats[core_cols],
                            structural=temp_feats[struct_cols],
                            long_only=True
                        )

                        if best_gate_params:
                            self.gate_params = best_gate_params
                            tprint_success(f"✅ Tuned Gate Params: {best_gate_params}")

                            # Regenerate with optimized params
                            chaser_feats = generate_dual_chaser_features(
                                df=df,
                                p_stable=oof_stable,
                                p_agg=oof_agg,
                                trade_returns=trade_ret_proxy,
                                gate_params=self.gate_params
                            )
                        else:
                            tprint_warning("⚠️ Gate tuning failed constraints. Using defaults.")
                            chaser_feats = temp_feats
                    else:
                        tprint_warning(f"⚠️ Missing columns for gate tuning: {set(core_cols+struct_cols) - set(temp_feats.columns)}. Using defaults.")
                        chaser_feats = temp_feats

                    # 5. Add to X_full
                    X_full = pd.concat([X_full, chaser_feats], axis=1).fillna(0)
                    tprint_success(f"✅ Added {chaser_feats.shape[1]} Dual Chaser features (OOF)")
                else:
                    tprint_warning("⚠️ Dual Chaser OOF generation failed.")
                    self.stable_chaser = None
            else:
                tprint_warning("⚠️ Dual Chaser: GMM failed or insufficient regimes. Skipping.")
                self.stable_chaser = None
        except Exception as e:
            tprint_error(f"❌ Dual Chaser failed: {e}")
            import traceback
            traceback.print_exc()
            self.stable_chaser = None

        # Downcast to float32
        X_full = downcast_float(X_full)
        X_full = X_full.fillna(0)

        # Extract volatility for weighting
        if 'volatility' in market_features.columns:
            volatility = market_features['volatility']
        else:
            tprint_warning("⚠️ Volatility not found in market_features. Using unweighted fallback (1.0).")
            volatility = pd.Series(np.ones(len(abs_returns)), index=abs_returns.index)

        weights = self._compute_financial_weights(abs_returns, volatility)
        
        tprint_info(f"📊 Processing Huber Teacher on {len(X_full.columns)} potential features...")

        # --- Prepare Huber Teacher ---
        huber_outputs = prepare_huber_teacher_outputs(
            X_train=X_full,
            y_train=y_true,
            pruning_percentile=15,
            corr_threshold=0.7,
            epsilons=[1.10],
            alphas=[3.0],
            irm_lambda=10.0,
            max_iter=2000
        )

        self.selected_features = huber_outputs['selected_features']
        self.huber_model = huber_outputs.get('huber_teacher', huber_outputs.get('huber_model'))
        self.huber_scaler = huber_outputs['scaler']
        self.huber_feature_columns = X_full.columns

        X_pruned = X_full[self.selected_features]
        warm_start_train = huber_outputs['warm_start']['train']

        # Calculate Residuals & Learnability Weights
        residuals = y_true - warm_start_train
        learnability_weights = self._calculate_learnability_weights(X_pruned, residuals, env_indices=env_indices)

        # Combine weights
        # Ensure weights series aligns with learnability_weights (which is numpy array)
        combined_weights = weights.values * learnability_weights if hasattr(weights, 'values') else weights * learnability_weights

        # Extract monotonic constraints (handle dict return from updated Huber)
        if isinstance(huber_outputs['monotonic_constraints'], dict):
             if 'monotonic_constraints_details' in huber_outputs:
                 # Use the pre-calculated selected list (aligned with selected_features)
                 monotonic_cst_tuple = huber_outputs['monotonic_constraints_details']['lightgbm_selected']
             else:
                 # Fallback: reconstruct
                 monotonic_cst_tuple = [huber_outputs['monotonic_constraints'][f] for f in self.selected_features]
        else:
             monotonic_cst_tuple = huber_outputs['monotonic_constraints']

        # Ensure tuple format
        monotonic_cst_tuple = tuple(int(x) for x in monotonic_cst_tuple)

        interaction_cst = huber_outputs['interaction_constraints']

        # DEBUG: Inspect constraints for XGBoost issue
        if interaction_cst:
            tprint_info(f"DEBUG: Interaction Constraints Sample: {interaction_cst[:2]}")
            tprint_info(f"DEBUG: Interaction Type: {type(interaction_cst)}")
            if len(interaction_cst) > 0:
                tprint_info(f"DEBUG: Interaction Inner Type: {type(interaction_cst[0])}, Element Type: {type(interaction_cst[0][0])}")
            tprint_info(f"DEBUG: Pruned Features: {self.selected_features[:5]}")

        tprint_info(f"✨ Feature pruning: {len(self.selected_features)}/{len(X_full.columns)} features selected")

        # --- Generate OOF Predictions for Weight Optimization ---
        # To avoid leakage (weights optimized on in-sample predictions), we must generate OOF predictions
        # for all student models before optimizing weights.
        tprint_info("🔄 Generating OOF Student Predictions for Weight Optimization...")

        # Initialize OOF arrays
        oof_preds = {
            'extratrees': np.zeros(len(y_true)),
            'lgbm': np.zeros(len(y_true)),
            'xgboost': np.zeros(len(y_true)),
            'catboost': np.zeros(len(y_true)),
            'ridge_alpha1': np.zeros(len(y_true)),
            'ridge_alpha5': np.zeros(len(y_true)),
            'ridge_alpha10': np.zeros(len(y_true))
        }

        # OOF Loop
        kf_internal = KFold(n_splits=5, shuffle=False)

        for fold_idx, (tr_idx, val_idx) in enumerate(kf_internal.split(X_pruned)):
            # Slice data
            X_tr, X_val = X_pruned.iloc[tr_idx], X_pruned.iloc[val_idx]
            y_tr, _ = y_true.iloc[tr_idx], y_true.iloc[val_idx] # y_val unused in loop
            ws_tr, ws_val = warm_start_train[tr_idx], warm_start_train[val_idx]
            # Weights
            w_tr = combined_weights[tr_idx] if hasattr(combined_weights, 'shape') else combined_weights # Handling if list vs array
            if hasattr(combined_weights, 'values'):
                 w_tr = combined_weights.values[tr_idx]
            elif isinstance(combined_weights, np.ndarray):
                 w_tr = combined_weights[tr_idx]

            # Residuals for ExtraTrees
            res_tr = residuals.values[tr_idx] if hasattr(residuals, 'values') else residuals[tr_idx]

            # 1. ExtraTrees OOF
            et_fold = clone(self.extratrees)
            et_fold.fit(X_tr, res_tr, sample_weight=w_tr)
            # Predict residual + add warm start
            oof_preds['extratrees'][val_idx] = et_fold.predict(X_val) + ws_val

            # 2. LGBM OOF
            lgbm_fold = LGBMRegressor(**self.lgbm_params)
            lgbm_fold.set_params(monotone_constraints=list(monotonic_cst_tuple), interaction_constraints=interaction_cst if interaction_cst else None)
            lgbm_fold.fit(X_tr, y_tr, sample_weight=w_tr, init_score=ws_tr)
            # LGBM predict(X) returns residual (sum of trees). Verified by script.
            # Must add warm start manually.
            oof_preds['lgbm'][val_idx] = lgbm_fold.predict(X_val) + ws_val

            # 3. XGBoost OOF
            xgb_fold = XGBRegressor(**self.xgb_params)
            try:
                xgb_fold.set_params(monotone_constraints=monotonic_cst_tuple, interaction_constraints=interaction_cst if interaction_cst else None)
                xgb_fold.fit(X_tr, y_tr, sample_weight=w_tr, base_margin=ws_tr)
            except (ValueError, KeyError):
                xgb_fold.set_params(interaction_constraints=None)
                xgb_fold.fit(X_tr, y_tr, sample_weight=w_tr, base_margin=ws_tr)

            # XGB predict(X, base_margin=...) returns FULL value. Verified by script.
            oof_preds['xgboost'][val_idx] = xgb_fold.predict(X_val, base_margin=ws_val)

            # 4. CatBoost OOF
            cb_fold = CatBoostRegressor(**self.catboost_params)
            cb_fold.set_params(monotone_constraints=list(monotonic_cst_tuple))
            cb_fold.fit(X_tr, y_tr, sample_weight=w_tr, baseline=ws_tr)
            # CatBoost predict(X) returns residual. Verified by script.
            # Must add warm start manually.
            oof_preds['catboost'][val_idx] = cb_fold.predict(X_val) + ws_val

            # 5. Ridge OOF (Full model, no warm start offset usually, unless we changed it. Code below trains on y_true)
            r1 = clone(self.ridge_alpha1)
            r1.fit(X_tr, y_tr, sample_weight=w_tr)
            oof_preds['ridge_alpha1'][val_idx] = r1.predict(X_val)

            r5 = clone(self.ridge_alpha5)
            r5.fit(X_tr, y_tr, sample_weight=w_tr)
            oof_preds['ridge_alpha5'][val_idx] = r5.predict(X_val)

            r10 = clone(self.ridge_alpha10)
            r10.fit(X_tr, y_tr, sample_weight=w_tr)
            oof_preds['ridge_alpha10'][val_idx] = r10.predict(X_val)

        
        # --- Dynamic Model Selection & Consensus (using OOF) ---
        tprint_info("🔍 Analyzing model correlations and selecting top 4 (using OOF)...")
        selection_results = self._analyze_model_correlations(oof_preds, y_true, abs_returns)

        self.selected_models = selection_results['selected_models']
        self.optimized_weights = selection_results['optimized_weights']

        # Store analysis results for logging
        self.model_selection_results = selection_results

        tprint_info(f"✅ Selected models: {self.selected_models}")
        tprint_info(f"📊 Optimized weights: {self.optimized_weights}")

        # --- Final Fit on Full Data (for Inference) ---
        tprint_info("🧠 Fitting Final Models on Full Data...")
        base_predictions = {}
        
        # 1. ExtraTrees (Full)
        self.extratrees.fit(X_pruned, residuals, sample_weight=combined_weights)
        et_residual_preds = self.extratrees.predict(X_pruned)
        et_preds = warm_start_train + et_residual_preds
        base_predictions['extratrees'] = et_preds # Store In-Sample for final calibration if needed, but we should probably use OOF for calibration too?
        # Using OOF for calibration is better.
        self.calibrators['extratrees'].fit(oof_preds['extratrees'], y_true)
        
        # 2. LGBM (Full)
        self.lgbm_model = LGBMRegressor(**self.lgbm_params)
        self.lgbm_model.set_params(
            monotone_constraints=list(monotonic_cst_tuple),
            interaction_constraints=interaction_cst if interaction_cst else None
        )
        self.lgbm_model.fit(
            X_pruned, y_true,
            sample_weight=combined_weights,
            init_score=warm_start_train
        )
        # Note: We store the in-sample prediction just for completeness in base_predictions,
        # but calibration should use OOF.
        lgbm_raw_preds = self.lgbm_model.predict(X_pruned)
        lgbm_preds = lgbm_raw_preds + warm_start_train
        base_predictions['lgbm'] = lgbm_preds
        self.calibrators['lgbm'].fit(oof_preds['lgbm'], y_true)

        # 3. XGB Regressor (Full)
        self.xgb_model = XGBRegressor(**self.xgb_params)
        try:
            self.xgb_model.set_params(
                monotone_constraints=monotonic_cst_tuple,
                interaction_constraints=interaction_cst if interaction_cst else None
            )
            self.xgb_model.fit(
                X_pruned, y_true,
                sample_weight=combined_weights,
                base_margin=warm_start_train
            )
        except (ValueError, KeyError) as e:
            tprint_error(f"XGBoost constraints failed: {e}. Retrying without interaction constraints.")
            self.xgb_model.set_params(interaction_constraints=None)
            self.xgb_model.fit(
                X_pruned, y_true,
                sample_weight=combined_weights,
                base_margin=warm_start_train
            )

        xgb_preds = self.xgb_model.predict(X_pruned, base_margin=warm_start_train)
        base_predictions['xgboost'] = xgb_preds
        self.calibrators['xgboost'].fit(oof_preds['xgboost'], y_true)

        # 4. CatBoost (Full)
        self.catboost_model = CatBoostRegressor(**self.catboost_params)
        self.catboost_model.set_params(monotone_constraints=list(monotonic_cst_tuple))
        self.catboost_model.fit(
            X_pruned, y_true,
            sample_weight=combined_weights,
            baseline=warm_start_train
        )
        cb_raw_preds = self.catboost_model.predict(X_pruned)
        catboost_preds = warm_start_train + cb_raw_preds
        base_predictions['catboost'] = catboost_preds
        self.calibrators['catboost'].fit(oof_preds['catboost'], y_true)
        
        # 5. Ridge Models (Full)
        self.ridge_alpha1.fit(X_pruned, y_true, sample_weight=combined_weights)
        ridge1_preds = self.ridge_alpha1.predict(X_pruned)
        base_predictions['ridge_alpha1'] = ridge1_preds
        self.calibrators['ridge_alpha1'].fit(oof_preds['ridge_alpha1'], y_true)

        self.ridge_alpha5.fit(X_pruned, y_true, sample_weight=combined_weights)
        ridge5_preds = self.ridge_alpha5.predict(X_pruned)
        base_predictions['ridge_alpha5'] = ridge5_preds
        self.calibrators['ridge_alpha5'].fit(oof_preds['ridge_alpha5'], y_true)

        self.ridge_alpha10.fit(X_pruned, y_true, sample_weight=combined_weights)
        ridge10_preds = self.ridge_alpha10.predict(X_pruned)
        base_predictions['ridge_alpha10'] = ridge10_preds
        self.calibrators['ridge_alpha10'].fit(oof_preds['ridge_alpha10'], y_true)
        
        # Build consensus with optimized weights (using OOF calibrated preds for metric calculation)
        # Note: We must calibrate the OOF preds first if we want to report clean metrics
        # The 'base_predictions' dict now contains IN-SAMPLE predictions from final models.
        # But for 'consensus_calibrator' fitting, we should strictly use OOF.

        # Calibrate OOF preds
        consensus_oof_raw = np.zeros(len(y_true))
        for model in self.selected_models:
             # Transform OOF using the calibrator (which was just fitted on OOF)
             # Wait, Isotonic is prone to overfitting if fitted on same data?
             # Standard practice: Fit Isotonic on OOF.
             # So cal_oof = iso.transform(oof) -- this is basically fitting on OOF.
             # Actually, if we fit on OOF, we are just mapping OOF->Target.
             # This is fine.
             cal_oof = self.calibrators[model].transform(oof_preds[model])
             consensus_oof_raw += self.optimized_weights[model] * cal_oof

        # Calibrate Consensus
        self.consensus_calibrator.fit(consensus_oof_raw, y_true)
        consensus_calibrated = self.consensus_calibrator.transform(consensus_oof_raw)

        # Store analysis results for logging
        self.model_selection_results = selection_results

        tprint_info(f"✅ Selected models: {self.selected_models}")
        tprint_info(f"📊 Optimized weights: {self.optimized_weights}")
        
        self.is_fitted = True
        self.final_predictions_ = consensus_calibrated
        self.feature_names = self.selected_features # Save for consistency check
        
        metrics = {
            'consensus_weighted_logloss': log_loss(y_true, consensus_calibrated, sample_weight=weights),
            'n_features_total': len(X_full.columns),
            'n_features_pruned': len(self.selected_features),
            'mean_conviction': consensus_calibrated.mean(),
            'selected_models': self.selected_models,
            'optimized_weights': self.optimized_weights,
            'correlation_matrix': selection_results['correlation_matrix'].tolist(),
            'pnl_metrics': selection_results['pnl_metrics']
        }
        
        tprint_success(f"✅ Layer 4 Engine trained: WL={metrics['consensus_weighted_logloss']:.4f}, Features={metrics['n_features_pruned']}")
        return metrics

    def predict_bet_size(self, df: pd.DataFrame, market_features: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("RiskEngine must be fitted")
            
        prob_feats = self._extract_layer3_prob_features(df)
        entropy_feats = self._extract_entropy_features(df) if ENTROPY_BARS_AVAILABLE else pd.DataFrame(index=df.index)
        
        X_full = pd.concat([prob_feats, entropy_feats, market_features], axis=1).fillna(0)

        # --- Layer 2.5 Chaser Integration ---
        try:
            chaser_avg_feats = average_layer25_predictions(df)
            X_full = pd.concat([X_full, chaser_avg_feats], axis=1)
        except Exception as e:
            tprint_warning(f"⚠️ Failed to add Layer 2.5 Chaser features: {e}")

        # --- Dual Chaser Feature Generation ---
        if self.stable_chaser is not None and self.dual_chaser_scaler is not None:
            try:
                # Filter out performance features
                chaser_cols = [c for c in X_full.columns if not c.startswith('perf_') and not c.startswith('meta_')]
                X_for_chaser = X_full[chaser_cols]

                if isinstance(X_for_chaser, pd.DataFrame):
                    X_chaser_scaled_np = self.dual_chaser_scaler.transform(X_for_chaser)
                    X_chaser_scaled = pd.DataFrame(
                        X_chaser_scaled_np,
                        columns=X_for_chaser.columns,
                        index=X_for_chaser.index
                    )
                else:
                    X_chaser_scaled = self.dual_chaser_scaler.transform(X_for_chaser)

                # Predict on test set using full models with appropriate features
                X_stable_in = X_chaser_scaled
                X_agg_in = X_chaser_scaled

                if hasattr(self.stable_chaser, 'selected_features_'):
                    # Ensure alignment (if X_stable_in is DataFrame)
                    if isinstance(X_stable_in, pd.DataFrame):
                        X_stable_in = X_stable_in[self.stable_chaser.selected_features_]
                    # If numpy, we hope indices match, but feature selection logic assumed DF.
                    # Since we reconstructed DF above, it should work.

                if hasattr(self.aggressive_chaser, 'selected_features_'):
                    if isinstance(X_agg_in, pd.DataFrame):
                        X_agg_in = X_agg_in[self.aggressive_chaser.selected_features_]

                # Convert to values for predict
                p_stable = pd.Series(self.stable_chaser.predict(X_stable_in.values), index=X_full.index)
                p_agg = pd.Series(self.aggressive_chaser.predict(X_agg_in.values), index=X_full.index)

                # Generate features (meta-health skipped or using placeholder as trade returns unknown for test)
                chaser_feats = generate_dual_chaser_features(
                    df=df,
                    p_stable=p_stable,
                    p_agg=p_agg,
                    trade_returns=None,
                    gate_params=self.gate_params
                )

                X_full = pd.concat([X_full, chaser_feats], axis=1).fillna(0)
            except Exception as e:
                tprint_warning(f"⚠️ Dual Chaser prediction failed: {e}")
                import traceback
                traceback.print_exc()
                pass

        # Downcast to float32
        X_full = downcast_float(X_full)
        X_full = X_full.fillna(0)

        if self.huber_feature_columns is not None:
            X_full = X_full.reindex(columns=self.huber_feature_columns, fill_value=0.0)

        # Prepare Warm Start for this new data
        X_scaled = pd.DataFrame(self.huber_scaler.transform(X_full), columns=X_full.columns)
        warm_start = self.huber_model.predict(X_scaled)
        
        # Select features
        X_pruned = X_full[self.selected_features]

        # 1. ExtraTrees
        et_residual_preds = self.extratrees.predict(X_pruned)
        et_preds = warm_start + et_residual_preds
        
        # 2. LGBM
        lgbm_raw = self.lgbm_model.predict(X_pruned)
        lgbm_preds = lgbm_raw + warm_start

        # 3. XGBoost
        xgb_preds = self.xgb_model.predict(X_pruned, base_margin=warm_start)
        
        # 4. CatBoost
        cb_raw = self.catboost_model.predict(X_pruned)
        catboost_preds = cb_raw + warm_start

        # 5. Ridge models
        ridge1_preds = self.ridge_alpha1.predict(X_pruned)
        ridge5_preds = self.ridge_alpha5.predict(X_pruned)
        ridge10_preds = self.ridge_alpha10.predict(X_pruned)

        # Collect raw predictions into a dictionary for easy access in loop
        raw_model_preds = {
            'extratrees': et_preds,
            'lgbm': lgbm_preds,
            'xgboost': xgb_preds,
            'catboost': catboost_preds,
            'ridge_alpha1': ridge1_preds,
            'ridge_alpha5': ridge5_preds,
            'ridge_alpha10': ridge10_preds
        }

        # Dynamic consensus with selected models
        # IMPORTANT: Calibrate predictions BEFORE averaging to match OOF training logic.
        # The weights were optimized on calibrated OOF predictions.
        consensus = np.zeros(len(X_pruned))
        for model in self.selected_models:
            # Apply calibration (transform)
            calibrated_pred = self.calibrators[model].transform(raw_model_preds[model])
            consensus += self.optimized_weights[model] * calibrated_pred
        
        return self.consensus_calibrator.transform(consensus)


def integrate_entropy_bars_into_layer4(
    df: pd.DataFrame,
    symbol: str = "ETHUSDT",
    exchange: str = "binance",
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Integrate entropy bars into Layer 4 processing.
    
    Args:
        df: Original DataFrame with market data
        symbol: Trading symbol
        exchange: Exchange name
        config: Configuration dictionary
        
    Returns:
        Tuple of (enhanced_df, entropy_bars_df)
    """
    cfg = config or {}
    
    try:
        # Determine date range from existing data
        if not df.empty and hasattr(df.index, 'min') and hasattr(df.index, 'max'):
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')
        else:
            # Default to last 30 days if no date range available
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        
        # --- Caching Mechanism ---
        cache_dir = Path("cache/entropy_bars")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create hash of parameters to uniquely identify cache
        cache_key = f"{symbol}_{exchange}_{start_date}_{end_date}_{cfg.get('entropy_bins', 10)}_{cfg.get('entropy_window', 100)}_{cfg.get('entropy_target_minutes', 15)}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = cache_dir / f"entropy_features_{cache_hash}.parquet"

        if cache_file.exists():
            tprint_info(f"⚡ Layer 4: Loading entropy features from cache: {cache_file}")
            try:
                entropy_features_all = pd.read_parquet(cache_file)

                # Filter to match current df index
                # Ensure index is datetime
                if not isinstance(entropy_features_all.index, pd.DatetimeIndex):
                    entropy_features_all.index = pd.to_datetime(entropy_features_all.index)

                # Reindex to match current df
                # This is fast
                entropy_features = entropy_features_all.reindex(df.index, method='ffill').fillna(0)

                enhanced_df = df.join(entropy_features, rsuffix='_entropy')
                return enhanced_df, pd.DataFrame() # We don't return raw bars from cache to save space
            except Exception as e:
                tprint_warning(f"⚠️ Failed to load cache, regenerating: {e}")

        # Try to fetch 1-min data if allowed
        use_proxy = False
        min_data = None
        
        if ENTROPY_BARS_AVAILABLE:
            tprint_info("🔧 Layer 4: Fetching 1-minute data for entropy bar generation")
            try:
                min_data = fetch_1min_data_for_entropy_bars(
                    symbol=symbol,
                    exchange=exchange,
                    start_date=start_date,
                    end_date=end_date,
                    data_dir=cfg.get('data_dir', 'historical_data')
                )
            except Exception as e:
                tprint_warning(f"⚠️ Layer 4: Failed to fetch 1-min data: {e}. Switching to Proxy Entropy.")
                use_proxy = True
        else:
            tprint_info("ℹ️ Layer 4: Entropy Bars module not available. Switching to Proxy Entropy.")
            use_proxy = True

        if use_proxy or min_data is None or min_data.empty:
            if not use_proxy:
                tprint_warning("⚠️ Layer 4: No 1-minute data available. Switching to Proxy Entropy.")

            # --- PROXY ENTROPY CALCULATION ---
            tprint_info("🔄 Layer 4: Calculating Proxy Entropy (Numba Optimized) on base timeframe...")

            # Calculate returns on the base dataframe
            if 'close' in df.columns:
                price_col = 'close'
            elif 'Close' in df.columns:
                price_col = 'Close'
            else:
                tprint_error("❌ Layer 4: No price column found for proxy entropy.")
                return df, pd.DataFrame()

            # Calculate log returns
            prices = df[price_col].values.astype(np.float64)
            returns = np.zeros_like(prices)
            returns[1:] = np.diff(np.log(prices + 1e-9))

            # Use Numba function
            proxy_entropy = compute_proxy_entropy_numba(
                returns,
                window=cfg.get('entropy_window', 100),
                n_bins=cfg.get('entropy_bins', 10)
            )

            entropy_features = pd.DataFrame(index=df.index)
            entropy_features['proxy_entropy'] = proxy_entropy

            # Cache the proxy features too
            try:
                entropy_features.to_parquet(cache_file)
                tprint_info(f"💾 Layer 4: Cached proxy entropy features to {cache_file}")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to cache proxy entropy features: {e}")

            enhanced_df = df.join(entropy_features, rsuffix='_entropy')
            return enhanced_df, pd.DataFrame()
        
        # Generate entropy bars (Normal Path)
        tprint_info("🔄 Layer 4: Generating entropy bars from 1-minute data")
        entropy_bars = generate_entropy_bars_from_ohlcv(
            ohlcv_data=min_data,
            n_bins=cfg.get('entropy_bins', 10),
            window_size=cfg.get('entropy_window', 100),
            target_minutes=cfg.get('entropy_target_minutes', 15),
            symbol=symbol,
            exchange=exchange
        )
        
        if entropy_bars.empty:
            tprint_warning("⚠️ Layer 4: Failed to generate entropy bars. Using empty features.")
            return df, pd.DataFrame()
        
        # Calculate specialized entropy features
        tprint_info("🎯 Layer 4: Calculating specialized entropy features")
        entropy_features = calculate_specialized_entropy_features(
            entropy_bars=entropy_bars,
            base_model_updates=df,  # Use df as proxy for base model updates
            specialist_prices=df['close'] if 'close' in df.columns else None,
            volatility_window=cfg.get('volatility_window', 20)
        )
        
        # Prepare feature set to merge
        features_to_merge = entropy_features.copy()
        
        # Add entropy bar OHLCV data as additional columns
        entropy_ohlcv_cols = ['close', 'volume', 'n_minutes', 'entropy_contribution']
        for col in entropy_ohlcv_cols:
            if col in entropy_bars.columns:
                 features_to_merge[f'entropy_{col}'] = entropy_bars[col]

        # Cache the raw features (before reindexing to specific df)
        try:
            features_to_merge.to_parquet(cache_file)
            tprint_info(f"💾 Layer 4: Cached entropy features to {cache_file}")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to cache entropy features: {e}")

        # Merge entropy features back to main dataframe
        # Vectorized reindex and ffill using join/reindex
        # reindex with method='ffill' is efficient

        aligned_features = features_to_merge.reindex(df.index, method='ffill').fillna(0)
        enhanced_df = df.join(aligned_features)
        
        tprint_success(f"✅ Layer 4: Integrated entropy bars: {len(entropy_bars)} bars, {len(entropy_features.columns)} features")
        
        return enhanced_df, entropy_bars
        
    except Exception as e:
        tprint_error(f"❌ Layer 4: Error integrating entropy bars: {e}")
        return df, pd.DataFrame()


class MetaLearnerFeatures:
    """
    Generates structural market features for Layer 4 meta-learning.
    Uses Numba-optimized SADF and CUSUM scores to detect structural breaks and regimes.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.window_sadf = self.config.get('window_sadf', 100)

    def generate(self, df: pd.DataFrame, raw_price_col: str = 'close', **kwargs) -> pd.DataFrame:
        """
        Generate structural features.
        """
        if raw_price_col not in df.columns:
            return pd.DataFrame(index=df.index)

        # Calculate returns
        prices = df[raw_price_col].values.astype(np.float64)
        returns = np.zeros_like(prices)
        returns[1:] = np.diff(np.log(prices + 1e-9))

        # Volatility (for weighting)
        vol_window = self.config.get('volatility_window', 20)
        volatility = pd.Series(returns).rolling(window=vol_window, min_periods=1).std().fillna(0.0).values

        # SADF Scores (Numba Optimized)
        sadf_scores = rolling_sadf_score_numba(returns, self.window_sadf)

        # CUSUM Scores (Numba Optimized)
        mean_ret = np.mean(returns)
        cusum_scores = rolling_cusum_scores_numba(returns, mean_ret)

        # Normalize
        max_sadf = np.max(sadf_scores)
        sadf_norm = sadf_scores / (max_sadf + 1e-9) if max_sadf > 0 else sadf_scores

        max_cusum = np.max(cusum_scores)
        cusum_norm = cusum_scores / (max_cusum + 1e-9) if max_cusum > 0 else cusum_scores

        features = pd.DataFrame(index=df.index)
        features['volatility'] = volatility
        features['sadf_score_norm'] = sadf_norm
        features['cusum_score_norm'] = cusum_norm

        # Integrate Anti-Explosion Features
        try:
            processed = apply_layer2_price_processing(df, price_col=raw_price_col, enable_price_features=True)
            # Select new features
            new_cols = [c for c in processed.columns if c not in df.columns]
            if new_cols:
                features = pd.concat([features, processed[new_cols]], axis=1)
                # tprint_info(f"   ✨ Layer 4: Added {len(new_cols)} Anti-Explosion features")
        except Exception as e:
            tprint_warning(f"   ⚠️ Layer 4: Anti-Explosion feature generation failed: {e}")

        return features


class ModelPerformanceFeatures:
    """
    Generates features based on the historical performance of the base models.
    Includes 'Skill' metrics and 'Risk-off' pressure indicators.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Configuration for Skill Feature
        self.skill_window = self.config.get('perf_skill_window', 100)  # Heavy smoothing
        self.skill_embargo = self.config.get('perf_skill_embargo', 2)  # Strict embargo

        # Configuration for Risk-off Feature
        self.risk_window = self.config.get('perf_risk_window', 50)
        self.risk_embargo = self.config.get('perf_risk_embargo', 2)
        self.dd_scaling = self.config.get('perf_dd_scaling', 0.1) # 'c' in tanh(dd/c)

    def generate(self, df: pd.DataFrame, pred_col: str, target_col: str) -> pd.DataFrame:
        if pred_col not in df.columns or target_col not in df.columns:
            return pd.DataFrame(index=df.index)

        preds = df[pred_col]
        targets = df[target_col]

        # 1. Skill Feature Components
        pred_dir = np.sign(preds - 0.5)
        strat_ret = pred_dir * targets
        hits = ((pred_dir * np.sign(targets)) > 0).astype(float)
        rolling_ic = preds.rolling(window=self.skill_window).corr(targets).fillna(0)

        # Apply Embargo
        strat_ret_shifted = strat_ret.shift(self.skill_embargo)
        hits_shifted = hits.shift(self.skill_embargo)
        rolling_ic_shifted = rolling_ic.shift(self.skill_embargo)

        # Smooth
        ewma_ret = strat_ret_shifted.ewm(span=self.skill_window, adjust=False).mean()
        ewma_hit = hits_shifted.ewm(span=self.skill_window, adjust=False).mean()
        ewma_ic = rolling_ic_shifted.ewm(span=self.skill_window, adjust=False).mean()

        # Skill Feature
        skill_raw = ewma_ret * ewma_hit * ewma_ic
        skill_score = np.tanh(skill_raw * 100.0)

        # 2. Risk-off Pressure Components
        cum_ret = strat_ret.cumsum().fillna(0)
        high_water_mark = cum_ret.expanding().max()
        drawdown = high_water_mark - cum_ret
        residual = np.abs(targets)

        drawdown_shifted = drawdown.shift(self.risk_embargo)
        residual_shifted = residual.shift(self.risk_embargo)

        dd_state = np.tanh(drawdown_shifted / self.dd_scaling)
        ewma_resid = residual_shifted.ewm(span=self.risk_window, adjust=False).mean()
        risk_off_pressure = dd_state * ewma_resid

        features = pd.DataFrame(index=df.index)
        features['perf_skill_score'] = skill_score.fillna(0)
        features['perf_risk_off_pressure'] = risk_off_pressure.fillna(0)

        return features


def train_layer4_simple_multimodel(
    oof_df: pd.DataFrame,
    market_data: pd.DataFrame,
    base_model_predictions: Optional[pd.DataFrame] = None,
    target_col: str = 'realized_return',
    n_folds: int = 5,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train Layer 4 using Layer 3 probability outputs, entropy bars, and market features.

    Integrated with Layer 4 Checkpoint Manager for robust state saving and resuming.
    """
    cfg = config or {}
    symbol = cfg.get('symbol', 'ETHUSDT')
    exchange = cfg.get('exchange', 'binance')
    
    tprint_info(f"🚀 Starting Layer 4 Training (Checkpoint Aware) for {symbol}...")
    
    # Initialize Checkpoint Manager
    manager = get_layer4_checkpoint_manager(symbol)
    start_step = manager.get_auto_resume_step(symbol)
    available_steps = manager.get_available_steps()
    
    # Helper to check if we should run a step
    def should_run(step_name):
        return available_steps.index(step_name) >= available_steps.index(start_step)

    # State variables (to be populated by running or loading)
    entropy_bars_df = pd.DataFrame()
    market_features = pd.DataFrame()
    oof_bet_sizes = None
    engine = SimpleMultiModelRiskEngine()
    final_metrics = {}
    oof_df_out = None
    
    # --- Step 0: Data Preparation ---
    step_name = 'data_preparation'
    if should_run(step_name):
        tprint_info(f"📍 Executing {step_name}...")

        # Integrate entropy bars if enabled
        if cfg.get('use_entropy_bars', True):
            oof_df, entropy_bars_df = integrate_entropy_bars_into_layer4(oof_df, symbol, exchange, cfg)
            cfg['entropy_bars_df'] = entropy_bars_df
        else:
            tprint_info("⏭️ Layer 4: Skipping entropy bars (disabled in config)")
            entropy_bars_df = pd.DataFrame()

        manager.save_checkpoint(step_name, {
            'meta_df': oof_df, # Naming convention from manager
            'entropy_bars_df': entropy_bars_df,
            'market_data': market_data
        }, symbol, cfg)
    else:
        tprint_info(f"⏭️ Resuming past {step_name}...")
        data = manager.load_checkpoint(step_name, symbol)
        if data:
            oof_df = data.get('meta_df', oof_df)
            entropy_bars_df = data.get('entropy_bars_df', pd.DataFrame())
            market_data = data.get('market_data', market_data)

    # --- Step 1: Confidence Filtering ---
    step_name = 'confidence_filtering'
    if should_run(step_name):
        tprint_info(f"📍 Executing {step_name}...")
        # Currently a pass-through (or logic could be added here)
        filtered_df = oof_df
        manager.save_checkpoint(step_name, {'filtered_meta_df': filtered_df}, symbol, cfg)
    else:
        tprint_info(f"⏭️ Resuming past {step_name}...")
        data = manager.load_checkpoint(step_name, symbol)
        if data:
            oof_df = data.get('filtered_meta_df', oof_df)

    # --- Step 2: Feature Engineering ---
    step_name = 'feature_engineering'
    if should_run(step_name):
        tprint_info(f"📍 Executing {step_name}...")

        # Generate market features
        generator = MetaLearnerFeatures(config=config)
        mkt_feats = generator.generate(
            df=oof_df.join(market_data, how='left', rsuffix='_mkt'),
            raw_price_col='close'
        )

        # Generate model performance features
        perf_generator = ModelPerformanceFeatures(config=config)
        pred_col = 'meta_prob' if 'meta_prob' in oof_df.columns else None

        if pred_col:
            perf_feats = perf_generator.generate(
                df=oof_df,
                pred_col=pred_col,
                target_col=target_col
            )
            market_features = pd.concat([mkt_feats, perf_feats], axis=1)
        else:
            tprint_warning("⚠️ Layer 4: 'meta_prob' not found, skipping performance features.")
            market_features = mkt_feats

        manager.save_checkpoint(step_name, {'market_features': market_features}, symbol, cfg)
    else:
        tprint_info(f"⏭️ Resuming past {step_name}...")
        data = manager.load_checkpoint(step_name, symbol)
        if data:
            market_features = data.get('market_features', pd.DataFrame())

    # --- Step 3: Gate Model Training (and OOF generation) ---
    step_name = 'gate_model_training'
    if should_run(step_name):
        tprint_info(f"📍 Executing {step_name}...")

        y_binary = (oof_df[target_col] > 0).astype(int)
        abs_returns = oof_df[target_col].abs()

        kf = KFold(n_splits=n_folds, shuffle=False)
        oof_bet_sizes = np.zeros(len(oof_df))

        # 1. CV Loop for OOF predictions
        tprint_info("🔄 Running Cross-Validation for OOF predictions...")
        # We use a temp engine for CV to not dirty the final engine state
        cv_engine = SimpleMultiModelRiskEngine()

        for train_idx, val_idx in kf.split(oof_df):
            cv_engine.train(
                df=oof_df.iloc[train_idx],
                market_features=market_features.iloc[train_idx],
                y_true=y_binary.iloc[train_idx],
                abs_returns=abs_returns.iloc[train_idx]
            )
            oof_bet_sizes[val_idx] = cv_engine.predict_bet_size(
                df=oof_df.iloc[val_idx],
                market_features=market_features.iloc[val_idx]
            )

        # 2. Final Fit on full data
        tprint_info("🧠 Training Final Gate Model...")
        final_metrics = engine.train(oof_df, market_features, y_binary, abs_returns)

        # Save state
        # Note: gate_models usually expects a dict of models, here we pass the engine state
        # The checkpoint manager is flexible with dicts
        manager.save_checkpoint(step_name, {
            'gate_models': engine.get_state(),
            'oof_predictions': oof_bet_sizes,
            'training_metrics': final_metrics
        }, symbol, cfg)
    else:
        tprint_info(f"⏭️ Resuming past {step_name}...")
        data = manager.load_checkpoint(step_name, symbol)
        if data:
            if 'gate_models' in data:
                engine.load_state(data['gate_models'])
            if 'oof_predictions' in data:
                oof_bet_sizes = data['oof_predictions']
            if 'training_metrics' in data:
                final_metrics = data['training_metrics']

    # --- Step 4: Gate Validation ---
    step_name = 'gate_validation'
    if should_run(step_name):
        tprint_info(f"📍 Executing {step_name}...")
        # Since we calculated OOF predictions in training, we can validate here
        # (Or recalculate metrics)
        manager.save_checkpoint(step_name, {'validation_metrics': final_metrics}, symbol, cfg)
    else:
        # Load metrics if needed
        pass

    # --- Step 5: Final Predictions ---
    step_name = 'final_predictions'
    if should_run(step_name):
        tprint_info(f"📍 Executing {step_name}...")

        oof_df_out = oof_df.copy()
        if oof_bet_sizes is not None:
            oof_df_out['layer4_prob'] = oof_bet_sizes
        else:
            tprint_warning("⚠️ No OOF predictions found for final output.")
            oof_df_out['layer4_prob'] = 0.5 # Fallback

        # Add entropy bars information to metrics
        if not entropy_bars_df.empty:
            final_metrics['entropy_bars_count'] = len(entropy_bars_df)
            final_metrics['entropy_features_count'] = len([col for col in oof_df_out.columns if col.startswith(('staleness_', 'drift_', 'lz_', 'trend_', 'entropy_', 'proxy_entropy'))])

        manager.save_checkpoint(step_name, {
            'final_predictions': oof_df_out,
            'final_metrics': final_metrics
        }, symbol, cfg)

        # Also save as Artifact
        manager.save_checkpoint('artifact_saving', {
            'model_state': engine.get_state(),
            'final_predictions': oof_df_out,
            'final_metrics': final_metrics
        }, symbol, cfg)

    else:
        tprint_info(f"⏭️ Resuming past {step_name}...")
        data = manager.load_checkpoint(step_name, symbol)
        if data:
            oof_df_out = data.get('final_predictions', pd.DataFrame())
            final_metrics = data.get('final_metrics', {})
    
    return oof_df_out, final_metrics
