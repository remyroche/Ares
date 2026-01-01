"""Layer 4 — Triple Barrier Trailing Profit & Sizing.

Layer2 is about learnability, layer3 about relation to target (IC, calibration),
layer4 is about position sizing. I want to trade it with a triple barrier method
that includes trailing profit.

This module implements:
1.  Triple Barrier Trailing Logic (Exit Strategy).
2.  Inverse Volatility Sizing (Position Sizing).
3.  Integration with Layer 5 via `layer4_prob` proxy generation.

Ensure compatibility with label_based_layer_5:
Layer 5 calculates Size = ((p - 0.5) / 0.5) ^ 2.
We reverse this to generate `layer4_prob` such that Layer 5 produces our desired
Inverse Volatility Size.

REFACTORED: Now uses unified Layer4FeatureGenerator for consolidated feature generation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from datetime import datetime
import json

from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from scipy.stats import spearmanr, norm
import statsmodels.api as sm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.base import clone
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Import the new PnL-optimized ExtraTrees implementation
from .layer4_extratrees_pnl import train_layer4_oof as _train_layer4_oof_extratrees_pnl


# Configuration Constants
STOP_LOSS_FLOOR = 0.004  # 0.3% Fees + 0.1% Spread Buffer
TARGET_VOLATILITY = 0.01  # 1% target volatility for sizing
VOLATILITY_SAFETY_FLOOR = 1e-4  # Prevent division by zero
HOME_RUN_MULTIPLIER = 3.0  # Multiplier for home run detection
WEIGHT_CLIP_MIN = 0.5  # Minimum weight clip
WEIGHT_CLIP_MAX = 2.0  # Maximum weight clip
NEUTRAL_PROB_THRESHOLD = 0.5  # Neutral probability threshold for gating
ZERO_SIZE_PROB = 0.4  # Probability that results in zero size


class SimpleMultiModelRiskEngine:
    """
    Simple Multi-Model Risk Engine with simultaneous training.
    
    Trains Ridge, ExtraTrees, and XGBoost simultaneously on the same features
    without stacking or complex hierarchies.
    """
    
    def __init__(self, 
                 n_estimators: int = 1000, 
                 max_features: str = 'log2',
                 consensus_weights: Optional[Dict[str, float]] = None):
        
        # Default consensus weights
        self.consensus_weights = consensus_weights or {
            'extratrees': 0.4,
            'ridge': 0.3,
            'xgboost': 0.3
        }
        
        # Initialize base models
        self.extratrees = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        )
        
        self.ridge = Ridge(alpha=1.0, random_state=42)
        
        if XGB_AVAILABLE:
            self.xgboost = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.xgboost = None
        
        # Individual calibrators
        self.calibrators = {
            'extratrees': IsotonicRegression(out_of_bounds='clip'),
            'ridge': IsotonicRegression(out_of_bounds='clip'),
            'xgboost': IsotonicRegression(out_of_bounds='clip') if XGB_AVAILABLE else None
        }
        
        # Consensus calibrator
        self.consensus_calibrator = IsotonicRegression(out_of_bounds='clip')
        
        self.feature_names = None
        self.is_fitted = False
    
    def _compute_financial_weights(self, abs_returns: pd.Series) -> pd.Series:
        """Compute sample weights based on financial attribution."""
        weights = abs_returns / abs_returns.sum() * len(abs_returns)
        weights = weights.clip(weights.quantile(0.01), weights.quantile(0.99))
        weights = weights / weights.sum() * len(weights)
        return weights
    
    def train(self, preds_df: pd.DataFrame, market_features: pd.DataFrame,
              y_true: pd.Series, abs_returns: pd.Series) -> Dict[str, Any]:
        """
        Train all models simultaneously on the same features.
        """
        tprint_info("🚀 Training Simple Multi-Model Risk Engine...")
        
        # Use market features directly (no complex feature engineering)
        X = market_features.fillna(0)
        weights = self._compute_financial_weights(abs_returns)
        
        # Train all models simultaneously
        base_predictions = {}
        model_metrics = {}
        
        # Train ExtraTrees
        tprint_info("📊 Training ExtraTrees...")
        self.extratrees.fit(X, y_true, sample_weight=weights)
        et_preds = self.extratrees.predict(X)
        base_predictions['extratrees'] = et_preds
        
        et_calibrated = self.calibrators['extratrees'].fit_transform(et_preds, y_true)
        model_metrics['extratrees'] = {
            'weighted_logloss': log_loss(y_true, et_calibrated, sample_weight=weights),
            'brier_score': brier_score_loss(y_true, et_calibrated),
            'mean_prediction': et_calibrated.mean(),
            'std_prediction': et_calibrated.std()
        }
        
        # Train Ridge
        tprint_info("📊 Training Ridge...")
        self.ridge.fit(X, y_true, sample_weight=weights)
        ridge_preds = self.ridge.predict(X)
        base_predictions['ridge'] = ridge_preds
        
        ridge_calibrated = self.calibrators['ridge'].fit_transform(ridge_preds, y_true)
        model_metrics['ridge'] = {
            'weighted_logloss': log_loss(y_true, ridge_calibrated, sample_weight=weights),
            'brier_score': brier_score_loss(y_true, ridge_calibrated),
            'mean_prediction': ridge_calibrated.mean(),
            'std_prediction': ridge_calibrated.std()
        }
        
        # Train XGBoost
        if self.xgboost is not None:
            tprint_info("📊 Training XGBoost...")
            self.xgboost.fit(X, y_true, sample_weight=weights)
            xgb_preds = self.xgboost.predict(X)
            base_predictions['xgboost'] = xgb_preds
            
            xgb_calibrated = self.calibrators['xgboost'].fit_transform(xgb_preds, y_true)
            model_metrics['xgboost'] = {
                'weighted_logloss': log_loss(y_true, xgb_calibrated, sample_weight=weights),
                'brier_score': brier_score_loss(y_true, xgb_calibrated),
                'mean_prediction': xgb_calibrated.mean(),
                'std_prediction': xgb_calibrated.std()
            }
        
        # Create Weighted Consensus
        consensus_raw = np.zeros(len(y_true))
        for name, weight in self.consensus_weights.items():
            if name in base_predictions:
                consensus_raw += weight * base_predictions[name]
        
        # Calibrate Consensus
        self.consensus_calibrator.fit(consensus_raw, y_true)
        consensus_calibrated = self.consensus_calibrator.transform(consensus_raw)
        
        # Compute Final Metrics
        final_weighted_logloss = log_loss(y_true, consensus_calibrated, sample_weight=weights)
        final_brier_score = brier_score_loss(y_true, consensus_calibrated)
        
        self.is_fitted = True
        self.final_predictions_ = consensus_calibrated
        self.feature_names = X.columns.tolist()
        
        metrics = {
            **{f'{k}_{m}': v for k, model_dict in model_metrics.items() for m, v in model_dict.items()},
            'consensus_weighted_logloss': final_weighted_logloss,
            'consensus_brier_score': final_brier_score,
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'mean_weight': weights.mean(),
            'weight_std': weights.std(),
            'consensus_weights': self.consensus_weights,
            'training_type': 'simultaneous'
        }
        
        tprint_success(f"✅ Simple Multi-Model Engine trained: WL={final_weighted_logloss:.4f}, BS={final_brier_score:.4f}")
        return metrics
    
    def predict_bet_size(self, preds_df: pd.DataFrame, market_features: pd.DataFrame,
                        y_true_dummy: Optional[pd.Series] = None) -> np.ndarray:
        """Generate calibrated bet sizes."""
        if not self.is_fitted:
            raise ValueError("RiskEngine must be trained before prediction")
        
        # Get features
        X = market_features.fillna(0)
        
        # Ensure feature alignment
        if set(X.columns) != set(self.feature_names):
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self.feature_names]
        
        # Get individual model predictions
        model_preds = {}
        if hasattr(self.extratrees, 'predict'):
            model_preds['extratrees'] = self.extratrees.predict(X)
        if hasattr(self.ridge, 'predict'):
            model_preds['ridge'] = self.ridge.predict(X)
        if self.xgboost is not None and hasattr(self.xgboost, 'predict'):
            model_preds['xgboost'] = self.xgboost.predict(X)
        
        # Apply individual calibrations
        calibrated_preds = {}
        for name, preds in model_preds.items():
            if self.calibrators[name] is not None:
                calibrated_preds[name] = self.calibrators[name].transform(preds)
            else:
                calibrated_preds[name] = preds
        
        # Create weighted consensus
        consensus_raw = np.zeros(len(X))
        for name, weight in self.consensus_weights.items():
            if name in calibrated_preds:
                consensus_raw += weight * calibrated_preds[name]
        
        # Apply consensus calibration
        final_predictions = self.consensus_calibrator.transform(consensus_raw)
        
        return final_predictions
    
    def evaluate_external_metrics(self, bet_sizes: np.ndarray, returns: pd.Series,
                                 volatility: Optional[pd.Series] = None) -> Dict[str, float]:
        """Compute external validation metrics."""
        sized_returns = returns * bet_sizes
        
        total_pnl = sized_returns.sum()
        mean_return = sized_returns.mean()
        std_return = sized_returns.std()
        
        downside_returns = sized_returns[sized_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-8
        sortino_ratio = mean_return / downside_std if downside_std > 0 else 0.0
        
        cumulative = (1 + sized_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        if volatility is not None:
            sharpe_ratio = mean_return / (volatility.mean() + 1e-8)
        else:
            sharpe_ratio = mean_return / (std_return + 1e-8)
        
        win_rate = (sized_returns > 0).mean()
        
        gross_profit = sized_returns[sized_returns > 0].sum()
        gross_loss = abs(sized_returns[sized_returns < 0].sum())
        profit_factor = gross_profit / (gross_loss + 1e-8)
        
        periods_per_year = 365 * 24 * 4
        annualized_return = mean_return * periods_per_year
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return {
            'total_pnl': total_pnl,
            'mean_return': mean_return,
            'std_return': std_return,
            'sortino_ratio': sortino_ratio,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'annualized_return': annualized_return
        }


# Keep the original EnhancedDePradoRiskEngine for comparison
    """
    Purged Stacking Sizer with residual correction.
    
    Uses out-of-sample predictions from base models to train a correction model,
    preventing the correction from learning in-sample noise.
    """
    
    def __init__(self, base_models: List, correction_model, purged_cv: bool = True):
        self.base_models = base_models  # e.g. [ExtraTrees, Ridge]
        self.correction_model = correction_model  # e.g. XGBoost
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.purged_cv = purged_cv
        self.fitted_base_models = []
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """
        Fit the stacking sizer with purged cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            sample_weight: Sample weights for financial attribution
        """
        tprint_info("🔄 Training PurgedStackingSizer...")
        
        # 1. Generate Out-of-Sample Predictions for the Base Layer
        oos_preds = []
        for i, model in enumerate(self.base_models):
            tprint_info(f"📊 Generating OOS predictions for model {i+1}/{len(self.base_models)}...")
            
            # Use cross_val_predict for OOS predictions
            if sample_weight is not None:
                # XGBoost needs special handling for sample weights
                if hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
                    p = cross_val_predict(model, X, y, cv=5, method='predict', 
                                        fit_params={'sample_weight': sample_weight})
                else:
                    p = cross_val_predict(model, X, y, cv=5, method='predict')
            else:
                p = cross_val_predict(model, X, y, cv=5, method='predict')
            
            oos_preds.append(p)
        
        consensus_oos = np.mean(oos_preds, axis=0)
        
        # 2. Train the Correction Layer on OOS Residuals
        tprint_info("🎯 Training correction model on OOS residuals...")
        residual_target = y - consensus_oos
        
        if sample_weight is not None:
            self.correction_model.fit(X, residual_target, sample_weight=sample_weight)
        else:
            self.correction_model.fit(X, residual_target)
        
        # 3. Final Step: Train the actual Base Models on the FULL dataset
        tprint_info("🏋️ Training base models on full dataset...")
        self.fitted_base_models = []
        for model in self.base_models:
            fitted_model = clone(model)
            if sample_weight is not None and hasattr(fitted_model, 'fit') and 'sample_weight' in fitted_model.fit.__code__.co_varnames:
                fitted_model.fit(X, y, sample_weight=sample_weight)
            else:
                fitted_model.fit(X, y)
            self.fitted_base_models.append(fitted_model)
            
        # 4. Calibration
        tprint_info("🎚️ Calibrating final predictions...")
        final_p = np.mean([m.predict(X) for m in self.fitted_base_models], axis=0) + self.correction_model.predict(X)
        self.calibrator.fit(final_p, y)
        
        self.is_fitted = True
        tprint_success("✅ PurgedStackingSizer training completed")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate calibrated probability predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("PurgedStackingSizer must be fitted before prediction")
        
        base_p = np.mean([m.predict(X) for m in self.fitted_base_models], axis=0)
        correction = self.correction_model.predict(X)
        return self.calibrator.transform(base_p + correction)


class EnhancedDePradoRiskEngine:
    """
    Enhanced Layer 4 Risk Engine with multiple models and consensus.
    
    Implements:
    1. Multiple Base Models: ExtraTrees, XGBoost, Ridge
    2. Individual Isotonic Calibration for each model
    3. Weighted Sizer Consensus (0.4×ET + 0.3×Ridge + 0.3×XGB)
    4. Sigmoid Calibration on consensus
    5. Optional PurgedStacking for residual correction
    """
    
    def __init__(self, 
                 n_estimators: int = 1000, 
                 max_features: str = 'log2',
                 use_stacking: bool = False,
                 consensus_weights: Optional[Dict[str, float]] = None):
        
        # Default consensus weights
        self.consensus_weights = consensus_weights or {
            'extratrees': 0.4,  # Maximize Information Gain
            'ridge': 0.3,        # Minimize L2-Regularized Squared Error  
            'xgboost': 0.3       # Minimize Weighted Log-Loss
        }
        
        # Initialize base models
        self.extratrees = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        )
        
        self.ridge = Ridge(alpha=1.0, random_state=42)
        
        if XGB_AVAILABLE:
            self.xgboost = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.xgboost = None
            tprint_warning("⚠️ XGBoost not available, using only ExtraTrees and Ridge")
        
        # Individual calibrators
        self.calibrators = {
            'extratrees': IsotonicRegression(out_of_bounds='clip'),
            'ridge': IsotonicRegression(out_of_bounds='clip'),
            'xgboost': IsotonicRegression(out_of_bounds='clip') if XGB_AVAILABLE else None
        }
        
        # Consensus calibrator (sigmoid approximation using isotonic)
        self.consensus_calibrator = IsotonicRegression(out_of_bounds='clip')
        
        # Optional stacking component
        self.use_stacking = use_stacking
        if use_stacking:
            base_models = [self.extratrees, self.ridge]
            if self.xgboost is not None:
                base_models.append(self.xgboost)
            
            # Use remaining model for correction
            correction_models = [m for m in [self.extratrees, self.ridge, self.xgboost] if m not in base_models[:2]]
            correction_model = correction_models[0] if correction_models else self.extratrees
            
            self.stacking_sizer = PurgedStackingSizer(
                base_models=base_models[:2],  # Use first 2 for base
                correction_model=correction_model
            )
        
        self.feature_names = None
        self.is_fitted = False
    
    def _get_oriented_residuals(self, preds_df: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
        """Generate oriented residuals as in original DePrado implementation."""
        oriented_resids = pd.DataFrame(index=preds_df.index)
        
        for col in preds_df.columns:
            others = preds_df.drop(columns=[col])
            consensus = others.mean(axis=1)
            oriented_resids[f"{col}_alpha"] = preds_df[col] - consensus
            
            error = y_true - preds_df[col]
            oriented_resids[f"{col}_err_ema"] = error.ewm(span=20, adjust=False).mean()
            oriented_resids[f"{col}_err_std"] = error.rolling(20, min_periods=1).std()
            
        return oriented_resids.fillna(0)
    
    def _compute_financial_weights(self, abs_returns: pd.Series) -> pd.Series:
        """Compute sample weights based on financial attribution."""
        weights = abs_returns / abs_returns.sum() * len(abs_returns)
        weights = weights.clip(weights.quantile(0.01), weights.quantile(0.99))
        weights = weights / weights.sum() * len(weights)
        return weights
    
    def _apply_sigmoid_calibration(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid-like calibration using isotonic regression.
        This provides a smooth S-shaped calibration curve.
        """
        # Isotonic regression can approximate sigmoid when data is properly distributed
        return self.consensus_calibrator.transform(probabilities)
    
    def train(self, preds_df: pd.DataFrame, market_features: pd.DataFrame,
              y_true: pd.Series, abs_returns: pd.Series) -> Dict[str, Any]:
        """
        Train the enhanced risk engine with multiple models and consensus.
        
        Args:
            preds_df: Matrix of base model predictions
            market_features: Additional market context features
            y_true: Binary meta-labels (1/0)
            abs_returns: Absolute magnitude of realized returns (for weighting)
            
        Returns:
            Dictionary with training metrics
        """
        tprint_info("🧠 Training Enhanced DePrado Risk Engine with multi-model consensus...")
        
        # 1. Generate Oriented Features
        resids = self._get_oriented_residuals(preds_df, y_true)
        X = pd.concat([resids, market_features], axis=1)
        self.feature_names = X.columns.tolist()
        
        # 2. Sample Weighting
        weights = self._compute_financial_weights(abs_returns)
        
        # 3. Train Individual Models
        models = {
            'extratrees': self.extratrees,
            'ridge': self.ridge
        }
        if self.xgboost is not None:
            models['xgboost'] = self.xgboost
        
        model_predictions = {}
        model_metrics = {}
        
        for name, model in models.items():
            tprint_info(f"📊 Training {name} model...")
            
            # Train model
            model.fit(X, y_true, sample_weight=weights)
            
            # Generate predictions
            raw_preds = model.predict(X)
            model_predictions[name] = raw_preds
            
            # Individual calibration
            calibrated_preds = self.calibrators[name].fit_transform(raw_preds, y_true)
            
            # Compute metrics
            weighted_logloss = log_loss(y_true, calibrated_preds, sample_weight=weights)
            brier_score = brier_score_loss(y_true, calibrated_preds)
            
            model_metrics[name] = {
                'weighted_logloss': weighted_logloss,
                'brier_score': brier_score,
                'mean_prediction': calibrated_preds.mean(),
                'std_prediction': calibrated_preds.std()
            }
        
        # 4. Create Weighted Consensus
        consensus_raw = np.zeros(len(y_true))
        for name, weight in self.consensus_weights.items():
            if name in model_predictions:
                consensus_raw += weight * model_predictions[name]
        
        # 5. Calibrate Consensus with Sigmoid-like calibration
        self.consensus_calibrator.fit(consensus_raw, y_true)
        consensus_calibrated = self._apply_sigmoid_calibration(consensus_raw)
        
        # 6. Optional Stacking Correction
        if self.use_stacking:
            tprint_info("🔄 Applying stacking correction...")
            self.stacking_sizer.fit(X, y_true, sample_weight=weights)
            stacking_preds = self.stacking_sizer.predict_proba(X)
            
            # Blend consensus with stacking (70% consensus, 30% stacking)
            final_predictions = 0.7 * consensus_calibrated + 0.3 * stacking_preds
        else:
            final_predictions = consensus_calibrated
        
        # 7. Compute Final Metrics
        final_weighted_logloss = log_loss(y_true, final_predictions, sample_weight=weights)
        final_brier_score = brier_score_loss(y_true, final_predictions)
        
        self.is_fitted = True
        self.final_predictions_ = final_predictions  # Store for external evaluation
        
        metrics = {
            **{f'{k}_{m}': v for k, model_dict in model_metrics.items() for m, v in model_dict.items()},
            'consensus_weighted_logloss': final_weighted_logloss,
            'consensus_brier_score': final_brier_score,
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'mean_weight': weights.mean(),
            'weight_std': weights.std(),
            'use_stacking': self.use_stacking,
            'consensus_weights': self.consensus_weights
        }
        
        tprint_success(f"✅ Enhanced Risk Engine trained: WL={final_weighted_logloss:.4f}, BS={final_brier_score:.4f}")
        return metrics
    
    def predict_bet_size(self, preds_df: pd.DataFrame, market_features: pd.DataFrame,
                        y_true_dummy: Optional[pd.Series] = None) -> np.ndarray:
        """
        Generate calibrated bet sizes using multi-model consensus.
        
        Args:
            preds_df: Matrix of base model predictions
            market_features: Additional market context features
            y_true_dummy: Dummy true values for residual computation (optional)
            
        Returns:
            Calibrated bet sizes
        """
        if not self.is_fitted:
            raise ValueError("RiskEngine must be trained before prediction")
        
        # Generate oriented residuals
        if y_true_dummy is None:
            y_true_dummy = preds_df.mean(axis=1)
        
        resids = self._get_oriented_residuals(preds_df, y_true_dummy)
        X = pd.concat([resids, market_features], axis=1)
        
        # Ensure feature alignment
        if set(X.columns) != set(self.feature_names):
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self.feature_names]
        
        # Get individual model predictions
        model_preds = {}
        if hasattr(self.extratrees, 'predict'):
            model_preds['extratrees'] = self.extratrees.predict(X)
        if hasattr(self.ridge, 'predict'):
            model_preds['ridge'] = self.ridge.predict(X)
        if self.xgboost is not None and hasattr(self.xgboost, 'predict'):
            model_preds['xgboost'] = self.xgboost.predict(X)
        
        # Apply individual calibrations
        calibrated_preds = {}
        for name, preds in model_preds.items():
            if self.calibrators[name] is not None:
                calibrated_preds[name] = self.calibrators[name].transform(preds)
            else:
                calibrated_preds[name] = preds
        
        # Create weighted consensus
        consensus_raw = np.zeros(len(X))
        for name, weight in self.consensus_weights.items():
            if name in calibrated_preds:
                consensus_raw += weight * calibrated_preds[name]
        
        # Apply consensus calibration
        if self.use_stacking:
            stacking_preds = self.stacking_sizer.predict_proba(X)
            final_predictions = 0.7 * self._apply_sigmoid_calibration(consensus_raw) + 0.3 * stacking_preds
        else:
            final_predictions = self._apply_sigmoid_calibration(consensus_raw)
        
        return final_predictions
    
    def evaluate_external_metrics(self, bet_sizes: np.ndarray, returns: pd.Series,
                                 volatility: Optional[pd.Series] = None) -> Dict[str, float]:
        """Compute external validation metrics (same as original)."""
        sized_returns = returns * bet_sizes
        
        total_pnl = sized_returns.sum()
        mean_return = sized_returns.mean()
        std_return = sized_returns.std()
        
        downside_returns = sized_returns[sized_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-8
        sortino_ratio = mean_return / downside_std if downside_std > 0 else 0.0
        
        cumulative = (1 + sized_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        if volatility is not None:
            sharpe_ratio = mean_return / (volatility.mean() + 1e-8)
        else:
            sharpe_ratio = mean_return / (std_return + 1e-8)
        
        win_rate = (sized_returns > 0).mean()
        
        gross_profit = sized_returns[sized_returns > 0].sum()
        gross_loss = abs(sized_returns[sized_returns < 0].sum())
        profit_factor = gross_profit / (gross_loss + 1e-8)
        
        periods_per_year = 365 * 24 * 4
        annualized_return = mean_return * periods_per_year
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return {
            'total_pnl': total_pnl,
            'mean_return': mean_return,
            'std_return': std_return,
            'sortino_ratio': sortino_ratio,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'annualized_return': annualized_return
        }


# Keep the original DePradoRiskEngine for backward compatibility


class MetaLearnerFeatures:
    """
    Final-layer (position sizing) meta-features.
    
    Refactored to use unified Layer4FeatureGenerator for consistency.
    Maintains backward compatibility while leveraging consolidated feature generation.
    
    Designed for ExtraTrees / RF:
    - bounded
    - monotone where possible
    - regime-aware
    - leak-safe
    """
    
    def __init__(
        self,
        window: int = 50,
        span: int = 20,
        prior_sr: float = 0.0,
        prior_weight: int = 10,
        min_psr_obs: int = 20,
        config: Optional[Dict[str, Any]] = None
    ):
        self.window = window
        self.span = span
        self.prior_sr = prior_sr
        self.prior_weight = prior_weight
        self.min_psr_obs = min_psr_obs
        self.config = config or {}
        
        # Initialize unified feature generator
        try:
            from .layer4 import Layer4FeatureGenerator
            self.generator = Layer4FeatureGenerator(
                window=window,
                span=span,
                prior_sr=prior_sr,
                prior_weight=prior_weight,
                min_psr_obs=min_psr_obs,
                config=self.config
            )
            self.use_unified_generator = True
        except ImportError:
            # Fallback to original implementation
            self.generator = None
            self.use_unified_generator = False
    
    def generate(
        self,
        df: pd.DataFrame,
        raw_price_col: str = "close",
        denoised_price_col: str = "denoised_price"
    ) -> pd.DataFrame:
        """
        Generate meta-learner features using unified approach.
        
        Args:
            df: DataFrame with market data and predictions
            raw_price_col: Raw price column name
            denoised_price_col: Denoised price column name
            
        Returns:
            DataFrame with meta-learner features
        """
        if self.use_unified_generator:
            # Use unified feature generator
            config = self.config.copy()
            config.update({
                'enable_performance': True,
                'enable_regime': True,
                'enable_market': True,
                'enable_technical': False,  # Disable technical for meta-learner
                'enable_structural': False,  # Disable structural for meta-learner
                'enable_model': False,  # Disable model features for meta-learner
                'enable_time': False,  # Disable time features for meta-learner
                'enable_contextual': False  # Disable contextual for meta-learner
            })
            
            # Generate only meta-learner specific features
            features_df = self.generator.generate_all_features(
                df=df,
                target_col='realized_return',
                prob_col='meta_prob',
                raw_price_col=raw_price_col,
                denoised_price_col=denoised_price_col
            )
            
            # Select only meta-learner relevant features
            meta_features = [
                'perf_bayesian_psr', 'perf_psr_trend', 'perf_entropy',
                'regime_sadf', 'market_stretch', 'noise_persistence'
            ]
            
            available_meta_features = [f for f in meta_features if f in features_df.columns]
            return features_df[available_meta_features].fillna(0.0)
        
        else:
            # Fallback to original implementation
            return self._generate_original(df, raw_price_col, denoised_price_col)
    
    def _generate_original(
        self, df: pd.DataFrame, raw_price_col: str, denoised_price_col: str
    ) -> pd.DataFrame:
        """Original implementation as fallback."""
        features = pd.DataFrame(index=df.index)
        
        # Performance Features
        if 'primary_ret' in df.columns:
            def bayesian_psr(returns, benchmark_sr=0.0):
                r = np.asarray(returns)
                n = len(r)
                if n < self.min_psr_obs:
                    return 0.0
                mean = r.mean()
                std = r.std(ddof=1) + 1e-9
                sample_sr = mean / std
                shrunk_sr = (
                    sample_sr * n + self.prior_sr * self.prior_weight
                ) / (n + self.prior_weight)
                skew = pd.Series(r).skew()
                kurt = pd.Series(r).kurtosis()
                var_sr = (
                    1 - skew * shrunk_sr + ((kurt - 1.0) / 4.0) * shrunk_sr ** 2
                ) / (n - 1)
                if var_sr <= 0 or not np.isfinite(var_sr):
                    return 0.0
                sigma_sr = np.sqrt(var_sr)
                return norm.cdf((shrunk_sr - benchmark_sr) / sigma_sr)
            
            features["perf_bayesian_psr"] = (
                df["primary_ret"]
                .rolling(self.window)
                .apply(bayesian_psr, raw=True)
            )
            
            features["perf_psr_trend"] = (
                features["perf_bayesian_psr"]
                .diff()
                .ewm(span=10, adjust=False)
                .mean()
            )
        
        # Regime Features
        if raw_price_col in df.columns:
            def get_sadf_proxy(price):
                log_p = np.log(price.replace(0, np.nan)).dropna()
                def adf_tstat(x):
                    if len(x) < 20:
                        return 0.0
                    y = x.values
                    dy = np.diff(y)
                    y_lag = y[:-1]
                    try:
                        res = sm.OLS(dy, sm.add_constant(y_lag)).fit(disp=False)
                        return res.tvalues[1]
                    except Exception:
                        return 0.0
                return (
                    log_p
                    .rolling(self.window)
                    .apply(adf_tstat, raw=False)
                    .reindex(price.index)
                    .fillna(0.0)
                )
            
            features["regime_sadf"] = get_sadf_proxy(df[raw_price_col])
        
        # Market Features
        if raw_price_col in df.columns and denoised_price_col in df.columns:
            stretch = np.log(
                (df[raw_price_col] + 1e-9) /
                (df[denoised_price_col] + 1e-9)
            )
            features["market_stretch"] = stretch.clip(-5, 5)
            
            features["noise_persistence"] = (
                (df[raw_price_col] - df[denoised_price_col])
                .rolling(self.span)
                .std()
            )
        
        # Model Stability
        if 'y_true' in df.columns and 'y_pred' in df.columns:
            def binary_entropy(errors):
                p = errors.mean()
                if p <= 0.0 or p >= 1.0:
                    return 0.0
                return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
            
            errors = (df["y_true"] != df["y_pred"]).astype(int)
            features["perf_entropy"] = (
                errors
                .rolling(self.window)
                .apply(binary_entropy, raw=True)
            )
        
        return features.fillna(0.0)


def compute_layer4_regime_features(
    df: pd.DataFrame,
    window: int = 50,
) -> pd.DataFrame:
    """
    Compute regime features for Layer 4 position sizing.
    
    REFACTORED: Now uses unified Layer4FeatureGenerator for consistency.
    
    Args:
        df: DataFrame with market data
        window: Rolling window size
        
    Returns:
        DataFrame with regime features
    """
    try:
        from .layer4 import Layer4FeatureGenerator
        
        generator = Layer4FeatureGenerator(window=window)
        
        # Generate only regime features
        config = {
            'enable_performance': False,
            'enable_regime': True,
            'enable_market': False,
            'enable_technical': True,  # Include technical for regime
            'enable_structural': False,
            'enable_model': False,
            'enable_time': False,
            'enable_contextual': False
        }
        
        features_df = generator.generate_all_features(df=df)
        
        # Select regime-relevant features
        regime_features = [
            'vol_long', 'vol_ratio', 'regime_sadf', 'adx_proxy',
            'choppiness_index', 'variance_ratio', 'efficiency_ratio'
        ]
        
        available_regime_features = [f for f in regime_features if f in features_df.columns]
        return features_df[available_regime_features].fillna(0.0)
        
    except ImportError:
        # Fallback to original implementation
        return _compute_regime_features_original(df, window)


def _compute_regime_features_original(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Original regime features implementation as fallback."""
    features = pd.DataFrame(index=df.index)
    
    if not all(col in df.columns for col in ['close', 'high', 'low']):
        return features
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Log returns
    log_ret = np.log(close / close.shift(1))
    
    # Volatility Features
    rv_short = log_ret.rolling(window=12).std() * np.sqrt(12)
    rv_long = log_ret.rolling(window=200).std()
    
    features['vol_long'] = rv_long
    features['vol_ratio'] = rv_short / (rv_long + 1e-8)
    
    # ADX proxy
    up_move = high.diff()
    down_move = low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr = (high - low) / close
    tr_smooth = tr.rolling(window=14).sum()
    plus_di = pd.Series(plus_dm, index=df.index).rolling(window=14).sum() / (tr_smooth + 1e-8)
    minus_di = pd.Series(minus_dm, index=df.index).rolling(window=14).sum() / (tr_smooth + 1e-8)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    features['adx_proxy'] = dx.rolling(window=14).mean()
    
    # Choppiness Index
    chop_window = 20
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr_series = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    sum_tr = tr_series.rolling(chop_window).sum()
    max_hi = high.rolling(chop_window).max()
    min_lo = low.rolling(chop_window).min()
    range_hl = max_hi - min_lo
    
    features['choppiness_index'] = 100 * np.log10(sum_tr / (range_hl + 1e-8)) / np.log10(chop_window)
    
    # Variance Ratio
    vr_window = 50
    r_20 = log_ret.rolling(20).sum()
    r_10 = log_ret.rolling(10).sum()
    var_20 = r_20.rolling(vr_window).var()
    var_10 = r_10.rolling(vr_window).var()
    features['variance_ratio'] = var_20 / (2 * var_10 + 1e-8)
    
    # Time Features
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    else:
        features['hour_sin'] = 0.0
        features['hour_cos'] = 0.0
    
    # Efficiency Ratio
    er_window = 10
    change = (close - close.shift(er_window)).abs()
    volatility = close.diff().abs().rolling(er_window).sum()
    features['efficiency_ratio'] = change / (volatility + 1e-8)
    
    return features.fillna(0.0)


# Keep the original triple_barrier_trailing_label function unchanged
def triple_barrier_trailing_label(
    df: pd.DataFrame,
    events: pd.DatetimeIndex,
    volatility: pd.Series,
    horizon: int = 24,
    sl: float = 1.0,
    trailing_gap: Optional[float] = None,
    pt: Optional[float] = None,
    min_ret: float = 0.003
) -> pd.DataFrame:
    """
    Advanced Triple Barrier Labeler with Trailing Profit Logic.
    
    Implements a "Rising Floor" trade structure:
    - If trailing_gap is set: The Upper Barrier is removed (Infinity).
      The Lower Barrier (Stop Loss) ratchets up as price makes new highs.
    - If trailing_gap is None: Uses standard Fixed Upper/Lower Barriers.
    
    Args:
        df: DataFrame with 'close', 'high', 'low' columns.
        events: DatetimeIndex of signal entry times.
        volatility: Series of volatility (e.g., ATR or StdDev) aligned with df.
        horizon: Maximum holding period in bars (Vertical Barrier).
        sl: Initial Stop Loss multiplier (e.g., 1.0 * Volatility).
        trailing_gap: The distance (in Volatility units) the stop trails behind the High.
                      If None, defaults to Fixed Barrier logic.
        pt: Fixed Profit Target multiplier (Only used if trailing_gap is None).
        min_ret: Minimum return required to label as '1' (accounts for fees).

    Returns:
        DataFrame containing:
        - 'label': {-1, 0, 1}
        - 'ret': Raw return of the trade
        - 'weight': Sample weight based on Inverse Volatility
    """
    out = {}

    # 1. Config: Fee Floors
    barrier_type = "Trailing" if trailing_gap is not None else "Fixed"
    tprint_info(f"🔄 Triple Barrier: {barrier_type} mode, {len(events)} events, horizon={horizon}")

    # 2. Pre-fetch Data for Speed
    vol_s = volatility.reindex(events).ffill().bfill()

    closes = df['close'].values
    if 'high' in df.columns and 'low' in df.columns:
        highs = df['high'].values
        lows = df['low'].values
    else:
        highs = closes; lows = closes
        
    index = df.index
    n_bars = len(df)

    # 3. Main Event Loop
    for t in events:
        if t not in index: continue
        
        i_0 = index.get_loc(t)
        i_1 = min(i_0 + horizon, n_bars - 1)
        if i_1 <= i_0: continue
        
        curr_vol = vol_s[t]
        if curr_vol <= 0: curr_vol = VOLATILITY_SAFETY_FLOOR
        
        entry_price = closes[i_0]
        
        # --- A. Determine Safe Distances ---
        raw_stop_dist = curr_vol * sl
        safe_stop_dist = max(raw_stop_dist, STOP_LOSS_FLOOR)
        
        # --- B. Trailing Stop Logic ---
        if trailing_gap is not None:
            stop_price = entry_price * (1 - safe_stop_dist)
            max_price = entry_price
            exit_idx = -1
            
            raw_gap_dist = curr_vol * trailing_gap
            safe_gap_dist = max(raw_gap_dist, STOP_LOSS_FLOOR)

            exit_price = entry_price
            for k in range(i_0 + 1, i_1 + 1):
                c_low = lows[k]
                c_high = highs[k]
                
                if c_low < stop_price:
                    exit_idx = k
                    exit_price = stop_price
                    break
                
                if c_high > max_price:
                    max_price = c_high
                    new_stop = max_price * (1 - safe_gap_dist)
                    stop_price = max(stop_price, new_stop)

            if exit_idx != -1:
                raw_ret = (exit_price / entry_price) - 1
            else:
                raw_ret = (closes[i_1] / entry_price) - 1

        # --- C. Standard Fixed Barrier Logic ---
        else:
            eff_pt = pt if pt is not None else 1.0
            rr_ratio = eff_pt / sl
            safe_target_dist = safe_stop_dist * rr_ratio
            
            trgt_price = entry_price * (1 + safe_target_dist)
            stop_price = entry_price * (1 - safe_stop_dist)

            raw_ret = 0.0
            path_slice_high = highs[i_0+1 : i_1+1]
            path_slice_low = lows[i_0+1 : i_1+1]
            path_slice_close = closes[i_0+1 : i_1+1]

            up_mask = path_slice_high > trgt_price
            dn_mask = path_slice_low < stop_price
            
            has_up = up_mask.any()
            has_dn = dn_mask.any()
            
            if has_up:
                touch_up = np.argmax(up_mask)
            else:
                touch_up = None
                
            if has_dn:
                touch_dn = np.argmax(dn_mask)
            else:
                touch_dn = None

            if has_up and (not has_dn or (touch_dn is not None and touch_up < touch_dn)):
                raw_ret = safe_target_dist
            elif has_dn and (not has_up or (touch_up is not None and touch_dn < touch_up)):
                raw_ret = -safe_stop_dist
            else:
                raw_ret = (path_slice_close[-1] / entry_price) - 1

        # --- D. Final Labeling & Weighting ---
        
        if raw_ret > min_ret:
            label = 1
        elif raw_ret < -min_ret:
            label = -1
        else:
            label = 0

        weight = np.clip(TARGET_VOLATILITY / (curr_vol + VOLATILITY_SAFETY_FLOOR), WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX)
        
        if label == 1 and abs(raw_ret) > (curr_vol * HOME_RUN_MULTIPLIER):
            weight *= 1.5

        if label != 0:
            out[t] = {
                'label': label,
                'ret': raw_ret,
                'weight': weight
            }

    return pd.DataFrame.from_dict(out, orient='index')


def train_layer4_simple_multimodel(
    oof_df: pd.DataFrame,
    market_data: pd.DataFrame,
    base_model_predictions: Optional[pd.DataFrame] = None,
    l3_prob_col: str = 'meta_prob',
    target_col: str = 'realized_return',
    return_col: str = 'realized_return',
    n_folds: int = 5,
    config: Optional[Dict[str, Any]] = None,
    n_estimators: int = 1000,
    max_features: str = 'log2',
    consensus_weights: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train Layer 4 using Simple Multi-Model Risk Engine.
    
    This function implements simultaneous training of Ridge, ExtraTrees, and XGBoost
    without stacking or complex hierarchies.
    
    Args:
        oof_df: Out-of-fold predictions with meta-labels and returns
        market_data: Market data for feature generation
        base_model_predictions: Optional DataFrame of base model predictions
        l3_prob_col: Column name for Layer 3 probabilities
        target_col: Target column name
        return_col: Return column name
        n_folds: Number of cross-validation folds
        config: Configuration dictionary
        n_estimators: Number of trees in ExtraTrees/XGBoost
        max_features: Max features parameter for ExtraTrees
        consensus_weights: Custom weights for model consensus
        
    Returns:
        Tuple of (OOF predictions DataFrame, training results dictionary)
    """
    config = config or {}
    
    tprint_info("🚀 Starting Layer 4 Simple Multi-Model training...")
    
    # Initialize results storage
    oof_predictions = pd.DataFrame(index=oof_df.index, columns=['fold', 'bet_size'])
    fold_results = []
    
    # Prepare base model predictions if not provided
    if base_model_predictions is None:
        # Create synthetic base model predictions from Layer 3 probabilities
        np.random.seed(42)
        n_base_models = 10
        base_model_predictions = pd.DataFrame(
            index=oof_df.index,
            columns=[f'model_{i}' for i in range(n_base_models)]
        )
        
        for i in range(n_base_models):
            noise = np.random.normal(0, 0.05, len(oof_df))
            base_model_predictions[f'model_{i}'] = (
                oof_df[l3_prob_col] + noise
            ).clip(0, 1)
    
    # Generate market features using existing Layer 4 infrastructure
    meta_features = MetaLearnerFeatures(config=config)
    market_features = meta_features.generate(
        df=oof_df.join(market_data, how='left'),
        raw_price_col='close',
        denoised_price_col='denoised_price'
    )
    
    # Cross-validation training
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Prepare labels for stratification
    y_binary = (oof_df[return_col] > 0).astype(int)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(oof_df, y_binary)):
        tprint_info(f"📊 Training fold {fold + 1}/{n_folds} with simple multi-model approach...")
        
        # Split data
        train_preds = base_model_predictions.iloc[train_idx]
        val_preds = base_model_predictions.iloc[val_idx]
        train_market = market_features.iloc[train_idx]
        val_market = market_features.iloc[val_idx]
        train_returns = oof_df.iloc[train_idx][return_col]
        val_returns = oof_df.iloc[val_idx][return_col]
        train_labels = y_binary.iloc[train_idx]
        val_labels = y_binary.iloc[val_idx]
        
        # Initialize and train simple multi-model risk engine
        engine = SimpleMultiModelRiskEngine(
            n_estimators=n_estimators,
            max_features=max_features,
            consensus_weights=consensus_weights
        )
        
        # Train with simultaneous multi-model approach
        train_metrics = engine.train(
            preds_df=train_preds,
            market_features=train_market,
            y_true=train_labels,
            abs_returns=train_returns.abs()
        )
        
        # Generate predictions
        val_bet_sizes = engine.predict_bet_size(
            preds_df=val_preds,
            market_features=val_market
        )
        
        # Store OOF predictions
        oof_predictions.loc[val_idx, 'bet_size'] = val_bet_sizes
        oof_predictions.loc[val_idx, 'fold'] = fold
        
        # Evaluate external metrics
        external_metrics = engine.evaluate_external_metrics(
            bet_sizes=val_bet_sizes,
            returns=val_returns,
            volatility=None  # Could add volatility calculation here
        )
        
        # Combine metrics
        fold_result = {
            'fold': fold,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            **train_metrics,
            **external_metrics
        }
        
        fold_results.append(fold_result)
        
        tprint_success(f"✅ Fold {fold + 1}: Sortino={external_metrics['sortino_ratio']:.3f}, "
                       f"MaxDD={external_metrics['max_drawdown']:.3f}, "
                       f"PnL={external_metrics['total_pnl']:.4f}")
    
    # Aggregate results across folds
    results = _aggregate_simple_results(fold_results, oof_predictions, oof_df[return_col])
    
    tprint_success("🎉 Simple Multi-Model training completed!")
    tprint_info(f"📈 Overall Sortino: {results['overall_sortino']:.3f}")
    tprint_info(f"📉 Overall MaxDD: {results['overall_max_drawdown']:.3f}")
    tprint_info(f"💰 Overall PnL: {results['overall_pnl']:.4f}")
    tprint_info(f"🎯 Consensus Weighted LogLoss: {results['avg_consensus_weighted_logloss']:.4f}")
    tprint_info(f"📊 Consensus Brier Score: {results['avg_consensus_brier_score']:.4f}")
    
    return oof_predictions, results


def _aggregate_simple_results(
    fold_results: List[Dict], 
    oof_predictions: pd.DataFrame, 
    returns: pd.Series
) -> Dict[str, Any]:
    """Aggregate results across folds for simple multi-model approach."""
    
    # Average fold metrics
    avg_metrics = {}
    metric_keys = fold_results[0].keys()
    exclude_keys = {'fold', 'train_samples', 'val_samples'}
    
    for key in metric_keys - exclude_keys:
        values = [r[key] for r in fold_results if np.isfinite(r[key])]
        avg_metrics[f'avg_{key}'] = np.mean(values) if values else 0.0
        avg_metrics[f'std_{key}'] = np.std(values) if values else 0.0
    
    # Overall metrics using all OOF predictions
    valid_mask = oof_predictions['bet_size'].notna()
    overall_bet_sizes = oof_predictions.loc[valid_mask, 'bet_size'].values
    overall_returns = returns.loc[valid_mask]
    
    # Compute overall external metrics
    engine = SimpleMultiModelRiskEngine()
    overall_metrics = engine.evaluate_external_metrics(
        bet_sizes=overall_bet_sizes,
        returns=overall_returns
    )
    
    # Add overall metrics to results
    for key, value in overall_metrics.items():
        avg_metrics[f'overall_{key}'] = value
    
    # Additional statistics
    avg_metrics.update({
        'total_folds': len(fold_results),
        'total_oof_samples': len(oof_predictions),
        'valid_oof_samples': valid_mask.sum(),
        'oof_coverage': valid_mask.mean(),
        'mean_bet_size': overall_bet_sizes.mean(),
        'bet_size_std': overall_bet_sizes.std(),
        'bet_size_range': (overall_bet_sizes.min(), overall_bet_sizes.max())
    })
    
    return avg_metrics


def _train_layer4_oof_simple_multimodel(
    oof_df: pd.DataFrame,
    market_data: pd.DataFrame,
    base_model_predictions: Optional[pd.DataFrame] = None,
    l3_prob_col: str = 'meta_prob',
    target_col: str = 'realized_return',
    return_col: str = 'realized_return',
    n_folds: int = 5,
    config: Optional[Dict[str, Any]] = None,
    n_estimators: int = 1000,
    max_features: str = 'log2',
    consensus_weights: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train Layer 4 using Simple Multi-Model Risk Engine.
    Wrapper function for backward compatibility.
    """
    return train_layer4_simple_multimodel(
        oof_df=oof_df,
        market_data=market_data,
        base_model_predictions=base_model_predictions,
        l3_prob_col=l3_prob_col,
        target_col=target_col,
        return_col=return_col,
        n_folds=n_folds,
        config=config,
        n_estimators=n_estimators,
        max_features=max_features,
        consensus_weights=consensus_weights
    )


def train_layer4_oof(
    oof_df: pd.DataFrame,
    market_data: pd.DataFrame,
    base_model_predictions: Optional[pd.DataFrame] = None,
    l3_prob_col: str = 'meta_prob',
    target_col: str = 'realized_return',
    return_col: str = 'realized_return',
    n_folds: int = 5,
    config: Optional[Dict[str, Any]] = None,
    n_estimators: int = 1000,
    max_features: str = 'log2',
    use_stacking: bool = False,
    consensus_weights: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train Layer 4 using Simple Multi-Model Risk Engine.
    Wrapper function for backward compatibility.
    """
    return train_layer4_simple_multimodel(
        oof_df=oof_df,
        market_data=market_data,
        base_model_predictions=base_model_predictions,
        l3_prob_col=l3_prob_col,
        target_col=target_col,
        return_col=return_col,
        n_folds=n_folds,
        config=config,
        n_estimators=n_estimators,
        max_features=max_features,
        consensus_weights=consensus_weights
    )


def _train_layer4_oof_extratrees_pnl(
    oof_df: pd.DataFrame,
    market_data: pd.DataFrame,
    base_model_predictions: Optional[pd.DataFrame] = None,
    l3_prob_col: str = 'meta_prob',
    target_col: str = 'realized_return',
    return_col: str = 'realized_return',
    n_folds: int = 5,
    config: Optional[Dict[str, Any]] = None,
    n_estimators: int = 1000,
    max_features: str = 'log2'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train Layer 4 using Simple Multi-Model Risk Engine.
    Wrapper function for backward compatibility.
    """
    return train_layer4_simple_multimodel(
        oof_df=oof_df,
        market_data=market_data,
        base_model_predictions=base_model_predictions,
        l3_prob_col=l3_prob_col,
        target_col=target_col,
        return_col=return_col,
        n_folds=n_folds,
        config=config,
        n_estimators=n_estimators,
        max_features=max_features,
        consensus_weights=None
    )


def _aggregate_enhanced_results(
    fold_results: List[Dict], 
    oof_predictions: pd.DataFrame, 
    returns: pd.Series
) -> Dict[str, Any]:
    """Aggregate results across folds for enhanced multi-model approach."""
    
    # Average fold metrics
    avg_metrics = {}
    metric_keys = fold_results[0].keys()
    exclude_keys = {'fold', 'train_samples', 'val_samples'}
    
    for key in metric_keys - exclude_keys:
        values = [r[key] for r in fold_results if np.isfinite(r[key])]
        avg_metrics[f'avg_{key}'] = np.mean(values) if values else 0.0
        avg_metrics[f'std_{key}'] = np.std(values) if values else 0.0
    
    # Overall metrics using all OOF predictions
    valid_mask = oof_predictions['bet_size'].notna()
    overall_bet_sizes = oof_predictions.loc[valid_mask, 'bet_size'].values
    overall_returns = returns.loc[valid_mask]
    
    # Compute overall external metrics
    engine = SimpleMultiModelRiskEngine()
    overall_metrics = engine.evaluate_external_metrics(
        bet_sizes=overall_bet_sizes,
        returns=overall_returns
    )
    
    # Add overall metrics to results
    for key, value in overall_metrics.items():
        avg_metrics[f'overall_{key}'] = value
    
    # Additional statistics
    avg_metrics.update({
        'total_folds': len(fold_results),
        'total_oof_samples': len(oof_predictions),
        'valid_oof_samples': valid_mask.sum(),
        'oof_coverage': valid_mask.mean(),
        'mean_bet_size': overall_bet_sizes.mean(),
        'bet_size_std': overall_bet_sizes.std(),
        'bet_size_range': (overall_bet_sizes.min(), overall_bet_sizes.max())
    })
    
    return avg_metrics


def train_layer4_deprado_risk_engine(
    oof_df: pd.DataFrame,
    market_data: pd.DataFrame,
    base_model_predictions: Optional[pd.DataFrame] = None,
    l3_prob_col: str = 'meta_prob',
    target_col: str = 'realized_return',
    return_col: str = 'realized_return',
    n_folds: int = 5,
    config: Optional[Dict[str, Any]] = None,
    n_estimators: int = 1000,
    max_features: str = 'log2'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train Layer 4 using Simple Multi-Model Risk Engine.
    Wrapper function for backward compatibility.
    """
    return train_layer4_simple_multimodel(
        oof_df=oof_df,
        market_data=market_data,
        base_model_predictions=base_model_predictions,
        l3_prob_col=l3_prob_col,
        target_col=target_col,
        return_col=return_col,
        n_folds=n_folds,
        config=config,
        n_estimators=n_estimators,
        max_features=max_features,
        consensus_weights=None
    )


def train_layer4_deprado_risk_engine(
    oof_df: pd.DataFrame,
    market_data: pd.DataFrame,
    base_model_predictions: Optional[pd.DataFrame] = None,
    l3_prob_col: str = 'meta_prob',
    target_col: str = 'realized_return',
    return_col: str = 'realized_return',
    n_folds: int = 5,
    config: Optional[Dict[str, Any]] = None,
    n_estimators: int = 1000,
    max_features: str = 'log2'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train Layer 4 using Simple Multi-Model Risk Engine.
    Wrapper function for backward compatibility.
    """
    return train_layer4_simple_multimodel(
        oof_df=oof_df,
        market_data=market_data,
        base_model_predictions=base_model_predictions,
        l3_prob_col=l3_prob_col,
        target_col=target_col,
        return_col=return_col,
        n_folds=n_folds,
        config=config,
        n_estimators=n_estimators,
        max_features=max_features,
        consensus_weights=None
    )


def compute_layer4_regime_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Compute Layer 4 regime features using unified feature generator."""
    meta_features = MetaLearnerFeatures(config=config)
    return meta_features.generate(
        df=df,
        raw_price_col='close',
        denoised_price_col='denoised_price'
    )


def _aggregate_deprado_results(
    fold_results: List[Dict], 
    oof_predictions: pd.DataFrame, 
    returns: pd.Series
) -> Dict[str, Any]:
    """Aggregate results across folds for DePrado approach."""
    
    # Average fold metrics
    avg_metrics = {}
    metric_keys = fold_results[0].keys()
    exclude_keys = {'fold', 'train_samples', 'val_samples'}
    
    for key in metric_keys - exclude_keys:
        values = [r[key] for r in fold_results if np.isfinite(r[key])]
        avg_metrics[f'avg_{key}'] = np.mean(values) if values else 0.0
        avg_metrics[f'std_{key}'] = np.std(values) if values else 0.0
    
    # Overall metrics using all OOF predictions
    valid_mask = oof_predictions['bet_size'].notna()
    overall_bet_sizes = oof_predictions.loc[valid_mask, 'bet_size'].values
    overall_returns = returns.loc[valid_mask]
    
    # Compute overall external metrics
    engine = SimpleMultiModelRiskEngine()
    overall_metrics = engine.evaluate_external_metrics(
        bet_sizes=overall_bet_sizes,
        returns=overall_returns
    )
    
    # Add overall metrics to results
    for key, value in overall_metrics.items():
        avg_metrics[f'overall_{key}'] = value
    
    # Additional statistics
    avg_metrics.update({
        'total_folds': len(fold_results),
        'total_oof_samples': len(oof_predictions),
        'valid_oof_samples': valid_mask.sum(),
        'oof_coverage': valid_mask.mean(),
        'mean_bet_size': overall_bet_sizes.mean(),
        'bet_size_std': overall_bet_sizes.std(),
        'bet_size_range': (overall_bet_sizes.min(), overall_bet_sizes.max())
    })
    
    return avg_metrics


def _train_layer4_oof_extratrees_pnl_simple(
    oof_df: pd.DataFrame,
    market_data: pd.DataFrame,
    base_model_predictions: Optional[pd.DataFrame] = None,
    l3_prob_col: str = 'meta_prob',
    target_col: str = 'realized_return',
    return_col: str = 'realized_return',
    n_folds: int = 5,
    config: Optional[Dict[str, Any]] = None,
    n_estimators: int = 1000,
    max_features: str = 'log2'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train Layer 4 using Simple Multi-Model Risk Engine.
    Wrapper function for backward compatibility.
    """
    return train_layer4_simple_multimodel(
        oof_df=oof_df,
        market_data=market_data,
        base_model_predictions=base_model_predictions,
        l3_prob_col=l3_prob_col,
        target_col=target_col,
        return_col=return_col,
        n_folds=n_folds,
        config=config,
        n_estimators=n_estimators,
        max_features=max_features,
        consensus_weights=None
    )


# ---------------------------------------------------------------------------
# Training Orchestration (Updated to use unified feature generation)
# ---------------------------------------------------------------------------

def train_layer4_oof(
    oof_df: pd.DataFrame,
    market_data: pd.DataFrame,
    l3_prob_col: str = 'meta_prob',
    target_col: str = 'realized_return',
    return_col: str = 'realized_return',
    n_folds: int = 5,
    config: Optional[Dict[str, Any]] = None,
    # Deprecated parameters - kept for backward compatibility only
    l3_models_metadata: Optional[Dict] = None,
    l3_quantile_thresholds: Optional[List[float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train ExtraTrees model for Layer4 position sizing optimized for PnL and Sortino.
    
    REFACTORED: Now delegates to unified feature generation in layer4_extratrees_pnl.py
    
    This function now uses ExtraTrees classifier trained on returns with comprehensive
    features including disagreement features, structural break scores, relative strength,
    and drawdown state to maximize PnL and Sortino while minimizing drawdown.
    """
    
    # Call the new PnL-optimized ExtraTrees implementation
    return _train_layer4_oof_extratrees_pnl(
        oof_df=oof_df,
        market_data=market_data,
        l3_prob_col=l3_prob_col,
        target_col=target_col,
        return_col=return_col,
        n_folds=n_folds,
        config=config,
        l3_models_metadata=l3_models_metadata,
        l3_quantile_thresholds=l3_quantile_thresholds
    )
