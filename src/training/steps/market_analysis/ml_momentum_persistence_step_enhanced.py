"""
Enhanced ML Momentum Persistence Step with MI Improvements

This enhanced version implements:
- Enhanced feature generation for MI improvement
- Real-time MI monitoring during training
- Hyperparameter optimization for MI > 0.02 target
- Data structure standardization
- Binary output enforcement
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.training.steps.market_analysis.ml_risk_regime_step import MLRiskRegimeStepHMM
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2
)
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType

logger = logging.getLogger(__name__)


class EnhancedMLMomentumPersistenceStep(SpecialistDiagnosticsMixinEnhancedV2, MLRiskRegimeStepHMM):

    @property
    def artifact_router(self):
        """Override artifact_router property for enhanced specialists."""
        if self._artifact_router is None:
            from src.utils.artifact_router import ArtifactRouter
            self._artifact_router = ArtifactRouter(
                base_dir="artifacts",
                versioned_store_dir="versioned_artifacts",
                historical_data_dir="historical_data",
                enable_versioned_artifacts=self.use_versioned_artifacts
            )
        return self._artifact_router

    """
    Enhanced Momentum Persistence Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Hyperparameter optimization for MI > 0.02
    - Standardized data structure output
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_ml_momentum_persistence_step"):
        """Initialize the enhanced momentum persistence step."""
        super().__init__()
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self._artifact_router = None
        self.use_versioned_artifacts = True
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedMLMomentumPersistenceStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        
    def _compute_enhanced_momentum_optimized_horizon_optimized_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute enhanced momentum features with MI improvements."""
        features = pd.DataFrame(index=df.index)
        
        # Basic momentum features
        returns = df['close'].pct_change()
        
        # Multi-timeframe momentum
        for window in [10,20,40]:
            features[f'momentum_{window}'] = returns.rolling(window).sum()
            features[f'momentum_accel_{window}'] = returns.rolling(window).sum() - returns.rolling(window*2).sum()
            features[f'momentum_vol_{window}'] = returns.rolling(window).std()
        
        # RSI-like momentum
        for window in [10,20,40]:
            gains = returns.clip(lower=0)
            losses = -returns.clip(upper=0)
            avg_gains = gains.rolling(window).mean()
            avg_losses = losses.rolling(window).mean()
            rs = avg_gains / (avg_losses + 1e-8)
            features[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # Price momentum indicators
        for window in [10, 20, 50]:
            sma = df['close'].rolling(window).mean()
            features[f'price_momentum_{window}'] = (df['close'] - sma) / sma
            features[f'price_momentum_cross_{window}'] = (df['close'] > sma).astype(int)
        
        # Volume momentum
        if 'volume' in df.columns:
            volume_change = df['volume'].pct_change()
            for window in [5, 10, 20]:
                features[f'volume_momentum_{window}'] = volume_change.rolling(window).sum()
                features[f'volume_price_momentum_{window}'] = (returns * volume_change).rolling(window).sum()
        
        return features
    
    def _generate_enhanced_features(self, df: pd.DataFrame, specialist_type: SpecialistType = None) -> pd.DataFrame:
        """Generate all enhanced features with manual redundancy reduction and poor performer enhancement."""
        # Basic momentum features
        momentum_features = self._compute_enhanced_momentum_optimized_horizon_optimized_momentum_features(df)
        
        # Enhanced features from pipeline
        enhanced_features = self.feature_pipeline.generate_enhanced_features(
            df, 'momentum', {'enhanced_features': True}
        )
        
        # Manual feature engineering to address redundancy and poor performers
        manual_features = self._create_manual_enhanced_features(df, enhanced_features)
        
        # Combine all features
        all_features = [momentum_features, enhanced_features, manual_features]
        
        # Combine all features with manual redundancy reduction
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            
            # Manual redundancy reduction and feature selection
            combined_features = self._apply_manual_feature_selection(combined_features)
            
            # Remove duplicates and clean
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            combined_features = combined_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            return combined_features
    
    def _create_manual_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create manual enhanced features to address redundancy and improve poor performers."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            # 1. Address smc|volume_force_breakout redundancy (0.632)
            # Create orthogonal regime features
            if 'smc_predicted' in enhanced_features.columns and 'vol_force_breakout' in enhanced_features.columns:
                smc = enhanced_features['smc_predicted']
                vol = enhanced_features['vol_force_breakout']
                
                # Standardize for orthogonal decomposition
                smc_std = (smc - smc.mean()) / (smc.std() + 1e-8)
                vol_std = (vol - vol.mean()) / (vol.std() + 1e-8)
                
                # Create orthogonal volume signal (remove smc component)
                orthogonal_vol = vol_std - (np.cov(vol_std, smc_std)[0,1] / np.var(smc_std)) * smc_std
                orthogonal_vol = orthogonal_vol / (orthogonal_vol.std() + 1e-8)
                manual_features['orthogonal_volume_regime'] = orthogonal_vol
                
                # Regime divergence (captures disagreement)
                regime_divergence = np.abs(smc_std - vol_std)
                manual_features['regime_divergence'] = regime_divergence
                
                # Regime consensus (captures agreement)
                regime_consensus = (smc_std + vol_std) / 2
                manual_features['regime_consensus'] = regime_consensus
            
            # 2. Improve risk_score (0.0131 MI, 0.0390 HSIC - POOR)
            # Enhanced multi-timeframe risk features
            short_vol = returns.rolling(5).std()
            medium_vol = returns.rolling(20).std()
            long_vol = returns.rolling(50).std()
            
            # Volume-adjusted risk
            volume_ma = volume.rolling(20).mean()
            volume_ratio = volume / (volume_ma + 1e-8)
            volume_adjusted_risk = medium_vol * (1 + np.log(volume_ratio + 1))
            manual_features['enhanced_volume_adjusted_risk'] = volume_adjusted_risk
            
            # Range-based risk
            range_ratio = (high - low) / close
            range_vol = range_ratio.rolling(20).std()
            manual_features['enhanced_range_based_risk'] = range_vol
            
            # Downside risk (focus on negative movements)
            downside_returns = returns.copy()
            downside_returns[downside_returns > 0] = 0
            downside_vol = downside_returns.rolling(20).std()
            manual_features['enhanced_downside_risk'] = downside_vol
            
            # Risk regime classification
            risk_zscore = (medium_vol - medium_vol.rolling(100).mean()) / (medium_vol.rolling(100).std() + 1e-8)
            manual_features['enhanced_risk_regime'] = np.where(risk_zscore > 1, 2, np.where(risk_zscore < -1, 0, 1))
            
            # 3. Improve path_risk_score (0.0082 MI, 0.0033 HSIC - VERY POOR)
            # Enhanced path risk features
            price_path = close.rolling(10).mean()
            
            # Path smoothness (detects erratic movements)
            path_smoothness = np.abs(price_path.diff().diff())
            manual_features['enhanced_path_smoothness'] = path_smoothness
            
            # Path acceleration (detects momentum changes)
            path_velocity = close.rolling(5).mean().diff()
            path_acceleration = path_velocity.diff()
            manual_features['enhanced_path_acceleration'] = path_acceleration
            
            # Path volatility (volatility of the path)
            path_vol = path_velocity.rolling(20).std()
            manual_features['enhanced_path_volatility'] = path_vol
            
            # Mean reversion strength
            price_mean = close.rolling(50).mean()
            # Use rolling correlation with proper syntax
            mean_reversion_strength = close.rolling(10).corr(price_mean)
            manual_features['enhanced_mean_reversion_strength'] = -mean_reversion_strength
            
            # Path breakout detection
            path_range = price_path.rolling(20).max() - price_path.rolling(20).min()
            path_breakout = np.abs(close - price_path) / (path_range + 1e-8)
            manual_features['enhanced_path_breakout'] = path_breakout
            
            # 4. Additional orthogonal momentum features
            # Momentum regime detection
            momentum_regime = (returns.rolling(20).mean() > 0).astype(int)
            manual_features['momentum_regime'] = momentum_regime
            
            # Volatility-adjusted momentum
            vol_adjusted_momentum = returns.rolling(10).mean() / (returns.rolling(10).std() + 1e-8)
            manual_features['vol_adjusted_momentum'] = vol_adjusted_momentum
            
            # Momentum persistence
            momentum_persistence = (returns.rolling(5).mean() * returns.rolling(10).mean()).rolling(5).sum()
            manual_features['momentum_persistence'] = momentum_persistence
            
        return manual_features
    
    def _apply_manual_feature_selection(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply manual feature selection to reduce redundancy and keep high-quality features."""
        if features.empty:
            return features
        
        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)
            self.logger.info(f"Removed {len(constant_features)} constant features")
        
        # Manual redundancy reduction - remove highly correlated features
        correlation_matrix = features.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find highly correlated pairs (>0.9)
        to_drop = []
        for column in upper_triangle.columns:
            correlated_features = upper_triangle[column][upper_triangle[column] > 0.9]
            if not correlated_features.empty:
                # Keep the feature that comes first alphabetically (deterministic)
                for correlated_feature in correlated_features.index:
                    if correlated_feature > column:  # Drop the later feature alphabetically
                        to_drop.append(correlated_feature)
        
        # Remove redundant features
        if to_drop:
            features = features.drop(columns=list(set(to_drop)))
            self.logger.info(f"Removed {len(set(to_drop))} redundant features: {list(set(to_drop))}")
        
        # Keep only the most informative features (limit to top 30 by variance)
        if len(features.columns) > 30:
            feature_variances = features.var()
            top_features = feature_variances.nlargest(30).index
            features = features[top_features]
            self.logger.info(f"Limited to top 30 features by variance")
        
        return features
    
    def _create_momentum_labels(self, df: pd.DataFrame, lookforward: int = 35) -> pd.Series:
        """Create momentum persistence labels."""
        returns = df['close'].pct_change()
        
        # Future momentum
        future_returns = returns.shift(-lookforward).rolling(lookforward).sum()
        
        # Binary label: positive future momentum
        labels = (future_returns > returns.rolling(25).std() * 0.5).astype(int)
        
        return labels
    
    def _compute_mi_during_training(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                  X_val: pd.DataFrame, y_val: pd.Series,
                                  model_predictions: np.ndarray) -> Dict[str, float]:
        """Compute MI metrics during training for monitoring."""
        mi_metrics = {}
        
        try:
            # Feature MI to target
            feature_mi_scores = []
            for col in X_train.select_dtypes(include=[np.number]).columns:
                mi_score = mutual_info_regression(
                    X_train[col].values.reshape(-1, 1), y_train.values
                )[0]
                feature_mi_scores.append(mi_score)
            
            if feature_mi_scores:
                mi_metrics['avg_feature_mi'] = np.mean(feature_mi_scores)
                mi_metrics['max_feature_mi'] = np.max(feature_mi_scores)
                mi_metrics['high_mi_features'] = sum(1 for mi in feature_mi_scores if mi > 0.02)
            
            # Prediction MI to target
            mi_metrics['prediction_mi'] = mutual_info_regression(
                model_predictions.reshape(-1, 1), y_val.values
            )[0]
            
            # MI improvement tracking
            self.mi_history.append(mi_metrics['prediction_mi'])
            
        except Exception as e:
            self.logger.warning(f"MI computation failed: {e}")
            mi_metrics = {'prediction_mi': 0.0, 'avg_feature_mi': 0.0}
        
        return mi_metrics
    
    def _optimize_hyperparameters_for_mi(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters specifically for MI improvement."""
        best_params = {}
        best_mi = 0.0
        
        # Parameter grid for MI optimization
        # Parameter grid for MI-focused optimization
        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [4, 6],
            "learning_rate": [0.03, 0.07, 0.1],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9],
            "gamma": [0, 0.1, 0.2],
            "reg_alpha": [0.1, 0.5, 1.0],
            "reg_lambda": [2, 5, 10],
            "min_child_weight": [20, 40]
        }
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for params in self._generate_param_combinations(param_grid, max_combinations=20):
            mi_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model with current parameters
                model = lgb.LGBMClassifier(
                    objective='binary',
                    random_state=42,
                    verbose=-1,
                    **params
                )
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                         callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
                
                # Compute MI
                val_pred = model.predict_proba(X_val)[:, 1]
                mi_score = mutual_info_regression(
                    val_pred.reshape(-1, 1), y_val.values
                )[0]
                mi_scores.append(mi_score)
            
            avg_mi = np.mean(mi_scores)
            
            if avg_mi > best_mi:
                best_mi = avg_mi
                best_params = params.copy()
                
                tprint_info(f"🔥 New best MI: {avg_mi:.4f} with params: {params}")
        
        tprint_success(f"✅ Best hyperparameters found: MI = {best_mi:.4f}")
        return best_params, best_mi
    

    def save(self, artifact_name: str, data, artifact_type: str = "data", data_category: str = "predictions"):
        """Custom save method for enhanced specialists."""
        try:
            # Use versioned store directly if available
            if hasattr(self, '_versioned_store') and self._versioned_store is not None:
                context = {
                    'symbol': self._current_context.get('symbol', 'UNKNOWN'),
                    'exchange': self._current_context.get('exchange', 'binance'),
                    'timeframe': self._current_context.get('timeframe', '15m'),
                    'direction': self._current_context.get('direction', 'long'),
                    'model': self._current_context.get('model', 'analyst'),
                    'step_name': self.step_name,
                }
                self._versioned_store.save(
                    artifact_name=artifact_name,
                    data=data,
                    artifact_type=artifact_type,
                    data_category=data_category,
                    context=context
                )
                self.logger.info(f"✅ Saved {artifact_name} to versioned store")
            else:
                self.logger.warning(f"⚠️ Cannot save {artifact_name}: no versioned store available")
        except Exception as e:
            self.logger.error(f"❌ Failed to save {artifact_name}: {e}")

    def _generate_param_combinations(self, param_grid: Dict[str, List], max_combinations: int = 20):
        """Generate parameter combinations for optimization."""
        import itertools
        import random
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        # Generate all combinations
        all_combinations = list(itertools.product(*values))
        
        # Randomly sample if too many
        if len(all_combinations) > max_combinations:
            all_combinations = random.sample(all_combinations, max_combinations)
        
        for combination in all_combinations:
            yield dict(zip(keys, combination))
    
    def _train_enhanced_momentum_model(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
           # Validate input data
        if len(features) < 100:
            tprint_warning("⚠️ Insufficient training samples, using fallback model")
            from sklearn.dummy import DummyClassifier
            dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
            dummy_model.fit(features, labels)
            return dummy_model, {'auc': 0.5, 'accuracy': 0.5, 'model_type': 'dummy_fallback'}
        
        # Check feature quality
        feature_variance = features.var()
        low_variance_features = feature_variance[feature_variance < 1e-6].index.tolist()
        if low_variance_features:
            tprint_warning(f"⚠️ Removing {len(low_variance_features)} low-variance features")
            features = features.drop(columns=low_variance_features)
        
        if features.empty:
            tprint_warning("⚠️ No valid features remaining, using fallback model")
            from sklearn.dummy import DummyClassifier
            dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
            dummy_model.fit(features, labels)
            return dummy_model, {'auc': 0.5, 'accuracy': 0.5, 'model_type': 'no_features_fallback'}
        
        # Check label balance
        label_counts = labels.value_counts()
        if len(label_counts) < 2:
            tprint_warning("⚠️ Insufficient label diversity, using fallback model")
            from sklearn.dummy import DummyClassifier
            dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
            dummy_model.fit(features, labels)
            return dummy_model, {'auc': 0.5, 'accuracy': 0.5, 'model_type': 'label_fallback'}
        
        # Check for constant predictions
        if label_counts.iloc[0] / len(labels) > 0.95:
            tprint_warning("⚠️ Highly imbalanced labels, using fallback model")
            from sklearn.dummy import DummyClassifier
            dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
            dummy_model.fit(features, labels)
            return dummy_model, {'auc': 0.5, 'accuracy': 0.5, 'model_type': 'imbalance_fallback'}
        
 