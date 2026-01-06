"""
Enhanced Specialist Mixin - Shared utilities for all enhanced specialist steps.

This mixin provides standardized methods for:
- Market data loading with caching
- XGBoost model training with proper API
- Artifact saving
- Feature generation
- MI optimization

All enhanced specialists should inherit from this mixin to ensure consistency.
"""

import logging
from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)


class EnhancedSpecialistMixin:
    """
    Mixin class providing shared utilities for enhanced specialist steps.
    
    This mixin should be used alongside BaseStep inheritance:
        class MyEnhancedStep(EnhancedSpecialistMixin, BaseStep):
            ...
    """
    
    # Class-level cache for market data (shared across instances)
    _shared_market_data_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    
    def _init_enhanced_specialist(self):
        """Initialize enhanced specialist attributes. Call in __init__."""
        self._market_data_cache = {}
        self.mi_history = []
        self.training_metrics = []
        self.feature_pipeline = None
    
    def load_market_data_cached(
        self, 
        config: Dict[str, Any], 
        timeframe: Optional[str] = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        Load market data with caching using BaseStep's standard method.
        
        Args:
            config: Configuration dictionary with symbol, exchange, timeframe
            timeframe: Optional override for timeframe
            
        Returns:
            Tuple of (market_data DataFrame, source string)
        """
        symbol = config.get("symbol", "ETHUSDT")
        exchange = config.get("exchange", "binance")
        tf = timeframe or config.get("timeframe", "15m")
        
        # Check instance cache
        cache_key = (symbol, exchange, tf)
        if cache_key in self._market_data_cache:
            logger.info(f"📦 Using cached market data for {symbol}/{exchange}/{tf}")
            return self._market_data_cache[cache_key], "cache"
        
        # Check class-level shared cache
        if cache_key in EnhancedSpecialistMixin._shared_market_data_cache:
            logger.info(f"📦 Using shared cached market data for {symbol}/{exchange}/{tf}")
            data = EnhancedSpecialistMixin._shared_market_data_cache[cache_key]
            self._market_data_cache[cache_key] = data
            return data, "shared_cache"
        
        # Load using BaseStep method
        merged_config = {**config, "timeframe": tf}
        market_data, source = self.load_market_data_or_fail(
            merged_config,
            pipeline_state={},
            allow_config_override=True,
        )
        
        # Cache the data
        self._market_data_cache[cache_key] = market_data
        EnhancedSpecialistMixin._shared_market_data_cache[cache_key] = market_data
        
        logger.info(f"✅ Loaded {len(market_data)} rows of market data for {symbol} from {source}")
        return market_data, source
    
    def train_xgb_classifier(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Optional[Dict[str, Any]] = None,
        n_splits: int = 5,
        early_stopping_rounds: int = 20,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train an XGBoost classifier with proper API usage.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            params: XGBoost parameters (optional)
            n_splits: Number of time series CV splits
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Tuple of (trained model, metrics dict)
        """
        import xgboost as xgb
        
        default_params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'min_child_weight': 20,
        }
        
        if params:
            default_params.update(params)
        
        # Create model with early_stopping_rounds in constructor (new API)
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=early_stopping_rounds,
            **default_params
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        metrics = {
            'cv_scores': [],
            'mi_scores': [],
        }
        
        best_model = None
        best_score = -np.inf
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit with eval_set for early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Compute metrics
            val_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # MI score
            mi_score = mutual_info_regression(
                val_pred_proba.reshape(-1, 1), 
                y_val.values
            )[0]
            metrics['mi_scores'].append(mi_score)
            
            # Track best model
            if mi_score > best_score:
                best_score = mi_score
                best_model = model
        
        metrics['mean_mi'] = np.mean(metrics['mi_scores'])
        metrics['best_mi'] = best_score
        
        logger.info(f"✅ XGB training complete: mean MI = {metrics['mean_mi']:.4f}, best MI = {best_score:.4f}")
        
        return best_model, metrics
    
    def optimize_xgb_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 20,
        n_splits: int = 3,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize XGBoost hyperparameters for MI improvement.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_trials: Number of random search trials
            n_splits: CV splits for evaluation
            
        Returns:
            Tuple of (best_params dict, best_mi score)
        """
        import xgboost as xgb
        
        # Parameter search space
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.03, 0.05, 0.07, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0.1, 0.5, 1.0],
            'reg_lambda': [2, 5, 10],
            'min_child_weight': [20, 30, 40],
        }
        
        best_params = {}
        best_mi = -np.inf
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for trial in range(n_trials):
            # Random sample parameters
            params = {k: np.random.choice(v) for k, v in param_grid.items()}
            
            mi_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    early_stopping_rounds=20,
                    **params
                )
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                val_pred = model.predict_proba(X_val)[:, 1]
                mi_score = mutual_info_regression(
                    val_pred.reshape(-1, 1), y_val.values
                )[0]
                mi_scores.append(mi_score)
            
            avg_mi = np.mean(mi_scores)
            
            if avg_mi > best_mi:
                best_mi = avg_mi
                best_params = params.copy()
                logger.info(f"🔥 Trial {trial+1}: New best MI = {avg_mi:.4f}")
        
        logger.info(f"✅ HPO complete: best MI = {best_mi:.4f}")
        return best_params, best_mi
    
    def generate_specialist_output(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        timestamps: pd.DatetimeIndex,
        specialist_name: str,
    ) -> pd.DataFrame:
        """
        Generate standardized specialist output DataFrame.
        
        Args:
            predictions: Binary predictions array
            probabilities: Probability predictions array
            timestamps: DatetimeIndex for the predictions
            specialist_name: Name of the specialist (e.g., 'risk', 'liquidity')
            
        Returns:
            Standardized DataFrame with specialist outputs
        """
        output_df = pd.DataFrame(index=timestamps)
        output_df['timestamp'] = timestamps
        output_df[f'{specialist_name}_predicted'] = predictions
        output_df[f'{specialist_name}_probability'] = probabilities
        output_df[f'{specialist_name}_score'] = probabilities  # Alias for compatibility
        
        return output_df
    
    def save_specialist_artifact(
        self,
        data: pd.DataFrame,
        artifact_name: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Save specialist artifact using BaseStep's artifact router.
        
        Args:
            data: DataFrame to save
            artifact_name: Name for the artifact
            config: Configuration dictionary
            metadata: Optional metadata dict
            
        Returns:
            Path to saved artifact or None if failed
        """
        try:
            path = self._save_artifact(
                artifact_name=artifact_name,
                data=data,
                data_category="features",
                metadata=metadata or {},
            )
            logger.info(f"✅ Saved artifact: {artifact_name} -> {path}")
            return path
        except Exception as e:
            logger.error(f"❌ Failed to save artifact {artifact_name}: {e}")
            return None
    
    def compute_feature_mi_scores(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute MI scores for all features against target.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            
        Returns:
            DataFrame with feature names and MI scores
        """
        # Align indices
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]
        
        # Handle NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        
        if len(X) < 100:
            logger.warning(f"⚠️ Insufficient samples for MI computation: {len(X)}")
            return pd.DataFrame({'feature': features.columns, 'mi_score': 0.0})
        
        mi_scores = []
        for col in X.columns:
            try:
                mi = mutual_info_regression(X[[col]].values, y.values)[0]
                mi_scores.append({'feature': col, 'mi_score': mi})
            except Exception as e:
                logger.warning(f"⚠️ MI computation failed for {col}: {e}")
                mi_scores.append({'feature': col, 'mi_score': 0.0})
        
        return pd.DataFrame(mi_scores).sort_values('mi_score', ascending=False)
    
    def create_binary_labels(
        self,
        data: pd.DataFrame,
        target_col: str = 'close',
        horizon: int = 4,
        threshold: float = 0.0,
    ) -> pd.Series:
        """
        Create binary labels based on future returns.
        
        Args:
            data: DataFrame with price data
            target_col: Column to use for return calculation
            horizon: Forward-looking periods
            threshold: Return threshold for positive label
            
        Returns:
            Binary label Series
        """
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        future_returns = data[target_col].pct_change(horizon).shift(-horizon)
        labels = (future_returns > threshold).astype(int)
        
        return labels
