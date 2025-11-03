"""
Multi-Task SR Quality Model

DATA-DRIVEN APPROACH: Instead of hand-crafted weighted scores,
train separate models for each quality component and learn optimal combination.

This replaces heuristic weights with learned weights based on actual performance.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class MultiTaskSRQualityModel:
    """
    Multi-task learning approach for SR quality prediction.
    
    KEY INNOVATION: Instead of quality_score = 0.25*bounce + 0.20*hold + ...
    We train:
    1. Separate models for each outcome (bounce, hold, trade, speed, volume)
    2. A meta-model that learns optimal combination weights
    
    This is DATA-DRIVEN because:
    - Weights are learned from actual performance data
    - Each component model specializes in its prediction
    - Meta-model discovers non-linear interactions
    """
    
    def __init__(self):
        self.component_models = {}  # Separate model for each component
        self.meta_model = None      # Learns optimal combination
        self.feature_names = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def train(self, training_data: pd.DataFrame, 
              n_folds: int = 5,
              num_boost_round: int = 500) -> Dict:
        """
        Train multi-task model.
        
        Process:
        1. Train 5 component models (bounce, hold, trade, speed, volume)
        2. Use predictions as features for meta-model
        3. Meta-model learns optimal combination
        
        Args:
            training_data: DataFrame with features and target components
            n_folds: CV folds
            num_boost_round: Boosting rounds
            
        Returns:
            Training metrics
        """
        
        self.logger.info("🚀 Training Multi-Task SR Quality Model")
        self.logger.info(f"   Samples: {len(training_data)}")
        
        # Separate features from targets
        feature_cols = [c for c in training_data.columns if c.startswith('feature_')]
        self.feature_names = feature_cols
        
        X = training_data[feature_cols].fillna(0.0)
        
        # TASK 1: Train component models
        component_targets = {
            'bounce': 'bounce_strength',      # Raw bounce metric
            'hold': 'hold_strength',          # Raw hold metric
            'trade': 'trade_profit',          # Raw trade profit
            'speed': 'rejection_speed',       # Raw rejection speed
            'volume': 'volume_quality'        # Raw volume metric
        }
        
        component_predictions = {}
        
        for component_name, target_col in component_targets.items():
            self.logger.info(f"\n📊 Training {component_name} model...")
            
            y = training_data[target_col]
            
            # Train component model with CV
            tscv = TimeSeriesSplit(n_splits=n_folds)
            fold_models = []
            cv_scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                train_data_lgb = lgb.Dataset(X_train, label=y_train)
                val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)
                
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'lambda_l1': 1.0,
                    'lambda_l2': 1.0,
                    'verbose': -1,
                    'seed': 42
                }
                
                model = lgb.train(
                    params,
                    train_data_lgb,
                    num_boost_round=num_boost_round,
                    valid_sets=[val_data_lgb],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)
                
                fold_models.append(model)
                cv_scores.append({'rmse': rmse, 'r2': r2})
                
            # Use best fold model
            best_idx = np.argmin([s['rmse'] for s in cv_scores])
            self.component_models[component_name] = fold_models[best_idx]
            
            avg_r2 = np.mean([s['r2'] for s in cv_scores])
            self.logger.info(f"   ✓ {component_name}: Avg R² = {avg_r2:.3f}")
            
            # Generate predictions for meta-model
            component_predictions[component_name] = self.component_models[component_name].predict(X)
        
        # TASK 2: Train meta-model on component predictions
        self.logger.info(f"\n🎯 Training meta-model (learns optimal combination)...")
        
        # Meta-features = predictions from component models
        meta_X = pd.DataFrame(component_predictions)
        
        # Meta-target = actual quality_score (or can use actual trading performance)
        meta_y = training_data['quality_score']
        
        # Add original features for context (optional)
        # This allows meta-model to adjust weights based on market conditions
        meta_X = pd.concat([
            meta_X,
            X[['feature_strength', 'feature_volatility', 'feature_market_trend']]
        ], axis=1)
        
        # Train meta-model
        tscv = TimeSeriesSplit(n_splits=n_folds)
        meta_fold_models = []
        meta_cv_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(meta_X)):
            X_train, X_val = meta_X.iloc[train_idx], meta_X.iloc[val_idx]
            y_train, y_val = meta_y.iloc[train_idx], meta_y.iloc[val_idx]
            
            train_data_lgb = lgb.Dataset(X_train, label=y_train)
            val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)
            
            # Simpler model for meta-learner (avoid overfitting)
            meta_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 15,  # Simpler model
                'learning_rate': 0.1,
                'lambda_l1': 2.0,  # More regularization
                'lambda_l2': 2.0,
                'verbose': -1,
                'seed': 42
            }
            
            model = lgb.train(
                meta_params,
                train_data_lgb,
                num_boost_round=100,  # Fewer rounds
                valid_sets=[val_data_lgb],
                callbacks=[lgb.early_stopping(20, verbose=False)]
            )
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            meta_fold_models.append(model)
            meta_cv_scores.append({'rmse': rmse, 'r2': r2})
        
        best_idx = np.argmin([s['rmse'] for s in meta_cv_scores])
        self.meta_model = meta_fold_models[best_idx]
        
        avg_r2 = np.mean([s['r2'] for s in meta_cv_scores])
        self.logger.info(f"   ✓ Meta-model: Avg R² = {avg_r2:.3f}")
        
        # ANALYZE LEARNED WEIGHTS
        self._analyze_learned_weights(meta_X.columns)
        
        return {
            'component_scores': cv_scores,
            'meta_scores': meta_cv_scores
        }
    
    def _analyze_learned_weights(self, feature_names):
        """Analyze what weights the meta-model learned."""
        
        importance = self.meta_model.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'importance_pct': importance / importance.sum() * 100
        }).sort_values('importance', ascending=False)
        
        self.logger.info(f"\n📊 LEARNED WEIGHTS (Data-Driven):")
        self.logger.info(f"   Compare to heuristic: bounce=25%, hold=20%, trade=20%, speed=20%, volume=15%")
        self.logger.info(f"")
        
        for idx, row in importance_df.head(10).iterrows():
            self.logger.info(f"   {row['feature']:<20} {row['importance_pct']:>6.1f}%")
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict quality scores using multi-task model.
        
        Process:
        1. Get predictions from each component model
        2. Feed to meta-model for optimal combination
        
        Args:
            features: Input features
            
        Returns:
            Predicted quality scores
        """
        
        X = features[self.feature_names].fillna(0.0)
        
        # Get component predictions
        component_preds = {}
        for component_name, model in self.component_models.items():
            component_preds[component_name] = model.predict(X)
        
        # Create meta-features
        meta_X = pd.DataFrame(component_preds)
        
        # Add context features if available
        context_features = ['feature_strength', 'feature_volatility', 'feature_market_trend']
        available_context = [f for f in context_features if f in features.columns]
        if available_context:
            meta_X = pd.concat([meta_X, features[available_context]], axis=1)
        
        # Meta-model prediction
        predictions = self.meta_model.predict(meta_X)
        
        return np.clip(predictions, 0, 1)


class AdaptiveWeightModel:
    """
    Alternative approach: Learn weights that adapt to market conditions.
    
    Instead of fixed weights, learn:
    quality_score = f(bounce, hold, trade, speed, volume | market_condition)
    
    Weights change based on:
    - Volatility regime (high vol → prioritize hold)
    - Trend regime (uptrend → different weights)
    - Timeframe
    """
    
    def __init__(self):
        self.model = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def train(self, training_data: pd.DataFrame) -> Dict:
        """
        Train adaptive weight model.
        
        Features include:
        - Raw component metrics (bounce, hold, etc.)
        - Market regime indicators (volatility, trend)
        - Interaction terms (bounce * volatility, etc.)
        
        Model learns different weight combinations for different conditions.
        """
        
        self.logger.info("🚀 Training Adaptive Weight Model")
        
        # Create interaction features
        X = self._create_interaction_features(training_data)
        y = training_data['quality_score']
        
        # Train with CV
        tscv = TimeSeriesSplit(n_splits=5)
        fold_models = []
        cv_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_data_lgb = lgb.Dataset(X_train, label=y_train)
            val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'lambda_l1': 1.0,
                'lambda_l2': 1.0,
                'verbose': -1,
                'seed': 42
            }
            
            model = lgb.train(
                params,
                train_data_lgb,
                num_boost_round=500,
                valid_sets=[val_data_lgb],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            cv_scores.append(r2)
            fold_models.append(model)
        
        best_idx = np.argmax(cv_scores)
        self.model = fold_models[best_idx]
        
        self.logger.info(f"   ✓ Avg R²: {np.mean(cv_scores):.3f}")
        
        # Analyze learned interactions
        self._analyze_interactions(X.columns)
        
        return {'cv_scores': cv_scores}
    
    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features for adaptive weighting.
        
        Examples:
        - bounce * is_high_volatility (bounce matters more in low vol)
        - hold * is_support (support holds differently than resistance)
        - trade * market_trend (trend affects trade quality)
        """
        
        features = []
        
        # Base components
        components = ['bounce_strength', 'hold_strength', 'trade_profit', 
                     'rejection_speed', 'volume_quality']
        
        for comp in components:
            if comp in data.columns:
                features.append(data[comp])
        
        # Market regime features
        regime_features = ['feature_is_high_volatility', 'feature_is_uptrend', 
                          'feature_market_volatility', 'feature_market_trend']
        
        for feat in regime_features:
            if feat in data.columns:
                features.append(data[feat])
        
        # Interaction terms
        if 'bounce_strength' in data.columns and 'feature_market_volatility' in data.columns:
            features.append(data['bounce_strength'] * data['feature_market_volatility'])
        
        if 'hold_strength' in data.columns and 'feature_is_high_volatility' in data.columns:
            features.append(data['hold_strength'] * data['feature_is_high_volatility'])
        
        if 'trade_profit' in data.columns and 'feature_market_trend' in data.columns:
            features.append(data['trade_profit'] * data['feature_market_trend'])
        
        # Combine
        X = pd.concat(features, axis=1)
        X.columns = [f"feature_{i}" for i in range(len(X.columns))]
        
        return X
    
    def _analyze_interactions(self, feature_names):
        """Analyze which interactions the model found important."""
        
        importance = self.model.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.logger.info(f"\n📊 KEY INTERACTIONS LEARNED:")
        for idx, row in importance_df.head(10).iterrows():
            self.logger.info(f"   {row['feature']}")
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict with adaptive weights."""
        X = self._create_interaction_features(features)
        return np.clip(self.model.predict(X), 0, 1)

