"""
Predictive SR Engine - Uses optimized weights to predict future SR level quality.

This module implements a comprehensive predictive system that uses the learned
weight optimization results to predict which SR levels will be most effective
in future market conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

from ..logger import system_logger
from .sr_backtesting_engine import SRBacktestingEngine, BacktestResult, SRLevel
from .weight_optimization_engine import WeightOptimizationEngine, WeightOptimizationConfig

@dataclass
class PredictiveConfig:
    """Configuration for predictive SR engine."""
    # Model parameters
    model_type: str = 'ensemble'  # 'ensemble', 'ridge', 'elastic_net', 'random_forest'
    ensemble_models: List[str] = field(default_factory=lambda: ['ridge', 'random_forest', 'gradient_boosting'])
    
    # Feature engineering
    include_market_context: bool = True
    include_time_features: bool = True
    include_volatility_features: bool = True
    include_volume_features: bool = True
    
    # Prediction parameters
    prediction_horizon_days: int = 30  # How far ahead to predict
    confidence_threshold: float = 0.7  # Minimum confidence for predictions
    quality_threshold: float = 0.6     # Minimum predicted quality to consider level "good"
    
    # Model training
    min_training_samples: int = 100
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    
    # Feature selection
    feature_importance_threshold: float = 0.01
    max_features: int = 20

@dataclass
class SRPrediction:
    """Prediction result for an SR level."""
    level: SRLevel
    predicted_quality: float
    confidence: float
    prediction_horizon: int
    key_factors: Dict[str, float]  # Feature contributions
    market_context: Dict[str, Any]
    prediction_date: datetime
    model_used: str

class PredictiveSREngine:
    """Engine for predicting future SR level quality using optimized weights."""
    
    def __init__(self, config: Optional[PredictiveConfig] = None):
        self.config = config or PredictiveConfig()
        self.logger = system_logger.getChild('PredictiveSREngine')
        
        # Core components
        self.backtesting_engine: Optional[SRBacktestingEngine] = None
        self.weight_optimizer: Optional[WeightOptimizationEngine] = None
        
        # Predictive models
        self.quality_predictor: Optional[Any] = None
        self.feature_scaler: Optional[StandardScaler] = None
        self.feature_importance: Dict[str, float] = {}
        
        # Training data
        self.training_data: List[BacktestResult] = []
        self.optimized_weights: Dict[str, float] = {}
        self.market_context_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.prediction_history: List[SRPrediction] = []
        self.model_performance: Dict[str, float] = {}
        
    def train_predictive_model(self, historical_data: pd.DataFrame, 
                             sr_levels: List[SRLevel],
                             optimize_weights: bool = True) -> Dict[str, Any]:
        """Train the predictive model using historical data and optimized weights."""
        try:
            self.logger.info(f"Training predictive model with {len(sr_levels)} SR levels")
            
            # Step 1: Run backtesting on historical data
            if not self.backtesting_engine:
                from .sr_backtesting_engine import get_backtesting_engine, BacktestConfig
                backtest_config = BacktestConfig()
                self.backtesting_engine = get_backtesting_engine(backtest_config)
            
            backtest_results = self.backtesting_engine.backtest_multiple_levels(sr_levels, historical_data)
            self.training_data = backtest_results
            
            self.logger.info(f"Backtesting completed. Average quality: {np.mean([r.quality_score for r in backtest_results]):.3f}")
            
            # Step 2: Optimize weights if requested
            if optimize_weights:
                self._optimize_weights(backtest_results, historical_data)
            
            # Step 3: Engineer predictive features
            feature_matrix, target_scores = self._engineer_predictive_features(
                backtest_results, historical_data
            )
            
            if len(feature_matrix) < self.config.min_training_samples:
                self.logger.warning(f"Insufficient training samples: {len(feature_matrix)} < {self.config.min_training_samples}")
                return {'status': 'insufficient_data'}
            
            # Step 4: Train predictive model
            model_result = self._train_quality_predictor(feature_matrix, target_scores)
            
            # Step 5: Validate model performance
            validation_result = self._validate_model(feature_matrix, target_scores)
            
            training_result = {
                'status': 'success',
                'training_samples': len(feature_matrix),
                'model_performance': model_result,
                'validation_performance': validation_result,
                'optimized_weights': self.optimized_weights,
                'feature_importance': self.feature_importance
            }
            
            self.logger.info(f"Predictive model training completed. R² Score: {validation_result.get('r2_score', 0.0):.3f}")
            
            return training_result
            
        except Exception as e:
            self.logger.error(f"Predictive model training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _optimize_weights(self, backtest_results: List[BacktestResult], 
                         historical_data: pd.DataFrame) -> None:
        """Optimize weights using the weight optimization engine."""
        try:
            if not self.weight_optimizer:
                from .weight_optimization_engine import get_weight_optimization_engine, WeightOptimizationConfig
                weight_config = WeightOptimizationConfig(
                    optimization_method='scipy_minimize',
                    primary_objective='r2_score',
                    secondary_objective='stability'
                )
                self.weight_optimizer = get_weight_optimization_engine(weight_config)
            
            optimization_result = self.weight_optimizer.optimize_weights(backtest_results, historical_data)
            
            if optimization_result and optimization_result.get('optimization_success', False):
                self.optimized_weights = optimization_result.get('best_weights', {})
                self.logger.info(f"Weight optimization completed. Best score: {optimization_result.get('best_score', 0.0):.4f}")
            else:
                self.logger.warning("Weight optimization failed, using default weights")
                self.optimized_weights = {}
                
        except Exception as e:
            self.logger.warning(f"Weight optimization failed: {e}")
            self.optimized_weights = {}
    
    def _engineer_predictive_features(self, backtest_results: List[BacktestResult], 
                                    historical_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Engineer features for predictive modeling."""
        try:
            features = []
            targets = []
            
            for result in backtest_results:
                # Extract SR level features
                level_features = self._extract_level_features(result)
                
                # Extract market context features
                if self.config.include_market_context:
                    market_features = self._extract_market_context_features(result, historical_data)
                    level_features.update(market_features)
                
                # Extract time-based features
                if self.config.include_time_features:
                    time_features = self._extract_time_features(result)
                    level_features.update(time_features)
                
                # Extract volatility features
                if self.config.include_volatility_features:
                    volatility_features = self._extract_volatility_features(result, historical_data)
                    level_features.update(volatility_features)
                
                # Extract volume features
                if self.config.include_volume_features:
                    volume_features = self._extract_volume_features(result, historical_data)
                    level_features.update(volume_features)
                
                # Convert to array
                feature_array = np.array(list(level_features.values()))
                features.append(feature_array)
                targets.append(result.quality_score)
            
            feature_matrix = np.array(features)
            target_scores = np.array(targets)
            
            self.logger.info(f"Engineered {feature_matrix.shape[1]} features for {len(features)} samples")
            
            return feature_matrix, target_scores
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return np.array([]), np.array([])
    
    def _extract_level_features(self, result: BacktestResult) -> Dict[str, float]:
        """Extract features from SR level backtest result."""
        return {
            'success_rate': result.success_rate,
            'avg_bounce_strength': result.avg_bounce_strength,
            'max_bounce_strength': result.max_bounce_strength,
            'total_touches': result.total_touches,
            'time_persistence': result.time_persistence,
            'total_volume_at_level': result.total_volume_at_level,
            'avg_hold_time': result.avg_hold_time,
            'penetration_depth': result.penetration_depth,
            'penetration_frequency': result.penetration_frequency,
            'pattern_consistency': result.pattern_consistency,
            'pattern_strength': result.pattern_strength,
            'order_flow_confirmation': result.order_flow_confirmation,
            'absorption_patterns': result.absorption_patterns,
            'structure_break': result.structure_break
        }
    
    def _extract_market_context_features(self, result: BacktestResult, 
                                       historical_data: pd.DataFrame) -> Dict[str, float]:
        """Extract market context features using existing step06 features."""
        try:
            # Use existing step06 features if available
            if 'step06_features' in historical_data.columns:
                # Extract step06 features from the data
                step06_features = historical_data['step06_features'].iloc[-1] if len(historical_data) > 0 else {}
                
                if isinstance(step06_features, dict):
                    # Calculate VWAP momentum if available
                    vwap_momentum = 0.0
                    if 'VWAP' in step06_features and 'close' in historical_data.columns:
                        current_price = historical_data['close'].iloc[-1] if len(historical_data) > 0 else 0.0
                        vwap_value = step06_features.get('VWAP', current_price)
                        vwap_momentum = (current_price - vwap_value) / vwap_value if vwap_value > 0 else 0.0
                    
                    return {
                        'market_regime': step06_features.get('Market_Regime', 0.0),
                        'volatility_regime': step06_features.get('ATR_14', 0.0),
                        'trend_strength': step06_features.get('SMA_5', 0.0) / step06_features.get('SMA_100', 1.0) - 1.0 if step06_features.get('SMA_100', 0) > 0 else 0.0,
                        'volume_regime': step06_features.get('Volume_Ratio', 0.0),
                        'time_of_day_effect': step06_features.get('Time_of_Day', 0.0),
                        'vwap_momentum': vwap_momentum,
                        'price_momentum': step06_features.get('Price_Momentum', 0.0),
                        'momentum_volume_interaction': step06_features.get('Price_Momentum', 0.0) * step06_features.get('Volume_Ratio', 0.0)
                    }
            
            # Fallback: Calculate basic features if step06 features not available
            if len(historical_data) > 0:
                # Volatility regime (using ATR-like calculation)
                returns = historical_data['close'].pct_change().dropna()
                volatility = returns.rolling(20).std()
                volatility_regime = np.mean(volatility) if len(volatility) > 0 else 0.0
                
                # Trend strength (using SMA-like calculation)
                sma_short = historical_data['close'].rolling(10).mean()
                sma_long = historical_data['close'].rolling(50).mean()
                trend_strength = abs(np.mean((sma_short - sma_long) / sma_long)) if len(sma_short) > 0 else 0.0
                
                # Volume regime
                volume_avg = historical_data['volume'].mean() if 'volume' in historical_data.columns else 1.0
                volume_regime = volume_avg / 1000000  # Normalize
                
                # Market momentum
                momentum = (historical_data['close'].iloc[-1] / historical_data['close'].iloc[0] - 1) if len(historical_data) > 0 else 0.0
                
                # Calculate VWAP momentum from price data
                vwap_momentum = 0.0
                if len(historical_data) > 20:  # Need enough data for VWAP calculation
                    # Simple VWAP calculation
                    vwap = (historical_data['close'] * historical_data['volume']).rolling(20).sum() / historical_data['volume'].rolling(20).sum()
                    current_price = historical_data['close'].iloc[-1]
                    current_vwap = vwap.iloc[-1]
                    vwap_momentum = (current_price - current_vwap) / current_vwap if current_vwap > 0 else 0.0
                
                return {
                    'market_regime': 0.0,  # Default neutral
                    'volatility_regime': volatility_regime,
                    'trend_strength': trend_strength,
                    'volume_regime': volume_regime,
                    'time_of_day_effect': 0.0,  # Default neutral
                    'vwap_momentum': vwap_momentum,
                    'price_momentum': momentum,
                    'momentum_volume_interaction': momentum * volume_regime
                }
            else:
                return {
                    'market_regime': 0.0,
                    'volatility_regime': 0.0,
                    'trend_strength': 0.0,
                    'volume_regime': 0.0,
                    'time_of_day_effect': 0.0,
                    'vwap_momentum': 0.0,
                    'price_momentum': 0.0,
                    'momentum_volume_interaction': 0.0
                }
                
        except Exception as e:
            self.logger.warning(f"Market context extraction failed: {e}")
            return {}
    
    def _extract_time_features(self, result: BacktestResult) -> Dict[str, float]:
        """Extract time-based features."""
        try:
            # Time since first touch
            time_since_first = (datetime.now() - result.first_touch).days if result.first_touch else 0
            
            # Time since last touch
            time_since_last = (datetime.now() - result.last_touch).days if result.last_touch else 0
            
            # Touch frequency (touches per day)
            total_days = max(1, (result.last_touch - result.first_touch).days) if result.first_touch and result.last_touch else 1
            touch_frequency = result.total_touches / total_days
            
            # Time of day effects (if available)
            hour_of_day = result.first_touch.hour if result.first_touch else 12
            day_of_week = result.first_touch.weekday() if result.first_touch else 0
            
            return {
                'time_since_first_touch': time_since_first,
                'time_since_last_touch': time_since_last,
                'touch_frequency_daily': touch_frequency,
                'hour_of_day': hour_of_day,
                'day_of_week': day_of_week
            }
            
        except Exception as e:
            self.logger.warning(f"Time feature extraction failed: {e}")
            return {}
    
    def _extract_volatility_features(self, result: BacktestResult, 
                                   historical_data: pd.DataFrame) -> Dict[str, float]:
        """Extract volatility-related features."""
        try:
            if len(historical_data) > 0:
                returns = historical_data['close'].pct_change().dropna()
                
                # Historical volatility
                historical_vol = returns.std() * np.sqrt(252)  # Annualized
                
                # Recent volatility (last 20 days)
                recent_vol = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else historical_vol
                
                # Volatility trend
                vol_trend = (recent_vol - historical_vol) / historical_vol if historical_vol > 0 else 0.0
                
                # Volatility at SR level
                level_volatility = self._calculate_level_volatility(result, historical_data)
                
                return {
                    'historical_volatility': historical_vol,
                    'recent_volatility': recent_vol,
                    'volatility_trend': vol_trend,
                    'level_volatility': level_volatility
                }
            else:
                return {
                    'historical_volatility': 0.0,
                    'recent_volatility': 0.0,
                    'volatility_trend': 0.0,
                    'level_volatility': 0.0
                }
                
        except Exception as e:
            self.logger.warning(f"Volatility feature extraction failed: {e}")
            return {}
    
    def _extract_volume_features(self, result: BacktestResult, 
                               historical_data: pd.DataFrame) -> Dict[str, float]:
        """Extract volume-related features."""
        try:
            if len(historical_data) > 0 and 'volume' in historical_data.columns:
                # Average volume
                avg_volume = historical_data['volume'].mean()
                
                # Volume trend
                recent_volume = historical_data['volume'].tail(20).mean() if len(historical_data) >= 20 else avg_volume
                volume_trend = (recent_volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0
                
                # Volume at SR level
                level_volume = self._calculate_level_volume(result, historical_data)
                
                # Volume confirmation ratio
                volume_confirmation = level_volume / avg_volume if avg_volume > 0 else 1.0
                
                return {
                    'avg_volume': avg_volume,
                    'volume_trend': volume_trend,
                    'level_volume': level_volume,
                    'volume_confirmation_ratio': volume_confirmation
                }
            else:
                return {
                    'avg_volume': 0.0,
                    'volume_trend': 0.0,
                    'level_volume': 0.0,
                    'volume_confirmation_ratio': 1.0
                }
                
        except Exception as e:
            self.logger.warning(f"Volume feature extraction failed: {e}")
            return {}
    
    def _calculate_level_volatility(self, result: BacktestResult, 
                                  historical_data: pd.DataFrame) -> float:
        """Calculate volatility specifically at the SR level."""
        try:
            if len(historical_data) == 0:
                return 0.0
            
            # Find data points near the SR level
            tolerance = 0.01  # 1% tolerance
            level_price = result.price
            
            near_level = historical_data[
                (historical_data['close'] >= level_price * (1 - tolerance)) &
                (historical_data['close'] <= level_price * (1 + tolerance))
            ]
            
            if len(near_level) > 1:
                returns = near_level['close'].pct_change().dropna()
                return returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Level volatility calculation failed: {e}")
            return 0.0
    
    def _calculate_level_volume(self, result: BacktestResult, 
                              historical_data: pd.DataFrame) -> float:
        """Calculate volume specifically at the SR level."""
        try:
            if len(historical_data) == 0 or 'volume' not in historical_data.columns:
                return 0.0
            
            # Find data points near the SR level
            tolerance = 0.01  # 1% tolerance
            level_price = result.price
            
            near_level = historical_data[
                (historical_data['close'] >= level_price * (1 - tolerance)) &
                (historical_data['close'] <= level_price * (1 + tolerance))
            ]
            
            if len(near_level) > 0:
                return near_level['volume'].mean()
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Level volume calculation failed: {e}")
            return 0.0
    
    def _train_quality_predictor(self, feature_matrix: np.ndarray, 
                               target_scores: np.ndarray) -> Dict[str, Any]:
        """Train the quality prediction model."""
        try:
            # Scale features
            self.feature_scaler = RobustScaler()
            scaled_features = self.feature_scaler.fit_transform(feature_matrix)
            
            # Train ensemble model
            if self.config.model_type == 'ensemble':
                models = []
                predictions = []
                
                for model_name in self.config.ensemble_models:
                    if model_name == 'ridge':
                        model = Ridge(alpha=1.0)
                    elif model_name == 'elastic_net':
                        model = ElasticNet(alpha=0.1, l1_ratio=0.5)
                    elif model_name == 'random_forest':
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    elif model_name == 'gradient_boosting':
                        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    else:
                        continue
                    
                    # Train model
                    model.fit(scaled_features, target_scores)
                    models.append((model_name, model))
                    
                    # Get predictions for ensemble
                    pred = model.predict(scaled_features)
                    predictions.append(pred)
                
                # Create ensemble predictor
                self.quality_predictor = models
                
                # Calculate feature importance (average across models)
                feature_importance = {}
                for model_name, model in models:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        importances = np.abs(model.coef_)
                    else:
                        continue
                    
                    for i, importance in enumerate(importances):
                        feature_name = f"feature_{i}"
                        if feature_name not in feature_importance:
                            feature_importance[feature_name] = []
                        feature_importance[feature_name].append(importance)
                
                # Average feature importance
                for feature_name in feature_importance:
                    feature_importance[feature_name] = np.mean(feature_importance[feature_name])
                
                self.feature_importance = feature_importance
                
                # Calculate ensemble predictions
                ensemble_predictions = np.mean(predictions, axis=0)
                
                # Calculate performance metrics
                r2 = r2_score(target_scores, ensemble_predictions)
                mse = mean_squared_error(target_scores, ensemble_predictions)
                mae = mean_absolute_error(target_scores, ensemble_predictions)
                
                return {
                    'model_type': 'ensemble',
                    'models_trained': len(models),
                    'r2_score': r2,
                    'mse': mse,
                    'mae': mae,
                    'feature_importance': feature_importance
                }
            
            else:
                # Single model training
                if self.config.model_type == 'ridge':
                    model = Ridge(alpha=1.0)
                elif self.config.model_type == 'elastic_net':
                    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
                elif self.config.model_type == 'random_forest':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model = Ridge(alpha=1.0)  # Default
                
                model.fit(scaled_features, target_scores)
                self.quality_predictor = model
                
                # Calculate feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_)
                else:
                    importances = np.ones(feature_matrix.shape[1]) / feature_matrix.shape[1]
                
                feature_importance = {f"feature_{i}": importance for i, importance in enumerate(importances)}
                self.feature_importance = feature_importance
                
                # Calculate performance metrics
                predictions = model.predict(scaled_features)
                r2 = r2_score(target_scores, predictions)
                mse = mean_squared_error(target_scores, predictions)
                mae = mean_absolute_error(target_scores, predictions)
                
                return {
                    'model_type': self.config.model_type,
                    'r2_score': r2,
                    'mse': mse,
                    'mae': mae,
                    'feature_importance': feature_importance
                }
                
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _validate_model(self, feature_matrix: np.ndarray, 
                       target_scores: np.ndarray) -> Dict[str, float]:
        """Validate the trained model using cross-validation."""
        try:
            if self.feature_scaler is None or self.quality_predictor is None:
                return {'status': 'no_model'}
            
            # Scale features
            scaled_features = self.feature_scaler.transform(feature_matrix)
            
            # Cross-validation
            cv_scores = []
            tscv = TimeSeriesSplit(n_splits=self.config.cross_validation_folds)
            
            for train_idx, val_idx in tscv.split(scaled_features):
                X_train, X_val = scaled_features[train_idx], scaled_features[val_idx]
                y_train, y_val = target_scores[train_idx], target_scores[val_idx]
                
                # Train model on fold
                if isinstance(self.quality_predictor, list):  # Ensemble
                    fold_models = []
                    for model_name, _ in self.quality_predictor:
                        if model_name == 'ridge':
                            model = Ridge(alpha=1.0)
                        elif model_name == 'elastic_net':
                            model = ElasticNet(alpha=0.1, l1_ratio=0.5)
                        elif model_name == 'random_forest':
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                        elif model_name == 'gradient_boosting':
                            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                        else:
                            continue
                        
                        model.fit(X_train, y_train)
                        fold_models.append(model)
                    
                    # Ensemble prediction
                    fold_predictions = []
                    for model in fold_models:
                        pred = model.predict(X_val)
                        fold_predictions.append(pred)
                    
                    fold_pred = np.mean(fold_predictions, axis=0)
                else:  # Single model
                    if self.config.model_type == 'ridge':
                        model = Ridge(alpha=1.0)
                    elif self.config.model_type == 'elastic_net':
                        model = ElasticNet(alpha=0.1, l1_ratio=0.5)
                    elif self.config.model_type == 'random_forest':
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    else:
                        model = Ridge(alpha=1.0)
                    
                    model.fit(X_train, y_train)
                    fold_pred = model.predict(X_val)
                
                # Calculate score
                fold_score = r2_score(y_val, fold_pred)
                cv_scores.append(fold_score)
            
            # Calculate validation metrics
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            return {
                'cv_r2_mean': cv_mean,
                'cv_r2_std': cv_std,
                'cv_scores': cv_scores,
                'validation_samples': len(target_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def predict_sr_quality(self, sr_level: SRLevel, 
                          current_market_data: pd.DataFrame,
                          prediction_horizon: Optional[int] = None) -> SRPrediction:
        """Predict the future quality of an SR level."""
        try:
            if self.quality_predictor is None or self.feature_scaler is None:
                raise ValueError("Model not trained. Call train_predictive_model first.")
            
            prediction_horizon = prediction_horizon or self.config.prediction_horizon_days
            
            # Create a temporary backtest result for feature extraction
            temp_result = BacktestResult(
                price=sr_level.price,
                level_type=sr_level.level_type,
                strength=sr_level.strength,
                first_touch=sr_level.first_touch,
                last_touch=sr_level.last_touch,
                total_touches=sr_level.touch_count,
                success_rate=0.0,  # Will be predicted
                avg_bounce_strength=0.0,
                max_bounce_strength=0.0,
                time_persistence=0.0,
                total_volume_at_level=0.0,
                avg_hold_time=0.0,
                penetration_depth=0.0,
                penetration_frequency=0.0,
                pattern_consistency=0.0,
                pattern_strength=0.0,
                order_flow_confirmation=0.0,
                absorption_patterns=0.0,
                structure_break=0.0,
                quality_score=0.0
            )
            
            # Extract features
            level_features = self._extract_level_features(temp_result)
            
            if self.config.include_market_context:
                market_features = self._extract_market_context_features(temp_result, current_market_data)
                level_features.update(market_features)
            
            if self.config.include_time_features:
                time_features = self._extract_time_features(temp_result)
                level_features.update(time_features)
            
            if self.config.include_volatility_features:
                volatility_features = self._extract_volatility_features(temp_result, current_market_data)
                level_features.update(volatility_features)
            
            if self.config.include_volume_features:
                volume_features = self._extract_volume_features(temp_result, current_market_data)
                level_features.update(volume_features)
            
            # Convert to array and scale
            feature_array = np.array(list(level_features.values())).reshape(1, -1)
            scaled_features = self.feature_scaler.transform(feature_array)
            
            # Make prediction
            if isinstance(self.quality_predictor, list):  # Ensemble
                predictions = []
                for model_name, model in self.quality_predictor:
                    pred = model.predict(scaled_features)[0]
                    predictions.append(pred)
                
                predicted_quality = np.mean(predictions)
                confidence = 1.0 - np.std(predictions)  # Lower std = higher confidence
            else:  # Single model
                predicted_quality = self.quality_predictor.predict(scaled_features)[0]
                confidence = 0.8  # Default confidence for single model
            
            # Calculate key factors (feature contributions)
            key_factors = self._calculate_feature_contributions(level_features, scaled_features[0])
            
            # Extract market context
            market_context = self._extract_market_context_features(temp_result, current_market_data)
            
            # Create prediction result
            prediction = SRPrediction(
                level=sr_level,
                predicted_quality=predicted_quality,
                confidence=confidence,
                prediction_horizon=prediction_horizon,
                key_factors=key_factors,
                market_context=market_context,
                prediction_date=datetime.now(),
                model_used=self.config.model_type
            )
            
            # Store prediction
            self.prediction_history.append(prediction)
            
            self.logger.info(f"Predicted quality for SR level at {sr_level.price}: {predicted_quality:.3f} (confidence: {confidence:.3f})")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"SR quality prediction failed: {e}")
            raise
    
    def _calculate_feature_contributions(self, features: Dict[str, float], 
                                       scaled_features: np.ndarray) -> Dict[str, float]:
        """Calculate the contribution of each feature to the prediction."""
        try:
            contributions = {}
            
            if isinstance(self.quality_predictor, list):  # Ensemble
                # Average contributions across models
                for model_name, model in self.quality_predictor:
                    if hasattr(model, 'coef_'):
                        coefs = model.coef_
                    elif hasattr(model, 'feature_importances_'):
                        coefs = model.feature_importances_
                    else:
                        continue
                    
                    for i, (feature_name, feature_value) in enumerate(features.items()):
                        if i < len(coefs):
                            contribution = coefs[i] * scaled_features[i]
                            if feature_name not in contributions:
                                contributions[feature_name] = []
                            contributions[feature_name].append(contribution)
                
                # Average contributions
                for feature_name in contributions:
                    contributions[feature_name] = np.mean(contributions[feature_name])
            else:  # Single model
                if hasattr(self.quality_predictor, 'coef_'):
                    coefs = self.quality_predictor.coef_
                elif hasattr(self.quality_predictor, 'feature_importances_'):
                    coefs = self.quality_predictor.feature_importances_
                else:
                    return {}
                
                for i, (feature_name, feature_value) in enumerate(features.items()):
                    if i < len(coefs):
                        contributions[feature_name] = coefs[i] * scaled_features[i]
            
            return contributions
            
        except Exception as e:
            self.logger.warning(f"Feature contribution calculation failed: {e}")
            return {}
    
    def get_high_quality_predictions(self, sr_levels: List[SRLevel], 
                                   current_market_data: pd.DataFrame,
                                   min_quality: Optional[float] = None,
                                   min_confidence: Optional[float] = None) -> List[SRPrediction]:
        """Get predictions for SR levels that are predicted to be high quality."""
        try:
            min_quality = min_quality or self.config.quality_threshold
            min_confidence = min_confidence or self.config.confidence_threshold
            
            predictions = []
            
            for sr_level in sr_levels:
                try:
                    prediction = self.predict_sr_quality(sr_level, current_market_data)
                    
                    if (prediction.predicted_quality >= min_quality and 
                        prediction.confidence >= min_confidence):
                        predictions.append(prediction)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to predict quality for level at {sr_level.price}: {e}")
                    continue
            
            # Sort by predicted quality (descending)
            predictions.sort(key=lambda x: x.predicted_quality, reverse=True)
            
            self.logger.info(f"Found {len(predictions)} high-quality SR level predictions")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"High quality prediction failed: {e}")
            return []
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get a summary of prediction performance and history."""
        if not self.prediction_history:
            return {'status': 'no_predictions'}
        
        predictions = self.prediction_history
        predicted_qualities = [p.predicted_quality for p in predictions]
        confidences = [p.confidence for p in predictions]
        
        return {
            'total_predictions': len(predictions),
            'avg_predicted_quality': np.mean(predicted_qualities),
            'avg_confidence': np.mean(confidences),
            'high_quality_predictions': len([p for p in predictions if p.predicted_quality >= self.config.quality_threshold]),
            'high_confidence_predictions': len([p for p in predictions if p.confidence >= self.config.confidence_threshold]),
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance
        }

def get_predictive_sr_engine(config: Optional[PredictiveConfig] = None) -> PredictiveSREngine:
    """Get a predictive SR engine instance."""
    return PredictiveSREngine(config)