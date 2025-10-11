"""
Robust Scoring Models for Regime Quality Prediction

This module provides regression and classification models trained on historical data
to predict regime quality (economic significance, trading viability, stability).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from src.utils.tprint import (

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)


@dataclass
class ScoringModelResult:
    """Result from scoring model prediction."""
    economic_significance: float
    trading_viability: float
    stability_score: float
    risk_score: float
    performance_score: float
    confidence_scores: Dict[str, float]
    model_metadata: Dict[str, Any]


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    target_metric: str
    train_score: float
    test_score: float
    cv_score: float
    cv_std: float
    feature_importance: Dict[str, float]
    model_metadata: Dict[str, Any]


class RobustScoringModels:
    """
    Robust scoring models for regime quality prediction.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize robust scoring models."""
        tprint_info("🚀 Initializing Robust Scoring Models")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Model parameters
        tprint_debug("⚙️ Setting model parameters...")
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_state', 42)
        self.cv_folds = config.get('cv_folds', 5)
        self.enable_feature_scaling = config.get('enable_feature_scaling', True)
        self.model_selection_strategy = config.get('model_selection_strategy', 'ensemble')
        tprint_success("✅ Model parameters configured")

        # Model configurations
        tprint_debug("🔧 Setting model configurations...")
        self.regression_models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'gradient_boosting': GradientBoostingRegressor(random_state=self.random_state),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'svr': SVR(kernel='rbf'),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=self.random_state, max_iter=1000)
        }
        
        self.classification_models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'svc': SVC(kernel='rbf', probability=True, random_state=self.random_state),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=self.random_state, max_iter=1000)
        }
        tprint_success(f"✅ {len(self.regression_models)} regression models, {len(self.classification_models)} classification models configured")

        # Scoring targets
        tprint_debug("🎯 Setting scoring targets...")
        self.scoring_targets = {
            'economic_significance': 'regression',
            'trading_viability': 'regression',
            'stability_score': 'regression',
            'risk_score': 'regression',
            'performance_score': 'regression',
            'regime_quality_class': 'classification'  # High/Medium/Low quality
        }
        tprint_success("✅ Scoring targets configured")

        # Initialize models
        self.trained_models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        tprint_success("✅ Robust Scoring Models initialized")
        self.logger.info("✅ Robust Scoring Models initialized")

    def train_scoring_models(self, 
                           historical_data: pd.DataFrame,
                           regime_features: np.ndarray,
                           regime_labels: np.ndarray,
                           regime_metrics: List[Dict[str, Any]]) -> Dict[str, ModelPerformance]:
        """
        Train scoring models on historical data.

        Args:
            historical_data: Historical market data
            regime_features: Feature matrix for regimes
            regime_labels: Regime labels
            regime_metrics: Historical regime metrics

        Returns:
            Dictionary of trained model performances
        """
        try:
            tprint("🔍 [ROBUST_SCORING] Starting scoring model training", color="blue", bold=True)
            tprint_debug(f"📊 [ROBUST_SCORING] Historical data shape: {historical_data.shape}")
            tprint_debug(f"📊 [ROBUST_SCORING] Regime features shape: {regime_features.shape}")
            tprint_debug(f"📊 [ROBUST_SCORING] Regime labels: {len(set(regime_labels))} unique regimes")
            self.logger.info("🔍 Starting scoring model training...")

            # Prepare training data
            tprint("📊 [ROBUST_SCORING] Preparing training data", color="cyan")
            X, y_targets, feature_names = self._prepare_training_data(
                historical_data, regime_features, regime_labels, regime_metrics
            )
            tprint_success(f"✅ [ROBUST_SCORING] Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
            tprint_debug(f"📊 [ROBUST_SCORING] Target variables: {list(y_targets.keys())}")

            # Train models for each target
            model_performances = {}
            
            for target_name, target_type in self.scoring_targets.items():
                tprint(f"🎯 [ROBUST_SCORING] Training model for {target_name} ({target_type})", color="cyan")
                
                if target_name in y_targets:
                    performance = self._train_single_model(
                        X, y_targets[target_name], target_name, target_type, feature_names
                    )
                    model_performances[target_name] = performance
                    tprint_success(f"✅ [ROBUST_SCORING] Model trained for {target_name}: {performance.test_score:.3f}")
                else:
                    tprint_warning(f"⚠️ [ROBUST_SCORING] Target {target_name} not found in data")

            tprint_success(f"🎉 [ROBUST_SCORING] Scoring model training completed successfully")
            tprint_performance(f"⚡ [ROBUST_SCORING] Final result: {len(model_performances)} models trained")
            
            return model_performances

        except Exception as e:
            tprint_error(f"❌ [ROBUST_SCORING] Scoring model training failed: {e}")
            tprint_debug(f"🔍 [ROBUST_SCORING] Error details: {str(e)}")
            self.logger.error(f"Scoring model training failed: {e}")
            raise

    def predict_regime_scores(self, 
                            features: np.ndarray,
                            market_data: pd.DataFrame,
                            regime_id: int) -> ScoringModelResult:
        """
        Predict regime scores using trained models.

        Args:
            features: Feature matrix for the regime
            market_data: Market data for the regime
            regime_id: Regime identifier

        Returns:
            ScoringModelResult with predicted scores
        """
        try:
            tprint(f"🔍 [ROBUST_SCORING] Predicting scores for regime {regime_id}", color="blue", bold=True)
            tprint_debug(f"📊 [ROBUST_SCORING] Features shape: {features.shape}")
            tprint_debug(f"📊 [ROBUST_SCORING] Market data shape: {market_data.shape}")
            self.logger.info(f"🔍 Predicting scores for regime {regime_id}...")

            # Prepare prediction features
            tprint("📊 [ROBUST_SCORING] Preparing prediction features", color="cyan")
            X_pred = self._prepare_prediction_features(features, market_data)
            tprint_success(f"✅ [ROBUST_SCORING] Prediction features prepared: {X_pred.shape}")

            # Make predictions
            tprint("🎯 [ROBUST_SCORING] Making predictions", color="cyan")
            predictions = {}
            confidence_scores = {}
            
            for target_name in self.scoring_targets.keys():
                if target_name in self.trained_models:
                    prediction, confidence = self._make_single_prediction(X_pred, target_name)
                    predictions[target_name] = prediction
                    confidence_scores[target_name] = confidence
                    tprint_debug(f"📈 [ROBUST_SCORING] {target_name}: {prediction:.3f} (confidence: {confidence:.3f})")
                else:
                    # Use default values if model not trained
                    predictions[target_name] = 0.5
                    confidence_scores[target_name] = 0.0
                    tprint_warning(f"⚠️ [ROBUST_SCORING] Model for {target_name} not available, using default")

            tprint_success(f"🎉 [ROBUST_SCORING] Score prediction completed successfully")
            tprint_performance(f"⚡ [ROBUST_SCORING] Final result: {len(predictions)} predictions made")
            
            return ScoringModelResult(
                economic_significance=predictions.get('economic_significance', 0.5),
                trading_viability=predictions.get('trading_viability', 0.5),
                stability_score=predictions.get('stability_score', 0.5),
                risk_score=predictions.get('risk_score', 0.5),
                performance_score=predictions.get('performance_score', 0.5),
                confidence_scores=confidence_scores,
                model_metadata={
                    'regime_id': regime_id,
                    'n_features': X_pred.shape[1],
                    'prediction_timestamp': datetime.now().isoformat(),
                    'models_used': list(self.trained_models.keys())
                }
            )

        except Exception as e:
            tprint_error(f"❌ [ROBUST_SCORING] Score prediction failed: {e}")
            tprint_debug(f"🔍 [ROBUST_SCORING] Error details: {str(e)}")
            self.logger.error(f"Score prediction failed: {e}")
            # Return default scores
            return ScoringModelResult(
                economic_significance=0.5,
                trading_viability=0.5,
                stability_score=0.5,
                risk_score=0.5,
                performance_score=0.5,
                confidence_scores={},
                model_metadata={'error': str(e)}
            )

    def _prepare_training_data(self, 
                             historical_data: pd.DataFrame,
                             regime_features: np.ndarray,
                             regime_labels: np.ndarray,
                             regime_metrics: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
        """Prepare training data for model training."""
        try:
            # Combine features and market data
            feature_list = []
            feature_names = []
            
            # Add regime features
            for i in range(regime_features.shape[1]):
                feature_list.append(regime_features[:, i])
                feature_names.append(f"regime_feature_{i}")
            
            # Add market data features
            market_features = self._extract_market_features(historical_data)
            for i, feature_name in enumerate(market_features.columns):
                feature_list.append(market_features.iloc[:, i].values)
                feature_names.append(f"market_{feature_name}")
            
            # Combine all features
            X = np.column_stack(feature_list)
            
            # Prepare target variables
            y_targets = {}
            
            # Extract targets from regime metrics
            for i, metrics in enumerate(regime_metrics):
                regime_id = metrics.get('regime_id', i)
                regime_mask = regime_labels == regime_id
                
                if np.any(regime_mask):
                    # Economic significance
                    if 'economic_significance' not in y_targets:
                        y_targets['economic_significance'] = np.zeros(len(regime_labels))
                    y_targets['economic_significance'][regime_mask] = metrics.get('economic_significance', 0.5)
                    
                    # Trading viability
                    if 'trading_viability' not in y_targets:
                        y_targets['trading_viability'] = np.zeros(len(regime_labels))
                    y_targets['trading_viability'][regime_mask] = metrics.get('trading_viability', 0.5)
                    
                    # Stability score
                    if 'stability_score' not in y_targets:
                        y_targets['stability_score'] = np.zeros(len(regime_labels))
                    y_targets['stability_score'][regime_mask] = metrics.get('stability_score', 0.5)
                    
                    # Risk score
                    if 'risk_score' not in y_targets:
                        y_targets['risk_score'] = np.zeros(len(regime_labels))
                    y_targets['risk_score'][regime_mask] = metrics.get('risk_score', 0.5)
                    
                    # Performance score
                    if 'performance_score' not in y_targets:
                        y_targets['performance_score'] = np.zeros(len(regime_labels))
                    y_targets['performance_score'][regime_mask] = metrics.get('performance_score', 0.5)
            
            # Create regime quality classification
            y_targets['regime_quality_class'] = self._create_quality_classification(y_targets)
            
            return X, y_targets, feature_names
            
        except Exception as e:
            self.logger.warning(f"Training data preparation failed: {e}")
            return np.array([]), {}, []

    def _extract_market_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Extract market features from historical data."""
        try:
            features = pd.DataFrame()
            
            # Price features
            if 'close' in market_data.columns:
                close_prices = market_data['close']
                features['returns'] = close_prices.pct_change().fillna(0)
                features['volatility'] = close_prices.pct_change().rolling(20).std().fillna(0)
                features['price_ma_5'] = close_prices.rolling(5).mean().fillna(method='bfill')
                features['price_ma_20'] = close_prices.rolling(20).mean().fillna(method='bfill')
            
            # Volume features
            if 'volume' in market_data.columns:
                volume = market_data['volume']
                features['volume_ma'] = volume.rolling(20).mean().fillna(method='bfill')
                features['volume_ratio'] = volume / features['volume_ma']
                features['volume_volatility'] = volume.pct_change().rolling(20).std().fillna(0)
            
            # Technical indicators
            if 'high' in market_data.columns and 'low' in market_data.columns:
                high = market_data['high']
                low = market_data['low']
                close = market_data['close']
                
                # RSI approximation
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                features['rsi'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                bb_middle = close.rolling(20).mean()
                bb_std = close.rolling(20).std()
                features['bb_upper'] = bb_middle + (bb_std * 2)
                features['bb_lower'] = bb_middle - (bb_std * 2)
                features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            return features.fillna(0)
            
        except Exception as e:
            self.logger.warning(f"Market features extraction failed: {e}")
            return pd.DataFrame()

    def _create_quality_classification(self, y_targets: Dict[str, np.ndarray]) -> np.ndarray:
        """Create regime quality classification."""
        try:
            # Combine multiple scores to create quality classification
            scores = []
            for target in ['economic_significance', 'trading_viability', 'stability_score', 'performance_score']:
                if target in y_targets:
                    scores.append(y_targets[target])
            
            if scores:
                combined_score = np.mean(scores, axis=0)
                
                # Classify into High/Medium/Low quality
                high_threshold = np.percentile(combined_score, 75)
                low_threshold = np.percentile(combined_score, 25)
                
                quality_class = np.zeros(len(combined_score))
                quality_class[combined_score >= high_threshold] = 2  # High
                quality_class[combined_score <= low_threshold] = 0   # Low
                quality_class[(combined_score > low_threshold) & (combined_score < high_threshold)] = 1  # Medium
                
                return quality_class
            else:
                return np.zeros(len(y_targets.get('economic_significance', [])))
                
        except Exception as e:
            self.logger.warning(f"Quality classification creation failed: {e}")
            return np.zeros(100)

    def _train_single_model(self, 
                          X: np.ndarray, 
                          y: np.ndarray, 
                          target_name: str, 
                          target_type: str,
                          feature_names: List[str]) -> ModelPerformance:
        """Train a single model for a specific target."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Scale features if enabled
            if self.enable_feature_scaling:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.scalers[target_name] = scaler
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Select and train model
            if target_type == 'regression':
                models = self.regression_models
            else:
                models = self.classification_models
            
            best_model = None
            best_score = -np.inf
            best_model_name = None
            
            for model_name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate model
                    if target_type == 'regression':
                        train_score = model.score(X_train_scaled, y_train)
                        test_score = model.score(X_test_scaled, y_test)
                    else:
                        train_score = model.score(X_train_scaled, y_train)
                        test_score = model.score(X_test_scaled, y_test)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=self.cv_folds)
                    cv_score = np.mean(cv_scores)
                    cv_std = np.std(cv_scores)
                    
                    # Select best model based on test score
                    if test_score > best_score:
                        best_score = test_score
                        best_model = model
                        best_model_name = model_name
                        
                except Exception as e:
                    self.logger.warning(f"Model {model_name} training failed for {target_name}: {e}")
                    continue
            
            if best_model is None:
                raise ValueError(f"No model could be trained for {target_name}")
            
            # Store trained model
            self.trained_models[target_name] = best_model
            
            # Calculate feature importance
            feature_importance = {}
            if hasattr(best_model, 'feature_importances_'):
                for i, importance in enumerate(best_model.feature_importances_):
                    feature_importance[feature_names[i]] = importance
            elif hasattr(best_model, 'coef_'):
                for i, coef in enumerate(best_model.coef_):
                    feature_importance[feature_names[i]] = abs(coef)
            
            return ModelPerformance(
                model_name=best_model_name,
                target_metric=target_name,
                train_score=train_score,
                test_score=test_score,
                cv_score=cv_score,
                cv_std=cv_std,
                feature_importance=feature_importance,
                model_metadata={
                    'target_type': target_type,
                    'n_features': X.shape[1],
                    'n_samples': X.shape[0]
                }
            )
            
        except Exception as e:
            self.logger.error(f"Single model training failed for {target_name}: {e}")
            raise

    def _prepare_prediction_features(self, features: np.ndarray, market_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction."""
        try:
            # Extract market features
            market_features = self._extract_market_features(market_data)
            
            # Combine regime features and market features
            if len(market_features) > 0:
                # Take the last row of market features (most recent)
                market_row = market_features.iloc[-1].values
                combined_features = np.concatenate([features.flatten(), market_row])
            else:
                combined_features = features.flatten()
            
            return combined_features.reshape(1, -1)
            
        except Exception as e:
            self.logger.warning(f"Prediction features preparation failed: {e}")
            return features.reshape(1, -1)

    def _make_single_prediction(self, X_pred: np.ndarray, target_name: str) -> Tuple[float, float]:
        """Make prediction for a single target."""
        try:
            if target_name not in self.trained_models:
                return 0.5, 0.0
            
            model = self.trained_models[target_name]
            
            # Scale features if scaler is available
            if target_name in self.scalers:
                X_pred_scaled = self.scalers[target_name].transform(X_pred)
            else:
                X_pred_scaled = X_pred
            
            # Make prediction
            prediction = model.predict(X_pred_scaled)[0]
            
            # Calculate confidence (simplified)
            confidence = 0.8  # Default confidence
            
            # For models that support probability prediction
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X_pred_scaled)[0]
                    confidence = np.max(proba)
                except:
                    pass
            
            return prediction, confidence
            
        except Exception as e:
            self.logger.warning(f"Single prediction failed for {target_name}: {e}")
            return 0.5, 0.0

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performances."""
        try:
            summary = {
                'n_trained_models': len(self.trained_models),
                'available_models': list(self.trained_models.keys()),
                'scalers_available': list(self.scalers.keys()),
                'label_encoders_available': list(self.label_encoders.keys())
            }
            return summary
        except Exception as e:
            self.logger.warning(f"Model performance summary failed: {e}")
            return {}


def create_robust_scoring_models(config: Dict[str, Any]) -> RobustScoringModels:
    """Create robust scoring models."""
    return RobustScoringModels(config)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
