"""
ML-Based Label Quality Assessor for Multi-Horizon Profit Labeling

This module provides machine learning-based assessment of label quality, replacing
heuristic scoring with data-driven ML models that learn from historical patterns
and market dynamics.

Key ML Components:
1. Random Forest Feature Importance for Label Relevance
2. XGBoost for Non-linear Label-Return Relationships
3. Neural Networks for Complex Pattern Detection
4. Ensemble Methods for Robust Quality Scores
5. Online Learning for Adaptive Quality Assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import joblib
import warnings
from datetime import datetime

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Optional XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Optional neural network libraries
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger


class MLModelType(Enum):
    """Enumeration of ML model types for label quality assessment."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    ENSEMBLE = "ensemble"


class QualityMetric(Enum):
    """Enumeration of quality assessment metrics."""
    PREDICTIVE_POWER = "predictive_power"
    INFORMATION_CONTENT = "information_content"
    STABILITY_SCORE = "stability_score"
    ECONOMIC_VALUE = "economic_value"
    FEATURE_IMPORTANCE = "feature_importance"
    NOISE_ROBUSTNESS = "noise_robustness"


@dataclass
class MLQualityAssessmentConfig:
    """Configuration for ML-based quality assessment."""
    # Model selection
    primary_model: MLModelType = MLModelType.ENSEMBLE
    ensemble_models: List[MLModelType] = field(default_factory=lambda: [
        MLModelType.RANDOM_FOREST,
        MLModelType.GRADIENT_BOOSTING,
        MLModelType.NEURAL_NETWORK
    ])
    
    # Feature engineering
    max_features: int = 50
    feature_selection_method: str = "mutual_info"  # "f_regression", "mutual_info"
    include_technical_indicators: bool = True
    include_market_microstructure: bool = True
    
    # Training parameters
    train_test_split: float = 0.7
    cv_folds: int = 5
    random_state: int = 42
    
    # Model hyperparameters
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    })
    
    gb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42
    })
    
    nn_params: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'max_iter': 500,
        'random_state': 42
    })
    
    # Quality assessment thresholds
    min_r2_score: float = 0.1
    min_feature_importance: float = 0.01
    stability_window: int = 100
    
    # Online learning parameters
    enable_online_learning: bool = True
    update_frequency: int = 100  # Update model every N samples
    adaptation_rate: float = 0.1


@dataclass
class MLQualityAssessmentResult:
    """Result container for ML quality assessment."""
    model_type: MLModelType
    quality_scores: Dict[QualityMetric, float]
    feature_importance: Dict[str, float]
    model_performance: Dict[str, float]
    predictions: np.ndarray
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class MLLabelQualityAssessor:
    """
    Machine Learning-based label quality assessor.
    
    This class uses various ML models to assess the quality of profit labels
    by learning patterns from historical data and predicting label effectiveness.
    
    Key Features:
    1. **Multi-Model Ensemble**: Combines multiple ML models for robust assessment
    2. **Feature Engineering**: Extracts relevant market features for prediction
    3. **Online Learning**: Adapts to new market conditions continuously
    4. **Quality Metrics**: Comprehensive quality assessment across multiple dimensions
    5. **Interpretability**: Provides feature importance and model explanations
    """
    
    def __init__(self, config: Optional[MLQualityAssessmentConfig] = None):
        """Initialize the ML label quality assessor."""
        self.config = config or MLQualityAssessmentConfig()
        self.logger = get_logger('MLLabelQualityAssessor')
        
        # Model storage
        self.models: Dict[MLModelType, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_selectors: Dict[str, Any] = {}
        
        # Assessment state
        self.assessment_history: List[MLQualityAssessmentResult] = []
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        
        # Online learning state
        self.online_update_counter: int = 0
        self.adaptation_buffer: List[Tuple[np.ndarray, np.ndarray]] = []
        
        self.logger.info('🤖 ML Label Quality Assessor initialized')
        self.logger.info(f'   → Primary model: {self.config.primary_model.value}')
        self.logger.info(f'   → Ensemble models: {[m.value for m in self.config.ensemble_models]}')
    
    def assess_label_quality(self,
                           labeled_data: pd.DataFrame,
                           market_data: pd.DataFrame,
                           target_column: str = 'overall_opportunity') -> MLQualityAssessmentResult:
        """
        Assess label quality using ML models.
        
        Args:
            labeled_data: DataFrame with profit labels
            market_data: Original market data (OHLCV)
            target_column: Column to assess quality for
            
        Returns:
            MLQualityAssessmentResult with comprehensive quality assessment
        """
        self.logger.info(f'🔍 Assessing label quality for {target_column}')
        
        # Prepare features and targets
        features, targets = self._prepare_features_and_targets(
            labeled_data, market_data, target_column
        )
        
        if len(features) < 100:
            self.logger.warning('⚠️ Insufficient data for ML assessment')
            return self._create_fallback_result(target_column)
        
        # Train models if not fitted
        if not self.is_fitted:
            self._fit_models(features, targets)
        
        # Generate predictions and quality scores
        predictions = self._generate_predictions(features)
        quality_scores = self._calculate_quality_scores(predictions, targets, features)
        feature_importance = self._calculate_feature_importance(features)
        model_performance = self._evaluate_model_performance(predictions, targets)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(features, targets)
        
        result = MLQualityAssessmentResult(
            model_type=self.config.primary_model,
            quality_scores=quality_scores,
            feature_importance=feature_importance,
            model_performance=model_performance,
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            metadata={
                'target_column': target_column,
                'feature_count': len(self.feature_names),
                'sample_count': len(features),
                'model_fitted': self.is_fitted
            }
        )
        
        # Store result
        self.assessment_history.append(result)
        
        # Online learning update if enabled
        if self.config.enable_online_learning:
            self._update_online_learning(features, targets)
        
        self.logger.info(f'✅ Quality assessment completed')
        self.logger.info(f'   → Primary quality score: {quality_scores.get(QualityMetric.PREDICTIVE_POWER, 0):.3f}')
        
        return result
    
    def predict_optimal_parameters(self,
                                 market_conditions: pd.DataFrame,
                                 parameter_space: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Predict optimal labeling parameters based on current market conditions.
        
        Args:
            market_conditions: Current market state features
            parameter_space: Dictionary of parameter ranges to optimize
            
        Returns:
            Dictionary of optimal parameter values
        """
        self.logger.info('🎯 Predicting optimal labeling parameters')
        
        if not self.is_fitted:
            self.logger.warning('⚠️ Models not fitted, using default parameters')
            return {param: np.mean(values) for param, values in parameter_space.items()}
        
        # Extract features from market conditions
        features = self._extract_market_condition_features(market_conditions)
        
        # Use ensemble model to predict optimal parameters
        optimal_params = {}
        
        for param_name, param_range in parameter_space.items():
            # Create feature combinations for parameter prediction
            param_features = self._create_parameter_prediction_features(features, param_name)
            
            if MLModelType.ENSEMBLE in self.models:
                # Use ensemble model for parameter prediction
                param_prediction = self.models[MLModelType.ENSEMBLE].predict(param_features.reshape(1, -1))[0]
                # Map prediction to parameter range
                param_value = np.interp(param_prediction, [0, 1], [min(param_range), max(param_range)])
                optimal_params[param_name] = float(param_value)
            else:
                # Fallback to mean value
                optimal_params[param_name] = float(np.mean(param_range))
        
        self.logger.info(f'   → Predicted parameters: {optimal_params}')
        return optimal_params
    
    def enhance_label_quality(self,
                            labeled_data: pd.DataFrame,
                            market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance label quality using ML-based adjustments.
        
        Args:
            labeled_data: Original labeled data
            market_data: Market data for context
            
        Returns:
            Enhanced labeled data with ML-adjusted quality scores
        """
        self.logger.info('🔧 Enhancing label quality with ML adjustments')
        
        enhanced_data = labeled_data.copy()
        
        # Get probability columns to enhance
        prob_columns = [col for col in labeled_data.columns if col.endswith('_prob')]
        
        for col in prob_columns:
            if col in labeled_data.columns:
                # Assess quality for this column
                assessment = self.assess_label_quality(labeled_data, market_data, col)
                
                # Apply ML-based quality adjustments
                quality_multiplier = assessment.quality_scores.get(
                    QualityMetric.PREDICTIVE_POWER, 1.0
                )
                
                # Adjust probabilities based on quality assessment
                enhanced_data[col] = labeled_data[col] * quality_multiplier
                
                # Add ML-based quality score column
                enhanced_data[f'{col}_ml_quality'] = assessment.predictions
                
                # Add confidence intervals
                if assessment.confidence_intervals:
                    lower_ci, upper_ci = assessment.confidence_intervals
                    enhanced_data[f'{col}_confidence_lower'] = lower_ci
                    enhanced_data[f'{col}_confidence_upper'] = upper_ci
        
        self.logger.info(f'✅ Enhanced {len(prob_columns)} probability columns')
        return enhanced_data
    
    def _prepare_features_and_targets(self,
                                    labeled_data: pd.DataFrame,
                                    market_data: pd.DataFrame,
                                    target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for ML training."""
        # Combine labeled and market data
        combined_data = pd.concat([market_data, labeled_data], axis=1)
        
        # Engineer features
        features_df = self._engineer_features(combined_data)
        
        # Select target
        if target_column not in labeled_data.columns:
            raise ValueError(f"Target column {target_column} not found in labeled data")
        
        targets = labeled_data[target_column].values
        
        # Handle missing values
        features_df = features_df.fillna(method='ffill').fillna(0)
        targets = np.nan_to_num(targets)
        
        # Align lengths
        min_length = min(len(features_df), len(targets))
        features = features_df.iloc[:min_length].values
        targets = targets[:min_length]
        
        # Store feature names
        self.feature_names = list(features_df.columns)
        
        return features, targets
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML models."""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        if 'close' in data.columns:
            features['price_change'] = data['close'].pct_change()
            features['price_volatility'] = data['close'].rolling(20).std()
            features['price_momentum'] = data['close'].rolling(10).mean() / data['close'].rolling(20).mean()
        
        # Volume-based features
        if 'volume' in data.columns:
            features['volume_change'] = data['volume'].pct_change()
            features['volume_ma'] = data['volume'].rolling(20).mean()
            features['volume_std'] = data['volume'].rolling(20).std()
        
        # Technical indicators if enabled
        if self.config.include_technical_indicators:
            features = self._add_technical_indicators(features, data)
        
        # Market microstructure features if enabled
        if self.config.include_market_microstructure:
            features = self._add_microstructure_features(features, data)
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            if 'close' in data.columns:
                features[f'price_lag_{lag}'] = data['close'].shift(lag)
            if 'volume' in data.columns:
                features[f'volume_lag_{lag}'] = data['volume'].shift(lag)
        
        return features
    
    def _add_technical_indicators(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features."""
        if 'close' in data.columns:
            # Moving averages
            for period in [5, 10, 20, 50]:
                features[f'ma_{period}'] = data['close'].rolling(period).mean()
                features[f'ma_ratio_{period}'] = data['close'] / features[f'ma_{period}']
            
            # RSI approximation
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_ma = data['close'].rolling(bb_period).mean()
            bb_std_val = data['close'].rolling(bb_period).std()
            features['bb_upper'] = bb_ma + (bb_std_val * bb_std)
            features['bb_lower'] = bb_ma - (bb_std_val * bb_std)
            features['bb_position'] = (data['close'] - bb_lower) / (features['bb_upper'] - bb_lower)
        
        return features
    
    def _add_microstructure_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        if all(col in data.columns for col in ['high', 'low', 'close']):
            # True Range
            features['true_range'] = np.maximum(
                data['high'] - data['low'],
                np.maximum(
                    abs(data['high'] - data['close'].shift(1)),
                    abs(data['low'] - data['close'].shift(1))
                )
            )
            
            # Average True Range
            features['atr'] = features['true_range'].rolling(14).mean()
            
            # High-Low spread
            features['hl_spread'] = (data['high'] - data['low']) / data['close']
            
            # Price position within bar
            features['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        return features
    
    def _fit_models(self, features: np.ndarray, targets: np.ndarray):
        """Fit ML models for quality assessment."""
        self.logger.info('🏋️ Fitting ML models for quality assessment')
        
        # Prepare data
        X_scaled, y_scaled = self._preprocess_data(features, targets)
        
        # Split data
        split_idx = int(len(X_scaled) * self.config.train_test_split)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        
        # Fit individual models
        for model_type in self.config.ensemble_models:
            try:
                model = self._create_model(model_type)
                model.fit(X_train, y_train)
                self.models[model_type] = model
                
                # Evaluate model
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                self.logger.info(f'   → {model_type.value}: train={train_score:.3f}, test={test_score:.3f}')
                
            except Exception as e:
                self.logger.warning(f'Failed to fit {model_type.value}: {e}')
        
        # Create ensemble model
        if len(self.models) > 1:
            self.models[MLModelType.ENSEMBLE] = self._create_ensemble_model()
        
        self.is_fitted = True
        self.logger.info(f'✅ Fitted {len(self.models)} models successfully')
    
    def _create_model(self, model_type: MLModelType):
        """Create ML model based on type."""
        if model_type == MLModelType.RANDOM_FOREST:
            return RandomForestRegressor(**self.config.rf_params)
        
        elif model_type == MLModelType.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(**self.config.gb_params)
        
        elif model_type == MLModelType.XGBOOST and XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.config.random_state
            )
        
        elif model_type == MLModelType.NEURAL_NETWORK:
            return MLPRegressor(**self.config.nn_params)
        
        elif model_type == MLModelType.LINEAR_REGRESSION:
            return LinearRegression()
        
        elif model_type == MLModelType.RIDGE_REGRESSION:
            return Ridge(alpha=1.0, random_state=self.config.random_state)
        
        else:
            # Fallback to Random Forest
            return RandomForestRegressor(**self.config.rf_params)
    
    def _create_ensemble_model(self):
        """Create ensemble model from fitted individual models."""
        from sklearn.ensemble import VotingRegressor
        
        estimators = []
        for model_type, model in self.models.items():
            if model_type != MLModelType.ENSEMBLE:
                estimators.append((model_type.value, model))
        
        return VotingRegressor(estimators=estimators)
    
    def _preprocess_data(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess features and targets for ML training."""
        # Feature scaling
        if 'feature_scaler' not in self.scalers:
            self.scalers['feature_scaler'] = RobustScaler()
            X_scaled = self.scalers['feature_scaler'].fit_transform(features)
        else:
            X_scaled = self.scalers['feature_scaler'].transform(features)
        
        # Feature selection
        if 'feature_selector' not in self.feature_selectors:
            if self.config.feature_selection_method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_regression, k=min(self.config.max_features, X_scaled.shape[1]))
            else:
                selector = SelectKBest(score_func=f_regression, k=min(self.config.max_features, X_scaled.shape[1]))
            
            X_selected = selector.fit_transform(X_scaled, targets)
            self.feature_selectors['feature_selector'] = selector
        else:
            X_selected = self.feature_selectors['feature_selector'].transform(X_scaled)
        
        # Target scaling (optional)
        y_processed = targets.copy()
        
        return X_selected, y_processed
    
    def _generate_predictions(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions using the primary model."""
        if not self.is_fitted:
            return np.zeros(len(features))
        
        # Preprocess features
        X_scaled, _ = self._preprocess_data(features, np.zeros(len(features)))
        
        # Use ensemble model if available, otherwise primary model
        if MLModelType.ENSEMBLE in self.models:
            return self.models[MLModelType.ENSEMBLE].predict(X_scaled)
        elif self.config.primary_model in self.models:
            return self.models[self.config.primary_model].predict(X_scaled)
        else:
            # Use first available model
            first_model = list(self.models.values())[0]
            return first_model.predict(X_scaled)
    
    def _calculate_quality_scores(self,
                                predictions: np.ndarray,
                                targets: np.ndarray,
                                features: np.ndarray) -> Dict[QualityMetric, float]:
        """Calculate comprehensive quality scores."""
        quality_scores = {}
        
        # Predictive Power (R² score)
        if len(predictions) > 0 and len(targets) > 0:
            r2 = max(0.0, r2_score(targets, predictions))
            quality_scores[QualityMetric.PREDICTIVE_POWER] = r2
        
        # Information Content (correlation)
        if len(predictions) > 1 and len(targets) > 1:
            correlation = np.corrcoef(predictions, targets)[0, 1]
            if not np.isnan(correlation):
                quality_scores[QualityMetric.INFORMATION_CONTENT] = abs(correlation)
        
        # Stability Score
        stability = self._calculate_stability_score(predictions)
        quality_scores[QualityMetric.STABILITY_SCORE] = stability
        
        # Economic Value (Sharpe-like ratio)
        economic_value = self._calculate_economic_value(predictions, targets)
        quality_scores[QualityMetric.ECONOMIC_VALUE] = economic_value
        
        # Noise Robustness
        noise_robustness = self._calculate_noise_robustness(predictions, features)
        quality_scores[QualityMetric.NOISE_ROBUSTNESS] = noise_robustness
        
        return quality_scores
    
    def _calculate_stability_score(self, predictions: np.ndarray) -> float:
        """Calculate stability score of predictions."""
        if len(predictions) < self.config.stability_window:
            return 0.5
        
        # Calculate rolling standard deviation
        window_size = min(self.config.stability_window, len(predictions) // 4)
        rolling_std = pd.Series(predictions).rolling(window_size).std()
        
        # Stability is inverse of coefficient of variation
        mean_std = rolling_std.mean()
        mean_pred = np.mean(predictions)
        
        if mean_pred != 0 and not np.isnan(mean_std):
            cv = mean_std / abs(mean_pred)
            stability = 1.0 / (1.0 + cv)
        else:
            stability = 0.5
        
        return max(0.0, min(1.0, stability))
    
    def _calculate_economic_value(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate economic value score."""
        if len(predictions) < 20 or len(targets) < 20:
            return 0.0
        
        # Create simple trading signals based on predictions
        pred_signals = (predictions > np.percentile(predictions, 70)).astype(int)
        target_returns = np.diff(targets)
        
        if len(target_returns) == 0:
            return 0.0
        
        # Calculate strategy returns
        strategy_returns = pred_signals[1:] * target_returns
        
        if np.std(strategy_returns) > 0:
            sharpe_like = np.mean(strategy_returns) / np.std(strategy_returns)
            return max(0.0, min(1.0, (sharpe_like + 1.0) / 2.0))
        
        return 0.0
    
    def _calculate_noise_robustness(self, predictions: np.ndarray, features: np.ndarray) -> float:
        """Calculate noise robustness score."""
        if len(predictions) < 50:
            return 0.5
        
        # Add small amount of noise to features and test prediction stability
        noise_levels = [0.01, 0.05, 0.1]
        robustness_scores = []
        
        for noise_level in noise_levels:
            try:
                # Add Gaussian noise to features
                noisy_features = features + np.random.normal(0, noise_level, features.shape)
                
                # Get predictions with noisy features
                noisy_predictions = self._generate_predictions(noisy_features)
                
                # Calculate correlation between original and noisy predictions
                if len(noisy_predictions) == len(predictions):
                    correlation = np.corrcoef(predictions, noisy_predictions)[0, 1]
                    if not np.isnan(correlation):
                        robustness_scores.append(abs(correlation))
                        
            except Exception:
                continue
        
        return np.mean(robustness_scores) if robustness_scores else 0.5
    
    def _calculate_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance scores."""
        importance_dict = {}
        
        # Get feature importance from tree-based models
        for model_type, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Map to feature names (considering feature selection)
                if 'feature_selector' in self.feature_selectors:
                    selected_features = self.feature_selectors['feature_selector'].get_support()
                    selected_names = [name for i, name in enumerate(self.feature_names) if selected_features[i]]
                else:
                    selected_names = self.feature_names[:len(importances)]
                
                for i, importance in enumerate(importances):
                    if i < len(selected_names):
                        feature_name = selected_names[i]
                        if feature_name not in importance_dict:
                            importance_dict[feature_name] = 0.0
                        importance_dict[feature_name] += importance
        
        # Normalize importance scores
        if importance_dict:
            total_importance = sum(importance_dict.values())
            if total_importance > 0:
                importance_dict = {k: v / total_importance for k, v in importance_dict.items()}
        
        return importance_dict
    
    def _evaluate_model_performance(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance metrics."""
        performance = {}
        
        if len(predictions) > 0 and len(targets) > 0:
            performance['mse'] = mean_squared_error(targets, predictions)
            performance['mae'] = mean_absolute_error(targets, predictions)
            performance['r2'] = r2_score(targets, predictions)
            
            # Custom metrics
            performance['mean_prediction'] = float(np.mean(predictions))
            performance['std_prediction'] = float(np.std(predictions))
            performance['prediction_range'] = float(np.ptp(predictions))
        
        return performance
    
    def _calculate_confidence_intervals(self, features: np.ndarray, targets: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Calculate prediction confidence intervals."""
        try:
            if not self.is_fitted or len(features) < 20:
                return None
            
            # Use bootstrap for confidence intervals
            n_bootstrap = 50
            bootstrap_predictions = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(features), size=len(features), replace=True)
                bootstrap_features = features[indices]
                
                # Get predictions
                bootstrap_pred = self._generate_predictions(bootstrap_features)
                bootstrap_predictions.append(bootstrap_pred)
            
            # Calculate confidence intervals
            bootstrap_predictions = np.array(bootstrap_predictions)
            lower_ci = np.percentile(bootstrap_predictions, 2.5, axis=0)
            upper_ci = np.percentile(bootstrap_predictions, 97.5, axis=0)
            
            return (lower_ci, upper_ci)
            
        except Exception:
            return None
    
    def _extract_market_condition_features(self, market_conditions: pd.DataFrame) -> np.ndarray:
        """Extract features from current market conditions."""
        # Engineer features similar to training
        features_df = self._engineer_features(market_conditions)
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # Take last row as current conditions
        return features_df.iloc[-1:].values
    
    def _create_parameter_prediction_features(self, features: np.ndarray, param_name: str) -> np.ndarray:
        """Create features for parameter prediction."""
        # Combine market features with parameter-specific context
        param_features = features.flatten()
        
        # Add parameter-specific context (encoded)
        param_encoding = hash(param_name) % 100 / 100.0  # Simple encoding
        param_features = np.append(param_features, param_encoding)
        
        return param_features
    
    def _update_online_learning(self, features: np.ndarray, targets: np.ndarray):
        """Update models with new data for online learning."""
        self.online_update_counter += 1
        
        # Add to adaptation buffer
        self.adaptation_buffer.append((features[-1:], targets[-1:]))
        
        # Update models when buffer is full
        if (self.online_update_counter % self.config.update_frequency == 0 and 
            len(self.adaptation_buffer) >= self.config.update_frequency):
            
            try:
                # Combine buffer data
                buffer_features = np.vstack([f for f, _ in self.adaptation_buffer])
                buffer_targets = np.concatenate([t for _, t in self.adaptation_buffer])
                
                # Partial fit for models that support it
                for model_type, model in self.models.items():
                    if hasattr(model, 'partial_fit'):
                        # Preprocess new data
                        X_scaled, _ = self._preprocess_data(buffer_features, buffer_targets)
                        model.partial_fit(X_scaled, buffer_targets)
                
                # Clear buffer
                self.adaptation_buffer = []
                
                self.logger.info('📈 Updated models with online learning')
                
            except Exception as e:
                self.logger.warning(f'Online learning update failed: {e}')
    
    def _create_fallback_result(self, target_column: str) -> MLQualityAssessmentResult:
        """Create fallback result when ML assessment fails."""
        return MLQualityAssessmentResult(
            model_type=self.config.primary_model,
            quality_scores={
                QualityMetric.PREDICTIVE_POWER: 0.5,
                QualityMetric.INFORMATION_CONTENT: 0.5,
                QualityMetric.STABILITY_SCORE: 0.5,
                QualityMetric.ECONOMIC_VALUE: 0.5,
                QualityMetric.NOISE_ROBUSTNESS: 0.5
            },
            feature_importance={},
            model_performance={},
            predictions=np.array([]),
            confidence_intervals=None,
            metadata={
                'target_column': target_column,
                'error': 'insufficient_data',
                'fallback': True
            }
        )
    
    def save_models(self, output_dir: Union[str, Path]):
        """Save trained models to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        models_path = output_dir / 'ml_models.joblib'
        joblib.dump(self.models, models_path)
        
        # Save scalers and feature selectors
        preprocessing_path = output_dir / 'preprocessing.joblib'
        joblib.dump({
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors,
            'feature_names': self.feature_names
        }, preprocessing_path)
        
        # Save configuration
        config_path = output_dir / 'ml_config.joblib'
        joblib.dump(self.config, config_path)
        
        self.logger.info(f'💾 Models saved to {output_dir}')
    
    def load_models(self, input_dir: Union[str, Path]):
        """Load trained models from disk."""
        input_dir = Path(input_dir)
        
        # Load models
        models_path = input_dir / 'ml_models.joblib'
        if models_path.exists():
            self.models = joblib.load(models_path)
        
        # Load preprocessing
        preprocessing_path = input_dir / 'preprocessing.joblib'
        if preprocessing_path.exists():
            preprocessing_data = joblib.load(preprocessing_path)
            self.scalers = preprocessing_data['scalers']
            self.feature_selectors = preprocessing_data['feature_selectors']
            self.feature_names = preprocessing_data['feature_names']
        
        # Load configuration
        config_path = input_dir / 'ml_config.joblib'
        if config_path.exists():
            self.config = joblib.load(config_path)
        
        self.is_fitted = len(self.models) > 0
        self.logger.info(f'📂 Models loaded from {input_dir}')


# Convenience functions
def assess_label_quality_ml(labeled_data: pd.DataFrame,
                           market_data: pd.DataFrame,
                           target_column: str = 'overall_opportunity',
                           config: Optional[MLQualityAssessmentConfig] = None) -> MLQualityAssessmentResult:
    """Convenience function for ML-based label quality assessment."""
    assessor = MLLabelQualityAssessor(config)
    return assessor.assess_label_quality(labeled_data, market_data, target_column)


def enhance_labels_with_ml(labeled_data: pd.DataFrame,
                          market_data: pd.DataFrame,
                          config: Optional[MLQualityAssessmentConfig] = None) -> pd.DataFrame:
    """Convenience function to enhance labels with ML."""
    assessor = MLLabelQualityAssessor(config)
    return assessor.enhance_label_quality(labeled_data, market_data)