"""
Analyst Base Trainer - Unified Training Architecture

This module provides the base trainer class for all Analyst model training,
consolidating common functionality and providing a unified interface.

Key Features:
- Unified training interface for all Analyst model types
- Common training patterns and lifecycle management
- Standardized configuration and validation
- Performance monitoring and checkpointing
- Error handling and recovery mechanisms
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from .base_trainer import BaseTrainer, TrainingConfig, TrainingResult, ValidationResult, PredictionResult, TrainingRole, ModelType
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug, tprint_performance
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.hardware.integrated_hardware_manager import (
    get_integrated_hardware_manager, WorkloadType
)
from src.utils.hardware.optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked
)
from src.utils.hardware.memory_optimized_decorators import (
    memory_optimized, comprehensive_memory_optimization, MemoryOptimizationLevel
)


class AnalystModelType(Enum):
    """Types of Analyst models."""
    LIGHTGBM = "lightgbm"
    LIGHTGBM_PATCHTST = "lightgbm_patchtst"
    CATBOOST = "catboost"
    STACKER_LGBM_CALIBRATED = "stacker_lgbm_calibrated"


@dataclass
class AnalystTrainingConfig(TrainingConfig):
    """Analyst-specific training configuration."""
    # Analyst-specific parameters
    enable_patchtst_features: bool = True
    enable_regime_features: bool = True
    enable_multi_timeframe: bool = True
    
    # Model-specific parameters
    lightgbm_params: Dict[str, Any] = field(default_factory=dict)
    catboost_params: Dict[str, Any] = field(default_factory=dict)
    stacker_params: Dict[str, Any] = field(default_factory=dict)
    
    # Feature engineering parameters
    patchtst_window_size: int = 96
    patchtst_patch_length: int = 16
    regime_lookback: int = 20
    
    # Validation parameters
    analyst_validation_split: float = 0.2
    analyst_cv_folds: int = 5


class AnalystBaseTrainer(BaseTrainer):
    """
    Base trainer for all Analyst model training.
    
    This class provides a unified interface for training different types of Analyst models
    while maintaining consistent patterns for configuration, validation, and error handling.
    """
    
    def __init__(self, config: AnalystTrainingConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the Analyst base trainer.
        
        Args:
            config: Analyst training configuration
            logger: Logger instance (optional)
        """
        # Set role to ANALYST
        config.role = TrainingRole.ANALYST
        
        super().__init__(config, logger)
        
        # Analyst-specific state
        self._analyst_state = {
            'patchtst_features_created': False,
            'regime_features_created': False,
            'multi_timeframe_features_created': False,
            'feature_engineering_completed': False
        }
        
        tprint_info(f"🔧 Initialized AnalystBaseTrainer for {config.timeframe}")
        self.logger.info(f"Initialized AnalystBaseTrainer for {config.timeframe}")
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TrainingResult(
            success=False,
            error_message="Analyst training failed"
        ),
        context="analyst training"
    )
    async def train(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> TrainingResult:
        """
        Train Analyst models with given data.
        
        Args:
            data: Training data
            targets: Target variables (optional, can be inferred from data)
            
        Returns:
            Training result with model and metrics
        """
        try:
            tprint_info("📊 Starting Analyst model training...")
            self.logger.info("Starting Analyst model training...")
            
            start_time = time.time()
            
            # Preprocess data
            processed_data, processed_targets = self._preprocess_data(data, targets)
            
            # Create features
            feature_data = await self._create_analyst_features(processed_data)
            
            # Train models
            training_results = {}
            for model_type in self.config.model_types:
                tprint_info(f"🔧 Training {model_type.value} model...")
                
                model_result = await self._train_single_model(
                    model_type, feature_data, processed_targets
                )
                training_results[model_type.value] = model_result
                
                if model_result.success:
                    tprint_success(f"✅ {model_type.value} model trained successfully")
                else:
                    tprint_error(f"❌ {model_type.value} model training failed: {model_result.error_message}")
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(training_results)
            
            # Update training state
            self._training_state['training_completed'] = True
            self._training_state['training_started'] = True
            
            training_time = time.time() - start_time
            self._update_performance_metrics('training', training_time)
            
            # Create result
            result = TrainingResult(
                success=True,
                model=training_results,
                metrics=overall_metrics,
                training_time=training_time,
                metadata={
                    'analyst_state': self._analyst_state.copy(),
                    'models_trained': list(training_results.keys()),
                    'feature_engineering_completed': self._analyst_state['feature_engineering_completed']
                }
            )
            
            tprint_success(f"✅ Analyst training completed in {training_time:.2f}s")
            self.logger.info(f"Analyst training completed in {training_time:.2f}s")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Analyst training failed: {e}")
            self.logger.error(f"Analyst training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time=time.time() - start_time if 'start_time' in locals() else 0.0
            )
    
    async def validate(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> ValidationResult:
        """
        Validate the trained Analyst models.
        
        Args:
            data: Validation data
            targets: Target variables (optional)
            
        Returns:
            Validation result with metrics
        """
        try:
            tprint_info("📊 Validating Analyst models...")
            self.logger.info("Validating Analyst models...")
            
            start_time = time.time()
            
            # Preprocess data
            processed_data, processed_targets = self._preprocess_data(data, targets)
            
            # Create features
            feature_data = await self._create_analyst_features(processed_data)
            
            # Validate models
            validation_metrics = {}
            for model_type in self.config.model_types:
                model_key = f"{model_type.value}_model"
                if model_key in self._model_state:
                    model = self._model_state[model_key]
                    metrics = await self._validate_single_model(model, model_type, feature_data, processed_targets)
                    validation_metrics[model_type.value] = metrics
            
            validation_time = time.time() - start_time
            self._update_performance_metrics('validation', validation_time)
            
            result = ValidationResult(
                success=True,
                metrics=validation_metrics,
                metadata={
                    'validation_time': validation_time,
                    'models_validated': list(validation_metrics.keys())
                }
            )
            
            tprint_success(f"✅ Analyst validation completed in {validation_time:.2f}s")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Analyst validation failed: {e}")
            self.logger.error(f"Analyst validation failed: {e}")
            return ValidationResult(
                success=False,
                error_message=str(e)
            )
    
    async def predict(self, data: pd.DataFrame) -> PredictionResult:
        """
        Make predictions with the trained Analyst models.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Prediction result
        """
        try:
            tprint_info("📊 Making Analyst predictions...")
            self.logger.info("Making Analyst predictions...")
            
            start_time = time.time()
            
            # Preprocess data
            processed_data, _ = self._preprocess_data(data, None)
            
            # Create features
            feature_data = await self._create_analyst_features(processed_data)
            
            # Make predictions
            predictions = {}
            probabilities = {}
            
            for model_type in self.config.model_types:
                model_key = f"{model_type.value}_model"
                if model_key in self._model_state:
                    model = self._model_state[model_key]
                    pred, prob = await self._predict_single_model(model, model_type, feature_data)
                    predictions[model_type.value] = pred
                    if prob is not None:
                        probabilities[model_type.value] = prob
            
            prediction_time = time.time() - start_time
            self._update_performance_metrics('prediction', prediction_time)
            
            result = PredictionResult(
                success=True,
                predictions=predictions,
                probabilities=probabilities if probabilities else None,
                metadata={
                    'prediction_time': prediction_time,
                    'models_used': list(predictions.keys())
                }
            )
            
            tprint_success(f"✅ Analyst predictions completed in {prediction_time:.2f}s")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Analyst prediction failed: {e}")
            self.logger.error(f"Analyst prediction failed: {e}")
            return PredictionResult(
                success=False,
                error_message=str(e)
            )
    
    def _create_model(self, model_type: ModelType) -> Any:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Model instance
        """
        try:
            if model_type == ModelType.LIGHTGBM:
                return self._create_lightgbm_model()
            elif model_type == ModelType.CATBOOST:
                return self._create_catboost_model()
            elif model_type == ModelType.NEURAL_NETWORK:
                return self._create_neural_network_model()
            else:
                raise ValueError(f"Unsupported model type for Analyst: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to create {model_type.value} model: {e}")
            return None
    
    def _get_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from model.
        
        Args:
            model: Trained model
            
        Returns:
            Feature importance dictionary
        """
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(model.feature_names_in_, model.feature_importances_))
            elif hasattr(model, 'get_feature_importance'):
                return model.get_feature_importance()
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    async def _create_analyst_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create Analyst-specific features.
        
        Args:
            data: Input data
            
        Returns:
            Data with Analyst features
        """
        try:
            tprint_info("🔧 Creating Analyst features...")
            
            feature_data = data.copy()
            
            # Create PatchTST features if enabled
            if self.config.enable_patchtst_features:
                feature_data = await self._create_patchtst_features(feature_data)
                self._analyst_state['patchtst_features_created'] = True
            
            # Create regime features if enabled
            if self.config.enable_regime_features:
                feature_data = await self._create_regime_features(feature_data)
                self._analyst_state['regime_features_created'] = True
            
            # Create multi-timeframe features if enabled
            if self.config.enable_multi_timeframe:
                feature_data = await self._create_multi_timeframe_features(feature_data)
                self._analyst_state['multi_timeframe_features_created'] = True
            
            self._analyst_state['feature_engineering_completed'] = True
            tprint_success(f"✅ Created {feature_data.shape[1]} features")
            
            return feature_data
            
        except Exception as e:
            tprint_error(f"❌ Feature creation failed: {e}")
            self.logger.error(f"Feature creation failed: {e}")
            return data
    
    async def _create_patchtst_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create PatchTST (Patch Time Series Transformer) features for time series analysis.
        
        This implementation creates time series patches and statistical features
        that are commonly used in transformer-based time series models.
        """
        try:
            tprint_debug("🔧 Creating PatchTST features...")
            
            feature_data = data.copy()
            
            # Create time series patches for transformer models
            if 'close' in data.columns and len(data) > 10:
                # Calculate patch-based features
                patch_size = 16  # Standard patch size for time series
                
                # Create rolling patches
                for i in range(patch_size, len(data)):
                    patch = data['close'].iloc[i-patch_size:i]
                    
                    # Statistical features from patches
                    feature_data.loc[i, 'patch_mean'] = patch.mean()
                    feature_data.loc[i, 'patch_std'] = patch.std()
                    feature_data.loc[i, 'patch_min'] = patch.min()
                    feature_data.loc[i, 'patch_max'] = patch.max()
                    feature_data.loc[i, 'patch_range'] = patch.max() - patch.min()
                    
                    # Trend features
                    feature_data.loc[i, 'patch_trend'] = (patch.iloc[-1] - patch.iloc[0]) / patch.iloc[0] if patch.iloc[0] != 0 else 0
                    
                    # Volatility features
                    feature_data.loc[i, 'patch_volatility'] = patch.pct_change().std()
                    
                    # Momentum features
                    feature_data.loc[i, 'patch_momentum'] = patch.iloc[-1] / patch.mean() - 1 if patch.mean() != 0 else 0
                
                # Fill NaN values for the first patch_size rows
                patch_columns = ['patch_mean', 'patch_std', 'patch_min', 'patch_max', 
                               'patch_range', 'patch_trend', 'patch_volatility', 'patch_momentum']
                for col in patch_columns:
                    feature_data[col] = feature_data[col].fillna(feature_data[col].mean())
            
            # Create additional time series features
            if 'close' in data.columns:
                # Moving averages with different windows
                for window in [5, 10, 20, 50]:
                    feature_data[f'sma_{window}'] = data['close'].rolling(window=window).mean()
                    feature_data[f'ema_{window}'] = data['close'].ewm(span=window).mean()
                
                # Price position within recent range
                feature_data['price_position_20'] = (data['close'] - data['close'].rolling(20).min()) / (data['close'].rolling(20).max() - data['close'].rolling(20).min())
                feature_data['price_position_50'] = (data['close'] - data['close'].rolling(50).min()) / (data['close'].rolling(50).max() - data['close'].rolling(50).min())
                
                # Volatility features
                feature_data['volatility_5'] = data['close'].pct_change().rolling(5).std()
                feature_data['volatility_20'] = data['close'].pct_change().rolling(20).std()
                
                # Fill NaN values
                feature_data = feature_data.fillna(method='bfill').fillna(method='ffill')
            
            tprint_success(f"✅ PatchTST features created - added {len(feature_data.columns) - len(data.columns)} new features")
            return feature_data
            
        except Exception as e:
            tprint_error(f"❌ PatchTST feature creation failed: {e}")
            self.logger.error(f"PatchTST feature creation failed: {e}")
            return data
    
    async def _create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create regime detection and classification features.
        
        This implementation creates features that help identify different market regimes
        such as trending, ranging, high volatility, and low volatility periods.
        """
        try:
            tprint_debug("🔧 Creating regime features...")
            
            feature_data = data.copy()
            
            if 'close' in data.columns and len(data) > 20:
                # Volatility regime features
                feature_data['volatility_20'] = data['close'].pct_change().rolling(20).std()
                feature_data['volatility_regime'] = pd.cut(feature_data['volatility_20'], 
                                                         bins=3, labels=['low_vol', 'medium_vol', 'high_vol'])
                
                # Trend regime features
                feature_data['trend_20'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
                feature_data['trend_regime'] = pd.cut(feature_data['trend_20'], 
                                                    bins=3, labels=['downtrend', 'sideways', 'uptrend'])
                
                # Price range regime features
                feature_data['range_20'] = (data['close'].rolling(20).max() - data['close'].rolling(20).min()) / data['close'].rolling(20).mean()
                feature_data['range_regime'] = pd.cut(feature_data['range_20'], 
                                                    bins=3, labels=['narrow_range', 'medium_range', 'wide_range'])
                
                # Momentum regime features
                feature_data['momentum_5'] = data['close'].pct_change(5)
                feature_data['momentum_10'] = data['close'].pct_change(10)
                feature_data['momentum_regime'] = pd.cut(feature_data['momentum_10'], 
                                                       bins=3, labels=['negative_momentum', 'neutral_momentum', 'positive_momentum'])
                
                # Market structure features
                feature_data['higher_highs'] = (data['close'] > data['close'].rolling(20).max().shift(1)).astype(int)
                feature_data['lower_lows'] = (data['close'] < data['close'].rolling(20).min().shift(1)).astype(int)
                
                # Support and resistance levels
                feature_data['resistance_level'] = data['close'].rolling(20).max()
                feature_data['support_level'] = data['close'].rolling(20).min()
                feature_data['price_vs_resistance'] = data['close'] / feature_data['resistance_level']
                feature_data['price_vs_support'] = data['close'] / feature_data['support_level']
                
                # Regime change detection
                feature_data['volatility_change'] = feature_data['volatility_20'].pct_change()
                feature_data['trend_change'] = feature_data['trend_20'].diff()
                
                # Market state indicators
                feature_data['is_trending'] = (feature_data['trend_20'].abs() > feature_data['trend_20'].rolling(50).std()).astype(int)
                feature_data['is_volatile'] = (feature_data['volatility_20'] > feature_data['volatility_20'].rolling(50).mean()).astype(int)
                
                # Fill NaN values
                feature_data = feature_data.fillna(method='bfill').fillna(method='ffill')
            
            tprint_success(f"✅ Regime features created - added {len(feature_data.columns) - len(data.columns)} new features")
            return feature_data
            
        except Exception as e:
            tprint_error(f"❌ Regime feature creation failed: {e}")
            self.logger.error(f"Regime feature creation failed: {e}")
            return data
    
    async def _create_multi_timeframe_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create multi-timeframe analysis features.
        
        This implementation creates features that capture patterns across different
        time horizons, from short-term to long-term market dynamics.
        """
        try:
            tprint_debug("🔧 Creating multi-timeframe features...")
            
            feature_data = data.copy()
            
            if 'close' in data.columns and len(data) > 50:
                # Short-term features (1-5 periods)
                feature_data['st_return_1'] = data['close'].pct_change(1)
                feature_data['st_return_3'] = data['close'].pct_change(3)
                feature_data['st_return_5'] = data['close'].pct_change(5)
                
                # Medium-term features (10-20 periods)
                feature_data['mt_return_10'] = data['close'].pct_change(10)
                feature_data['mt_return_15'] = data['close'].pct_change(15)
                feature_data['mt_return_20'] = data['close'].pct_change(20)
                
                # Long-term features (30-50 periods)
                feature_data['lt_return_30'] = data['close'].pct_change(30)
                feature_data['lt_return_40'] = data['close'].pct_change(40)
                feature_data['lt_return_50'] = data['close'].pct_change(50)
                
                # Multi-timeframe moving averages
                for period in [5, 10, 20, 30, 50]:
                    feature_data[f'sma_{period}'] = data['close'].rolling(period).mean()
                    feature_data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
                
                # Multi-timeframe volatility
                for period in [5, 10, 20, 30]:
                    feature_data[f'volatility_{period}'] = data['close'].pct_change().rolling(period).std()
                
                # Timeframe relationships
                feature_data['st_vs_mt_trend'] = feature_data['st_return_5'] - feature_data['mt_return_20']
                feature_data['mt_vs_lt_trend'] = feature_data['mt_return_20'] - feature_data['lt_return_50']
                feature_data['st_vs_lt_trend'] = feature_data['st_return_5'] - feature_data['lt_return_50']
                
                # Multi-timeframe momentum
                feature_data['momentum_ratio_5_20'] = feature_data['st_return_5'] / (feature_data['mt_return_20'] + 1e-8)
                feature_data['momentum_ratio_10_30'] = feature_data['mt_return_10'] / (feature_data['lt_return_30'] + 1e-8)
                
                # Multi-timeframe volatility ratios
                feature_data['vol_ratio_5_20'] = feature_data['volatility_5'] / (feature_data['volatility_20'] + 1e-8)
                feature_data['vol_ratio_10_30'] = feature_data['volatility_10'] / (feature_data['volatility_30'] + 1e-8)
                
                # Cross-timeframe price position
                feature_data['price_position_5'] = (data['close'] - data['close'].rolling(5).min()) / (data['close'].rolling(5).max() - data['close'].rolling(5).min())
                feature_data['price_position_20'] = (data['close'] - data['close'].rolling(20).min()) / (data['close'].rolling(20).max() - data['close'].rolling(20).min())
                feature_data['price_position_50'] = (data['close'] - data['close'].rolling(50).min()) / (data['close'].rolling(50).max() - data['close'].rolling(50).min())
                
                # Timeframe divergence detection
                feature_data['price_momentum_divergence'] = (feature_data['st_return_5'] > 0).astype(int) != (feature_data['mt_return_20'] > 0).astype(int)
                feature_data['volatility_divergence'] = (feature_data['volatility_5'] > feature_data['volatility_20']).astype(int) != (feature_data['volatility_10'] > feature_data['volatility_30']).astype(int)
                
                # Multi-timeframe trend strength
                feature_data['trend_strength_5'] = feature_data['st_return_5'].abs()
                feature_data['trend_strength_20'] = feature_data['mt_return_20'].abs()
                feature_data['trend_strength_50'] = feature_data['lt_return_50'].abs()
                
                # Timeframe consistency
                feature_data['trend_consistency'] = ((feature_data['st_return_5'] > 0) == (feature_data['mt_return_20'] > 0) == (feature_data['lt_return_50'] > 0)).astype(int)
                
                # Fill NaN values
                feature_data = feature_data.fillna(method='bfill').fillna(method='ffill')
            
            tprint_success(f"✅ Multi-timeframe features created - added {len(feature_data.columns) - len(data.columns)} new features")
            return feature_data
            
        except Exception as e:
            tprint_error(f"❌ Multi-timeframe feature creation failed: {e}")
            self.logger.error(f"Multi-timeframe feature creation failed: {e}")
            return data
    
    def _create_lightgbm_model(self):
        """Create LightGBM model."""
        try:
            import lightgbm as lgb
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                **self.config.lightgbm_params
            }
            
            return lgb.LGBMClassifier(**params)
            
        except ImportError:
            self.logger.error("LightGBM not available")
            return None
    
    def _create_catboost_model(self):
        """Create CatBoost model."""
        try:
            import catboost as cb
            
            params = {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'verbose': False,
                **self.config.catboost_params
            }
            
            return cb.CatBoostClassifier(**params)
            
        except ImportError:
            self.logger.error("CatBoost not available")
            return None
    
    def _create_neural_network_model(self):
        """Create neural network model for Analyst predictions.
        
        This implementation creates a simple feedforward neural network
        using scikit-learn's MLPRegressor for regression tasks.
        """
        try:
            tprint_debug("🔧 Creating neural network model...")
            
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Create a simple neural network model
            model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),  # Three hidden layers
                activation='relu',
                solver='adam',
                alpha=0.001,  # L2 regularization
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=1000,
                shuffle=True,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-4
            )
            
            # Create a scaler for input normalization
            scaler = StandardScaler()
            
            # Wrap model and scaler in a custom class
            class NeuralNetworkModel:
                def __init__(self, model, scaler):
                    self.model = model
                    self.scaler = scaler
                    self.is_fitted = False
                    self.feature_names_ = None
                
                def fit(self, X, y):
                    """Fit the neural network model."""
                    try:
                        # Store feature names if available
                        if hasattr(X, 'columns'):
                            self.feature_names_ = list(X.columns)
                        
                        # Scale the features
                        X_scaled = self.scaler.fit_transform(X)
                        
                        # Fit the model
                        self.model.fit(X_scaled, y)
                        self.is_fitted = True
                        
                        return self
                    except Exception as e:
                        tprint_error(f"❌ Neural network fitting failed: {e}")
                        raise
                
                def predict(self, X):
                    """Make predictions."""
                    if not self.is_fitted:
                        raise ValueError("Model must be fitted before prediction")
                    
                    try:
                        # Scale the features
                        X_scaled = self.scaler.transform(X)
                        
                        # Make predictions
                        predictions = self.model.predict(X_scaled)
                        return predictions
                    except Exception as e:
                        tprint_error(f"❌ Neural network prediction failed: {e}")
                        raise
                
                def get_params(self, deep=True):
                    """Get model parameters."""
                    return {
                        'hidden_layer_sizes': self.model.hidden_layer_sizes,
                        'activation': self.model.activation,
                        'solver': self.model.solver,
                        'alpha': self.model.alpha,
                        'is_fitted': self.is_fitted
                    }
                
                def set_params(self, **params):
                    """Set model parameters."""
                    for key, value in params.items():
                        if hasattr(self.model, key):
                            setattr(self.model, key, value)
                    return self
                
                def get_feature_importance(self):
                    """Get feature importance (not directly available for MLP)."""
                    if self.is_fitted and hasattr(self.model, 'coefs_'):
                        # Approximate feature importance using first layer weights
                        if len(self.model.coefs_) > 0:
                            importance = np.abs(self.model.coefs_[0]).mean(axis=1)
                            if self.feature_names_:
                                return dict(zip(self.feature_names_, importance))
                            else:
                                return {f'feature_{i}': imp for i, imp in enumerate(importance)}
                    return None
            
            neural_model = NeuralNetworkModel(model, scaler)
            
            tprint_success("✅ Neural network model created successfully")
            return neural_model
            
        except ImportError as e:
            tprint_warning(f"⚠️ scikit-learn not available for neural network: {e}")
            return None
        except Exception as e:
            tprint_error(f"❌ Failed to create neural network model: {e}")
            self.logger.error(f"Failed to create neural network model: {e}")
            return None
    
    async def _train_single_model(self, model_type: ModelType, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train a single model."""
        try:
            model = self._create_model(model_type)
            if model is None:
                return TrainingResult(
                    success=False,
                    error_message=f"Failed to create {model_type.value} model"
                )
            
            # Train the model
            model.fit(data, targets)
            
            # Store model
            model_key = f"{model_type.value}_model"
            self._model_state[model_key] = model
            
            # Get feature importance
            feature_importance = self._get_feature_importance(model)
            
            return TrainingResult(
                success=True,
                model=model,
                feature_importance=feature_importance,
                metadata={'model_type': model_type.value}
            )
            
        except Exception as e:
            return TrainingResult(
                success=False,
                error_message=str(e)
            )
    
    async def _validate_single_model(self, model: Any, model_type: ModelType, data: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """Validate a single model."""
        try:
            predictions = model.predict(data)
            probabilities = model.predict_proba(data)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'accuracy': accuracy_score(targets, predictions),
                'precision': precision_score(targets, predictions, average='weighted'),
                'recall': recall_score(targets, predictions, average='weighted'),
                'f1': f1_score(targets, predictions, average='weighted')
            }
            
            if probabilities is not None:
                metrics['roc_auc'] = roc_auc_score(targets, probabilities)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Validation failed for {model_type.value}: {e}")
            return {}
    
    async def _predict_single_model(self, model: Any, model_type: ModelType, data: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with a single model."""
        try:
            predictions = model.predict(data)
            probabilities = model.predict_proba(data)[:, 1] if hasattr(model, 'predict_proba') else None
            
            return predictions, probabilities
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {model_type.value}: {e}")
            return np.array([]), None
    
    def _calculate_overall_metrics(self, training_results: Dict[str, TrainingResult]) -> Dict[str, float]:
        """Calculate overall training metrics."""
        successful_models = [r for r in training_results.values() if r.success]
        
        return {
            'total_models': len(training_results),
            'successful_models': len(successful_models),
            'success_rate': len(successful_models) / len(training_results) if training_results else 0.0
        }
    
    def get_analyst_summary(self) -> Dict[str, Any]:
        """Get Analyst-specific training summary."""
        base_summary = self.get_training_summary()
        base_summary.update({
            'analyst_state': self._analyst_state.copy(),
            'feature_engineering': {
                'patchtst_features': self._analyst_state['patchtst_features_created'],
                'regime_features': self._analyst_state['regime_features_created'],
                'multi_timeframe_features': self._analyst_state['multi_timeframe_features_created'],
                'completed': self._analyst_state['feature_engineering_completed']
            }
        })
        return base_summary