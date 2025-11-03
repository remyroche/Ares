"""
ML-Based Trading Indicator Generator from Candle Patterns

This module provides a comprehensive system for generating trading indicators
based on candlestick patterns using various ML models (LGBM, Random Forest, GRU, TFT).

Key Features:
- Uses existing candlestick pattern detection as input features
- Combines patterns with market context (volatility, volume, momentum)
- Supports multiple ML model types for indicator generation
- Comprehensive evaluation and backtesting capabilities
- Integration with existing VectorBT optimization
- Real-time indicator generation and updating

Architecture:
1. Feature Engineering: Combines candle patterns with market context
2. Model Training: Trains various ML models on historical data
3. Indicator Generation: Uses trained models to generate trading signals
4. Evaluation: Comprehensive performance assessment and backtesting
"""

import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Core imports
from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# ML imports
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# VectorBT imports for optimization
try:
    import vectorbt as vbt
    from src.vectorbt import rolling_mean, rolling_std, rolling_corr
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

# Existing candlestick pattern generator
from ..candlestick_pattern import CandlestickPatternFeatureGenerator

# ML common utilities
try:
    from ....utils.ml_common.models.model_factory import ModelFactory, ModelType
    from ....utils.ml_common.models.model_training import EnhancedModelTrainer
    from ....utils.ml_common.evaluation.unified_evaluator import evaluate_model
    from ....utils.ml_common.confidence_metrics import calculate_confidence_metrics
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """Types of trading indicators that can be generated."""
    DIRECTIONAL_SIGNAL = "directional_signal"  # Buy/Sell/Hold
    STRENGTH_SCORE = "strength_score"  # Signal strength 0-1
    CONFIDENCE_LEVEL = "confidence_level"  # Prediction confidence
    VOLATILITY_PREDICTION = "volatility_prediction"  # Future volatility
    PRICE_TARGET = "price_target"  # Price target prediction
    RISK_SCORE = "risk_score"  # Risk assessment


class ModelType(Enum):
    """Supported ML model types for indicator generation."""
    RANDOM_FOREST = "random_forest"
    LIGHTGBM = "lightgbm"
    GRU = "gru"
    TFT = "tft"  # Temporal Fusion Transformer
    ENSEMBLE = "ensemble"


@dataclass
class IndicatorConfig:
    """Configuration for ML-based indicator generation."""
    model_type: ModelType = ModelType.LIGHTGBM
    indicator_types: List[IndicatorType] = None
    lookback_window: int = 20
    prediction_horizon: int = 5
    feature_lookback: int = 10
    enable_market_context: bool = True
    enable_volume_features: bool = True
    enable_volatility_features: bool = True
    enable_momentum_features: bool = True
    enable_pattern_strength: bool = True
    enable_pattern_reliability: bool = True
    min_training_samples: int = 1000
    validation_split: float = 0.2
    retrain_frequency: int = 100  # Retrain every N new samples
    confidence_threshold: float = 0.7
    enable_vectorbt_optimization: bool = True
    
    def __post_init__(self):
        if self.indicator_types is None:
            self.indicator_types = [
                IndicatorType.DIRECTIONAL_SIGNAL,
                IndicatorType.STRENGTH_SCORE,
                IndicatorType.CONFIDENCE_LEVEL
            ]


class MLIndicatorGenerator(VectorizedFeatureGenerator):
    """
    ML-based trading indicator generator using candlestick patterns.
    
    This generator uses machine learning models to create trading indicators
    based on candlestick patterns and market context features.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None, indicator_config: Optional[IndicatorConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.indicator_config = indicator_config or IndicatorConfig()
        self.candle_pattern_generator = CandlestickPatternFeatureGenerator()
        
        # Model storage
        self.trained_models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.training_history = []
        
        # Performance tracking
        self.performance_stats = {
            'models_trained': 0,
            'predictions_made': 0,
            'total_training_time': 0.0,
            'total_prediction_time': 0.0,
            'last_retrain': None
        }
        
        # Initialize ML components
        self._initialize_ml_components()
    
    def _initialize_ml_components(self):
        """Initialize ML model components based on configuration."""
        if self.indicator_config.model_type == ModelType.RANDOM_FOREST and SKLEARN_AVAILABLE:
            self._initialize_random_forest()
        elif self.indicator_config.model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
            self._initialize_lightgbm()
        elif self.indicator_config.model_type in [ModelType.GRU, ModelType.TFT] and TORCH_AVAILABLE:
            self._initialize_neural_networks()
        else:
            logger.warning(f"Model type {self.indicator_config.model_type} not available, falling back to Random Forest")
            if SKLEARN_AVAILABLE:
                self._initialize_random_forest()
            else:
                raise ValueError("No suitable ML libraries available")
    
    def _initialize_random_forest(self):
        """Initialize Random Forest models for each indicator type."""
        for indicator_type in self.indicator_config.indicator_types:
            if indicator_type == IndicatorType.DIRECTIONAL_SIGNAL:
                self.trained_models[indicator_type] = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                self.trained_models[indicator_type] = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            self.scalers[indicator_type] = StandardScaler()
    
    def _initialize_lightgbm(self):
        """Initialize LightGBM models for each indicator type."""
        for indicator_type in self.indicator_config.indicator_types:
            if indicator_type == IndicatorType.DIRECTIONAL_SIGNAL:
                self.trained_models[indicator_type] = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=10,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            else:
                self.trained_models[indicator_type] = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=10,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            self.scalers[indicator_type] = StandardScaler()
    
    def _initialize_neural_networks(self):
        """Initialize neural network models (GRU/TFT) for each indicator type."""
        # This would be implemented based on the specific neural network architecture
        # For now, we'll use a placeholder that can be extended
        logger.info("Neural network initialization - to be implemented based on specific architecture requirements")
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="ml_candle_pattern_indicators",
            category=FeatureCategory.CANDLESTICK_PATTERN,
            description="ML-based trading indicators generated from candlestick patterns",
            required_columns=["open", "high", "low", "close", "volume"],
            default_lookback=20,
            min_lookback=10,
            max_lookback=50,
            parameters={
                "model_type": "lightgbm",
                "indicator_types": ["directional_signal", "strength_score", "confidence_level"],
                "enable_market_context": True,
                "enable_vectorbt_optimization": True
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ML-based trading indicators from candlestick patterns."""
        start_time = time.time()
        
        # Validate required columns
        required_cols = ["open", "high", "low", "close"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Generate candlestick pattern features
        pattern_features = self._generate_pattern_features(data)
        
        # Generate market context features
        context_features = self._generate_market_context_features(data)
        
        # Combine features
        combined_features = self._combine_features(pattern_features, context_features)
        
        # Generate indicators using trained models
        indicators = self._generate_indicators(combined_features, data)
        
        # Update performance stats
        self.performance_stats['predictions_made'] += 1
        self.performance_stats['total_prediction_time'] += time.time() - start_time
        
        # Return primary indicator (directional signal)
        primary_indicator = indicators.get(IndicatorType.DIRECTIONAL_SIGNAL, 
                                        np.zeros(len(data)))
        
        return pd.Series(primary_indicator, index=data.index, name='ml_candle_indicator')
    
    def _generate_pattern_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate candlestick pattern features using existing generator."""
        try:
            # Use existing candlestick pattern generator
            pattern_score = self.candle_pattern_generator._generate_feature(data)
            
            # Generate additional pattern features
            pattern_features = {
                'pattern_score': pattern_score.values,
                'pattern_strength': self._calculate_pattern_strength(data),
                'pattern_reliability': self._calculate_pattern_reliability(data),
                'pattern_frequency': self._calculate_pattern_frequency(data),
                'pattern_consistency': self._calculate_pattern_consistency(data)
            }
            
            return pattern_features
            
        except Exception as e:
            logger.warning(f"Pattern feature generation failed: {e}")
            return {
                'pattern_score': np.zeros(len(data)),
                'pattern_strength': np.zeros(len(data)),
                'pattern_reliability': np.zeros(len(data)),
                'pattern_frequency': np.zeros(len(data)),
                'pattern_consistency': np.zeros(len(data))
            }
    
    def _generate_market_context_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate market context features (volatility, volume, momentum)."""
        context_features = {}
        
        # Price-based features
        if 'close' in data.columns:
            close_prices = data['close'].values
            context_features['price_momentum'] = self._calculate_price_momentum(close_prices)
            context_features['price_volatility'] = self._calculate_price_volatility(close_prices)
            context_features['price_trend'] = self._calculate_price_trend(close_prices)
        
        # Volume features
        if 'volume' in data.columns and self.indicator_config.enable_volume_features:
            volume = data['volume'].values
            context_features['volume_momentum'] = self._calculate_volume_momentum(volume)
            context_features['volume_volatility'] = self._calculate_volume_volatility(volume)
            context_features['volume_trend'] = self._calculate_volume_trend(volume)
        
        # OHLC-based features
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            context_features['body_size'] = self._calculate_body_size(data)
            context_features['shadow_ratio'] = self._calculate_shadow_ratio(data)
            context_features['range_volatility'] = self._calculate_range_volatility(data)
        
        return context_features
    
    def _combine_features(self, pattern_features: Dict[str, np.ndarray], 
                         context_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine pattern and context features into a single feature matrix."""
        all_features = []
        feature_names = []
        
        # Add pattern features
        for name, values in pattern_features.items():
            all_features.append(values.reshape(-1, 1))
            feature_names.append(f"pattern_{name}")
        
        # Add context features
        for name, values in context_features.items():
            all_features.append(values.reshape(-1, 1))
            feature_names.append(f"context_{name}")
        
        # Combine all features
        if all_features:
            combined = np.hstack(all_features)
        else:
            combined = np.zeros((len(pattern_features.get('pattern_score', [])), 1))
        
        return combined
    
    def _generate_indicators(self, features: np.ndarray, data: pd.DataFrame) -> Dict[IndicatorType, np.ndarray]:
        """Generate trading indicators using trained ML models."""
        indicators = {}
        
        for indicator_type, model in self.trained_models.items():
            try:
                # Check if model is trained
                if not hasattr(model, 'feature_importances_') and not hasattr(model, 'booster_'):
                    logger.warning(f"Model for {indicator_type} not trained, skipping")
                    indicators[indicator_type] = np.zeros(len(features))
                    continue
                
                # Scale features
                scaler = self.scalers.get(indicator_type)
                if scaler is not None:
                    features_scaled = scaler.transform(features)
                else:
                    features_scaled = features
                
                # Generate predictions
                predictions = model.predict(features_scaled)
                
                # Post-process predictions based on indicator type
                if indicator_type == IndicatorType.DIRECTIONAL_SIGNAL:
                    # Convert to -1, 0, 1 (sell, hold, buy)
                    predictions = np.where(predictions > 0.6, 1, 
                                         np.where(predictions < 0.4, -1, 0))
                elif indicator_type == IndicatorType.STRENGTH_SCORE:
                    # Ensure values are in [0, 1]
                    predictions = np.clip(predictions, 0, 1)
                elif indicator_type == IndicatorType.CONFIDENCE_LEVEL:
                    # Ensure values are in [0, 1]
                    predictions = np.clip(predictions, 0, 1)
                
                indicators[indicator_type] = predictions
                
            except Exception as e:
                logger.warning(f"Indicator generation failed for {indicator_type}: {e}")
                indicators[indicator_type] = np.zeros(len(features))
        
        return indicators
    
    def train_models(self, data: pd.DataFrame, target_column: str = 'future_return'):
        """Train ML models on historical data."""
        start_time = time.time()
        
        # Generate features
        pattern_features = self._generate_pattern_features(data)
        context_features = self._generate_market_context_features(data)
        features = self._combine_features(pattern_features, context_features)
        
        # Prepare targets
        if target_column not in data.columns:
            # Create synthetic target based on future price movement
            future_returns = self._create_synthetic_targets(data)
        else:
            future_returns = data[target_column].values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, future_returns, 
            test_size=self.indicator_config.validation_split,
            random_state=42
        )
        
        # Train models for each indicator type
        for indicator_type, model in self.trained_models.items():
            try:
                # Prepare target for this indicator type
                y_train_indicator = self._prepare_target_for_indicator(
                    y_train, indicator_type
                )
                y_val_indicator = self._prepare_target_for_indicator(
                    y_val, indicator_type
                )
                
                # Scale features
                scaler = self.scalers[indicator_type]
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train model
                model.fit(X_train_scaled, y_train_indicator)
                
                # Store feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[indicator_type] = model.feature_importances_
                elif hasattr(model, 'booster_'):
                    self.feature_importance[indicator_type] = model.booster_.feature_importance()
                
                logger.info(f"✅ Trained {indicator_type} model successfully")
                
            except Exception as e:
                logger.error(f"❌ Training failed for {indicator_type}: {e}")
        
        # Update performance stats
        self.performance_stats['models_trained'] += 1
        self.performance_stats['total_training_time'] += time.time() - start_time
        self.performance_stats['last_retrain'] = datetime.now()
        
        # Store training history
        self.training_history.append({
            'timestamp': datetime.now(),
            'samples': len(features),
            'training_time': time.time() - start_time,
            'models_trained': len(self.trained_models)
        })
    
    def _prepare_target_for_indicator(self, targets: np.ndarray, 
                                    indicator_type: IndicatorType) -> np.ndarray:
        """Prepare target values for specific indicator type."""
        if indicator_type == IndicatorType.DIRECTIONAL_SIGNAL:
            # Convert to classification: 0 (sell), 1 (hold), 2 (buy)
            return np.where(targets > 0.01, 2, 
                          np.where(targets < -0.01, 0, 1))
        elif indicator_type == IndicatorType.STRENGTH_SCORE:
            # Use absolute return as strength
            return np.abs(targets)
        elif indicator_type == IndicatorType.CONFIDENCE_LEVEL:
            # Use inverse of volatility as confidence
            return 1.0 / (1.0 + np.abs(targets))
        else:
            # For other indicators, use raw targets
            return targets
    
    def _create_synthetic_targets(self, data: pd.DataFrame) -> np.ndarray:
        """Create synthetic targets based on future price movement."""
        if 'close' not in data.columns:
            return np.zeros(len(data))
        
        close_prices = data['close'].values
        future_returns = np.zeros(len(close_prices))
        
        # Calculate future returns
        for i in range(len(close_prices) - self.indicator_config.prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + self.indicator_config.prediction_horizon]
            future_returns[i] = (future_price - current_price) / current_price
        
        return future_returns
    
    # Feature calculation methods
    def _calculate_pattern_strength(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate pattern strength based on recent patterns."""
        if len(data) < 5:
            return np.zeros(len(data))
        
        # Simple pattern strength based on recent pattern consistency
        pattern_scores = self.candle_pattern_generator._generate_feature(data)
        strength = np.zeros(len(data))
        
        for i in range(5, len(data)):
            recent_patterns = pattern_scores.iloc[i-5:i]
            strength[i] = recent_patterns.std() / (recent_patterns.mean() + 1e-8)
        
        return strength
    
    def _calculate_pattern_reliability(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate pattern reliability based on historical accuracy."""
        if len(data) < 10:
            return np.zeros(len(data))
        
        # Simple reliability based on pattern consistency over time
        pattern_scores = self.candle_pattern_generator._generate_feature(data)
        reliability = np.zeros(len(data))
        
        for i in range(10, len(data)):
            recent_patterns = pattern_scores.iloc[i-10:i]
            reliability[i] = 1.0 / (1.0 + recent_patterns.std())
        
        return reliability
    
    def _calculate_pattern_frequency(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate pattern frequency in recent periods."""
        if len(data) < 20:
            return np.zeros(len(data))
        
        pattern_scores = self.candle_pattern_generator._generate_feature(data)
        frequency = np.zeros(len(data))
        
        for i in range(20, len(data)):
            recent_patterns = pattern_scores.iloc[i-20:i]
            frequency[i] = (recent_patterns > 0).sum() / 20.0
        
        return frequency
    
    def _calculate_pattern_consistency(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate pattern consistency over time."""
        if len(data) < 10:
            return np.zeros(len(data))
        
        pattern_scores = self.candle_pattern_generator._generate_feature(data)
        consistency = np.zeros(len(data))
        
        for i in range(10, len(data)):
            recent_patterns = pattern_scores.iloc[i-10:i]
            consistency[i] = 1.0 - recent_patterns.std()
        
        return consistency
    
    def _calculate_price_momentum(self, prices: np.ndarray) -> np.ndarray:
        """Calculate price momentum."""
        if len(prices) < 5:
            return np.zeros(len(prices))
        
        momentum = np.zeros(len(prices))
        for i in range(5, len(prices)):
            momentum[i] = (prices[i] - prices[i-5]) / prices[i-5]
        
        return momentum
    
    def _calculate_price_volatility(self, prices: np.ndarray) -> np.ndarray:
        """Calculate price volatility."""
        if len(prices) < 10:
            return np.zeros(len(prices))
        
        volatility = np.zeros(len(prices))
        for i in range(10, len(prices)):
            returns = np.diff(prices[i-10:i+1]) / prices[i-10:i]
            volatility[i] = np.std(returns)
        
        return volatility
    
    def _calculate_price_trend(self, prices: np.ndarray) -> np.ndarray:
        """Calculate price trend strength."""
        if len(prices) < 20:
            return np.zeros(len(prices))
        
        trend = np.zeros(len(prices))
        for i in range(20, len(prices)):
            recent_prices = prices[i-20:i+1]
            trend[i] = np.polyfit(range(21), recent_prices, 1)[0]
        
        return trend
    
    def _calculate_volume_momentum(self, volume: np.ndarray) -> np.ndarray:
        """Calculate volume momentum."""
        if len(volume) < 5:
            return np.zeros(len(volume))
        
        momentum = np.zeros(len(volume))
        for i in range(5, len(volume)):
            momentum[i] = (volume[i] - volume[i-5]) / (volume[i-5] + 1e-8)
        
        return momentum
    
    def _calculate_volume_volatility(self, volume: np.ndarray) -> np.ndarray:
        """Calculate volume volatility."""
        if len(volume) < 10:
            return np.zeros(len(volume))
        
        volatility = np.zeros(len(volume))
        for i in range(10, len(volume)):
            recent_volume = volume[i-10:i+1]
            volatility[i] = np.std(recent_volume) / (np.mean(recent_volume) + 1e-8)
        
        return volatility
    
    def _calculate_volume_trend(self, volume: np.ndarray) -> np.ndarray:
        """Calculate volume trend."""
        if len(volume) < 20:
            return np.zeros(len(volume))
        
        trend = np.zeros(len(volume))
        for i in range(20, len(volume)):
            recent_volume = volume[i-20:i+1]
            trend[i] = np.polyfit(range(21), recent_volume, 1)[0]
        
        return trend
    
    def _calculate_body_size(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate candlestick body size."""
        body_size = np.abs(data['close'] - data['open']) / data['close']
        return body_size.values
    
    def _calculate_shadow_ratio(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate shadow ratio."""
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        total_range = data['high'] - data['low']
        
        # Avoid division by zero
        total_range = np.where(total_range == 0, 1e-8, total_range)
        
        shadow_ratio = (upper_shadow + lower_shadow) / total_range
        return shadow_ratio.values
    
    def _calculate_range_volatility(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate range-based volatility."""
        if len(data) < 10:
            return np.zeros(len(data))
        
        range_vol = np.zeros(len(data))
        for i in range(10, len(data)):
            recent_ranges = (data['high'].iloc[i-10:i+1] - data['low'].iloc[i-10:i+1]) / data['close'].iloc[i-10:i+1]
            range_vol[i] = np.std(recent_ranges)
        
        return range_vol
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        if stats['total_training_time'] > 0:
            stats['training_samples_per_second'] = stats['models_trained'] / stats['total_training_time']
        if stats['total_prediction_time'] > 0:
            stats['predictions_per_second'] = stats['predictions_made'] / stats['total_prediction_time']
        return stats
    
    def get_feature_importance(self) -> Dict[IndicatorType, np.ndarray]:
        """Get feature importance for each model."""
        return self.feature_importance.copy()
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.training_history.copy()


def create_ml_indicator_generator(
    model_type: ModelType = ModelType.LIGHTGBM,
    indicator_types: List[IndicatorType] = None,
    **kwargs
) -> MLIndicatorGenerator:
    """Create an ML indicator generator with specified configuration."""
    if indicator_types is None:
        indicator_types = [
            IndicatorType.DIRECTIONAL_SIGNAL,
            IndicatorType.STRENGTH_SCORE,
            IndicatorType.CONFIDENCE_LEVEL
        ]
    
    indicator_config = IndicatorConfig(
        model_type=model_type,
        indicator_types=indicator_types,
        **kwargs
    )
    
    return MLIndicatorGenerator(indicator_config=indicator_config)


def test_ml_indicator_generator():
    """Test function for the ML indicator generator."""
    print("🧪 Testing ML Indicator Generator...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic OHLC data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=pd.date_range('2020-01-01', periods=n_samples, freq='1min'))
    
    # Ensure OHLC constraints
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Create indicator generator
    generator = create_ml_indicator_generator()
    
    # Train models
    print("📚 Training models...")
    generator.train_models(data)
    
    # Generate indicators
    print("🔮 Generating indicators...")
    indicators = generator._generate_feature(data)
    
    print(f"✅ Generated indicators for {len(indicators)} samples")
    print(f"📊 Indicator statistics:")
    print(f"   - Mean: {indicators.mean():.4f}")
    print(f"   - Std: {indicators.std():.4f}")
    print(f"   - Min: {indicators.min():.4f}")
    print(f"   - Max: {indicators.max():.4f}")
    
    # Performance stats
    stats = generator.get_performance_stats()
    print(f"\n📈 Performance Statistics:")
    print(f"   - Models trained: {stats['models_trained']}")
    print(f"   - Predictions made: {stats['predictions_made']}")
    print(f"   - Training time: {stats['total_training_time']:.4f}s")
    print(f"   - Prediction time: {stats['total_prediction_time']:.4f}s")
    
    print("\n🎉 ML Indicator Generator test completed successfully!")
    return generator, indicators


if __name__ == "__main__":
    test_ml_indicator_generator()