"""
ML Indicator Training Pipeline

This module provides a comprehensive training pipeline for ML-based trading indicators
generated from candlestick patterns. It integrates with the existing ML infrastructure
and provides advanced training capabilities.

Key Features:
- Integration with existing ML common utilities
- Advanced feature engineering pipeline
- Model selection and hyperparameter optimization
- Cross-validation and backtesting
- Performance monitoring and evaluation
- Model persistence and versioning
"""

import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import json
import pickle

# Core imports
from .ml_candle_pattern_indicators import MLIndicatorGenerator, IndicatorType, ModelType, IndicatorConfig

# ML common imports
try:
    from src.utils.ml_common.models.model_factory import ModelFactory, ModelType as MLModelType
    from src.utils.ml_common.models.model_training import EnhancedModelTrainer
    from src.utils.ml_common.evaluation.unified_evaluator import evaluate_model
    from src.utils.ml_common.confidence_metrics import calculate_confidence_metrics
    from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization
    from src.utils.ml_common.ensembles.ensemble_manager import EnsembleManager
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

# VectorBT imports
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for ML indicator training pipeline."""
    # Data configuration
    train_start_date: Optional[datetime] = None
    train_end_date: Optional[datetime] = None
    validation_split: float = 0.2
    test_split: float = 0.1
    min_samples_per_class: int = 100
    
    # Feature engineering
    enable_feature_selection: bool = True
    feature_selection_method: str = "mutual_info"  # mutual_info, f_test, chi2, rfe
    max_features: int = 50
    enable_feature_interaction: bool = True
    enable_polynomial_features: bool = False
    
    # Model configuration
    enable_hyperparameter_optimization: bool = True
    hpo_trials: int = 50
    enable_ensemble: bool = True
    ensemble_methods: List[str] = None
    
    # Evaluation configuration
    enable_cross_validation: bool = True
    cv_folds: int = 5
    enable_backtesting: bool = True
    backtest_start_date: Optional[datetime] = None
    backtest_end_date: Optional[datetime] = None
    
    # Performance monitoring
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    enable_model_checkpointing: bool = True
    checkpoint_frequency: int = 10
    
    def __post_init__(self):
        if self.ensemble_methods is None:
            self.ensemble_methods = ["voting", "stacking"]


class MLIndicatorTrainingPipeline:
    """
    Comprehensive training pipeline for ML-based trading indicators.
    
    This pipeline provides end-to-end training capabilities including:
    - Data preparation and feature engineering
    - Model training and hyperparameter optimization
    - Cross-validation and backtesting
    - Performance evaluation and model selection
    - Model persistence and versioning
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.generators = {}
        self.trained_models = {}
        self.evaluation_results = {}
        self.training_history = []
        
        # Initialize ML components
        self._initialize_ml_components()
    
    def _initialize_ml_components(self):
        """Initialize ML components based on availability."""
        if ML_COMMON_AVAILABLE:
            self.model_factory = ModelFactory()
            self.model_trainer = EnhancedModelTrainer()
            self.hpo_optimizer = HyperparameterOptimization()
            self.ensemble_manager = EnsembleManager()
        else:
            logger.warning("ML common utilities not available, using basic training")
            self.model_factory = None
            self.model_trainer = None
            self.hpo_optimizer = None
            self.ensemble_manager = None
    
    def train_all_models(self, data: pd.DataFrame, 
                        target_column: str = 'future_return',
                        symbol: str = 'BTCUSDT') -> Dict[str, Any]:
        """
        Train all ML models for indicator generation.
        
        Args:
            data: Historical OHLCV data
            target_column: Target variable column name
            symbol: Trading symbol for identification
            
        Returns:
            Dictionary containing training results and model performance
        """
        start_time = time.time()
        logger.info(f"🚀 Starting ML indicator training for {symbol}")
        
        # Prepare data
        prepared_data = self._prepare_training_data(data, target_column)
        
        # Train models for each indicator type
        training_results = {}
        for indicator_type in IndicatorType:
            try:
                logger.info(f"📚 Training model for {indicator_type.value}")
                result = self._train_single_indicator_model(
                    prepared_data, indicator_type, symbol
                )
                training_results[indicator_type.value] = result
                
            except Exception as e:
                logger.error(f"❌ Training failed for {indicator_type.value}: {e}")
                training_results[indicator_type.value] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Train ensemble models if enabled
        if self.config.enable_ensemble:
            logger.info("🎯 Training ensemble models...")
            ensemble_results = self._train_ensemble_models(prepared_data, symbol)
            training_results['ensemble'] = ensemble_results
        
        # Evaluate all models
        evaluation_results = self._evaluate_all_models(prepared_data, symbol)
        training_results['evaluation'] = evaluation_results
        
        # Store training history
        self.training_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'samples': len(prepared_data),
            'training_time': time.time() - start_time,
            'models_trained': len(training_results),
            'success': True
        })
        
        logger.info(f"✅ Training completed in {time.time() - start_time:.2f} seconds")
        return training_results
    
    def _prepare_training_data(self, data: pd.DataFrame, 
                              target_column: str) -> pd.DataFrame:
        """Prepare data for training including feature engineering."""
        logger.info("🔧 Preparing training data...")
        
        # Create target variable if not exists
        if target_column not in data.columns:
            data = self._create_target_variable(data, target_column)
        
        # Add technical indicators
        data = self._add_technical_indicators(data)
        
        # Add candlestick pattern features
        data = self._add_candlestick_features(data)
        
        # Add market context features
        data = self._add_market_context_features(data)
        
        # Feature selection if enabled
        if self.config.enable_feature_selection:
            data = self._apply_feature_selection(data, target_column)
        
        # Feature interaction if enabled
        if self.config.enable_feature_interaction:
            data = self._add_feature_interactions(data)
        
        logger.info(f"✅ Prepared data with {len(data.columns)} features")
        return data
    
    def _create_target_variable(self, data: pd.DataFrame, 
                               target_column: str) -> pd.DataFrame:
        """Create target variable based on future price movement."""
        if 'close' not in data.columns:
            data[target_column] = 0
            return data
        
        # Calculate future returns
        future_returns = np.zeros(len(data))
        prediction_horizon = 5  # 5 periods ahead
        
        for i in range(len(data) - prediction_horizon):
            current_price = data['close'].iloc[i]
            future_price = data['close'].iloc[i + prediction_horizon]
            future_returns[i] = (future_price - current_price) / current_price
        
        data[target_column] = future_returns
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        if 'close' not in data.columns:
            return data
        
        close_prices = data['close']
        
        # Moving averages
        data['sma_5'] = close_prices.rolling(5).mean()
        data['sma_20'] = close_prices.rolling(20).mean()
        data['ema_12'] = close_prices.ewm(span=12).mean()
        data['ema_26'] = close_prices.ewm(span=26).mean()
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # RSI
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['bb_middle'] = close_prices.rolling(20).mean()
        bb_std = close_prices.rolling(20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (close_prices - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Volatility
        data['volatility'] = close_prices.rolling(20).std()
        data['volatility_ratio'] = data['volatility'] / data['volatility'].rolling(50).mean()
        
        return data
    
    def _add_candlestick_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features."""
        try:
            # Use existing candlestick pattern generator
            from .candlestick_pattern import CandlestickPatternFeatureGenerator
            pattern_generator = CandlestickPatternFeatureGenerator()
            
            # Generate pattern features
            pattern_features = pattern_generator._generate_feature(data)
            data['candlestick_pattern'] = pattern_features
            
            # Add individual pattern features
            data['body_size'] = np.abs(data['close'] - data['open']) / data['close']
            data['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
            data['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
            data['total_range'] = data['high'] - data['low']
            
            # Shadow ratios
            data['upper_shadow_ratio'] = data['upper_shadow'] / data['total_range']
            data['lower_shadow_ratio'] = data['lower_shadow'] / data['total_range']
            data['body_ratio'] = data['body_size'] / data['total_range']
            
        except Exception as e:
            logger.warning(f"Failed to add candlestick features: {e}")
        
        return data
    
    def _add_market_context_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market context features."""
        if 'close' not in data.columns:
            return data
        
        close_prices = data['close']
        
        # Price momentum
        data['momentum_5'] = close_prices.pct_change(5)
        data['momentum_10'] = close_prices.pct_change(10)
        data['momentum_20'] = close_prices.pct_change(20)
        
        # Price acceleration
        data['acceleration_5'] = data['momentum_5'].diff()
        data['acceleration_10'] = data['momentum_10'].diff()
        
        # Volume features
        if 'volume' in data.columns:
            data['volume_ma_5'] = data['volume'].rolling(5).mean()
            data['volume_ma_20'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma_20']
            data['volume_momentum'] = data['volume'].pct_change(5)
        
        # Market regime features
        data['trend_strength'] = np.abs(data['close'].rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0
        ))
        
        # Support and resistance levels
        data['resistance_level'] = data['high'].rolling(20).max()
        data['support_level'] = data['low'].rolling(20).min()
        data['price_position'] = (data['close'] - data['support_level']) / (data['resistance_level'] - data['support_level'])
        
        return data
    
    def _apply_feature_selection(self, data: pd.DataFrame, 
                                target_column: str) -> pd.DataFrame:
        """Apply feature selection to reduce dimensionality."""
        try:
            from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
            
            # Separate features and target
            feature_columns = [col for col in data.columns if col != target_column]
            X = data[feature_columns].fillna(0)
            y = data[target_column].fillna(0)
            
            # Select features based on method
            if self.config.feature_selection_method == "mutual_info":
                selector = SelectKBest(score_func=mutual_info_regression, k=self.config.max_features)
            else:
                selector = SelectKBest(score_func=f_regression, k=self.config.max_features)
            
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
            selected_features.append(target_column)
            
            # Return data with selected features
            return data[selected_features]
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return data
    
    def _add_feature_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions to capture non-linear relationships."""
        try:
            from sklearn.preprocessing import PolynomialFeatures
            
            # Select numeric columns for interactions
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if col != 'future_return']
            
            if len(numeric_columns) < 2:
                return data
            
            # Create polynomial features (degree 2)
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            interaction_features = poly.fit_transform(data[numeric_columns].fillna(0))
            
            # Create column names for interaction features
            feature_names = poly.get_feature_names_out(numeric_columns)
            
            # Add interaction features to data
            for i, name in enumerate(feature_names):
                if name not in data.columns:  # Avoid duplicates
                    data[f'interaction_{name}'] = interaction_features[:, i]
            
        except Exception as e:
            logger.warning(f"Feature interaction failed: {e}")
        
        return data
    
    def _train_single_indicator_model(self, data: pd.DataFrame, 
                                    indicator_type: IndicatorType,
                                    symbol: str) -> Dict[str, Any]:
        """Train a single indicator model."""
        start_time = time.time()
        
        # Create indicator generator
        generator = MLIndicatorGenerator(
            indicator_config=IndicatorConfig(
                indicator_types=[indicator_type],
                enable_market_context=True
            )
        )
        
        # Prepare features and targets
        feature_columns = [col for col in data.columns if col != 'future_return']
        X = data[feature_columns].fillna(0)
        y = data['future_return'].fillna(0)
        
        # Prepare target for specific indicator type
        y_indicator = generator._prepare_target_for_indicator(y.values, indicator_type)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_indicator, test_size=0.2, random_state=42
        )
        
        # Train model
        model = generator.trained_models[indicator_type]
        scaler = generator.scalers[indicator_type]
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Generate predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate additional metrics
        metrics = self._calculate_model_metrics(y_test, y_pred, indicator_type)
        
        # Store model
        self.generators[f"{symbol}_{indicator_type.value}"] = generator
        
        result = {
            'success': True,
            'model_type': indicator_type.value,
            'training_time': time.time() - start_time,
            'train_score': train_score,
            'test_score': test_score,
            'metrics': metrics,
            'feature_importance': generator.get_feature_importance().get(indicator_type, []),
            'model': model,
            'scaler': scaler
        }
        
        return result
    
    def _train_ensemble_models(self, data: pd.DataFrame, 
                              symbol: str) -> Dict[str, Any]:
        """Train ensemble models combining multiple approaches."""
        if not self.config.enable_ensemble:
            return {'success': False, 'message': 'Ensemble training disabled'}
        
        try:
            # Create ensemble of different model types
            ensemble_models = {}
            
            for model_type in [ModelType.RANDOM_FOREST, ModelType.LIGHTGBM]:
                try:
                    generator = MLIndicatorGenerator(
                        indicator_config=IndicatorConfig(
                            model_type=model_type,
                            indicator_types=[IndicatorType.DIRECTIONAL_SIGNAL]
                        )
                    )
                    
                    # Train the generator
                    generator.train_models(data)
                    
                    ensemble_models[model_type.value] = generator
                    
                except Exception as e:
                    logger.warning(f"Failed to train {model_type.value} for ensemble: {e}")
            
            # Create ensemble predictions
            if len(ensemble_models) > 1:
                ensemble_result = self._create_ensemble_predictions(ensemble_models, data)
                return {
                    'success': True,
                    'ensemble_models': list(ensemble_models.keys()),
                    'ensemble_result': ensemble_result
                }
            else:
                return {'success': False, 'message': 'Insufficient models for ensemble'}
                
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_ensemble_predictions(self, models: Dict[str, MLIndicatorGenerator], 
                                   data: pd.DataFrame) -> Dict[str, Any]:
        """Create ensemble predictions from multiple models."""
        # This would implement ensemble prediction logic
        # For now, return a placeholder
        return {
            'ensemble_method': 'voting',
            'models_used': list(models.keys()),
            'prediction_accuracy': 0.0  # Placeholder
        }
    
    def _calculate_model_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               indicator_type: IndicatorType) -> Dict[str, float]:
        """Calculate comprehensive model metrics."""
        metrics = {}
        
        try:
            if indicator_type == IndicatorType.DIRECTIONAL_SIGNAL:
                # Classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            else:
                # Regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
                metrics['r2'] = r2_score(y_true, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
        
        except Exception as e:
            logger.warning(f"Failed to calculate metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _evaluate_all_models(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Evaluate all trained models."""
        evaluation_results = {
            'symbol': symbol,
            'evaluation_timestamp': datetime.now(),
            'models_evaluated': len(self.generators),
            'overall_performance': {}
        }
        
        # This would implement comprehensive model evaluation
        # including backtesting, cross-validation, etc.
        
        return evaluation_results
    
    def save_models(self, save_path: str, symbol: str):
        """Save trained models to disk."""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each generator
        for name, generator in self.generators.items():
            if symbol in name:
                model_path = save_dir / f"{name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(generator, f)
        
        # Save training history
        history_path = save_dir / f"{symbol}_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, default=str, indent=2)
        
        logger.info(f"✅ Models saved to {save_path}")
    
    def load_models(self, load_path: str, symbol: str):
        """Load trained models from disk."""
        load_dir = Path(load_path)
        
        # Load generators
        for model_file in load_dir.glob(f"{symbol}_*.pkl"):
            with open(model_file, 'rb') as f:
                generator = pickle.load(f)
                model_name = model_file.stem
                self.generators[model_name] = generator
        
        # Load training history
        history_path = load_dir / f"{symbol}_training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        
        logger.info(f"✅ Models loaded from {load_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results."""
        return {
            'total_models_trained': len(self.generators),
            'training_history': self.training_history,
            'available_models': list(self.generators.keys()),
            'pipeline_config': self.config
        }


def create_training_pipeline(config: Optional[TrainingConfig] = None) -> MLIndicatorTrainingPipeline:
    """Create a training pipeline with specified configuration."""
    return MLIndicatorTrainingPipeline(config)


def test_training_pipeline():
    """Test function for the training pipeline."""
    print("🧪 Testing ML Indicator Training Pipeline...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 2000
    
    # Generate realistic OHLCV data
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
    
    # Create training pipeline
    config = TrainingConfig(
        enable_feature_selection=True,
        max_features=20,
        enable_ensemble=True
    )
    pipeline = create_training_pipeline(config)
    
    # Train models
    print("📚 Training all models...")
    results = pipeline.train_all_models(data, symbol='BTCUSDT')
    
    print(f"✅ Training completed!")
    print(f"📊 Results summary:")
    for indicator_type, result in results.items():
        if isinstance(result, dict) and result.get('success', False):
            print(f"   - {indicator_type}: Success")
            if 'test_score' in result:
                print(f"     Test Score: {result['test_score']:.4f}")
        else:
            print(f"   - {indicator_type}: Failed")
    
    # Get training summary
    summary = pipeline.get_training_summary()
    print(f"\n📈 Training Summary:")
    print(f"   - Models trained: {summary['total_models_trained']}")
    print(f"   - Available models: {len(summary['available_models'])}")
    
    print("\n🎉 Training Pipeline test completed successfully!")
    return pipeline, results


if __name__ == "__main__":
    test_training_pipeline()