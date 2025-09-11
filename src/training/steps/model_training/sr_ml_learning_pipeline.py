"""
Support/Resistance ML Learning Pipeline

This module provides machine learning-based support and resistance level
prediction and validation using advanced ML techniques.

Key Features:
- ML model training for SR level prediction
- Feature engineering for SR analysis
- Model validation and performance metrics
- Data quality validation using existing utilities
- Integration with ML commons for enhanced analysis
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle

from src.utils.logger import system_logger
from src.utils.data.processing.transformers import DataFrameValidator, DataQualityReport
from src.utils.data.quality.data_quality import DataQualityFramework as EnhancedDataQualityValidator, QualityResult

from src.utils.ml_common.data_quality import DataQualityUtilities
from src.utils.ml_common.feature_selection import FeatureSelectionFramework
from src.utils.ml_common.model_training import ModelTrainingUtilities
from src.utils.core.common import CommonOperations
from src.utils.math_validation import MathValidation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

logger = system_logger.getChild('SRMLLearningPipeline')

@dataclass
class SRMLConfig:
    """Configuration for SR ML learning."""
    # Model parameters
    model_type: str = 'random_forest'  # 'random_forest', 'xgboost', 'lightgbm', 'neural_network'
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    
    # Feature engineering
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    technical_indicators: List[str] = field(default_factory=lambda: ['rsi', 'macd', 'bollinger', 'atr'])
    price_features: List[str] = field(default_factory=lambda: ['returns', 'volatility', 'momentum'])
    
    # Model training
    n_estimators: int = 100
    max_depth: int = 10
    learning_rate: float = 0.1
    early_stopping_rounds: int = 10
    
    # Validation
    cv_folds: int = 5
    scoring_metrics: List[str] = field(default_factory=lambda: ['accuracy', 'precision', 'recall', 'f1'])
    
    # Data quality
    enable_data_quality_validation: bool = True
    quality_thresholds: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SRMLResult:
    """Result of SR ML learning."""
    models: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    feature_importance: Dict[str, Any]
    predictions: Optional[pd.DataFrame] = None
    quality_report: Optional[QualityResult] = None
    training_history: Dict[str, Any] = field(default_factory=dict)

class SRMLLearningPipeline:
    """
    Support/Resistance ML Learning Pipeline.
    
    Provides machine learning-based SR level prediction and validation.
    """
    
    def __init__(self, config: Optional[SRMLConfig] = None):
        """Initialize SR ML learning pipeline."""
        self.config = config or SRMLConfig()
        self.logger = logger.getChild('SRMLLearningPipeline')
        self.common_ops = CommonOperations()
        self.math_validator = MathValidation()
        
        # Initialize ML utilities
        self.data_quality_validator = EnhancedDataQualityValidator()
        self.ml_data_quality = None
        self.feature_selector = None
        self.model_trainer = None
        
        try:
            self.ml_data_quality = DataQualityUtilities()
            self.feature_selector = FeatureSelectionFramework()
            self.model_trainer = ModelTrainingUtilities()
            self.logger.info("✅ ML utilities initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ ML utilities not available: {e}")
    
    async def train_sr_models(
        self,
        data_dir: str,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> SRMLResult:
        """
        Train ML models for SR level prediction.
        
        Args:
            data_dir: Data directory path
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            
        Returns:
            SRMLResult with trained models and metrics
        """
        self.logger.info(f"🤖 Starting SR ML learning for {symbol} on {exchange} ({timeframe})")
        
        try:
            # Load and validate data
            data = await self._load_and_validate_data(data_dir, symbol, exchange, timeframe)
            
            # Perform data quality validation
            quality_report = None
            if self.config.enable_data_quality_validation:
                quality_report = await self._validate_data_quality(data, symbol, exchange)
            
            # Generate SR labels
            sr_labels = await self._generate_sr_labels(data)
            
            # Engineer features
            features = await self._engineer_features(data)
            
            # Prepare training data
            X, y = await self._prepare_training_data(features, sr_labels)
            
            # Train models
            models, training_history = await self._train_models(X, y)
            
            # Evaluate models
            performance_metrics = await self._evaluate_models(models, X, y)
            
            # Get feature importance
            feature_importance = await self._get_feature_importance(models, features.columns)
            
            # Generate predictions
            predictions = await self._generate_predictions(models, X, y)
            
            result = SRMLResult(
                models=models,
                performance_metrics=performance_metrics,
                feature_importance=feature_importance,
                predictions=predictions,
                quality_report=quality_report,
                training_history=training_history
            )
            
            self.logger.info(f"✅ SR ML learning completed: {len(models)} models trained")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ SR ML learning failed: {e}")
            raise
    
    async def _load_and_validate_data(
        self,
        data_dir: str,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> pd.DataFrame:
        """Load and validate market data."""
        # Construct file path
        file_path = Path(data_dir) / f"aggtrades_{exchange}_{symbol}_consolidated.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data using standardized handler
        data = standardized_parquet_handler.read_parquet_standardized(file_path)
        
        # Basic validation
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort by timestamp if available
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp').reset_index(drop=True)
        
        self.logger.info(f"📊 Loaded {len(data)} data points for SR ML learning")
        return data
    
    async def _validate_data_quality(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str
    ) -> QualityResult:
        """Validate data quality using existing utilities."""
        self.logger.info("🔍 Performing data quality validation for ML training")
        
        try:
            # Use enhanced data quality validator
            quality_result = self.data_quality_validator.validate_dataframe(data)
            
            # Use ML data quality utilities if available
            if self.ml_data_quality:
                try:
                    ml_quality_report = await self.ml_data_quality.perform_comprehensive_validation(
                        data, symbol=symbol, exchange=exchange
                    )
                    
                    # Merge ML quality insights
                    if ml_quality_report.get('has_critical_issues', False):
                        for issue in ml_quality_report.get('critical_issues', []):
                            quality_result.add_issue('ml_critical', issue)
                    
                    if ml_quality_report.get('warnings', []):
                        for warning in ml_quality_report.get('warnings', []):
                            quality_result.add_warning('ml_warning', warning)
                    
                    self.logger.info("✅ ML-enhanced data quality validation completed")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ ML data quality validation failed: {e}")
            
            # Log quality results
            if quality_result.passed:
                self.logger.info("✅ Data quality validation passed")
            else:
                self.logger.warning(f"⚠️ Data quality issues found: {len(quality_result.issues)} issues, {len(quality_result.warnings)} warnings")
                for issue in quality_result.issues[:5]:  # Log first 5 issues
                    self.logger.warning(f"  - {issue}")
            
            return quality_result
            
        except Exception as e:
            self.logger.error(f"❌ Data quality validation failed: {e}")
            # Return a basic quality result
            return QualityResult(passed=False, issues=[f"Validation failed: {e}"])
    
    async def _generate_sr_labels(self, data: pd.DataFrame) -> pd.Series:
        """Generate SR labels for ML training."""
        self.logger.info("🏷️ Generating SR labels for ML training")
        
        try:
            # Simple SR label generation based on price action
            labels = pd.Series(0, index=data.index)  # 0 = neutral
            
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Look for support and resistance patterns
            for i in range(20, len(data) - 20):
                # Check for resistance (price rejection at highs)
                recent_high = np.max(high[i-20:i])
                if high[i] >= recent_high * 0.99 and close[i] < high[i] * 0.95:
                    labels.iloc[i] = 1  # Resistance
                
                # Check for support (price rejection at lows)
                recent_low = np.min(low[i-20:i])
                if low[i] <= recent_low * 1.01 and close[i] > low[i] * 1.05:
                    labels.iloc[i] = -1  # Support
            
            # Balance the dataset
            support_count = (labels == -1).sum()
            resistance_count = (labels == 1).sum()
            neutral_count = (labels == 0).sum()
            
            self.logger.info(f"🏷️ Generated labels: {support_count} support, {resistance_count} resistance, {neutral_count} neutral")
            
            return labels
            
        except Exception as e:
            self.logger.error(f"❌ SR label generation failed: {e}")
            # Return neutral labels as fallback
            return pd.Series(0, index=data.index)
    
    async def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML training."""
        self.logger.info("🔧 Engineering features for SR ML learning")
        
        try:
            features = pd.DataFrame(index=data.index)
            
            # Price-based features
            if 'returns' in self.config.price_features:
                features['returns_1'] = data['close'].pct_change(1)
                features['returns_5'] = data['close'].pct_change(5)
                features['returns_20'] = data['close'].pct_change(20)
            
            if 'volatility' in self.config.price_features:
                features['volatility_5'] = data['close'].rolling(5).std()
                features['volatility_20'] = data['close'].rolling(20).std()
            
            if 'momentum' in self.config.price_features:
                features['momentum_5'] = data['close'] / data['close'].shift(5) - 1
                features['momentum_20'] = data['close'] / data['close'].shift(20) - 1
            
            # Technical indicators
            if 'rsi' in self.config.technical_indicators:
                features['rsi_14'] = self._calculate_rsi(data['close'], 14)
                features['rsi_21'] = self._calculate_rsi(data['close'], 21)
            
            if 'macd' in self.config.technical_indicators:
                macd_line, signal_line, histogram = self._calculate_macd(data['close'])
                features['macd'] = macd_line
                features['macd_signal'] = signal_line
                features['macd_histogram'] = histogram
            
            if 'bollinger' in self.config.technical_indicators:
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['close'])
                features['bb_upper'] = bb_upper
                features['bb_middle'] = bb_middle
                features['bb_lower'] = bb_lower
                features['bb_width'] = (bb_upper - bb_lower) / bb_middle
                features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
            
            if 'atr' in self.config.technical_indicators:
                features['atr_14'] = self._calculate_atr(data, 14)
                features['atr_21'] = self._calculate_atr(data, 21)
            
            # Volume features
            features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            features['volume_trend'] = data['volume'].rolling(5).mean() / data['volume'].rolling(20).mean()
            
            # Price position features
            features['price_position_20'] = (data['close'] - data['low'].rolling(20).min()) / (data['high'].rolling(20).max() - data['low'].rolling(20).min())
            features['price_position_50'] = (data['close'] - data['low'].rolling(50).min()) / (data['high'].rolling(50).max() - data['low'].rolling(50).min())
            
            # Lookback features
            for period in self.config.lookback_periods:
                features[f'high_{period}'] = data['high'].rolling(period).max()
                features[f'low_{period}'] = data['low'].rolling(period).min()
                features[f'close_{period}'] = data['close'].rolling(period).mean()
            
            # Remove rows with NaN values
            features = features.dropna()
            
            self.logger.info(f"🔧 Engineered {len(features.columns)} features")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {e}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr
    
    async def _prepare_training_data(
        self,
        features: pd.DataFrame,
        labels: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data."""
        self.logger.info("📊 Preparing training data")
        
        try:
            # Align features and labels
            common_index = features.index.intersection(labels.index)
            X = features.loc[common_index]
            y = labels.loc[common_index]
            
            # Remove any remaining NaN values
            valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[valid_mask]
            y = y[valid_mask]
            
            self.logger.info(f"📊 Prepared training data: {len(X)} samples, {len(X.columns)} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"❌ Training data preparation failed: {e}")
            raise
    
    async def _train_models(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Train ML models."""
        self.logger.info(f"🤖 Training {self.config.model_type} model")
        
        try:
            models = {}
            training_history = {}
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )
            
            # Train model based on type
            if self.config.model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    random_state=self.config.random_state
                )
                model.fit(X_train, y_train)
                models['random_forest'] = model
                
            elif self.config.model_type == 'xgboost':
                try:
                    import xgboost as xgb
                    model = xgb.XGBClassifier(
                        n_estimators=self.config.n_estimators,
                        max_depth=self.config.max_depth,
                        learning_rate=self.config.learning_rate,
                        random_state=self.config.random_state
                    )
                    model.fit(X_train, y_train)
                    models['xgboost'] = model
                except ImportError:
                    self.logger.warning("⚠️ XGBoost not available, using Random Forest")
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(random_state=self.config.random_state)
                    model.fit(X_train, y_train)
                    models['random_forest'] = model
            
            elif self.config.model_type == 'lightgbm':
                try:
                    import lightgbm as lgb
                    model = lgb.LGBMClassifier(
                        n_estimators=self.config.n_estimators,
                        max_depth=self.config.max_depth,
                        learning_rate=self.config.learning_rate,
                        random_state=self.config.random_state
                    )
                    model.fit(X_train, y_train)
                    models['lightgbm'] = model
                except ImportError:
                    self.logger.warning("⚠️ LightGBM not available, using Random Forest")
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(random_state=self.config.random_state)
                    model.fit(X_train, y_train)
                    models['random_forest'] = model
            
            else:
                # Default to Random Forest
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(random_state=self.config.random_state)
                model.fit(X_train, y_train)
                models['random_forest'] = model
            
            # Store training history
            training_history = {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'feature_count': len(X.columns),
                'model_type': self.config.model_type
            }
            
            self.logger.info(f"✅ Model training completed: {len(models)} models trained")
            return models, training_history
            
        except Exception as e:
            self.logger.error(f"❌ Model training failed: {e}")
            raise
    
    async def _evaluate_models(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate model performance."""
        self.logger.info("📊 Evaluating model performance")
        
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            performance_metrics = {}
            
            for model_name, model in models.items():
                # Cross-validation scores
                cv_scores = cross_val_score(model, X, y, cv=self.config.cv_folds, scoring='accuracy')
                
                # Predictions
                y_pred = model.predict(X)
                
                # Calculate metrics
                metrics = {
                    'cv_accuracy_mean': cv_scores.mean(),
                    'cv_accuracy_std': cv_scores.std(),
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
                }
                
                performance_metrics[model_name] = metrics
            
            self.logger.info("✅ Model evaluation completed")
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Model evaluation failed: {e}")
            return {}
    
    async def _get_feature_importance(
        self,
        models: Dict[str, Any],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Get feature importance from models."""
        self.logger.info("📊 Extracting feature importance")
        
        try:
            feature_importance = {}
            
            for model_name, model in models.items():
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    importance_dict = dict(zip(feature_names, importance))
                    # Sort by importance
                    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
                    feature_importance[model_name] = sorted_importance
                else:
                    feature_importance[model_name] = {}
            
            self.logger.info("✅ Feature importance extracted")
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"❌ Feature importance extraction failed: {e}")
            return {}
    
    async def _generate_predictions(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """Generate predictions from models."""
        self.logger.info("🔮 Generating predictions")
        
        try:
            predictions = pd.DataFrame(index=X.index)
            predictions['actual'] = y
            
            for model_name, model in models.items():
                y_pred = model.predict(X)
                y_pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
                
                predictions[f'{model_name}_pred'] = y_pred
                if y_pred_proba is not None:
                    predictions[f'{model_name}_proba'] = y_pred_proba.max(axis=1)
            
            self.logger.info("✅ Predictions generated")
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ Prediction generation failed: {e}")
            return pd.DataFrame()

# Convenience function
async def train_sr_models(
    data_dir: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    config: Optional[SRMLConfig] = None
) -> SRMLResult:
    """Convenience function to train SR models."""
    pipeline = SRMLLearningPipeline(config)
    return await pipeline.train_sr_models(data_dir, symbol, exchange, timeframe)