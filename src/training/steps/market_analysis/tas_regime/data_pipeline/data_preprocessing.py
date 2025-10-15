"""
Data Preprocessing for TAS

Comprehensive data preprocessing system for tree architecture search including
data cleaning, normalization, feature engineering, and data quality assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import existing utilities from the codebase
try:
    from src.utils.data.processing.data_processing import DataProcessor
    DATA_PROCESSOR_AVAILABLE = True
except ImportError:
    DATA_PROCESSOR_AVAILABLE = False

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

try:
    from src.utils.hardware import (
        get_hardware_accelerator,
        get_memory_manager,
        get_performance_monitor
    )
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class PreprocessingStep(Enum):
    """Preprocessing steps."""
    CLEANING = "cleaning"
    NORMALIZATION = "normalization"
    FEATURE_ENGINEERING = "feature_engineering"
    OUTLIER_DETECTION = "outlier_detection"
    MISSING_DATA_HANDLING = "missing_data_handling"
    TIMESTAMP_REGULARIZATION = "timestamp_regularization"
    DATA_VALIDATION = "data_validation"

@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    
    # Preprocessing steps
    preprocessing_steps: List[PreprocessingStep] = field(default_factory=lambda: [
        PreprocessingStep.CLEANING,
        PreprocessingStep.NORMALIZATION,
        PreprocessingStep.FEATURE_ENGINEERING,
        PreprocessingStep.OUTLIER_DETECTION,
        PreprocessingStep.MISSING_DATA_HANDLING,
        PreprocessingStep.TIMESTAMP_REGULARIZATION,
        PreprocessingStep.DATA_VALIDATION
    ])
    
    # Data cleaning
    enable_data_cleaning: bool = True
    remove_duplicates: bool = True
    handle_missing_values: bool = True
    handle_infinite_values: bool = True
    convert_data_types: bool = True
    
    # Normalization
    enable_normalization: bool = True
    normalization_method: str = "standard"  # "standard", "robust", "minmax", "quantile"
    normalize_features: bool = True
    normalize_prices: bool = False
    
    # Feature engineering
    enable_feature_engineering: bool = True
    technical_indicators: bool = True
    price_features: bool = True
    volume_features: bool = True
    volatility_features: bool = True
    momentum_features: bool = True
    trend_features: bool = True
    
    # Outlier detection
    enable_outlier_detection: bool = True
    outlier_method: str = "zscore"  # "zscore", "iqr", "isolation_forest", "local_outlier_factor"
    outlier_threshold: float = 3.0
    outlier_handling: str = "cap"  # "cap", "remove", "winsorize"
    
    # Missing data handling
    enable_missing_data_handling: bool = True
    missing_data_method: str = "interpolate"  # "interpolate", "forward_fill", "backward_fill", "drop"
    max_missing_ratio: float = 0.1  # Maximum ratio of missing values to allow
    
    # Timestamp regularization
    enable_timestamp_regularization: bool = True
    expected_interval: Optional[timedelta] = None
    tolerance_seconds: int = 30
    regularization_method: str = "forward_fill"
    
    # Data validation
    enable_data_validation: bool = True
    validate_ohlc_consistency: bool = True
    validate_price_positive: bool = True
    validate_volume_non_negative: bool = True
    validate_timestamp_order: bool = True
    
    # Hardware acceleration
    enable_hardware_acceleration: bool = True
    enable_matrix_operations: bool = True
    enable_batch_processing: bool = True
    
    # Output configuration
    save_preprocessed_data: bool = True
    output_directory: str = "preprocessed_data"
    cache_intermediate_results: bool = True

@dataclass
class PreprocessingResult:
    """Result of data preprocessing."""
    
    # Processed data
    processed_data: pd.DataFrame
    original_data: pd.DataFrame
    data_shape: Tuple[int, int]
    data_columns: List[str]
    data_types: Dict[str, str]
    
    # Preprocessing statistics
    preprocessing_steps_applied: List[str]
    preprocessing_metadata: Dict[str, Any]
    data_quality_improvement: float
    
    # Feature engineering results
    technical_indicators: Dict[str, pd.Series]
    engineered_features: List[str]
    feature_statistics: Dict[str, Any]
    
    # Data quality metrics
    original_quality_score: float
    final_quality_score: float
    missing_values_before: Dict[str, int]
    missing_values_after: Dict[str, int]
    outliers_detected: int
    outliers_handled: int
    
    # Performance metrics
    preprocessing_time: float
    memory_usage: float
    hardware_acceleration_used: bool
    matrix_operations_used: bool
    
    # Metadata
    config: PreprocessingConfig
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class DataPreprocessor:
    """
    Comprehensive data preprocessor for TAS.
    
    Provides data cleaning, normalization, feature engineering,
    and data quality assessment for tree architecture search.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """Initialize data preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize processors
        self.data_processor = None
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None
        self.hardware_accelerator = None
        self.memory_manager = None
        self.performance_monitor = None
        
        # Initialize available processors
        self._initialize_processors()
        
        self.logger.info("✅ Data Preprocessor initialized")
        self.logger.info(f"📊 Preprocessing steps: {[step.value for step in config.preprocessing_steps]}")
        self.logger.info(f"📊 Hardware acceleration: {config.enable_hardware_acceleration}")
        self.logger.info(f"📊 Matrix operations: {config.enable_matrix_operations}")
    
    def _initialize_processors(self):
        """Initialize available processors."""
        # Initialize data processor if available
        if DATA_PROCESSOR_AVAILABLE:
            try:
                self.data_processor = DataProcessor()
                self.logger.info("✅ Data processor initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Data processor not available: {e}")
        
        # Initialize matrix operations if available
        if MATRIX_OPERATIONS_AVAILABLE and self.config.enable_matrix_operations:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available: {e}")
        
        # Initialize hardware acceleration if available
        if HARDWARE_ACCELERATION_AVAILABLE and self.config.enable_hardware_acceleration:
            try:
                self.hardware_accelerator = get_hardware_accelerator()
                self.memory_manager = get_memory_manager()
                self.performance_monitor = get_performance_monitor()
                self.logger.info("✅ Hardware acceleration initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available: {e}")
    
    def preprocess_data(self, data: pd.DataFrame) -> PreprocessingResult:
        """
        Preprocess data for TAS.
        
        Args:
            data: Input data to preprocess
            
        Returns:
            Preprocessing result
        """
        self.logger.info("🚀 Starting data preprocessing")
        start_time = datetime.now()
        
        try:
            # Store original data
            original_data = data.copy()
            
            # Calculate original data quality
            original_quality = self._calculate_data_quality(original_data)
            
            # Initialize preprocessing state
            preprocessing_metadata = {
                'steps_applied': [],
                'quality_improvements': {},
                'feature_engineering': {},
                'outlier_handling': {},
                'missing_data_handling': {}
            }
            
            # Apply preprocessing steps
            processed_data = data.copy()
            
            for step in self.config.preprocessing_steps:
                if step == PreprocessingStep.CLEANING and self.config.enable_data_cleaning:
                    processed_data = self._apply_data_cleaning(processed_data, preprocessing_metadata)
                
                elif step == PreprocessingStep.NORMALIZATION and self.config.enable_normalization:
                    processed_data = self._apply_normalization(processed_data, preprocessing_metadata)
                
                elif step == PreprocessingStep.FEATURE_ENGINEERING and self.config.enable_feature_engineering:
                    processed_data = self._apply_feature_engineering(processed_data, preprocessing_metadata)
                
                elif step == PreprocessingStep.OUTLIER_DETECTION and self.config.enable_outlier_detection:
                    processed_data = self._apply_outlier_detection(processed_data, preprocessing_metadata)
                
                elif step == PreprocessingStep.MISSING_DATA_HANDLING and self.config.enable_missing_data_handling:
                    processed_data = self._apply_missing_data_handling(processed_data, preprocessing_metadata)
                
                elif step == PreprocessingStep.TIMESTAMP_REGULARIZATION and self.config.enable_timestamp_regularization:
                    processed_data = self._apply_timestamp_regularization(processed_data, preprocessing_metadata)
                
                elif step == PreprocessingStep.DATA_VALIDATION and self.config.enable_data_validation:
                    processed_data = self._apply_data_validation(processed_data, preprocessing_metadata)
            
            # Calculate final data quality
            final_quality = self._calculate_data_quality(processed_data)
            
            # Calculate quality improvement
            quality_improvement = final_quality - original_quality
            
            # Calculate performance metrics
            preprocessing_time = (datetime.now() - start_time).total_seconds()
            memory_usage = processed_data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            # Create comprehensive result
            result = PreprocessingResult(
                # Processed data
                processed_data=processed_data,
                original_data=original_data,
                data_shape=processed_data.shape,
                data_columns=list(processed_data.columns),
                data_types=processed_data.dtypes.to_dict(),
                
                # Preprocessing statistics
                preprocessing_steps_applied=preprocessing_metadata['steps_applied'],
                preprocessing_metadata=preprocessing_metadata,
                data_quality_improvement=quality_improvement,
                
                # Feature engineering results
                technical_indicators=preprocessing_metadata.get('feature_engineering', {}).get('technical_indicators', {}),
                engineered_features=preprocessing_metadata.get('feature_engineering', {}).get('engineered_features', []),
                feature_statistics=preprocessing_metadata.get('feature_engineering', {}).get('feature_statistics', {}),
                
                # Data quality metrics
                original_quality_score=original_quality,
                final_quality_score=final_quality,
                missing_values_before=preprocessing_metadata.get('missing_data_handling', {}).get('before', {}),
                missing_values_after=preprocessing_metadata.get('missing_data_handling', {}).get('after', {}),
                outliers_detected=preprocessing_metadata.get('outlier_handling', {}).get('detected', 0),
                outliers_handled=preprocessing_metadata.get('outlier_handling', {}).get('handled', 0),
                
                # Performance metrics
                preprocessing_time=preprocessing_time,
                memory_usage=memory_usage,
                hardware_acceleration_used=self.hardware_accelerator is not None,
                matrix_operations_used=self.matrix_ops is not None,
                
                # Metadata
                config=self.config
            )
            
            # Save preprocessed data if configured
            if self.config.save_preprocessed_data:
                self._save_preprocessed_data(result)
            
            self.logger.info(f"✅ Data preprocessing completed in {result.preprocessing_time:.2f}s")
            self.logger.info(f"📊 Data shape: {result.data_shape}")
            self.logger.info(f"📊 Quality improvement: {result.data_quality_improvement:.3f}")
            self.logger.info(f"📊 Steps applied: {len(result.preprocessing_steps_applied)}")
            self.logger.info(f"📊 Memory usage: {result.memory_usage:.2f} MB")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Data preprocessing failed: {e}")
            raise
    
    def _apply_data_cleaning(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Apply data cleaning."""
        self.logger.info("🧹 Applying data cleaning...")
        
        try:
            if self.data_processor:
                cleaned_data, cleaning_metadata = self.data_processor.clean_data(data)
                metadata['steps_applied'].append('data_cleaning')
                metadata['quality_improvements']['data_cleaning'] = cleaning_metadata
                return cleaned_data
            else:
                # Fallback to basic cleaning
                return self._basic_data_cleaning(data, metadata)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Data cleaning failed: {e}")
            return data
    
    def _basic_data_cleaning(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Apply basic data cleaning."""
        cleaned_data = data.copy()
        
        # Remove duplicates
        if self.config.remove_duplicates:
            initial_rows = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            removed_duplicates = initial_rows - len(cleaned_data)
            if removed_duplicates > 0:
                metadata['steps_applied'].append('remove_duplicates')
                metadata['quality_improvements']['duplicates_removed'] = removed_duplicates
        
        # Handle missing values
        if self.config.handle_missing_values:
            missing_before = cleaned_data.isnull().sum().to_dict()
            cleaned_data = cleaned_data.fillna(cleaned_data.median())
            missing_after = cleaned_data.isnull().sum().to_dict()
            metadata['steps_applied'].append('handle_missing_values')
            metadata['quality_improvements']['missing_values_handled'] = {
                'before': missing_before,
                'after': missing_after
            }
        
        # Handle infinite values
        if self.config.handle_infinite_values:
            inf_mask = np.isinf(cleaned_data.select_dtypes(include=[np.number]))
            if inf_mask.any().any():
                cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan).fillna(cleaned_data.median())
                metadata['steps_applied'].append('handle_infinite_values')
        
        # Convert data types
        if self.config.convert_data_types:
            for col in cleaned_data.columns:
                if cleaned_data[col].dtype == 'object':
                    try:
                        cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                    except:
                        pass
            metadata['steps_applied'].append('convert_data_types')
        
        return cleaned_data
    
    def _apply_normalization(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Apply data normalization."""
        self.logger.info("📊 Applying data normalization...")
        
        try:
            normalized_data = data.copy()
            
            # Normalize features if configured
            if self.config.normalize_features:
                numeric_cols = normalized_data.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if self.config.normalization_method == "standard":
                        normalized_data[col] = (normalized_data[col] - normalized_data[col].mean()) / normalized_data[col].std()
                    elif self.config.normalization_method == "robust":
                        median = normalized_data[col].median()
                        mad = np.median(np.abs(normalized_data[col] - median))
                        normalized_data[col] = (normalized_data[col] - median) / mad
                    elif self.config.normalization_method == "minmax":
                        min_val = normalized_data[col].min()
                        max_val = normalized_data[col].max()
                        normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
                
                metadata['steps_applied'].append('normalize_features')
                metadata['quality_improvements']['normalization'] = {
                    'method': self.config.normalization_method,
                    'columns_normalized': len(numeric_cols)
                }
            
            return normalized_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Normalization failed: {e}")
            return data
    
    def _apply_feature_engineering(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Apply feature engineering."""
        self.logger.info("🔧 Applying feature engineering...")
        
        try:
            engineered_data = data.copy()
            technical_indicators = {}
            engineered_features = []
            
            # Technical indicators
            if self.config.technical_indicators:
                technical_indicators.update(self._calculate_technical_indicators(engineered_data))
                engineered_features.extend(list(technical_indicators.keys()))
            
            # Price features
            if self.config.price_features:
                price_features = self._calculate_price_features(engineered_data)
                engineered_data = pd.concat([engineered_data, price_features], axis=1)
                engineered_features.extend(list(price_features.columns))
            
            # Volume features
            if self.config.volume_features:
                volume_features = self._calculate_volume_features(engineered_data)
                engineered_data = pd.concat([engineered_data, volume_features], axis=1)
                engineered_features.extend(list(volume_features.columns))
            
            # Volatility features
            if self.config.volatility_features:
                volatility_features = self._calculate_volatility_features(engineered_data)
                engineered_data = pd.concat([engineered_data, volatility_features], axis=1)
                engineered_features.extend(list(volatility_features.columns))
            
            # Momentum features
            if self.config.momentum_features:
                momentum_features = self._calculate_momentum_features(engineered_data)
                engineered_data = pd.concat([engineered_data, momentum_features], axis=1)
                engineered_features.extend(list(momentum_features.columns))
            
            # Trend features
            if self.config.trend_features:
                trend_features = self._calculate_trend_features(engineered_data)
                engineered_data = pd.concat([engineered_data, trend_features], axis=1)
                engineered_features.extend(list(trend_features.columns))
            
            metadata['steps_applied'].append('feature_engineering')
            metadata['feature_engineering'] = {
                'technical_indicators': technical_indicators,
                'engineered_features': engineered_features,
                'feature_statistics': self._calculate_feature_statistics(engineered_data)
            }
            
            return engineered_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature engineering failed: {e}")
            return data
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators."""
        indicators = {}
        
        try:
            # Moving averages
            if 'close' in data.columns:
                indicators['sma_20'] = rolling_mean(data["close"], window=20) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=20).mean()
                indicators['sma_50'] = rolling_mean(data["close"], window=50) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=50).mean()
                indicators['ema_20'] = data['close'].ewm(span=20).mean()
                indicators['ema_50'] = data['close'].ewm(span=50).mean()
            
            # RSI
            if 'close' in data.columns:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            if 'close' in data.columns:
                sma_20 = rolling_mean(data["close"], window=20) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=20).mean()
                std_20 = rolling_std(data["close"], window=20) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=20).std()
                indicators['bb_upper'] = sma_20 + (std_20 * 2)
                indicators['bb_lower'] = sma_20 - (std_20 * 2)
                indicators['bb_middle'] = sma_20
            
            # MACD
            if 'close' in data.columns:
                ema_12 = data['close'].ewm(span=12).mean()
                ema_26 = data['close'].ewm(span=26).mean()
                indicators['macd'] = ema_12 - ema_26
                indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
                indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
        except Exception as e:
            self.logger.warning(f"⚠️ Technical indicators calculation failed: {e}")
        
        return indicators
    
    def _calculate_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based features."""
        features = pd.DataFrame(index=data.index)
        
        try:
            if 'close' in data.columns:
                # Price returns
                features['price_return'] = data['close'].pct_change()
                features['log_return'] = np.log(data['close'] / data['close'].shift(1))
                
                # Price ratios
                if 'open' in data.columns:
                    features['open_close_ratio'] = data['open'] / data['close']
                if 'high' in data.columns:
                    features['high_close_ratio'] = data['high'] / data['close']
                if 'low' in data.columns:
                    features['low_close_ratio'] = data['low'] / data['close']
                
                # Price position within range
                if all(col in data.columns for col in ['high', 'low']):
                    features['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
            
        except Exception as e:
            self.logger.warning(f"⚠️ Price features calculation failed: {e}")
        
        return features
    
    def _calculate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        features = pd.DataFrame(index=data.index)
        
        try:
            if 'volume' in data.columns:
                # Volume returns
                features['volume_return'] = data['volume'].pct_change()
                features['log_volume'] = np.log(data['volume'] + 1)
                
                # Volume moving averages
                features['volume_sma_20'] = rolling_mean(data["volume"], window=20) if VECTORBT_AVAILABLE and len(data) > 1000 else data["volume"].rolling(window=20).mean()
                features['volume_sma_50'] = rolling_mean(data["volume"], window=50) if VECTORBT_AVAILABLE and len(data) > 1000 else data["volume"].rolling(window=50).mean()
                
                # Volume ratios
                features['volume_ratio_20'] = data['volume'] / features['volume_sma_20']
                features['volume_ratio_50'] = data['volume'] / features['volume_sma_50']
                
                # Volume volatility
                features['volume_volatility'] = rolling_std(data["volume"], window=20) if VECTORBT_AVAILABLE and len(data) > 1000 else data["volume"].rolling(window=20).std()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volume features calculation failed: {e}")
        
        return features
    
    def _calculate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based features."""
        features = pd.DataFrame(index=data.index)
        
        try:
            if 'close' in data.columns:
                # Rolling volatility
                features['volatility_20'] = rolling_std(data["close"], window=20) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=20).std()
                features['volatility_50'] = rolling_std(data["close"], window=50) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=50).std()
                
                # Volatility ratios
                features['volatility_ratio'] = features['volatility_20'] / features['volatility_50']
                
                # GARCH-like features
                returns = data['close'].pct_change()
                features['squared_returns'] = returns ** 2
                features['abs_returns'] = np.abs(returns)
                
                # Volatility of volatility
                features['vol_of_vol'] = features['volatility_20'].rolling(window=20).std()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volatility features calculation failed: {e}")
        
        return features
    
    def _calculate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based features."""
        features = pd.DataFrame(index=data.index)
        
        try:
            if 'close' in data.columns:
                # Price momentum
                features['momentum_5'] = data['close'] / data['close'].shift(5) - 1
                features['momentum_10'] = data['close'] / data['close'].shift(10) - 1
                features['momentum_20'] = data['close'] / data['close'].shift(20) - 1
                
                # Rate of change
                features['roc_5'] = data['close'].pct_change(5)
                features['roc_10'] = data['close'].pct_change(10)
                features['roc_20'] = data['close'].pct_change(20)
                
                # Momentum oscillators
                if 'high' in data.columns and 'low' in data.columns:
                    # Stochastic oscillator
                    lowest_low = data['low'].rolling(window=14).min()
                    highest_high = data['high'].rolling(window=14).max()
                    features['stoch_k'] = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
                    features['stoch_d'] = features['stoch_k'].rolling(window=3).mean()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Momentum features calculation failed: {e}")
        
        return features
    
    def _calculate_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-based features."""
        features = pd.DataFrame(index=data.index)
        
        try:
            if 'close' in data.columns:
                # Trend direction
                features['trend_5'] = np.where(data['close'] > data['close'].shift(5), 1, -1)
                features['trend_10'] = np.where(data['close'] > data['close'].shift(10), 1, -1)
                features['trend_20'] = np.where(data['close'] > data['close'].shift(20), 1, -1)
                
                # Trend strength
                features['trend_strength_5'] = np.abs(data['close'] - data['close'].shift(5))
                features['trend_strength_10'] = np.abs(data['close'] - data['close'].shift(10))
                features['trend_strength_20'] = np.abs(data['close'] - data['close'].shift(20))
                
                # Linear regression slope
                features['trend_slope_20'] = data['close'].rolling(window=20).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan
                )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trend features calculation failed: {e}")
        
        return features
    
    def _calculate_feature_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate feature statistics."""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            stats = {}
            
            for col in numeric_cols:
                stats[col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'median': float(data[col].median()),
                    'skewness': float(data[col].skew()),
                    'kurtosis': float(data[col].kurtosis())
                }
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature statistics calculation failed: {e}")
            return {}
    
    def _apply_outlier_detection(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Apply outlier detection and handling."""
        self.logger.info("🔍 Applying outlier detection...")
        
        try:
            outlier_data = data.copy()
            outliers_detected = 0
            outliers_handled = 0
            
            numeric_cols = outlier_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if self.config.outlier_method == "zscore":
                    z_scores = np.abs((outlier_data[col] - outlier_data[col].mean()) / outlier_data[col].std())
                    outlier_mask = z_scores > self.config.outlier_threshold
                elif self.config.outlier_method == "iqr":
                    Q1 = outlier_data[col].quantile(0.25)
                    Q3 = outlier_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_mask = (outlier_data[col] < Q1 - 1.5 * IQR) | (outlier_data[col] > Q3 + 1.5 * IQR)
                else:
                    continue
                
                outliers_detected += outlier_mask.sum()
                
                if self.config.outlier_handling == "cap":
                    # Cap outliers at threshold
                    if self.config.outlier_method == "zscore":
                        mean_val = outlier_data[col].mean()
                        std_val = outlier_data[col].std()
                        outlier_data.loc[outlier_mask, col] = mean_val + np.sign(outlier_data.loc[outlier_mask, col] - mean_val) * self.config.outlier_threshold * std_val
                    elif self.config.outlier_method == "iqr":
                        Q1 = outlier_data[col].quantile(0.25)
                        Q3 = outlier_data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outlier_data.loc[outlier_data[col] < Q1 - 1.5 * IQR, col] = Q1 - 1.5 * IQR
                        outlier_data.loc[outlier_data[col] > Q3 + 1.5 * IQR, col] = Q3 + 1.5 * IQR
                    
                    outliers_handled += outlier_mask.sum()
                elif self.config.outlier_handling == "remove":
                    outlier_data = outlier_data[~outlier_mask]
                    outliers_handled += outlier_mask.sum()
                elif self.config.outlier_handling == "winsorize":
                    # Winsorize outliers
                    if self.config.outlier_method == "iqr":
                        Q1 = outlier_data[col].quantile(0.25)
                        Q3 = outlier_data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outlier_data.loc[outlier_data[col] < Q1 - 1.5 * IQR, col] = Q1 - 1.5 * IQR
                        outlier_data.loc[outlier_data[col] > Q3 + 1.5 * IQR, col] = Q3 + 1.5 * IQR
                        outliers_handled += outlier_mask.sum()
            
            metadata['steps_applied'].append('outlier_detection')
            metadata['outlier_handling'] = {
                'detected': outliers_detected,
                'handled': outliers_handled,
                'method': self.config.outlier_method,
                'handling': self.config.outlier_handling
            }
            
            return outlier_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Outlier detection failed: {e}")
            return data
    
    def _apply_missing_data_handling(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Apply missing data handling."""
        self.logger.info("🔧 Applying missing data handling...")
        
        try:
            missing_data = data.copy()
            missing_before = missing_data.isnull().sum().to_dict()
            
            # Check if missing data ratio is acceptable
            total_values = len(missing_data) * len(missing_data.columns)
            missing_values = missing_data.isnull().sum().sum()
            missing_ratio = missing_values / total_values
            
            if missing_ratio > self.config.max_missing_ratio:
                self.logger.warning(f"⚠️ High missing data ratio: {missing_ratio:.3f} > {self.config.max_missing_ratio}")
            
            # Handle missing data
            if self.config.missing_data_method == "interpolate":
                missing_data = missing_data.interpolate()
            elif self.config.missing_data_method == "forward_fill":
                missing_data = missing_data.fillna(method='ffill')
            elif self.config.missing_data_method == "backward_fill":
                missing_data = missing_data.fillna(method='bfill')
            elif self.config.missing_data_method == "drop":
                missing_data = missing_data.dropna()
            
            missing_after = missing_data.isnull().sum().to_dict()
            
            metadata['steps_applied'].append('missing_data_handling')
            metadata['missing_data_handling'] = {
                'before': missing_before,
                'after': missing_after,
                'method': self.config.missing_data_method,
                'missing_ratio': missing_ratio
            }
            
            return missing_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Missing data handling failed: {e}")
            return data
    
    def _apply_timestamp_regularization(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Apply timestamp regularization."""
        self.logger.info("⏰ Applying timestamp regularization...")
        
        try:
            if self.data_processor:
                regularized_data = self.data_processor.regularize_timestamps(
                    data,
                    expected_interval=self.config.expected_interval,
                    tolerance_seconds=self.config.tolerance_seconds,
                    method=self.config.regularization_method
                )
                metadata['steps_applied'].append('timestamp_regularization')
                return regularized_data
            else:
                return data
                
        except Exception as e:
            self.logger.warning(f"⚠️ Timestamp regularization failed: {e}")
            return data
    
    def _apply_data_validation(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Apply data validation."""
        self.logger.info("✅ Applying data validation...")
        
        try:
            validated_data = data.copy()
            validation_issues = []
            
            # Validate OHLC consistency
            if self.config.validate_ohlc_consistency:
                if all(col in validated_data.columns for col in ['open', 'high', 'low', 'close']):
                    # Check high >= max(open, close)
                    high_consistency = validated_data['high'] >= np.maximum(validated_data['open'], validated_data['close'])
                    if not high_consistency.all():
                        validation_issues.append("High price consistency issues")
                    
                    # Check low <= min(open, close)
                    low_consistency = validated_data['low'] <= np.minimum(validated_data['open'], validated_data['close'])
                    if not low_consistency.all():
                        validation_issues.append("Low price consistency issues")
            
            # Validate price positivity
            if self.config.validate_price_positive:
                price_cols = ['open', 'high', 'low', 'close']
                for col in price_cols:
                    if col in validated_data.columns:
                        if (validated_data[col] <= 0).any():
                            validation_issues.append(f"Negative prices in {col}")
            
            # Validate volume non-negativity
            if self.config.validate_volume_non_negative:
                if 'volume' in validated_data.columns:
                    if (validated_data['volume'] < 0).any():
                        validation_issues.append("Negative volume values")
            
            # Validate timestamp order
            if self.config.validate_timestamp_order:
                if isinstance(validated_data.index, pd.DatetimeIndex):
                    if not validated_data.index.is_monotonic_increasing:
                        validation_issues.append("Timestamp order issues")
                        validated_data = validated_data.sort_index()
            
            metadata['steps_applied'].append('data_validation')
            metadata['data_validation'] = {
                'issues_found': validation_issues,
                'validation_passed': len(validation_issues) == 0
            }
            
            return validated_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Data validation failed: {e}")
            return data
    
    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
        """Calculate data quality score."""
        try:
            # Calculate missing values ratio
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            
            # Calculate outlier ratio (simplified)
            outlier_ratio = 0
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_ratio += (z_scores > 3).sum() / len(data)
            outlier_ratio /= len(numeric_cols) if len(numeric_cols) > 0 else 1
            
            # Calculate data quality score
            quality_score = 1.0 - missing_ratio - outlier_ratio
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Data quality calculation failed: {e}")
            return 0.0
    
    def _save_preprocessed_data(self, result: PreprocessingResult):
        """Save preprocessed data to file."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save preprocessed data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"preprocessed_data_{timestamp}.parquet"
            filepath = output_dir / filename
            
            result.processed_data.to_parquet(filepath)
            
            # Save metadata
            metadata_file = output_dir / f"preprocessing_metadata_{timestamp}.json"
            import json

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

except ImportError:
    
    cp = None
            metadata = {
                'data_shape': result.data_shape,
                'preprocessing_steps_applied': result.preprocessing_steps_applied,
                'data_quality_improvement': result.data_quality_improvement,
                'original_quality_score': result.original_quality_score,
                'final_quality_score': result.final_quality_score,
                'preprocessing_time': result.preprocessing_time,
                'memory_usage': result.memory_usage,
                'hardware_acceleration_used': result.hardware_acceleration_used,
                'matrix_operations_used': result.matrix_operations_used
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"📁 Preprocessed data saved to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save preprocessed data: {e}")
    
    def export_data(self, result: PreprocessingResult, filepath: str):
        """Export preprocessed data to file."""
        try:
            result.processed_data.to_csv(filepath)
            self.logger.info(f"📁 Preprocessed data exported to {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Failed to export data: {e}")

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
