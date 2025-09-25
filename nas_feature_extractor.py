"""
Neural Architecture Search (NAS) Feature Extractor

This module provides a comprehensive NASFeatureExtractor class that combines
neural architecture search with advanced feature engineering capabilities,
optimized for Apple Silicon (M1/M2/M3) hardware.
"""

import logging
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import json
import pickle

# Import utility modules
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, 
    safe_convert_dtypes, calculate_data_quality_metrics,
    safe_merge_dataframes, create_summary_statistics,
    optimize_dataframe_dtypes, get_dataframe_info,
    integrate_with_m1_optimizers, memory_checkpoint, gpu_context
)
from src.utils.common_utilities import (
    safe_dataframe_operation as safe_df_op,
    validate_dataframe_columns as validate_df_cols,
    safe_convert_dtypes as safe_convert,
    calculate_data_quality_metrics as calc_quality,
    create_summary_statistics as create_summary
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power,
    validate_finite, validate_positive, validate_range,
    safe_correlation, safe_covariance, safe_mean, safe_std,
    MathValidation
)
from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, tprint_with_level, LogLevel
)
from src.utils.hardware.m1_gpu_utils import (
    get_m1_gpu_manager, is_m1_available, is_mps_available,
    optimize_dataframe_for_m1, create_m1_optimized_array
)
from src.utils.hardware.m1_memory_optimizer import (
    get_m1_memory_optimizer, optimize_dataframe_memory
)
from src.utils.hardware.m1_cpu_optimizer import (
    get_m1_cpu_optimizer, parallel_map_m1, create_m1_optimized_thread_pool
)

# Setup logging
logger = logging.getLogger(__name__)

class NASFeatureExtractor:
    """
    Neural Architecture Search Feature Extractor with M1 optimization.
    
    This class provides comprehensive feature engineering capabilities
    combined with neural architecture search for optimal feature selection
    and transformation.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 enable_m1_optimization: bool = True,
                 memory_limit_gb: Optional[float] = None):
        """
        Initialize NASFeatureExtractor.
        
        Args:
            config: Configuration dictionary
            enable_m1_optimization: Enable M1 hardware optimizations
            memory_limit_gb: Memory limit in GB for M1 optimization
        """
        self.config = config or {}
        self.enable_m1_optimization = enable_m1_optimization
        self.memory_limit_gb = memory_limit_gb
        
        # Initialize components
        self.logger = logger.getChild('NASFeatureExtractor')
        self.math_validator = MathValidation()
        
        # Initialize M1 optimizers if enabled
        if self.enable_m1_optimization:
            self._initialize_m1_optimizers()
        
        # Initialize serializers
        self.json_serializer = JSONSerializer()
        self.pickle_serializer = PickleSerializer()
        self.parquet_serializer = ParquetSerializer()
        self.universal_serializer = UniversalSerializer()
        
        # Feature engineering state
        self.feature_pipeline = []
        self.extracted_features = {}
        self.feature_importance = {}
        self.architecture_history = []
        
        # Performance tracking
        self.performance_metrics = {}
        self.optimization_stats = {}
        
        tprint_success("🧠 NASFeatureExtractor initialized successfully")
    
    def _initialize_m1_optimizers(self):
        """Initialize M1 hardware optimizers."""
        try:
            # Initialize M1 components
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer(self.memory_limit_gb)
            self.cpu_optimizer = get_m1_cpu_optimizer()
            
            # Start memory monitoring
            self.memory_optimizer.start_monitoring()
            
            # Optimize numpy for M1
            self.cpu_optimizer.optimize_numpy_operations()
            
            # Get hardware info
            gpu_info = self.gpu_manager.get_gpu_info()
            cpu_info = self.cpu_optimizer.get_cpu_info()
            
            tprint_info(f"🧠 M1 Hardware Status:")
            tprint_info(f"   - M1 Available: {is_m1_available()}")
            tprint_info(f"   - MPS Available: {is_mps_available()}")
            tprint_info(f"   - Performance Cores: {cpu_info.get('performance_cores', 'Unknown')}")
            tprint_info(f"   - Memory Monitoring: Active")
            
            self.optimization_stats['m1_integration'] = {
                'gpu_available': is_mps_available(),
                'memory_optimizer': True,
                'cpu_optimizer': True,
                'gpu_info': gpu_info,
                'cpu_info': cpu_info
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ M1 optimization initialization failed: {e}")
            self.enable_m1_optimization = False
    
    def extract_features(self, 
                        data: pd.DataFrame,
                        target_column: Optional[str] = None,
                        feature_types: List[str] = None,
                        max_features: int = 100,
                        enable_architectural_search: bool = True) -> Dict[str, Any]:
        """
        Extract features using neural architecture search.
        
        Args:
            data: Input DataFrame
            target_column: Target column for supervised feature selection
            feature_types: Types of features to extract
            max_features: Maximum number of features to extract
            enable_architectural_search: Enable NAS for feature selection
            
        Returns:
            Dictionary with extracted features and metadata
        """
        start_time = time.time()
        
        with tprint_timer("Feature Extraction"):
            try:
                # Validate input data
                if not self._validate_input_data(data):
                    return {'success': False, 'error': 'Invalid input data'}
                
                # Optimize data for M1 if enabled
                if self.enable_m1_optimization:
                    with memory_checkpoint("feature_extraction"):
                        data = self._optimize_data_for_m1(data)
                
                # Initialize feature extraction pipeline
                feature_pipeline = self._create_feature_pipeline(
                    feature_types or ['technical', 'statistical', 'temporal']
                )
                
                # Extract features
                extracted_features = {}
                feature_importance = {}
                
                for feature_type in feature_pipeline:
                    tprint_info(f"🔧 Extracting {feature_type} features...")
                    
                    with tprint_timer(f"{feature_type}_features"):
                        features = self._extract_feature_type(
                            data, feature_type, target_column
                        )
                        
                        if features is not None and not features.empty:
                            extracted_features[feature_type] = features
                            
                            # Calculate feature importance if target provided
                            if target_column and target_column in data.columns:
                                importance = self._calculate_feature_importance(
                                    features, data[target_column]
                                )
                                feature_importance[feature_type] = importance
                
                # Combine all features
                combined_features = self._combine_features(extracted_features)
                
                # Apply neural architecture search if enabled
                if enable_architectural_search and len(combined_features.columns) > max_features:
                    tprint_info("🧠 Applying Neural Architecture Search for feature selection...")
                    
                    with tprint_timer("nas_feature_selection"):
                        selected_features = self._apply_nas_feature_selection(
                            combined_features, data[target_column] if target_column else None,
                            max_features
                        )
                else:
                    selected_features = combined_features
                
                # Store results
                self.extracted_features = extracted_features
                self.feature_importance = feature_importance
                
                # Calculate performance metrics
                extraction_time = time.time() - start_time
                self.performance_metrics = {
                    'extraction_time': extraction_time,
                    'total_features': len(combined_features.columns),
                    'selected_features': len(selected_features.columns),
                    'feature_types': list(extracted_features.keys()),
                    'm1_optimized': self.enable_m1_optimization
                }
                
                tprint_success(f"✅ Feature extraction completed: {len(selected_features.columns)} features in {extraction_time:.2f}s")
                
                return {
                    'success': True,
                    'features': selected_features,
                    'feature_importance': feature_importance,
                    'performance_metrics': self.performance_metrics,
                    'extraction_pipeline': self.feature_pipeline
                }
                
            except Exception as e:
                tprint_error(f"❌ Feature extraction failed: {e}")
                return {'success': False, 'error': str(e)}
    
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for feature extraction."""
        try:
            if data is None or data.empty:
                tprint_error("❌ Input data is empty or None")
                return False
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                tprint_warning(f"⚠️ Missing required columns: {missing_columns}")
                # Continue with available columns
            
            # Validate data quality
            quality_metrics = calculate_data_quality_metrics(data)
            
            if quality_metrics.get('missing_percentage', 0) > 50:
                tprint_warning("⚠️ High percentage of missing values detected")
            
            if quality_metrics.get('duplicate_percentage', 0) > 10:
                tprint_warning("⚠️ High percentage of duplicate rows detected")
            
            tprint_info(f"📊 Data validation passed: {len(data)} rows, {len(data.columns)} columns")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            return False
    
    def _optimize_data_for_m1(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for M1 hardware."""
        try:
            # Apply M1 memory optimization
            optimized_data = optimize_dataframe_for_m1(data)
            
            # Apply memory optimization
            optimized_data = self.memory_optimizer.optimize_dataframe_memory(optimized_data)
            
            tprint_info("🧠 Data optimized for M1 hardware")
            return optimized_data
            
        except Exception as e:
            tprint_warning(f"⚠️ M1 optimization failed: {e}")
            return data
    
    def _create_feature_pipeline(self, feature_types: List[str]) -> List[str]:
        """Create feature extraction pipeline."""
        pipeline = []
        
        for feature_type in feature_types:
            if feature_type == 'technical':
                pipeline.extend(['sma', 'ema', 'rsi', 'macd', 'bollinger_bands'])
            elif feature_type == 'statistical':
                pipeline.extend(['rolling_stats', 'volatility', 'momentum'])
            elif feature_type == 'temporal':
                pipeline.extend(['time_features', 'lag_features', 'seasonality'])
            elif feature_type == 'price_action':
                pipeline.extend(['candlestick_patterns', 'support_resistance'])
        
        self.feature_pipeline = pipeline
        return pipeline
    
    def _extract_feature_type(self, 
                             data: pd.DataFrame, 
                             feature_type: str,
                             target_column: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Extract specific type of features."""
        try:
            if feature_type == 'sma':
                return self._extract_sma_features(data)
            elif feature_type == 'ema':
                return self._extract_ema_features(data)
            elif feature_type == 'rsi':
                return self._extract_rsi_features(data)
            elif feature_type == 'macd':
                return self._extract_macd_features(data)
            elif feature_type == 'bollinger_bands':
                return self._extract_bollinger_bands_features(data)
            elif feature_type == 'rolling_stats':
                return self._extract_rolling_stats_features(data)
            elif feature_type == 'volatility':
                return self._extract_volatility_features(data)
            elif feature_type == 'momentum':
                return self._extract_momentum_features(data)
            elif feature_type == 'time_features':
                return self._extract_time_features(data)
            elif feature_type == 'lag_features':
                return self._extract_lag_features(data)
            elif feature_type == 'seasonality':
                return self._extract_seasonality_features(data)
            else:
                tprint_warning(f"⚠️ Unknown feature type: {feature_type}")
                return None
                
        except Exception as e:
            tprint_error(f"❌ Feature extraction failed for {feature_type}: {e}")
            return None
    
    def _extract_sma_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract Simple Moving Average features."""
        features = pd.DataFrame(index=data.index)
        
        periods = [5, 10, 20, 50, 100, 200]
        
        for period in periods:
            if len(data) >= period:
                features[f'sma_{period}'] = data['close'].rolling(window=period).mean()
                features[f'sma_{period}_ratio'] = data['close'] / features[f'sma_{period}']
        
        return features.dropna()
    
    def _extract_ema_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract Exponential Moving Average features."""
        features = pd.DataFrame(index=data.index)
        
        periods = [5, 10, 20, 50, 100]
        
        for period in periods:
            if len(data) >= period:
                features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
                features[f'ema_{period}_ratio'] = data['close'] / features[f'ema_{period}']
        
        return features.dropna()
    
    def _extract_rsi_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract RSI features."""
        features = pd.DataFrame(index=data.index)
        
        periods = [14, 21, 30]
        
        for period in periods:
            if len(data) >= period:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                features[f'rsi_{period}'] = rsi
                features[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)
                features[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)
        
        return features.dropna()
    
    def _extract_macd_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract MACD features."""
        features = pd.DataFrame(index=data.index)
        
        if len(data) >= 26:
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            features['macd'] = macd_line
            features['macd_signal'] = signal_line
            features['macd_histogram'] = histogram
            features['macd_crossover'] = ((macd_line > signal_line) & 
                                        (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
        
        return features.dropna()
    
    def _extract_bollinger_bands_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract Bollinger Bands features."""
        features = pd.DataFrame(index=data.index)
        
        periods = [20, 50]
        std_multipliers = [2, 2.5]
        
        for period in periods:
            if len(data) >= period:
                sma = data['close'].rolling(window=period).mean()
                std = data['close'].rolling(window=period).std()
                
                for std_mult in std_multipliers:
                    upper_band = sma + (std * std_mult)
                    lower_band = sma - (std * std_mult)
                    
                    features[f'bb_upper_{period}_{std_mult}'] = upper_band
                    features[f'bb_lower_{period}_{std_mult}'] = lower_band
                    features[f'bb_width_{period}_{std_mult}'] = upper_band - lower_band
                    features[f'bb_position_{period}_{std_mult}'] = (data['close'] - lower_band) / (upper_band - lower_band)
        
        return features.dropna()
    
    def _extract_rolling_stats_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract rolling statistical features."""
        features = pd.DataFrame(index=data.index)
        
        periods = [5, 10, 20, 50]
        
        for period in periods:
            if len(data) >= period:
                rolling = data['close'].rolling(window=period)
                
                features[f'rolling_mean_{period}'] = rolling.mean()
                features[f'rolling_std_{period}'] = rolling.std()
                features[f'rolling_min_{period}'] = rolling.min()
                features[f'rolling_max_{period}'] = rolling.max()
                features[f'rolling_skew_{period}'] = rolling.skew()
                features[f'rolling_kurt_{period}'] = rolling.kurt()
        
        return features.dropna()
    
    def _extract_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract volatility features."""
        features = pd.DataFrame(index=data.index)
        
        periods = [5, 10, 20, 50]
        
        for period in periods:
            if len(data) >= period:
                returns = data['close'].pct_change()
                
                features[f'volatility_{period}'] = returns.rolling(window=period).std()
                features[f'volatility_annualized_{period}'] = features[f'volatility_{period}'] * np.sqrt(252)
                features[f'volatility_ratio_{period}'] = features[f'volatility_{period}'] / features[f'volatility_{period}'].rolling(window=period*2).mean()
        
        return features.dropna()
    
    def _extract_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract momentum features."""
        features = pd.DataFrame(index=data.index)
        
        periods = [5, 10, 20, 50]
        
        for period in periods:
            if len(data) >= period:
                features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
                features[f'roc_{period}'] = data['close'].pct_change(periods=period)
                features[f'price_velocity_{period}'] = data['close'].diff(period) / period
        
        return features.dropna()
    
    def _extract_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features."""
        features = pd.DataFrame(index=data.index)
        
        if isinstance(data.index, pd.DatetimeIndex):
            features['hour'] = data.index.hour
            features['day_of_week'] = data.index.dayofweek
            features['day_of_month'] = data.index.day
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            features['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
            features['is_month_start'] = (data.index.day <= 3).astype(int)
            features['is_month_end'] = (data.index.day >= 28).astype(int)
        
        return features
    
    def _extract_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract lag features."""
        features = pd.DataFrame(index=data.index)
        
        lags = [1, 2, 3, 5, 10, 20]
        
        for lag in lags:
            if len(data) >= lag:
                features[f'close_lag_{lag}'] = data['close'].shift(lag)
                features[f'volume_lag_{lag}'] = data['volume'].shift(lag)
                features[f'return_lag_{lag}'] = data['close'].pct_change().shift(lag)
        
        return features.dropna()
    
    def _extract_seasonality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract seasonality features."""
        features = pd.DataFrame(index=data.index)
        
        if isinstance(data.index, pd.DatetimeIndex):
            # Fourier features for seasonality
            for period in [24, 168, 720]:  # Daily, weekly, monthly
                if len(data) >= period * 2:
                    t = np.arange(len(data))
                    features[f'seasonal_sin_{period}'] = np.sin(2 * np.pi * t / period)
                    features[f'seasonal_cos_{period}'] = np.cos(2 * np.pi * t / period)
        
        return features.dropna()
    
    def _calculate_feature_importance(self, 
                                    features: pd.DataFrame, 
                                    target: pd.Series) -> Dict[str, float]:
        """Calculate feature importance using correlation."""
        importance = {}
        
        for column in features.columns:
            try:
                # Calculate correlation with target
                corr = safe_correlation(features[column].values, target.values)
                importance[column] = abs(corr) if not np.isnan(corr) else 0.0
            except Exception as e:
                self.logger.warning(f"Could not calculate importance for {column}: {e}")
                importance[column] = 0.0
        
        return importance
    
    def _combine_features(self, extracted_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine all extracted features."""
        if not extracted_features:
            return pd.DataFrame()
        
        # Start with the first feature set
        combined = list(extracted_features.values())[0]
        
        # Add remaining feature sets
        for feature_type, features in list(extracted_features.items())[1:]:
            if not features.empty:
                combined = safe_merge_dataframes(
                    combined, features, 
                    left_index=True, right_index=True, how='outer'
                )
        
        return combined.fillna(0)
    
    def _apply_nas_feature_selection(self, 
                                   features: pd.DataFrame,
                                   target: Optional[pd.Series],
                                   max_features: int) -> pd.DataFrame:
        """Apply Neural Architecture Search for feature selection."""
        try:
            if target is None:
                # Use variance-based selection
                feature_variance = features.var()
                selected_features = feature_variance.nlargest(max_features).index
            else:
                # Use correlation-based selection
                correlations = features.corrwith(target).abs()
                selected_features = correlations.nlargest(max_features).index
            
            selected_df = features[selected_features]
            
            tprint_info(f"🧠 NAS selected {len(selected_features)} features from {len(features.columns)}")
            return selected_df
            
        except Exception as e:
            tprint_warning(f"⚠️ NAS feature selection failed: {e}")
            # Return top features by variance as fallback
            feature_variance = features.var()
            selected_features = feature_variance.nlargest(max_features).index
            return features[selected_features]
    
    def save_features(self, 
                     filepath: str,
                     format: str = 'auto',
                     include_metadata: bool = True) -> bool:
        """Save extracted features to file."""
        try:
            if not self.extracted_features:
                tprint_warning("⚠️ No features to save")
                return False
            
            # Prepare data for saving
            save_data = {
                'features': self.extracted_features,
                'feature_importance': self.feature_importance,
                'performance_metrics': self.performance_metrics,
                'optimization_stats': self.optimization_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save using universal serializer
            success = self.universal_serializer.save(save_data, filepath, format)
            
            if success:
                tprint_success(f"💾 Features saved to {filepath}")
            else:
                tprint_error(f"❌ Failed to save features to {filepath}")
            
            return success
            
        except Exception as e:
            tprint_error(f"❌ Save features failed: {e}")
            return False
    
    def load_features(self, filepath: str) -> bool:
        """Load extracted features from file."""
        try:
            data = self.universal_serializer.load(filepath)
            
            if data is None:
                tprint_error(f"❌ Failed to load features from {filepath}")
                return False
            
            # Restore state
            self.extracted_features = data.get('features', {})
            self.feature_importance = data.get('feature_importance', {})
            self.performance_metrics = data.get('performance_metrics', {})
            self.optimization_stats = data.get('optimization_stats', {})
            
            tprint_success(f"📂 Features loaded from {filepath}")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Load features failed: {e}")
            return False
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of extracted features."""
        summary = {
            'total_feature_types': len(self.extracted_features),
            'feature_types': list(self.extracted_features.keys()),
            'total_features': sum(len(df.columns) for df in self.extracted_features.values()),
            'performance_metrics': self.performance_metrics,
            'optimization_stats': self.optimization_stats
        }
        
        # Add feature importance summary
        if self.feature_importance:
            summary['feature_importance'] = {
                feature_type: {
                    'top_features': sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                }
                for feature_type, importance in self.feature_importance.items()
            }
        
        return summary
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.enable_m1_optimization and hasattr(self, 'memory_optimizer'):
                self.memory_optimizer.stop_monitoring()
            
            tprint_info("🧹 NASFeatureExtractor cleanup completed")
            
        except Exception as e:
            tprint_warning(f"⚠️ Cleanup failed: {e}")


# Convenience functions
def create_nas_extractor(config: Optional[Dict[str, Any]] = None,
                        enable_m1_optimization: bool = True,
                        memory_limit_gb: Optional[float] = None) -> NASFeatureExtractor:
    """Create a NASFeatureExtractor instance."""
    return NASFeatureExtractor(config, enable_m1_optimization, memory_limit_gb)


def extract_features_with_nas(data: pd.DataFrame,
                            target_column: Optional[str] = None,
                            feature_types: List[str] = None,
                            max_features: int = 100,
                            enable_architectural_search: bool = True,
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract features using NAS in one function call."""
    extractor = create_nas_extractor(config)
    
    try:
        result = extractor.extract_features(
            data, target_column, feature_types, max_features, enable_architectural_search
        )
        return result
    finally:
        extractor.cleanup()


if __name__ == "__main__":
    # Example usage
    tprint_info("🧠 NASFeatureExtractor Example")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    sample_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Ensure high >= low and proper OHLC relationships
    sample_data['high'] = np.maximum(sample_data['high'], sample_data[['open', 'close']].max(axis=1))
    sample_data['low'] = np.minimum(sample_data['low'], sample_data[['open', 'close']].min(axis=1))
    
    # Create extractor
    extractor = create_nas_extractor()
    
    # Extract features
    result = extractor.extract_features(
        sample_data,
        target_column=None,
        feature_types=['technical', 'statistical'],
        max_features=50
    )
    
    if result['success']:
        tprint_success("✅ Feature extraction completed successfully")
        tprint_info(f"📊 Extracted {len(result['features'].columns)} features")
        
        # Get summary
        summary = extractor.get_feature_summary()
        tprint_structured(summary)
        
        # Save features
        extractor.save_features('nas_features.pkl')
    
    # Cleanup
    extractor.cleanup()