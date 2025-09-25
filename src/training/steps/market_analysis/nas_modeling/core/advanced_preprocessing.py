"""
Advanced Preprocessing Module

This module provides comprehensive preprocessing capabilities for financial data,
integrating with the existing utility infrastructure for optimal performance on Apple Silicon.

Features:
- Advanced feature engineering with regime-aware processing
- M1/M2/M3 hardware optimization
- Memory-efficient data processing
- Cross-validation and lookahead bias prevention
- Hyperparameter optimization integration
- Matrix operations and mathematical validation
- Comprehensive logging and monitoring
"""

import logging
import time
import gc
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from contextlib import contextmanager
import warnings

# Core data processing
import pandas as pd
import numpy as np

# Import utility modules
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, ensure_directory, safe_file_exists,
    validate_dataframe, validate_dataframe_columns, optimize_dataframe_dtypes,
    calculate_data_quality_metrics, create_data_quality_report,
    safe_divide, safe_log, safe_sqrt, safe_mean, safe_std,
    memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
    integrate_with_m1_optimizers, cleanup_m1_optimizers
)

from src.utils.common_utilities import (
    safe_dataframe_operation, safe_convert_dtypes, safe_merge_dataframes,
    safe_drop_columns, safe_rename_columns, validate_timestamp_column,
    safe_timestamp_conversion, get_dataframe_info, create_summary_statistics
)

from src.utils.math_validation import (
    validate_finite, validate_positive, validate_range, validate_numeric_array,
    safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
    safe_correlation, safe_covariance, safe_percentile, validate_correlation_matrix,
    safe_matrix_inverse, math_safe
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_progress, tprint_performance, tprint_timer, LogLevel
)

# Import specialized utilities
try:
    from src.utils.data.klines_parquet import KlinesParquetManager, validate_klines_data
    KLINES_AVAILABLE = True
except ImportError:
    KLINES_AVAILABLE = False
    tprint_warning("Klines parquet utilities not available")

try:
    from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    tprint_warning("Matrix operations not available")

try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    tprint_warning("M1 hardware optimizations not available")

try:
    from src.utils.ml_common.validation.cv_utils import CrossValidationUtils
    from src.utils.ml_common.validation.temporal_validation import TemporalValidation
    ML_VALIDATION_AVAILABLE = True
except ImportError:
    ML_VALIDATION_AVAILABLE = False
    tprint_warning("ML validation utilities not available")

try:
    from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimizer
    from src.utils.ml_common.optimization.grid_utils import GridSearchOptimizer
    ML_OPTIMIZATION_AVAILABLE = True
except ImportError:
    ML_OPTIMIZATION_AVAILABLE = False
    tprint_warning("ML optimization utilities not available")

# Setup logging
logger = logging.getLogger(__name__)


class AdvancedPreprocessor:
    """
    Advanced Preprocessor for financial data with comprehensive feature engineering,
    hardware optimization, and ML integration capabilities.
    
    This class provides a unified interface for advanced data preprocessing operations
    optimized for Apple Silicon hardware with extensive validation and monitoring.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 enable_hardware_optimization: bool = True,
                 enable_validation: bool = True,
                 enable_monitoring: bool = True,
                 cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the Advanced Preprocessor.
        
        Args:
            config: Configuration dictionary for preprocessing options
            enable_hardware_optimization: Enable M1/M2/M3 hardware optimizations
            enable_validation: Enable data validation and quality checks
            enable_monitoring: Enable performance monitoring and logging
            cache_dir: Directory for caching preprocessed data
        """
        self.config = config or {}
        self.enable_hardware_optimization = enable_hardware_optimization
        self.enable_validation = enable_validation
        self.enable_monitoring = enable_monitoring
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            ensure_directory(self.cache_dir)
        else:
            self.cache_dir = Path("data_cache/preprocessed")
            ensure_directory(self.cache_dir)
        
        # Initialize hardware optimizations
        self._setup_hardware_optimizations()
        
        # Initialize utility components
        self._setup_utilities()
        
        # Initialize serialization
        self.serializer = UniversalSerializer()
        
        # Performance tracking
        self.performance_metrics = {}
        self.operation_history = []
        
        tprint_success("AdvancedPreprocessor initialized successfully")
    
    def _setup_hardware_optimizations(self):
        """Setup hardware optimizations for M1/M2/M3."""
        if not self.enable_hardware_optimization or not HARDWARE_AVAILABLE:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            return
        
        try:
            self.gpu_manager = M1GPUManager()
            self.memory_optimizer = M1MemoryOptimizer()
            self.cpu_optimizer = M1CPUOptimizer()
            
            # Start memory monitoring
            if self.memory_optimizer:
                self.memory_optimizer.start_monitoring()
            
            tprint_info("Hardware optimizations initialized")
        except Exception as e:
            tprint_warning(f"Failed to initialize hardware optimizations: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _setup_utilities(self):
        """Setup utility components."""
        # Matrix operations
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = UnifiedMatrixOperations(
                enable_gpu=self.enable_hardware_optimization,
                enable_memory_optimization=self.enable_hardware_optimization,
                enable_parallel=True
            )
        else:
            self.matrix_ops = None
        
        # Klines data manager
        if KLINES_AVAILABLE:
            self.klines_manager = KlinesParquetManager()
        else:
            self.klines_manager = None
        
        # ML validation
        if ML_VALIDATION_AVAILABLE:
            self.cv_utils = CrossValidationUtils()
            self.temporal_validator = TemporalValidation()
        else:
            self.cv_utils = None
            self.temporal_validator = None
        
        # ML optimization
        if ML_OPTIMIZATION_AVAILABLE:
            self.hpo_optimizer = HyperparameterOptimizer()
            self.grid_optimizer = GridSearchOptimizer()
        else:
            self.hpo_optimizer = None
            self.grid_optimizer = None
    
    @contextmanager
    def _performance_timer(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.perf_counter()
        start_memory = get_memory_usage() if self.enable_monitoring else 0
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = get_memory_usage() if self.enable_monitoring else 0
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            if operation_name not in self.performance_metrics:
                self.performance_metrics[operation_name] = {
                    'total_time': 0.0,
                    'call_count': 0,
                    'avg_time': 0.0,
                    'total_memory_delta': 0.0,
                    'avg_memory_delta': 0.0
                }
            
            metrics = self.performance_metrics[operation_name]
            metrics['total_time'] += duration
            metrics['call_count'] += 1
            metrics['avg_time'] = metrics['total_time'] / metrics['call_count']
            metrics['total_memory_delta'] += memory_delta
            metrics['avg_memory_delta'] = metrics['total_memory_delta'] / metrics['call_count']
            
            self.operation_history.append({
                'operation': operation_name,
                'duration': duration,
                'memory_delta': memory_delta,
                'timestamp': time.time()
            })
            
            if self.enable_monitoring:
                tprint_performance(f"{operation_name}", duration)
    
    def validate_input_data(self, df: pd.DataFrame, 
                          required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate input data for preprocessing.
        
        Args:
            df: Input DataFrame
            required_columns: List of required columns
            
        Returns:
            Validation results dictionary
        """
        with self._performance_timer("validate_input_data"):
            if not self.enable_validation:
                return {'valid': True, 'warnings': [], 'errors': []}
            
            results = {
                'valid': True,
                'warnings': [],
                'errors': [],
                'quality_metrics': {},
                'data_info': {}
            }
            
            # Basic validation
            if not validate_dataframe(df):
                results['valid'] = False
                results['errors'].append("Invalid DataFrame")
                return results
            
            # Required columns validation
            if required_columns:
                if not validate_dataframe_columns(df, required_columns):
                    results['valid'] = False
                    results['errors'].append(f"Missing required columns: {required_columns}")
            
            # Data quality metrics
            try:
                results['quality_metrics'] = calculate_data_quality_metrics(df)
                results['data_info'] = get_dataframe_info(df)
                
                # Check for critical issues
                if results['quality_metrics'].get('missing_percentage', 0) > 50:
                    results['warnings'].append("High percentage of missing values")
                
                if results['quality_metrics'].get('duplicate_percentage', 0) > 10:
                    results['warnings'].append("High percentage of duplicate rows")
                    
            except Exception as e:
                results['warnings'].append(f"Could not calculate quality metrics: {e}")
            
            # Klines-specific validation
            if KLINES_AVAILABLE and 'close' in df.columns:
                try:
                    klines_validation = validate_klines_data(df)
                    if not klines_validation['valid']:
                        results['errors'].extend(klines_validation['errors'])
                    results['warnings'].extend(klines_validation['warnings'])
                except Exception as e:
                    results['warnings'].append(f"Klines validation failed: {e}")
            
            return results
    
    def clean_data(self, df: pd.DataFrame, 
                   cleaning_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Clean and prepare data for preprocessing.
        
        Args:
            df: Input DataFrame
            cleaning_config: Configuration for cleaning operations
            
        Returns:
            Cleaned DataFrame
        """
        with self._performance_timer("clean_data"):
            config = cleaning_config or {}
            
            # Create a copy to avoid modifying original
            cleaned_df = df.copy()
            
            # Remove duplicates
            if config.get('remove_duplicates', True):
                initial_rows = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                removed_duplicates = initial_rows - len(cleaned_df)
                if removed_duplicates > 0:
                    tprint_info(f"Removed {removed_duplicates} duplicate rows")
            
            # Handle missing values
            missing_strategy = config.get('missing_strategy', 'interpolate')
            if missing_strategy == 'drop':
                cleaned_df = cleaned_df.dropna()
            elif missing_strategy == 'interpolate':
                # Use safe interpolation for numeric columns
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if cleaned_df[col].isnull().any():
                        cleaned_df[col] = cleaned_df[col].interpolate(method='linear')
            elif missing_strategy == 'fill':
                fill_value = config.get('fill_value', 0)
                cleaned_df = cleaned_df.fillna(fill_value)
            
            # Optimize data types
            if config.get('optimize_dtypes', True):
                cleaned_df = optimize_dataframe_dtypes(cleaned_df)
            
            # Remove outliers if configured
            outlier_threshold = config.get('outlier_threshold', None)
            if outlier_threshold:
                cleaned_df = self._remove_outliers(cleaned_df, outlier_threshold)
            
            return cleaned_df
    
    def _remove_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
        
        return df
    
    def engineer_features(self, df: pd.DataFrame,
                         feature_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Engineer advanced features for financial data.
        
        Args:
            df: Input DataFrame
            feature_config: Configuration for feature engineering
            
        Returns:
            DataFrame with engineered features
        """
        with self._performance_timer("engineer_features"):
            config = feature_config or {}
            engineered_df = df.copy()
            
            # Price-based features
            if config.get('price_features', True):
                engineered_df = self._add_price_features(engineered_df)
            
            # Technical indicators
            if config.get('technical_indicators', True):
                engineered_df = self._add_technical_indicators(engineered_df)
            
            # Volatility features
            if config.get('volatility_features', True):
                engineered_df = self._add_volatility_features(engineered_df)
            
            # Volume features
            if config.get('volume_features', True) and 'volume' in engineered_df.columns:
                engineered_df = self._add_volume_features(engineered_df)
            
            # Time-based features
            if config.get('time_features', True):
                engineered_df = self._add_time_features(engineered_df)
            
            # Regime-based features
            if config.get('regime_features', False):
                engineered_df = self._add_regime_features(engineered_df)
            
            return engineered_df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return df
        
        # Price ratios and spreads
        df['price_range'] = df['high'] - df['low']
        df['price_spread'] = (df['high'] - df['low']) / df['close']
        df['close_open_ratio'] = df['close'] / df['open']
        df['high_low_ratio'] = df['high'] / df['low']
        
        # Price momentum
        for window in [5, 10, 20]:
            df[f'price_change_{window}'] = df['close'].pct_change(window)
            df[f'price_momentum_{window}'] = df['close'] / df['close'].shift(window)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        if 'close' not in df.columns:
            return df
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # Bollinger Bands
        window = 20
        sma = df['close'].rolling(window=window).mean()
        std = df['close'].rolling(window=window).std()
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        if 'close' not in df.columns:
            return df
        
        # Rolling volatility
        for window in [5, 10, 20]:
            returns = df['close'].pct_change()
            df[f'volatility_{window}'] = returns.rolling(window=window).std()
        
        # GARCH-like features
        returns = df['close'].pct_change()
        df['volatility_clustering'] = returns.rolling(window=10).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0
        )
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        if 'volume' not in df.columns:
            return df
        
        # Volume moving averages
        for window in [5, 10, 20]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
        
        # Volume-price features
        if 'close' in df.columns:
            df['volume_price'] = df['volume'] * df['close']
            df['volume_weighted_price'] = df['volume_price'] / df['volume']
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # Time components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime-based features (placeholder for advanced regime detection)."""
        # This would integrate with regime detection algorithms
        # For now, add simple trend features
        if 'close' not in df.columns:
            return df
        
        # Trend features
        for window in [20, 50]:
            df[f'trend_{window}'] = df['close'].rolling(window=window).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1
            )
        
        return df
    
    def scale_features(self, df: pd.DataFrame,
                      scaling_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Scale features for machine learning.
        
        Args:
            df: Input DataFrame
            scaling_config: Configuration for scaling operations
            
        Returns:
            DataFrame with scaled features
        """
        with self._performance_timer("scale_features"):
            config = scaling_config or {}
            scaling_method = config.get('method', 'standard')
            
            scaled_df = df.copy()
            
            # Select numeric columns for scaling
            numeric_cols = scaled_df.select_dtypes(include=[np.number]).columns
            
            if scaling_method == 'standard':
                # Z-score normalization
                for col in numeric_cols:
                    if scaled_df[col].std() > 0:
                        scaled_df[col] = (scaled_df[col] - scaled_df[col].mean()) / scaled_df[col].std()
            
            elif scaling_method == 'minmax':
                # Min-max scaling
                for col in numeric_cols:
                    col_min = scaled_df[col].min()
                    col_max = scaled_df[col].max()
                    if col_max > col_min:
                        scaled_df[col] = (scaled_df[col] - col_min) / (col_max - col_min)
            
            elif scaling_method == 'robust':
                # Robust scaling using median and IQR
                for col in numeric_cols:
                    col_median = scaled_df[col].median()
                    col_iqr = scaled_df[col].quantile(0.75) - scaled_df[col].quantile(0.25)
                    if col_iqr > 0:
                        scaled_df[col] = (scaled_df[col] - col_median) / col_iqr
            
            return scaled_df
    
    def select_features(self, df: pd.DataFrame,
                       target_column: Optional[str] = None,
                       selection_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Select relevant features for modeling.
        
        Args:
            df: Input DataFrame
            target_column: Target column for supervised feature selection
            selection_config: Configuration for feature selection
            
        Returns:
            DataFrame with selected features
        """
        with self._performance_timer("select_features"):
            config = selection_config or {}
            selection_method = config.get('method', 'correlation')
            
            selected_df = df.copy()
            
            if target_column and target_column in df.columns:
                # Supervised feature selection
                if selection_method == 'correlation':
                    selected_df = self._correlation_feature_selection(selected_df, target_column, config)
                elif selection_method == 'mutual_info':
                    selected_df = self._mutual_info_feature_selection(selected_df, target_column, config)
            else:
                # Unsupervised feature selection
                if selection_method == 'variance':
                    selected_df = self._variance_feature_selection(selected_df, config)
                elif selection_method == 'correlation_matrix':
                    selected_df = self._correlation_matrix_feature_selection(selected_df, config)
            
            return selected_df
    
    def _correlation_feature_selection(self, df: pd.DataFrame, target_column: str, config: Dict) -> pd.DataFrame:
        """Select features based on correlation with target."""
        threshold = config.get('correlation_threshold', 0.1)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = {}
        
        for col in numeric_cols:
            if col != target_column:
                try:
                    corr = safe_correlation(df[col].values, df[target_column].values)
                    correlations[col] = abs(corr)
                except Exception:
                    continue
        
        # Select features above threshold
        selected_features = [col for col, corr in correlations.items() if corr > threshold]
        selected_features.append(target_column)
        
        return df[selected_features]
    
    def _mutual_info_feature_selection(self, df: pd.DataFrame, target_column: str, config: Dict) -> pd.DataFrame:
        """Select features based on mutual information (placeholder)."""
        # This would use sklearn's mutual information functions
        # For now, return correlation-based selection
        return self._correlation_feature_selection(df, target_column, config)
    
    def _variance_feature_selection(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Select features based on variance threshold."""
        threshold = config.get('variance_threshold', 0.01)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_features = []
        
        for col in numeric_cols:
            if df[col].var() > threshold:
                selected_features.append(col)
        
        return df[selected_features]
    
    def _correlation_matrix_feature_selection(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Remove highly correlated features."""
        threshold = config.get('correlation_threshold', 0.95)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            if feat1 not in features_to_remove:
                features_to_remove.add(feat2)
        
        selected_features = [col for col in df.columns if col not in features_to_remove]
        return df[selected_features]
    
    def create_train_test_split(self, df: pd.DataFrame,
                               test_size: float = 0.2,
                               validation_size: float = 0.1,
                               temporal_split: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for test set
            validation_size: Proportion of data for validation set
            temporal_split: Whether to use temporal splitting
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        with self._performance_timer("create_train_test_split"):
            if temporal_split and isinstance(df.index, pd.DatetimeIndex):
                # Temporal split preserves time order
                total_size = len(df)
                test_start = int(total_size * (1 - test_size))
                val_start = int(total_size * (1 - test_size - validation_size))
                
                train_df = df.iloc[:val_start].copy()
                val_df = df.iloc[val_start:test_start].copy()
                test_df = df.iloc[test_start:].copy()
            else:
                # Random split
                from sklearn.model_selection import train_test_split
                
                train_val_df, test_df = train_test_split(
                    df, test_size=test_size, random_state=42
                )
                
                train_df, val_df = train_test_split(
                    train_val_df, test_size=validation_size/(1-test_size), random_state=42
                )
            
            tprint_info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            return train_df, val_df, test_df
    
    def optimize_hyperparameters(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                target_column: str,
                                optimization_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters for preprocessing.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            target_column: Target column name
            optimization_config: Configuration for optimization
            
        Returns:
            Optimized hyperparameters
        """
        with self._performance_timer("optimize_hyperparameters"):
            if not ML_OPTIMIZATION_AVAILABLE:
                tprint_warning("ML optimization not available, returning default config")
                return self.config
            
            config = optimization_config or {}
            
            # Define parameter space for preprocessing
            param_space = {
                'feature_selection_threshold': [0.05, 0.1, 0.15, 0.2],
                'scaling_method': ['standard', 'minmax', 'robust'],
                'outlier_threshold': [2.0, 2.5, 3.0],
                'volatility_window': [5, 10, 20],
                'technical_window': [10, 20, 30]
            }
            
            # Use grid search or Bayesian optimization
            optimization_method = config.get('method', 'grid')
            
            if optimization_method == 'grid' and self.grid_optimizer:
                best_params = self.grid_optimizer.optimize(
                    train_df, val_df, target_column, param_space
                )
            elif optimization_method == 'bayesian' and self.hpo_optimizer:
                best_params = self.hpo_optimizer.optimize(
                    train_df, val_df, target_column, param_space
                )
            else:
                # Default parameters
                best_params = {
                    'feature_selection_threshold': 0.1,
                    'scaling_method': 'standard',
                    'outlier_threshold': 3.0,
                    'volatility_window': 20,
                    'technical_window': 20
                }
            
            tprint_info(f"Optimized hyperparameters: {best_params}")
            return best_params
    
    def preprocess_pipeline(self, df: pd.DataFrame,
                          pipeline_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            pipeline_config: Configuration for the preprocessing pipeline
            
        Returns:
            Dictionary containing preprocessed data and metadata
        """
        with self._performance_timer("preprocess_pipeline"):
            config = pipeline_config or {}
            
            # Validate input
            validation_results = self.validate_input_data(df, config.get('required_columns'))
            if not validation_results['valid']:
                tprint_error(f"Input validation failed: {validation_results['errors']}")
                return {'success': False, 'errors': validation_results['errors']}
            
            # Clean data
            cleaned_df = self.clean_data(df, config.get('cleaning', {}))
            
            # Engineer features
            engineered_df = self.engineer_features(cleaned_df, config.get('features', {}))
            
            # Scale features
            scaled_df = self.scale_features(engineered_df, config.get('scaling', {}))
            
            # Select features
            selected_df = self.select_features(
                scaled_df, 
                config.get('target_column'),
                config.get('selection', {})
            )
            
            # Create splits if target column specified
            splits = None
            if config.get('target_column') and config.get('create_splits', True):
                splits = self.create_train_test_split(
                    selected_df,
                    config.get('test_size', 0.2),
                    config.get('validation_size', 0.1),
                    config.get('temporal_split', True)
                )
            
            # Cache results if configured
            if config.get('cache_results', False):
                cache_key = self._generate_cache_key(df, config)
                self._cache_preprocessed_data(selected_df, cache_key)
            
            result = {
                'success': True,
                'preprocessed_data': selected_df,
                'splits': splits,
                'validation_results': validation_results,
                'performance_metrics': self.performance_metrics.copy(),
                'config_used': config
            }
            
            tprint_success("Preprocessing pipeline completed successfully")
            return result
    
    def _generate_cache_key(self, df: pd.DataFrame, config: Dict) -> str:
        """Generate cache key for preprocessed data."""
        import hashlib
        
        # Create hash from DataFrame info and config
        data_hash = hashlib.md5(
            f"{df.shape}{list(df.columns)}{str(config)}".encode()
        ).hexdigest()
        
        return f"preprocessed_{data_hash}"
    
    def _cache_preprocessed_data(self, df: pd.DataFrame, cache_key: str):
        """Cache preprocessed data."""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        try:
            df.to_parquet(cache_file)
            tprint_info(f"Cached preprocessed data to {cache_file}")
        except Exception as e:
            tprint_warning(f"Failed to cache data: {e}")
    
    def load_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load cached preprocessed data."""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception as e:
                tprint_warning(f"Failed to load cached data: {e}")
        return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all operations."""
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'operation_history': self.operation_history.copy(),
            'total_operations': len(self.operation_history),
            'average_operation_time': np.mean([op['duration'] for op in self.operation_history]) if self.operation_history else 0
        }
    
    def cleanup(self):
        """Cleanup resources and stop monitoring."""
        if self.memory_optimizer:
            self.memory_optimizer.stop_monitoring()
        
        cleanup_m1_optimizers()
        tprint_info("AdvancedPreprocessor cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Convenience functions
def create_preprocessor(config: Optional[Dict[str, Any]] = None, **kwargs) -> AdvancedPreprocessor:
    """
    Create an AdvancedPreprocessor instance with default configuration.
    
    Args:
        config: Configuration dictionary
        **kwargs: Additional keyword arguments
        
    Returns:
        AdvancedPreprocessor instance
    """
    return AdvancedPreprocessor(config=config, **kwargs)


def preprocess_financial_data(df: pd.DataFrame, 
                            config: Optional[Dict[str, Any]] = None,
                            **kwargs) -> Dict[str, Any]:
    """
    Convenience function for preprocessing financial data.
    
    Args:
        df: Input DataFrame
        config: Preprocessing configuration
        **kwargs: Additional keyword arguments
        
    Returns:
        Preprocessing results
    """
    with AdvancedPreprocessor(config=config, **kwargs) as preprocessor:
        return preprocessor.preprocess_pipeline(df, config)


# Export main class and functions
__all__ = [
    'AdvancedPreprocessor',
    'create_preprocessor',
    'preprocess_financial_data'
]
