"""
NAS/TAS Data Manager

This module provides comprehensive data management for Neural Architecture Search
and Trading Architecture Search with extensive integration of utility modules
for optimal performance, data processing, and hardware optimization.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

# Extensive use of common utilities
from ...common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, optimize_dataframe_dtypes,
    safe_to_parquet, safe_read_parquet, integrate_with_m1_optimizers,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    cleanup_m1_optimizers, memory_checkpoint, gpu_context, optimize_memory,
    get_memory_usage, safe_copy, safe_deepcopy, safe_resample, align_dataframes,
    validate_dataframe_schema, guard_dataframe_nulls, timed_operation,
    format_bytes, parallel_map, chunked_iterable
)

from ...common_utilities import (
    CommonUtilities, safe_dataframe_operation as cu_safe_dataframe_operation,
    validate_dataframe_columns as cu_validate_dataframe_columns,
    calculate_data_quality_metrics as cu_calculate_data_quality_metrics,
    safe_merge_dataframes as cu_safe_merge_dataframes,
    safe_groupby_operation as cu_safe_groupby_operation,
    safe_apply_function as cu_safe_apply_function,
    create_summary_statistics as cu_create_summary_statistics,
    safe_drop_columns as cu_safe_drop_columns,
    safe_rename_columns as cu_safe_rename_columns,
    validate_timestamp_column as cu_validate_timestamp_column,
    safe_timestamp_conversion as cu_safe_timestamp_conversion,
    get_dataframe_info as cu_get_dataframe_info,
    safe_filter_dataframe as cu_safe_filter_dataframe,
    create_data_quality_report as cu_create_data_quality_report
)

from ...math_validation import (
    MathValidation, safe_divide as mv_safe_divide, safe_log as mv_safe_log,
    safe_sqrt as mv_safe_sqrt, safe_power as mv_safe_power,
    validate_finite as mv_validate_finite, validate_positive as mv_validate_positive,
    validate_range as mv_validate_range, safe_kelly_calculation as mv_safe_kelly_calculation,
    safe_weighted_average as mv_safe_weighted_average, safe_percentage_change as mv_safe_percentage_change,
    safe_correlation, safe_covariance, safe_mean as mv_safe_mean, safe_std as mv_safe_std,
    safe_percentile, validate_correlation_matrix, safe_matrix_inverse, math_safe,
    validate_numeric_array
)

from ...tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_with_level, tprint_timer, tprint_logged, configure_tprint,
    get_tprint_config, tprint_context, LogLevel
)

from ...data.klines_parquet import (
    KlinesParquetManager, get_klines_manager, read_ethusdt_data,
    save_klines_to_parquet, load_klines_from_parquet, validate_klines_data
)

from ...serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

# Import data processing utilities
from ...data.processing.data_processing import DataProcessor
from ...data.basic_returns_engineer import BasicReturnsEngineer
from ...data.feature_engineer import FeatureEngineer
from ...data.gap_detector import GapDetector
from ...data.unified_data_utils import UnifiedDataUtils

# Import matrix operations
from ...matrix_operations.unified_operations import MatrixOperations
from ...matrix_operations.enhanced_operations import EnhancedMatrixOperations
from ...matrix_operations.batch_operations import BatchMatrixOperations
from ...matrix_operations.vectorized_core import VectorizedCore
from ...matrix_operations.convenience import MatrixConvenience

# Import hardware utilities
from ...hardware.m1_gpu_utils import M1GPUManager, is_m1_available, is_mps_available
from ...hardware.m1_memory_optimizer import M1MemoryOptimizer
from ...hardware.m1_cpu_optimizer import M1CPUOptimizer

# Setup logging with tprint integration
logger = logging.getLogger(__name__)

@tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
class NASDataManager:
    """
    NAS/TAS Data Manager with extensive utility integration.
    
    This data manager provides comprehensive data management capabilities with:
    - Extensive use of common operations for data processing
    - Math validation for safe computations
    - Comprehensive logging with tprint
    - Data management with klines parquet utilities
    - Serialization for data persistence
    - M1 hardware optimization
    - Matrix operations for high-performance computations
    - Data processing pipeline integration
    - Feature engineering and data quality assurance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the NAS/TAS Data Manager with extensive utility integration.
        
        Args:
            config: Configuration dictionary for data manager
        """
        tprint_info("🚀 Initializing NAS/TAS Data Manager with extensive utility integration")
        
        # Initialize configuration
        self.config = config or {}
        self.logger = logger.getChild("NASDataManager")
        
        # Initialize utility classes
        tprint_debug("🔧 Initializing utility classes")
        self.common_ops = CommonUtilities()
        self.math_validator = MathValidation()
        self.klines_manager = get_klines_manager()
        self.serializer = UniversalSerializer()
        
        # Initialize data processing utilities
        tprint_debug("🔧 Initializing data processing utilities")
        self.data_processor = DataProcessor()
        self.returns_engineer = BasicReturnsEngineer()
        self.feature_engineer = FeatureEngineer()
        self.gap_detector = GapDetector()
        self.unified_data_utils = UnifiedDataUtils()
        
        # Initialize matrix operations
        tprint_debug("🔧 Initializing matrix operations")
        self.matrix_ops = MatrixOperations()
        self.enhanced_matrix_ops = EnhancedMatrixOperations()
        self.batch_matrix_ops = BatchMatrixOperations()
        self.vectorized_core = VectorizedCore()
        self.matrix_convenience = MatrixConvenience()
        
        # Initialize M1 hardware optimizations
        tprint_debug("🔧 Initializing M1 hardware optimizations")
        self.m1_integration = integrate_with_m1_optimizers()
        if self.m1_integration['success']:
            tprint_success("✅ M1 integration successful")
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
        else:
            tprint_warning("⚠️ M1 integration failed, using fallback")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
        
        # Initialize data storage
        self.data_cache = {}
        self.quality_metrics = {}
        self.processing_history = []
        
        tprint_success("✅ NAS/TAS Data Manager initialized successfully")
    
    @tprint_timer("Data Loading")
    def load_data(
        self,
        symbol: str = "ETHUSDT",
        interval: str = "1m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        data_type: str = "processed"
    ) -> Optional[pd.DataFrame]:
        """Load data using extensive utility integration.
        
        Args:
            symbol: Trading symbol to load
            interval: Data interval
            start_date: Start date for data loading
            end_date: End date for data loading
            data_type: Type of data to load (raw, processed, features)
            
        Returns:
            Loaded DataFrame or None if loading fails
        """
        tprint_info(f"📊 Loading {data_type} data for {symbol} {interval}")
        
        try:
            # Check cache first
            cache_key = f"{symbol}_{interval}_{start_date}_{end_date}_{data_type}"
            if cache_key in self.data_cache:
                tprint_info("📋 Using cached data")
                return self.data_cache[cache_key]
            
            # Load data using klines parquet manager
            with memory_checkpoint("data_loading"):
                data = self.klines_manager.read_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    data_type=data_type
                )
            
            if data is None or data.empty:
                tprint_error(f"❌ No data loaded for {symbol} {interval}")
                return None
            
            tprint_info(f"📊 Loaded {len(data)} records")
            
            # Validate data using common utilities
            tprint_debug("🔍 Validating data quality")
            validation_result = validate_klines_data(data)
            
            if not validation_result['valid']:
                tprint_error(f"❌ Data validation failed: {validation_result['errors']}")
                return None
            
            # Apply data quality metrics
            quality_metrics = calculate_data_quality_metrics(data)
            self.quality_metrics[cache_key] = quality_metrics
            tprint_info(f"📈 Data quality metrics: {quality_metrics}")
            
            # Optimize data types for memory efficiency
            tprint_debug("🔧 Optimizing data types")
            data = optimize_dataframe_dtypes(data)
            
            # Guard against null values
            data = guard_dataframe_nulls(data, threshold=0.1)
            
            # Cache the data
            self.data_cache[cache_key] = data
            
            tprint_success(f"✅ Data loaded and validated: {len(data)} records")
            return data
            
        except Exception as e:
            tprint_error(f"❌ Error loading data: {e}")
            self.logger.exception("Data loading error")
            return None
    
    @tprint_timer("Data Processing")
    def process_data(
        self,
        data: pd.DataFrame,
        processing_type: str = "full",
        apply_feature_engineering: bool = True,
        apply_returns_engineering: bool = True,
        detect_gaps: bool = True
    ) -> Optional[pd.DataFrame]:
        """Process data using extensive utility integration.
        
        Args:
            data: Input data to process
            processing_type: Type of processing (full, basic, features_only)
            apply_feature_engineering: Whether to apply feature engineering
            apply_returns_engineering: Whether to apply returns engineering
            detect_gaps: Whether to detect data gaps
            
        Returns:
            Processed DataFrame or None if processing fails
        """
        tprint_info(f"🔧 Processing data with type: {processing_type}")
        
        try:
            # Validate input data
            if not validate_dataframe_columns(data, ['open', 'high', 'low', 'close', 'volume']):
                tprint_error("❌ Invalid data columns for processing")
                return None
            
            # Make a copy to avoid modifying original data
            processed_data = safe_copy(data)
            
            # Apply returns engineering if requested
            if apply_returns_engineering:
                tprint_debug("🔧 Applying returns engineering")
                with memory_checkpoint("returns_engineering"):
                    processed_data = self.returns_engineer.add_basic_returns(processed_data)
            
            # Detect gaps if requested
            if detect_gaps:
                tprint_debug("🔍 Detecting data gaps")
                with memory_checkpoint("gap_detection"):
                    gaps = self.gap_detector.detect_gaps(processed_data)
                    if gaps:
                        tprint_info(f"🔍 Detected {len(gaps)} gaps in data")
                        # Store gap information for later use
                        self.processing_history.append({
                            'timestamp': time.time(),
                            'operation': 'gap_detection',
                            'gaps_found': len(gaps),
                            'gaps': gaps
                        })
            
            # Apply feature engineering if requested
            if apply_feature_engineering:
                tprint_debug("🔧 Applying feature engineering")
                with memory_checkpoint("feature_engineering"):
                    processed_data = self.feature_engineer.add_technical_indicators(processed_data)
                    processed_data = self.feature_engineer.add_price_features(processed_data)
                    processed_data = self.feature_engineer.add_volume_features(processed_data)
                    processed_data = self.feature_engineer.add_time_features(processed_data)
            
            # Apply unified data processing
            if processing_type == "full":
                tprint_debug("🔧 Applying unified data processing")
                with memory_checkpoint("unified_processing"):
                    processed_data = self.unified_data_utils.standardize_data(processed_data)
                    processed_data = self.unified_data_utils.add_derived_features(processed_data)
            
            # Validate processed data
            if not validate_dataframe_columns(processed_data, ['open', 'high', 'low', 'close', 'volume']):
                tprint_error("❌ Processed data missing required columns")
                return None
            
            # Calculate final data quality metrics
            final_quality_metrics = calculate_data_quality_metrics(processed_data)
            tprint_info(f"📈 Final data quality metrics: {final_quality_metrics}")
            
            # Store processing history
            self.processing_history.append({
                'timestamp': time.time(),
                'operation': 'data_processing',
                'processing_type': processing_type,
                'input_shape': data.shape,
                'output_shape': processed_data.shape,
                'quality_metrics': final_quality_metrics
            })
            
            tprint_success(f"✅ Data processed successfully: {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            tprint_error(f"❌ Error processing data: {e}")
            self.logger.exception("Data processing error")
            return None
    
    @tprint_timer("Feature Engineering")
    def engineer_features(
        self,
        data: pd.DataFrame,
        feature_types: List[str] = None,
        custom_features: Optional[Dict[str, Any]] = None
    ) -> Optional[pd.DataFrame]:
        """Engineer features using extensive utility integration.
        
        Args:
            data: Input data for feature engineering
            feature_types: Types of features to engineer
            custom_features: Custom feature definitions
            
        Returns:
            DataFrame with engineered features or None if engineering fails
        """
        tprint_info("🔧 Engineering features with extensive utility integration")
        
        try:
            # Validate input data
            if not validate_dataframe_columns(data, ['open', 'high', 'low', 'close', 'volume']):
                tprint_error("❌ Invalid data columns for feature engineering")
                return None
            
            # Default feature types
            if feature_types is None:
                feature_types = ['technical', 'price', 'volume', 'time', 'statistical']
            
            # Make a copy to avoid modifying original data
            feature_data = safe_copy(data)
            
            # Apply feature engineering based on types
            for feature_type in feature_types:
                tprint_debug(f"🔧 Engineering {feature_type} features")
                
                with memory_checkpoint(f"feature_engineering_{feature_type}"):
                    if feature_type == 'technical':
                        feature_data = self.feature_engineer.add_technical_indicators(feature_data)
                    elif feature_type == 'price':
                        feature_data = self.feature_engineer.add_price_features(feature_data)
                    elif feature_type == 'volume':
                        feature_data = self.feature_engineer.add_volume_features(feature_data)
                    elif feature_type == 'time':
                        feature_data = self.feature_engineer.add_time_features(feature_data)
                    elif feature_type == 'statistical':
                        feature_data = self._add_statistical_features(feature_data)
                    elif feature_type == 'matrix':
                        feature_data = self._add_matrix_features(feature_data)
            
            # Apply custom features if provided
            if custom_features:
                tprint_debug("🔧 Applying custom features")
                with memory_checkpoint("custom_feature_engineering"):
                    feature_data = self._apply_custom_features(feature_data, custom_features)
            
            # Validate engineered features
            if feature_data.empty:
                tprint_error("❌ Feature engineering resulted in empty data")
                return None
            
            # Calculate feature quality metrics
            feature_quality = self._calculate_feature_quality(feature_data)
            tprint_info(f"📈 Feature quality metrics: {feature_quality}")
            
            # Store feature engineering history
            self.processing_history.append({
                'timestamp': time.time(),
                'operation': 'feature_engineering',
                'feature_types': feature_types,
                'custom_features': custom_features is not None,
                'input_shape': data.shape,
                'output_shape': feature_data.shape,
                'feature_quality': feature_quality
            })
            
            tprint_success(f"✅ Feature engineering completed: {feature_data.shape}")
            return feature_data
            
        except Exception as e:
            tprint_error(f"❌ Error in feature engineering: {e}")
            self.logger.exception("Feature engineering error")
            return None
    
    def _add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features using matrix operations."""
        try:
            tprint_debug("🔧 Adding statistical features")
            
            # Extract numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    # Calculate rolling statistics
                    data[f'{col}_rolling_mean_5'] = safe_rolling(data[col], 5).mean()
                    data[f'{col}_rolling_std_5'] = safe_rolling(data[col], 5).std()
                    data[f'{col}_rolling_mean_20'] = safe_rolling(data[col], 20).mean()
                    data[f'{col}_rolling_std_20'] = safe_rolling(data[col], 20).std()
                    
                    # Calculate percentiles
                    data[f'{col}_percentile_25'] = safe_rolling(data[col], 20).quantile(0.25)
                    data[f'{col}_percentile_75'] = safe_rolling(data[col], 20).quantile(0.75)
                    
                    # Calculate skewness and kurtosis
                    data[f'{col}_skewness'] = safe_rolling(data[col], 20).skew()
                    data[f'{col}_kurtosis'] = safe_rolling(data[col], 20).kurt()
            
            return data
            
        except Exception as e:
            tprint_error(f"❌ Error adding statistical features: {e}")
            return data
    
    def _add_matrix_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add matrix-based features using matrix operations utilities."""
        try:
            tprint_debug("🔧 Adding matrix features")
            
            # Extract numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_data = data[numeric_cols].values
            
            # Handle NaN values
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Use matrix operations for feature engineering
            with memory_checkpoint("matrix_feature_engineering"):
                # Normalize features
                normalized_features = self.matrix_ops.normalize_matrix(feature_data)
                
                # Add polynomial features
                polynomial_features = self.enhanced_matrix_ops.add_polynomial_features(
                    normalized_features, degree=2
                )
                
                # Add technical features
                technical_features = self.enhanced_matrix_ops.add_technical_features(
                    normalized_features
                )
                
                # Add trading features
                trading_features = self.matrix_convenience.add_trading_features(
                    technical_features
                )
            
            # Create new columns for matrix features
            n_features = trading_features.shape[1]
            for i in range(n_features):
                if i < len(numeric_cols):
                    # Use original column names for first few features
                    col_name = f"{numeric_cols[i]}_matrix_feature"
                else:
                    col_name = f"matrix_feature_{i}"
                
                data[col_name] = trading_features[:, i]
            
            return data
            
        except Exception as e:
            tprint_error(f"❌ Error adding matrix features: {e}")
            return data
    
    def _apply_custom_features(self, data: pd.DataFrame, custom_features: Dict[str, Any]) -> pd.DataFrame:
        """Apply custom features using safe operations."""
        try:
            tprint_debug("🔧 Applying custom features")
            
            for feature_name, feature_config in custom_features.items():
                try:
                    feature_type = feature_config.get('type', 'rolling')
                    window = feature_config.get('window', 20)
                    column = feature_config.get('column', 'close')
                    
                    if feature_type == 'rolling_mean':
                        data[f'{feature_name}'] = safe_rolling(data[column], window).mean()
                    elif feature_type == 'rolling_std':
                        data[f'{feature_name}'] = safe_rolling(data[column], window).std()
                    elif feature_type == 'rolling_max':
                        data[f'{feature_name}'] = safe_rolling(data[column], window).max()
                    elif feature_type == 'rolling_min':
                        data[f'{feature_name}'] = safe_rolling(data[column], window).min()
                    elif feature_type == 'percentage_change':
                        data[f'{feature_name}'] = safe_percentage_change(data[column], window)
                    elif feature_type == 'correlation':
                        other_column = feature_config.get('other_column', 'volume')
                        data[f'{feature_name}'] = safe_rolling(data[column], window).corr(data[other_column])
                    
                    tprint_debug(f"✅ Added custom feature: {feature_name}")
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Error adding custom feature {feature_name}: {e}")
                    continue
            
            return data
            
        except Exception as e:
            tprint_error(f"❌ Error applying custom features: {e}")
            return data
    
    def _calculate_feature_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate feature quality metrics using math validation utilities."""
        try:
            # Extract numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            quality_metrics = {
                'total_features': len(numeric_cols),
                'feature_correlations': {},
                'feature_variance': {},
                'feature_completeness': {},
                'feature_outliers': {}
            }
            
            # Calculate correlations
            for col in numeric_cols:
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    # Calculate correlation with price
                    if 'close' in data.columns:
                        corr = safe_correlation(data[col], data['close'])
                        quality_metrics['feature_correlations'][col] = corr
                    
                    # Calculate variance
                    variance = safe_std(data[col]) ** 2
                    quality_metrics['feature_variance'][col] = variance
                    
                    # Calculate completeness
                    completeness = safe_divide(data[col].count(), len(data))
                    quality_metrics['feature_completeness'][col] = completeness
                    
                    # Calculate outliers (using IQR method)
                    q25 = safe_percentile(data[col], 25.0)
                    q75 = safe_percentile(data[col], 75.0)
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    
                    outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                    outlier_rate = safe_divide(outliers, len(data))
                    quality_metrics['feature_outliers'][col] = outlier_rate
            
            return quality_metrics
            
        except Exception as e:
            tprint_error(f"❌ Error calculating feature quality: {e}")
            return {}
    
    @tprint_timer("Data Validation")
    def validate_data(
        self,
        data: pd.DataFrame,
        validation_level: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Validate data using extensive utility integration.
        
        Args:
            data: Data to validate
            validation_level: Level of validation (basic, comprehensive, strict)
            
        Returns:
            Dictionary with validation results
        """
        tprint_info(f"🔍 Validating data with level: {validation_level}")
        
        try:
            validation_results = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'quality_metrics': {},
                'recommendations': []
            }
            
            # Basic validation
            if not validate_dataframe_columns(data, ['open', 'high', 'low', 'close', 'volume']):
                validation_results['valid'] = False
                validation_results['errors'].append("Missing required columns")
            
            # Data quality metrics
            quality_metrics = calculate_data_quality_metrics(data)
            validation_results['quality_metrics'] = quality_metrics
            
            # Comprehensive validation
            if validation_level in ['comprehensive', 'strict']:
                # Check for null values
                null_counts = data.isnull().sum()
                high_null_cols = null_counts[null_counts > len(data) * 0.1]
                if not high_null_cols.empty:
                    validation_results['warnings'].append(f"High null values in columns: {list(high_null_cols.index)}")
                
                # Check for infinite values
                inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
                high_inf_cols = inf_counts[inf_counts > 0]
                if not high_inf_cols.empty:
                    validation_results['warnings'].append(f"Infinite values in columns: {list(high_inf_cols.index)}")
                
                # Check for duplicate rows
                duplicates = data.duplicated().sum()
                if duplicates > 0:
                    validation_results['warnings'].append(f"Found {duplicates} duplicate rows")
                
                # Check data types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in data.columns:
                        if not pd.api.types.is_numeric_dtype(data[col]):
                            validation_results['errors'].append(f"Column {col} is not numeric")
            
            # Strict validation
            if validation_level == 'strict':
                # Check for negative prices
                price_cols = ['open', 'high', 'low', 'close']
                for col in price_cols:
                    if col in data.columns:
                        negative_prices = (data[col] <= 0).sum()
                        if negative_prices > 0:
                            validation_results['errors'].append(f"Found {negative_prices} negative/zero prices in {col}")
                
                # Check for negative volume
                if 'volume' in data.columns:
                    negative_volume = (data['volume'] < 0).sum()
                    if negative_volume > 0:
                        validation_results['errors'].append(f"Found {negative_volume} negative volumes")
                
                # Check OHLC relationships
                if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    invalid_ohlc = (
                        (data['high'] < data['low']) |
                        (data['high'] < data['open']) |
                        (data['high'] < data['close']) |
                        (data['low'] > data['open']) |
                        (data['low'] > data['close'])
                    ).sum()
                    
                    if invalid_ohlc > 0:
                        validation_results['errors'].append(f"Found {invalid_ohlc} invalid OHLC relationships")
            
            # Generate recommendations
            if validation_results['quality_metrics'].get('completeness', 1.0) < 0.95:
                validation_results['recommendations'].append("Consider data cleaning for better completeness")
            
            if validation_results['quality_metrics'].get('consistency', 1.0) < 0.9:
                validation_results['recommendations'].append("Consider data standardization for better consistency")
            
            # Final validation status
            if validation_results['errors']:
                validation_results['valid'] = False
            
            tprint_info(f"🔍 Validation completed: {'✅ Valid' if validation_results['valid'] else '❌ Invalid'}")
            if validation_results['errors']:
                tprint_error(f"❌ Validation errors: {validation_results['errors']}")
            if validation_results['warnings']:
                tprint_warning(f"⚠️ Validation warnings: {validation_results['warnings']}")
            
            return validation_results
            
        except Exception as e:
            tprint_error(f"❌ Error validating data: {e}")
            return {'valid': False, 'errors': [str(e)], 'warnings': [], 'quality_metrics': {}}
    
    @tprint_timer("Data Serialization")
    def save_data(
        self,
        data: pd.DataFrame,
        filepath: str,
        format: str = "auto"
    ) -> bool:
        """Save data using serialization utilities.
        
        Args:
            data: Data to save
            filepath: Path to save data
            format: Format to save in (auto, parquet, csv, json)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            tprint_info(f"💾 Saving data to {filepath}")
            
            # Add metadata
            data_with_metadata = {
                'data': data,
                'metadata': {
                    'timestamp': time.time(),
                    'data_manager_version': '1.0.0',
                    'm1_integration': self.m1_integration,
                    'memory_usage': get_memory_usage(),
                    'data_shape': data.shape,
                    'data_columns': list(data.columns),
                    'processing_history': self.processing_history[-10:]  # Last 10 operations
                }
            }
            
            # Save using universal serializer
            success = self.serializer.save(data_with_metadata, filepath)
            
            if success:
                tprint_success(f"✅ Data saved successfully to {filepath}")
            else:
                tprint_error(f"❌ Failed to save data to {filepath}")
            
            return success
            
        except Exception as e:
            tprint_error(f"❌ Error saving data: {e}")
            return False
    
    def load_data_from_file(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load data from file using serialization utilities.
        
        Args:
            filepath: Path to load data from
            
        Returns:
            Loaded data or None if loading fails
        """
        try:
            tprint_info(f"📂 Loading data from {filepath}")
            
            # Load using universal serializer
            data_with_metadata = self.serializer.load(filepath)
            
            if data_with_metadata and 'data' in data_with_metadata:
                data = data_with_metadata['data']
                metadata = data_with_metadata.get('metadata', {})
                
                tprint_info(f"📊 Loaded data: {data.shape}")
                tprint_info(f"📊 Metadata: {metadata}")
                
                tprint_success(f"✅ Data loaded successfully from {filepath}")
                return data
            else:
                tprint_error(f"❌ Failed to load data from {filepath}")
                return None
                
        except Exception as e:
            tprint_error(f"❌ Error loading data: {e}")
            return None
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary using utility integration."""
        try:
            summary = {
                'cache_size': len(self.data_cache),
                'quality_metrics_count': len(self.quality_metrics),
                'processing_history_length': len(self.processing_history),
                'm1_integration': self.m1_integration,
                'memory_usage': get_memory_usage(),
                'cached_datasets': list(self.data_cache.keys()),
                'recent_processing': self.processing_history[-5:] if self.processing_history else []
            }
            
            return summary
            
        except Exception as e:
            tprint_error(f"❌ Error getting data summary: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources and M1 optimizations."""
        try:
            tprint_info("🧹 Cleaning up NAS/TAS Data Manager resources")
            
            # Cleanup M1 optimizers
            cleanup_m1_optimizers()
            
            # Clear caches
            self.data_cache.clear()
            self.quality_metrics.clear()
            self.processing_history.clear()
            
            tprint_success("✅ NAS/TAS Data Manager cleanup completed")
            
        except Exception as e:
            tprint_error(f"❌ Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


# Convenience function for quick data manager usage
def create_nas_data_manager(config: Optional[Dict[str, Any]] = None) -> NASDataManager:
    """Create a NAS/TAS Data Manager instance with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured NASDataManager instance
    """
    return NASDataManager(config)


# Example usage
if __name__ == "__main__":
    # Configure tprint for better output
    from ...tprint import TPrintConfig, configure_tprint
    
    config = TPrintConfig(
        use_colors=True,
        output_to_console=True,
        enable_structured_logging=True
    )
    configure_tprint(config)
    
    # Create and use data manager
    with create_nas_data_manager() as data_manager:
        # Load data
        data = data_manager.load_data("ETHUSDT", "1m")
        
        if data is not None:
            # Process data
            processed_data = data_manager.process_data(
                data, 
                processing_type="full",
                apply_feature_engineering=True
            )
            
            # Engineer features
            if processed_data is not None:
                feature_data = data_manager.engineer_features(
                    processed_data,
                    feature_types=['technical', 'price', 'volume', 'time', 'statistical']
                )
                
                # Validate data
                if feature_data is not None:
                    validation_results = data_manager.validate_data(feature_data, "comprehensive")
                    tprint_structured(validation_results, LogLevel.INFO)
                    
                    # Save processed data
                    data_manager.save_data(feature_data, "processed_data.json")