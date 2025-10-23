"""
Unified Data Utilities Interface

This module provides a single, comprehensive interface for all data processing,
quality validation, and cleaning operations. It consolidates functionality from
multiple specialized modules into a unified API.

Usage:
    from src.utils.data import UnifiedDataUtils

    # Initialize the unified interface
    data_utils = UnifiedDataUtils()

    # Process and validate data in one go
    processed_data = data_utils.process_and_validate(
        data=raw_data,
        validate_quality=True,
        clean_missing_values=True,
        detect_outliers=True,
        optimize_dtypes=True
    )
"""

import logging
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

from .quality.data_quality import DataQualityFramework, QualityResult, QualityThresholds
from .processing.data_processing import DataProcessor
from .quality.data_cleaning import DataCleaner
from .processing.transformers import DataStreamingManager
from src.utils.logger import system_logger
from src.utils.tprint import tprint_data_preview, tprint_data_format

logger = logging.getLogger(__name__)

class UnifiedDataUtils:
    """
    Unified interface for all data processing, quality validation, and cleaning operations.

    This class provides a single entry point for:
    - Data quality validation and scoring
    - Data cleaning and preprocessing
    - Missing value handling
    - Outlier detection and handling
    - Data type optimization
    - Cross-step validation
    - Data streaming and chunking
    """

    def __init__(
        self,
        quality_thresholds: Optional[QualityThresholds] = None,
        enable_streaming: bool = True,
        chunk_size: int = 10000,
        memory_threshold: float = 0.8
    ) -> None:
        """
        Initialize the unified data utilities interface.

        Args:
            quality_thresholds: Custom quality validation thresholds
            enable_streaming: Whether to enable data streaming for large datasets
            chunk_size: Size of data chunks for streaming
            memory_threshold: Memory usage threshold for streaming
        """
        self.logger = system_logger.getChild('UnifiedDataUtils')

        # Initialize core components
        self.quality_framework = DataQualityFramework(quality_thresholds)
        self.data_processor = DataProcessor()
        self.data_cleaner = DataCleaner()
        self._cross_step_validator = None

        # Initialize streaming manager if enabled
        if enable_streaming:
            self.streaming_manager = DataStreamingManager(
                chunk_size=chunk_size,
                memory_threshold=memory_threshold
            )
        else:
            self.streaming_manager = None

        self.logger.info('🚀 Unified Data Utils initialized')
    
    @property
    def cross_step_validator(self):
        """Lazy import of CrossStepValidator to avoid circular imports."""
        if self._cross_step_validator is None:
            from .validation.validators import CrossStepValidator
            self._cross_step_validator = CrossStepValidator()
        return self._cross_step_validator

    def process_and_validate(
        self,
        data: pd.DataFrame,
        validate_quality: bool = True,
        clean_missing_values: bool = True,
        detect_outliers: bool = True,
        optimize_dtypes: bool = True,
        regularize_timestamps: bool = True,
        context: str = '',
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: str = '1m'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process and validate data using all available utilities.

        Args:
            data: Input DataFrame
            validate_quality: Whether to perform quality validation
            clean_missing_values: Whether to clean missing values
            detect_outliers: Whether to detect outliers
            optimize_dtypes: Whether to optimize data types
            regularize_timestamps: Whether to regularize timestamps
            context: Context string for logging
            symbol: Trading symbol for data download (if needed)
            exchange: Exchange name for data download (if needed)
            timeframe: Timeframe for data download (if needed)

        Returns:
            Tuple of (processed_data, processing_report)
        """
        start_time = datetime.now()
        processing_report = {
            'start_time': start_time.isoformat(),
            'original_shape': data.shape,
            'steps_completed': [],
            'quality_results': {},
            'cleaning_results': {},
            'optimization_results': {},
            'errors': [],
            'warnings': []
        }

        try:
            processed_data = data.copy()
            tprint_data_preview(processed_data, f"input_data_{context}")
            tprint_data_format(processed_data, f"input_data_format_{context}", level="DEBUG")

            # Step 1: Quality validation
            if validate_quality:
                self.logger.info('🔍 Performing quality validation...')
                tprint_data_preview(processed_data, f"before_quality_validation_{context}")
                tprint_data_format(processed_data, f"before_quality_validation_format_{context}", level="DEBUG")
                quality_result = self.quality_framework.validate_dataframe_quality(processed_data, context)
                processing_report['quality_results'] = quality_result.get_summary()
                processing_report['steps_completed'].append('quality_validation')

                if not quality_result.passed:
                    processing_report['warnings'].append(f'Quality validation failed: {len(quality_result.issues)} issues found')

            # Step 2: Regularize timestamps
            if regularize_timestamps and 'timestamp' in processed_data.columns:
                self.logger.info('⏰ Regularizing timestamps...')
                processed_data = self.data_processor.regularize_timestamps(processed_data)
                tprint_data_preview(processed_data, f"after_timestamp_regularization_{context}")
                tprint_data_format(processed_data, f"after_timestamp_regularization_format_{context}", level="DEBUG")
                processing_report['steps_completed'].append('timestamp_regularization')

            # Step 3: Clean missing values
            if clean_missing_values and 'timestamp' in processed_data.columns:
                self.logger.info('🧹 Cleaning missing values...')
                processed_data = self.data_cleaner.handle_missing_values_intelligently(
                    processed_data,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe
                )
                tprint_data_preview(processed_data, f"after_missing_value_cleaning_{context}")
                tprint_data_format(processed_data, f"after_missing_value_cleaning_format_{context}", level="DEBUG")
                processing_report['steps_completed'].append('missing_value_cleaning')

            # Step 4: Detect outliers
            if detect_outliers:
                self.logger.info('🔍 Detecting outliers...')
                outliers = self.data_cleaner.detect_outliers(processed_data, raise_errors=False)
                processing_report['cleaning_results']['outliers_detected'] = len(outliers)
                processing_report['steps_completed'].append('outlier_detection')

                if outliers:
                    processing_report['warnings'].append(f'Outliers detected: {len(outliers)} groups')

            # Step 5: Optimize data types
            if optimize_dtypes:
                self.logger.info('🔧 Optimizing data types...')
                original_memory = processed_data.memory_usage(deep=True).sum()
                processed_data = self.data_processor.optimize_dataframe_dtypes(processed_data)
                final_memory = processed_data.memory_usage(deep=True).sum()
                memory_reduction = (original_memory - final_memory) / original_memory * 100

                processing_report['optimization_results'] = {
                    'original_memory_mb': original_memory / 1024 / 1024,
                    'final_memory_mb': final_memory / 1024 / 1024,
                    'memory_reduction_percent': memory_reduction
                }
                tprint_data_preview(processed_data, f"after_dtype_optimization_{context}")
                processing_report['steps_completed'].append('dtype_optimization')

            # Final quality check
            if validate_quality:
                self.logger.info('🔍 Performing final quality validation...')
                final_quality_result = self.quality_framework.validate_dataframe_quality(processed_data, f"{context}_final")
                processing_report['quality_results']['final'] = final_quality_result.get_summary()

            processing_report['final_shape'] = processed_data.shape
            processing_report['end_time'] = datetime.now().isoformat()
            processing_report['processing_time_seconds'] = (datetime.now() - start_time).total_seconds()
            processing_report['success'] = True

            self.logger.info(f'✅ Data processing completed successfully in {processing_report["processing_time_seconds"]:.2f}s')
            self.logger.info(f'   Original shape: {processing_report["original_shape"]} → Final shape: {processing_report["final_shape"]}')
            tprint_data_preview(processed_data, f"final_processed_data_{context}")
            tprint_data_format(processed_data, f"final_processed_data_format_{context}", level="INFO")

            return processed_data, processing_report

        except Exception as e:
            processing_report['success'] = False
            processing_report['errors'].append(str(e))
            processing_report['end_time'] = datetime.now().isoformat()
            processing_report['processing_time_seconds'] = (datetime.now() - start_time).total_seconds()

            self.logger.exception(f'❌ Error in data processing: {e}')
            # Add format debugging for error troubleshooting
            tprint_data_format(data, f"error_data_format_{context}", level="ERROR")
            return data, processing_report

    def validate_data_quality(
        self,
        data: pd.DataFrame,
        context: str = '',
        validation_rules: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate data quality using the comprehensive framework.

        Args:
            data: DataFrame to validate
            context: Context string for logging
            validation_rules: Specific validation rules to apply

        Returns:
            Validation results dictionary
        """
        return self.quality_framework.validate_data(data, validation_rules)

    def clean_data(
        self,
        data: pd.DataFrame,
        timestamp_column: str = 'timestamp',
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: str = '1m',
        detect_outliers: bool = True,
        outlier_method: str = 'zscore',
        outlier_threshold: float = 3.0
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean data by handling missing values and detecting outliers.

        Args:
            data: DataFrame to clean
            timestamp_column: Name of timestamp column
            symbol: Trading symbol for data download
            exchange: Exchange name for data download
            timeframe: Timeframe for data download
            detect_outliers: Whether to detect outliers
            outlier_method: Method for outlier detection
            outlier_threshold: Threshold for outlier detection

        Returns:
            Tuple of (cleaned_data, cleaning_report)
        """
        cleaning_report = {
            'original_shape': data.shape,
            'steps_completed': [],
            'outliers_detected': 0,
            'gaps_filled': 0
        }

        try:
            cleaned_data = data.copy()

            # Handle missing values
            if timestamp_column in cleaned_data.columns:
                cleaned_data = self.data_cleaner.handle_missing_values_intelligently(
                    cleaned_data,
                    timestamp_column=timestamp_column,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe
                )
                cleaning_report['steps_completed'].append('missing_value_handling')

            # Detect outliers
            if detect_outliers:
                outliers = self.data_cleaner.detect_outliers(
                    cleaned_data,
                    method=outlier_method,
                    threshold=outlier_threshold,
                    raise_errors=False
                )
                cleaning_report['outliers_detected'] = len(outliers)
                cleaning_report['steps_completed'].append('outlier_detection')

            cleaning_report['final_shape'] = cleaned_data.shape
            cleaning_report['success'] = True

            return cleaned_data, cleaning_report

        except Exception as e:
            cleaning_report['success'] = False
            cleaning_report['error'] = str(e)
            self.logger.exception(f'Error in data cleaning: {e}')
            return data, cleaning_report

    def optimize_data(
        self,
        data: pd.DataFrame,
        stage: str = 'output',
        preserve_categorical: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Optimize data types and memory usage.

        Args:
            data: DataFrame to optimize
            stage: Pipeline stage ('input', 'intermediate', 'output')
            preserve_categorical: Whether to preserve categorical columns

        Returns:
            Tuple of (optimized_data, optimization_report)
        """
        optimization_report = {
            'original_shape': data.shape,
            'original_memory_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'stage': stage
        }

        try:
            if stage == 'output':
                optimized_data = self.data_processor.apply_feature_specific_optimization(data)
            else:
                optimized_data = self.data_processor.optimize_dataframe_dtypes(
                    data,
                    preserve_categorical=preserve_categorical
                )

            optimization_report['final_memory_mb'] = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024
            optimization_report['memory_reduction_percent'] = (
                (optimization_report['original_memory_mb'] - optimization_report['final_memory_mb']) /
                optimization_report['original_memory_mb'] * 100
            )
            optimization_report['final_shape'] = optimized_data.shape
            optimization_report['success'] = True

            return optimized_data, optimization_report

        except Exception as e:
            optimization_report['success'] = False
            optimization_report['error'] = str(e)
            self.logger.exception(f'Error in data optimization: {e}')
            return data, optimization_report

    def process_large_dataset(
        self,
        data: pd.DataFrame,
        processing_func: callable,
        combine_results: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Process large datasets using streaming and chunking.

        Args:
            data: DataFrame to process
            processing_func: Function to apply to each chunk
            combine_results: Whether to combine results into single DataFrame
            progress_callback: Optional progress callback function

        Returns:
            Processed DataFrame or list of processed chunks
        """
        if self.streaming_manager is None:
            self.logger.warning('Streaming manager not enabled, processing in memory')
            return processing_func(data)

        return self.streaming_manager.process_large_dataset(
            data,
            processing_func,
            combine_results=combine_results,
            progress_callback=progress_callback
        )

    def validate_step_transition(
        self,
        from_step: str,
        to_step: str,
        input_data: pd.DataFrame,
        output_data: pd.DataFrame,
        step_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate data consistency between pipeline steps.

        Args:
            from_step: Name of the source step
            to_step: Name of the destination step
            input_data: Input DataFrame
            output_data: Output DataFrame
            step_metadata: Additional step metadata

        Returns:
            Validation results dictionary
        """
        return self.cross_step_validator.validate_step_transition(
            from_step,
            to_step,
            input_data,
            output_data,
            step_metadata
        )

    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all processing capabilities and current state.

        Returns:
            Summary dictionary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'components': {
                'quality_framework': 'DataQualityFramework',
                'data_processor': 'DataProcessor',
                'data_cleaner': 'DataCleaner',
                'cross_step_validator': 'CrossStepValidator',
                'streaming_manager': 'DataStreamingManager' if self.streaming_manager else None
            },
            'capabilities': [
                'data_quality_validation',
                'missing_value_handling',
                'outlier_detection',
                'data_type_optimization',
                'timestamp_regularization',
                'cross_step_validation',
                'large_dataset_streaming'
            ],
            'streaming_enabled': self.streaming_manager is not None
        }

        if self.streaming_manager:
            summary['streaming_config'] = {
                'chunk_size': self.streaming_manager.chunk_size,
                'memory_threshold': self.streaming_manager.memory_threshold
            }

        return summary

# Create global instance for convenience
unified_data_utils = UnifiedDataUtils()
