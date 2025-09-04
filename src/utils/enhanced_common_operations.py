#!/usr/bin/env python3
"""
Enhanced Common Operations

This module provides enhanced common operations with comprehensive validation,
error handling, and safety features for the trading pipeline.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import pandas as pd
import numpy as np

from src.utils.common_operations import (
    format_datetime,
    get_current_datetime,
    safe_file_exists,
    safe_json_load,
    safe_json_dump,
    ensure_directory,
    safe_float,
    safe_int,
)
from src.utils.logger import system_logger
from src.utils.operation_protection_decorators import (
    validate_data_format,
    validate_data_access,
    safe_operation,
    performance_monitor,
)


class DataValidationError(Exception):
    """Exception for data validation failures."""
    pass


class DataIntegrityError(Exception):
    """Exception for data integrity failures."""
    pass


class DataProcessingError(Exception):
    """Exception for data processing failures."""
    pass


def validate_dataframe_integrity(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
    """
    Validate DataFrame integrity with comprehensive checks.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required columns
        
    Returns:
        Dictionary with validation results
    """
    logger = system_logger.getChild("DataFrameValidator")
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {},
        'recommendations': []
    }
    
    try:
        # Check if DataFrame is empty
        if df.empty:
            validation_results['errors'].append("DataFrame is empty")
            validation_results['is_valid'] = False
            return validation_results
        
        # Check required columns
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_results['errors'].append(f"Missing required columns: {missing_columns}")
                validation_results['is_valid'] = False
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results['warnings'].append(f"Found {duplicate_count} duplicate rows")
            validation_results['recommendations'].append("Consider removing duplicate rows")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        total_missing = missing_values.sum()
        if total_missing > 0:
            validation_results['warnings'].append(f"Found {total_missing} missing values")
            validation_results['statistics']['missing_values'] = missing_values.to_dict()
            validation_results['recommendations'].append("Consider handling missing values")
        
        # Check for infinite values in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            infinite_count = np.isinf(df[numeric_columns]).sum().sum()
            if infinite_count > 0:
                validation_results['errors'].append(f"Found {infinite_count} infinite values")
                validation_results['is_valid'] = False
        
        # Check for negative values in price/volume columns
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['price', 'volume', 'amount']):
                if df[col].dtype in [np.float64, np.int64]:
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        validation_results['warnings'].append(f"Found {negative_count} negative values in {col}")
        
        # Calculate basic statistics
        validation_results['statistics'].update({
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': list(numeric_columns)
        })
        
        logger.info(f"DataFrame validation completed: {validation_results['is_valid']}")
        
    except Exception as e:
        logger.exception(f"DataFrame validation failed: {e}")
        validation_results['errors'].append(f"Validation error: {str(e)}")
        validation_results['is_valid'] = False
    
    return validation_results


@validate_data_format(allow_empty=False)
@performance_monitor(performance_threshold=5.0)
def load_and_validate_data(file_path: str, required_columns: List[str] = None) -> pd.DataFrame:
    """
    Load and validate data from file with comprehensive checks.
    
    Args:
        file_path: Path to the data file
        required_columns: List of required columns
        
    Returns:
        Validated DataFrame
    """
    logger = system_logger.getChild("DataLoader")
    
    if not safe_file_exists(file_path):
        raise DataValidationError(f"File does not exist: {file_path}")
    
    try:
        # Load data based on file extension
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise DataValidationError(f"Unsupported file format: {file_path}")
        
        # Validate DataFrame integrity
        validation_results = validate_dataframe_integrity(df, required_columns)
        
        if not validation_results['is_valid']:
            error_msg = f"Data validation failed: {validation_results['errors']}"
            logger.error(error_msg)
            raise DataValidationError(error_msg)
        
        if validation_results['warnings']:
            logger.warning(f"Data validation warnings: {validation_results['warnings']}")
        
        logger.info(f"Successfully loaded and validated data from {file_path}")
        return df
        
    except Exception as e:
        logger.exception(f"Failed to load data from {file_path}: {e}")
        raise DataProcessingError(f"Failed to load data: {e}") from e


@validate_data_format(allow_empty=False)
@performance_monitor(performance_threshold=10.0)
def clean_and_prepare_data(df: pd.DataFrame, cleaning_config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Clean and prepare data with comprehensive preprocessing.
    
    Args:
        df: Input DataFrame
        cleaning_config: Configuration for cleaning operations
        
    Returns:
        Cleaned DataFrame
    """
    logger = system_logger.getChild("DataCleaner")
    
    if cleaning_config is None:
        cleaning_config = {
            'remove_duplicates': True,
            'handle_missing': 'forward_fill',
            'remove_outliers': False,
            'normalize_columns': False
        }
    
    try:
        cleaned_df = df.copy()
        
        # Remove duplicates
        if cleaning_config.get('remove_duplicates', True):
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            removed_count = initial_count - len(cleaned_df)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} duplicate rows")
        
        # Handle missing values
        missing_strategy = cleaning_config.get('handle_missing', 'forward_fill')
        if missing_strategy == 'forward_fill':
            cleaned_df = cleaned_df.fillna(method='ffill')
        elif missing_strategy == 'backward_fill':
            cleaned_df = cleaned_df.fillna(method='bfill')
        elif missing_strategy == 'interpolate':
            cleaned_df = cleaned_df.interpolate()
        elif missing_strategy == 'drop':
            cleaned_df = cleaned_df.dropna()
        
        # Remove outliers (if enabled)
        if cleaning_config.get('remove_outliers', False):
            numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
                if outlier_count > 0:
                    logger.info(f"Found {outlier_count} outliers in {col}")
                    cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
        
        # Normalize columns (if enabled)
        if cleaning_config.get('normalize_columns', False):
            numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if cleaned_df[col].std() > 0:  # Avoid division by zero
                    cleaned_df[col] = (cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std()
        
        logger.info(f"Data cleaning completed. Shape: {df.shape} -> {cleaned_df.shape}")
        return cleaned_df
        
    except Exception as e:
        logger.exception(f"Data cleaning failed: {e}")
        raise DataProcessingError(f"Data cleaning failed: {e}") from e


@validate_data_analysis(required_outputs=['statistics', 'insights', 'recommendations'])
@performance_monitor(performance_threshold=15.0)
def analyze_data_quality(df: pd.DataFrame, analysis_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analyze data quality with comprehensive metrics.
    
    Args:
        df: DataFrame to analyze
        analysis_config: Configuration for analysis
        
    Returns:
        Dictionary with analysis results
    """
    logger = system_logger.getChild("DataQualityAnalyzer")
    
    if analysis_config is None:
        analysis_config = {
            'include_correlation': True,
            'include_distribution': True,
            'include_trend_analysis': True
        }
    
    try:
        analysis_results = {
            'statistics': {},
            'insights': [],
            'recommendations': [],
            'quality_score': 0.0
        }
        
        # Basic statistics
        analysis_results['statistics'] = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        # Numeric column analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            numeric_stats = df[numeric_columns].describe()
            analysis_results['statistics']['numeric_summary'] = numeric_stats.to_dict()
            
            # Correlation analysis
            if analysis_config.get('include_correlation', True) and len(numeric_columns) > 1:
                correlation_matrix = df[numeric_columns].corr()
                analysis_results['statistics']['correlation_matrix'] = correlation_matrix.to_dict()
                
                # Find high correlations
                high_correlations = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > 0.8:
                            high_correlations.append({
                                'columns': [correlation_matrix.columns[i], correlation_matrix.columns[j]],
                                'correlation': corr_value
                            })
                
                if high_correlations:
                    analysis_results['insights'].append(f"Found {len(high_correlations)} high correlations (>0.8)")
                    analysis_results['statistics']['high_correlations'] = high_correlations
        
        # Data quality insights
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_percentage > 10:
            analysis_results['insights'].append(f"High missing data percentage: {missing_percentage:.2f}%")
            analysis_results['recommendations'].append("Consider data imputation strategies")
        
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        if duplicate_percentage > 5:
            analysis_results['insights'].append(f"High duplicate percentage: {duplicate_percentage:.2f}%")
            analysis_results['recommendations'].append("Consider removing duplicate rows")
        
        # Calculate quality score
        quality_score = 100.0
        quality_score -= missing_percentage * 2  # Penalize missing data
        quality_score -= duplicate_percentage * 1  # Penalize duplicates
        quality_score = max(0, quality_score)  # Ensure non-negative
        
        analysis_results['quality_score'] = quality_score
        
        if quality_score < 70:
            analysis_results['insights'].append(f"Low data quality score: {quality_score:.2f}")
            analysis_results['recommendations'].append("Improve data quality before proceeding")
        
        logger.info(f"Data quality analysis completed. Quality score: {quality_score:.2f}")
        return analysis_results
        
    except Exception as e:
        logger.exception(f"Data quality analysis failed: {e}")
        raise DataProcessingError(f"Data quality analysis failed: {e}") from e


@validate_data_access(required_directories=['data_cache'])
@performance_monitor(performance_threshold=30.0)
def save_processed_data(df: pd.DataFrame, output_path: str, save_config: Dict[str, Any] = None) -> bool:
    """
    Save processed data with comprehensive validation.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
        save_config: Configuration for saving
        
    Returns:
        True if successful, False otherwise
    """
    logger = system_logger.getChild("DataSaver")
    
    if save_config is None:
        save_config = {
            'format': 'parquet',
            'compression': 'snappy',
            'index': False,
            'validate_before_save': True
        }
    
    try:
        # Validate DataFrame before saving
        if save_config.get('validate_before_save', True):
            validation_results = validate_dataframe_integrity(df)
            if not validation_results['is_valid']:
                raise DataValidationError(f"DataFrame validation failed: {validation_results['errors']}")
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        ensure_directory(output_dir)
        
        # Save based on format
        if save_config['format'] == 'parquet':
            df.to_parquet(output_path, compression=save_config.get('compression', 'snappy'), index=save_config.get('index', False))
        elif save_config['format'] == 'csv':
            df.to_csv(output_path, index=save_config.get('index', False))
        elif save_config['format'] == 'json':
            df.to_json(output_path, orient='records', index=save_config.get('index', False))
        else:
            raise DataProcessingError(f"Unsupported save format: {save_config['format']}")
        
        # Verify file was created
        if not safe_file_exists(output_path):
            raise DataProcessingError(f"Failed to create output file: {output_path}")
        
        file_size = Path(output_path).stat().st_size
        logger.info(f"Successfully saved data to {output_path} (size: {file_size} bytes)")
        return True
        
    except Exception as e:
        logger.exception(f"Failed to save data to {output_path}: {e}")
        raise DataProcessingError(f"Failed to save data: {e}") from e


@safe_operation(max_retries=3, retry_delay=1.0)
@performance_monitor(performance_threshold=5.0)
def calculate_data_hash(df: pd.DataFrame, algorithm: str = 'sha256') -> str:
    """
    Calculate hash of DataFrame for integrity checking.
    
    Args:
        df: DataFrame to hash
        algorithm: Hash algorithm to use
        
    Returns:
        Hash string
    """
    logger = system_logger.getChild("DataHasher")
    
    try:
        # Convert DataFrame to string representation
        df_string = df.to_string()
        
        # Calculate hash
        if algorithm == 'sha256':
            hash_obj = hashlib.sha256(df_string.encode())
        elif algorithm == 'md5':
            hash_obj = hashlib.md5(df_string.encode())
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        hash_value = hash_obj.hexdigest()
        logger.info(f"Calculated {algorithm} hash: {hash_value[:16]}...")
        return hash_value
        
    except Exception as e:
        logger.exception(f"Failed to calculate data hash: {e}")
        raise DataProcessingError(f"Failed to calculate data hash: {e}") from e


def create_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create comprehensive data summary.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary with data summary
    """
    logger = system_logger.getChild("DataSummarizer")
    
    try:
        summary = {
            'basic_info': {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'infinite_values': np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            },
            'statistics': {}
        }
        
        # Add statistics for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            summary['statistics']['numeric'] = df[numeric_columns].describe().to_dict()
        
        # Add statistics for categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            summary['statistics']['categorical'] = {}
            for col in categorical_columns:
                summary['statistics']['categorical'][col] = {
                    'unique_count': df[col].nunique(),
                    'most_common': df[col].value_counts().head().to_dict()
                }
        
        logger.info(f"Created data summary for DataFrame with shape {df.shape}")
        return summary
        
    except Exception as e:
        logger.exception(f"Failed to create data summary: {e}")
        raise DataProcessingError(f"Failed to create data summary: {e}") from e


def validate_pipeline_step_output(step_name: str, output_data: Any, expected_type: type = None) -> bool:
    """
    Validate pipeline step output.
    
    Args:
        step_name: Name of the pipeline step
        output_data: Output data to validate
        expected_type: Expected data type
        
    Returns:
        True if valid, False otherwise
    """
    logger = system_logger.getChild("PipelineStepValidator")
    
    try:
        # Check if output is None
        if output_data is None:
            logger.error(f"Pipeline step {step_name} returned None")
            return False
        
        # Check expected type
        if expected_type and not isinstance(output_data, expected_type):
            logger.error(f"Pipeline step {step_name} returned wrong type: {type(output_data)}, expected: {expected_type}")
            return False
        
        # Additional validation for DataFrames
        if isinstance(output_data, pd.DataFrame):
            if output_data.empty:
                logger.warning(f"Pipeline step {step_name} returned empty DataFrame")
                return False
            
            # Check for required columns if specified
            if hasattr(output_data, '_required_columns'):
                missing_columns = [col for col in output_data._required_columns if col not in output_data.columns]
                if missing_columns:
                    logger.error(f"Pipeline step {step_name} missing required columns: {missing_columns}")
                    return False
        
        logger.info(f"Pipeline step {step_name} output validation passed")
        return True
        
    except Exception as e:
        logger.exception(f"Pipeline step {step_name} output validation failed: {e}")
        return False


# Export commonly used functions
__all__ = [
    'validate_dataframe_integrity',
    'load_and_validate_data',
    'clean_and_prepare_data',
    'analyze_data_quality',
    'save_processed_data',
    'calculate_data_hash',
    'create_data_summary',
    'validate_pipeline_step_output',
    'DataValidationError',
    'DataIntegrityError',
    'DataProcessingError'
]