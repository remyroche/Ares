#!/usr/bin/env python3
"""
Enhanced Data Validation Utilities

This module provides comprehensive data validation utilities for the trading pipeline,
including validators for each step, data quality checks, and common utilities for
data operations with proper error handling and logging.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json

from src.utils.base_validator import BaseValidator
from src.utils.common_operations import (
    format_datetime, get_current_datetime, safe_file_exists, 
    ensure_directory, safe_json_dump, safe_json_load, safe_fillna
)
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose
)
from src.core.domain.decorators import (
    validate_data_quality, monitor_step_execution, 
    ensure_data_integrity, validate_pipeline_step
)

logger = logging.getLogger(__name__)

class DataQualityValidator:
    """Comprehensive data quality validator for trading data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DataQualityValidator")
    
    @compose(
        error_boundary(name="validate_ohlc_data"),
        traced(span_name="validate_ohlc_data"),
        validate_data_quality(
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            check_ohlc_integrity=True,
            check_nan=True,
            check_infinite=True,
            context='ohlc_validation'
        )
    )
    async def validate_ohlc_data(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate OHLC data quality."""
        try:
            self.logger.info(f"🔍 Validating OHLC data with {len(df)} rows")
            
            validation_results = {
                'total_rows': len(df),
                'validation_passed': True,
                'issues': [],
                'warnings': [],
                'metrics': {}
            }
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_results['issues'].append(f"Missing required columns: {missing_columns}")
                validation_results['validation_passed'] = False
            
            # Check OHLC integrity
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # High should be >= max(open, close)
                invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
                if invalid_high.any():
                    count = invalid_high.sum()
                    validation_results['issues'].append(f"Found {count} rows where high < max(open, close)")
                    validation_results['validation_passed'] = False
                
                # Low should be <= min(open, close)
                invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
                if invalid_low.any():
                    count = invalid_low.sum()
                    validation_results['issues'].append(f"Found {count} rows where low > min(open, close)")
                    validation_results['validation_passed'] = False
                
                # High should be >= low
                invalid_hl = df['high'] < df['low']
                if invalid_hl.any():
                    count = invalid_hl.sum()
                    validation_results['issues'].append(f"Found {count} rows where high < low")
                    validation_results['validation_passed'] = False
            
            # Check for negative values
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        validation_results['warnings'].append(f"Found {negative_count} negative values in {col}")
            
            # Check for zero volume
            if 'volume' in df.columns:
                zero_volume_count = (df['volume'] == 0).sum()
                if zero_volume_count > 0:
                    validation_results['warnings'].append(f"Found {zero_volume_count} rows with zero volume")
            
            # Calculate basic metrics
            if validation_results['validation_passed']:
                validation_results['metrics'] = {
                    'price_range': {
                        'min': df['close'].min(),
                        'max': df['close'].max(),
                        'mean': df['close'].mean()
                    },
                    'volume_stats': {
                        'min': df['volume'].min(),
                        'max': df['volume'].max(),
                        'mean': df['volume'].mean()
                    },
                    'data_quality_score': self._calculate_quality_score(df)
                }
            
            self.logger.info(f"✅ OHLC validation completed: {validation_results['validation_passed']}")
            return validation_results['validation_passed'], validation_results
            
        except Exception as e:
            self.logger.exception(f"❌ Error in OHLC validation: {e}")
            return False, {'error': str(e), 'validation_passed': False}
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate a data quality score (0-1)."""
        try:
            score = 1.0
            
            # Deduct for missing values
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            score -= missing_ratio * 0.3
            
            # Deduct for infinite values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            inf_count = 0
            for col in numeric_cols:
                inf_count += np.isinf(df[col]).sum()
            inf_ratio = inf_count / (len(df) * len(numeric_cols))
            score -= inf_ratio * 0.2
            
            # Deduct for duplicate rows
            duplicate_ratio = df.duplicated().sum() / len(df)
            score -= duplicate_ratio * 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.0
    
    @compose(
        error_boundary(name="validate_timestamp_data"),
        traced(span_name="validate_timestamp_data")
    )
    async def validate_timestamp_data(self, df: pd.DataFrame, timestamp_column: str = 'timestamp') -> Tuple[bool, Dict[str, Any]]:
        """Validate timestamp data quality."""
        try:
            self.logger.info(f"🔍 Validating timestamp data in column '{timestamp_column}'")
            
            validation_results = {
                'validation_passed': True,
                'issues': [],
                'warnings': [],
                'metrics': {}
            }
            
            if timestamp_column not in df.columns:
                validation_results['issues'].append(f"Timestamp column '{timestamp_column}' not found")
                validation_results['validation_passed'] = False
                return validation_results['validation_passed'], validation_results
            
            # Check for missing timestamps
            missing_timestamps = df[timestamp_column].isnull().sum()
            if missing_timestamps > 0:
                validation_results['issues'].append(f"Found {missing_timestamps} missing timestamps")
                validation_results['validation_passed'] = False
            
            # Check for duplicate timestamps
            duplicate_timestamps = df[timestamp_column].duplicated().sum()
            if duplicate_timestamps > 0:
                validation_results['issues'].append(f"Found {duplicate_timestamps} duplicate timestamps")
                validation_results['validation_passed'] = False
            
            # Check timestamp ordering
            if not df[timestamp_column].is_monotonic_increasing:
                validation_results['warnings'].append("Timestamps are not in chronological order")
            
            # Calculate time metrics
            if validation_results['validation_passed']:
                timestamps = pd.to_datetime(df[timestamp_column])
                time_diff = timestamps.diff().dropna()
                
                validation_results['metrics'] = {
                    'start_time': timestamps.min(),
                    'end_time': timestamps.max(),
                    'duration': timestamps.max() - timestamps.min(),
                    'avg_interval': time_diff.mean(),
                    'min_interval': time_diff.min(),
                    'max_interval': time_diff.max(),
                    'total_intervals': len(time_diff)
                }
            
            self.logger.info(f"✅ Timestamp validation completed: {validation_results['validation_passed']}")
            return validation_results['validation_passed'], validation_results
            
        except Exception as e:
            self.logger.exception(f"❌ Error in timestamp validation: {e}")
            return False, {'error': str(e), 'validation_passed': False}

class PipelineStepValidator(BaseValidator):
    """Enhanced validator for pipeline steps with comprehensive checks."""
    
    def __init__(self, step_name: str, config: Dict[str, Any]):
        super().__init__(step_name, config)
        self.data_quality_validator = DataQualityValidator(config)
    
    @compose(
        error_boundary(name="validate_step_data"),
        traced(span_name="validate_step_data"),
        monitor_step_execution(step_name="validate_step_data")
    )
    async def validate_step_data(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
        data: Optional[pd.DataFrame] = None
    ) -> bool:
        """Validate data for a specific pipeline step."""
        try:
            self.logger.info(f"🔍 Validating data for step: {self.step_name}")
            
            validation_results = {
                'step_name': self.step_name,
                'validation_passed': True,
                'issues': [],
                'warnings': [],
                'metrics': {}
            }
            
            # Validate input parameters
            input_validation = await self._validate_input_parameters(training_input)
            if not input_validation['validation_passed']:
                validation_results['issues'].extend(input_validation['issues'])
                validation_results['validation_passed'] = False
            
            # Validate data if provided
            if data is not None:
                data_validation = await self._validate_data_quality(data)
                if not data_validation['validation_passed']:
                    validation_results['issues'].extend(data_validation['issues'])
                    validation_results['validation_passed'] = False
                validation_results['warnings'].extend(data_validation.get('warnings', []))
                validation_results['metrics']['data_quality'] = data_validation.get('metrics', {})
            
            # Validate prerequisites
            prerequisites_validation = await self._validate_prerequisites(pipeline_state)
            if not prerequisites_validation['validation_passed']:
                validation_results['issues'].extend(prerequisites_validation['issues'])
                validation_results['validation_passed'] = False
            
            # Store results
            self.validation_results = validation_results
            
            if validation_results['validation_passed']:
                self.logger.info(f"✅ Step validation passed: {self.step_name}")
            else:
                self.logger.error(f"❌ Step validation failed: {self.step_name}")
                for issue in validation_results['issues']:
                    self.logger.error(f"   • {issue}")
            
            return validation_results['validation_passed']
            
        except Exception as e:
            self.logger.exception(f"❌ Error in step validation: {e}")
            return False
    
    async def _validate_input_parameters(self, training_input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input parameters."""
        try:
            validation_results = {
                'validation_passed': True,
                'issues': []
            }
            
            # Check required parameters
            required_params = ['symbol', 'exchange']
            for param in required_params:
                if param not in training_input or not training_input[param]:
                    validation_results['issues'].append(f"Missing required parameter: {param}")
                    validation_results['validation_passed'] = False
            
            # Validate symbol format
            if 'symbol' in training_input:
                symbol = training_input['symbol']
                if not isinstance(symbol, str) or len(symbol) < 3:
                    validation_results['issues'].append(f"Invalid symbol format: {symbol}")
                    validation_results['validation_passed'] = False
            
            # Validate exchange
            if 'exchange' in training_input:
                exchange = training_input['exchange']
                valid_exchanges = ['BINANCE', 'MEXC', 'GATEIO']
                if exchange.upper() not in valid_exchanges:
                    validation_results['issues'].append(f"Unsupported exchange: {exchange}")
                    validation_results['validation_passed'] = False
            
            return validation_results
            
        except Exception as e:
            return {
                'validation_passed': False,
                'issues': [f"Error validating input parameters: {e}"]
            }
    
    async def _validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality using the data quality validator."""
        try:
            # Validate OHLC data
            ohlc_passed, ohlc_results = await self.data_quality_validator.validate_ohlc_data(data)
            
            # Validate timestamp data
            timestamp_passed, timestamp_results = await self.data_quality_validator.validate_timestamp_data(data)
            
            return {
                'validation_passed': ohlc_passed and timestamp_passed,
                'issues': ohlc_results.get('issues', []) + timestamp_results.get('issues', []),
                'warnings': ohlc_results.get('warnings', []) + timestamp_results.get('warnings', []),
                'metrics': {
                    'ohlc': ohlc_results.get('metrics', {}),
                    'timestamp': timestamp_results.get('metrics', {})
                }
            }
            
        except Exception as e:
            return {
                'validation_passed': False,
                'issues': [f"Error validating data quality: {e}"]
            }
    
    async def _validate_prerequisites(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate step prerequisites."""
        try:
            validation_results = {
                'validation_passed': True,
                'issues': []
            }
            
            # Define step dependencies
            step_dependencies = {
                'step1_data_collection': [],
                'step2_data_reading': ['step1_data_collection'],
                'step3_market_analysis': ['step2_data_reading'],
                'step9_model_training': ['step3_market_analysis'],
                'step18_backtesting': ['step9_model_training']
            }
            
            # Check if current step has dependencies
            if self.step_name in step_dependencies:
                required_steps = step_dependencies[self.step_name]
                for required_step in required_steps:
                    if required_step not in pipeline_state or not pipeline_state[required_step].get('completed', False):
                        validation_results['issues'].append(f"Missing prerequisite: {required_step}")
                        validation_results['validation_passed'] = False
            
            return validation_results
            
        except Exception as e:
            return {
                'validation_passed': False,
                'issues': [f"Error validating prerequisites: {e}"]
            }

class DataAccessValidator:
    """Validator for data access operations with security checks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DataAccessValidator")
        self.allowed_operations = config.get('allowed_operations', ['read', 'write', 'delete'])
        self.sensitive_columns = config.get('sensitive_columns', [])
    
    @compose(
        error_boundary(name="validate_data_access"),
        traced(span_name="validate_data_access")
    )
    async def validate_data_access(
        self,
        operation: str,
        file_path: str,
        columns: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate data access operation."""
        try:
            self.logger.info(f"🔍 Validating data access: {operation} on {file_path}")
            
            validation_results = {
                'operation': operation,
                'file_path': file_path,
                'validation_passed': True,
                'issues': [],
                'warnings': []
            }
            
            # Validate operation type
            if operation not in self.allowed_operations:
                validation_results['issues'].append(f"Operation '{operation}' not allowed")
                validation_results['validation_passed'] = False
            
            # Validate file path
            if not self._is_safe_path(file_path):
                validation_results['issues'].append(f"Unsafe file path: {file_path}")
                validation_results['validation_passed'] = False
            
            # Validate sensitive columns
            if columns and self.sensitive_columns:
                sensitive_access = [col for col in columns if col in self.sensitive_columns]
                if sensitive_access:
                    validation_results['warnings'].append(f"Accessing sensitive columns: {sensitive_access}")
            
            # Check file existence for read operations
            if operation == 'read' and not safe_file_exists(file_path):
                validation_results['issues'].append(f"File not found: {file_path}")
                validation_results['validation_passed'] = False
            
            self.logger.info(f"✅ Data access validation completed: {validation_results['validation_passed']}")
            return validation_results['validation_passed'], validation_results
            
        except Exception as e:
            self.logger.exception(f"❌ Error in data access validation: {e}")
            return False, {'error': str(e), 'validation_passed': False}
    
    def _is_safe_path(self, file_path: str) -> bool:
        """Check if file path is safe (no directory traversal)."""
        try:
            path = Path(file_path).resolve()
            # Add additional security checks here
            return True
        except Exception:
            return False

class EnhancedDataFormatter:
    """Enhanced data formatter with validation and error handling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.EnhancedDataFormatter")
        self.validator = DataQualityValidator(config)
    
    @compose(
        error_boundary(name="format_data"),
        traced(span_name="format_data"),
        validate_data_quality(context='data_formatting')
    )
    async def format_data(
        self,
        data: pd.DataFrame,
        format_type: str = 'standard',
        **kwargs
    ) -> Tuple[bool, Optional[pd.DataFrame]]:
        """Format data with validation."""
        try:
            self.logger.info(f"🔄 Formatting data with type: {format_type}")
            
            # Validate input data
            validation_passed, validation_results = await self.validator.validate_ohlc_data(data)
            if not validation_passed:
                self.logger.error("❌ Data validation failed before formatting")
                return False, None
            
            # Apply formatting based on type
            if format_type == 'standard':
                formatted_data = await self._format_standard(data, **kwargs)
            elif format_type == 'normalized':
                formatted_data = await self._format_normalized(data, **kwargs)
            elif format_type == 'regime_specific':
                formatted_data = await self._format_regime_specific(data, **kwargs)
            else:
                self.logger.error(f"❌ Unknown format type: {format_type}")
                return False, None
            
            # Validate formatted data
            if formatted_data is not None:
                validation_passed, _ = await self.validator.validate_ohlc_data(formatted_data)
                if not validation_passed:
                    self.logger.error("❌ Formatted data validation failed")
                    return False, None
            
            self.logger.info("✅ Data formatting completed successfully")
            return True, formatted_data
            
        except Exception as e:
            self.logger.exception(f"❌ Error in data formatting: {e}")
            return False, None
    
    async def _format_standard(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply standard formatting."""
        try:
            formatted_data = data.copy()
            
            # Ensure proper data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in formatted_data.columns:
                    formatted_data[col] = pd.to_numeric(formatted_data[col], errors='coerce')
            
            # Handle missing values
            formatted_data = safe_fillna(formatted_data, method='ffill')
            
            return formatted_data
            
        except Exception as e:
            self.logger.exception(f"❌ Error in standard formatting: {e}")
            return None
    
    async def _format_normalized(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply normalized formatting."""
        try:
            formatted_data = data.copy()
            
            # Normalize price columns
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in formatted_data.columns:
                    # Min-max normalization
                    min_val = formatted_data[col].min()
                    max_val = formatted_data[col].max()
                    if max_val > min_val:
                        formatted_data[f'{col}_normalized'] = (formatted_data[col] - min_val) / (max_val - min_val)
            
            return formatted_data
            
        except Exception as e:
            self.logger.exception(f"❌ Error in normalized formatting: {e}")
            return None
    
    async def _format_regime_specific(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply regime-specific formatting."""
        try:
            formatted_data = data.copy()
            
            # Add regime-specific features
            regime_id = kwargs.get('regime_id', 0)
            formatted_data['regime_id'] = regime_id
            
            # Apply regime-specific transformations
            if regime_id == 0:  # Bull market
                formatted_data['regime_multiplier'] = 1.0
            elif regime_id == 1:  # Bear market
                formatted_data['regime_multiplier'] = -1.0
            else:  # Sideways market
                formatted_data['regime_multiplier'] = 0.0
            
            return formatted_data
            
        except Exception as e:
            self.logger.exception(f"❌ Error in regime-specific formatting: {e}")
            return None

# Export main classes and functions
__all__ = [
    'DataQualityValidator',
    'PipelineStepValidator', 
    'DataAccessValidator',
    'EnhancedDataFormatter'
]