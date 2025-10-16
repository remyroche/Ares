"""
Validation mixin for data validation and error handling.

This mixin provides comprehensive data validation, error handling,
and input sanitization capabilities for all features_common components.
"""

import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import pandas as pd
import numpy as np

from ..config import get_unified_config

logger = logging.getLogger(__name__)

class ValidationMixin:
    """
    Mixin class providing data validation and error handling.
    
    This mixin can be added to any class to provide comprehensive
    data validation, input sanitization, and error handling capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize validation mixin."""
        super().__init__(*args, **kwargs)
        
        # Get unified configuration
        self.config = get_unified_config()
        
        # Validation statistics
        self._validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'warnings_issued': 0,
            'errors_handled': 0
        }
        
        # Validation rules
        self._validation_rules = {
            'check_numeric': True,
            'check_finite': True,
            'check_shape': True,
            'check_index': True,
            'check_dtypes': True,
            'warn_on_na': True,
            'strict_mode': False
        }
    
    def validate_data(self, 
                     data: Union[pd.Series, pd.DataFrame],
                     data_name: str = "data",
                     **validation_options) -> Tuple[bool, List[str]]:
        """
        Validate input data with comprehensive checks.
        
        Args:
            data: Data to validate
            data_name: Name of the data for error messages
            **validation_options: Additional validation options
            
        Returns:
            Tuple of (is_valid, list_of_warnings)
            
        Raises:
            ValueError: If critical validation fails
        """
        from ..utils import TPRINT_AVAILABLE, tprint
        
        if TPRINT_AVAILABLE:
            tprint(f"🔍 [ValidationMixin] Starting validation for {data_name}", color="cyan")
        
        self._validation_stats['total_validations'] += 1
        
        warnings = []
        is_valid = True
        
        try:
            # Check if data is empty
            if TPRINT_AVAILABLE:
                tprint(f"🔍 [ValidationMixin] Checking if {data_name} is empty", color="blue")
            
            if self._is_empty(data):
                warning_msg = f"{data_name} is empty"
                warnings.append(warning_msg)
                if TPRINT_AVAILABLE:
                    tprint(f"⚠️  [ValidationMixin] {warning_msg}", color="yellow")
                if self._validation_rules['strict_mode']:
                    is_valid = False
                    if TPRINT_AVAILABLE:
                        tprint(f"❌ [ValidationMixin] Empty data in strict mode", color="red")
            
            # Check data type
            if TPRINT_AVAILABLE:
                tprint(f"🔍 [ValidationMixin] Checking data type for {data_name}", color="blue")
            
            if not isinstance(data, (pd.Series, pd.DataFrame)):
                warning_msg = f"{data_name} must be a pandas Series or DataFrame"
                warnings.append(warning_msg)
                if TPRINT_AVAILABLE:
                    tprint(f"❌ [ValidationMixin] {warning_msg}", color="red")
                is_valid = False
            
            # Check for numeric data
            if self._validation_rules['check_numeric']:
                if TPRINT_AVAILABLE:
                    tprint(f"🔍 [ValidationMixin] Checking numeric data for {data_name}", color="blue")
                
                if isinstance(data, pd.Series):
                    if not pd.api.types.is_numeric_dtype(data):
                        warning_msg = f"{data_name} must be numeric"
                        warnings.append(warning_msg)
                        if TPRINT_AVAILABLE:
                            tprint(f"⚠️  [ValidationMixin] {warning_msg}", color="yellow")
                        if self._validation_rules['strict_mode']:
                            is_valid = False
                            if TPRINT_AVAILABLE:
                                tprint(f"❌ [ValidationMixin] Non-numeric data in strict mode", color="red")
                elif isinstance(data, pd.DataFrame):
                    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
                    if non_numeric_cols:
                        warning_msg = f"{data_name} contains non-numeric columns: {non_numeric_cols}"
                        warnings.append(warning_msg)
                        if TPRINT_AVAILABLE:
                            tprint(f"⚠️  [ValidationMixin] {warning_msg}", color="yellow")
                        if self._validation_rules['strict_mode']:
                            is_valid = False
                            if TPRINT_AVAILABLE:
                                tprint(f"❌ [ValidationMixin] Non-numeric columns in strict mode", color="red")
            
            # Check for finite values
            if self._validation_rules['check_finite']:
                if isinstance(data, pd.Series):
                    if not np.isfinite(data).all():
                        inf_count = np.isinf(data).sum()
                        nan_count = data.isna().sum()
                        warnings.append(f"{data_name} contains {inf_count} infinite and {nan_count} NaN values")
                        if self._validation_rules['strict_mode']:
                            is_valid = False
                elif isinstance(data, pd.DataFrame):
                    inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
                    nan_count = data.select_dtypes(include=[np.number]).isna().sum().sum()
                    if inf_count > 0 or nan_count > 0:
                        warnings.append(f"{data_name} contains {inf_count} infinite and {nan_count} NaN values")
                        if self._validation_rules['strict_mode']:
                            is_valid = False
            
            # Check data shape
            if self._validation_rules['check_shape']:
                if isinstance(data, pd.Series):
                    if len(data) == 0:
                        warnings.append(f"{data_name} has zero length")
                        is_valid = False
                elif isinstance(data, pd.DataFrame):
                    if data.shape[0] == 0 or data.shape[1] == 0:
                        warnings.append(f"{data_name} has zero dimensions: {data.shape}")
                        is_valid = False
            
            # Check index
            if self._validation_rules['check_index']:
                if isinstance(data, pd.Series):
                    if not isinstance(data.index, pd.Index):
                        warnings.append(f"{data_name} index is not a pandas Index")
                        if self._validation_rules['strict_mode']:
                            is_valid = False
                elif isinstance(data, pd.DataFrame):
                    if not isinstance(data.index, pd.Index):
                        warnings.append(f"{data_name} index is not a pandas Index")
                        if self._validation_rules['strict_mode']:
                            is_valid = False
            
            # Check data types
            if self._validation_rules['check_dtypes']:
                if isinstance(data, pd.Series):
                    if data.dtype == 'object':
                        warnings.append(f"{data_name} has object dtype - may cause performance issues")
                elif isinstance(data, pd.DataFrame):
                    object_cols = data.select_dtypes(include=['object']).columns.tolist()
                    if object_cols:
                        warnings.append(f"{data_name} has object dtype columns: {object_cols}")
            
            # Check for NA values
            if self._validation_rules['warn_on_na']:
                if isinstance(data, pd.Series):
                    na_count = data.isna().sum()
                    if na_count > 0:
                        warnings.append(f"{data_name} contains {na_count} NA values")
                elif isinstance(data, pd.DataFrame):
                    na_count = data.isna().sum().sum()
                    if na_count > 0:
                        warnings.append(f"{data_name} contains {na_count} NA values")
            
            # Update statistics
            if is_valid:
                self._validation_stats['successful_validations'] += 1
            else:
                self._validation_stats['failed_validations'] += 1
            
            if warnings:
                self._validation_stats['warnings_issued'] += len(warnings)
            
            return is_valid, warnings
            
        except Exception as e:
            error_msg = f"Validation failed for {data_name}: {e}"
            logger.error(error_msg)
            self._validation_stats['errors_handled'] += 1
            self._validation_stats['failed_validations'] += 1
            return False, [error_msg]
    
    def _is_empty(self, data: Union[pd.Series, pd.DataFrame]) -> bool:
        """Check if data is empty."""
        if isinstance(data, pd.Series):
            return len(data) == 0
        elif isinstance(data, pd.DataFrame):
            return data.shape[0] == 0 or data.shape[1] == 0
        else:
            return True
    
    def sanitize_data(self, 
                     data: Union[pd.Series, pd.DataFrame],
                     data_name: str = "data",
                     **sanitization_options) -> Union[pd.Series, pd.DataFrame]:
        """
        Sanitize input data by handling common issues.
        
        Args:
            data: Data to sanitize
            data_name: Name of the data for logging
            **sanitization_options: Additional sanitization options
            
        Returns:
            Sanitized data
        """
        try:
            sanitized_data = data.copy()
            
            # Handle infinite values
            if isinstance(sanitized_data, pd.Series):
                if not np.isfinite(sanitized_data).all():
                    inf_count = np.isinf(sanitized_data).sum()
                    logger.warning(f"Replacing {inf_count} infinite values in {data_name}")
                    sanitized_data = sanitized_data.replace([np.inf, -np.inf], np.nan)
            
            elif isinstance(sanitized_data, pd.DataFrame):
                numeric_cols = sanitized_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if not np.isfinite(sanitized_data[col]).all():
                        inf_count = np.isinf(sanitized_data[col]).sum()
                        logger.warning(f"Replacing {inf_count} infinite values in {data_name}[{col}]")
                        sanitized_data[col] = sanitized_data[col].replace([np.inf, -np.inf], np.nan)
            
            # Handle data types
            if sanitization_options.get('convert_to_numeric', True):
                if isinstance(sanitized_data, pd.Series):
                    if not pd.api.types.is_numeric_dtype(sanitized_data):
                        try:
                            sanitized_data = pd.to_numeric(sanitized_data, errors='coerce')
                            logger.info(f"Converted {data_name} to numeric")
                        except Exception as e:
                            logger.warning(f"Failed to convert {data_name} to numeric: {e}")
                
                elif isinstance(sanitized_data, pd.DataFrame):
                    for col in sanitized_data.columns:
                        if not pd.api.types.is_numeric_dtype(sanitized_data[col]):
                            try:
                                sanitized_data[col] = pd.to_numeric(sanitized_data[col], errors='coerce')
                                logger.info(f"Converted {data_name}[{col}] to numeric")
                            except Exception as e:
                                logger.warning(f"Failed to convert {data_name}[{col}] to numeric: {e}")
            
            # Handle NA values
            if sanitization_options.get('handle_na', True):
                na_strategy = sanitization_options.get('na_strategy', 'warn')
                
                if isinstance(sanitized_data, pd.Series):
                    na_count = sanitized_data.isna().sum()
                    if na_count > 0:
                        if na_strategy == 'warn':
                            logger.warning(f"{data_name} contains {na_count} NA values")
                        elif na_strategy == 'drop':
                            sanitized_data = sanitized_data.dropna()
                            logger.info(f"Dropped {na_count} NA values from {data_name}")
                        elif na_strategy == 'fill':
                            fill_value = sanitization_options.get('fill_value', 0)
                            sanitized_data = sanitized_data.fillna(fill_value)
                            logger.info(f"Filled {na_count} NA values in {data_name} with {fill_value}")
                
                elif isinstance(sanitized_data, pd.DataFrame):
                    na_count = sanitized_data.isna().sum().sum()
                    if na_count > 0:
                        if na_strategy == 'warn':
                            logger.warning(f"{data_name} contains {na_count} NA values")
                        elif na_strategy == 'drop':
                            sanitized_data = sanitized_data.dropna()
                            logger.info(f"Dropped rows with NA values from {data_name}")
                        elif na_strategy == 'fill':
                            fill_value = sanitization_options.get('fill_value', 0)
                            sanitized_data = sanitized_data.fillna(fill_value)
                            logger.info(f"Filled {na_count} NA values in {data_name} with {fill_value}")
            
            return sanitized_data
            
        except Exception as e:
            logger.error(f"Data sanitization failed for {data_name}: {e}")
            return data
    
    def validate_and_sanitize(self, 
                             data: Union[pd.Series, pd.DataFrame],
                             data_name: str = "data",
                             **options) -> Tuple[Union[pd.Series, pd.DataFrame], bool, List[str]]:
        """
        Validate and sanitize data in one operation.
        
        Args:
            data: Data to validate and sanitize
            data_name: Name of the data for error messages
            **options: Additional options for validation and sanitization
            
        Returns:
            Tuple of (sanitized_data, is_valid, warnings)
        """
        # First validate
        is_valid, warnings = self.validate_data(data, data_name, **options)
        
        # Then sanitize
        sanitized_data = self.sanitize_data(data, data_name, **options)
        
        return sanitized_data, is_valid, warnings
    
    def set_validation_rules(self, **rules) -> None:
        """Set validation rules."""
        for key, value in rules.items():
            if key in self._validation_rules:
                self._validation_rules[key] = value
            else:
                logger.warning(f"Unknown validation rule: {key}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = self._validation_stats.copy()
        
        # Calculate success rate
        if stats['total_validations'] > 0:
            stats['success_rate'] = stats['successful_validations'] / stats['total_validations']
            stats['failure_rate'] = stats['failed_validations'] / stats['total_validations']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        return stats
    
    def reset_validation_stats(self) -> None:
        """Reset validation statistics."""
        self._validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'warnings_issued': 0,
            'errors_handled': 0
        }
    
    def get_validation_recommendations(self) -> List[str]:
        """Get recommendations for improving data quality."""
        recommendations = []
        stats = self.get_validation_stats()
        
        # Check success rate
        if stats['success_rate'] < 0.8:
            recommendations.append("Low validation success rate - check data quality")
        
        # Check warning rate
        if stats['warnings_issued'] > stats['total_validations'] * 0.5:
            recommendations.append("High warning rate - consider data preprocessing")
        
        # Check error rate
        if stats['errors_handled'] > 0:
            recommendations.append("Validation errors detected - check error logs")
        
        return recommendations