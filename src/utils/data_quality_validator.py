"""
Data Quality Validator for Enhanced Training Manager
Provides validation for data integrity, schema compliance, and quality checks
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a data validation check"""
    is_valid: bool
    message: str
    details: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


class DataQualityValidator:
    """Validates data quality and integrity for training pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DataQualityValidator")
        self.validation_results = {}
    
    def validate_dataframe(self, df: pd.DataFrame, name: str = "dataframe") -> ValidationResult:
        """Validate a pandas DataFrame for common data quality issues"""
        errors = []
        warnings = []
        details = {}
        
        try:
            # Check for null values
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                warnings.append(f"Found {null_counts.sum()} null values")
                details['null_counts'] = null_counts.to_dict()
            
            # Check for infinite values
            inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
            if inf_counts.sum() > 0:
                errors.append(f"Found {inf_counts.sum()} infinite values")
                details['inf_counts'] = inf_counts.to_dict()
            
            # Check for duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                warnings.append(f"Found {duplicate_count} duplicate rows")
                details['duplicate_count'] = duplicate_count
            
            # Check data types
            details['dtypes'] = df.dtypes.to_dict()
            
            # Check shape
            details['shape'] = df.shape
            
            # Check for constant columns
            constant_columns = []
            for col in df.columns:
                if df[col].nunique() <= 1:
                    constant_columns.append(col)
            
            if constant_columns:
                warnings.append(f"Found {len(constant_columns)} constant columns: {constant_columns}")
                details['constant_columns'] = constant_columns
            
            is_valid = len(errors) == 0
            
            result = ValidationResult(
                is_valid=is_valid,
                message=f"DataFrame '{name}' validation {'passed' if is_valid else 'failed'}",
                details=details,
                errors=errors,
                warnings=warnings
            )
            
            self.validation_results[name] = result
            self.logger.info(f"Validation result for {name}: {result.message}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error validating DataFrame '{name}': {str(e)}"
            self.logger.error(error_msg)
            return ValidationResult(
                is_valid=False,
                message=error_msg,
                details={},
                errors=[error_msg],
                warnings=[]
            )
    
    def validate_training_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate training data structure and content"""
        errors = []
        warnings = []
        details = {}
        
        try:
            # Check required keys
            required_keys = ['symbol', 'exchange', 'timeframe']
            for key in required_keys:
                if key not in data:
                    errors.append(f"Missing required key: {key}")
            
            # Validate symbol
            if 'symbol' in data:
                symbol = data['symbol']
                if not isinstance(symbol, str) or len(symbol.strip()) == 0:
                    errors.append("Symbol must be a non-empty string")
                else:
                    details['symbol'] = symbol
            
            # Validate exchange
            if 'exchange' in data:
                exchange = data['exchange']
                if not isinstance(exchange, str) or len(exchange.strip()) == 0:
                    errors.append("Exchange must be a non-empty string")
                else:
                    details['exchange'] = exchange
            
            # Validate timeframe
            if 'timeframe' in data:
                timeframe = data['timeframe']
                valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
                if timeframe not in valid_timeframes:
                    warnings.append(f"Timeframe '{timeframe}' not in standard list: {valid_timeframes}")
                details['timeframe'] = timeframe
            
            # Validate lookback_days if present
            if 'lookback_days' in data:
                lookback = data['lookback_days']
                if not isinstance(lookback, (int, float)) or lookback <= 0:
                    errors.append("lookback_days must be a positive number")
                else:
                    details['lookback_days'] = lookback
            
            is_valid = len(errors) == 0
            
            result = ValidationResult(
                is_valid=is_valid,
                message=f"Training data validation {'passed' if is_valid else 'failed'}",
                details=details,
                errors=errors,
                warnings=warnings
            )
            
            self.validation_results['training_data'] = result
            self.logger.info(f"Training data validation: {result.message}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error validating training data: {str(e)}"
            self.logger.error(error_msg)
            return ValidationResult(
                is_valid=False,
                message=error_msg,
                details={},
                errors=[error_msg],
                warnings=[]
            )
    
    def validate_pipeline_state(self, state: Dict[str, Any]) -> ValidationResult:
        """Validate pipeline state structure"""
        errors = []
        warnings = []
        details = {}
        
        try:
            # Check if state is a dictionary
            if not isinstance(state, dict):
                errors.append("Pipeline state must be a dictionary")
                return ValidationResult(
                    is_valid=False,
                    message="Invalid pipeline state type",
                    details={},
                    errors=errors,
                    warnings=warnings
                )
            
            # Check for required state keys
            required_keys = ['current_step', 'completed_steps', 'step_results']
            for key in required_keys:
                if key not in state:
                    warnings.append(f"Missing state key: {key}")
            
            # Validate current_step
            if 'current_step' in state:
                current_step = state['current_step']
                if not isinstance(current_step, (str, int)):
                    errors.append("current_step must be a string or integer")
                else:
                    details['current_step'] = current_step
            
            # Validate completed_steps
            if 'completed_steps' in state:
                completed_steps = state['completed_steps']
                if not isinstance(completed_steps, list):
                    errors.append("completed_steps must be a list")
                else:
                    details['completed_steps'] = completed_steps
            
            # Validate step_results
            if 'step_results' in state:
                step_results = state['step_results']
                if not isinstance(step_results, dict):
                    errors.append("step_results must be a dictionary")
                else:
                    details['step_results'] = step_results
            
            is_valid = len(errors) == 0
            
            result = ValidationResult(
                is_valid=is_valid,
                message=f"Pipeline state validation {'passed' if is_valid else 'failed'}",
                details=details,
                errors=errors,
                warnings=warnings
            )
            
            self.validation_results['pipeline_state'] = result
            self.logger.info(f"Pipeline state validation: {result.message}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error validating pipeline state: {str(e)}"
            self.logger.error(error_msg)
            return ValidationResult(
                is_valid=False,
                message=error_msg,
                details={},
                errors=[error_msg],
                warnings=[]
            )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results"""
        summary = {
            'total_validations': len(self.validation_results),
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'results': {}
        }
        
        for name, result in self.validation_results.items():
            summary['results'][name] = {
                'is_valid': result.is_valid,
                'message': result.message,
                'error_count': len(result.errors),
                'warning_count': len(result.warnings)
            }
            
            if result.is_valid:
                summary['passed'] += 1
            else:
                summary['failed'] += 1
            
            summary['warnings'] += len(result.warnings)
        
        return summary
    
    def clear_results(self):
        """Clear all validation results"""
        self.validation_results.clear()
        self.logger.info("Validation results cleared")
