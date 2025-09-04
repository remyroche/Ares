#!/usr/bin/env python3
"""
Pipeline Validation Utilities

This module provides validation utilities for the model training pipeline
using the existing core decorators and common operations.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from src.core.decorators import (
    handles_errors,
    retry,
    timeout,
    log_execution_time,
    traced,
    validates,
    validate_dataframe,
)
from src.utils.common_operations import (
    validate_dataframe_integrity,
    validate_pipeline_step_output,
    safe_file_exists,
    ensure_directory,
)
from src.utils.logger import system_logger


class PipelineValidator:
    """Pipeline validator using existing decorators and utilities."""
    
    def __init__(self):
        self.logger = system_logger.getChild("PipelineValidator")
        self.validation_results = []
    
    @handles_errors(
        fallback=False,
        log_level="ERROR",
        include_traceback=True
    )
    @retry(
        max_attempts=3,
        backoff_factor=1.5,
        exceptions=(ValueError, FileNotFoundError)
    )
    @timeout(seconds=300)  # 5 minute timeout
    @log_execution_time
    @traced
    @validates(strict=True)
    async def validate_data_loading(
        self, 
        symbol: str, 
        exchange: str, 
        data_dir: str
    ) -> Dict[str, Any]:
        """Validate data loading step."""
        self.logger.info(f"Validating data loading for {symbol} on {exchange}")
        
        # Check data directory exists
        if not safe_file_exists(data_dir):
            ensure_directory(data_dir)
            self.logger.info(f"Created data directory: {data_dir}")
        
        # Check for required data files
        required_files = [
            f"aggtrades_{exchange}_{symbol}_consolidated.parquet",
            f"volume_{exchange}_{symbol}_consolidated.parquet"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = f"{data_dir}/{file_name}"
            if not safe_file_exists(file_path):
                missing_files.append(file_name)
        
        validation_result = {
            'step': 'data_loading',
            'symbol': symbol,
            'exchange': exchange,
            'data_dir': data_dir,
            'missing_files': missing_files,
            'is_valid': len(missing_files) == 0,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if missing_files:
            self.logger.warning(f"Missing data files: {missing_files}")
        else:
            self.logger.info("Data loading validation passed")
        
        self.validation_results.append(validation_result)
        return validation_result
    
    @handles_errors(
        fallback=False,
        log_level="ERROR",
        include_traceback=True
    )
    @retry(
        max_attempts=2,
        backoff_factor=2.0,
        exceptions=(ValueError, pd.errors.EmptyDataError)
    )
    @timeout(seconds=180)  # 3 minute timeout
    @log_execution_time
    @traced
    @validates(strict=True)
    @validate_dataframe
    async def validate_data_quality(
        self, 
        df: pd.DataFrame, 
        required_columns: List[str] = None
    ) -> Dict[str, Any]:
        """Validate data quality using existing utilities."""
        self.logger.info("Validating data quality")
        
        # Use existing validation utility
        integrity_results = validate_dataframe_integrity(df, required_columns)
        
        validation_result = {
            'step': 'data_quality',
            'is_valid': integrity_results['is_valid'],
            'errors': integrity_results['errors'],
            'warnings': integrity_results['warnings'],
            'statistics': integrity_results['statistics'],
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if integrity_results['is_valid']:
            self.logger.info("Data quality validation passed")
        else:
            self.logger.error(f"Data quality validation failed: {integrity_results['errors']}")
        
        self.validation_results.append(validation_result)
        return validation_result
    
    @handles_errors(
        fallback=False,
        log_level="ERROR",
        include_traceback=True
    )
    @retry(
        max_attempts=2,
        backoff_factor=1.5,
        exceptions=(ValueError, KeyError)
    )
    @timeout(seconds=120)  # 2 minute timeout
    @log_execution_time
    @traced
    @validates(strict=True)
    async def validate_model_training_output(
        self, 
        training_result: Dict[str, Any], 
        expected_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Validate model training output."""
        self.logger.info("Validating model training output")
        
        if expected_metrics is None:
            expected_metrics = ['accuracy', 'loss']
        
        validation_result = {
            'step': 'model_training_output',
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'missing_metrics': [],
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Check if training result is valid
        if not isinstance(training_result, dict):
            validation_result['errors'].append("Training result must be a dictionary")
            validation_result['is_valid'] = False
        else:
            # Check for required metrics
            for metric in expected_metrics:
                if metric not in training_result:
                    validation_result['missing_metrics'].append(metric)
                    validation_result['warnings'].append(f"Missing metric: {metric}")
            
            # Validate metric values
            for metric_name, metric_value in training_result.items():
                if isinstance(metric_value, (int, float)):
                    if np.isnan(metric_value) or np.isinf(metric_value):
                        validation_result['errors'].append(f"Invalid metric value for {metric_name}: {metric_value}")
                        validation_result['is_valid'] = False
        
        if validation_result['is_valid']:
            self.logger.info("Model training output validation passed")
        else:
            self.logger.error(f"Model training output validation failed: {validation_result['errors']}")
        
        self.validation_results.append(validation_result)
        return validation_result
    
    @handles_errors(
        fallback=False,
        log_level="ERROR",
        include_traceback=True
    )
    @timeout(seconds=60)  # 1 minute timeout
    @log_execution_time
    @traced
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        if not self.validation_results:
            return {
                'total_validations': 0,
                'passed': 0,
                'failed': 0,
                'success_rate': 0.0
            }
        
        total_validations = len(self.validation_results)
        passed = sum(1 for result in self.validation_results if result.get('is_valid', False))
        failed = total_validations - passed
        success_rate = passed / total_validations if total_validations > 0 else 0.0
        
        summary = {
            'total_validations': total_validations,
            'passed': passed,
            'failed': failed,
            'success_rate': success_rate,
            'validation_results': self.validation_results
        }
        
        self.logger.info(f"Validation summary: {passed}/{total_validations} passed ({success_rate:.2%})")
        return summary


# Global validator instance
pipeline_validator = PipelineValidator()


# Convenience functions using the validator
async def validate_pipeline_step(
    step_name: str, 
    data: Any, 
    validation_type: str = "general",
    **kwargs
) -> Dict[str, Any]:
    """Validate a pipeline step using the appropriate validator."""
    if validation_type == "data_loading":
        return await pipeline_validator.validate_data_loading(**kwargs)
    elif validation_type == "data_quality":
        return await pipeline_validator.validate_data_quality(data, **kwargs)
    elif validation_type == "model_training_output":
        return await pipeline_validator.validate_model_training_output(data, **kwargs)
    else:
        # General validation
        return {
            'step': step_name,
            'is_valid': data is not None,
            'validation_type': validation_type,
            'timestamp': pd.Timestamp.now().isoformat()
        }


def get_pipeline_validation_summary() -> Dict[str, Any]:
    """Get the current validation summary."""
    return pipeline_validator.get_validation_summary()