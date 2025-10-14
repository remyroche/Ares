"""
Data Validation Stage for Unified Data-Driven Pipeline

This module handles comprehensive data validation and quality assessment
for the unified pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
from pathlib import Path

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import enhanced components
from ..enhanced_components.enhanced_statistical_framework import EnhancedStatisticalFramework
from ..enhanced_components.enhanced_schema_validation import EnhancedSchemaValidator
from ..enhanced_components.advanced_validation import AdvancedInputValidator, ValidationLevel, ValidationStatus
from ..enhanced_components.detailed_pipeline_reporter import DetailedPipelineReporter


@dataclass
class DataValidationResult:
    """Result from data validation stage."""
    
    is_valid: bool
    validation_level: ValidationLevel
    quality_score: float
    issues: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    processing_time: float
    memory_usage: float


class DataValidationStage:
    """Data validation stage for the unified pipeline."""
    
    def __init__(self, config: Any, logger: Optional[logging.Logger] = None):
        """Initialize the data validation stage.
        
        Args:
            config: Pipeline configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize validation components
        self.quality_framework = EnhancedStatisticalFramework()
        self.schema_validator = EnhancedSchemaValidator()
        self.advanced_validator = AdvancedInputValidator()
        self.detailed_reporter = DetailedPipelineReporter(outcomes_dir="outcomes")
        
        tprint_info("🔍 Data validation stage initialized")
    
    def validate_dataframe_quality(self, 
                                 data: pd.DataFrame, 
                                 context: str = "pipeline_input") -> DataValidationResult:
        """Validate DataFrame quality using comprehensive framework.
        
        Args:
            data: Input DataFrame to validate
            context: Context for validation (e.g., "pipeline_input_15m")
            
        Returns:
            DataValidationResult with validation details
        """
        start_time = time.time()
        
        try:
            tprint_info(f"🔍 Validating DataFrame quality for context: {context}")
            
            # Start detailed reporting
            self.detailed_reporter.start_step("data_validation", len(data.columns))
            
            # Perform quality validation
            quality_result = self.quality_framework.validate_dataframe_quality(
                data, context=context
            )
            
            # Calculate processing time and memory usage
            processing_time = time.time() - start_time
            memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            # Create result
            result = DataValidationResult(
                is_valid=quality_result.passed,
                validation_level=ValidationLevel.STANDARD,
                quality_score=quality_result.quality_score,
                issues=quality_result.issues,
                warnings=quality_result.warnings,
                metadata={
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'dtypes': data.dtypes.to_dict(),
                    'context': context,
                    'memory_usage_mb': memory_usage
                },
                processing_time=processing_time,
                memory_usage=memory_usage
            )
            
            # End detailed reporting
            self.detailed_reporter.end_step(
                "data_validation", 
                len(data.columns),
                processing_time,
                memory_usage,
                quality_result.passed
            )
            
            if quality_result.passed:
                tprint_success(f"✅ Data validation passed (quality score: {quality_result.quality_score:.3f})")
            else:
                tprint_warning(f"⚠️ Data validation issues detected: {len(quality_result.issues)} issues")
                for issue in quality_result.issues[:3]:  # Show first 3 issues
                    tprint_warning(f"  - {issue}")
                if len(quality_result.issues) > 3:
                    tprint_warning(f"  ... and {len(quality_result.issues) - 3} more issues")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            tprint_error(f"❌ Data validation failed: {e}")
            
            return DataValidationResult(
                is_valid=False,
                validation_level=ValidationLevel.STANDARD,
                quality_score=0.0,
                issues=[f"Validation error: {str(e)}"],
                warnings=[],
                metadata={'error': str(e), 'context': context},
                processing_time=processing_time,
                memory_usage=0.0
            )
    
    def validate_schema(self, 
                       data: pd.DataFrame, 
                       required_columns: List[str]) -> bool:
        """Validate DataFrame schema against required columns.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if schema is valid, False otherwise
        """
        try:
            return self.schema_validator.validate_schema(data, required_columns)
        except Exception as e:
            tprint_error(f"❌ Schema validation failed: {e}")
            return False
    
    def validate_data_types(self, 
                           data: pd.DataFrame, 
                           expected_types: Optional[Dict[str, str]] = None) -> bool:
        """Validate DataFrame data types.
        
        Args:
            data: DataFrame to validate
            expected_types: Optional dictionary of column names to expected types
            
        Returns:
            True if data types are valid, False otherwise
        """
        try:
            if expected_types is None:
                # Basic validation - check for numeric types
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                return len(numeric_cols) > 0
            
            # Validate specific types
            for col, expected_type in expected_types.items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    if expected_type not in actual_type:
                        tprint_warning(f"Column {col} has type {actual_type}, expected {expected_type}")
                        return False
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Data type validation failed: {e}")
            return False
    
    def validate_time_series_properties(self, 
                                      data: pd.DataFrame, 
                                      time_column: str = 'timestamp') -> bool:
        """Validate time series properties.
        
        Args:
            data: DataFrame to validate
            time_column: Name of the time column
            
        Returns:
            True if time series properties are valid, False otherwise
        """
        try:
            if time_column not in data.columns:
                tprint_warning(f"Time column '{time_column}' not found")
                return False
            
            # Check if time column is datetime
            if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
                tprint_warning(f"Time column '{time_column}' is not datetime type")
                return False
            
            # Check for monotonic time index
            if not data[time_column].is_monotonic_increasing:
                tprint_warning("Time column is not monotonically increasing")
                return False
            
            # Check for duplicate timestamps
            if data[time_column].duplicated().any():
                tprint_warning("Duplicate timestamps found")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Time series validation failed: {e}")
            return False
    
    def get_validation_summary(self, result: DataValidationResult) -> Dict[str, Any]:
        """Get a summary of validation results.
        
        Args:
            result: DataValidationResult to summarize
            
        Returns:
            Dictionary with validation summary
        """
        return {
            'is_valid': result.is_valid,
            'quality_score': result.quality_score,
            'total_issues': len(result.issues),
            'total_warnings': len(result.warnings),
            'processing_time': result.processing_time,
            'memory_usage_mb': result.memory_usage,
            'data_shape': result.metadata.get('shape', (0, 0)),
            'column_count': result.metadata.get('columns', []).__len__()
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        tprint_debug("🧹 Cleaning up data validation stage")
        # Add any cleanup logic here if needed


def create_data_validation_stage(config: Any, logger: Optional[logging.Logger] = None) -> DataValidationStage:
    """Create a data validation stage instance.
    
    Args:
        config: Pipeline configuration
        logger: Optional logger instance
        
    Returns:
        DataValidationStage instance
    """
    return DataValidationStage(config, logger)