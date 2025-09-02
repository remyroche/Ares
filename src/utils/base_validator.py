"""
Base Validator for Pipeline Validation

This module provides the base validator class that other validators can inherit from
for consistent validation behavior across the pipeline.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from src.utils.logger import system_logger

class BaseValidator(ABC):
    """Base validator class for pipeline validation."""
    
    def __init__(self, step_name: str, config: Dict[str, Any]) -> None:
        """Initialize BaseValidator."""
        self.step_name = step_name
        self.config = config
        self.logger = logging.getLogger(f"AresGlobal.{self.__class__.__name__}")
        self.validation_results: Dict[str, Dict[str, Any]] = {}
    
    def print(self, message: str) -> None:
        """Print a message using the logger."""
        self.logger.info(message)
    
    @abstractmethod
    async def validate(self) -> bool:
        """Validate the step. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def validate_error_absence(self, step_result: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validate that there are no critical errors in the step result."""
        try:
            errors = step_result.get("errors", [])
            warnings = step_result.get("warnings", [])
            
            critical_errors = [
                e for e in errors if isinstance(e, dict) and e.get("severity") == "CRITICAL"
            ]
            
            metrics: Dict[str, Any] = {
                "total_errors": len(errors),
                "total_warnings": len(warnings),
                "critical_errors": len(critical_errors),
                "has_critical_errors": len(critical_errors) > 0,
                "error_messages": errors,
                "warning_messages": warnings,
            }
            
            passed = len(critical_errors) == 0
            if not passed:
                self.logger.warning(
                    f"⚠️ Step {self.step_name} has {len(critical_errors)} critical errors",
                )
            
            return passed, metrics
            
        except Exception as e:  # pragma: no cover - defensive logging
            self.print(f"❌ Error in error absence validation: {e}")
            return False, {"error": str(e)}
    
    def validate_file_exists(self, file_path: str, file_type: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate that a file exists."""
        try:
            exists = os.path.exists(file_path)
            metrics: Dict[str, Any] = {
                "file_path": file_path,
                "file_type": file_type,
                "exists": exists,
            }
            
            if not exists:
                self.logger.warning(
                    f"⚠️ {file_type} not found: {file_path}",
                )
            
            return exists, metrics
            
        except Exception as e:  # pragma: no cover - defensive logging
            self.print(f"❌ Error checking file existence: {e}")
            return False, {"error": str(e)}
    
    def validate_dataframe_quality(
        self,
        df: Any,
        min_rows: int = 100,
        required_columns: Optional[List[str]] = None,
        check_data_types: bool = True,
        check_value_ranges: bool = True,
        check_duplicates: bool = True,
        check_temporal_consistency: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate dataframe quality."""
        try:
            metrics: Dict[str, Any] = {
                "total_rows": int(len(df)),
                "total_columns": int(len(df.columns)),
                "has_minimum_rows": len(df) >= min_rows,
                "missing_columns": [],
                "null_counts": {},
                "data_type_issues": {},
                "value_range_issues": {},
                "duplicate_rows": 0,
                "temporal_issues": {},
                "critical_issues": [],
            }
            
            # Check minimum rows
            if len(df) < min_rows:
                metrics["critical_issues"].append(f"Too few rows: {len(df)} < {min_rows}")
            
            # Check required columns
            if required_columns:
                missing_cols = [col for col in required_columns if col not in df.columns]
                metrics["missing_columns"] = missing_cols
                if missing_cols:
                    metrics["critical_issues"].append(f"Missing required columns: {missing_cols}")
            
            # Check for null values
            if hasattr(df, 'isnull'):
                null_counts = df.isnull().sum()
                metrics["null_counts"] = null_counts.to_dict()
                
                # Check for columns with too many nulls
                high_null_cols = null_counts[null_counts > len(df) * 0.5]  # More than 50% null
                if len(high_null_cols) > 0:
                    metrics["critical_issues"].append(f"High null columns: {list(high_null_cols.index)}")
            
            # Check for duplicates
            if check_duplicates and hasattr(df, 'duplicated'):
                duplicate_count = df.duplicated().sum()
                metrics["duplicate_rows"] = duplicate_count
                if duplicate_count > len(df) * 0.1:  # More than 10% duplicates
                    metrics["critical_issues"].append(f"Too many duplicates: {duplicate_count}")
            
            # Determine if validation passed
            passed = len(metrics["critical_issues"]) == 0
            
            return passed, metrics
            
        except Exception as e:
            self.print(f"❌ Error in dataframe quality validation: {e}")
            return False, {"error": str(e)}
    
    def validate_step_prerequisites(self, symbol: str, exchange: str, timeframe: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate step prerequisites."""
        try:
            # Basic validation - can be overridden by subclasses
            metrics = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "prerequisites_met": True,
            }
            
            # Check if required parameters are provided
            if not all([symbol, exchange, timeframe]):
                metrics["prerequisites_met"] = False
                metrics["error"] = "Missing required parameters"
                return False, metrics
            
            return True, metrics
            
        except Exception as e:
            self.print(f"❌ Error in prerequisites validation: {e}")
            return False, {"error": str(e)}
    
    def validate_step_output(self, symbol: str, exchange: str, timeframe: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate step output."""
        try:
            # Basic validation - can be overridden by subclasses
            metrics = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "output_valid": True,
            }
            
            return True, metrics
            
        except Exception as e:
            self.print(f"❌ Error in output validation: {e}")
            return False, {"error": str(e)}

# Export the main class
__all__ = ["BaseValidator"]
