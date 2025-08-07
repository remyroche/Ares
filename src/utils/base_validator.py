"""
Base validator class for training step validators.
"""

import logging
from typing import Any, Dict, Tuple
from abc import ABC, abstractmethod


class BaseValidator(ABC):
    """Base class for all step validators."""
    
    def __init__(self, step_name: str, config: Dict[str, Any]):
        self.step_name = step_name
        self.config = config
        self.logger = logging.getLogger(f"AresGlobal.{self.__class__.__name__}")
        self.validation_results = {}
    
    @abstractmethod
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> bool:
        """
        Validate a training step.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        pass
    
    def validate_error_absence(self, step_result: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that the step completed without errors.
        
        Args:
            step_result: Step result dictionary
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (passed, metrics)
        """
        try:
            # Check for errors in step result
            errors = step_result.get("errors", [])
            warnings = step_result.get("warnings", [])
            
            # Check for critical errors
            critical_errors = [e for e in errors if isinstance(e, dict) and e.get("severity") == "CRITICAL"]
            
            metrics = {
                "total_errors": len(errors),
                "total_warnings": len(warnings),
                "critical_errors": len(critical_errors),
                "has_critical_errors": len(critical_errors) > 0,
                "error_messages": errors,
                "warning_messages": warnings
            }
            
            # Step passes if no critical errors
            passed = len(critical_errors) == 0
            
            if not passed:
                self.logger.warning(f"⚠️ Step {self.step_name} has {len(critical_errors)} critical errors")
            
            return passed, metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error in error absence validation: {e}")
            return False, {"error": str(e)}
    
    def validate_file_exists(self, file_path: str, file_type: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that a file exists.
        
        Args:
            file_path: Path to the file
            file_type: Type of file for logging
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (passed, metrics)
        """
        try:
            import os
            exists = os.path.exists(file_path)
            
            metrics = {
                "file_path": file_path,
                "file_type": file_type,
                "exists": exists,
                "file_size": os.path.getsize(file_path) if exists else 0
            }
            
            if not exists:
                self.logger.error(f"❌ {file_type} file not found: {file_path}")
            
            return exists, metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error in file existence validation: {e}")
            return False, {"error": str(e)}
    
    def validate_data_quality(self, data: Any, data_name: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate basic data quality metrics.
        
        Args:
            data: Data to validate
            data_name: Name of the data for logging
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (passed, metrics)
        """
        try:
            import pandas as pd
            import numpy as np
            
            if isinstance(data, pd.DataFrame):
                metrics = {
                    "data_type": "DataFrame",
                    "shape": data.shape,
                    "columns": list(data.columns),
                    "dtypes": data.dtypes.to_dict(),
                    "null_counts": data.isnull().sum().to_dict(),
                    "memory_usage": data.memory_usage(deep=True).sum(),
                    "has_duplicates": data.duplicated().sum(),
                    "empty": data.empty
                }
                
                # Basic quality checks
                passed = (
                    not data.empty and
                    data.shape[0] > 0 and
                    data.shape[1] > 0
                )
                
            elif isinstance(data, (list, tuple)):
                metrics = {
                    "data_type": type(data).__name__,
                    "length": len(data),
                    "empty": len(data) == 0
                }
                passed = len(data) > 0
                
            elif isinstance(data, dict):
                metrics = {
                    "data_type": "dict",
                    "keys": list(data.keys()),
                    "length": len(data),
                    "empty": len(data) == 0
                }
                passed = len(data) > 0
                
            else:
                metrics = {
                    "data_type": type(data).__name__,
                    "value": str(data)[:100]  # Truncate long values
                }
                passed = data is not None
            
            if not passed:
                self.logger.warning(f"⚠️ {data_name} quality validation failed")
            
            return passed, metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error in data quality validation: {e}")
            return False, {"error": str(e)}
    
    def validate_outcome_favorability(self, step_result: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that the step outcome is favorable.
        
        Args:
            step_result: Step result dictionary
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (passed, metrics)
        """
        try:
            # Extract outcome metrics
            metrics = step_result.get("metrics", {})
            performance = step_result.get("performance", {})
            
            # Check for success indicators
            success_indicators = [
                step_result.get("success", False),
                step_result.get("completed", False),
                step_result.get("status") == "SUCCESS"
            ]
            
            # Check for error indicators
            error_indicators = [
                step_result.get("error") is not None,
                step_result.get("failed", False),
                step_result.get("status") == "FAILED"
            ]
            
            # Determine if outcome is favorable
            has_success = any(success_indicators)
            has_errors = any(error_indicators)
            
            # Outcome is favorable if there's success and no errors
            passed = has_success and not has_errors
            
            outcome_metrics = {
                "has_success_indicators": has_success,
                "has_error_indicators": has_errors,
                "success_indicators": success_indicators,
                "error_indicators": error_indicators,
                "step_metrics": metrics,
                "performance_metrics": performance,
                "favorable": passed
            }
            
            if not passed:
                self.logger.warning(f"⚠️ Step {self.step_name} outcome is not favorable")
            
            return passed, outcome_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error in outcome favorability validation: {e}")
            return False, {"error": str(e)}
