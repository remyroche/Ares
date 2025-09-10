from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""Common types and result structures for Step 4 implementations.

This module provides standardized return types and result structures
to ensure consistent error handling across all Step 4 components.
"""

from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class StepResultStatus(Enum):
    """Status enumeration for step execution results."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL_SUCCESS = "partial_success"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Standardized result structure for all Step 4 operations."""
    
    status: StepResultStatus
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    execution_time: Optional[float] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def success_result(cls, data: Any = None, metadata: Dict[str, Any] = None, 
                      execution_time: float = None, warnings: List[str] = None) -> 'StepResult':
        """Create a successful result."""
        return cls(
            status = StepResultStatus.SUCCESS,
            success = True,
            data = data,
            metadata = metadata or {},
            execution_time = execution_time,
            warnings = warnings or []
        )
    
    @classmethod
    def failure_result(cls, error: str, error_type: str = None, 
                      metadata: Dict[str, Any] = None, execution_time: float = None) -> 'StepResult':
        """Create a failure result."""
        return cls(
            status = StepResultStatus.FAILURE,
            success = False,
            error = error,
            error_type = error_type,
            metadata = metadata or {},
            execution_time = execution_time
        )
    
    @classmethod
    def partial_success_result(cls, data: Any = None, warnings: List[str] = None,
                              metadata: Dict[str, Any] = None, execution_time: float = None) -> 'StepResult':
        """Create a partial success result."""
        return cls(
            status = StepResultStatus.PARTIAL_SUCCESS,
            success = True,  # Partial success is still considered successful
            data = data,
            warnings = warnings or [],
            metadata = metadata or {},
            execution_time = execution_time
        )


@dataclass
class RegimeDataResult(StepResult):
    """Result structure specifically for regime data operations."""
    
    unified_data: Optional[pd.DataFrame] = None
    regime_stats: Optional[Dict[str, Any]] = None
    saved_path: Optional[str] = None
    regime_count: Optional[int] = None
    data_retention_ratio: Optional[float] = None


@dataclass
class TripleBarrierResult(StepResult):
    """Result structure specifically for triple barrier operations."""
    
    labeled_data: Optional[pd.DataFrame] = None
    label_stats: Optional[Dict[str, Any]] = None
    output_path: Optional[str] = None
    profit_target: Optional[float] = None
    stop_loss: Optional[float] = None
    max_holding_period: Optional[int] = None


def standardize_result(result: Union[bool, Dict[str, Any], StepResult], 
                      operation_name: str = "operation") -> StepResult:
    """Convert various result types to standardized StepResult.
    
    Args:
        result: Result from function (bool, dict, or StepResult)
        operation_name: Name of the operation for error context
        
    Returns:
        Standardized StepResult
    """
    if isinstance(result, StepResult):
        return result
    
    if isinstance(result, bool):
        if result:
            return StepResult.success_result(metadata={"operation": operation_name})
        else:
            return StepResult.failure_result(
                error = f"{operation_name} failed", 
                error_type="OperationFailure",
                metadata={"operation": operation_name}
            )
    
    if isinstance(result, dict):
        success = result.get('success', False)
        if success:
            return StepResult.success_result(
                data = result.get('data'),
                metadata = result.get('metadata', {}),
                warnings = result.get('warnings', [])
            )
        else:
            return StepResult.failure_result(
                error = result.get('error', f"{operation_name} failed"),
                error_type = result.get('error_type', "OperationFailure"),
                metadata = result.get('metadata', {})
            )
    
    # Fallback for unknown types
    return StepResult.failure_result(
        error = f"Unknown result type: {type(result)}",
        error_type="TypeError",
        metadata={"operation": operation_name, "result_type": str(type(result))}
    )