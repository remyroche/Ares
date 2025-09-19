"""
Step06 Validation Orchestrator

This module provides orchestration for step06 validation operations,
coordinating validation across different components and steps.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Callable, Union
from functools import wraps
from enum import Enum
from dataclasses import dataclass, field

# Setup logging
logger = logging.getLogger(__name__)

class ValidationPhase(Enum):
    """Validation phases for step06 operations."""
    PRE_PROCESSING = "pre_processing"
    PROCESSING = "processing"
    POST_PROCESSING = "post_processing"
    FINAL_VALIDATION = "final_validation"

class ValidationResult(Enum):
    """Validation results."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"

@dataclass
class ValidationStep:
    """Individual validation step."""
    name: str
    phase: ValidationPhase
    validator: Callable
    required: bool = True
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationReport:
    """Validation report for step06 operations."""
    step_name: str
    phase: ValidationPhase
    result: ValidationResult
    message: str
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class Step06ValidationOrchestrator:
    """Orchestrates step06 validation operations."""
    
    def __init__(self):
        self.validation_steps: List[ValidationStep] = []
        self.validation_reports: List[ValidationReport] = []
        self.current_phase: Optional[ValidationPhase] = None
    
    def add_validation_step(self, step: ValidationStep):
        """Add a validation step."""
        self.validation_steps.append(step)
        logger.info(f"Added validation step: {step.name} for phase {step.phase.value}")
    
    def run_validation_phase(self, phase: ValidationPhase) -> List[ValidationReport]:
        """Run validation for a specific phase."""
        self.current_phase = phase
        phase_reports = []
        
        phase_steps = [step for step in self.validation_steps if step.phase == phase]
        logger.info(f"Running {len(phase_steps)} validation steps for phase {phase.value}")
        
        for step in phase_steps:
            try:
                start_time = time.time()
                
                # Run validation with timeout if specified
                if step.timeout:
                    result = self._run_with_timeout(step.validator, step.timeout)
                else:
                    result = step.validator()
                
                execution_time = time.time() - start_time
                
                # Determine validation result
                if result:
                    validation_result = ValidationResult.PASS
                    message = f"Validation {step.name} passed"
                elif not result:
                    validation_result = ValidationResult.FAIL
                    message = f"Validation {step.name} failed"
                elif isinstance(result, str):
                    validation_result = ValidationResult.WARNING
                    message = result
                else:
                    validation_result = ValidationResult.PASS
                    message = f"Validation {step.name} completed"
                
                report = ValidationReport(
                    step_name=step.name,
                    phase=phase,
                    result=validation_result,
                    message=message,
                    execution_time=execution_time,
                    metadata=step.metadata
                )
                
                phase_reports.append(report)
                self.validation_reports.append(report)
                
                logger.info(f"Validation {step.name}: {validation_result.value} - {message}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                report = ValidationReport(
                    step_name=step.name,
                    phase=phase,
                    result=ValidationResult.FAIL,
                    message=f"Validation {step.name} failed with error: {str(e)}",
                    execution_time=execution_time,
                    metadata=step.metadata
                )
                
                phase_reports.append(report)
                self.validation_reports.append(report)
                
                logger.error(f"Validation {step.name} failed: {str(e)}")
                
                # If step is required, stop validation
                if step.required:
                    logger.error(f"Required validation {step.name} failed, stopping validation")
                    break
        
        return phase_reports
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run full validation across all phases."""
        logger.info("Starting full step06 validation")
        
        all_reports = []
        for phase in ValidationPhase:
            phase_reports = self.run_validation_phase(phase)
            all_reports.extend(phase_reports)
        
        # Generate summary
        summary = self._generate_validation_summary(all_reports)
        
        logger.info(f"Step06 validation completed. Summary: {summary}")
        return summary
    
    def _run_with_timeout(self, func: Callable, timeout: float) -> Any:
        """Run function with timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Validation timed out after {timeout} seconds")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func()
            return result
        finally:
            signal.alarm(0)
    
    def _generate_validation_summary(self, reports: List[ValidationReport]) -> Dict[str, Any]:
        """Generate validation summary."""
        total_steps = len(reports)
        passed = len([r for r in reports if r.result == ValidationResult.PASS])
        failed = len([r for r in reports if r.result == ValidationResult.FAIL])
        warnings = len([r for r in reports if r.result == ValidationResult.WARNING])
        
        total_time = sum(r.execution_time for r in reports)
        
        return {
            'total_steps': total_steps,
            'passed': passed,
            'failed': failed,
            'warnings': warnings,
            'success_rate': passed / total_steps if total_steps > 0 else 0,
            'total_execution_time': total_time,
            'phase_summary': self._get_phase_summary(reports)
        }
    
    def _get_phase_summary(self, reports: List[ValidationReport]) -> Dict[str, Any]:
        """Get summary by phase."""
        phase_summary = {}
        for phase in ValidationPhase:
            phase_reports = [r for r in reports if r.phase == phase]
            if phase_reports:
                phase_summary[phase.value] = {
                    'total_steps': len(phase_reports),
                    'passed': len([r for r in phase_reports if r.result == ValidationResult.PASS]),
                    'failed': len([r for r in phase_reports if r.result == ValidationResult.FAIL]),
                    'warnings': len([r for r in phase_reports if r.result == ValidationResult.WARNING])
                }
        return phase_summary
    
    def get_validation_reports(self) -> List[ValidationReport]:
        """Get all validation reports."""
        return self.validation_reports
    
    def clear_validation_reports(self):
        """Clear validation reports."""
        self.validation_reports.clear()
        logger.info("Cleared validation reports")

# Global orchestrator instance
step06_validation_orchestrator = Step06ValidationOrchestrator()

def step06_validation_decorator(phase: ValidationPhase, required: bool = True, timeout: Optional[float] = None):
    """Decorator for step06 validation functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            step = ValidationStep(
                name=func.__name__,
                phase=phase,
                validator=lambda: func(*args, **kwargs),
                required=required,
                timeout=timeout
            )
            
            step06_validation_orchestrator.add_validation_step(step)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def run_step06_validation() -> Dict[str, Any]:
    """Run step06 validation."""
    return step06_validation_orchestrator.run_full_validation()

def get_step06_validation_reports() -> List[ValidationReport]:
    """Get step06 validation reports."""
    return step06_validation_orchestrator.get_validation_reports()
