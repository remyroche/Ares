"""Base step utilities."""
from typing import Any, Dict, List, Tuple
from src.utils.logger import system_logger
import logging

class BaseStep:
    """Base step class."""

    def __init__(self, config: Dict[str, Any], step_number: str, step_name: str) -> None:
        """Initialize base step.

        Args:
            config: Configuration dictionary
            step_number: Step number (e.g., "02")
            step_name: Step name (e.g., "data_reading")
        """
        self.config = config
        self.step_number = step_number
        self.step_name = step_name
        try:
            self.logger = system_logger.getChild(f'Step{step_number}_{step_name}')
        except Exception:
            self.logger = None

    def _initialize_step(self) -> None:
        """Hook for step-specific initialization in subclasses."""
        pass

    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Optional input validation. Subclasses may override.

        Returns (is_valid, errors).
        """
        return (True, [])

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Optional output validation. Subclasses may override.

        Returns (is_valid, errors).
        """
        return (True, [])

    async def initialize(self) -> None:
        """Async-friendly initialization wrapper."""
        try:
            self._initialize_step()
            if self.logger:
                self.logger.info('Step initialized')
        except Exception as exc:
            if self.logger:
                self.logger.exception(f'Initialization error: {exc}')
            raise

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Standard execution wrapper invoking execute_logic with validations."""
        try:
            is_valid, input_errors = self.validate_inputs(training_input, pipeline_state)
            if not is_valid:
                if self.logger:
                    self.logger.error(f'Input validation failed: {input_errors}')
                raise ValueError(f'Input validation failed: {input_errors}')
            execute_logic = getattr(self, 'execute_logic', None)
            if execute_logic is None:
                raise NotImplementedError('execute_logic is not implemented in this step')
            result = await execute_logic(training_input, pipeline_state)
            out_valid, output_errors = self.validate_outputs(pipeline_state)
            if not out_valid and self.logger:
                self.logger.warning(f'Output validation issues: {output_errors}')
            return result
        except Exception as exc:
            if self.logger:
                self.logger.exception(f'Step execution error: {exc}')
            raise