"""BaseStep wrapper for Step 17 Parameter Optimization.

Bridges existing parameter optimization implementations to the BaseStep interface.
"""

from typing import Any, Dict, Tuple

from src.core.decorators import handles_errors
from src.training.base_step import BaseStep
from src.utils.logger import system_logger


class ParameterOptimizationStep(BaseStep):
    """Step 17: Parameter Optimization using BaseStep contract."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "17", "parameter_optimization")

    def _initialize_step(self) -> None:
        self.logger.info("✅ Parameter optimization wrapper initialized")

    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        return True, []

    @handles_errors(
        exceptions=(Exception,),
        default_return={"success": False},
        context="parameter optimization execution",
    )
    async def execute_logic(
        self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Prefer the newer categorized implementation if available, fallback to legacy
        try:
            from src.training.steps.step17_final_parameters_optimization_new import (
                FinalParametersOptimizationStepNew,
            )
            impl = FinalParametersOptimizationStepNew(self.config)
        except Exception:
            from src.training.steps.validation.step17_final_parameters_optimization import (
                FinalParametersOptimizationStep,
            )
            impl = FinalParametersOptimizationStep(self.config)

        # Prepare compatibility input/state
        result = await impl.execute(training_input, dict(pipeline_state))

        # Normalize outputs into pipeline_state
        if isinstance(result, dict):
            # Allow both styles: direct outputs or embedded
            if "final_parameters" in result:
                pipeline_state["optimized_models"] = result["final_parameters"]
            if "optimization_report" in result:
                pipeline_state["step17_parameter_optimization_report"] = result["optimization_report"]

        return pipeline_state

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        errors: list = []
        if "optimized_models" not in pipeline_state:
            errors.append("Missing optimized_models")
        return len(errors) == 0, errors

    def get_required_inputs(self) -> list:
        return ["calibrated_models"]

    def get_produced_outputs(self) -> list:
        return ["optimized_models", "step17_parameter_optimization_report"]

    def get_dependencies(self) -> list:
        return ["16_confidence_calibration"]

