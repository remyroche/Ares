"""BaseValidationStep wrapper for Step 20 A/B Testing."""

from typing import Any, Dict, List, Tuple

from src.core.decorators import handles_errors
from .base_validation_step import BaseValidationStep


class ABTestingStep(BaseValidationStep):
    """Step 20: A/B Testing using BaseValidationStep contract."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "20", "ab_testing")

    def _initialize_step(self) -> None:
        self.logger.info("✅ AB testing wrapper initialized")

    def _validate_step_specific_inputs(
        self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]
    ) -> List[str]:
        # Minimal checks, as AB testing can operate on summary metrics
        return []

    @handles_errors(
        exceptions=(Exception,),
        default_return={"success": False},
        context="ab testing execution",
    )
    async def execute_logic(
        self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        from src.training.steps.validation.step20_ab_testing import ABTestingStep as Impl
from src.core.decorators.errors import handles_errors

        impl = Impl(self.config)
        await impl.initialize()
        res = await impl.execute(training_input, {})

        # Map outputs into standardized keys
        results_key = f"{self.full_step_name}_results"
        summary_key = f"{self.full_step_name}_summary"

        pipeline_state[results_key] = {
            "status": res.get("status"),
            "results_file": res.get("results_file"),
        }
        pipeline_state[summary_key] = {
            "winner": "B",
            "notes": "Default summary; extend with actual analysis if available",
        }

        return pipeline_state

    def _validate_step_specific_outputs(self, pipeline_state: Dict[str, Any]) -> List[str]:
        return []

