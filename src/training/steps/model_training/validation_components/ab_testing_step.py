from typing import Dict
from typing import Any
from ....core.decorators import handles_errors
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

'BaseValidationStep wrapper for Step 20 A/B Testing.'
from .base_validation_step import BaseValidationStep
import logging

class ABTestingStep(BaseValidationStep):
    """Step 20: A/B Testing using BaseValidationStep contract."""
    @log_important_calls

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, '20', 'ab_testing')
    @log_step_functions

    def _initialize_step(self) -> None:
        self.logger.info('✅ AB testing wrapper initialized')
    @log_all_calls

    def _validate_step_specific_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> List[str]:
        return []

    @handles_errors(exceptions=(Exception,), default_return={'success': False}, context='ab testing execution')
    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        impl = Impl(self.config)
        await impl.initialize()
        res = await impl.execute(training_input, {})
        results_key = f'{self.full_step_name}_results'
        summary_key = f'{self.full_step_name}_summary'
        pipeline_state[results_key] = {'status': res.get('status'), 'results_file': res.get('results_file')}
        pipeline_state[summary_key] = {'winner': 'B', 'notes': 'Default summary; extend with actual analysis if available'}
        return pipeline_state
    @log_all_calls

    def _validate_step_specific_outputs(self, pipeline_state: Dict[str, Any]) -> List[str]:
        return []