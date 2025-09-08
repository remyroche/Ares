from ..standardized_parquet_handler import standardized_parquet_handler
"""BaseStep wrapper for Step 08 Advanced Feature Selection.

This adapter wraps the heavy Step08 implementation so it fits the BaseStep
contract used by the pipeline orchestration.
"""
from typing import Dict, List, Optional, Union, Any, Tuple
from src.core.decorators import handles_errors
from src.training.base_step import BaseStep
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

class AdvancedFeatureSelectionStep(BaseStep):
    """Step 08: Advanced Feature Selection using BaseStep contract."""

    @log_important_calls
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, '08', 'advanced_feature_selection')
        try:
            from src.utils.logger import get_logger
            import logging
            self.logger = get_logger('AdvancedFeatureSelectionStep')
        except Exception:
            self.logger = None

    @log_step_functions
    def _initialize_step(self) -> None:
        self.logger.info('✅ Advanced feature selection wrapper initialized')

    @log_step_functions
    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        errors = []
        if 'engineered_data' not in pipeline_state:
            self.logger.warning('No engineered_data in memory; relying on filtered feature parquet files if available')
        for key in ['symbol', 'exchange', 'timeframe', 'data_dir']:
            if key not in training_input:
                self.logger.warning(f'Missing training_input key: {key}')
        return (len(errors) == 0, errors)

    @handles_errors(exceptions=(Exception,), default_return={'success': False}, context='advanced feature selection execution')
    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        result_state = await self._execute_feature_selection(training_input, pipeline_state)
        pipeline_state.update(result_state)
        return pipeline_state

    async def _execute_feature_selection(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the advanced feature selection step."""
        from src.training.steps.market_analysis.step08_advanced_feature_selection import Step08AdvancedFeatureSelection
        step_impl = Step08AdvancedFeatureSelection(self.config)
        result_state = await step_impl.execute(training_input, pipeline_state)
        self.logger.info('✅ Step08 legacy implementation executed')
        return result_state

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        errors: list = []
        if 'step08_advanced_feature_selection' not in pipeline_state:
            errors.append('Missing step08_advanced_feature_selection results')
        else:
            status = pipeline_state['step08_advanced_feature_selection'].get('status')
            if status != 'completed':
                errors.append('Step 08 status not completed')
        return (len(errors) == 0, errors)

    def get_required_inputs(self) -> list:
        return ['engineered_data (or feature parquet files)']

    def get_produced_outputs(self) -> list:
        return ['step08_advanced_feature_selection']

    def get_dependencies(self) -> list:
        return ['07_enhanced_matrix_operations']