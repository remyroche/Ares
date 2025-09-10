from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""BaseStep wrapper for Step 08 Advanced Feature Selection.

This adapter wraps the heavy Step08 implementation so it fits the BaseStep
contract used by the pipeline orchestration.
"""
from typing import Dict, List, Optional, Union, Any, Tuple
from src.core.decorators import handles_errors
from src.training.base_step import BaseStep
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
import time
from datetime import datetime
import logging

class AdvancedFeatureSelectionStep(BaseStep):
    """Step 08: Advanced Feature Selection using BaseStep contract."""

    @log_important_calls
    def __init__(self, config: Dict[str, Any]) -> None:
        start_time = time.time()
        
        # Initialize logger first
        try:
            from src.utils.logger import get_logger
            self.logger = get_logger('AdvancedFeatureSelectionStep')
            self.logger.info('🚀 Initializing Advanced Feature Selection Step wrapper...')
        except Exception as e:
            self.logger = logging.getLogger('AdvancedFeatureSelectionStep')
            self.logger.warning(f'⚠️ Using fallback logger: {e}')
        
        # Initialize base step
        super().__init__(config, '08', 'advanced_feature_selection')
        
        # Log initialization details
        self.logger.info(f'📋 Configuration keys: {list(config.keys()) if config else "None"}')
        self.logger.info(f'🔧 Step ID: 08, Step name: advanced_feature_selection')
        
        init_time = time.time() - start_time
        self.logger.info(f'✅ Advanced Feature Selection Step wrapper initialized in {init_time:.3f}s')

    @log_step_functions
    def _initialize_step(self) -> None:
        self.logger.info('✅ Advanced feature selection wrapper initialized')

    @log_step_functions
    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        validation_start = time.time()
        self.logger.info('🔍 Validating inputs for Advanced Feature Selection Step...')
        
        errors = []
        warnings = []
        
        # Check pipeline state
        if 'engineered_data' not in pipeline_state:
            warning_msg = 'No engineered_data in memory; relying on filtered feature parquet files if available'
            self.logger.warning(f'⚠️ {warning_msg}')
            warnings.append(warning_msg)
        else:
            self.logger.info('✅ Engineered data found in pipeline state')
        
        # Check training input keys
        required_keys = ['symbol', 'exchange', 'timeframe', 'data_dir']
        missing_keys = []
        
        for key in required_keys:
            if key not in training_input:
                error_msg = f'Missing training_input key: {key}'
                self.logger.warning(f'⚠️ {error_msg}')
                missing_keys.append(key)
                errors.append(error_msg)
            else:
                self.logger.info(f'✅ Found required key: {key} = {training_input[key]}')
        
        # Log validation summary
        validation_time = time.time() - validation_start
        is_valid = len(errors) == 0
        
        if is_valid:
            self.logger.info(f'✅ Input validation passed in {validation_time:.3f}s')
            if warnings:
                self.logger.info(f'⚠️ {len(warnings)} warnings found')
        else:
            self.logger.error(f'❌ Input validation failed in {validation_time:.3f}s')
            self.logger.error(f'❌ {len(errors)} errors found: {errors}')
        
        return (is_valid, errors)

    @handles_errors(exceptions=(Exception,), default_return={'success': False}, context='advanced feature selection execution')
    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        execution_start = time.time()
        self.logger.info('🎯 Starting Advanced Feature Selection Step execution...')
        
        try:
            # Execute the feature selection
            result_state = await self._execute_feature_selection(training_input, pipeline_state)
            
            # Update pipeline state
            pipeline_state.update(result_state)
            
            execution_time = time.time() - execution_start
            self.logger.info(f'✅ Advanced Feature Selection Step execution completed in {execution_time:.3f}s')
            
            return pipeline_state
            
        except Exception as e:
            execution_time = time.time() - execution_start
            self.logger.error(f'❌ Advanced Feature Selection Step execution failed after {execution_time:.3f}s: {e}')
            raise

    async def _execute_feature_selection(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the advanced feature selection step."""
        try:
            self.logger.info('🔧 Executing advanced feature selection implementation...')
            
            # Import and initialize the step implementation
            import_start = time.time()
            from src.training.steps.market_analysis.step08_advanced_feature_selection import Step08AdvancedFeatureSelection
            import_time = time.time() - import_start
            self.logger.info(f'📦 Step08 implementation imported in {import_time:.3f}s')
            
            # Initialize the step
            init_start = time.time()
            step_impl = Step08AdvancedFeatureSelection(self.config)
            init_time = time.time() - init_start
            self.logger.info(f'🔧 Step08 implementation initialized in {init_time:.3f}s')
            
            # Execute the step
            execution_start = time.time()
            result_state = await step_impl.execute(training_input, pipeline_state)
            execution_time = time.time() - execution_start
            
            self.logger.info(f'✅ Step08 legacy implementation executed in {execution_time:.3f}s')
            self.logger.info(f'📊 Result state keys: {list(result_state.keys()) if isinstance(result_state, dict) else "N/A"}')
            
            return result_state
            
        except ImportError as e:
            self.logger.error(f'❌ Failed to import Step08AdvancedFeatureSelection: {e}')
            raise
        except Exception as e:
            self.logger.error(f'❌ Error executing feature selection: {e}')
            self.logger.exception(f'🔍 Full error details:')
            raise

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        validation_start = time.time()
        self.logger.info('🔍 Validating outputs for Advanced Feature Selection Step...')
        
        errors = []
        
        # Check for step08 results
        if 'step08_advanced_feature_selection' not in pipeline_state:
            error_msg = 'Missing step08_advanced_feature_selection results'
            self.logger.error(f'❌ {error_msg}')
            errors.append(error_msg)
        else:
            self.logger.info('✅ Found step08_advanced_feature_selection results')
            
            # Check status
            step_results = pipeline_state['step08_advanced_feature_selection']
            status = step_results.get('status')
            
            if status != 'completed':
                error_msg = f'Step 08 status not completed: {status}'
                self.logger.error(f'❌ {error_msg}')
                errors.append(error_msg)
            else:
                self.logger.info('✅ Step 08 status is completed')
                
                # Log additional result details
                if isinstance(step_results, dict):
                    self.logger.info(f'📊 Result keys: {list(step_results.keys())}')
                    if 'selected_features' in step_results:
                        feature_count = len(step_results['selected_features']) if isinstance(step_results['selected_features'], list) else 'N/A'
                        self.logger.info(f'📈 Selected features count: {feature_count}')
        
        validation_time = time.time() - validation_start
        is_valid = len(errors) == 0
        
        if is_valid:
            self.logger.info(f'✅ Output validation passed in {validation_time:.3f}s')
        else:
            self.logger.error(f'❌ Output validation failed in {validation_time:.3f}s')
            self.logger.error(f'❌ {len(errors)} errors found: {errors}')
        
        return (is_valid, errors)

    def get_required_inputs(self) -> list:
        inputs = ['engineered_data (or feature parquet files)']
        self.logger.info(f'📋 Required inputs: {inputs}')
        return inputs

    def get_produced_outputs(self) -> list:
        outputs = ['step08_advanced_feature_selection']
        self.logger.info(f'📤 Produced outputs: {outputs}')
        return outputs

    def get_dependencies(self) -> list:
        dependencies = ['07_enhanced_matrix_operations']
        self.logger.info(f'🔗 Dependencies: {dependencies}')
        return dependencies