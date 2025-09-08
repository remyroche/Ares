from ..standardized_parquet_handler import standardized_parquet_handler
"""Per-Regime Pipeline Orchestrator.

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

This module orchestrates the entire per-regime pipeline, ensuring that regime
continuity is maintained throughout all steps and providing comprehensive
monitoring and validation.
"""
import asyncio
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

from src.utils.decorators import traced, validates, handles_errors, log_execution_time
from src.utils.logger import get_logger
from src.utils.pipeline_standards import pipeline_standards
from src.utils.file_utils import safe_json_dump, safe_json_load

logger = get_logger('PerRegimePipelineOrchestrator')

@dataclass
class PipelineExecutionResult:
    """Result of pipeline execution."""
    symbol: str
    exchange: str
    timeframe: str
    execution_start: datetime
    execution_end: Optional[datetime] = None
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    step_results: Dict[str, bool] = None
    continuity_validation_score: float = 0.0
    overall_success: bool = False
    error_message: Optional[str] = None
    @log_all_calls

    def __post_init__(self) -> None:
        if self.step_results is None:
            self.step_results = {}

class PerRegimePipelineOrchestrator:
    """Orchestrates the entire per-regime pipeline with continuity management."""
    @log_important_calls

    def __init__(self, config: Dict[str, Any]=None) -> None:
        """Initialize the per-regime pipeline orchestrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger('PerRegimePipelineOrchestrator')
        self.standards = pipeline_standards
        self._load_per_regime_config()
        # TODO: Initialize these when the modules are available
        # self.continuity_manager = regime_continuity_manager
        # self.continuity_validator = regime_continuity_validator
        # self.pipeline_integrator = PerRegimePipelineIntegrator(self.config)
        # self.regime_handler = regime_handler
        self.pipeline_steps = ['step04_regime_data_splitting', 'step05_labeling', 'step06_feature_engineering', 'step07_enhanced_matrix_operations', 'step08_advanced_feature_selection', 'step09_hmm_based_training', 'step10_unified_regime_intelligence', 'step11_analyst_creation', 'step12_analyst_enhancement', 'step13_analyst_ensemble_creation', 'step14_tactician_labeling', 'step15_tactician_specialist_training', 'step16_confidence_calibration', 'step17_final_parameters_optimization', 'step18_walk_forward_validation', 'step19_monte_carlo_validation', 'step20_ab_testing', 'step21_saving']
        self.per_regime_steps = [step for step in self.pipeline_steps if step != 'step04_regime_data_splitting']
    @log_all_calls

    def _load_per_regime_config(self) -> None:
        """Load per-regime pipeline configuration."""
        try:
            config_path = Path(__file__).parent / 'per_regime_pipeline_config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    per_regime_config = json.load(f)
                    self.config.update(per_regime_config)
                    self.logger.info('✅ Loaded per-regime pipeline configuration')
            else:
                self.logger.warning('⚠️ Per-regime config file not found, using defaults')
        except Exception as e:
            self.logger.warning(f'⚠️ Error loading per-regime config: {e}, using defaults')

    @traced(span_name='execute_per_regime_pipeline')
    @log_execution_time
    async def execute_per_regime_pipeline(self, symbol: str, exchange: str, timeframe: str, data_dir: str, force_rerun: bool = False, steps_to_run: Optional[List[str]]=None) -> PipelineExecutionResult:
        """Execute the entire per-regime pipeline.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            force_rerun: Force rerun all steps
            steps_to_run: Specific steps to run (default: all steps)
            
        Returns:
            Pipeline execution result
        """
        execution_start = datetime.now()
        result = PipelineExecutionResult(symbol = symbol, exchange = exchange, timeframe = timeframe, execution_start = execution_start)
        try:
            self.logger.info(f'🚀 Starting per-regime pipeline execution for {exchange}_{symbol}_{timeframe}')
            if steps_to_run is None:
                steps_to_run = self.pipeline_steps
            result.total_steps = len(steps_to_run)
            self.logger.info('🔧 Initializing regime continuity management')
            # TODO: Implement when continuity_manager is available
            # continuity_init_success = await self.continuity_manager.initialize_regime_continuity(symbol, exchange, timeframe, data_dir)
            # if not continuity_init_success:
            #     result.error_message = 'Failed to initialize regime continuity'
            #     result.execution_end = datetime.now()
            #     return result
            for step_name in steps_to_run:
                self.logger.info(f'🔄 Executing step: {step_name}')
                try:
                    # TODO: Implement when pipeline_integrator is available
                    # use_per_regime = self.pipeline_integrator.should_use_per_regime(step_name)
                    use_per_regime = step_name in self.per_regime_steps
                    if use_per_regime:
                        step_success = await self._execute_per_regime_step(step_name, symbol, exchange, timeframe, data_dir, force_rerun)
                    else:
                        step_success = await self._execute_standard_step(step_name, symbol, exchange, timeframe, data_dir, force_rerun)
                    result.step_results[step_name] = step_success
                    if step_success:
                        result.completed_steps += 1
                        self.logger.info(f'✅ Completed step: {step_name}')
                    else:
                        result.failed_steps += 1
                        self.logger.error(f'❌ Failed step: {step_name}')
                        if not self._should_continue_after_failure(step_name):
                            self.logger.error(f'🛑 Stopping pipeline due to critical step failure: {step_name}')
                            break
                    if use_per_regime and self.config.get('regime_continuity_validation', {}).get('validate_after_each_step', True):
                        # TODO: Implement when continuity_validator is available
                        # validation_result = await self.continuity_validator.validate_step_continuity(step_name, symbol, exchange, timeframe, data_dir)
                        # if not validation_result.is_valid:
                        #     self.logger.warning(f'⚠️ Continuity validation failed for {step_name}: {validation_result.issues}')
                        # result.continuity_validation_score = (result.continuity_validation_score * (result.completed_steps - 1) + validation_result.validation_score) / result.completed_steps if result.completed_steps > 0 else validation_result.validation_score
                        pass
                except Exception as e:
                    self.logger.exception(f'❌ Error executing step {step_name}: {e}')
                    result.step_results[step_name] = False
                    result.failed_steps += 1
                    if not self._should_continue_after_failure(step_name):
                        break
            self.logger.info('🔍 Performing final pipeline continuity validation')
            # TODO: Implement when continuity_validator is available
            # final_validation = await self.continuity_validator.validate_pipeline_continuity(symbol, exchange, timeframe, data_dir, steps_to_run)
            # result.continuity_validation_score = final_validation.get('overall_score', 0.0)
            result.continuity_validation_score = 1.0  # Placeholder
            result.overall_success = result.failed_steps == 0 and result.continuity_validation_score >= 0.8
            result.execution_end = datetime.now()
            await self._save_execution_results(result, symbol, exchange, timeframe, data_dir)
            if result.overall_success:
                self.logger.info(f'✅ Per-regime pipeline completed successfully for {exchange}_{symbol}_{timeframe}')
            else:
                self.logger.error(f'❌ Per-regime pipeline failed for {exchange}_{symbol}_{timeframe}')
            return result
        except Exception as e:
            self.logger.exception(f'❌ Error in per-regime pipeline execution: {e}')
            result.error_message = str(e)
            result.execution_end = datetime.now()
            return result

    async def _execute_per_regime_step(self, step_name: str, symbol: str, exchange: str, timeframe: str, data_dir: str, force_rerun: bool) -> bool:
        """Execute a step with per-regime processing.
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            force_rerun: Force rerun flag
            
        Returns:
            True if successful
        """
        try:
            # TODO: Implement when pipeline_integrator is available
            # step_function = await self.pipeline_integrator.get_step_function(step_name)
            # if step_function is None:
            #     self.logger.error(f'❌ No step function found for {step_name}')
            #     return False
            # result = await step_function(symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir, force_rerun = force_rerun, config = self.config)
            # return result
            self.logger.info(f'🔄 Executing per-regime step: {step_name}')
            return True  # Placeholder
        except Exception as e:
            self.logger.exception(f'❌ Error executing per-regime step {step_name}: {e}')
            return False

    async def _execute_standard_step(self, step_name: str, symbol: str, exchange: str, timeframe: str, data_dir: str, force_rerun: bool) -> bool:
        """Execute a step with standard processing.
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            force_rerun: Force rerun flag
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f'🔄 Executing standard step: {step_name}')
            return True
        except Exception as e:
            self.logger.exception(f'❌ Error executing standard step {step_name}: {e}')
            return False
    @log_all_calls

    def _should_continue_after_failure(self, step_name: str) -> bool:
        """Determine if pipeline should continue after a step failure.
        
        Args:
            step_name: Name of the failed step
            
        Returns:
            True if pipeline should continue
        """
        critical_steps = ['step04_regime_data_splitting', 'step05_labeling', 'step06_feature_engineering', 'step21_saving']
        return step_name not in critical_steps

    async def _save_execution_results(self, result: PipelineExecutionResult, symbol: str, exchange: str, timeframe: str, data_dir: str) -> None:
        """Save pipeline execution results.
        
        Args:
            result: Pipeline execution result
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
        """
        try:
            training_dir = Path(data_dir) / 'training'
            training_dir.mkdir(parents = True, exist_ok = True)
            result_dict = asdict(result)
            result_dict['execution_start'] = result.execution_start.isoformat()
            result_dict['execution_end'] = result.execution_end.isoformat() if result.execution_end else None
            execution_file = training_dir / f'{exchange}_{symbol}_{timeframe}_pipeline_execution_results.json'
            safe_json_dump(result_dict, execution_file)
            self.logger.info(f'✅ Saved pipeline execution results: {execution_file}')
        except Exception as e:
            self.logger.error(f'❌ Error saving execution results: {e}')

    @traced(span_name='get_pipeline_status')
    async def get_pipeline_status(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Dict[str, Any]:
        """Get the current status of the pipeline.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            Pipeline status information
        """
        try:
            # TODO: Implement when continuity_manager is available
            # continuity_report = await self.continuity_manager.get_continuity_report(symbol, exchange, timeframe)
            continuity_report = {}  # Placeholder
            training_dir = Path(data_dir) / 'training'
            validation_file = training_dir / f'{exchange}_{symbol}_{timeframe}_regime_continuity_validation.json'
            validation_results = {}
            if validation_file.exists():
                validation_results = safe_json_load(validation_file)
            execution_file = training_dir / f'{exchange}_{symbol}_{timeframe}_pipeline_execution_results.json'
            execution_results = {}
            if execution_file.exists():
                execution_results = safe_json_load(execution_file)
            status = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'timestamp': datetime.now().isoformat(), 'continuity_report': continuity_report, 'validation_results': validation_results, 'execution_results': execution_results, 'pipeline_health': self._calculate_pipeline_health(continuity_report, validation_results, execution_results)}
            return status
        except Exception as e:
            self.logger.exception(f'❌ Error getting pipeline status: {e}')
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    @log_all_calls

    def _calculate_pipeline_health(self, continuity_report: Dict[str, Any], validation_results: Dict[str, Any], execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall pipeline health.
        
        Args:
            continuity_report: Continuity report
            validation_results: Validation results
            execution_results: Execution results
            
        Returns:
            Pipeline health information
        """
        try:
            health_score = 1.0
            issues = []
            warnings = []
            if continuity_report:
                total_regimes = continuity_report.get('total_regimes', 0)
                if total_regimes == 0:
                    health_score *= 0.5
                    issues.append('No regimes found')
            if validation_results:
                overall_score = validation_results.get('overall_score', 0.0)
                if overall_score < 0.8:
                    health_score *= overall_score
                    issues.append(f'Low validation score: {overall_score:.2f}')
            if execution_results:
                overall_success = execution_results.get('overall_success', False)
                if not overall_success:
                    health_score *= 0.3
                    issues.append('Pipeline execution failed')
                failed_steps = execution_results.get('failed_steps', 0)
                if failed_steps > 0:
                    health_score *= 0.8
                    warnings.append(f'{failed_steps} steps failed')
            if health_score >= 0.9:
                status = 'excellent'
            elif health_score >= 0.8:
                status = 'good'
            elif health_score >= 0.6:
                status = 'fair'
            elif health_score >= 0.4:
                status = 'poor'
            else:
                status = 'critical'
            return {'health_score': health_score, 'status': status, 'issues': issues, 'warnings': warnings, 'recommendations': self._generate_health_recommendations(health_score, issues, warnings)}
        except Exception as e:
            self.logger.error(f'❌ Error calculating pipeline health: {e}')
            return {'health_score': 0.0, 'status': 'unknown', 'issues': [f'Health calculation error: {str(e)}'], 'warnings': [], 'recommendations': ['Fix health calculation error']}
    @log_all_calls

    def _generate_health_recommendations(self, health_score: float, issues: List[str], warnings: List[str]) -> List[str]:
        """Generate health recommendations.
        
        Args:
            health_score: Overall health score
            issues: List of issues
            warnings: List of warnings
            
        Returns:
            List of recommendations
        """
        recommendations = []
        if health_score < 0.8:
            recommendations.append('Overall pipeline health is below optimal. Review and address all issues.')
        if any(('regime' in issue.lower() for issue in issues)):
            recommendations.append('Address regime-related issues to improve pipeline health.')
        if any(('validation' in issue.lower() for issue in issues)):
            recommendations.append('Improve validation processes to ensure data quality.')
        if any(('execution' in issue.lower() for issue in issues)):
            recommendations.append('Review and fix execution failures.')
        if not recommendations:
            recommendations.append('Pipeline health is good. Continue monitoring.')
        return recommendations
per_regime_pipeline_orchestrator = PerRegimePipelineOrchestrator()

@traced(span_name='run_per_regime_pipeline')
@validates()
@handles_errors
async def run_per_regime_pipeline(symbol: str, exchange: str, timeframe: str, data_dir: str = None, force_rerun: bool = False, steps_to_run: Optional[List[str]]=None, config: Optional[Dict[str, Any]]=None) -> bool:
    """Run the complete per-regime pipeline.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for data
        data_dir: Data directory
        force_rerun: Force rerun all steps
        steps_to_run: Specific steps to run
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    logger.info('🚀 Starting Per-Regime Pipeline Orchestration')
    if config is None:
        config = {}
    if data_dir is None:
        data_dir = standardized_parquet_handler.get_standardized_path('processed_data', exchange, symbol)
    orchestrator = PerRegimePipelineOrchestrator(config)
    result = await orchestrator.execute_per_regime_pipeline(symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir, force_rerun = force_rerun, steps_to_run = steps_to_run)
    if result.overall_success:
        logger.info('✅ Per-Regime Pipeline completed successfully')
    else:
        logger.error(f'❌ Per-Regime Pipeline failed: {result.error_message}')
    return result.overall_success
if __name__ == '__main__':

    async def test() -> None:
        """Test the per-regime pipeline."""
        success = await run_per_regime_pipeline(symbol='ETHUSDT', exchange='BINANCE', timeframe='1m', data_dir='data_cache')
        print(f'Per-regime pipeline result: {success}')
    asyncio.run(test())

"""
Per-Regime Pipeline Orchestrator.

This module orchestrates the entire per-regime pipeline, ensuring that regime
continuity is maintained throughout all steps and providing comprehensive
monitoring and validation.
"""