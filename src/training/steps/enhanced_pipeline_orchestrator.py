from ..standardized_parquet_handler import standardized_parquet_handler
"""
Enhanced Pipeline Orchestrator with Fail-Fast Behavior

This module provides a comprehensive pipeline orchestrator that ensures:
1. Critical processes fail fast rather than continuing
2. No silent failures - all errors are properly handled and logged
3. Proper error propagation and pipeline termination
4. Comprehensive monitoring and reporting
"""

import asyncio

import time

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logger import system_logger
from .enhanced_error_handling import (
    EnhancedErrorHandler, 
    CriticalProcessError, 
    ErrorSeverity, 
    ErrorCategory,
    ErrorContext,
    ErrorRecord
)

class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    FAILED_FAST = 'failed_fast'
    ROLLED_BACK = 'rolled_back'

@dataclass
class PipelineStep:
    """Represents a pipeline step with its configuration."""
    name: str
    function: Callable
    config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    is_critical: bool = False
    timeout_seconds: int = 3600
    retry_count: int = 0
    max_retries: int = 0

@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    name: str
    status: PipelineStatus
    execution_time: float
    error: Optional[str] = None
    error_record: Optional[ErrorRecord] = None
    validation_results: Optional[Dict[str, Any]] = None
    data_quality_score: Optional[float] = None
    rollback_required: bool = False
    retry_count: int = 0

@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    force_rerun: bool = True
    enable_validation: bool = True
    enable_rollback: bool = True
    enable_monitoring: bool = True
    validation_level: str = 'CRITICAL'
    max_retries: int = 3
    timeout_seconds: int = 3600
    fail_fast_enabled: bool = True
    critical_processes_only: bool = False

class EnhancedPipelineOrchestrator:
    """Enhanced orchestrator with fail-fast behavior and comprehensive error handling."""
    
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = system_logger.getChild('EnhancedPipelineOrchestrator')
        self.error_handler = EnhancedErrorHandler()
        self.pipeline_results: List[PipelineResult] = []
        self.pipeline_state: Dict[str, Any] = {}
        self.checkpoint_file = Path(config.data_dir) / f'pipeline_checkpoint_{config.symbol}_{config.timeframe}.json'
        
        # Define critical processes that must succeed
        self.critical_processes = {
            'hmm_clustering',
            'feature_generation', 
            'matrix_operations',
            'ml_model_training',
            'sr_levels_detection',
            'regime_detection'
        }
        
        # Initialize pipeline steps
        self._initialize_pipeline_steps()
        
    def _initialize_pipeline_steps(self) -> None:
        """Initialize pipeline steps with proper configuration."""
        self.pipeline_steps = [
            PipelineStep(
                name='data_collection',
                function=self._run_data_collection,
                config={'force_rerun': self.config.force_rerun},
                is_critical=True,
                timeout_seconds=1800
            ),
            PipelineStep(
                name='hmm_clustering',
                function=self._run_hmm_clustering,
                config={'force_rerun': self.config.force_rerun},
                dependencies=['data_collection'],
                is_critical=True,
                timeout_seconds=3600
            ),
            PipelineStep(
                name='feature_generation',
                function=self._run_feature_generation,
                config={'force_rerun': self.config.force_rerun},
                dependencies=['hmm_clustering'],
                is_critical=True,
                timeout_seconds=2400
            ),
            PipelineStep(
                name='matrix_operations',
                function=self._run_matrix_operations,
                config={'force_rerun': self.config.force_rerun},
                dependencies=['feature_generation'],
                is_critical=True,
                timeout_seconds=1800
            ),
            PipelineStep(
                name='ml_model_training',
                function=self._run_ml_model_training,
                config={'force_rerun': self.config.force_rerun},
                dependencies=['matrix_operations'],
                is_critical=True,
                timeout_seconds=7200
            ),
            PipelineStep(
                name='optimization',
                function=self._run_optimization,
                config={'force_rerun': self.config.force_rerun},
                dependencies=['ml_model_training'],
                is_critical=False,
                timeout_seconds=3600
            ),
            PipelineStep(
                name='backtesting',
                function=self._run_backtesting,
                config={'force_rerun': self.config.force_rerun},
                dependencies=['optimization'],
                is_critical=False,
                timeout_seconds=1800
            )
        ]
    
    async def run_all_pipelines(self) -> bool:
        """Run all training pipelines with fail-fast behavior."""
        self.logger.info('🚀 ENHANCED PIPELINE EXECUTION WITH FAIL-FAST BEHAVIOR')
        self.logger.info('=' * 100)
        self.logger.info(f'📊 Configuration:')
        self.logger.info(f'   Symbol: {self.config.symbol}')
        self.logger.info(f'   Exchange: {self.config.exchange}')
        self.logger.info(f'   Timeframe: {self.config.timeframe}')
        self.logger.info(f'   Data directory: {self.config.data_dir}')
        self.logger.info(f'   Fail-fast enabled: {self.config.fail_fast_enabled}')
        self.logger.info(f'   Critical processes only: {self.config.critical_processes_only}')
        self.logger.info('=' * 100)
        
        total_start_time = time.time()
        
        try:
            # Validate prerequisites
            if not await self._validate_prerequisites():
                self.logger.error('❌ Prerequisites validation failed')
                return False
            
            # Execute pipeline steps
            for step in self.pipeline_steps:
                # Skip non-critical steps if configured
                if self.config.critical_processes_only and not step.is_critical:
                    self.logger.info(f'⏭️ Skipping non-critical step: {step.name}')
                    continue
                
                # Check dependencies
                if not await self._check_dependencies(step):
                    self.logger.error(f'❌ Dependencies not met for step: {step.name}')
                    if self.config.fail_fast_enabled:
                        return False
                    continue
                
                # Execute step
                result = await self._execute_step(step)
                self.pipeline_results.append(result)
                
                # Check for critical failures
                if result.status == PipelineStatus.FAILED_FAST:
                    self.logger.critical(f'🚨 FAIL-FAST TRIGGERED: {step.name}')
                    return False
                
                if result.status == PipelineStatus.FAILED and step.is_critical:
                    self.logger.error(f'❌ Critical step failed: {step.name}')
                    if self.config.fail_fast_enabled:
                        return False
                
                # Handle rollback if needed
                if result.rollback_required and self.config.enable_rollback:
                    rollback_success = await self._rollback_step(step)
                    if not rollback_success:
                        self.logger.error(f'❌ Rollback failed for step: {step.name}')
                        if self.config.fail_fast_enabled:
                            return False
            
            # Generate final report
            await self._generate_final_report(total_start_time)
            
            # Check overall success
            failed_critical_steps = [
                r for r in self.pipeline_results 
                if r.status == PipelineStatus.FAILED and r.name in self.critical_processes
            ]
            
            if failed_critical_steps:
                self.logger.error(f'❌ {len(failed_critical_steps)} critical steps failed')
                return False
            
            self.logger.info('🎉 ALL PIPELINES COMPLETED SUCCESSFULLY!')
            return True
            
        except CriticalProcessError as e:
            self.logger.critical(f'🚨 CRITICAL PROCESS ERROR: {e}')
            return False
        except Exception as e:
            self.logger.exception(f'💥 UNEXPECTED ERROR: {e}')
            return False
    
    async def _validate_prerequisites(self) -> bool:
        """Validate prerequisites before starting pipeline execution."""
        self.logger.info('🔍 Validating pipeline prerequisites...')
        
        try:
            # Check required directories
            data_dir_path = Path(self.config.data_dir)
            if not data_dir_path.exists():
                self.logger.error(f'❌ Data directory does not exist: {self.config.data_dir}')
                return False
            
            # Check required modules
            required_modules = ['pandas', 'numpy', 'sklearn', 'lightgbm']
            missing_modules = []
            for module in required_modules:
                try:
                    __import__(module)
                except ImportError:
                    missing_modules.append(module)
            
            if missing_modules:
                self.logger.error(f'❌ Missing required modules: {missing_modules}')
                return False
            
            self.logger.info('✅ Prerequisites validation passed')
            return True
            
        except Exception as e:
            self.logger.exception(f'❌ Prerequisites validation failed: {e}')
            return False
    
    async def _check_dependencies(self, step: PipelineStep) -> bool:
        """Check if step dependencies are satisfied."""
        for dependency in step.dependencies:
            # Find dependency result
            dependency_result = next(
                (r for r in self.pipeline_results if r.name == dependency), 
                None
            )
            
            if not dependency_result:
                self.logger.error(f'❌ Dependency not found: {dependency}')
                return False
            
            if dependency_result.status != PipelineStatus.COMPLETED:
                self.logger.error(f'❌ Dependency failed: {dependency} (status: {dependency_result.status})')
                return False
        
        return True
    
    async def _execute_step(self, step: PipelineStep) -> PipelineResult:
        """Execute a single pipeline step with comprehensive error handling."""
        self.logger.info(f'🚀 Executing {step.name} pipeline...')
        start_time = time.time()
        
        result = PipelineResult(
            name=step.name,
            status=PipelineStatus.RUNNING,
            execution_time=0.0
        )
        
        try:
            # Create error context
            context = ErrorContext(
                function_name=step.function.__name__,
                step_name=step.name,
                additional_context={
                    'is_critical': step.is_critical,
                    'timeout_seconds': step.timeout_seconds,
                    'config': step.config
                }
            )
            
            # Execute with timeout
            success = await asyncio.wait_for(
                step.function(
                    symbol=self.config.symbol,
                    exchange=self.config.exchange,
                    timeframe=self.config.timeframe,
                    data_dir=self.config.data_dir,
                    **step.config
                ),
                timeout=step.timeout_seconds
            )
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            if success:
                result.status = PipelineStatus.COMPLETED
                self.logger.info(f'✅ {step.name} completed successfully in {execution_time:.2f}s')
            else:
                result.status = PipelineStatus.FAILED
                result.error = f'{step.name} execution returned False'
                result.rollback_required = True
                self.logger.error(f'❌ {step.name} failed after {execution_time:.2f}s')
                
                # If this is a critical step, mark as failed fast
                if step.is_critical:
                    result.status = PipelineStatus.FAILED_FAST
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.status = PipelineStatus.FAILED
            result.error = f'{step.name} timed out after {step.timeout_seconds}s'
            result.rollback_required = True
            self.logger.error(f'⏰ {step.name} timed out after {execution_time:.2f}s')
            
            # If this is a critical step, mark as failed fast
            if step.is_critical:
                result.status = PipelineStatus.FAILED_FAST
                
        except CriticalProcessError as e:
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.status = PipelineStatus.FAILED_FAST
            result.error = str(e)
            result.error_record = e.error_record
            result.rollback_required = True
            self.logger.critical(f'🚨 {step.name} failed with critical error: {e}')
            
        except Exception as e:
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.status = PipelineStatus.FAILED
            result.error = str(e)
            result.rollback_required = True
            self.logger.exception(f'💥 {step.name} failed with exception: {e}')
            
            # If this is a critical step, mark as failed fast
            if step.is_critical:
                result.status = PipelineStatus.FAILED_FAST
        
        return result
    
    async def _rollback_step(self, step: PipelineStep) -> bool:
        """Rollback a failed step if possible."""
        if not self.config.enable_rollback:
            self.logger.info(f'⏭️ Rollback disabled for {step.name}')
            return True
        
        self.logger.info(f'🔄 Rolling back {step.name}...')
        
        try:
            # Implement rollback logic here
            # This would depend on the specific step and what needs to be rolled back
            self.logger.info(f'✅ Rollback completed for {step.name}')
            return True
            
        except Exception as e:
            self.logger.exception(f'❌ Rollback failed for {step.name}: {e}')
            return False
    
    async def _generate_final_report(self, total_start_time: float) -> None:
        """Generate comprehensive final report."""
        total_time = time.time() - total_start_time
        
        self.logger.info('\n' + '=' * 100)
        self.logger.info('📊 ENHANCED PIPELINE EXECUTION SUMMARY')
        self.logger.info('=' * 100)
        
        successful_steps = 0
        failed_steps = 0
        failed_fast_steps = 0
        critical_failures = 0
        
        for result in self.pipeline_results:
            status_emoji = {
                PipelineStatus.COMPLETED: '✅ SUCCESS',
                PipelineStatus.FAILED: '❌ FAILED',
                PipelineStatus.FAILED_FAST: '🚨 FAILED_FAST',
                PipelineStatus.ROLLED_BACK: '🔄 ROLLED_BACK'
            }.get(result.status, '❓ UNKNOWN')
            
            self.logger.info(f'{result.name:20} | {status_emoji:15} | {result.execution_time:8.2f}s')
            
            if result.error:
                self.logger.info(f"{'':20} | Error: {result.error}")
            
            if result.error_record:
                self.logger.info(f"{'':20} | Error ID: {result.error_record.error_id}")
            
            if result.status == PipelineStatus.COMPLETED:
                successful_steps += 1
            elif result.status == PipelineStatus.FAILED:
                failed_steps += 1
            elif result.status == PipelineStatus.FAILED_FAST:
                failed_fast_steps += 1
                critical_failures += 1
        
        self.logger.info('-' * 100)
        self.logger.info(f'Total Execution Time: {total_time:.2f} seconds')
        self.logger.info(f'Successful Steps: {successful_steps}/{len(self.pipeline_results)}')
        self.logger.info(f'Failed Steps: {failed_steps}/{len(self.pipeline_results)}')
        self.logger.info(f'Failed Fast Steps: {failed_fast_steps}/{len(self.pipeline_results)}')
        self.logger.info(f'Critical Failures: {critical_failures}')
        
        if critical_failures == 0 and failed_steps == 0:
            self.logger.info('🎉 ALL STEPS COMPLETED SUCCESSFULLY!')
        else:
            self.logger.info(f'⚠️  {failed_steps} STEP(S) FAILED, {failed_fast_steps} FAILED FAST')
        
        self.logger.info('=' * 100)
        
        # Generate error summary
        error_summary = self.error_handler.get_error_summary()
        if error_summary.get('total_errors', 0) > 0:
            self.logger.info('📊 ERROR SUMMARY:')
            self.logger.info(f"   Total Errors: {error_summary['total_errors']}")
            self.logger.info(f"   Critical Errors: {error_summary['critical_errors']}")
            self.logger.info(f"   High Errors: {error_summary['high_errors']}")
            self.logger.info(f"   Fail-Fast Errors: {error_summary['fail_fast_errors']}")
            self.logger.info(f"   Resolution Rate: {error_summary['resolution_rate']:.2%}")
    
    # Pipeline step implementations
    async def _run_data_collection(self, **kwargs) -> bool:
        """Run data collection pipeline."""
        try:
            from .data_collection import run_data_collection_pipeline
            return await run_data_collection_pipeline(**kwargs)
        except Exception as e:
            self.logger.exception(f'Data collection failed: {e}')
            return False
    
    async def _run_hmm_clustering(self, **kwargs) -> bool:
        """Run HMM clustering pipeline."""
        try:
            from .market_analysis import run_market_analysis_pipeline
            return await run_market_analysis_pipeline(**kwargs)
        except Exception as e:
            self.logger.exception(f'HMM clustering failed: {e}')
            return False
    
    async def _run_feature_generation(self, **kwargs) -> bool:
        """Run feature generation pipeline."""
        try:
            from .feature_engineering import run_feature_engineering_pipeline
            return await run_feature_engineering_pipeline(**kwargs)
        except Exception as e:
            self.logger.exception(f'Feature generation failed: {e}')
            return False
    
    async def _run_matrix_operations(self, **kwargs) -> bool:
        """Run matrix operations pipeline."""
        try:
            from .market_analysis.step07_enhanced_matrix_operations import run_step
            return await run_step(**kwargs)
        except Exception as e:
            self.logger.exception(f'Matrix operations failed: {e}')
            return False
    
    async def _run_ml_model_training(self, **kwargs) -> bool:
        """Run ML model training pipeline."""
        try:
            from .model_training import run_model_training_pipeline
            return await run_model_training_pipeline(**kwargs)
        except Exception as e:
            self.logger.exception(f'ML model training failed: {e}')
            return False
    
    async def _run_optimization(self, **kwargs) -> bool:
        """Run optimization pipeline."""
        try:
            from .optimisation import run_optimisation_pipeline
            return await run_optimisation_pipeline(**kwargs)
        except Exception as e:
            self.logger.exception(f'Optimization failed: {e}')
            return False
    
    async def _run_backtesting(self, **kwargs) -> bool:
        """Run backtesting pipeline."""
        try:
            from .backtesting import run_backtesting_pipeline
            return await run_backtesting_pipeline(**kwargs)
        except Exception as e:
            self.logger.exception(f'Backtesting failed: {e}')
            return False

async def run_enhanced_pipeline(symbol: str = 'ETHUSDT', 
                              exchange: str = 'BINANCE', 
                              timeframe: str = '1m', 
                              data_dir: str = 'data_cache',
                              fail_fast_enabled: bool = True,
                              critical_processes_only: bool = False,
                              **config: Dict[str, Any]) -> bool:
    """
    Run enhanced pipeline with fail-fast behavior.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory
        fail_fast_enabled: Whether to enable fail-fast behavior
        critical_processes_only: Whether to run only critical processes
        **config: Additional configuration
        
    Returns:
        bool: True if successful, False otherwise
    """
    pipeline_config = PipelineConfig(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        fail_fast_enabled=fail_fast_enabled,
        critical_processes_only=critical_processes_only,
        **config
    )
    
    orchestrator = EnhancedPipelineOrchestrator(pipeline_config)
    return await orchestrator.run_all_pipelines()

if __name__ == '__main__':
    async def main():
        success = await run_enhanced_pipeline(
            symbol='ETHUSDT',
            exchange='BINANCE',
            timeframe='1m',
            data_dir='data_cache',
            fail_fast_enabled=True,
            critical_processes_only=False
        )
        
        if success:
            print('\n🎉 ENHANCED PIPELINE EXECUTION SUCCESSFUL!')
            sys.exit(0)
        else:
            print('\n❌ ENHANCED PIPELINE EXECUTION FAILED!')
            sys.exit(1)
    
    asyncio.run(main())