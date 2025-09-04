#!/usr/bin/env python3
"""
Pipeline Integration and Orchestration

This module provides the main integration point for all pipeline enhancements,
ensuring effective execution with comprehensive validation, monitoring, and protection.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.core.decorators.errors import handles_errors, error_boundary
from src.core.decorators.logging import logs_execution
from src.utils.pipeline_validator_framework import (
    validator_orchestrator,
    ValidationLevel,
    ValidationResult
)
from src.utils.pipeline_decorators import (
    pipeline_step,
    step_dependency_check,
    performance_monitor
)
from src.utils.pipeline_utilities import (
    pipeline_utilities,
    DataFormat,
    DataMetadata
)
from src.utils.pipeline_state_manager import (
    pipeline_state_manager,
    PipelineState,
    CheckpointStatus
)
from src.utils.pipeline_monitoring import (
    pipeline_monitor,
    LogLevel,
    MetricType
)
from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    safe_file_exists,
    ensure_directory
)


class EnhancedPipelineIntegration:
    """Main integration class for enhanced pipeline execution."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str = "1m", data_dir: str = "data_cache"):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.data_dir = data_dir
        self.logger = logging.getLogger("enhanced_pipeline_integration")
        self.pipeline_id = f"{symbol}_{exchange}_{timeframe}_{int(time.time())}"
        
        # Initialize all components
        self._initialize_components()
    
    @handles_errors(Exception, fallback=False)
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        
        try:
            # Start monitoring
            if not pipeline_monitor.start_monitoring():
                self.logger.warning("Failed to start pipeline monitoring")
            
            # Initialize pipeline state
            self.pipeline_state = pipeline_state_manager.initialize_pipeline(
                pipeline_id=self.pipeline_id,
                pipeline_name=f"Enhanced Pipeline {self.symbol}",
                configuration={
                    "symbol": self.symbol,
                    "exchange": self.exchange,
                    "timeframe": self.timeframe,
                    "data_dir": self.data_dir
                }
            )
            
            # Record pipeline start
            pipeline_monitor.record_pipeline_event(
                event_type="pipeline_started",
                pipeline_id=self.pipeline_id,
                message=f"Enhanced pipeline started for {self.symbol} on {self.exchange}",
                context={
                    "timeframe": self.timeframe,
                    "data_dir": self.data_dir
                }
            )
            
            self.logger.info(f"Enhanced pipeline integration initialized for {self.pipeline_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    @handles_errors(Exception, fallback=False)
    @logs_execution("enhanced_pipeline_execution")
    async def execute_enhanced_pipeline(
        self,
        pipeline_steps: List[Dict[str, Any]],
        **config: Dict[str, Any]
    ) -> bool:
        """Execute the enhanced pipeline with comprehensive validation and monitoring."""
        
        try:
            # Update pipeline state to running
            pipeline_state_manager.update_pipeline_state(
                self.pipeline_id,
                PipelineState.RUNNING
            )
            
            total_steps = len(pipeline_steps)
            successful_steps = 0
            failed_steps = 0
            
            self.logger.info(f"Starting enhanced pipeline execution with {total_steps} steps")
            
            for step_index, step_config in enumerate(pipeline_steps):
                step_name = step_config.get("name", f"step_{step_index}")
                step_func = step_config.get("function")
                step_dependencies = step_config.get("dependencies", [])
                validation_level = ValidationLevel(step_config.get("validation_level", "critical"))
                
                if not step_func:
                    self.logger.error(f"No function provided for step {step_name}")
                    failed_steps += 1
                    continue
                
                # Update current step
                pipeline_state_manager.update_pipeline_state(
                    self.pipeline_id,
                    PipelineState.RUNNING,
                    current_step=step_name,
                    progress_percentage=(step_index / total_steps) * 100
                )
                
                # Record step start
                pipeline_monitor.record_pipeline_event(
                    event_type="step_started",
                    pipeline_id=self.pipeline_id,
                    step_name=step_name,
                    message=f"Starting step {step_name}",
                    context={"step_index": step_index, "total_steps": total_steps}
                )
                
                step_start_time = time.time()
                
                try:
                    # Execute step with comprehensive validation
                    success = await self._execute_step_with_validation(
                        step_name=step_name,
                        step_func=step_func,
                        step_dependencies=step_dependencies,
                        validation_level=validation_level,
                        step_config=step_config,
                        **config
                    )
                    
                    step_execution_time = time.time() - step_start_time
                    
                    if success:
                        successful_steps += 1
                        
                        # Record step success
                        pipeline_monitor.record_step_performance(
                            pipeline_id=self.pipeline_id,
                            step_name=step_name,
                            execution_time=step_execution_time,
                            success=True
                        )
                        
                        # Create checkpoint
                        checkpoint_id = pipeline_state_manager.create_step_checkpoint(
                            pipeline_id=self.pipeline_id,
                            step_name=step_name,
                            data={"step_completed": True, "execution_time": step_execution_time},
                            dependencies=step_dependencies,
                            validation_results=await self._get_step_validation_results(step_name),
                            performance_metrics={"execution_time": step_execution_time}
                        )
                        
                        if checkpoint_id:
                            self.logger.info(f"Created checkpoint {checkpoint_id} for step {step_name}")
                        
                        self.logger.info(f"✅ Step {step_name} completed successfully in {step_execution_time:.2f}s")
                        
                    else:
                        failed_steps += 1
                        
                        # Record step failure
                        pipeline_monitor.record_step_performance(
                            pipeline_id=self.pipeline_id,
                            step_name=step_name,
                            execution_time=step_execution_time,
                            success=False
                        )
                        
                        # Record error
                        pipeline_state_manager.update_pipeline_state(
                            self.pipeline_id,
                            PipelineState.FAILED,
                            current_step=step_name,
                            error={"message": f"Step {step_name} returned False", "execution_time": step_execution_time}
                        )
                        
                        self.logger.error(f"❌ Step {step_name} failed after {step_execution_time:.2f}s")
                        
                        # Check if we should continue or stop
                        if step_config.get("critical", True):
                            self.logger.error(f"Critical step {step_name} failed, stopping pipeline")
                            break
                        else:
                            self.logger.warning(f"Non-critical step {step_name} failed, continuing pipeline")
                
                except Exception as e:
                    step_execution_time = time.time() - step_start_time
                    failed_steps += 1
                    
                    # Record step failure with exception
                    pipeline_monitor.record_step_performance(
                        pipeline_id=self.pipeline_id,
                        step_name=step_name,
                        execution_time=step_execution_time,
                        success=False
                    )
                    
                    # Record error
                    pipeline_state_manager.update_pipeline_state(
                        self.pipeline_id,
                        PipelineState.FAILED,
                        current_step=step_name,
                        error={"message": str(e), "exception": type(e).__name__, "execution_time": step_execution_time}
                    )
                    
                    self.logger.error(f"💥 Step {step_name} failed with exception: {e}")
                    
                    # Check if we should continue or stop
                    if step_config.get("critical", True):
                        self.logger.error(f"Critical step {step_name} failed with exception, stopping pipeline")
                        break
                    else:
                        self.logger.warning(f"Non-critical step {step_name} failed with exception, continuing pipeline")
            
            # Final pipeline state
            if failed_steps == 0:
                final_state = PipelineState.COMPLETED
                self.logger.info(f"🎉 Pipeline completed successfully: {successful_steps}/{total_steps} steps")
            else:
                final_state = PipelineState.FAILED
                self.logger.error(f"⚠️ Pipeline completed with failures: {successful_steps}/{total_steps} steps successful, {failed_steps} failed")
            
            # Update final pipeline state
            pipeline_state_manager.update_pipeline_state(
                self.pipeline_id,
                final_state,
                progress_percentage=100.0
            )
            
            # Record pipeline completion
            pipeline_monitor.record_pipeline_event(
                event_type="pipeline_completed",
                pipeline_id=self.pipeline_id,
                message=f"Pipeline completed with {successful_steps}/{total_steps} successful steps",
                context={
                    "successful_steps": successful_steps,
                    "failed_steps": failed_steps,
                    "total_steps": total_steps,
                    "final_state": final_state.value
                }
            )
            
            return failed_steps == 0
            
        except Exception as e:
            # Record pipeline failure
            pipeline_state_manager.update_pipeline_state(
                self.pipeline_id,
                PipelineState.FAILED,
                error={"message": str(e), "exception": type(e).__name__}
            )
            
            pipeline_monitor.record_pipeline_event(
                event_type="pipeline_failed",
                pipeline_id=self.pipeline_id,
                message=f"Pipeline failed with exception: {e}",
                context={"exception": type(e).__name__}
            )
            
            self.logger.error(f"Pipeline execution failed: {e}")
            return False
    
    @handles_errors(Exception, fallback=False)
    async def _execute_step_with_validation(
        self,
        step_name: str,
        step_func: callable,
        step_dependencies: List[str],
        validation_level: ValidationLevel,
        step_config: Dict[str, Any],
        **config: Dict[str, Any]
    ) -> bool:
        """Execute a single step with comprehensive validation."""
        
        try:
            # Pre-execution validation
            await self._validate_step_prerequisites(step_name, step_dependencies, validation_level)
            
            # Execute step with monitoring
            with pipeline_utilities.safe_data_operation(
                f"step_{step_name}",
                self._get_step_output_file(step_name)
            ):
                # Call the step function
                if asyncio.iscoroutinefunction(step_func):
                    result = await step_func(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        data_dir=self.data_dir,
                        **step_config,
                        **config
                    )
                else:
                    result = step_func(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        data_dir=self.data_dir,
                        **step_config,
                        **config
                    )
            
            # Post-execution validation
            await self._validate_step_output(step_name, validation_level)
            
            return result if isinstance(result, bool) else True
            
        except Exception as e:
            self.logger.error(f"Step execution failed for {step_name}: {e}")
            raise
    
    @handles_errors(Exception, fallback={})
    async def _validate_step_prerequisites(
        self,
        step_name: str,
        step_dependencies: List[str],
        validation_level: ValidationLevel
    ) -> Dict[str, Any]:
        """Validate step prerequisites."""
        
        try:
            validation_results = await validator_orchestrator.validate_pipeline_step(
                step_name=f"{step_name}_prerequisites",
                data=None,
                context={
                    "step_name": step_name,
                    "dependencies": step_dependencies,
                    "validation_level": validation_level.value,
                    "symbol": self.symbol,
                    "exchange": self.exchange,
                    "timeframe": self.timeframe,
                    "data_dir": self.data_dir
                },
                validators_to_run=["step_dependency"]
            )
            
            # Check validation results
            for validator_name, report in validation_results.items():
                if report.result == ValidationResult.FAILED:
                    raise ValueError(f"Prerequisites validation failed for {step_name}: {report.message}")
                elif report.result == ValidationResult.WARNING:
                    self.logger.warning(f"Prerequisites validation warning for {step_name}: {report.message}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Prerequisites validation failed for {step_name}: {e}")
            raise
    
    @handles_errors(Exception, fallback={})
    async def _validate_step_output(
        self,
        step_name: str,
        validation_level: ValidationLevel
    ) -> Dict[str, Any]:
        """Validate step output."""
        
        try:
            output_file = self._get_step_output_file(step_name)
            if not safe_file_exists(output_file):
                self.logger.warning(f"Output file not found for {step_name}: {output_file}")
                return {"warning": "Output file not found"}
            
            # Load output data for validation
            output_data = pipeline_utilities.format_manager.read_data(output_file)
            
            validation_results = await validator_orchestrator.validate_pipeline_step(
                step_name=f"{step_name}_output",
                data=output_data,
                context={
                    "step_name": step_name,
                    "output_file": output_file,
                    "validation_level": validation_level.value
                },
                validators_to_run=["data_format", "data_quality"]
            )
            
            # Check validation results
            for validator_name, report in validation_results.items():
                if report.result == ValidationResult.FAILED:
                    self.logger.error(f"Output validation failed for {step_name}: {report.message}")
                elif report.result == ValidationResult.WARNING:
                    self.logger.warning(f"Output validation warning for {step_name}: {report.message}")
            
            return validation_results
            
        except Exception as e:
            self.logger.warning(f"Could not validate output for {step_name}: {e}")
            return {"error": str(e)}
    
    @handles_errors(Exception, fallback={})
    async def _get_step_validation_results(self, step_name: str) -> Dict[str, Any]:
        """Get validation results for a step."""
        
        try:
            validation_summary = validator_orchestrator.get_validation_summary()
            return {
                "validation_summary": validation_summary,
                "step_name": step_name,
                "timestamp": format_datetime(get_current_datetime())
            }
        except Exception as e:
            self.logger.warning(f"Could not get validation results for {step_name}: {e}")
            return {"error": str(e)}
    
    def _get_step_output_file(self, step_name: str) -> str:
        """Get expected output file for a step."""
        return f"{self.data_dir}/{step_name}_{self.symbol}_{self.timeframe}.parquet"
    
    @handles_errors(Exception, fallback={})
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        
        try:
            # Get pipeline state
            pipeline_state = self.pipeline_state
            
            # Get monitoring summary
            monitoring_summary = pipeline_monitor.get_monitoring_summary()
            
            # Get state manager summary
            state_summary = pipeline_state_manager.get_pipeline_status_summary()
            
            # Get validation summary
            validation_summary = validator_orchestrator.get_validation_summary()
            
            return {
                "pipeline_id": self.pipeline_id,
                "symbol": self.symbol,
                "exchange": self.exchange,
                "timeframe": self.timeframe,
                "data_dir": self.data_dir,
                "pipeline_state": pipeline_state.to_dict(),
                "monitoring_summary": monitoring_summary,
                "state_summary": state_summary,
                "validation_summary": validation_summary,
                "timestamp": format_datetime(get_current_datetime())
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get pipeline status: {e}")
            return {"error": str(e)}
    
    @handles_errors(Exception, fallback=None)
    def save_pipeline_report(self, file_path: str) -> None:
        """Save comprehensive pipeline report."""
        
        try:
            # Get pipeline status
            status = self.get_pipeline_status()
            
            # Save monitoring report
            monitoring_report_path = Path(file_path).parent / f"{Path(file_path).stem}_monitoring.json"
            pipeline_monitor.save_monitoring_report(str(monitoring_report_path))
            
            # Save pipeline status
            safe_json_dump(status, file_path, indent=2)
            
            self.logger.info(f"Pipeline report saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save pipeline report: {e}")
    
    @handles_errors(Exception, fallback=None)
    def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        
        try:
            # Stop monitoring
            pipeline_monitor.stop_monitoring()
            
            # Record pipeline cleanup
            pipeline_monitor.record_pipeline_event(
                event_type="pipeline_cleanup",
                pipeline_id=self.pipeline_id,
                message="Pipeline cleanup completed"
            )
            
            self.logger.info(f"Pipeline cleanup completed for {self.pipeline_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup pipeline: {e}")


# Convenience function for running enhanced pipeline
async def run_enhanced_pipeline(
    symbol: str,
    exchange: str,
    pipeline_steps: List[Dict[str, Any]],
    timeframe: str = "1m",
    data_dir: str = "data_cache",
    **config: Dict[str, Any]
) -> bool:
    """Run enhanced pipeline with comprehensive validation and monitoring."""
    
    integration = EnhancedPipelineIntegration(symbol, exchange, timeframe, data_dir)
    
    try:
        success = await integration.execute_enhanced_pipeline(pipeline_steps, **config)
        
        # Save final report
        report_path = f"{data_dir}/enhanced_pipeline_report_{symbol}_{timeframe}.json"
        integration.save_pipeline_report(report_path)
        
        return success
        
    finally:
        integration.cleanup()