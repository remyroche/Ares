#!/usr/bin/env python3
"""
Optimisation Pipeline Orchestrator

Robust orchestrator for the optimisation pipeline with:
- Step execution management
- Validation and error recovery
- State transitions and persistence
- Performance monitoring
- Comprehensive logging and reporting
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import traceback

from src.utils.logger import system_logger
from src.utils.pipeline_protection_framework import (
    initialize_pipeline_protection,
    get_pipeline_protection,
    get_state_manager,
    get_monitor,
    PipelineState,
    ValidationLevel,
    OperationType
)
from src.training.steps.optimisation.optimisation_pipeline_validator import (
    OptimisationPipelineValidator,
    ConfidenceCalibrationValidator,
    ParameterOptimizationValidator
)
from src.training.steps.optimisation.optimisation_decorators import (
    protect_optimisation_operation,
    protect_data_operation
)
from src.training.steps.optimisation.optimisation_utilities import (
    initialize_optimisation_utilities,
    get_data_formatting_utils,
    get_analysis_operations_utils,
    get_data_access_control,
    get_pipeline_state_manager,
    get_performance_optimizer
)
from src.utils.common_operations import (
    ensure_directory,
    safe_file_exists,
    safe_json_dump,
    safe_json_load,
    format_datetime,
    get_current_datetime
)


class OptimisationPipelineOrchestrator:
    """Orchestrator for the optimisation pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("OptimisationPipelineOrchestrator")
        
        # Initialize components
        self._initialize_components()
        
        # Pipeline state
        self.pipeline_state: Optional[PipelineState] = None
        self.execution_history: List[Dict[str, Any]] = []
        self.current_step: Optional[str] = None
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.step_times: Dict[str, float] = {}
        
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        try:
            self.logger.info("🔧 Initializing optimisation pipeline components...")
            
            # Initialize protection framework
            initialize_pipeline_protection(self.config)
            
            # Initialize utilities
            initialize_optimisation_utilities(self.config)
            
            # Initialize validators
            self.pipeline_validator = OptimisationPipelineValidator(self.config)
            self.confidence_validator = ConfidenceCalibrationValidator(self.config)
            self.parameter_validator = ParameterOptimizationValidator(self.config)
            
            # Get component instances
            self.protection = get_pipeline_protection()
            self.state_manager = get_state_manager()
            self.monitor = get_monitor()
            
            self.logger.info("✅ All pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.exception(f"❌ Component initialization failed: {e}")
            raise
    
    async def execute_pipeline(self, 
                             symbol: str,
                             exchange: str,
                             timeframe: str = "1m",
                             data_dir: str = "data_cache",
                             **kwargs) -> Dict[str, Any]:
        """Execute the complete optimisation pipeline."""
        try:
            self.logger.info("🚀 Starting optimisation pipeline execution...")
            self.start_time = get_current_datetime()
            
            # Prepare training input
            training_input = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "data_dir": data_dir,
                **kwargs
            }
            
            # Initialize pipeline state
            await self._initialize_pipeline_state(training_input)
            
            # Execute pipeline steps
            pipeline_result = await self._execute_pipeline_steps(training_input)
            
            # Generate final report
            final_report = await self._generate_final_report(pipeline_result)
            
            # Save pipeline state
            await self._save_pipeline_state()
            
            self.logger.info("✅ Optimisation pipeline execution completed successfully")
            return final_report
            
        except Exception as e:
            self.logger.exception(f"❌ Pipeline execution failed: {e}")
            await self._handle_pipeline_failure(e)
            raise
    
    async def _initialize_pipeline_state(self, training_input: Dict[str, Any]) -> None:
        """Initialize pipeline state."""
        try:
            self.logger.info("🔧 Initializing pipeline state...")
            
            # Load existing state or create new one
            self.pipeline_state = await self.state_manager.initialize_state()
            
            # Update state with training input
            self.pipeline_state.data_checkpoints["training_input"] = training_input
            
            # Initialize execution tracking
            self.execution_history = []
            self.step_times = {}
            
            self.logger.info("✅ Pipeline state initialized")
            
        except Exception as e:
            self.logger.exception(f"❌ Pipeline state initialization failed: {e}")
            raise
    
    async def _execute_pipeline_steps(self, training_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all pipeline steps."""
        try:
            self.logger.info("🔄 Executing pipeline steps...")
            
            # Define pipeline steps
            pipeline_steps = [
                ("pre_validation", self._execute_pre_validation),
                ("confidence_calibration", self._execute_confidence_calibration),
                ("parameter_optimization", self._execute_parameter_optimization),
                ("post_validation", self._execute_post_validation),
                ("final_optimization", self._execute_final_optimization)
            ]
            
            pipeline_result = {
                "overall_success": False,
                "step_results": {},
                "execution_time": 0,
                "errors": [],
                "warnings": []
            }
            
            # Execute each step
            for step_name, step_function in pipeline_steps:
                try:
                    self.logger.info(f"🔄 Executing step: {step_name}")
                    self.current_step = step_name
                    
                    # Update pipeline state
                    if self.pipeline_state:
                        self.pipeline_state.update_step(step_name)
                    
                    # Execute step with protection
                    step_start_time = time.time()
                    step_result = await step_function(training_input)
                    step_duration = time.time() - step_start_time
                    
                    # Record step timing
                    self.step_times[step_name] = step_duration
                    self.monitor.record_metric("step_duration", step_duration, {"step": step_name})
                    
                    # Store step result
                    pipeline_result["step_results"][step_name] = step_result
                    
                    # Check step success
                    if not step_result.get("success", False):
                        error_msg = f"Step {step_name} failed: {step_result.get('error', 'Unknown error')}"
                        pipeline_result["errors"].append(error_msg)
                        self.logger.error(f"❌ {error_msg}")
                        
                        # Decide whether to continue or stop
                        if step_name in ["pre_validation", "confidence_calibration"]:
                            self.logger.error(f"💥 Critical step {step_name} failed, stopping pipeline")
                            break
                        else:
                            self.logger.warning(f"⚠️ Non-critical step {step_name} failed, continuing")
                    
                    else:
                        self.logger.info(f"✅ Step {step_name} completed successfully ({step_duration:.2f}s)")
                    
                    # Record execution history
                    self.execution_history.append({
                        "step": step_name,
                        "success": step_result.get("success", False),
                        "duration": step_duration,
                        "timestamp": get_current_datetime().isoformat(),
                        "result": step_result
                    })
                    
                except Exception as e:
                    error_msg = f"Step {step_name} execution failed: {str(e)}"
                    pipeline_result["errors"].append(error_msg)
                    self.logger.exception(f"❌ {error_msg}")
                    
                    # Record failed execution
                    self.execution_history.append({
                        "step": step_name,
                        "success": False,
                        "duration": 0,
                        "timestamp": get_current_datetime().isoformat(),
                        "error": str(e)
                    })
                    
                    # Stop on critical step failures
                    if step_name in ["pre_validation", "confidence_calibration"]:
                        break
            
            # Calculate overall success
            critical_steps = ["pre_validation", "confidence_calibration", "parameter_optimization"]
            critical_success = all(
                pipeline_result["step_results"].get(step, {}).get("success", False)
                for step in critical_steps
                if step in pipeline_result["step_results"]
            )
            
            pipeline_result["overall_success"] = critical_success
            pipeline_result["execution_time"] = sum(self.step_times.values())
            
            self.logger.info(f"✅ Pipeline steps execution completed: success={pipeline_result['overall_success']}")
            return pipeline_result
            
        except Exception as e:
            self.logger.exception(f"❌ Pipeline steps execution failed: {e}")
            raise
    
    @protect_optimisation_operation()
    async def _execute_pre_validation(self, training_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pre-validation step."""
        try:
            self.logger.info("🔍 Executing pre-validation...")
            
            # Validate pipeline dependencies and data
            validation_result = await self.pipeline_validator.validate(
                training_input, 
                self.pipeline_state.data_checkpoints if self.pipeline_state else {}
            )
            
            if validation_result:
                self.logger.info("✅ Pre-validation passed")
                return {"success": True, "validation_result": validation_result}
            else:
                self.logger.error("❌ Pre-validation failed")
                return {"success": False, "error": "Pre-validation failed"}
                
        except Exception as e:
            self.logger.exception(f"❌ Pre-validation execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    @protect_optimisation_operation()
    async def _execute_confidence_calibration(self, training_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute confidence calibration step."""
        try:
            self.logger.info("🎯 Executing confidence calibration...")
            
            # Import and execute confidence calibration
            from src.training.steps.optimisation.step16_confidence_calibration_per_regime import (
                ConfidenceCalibrationPerRegimeStep
            )
            
            calibrator = ConfidenceCalibrationPerRegimeStep()
            
            # Execute calibration
            calibration_result = await calibrator.calibrate_confidence(
                training_input["symbol"],
                training_input["exchange"],
                training_input["timeframe"],
                training_input["data_dir"]
            )
            
            if calibration_result:
                # Validate calibration result
                validation_result = await self.confidence_validator.validate(
                    training_input,
                    {"confidence_calibration": {"success": True}}
                )
                
                if validation_result:
                    self.logger.info("✅ Confidence calibration completed successfully")
                    return {"success": True, "calibration_result": calibration_result}
                else:
                    self.logger.error("❌ Confidence calibration validation failed")
                    return {"success": False, "error": "Calibration validation failed"}
            else:
                self.logger.error("❌ Confidence calibration failed")
                return {"success": False, "error": "Calibration execution failed"}
                
        except Exception as e:
            self.logger.exception(f"❌ Confidence calibration execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    @protect_optimisation_operation()
    async def _execute_parameter_optimization(self, training_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parameter optimization step."""
        try:
            self.logger.info("🔧 Executing parameter optimization...")
            
            # Import and execute parameter optimization
            from src.training.steps.optimisation.step17_final_parameters_optimization_new import (
                FinalParametersOptimizationStep
            )
            
            optimizer = FinalParametersOptimizationStep()
            
            # Execute optimization
            optimization_result = await optimizer.optimize_parameters(
                training_input["symbol"],
                training_input["exchange"],
                training_input["timeframe"],
                training_input["data_dir"]
            )
            
            if optimization_result:
                # Validate optimization result
                validation_result = await self.parameter_validator.validate(
                    training_input,
                    {"parameter_optimization": {"success": True}}
                )
                
                if validation_result:
                    self.logger.info("✅ Parameter optimization completed successfully")
                    return {"success": True, "optimization_result": optimization_result}
                else:
                    self.logger.error("❌ Parameter optimization validation failed")
                    return {"success": False, "error": "Optimization validation failed"}
            else:
                self.logger.error("❌ Parameter optimization failed")
                return {"success": False, "error": "Optimization execution failed"}
                
        except Exception as e:
            self.logger.exception(f"❌ Parameter optimization execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    @protect_optimisation_operation()
    async def _execute_post_validation(self, training_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute post-validation step."""
        try:
            self.logger.info("🔍 Executing post-validation...")
            
            # Validate all step outputs
            validation_result = await self.pipeline_validator.validate(
                training_input,
                self.pipeline_state.data_checkpoints if self.pipeline_state else {}
            )
            
            if validation_result:
                self.logger.info("✅ Post-validation passed")
                return {"success": True, "validation_result": validation_result}
            else:
                self.logger.error("❌ Post-validation failed")
                return {"success": False, "error": "Post-validation failed"}
                
        except Exception as e:
            self.logger.exception(f"❌ Post-validation execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    @protect_optimisation_operation()
    async def _execute_final_optimization(self, training_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute final optimization step."""
        try:
            self.logger.info("🎯 Executing final optimization...")
            
            # Combine all optimization results
            final_result = {
                "symbol": training_input["symbol"],
                "exchange": training_input["exchange"],
                "timeframe": training_input["timeframe"],
                "optimization_timestamp": get_current_datetime().isoformat(),
                "pipeline_success": True
            }
            
            # Add step results
            if self.pipeline_state:
                final_result["step_results"] = self.pipeline_state.validation_results
            
            # Add performance metrics
            final_result["performance_metrics"] = {
                "total_execution_time": sum(self.step_times.values()),
                "step_times": self.step_times,
                "memory_usage": self.monitor.get_metrics_summary().get("memory_usage", {}),
                "error_count": len([h for h in self.execution_history if not h.get("success", False)])
            }
            
            # Save final results
            results_file = Path(training_input["data_dir"]) / f"final_optimization_results_{training_input['symbol']}.json"
            safe_json_dump(final_result, results_file, indent=2)
            
            self.logger.info("✅ Final optimization completed successfully")
            return {"success": True, "final_result": final_result}
            
        except Exception as e:
            self.logger.exception(f"❌ Final optimization execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_final_report(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final pipeline report."""
        try:
            self.logger.info("📊 Generating final pipeline report...")
            
            total_time = (get_current_datetime() - self.start_time).total_seconds() if self.start_time else 0
            
            report = {
                "pipeline_summary": {
                    "symbol": pipeline_result.get("symbol", "unknown"),
                    "exchange": pipeline_result.get("exchange", "unknown"),
                    "overall_success": pipeline_result.get("overall_success", False),
                    "total_execution_time": total_time,
                    "start_time": self.start_time.isoformat() if self.start_time else None,
                    "end_time": get_current_datetime().isoformat()
                },
                "step_summary": {
                    step: {
                        "success": result.get("success", False),
                        "duration": self.step_times.get(step, 0),
                        "error": result.get("error") if not result.get("success", False) else None
                    }
                    for step, result in pipeline_result.get("step_results", {}).items()
                },
                "performance_metrics": {
                    "total_steps": len(pipeline_result.get("step_results", {})),
                    "successful_steps": len([r for r in pipeline_result.get("step_results", {}).values() if r.get("success", False)]),
                    "failed_steps": len([r for r in pipeline_result.get("step_results", {}).values() if not r.get("success", False)]),
                    "average_step_time": sum(self.step_times.values()) / len(self.step_times) if self.step_times else 0,
                    "total_errors": len(pipeline_result.get("errors", [])),
                    "total_warnings": len(pipeline_result.get("warnings", []))
                },
                "execution_history": self.execution_history,
                "errors": pipeline_result.get("errors", []),
                "warnings": pipeline_result.get("warnings", []),
                "recommendations": self._generate_recommendations(pipeline_result)
            }
            
            # Save report
            report_file = Path(self.config.get("data_dir", "data_cache")) / f"optimisation_pipeline_report_{get_current_datetime().strftime('%Y%m%d_%H%M%S')}.json"
            safe_json_dump(report, report_file, indent=2)
            
            self.logger.info(f"✅ Final report generated: {report_file}")
            return report
            
        except Exception as e:
            self.logger.exception(f"❌ Final report generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, pipeline_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on pipeline results."""
        recommendations = []
        
        try:
            # Check for performance issues
            if pipeline_result.get("execution_time", 0) > 3600:  # More than 1 hour
                recommendations.append("Consider optimizing pipeline performance - execution time exceeded 1 hour")
            
            # Check for high error rates
            error_count = len(pipeline_result.get("errors", []))
            if error_count > 3:
                recommendations.append(f"High error count ({error_count}) - review error logs and improve error handling")
            
            # Check for failed steps
            failed_steps = [step for step, result in pipeline_result.get("step_results", {}).items() 
                          if not result.get("success", False)]
            if failed_steps:
                recommendations.append(f"Failed steps detected: {failed_steps} - review and fix these steps")
            
            # Check for memory usage
            memory_metrics = self.monitor.get_metrics_summary().get("memory_usage", {})
            if memory_metrics.get("latest", 0) > 1000:  # More than 1GB
                recommendations.append("High memory usage detected - consider memory optimization")
            
            if not recommendations:
                recommendations.append("Pipeline executed successfully - no specific recommendations")
            
        except Exception as e:
            self.logger.exception(f"Error generating recommendations: {e}")
            recommendations.append("Error generating recommendations - check logs")
        
        return recommendations
    
    async def _save_pipeline_state(self) -> None:
        """Save pipeline state."""
        try:
            if self.pipeline_state:
                # Update state with execution results
                self.pipeline_state.performance_metrics.update({
                    "total_execution_time": sum(self.step_times.values()),
                    "step_times": self.step_times,
                    "execution_history": self.execution_history
                })
                
                # Save state
                await self.state_manager.save_state()
                
        except Exception as e:
            self.logger.exception(f"Error saving pipeline state: {e}")
    
    async def _handle_pipeline_failure(self, error: Exception) -> None:
        """Handle pipeline failure."""
        try:
            self.logger.error("💥 Pipeline failure detected - initiating recovery procedures")
            
            # Record failure in state
            if self.pipeline_state:
                self.pipeline_state.add_error({
                    "type": "pipeline_failure",
                    "error": str(error),
                    "step": self.current_step or "unknown",
                    "traceback": traceback.format_exc()
                })
            
            # Save state for debugging
            await self._save_pipeline_state()
            
            # Generate failure report
            failure_report = {
                "failure_time": get_current_datetime().isoformat(),
                "current_step": self.current_step,
                "error": str(error),
                "execution_history": self.execution_history,
                "step_times": self.step_times
            }
            
            failure_file = Path(self.config.get("data_dir", "data_cache")) / f"pipeline_failure_report_{get_current_datetime().strftime('%Y%m%d_%H%M%S')}.json"
            safe_json_dump(failure_report, failure_file, indent=2)
            
            self.logger.info(f"💾 Failure report saved: {failure_file}")
            
        except Exception as e:
            self.logger.exception(f"Error handling pipeline failure: {e}")


# Convenience function for pipeline execution
async def run_optimisation_pipeline(symbol: str,
                                  exchange: str,
                                  timeframe: str = "1m",
                                  data_dir: str = "data_cache",
                                  config: Optional[Dict[str, Any]] = None,
                                  **kwargs) -> Dict[str, Any]:
    """Run the optimisation pipeline with comprehensive protection."""
    
    if config is None:
        config = {}
    
    # Initialize orchestrator
    orchestrator = OptimisationPipelineOrchestrator(config)
    
    # Execute pipeline
    result = await orchestrator.execute_pipeline(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        **kwargs
    )
    
    return result