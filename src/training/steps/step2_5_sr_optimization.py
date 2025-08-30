#!/usr/bin/env python3
"""Step 2.5: S/R Detection Optimization.

This module performs comprehensive S/R detection optimization before HMM clustering
to ensure that all subsequent steps use optimized parameters for S/R features.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import json
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.centralized_decorators import (
    comprehensive_data_validation,
    handle_errors,
    memory_efficient,
    resource_monitor,
    secure_data_processing,
    validate_data_structure,
    with_tracing_span,
    quality_gate,
    monitor_feature_engineering,
    ensure_data_integrity,
    monitor_step_execution,
    secure_step_execution,
    validate_pipeline_step
)
from src.utils.logger import system_logger
from src.tactician.sr_detection_optimization import SRDetectionOptimizer

logger = system_logger.getChild("Step2_5SROptimization")


class SROptimizationStep:
    """Step 2.5: S/R Detection Optimization with comprehensive parameter optimization."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("SROptimizationStep")
        self.start_time = None
        self.step_timings = {}
        self.optimizer = None
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize S/R optimization components."""
        self.logger.info("🔧 Initializing S/R optimization components...")
        try:
            # Initialize S/R detection optimizer
            self.optimizer = SRDetectionOptimizer(self.config)
            self.logger.info("✅ S/R detection optimizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize S/R optimization components: {e}")
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="sr_optimization_initialization"
    )
    async def initialize(self) -> bool:
        """Initialize the S/R optimization step."""
        try:
            self.logger.info("🚀 Initializing S/R optimization step...")
            
            # Initialize the optimizer
            if not await self.optimizer.initialize():
                self.logger.error("Failed to initialize S/R detection optimizer")
                return False
            
            self.logger.info("✅ S/R optimization step initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize S/R optimization step: {e}")
            return False

    @monitor_step_execution
    @secure_step_execution
    @validate_pipeline_step
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="sr_optimization_execution"
    )
    async def execute(self) -> bool:
        """Execute the S/R optimization step."""
        try:
            self.logger.info("🎯 Starting S/R detection optimization...")
            self.start_time = time.time()
            
            # Perform comprehensive S/R optimization
            optimization_result = await self._perform_sr_optimization()
            
            if not optimization_result:
                self.logger.error("S/R optimization failed")
                return False
            
            # Save optimization results for subsequent steps
            await self._save_optimization_results(optimization_result)
            
            # Update configuration with optimized parameters
            await self._update_config_with_optimized_params(optimization_result)
            
            execution_time = time.time() - self.start_time
            self.logger.info(f"✅ S/R optimization completed successfully in {execution_time:.2f}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute S/R optimization: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="sr_optimization_performance"
    )
    async def _perform_sr_optimization(self) -> Optional[Any]:
        """Perform comprehensive S/R detection optimization."""
        try:
            self.logger.info("🔍 Performing comprehensive S/R detection optimization...")
            
            # Run multi-method ensemble optimization
            self.logger.info("📊 Running multi-method ensemble optimization...")
            ensemble_result = await self.optimizer.optimize_multi_method_ensemble()
            
            # Run advanced strength scoring optimization
            self.logger.info("⚖️ Running advanced strength scoring optimization...")
            strength_result = await self.optimizer.optimize_advanced_strength_scoring()
            
            # Run multi-timeframe confluence optimization
            self.logger.info("🕐 Running multi-timeframe confluence optimization...")
            timeframe_result = await self.optimizer.optimize_multi_timeframe_confluence()
            
            # Run advanced S/R method optimization
            self.logger.info("🔬 Running advanced S/R method optimization...")
            advanced_result = await self.optimizer.optimize_advanced_sr_methods()
            
            # Run DBSCAN clustering optimization
            self.logger.info("🎯 Running DBSCAN clustering optimization...")
            dbscan_result = await self.optimizer.optimize_dbscan_clustering()
            
            # Combine all optimization results
            combined_result = await self._combine_optimization_results([
                ensemble_result,
                strength_result,
                timeframe_result,
                advanced_result,
                dbscan_result
            ])
            
            self.logger.info("✅ Comprehensive S/R optimization completed")
            return combined_result
            
        except Exception as e:
            self.logger.error(f"Failed to perform S/R optimization: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="sr_optimization_combination"
    )
    async def _combine_optimization_results(self, results: List[Any]) -> Optional[Any]:
        """Combine multiple optimization results into a single optimized configuration."""
        try:
            self.logger.info("🔗 Combining optimization results...")
            
            # Filter out None results
            valid_results = [r for r in results if r is not None]
            
            if not valid_results:
                self.logger.warning("No valid optimization results to combine")
                return None
            
            # Create combined result
            combined_result = {
                "method_weights": {},
                "strength_weights": {},
                "dbscan_params": {},
                "timeframe_weights": {},
                "advanced_params": {},
                "performance_metrics": {
                    "optimization_score": 0.0,
                    "sharpe_ratio": 0.0,
                    "win_rate": 0.0,
                    "max_drawdown": 0.0,
                    "profit_factor": 0.0,
                    "signal_clarity": 0.0,
                },
                "validation_metrics": {
                    "cross_validation_score": 0.0,
                    "out_of_sample_score": 0.0,
                    "statistical_significance": 0.0,
                },
                "metadata": {
                    "optimization_time": 0.0,
                    "n_trials": 0,
                    "best_trial_number": 0,
                    "optimization_method": "combined",
                    "market_regime": "combined",
                    "timestamp": time.time(),
                }
            }
            
            # Aggregate parameters from all results
            for result in valid_results:
                if hasattr(result, 'method_weights'):
                    combined_result["method_weights"].update(result.method_weights)
                if hasattr(result, 'strength_weights'):
                    combined_result["strength_weights"].update(result.strength_weights)
                if hasattr(result, 'dbscan_params'):
                    combined_result["dbscan_params"].update(result.dbscan_params)
                if hasattr(result, 'timeframe_weights'):
                    combined_result["timeframe_weights"].update(result.timeframe_weights)
                if hasattr(result, 'advanced_params'):
                    combined_result["advanced_params"].update(result.advanced_params)
                
                # Aggregate performance metrics
                if hasattr(result, 'optimization_score'):
                    combined_result["performance_metrics"]["optimization_score"] = max(
                        combined_result["performance_metrics"]["optimization_score"],
                        result.optimization_score
                    )
                if hasattr(result, 'sharpe_ratio'):
                    combined_result["performance_metrics"]["sharpe_ratio"] = max(
                        combined_result["performance_metrics"]["sharpe_ratio"],
                        result.sharpe_ratio
                    )
                if hasattr(result, 'win_rate'):
                    combined_result["performance_metrics"]["win_rate"] = max(
                        combined_result["performance_metrics"]["win_rate"],
                        result.win_rate
                    )
            
            self.logger.info("✅ Optimization results combined successfully")
            return combined_result
            
        except Exception as e:
            self.logger.error(f"Failed to combine optimization results: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="sr_optimization_save"
    )
    async def _save_optimization_results(self, optimization_result: Any) -> bool:
        """Save optimization results for subsequent steps."""
        try:
            self.logger.info("💾 Saving optimization results...")
            
            # Create optimization results directory
            results_dir = Path("data/optimization")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save optimization results
            results_file = results_dir / "sr_optimization_results.json"
            
            # Convert to dictionary if it's an OptimizationResult object
            if hasattr(optimization_result, 'to_dict'):
                results_data = optimization_result.to_dict()
            else:
                results_data = optimization_result
            
            # Add metadata
            results_data["metadata"]["step"] = "step2_5_sr_optimization"
            results_data["metadata"]["timestamp"] = time.time()
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            # Also save to the expected location for SR predictor
            sr_results_file = Path("optimization_results.json")
            with open(sr_results_file, 'w') as f:
                json.dump({"best_result": results_data}, f, indent=2, default=str)
            
            self.logger.info(f"✅ Optimization results saved to {results_file} and {sr_results_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save optimization results: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="sr_config_update"
    )
    async def _update_config_with_optimized_params(self, optimization_result: Any) -> bool:
        """Update configuration with optimized parameters."""
        try:
            self.logger.info("⚙️ Updating configuration with optimized parameters...")
            
            # Ensure SR configuration exists
            if "sr_breakout_predictor" not in self.config:
                self.config["sr_breakout_predictor"] = {}
            
            # Set use_optimized_params to True
            self.config["sr_breakout_predictor"]["use_optimized_params"] = True
            
            # Set optimization results file path
            self.config["sr_breakout_predictor"]["optimization_results_file"] = "optimization_results.json"
            
            # Update SR detection optimization config
            if "sr_detection_optimization" not in self.config:
                self.config["sr_detection_optimization"] = {}
            
            # Add optimized parameters to config
            if hasattr(optimization_result, 'method_weights'):
                self.config["sr_detection_optimization"]["optimized_method_weights"] = optimization_result.method_weights
            if hasattr(optimization_result, 'strength_weights'):
                self.config["sr_detection_optimization"]["optimized_strength_weights"] = optimization_result.strength_weights
            if hasattr(optimization_result, 'dbscan_params'):
                self.config["sr_detection_optimization"]["optimized_dbscan_params"] = optimization_result.dbscan_params
            if hasattr(optimization_result, 'timeframe_weights'):
                self.config["sr_detection_optimization"]["optimized_timeframe_weights"] = optimization_result.timeframe_weights
            if hasattr(optimization_result, 'advanced_params'):
                self.config["sr_detection_optimization"]["optimized_advanced_params"] = optimization_result.advanced_params
            
            self.logger.info("✅ Configuration updated with optimized parameters")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="sr_optimization_cleanup"
    )
    async def cleanup(self) -> bool:
        """Clean up resources after optimization."""
        try:
            self.logger.info("🧹 Cleaning up S/R optimization resources...")
            
            # Clean up optimizer
            if self.optimizer:
                # Add cleanup method if available
                if hasattr(self.optimizer, 'cleanup'):
                    await self.optimizer.cleanup()
            
            self.logger.info("✅ S/R optimization cleanup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup S/R optimization: {e}")
            return False


@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="step2_5_sr_optimization"
)
async def run_step(config: dict[str, Any]) -> bool:
    """Run the S/R optimization step."""
    try:
        logger.info("🚀 Starting Step 2.5: S/R Detection Optimization")
        
        # Create and initialize the step
        step = SROptimizationStep(config)
        
        # Initialize the step
        if not await step.initialize():
            logger.error("Failed to initialize S/R optimization step")
            return False
        
        # Execute the step
        success = await step.execute()
        
        # Cleanup
        await step.cleanup()
        
        if success:
            logger.info("✅ Step 2.5: S/R Detection Optimization completed successfully")
        else:
            logger.error("❌ Step 2.5: S/R Detection Optimization failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to run S/R optimization step: {e}")
        return False


if __name__ == "__main__":
    # Test the step
    import asyncio
    
    # Load test configuration
    test_config = {
        "sr_detection_optimization": {
            "n_trials": 10,  # Reduced for testing
            "cv_folds": 3,
            "test_size": 0.2,
            "optimization_timeout": 300,  # 5 minutes for testing
            "performance_thresholds": {
                "min_sharpe_ratio": 0.3,
                "max_drawdown": -0.2,
                "min_win_rate": 0.5,
                "min_profit_factor": 1.2,
                "min_signal_clarity": 0.05,
            }
        },
        "sr_breakout_predictor": {
            "use_optimized_params": True,
            "enable_detailed_reporting": True,
            "report_directory": "reports/sr_analysis",
            "report_format": "json",
            "report_retention_days": 30
        }
    }
    
    # Run the step
    success = asyncio.run(run_step(test_config))
    print(f"Step execution {'successful' if success else 'failed'}")