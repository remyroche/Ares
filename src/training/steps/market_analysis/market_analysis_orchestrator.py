#!/usr/bin/env python3

"""
Market Analysis Orchestrator

This orchestrator manages the execution of the market analysis workflow within the pre-training pipeline.
It coordinates the execution of the following steps in sequence:
1. sr_parameter_optimization
2. sr_detection  
3. sr_clustering
4. hdbscan_clustering
5. regime_clustering
6. regime_models_training
7. regime_ensemble_training
8. regime_data_splitting

The orchestrator ensures proper data flow between steps and handles error recovery.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
import sys
sys.path.insert(0, str(project_root))

# Import tprint utilities for enhanced logging
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_error, tprint_warning, 
    tprint_debug, tprint_performance, tprint_structured, tprint_data_preview,
    tprint_data_format, tprint_progress, tprint_timer
)

# Import market analysis components
from src.training.steps.market_analysis.components.sr_parameter_optimization import SRParameterOptimizationComponent
from src.training.steps.market_analysis.components.sr_detection import SRDetectionComponent
from src.training.steps.market_analysis.components.sr_clustering import SRClusteringComponent
from src.training.steps.market_analysis.hdbscan_regime_discovery_step import HDBSCANRegimeDiscoveryStep
from src.training.steps.market_analysis.components.regime_clustering import RegimeClusteringComponent
from src.training.steps.market_analysis.components.regime_models_training import RegimeModelsTrainingComponent
from src.training.steps.market_analysis.components.regime_ensemble_training import RegimeEnsembleTrainingComponent
from src.training.steps.market_analysis.components.regime_data_splitting import RegimeDataSplittingComponent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ares.market_analysis_orchestrator")

# Initialize tprint with configuration
tprint_info("🎯 Initializing Market Analysis Orchestrator")

class MarketAnalysisOrchestrator:
    """
    Orchestrates the market analysis workflow within the pre-training pipeline.
    """
    
    def __init__(self):
        """Initialize the market analysis orchestrator."""
        self.logger = logger
        tprint_info("MarketAnalysisOrchestrator initialized")
        
        # Define the market analysis workflow steps
        self.workflow_steps = [
            "sr_parameter_optimization",
            "sr_detection", 
            "sr_clustering",
            "hdbscan_clustering",
            "regime_clustering",
            "regime_models_training",
            "regime_ensemble_training",
            "regime_data_splitting"
        ]
        
        # Initialize component instances
        self.components = {}
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all market analysis components."""
        tprint_info("Initializing market analysis components")
        
        try:
            # Initialize each component
            self.components = {
                "sr_parameter_optimization": SRParameterOptimizationComponent(),
                "sr_detection": SRDetectionComponent(),
                "sr_clustering": SRClusteringComponent(),
                "hdbscan_clustering": HDBSCANRegimeDiscoveryStep(),
                "regime_clustering": RegimeClusteringComponent(),
                "regime_models_training": RegimeModelsTrainingComponent(),
                "regime_ensemble_training": RegimeEnsembleTrainingComponent(),
                "regime_data_splitting": RegimeDataSplittingComponent()
            }
            
            tprint_success("All market analysis components initialized successfully")
            
        except Exception as e:
            tprint_error(f"Failed to initialize components: {e}")
            raise
    
    async def execute_market_analysis_workflow(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete market analysis workflow.
        
        Args:
            config: Configuration dictionary containing symbol, exchange, timeframes, etc.
            
        Returns:
            Execution results for all steps
        """
        tprint_info("🚀 Starting market analysis workflow")
        tprint_data_preview(config, "market_analysis_config")
        
        # Initialize pipeline state to pass data between steps
        pipeline_state = {
            "config": config,
            "artifacts": {},
            "metadata": {},
            "symbol": config.get("symbol", "UNKNOWN"),
            "exchange": config.get("exchange", "UNKNOWN"),
            "timeframe": config.get("timeframe", "UNKNOWN"),
            "direction": config.get("direction", "long"),
            "mode": config.get("mode", "full")
        }
        
        tprint_structured({
            "total_steps": len(self.workflow_steps),
            "step_sequence": self.workflow_steps,
            "symbol": config.get("symbol"),
            "exchange": config.get("exchange"),
            "timeframe": config.get("timeframe")
        })
        
        results = {}
        successful_steps = 0
        
        for i, step_name in enumerate(self.workflow_steps, 1):
            tprint_progress(i, len(self.workflow_steps), f"Executing: {step_name}")
            
            try:
                success, step_output = await self._execute_step(step_name, config, pipeline_state)
                
                results[step_name] = {
                    "success": success,
                    "output": step_output,
                    "step_number": i,
                    "timestamp": datetime.now().isoformat()
                }
                
                if success:
                    successful_steps += 1
                    tprint_success(f"Step {i}/{len(self.workflow_steps)} completed: {step_name}")
                    
                    # Update pipeline state with artifacts from this step
                    if isinstance(step_output, dict) and "artifacts" in step_output:
                        pipeline_state["artifacts"].update(step_output["artifacts"])
                        tprint_debug(f"Updated pipeline state with artifacts from {step_name}")
                else:
                    tprint_error(f"Market analysis failed at step: {step_name}")
                    break
                    
            except Exception as e:
                error_msg = f"Exception in step {step_name}: {str(e)}"
                tprint_error(error_msg)
                
                results[step_name] = {
                    "success": False,
                    "output": {"error": error_msg},
                    "step_number": i,
                    "timestamp": datetime.now().isoformat()
                }
                break
        
        tprint_structured({
            "workflow": "market_analysis",
            "successful_steps": successful_steps,
            "total_steps": len(self.workflow_steps),
            "success_rate": f"{(successful_steps/len(self.workflow_steps))*100:.1f}%"
        })
        
        return results
    
    async def _execute_step(self, step_name: str, config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Execute a single market analysis step.
        
        Args:
            step_name: Name of the step to execute
            config: Configuration dictionary
            pipeline_state: Current pipeline state with artifacts
            
        Returns:
            Tuple of (success, output)
        """
        with tprint_timer(f"Step execution: {step_name}"):
            try:
                tprint_info(f"Executing market analysis step: {step_name}")
                
                # Get the component for this step
                if step_name not in self.components:
                    raise ValueError(f"Unknown step: {step_name}")
                
                component = self.components[step_name]
                
                # Execute the step
                if hasattr(component, 'execute'):
                    # For BaseStep-based components
                    result = await component.execute(config)
                elif hasattr(component, 'run'):
                    # For other component types
                    result = await component.run(config, pipeline_state)
                else:
                    raise ValueError(f"Component {step_name} has no execute or run method")
                
                # Check if execution was successful
                if isinstance(result, dict):
                    success = result.get("success", True)
                    if "error" in result:
                        success = False
                else:
                    success = True
                
                if success:
                    tprint_success(f"Step '{step_name}' completed successfully")
                    tprint_debug(f"Output: {result}")
                else:
                    tprint_error(f"Step '{step_name}' failed")
                    if isinstance(result, dict) and "error" in result:
                        tprint_error(f"Error: {result['error']}")
                
                return success, result
                
            except Exception as e:
                error_msg = f"Failed to execute step '{step_name}': {str(e)}"
                tprint_error(error_msg)
                return False, {"error": error_msg}
    
    def print_summary(self, results: Dict[str, Any]):
        """
        Print execution summary using tprint.
        
        Args:
            results: Execution results
        """
        total_steps = len(results)
        successful_steps = sum(1 for r in results.values() if r.get("success", False))
        
        # Create summary data structure
        summary_data = {
            "workflow": "MARKET_ANALYSIS",
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "failed_steps": total_steps - successful_steps,
            "success_rate": f"{(successful_steps/total_steps)*100:.1f}%",
            "step_details": []
        }
        
        # Add step details
        for step_name, result in results.items():
            step_detail = {
                "step_number": result.get("step_number", "?"),
                "step_name": step_name,
                "status": "SUCCESS" if result.get("success", False) else "FAILED",
                "success": result.get("success", False),
                "timestamp": result.get("timestamp", "Unknown")
            }
            summary_data["step_details"].append(step_detail)
        
        # Print structured summary
        tprint_structured(summary_data)
        
        # Print individual step status
        tprint_info("Market Analysis Step Details:")
        for step_name, result in results.items():
            status = "✅ SUCCESS" if result.get("success", False) else "❌ FAILED"
            step_num = result.get("step_number", "?")
            timestamp = result.get("timestamp", "Unknown")
            tprint_info(f"  {step_num:2d}. {step_name:<30} {status} ({timestamp})")


async def main():
    """Main entry point for testing the orchestrator."""
    tprint_info("🎯 Starting Market Analysis Orchestrator...")
    
    # Create orchestrator
    orchestrator = MarketAnalysisOrchestrator()
    
    # Test configuration
    test_config = {
        "symbol": "ETHUSDT",
        "exchange": "binance", 
        "timeframe": "15m",
        "direction": "long",
        "mode": "full"
    }
    
    tprint_data_preview(test_config, "test_config")
    
    try:
        # Execute the workflow
        results = await orchestrator.execute_market_analysis_workflow(test_config)
        orchestrator.print_summary(results)
        
    except Exception as e:
        tprint_error(f"Execution failed: {e}")
        return False
    
    tprint_success("✅ Market Analysis Orchestrator completed")
    return True


if __name__ == "__main__":
    asyncio.run(main())