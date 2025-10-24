#!/usr/bin/env python3

"""
Market Analysis Step

This step integrates the market analysis workflow into the ares_launcher.py system.
It serves as the entry point for the market analysis workflow within the pre-training pipeline.
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

# Import BaseStep
from src.training.steps.base_step import BaseStep

# Import tprint utilities for enhanced logging
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_error, tprint_warning, 
    tprint_debug, tprint_performance, tprint_structured, tprint_data_preview,
    tprint_data_format, tprint_progress, tprint_timer
)

# Import the market analysis orchestrator
from .market_analysis_orchestrator import MarketAnalysisOrchestrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ares.market_analysis_step")

class MarketAnalysisStep(BaseStep):
    """
    Market Analysis Step for the pre-training pipeline.
    
    This step executes the complete market analysis workflow including:
    - SR parameter optimization
    - SR detection
    - SR clustering
    - HDBSCAN clustering
    - Regime clustering
    - Regime models training
    - Regime ensemble training
    - Regime data splitting
    """
    
    def __init__(self, step_name: str = "market_analysis"):
        """Initialize the market analysis step."""
        super().__init__(step_name)
        self.logger = logger
        tprint_info("MarketAnalysisStep initialized")
        
        # Initialize the orchestrator
        self.orchestrator = MarketAnalysisOrchestrator()
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this step must produce."""
        return [
            'sr_parameter_optimization_result',
            'sr_detection_result', 
            'sr_clustering_result',
            'hdbscan_clustering_result',
            'regime_clustering_result',
            'regime_models_training_result',
            'regime_ensemble_training_result',
            'regime_data_splitting_result'
        ]
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the market analysis workflow.
        
        Args:
            config: Configuration dictionary containing symbol, exchange, timeframes, etc.
            
        Returns:
            Execution result with artifacts and metrics
        """
        tprint_info("🚀 Starting Market Analysis Step")
        tprint_data_preview(config, "market_analysis_step_config")
        
        try:
            # Execute the market analysis workflow
            workflow_results = await self.orchestrator.execute_market_analysis_workflow(config)
            
            # Check if all steps were successful
            successful_steps = sum(1 for r in workflow_results.values() if r.get("success", False))
            total_steps = len(workflow_results)
            
            # Create comprehensive artifacts from all steps
            artifacts = {}
            metadata = {
                "workflow_execution_time": datetime.now().isoformat(),
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "success_rate": f"{(successful_steps/total_steps)*100:.1f}%",
                "step_results": workflow_results
            }
            
            # Extract artifacts from each successful step
            for step_name, step_result in workflow_results.items():
                if step_result.get("success", False) and isinstance(step_result.get("output"), dict):
                    step_output = step_result["output"]
                    if "artifacts" in step_output:
                        artifacts.update(step_output["artifacts"])
                        tprint_debug(f"Extracted artifacts from {step_name}")
            
            # Determine overall success
            overall_success = successful_steps == total_steps
            
            if overall_success:
                tprint_success("✅ Market Analysis Step completed successfully")
                tprint_structured({
                    "status": "completed",
                    "successful_steps": successful_steps,
                    "total_steps": total_steps,
                    "artifacts_count": len(artifacts)
                })
            else:
                tprint_error(f"❌ Market Analysis Step completed with failures ({successful_steps}/{total_steps} steps successful)")
                tprint_structured({
                    "status": "partial_failure",
                    "successful_steps": successful_steps,
                    "total_steps": total_steps,
                    "failed_steps": total_steps - successful_steps
                })
            
            # Print detailed summary
            self.orchestrator.print_summary(workflow_results)
            
            return {
                "success": overall_success,
                "artifacts": artifacts,
                "metadata": metadata,
                "workflow_results": workflow_results
            }
            
        except Exception as e:
            error_msg = f"Market Analysis Step failed: {str(e)}"
            tprint_error(error_msg)
            
            return {
                "success": False,
                "artifacts": {},
                "metadata": {
                    "error": error_msg,
                    "workflow_execution_time": datetime.now().isoformat()
                },
                "workflow_results": {}
            }


async def main():
    """Main entry point for testing the step."""
    tprint_info("🎯 Starting Market Analysis Step...")
    
    # Create step instance
    step = MarketAnalysisStep()
    
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
        # Execute the step
        result = await step.execute(test_config)
        
        if result["success"]:
            tprint_success("✅ Market Analysis Step completed successfully")
        else:
            tprint_error("❌ Market Analysis Step failed")
            
        tprint_structured(result)
        
    except Exception as e:
        tprint_error(f"Execution failed: {e}")
        return False
    
    return True


# Register the step
def register_market_analysis_step():
    """Register the market analysis step."""
    from src.training.steps.base_step import step_registry
    
    step_registry.register("market_analysis", MarketAnalysisStep)
    print("✅ Market analysis step registered")


# Auto-register when module is imported
register_market_analysis_step()


if __name__ == "__main__":
    asyncio.run(main())