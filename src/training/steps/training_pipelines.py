#!/usr/bin/env python3

"""
Training Pipelines Orchestrator

This script provides a unified interface for orchestrating the complete training pipeline
with support for different stages: pre_training, models_training, and backtesting.

Key Features:
- Command-line interface with comprehensive argument parsing
- Stage-based execution with proper sequencing
- Model type selection (tactician vs analyst)
- Direction support (short vs long)
- Timeframe and mode configuration
- Symbol and exchange configuration
- Integration with ares_launcher.py for step execution

Usage:
    python training_pipelines.py --stage pre_training --model tactician --direction long --timeframe 15m --mode full --symbol ETHUSDT --exchange binance
    python training_pipelines.py --stage models_training --model analyst --direction short --timeframe 1h --mode light --symbol BTCUSDT
    python training_pipelines.py --stage backtesting --model tactician --direction long --timeframe 15m --mode full --symbol ETHUSDT
"""

import asyncio
import argparse
import logging
import os
import sys
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import tprint utilities for enhanced logging
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_error, tprint_warning, 
    tprint_debug, tprint_performance, tprint_structured, tprint_data_preview,
    tprint_data_format, tprint_progress, tprint_timer
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ares.training_pipelines")

# Initialize tprint with configuration
tprint_info("🎯 Initializing Training Pipelines Orchestrator")

class TrainingPipelineOrchestrator:
    """
    Orchestrates the complete training pipeline with support for different stages.
    """
    
    def __init__(self):
        """Initialize the training pipeline orchestrator."""
        self.logger = logger
        self.ares_launcher_path = project_root / "src" / "launcher" / "ares_launcher.py"
        tprint_info("TrainingPipelineOrchestrator initialized")
        
    def validate_ares_launcher(self) -> bool:
        """Validate that ares_launcher.py exists and is accessible."""
        tprint_info("Validating ares_launcher.py availability")
        
        if not self.ares_launcher_path.exists():
            tprint_error(f"ares_launcher.py not found at {self.ares_launcher_path}")
            return False
        
        tprint_success("ares_launcher.py found and accessible")
        return True
    
    def build_step_command(self, step_name: str, config: Dict[str, Any]) -> List[str]:
        """
        Build command for executing a single step via ares_launcher.py.
        
        Args:
            step_name: Name of the step to execute
            config: Configuration dictionary
            
        Returns:
            List of command arguments
        """
        tprint_debug(f"Building command for step: {step_name}")
        
        cmd = [
            "python3", str(self.ares_launcher_path),
            step_name,
            "--symbol", config["symbol"],
            "--exchange", config["exchange"],
            "--timeframe", config["timeframe"],
            "--direction", config["direction"],
            "--execution-mode", config["mode"]
        ]
        
        # Log command structure for debugging
        tprint_structured({
            "step_name": step_name,
            "command": " ".join(cmd),
            "config": config
        })
        
        return cmd
    
    async def execute_step(self, step_name: str, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Execute a single step using ares_launcher.py.
        
        Args:
            step_name: Name of the step to execute
            config: Configuration dictionary
            
        Returns:
            Tuple of (success, output)
        """
        with tprint_timer(f"Step execution: {step_name}"):
            try:
                tprint_info(f"Executing step: {step_name}")
                
                cmd = self.build_step_command(step_name, config)
                tprint_debug(f"Command: {' '.join(cmd)}")
                
                # Execute the command
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(project_root)
                )
                
                stdout, stderr = await process.communicate()
                
                # Decode output
                stdout_str = stdout.decode('utf-8') if stdout else ""
                stderr_str = stderr.decode('utf-8') if stderr else ""
                
                # Check if execution was successful
                success = process.returncode == 0
                
                if success:
                    tprint_success(f"Step '{step_name}' completed successfully")
                    tprint_debug(f"Output: {stdout_str}")
                    
                    # Log data format compatibility if output contains data
                    if stdout_str and ("data" in stdout_str.lower() or "result" in stdout_str.lower()):
                        tprint_data_format(stdout_str, f"step_{step_name}_output")
                else:
                    tprint_error(f"Step '{step_name}' failed with return code {process.returncode}")
                    tprint_error(f"Error output: {stderr_str}")
                
                return success, stdout_str + stderr_str
                
            except Exception as e:
                error_msg = f"Failed to execute step '{step_name}': {str(e)}"
                tprint_error(error_msg)
                return False, error_msg
    
    async def execute_pre_training_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pre_training stage with feature generation pipeline.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Execution results
        """
        tprint_info("🚀 Starting pre_training stage")
        tprint_data_preview(config, "pre_training_config")
        
        # Define the pre_training step sequence
        steps = [
            "feature_generation_data_validation_step",
            "feature_generation_labeling_integration_step", 
            "feature_generation_feature_generation_step",
            "feature_generation_period_lookback_optimization_step",
            "feature_generation_feature_selection_step",
            "market_analysis",  # Add market analysis step
        ]
        
        # Add model-specific interaction generation step
        if config.get("model", "analyst").lower() == "tactician":
            steps.append("feature_generation_interaction_generation_step_tactician")
            tprint_info("Using tactician interaction generation step")
        else:
            steps.append("feature_generation_interaction_generation_step_analyst")
            tprint_info("Using analyst interaction generation step")
        
        # Add final steps
        steps.extend([
            "feature_generation_final_feature_selection_step",
            "feature_generation_final_validation_step"
        ])
        
        tprint_structured({
            "total_steps": len(steps),
            "step_sequence": steps,
            "model_type": config.get("model", "analyst")
        })
        
        results = {}
        successful_steps = 0
        
        for i, step_name in enumerate(steps, 1):
            tprint_progress(i, len(steps), f"Executing: {step_name}")
            
            success, output = await self.execute_step(step_name, config)
            results[step_name] = {
                "success": success,
                "output": output,
                "step_number": i
            }
            
            if success:
                successful_steps += 1
                tprint_success(f"Step {i}/{len(steps)} completed: {step_name}")
            else:
                tprint_error(f"Pre-training failed at step: {step_name}")
                break
        
        tprint_structured({
            "stage": "pre_training",
            "successful_steps": successful_steps,
            "total_steps": len(steps),
            "success_rate": f"{(successful_steps/len(steps))*100:.1f}%"
        })
        
        return results
    
    async def execute_models_training_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the models_training stage with analyst and tactician training.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Execution results
        """
        tprint_info("🚀 Starting models_training stage")
        tprint_data_preview(config, "models_training_config")
        
        # Define the models_training step sequence
        steps = [
            "train_analyst_base",
            "train_analyst_ensemble", 
            "train_tactician_base",
            "train_tactician_ensemble"
        ]
        
        tprint_structured({
            "total_steps": len(steps),
            "step_sequence": steps,
            "model_type": config.get("model", "analyst")
        })
        
        results = {}
        successful_steps = 0
        
        for i, step_name in enumerate(steps, 1):
            tprint_progress(i, len(steps), f"Executing: {step_name}")
            
            success, output = await self.execute_step(step_name, config)
            results[step_name] = {
                "success": success,
                "output": output,
                "step_number": i
            }
            
            if success:
                successful_steps += 1
                tprint_success(f"Step {i}/{len(steps)} completed: {step_name}")
            else:
                tprint_error(f"Models training failed at step: {step_name}")
                break
        
        tprint_structured({
            "stage": "models_training",
            "successful_steps": successful_steps,
            "total_steps": len(steps),
            "success_rate": f"{(successful_steps/len(steps))*100:.1f}%"
        })
        
        return results
    
    async def execute_backtesting_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the backtesting stage with validation and reporting pipeline.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Execution results
        """
        tprint_info("🚀 Starting backtesting stage")
        tprint_data_preview(config, "backtesting_config")
        
        # Define the backtesting step sequence
        steps = [
            "basic_backtesting_pre",
            "final_parameters_optimization",
            "basic_backtesting_post", 
            "walk_forward_validation",
            "monte_carlo_simulation",
            "ab_testing",
            "reporting"
        ]
        
        tprint_structured({
            "total_steps": len(steps),
            "step_sequence": steps,
            "model_type": config.get("model", "analyst")
        })
        
        results = {}
        successful_steps = 0
        
        for i, step_name in enumerate(steps, 1):
            tprint_progress(i, len(steps), f"Executing: {step_name}")
            
            success, output = await self.execute_step(step_name, config)
            results[step_name] = {
                "success": success,
                "output": output,
                "step_number": i
            }
            
            if success:
                successful_steps += 1
                tprint_success(f"Step {i}/{len(steps)} completed: {step_name}")
            else:
                tprint_error(f"Backtesting failed at step: {step_name}")
                break
        
        tprint_structured({
            "stage": "backtesting",
            "successful_steps": successful_steps,
            "total_steps": len(steps),
            "success_rate": f"{(successful_steps/len(steps))*100:.1f}%"
        })
        
        return results
    
    async def execute_stage(self, stage: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific stage based on the stage name.
        
        Args:
            stage: Name of the stage to execute
            config: Configuration dictionary
            
        Returns:
            Execution results
        """
        stage_handlers = {
            "pre_training": self.execute_pre_training_stage,
            "models_training": self.execute_models_training_stage,
            "backtesting": self.execute_backtesting_stage
        }
        
        if stage not in stage_handlers:
            raise ValueError(f"Unknown stage: {stage}. Available stages: {list(stage_handlers.keys())}")
        
        return await stage_handlers[stage](config)
    
    def print_summary(self, results: Dict[str, Any], stage: str):
        """
        Print execution summary using tprint.
        
        Args:
            results: Execution results
            stage: Stage name
        """
        total_steps = len(results)
        successful_steps = sum(1 for r in results.values() if r.get("success", False))
        
        # Create summary data structure
        summary_data = {
            "stage": stage.upper(),
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
                "success": result.get("success", False)
            }
            summary_data["step_details"].append(step_detail)
        
        # Print structured summary
        tprint_structured(summary_data)
        
        # Print individual step status
        tprint_info("Step Details:")
        for step_name, result in results.items():
            status = "✅ SUCCESS" if result.get("success", False) else "❌ FAILED"
            step_num = result.get("step_number", "?")
            tprint_info(f"  {step_num:2d}. {step_name:<50} {status}")


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Training Pipelines Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-training stage with tactician model
  python training_pipelines.py --stage pre_training --model tactician --direction long --timeframe 15m --mode full --symbol ETHUSDT --exchange binance
  
  # Models training stage with analyst model
  python training_pipelines.py --stage models_training --model analyst --direction short --timeframe 1h --mode light --symbol BTCUSDT
  
  # Backtesting stage with tactician model
  python training_pipelines.py --stage backtesting --model tactician --direction long --timeframe 15m --mode full --symbol ETHUSDT
  
  # Full pipeline execution (all stages)
  python training_pipelines.py --stage all --model tactician --direction long --timeframe 15m --mode full --symbol ETHUSDT
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--stage", 
        type=str, 
        required=True,
        choices=["pre_training", "models_training", "backtesting", "all"],
        help="Stage to execute (required)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        choices=["tactician", "analyst"],
        default="analyst",
        help="Model type (default: analyst)"
    )
    
    # Trading configuration
    parser.add_argument(
        "--direction",
        type=str,
        choices=["short", "long"],
        default="long",
        help="Trading direction (default: long)"
    )
    
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15m",
        help="Timeframe for training (default: 15m)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["light", "blank", "full"],
        default="full",
        help="Execution mode (default: full)"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default="ETHUSDT",
        help="Trading symbol (default: ETHUSDT)"
    )
    
    parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        help="Exchange name (default: binance)"
    )
    
    # Utility options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running"
    )
    
    return parser


async def main():
    """Main entry point."""
    tprint_info("🎯 Starting Training Pipelines Orchestrator...")
    
    # Create CLI parser
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        tprint_info("Verbose logging enabled")
    
    # Create orchestrator
    orchestrator = TrainingPipelineOrchestrator()
    
    # Validate ares_launcher
    if not orchestrator.validate_ares_launcher():
        tprint_error("Cannot proceed without ares_launcher.py")
        sys.exit(1)
    
    # Build configuration
    config = {
        "model": args.model,
        "direction": args.direction,
        "timeframe": args.timeframe,
        "mode": args.mode,
        "symbol": args.symbol,
        "exchange": args.exchange
    }
    
    tprint_data_preview(config, "execution_config")
    
    # Handle dry run
    if args.dry_run:
        tprint_info("DRY RUN - Would execute the following:")
        tprint_structured({
            "stage": args.stage,
            "configuration": config
        })
        
        if args.stage == "all":
            stages = ["pre_training", "models_training", "backtesting"]
            tprint_info("Stages to execute:")
            for stage in stages:
                tprint_info(f"  - {stage}")
        else:
            tprint_info(f"Stage to execute: {args.stage}")
        return
    
    try:
        if args.stage == "all":
            # Execute all stages in sequence
            tprint_info("Executing all stages in sequence...")
            
            all_results = {}
            stages = ["pre_training", "models_training", "backtesting"]
            
            for stage in stages:
                tprint_info(f"Executing stage: {stage}")
                results = await orchestrator.execute_stage(stage, config)
                all_results[stage] = results
                orchestrator.print_summary(results, stage)
                
                # Check if stage failed and stop if needed
                successful = sum(1 for r in results.values() if r.get("success", False))
                total = len(results)
                if successful < total:
                    tprint_error(f"Stage '{stage}' had failures. Stopping execution.")
                    break
            
            # Print overall summary
            tprint_info("OVERALL EXECUTION SUMMARY")
            overall_data = {
                "summary_type": "overall_execution",
                "stages": []
            }
            
            for stage, results in all_results.items():
                successful = sum(1 for r in results.values() if r.get("success", False))
                total = len(results)
                status = "COMPLETED" if successful == total else "FAILED"
                stage_summary = {
                    "stage": stage.upper(),
                    "successful_steps": successful,
                    "total_steps": total,
                    "status": status
                }
                overall_data["stages"].append(stage_summary)
                tprint_info(f"{stage.upper():<20} {successful:2d}/{total:2d} steps {status}")
            
            tprint_structured(overall_data)
            
        else:
            # Execute single stage
            tprint_info(f"Executing stage: {args.stage}")
            results = await orchestrator.execute_stage(args.stage, config)
            orchestrator.print_summary(results, args.stage)
            
    except Exception as e:
        tprint_error(f"Execution failed: {e}")
        sys.exit(1)
    
    tprint_success("✅ Training Pipelines Orchestrator completed")


if __name__ == "__main__":
    asyncio.run(main())