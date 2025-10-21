#!/usr/bin/env python3

"""
Simplified Ares Launcher - Autonomous Step Execution

This simplified launcher provides clean orchestration of autonomous pipeline steps
using the step registry pattern. Each step is independent and uses artifact_manager
for data persistence and outcome file generation.

Key Features:
- Simple step registry pattern
- Autonomous step execution
- Artifact management via artifact_manager
- Markdown outcome reports
- Clean CLI interface
- Legacy compatibility maintained
"""

import asyncio
import json
import logging
import os
import sys
import argparse
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ares.launcher")

# Import step registry and base step
from src.training.steps.base_step import step_registry, BaseStep

# Step packages will be imported lazily to avoid circular imports


class SimplifiedAresLauncher:
    """
    Simplified Ares Launcher using step registry pattern.
    
    Provides clean orchestration of autonomous steps with artifact management
    and outcome file generation.
    """
    
    def __init__(self):
        """Initialize the simplified launcher."""
        self.logger = logger
        self.step_registry = step_registry
        
    def register_step(self, step_name: str, step_class: type):
        """
        Register a step class.
        
        Args:
            step_name: Unique name for the step
            step_class: Step class that inherits from BaseStep
        """
        self.step_registry.register(step_name, step_class)
        self.logger.info(f"Registered step: {step_name}")
    
    async def run_step(self, step_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single autonomous step.
        
        Args:
            step_name: Name of the step to run
            config: Configuration dictionary
            
        Returns:
            Execution result from the step
        """
        try:
            self.logger.info(f"Starting execution of step: {step_name}")
            
            # Get step class from registry
            step_class = self.step_registry.get_step(step_name)
            
            # Create step instance with config
            step_instance = step_class(step_name, config)
            
            # Run the step
            result = await step_instance.run(config)
            
            # Log completion
            if result.get('success', False):
                self.logger.info(f"✅ Successfully completed step: {step_name}")
            else:
                self.logger.error(f"❌ Failed to complete step: {step_name}")
            
            return result
            
        except KeyError as e:
            error_msg = f"Step '{step_name}' not found in registry. Available steps: {self.step_registry.list_steps()}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'artifacts': [],
                'metrics': {}
            }
        except Exception as e:
            error_msg = f"Failed to run step '{step_name}': {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'artifacts': [],
                'metrics': {}
            }
    
    async def run_steps(self, step_names: List[str], config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Run multiple steps sequentially.
        
        Args:
            step_names: List of step names to run
            config: Configuration dictionary
            
        Returns:
            Dictionary mapping step names to their execution results
        """
        results = {}
        
        for step_name in step_names:
            self.logger.info(f"Running step {step_names.index(step_name) + 1}/{len(step_names)}: {step_name}")
            
            result = await self.run_step(step_name, config)
            results[step_name] = result
            
            # Stop on first failure unless configured otherwise
            if not result.get('success', False):
                self.logger.error(f"Stopping execution due to failure in step: {step_name}")
                break
        
        return results
    
    def run_stage(self, stage_name: str, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Run all steps in a specific stage.
        
        Args:
            stage_name: Name of the stage (DATA_COLLECTION, MARKET_ANALYSIS, etc.)
            config: Configuration dictionary
            
        Returns:
            Dictionary mapping step names to their execution results
        """
        # Define stage step mappings
        stage_steps = {
            'DATA_COLLECTION': [
                'data_download', 'data_conversion', 'data_validation', 'data_preparation',
                'feature_engineering', 'data_resampling', 'gap_filling', 'data_quality_check',
                'data_integration', 'data_storage', 'data_monitoring', 'data_export'
            ],
            'MARKET_ANALYSIS': [
                'sr_parameter_optimization', 'sr_detection', 'sr_clustering',
                'hdbscan_regime_discovery',  # New HDBSCAN-based regime discovery
                'regime_clustering',  # New regime clustering step using clusters/ folder
                'regime_models_training', 'regime_ensemble_training', 'regime_data_splitting',
                'model_persistence'  # Model persistence step
            ],
            'PRE_TRAINING': [
                'feature_generation_data_validation_step',
                'feature_generation_labeling_integration_step',
                'feature_generation_feature_generation_step',
                'feature_generation_period_lookback_optimization_step',
                'feature_generation_feature_selection_step',
                'feature_generation_interaction_generation_step_analyst',
                'feature_generation_interaction_generation_step_tactician',
                'feature_generation_final_feature_selection_step',
                'feature_generation_final_validation_step'
            ],
            'MODEL_TRAINING': [
                'analyst_base_training', 'analyst_ensemble_training',
                'tactician_base_training', 'tactician_ensemble_training'
            ],
            'BACKTESTING': [
                'basic_backtesting_pre', 'final_parameters_optimization',
                'basic_backtesting_post', 'walk_forward_validation',
                'monte_carlo_simulation', 'ab_testing', 'reporting'
            ]
        }
        
        if stage_name not in stage_steps:
            error_msg = f"Unknown stage: {stage_name}. Available stages: {list(stage_steps.keys())}"
            self.logger.error(error_msg)
            return {}
        
        step_names = stage_steps[stage_name]
        self.logger.info(f"Running stage '{stage_name}' with {len(step_names)} steps")
        
        return self.run_steps(step_names, config)
    
    def list_steps(self) -> List[str]:
        """
        List all registered steps.
        
        Returns:
            List of registered step names
        """
        return self.step_registry.list_steps()
    
    def list_stages(self) -> List[str]:
        """
        List all available stages.
        
        Returns:
            List of stage names
        """
        return ['DATA_COLLECTION', 'MARKET_ANALYSIS', 'PRE_TRAINING', 'MODEL_TRAINING', 'BACKTESTING']
    
    def is_step_available(self, step_name: str) -> bool:
        """
        Check if a specific step is available.
        
        Args:
            step_name: Name of the step to check
            
        Returns:
            True if step is available, False otherwise
        """
        try:
            self.step_registry.get_step(step_name)
            return True
        except KeyError:
            return False


def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Simplified Ares Launcher - Autonomous Step Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single step (new simplified format)
  python ares_launcher.py feature_generation_period_lookback_optimization_step --symbol ETHUSDT --execution-mode light
  
  # Run with additional parameters
  python ares_launcher.py feature_generation_data_validation_step --symbol ETHUSDT --timeframe 15m --exchange binance --execution-mode full
  
  # Run multiple steps
  python ares_launcher.py --steps data_download,data_conversion --symbol ETHUSDT
  
  # Run entire stage
  python ares_launcher.py --stage DATA_COLLECTION --symbol ETHUSDT
  
  # Model training steps
  python ares_launcher.py --train-analyst-base --symbol ETHUSDT --timeframe 15m --direction longs
  
  # List available steps
  python ares_launcher.py --list-steps
        """
    )
    
    # Step execution options
    step_group = parser.add_mutually_exclusive_group(required=True)
    step_group.add_argument('step_name', nargs='?', type=str, help='Name of the step to run (positional argument)')
    step_group.add_argument('--steps', type=str, help='Run multiple steps (comma-separated)')
    step_group.add_argument('--stage', type=str, help='Run entire stage')
    step_group.add_argument('--mode', type=str, help='Legacy mode (sequential, etc.)')
    step_group.add_argument('--sub_pipeline', type=str, help='Legacy sub-pipeline execution')
    step_group.add_argument('--list-steps', action='store_true', help='List all registered steps')
    step_group.add_argument('--list-stages', action='store_true', help='List all available stages')
    
    # Model training options (included in main group)
    step_group.add_argument('--train-analyst-base', action='store_true', help='Train analyst base models')
    step_group.add_argument('--train-analyst-ensemble', action='store_true', help='Train analyst ensemble models')
    step_group.add_argument('--train-tactician-base', action='store_true', help='Train tactician base models')
    step_group.add_argument('--train-tactician-ensemble', action='store_true', help='Train tactician ensemble models')
    
    # Regime discovery options
    regime_group = parser.add_mutually_exclusive_group()
    regime_group.add_argument('--hdbscan-regime-discovery', action='store_true', help='Run HDBSCAN regime discovery (replaces NAS/TAS)')
    regime_group.add_argument('--legacy-nas-tas', action='store_true', help='Run legacy NAS/TAS regime discovery (deprecated)')
    
    # Common parameters
    parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., ETHUSDT)')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe for training')
    parser.add_argument('--direction', type=str, choices=['longs', 'shorts', 'both'], default='longs', help='Trading direction')
    parser.add_argument('--execution-mode', type=str, choices=['full', 'light', 'blank'], default='light', help='Execution mode')
    
    # Utility options
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    return parser


def load_step_modules():
    """Lazily load step modules to avoid circular imports with enhanced resilience."""
    loaded_modules = []
    failed_modules = []
    partial_modules = []
    
    # List of modules to load in order of dependency
    modules_to_load = [
        ("data_collection", "src.training.steps.data_collection"),
        ("pre_training", "src.training.steps.pre_training"),
        ("market_analysis", "src.training.steps.market_analysis"),
        ("model_training", "src.training.steps.model_training"),
        ("backtesting", "src.training.steps.backtesting"),
    ]
    
    for module_name, module_path in modules_to_load:
        try:
            # Try to import the module
            module = __import__(module_path)
            loaded_modules.append(module_name)
            print(f"✅ Loaded {module_name}")
            
        except ImportError as e:
            error_msg = str(e)
            failed_modules.append((module_name, error_msg))
            
            # Check if it's a missing dependency vs structural issue
            if any(dep in error_msg.lower() for dep in ['matplotlib', 'hmmlearn', 'optuna', 'vectorbt', 'shap']):
                print(f"⚠️ Failed to load {module_name}: Missing dependency - {error_msg}")
            else:
                print(f"❌ Failed to load {module_name}: Import error - {error_msg}")
                
        except Exception as e:
            error_msg = str(e)
            failed_modules.append((module_name, error_msg))
            print(f"❌ Error loading {module_name}: {error_msg}")
    
    # Try to load individual step components if main modules fail
    if not loaded_modules:
        print("🔄 Attempting to load individual step components...")
        individual_steps = [
            ("tactician_ensemble_training", "src.training.steps.models_training.components.tactician_ensemble_training"),
            ("analyst_ensemble_training", "src.training.steps.models_training.components.analyst_ensemble_training"),
            ("tactician_base_training", "src.training.steps.models_training.components.tactician_base_training"),
            ("analyst_base_training", "src.training.steps.models_training.components.analyst_base_training"),
        ]
        
        for step_name, step_path in individual_steps:
            try:
                __import__(step_path)
                partial_modules.append(step_name)
                print(f"✅ Loaded individual step: {step_name}")
            except Exception as e:
                print(f"⚠️ Failed to load individual step {step_name}: {e}")
    
    # Report results
    if loaded_modules:
        print(f"✅ Successfully loaded {len(loaded_modules)} step modules: {', '.join(loaded_modules)}")
        if partial_modules:
            print(f"✅ Also loaded {len(partial_modules)} individual steps: {', '.join(partial_modules)}")
        return True
    elif partial_modules:
        print(f"⚠️ Partial success: Loaded {len(partial_modules)} individual steps: {', '.join(partial_modules)}")
        return True
    else:
        print("❌ No step modules could be loaded")
        if failed_modules:
            print("Failed modules:")
            for module_name, error in failed_modules:
                print(f"  - {module_name}: {error}")
        return False

def main():
    """Main entry point."""
    logger.info("🎯 Starting Simplified Ares Launcher...")
    
    # Load step modules lazily with enhanced resilience
    step_loading_success = load_step_modules()
    if not step_loading_success:
        print("⚠️ Warning: Some step modules failed to load, but continuing with available modules...")
        # Don't exit immediately - let the launcher try to work with what's available
    
    # Create CLI parser
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create launcher instance
    launcher = SimplifiedAresLauncher()
    
    # Handle utility commands
    if args.list_steps:
        steps = launcher.list_steps()
        print("Registered steps:")
        for step in steps:
            print(f"  - {step}")
        return
    
    if args.list_stages:
        stages = launcher.list_stages()
        print("Available stages:")
        for stage in stages:
            print(f"  - {stage}")
        return
    
    # Validate required arguments for execution commands
    if args.step_name and not args.symbol:
        parser.error("--symbol is required for step execution")
    
    # Build configuration
    config = {
        'symbol': args.symbol,
        'exchange': args.exchange,
        'timeframe': args.timeframe,
        'direction': args.direction,
        'execution_mode': args.execution_mode
    }
    
    # Handle different execution modes
    try:
        if args.step_name:
            # Single step execution (new simplified format)
            logger.info(f"Running single step: {args.step_name}")
            result = asyncio.run(launcher.run_step(args.step_name, config))
            print(f"Step '{args.step_name}' completed: {'✅ Success' if result.get('success') else '❌ Failed'}")
            
        elif args.steps:
            # Multiple steps execution
            step_names = [s.strip() for s in args.steps.split(',')]
            logger.info(f"Running multiple steps: {step_names}")
            results = asyncio.run(launcher.run_steps(step_names, config))
            
            # Print summary
            successful = sum(1 for r in results.values() if r.get('success', False))
            total = len(results)
            print(f"Steps completed: {successful}/{total}")
            
        elif args.stage:
            # Stage execution
            logger.info(f"Running stage: {args.stage}")
            results = launcher.run_stage(args.stage, config)
            
            # Print summary
            successful = sum(1 for r in results.values() if r.get('success', False))
            total = len(results)
            print(f"Stage '{args.stage}' completed: {successful}/{total} steps successful")
            
        elif args.mode == 'sequential' and args.sub_pipeline:
            # Legacy sequential sub-pipeline execution
            logger.info(f"Running legacy sub-pipeline: {args.sub_pipeline}")
            result = asyncio.run(launcher.run_step(args.sub_pipeline, config))
            print(f"Sub-pipeline '{args.sub_pipeline}' completed: {'✅ Success' if result.get('success') else '❌ Failed'}")
            
        elif any([args.train_analyst_base, args.train_analyst_ensemble, args.train_tactician_base, args.train_tactician_ensemble]):
            # Model training execution
            if args.train_analyst_base:
                step_name = 'analyst_base_training'
            elif args.train_analyst_ensemble:
                step_name = 'analyst_ensemble_training'
            elif args.train_tactician_base:
                step_name = 'tactician_base_training'
            elif args.train_tactician_ensemble:
                step_name = 'tactician_ensemble_training'
            
            # Check if step is available
            if not launcher.is_step_available(step_name):
                print(f"❌ Error: Step '{step_name}' is not available.")
                print("Available steps:")
                for step in launcher.list_steps():
                    print(f"  - {step}")
                return 1
            
            logger.info(f"Running model training: {step_name}")
            result = asyncio.run(launcher.run_step(step_name, config))
            print(f"Model training '{step_name}' completed: {'✅ Success' if result.get('success') else '❌ Failed'}")
            
        elif args.hdbscan_regime_discovery:
            # HDBSCAN regime discovery execution
            logger.info("Running HDBSCAN regime discovery (replaces NAS/TAS)")
            result = asyncio.run(launcher.run_step('hdbscan_regime_discovery', config))
            print(f"HDBSCAN regime discovery completed: {'✅ Success' if result.get('success') else '❌ Failed'}")
            
        elif args.legacy_nas_tas:
            # Legacy NAS/TAS regime discovery execution (deprecated)
            logger.warning("Running legacy NAS/TAS regime discovery (deprecated - use --hdbscan-regime-discovery instead)")
            # For now, redirect to HDBSCAN until legacy is fully removed
            result = asyncio.run(launcher.run_step('hdbscan_regime_discovery', config))
            print(f"Legacy NAS/TAS regime discovery (redirected to HDBSCAN) completed: {'✅ Success' if result.get('success') else '❌ Failed'}")
            
        else:
            parser.error("Please specify a valid execution mode")
            
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)
    
    logger.info("✅ Simplified Ares Launcher completed")


if __name__ == "__main__":
    main()