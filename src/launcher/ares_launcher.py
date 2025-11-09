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
import re
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path


FEATURE_GENERATION_STEP_FLAGS = [
    'feature_generation_data_validation_step',
    'feature_generation_labeling_integration_step',
    'feature_generation_feature_generation_step',
    'feature_generation_period_lookback_optimization_step',
    'feature_generation_interaction_generation_step',
    'feature_generation_final_feature_selection_step',
    'feature_generation_final_validation_step',
]

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

# Import step packages to register them
import src.training.steps.data_collection  # Registers DATA_COLLECTION steps
import src.training.steps.market_analysis  # Registers MARKET_ANALYSIS steps
import src.training.steps.pre_training  # Registers PRE_TRAINING steps
import src.training.steps.model_training  # Registers MODEL_TRAINING steps
# import src.training.steps.backtesting  # Registers BACKTESTING steps - temporarily disabled due to import issues


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
            
            # Create step instance
            step_instance = step_class(step_name)
            
            # Run the step (async)
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
    
    async def run_stage(self, stage_name: str, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
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
                'sr_detection', 'sr_clustering', 'sr_parameter_optimization',  # Fixed order: detection -> clustering -> optimization
                'statsmodel_clustering_pipeline',  # Statsmodel Markov-switching clustering
                'sticky_finite_hmm_regime_discovery',  # Sticky Finite HMM regime discovery (K=5, VB inference)
                'rolling_hmm_regime_discovery',  # Rolling HMM regime discovery with EWMA features and HPO
                'regime_feature_selection',  # Enhanced regime feature selection
                'regime_models_training', 'regime_ensemble_training'
            ],
            'PRE_TRAINING': [
                'feature_generation_data_validation_step',
                'feature_generation_labeling_integration_step',
                'feature_generation_feature_generation_step',
                'feature_generation_period_lookback_optimization_step',
                'feature_generation_interaction_generation_step',
                'feature_generation_final_feature_selection_step',
                'feature_generation_final_validation_step'
            ],
            'MODEL_TRAINING': [
                'analyst_base_training',
                'analyst_ensemble_training',
                'tactician_base_training',
                'tactician_ensemble_training'
            ],
            'BACKTESTING': [
                'feature_generation_data_validation_step',
                'feature_generation_labeling_integration_step',
                'feature_generation_feature_generation_step',
                'feature_generation_period_lookback_optimization_step',
                'feature_generation_interaction_generation_step',
                'feature_generation_final_feature_selection_step',
                'feature_generation_final_validation_step'
            ]
        }
        
        if stage_name not in stage_steps:
            error_msg = f"Unknown stage: {stage_name}. Available stages: {list(stage_steps.keys())}"
            self.logger.error(error_msg)
            return {}
        
        step_names = stage_steps[stage_name]
        self.logger.info(f"Running stage '{stage_name}' with {len(step_names)} steps")
        
        return await self.run_steps(step_names, config)
    
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


def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Simplified Ares Launcher - Autonomous Step Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single step (positional argument)
  python ares_launcher.py regime_models_training --symbol ETHUSDT --execution-mode light
  python ares_launcher.py regime_ensemble_training --symbol ETHUSDT --execution-mode light

  # Run a single step (named argument)
  python ares_launcher.py --step data_download --symbol ETHUSDT --exchange binance

  # Run multiple steps
  python ares_launcher.py --steps data_download,data_conversion --symbol ETHUSDT

  # Run entire stage
  python ares_launcher.py --stage DATA_COLLECTION --symbol ETHUSDT

  # PRE_TRAINING steps (maintain compatibility)
  python ares_launcher.py --step feature_generation_data_validation_step --symbol ETHUSDT --execution-mode light

  # MODEL_TRAINING steps (maintain compatibility)
  python ares_launcher.py --train-analyst-base --symbol ETHUSDT --timeframe 15m --direction long

  # FEATURE GENERATION INTERACTION GENERATION (differentiated modes)
  python ares_launcher.py --run-tactician-interaction --symbol ETHUSDT --timeframe 15m
  python ares_launcher.py --run-analyst-interaction --symbol ETHUSDT --timeframe 15m
  python ares_launcher.py --run-both-interaction-modes --symbol ETHUSDT --timeframe 15m

  # REGIME TRAINING (ML models and ensembles)
  python ares_launcher.py --regime-models-training --symbol ETHUSDT --execution-mode light
  python ares_launcher.py --regime-ensemble-training --symbol ETHUSDT --execution-mode light

  # Legacy compatibility
  python ares_launcher.py --mode sequential --sub_pipeline feature_generation_data_validation_step --symbol ETHUSDT
        """
    )

    # Positional argument for step name (optional)
    parser.add_argument('command', nargs='?', type=str, help='Step name to execute (e.g., regime_models_training, regime_ensemble_training)')

    # Step execution options
    step_group = parser.add_mutually_exclusive_group(required=False)
    step_group.add_argument('--step', type=str, help='Run a single step')
    step_group.add_argument('--steps', type=str, help='Run multiple steps (comma-separated)')
    step_group.add_argument('--stage', type=str, help='Run entire stage')
    step_group.add_argument('--mode', type=str, help='Legacy mode (sequential, etc.)')
    step_group.add_argument('--sub_pipeline', type=str, help='Legacy sub-pipeline execution')
    
    # Model training options (maintain compatibility)
    training_group = parser.add_mutually_exclusive_group()
    training_group.add_argument('--train-analyst-base', action='store_true', help='Train analyst base models')
    training_group.add_argument('--train-analyst-ensemble', action='store_true', help='Train analyst ensemble models')
    training_group.add_argument('--train-tactician-base', action='store_true', help='Train tactician base models')
    training_group.add_argument('--train-tactician-ensemble', action='store_true', help='Train tactician ensemble models')
    
    # Feature generation interaction generation options
    interaction_group = parser.add_mutually_exclusive_group()
    interaction_group.add_argument('--run-tactician-interaction', action='store_true', help='Run feature generation interaction generation in Tactician mode (MI-based)')
    interaction_group.add_argument('--run-analyst-interaction', action='store_true', help='Run feature generation interaction generation in Analyst mode (CMI-based)')
    interaction_group.add_argument('--run-both-interaction-modes', action='store_true', help='Run feature generation interaction generation in both Tactician and Analyst modes')
    
    # Regime discovery options
    regime_group = parser.add_mutually_exclusive_group()
    regime_group.add_argument('--rolling-hmm-regime-discovery', action='store_true', help='Run Rolling HMM regime discovery with EWMA features and HPO')
    regime_group.add_argument('--regime-models-training', action='store_true', help='Train machine learning models for regime classification')
    regime_group.add_argument('--regime-ensemble-training', action='store_true', help='Train ensemble models for regime classification using meta-learning')

    # Feature generation step shortcuts
    feature_group = parser.add_argument_group('Feature generation step shortcuts')
    for flag in FEATURE_GENERATION_STEP_FLAGS:
        friendly_name = flag.replace('_', ' ')
        feature_group.add_argument(
            f'--{flag}',
            action='store_true',
            help=f"Run the '{friendly_name}' step"
        )
    
    # Common parameters
    parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., ETHUSDT)')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe for training')
    parser.add_argument('--direction', type=str, choices=['long', 'short', 'both'], default='long', help='Trading direction')
    parser.add_argument('--execution-mode', type=str, choices=['full', 'light', 'blank'], default='light', help='Execution mode')
    
    # Legacy compatibility options
    parser.add_argument('--start-from-step-name', type=str, help='Legacy: start from specific step')
    parser.add_argument('--stop-at-step', type=int, help='Legacy: stop at specific step number')
    
    # Utility options
    parser.add_argument('--list-steps', action='store_true', help='List all registered steps')
    parser.add_argument('--list-stages', action='store_true', help='List all available stages')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    return parser


def cleanup_duplicate_files(directories: List[str], keep_count: int = 5):
    """
    Clean up duplicate files in specified directories.
    
    Duplicates are identified by base filename (without datetime suffix),
    and only the keep_count youngest files are kept.
    
    Args:
        directories: List of directory paths to clean
        keep_count: Number of youngest files to keep per group
    """
    logger.info("🧹 Starting cleanup of duplicate files...")
    
    # Pattern to match datetime suffixes like: _20251026_223845
    datetime_pattern = re.compile(r'_(\d{8}_\d{6})(\.[a-zA-Z]+)?$')
    
    total_deleted = 0
    total_skipped = 0
    
    for directory in directories:
        if not os.path.exists(directory):
            logger.debug(f"Directory does not exist, skipping: {directory}")
            continue
        
        # Special handling for logs directory - keep 100 youngest
        current_keep_count = 100 if directory == 'logs' else keep_count
        logger.info(f"Cleaning directory: {directory} (keeping {current_keep_count} youngest files per group)")
        
        # Get all files in the directory
        try:
            all_files = list(Path(directory).glob('*'))
            files = [f for f in all_files if f.is_file()]
        except Exception as e:
            logger.error(f"Error reading directory {directory}: {e}")
            continue
        
        if not files:
            logger.debug(f"No files found in {directory}")
            continue
        
        # Group files by base name (without datetime suffix)
        file_groups = {}
        
        for file_path in files:
            file_name = file_path.name
            
            # Try to extract base name and datetime
            match = datetime_pattern.search(file_name)
            if match:
                # Has datetime suffix
                datetime_str = match.group(1)
                extension = match.group(2) if match.group(2) else ''
                # Extract the base name before the datetime
                base_name = file_name[:match.start()] + extension
            else:
                # No datetime suffix - treat as unique
                base_name = file_name
                datetime_str = None
            
            if base_name not in file_groups:
                file_groups[base_name] = []
            
            file_groups[base_name].append({
                'path': file_path,
                'name': file_name,
                'datetime': datetime_str,
                'mtime': file_path.stat().st_mtime
            })
        
        # Process each group
        for base_name, files_in_group in file_groups.items():
            if len(files_in_group) <= current_keep_count:
                # Not enough files to clean
                continue
            
            # Sort by modification time (newest first)
            files_in_group.sort(key=lambda x: x['mtime'], reverse=True)
            
            # Keep the youngest current_keep_count files, delete the rest
            to_keep = files_in_group[:current_keep_count]
            to_delete = files_in_group[current_keep_count:]
            
            logger.info(f"  Group '{base_name}': {len(files_in_group)} files, keeping {len(to_keep)}, deleting {len(to_delete)}")
            
            for file_info in to_delete:
                try:
                    file_info['path'].unlink()
                    total_deleted += 1
                    logger.debug(f"    Deleted: {file_info['name']}")
                except Exception as e:
                    logger.error(f"    Failed to delete {file_info['name']}: {e}")
                    total_skipped += 1
    
    logger.info(f"✅ Cleanup complete: {total_deleted} files deleted, {total_skipped} files skipped")


async def main():
    """Main entry point."""
    logger.info("🎯 Starting Simplified Ares Launcher...")
    
    # Run cleanup before anything else
    directories_to_clean = [
        'logs',
        'artifacts',
        'outcomes'
    ]
    cleanup_duplicate_files(directories_to_clean, keep_count=5)
    
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
    
    # Handle positional command argument
    if args.command:
        # Check if the command matches a registered step
        if args.command in launcher.list_steps():
            # Treat positional argument as --step
            args.step = args.command
            logger.info(f"Detected positional command: {args.command}")
        else:
            print(f"Error: Unknown command '{args.command}'")
            print(f"Available steps: {', '.join(launcher.list_steps())}")
            print("Use --list-steps to see all registered steps")
            return

    # Map feature generation shortcut flags to step execution
    feature_step_flags = [flag for flag in FEATURE_GENERATION_STEP_FLAGS if getattr(args, flag, False)]
    if feature_step_flags:
        logger.info(f"Detected feature generation shortcuts: {feature_step_flags}")

        # Only apply feature generation shortcuts if no other step was specified via positional command
        # This prevents feature generation steps from interfering with other steps like regime_models_training
        if args.command:
            logger.warning(f"⚠️ Ignoring feature generation shortcuts because positional command '{args.command}' was provided")
        elif args.steps:
            existing_steps = [s.strip() for s in args.steps.split(',') if s.strip()]
            combined_steps = existing_steps + feature_step_flags
            args.steps = ','.join(combined_steps)
        elif args.step:
            combined_steps = [args.step] + feature_step_flags
            args.steps = ','.join(combined_steps)
            args.step = None
        else:
            if len(feature_step_flags) == 1:
                args.step = feature_step_flags[0]
            else:
                args.steps = ','.join(feature_step_flags)

        # Reset shortcut flags to avoid re-processing later
        for flag in feature_step_flags:
            setattr(args, flag, False)

    # Check if any execution mode is specified
    has_execution_mode = any([
        args.step, args.steps, args.stage, args.mode, args.sub_pipeline,
        args.train_analyst_base, args.train_analyst_ensemble,
        args.train_tactician_base, args.train_tactician_ensemble,
        args.run_tactician_interaction, args.run_analyst_interaction, args.run_both_interaction_modes,
        args.rolling_hmm_regime_discovery,
        args.regime_models_training, args.regime_ensemble_training
    ])

    if not has_execution_mode:
        print("No execution mode specified. Use --help to see available options.")
        print("Available utility commands:")
        print("  --list-steps    List all registered steps")
        print("  --list-stages   List all available stages")
        return

    # Validate required parameters for execution modes
    if not args.symbol and has_execution_mode:
        parser.error("--symbol is required when running execution modes")
    
    # Build configuration
    config = {
        'symbol': args.symbol,
        'exchange': args.exchange,
        'timeframe': args.timeframe,
        'direction': args.direction,
        'execution_mode': args.execution_mode
    }
    
    # Import tprint for troubleshooting output
    try:
        from src.utils.tprint import tprint
    except ImportError:
        # Fallback if tprint is not available
        def tprint(*args, **kwargs):
            print(*args)
    
    # Add mode-specific days configuration for regime models training
    tprint(f"🔧 CONFIG: Setting execution_mode={args.execution_mode} with mode-specific days", "INFO")
    if args.execution_mode == 'blank':
        config['blank_mode_days'] = 180
        tprint(f"🔧 CONFIG: Setting blank_mode_days=180", "INFO")
    elif args.execution_mode == 'light':
        config['light_mode_days'] = 20
        tprint(f"🔧 CONFIG: Setting light_mode_days=20", "INFO")
    
    # Add both as defaults for all modes (can be overridden by mode-specific values above)
    config.setdefault('blank_mode_days', 180)
    config.setdefault('light_mode_days', 20)
    
    # Set execution mode for HPO optimizations
    from src.utils.ml_common.optimization import set_execution_mode
    set_execution_mode(args.execution_mode)
    logger.info(f"🔧 Execution mode set to: {args.execution_mode.upper()}")
    
    # Handle different execution modes
    try:
        if args.step:
            # Single step execution
            logger.info(f"Running single step: {args.step}")
            result = await launcher.run_step(args.step, config)
            print(f"Step '{args.step}' completed: {'✅ Success' if result.get('success') else '❌ Failed'}")
            
        elif args.steps:
            # Multiple steps execution
            step_names = [s.strip() for s in args.steps.split(',')]
            logger.info(f"Running multiple steps: {step_names}")
            results = await launcher.run_steps(step_names, config)
            
            # Print summary
            successful = sum(1 for r in results.values() if r.get('success', False))
            total = len(results)
            print(f"Steps completed: {successful}/{total}")
            
        elif args.stage:
            # Stage execution
            logger.info(f"Running stage: {args.stage}")
            results = await launcher.run_stage(args.stage, config)
            
            # Print summary
            successful = sum(1 for r in results.values() if r.get('success', False))
            total = len(results)
            print(f"Stage '{args.stage}' completed: {successful}/{total} steps successful")
            
        elif args.mode == 'sequential' and args.sub_pipeline:
            # Legacy sequential sub-pipeline execution
            logger.info(f"Running legacy sub-pipeline: {args.sub_pipeline}")
            result = await launcher.run_step(args.sub_pipeline, config)
            print(f"Sub-pipeline '{args.sub_pipeline}' completed: {'✅ Success' if result.get('success') else '❌ Failed'}")
            
        elif any([args.train_analyst_base, args.train_analyst_ensemble, args.train_tactician_base, args.train_tactician_ensemble]):
            # Model training execution using specific training steps
            if args.train_analyst_base:
                step_name = 'analyst_base_training'
                training_type = 'analyst_base'
                config['execution_context'] = 'analyst'
            elif args.train_analyst_ensemble:
                step_name = 'analyst_ensemble_training'
                training_type = 'analyst_ensemble'
                config['execution_context'] = 'analyst'
            elif args.train_tactician_base:
                step_name = 'tactician_base_training'
                training_type = 'tactician_base'
                config['execution_context'] = 'tactician'
            elif args.train_tactician_ensemble:
                step_name = 'tactician_ensemble_training'
                training_type = 'tactician_ensemble'
                config['execution_context'] = 'tactician'
            
            # Add training type to config
            config['training_type'] = training_type
            
            logger.info(f"Running model training: {training_type}")
            result = await launcher.run_step(step_name, config)
            print(f"Model training '{training_type}' completed: {'✅ Success' if result.get('success') else '❌ Failed'}")
            
        elif args.rolling_hmm_regime_discovery:
            # Rolling HMM regime discovery execution
            logger.info("Running Rolling HMM regime discovery with EWMA features and HPO")
            # Ensure HPO is enabled by default unless explicitly disabled
            config['enable_auto_tuning'] = True
            result = await launcher.run_step('rolling_hmm_regime_discovery', config)
            print(f"Rolling HMM regime discovery completed: {'✅ Success' if result.get('success') else '❌ Failed'}")

        elif args.regime_models_training:
            # Regime models training execution
            logger.info("Training machine learning models for regime classification")
            result = await launcher.run_step('regime_models_training', config)
            print(f"Regime models training completed: {'✅ Success' if result.get('success') else '❌ Failed'}")

        elif args.regime_ensemble_training:
            # Regime ensemble training execution
            logger.info("Training ensemble models for regime classification using meta-learning")
            result = await launcher.run_step('regime_ensemble_training', config)
            print(f"Regime ensemble training completed: {'✅ Success' if result.get('success') else '❌ Failed'}")

        elif args.run_tactician_interaction:
            # Tactician mode interaction generation (MI-based)
            logger.info("Running feature generation interaction generation in Tactician mode (MI-based)")
            config['execution_context'] = 'tactician'
            config['interaction_generation_mode'] = 'tactician'
            result = await launcher.run_step('feature_generation_interaction_generation_step', config)
            print(f"Tactician interaction generation completed: {'✅ Success' if result.get('success') else '❌ Failed'}")
            
        elif args.run_analyst_interaction:
            # Analyst mode interaction generation (CMI-based)
            logger.info("Running feature generation interaction generation in Analyst mode (CMI-based)")
            config['execution_context'] = 'analyst'
            config['interaction_generation_mode'] = 'analyst'
            result = await launcher.run_step('feature_generation_interaction_generation_step', config)
            print(f"Analyst interaction generation completed: {'✅ Success' if result.get('success') else '❌ Failed'}")
            
        elif args.run_both_interaction_modes:
            # Both modes interaction generation
            logger.info("Running feature generation interaction generation in both Tactician and Analyst modes")
            
            # Run Tactician mode first
            logger.info("Step 1/2: Running Tactician mode (MI-based)")
            tactician_config = config.copy()
            tactician_config['execution_context'] = 'tactician'
            tactician_config['interaction_generation_mode'] = 'tactician'
            tactician_result = await launcher.run_step('feature_generation_interaction_generation_step', tactician_config)
            print(f"Tactician interaction generation completed: {'✅ Success' if tactician_result.get('success') else '❌ Failed'}")
            
            # Run Analyst mode second
            logger.info("Step 2/2: Running Analyst mode (CMI-based)")
            analyst_config = config.copy()
            analyst_config['execution_context'] = 'analyst'
            analyst_config['interaction_generation_mode'] = 'analyst'
            analyst_result = await launcher.run_step('feature_generation_interaction_generation_step', analyst_config)
            print(f"Analyst interaction generation completed: {'✅ Success' if analyst_result.get('success') else '❌ Failed'}")
            
            # Summary
            tactician_success = tactician_result.get('success', False)
            analyst_success = analyst_result.get('success', False)
            print(f"Both modes completed: Tactician={'✅' if tactician_success else '❌'}, Analyst={'✅' if analyst_success else '❌'}")
            
        else:
            parser.error("Please specify a valid execution mode")
            
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)
    
    logger.info("✅ Simplified Ares Launcher completed")


if __name__ == "__main__":
    asyncio.run(main())