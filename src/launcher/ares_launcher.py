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

# Centralized lookback configuration for execution modes
MODE_LOOKBACK_DAYS: Dict[str, int] = {
    "light": 30,     # 30 days
    "blank": 360,    # 1 year
    "full": 365 * 3  # 3 years
}

def get_mode_lookback_days(mode: str) -> int:
    """Return centralized lookback days for a given execution mode."""
    return MODE_LOOKBACK_DAYS.get(mode, MODE_LOOKBACK_DAYS["light"])


FEATURE_GENERATION_STEP_FLAGS = [
    'feature_generation_data_validation_step',
    'feature_generation_labeling_integration_step',
    'feature_generation_feature_generation_step',
    'feature_generation_period_lookback_optimization_step',
    'feature_generation_interaction_generation_step',
    'regime_aware_feature_interaction_generation_step',
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
from src.utils.versioned_artifacts import VersionedArtifactStore

# Import step packages to register them
import src.training.steps.data_collection  # Registers DATA_COLLECTION steps

try:
    import src.training.steps.market_analysis  # Registers MARKET_ANALYSIS steps
except Exception as e:
    logger.warning(
        "MARKET_ANALYSIS steps could not be imported and will be unavailable: %s",
        e,
    )

try:
    import src.training.steps.pre_training  # Registers PRE_TRAINING steps
except Exception as e:
    logger.warning(
        "PRE_TRAINING steps could not be imported and will be unavailable: %s",
        e,
    )

try:
    import src.training.steps.model_training  # Registers MODEL_TRAINING steps
except Exception as e:
    logger.warning(
        "MODEL_TRAINING steps could not be imported and will be unavailable: %s",
        e,
    )

try:
    import src.training.steps.analyst_base_backtest_step  # Registers ANALYST_BASE_BACKTEST step
except Exception as e:
    logger.warning(
        "ANALYST_BASE_BACKTEST step could not be imported and will be unavailable: %s",
        e,
    )

try:
    import src.training.steps.labeling  # Registers LABELING steps (includes META_GATED_BACKTEST)
except Exception as e:
    logger.warning(
        "LABELING steps could not be imported and will be unavailable: %s",
        e,
    )

try:
    import src.training.steps.backtesting  # Registers BACKTESTING steps
except Exception as e:
    logger.warning(
        "BACKTESTING steps could not be imported and will be unavailable: %s",
        e,
    )


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
                'regime_ensemble_training'
            ],
            'PRE_TRAINING': [
                'feature_generation_data_validation_step',
                'feature_generation_labeling_integration_step',
                'feature_generation_feature_generation_step',
                'feature_generation_period_lookback_optimization_step',
                'feature_generation_interaction_generation_step',
                'regime_aware_feature_interaction_generation_step',
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
                'regime_aware_feature_interaction_generation_step',
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
  python ares_launcher.py --regime-ensemble-training --symbol ETHUSDT --execution-mode light

  # Legacy compatibility
  python ares_launcher.py --mode sequential --sub_pipeline feature_generation_data_validation_step --symbol ETHUSDT
        """
    )

    # Positional argument for step name (optional)
    parser.add_argument('command', nargs='?', type=str, help='Step name to execute (e.g., regime_ensemble_training)')

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
    training_group.add_argument('--train-gate', action='store_true', help='Train gate model')
    
    # Feature generation interaction generation options
    interaction_group = parser.add_mutually_exclusive_group()
    interaction_group.add_argument('--run-tactician-interaction', action='store_true', help='Run feature generation interaction generation in Tactician mode (MI-based)')
    interaction_group.add_argument('--run-analyst-interaction', action='store_true', help='Run feature generation interaction generation in Analyst mode (CMI-based)')
    interaction_group.add_argument('--run-both-interaction-modes', action='store_true', help='Run feature generation interaction generation in both Tactician and Analyst modes')
    
    # Regime discovery options
    regime_group = parser.add_mutually_exclusive_group()
    regime_group.add_argument('--rolling-hmm-regime-discovery', action='store_true', help='Run Rolling HMM regime discovery with EWMA features and HPO')
    regime_group.add_argument('--regime-ensemble-training', action='store_true', help='Train ensemble models for regime classification using meta-learning')
    regime_group.add_argument('--hmm-macro-regime', action='store_true', help='Run HMM macro alpha / macro regime step from Rolling HMM outputs')
    regime_group.add_argument('--xgb-meso-regime', action='store_true', help='Run XGB Meso Trend regime step')
    regime_group.add_argument('--final_parameters_optimization', action='store_true', help='Run final parameters optimization')

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
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe for training / base features')
    # Default regime timeframe to 15m so regime-aware steps (HMM/alpha, regime ensemble,
    # unified training) use the same timeframe as base features.
    parser.add_argument('--regime-timeframe', type=str, default='15m', help='Timeframe used for regime detection/ensemble (default: 15m)')
    parser.add_argument('--direction', type=str, choices=['long', 'short', 'both'], default='long', help='Trading direction')
    parser.add_argument('--execution-mode', type=str, choices=['full', 'light', 'blank'], default='light', help='Execution mode')
    parser.add_argument(
        '--min-interaction-mi-lift',
        type=float,
        default=None,
        help='Minimum MI lift required for interaction selection; overrides default when provided'
    )

    # Alpha/HPO options (used by hmm_ml_alpha_step and other alpha-related steps)
    parser.add_argument(
        '--alpha-enable-hpo',
        action='store_true',
        help='Enable hierarchical HPO for alpha model training (e.g., in hmm_ml_alpha_step)'
    )

    # Global HPO toggle (used by ml_risk_regime_step and unified model training)
    parser.add_argument(
        '--enable-hpo',
        action='store_true',
        help='Enable hyperparameter optimization for compatible steps (e.g., ml_risk_regime_step, unified model training)'
    )

    # Labeling HPO integration (used by feature_generation_meta_labeling_step)
    parser.add_argument(
        '--enable-labeling-hpo-params',
        action='store_true',
        help='Use latest meta-labeling HPO best-params when running feature_generation_meta_labeling_step'
    )
    parser.add_argument(
        '--labeling-hpo-use-best-params',
        action='store_true',
        help='Alias for --enable-labeling-hpo-params; use latest meta-labeling HPO best-params in labeling steps'
    )

    # Force recomputation of multi-stage labeling HPO (ignore cached best params)
    parser.add_argument(
        '--force-hpo',
        action='store_true',
        help='Force recomputation of multi-stage labeling HPO for compatible steps (ignore cached results)'
    )

    parser.add_argument(
        '--labeling-hpo-start-at',
        type=str,
        default=None,
        choices=[
            'layer0', 'layer1', 'layer2', 'feature_selection', 'layer3',
            'layer4',
            'stage0', 'kalman', 'weighting', 'trading', 'model',
            '0', '1', '2', '3', '4', 'fs'
        ],
        help=(
            'Start multi-stage meta-labeling HPO at a specific step. Steps before the start point '
            'reuse the latest persisted best-params, while the start point and subsequent steps '
            're-run HPO. Choices: layer0/layer1/layer2/feature_selection/layer3 (aliases: stage0, kalman, weighting, trading, model, fs, 0-3).'
        )
    )
    
    parser.add_argument(
        '--meta-permutation-test',
        action='store_true',
        help='Enable permutation test diagnostics in meta_gated_backtest'
    )
    parser.add_argument(
        '--meta-permutation-repeats',
        type=int,
        default=None,
        help='Number of permutation repeats for meta_gated_backtest'
    )
    parser.add_argument(
        '--meta-forward-walk-n-windows',
        type=int,
        default=None,
        help='Number of forward-walk evaluation windows for meta_gated_backtest'
    )
    
    # Weighted meta-labeling utilities
    parser.add_argument(
        '--save-labeled-data-csv',
        action='store_true',
        help='Save labeled data CSVs in outcomes/ (e.g., weighted_labeled_data_*)'
    )
    
    # Legacy compatibility options
    parser.add_argument('--start-from-step-name', type=str, help='Legacy: start from specific step')
    parser.add_argument('--stop-at-step', type=int, help='Legacy: stop at specific step number')
    
    # Utility options
    parser.add_argument('--list-steps', action='store_true', help='List all registered steps')
    parser.add_argument('--list-stages', action='store_true', help='List all available stages')
    parser.add_argument(
        '--cleanup-only',
        action='store_true',
        help=(
            'Run the launcher cleanup routines only (no step execution) and exit. '
            "Includes duplicate-file cleanup and versioned artifact store pruning unless '--cleanup-skip-versioned-artifacts' is set."
        )
    )
    parser.add_argument(
        '--cleanup-duplicates-keep-count',
        type=int,
        default=3,
        help='For duplicate-file cleanup, keep this many newest files per base name (default: 3; logs uses 100 regardless).'
    )
    parser.add_argument(
        '--cleanup-keep-per-base',
        type=int,
        default=5,
        help='For VersionedArtifactStore pruning, keep this many most recent versions per base (default: 5).'
    )
    parser.add_argument(
        '--cleanup-repair-versioned-artifacts',
        action='store_true',
        help=(
            'When running cleanup-only, reconcile VersionedArtifactStore metadata.json with store.h5 '
            'before pruning (removes metadata-only phantom versions and adds minimal metadata for HDF5-only versions).'
        )
    )
    parser.add_argument(
        '--cleanup-regime-models-keep-count',
        type=int,
        default=1,
        help=(
            'Rotate timestamped files under versioned_artifacts/regime_models by base name and keep this many newest per group '
            '(default: 1).'
        )
    )
    parser.add_argument(
        '--cleanup-skip-regime-models',
        action='store_true',
        help='When running --cleanup-only, skip rotation of versioned_artifacts/regime_models.'
    )
    parser.add_argument(
        '--cleanup-skip-versioned-artifacts',
        action='store_true',
        help='When running --cleanup-only, skip versioned_artifacts store pruning.'
    )
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
    
    # Pattern to match datetime suffixes like: _20251026_223845 or _20251026_223845_123
    datetime_pattern = re.compile(r'_(\d{8}_\d{6})(?:_\d+)?(\.[a-zA-Z]+)?$')
    
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

            # Guard against race conditions where a file may be deleted
            # between globbing and stat(). If this happens, simply skip the
            # missing file rather than failing the entire launcher on
            # FileNotFoundError.
            try:
                mtime = file_path.stat().st_mtime
            except FileNotFoundError:
                logger.debug(f"File disappeared during cleanup scan, skipping: {file_name}")
                continue
            
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
                'mtime': mtime,
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


def cleanup_regime_models_artifacts(keep_count: int = 1) -> None:
    """Rotate timestamped detector artifacts under versioned_artifacts/regime_models."""
    root = Path("versioned_artifacts/regime_models")
    if not root.exists():
        logger.debug("versioned_artifacts/regime_models directory does not exist, skipping")
        return

    logger.info(
        "🧹 Rotating regime_models artifacts: %s (keeping %d newest files per base name)",
        str(root),
        keep_count,
    )

    datetime_pattern = re.compile(r'_(\d{8}_\d{6})(?:_\d+)?(\.[a-zA-Z]+)?$')

    files = [f for f in root.glob('*') if f.is_file()]
    if not files:
        logger.info("No files found in versioned_artifacts/regime_models")
        return

    file_groups: Dict[str, List[Dict[str, Any]]] = {}
    total_deleted = 0
    total_skipped = 0

    for file_path in files:
        file_name = file_path.name
        try:
            mtime = file_path.stat().st_mtime
        except FileNotFoundError:
            continue

        match = datetime_pattern.search(file_name)
        if match:
            extension = match.group(2) if match.group(2) else ''
            base_name = file_name[:match.start()] + extension
        else:
            base_name = file_name

        file_groups.setdefault(base_name, []).append({
            'path': file_path,
            'name': file_name,
            'mtime': mtime,
        })

    for base_name, files_in_group in file_groups.items():
        if len(files_in_group) <= keep_count:
            continue

        files_in_group.sort(key=lambda x: x['mtime'], reverse=True)
        to_delete = files_in_group[keep_count:]

        logger.info(
            "  Group '%s': %d files, keeping %d, deleting %d",
            base_name,
            len(files_in_group),
            keep_count,
            len(to_delete),
        )

        for file_info in to_delete:
            try:
                file_info['path'].unlink()
                total_deleted += 1
            except Exception as e:
                logger.error(f"    Failed to delete {file_info['name']}: {e}")
                total_skipped += 1

    logger.info(
        "✅ Regime model artifact rotation complete: %d files deleted, %d files skipped",
        total_deleted,
        total_skipped,
    )


def cleanup_versioned_artifact_stores(keep_per_base: int = 5) -> None:
    """Prune old versions inside each VersionedArtifactStore."""
    logger.info(f"Starting cleanup of versioned artifact stores (keeping {keep_per_base} versions per base)")

    root = Path("versioned_artifacts")
    if not root.exists():
        logger.debug("versioned_artifacts directory does not exist, skipping store cleanup")
        return

    for store_dir in root.iterdir():
        if not store_dir.is_dir():
            continue

        h5_path = store_dir / "store.h5"
        if not h5_path.exists():
            continue

        try:
            logger.info(f"Cleaning VersionedArtifactStore at {store_dir}")
            store = VersionedArtifactStore(store_path=store_dir)
            summary = store.prune_versions(keep_per_base=keep_per_base)
            logger.info(
                "Pruned store %s: h5_only_removed=%d, meta_only_removed=%d, versions_pruned=%d (keep_per_base=%d)",
                store_dir.name,
                summary.get("h5_only_removed", 0),
                summary.get("meta_only_removed", 0),
                summary.get("versions_pruned", 0),
                keep_per_base,
            )
        except Exception as e:
            logger.error(f"Failed to clean VersionedArtifactStore at {store_dir}: {e}")


def cleanup_versioned_artifact_stores_repair(keep_per_base: int = 5) -> None:
    """Repair and then prune old versions inside each VersionedArtifactStore."""
    logger.info(
        "Starting REPAIR+cleanup of versioned artifact stores (keeping %d versions per base)",
        keep_per_base,
    )

    root = Path("versioned_artifacts")
    if not root.exists():
        logger.debug("versioned_artifacts directory does not exist, skipping store cleanup")
        return

    for store_dir in root.iterdir():
        if not store_dir.is_dir():
            continue

        h5_path = store_dir / "store.h5"
        if not h5_path.exists():
            continue

        try:
            logger.info(f"Repairing VersionedArtifactStore at {store_dir}")
            store = VersionedArtifactStore(store_path=store_dir)
            repair_summary = store.reconcile_metadata_with_hdf5()
            logger.info(
                "Repaired store %s: meta_only_removed=%d, h5_only_added=%d",
                store_dir.name,
                repair_summary.get("meta_only_removed", 0),
                repair_summary.get("h5_only_added", 0),
            )

            prune_summary = store.prune_versions(keep_per_base=keep_per_base)
            logger.info(
                "Pruned store %s: h5_only_removed=%d, meta_only_removed=%d, versions_pruned=%d (keep_per_base=%d)",
                store_dir.name,
                prune_summary.get("h5_only_removed", 0),
                prune_summary.get("meta_only_removed", 0),
                prune_summary.get("versions_pruned", 0),
                keep_per_base,
            )
        except Exception as e:
            logger.error(f"Failed to repair/clean VersionedArtifactStore at {store_dir}: {e}")


async def main():
    """Main entry point."""
    logger.info("Starting Simplified Ares Launcher...")
    
    # Create CLI parser
    parser = create_cli_parser()
    args = parser.parse_args()

    # Map positional command argument for utility commands
    if args.command in ("cleaner", "cleanup"):
        args.cleanup_only = True
        args.command = None
        logger.info("Detected cleanup-only utility command")

    # Run cleanup before anything else (skip if ARES_SKIP_CLEANUP is set)
    # NOTE: Cleanup is intentionally skipped for normal runs on versioned_artifacts
    # because it can be slow when many files exist. Use --cleanup-only to run the
    # cleaner explicitly.
    skip_cleanup = os.environ.get('ARES_SKIP_CLEANUP', '0') == '1'
    if args.cleanup_only:
        directories_to_clean = [
            'logs',
            'artifacts',
            'outcomes'
        ]
        cleanup_duplicate_files(directories_to_clean, keep_count=int(args.cleanup_duplicates_keep_count))

        if not getattr(args, 'cleanup_skip_regime_models', False):
            cleanup_regime_models_artifacts(keep_count=int(args.cleanup_regime_models_keep_count))

        if not args.cleanup_skip_versioned_artifacts:
            if getattr(args, 'cleanup_repair_versioned_artifacts', False):
                cleanup_versioned_artifact_stores_repair(keep_per_base=int(args.cleanup_keep_per_base))
            else:
                cleanup_versioned_artifact_stores(keep_per_base=int(args.cleanup_keep_per_base))
        return
    elif not skip_cleanup:
        directories_to_clean = [
            'logs',
            'artifacts',
            # 'versioned_artifacts',  # Skip - too many files, causes long startup
            'outcomes'
        ]
        cleanup_duplicate_files(directories_to_clean, keep_count=3)
        # cleanup_versioned_artifact_stores(keep_per_base=5)  # Skip - too slow with many artifacts
    else:
        logger.info("⏭️ Skipping cleanup (ARES_SKIP_CLEANUP=1)")
    
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
        # This prevents feature generation steps from interfering with other steps like regime_ensemble_training
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
        args.regime_ensemble_training,
        args.hmm_macro_regime, args.xgb_meso_regime, args.final_parameters_optimization,
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
        'execution_mode': args.execution_mode,
        # Ensure regime_timeframe is always populated; default to 15m for regime
        # discovery/alpha models unless explicitly overridden on the CLI.
        'regime_timeframe': args.regime_timeframe or '15m',
    }

    # Optional: force recomputation of multi-stage labeling HPO (ignore cached params)
    if getattr(args, 'force_hpo', False):
        config['force_hpo'] = True

    # Optional: start multi-stage labeling HPO at a specific stage
    if getattr(args, 'labeling_hpo_start_at', None) is not None:
        config['labeling_hpo_start_at'] = args.labeling_hpo_start_at

    # Hard-cap feature selection at ~200 features for full and blank modes
    if args.execution_mode in ("full", "blank"):
        config.setdefault('target_n_features_selector', 200)
        # Disable LGBM gating to allow more features through to final selection
        config.setdefault('enable_lgbm_feature_gating', False)

        # --- Layer 2 Performance Optimizations ---
        # 50% sampling for learnability probes (massive speedup)
        config.setdefault('layer2_probe_sampling_rate', 0.5)
        # Limit probes to top 30 features 
        config.setdefault('layer2_probe_feature_limit', 30)
        # Allow Linear model to skip LGBM if it's already conclusive
        config.setdefault('layer2_probe_linear_only_auc', 0.65)
        # Tighter profitability floor for L2 trials
        config.setdefault('layer2_min_pos_rate', 0.10)

    # Optional interaction-generation configuration
    if getattr(args, 'min_interaction_mi_lift', None) is not None:
        config['min_interaction_mi_lift'] = float(args.min_interaction_mi_lift)

    # Optional alpha/HPO configuration (used by hmm_ml_alpha_step and related steps)
    if getattr(args, 'alpha_enable_hpo', False):
        config['alpha_enable_hpo'] = True

    # Optional labeling HPO configuration (used by feature_generation_meta_labeling_step)
    if getattr(args, 'enable_labeling_hpo_params', False) or getattr(args, 'labeling_hpo_use_best_params', False):
        config['enable_labeling_hpo_params'] = True
    
    # Optional global HPO configuration (used by ml_risk_regime_step, ml_map_regime_step and unified training)
    if getattr(args, 'enable_hpo', False):
        config['enable_hpo'] = True
        config['risk_enable_hpo'] = True

    # Map XGB/HPO behaviour for specific steps when --enable-hpo is provided.
    if getattr(args, 'enable_hpo', False):
        target_steps: List[str] = []
        if args.step:
            target_steps = [args.step]
        elif args.steps:
            target_steps = [s.strip() for s in args.steps.split(',') if s.strip()]
        elif args.command:
            target_steps = [args.command]

        # When running the MAP regime step with --enable-hpo, enable XGB
        # training and feature pruning so we exercise the WCoV objective and
        # pruning logic.
        if 'ml_map_regime_step' in target_steps:
            config['map_xgb_enable_training'] = True
            config['map_xgb_enable_feature_pruning'] = True

        # When running the sr_labeling_xgb (meta-labeling HPO) step with
        # --enable-hpo, automatically enable XGB model HPO inside the step.
        # Keep compatibility with the legacy step name
        # 'meta_labeling_hpo_experiment'.
        if 'sr_labeling_xgb' in target_steps or 'meta_labeling_hpo_experiment' in target_steps:
            config['enable_xgb_model_hpo'] = True

    # Optional meta-gated backtest diagnostics configuration
    # These flags are no-ops for other steps; MetaGatedBacktestStep will
    # consume them when present.
    if getattr(args, 'meta_permutation_test', False):
        config['permutation_test'] = True
    if getattr(args, 'meta_permutation_repeats', None) is not None:
        config['permutation_repeats'] = args.meta_permutation_repeats
    if getattr(args, 'meta_forward_walk_n_windows', None) is not None:
        config['forward_walk_n_windows'] = args.meta_forward_walk_n_windows

    # Optional artifact export used by weighted_meta_labeling_step
    if getattr(args, 'save_labeled_data_csv', False):
        config['save_labeled_data_csv'] = True
    
    # Import tprint for troubleshooting output
    try:
        from src.utils.tprint import tprint
    except ImportError:
        # Fallback if tprint is not available
        def tprint(*args, **kwargs):
            print(*args)
    
    # Add centralized lookback days to config
    lookback_days = get_mode_lookback_days(args.execution_mode)
    config['lookback_days'] = lookback_days
    tprint(f"🔧 CONFIG: Centralized lookback_days={lookback_days} for mode={args.execution_mode}", "INFO")
    tprint("🔒 LEAKAGE GUARD: Use non-zero embargo (gap) and OOF predictions for stacking; avoid joining future-derived columns.", "INFO")
    
    # Set execution mode globally
    os.environ["EXECUTION_MODE"] = args.execution_mode

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
            
        elif any([args.train_analyst_base, args.train_analyst_ensemble, args.train_tactician_base, args.train_tactician_ensemble, args.train_gate]):
            # Model training execution using specific training steps
            run_backtest_after_training = False

            if args.train_gate:
                step_name = 'gate_training_step'
                training_type = 'gate'
                # Gate training has its own context
                config['execution_context'] = 'gate'
            elif args.train_analyst_base:
                step_name = 'analyst_base_training'
                training_type = 'analyst_base'
                config['execution_context'] = 'analyst'
                config['feature_set_size'] = 40
                # For analyst base, also run simple analyst_base_backtest step afterwards
                run_backtest_after_training = True
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
            training_result = await launcher.run_step(step_name, config)
            print(f"Model training '{training_type}' completed: {'✅ Success' if training_result.get('success') else '❌ Failed'}")

            # Optionally run analyst base backtest using OOS predictions
            if run_backtest_after_training and training_result.get('success'):
                logger.info("Running analyst base backtest using OOS predictions")
                backtest_result = await launcher.run_step('analyst_base_backtest', config)
                print(f"Analyst base backtest completed: {'✅ Success' if backtest_result.get('success') else '❌ Failed'}")

        elif args.rolling_hmm_regime_discovery:
            # Rolling HMM regime discovery execution
            logger.info("Running Rolling HMM regime discovery with EWMA features and HPO")
            # Ensure HPO is enabled by default unless explicitly disabled
            config['enable_auto_tuning'] = True
            result = await launcher.run_step('rolling_hmm_regime_discovery', config)
            print(f"Rolling HMM regime discovery completed: {'✅ Success' if result.get('success') else '❌ Failed'}")

        elif args.hmm_macro_regime:
            logger.info("Running HMM macro regime alpha step from Rolling HMM outputs")
            result = await launcher.run_step('hmm_macro_regime', config)
            print(f"HMM macro regime step completed: {'✅ Success' if result.get('success') else '❌ Failed'}")

        elif args.xgb_meso_regime:
            logger.info("Running XGB Meso Trend regime step")
            result = await launcher.run_step("xgb_meso_regime", config)
            print(f"XGB Meso Trend regime step completed: {'✅ Success' if result.get('success') else '❌ Failed'}")
        elif args.regime_ensemble_training:
            # Regime ensemble training execution
            logger.info("Training ensemble models for regime classification")
            result = await launcher.run_step('regime_ensemble_training', config)
            print(f"Regime ensemble training completed: {'✅ Success' if result.get('success') else '❌ Failed'}")

        elif args.final_parameters_optimization:
            # Final parameters optimization execution
            logger.info("Running final parameters optimization")
            result = await launcher.run_step('final_parameters_optimization', config)
            print(f"Final parameters optimization completed: {'✅ Success' if result.get('success') else '❌ Failed'}")

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
