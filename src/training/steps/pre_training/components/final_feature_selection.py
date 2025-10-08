"""
Final Feature Selection Component.

This component performs multi-stage feature selection (120→100→80→60) as the final step
in the market analysis pipeline.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult, ComponentError
from .component_factory import register_component
from src.utils.logger import system_logger
from src.utils.tprint import tprint
from ...market_analysis.logging_standards import (
    get_logger, log_info, log_warning, log_error, log_success, log_debug,
    LoggingContext, log_step_progress, log_data_info, log_validation_result
)

# Import optimized process engine
from ...market_analysis.optimized_process_engines import OptimizedFeatureSelectionEngine, ProcessType

# Import hardware optimization tools
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.adaptive_optimization_engine import AdaptiveOptimizationEngine
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
from src.training.config.data_locator import DataLocator


CONFIG_ROOT_ENV = "ARES_CONFIG_ROOT"
"""Environment variable that can override the repo-relative config root."""

DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[4] / "config"
"""Default location of repository configuration files relative to this module."""

FEATURE_SELECTION_CONFIG_PATH = Path(
    os.environ.get(CONFIG_ROOT_ENV, DEFAULT_CONFIG_ROOT)
) / "feature_selection_config.yaml"
"""Resolved path to the feature selection YAML profile."""


@register_component('final_feature_selection')
class FinalFeatureSelectionComponent(BasePreTrainingComponent):
    """
    Final Feature Selection Component.

    Performs multi-stage feature selection as the final step in the pipeline.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the final feature selection component."""
        super().__init__(config)
        # Use standardized logging
        self.logger = get_logger('FinalFeatureSelectionComponent')

        # Initialize hardware optimization tools
        tprint("🔧 [FinalFeatureSelection] Initializing hardware optimization tools...")
        self.memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=8.0)
        self.adaptive_engine = AdaptiveOptimizationEngine()
        self.hardware_manager = UnifiedHardwareManager()

        # Initialize optimized process engine with hardware acceleration
        tprint("🔧 [FinalFeatureSelection] Initializing optimized feature selection engine...")
        self.optimized_engine = OptimizedFeatureSelectionEngine(
            use_hardware_accel=True,
            cache_size=1000
        )
        tprint("✅ [FinalFeatureSelection] Hardware optimization tools and feature selection engine initialized")

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        tprint("📦 [FinalFeatureSelection] Reporting required artifacts: final_feature_selection_result")
        return ['final_feature_selection_result']

    def _load_model_specific_config(self, model_type: str) -> Dict[str, Any]:
        """Load model-specific configuration from YAML file."""
        try:
            import yaml

            # Try to load from the feature selection config file
            config_path = FEATURE_SELECTION_CONFIG_PATH
            log_debug(
                f"Resolving feature selection config for '{model_type}' via {config_path}"
            )
            tprint(
                f"🧩 [FinalFeatureSelection] Loading model-specific config for '{model_type}' from {config_path}"
            )
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)

                if config_data and 'feature_selection' in config_data:
                    fs_config = config_data['feature_selection']

                    # Check if model has a specific profile
                    if 'model_profiles' in fs_config and model_type in fs_config['model_profiles']:
                        model_config = fs_config['model_profiles'][model_type]

                        log_success(
                            f"Loaded feature selection profile '{model_type}' from {config_path}"
                        )
                        tprint(
                            f"✅ [FinalFeatureSelection] Loaded profile '{model_type}' from YAML configuration"
                        )

                        # Map YAML config to expected format
                        stage_targets = [
                            model_config.get('target_features', 80) - 20,  # stage_1_target
                            model_config.get('target_features', 80) - 15,  # stage_2_target
                            model_config.get('target_features', 80) - 10   # stage_3_target
                        ]

                        return {
                            'target_features': model_config.get('target_features', 80),
                            'min_features': model_config.get('min_features', 60),
                            'max_features': model_config.get('max_features', 100),
                            'stage_targets': stage_targets,
                            'priority_categories': model_config.get('priority_categories', ['momentum', 'volatility', 'microstructure'])
                        }

                    # Use default settings if no model profile found
                    elif model_type == 'default':
                        return {
                            'target_features': fs_config.get('target_features', 80),
                            'min_features': fs_config.get('min_features', 60),
                            'max_features': fs_config.get('max_features', 100),
                            'stage_targets': [95, 75, 65],
                            'priority_categories': ['momentum', 'volatility', 'microstructure']
                        }

            # Fallback to hardcoded defaults if YAML loading fails
            log_warning(
                f"Could not load model-specific config for {model_type}, using defaults. "
                f"Searched path: {config_path}"
            )
            tprint(
                f"⚠️ [FinalFeatureSelection] Using default configuration for '{model_type}'"
            )
            return {
                'target_features': 80,
                'min_features': 60,
                'max_features': 100,
                'stage_targets': [95, 75, 65],
                'priority_categories': ['momentum', 'volatility', 'microstructure']
            }

        except Exception as e:
            log_error(
                f"Error loading model-specific config for {model_type}: {e}. "
                f"Searched path: {FEATURE_SELECTION_CONFIG_PATH}"
            )
            tprint(
                f"❌ [FinalFeatureSelection] Error loading config for '{model_type}': {e}. Using defaults."
            )
            return {
                'target_features': 80,
                'min_features': 60,
                'max_features': 100,
                'stage_targets': [95, 75, 65],
                'priority_categories': ['momentum', 'volatility', 'microstructure']
            }

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute final feature selection.

        Args:
            data: Market data for feature selection
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with feature selection results
        """
        log_info('🎯 Starting Final Feature Selection')
        tprint('🚀 [FinalFeatureSelection] Starting execute routine')

        try:
            # Check memory pressure and apply optimizations
            memory_pressure = getattr(self.memory_optimizer, 'memory_pressure', 0.0)
            if memory_pressure > 0.75:
                log_warning(f'🧠 High memory pressure detected ({memory_pressure:.2f}), applying memory optimizations')
                self.memory_optimizer._apply_memory_optimizations()
                tprint(
                    f"🧠 [FinalFeatureSelection] High memory pressure detected ({memory_pressure:.2f}); optimizations applied"
                )

            # Get optimal hardware configuration for feature selection
            hardware_config = self.hardware_manager.get_optimal_config('feature_selection')
            log_debug(f'📊 Hardware configuration: {hardware_config}')
            tprint(f'🛠️ [FinalFeatureSelection] Hardware configuration resolved: {hardware_config}')

            # Adapt optimization strategy based on current conditions
            adaptive_strategy = self.adaptive_engine.get_optimal_strategy('feature_selection', {
                'memory_pressure': memory_pressure,
                'hardware_config': hardware_config
            })
            log_debug(f'🎯 Adaptive strategy: {adaptive_strategy}')
            tprint(f'🎯 [FinalFeatureSelection] Adaptive strategy selected: {adaptive_strategy}')

            # Import the final feature selection step
            from ..final_feature_selection_step import run_final_feature_selection_step

            # Resolve symbol from config or pipeline state
            symbol = getattr(self.config, 'symbol', None)
            if symbol is None and 'symbol' in pipeline_state:
                symbol = pipeline_state['symbol']
            if symbol is None:
                raise ValueError("Symbol must be provided in config or pipeline state")

            # Resolve exchange from config or pipeline state
            exchange = getattr(self.config, 'exchange', None)
            if exchange is None and 'exchange' in pipeline_state:
                exchange = pipeline_state['exchange']
            if exchange is None:
                exchange = 'binance'  # Default exchange

            # Resolve timeframe from config or pipeline state
            timeframe = getattr(self.config, 'timeframe', None)
            if timeframe is None and 'timeframe' in pipeline_state:
                timeframe = pipeline_state['timeframe']
            if timeframe is None:
                timeframe = '15m'  # Default timeframe

            # Resolve data locator and directory information
            data_locator: Optional[DataLocator] = getattr(self.config, 'data_locator', None)
            if data_locator is None:
                data_locator = pipeline_state.get('data_locator')

            data_dir = getattr(self.config, 'data_dir', None)
            if data_dir is None:
                data_dir = pipeline_state.get('data_dir')

            data_dir_key = getattr(self.config, 'data_dir_key', None) or pipeline_state.get('data_dir_key', 'market_data')
            final_features_dir_key = (
                getattr(self.config, 'final_feature_selection_dir_key', None)
                or pipeline_state.get('final_feature_selection_dir_key', 'final_feature_selection')
            )

            if data_dir is None and isinstance(data_locator, DataLocator):
                data_dir = str(data_locator.data_path(data_dir_key))

            if data_dir is None:
                raise ValueError("Data directory could not be resolved for final feature selection")

            final_features_dir_override = (
                getattr(self.config, 'final_feature_selection_dir', None)
                or pipeline_state.get('final_feature_selection_dir')
            )

            output_directory_override = getattr(self.config, 'output_directory', None)
            if output_directory_override is None:
                output_directory_override = pipeline_state.get('generated_dir')

            tprint(
                "📥 [FinalFeatureSelection] Resolved execution context "
                f"symbol={symbol}, exchange={exchange}, timeframe={timeframe}, data_dir={data_dir}"
            )

            # Resolve the model profile for feature selection from config or pipeline state
            model_type = None
            if self.config.custom_params:
                model_type = self.config.custom_params.get('model_type')
            if model_type is None:
                model_type = pipeline_state.get('model_type') if pipeline_state else None
            if not model_type:
                model_type = 'default'

            # Load model-specific configuration with hardware optimizations
            final_feature_selection_config = self._load_model_specific_config(model_type)
            tprint(
                "🧾 [FinalFeatureSelection] Final feature selection config prepared: "
                f"{final_feature_selection_config}"
            )

            if model_type != 'default':
                log_success(
                    f"Feature selection will use model-specific profile '{model_type}'"
                )
            else:
                log_info("Feature selection will use the default profile")

            # Apply adaptive optimizations to config
            if adaptive_strategy:
                final_feature_selection_config.update({
                    'hardware_accelerated': adaptive_strategy.get('hardware_accelerated', True),
                    'memory_efficient': adaptive_strategy.get('memory_efficient', True),
                    'parallel_processing': adaptive_strategy.get('parallel_processing', False)
                })

            # Execute final feature selection with hardware optimization
            log_info(f'🚀 Executing feature selection with hardware optimizations...')
            tprint('🚀 [FinalFeatureSelection] Executing feature selection step')
            runtime_config = dict(final_feature_selection_config)
            runtime_config['data_dir_key'] = data_dir_key
            runtime_config['final_features_dir_key'] = final_features_dir_key
            if final_features_dir_override:
                runtime_config['final_features_dir'] = final_features_dir_override
            if output_directory_override and 'output_directory' not in runtime_config:
                runtime_config['output_directory'] = output_directory_override
            if isinstance(data_locator, DataLocator):
                runtime_config['data_locator'] = data_locator
                runtime_config.setdefault(
                    'output_directory_key',
                    pipeline_state.get('generated_dir_key', 'market_analysis'),
                )

            success = await run_final_feature_selection_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                config=runtime_config
            )

            if success:
                # Create result artifacts with hardware performance metrics
                performance_metrics = {
                    'memory_pressure_before': memory_pressure,
                    'memory_pressure_after': getattr(self.memory_optimizer, 'memory_pressure', 0.0),
                    'hardware_config_used': hardware_config,
                    'adaptive_strategy_used': adaptive_strategy
                }

                artifacts = {
                    'final_feature_selection_result': {
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'data_dir': data_dir,
                        'feature_selection_config': final_feature_selection_config,
                        'execution_mode': 'component',
                        'success': True,
                        'stage_reduction': {
                            'initial': 120,
                            'stage_1': 100,
                            'stage_2': 80,
                            'stage_3': 60
                        },
                        'hardware_performance': performance_metrics
                    }
                }

                log_success(f'✅ Final feature selection completed successfully with hardware optimizations')
                log_info(f'📊 Performance metrics: {performance_metrics}')
                tprint(
                    "✅ [FinalFeatureSelection] Feature selection succeeded with metrics "
                    f"{performance_metrics}"
                )

                # Clean up memory after processing
                self.memory_optimizer._light_memory_cleanup()
                tprint('🧹 [FinalFeatureSelection] Performed post-execution memory cleanup')

                # Save artifacts persistently using the artifact manager
                persistence_error: Optional[str] = None
                artifacts_saved_persistently = False
                saved_files: Dict[str, str] = {}
                failure_reason: Optional[str] = None

                try:
                    saved_files = await self.save_artifacts(artifacts, {
                        'component_type': 'final_feature_selection',
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe
                    })
                    
                    if saved_files:
                        artifacts_saved_persistently = True
                        log_success(
                            f"💾 [FINAL_FEATURE_SELECTION] Artifacts saved persistently: {list(saved_files.keys())}"
                        )
                        tprint(
                            "💾 [FinalFeatureSelection] Artifacts saved successfully: "
                            f"{list(saved_files.keys())}"
                        )
                    else:
                        log_error("❌ [FINAL_FEATURE_SELECTION] Artifact manager returned no file paths")
                        tprint("❌ [FinalFeatureSelection] Failed to persist artifacts: no file paths returned")
                        
                except Exception as e:
                    persistence_error = str(e)
                    failure_reason = f"Artifact saving failed: {e}"
                    log_warning(f"⚠️ [FINAL_FEATURE_SELECTION] Exception while saving artifacts persistently: {e}")
                    tprint(f"⚠️ [FinalFeatureSelection] Artifact save error: {e}")

                component_success = success and artifacts_saved_persistently

                result_error: Optional[Exception] = None
                warnings: List[str] = []
                if not component_success:
                    failure_reason = failure_reason or persistence_error or "Artifacts were not persisted"
                    result_error = ComponentError(failure_reason)
                    warnings.append(failure_reason)
                    log_error(f"❌ [FINAL_FEATURE_SELECTION] {failure_reason}")

                return ComponentResult(
                    success=component_success,
                    artifacts=artifacts,
                    error=result_error,
                    warnings=warnings,
                    execution_time=0.0,
                    metrics={},
                    metadata={
                        'component_type': 'final_feature_selection',
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'artifacts_saved_persistently': artifacts_saved_persistently,
                        'artifact_persistence_paths': saved_files,
                        **({'artifact_persistence_error': persistence_error} if persistence_error else {})
                    }
                )
            else:
                log_error('Final feature selection failed')
                tprint('❌ [FinalFeatureSelection] Feature selection execution returned failure')

                # Clean up memory even on failure
                self.memory_optimizer._light_memory_cleanup()
                tprint('🧹 [FinalFeatureSelection] Memory cleanup performed after failure')

                failure_message = "Final feature selection execution failed"
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error=ComponentError(failure_message),
                    warnings=[failure_message],
                    execution_time=0.0,
                    metrics={},
                    metadata={
                        'component_type': 'final_feature_selection',
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'memory_pressure': getattr(self.memory_optimizer, 'memory_pressure', 0.0)
                    }
                )

        except Exception as e:
            log_error(f'Final feature selection failed with exception: {e}')
            tprint(f'❌ [FinalFeatureSelection] Exception during execution: {e}')

            # Clean up memory on exception
            try:
                self.memory_optimizer._light_memory_cleanup()
                tprint('🧹 [FinalFeatureSelection] Memory cleanup performed after exception')
            except Exception as cleanup_error:
                tprint(f'⚠️ [FinalFeatureSelection] Memory cleanup failed (non-critical): {cleanup_error}')

            failure_message = str(e)
            return ComponentResult(
                success=False,
                artifacts={},
                error=ComponentError(failure_message),
                warnings=[failure_message],
                execution_time=0.0,
                metrics={},
                metadata={
                    'component_type': 'final_feature_selection',
                    'symbol': symbol if 'symbol' in locals() else 'unknown',
                    'exchange': exchange if 'exchange' in locals() else 'unknown',
                    'timeframe': timeframe if 'timeframe' in locals() else 'unknown',
                    'memory_pressure': getattr(self.memory_optimizer, 'memory_pressure', 0.0)
                }
            )

    def cleanup(self):
        """Clean up hardware optimization resources."""
        try:
            log_info('🧹 Cleaning up hardware optimization resources...')
            tprint('🧹 [FinalFeatureSelection] Cleanup initiated')
            self.memory_optimizer._light_memory_cleanup()
            log_info('✅ Hardware optimization resources cleaned up')
            tprint('✅ [FinalFeatureSelection] Cleanup completed')
        except Exception as e:
            log_warning(f'⚠️ Error during hardware cleanup: {e}')
            tprint(f'⚠️ [FinalFeatureSelection] Cleanup encountered an error: {e}')
