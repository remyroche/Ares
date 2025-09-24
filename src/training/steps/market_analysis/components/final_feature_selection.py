"""
Final Feature Selection Component.

This component performs multi-stage feature selection (120→100→80→60) as the final step
in the market analysis pipeline.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger
from ..logging_standards import (
    get_logger, log_info, log_warning, log_error, log_success, log_debug,
    LoggingContext, log_step_progress, log_data_info, log_validation_result
)


class FinalFeatureSelectionComponent(BaseMarketAnalysisComponent):
    """
    Final Feature Selection Component.

    Performs multi-stage feature selection as the final step in the pipeline.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the final feature selection component."""
        super().__init__(config)
        # Use standardized logging
        self.logger = get_logger('FinalFeatureSelectionComponent')

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['final_feature_selection_result']

    def _load_model_specific_config(self, model_type: str) -> Dict[str, Any]:
        """Load model-specific configuration from YAML file."""
        try:
            import yaml

            # Try to load from the feature selection config file
            config_path = Path("/workspace/src/config/feature_selection_config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)

                if config_data and 'feature_selection' in config_data:
                    fs_config = config_data['feature_selection']

                    # Check if model has a specific profile
                    if 'model_profiles' in fs_config and model_type in fs_config['model_profiles']:
                        model_config = fs_config['model_profiles'][model_type]

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
            log_warning(f"Could not load model-specific config for {model_type}, using defaults")
            return {
                'target_features': 80,
                'min_features': 60,
                'max_features': 100,
                'stage_targets': [95, 75, 65],
                'priority_categories': ['momentum', 'volatility', 'microstructure']
            }

        except Exception as e:
            log_error(f"Error loading model-specific config for {model_type}: {e}")
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

        try:
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
                timeframe = '1m'  # Default timeframe

            # Resolve data directory from config or pipeline state
            data_dir = getattr(self.config, 'data_dir', None)
            if data_dir is None and 'data_dir' in pipeline_state:
                data_dir = pipeline_state['data_dir']
            if data_dir is None:
                data_dir = 'historical_data'  # Default data directory

            # Load model-specific configuration
            final_feature_selection_config = self._load_model_specific_config('default')

            # Execute final feature selection
            success = await run_final_feature_selection_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                config=final_feature_selection_config
            )

            if success:
                # Create result artifacts
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
                        }
                    }
                }

                log_success('Final feature selection completed successfully')
                return ComponentResult(
                    success=True,
                    artifacts=artifacts,
                    error_message=None,
                    execution_time=0.0,
                    metadata={
                        'component_type': 'final_feature_selection',
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe
                    }
                )
            else:
                log_error('Final feature selection failed')
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message="Final feature selection execution failed",
                    execution_time=0.0,
                    metadata={
                        'component_type': 'final_feature_selection',
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe
                    }
                )

        except Exception as e:
            log_error(f'Final feature selection failed with exception: {e}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                execution_time=0.0,
                metadata={
                    'component_type': 'final_feature_selection',
                    'symbol': symbol if 'symbol' in locals() else 'unknown',
                    'exchange': exchange if 'exchange' in locals() else 'unknown',
                    'timeframe': timeframe if 'timeframe' in locals() else 'unknown'
                }
            )
