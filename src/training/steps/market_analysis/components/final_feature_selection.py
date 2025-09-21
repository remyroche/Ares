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


class FinalFeatureSelectionComponent(BaseMarketAnalysisComponent):
    """
    Final Feature Selection Component.

    Performs multi-stage feature selection as the final step in the pipeline.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the final feature selection component."""
        super().__init__(config)
        self.logger = system_logger.getChild('FinalFeatureSelectionComponent')

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['final_feature_selection_result']

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute final feature selection.

        Args:
            data: Market data for feature selection
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with feature selection results
        """
        self.logger.info('🎯 Starting Final Feature Selection')

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

            # Prepare configuration for final feature selection
            final_feature_selection_config = {
                'initial_features': self.config.custom_params.get('initial_features', 120) if self.config.custom_params else 120,
                'stage_1_target': self.config.custom_params.get('stage_1_target', 100) if self.config.custom_params else 100,
                'stage_2_target': self.config.custom_params.get('stage_2_target', 80) if self.config.custom_params else 80,
                'stage_3_target': self.config.custom_params.get('stage_3_target', 60) if self.config.custom_params else 60,
                'rf_n_estimators': self.config.custom_params.get('rf_n_estimators', 100) if self.config.custom_params else 100,
                'cv_folds': self.config.custom_params.get('cv_folds', 5) if self.config.custom_params else 5,
                'save_analysis': self.config.custom_params.get('save_analysis', True) if self.config.custom_params else True,
                'verbose': self.config.custom_params.get('verbose', True) if self.config.custom_params else True
            }

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

                self.logger.info('✅ Final feature selection completed successfully')
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
                self.logger.error('❌ Final feature selection failed')
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
            self.logger.exception(f'❌ Final feature selection failed with exception: {e}')
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
