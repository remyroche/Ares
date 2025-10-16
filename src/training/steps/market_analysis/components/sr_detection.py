"""
SR Detection Component.

This component detects Support/Resistance levels using optimized parameters.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger

class SRDetectionComponent(BaseMarketAnalysisComponent):
    """
    SR Detection Component.

    Detects Support/Resistance levels using optimized parameters.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the SR detection component."""
        super().__init__(config)
        self.logger = system_logger.getChild('SRDetection')

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['sr_detection_result']

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute SR detection.

        Args:
            data: Market data for detection
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with detection results
        """
        self.logger.info('📊 Starting SR Detection')

        try:
            # Import SR detection step
            from ..sr_detection import SRDetectionStep

            # Get market data
            market_data = await self._load_market_data(data)
            if market_data is None or len(market_data) == 0:
                raise ValueError("No market data available for SR detection")

            # Get optimized parameters from previous stage
            optimized_parameters = pipeline_state.get('optimized_parameters', {})
            quality_thresholds = pipeline_state.get('quality_thresholds', {})

            # Create training input
            training_input = {
                'symbol': self.config.symbol,
                'exchange': self.config.exchange,
                'timeframe': self.config.timeframe,
                'optimized_parameters': optimized_parameters,
                'quality_thresholds': quality_thresholds
            }

            # Create pipeline state for SR detection
            sr_pipeline_state = {
                'dataframe': market_data,
                'optimized_parameters': optimized_parameters,
                'quality_thresholds': quality_thresholds
            }

            # Initialize and execute SR detection step
            sr_detection_step = SRDetectionStep(training_input)
            result = await sr_detection_step.execute(training_input, sr_pipeline_state)

            if result.get('success', False):
                sr_levels = result.get('sr_levels', {})
                sr_metrics = result.get('sr_metrics', {})

                # Validate that we have SR levels
                all_levels = sr_levels.get('all_levels', [])
                if not all_levels:
                    raise ValueError("SR detection completed but no levels were detected")

                # Create single consolidated artifact
                artifacts = {
                    'sr_detection_result': {
                        'sr_levels': all_levels,
                        'sr_metrics': sr_metrics,
                        'detection_summary': {
                            'total_levels': len(all_levels),
                            'support_levels': len([l for l in all_levels if l.get('type') == 'support']),
                            'resistance_levels': len([l for l in all_levels if l.get('type') == 'resistance']),
                            'detection_time': result.get('execution_time', 0.0)
                        },
                        'metadata': {
                            'symbol': self.config.symbol,
                            'exchange': self.config.exchange,
                            'timeframe': self.config.timeframe,
                            'data_points': len(market_data) if market_data is not None else 0,
                            'execution_timestamp': datetime.now().isoformat()
                        }
                    }
                }

                self.logger.info(f'✅ SR Detection completed: {len(all_levels)} levels detected')
                return ComponentResult(
                    success=True,
                    artifacts=artifacts,
                    metadata={
                        'symbol': self.config.symbol,
                        'exchange': self.config.exchange,
                        'timeframe': self.config.timeframe,
                        'data_points': len(market_data)
                    }
                )
            else:
                error_msg = result.get('error', 'Unknown error in SR detection')
                raise ValueError(f"SR detection failed: {error_msg}")

        except Exception as e:
            self.logger.error(f'❌ SR Detection failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e)
            )

    async def _load_market_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Load and prepare market data for detection."""
        if data is None:
            return None

        if isinstance(data, pd.DataFrame):
            return data.copy()

        # Handle other data types if needed
        return None
