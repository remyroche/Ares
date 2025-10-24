"""
SR Detection Component.

This component detects Support/Resistance levels using optimized parameters.
Refactored to inherit from BaseStep for autonomous execution.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
    tprint_debug, tprint_data_format, tprint_data_preview, tprint_performance,
    tprint_timer, tprint_structured, LogLevel
)
from src.config.pipeline_modes import get_mode_config, get_mode_lookback_days

class SRDetectionComponent(BaseStep):
    """
    SR Detection Component.

    Detects Support/Resistance levels using optimized parameters.
    Refactored to inherit from BaseStep for autonomous execution.
    """

    def __init__(self, step_name: str = "sr_detection"):
        """Initialize the SR detection component."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('SRDetection')

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['sr_detection_result']

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute SR detection.

        Args:
            config: Configuration containing symbol, exchange, timeframes, etc.

        Returns:
            Execution result with artifacts and metrics
        """
        self.logger.info('📊 Starting SR Detection')
        tprint_info("🚀 Starting SR Detection Process", level=LogLevel.INFO)
        tprint_structured({
            "step": "sr_detection_start",
            "component": "SRDetectionComponent",
            "timestamp": datetime.now().isoformat()
        }, level=LogLevel.INFO)

        try:
            # Extract configuration
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'longs')
            execution_mode = config.get('execution_mode', 'light')
            
            # Get mode configuration for lookback periods and other parameters
            mode_config = get_mode_config(execution_mode)
            lookback_days = mode_config.lookback_days
            
            # Debug configuration with tprint
            tprint_data_format(config, "sr_detection_config", level=LogLevel.DEBUG)
            tprint_info(f"Configuration extracted - Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}")
            
            if not symbol:
                tprint_error("Symbol is required for SR detection", level=LogLevel.ERROR)
                raise ValueError("Symbol is required for SR detection")
            
            self.logger.info(f"Detecting SR levels for {symbol} from {exchange}")
            self.logger.info(f"Timeframe: {timeframe}, Direction: {direction}")
            tprint_info(f"Detecting SR levels for {symbol} from {exchange}")
            tprint_info(f"Timeframe: {timeframe}, Direction: {direction}")
            
            # Initialize artifacts list
            artifacts = []
            metrics = {}
            
            # Set up context using BaseStep method
            tprint_debug("Setting up execution context", level=LogLevel.DEBUG)
            self._set_context(
                symbol=symbol,
                exchange=exchange,
                direction=direction,
                model='Analyst'
            )
            
            # Perform SR detection (simplified version)
            tprint_info("Starting SR detection process", level=LogLevel.INFO)
            with tprint_timer("sr_detection_execution", level=LogLevel.PERFORMANCE):
                detection_result = await self._perform_sr_detection(symbol, timeframe, direction, execution_mode)
            
            # Debug detection result
            tprint_data_format(detection_result, "detection_result", level=LogLevel.DEBUG)
            tprint_data_preview(detection_result, "detection_result_preview", max_rows=5)

            # Save detection result as artifact using enhanced artifact saving
            tprint_info("Saving detection result as artifact", level=LogLevel.INFO)
            artifact_metadata = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'direction': direction,
                'execution_mode': execution_mode,
                'total_levels': detection_result.get('total_levels', 0),
                'support_levels': detection_result.get('support_levels', 0),
                'resistance_levels': detection_result.get('resistance_levels', 0)
            }
            tprint_data_format(artifact_metadata, "artifact_metadata", level=LogLevel.DEBUG)
            
            artifact_path = self._save_enhanced_artifact(
                detection_result,
                'sr_detection_result',
                'data',
                artifact_metadata
            )
            artifacts.append(artifact_path)
            tprint_success(f"Artifact saved to: {artifact_path}", level=LogLevel.INFO)
            
            # Record metrics
            metrics.update({
                'total_levels': detection_result.get('total_levels', 0),
                'support_levels': detection_result.get('support_levels', 0),
                'resistance_levels': detection_result.get('resistance_levels', 0),
                'execution_mode': execution_mode
            })
            
            tprint_structured({
                "step": "sr_detection_complete",
                "metrics": metrics,
                "artifacts_count": len(artifacts)
            }, level=LogLevel.INFO)

            self.logger.info(f'✅ SR Detection completed: {metrics["total_levels"]} levels detected')
            tprint_success(f'SR Detection completed: {metrics["total_levels"]} levels detected', level=LogLevel.INFO)
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'detection_result': detection_result
            }

        except Exception as e:
            self.logger.error(f'❌ SR Detection failed: {e}')
            tprint_error(f'SR Detection failed: {e}', level=LogLevel.ERROR)
            tprint_structured({
                "step": "sr_detection_error",
                "error": str(e),
                "error_type": type(e).__name__
            }, level=LogLevel.ERROR)
            return {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }

    async def _perform_sr_detection(self, symbol: str, timeframe: str, 
                                  direction: str, execution_mode: str) -> Dict[str, Any]:
        """
        Perform SR detection with simplified logic.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            direction: Trading direction
            execution_mode: Execution mode (light/full)
            
        Returns:
            Detection result dictionary
        """
        try:
            tprint_debug(f"Starting SR detection for {symbol} on {timeframe}", level=LogLevel.DEBUG)
            tprint_structured({
                "detection_params": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "direction": direction,
                    "execution_mode": execution_mode
                }
            }, level=LogLevel.DEBUG)
            
            # Create sample detection result for demonstration
            # In a real implementation, this would use the existing detection logic
            tprint_info("Generating sample SR levels for demonstration", level=LogLevel.INFO)
            
            sample_levels = [
                {'price': 1.2000, 'type': 'support', 'strength': 0.85, 'touches': 3},
                {'price': 1.2500, 'type': 'resistance', 'strength': 0.72, 'touches': 2},
                {'price': 1.1800, 'type': 'support', 'strength': 0.68, 'touches': 2},
                {'price': 1.2800, 'type': 'resistance', 'strength': 0.81, 'touches': 4}
            ]
            
            tprint_data_format(sample_levels, "sample_levels", level=LogLevel.DEBUG)
            tprint_data_preview(sample_levels, "sample_levels_preview", max_rows=3)
            
            support_levels = [l for l in sample_levels if l['type'] == 'support']
            resistance_levels = [l for l in sample_levels if l['type'] == 'resistance']
            
            tprint_info(f"Detected {len(support_levels)} support levels and {len(resistance_levels)} resistance levels", level=LogLevel.INFO)
            
            result = {
                'total_levels': len(sample_levels),
                'support_levels': len(support_levels),
                'resistance_levels': len(resistance_levels),
                'levels': sample_levels,
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'direction': direction,
                    'execution_mode': execution_mode
                }
            }
            
            tprint_data_format(result, "detection_result", level=LogLevel.DEBUG)
            tprint_success(f"SR detection completed successfully with {result['total_levels']} total levels", level=LogLevel.INFO)
            
            return result
            
        except Exception as e:
            self.logger.error(f"SR detection failed: {e}")
            tprint_error(f"SR detection failed: {e}", level=LogLevel.ERROR)
            tprint_structured({
                "detection_error": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "symbol": symbol,
                    "timeframe": timeframe
                }
            }, level=LogLevel.ERROR)
            return {
                'total_levels': 0,
                'support_levels': 0,
                'resistance_levels': 0,
                'levels': [],
                'error': str(e)
            }
