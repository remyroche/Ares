"""
Analyst Base Training Step.

This step trains base analyst models.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint

logger = logging.getLogger(__name__)


class AnalystBaseTrainingStep(BaseStep):
    """
    Analyst Base Training Step.

    Trains base analyst models for price prediction.
    """

    def __init__(self, step_name: str = "analyst_base_training"):
        """Initialize the analyst base training step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('AnalystBaseTraining')

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute analyst base model training.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - direction: Trading direction ('long', 'short', 'both')

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        tprint(f"🧠 Starting analyst base training for {config.get('symbol', 'UNKNOWN')}", "INFO")

        try:
            # Import and call unified training step
            from .unified_models_training_step import UnifiedModelsTrainingStep
            
            # Set training type for unified step
            config['training_type'] = 'analyst_base'
            config['execution_context'] = 'analyst'
            
            # Create and execute unified training step
            unified_step = UnifiedModelsTrainingStep()
            result = await unified_step.execute(config)
            
            return result

        except Exception as e:
            error_msg = f"Analyst base training failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.error(error_msg)

            return {
                'success': False,
                'artifacts': {},
                'metrics': {},
                'error': error_msg
            }

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)


# Register the step
def register_analyst_base_training_step():
    """Register the analyst base training step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("analyst_base_training", AnalystBaseTrainingStep)
    tprint("✅ Analyst base training step registered", "SUCCESS")


# Auto-register when module is imported
register_analyst_base_training_step()
