"""
Tactician Ensemble Training Step.

This step trains ensemble tactician models.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint

logger = logging.getLogger(__name__)


class TacticianEnsembleTrainingStep(BaseStep):
    """
    Tactician Ensemble Training Step.

    Trains ensemble tactician models using outputs from base models.
    """

    def __init__(self, step_name: str = "tactician_ensemble_training"):
        """Initialize the tactician ensemble training step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('TacticianEnsembleTraining')

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tactician ensemble model training.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - direction: Trading direction ('longs', 'shorts', 'both')

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        tprint(f"🧠 Starting tactician ensemble training for {config.get('symbol', 'UNKNOWN')}", "INFO")

        try:
            # Import and call unified training step
            from .unified_models_training_step import UnifiedModelsTrainingStep
            
            # Set training type for unified step
            config['training_type'] = 'tactician_ensemble'
            config['execution_context'] = 'tactician'
            
            # Create and execute unified training step
            unified_step = UnifiedModelsTrainingStep()
            result = await unified_step.execute(config)
            
            return result

        except Exception as e:
            error_msg = f"Tactician ensemble training failed: {str(e)}"
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
def register_tactician_ensemble_training_step():
    """Register the tactician ensemble training step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("tactician_ensemble_training", TacticianEnsembleTrainingStep)
    tprint("✅ Tactician ensemble training step registered", "SUCCESS")


# Auto-register when module is imported
register_tactician_ensemble_training_step()
