"""
HMM Regime Integration Step

This step runs after HMM training to ensure all subsequent ML models
use the HMM-retagged regime data instead of the original MARKET_ANALYSIS tags.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_important_calls, log_all_calls
from .regime_data_integration import RegimeDataIntegrator

logger = logging.getLogger(__name__)

class HMMRegimeIntegrationStep:
    """Step to integrate HMM-retagged regime data into the pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize HMM regime integration step.
        
        Args:
            config: Step configuration
        """
        self.config = config or {}
        self.logger = system_logger.getChild('HMMRegimeIntegrationStep')
        self.regime_integrator = RegimeDataIntegrator(config)
        
    @log_important_calls
    async def execute(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute HMM regime integration step.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with integrated HMM regime data
        """
        try:
            symbol = training_input.get('symbol', 'UNKNOWN')
            exchange = training_input.get('exchange', 'UNKNOWN')
            timeframe = training_input.get('timeframe', 'UNKNOWN')
            
            self.logger.info(f"🔄 Starting HMM regime integration for {symbol}/{exchange}/{timeframe}")
            
            # Check if HMM training was completed
            if not pipeline_state.get('hmm_training_completed', False):
                self.logger.warning("⚠️ HMM training not completed, skipping regime integration")
                return pipeline_state
            
            # Check if HMM retagging was completed
            if not pipeline_state.get('hmm_retagging_completed', False):
                self.logger.warning("⚠️ HMM retagging not completed, skipping regime integration")
                return pipeline_state
            
            # Integrate HMM regime data
            updated_pipeline_state = await self.regime_integrator.integrate_hmm_regime_data(
                pipeline_state, symbol, exchange, timeframe
            )
            
            # Mark step as completed
            updated_pipeline_state.update({
                'hmm_regime_integration_completed': True,
                'hmm_regime_integration_timestamp': datetime.now().isoformat(),
                'regime_data_source': 'hmm_retagged',
                'subsequent_models_use_hmm_regimes': True
            })
            
            self.logger.info("✅ HMM regime integration completed successfully")
            
            return updated_pipeline_state
            
        except Exception as e:
            self.logger.error(f"❌ HMM regime integration failed: {e}")
            pipeline_state.update({
                'hmm_regime_integration_completed': False,
                'hmm_regime_integration_error': str(e)
            })
            return pipeline_state
    
    @log_all_calls
    def get_step_info(self) -> Dict[str, Any]:
        """Get information about this step."""
        return {
            'step_name': 'hmm_regime_integration',
            'description': 'Integrates HMM-retagged regime data into pipeline for subsequent steps',
            'dependencies': ['hmm_training'],
            'outputs': ['updated_pipeline_state', 'regime_data_integration_completed'],
            'config': self.config
        }

# Global instance for convenience
hmm_regime_integration_step = HMMRegimeIntegrationStep()