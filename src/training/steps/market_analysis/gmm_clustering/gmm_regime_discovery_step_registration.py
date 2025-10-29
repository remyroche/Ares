"""
GMM Regime Discovery Step Registration

This module registers the GMM regime discovery step with the step registry.
"""

from src.training.steps.base_step import step_registry
from src.training.steps.market_analysis.gmm_clustering.gmm_regime_discovery_step import GMMRegimeDiscoveryStep

@step_registry.register('gmm_regime_discovery')
class GMMRegimeDiscoveryStepRegistration:
    """Registration for GMM regime discovery step."""
    
    def __init__(self):
        self.step_class = GMMRegimeDiscoveryStep
        self.step_name = 'gmm_regime_discovery'
        self.category = 'MARKET_ANALYSIS'
        self.description = 'GMM-based regime discovery with correlation-based feature reduction'
    
    def create_step(self, config: dict) -> GMMRegimeDiscoveryStep:
        """Create a GMM regime discovery step instance."""
        return GMMRegimeDiscoveryStep(
            n_components_range=config.get('n_components_range', (5, 9)),
            correlation_threshold=config.get('correlation_threshold', 0.85),
            random_state=config.get('random_state', 42)
        )
    
    def get_required_config(self) -> list:
        """Get required configuration parameters."""
        return ['symbol', 'exchange', 'timeframe']
    
    def get_optional_config(self) -> list:
        """Get optional configuration parameters."""
        return [
            'n_components_range',
            'correlation_threshold', 
            'random_state',
            'execution_mode'
        ]
