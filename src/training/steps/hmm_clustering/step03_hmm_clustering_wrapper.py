#!/usr/bin/env python3
"""Wrapper for HMM Clustering Step to ensure pipeline compatibility.

This wrapper ensures that the enhanced HMM regime discovery step integrates
seamlessly with the existing pipeline structure.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, validates
from .step03_enhanced_hmm_regime_discovery import EnhancedHMMRegimeDiscoveryStep

logger = system_logger.getChild("HMMClusteringWrapper")


class HMMRegimeDiscoveryStep:
    """Wrapper class for enhanced HMM regime discovery to maintain compatibility."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the HMM regime discovery step.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('HMMRegimeDiscoveryStep')
        
        # Initialize the enhanced step
        self.enhanced_step = EnhancedHMMRegimeDiscoveryStep(config)
        
    @handles_errors(fallback=False)
    async def initialize(self) -> None:
        """Initialize the step."""
        self.logger.info('🚀 Initializing HMM Regime Discovery Step (Enhanced)...')
        await self.enhanced_step.initialize()
        self.logger.info('✅ HMM Regime Discovery Step initialized successfully')
        
    @validates(step_name='hmm_regime_discovery', validation_level='CRITICAL')
    @handles_errors(default_return={'success': False}, context='hmm_regime_discovery.execute')
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the HMM regime discovery step.
        
        Args:
            training_input: Training input configuration
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with regime discovery results
        """
        self.logger.info('🎯 Executing HMM Regime Discovery (Enhanced Version)...')
        
        try:
            # Execute the enhanced step
            result = await self.enhanced_step.execute(training_input, pipeline_state)
            
            # Ensure backward compatibility
            if result.get('enhanced_hmm_regime_discovery_completed', False):
                # Map enhanced results to standard format
                result['hmm_regime_discovery_completed'] = True
                result['success'] = True
                
                # Ensure required outputs exist
                if 'regime_labels' not in result and 'regime_states' in result:
                    result['regime_labels'] = result['regime_states']
                    
                # Create composite DataFrame if needed
                if 'composite_df' not in result and 'regime_states' in result:
                    composite_df = pd.DataFrame({
                        'composite_cluster_id': result['regime_states'],
                        'timestamp': pd.date_range(
                            start='2024-01-01', 
                            periods=len(result['regime_states']), 
                            freq='1min'
                        )
                    })
                    result['composite_df'] = composite_df
                    
                # Add metrics for compatibility
                if 'metrics' not in result:
                    result['metrics'] = {
                        'hmm_states': result.get('n_regimes', 0),
                        'composite_clusters': result.get('n_regimes', 0),
                        'total_periods': len(result.get('regime_states', [])),
                        'ensemble_quality': result.get('ensemble_quality', {}),
                        'economic_significance': result.get('economic_significance', False)
                    }
                    
                self.logger.info('✅ HMM Regime Discovery completed successfully')
                self.logger.info(f'   - Regimes discovered: {result.get("n_regimes", 0)}')
                self.logger.info(f'   - Periods analyzed: {len(result.get("regime_states", []))}')
                self.logger.info(f'   - Economic significance: {result.get("economic_significance", False)}')
                
            else:
                self.logger.error('❌ Enhanced HMM regime discovery failed')
                result['hmm_regime_discovery_completed'] = False
                result['success'] = False
                
            return result
            
        except Exception as e:
            self.logger.exception(f'❌ Error in HMM regime discovery wrapper: {e}')
            pipeline_state['hmm_regime_discovery_completed'] = False
            pipeline_state['success'] = False
            pipeline_state['regime_discovery_error'] = str(e)
            return pipeline_state


# Convenience function for direct execution
async def run_hmm_clustering_step(
    symbol: str,
    exchange: str, 
    timeframe: str = "1m",
    data_dir: Optional[str] = None,
    force_rerun: bool = False,
    **kwargs: Any
) -> bool:
    """Run the HMM clustering step directly.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory
        force_rerun: Force re-run
        **kwargs: Additional arguments
        
    Returns:
        Success status
    """
    from .step03_enhanced_hmm_regime_discovery import run_enhanced_step
    
    return await run_enhanced_step(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        success = await run_hmm_clustering_step(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            n_trials=50,
            timeout_minutes=15
        )
        
        if success:
            print("✅ HMM clustering completed successfully!")
        else:
            print("❌ HMM clustering failed")
            
    asyncio.run(main())