"""
Regime Data Splitting Component.

This component tags data by regimes discovered in previous stages.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Handle optional dependencies gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger


class RegimeDataSplittingComponent(BaseMarketAnalysisComponent):
    """
    Regime Data Splitting Component.
    
    Tags data by regimes discovered in previous stages.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the regime data splitting component."""
        super().__init__(config)
        self.logger = system_logger.getChild('RegimeDataSplitting')
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['regime_data_splitting_result']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute regime data splitting.
        
        Args:
            data: Market data for regime tagging
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with regime data splitting results
        """
        self.logger.info('✂️ Starting Regime Data Splitting')
        
        try:
            # Import regime data processing utilities
            from src.utils.ml_common.data_processing.regime_data_processing import EnhancedRegimeDataProcessor, RegimeProcessingConfig
            
            # Get market data
            market_data = await self._load_market_data(data)
            if market_data is None or market_data.empty:
                raise ValueError("No market data available for regime data splitting")
            
            # Get regime discovery results from previous stage
            hmm_regime_discovery = pipeline_state.get('hmm_regime_discovery_result', {})
            if not hmm_regime_discovery:
                raise ValueError("No HMM regime discovery results available for data splitting")
            
            # Configure regime data processing
            processing_config = RegimeProcessingConfig(
                regime_assignment_method='hmm_based',
                regime_transition_threshold=0.1,
                min_regime_duration=5,
                smoothing_window=3,
                outlier_detection=True,
                outlier_threshold=2.0,
                
                # Data validation
                validate_regime_assignments=True,
                check_regime_continuity=True,
                fill_missing_assignments=True,
                
                # Hardware optimization
                enable_parallel_processing=True,
                enable_gpu_acceleration=True,
                memory_limit_gb=8.0
            )
            
            # Create regime data processor
            regime_processor = EnhancedRegimeDataProcessor()
            
            # Perform regime data splitting
            splitting_result = await self._perform_regime_data_splitting(
                regime_processor, market_data, hmm_regime_discovery, processing_config
            )
            
            # Extract results
            regime_data = splitting_result.get('regime_data', {})
            regime_stats = splitting_result.get('regime_stats', {})
            processing_metrics = splitting_result.get('processing_metrics', {})
            
            # Validate that we have regime data
            if not regime_data or not regime_stats:
                raise ValueError("Regime data splitting completed but no regime data was created")
            
            # Create single consolidated artifact
            artifacts = {
                'regime_data_splitting_result': {
                    'regime_data': regime_data,
                    'regime_stats': regime_stats,
                    'processing_metrics': processing_metrics,
                    'splitting_summary': {
                        'total_data_points': len(market_data) if market_data is not None else 0,
                        'regime_count': regime_stats.get('total_regimes', 0),
                        'regime_distribution': regime_stats.get('regime_distribution', {}),
                        'processing_time': splitting_result.get('processing_time', 0.0)
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
            
            self.logger.info(f'✅ Regime Data Splitting completed: {regime_stats.get("total_regimes", 0)} regimes processed')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'regime_count': regime_stats.get('total_regimes', 0)
                }
            )
            
        except Exception as e:
            self.logger.error(f'❌ Regime Data Splitting failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e)
            )
    
    async def _load_market_data(self, data: Any) -> Optional[Any]:
        """Load and prepare market data for regime splitting."""
        if data is None:
            return None
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return data.copy()
        
        # Handle other data types if needed
        return data
    
    async def _perform_regime_data_splitting(
        self, 
        regime_processor: Any, 
        market_data: Any, 
        hmm_regime_discovery: Dict[str, Any],
        config: Any
    ) -> Dict[str, Any]:
        """Perform the actual regime data splitting process."""
        try:
            # Prepare data for regime processing
            prepared_data = self._prepare_data_for_regime_splitting(market_data, hmm_regime_discovery)
            
            # Perform regime data processing
            processing_result = await regime_processor.process_regime_data(prepared_data, config)
            
            return processing_result
            
        except Exception as e:
            self.logger.error(f"Regime data splitting process failed: {e}")
            # Return fallback splitting result
            return {
                'regime_data': {},
                'regime_stats': {
                    'total_regimes': 0,
                    'regime_distribution': {},
                    'processing_method': 'fallback',
                    'error': str(e)
                },
                'processing_metrics': {
                    'processing_method': 'fallback',
                    'error': str(e)
                },
                'processing_time': 0.0
            }
    
    def _prepare_data_for_regime_splitting(self, data: Any, hmm_regime_discovery: Dict[str, Any]) -> Any:
        """Prepare market data and regime discovery results for splitting."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            self.logger.warning("Pandas not available or data is not a DataFrame, using fallback")
            return {
                'market_data': data,
                'hmm_regime_discovery': hmm_regime_discovery
            }
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing columns for regime splitting: {missing_columns}")
            # Use available columns or create fallback data
            for col in missing_columns:
                if col == 'volume':
                    data[col] = 1000  # Default volume
                else:
                    data[col] = data.get('close', 100.0)  # Use close price as fallback
        
        return {
            'market_data': data,
            'hmm_regime_discovery': hmm_regime_discovery
        }