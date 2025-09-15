"""
Triple Barrier Labeling Component.

This component applies the triple barrier method for data labeling.

DEPRECATED: This component is deprecated. Use unified_triple_barrier_labeler.py instead.
This file is kept for backward compatibility and will be removed in a future version.
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


class TripleBarrierLabelingComponent(BaseMarketAnalysisComponent):
    """
    Triple Barrier Labeling Component.
    
    Applies the triple barrier method for data labeling.
    
    DEPRECATED: Use UnifiedTripleBarrierLabeler from unified_triple_barrier_labeler.py instead.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the triple barrier labeling component."""
        import warnings
        warnings.warn(
            "TripleBarrierLabelingComponent is deprecated. Use UnifiedTripleBarrierLabeler from unified_triple_barrier_labeler.py instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(config)
        self.logger = system_logger.getChild('TripleBarrierLabeling')
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['triple_barrier_labeling_result']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute triple barrier labeling.
        
        Args:
            data: Market data for labeling
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with triple barrier labeling results
        """
        self.logger.info('🏷️ Starting Triple Barrier Labeling')
        
        try:
            # Import triple barrier labeling utilities
            from src.utils.ml_common.data_processing.data_labeling import EnhancedDataLabeler, TripleBarrierConfig, LabelingMethod
            
            # Get market data
            market_data = await self._load_market_data(data)
            if market_data is None or market_data.empty:
                raise ValueError("No market data available for triple barrier labeling")
            
            # Get regime data from previous stage
            regime_data_splitting = pipeline_state.get('regime_data_splitting_result', {})
            if not regime_data_splitting:
                raise ValueError("No regime data splitting results available for labeling")
            
            # Configure triple barrier labeling
            labeling_config = TripleBarrierConfig(
                labeling_method=LabelingMethod.TRIPLE_BARRIER,
                profit_taking_threshold=0.02,  # 2% profit taking
                stop_loss_threshold=0.01,      # 1% stop loss
                time_horizon=20,               # 20 periods max hold
                volatility_lookback=20,        # 20 periods for volatility calculation
                min_trade_duration=1,          # Minimum 1 period hold
                max_trade_duration=50,         # Maximum 50 periods hold
                
                # Regime-aware labeling
                enable_regime_aware_labeling=True,
                regime_specific_thresholds=True,
                regime_transition_handling='conservative',
                
                # Data validation
                validate_labels=True,
                check_label_distribution=True,
                balance_labels=True,
                
                # Hardware optimization
                enable_parallel_processing=True,
                enable_gpu_acceleration=True,
                memory_limit_gb=8.0
            )
            
            # Create data labeler
            data_labeler = EnhancedDataLabeler()
            
            # Perform triple barrier labeling
            labeling_result = await self._perform_triple_barrier_labeling(
                data_labeler, market_data, regime_data_splitting, labeling_config
            )
            
            # Extract results
            labeled_data = labeling_result.get('labeled_data', {})
            labeling_metrics = labeling_result.get('labeling_metrics', {})
            label_distribution = labeling_result.get('label_distribution', {})
            
            # Validate that we have labeled data
            if not labeled_data or not labeling_metrics:
                raise ValueError("Triple barrier labeling completed but no labeled data was created")
            
            # Create single consolidated artifact
            artifacts = {
                'triple_barrier_labeling_result': {
                    'labeled_data': labeled_data,
                    'labeling_metrics': labeling_metrics,
                    'label_distribution': label_distribution,
                    'labeling_summary': {
                        'total_data_points': len(market_data) if market_data is not None else 0,
                        'labeled_points': labeling_metrics.get('total_labels', 0),
                        'label_distribution': label_distribution,
                        'labeling_time': labeling_result.get('labeling_time', 0.0)
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
            
            self.logger.info(f'✅ Triple Barrier Labeling completed: {labeling_metrics.get("total_labels", 0)} labels created')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'labels_created': labeling_metrics.get('total_labels', 0)
                }
            )
            
        except Exception as e:
            self.logger.error(f'❌ Triple Barrier Labeling failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e)
            )
    
    async def _load_market_data(self, data: Any) -> Optional[Any]:
        """Load and prepare market data for labeling."""
        if data is None:
            return None
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return data.copy()
        
        # Handle other data types if needed
        return data
    
    async def _perform_triple_barrier_labeling(
        self, 
        data_labeler: Any, 
        market_data: Any, 
        regime_data_splitting: Dict[str, Any],
        config: Any
    ) -> Dict[str, Any]:
        """Perform the actual triple barrier labeling process."""
        try:
            # Prepare data for labeling
            prepared_data = self._prepare_data_for_labeling(market_data, regime_data_splitting)
            
            # Perform triple barrier labeling
            labeling_result = await data_labeler.apply_triple_barrier_labeling(prepared_data, config)
            
            return labeling_result
            
        except Exception as e:
            self.logger.error(f"Triple barrier labeling process failed: {e}")
            # Return fallback labeling result
            return {
                'labeled_data': {},
                'labeling_metrics': {
                    'total_labels': 0,
                    'labeling_method': 'fallback',
                    'error': str(e)
                },
                'label_distribution': {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                },
                'labeling_time': 0.0
            }
    
    def _prepare_data_for_labeling(self, data: Any, regime_data_splitting: Dict[str, Any]) -> Any:
        """Prepare market data and regime data for labeling."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            self.logger.warning("Pandas not available or data is not a DataFrame, using fallback")
            return {
                'market_data': data,
                'regime_data_splitting': regime_data_splitting
            }
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing columns for labeling: {missing_columns}")
            # Use available columns or create fallback data
            for col in missing_columns:
                if col == 'volume':
                    data[col] = 1000  # Default volume
                else:
                    data[col] = data.get('close', 100.0)  # Use close price as fallback
        
        return {
            'market_data': data,
            'regime_data_splitting': regime_data_splitting
        }