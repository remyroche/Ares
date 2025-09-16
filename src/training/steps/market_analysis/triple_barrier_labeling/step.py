"""
Triple Barrier Labeling Step Implementation

This module provides the step implementation for triple barrier labeling in the market analysis pipeline.
"""

import logging
import time
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

from src.utils.logger import system_logger
from src.utils.tprint import tprint
from src.training.steps.market_analysis.triple_barrier_labeling.unified_labeler import (
    UnifiedTripleBarrierLabeler, TripleBarrierConfig
)


class TripleBarrierLabelingStep:
    """
    Triple Barrier Labeling Step for Market Analysis Pipeline.
    
    This step applies triple barrier labeling to market data with regime awareness.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the triple barrier labeling step."""
        self.config = config or {}
        self.logger = system_logger.getChild('TripleBarrierLabelingStep')
        
        # Validate dependencies
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy is required for triple barrier labeling")
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for triple barrier labeling")
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute triple barrier labeling step.
        
        Args:
            data: Market data for labeling
            pipeline_state: Current pipeline state
            
        Returns:
            Dict containing labeling results and artifacts
        """
        tprint('🏷️ Starting Triple Barrier Labeling Step')
        self.logger.info('🏷️ Starting Triple Barrier Labeling')
        start_time = time.time()
        
        try:
            # Step 1: Validate inputs
            validation_result = await self._validate_inputs(data, pipeline_state)
            if not validation_result['valid']:
                return self._create_error_result("Input validation failed", validation_result['errors'])
            
            # Step 2: Load and prepare data
            market_data = await self._load_market_data(data)
            if market_data is None or market_data.empty:
                return self._create_error_result("No market data available for labeling")
            
            # Step 3: Get regime data from pipeline state
            regime_data = pipeline_state.get('regime_data', {})
            regime_assignments = regime_data.get('regime_assignments', [])
            
            # Step 4: Configure triple barrier labeling
            labeling_config = self._create_labeling_config()
            
            # Step 5: Initialize unified labeler
            labeler = UnifiedTripleBarrierLabeler(labeling_config)
            
            # Step 6: Apply triple barrier labeling
            labeling_result = await self._apply_triple_barrier_labeling(
                labeler, market_data, regime_assignments
            )
            
            # Step 7: Create results
            execution_time = time.time() - start_time
            
            results = {
                'status': 'completed',
                'execution_time': execution_time,
                'artifacts': {
                    'triple_barrier_labeling_result': {
                        'labeled_data': labeling_result.get('labeled_data', {}),
                        'labeling_metrics': labeling_result.get('labeling_metrics', {}),
                        'barrier_config': labeling_result.get('barrier_config', {}),
                        'regime_analysis': labeling_result.get('regime_analysis', {})
                    }
                },
                'metadata': {
                    'total_samples': len(market_data),
                    'regimes_processed': len(set(regime_assignments)) if regime_assignments else 0,
                    'labeling_method': 'triple_barrier_unified',
                    'config': self.config,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            tprint(f'✅ Triple Barrier Labeling completed in {execution_time:.2f}s')
            self.logger.info(f'✅ Triple Barrier Labeling completed in {execution_time:.2f}s')
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint(f'❌ Triple Barrier Labeling failed: {e}')
            self.logger.error(f'❌ Triple Barrier Labeling failed: {e}')
            
            return self._create_error_result(
                "Triple barrier labeling execution failed", 
                [str(e)],
                execution_time
            )
    
    async def _validate_inputs(self, data: Any, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and pipeline state."""
        errors = []
        
        if data is None:
            errors.append("Input data is None")
        
        if not pipeline_state:
            errors.append("Pipeline state is empty")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    async def _load_market_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Load and prepare market data."""
        try:
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, dict) and 'data' in data:
                return data['data']
            elif hasattr(data, 'to_dataframe'):
                return data.to_dataframe()
            else:
                self.logger.warning("Unknown data type, attempting to convert to DataFrame")
                return pd.DataFrame(data) if data is not None else None
        except Exception as e:
            self.logger.error(f"Failed to load market data: {e}")
            return None
    
    def _create_labeling_config(self) -> TripleBarrierConfig:
        """Create triple barrier labeling configuration."""
        return TripleBarrierConfig(
            profit_take_multiplier=self.config.get('profit_take_multiplier', 0.002),
            stop_loss_multiplier=self.config.get('stop_loss_multiplier', 0.001),
            transaction_cost=self.config.get('transaction_cost', 0.0008),
            regime_aware=self.config.get('regime_aware', True),
            enable_hardware_optimizations=self.config.get('enable_hardware_optimizations', True)
        )
    
    async def _apply_triple_barrier_labeling(
        self, 
        labeler: UnifiedTripleBarrierLabeler, 
        market_data: pd.DataFrame,
        regime_assignments: List[int]
    ) -> Dict[str, Any]:
        """Apply triple barrier labeling to market data."""
        try:
            # Prepare data for labeling
            if 'close' not in market_data.columns:
                # Try to find price column
                price_columns = ['price', 'close_price', 'last_price']
                price_col = None
                for col in price_columns:
                    if col in market_data.columns:
                        price_col = col
                        break
                
                if price_col is None:
                    raise ValueError("No price column found in market data")
                
                market_data = market_data.copy()
                market_data['close'] = market_data[price_col]
            
            # Apply labeling
            labeling_result = labeler.apply_labeling(market_data)
            
            # Convert TripleBarrierResult to expected format
            if labeling_result.success:
                return {
                    'labeled_data': labeling_result.labeled_data,
                    'labeling_metrics': {
                        'total_labels_generated': labeling_result.total_labels_generated,
                        'label_distribution': labeling_result.label_distribution,
                        'data_quality_score': labeling_result.data_quality_score,
                        'execution_duration': labeling_result.execution_duration
                    },
                    'barrier_config': {
                        'profit_take_multiplier': labeling_result.labeled_data.get('potential_profit_pct', {}).mean() if labeling_result.labeled_data is not None else 0,
                        'stop_loss_multiplier': self.config.get('stop_loss_multiplier', 0.001),
                        'transaction_cost': self.config.get('transaction_cost', 0.0008)
                    },
                    'regime_analysis': {
                        'regimes_processed': len(set(regime_assignments)) if regime_assignments else 0,
                        'regime_coverage': labeling_result.regime_coverage
                    }
                }
            else:
                raise Exception(f"Labeling failed: {labeling_result.error_message}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply triple barrier labeling: {e}")
            return {
                'labeled_data': {},
                'labeling_metrics': {'error': str(e)},
                'barrier_config': {},
                'regime_analysis': {}
            }
    
    def _create_error_result(
        self, 
        error_message: str, 
        errors: List[str] = None, 
        execution_time: float = 0.0
    ) -> Dict[str, Any]:
        """Create error result."""
        return {
            'status': 'failed',
            'error': error_message,
            'errors': errors or [error_message],
            'execution_time': execution_time,
            'artifacts': {
                'triple_barrier_labeling_result': {
                    'labeled_data': {},
                    'labeling_metrics': {'error': error_message},
                    'barrier_config': {},
                    'regime_analysis': {}
                }
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config
            }
        }


# Convenience function for pipeline integration
async def execute_triple_barrier_labeling_step(
    data: Any, 
    pipeline_state: Dict[str, Any], 
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute triple barrier labeling step."""
    step = TripleBarrierLabelingStep(config)
    return await step.execute(data, pipeline_state)