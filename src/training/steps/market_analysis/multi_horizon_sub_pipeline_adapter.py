"""
Multi-Horizon Sub-Pipeline Adapter

This module provides an adapter to integrate multi-horizon labeling into the existing
sub-pipeline system, replacing the triple barrier method.

Key features:
- Drop-in replacement for triple barrier labeling
- Maintains compatibility with existing sub-pipeline
- Provides enhanced labeling with reversal capture
- Optimized for short-term, high-frequency trading
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

# Optimized imports using common utilities
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time
from src.utils.common_operations import (
    validate_dataframe, validate_dataframe_columns, safe_dataframe_operation,
    safe_fillna, safe_convert_dtypes, safe_merge_dataframes,
    calculate_data_quality_metrics, create_summary_statistics,
    timed_operation, memory_checkpoint, gpu_context,
    integrate_with_m1_optimizers
)
from src.utils.math_validation import (
    safe_mean, safe_std, validate_finite, safe_percentage_change
)
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.serialization_utils import UniversalSerializer

# Import the multi-horizon labeler
from .multi_horizon_profit_labeler import (
    MultiHorizonProfitLabeler, 
    MultiHorizonConfig,
    apply_multi_horizon_labeling
)

class MultiHorizonSubPipelineAdapter:
    """
    Adapter for integrating multi-horizon labeling into existing sub-pipeline.
    
    This adapter provides a drop-in replacement for the triple barrier labeling
    step while maintaining compatibility with the existing pipeline structure.
    """
    
    def __init__(self):
        """Initialize the adapter with hardware optimizations."""
        self.logger = get_logger('MultiHorizonSubPipelineAdapter')
        
        # Initialize hardware optimizers
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        self.serializer = UniversalSerializer()
        
        # Optimize CPU for data processing
        if self.cpu_optimizer:
            self.cpu_optimizer.optimize_numpy_operations()
        
        self.logger.info('🔄 Multi-Horizon Sub-Pipeline Adapter initialized with M1 optimizations')
    
    @timed_operation
    @traced(span_name='execute_multi_horizon_labeling_step')
    @validates()
    @handles_errors(exceptions=(Exception,), default_return={'status': 'failed', 'error': 'Unknown error'})
    @log_execution_time()
    def execute_multi_horizon_labeling_step(self,
                                          data: pd.DataFrame,
                                          regime_labels: Optional[pd.Series] = None,
                                          config: Optional[Dict[str, Any]] = None,
                                          symbol: Optional[str] = None,
                                          exchange: Optional[str] = None,
                                          timeframe: Optional[str] = None,
                                          mode: str = 'full') -> Dict[str, Any]:
        """
        Execute multi-horizon labeling step compatible with sub-pipeline.
        
        This method provides the same interface as the original triple barrier
        labeling step but uses the new multi-horizon approach.
        
        Args:
            data: Input OHLCV data
            regime_labels: Optional regime labels
            config: Configuration dictionary
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            mode: Execution mode
            
        Returns:
            Dictionary with labeling results compatible with sub-pipeline
        """
        self.logger.info(f'🎯 Executing multi-horizon labeling step for {symbol or "unknown"} on {timeframe or "unknown"}')
        
        try:
            # Validate input data with enhanced validation
            if not validate_dataframe(data):
                return {
                    'status': 'failed',
                    'error': 'Invalid or empty DataFrame provided',
                    'artifacts': {}
                }
            
            # Optimize data memory usage
            if self.memory_optimizer:
                data = self.memory_optimizer.optimize_dataframe_memory(data)
            
            # Use memory checkpoint for large operations
            with memory_checkpoint('multi_horizon_labeling'):
                # Create multi-horizon configuration
                labeling_config = self._create_labeling_config(config)
                
                # Apply multi-horizon labeling with safe operations
                labeled_data = safe_dataframe_operation(
                    data, 
                    apply_multi_horizon_labeling, 
                    labeling_config
                )
                
                # Calculate labeling metrics with enhanced validation
                labeling_metrics = self._calculate_labeling_metrics(labeled_data, data)
            
            # Create result compatible with sub-pipeline with enhanced metrics
            result = {
                'status': 'completed',
                'execution_time': datetime.now().isoformat(),
                'artifacts': {
                    'multi_horizon_labeling_result': {
                        'labeled_data': labeled_data,
                        'labeling_metrics': labeling_metrics,
                        'config': labeling_config.__dict__,
                        'method': 'multi_horizon_profit_labeling',
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'data_quality': calculate_data_quality_metrics(labeled_data),
                        'summary_stats': create_summary_statistics(labeled_data)
                    }
                }
            }
            
            self.logger.info(f'✅ Multi-horizon labeling completed: {len(labeled_data)} samples, {labeled_data.shape[1]} features')
            return result
            
        except Exception as e:
            self.logger.error(f'❌ Multi-horizon labeling failed: {e}')
            return {
                'status': 'failed',
                'error': str(e),
                'artifacts': {}
            }
    
    def _create_labeling_config(self, config: Optional[Dict[str, Any]] = None) -> MultiHorizonConfig:
        """Create multi-horizon configuration from sub-pipeline config."""
        labeling_config = MultiHorizonConfig()
        
        if config:
            # Update profit targets if specified
            if 'profit_targets' in config:
                labeling_config.profit_targets = config['profit_targets']
            
            # Update time horizons if specified
            if 'time_horizons' in config:
                labeling_config.time_horizons = config['time_horizons']
            
            # Update other parameters
            if 'transaction_cost' in config:
                labeling_config.transaction_cost = config['transaction_cost']
            
            if 'enable_quality_scoring' in config:
                labeling_config.enable_quality_scoring = config['enable_quality_scoring']
            
            if 'leverage_aware' in config:
                labeling_config.leverage_aware = config['leverage_aware']
        
        return labeling_config
    
    def _calculate_labeling_metrics(self, labeled_data: pd.DataFrame, 
                                  original_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive labeling metrics with safe operations."""
        metrics = {
            'total_samples': len(labeled_data),
            'original_samples': len(original_data),
            'total_features': labeled_data.shape[1],
            'new_features_added': labeled_data.shape[1] - original_data.shape[1],
            'labeling_method': 'multi_horizon_profit_labeling'
        }
        
        # Calculate target-specific metrics with safe operations
        target_columns = [col for col in labeled_data.columns if col.endswith('_prob')]
        metrics['probability_targets'] = len(target_columns)
        
        # Calculate composite score metrics with safe operations
        composite_columns = [
            'overall_opportunity', 'leverage_adjusted_score', 
            'immediate_opportunity', 'short_term_opportunity',
            'reversal_capture_score', 'reassessment_frequency'
        ]
        
        for col in composite_columns:
            if col in labeled_data.columns:
                values = labeled_data[col].dropna()
                if len(values) > 0:
                    metrics[f'{col}_mean'] = validate_finite(safe_mean(values, default=0.0), f'{col}_mean')
                    metrics[f'{col}_std'] = validate_finite(safe_std(values, default=0.0), f'{col}_std')
                    high_quality_count = (values > 0.7).sum()
                    metrics[f'{col}_high_quality_ratio'] = safe_divide(high_quality_count, len(values), default=0.0)
        
        # Overall quality metrics with safe operations
        if 'overall_opportunity' in labeled_data.columns:
            overall_opp = labeled_data['overall_opportunity'].dropna()
            if len(overall_opp) > 0:
                high_opp_count = (overall_opp > 0.7).sum()
                metrics['high_opportunity_samples'] = int(high_opp_count)
                metrics['high_opportunity_ratio'] = safe_divide(high_opp_count, len(overall_opp), default=0.0)
                metrics['average_opportunity_score'] = validate_finite(safe_mean(overall_opp, default=0.0), 'average_opportunity_score')
        
        return metrics

# Convenience function for sub-pipeline integration
def execute_multi_horizon_labeling_step(data: pd.DataFrame,
                                       regime_labels: Optional[pd.Series] = None,
                                       config: Optional[Dict[str, Any]] = None,
                                       symbol: Optional[str] = None,
                                       exchange: Optional[str] = None,
                                       timeframe: Optional[str] = None,
                                       mode: str = 'full') -> Dict[str, Any]:
    """
    Execute multi-horizon labeling step (sub-pipeline compatible).
    
    This function provides a drop-in replacement for the original triple barrier
    labeling step in the sub-pipeline system.
    """
    adapter = MultiHorizonSubPipelineAdapter()
    return adapter.execute_multi_horizon_labeling_step(
        data=data,
        regime_labels=regime_labels,
        config=config,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        mode=mode
    )

# Test function
if __name__ == '__main__':
    from src.utils.tprint import tprint
    import numpy as np
    
    tprint('🧪 Testing Multi-Horizon Sub-Pipeline Adapter')
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=500, freq='5min')
    np.random.seed(42)
    
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.002, 500)
    prices = [base_price]
    
    for ret in returns[:-1]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 500)
    }, index=dates)
    
    # Ensure OHLC consistency
    for i in range(len(data)):
        data.loc[data.index[i], 'high'] = max(data.iloc[i][['open', 'high', 'low', 'close']])
        data.loc[data.index[i], 'low'] = min(data.iloc[i][['open', 'high', 'low', 'close']])
    
    # Test sub-pipeline adapter
    tprint('\n🔄 Testing sub-pipeline adapter...')
    
    config = {
        'profit_targets': {
            'micro': 0.003,
            'small': 0.005,
            'medium': 0.007,
            'good': 0.010
        },
        'time_horizons': {
            'immediate': 2,
            'short': 4
        },
        'transaction_cost': 0.0008
    }
    
    result = execute_multi_horizon_labeling_step(
        data=data,
        config=config,
        symbol='TESTUSDT',
        exchange='test',
        timeframe='5m'
    )
    
    if result['status'] == 'completed':
        tprint('✅ Sub-pipeline adapter test successful!')
        
        artifacts = result['artifacts']['multi_horizon_labeling_result']
        metrics = artifacts['labeling_metrics']
        
        tprint(f'📊 Results:')
        tprint(f'   → Status: {result["status"]}')
        tprint(f'   → Total samples: {metrics["total_samples"]}')
        tprint(f'   → New features: {metrics["new_features_added"]}')
        tprint(f'   → Probability targets: {metrics["probability_targets"]}')
        tprint(f'   → High opportunity ratio: {metrics.get("high_opportunity_ratio", 0):.1%}')
        
    else:
        tprint(f'❌ Sub-pipeline adapter test failed: {result.get("error", "Unknown error")}')
    
    tprint('✅ Multi-Horizon Sub-Pipeline Adapter test completed!')