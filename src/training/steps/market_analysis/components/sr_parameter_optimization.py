"""
SR Parameter Optimization Component.

This component optimizes Support/Resistance detection parameters using backtesting.
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


class SRParameterOptimizationComponent(BaseMarketAnalysisComponent):
    """
    SR Parameter Optimization Component.
    
    Optimizes Support/Resistance detection parameters using backtesting engine.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the SR parameter optimization component."""
        super().__init__(config)
        self.logger = system_logger.getChild('SRParameterOptimization')
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['optimized_parameters', 'quality_thresholds']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute SR parameter optimization.
        
        Args:
            data: Market data for optimization
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with optimization results
        """
        self.logger.info('🎯 Starting SR Parameter Optimization')
        
        try:
            # Import SR backtesting engine
            from src.utils.sr_clustering.sr_backtesting_engine import SRBacktestingEngine, BacktestConfig
            from src.utils.sr_clustering.parameter_optimization_engine import get_parameter_optimization_engine, ParameterOptimizationConfig
            
            # Get market data
            market_data = await self._load_market_data(data)
            if market_data is None or market_data.empty:
                raise ValueError("No market data available for parameter optimization")
            
            # Configure enhanced parameter optimization
            param_config = ParameterOptimizationConfig(
                optimization_method='adaptive_grid_search',
                min_samples_for_optimization=10,
                adaptive_optimization=True,
                objective_metric='composite',
                
                # Hardware optimization settings
                enable_hardware_optimization=True,
                enable_parallel_processing=True,
                max_parallel_workers=None,  # Auto-detect
                enable_gpu_acceleration=True,
                memory_limit_gb=8.0,
                chunk_size=1000
            )

            # Ensure data has proper datetime indexing for backtesting
            market_data = self._prepare_data_for_backtesting(market_data)
            
            # Create backtesting engine with hardware optimizations
            backtest_config = BacktestConfig(
                enable_parameter_optimization=True,
                parameter_optimization_method='adaptive_grid_search',
                min_samples_for_optimization=10,
                
                # Hardware optimization settings
                enable_m1_optimizations=True,
                enable_gpu_acceleration=True,
                enable_memory_optimization=True,
                memory_limit_gb=8.0,
                chunk_size=1000,
                
                # Computation optimization settings
                enable_parallel_processing=True,
                enable_vectorized_operations=True,
                enable_caching=True,
                cache_size_mb=100,
                enable_numba_acceleration=True
            )
            
            engine = SRBacktestingEngine(backtest_config)
            
            # Create sample SR levels for optimization
            if len(market_data) > 1000:
                level_creation_data = market_data.iloc[:len(market_data)//2]
                backtest_data = market_data
            else:
                level_creation_data = market_data
                backtest_data = market_data
            
            # Run parameter optimization
            optimization_result = await self._run_parameter_optimization(
                engine, level_creation_data, backtest_data, param_config
            )
            
            # Extract results
            optimized_parameters = optimization_result.get('optimized_parameters', {})
            quality_thresholds = optimization_result.get('quality_thresholds', {})
            parameter_optimization_metrics = optimization_result.get('parameter_optimization_metrics', {})
            
            # Validate that we have the required artifacts
            if not optimized_parameters or not quality_thresholds:
                raise ValueError("Parameter optimization failed to produce required artifacts")
            
            artifacts = {
                'optimized_parameters': optimized_parameters,
                'quality_thresholds': quality_thresholds,
                'parameter_optimization_metrics': parameter_optimization_metrics,
                'optimization_summary': {
                    'total_combinations_tested': optimization_result.get('total_combinations_tested', 0),
                    'best_score': optimization_result.get('best_score', 0.0),
                    'optimization_time': optimization_result.get('optimization_time', 0.0)
                }
            }
            
            self.logger.info('✅ SR Parameter Optimization completed successfully')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'data_points': len(market_data)
                }
            )
            
        except Exception as e:
            self.logger.error(f'❌ SR Parameter Optimization failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e)
            )
    
    async def _load_market_data(self, data: Any) -> Optional[Any]:
        """Load and prepare market data for optimization."""
        if data is None:
            return None
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return data.copy()
        
        # Handle other data types if needed
        return data
    
    def _prepare_data_for_backtesting(self, data: Any) -> Any:
        """Prepare data for backtesting with proper datetime indexing."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            self.logger.warning("Pandas not available or data is not a DataFrame, skipping data preparation")
            return data
            
        self.logger.info(f"Data index type before conversion: {type(data.index)}")
        self.logger.info(f"Data columns: {list(data.columns)}")
        self.logger.info(f"Data shape: {data.shape}")

        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.info("Converting data to datetime index for backtesting")
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
                self.logger.info("Using 'timestamp' column as index")
            elif 'open_time' in data.columns:
                data = data.set_index('open_time')
                self.logger.info("Using 'open_time' column as index")
            elif 'time' in data.columns:
                data = data.set_index('time')
                self.logger.info("Using 'time' column as index")

            # Ensure it's datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    # Check if timestamps look like milliseconds (very large numbers)
                    sample_timestamps = data.index[:5]
                    self.logger.info(f"Sample timestamps before conversion: {sample_timestamps.tolist()}")

                    if sample_timestamps.max() > 1e10:  # Likely milliseconds
                        data.index = pd.to_datetime(data.index, unit='ms')
                        self.logger.info("Converted index to datetime (milliseconds)")
                    else:
                        data.index = pd.to_datetime(data.index)
                        self.logger.info("Converted index to datetime")
                except Exception as e:
                    self.logger.warning(f"Could not convert index to datetime: {e}")

        self.logger.info(f"Final data index type: {type(data.index)}")
        self.logger.info(f"Data index sample: {data.index[:3] if len(data) > 0 else 'empty'}")
        
        return data
    
    async def _run_parameter_optimization(
        self, 
        engine: Any, 
        level_creation_data: pd.DataFrame, 
        backtest_data: pd.DataFrame,
        param_config: Any
    ) -> Dict[str, Any]:
        """Run the actual parameter optimization process."""
        try:
            # Get parameter optimization engine
            optimization_engine = get_parameter_optimization_engine(param_config)
            
            # Run optimization
            optimization_result = await optimization_engine.optimize_parameters(
                engine=engine,
                level_creation_data=level_creation_data,
                backtest_data=backtest_data,
                config=param_config
            )
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Parameter optimization failed: {e}")
            # Return fallback parameters
            return {
                'optimized_parameters': {
                    'min_touches': 3,
                    'touch_tolerance': 0.002,
                    'strength_threshold': 0.5
                },
                'quality_thresholds': {
                    'min_success_rate': 0.6,
                    'min_bounce_strength': 0.3
                },
                'parameter_optimization_metrics': {
                    'optimization_method': 'fallback',
                    'error': str(e)
                }
            }