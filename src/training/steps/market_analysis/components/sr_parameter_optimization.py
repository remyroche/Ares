"""
SR Parameter Optimization Step.

This step optimizes Support/Resistance detection parameters using backtesting.
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

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger

# Import SR clustering components
try:
    from src.utils.sr_clustering.sr_backtesting_engine import SRBacktestingEngine, BacktestConfig
    from src.utils.sr_clustering.parameter_optimization_engine import get_parameter_optimization_engine, ParameterOptimizationConfig
    SR_CLUSTERING_AVAILABLE = True
except ImportError as e:
    SR_CLUSTERING_AVAILABLE = False
    SRBacktestingEngine = None
    BacktestConfig = None
    get_parameter_optimization_engine = None
    ParameterOptimizationConfig = None
    print(f"Warning: SR clustering components not available: {e}")

class SRParameterOptimizationStep(BaseStep):
    """
    SR Parameter Optimization Step.

    Optimizes Support/Resistance detection parameters using backtesting engine.
    """

    def __init__(self, step_name: str = "sr_parameter_optimization"):
        """Initialize the SR parameter optimization step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('SRParameterOptimization')

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this step must produce."""
        return ['sr_parameter_optimization_result']

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute SR parameter optimization.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - execution_mode: 'full', 'light', or 'blank'

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        self.logger.info('🎯 Starting SR Parameter Optimization')

        try:
            # Check if SR clustering components are available
            if not SR_CLUSTERING_AVAILABLE:
                error_msg = "SR clustering components not available"
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'artifacts': {},
                    'metrics': {},
                    'error': error_msg
                }

            # Get and validate market data
            market_data = await self._load_market_data(config)
            if not self._validate_market_data(market_data):
                error_msg = "Invalid market data for parameter optimization"
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'artifacts': {},
                    'metrics': {},
                    'error': error_msg
                }

            # Configure enhanced parameter optimization with validation
            param_config = self._create_validated_param_config()

            # Ensure data has proper datetime indexing for backtesting
            market_data = self._prepare_data_for_backtesting(market_data)

            # Create backtesting engine with validated hardware optimizations
            backtest_config = self._create_validated_backtest_config()

            engine = SRBacktestingEngine(backtest_config)

            # Create sample SR levels for optimization with proper data splitting
            level_creation_data, backtest_data = self._split_data_for_optimization(market_data)

            # Run parameter optimization
            optimization_result = await self._run_parameter_optimization(
                engine, level_creation_data, backtest_data, param_config
            )

            # Extract results
            optimized_parameters = optimization_result.get('optimized_parameters', {})
            quality_thresholds = optimization_result.get('quality_thresholds', {})
            parameter_optimization_metrics = optimization_result.get('parameter_optimization_metrics', {})

            # Validate that we have the required data
            if not optimized_parameters or not quality_thresholds:
                raise ValueError("Parameter optimization failed to produce required data")

            # Create single consolidated artifact
            artifacts = {
                'sr_parameter_optimization_result': {
                    'optimized_parameters': optimized_parameters,
                    'quality_thresholds': quality_thresholds,
                    'parameter_optimization_metrics': parameter_optimization_metrics,
                    'optimization_summary': {
                        'total_combinations_tested': optimization_result.get('total_combinations_tested', 0),
                        'best_score': optimization_result.get('best_score', 0.0),
                        'optimization_time': optimization_result.get('optimization_time', 0.0)
                    },
                    'metadata': {
                        'symbol': config['symbol'],
                        'exchange': config['exchange'],
                        'timeframe': config['timeframe'],
                        'data_points': len(market_data) if market_data is not None else 0,
                        'execution_timestamp': datetime.now().isoformat()
                    }
                }
            }

            # Calculate metrics
            metrics = {
                'data_points': len(market_data) if market_data is not None else 0,
                'optimization_time': optimization_result.get('optimization_time', 0.0),
                'best_score': optimization_result.get('best_score', 0.0),
                'total_combinations_tested': optimization_result.get('total_combinations_tested', 0)
            }

            self.logger.info('✅ SR Parameter Optimization completed successfully')
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics
            }

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            self.logger.error(f'❌ SR Parameter Optimization failed: {error_type}: {error_msg}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')

            # Return BaseStep format
            return {
                'success': False,
                'artifacts': {},
                'metrics': {},
                'error': f"SR Parameter Optimization failed: {error_type}: {error_msg}"
            }

    async def _load_market_data(self, config: Dict[str, Any]) -> Optional[Any]:
        """Load and prepare market data for optimization with memory optimization."""
        try:
            # Import klines manager here to avoid circular imports
            from src.utils.data.klines_parquet import get_klines_manager

            # Get klines manager
            klines_manager = get_klines_manager(data_dir=config.get('data_dir', 'historical_data'))

            # Parse date filters if provided
            start_date = None
            end_date = None

            if 'start_date' in config and config['start_date']:
                start_date = pd.to_datetime(config['start_date'])
            if 'end_date' in config and config['end_date']:
                end_date = pd.to_datetime(config['end_date'])

            # Load data
            market_data = klines_manager.read_data(
                symbol=config['symbol'],
                interval=config['timeframe'],
                data_type="processed",
                start_date=start_date,
                end_date=end_date
            )

            if market_data is not None and len(market_data) > 0:
                # Ensure timestamp column exists
                if 'timestamp' not in market_data.columns and isinstance(market_data.index, pd.DatetimeIndex):
                    market_data = market_data.copy()
                    market_data['timestamp'] = market_data.index

                return market_data
            else:
                return None

        except Exception as e:
            self.logger.error(f"Failed to load market data: {e}")
            return None

    def _validate_market_data(self, data: Any) -> bool:
        """Validate market data for optimization requirements."""
        if data is None:
            self.logger.error("Market data is None")
            return False

        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            # Check if DataFrame is empty
            if len(data) == 0:
                self.logger.error("Market data DataFrame is empty")
                return False

            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False

            # Check minimum data points
            if len(data) < 100:
                self.logger.error(f"Insufficient data points for optimization: {len(data)} < 100")
                return False

            # Check for NaN values in critical columns
            critical_columns = ['open', 'high', 'low', 'close']
            for col in critical_columns:
                if data[col].isna().any():
                    self.logger.error(f"Found NaN values in critical column: {col}")
                    return False

            # Check for reasonable price values
            for col in critical_columns:
                if (data[col] <= 0).any():
                    self.logger.error(f"Found non-positive values in column: {col}")
                    return False

            self.logger.info(f"Market data validation passed: {len(data)} rows, columns: {list(data.columns)}")
            return True

        # For non-DataFrame data, assume it's valid if not None
        self.logger.warning("Non-DataFrame data provided, validation limited")
        return True

    def _prepare_data_for_backtesting(self, data: Any) -> Any:
        """Prepare data for backtesting with proper datetime indexing."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            return data

        # Process data for backtesting (similar to the original method but simplified)
        if not isinstance(data.index, pd.DatetimeIndex):
            # Try to find timestamp column and set as index
            timestamp_columns = ['timestamp', 'open_time', 'time', 'datetime', 'date']
            for col in timestamp_columns:
                if col in data.columns:
                    data = data.set_index(col)
                    break

        # Convert to datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index, utc=False, errors='coerce')
                data = data.dropna()  # Remove invalid dates
            except Exception as e:
                self.logger.error(f"Failed to convert index to datetime: {e}")
                data.index = pd.RangeIndex(start=0, stop=len(data))

        return data

    def _create_validated_param_config(self) -> Any:
        """Create parameter optimization config with hardware capability validation."""
        if not SR_CLUSTERING_AVAILABLE or ParameterOptimizationConfig is None:
            raise RuntimeError("ParameterOptimizationConfig not available")

        # Get current data for configuration
        current_data = getattr(self, '_current_data', None)

        # Check GPU availability
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
        except ImportError:
            pass

        # Determine optimal memory settings
        memory_limit_gb = 4.0  # Conservative default
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            memory_limit_gb = min(available_memory_gb * 0.5, 8.0)
        except ImportError:
            pass

        return ParameterOptimizationConfig(
            optimization_method='adaptive_grid_search',
            min_samples_for_optimization=10,
            adaptive_optimization=True,
            objective_metric='composite',

            # Hardware optimization settings
            enable_hardware_optimization=True,
            enable_parallel_processing=True,
            max_parallel_workers=None,  # Auto-detect
            enable_gpu_acceleration=gpu_available,
            memory_limit_gb=memory_limit_gb,
            chunk_size=min(1000, max(100, int(len(current_data) / 10) if current_data is not None else 100))
        )

    def _create_validated_backtest_config(self) -> Any:
        """Create backtesting config with hardware capability validation."""
        if not SR_CLUSTERING_AVAILABLE or BacktestConfig is None:
            raise RuntimeError("BacktestConfig not available")

        # Get current data for configuration
        current_data = getattr(self, '_current_data', None)

        # Check GPU availability
        gpu_available = False
        try:
            gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
        except ImportError:
            pass

        # Determine optimal memory settings
        memory_limit_gb = 4.0  # Conservative default
        try:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            memory_limit_gb = min(available_memory_gb * 0.5, 8.0)
        except ImportError:
            pass

        return BacktestConfig(
            enable_parameter_optimization=True,
            parameter_optimization_method='adaptive_grid_search',
            min_samples_for_optimization=10,

            # Hardware optimization settings
            enable_m1_optimizations=True,
            enable_gpu_acceleration=gpu_available,
            enable_memory_optimization=True,
            memory_limit_gb=memory_limit_gb,
            chunk_size=min(1000, max(100, int(len(current_data) / 10) if current_data is not None else 100)),

            # Computation optimization settings
            enable_parallel_processing=True,
            enable_vectorized_operations=True,
            enable_caching=True,
            cache_size_mb=min(100, max(10, int(memory_limit_gb * 10))),
            enable_numba_acceleration=True
        )

    def _split_data_for_optimization(self, market_data: Any) -> Tuple[Any, Any]:
        """Split data properly to avoid data leakage during optimization."""
        if not PANDAS_AVAILABLE or not isinstance(market_data, pd.DataFrame):
            return market_data, market_data

        # Use 70% for training (level creation) and 30% for testing (backtesting)
        split_point = int(len(market_data) * 0.7)

        if split_point < 100:
            return market_data, market_data

        level_creation_data = market_data.iloc[:split_point]
        backtest_data = market_data.iloc[split_point:]

        self.logger.info(f"Data split: {len(level_creation_data)} rows for training, {len(backtest_data)} rows for testing")
        return level_creation_data, backtest_data

    def _get_current_data(self):
        """Get current data reference for configuration methods."""
        return getattr(self, '_current_data', None)

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)
