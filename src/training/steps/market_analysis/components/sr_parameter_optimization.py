"""
SR Parameter Optimization Step.

BaseStep-based implementation for optimizing Support/Resistance detection parameters.
Migrated from the component pattern to use the new BaseStep architecture.
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
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
    tprint_debug, tprint_data_format, tprint_data_preview, tprint_performance,
    tprint_timer, tprint_structured, LogLevel
)

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
    SR Parameter Optimization Step using BaseStep pattern.

    Optimizes Support/Resistance detection parameters using backtesting engine.
    """

    def __init__(self, step_name: str = "sr_parameter_optimization"):
        """Initialize the SR parameter optimization step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('SRParameterOptimization')

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute SR parameter optimization.

        Args:
            config: Configuration dictionary with parameters:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - data_dir: Data directory path
                - start_date: Start date (optional)
                - end_date: End date (optional)
                - market_data: Market data (optional)
                - sr_levels: SR levels (optional)
                - parameter_grid: Parameter grid for optimization (optional)
                - optimization_metric: Metric to optimize (default: 'accuracy')

        Returns:
            Dictionary with optimization results and best parameters
        """
        start_time = datetime.now()
        
        try:
            tprint(f"🔍 Starting SR parameter optimization for {config.get('symbol', 'UNKNOWN')}", "INFO")
            tprint_structured({
                "step": "sr_parameter_optimization_start",
                "component": "SRParameterOptimizationStep",
                "symbol": config.get('symbol', 'UNKNOWN'),
                "timestamp": start_time.isoformat()
            }, level=LogLevel.INFO)
            
            # Set context for artifact management
            tprint_debug("Setting up execution context", level=LogLevel.DEBUG)
            self._set_context(
                symbol=config.get('symbol', 'ETHUSDT'),
                exchange=config.get('exchange', 'binance'),
                direction=config.get('direction', 'both'),
                model=config.get('model', 'default')
            )

            # Check if SR clustering components are available
            if not SR_CLUSTERING_AVAILABLE:
                error_msg = "SR clustering components not available"
                tprint(f"❌ {error_msg}", "ERROR")
                tprint_error(f"SR clustering components not available", level=LogLevel.ERROR)
                tprint_structured({
                    "step": "sr_parameter_optimization_error",
                    "error": error_msg,
                    "error_type": "ImportError"
                }, level=LogLevel.ERROR)
                return {
                    'success': False,
                    'error': error_msg,
                    'optimization_results': {},
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }

            # Load market data
            tprint_info("Loading market data for optimization", level=LogLevel.INFO)
            market_data = self._load_market_data(config)
            if market_data is None:
                tprint_error("No market data found", level=LogLevel.ERROR)
                raise ValueError("No market data found")
            
            tprint(f"✅ Loaded market data: {market_data.shape[0]} rows, {market_data.shape[1]} columns", "SUCCESS")
            tprint_success(f"Loaded market data: {market_data.shape[0]} rows, {market_data.shape[1]} columns", level=LogLevel.INFO)
            
            # Preview loaded market data for troubleshooting
            tprint_data_preview(market_data, "market_data_for_optimization", max_rows=5)
            tprint_data_format(market_data, "market_data_format", level=LogLevel.DEBUG)

            # Load SR levels
            tprint_info("Loading SR levels for optimization", level=LogLevel.INFO)
            sr_levels = self._load_sr_levels(config)
            if sr_levels is None:
                tprint_error("No SR levels found", level=LogLevel.ERROR)
                raise ValueError("No SR levels found")
            
            tprint(f"✅ Loaded SR levels: {len(sr_levels)} levels", "SUCCESS")
            tprint_success(f"Loaded SR levels: {len(sr_levels)} levels", level=LogLevel.INFO)
            
            # Preview loaded SR levels for troubleshooting
            tprint_data_preview(sr_levels, "sr_levels_for_optimization", max_rows=10)
            tprint_data_format(sr_levels, "sr_levels_format", level=LogLevel.DEBUG)

            # Get parameter grid
            tprint_info("Getting parameter grid for optimization", level=LogLevel.INFO)
            parameter_grid = self._get_parameter_grid(config)
            tprint_data_format(parameter_grid, "parameter_grid", level=LogLevel.DEBUG)

            # Configure enhanced parameter optimization with validation
            tprint_info("Creating validated parameter configuration", level=LogLevel.INFO)
            param_config = self._create_validated_param_config()
            tprint_data_format(param_config, "param_config", level=LogLevel.DEBUG)

            # Ensure data has proper datetime indexing for backtesting
            tprint_info("Preparing data for backtesting", level=LogLevel.INFO)
            market_data = self._prepare_data_for_backtesting(market_data)
            
            # Preview prepared data for backtesting
            tprint_data_preview(market_data, "prepared_data_for_backtesting", max_rows=5)
            tprint_data_format(market_data, "prepared_data_format", level=LogLevel.DEBUG)

            # Create backtesting engine with validated hardware optimizations
            tprint_info("Creating backtesting engine", level=LogLevel.INFO)
            backtest_config = self._create_validated_backtest_config()
            tprint_data_format(backtest_config, "backtest_config", level=LogLevel.DEBUG)

            engine = SRBacktestingEngine(backtest_config)
            tprint_success("Backtesting engine created successfully", level=LogLevel.INFO)

            # Create sample SR levels for optimization with proper data splitting
            tprint_info("Splitting data for optimization", level=LogLevel.INFO)
            level_creation_data, backtest_data = self._split_data_for_optimization(market_data)
            
            # Preview split data for troubleshooting
            tprint_data_preview(level_creation_data, "level_creation_data", max_rows=5)
            tprint_data_preview(backtest_data, "backtest_data", max_rows=5)
            tprint_data_format(level_creation_data, "level_creation_data_format", level=LogLevel.DEBUG)
            tprint_data_format(backtest_data, "backtest_data_format", level=LogLevel.DEBUG)

            # Run parameter optimization
            tprint_info("Starting parameter optimization process", level=LogLevel.INFO)
            with tprint_timer("parameter_optimization", level=LogLevel.PERFORMANCE):
                optimization_result = await self._run_parameter_optimization(
                    engine, level_creation_data, backtest_data, param_config
            )

            # Extract results
            tprint_info("Extracting optimization results", level=LogLevel.INFO)
            optimized_parameters = optimization_result.get('optimized_parameters', {})
            quality_thresholds = optimization_result.get('quality_thresholds', {})
            parameter_optimization_metrics = optimization_result.get('parameter_optimization_metrics', {})
            
            tprint_data_format(optimization_result, "optimization_result", level=LogLevel.DEBUG)
            tprint_data_preview(optimized_parameters, "optimized_parameters", max_rows=5)

            # Validate that we have the required data
            if not optimized_parameters or not quality_thresholds:
                tprint_error("Parameter optimization failed to produce required data", level=LogLevel.ERROR)
                raise ValueError("Parameter optimization failed to produce required data")

            # Save optimization results using artifact manager
            tprint_info("Saving optimization results as artifact", level=LogLevel.INFO)
            artifact_data = {
                'optimized_parameters': optimized_parameters,
                'quality_thresholds': quality_thresholds,
                'parameter_optimization_metrics': parameter_optimization_metrics,
                'optimization_summary': {
                    'total_combinations_tested': optimization_result.get('total_combinations_tested', 0),
                    'best_score': optimization_result.get('best_score', 0.0),
                    'optimization_time': optimization_result.get('optimization_time', 0.0)
                },
                'metadata': {
                    'symbol': config.get('symbol', 'ETHUSDT'),
                    'exchange': config.get('exchange', 'binance'),
                    'timeframe': config.get('timeframe', '15m'),
                    'data_points': len(market_data) if market_data is not None else 0,
                    'execution_timestamp': datetime.now().isoformat()
                }
            }
            tprint_data_format(artifact_data, "artifact_data", level=LogLevel.DEBUG)
            self._save_artifact(artifact_data, 'sr_parameter_optimization_result')
            tprint_success("Optimization results saved as artifact", level=LogLevel.INFO)

            # Calculate metrics
            tprint_info("Calculating optimization metrics", level=LogLevel.INFO)
            metrics = self._calculate_optimization_metrics(optimization_result, start_time, config)
            tprint_data_format(metrics, "optimization_metrics", level=LogLevel.DEBUG)

            # Create outcome report
            tprint_info("Creating outcome report", level=LogLevel.INFO)
            outcome_report = self._create_outcome_report(optimization_result, metrics, config)
            tprint_data_format(outcome_report, "outcome_report", level=LogLevel.DEBUG)

            tprint(f"✅ SR parameter optimization completed", "SUCCESS")
            tprint_success("SR parameter optimization completed successfully", level=LogLevel.INFO)
            tprint_structured({
                "step": "sr_parameter_optimization_complete",
                "metrics": metrics,
                "optimization_summary": optimization_result.get('optimization_summary', {})
            }, level=LogLevel.INFO)

            return {
                'success': True,
                'optimization_results': optimization_result,
                'metrics': metrics,
                'outcome_report': outcome_report,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }

        except Exception as e:
            error_msg = f"SR parameter optimization failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            tprint_error(f"SR parameter optimization failed: {str(e)}", level=LogLevel.ERROR)
            tprint_structured({
                "step": "sr_parameter_optimization_error",
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }, level=LogLevel.ERROR)
            self.logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'optimization_results': {},
                'metrics': {},
                'processing_time': (datetime.now() - start_time).total_seconds()
            }

    async def _load_market_data(self, data: Any) -> Optional[Any]:
        """Load and prepare market data for optimization with memory optimization."""
        tprint_debug("Loading market data for optimization", level=LogLevel.DEBUG)
        if data is None:
            tprint_warning("No market data provided", level=LogLevel.WARNING)
            return None

        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            # Check data size and optimize memory usage
            data_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
            self.logger.info(f"Data size: {data_size_mb:.2f} MB")
            tprint_info(f"Data size: {data_size_mb:.2f} MB", level=LogLevel.INFO)
            
            # Preview loaded market data for troubleshooting
            tprint_data_preview(data, "loaded_market_data", max_rows=5)
            tprint_data_format(data, "loaded_market_data_format", level=LogLevel.DEBUG)

            # For large datasets, optimize memory usage
            if data_size_mb > 100:  # Large dataset (> 100MB)
                self.logger.info("Large dataset detected, optimizing memory usage")
                tprint_info("Large dataset detected, optimizing memory usage", level=LogLevel.INFO)

                # Convert float64 to float32 where possible to save memory
                numeric_columns = data.select_dtypes(include=[np.float64]).columns
                for col in numeric_columns:
                    if col in ['open', 'high', 'low', 'close']:  # Price columns
                        data[col] = data[col].astype(np.float32)

                # Convert int64 to int32 where possible
                int_columns = data.select_dtypes(include=[np.int64]).columns
                for col in int_columns:
                    if col == 'volume':
                        data[col] = data[col].astype(np.int32)

                optimized_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
                self.logger.info(f"Optimized data size: {optimized_size_mb:.2f} MB (saved {data_size_mb - optimized_size_mb:.2f} MB)")

            # Store reference to current data for configuration methods
            self._current_data = data
            return data

        # Handle other data types if needed
        return data

    def _prepare_data_for_backtesting(self, data: Any) -> Any:
        """Prepare data for backtesting with proper datetime indexing and robust error handling."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            self.logger.warning("Pandas not available or data is not a DataFrame, skipping data preparation")
            return data

        self.logger.info(f"Data index type before conversion: {type(data.index)}")
        self.logger.info(f"Data columns: {list(data.columns)}")
        self.logger.info(f"Data shape: {data.shape}")
        
        # Preview data before backtest preparation
        tprint_data_preview(data, "data_before_backtest_prep", max_rows=5, level="DEBUG")

        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.info("Converting data to datetime index for backtesting")

            # Try to find a suitable timestamp column
            timestamp_columns = ['timestamp', 'open_time', 'time', 'datetime', 'date']
            timestamp_col = None

            for col in timestamp_columns:
                if col in data.columns:
                    timestamp_col = col
                    break

            if timestamp_col:
                data = data.set_index(timestamp_col)
                self.logger.info(f"Using '{timestamp_col}' column as index")
            else:
                self.logger.warning("No suitable timestamp column found, using existing index")

            # Convert index to datetime with robust error handling
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    # Check if timestamps look like milliseconds (very large numbers)
                    sample_timestamps = data.index[:5]
                    self.logger.info(f"Sample timestamps before conversion: {sample_timestamps.tolist()}")

                    # Handle timezone info if present
                    if hasattr(data.index, 'tz') and data.index.tz is not None:
                        data.index = data.index.tz_convert('UTC').tz_localize(None)
                        self.logger.info("Converted timezone-aware index to UTC")

                    # Determine timestamp format and convert
                    max_timestamp = sample_timestamps.max()
                    min_timestamp = sample_timestamps.min()

                    if max_timestamp > 1e12:  # Likely microseconds
                        data.index = pd.to_datetime(data.index, unit='us', utc=False)
                        self.logger.info("Converted index to datetime (microseconds)")
                    elif max_timestamp > 1e10:  # Likely milliseconds
                        data.index = pd.to_datetime(data.index, unit='ms', utc=False)
                        self.logger.info("Converted index to datetime (milliseconds)")
                    elif max_timestamp > 1e9:  # Likely seconds
                        data.index = pd.to_datetime(data.index, unit='s', utc=False)
                        self.logger.info("Converted index to datetime (seconds)")
                    else:
                        # Try parsing as regular datetime string
                        data.index = pd.to_datetime(data.index, utc=False, errors='coerce')
                        self.logger.info("Converted index to datetime (string parsing)")

                        # Check for any NaT values (failed conversions)
                        if data.index.isna().any():
                            self.logger.warning(f"Found {data.index.isna().sum()} invalid timestamps after conversion")
                            # Drop rows with invalid timestamps
                            data = data.dropna()
                            self.logger.info(f"Dropped invalid timestamp rows, remaining: {len(data)}")

                except Exception as e:
                    self.logger.error(f"Failed to convert index to datetime: {e}")
                    # Create a simple integer-based index as fallback
                    data.index = pd.RangeIndex(start=0, stop=len(data))
                    self.logger.warning("Created fallback integer index")

        # Validate the final datetime index
        if isinstance(data.index, pd.DatetimeIndex):
            # Check for reasonable date range (not too far in past/future)
            min_date = data.index.min()
            max_date = data.index.max()
            current_date = pd.Timestamp.now()

            if min_date < current_date - pd.Timedelta(days=365*10):  # More than 10 years ago
                self.logger.warning(f"Unusual early date in data: {min_date}")
            if max_date > current_date + pd.Timedelta(days=365):  # More than 1 year in future
                self.logger.warning(f"Unusual future date in data: {max_date}")

            # Check for duplicate timestamps
            if data.index.duplicated().any():
                duplicate_count = data.index.duplicated().sum()
                self.logger.warning(f"Found {duplicate_count} duplicate timestamps")
                # Keep first occurrence of duplicates
                data = data[~data.index.duplicated(keep='first')]
                self.logger.info(f"Removed duplicate timestamps, remaining: {len(data)}")

        self.logger.info(f"Final data index type: {type(data.index)}")
        self.logger.info(f"Data index sample: {data.index[:3] if len(data) > 0 else 'empty'}")
        
        # Preview data after backtest preparation
        tprint_data_preview(data, "data_after_backtest_prep", max_rows=5, level="DEBUG")

        return data

    async def _run_parameter_optimization(
        self,
        engine: Any,
        level_creation_data: pd.DataFrame,
        backtest_data: pd.DataFrame,
        param_config: Any
    ) -> Dict[str, Any]:
        """Run the actual parameter optimization process with robust error handling."""
        try:
            # Import the function here to ensure it's available

            # Get parameter optimization engine
            optimization_engine = get_parameter_optimization_engine(param_config)

            # Check if the optimization method is async
            import inspect
            optimize_method = getattr(optimization_engine, 'optimize_parameters', None)
            if optimize_method is None:
                raise AttributeError("optimize_parameters method not found on optimization engine")

            is_async = inspect.iscoroutinefunction(optimize_method)

            # Run optimization with proper async handling
            if is_async:
                optimization_result = await optimization_engine.optimize_parameters(
                    engine=engine,
                    level_creation_data=level_creation_data,
                    backtest_data=backtest_data,
                    config=param_config
                )
            else:
                # Run in thread pool for non-async methods
                loop = asyncio.get_event_loop()
                optimization_result = await loop.run_in_executor(
                    None,
                    lambda: optimization_engine.optimize_parameters(
                        engine=engine,
                        level_creation_data=level_creation_data,
                        backtest_data=backtest_data,
                        config=param_config
                    )
                )

            # Validate optimization result
            if not isinstance(optimization_result, dict):
                raise ValueError(f"Invalid optimization result type: {type(optimization_result)}")

            required_keys = ['optimized_parameters', 'quality_thresholds']
            missing_keys = [key for key in required_keys if key not in optimization_result]
            if missing_keys:
                raise ValueError(f"Missing required keys in optimization result: {missing_keys}")

            self.logger.info("Parameter optimization completed successfully")
            return optimization_result

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            self.logger.error(f"Parameter optimization failed: {error_type}: {error_msg}")

            # Return fallback parameters with more context
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
                    'error_type': error_type,
                    'error': error_msg,
                    'fallback_reason': 'Optimization engine failure'
                }
            }

    def _validate_market_data(self, data: Any) -> bool:
        """
        Validate market data for optimization requirements.

        Args:
            data: Market data to validate

        Returns:
            True if data is valid, False otherwise
        """
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

    def _create_validated_param_config(self):
        """Create parameter optimization config with hardware capability validation."""
        if not SR_CLUSTERING_AVAILABLE or ParameterOptimizationConfig is None:
            raise RuntimeError("ParameterOptimizationConfig not available")

        # Check GPU availability
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
        except ImportError:
            pass

        # Determine optimal memory settings based on available memory
        memory_limit_gb = 4.0  # Conservative default
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            memory_limit_gb = min(available_memory_gb * 0.5, 8.0)  # Use max 50% of available memory
        except ImportError:
            pass

        return ParameterOptimizationConfig(
            optimization_method='adaptive_grid_search',
            min_samples_for_optimization=10,
            adaptive_optimization=True,
            objective_metric='composite',

            # Hardware optimization settings with validation
            enable_hardware_optimization=True,
            enable_parallel_processing=True,
            max_parallel_workers=None,  # Auto-detect
            enable_gpu_acceleration=gpu_available,
            memory_limit_gb=memory_limit_gb,
            chunk_size=min(1000, max(100, int(len(self._get_current_data()) / 10) if hasattr(self, '_current_data') and self._current_data is not None else 100))
        )

    def _create_validated_backtest_config(self):
        """Create backtesting config with hardware capability validation."""
        if not SR_CLUSTERING_AVAILABLE or BacktestConfig is None:
            raise RuntimeError("BacktestConfig not available")

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

            # Hardware optimization settings with validation
            enable_m1_optimizations=True,
            enable_gpu_acceleration=gpu_available,
            enable_memory_optimization=True,
            memory_limit_gb=memory_limit_gb,
            chunk_size=min(1000, max(100, int(len(self._get_current_data()) / 10) if hasattr(self, '_current_data') and self._current_data is not None else 100)),

            # Computation optimization settings
            enable_parallel_processing=True,
            enable_vectorized_operations=True,
            enable_caching=True,
            cache_size_mb=min(100, max(10, int(memory_limit_gb * 10))),
            enable_numba_acceleration=True
        )

    def _split_data_for_optimization(self, market_data: Any) -> Tuple[Any, Any]:
        """
        Split data properly to avoid data leakage during optimization.

        Args:
            market_data: Full market dataset

        Returns:
            Tuple of (training_data, testing_data)
        """
        if not PANDAS_AVAILABLE or not isinstance(market_data, pd.DataFrame):
            self.logger.warning("Non-DataFrame data provided, using same data for training and testing")
            return market_data, market_data

        # Use 70% for training (level creation) and 30% for testing (backtesting)
        # This prevents data leakage by ensuring backtesting data is never seen during optimization
        split_point = int(len(market_data) * 0.7)

        if split_point < 100:
            self.logger.warning("Dataset too small for proper splitting, using same data")
            return market_data, market_data

        level_creation_data = market_data.iloc[:split_point]
        backtest_data = market_data.iloc[split_point:]

        self.logger.info(f"Data split: {len(level_creation_data)} rows for training, {len(backtest_data)} rows for testing")
        
        # Preview split data for troubleshooting
        tprint_data_preview(level_creation_data, "training_data_split", max_rows=5, level="DEBUG")
        tprint_data_preview(backtest_data, "testing_data_split", max_rows=5, level="DEBUG")
        
        return level_creation_data, backtest_data

    def _get_current_data(self):
        """Get current data reference for configuration methods."""
        return getattr(self, '_current_data', None)

    def _load_market_data(self, config: Dict[str, Any]) -> Optional[Any]:
        """Load market data from artifacts or config."""
        try:
            # Try to load from artifacts first
            market_data = self._load_dataframe('market_data')
            if market_data is not None:
                return market_data
            
            # Try alternative artifact names
            market_data = self._load_dataframe('processed_data') or self._load_dataframe('data')
            if market_data is not None:
                return market_data
            
            # Try to load from config
            if 'market_data' in config:
                return pd.DataFrame(config['market_data'])
            
            return None
            
        except Exception as e:
            tprint(f"⚠️ Failed to load market data: {e}", "WARNING")
            return None

    def _load_sr_levels(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load SR levels from artifacts or config."""
        try:
            # Try to load from artifacts first
            sr_levels = self._get_artifact('sr_levels')
            if sr_levels is not None:
                return sr_levels
            
            # Try alternative artifact names
            sr_levels = self._get_artifact('support_resistance_levels') or self._get_artifact('sr_data')
            if sr_levels is not None:
                return sr_levels
            
            # Try to load from config
            if 'sr_levels' in config:
                return config['sr_levels']
            
            return None
            
        except Exception as e:
            tprint(f"⚠️ Failed to load SR levels: {e}", "WARNING")
            return None

    def _get_parameter_grid(self, config: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Get parameter grid for optimization."""
        default_grid = {
            'min_touches': [2, 3, 4, 5],
            'tolerance_pct': [0.3, 0.5, 0.7, 1.0],
            'lookback_periods': [50, 100, 150, 200],
            'min_strength': [0.1, 0.2, 0.3, 0.4],
            'max_levels': [10, 20, 30, 50]
        }
        
        # Use config parameter grid if provided
        if 'parameter_grid' in config:
            return config['parameter_grid']
        
        return default_grid

    def _calculate_optimization_metrics(self, results: Dict[str, Any], start_time: datetime, config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimization metrics."""
        try:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            metrics = {
                'processing_time_seconds': processing_time,
                'best_score': results.get('best_score', 0.0),
                'total_combinations_tested': results.get('total_combinations_tested', 0),
                'optimization_time': results.get('optimization_time', 0.0),
                'success': True
            }
            
            return metrics
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate metrics: {e}", "WARNING")
            return {'success': False, 'error': str(e)}

    def _create_outcome_report(self, results: Dict[str, Any], metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create outcome report markdown."""
        try:
            report = f"""# SR Parameter Optimization Outcome Report

## Execution Summary
- **Symbol**: {config.get('symbol', 'UNKNOWN')}
- **Exchange**: {config.get('exchange', 'UNKNOWN')}
- **Timeframe**: {config.get('timeframe', 'UNKNOWN')}
- **Processing Time**: {metrics.get('processing_time_seconds', 0):.2f} seconds
- **Success**: {'✅ Yes' if metrics.get('success', False) else '❌ No'}

## Optimization Results
- **Best Score**: {metrics.get('best_score', 0):.3f}
- **Combinations Tested**: {metrics.get('total_combinations_tested', 0)}
- **Optimization Time**: {metrics.get('optimization_time', 0):.2f} seconds

## Best Parameters
"""
            
            optimized_params = results.get('optimized_parameters', {})
            for param, value in optimized_params.items():
                report += f"- **{param}**: {value}\n"
            
            report += f"""
## Quality Thresholds
"""
            
            quality_thresholds = results.get('quality_thresholds', {})
            for threshold, value in quality_thresholds.items():
                report += f"- **{threshold}**: {value}\n"
            
            report += f"""
## Generated Artifacts
- Optimization results
- Best parameters
- Quality thresholds
- Optimization metadata

---
*Generated by SR Parameter Optimization Step at {datetime.now().isoformat()}*
"""
            
            return report
            
        except Exception as e:
            tprint(f"⚠️ Failed to create outcome report: {e}", "WARNING")
            return f"# SR Parameter Optimization Outcome Report\n\nError creating report: {str(e)}"


# Register the step
def register_sr_parameter_optimization_step():
    """Register the SR parameter optimization step."""
    from src.training.steps.base_step import step_registry
    
    step_registry.register("sr_parameter_optimization", SRParameterOptimizationStep)
    tprint("✅ SR parameter optimization step registered", "SUCCESS")


# Auto-register when module is imported
register_sr_parameter_optimization_step()
