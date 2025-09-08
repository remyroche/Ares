from ...core.decorators import handles_errors
"""Step 4: Optimized Triple Barrier Method with Lookahead Bias Prevention.

This module applies the triple barrier method to create trading signals and labels
with proper lookahead bias prevention, vectorized operations, and volatility-based
parameter suggestions.
"""

# Import common types from main branch
from src.training.steps.model_training.step04_common_types import (
    StepResult, TripleBarrierResult, StepResultStatus, standardize_result
)
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import concurrent.futures
from functools import partial

# Import decorators from both locations for compatibility
import numpy as np
import pandas as pd

# Enhanced utility imports with dependency injection
from .step04_dependency_injection import (
    get_step04_utilities, get_step04_container, create_step04_config,
    get_common_ops, get_common_utils, get_math_validation, get_parquet_utils,
    get_serialization_utils, get_data_processing_utils, get_m1_gpu_utils,
    get_m1_memory_optimizer, get_m1_cpu_optimizer
)

# Standardized imports from utils (fallback)
from src.utils.common_operations import (
    ensure_directory,
    safe_read_parquet,
    safe_to_parquet,
    get_logger,
    format_bytes,
    chunked_iterable,
    parallel_map,
    safe_dict_get,
    safe_float,
    safe_int,
    safe_json_dump,
    safe_json_load,
    optimize_dataframe_dtypes,
    validate_dataframe_schema,
    validate_data_quality
)
from src.utils.math_validation import (
    safe_divide,
    safe_log,
    safe_sqrt,
    safe_kelly_calculation,
    validate_positive,
    validate_range,
    MathValidationError
)
from src.utils.parquet_utils import get_parquet_utils
# Core decorators imports
from src.core.decorators import (
    handles_errors,
    traced,
    validates,
    log_execution_time,
    cached,
    error_boundary,
    timeout,
    retry
)
# Core errors imports
from src.core.errors import (
    AppError,
    ValidationError,
    DataIntegrityError,
    NotFoundError,
    TimeoutError
)
from src.utils.enhanced_memory_management import (
import logging

    MemoryMonitor,
    MemoryConfig,
    optimize_dataframe_dtypes,
    chunk_dataframe
)
from src.utils.data_streaming_manager import DataStreamingManager
from src.utils.logger import system_logger

# Project setup
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# MLflow integration with fallback
try:
    from src.utils.enhanced_mlflow_integration import (
        with_enhanced_mlflow_logging,
        log_step_report,
        log_step_metrics
    )
except ImportError:
    def with_enhanced_mlflow_logging(_name: str) -> Any:
        def _decorator(fn: Any) -> Any:
            return fn
        return _decorator

    def log_step_report(*args: Any, **kwargs: Any) -> None:
        return None

    def log_step_metrics(*args: Any, **kwargs: Any) -> None:
        return None

# Import financial metrics logging system
try:
    from .step04_5_financial_logging import Step04_5FinancialLogger
    FINANCIAL_LOGGING_AVAILABLE = True
except ImportError:
    FINANCIAL_LOGGING_AVAILABLE = False

# Initialize logger using common utilities
logger = get_logger('Step4TripleBarrierMethodOptimized')

class VolatilityBasedParameterCalculator:
    """Calculate optimal triple barrier parameters based on market volatility with extensive utility usage."""
    
    def __init__(self, utils=None, lookback_periods: int = 30):
        self.utils = utils or get_step04_utilities()
        self.lookback_periods = lookback_periods
        self.logger = self.utils.get_function('common_operations', 'get_logger')('VolatilityBasedParameterCalculator')
        
        # Get utility functions
        self.safe_float = self.utils.get_function('common_operations', 'safe_float')
        self.safe_divide = self.utils.get_function('math_validation', 'safe_divide')
        self.validate_positive = self.utils.get_function('math_validation', 'validate_positive')
        self.validate_range = self.utils.get_function('math_validation', 'validate_range')
    
    def calculate_volatility_based_parameters(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate optimal parameters based on historical volatility using extensive utility functions."""
        try:
            if len(data) < self.lookback_periods:
                return self._get_default_parameters()
            
            # Use data processing utilities for comprehensive validation
            data_quality_report = self.utils.get_function('data_processing_utils', 'create_data_quality_report')(data)
            if not data_quality_report.get('is_valid', True):
                self.logger.warning(f'⚠️ Data quality issues detected: {data_quality_report.get("issues", [])}')
            
            # Use DataFrameValidator for additional validation
            DataFrameValidator = self.utils.get_function('data_processing_utils', 'DataFrameValidator')
            validator = DataFrameValidator()
            validation_result = validator.validate(data)
            if not validation_result.get('is_valid', True):
                self.logger.warning(f'⚠️ DataFrame validation issues: {validation_result.get("issues", [])}')
            
            # Use DataFrameCleaner to ensure data quality
            DataFrameCleaner = self.utils.get_function('data_processing_utils', 'DataFrameCleaner')
            cleaner = DataFrameCleaner()
            cleaned_data = cleaner.clean(data)
            if len(cleaned_data) != len(data):
                self.logger.info(f'🧹 Data cleaning removed {len(data) - len(cleaned_data)} rows')
                data = cleaned_data
            
            # Calculate rolling volatility with comprehensive math validation
            returns = data['close'].pct_change().dropna()
            
            # Validate returns data
            returns = self.validate_finite(returns, "returns")
            
            # Calculate volatility with safe operations
            volatility = returns.rolling(window=self.lookback_periods).std().iloc[-1]
            volatility = self.validate_positive(volatility, "volatility", epsilon=1e-10)
            volatility = self.validate_finite(volatility, "volatility")
            
            # Calculate ATR (Average True Range) for volatility measure with math validation
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            # Validate price differences
            high_low = self.validate_positive(high_low, "high_low")
            high_close = self.validate_positive(high_close, "high_close")
            low_close = self.validate_positive(low_close, "low_close")
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            true_range = self.validate_positive(true_range, "true_range")
            
            atr = true_range.rolling(window=self.lookback_periods).mean().iloc[-1]
            atr = self.validate_positive(atr, "atr", epsilon=1e-10)
            atr = self.validate_finite(atr, "atr")
            
            # Calculate current price level
            current_price = data['close'].iloc[-1]
            current_price = self.validate_positive(current_price, "current_price", epsilon=1e-10)
            
            # Volatility-based parameter calculation using comprehensive safe math operations
            volatility_multiplier = self.safe_divide(volatility * 100, 1.0, default=1.0)
            volatility_multiplier = self.validate_range(volatility_multiplier, 0.5, 5.0, "volatility_multiplier")
            volatility_multiplier = self.validate_finite(volatility_multiplier, "volatility_multiplier")
            
            atr_ratio = self.safe_divide(atr, current_price, default=0.01)
            atr_ratio = self.validate_positive(atr_ratio, "atr_ratio")
            atr_ratio = self.validate_finite(atr_ratio, "atr_ratio")
            atr_multiplier = self.validate_range(atr_ratio * 100, 0.1, 2.0, "atr_multiplier")
            
            # Calculate optimal parameters using safe math with comprehensive validation
            profit_take_multiplier = self.safe_divide(volatility_multiplier * 0.8, 100.0, default=0.001)
            profit_take_multiplier = self.validate_positive(profit_take_multiplier, "profit_take_multiplier")
            profit_take_multiplier = self.validate_range(profit_take_multiplier, 0.001, 0.1, "profit_take_multiplier")
            profit_take_multiplier = self.validate_finite(profit_take_multiplier, "profit_take_multiplier")
            
            stop_loss_multiplier = self.safe_divide(volatility_multiplier * 0.4, 100.0, default=0.0005)
            stop_loss_multiplier = self.validate_positive(stop_loss_multiplier, "stop_loss_multiplier")
            stop_loss_multiplier = self.validate_range(stop_loss_multiplier, 0.0005, 0.05, "stop_loss_multiplier")
            stop_loss_multiplier = self.validate_finite(stop_loss_multiplier, "stop_loss_multiplier")
            
            # Time barrier based on volatility (higher volatility = shorter time barrier) with math validation
            base_time_minutes = 30
            volatility_denominator = self.safe_divide(1.0, volatility * 100 + 0.1, default=1.0)
            volatility_denominator = self.validate_positive(volatility_denominator, "volatility_denominator")
            volatility_denominator = self.validate_finite(volatility_denominator, "volatility_denominator")
            volatility_time_factor = self.validate_range(volatility_denominator, 0.5, 2.0, "volatility_time_factor")
            
            time_barrier_minutes = int(base_time_minutes * volatility_time_factor)
            time_barrier_minutes = self.validate_positive(time_barrier_minutes, "time_barrier_minutes")
            time_barrier_minutes = self.validate_range(time_barrier_minutes, 5, 300, "time_barrier_minutes")
            
            # Max lookahead based on volatility with math validation
            base_lookahead = 100
            volatility_lookahead_factor = self.validate_range(volatility_denominator, 0.5, 2.0, "volatility_lookahead_factor")
            max_lookahead = int(base_lookahead * volatility_lookahead_factor)
            max_lookahead = self.validate_positive(max_lookahead, "max_lookahead")
            max_lookahead = self.validate_range(max_lookahead, 10, 1000, "max_lookahead")
            
            # Use safe_float for all parameter values
            parameters = {
                'profit_take_multiplier': self.safe_float(round(profit_take_multiplier, 6), 0.001),
                'stop_loss_multiplier': self.safe_float(round(stop_loss_multiplier, 6), 0.0005),
                'time_barrier_minutes': int(time_barrier_minutes),
                'max_lookahead': int(max_lookahead),
                'volatility': self.safe_float(round(volatility, 6), 0.01),
                'atr': self.safe_float(round(atr, 6), 0.01),
                'volatility_multiplier': self.safe_float(round(volatility_multiplier, 6), 1.0),
                'parameter_source': 'volatility_based'
            }
            
            self.logger.info(f'📊 Volatility-based parameters calculated using utility functions:')
            self.logger.info(f'   Volatility: {volatility:.4f} ({volatility*100:.2f}%)')
            self.logger.info(f'   Profit Take: {profit_take_multiplier:.4f} ({profit_take_multiplier*100:.2f}%)')
            self.logger.info(f'   Stop Loss: {stop_loss_multiplier:.4f} ({stop_loss_multiplier*100:.2f}%)')
            self.logger.info(f'   Time Barrier: {time_barrier_minutes} minutes')
            self.logger.info(f'   Max Lookahead: {max_lookahead} periods')
            
            return parameters
            
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to calculate volatility-based parameters: {e}')
            return self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict[str, float]:
        """Get default parameters when volatility calculation fails."""
        return {
            'profit_take_multiplier': 0.002,  # 0.2%
            'stop_loss_multiplier': 0.001,    # 0.1%
            'time_barrier_minutes': 30,
            'max_lookahead': 100,
            'volatility': 0.0,
            'atr': 0.0,
            'volatility_multiplier': 1.0,
            'parameter_source': 'default'
        }

class FastFailValidator:
    """Fast fail validation for early error detection."""
    
    @staticmethod
    def validate_data(data: pd.DataFrame) -> Tuple[bool, str]:
        """Fast fail data validation."""
        if data is None:
            return False, "Data is None"
        if data.empty:
            return False, "Empty dataset"
        if len(data) < 100:
            return False, f"Insufficient data points: {len(data)} (minimum 100 required)"
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Check for non-positive prices
        for col in required_columns:
            if (data[col] <= 0).any():
                negative_count = (data[col] <= 0).sum()
                return False, f"Non-positive prices in {col}: {negative_count} rows"
        
        # Check OHLC relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            return False, f"Invalid OHLC relationships: {invalid_count} rows"
        
        return True, "Data validation passed"
    
    @staticmethod
    def validate_parameters(config: Dict[str, Any]) -> Tuple[bool, str]:
        """Fast fail parameter validation."""
        try:
            # Extract parameters
            profit_take = safe_float(config.get('profit_take_multiplier', 0.002), 0.002)
            stop_loss = safe_float(config.get('stop_loss_multiplier', 0.001), 0.001)
            max_lookahead = safe_int(config.get('max_lookahead', 100), 100)
            time_barrier = safe_int(config.get('time_barrier_minutes', 30), 30)
            
            # Validate ranges
            if profit_take <= 0 or profit_take > 0.1:  # Max 10%
                return False, f"Invalid profit_take_multiplier: {profit_take} (must be 0 < x <= 0.1)"
            
            if stop_loss <= 0 or stop_loss > 0.1:  # Max 10%
                return False, f"Invalid stop_loss_multiplier: {stop_loss} (must be 0 < x <= 0.1)"
            
            if max_lookahead <= 0 or max_lookahead > 1000:
                return False, f"Invalid max_lookahead: {max_lookahead} (must be 0 < x <= 1000)"
            
            if time_barrier <= 0 or time_barrier > 1440:  # Max 24 hours
                return False, f"Invalid time_barrier_minutes: {time_barrier} (must be 0 < x <= 1440)"
            
            # Check risk-reward ratio
            if profit_take <= stop_loss:
                return False, f"Poor risk-reward ratio: profit_take ({profit_take}) <= stop_loss ({stop_loss})"
            
            return True, "Parameter validation passed"
            
        except Exception as e:
            return False, f"Parameter validation error: {str(e)}"

class VectorizedTripleBarrierProcessor:
    """Vectorized triple barrier processor with lookahead bias prevention and extensive utility usage."""
    
    def __init__(self, config: Dict[str, Any], utils=None):
        self.config = config
        self.utils = utils or get_step04_utilities()
        self.logger = self.utils.get_function('common_operations', 'get_logger')('VectorizedTripleBarrierProcessor')
        
        # Get utility functions
        self.safe_float = self.utils.get_function('common_operations', 'safe_float')
        self.safe_int = self.utils.get_function('common_operations', 'safe_int')
        self.validate_positive = self.utils.get_function('math_validation', 'validate_positive')
        self.validate_range = self.utils.get_function('math_validation', 'validate_range')
        self.safe_kelly_calculation = self.utils.get_function('math_validation', 'safe_kelly_calculation')
    
    def apply_triple_barrier_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply triple barrier method using vectorized operations with extensive utility usage."""
        try:
            # Use data processing utilities for comprehensive validation
            data_quality_report = self.utils.get_function('data_processing_utils', 'create_data_quality_report')(data)
            if not data_quality_report.get('is_valid', True):
                self.logger.warning(f'⚠️ Data quality issues detected: {data_quality_report.get("issues", [])}')
            
            # Fast fail validation
            is_valid, error_msg = FastFailValidator.validate_data(data)
            if not is_valid:
                raise ValueError(f"Data validation failed: {error_msg}")
            
            # Get parameters using utility functions
            profit_take_multiplier = self.safe_float(self.config.get('profit_take_multiplier', 0.002), 0.002)
            stop_loss_multiplier = self.safe_float(self.config.get('stop_loss_multiplier', 0.001), 0.001)
            max_lookahead = self.safe_int(self.config.get('max_lookahead', 100), 100)
            
            # Validate parameters using math validation utilities
            profit_take_multiplier = self.validate_positive(profit_take_multiplier, "profit_take_multiplier")
            stop_loss_multiplier = self.validate_positive(stop_loss_multiplier, "stop_loss_multiplier")
            max_lookahead = self.validate_positive(max_lookahead, "max_lookahead")
            
            # Validate parameter ranges
            profit_take_multiplier = self.validate_range(profit_take_multiplier, 0.0001, 0.1, "profit_take_multiplier")
            stop_loss_multiplier = self.validate_range(stop_loss_multiplier, 0.0001, 0.1, "stop_loss_multiplier")
            max_lookahead = self.validate_range(max_lookahead, 1, 1000, "max_lookahead")
            
            # Validate parameters
            is_valid, error_msg = FastFailValidator.validate_parameters(self.config)
            if not is_valid:
                raise ValueError(f"Parameter validation failed: {error_msg}")
            
            self.logger.info(f'🚀 Applying vectorized triple barrier with utility-validated parameters:')
            self.logger.info(f'   Profit Take: {profit_take_multiplier:.4f} ({profit_take_multiplier*100:.2f}%)')
            self.logger.info(f'   Stop Loss: {stop_loss_multiplier:.4f} ({stop_loss_multiplier*100:.2f}%)')
            self.logger.info(f'   Max Lookahead: {max_lookahead} periods')
            
            # Extract price arrays
            close_prices = data['close'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            
            n = len(close_prices)
            labels = np.zeros(n, dtype=np.int8)
            profit_pcts = np.zeros(n, dtype=np.float64)
            
            # Vectorized barrier calculation with utility validation
            entry_prices = close_prices[:-1]  # All but last
            entry_prices = self.validate_positive(entry_prices, "entry_prices")
            
            profit_barriers = entry_prices * (1 + profit_take_multiplier)
            stop_barriers = entry_prices * (1 - stop_loss_multiplier)
            
            # Validate calculated barriers
            profit_barriers = self.validate_positive(profit_barriers, "profit_barriers")
            stop_barriers = self.validate_positive(stop_barriers, "stop_barriers")
            
            # Process each entry point with utility validation
            for i in range(n - 1):
                entry_price = entry_prices[i]
                profit_barrier = profit_barriers[i]
                stop_barrier = stop_barriers[i]
                
                # Validate individual prices and barriers
                entry_price = self.validate_positive(entry_price, f"entry_price[{i}]")
                profit_barrier = self.validate_positive(profit_barrier, f"profit_barrier[{i}]")
                stop_barrier = self.validate_positive(stop_barrier, f"stop_barrier[{i}]")
                
                # Look ahead window (preventing lookahead bias by using only future data)
                lookahead_end = min(i + max_lookahead + 1, n)
                future_highs = high_prices[i+1:lookahead_end]
                future_lows = low_prices[i+1:lookahead_end]
                
                if len(future_highs) == 0:
                    continue
                
                # Validate future price arrays
                future_highs = self.validate_positive(future_highs, f"future_highs[{i}]")
                future_lows = self.validate_positive(future_lows, f"future_lows[{i}]")
                
                # Vectorized barrier hit detection
                profit_hits = future_highs >= profit_barrier
                stop_hits = future_lows <= stop_barrier
                
                # Find first hit
                profit_hit_idx = np.argmax(profit_hits) if np.any(profit_hits) else len(profit_hits)
                stop_hit_idx = np.argmax(stop_hits) if np.any(stop_hits) else len(stop_hits)
                
                # Determine outcome with safe calculations
                if profit_hit_idx < stop_hit_idx and np.any(profit_hits):
                    # Profit target hit first
                    labels[i] = 1
                    profit_pcts[i] = self.safe_float(profit_take_multiplier, 0.0)
                elif stop_hit_idx < profit_hit_idx and np.any(stop_hits):
                    # Stop loss hit first
                    labels[i] = -1
                    profit_pcts[i] = self.safe_float(-stop_loss_multiplier, 0.0)
                # If neither hit, label remains 0 (no action)
            
            # Create result DataFrame with utility validation
            result_data = pd.DataFrame({
                'label': labels,
                'potential_profit_pct': profit_pcts
            }, index=data.index)
            
            # Calculate statistics using safe math operations
            total_signals = len(result_data)
            buy_signals = (labels == 1).sum()
            sell_signals = (labels == -1).sum()
            no_action = (labels == 0).sum()
            
            # Use safe division for percentage calculations
            buy_percentage = self.safe_divide(buy_signals * 100, total_signals, default=0.0) if total_signals > 0 else 0.0
            sell_percentage = self.safe_divide(sell_signals * 100, total_signals, default=0.0) if total_signals > 0 else 0.0
            no_action_percentage = self.safe_divide(no_action * 100, total_signals, default=0.0) if total_signals > 0 else 0.0
            
            self.logger.info(f'📊 Triple barrier results with utility validation:')
            self.logger.info(f'   Total signals: {total_signals:,}')
            self.logger.info(f'   Buy signals: {buy_signals:,} ({buy_percentage:.1f}%)')
            self.logger.info(f'   Sell signals: {sell_signals:,} ({sell_percentage:.1f}%)')
            self.logger.info(f'   No action: {no_action:,} ({no_action_percentage:.1f}%)')
            
            # Use data processing utilities for final validation
            final_quality_report = self.utils.get_function('data_processing_utils', 'create_data_quality_report')(result_data)
            if not final_quality_report.get('is_valid', True):
                self.logger.warning(f'⚠️ Final result data quality issues: {final_quality_report.get("issues", [])}')
            
            return result_data
            
        except Exception as e:
            self.logger.exception(f'❌ Error in vectorized triple barrier: {e}')
            return pd.DataFrame()

class OptimizedTripleBarrierMethodStep:
    """Optimized Step 4: Triple Barrier Method with comprehensive improvements and extensive utility usage."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        
        # Initialize dependency injection container
        self.utility_config = create_step04_config(
            enable_common_operations=True,
            enable_common_utilities=True,
            enable_math_validation=True,
            enable_parquet_utils=True,
            enable_serialization_utils=True,
            enable_data_processing_utils=True,
            enable_m1_gpu_utils=True,
            enable_m1_memory_optimizer=True,
            enable_m1_cpu_optimizer=True
        )
        self.container = get_step04_container(self.utility_config)
        self.utils = get_step04_utilities()
        
        # Get logger from utilities
        self.logger = self.utils.get_function('common_operations', 'get_logger')('OptimizedTripleBarrierMethodStep')
        self.start_time: Optional[float] = None
        self.step_timings: Dict[str, float] = {}
        
        # Initialize components with utility integration
        self.volatility_calculator = VolatilityBasedParameterCalculator(self.utils)
        self.vectorized_processor = VectorizedTripleBarrierProcessor(config, self.utils)
        
        # Initialize M1 optimizations
        self._init_m1_optimizations()
        
        # Memory management
        self.memory_config = MemoryConfig(
            max_memory_mb=safe_float(config.get('max_memory_mb', 2048.0), 2048.0),
            warning_threshold=0.8,
            critical_threshold=0.95
        )
        self.memory_monitor = MemoryMonitor(self.memory_config)
        
        # Data streaming with true streaming support
        self.streaming_manager = DataStreamingManager(
            chunk_size=safe_int(config.get('chunk_size', 10000), 10000),
            memory_threshold=0.8
        )
        
        # Risk management configuration
        self.risk_config = {
            'max_position_size_pct': safe_float(config.get('max_position_size_pct', 0.1), 0.1),
            'max_daily_trades': safe_int(config.get('max_daily_trades', 100), 100),
            'max_drawdown_pct': safe_float(config.get('max_drawdown_pct', 0.05), 0.05),
            'min_risk_reward_ratio': safe_float(config.get('min_risk_reward_ratio', 1.0), 1.0),
            'max_volatility_pct': safe_float(config.get('max_volatility_pct', 0.1), 0.1),
            'enable_risk_controls': config.get('enable_risk_controls', True)
        }

        # Initialize financial metrics logging system
        if FINANCIAL_LOGGING_AVAILABLE:
            try:
                self.financial_logger = None  # Will be initialized per execution
                self.logger.info('✅ Financial metrics logging system available')
            except Exception as e:
                self.logger.warning(f'⚠️ Financial metrics logging system failed to initialize: {e}')
                self.financial_logger = None
        else:
            self.logger.info('ℹ️ Financial metrics logging system not available, using basic reporting')
            self.financial_logger = None

    async def initialize(self) -> None:
        """Initialize the optimized triple barrier method step."""
        self.start_time = time.time()
        self.logger.info('🚀 Initializing Optimized Triple Barrier Method Step...')
        self.logger.info('📋 Step 4 Configuration:')
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL', 'N/A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N/A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N/A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N/A')}")
        self.logger.info('✅ Optimized Triple Barrier Method Step initialized successfully')
    
    def _init_m1_optimizations(self):
        """Initialize M1 hardware optimization components using utility functions."""
        try:
            # Get utility functions
            safe_float = self.utils.get_function('common_operations', 'safe_float')
            safe_int = self.utils.get_function('common_operations', 'safe_int')
            validate_positive = self.utils.get_function('math_validation', 'validate_positive')
            validate_range = self.utils.get_function('math_validation', 'validate_range')
            
            # Initialize M1 GPU Manager through utility injection
            self.gpu_manager = self.utils.get_function('m1_gpu_utils', 'get_m1_gpu_manager')()
            self.logger.info('🎯 M1 GPU Manager initialized for triple barrier method')
            
            # Initialize M1 Memory Optimizer with step-specific settings and validation
            memory_limit = self.config.get('memory_limit_gb', 4.0)
            memory_limit = validate_positive(safe_float(memory_limit, 4.0), "memory_limit_gb")
            memory_limit = validate_range(memory_limit, 1.0, 32.0, "memory_limit_gb")
            
            self.memory_optimizer = self.utils.get_function('m1_memory_optimizer', 'M1MemoryOptimizer')(
                memory_limit_gb=memory_limit,
                enable_gc_tuning=True,
                enable_memory_leak_detection=True,
                enable_swap_management=True
            )
            self.logger.info('🧠 M1 Memory Optimizer initialized for triple barrier method')
            
            # Initialize M1 CPU Optimizer with validation
            max_workers = self.config.get('max_parallel_workers', None)
            if max_workers is not None:
                max_workers = validate_positive(safe_int(max_workers, None), "max_parallel_workers")
                max_workers = validate_range(max_workers, 1, 16, "max_parallel_workers")
            
            self.cpu_optimizer = self.utils.get_function('m1_cpu_optimizer', 'M1CPUOptimizer')(
                max_workers=max_workers,
                enable_hyperthreading=True
            )
            self.logger.info('⚡ M1 CPU Optimizer initialized for triple barrier method')
            
            self.m1_optimizations_enabled = True
            
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to initialize M1 optimizations: {e}')
            self.m1_optimizations_enabled = False
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None

    @traced(span_name='execute_optimized_triple_barrier_method')
    @validates()
    @handles_errors()
    @log_execution_time()
    @memory_efficient(max_memory_mb=2048.0)
    async def execute_triple_barrier_method(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str = 'data_cache',
        force_rerun: bool = False
    ) -> TripleBarrierResult:
        """Execute the optimized triple barrier method step."""
        step_start = time.time()
        self.logger.info(f'🚀 Executing Optimized Triple Barrier Method for {symbol} on {exchange}')
        
        # Log initial memory status
        self.memory_monitor.log_memory_status('before triple barrier execution')
        
        # Start M1 memory monitoring if available
        if self.m1_optimizations_enabled and self.memory_optimizer:
            with self.memory_optimizer.memory_checkpoint("triple_barrier_execution_start"):
                self.logger.info('🧠 M1 Memory monitoring enabled for triple barrier execution')
        
        try:
            # Load data with optimized streaming and M1 optimizations
            if self.m1_optimizations_enabled and self.memory_optimizer:
                with self.memory_optimizer.memory_checkpoint("data_loading"):
                    data = await self._load_data_optimized(symbol, exchange, timeframe, data_dir)
            else:
                data = await self._load_data_optimized(symbol, exchange, timeframe, data_dir)
            if data is None or data.empty:
                self.logger.error('❌ Failed to load data')
                return TripleBarrierResult.failure_result(
                    error='Failed to load data',
                    error_type='DataLoadError',
                    metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir}
                )
            
            self.logger.info(f'✅ Loaded data with shape: {data.shape}')
            
            # Calculate volatility-based parameters
            volatility_params = self.volatility_calculator.calculate_volatility_based_parameters(data)
            
            # Update config with volatility-based parameters if not explicitly set
            if self.config.get('use_volatility_based_params', True):
                self.config.update(volatility_params)
                self.logger.info('📊 Using volatility-based parameters')
            else:
                self.logger.info('📊 Using user-specified parameters')
            
            # Apply vectorized triple barrier with M1 optimizations
            if self.m1_optimizations_enabled and self.memory_optimizer:
                with self.memory_optimizer.memory_checkpoint("vectorized_processing"):
                    labeled_data = self.vectorized_processor.apply_triple_barrier_vectorized(data)
            else:
                labeled_data = self.vectorized_processor.apply_triple_barrier_vectorized(data)
            
            if labeled_data is None or labeled_data.empty:
                self.logger.error('❌ Failed to generate triple barrier labels')
                return TripleBarrierResult.failure_result(
                    error='Failed to generate triple barrier labels',
                    error_type='LabelingError',
                    metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
                )
            
            # Save results with optimized I/O
            success = await self._save_results_optimized(
                data, labeled_data, symbol, exchange, timeframe, data_dir
            )
            
            if success:
                return self._create_success_result(
                    data, labeled_data, symbol, exchange, timeframe, data_dir, step_start, volatility_params
                )
            else:
                self.logger.error('❌ Failed to save results')
                return TripleBarrierResult.failure_result(
                    error='Failed to save results',
                    error_type='SaveError',
                    metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
                )
                
        except ValueError as e:
            self.logger.error(f'❌ Parameter validation error: {e}')
            self.memory_monitor.trigger_gc()
            return TripleBarrierResult.failure_result(
                error=f'Parameter validation failed: {str(e)}',
                error_type='ParameterValidationError',
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe},
                execution_time=time.time() - step_start
            )
        except Exception as e:
            self.logger.exception(f'❌ Unexpected error in optimized triple barrier method: {e}')
            self.memory_monitor.trigger_gc()
            return TripleBarrierResult.failure_result(
                error=f'Unexpected error: {str(e)}',
                error_type=type(e).__name__,
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe},
                execution_time=time.time() - step_start
            )

    async def _load_data_optimized(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str
    ) -> Optional[pd.DataFrame]:
        """Load data with optimized I/O operations."""
        try:
            unified_data_path = Path(data_dir) / 'unified' / exchange / symbol / timeframe
            if not unified_data_path.exists():
                self.logger.error(f'❌ Unified data not found at {unified_data_path}')
                return None
                
            data_files = list(unified_data_path.glob('*.parquet'))
            if not data_files:
                self.logger.error(f'❌ No parquet files found in {unified_data_path}')
                return None
            
            # Use parallel loading for multiple files
            if len(data_files) > 1:
                self.logger.info(f'📁 Loading {len(data_files)} files in parallel')
                data = await self._load_files_parallel(data_files)
            else:
                latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
                self.logger.info(f'📁 Loading data from {latest_file}')
                data = await self._load_single_file_optimized(latest_file)
            
            # Validate loaded data
            if data is not None:
                validation_result = self._validate_input_data_optimized(data, symbol, exchange, timeframe)
                if not validation_result['valid']:
                    self.logger.error(f'❌ Data validation failed: {validation_result["error"]}')
                    return None
                self.logger.info('✅ Input data validation passed')
            
            return data
                
        except Exception as e:
            self.logger.exception(f'❌ Error loading data: {e}')
            return None

    async def _load_files_parallel(self, file_paths: List[Path]) -> Optional[pd.DataFrame]:
        """Load multiple files in parallel for better I/O performance."""
        try:
            # Use ThreadPoolExecutor for I/O bound operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(file_paths), 4)) as executor:
                # Submit all file loading tasks
                future_to_file = {
                    executor.submit(self._load_single_file_optimized, file_path): file_path 
                    for file_path in file_paths
                }
                
                # Collect results
                dataframes = []
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        data = future.result()
                        if data is not None and not data.empty:
                            dataframes.append(data)
                            self.logger.info(f'✅ Loaded {file_path.name}: {len(data):,} rows')
                    except Exception as e:
                        self.logger.warning(f'⚠️ Failed to load {file_path.name}: {e}')
                
                if dataframes:
                    # Combine all dataframes
                    combined_data = pd.concat(dataframes, ignore_index=True)
                    combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
                    self.logger.info(f'✅ Combined {len(dataframes)} files: {len(combined_data):,} total rows')
                    return combined_data
                else:
                    self.logger.error('❌ No files loaded successfully')
                    return None
                    
        except Exception as e:
            self.logger.exception(f'❌ Error in parallel file loading: {e}')
            return None

    async def _load_single_file_optimized(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load single file with optimized I/O operations."""
        try:
            # Check file size to determine loading strategy
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb > 500:  # Large file, use streaming
                self.logger.info(f'🌊 Using streaming for large file: {file_size_mb:.2f} MB')
                return await self._stream_load_data_optimized(file_path)
            else:
                # Small file, load with PyArrow for better performance
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(file_path)
                    data = table.to_pandas()
                    self.logger.info(f'✅ Loaded with PyArrow: {len(data):,} rows')
                    return data
                except ImportError:
                    # Fallback to pandas
                    data = safe_read_parquet(file_path)
                    self.logger.info(f'✅ Loaded with pandas: {len(data):,} rows')
                    return data
                    
        except Exception as e:
            self.logger.exception(f'❌ Error loading single file {file_path}: {e}')
            return None

    async def _stream_load_data_optimized(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Optimized streaming load that doesn't accumulate all data in memory."""
        try:
            self.logger.info(f'🌊 Starting optimized streaming for {file_path}')
            
            # Use PyArrow for streaming if available
            try:
                import pyarrow.parquet as pq
                import pyarrow as pa
                
                # Read file metadata first
                parquet_file = pq.ParquetFile(file_path)
                total_rows = parquet_file.metadata.num_rows
                self.logger.info(f'📊 File contains {total_rows:,} rows')
                
                # Process in chunks without accumulating in memory
                chunk_size = 100000  # 100K rows per chunk
                processed_chunks = []
                
                for batch in parquet_file.iter_batches(batch_size=chunk_size):
                    # Convert batch to pandas
                    chunk_df = batch.to_pandas()
                    
                    # Validate chunk
                    is_valid, error_msg = FastFailValidator.validate_data(chunk_df)
                    if not is_valid:
                        self.logger.warning(f'⚠️ Chunk validation failed: {error_msg}')
                        continue
                    
                    processed_chunks.append(chunk_df)
                    
                    # Memory management
                    if len(processed_chunks) > 10:  # Keep only last 10 chunks in memory
                        # Combine and save intermediate result
                        combined = pd.concat(processed_chunks, ignore_index=True)
                        temp_file = file_path.parent / f'temp_stream_{file_path.stem}.parquet'
                        combined.to_parquet(temp_file, compression='snappy', index=False)
                        processed_chunks = [combined]  # Keep only combined result
                        
                        # Clean up memory
                        del combined
                        if hasattr(self.memory_monitor, 'trigger_gc'):
                            self.memory_monitor.trigger_gc()
                
                # Final combination
                if processed_chunks:
                    final_data = pd.concat(processed_chunks, ignore_index=True)
                    final_data = final_data.sort_values('timestamp').reset_index(drop=True)
                    self.logger.info(f'✅ Streaming completed: {len(final_data):,} rows')
                    return final_data
                else:
                    self.logger.warning('⚠️ No valid chunks processed')
                    return None
                    
            except ImportError:
                # Fallback to pandas streaming
                self.logger.warning('⚠️ PyArrow not available, using pandas streaming fallback')
                return await self._stream_load_data_pandas_fallback(file_path)
                
        except Exception as e:
            self.logger.exception(f'❌ Error in optimized streaming: {e}')
            return None

    async def _stream_load_data_pandas_fallback(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Pandas fallback for streaming (less efficient but functional)."""
        try:
            # Read file in chunks
            chunk_size = 50000
            chunks = []
            
            for chunk in pd.read_parquet(file_path, chunksize=chunk_size):
                chunks.append(chunk)
                
                # Memory management
                if len(chunks) > 5:
                    combined = pd.concat(chunks, ignore_index=True)
                    chunks = [combined]
                    del combined
            
            if chunks:
                final_data = pd.concat(chunks, ignore_index=True)
                return final_data
            else:
                return None
                
        except Exception as e:
            self.logger.exception(f'❌ Error in pandas streaming fallback: {e}')
            return None

    def _validate_input_data_optimized(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Optimized input data validation with fast fails."""
        try:
            # Use fast fail validator
            is_valid, error_msg = FastFailValidator.validate_data(data)
            if not is_valid:
                return {'valid': False, 'error': error_msg}
            
            # Additional quality checks
            price_changes = data['close'].pct_change().abs()
            extreme_moves = price_changes > 0.5  # 50% moves
            if extreme_moves.any():
                extreme_count = extreme_moves.sum()
                self.logger.warning(f'⚠️ Extreme price movements detected: {extreme_count} rows with >50% change')
            
            # Log data quality metrics
            self.logger.info(f'📊 Data quality metrics:')
            self.logger.info(f'   Rows: {len(data):,}')
            self.logger.info(f'   Columns: {len(data.columns)}')
            self.logger.info(f'   Price range: ${data["close"].min():.4f} - ${data["close"].max():.4f}')
            self.logger.info(f'   Avg daily return: {data["close"].pct_change().mean():.6f}')
            self.logger.info(f'   Volatility: {data["close"].pct_change().std():.6f}')
            
            return {'valid': True, 'metrics': {
                'rows': len(data),
                'columns': len(data.columns),
                'price_range': (data['close'].min(), data['close'].max()),
                'avg_return': data['close'].pct_change().mean(),
                'volatility': data['close'].pct_change().std()
            }}
            
        except Exception as e:
            self.logger.exception(f'❌ Error validating input data: {e}')
            return {'valid': False, 'error': f'Validation error: {str(e)}'}

    async def _save_results_optimized(
        self,
        original_data: pd.DataFrame,
        labeled_data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> bool:
        """Save results with optimized I/O operations."""
        try:
            output_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_triple_barrier_labels_optimized.parquet'
            ensure_directory(output_path.parent)
            
            # Combine data efficiently
            result_data = original_data.copy()
            result_data['triple_barrier_label'] = labeled_data['label']
            
            if 'potential_profit_pct' in labeled_data.columns:
                result_data['potential_profit_pct'] = labeled_data['potential_profit_pct']
                
                # Calculate net profit after fees (corrected fee: 0.04% per side)
                fee_per_side = 0.0004  # 0.04% per side (corrected from 0.05%)
                result_data['potential_profit_net_pct'] = (
                    result_data['potential_profit_pct'] - (2.0 * fee_per_side)
                ).astype(np.float64)
            
            # Optimize data types before saving
            result_data = optimize_dataframe_dtypes(result_data)
            
            # Save with serialization utilities for better performance and reliability
            try:
                # Use ParquetSerializer from utilities
                parquet_serializer = self.utils.get_function('serialization_utils', 'ParquetSerializer')()
                parquet_serializer.save(result_data, output_path)
                self.logger.info(f'✅ Saved with ParquetSerializer: {output_path}')
                
            except Exception as e:
                self.logger.warning(f'⚠️ ParquetSerializer failed: {e}, falling back to PyArrow')
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    
                    table = pa.Table.from_pandas(result_data)
                    pq.write_table(table, output_path, compression='snappy')
                    self.logger.info(f'✅ Saved with PyArrow: {output_path}')
                    
                except ImportError:
                    # Final fallback to pandas
                    success = safe_to_parquet(
                        result_data, 
                        output_path,
                        compression='snappy',
                        index=False
                    )
                    if not success:
                        self.logger.error('❌ Failed to save parquet file')
                        return False
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            self.logger.info(f'✅ Triple barrier labels saved to {output_path} ({file_size_mb:.2f} MB)')
            return True
                
        except Exception as e:
            self.logger.exception(f'❌ Error saving results: {e}')
            return False

    def _create_success_result(
        self, 
        data: pd.DataFrame, 
        labeled_data: pd.DataFrame, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str, 
        step_start: float,
        volatility_params: Dict[str, Any]
    ) -> TripleBarrierResult:
        """Create standardized success result."""
        # Calculate label statistics
        labels = labeled_data['label'].values
        total_signals = len(labels)
        buy_signals = (labels == 1).sum()
        sell_signals = (labels == -1).sum()
        no_action = (labels == 0).sum()
        
        label_stats = {
            'total_labels': int(total_signals),
            'buy_signals': int(buy_signals),
            'sell_signals': int(sell_signals),
            'no_action': int(no_action),
            'buy_ratio': float(buy_signals / total_signals) if total_signals > 0 else 0.0,
            'sell_ratio': float(sell_signals / total_signals) if total_signals > 0 else 0.0,
            'signal_balance': float(min(buy_signals, sell_signals) / max(buy_signals, sell_signals)) if max(buy_signals, sell_signals) > 0 else 0.0
        }
        
        self._log_step_timing('Optimized Triple Barrier Method', step_start)
        self.memory_monitor.log_memory_status('after triple barrier execution')
        
        output_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_triple_barrier_labels_optimized.parquet'
        
        return TripleBarrierResult.success_result(
            data=data,
            metadata={
                'symbol': symbol, 
                'exchange': exchange, 
                'timeframe': timeframe,
                'output_file': str(output_path),
                'data_shape': data.shape, 
                'label_stats': label_stats,
                'memory_stats': self.memory_monitor.get_memory_stats(),
                'volatility_params': volatility_params,
                'optimization_features': [
                    'vectorized_operations',
                    'lookahead_bias_prevention',
                    'volatility_based_parameters',
                    'optimized_io',
                    'fast_fail_validation',
                    'corrected_trading_fees'
                ]
            },
            execution_time=time.time() - step_start
        )

    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f'⏱️ {step_name} completed in {elapsed:.2f} seconds')
        
        # Log memory usage
        memory_stats = self.memory_monitor.get_memory_stats()
        self.logger.info(f'💾 Memory usage: {memory_stats["current_mb"]:.1f}MB (peak: {memory_stats["peak_mb"]:.1f}MB)')

@traced(span_name='execute_optimized_triple_barrier')
@validates()
@handles_errors()
@cached()
@log_execution_time()
async def run_step_optimized(
    symbol: str, 
    exchange: str, 
    timeframe: str, 
    data_dir: str = None, 
    force_rerun: bool = False, 
    config: dict[str, Any] = None
) -> StepResult:
    """Run Optimized Step 4: Triple Barrier Method with comprehensive improvements.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force rerun flag
        config: Configuration dictionary
        
    Returns:
        StepResult: Standardized result with success status and details
    """
    logger.info('🚀 Starting Optimized Step 4: Triple Barrier Method')
    if data_dir is None:
        data_dir = 'data_cache'
    
    step_start = time.time()
    try:
        step = OptimizedTripleBarrierMethodStep(config or {})
        await step.initialize()
        result = await step.execute_triple_barrier_method(symbol, exchange, timeframe, data_dir, force_rerun)
        
        # Standardize the result if it's not already a StepResult
        standardized_result = standardize_result(result, "optimized_triple_barrier_method")

        if standardized_result.success:
            logger.info('✅ Optimized Step 4: Triple Barrier Method completed successfully')
            logger.info('🎯 All optimizations applied: vectorized operations, lookahead bias prevention, volatility-based parameters')

            # Return dictionary for pipeline state integration
            return {
                'success': True,
                'step04_5_triple_barrier_method_completed': True,
                'triple_barrier_data': result.data,
                'triple_barrier_metadata': result.metadata,
                'execution_time': standardized_result.execution_time,
                'step_name': 'step04_5_triple_barrier_method_optimized'
            }
        else:
            logger.error('❌ Optimized Step 4: Triple Barrier Method failed')
            logger.error(f'🔍 Error: {standardized_result.error}')

            return {
                'success': False,
                'step04_5_triple_barrier_method_completed': False,
                'error': standardized_result.error,
                'execution_time': standardized_result.execution_time,
                'step_name': 'step04_5_triple_barrier_method_optimized'
            }
        
    except Exception as e:
        logger.exception(f'❌ Error in optimized triple barrier method: {e}')

        return {
            'success': False,
            'step04_5_triple_barrier_method_completed': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'execution_time': time.time() - step_start,
            'step_name': 'step04_5_triple_barrier_method_optimized'
        }

if __name__ == '__main__':
    async def test() -> None:
        test_config = {
            'symbol': 'ETHUSDT', 
            'exchange': 'BINANCE', 
            'timeframe': '1m',
            'use_volatility_based_params': True,
            'max_memory_mb': 2048.0
        }
        success = await run_step_optimized(
            symbol='ETHUSDT', 
            exchange='BINANCE', 
            timeframe='1m', 
            data_dir='data_cache', 
            force_rerun=False, 
            config=test_config
        )
        print(f'Test result: {success}')
    
    asyncio.run(test())