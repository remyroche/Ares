"""
Feature Generation Period + Lookback Optimization Step

This step combines period optimization and lookback optimization to optimize both
concurrently, ensuring at least 2 periods per feature with no recency bias.

Key Features:
- Concurrent period and lookback optimization
- Minimum 2 periods per feature
- No recency bias or adaptive windows
- Correlation threshold >0.85 for redundancy
- Top 1 period/lookback used as default for trading
- Top 3 periods/lookback used for interaction generation
"""

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import traceback
import asyncio

from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline_runner import (
    run_period_lookback_optimization_step
)
from src.training.steps.pre_training.components.base_component import (
    BasePreTrainingComponent, ComponentConfig, ComponentResult
)
from dataclasses import field
from src.utils.common_operations import safe_dataframe_operation
from src.utils.matrix_operations import safe_matrix_multiply, optimize_dataframe

# Import tprint utilities for enhanced logging
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug,
        tprint_performance, tprint_step, tprint_result
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)
    def tprint_step(*args, **kwargs): print("STEP:", *args, **kwargs)
    def tprint_result(*args, **kwargs): print("RESULT:", *args, **kwargs)

@dataclass
class PeriodLookbackOptimizationConfig(ComponentConfig):
    """Configuration for period + lookback optimization step."""
    
    min_periods: int = 2
    correlation_threshold: float = 0.85
    no_recency_bias: bool = True
    top_1_trading: bool = True
    top_3_interactions: bool = True
    log_level: str = "INFO"
    timeout_seconds: int = 300

@dataclass
class PeriodLookbackOptimizationResult:
    """Result of period + lookback optimization step."""

    success: bool
    optimized_periods: int
    optimized_lookbacks: int
    optimization_metadata: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None

class FeatureGenerationPeriodLookbackOptimizationStep(BasePreTrainingComponent):
    """Period + lookback optimization step that calls the consolidated pipeline."""

    def __init__(self, config: Optional[PeriodLookbackOptimizationConfig] = None):
        """Initialize the period + lookback optimization step."""
        super().__init__(config or PeriodLookbackOptimizationConfig())
        self.logger = logging.getLogger(__name__)
        
        # Apply config settings
        if hasattr(self.config, 'log_level') and self.config.log_level:
            self.logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
        
        # Set up constraint validation parameters
        self.min_periods = self.config.min_periods
        self.correlation_threshold = self.config.correlation_threshold
        self.no_recency_bias = self.config.no_recency_bias
        self.top_1_trading = self.config.top_1_trading
        self.top_3_interactions = self.config.top_3_interactions

    async def execute(self,
                     training_input: Dict[str, Any],
                     pipeline_state: Dict[str, Any]) -> ComponentResult:
        """Execute period + lookback optimization step using consolidated pipeline."""

        self.logger.info("📊 Starting period + lookback optimization step using consolidated pipeline")

        # Extract parameters from training_input
        data = training_input.get('data')
        symbol = training_input.get('symbol', 'ETHUSDT')
        timeframe = training_input.get('timeframe', '15m')
        direction = training_input.get('direction', 'longs')
        intensity = training_input.get('intensity', 'blank')
        lookback_days = training_input.get('lookback_days')
        start_date = training_input.get('start_date')
        end_date = training_input.get('end_date')
        exchange = training_input.get('exchange', 'binance')
        custom_overrides = training_input.get('custom_overrides')

        try:
            # Input validation
            if data is None:
                error_msg = "Data is required for period + lookback optimization"
                if TPRINT_AVAILABLE:
                    tprint_error(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(
                    success=False,
                    artifacts={},
                    metadata={'constraint_violations': ['missing_data']},
                    error_message=error_msg
                )

            if len(data) < 100:
                error_msg = f"Data must have at least 100 rows, got {len(data)}"
                if TPRINT_AVAILABLE:
                    tprint_error(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(
                    success=False,
                    artifacts={},
                    metadata={'constraint_violations': ['insufficient_data']},
                    error_message=error_msg
                )

            if TPRINT_AVAILABLE:
                tprint_step(f"🚀 Starting period + lookback optimization for {symbol} {timeframe} {direction}")

            # Call the consolidated pipeline runner with timeout
            try:
                result = await asyncio.wait_for(
                    run_period_lookback_optimization_step(
                        data=data,
                        symbol=symbol,
                        timeframe=timeframe,
                        direction=direction,
                        intensity=intensity,
                        lookback_days=lookback_days,
                        start_date=start_date,
                        end_date=end_date,
                        exchange=exchange,
                        custom_overrides=custom_overrides
                    ),
                    timeout=self.config.timeout_seconds
                )
            except asyncio.TimeoutError:
                error_msg = f"Period + lookback optimization timed out after {self.config.timeout_seconds} seconds"
                if TPRINT_AVAILABLE:
                    tprint_error(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(
                    success=False,
                    artifacts={},
                    metadata={'constraint_violations': ['timeout']},
                    error_message=error_msg
                )

            # Validate constraints from the result
            constraint_violations = []
            optimization_metadata = result.get('optimization_metadata', {})
            
            # Check minimum periods per feature
            optimized_periods = result.get('optimized_periods', 0)
            if optimized_periods < self.min_periods:
                constraint_violations.append('min_periods_violation')
                if TPRINT_AVAILABLE:
                    tprint_warning(f"⚠️ Constraint violation: Only {optimized_periods} periods, minimum {self.min_periods} required")

            # Check correlation threshold
            correlation_threshold = optimization_metadata.get('correlation_threshold', self.correlation_threshold)
            if correlation_threshold <= self.correlation_threshold:
                constraint_violations.append('correlation_threshold_violation')
                if TPRINT_AVAILABLE:
                    tprint_warning(f"⚠️ Constraint violation: Correlation threshold {correlation_threshold} <= {self.correlation_threshold}")

            # Check for recency bias (should be False or not present)
            has_recency_bias = optimization_metadata.get('has_recency_bias', False)
            if self.no_recency_bias and has_recency_bias:
                constraint_violations.append('recency_bias_violation')
                if TPRINT_AVAILABLE:
                    tprint_warning("⚠️ Constraint violation: Recency bias detected")

            # Check for top-1/top-3 outputs in artifacts
            artifacts = result.get('artifacts', {})
            if self.top_1_trading and 'top_1_selection' not in artifacts:
                constraint_violations.append('missing_top_1_selection')
                if TPRINT_AVAILABLE:
                    tprint_warning("⚠️ Constraint violation: Top 1 selection missing from artifacts")
            
            if self.top_3_interactions and 'top_3_selections' not in artifacts:
                constraint_violations.append('missing_top_3_selections')
                if TPRINT_AVAILABLE:
                    tprint_warning("⚠️ Constraint violation: Top 3 selections missing from artifacts")

            # Convert result to ComponentResult with improved metadata
            component_result = ComponentResult(
                success=result['success'] and len(constraint_violations) == 0,
                artifacts=result.get('artifacts', {}),
                metadata={
                    'runner': {
                        'optimized_periods': result.get('optimized_periods', 0),
                        'optimized_lookbacks': result.get('optimized_lookbacks', 0),
                        'optimization_metadata': optimization_metadata,
                        **result.get('metadata', {})
                    },
                    'constraints': {
                        'min_periods': self.min_periods,
                        'correlation_threshold': self.correlation_threshold,
                        'no_recency_bias': self.no_recency_bias,
                        'top_1_trading': self.top_1_trading,
                        'top_3_interactions': self.top_3_interactions
                    },
                    'constraint_violations': constraint_violations,
                    'validation_passed': len(constraint_violations) == 0
                },
                error_message=result.get('error_message')
            )

            if component_result.success:
                if TPRINT_AVAILABLE:
                    tprint_success(f"✅ Period + lookback optimization completed successfully with {component_result.metadata['runner']['optimized_periods']} periods and {component_result.metadata['runner']['optimized_lookbacks']} lookbacks")
                self.logger.info(f"✅ Period + lookback optimization completed successfully with {component_result.metadata['runner']['optimized_periods']} periods and {component_result.metadata['runner']['optimized_lookbacks']} lookbacks")
            else:
                error_details = f"Optimization failed: {component_result.error_message}" if component_result.error_message else f"Constraint violations: {constraint_violations}"
                if TPRINT_AVAILABLE:
                    tprint_error(f"❌ {error_details}")
                self.logger.error(f"❌ Period + lookback optimization failed: {error_details}")

            return component_result

        except asyncio.TimeoutError:
            error_msg = f"Period + lookback optimization step timed out after {self.config.timeout_seconds} seconds"
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg)
            return ComponentResult(
                success=False,
                artifacts={},
                metadata={'constraint_violations': ['timeout']},
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"Period + lookback optimization step failed with exception: {str(e)}"
            traceback_str = traceback.format_exc()
            
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ {error_msg}")
                tprint_debug(f"Traceback: {traceback_str}")
            
            self.logger.error(f"❌ {error_msg}")
            self.logger.debug(f"Traceback: {traceback_str}")
            
            return ComponentResult(
                success=False,
                artifacts={},
                metadata={'constraint_violations': ['execution_error']},
                error_message=error_msg
            )

    # Required utility methods for BasePreTrainingComponent
    def safe_dataframe_operation(self, operation_func, *args, **kwargs):
        """Safe dataframe operation wrapper."""
        return safe_dataframe_operation(operation_func, *args, **kwargs)

    def safe_matrix_multiply(self, a, b):
        """Safe matrix multiplication."""
        return safe_matrix_multiply(a, b)

    def optimize_dataframe_for_matrix_ops(self, df):
        """Optimize dataframe for matrix operations."""
        return optimize_dataframe(df)

# Command handler for ares_launcher integration
async def handle_feature_generation_period_lookback_optimization_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
    use_sample_data: bool = False,
    random_seed: Optional[int] = None,
    **kwargs
) -> ComponentResult:
    """
    Handle feature generation period + lookback optimization step command.

    Args:
        symbol: Trading symbol (default: "ETHUSDT")
        timeframe: Timeframe (default: "15m")
        direction: Direction (default: "longs")
        intensity: Pipeline intensity (default: "blank")
        lookback_days: Lookback days (optional)
        start_date: Start date (optional)
        end_date: End date (optional)
        exchange: Exchange (default: "binance")
        custom_overrides: Custom configuration overrides (optional)
        use_sample_data: Whether to use sample data for testing (default: False)
        random_seed: Random seed for reproducible sample data (optional)
        **kwargs: Additional arguments

    Returns:
        ComponentResult with optimization results
    """
    # Input validation
    if not symbol or not isinstance(symbol, str):
        raise ValueError("symbol must be a non-empty string")
    if not timeframe or not isinstance(timeframe, str):
        raise ValueError("timeframe must be a non-empty string")
    if direction not in ['longs', 'shorts', 'both']:
        raise ValueError("direction must be one of: 'longs', 'shorts', 'both'")
    if intensity not in ['blank', 'light', 'medium', 'heavy']:
        raise ValueError("intensity must be one of: 'blank', 'light', 'medium', 'heavy'")
    if exchange not in ['binance', 'coinbase', 'kraken']:
        raise ValueError("exchange must be one of: 'binance', 'coinbase', 'kraken'")

    # Create sample data for optimization if requested
    data = None
    if use_sample_data:
        if random_seed is not None:
            np.random.seed(random_seed)
        
        data = pd.DataFrame({
            'open': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 105,
            'low': np.random.randn(1000).cumsum() + 95,
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 1000)
        })
        
        if TPRINT_AVAILABLE:
            tprint_info(f"Using sample data with {len(data)} rows (random_seed={random_seed})")
    else:
        if TPRINT_AVAILABLE:
            tprint_warning("No data provided and use_sample_data=False. This may cause errors.")

    # Create step instance with config and execute
    config = PeriodLookbackOptimizationConfig(
        min_periods=2,
        correlation_threshold=0.85,
        no_recency_bias=True,
        top_1_trading=True,
        top_3_interactions=True,
        log_level="INFO",
        timeout_seconds=300
    )
    step = FeatureGenerationPeriodLookbackOptimizationStep(config)
    
    # Prepare training_input and pipeline_state
    training_input = {
        'data': data,
        'symbol': symbol,
        'timeframe': timeframe,
        'direction': direction,
        'intensity': intensity,
        'lookback_days': lookback_days,
        'start_date': start_date,
        'end_date': end_date,
        'exchange': exchange,
        'custom_overrides': custom_overrides
    }
    pipeline_state = {}

    return await step.execute(training_input, pipeline_state)

# Register component with factory
def _register_feature_generation_period_lookback_optimization_step():
    """Register the feature generation period + lookback optimization step component with the factory."""
    try:
        from src.training.steps.pre_training.components import ComponentFactory
        ComponentFactory.register_component(
            'feature_generation_period_lookback_optimization_step',
            FeatureGenerationPeriodLookbackOptimizationStep
        )
    except ImportError:
        # Component factory not available, skip registration
        pass

# Register the component when module is imported
_register_feature_generation_period_lookback_optimization_step()