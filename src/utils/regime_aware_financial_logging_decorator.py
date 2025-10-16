"""
Regime-Aware Financial Logging Decorator

This decorator automatically adds per-HMM regime logging and fail-fast validation
to training steps that come after HMM-based data splitting (step08).
"""

import functools

from typing import Any, Dict, List, Optional, Callable, Union
import pandas as pd

# Import the enhanced financial metrics logger
try:
    from src.utils.enhanced_financial_metrics_logger import (
        get_enhanced_financial_metrics_logger,
        enhanced_financial_metrics_context,
        validate_and_log_regime_data
    )
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False

# Import the base financial metrics logger as fallback
try:
    from src.utils.financial_metrics_logger import get_financial_metrics_logger
    BASE_LOGGING_AVAILABLE = True
except ImportError:
    BASE_LOGGING_AVAILABLE = False

# Import the main logger
try:
    from src.utils.logger import system_logger
    import logging

except ImportError:
    system_logger = None

def regime_aware_financial_logging(
    step_name: str,
    enable_regime_validation: bool = True,
    enable_fail_fast: bool = True,
    min_regime_samples: int = 100,
    max_regime_imbalance: float = 0.8,
    regime_column: str = 'composite_cluster_id',
    expected_regimes: Optional[List[str]] = None,
    min_data_quality: float = 0.7,
    log_regime_distribution: bool = True,
    log_regime_performance: bool = True,
    log_regime_transitions: bool = True
):
    """
    Decorator to add regime-aware financial logging to training steps.

    This decorator automatically:
    1. Validates regime data for steps after HMM-based data splitting
    2. Logs per-regime metrics with fail-fast validation
    3. Prevents empty running or important degradation
    4. Provides comprehensive regime-specific financial metrics

    Args:
        step_name: Name of the training step
        enable_regime_validation: Enable regime data validation
        enable_fail_fast: Enable fail-fast behavior
        min_regime_samples: Minimum samples required per regime
        max_regime_imbalance: Maximum allowed regime imbalance ratio
        regime_column: Name of the regime column in data
        expected_regimes: List of expected regime IDs
        min_data_quality: Minimum required data quality score
        log_regime_distribution: Log regime distribution metrics
        log_regime_performance: Log regime-specific performance metrics
        log_regime_transitions: Log regime transition metrics

    Usage:
        @regime_aware_financial_logging(
            step_name="Step09_HMM_Based_Training",
            enable_fail_fast=True,
            min_regime_samples=100
        )
        async def execute(self, training_input, pipeline_state):
            # Step implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get the step instance (first argument is usually 'self')
            step_instance = args[0] if args else None

            # Extract configuration from step instance or kwargs
            config = {}
            if step_instance and hasattr(step_instance, 'config'):
                config = step_instance.config
            elif 'config' in kwargs:
                config = kwargs['config']

            # Get symbol, exchange, timeframe from config
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'BINANCE')
            timeframe = config.get('timeframe', '1m')

            # Initialize logging
            logger = None
            if ENHANCED_LOGGING_AVAILABLE:
                logger = get_enhanced_financial_metrics_logger()
            elif BASE_LOGGING_AVAILABLE:
                logger = get_financial_metrics_logger()

            # Get system logger for fallback
            system_log = system_logger.getChild(f'RegimeAwareLogging.{step_name}') if system_logger else None

            try:
                # Extract data from arguments for validation
                data = None
                training_input = kwargs.get('training_input', {})
                pipeline_state = kwargs.get('pipeline_state', {})

                # Try to find data in various locations
                if 'dataframe' in pipeline_state:
                    data = pipeline_state['dataframe']
                elif 'data' in training_input:
                    data = training_input['data']
                elif 'data' in pipeline_state:
                    data = pipeline_state['data']
                elif len(args) > 1 and isinstance(args[1], pd.DataFrame):
                    data = args[1]

                # Use enhanced financial metrics context if available
                if ENHANCED_LOGGING_AVAILABLE:
                    with enhanced_financial_metrics_context(
                        step_name=step_name,
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        data=data,
                        expected_regimes=expected_regimes
                    ) as enhanced_logger:
                        # Log step start with regime validation
                        if enable_regime_validation and data is not None:
                            validation_success = validate_and_log_regime_data(
                                symbol=symbol,
                                exchange=exchange,
                                timeframe=timeframe,
                                step_name=step_name,
                                data=data,
                                regime_column=regime_column
                            )

                            if not validation_success and enable_fail_fast:
                                error_msg = f"Regime validation failed for {step_name}"
                                if system_log:
                                    system_log.error(f"🚨 {error_msg}")
                                raise RuntimeError(error_msg)

                        # Execute the original function
                        result = await func(*args, **kwargs)

                        # Log regime-specific metrics from result
                        if result and isinstance(result, dict):
                            await _log_regime_metrics_from_result(
                                enhanced_logger=enhanced_logger,
                                result=result,
                                symbol=symbol,
                                exchange=exchange,
                                timeframe=timeframe,
                                step_name=step_name,
                                data=data,
                                log_regime_distribution=log_regime_distribution,
                                log_regime_performance=log_regime_performance,
                                log_regime_transitions=log_regime_transitions
                            )

                        return result

                else:
                    # Fallback to base logging
                    if logger:
                        logger.log_step_start(step_name, symbol, exchange, timeframe)

                    # Execute the original function
                    result = await func(*args, **kwargs)

                    # Log basic metrics
                    if logger and result and isinstance(result, dict):
                        await _log_basic_metrics_from_result(
                            logger=logger,
                            result=result,
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=timeframe,
                            step_name=step_name
                        )

                    if logger:
                        logger.log_step_end(step_name, symbol, exchange, timeframe, success=True)

                    return result

            except Exception as e:
                # Log error
                if logger:
                    logger.log_step_end(step_name, symbol, exchange, timeframe, success=False, error_message=str(e))

                if system_log:
                    system_log.error(f"❌ {step_name} failed: {e}")

                raise

        return wrapper
    return decorator

async def _log_regime_metrics_from_result(
    enhanced_logger,
    result: Dict[str, Any],
    symbol: str,
    exchange: str,
    timeframe: str,
    step_name: str,
    data: Optional[pd.DataFrame] = None,
    log_regime_distribution: bool = True,
    log_regime_performance: bool = True,
    log_regime_transitions: bool = True
) -> None:
    """Log regime-specific metrics from step result."""
    try:
        # Log regime distribution metrics
        regime_column = 'composite_cluster_id'  # Default regime column name
        if log_regime_distribution and data is not None and regime_column in data.columns:
            regime_data = data[regime_column].dropna()
            regime_counts = regime_data.value_counts()

            for regime_id, count in regime_counts.items():
                enhanced_logger.log_financial_metric_with_regime_validation(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name=f"regime_{regime_id}_sample_count",
                    metric_value=float(count),
                    metric_type="regime",
                    step_name=step_name,
                    regime_id=str(regime_id),
                    data=data
                )

        # Log regime performance metrics
        if log_regime_performance:
            # Look for regime-specific performance metrics in result
            regime_metrics = {}

            # Check for common regime performance patterns
            for key, value in result.items():
                if isinstance(value, dict):
                    # Check if this looks like regime-specific metrics
                    if any(regime_key in key.lower() for regime_key in ['regime', 'cluster', 'state']):
                        regime_metrics[key] = value
                    elif isinstance(value, dict) and any(str(k).isdigit() for k in value.keys()):
                        # Looks like regime_id -> metrics mapping
                        regime_metrics[key] = value

            # Log regime-specific metrics
            if regime_metrics:
                enhanced_logger.log_per_regime_metrics(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    step_name=step_name,
                    regime_metrics=regime_metrics,
                    data=data
                )

        # Log regime transition metrics
        if log_regime_transitions and 'regime_transitions' in result:
            transitions = result['regime_transitions']
            if isinstance(transitions, dict):
                for transition_key, transition_value in transitions.items():
                    enhanced_logger.log_financial_metric_with_regime_validation(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        metric_name=f"regime_transition_{transition_key}",
                        metric_value=float(transition_value),
                        metric_type="regime",
                        step_name=step_name,
                        data=data
                    )

        # Log overall step performance
        if 'success' in result:
            enhanced_logger.log_financial_metric_with_regime_validation(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                metric_name="step_success",
                metric_value=1.0 if result['success'] else 0.0,
                metric_type="performance",
                step_name=step_name,
                data=data
            )

        # Log execution time if available
        if 'execution_time' in result:
            enhanced_logger.log_financial_metric_with_regime_validation(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                metric_name="execution_time_seconds",
                metric_value=float(result['execution_time']),
                metric_type="performance",
                step_name=step_name,
                data=data
            )

    except Exception as e:
        if system_logger:
            system_logger.getChild('RegimeMetricsLogging').warning(f"Failed to log regime metrics: {e}")

async def _log_basic_metrics_from_result(
    logger,
    result: Dict[str, Any],
    symbol: str,
    exchange: str,
    timeframe: str,
    step_name: str
) -> None:
    """Log basic metrics from step result using base logger."""
    try:
        # Log success status
        if 'success' in result:
            logger.log_financial_metric(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                metric_name="step_success",
                metric_value=1.0 if result['success'] else 0.0,
                metric_type="performance",
                step_name=step_name
            )

        # Log execution time if available
        if 'execution_time' in result:
            logger.log_financial_metric(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                metric_name="execution_time_seconds",
                metric_value=float(result['execution_time']),
                metric_type="performance",
                step_name=step_name
            )

    except Exception as e:
        if system_logger:
            system_logger.getChild('BasicMetricsLogging').warning(f"Failed to log basic metrics: {e}")

def is_post_hmm_step(step_name: str) -> bool:
    """
    Check if a step comes after HMM-based data splitting (step08).

    Args:
        step_name: Name of the step

    Returns:
        True if the step comes after step08, False otherwise
    """
    try:
        # Extract step number from step name
        if 'step' in step_name.lower():
            step_parts = step_name.lower().split('step')
            if len(step_parts) > 1:
                step_number_str = step_parts[1]
                # Extract numeric part
                step_number = ''
                for char in step_number_str:
                    if char.isdigit():
                        step_number += char
                    else:
                        break

                if step_number:
                    step_num = int(step_number)
                    return step_num > 8  # Steps after step08

        return False
    except (ValueError, IndexError):
        return False

def auto_regime_aware_logging(
    enable_regime_validation: bool = True,
    enable_fail_fast: bool = True,
    min_regime_samples: int = 100,
    max_regime_imbalance: float = 0.8,
    regime_column: str = 'composite_cluster_id',
    min_data_quality: float = 0.7
):
    """
    Auto-decorator that applies regime-aware logging only to post-HMM steps.

    This decorator automatically detects if a step comes after HMM-based data splitting
    and applies regime-aware logging only to those steps.

    Args:
        enable_regime_validation: Enable regime data validation
        enable_fail_fast: Enable fail-fast behavior
        min_regime_samples: Minimum samples required per regime
        max_regime_imbalance: Maximum allowed regime imbalance ratio
        regime_column: Name of the regime column in data
        min_data_quality: Minimum required data quality score

    Usage:
        @auto_regime_aware_logging()
        async def execute(self, training_input, pipeline_state):
            # Step implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get step name from function or class
            step_name = func.__name__
            if args and hasattr(args[0], '__class__'):
                class_name = args[0].__class__.__name__
                if 'Step' in class_name:
                    step_name = class_name

            # Check if this is a post-HMM step
            if is_post_hmm_step(step_name):
                # Apply regime-aware logging
                regime_decorator = regime_aware_financial_logging(
                    step_name=step_name,
                    enable_regime_validation=enable_regime_validation,
                    enable_fail_fast=enable_fail_fast,
                    min_regime_samples=min_regime_samples,
                    max_regime_imbalance=max_regime_imbalance,
                    regime_column=regime_column,
                    min_data_quality=min_data_quality
                )

                # Apply the decorator and execute
                decorated_func = regime_decorator(func)
                return await decorated_func(*args, **kwargs)
            else:
                # Execute without regime-aware logging
                return await func(*args, **kwargs)

        return wrapper
    return decorator

# Export main decorators and functions
__all__ = [
    'regime_aware_financial_logging',
    'auto_regime_aware_logging',
    'is_post_hmm_step'
]
