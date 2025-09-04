"""Regime Continuity Decorator for Pipeline Steps.

This decorator ensures that all pipeline steps maintain regime continuity
and process data on a per-regime basis when appropriate.
"""

import asyncio
import functools
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.logger import getChild as get_logger
from src.training.steps.regime_continuity_manager import (
    regime_continuity_manager,
    RegimeStatus,
    StepRegimeContext
)
from src.training.steps.regime_handler import regime_handler
from src.core.decorators import traced, handles_errors


logger = get_logger('RegimeContinuityDecorator')


def ensure_regime_continuity(
    step_name: str,
    per_regime_required: bool = True,
    regime_aware: bool = True
):
    """Decorator to ensure regime continuity in pipeline steps.
    
    Args:
        step_name: Name of the step
        per_regime_required: Whether the step must process each regime separately
        regime_aware: Whether the step should be aware of regime context
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract common parameters
            symbol = kwargs.get('symbol') or (args[0] if len(args) > 0 else None)
            exchange = kwargs.get('exchange') or (args[1] if len(args) > 1 else None)
            timeframe = kwargs.get('timeframe') or (args[2] if len(args) > 2 else None)
            data_dir = kwargs.get('data_dir') or (args[3] if len(args) > 3 else None)
            
            if not all([symbol, exchange, timeframe, data_dir]):
                logger.error(f"❌ Missing required parameters for {step_name}")
                return False
            
            logger.info(f"🚀 Starting {step_name} with regime continuity")
            
            try:
                # Initialize regime continuity if not already done
                if not regime_continuity_manager.regime_metadata:
                    success = await regime_continuity_manager.initialize_regime_continuity(
                        symbol, exchange, timeframe, data_dir
                    )
                    if not success:
                        logger.error(f"❌ Failed to initialize regime continuity for {step_name}")
                        return False
                
                # Check if step should use per-regime processing
                if per_regime_required and regime_aware:
                    return await _execute_per_regime_step(
                        func, step_name, symbol, exchange, timeframe, data_dir, args, kwargs
                    )
                else:
                    return await _execute_standard_step(
                        func, step_name, symbol, exchange, timeframe, data_dir, args, kwargs
                    )
                    
            except Exception as e:
                logger.exception(f"❌ Error in {step_name} with regime continuity: {e}")
                return False
        
        return wrapper
    return decorator


async def _execute_per_regime_step(
    func: Callable,
    step_name: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str,
    args: tuple,
    kwargs: dict
) -> bool:
    """Execute a step with per-regime processing.
    
    Args:
        func: The step function
        step_name: Name of the step
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory
        args: Function arguments
        kwargs: Function keyword arguments
        
    Returns:
        True if successful
    """
    try:
        # Get regime IDs
        regime_ids = list(regime_continuity_manager.regime_metadata.keys())
        if not regime_ids:
            logger.error(f"❌ No regimes found for {step_name}")
            return False
        
        logger.info(f"📊 Processing {len(regime_ids)} regimes for {step_name}")
        
        # Process each regime
        regime_results = {}
        successful_regimes = 0
        
        for regime_id in regime_ids:
            logger.info(f"🔄 Processing regime {regime_id} for {step_name}")
            
            try:
                # Update step status to in progress
                await regime_continuity_manager.update_step_status(
                    step_name, regime_id, RegimeStatus.IN_PROGRESS
                )
                
                # Get regime context
                context = await regime_continuity_manager.get_regime_context(step_name, regime_id)
                if not context:
                    logger.error(f"❌ No context found for regime {regime_id} in {step_name}")
                    await regime_continuity_manager.update_step_status(
                        step_name, regime_id, RegimeStatus.FAILED,
                        error_message="No context found"
                    )
                    continue
                
                # Prepare regime-specific arguments
                regime_kwargs = kwargs.copy()
                regime_kwargs.update({
                    'regime_id': regime_id,
                    'regime_context': context,
                    'per_regime': True
                })
                
                # Execute the step function for this regime
                result = await func(*args, **regime_kwargs)
                
                if result:
                    regime_results[regime_id] = result
                    successful_regimes += 1
                    
                    # Update step status to completed
                    await regime_continuity_manager.update_step_status(
                        step_name, regime_id, RegimeStatus.COMPLETED,
                        metadata={'result': str(result) if result else None}
                    )
                    
                    logger.info(f"✅ Completed regime {regime_id} for {step_name}")
                else:
                    # Update step status to failed
                    await regime_continuity_manager.update_step_status(
                        step_name, regime_id, RegimeStatus.FAILED,
                        error_message="Step function returned False"
                    )
                    
                    logger.error(f"❌ Failed regime {regime_id} for {step_name}")
                
            except Exception as e:
                logger.exception(f"❌ Error processing regime {regime_id} for {step_name}: {e}")
                await regime_continuity_manager.update_step_status(
                    step_name, regime_id, RegimeStatus.FAILED,
                    error_message=str(e)
                )
        
        # Validate continuity
        continuity_valid = await regime_continuity_manager.validate_regime_continuity(
            step_name, symbol, exchange, timeframe, data_dir
        )
        
        if not continuity_valid:
            logger.warning(f"⚠️ Regime continuity validation failed for {step_name}")
        
        # Aggregate results if needed
        if regime_results:
            await _aggregate_regime_results(
                step_name, regime_results, symbol, exchange, timeframe, data_dir
            )
        
        success_rate = successful_regimes / len(regime_ids)
        logger.info(f"📊 {step_name} completed: {successful_regimes}/{len(regime_ids)} regimes successful ({success_rate:.1%})")
        
        return success_rate >= 0.8  # Require at least 80% success rate
        
    except Exception as e:
        logger.exception(f"❌ Error in per-regime execution for {step_name}: {e}")
        return False


async def _execute_standard_step(
    func: Callable,
    step_name: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str,
    args: tuple,
    kwargs: dict
) -> bool:
    """Execute a step with standard processing (not per-regime).
    
    Args:
        func: The step function
        step_name: Name of the step
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory
        args: Function arguments
        kwargs: Function keyword arguments
        
    Returns:
        True if successful
    """
    try:
        logger.info(f"🔄 Executing {step_name} with standard processing")
        
        # Add regime awareness if needed
        if 'regime_aware' in kwargs:
            kwargs['regime_aware'] = True
        
        # Execute the step function
        result = await func(*args, **kwargs)
        
        if result:
            logger.info(f"✅ Completed {step_name} with standard processing")
        else:
            logger.error(f"❌ Failed {step_name} with standard processing")
        
        return result
        
    except Exception as e:
        logger.exception(f"❌ Error in standard execution for {step_name}: {e}")
        return False


async def _aggregate_regime_results(
    step_name: str,
    regime_results: Dict[int, Any],
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str
) -> None:
    """Aggregate results from per-regime processing.
    
    Args:
        step_name: Name of the step
        regime_results: Results from each regime
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory
    """
    try:
        logger.info(f"🔄 Aggregating results for {step_name}")
        
        # Create aggregated output path
        training_dir = Path(data_dir) / 'training'
        aggregated_path = training_dir / f'{exchange}_{symbol}_{timeframe}_{step_name}_aggregated.parquet'
        
        # Handle different types of results
        if all(isinstance(result, pd.DataFrame) for result in regime_results.values()):
            # Aggregate DataFrames
            dfs = []
            for regime_id, df in regime_results.items():
                if df is not None and not df.empty:
                    df_copy = df.copy()
                    df_copy['source_regime_id'] = regime_id
                    dfs.append(df_copy)
            
            if dfs:
                aggregated_df = pd.concat(dfs, ignore_index=True)
                aggregated_df = aggregated_df.sort_values('timestamp').reset_index(drop=True)
                aggregated_df.to_parquet(aggregated_path, index=False)
                logger.info(f"✅ Aggregated {len(dfs)} regime DataFrames: {aggregated_path}")
        
        elif all(isinstance(result, dict) for result in regime_results.values()):
            # Aggregate dictionaries
            aggregated_dict = {
                'step_name': step_name,
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'aggregated_at': datetime.now().isoformat(),
                'regime_results': regime_results,
                'total_regimes': len(regime_results),
                'successful_regimes': len([r for r in regime_results.values() if r is not None])
            }
            
            import json
            aggregated_json_path = training_dir / f'{exchange}_{symbol}_{timeframe}_{step_name}_aggregated.json'
            with open(aggregated_json_path, 'w') as f:
                json.dump(aggregated_dict, f, indent=2, default=str)
            
            logger.info(f"✅ Aggregated {len(regime_results)} regime dictionaries: {aggregated_json_path}")
        
        else:
            logger.warning(f"⚠️ Unknown result types for {step_name}, skipping aggregation")
        
    except Exception as e:
        logger.exception(f"❌ Error aggregating results for {step_name}: {e}")


def get_regime_aware_step_function(
    step_name: str,
    per_regime_required: bool = True,
    regime_aware: bool = True
) -> Callable:
    """Get a regime-aware version of a step function.
    
    Args:
        step_name: Name of the step
        per_regime_required: Whether per-regime processing is required
        regime_aware: Whether the step should be regime-aware
        
    Returns:
        Decorated step function
    """
    def decorator(func: Callable) -> Callable:
        return ensure_regime_continuity(
            step_name=step_name,
            per_regime_required=per_regime_required,
            regime_aware=regime_aware
        )(func)
    
    return decorator


# Convenience decorators for different step types
def per_regime_step(step_name: str):
    """Decorator for steps that must process each regime separately."""
    return ensure_regime_continuity(
        step_name=step_name,
        per_regime_required=True,
        regime_aware=True
    )


def regime_aware_step(step_name: str):
    """Decorator for steps that should be aware of regime context but don't need per-regime processing."""
    return ensure_regime_continuity(
        step_name=step_name,
        per_regime_required=False,
        regime_aware=True
    )


def standard_step(step_name: str):
    """Decorator for steps that don't need regime awareness."""
    return ensure_regime_continuity(
        step_name=step_name,
        per_regime_required=False,
        regime_aware=False
    )