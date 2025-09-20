"""
Automatic Step Trigger for Backtesting Pipeline

This module provides a simple interface for automatically triggering all backtesting steps
when one step completes. It uses the existing sub_pipeline.py infrastructure.

Usage:
    # Execute all steps automatically from the beginning
    result = await auto_execute_all_backtesting_steps(symbol, exchange, timeframe)
    
    # Execute from a specific step (will trigger all subsequent steps)
    result = await auto_execute_from_step('walk_forward_validation', symbol, exchange, timeframe)
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from .sub_pipeline import BacktestingSubPipeline, SubPipelineConfig, ExecutionMode
from src.utils.logger import system_logger

logger = system_logger.getChild('BacktestingAutoStepTrigger')

async def auto_execute_all_backtesting_steps(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str = "historical_data",
    force_rerun: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Automatically execute all 7 backtesting steps from the beginning.
    
    When each step completes successfully, it automatically triggers the next step.
    
    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT', 'BTCUSDT')
        exchange: Exchange name (e.g., 'BINANCE', 'BYBIT')
        timeframe: Data timeframe (e.g., '1m', '5m', '1h')
        data_dir: Data directory path (default: historical_data)
        force_rerun: Whether to force rerun existing artifacts (default: False)
        config: Optional configuration dictionary
        
    Returns:
        Dict with execution results and summary
    """
    logger.info('🚀 Starting automatic execution of all backtesting steps')
    logger.info(f'📊 Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}')
    
    # Create sub-pipeline configuration
    if config is None:
        config = {}
    
    sub_config = SubPipelineConfig(
        mode=ExecutionMode.FULL,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun,
        single_stage_only=False,  # Enable automatic triggering
        **config
    )
    
    # Create and execute sub-pipeline
    sub_pipeline = BacktestingSubPipeline(sub_config)
    result = await sub_pipeline.execute_sub_pipeline_with_next('basic_backtesting_pre', sub_config)
    
    # Get execution summary
    summary = sub_pipeline.get_execution_summary()
    
    logger.info('🎉 Automatic execution completed')
    logger.info(f'✅ Successful steps: {summary["successful_sub_pipelines"]}/{summary["total_sub_pipelines"]}')
    logger.info(f'⏱️ Total execution time: {summary["total_execution_time"]:.2f} seconds')
    
    return {
        'success': result.success,
        'execution_summary': summary,
        'total_steps_executed': summary['total_sub_pipelines'],
        'successful_steps': summary['successful_sub_pipelines'],
        'failed_steps': summary['failed_sub_pipelines'],
        'total_execution_time': summary['total_execution_time']
    }

async def auto_execute_from_step(
    step_name: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str = "historical_data",
    force_rerun: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Automatically execute backtesting steps starting from a specific step.
    
    When the specified step completes successfully, it automatically triggers all subsequent steps.
    
    Args:
        step_name: Name of the step to start from (e.g., 'walk_forward_validation')
        symbol: Trading symbol (e.g., 'ETHUSDT', 'BTCUSDT')
        exchange: Exchange name (e.g., 'BINANCE', 'BYBIT')
        timeframe: Data timeframe (e.g., '1m', '5m', '1h')
        data_dir: Data directory path (default: historical_data)
        force_rerun: Whether to force rerun existing artifacts (default: False)
        config: Optional configuration dictionary
        
    Returns:
        Dict with execution results and summary
    """
    logger.info(f'🚀 Starting automatic execution from step: {step_name}')
    logger.info(f'📊 Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}')
    
    # Create sub-pipeline configuration
    if config is None:
        config = {}
    
    sub_config = SubPipelineConfig(
        mode=ExecutionMode.FULL,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun,
        single_stage_only=False,  # Enable automatic triggering
        **config
    )
    
    # Create and execute sub-pipeline from specified step
    sub_pipeline = BacktestingSubPipeline(sub_config)
    result = await sub_pipeline.execute_sub_pipeline_with_next(step_name, sub_config)
    
    # Get execution summary
    summary = sub_pipeline.get_execution_summary()
    
    logger.info('🎉 Automatic execution completed')
    logger.info(f'✅ Successful steps: {summary["successful_sub_pipelines"]}/{summary["total_sub_pipelines"]}')
    logger.info(f'⏱️ Total execution time: {summary["total_execution_time"]:.2f} seconds')
    
    return {
        'success': result.success,
        'starting_step': step_name,
        'execution_summary': summary,
        'total_steps_executed': summary['total_sub_pipelines'],
        'successful_steps': summary['successful_sub_pipelines'],
        'failed_steps': summary['failed_sub_pipelines'],
        'total_execution_time': summary['total_execution_time']
    }

def get_available_steps() -> list:
    """
    Get list of all available backtesting steps.
    
    Returns:
        List of step names in execution order
    """
    return [
        'basic_backtesting_pre',
        'final_parameters_optimization',
        'basic_backtesting_post',
        'walk_forward_validation',
        'monte_carlo_simulation',
        'ab_testing',
        'reporting'
    ]

async def main():
    """Example usage of automatic step triggering."""
    # Example 1: Execute all steps from the beginning
    logger.info('Example 1: Executing all backtesting steps from the beginning')
    result = await auto_execute_all_backtesting_steps(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        force_rerun=True
    )
    
    if result['success']:
        logger.info('✅ All backtesting steps completed successfully!')
    else:
        logger.info('❌ Some backtesting steps failed')
    
    # Example 2: Execute from a specific step
    logger.info('Example 2: Executing from walk_forward_validation step')
    result = await auto_execute_from_step(
        step_name='walk_forward_validation',
        symbol="ETHUSDT", 
        exchange="BINANCE",
        timeframe="1m",
        force_rerun=True
    )
    
    if result['success']:
        logger.info('✅ Backtesting steps from walk_forward_validation completed successfully!')
    else:
        logger.info('❌ Some backtesting steps failed')

if __name__ == '__main__':
    asyncio.run(main())
