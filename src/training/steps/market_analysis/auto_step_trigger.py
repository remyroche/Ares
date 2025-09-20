"""
Automatic Step Trigger for Market Analysis Pipeline

This module provides a simple interface for automatically triggering all market analysis steps
when one step completes. It uses the existing sub_pipeline.py infrastructure.

Usage:
    # Execute all steps automatically from the beginning
    result = await auto_execute_all_market_analysis_steps(config)
    
    # Execute from a specific step (will trigger all subsequent steps)
    result = await auto_execute_from_step('hmm_clustering', config)
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from .sub_pipeline import MarketAnalysisSubPipeline, SubPipelineConfig, ExecutionMode
from src.utils.logger import system_logger

logger = system_logger.getChild('AutoStepTrigger')

async def auto_execute_all_market_analysis_steps(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str = "historical_data",
    force_rerun: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Automatically execute all 11 market analysis steps from the beginning.
    
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
    logger.info('🚀 Starting automatic execution of all market analysis steps')
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
    sub_pipeline = MarketAnalysisSubPipeline(sub_config)
    result = await sub_pipeline.execute_all_steps_from_start(sub_config)
    
    logger.info('🎉 Automatic execution completed')
    logger.info(f'✅ Successful steps: {result["successful_steps"]}/{result["total_steps_executed"]}')
    logger.info(f'⏱️ Total execution time: {result["total_execution_time"]:.2f} seconds')
    
    return result

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
    Automatically execute market analysis steps starting from a specific step.
    
    When the specified step completes successfully, it automatically triggers all subsequent steps.
    
    Args:
        step_name: Name of the step to start from (e.g., 'hmm_clustering')
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
    sub_pipeline = MarketAnalysisSubPipeline(sub_config)
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
    Get list of all available market analysis steps.
    
    Returns:
        List of step names in execution order
    """
    return [
        'sr_parameter_optimization',
        'sr_detection', 
        'sr_clustering',
        'hmm_regime_discovery',
        'hmm_clustering',
        'hmm_models_training',
        'hmm_ensemble_training',
        'regime_data_splitting',
        'multi_horizon_profit_labeler',
        'feature_lookback_optimization',
        'pid_based_feature_generation'
    ]

async def main():
    """Example usage of automatic step triggering."""
    # Example 1: Execute all steps from the beginning
    logger.info('Example 1: Executing all steps from the beginning')
    result = await auto_execute_all_market_analysis_steps(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        force_rerun=True
    )
    
    if result['success']:
        logger.info('✅ All steps completed successfully!')
    else:
        logger.info('❌ Some steps failed')
    
    # Example 2: Execute from a specific step
    logger.info('Example 2: Executing from hmm_clustering step')
    result = await auto_execute_from_step(
        step_name='hmm_clustering',
        symbol="ETHUSDT", 
        exchange="BINANCE",
        timeframe="1m",
        force_rerun=True
    )
    
    if result['success']:
        logger.info('✅ Steps from hmm_clustering completed successfully!')
    else:
        logger.info('❌ Some steps failed')

if __name__ == '__main__':
    asyncio.run(main())
