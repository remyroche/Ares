#!/usr/bin/env python3
"""
Example: Automatic Step Triggering in Market Analysis Pipeline

This example demonstrates how to use the automatic step triggering system
where each step automatically triggers the next step when it completes successfully.

The system includes all 11 market analysis steps:
1. sr_parameter_optimization - Optimize SR detection levels
2. sr_detection - Detect Support/Resistance levels
3. sr_clustering - Generate SR clusters
4. hmm_regime_discovery - Discover market regimes
5. hmm_clustering - HMM-based regime clustering
6. hmm_models_training - Base models training, HPO, saving, metrics
7. hmm_ensemble_training - Meta-model, HPO, saving, metrics
8. regime_data_splitting - Tag data by regimes
9. multi_horizon_profit_labeler - Apply triple barrier method
10. feature_lookback_optimization - Optimize feature lookback periods
11. pid_based_feature_generation - Cross timeframe interaction features
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.market_analysis.auto_step_trigger import (
    auto_execute_all_market_analysis_steps,
    auto_execute_from_step,
    get_available_steps
)
from src.training.steps.market_analysis.enhanced_market_analysis_orchestrator import (
    run_auto_triggering_market_analysis_pipeline
)
from src.utils.tprint import tprint

async def example_1_execute_all_steps():
    """Example 1: Execute all steps automatically from the beginning."""
    tprint("🚀 EXAMPLE 1: Execute all 11 steps automatically from the beginning")
    tprint("=" * 80)
    
    # Parameters are now required - no defaults
    symbol = "ETHUSDT"  # Could be any symbol like BTCUSDT, ADAUSDT, etc.
    exchange = "BINANCE"  # Could be BYBIT, KRAKEN, etc.
    timeframe = "1m"  # Could be 5m, 15m, 1h, etc.
    
    result = await auto_execute_all_market_analysis_steps(
        symbol=symbol,
        exchange=exchange, 
        timeframe=timeframe,
        data_dir="historical_data",
        force_rerun=True,
        config={
            'parallel_processing': True,
            'validation_enabled': True,
            'monitoring_enabled': True
        }
    )
    
    tprint("📊 RESULTS:")
    tprint(f"   ✅ Success: {result['success']}")
    tprint(f"   📈 Steps executed: {result['total_steps_executed']}")
    tprint(f"   ✅ Successful steps: {result['successful_steps']}")
    tprint(f"   ❌ Failed steps: {result['failed_steps']}")
    tprint(f"   ⏱️ Total time: {result['total_execution_time']:.2f} seconds")
    
    return result

async def example_2_execute_from_step():
    """Example 2: Execute from a specific step (hmm_clustering)."""
    tprint("🚀 EXAMPLE 2: Execute from hmm_clustering step (will trigger steps 5-11)")
    tprint("=" * 80)
    
    # Parameters are now required - no defaults
    symbol = "ETHUSDT"  # Could be any symbol like BTCUSDT, ADAUSDT, etc.
    exchange = "BINANCE"  # Could be BYBIT, KRAKEN, etc.
    timeframe = "1m"  # Could be 5m, 15m, 1h, etc.
    
    result = await auto_execute_from_step(
        step_name='hmm_clustering',
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe, 
        data_dir="historical_data",
        force_rerun=True,
        config={
            'parallel_processing': True,
            'validation_enabled': True
        }
    )
    
    tprint("📊 RESULTS:")
    tprint(f"   ✅ Success: {result['success']}")
    tprint(f"   🎯 Starting step: {result['starting_step']}")
    tprint(f"   📈 Steps executed: {result['total_steps_executed']}")
    tprint(f"   ✅ Successful steps: {result['successful_steps']}")
    tprint(f"   ❌ Failed steps: {result['failed_steps']}")
    tprint(f"   ⏱️ Total time: {result['total_execution_time']:.2f} seconds")
    
    return result

async def example_3_using_orchestrator():
    """Example 3: Using the enhanced orchestrator with auto-triggering."""
    tprint("🚀 EXAMPLE 3: Using enhanced orchestrator with auto-triggering")
    tprint("=" * 80)
    
    result = await run_auto_triggering_market_analysis_pipeline(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        data_dir="historical_data",
        start_from_step=None,  # Start from beginning
        force_rerun=True,
        parallel_processing=True,
        validation_enabled=True,
        monitoring_enabled=True
    )
    
    tprint("📊 RESULTS:")
    tprint(f"   ✅ Success: {result['success']}")
    tprint(f"   📈 Steps executed: {result['total_steps_executed']}")
    tprint(f"   ✅ Successful steps: {result['successful_steps']}")
    tprint(f"   ❌ Failed steps: {result['failed_steps']}")
    tprint(f"   ⏱️ Total time: {result['total_execution_time']:.2f} seconds")
    
    return result

async def example_4_execute_from_specific_step():
    """Example 4: Execute from a specific step using orchestrator."""
    tprint("🚀 EXAMPLE 4: Execute from regime_data_splitting using orchestrator")
    tprint("=" * 80)
    
    result = await run_auto_triggering_market_analysis_pipeline(
        symbol="ETHUSDT",
        exchange="BINANCE", 
        timeframe="1m",
        data_dir="historical_data",
        start_from_step="regime_data_splitting",  # Start from step 8
        force_rerun=True,
        parallel_processing=True,
        validation_enabled=True
    )
    
    tprint("📊 RESULTS:")
    tprint(f"   ✅ Success: {result['success']}")
    tprint(f"   📈 Steps executed: {result['total_steps_executed']}")
    tprint(f"   ✅ Successful steps: {result['successful_steps']}")
    tprint(f"   ❌ Failed steps: {result['failed_steps']}")
    tprint(f"   ⏱️ Total time: {result['total_execution_time']:.2f} seconds")
    
    return result

def show_available_steps():
    """Show all available steps in order."""
    tprint("📋 AVAILABLE MARKET ANALYSIS STEPS:")
    tprint("=" * 80)
    
    steps = get_available_steps()
    step_descriptions = {
        'sr_parameter_optimization': 'Optimize SR detection levels',
        'sr_detection': 'Detect Support/Resistance levels',
        'sr_clustering': 'Generate SR clusters',
        'hmm_regime_discovery': 'Discover market regimes',
        'hmm_clustering': 'HMM-based regime clustering',
        'hmm_models_training': 'Base models training, HPO, saving, metrics',
        'hmm_ensemble_training': 'Meta-model, HPO, saving, metrics',
        'regime_data_splitting': 'Tag data by regimes',
        'multi_horizon_profit_labeler': 'Apply triple barrier method',
        'feature_lookback_optimization': 'Optimize feature lookback periods',
        'pid_based_feature_generation': 'Cross timeframe interaction features'
    }
    
    for i, step in enumerate(steps, 1):
        description = step_descriptions.get(step, 'No description available')
        tprint(f"   {i:2d}. {step:<35} - {description}")
    
    tprint("=" * 80)

async def main():
    """Run all examples."""
    tprint("🎯 AUTOMATIC STEP TRIGGERING EXAMPLES")
    tprint("=" * 80)
    tprint("This demonstrates how each step automatically triggers the next step")
    tprint("when it completes successfully, ensuring seamless execution.")
    tprint("=" * 80)
    
    # Show available steps
    show_available_steps()
    
    # Run examples (uncomment the ones you want to test)
    
    # Example 1: Execute all steps from beginning
    # tprint("\n" + "="*80)
    # await example_1_execute_all_steps()
    
    # Example 2: Execute from specific step
    # tprint("\n" + "="*80) 
    # await example_2_execute_from_step()
    
    # Example 3: Using orchestrator
    # tprint("\n" + "="*80)
    # await example_3_using_orchestrator()
    
    # Example 4: Execute from specific step using orchestrator
    # tprint("\n" + "="*80)
    # await example_4_execute_from_specific_step()
    
    tprint("\n🎉 EXAMPLES COMPLETED")
    tprint("=" * 80)
    tprint("To run the examples, uncomment the desired example calls in main()")
    tprint("=" * 80)

if __name__ == '__main__':
    asyncio.run(main())
