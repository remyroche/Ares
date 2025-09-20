#!/usr/bin/env python3
"""
Example: Automatic Step Triggering in Backtesting Pipeline

This example demonstrates how to use the automatic step triggering system
where each step automatically triggers the next step when it completes successfully.

The system includes all 7 backtesting steps:
1. basic_backtesting_pre - Pre-optimization baseline backtesting
2. final_parameters_optimization - System-wide parameter optimization
3. basic_backtesting_post - Post-optimization comparison backtesting
4. walk_forward_validation - Walk-forward backtesting
5. monte_carlo_simulation - Monte Carlo backtesting
6. ab_testing - A/B testing for strategies
7. reporting - Comprehensive reporting
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.backtesting.auto_step_trigger import (
    auto_execute_all_backtesting_steps,
    auto_execute_from_step,
    get_available_steps
)
from src.utils.tprint import tprint

async def example_1_execute_all_steps():
    """Example 1: Execute all steps automatically from the beginning."""
    tprint("🚀 EXAMPLE 1: Execute all 7 backtesting steps automatically from the beginning")
    tprint("=" * 80)
    
    # Parameters are now required - no defaults
    symbol = "ETHUSDT"  # Could be any symbol like BTCUSDT, ADAUSDT, etc.
    exchange = "BINANCE"  # Could be BYBIT, KRAKEN, etc.
    timeframe = "1m"  # Could be 5m, 15m, 1h, etc.
    
    result = await auto_execute_all_backtesting_steps(
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
    """Example 2: Execute from a specific step (walk_forward_validation)."""
    tprint("🚀 EXAMPLE 2: Execute from walk_forward_validation step (will trigger steps 4-7)")
    tprint("=" * 80)
    
    # Parameters are now required - no defaults
    symbol = "ETHUSDT"  # Could be any symbol like BTCUSDT, ADAUSDT, etc.
    exchange = "BINANCE"  # Could be BYBIT, KRAKEN, etc.
    timeframe = "1m"  # Could be 5m, 15m, 1h, etc.
    
    result = await auto_execute_from_step(
        step_name='walk_forward_validation',
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

async def example_3_different_timeframes():
    """Example 3: Execute with different timeframes for different purposes."""
    tprint("🚀 EXAMPLE 3: Execute with different timeframes")
    tprint("=" * 80)
    
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    
    # Different timeframes for different analysis purposes
    timeframes = {
        "1m": "High-frequency analysis",
        "5m": "Medium-term analysis", 
        "15m": "Short-term analysis",
        "1h": "Long-term analysis"
    }
    
    results = {}
    for timeframe, description in timeframes.items():
        tprint(f"📊 Executing backtesting for {timeframe} ({description})")
        
        result = await auto_execute_all_backtesting_steps(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir="historical_data",
            force_rerun=True,
            config={'validation_enabled': True}
        )
        
        results[timeframe] = result
        tprint(f"   ✅ {timeframe}: {result['successful_steps']}/{result['total_steps_executed']} steps completed")
    
    tprint("📊 ALL TIMEFRAMES COMPLETED:")
    for timeframe, result in results.items():
        tprint(f"   {timeframe}: {result['successful_steps']}/{result['total_steps_executed']} steps, {result['total_execution_time']:.1f}s")
    
    return results

def show_available_steps():
    """Show all available steps in order."""
    tprint("📋 AVAILABLE BACKTESTING STEPS:")
    tprint("=" * 80)
    
    steps = get_available_steps()
    step_descriptions = {
        'basic_backtesting_pre': 'Pre-optimization baseline backtesting',
        'final_parameters_optimization': 'System-wide parameter optimization',
        'basic_backtesting_post': 'Post-optimization comparison backtesting',
        'walk_forward_validation': 'Walk-forward backtesting',
        'monte_carlo_simulation': 'Monte Carlo backtesting',
        'ab_testing': 'A/B testing for strategies',
        'reporting': 'Comprehensive reporting'
    }
    
    for i, step in enumerate(steps, 1):
        description = step_descriptions.get(step, 'No description available')
        tprint(f"   {i:2d}. {step:<35} - {description}")
    
    tprint("=" * 80)

async def main():
    """Run all examples."""
    tprint("🎯 AUTOMATIC STEP TRIGGERING EXAMPLES - BACKTESTING")
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
    
    # Example 3: Different timeframes
    # tprint("\n" + "="*80)
    # await example_3_different_timeframes()
    
    tprint("\n🎉 EXAMPLES COMPLETED")
    tprint("=" * 80)
    tprint("To run the examples, uncomment the desired example calls in main()")
    tprint("=" * 80)

if __name__ == '__main__':
    asyncio.run(main())
