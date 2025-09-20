#!/usr/bin/env python3
"""
Example: Automatic Step Triggering in Model Training Pipeline

This example demonstrates how to use the automatic step triggering system
where each step automatically triggers the next step when it completes successfully.

The system includes all 5 model training steps:
1. analyst_model_training - Per-regime individual model training with HPO, saving, and metrics
2. analyst_ensemble_training - Per-regime ensemble training with HPO, saving, and metrics
3. tactician_lookback_optimization - Lookback optimization for tactician models
4. tactician_models_training - All-regime individual model training with HPO, saving, and metrics
5. tactician_ensemble_training - All-regime ensemble training with HPO, saving, and metrics
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.model_training.auto_step_trigger import (
    auto_execute_all_model_training_steps,
    auto_execute_from_step,
    get_available_steps
)
from src.utils.tprint import tprint

async def example_1_execute_all_steps():
    """Example 1: Execute all steps automatically from the beginning."""
    tprint("🚀 EXAMPLE 1: Execute all 5 model training steps automatically from the beginning")
    tprint("=" * 80)
    
    # Parameters are now required - no defaults
    symbol = "ETHUSDT"  # Could be any symbol like BTCUSDT, ADAUSDT, etc.
    exchange = "BINANCE"  # Could be BYBIT, KRAKEN, etc.
    timeframe = "1m"  # Could be 5m, 15m, 1h, etc.
    
    result = await auto_execute_all_model_training_steps(
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
    """Example 2: Execute from a specific step (tactician_models_training)."""
    tprint("🚀 EXAMPLE 2: Execute from tactician_models_training step (will trigger steps 4-5)")
    tprint("=" * 80)
    
    # Parameters are now required - no defaults
    symbol = "ETHUSDT"  # Could be any symbol like BTCUSDT, ADAUSDT, etc.
    exchange = "BINANCE"  # Could be BYBIT, KRAKEN, etc.
    timeframe = "1m"  # Could be 5m, 15m, 1h, etc.
    
    result = await auto_execute_from_step(
        step_name='tactician_models_training',
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
    """Example 3: Execute with different timeframes for different model types."""
    tprint("🚀 EXAMPLE 3: Execute with different timeframes for different model types")
    tprint("=" * 80)
    
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    
    # Different timeframes for different model types
    timeframes = {
        "1m": "High-frequency models (Analyst focus)",
        "5m": "Medium-term models (Tactician focus)", 
        "1h": "Long-term models (Strategic focus)"
    }
    
    results = {}
    for timeframe, description in timeframes.items():
        tprint(f"📊 Executing model training for {timeframe} ({description})")
        
        result = await auto_execute_all_model_training_steps(
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

async def example_4_analyst_vs_tactician():
    """Example 4: Execute analyst steps vs tactician steps separately."""
    tprint("🚀 EXAMPLE 4: Execute analyst vs tactician steps separately")
    tprint("=" * 80)
    
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    
    # Execute analyst steps (steps 1-2)
    tprint("📊 Executing Analyst Steps (Per-regime models)")
    analyst_result = await auto_execute_from_step(
        step_name='analyst_model_training',
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir="historical_data",
        force_rerun=True,
        config={'validation_enabled': True}
    )
    
    tprint(f"   ✅ Analyst Steps: {analyst_result['successful_steps']}/{analyst_result['total_steps_executed']} steps completed")
    
    # Execute tactician steps (steps 3-5)
    tprint("📊 Executing Tactician Steps (All-regime models)")
    tactician_result = await auto_execute_from_step(
        step_name='tactician_lookback_optimization',
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir="historical_data",
        force_rerun=True,
        config={'validation_enabled': True}
    )
    
    tprint(f"   ✅ Tactician Steps: {tactician_result['successful_steps']}/{tactician_result['total_steps_executed']} steps completed")
    
    return {
        'analyst': analyst_result,
        'tactician': tactician_result
    }

def show_available_steps():
    """Show all available steps in order."""
    tprint("📋 AVAILABLE MODEL TRAINING STEPS:")
    tprint("=" * 80)
    
    steps = get_available_steps()
    step_descriptions = {
        'analyst_model_training': 'Per-regime individual model training with HPO, saving, and metrics',
        'analyst_ensemble_training': 'Per-regime ensemble training with HPO, saving, and metrics',
        'tactician_lookback_optimization': 'Lookback optimization for tactician models',
        'tactician_models_training': 'All-regime individual model training with HPO, saving, and metrics',
        'tactician_ensemble_training': 'All-regime ensemble training with HPO, saving, and metrics'
    }
    
    for i, step in enumerate(steps, 1):
        description = step_descriptions.get(step, 'No description available')
        tprint(f"   {i:2d}. {step:<35} - {description}")
    
    tprint("=" * 80)

async def main():
    """Run all examples."""
    tprint("🎯 AUTOMATIC STEP TRIGGERING EXAMPLES - MODEL TRAINING")
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
    
    # Example 4: Analyst vs Tactician
    # tprint("\n" + "="*80)
    # await example_4_analyst_vs_tactician()
    
    tprint("\n🎉 EXAMPLES COMPLETED")
    tprint("=" * 80)
    tprint("To run the examples, uncomment the desired example calls in main()")
    tprint("=" * 80)

if __name__ == '__main__':
    asyncio.run(main())
