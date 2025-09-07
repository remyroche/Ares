#!/usr/bin/env python3
"""Run step02_5_sr_optimization by bypassing dependency checking."""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the step directly
from src.training.steps.data_collection.data_preparation.step02_5_sr_optimization import SROptimizationStep
from src.config import CONFIG

async def run_step02_5():
    """Run step02_5_sr_optimization with minimal setup."""
    print("🚀 Running step02_5_sr_optimization with bypassed dependencies...")
    
    # Create configuration
    config = CONFIG.copy()
    config.update({
        'symbol': 'ETHUSDT',
        'exchange': 'BINANCE',
        'timeframe': '1m',
        'training_mode': 'light',
        'LIGHT_TRAINING_MODE': 1
    })
    
    # Create step instance
    step = SROptimizationStep(config)
    
    # Prepare training input
    training_input = {
        'symbol': 'ETHUSDT',
        'exchange': 'BINANCE',
        'timeframe': '1m',
        'data_dir': 'data_cache'
    }
    
    # Prepare pipeline state (simulate completed previous steps)
    pipeline_state = {
        'data_info': {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m'
        }
    }
    
    try:
        # Initialize the step
        print("🔧 Initializing step...")
        await step.initialize()
        
        # Execute the step
        print("🔄 Executing step...")
        result = await step.execute(training_input, pipeline_state)
        
        if result.get('success', False):
            print("✅ Step02_5_sr_optimization completed successfully!")
            print(f"📊 Results summary:")
            print(f"   - SR levels found: {len(result.get('sr_levels', []))}")
            print(f"   - Optimization results: {result.get('sr_optimization_results', {})}")
            print(f"   - Execution time: {result.get('execution_time', 0):.2f}s")
        else:
            print("❌ Step02_5_sr_optimization failed!")
            print(f"🔍 Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception during step execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_step02_5())
