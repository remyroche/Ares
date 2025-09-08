#!/usr/bin/env python3
"""Direct execution of step02_5_sr_optimization without dependency checking."""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.data_collection.data_preparation.step02_5_sr_optimization import SROptimizationStep
from src.config import CONFIG
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards
import logging

async def main():
    """Run step02_5_sr_optimization directly."""
    print("🚀 Running step02_5_sr_optimization directly...")
    
    # Create configuration
    config = CONFIG.copy()
    config.update({
        'symbol': 'ETHUSDT',
        'exchange': 'BINANCE',
        'timeframe': '1m',
        'training_mode': 'light',
        'LIGHT_TRAINING_MODE': 1
    })
    
    # Initialize the step (without decorators to avoid async issues)
    step = SROptimizationStep.__new__(SROptimizationStep)
    step.config = config
    step.step_number = '2_5'
    step.step_name = 'sr_optimization'
    step.logger = system_logger.getChild('SROptimizationStep')
    step.standards = PipelineStandards(step.logger)
    step.sr_optimization_config = config.get('sr_optimization', {'min_touches': 2, 'tolerance_pct': 0.5, 'lookback_periods': 100})
    step.start_time = None
    step.instance_call_tracker = {'method_calls': 0, 'method_history': [], 'performance_metrics': {}}
    
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
        await step.initialize()
        
        # Execute the step
        result = await step.execute(training_input, pipeline_state)
        
        if result.get('success', False):
            print("✅ Step02_5_sr_optimization completed successfully!")
            print(f"📊 Results: {result}")
        else:
            print("❌ Step02_5_sr_optimization failed!")
            print(f"🔍 Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception during step execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
