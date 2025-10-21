#!/usr/bin/env python3
"""
Mode-Aware Data Fetching Example

This example demonstrates how to use the enhanced artifact_manager and BaseStep
with ares_launcher's mode system (full/blank/light) to control data fetching
based on execution mode.

The system automatically applies the correct lookback period:
- full: 1460 days (4 years)
- blank: 180 days (6 months)  
- light: 20 days

This makes it much easier for all steps to comply with the mode requirements.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.artifact_manager import ArtifactManager
from src.training.steps.base_step import BaseStep
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning


class ExampleStep(BaseStep):
    """Example step that demonstrates mode-aware data fetching."""
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the example step with mode-aware data fetching."""
        symbol = config.get('symbol', 'ETHUSDT')
        timeframe = config.get('timeframe', '15m')
        execution_mode = config.get('execution_mode', 'light')
        
        tprint_info(f"🚀 Running ExampleStep in {execution_mode.upper()} mode")
        
        # Get mode configuration
        mode_config = self._get_mode_config(execution_mode)
        lookback_days = self._get_mode_lookback_days(execution_mode)
        
        tprint_info(f"📊 Mode configuration:")
        tprint_info(f"   - Lookback days: {lookback_days}")
        tprint_info(f"   - Description: {mode_config.get('description', 'N/A')}")
        tprint_info(f"   - Computational intensity: {mode_config.get('computational_intensity', 'N/A')}")
        
        # Load data using mode-aware fetching
        tprint_info(f"📥 Loading data for {symbol} ({timeframe}) in {execution_mode.upper()} mode...")
        
        data = self._load_klines_with_mode(
            symbol=symbol,
            interval=timeframe,
            mode=execution_mode,
            data_type="raw"
        )
        
        if data is not None:
            tprint_success(f"✅ Data loaded successfully!")
            tprint_info(f"   - Records: {len(data)}")
            tprint_info(f"   - Columns: {list(data.columns)}")
            tprint_info(f"   - Date range: {data.index.min()} to {data.index.max()}")
            tprint_info(f"   - Memory usage: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Save the data as an artifact
            self._save_dataframe(data, f"mode_aware_data_{execution_mode}")
            
            return {
                'success': True,
                'records_loaded': len(data),
                'lookback_days': lookback_days,
                'execution_mode': execution_mode,
                'artifacts': [f"mode_aware_data_{execution_mode}"]
            }
        else:
            tprint_warning(f"⚠️ No data loaded for {symbol} ({timeframe}) in {execution_mode.upper()} mode")
            return {
                'success': False,
                'error': f'No data available for {symbol} ({timeframe}) in {execution_mode} mode',
                'lookback_days': lookback_days,
                'execution_mode': execution_mode
            }


async def demonstrate_mode_aware_fetching():
    """Demonstrate mode-aware data fetching across different modes."""
    
    tprint("🎯 Mode-Aware Data Fetching Demonstration")
    tprint("=" * 50)
    
    # Test configuration
    test_config = {
        'symbol': 'ETHUSDT',
        'timeframe': '15m',
        'exchange': 'binance',
        'direction': 'long',
        'model': 'Analyst'
    }
    
    # Test different execution modes
    modes = ['light', 'blank', 'full']
    
    for mode in modes:
        tprint(f"\n🔄 Testing {mode.upper()} mode...")
        tprint("-" * 30)
        
        # Create step with mode-specific config
        config = test_config.copy()
        config['execution_mode'] = mode
        
        try:
            # Create and run the example step
            step = ExampleStep(f"example_step_{mode}", config)
            result = await step.run(config)
            
            if result['success']:
                tprint_success(f"✅ {mode.upper()} mode completed successfully!")
                tprint_info(f"   - Records loaded: {result['records_loaded']}")
                tprint_info(f"   - Lookback days: {result['lookback_days']}")
            else:
                tprint_warning(f"⚠️ {mode.upper()} mode failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            tprint_warning(f"⚠️ {mode.upper()} mode failed with exception: {e}")
    
    tprint("\n🎉 Mode-aware data fetching demonstration completed!")


def demonstrate_artifact_manager_modes():
    """Demonstrate mode-aware functionality directly with ArtifactManager."""
    
    tprint("\n🔧 ArtifactManager Mode-Aware Functionality")
    tprint("=" * 50)
    
    # Create artifact manager
    config = {
        'enable_compression': True,
        'enable_caching': True,
        'enable_memory_optimization': True
    }
    
    artifact_manager = ArtifactManager(config)
    
    # Test different modes
    modes = ['light', 'blank', 'full']
    
    for mode in modes:
        tprint(f"\n📊 Testing {mode.upper()} mode with ArtifactManager...")
        
        # Set execution mode
        artifact_manager.set_execution_mode(mode)
        
        # Get mode configuration
        mode_config = artifact_manager.get_mode_config(mode)
        lookback_days = artifact_manager.get_mode_lookback_days(mode)
        
        tprint_info(f"   - Mode: {mode}")
        tprint_info(f"   - Lookback days: {lookback_days}")
        tprint_info(f"   - Description: {mode_config.get('description', 'N/A')}")
        tprint_info(f"   - Computational intensity: {mode_config.get('computational_intensity', 'N/A')}")
        
        # Test data loading (this will fail if no data is available, but shows the interface)
        try:
            data = artifact_manager.load_klines_with_mode(
                symbol='ETHUSDT',
                interval='15m',
                mode=mode
            )
            
            if data is not None:
                tprint_success(f"   ✅ Data loaded: {len(data)} records")
            else:
                tprint_warning(f"   ⚠️ No data available for {mode} mode")
                
        except Exception as e:
            tprint_warning(f"   ⚠️ Data loading failed: {e}")
    
    tprint("\n🎉 ArtifactManager mode demonstration completed!")


async def main():
    """Main demonstration function."""
    try:
        # Demonstrate BaseStep mode-aware functionality
        await demonstrate_mode_aware_fetching()
        
        # Demonstrate ArtifactManager mode-aware functionality
        demonstrate_artifact_manager_modes()
        
    except Exception as e:
        tprint_warning(f"⚠️ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())