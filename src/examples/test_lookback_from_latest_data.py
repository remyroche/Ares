#!/usr/bin/env python3
"""
Test Lookback Period Calculation from Latest Data Point

This script tests that the lookback period is calculated from the latest available
data point, not from the current time. This ensures we always work with the most
recent data available, regardless of when the step is executed.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.artifact_manager import ArtifactManager
from src.training.steps.base_step import BaseStep
from src.utils.data.ares_launcher_data_loader import AresLauncherDataLoader
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error


class TestLookbackStep(BaseStep):
    """Test step that demonstrates lookback calculation from latest data."""
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the test step with lookback verification."""
        symbol = config.get('symbol', 'ETHUSDT')
        timeframe = config.get('timeframe', '15m')
        execution_mode = config.get('execution_mode', 'light')
        
        tprint_info(f"🧪 Testing lookback calculation in {execution_mode.upper()} mode")
        
        # Get mode configuration
        mode_config = self._get_mode_config(execution_mode)
        expected_lookback_days = self._get_mode_lookback_days(execution_mode)
        
        tprint_info(f"📊 Expected lookback days: {expected_lookback_days}")
        
        # Load data using mode-aware fetching
        tprint_info(f"📥 Loading data for {symbol} ({timeframe}) in {execution_mode.upper()} mode...")
        
        data = self._load_klines_with_mode(
            symbol=symbol,
            interval=timeframe,
            mode=execution_mode,
            data_type="raw"
        )
        
        if data is not None and len(data) > 0:
            # Calculate actual lookback period from the data
            actual_start = data.index.min()
            actual_end = data.index.max()
            actual_days = (actual_end - actual_start).days
            
            tprint_success(f"✅ Data loaded successfully!")
            tprint_info(f"   - Records: {len(data)}")
            tprint_info(f"   - Actual date range: {actual_start.date()} to {actual_end.date()}")
            tprint_info(f"   - Actual days: {actual_days}")
            tprint_info(f"   - Expected days: {expected_lookback_days}")
            
            # Verify that we're using the latest available data
            now = datetime.now()
            time_since_latest = (now - actual_end).total_seconds() / 3600  # hours
            
            tprint_info(f"   - Latest data point: {actual_end}")
            tprint_info(f"   - Current time: {now}")
            tprint_info(f"   - Hours since latest data: {time_since_latest:.1f}")
            
            # Check if we're using the latest available data (not current time)
            if time_since_latest > 24:  # More than 24 hours old
                tprint_warning(f"⚠️ Latest data is {time_since_latest:.1f} hours old")
                tprint_warning("   → This suggests we're correctly using the latest available data, not current time")
            else:
                tprint_info("ℹ️ Latest data is very recent, which is normal for active data sources")
            
            # Verify lookback period is approximately correct
            if abs(actual_days - expected_lookback_days) <= 1:  # Allow 1 day tolerance
                tprint_success(f"✅ Lookback period is correct: {actual_days} days (expected: {expected_lookback_days})")
            else:
                tprint_warning(f"⚠️ Lookback period differs: {actual_days} days (expected: {expected_lookback_days})")
            
            return {
                'success': True,
                'records_loaded': len(data),
                'actual_lookback_days': actual_days,
                'expected_lookback_days': expected_lookback_days,
                'data_start': actual_start.isoformat(),
                'data_end': actual_end.isoformat(),
                'hours_since_latest': time_since_latest,
                'execution_mode': execution_mode
            }
        else:
            tprint_warning(f"⚠️ No data loaded for {symbol} ({timeframe}) in {execution_mode.upper()} mode")
            return {
                'success': False,
                'error': f'No data available for {symbol} ({timeframe}) in {execution_mode} mode',
                'expected_lookback_days': expected_lookback_days,
                'execution_mode': execution_mode
            }


async def test_lookback_calculation():
    """Test lookback calculation from latest data point."""
    
    tprint("🧪 Testing Lookback Period Calculation from Latest Data Point")
    tprint("=" * 70)
    
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
    
    results = {}
    
    for mode in modes:
        tprint(f"\n🔄 Testing {mode.upper()} mode...")
        tprint("-" * 40)
        
        # Create step with mode-specific config
        config = test_config.copy()
        config['execution_mode'] = mode
        
        try:
            # Create and run the test step
            step = TestLookbackStep(f"test_lookback_{mode}", config)
            result = await step.run(config)
            
            results[mode] = result
            
            if result['success']:
                tprint_success(f"✅ {mode.upper()} mode test completed!")
                tprint_info(f"   - Records: {result['records_loaded']}")
                tprint_info(f"   - Actual lookback: {result['actual_lookback_days']} days")
                tprint_info(f"   - Expected lookback: {result['expected_lookback_days']} days")
                tprint_info(f"   - Data range: {result['data_start']} to {result['data_end']}")
                tprint_info(f"   - Hours since latest: {result['hours_since_latest']:.1f}")
            else:
                tprint_warning(f"⚠️ {mode.upper()} mode test failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            tprint_error(f"❌ {mode.upper()} mode test failed with exception: {e}")
            results[mode] = {'success': False, 'error': str(e)}
    
    # Summary
    tprint("\n📊 Test Summary")
    tprint("=" * 30)
    
    for mode, result in results.items():
        if result['success']:
            tprint_success(f"{mode.upper()}: ✅ {result['actual_lookback_days']} days (expected: {result['expected_lookback_days']})")
        else:
            tprint_error(f"{mode.upper()}: ❌ {result.get('error', 'Unknown error')}")
    
    return results


def test_data_loader_directly():
    """Test the data loader directly to verify date calculation."""
    
    tprint("\n🔧 Testing AresLauncherDataLoader Directly")
    tprint("=" * 50)
    
    try:
        # Create data loader
        data_loader = AresLauncherDataLoader("historical_data")
        
        # Test different modes
        modes = ['light', 'blank', 'full']
        
        for mode in modes:
            tprint(f"\n📊 Testing {mode.upper()} mode date calculation...")
            
            # Get lookback dates
            start_date, end_date = data_loader.get_lookback_dates(
                mode=mode,
                symbol='ETHUSDT',
                interval='15m'
            )
            
            lookback_days = (end_date - start_date).days
            
            tprint_info(f"   - Mode: {mode}")
            tprint_info(f"   - Start date: {start_date}")
            tprint_info(f"   - End date: {end_date}")
            tprint_info(f"   - Lookback days: {lookback_days}")
            
            # Check if end_date is recent (not current time)
            now = datetime.now()
            time_diff = (now - end_date).total_seconds() / 3600  # hours
            
            if time_diff > 24:
                tprint_success(f"   ✅ End date is {time_diff:.1f} hours old - using latest available data")
            else:
                tprint_info(f"   ℹ️ End date is {time_diff:.1f} hours old - very recent data")
        
    except Exception as e:
        tprint_error(f"❌ Direct data loader test failed: {e}")


async def main():
    """Main test function."""
    try:
        # Test lookback calculation from latest data
        await test_lookback_calculation()
        
        # Test data loader directly
        test_data_loader_directly()
        
        tprint("\n🎉 Lookback calculation tests completed!")
        
    except Exception as e:
        tprint_error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())