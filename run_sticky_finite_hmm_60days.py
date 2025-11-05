#!/usr/bin/env python3
"""
Run Sticky Finite HMM with 60 days of ETHUSDT data.
This script loads 60 days of data and runs the complete pipeline.
"""

import sys
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, 'src')

from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_regime_discovery_step import (
    StickyFiniteHMMRegimeDiscoveryStep
)
from src.utils.data_loader import DataLoader

def filter_data_by_days(data: pd.DataFrame, days: int = 60) -> pd.DataFrame:
    """Filter DataFrame to last N days."""
    if data is None or data.empty:
        return data
    
    # Ensure timestamp column exists and is datetime
    if 'timestamp' not in data.columns:
        return data
    
    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Sort by timestamp
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    # Filter to last N days
    cutoff_date = data['timestamp'].max() - timedelta(days=days)
    filtered_data = data[data['timestamp'] >= cutoff_date].reset_index(drop=True)
    
    print(f"📊 Filtered to {days} days: {len(filtered_data)} rows (from {len(data)} total)")
    return filtered_data

async def run_sticky_finite_hmm_60_days():
    """Run Sticky Finite HMM with 60 days of data."""
    
    print("=" * 80)
    print("STICKY FINITE HMM REGIME DISCOVERY - 60 DAYS")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print(f"Data: ETHUSDT 1h - Last 60 days")
    print("=" * 80)
    
    # Load data first to filter it
    print("\n📂 Loading ETHUSDT data...")
    data_loader = DataLoader()
    market_data = data_loader.load_ethusdt_1h_data()
    
    if market_data is None or market_data.empty:
        print("❌ Failed to load ETHUSDT data")
        return False
    
    # Filter to 60 days
    market_data = filter_data_by_days(market_data, days=60)
    
    # Create the step
    step = StickyFiniteHMMRegimeDiscoveryStep()
    
    # Configure for 60 days
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '1h',
        'execution_mode': 'full',  # Use full mode to avoid light mode filter
        'auto_tuning': {
            'enabled': True,
            'n_rounds': 2,
            'trials_per_round': 30,  # Reduced for faster execution
            'timeout_minutes': 15,
            'optimization_level': 'balanced'
        },
        'clustering': {
            'k': 5,  # Fixed number of states
            'method': 'sticky_finite_hmm',
            'enhanced_integration': True,
            'feature_generation': {
                'max_features': 100,  # Reduced for faster execution
                'enable_all_categories': True
            }
        }
    }
    
    try:
        # Execute the step with pre-loaded data by monkey-patching the DataLoader
        # This is a cleaner approach than accessing protected methods
        
        # Save original DataLoader method
        original_load_ethusdt = DataLoader.load_ethusdt_1h_data
        
        # Create a patched version that returns our filtered data
        def patched_load_ethusdt_1h_data(self, data_dir="historical_data"):
            return market_data
        
        # Apply the patch
        DataLoader.load_ethusdt_1h_data = patched_load_ethusdt_1h_data
        
        result = await step.execute(config)
        
        # Restore original method
        DataLoader.load_ethusdt_1h_data = original_load_ethusdt
        
        if result:
            print("\n" + "=" * 80)
            print("✅ STICKY FINITE HMM COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"End time: {datetime.now()}")
            print(f"Results saved to artifacts")
            print("=" * 80)
            return True
        else:
            print("\n" + "=" * 80)
            print("❌ STICKY FINITE HMM FAILED")
            print("=" * 80)
            return False
            
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(run_sticky_finite_hmm_60_days())
    sys.exit(0 if success else 1)
