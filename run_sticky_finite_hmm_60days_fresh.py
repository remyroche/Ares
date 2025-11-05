#!/usr/bin/env python3
"""
Run Sticky Finite HMM with 60 days of data and fresh module imports.
This script ensures the fixed regime generators are loaded properly.
"""

import sys
from datetime import datetime, timedelta
import pandas as pd

# Clear all cached modules related to feature generation
modules_to_clear = []
for module_name in sys.modules:
    if any(x in module_name for x in [
        'src.feature_generation.categories.advanced_regime_features',
        'src.feature_generation.categories',
        'src.feature_generation',
        'src.training.steps.market_analysis.sticky_finite_hmm_clustering'
    ]):
        modules_to_clear.append(module_name)

for module_name in modules_to_clear:
    del sys.modules[module_name]

print(f"Cleared {len(modules_to_clear)} cached modules")

# Add src to path
sys.path.insert(0, 'src')

# Import core components
from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_regime_discovery_step import (
    StickyFiniteHMMRegimeDiscoveryStep
)
from src.utils.data_loader import DataLoader

def filter_data_by_days(data: pd.DataFrame | pd.Series, days: int = 60) -> pd.DataFrame:
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

async def run_sticky_finite_hmm_90_days_fresh():
    """Run Sticky Finite HMM with 90 days of data and fresh imports."""
    
    print("=" * 80)
    print("STICKY FINITE HMM REGIME DISCOVERY - 90 DAYS (FRESH IMPORTS)")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print(f"Data: ETHUSDT 1h - Last 90 days")
    print("=" * 80)
    
    # Load data first to filter it
    print("\n📂 Loading ETHUSDT data...")
    data_loader = DataLoader()
    market_data = data_loader.load_ethusdt_1h_data()
    
    if market_data is None or market_data.empty:
        print("❌ Failed to load ETHUSDT data")
        return False
    
    # Filter to 90 days
    market_data = filter_data_by_days(market_data, days=90)
    
    # Create the step
    step = StickyFiniteHMMRegimeDiscoveryStep(
        step_name="sticky_finite_hmm_90_days_fresh"
    )
    
    # Configure with all original settings but for 90 days
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '1h',
        'regime_timeframe': '1h',  # Use 1h for regime detection
        'execution_mode': 'full',   # Full mode for complete analysis
        'enable_auto_tuning': True, # Enable auto-tuning
        'auto_tuning_config': {
            'use_hierarchical': True,      # Use hierarchical optimization
            'use_multi_objective': False,  # Single objective for speed
            'n_rounds': 2,                 # 2 optimization rounds
            'tpe_trials': 50,              # 50 TPE trials
            'timeout': 1800                # 30 min timeout
        },
        'sticky_finite_hmm_params': {
            'min_features': 100,   # Ensure 100+ features
            'max_features': 150,   # Allow up to 150 features
            'enable_pca': True,
            'pca_components': 20,  # Use 20 PCA components
            'num_iters': 800,      # Sufficient iterations
            'compute_posteriors': True  # Full computation
        }
    }
    
    try:
        # Execute the step with pre-loaded data by monkey-patching the DataLoader
        # This avoids accessing protected methods
        
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
    success = asyncio.run(run_sticky_finite_hmm_90_days_fresh())
    sys.exit(0 if success else 1)
