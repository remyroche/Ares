#!/usr/bin/env python3
"""
Test script for CLI with existing data
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_sample_data():
    """Create sample data for testing."""
    print("Creating sample data...")
    
    # Create sample OHLCV data
    np.random.seed(42)
    n_samples = 500
    
    # Generate price data
    price = 100.0
    prices = [price]
    
    for i in range(1, n_samples):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        price = price * (1 + change)
        prices.append(price)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Add datetime index
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    data.index = dates
    
    # Save to file
    data_file = Path("sample_data.parquet")
    data.to_parquet(data_file, index=True)
    
    print(f"✅ Sample data created and saved to {data_file}")
    print(f"📊 Data shape: {data.shape}")
    print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
    
    return data_file

def test_cli_with_data():
    """Test CLI with existing data."""
    try:
        # Import CLI
        from src.training.steps.market_analysis.statsmodel_clustering.cli import StatsmodelClusteringCLI
        
        # Create sample data
        data_file = create_sample_data()
        
        # Create CLI instance
        cli = StatsmodelClusteringCLI()
        
        # Test cluster command with sample data
        print("\n🔬 Testing cluster command with sample data...")
        
        import asyncio
        
        async def run_test():
            # Create args for cluster command
            args = [
                'cluster',
                '--symbol', 'TEST',
                '--data-file', str(data_file),
                '--regimes', '3',
                '--output-dir', 'test_output'
            ]
            
            # Run CLI
            result = await cli.run(args)
            return result
        
        # Run the test
        result = asyncio.run(run_test())
        
        if result == 0:
            print("✅ CLI test with data passed!")
            return True
        else:
            print(f"❌ CLI test with data failed with exit code: {result}")
            return False
            
    except Exception as e:
        print(f"❌ CLI test with data failed with exception: {e}")
        return False

def main():
    """Run CLI test with data."""
    print("🧪 Running CLI test with data...\n")
    
    result = test_cli_with_data()
    
    if result:
        print("\n🎉 CLI test with data passed!")
        return 0
    else:
        print("\n❌ CLI test with data failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)