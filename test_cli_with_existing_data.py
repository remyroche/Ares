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

def find_existing_data():
    """Find existing data files."""
    print("Looking for existing data files...")
    
    # Common data directories
    data_dirs = [
        "data",
        "data/raw",
        "data/processed",
        "artifacts",
        "artifacts/Analyst"
    ]
    
    # Look for parquet or csv files
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if data_path.exists():
            print(f"🔍 Checking directory: {data_path}")
            
            # Look for parquet files first
            for file_path in data_path.rglob("*.parquet"):
                if "BTCUSDT" in str(file_path) or "ETHUSDT" in str(file_path):
                    print(f"✅ Found data file: {file_path}")
                    return file_path
            
            # Look for csv files if no parquet found
            for file_path in data_path.rglob("*.csv"):
                if "BTCUSDT" in str(file_path) or "ETHUSDT" in str(file_path):
                    print(f"✅ Found data file: {file_path}")
                    return file_path
    
    print("❌ No existing data files found")
    return None

def test_cli_with_existing_data():
    """Test CLI with existing data."""
    try:
        # Find existing data
        data_file = find_existing_data()
        
        if not data_file:
            print("❌ No existing data found, cannot run CLI test")
            return False
        
        # Import CLI
        from src.training.steps.market_analysis.statsmodel_clustering.cli import StatsmodelClusteringCLI
        
        # Create CLI instance
        cli = StatsmodelClusteringCLI()
        
        # Test cluster command with existing data
        print("\n🔬 Testing cluster command with existing data...")
        
        import asyncio
        
        async def run_test():
            # Create args for cluster command
            args = [
                'cluster',
                '--symbol', 'BTCUSDT',
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
            print("✅ CLI test with existing data passed!")
            return True
        else:
            print(f"❌ CLI test with existing data failed with exit code: {result}")
            return False
            
    except Exception as e:
        print(f"❌ CLI test with existing data failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run CLI test with existing data."""
    print("🧪 Running CLI test with existing data...\n")
    
    result = test_cli_with_existing_data()
    
    if result:
        print("\n🎉 CLI test with existing data passed!")
        return 0
    else:
        print("\n❌ CLI test with existing data failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)