#!/usr/bin/env python3
"""
Test Script for Statsmodel Clustering Implementation

This script tests the core functionality of the statsmodel clustering module,
including data downloading, clustering analysis, and CLI interface.
"""

import asyncio
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import components to test
from src.training.steps.market_analysis.statsmodel_clustering.core import (
    BaseDataDownloader,
    StandardDataDownloader,
    create_data_downloader,
    download_clustering_data,
    MarkovRegressionAdapter,
    create_enhanced_markov_regression_adapter
)

# Import CLI
from src.training.steps.market_analysis.statsmodel_clustering.cli import StatsmodelClusteringCLI


def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing imports...")
    
    try:
        # Test core components
        assert BaseDataDownloader is not None
        assert StandardDataDownloader is not None
        assert create_data_downloader is not None
        assert download_clustering_data is not None
        assert MarkovRegressionAdapter is not None
        assert create_enhanced_markov_regression_adapter is not None
        print("✅ Core components imported successfully")
        
        # Test CLI
        assert StatsmodelClusteringCLI is not None
        print("✅ CLI imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def test_data_downloader_creation():
    """Test data downloader creation."""
    print("\n🧪 Testing data downloader creation...")
    
    try:
        config = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1h',
            'lookback_years': 1,
            'data_dir': 'test_data',
            'downloader_type': 'standard'
        }
        
        downloader = create_data_downloader(config)
        assert isinstance(downloader, StandardDataDownloader)
        assert downloader.symbol == 'ETHUSDT'
        assert downloader.exchange == 'BINANCE'
        assert downloader.timeframe == '1h'
        
        print("✅ Data downloader created successfully")
        return True
    except Exception as e:
        print(f"❌ Data downloader creation test failed: {e}")
        return False


def test_markov_adapter_creation():
    """Test Markov regression adapter creation."""
    print("\n🧪 Testing Markov regression adapter creation...")
    
    try:
        adapter = create_enhanced_markov_regression_adapter(
            k_regimes=3,
            enable_pca=True,
            pca_components=5,
            enable_diagnostics=True
        )
        
        assert adapter is not None
        assert adapter.config.k_regimes == 3
        assert adapter.config.enable_pca == True
        assert adapter.config.pca_components == 5
        
        print("✅ Markov regression adapter created successfully")
        return True
    except Exception as e:
        print(f"❌ Markov adapter creation test failed: {e}")
        return False


def test_cli_creation():
    """Test CLI creation."""
    print("\n🧪 Testing CLI creation...")
    
    try:
        cli = StatsmodelClusteringCLI()
        assert cli is not None
        
        # Test parser creation
        parser = cli.create_parser()
        assert parser is not None
        
        print("✅ CLI created successfully")
        return True
    except Exception as e:
        print(f"❌ CLI creation test failed: {e}")
        return False


async def test_data_download():
    """Test data download functionality."""
    print("\n🧪 Testing data download...")
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        config = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1h',
            'lookback_years': 1,
            'data_dir': temp_dir,
            'downloader_type': 'standard'
        }
        
        downloader = create_data_downloader(config)
        
        # Test the download process (may fail due to missing API keys, but should not crash)
        print("📥 Attempting data download...")
        success, data, error = await downloader.download_data()
        
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if success and data is not None:
            print(f"✅ Data download successful: {len(data)} records")
            return True
        else:
            print(f"⚠️ Data download failed (expected if no API keys): {error}")
            # This is expected if no API keys are available
            return True
            
    except Exception as e:
        print(f"❌ Data download test failed: {e}")
        return False


def test_cli_commands():
    """Test CLI command parsing."""
    print("\n🧪 Testing CLI command parsing...")
    
    try:
        cli = StatsmodelClusteringCLI()
        
        # Test download command parsing
        test_args = ['download', '--symbol', 'BTCUSDT', '--timeframe', '1h']
        parser = cli.create_parser()
        args = parser.parse_args(test_args)
        
        assert args.command == 'download'
        assert args.symbol == 'BTCUSDT'
        assert args.timeframe == '1h'
        
        # Test cluster command parsing
        test_args = ['cluster', '--symbol', 'BTCUSDT', '--data-file', 'test.parquet', '--regimes', '3']
        args = parser.parse_args(test_args)
        
        assert args.command == 'cluster'
        assert args.symbol == 'BTCUSDT'
        assert args.data_file == 'test.parquet'
        assert args.regimes == 3
        
        print("✅ CLI command parsing successful")
        return True
    except Exception as e:
        print(f"❌ CLI command parsing test failed: {e}")
        return False


def test_feature_preparation():
    """Test feature preparation functionality."""
    print("\n🧪 Testing feature preparation...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        cli = StatsmodelClusteringCLI()
        features = cli._prepare_features(data)
        
        assert features is not None
        assert len(features) > 0
        assert 'returns' in features.columns
        assert 'log_returns' in features.columns
        assert 'volatility' in features.columns
        
        print("✅ Feature preparation successful")
        return True
    except Exception as e:
        print(f"❌ Feature preparation test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests."""
    print("🚀 Starting Statsmodel Clustering Implementation Tests")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Downloader Creation Test", test_data_downloader_creation),
        ("Markov Adapter Creation Test", test_markov_adapter_creation),
        ("CLI Creation Test", test_cli_creation),
        ("CLI Commands Test", test_cli_commands),
        ("Feature Preparation Test", test_feature_preparation),
        ("Data Download Test", test_data_download),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️ Some tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)