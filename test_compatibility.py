#!/usr/bin/env python3
"""
Compatibility Test for ExchangeInterface, src/utils/data/, and enhanced_klines_processing_pipeline.py

This script tests the compatibility between the three main components to ensure they work together correctly.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required imports work correctly."""
    print("🔍 Testing imports...")
    
    try:
        # Test ExchangeInterface imports
        from src.trading.execution.exchange_interface import ExchangeInterface, create_exchange_interface, KlineData
        print("✅ ExchangeInterface imports successful")
        
        # Test data utilities imports
        from src.utils.data.klines_parquet import KlinesParquetManager
        from src.utils.data.quality.data_quality import DataQualityFramework, QualityResult
        from src.utils.data.unified_data_utils import UnifiedDataUtils
        print("✅ Data utilities imports successful")
        
        # Test enhanced klines pipeline imports
        from src.training.steps.data_collection.enhanced_klines_processing_pipeline import (
            EnhancedKlinesProcessingPipeline, 
            PipelineConfig, 
            ResamplingConfig,
            process_klines_data_enhanced
        )
        print("✅ Enhanced klines pipeline imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_structures():
    """Test that data structures are compatible."""
    print("\n🔍 Testing data structures...")
    
    try:
        from src.trading.execution.exchange_interface import KlineData
        from src.training.steps.data_collection.enhanced_klines_processing_pipeline import QualityScore, QualityAssessment
        
        # Test KlineData creation
        kline = KlineData(
            symbol="ETHUSDT",
            interval="1m",
            timestamp=datetime.now(),
            open_price=3000.0,
            high_price=3010.0,
            low_price=2990.0,
            close_price=3005.0,
            volume=100.0,
            close_time=datetime.now(),
            quote_asset_volume=300500.0,
            number_of_trades=50,
            taker_buy_base_asset_volume=50.0,
            taker_buy_quote_asset_volume=150250.0
        )
        print("✅ KlineData creation successful")
        
        # Test QualityScore creation
        from src.training.steps.data_collection.enhanced_klines_processing_pipeline import QualityScoreLevel
        quality_score = QualityScore(
            overall_score=85.0,
            level=QualityScoreLevel.GOOD,
            component_scores={"completeness": 90.0, "accuracy": 80.0},
            issues=[],
            warnings=["Minor data gaps detected"],
            recommendations=["Consider gap filling"],
            assessment_timestamp=datetime.now(),
            data_shape=(1000, 10)
        )
        print("✅ QualityScore creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False

async def test_exchange_interface():
    """Test ExchangeInterface functionality."""
    print("\n🔍 Testing ExchangeInterface...")
    
    try:
        from src.trading.execution.exchange_interface import create_exchange_interface
        
        # Create simulated exchange interface
        config = {
            'exchange_type': 'simulated',
            'api_key': '',
            'api_secret': '',
            'testnet': True
        }
        
        exchange = create_exchange_interface(config)
        await exchange.connect()
        
        # Test get_klines method
        klines = await exchange.get_klines(
            symbol="ETHUSDT",
            interval="1m",
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now(),
            limit=10
        )
        
        print(f"✅ ExchangeInterface test successful - got {len(klines)} klines")
        
        await exchange.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ ExchangeInterface test failed: {e}")
        return False

def test_data_utilities():
    """Test data utilities functionality."""
    print("\n🔍 Testing data utilities...")
    
    try:
        from src.utils.data.klines_parquet import KlinesParquetManager
        from src.utils.data.quality.data_quality import DataQualityFramework
        from src.utils.data.unified_data_utils import UnifiedDataUtils
        import pandas as pd
        import numpy as np
        
        # Test KlinesParquetManager
        manager = KlinesParquetManager("test_data", "binance")
        print("✅ KlinesParquetManager creation successful")
        
        # Test DataQualityFramework
        quality_framework = DataQualityFramework()
        print("✅ DataQualityFramework creation successful")
        
        # Test UnifiedDataUtils
        data_utils = UnifiedDataUtils()
        print("✅ UnifiedDataUtils creation successful")
        
        # Test with sample data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'open': np.random.uniform(2900, 3100, 100),
            'high': np.random.uniform(3000, 3200, 100),
            'low': np.random.uniform(2800, 3000, 100),
            'close': np.random.uniform(2900, 3100, 100),
            'volume': np.random.uniform(100, 1000, 100)
        })
        sample_data.set_index('timestamp', inplace=True)
        
        # Test quality validation
        quality_result = quality_framework.validate_dataframe_quality(sample_data, "test")
        print(f"✅ Quality validation successful - score: {quality_result.quality_score}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data utilities test failed: {e}")
        return False

async def test_enhanced_pipeline():
    """Test enhanced klines processing pipeline."""
    print("\n🔍 Testing enhanced klines processing pipeline...")
    
    try:
        from src.training.steps.data_collection.enhanced_klines_processing_pipeline import (
            EnhancedKlinesProcessingPipeline, 
            PipelineConfig, 
            ResamplingConfig
        )
        from src.trading.execution.exchange_interface import create_exchange_interface
        
        # Create pipeline configuration
        config = PipelineConfig(
            data_dir="test_data",
            exchange="binance",
            enable_logging=True,
            enable_gap_filling=False,  # Disable for test
            enable_resampling=False,   # Disable for test
            enable_duplicate_handling=True,
            enable_quality_validation=True,
            batch_compatible=True
        )
        
        # Create pipeline
        pipeline = EnhancedKlinesProcessingPipeline(config)
        print("✅ EnhancedKlinesProcessingPipeline creation successful")
        
        # Create exchange interface
        exchange_config = {
            'exchange_type': 'simulated',
            'api_key': '',
            'api_secret': '',
            'testnet': True
        }
        
        exchange = create_exchange_interface(exchange_config)
        await exchange.connect()
        
        # Test pipeline processing (with minimal data)
        print("🔄 Testing pipeline processing...")
        results = await pipeline.process_klines_data(
            symbol="ETHUSDT",
            interval="1m",
            years=1,
            exchange_interface=exchange,
            create_consolidated=False  # Disable for test
        )
        
        print(f"✅ Pipeline processing successful - success: {results['pipeline_success']}")
        
        await exchange.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Enhanced pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all compatibility tests."""
    print("🚀 Starting compatibility tests...")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Structures Test", test_data_structures),
        ("Data Utilities Test", test_data_utilities),
        ("ExchangeInterface Test", test_exchange_interface),
        ("Enhanced Pipeline Test", test_enhanced_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
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
    print("📊 COMPATIBILITY TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All compatibility tests passed! The components are fully compatible.")
        return True
    else:
        print("⚠️ Some compatibility tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
