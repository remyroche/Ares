#!/usr/bin/env python3
"""
Import Compatibility Test Script

This script tests that all imports work correctly between components.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required imports work."""
    print("🧪 Testing Import Compatibility...")
    
    tests = []
    
    # Test ExchangeInterface imports
    try:
        from src.trading.execution.exchange_interface import ExchangeInterface, create_exchange_interface
        print("✅ ExchangeInterface imports successful")
        tests.append(("ExchangeInterface", True))
    except Exception as e:
        print(f"❌ ExchangeInterface imports failed: {e}")
        tests.append(("ExchangeInterface", False))
    
    # Test KlinesParquetManager imports
    try:
        from src.utils.kline_parquet import KlinesParquetManager, StorageConfig, KlinesMetadata
        print("✅ KlinesParquetManager imports successful")
        tests.append(("KlinesParquetManager", True))
    except Exception as e:
        print(f"❌ KlinesParquetManager imports failed: {e}")
        tests.append(("KlinesParquetManager", False))
    
    # Test Enhanced Klines Pipeline imports
    try:
        from src.training.steps.data_collection.enhanced_klines_processing_pipeline import (
            EnhancedKlinesProcessingPipeline,
            PipelineConfig,
            ResamplingConfig
        )
        print("✅ Enhanced Klines Pipeline imports successful")
        tests.append(("Enhanced Klines Pipeline", True))
    except Exception as e:
        print(f"❌ Enhanced Klines Pipeline imports failed: {e}")
        tests.append(("Enhanced Klines Pipeline", False))
    
    # Test Data Quality utilities imports
    try:
        from src.utils.data.quality.data_quality import DataQualityFramework, QualityThresholds, QualityResult
        print("✅ Data Quality Framework imports successful")
        tests.append(("Data Quality Framework", True))
    except Exception as e:
        print(f"❌ Data Quality Framework imports failed: {e}")
        tests.append(("Data Quality Framework", False))
    
    try:
        from src.utils.data.quality.comprehensive_quality_scorer import ComprehensiveQualityScorer, QualityScore, QualityScoreLevel
        print("✅ Comprehensive Quality Scorer imports successful")
        tests.append(("Comprehensive Quality Scorer", True))
    except Exception as e:
        print(f"❌ Comprehensive Quality Scorer imports failed: {e}")
        tests.append(("Comprehensive Quality Scorer", False))
    
    try:
        from src.utils.data.quality.comprehensive_duplicate_analyzer import ComprehensiveDuplicateAnalyzer, analyze_duplicates_comprehensive
        print("✅ Comprehensive Duplicate Analyzer imports successful")
        tests.append(("Comprehensive Duplicate Analyzer", True))
    except Exception as e:
        print(f"❌ Comprehensive Duplicate Analyzer imports failed: {e}")
        tests.append(("Comprehensive Duplicate Analyzer", False))
    
    try:
        from src.utils.data.quality.advanced_quality_metrics import AdvancedQualityMetrics, QualityAssessment
        print("✅ Advanced Quality Metrics imports successful")
        tests.append(("Advanced Quality Metrics", True))
    except Exception as e:
        print(f"❌ Advanced Quality Metrics imports failed: {e}")
        tests.append(("Advanced Quality Metrics", False))
    
    try:
        from src.utils.data.quality.data_cleaning import DataCleaner
        print("✅ Data Cleaner imports successful")
        tests.append(("Data Cleaner", True))
    except Exception as e:
        print(f"❌ Data Cleaner imports failed: {e}")
        tests.append(("Data Cleaner", False))
    
    try:
        from src.utils.data.quality.statistical_distribution_validation import StatisticalValidator
        print("✅ Statistical Validator imports successful")
        tests.append(("Statistical Validator", True))
    except Exception as e:
        print(f"❌ Statistical Validator imports failed: {e}")
        tests.append(("Statistical Validator", False))
    
    try:
        from src.utils.data.quality.quality_alert_system import QualityAlertSystem
        print("✅ Quality Alert System imports successful")
        tests.append(("Quality Alert System", True))
    except Exception as e:
        print(f"❌ Quality Alert System imports failed: {e}")
        tests.append(("Quality Alert System", False))
    
    # Test Unified Exchange Standardizer imports
    try:
        from exchanges.shared.unified_ohlcv_standardizer import UnifiedExchangeStandardizer, ExchangeType
        print("✅ Unified Exchange Standardizer imports successful")
        tests.append(("Unified Exchange Standardizer", True))
    except Exception as e:
        print(f"❌ Unified Exchange Standardizer imports failed: {e}")
        tests.append(("Unified Exchange Standardizer", False))
    
    return tests

def main():
    """Run import compatibility tests."""
    print("🚀 Starting Import Compatibility Tests...")
    print("=" * 60)
    
    tests = test_imports()
    
    print("\n" + "=" * 60)
    print("📊 Import Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in tests:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:35} | {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Total: {passed}/{total} imports successful")
    
    if passed == total:
        print("🎉 All import tests PASSED! All components can be imported successfully.")
        return True
    else:
        print("⚠️ Some import tests FAILED. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)