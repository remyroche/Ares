#!/usr/bin/env python3
"""
Test script for enhanced data quality integration in enhanced_klines_processing_pipeline.py

This script tests the integration of src/utils/data/quality/ utilities with the
enhanced klines processing pipeline.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_test_data():
    """Create test klines data with various quality issues."""
    # Create a date range
    start_date = datetime.now() - timedelta(days=30)
    dates = pd.date_range(start=start_date, periods=1000, freq='1H')
    
    # Create base OHLCV data
    np.random.seed(42)  # For reproducible tests
    base_price = 100.0
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        # Simulate price movement
        price_change = np.random.normal(0, 0.01) * base_price
        base_price += price_change
        
        # Create OHLCV data
        open_price = base_price
        high_price = base_price + abs(np.random.normal(0, 0.005)) * base_price
        low_price = base_price - abs(np.random.normal(0, 0.005)) * base_price
        close_price = base_price + np.random.normal(0, 0.002) * base_price
        volume = np.random.exponential(1000)
        
        prices.append([open_price, high_price, low_price, close_price])
        volumes.append(volume)
    
    # Create DataFrame
    df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close'], index=dates)
    df['volume'] = volumes
    
    # Add some quality issues for testing
    # 1. Add some null values
    df.loc[df.index[10:15], 'volume'] = np.nan
    
    # 2. Add some negative values
    df.loc[df.index[20:25], 'low'] = -1.0
    
    # 3. Add some zero volumes
    df.loc[df.index[30:35], 'volume'] = 0
    
    # 4. Add some duplicates
    duplicate_data = df.iloc[100:105].copy()
    df = pd.concat([df, duplicate_data])
    df = df.sort_index()
    
    return df

def test_quality_utilities_import():
    """Test that all quality utilities can be imported."""
    print("🧪 Testing quality utilities import...")
    
    try:
        from src.utils.data.quality.data_quality import DataQualityFramework, QualityThresholds, QualityResult
        from src.utils.data.quality.comprehensive_quality_scorer import ComprehensiveQualityScorer, QualityScore, QualityScoreLevel
        from src.utils.data.quality.advanced_quality_metrics import AdvancedQualityMetrics, QualityAssessment
        from src.utils.data.quality.data_cleaning import DataCleaner
        from src.utils.data.quality.statistical_distribution_validation import StatisticalValidator
        from src.utils.data.quality.quality_alert_system import QualityAlertSystem
        print("✅ All quality utilities imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import quality utilities: {e}")
        return False

def test_pipeline_import():
    """Test that the enhanced pipeline can be imported."""
    print("🧪 Testing enhanced pipeline import...")
    
    try:
        from src.training.steps.data_collection.enhanced_klines_processing_pipeline import EnhancedKlinesProcessingPipeline
        print("✅ Enhanced pipeline imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import enhanced pipeline: {e}")
        return False

def test_quality_framework():
    """Test the data quality framework."""
    print("🧪 Testing data quality framework...")
    
    try:
        from src.utils.data.quality.data_quality import DataQualityFramework, QualityThresholds
        
        # Create test data
        df = create_test_data()
        
        # Initialize framework
        framework = DataQualityFramework()
        thresholds = QualityThresholds(
            null_percentage_threshold=5.0,
            negative_value_threshold=0.0,
            zero_volume_threshold=10.0,
            temporal_consistency_threshold=0.95,
            price_consistency_threshold=0.98
        )
        
        # Test validation
        result = framework.validate_data(df, thresholds)
        print(f"✅ Quality framework test passed - Score: {result.score:.2f}")
        print(f"   Issues: {len(result.issues)}, Warnings: {len(result.warnings)}")
        return True
    except Exception as e:
        print(f"❌ Quality framework test failed: {e}")
        return False

def test_comprehensive_quality_scorer():
    """Test the comprehensive quality scorer."""
    print("🧪 Testing comprehensive quality scorer...")
    
    try:
        from src.utils.data.quality.comprehensive_quality_scorer import ComprehensiveQualityScorer
        
        # Create test data
        df = create_test_data()
        
        # Initialize scorer
        scorer = ComprehensiveQualityScorer()
        
        # Test scoring
        score = scorer.score_data_quality(df, "BTCUSDT", "1h")
        print(f"✅ Quality scorer test passed - Score: {score.overall_score:.2f} ({score.level.value})")
        print(f"   Component scores: {list(score.component_scores.keys())}")
        print(f"   Issues: {len(score.issues)}, Warnings: {len(score.warnings)}")
        return True
    except Exception as e:
        print(f"❌ Quality scorer test failed: {e}")
        return False

def test_advanced_quality_metrics():
    """Test the advanced quality metrics."""
    print("🧪 Testing advanced quality metrics...")
    
    try:
        from src.utils.data.quality.advanced_quality_metrics import AdvancedQualityMetrics
        
        # Create test data
        df = create_test_data()
        
        # Initialize metrics
        metrics = AdvancedQualityMetrics()
        
        # Test assessment
        assessment = metrics.assess_quality(df)
        print(f"✅ Advanced metrics test passed - Score: {assessment.overall_score:.2f}")
        print(f"   Issues: {assessment.issues_found}, Warnings: {assessment.warnings_found}")
        print(f"   Critical issues: {assessment.critical_issues}")
        return True
    except Exception as e:
        print(f"❌ Advanced metrics test failed: {e}")
        return False

async def test_pipeline_quality_integration():
    """Test the pipeline's quality integration."""
    print("🧪 Testing pipeline quality integration...")
    
    try:
        from src.training.steps.data_collection.enhanced_klines_processing_pipeline import EnhancedKlinesProcessingPipeline
        
        # Create test data
        df = create_test_data()
        
        # Initialize pipeline (without logging to avoid output noise)
        pipeline = EnhancedKlinesProcessingPipeline('binance', enable_logging=False)
        
        # Test comprehensive quality scoring method
        quality_score = await pipeline.get_comprehensive_quality_score(df, "BTCUSDT", "1h")
        print(f"✅ Pipeline quality integration test passed")
        print(f"   Quality score: {quality_score.overall_score:.2f} ({quality_score.level.value})")
        print(f"   Data shape: {quality_score.data_shape}")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline quality integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_cleaning():
    """Test the data cleaning utilities."""
    print("🧪 Testing data cleaning utilities...")
    
    try:
        from src.utils.data.quality.data_cleaning import DataCleaner
        
        # Create test data with issues
        df = create_test_data()
        
        # Initialize cleaner
        cleaner = DataCleaner()
        
        # Test cleaning
        cleaned_df = cleaner.clean_data(df)
        print(f"✅ Data cleaning test passed")
        print(f"   Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
        print(f"   Original nulls: {df.isnull().sum().sum()}, Cleaned nulls: {cleaned_df.isnull().sum().sum()}")
        
        return True
    except Exception as e:
        print(f"❌ Data cleaning test failed: {e}")
        return False

def test_statistical_validator():
    """Test the statistical validator."""
    print("🧪 Testing statistical validator...")
    
    try:
        from src.utils.data.quality.statistical_distribution_validation import StatisticalValidator
        
        # Create test data
        df = create_test_data()
        
        # Initialize validator
        validator = StatisticalValidator()
        
        # Test validation
        validation_results = validator.validate_distributions(df)
        print(f"✅ Statistical validator test passed")
        print(f"   Validation results: {list(validation_results.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Statistical validator test failed: {e}")
        return False

async def run_all_tests():
    """Run all quality integration tests."""
    print("🚀 Starting Enhanced Data Quality Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Quality Utilities Import", test_quality_utilities_import, False),
        ("Pipeline Import", test_pipeline_import, False),
        ("Quality Framework", test_quality_framework, False),
        ("Comprehensive Quality Scorer", test_comprehensive_quality_scorer, False),
        ("Advanced Quality Metrics", test_advanced_quality_metrics, False),
        ("Data Cleaning", test_data_cleaning, False),
        ("Statistical Validator", test_statistical_validator, False),
        ("Pipeline Quality Integration", test_pipeline_quality_integration, True),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func, is_async in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if is_async:
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Data quality integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_all_tests())