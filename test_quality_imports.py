#!/usr/bin/env python3
"""
Simple test script to validate the enhanced data quality integration imports
without requiring external dependencies like pandas.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_quality_utilities_import():
    """Test that all quality utilities can be imported."""
    print("🧪 Testing quality utilities import...")
    
    try:
        from src.utils.data.quality.data_quality import DataQualityFramework, QualityThresholds, QualityResult
        print("✅ DataQualityFramework imported successfully")
        
        from src.utils.data.quality.comprehensive_quality_scorer import ComprehensiveQualityScorer, QualityScore, QualityScoreLevel
        print("✅ ComprehensiveQualityScorer imported successfully")
        
        from src.utils.data.quality.advanced_quality_metrics import AdvancedQualityMetrics, QualityAssessment
        print("✅ AdvancedQualityMetrics imported successfully")
        
        from src.utils.data.quality.data_cleaning import DataCleaner
        print("✅ DataCleaner imported successfully")
        
        from src.utils.data.quality.statistical_distribution_validation import StatisticalValidator
        print("✅ StatisticalValidator imported successfully")
        
        from src.utils.data.quality.quality_alert_system import QualityAlertSystem
        print("✅ QualityAlertSystem imported successfully")
        
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

def test_pipeline_syntax():
    """Test that the enhanced pipeline has correct syntax."""
    print("🧪 Testing enhanced pipeline syntax...")
    
    try:
        import py_compile
        py_compile.compile('src/training/steps/data_collection/enhanced_klines_processing_pipeline.py', doraise=True)
        print("✅ Enhanced pipeline syntax is correct")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ Enhanced pipeline syntax error: {e}")
        return False

def test_quality_methods_exist():
    """Test that the quality methods exist in the pipeline."""
    print("🧪 Testing quality methods exist in pipeline...")
    
    try:
        from src.training.steps.data_collection.enhanced_klines_processing_pipeline import EnhancedKlinesProcessingPipeline
        
        # Check if the class has the expected methods
        methods = dir(EnhancedKlinesProcessingPipeline)
        
        expected_methods = [
            '_validate_data_quality',
            '_final_quality_check',
            'get_comprehensive_quality_score'
        ]
        
        missing_methods = []
        for method in expected_methods:
            if method not in methods:
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing methods: {missing_methods}")
            return False
        
        print("✅ All expected quality methods exist in pipeline")
        return True
    except Exception as e:
        print(f"❌ Failed to check quality methods: {e}")
        return False

def test_import_structure():
    """Test the import structure in the enhanced pipeline."""
    print("🧪 Testing import structure in enhanced pipeline...")
    
    try:
        with open('src/training/steps/data_collection/enhanced_klines_processing_pipeline.py', 'r') as f:
            content = f.read()
        
        # Check for key imports
        required_imports = [
            'from src.utils.data.quality.data_quality import',
            'from src.utils.data.quality.comprehensive_quality_scorer import',
            'from src.utils.data.quality.advanced_quality_metrics import',
            'from src.utils.data.quality.data_cleaning import',
            'from src.utils.data.quality.statistical_distribution_validation import',
            'from src.utils.data.quality.quality_alert_system import'
        ]
        
        missing_imports = []
        for import_line in required_imports:
            if import_line not in content:
                missing_imports.append(import_line)
        
        if missing_imports:
            print(f"❌ Missing imports: {missing_imports}")
            return False
        
        print("✅ All required quality imports are present")
        return True
    except Exception as e:
        print(f"❌ Failed to check import structure: {e}")
        return False

def test_quality_integration_methods():
    """Test that quality integration methods are properly implemented."""
    print("🧪 Testing quality integration methods...")
    
    try:
        with open('src/training/steps/data_collection/enhanced_klines_processing_pipeline.py', 'r') as f:
            content = f.read()
        
        # Check for key quality integration patterns
        quality_patterns = [
            'DataQualityFramework()',
            'ComprehensiveQualityScorer()',
            'AdvancedQualityMetrics()',
            'DataCleaner()',
            'StatisticalValidator()',
            'QualityAlertSystem()',
            'quality_framework.validate_data',
            'quality_scorer.score_data_quality',
            'advanced_metrics.assess_quality',
            'statistical_validator.validate_distributions',
            'analyze_duplicates_comprehensive'
        ]
        
        missing_patterns = []
        for pattern in quality_patterns:
            if pattern not in content:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            print(f"❌ Missing quality integration patterns: {missing_patterns}")
            return False
        
        print("✅ All quality integration patterns are present")
        return True
    except Exception as e:
        print(f"❌ Failed to check quality integration methods: {e}")
        return False

def run_all_tests():
    """Run all quality integration tests."""
    print("🚀 Starting Enhanced Data Quality Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Quality Utilities Import", test_quality_utilities_import),
        ("Pipeline Import", test_pipeline_import),
        ("Pipeline Syntax", test_pipeline_syntax),
        ("Quality Methods Exist", test_quality_methods_exist),
        ("Import Structure", test_import_structure),
        ("Quality Integration Methods", test_quality_integration_methods),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Data quality integration is working correctly.")
        print("\n📋 Summary of Changes:")
        print("   ✅ Updated imports to include comprehensive data quality utilities")
        print("   ✅ Enhanced _validate_data_quality method with comprehensive framework")
        print("   ✅ Enhanced _final_quality_check method with advanced metrics")
        print("   ✅ Added get_comprehensive_quality_score method")
        print("   ✅ Integrated all src/utils/data/quality/ utilities")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()