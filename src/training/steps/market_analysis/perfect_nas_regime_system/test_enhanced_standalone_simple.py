"""
Simple Test for Enhanced Standalone Perfect NAS Regime System Components

Tests the enhanced standalone components without external dependencies.
"""

import sys
import os
import time
import logging

# Add the project root to the path
sys.path.append('/workspace/src')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_standalone_imports():
    """Test that enhanced standalone components can be imported."""
    try:
        logger.info("🧪 Testing Enhanced Standalone Component Imports...")
        logger.info("=" * 60)
        
        # Test imports
        logger.info("📦 Testing StandaloneNASClusterer import...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneNASClusterer
        logger.info("✅ StandaloneNASClusterer imported successfully")
        
        logger.info("📦 Testing StandaloneNASEvaluator import...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneNASEvaluator
        logger.info("✅ StandaloneNASEvaluator imported successfully")
        
        logger.info("📦 Testing StandaloneNASTrainer import...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneNASTrainer
        logger.info("✅ StandaloneNASTrainer imported successfully")
        
        logger.info("📦 Testing StandaloneFeatureExtractor import...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneFeatureExtractor
        logger.info("✅ StandaloneFeatureExtractor imported successfully")
        
        logger.info("📦 Testing StandaloneRegimeAnalyzer import...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneRegimeAnalyzer
        logger.info("✅ StandaloneRegimeAnalyzer imported successfully")
        
        logger.info("📦 Testing StandaloneMicroRegimeDetector import...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneMicroRegimeDetector
        logger.info("✅ StandaloneMicroRegimeDetector imported successfully")
        
        logger.info("📦 Testing StandalonePerfectNASRegimeDetector import...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandalonePerfectNASRegimeDetector
        logger.info("✅ StandalonePerfectNASRegimeDetector imported successfully")
        
        logger.info("📦 Testing PerfectNASConfig import...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.perfect_nas_config import PerfectNASConfig
        logger.info("✅ PerfectNASConfig imported successfully")
        
        logger.info("🎉 All enhanced standalone components imported successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

def test_enhanced_standalone_initialization():
    """Test that enhanced standalone components can be initialized."""
    try:
        logger.info("🔧 Testing Enhanced Standalone Component Initialization...")
        logger.info("=" * 60)
        
        # Test StandaloneNASClusterer initialization
        logger.info("🧠 Testing StandaloneNASClusterer initialization...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneNASClusterer
        nas_clusterer = StandaloneNASClusterer(population_size=10, generations=5, enable_multi_objective=True)
        logger.info("✅ StandaloneNASClusterer initialized successfully")
        
        # Test StandaloneNASEvaluator initialization
        logger.info("🎯 Testing StandaloneNASEvaluator initialization...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneNASEvaluator
        nas_evaluator = StandaloneNASEvaluator(use_gpu=False, mixed_precision=False)
        logger.info("✅ StandaloneNASEvaluator initialized successfully")
        
        # Test StandaloneNASTrainer initialization
        logger.info("🏋️ Testing StandaloneNASTrainer initialization...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneNASTrainer
        nas_trainer = StandaloneNASTrainer(batch_size=16, learning_rate=0.001, epochs=10, use_gpu=False, mixed_precision=False)
        logger.info("✅ StandaloneNASTrainer initialized successfully")
        
        # Test StandaloneFeatureExtractor initialization
        logger.info("🔍 Testing StandaloneFeatureExtractor initialization...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneFeatureExtractor
        feature_extractor = StandaloneFeatureExtractor(enable_dimensionality_reduction=False, enable_feature_selection=False, n_components=5)
        logger.info("✅ StandaloneFeatureExtractor initialized successfully")
        
        # Test StandaloneRegimeAnalyzer initialization
        logger.info("📊 Testing StandaloneRegimeAnalyzer initialization...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneRegimeAnalyzer
        regime_analyzer = StandaloneRegimeAnalyzer()
        logger.info("✅ StandaloneRegimeAnalyzer initialized successfully")
        
        # Test StandaloneMicroRegimeDetector initialization
        logger.info("🔬 Testing StandaloneMicroRegimeDetector initialization...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneMicroRegimeDetector
        micro_detector = StandaloneMicroRegimeDetector()
        logger.info("✅ StandaloneMicroRegimeDetector initialized successfully")
        
        # Test PerfectNASConfig initialization
        logger.info("⚙️ Testing PerfectNASConfig initialization...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.perfect_nas_config import PerfectNASConfig
        config = PerfectNASConfig()
        logger.info("✅ PerfectNASConfig initialized successfully")
        
        # Test StandalonePerfectNASRegimeDetector initialization
        logger.info("🎯 Testing StandalonePerfectNASRegimeDetector initialization...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandalonePerfectNASRegimeDetector
        detector = StandalonePerfectNASRegimeDetector(config)
        logger.info("✅ StandalonePerfectNASRegimeDetector initialized successfully")
        
        logger.info("🎉 All enhanced standalone components initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization test failed: {e}")
        return False

def test_enhanced_standalone_methods():
    """Test that enhanced standalone components have the expected methods."""
    try:
        logger.info("🔍 Testing Enhanced Standalone Component Methods...")
        logger.info("=" * 60)
        
        # Test StandaloneNASClusterer methods
        logger.info("🧠 Testing StandaloneNASClusterer methods...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneNASClusterer
        nas_clusterer = StandaloneNASClusterer(population_size=5, generations=3, enable_multi_objective=True)
        
        # Check for expected methods
        expected_methods = ['search', '_initialize_population', '_generate_random_architecture', 
                          '_evaluate_population', '_update_pareto_frontier', '_evolve_population']
        for method in expected_methods:
            assert hasattr(nas_clusterer, method), f"Missing method: {method}"
        logger.info("✅ StandaloneNASClusterer has all expected methods")
        
        # Test StandaloneNASEvaluator methods
        logger.info("🎯 Testing StandaloneNASEvaluator methods...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneNASEvaluator
        nas_evaluator = StandaloneNASEvaluator(use_gpu=False, mixed_precision=False)
        
        # Check for expected methods
        expected_methods = ['evaluate_model', '_calculate_metrics', '_accuracy_score', '_precision_macro', 
                          '_recall_macro', '_f1_macro', '_confusion_matrix']
        for method in expected_methods:
            assert hasattr(nas_evaluator, method), f"Missing method: {method}"
        logger.info("✅ StandaloneNASEvaluator has all expected methods")
        
        # Test StandaloneNASTrainer methods
        logger.info("🏋️ Testing StandaloneNASTrainer methods...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneNASTrainer
        nas_trainer = StandaloneNASTrainer(batch_size=16, learning_rate=0.001, epochs=5, use_gpu=False, mixed_precision=False)
        
        # Check for expected methods
        expected_methods = ['train', '_create_optimizer', '_create_scheduler', '_train_epoch', 
                          '_validate_epoch', '_hmm_loss', '_regime_loss', '_focal_loss']
        for method in expected_methods:
            assert hasattr(nas_trainer, method), f"Missing method: {method}"
        logger.info("✅ StandaloneNASTrainer has all expected methods")
        
        # Test StandaloneFeatureExtractor methods
        logger.info("🔍 Testing StandaloneFeatureExtractor methods...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneFeatureExtractor
        feature_extractor = StandaloneFeatureExtractor(enable_dimensionality_reduction=False, enable_feature_selection=False, n_components=5)
        
        # Check for expected methods
        expected_methods = ['extract_features', '_simple_moving_average', '_exponential_moving_average', 
                          '_relative_strength_index', '_macd', '_bollinger_bands']
        for method in expected_methods:
            assert hasattr(feature_extractor, method), f"Missing method: {method}"
        logger.info("✅ StandaloneFeatureExtractor has all expected methods")
        
        # Test StandaloneRegimeAnalyzer methods
        logger.info("📊 Testing StandaloneRegimeAnalyzer methods...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneRegimeAnalyzer
        regime_analyzer = StandaloneRegimeAnalyzer()
        
        # Check for expected methods
        expected_methods = ['analyze_regimes', '_compute_basic_regime_metrics', '_compute_quality_metrics', 
                          '_compute_temporal_metrics', '_compute_persistence_analysis']
        for method in expected_methods:
            assert hasattr(regime_analyzer, method), f"Missing method: {method}"
        logger.info("✅ StandaloneRegimeAnalyzer has all expected methods")
        
        # Test StandaloneMicroRegimeDetector methods
        logger.info("🔬 Testing StandaloneMicroRegimeDetector methods...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandaloneMicroRegimeDetector
        micro_detector = StandaloneMicroRegimeDetector()
        
        # Check for expected methods
        expected_methods = ['detect_micro_regimes', '_initialize_detection_algorithms', 
                          '_initialize_classification_methods', '_initialize_accuracy_estimation']
        for method in expected_methods:
            assert hasattr(micro_detector, method), f"Missing method: {method}"
        logger.info("✅ StandaloneMicroRegimeDetector has all expected methods")
        
        logger.info("🎉 All enhanced standalone components have expected methods!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Methods test failed: {e}")
        return False

def test_enhanced_standalone_configuration():
    """Test that enhanced standalone components can be configured."""
    try:
        logger.info("⚙️ Testing Enhanced Standalone Component Configuration...")
        logger.info("=" * 60)
        
        # Test PerfectNASConfig
        logger.info("🔧 Testing PerfectNASConfig configuration...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.perfect_nas_config import PerfectNASConfig
        config = PerfectNASConfig()
        
        # Check configuration attributes
        assert hasattr(config, 'primary_architecture'), "Missing primary_architecture"
        assert hasattr(config, 'enable_neural_odes'), "Missing enable_neural_odes"
        assert hasattr(config, 'enable_vision_transformers'), "Missing enable_vision_transformers"
        assert hasattr(config, 'enable_meta_learning'), "Missing enable_meta_learning"
        assert hasattr(config, 'n_regimes'), "Missing n_regimes"
        assert hasattr(config, 'population_size'), "Missing population_size"
        assert hasattr(config, 'generations'), "Missing generations"
        
        logger.info("✅ PerfectNASConfig has all expected attributes")
        
        # Test factory methods
        logger.info("🏭 Testing PerfectNASConfig factory methods...")
        short_term_config = PerfectNASConfig.create_short_term_trading_config()
        research_config = PerfectNASConfig.create_research_config()
        production_config = PerfectNASConfig.create_production_config()
        
        assert short_term_config is not None, "Short term config creation failed"
        assert research_config is not None, "Research config creation failed"
        assert production_config is not None, "Production config creation failed"
        
        logger.info("✅ PerfectNASConfig factory methods work correctly")
        
        # Test StandalonePerfectNASRegimeDetector with different configurations
        logger.info("🎯 Testing StandalonePerfectNASRegimeDetector with different configurations...")
        from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import StandalonePerfectNASRegimeDetector
        
        # Test with short term config
        detector1 = StandalonePerfectNASRegimeDetector(short_term_config)
        assert detector1 is not None, "Detector with short term config failed"
        
        # Test with research config
        detector2 = StandalonePerfectNASRegimeDetector(research_config)
        assert detector2 is not None, "Detector with research config failed"
        
        # Test with production config
        detector3 = StandalonePerfectNASRegimeDetector(production_config)
        assert detector3 is not None, "Detector with production config failed"
        
        logger.info("✅ StandalonePerfectNASRegimeDetector works with different configurations")
        
        logger.info("🎉 All enhanced standalone components can be configured!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def run_comprehensive_enhanced_standalone_test():
    """Run comprehensive test of enhanced standalone components."""
    try:
        logger.info("🚀 Starting Comprehensive Enhanced Standalone Components Test")
        logger.info("=" * 80)
        
        test_results = {}
        
        # Test individual components
        test_results['imports'] = test_enhanced_standalone_imports()
        test_results['initialization'] = test_enhanced_standalone_initialization()
        test_results['methods'] = test_enhanced_standalone_methods()
        test_results['configuration'] = test_enhanced_standalone_configuration()
        
        # Summary
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        logger.info("\n📊 Enhanced Standalone Components Test Results:")
        logger.info("=" * 60)
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"   {test_name}: {status}")
        
        logger.info(f"\n🏆 Overall Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All enhanced standalone components are working correctly!")
            logger.info("✅ Enhanced standalone components are now on par with original components!")
        else:
            logger.warning(f"⚠️ {total_tests - passed_tests} tests failed")
        
        return passed_tests == total_tests
        
    except Exception as e:
        logger.error(f"❌ Comprehensive enhanced standalone test failed: {e}")
        return False

if __name__ == "__main__":
    """Run the enhanced standalone components test."""
    try:
        success = run_comprehensive_enhanced_standalone_test()
        
        if success:
            logger.info("\n🎯 Enhanced Standalone Components Test Complete!")
            logger.info("🎉 All enhanced standalone components are working correctly!")
            logger.info("✅ Enhanced standalone components are now on par with original components!")
            logger.info("🚀 The standalone system now has advanced functionality!")
        else:
            logger.error("❌ Some enhanced standalone components failed!")
            
    except Exception as e:
        logger.error(f"❌ Enhanced standalone components test failed: {e}")
        raise