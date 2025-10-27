"""
Comprehensive Test Suite for Enhanced SR Detection System

This test suite validates the enhanced SR detection system with:
- VectorBT optimization integration
- SHAP/LIME explainability
- Advanced validation with temporal CV and data leakage detection
- HPO integration for parameter optimization
- Hardware optimization for M1 Mac performance

Author: AI Assistant
Date: 2024
"""

import asyncio
import logging
import time
import unittest
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Test imports
try:
    from src.tactician.sr_levels.enhanced_sr_detection_optimized import (
        EnhancedSROptimizedDetector, SROptimizationConfig, SRLevel
    )
    from src.training.steps.market_analysis.components.sr_detection_enhanced import (
        EnhancedSRDetectionStep
    )
    ENHANCED_SR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced SR detection not available: {e}")
    ENHANCED_SR_AVAILABLE = False

# VectorBT and optimization imports
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, OperationType, OptimizationStrategy
    )
    VECTORIZATION_AVAILABLE = True
except ImportError:
    VECTORIZATION_AVAILABLE = False

# Hardware optimization imports
try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager, WorkloadType, OptimizationLevel
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# ML explainability imports
try:
    from src.utils.ml_common.explainability.shap_lime_integration import (
        SHAPLIMEExplainer, ExplanationConfig
    )
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False

# Validation imports
try:
    from src.utils.ml_common.validation.temporal_cross_validation import temporal_cross_validation
    from src.utils.ml_common.validation.data_leakage_detector import DataLeakageDetector
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

# HPO imports
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    from src.utils.ml_common.optimization.hpo_utils import HPOConfig
    HPO_AVAILABLE = True
except ImportError:
    HPO_AVAILABLE = False

class TestEnhancedSRDetection(unittest.TestCase):
    """Test suite for enhanced SR detection system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with sample data."""
        print("\n" + "="*80)
        print("🧪 ENHANCED SR DETECTION TEST SUITE")
        print("="*80)
        
        # Create sample market data
        cls.sample_data = cls._create_sample_market_data()
        
        # Test configuration
        cls.test_config = SROptimizationConfig(
            min_touches=2,
            tolerance_pct=0.5,
            lookback_periods=50,
            enable_vectorbt=True,
            enable_hardware_optimization=True,
            enable_explainability=True,
            enable_validation=True,
            enable_hpo=True
        )
        
        print(f"✅ Test setup completed")
        print(f"📊 Sample data: {len(cls.sample_data)} rows")
        print(f"🔧 Test config: {cls.test_config}")
    
    @staticmethod
    def _create_sample_market_data() -> pd.DataFrame:
        """Create sample market data for testing."""
        # Create realistic market data with clear SR levels
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='15T')
        np.random.seed(42)
        
        # Create base price with trend
        base_price = 2000.0
        trend = np.linspace(0, 0.1, len(dates))  # 10% upward trend
        noise = np.random.normal(0, 0.001, len(dates))
        returns = trend + noise
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Add clear support and resistance levels
        support_levels = [1950, 1980, 2010]
        resistance_levels = [2050, 2080, 2110]
        
        # Modify prices to touch these levels
        for i, price in enumerate(prices):
            # Check for support levels
            for support in support_levels:
                if abs(price - support) < 20:
                    prices[i] = support + np.random.normal(0, 5)
            
            # Check for resistance levels
            for resistance in resistance_levels:
                if abs(price - resistance) < 20:
                    prices[i] = resistance + np.random.normal(0, 5)
        
        # Create OHLCV data
        market_data = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, len(dates))
        }, index=dates)
        
        # Ensure high >= low
        market_data['high'] = np.maximum(market_data['high'], market_data['low'])
        
        return market_data
    
    def test_enhanced_sr_detector_initialization(self):
        """Test enhanced SR detector initialization."""
        print("\n🔧 Testing Enhanced SR Detector Initialization...")
        
        if not ENHANCED_SR_AVAILABLE:
            self.skipTest("Enhanced SR detection not available")
        
        try:
            # Test with default config
            detector = EnhancedSROptimizedDetector()
            self.assertIsInstance(detector, EnhancedSROptimizedDetector)
            print("✅ Default initialization successful")
            
            # Test with custom config
            detector_custom = EnhancedSROptimizedDetector(self.test_config)
            self.assertIsInstance(detector_custom, EnhancedSROptimizedDetector)
            print("✅ Custom config initialization successful")
            
            # Test optimization status
            status = detector.get_optimization_status()
            self.assertIsInstance(status, dict)
            print(f"✅ Optimization status: {status}")
            
        except Exception as e:
            self.fail(f"Initialization failed: {e}")
    
    def test_sr_level_detection(self):
        """Test SR level detection functionality."""
        print("\n🎯 Testing SR Level Detection...")
        
        if not ENHANCED_SR_AVAILABLE:
            self.skipTest("Enhanced SR detection not available")
        
        try:
            detector = EnhancedSROptimizedDetector(self.test_config)
            sr_levels = detector.detect_sr_levels(self.sample_data)
            
            self.assertIsInstance(sr_levels, list)
            print(f"✅ Detected {len(sr_levels)} SR levels")
            
            # Validate SR level structure
            for level in sr_levels:
                self.assertIsInstance(level, SRLevel)
                self.assertIn(level.level_type, ['support', 'resistance'])
                self.assertGreaterEqual(level.strength, 0.0)
                self.assertLessEqual(level.strength, 1.0)
                self.assertGreaterEqual(level.touches, 1)
                self.assertGreaterEqual(level.quality_score, 0.0)
                self.assertLessEqual(level.quality_score, 1.0)
            
            print("✅ SR level structure validation passed")
            
            # Test performance metrics
            metrics = detector.get_performance_metrics()
            self.assertIsInstance(metrics, dict)
            print(f"✅ Performance metrics: {metrics}")
            
        except Exception as e:
            self.fail(f"SR level detection failed: {e}")
    
    def test_vectorbt_integration(self):
        """Test VectorBT integration."""
        print("\n⚡ Testing VectorBT Integration...")
        
        if not VECTORIZATION_AVAILABLE:
            self.skipTest("VectorBT not available")
        
        try:
            # Test VectorBT manager initialization
            vectorization_manager = UnifiedVectorizationManager()
            self.assertIsInstance(vectorization_manager, UnifiedVectorizationManager)
            print("✅ VectorBT manager initialized")
            
            # Test operation type selection
            operation_type = OperationType.TECHNICAL_INDICATORS
            strategy = vectorization_manager._select_optimal_strategy(
                operation_type, len(self.sample_data), self.sample_data.shape
            )
            self.assertIsInstance(strategy, OptimizationStrategy)
            print(f"✅ Optimal strategy selected: {strategy}")
            
        except Exception as e:
            self.fail(f"VectorBT integration failed: {e}")
    
    def test_hardware_optimization(self):
        """Test hardware optimization."""
        print("\n🖥️ Testing Hardware Optimization...")
        
        if not HARDWARE_OPTIMIZATION_AVAILABLE:
            self.skipTest("Hardware optimization not available")
        
        try:
            # Test hardware manager initialization
            hardware_manager = UnifiedHardwareManager()
            hardware_manager.initialize()
            self.assertIsInstance(hardware_manager, UnifiedHardwareManager)
            print("✅ Hardware manager initialized")
            
            # Test workload optimization
            hardware_manager.optimize_for_workload(
                WorkloadType.ML_TRAINING, 
                OptimizationLevel.BALANCED
            )
            print("✅ Workload optimization successful")
            
        except Exception as e:
            self.fail(f"Hardware optimization failed: {e}")
    
    def test_ml_explainability(self):
        """Test ML explainability integration."""
        print("\n🧠 Testing ML Explainability...")
        
        if not EXPLAINABILITY_AVAILABLE:
            self.skipTest("ML explainability not available")
        
        try:
            # Test explainer initialization
            explanation_config = ExplanationConfig(
                enable_shap=True,
                enable_lime=True,
                shap_sample_size=50,
                lime_sample_size=500
            )
            explainer = SHAPLIMEExplainer(explanation_config)
            self.assertIsInstance(explainer, SHAPLIMEExplainer)
            print("✅ Explainability initialized")
            
        except Exception as e:
            self.fail(f"ML explainability failed: {e}")
    
    def test_validation_components(self):
        """Test validation components."""
        print("\n🔍 Testing Validation Components...")
        
        if not VALIDATION_AVAILABLE:
            self.skipTest("Validation components not available")
        
        try:
            # Test data leakage detector
            leakage_detector = DataLeakageDetector()
            self.assertIsInstance(leakage_detector, DataLeakageDetector)
            print("✅ Data leakage detector initialized")
            
            # Test temporal cross validation
            cv_result = temporal_cross_validation(
                X=self.sample_data[['open', 'high', 'low', 'close']].values,
                y=self.sample_data['close'].values,
                n_splits=3,
                gap=5
            )
            self.assertIsInstance(cv_result, dict)
            print("✅ Temporal cross validation successful")
            
        except Exception as e:
            self.fail(f"Validation components failed: {e}")
    
    def test_hpo_integration(self):
        """Test HPO integration."""
        print("\n🎯 Testing HPO Integration...")
        
        if not HPO_AVAILABLE:
            self.skipTest("HPO not available")
        
        try:
            # Test HPO optimizer initialization
            hpo_config = HPOConfig(
                n_trials=10,
                timeout=60,
                direction='maximize'
            )
            hpo_optimizer = BayesianTPEOptimizer(hpo_config)
            self.assertIsInstance(hpo_optimizer, BayesianTPEOptimizer)
            print("✅ HPO optimizer initialized")
            
        except Exception as e:
            self.fail(f"HPO integration failed: {e}")
    
    def test_enhanced_sr_detection_step(self):
        """Test enhanced SR detection step."""
        print("\n📊 Testing Enhanced SR Detection Step...")
        
        if not ENHANCED_SR_AVAILABLE:
            self.skipTest("Enhanced SR detection not available")
        
        try:
            # Test step initialization
            step = EnhancedSRDetectionStep()
            self.assertIsInstance(step, EnhancedSRDetectionStep)
            print("✅ Step initialization successful")
            
            # Test input validation
            input_data = {
                'market_data': self.sample_data,
                'config': {
                    'sr_detection': {
                        'min_touches': 2,
                        'tolerance_pct': 0.5,
                        'enable_vectorbt': True
                    }
                }
            }
            
            is_valid = step.validate_input(input_data)
            self.assertTrue(is_valid)
            print("✅ Input validation passed")
            
            # Test step execution
            result = asyncio.run(step.execute(input_data))
            self.assertIsInstance(result, dict)
            self.assertIn('sr_levels', result)
            self.assertIn('metadata', result)
            self.assertIn('performance_metrics', result)
            print(f"✅ Step execution successful: {len(result['sr_levels'])} levels detected")
            
            # Test step info
            step_info = step.get_step_info()
            self.assertIsInstance(step_info, dict)
            print(f"✅ Step info: {step_info}")
            
        except Exception as e:
            self.fail(f"Enhanced SR detection step failed: {e}")
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking."""
        print("\n⚡ Testing Performance Benchmarking...")
        
        if not ENHANCED_SR_AVAILABLE:
            self.skipTest("Enhanced SR detection not available")
        
        try:
            # Test with different data sizes
            data_sizes = [100, 500, 1000, 2000]
            performance_results = {}
            
            for size in data_sizes:
                # Create subset of data
                subset_data = self.sample_data.head(size)
                
                # Test detection time
                start_time = time.time()
                detector = EnhancedSROptimizedDetector(self.test_config)
                sr_levels = detector.detect_sr_levels(subset_data)
                detection_time = time.time() - start_time
                
                performance_results[size] = {
                    'detection_time': detection_time,
                    'levels_detected': len(sr_levels),
                    'levels_per_second': len(sr_levels) / detection_time if detection_time > 0 else 0
                }
                
                print(f"✅ Data size {size}: {detection_time:.3f}s, {len(sr_levels)} levels")
            
            # Validate performance scaling
            self.assertIsInstance(performance_results, dict)
            print(f"✅ Performance benchmarking completed: {performance_results}")
            
        except Exception as e:
            self.fail(f"Performance benchmarking failed: {e}")
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        print("\n🛡️ Testing Error Handling...")
        
        if not ENHANCED_SR_AVAILABLE:
            self.skipTest("Enhanced SR detection not available")
        
        try:
            detector = EnhancedSROptimizedDetector(self.test_config)
            
            # Test with invalid data
            invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
            result = detector.detect_sr_levels(invalid_data)
            self.assertEqual(result, [])
            print("✅ Invalid data handling passed")
            
            # Test with empty data
            empty_data = pd.DataFrame()
            result = detector.detect_sr_levels(empty_data)
            self.assertEqual(result, [])
            print("✅ Empty data handling passed")
            
            # Test with insufficient data
            insufficient_data = self.sample_data.head(5)
            result = detector.detect_sr_levels(insufficient_data)
            self.assertIsInstance(result, list)
            print("✅ Insufficient data handling passed")
            
        except Exception as e:
            self.fail(f"Error handling failed: {e}")
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        print("\n⚙️ Testing Configuration Validation...")
        
        if not ENHANCED_SR_AVAILABLE:
            self.skipTest("Enhanced SR detection not available")
        
        try:
            # Test valid configuration
            valid_config = SROptimizationConfig(
                min_touches=2,
                tolerance_pct=0.5,
                lookback_periods=100
            )
            detector = EnhancedSROptimizedDetector(valid_config)
            self.assertIsInstance(detector, EnhancedSROptimizedDetector)
            print("✅ Valid configuration passed")
            
            # Test edge case configurations
            edge_config = SROptimizationConfig(
                min_touches=1,
                tolerance_pct=0.01,
                lookback_periods=10
            )
            detector_edge = EnhancedSROptimizedDetector(edge_config)
            self.assertIsInstance(detector_edge, EnhancedSROptimizedDetector)
            print("✅ Edge case configuration passed")
            
        except Exception as e:
            self.fail(f"Configuration validation failed: {e}")

def run_comprehensive_tests():
    """Run comprehensive test suite."""
    print("\n" + "="*80)
    print("🚀 RUNNING COMPREHENSIVE ENHANCED SR DETECTION TESTS")
    print("="*80)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedSRDetection)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()

def run_quick_tests():
    """Run quick validation tests."""
    print("\n" + "="*80)
    print("⚡ RUNNING QUICK VALIDATION TESTS")
    print("="*80)
    
    try:
        # Test basic functionality
        if ENHANCED_SR_AVAILABLE:
            print("✅ Enhanced SR detection available")
            
            # Test detector initialization
            detector = EnhancedSROptimizedDetector()
            print("✅ Detector initialization successful")
            
            # Test with sample data
            sample_data = TestEnhancedSRDetection._create_sample_market_data()
            sr_levels = detector.detect_sr_levels(sample_data)
            print(f"✅ SR detection successful: {len(sr_levels)} levels")
            
            # Test step functionality
            step = EnhancedSRDetectionStep()
            input_data = {
                'market_data': sample_data,
                'config': {'sr_detection': {}}
            }
            result = asyncio.run(step.execute(input_data))
            print(f"✅ Step execution successful: {len(result['sr_levels'])} levels")
            
        else:
            print("❌ Enhanced SR detection not available")
            return False
        
        print("\n✅ All quick tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Quick tests failed: {e}")
        return False

if __name__ == "__main__":
    # Run quick tests first
    if run_quick_tests():
        print("\n" + "="*80)
        print("🎉 QUICK TESTS PASSED - RUNNING COMPREHENSIVE TESTS")
        print("="*80)
        
        # Run comprehensive tests
        success = run_comprehensive_tests()
        
        if success:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print("\n❌ SOME TESTS FAILED!")
    else:
        print("\n❌ QUICK TESTS FAILED - SKIPPING COMPREHENSIVE TESTS")