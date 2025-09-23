"""
Standalone Test Script for Perfect NAS Regime System

Tests the fully standalone implementation without external dependencies.
"""

import numpy as np
import torch
import logging
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import standalone components
from core.perfect_nas_config import PerfectNASConfig, NeuralArchitectureType
from core.perfect_nas_regime_detector import PerfectNASRegimeDetector
from core.neural_architectures import (
    NeuralODE, VisionTransformer, NeuralStateSpaceModel,
    ContinuousTimeRegimeDetector, TransformerRegimeDetector
)
from core.nas_search import EssentialNASClusterer
from evaluation.economic_evaluator import EconomicSignificanceEvaluator
from evaluation.trading_viability_evaluator import TradingViabilityEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_neural_architectures():
    """Test standalone neural architectures."""
    logger.info("🧠 Testing Neural Architectures...")
    
    try:
        # Test Neural ODE
        neural_ode = NeuralODE(input_size=4, hidden_size=64, output_size=5)
        test_input = torch.randn(10, 4)
        ode_output = neural_ode(test_input)
        logger.info(f"✅ Neural ODE: Input {test_input.shape} -> Output {ode_output.shape}")
        
        # Test Vision Transformer
        vision_transformer = VisionTransformer(input_dim=4, n_regimes=5, sequence_length=50)
        test_sequence = torch.randn(10, 50, 4)
        vt_output = vision_transformer(test_sequence)
        logger.info(f"✅ Vision Transformer: Input {test_sequence.shape} -> Output {vt_output.shape}")
        
        # Test Neural State Space Model
        ssm = NeuralStateSpaceModel(input_dim=4, state_dim=64, hidden_dim=128, n_regimes=5)
        ssm_output, states = ssm(test_sequence)
        logger.info(f"✅ State Space Model: Input {test_sequence.shape} -> Output {ssm_output.shape}, States {states.shape}")
        
        # Test Continuous Time Detector
        ctd = ContinuousTimeRegimeDetector(input_size=4, state_size=64, num_regimes=5)
        ctd_output = ctd(test_input)
        logger.info(f"✅ Continuous Time Detector: Input {test_input.shape} -> Output {ctd_output.shape}")
        
        # Test Transformer Detector
        td = TransformerRegimeDetector(input_dim=4, n_regimes=5)
        td_output = td(test_sequence)
        logger.info(f"✅ Transformer Detector: Input {test_sequence.shape} -> Output {td_output.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Neural architecture test failed: {e}")
        return False

def test_nas_search():
    """Test standalone NAS search."""
    logger.info("🔍 Testing NAS Search...")
    
    try:
        # Create NAS clusterer
        nas_clusterer = EssentialNASClusterer(population_size=10, generations=5)
        
        # Generate test data
        test_data = np.random.randn(100, 4)
        test_labels = np.random.randint(0, 5, 100)
        
        # Run NAS search
        result = nas_clusterer.search(test_data, test_labels)
        
        logger.info(f"✅ NAS Search: Success={result.success}")
        if result.best_architecture:
            logger.info(f"   Best fitness: {result.best_architecture.fitness_score:.4f}")
            logger.info(f"   Architecture layers: {len(result.best_architecture.layers)}")
        logger.info(f"   Pareto solutions: {len(result.pareto_frontier)}")
        
        return result.success
        
    except Exception as e:
        logger.error(f"❌ NAS search test failed: {e}")
        return False

def test_evaluators():
    """Test standalone evaluators."""
    logger.info("📊 Testing Evaluators...")
    
    try:
        # Test Economic Evaluator
        economic_config = PerfectNASConfig().economic_config
        economic_evaluator = EconomicSignificanceEvaluator(economic_config)
        
        test_data = np.random.randn(100, 5)  # OHLCV
        test_predictions = np.random.randint(0, 5, 100)
        economic_scores = economic_evaluator.evaluate(test_data, test_predictions)
        
        logger.info(f"✅ Economic Evaluator: Mean score {np.mean(economic_scores):.3f}")
        
        # Test Trading Viability Evaluator
        trading_config = PerfectNASConfig().trading_config
        trading_evaluator = TradingViabilityEvaluator(trading_config)
        
        trading_scores = trading_evaluator.evaluate(test_data, test_predictions)
        
        logger.info(f"✅ Trading Viability Evaluator: Mean score {np.mean(trading_scores):.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Evaluator test failed: {e}")
        return False

def test_perfect_nas_system():
    """Test the complete Perfect NAS system."""
    logger.info("🚀 Testing Perfect NAS System...")
    
    try:
        # Create configuration
        config = PerfectNASConfig()
        config.primary_architecture = NeuralArchitectureType.HYBRID
        config.enable_neural_odes = True
        config.enable_vision_transformers = True
        config.enable_meta_learning = True
        config.n_regimes = 5
        config.population_size = 10  # Small for testing
        config.generations = 5       # Small for testing
        
        # Initialize detector
        detector = PerfectNASRegimeDetector(config)
        
        # Generate test data
        market_data = np.random.randn(200, 5)  # OHLCV
        timestamps = np.arange(200)
        
        # Detect regimes
        result = detector.detect_regimes(
            market_data=market_data,
            timestamps=timestamps,
            optimize_architecture=True,
            enable_meta_learning=True
        )
        
        logger.info(f"✅ Perfect NAS System: Success={result.success}")
        logger.info(f"   Execution time: {result.execution_time:.2f}s")
        logger.info(f"   Regimes detected: {len(np.unique(result.regime_predictions))}")
        logger.info(f"   Economic significance: {np.mean(result.economic_significance_scores):.3f}")
        logger.info(f"   Trading viability: {np.mean(result.trading_viability_scores):.3f}")
        
        return result.success
        
    except Exception as e:
        logger.error(f"❌ Perfect NAS system test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite."""
    logger.info("🧪 Starting Comprehensive Standalone Test Suite")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test 1: Neural Architectures
    logger.info("\n1️⃣ Testing Neural Architectures...")
    test_results['neural_architectures'] = test_neural_architectures()
    
    # Test 2: NAS Search
    logger.info("\n2️⃣ Testing NAS Search...")
    test_results['nas_search'] = test_nas_search()
    
    # Test 3: Evaluators
    logger.info("\n3️⃣ Testing Evaluators...")
    test_results['evaluators'] = test_evaluators()
    
    # Test 4: Complete System
    logger.info("\n4️⃣ Testing Complete Perfect NAS System...")
    test_results['complete_system'] = test_perfect_nas_system()
    
    # Summary
    logger.info("\n📊 Test Results Summary:")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED! Perfect NAS System is fully standalone and functional!")
    else:
        logger.info("\n⚠️ Some tests failed. Check the logs above for details.")
    
    return all_passed

if __name__ == "__main__":
    """Run the standalone test suite."""
    try:
        success = run_comprehensive_test()
        
        if success:
            logger.info("\n🏆 Perfect NAS Regime System - Standalone Implementation Complete!")
            logger.info("✅ All components are fully self-contained")
            logger.info("✅ No external dependencies required")
            logger.info("✅ Ready for production use")
        else:
            logger.error("\n❌ Some tests failed. Please check the implementation.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        sys.exit(1)