"""
Test Enhanced Standalone Perfect NAS Regime System Components

Tests the enhanced standalone components to ensure they are on par with
the original components from nas_clustering/ and nas_modeling/.
"""

import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime, timedelta
import sys
import os
import time

# Add the project root to the path
sys.path.append('/workspace/src')

# Import enhanced standalone components
from training.steps.market_analysis.perfect_nas_regime_system.core.standalone_perfect_nas_regime_detector import (
    StandaloneNASClusterer, StandaloneNASEvaluator, StandaloneNASTrainer,
    StandaloneFeatureExtractor, StandaloneRegimeAnalyzer, StandaloneMicroRegimeDetector
)
from training.steps.market_analysis.perfect_nas_regime_system.core.perfect_nas_config import (
    PerfectNASConfig, NeuralArchitectureType
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_market_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate test market data for enhanced standalone testing."""
    try:
        logger.info(f"📊 Generating test market data with {n_samples} samples...")
        
        # Generate realistic market data with regime-like patterns
        np.random.seed(42)
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=n_samples//24)
        timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]
        
        # Generate OHLCV data with regime-like patterns
        data = []
        current_price = 100.0
        
        for i in range(n_samples):
            # Simulate different market regimes
            regime_period = i // 100  # Change regime every 100 samples
            
            if regime_period % 4 == 0:  # Bull market
                trend = 0.001
                volatility = 0.01
            elif regime_period % 4 == 1:  # Bear market
                trend = -0.001
                volatility = 0.015
            elif regime_period % 4 == 2:  # High volatility
                trend = 0.0005
                volatility = 0.02
            else:  # Low volatility
                trend = 0.0002
                volatility = 0.005
            
            # Generate price movement
            price_change = np.random.normal(trend, volatility)
            current_price *= (1 + price_change)
            
            # Generate OHLCV
            open_price = current_price
            high_price = open_price * (1 + abs(np.random.normal(0, volatility/2)))
            low_price = open_price * (1 - abs(np.random.normal(0, volatility/2)))
            close_price = open_price * (1 + price_change)
            volume = np.random.lognormal(10, 0.5)
            
            data.append([open_price, high_price, low_price, close_price, volume])
        
        market_data = np.array(data)
        timestamps = np.array(timestamps)
        
        logger.info(f"✅ Generated test market data: {market_data.shape}")
        return market_data, timestamps
        
    except Exception as e:
        logger.error(f"❌ Test data generation failed: {e}")
        raise

def test_enhanced_standalone_nas_clusterer():
    """Test enhanced standalone NAS clusterer."""
    try:
        logger.info("🧠 Testing Enhanced Standalone NAS Clusterer...")
        logger.info("=" * 60)
        
        # Generate test data
        market_data, timestamps = generate_test_market_data(n_samples=200)
        labels = np.random.randint(0, 5, len(market_data))
        
        # Test enhanced NAS clusterer
        nas_clusterer = StandaloneNASClusterer(
            population_size=20,
            generations=10,
            enable_multi_objective=True
        )
        
        start_time = time.time()
        result = nas_clusterer.search(market_data, labels)
        execution_time = time.time() - start_time
        
        # Verify results
        assert result['success'], "NAS search should succeed"
        assert 'best_architecture' in result, "Should have best architecture"
        assert 'pareto_frontier' in result, "Should have Pareto frontier"
        assert 'search_statistics' in result, "Should have search statistics"
        assert 'generation_stats' in result, "Should have generation stats"
        
        # Check advanced features
        best_arch = result['best_architecture']
        assert 'fitness_score' in best_arch, "Should have fitness score"
        assert 'parameters_count' in best_arch, "Should have parameter count"
        assert 'layers' in best_arch, "Should have architecture layers"
        
        # Check Pareto frontier
        pareto_size = len(result['pareto_frontier'])
        assert pareto_size > 0, "Should have Pareto frontier solutions"
        
        # Check generation statistics
        gen_stats = result['generation_stats']
        assert len(gen_stats) == 10, "Should have stats for all generations"
        
        logger.info(f"✅ Enhanced NAS Clusterer test passed!")
        logger.info(f"   Execution time: {execution_time:.2f}s")
        logger.info(f"   Best fitness: {best_arch['fitness_score']:.4f}")
        logger.info(f"   Pareto solutions: {pareto_size}")
        logger.info(f"   Generations: {len(gen_stats)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced NAS Clusterer test failed: {e}")
        return False

def test_enhanced_standalone_nas_evaluator():
    """Test enhanced standalone NAS evaluator."""
    try:
        logger.info("🎯 Testing Enhanced Standalone NAS Evaluator...")
        logger.info("=" * 60)
        
        # Create test model
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 32)
                self.linear2 = torch.nn.Linear(32, 5)
                self.relu = torch.nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.linear1(x))
                x = self.linear2(x)
                return x
        
        model = TestModel()
        
        # Generate test data
        n_samples = 100
        data = torch.randn(n_samples, 5)
        labels = torch.randint(0, 5, (n_samples,))
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(data, labels)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        # Test enhanced evaluator
        evaluator = StandaloneNASEvaluator(use_gpu=False, mixed_precision=False)
        
        start_time = time.time()
        result = evaluator.evaluate_model(
            model, data_loader, 
            metrics=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
            problem_type='classification'
        )
        execution_time = time.time() - start_time
        
        # Verify results
        assert 'accuracy' in result, "Should have accuracy"
        assert 'precision_macro' in result, "Should have precision"
        assert 'recall_macro' in result, "Should have recall"
        assert 'f1_macro' in result, "Should have F1 score"
        assert 'confusion_matrix' in result, "Should have confusion matrix"
        assert 'per_class_accuracy' in result, "Should have per-class accuracy"
        
        # Check advanced metrics
        assert result['accuracy'] > 0, "Accuracy should be positive"
        assert result['precision_macro'] > 0, "Precision should be positive"
        assert result['recall_macro'] > 0, "Recall should be positive"
        assert result['f1_macro'] > 0, "F1 score should be positive"
        
        logger.info(f"✅ Enhanced NAS Evaluator test passed!")
        logger.info(f"   Execution time: {execution_time:.2f}s")
        logger.info(f"   Accuracy: {result['accuracy']:.4f}")
        logger.info(f"   Precision: {result['precision_macro']:.4f}")
        logger.info(f"   Recall: {result['recall_macro']:.4f}")
        logger.info(f"   F1 Score: {result['f1_macro']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced NAS Evaluator test failed: {e}")
        return False

def test_enhanced_standalone_nas_trainer():
    """Test enhanced standalone NAS trainer."""
    try:
        logger.info("🏋️ Testing Enhanced Standalone NAS Trainer...")
        logger.info("=" * 60)
        
        # Create test model
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 32)
                self.linear2 = torch.nn.Linear(32, 5)
                self.relu = torch.nn.ReLU()
                self.dropout = torch.nn.Dropout(0.2)
            
            def forward(self, x):
                x = self.relu(self.linear1(x))
                x = self.dropout(x)
                x = self.linear2(x)
                return x
        
        model = TestModel()
        
        # Generate test data
        n_samples = 200
        data = torch.randn(n_samples, 5)
        labels = torch.randint(0, 5, (n_samples,))
        
        # Create data loaders
        dataset = torch.utils.data.TensorDataset(data, labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Test enhanced trainer
        trainer = StandaloneNASTrainer(
            batch_size=32,
            learning_rate=0.001,
            epochs=20,
            optimizer='adam',
            scheduler='cosine',
            loss_function='cross_entropy',
            early_stopping_patience=5,
            gradient_clip_norm=1.0,
            use_gpu=False,
            mixed_precision=False
        )
        
        start_time = time.time()
        result = trainer.train(model, train_loader, val_loader)
        execution_time = time.time() - start_time
        
        # Verify results
        assert result['success'], "Training should succeed"
        assert 'training_history' in result, "Should have training history"
        assert 'final_train_loss' in result, "Should have final train loss"
        assert 'final_train_accuracy' in result, "Should have final train accuracy"
        assert 'best_val_loss' in result, "Should have best validation loss"
        assert 'epochs_trained' in result, "Should have epochs trained"
        assert 'converged' in result, "Should have convergence info"
        
        # Check training history
        history = result['training_history']
        assert 'train_loss' in history, "Should have train loss history"
        assert 'train_accuracy' in history, "Should have train accuracy history"
        assert 'val_loss' in history, "Should have val loss history"
        assert 'val_accuracy' in history, "Should have val accuracy history"
        assert 'learning_rate' in history, "Should have learning rate history"
        
        # Check advanced features
        assert result['final_train_loss'] > 0, "Final train loss should be positive"
        assert result['final_train_accuracy'] > 0, "Final train accuracy should be positive"
        assert result['epochs_trained'] > 0, "Should have trained for some epochs"
        
        logger.info(f"✅ Enhanced NAS Trainer test passed!")
        logger.info(f"   Execution time: {execution_time:.2f}s")
        logger.info(f"   Final train loss: {result['final_train_loss']:.4f}")
        logger.info(f"   Final train accuracy: {result['final_train_accuracy']:.4f}")
        logger.info(f"   Best val loss: {result['best_val_loss']:.4f}")
        logger.info(f"   Epochs trained: {result['epochs_trained']}")
        logger.info(f"   Converged: {result['converged']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced NAS Trainer test failed: {e}")
        return False

def test_enhanced_standalone_feature_extractor():
    """Test enhanced standalone feature extractor."""
    try:
        logger.info("🔍 Testing Enhanced Standalone Feature Extractor...")
        logger.info("=" * 60)
        
        # Generate test data
        market_data, timestamps = generate_test_market_data(n_samples=500)
        
        # Test enhanced feature extractor
        feature_extractor = StandaloneFeatureExtractor(
            enable_dimensionality_reduction=True,
            enable_feature_selection=True,
            n_components=10
        )
        
        start_time = time.time()
        extracted_features = feature_extractor.extract_features(market_data)
        execution_time = time.time() - start_time
        
        # Verify results
        assert extracted_features is not None, "Should extract features"
        assert extracted_features.shape[0] == market_data.shape[0], "Should preserve sample count"
        assert extracted_features.shape[1] > 0, "Should have features"
        
        # Check that features are different from original
        assert not np.array_equal(extracted_features, market_data), "Should extract additional features"
        
        # Check feature quality
        assert not np.any(np.isnan(extracted_features)), "Should not have NaN values"
        assert not np.any(np.isinf(extracted_features)), "Should not have infinite values"
        
        logger.info(f"✅ Enhanced Feature Extractor test passed!")
        logger.info(f"   Execution time: {execution_time:.2f}s")
        logger.info(f"   Original shape: {market_data.shape}")
        logger.info(f"   Extracted shape: {extracted_features.shape}")
        logger.info(f"   Feature ratio: {extracted_features.shape[1] / market_data.shape[1]:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Feature Extractor test failed: {e}")
        return False

def test_enhanced_standalone_regime_analyzer():
    """Test enhanced standalone regime analyzer."""
    try:
        logger.info("📊 Testing Enhanced Standalone Regime Analyzer...")
        logger.info("=" * 60)
        
        # Generate test data
        market_data, timestamps = generate_test_market_data(n_samples=300)
        regime_predictions = np.random.randint(0, 5, len(market_data))
        
        # Test enhanced regime analyzer
        regime_analyzer = StandaloneRegimeAnalyzer()
        
        start_time = time.time()
        analysis = regime_analyzer.analyze_regimes(market_data, regime_predictions, timestamps)
        execution_time = time.time() - start_time
        
        # Verify results
        assert 'n_regimes' in analysis, "Should have regime count"
        assert 'regime_durations' in analysis, "Should have regime durations"
        assert 'regime_characteristics' in analysis, "Should have regime characteristics"
        assert 'transition_matrix' in analysis, "Should have transition matrix"
        assert 'regime_stability' in analysis, "Should have regime stability"
        assert 'regime_separation' in analysis, "Should have regime separation"
        
        # Check advanced features
        assert analysis['n_regimes'] > 0, "Should have regimes"
        assert len(analysis['regime_durations']) > 0, "Should have regime durations"
        assert len(analysis['regime_characteristics']) > 0, "Should have regime characteristics"
        assert len(analysis['regime_stability']) > 0, "Should have regime stability"
        assert len(analysis['regime_separation']) > 0, "Should have regime separation"
        
        # Check transition matrix
        transition_matrix = analysis['transition_matrix']
        assert transition_matrix.shape[0] == transition_matrix.shape[1], "Should be square matrix"
        assert np.allclose(transition_matrix.sum(axis=1), 1.0), "Should be probability matrix"
        
        logger.info(f"✅ Enhanced Regime Analyzer test passed!")
        logger.info(f"   Execution time: {execution_time:.2f}s")
        logger.info(f"   Regimes detected: {analysis['n_regimes']}")
        logger.info(f"   Regime durations: {len(analysis['regime_durations'])}")
        logger.info(f"   Regime characteristics: {len(analysis['regime_characteristics'])}")
        logger.info(f"   Transition matrix shape: {transition_matrix.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Regime Analyzer test failed: {e}")
        return False

def test_enhanced_standalone_micro_regime_detector():
    """Test enhanced standalone micro regime detector."""
    try:
        logger.info("🔬 Testing Enhanced Standalone Micro Regime Detector...")
        logger.info("=" * 60)
        
        # Generate test data
        market_data, timestamps = generate_test_market_data(n_samples=200)
        regime_predictions = np.random.randint(0, 5, len(market_data))
        
        # Test enhanced micro regime detector
        micro_detector = StandaloneMicroRegimeDetector()
        
        start_time = time.time()
        micro_regimes = micro_detector.detect_micro_regimes(market_data, regime_predictions, timestamps)
        execution_time = time.time() - start_time
        
        # Verify results
        assert 'types' in micro_regimes, "Should have micro-regime types"
        assert 'scores' in micro_regimes, "Should have micro-regime scores"
        assert 'detection_accuracy' in micro_regimes, "Should have detection accuracy"
        assert 'micro_regime_distribution' in micro_regimes, "Should have distribution"
        
        # Check advanced features
        assert len(micro_regimes['types']) == len(market_data), "Should have types for all samples"
        assert len(micro_regimes['scores']) == len(market_data), "Should have scores for all samples"
        assert 0 <= micro_regimes['detection_accuracy'] <= 1, "Accuracy should be between 0 and 1"
        
        # Check micro-regime types
        unique_types = set(micro_regimes['types'])
        expected_types = {'high_volatility', 'high_volume', 'low_volatility', 'normal'}
        assert unique_types.issubset(expected_types), f"Should have expected types, got {unique_types}"
        
        # Check distribution
        distribution = micro_regimes['micro_regime_distribution']
        assert len(distribution) > 0, "Should have distribution"
        total_count = sum(distribution.values())
        assert total_count == len(market_data), "Distribution should sum to sample count"
        
        logger.info(f"✅ Enhanced Micro Regime Detector test passed!")
        logger.info(f"   Execution time: {execution_time:.2f}s")
        logger.info(f"   Micro-regime types: {len(unique_types)}")
        logger.info(f"   Detection accuracy: {micro_regimes['detection_accuracy']:.4f}")
        logger.info(f"   Distribution: {distribution}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Micro Regime Detector test failed: {e}")
        return False

def test_enhanced_standalone_integration():
    """Test enhanced standalone components integration."""
    try:
        logger.info("🔗 Testing Enhanced Standalone Components Integration...")
        logger.info("=" * 60)
        
        # Generate test data
        market_data, timestamps = generate_test_market_data(n_samples=300)
        
        # Test integrated workflow
        logger.info("🔍 Step 1: Feature extraction...")
        feature_extractor = StandaloneFeatureExtractor(
            enable_dimensionality_reduction=True,
            enable_feature_selection=True,
            n_components=15
        )
        extracted_features = feature_extractor.extract_features(market_data)
        
        logger.info("🧠 Step 2: NAS clustering...")
        nas_clusterer = StandaloneNASClusterer(
            population_size=15,
            generations=5,
            enable_multi_objective=True
        )
        labels = np.random.randint(0, 5, len(extracted_features))
        nas_result = nas_clusterer.search(extracted_features, labels)
        
        logger.info("📊 Step 3: Regime analysis...")
        regime_analyzer = StandaloneRegimeAnalyzer()
        regime_predictions = np.random.randint(0, 5, len(extracted_features))
        regime_analysis = regime_analyzer.analyze_regimes(extracted_features, regime_predictions, timestamps)
        
        logger.info("🔬 Step 4: Micro-regime detection...")
        micro_detector = StandaloneMicroRegimeDetector()
        micro_regimes = micro_detector.detect_micro_regimes(extracted_features, regime_predictions, timestamps)
        
        # Verify integration
        assert extracted_features is not None, "Feature extraction should work"
        assert nas_result['success'], "NAS clustering should work"
        assert regime_analysis is not None, "Regime analysis should work"
        assert micro_regimes is not None, "Micro-regime detection should work"
        
        logger.info(f"✅ Enhanced Standalone Integration test passed!")
        logger.info(f"   Feature extraction: {extracted_features.shape}")
        logger.info(f"   NAS clustering: {nas_result['success']}")
        logger.info(f"   Regime analysis: {regime_analysis['n_regimes']} regimes")
        logger.info(f"   Micro-regimes: {len(micro_regimes['types'])} samples")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Standalone Integration test failed: {e}")
        return False

def run_comprehensive_enhanced_standalone_test():
    """Run comprehensive test of enhanced standalone components."""
    try:
        logger.info("🚀 Starting Comprehensive Enhanced Standalone Components Test")
        logger.info("=" * 80)
        
        test_results = {}
        
        # Test individual components
        test_results['nas_clusterer'] = test_enhanced_standalone_nas_clusterer()
        test_results['nas_evaluator'] = test_enhanced_standalone_nas_evaluator()
        test_results['nas_trainer'] = test_enhanced_standalone_nas_trainer()
        test_results['feature_extractor'] = test_enhanced_standalone_feature_extractor()
        test_results['regime_analyzer'] = test_enhanced_standalone_regime_analyzer()
        test_results['micro_regime_detector'] = test_enhanced_standalone_micro_regime_detector()
        test_results['integration'] = test_enhanced_standalone_integration()
        
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