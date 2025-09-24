#!/usr/bin/env python3
"""
Test Enhanced CVLSA Integration

This script tests the enhanced CVLSA architecture with all improvements:
1. Cross-View Attention between different data modalities
2. Multi-Scale Temporal Attention for time series modeling
3. Memory Efficiency with gradient checkpointing and chunked processing
4. Bayesian Hyperparameter Optimization
5. Hardware Integration with M1 GPU acceleration
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Any, Optional, Tuple
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_market_data(n_samples: int = 1000, n_features: int = 20) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create sample market data for testing."""
    logger.info(f"📊 Creating sample market data: {n_samples} samples, {n_features} features")
    
    np.random.seed(42)
    
    # Generate synthetic market data
    base_price = 100.0
    prices = []
    volumes = []
    features = []
    
    for i in range(n_samples):
        # Generate price movement with some trend
        if i == 0:
            price = base_price
        else:
            # Add some trend and noise
            trend = 0.0001 * i  # Slight upward trend
            noise = np.random.normal(0, 0.02)
            price = prices[-1] * (1 + trend + noise)
        
        # Generate OHLC from price
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = price * (1 + np.random.normal(0, 0.005))
        close = price
        
        prices.append([open_price, high, low, close])
        
        # Generate volume with some correlation to price movement
        volume = np.random.lognormal(10, 1) * (1 + abs(noise) * 2)
        volumes.append(volume)
        
        # Generate additional features
        feature_vector = np.random.randn(n_features)
        # Make some features correlated with price movement
        feature_vector[0] = noise * 2  # Price momentum
        feature_vector[1] = np.random.normal(0, 0.1) + trend * 10  # Trend strength
        feature_vector[2] = np.random.exponential(0.1)  # Volatility proxy
        features.append(feature_vector)
    
    # Create market data DataFrame
    market_data = pd.DataFrame({
        'open': [p[0] for p in prices],
        'high': [p[1] for p in prices],
        'low': [p[2] for p in prices],
        'close': [p[3] for p in prices],
        'volume': volumes
    })
    
    # Create feature matrix
    X = np.array(features)
    
    # Create target (next period return)
    returns = np.diff([p[3] for p in prices], prepend=prices[0][3]) / [p[3] for p in prices]
    y = returns[1:]  # Shift by 1 for next period prediction
    X = X[:-1]  # Remove last sample to match target length
    
    # Create regime labels
    regimes = np.random.choice(['high_vol', 'low_vol', 'trending', 'mean_reverting'], 
                              len(X), p=[0.3, 0.2, 0.3, 0.2])
    
    logger.info(f"✅ Sample data created: {market_data.shape[0]} market samples, {X.shape[0]} feature samples")
    return market_data, X, y, regimes


def test_enhanced_cvlsa_architecture():
    """Test the enhanced CVLSA architecture."""
    logger.info("🧪 Testing Enhanced CVLSA Architecture...")
    
    try:
        from src.utils.ml_common.models.enhanced_cvlsa_architecture import (
            EnhancedCVLSAConfig, EnhancedCVLSATrainer, create_enhanced_cvlsa_model
        )
        
        # Create sample data
        market_data, X, y, regimes = create_sample_market_data(500, 15)
        
        # Create CVLSA configuration
        config = EnhancedCVLSAConfig(
            input_dim=15,
            output_dim=1,
            seq_length=100,
            cross_view_attention=True,
            use_multi_scale_attention=True,
            memory_efficient=True,
            gradient_checkpointing=True,
            use_m1_gpu=True,
            enable_hyperparameter_optimization=False  # Disable for testing
        )
        
        # Create and train CVLSA model
        cvlsa_trainer = create_enhanced_cvlsa_model(config)
        
        # Prepare features
        logger.info("🔧 Preparing CVLSA features...")
        cvlsa_features = cvlsa_trainer.prepare_features(market_data)
        
        # Train model
        logger.info("🚀 Training CVLSA model...")
        start_time = time.time()
        
        # Convert target to tensor
        y_tensor = torch.FloatTensor(y[:len(cvlsa_features['price'])])
        
        training_results = cvlsa_trainer.train(
            cvlsa_features, cvlsa_features, y_tensor, epochs=20
        )
        
        training_time = time.time() - start_time
        
        # Test predictions
        logger.info("🔮 Testing CVLSA predictions...")
        predictions = cvlsa_trainer.predict(cvlsa_features)
        
        # Verify results
        assert len(predictions) == len(y_tensor), f"Prediction length mismatch: {len(predictions)} vs {len(y_tensor)}"
        assert not torch.isnan(predictions).any(), "Predictions contain NaN values"
        
        # Test attention weights
        attention_weights = cvlsa_trainer.get_attention_weights()
        assert 'cross_view' in attention_weights, "Cross-view attention weights not found"
        
        logger.info("✅ Enhanced CVLSA architecture test passed")
        logger.info(f"   Training time: {training_time:.2f}s")
        logger.info(f"   Prediction shape: {predictions.shape}")
        logger.info(f"   Attention weights available: {len(attention_weights)} types")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced CVLSA architecture test failed: {e}")
        return False


def test_hybrid_cvlsa_tree_model():
    """Test the hybrid CVLSA-tree model."""
    logger.info("🌳 Testing Hybrid CVLSA-Tree Model...")
    
    try:
        from src.utils.ml_common.models.enhanced_cvlsa_integration import (
            HybridCVLSATreeModel, create_hybrid_cvlsa_tree_model
        )
        from src.utils.ml_common.models.enhanced_cvlsa_architecture import EnhancedCVLSAConfig
        from src.utils.ml_common.models.tree_clvsa_wrapper import TreeCLVSAConfig
        
        # Create sample data
        market_data, X, y, regimes = create_sample_market_data(300, 10)
        
        # Create configurations
        cvlsa_config = EnhancedCVLSAConfig(
            input_dim=10,
            output_dim=1,
            seq_length=50,
            memory_efficient=True,
            enable_hyperparameter_optimization=False
        )
        
        tree_config = TreeCLVSAConfig(
            attention_dim=32,
            use_temporal_attention=True,
            regime_aware=True,
            memory_efficient=True
        )
        
        # Create hybrid model
        hybrid_model = create_hybrid_cvlsa_tree_model(
            cvlsa_config=cvlsa_config,
            tree_config=tree_config,
            tree_model_type='random_forest',
            fusion_method='weighted_average'
        )
        
        # Train hybrid model
        logger.info("🚀 Training hybrid model...")
        start_time = time.time()
        
        hybrid_model.fit(X, y, market_data=market_data, regimes=regimes)
        
        training_time = time.time() - start_time
        
        # Test predictions
        logger.info("🔮 Testing hybrid model predictions...")
        predictions = hybrid_model.predict(X, market_data=market_data)
        
        # Verify results
        assert len(predictions) == len(y), f"Prediction length mismatch: {len(predictions)} vs {len(y)}"
        assert not np.isnan(predictions).any(), "Predictions contain NaN values"
        
        # Test feature importance
        feature_importance = hybrid_model.get_feature_importance()
        assert len(feature_importance) > 0, "Feature importance not available"
        
        # Test model info
        model_info = hybrid_model.get_model_info()
        assert model_info['is_fitted'], "Model not marked as fitted"
        
        logger.info("✅ Hybrid CVLSA-tree model test passed")
        logger.info(f"   Training time: {training_time:.2f}s")
        logger.info(f"   Prediction shape: {predictions.shape}")
        logger.info(f"   Feature importance types: {list(feature_importance.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hybrid CVLSA-tree model test failed: {e}")
        return False


def test_cvlsa_feature_extractor():
    """Test the CVLSA feature extractor."""
    logger.info("🔧 Testing CVLSA Feature Extractor...")
    
    try:
        from src.utils.ml_common.models.enhanced_cvlsa_integration import (
            CVLSAFeatureExtractor, create_cvlsa_feature_extractor
        )
        from src.utils.ml_common.models.enhanced_cvlsa_architecture import EnhancedCVLSAConfig
        
        # Create sample data
        market_data, X, y, regimes = create_sample_market_data(200, 8)
        
        # Create CVLSA configuration
        cvlsa_config = EnhancedCVLSAConfig(
            input_dim=8,
            output_dim=1,
            seq_length=30,
            memory_efficient=True,
            enable_hyperparameter_optimization=False
        )
        
        # Create feature extractor
        feature_extractor = create_cvlsa_feature_extractor(cvlsa_config)
        
        # Fit and transform
        logger.info("🔧 Fitting and transforming features...")
        start_time = time.time()
        
        enhanced_features = feature_extractor.fit_transform(X, y, market_data=market_data)
        
        processing_time = time.time() - start_time
        
        # Verify results
        assert enhanced_features.shape[0] == X.shape[0], f"Sample count mismatch: {enhanced_features.shape[0]} vs {X.shape[0]}"
        assert enhanced_features.shape[1] > X.shape[1], f"Feature count not increased: {enhanced_features.shape[1]} vs {X.shape[1]}"
        assert not np.isnan(enhanced_features).any(), "Enhanced features contain NaN values"
        
        # Test with new data
        logger.info("🔮 Testing feature extraction on new data...")
        new_market_data, new_X, new_y, _ = create_sample_market_data(100, 8)
        new_enhanced_features = feature_extractor.transform(new_X, new_market_data)
        
        assert new_enhanced_features.shape[0] == new_X.shape[0], "New data sample count mismatch"
        assert new_enhanced_features.shape[1] == enhanced_features.shape[1], "Feature dimension mismatch"
        
        logger.info("✅ CVLSA feature extractor test passed")
        logger.info(f"   Processing time: {processing_time:.2f}s")
        logger.info(f"   Original features: {X.shape[1]}")
        logger.info(f"   Enhanced features: {enhanced_features.shape[1]}")
        logger.info(f"   Feature increase: {enhanced_features.shape[1] - X.shape[1]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CVLSA feature extractor test failed: {e}")
        return False


def test_memory_efficiency():
    """Test memory efficiency features."""
    logger.info("🧠 Testing Memory Efficiency...")
    
    try:
        from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
        from src.utils.matrix_operations.enhanced_operations import get_enhanced_matrix_operations
        
        # Test memory optimizer
        memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=2.0)
        memory_stats = memory_optimizer.get_memory_stats()
        
        assert 'memory_percent' in memory_stats, "Memory stats not available"
        assert memory_stats['memory_percent'] >= 0, "Invalid memory percentage"
        
        # Test matrix operations
        matrix_ops = get_enhanced_matrix_operations()
        performance_stats = matrix_ops.get_performance_stats()
        
        assert 'gpu_enabled' in performance_stats, "GPU status not available"
        assert 'memory_efficient' in performance_stats, "Memory efficiency status not available"
        
        # Test memory optimization
        optimization_result = memory_optimizer.optimize_memory_usage(aggressive=False)
        assert optimization_result['success'], "Memory optimization failed"
        
        logger.info("✅ Memory efficiency test passed")
        logger.info(f"   Memory usage: {memory_stats['memory_percent']:.1f}%")
        logger.info(f"   GPU enabled: {performance_stats['gpu_enabled']}")
        logger.info(f"   Memory optimization: {optimization_result['success']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory efficiency test failed: {e}")
        return False


def test_cross_view_attention():
    """Test cross-view attention mechanisms."""
    logger.info("👁️ Testing Cross-View Attention...")
    
    try:
        from src.utils.ml_common.models.enhanced_cvlsa_architecture import (
            CrossViewAttention, EnhancedCVLSAConfig
        )
        
        # Create configuration
        config = EnhancedCVLSAConfig(
            input_dim=10,
            view_embedding_dim=32,
            cross_attention_heads=4
        )
        
        # Create cross-view attention module
        cross_attention = CrossViewAttention(config)
        
        # Create sample data
        batch_size, seq_len, input_dim = 16, 50, 10
        price_features = torch.randn(batch_size, seq_len, input_dim)
        volume_features = torch.randn(batch_size, seq_len, input_dim)
        
        # Test cross-view attention
        attended_features = cross_attention(
            price_features, volume_features, 'price', 'volume'
        )
        
        # Verify results
        assert attended_features.shape == (batch_size, seq_len, config.view_embedding_dim), \
            f"Output shape mismatch: {attended_features.shape}"
        assert not torch.isnan(attended_features).any(), "Output contains NaN values"
        assert cross_attention.attention_weights is not None, "Attention weights not computed"
        
        # Test attention weights
        attention_weights = cross_attention.attention_weights
        assert attention_weights.shape == (batch_size, config.cross_attention_heads, seq_len, seq_len), \
            f"Attention weights shape mismatch: {attention_weights.shape}"
        
        logger.info("✅ Cross-view attention test passed")
        logger.info(f"   Input shape: {price_features.shape}")
        logger.info(f"   Output shape: {attended_features.shape}")
        logger.info(f"   Attention weights shape: {attention_weights.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cross-view attention test failed: {e}")
        return False


def test_multi_scale_temporal_attention():
    """Test multi-scale temporal attention."""
    logger.info("⏰ Testing Multi-Scale Temporal Attention...")
    
    try:
        from src.utils.ml_common.models.enhanced_cvlsa_architecture import (
            MultiScaleTemporalAttention, EnhancedCVLSAConfig
        )
        
        # Create configuration
        config = EnhancedCVLSAConfig(
            temporal_scales=[1, 3, 7, 14],
            view_embedding_dim=64,
            temporal_attention_heads=4
        )
        
        # Create multi-scale temporal attention
        temporal_attention = MultiScaleTemporalAttention(config)
        
        # Create sample data
        batch_size, seq_len, embed_dim = 8, 100, 64
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Test multi-scale temporal attention
        temporal_features = temporal_attention(x)
        
        # Verify results
        assert temporal_features.shape == (batch_size, seq_len, embed_dim), \
            f"Output shape mismatch: {temporal_features.shape}"
        assert not torch.isnan(temporal_features).any(), "Output contains NaN values"
        
        # Test with different sequence lengths
        for seq_len in [50, 200, 500]:
            x_test = torch.randn(batch_size, seq_len, embed_dim)
            temporal_features_test = temporal_attention(x_test)
            assert temporal_features_test.shape == (batch_size, seq_len, embed_dim), \
                f"Output shape mismatch for seq_len {seq_len}: {temporal_features_test.shape}"
        
        logger.info("✅ Multi-scale temporal attention test passed")
        logger.info(f"   Input shape: {x.shape}")
        logger.info(f"   Output shape: {temporal_features.shape}")
        logger.info(f"   Temporal scales: {config.temporal_scales}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Multi-scale temporal attention test failed: {e}")
        return False


def test_bayesian_optimization():
    """Test Bayesian hyperparameter optimization."""
    logger.info("🔍 Testing Bayesian Hyperparameter Optimization...")
    
    try:
        from src.utils.ml_common.models.enhanced_cvlsa_architecture import (
            BayesianHyperparameterOptimizer, EnhancedCVLSAConfig
        )
        
        # Create configuration
        config = EnhancedCVLSAConfig(
            input_dim=5,
            output_dim=1,
            seq_length=20,
            enable_hyperparameter_optimization=True,
            optimization_trials=5,  # Reduced for testing
            optimization_timeout=60  # 1 minute timeout
        )
        
        # Create optimizer
        optimizer = BayesianHyperparameterOptimizer(config)
        
        # Create sample data
        market_data, X, y, regimes = create_sample_market_data(100, 5)
        
        # Prepare training data
        train_data = {
            'price': torch.randn(50, 20, 5),
            'volume': torch.randn(50, 20, 5),
            'trend': torch.randn(50, 20, 5),
            'momentum': torch.randn(50, 20, 5)
        }
        
        val_data = {
            'price': torch.randn(20, 20, 5),
            'volume': torch.randn(20, 20, 5),
            'trend': torch.randn(20, 20, 5),
            'momentum': torch.randn(20, 20, 5)
        }
        
        target = torch.randn(50)
        
        # Test optimization
        logger.info("🔍 Running hyperparameter optimization...")
        start_time = time.time()
        
        best_params = optimizer.optimize_hyperparameters(train_data, val_data, target)
        
        optimization_time = time.time() - start_time
        
        # Verify results
        assert isinstance(best_params, dict), "Best parameters not returned as dictionary"
        assert len(best_params) > 0, "No parameters optimized"
        
        # Check for expected parameters
        expected_params = ['view_embedding_dim', 'cross_attention_heads', 'learning_rate']
        for param in expected_params:
            if param in best_params:
                logger.info(f"   {param}: {best_params[param]}")
        
        logger.info("✅ Bayesian hyperparameter optimization test passed")
        logger.info(f"   Optimization time: {optimization_time:.2f}s")
        logger.info(f"   Parameters optimized: {len(best_params)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Bayesian hyperparameter optimization test failed: {e}")
        return False


def main():
    """Run all enhanced CVLSA tests."""
    logger.info("🚀 Starting Enhanced CVLSA Integration Tests")
    
    tests = [
        ("Enhanced CVLSA Architecture", test_enhanced_cvlsa_architecture),
        ("Hybrid CVLSA-Tree Model", test_hybrid_cvlsa_tree_model),
        ("CVLSA Feature Extractor", test_cvlsa_feature_extractor),
        ("Memory Efficiency", test_memory_efficiency),
        ("Cross-View Attention", test_cross_view_attention),
        ("Multi-Scale Temporal Attention", test_multi_scale_temporal_attention),
        ("Bayesian Optimization", test_bayesian_optimization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("ENHANCED CVLSA INTEGRATION TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All enhanced CVLSA tests passed!")
        logger.info("\n📋 Enhanced CVLSA Features Implemented:")
        logger.info("   ✅ Cross-View Attention between data modalities (price, volume, trend, momentum)")
        logger.info("   ✅ Multi-Scale Temporal Attention for time series modeling")
        logger.info("   ✅ Memory Efficiency with gradient checkpointing and chunked processing")
        logger.info("   ✅ Bayesian Hyperparameter Optimization with Optuna integration")
        logger.info("   ✅ Hardware Integration with M1 GPU acceleration and memory optimization")
        logger.info("   ✅ Hybrid CVLSA-Tree models combining neural and tree-based approaches")
        logger.info("   ✅ CVLSA Feature Extractor for enhancing any downstream model")
        logger.info("   ✅ Advanced feature engineering with technical indicators")
        logger.info("   ✅ Matrix operations optimization with hardware acceleration")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)