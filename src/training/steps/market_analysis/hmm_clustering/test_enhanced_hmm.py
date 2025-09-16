#!/usr/bin/env python3
"""
Test suite for Enhanced HMM Clustering with Common Utilities Integration.

This module provides comprehensive tests for the enhanced HMM clustering
implementation and its integration with common utilities.
"""

import unittest
import tempfile
import os
from typing import Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path

# Import the modules to test
from enhanced_hmm_clustering import (
    EnhancedHMMClustering,
    HMMClusteringConfig,
    HMMClusteringResults,
    create_enhanced_hmm_clustering
)
from integration_example import HMMClusteringIntegration

class TestHMMClusteringConfig(unittest.TestCase):
    """Test HMMClusteringConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = HMMClusteringConfig()
        
        self.assertEqual(config.n_components, 3)
        self.assertEqual(config.covariance_type, 'full')
        self.assertEqual(config.n_iter, 100)
        self.assertEqual(config.random_state, 42)
        self.assertTrue(config.use_gpu)
        self.assertTrue(config.enable_validation)
        self.assertTrue(config.enable_optimization)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = HMMClusteringConfig(
            n_components=5,
            covariance_type='tied',
            n_iter=200,
            random_state=123,
            use_gpu=False,
            memory_limit_gb=8.0
        )
        
        self.assertEqual(config.n_components, 5)
        self.assertEqual(config.covariance_type, 'tied')
        self.assertEqual(config.n_iter, 200)
        self.assertEqual(config.random_state, 123)
        self.assertFalse(config.use_gpu)
        self.assertEqual(config.memory_limit_gb, 8.0)

class TestEnhancedHMMClustering(unittest.TestCase):
    """Test EnhancedHMMClustering class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = HMMClusteringConfig(
            n_components=3,
            n_iter=50,  # Reduced for faster testing
            random_state=42
        )
        self.hmm_clustering = EnhancedHMMClustering(self.config)
        
        # Create sample data
        np.random.seed(42)
        n_samples = 500
        n_features = 4
        
        # Generate 3 distinct clusters
        cluster1 = np.random.multivariate_normal([0, 0, 0, 0], np.eye(4), n_samples // 3)
        cluster2 = np.random.multivariate_normal([3, 3, 3, 3], np.eye(4), n_samples // 3)
        cluster3 = np.random.multivariate_normal([-3, -3, -3, -3], np.eye(4), n_samples - 2 * (n_samples // 3))
        
        self.sample_data = np.vstack([cluster1, cluster2, cluster3])
        self.sample_df = pd.DataFrame(
            self.sample_data,
            columns=[f'feature_{i}' for i in range(n_features)]
        )
    
    def test_initialization(self):
        """Test HMM clustering initialization."""
        self.assertIsNotNone(self.hmm_clustering.config)
        self.assertFalse(self.hmm_clustering.is_trained)
        self.assertEqual(len(self.hmm_clustering.training_history), 0)
    
    def test_validate_input_data_numpy(self):
        """Test input data validation with numpy array."""
        validated_data = self.hmm_clustering._validate_input_data(self.sample_data)
        
        self.assertIsInstance(validated_data, np.ndarray)
        self.assertEqual(validated_data.shape, self.sample_data.shape)
        self.assertTrue(np.all(np.isfinite(validated_data)))
    
    def test_validate_input_data_dataframe(self):
        """Test input data validation with DataFrame."""
        validated_data = self.hmm_clustering._validate_input_data(self.sample_df)
        
        self.assertIsInstance(validated_data, np.ndarray)
        self.assertEqual(validated_data.shape, self.sample_df.shape)
        self.assertTrue(np.all(np.isfinite(validated_data)))
    
    def test_validate_input_data_invalid(self):
        """Test input data validation with invalid data."""
        # Test with 1D array
        with self.assertRaises(ValueError):
            self.hmm_clustering._validate_input_data(np.array([1, 2, 3]))
        
        # Test with insufficient samples
        with self.assertRaises(ValueError):
            self.hmm_clustering._validate_input_data(np.array([[1, 2], [3, 4]]))
    
    def test_calculate_cluster_balance(self):
        """Test cluster balance calculation."""
        # Perfectly balanced clusters
        balanced_labels = np.array([0, 0, 1, 1, 2, 2])
        balance = self.hmm_clustering._calculate_cluster_balance(balanced_labels)
        self.assertEqual(balance, 0.0)
        
        # Unbalanced clusters
        unbalanced_labels = np.array([0, 0, 0, 0, 1, 2])
        balance = self.hmm_clustering._calculate_cluster_balance(unbalanced_labels)
        self.assertGreater(balance, 0.0)
        
        # Single cluster
        single_labels = np.array([0, 0, 0, 0])
        balance = self.hmm_clustering._calculate_cluster_balance(single_labels)
        self.assertEqual(balance, 1.0)
    
    def test_fit_numpy_data(self):
        """Test fitting with numpy data."""
        results = self.hmm_clustering.fit(self.sample_data)
        
        self.assertIsInstance(results, HMMClusteringResults)
        self.assertTrue(self.hmm_clustering.is_trained)
        self.assertEqual(len(self.hmm_clustering.training_history), 1)
        
        # Check results properties
        self.assertIsNotNone(results.model)
        self.assertIsInstance(results.labels, np.ndarray)
        self.assertGreater(results.training_time, 0)
        self.assertIsInstance(results.silhouette_score, float)
    
    def test_fit_dataframe_data(self):
        """Test fitting with DataFrame data."""
        results = self.hmm_clustering.fit(self.sample_df)
        
        self.assertIsInstance(results, HMMClusteringResults)
        self.assertTrue(self.hmm_clustering.is_trained)
        
        # Check that labels are reasonable
        unique_labels = np.unique(results.labels)
        self.assertGreaterEqual(len(unique_labels), 1)
        self.assertLessEqual(len(unique_labels), self.config.n_components)
    
    def test_predict_before_training(self):
        """Test prediction before training raises error."""
        with self.assertRaises(ValueError):
            self.hmm_clustering.predict(self.sample_data)
    
    def test_predict_after_training(self):
        """Test prediction after training."""
        # Train the model first
        self.hmm_clustering.fit(self.sample_data)
        
        # Make predictions
        predictions = self.hmm_clustering.predict(self.sample_data)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.sample_data))
        
        # Check that predictions are valid cluster labels
        unique_predictions = np.unique(predictions)
        self.assertGreaterEqual(len(unique_predictions), 1)
        self.assertLessEqual(len(unique_predictions), self.config.n_components)
    
    def test_predict_proba_after_training(self):
        """Test probability prediction after training."""
        # Train the model first
        self.hmm_clustering.fit(self.sample_data)
        
        # Make probability predictions
        probabilities = self.hmm_clustering.predict_proba(self.sample_data)
        
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape[0], len(self.sample_data))
        self.assertEqual(probabilities.shape[1], self.config.n_components)
        
        # Check that probabilities sum to 1
        prob_sums = np.sum(probabilities, axis=1)
        np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-10)
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        # Train the model first
        self.hmm_clustering.fit(self.sample_data)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.hmm_clustering.save_model(tmp_path)
            self.assertTrue(success)
            
            # Create new instance and load model
            new_hmm = EnhancedHMMClustering(self.config)
            load_success = new_hmm.load_model(tmp_path)
            self.assertTrue(load_success)
            
            # Check that loaded model is trained
            self.assertTrue(new_hmm.is_trained)
            self.assertEqual(len(new_hmm.training_history), 1)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_get_performance_summary(self):
        """Test performance summary generation."""
        # Before training
        summary = self.hmm_clustering.get_performance_summary()
        self.assertEqual(summary, {})
        
        # After training
        self.hmm_clustering.fit(self.sample_data)
        summary = self.hmm_clustering.get_performance_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('training_metrics', summary)
        self.assertIn('clustering_metrics', summary)
        self.assertIn('hardware_info', summary)

class TestHMMClusteringIntegration(unittest.TestCase):
    """Test HMMClusteringIntegration class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.integration = HMMClusteringIntegration()
        
        # Create sample market data
        np.random.seed(42)
        n_samples = 1000
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
        
        # Generate realistic market data
        returns = np.random.normal(0, 0.02, n_samples)
        prices = 100 * np.exp(np.cumsum(returns))
        volume = np.random.lognormal(10, 0.5, n_samples)
        
        self.market_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
            'close': prices,
            'volume': volume
        })
    
    def test_initialization(self):
        """Test integration initialization."""
        self.assertIsNotNone(self.integration.hmm_clustering)
        self.assertIsNotNone(self.integration.json_serializer)
        self.assertIsNotNone(self.integration.pickle_serializer)
    
    def test_prepare_features(self):
        """Test feature preparation."""
        features = self.integration.prepare_features(self.market_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features), 0)
        
        # Check that features are numeric
        for col in features.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(features[col]))
        
        # Check for common features
        expected_features = ['returns', 'log_returns', 'volatility', 'volume_ratio']
        for feature in expected_features:
            if feature in features.columns:
                self.assertFalse(features[feature].isna().all())
    
    def test_calculate_rsi(self):
        """Test RSI calculation."""
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        rsi = self.integration._calculate_rsi(prices, window=5)
        
        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(prices))
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            self.assertTrue((valid_rsi >= 0).all())
            self.assertTrue((valid_rsi <= 100).all())
    
    def test_calculate_macd(self):
        """Test MACD calculation."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        macd = self.integration._calculate_macd(prices)
        
        self.assertIsInstance(macd, pd.Series)
        self.assertEqual(len(macd), len(prices))
    
    def test_calculate_bollinger_position(self):
        """Test Bollinger Bands position calculation."""
        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98
        })
        position = self.integration._calculate_bollinger_position(data)
        
        self.assertIsInstance(position, pd.Series)
        self.assertEqual(len(position), len(data))
        # Position should be between 0 and 1
        valid_position = position.dropna()
        if len(valid_position) > 0:
            self.assertTrue((valid_position >= 0).all())
            self.assertTrue((valid_position <= 1).all())
    
    def test_run_comprehensive_analysis(self):
        """Test comprehensive analysis."""
        results = self.integration.run_comprehensive_analysis(
            self.market_data, 
            optimize_hyperparams=False  # Disable for faster testing
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('hmm_results', results)
        self.assertIn('performance_summary', results)
        self.assertIn('feature_quality', results)
        self.assertIn('hardware_utilization', results)
    
    def test_generate_report(self):
        """Test report generation."""
        # Create mock results
        mock_results = {
            'hmm_results': {
                'training_time': 1.5,
                'log_likelihood': -1000.0,
                'aic': 2000.0,
                'bic': 2100.0,
                'silhouette_score': 0.5,
                'calinski_harabasz_score': 100.0,
                'davies_bouldin_score': 1.5
            },
            'hardware_utilization': {
                'gpu_used': True,
                'memory_optimized': True,
                'cpu_optimized': True
            },
            'hyperparameter_optimization': {
                'n_components': 3,
                'covariance_type': 'full'
            },
            'analysis_timestamp': 1234567890
        }
        
        report = self.integration.generate_report(mock_results)
        
        self.assertIsInstance(report, str)
        self.assertIn('HMM Clustering Analysis Report', report)
        self.assertIn('Model Performance', report)
        self.assertIn('Clustering Quality', report)
        self.assertIn('Hardware Utilization', report)
    
    def test_save_and_load_analysis_results(self):
        """Test saving and loading analysis results."""
        # Create mock results
        mock_results = {
            'hmm_results': {
                'labels': [0, 1, 2, 0, 1],
                'probabilities': [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
                'training_time': 1.5,
                'silhouette_score': 0.5
            },
            'performance_summary': {
                'training_metrics': {'log_likelihood': -1000.0},
                'clustering_metrics': {'silhouette_score': 0.5}
            },
            'analysis_timestamp': 1234567890
        }
        
        # Test JSON serialization
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save results
            success = self.integration.save_analysis_results(mock_results, tmp_path)
            self.assertTrue(success)
            
            # Load results
            loaded_results = self.integration.load_analysis_results(tmp_path)
            self.assertIsNotNone(loaded_results)
            self.assertIn('hmm_results', loaded_results)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

class TestFactoryFunction(unittest.TestCase):
    """Test factory function."""
    
    def test_create_enhanced_hmm_clustering_default(self):
        """Test factory function with default config."""
        hmm_clustering = create_enhanced_hmm_clustering()
        
        self.assertIsInstance(hmm_clustering, EnhancedHMMClustering)
        self.assertIsInstance(hmm_clustering.config, HMMClusteringConfig)
    
    def test_create_enhanced_hmm_clustering_custom(self):
        """Test factory function with custom config."""
        config = HMMClusteringConfig(n_components=5, use_gpu=False)
        hmm_clustering = create_enhanced_hmm_clustering(config)
        
        self.assertIsInstance(hmm_clustering, EnhancedHMMClustering)
        self.assertEqual(hmm_clustering.config.n_components, 5)
        self.assertFalse(hmm_clustering.config.use_gpu)

def run_performance_test():
    """Run a performance test to demonstrate the utilities."""
    print("🚀 Running Enhanced HMM Clustering Performance Test...")
    
    # Create large dataset
    np.random.seed(42)
    n_samples = 5000
    n_features = 10
    
    # Generate data with 4 clusters
    cluster1 = np.random.multivariate_normal([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.eye(10), n_samples // 4)
    cluster2 = np.random.multivariate_normal([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], np.eye(10), n_samples // 4)
    cluster3 = np.random.multivariate_normal([-2, -2, -2, -2, -2, -2, -2, -2, -2, -2], np.eye(10), n_samples // 4)
    cluster4 = np.random.multivariate_normal([4, 4, 4, 4, 4, 4, 4, 4, 4, 4], np.eye(10), n_samples - 3 * (n_samples // 4))
    
    data = np.vstack([cluster1, cluster2, cluster3, cluster4])
    
    # Test with different configurations
    configs = [
        HMMClusteringConfig(n_components=2, n_iter=50, use_gpu=False),
        HMMClusteringConfig(n_components=4, n_iter=100, use_gpu=True),
        HMMClusteringConfig(n_components=6, n_iter=200, use_gpu=True, enable_optimization=True)
    ]
    
    for i, config in enumerate(configs):
        print(f"\n📊 Testing Configuration {i+1}: {config.n_components} components, {config.n_iter} iterations")
        
        hmm_clustering = EnhancedHMMClustering(config)
        
        import time
        start_time = time.time()
        
        try:
            results = hmm_clustering.fit(data)
            training_time = time.time() - start_time
            
            print(f"✅ Training completed in {training_time:.2f} seconds")
            print(f"   Silhouette Score: {results.silhouette_score:.3f}")
            print(f"   AIC: {results.aic:.2f}")
            print(f"   BIC: {results.bic:.2f}")
            
            # Test prediction performance
            pred_start = time.time()
            predictions = hmm_clustering.predict(data)
            pred_time = time.time() - pred_start
            
            print(f"   Prediction time: {pred_time:.4f} seconds")
            print(f"   Unique clusters found: {len(np.unique(predictions))}")
            
        except Exception as e:
            print(f"❌ Configuration {i+1} failed: {e}")
    
    print("\n🎉 Performance test completed!")

if __name__ == '__main__':
    # Run unit tests
    print("🧪 Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance test
    print("\n" + "="*50)
    run_performance_test()