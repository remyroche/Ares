"""
Tests for Enhanced Profit Labeling System

This module contains comprehensive tests for the enhanced profit labeling system
to ensure all components work correctly and integrate properly.

Author: AI Assistant
Date: 2025-01-10
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.training.steps.pre_training.profit_labeling.enhanced_profit_labeling_system import (
    EnhancedProfitLabelingSystem, ProfitLabelingConfig
)


class TestProfitLabelingConfig(unittest.TestCase):
    """Test the ProfitLabelingConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProfitLabelingConfig()
        
        self.assertEqual(config.symbols, ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(config.timeframes, ["1h", "4h", "1d"])
        self.assertEqual(config.max_features, 1000)
        self.assertTrue(config.enable_bayesian_optimization)
        self.assertFalse(config.enable_gpu)
        self.assertTrue(config.enable_parallel)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProfitLabelingConfig(
            symbols=["BTCUSDT"],
            timeframes=["1h"],
            max_features=100,
            enable_bayesian_optimization=False
        )
        
        self.assertEqual(config.symbols, ["BTCUSDT"])
        self.assertEqual(config.timeframes, ["1h"])
        self.assertEqual(config.max_features, 100)
        self.assertFalse(config.enable_bayesian_optimization)


class TestEnhancedProfitLabelingSystem(unittest.TestCase):
    """Test the EnhancedProfitLabelingSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ProfitLabelingConfig(
            symbols=["BTCUSDT"],
            timeframes=["1h"],
            max_features=50,
            enable_bayesian_optimization=False
        )
        
        # Mock the system to avoid dependency issues
        with patch.multiple(
            'src.training.steps.pre_training.profit_labeling.enhanced_profit_labeling_system',
            DATA_UTILS_AVAILABLE=False,
            FEATURE_UTILS_AVAILABLE=False,
            ML_UTILS_AVAILABLE=False,
            HARDWARE_UTILS_AVAILABLE=False,
            TPRINT_AVAILABLE=False
        ):
            self.system = EnhancedProfitLabelingSystem(self.config)
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.system.config)
        self.assertIsNotNone(self.system.logger)
        self.assertIsNone(self.system.klines_manager)  # Mocked as unavailable
        self.assertIsNone(self.system.rolling_optimizer)  # Mocked as unavailable
    
    def test_generate_mock_data(self):
        """Test mock data generation."""
        mock_data = self.system._generate_mock_data("BTCUSDT", "1h")
        
        self.assertIsInstance(mock_data, pd.DataFrame)
        self.assertGreater(len(mock_data), 0)
        self.assertIn('open', mock_data.columns)
        self.assertIn('high', mock_data.columns)
        self.assertIn('low', mock_data.columns)
        self.assertIn('close', mock_data.columns)
        self.assertIn('volume', mock_data.columns)
        self.assertEqual(mock_data['symbol'].iloc[0], "BTCUSDT")
        self.assertEqual(mock_data['timeframe'].iloc[0], "1h")
    
    def test_generate_basic_features(self):
        """Test basic feature generation."""
        # Create mock data
        mock_data = self.system._generate_mock_data("BTCUSDT", "1h")
        
        # Generate features
        features = self.system._generate_basic_features(mock_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertIn('returns', features.columns)
        self.assertIn('log_returns', features.columns)
        self.assertIn('volatility', features.columns)
        self.assertIn('sma_20', features.columns)
        self.assertIn('rsi', features.columns)
    
    def test_calculate_rsi(self):
        """Test RSI calculation."""
        # Create test data
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        
        rsi = self.system._calculate_rsi(prices, window=5)
        
        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(prices))
        self.assertTrue(rsi.isna().sum() > 0)  # Should have NaN values initially
    
    def test_calculate_balance_score(self):
        """Test balance score calculation."""
        # Create test data with balanced classes
        balanced_data = pd.DataFrame({
            'label1': [0, 1, 0, 1, 0, 1],
            'label2': [0, 0, 1, 1, 0, 1]
        })
        
        balance_score = self.system._calculate_balance_score(balanced_data)
        
        self.assertIsInstance(balance_score, float)
        self.assertGreaterEqual(balance_score, 0.0)
        self.assertLessEqual(balance_score, 1.0)
    
    def test_calculate_stability_score(self):
        """Test stability score calculation."""
        # Create test data
        stable_data = pd.DataFrame({
            'label1': [1, 1, 1, 1, 1],
            'label2': [0.5, 0.5, 0.5, 0.5, 0.5]
        })
        
        stability_score = self.system._calculate_stability_score(stable_data)
        
        self.assertIsInstance(stability_score, float)
        self.assertGreaterEqual(stability_score, 0.0)
        self.assertLessEqual(stability_score, 1.0)
    
    def test_calculate_evaluation_metrics(self):
        """Test evaluation metrics calculation."""
        # Create test data
        features = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        labels = pd.DataFrame({
            'label1': np.random.randint(0, 2, 100),
            'label2': np.random.randn(100)
        })
        
        metrics = self.system._calculate_evaluation_metrics(features, labels)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('label_count', metrics)
        self.assertIn('feature_count', metrics)
        self.assertIn('label_mean', metrics)
        self.assertIn('label_std', metrics)
        self.assertIn('label_balance', metrics)
        self.assertIn('label_stability', metrics)
    
    @patch('src.training.steps.pre_training.profit_labeling.enhanced_profit_labeling_system.tprint_info')
    def test_load_data_mock(self, mock_tprint):
        """Test data loading with mock data."""
        data = self.system.load_data()
        
        self.assertIsInstance(data, dict)
        self.assertGreater(len(data), 0)
        
        # Check that mock data was generated
        for key, df in data.items():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
    
    @patch('src.training.steps.pre_training.profit_labeling.enhanced_profit_labeling_system.tprint_info')
    def test_generate_features_mock(self, mock_tprint):
        """Test feature generation with mock data."""
        # Create mock data
        data = {
            'BTCUSDT_1h': self.system._generate_mock_data("BTCUSDT", "1h")
        }
        
        features = self.system.generate_features(data)
        
        self.assertIsInstance(features, dict)
        self.assertIn('BTCUSDT_1h', features)
        self.assertIsInstance(features['BTCUSDT_1h'], pd.DataFrame)
    
    @patch('src.training.steps.pre_training.profit_labeling.enhanced_profit_labeling_system.tprint_info')
    def test_generate_labels_mock(self, mock_tprint):
        """Test label generation with mock data."""
        # Create mock data
        data = {
            'BTCUSDT_1h': self.system._generate_mock_data("BTCUSDT", "1h")
        }
        
        # Mock the profit labeler
        with patch.object(self.system, 'profit_labeler') as mock_labeler:
            mock_labeler.generate_labels.return_value = pd.DataFrame({
                'label1': np.random.randn(100),
                'label2': np.random.randint(0, 2, 100)
            })
            
            labels = self.system.generate_labels(data)
            
            self.assertIsInstance(labels, dict)
            self.assertIn('BTCUSDT_1h', labels)
            self.assertIsInstance(labels['BTCUSDT_1h'], pd.DataFrame)
    
    @patch('src.training.steps.pre_training.profit_labeling.enhanced_profit_labeling_system.tprint_info')
    def test_select_features_mock(self, mock_tprint):
        """Test feature selection with mock data."""
        # Create mock data
        features = {
            'BTCUSDT_1h': pd.DataFrame({
                'feature1': np.random.randn(100),
                'feature2': np.random.randn(100),
                'feature3': np.random.randn(100)
            })
        }
        labels = {
            'BTCUSDT_1h': pd.DataFrame({
                'label1': np.random.randn(100)
            })
        }
        
        selected_features = self.system.select_features(features, labels)
        
        self.assertIsInstance(selected_features, dict)
        self.assertIn('BTCUSDT_1h', selected_features)
        self.assertIsInstance(selected_features['BTCUSDT_1h'], list)
    
    @patch('src.training.steps.pre_training.profit_labeling.enhanced_profit_labeling_system.tprint_info')
    def test_evaluate_labels_mock(self, mock_tprint):
        """Test label evaluation with mock data."""
        # Create mock data
        features = {
            'BTCUSDT_1h': pd.DataFrame({
                'feature1': np.random.randn(100),
                'feature2': np.random.randn(100)
            })
        }
        labels = {
            'BTCUSDT_1h': pd.DataFrame({
                'label1': np.random.randn(100),
                'label2': np.random.randint(0, 2, 100)
            })
        }
        
        evaluation = self.system.evaluate_labels(features, labels)
        
        self.assertIsInstance(evaluation, dict)
        self.assertIn('BTCUSDT_1h', evaluation)
        self.assertIsInstance(evaluation['BTCUSDT_1h'], dict)
    
    def test_save_results_no_serializer(self):
        """Test saving results when serializer is not available."""
        results = {'test': 'data'}
        
        # Should not raise an exception
        self.system.save_results(results, 'test.json')
    
    @patch('src.training.steps.pre_training.profit_labeling.enhanced_profit_labeling_system.tprint_info')
    def test_run_full_pipeline_mock(self, mock_tprint):
        """Test full pipeline execution with mocked components."""
        # Mock all the pipeline steps
        with patch.object(self.system, 'load_data') as mock_load, \
             patch.object(self.system, 'generate_features') as mock_features, \
             patch.object(self.system, 'generate_labels') as mock_labels, \
             patch.object(self.system, 'select_features') as mock_select, \
             patch.object(self.system, 'evaluate_labels') as mock_evaluate, \
             patch.object(self.system, 'save_results') as mock_save:
            
            # Set up mock returns
            mock_load.return_value = {'BTCUSDT_1h': pd.DataFrame()}
            mock_features.return_value = {'BTCUSDT_1h': pd.DataFrame()}
            mock_labels.return_value = {'BTCUSDT_1h': pd.DataFrame()}
            mock_select.return_value = {'BTCUSDT_1h': ['feature1', 'feature2']}
            mock_evaluate.return_value = {'BTCUSDT_1h': {'score': 0.8}}
            
            # Run pipeline
            results = self.system.run_full_pipeline()
            
            # Check results structure
            self.assertIsInstance(results, dict)
            self.assertIn('config', results)
            self.assertIn('timestamp', results)
            self.assertIn('data', results)
            self.assertIn('features', results)
            self.assertIn('labels', results)
            self.assertIn('selected_features', results)
            self.assertIn('evaluation', results)
            
            # Verify all methods were called
            mock_load.assert_called_once()
            mock_features.assert_called_once()
            mock_labels.assert_called_once()
            mock_select.assert_called_once()
            mock_evaluate.assert_called_once()
            mock_save.assert_called_once()


class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = ProfitLabelingConfig(
            symbols=["BTCUSDT"],
            timeframes=["1h"],
            max_features=100
        )
        self.assertIsNotNone(config)
        
        # Test invalid configuration handling
        with self.assertRaises(TypeError):
            ProfitLabelingConfig(invalid_param="test")
    
    def test_mock_data_consistency(self):
        """Test that mock data is consistent across calls."""
        config = ProfitLabelingConfig(symbols=["BTCUSDT"], timeframes=["1h"])
        
        with patch.multiple(
            'src.training.steps.pre_training.profit_labeling.enhanced_profit_labeling_system',
            DATA_UTILS_AVAILABLE=False,
            FEATURE_UTILS_AVAILABLE=False,
            ML_UTILS_AVAILABLE=False,
            HARDWARE_UTILS_AVAILABLE=False,
            TPRINT_AVAILABLE=False
        ):
            system = EnhancedProfitLabelingSystem(config)
            
            # Generate mock data multiple times
            data1 = system._generate_mock_data("BTCUSDT", "1h")
            data2 = system._generate_mock_data("BTCUSDT", "1h")
            
            # Should have same structure
            self.assertEqual(list(data1.columns), list(data2.columns))
            self.assertEqual(data1.shape[0], data2.shape[0])


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestProfitLabelingConfig))
    test_suite.addTest(unittest.makeSuite(TestEnhancedProfitLabelingSystem))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        exit(1)