"""
Tests for Hybrid Orchestrator.
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, Any

from ..integration.hybrid_orchestrator import HybridOrchestrator
from ..config.hybrid_config import HybridRegimeConfig, ClusteringMethod, IntegrationStrategy


class TestHybridOrchestrator(unittest.TestCase):
    """Test cases for Hybrid Orchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = HybridRegimeConfig(
            n_regimes=6,
            economic_modeling_enabled=True,
            financial_modeling_enabled=True,
            clustering_method=ClusteringMethod.KMEANS,
            integration_strategy=IntegrationStrategy.WEIGHTED_AVERAGE
        )
        
        self.orchestrator = HybridOrchestrator(self.config)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        # Create sample TAS inputs
        self.tas_inputs = {
            'regime_predictions': np.random.randint(0, 6, 100),
            'regime_probabilities': np.random.dirichlet([1] * 6, 100),
            'regime_stability_scores': np.random.uniform(0.6, 0.9, 100),
            'economic_significance_scores': np.random.uniform(0.5, 0.8, 100),
            'trading_viability_scores': np.random.uniform(0.4, 0.7, 100)
        }
        
        # Create sample NAS inputs
        self.nas_inputs = {
            'regime_predictions': np.random.randint(0, 6, 100),
            'regime_probabilities': np.random.dirichlet([1] * 6, 100),
            'regime_stability_scores': np.random.uniform(0.7, 0.95, 100),
            'economic_significance_scores': np.random.uniform(0.6, 0.9, 100),
            'trading_viability_scores': np.random.uniform(0.5, 0.8, 100)
        }
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        self.assertIsNotNone(self.orchestrator)
        self.assertIsNotNone(self.orchestrator.regime_detector)
        self.assertIsNotNone(self.orchestrator.tas_integration)
        self.assertIsNotNone(self.orchestrator.nas_integration)
        self.assertIsNotNone(self.orchestrator.economic_evaluator)
        self.assertIsNotNone(self.orchestrator.regime_tagger)
    
    def test_process_regime_detection_success(self):
        """Test successful regime detection processing."""
        result = self.orchestrator.process_regime_detection(
            market_data=self.sample_data,
            tas_inputs=self.tas_inputs,
            nas_inputs=self.nas_inputs,
            enable_tagging=True,
            save_results=False
        )
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.regime_predictions), 100)
        self.assertEqual(len(result.regime_probabilities), 100)
        self.assertGreater(len(result.regime_labels), 0)
        self.assertEqual(len(result.economic_significance_scores), 100)
        self.assertEqual(len(result.financial_significance_scores), 100)
        self.assertEqual(len(result.trading_viability_scores), 100)
        self.assertIsNotNone(result.tagged_data)
        self.assertGreater(result.execution_time, 0)
    
    def test_process_regime_detection_without_inputs(self):
        """Test regime detection without TAS/NAS inputs."""
        result = self.orchestrator.process_regime_detection(
            market_data=self.sample_data,
            enable_tagging=False,
            save_results=False
        )
        
        # Should still work but may have different results
        self.assertTrue(result.success)
        self.assertEqual(len(result.regime_predictions), 100)
    
    def test_process_regime_detection_without_tagging(self):
        """Test regime detection without tagging."""
        result = self.orchestrator.process_regime_detection(
            market_data=self.sample_data,
            tas_inputs=self.tas_inputs,
            nas_inputs=self.nas_inputs,
            enable_tagging=False,
            save_results=False
        )
        
        self.assertTrue(result.success)
        self.assertIsNone(result.tagged_data)
    
    def test_get_performance_summary(self):
        """Test performance summary retrieval."""
        # Run a detection first
        self.orchestrator.process_regime_detection(
            market_data=self.sample_data,
            tas_inputs=self.tas_inputs,
            nas_inputs=self.nas_inputs,
            enable_tagging=False,
            save_results=False
        )
        
        summary = self.orchestrator.get_performance_summary()
        
        self.assertIn('latest_execution_time', summary)
        self.assertIn('latest_n_samples', summary)
        self.assertIn('latest_n_regimes', summary)
        self.assertIn('total_runs', summary)
        self.assertEqual(summary['total_runs'], 1)
    
    def test_get_regime_summary(self):
        """Test regime summary retrieval."""
        # Run a detection first
        self.orchestrator.process_regime_detection(
            market_data=self.sample_data,
            tas_inputs=self.tas_inputs,
            nas_inputs=self.nas_inputs,
            enable_tagging=False,
            save_results=False
        )
        
        summary = self.orchestrator.get_regime_summary()
        
        self.assertIn('n_samples', summary)
        self.assertIn('n_regimes', summary)
        self.assertIn('regime_labels', summary)
        self.assertIn('average_economic_significance', summary)
        self.assertIn('average_financial_significance', summary)
        self.assertIn('average_trading_viability', summary)
        self.assertIn('execution_time', summary)
        self.assertIn('success', summary)
        self.assertTrue(summary['success'])
    
    def test_get_tagged_data(self):
        """Test tagged data retrieval."""
        # Run a detection with tagging
        self.orchestrator.process_regime_detection(
            market_data=self.sample_data,
            tas_inputs=self.tas_inputs,
            nas_inputs=self.nas_inputs,
            enable_tagging=True,
            save_results=False
        )
        
        tagged_data = self.orchestrator.get_tagged_data()
        
        self.assertIsNotNone(tagged_data)
        self.assertIsInstance(tagged_data, pd.DataFrame)
        self.assertEqual(len(tagged_data), 100)
        
        # Check for regime-related columns
        expected_columns = ['regime_id', 'regime_label', 'economic_significance', 'financial_significance']
        for col in expected_columns:
            self.assertIn(col, tagged_data.columns)
    
    def test_error_handling(self):
        """Test error handling with invalid data."""
        # Test with empty data
        empty_data = pd.DataFrame()
        
        result = self.orchestrator.process_regime_detection(
            market_data=empty_data,
            enable_tagging=False,
            save_results=False
        )
        
        # Should handle gracefully
        self.assertIsNotNone(result)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)


if __name__ == '__main__':
    unittest.main()