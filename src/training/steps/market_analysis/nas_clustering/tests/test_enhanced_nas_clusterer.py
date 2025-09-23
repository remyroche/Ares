"""
Tests for Enhanced NAS Clusterer

This module provides comprehensive tests for the enhanced NAS clusterer with true
Neural Architecture Search capabilities.
"""

import unittest
import numpy as np
import pandas as pd
import time
import logging
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
from ..core.enhanced_nas_clusterer import EnhancedNASClusterer, EnhancedNASClusteringResult
from ..core.nas_config import NASClusteringConfig, NASArchitectureType
from ..core.nas_search.search_space import SearchSpace, LayerType, ActivationFunction

logger = logging.getLogger(__name__)


class TestEnhancedNASClusterer(unittest.TestCase):
    """Test cases for EnhancedNASClusterer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NASClusteringConfig(
            timeframe="15m",
            n_regimes=5,
            nas_architecture_type=NASArchitectureType.HYBRID,
            enable_true_nas=True,
            nas_generations=10,
            nas_population_size=20,
            enable_multi_objective=True,
            enable_hardware_acceleration=False,  # Disable for testing
            enable_matrix_optimization=False
        )
        
        # Create test data
        np.random.seed(42)
        self.test_data = np.random.randn(100, 10)
        self.test_timestamps = np.arange(100)
        
        # Mock hardware and matrix operations
        self.mock_matrix_ops = Mock()
        self.mock_hardware_manager = Mock()
    
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedMatrixOperations')
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedHardwareManager')
    def test_initialization(self, mock_hardware, mock_matrix_ops):
        """Test clusterer initialization."""
        mock_matrix_ops.return_value = self.mock_matrix_ops
        mock_hardware.return_value = self.mock_hardware_manager
        
        clusterer = EnhancedNASClusterer(self.config)
        
        self.assertIsNotNone(clusterer.config)
        self.assertEqual(clusterer.nas_architecture_type, NASArchitectureType.HYBRID)
        self.assertEqual(clusterer.n_regimes, 5)
        self.assertTrue(clusterer.enable_true_nas)
        self.assertTrue(clusterer.enable_multi_objective)
    
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedMatrixOperations')
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedHardwareManager')
    def test_initialization_without_nas(self, mock_hardware, mock_matrix_ops):
        """Test clusterer initialization with NAS disabled."""
        mock_matrix_ops.return_value = self.mock_matrix_ops
        mock_hardware.return_value = self.mock_hardware_manager
        
        config_no_nas = NASClusteringConfig(
            timeframe="15m",
            n_regimes=5,
            enable_true_nas=False,
            enable_hardware_acceleration=False
        )
        
        clusterer = EnhancedNASClusterer(config_no_nas)
        
        self.assertFalse(clusterer.enable_true_nas)
    
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedMatrixOperations')
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedHardwareManager')
    def test_search_space_initialization(self, mock_hardware, mock_matrix_ops):
        """Test search space initialization for different architecture types."""
        mock_matrix_ops.return_value = self.mock_matrix_ops
        mock_hardware.return_value = self.mock_hardware_manager
        
        # Test volatility-focused architecture
        config_vol = NASClusteringConfig(
            timeframe="15m",
            nas_architecture_type=NASArchitectureType.VOLATILITY_FOCUSED,
            enable_hardware_acceleration=False
        )
        
        clusterer_vol = EnhancedNASClusterer(config_vol)
        self.assertIsNotNone(clusterer_vol.search_space)
        self.assertIn(LayerType.LSTM, clusterer_vol.search_space.available_layer_types)
    
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedMatrixOperations')
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedHardwareManager')
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.EvolutionaryArchitectureSearch')
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.RegimeDetectionMultiObjective')
    def test_true_nas_enabled(self, mock_multi_obj, mock_evolutionary, mock_hardware, mock_matrix_ops):
        """Test clusterer with true NAS enabled."""
        mock_matrix_ops.return_value = self.mock_matrix_ops
        mock_hardware.return_value = self.mock_hardware_manager
        
        # Mock evolutionary search
        mock_evolutionary_instance = Mock()
        mock_evolutionary_instance.search.return_value = Mock(fitness_score=0.8)
        mock_evolutionary.return_value = mock_evolutionary_instance
        
        # Mock multi-objective optimizer
        mock_multi_obj_instance = Mock()
        mock_pareto_frontier = Mock()
        mock_pareto_frontier.get_best_solutions.return_value = [Mock(architecture=Mock())]
        mock_multi_obj_instance.optimize_nsga2.return_value = mock_pareto_frontier
        mock_multi_obj_instance.get_optimization_summary.return_value = {'total_solutions': 10}
        mock_multi_obj.return_value = mock_multi_obj_instance
        
        clusterer = EnhancedNASClusterer(self.config)
        
        self.assertTrue(clusterer.enable_true_nas)
        self.assertIsNotNone(clusterer.evolutionary_search)
        self.assertIsNotNone(clusterer.multi_objective_optimizer)
    
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedMatrixOperations')
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedHardwareManager')
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.NASFeatureExtractor')
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.MicroRegimeDetector')
    def test_traditional_clustering_fallback(self, mock_micro_regime, mock_feature_extractor, 
                                           mock_hardware, mock_matrix_ops):
        """Test traditional clustering fallback."""
        mock_matrix_ops.return_value = self.mock_matrix_ops
        mock_hardware.return_value = self.mock_hardware_manager
        
        # Mock feature extractor
        mock_feature_result = Mock()
        mock_feature_result.features = np.random.randn(100, 5)
        mock_feature_extractor.return_value.extract_features.return_value = mock_feature_result
        
        # Mock micro-regime detector
        mock_micro_regime_result = Mock()
        mock_micro_regime.return_value.detect_micro_regimes.return_value = mock_micro_regime_result
        
        # Create clusterer with NAS disabled
        config_no_nas = NASClusteringConfig(
            timeframe="15m",
            n_regimes=5,
            enable_true_nas=False,
            enable_hardware_acceleration=False
        )
        
        clusterer = EnhancedNASClusterer(config_no_nas)
        
        # Test clustering
        result = clusterer.cluster(self.test_data, self.test_timestamps)
        
        self.assertIsInstance(result, EnhancedNASClusteringResult)
        self.assertTrue(result.success)
        self.assertEqual(result.metadata['method'], 'enhanced_nas_clustering')
    
    def test_nas_label_generation(self):
        """Test NAS label generation."""
        with patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedMatrixOperations'):
            clusterer = EnhancedNASClusterer(self.config)
            
            labels = clusterer._generate_nas_labels(self.test_data, self.test_timestamps)
            
            self.assertEqual(len(labels), len(self.test_data))
            self.assertTrue(np.all(labels >= 0))
            self.assertTrue(np.all(labels < clusterer.n_regimes))
    
    def test_network_type_determination(self):
        """Test network type determination from architecture."""
        with patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedMatrixOperations'):
            clusterer = EnhancedNASClusterer(self.config)
            
            # Create mock architecture
            mock_architecture = Mock()
            mock_layer1 = Mock()
            mock_layer1.layer_type.value = 'lstm'
            mock_layer2 = Mock()
            mock_layer2.layer_type.value = 'dense'
            mock_architecture.layers = [mock_layer1, mock_layer2]
            
            network_type = clusterer._determine_network_type_from_architecture(mock_architecture)
            
            self.assertEqual(network_type, 'volatility')  # LSTM-heavy should be volatility
    
    def test_economic_significance_calculation(self):
        """Test economic significance score calculation."""
        with patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedMatrixOperations'):
            clusterer = EnhancedNASClusterer(self.config)
            
            labels = np.array([0, 0, 1, 1, 2, 2])
            economic_scores = clusterer._calculate_economic_significance_scores(self.test_data[:6], labels)
            
            self.assertEqual(len(economic_scores), len(labels))
            self.assertTrue(np.all(economic_scores >= 0))
            self.assertTrue(np.all(economic_scores <= 1))
    
    def test_trading_viability_calculation(self):
        """Test trading viability score calculation."""
        with patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedMatrixOperations'):
            clusterer = EnhancedNASClusterer(self.config)
            
            labels = np.array([0, 0, 1, 1, 2, 2])
            trading_scores = clusterer._calculate_trading_viability_scores(self.test_data[:6], labels)
            
            self.assertEqual(len(trading_scores), len(labels))
            self.assertTrue(np.all(trading_scores >= 0))
            self.assertTrue(np.all(trading_scores <= 1))
    
    def test_regime_transition_calculation(self):
        """Test regime transition matrix calculation."""
        with patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedMatrixOperations'):
            clusterer = EnhancedNASClusterer(self.config)
            
            labels = np.array([0, 0, 1, 1, 2, 2, 0, 1])
            transition_matrix = clusterer._calculate_regime_transitions(labels)
            
            self.assertEqual(transition_matrix.shape, (3, 3))  # 3 unique regimes
            self.assertTrue(np.allclose(transition_matrix.sum(axis=1), 1.0))  # Rows sum to 1


class TestTrueNASIntegration(unittest.TestCase):
    """Test cases for true NAS integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NASClusteringConfig(
            timeframe="15m",
            n_regimes=5,
            enable_true_nas=True,
            nas_generations=5,
            nas_population_size=10,
            enable_hardware_acceleration=False
        )
        
        self.test_data = np.random.randn(50, 8)
        self.test_timestamps = np.arange(50)
    
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedMatrixOperations')
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedHardwareManager')
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.EvolutionaryArchitectureSearch')
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.RegimeNetworkFactory')
    def test_true_nas_execution(self, mock_factory, mock_evolutionary, mock_hardware, mock_matrix_ops):
        """Test true NAS execution."""
        # Mock dependencies
        mock_matrix_ops.return_value = Mock()
        mock_hardware.return_value = Mock()
        
        # Mock evolutionary search
        mock_evolutionary_instance = Mock()
        mock_best_architecture = Mock()
        mock_best_architecture.fitness_score = 0.85
        mock_best_architecture.layers = [Mock(), Mock()]
        mock_best_architecture.connections = [Mock()]
        mock_evolutionary_instance.search.return_value = mock_best_architecture
        mock_evolutionary_instance.get_search_statistics.return_value = {'total_generations': 5}
        mock_evolutionary.return_value = mock_evolutionary_instance
        
        # Mock neural network factory
        mock_network = Mock()
        mock_network.train_network.return_value = {'final_accuracy': 0.8, 'training_time': 10.0}
        mock_network.predict.return_value = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        mock_factory.create_network.return_value = mock_network
        
        clusterer = EnhancedNASClusterer(self.config)
        
        # Mock feature extraction and micro-regime detection
        with patch.object(clusterer, 'feature_extractor') as mock_feature_extractor, \
             patch.object(clusterer, 'micro_regime_detector') as mock_micro_regime_detector:
            
            mock_feature_result = Mock()
            mock_feature_result.features = self.test_data
            mock_feature_extractor.extract_features.return_value = mock_feature_result
            
            mock_micro_regime_result = Mock()
            mock_micro_regime_detector.detect_micro_regimes.return_value = mock_micro_regime_result
            
            # Test clustering with true NAS
            result = clusterer.cluster(self.test_data, self.test_timestamps)
            
            self.assertIsInstance(result, EnhancedNASClusteringResult)
            self.assertTrue(result.success)
            self.assertIsNotNone(result.best_architecture)
            self.assertIsNotNone(result.neural_network_performance)
    
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedMatrixOperations')
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedHardwareManager')
    def test_nas_fallback_to_traditional(self, mock_hardware, mock_matrix_ops):
        """Test fallback to traditional clustering when NAS fails."""
        mock_matrix_ops.return_value = Mock()
        mock_hardware.return_value = Mock()
        
        clusterer = EnhancedNASClusterer(self.config)
        
        # Force NAS to fail by setting evolutionary_search to None
        clusterer.evolutionary_search = None
        
        # Mock feature extraction and micro-regime detection
        with patch.object(clusterer, 'feature_extractor') as mock_feature_extractor, \
             patch.object(clusterer, 'micro_regime_detector') as mock_micro_regime_detector:
            
            mock_feature_result = Mock()
            mock_feature_result.features = self.test_data
            mock_feature_extractor.extract_features.return_value = mock_feature_result
            
            mock_micro_regime_result = Mock()
            mock_micro_regime_detector.detect_micro_regimes.return_value = mock_micro_regime_result
            
            # Test clustering with fallback
            result = clusterer.cluster(self.test_data, self.test_timestamps)
            
            self.assertIsInstance(result, EnhancedNASClusteringResult)
            self.assertTrue(result.success)
            self.assertEqual(result.metadata['method'], 'enhanced_nas_clustering')


class TestMultiObjectiveOptimization(unittest.TestCase):
    """Test cases for multi-objective optimization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NASClusteringConfig(
            timeframe="15m",
            n_regimes=5,
            enable_true_nas=True,
            enable_multi_objective=True,
            nas_generations=5,
            nas_population_size=10,
            enable_hardware_acceleration=False
        )
        
        self.test_data = np.random.randn(50, 8)
        self.test_timestamps = np.arange(50)
    
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedMatrixOperations')
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedHardwareManager')
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.RegimeDetectionMultiObjective')
    def test_multi_objective_optimization_enabled(self, mock_multi_obj, mock_hardware, mock_matrix_ops):
        """Test clusterer with multi-objective optimization enabled."""
        mock_matrix_ops.return_value = Mock()
        mock_hardware.return_value = Mock()
        
        # Mock multi-objective optimizer
        mock_optimizer = Mock()
        mock_pareto_frontier = Mock()
        mock_pareto_frontier.get_best_solutions.return_value = [Mock(architecture=Mock())]
        mock_optimizer.optimize_nsga2.return_value = mock_pareto_frontier
        mock_optimizer.get_optimization_summary.return_value = {
            'total_solutions': 10,
            'num_fronts': 3
        }
        mock_multi_obj.return_value = mock_optimizer
        
        clusterer = EnhancedNASClusterer(self.config)
        
        self.assertTrue(clusterer.enable_multi_objective)
        self.assertIsNotNone(clusterer.multi_objective_optimizer)
    
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedMatrixOperations')
    @patch('src.training.steps.market_analysis.nas_clustering.core.enhanced_nas_clusterer.UnifiedHardwareManager')
    def test_multi_objective_optimization_disabled(self, mock_hardware, mock_matrix_ops):
        """Test clusterer with multi-objective optimization disabled."""
        mock_matrix_ops.return_value = Mock()
        mock_hardware.return_value = Mock()
        
        config_no_multi_obj = NASClusteringConfig(
            timeframe="15m",
            n_regimes=5,
            enable_true_nas=True,
            enable_multi_objective=False,
            enable_hardware_acceleration=False
        )
        
        clusterer = EnhancedNASClusterer(config_no_multi_obj)
        
        self.assertFalse(clusterer.enable_multi_objective)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main()