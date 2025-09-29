"""
Test Enhanced NAS Implementations

This script tests the advanced neural architectures and enhanced search strategies
to ensure they work correctly and provide the expected functionality.
"""

import torch
import numpy as np
import logging
from pathlib import Path
import sys
import traceback

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from src.training.steps.market_analysis.nas_regime.core.advanced_neural_architectures import (
    AdvancedArchitectureConfig, ArchitectureType, create_advanced_architecture,
    AdvancedArchitectureManager, TransformerRegimeDetector, GraphNeuralNetworkRegimeDetector,
    TemporalConvolutionalRegimeDetector, HybridTransformerGNN
)

from src.training.steps.market_analysis.nas_regime.core.enhanced_search_strategies import (
    SearchStrategyConfig, SearchStrategyType, create_search_strategy,
    create_enhanced_search_manager, ReinforcementLearningSearch,
    ProgressiveArchitectureSearch, MultiObjectiveEvolutionarySearch
)

from src.training.steps.market_analysis.nas_regime.core.enhanced_nas_integration import (
    EnhancedNASConfig, EnhancedNASSystem, create_enhanced_nas_system,
    SearchSpace, Architecture, Layer, PerformanceEvaluator
)

from src.utils.tprint import tprint, tprint_success, tprint_info, tprint_warning, tprint_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRunner:
    """Test runner for Enhanced NAS implementations."""
    
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        
    def run_test(self, test_name: str, test_func):
        """Run a single test."""
        tprint(f"🧪 Running test: {test_name}", color="blue")
        
        try:
            result = test_func()
            if result:
                tprint_success(f"✅ {test_name} PASSED")
                self.test_results[test_name] = {"status": "PASSED", "result": result}
                self.passed_tests += 1
            else:
                tprint_error(f"❌ {test_name} FAILED: No result returned")
                self.test_results[test_name] = {"status": "FAILED", "error": "No result returned"}
                self.failed_tests += 1
        except Exception as e:
            tprint_error(f"❌ {test_name} FAILED: {e}")
            self.test_results[test_name] = {"status": "FAILED", "error": str(e)}
            self.failed_tests += 1
            logger.exception(f"Full traceback for {test_name}:")
    
    def run_all_tests(self):
        """Run all tests."""
        tprint("🚀 Starting Enhanced NAS Implementation Tests", color="cyan", bold=True)
        tprint("=" * 60, color="cyan")
        
        # Test advanced neural architectures
        self.run_test("Test Transformer Regime Detector", test_transformer_regime_detector)
        self.run_test("Test Graph Neural Network Regime Detector", test_gnn_regime_detector)
        self.run_test("Test Temporal Convolutional Regime Detector", test_temporal_conv_regime_detector)
        self.run_test("Test Hybrid Transformer GNN", test_hybrid_transformer_gnn)
        self.run_test("Test Advanced Architecture Manager", test_advanced_architecture_manager)
        
        # Test enhanced search strategies
        self.run_test("Test Reinforcement Learning Search", test_reinforcement_learning_search)
        self.run_test("Test Progressive Architecture Search", test_progressive_architecture_search)
        self.run_test("Test Multi-Objective Evolutionary Search", test_multi_objective_evolutionary_search)
        
        # Test enhanced NAS integration
        self.run_test("Test Enhanced NAS System", test_enhanced_nas_system)
        self.run_test("Test Search Space", test_search_space)
        self.run_test("Test Performance Evaluator", test_performance_evaluator)
        
        # Test integration
        self.run_test("Test End-to-End Integration", test_end_to_end_integration)
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        tprint("\n" + "=" * 60, color="cyan")
        tprint("📊 TEST SUMMARY", color="cyan", bold=True)
        tprint("=" * 60, color="cyan")
        
        total_tests = self.passed_tests + self.failed_tests
        tprint(f"Total Tests: {total_tests}", color="blue")
        tprint(f"Passed: {self.passed_tests}", color="green")
        tprint(f"Failed: {self.failed_tests}", color="red")
        tprint(f"Success Rate: {(self.passed_tests/total_tests)*100:.1f}%", color="blue")
        
        if self.failed_tests > 0:
            tprint("\n❌ FAILED TESTS:", color="red")
            for test_name, result in self.test_results.items():
                if result["status"] == "FAILED":
                    tprint(f"   {test_name}: {result['error']}", color="red")
        
        tprint("\n" + "=" * 60, color="cyan")


def test_transformer_regime_detector():
    """Test Transformer-based regime detector."""
    try:
        config = AdvancedArchitectureConfig()
        config.architecture_type = ArchitectureType.TRANSFORMER_REGIME
        config.input_dim = 64
        config.hidden_dim = 128
        config.num_heads = 4
        config.num_layers = 3
        config.num_regimes = 5
        
        # Create model
        model = TransformerRegimeDetector(config)
        
        # Test forward pass
        batch_size, seq_len, input_dim = 2, 10, 64
        x = torch.randn(batch_size, seq_len, input_dim)
        regime_labels = torch.randint(0, 5, (batch_size, seq_len))
        
        output = model(x, regime_labels)
        
        # Check output structure
        assert 'regime_logits' in output
        assert 'hidden_states' in output
        assert output['regime_logits'].shape == (batch_size, seq_len, config.num_regimes)
        assert output['hidden_states'].shape == (batch_size, seq_len, config.hidden_dim)
        
        tprint(f"   ✓ Transformer regime detector created successfully")
        tprint(f"   ✓ Forward pass completed: {output['regime_logits'].shape}")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ Transformer regime detector test failed: {e}")
        return False


def test_gnn_regime_detector():
    """Test Graph Neural Network regime detector."""
    try:
        config = AdvancedArchitectureConfig()
        config.architecture_type = ArchitectureType.GRAPH_NEURAL_NETWORK
        config.input_dim = 32
        config.hidden_dim = 64
        config.num_heads = 4
        config.num_layers = 2
        config.num_regimes = 4
        
        # Create model
        model = GraphNeuralNetworkRegimeDetector(config)
        
        # Test forward pass
        batch_size, num_nodes, seq_len, input_dim = 2, 5, 8, 32
        node_features = torch.randn(batch_size, num_nodes, seq_len, input_dim)
        adjacency_matrix = torch.rand(num_nodes, num_nodes)
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2  # Make symmetric
        
        output = model(node_features, adjacency_matrix)
        
        # Check output structure
        assert 'regime_logits' in output
        assert 'node_embeddings' in output
        assert output['regime_logits'].shape == (batch_size, seq_len, config.num_regimes)
        assert output['node_embeddings'].shape == (batch_size, num_nodes, seq_len, config.hidden_dim)
        
        tprint(f"   ✓ GNN regime detector created successfully")
        tprint(f"   ✓ Forward pass completed: {output['regime_logits'].shape}")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ GNN regime detector test failed: {e}")
        return False


def test_temporal_conv_regime_detector():
    """Test Temporal Convolutional regime detector."""
    try:
        config = AdvancedArchitectureConfig()
        config.architecture_type = ArchitectureType.TEMPORAL_CONVOLUTIONAL
        config.input_dim = 32
        config.hidden_dim = 64
        config.num_heads = 4
        config.num_layers = 3
        config.num_regimes = 4
        
        # Create model
        model = TemporalConvolutionalRegimeDetector(config)
        
        # Test forward pass
        batch_size, seq_len, input_dim = 2, 16, 32
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output = model(x)
        
        # Check output structure
        assert 'regime_logits' in output
        assert 'temporal_features' in output
        assert output['regime_logits'].shape == (batch_size, seq_len, config.num_regimes)
        assert output['temporal_features'].shape == (batch_size, seq_len, config.hidden_dim)
        
        tprint(f"   ✓ Temporal convolutional regime detector created successfully")
        tprint(f"   ✓ Forward pass completed: {output['regime_logits'].shape}")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ Temporal convolutional regime detector test failed: {e}")
        return False


def test_hybrid_transformer_gnn():
    """Test Hybrid Transformer-GNN architecture."""
    try:
        config = AdvancedArchitectureConfig()
        config.architecture_type = ArchitectureType.HYBRID_TRANSFORMER_GNN
        config.input_dim = 32
        config.hidden_dim = 64
        config.num_heads = 4
        config.num_layers = 2
        config.num_regimes = 4
        
        # Create model
        model = HybridTransformerGNN(config)
        
        # Test forward pass
        batch_size, seq_len, input_dim = 2, 8, 32
        x = torch.randn(batch_size, seq_len, input_dim)
        adjacency_matrix = torch.rand(seq_len, seq_len)
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2  # Make symmetric
        
        output = model(x, adjacency_matrix)
        
        # Check output structure
        assert 'regime_logits' in output
        assert 'transformer_logits' in output
        assert 'temporal_logits' in output
        assert 'gnn_logits' in output
        assert output['regime_logits'].shape == (batch_size, seq_len, config.num_regimes)
        
        tprint(f"   ✓ Hybrid Transformer-GNN created successfully")
        tprint(f"   ✓ Forward pass completed: {output['regime_logits'].shape}")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ Hybrid Transformer-GNN test failed: {e}")
        return False


def test_advanced_architecture_manager():
    """Test Advanced Architecture Manager."""
    try:
        config = AdvancedArchitectureConfig()
        config.architecture_type = ArchitectureType.TRANSFORMER_REGIME
        config.input_dim = 32
        config.hidden_dim = 64
        config.num_regimes = 4
        
        # Create manager
        manager = AdvancedArchitectureManager(config)
        
        # Test forward pass
        batch_size, seq_len, input_dim = 1, 5, 32
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output = manager.forward(x)
        
        # Check output structure
        assert 'regime_logits' in output
        assert output['regime_logits'].shape == (batch_size, seq_len, config.num_regimes)
        
        # Test architecture info
        info = manager.get_architecture_info()
        assert 'architecture_type' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        
        tprint(f"   ✓ Advanced architecture manager created successfully")
        tprint(f"   ✓ Architecture info: {info['total_parameters']} total parameters")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ Advanced architecture manager test failed: {e}")
        return False


def test_reinforcement_learning_search():
    """Test Reinforcement Learning-based search."""
    try:
        # Create search space
        operations = ["conv1d", "linear", "lstm", "attention"]
        search_space = SearchSpace(operations, max_layers=5)
        
        # Create performance evaluator
        config = EnhancedNASConfig()
        evaluator = PerformanceEvaluator(config)
        
        # Create RL search strategy
        search_config = SearchStrategyConfig()
        search_config.strategy_type = SearchStrategyType.REINFORCEMENT_LEARNING
        search_config.rl_learning_rate = 0.01
        search_config.max_search_iterations = 10
        
        rl_search = ReinforcementLearningSearch(search_space, evaluator, search_config)
        
        # Run search
        result = rl_search.search(max_episodes=5)
        
        # Check result structure
        assert 'best_architecture' in result
        assert 'best_performance' in result
        assert 'search_history' in result
        
        tprint(f"   ✓ Reinforcement learning search completed")
        tprint(f"   ✓ Best performance: {result['best_performance']:.4f}")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ Reinforcement learning search test failed: {e}")
        return False


def test_progressive_architecture_search():
    """Test Progressive Architecture Search."""
    try:
        # Create search space
        operations = ["conv1d", "linear", "lstm"]
        search_space = SearchSpace(operations, max_layers=3)
        
        # Create performance evaluator
        config = EnhancedNASConfig()
        evaluator = PerformanceEvaluator(config)
        
        # Create progressive search strategy
        search_config = SearchStrategyConfig()
        search_config.strategy_type = SearchStrategyType.PROGRESSIVE_SEARCH
        search_config.progressive_initial_ops = 1
        search_config.progressive_max_ops = 3
        search_config.progressive_evolution_rounds = 2
        search_config.mo_population_size = 5
        
        progressive_search = ProgressiveArchitectureSearch(search_space, evaluator, search_config)
        
        # Run search
        result = progressive_search.search()
        
        # Check result structure
        assert 'best_architecture' in result
        assert 'best_performance' in result
        assert 'search_history' in result
        
        tprint(f"   ✓ Progressive architecture search completed")
        tprint(f"   ✓ Best performance: {result['best_performance']:.4f}")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ Progressive architecture search test failed: {e}")
        return False


def test_multi_objective_evolutionary_search():
    """Test Multi-Objective Evolutionary Search."""
    try:
        # Create search space
        operations = ["conv1d", "linear", "lstm"]
        search_space = SearchSpace(operations, max_layers=3)
        
        # Create performance evaluator
        config = EnhancedNASConfig()
        evaluator = PerformanceEvaluator(config)
        
        # Create multi-objective search strategy
        search_config = SearchStrategyConfig()
        search_config.strategy_type = SearchStrategyType.MULTI_OBJECTIVE_EVOLUTIONARY
        search_config.mo_population_size = 5
        search_config.mo_generations = 3
        
        mo_search = MultiObjectiveEvolutionarySearch(search_space, evaluator, search_config)
        
        # Run search
        result = mo_search.search()
        
        # Check result structure
        assert 'pareto_front' in result
        assert 'search_history' in result
        assert 'best_architectures' in result
        
        tprint(f"   ✓ Multi-objective evolutionary search completed")
        tprint(f"   ✓ Pareto front size: {len(result['pareto_front'])}")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ Multi-objective evolutionary search test failed: {e}")
        return False


def test_enhanced_nas_system():
    """Test Enhanced NAS System."""
    try:
        # Create configuration
        config = EnhancedNASConfig()
        config.architecture_config.architecture_type = ArchitectureType.TRANSFORMER_REGIME
        config.search_config.strategy_type = SearchStrategyType.REINFORCEMENT_LEARNING
        config.max_search_iterations = 10
        config.output_dir = "test_results"
        
        # Create system
        nas_system = create_enhanced_nas_system(config)
        
        # Run search
        result = nas_system.search()
        
        # Check result structure
        assert hasattr(result, 'success')
        assert hasattr(result, 'best_architecture')
        assert hasattr(result, 'best_performance')
        assert hasattr(result, 'execution_time')
        
        tprint(f"   ✓ Enhanced NAS system created successfully")
        tprint(f"   ✓ Search completed: Success = {result.success}")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ Enhanced NAS system test failed: {e}")
        return False


def test_search_space():
    """Test Search Space functionality."""
    try:
        operations = ["conv1d", "linear", "lstm", "attention"]
        search_space = SearchSpace(operations, max_layers=5)
        
        # Test empty architecture creation
        empty_arch = search_space.create_empty_architecture()
        assert len(empty_arch.layers) == 0
        
        # Test operation application
        arch = search_space.create_empty_architecture()
        arch = search_space.apply_operation(arch, 0)
        assert len(arch.layers) == 1
        assert arch.layers[0].operation_id == 0
        
        # Test random architecture sampling
        random_arch = search_space.sample_random_architecture()
        assert len(random_arch.layers) > 0
        assert len(random_arch.layers) <= search_space.max_layers
        
        tprint(f"   ✓ Search space created successfully")
        tprint(f"   ✓ Random architecture sampled: {len(random_arch.layers)} layers")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ Search space test failed: {e}")
        return False


def test_performance_evaluator():
    """Test Performance Evaluator."""
    try:
        config = EnhancedNASConfig()
        evaluator = PerformanceEvaluator(config)
        
        # Create test architecture
        operations = ["conv1d", "linear", "lstm"]
        search_space = SearchSpace(operations, max_layers=3)
        arch = search_space.sample_random_architecture()
        
        # Test evaluation
        performance1 = evaluator(arch)
        performance2 = evaluator(arch)  # Should use cache
        
        assert 0.0 <= performance1 <= 1.0
        assert performance1 == performance2  # Should be cached
        
        tprint(f"   ✓ Performance evaluator created successfully")
        tprint(f"   ✓ Architecture performance: {performance1:.4f}")
        tprint(f"   ✓ Cache hit rate: {len(evaluator.evaluation_cache)}/{evaluator.evaluation_count}")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ Performance evaluator test failed: {e}")
        return False


def test_end_to_end_integration():
    """Test end-to-end integration."""
    try:
        # Create configuration
        config = EnhancedNASConfig()
        config.architecture_config.architecture_type = ArchitectureType.TRANSFORMER_REGIME
        config.search_config.strategy_type = SearchStrategyType.PROGRESSIVE_SEARCH
        config.max_search_iterations = 5
        config.output_dir = "test_integration_results"
        
        # Create system
        nas_system = create_enhanced_nas_system(config)
        
        # Run search
        result = nas_system.search()
        
        # Check that we got a valid result
        assert result is not None
        assert hasattr(result, 'success')
        
        if result.success:
            assert result.best_architecture is not None
            assert result.best_performance >= 0.0
            assert result.execution_time > 0.0
        
        tprint(f"   ✓ End-to-end integration completed")
        tprint(f"   ✓ Result success: {result.success}")
        
        return True
        
    except Exception as e:
        tprint_error(f"   ❌ End-to-end integration test failed: {e}")
        return False


def main():
    """Main test function."""
    try:
        # Create output directories
        test_dirs = ["test_results", "test_integration_results"]
        for test_dir in test_dirs:
            Path(test_dir).mkdir(parents=True, exist_ok=True)
        
        # Run all tests
        test_runner = TestRunner()
        test_runner.run_all_tests()
        
        return test_runner.passed_tests > 0
        
    except Exception as e:
        tprint_error(f"❌ Test suite failed: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        tprint_success("🎉 Test suite completed successfully!")
        sys.exit(0)
    else:
        tprint_error("❌ Test suite failed!")
        sys.exit(1)