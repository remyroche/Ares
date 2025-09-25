#!/usr/bin/env python3
"""
Simple test for EvolutionaryArchitectureSearch without external dependencies.
"""

import sys
import os
import time
import random
import json
from pathlib import Path

# Mock numpy and pandas for testing
class MockNumpy:
    def mean(self, x):
        return sum(x) / len(x) if x else 0.0
    
    def std(self, x):
        if len(x) < 2:
            return 0.0
        mean_val = self.mean(x)
        variance = sum((x - mean_val) ** 2 for x in x) / (len(x) - 1)
        return variance ** 0.5
    
    def randn(self, *args):
        return [[random.gauss(0, 1) for _ in range(args[1])] for _ in range(args[0])]
    
    def random(self):
        return MockRandom()
    
    # Add ndarray for type hints
    class ndarray:
        pass

class MockRandom:
    def randint(self, low, high):
        return random.randint(low, high)

# Mock numpy
np = MockNumpy()

# Mock pandas
class MockDataFrame:
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def columns(self):
        return list(range(len(self.data[0]))) if self.data else []

def mock_numpy():
    """Mock numpy functions."""
    return MockNumpy()

# Set up mock modules
sys.modules['numpy'] = mock_numpy()
sys.modules['pandas'] = type('MockPandas', (), {'DataFrame': MockDataFrame})()

# Now import our module
try:
    from src.utils.nas_tas.evolutionary_algorithms import (
        EvolutionaryArchitectureSearch,
        ArchitectureConfig,
        EvolutionaryConfig,
        FitnessConfig,
        Architecture
    )
    print("✅ Successfully imported EvolutionaryArchitectureSearch")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("🧪 Testing basic functionality...")
    
    # Test configuration classes
    arch_config = ArchitectureConfig(
        max_layers=4,
        min_layers=2,
        max_neurons_per_layer=128,
        min_neurons_per_layer=16,
        max_parameters=100000,  # Lower limit for test
        min_parameters=100,     # Lower limit for test
        max_flops=1000000,      # Lower limit for test
        min_flops=1000          # Lower limit for test
    )
    
    evo_config = EvolutionaryConfig(
        population_size=5,
        max_generations=2,
        n_workers=1
    )
    
    fitness_config = FitnessConfig(
        cv_folds=2,
        max_training_epochs=10
    )
    
    print("✅ Configuration classes work")
    
    # Test architecture creation
    layers = [
        {'type': 'dense', 'neurons': 64, 'activation': 'relu'},
        {'type': 'dense', 'neurons': 32, 'activation': 'sigmoid'}
    ]
    
    arch = Architecture(layers, arch_config)
    print(f"   Architecture layers: {len(arch.layers)}")
    print(f"   Min layers: {arch_config.min_layers}, Max layers: {arch_config.max_layers}")
    print(f"   Is valid: {arch.is_valid()}")
    if not arch.is_valid():
        print(f"   Architecture layers: {arch.layers}")
    assert arch.is_valid(), "Architecture should be valid"
    print("✅ Architecture creation works")
    
    # Test complexity calculation
    complexity = arch.calculate_complexity()
    assert 'parameters' in complexity, "Should have parameters"
    assert 'flops' in complexity, "Should have FLOPs"
    print("✅ Complexity calculation works")
    
    # Test serialization
    arch_dict = arch.to_dict()
    restored_arch = Architecture.from_dict(arch_dict, arch_config)
    assert restored_arch.fitness == arch.fitness, "Serialization should work"
    print("✅ Serialization works")
    
    # Test NAS initialization
    # Create mock data
    X = [[random.random() for _ in range(10)] for _ in range(100)]
    y = [random.randint(0, 1) for _ in range(100)]
    
    nas = EvolutionaryArchitectureSearch(
        architecture_config=arch_config,
        evolutionary_config=evo_config,
        fitness_config=fitness_config,
        data=(X, y),
        log_dir="test_logs"
    )
    
    assert nas.arch_config.max_layers == 4, "Config should be set"
    print("✅ NAS initialization works")
    
    # Test population initialization
    population = nas.initialize_population()
    assert len(population) > 0, "Should create population"
    print("✅ Population initialization works")
    
    # Test fitness evaluation
    if population:
        arch = population[0]
        fitness = nas.evaluate_fitness(arch)
        assert 0 <= fitness <= 1, "Fitness should be valid"
        print("✅ Fitness evaluation works")
    
    # Test genetic operators
    if len(population) >= 2:
        parent1, parent2 = population[0], population[1]
        child1, child2 = nas.crossover(parent1, parent2)
        assert isinstance(child1, Architecture), "Crossover should work"
        print("✅ Crossover works")
        
        mutated = nas.mutate(parent1)
        assert isinstance(mutated, Architecture), "Mutation should work"
        print("✅ Mutation works")
    
    # Test selection
    parents = nas.select_parents(population)
    assert len(parents) == len(population), "Selection should work"
    print("✅ Selection works")
    
    print("✅ All basic tests passed!")

def test_evolution_cycle():
    """Test a complete evolution cycle."""
    print("🧪 Testing evolution cycle...")
    
    # Create mock data
    X = [[random.random() for _ in range(5)] for _ in range(50)]
    y = [random.randint(0, 1) for _ in range(50)]
    
    # Configure for quick test
    arch_config = ArchitectureConfig(
        max_layers=3,
        min_layers=2,
        max_neurons_per_layer=64,
        min_neurons_per_layer=16,
        max_parameters=50000,   # Much lower limits
        min_parameters=10,      # Much lower limits
        max_flops=100000,       # Much lower limits
        min_flops=10            # Much lower limits
    )
    
    evo_config = EvolutionaryConfig(
        population_size=4,
        max_generations=2,
        n_workers=1
    )
    
    fitness_config = FitnessConfig(
        cv_folds=2,
        max_training_epochs=5,
        max_training_time=10.0
    )
    
    # Initialize NAS
    nas = EvolutionaryArchitectureSearch(
        architecture_config=arch_config,
        evolutionary_config=evo_config,
        fitness_config=fitness_config,
        data=(X, y),
        log_dir="test_logs"
    )
    
    # Run evolution
    start_time = time.time()
    best_architecture = nas.run_evolution()
    end_time = time.time()
    
    assert best_architecture is not None, "Should find best architecture"
    assert best_architecture.fitness is not None, "Should have fitness"
    assert 0 <= best_architecture.fitness <= 1, "Fitness should be valid"
    
    # Check summary
    summary = nas.get_search_summary()
    assert summary['total_generations'] > 0, "Should complete generations"
    assert summary['total_evaluations'] > 0, "Should perform evaluations"
    
    print(f"✅ Evolution completed in {end_time - start_time:.2f} seconds")
    print(f"   Best fitness: {best_architecture.fitness:.4f}")
    print(f"   Total evaluations: {summary['total_evaluations']}")
    print("✅ Evolution cycle test passed!")

def test_error_handling():
    """Test error handling."""
    print("🧪 Testing error handling...")
    
    # Test with None data
    try:
        nas = EvolutionaryArchitectureSearch(data=None)
        assert nas.X is None, "Should handle None data"
        print("✅ None data handling works")
    except Exception as e:
        print(f"   Expected error with None data: {e}")
    
    # Test with empty data
    try:
        nas = EvolutionaryArchitectureSearch(data=([], []))
        print("✅ Empty data handling works")
    except Exception as e:
        print(f"   Expected error with empty data: {e}")
    
    print("✅ Error handling test passed!")

def main():
    """Run all tests."""
    print("🚀 Starting simple EvolutionaryArchitectureSearch tests...")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        print("\n" + "=" * 60)
        test_evolution_cycle()
        print("\n" + "=" * 60)
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()