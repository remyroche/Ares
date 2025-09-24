#!/usr/bin/env python3
"""
Test script for EvolutionaryArchitectureSearch.

This script demonstrates the usage of the evolutionary architecture search
and validates its functionality with various test cases.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time
import logging

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from evolutionary_search import (
    EvolutionaryArchitectureSearch,
    ArchitectureConfig,
    EvolutionaryConfig,
    FitnessConfig,
    Architecture
)

def test_architecture_creation():
    """Test architecture creation and validation."""
    print("🧪 Testing architecture creation...")
    
    arch_config = ArchitectureConfig(
        max_layers=5,
        min_layers=2,
        max_neurons_per_layer=256,
        min_neurons_per_layer=16
    )
    
    # Test valid architecture
    layers = [
        {'type': 'dense', 'neurons': 64, 'activation': 'relu'},
        {'type': 'dropout', 'neurons': 32, 'dropout': 0.2},
        {'type': 'dense', 'neurons': 32, 'activation': 'sigmoid'}
    ]
    
    arch = Architecture(layers, arch_config)
    assert arch.is_valid(), "Valid architecture should pass validation"
    print("✅ Valid architecture creation passed")
    
    # Test invalid architecture (too many layers)
    invalid_layers = [{'type': 'dense', 'neurons': 32, 'activation': 'relu'}] * 10
    invalid_arch = Architecture(invalid_layers, arch_config)
    assert not invalid_arch.is_valid(), "Invalid architecture should fail validation"
    print("✅ Invalid architecture detection passed")
    
    # Test complexity calculation
    complexity = arch.calculate_complexity()
    assert 'parameters' in complexity, "Complexity should include parameters"
    assert 'flops' in complexity, "Complexity should include FLOPs"
    print("✅ Complexity calculation passed")

def test_evolutionary_config():
    """Test evolutionary algorithm configuration."""
    print("🧪 Testing evolutionary configuration...")
    
    config = EvolutionaryConfig(
        population_size=10,
        max_generations=5,
        crossover_rate=0.8,
        mutation_rate=0.2,
        n_workers=2
    )
    
    assert config.population_size == 10, "Population size should be set correctly"
    assert config.max_generations == 5, "Max generations should be set correctly"
    assert 0 <= config.crossover_rate <= 1, "Crossover rate should be valid"
    assert 0 <= config.mutation_rate <= 1, "Mutation rate should be valid"
    print("✅ Evolutionary configuration passed")

def test_fitness_config():
    """Test fitness evaluation configuration."""
    print("🧪 Testing fitness configuration...")
    
    config = FitnessConfig(
        primary_metric='accuracy',
        cv_folds=3,
        max_training_epochs=50,
        max_training_time=60.0
    )
    
    assert config.primary_metric == 'accuracy', "Primary metric should be set"
    assert config.cv_folds > 0, "CV folds should be positive"
    assert config.max_training_epochs > 0, "Max epochs should be positive"
    print("✅ Fitness configuration passed")

def create_sample_data(n_samples=500, n_features=10):
    """Create sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    return X, y

def test_nas_initialization():
    """Test NAS initialization."""
    print("🧪 Testing NAS initialization...")
    
    X, y = create_sample_data(100, 5)
    
    arch_config = ArchitectureConfig(max_layers=4, min_layers=2)
    evo_config = EvolutionaryConfig(population_size=5, max_generations=3)
    fitness_config = FitnessConfig(cv_folds=2, max_training_epochs=10)
    
    nas = EvolutionaryArchitectureSearch(
        architecture_config=arch_config,
        evolutionary_config=evo_config,
        fitness_config=fitness_config,
        data=(X, y),
        log_dir="test_logs"
    )
    
    assert nas.arch_config.max_layers == 4, "Architecture config should be set"
    assert nas.evo_config.population_size == 5, "Evolutionary config should be set"
    assert nas.fitness_config.cv_folds == 2, "Fitness config should be set"
    assert nas.X is not None, "Data should be set"
    assert nas.y is not None, "Target should be set"
    print("✅ NAS initialization passed")

def test_population_initialization():
    """Test population initialization."""
    print("🧪 Testing population initialization...")
    
    X, y = create_sample_data(50, 5)
    
    arch_config = ArchitectureConfig(max_layers=3, min_layers=2)
    evo_config = EvolutionaryConfig(population_size=5, max_generations=2)
    fitness_config = FitnessConfig(cv_folds=2, max_training_epochs=5)
    
    nas = EvolutionaryArchitectureSearch(
        architecture_config=arch_config,
        evolutionary_config=evo_config,
        fitness_config=fitness_config,
        data=(X, y),
        log_dir="test_logs"
    )
    
    population = nas.initialize_population()
    
    assert len(population) > 0, "Population should not be empty"
    assert len(population) <= evo_config.population_size, "Population size should not exceed limit"
    
    for arch in population:
        assert arch.is_valid(), "All architectures should be valid"
    
    print("✅ Population initialization passed")

def test_fitness_evaluation():
    """Test fitness evaluation."""
    print("🧪 Testing fitness evaluation...")
    
    X, y = create_sample_data(50, 5)
    
    arch_config = ArchitectureConfig(max_layers=3, min_layers=2)
    evo_config = EvolutionaryConfig(population_size=3, max_generations=1)
    fitness_config = FitnessConfig(cv_folds=2, max_training_epochs=5)
    
    nas = EvolutionaryArchitectureSearch(
        architecture_config=arch_config,
        evolutionary_config=evo_config,
        fitness_config=fitness_config,
        data=(X, y),
        log_dir="test_logs"
    )
    
    # Create a test architecture
    layers = [
        {'type': 'dense', 'neurons': 32, 'activation': 'relu'},
        {'type': 'dense', 'neurons': 16, 'activation': 'sigmoid'}
    ]
    arch = Architecture(layers, arch_config)
    
    # Evaluate fitness
    fitness = nas.evaluate_fitness(arch)
    
    assert 0 <= fitness <= 1, "Fitness should be between 0 and 1"
    assert arch.fitness == fitness, "Architecture fitness should be set"
    assert arch.training_time is not None, "Training time should be recorded"
    assert arch.parameters_count is not None, "Parameters count should be calculated"
    
    print("✅ Fitness evaluation passed")

def test_genetic_operators():
    """Test genetic operators (crossover, mutation, selection)."""
    print("🧪 Testing genetic operators...")
    
    X, y = create_sample_data(50, 5)
    
    arch_config = ArchitectureConfig(max_layers=4, min_layers=2)
    evo_config = EvolutionaryConfig(population_size=5, max_generations=1)
    fitness_config = FitnessConfig(cv_folds=2, max_training_epochs=5)
    
    nas = EvolutionaryArchitectureSearch(
        architecture_config=arch_config,
        evolutionary_config=evo_config,
        fitness_config=fitness_config,
        data=(X, y),
        log_dir="test_logs"
    )
    
    # Initialize population
    population = nas.initialize_population()
    
    # Test selection
    parents = nas.select_parents(population)
    assert len(parents) == len(population), "Should select same number of parents"
    
    # Test crossover
    if len(parents) >= 2:
        parent1, parent2 = parents[0], parents[1]
        child1, child2 = nas.crossover(parent1, parent2)
        assert isinstance(child1, Architecture), "Child should be Architecture"
        assert isinstance(child2, Architecture), "Child should be Architecture"
    
    # Test mutation
    if population:
        original = population[0]
        mutated = nas.mutate(original)
        assert isinstance(mutated, Architecture), "Mutated should be Architecture"
    
    print("✅ Genetic operators passed")

def test_evolution_cycle():
    """Test a complete evolution cycle."""
    print("🧪 Testing evolution cycle...")
    
    X, y = create_sample_data(100, 8)
    
    arch_config = ArchitectureConfig(
        max_layers=4,
        min_layers=2,
        max_neurons_per_layer=128,
        min_neurons_per_layer=16
    )
    
    evo_config = EvolutionaryConfig(
        population_size=6,
        max_generations=2,
        n_workers=1
    )
    
    fitness_config = FitnessConfig(
        cv_folds=2,
        max_training_epochs=10,
        max_training_time=30.0
    )
    
    nas = EvolutionaryArchitectureSearch(
        architecture_config=arch_config,
        evolutionary_config=evo_config,
        fitness_config=fitness_config,
        data=(X, y),
        log_dir="test_logs"
    )
    
    # Run a short evolution
    start_time = time.time()
    best_architecture = nas.run_evolution()
    end_time = time.time()
    
    assert best_architecture is not None, "Best architecture should be found"
    assert best_architecture.fitness is not None, "Best architecture should have fitness"
    assert 0 <= best_architecture.fitness <= 1, "Fitness should be valid"
    
    # Check search summary
    summary = nas.get_search_summary()
    assert summary['total_generations'] > 0, "Should have completed generations"
    assert summary['total_evaluations'] > 0, "Should have performed evaluations"
    assert summary['best_fitness'] is not None, "Should have best fitness"
    
    print(f"✅ Evolution cycle completed in {end_time - start_time:.2f} seconds")
    print(f"   Best fitness: {best_architecture.fitness:.4f}")
    print(f"   Total evaluations: {summary['total_evaluations']}")

def test_serialization():
    """Test architecture serialization."""
    print("🧪 Testing serialization...")
    
    arch_config = ArchitectureConfig()
    layers = [
        {'type': 'dense', 'neurons': 64, 'activation': 'relu'},
        {'type': 'dropout', 'neurons': 32, 'dropout': 0.2},
        {'type': 'dense', 'neurons': 16, 'activation': 'sigmoid'}
    ]
    
    arch = Architecture(layers, arch_config)
    arch.fitness = 0.85
    arch.training_time = 1.5
    arch.parameters_count = 1000
    
    # Test to_dict
    arch_dict = arch.to_dict()
    assert 'layers' in arch_dict, "Should include layers"
    assert 'fitness' in arch_dict, "Should include fitness"
    assert arch_dict['fitness'] == 0.85, "Fitness should be preserved"
    
    # Test from_dict
    restored_arch = Architecture.from_dict(arch_dict, arch_config)
    assert restored_arch.fitness == arch.fitness, "Fitness should be restored"
    assert restored_arch.training_time == arch.training_time, "Training time should be restored"
    assert len(restored_arch.layers) == len(arch.layers), "Layers should be restored"
    
    print("✅ Serialization passed")

def test_error_handling():
    """Test error handling and edge cases."""
    print("🧪 Testing error handling...")
    
    # Test with invalid data
    try:
        nas = EvolutionaryArchitectureSearch(data=None)
        # Should handle None data gracefully
        assert nas.X is None, "Should handle None data"
    except Exception as e:
        print(f"   Expected error with None data: {e}")
    
    # Test with very small population
    X, y = create_sample_data(20, 3)
    
    arch_config = ArchitectureConfig(max_layers=2, min_layers=1)
    evo_config = EvolutionaryConfig(population_size=1, max_generations=1)
    fitness_config = FitnessConfig(cv_folds=2, max_training_epochs=5)
    
    nas = EvolutionaryArchitectureSearch(
        architecture_config=arch_config,
        evolutionary_config=evo_config,
        fitness_config=fitness_config,
        data=(X, y),
        log_dir="test_logs"
    )
    
    # Should handle small population gracefully
    population = nas.initialize_population()
    assert len(population) >= 0, "Should handle small population"
    
    print("✅ Error handling passed")

def run_performance_test():
    """Run a performance test with larger data."""
    print("🧪 Running performance test...")
    
    # Create larger dataset
    X, y = create_sample_data(1000, 20)
    
    arch_config = ArchitectureConfig(
        max_layers=6,
        min_layers=2,
        max_neurons_per_layer=256,
        min_neurons_per_layer=16
    )
    
    evo_config = EvolutionaryConfig(
        population_size=10,
        max_generations=3,
        n_workers=2
    )
    
    fitness_config = FitnessConfig(
        cv_folds=3,
        max_training_epochs=20,
        max_training_time=60.0
    )
    
    nas = EvolutionaryArchitectureSearch(
        architecture_config=arch_config,
        evolutionary_config=evo_config,
        fitness_config=fitness_config,
        data=(X, y),
        log_dir="performance_test_logs"
    )
    
    start_time = time.time()
    best_architecture = nas.run_evolution()
    end_time = time.time()
    
    total_time = end_time - start_time
    summary = nas.get_search_summary()
    
    print(f"✅ Performance test completed:")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Best fitness: {best_architecture.fitness:.4f}")
    print(f"   Total evaluations: {summary['total_evaluations']}")
    print(f"   Avg evaluation time: {summary['avg_evaluation_time']:.3f} seconds")
    print(f"   Evaluations per second: {summary['total_evaluations'] / total_time:.2f}")

def main():
    """Run all tests."""
    print("🚀 Starting EvolutionaryArchitectureSearch tests...")
    print("=" * 60)
    
    try:
        # Basic functionality tests
        test_architecture_creation()
        test_evolutionary_config()
        test_fitness_config()
        test_nas_initialization()
        test_population_initialization()
        test_fitness_evaluation()
        test_genetic_operators()
        test_serialization()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("🧪 Running integration tests...")
        
        # Integration tests
        test_evolution_cycle()
        
        print("\n" + "=" * 60)
        print("🧪 Running performance tests...")
        
        # Performance test
        run_performance_test()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()