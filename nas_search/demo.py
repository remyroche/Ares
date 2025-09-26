#!/usr/bin/env python3
"""
Demonstration script for EvolutionaryArchitectureSearch.

This script shows how to use the evolutionary architecture search
for neural architecture optimization.
"""

import sys
import time
import random
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from evolutionary_search import (
    EvolutionaryArchitectureSearch,
    ArchitectureConfig,
    EvolutionaryConfig,
    FitnessConfig
)

def create_sample_data(n_samples=1000, n_features=20):
    """Create sample data for demonstration."""
    # Generate realistic mock data with proper statistical properties
    print(f"📊 Generating {n_samples} samples with {n_features} features...")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Generate features with realistic distributions
    X = []
    for i in range(n_samples):
        sample = []
        for j in range(n_features):
            # Mix of different distributions for more realistic data
            if j % 4 == 0:
                # Normal distribution
                sample.append(random.gauss(0, 1))
            elif j % 4 == 1:
                # Uniform distribution
                sample.append(random.uniform(-2, 2))
            elif j % 4 == 2:
                # Exponential distribution
                sample.append(random.expovariate(1.0))
            else:
                # Beta distribution (approximation)
                x = random.gammavariate(2, 1.0)
                y = random.gammavariate(2, 1.0)
                sample.append(x / (x + y) * 4 - 2)  # Scale to [-2, 2]
        X.append(sample)
    
    # Generate target variable with some correlation to features
    y = []
    for i, sample in enumerate(X):
        # Create target based on weighted combination of features
        target_score = 0
        for j, feature in enumerate(sample):
            weight = 1.0 / (j + 1)  # Decreasing weights
            target_score += weight * feature
        
        # Add some noise
        noise = random.gauss(0, 0.5)
        target_score += noise
        
        # Convert to binary classification
        y.append(1 if target_score > 0 else 0)
    
    # Ensure balanced classes
    class_counts = [y.count(0), y.count(1)]
    if abs(class_counts[0] - class_counts[1]) > n_samples * 0.1:  # If imbalance > 10%
        print(f"⚠️  Class imbalance detected: {class_counts[0]} vs {class_counts[1]}")
        # Adjust threshold to balance classes
        sorted_indices = sorted(range(len(y)), key=lambda i: sum(X[i]))
        target_balance = n_samples // 2
        for i in range(n_samples):
            y[i] = 1 if i >= target_balance else 0
    
    print(f"✅ Generated data with class distribution: {[y.count(0), y.count(1)]}")
    
    # Add some missing values for realism (5% of data)
    missing_count = int(n_samples * n_features * 0.05)
    for _ in range(missing_count):
        row_idx = random.randint(0, n_samples - 1)
        col_idx = random.randint(0, n_features - 1)
        X[row_idx][col_idx] = None
    
    print(f"📊 Added {missing_count} missing values for realism")
    
    return X, y

def demonstrate_basic_usage():
    """Demonstrate basic usage of the evolutionary search."""
    print("🚀 Evolutionary Architecture Search Demo")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample dataset...")
    X, y = create_sample_data(500, 15)
    print(f"   Dataset size: {len(X)} samples, {len(X[0])} features")
    
    # Configure architecture constraints
    print("\n🏗️ Configuring architecture constraints...")
    arch_config = ArchitectureConfig(
        max_layers=6,
        min_layers=2,
        max_neurons_per_layer=256,
        min_neurons_per_layer=16,
        max_parameters=50000,
        min_parameters=100,
        max_flops=1000000,
        min_flops=1000
    )
    print(f"   Max layers: {arch_config.max_layers}")
    print(f"   Max neurons per layer: {arch_config.max_neurons_per_layer}")
    print(f"   Max parameters: {arch_config.max_parameters}")
    
    # Configure evolutionary algorithm
    print("\n🧬 Configuring evolutionary algorithm...")
    evo_config = EvolutionaryConfig(
        population_size=20,
        max_generations=10,
        elite_size=3,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=0.2,
        n_workers=2
    )
    print(f"   Population size: {evo_config.population_size}")
    print(f"   Max generations: {evo_config.max_generations}")
    print(f"   Crossover rate: {evo_config.crossover_rate}")
    print(f"   Mutation rate: {evo_config.mutation_rate}")
    
    # Configure fitness evaluation
    print("\n📈 Configuring fitness evaluation...")
    fitness_config = FitnessConfig(
        primary_metric='accuracy',
        cv_folds=3,
        max_training_epochs=50,
        max_training_time=30.0,
        max_memory_usage=4.0
    )
    print(f"   Primary metric: {fitness_config.primary_metric}")
    print(f"   CV folds: {fitness_config.cv_folds}")
    print(f"   Max training time: {fitness_config.max_training_time}s")
    
    # Initialize search
    print("\n🔧 Initializing evolutionary search...")
    nas = EvolutionaryArchitectureSearch(
        architecture_config=arch_config,
        evolutionary_config=evo_config,
        fitness_config=fitness_config,
        data=(X, y),
        log_dir="demo_results"
    )
    print("   ✅ Search initialized successfully")
    
    # Run evolution
    print("\n🚀 Starting evolutionary search...")
    start_time = time.time()
    
    try:
        best_architecture = nas.run_evolution()
        end_time = time.time()
        
        if best_architecture:
            print(f"\n🏆 Search completed successfully!")
            print(f"   Total time: {end_time - start_time:.2f} seconds")
            print(f"   Best fitness: {best_architecture.fitness:.4f}")
            print(f"   Architecture layers: {len(best_architecture.layers)}")
            print(f"   Parameters: {best_architecture.parameters_count}")
            print(f"   FLOPs: {best_architecture.flops_count}")
            print(f"   Training time: {best_architecture.training_time:.3f}s")
            print(f"   Memory usage: {best_architecture.memory_usage:.2f} GB")
            
            # Show architecture details
            print(f"\n🏗️ Best architecture details:")
            for i, layer in enumerate(best_architecture.layers):
                print(f"   Layer {i+1}: {layer['type']} with {layer['neurons']} neurons, activation: {layer['activation']}")
            
            # Show search summary
            summary = nas.get_search_summary()
            print(f"\n📊 Search summary:")
            print(f"   Total generations: {summary['total_generations']}")
            print(f"   Total evaluations: {summary['total_evaluations']}")
            print(f"   Average fitness: {summary['avg_fitness']:.4f}")
            print(f"   Final diversity: {summary['final_diversity']:.4f}")
            print(f"   Average evaluation time: {summary['avg_evaluation_time']:.3f}s")
            
        else:
            print("❌ Search failed - no valid architectures found")
            
    except Exception as e:
        print(f"❌ Search failed with error: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_advanced_features():
    """Demonstrate advanced features."""
    print("\n" + "=" * 50)
    print("🔬 Advanced Features Demo")
    print("=" * 50)
    
    # Create data
    X, y = create_sample_data(200, 10)
    
    # Advanced configuration
    arch_config = ArchitectureConfig(
        max_layers=8,
        min_layers=3,
        max_neurons_per_layer=512,
        min_neurons_per_layer=32,
        layer_types=['dense', 'conv1d', 'lstm', 'dropout', 'batch_norm'],
        activation_functions=['relu', 'tanh', 'sigmoid', 'gelu', 'swish'],
        max_parameters=100000,
        min_parameters=1000,
        max_flops=5000000,
        min_flops=10000
    )
    
    evo_config = EvolutionaryConfig(
        population_size=15,
        max_generations=5,
        elite_size=2,
        tournament_size=4,
        crossover_rate=0.9,
        mutation_rate=0.3,
        mutation_strength=0.2,
        selection_pressure=2.5,
        diversity_weight=0.2,
        early_stopping_patience=3,
        convergence_threshold=1e-4,
        n_workers=2,
        use_parallel_evaluation=True
    )
    
    fitness_config = FitnessConfig(
        primary_metric='accuracy',
        secondary_metrics=['precision', 'recall', 'f1_score'],
        cv_folds=5,
        use_stratified_cv=True,
        max_training_epochs=100,
        early_stopping_patience=15,
        learning_rate=0.001,
        batch_size=32,
        max_training_time=60.0,
        max_memory_usage=8.0,
        min_accuracy_threshold=0.6
    )
    
    print("🔧 Advanced configuration:")
    print(f"   Layer types: {arch_config.layer_types}")
    print(f"   Activation functions: {arch_config.activation_functions}")
    print(f"   Selection pressure: {evo_config.selection_pressure}")
    print(f"   Diversity weight: {evo_config.diversity_weight}")
    print(f"   Secondary metrics: {fitness_config.secondary_metrics}")
    
    # Initialize search
    nas = EvolutionaryArchitectureSearch(
        architecture_config=arch_config,
        evolutionary_config=evo_config,
        fitness_config=fitness_config,
        data=(X, y),
        log_dir="advanced_demo_results"
    )
    
    print("\n🚀 Running advanced search...")
    start_time = time.time()
    
    try:
        best_architecture = nas.run_evolution()
        end_time = time.time()
        
        if best_architecture:
            print(f"\n🏆 Advanced search completed!")
            print(f"   Time: {end_time - start_time:.2f}s")
            print(f"   Best fitness: {best_architecture.fitness:.4f}")
            print(f"   Architecture complexity: {best_architecture.parameters_count} parameters")
            
            # Show validation metrics
            if best_architecture.validation_metrics:
                print(f"\n📊 Validation metrics:")
                for metric, value in best_architecture.validation_metrics.items():
                    print(f"   {metric}: {value:.4f}")
        else:
            print("❌ Advanced search failed")
            
    except Exception as e:
        print(f"❌ Advanced search failed: {e}")

def main():
    """Run the demonstration."""
    print("🧬 Evolutionary Architecture Search (EAS) Demonstration")
    print("=" * 60)
    print("This demo shows how to use the evolutionary architecture search")
    print("for neural architecture optimization.")
    print("=" * 60)
    
    try:
        # Basic demonstration
        demonstrate_basic_usage()
        
        # Advanced features demonstration
        demonstrate_advanced_features()
        
        print("\n" + "=" * 60)
        print("✅ Demonstration completed successfully!")
        print("\nKey features demonstrated:")
        print("  🧬 Evolutionary algorithm with genetic operators")
        print("  🏗️ Architecture constraints and validation")
        print("  📊 Fitness evaluation with multiple metrics")
        print("  🔧 Hardware optimization and parallel processing")
        print("  📈 Progress tracking and logging")
        print("  💾 Result serialization and persistence")
        print("  🛡️ Error handling and graceful degradation")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()