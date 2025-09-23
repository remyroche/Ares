#!/usr/bin/env python3
"""
Advanced Tree Architecture Search (TAS) Demo

This script demonstrates the advanced capabilities of the enhanced TAS system
including Bayesian optimization, meta-learning, and hierarchical architectures.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path

# Import our enhanced Tree Architecture Search
from src.utils.ml_common.optimization.tree_architecture_search import (
    TreeArchitectureSearch,
    TreeArchitectureConfig,
    TreeArchitectureCandidate,
    search_tree_architecture
)


def create_complex_dataset(n_samples: int = 2000, n_features: int = 50, problem_type: str = 'mixed'):
    """Create a complex dataset for advanced TAS testing."""
    if problem_type == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            n_redundant=n_features // 4,
            n_clusters_per_class=2,
            n_classes=5,
            flip_y=0.05,  # Add noise
            class_sep=1.0,
            random_state=42
        )
    elif problem_type == 'regression':
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            noise=0.2,
            random_state=42
        )
        # Add non-linear relationships
        y = y + 0.1 * y**2 + 0.05 * np.sin(y)
    else:  # mixed - create multiple datasets
        datasets = []
        for i in range(3):
            X, y = make_classification(
                n_samples=n_samples // 3,
                n_features=n_features,
                n_informative=n_features // 2,
                n_redundant=n_features // 4,
                n_classes=3,
                random_state=42 + i
            )
            datasets.append((X, y))
        X = np.vstack([d[0] for d in datasets])
        y = np.hstack([d[1] for d in datasets])

    return X, y


def create_meta_learning_data():
    """Create sample meta-learning data for demonstration."""
    sample_data = [
        {
            'dataset_id': 'synthetic_1',
            'meta_features': {
                'n_samples': 1000,
                'n_features': 20,
                'n_classes': 3,
                'feature_noise': 1.2,
                'target_entropy': 1.1,
                'feature_correlation': 0.3
            },
            'best_architecture': {
                'n_trees': 150,
                'max_depth': 12,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'splitting_strategy': 'gini'
            },
            'score': 0.92,
            'training_time': 45.2,
            'model_size_mb': 2.1
        },
        {
            'dataset_id': 'synthetic_2',
            'meta_features': {
                'n_samples': 5000,
                'n_features': 100,
                'n_classes': 2,
                'feature_noise': 0.8,
                'target_entropy': 0.6,
                'feature_correlation': 0.7
            },
            'best_architecture': {
                'n_trees': 300,
                'max_depth': 8,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'log2',
                'splitting_strategy': 'entropy'
            },
            'score': 0.88,
            'training_time': 120.5,
            'model_size_mb': 8.3
        }
    ]

    # Save to temporary file
    meta_path = Path("/tmp/tas_meta_learning.json")
    with open(meta_path, 'w') as f:
        json.dump(sample_data, f, indent=2)

    return str(meta_path)


def demonstrate_advanced_tas():
    """Demonstrate advanced TAS capabilities."""
    print("🌲 Advanced Tree Architecture Search (TAS) Demo")
    print("=" * 60)

    # Create complex dataset
    print("\n1. Creating complex multi-class dataset...")
    X, y = create_complex_dataset(n_samples=3000, n_features=50, problem_type='classification')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"   Training set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    print(f"   Classes: {np.unique(y)}")
    print(f"   Features: {X_train.shape[1]}")

    # Create meta-learning data
    meta_path = create_meta_learning_data()
    print(f"   Meta-learning data: {len(json.load(open(meta_path)))} examples")

    # Test different search methods
    search_methods = ['evolutionary', 'bayesian', 'meta_learning', 'hierarchical', 'hybrid']
    results = {}

    for method in search_methods:
        print(f"\n2. Running {method.upper()} search...")

        # Configure TAS based on method
        if method == 'bayesian':
            config = TreeArchitectureConfig(
                n_bayesian_iterations=20,
                n_initial_points=8,
                acquisition_function='EI',
                xi=0.01,
                enable_meta_learning=False
            )
        elif method == 'hierarchical':
            config = TreeArchitectureConfig(
                n_trials=15,
                enable_meta_learning=False
            )
        elif method == 'meta_learning':
            config = TreeArchitectureConfig(
                n_trials=10,
                meta_learning_path=meta_path,
                enable_meta_learning=True
            )
        else:  # evolutionary and hybrid
            config = TreeArchitectureConfig(
                n_trials=15,
                enable_meta_learning=True,
                meta_learning_path=meta_path
            )

        # Run search
        start_time = datetime.now()
        best_architecture = search_tree_architecture(
            X_train, y_train, X_val, y_val,
            config=config,
            search_method=method
        )
        search_time = (datetime.now() - start_time).total_seconds()

        # Store results
        results[method] = {
            'architecture': best_architecture,
            'search_time': search_time
        }

        print(f"   ✅ {method.upper()} completed in {search_time:.1f} seconds")
        print(f"   📊 Score: {best_architecture.overall_score:.4f}, Accuracy: {best_architecture.accuracy:.4f}")
        print(f"   🏗️ Method: {best_architecture.search_method}")
        print(f"   🌳 Trees: {best_architecture.n_trees}, Depth: {best_architecture.max_depth}")
        if best_architecture.is_hierarchical:
            print(f"   📈 Ensemble Type: {best_architecture.ensemble_type}")

    # Compare results
    print("\n3. Comparing Search Methods")
    print("-" * 40)

    comparison_data = []
    for method, result in results.items():
        arch = result['architecture']
        comparison_data.append({
            'Method': method.upper(),
            'Overall Score': arch.overall_score,
            'Accuracy': arch.accuracy,
            'Efficiency': arch.efficiency_score,
            'Interpretability': arch.interpretability_score,
            'Search Time (s)': result['search_time'],
            'Training Time (s)': arch.training_time,
            'Model Size (MB)': arch.model_size_mb,
            'Architecture Type': arch.search_method,
            'Is Hierarchical': arch.is_hierarchical,
            'Ensemble Type': arch.ensemble_type if arch.is_hierarchical else 'Single'
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.round(4).to_string(index=False))

    # Find best method
    best_method = comparison_df.loc[comparison_df['Overall Score'].idxmax(), 'Method']
    best_score = comparison_df['Overall Score'].max()

    print(f"\n🏆 Best Method: {best_method} (Score: {best_score:.4f})")

    # Visualize comparison
    print("\n4. Generating Comparison Visualizations...")
    plt.figure(figsize=(15, 10))

    # Overall score comparison
    plt.subplot(2, 3, 1)
    methods = comparison_df['Method']
    scores = comparison_df['Overall Score']
    plt.bar(methods, scores, color='skyblue')
    plt.title('Overall Score by Method')
    plt.ylabel('Score')
    plt.xticks(rotation=45)

    # Accuracy comparison
    plt.subplot(2, 3, 2)
    plt.bar(methods, comparison_df['Accuracy'], color='lightgreen')
    plt.title('Accuracy by Method')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)

    # Search time comparison
    plt.subplot(2, 3, 3)
    plt.bar(methods, comparison_df['Search Time (s)'], color='orange')
    plt.title('Search Time by Method')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)

    # Training time vs Model size
    plt.subplot(2, 3, 4)
    plt.scatter(comparison_df['Training Time (s)'], comparison_df['Model Size (MB)'],
                s=comparison_df['Overall Score']*200, alpha=0.6)
    for i, method in enumerate(methods):
        plt.annotate(method, (comparison_df.iloc[i]['Training Time (s)'],
                             comparison_df.iloc[i]['Model Size (MB)']))
    plt.xlabel('Training Time (s)')
    plt.ylabel('Model Size (MB)')
    plt.title('Training Time vs Model Size')

    # Multi-objective radar chart
    plt.subplot(2, 3, 5, projection='polar')
    objectives = ['Accuracy', 'Efficiency', 'Interpretability']
    angles = np.linspace(0, 2*np.pi, len(objectives), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    for i, (_, row) in enumerate(comparison_df.iterrows()):
        values = [row[obj] for obj in objectives]
        values += values[:1]  # Complete the loop
        plt.polar(angles, values, label=row['Method'])

    plt.thetagrids(np.degrees(angles[:-1]), objectives)
    plt.title('Multi-Objective Comparison')
    plt.legend(bbox_to_anchor=(1.3, 1))

    # Hierarchical vs Single model comparison
    plt.subplot(2, 3, 6)
    hierarchical_data = comparison_df[comparison_df['Is Hierarchical']]
    single_data = comparison_df[~comparison_df['Is Hierarchical']]

    if len(hierarchical_data) > 0:
        plt.bar(['Single Models', 'Hierarchical'], [
            single_data['Overall Score'].mean() if len(single_data) > 0 else 0,
            hierarchical_data['Overall Score'].mean()
        ], color=['lightcoral', 'lightblue'])
        plt.title('Single vs Hierarchical Models')
        plt.ylabel('Average Score')

    plt.tight_layout()
    plt.savefig('/workspace/tas_advanced_comparison.png', dpi=300, bbox_inches='tight')
    print("   📊 Visualization saved to: /workspace/tas_advanced_comparison.png")

    return results


def demonstrate_hierarchical_ensembles():
    """Demonstrate hierarchical ensemble capabilities."""
    print("\n" + "=" * 60)
    print("🏗️ Hierarchical Ensemble Deep Dive")
    print("=" * 60)

    # Create high-dimensional data for parallel ensemble
    print("\n1. Testing Parallel Ensemble (High-Dimensional Data)...")
    X_high, y_high = create_complex_dataset(n_samples=1000, n_features=100, problem_type='classification')
    X_train_h, X_val_h, y_train_h, y_val_h = train_test_split(X_high, y_high, test_size=0.2, random_state=42)

    config_parallel = TreeArchitectureConfig(
        n_trials=10,
        enable_meta_learning=False
    )

    parallel_arch = search_tree_architecture(
        X_train_h, y_train_h, X_val_h, y_val_h,
        config=config_parallel,
        search_method="hierarchical"
    )

    print("   🏗️ Parallel Ensemble Results:")
    print(f"   - Ensemble Type: {parallel_arch.ensemble_type}")
    print(f"   - Hierarchy Levels: {len(parallel_arch.hierarchy_levels)}")
    print(f"   - Overall Score: {parallel_arch.overall_score:.4f}, Accuracy: {parallel_arch.accuracy:.4f}")
    print("   - Hierarchy Structure:")
    for level in parallel_arch.hierarchy_levels:
        print(f"     Level {level['level']}: {level['model_type']} ({level.get('n_models', 'N/A')} models)")

    # Create multi-class data for adaptive ensemble
    print("\n2. Testing Adaptive Ensemble (Multi-Class Data)...")
    X_multi, y_multi = create_complex_dataset(n_samples=1500, n_features=30, problem_type='mixed')
    X_train_m, X_val_m, y_train_m, y_val_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

    config_adaptive = TreeArchitectureConfig(
        n_trials=10,
        enable_meta_learning=False
    )

    adaptive_arch = search_tree_architecture(
        X_train_m, y_train_m, X_val_m, y_val_m,
        config=config_adaptive,
        search_method="hierarchical"
    )

    print("   🧠 Adaptive Ensemble Results:")
    print(f"   - Ensemble Type: {adaptive_arch.ensemble_type}")
    print(f"   - Overall Score: {adaptive_arch.overall_score:.4f}, Accuracy: {adaptive_arch.accuracy:.4f}")
    print("   - Hierarchy Structure:")
    for level in adaptive_arch.hierarchy_levels:
        print(f"     Level {level['level']}: {level['model_type']}")
        if 'specializations' in level:
            print(f"       Specializations: {len(level['specializations'])} types")

    return parallel_arch, adaptive_arch


def demonstrate_bayesian_optimization():
    """Demonstrate Bayesian optimization capabilities."""
    print("\n" + "=" * 60)
    print("🔍 Bayesian Optimization Deep Dive")
    print("=" * 60)

    # Create regression dataset for Bayesian optimization
    print("\n1. Running Bayesian Optimization on Regression Task...")
    X_reg, y_reg = create_complex_dataset(n_samples=1000, n_features=25, problem_type='regression')
    X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # Test different acquisition functions
    acquisition_functions = ['EI', 'UCB', 'PI']
    bayesian_results = {}

    for acq_func in acquisition_functions:
        print(f"   Testing acquisition function: {acq_func}")

        config = TreeArchitectureConfig(
            n_bayesian_iterations=25,
            n_initial_points=10,
            acquisition_function=acq_func,
            xi=0.01,
            enable_meta_learning=False
        )

        start_time = datetime.now()
        best_arch = search_tree_architecture(
            X_train_r, y_train_r, X_val_r, y_val_r,
            config=config,
            search_method="bayesian"
        )
        search_time = (datetime.now() - start_time).total_seconds()

        bayesian_results[acq_func] = {
            'architecture': best_arch,
            'search_time': search_time
        }

        print(f"   ✅ {acq_func}: Score {best_arch.overall_score:.4f}, Time {search_time:.1f} seconds")

    print("\n2. Bayesian Optimization Comparison:")
    for acq_func, result in bayesian_results.items():
        arch = result['architecture']
        print(f"   {acq_func}: Score {arch.overall_score:.4f}, Search time {result['search_time']:.1f} seconds")

    return bayesian_results


def generate_comprehensive_report(results):
    """Generate a comprehensive report of the advanced TAS capabilities."""
    print("\n" + "=" * 60)
    print("📋 Comprehensive TAS Report")
    print("=" * 60)

    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_methods_tested': len(results),
            'best_method': max(results.keys(), key=lambda x: results[x]['architecture'].overall_score),
            'best_score': max([r['architecture'].overall_score for r in results.values()]),
            'fastest_method': min(results.keys(), key=lambda x: results[x]['search_time']),
            'fastest_time': min([r['search_time'] for r in results.values()])
        },
        'method_details': {}
    }

    for method, result in results.items():
        arch = result['architecture']
        report['method_details'][method] = {
            'overall_score': arch.overall_score,
            'accuracy': arch.accuracy,
            'efficiency': arch.efficiency_score,
            'interpretability': arch.interpretability_score,
            'search_time': result['search_time'],
            'training_time': arch.training_time,
            'model_size_mb': arch.model_size_mb,
            'architecture': {
                'n_trees': arch.n_trees,
                'max_depth': arch.max_depth,
                'search_method': arch.search_method,
                'is_hierarchical': arch.is_hierarchical,
                'ensemble_type': arch.ensemble_type if arch.is_hierarchical else 'single'
            }
        }

    # Save report
    with open('/workspace/tas_advanced_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("   📊 Report saved to: /workspace/tas_advanced_report.json")
    print("\n📈 Summary:")
    print(f"   • Best Method: {report['summary']['best_method']}")
    print(f"   • Best Score: {report['summary']['best_score']:.4f}")
    print(f"   • Fastest Method: {report['summary']['fastest_method']}")
    print(f"   • Fastest Time: {report['summary']['fastest_time']:.1f} seconds")

    return report


if __name__ == "__main__":
    try:
        # Run comprehensive demonstration
        print("🚀 Starting Advanced TAS Demonstration...")

        # Main comparison
        results = demonstrate_advanced_tas()

        # Hierarchical ensembles deep dive
        parallel_arch, adaptive_arch = demonstrate_hierarchical_ensembles()

        # Bayesian optimization deep dive
        bayesian_results = demonstrate_bayesian_optimization()

        # Generate comprehensive report
        report = generate_comprehensive_report(results)

        print("\n" + "=" * 60)
        print("✅ Advanced TAS Demo Complete!")
        print("=" * 60)

        print("\n🎯 Key Insights:")
        print("• Hierarchical ensembles excel on complex, high-dimensional data")
        print("• Bayesian optimization provides sample-efficient search")
        print("• Meta-learning accelerates convergence with historical data")
        print("• Multi-objective optimization balances accuracy, efficiency, and interpretability")
        print("• Hybrid approach combines the best of all methods")

        print("\n📁 Files Generated:")
        print("• /workspace/tas_advanced_comparison.png - Visual comparison")
        print("• /workspace/tas_advanced_report.json - Detailed report")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()