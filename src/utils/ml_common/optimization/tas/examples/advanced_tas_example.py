"""
Advanced TAS Example

Comprehensive example demonstrating the advanced Tree Architecture Search system
with meta-learning, hardware optimization, uncertainty estimation, regime analysis,
and real-time adaptation capabilities.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import TAS components
from src.utils.ml_common.optimization.tas import (
    TreeArchitectureSearchEngine, TASEngineConfig, 
    SearchStrategy, OptimizationMode
)
from src.utils.ml_common.optimization.tas.meta_learning import (
    TreeMetaLearning, MetaLearningConfig
)
from src.utils.ml_common.optimization.tas.regime_analysis import (
    TreeRegimeAnalyzer
)
from src.utils.ml_common.optimization.tas.utils import (
    TreeVisualizer, TreeLogger
)


def create_sample_data(n_samples: int = 1000, n_features: int = 20, 
                      task_type: str = "classification") -> Tuple[np.ndarray, np.ndarray]:
    """Create sample data for demonstration."""
    if task_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            noise=0.1,
            random_state=42
        )
    
    return X, y


def create_regime_data(n_samples: int = 2000, n_regimes: int = 3) -> Dict[str, Any]:
    """Create regime-aware data for demonstration."""
    # Create different regimes with different characteristics
    regimes = []
    regime_labels = []
    
    for regime_id in range(n_regimes):
        # Create regime-specific data
        if regime_id == 0:  # High volatility regime
            X_regime, y_regime = make_classification(
                n_samples=n_samples // n_regimes,
                n_features=20,
                n_informative=10,
                n_redundant=10,
                n_classes=2,
                class_sep=0.5,  # Lower separation for high volatility
                random_state=42 + regime_id
            )
        elif regime_id == 1:  # Low volatility regime
            X_regime, y_regime = make_classification(
                n_samples=n_samples // n_regimes,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                n_classes=2,
                class_sep=2.0,  # Higher separation for low volatility
                random_state=42 + regime_id
            )
        else:  # Trending regime
            X_regime, y_regime = make_classification(
                n_samples=n_samples // n_regimes,
                n_features=20,
                n_informative=12,
                n_redundant=8,
                n_classes=2,
                class_sep=1.0,  # Medium separation for trending
                random_state=42 + regime_id
            )
        
        regimes.append(X_regime)
        regime_labels.extend([regime_id] * len(X_regime))
    
    # Combine all regimes
    X_combined = np.vstack(regimes)
    y_combined = np.hstack([regime[1] for regime in zip(regimes, [make_classification(
        n_samples=len(regime), n_features=20, n_informative=10, n_redundant=10, n_classes=2, random_state=42
    )[1] for regime in regimes])])
    
    return {
        'X': X_combined,
        'y': y_combined,
        'regime_labels': np.array(regime_labels),
        'regime_characteristics': {
            'volatility': [0.8, 0.2, 0.5],  # High, Low, Medium
            'trend_strength': [0.2, 0.1, 0.7],  # Low, Low, High
            'regime_duration': [500, 800, 700]  # Different durations
        }
    }


def demonstrate_basic_tas():
    """Demonstrate basic TAS functionality."""
    logger.info("🚀 Demonstrating Basic TAS")
    
    # Create sample data
    X, y = create_sample_data(n_samples=1000, n_features=20, task_type="classification")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create basic TAS configuration
    config = TASEngineConfig(
        search_strategy=SearchStrategy.BAYESIAN,
        optimization_mode=OptimizationMode.SINGLE_OBJECTIVE,
        enable_meta_learning=False,
        enable_hardware_optimization=False,
        enable_uncertainty_estimation=False,
        enable_regime_analysis=False,
        enable_real_time_adaptation=False,
        max_search_time=300,  # 5 minutes
        max_evaluations=50,
        verbose=True
    )
    
    # Create TAS engine
    engine = TreeArchitectureSearchEngine(config)
    
    # Perform search
    logger.info("🔍 Starting basic TAS search...")
    result = engine.search(
        train_data=(X_train, y_train),
        validation_data=(X_val, y_val),
        test_data=(X_test, y_test)
    )
    
    # Display results
    logger.info(f"✅ Basic TAS completed")
    logger.info(f"🏆 Best architecture: {result.best_architecture}")
    logger.info(f"🎯 Best score: {result.best_score:.4f}")
    logger.info(f"⏱️ Execution time: {result.execution_time:.2f}s")
    logger.info(f"🔢 Evaluations: {result.n_evaluations}")
    
    return result


def demonstrate_advanced_tas():
    """Demonstrate advanced TAS with all features enabled."""
    logger.info("🚀 Demonstrating Advanced TAS")
    
    # Create sample data
    X, y = create_sample_data(n_samples=2000, n_features=30, task_type="classification")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create advanced TAS configuration
    config = TASEngineConfig(
        search_strategy=SearchStrategy.HYBRID,
        optimization_mode=OptimizationMode.REGIME_AWARE,
        enable_meta_learning=True,
        enable_hardware_optimization=True,
        enable_uncertainty_estimation=True,
        enable_regime_analysis=True,
        enable_real_time_adaptation=True,
        enable_continual_learning=True,
        max_search_time=600,  # 10 minutes
        max_evaluations=100,
        parallel_evaluations=4,
        memory_limit_gb=4.0,
        verbose=True
    )
    
    # Create TAS engine
    engine = TreeArchitectureSearchEngine(config)
    
    # Perform advanced search
    logger.info("🔍 Starting advanced TAS search...")
    result = engine.search(
        train_data=(X_train, y_train),
        validation_data=(X_val, y_val),
        test_data=(X_test, y_test)
    )
    
    # Display results
    logger.info(f"✅ Advanced TAS completed")
    logger.info(f"🏆 Best architecture: {result.best_architecture}")
    logger.info(f"🎯 Best score: {result.best_score:.4f}")
    logger.info(f"⏱️ Execution time: {result.execution_time:.2f}s")
    logger.info(f"🔢 Evaluations: {result.n_evaluations}")
    
    # Display advanced features
    if result.uncertainty_estimates:
        logger.info(f"🎲 Uncertainty estimates: {result.uncertainty_estimates}")
    
    if result.regime_analysis:
        logger.info(f"📊 Regime analysis: {result.regime_analysis}")
    
    return result


def demonstrate_regime_aware_tas():
    """Demonstrate regime-aware TAS functionality."""
    logger.info("🚀 Demonstrating Regime-Aware TAS")
    
    # Create regime data
    regime_data = create_regime_data(n_samples=3000, n_regimes=3)
    X, y = regime_data['X'], regime_data['y']
    regime_labels = regime_data['regime_labels']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Split regime labels accordingly
    train_regime_labels = regime_labels[:len(X_train)]
    val_regime_labels = regime_labels[len(X_train):len(X_train) + len(X_val)]
    test_regime_labels = regime_labels[len(X_train) + len(X_val):]
    
    # Create regime-aware TAS configuration
    config = TASEngineConfig(
        search_strategy=SearchStrategy.BAYESIAN,
        optimization_mode=OptimizationMode.REGIME_AWARE,
        enable_regime_analysis=True,
        enable_uncertainty_estimation=True,
        max_search_time=600,
        max_evaluations=100,
        verbose=True
    )
    
    # Create TAS engine
    engine = TreeArchitectureSearchEngine(config)
    
    # Prepare regime data
    regime_info = {
        'regime_labels': {
            'train': train_regime_labels,
            'val': val_regime_labels,
            'test': test_regime_labels
        },
        'regime_characteristics': regime_data['regime_characteristics']
    }
    
    # Perform regime-aware search
    logger.info("🔍 Starting regime-aware TAS search...")
    result = engine.search(
        train_data=(X_train, y_train),
        validation_data=(X_val, y_val),
        test_data=(X_test, y_test),
        regime_data=regime_info
    )
    
    # Display results
    logger.info(f"✅ Regime-aware TAS completed")
    logger.info(f"🏆 Best architecture: {result.best_architecture}")
    logger.info(f"🎯 Best score: {result.best_score:.4f}")
    logger.info(f"⏱️ Execution time: {result.execution_time:.2f}s")
    
    # Display regime analysis
    if result.regime_analysis:
        logger.info(f"📊 Regime analysis results:")
        for regime_id, analysis in result.regime_analysis.items():
            logger.info(f"   Regime {regime_id}: {analysis}")
    
    return result


def demonstrate_meta_learning():
    """Demonstrate meta-learning capabilities."""
    logger.info("🚀 Demonstrating Meta-Learning TAS")
    
    # Create multiple tasks for meta-learning
    meta_train_tasks = []
    meta_val_tasks = []
    
    for task_id in range(5):
        # Create different tasks with different characteristics
        X, y = create_sample_data(
            n_samples=500, 
            n_features=20, 
            task_type="classification"
        )
        
        # Split into support and query sets
        X_support, X_query, y_support, y_query = train_test_split(
            X, y, test_size=0.3, random_state=42 + task_id
        )
        
        task = {
            'task_id': task_id,
            'task_type': 'classification',
            'support_data': (X_support, y_support),
            'query_data': (X_query, y_query)
        }
        
        if task_id < 3:
            meta_train_tasks.append(task)
        else:
            meta_val_tasks.append(task)
    
    # Configure meta-learning
    meta_config = MetaLearningConfig(
        meta_learning_rate=0.001,
        num_inner_steps=5,
        num_outer_steps=50,
        num_shots=5,
        num_ways=3
    )
    
    # Create meta-learner
    meta_learner = TreeMetaLearning(meta_config)
    
    # Meta-train
    logger.info("🧠 Starting meta-training...")
    meta_results = meta_learner.meta_train(meta_train_tasks, meta_val_tasks)
    
    logger.info(f"✅ Meta-training completed")
    logger.info(f"🎯 Final performance: {meta_results['final_performance']:.4f}")
    logger.info(f"⏱️ Execution time: {meta_results['execution_time']:.2f}s")
    
    # Demonstrate few-shot adaptation
    logger.info("🔄 Demonstrating few-shot adaptation...")
    
    # Create new task for adaptation
    X_new, y_new = create_sample_data(n_samples=200, n_features=20, task_type="classification")
    X_support_new, X_query_new, y_support_new, y_query_new = train_test_split(
        X_new, y_new, test_size=0.4, random_state=42
    )
    
    # Perform few-shot adaptation
    adaptation_results = meta_learner.few_shot_adaptation(
        support_data=(X_support_new, y_support_new),
        query_data=(X_query_new, y_query_new),
        adaptation_method="maml"
    )
    
    logger.info(f"✅ Few-shot adaptation completed")
    logger.info(f"🎯 Adaptation results: {adaptation_results}")
    
    return meta_results, adaptation_results


def demonstrate_real_time_adaptation():
    """Demonstrate real-time adaptation capabilities."""
    logger.info("🚀 Demonstrating Real-Time Adaptation TAS")
    
    # Create initial data
    X, y = create_sample_data(n_samples=1000, n_features=20, task_type="classification")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create real-time TAS configuration
    config = TASEngineConfig(
        search_strategy=SearchStrategy.BAYESIAN,
        optimization_mode=OptimizationMode.REAL_TIME,
        enable_real_time_adaptation=True,
        enable_uncertainty_estimation=True,
        max_search_time=300,
        max_evaluations=50,
        verbose=True
    )
    
    # Create TAS engine
    engine = TreeArchitectureSearchEngine(config)
    
    # Initial search
    logger.info("🔍 Starting initial TAS search...")
    result = engine.search(
        train_data=(X_train, y_train),
        validation_data=(X_val, y_val),
        test_data=(X_test, y_test)
    )
    
    logger.info(f"✅ Initial search completed")
    logger.info(f"🏆 Best architecture: {result.best_architecture}")
    logger.info(f"🎯 Best score: {result.best_score:.4f}")
    
    # Simulate real-time adaptation
    logger.info("🔄 Demonstrating real-time adaptation...")
    
    # Create new data (simulating new market conditions)
    X_new, y_new = create_sample_data(n_samples=200, n_features=20, task_type="classification")
    
    # Adapt to new data
    adapted_architecture = engine.adapt_to_new_data(
        new_data=(X_new, y_new),
        current_architecture=result.best_architecture
    )
    
    logger.info(f"✅ Real-time adaptation completed")
    logger.info(f"🔄 Adapted architecture: {adapted_architecture}")
    logger.info(f"🎯 Adapted score: {adapted_architecture.overall_score:.4f}")
    
    return result, adapted_architecture


def demonstrate_visualization():
    """Demonstrate visualization capabilities."""
    logger.info("🚀 Demonstrating TAS Visualization")
    
    # Create sample data
    X, y = create_sample_data(n_samples=1000, n_features=20, task_type="classification")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create TAS configuration
    config = TASEngineConfig(
        search_strategy=SearchStrategy.BAYESIAN,
        optimization_mode=OptimizationMode.SINGLE_OBJECTIVE,
        max_search_time=300,
        max_evaluations=50,
        verbose=True
    )
    
    # Create TAS engine
    engine = TreeArchitectureSearchEngine(config)
    
    # Perform search
    result = engine.search(
        train_data=(X_train, y_train),
        validation_data=(X_val, y_val),
        test_data=(X_test, y_test)
    )
    
    # Create visualizer
    visualizer = TreeVisualizer()
    
    # Visualize search progress
    logger.info("📊 Creating visualizations...")
    
    try:
        # Plot search progress
        visualizer.plot_search_progress(result.search_history)
        plt.title("TAS Search Progress")
        plt.show()
        
        # Plot architecture comparison
        if len(result.search_history) > 1:
            architectures = [entry['architecture'] for entry in result.search_history[-5:]]
            visualizer.plot_architecture_comparison(architectures)
            plt.title("Architecture Comparison")
            plt.show()
        
        logger.info("✅ Visualizations created successfully")
        
    except Exception as e:
        logger.warning(f"⚠️ Visualization failed: {e}")
    
    return result


def main():
    """Main demonstration function."""
    logger.info("🚀 Starting Advanced TAS Demonstration")
    logger.info("=" * 60)
    
    try:
        # 1. Basic TAS
        logger.info("\n1. Basic TAS Demonstration")
        logger.info("-" * 40)
        basic_result = demonstrate_basic_tas()
        
        # 2. Advanced TAS
        logger.info("\n2. Advanced TAS Demonstration")
        logger.info("-" * 40)
        advanced_result = demonstrate_advanced_tas()
        
        # 3. Regime-Aware TAS
        logger.info("\n3. Regime-Aware TAS Demonstration")
        logger.info("-" * 40)
        regime_result = demonstrate_regime_aware_tas()
        
        # 4. Meta-Learning
        logger.info("\n4. Meta-Learning Demonstration")
        logger.info("-" * 40)
        meta_results, adaptation_results = demonstrate_meta_learning()
        
        # 5. Real-Time Adaptation
        logger.info("\n5. Real-Time Adaptation Demonstration")
        logger.info("-" * 40)
        real_time_result, adapted_architecture = demonstrate_real_time_adaptation()
        
        # 6. Visualization
        logger.info("\n6. Visualization Demonstration")
        logger.info("-" * 40)
        viz_result = demonstrate_visualization()
        
        # Summary
        logger.info("\n📊 Demonstration Summary")
        logger.info("=" * 60)
        logger.info(f"✅ Basic TAS: {basic_result.best_score:.4f} (Score)")
        logger.info(f"✅ Advanced TAS: {advanced_result.best_score:.4f} (Score)")
        logger.info(f"✅ Regime-Aware TAS: {regime_result.best_score:.4f} (Score)")
        logger.info(f"✅ Meta-Learning: {meta_results['final_performance']:.4f} (Performance)")
        logger.info(f"✅ Real-Time Adaptation: {adapted_architecture.overall_score:.4f} (Score)")
        logger.info(f"✅ Visualization: Completed")
        
        logger.info("\n🎉 Advanced TAS Demonstration Completed Successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()