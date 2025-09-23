"""
Neural Architecture Search (NAS) Usage Example

This script demonstrates how to use the NAS system for market analysis,
including architecture search, evaluation, and comparison.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import time
from pathlib import Path

from .core.nas_search import NASArchitectureSearch, NASSearchConfig, SearchResult
from .core.nas_model import NASModel, HMM_NAS_Model, Regime_NAS_Model
from .core.nas_trainer import NASTrainer, TrainingConfig
from .core.nas_evaluator import NASEvaluator, EvaluationConfig

from .search.search_space import SearchSpace
from .search.random_search import RandomSearch, RandomSearchConfig
from .search.bayesian_search import BayesianSearch, BayesianSearchConfig

from .evaluation.nas_metrics import NASMetrics, NASMetricsConfig, ArchitectureMetrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 1000,
                      input_dim: int = 100,
                      output_dim: int = 5,
                      problem_type: str = "classification") -> tuple:
    """
    Create sample market data for demonstration.

    Args:
        n_samples: Number of samples
        input_dim: Input dimension
        output_dim: Output dimension
        problem_type: Type of problem

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    np.random.seed(42)

    # Generate synthetic market features
    X = np.random.randn(n_samples, input_dim).astype(np.float32)

    # Add some market-like patterns
    # Trend component
    trend = np.linspace(0, 2, n_samples)[:, np.newaxis]
    # Volatility component
    volatility = np.random.randn(n_samples, 10) * 0.5
    # Volume-like features
    volume = np.random.exponential(1, (n_samples, 5))

    X = np.concatenate([X, trend, volatility, volume], axis=1).astype(np.float32)

    # Generate targets based on problem type
    if problem_type == "classification":
        # Create regime-like labels based on feature patterns
        y = np.zeros(n_samples, dtype=np.int64)
        y[n_samples//3:2*n_samples//3] = 1
        y[2*n_samples//3:] = 2
    elif problem_type == "regression":
        # Create continuous targets
        y = np.sum(X[:, :10], axis=1) + np.random.randn(n_samples) * 0.1
        y = y.astype(np.float32)
    else:
        # Default to classification
        y = np.random.randint(0, output_dim, n_samples, dtype=np.int64)

    # Split data
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    n_test = n_samples - n_train - n_val

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]

    logger.info(f"📊 Created sample data: {n_samples} samples, {input_dim} features, {output_dim} classes")
    logger.info(f"📈 Train: {n_train}, Val: {n_val}, Test: {n_test}")

    return X_train, y_train, X_val, y_val, X_test, y_test

def create_data_loaders(X_train, y_train, X_val, y_val, X_test=None, y_test=None,
                       batch_size: int = 32) -> tuple:
    """
    Create PyTorch data loaders.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data (optional)
        batch_size: Batch size

    Returns:
        Tuple of data loaders
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train) if isinstance(y_train[0], (int, np.integer)) else torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val) if isinstance(y_val[0], (int, np.integer)) else torch.FloatTensor(y_val)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = None
    if X_test is not None and y_test is not None:
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test) if isinstance(y_test[0], (int, np.integer)) else torch.FloatTensor(y_test)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def example_basic_nas_search():
    """Example of basic NAS search."""
    logger.info("🚀 Starting Basic NAS Search Example")

    # Create sample data
    X_train, y_train, X_val, y_val, X_test, y_test = create_sample_data(
        n_samples=2000, input_dim=50, output_dim=3, problem_type="classification"
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64
    )

    # Configure NAS search
    search_config = NASSearchConfig(
        max_iterations=50,  # Small number for demo
        max_time_seconds=600,  # 10 minutes
        search_strategy="random",
        primary_metric="accuracy",
        minimize_metric=False,
        use_gpu=torch.cuda.is_available(),
        batch_size=64
    )

    # Create NAS search engine
    nas_search = NASArchitectureSearch(search_config)

    # Perform architecture search
    search_result = nas_search.search(
        train_data=(X_train, y_train),
        validation_data=(X_val, y_val),
        test_data=(X_test, y_test),
        problem_type="classification",
        input_shape=(64, 50)
    )

    # Display results
    logger.info("🎯 Search Results:")
    logger.info(f"   Best Architecture: {search_result.best_architecture.name}")
    logger.info(f"   Best Score: {search_result.best_score:.4f}")
    logger.info(f"   Execution Time: {search_result.execution_time:.2f}s")
    logger.info(f"   Evaluations: {search_result.n_evaluations}")

    # Save results
    output_dir = Path("nas_results/basic_search")
    nas_search.save_search_results(search_result, str(output_dir))

    return search_result

def example_bayesian_search():
    """Example of Bayesian optimization search."""
    logger.info("🧠 Starting Bayesian Search Example")

    # Create sample data
    X_train, y_train, X_val, y_val, X_test, y_test = create_sample_data(
        n_samples=1500, input_dim=30, output_dim=4, problem_type="classification"
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32
    )

    # Configure Bayesian search
    search_config = NASSearchConfig(
        max_iterations=30,
        search_strategy="bayesian",
        primary_metric="accuracy",
        use_gpu=torch.cuda.is_available(),
        batch_size=32
    )

    # Create Bayesian search
    bayesian_search = NASArchitectureSearch(search_config)

    # Perform search
    search_result = bayesian_search.search(
        train_data=(X_train, y_train),
        validation_data=(X_val, y_val),
        test_data=(X_test, y_test),
        problem_type="classification"
    )

    # Display results
    logger.info("🎯 Bayesian Search Results:")
    logger.info(f"   Best Architecture: {search_result.best_architecture.name}")
    logger.info(f"   Best Score: {search_result.best_score:.4f}")
    logger.info(f"   Search Strategy: {search_result.metadata.get('search_strategy', 'unknown')}")

    return search_result

def example_hmm_nas():
    """Example of NAS for HMM state modeling."""
    logger.info("🔍 Starting HMM NAS Example")

    # Create HMM-specific data
    n_samples = 1000
    n_states = 4
    n_features = 20

    # Generate HMM-like data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Generate state sequence with persistence
    states = np.zeros(n_samples, dtype=np.int64)
    current_state = 0
    state_lengths = np.random.exponential(50, n_samples//10).astype(int)

    idx = 0
    for length in state_lengths:
        if idx + length > n_samples:
            length = n_samples - idx
        states[idx:idx+length] = current_state
        current_state = (current_state + 1) % n_states
        idx += length

    # Add state-specific features
    for state in range(n_states):
        mask = states == state
        X[mask] += np.random.randn(n_features) * 0.5  # State-specific noise

    # Split data
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)

    X_train = X[:n_train]
    y_train = states[:n_train]
    X_val = X[n_train:n_train+n_val]
    y_val = states[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    y_test = states[n_train+n_val:]

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32
    )

    # Configure NAS for HMM
    search_config = NASSearchConfig(
        max_iterations=20,
        search_strategy="random",
        primary_metric="accuracy",
        use_gpu=torch.cuda.is_available(),
        batch_size=32
    )

    # Create NAS search for HMM
    hmm_nas = NASArchitectureSearch(search_config)

    # Perform HMM-specific search
    search_result = hmm_nas.search(
        train_data=(X_train, y_train),
        validation_data=(X_val, y_val),
        test_data=(X_test, y_test),
        problem_type="hmm",
        input_shape=(32, n_features)
    )

    # Display HMM results
    logger.info("🎯 HMM NAS Results:")
    logger.info(f"   Best Architecture: {search_result.best_architecture.name}")
    logger.info(f"   Best Score: {search_result.best_score:.4f}")
    logger.info(f"   Problem Type: {search_result.metadata.get('problem_type', 'unknown')}")

    return search_result

def example_regime_detection_nas():
    """Example of NAS for market regime detection."""
    logger.info("📈 Starting Regime Detection NAS Example")

    # Create regime-specific data
    n_samples = 2000
    n_regimes = 6
    n_features = 40

    # Generate regime data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Create regime patterns
    regimes = np.zeros(n_samples, dtype=np.int64)

    # Regime 1: Trending upward
    regime1_mask = np.arange(n_samples) % n_regimes == 0
    X[regime1_mask, :5] += np.linspace(0, 1, n_samples)[regime1_mask][:, np.newaxis]
    regimes[regime1_mask] = 0

    # Regime 2: High volatility
    regime2_mask = np.arange(n_samples) % n_regimes == 1
    X[regime2_mask, :5] += np.random.randn(np.sum(regime2_mask), 5) * 2.0
    regimes[regime2_mask] = 1

    # Regime 3: Sideways
    regime3_mask = np.arange(n_samples) % n_regimes == 2
    X[regime3_mask, :5] += np.random.randn(np.sum(regime3_mask), 5) * 0.1
    regimes[regime3_mask] = 2

    # Regime 4: Trending downward
    regime4_mask = np.arange(n_samples) % n_regimes == 3
    X[regime4_mask, :5] += np.linspace(0, -1, n_samples)[regime4_mask][:, np.newaxis]
    regimes[regime4_mask] = 3

    # Regime 5: Mixed
    regime5_mask = np.arange(n_samples) % n_regimes == 4
    X[regime5_mask, :5] += np.random.randn(np.sum(regime5_mask), 5) * 0.5
    regimes[regime5_mask] = 4

    # Regime 6: Extreme volatility
    regime6_mask = np.arange(n_samples) % n_regimes == 5
    X[regime6_mask, :5] += np.random.randn(np.sum(regime6_mask), 5) * 3.0
    regimes[regime6_mask] = 5

    # Split data
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)

    X_train = X[:n_train]
    y_train = regimes[:n_train]
    X_val = X[n_train:n_train+n_val]
    y_val = regimes[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    y_test = regimes[n_train+n_val:]

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64
    )

    # Configure NAS for regime detection
    search_config = NASSearchConfig(
        max_iterations=25,
        search_strategy="random",
        primary_metric="accuracy",
        use_gpu=torch.cuda.is_available(),
        batch_size=64
    )

    # Create NAS search for regime detection
    regime_nas = NASArchitectureSearch(search_config)

    # Perform search
    search_result = regime_nas.search(
        train_data=(X_train, y_train),
        validation_data=(X_val, y_val),
        test_data=(X_test, y_test),
        problem_type="regime_detection",
        input_shape=(64, n_features)
    )

    # Display results
    logger.info("🎯 Regime Detection NAS Results:")
    logger.info(f"   Best Architecture: {search_result.best_architecture.name}")
    logger.info(f"   Best Score: {search_result.best_score:.4f}")
    logger.info(f"   Number of Regimes: {n_regimes}")

    return search_result

def example_architecture_evaluation():
    """Example of comprehensive architecture evaluation."""
    logger.info("🔬 Starting Architecture Evaluation Example")

    # Create sample data
    X_train, y_train, X_val, y_val, X_test, y_test = create_sample_data(
        n_samples=1000, input_dim=25, output_dim=3, problem_type="classification"
    )

    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32
    )

    # Create search space
    search_space = SearchSpace()

    # Generate a few architectures
    architectures = []
    for i in range(5):
        arch = search_space.generate_random_architecture(
            input_dim=25, output_dim=3, problem_type="classification"
        )
        arch.name = f"test_architecture_{i+1}"
        architectures.append(arch)

    # Evaluate architectures
    evaluation_config = EvaluationConfig(
        batch_size=32,
        use_gpu=torch.cuda.is_available(),
        compute_confusion_matrix=True,
        compute_per_class_metrics=True
    )

    evaluator = NASEvaluator(evaluation_config)
    metrics_list = []

    for arch in architectures:
        logger.info(f"📊 Evaluating {arch.name}")

        # Create model
        model = NASModel.create_from_config(arch, "classification")

        # Train model
        trainer_config = TrainingConfig(epochs=10, batch_size=32)
        trainer = NASTrainer(trainer_config)
        trained_model = trainer.train(model, train_loader, val_loader, "classification")

        # Evaluate model
        metrics = evaluator.evaluate_architecture(
            trained_model.model, train_loader, val_loader, test_loader,
            arch.name, "classification"
        )

        metrics_list.append(metrics)

        logger.info(f"✅ {arch.name}: Accuracy = {metrics.accuracy:.4f}, Params = {metrics.num_parameters:,}")

    # Compare architectures
    nas_metrics = NASMetrics(NASMetricsConfig(primary_metric="accuracy"))
    comparison = nas_metrics.compare_architectures(metrics_list, "accuracy")

    logger.info("🏆 Architecture Comparison:")
    logger.info(f"   Best: {comparison['best_architecture']}")
    logger.info(f"   Score: {comparison['best_score']:.4f}")
    logger.info(f"   Average Accuracy: {comparison['average_metrics']['avg_accuracy']:.4f}")

    return comparison

def main():
    """Main function demonstrating NAS usage."""
    logger.info("🚀 NAS Modeling System - Usage Examples")
    logger.info("=" * 50)

    # Run examples
    try:
        # Basic NAS search
        logger.info("\\n1. Running Basic NAS Search...")
        basic_result = example_basic_nas_search()

        # Bayesian search
        logger.info("\\n2. Running Bayesian Search...")
        bayesian_result = example_bayesian_search()

        # HMM NAS
        logger.info("\\n3. Running HMM NAS...")
        hmm_result = example_hmm_nas()

        # Regime detection NAS
        logger.info("\\n4. Running Regime Detection NAS...")
        regime_result = example_regime_detection_nas()

        # Architecture evaluation
        logger.info("\\n5. Running Architecture Evaluation...")
        evaluation_result = example_architecture_evaluation()

        logger.info("\\n✅ All NAS examples completed successfully!")
        logger.info("=" * 50)

        # Summary
        logger.info("📋 Summary of Results:")
        logger.info(f"   Basic NAS - Best Score: {basic_result.best_score:.4f}")
        logger.info(f"   Bayesian - Best Score: {bayesian_result.best_score:.4f}")
        logger.info(f"   HMM NAS - Best Score: {hmm_result.best_score:.4f}")
        logger.info(f"   Regime NAS - Best Score: {regime_result.best_score:.4f}")

        return {
            'basic_nas': basic_result,
            'bayesian_search': bayesian_result,
            'hmm_nas': hmm_result,
            'regime_detection': regime_result,
            'evaluation': evaluation_result
        }

    except Exception as e:
        logger.error(f"❌ Error in NAS examples: {e}")
        raise

if __name__ == "__main__":
    main()