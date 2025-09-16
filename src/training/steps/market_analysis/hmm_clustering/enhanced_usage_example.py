#!/usr/bin/env python3
"""
Enhanced HMM Clustering Usage Example

This example demonstrates how to use the enhanced HMM clustering modules
with full integration of all available common utilities.
"""

import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Import enhanced HMM clustering modules
from hmm_executor import (
    create_hmm_dependencies,
    train_hmm_optimized,
    save_hmm_results,
    validate_hmm_model
)
from hmm_utils import HMMCommonUtilities
from clustering_executor import (
    create_clustering_dependencies,
    kmeans_standard,
    kmeans_minibatch,
    save_clustering_results
)

# Import common utilities
from src.utils.common_operations import (
    get_m1_gpu_manager,
    get_m1_memory_optimizer,
    get_m1_cpu_optimizer,
    validate_dataframe_columns,
    calculate_data_quality_metrics
)
from src.utils.common_utilities import safe_convert_dtypes
from src.utils.math_validation import safe_divide, safe_log
from src.utils.serialization_utils import JSONSerializer, PickleSerializer
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
from src.utils.ml_common.hmm_regime_detection import HMMRegimeDetector
from src.utils.ml_common.validation.cross_validation import TimeSeriesCrossValidator
from src.utils.ml_common.optimization.hyperparameter_optimization import HyperparameterOptimizer
from src.utils.logger import system_logger

# Setup logging
logger = system_logger.getChild('EnhancedHMMExample')

def generate_sample_market_data(n_samples: int = 2000) -> pd.DataFrame:
    """Generate realistic sample market data for testing."""
    logger.info(f"📊 Generating sample market data: {n_samples} samples")
    
    np.random.seed(42)
    
    # Generate time series data
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    # Generate price data with trends and volatility clusters
    returns = np.random.normal(0, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate volume data
    volume = np.random.lognormal(10, 0.5, n_samples)
    
    # Create OHLCV data
    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': volume
    })
    
    logger.info(f"✅ Sample market data generated: {market_data.shape}")
    return market_data

def demonstrate_enhanced_hmm_clustering():
    """Demonstrate enhanced HMM clustering with common utilities integration."""
    logger.info("🚀 Starting Enhanced HMM Clustering Demonstration")
    
    try:
        # Generate sample data
        market_data = generate_sample_market_data(1000)
        
        # Initialize HMM common utilities
        hmm_utils = HMMCommonUtilities()
        
        # Prepare features with validation
        logger.info("🔧 Preparing features with validation...")
        features_prepared = hmm_utils.prepare_features_with_validation(market_data)
        
        # Calculate technical indicators
        logger.info("📊 Calculating technical indicators...")
        features_with_indicators = hmm_utils.calculate_technical_indicators_safe(features_prepared)
        
        # Select numeric features for HMM training
        numeric_features = features_with_indicators.select_dtypes(include=[np.number]).columns.tolist()
        features_array = features_with_indicators[numeric_features].values
        
        logger.info(f"Features prepared: {features_array.shape}")
        
        # Create HMM dependencies with common utilities
        hmm_deps = create_hmm_dependencies()
        
        # Train HMM with optimization
        logger.info("🎯 Training HMM with hardware optimization...")
        hmm_results = train_hmm_optimized(
            features=features_with_indicators[numeric_features],
            n_components=3,
            covariance_type='full',
            n_iter=100,
            random_state=42,
            deps=hmm_deps
        )
        
        logger.info("✅ HMM training completed!")
        logger.info(f"   Used GPU: {hmm_results.get('used_gpu', False)}")
        logger.info(f"   Score: {hmm_results.get('score', 0):.3f}")
        logger.info(f"   Converged: {hmm_results.get('validation_metrics', {}).get('converged', False)}")
        
        # Validate HMM model
        logger.info("🔍 Validating HMM model...")
        validation_results = validate_hmm_model(
            hmm_results['model'],
            features_array,
            3,
            hmm_deps.logger
        )
        logger.info(f"Validation results: {validation_results}")
        
        # Save HMM results
        logger.info("💾 Saving HMM results...")
        save_success = save_hmm_results(hmm_results, 'hmm_results.json', hmm_deps)
        if save_success:
            logger.info("✅ HMM results saved successfully")
        
        # Demonstrate clustering with common utilities
        logger.info("🎭 Demonstrating clustering with common utilities...")
        
        # Create clustering dependencies
        clustering_deps = create_clustering_dependencies()
        
        # Run KMeans clustering
        kmeans_results = kmeans_standard(
            features_array=features_array,
            n_clusters=3,
            random_state=42,
            logger=clustering_deps.logger,
            deps=clustering_deps
        )
        
        logger.info("✅ KMeans clustering completed!")
        logger.info(f"   Quality metrics: {kmeans_results['quality_metrics']}")
        logger.info(f"   Used optimization: {kmeans_results.get('used_optimization', False)}")
        
        # Run MiniBatch KMeans clustering
        minibatch_results = kmeans_minibatch(
            features_array=features_array,
            n_clusters=3,
            random_state=42,
            logger=clustering_deps.logger,
            deps=clustering_deps
        )
        
        logger.info("✅ MiniBatch KMeans clustering completed!")
        logger.info(f"   Quality metrics: {minibatch_results['quality_metrics']}")
        
        # Save clustering results
        save_clustering_results(kmeans_results, 'kmeans_results.json', clustering_deps)
        save_clustering_results(minibatch_results, 'minibatch_results.json', clustering_deps)
        
        # Demonstrate cross-validation
        logger.info("🔄 Running cross-validation...")
        cv_results = hmm_utils.run_cross_validation(
            hmm_results['model'],
            features_array,
            cv_folds=5
        )
        logger.info(f"Cross-validation results: {cv_results}")
        
        # Demonstrate hyperparameter optimization
        logger.info("🔧 Running hyperparameter optimization...")
        param_grid = {
            'n_components': [2, 3, 4],
            'covariance_type': ['full', 'tied'],
            'n_iter': [50, 100]
        }
        
        best_params = hmm_utils.optimize_hyperparameters(
            model_class=type(hmm_results['model']),
            data=features_array,
            param_grid=param_grid
        )
        logger.info(f"Best parameters: {best_params}")
        
        # Create comprehensive results summary
        comprehensive_results = {
            'hmm_results': hmm_results,
            'validation_results': validation_results,
            'kmeans_results': kmeans_results,
            'minibatch_results': minibatch_results,
            'cv_results': cv_results,
            'best_params': best_params,
            'data_quality': calculate_data_quality_metrics(features_with_indicators),
            'timestamp': time.time()
        }
        
        # Save comprehensive results
        hmm_utils.save_results(comprehensive_results, 'comprehensive_results.json')
        
        logger.info("🎉 Enhanced HMM Clustering Demonstration completed successfully!")
        
        return comprehensive_results
        
    except Exception as e:
        logger.error(f"❌ Enhanced HMM Clustering Demonstration failed: {e}")
        raise

def demonstrate_common_utilities_integration():
    """Demonstrate integration with all common utilities."""
    logger.info("🔧 Demonstrating Common Utilities Integration")
    
    try:
        # Initialize all common utilities
        logger.info("Initializing common utilities...")
        
        # Hardware utilities
        gpu_manager = get_m1_gpu_manager()
        memory_optimizer = get_m1_memory_optimizer()
        cpu_optimizer = get_m1_cpu_optimizer()
        
        # Data operations
        matrix_ops = UnifiedMatrixOperations()
        json_serializer = JSONSerializer()
        pickle_serializer = PickleSerializer()
        
        # ML utilities
        hmm_regime_detector = HMMRegimeDetector()
        cv_validator = TimeSeriesCrossValidator()
        hpo_optimizer = HyperparameterOptimizer()
        
        # Log utility status
        logger.info("Common utilities status:")
        logger.info(f"   GPU Manager: {'Available' if gpu_manager else 'Not Available'}")
        logger.info(f"   Memory Optimizer: {'Available' if memory_optimizer else 'Not Available'}")
        logger.info(f"   CPU Optimizer: {'Available' if cpu_optimizer else 'Not Available'}")
        logger.info(f"   Matrix Operations: {'Available' if matrix_ops else 'Not Available'}")
        logger.info(f"   HMM Regime Detector: {'Available' if hmm_regime_detector else 'Not Available'}")
        logger.info(f"   CV Validator: {'Available' if cv_validator else 'Not Available'}")
        logger.info(f"   HPO Optimizer: {'Available' if hpo_optimizer else 'Not Available'}")
        
        # Generate sample data
        sample_data = generate_sample_market_data(500)
        
        # Demonstrate data operations
        logger.info("Demonstrating data operations...")
        
        # Validate DataFrame
        is_valid = validate_dataframe_columns(sample_data, sample_data.columns.tolist())
        logger.info(f"DataFrame validation: {'Passed' if is_valid else 'Failed'}")
        
        # Calculate data quality metrics
        quality_metrics = calculate_data_quality_metrics(sample_data)
        logger.info(f"Data quality metrics: {quality_metrics}")
        
        # Convert dtypes
        dtype_mapping = {col: 'float32' for col in sample_data.select_dtypes(include=[np.number]).columns}
        sample_data_optimized = safe_convert_dtypes(sample_data, dtype_mapping)
        logger.info("Data types optimized for performance")
        
        # Demonstrate safe math operations
        logger.info("Demonstrating safe math operations...")
        
        if 'close' in sample_data.columns:
            returns = sample_data['close'].pct_change()
            log_returns = safe_log(returns, 0.0)
            volatility = returns.rolling(20).std()
            
            logger.info(f"Calculated returns, log returns, and volatility")
            logger.info(f"   Returns range: {returns.min():.4f} to {returns.max():.4f}")
            logger.info(f"   Log returns range: {log_returns.min():.4f} to {log_returns.max():.4f}")
            logger.info(f"   Volatility range: {volatility.min():.4f} to {volatility.max():.4f}")
        
        # Demonstrate matrix operations
        if matrix_ops:
            logger.info("Demonstrating matrix operations...")
            numeric_data = sample_data.select_dtypes(include=[np.number]).values
            
            if hasattr(matrix_ops, 'optimize_for_clustering'):
                optimized_data = matrix_ops.optimize_for_clustering(numeric_data)
                logger.info(f"Matrix optimized: {numeric_data.shape} -> {optimized_data.shape}")
        
        # Demonstrate serialization
        logger.info("Demonstrating serialization...")
        
        test_data = {
            'sample_data_shape': sample_data.shape,
            'quality_metrics': quality_metrics,
            'timestamp': time.time()
        }
        
        # Save as JSON
        json_success = json_serializer.save(test_data, 'test_data.json')
        logger.info(f"JSON serialization: {'Success' if json_success else 'Failed'}")
        
        # Save as Pickle
        pickle_success = pickle_serializer.save(test_data, 'test_data.pkl')
        logger.info(f"Pickle serialization: {'Success' if pickle_success else 'Failed'}")
        
        logger.info("✅ Common Utilities Integration demonstration completed!")
        
    except Exception as e:
        logger.error(f"❌ Common Utilities Integration demonstration failed: {e}")
        raise

def main():
    """Main function to run all demonstrations."""
    logger.info("🚀 Starting Enhanced HMM Clustering with Common Utilities Integration")
    
    try:
        # Demonstrate common utilities integration
        demonstrate_common_utilities_integration()
        
        # Demonstrate enhanced HMM clustering
        results = demonstrate_enhanced_hmm_clustering()
        
        # Print summary
        logger.info("📊 Demonstration Summary:")
        logger.info(f"   HMM Score: {results['hmm_results'].get('score', 0):.3f}")
        logger.info(f"   HMM Converged: {results['validation_results'].get('converged', False)}")
        logger.info(f"   KMeans Balance Score: {results['kmeans_results']['quality_metrics'].get('regime_balance_score', 0):.3f}")
        logger.info(f"   MiniBatch Balance Score: {results['minibatch_results']['quality_metrics'].get('regime_balance_score', 0):.3f}")
        logger.info(f"   CV Mean Score: {results['cv_results'].get('mean_score', 0):.3f}")
        logger.info(f"   Best Parameters: {results['best_params']}")
        
        logger.info("🎉 All demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise

if __name__ == "__main__":
    main()