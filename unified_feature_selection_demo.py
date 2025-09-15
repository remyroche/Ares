#!/usr/bin/env python3
"""
Unified Feature Selection Framework Demo

This script demonstrates the unified feature selection framework that:
1. Consolidates all existing feature selection methods
2. Leverages matrix operations for efficient computations
3. Provides backwards compatibility
4. Generates feature sets of different sizes (120, 100, 80, 60)
5. Supports both price prediction and HMM regime prediction

Author: AI Assistant
Date: 2024-01-XX
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the unified framework
try:
    from src.utils.ml_common.unified_feature_selection import (
        UnifiedFeatureSelector, UnifiedFeatureSelectionConfig,
        create_unified_selector, select_features_unified, generate_feature_sets
    )
    from src.utils.ml_common.matrix_feature_operations import (
        MatrixFeatureOperations, create_matrix_feature_operations
    )
    from src.utils.ml_common.backwards_compatibility import (
        BackwardsCompatibilityWrapper, show_migration_guide
    )
    UNIFIED_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import unified framework: {e}")
    UNIFIED_FRAMEWORK_AVAILABLE = False


def generate_sample_data(n_samples: int = 10000, n_features: int = 500) -> tuple:
    """
    Generate sample financial data for demonstration.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    logger.info(f"Generating sample data: {n_samples} samples, {n_features} features")
    
    np.random.seed(42)
    
    # Generate feature matrix with different types of features
    X = np.random.randn(n_samples, n_features)
    
    # Add some structure to make features more realistic
    # Price-based features (high correlation with target)
    X[:, :50] = X[:, :50] * 0.1 + np.random.randn(n_samples, 50) * 0.9
    
    # Technical indicator features (medium correlation)
    X[:, 50:150] = X[:, 50:150] * 0.3 + np.random.randn(n_samples, 100) * 0.7
    
    # Noise features (low correlation)
    X[:, 150:] = np.random.randn(n_samples, n_features - 150)
    
    # Generate target variable (price prediction)
    # Target is influenced by first 50 features
    y_price = np.sum(X[:, :50], axis=1) * 0.1 + np.random.randn(n_samples) * 0.5
    
    # Generate HMM regime labels (classification)
    # Create 3 regimes based on volatility
    volatility = np.std(X[:, :50], axis=1)
    regime_thresholds = np.percentile(volatility, [33, 67])
    y_regime = np.zeros(n_samples, dtype=int)
    y_regime[volatility > regime_thresholds[1]] = 2  # High volatility
    y_regime[volatility > regime_thresholds[0]] = 1  # Medium volatility
    # Low volatility remains 0
    
    # Generate feature names
    feature_names = []
    
    # Price-based features
    for i in range(50):
        feature_names.append(f'price_feature_{i}')
    
    # Technical indicator features
    for i in range(100):
        feature_names.append(f'technical_feature_{i}')
    
    # Noise features
    for i in range(n_features - 150):
        feature_names.append(f'noise_feature_{i}')
    
    return X, y_price, y_regime, feature_names


def demonstrate_unified_framework():
    """Demonstrate the unified feature selection framework."""
    logger.info("🚀 Demonstrating Unified Feature Selection Framework")
    
    if not UNIFIED_FRAMEWORK_AVAILABLE:
        logger.error("❌ Unified framework not available")
        return
    
    # Generate sample data
    X, y_price, y_regime, feature_names = generate_sample_data()
    
    logger.info(f"📊 Generated data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"🎯 Price target range: {y_price.min():.3f} to {y_price.max():.3f}")
    logger.info(f"🎯 Regime distribution: {np.bincount(y_regime)}")
    
    # Demo 1: Price prediction with multiple feature set sizes
    logger.info("\n" + "="*60)
    logger.info("📈 DEMO 1: Price Prediction Feature Selection")
    logger.info("="*60)
    
    config_price = UnifiedFeatureSelectionConfig(
        target_features=120,
        task_type="regression",
        prediction_target="price",
        primary_method="hybrid",
        save_results=True,
        output_dir="demo_results/price_prediction"
    )
    
    selector_price = UnifiedFeatureSelector(config_price)
    results_price = selector_price.select_features(
        X, y_price, feature_names, target_sizes=[120, 100, 80, 60]
    )
    
    # Display results
    for size_name, result in results_price.items():
        logger.info(f"✅ {size_name}: {len(result['selected_features'])} features selected")
        logger.info(f"   Method: {result['method']}")
        logger.info(f"   Selection ratio: {result['selection_ratio']:.3f}")
    
    # Demo 2: HMM regime prediction
    logger.info("\n" + "="*60)
    logger.info("🎯 DEMO 2: HMM Regime Prediction Feature Selection")
    logger.info("="*60)
    
    config_regime = UnifiedFeatureSelectionConfig(
        target_features=100,
        task_type="classification",
        prediction_target="hmm_regime",
        primary_method="hybrid",
        save_results=True,
        output_dir="demo_results/hmm_regime"
    )
    
    selector_regime = UnifiedFeatureSelector(config_regime)
    results_regime = selector_regime.select_features(X, y_regime, feature_names)
    
    # Display HMM results
    if 'hmm_regime_top_100' in results_regime:
        hmm_result = results_regime['hmm_regime_top_100']
        logger.info(f"✅ HMM regime features: {len(hmm_result['selected_features'])} features selected")
        logger.info(f"   Method: {hmm_result['method']}")
        
        # Show regime analysis
        if 'regime_analysis' in hmm_result:
            regime_analysis = hmm_result['regime_analysis']
            logger.info(f"   Regimes detected: {regime_analysis['n_regimes']}")
            logger.info(f"   Unique regimes: {regime_analysis['unique_regimes']}")
    
    # Demo 3: Matrix operations integration
    logger.info("\n" + "="*60)
    logger.info("⚡ DEMO 3: Matrix Operations Integration")
    logger.info("="*60)
    
    matrix_ops = create_matrix_feature_operations(use_gpu=True, use_parallel=True)
    
    # Test correlation matrix computation
    start_time = time.time()
    corr_matrix = matrix_ops.correlation_matrix(X, method="pearson", feature_names=feature_names)
    corr_time = time.time() - start_time
    logger.info(f"✅ Correlation matrix computed in {corr_time:.3f}s")
    logger.info(f"   Matrix shape: {corr_matrix.shape}")
    
    # Test hierarchical clustering
    start_time = time.time()
    clustering_result = matrix_ops.hierarchical_clustering_correlation(
        X, correlation_threshold=0.95, feature_names=feature_names
    )
    clustering_time = time.time() - start_time
    logger.info(f"✅ Hierarchical clustering completed in {clustering_time:.3f}s")
    logger.info(f"   Clusters found: {clustering_result['n_clusters']}")
    logger.info(f"   Representative features: {clustering_result['n_representatives']}")
    
    # Demo 4: Backwards compatibility
    logger.info("\n" + "="*60)
    logger.info("🔄 DEMO 4: Backwards Compatibility")
    logger.info("="*60)
    
    # Show migration guide
    show_migration_guide()
    
    # Test legacy interface
    legacy_selector = BackwardsCompatibilityWrapper()
    legacy_selector.fit(X, y_price)
    legacy_features = legacy_selector.get_feature_names_out()
    logger.info(f"✅ Legacy interface: {len(legacy_features)} features selected")
    
    # Demo 5: Convenience functions
    logger.info("\n" + "="*60)
    logger.info("🎯 DEMO 5: Convenience Functions")
    logger.info("="*60)
    
    # Test convenience function
    start_time = time.time()
    convenience_result = select_features_unified(
        X, y_price, feature_names, target_features=80, task_type="regression"
    )
    convenience_time = time.time() - start_time
    logger.info(f"✅ Convenience function completed in {convenience_time:.3f}s")
    
    # Test feature set generation
    start_time = time.time()
    feature_sets = generate_feature_sets(
        X, y_price, feature_names, target_sizes=[120, 100, 80, 60]
    )
    generation_time = time.time() - start_time
    logger.info(f"✅ Feature set generation completed in {generation_time:.3f}s")
    
    for set_name, features in feature_sets.items():
        logger.info(f"   {set_name}: {len(features)} features")
    
    return {
        'price_results': results_price,
        'regime_results': results_regime,
        'matrix_operations': {
            'correlation_time': corr_time,
            'clustering_time': clustering_time
        },
        'convenience_time': convenience_time,
        'generation_time': generation_time,
        'feature_sets': feature_sets
    }


def demonstrate_random_forest_refinement():
    """Demonstrate Random Forest-based feature refinement."""
    logger.info("\n" + "="*60)
    logger.info("🌲 DEMO: Random Forest Feature Refinement")
    logger.info("="*60)
    
    if not UNIFIED_FRAMEWORK_AVAILABLE:
        logger.error("❌ Unified framework not available")
        return
    
    # Generate sample data
    X, y_price, y_regime, feature_names = generate_sample_data()
    
    # Get initial 120 features
    config = UnifiedFeatureSelectionConfig(
        target_features=120,
        task_type="regression",
        primary_method="hybrid"
    )
    
    selector = UnifiedFeatureSelector(config)
    results = selector.select_features(X, y_price, feature_names, target_sizes=[120])
    
    # Get the 120 features
    top_120_features = results['top_120']['selected_features']
    logger.info(f"📊 Initial 120 features selected")
    
    # Refine using Random Forest for different sizes
    from sklearn.ensemble import RandomForestRegressor
    
    # Get feature indices
    feature_indices = [feature_names.index(feat) for feat in top_120_features]
    X_selected = X[:, feature_indices]
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_selected, y_price)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Create feature importance mapping
    feature_importance_map = {
        top_120_features[i]: importances[i] 
        for i in range(len(top_120_features))
    }
    
    # Sort by importance
    sorted_features = sorted(feature_importance_map.items(), key=lambda x: x[1], reverse=True)
    
    # Create refined feature sets
    refined_sets = {}
    for target_size in [100, 80, 60]:
        refined_features = [feat for feat, _ in sorted_features[:target_size]]
        refined_sets[f'rf_refined_{target_size}'] = refined_features
        logger.info(f"✅ RF refined {target_size} features: {len(refined_features)} features")
    
    return refined_sets


def demonstrate_hmm_regime_selection():
    """Demonstrate HMM regime-specific feature selection."""
    logger.info("\n" + "="*60)
    logger.info("🎯 DEMO: HMM Regime-Specific Feature Selection")
    logger.info("="*60)
    
    if not UNIFIED_FRAMEWORK_AVAILABLE:
        logger.error("❌ Unified framework not available")
        return
    
    # Generate sample data
    X, y_price, y_regime, feature_names = generate_sample_data()
    
    # Configure for HMM regime prediction
    config = UnifiedFeatureSelectionConfig(
        target_features=100,
        task_type="classification",
        prediction_target="hmm_regime",
        primary_method="hybrid"
    )
    
    selector = UnifiedFeatureSelector(config)
    results = selector.select_features(X, y_regime, feature_names)
    
    # Get HMM regime features
    hmm_features = selector.get_hmm_regime_features()
    logger.info(f"✅ HMM regime features: {len(hmm_features)} features selected")
    
    # Analyze regime separation
    if 'hmm_regime_top_100' in results:
        regime_analysis = results['hmm_regime_top_100']['regime_analysis']
        
        logger.info(f"📊 Regime Analysis:")
        logger.info(f"   Number of regimes: {regime_analysis['n_regimes']}")
        logger.info(f"   Unique regimes: {regime_analysis['unique_regimes']}")
        
        # Show top regime-separating features
        separation_scores = regime_analysis['regime_separation_scores']
        top_separating_features = sorted(
            separation_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        logger.info("🏆 Top 10 regime-separating features:")
        for i, (feature, score) in enumerate(top_separating_features, 1):
            logger.info(f"   {i:2d}. {feature}: {score:.4f}")
    
    return hmm_features


def main():
    """Main demonstration function."""
    logger.info("🎯 Starting Unified Feature Selection Framework Demo")
    logger.info("="*80)
    
    try:
        # Run all demonstrations
        results = {}
        
        # Main unified framework demo
        results['unified_demo'] = demonstrate_unified_framework()
        
        # Random Forest refinement demo
        results['rf_refinement'] = demonstrate_random_forest_refinement()
        
        # HMM regime selection demo
        results['hmm_regime'] = demonstrate_hmm_regime_selection()
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("📊 DEMONSTRATION SUMMARY")
        logger.info("="*80)
        
        logger.info("✅ All demonstrations completed successfully!")
        logger.info("🎯 Key Features Demonstrated:")
        logger.info("   - Unified feature selection framework")
        logger.info("   - Multiple feature set sizes (120, 100, 80, 60)")
        logger.info("   - HMM regime-specific selection")
        logger.info("   - Matrix operations integration")
        logger.info("   - Backwards compatibility")
        logger.info("   - Random Forest refinement")
        logger.info("   - Convenience functions")
        
        # Save summary results
        summary_file = Path("demo_results/demo_summary.json")
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(summary_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = value
                else:
                    serializable_results[key] = str(value)
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"💾 Summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()