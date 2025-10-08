"""
Enhanced NAS-TAS Clustering Example

This example demonstrates the enhanced data-driven improvements for NAS-TAS evaluation and clustering:
- Regime evaluation metrics (Sharpe, Sortino, max drawdown, hit rate, payoff ratio)
- Feature correlation handling (PCA, VIF)
- Cross-validation for clustering parameters
- Robust scoring models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Import enhanced modules
from src.training.steps.market_analysis.hybrid_nas_tas_regime.core.enhanced_economic_clustering import (
    create_enhanced_economic_clusterer
)
from src.training.steps.market_analysis.hybrid_nas_tas_regime.evaluation.enhanced_regime_evaluator import (
    create_enhanced_regime_evaluator
)
from src.training.steps.market_analysis.hybrid_nas_tas_regime.evaluation.clustering_cross_validation import (
    create_clustering_cross_validator
)
from src.training.steps.market_analysis.hybrid_nas_tas_regime.evaluation.robust_scoring_models import (
    create_robust_scoring_models
)
from src.feature_selection.dimensionality import create_pca_module, create_vif_module

logger = logging.getLogger(__name__)


def generate_sample_data(n_samples: int = 1000, n_features: int = 50) -> tuple:
    """Generate sample market data and features for demonstration."""
    print("📊 Generating sample data...")
    
    # Generate market data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='15T')
    
    # Generate price data with regime changes
    price_changes = np.random.normal(0, 0.01, n_samples)
    # Add regime-specific patterns
    regime_periods = n_samples // 4
    for i in range(4):
        start_idx = i * regime_periods
        end_idx = min((i + 1) * regime_periods, n_samples)
        if i == 0:  # High volatility regime
            price_changes[start_idx:end_idx] *= 2.0
        elif i == 1:  # Trending regime
            price_changes[start_idx:end_idx] += 0.002
        elif i == 2:  # Mean-reverting regime
            price_changes[start_idx:end_idx] *= -0.5
        # i == 3: Normal regime
    
    prices = 100 * np.cumprod(1 + price_changes)
    
    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 0.5, n_samples)
    })
    
    # Generate features
    features = np.random.randn(n_samples, n_features)
    
    # Add some correlation between features
    for i in range(0, n_features, 5):
        if i + 1 < n_features:
            features[:, i+1] = 0.8 * features[:, i] + 0.2 * np.random.randn(n_samples)
    
    print(f"✅ Generated {n_samples} samples with {n_features} features")
    return market_data, features


def demonstrate_enhanced_regime_evaluation():
    """Demonstrate enhanced regime evaluation metrics."""
    print("\n" + "="*60)
    print("🔍 ENHANCED REGIME EVALUATION DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    market_data, features = generate_sample_data(500, 30)
    
    # Create sample regime labels
    n_regimes = 4
    regime_labels = np.random.randint(0, n_regimes, len(market_data))
    
    # Configure enhanced regime evaluator
    evaluator_config = {
        'risk_free_rate': 0.02,
        'confidence_level': 0.95,
        'min_regime_size': 10,
        'min_sharpe_threshold': 0.5,
        'min_sortino_threshold': 0.3,
        'max_drawdown_threshold': 0.2,
        'min_hit_rate_threshold': 0.4,
        'min_payoff_ratio_threshold': 1.0
    }
    
    # Create and run evaluator
    evaluator = create_enhanced_regime_evaluator(evaluator_config)
    
    print("📊 Evaluating regimes with enhanced metrics...")
    evaluation_result = evaluator.evaluate_regimes(market_data, regime_labels)
    
    print(f"✅ Evaluation completed:")
    print(f"   - Number of regimes: {len(evaluation_result.regime_metrics)}")
    print(f"   - Overall quality score: {evaluation_result.overall_quality_score:.3f}")
    print(f"   - Regime rankings available: {list(evaluation_result.regime_rankings.keys())}")
    
    # Display regime metrics
    print("\n📈 Regime Metrics:")
    for i, metric in enumerate(evaluation_result.regime_metrics):
        print(f"   Regime {i}:")
        print(f"     - Mean Return: {metric.mean_return:.4f}")
        print(f"     - Volatility: {metric.volatility:.4f}")
        print(f"     - Sharpe Ratio: {metric.sharpe_ratio:.3f}")
        print(f"     - Sortino Ratio: {metric.sortino_ratio:.3f}")
        print(f"     - Max Drawdown: {metric.max_drawdown:.3f}")
        print(f"     - Hit Rate: {metric.hit_rate:.3f}")
        print(f"     - Payoff Ratio: {metric.payoff_ratio:.3f}")
        print(f"     - Economic Significance: {metric.economic_significance:.3f}")
        print(f"     - Trading Viability: {metric.trading_viability:.3f}")


def demonstrate_feature_correlation_handling():
    """Demonstrate feature correlation handling with PCA and VIF."""
    print("\n" + "="*60)
    print("🔧 FEATURE CORRELATION HANDLING DEMONSTRATION")
    print("="*60)
    
    # Generate sample data with high correlation
    market_data, features = generate_sample_data(300, 100)
    
    # Configure PCA module
    pca_config = {
        'n_components': None,
        'variance_threshold': 0.95,
        'correlation_threshold': 0.9,
        'enable_correlation_filtering': True,
        'enable_variance_filtering': True
    }
    
    # Configure VIF module
    vif_config = {
        'vif_threshold': 10.0,
        'correlation_threshold': 0.9,
        'enable_correlation_filtering': True,
        'enable_variance_filtering': True,
        'stepwise_removal': True
    }
    
    print("🔧 Applying VIF-based feature selection...")
    vif_module = create_vif_module(vif_config)
    vif_result = vif_module.apply_vif_feature_selection(features)
    
    print(f"✅ VIF feature selection:")
    print(f"   - Original features: {features.shape[1]}")
    print(f"   - Selected features: {len(vif_result['selected_features'])}")
    print(f"   - VIF scores: {len(vif_result.get('vif_scores', {}))}")
    
    if vif_result['success']:
        selected_indices = vif_result['selected_indices']
        features_selected = features[:, selected_indices]
        
        print("\n🔧 Applying PCA for dimensionality reduction...")
        pca_module = create_pca_module(pca_config)
        pca_result = pca_module.apply_pca_feature_selection(features_selected)
        
        print(f"✅ PCA feature selection:")
        print(f"   - Input features: {features_selected.shape[1]}")
        print(f"   - Selected features: {len(pca_result['selected_features'])}")
        print(f"   - Explained variance ratio: {pca_result.get('explained_variance_ratio', [])[:5]}")
        print(f"   - Cumulative variance: {pca_result.get('cumulative_variance', [])[:5]}")


def demonstrate_clustering_cross_validation():
    """Demonstrate cross-validation for clustering parameters."""
    print("\n" + "="*60)
    print("🎯 CLUSTERING CROSS-VALIDATION DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    market_data, features = generate_sample_data(400, 25)
    
    # Configure cross-validation
    cv_config = {
        'cv_folds': 5,
        'test_size': 0.2,
        'random_state': 42,
        'scoring_metric': 'silhouette',
        'enable_time_series_cv': True,
        'n_regimes_range': list(range(2, 8)),
        'weight_ranges': {
            'economic_significance_weight': np.arange(0.1, 0.6, 0.1),
            'momentum_weight': np.arange(0.1, 0.6, 0.1),
            'volume_weight': np.arange(0.1, 0.6, 0.1)
        },
        'algorithm_options': ['kmeans', 'hierarchical', 'gmm']
    }
    
    print("🎯 Optimizing clustering parameters...")
    cv_validator = create_clustering_cross_validator(cv_config)
    cv_result = cv_validator.optimize_clustering_parameters(features, market_data)
    
    print(f"✅ Cross-validation completed:")
    print(f"   - Best parameters: {cv_result.best_params}")
    print(f"   - Best score: {cv_result.best_score:.3f}")
    print(f"   - CV scores: {len(cv_result.cv_scores)} metrics")
    print(f"   - Validation metrics: {list(cv_result.validation_metrics.keys())}")
    print(f"   - Stability scores: {list(cv_result.stability_scores.keys())}")


def demonstrate_robust_scoring_models():
    """Demonstrate robust scoring models."""
    print("\n" + "="*60)
    print("🤖 ROBUST SCORING MODELS DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    market_data, features = generate_sample_data(600, 20)
    
    # Create sample regime labels and metrics
    n_regimes = 3
    regime_labels = np.random.randint(0, n_regimes, len(market_data))
    
    # Create sample regime metrics
    regime_metrics = []
    for i in range(n_regimes):
        regime_metrics.append({
            'regime_id': i,
            'economic_significance': np.random.uniform(0.3, 0.9),
            'trading_viability': np.random.uniform(0.4, 0.8),
            'stability_score': np.random.uniform(0.5, 0.9),
            'risk_score': np.random.uniform(0.2, 0.8),
            'performance_score': np.random.uniform(0.3, 0.9)
        })
    
    # Configure scoring models
    scoring_config = {
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 5,
        'enable_feature_scaling': True,
        'model_selection_strategy': 'ensemble'
    }
    
    print("🤖 Training robust scoring models...")
    scoring_models = create_robust_scoring_models(scoring_config)
    model_performances = scoring_models.train_scoring_models(
        market_data, features, regime_labels, regime_metrics
    )
    
    print(f"✅ Scoring models trained:")
    print(f"   - Models trained: {len(model_performances)}")
    for target, performance in model_performances.items():
        print(f"   - {target}: {performance.test_score:.3f} (CV: {performance.cv_score:.3f})")
    
    # Test prediction
    print("\n🔮 Testing regime score prediction...")
    test_features = features[:10]
    test_market_data = market_data[:10]
    
    for i in range(3):
        regime_features = test_features[i:i+1]
        regime_market_data = test_market_data[i:i+1]
        
        scoring_result = scoring_models.predict_regime_scores(
            regime_features, regime_market_data, i
        )
        
        print(f"   Regime {i} predictions:")
        print(f"     - Economic Significance: {scoring_result.economic_significance:.3f}")
        print(f"     - Trading Viability: {scoring_result.trading_viability:.3f}")
        print(f"     - Stability Score: {scoring_result.stability_score:.3f}")
        print(f"     - Risk Score: {scoring_result.risk_score:.3f}")
        print(f"     - Performance Score: {scoring_result.performance_score:.3f}")


def demonstrate_enhanced_clustering():
    """Demonstrate the complete enhanced clustering system."""
    print("\n" + "="*60)
    print("🚀 ENHANCED CLUSTERING SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    market_data, features = generate_sample_data(500, 40)
    
    # Configure enhanced clustering
    clustering_config = {
        'n_regimes': 4,
        'primary_algorithm': 'economic_adaptive',
        'enable_feature_selection': True,
        'enable_cross_validation': True,
        'enable_scoring_models': True,
        'evaluator_config': {
            'risk_free_rate': 0.02,
            'min_regime_size': 10
        },
        'cv_config': {
            'cv_folds': 3,
            'scoring_metric': 'silhouette',
            'n_regimes_range': list(range(2, 6))
        },
        'scoring_config': {
            'test_size': 0.2,
            'enable_feature_scaling': True
        },
        'pca_config': {
            'variance_threshold': 0.95,
            'correlation_threshold': 0.9
        },
        'vif_config': {
            'vif_threshold': 10.0,
            'correlation_threshold': 0.9
        }
    }
    
    print("🚀 Running enhanced economic clustering...")
    clusterer = create_enhanced_economic_clusterer(clustering_config)
    clustering_result = clusterer.cluster_with_enhanced_evaluation(
        features, market_data, market_data  # Use market_data as historical data
    )
    
    print(f"✅ Enhanced clustering completed:")
    print(f"   - Number of clusters: {len(set(clustering_result.labels))}")
    print(f"   - Overall quality score: {clustering_result.overall_quality_score:.3f}")
    print(f"   - Algorithm used: {clustering_result.algorithm_used}")
    print(f"   - Feature selection: {clustering_result.feature_selection_info.get('n_selected_features', 'N/A')} features")
    print(f"   - Cross-validation: {clustering_result.cross_validation_results.get('best_score', 0.0):.3f}")
    print(f"   - Scoring models: {clustering_result.scoring_model_results.get('models_trained', 0)} models")
    
    # Display regime rankings
    print("\n📊 Regime Rankings:")
    for ranking_type, rankings in clustering_result.regime_rankings.items():
        print(f"   {ranking_type}: {rankings}")
    
    # Display economic metrics
    print("\n💰 Economic Metrics:")
    print(f"   - Regime sizes: {clustering_result.economic_metrics.get('regime_sizes', [])}")
    print(f"   - Risk-adjusted rankings: {list(clustering_result.economic_metrics.get('risk_adjusted_rankings', {}).keys())}")
    print(f"   - Economic rankings: {list(clustering_result.economic_metrics.get('economic_rankings', {}).keys())}")
    print(f"   - Trading rankings: {list(clustering_result.economic_metrics.get('trading_rankings', {}).keys())}")


def main():
    """Run all demonstrations."""
    print("🎯 ENHANCED NAS-TAS EVALUATION AND CLUSTERING DEMONSTRATION")
    print("="*80)
    
    try:
        # Demonstrate individual components
        demonstrate_enhanced_regime_evaluation()
        demonstrate_feature_correlation_handling()
        demonstrate_clustering_cross_validation()
        demonstrate_robust_scoring_models()
        
        # Demonstrate complete system
        demonstrate_enhanced_clustering()
        
        print("\n" + "="*80)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nKey improvements demonstrated:")
        print("✅ Enhanced regime evaluation metrics (Sharpe, Sortino, max drawdown, hit rate, payoff ratio)")
        print("✅ Feature correlation handling (PCA, VIF)")
        print("✅ Cross-validation for clustering parameters")
        print("✅ Robust scoring models for regime quality prediction")
        print("✅ Integrated enhanced economic clustering system")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()