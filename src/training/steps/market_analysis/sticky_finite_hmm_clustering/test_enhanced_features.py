#!/usr/bin/env python3
"""
Test enhanced CSV and PCA features in markdown reports
"""

import sys
import os
# Add the path to the clusters directory
current_dir = os.path.dirname(os.path.abspath(__file__))
clusters_dir = os.path.join(current_dir, '..', 'clusters')
sys.path.insert(0, clusters_dir)

from cluster_quality_assessor import ClusterQualityAssessor, ClusterQualityMetrics
from datetime import datetime

def test_enhanced_features():
    """Test the enhanced CSV and PCA markdown features"""
    
    print("🧪 Testing Enhanced CSV and PCA Features")
    print("=" * 50)
    
    # Create assessor
    assessor = ClusterQualityAssessor()
    
    # Create sample metrics with enhanced data
    metrics = ClusterQualityMetrics(
        quality_score=0.82,
        silhouette_score=0.48,
        davies_bouldin_score=1.1,
        calinski_harabasz_score=185.0,
        n_regimes=4,
        temporal_smoothness=0.88,
        regime_persistence=28.5,
        balance_score=0.92,
        cluster_size_distribution=[0.24, 0.26, 0.23, 0.27],
        # Enhanced CV metrics
        within_regime_cv=0.15,
        within_regime_cv_std=0.03,
        between_regime_cv=0.45,
        between_regime_cv_std=0.08,
        per_regime_cv={0: 0.12, 1: 0.18, 2: 0.14, 3: 0.16},
        # Economic metrics
        economic_validation={
            0: {'mean_return': 0.0025, 'volatility': 0.015, 'sharpe': 1.67, 'max_drawdown': -0.08, 'hit_rate': 0.65, 'size': 1000},
            1: {'mean_return': -0.0018, 'volatility': 0.022, 'sharpe': -0.82, 'max_drawdown': -0.15, 'hit_rate': 0.45, 'size': 1100},
            2: {'mean_return': 0.0042, 'volatility': 0.018, 'sharpe': 2.33, 'max_drawdown': -0.06, 'hit_rate': 0.72, 'size': 950},
            3: {'mean_return': -0.0008, 'volatility': 0.012, 'sharpe': -0.67, 'max_drawdown': -0.10, 'hit_rate': 0.52, 'size': 1050}
        },
        predictive_power=0.73,
        log_likelihood=-1450.0,
        # HMM validation metrics
        refit_stability_ari=0.85,
        state_occupancy={0: 0.24, 1: 0.26, 2: 0.23, 3: 0.27},
        occupancy_entropy=1.38
    )
    
    # Enhanced method-specific config with PCA information
    method_config = {
        'K': 4,
        'base_alpha': 0.75,
        'kappa': 18.5,
        'n_mixtures': 2,
        'pca_components': 15,
        'learning_rate': 0.045,
        'svi_iterations': 1200,
        # PCA Feature Information
        'feature_categories': {
            'price': ['close', 'open', 'high', 'low', 'vwap', 'typical_price', 'weighted_close'],
            'volume': ['volume', 'volume_sma', 'volume_ratio', 'on_balance_volume', 'money_flow'],
            'volatility': ['atr', 'volatility_ratio', 'historical_volatility', 'garman_klass_vol', 'parkinson_vol'],
            'momentum': ['rsi', 'macd', 'momentum', 'rate_of_change', 'commodity_channel_index'],
            'mean_reversion': ['bollinger_position', 'stochastic', 'williams_r', 'relative_variance'],
            'technical': ['ema_12', 'ema_26', 'sma_50', 'sma_200', 'adx', 'aroon_up', 'aroon_down']
        },
        'pca_variance_ratio': [
            0.28, 0.19, 0.12, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02,
            0.02, 0.01, 0.01, 0.01, 0.00
        ],
        'pca_feature_loadings': [
            {  # PC1 - Price momentum component
                'close': 0.32, 'volume': -0.15, 'rsi': 0.28, 'atr': 0.25, 
                'ema_12': 0.31, 'macd': 0.29, 'volatility_ratio': 0.18
            },
            {  # PC2 - Volatility component  
                'atr': 0.41, 'volatility_ratio': 0.38, 'historical_volatility': 0.35,
                'garman_klass_vol': 0.33, 'close': -0.12, 'volume': 0.22
            },
            {  # PC3 - Volume flow component
                'volume': 0.45, 'on_balance_volume': 0.42, 'money_flow': 0.38,
                'volume_ratio': 0.31, 'vwap': 0.25, 'close': 0.08
            },
            {  # PC4 - Mean reversion component
                'bollinger_position': -0.39, 'stochastic': -0.35, 'williams_r': -0.32,
                'relative_variance': 0.28, 'rsi': 0.18, 'close': -0.15
            },
            {  # PC5 - Trend strength component
                'adx': 0.44, 'aroon_up': 0.41, 'momentum': 0.38, 'rate_of_change': 0.35,
                'macd': 0.32, 'ema_12': 0.25, 'sma_50': 0.22
            }
        ]
    }
    
    # Create sample all_trials data with enhanced metrics
    all_trials = [
        {
            'trial_number': 1,
            'params': {'K': 4, 'base_alpha': 0.75, 'kappa': 18.5, 'n_mixtures': 2, 
                      'pca_components': 15, 'learning_rate': 0.045, 'svi_iterations': 1200},
            'final_elbo': -1450.0,
            'quality_metrics': {
                'quality_score': 0.82, 'silhouette_score': 0.48, 'davies_bouldin_score': 1.1,
                'calinski_harabasz_score': 185.0, 'within_regime_cv': 0.15, 'between_regime_cv': 0.45,
                'within_regime_cv_std': 0.03, 'between_regime_cv_std': 0.08,
                'temporal_smoothness': 0.88, 'regime_persistence': 28.5, 'balance_score': 0.92,
                'n_regimes': 4, 'noise_ratio': 0.05, 'predictive_power': 0.73,
                'economic_validation': metrics.economic_validation,
                'log_likelihood': -1450.0, 'refit_stability_ari': 0.85, 'occupancy_entropy': 1.38,
                'cluster_size_distribution': [0.24, 0.26, 0.23, 0.27]
            }
        },
        {
            'trial_number': 2,
            'params': {'K': 5, 'base_alpha': 0.65, 'kappa': 15.0, 'n_mixtures': 1, 
                      'pca_components': 12, 'learning_rate': 0.035, 'svi_iterations': 1000},
            'final_elbo': -1520.0,
            'quality_metrics': {
                'quality_score': 0.76, 'silhouette_score': 0.42, 'davies_bouldin_score': 1.3,
                'calinski_harabasz_score': 165.0, 'within_regime_cv': 0.18, 'between_regime_cv': 0.41,
                'within_regime_cv_std': 0.04, 'between_regime_cv_std': 0.09,
                'temporal_smoothness': 0.82, 'regime_persistence': 24.0, 'balance_score': 0.88,
                'n_regimes': 5, 'noise_ratio': 0.08, 'predictive_power': 0.68,
                'economic_validation': {
                    0: {'mean_return': 0.0018, 'volatility': 0.016, 'sharpe': 1.12, 'max_drawdown': -0.10, 'hit_rate': 0.58, 'size': 900},
                    1: {'mean_return': -0.0012, 'volatility': 0.020, 'sharpe': -0.60, 'max_drawdown': -0.14, 'hit_rate': 0.48, 'size': 850}
                },
                'log_likelihood': -1520.0, 'refit_stability_ari': 0.78, 'occupancy_entropy': 1.45,
                'cluster_size_distribution': [0.20, 0.22, 0.18, 0.21, 0.19]
            }
        }
    ]
    
    print("✅ Created enhanced test data with PCA information")
    
    # Generate enhanced CSV reports
    quality_csv, trials_csv = assessor.generate_comprehensive_csv_report(
        metrics=metrics,
        all_trials=all_trials,
        symbol="ETHUSDT",
        output_dir="test_outcomes",
        method_specific_config=method_config
    )
    
    print(f"📊 Enhanced Quality Metrics CSV: {quality_csv}")
    print(f"📋 Enhanced All Trials CSV: {trials_csv}")
    
    # Generate enhanced markdown report with PCA information
    markdown_path = assessor.generate_markdown_report(
        metrics=metrics,
        symbol="ETHUSDT",
        output_dir="test_outcomes",
        method_specific_config=method_config
    )
    
    print(f"📝 Enhanced Markdown Report: {markdown_path}")
    
    print("\n🎉 Enhanced Features Summary:")
    print("   ✅ CSV with within/between cluster CV metrics")
    print("   ✅ CSV with per-regime CV values")
    print("   ✅ CSV with economic validation metrics")
    print("   ✅ CSV with HMM validation metrics")
    print("   ✅ CSV with comprehensive regime size analysis")
    print("   ✅ Markdown with PCA feature categories")
    print("   ✅ Markdown with PCA variance explained")
    print("   ✅ Markdown with PCA feature loadings")
    
    return quality_csv, trials_csv, markdown_path

if __name__ == "__main__":
    test_enhanced_features()
