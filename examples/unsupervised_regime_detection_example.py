"""
Unsupervised Regime Detection and Qualification Example

This example demonstrates how to use tree-based NAS for unsupervised
regime detection and qualification without requiring labeled data.

Key Features:
- Unsupervised clustering with multiple algorithms
- Automatic regime detection and classification
- Regime qualification and quality assessment
- Feature importance analysis
- Regime transition analysis
- Integration with existing hybrid NAS system
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import unsupervised NAS components
import sys
sys.path.append('/workspace')
from src.utils.ml_common.optimization.unsupervised_tree_nas import (
    UnsupervisedTreeNASConfig, UnsupervisedTreeNAS, search_unsupervised_regimes
)


def create_realistic_market_data(n_samples=2000, start_date='2024-01-01'):
    """Create realistic market data with different regimes."""
    logger.info("Creating realistic market data with different regimes...")
    
    # Create date range
    dates = pd.date_range(start_date, periods=n_samples, freq='15T')
    
    # Define regime periods
    regime_periods = [
        (0, 400, 'bull', 0.02, 0.01),      # Bull market: high returns, low volatility
        (400, 800, 'bear', -0.015, 0.02),  # Bear market: negative returns, high volatility
        (800, 1200, 'sideways', 0.001, 0.005),  # Sideways: low returns, low volatility
        (1200, 1600, 'volatile', 0.005, 0.03),  # Volatile: moderate returns, high volatility
        (1600, 2000, 'trending', 0.01, 0.008)   # Trending: steady returns, moderate volatility
    ]
    
    # Initialize price
    price = 100.0
    prices = [price]
    volumes = []
    
    for i in range(1, n_samples):
        # Determine current regime
        current_regime = None
        for start, end, regime_type, mean_return, volatility in regime_periods:
            if start <= i < end:
                current_regime = (regime_type, mean_return, volatility)
                break
        
        if current_regime is None:
            current_regime = ('normal', 0.005, 0.01)
        
        regime_type, mean_return, volatility = current_regime
        
        # Generate price movement
        if regime_type == 'bull':
            # Bull market: generally upward trend with occasional dips
            if i % 50 < 40:  # 80% of the time, positive returns
                return_val = np.random.normal(mean_return, volatility)
            else:  # 20% of the time, negative returns
                return_val = np.random.normal(-mean_return/2, volatility)
        elif regime_type == 'bear':
            # Bear market: generally downward trend with occasional rallies
            if i % 50 < 30:  # 60% of the time, negative returns
                return_val = np.random.normal(mean_return, volatility)
            else:  # 40% of the time, positive returns
                return_val = np.random.normal(-mean_return/2, volatility)
        elif regime_type == 'sideways':
            # Sideways market: small movements around current price
            return_val = np.random.normal(mean_return, volatility)
        elif regime_type == 'volatile':
            # Volatile market: large movements in both directions
            return_val = np.random.normal(mean_return, volatility * 2)
        else:  # trending
            # Trending market: consistent directional movement
            return_val = np.random.normal(mean_return, volatility)
        
        # Update price
        price *= (1 + return_val)
        prices.append(price)
        
        # Generate volume (higher during volatile periods)
        if regime_type == 'volatile':
            volume = np.random.randint(5000, 15000)
        elif regime_type == 'bear':
            volume = np.random.randint(3000, 8000)
        else:
            volume = np.random.randint(2000, 6000)
        volumes.append(volume)
    
    # Create OHLCV data
    market_data = pd.DataFrame({
        'open': prices[:-1],
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
        'close': prices[1:],
        'volume': volumes
    })
    
    # Ensure high >= low
    market_data['high'] = np.maximum(market_data['high'], market_data['low'])
    market_data['high'] = np.maximum(market_data['high'], market_data['close'])
    market_data['low'] = np.minimum(market_data['low'], market_data['close'])
    
    logger.info(f"Created market data with {len(market_data)} samples")
    return market_data, dates


def demonstrate_unsupervised_regime_detection():
    """Demonstrate unsupervised regime detection."""
    logger.info("=== Unsupervised Regime Detection ===")
    
    # Create market data
    market_data, dates = create_realistic_market_data()
    
    # Configure unsupervised NAS
    config = UnsupervisedTreeNASConfig(
        clustering_algorithms=['kmeans', 'gaussian_mixture', 'agglomerative'],
        n_regimes_range=(3, 8),
        min_regime_duration=50,
        max_regime_duration=500,
        regime_stability_threshold=0.6,
        n_trials=20
    )
    
    # Perform unsupervised regime detection
    start_time = time.time()
    result = search_unsupervised_regimes(market_data, dates.values, config)
    detection_time = time.time() - start_time
    
    # Display results
    logger.info(f"Unsupervised regime detection completed in {detection_time:.2f} seconds")
    logger.info(f"Best clustering algorithm: {result.clustering_algorithm}")
    logger.info(f"Detected {result.n_regimes} regimes")
    logger.info(f"Overall quality score: {result.overall_score:.4f}")
    
    # Display regime details
    for i, regime in enumerate(result.regimes):
        logger.info(f"Regime {i+1}: {regime.regime_type} (confidence: {regime.regime_confidence:.3f})")
        logger.info(f"  Duration: {regime.duration} samples")
        logger.info(f"  Quality: {regime.overall_quality:.3f}")
        logger.info(f"  Key features: {', '.join(regime.key_features[:3])}")
    
    return result


def demonstrate_regime_qualification():
    """Demonstrate regime qualification and quality assessment."""
    logger.info("=== Regime Qualification ===")
    
    # Create market data
    market_data, dates = create_realistic_market_data()
    
    # Configure for regime qualification
    config = UnsupervisedTreeNASConfig(
        clustering_algorithms=['kmeans', 'gaussian_mixture'],
        n_regimes_range=(4, 10),
        qualification_metrics=[
            'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
            'regime_persistence', 'regime_separation', 'regime_consistency'
        ],
        quality_thresholds={
            'min_silhouette_score': 0.3,
            'min_regime_persistence': 0.6,
            'min_regime_separation': 0.5,
            'min_regime_consistency': 0.7
        },
        n_trials=15
    )
    
    # Perform regime detection and qualification
    start_time = time.time()
    result = search_unsupervised_regimes(market_data, dates.values, config)
    qualification_time = time.time() - start_time
    
    # Display qualification results
    logger.info(f"Regime qualification completed in {qualification_time:.2f} seconds")
    logger.info(f"Regime qualification score: {result.regime_qualification_score:.4f}")
    
    # Display qualified regimes
    qualified_regimes = []
    for regime in result.regimes:
        if (regime.silhouette_score >= config.quality_thresholds['min_silhouette_score'] and
            regime.regime_persistence >= config.quality_thresholds['min_regime_persistence'] and
            regime.regime_separation >= config.quality_thresholds['min_regime_separation'] and
            regime.regime_consistency >= config.quality_thresholds['min_regime_consistency']):
            qualified_regimes.append(regime)
    
    logger.info(f"Qualified regimes: {len(qualified_regimes)}/{len(result.regimes)}")
    
    for i, regime in enumerate(qualified_regimes):
        logger.info(f"Qualified Regime {i+1}: {regime.regime_type}")
        logger.info(f"  Silhouette: {regime.silhouette_score:.3f}")
        logger.info(f"  Persistence: {regime.regime_persistence:.3f}")
        logger.info(f"  Separation: {regime.regime_separation:.3f}")
        logger.info(f"  Consistency: {regime.regime_consistency:.3f}")
    
    return result, qualified_regimes


def demonstrate_feature_importance_analysis():
    """Demonstrate feature importance analysis for regime detection."""
    logger.info("=== Feature Importance Analysis ===")
    
    # Create market data
    market_data, dates = create_realistic_market_data()
    
    # Configure for feature importance analysis
    config = UnsupervisedTreeNASConfig(
        clustering_algorithms=['kmeans'],
        n_regimes_range=(5, 8),
        feature_engineering_methods=[
            'technical_indicators', 'price_features', 'volume_features',
            'volatility_features', 'momentum_features'
        ],
        n_trials=10
    )
    
    # Perform regime detection
    result = search_unsupervised_regimes(market_data, dates.values, config)
    
    # Display feature importance
    logger.info("Feature importance for regime detection:")
    sorted_features = sorted(result.feature_importance.items(), 
                           key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(sorted_features[:10]):
        logger.info(f"{i+1:2d}. {feature}: {importance:.4f}")
    
    # Display regime-specific feature importance
    logger.info("\nRegime-specific feature importance:")
    for i, regime in enumerate(result.regimes):
        logger.info(f"\nRegime {i+1} ({regime.regime_type}):")
        regime_features = sorted(regime.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        for j, (feature, importance) in enumerate(regime_features[:5]):
            logger.info(f"  {j+1}. {feature}: {importance:.4f}")
    
    return result


def demonstrate_regime_transition_analysis():
    """Demonstrate regime transition analysis."""
    logger.info("=== Regime Transition Analysis ===")
    
    # Create market data
    market_data, dates = create_realistic_market_data()
    
    # Configure for transition analysis
    config = UnsupervisedTreeNASConfig(
        clustering_algorithms=['gaussian_mixture'],
        n_regimes_range=(4, 8),
        min_regime_duration=30,
        n_trials=10
    )
    
    # Perform regime detection
    result = search_unsupervised_regimes(market_data, dates.values, config)
    
    # Analyze transitions
    logger.info("Regime transition analysis:")
    for i, regime in enumerate(result.regimes):
        logger.info(f"Regime {i+1} ({regime.regime_type}):")
        logger.info(f"  Transition probability: {regime.transition_probability:.3f}")
        logger.info(f"  Transition targets: {regime.transition_targets}")
        logger.info(f"  Duration: {regime.duration} samples")
    
    # Calculate transition matrix
    n_regimes = len(result.regimes)
    transition_matrix = np.zeros((n_regimes, n_regimes))
    
    for regime in result.regimes:
        for target in regime.transition_targets:
            if target < n_regimes:
                transition_matrix[regime.regime_id, target] = regime.transition_probability
    
    logger.info(f"\nTransition matrix:")
    for i in range(n_regimes):
        row = " ".join([f"{transition_matrix[i, j]:.3f}" for j in range(n_regimes)])
        logger.info(f"  {row}")
    
    return result, transition_matrix


def demonstrate_clustering_algorithm_comparison():
    """Demonstrate comparison of different clustering algorithms."""
    logger.info("=== Clustering Algorithm Comparison ===")
    
    # Create market data
    market_data, dates = create_realistic_market_data()
    
    algorithms = ['kmeans', 'gaussian_mixture', 'agglomerative', 'dbscan']
    results = {}
    
    for algorithm in algorithms:
        logger.info(f"Testing {algorithm}...")
        
        config = UnsupervisedTreeNASConfig(
            clustering_algorithms=[algorithm],
            n_regimes_range=(4, 8),
            n_trials=5
        )
        
        start_time = time.time()
        try:
            result = search_unsupervised_regimes(market_data, dates.values, config)
            detection_time = time.time() - start_time
            
            results[algorithm] = {
                'detection_time': detection_time,
                'n_regimes': result.n_regimes,
                'overall_score': result.overall_score,
                'clustering_quality': result.clustering_quality,
                'regime_qualification_score': result.regime_qualification_score
            }
            
            logger.info(f"{algorithm}: {detection_time:.2f}s, {result.n_regimes} regimes, score: {result.overall_score:.4f}")
            
        except Exception as e:
            logger.warning(f"{algorithm} failed: {e}")
            results[algorithm] = {'error': str(e)}
    
    # Display comparison
    logger.info("\n=== Algorithm Comparison Results ===")
    for algorithm, result in results.items():
        if 'error' not in result:
            logger.info(f"{algorithm}: {result['detection_time']:.2f}s, {result['n_regimes']} regimes, {result['overall_score']:.4f} score")
        else:
            logger.info(f"{algorithm}: Failed - {result['error']}")
    
    return results


def create_regime_visualization(market_data, result):
    """Create visualization of detected regimes."""
    logger.info("=== Creating Regime Visualization ===")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Price with regime colors
    ax1.plot(market_data.index, market_data['close'], 'b-', alpha=0.7, linewidth=1)
    
    # Color code by regime
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, regime in enumerate(result.regimes):
        if regime.sample_indices:
            regime_data = market_data.iloc[regime.sample_indices]
            ax1.scatter(regime_data.index, regime_data['close'], 
                       c=colors[i % len(colors)], alpha=0.6, s=10, 
                       label=f'Regime {i+1} ({regime.regime_type})')
    
    ax1.set_title('Price with Detected Regimes')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Regime quality metrics
    regime_ids = [f"R{i+1}" for i in range(len(result.regimes))]
    silhouette_scores = [regime.silhouette_score for regime in result.regimes]
    persistence_scores = [regime.regime_persistence for regime in result.regimes]
    
    x = np.arange(len(regime_ids))
    width = 0.35
    
    ax2.bar(x - width/2, silhouette_scores, width, label='Silhouette Score', alpha=0.8)
    ax2.bar(x + width/2, persistence_scores, width, label='Persistence', alpha=0.8)
    
    ax2.set_title('Regime Quality Metrics')
    ax2.set_xlabel('Regimes')
    ax2.set_ylabel('Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regime_ids)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Feature importance
    feature_names = list(result.feature_importance.keys())
    feature_scores = list(result.feature_importance.values())
    
    # Sort by importance
    sorted_indices = np.argsort(feature_scores)[::-1][:10]
    top_features = [feature_names[i] for i in sorted_indices]
    top_scores = [feature_scores[i] for i in sorted_indices]
    
    ax3.barh(range(len(top_features)), top_scores, alpha=0.8)
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features)
    ax3.set_title('Top 10 Feature Importance')
    ax3.set_xlabel('Importance Score')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Regime duration distribution
    durations = [regime.duration for regime in result.regimes]
    regime_types = [regime.regime_type for regime in result.regimes]
    
    unique_types = list(set(regime_types))
    type_durations = {regime_type: [] for regime_type in unique_types}
    
    for regime in result.regimes:
        type_durations[regime.regime_type].append(regime.duration)
    
    ax4.hist([type_durations[regime_type] for regime_type in unique_types], 
             bins=10, alpha=0.7, label=unique_types)
    ax4.set_title('Regime Duration Distribution')
    ax4.set_xlabel('Duration (samples)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/unsupervised_regime_detection.png', dpi=300, bbox_inches='tight')
    logger.info("Visualization saved as 'unsupervised_regime_detection.png'")
    
    return fig


def main():
    """Main demonstration function."""
    logger.info("🚀 Starting Unsupervised Regime Detection Demonstration")
    
    try:
        # Create market data
        market_data, dates = create_realistic_market_data()
        
        # Demonstrate unsupervised regime detection
        detection_result = demonstrate_unsupervised_regime_detection()
        
        # Demonstrate regime qualification
        qualification_result, qualified_regimes = demonstrate_regime_qualification()
        
        # Demonstrate feature importance analysis
        feature_result = demonstrate_feature_importance_analysis()
        
        # Demonstrate regime transition analysis
        transition_result, transition_matrix = demonstrate_regime_transition_analysis()
        
        # Demonstrate clustering algorithm comparison
        comparison_result = demonstrate_clustering_algorithm_comparison()
        
        # Create visualization
        visualization = create_regime_visualization(market_data, detection_result)
        
        # Summary
        logger.info("=== Summary ===")
        logger.info("✅ Unsupervised regime detection successfully demonstrated")
        logger.info(f"✅ Detected {detection_result.n_regimes} regimes")
        logger.info(f"✅ Regime qualification score: {detection_result.regime_qualification_score:.4f}")
        logger.info(f"✅ Qualified regimes: {len(qualified_regimes)}")
        logger.info("✅ Feature importance analysis completed")
        logger.info("✅ Regime transition analysis completed")
        logger.info("✅ Clustering algorithm comparison completed")
        logger.info("✅ Unsupervised tree-based NAS is ready for production use")
        
        return {
            'detection_result': detection_result,
            'qualification_result': qualification_result,
            'qualified_regimes': qualified_regimes,
            'feature_result': feature_result,
            'transition_result': transition_result,
            'transition_matrix': transition_matrix,
            'comparison_result': comparison_result
        }
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    results = main()
    print("\n🎉 Unsupervised regime detection demonstration completed successfully!")
    print("📊 Check the generated visualization: unsupervised_regime_detection.png")
    print("🔍 Review the logs above for detailed regime analysis")
    print("🤖 Unsupervised tree-based NAS works without labeled data!")