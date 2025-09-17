"""
Refined Example: Market Dimension Discovery for Regime Clustering

This example demonstrates the refined approach to market regime research:
1. Use existing feature engineering pipeline
2. Discover implicit market dimensions from features
3. Assess economic significance of dimensions
4. Apply clustering with dimension awareness
5. Validate regime quality
6. Integrate with HMM systems (timing comparison)

Research Question: What implicit market dimensions are captured by our features
and which are most economically significant for trading regime identification?
"""

import numpy as np
import pandas as pd
import asyncio
from pathlib import Path
import logging

# Import the refined regime clustering framework
from src.research.clusters import (
    MarketDimensionAnalyzer, DimensionAnalysisConfig, MarketDimension,
    RegimeClusterer, ClusteringConfig, ClusteringMethod,
    RegimeFeatureImportance, ImportanceConfig, ImportanceMethod,
    RegimeValidationMetrics, ValidationConfig,
    HMMIntegrationLayer, IntegrationConfig, IntegrationMethod,
    RegimeVisualization, VisualizationConfig
)

from src.utils.logger import system_logger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = system_logger.getChild('RefinedRegimeResearch')


def generate_sample_data_with_features(n_samples: int = 1000) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate sample market data with realistic feature patterns.
    Simulates the output of your feature engineering pipeline.
    """
    logger.info(f"🎲 Generating sample market data with features ({n_samples} samples)")
    
    # Create datetime index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
    
    # Generate base OHLCV data with regime-like behavior
    np.random.seed(42)
    
    # Three regimes with different characteristics
    regime_1_samples = n_samples // 3  # Low vol, trending
    regime_2_samples = n_samples // 3  # High vol, sideways
    regime_3_samples = n_samples - regime_1_samples - regime_2_samples  # Medium vol, mean-reverting
    
    # Regime 1: Low volatility, trending (momentum dimension dominant)
    price_1 = 100 + np.cumsum(np.random.normal(0.01, 0.005, regime_1_samples))
    vol_1 = np.random.lognormal(10, 0.3, regime_1_samples)
    
    # Regime 2: High volatility, sideways (volatility dimension dominant)
    price_2 = price_1[-1] + np.cumsum(np.random.normal(0, 0.02, regime_2_samples))
    vol_2 = np.random.lognormal(12, 1.2, regime_2_samples)
    
    # Regime 3: Medium volatility, mean-reverting (correlation dimension dominant)
    price_3 = price_2[-1] + np.cumsum(np.random.normal(-0.005, 0.01, regime_3_samples))
    vol_3 = np.random.lognormal(11, 0.7, regime_3_samples)
    
    # Combine regimes
    close_prices = np.concatenate([price_1, price_2, price_3])
    volumes = np.concatenate([vol_1, vol_2, vol_3])
    
    # Generate OHLC
    noise = np.random.normal(0, 0.001, n_samples)
    high_prices = close_prices * (1 + np.abs(noise) + 0.001)
    low_prices = close_prices * (1 - np.abs(noise) - 0.001)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # Create base market data
    market_data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    # Simulate comprehensive feature engineering output
    features = market_data.copy()
    
    # Returns and basic features
    features['returns'] = features['close'].pct_change()
    features['log_returns'] = np.log(features['close']).diff()
    
    # Momentum dimension features (simulating existing pipeline output)
    for period in [5, 10, 20, 50]:
        features[f'sma_{period}'] = features['close'].rolling(period).mean()
        features[f'ema_{period}'] = features['close'].ewm(span=period).mean()
        features[f'roc_{period}'] = features['close'].pct_change(period)
        features[f'momentum_{period}'] = features['close'] / features['close'].shift(period) - 1
    
    # Volatility dimension features
    for period in [10, 20, 50]:
        features[f'volatility_{period}'] = features['returns'].rolling(period).std()
        features[f'atr_{period}'] = (features['high'] - features['low']).rolling(period).mean()
        features[f'vol_ratio_{period}'] = features[f'volatility_{period}'] / features['volatility_20']
    
    # Volume dimension features
    for period in [5, 10, 20]:
        features[f'volume_sma_{period}'] = features['volume'].rolling(period).mean()
        features[f'volume_ratio_{period}'] = features['volume'] / features[f'volume_sma_{period}']
        features[f'vwap_{period}'] = (features['close'] * features['volume']).rolling(period).sum() / features['volume'].rolling(period).sum()
    
    # Liquidity dimension features (proxies)
    features['spread_proxy'] = (features['high'] - features['low']) / features['close']
    features['impact_proxy'] = features['volume'] / (features['high'] - features['low'])
    for period in [10, 20]:
        features[f'liquidity_proxy_{period}'] = features['volume'].rolling(period).mean() / features['spread_proxy'].rolling(period).mean()
    
    # Microstructure dimension features (proxies from OHLCV)
    features['trade_intensity_proxy'] = features['volume'] / (features['high'] - features['low'])
    features['order_flow_proxy'] = (features['close'] - features['low']) / (features['high'] - features['low'])
    for period in [5, 10]:
        features[f'microstructure_proxy_{period}'] = features['order_flow_proxy'].rolling(period).mean()
    
    # Correlation dimension features
    for lag in [1, 5, 10]:
        features[f'autocorr_lag_{lag}'] = features['returns'].rolling(50).apply(lambda x: x.autocorr(lag))
    for period in [20, 50]:
        features[f'vol_price_corr_{period}'] = features['volume'].rolling(period).corr(features['close'])
        features[f'high_low_corr_{period}'] = features['high'].rolling(period).corr(features['low'])
    
    # Fill NaN values
    features = features.fillna(method='ffill').fillna(0)
    
    # True regime labels (unknown in practice)
    true_regimes = np.concatenate([
        np.full(regime_1_samples, 0),  # Momentum regime
        np.full(regime_2_samples, 1),  # Volatility regime
        np.full(regime_3_samples, 2)   # Correlation/mean-reversion regime
    ])
    
    logger.info(f"✅ Generated {len(features.columns)} features across {len(np.unique(true_regimes))} regimes")
    
    return features, true_regimes


async def research_workflow_dimension_first(market_data: pd.DataFrame) -> dict:
    """
    Research workflow: Discover dimensions first, then apply HMM.
    
    This approach answers: What dimensions should we focus on before applying HMM?
    """
    logger.info("🔬 Research Workflow: DIMENSION FIRST → HMM")
    
    results = {}
    
    # Step 1: Discover implicit market dimensions from existing features
    logger.info("📊 Step 1: Discovering implicit market dimensions")
    dimension_analyzer = MarketDimensionAnalyzer(DimensionAnalysisConfig(
        use_pca=True,
        use_mutual_information=True,
        use_feature_importance=True
    ))
    
    dimension_results = dimension_analyzer.analyze_all_dimensions(
        market_data, use_existing_features=True
    )
    
    # Get top economically significant dimensions
    top_dimensions = dimension_analyzer.get_top_dimensions(3)
    results['dimension_analysis'] = {
        'all_dimensions': {dim.value: metrics.to_dict() for dim, metrics in dimension_results.items()},
        'top_dimensions': [(dim.value, metrics.to_dict()) for dim, metrics in top_dimensions],
        'economic_significance_ranking': sorted(
            [(dim.value, metrics.metrics.get('regime_separability', 0)) 
             for dim, metrics in dimension_results.items()],
            key=lambda x: x[1], reverse=True
        )
    }
    
    logger.info("📊 Top economically significant dimensions:")
    for dim_name, separability in results['dimension_analysis']['economic_significance_ranking'][:3]:
        logger.info(f"   {dim_name}: economic separability = {separability:.3f}")
    
    # Step 2: Focus clustering on most important dimension features
    logger.info("🔍 Step 2: Clustering with dimension-aware feature selection")
    
    # Extract features from top dimensions
    top_dimension_features = []
    for dimension, metrics in top_dimensions:
        top_dimension_features.extend(metrics.feature_names)
    
    # Use top dimension features for clustering
    selected_features = market_data[top_dimension_features].fillna(0)
    
    clusterer = RegimeClusterer(ClusteringConfig(
        n_clusters=3,  # We know there are 3 regimes
        ensemble_methods=[ClusteringMethod.KMEANS, ClusteringMethod.GMM, ClusteringMethod.HIERARCHICAL]
    ))
    
    clustering_results = clusterer.run_all_methods(
        selected_features.values, 
        analyze_dimensions=True,
        feature_names=list(selected_features.columns)
    )
    
    best_method, best_result = clusterer.get_best_method()
    results['clustering'] = {
        'best_method': best_method.value,
        'n_clusters': best_result.n_clusters,
        'silhouette_score': best_result.metrics.get('silhouette_score', 0),
        'intrinsic_dimensionality': best_result.metadata.get('dimension_analysis', {}).get('intrinsic_dimensionality_estimate', 'N/A'),
        'regime_labels': best_result.labels
    }
    
    logger.info(f"✅ Best clustering: {best_method.value} with {best_result.n_clusters} clusters")
    logger.info(f"   Silhouette score: {best_result.metrics.get('silhouette_score', 0):.3f}")
    logger.info(f"   Intrinsic dimensionality: {results['clustering']['intrinsic_dimensionality']}")
    
    return results


async def research_workflow_hmm_first(market_data: pd.DataFrame) -> dict:
    """
    Research workflow: Apply HMM first, then analyze dimensions.
    
    This approach answers: What dimensions does HMM implicitly discover?
    """
    logger.info("🔬 Research Workflow: HMM FIRST → Dimension Analysis")
    
    results = {}
    
    # Step 1: Apply HMM to discover regimes
    logger.info("🔄 Step 1: Applying HMM regime discovery")
    
    # Simulate HMM results (in practice, use your existing HMM system)
    # For this example, we'll create mock HMM results
    n_samples = len(market_data)
    mock_hmm_regimes = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.35, 0.25])
    
    results['hmm_results'] = {
        'regime_labels': mock_hmm_regimes,
        'n_regimes': len(np.unique(mock_hmm_regimes)),
        'regime_distribution': dict(zip(*np.unique(mock_hmm_regimes, return_counts=True)))
    }
    
    logger.info(f"🔄 HMM discovered {results['hmm_results']['n_regimes']} regimes")
    
    # Step 2: Analyze which dimensions are captured by HMM regimes
    logger.info("📊 Step 2: Analyzing dimensions captured by HMM regimes")
    
    dimension_analyzer = MarketDimensionAnalyzer()
    dimension_results = dimension_analyzer.analyze_all_dimensions(
        market_data, regime_labels=mock_hmm_regimes, use_existing_features=True
    )
    
    results['post_hmm_dimension_analysis'] = {
        'dimensions_by_discriminability': sorted(
            [(dim.value, metrics.regime_discriminability) for dim, metrics in dimension_results.items()],
            key=lambda x: x[1], reverse=True
        ),
        'dimensions_by_economic_significance': sorted(
            [(dim.value, metrics.metrics.get('regime_separability', 0)) 
             for dim, metrics in dimension_results.items()],
            key=lambda x: x[1], reverse=True
        )
    }
    
    logger.info("📊 Dimensions best captured by HMM:")
    for dim_name, discriminability in results['post_hmm_dimension_analysis']['dimensions_by_discriminability'][:3]:
        logger.info(f"   {dim_name}: discriminability = {discriminability:.3f}")
    
    return results


async def research_workflow_comparative(market_data: pd.DataFrame, true_regimes: np.ndarray) -> dict:
    """
    Comparative analysis: Compare dimension-first vs HMM-first approaches.
    """
    logger.info("🔬 Research Workflow: COMPARATIVE ANALYSIS")
    
    # Run both workflows
    dimension_first_results = await research_workflow_dimension_first(market_data)
    hmm_first_results = await research_workflow_hmm_first(market_data)
    
    # Compare results
    comparison = {}
    
    # Compare regime discovery quality
    if 'regime_labels' in dimension_first_results.get('clustering', {}):
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        dim_first_labels = dimension_first_results['clustering']['regime_labels']
        hmm_first_labels = hmm_first_results['hmm_results']['regime_labels']
        
        # Agreement with true regimes
        dim_first_agreement = adjusted_rand_score(true_regimes, dim_first_labels)
        hmm_first_agreement = adjusted_rand_score(true_regimes, hmm_first_labels)
        
        # Agreement between methods
        method_agreement = adjusted_rand_score(dim_first_labels, hmm_first_labels)
        
        comparison['regime_quality'] = {
            'dimension_first_vs_true': float(dim_first_agreement),
            'hmm_first_vs_true': float(hmm_first_agreement),
            'method_agreement': float(method_agreement),
            'better_method': 'dimension_first' if dim_first_agreement > hmm_first_agreement else 'hmm_first'
        }
    
    # Compare dimension discovery
    comparison['dimension_insights'] = {
        'dimension_first_top_dims': [dim for dim, _ in dimension_first_results['dimension_analysis']['economic_significance_ranking'][:3]],
        'hmm_first_captured_dims': [dim for dim, _ in hmm_first_results['post_hmm_dimension_analysis']['dimensions_by_discriminability'][:3]],
        'consistent_dimensions': list(set(
            [dim for dim, _ in dimension_first_results['dimension_analysis']['economic_significance_ranking'][:3]]
        ).intersection(set(
            [dim for dim, _ in hmm_first_results['post_hmm_dimension_analysis']['dimensions_by_discriminability'][:3]]
        )))
    }
    
    logger.info("🔬 Comparative Analysis Results:")
    logger.info(f"   Better regime discovery: {comparison['regime_quality']['better_method']}")
    logger.info(f"   Method agreement: {comparison['regime_quality']['method_agreement']:.3f}")
    logger.info(f"   Consistent dimensions: {comparison['dimension_insights']['consistent_dimensions']}")
    
    return {
        'dimension_first': dimension_first_results,
        'hmm_first': hmm_first_results,
        'comparison': comparison
    }


async def main():
    """Main research workflow."""
    logger.info("🚀 Starting Refined Market Regime Research")
    logger.info("=" * 60)
    
    try:
        # Generate sample data with comprehensive features
        market_data, true_regimes = generate_sample_data_with_features(1000)
        
        logger.info("🎯 Research Question: What implicit market dimensions are most economically significant?")
        logger.info(f"📊 Data: {market_data.shape[0]} samples, {market_data.shape[1]} features")
        logger.info(f"🎯 True regimes: {len(np.unique(true_regimes))} (Momentum, Volatility, Correlation)")
        logger.info("")
        
        # Run comparative research workflow
        results = await research_workflow_comparative(market_data, true_regimes)
        
        # Summary of findings
        logger.info("🎉 RESEARCH FINDINGS SUMMARY")
        logger.info("=" * 60)
        
        # Dimension insights
        comparison = results['comparison']
        logger.info("📊 DIMENSION DISCOVERY:")
        logger.info(f"   Dimension-First Top 3: {comparison['dimension_insights']['dimension_first_top_dims']}")
        logger.info(f"   HMM-Captured Top 3: {comparison['dimension_insights']['hmm_first_captured_dims']}")
        logger.info(f"   Consistent Dimensions: {comparison['dimension_insights']['consistent_dimensions']}")
        logger.info("")
        
        # Regime quality
        logger.info("🎯 REGIME DISCOVERY QUALITY:")
        logger.info(f"   Dimension-First Agreement: {comparison['regime_quality']['dimension_first_vs_true']:.3f}")
        logger.info(f"   HMM-First Agreement: {comparison['regime_quality']['hmm_first_vs_true']:.3f}")
        logger.info(f"   Better Approach: {comparison['regime_quality']['better_method']}")
        logger.info(f"   Method Agreement: {comparison['regime_quality']['method_agreement']:.3f}")
        logger.info("")
        
        # Economic significance insights
        dim_first = results['dimension_first']
        logger.info("💰 ECONOMIC SIGNIFICANCE RANKING:")
        for i, (dim_name, separability) in enumerate(dim_first['dimension_analysis']['economic_significance_ranking'][:5], 1):
            logger.info(f"   {i}. {dim_name}: {separability:.3f}")
        logger.info("")
        
        # Recommendations
        logger.info("💡 RESEARCH RECOMMENDATIONS:")
        
        better_method = comparison['regime_quality']['better_method']
        if better_method == 'dimension_first':
            logger.info("   ✅ Use DIMENSION-FIRST approach:")
            logger.info("     1. Discover economically significant dimensions first")
            logger.info("     2. Focus feature engineering on top dimensions")
            logger.info("     3. Apply clustering/HMM to dimension-selected features")
        else:
            logger.info("   ✅ Use HMM-FIRST approach:")
            logger.info("     1. Apply HMM to discover regimes")
            logger.info("     2. Analyze which dimensions HMM captures")
            logger.info("     3. Enhance HMM with additional dimension insights")
        
        consistent_dims = comparison['dimension_insights']['consistent_dimensions']
        if consistent_dims:
            logger.info(f"   🎯 Focus on consistent dimensions: {consistent_dims}")
            logger.info("     These dimensions are important regardless of approach")
        
        logger.info("")
        logger.info("🔬 Next Steps:")
        logger.info("   1. Apply this workflow to your real market data")
        logger.info("   2. Integrate with your existing feature engineering pipeline")
        logger.info("   3. Use findings to enhance HMM regime discovery")
        logger.info("   4. Validate economic significance in backtesting")
        
        return {
            'success': True,
            'key_findings': {
                'better_approach': better_method,
                'top_dimensions': comparison['dimension_insights']['dimension_first_top_dims'],
                'consistent_dimensions': consistent_dims,
                'regime_quality': comparison['regime_quality']
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Research workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    # Run the refined research workflow
    result = asyncio.run(main())
    
    if result['success']:
        print("\n✅ Research completed successfully!")
        print("Key insights available in the logs above.")
    else:
        print(f"\n❌ Research failed: {result['error']}")