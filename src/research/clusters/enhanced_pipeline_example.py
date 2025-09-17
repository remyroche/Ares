"""
Enhanced Pipeline Example: Complete Economic Relevance Analysis

This example demonstrates the enhanced pipeline that answers your key research questions:

1. Which implicit dimensions (beyond volume/volatility) influence price action?
2. How do dimensions support momentum vs mean reversion strategies?
3. Which dimensions modulate volatility and affect price dynamics?
4. What is the economic significance of each dimension for trading?

Pipeline: Many Features → Statistical Analysis → Market Dimensions → Economic Relevance → Clustering
"""

import numpy as np
import pandas as pd
import asyncio
from pathlib import Path
import logging

from src.research.clusters import (
    MarketDimensionAnalyzer, DimensionAnalysisConfig,
    RegimeClusterer, ClusteringConfig, ClusteringMethod,
    RegimeValidationMetrics, ValidationConfig
)

from src.utils.logger import system_logger

logging.basicConfig(level=logging.INFO)
logger = system_logger.getChild('EnhancedPipelineExample')


def generate_comprehensive_feature_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate comprehensive feature data simulating your feature engineering pipeline.
    
    This simulates many features to capture various market dimensions, including
    some beyond traditional volume/volatility that might influence price action.
    """
    logger.info(f"🎲 Generating comprehensive feature data ({n_samples} samples)")
    
    # Create datetime index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
    
    # Generate base OHLCV with realistic patterns
    np.random.seed(42)
    
    # Base price evolution
    price_base = 100
    price_trend = 0.0001  # Small upward trend
    price_noise = 0.01
    
    prices = [price_base]
    volumes = []
    
    for i in range(n_samples - 1):
        # Price evolution with regime-like behavior
        if i < n_samples // 3:  # Momentum regime
            trend_strength = 0.002
            vol_level = 0.008
        elif i < 2 * n_samples // 3:  # High volatility regime  
            trend_strength = 0.0
            vol_level = 0.025
        else:  # Mean reversion regime
            trend_strength = -0.001
            vol_level = 0.012
        
        price_change = trend_strength + np.random.normal(0, vol_level)
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
        
        # Volume with microstructure effects
        base_volume = 1000000
        volume_noise = np.random.lognormal(0, 0.5)
        # Volume increases with volatility and near significant levels
        vol_multiplier = 1 + abs(price_change) * 10
        volumes.append(base_volume * volume_noise * vol_multiplier)
    
    prices = np.array(prices)
    volumes = np.array(volumes)
    
    # Generate OHLC
    noise = np.random.normal(0, 0.001, n_samples)
    high_prices = prices * (1 + np.abs(noise) + 0.002)
    low_prices = prices * (1 - np.abs(noise) - 0.002)
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]
    
    # Create comprehensive feature set
    market_data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Generate MANY features to capture various dimensions
    features = market_data.copy()
    
    # === MOMENTUM DIMENSION FEATURES ===
    for period in [5, 10, 20, 50, 100]:
        features[f'sma_{period}'] = features['close'].rolling(period).mean()
        features[f'ema_{period}'] = features['close'].ewm(span=period).mean()
        features[f'roc_{period}'] = features['close'].pct_change(period)
        features[f'momentum_{period}'] = features['close'] / features['close'].shift(period) - 1
        features[f'macd_{period}'] = features[f'ema_{period//2}'] - features[f'ema_{period}']
    
    # RSI family
    for period in [14, 21, 50]:
        delta = features['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta).where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # === VOLATILITY DIMENSION FEATURES ===
    features['returns'] = features['close'].pct_change()
    for period in [5, 10, 20, 50, 100]:
        features[f'volatility_{period}'] = features['returns'].rolling(period).std()
        features[f'atr_{period}'] = (features['high'] - features['low']).rolling(period).mean()
        features[f'vol_ratio_{period}'] = features[f'volatility_{period}'] / features['volatility_20']
        
        # GARCH-like features
        features[f'vol_clustering_{period}'] = features[f'volatility_{period}'].rolling(period).std()
    
    # Garman-Klass volatility
    features['gk_volatility'] = np.sqrt(
        0.5 * (np.log(features['high'] / features['low'])) ** 2 - 
        (2 * np.log(2) - 1) * (np.log(features['close'] / features['open'])) ** 2
    )
    
    # === VOLUME DIMENSION FEATURES ===
    for period in [5, 10, 20, 50]:
        features[f'volume_sma_{period}'] = features['volume'].rolling(period).mean()
        features[f'volume_ratio_{period}'] = features['volume'] / features[f'volume_sma_{period}']
        features[f'vwap_{period}'] = (features['close'] * features['volume']).rolling(period).sum() / features['volume'].rolling(period).sum()
        features[f'volume_roc_{period}'] = features['volume'].pct_change(period)
    
    # On-Balance Volume
    features['obv'] = (features['volume'] * np.sign(features['close'].diff())).cumsum()
    
    # === LIQUIDITY DIMENSION FEATURES ===
    features['spread_proxy'] = (features['high'] - features['low']) / features['close']
    features['impact_proxy'] = features['volume'] / (features['high'] - features['low'])
    for period in [10, 20, 50]:
        features[f'liquidity_proxy_{period}'] = features['volume'].rolling(period).mean() / features['spread_proxy'].rolling(period).mean()
        features[f'amihud_illiquidity_{period}'] = (abs(features['returns']) / features['volume']).rolling(period).mean()
    
    # === MICROSTRUCTURE DIMENSION FEATURES ===
    features['trade_intensity_proxy'] = features['volume'] / (features['high'] - features['low'])
    features['order_flow_proxy'] = (features['close'] - features['low']) / (features['high'] - features['low'])
    features['price_efficiency_proxy'] = features['close'].rolling(20).std() / features['close'].rolling(20).mean()
    
    for period in [5, 10, 20]:
        features[f'microstructure_proxy_{period}'] = features['order_flow_proxy'].rolling(period).mean()
        features[f'tick_rule_{period}'] = np.sign(features['close'].diff()).rolling(period).mean()
    
    # === CORRELATION DIMENSION FEATURES ===
    for lag in [1, 5, 10, 20]:
        features[f'autocorr_lag_{lag}'] = features['returns'].rolling(50).apply(lambda x: x.autocorr(lag))
    
    for period in [20, 50, 100]:
        features[f'vol_price_corr_{period}'] = features['volume'].rolling(period).corr(features['close'])
        features[f'high_low_corr_{period}'] = features['high'].rolling(period).corr(features['low'])
        features[f'return_vol_corr_{period}'] = features['returns'].rolling(period).corr(features[f'volatility_{period}'])
    
    # === ADVANCED FEATURES (potentially new dimensions) ===
    
    # Fractal dimension proxy
    for period in [20, 50]:
        features[f'fractal_dimension_{period}'] = features['close'].rolling(period).apply(
            lambda x: len(x) / (1 + np.log(len(x))) if len(x) > 1 else 1
        )
    
    # Hurst exponent proxy
    for period in [50, 100]:
        features[f'hurst_proxy_{period}'] = features['returns'].rolling(period).apply(
            lambda x: 0.5 + np.corrcoef(np.arange(len(x)), np.cumsum(x))[0,1] * 0.5 if len(x) > 10 else 0.5
        )
    
    # Market stress indicators
    features['market_stress'] = (features['volatility_20'] > features['volatility_20'].rolling(100).quantile(0.8)).astype(int)
    features['volume_stress'] = (features['volume'] > features['volume'].rolling(100).quantile(0.8)).astype(int)
    
    # Cross-timeframe alignment (simulated)
    for ratio in [(5, 20), (20, 50), (50, 100)]:
        short_ma = features[f'sma_{ratio[0]}']
        long_ma = features[f'sma_{ratio[1]}']
        features[f'ma_alignment_{ratio[0]}_{ratio[1]}'] = (short_ma > long_ma).astype(int)
        features[f'ma_divergence_{ratio[0]}_{ratio[1]}'] = (short_ma - long_ma) / long_ma
    
    # Regime transition indicators
    features['volatility_regime_change'] = (features['volatility_20'].diff().abs() > features['volatility_20'].rolling(50).std()).astype(int)
    features['volume_regime_change'] = (features['volume'].pct_change().abs() > features['volume'].pct_change().rolling(50).std()).astype(int)
    
    # Market efficiency proxies
    for period in [10, 20, 50]:
        features[f'efficiency_ratio_{period}'] = abs(features['close'].diff(period)) / features['close'].rolling(period).apply(lambda x: np.sum(np.abs(np.diff(x))))
        features[f'random_walk_deviation_{period}'] = features['close'] / (features['close'].shift(period) * (1 + features['returns'].rolling(period).mean() * period))
    
    # Information flow proxies
    features['price_volume_sync'] = features['returns'].rolling(10).corr(features['volume'].pct_change())
    features['information_arrival_proxy'] = features['volume'] * abs(features['returns'])
    
    # Fill NaN values
    features = features.fillna(method='ffill').fillna(0)
    
    logger.info(f"✅ Generated {len(features.columns)} comprehensive features")
    logger.info(f"   📊 Feature categories: momentum, volatility, volume, liquidity, microstructure, correlation, advanced")
    
    return features


async def run_enhanced_pipeline_analysis(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Run the enhanced pipeline with economic relevance analysis."""
    
    logger.info("🚀 Starting Enhanced Pipeline Analysis")
    logger.info("=" * 60)
    logger.info("Pipeline: Many Features → Statistical Analysis → Market Dimensions → Economic Relevance → Clustering")
    logger.info("")
    
    # Initialize analyzer
    analyzer = MarketDimensionAnalyzer()
    
    # Run complete coherent pipeline with economic relevance
    pipeline_results = analyzer.analyze_coherent_pipeline(market_data)
    
    # Extract key insights
    insights = {}
    
    # 1. Feature generation insights
    feature_gen = pipeline_results.get('feature_generation', {})
    insights['feature_generation'] = {
        'total_features': feature_gen.get('n_features_generated', 0),
        'feature_coverage': 'comprehensive' if feature_gen.get('n_features_generated', 0) > 50 else 'limited'
    }
    
    # 2. Statistical analysis insights  
    statistical = pipeline_results.get('statistical_analysis', {})
    if 'results' in statistical:
        pca_result = statistical['results'].get('principal_component_analysis', {})
        insights['statistical_analysis'] = {
            'intrinsic_dimensionality': pca_result.get('n_components', 'N/A'),
            'variance_explained': pca_result.get('explained_variance', 'N/A'),
            'dimensionality_reduction_effective': pca_result.get('explained_variance', 0) > 0.9
        }
    
    # 3. Market dimension insights
    market_dims = pipeline_results.get('market_dimensions', {})
    insights['market_dimensions'] = {
        'dimensions_discovered': market_dims.get('dimensions_discovered', []),
        'top_3_dimensions': [dim for dim, score in market_dims.get('top_dimensions', [])[:3]]
    }
    
    # 4. Economic relevance insights (KEY NEW STEP)
    economic_relevance = pipeline_results.get('economic_relevance', {})
    if 'beyond_volume_volatility_insights' in economic_relevance:
        beyond_vol_vol = economic_relevance['beyond_volume_volatility_insights']
        insights['economic_relevance'] = {
            'dimensions_beyond_vol_volatility': list(beyond_vol_vol.keys()),
            'new_price_action_influences': {
                dim: data['key_influences'] 
                for dim, data in beyond_vol_vol.items()
            },
            'novel_trading_applications': {
                dim: data['trading_applications']
                for dim, data in beyond_vol_vol.items()
            }
        }
        
        # Log key findings
        logger.info("🎯 KEY RESEARCH FINDINGS:")
        logger.info("=" * 40)
        
        if beyond_vol_vol:
            logger.info(f"✅ Found {len(beyond_vol_vol)} dimensions beyond volume/volatility with price action influence:")
            for dim, data in beyond_vol_vol.items():
                logger.info(f"   🔍 {dim.upper()}: relevance={data['relevance_score']:.3f}")
                logger.info(f"      - Key influences: {', '.join(data['key_influences'])}")
                logger.info(f"      - Trading applications: {', '.join(data['trading_applications'][:2])}")
                logger.info("")
        else:
            logger.info("⚠️ No dimensions beyond volume/volatility show significant price action influence")
            logger.info("   - Volume and volatility remain the primary price action drivers")
            logger.info("   - Consider expanding feature engineering to capture more market dynamics")
            logger.info("")
    
    # 5. Pipeline coherence insights
    coherence = pipeline_results.get('coherence_analysis', {})
    insights['pipeline_coherence'] = {
        'overall_score': coherence.get('overall_coherence_score', 0),
        'recommendations': coherence.get('recommendations', []),
        'pipeline_quality': 'strong' if coherence.get('overall_coherence_score', 0) > 0.7 else 'moderate' if coherence.get('overall_coherence_score', 0) > 0.5 else 'weak'
    }
    
    return {
        'pipeline_results': pipeline_results,
        'insights': insights
    }


async def demonstrate_economic_relevance_discovery():
    """Demonstrate the discovery of economically relevant dimensions."""
    
    logger.info("🔬 RESEARCH QUESTION: Which dimensions beyond volume/volatility influence price action?")
    logger.info("")
    
    # Generate comprehensive feature data
    market_data = generate_comprehensive_feature_data(1000)
    
    # Run enhanced pipeline analysis
    results = await run_enhanced_pipeline_analysis(market_data)
    insights = results['insights']
    
    # Analysis Summary
    logger.info("📊 ANALYSIS SUMMARY:")
    logger.info("=" * 40)
    logger.info(f"Features Generated: {insights['feature_generation']['total_features']}")
    logger.info(f"Intrinsic Dimensionality: {insights['statistical_analysis']['intrinsic_dimensionality']}")
    logger.info(f"Market Dimensions: {len(insights['market_dimensions']['dimensions_discovered'])}")
    logger.info(f"Pipeline Quality: {insights['pipeline_coherence']['pipeline_quality']}")
    logger.info("")
    
    # Economic Relevance Findings
    econ_relevance = insights.get('economic_relevance', {})
    
    logger.info("💰 ECONOMIC RELEVANCE FINDINGS:")
    logger.info("=" * 40)
    
    beyond_vol_vol = econ_relevance.get('dimensions_beyond_vol_volatility', [])
    if beyond_vol_vol:
        logger.info(f"🎯 DISCOVERY: {len(beyond_vol_vol)} dimensions beyond volume/volatility influence price action!")
        logger.info("")
        
        for dim in beyond_vol_vol:
            influences = econ_relevance['new_price_action_influences'].get(dim, [])
            applications = econ_relevance['novel_trading_applications'].get(dim, [])
            
            logger.info(f"📈 {dim.upper()} DIMENSION:")
            logger.info(f"   - Price Action Influences: {', '.join(influences)}")
            logger.info(f"   - Trading Applications: {', '.join(applications[:2])}")
            logger.info("")
        
        # Research implications
        logger.info("🔬 RESEARCH IMPLICATIONS:")
        logger.info("✅ Market has additional exploitable dimensions beyond volume/volatility")
        logger.info("✅ These dimensions can enhance momentum and mean reversion strategies") 
        logger.info("✅ Regime-specific ML models should incorporate these dimensions")
        logger.info("")
        
    else:
        logger.info("📊 FINDING: Volume and volatility remain the primary price action drivers")
        logger.info("⚠️ No additional dimensions show significant economic relevance")
        logger.info("💡 Consider:")
        logger.info("   - Expanding feature engineering to capture more market dynamics")
        logger.info("   - Using higher frequency data for microstructure effects")
        logger.info("   - Adding cross-asset or sentiment features")
        logger.info("")
    
    # Pipeline Recommendations
    logger.info("💡 PIPELINE RECOMMENDATIONS:")
    logger.info("=" * 40)
    
    for rec in insights['pipeline_coherence']['recommendations']:
        logger.info(f"   {rec}")
    
    logger.info("")
    
    # Next Steps for ML Training
    logger.info("🤖 NEXT STEPS FOR ML MODEL TRAINING:")
    logger.info("=" * 40)
    
    if beyond_vol_vol:
        logger.info("✅ Proceed with regime-specific ML models using:")
        logger.info("   1. Traditional volume and volatility dimensions")
        logger.info(f"   2. Newly discovered relevant dimensions: {', '.join(beyond_vol_vol)}")
        logger.info("   3. Focus on regime-specific strategies for each dimension")
    else:
        logger.info("⚠️ Consider:")
        logger.info("   1. Single ML model approach (limited regime benefits)")
        logger.info("   2. Focus on volume/volatility regime identification")
        logger.info("   3. Expand feature engineering before regime-specific training")
    
    return results


async def main():
    """Main enhanced pipeline demonstration."""
    
    logger.info("🚀 Enhanced Market Regime Research Pipeline")
    logger.info("🎯 Goal: Discover dimensions beyond volume/volatility that influence price action")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # Run the enhanced pipeline analysis
        results = await demonstrate_economic_relevance_discovery()
        
        logger.info("🎉 ENHANCED PIPELINE ANALYSIS COMPLETED!")
        logger.info("=" * 80)
        logger.info("")
        logger.info("📊 Key Outputs Generated:")
        logger.info("   1. Comprehensive feature analysis")
        logger.info("   2. Statistical dimensionality analysis (PCA, FA, ICA)")
        logger.info("   3. Market dimension discovery")
        logger.info("   4. Economic relevance analysis (NEW)")
        logger.info("   5. Pipeline coherence validation")
        logger.info("")
        logger.info("🔬 Research Question Answered:")
        logger.info("   Which dimensions beyond volume/volatility influence price action?")
        logger.info("")
        logger.info("📈 Ready for regime-specific ML model training based on findings!")
        
        return {'success': True, 'results': results}
        
    except Exception as e:
        logger.error(f"❌ Enhanced pipeline analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    # Run the enhanced pipeline
    result = asyncio.run(main())
    
    if result['success']:
        print("\n✅ Enhanced pipeline analysis completed successfully!")
        print("Check the detailed logs above for research findings.")
    else:
        print(f"\n❌ Analysis failed: {result['error']}")