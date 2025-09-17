"""
Example Usage of Market Regime Clustering Research Framework.

This script demonstrates how to use the comprehensive regime clustering
research framework for market analysis. It shows the complete workflow
from data preparation to final analysis and visualization.

Usage:
    python example_usage.py

This example will:
1. Load sample market data
2. Analyze market dimensions
3. Run clustering analysis
4. Perform feature importance analysis
5. Validate regime quality
6. Train ML models on regimes
7. Integrate with existing HMM systems
8. Generate comprehensive visualizations and reports
"""

import numpy as np
import pandas as pd
import asyncio
from pathlib import Path
import logging

# Import the regime clustering framework
from src.research.clusters import (
    MarketDimensionAnalyzer, DimensionAnalysisConfig,
    RegimeClusterer, ClusteringConfig, ClusteringMethod,
    RegimeFeatureImportance, ImportanceConfig, ImportanceMethod,
    RegimeValidationMetrics, ValidationConfig,
    HMMIntegrationLayer, IntegrationConfig, IntegrationMethod,
    RegimeVisualization, VisualizationConfig
)

from src.utils.logger import system_logger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = system_logger.getChild('RegimeClusteringExample')


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate sample market data for demonstration.
    
    In practice, you would load real market data from your data source.
    """
    logger.info(f"🎲 Generating sample market data ({n_samples} samples)")
    
    # Create datetime index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
    
    # Generate synthetic OHLCV data with regime-like behavior
    np.random.seed(42)
    
    # Create three distinct regimes with different characteristics
    regime_1_samples = n_samples // 3
    regime_2_samples = n_samples // 3
    regime_3_samples = n_samples - regime_1_samples - regime_2_samples
    
    # Regime 1: Low volatility, steady uptrend
    price_1 = 100 + np.cumsum(np.random.normal(0.01, 0.005, regime_1_samples))
    vol_1 = np.random.lognormal(10, 0.5, regime_1_samples)
    
    # Regime 2: High volatility, sideways
    price_2 = price_1[-1] + np.cumsum(np.random.normal(0, 0.02, regime_2_samples))
    vol_2 = np.random.lognormal(12, 1.0, regime_2_samples)
    
    # Regime 3: Medium volatility, downtrend
    price_3 = price_2[-1] + np.cumsum(np.random.normal(-0.005, 0.01, regime_3_samples))
    vol_3 = np.random.lognormal(11, 0.7, regime_3_samples)
    
    # Combine regimes
    close_prices = np.concatenate([price_1, price_2, price_3])
    volumes = np.concatenate([vol_1, vol_2, vol_3])
    
    # Generate OHLC from close prices
    noise = np.random.normal(0, 0.001, n_samples)
    high_prices = close_prices * (1 + np.abs(noise) + 0.001)
    low_prices = close_prices * (1 - np.abs(noise) - 0.001)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # Create DataFrame
    market_data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    # Add some technical indicators
    market_data['returns'] = market_data['close'].pct_change()
    market_data['sma_20'] = market_data['close'].rolling(20).mean()
    market_data['volatility'] = market_data['returns'].rolling(20).std()
    
    # Create true regime labels for comparison (in practice, these would be unknown)
    true_regimes = np.concatenate([
        np.full(regime_1_samples, 0),
        np.full(regime_2_samples, 1),
        np.full(regime_3_samples, 2)
    ])
    
    return market_data.fillna(method='ffill'), true_regimes


async def run_dimension_analysis_example(market_data: pd.DataFrame) -> dict:
    """Example of market dimension analysis."""
    logger.info("📊 Running Market Dimension Analysis Example")
    
    # Configure dimension analysis
    config = DimensionAnalysisConfig(
        lookback_periods=[5, 10, 20, 50],
        volume_windows=[5, 10, 20],
        volatility_windows=[10, 20, 50],
        use_pca=True,
        use_mutual_information=True
    )
    
    # Initialize analyzer
    analyzer = MarketDimensionAnalyzer(config)
    
    # Run analysis
    dimension_results = analyzer.analyze_all_dimensions(market_data)
    
    # Get top dimensions
    top_dimensions = analyzer.get_top_dimensions(3)
    
    logger.info(f"✅ Analyzed {len(dimension_results)} market dimensions")
    for dimension, metrics in top_dimensions:
        logger.info(f"   {dimension.value}: composite_score={metrics.metrics.get('composite_score', 0):.3f}")
    
    # Generate report
    report = analyzer.generate_analysis_report()
    
    return {
        'dimension_results': {dim.value: metrics.to_dict() for dim, metrics in dimension_results.items()},
        'top_dimensions': [(dim.value, metrics.to_dict()) for dim, metrics in top_dimensions],
        'analysis_report': report
    }


async def run_clustering_analysis_example(market_data: pd.DataFrame) -> dict:
    """Example of regime clustering analysis."""
    logger.info("🔍 Running Regime Clustering Analysis Example")
    
    # Configure clustering
    config = ClusteringConfig(
        n_clusters=3,  # We know there are 3 regimes in our sample data
        min_cluster_size=50,
        max_clusters=10
    )
    
    # Initialize clusterer
    clusterer = RegimeClusterer(config)
    
    # Prepare features
    features = market_data[['open', 'high', 'low', 'close', 'volume', 'returns', 'volatility']].fillna(0)
    
    # Run all clustering methods
    clustering_results = clusterer.run_all_methods(features.values)
    
    # Get best method
    best_method, best_result = clusterer.get_best_method()
    
    logger.info(f"✅ Tested {len(clustering_results)} clustering methods")
    logger.info(f"   Best method: {best_method.value} with {best_result.n_clusters} clusters")
    logger.info(f"   Silhouette score: {best_result.metrics.get('silhouette_score', 0):.3f}")
    
    # Generate comparison
    comparison_df = clusterer.compare_methods()
    
    return {
        'best_method': best_method.value,
        'best_result': best_result.to_dict(),
        'all_results': {method.value: result.to_dict() for method, result in clustering_results.items()},
        'comparison_df': comparison_df.to_dict() if not comparison_df.empty else {},
        'regime_labels': best_result.labels,
        'n_clusters': best_result.n_clusters
    }


async def run_feature_importance_example(market_data: pd.DataFrame, regime_labels: np.ndarray) -> dict:
    """Example of feature importance analysis."""
    logger.info("⚖️ Running Feature Importance Analysis Example")
    
    # Configure feature importance analysis
    config = ImportanceConfig(
        cross_validation_folds=3,
        ensemble_methods=[
            ImportanceMethod.MUTUAL_INFORMATION,
            ImportanceMethod.RANDOM_FOREST,
            ImportanceMethod.PERMUTATION
        ]
    )
    
    # Initialize analyzer
    analyzer = RegimeFeatureImportance(config)
    
    # Prepare features
    features = market_data[['open', 'high', 'low', 'close', 'volume', 'returns', 'volatility']].fillna(0)
    
    # Run all importance methods
    importance_results = analyzer.analyze_all_methods(features, regime_labels)
    
    # Get consensus features
    consensus_features = analyzer.get_consensus_features(10, min_methods=2)
    
    logger.info(f"✅ Analyzed feature importance using {len(importance_results)} methods")
    logger.info("   Top consensus features:")
    for i, (feature, score, n_methods) in enumerate(consensus_features[:5], 1):
        logger.info(f"     {i}. {feature}: {score:.3f} (agreed by {n_methods} methods)")
    
    # Generate report
    report = analyzer.generate_importance_report()
    
    return {
        'importance_results': {method.value: result.to_dict() for method, result in importance_results.items()},
        'consensus_features': consensus_features,
        'analysis_report': report
    }


async def run_validation_example(market_data: pd.DataFrame, regime_labels: np.ndarray) -> dict:
    """Example of regime validation."""
    logger.info("✅ Running Regime Validation Example")
    
    # Configure validation
    config = ValidationConfig(
        significance_level=0.05,
        min_regime_duration=10,
        risk_free_rate=0.02
    )
    
    # Initialize validator
    validator = RegimeValidationMetrics(config)
    
    # Run all validation metrics
    validation_results = validator.validate_all_metrics(market_data, regime_labels)
    
    # Calculate composite score
    composite_score = validator.calculate_composite_score()
    
    # Get validation summary
    summary = validator.get_validation_summary()
    
    logger.info(f"✅ Validated regimes using {len(validation_results)} metrics")
    logger.info(f"   Composite validation score: {composite_score:.3f}")
    logger.info(f"   Significant metrics: {summary.get('significant_metrics', 0)}")
    
    # Generate report
    report = validator.generate_validation_report()
    
    return {
        'validation_results': {metric.value: result.to_dict() for metric, result in validation_results.items()},
        'composite_score': composite_score,
        'summary': summary,
        'validation_report': report
    }


async def run_trading_calibration_example(market_data: pd.DataFrame, regime_labels: np.ndarray) -> dict:
    """Example of trading calibration for discovered regimes."""
    logger.info("💼 Running Trading Calibration Example")
    
    # Run economic validation
    from src.research.clusters import EconomicValidator, generate_complete_trading_calibration_report
    
    economic_validator = EconomicValidator()
    economic_results = economic_validator.validate_regime_economics(market_data, regime_labels)
    
    # Convert to serializable format
    economic_dict = {
        metric.value: result.to_dict() 
        for metric, result in economic_results.items()
    }
    
    # Generate trading calibration report
    trading_report = generate_complete_trading_calibration_report(economic_dict)
    
    # Extract key trading insights
    economically_significant = sum(1 for result in economic_results.values() if result.economic_significance)
    total_metrics = len(economic_results)
    
    logger.info(f"✅ Economic validation completed: {economically_significant}/{total_metrics} metrics significant")
    
    # Determine ML training recommendation
    significance_rate = economically_significant / total_metrics
    if significance_rate >= 0.7:
        ml_recommendation = "Train separate ML models per regime"
    elif significance_rate >= 0.4:
        ml_recommendation = "Consider selective regime-based modeling"
    else:
        ml_recommendation = "Single ML model approach recommended"
    
    logger.info(f"🤖 ML Training Recommendation: {ml_recommendation}")
    
    return {
        'economic_results': economic_dict,
        'trading_calibration_report': trading_report,
        'ml_training_recommendation': ml_recommendation,
        'economic_significance_rate': significance_rate
    }


async def run_integration_example(market_data: pd.DataFrame) -> dict:
    """Example of HMM integration."""
    logger.info("🔄 Running HMM Integration Example")
    
    # Configure integration
    config = IntegrationConfig(
        method=IntegrationMethod.COMPARATIVE,  # Compare methods
        clustering_n_clusters=3,
        analyze_dimensions=True,
        analyze_feature_importance=True,
        validate_results=True,
        save_results=False  # Don't save for example
    )
    
    # Initialize integration layer
    integration = HMMIntegrationLayer(config)
    
    try:
        # Run integration analysis
        result = await integration.run_integration_analysis(market_data)
        
        logger.info(f"✅ Integration analysis completed")
        logger.info(f"   Method: {result.method.value}")
        logger.info(f"   Components analyzed: {len([k for k, v in result.__dict__.items() if v is not None])}")
        
        # Log recommendations
        if result.recommendations:
            logger.info("   Key recommendations:")
            for i, rec in enumerate(result.recommendations[:3], 1):
                logger.info(f"     {i}. {rec}")
        
        # Generate report
        report = integration.generate_integration_report(result)
        
        return {
            'integration_result': result.to_dict(),
            'integration_report': report
        }
        
    except Exception as e:
        logger.error(f"Integration analysis failed: {e}")
        return {'integration_result': None, 'error': str(e)}


async def run_visualization_example(market_data: pd.DataFrame, 
                                   regime_labels: np.ndarray,
                                   analysis_results: dict) -> dict:
    """Example of visualization generation."""
    logger.info("📊 Running Visualization Example")
    
    # Configure visualization
    config = VisualizationConfig(
        save_plots=False,  # Don't save for example
        use_plotly=False,  # Use matplotlib for simplicity
        figure_size=(10, 6)
    )
    
    # Initialize visualizer
    visualizer = RegimeVisualization(config)
    
    try:
        # Create regime timeseries plot
        fig1 = visualizer.plot_regime_timeseries(market_data, regime_labels)
        logger.info("   ✅ Created regime timeseries plot")
        
        # Create clustering quality comparison (if available)
        if 'clustering_results' in analysis_results and 'all_results' in analysis_results['clustering_results']:
            clustering_metrics = {}
            for method, result in analysis_results['clustering_results']['all_results'].items():
                if 'metrics' in result:
                    clustering_metrics[method] = result['metrics']
            
            if clustering_metrics:
                fig2 = visualizer.plot_clustering_quality_comparison(clustering_metrics)
                logger.info("   ✅ Created clustering quality comparison")
        
        # Create feature importance heatmap (if available)
        if 'feature_importance' in analysis_results and 'importance_results' in analysis_results['feature_importance']:
            importance_data = {}
            for method, result in analysis_results['feature_importance']['importance_results'].items():
                if 'feature_names' in result and 'importance_scores' in result:
                    feature_names = result['feature_names']
                    importance_scores = result['importance_scores']
                    importance_data[method] = dict(zip(feature_names, importance_scores))
            
            if importance_data:
                fig3 = visualizer.plot_feature_importance_heatmap(importance_data, top_n=10)
                logger.info("   ✅ Created feature importance heatmap")
        
        # Create validation metrics comparison (if available)
        if 'validation_metrics' in analysis_results and 'validation_results' in analysis_results['validation_metrics']:
            fig4 = visualizer.plot_validation_metrics_comparison(
                analysis_results['validation_metrics']['validation_results']
            )
            logger.info("   ✅ Created validation metrics comparison")
        
        # Create regime transitions plot
        fig5 = visualizer.plot_regime_transitions(regime_labels)
        logger.info("   ✅ Created regime transitions plot")
        
        logger.info("📊 Visualization example completed successfully")
        
        return {
            'visualizations_created': [
                'regime_timeseries',
                'clustering_quality_comparison',
                'feature_importance_heatmap',
                'validation_metrics_comparison',
                'regime_transitions'
            ]
        }
        
    except Exception as e:
        logger.error(f"Visualization example failed: {e}")
        return {'visualizations_created': [], 'error': str(e)}


async def main():
    """Main example function."""
    logger.info("🚀 Starting Market Regime Clustering Research Framework Example")
    
    try:
        # Step 1: Generate sample data
        market_data, true_regimes = generate_sample_data(1000)
        logger.info(f"📊 Generated sample data: {market_data.shape}")
        
        # Step 2: Dimension Analysis
        dimension_results = await run_dimension_analysis_example(market_data)
        
        # Step 3: Clustering Analysis
        clustering_results = await run_clustering_analysis_example(market_data)
        discovered_regimes = clustering_results['regime_labels']
        
        # Step 4: Feature Importance Analysis
        feature_importance_results = await run_feature_importance_example(market_data, discovered_regimes)
        
        # Step 5: Validation Analysis
        validation_results = await run_validation_example(market_data, discovered_regimes)
        
        # Step 6: Trading Calibration
        trading_calibration_results = await run_trading_calibration_example(market_data, discovered_regimes)
        
        # Step 7: Integration Analysis
        integration_results = await run_integration_example(market_data)
        
        # Step 8: Visualization
        analysis_results = {
            'dimension_analysis': dimension_results,
            'clustering_results': clustering_results,
            'feature_importance': feature_importance_results,
            'validation_metrics': {'validation_results': validation_results['validation_results']},
            'trading_calibration': trading_calibration_results,
            'integration': integration_results
        }
        
        visualization_results = await run_visualization_example(
            market_data, discovered_regimes, analysis_results
        )
        
        # Final Summary
        logger.info("🎉 Example completed successfully!")
        logger.info("=" * 60)
        logger.info("SUMMARY:")
        logger.info(f"  📊 Data points analyzed: {len(market_data)}")
        logger.info(f"  🔍 True regimes: {len(np.unique(true_regimes))}")
        logger.info(f"  🎯 Discovered regimes: {clustering_results['n_clusters']}")
        logger.info(f"  📈 Top dimension: {dimension_results['top_dimensions'][0][0] if dimension_results['top_dimensions'] else 'N/A'}")
        logger.info(f"  ⚖️ Top feature: {feature_importance_results['consensus_features'][0][0] if feature_importance_results['consensus_features'] else 'N/A'}")
        logger.info(f"  ✅ Validation score: {validation_results['composite_score']:.3f}")
        logger.info(f"  💼 ML Training Recommendation: {trading_calibration_results.get('ml_training_recommendation', 'N/A')}")
        logger.info(f"  📊 Visualizations: {len(visualization_results['visualizations_created'])}")
        logger.info("=" * 60)
        
        # Agreement between true and discovered regimes
        from sklearn.metrics import adjusted_rand_score
        agreement = adjusted_rand_score(true_regimes, discovered_regimes)
        logger.info(f"🎯 Agreement with true regimes: {agreement:.3f}")
        
        return {
            'success': True,
            'results': analysis_results,
            'agreement_score': agreement
        }
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    # Run the example
    result = asyncio.run(main())
    
    if result['success']:
        print("\n✅ Example completed successfully!")
        print("Check the logs above for detailed results.")
    else:
        print(f"\n❌ Example failed: {result['error']}")