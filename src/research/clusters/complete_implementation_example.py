"""
Complete Implementation Example: Fully Enhanced Market Regime Research Framework

This example demonstrates the complete implementation with ALL enhancements:

✅ 1. Trading-calibrated economic metrics (9 metrics with empirical thresholds)
✅ 2. Lookahead bias prevention (strict temporal separation)
✅ 3. Metric orthogonalization (reduced redundancy)
✅ 4. Comprehensive feature integration (ALL features from feature_engineering/)
✅ 5. Statistical robustness (PCA, AIC, BIC, bootstrap validation)
✅ 6. Economic relevance analysis (beyond volume/volatility discovery)

Research Question: "Which market regimes justify training different ML models, 
and what are the concrete trading rules for each regime?"
"""

import numpy as np
import pandas as pd
import asyncio
from pathlib import Path
import logging
import json

# Import complete enhanced framework
from src.research.clusters import (
    # Core components
    MarketDimensionAnalyzer, DimensionAnalysisConfig,
    RegimeClusterer, ClusteringConfig, ClusteringMethod,
    RegimeValidationMetrics, ValidationConfig,
    
    # Enhanced components
    EconomicValidator, EconomicValidationConfig, EconomicMetric,
    TradingMetricCalibrator, TradingCalibration, generate_complete_trading_calibration_report,
    LookaheadBiasPrevention, create_bias_free_analysis_wrapper,
    MetricOrthogonalizer, OrthogonalMetric,
    ComprehensiveFeatureGenerator,
    StatisticalDimensionAnalyzer, DimensionalityMethod,
    analyze_all_dimensions_economic_relevance
)

from src.utils.logger import system_logger

logging.basicConfig(level=logging.INFO)
logger = system_logger.getChild('CompleteImplementationExample')


def generate_realistic_market_data(n_samples: int = 2000) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate realistic market data with distinct regime characteristics."""
    
    logger.info(f"🎲 Generating realistic market data with {n_samples} samples")
    
    # Create datetime index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
    
    # Generate three distinct regimes with realistic characteristics
    np.random.seed(42)
    
    # Regime 1: Trending market (40% of data) - momentum dimension dominant
    regime_1_samples = int(n_samples * 0.4)
    trend_strength = 0.0008
    vol_level = 0.012
    price_1 = 100 + np.cumsum(np.random.normal(trend_strength, vol_level, regime_1_samples))
    vol_1 = np.random.lognormal(10, 0.4, regime_1_samples)
    
    # Regime 2: High volatility market (30% of data) - volatility dimension dominant  
    regime_2_samples = int(n_samples * 0.3)
    trend_strength = 0.0
    vol_level = 0.025
    price_2 = price_1[-1] + np.cumsum(np.random.normal(trend_strength, vol_level, regime_2_samples))
    vol_2 = np.random.lognormal(12, 1.0, regime_2_samples)
    
    # Regime 3: Mean reverting market (30% of data) - correlation dimension dominant
    regime_3_samples = n_samples - regime_1_samples - regime_2_samples
    mean_price = price_2[-1]
    reversion_strength = 0.02
    vol_level = 0.015
    
    price_3 = [price_2[-1]]
    for i in range(regime_3_samples - 1):
        deviation = price_3[-1] - mean_price
        reversion = -deviation * reversion_strength
        noise = np.random.normal(0, vol_level)
        new_price = price_3[-1] * (1 + reversion + noise)
        price_3.append(new_price)
    
    vol_3 = np.random.lognormal(11, 0.6, regime_3_samples)
    
    # Combine all regimes
    close_prices = np.concatenate([price_1, price_2, price_3])
    volumes = np.concatenate([vol_1, vol_2, vol_3])
    
    # Generate OHLC with realistic spreads
    noise = np.random.normal(0, 0.0008, n_samples)
    high_prices = close_prices * (1 + np.abs(noise) + 0.003)
    low_prices = close_prices * (1 - np.abs(noise) - 0.003)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # Add realistic microstructure data
    taker_buy_ratios = np.random.beta(2, 2, n_samples)  # Realistic taker buy ratios
    taker_buy_volumes = volumes * taker_buy_ratios
    
    # Create comprehensive market data
    market_data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes,
        'taker_buy_base_asset_volume': taker_buy_volumes
    }, index=dates)
    
    # True regime labels (for validation)
    true_regimes = np.concatenate([
        np.full(regime_1_samples, 0),  # Trending regime
        np.full(regime_2_samples, 1),  # High volatility regime
        np.full(regime_3_samples, 2)   # Mean reverting regime
    ])
    
    logger.info(f"✅ Generated realistic market data:")
    logger.info(f"   📊 Regime 0 (Trending): {regime_1_samples} samples ({regime_1_samples/n_samples:.1%})")
    logger.info(f"   📊 Regime 1 (High Vol): {regime_2_samples} samples ({regime_2_samples/n_samples:.1%})")
    logger.info(f"   📊 Regime 2 (Mean Rev): {regime_3_samples} samples ({regime_3_samples/n_samples:.1%})")
    
    return market_data, true_regimes


@create_bias_free_analysis_wrapper
async def run_complete_enhanced_analysis(market_data: pd.DataFrame, 
                                       true_regimes: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Run complete enhanced analysis with all improvements.
    
    This function is wrapped with bias prevention to ensure temporal separation.
    """
    
    logger.info("🚀 Starting Complete Enhanced Market Regime Analysis")
    logger.info("=" * 80)
    
    results = {}
    
    # === STEP 1: COMPREHENSIVE FEATURE GENERATION ===
    logger.info("📊 Step 1: Comprehensive Feature Generation (ALL features)")
    
    feature_generator = ComprehensiveFeatureGenerator()
    comprehensive_features = feature_generator.generate_all_available_features(market_data)
    
    feature_categories = feature_generator.get_feature_categories(comprehensive_features)
    results['feature_generation'] = {
        'total_features': len(comprehensive_features.columns),
        'feature_categories': {cat: len(features) for cat, features in feature_categories.items() if features},
        'feature_names': list(comprehensive_features.columns)
    }
    
    logger.info(f"   ✅ Generated {len(comprehensive_features.columns)} comprehensive features")
    for category, count in results['feature_generation']['feature_categories'].items():
        logger.info(f"      {category}: {count} features")
    
    # === STEP 2: STATISTICAL DIMENSIONALITY ANALYSIS ===
    logger.info("🔬 Step 2: Statistical Dimensionality Analysis (PCA, FA, ICA)")
    
    statistical_analyzer = StatisticalDimensionAnalyzer()
    statistical_results = statistical_analyzer.analyze_dimensions(
        comprehensive_features,
        methods=[DimensionalityMethod.PCA, DimensionalityMethod.FACTOR_ANALYSIS, DimensionalityMethod.ICA],
        n_components=None  # Auto-determine
    )
    
    results['statistical_analysis'] = {}
    for method, result in statistical_results.items():
        results['statistical_analysis'][method.value] = {
            'n_components': result.n_components,
            'explained_variance': float(np.sum(result.explained_variance_ratio)) if result.explained_variance_ratio is not None else None,
            'statistical_tests': result.statistical_tests,
            'interpretation': result.interpretation
        }
    
    # Get PCA results for dimensionality reduction
    pca_result = statistical_results.get(DimensionalityMethod.PCA)
    if pca_result:
        intrinsic_dimensionality = pca_result.n_components
        explained_variance = np.sum(pca_result.explained_variance_ratio)
        logger.info(f"   📊 Intrinsic dimensionality: {intrinsic_dimensionality} components")
        logger.info(f"   📊 Explained variance: {explained_variance:.1%}")
    
    # === STEP 3: MARKET DIMENSION DISCOVERY ===
    logger.info("🔍 Step 3: Market Dimension Discovery")
    
    dimension_analyzer = MarketDimensionAnalyzer()
    dimension_results = dimension_analyzer.analyze_all_dimensions(
        comprehensive_features, regime_labels=true_regimes, use_existing_features=True
    )
    
    results['market_dimensions'] = {
        'dimensions_discovered': [dim.value for dim in dimension_results.keys()],
        'dimension_metrics': {dim.value: metrics.to_dict() for dim, metrics in dimension_results.items()},
        'top_dimensions': [(dim.value, metrics.metrics.get('composite_score', 0)) 
                          for dim, metrics in sorted(dimension_results.items(), 
                                                   key=lambda x: x[1].metrics.get('composite_score', 0), 
                                                   reverse=True)]
    }
    
    logger.info(f"   ✅ Discovered {len(dimension_results)} market dimensions")
    for dim_name, score in results['market_dimensions']['top_dimensions'][:3]:
        logger.info(f"      {dim_name}: {score:.3f}")
    
    # === STEP 4: ECONOMIC RELEVANCE ANALYSIS ===
    logger.info("💰 Step 4: Economic Relevance Analysis (Beyond Volume/Volatility)")
    
    # Create dimension feature groups
    dimension_feature_groups = {}
    for dimension, metrics in dimension_results.items():
        dimension_features = comprehensive_features[metrics.feature_names]
        dimension_feature_groups[dimension.value] = dimension_features
    
    # Analyze economic relevance
    relevance_results = analyze_all_dimensions_economic_relevance(
        market_data, dimension_feature_groups
    )
    
    # Find dimensions beyond volume/volatility with economic relevance
    beyond_vol_volatility = {}
    for dim_name, relevance in relevance_results.items():
        if 'volume' not in dim_name.lower() and 'volatility' not in dim_name.lower():
            if relevance.overall_relevance_score > 0.15:
                beyond_vol_volatility[dim_name] = {
                    'relevance_score': relevance.overall_relevance_score,
                    'key_influences': [
                        influence.value for influence, score in relevance.price_action_influences.items()
                        if score > 0.2
                    ],
                    'trading_applications': relevance.trading_applications[:3]
                }
    
    results['economic_relevance'] = {
        'total_dimensions_analyzed': len(relevance_results),
        'beyond_volume_volatility': beyond_vol_volatility,
        'volume_volatility_baseline': {
            dim: rel.overall_relevance_score 
            for dim, rel in relevance_results.items() 
            if 'volume' in dim.lower() or 'volatility' in dim.lower()
        }
    }
    
    logger.info(f"   💰 Economic relevance analysis completed:")
    logger.info(f"      🔍 {len(beyond_vol_volatility)} dimensions beyond volume/volatility show relevance")
    
    if beyond_vol_volatility:
        logger.info("      🎯 Relevant dimensions beyond volume/volatility:")
        for dim, data in beyond_vol_volatility.items():
            logger.info(f"         {dim}: {data['relevance_score']:.3f} ({', '.join(data['key_influences'])})")
    else:
        logger.info("      ⚠️ No dimensions beyond volume/volatility show significant relevance")
    
    # === STEP 5: CLUSTERING WITH STATISTICAL VALIDATION ===
    logger.info("🔍 Step 5: Clustering with Statistical Validation (AIC, BIC, Gap)")
    
    # Use economically relevant features for clustering
    if beyond_vol_volatility:
        # Use features from relevant dimensions
        relevant_features = []
        for dim_name in beyond_vol_volatility.keys():
            if dim_name in dimension_feature_groups:
                relevant_features.extend(dimension_feature_groups[dim_name].columns)
        
        # Add volume/volatility features
        for dim_name, score in results['economic_relevance']['volume_volatility_baseline'].items():
            if dim_name in dimension_feature_groups:
                relevant_features.extend(dimension_feature_groups[dim_name].columns)
        
        clustering_features = comprehensive_features[list(set(relevant_features))].fillna(0)
    else:
        # Use top statistical components if no economic relevance found
        if pca_result and pca_result.n_components < len(comprehensive_features.columns):
            clustering_features = pd.DataFrame(
                pca_result.transformed_data,
                columns=[f'PC{i+1}' for i in range(pca_result.n_components)],
                index=comprehensive_features.index
            )
        else:
            clustering_features = comprehensive_features.fillna(0)
    
    # Run clustering with statistical validation
    clusterer = RegimeClusterer(ClusteringConfig(n_clusters=3))
    clustering_results = clusterer.run_all_methods(
        clustering_features.values,
        analyze_dimensions=True,
        feature_names=list(clustering_features.columns)
    )
    
    best_method, best_result = clusterer.get_best_method()
    
    results['clustering'] = {
        'best_method': best_method.value,
        'n_clusters': best_result.n_clusters,
        'silhouette_score': best_result.metrics.get('silhouette_score', 0),
        'aic_score': best_result.metrics.get('aic_diag', 0),
        'bic_score': best_result.metrics.get('bic_diag', 0),
        'gap_statistic': best_result.metrics.get('gap_statistic', 0),
        'regime_labels': best_result.labels,
        'clustering_features_used': len(clustering_features.columns)
    }
    
    logger.info(f"   ✅ Best clustering: {best_method.value}")
    logger.info(f"      Clusters: {best_result.n_clusters}, Silhouette: {best_result.metrics.get('silhouette_score', 0):.3f}")
    logger.info(f"      AIC: {best_result.metrics.get('aic_diag', 0):.1f}, BIC: {best_result.metrics.get('bic_diag', 0):.1f}")
    
    # === STEP 6: COMPREHENSIVE ECONOMIC VALIDATION ===
    logger.info("💰 Step 6: Comprehensive Economic Validation (9 Enhanced Metrics)")
    
    # Run enhanced economic validation
    economic_validator = EconomicValidator(EconomicValidationConfig())
    economic_validation_results = economic_validator.validate_regime_economics(
        market_data, best_result.labels
    )
    
    # Convert to serializable format
    economic_results_dict = {
        metric.value: result.to_dict() 
        for metric, result in economic_validation_results.items()
    }
    
    # Calculate economic significance summary
    economically_significant_metrics = sum(
        1 for result in economic_validation_results.values() 
        if result.economic_significance
    )
    total_metrics = len(economic_validation_results)
    economic_significance_rate = economically_significant_metrics / total_metrics
    
    results['economic_validation'] = {
        'total_metrics': total_metrics,
        'economically_significant': economically_significant_metrics,
        'significance_rate': economic_significance_rate,
        'overall_quality': 'strong' if economic_significance_rate >= 0.7 else 'moderate' if economic_significance_rate >= 0.4 else 'weak',
        'detailed_results': economic_results_dict
    }
    
    logger.info(f"   💰 Economic validation: {economically_significant_metrics}/{total_metrics} metrics significant ({economic_significance_rate:.1%})")
    logger.info(f"   📊 Overall economic quality: {results['economic_validation']['overall_quality']}")
    
    # === STEP 7: METRIC ORTHOGONALIZATION ===
    logger.info("🔧 Step 7: Metric Orthogonalization (Reduce Redundancy)")
    
    orthogonalizer = MetricOrthogonalizer()
    orthogonal_metrics = orthogonalizer.orthogonalize_metrics(economic_results_dict)
    
    # Calculate orthogonalization quality
    orthogonalization_quality = orthogonalizer.calculate_orthogonalization_quality(
        economic_results_dict, orthogonal_metrics
    )
    
    results['metric_orthogonalization'] = {
        'original_metrics': total_metrics,
        'orthogonal_metrics': len(orthogonal_metrics),
        'compression_ratio': orthogonalization_quality['compression_ratio'],
        'information_preservation': orthogonalization_quality['information_preservation'],
        'average_independence': orthogonalization_quality['average_independence'],
        'overall_quality': orthogonalization_quality['overall_orthogonalization_quality'],
        'orthogonal_results': {metric.value: result.to_dict() for metric, result in orthogonal_metrics.items()}
    }
    
    logger.info(f"   🔧 Orthogonalization: {total_metrics} → {len(orthogonal_metrics)} metrics")
    logger.info(f"   📊 Information preservation: {orthogonalization_quality['information_preservation']:.1%}")
    logger.info(f"   📊 Average independence: {orthogonalization_quality['average_independence']:.3f}")
    
    # === STEP 8: TRADING CALIBRATION ===
    logger.info("💼 Step 8: Trading Calibration (Actionable Trading Rules)")
    
    # Generate trading calibration
    trading_calibration_report = generate_complete_trading_calibration_report(economic_results_dict)
    
    # Extract key trading insights
    calibrator = TradingMetricCalibrator()
    
    # Get regime-specific calibrations for key metrics
    instability_metric = economic_results_dict.get('price_instability_influence')
    if instability_metric and 'regime_specific_values' in instability_metric:
        instability_calibrations = calibrator.calibrate_price_instability_influence(
            instability_metric['value'], 
            instability_metric['regime_specific_values']
        )
        
        # Extract trading rules
        trading_rules = {}
        for regime, calibration in instability_calibrations.items():
            trading_rules[regime] = {
                'position_size_multiplier': calibration.position_sizing_multiplier,
                'stop_loss_multiplier': calibration.stop_loss_multiplier,
                'expected_sharpe_impact': calibration.sharpe_impact,
                'expected_drawdown_impact': calibration.max_drawdown_impact,
                'confidence_level': calibration.confidence_level
            }
        
        results['trading_calibration'] = {
            'regime_specific_rules': trading_rules,
            'calibration_report': trading_calibration_report
        }
    
    # === STEP 9: VALIDATION WITH TRUE REGIMES ===
    if true_regimes is not None:
        logger.info("✅ Step 9: Validation Against True Regimes")
        
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        # Calculate agreement with true regimes
        discovered_regimes = best_result.labels
        min_len = min(len(true_regimes), len(discovered_regimes))
        
        agreement_ari = adjusted_rand_score(true_regimes[:min_len], discovered_regimes[:min_len])
        agreement_nmi = normalized_mutual_info_score(true_regimes[:min_len], discovered_regimes[:min_len])
        
        results['validation_against_truth'] = {
            'adjusted_rand_score': float(agreement_ari),
            'normalized_mutual_info': float(agreement_nmi),
            'agreement_quality': 'excellent' if agreement_ari > 0.7 else 'good' if agreement_ari > 0.5 else 'moderate' if agreement_ari > 0.3 else 'poor'
        }
        
        logger.info(f"   ✅ Agreement with true regimes:")
        logger.info(f"      ARI: {agreement_ari:.3f}, NMI: {agreement_nmi:.3f}")
        logger.info(f"      Quality: {results['validation_against_truth']['agreement_quality']}")
    
    return results


async def demonstrate_complete_framework():
    """Demonstrate the complete enhanced framework."""
    
    logger.info("🎯 COMPLETE ENHANCED FRAMEWORK DEMONSTRATION")
    logger.info("🔬 Research Question: Which regimes justify different ML models?")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # Generate realistic market data
        market_data, true_regimes = generate_realistic_market_data(2000)
        
        # Run complete enhanced analysis
        results = await run_complete_enhanced_analysis(market_data, true_regimes)
        
        # === RESEARCH FINDINGS SUMMARY ===
        logger.info("🎉 COMPLETE RESEARCH FINDINGS")
        logger.info("=" * 80)
        
        # Feature analysis
        feature_gen = results['feature_generation']
        logger.info(f"📊 FEATURE ANALYSIS:")
        logger.info(f"   Total features generated: {feature_gen['total_features']}")
        logger.info(f"   Feature categories: {len(feature_gen['feature_categories'])}")
        logger.info("")
        
        # Statistical analysis
        statistical = results['statistical_analysis']
        pca_results = statistical.get('principal_component_analysis', {})
        logger.info(f"🔬 STATISTICAL ANALYSIS:")
        logger.info(f"   Intrinsic dimensionality: {pca_results.get('n_components', 'N/A')} components")
        logger.info(f"   Variance explained: {pca_results.get('explained_variance', 0):.1%}")
        logger.info(f"   KMO adequacy: {pca_results.get('statistical_tests', {}).get('kaiser_meyer_olkin', 'N/A'):.3f}")
        logger.info("")
        
        # Economic relevance findings
        economic_rel = results['economic_relevance']
        beyond_vol_vol = economic_rel['beyond_volume_volatility']
        logger.info(f"💰 ECONOMIC RELEVANCE FINDINGS:")
        logger.info(f"   Dimensions beyond volume/volatility: {len(beyond_vol_vol)}")
        
        if beyond_vol_vol:
            logger.info("   🎯 DISCOVERY: Additional economically relevant dimensions found!")
            for dim, data in beyond_vol_vol.items():
                logger.info(f"      {dim}: {data['relevance_score']:.3f} - {', '.join(data['key_influences'])}")
        else:
            logger.info("   📊 Volume and volatility remain primary price action drivers")
        logger.info("")
        
        # Economic validation results
        economic_val = results['economic_validation']
        logger.info(f"📈 ECONOMIC VALIDATION:")
        logger.info(f"   Economic significance rate: {economic_val['significance_rate']:.1%}")
        logger.info(f"   Overall quality: {economic_val['overall_quality']}")
        logger.info(f"   Significant metrics: {economic_val['economically_significant']}/{economic_val['total_metrics']}")
        logger.info("")
        
        # Clustering quality
        clustering = results['clustering']
        logger.info(f"🔍 CLUSTERING QUALITY:")
        logger.info(f"   Best method: {clustering['best_method']}")
        logger.info(f"   Silhouette score: {clustering['silhouette_score']:.3f}")
        logger.info(f"   AIC: {clustering['aic_score']:.1f}, BIC: {clustering['bic_score']:.1f}")
        logger.info("")
        
        # Validation against truth
        if 'validation_against_truth' in results:
            validation = results['validation_against_truth']
            logger.info(f"✅ VALIDATION AGAINST TRUE REGIMES:")
            logger.info(f"   Agreement quality: {validation['agreement_quality']}")
            logger.info(f"   ARI: {validation['adjusted_rand_score']:.3f}")
            logger.info(f"   NMI: {validation['normalized_mutual_info']:.3f}")
            logger.info("")
        
        # Trading implications
        if 'trading_calibration' in results:
            trading_cal = results['trading_calibration']
            logger.info(f"💼 TRADING CALIBRATION:")
            logger.info(f"   Regime-specific rules generated: {len(trading_cal['regime_specific_rules'])}")
            
            for regime, rules in trading_cal['regime_specific_rules'].items():
                logger.info(f"   Regime {regime}:")
                logger.info(f"      Position size: {rules['position_size_multiplier']:.2f}x")
                logger.info(f"      Stop loss: {rules['stop_loss_multiplier']:.2f}x ATR")
                logger.info(f"      Expected Sharpe impact: {rules['expected_sharpe_impact']:+.2f}")
        logger.info("")
        
        # === FINAL RESEARCH DECISION ===
        logger.info("🎯 FINAL RESEARCH DECISION:")
        logger.info("=" * 40)
        
        # Decision framework
        economic_quality = economic_val['overall_quality']
        beyond_vol_vol_count = len(beyond_vol_vol)
        agreement_quality = results.get('validation_against_truth', {}).get('agreement_quality', 'unknown')
        
        if economic_quality == 'strong' and beyond_vol_vol_count >= 1:
            decision = "✅ TRAIN REGIME-SPECIFIC ML MODELS"
            rationale = [
                f"Strong economic foundation ({economic_val['significance_rate']:.1%} significance)",
                f"Additional relevant dimensions discovered: {list(beyond_vol_vol.keys())}",
                f"Good regime discovery quality: {agreement_quality}"
            ]
        elif economic_quality in ['strong', 'moderate'] and beyond_vol_vol_count == 0:
            decision = "⚠️ FOCUS ON VOLUME/VOLATILITY REGIME MODELS"
            rationale = [
                f"Economic foundation: {economic_quality}",
                "No additional dimensions beyond volume/volatility found",
                "Regime-specific models justified but limited scope"
            ]
        else:
            decision = "❌ CONSIDER SINGLE MODEL APPROACH"
            rationale = [
                f"Weak economic foundation ({economic_val['significance_rate']:.1%} significance)",
                "Limited regime-specific benefits detected",
                "Single model may be more efficient"
            ]
        
        logger.info(decision)
        logger.info("Rationale:")
        for reason in rationale:
            logger.info(f"   - {reason}")
        logger.info("")
        
        # Next steps
        logger.info("🚀 RECOMMENDED NEXT STEPS:")
        if "TRAIN REGIME-SPECIFIC" in decision:
            logger.info("   1. Implement regime-specific ML model training")
            logger.info("   2. Use discovered relevant dimensions as features")
            logger.info("   3. Apply regime-specific trading rules")
            logger.info("   4. Monitor regime transitions for model switching")
        elif "VOLUME/VOLATILITY" in decision:
            logger.info("   1. Focus regime identification on volume/volatility dimensions")
            logger.info("   2. Train models specific to volume/volatility regimes")
            logger.info("   3. Consider expanding feature engineering for additional dimensions")
        else:
            logger.info("   1. Use single ML model approach")
            logger.info("   2. Consider alternative regime identification methods")
            logger.info("   3. Focus on feature engineering improvements")
        
        # Save complete results
        output_dir = Path("regime_analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        with open(output_dir / "complete_analysis_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save trading calibration report
        if 'trading_calibration' in results:
            with open(output_dir / "trading_calibration_report.md", 'w') as f:
                f.write(results['trading_calibration']['calibration_report'])
        
        logger.info(f"💾 Complete results saved to {output_dir}")
        
        return {
            'success': True,
            'decision': decision,
            'economic_quality': economic_quality,
            'beyond_vol_volatility_count': beyond_vol_vol_count,
            'agreement_quality': agreement_quality,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"❌ Complete framework demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


async def main():
    """Main demonstration of complete enhanced framework."""
    
    logger.info("🚀 COMPLETE ENHANCED MARKET REGIME RESEARCH FRAMEWORK")
    logger.info("🎯 Fully Implemented with All Enhancements")
    logger.info("=" * 100)
    logger.info("")
    logger.info("ENHANCEMENTS INCLUDED:")
    logger.info("✅ Trading-calibrated economic metrics (empirical thresholds)")
    logger.info("✅ Lookahead bias prevention (strict temporal separation)")
    logger.info("✅ Metric orthogonalization (reduced redundancy)")
    logger.info("✅ Comprehensive feature integration (ALL feature_engineering/ features)")
    logger.info("✅ Statistical robustness (PCA, AIC, BIC, bootstrap)")
    logger.info("✅ Economic relevance analysis (beyond volume/volatility discovery)")
    logger.info("✅ 9 enhanced price action metrics with 3 missing critical metrics")
    logger.info("=" * 100)
    logger.info("")
    
    # Run complete demonstration
    result = await demonstrate_complete_framework()
    
    if result['success']:
        logger.info("")
        logger.info("🎉 COMPLETE FRAMEWORK DEMONSTRATION SUCCESSFUL!")
        logger.info("=" * 80)
        logger.info("")
        logger.info("📊 KEY OUTCOMES:")
        logger.info(f"   Decision: {result['decision']}")
        logger.info(f"   Economic Quality: {result['economic_quality']}")
        logger.info(f"   Beyond Vol/Volatility Dimensions: {result['beyond_vol_volatility_count']}")
        logger.info(f"   Regime Discovery Quality: {result['agreement_quality']}")
        logger.info("")
        logger.info("📁 OUTPUT FILES GENERATED:")
        logger.info("   - complete_analysis_results.json (detailed results)")
        logger.info("   - trading_calibration_report.md (actionable trading rules)")
        logger.info("")
        logger.info("🚀 FRAMEWORK IS READY FOR YOUR RESEARCH!")
        logger.info("   Apply to your real market data using the same workflow")
        
    else:
        logger.error(f"❌ Framework demonstration failed: {result['error']}")
    
    return result


if __name__ == "__main__":
    # Run the complete enhanced framework
    result = asyncio.run(main())
    
    if result['success']:
        print("\n🎉 Complete enhanced framework demonstration successful!")
        print("Check the detailed logs and generated files for comprehensive results.")
    else:
        print(f"\n❌ Demonstration failed: {result['error']}")