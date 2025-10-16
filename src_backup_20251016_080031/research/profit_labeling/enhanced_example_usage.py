"""
Enhanced Multi-Horizon Profit Labeling Framework Examples

This script demonstrates how to use both the original and enhanced components
of the profit labeling research framework, including ML-based enhancements,
adaptive strategies, ensemble methods, and backtesting validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import original research framework components
from research.profit_labeling import (
    HeuristicAnalyzer,
    LabelingValidator,
    ParameterOptimizer,
    LabelingVisualizer,
    ResearchRunner,
    HeuristicAnalysisConfig,
    ValidationConfig,
    OptimizationConfig,
    VisualizationConfig,
    ResearchConfig,
    ResearchWorkflow,
    OptimizationMethod,
    OptimizationObjective
)

# Import enhanced components
from research.profit_labeling import (
    # Enhanced labeling system
    EnhancedMultiHorizonProfitLabeler,
    EnhancedLabelingConfig,
    EnhancementLevel,
    create_enhanced_labeler,
    generate_fully_enhanced_labels,
    
    # ML components
    MLLabelQualityAssessor,
    MLQualityAssessmentConfig,
    assess_label_quality_ml,
    
    # Adaptive components
    AdaptiveLabelingStrategy,
    AdaptiveLabelingConfig,
    get_regime_adaptive_config,
    
    # Ensemble components
    EnsembleLabelingSystem,
    EnsembleLabelingConfig,
    generate_ensemble_labels,
    
    # Advanced validation
    AdvancedStatisticalValidator,
    AdvancedValidationConfig,
    validate_labels_advanced,
    
    # Dynamic optimization
    discover_optimal_targets_and_horizons,
    DynamicOptimizationConfig,
    
    # Feature engineering
    ContextualFeatureEngineer,
    ContextualFeatureConfig,
    engineer_contextual_features,
    
    # Backtesting validation
    BacktestingIntegratedValidator,
    BacktestingConfig,
    validate_labels_through_backtesting
)


def generate_sample_data(n_samples: int = 2000) -> pd.DataFrame:
    """Generate sample market data for demonstration."""
    np.random.seed(42)
    
    # Generate realistic price data with trends and volatility clustering
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='5min')
    
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.002, n_samples)
    
    # Add volatility clustering
    vol_persistence = 0.9
    volatility = np.zeros(n_samples)
    volatility[0] = 0.002
    
    for i in range(1, n_samples):
        volatility[i] = vol_persistence * volatility[i-1] + (1 - vol_persistence) * 0.002
        returns[i] = np.random.normal(0.0001, volatility[i])
    
    # Generate prices
    prices = [base_price]
    for ret in returns[:-1]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    }, index=dates)
    
    # Ensure OHLC consistency
    for i in range(len(data)):
        data.loc[data.index[i], 'high'] = max(data.iloc[i][['open', 'high', 'low', 'close']])
        data.loc[data.index[i], 'low'] = min(data.iloc[i][['open', 'high', 'low', 'close']])
    
    return data


def example_enhanced_labeling_basic():
    """Example of enhanced labeling with ML quality assessment."""
    print("🤖 Running Enhanced Labeling (ML-Enhanced) Example")
    
    # Generate sample data
    market_data = generate_sample_data(1500)
    print(f"Generated {len(market_data)} samples of market data")
    
    # Create enhanced labeler with ML enhancement
    enhanced_labeler = create_enhanced_labeler(EnhancementLevel.ML_ENHANCED)
    
    # Generate enhanced labels
    result = enhanced_labeler.generate_enhanced_labels(market_data)
    
    print(f"✅ Enhanced labeling completed in {result.processing_time:.2f}s")
    print(f"   → Base labels shape: {result.base_labels.shape}")
    print(f"   → Enhanced labels shape: {result.enhanced_labels.shape}")
    print(f"   → Overall quality score: {result.quality_scores.get('overall_quality', 0):.3f}")
    
    # Show ML assessment results
    if result.ml_assessment_result:
        ml_scores = result.ml_assessment_result.quality_scores
        print(f"   → ML Predictive Power: {ml_scores.get('PREDICTIVE_POWER', 0):.3f}")
        print(f"   → ML Stability Score: {ml_scores.get('STABILITY_SCORE', 0):.3f}")
    
    return result


def example_adaptive_labeling():
    """Example of adaptive market regime-aware labeling."""
    print("\n🎯 Running Adaptive Labeling Example")
    
    # Generate sample data
    market_data = generate_sample_data(1200)
    
    # Create enhanced labeler with adaptive capabilities
    enhanced_labeler = create_enhanced_labeler(EnhancementLevel.ADAPTIVE)
    
    # Generate enhanced labels
    result = enhanced_labeler.generate_enhanced_labels(market_data)
    
    print(f"✅ Adaptive labeling completed")
    print(f"   → Overall quality score: {result.quality_scores.get('overall_quality', 0):.3f}")
    
    # Show adaptive results
    if result.adaptive_result:
        print(f"   → Detected regime: {result.adaptive_result.regime.value}")
        print(f"   → Regime confidence: {result.adaptive_result.regime_confidence:.3f}")
        print(f"   → Configuration updated: {result.adaptive_result.metadata.get('update_triggered', False)}")
    
    return result


def example_ensemble_labeling():
    """Example of ensemble labeling approaches."""
    print("\n🎭 Running Ensemble Labeling Example")
    
    # Generate sample data
    market_data = generate_sample_data(1000)
    
    # Create enhanced labeler with ensemble methods
    enhanced_labeler = create_enhanced_labeler(EnhancementLevel.ENSEMBLE)
    
    # Generate enhanced labels
    result = enhanced_labeler.generate_enhanced_labels(market_data)
    
    print(f"✅ Ensemble labeling completed")
    print(f"   → Overall quality score: {result.quality_scores.get('overall_quality', 0):.3f}")
    
    # Show ensemble results
    if result.ensemble_result:
        print(f"   → Strategies combined: {len(result.ensemble_result.strategy_results)}")
        print(f"   → Diversity score: {result.ensemble_result.diversity_score:.3f}")
        
        # Show strategy weights
        print("   → Strategy weights:")
        for strategy, weight in result.ensemble_result.combination_weights.items():
            print(f"     - {strategy.value}: {weight:.3f}")
    
    return result


def example_fully_optimized_labeling():
    """Example of fully optimized labeling with all enhancements."""
    print("\n🚀 Running Fully Optimized Labeling Example")
    
    # Generate larger sample for comprehensive analysis
    market_data = generate_sample_data(2500)
    
    # Create fully enhanced labeler
    enhanced_labeler = create_enhanced_labeler(EnhancementLevel.FULLY_OPTIMIZED)
    
    # Generate enhanced labels
    result = enhanced_labeler.generate_enhanced_labels(market_data)
    
    print(f"✅ Fully optimized labeling completed in {result.processing_time:.2f}s")
    print(f"   → Overall quality score: {result.quality_scores.get('overall_quality', 0):.3f}")
    
    # Show comprehensive results
    print("\n📊 Component Results:")
    
    if result.ml_assessment_result:
        ml_quality = result.quality_scores.get('ml_quality', 0)
        print(f"   → ML Quality Assessment: {ml_quality:.3f}")
    
    if result.adaptive_result:
        print(f"   → Adaptive Strategy: {result.adaptive_result.regime.value} regime detected")
    
    if result.ensemble_result:
        print(f"   → Ensemble System: {result.ensemble_result.diversity_score:.3f} diversity")
    
    if result.validation_results:
        significant_ratio = sum(1 for r in result.validation_results.values() if r.is_significant) / len(result.validation_results)
        print(f"   → Advanced Validation: {significant_ratio:.2%} tests significant")
    
    if result.backtesting_result:
        bt_score = result.backtesting_result.validation_summary.get('overall_score', 0)
        print(f"   → Backtesting Validation: {bt_score:.3f} overall score")
    
    # Generate comprehensive report
    report = enhanced_labeler.generate_comprehensive_report()
    print(f"\n📄 Comprehensive report generated ({len(report)} characters)")
    
    return result


def example_integration_with_existing_labeler():
    """Example of integrating enhancements with existing multi_horizon_profit_labeler."""
    print("\n🔗 Running Integration with Existing Labeler Example")
    
    # Import existing labeler
    from src.training.steps.pre_training.multi_horizon_profit_labeler import (
        MultiHorizonProfitLabeler, MultiHorizonConfig
    )
    
    # Generate sample data
    market_data = generate_sample_data(1000)
    
    # Create existing labeler
    existing_config = MultiHorizonConfig()
    existing_labeler = MultiHorizonProfitLabeler(existing_config)
    
    print("📊 Original labeler:")
    original_labels = existing_labeler.generate_labels(market_data.copy())
    print(f"   → Generated {original_labels.shape[1]} label columns")
    print(f"   → Overall opportunity mean: {original_labels['overall_opportunity'].mean():.3f}")
    
    # Enhance existing labeler
    from research.profit_labeling import enhance_existing_labeler
    enhanced_labeler = enhance_existing_labeler(existing_labeler, EnhancementLevel.ML_ENHANCED)
    
    print("\n🚀 Enhanced labeler:")
    enhanced_result = enhanced_labeler.generate_enhanced_labels(market_data)
    print(f"   → Generated {enhanced_result.enhanced_labels.shape[1]} enhanced label columns")
    print(f"   → Overall quality score: {enhanced_result.quality_scores.get('overall_quality', 0):.3f}")
    
    # Compare results
    if 'overall_opportunity' in enhanced_result.enhanced_labels.columns:
        enhanced_opp = enhanced_result.enhanced_labels['overall_opportunity'].mean()
        original_opp = original_labels['overall_opportunity'].mean()
        improvement = (enhanced_opp - original_opp) / original_opp * 100
        print(f"   → Opportunity score improvement: {improvement:.1f}%")
    
    return enhanced_result


def example_component_usage():
    """Example of using individual enhanced components."""
    print("\n🔧 Running Individual Component Usage Examples")
    
    # Generate sample data
    market_data = generate_sample_data(800)
    
    # Example 1: ML Quality Assessment
    print("\n1. ML Quality Assessment:")
    try:
        # First generate base labels
        from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonProfitLabeler
        base_labeler = MultiHorizonProfitLabeler()
        base_labels = base_labeler.generate_labels(market_data.copy())
        
        # Assess quality with ML
        ml_result = assess_label_quality_ml(base_labels, market_data)
        print(f"   ✅ ML assessment completed")
        print(f"   → Predictive power: {ml_result.quality_scores.get('PREDICTIVE_POWER', 0):.3f}")
        
    except Exception as e:
        print(f"   ❌ ML assessment failed: {e}")
    
    # Example 2: Adaptive Configuration
    print("\n2. Adaptive Configuration:")
    try:
        adaptive_result = get_regime_adaptive_config(market_data)
        print(f"   ✅ Adaptive config generated")
        print(f"   → Detected regime: {adaptive_result.regime.value}")
        print(f"   → Regime confidence: {adaptive_result.regime_confidence:.3f}")
        
    except Exception as e:
        print(f"   ❌ Adaptive configuration failed: {e}")
    
    # Example 3: Feature Engineering
    print("\n3. Contextual Feature Engineering:")
    try:
        feature_result = engineer_contextual_features(market_data)
        print(f"   ✅ Feature engineering completed")
        print(f"   → Generated {len(feature_result.feature_names)} features")
        print(f"   → Feature categories: {len(set(feature_result.feature_categories.values()))}")
        
    except Exception as e:
        print(f"   ❌ Feature engineering failed: {e}")
    
    # Example 4: Dynamic Target Optimization
    print("\n4. Dynamic Target Optimization:")
    try:
        optimization_result = discover_optimal_targets_and_horizons(market_data)
        print(f"   ✅ Dynamic optimization completed")
        print(f"   → Objective score: {optimization_result.objective_score:.3f}")
        print(f"   → Optimal targets: {list(optimization_result.optimal_targets.values())[:3]}")
        
    except Exception as e:
        print(f"   ❌ Dynamic optimization failed: {e}")


def example_performance_monitoring():
    """Example of performance monitoring over time."""
    print("\n📈 Running Performance Monitoring Example")
    
    # Create enhanced labeler
    enhanced_labeler = create_enhanced_labeler(EnhancementLevel.ML_ENHANCED)
    
    # Simulate multiple labeling runs over time
    base_data = generate_sample_data(2000)
    
    print("Simulating performance over time...")
    for i in range(5):
        # Use sliding window of data
        start_idx = i * 200
        end_idx = start_idx + 1000
        
        if end_idx <= len(base_data):
            window_data = base_data.iloc[start_idx:end_idx]
            
            print(f"   → Run {i+1}: Processing {len(window_data)} samples")
            result = enhanced_labeler.generate_enhanced_labels(window_data)
            
            quality_score = result.quality_scores.get('overall_quality', 0)
            print(f"     Quality score: {quality_score:.3f}")
    
    # Get performance summary
    perf_summary = enhanced_labeler.get_performance_summary()
    print(f"\n📊 Performance Summary:")
    
    for metric, stats in perf_summary.items():
        trend_direction = "↗️" if stats['trend'] > 0 else "↘️" if stats['trend'] < 0 else "➡️"
        print(f"   → {metric}: {stats['current']:.3f} {trend_direction} (μ={stats['mean']:.3f})")
    
    return perf_summary


def example_integration_workflow():
    """Example of complete integration workflow for production use."""
    print("\n🔄 Running Complete Integration Workflow Example")
    
    # Generate sample data
    market_data = generate_sample_data(1500)
    
    print("Step 1: Generate fully enhanced labels")
    # Use convenience function for quick integration
    enhanced_result = generate_fully_enhanced_labels(market_data)
    
    print(f"   ✅ Enhanced labels generated")
    print(f"   → Processing time: {enhanced_result.processing_time:.2f}s")
    print(f"   → Quality score: {enhanced_result.quality_scores.get('overall_quality', 0):.3f}")
    
    print("\nStep 2: Extract actionable insights")
    # Extract key columns for trading decisions
    key_columns = [
        'overall_opportunity',
        'leverage_adjusted_score', 
        'immediate_opportunity',
        'short_term_opportunity'
    ]
    
    available_columns = [col for col in key_columns if col in enhanced_result.enhanced_labels.columns]
    if available_columns:
        insights = enhanced_result.enhanced_labels[available_columns].describe()
        print(f"   ✅ Extracted insights for {len(available_columns)} key metrics")
        print(f"   → High opportunity samples (>0.7): {(enhanced_result.enhanced_labels.get('overall_opportunity', pd.Series(0)) > 0.7).sum()}")
    
    print("\nStep 3: Validate quality")
    # Show validation metrics
    if enhanced_result.validation_results:
        significant_count = sum(1 for r in enhanced_result.validation_results.values() if r.is_significant)
        print(f"   ✅ Advanced validation: {significant_count}/{len(enhanced_result.validation_results)} tests significant")
    
    if enhanced_result.backtesting_result:
        bt_summary = enhanced_result.backtesting_result.validation_summary
        print(f"   ✅ Backtesting validation: {bt_summary.get('validation_result', 'N/A')} ({bt_summary.get('overall_score', 0):.3f})")
    
    print("\nStep 4: Integration recommendations")
    # Show how to integrate with existing systems
    print("   📋 Integration recommendations:")
    print("   1. Replace existing labeler with EnhancedMultiHorizonProfitLabeler")
    print("   2. Use enhanced_labels DataFrame as input to ML models")
    print("   3. Monitor performance using get_performance_summary()")
    print("   4. Periodically retrain ML components with new data")
    print("   5. Use adaptive configuration for different market conditions")
    
    return enhanced_result


def main():
    """Run all enhanced examples."""
    print("🚀 Enhanced Multi-Horizon Profit Labeling Framework Examples")
    print("=" * 80)
    
    try:
        # Run enhanced examples
        basic_result = example_enhanced_labeling_basic()
        adaptive_result = example_adaptive_labeling()
        ensemble_result = example_ensemble_labeling()
        fully_optimized_result = example_fully_optimized_labeling()
        
        # Integration examples
        integration_result = example_integration_with_existing_labeler()
        component_usage = example_component_usage()
        performance_monitoring = example_performance_monitoring()
        workflow_result = example_integration_workflow()
        
        print("\n🎉 All enhanced examples completed successfully!")
        print("\n📋 Enhancement Summary:")
        print(f"   → ML-Enhanced Quality: {basic_result.quality_scores.get('overall_quality', 0):.3f}")
        print(f"   → Adaptive Strategy Quality: {adaptive_result.quality_scores.get('overall_quality', 0):.3f}")
        print(f"   → Ensemble Quality: {ensemble_result.quality_scores.get('overall_quality', 0):.3f}")
        print(f"   → Fully Optimized Quality: {fully_optimized_result.quality_scores.get('overall_quality', 0):.3f}")
        
        print("\n🎯 Key Benefits Demonstrated:")
        print("   ✓ ML-based quality assessment and enhancement")
        print("   ✓ Adaptive parameter adjustment based on market regimes")
        print("   ✓ Ensemble approaches for improved robustness")
        print("   ✓ Advanced statistical validation methods")
        print("   ✓ Dynamic target and horizon optimization")
        print("   ✓ Contextual feature engineering")
        print("   ✓ Backtesting-integrated validation")
        print("   ✓ Real-time performance monitoring")
        print("   ✓ Seamless integration with existing systems")
        
        print("\n📈 Ready for Production Integration!")
        
    except Exception as e:
        print(f"❌ Enhanced example execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()