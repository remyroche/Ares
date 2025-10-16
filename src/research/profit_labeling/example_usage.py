#!/usr/bin/env python3
"""
Example Usage of Multi-Horizon Profit Labeling Research Framework

This script demonstrates how to use the profit labeling research framework
to analyze and optimize labeling heuristics, similar to how we analyze
HMM clustering effectiveness.

Run this script to see the framework in action:
    python src/research/profit_labeling/example_usage.py
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import the research framework
from research.profit_labeling import (
    HeuristicAnalyzer,
    HeuristicAnalysisConfig,
    LabelingValidator,
    ValidationConfig,
    ParameterOptimizer,
    OptimizationConfig,
    OptimizationMethod,
    OptimizationObjective,
    LabelingVisualizer,
    VisualizationConfig,
    ResearchRunner,
    ResearchConfig,
    ResearchWorkflow,
    run_quick_profit_labeling_analysis
)

def generate_sample_market_data(n_samples: int = 2000) -> pd.DataFrame:
    """Generate realistic sample market data for demonstration."""
    print(f"📊 Generating {n_samples} samples of realistic market data...")

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate dates
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='5min')

    # Generate realistic price data with volatility clustering
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.002, n_samples)

    # Add volatility clustering (GARCH-like behavior)
    vol_persistence = 0.9
    volatility = np.zeros(n_samples)
    volatility[0] = 0.002

    for i in range(1, n_samples):
        # Volatility clustering
        volatility[i] = vol_persistence * volatility[i-1] + (1 - vol_persistence) * 0.002
        returns[i] = np.random.normal(0.0001, volatility[i])

    # Generate price series
    prices = [base_price]
    for ret in returns[:-1]:
        prices.append(prices[-1] * (1 + ret))

    # Create OHLCV data
    market_data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    }, index=dates)

    # Ensure OHLC consistency
    for i in range(len(market_data)):
        market_data.loc[market_data.index[i], 'high'] = max(
            market_data.iloc[i][['open', 'high', 'low', 'close']]
        )
        market_data.loc[market_data.index[i], 'low'] = min(
            market_data.iloc[i][['open', 'high', 'low', 'close']]
        )

    print(f"   ✅ Generated data: {len(market_data)} samples from {dates[0].date()} to {dates[-1].date()}")
    return market_data

def example_1_quick_analysis():
    """Example 1: Quick heuristic analysis for rapid insights."""
    print("\n" + "="*60)
    print("📈 EXAMPLE 1: Quick Heuristic Analysis")
    print("="*60)

    # Generate sample data
    market_data = generate_sample_market_data(1500)

    # Run quick analysis
    print("\n🔍 Running quick heuristic analysis...")
    result = run_quick_profit_labeling_analysis(market_data, "heuristic")

    print(f"   ✅ Analysis completed in {result.execution_time:.1f} seconds")

    if result.heuristic_results:
        print(f"   📊 Analyzed {len(result.heuristic_results)} heuristic components")

        # Show some key results
        for key, analysis_result in list(result.heuristic_results.items())[:3]:
            print(f"   → {key}: {analysis_result.metric_value:.3f} - {analysis_result.interpretation}")

    return result

def example_2_comprehensive_validation():
    """Example 2: Comprehensive validation testing."""
    print("\n" + "="*60)
    print("🔬 EXAMPLE 2: Comprehensive Validation Testing")
    print("="*60)

    # Generate sample data
    market_data = generate_sample_market_data(2000)

    # Configure comprehensive validation
    validation_config = ValidationConfig(
        validate_consistency=True,
        validate_stability=True,
        validate_predictiveness=True,
        validate_significance=True,
        validate_bias=True,
        significance_level=0.05,
        bootstrap_iterations=100,  # Reduced for demo speed
        confidence_level=0.95
    )

    print("\n🧪 Running comprehensive validation testing...")
    validator = LabelingValidator(validation_config)
    validation_results = validator.validate_labeling_quality(market_data)

    print(f"   ✅ Validation completed: {len(validation_results)} tests")

    # Analyze results
    significant_count = sum(1 for r in validation_results.values() if r.is_significant)
    print(f"   📊 Statistical significance: {significant_count}/{len(validation_results)} tests significant")

    # Show key validation results
    for key, result in list(validation_results.items())[:3]:
        status = "✅ PASS" if result.is_significant else "⚠️ REVIEW"
        print(f"   → {key}: {result.value:.3f} {status}")
        print(f"     {result.interpretation}")

    return validation_results

def example_3_parameter_optimization():
    """Example 3: Parameter optimization study."""
    print("\n" + "="*60)
    print("🎯 EXAMPLE 3: Parameter Optimization Study")
    print("="*60)

    # Generate sample data
    market_data = generate_sample_market_data(1800)

    # Configure optimization
    optimization_config = OptimizationConfig(
        method=OptimizationMethod.RANDOM_SEARCH,  # Fast method for demo
        objective=OptimizationObjective.PREDICTIVE_POWER,
        profit_targets_range={
            'micro': (0.002, 0.005),    # 0.2% to 0.5%
            'small': (0.003, 0.008),    # 0.3% to 0.8%
            'medium': (0.005, 0.012),   # 0.5% to 1.2%
            'good': (0.008, 0.020)      # 0.8% to 2.0%
        },
        time_horizons_range={
            'immediate': (1, 4),        # 1 to 4 periods
            'short': (2, 8)             # 2 to 8 periods
        },
        random_search_iterations=20,    # Reduced for demo speed
        validation_split=0.3
    )

    print("\n🚀 Running parameter optimization...")
    optimizer = ParameterOptimizer(optimization_config)
    optimization_result = optimizer.optimize_parameters(market_data)

    print(f"   ✅ Optimization completed in {optimization_result.metadata.get('optimization_time', 0):.1f} seconds")
    print(f"   🎯 Best score: {optimization_result.best_score:.4f}")
    print(f"   📈 Method: {optimization_result.method.value}")

    # Show best parameters
    print("\n   🔧 Best Parameters:")
    for param, value in optimization_result.best_params.items():
        if isinstance(value, float):
            print(f"      → {param}: {value:.4f}")
        else:
            print(f"      → {param}: {value}")

    return optimization_result

def example_4_comparative_analysis():
    """Example 4: Comparative analysis of different configurations."""
    print("\n" + "="*60)
    print("📊 EXAMPLE 4: Comparative Analysis")
    print("="*60)

    # Generate sample data
    market_data = generate_sample_market_data(1600)

    # Define test configurations
    from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonConfig

    test_configs = {
        'conservative': MultiHorizonConfig(),
        'aggressive': MultiHorizonConfig(),
        'quality_focused': MultiHorizonConfig()
    }

    # Conservative: smaller targets, shorter horizons
    test_configs['conservative'].profit_targets = {
        'micro': 0.002, 'small': 0.003, 'medium': 0.005, 'good': 0.008
    }
    test_configs['conservative'].time_horizons = {'immediate': 1, 'short': 2}

    # Aggressive: larger targets, longer horizons
    test_configs['aggressive'].profit_targets = {
        'micro': 0.005, 'small': 0.008, 'medium': 0.012, 'good': 0.020
    }
    test_configs['aggressive'].time_horizons = {'immediate': 3, 'short': 6}

    # Quality focused: enhanced quality scoring
    test_configs['quality_focused'].enable_quality_scoring = True
    test_configs['quality_focused'].speed_weight = 0.2
    test_configs['quality_focused'].risk_weight = 0.5
    test_configs['quality_focused'].profitability_weight = 0.3

    print(f"\n🔍 Comparing {len(test_configs)} different configurations...")

    comparison_results = {}
    analyzer = HeuristicAnalyzer(HeuristicAnalysisConfig(
        bootstrap_samples=50  # Reduced for demo speed
    ))

    for config_name, config in test_configs.items():
        print(f"   → Testing {config_name} configuration...")
        results = analyzer.analyze_labeling_heuristics(market_data, config)

        # Calculate average effectiveness
        effectiveness_scores = [
            r.metric_value for r in results.values()
            if 'effectiveness' in str(r.analysis_type)
        ]
        avg_effectiveness = np.mean(effectiveness_scores) if effectiveness_scores else 0.0

        comparison_results[config_name] = {
            'avg_effectiveness': avg_effectiveness,
            'components_analyzed': len(results)
        }

        print(f"      ✅ Average effectiveness: {avg_effectiveness:.3f}")

    # Find best configuration
    best_config = max(comparison_results.items(), key=lambda x: x[1]['avg_effectiveness'])
    print(f"\n   🏆 Best configuration: {best_config[0]} (effectiveness: {best_config[1]['avg_effectiveness']:.3f})")

    return comparison_results

def example_5_complete_research_pipeline():
    """Example 5: Complete research pipeline with all components."""
    print("\n" + "="*60)
    print("🔄 EXAMPLE 5: Complete Research Pipeline")
    print("="*60)

    # Generate sample data
    market_data = generate_sample_market_data(2500)

    # Configure complete research pipeline
    research_config = ResearchConfig(
        workflows=[ResearchWorkflow.COMPLETE_PIPELINE],
        generate_reports=True,
        generate_visualizations=False,  # Disable for demo
        output_dir="example_research_output",
        # Reduced settings for demo speed
        heuristic_config=HeuristicAnalysisConfig(bootstrap_samples=50),
        validation_config=ValidationConfig(bootstrap_iterations=50),
        optimization_config=OptimizationConfig(
            method=OptimizationMethod.RANDOM_SEARCH,
            random_search_iterations=10
        )
    )

    print("\n🚀 Running complete research pipeline...")
    print("   This includes: Heuristic Analysis → Validation → Optimization → Reporting")

    runner = ResearchRunner(research_config)
    results = runner.run_research(market_data)

    # Analyze complete results
    complete_result = results['complete_pipeline']
    print(f"\n   ✅ Complete pipeline finished in {complete_result.execution_time:.1f} seconds")

    # Summary of results
    if complete_result.heuristic_results:
        print(f"   📈 Heuristic Analysis: {len(complete_result.heuristic_results)} components analyzed")

    if complete_result.validation_results:
        significant_count = sum(1 for r in complete_result.validation_results.values()
                              if r.is_significant)
        print(f"   🔬 Validation: {significant_count}/{len(complete_result.validation_results)} tests significant")

    if complete_result.optimization_results:
        best_result = max(complete_result.optimization_results.values(),
                         key=lambda x: x.best_score)
        print(f"   🎯 Optimization: Best score {best_result.best_score:.4f}")

    print(f"\n   📄 Research report saved to: {research_config.output_dir}/")

    return results

def main():
    """Run all examples demonstrating the research framework."""
    print("🔬 Multi-Horizon Profit Labeling Research Framework Examples")
    print("=" * 80)
    print("This demonstration shows how to analyze profit labeling heuristics")
    print("from a data-driven perspective, similar to HMM clustering research.")
    print("=" * 80)

    try:
        # Run examples
        example_1_quick_analysis()
        example_2_comprehensive_validation()
        example_3_parameter_optimization()
        example_4_comparative_analysis()
        example_5_complete_research_pipeline()

        # Summary
        print("\n" + "="*60)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📚 What you've seen:")
        print("   1. ⚡ Quick heuristic analysis for rapid insights")
        print("   2. 🔬 Comprehensive validation testing")
        print("   3. 🎯 Parameter optimization studies")
        print("   4. 📊 Comparative analysis of configurations")
        print("   5. 🔄 Complete end-to-end research pipeline")

        print("\n💡 Next Steps:")
        print("   → Use this framework to analyze your own market data")
        print("   → Customize configurations for your specific use cases")
        print("   → Integrate findings into your ML training pipeline")
        print("   → Set up automated research runs for different market conditions")

        print(f"\n📁 Example outputs saved to: ./example_research_output/")
        print("   Check the generated reports and analysis results!")

    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        print("   This might be due to missing dependencies or configuration issues.")
        print("   Please check the framework installation and requirements.")
        raise

if __name__ == '__main__':
    main()
