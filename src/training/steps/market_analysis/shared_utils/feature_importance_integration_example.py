"""
Feature Importance Integration Example

This example demonstrates how to integrate feature importance analysis
into the broader market analysis pipeline for regime discovery.

This example shows:
1. Basic feature importance analysis
2. Pipeline integration with clustering
3. Component enhancement
4. Report generation and insights extraction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

# Import the feature importance integration modules
from .feature_importance_integration import (
    FeatureImportanceIntegrationManager, FeatureImportanceIntegrationConfig,
    FeatureImportancePipelineHook
)
from .feature_importance_pipeline_utils import (
    create_feature_importance_config_for_pipeline,
    analyze_features_for_clustering,
    enhance_pipeline_component_with_feature_importance,
    create_feature_importance_report_summary,
    extract_regime_insights_from_analysis,
    validate_feature_importance_integration
)
from .balanced_feature_extractor import analyze_regime_feature_importance

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_feature_importance_analysis(n_regimes: int = 3):
    """
    Example 1: Basic feature importance analysis for regime discovery.

    This demonstrates the core functionality of feature importance analysis
    for understanding which features are most discriminative between regimes.

    Args:
        n_regimes: Number of regimes to simulate (default: 3)
    """
    print("🔍 Example 1: Basic Feature Importance Analysis")
    print("=" * 60)

    # Create synthetic market data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    # Generate synthetic features
    features = np.random.randn(n_samples, n_features)

    # Create synthetic regime labels (simulating market regimes)
    regime_labels = np.random.choice(n_regimes, size=n_samples)

    # Create meaningful feature names
    feature_names = [
        'close_price', 'volume', 'price_volatility', 'momentum_5', 'momentum_10',
        'rsi_14', 'macd_signal', 'bb_upper', 'bb_lower', 'atr_14',
        'volume_ratio', 'price_position', 'trend_strength', 'volatility_ratio',
        'returns_1', 'returns_5', 'volume_surge', 'price_gap', 'support_level', 'resistance_level'
    ][:n_features]  # Use only as many as we have features

    # If we need more names, add generic ones
    while len(feature_names) < n_features:
        feature_names.append(f'feature_{len(feature_names)}')

    # Add some regime-specific characteristics to make it realistic
    for regime_id in range(n_regimes):
        regime_mask = regime_labels == regime_id
        # Make certain features more distinctive for each regime
        # Use modulo arithmetic to distribute characteristics across features
        base_feature_idx = (regime_id * 2) % n_features

        # Assign characteristics to the first few features for each regime
        for offset in range(min(2, n_features - base_feature_idx)):
            feature_idx = base_feature_idx + offset
            if feature_idx < n_features:
                # Add regime-specific characteristics
                if regime_id == 0:  # High volatility regime
                    if offset == 0:
                        features[regime_mask, feature_idx] += 2.0  # High volatility feature
                    elif offset == 1:
                        features[regime_mask, feature_idx] -= 1.5  # Low return feature
                elif regime_id == 1:  # Trending regime
                    if offset == 0:
                        features[regime_mask, feature_idx] += 1.8  # Strong trend feature
                    elif offset == 1:
                        features[regime_mask, feature_idx] += 1.2  # Momentum feature
                else:  # Other regimes (range-bound, mixed characteristics)
                    if offset == 0:
                        features[regime_mask, feature_idx] += 1.6  # Range feature
                    elif offset == 1:
                        features[regime_mask, feature_idx] -= 1.8  # Low volatility feature

    print(f"📊 Dataset: {n_samples} samples, {n_features} features, {n_regimes} regimes")

    # Perform feature importance analysis
    print("\n🔍 Performing feature importance analysis...")
    analysis_result = analyze_regime_feature_importance(
        features=features,
        feature_names=feature_names,
        regime_labels=regime_labels,
        method="mutual_information"
    )

    if analysis_result:
        # Display results
        print("✅ Analysis completed successfully!")
        print(f"📈 Found {len(analysis_result.get('feature_importance_ranking', []))} feature rankings")

        # Show top features
        feature_ranking = analysis_result.get('feature_importance_ranking', [])
        if feature_ranking:
            print("\n🏆 Top 5 Most Important Features:")
            for i, (feature_name, importance_score) in enumerate(feature_ranking[:5]):
                print(f"   {i+1}. {feature_name}: {importance_score:.4f}")

        # Show regime characteristics
        regime_profiles = analysis_result.get('regime_feature_profiles', {})
        print(f"\n🏛️ Regime Analysis ({len(regime_profiles)} regimes):")
        for regime_id, profile in regime_profiles.items():
            dominant_features = profile.get('dominant_features', [])[:3]
            print(f"   {regime_id}: dominated by {', '.join(dominant_features)}")

        # Show interpretation
        interpretation = analysis_result.get('interpretation', '')
        if interpretation:
            print(f"\n💡 Interpretation: {interpretation}")

        return analysis_result
    else:
        print("❌ Analysis failed")
        return None


def example_pipeline_integration():
    """
    Example 2: Full pipeline integration with clustering.

    This demonstrates how to integrate feature importance analysis
    into a complete clustering pipeline.
    """
    print("\n🔗 Example 2: Pipeline Integration with Clustering")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 15
    features = np.random.randn(n_samples, n_features)
    feature_names = [
        'close_price', 'volume', 'price_volatility', 'momentum_5', 'momentum_10',
        'rsi_14', 'macd_signal', 'bb_upper', 'bb_lower', 'atr_14',
        'volume_ratio', 'price_position', 'trend_strength', 'volatility_ratio',
        'returns_1'
    ][:n_features]

    while len(feature_names) < n_features:
        feature_names.append(f'feature_{len(feature_names)}')

    # Create regime labels
    regime_labels = np.random.choice(3, size=n_samples)

    # Create a simple clusterer (KMeans)
    from sklearn.cluster import KMeans
    clusterer = KMeans(n_clusters=3, random_state=42, n_init=10)

    print(f"📊 Dataset: {n_samples} samples, {n_features} features")
    print(f"🤖 Using clusterer: {type(clusterer).__name__}")

    # Configure feature importance for pipeline
    config = create_feature_importance_config_for_pipeline(
        enable_pre_clustering=True,
        enable_post_clustering=True,
        enable_regime_characterization=True,
        importance_methods=["mutual_information", "f_classif"]
    )

    print("⚙️ Configuration: Pre-clustering analysis, Post-clustering analysis, Regime characterization")

    # Run complete analysis pipeline
    pipeline_results = analyze_features_for_clustering(
        features=features,
        feature_names=feature_names,
        clusterer=clusterer,
        config=config
    )

    if pipeline_results and pipeline_results['summary']['analysis_completed']:
        print("✅ Pipeline analysis completed successfully!")

        # Show clustering results
        clustering_info = pipeline_results['clustering_results']['clustering_info']
        print(f"📊 Clustering: {clustering_info['n_clusters']} clusters found")

        # Show pre-clustering analysis
        pre_analysis = pipeline_results['pre_clustering_analysis']
        if pre_analysis:
            print("🔍 Pre-clustering analysis completed")

        # Show post-clustering analysis
        post_analysis = pipeline_results['post_clustering_analysis']
        if post_analysis:
            print("🔍 Post-clustering analysis completed")
            feature_ranking = post_analysis.get('feature_importance_ranking', [])
            if feature_ranking:
                print("🏆 Top 3 features:")
                for i, (name, score) in enumerate(feature_ranking[:3]):
                    print(f"   {i+1}. {name}: {score:.4f}")

        return pipeline_results
    else:
        print("❌ Pipeline analysis failed")
        return None


def example_component_enhancement():
    """
    Example 3: Enhancing existing pipeline components.

    This demonstrates how to enhance existing components with feature
    importance analysis without modifying their core logic.
    """
    print("\n🔧 Example 3: Component Enhancement")
    print("=" * 60)

    # Create a mock component
    class MockRegimeDiscoveryComponent:
        def __init__(self):
            self.name = "Mock Regime Discovery Component"

        def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
            """Mock execution method."""
            print(f"🔄 {self.name}: Processing data...")

            # Simulate some processing
            result = {
                'regime_predictions': np.random.choice(3, size=100),
                'execution_time': 1.5,
                'regime_count': 3,
                'success': True
            }

            print(f"✅ {self.name}: Processing completed")
            return result

    # Create component instance
    component = MockRegimeDiscoveryComponent()

    print(f"🏗️ Original component: {component.name}")

    # Enhance the component with feature importance
    enhanced_component = enhance_pipeline_component_with_feature_importance(
        component_instance=component,
        method_name="execute"
    )

    print("✨ Component enhanced with feature importance analysis")

    # Test the enhanced component
    test_data = {'market_data': pd.DataFrame({'close': np.random.randn(100)})}

    print("\n🧪 Testing enhanced component...")
    result = enhanced_component.execute(test_data)

    if result and 'feature_importance_summary' in result:
        print("✅ Feature importance integration successful!")
        print(f"📊 Feature importance summary: {result['feature_importance_summary']}")
    else:
        print("⚠️ Feature importance integration may have failed")

    return result


def example_report_generation():
    """
    Example 4: Report generation and insights extraction.

    This demonstrates how to generate comprehensive reports and extract
    actionable insights from feature importance analysis.
    """
    print("\n📊 Example 4: Report Generation and Insights")
    print("=" * 60)

    # Use results from previous examples
    analysis_results = example_basic_feature_importance_analysis()

    if not analysis_results:
        print("❌ Cannot generate report - analysis failed")
        return None

    print("\n📋 Generating comprehensive report...")

    # Create summary report
    report_summary = create_feature_importance_report_summary(
        analysis_results,
        include_detailed_profiles=True
    )

    if report_summary:
        print("✅ Report summary generated successfully!")

        # Display key insights
        key_insights = report_summary.get('key_insights', {})
        main_findings = key_insights.get('main_findings', [])

        print("\n💡 Key Findings:")
        for finding in main_findings:
            print(f"   • {finding}")

        # Show feature rankings
        feature_rankings = report_summary.get('feature_rankings', {})
        if feature_rankings:
            top_features = feature_rankings.get('top_10_features', [])[:5]
            print(f"\n🏆 Top Features: {', '.join(top_features)}")

        # Show regime characteristics
        regime_chars = report_summary.get('regime_characteristics', {})
        print(f"\n🏛️ Regime Characteristics ({len(regime_chars)} regimes analyzed)")

        return report_summary
    else:
        print("❌ Report generation failed")
        return None


def example_insights_extraction():
    """
    Example 5: Extracting actionable insights.

    This demonstrates how to extract actionable insights from feature
    importance analysis for trading strategy development.
    """
    print("\n🎯 Example 5: Actionable Insights Extraction")
    print("=" * 60)

    # Get analysis results
    analysis_results = example_basic_feature_importance_analysis()

    if not analysis_results:
        print("❌ Cannot extract insights - analysis failed")
        return None

    print("\n🔍 Extracting actionable insights...")

    # Extract insights
    insights = extract_regime_insights_from_analysis(
        analysis_results,
        top_k_features=3
    )

    if insights and 'error' not in insights:
        print("✅ Insights extraction completed!")

        # Display regime discriminators
        regime_discriminators = insights.get('regime_discriminators', {})
        print(f"\n🏛️ Regime Discriminators ({len(regime_discriminators)} regimes):")
        for regime_id, discriminators in regime_discriminators.items():
            print(f"   {regime_id}: {', '.join(discriminators)}")

        # Display stability indicators
        stability_indicators = insights.get('regime_stability_indicators', {})
        print("\n📊 Regime Stability:")
        for regime_id, stability in stability_indicators.items():
            variance_mean = stability.get('feature_variance_mean', 0)
            print(f"   {regime_id}: variance = {variance_mean:.4f}")

        # Display recommendations
        recommendations = insights.get('actionable_recommendations', [])
        print("\n💡 Actionable Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

        return insights
    else:
        print("❌ Insights extraction failed")
        return None


def example_validation():
    """
    Example 6: Validation of feature importance integration.

    This demonstrates how to validate that feature importance analysis
    is working correctly in your pipeline.
    """
    print("\n✅ Example 6: Feature Importance Validation")
    print("=" * 60)

    # Create test data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    test_features = np.random.randn(n_samples, n_features)
    test_labels = np.random.choice(3, size=n_samples)
    feature_names = [
        'close_price', 'volume', 'price_volatility', 'momentum_5', 'momentum_10',
        'rsi_14', 'macd_signal', 'bb_upper', 'bb_lower', 'atr_14'
    ][:n_features]

    while len(feature_names) < n_features:
        feature_names.append(f'feature_{len(feature_names)}')

    print(f"🧪 Test Dataset: {n_samples} samples, {n_features} features, {len(np.unique(test_labels))} regimes")

    # Run validation
    validation_results = validate_feature_importance_integration(
        test_features=test_features,
        test_labels=test_labels,
        feature_names=feature_names
    )

    print("\n🔍 Validation Results:")
    for key, value in validation_results.items():
        status = "✅" if value else "❌"
        print(f"   {key}: {status} {value}")

    if validation_results.get('integration_successful', False):
        print("\n🎉 Feature importance integration is working correctly!")
        print("   Ready for production use in your pipeline.")
    else:
        print("\n⚠️ Feature importance integration needs attention.")
        print(f"   Error: {validation_results.get('error', 'Unknown error')}")

    return validation_results


def main():
    """
    Run all examples demonstrating feature importance integration.
    """
    print("🚀 Feature Importance Integration Examples")
    print("=" * 70)
    print("This demonstrates comprehensive feature importance analysis")
    print("integration into the market analysis pipeline.\n")

    try:
        # Run all examples
        results = {}

        results['basic_analysis'] = example_basic_feature_importance_analysis()
        results['pipeline_integration'] = example_pipeline_integration()
        results['component_enhancement'] = example_component_enhancement()
        results['report_generation'] = example_report_generation()
        results['insights_extraction'] = example_insights_extraction()
        results['validation'] = example_validation()

        print("\n" + "=" * 70)
        print("🎉 All Examples Completed Successfully!")
        print("\n📚 Key Takeaways:")
        print("   1. Feature importance analysis reveals which features")
        print("      are most discriminative between market regimes")
        print("   2. Integration is seamless and enhances existing pipelines")
        print("   3. Results provide actionable insights for trading strategy")
        print("   4. Validation ensures reliable integration")
        print("   5. Reports can be customized for different use cases")

        return results

    except Exception as e:
        print(f"\n❌ Examples failed with error: {e}")
        return None


if __name__ == "__main__":
    main()
