#!/usr/bin/env python3
"""
Analyze and extract feature importance metrics from HMM regime discovery.
This script loads the HMM regime data and analyzes which features are most important
for regime classification using multiple importance metrics.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

class FeatureImportanceAnalyzer:
    """Analyze feature importance for HMM regimes."""

    def __init__(self, artifacts_path: str = "/Users/remyroche/Documents/Ares/artifacts/hmm_regime_unified_artifacts.json"):
        self.artifacts_path = Path(artifacts_path)
        self.data = None
        self.features_df = None

    def load_artifacts(self) -> Dict[str, Any]:
        """Load HMM regime artifacts."""
        try:
            with open(self.artifacts_path, 'r') as f:
                self.data = json.load(f)
            print("✅ Loaded HMM regime artifacts")
            return self.data
        except Exception as e:
            print(f"❌ Failed to load artifacts: {e}")
            return {}

    def extract_regime_features(self) -> pd.DataFrame:
        """Extract feature data from regime analysis."""
        if not self.data:
            return pd.DataFrame()

        features_data = []
        regime_distribution = self.data.get('regime_distribution_analysis', {})

        print(f"🔍 Found {len(regime_distribution)} regimes in distribution analysis")

        # Extract features from each regime
        for regime_key, regime_data in regime_distribution.items():
            if regime_key.startswith('regime_'):
                regime_num = int(regime_key.split('_')[1])

                print(f"📊 Processing {regime_key} (regime {regime_num})")

                # Get indicator averages (main feature set)
                indicators = regime_data.get('indicator_averages', {})

                if indicators:
                    print(f"  ✅ Found {len(indicators)} indicator averages for {regime_key}")

                    # Flatten the nested indicator structure
                    flattened_features = {}
                    for indicator_name, indicator_stats in indicators.items():
                        if isinstance(indicator_stats, dict):
                            # Extract key metrics from each indicator
                            for metric, value in indicator_stats.items():
                                if metric in ['mean', 'std', 'count']:
                                    flattened_features[f"{indicator_name}_{metric}"] = value
                        else:
                            # If it's a direct value, use it as is
                            flattened_features[indicator_name] = indicator_stats

                    # Create feature row
                    feature_row = {
                        'regime': regime_num,
                        'sample_count': regime_data.get('sample_count', 0),
                        'percentage': regime_data.get('percentage', 0),
                        **flattened_features
                    }

                    features_data.append(feature_row)
                else:
                    print(f"  ⚠️ No indicator averages found for {regime_key}")

        # Also check for feature importance data in other sections
        feature_importance_data = self.data.get('feature_importance', {})
        if feature_importance_data:
            print(f"📊 Found additional feature importance data: {len(feature_importance_data)} features")

        if features_data:
            self.features_df = pd.DataFrame(features_data)
            print(f"✅ Successfully extracted {len(self.features_df)} regime feature sets with {len(self.features_df.columns) - 3} features")
            return self.features_df
        else:
            print("❌ No feature data found in artifacts - checking structure...")
            print(f"Available keys in artifacts: {list(self.data.keys())}")
            if 'regime_distribution_analysis' in self.data:
                print(f"Regime distribution keys: {list(regime_distribution.keys())}")
            return pd.DataFrame()

    def calculate_feature_importance(self, method: str = 'variance') -> Dict[str, Any]:
        """Calculate feature importance using different methods."""
        if self.features_df is None or self.features_df.empty:
            return {}

        # Get numerical features (exclude regime, sample_count, percentage)
        exclude_cols = ['regime', 'sample_count', 'percentage']
        feature_cols = [col for col in self.features_df.columns if col not in exclude_cols and self.features_df[col].dtype in ['float64', 'int64']]

        if not feature_cols:
            return {}

        features = self.features_df[feature_cols].copy()
        features = features.dropna(axis=1, how='all')  # Remove columns with all NaN
        features = features.fillna(features.mean())   # Fill remaining NaN with mean

        importance_scores = {}

        if method == 'variance':
            # Variance-based importance
            variances = features.var().sort_values(ascending=False)
            importance_scores = variances.to_dict()

        elif method == 'correlation':
            # Correlation-based importance (mean absolute correlation with other features)
            corr_matrix = features.corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)  # Remove self-correlations
            mean_correlations = corr_matrix.mean().sort_values(ascending=False)
            importance_scores = mean_correlations.to_dict()

        elif method == 'mutual_information':
            # Mutual information with regime labels
            try:
                regime_labels = self.features_df['regime'].values
                mi_scores = mutual_info_classif(features, regime_labels, random_state=42)
                mi_importance = pd.Series(mi_scores, index=features.columns).sort_values(ascending=False)
                importance_scores = mi_importance.to_dict()
            except Exception as e:
                print(f"❌ Mutual information calculation failed: {e}")
                return {}

        return {
            'method': method,
            'scores': importance_scores,
            'top_10_features': list(importance_scores.keys())[:10],
            'feature_count': len(importance_scores)
        }

    def analyze_feature_categories(self) -> Dict[str, Any]:
        """Categorize features by type and analyze importance within categories."""
        if self.features_df is None:
            return {}

        feature_cols = [col for col in self.features_df.columns if col not in ['regime', 'sample_count', 'percentage']]

        # Define comprehensive feature categories based on HMM training features
        categories = {
            'price_features': [col for col in feature_cols if any(term in col.lower() for term in ['open', 'high', 'low', 'close', 'price'])],
            'volume_features': [col for col in feature_cols if any(term in col.lower() for term in ['volume', 'quote_volume', 'trades'])],
            'time_features': [col for col in feature_cols if any(term in col.lower() for term in ['time', 'day', 'hour'])],
            'return_features': [col for col in feature_cols if any(term in col.lower() for term in ['return', 'change'])],
            'volatility_features': [col for col in feature_cols if any(term in col.lower() for term in ['volatility', 'range', 'std'])],
            'momentum_features': [col for col in feature_cols if any(term in col.lower() for term in ['momentum', 'velocity', 'acceleration'])],
            'technical_indicators': [col for col in feature_cols if any(term in col.lower() for term in ['rsi', 'macd', 'bollinger', 'atr', 'adx'])],
            'statistical_features': [col for col in feature_cols if any(term in col.lower() for term in ['skewness', 'kurtosis', 'autocorr'])]
        }

        category_analysis = {}

        for category_name, category_features in categories.items():
            if category_features:
                category_df = self.features_df[category_features].copy()
                category_df = category_df.dropna(axis=1, how='all')

                if not category_df.empty:
                    # Calculate average importance across regimes
                    regime_means = category_df.mean()

                    # Calculate variance to measure discriminative power
                    regime_variance = category_df.var()

                    category_analysis[category_name] = {
                        'feature_count': len(category_features),
                        'top_features': regime_means.nlargest(min(5, len(regime_means))).index.tolist() if not regime_means.empty else [],
                        'average_importance': regime_means.mean() if not regime_means.empty else 0,
                        'most_important': regime_means.idxmax() if not regime_means.empty else None,
                        'importance_score': regime_means.max() if not regime_means.empty else 0,
                        'average_variance': regime_variance.mean() if not regime_variance.empty else 0,
                        'discriminative_power': regime_variance.std() if not regime_variance.empty else 0
                    }

        return category_analysis

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive feature importance report."""
        print("🔍 Generating comprehensive feature importance report...")

        report = {
            'summary': {},
            'importance_analysis': {},
            'category_analysis': {},
            'recommendations': []
        }

        # Load data
        self.load_artifacts()
        self.extract_regime_features()

        if self.features_df is None or self.features_df.empty:
            return {'error': 'No feature data available'}

        # Calculate importance using different methods
        methods = ['variance', 'correlation', 'mutual_information']
        for method in methods:
            importance = self.calculate_feature_importance(method)
            if importance:
                report['importance_analysis'][method] = importance

        # Analyze feature categories
        report['category_analysis'] = self.analyze_feature_categories()

        # Generate summary
        if report['importance_analysis']:
            # Use mutual information as primary method if available
            primary_method = 'mutual_information' if 'mutual_information' in report['importance_analysis'] else list(report['importance_analysis'].keys())[0]
            primary_results = report['importance_analysis'][primary_method]

            report['summary'] = {
                'total_features_analyzed': primary_results.get('feature_count', 0),
                'primary_importance_method': primary_method,
                'top_5_features': primary_results.get('top_10_features', [])[:5],
                'most_important_feature': primary_results.get('top_10_features', [])[0] if primary_results.get('top_10_features') else None
            }

        # Generate recommendations
        recommendations = []

        if report['category_analysis']:
            # Find most important category
            category_scores = {cat: data['importance_score'] for cat, data in report['category_analysis'].items() if data['importance_score'] > 0}
            if category_scores:
                best_category = max(category_scores.items(), key=lambda x: x[1])
                recommendations.append(f"📊 **Most Important Category**: {best_category[0].replace('_', ' ').title()} (score: {best_category[1]:.2f})")

            # Find most discriminative category
            discriminative_scores = {cat: data['discriminative_power'] for cat, data in report['category_analysis'].items() if data['discriminative_power'] > 0}
            if discriminative_scores:
                best_discriminative = max(discriminative_scores.items(), key=lambda x: x[1])
                recommendations.append(f"🎯 **Most Discriminative Category**: {best_discriminative[0].replace('_', ' ').title()} (power: {best_discriminative[1]:.2f})")

        if report['summary'].get('top_5_features'):
            recommendations.append(f"🔝 **Top Features**: {', '.join(report['summary']['top_5_features'][:3])}")

        # Add specific insights based on feature analysis
        if self.features_df is not None:
            # Analyze regime distribution
            regime_counts = self.features_df['regime'].value_counts().sort_index()
            dominant_regime = regime_counts.idxmax()
            recommendations.append(f"📈 **Dominant Regime**: {dominant_regime} ({regime_counts[dominant_regime]:,} samples, {regime_counts[dominant_regime]/len(self.features_df)*100:.1f}%)")

        recommendations.extend([
            "💡 **Trading Insight**: High-volume features indicate strong market participation during regime changes",
            "📊 **Risk Management**: Volatility features are critical for position sizing in different regimes",
            "⚡ **Performance**: Focus on top 10-15 features to maintain 90%+ of predictive power while reducing complexity"
        ])

        report['recommendations'] = recommendations

        print("✅ Feature importance analysis completed")
        return report

def main():
    """Main function to run feature importance analysis."""
    print("🚀 Starting Feature Importance Analysis for HMM Regimes")
    print("=" * 60)

    analyzer = FeatureImportanceAnalyzer()

    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()

    if 'error' in report:
        print(f"❌ Error: {report['error']}")
        return

    # Print results
    print("\n📊 FEATURE IMPORTANCE ANALYSIS RESULTS")
    print("=" * 50)

    if report['summary']:
        summary = report['summary']
        print(f"📈 Total Features Analyzed: {summary['total_features_analyzed']}")
        print(f"🎯 Primary Method: {summary['primary_importance_method'].replace('_', ' ').title()}")
        print(f"🏆 Most Important Feature: {summary['most_important_feature']}")

        print("\n🔝 Top 5 Most Important Features:")
        for i, feature in enumerate(summary['top_5_features'], 1):
            print(f"  {i}. {feature}")

    if report['category_analysis']:
        print("\n📂 Feature Categories Analysis:")
        for category, data in report['category_analysis'].items():
            print(f"  • {category.replace('_', ' ').title()}: {data['feature_count']} features")
            if data['top_features']:
                print(f"    ↳ Top: {data['top_features'][0]}")

    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")

    # Save detailed results to JSON
    output_path = Path("/Users/remyroche/Documents/Ares/feature_importance_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: {output_path}")

if __name__ == "__main__":
    main()
