#!/usr/bin/env python3
"""
Example script demonstrating autoencoder feature importance analysis.

This script shows how to:
1. Generate autoencoder features
2. Analyze feature importance using multiple methods
3. Get recommendations for feature selection
4. Access analysis results programmatically
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(project_root))

from src.analyst.autoencoder_feature_generator import AutoencoderFeatureGenerator
from src.utils.logger import setup_logging


def create_sample_data(
    n_samples: int = 1000, n_features: int = 50
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create sample data for demonstration."""
    np.random.seed(42)

    # Create sample features
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Create sample labels (binary classification)
    labels = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    # Create sample regime labels
    regime_labels = np.random.choice(
        ["bull", "bear", "sideways"], size=n_samples, p=[0.4, 0.3, 0.3]
    )

    return features, labels, regime_labels


def demonstrate_feature_analysis():
    """Demonstrate the autoencoder feature importance analysis."""

    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting autoencoder feature importance analysis demonstration...")

    # Create sample data
    logger.info("📊 Creating sample data...")
    features_df, labels, regime_labels = create_sample_data(
        n_samples=1000, n_features=50
    )
    logger.info(
        f"📊 Sample data created: {features_df.shape[0]} samples, {features_df.shape[1]} features"
    )

    # Initialize autoencoder feature generator
    logger.info("🔧 Initializing autoencoder feature generator...")
    generator = AutoencoderFeatureGenerator()

    # Generate autoencoder features with analysis enabled
    logger.info("🎯 Generating autoencoder features with importance analysis...")
    enhanced_features = generator.generate_features(
        features_df=features_df,
        regime_name="demo_regime",
        labels=labels,
        regime_labels=regime_labels,
        enable_analysis=True,
    )

    logger.info(
        f"✅ Enhanced features generated: {enhanced_features.shape[1]} total features"
    )

    # Access analysis results
    logger.info("📊 Accessing analysis results...")
    analysis_results = generator.get_last_analysis_results()

    if analysis_results:
        logger.info("🎉 Feature importance analysis completed successfully!")

        # Get feature rankings
        logger.info("🏆 Getting feature rankings...")
        ensemble_ranking = generator.get_feature_ranking(method="ensemble")
        if not ensemble_ranking.empty:
            logger.info("📈 Top 10 autoencoder features by ensemble importance:")
            for i, row in ensemble_ranking.head(10).iterrows():
                logger.info(
                    f"   {i+1}. {row['feature']}: {row['ensemble_importance']:.4f}"
                )

        # Get stable features
        logger.info("📈 Getting stable features...")
        stable_features = generator.get_stable_features(threshold=0.7)
        logger.info(f"📊 Stable features (threshold=0.7): {len(stable_features)}")
        if stable_features:
            logger.info(f"   📊 Stable features: {stable_features[:5]}...")

        # Get high correlation features
        logger.info("📊 Getting high correlation features...")
        high_corr_features = generator.get_high_correlation_features(threshold=0.5)
        logger.info(
            f"📊 High correlation features (threshold=0.5): {len(high_corr_features)}"
        )
        if high_corr_features:
            logger.info(f"   📊 High correlation features: {high_corr_features[:5]}...")

        # Get recommendations
        logger.info("💡 Getting recommendations...")
        recommendations = generator.get_recommendations()
        logger.info(f"📊 Number of recommendations: {len(recommendations)}")
        for i, rec in enumerate(recommendations[:5], 1):
            logger.info(f"   {i}. {rec}")

        # Demonstrate accessing detailed analysis results
        logger.info("🔍 Accessing detailed analysis results...")

        if "summary_statistics" in analysis_results:
            summary = analysis_results["summary_statistics"]
            logger.info("📊 Summary Statistics:")
            logger.info(
                f"   📈 Mean importance: {summary.get('mean_importance', 0):.4f}"
            )
            logger.info(
                f"   📈 Mean correlation: {summary.get('mean_correlation', 0):.4f}"
            )
            logger.info(f"   📈 Mean stability: {summary.get('mean_stability', 0):.4f}")
            logger.info(
                f"   📈 Stable features count: {summary.get('stable_count', 0)}"
            )
            logger.info(
                f"   📈 High correlation features count: {summary.get('high_corr_count', 0)}"
            )

        if "regime_analysis" in analysis_results:
            regime_analysis = analysis_results["regime_analysis"]
            if "consistent_features" in regime_analysis:
                consistent_features = regime_analysis["consistent_features"]
                logger.info(
                    f"🔄 Regime-consistent features: {len(consistent_features)}"
                )
                if consistent_features:
                    logger.info(
                        f"   📊 Consistent features: {consistent_features[:5]}..."
                    )

        # Demonstrate feature selection based on analysis
        logger.info("🎯 Demonstrating feature selection based on analysis...")

        # Select features based on multiple criteria
        top_features = (
            ensemble_ranking.head(10)["feature"].tolist()
            if not ensemble_ranking.empty
            else []
        )
        stable_and_important = list(set(top_features) & set(stable_features))

        logger.info(f"📊 Feature selection results:")
        logger.info(f"   📈 Top 10 features by importance: {len(top_features)}")
        logger.info(f"   📈 Stable features: {len(stable_features)}")
        logger.info(
            f"   📈 Features that are both top and stable: {len(stable_and_important)}"
        )

        if stable_and_important:
            logger.info(f"   📊 Recommended features: {stable_and_important}")

    else:
        logger.warning(
            "⚠️ No analysis results available. Analysis may have failed or been disabled."
        )

    logger.info("✅ Autoencoder feature importance analysis demonstration completed!")


def demonstrate_configuration_options():
    """Demonstrate different configuration options for feature analysis."""

    logger = logging.getLogger(__name__)
    logger.info("🔧 Demonstrating configuration options...")

    # Example configuration with custom analysis settings
    custom_config = {
        "feature_analysis": {
            "enable_analysis": True,
            "high_correlation_threshold": 0.6,  # More strict
            "low_correlation_threshold": 0.05,  # More strict
            "stability_window": 50,  # Smaller window
            "stability_threshold": 0.8,  # Higher threshold
            "regime_analysis_enabled": True,
            "comparison_with_original": True,
        }
    }

    # Create generator with custom config
    generator = AutoencoderFeatureGenerator(config=custom_config)

    # Create sample data
    features_df, labels, regime_labels = create_sample_data(
        n_samples=500, n_features=30
    )

    # Generate features with custom analysis settings
    enhanced_features = generator.generate_features(
        features_df=features_df,
        regime_name="custom_config_demo",
        labels=labels,
        regime_labels=regime_labels,
        enable_analysis=True,
    )

    logger.info(f"✅ Custom configuration demonstration completed!")
    logger.info(f"📊 Enhanced features: {enhanced_features.shape[1]} total features")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_feature_analysis()
    print("\n" + "=" * 80 + "\n")
    demonstrate_configuration_options()

    print("\n🎉 All demonstrations completed successfully!")
    print("📚 Check the logs above for detailed analysis results.")
