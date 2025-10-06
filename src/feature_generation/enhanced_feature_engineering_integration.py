"""
Enhanced Feature Engineering Integration

This module demonstrates how to integrate all the new feature engineering capabilities:
- Normalization & stationarity features
- Cross-timeframe aggregation improvements
- Interaction & composite features
- Representation learning features

Usage:
    from src.feature_generation.enhanced_feature_engineering_integration import EnhancedFeatureEngineer

    engineer = EnhancedFeatureEngineer()
    features = await engineer.generate_comprehensive_features(data_dict)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
import logging
from datetime import datetime

from .core.feature_generator import FeatureCategory
from .categories import (
    # Normalization features
    NormalizationFeatureGenerator,
    RollingZScoreGenerator,
    VolatilityScalingGenerator,
    CrossSectionalNormalizer,
    # Cross-timeframe features
    CrossTimeframeFractionalChangeGenerator,
    CrossTimeframeAlignmentGenerator,
    CrossTimeframeLearnedProjectionGenerator,
    # Interaction features
    RegimeDependentFeatureGenerator,
    CointegrationResidualGenerator,
    StructuralRatioGenerator,
    PairwiseInteractionGenerator,
    # Representation learning features
    PatchTSTRepresentationGenerator,
    TFTEncoderRepresentationGenerator,
    AutoencoderRepresentationGenerator,
    ContrastiveLearningGenerator,
)

logger = logging.getLogger(__name__)


class EnhancedFeatureEngineer:
    """Enhanced feature engineering system integrating all new capabilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize feature generators
        self.normalization_generators = self._initialize_normalization_generators()
        self.cross_timeframe_generators = self._initialize_cross_timeframe_generators()
        self.interaction_generators = self._initialize_interaction_generators()
        self.representation_generators = self._initialize_representation_generators()

        self.logger.info("🚀 Enhanced Feature Engineer initialized with all new capabilities")

    def _initialize_normalization_generators(self) -> List:
        """Initialize normalization feature generators."""
        generators = [
            NormalizationFeatureGenerator(),
            RollingZScoreGenerator(window=20, column="close"),
            RollingZScoreGenerator(window=50, column="close"),
            VolatilityScalingGenerator(window=20, column="close"),
            CrossSectionalNormalizer(group_by="price", method="zscore"),
        ]
        return generators

    def _initialize_cross_timeframe_generators(self) -> List:
        """Initialize enhanced cross-timeframe feature generators."""
        generators = [
            CrossTimeframeFractionalChangeGenerator(fast_tf=5, slow_tf=15, feature_type="volatility"),
            CrossTimeframeFractionalChangeGenerator(fast_tf=1, slow_tf=5, feature_type="momentum"),
            CrossTimeframeAlignmentGenerator(source_tf=1, target_tf=5, alignment_method="lag"),
            CrossTimeframeLearnedProjectionGenerator(timeframes=[1, 5, 15], n_components=3),
        ]
        return generators

    def _initialize_interaction_generators(self) -> List:
        """Initialize interaction feature generators."""
        generators = [
            RegimeDependentFeatureGenerator(regime_detector="volatility", feature_type="momentum"),
            CointegrationResidualGenerator(pair_assets=["BTCUSDT", "ETHUSDT"], window=60),
            StructuralRatioGenerator(ratio_type="bid_ask_imbalance", window=20),
            PairwiseInteractionGenerator(feature1="rsi", feature2="volume", interaction_type="product"),
        ]
        return generators

    def _initialize_representation_generators(self) -> List:
        """Initialize representation learning feature generators."""
        generators = [
            PatchTSTRepresentationGenerator(patch_length=16, num_patches=8, embedding_dim=64),
            TFTEncoderRepresentationGenerator(seq_length=60, hidden_size=64, num_heads=4),
            AutoencoderRepresentationGenerator(encoding_dim=32, sequence_length=60),
            ContrastiveLearningGenerator(embedding_dim=64, temperature=0.1),
        ]
        return generators

    async def generate_comprehensive_features(
        self,
        data_dict: Dict[str, pd.DataFrame],
        include_categories: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive features using all enhanced capabilities.

        Args:
            data_dict: Dictionary with timeframe -> DataFrame mapping
            include_categories: List of categories to include (if None, include all)

        Returns:
            Dictionary with timeframe -> enhanced features DataFrame mapping
        """
        if include_categories is None:
            include_categories = ["normalization", "cross_timeframe", "interaction", "representation"]

        try:
            self.logger.info(f"🎯 Generating comprehensive features for {len(data_dict)} timeframes")

            enhanced_features_dict = {}

            for timeframe, data in data_dict.items():
                if data.empty:
                    self.logger.warning(f"Empty data for {timeframe}, skipping")
                    continue

                self.logger.info(f"📊 Processing {timeframe} with {len(data)} samples")

                # Generate features by category
                features_dict = {}

                if "normalization" in include_categories:
                    features_dict.update(await self._generate_normalization_features(data))

                if "cross_timeframe" in include_categories:
                    features_dict.update(await self._generate_cross_timeframe_features(data))

                if "interaction" in include_categories:
                    features_dict.update(await self._generate_interaction_features(data))

                if "representation" in include_categories:
                    features_dict.update(await self._generate_representation_features(data))

                # Combine all features
                if features_dict:
                    combined_features = pd.DataFrame(features_dict, index=data.index)
                    enhanced_features_dict[timeframe] = combined_features

                    self.logger.info(f"✅ Generated {len(features_dict)} enhanced features for {timeframe}")
                else:
                    self.logger.warning(f"⚠️ No features generated for {timeframe}")

            return enhanced_features_dict

        except Exception as e:
            self.logger.exception(f"Error generating comprehensive features: {e}")
            return {}

    async def _generate_normalization_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate normalization features for a single timeframe."""
        features = {}

        try:
            for generator in self.normalization_generators:
                try:
                    feature_series = generator.generate_feature(data)
                    if feature_series is not None and not feature_series.empty:
                        features[generator.config.name] = feature_series.values
                except Exception as e:
                    self.logger.warning(f"Error generating {generator.config.name}: {e}")

            self.logger.info(f"Generated {len(features)} normalization features")
            return features

        except Exception as e:
            self.logger.error(f"Error in normalization feature generation: {e}")
            return {}

    async def _generate_cross_timeframe_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate cross-timeframe features for a single timeframe."""
        features = {}

        try:
            for generator in self.cross_timeframe_generators:
                try:
                    feature_series = generator.generate_feature(data)
                    if feature_series is not None and not feature_series.empty:
                        features[generator.config.name] = feature_series.values
                except Exception as e:
                    self.logger.warning(f"Error generating {generator.config.name}: {e}")

            self.logger.info(f"Generated {len(features)} cross-timeframe features")
            return features

        except Exception as e:
            self.logger.error(f"Error in cross-timeframe feature generation: {e}")
            return {}

    async def _generate_interaction_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate interaction features for a single timeframe."""
        features = {}

        try:
            for generator in self.interaction_generators:
                try:
                    feature_series = generator.generate_feature(data)
                    if feature_series is not None and not feature_series.empty:
                        features[generator.config.name] = feature_series.values
                except Exception as e:
                    self.logger.warning(f"Error generating {generator.config.name}: {e}")

            self.logger.info(f"Generated {len(features)} interaction features")
            return features

        except Exception as e:
            self.logger.error(f"Error in interaction feature generation: {e}")
            return {}

    async def _generate_representation_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate representation learning features for a single timeframe."""
        features = {}

        try:
            for generator in self.representation_generators:
                try:
                    feature_series = generator.generate_feature(data)
                    if feature_series is not None and not feature_series.empty:
                        features[generator.config.name] = feature_series.values
                except Exception as e:
                    self.logger.warning(f"Error generating {generator.config.name}: {e}")

            self.logger.info(f"Generated {len(features)} representation features")
            return features

        except Exception as e:
            self.logger.error(f"Error in representation feature generation: {e}")
            return {}

    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of available features."""
        return {
            "normalization_generators": len(self.normalization_generators),
            "cross_timeframe_generators": len(self.cross_timeframe_generators),
            "interaction_generators": len(self.interaction_generators),
            "representation_generators": len(self.representation_generators),
            "total_generators": (
                len(self.normalization_generators) +
                len(self.cross_timeframe_generators) +
                len(self.interaction_generators) +
                len(self.representation_generators)
            ),
            "initialization_time": datetime.now().isoformat()
        }

    def validate_features(self, features_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate generated features."""
        validation_results = {}

        for timeframe, features in features_dict.items():
            validation_results[timeframe] = {
                "num_features": len(features.columns),
                "num_samples": len(features),
                "missing_values": features.isnull().sum().sum(),
                "infinite_values": np.isinf(features).sum().sum(),
                "feature_names": features.columns.tolist()[:10],  # First 10 names
            }

        return validation_results


# Example usage function
def example_usage():
    """Example of how to use the enhanced feature engineering system."""

    # Sample data (replace with real market data)
    sample_data = {
        "1m": pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=1000, freq="1min"),
            "close": np.random.randn(1000).cumsum() + 100,
            "high": np.random.randn(1000).cumsum() + 102,
            "low": np.random.randn(1000).cumsum() + 98,
            "open": np.random.randn(1000).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 1000)
        }).set_index("timestamp"),
        "5m": pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=200, freq="5min"),
            "close": np.random.randn(200).cumsum() + 100,
            "high": np.random.randn(200).cumsum() + 102,
            "low": np.random.randn(200).cumsum() + 98,
            "open": np.random.randn(200).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 200)
        }).set_index("timestamp")
    }

    # Initialize enhanced feature engineer
    engineer = EnhancedFeatureEngineer()

    # Generate comprehensive features
    import asyncio
    features = asyncio.run(engineer.generate_comprehensive_features(sample_data))

    # Print summary
    summary = engineer.get_feature_summary()
    print(f"Generated features using {summary['total_generators']} generators")

    for timeframe, feature_df in features.items():
        print(f"{timeframe}: {len(feature_df.columns)} features, {len(feature_df)} samples")

    return features


if __name__ == "__main__":
    # Run example
    example_usage()