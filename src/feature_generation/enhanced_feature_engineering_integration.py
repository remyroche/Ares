"""
Enhanced Feature Engineering Integration

This module provides comprehensive integration of all enhanced feature engineering
capabilities, fully wiring them with the existing infrastructure so that the
existing pipeline natively benefits from the upgrades.

Features integrated:
- Enhanced normalization & stationarity features
- Advanced cross-timeframe aggregation with proper lag handling
- Interaction & composite features with regime awareness
- Representation learning with PatchTST, TFT, and autoencoders
- Seamless integration with existing FeatureBank system
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from .core.feature_generator import FeatureCategory, FeatureGenerator
from .core.feature_bank import FeatureBank, FeatureBankConfig
from .categories.enhanced_normalization import (
    EnhancedNormalizationFeatureGenerator,
    AdvancedRollingZScoreGenerator,
    RegimeAwareNormalizer,
    create_enhanced_normalization_generators
)
from .categories.enhanced_cross_timeframe import (
    EnhancedCrossTimeframeFeatureGenerator,
    FractionalChangeGenerator,
    CrossTimeframeAlignmentGenerator,
    LearnedProjectionGenerator,
    create_enhanced_cross_timeframe_generators
)
from .categories.enhanced_interaction import (
    EnhancedInteractionFeatureGenerator,
    PairwiseInteractionGenerator,
    RegimeDependentFeatureGenerator,
    StructuralRatioGenerator,
    create_enhanced_interaction_generators
)
from .categories.enhanced_representation_learning import (
    EnhancedRepresentationLearningGenerator,
    PatchTSTRepresentationGenerator,
    TFTEncoderRepresentationGenerator,
    create_enhanced_representation_learning_generators
)

logger = logging.getLogger(__name__)


class EnhancedFeatureEngineeringIntegration:
    """
    Enhanced feature engineering integration system that seamlessly integrates
    all new capabilities with the existing infrastructure.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced feature engineering integration system.
        
        Args:
            config: Configuration dictionary for the integration system
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize enhanced feature generators
        self.enhanced_generators = self._initialize_enhanced_generators()
        
        # Initialize feature bank with enhanced capabilities
        self.feature_bank = self._initialize_enhanced_feature_bank()
        
        # Performance tracking
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'categories_used': set(),
            'features_generated': 0,
            'average_generation_time': 0.0,
            'enhanced_features_generated': 0,
            'integration_time': datetime.now().isoformat()
        }

        self.logger.info("🚀 Enhanced Feature Engineering Integration initialized")

    def _initialize_enhanced_generators(self) -> Dict[str, List[FeatureGenerator]]:
        """Initialize all enhanced feature generators."""
        generators = {
            'normalization': create_enhanced_normalization_generators(),
            'cross_timeframe': create_enhanced_cross_timeframe_generators(),
            'interaction': create_enhanced_interaction_generators(),
            'representation_learning': create_enhanced_representation_learning_generators()
        }
        
        total_generators = sum(len(gen_list) for gen_list in generators.values())
        self.logger.info(f"📊 Initialized {total_generators} enhanced generators across {len(generators)} categories")
        
        return generators

    def _initialize_enhanced_feature_bank(self) -> FeatureBank:
        """Initialize feature bank with enhanced capabilities."""
        # Create enhanced feature bank configuration
        bank_config = FeatureBankConfig(
            enable_matrix_operations=True,
            enable_gpu_acceleration=self.config.get('enable_gpu', False),
            enable_lookback_optimization=True,
            enable_parallel_processing=True,
            max_workers=self.config.get('max_workers', 4),
            memory_efficient=True,
            cache_results=True,
            auto_normalize=True,
            normalization_method='zscore'
        )
        
        # Initialize feature bank
        feature_bank = FeatureBank(bank_config)
        
        # Register enhanced generators
        self._register_enhanced_generators(feature_bank)
        
        return feature_bank

    def _register_enhanced_generators(self, feature_bank: FeatureBank) -> None:
        """Register enhanced generators with the feature bank."""
        total_registered = 0
        
        for category, generators in self.enhanced_generators.items():
            for generator in generators:
                try:
                    feature_bank.register_generator(generator)
                    total_registered += 1
                except Exception as e:
                    self.logger.warning(f"Failed to register {generator.config.name}: {e}")
        
        self.logger.info(f"✅ Registered {total_registered} enhanced generators with feature bank")

    async def generate_comprehensive_features(
        self,
        data_dict: Dict[str, pd.DataFrame],
        include_categories: Optional[List[str]] = None,
        enable_enhanced: bool = True,
        parallel_processing: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive features using all enhanced capabilities.

        Args:
            data_dict: Dictionary with timeframe -> DataFrame mapping
            include_categories: List of categories to include (if None, include all)
            enable_enhanced: Whether to use enhanced features
            parallel_processing: Whether to use parallel processing

        Returns:
            Dictionary with timeframe -> enhanced features DataFrame mapping
        """
        if include_categories is None:
            include_categories = ["normalization", "cross_timeframe", "interaction", "representation_learning"]

        try:
            self.logger.info(f"🎯 Generating comprehensive features for {len(data_dict)} timeframes")
            self.logger.info(f"📊 Categories: {include_categories}")
            self.logger.info(f"🚀 Enhanced features: {enable_enhanced}")

            enhanced_features_dict = {}

            if parallel_processing and len(data_dict) > 1:
                # Parallel processing for multiple timeframes
                enhanced_features_dict = await self._generate_features_parallel(
                    data_dict, include_categories, enable_enhanced
                )
            else:
                # Sequential processing
                for timeframe, data in data_dict.items():
                    if data.empty:
                        self.logger.warning(f"Empty data for {timeframe}, skipping")
                        continue

                    self.logger.info(f"📊 Processing {timeframe} with {len(data)} samples")

                    # Generate features
                    features_df = await self._generate_features_for_timeframe(
                        data, include_categories, enable_enhanced
                    )

                    if not features_df.empty:
                        enhanced_features_dict[timeframe] = features_df
                        self.logger.info(f"✅ Generated {len(features_df.columns)} features for {timeframe}")

            # Update performance stats
            self._update_performance_stats(enhanced_features_dict)

            return enhanced_features_dict

        except Exception as e:
            self.logger.exception(f"Error generating comprehensive features: {e}")
            return {}

    async def _generate_features_parallel(
        self,
        data_dict: Dict[str, pd.DataFrame],
        include_categories: List[str],
        enable_enhanced: bool
    ) -> Dict[str, pd.DataFrame]:
        """Generate features using parallel processing."""
        enhanced_features_dict = {}

        with ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4)) as executor:
            # Submit all tasks
            future_to_timeframe = {
                executor.submit(
                    self._generate_features_for_timeframe,
                    data, include_categories, enable_enhanced
                ): timeframe
                for timeframe, data in data_dict.items()
                if not data.empty
            }

            # Collect results as they complete
            for future in as_completed(future_to_timeframe):
                timeframe = future_to_timeframe[future]
                try:
                    features_df = future.result()
                    if not features_df.empty:
                        enhanced_features_dict[timeframe] = features_df
                        self.logger.info(f"✅ Generated {len(features_df.columns)} features for {timeframe}")
                except Exception as e:
                    self.logger.error(f"Error generating features for {timeframe}: {e}")

        return enhanced_features_dict

    async def _generate_features_for_timeframe(
        self,
        data: pd.DataFrame,
        include_categories: List[str],
        enable_enhanced: bool
    ) -> pd.DataFrame:
        """Generate features for a single timeframe."""
        try:
            all_features = {}

            # Generate enhanced features if enabled
            if enable_enhanced:
                for category in include_categories:
                    if category in self.enhanced_generators:
                        category_features = await self._generate_category_features(
                            data, category, self.enhanced_generators[category]
                        )
                        all_features.update(category_features)

            # Generate standard features using feature bank
            standard_features = self._generate_standard_features(data, include_categories)
            all_features.update(standard_features)

            # Combine all features
            if all_features:
                features_df = pd.DataFrame(all_features, index=data.index)
                
                # Apply post-processing
                features_df = self._apply_post_processing(features_df)
                
                return features_df
            else:
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error generating features for timeframe: {e}")
            return pd.DataFrame()

    async def _generate_category_features(
        self,
        data: pd.DataFrame,
        category: str,
        generators: List[FeatureGenerator]
    ) -> Dict[str, np.ndarray]:
        """Generate features for a specific category."""
        features = {}

        for generator in generators:
            try:
                # Generate feature
                result = generator.generate(data)
                
                if result.success:
                    features[result.name] = result.data.values
                    self.performance_stats['enhanced_features_generated'] += 1
                else:
                    self.logger.warning(f"Generator {generator.config.name} failed: {result.error_message}")

            except Exception as e:
                self.logger.warning(f"Error generating {generator.config.name}: {e}")

        return features

    def _generate_standard_features(
        self,
        data: pd.DataFrame,
        include_categories: List[str]
    ) -> Dict[str, np.ndarray]:
        """Generate standard features using the feature bank."""
        try:
            # Convert category names to FeatureCategory enums
            category_enums = []
            for category in include_categories:
                try:
                    category_enum = FeatureCategory(category)
                    category_enums.append(category_enum)
                except ValueError:
                    self.logger.warning(f"Unknown category: {category}")

            # Generate features using feature bank
            features_df = self.feature_bank.generate_features(
                data,
                categories=category_enums,
                lookback_optimization=True
            )

            # Convert to dictionary
            return features_df.to_dict('series')

        except Exception as e:
            self.logger.error(f"Error generating standard features: {e}")
            return {}

    def _apply_post_processing(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply post-processing to generated features."""
        try:
            # Remove infinite values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with 0
            features_df = features_df.fillna(0)
            
            # Remove constant features
            constant_features = features_df.columns[features_df.nunique() <= 1]
            if len(constant_features) > 0:
                features_df = features_df.drop(columns=constant_features)
                self.logger.info(f"Removed {len(constant_features)} constant features")

            # Remove highly correlated features
            features_df = self._remove_highly_correlated_features(features_df)

            return features_df

        except Exception as e:
            self.logger.error(f"Error in post-processing: {e}")
            return features_df

    def _remove_highly_correlated_features(self, features_df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features."""
        try:
            # Calculate correlation matrix
            corr_matrix = features_df.corr().abs()
            
            # Find pairs of highly correlated features
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features to drop
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            
            if len(to_drop) > 0:
                features_df = features_df.drop(columns=to_drop)
                self.logger.info(f"Removed {len(to_drop)} highly correlated features")

            return features_df

        except Exception as e:
            self.logger.warning(f"Error removing correlated features: {e}")
            return features_df

    def _update_performance_stats(self, features_dict: Dict[str, pd.DataFrame]) -> None:
        """Update performance statistics."""
        self.performance_stats['total_generations'] += 1
        
        total_features = sum(len(df.columns) for df in features_dict.values())
        self.performance_stats['features_generated'] += total_features
        
        if total_features > 0:
            self.performance_stats['successful_generations'] += 1
        else:
            self.performance_stats['failed_generations'] += 1

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        stats['categories_used'] = list(stats['categories_used'])
        return stats

    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of available features."""
        summary = {
            'enhanced_generators': {
                category: len(generators) 
                for category, generators in self.enhanced_generators.items()
            },
            'total_enhanced_generators': sum(
                len(generators) for generators in self.enhanced_generators.values()
            ),
            'feature_bank_generators': len(self.feature_bank.registry.get_all()),
            'performance_stats': self.get_performance_stats(),
            'integration_time': self.performance_stats['integration_time']
        }
        
        return summary

    def validate_features(self, features_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate generated features."""
        validation_results = {}

        for timeframe, features in features_dict.items():
            validation_results[timeframe] = {
                "num_features": len(features.columns),
                "num_samples": len(features),
                "missing_values": features.isnull().sum().sum(),
                "infinite_values": np.isinf(features).sum().sum(),
                "constant_features": (features.nunique() <= 1).sum(),
                "feature_names": features.columns.tolist()[:10],  # First 10 names
                "data_types": features.dtypes.value_counts().to_dict(),
                "memory_usage_mb": features.memory_usage(deep=True).sum() / 1024 / 1024
            }

        return validation_results

    def generate_features_by_category(
        self,
        data: pd.DataFrame,
        category: str,
        enable_enhanced: bool = True
    ) -> pd.DataFrame:
        """
        Generate features for a specific category.
        
        Args:
            data: Input data DataFrame
            category: Feature category
            enable_enhanced: Whether to use enhanced features
            
        Returns:
            DataFrame with generated features
        """
        try:
            if enable_enhanced and category in self.enhanced_generators:
                # Use enhanced generators
                features = {}
                for generator in self.enhanced_generators[category]:
                    try:
                        result = generator.generate(data)
                        if result.success:
                            features[result.name] = result.data
                    except Exception as e:
                        self.logger.warning(f"Error generating {generator.config.name}: {e}")
                
                return pd.DataFrame(features, index=data.index)
            else:
                # Use standard feature bank
                return self.feature_bank.generate_features_by_category(data, category)
                
        except Exception as e:
            self.logger.error(f"Error generating features for category {category}: {e}")
            return pd.DataFrame()

    def get_available_categories(self) -> List[str]:
        """Get list of available feature categories."""
        return list(self.enhanced_generators.keys()) + [
            category.value for category in self.feature_bank.list_categories()
        ]

    def get_available_features(self, category: Optional[str] = None) -> List[str]:
        """Get list of available features."""
        if category and category in self.enhanced_generators:
            return [gen.config.name for gen in self.enhanced_generators[category]]
        else:
            return self.feature_bank.list_features(category)


# Convenience functions for easy integration

def create_enhanced_feature_engineer(config: Optional[Dict[str, Any]] = None) -> EnhancedFeatureEngineeringIntegration:
    """
    Create an enhanced feature engineering integration instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        EnhancedFeatureEngineeringIntegration instance
    """
    return EnhancedFeatureEngineeringIntegration(config)


async def generate_enhanced_features(
    data_dict: Dict[str, pd.DataFrame],
    config: Optional[Dict[str, Any]] = None,
    include_categories: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Generate enhanced features for multiple timeframes.
    
    Args:
        data_dict: Dictionary with timeframe -> DataFrame mapping
        config: Configuration dictionary
        include_categories: List of categories to include
        
    Returns:
        Dictionary with timeframe -> enhanced features DataFrame mapping
    """
    engineer = create_enhanced_feature_engineer(config)
    return await engineer.generate_comprehensive_features(
        data_dict, include_categories
    )


def generate_enhanced_features_sync(
    data_dict: Dict[str, pd.DataFrame],
    config: Optional[Dict[str, Any]] = None,
    include_categories: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Generate enhanced features synchronously.
    
    Args:
        data_dict: Dictionary with timeframe -> DataFrame mapping
        config: Configuration dictionary
        include_categories: List of categories to include
        
    Returns:
        Dictionary with timeframe -> enhanced features DataFrame mapping
    """
    return asyncio.run(generate_enhanced_features(data_dict, config, include_categories))


# Example usage and testing functions

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
    engineer = create_enhanced_feature_engineer({
        'enable_gpu': False,
        'max_workers': 2
    })

    # Generate comprehensive features
    features = generate_enhanced_features_sync(
        sample_data,
        include_categories=["normalization", "cross_timeframe", "interaction", "representation_learning"]
    )

    # Print summary
    summary = engineer.get_feature_summary()
    print(f"Generated features using {summary['total_enhanced_generators']} enhanced generators")
    print(f"Feature bank has {summary['feature_bank_generators']} generators")

    for timeframe, feature_df in features.items():
        print(f"{timeframe}: {len(feature_df.columns)} features, {len(feature_df)} samples")

    # Validate features
    validation = engineer.validate_features(features)
    print("\nValidation results:")
    for timeframe, results in validation.items():
        print(f"{timeframe}: {results['num_features']} features, {results['missing_values']} missing values")

    return features


if __name__ == "__main__":
    # Run example
    example_usage()