#!/usr/bin/env python3
"""
Test script for enhanced feature engineering capabilities.
"""

import sys
import os

# Add the workspace to the path
sys.path.insert(0, '/workspace')

def test_imports():
    """Test that all new feature generators can be imported."""
    print("Testing imports...")

    try:
        # Test normalization features
        from src.feature_generation.categories.normalization import (
            NormalizationFeatureGenerator,
            RollingZScoreGenerator,
            VolatilityScalingGenerator,
            CrossSectionalNormalizer
        )
        print("✅ Normalization features imported successfully")

        # Test cross-timeframe features
        from src.feature_generation.categories.cross_timeframe import (
            CrossTimeframeFractionalChangeGenerator,
            CrossTimeframeAlignmentGenerator,
            CrossTimeframeLearnedProjectionGenerator
        )
        print("✅ Enhanced cross-timeframe features imported successfully")

        # Test interaction features
        from src.feature_generation.categories.interaction import (
            RegimeDependentFeatureGenerator,
            CointegrationResidualGenerator,
            StructuralRatioGenerator,
            PairwiseInteractionGenerator
        )
        print("✅ Enhanced interaction features imported successfully")

        # Test representation learning features
        from src.feature_generation.categories.representation_learning import (
            PatchTSTRepresentationGenerator,
            TFTEncoderRepresentationGenerator,
            AutoencoderRepresentationGenerator,
            ContrastiveLearningGenerator
        )
        print("✅ Representation learning features imported successfully")

        # Test integration module
        from src.feature_generation.enhanced_feature_engineering_integration import EnhancedFeatureEngineer
        print("✅ Enhanced Feature Engineer imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_initialization():
    """Test that feature generators can be initialized."""
    print("\nTesting initialization...")

    try:
        from src.feature_generation.enhanced_feature_engineering_integration import EnhancedFeatureEngineer

        engineer = EnhancedFeatureEngineer()
        summary = engineer.get_feature_summary()

        print(f"✅ Enhanced Feature Engineer initialized")
        print(f"📊 Total generators: {summary['total_generators']}")
        print(f"🔧 Normalization: {summary['normalization_generators']}")
        print(f"⏰ Cross-timeframe: {summary['cross_timeframe_generators']}")
        print(f"🔗 Interaction: {summary['interaction_generators']}")
        print(f"🧠 Representation: {summary['representation_generators']}")

        return True

    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False

def test_feature_generation():
    """Test feature generation with sample data."""
    print("\nTesting feature generation...")

    try:
        import pandas as pd
        import numpy as np
        from src.feature_generation.enhanced_feature_engineering_integration import EnhancedFeatureEngineer

        # Create sample data
        sample_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'open': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })

        engineer = EnhancedFeatureEngineer()

        # Test normalization features
        norm_features = engineer._generate_normalization_features(sample_data)
        print(f"✅ Generated {len(norm_features)} normalization features")

        # Test cross-timeframe features
        ctf_features = engineer._generate_cross_timeframe_features(sample_data)
        print(f"✅ Generated {len(ctf_features)} cross-timeframe features")

        # Test interaction features
        interaction_features = engineer._generate_interaction_features(sample_data)
        print(f"✅ Generated {len(interaction_features)} interaction features")

        # Test representation features
        repr_features = engineer._generate_representation_features(sample_data)
        print(f"✅ Generated {len(repr_features)} representation features")

        return True

    except Exception as e:
        print(f"❌ Feature generation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Feature Engineering System")
    print("=" * 50)

    # Test imports
    if not test_imports():
        print("❌ Import tests failed")
        return 1

    # Test initialization
    if not test_initialization():
        print("❌ Initialization tests failed")
        return 1

    # Test feature generation
    if not test_feature_generation():
        print("❌ Feature generation tests failed")
        return 1

    print("\n🎉 All tests passed! Enhanced feature engineering system is working correctly.")
    return 0

if __name__ == "__main__":
    exit(main())