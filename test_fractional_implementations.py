# test_fractional_implementations.py

"""Test suite for fractional labeling and fractional differentiation implementations."""

import numpy as np
import pandas as pd
import pytest
from typing import Dict, Any

from src.training.steps.step4_analyst_labeling_feature_engineering_components.fractional_triple_barrier_labeling import (
    FractionalTripleBarrierLabeling
)
from src.training.steps.fractional_differentiation import (
    FractionalDifferentiation,
    FractionalFeatureGenerator
)


class TestFractionalTripleBarrierLabeling:
    """Test suite for fractional triple barrier labeling."""

    def setup_method(self):
        """Set up test data."""
        # Create synthetic OHLCV data
        np.random.seed(42)
        n_samples = 1000

        # Generate price data with some trend and volatility
        base_price = 100
        returns = np.random.normal(0.0001, 0.02, n_samples)  # Small positive drift
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Create OHLCV data
        self.test_data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_samples)
        })

        # Ensure high >= close >= low
        self.test_data['high'] = np.maximum(self.test_data['high'], self.test_data['close'])
        self.test_data['low'] = np.minimum(self.test_data['low'], self.test_data['close'])

        # Create regime labels (simplified)
        self.regime_labels = np.random.choice([0, 1, 2], size=n_samples)

        # Create volatility series
        self.volatility_series = pd.Series(
            np.random.uniform(0.01, 0.05, n_samples),
            index=self.test_data.index
        )

    def test_fractional_labeling_initialization(self):
        """Test fractional labeling initialization."""
        labeler = FractionalTripleBarrierLabeling()

        assert labeler.base_labeler is not None
        assert labeler.fractional_config is not None
        assert "enable_distance_scaling" in labeler.fractional_config
        assert "enable_time_decay" in labeler.fractional_config
        assert "enable_volatility_normalization" in labeler.fractional_config

    def test_fractional_labeling_basic(self):
        """Test basic fractional labeling functionality."""
        labeler = FractionalTripleBarrierLabeling()

        result = labeler.apply_fractional_triple_barrier_labeling(self.test_data)

        # Check that we get the expected columns
        expected_columns = [
            'fractional_label', 'confidence_score', 'barrier_distance',
            'time_decay_score', 'volatility_score'
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        # Check that fractional labels are in expected range
        assert result['fractional_label'].min() >= -1
        assert result['fractional_label'].max() <= 1

        # Check that confidence scores are in expected range
        assert result['confidence_score'].min() >= 0
        assert result['confidence_score'].max() <= 1

    def test_fractional_labeling_with_regime_labels(self):
        """Test fractional labeling with regime labels."""
        labeler = FractionalTripleBarrierLabeling()

        result = labeler.apply_fractional_triple_barrier_labeling(
            self.test_data,
            regime_labels=self.regime_labels
        )

        assert len(result) > 0
        assert 'fractional_label' in result.columns

    def test_fractional_labeling_with_volatility(self):
        """Test fractional labeling with volatility series."""
        labeler = FractionalTripleBarrierLabeling()

        result = labeler.apply_fractional_triple_barrier_labeling(
            self.test_data,
            volatility_series=self.volatility_series
        )

        assert len(result) > 0
        assert 'volatility_score' in result.columns

    def test_fractional_labeling_statistics(self):
        """Test fractional labeling statistics."""
        labeler = FractionalTripleBarrierLabeling()

        result = labeler.apply_fractional_triple_barrier_labeling(self.test_data)
        stats = labeler.get_fractional_label_statistics(result)

        expected_keys = [
            'total_samples', 'fractional_label_mean', 'fractional_label_std',
            'confidence_mean', 'confidence_std', 'positive_labels',
            'negative_labels', 'neutral_labels'
        ]

        for key in expected_keys:
            assert key in stats, f"Missing statistic: {key}"

        assert stats['total_samples'] > 0
        assert stats['fractional_label_mean'] >= -1 and stats['fractional_label_mean'] <= 1

    def test_fractional_labeling_configuration(self):
        """Test fractional labeling with custom configuration."""
        custom_config = {
            "enable_distance_scaling": True,
            "enable_time_decay": False,
            "enable_volatility_normalization": True,
            "distance_weight": 0.6,
            "time_weight": 0.0,
            "volatility_weight": 0.4,
            "min_confidence_threshold": 0.2,
            "max_confidence_threshold": 0.9,
        }

        labeler = FractionalTripleBarrierLabeling(fractional_config=custom_config)
        result = labeler.apply_fractional_triple_barrier_labeling(self.test_data)

        # Check that time decay scores are zero (disabled)
        assert result['time_decay_score'].sum() == 0

        # Check that confidence threshold was applied
        assert result['confidence_score'].min() >= 0.2


class TestFractionalDifferentiation:
    """Test suite for fractional differentiation."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 500

        # Create synthetic time series with trend and noise
        t = np.arange(n_samples)
        trend = 0.001 * t  # Linear trend
        noise = np.random.normal(0, 0.1, n_samples)
        self.test_series = pd.Series(trend + noise, name='test_series')

        # Create price-like data
        self.price_series = pd.Series(
            np.cumsum(np.random.normal(0.001, 0.02, n_samples)) + 100,
            name='price'
        )

    def test_fractional_differentiation_initialization(self):
        """Test fractional differentiation initialization."""
        frac_diff = FractionalDifferentiation(d=0.5, window=50)

        assert frac_diff.d == 0.5
        assert frac_diff.window == 50
        assert len(frac_diff.weights) == 50
        assert frac_diff.weights[0] == -0.5  # First weight should be -d

    def test_fractional_differentiation_basic(self):
        """Test basic fractional differentiation."""
        frac_diff = FractionalDifferentiation(d=0.5, window=20)

        result = frac_diff.fractional_diff(self.test_series)

        assert len(result) == len(self.test_series)
        assert not result.isnull().all()
        assert result.name == 'test_series_frac_diff_0.5'

    def test_fractional_differentiation_short_series(self):
        """Test fractional differentiation with short series."""
        short_series = self.test_series.iloc[:10]  # Less than window size
        frac_diff = FractionalDifferentiation(d=0.5, window=20)

        result = frac_diff.fractional_diff(short_series)

        # Should fall back to simple differentiation
        assert len(result) == len(short_series)
        assert result.name == 'test_series_frac_diff_0.5'

    def test_fractional_differentiation_order_optimization(self):
        """Test fractional order optimization."""
        frac_diff = FractionalDifferentiation(d=0.5, window=20, optimize_order=True)

        result, optimal_d = frac_diff.apply_with_optimization(self.price_series)

        assert optimal_d > 0 and optimal_d < 1
        assert len(result) == len(self.price_series)
        assert result.name == f'price_frac_diff_{optimal_d:.3f}'

    def test_batch_fractional_differentiation(self):
        """Test batch fractional differentiation."""
        # Create test DataFrame
        test_df = pd.DataFrame({
            'price1': self.price_series,
            'price2': self.price_series * 1.1,
            'volume': np.random.randint(1000, 10000, len(self.price_series))
        })

        frac_diff = FractionalDifferentiation(d=0.5, window=20, optimize_order=False)

        result_df, optimization_results = frac_diff.batch_fractional_diff(
            test_df,
            columns=['price1', 'price2']
        )

        # Check that new columns were added
        expected_columns = ['price1_frac_diff_0.500', 'price2_frac_diff_0.500']
        for col in expected_columns:
            assert col in result_df.columns

        # Check optimization results
        assert len(optimization_results) == 2
        assert 'price1' in optimization_results
        assert 'price2' in optimization_results


class TestFractionalFeatureGenerator:
    """Test suite for fractional feature generator."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 300

        # Create OHLCV test data
        self.test_data = pd.DataFrame({
            'open': np.cumsum(np.random.normal(0.001, 0.02, n_samples)) + 100,
            'high': np.cumsum(np.random.normal(0.001, 0.02, n_samples)) + 101,
            'low': np.cumsum(np.random.normal(0.001, 0.02, n_samples)) + 99,
            'close': np.cumsum(np.random.normal(0.001, 0.02, n_samples)) + 100,
            'volume': np.random.randint(1000, 10000, n_samples)
        })

    def test_fractional_feature_generator_initialization(self):
        """Test fractional feature generator initialization."""
        generator = FractionalFeatureGenerator()

        assert generator.config is not None
        assert generator.fractional_diff is not None
        assert generator.config["enable_fractional_diff"] is True

    def test_fractional_feature_generation(self):
        """Test fractional feature generation."""
        generator = FractionalFeatureGenerator()

        result = generator.generate_features(self.test_data)

        # Check that original columns are preserved
        original_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in original_columns:
            assert col in result.columns

        # Check that fractional differentiation features were added
        frac_diff_columns = [col for col in result.columns if 'frac_diff' in col]
        assert len(frac_diff_columns) > 0

    def test_fractional_feature_generator_disabled(self):
        """Test fractional feature generator when disabled."""
        config = {"enable_fractional_diff": False}
        generator = FractionalFeatureGenerator(config)

        result = generator.generate_features(self.test_data)

        # Should return original data unchanged
        assert len(result.columns) == len(self.test_data.columns)
        assert all(col in result.columns for col in self.test_data.columns)

    def test_fractional_feature_statistics(self):
        """Test fractional feature statistics."""
        generator = FractionalFeatureGenerator()

        result = generator.generate_features(self.test_data)
        stats = generator.get_feature_statistics(result)

        assert "total_frac_diff_features" in stats
        assert "frac_diff_columns" in stats
        assert "feature_statistics" in stats

        assert stats["total_frac_diff_features"] > 0
        assert len(stats["frac_diff_columns"]) > 0


def test_integration_fractional_labeling_and_differentiation():
    """Integration test for fractional labeling and differentiation."""
    # Create test data
    np.random.seed(42)
    n_samples = 200

    test_data = pd.DataFrame({
        'open': np.cumsum(np.random.normal(0.001, 0.02, n_samples)) + 100,
        'high': np.cumsum(np.random.normal(0.001, 0.02, n_samples)) + 101,
        'low': np.cumsum(np.random.normal(0.001, 0.02, n_samples)) + 99,
        'close': np.cumsum(np.random.normal(0.001, 0.02, n_samples)) + 100,
        'volume': np.random.randint(1000, 10000, n_samples)
    })

    # Step 1: Apply fractional differentiation
    feature_generator = FractionalFeatureGenerator()
    data_with_frac_features = feature_generator.generate_features(test_data)

    # Step 2: Apply fractional labeling
    labeler = FractionalTripleBarrierLabeling()
    labeled_data = labeler.apply_fractional_triple_barrier_labeling(data_with_frac_features)

    # Check that both fractional features and labels are present
    frac_diff_columns = [col for col in labeled_data.columns if 'frac_diff' in col]
    fractional_label_columns = [col for col in labeled_data.columns if 'fractional_label' in col]

    assert len(frac_diff_columns) > 0, "No fractional differentiation features found"
    assert len(fractional_label_columns) > 0, "No fractional labels found"

    # Check that we have both original and enhanced data
    assert 'close' in labeled_data.columns
    assert 'fractional_label' in labeled_data.columns
    assert 'confidence_score' in labeled_data.columns


if __name__ == "__main__":
    # Run tests
    print("🧪 Running fractional implementation tests...")

    # Test fractional labeling
    print("\n📊 Testing Fractional Triple Barrier Labeling...")
    test_labeling = TestFractionalTripleBarrierLabeling()
    test_labeling.setup_method()

    test_labeling.test_fractional_labeling_initialization()
    test_labeling.test_fractional_labeling_basic()
    test_labeling.test_fractional_labeling_with_regime_labels()
    test_labeling.test_fractional_labeling_with_volatility()
    test_labeling.test_fractional_labeling_statistics()
    test_labeling.test_fractional_labeling_configuration()

    print("✅ Fractional labeling tests passed!")

    # Test fractional differentiation
    print("\n📈 Testing Fractional Differentiation...")
    test_diff = TestFractionalDifferentiation()
    test_diff.setup_method()

    test_diff.test_fractional_differentiation_initialization()
    test_diff.test_fractional_differentiation_basic()
    test_diff.test_fractional_differentiation_short_series()
    test_diff.test_fractional_differentiation_order_optimization()
    test_diff.test_batch_fractional_differentiation()

    print("✅ Fractional differentiation tests passed!")

    # Test fractional feature generator
    print("\n🔧 Testing Fractional Feature Generator...")
    test_generator = TestFractionalFeatureGenerator()
    test_generator.setup_method()

    test_generator.test_fractional_feature_generator_initialization()
    test_generator.test_fractional_feature_generation()
    test_generator.test_fractional_feature_generator_disabled()
    test_generator.test_fractional_feature_statistics()

    print("✅ Fractional feature generator tests passed!")

    # Integration test
    print("\n🔗 Testing Integration...")
    test_integration_fractional_labeling_and_differentiation()

    print("✅ Integration test passed!")
    print("\n🎉 All fractional implementation tests completed successfully!")