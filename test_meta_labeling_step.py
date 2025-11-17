"""
Test suite for Feature Generation Meta-Labeling Step.

This module tests the meta-labeling functionality including:
- Purging training indices to avoid lookahead bias
- Generating primary signals from technical indicators
- Creating meta-labels from signals
- Training meta-models with proper time-series CV
- Translating meta-labels to targets for downstream steps
"""

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    print("Warning: pytest not available, skipping pytest-based tests")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the meta-labeling components
from src.training.steps.pre_training.feature_generation_meta_labeling_step import (
    purge_training_idxs,
    compute_rsi,
    generate_primary_signals,
    create_meta_labels,
    create_meta_features,
    translate_metalabels_to_targets,
    FeatureGenerationMetaLabelingStep
)


class TestPurgeTrainingIdxs:
    """Test the purging functionality to avoid lookahead bias."""

    def test_purge_basic(self):
        """Test basic purging of overlapping indices."""
        train_idxs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        val_start_idx = 10
        val_end_idx = 15
        horizon = 3

        purged = purge_training_idxs(train_idxs, val_start_idx, val_end_idx, horizon)

        # Indices 7, 8, 9 should be removed because their horizon reaches validation
        # 7 + 3 = 10 (touches validation start)
        # 8 + 3 = 11 (in validation)
        # 9 + 3 = 12 (in validation)
        assert 7 not in purged
        assert 8 not in purged
        assert 9 not in purged
        assert all(i in purged for i in [0, 1, 2, 3, 4, 5, 6])

    def test_purge_no_overlap(self):
        """Test when there's no overlap."""
        train_idxs = np.array([0, 1, 2, 3, 4])
        val_start_idx = 20
        val_end_idx = 25
        horizon = 5

        purged = purge_training_idxs(train_idxs, val_start_idx, val_end_idx, horizon)

        # No indices should be removed
        assert len(purged) == len(train_idxs)
        assert np.array_equal(purged, train_idxs)

    def test_purge_all_overlap(self):
        """Test when all indices overlap."""
        train_idxs = np.array([8, 9, 10, 11, 12])
        val_start_idx = 10
        val_end_idx = 15
        horizon = 3

        purged = purge_training_idxs(train_idxs, val_start_idx, val_end_idx, horizon)

        # All indices should be removed
        assert len(purged) == 0


class TestComputeRSI:
    """Test RSI calculation."""

    def test_rsi_basic(self):
        """Test basic RSI calculation."""
        # Create synthetic price data with clear trend
        prices = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118,
                           120, 122, 124, 126, 128, 130, 132, 134, 136, 138])

        rsi = compute_rsi(prices, period=14)

        # Strong uptrend should have high RSI
        assert rsi.iloc[-1] > 70, "Strong uptrend should have RSI > 70"

    def test_rsi_downtrend(self):
        """Test RSI in downtrend."""
        prices = pd.Series([140, 138, 136, 134, 132, 130, 128, 126, 124, 122,
                           120, 118, 116, 114, 112, 110, 108, 106, 104, 102])

        rsi = compute_rsi(prices, period=14)

        # Strong downtrend should have low RSI
        assert rsi.iloc[-1] < 30, "Strong downtrend should have RSI < 30"

    def test_rsi_range(self):
        """Test that RSI stays in 0-100 range."""
        # Random prices
        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(100).cumsum())

        rsi = compute_rsi(prices, period=14)

        valid_rsi = rsi[~rsi.isna()]
        assert valid_rsi.min() >= 0, "RSI should be >= 0"
        assert valid_rsi.max() <= 100, "RSI should be <= 100"


class TestGeneratePrimarySignals:
    """Test primary signal generation."""

    def test_signals_basic(self):
        """Test basic signal generation."""
        # Create synthetic market data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
        df = pd.DataFrame({
            'close': 100 + np.random.randn(100).cumsum(),
            'high': 0,
            'low': 0,
            'volume': 1000
        }, index=dates)

        df['high'] = df['close'] * 1.01
        df['low'] = df['close'] * 0.99

        signals = generate_primary_signals(df)

        # Check that signals are generated
        assert 'rsi' in signals.columns
        assert 'ma' in signals.columns
        assert 'mom' in signals.columns
        assert 'consensus' in signals.columns

        # Check that signals are in valid range
        assert signals['rsi'].isin([-1, 0, 1]).all()
        assert signals['ma'].isin([-1, 0, 1]).all()
        assert signals['mom'].isin([-1, 0, 1]).all()

    def test_signals_uptrend(self):
        """Test signals in clear uptrend."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
        # Strong uptrend
        close_prices = np.linspace(100, 150, 100)
        df = pd.DataFrame({
            'close': close_prices,
            'high': close_prices * 1.01,
            'low': close_prices * 0.99,
            'volume': 1000
        }, index=dates)

        signals = generate_primary_signals(df)

        # In uptrend, MA signal should eventually be positive
        assert signals['ma'].iloc[-10:].mean() > 0, "Uptrend should have positive MA signals"


class TestCreateMetaLabels:
    """Test meta-label creation."""

    def test_metalabels_basic(self):
        """Test basic meta-label creation."""
        # Create synthetic data with clear profit opportunities
        dates = pd.date_range(start='2023-01-01', periods=50, freq='15min')
        df = pd.DataFrame({
            'close': [100] * 10 + [110] * 10 + [100] * 10 + [90] * 10 + [100] * 10
        }, index=dates)

        # Create signals
        signals = pd.DataFrame({
            'consensus': [1] + [0] * 49  # Only one signal at start
        }, index=dates)

        # Price goes from 100 to 110, so should be profitable
        meta_labels = create_meta_labels(
            df,
            signals,
            profit_threshold=0.08,  # 8% profit target
            stop_threshold=0.05,    # 5% stop loss
            horizon=20
        )

        # First bar should have a positive meta-label
        assert meta_labels.iloc[0] == 1.0, "Profitable signal should have meta-label = 1"

    def test_metalabels_stop_loss(self):
        """Test meta-labels with stop loss."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='15min')
        df = pd.DataFrame({
            'close': [100] * 10 + [90] * 40  # Price drops
        }, index=dates)

        signals = pd.DataFrame({
            'consensus': [1] + [0] * 49  # Long signal at start
        }, index=dates)

        # Price drops, should hit stop loss
        meta_labels = create_meta_labels(
            df,
            signals,
            profit_threshold=0.10,  # 10% profit (won't hit)
            stop_threshold=0.05,    # 5% stop (will hit)
            horizon=20
        )

        # First bar should have negative meta-label (stopped out)
        assert meta_labels.iloc[0] == 0.0, "Stopped signal should have meta-label = 0"

    def test_metalabels_no_signal(self):
        """Test that no label is created when there's no signal."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='15min')
        df = pd.DataFrame({
            'close': 100 + np.random.randn(50)
        }, index=dates)

        signals = pd.DataFrame({
            'consensus': [0] * 50  # No signals
        }, index=dates)

        meta_labels = create_meta_labels(df, signals)

        # All labels should be NaN
        assert meta_labels.isna().all(), "No signals should result in all NaN labels"


class TestCreateMetaFeatures:
    """Test meta-feature creation."""

    def test_features_basic(self):
        """Test basic feature creation."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
        df = pd.DataFrame({
            'close': 100 + np.random.randn(100).cumsum(),
            'high': 0,
            'low': 0,
            'volume': 1000 + np.random.randn(100) * 100
        }, index=dates)

        df['high'] = df['close'] * 1.01
        df['low'] = df['close'] * 0.99

        signals = generate_primary_signals(df)
        features = create_meta_features(df, signals, volume_available=True)

        # Check that features are created
        assert 'signal_strength' in features.columns
        assert 'volatility_5' in features.columns
        assert 'volume_ratio' in features.columns

        # Check that features are numeric
        assert features['signal_strength'].dtype in [np.float64, np.float32, np.int64]

    def test_features_without_volume(self):
        """Test feature creation without volume."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
        df = pd.DataFrame({
            'close': 100 + np.random.randn(100).cumsum(),
            'high': 0,
            'low': 0
        }, index=dates)

        df['high'] = df['close'] * 1.01
        df['low'] = df['close'] * 0.99

        signals = generate_primary_signals(df)
        features = create_meta_features(df, signals, volume_available=False)

        # Volume features should be constant
        assert (features['volume_ratio'] == 1.0).all()


class TestTranslateMetaLabelsToTargets:
    """Test translation from meta-labels to targets."""

    def test_translation_basic(self):
        """Test basic translation."""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='15min')
        meta_labels = pd.Series([1.0, 0.0, 1.0, np.nan, 1.0, 0.0, np.nan, 1.0, 0.0, 1.0], index=dates)
        signals = pd.DataFrame({
            'consensus': [1, 1, -1, 0, 1, -1, 0, -1, 1, 1]
        }, index=dates)
        probabilities = np.array([0.9, 0.4, 0.8, 0.5, 0.7, 0.3, 0.5, 0.85, 0.45, 0.95])

        target_long, target_short = translate_metalabels_to_targets(
            meta_labels,
            signals,
            probabilities,
            threshold=0.6
        )

        # Check that targets are created
        assert len(target_long) == len(meta_labels)
        assert len(target_short) == len(meta_labels)

        # High probability long signal should create positive long target
        assert target_long.iloc[0] > 0, "High prob long signal should have positive long target"

        # Low probability should not create target
        assert target_long.iloc[1] == 0, "Low probability should not create target"

        # High probability short signal should create positive short target
        assert target_short.iloc[2] > 0, "High prob short signal should have positive short target"


class TestFeatureGenerationMetaLabelingStep:
    """Test the full meta-labeling step."""

    @pytest.mark.asyncio
    async def test_step_basic(self):
        """Test basic step execution."""
        step = FeatureGenerationMetaLabelingStep()

        # This test would require actual market data
        # For now, just test that the step initializes
        assert step.step_name == "feature_generation_meta_labeling_step"
        assert step.logger is not None

    def test_step_config_validation(self):
        """Test config validation."""
        step = FeatureGenerationMetaLabelingStep()

        # Test with missing config
        import asyncio
        result = asyncio.run(step.execute({}))

        assert result['success'] is False
        assert 'error' in result
        assert 'Missing required config keys' in result['error']


def test_integration_full_pipeline():
    """
    Integration test for the full meta-labeling pipeline.

    This test creates synthetic data and runs through the entire process:
    1. Generate primary signals
    2. Create meta-labels
    3. Create meta-features
    4. Verify outputs
    """
    # Create synthetic market data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=500, freq='15min')

    # Create price data with some trends
    close_prices = 100.0
    close_data = []
    for i in range(500):
        # Add trend and noise
        trend = 0.001 if i % 100 < 50 else -0.001
        close_prices *= (1 + trend + np.random.randn() * 0.005)
        close_data.append(close_prices)

    df = pd.DataFrame({
        'close': close_data,
        'high': [p * 1.01 for p in close_data],
        'low': [p * 0.99 for p in close_data],
        'volume': 1000 + np.random.randn(500) * 100
    }, index=dates)

    # Generate primary signals
    print("Generating primary signals...")
    signals = generate_primary_signals(df)

    # Create meta-labels
    print("Creating meta-labels...")
    meta_labels = create_meta_labels(
        df,
        signals,
        profit_threshold=0.02,
        stop_threshold=0.01,
        horizon=16
    )

    # Create meta-features
    print("Creating meta-features...")
    features = create_meta_features(df, signals, volume_available=True)

    # Verify outputs
    print(f"Signals shape: {signals.shape}")
    print(f"Meta-labels: {(~meta_labels.isna()).sum()} labeled out of {len(meta_labels)}")
    print(f"Positive rate: {(meta_labels == 1.0).sum() / (~meta_labels.isna()).sum():.2%}")
    print(f"Features shape: {features.shape}")

    # Basic assertions
    assert signals.shape[0] == df.shape[0], "Signals should match data length"
    assert (~meta_labels.isna()).sum() > 0, "Should have some meta-labels"
    assert features.shape[0] == df.shape[0], "Features should match data length"
    assert features.shape[1] > 5, "Should have multiple features"

    print("\n✅ Integration test passed!")


if __name__ == "__main__":
    # Run integration test
    print("Running integration test...")
    test_integration_full_pipeline()

    print("\n" + "="*80)
    print("All tests would be run with: pytest test_meta_labeling_step.py")
    print("="*80)
