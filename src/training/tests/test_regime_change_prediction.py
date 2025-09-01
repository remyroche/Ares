# src/training/tests/test_regime_change_prediction.py

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.training.steps.step05_hmm_based_training import HMMBasedTrainingStep
from src.training.steps.step09_5_hmm_lm_generalist_training import (
    HMMLMGeneralistTrainingStep,
)


class TestRegimeChangePrediction:
    """Test suite for regime change prediction functionality."""

    @pytest.fixture
    def sample_hmm_data(self):
        """Create sample HMM data for testing."""
        # Create sample data with regime changes
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="1H")

        # Create regime changes every few hours
        regimes = []
        for i in range(len(dates)):
            regime = (i // 6) % 5  # Change regime every 6 hours, 5 different regimes
            regimes.append(regime)

        # Create intensity scores
        intensity_data = {}
        for i in range(20):  # 20 intensity features
            intensity_data[f"intensity_cluster_{i}"] = np.random.rand(len(dates))

        # Create regime probability features
        regime_data = {}
        for i in range(5):  # 5 regime states
            regime_data[f"momentum_p_state_{i}"] = np.random.rand(len(dates))
            regime_data[f"volatility_p_state_{i}"] = np.random.rand(len(dates))

        return pd.DataFrame(
            {
                "timestamp": dates,
                "composite_cluster_id": regimes,
                **intensity_data,
                **regime_data,
            },
        )


    @pytest.fixture
    def sample_feature_data(self):
        """Create sample feature data for testing."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="1H")

        # Create sample features
        features = {}
        for i in range(50):  # 50 features
            features[f"feature_{i}"] = np.random.rand(len(dates))

        return pd.DataFrame({"timestamp": dates, **features})


    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_regime_change_detection(self, sample_hmm_data) -> None:
        """Test regime change detection functionality."""
        # Create step instance
        config = {
            "HMM_LM": {
                "generalist": {
                    "hmm_states": 5,
                    "sequence_length": 20,
                    "timeframes": ["1m", "5m", "15m", "30m"],
                },
            },
        }

        step = HMMLMGeneralistTrainingStep(config)

        # Test regime change detection
        regime_changes = step._detect_regime_changes(sample_hmm_data)

        # Should detect regime changes
        assert len(regime_changes) > 0

        # Should have regime change events
        change_events = [event for event in regime_changes if event != "<PAD>"]
        assert len(change_events) > 0

        # Should have enter/exit events
        enter_events = [event for event in change_events if "enter_regime" in event]
        exit_events = [event for event in change_events if "exit_regime" in event]
        assert len(enter_events) > 0
        assert len(exit_events) > 0

    def test_vocabulary_creation(self) -> None:
        """Test regime change vocabulary creation."""
        config = {
            "HMM_LM": {
                "generalist": {
                    "hmm_states": 3,
                    "sequence_length": 20,
                    "timeframes": ["1m", "5m"],
                },
            },
        }

        step = HMMLMGeneralistTrainingStep(config)

        # Check vocabulary size
        expected_size = 3 * 2 + 4  # 3 states * 2 (enter/exit) + 4 special tokens
        assert len(step.regime_change_vocab) == expected_size

        # Check vocabulary content
        assert "enter_regime_0" in step.regime_change_vocab
        assert "exit_regime_0" in step.regime_change_vocab
        assert "enter_regime_1" in step.regime_change_vocab
        assert "exit_regime_1" in step.regime_change_vocab
        assert "enter_regime_2" in step.regime_change_vocab
        assert "exit_regime_2" in step.regime_change_vocab
        assert "<PAD>" in step.regime_change_vocab
        assert "<UNK>" in step.regime_change_vocab
        assert "<START>" in step.regime_change_vocab
        assert "<END>" in step.regime_change_vocab

    def test_sequence_creation(self, sample_hmm_data) -> None:
        """Test sequence creation for training."""
        config = {
            "HMM_LM": {
                "generalist": {
                    "hmm_states": 5,
                    "sequence_length": 10,
                    "timeframes": ["1m"],
                },
            },
        }

        step = HMMLMGeneralistTrainingStep(config)

        # Create sequences
        sequences = step._create_regime_change_sequences({"1m": sample_hmm_data})

        # Should create sequences
        assert len(sequences) > 0

        # Check sequence structure
        for seq in sequences:
            assert "sequence" in seq
            assert "target" in seq
            assert "timestamp" in seq
            assert "timeframe" in seq

            # Check sequence data
            assert len(seq["sequence"]) == 10  # sequence_length
            assert seq["target"] in step.regime_change_vocab

    def test_feature_preparation(self, sample_hmm_data) -> None:
        """Test feature preparation for language model."""
        config = {
            "HMM_LM": {
                "generalist": {
                    "hmm_states": 5,
                    "sequence_length": 10,
                    "timeframes": ["1m"],
                },
            },
        }

        step = HMMLMGeneralistTrainingStep(config)

        # Test feature conversion
        features = step._sequence_to_features(sample_hmm_data)

        # Should return numpy array
        assert isinstance(features, np.ndarray)

        # Should have correct shape
        assert features.shape[0] == 10  # sequence_length
        assert features.shape[1] > 0  # number of features

    @pytest.mark.asyncio
    async def test_hmm_based_training_step(
        self, sample_hmm_data, sample_feature_data, temp_data_dir,
    ) -> None:
        """Test HMM-based training step with regime change features."""
        config = {
            "HMM_LM": {
                "specialist_models": {
                    "1m": {"architecture": "CNN"},
                    "5m": {"architecture": "TCN"},
                    "15m": {"architecture": "Transformer"},
                    "30m": {"architecture": "LightGBM"},
                },
            },
        }

        step = HMMBasedTrainingStep(config)
        await step.initialize()

        # Save sample data
        sample_hmm_data.to_parquet(os.path.join(temp_data_dir, "test_hmm_data.parquet"))
        sample_feature_data.to_parquet(
            os.path.join(temp_data_dir, "test_feature_data.parquet"),
        )

        # Test regime change feature addition
        enhanced_data = await step._add_regime_change_features(sample_hmm_data, "1m")

        # Should have regime change features
        assert "regime_change" in enhanced_data.columns
        assert "regime_change_next" in enhanced_data.columns
        assert "regime_change_prev" in enhanced_data.columns
        assert "regime_stability" in enhanced_data.columns
        assert "regime_volatility" in enhanced_data.columns

    def test_config_integration(self) -> None:
        """Test configuration integration."""
        from src.config import get_complete_config

        config = get_complete_config()

        # Check if HMM_LM config is present
        assert "HMM_LM" in config

        hmm_lm_config = config["HMM_LM"]

        # Check generalist config
        assert "generalist" in hmm_lm_config
        generalist = hmm_lm_config["generalist"]
        assert "enabled" in generalist
        assert "hmm_states" in generalist
        assert "sequence_length" in generalist
        assert "timeframes" in generalist

        # Check specialist models config
        assert "specialist_models" in hmm_lm_config
        specialist = hmm_lm_config["specialist_models"]
        assert "1m" in specialist
        assert "5m" in specialist
        assert "15m" in specialist
        assert "30m" in specialist

        # Check model architectures
        assert specialist["1m"]["architecture"] == "CNN"
        assert specialist["5m"]["architecture"] == "TCN"
        assert specialist["15m"]["architecture"] == "Transformer"
        assert specialist["30m"]["architecture"] == "LightGBM"

    def test_step_order_integration(self) -> None:
        """Test step order integration in training pipeline."""
        from src.config import get_complete_config

        config = get_complete_config()
        hmm_lm_config = config["HMM_LM"]

        # Check training pipeline config
        assert "training_pipeline" in hmm_lm_config
        pipeline_config = hmm_lm_config["training_pipeline"]

        # Check step order
        assert "step_order" in pipeline_config
        step_order = pipeline_config["step_order"]

        # Should include new step 9.5
        assert "9.5" in step_order

        # Check step order is correct
        expected_order = ["5", "9", "9.5", "6", "10"]
        assert step_order == expected_order


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
