from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd

"""Unit tests for Step 10: Unified Regime Intelligence."""
try:
    import pytest
except ImportError:
    pytest = None
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from src.training.steps.model_training.step10_unified_regime_intelligence import UnifiedRegimeIntelligenceStep, RegimeIntelligenceAnalyzer, RegimeMetricsCalculator, RegimeTransitionAnalyzer
from copy import copy
import asyncio

class TestRegimeIntelligenceAnalyzer:
    """Test cases for RegimeIntelligenceAnalyzer."""

    @pytest.fixture
    def analyzer(self) -> None:
        """Create analyzer instance."""
        config = {'timeframes': ['5m', '15m', '30m'], 'intensity_threshold': 0.7, 'transition_threshold': 0.8}
        return RegimeIntelligenceAnalyzer(config)

    @pytest.fixture
    def sample_data(self) -> None:
        """Create sample data for testing."""
        hmm_states = {'5m': np.random.randint(0, 5, size = 100), '15m': np.random.randint(0, 5, size = 100), '30m': np.random.randint(0, 5, size = 100)}
        dates = pd.date_range(start='2024-01-01', periods = 100, freq='5min')
        market_features = pd.DataFrame({'returns': np.random.randn(100), 'volume': np.random.rand(100) * 1000000, 'volatility': np.random.rand(100) * 0.02}, index = dates)
        return (hmm_states, market_features)

    def test_analyze_regime_states(self, analyzer: Any, sample_data: Any) -> None:
        """Test regime state analysis."""
        hmm_states, market_features = sample_data
        results = analyzer.analyze_regime_states(hmm_states, market_features)
        assert 'regime_states' in results
        assert 'intensity_scores' in results
        assert 'transition_probabilities' in results
        assert 'alignment_scores' in results
        for tf in analyzer.timeframes:
            assert tf in results['regime_states']
            assert tf in results['intensity_scores']
            assert tf in results['transition_probabilities']

    def test_intensity_calculation(self, analyzer: Any, sample_data: Any) -> None:
        """Test intensity score calculation."""
        hmm_states, market_features = sample_data
        intensity = analyzer._calculate_intensity_scores(hmm_states['5m'], market_features)
        assert isinstance(intensity, np.ndarray)
        assert len(intensity) == len(hmm_states['5m'])
        assert np.all((intensity >= 0) & (intensity <= 1))

    def test_empty_hmm_states(self, analyzer: Any) -> None:
        """Test handling of empty HMM states."""
        empty_states = {}
        market_features = pd.DataFrame()
        results = analyzer.analyze_regime_states(empty_states, market_features)
        assert results['regime_states'] == {}
        assert results['intensity_scores'] == {}

class TestRegimeMetricsCalculator:
    """Test cases for RegimeMetricsCalculator."""

    @pytest.fixture
    def calculator(self) -> None:
        """Create calculator instance."""
        config = {'num_regimes': 5}
        return RegimeMetricsCalculator(config)

    @pytest.fixture
    def sample_regime_data(self) -> None:
        """Create sample regime analysis data."""
        return {'regime_states': {'15m': np.random.randint(0, 5, size = 100)}, 'intensity_scores': {'15m': np.random.rand(100)}}

    @pytest.fixture
    def sample_price_data(self) -> None:
        """Create sample price data."""
        dates = pd.date_range(start='2024-01-01', periods = 100, freq='5min')
        return pd.DataFrame({'close': 100 + np.random.randn(100).cumsum(), 'high': 101 + np.random.randn(100).cumsum(), 'low': 99 + np.random.randn(100).cumsum(), 'volume': np.random.rand(100) * 1000000}, index = dates)

    def test_calculate_regime_metrics(self, calculator: Any, sample_regime_data: Any, sample_price_data: Any) -> None:
        """Test regime metrics calculation."""
        metrics = calculator.calculate_regime_metrics(sample_regime_data, sample_price_data)
        assert 'per_regime_metrics' in metrics
        assert 'transition_metrics' in metrics
        assert 'overall_metrics' in metrics
        for i in range(5):
            regime_key = f'regime_{i}'
            assert regime_key in metrics['per_regime_metrics']
            regime_metrics = metrics['per_regime_metrics'][regime_key]
            assert 'return' in regime_metrics
            assert 'volatility' in regime_metrics
            assert 'sharpe_ratio' in regime_metrics

    def test_transition_metrics(self, calculator: Any, sample_regime_data: Any, sample_price_data: Any) -> None:
        """Test transition metrics calculation."""
        metrics = calculator.calculate_regime_metrics(sample_regime_data, sample_price_data)
        transition_metrics = metrics['transition_metrics']
        assert 'avg_transition_cost' in transition_metrics
        assert 'false_transition_rate' in transition_metrics
        assert 'transition_timing_accuracy' in transition_metrics

class TestRegimeTransitionAnalyzer:
    """Test cases for RegimeTransitionAnalyzer."""

    @pytest.fixture
    def analyzer(self) -> None:
        """Create analyzer instance."""
        config = {'num_regimes': 5}
        return RegimeTransitionAnalyzer(config)

    @pytest.fixture
    def sample_sequence(self) -> None:
        """Create sample regime sequence."""
        sequence = []
        for _ in range(20):
            regime = np.random.randint(0, 5)
            duration = np.random.randint(5, 15)
            sequence.extend([regime] * duration)
        return np.array(sequence)

    def test_analyze_transitions(self, analyzer: Any, sample_sequence: Any) -> None:
        """Test transition analysis."""
        features = pd.DataFrame({'feature1': np.random.randn(len(sample_sequence))})
        results = analyzer.analyze_transitions(sample_sequence, features)
        assert 'transition_matrix' in results
        assert 'current_regime' in results
        assert 'next_regime_probabilities' in results
        assert 'transition_indicators' in results
        assert 'stability_score' in results
        assert results['transition_matrix'] is not None
        assert results['transition_matrix'].shape == (5, 5)
        row_sums = results['transition_matrix'].sum(axis = 1)
        np.testing.assert_allclose(row_sums[row_sums > 0], 1.0, rtol = 1e-06)

    def test_build_transition_matrix(self, analyzer: Any) -> None:
        """Test transition matrix construction."""
        sequence = np.array([0, 1, 1, 2, 0, 1, 2, 2, 0])
        matrix = analyzer._build_transition_matrix(sequence)
        assert matrix.shape == (5, 5)
        assert np.isclose(matrix[0, 1], 2 / 3)
        assert np.isclose(matrix[1, 1], 1 / 3)

    def test_stability_score(self, analyzer: Any) -> None:
        """Test stability score calculation."""
        stable_sequence = np.array([2] * 50)
        score = analyzer._calculate_stability_score(stable_sequence, 20)
        assert score == 1.0
        unstable_sequence = np.array([i % 5 for i in range(50)])
        score = analyzer._calculate_stability_score(unstable_sequence, 20)
        assert 0 <= score < 1.0

class TestUnifiedRegimeIntelligenceStep:
    """Test cases for UnifiedRegimeIntelligenceStep."""

    @pytest.fixture
    def step(self) -> None:
        """Create step instance."""
        config = {'model': {'sequence_length': 20, 'batch_size': 32, 'learning_rate': 0.0001, 'epochs': 10}, 'artifacts_dir': 'test_artifacts'}
        return UnifiedRegimeIntelligenceStep(config)

    @pytest.fixture
    def valid_pipeline_state(self) -> None:
        """Create valid pipeline state."""
        dates = pd.date_range(start='2024-01-01', periods = 100, freq='5min')
        return {'hmm_states': {'5m': np.random.randint(0, 5, size = 100), '15m': np.random.randint(0, 5, size = 100)}, 'market_features': pd.DataFrame({'returns': np.random.randn(100), 'volume': np.random.rand(100) * 1000000}, index = dates), 'price_data': pd.DataFrame({'close': 100 + np.random.randn(100).cumsum()}, index = dates), 'regime_labels': np.random.randint(0, 5, size = 100)}

    def test_initialization(self, step: Any) -> None:
        """Test step initialization."""
        assert step.step_number == '10'
        assert step.step_name == 'unified_regime_intelligence'
        assert step.regime_analyzer is not None
        assert step.metrics_calculator is not None
        assert step.transition_analyzer is not None

    def test_get_required_inputs(self, step: Any) -> None:
        """Test required inputs."""
        inputs = step.get_required_inputs()
        assert 'hmm_states' in inputs
        assert 'market_features' in inputs
        assert 'price_data' in inputs
        assert 'regime_labels' in inputs

    def test_get_produced_outputs(self, step: Any) -> None:
        """Test produced outputs."""
        outputs = step.get_produced_outputs()
        assert 'regime_model' in outputs
        assert 'regime_analysis' in outputs
        assert 'regime_metrics' in outputs
        assert 'transition_analysis' in outputs
        assert 'regime_predictions' in outputs

    def test_validate_inputs_valid(self, step: Any, valid_pipeline_state: Any) -> None:
        """Test input validation with valid inputs."""
        is_valid, errors = step.validate_inputs({}, valid_pipeline_state)
        assert is_valid
        assert len(errors) == 0

    def test_validate_inputs_missing_data(self, step: Any) -> None:
        """Test input validation with missing data."""
        incomplete_state = {'hmm_states': {}}
        is_valid, errors = step.validate_inputs({}, incomplete_state)
        assert not is_valid
        assert any(('market_features' in error for error in errors))

    def test_validate_inputs_wrong_type(self, step: Any, valid_pipeline_state: Any) -> None:
        """Test input validation with wrong data types."""
        invalid_state = valid_pipeline_state.copy()
        invalid_state['hmm_states'] = 'not a dict'
        is_valid, errors = step.validate_inputs({}, invalid_state)
        assert not is_valid
        assert any(('dictionary' in error for error in errors))

    @pytest.mark.asyncio
    async def test_execute_logic(self, step: Any, valid_pipeline_state: Any) -> None:
        """Test execution logic."""
        training_input = {'train_model': False}
        with patch.object(step, '_train_regime_model', new_callable = AsyncMock) as mock_train:
            mock_train.return_value = (Mock(), {'loss': [0.5]})
            result = await step.execute_logic(training_input, valid_pipeline_state)
        assert 'regime_analysis' in result
        assert 'regime_metrics' in result
        assert 'transition_analysis' in result
        assert 'num_regimes' in result

    def test_validate_outputs_valid(self, step: Any) -> None:
        """Test output validation with valid outputs."""
        valid_outputs = {'regime_analysis': {'regime_states': {}, 'intensity_scores': {}, 'transition_probabilities': {}}, 'regime_metrics': {'per_regime_metrics': {}, 'transition_metrics': {}, 'overall_metrics': {}}, 'transition_analysis': {'transition_matrix': np.array([[0.5, 0.5], [0.3, 0.7]])}}
        is_valid, errors = step.validate_outputs(valid_outputs)
        assert is_valid
        assert len(errors) == 0

    def test_validate_outputs_missing(self, step: Any) -> None:
        """Test output validation with missing outputs."""
        incomplete_outputs = {'regime_analysis': {}}
        is_valid, errors = step.validate_outputs(incomplete_outputs)
        assert not is_valid
        assert any(('regime_metrics' in error for error in errors))

    def test_make_json_serializable(self, step: Any) -> None:
        """Test JSON serialization helper."""
        data = {'array': np.array([1, 2, 3]), 'float32': np.float32(1.5), 'nested': {'array': np.array([[1, 2], [3, 4]])}}
        serializable = step._make_json_serializable(data)
        assert serializable['array'] == [1, 2, 3]
        assert isinstance(serializable['float32'], float)
        assert serializable['nested']['array'] == [[1, 2], [3, 4]]
if __name__ == '__main__':
    pytest.main([__file__, '-v'])