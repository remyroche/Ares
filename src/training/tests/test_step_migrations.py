"""Unit tests for migrated steps."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import asyncio
from typing import Dict, List, Optional, Union, Any, Tuple

class TestStep02DataReading:
    """Test Step 2: Data Reading."""

    @pytest.mark.asyncio
    async def test_data_reading_with_valid_data(self, sample_config: Any, sample_training_input: Any, sample_market_data: Any, test_data_dir: Any) -> None:
        """Test data reading with valid input data."""
        from src.training.steps.data_preparation.step02_data_reading import DataReadingStep
        data_path = test_data_dir / 'data' / 'sample_data.parquet'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_market_data.to_parquet(data_path)
        step = DataReadingStep(sample_config)
        pipeline_state = {'raw_market_data': str(data_path)}
        result = await step.execute(sample_training_input, pipeline_state)
        assert result['success'] is True
        assert 'validated_data' in result
        assert 'data_validation_results' in result
        assert isinstance(result['validated_data'], pd.DataFrame)
        validation = result['data_validation_results']
        assert validation['has_required_columns'] is True
        assert validation['missing_data_pct'] == 0
        assert validation['data_quality_score'] > 80

    @pytest.mark.asyncio
    async def test_data_reading_with_missing_file(self, sample_config: Any, sample_training_input: Any) -> None:
        """Test data reading with missing file."""
        from src.training.steps.data_preparation.step02_data_reading import DataReadingStep
        step = DataReadingStep(sample_config)
        pipeline_state = {'raw_market_data': '/path/to/missing/file.parquet'}
        result = await step.execute(sample_training_input, pipeline_state)
        assert result['success'] is False
        assert 'validation_errors' in result or 'error' in result

class TestStep03HMMRegimeDiscovery:
    """Test Step 3: HMM Regime Discovery."""

    @pytest.mark.asyncio
    async def test_hmm_regime_discovery(self, sample_config: Any, sample_training_input: Any, sample_features: List[Any]) -> None:
        """Test HMM regime discovery with features."""
        from src.training.steps.market_analysis.step03_hmm_regime_discovery import HMMRegimeDiscoveryStep
        sample_config['n_regimes'] = 3
        step = HMMRegimeDiscoveryStep(sample_config)
        pipeline_state = {'validated_data': sample_features, 'data_validation_results': {'data_quality_score': 90}}
        result = await step.execute(sample_training_input, pipeline_state)
        assert result['success'] is True
        assert 'features' in result
        assert 'hmm_results' in result
        assert 'regime_labels' in result
        assert 'regime_characteristics' in result
        labels = result['regime_labels']
        assert len(labels) == len(result['features'])
        assert len(np.unique(labels)) <= 3

    @pytest.mark.asyncio
    async def test_hmm_with_invalid_regimes(self, sample_config: Any, sample_training_input: Any, sample_features: List[Any]) -> None:
        """Test HMM with invalid number of regimes."""
        from src.training.steps.market_analysis.step03_hmm_regime_discovery import HMMRegimeDiscoveryStep
        sample_config['n_regimes'] = 15
        step = HMMRegimeDiscoveryStep(sample_config)
        pipeline_state = {'validated_data': sample_features}
        is_valid, errors = step.validate_inputs(sample_training_input, pipeline_state)
        assert is_valid is False
        assert any(('n_regimes' in error for error in errors))

class TestStep04RegimeDataSplitting:
    """Test Step 4: Regime Data Splitting."""

    @pytest.mark.asyncio
    async def test_regime_data_splitting(self, sample_config: Any, sample_training_input: Any, sample_features: List[Any], sample_regime_labels: List[Any]) -> None:
        """Test regime data splitting."""
        from src.training.steps.market_analysis.step04_regime_data_splitting import RegimeDataSplittingStep
        step = RegimeDataSplittingStep(sample_config)
        pipeline_state = {'features': sample_features, 'regime_labels': sample_regime_labels, 'validated_data': sample_features}
        result = await step.execute(sample_training_input, pipeline_state)
        assert result['success'] is True
        assert 'unified_data' in result
        assert 'train_data' in result
        assert 'val_data' in result
        assert 'test_data' in result
        assert 'regime_statistics' in result
        total_len = len(sample_features)
        train_len = len(result['train_data'])
        val_len = len(result['val_data'])
        test_len = len(result['test_data'])
        assert train_len + val_len + test_len == total_len
        assert 0.65 < train_len / total_len < 0.75
        assert 0.1 < val_len / total_len < 0.2
        assert 0.1 < test_len / total_len < 0.2

    @pytest.mark.asyncio
    async def test_stratified_splitting(self, sample_config: Any, sample_training_input: Any, sample_features: List[Any], sample_regime_labels: List[Any]) -> None:
        """Test that stratified splitting maintains regime distribution."""
        from src.training.steps.market_analysis.step04_regime_data_splitting import RegimeDataSplittingStep
        sample_config['stratify_by_regime'] = True
        step = RegimeDataSplittingStep(sample_config)
        pipeline_state = {'features': sample_features, 'regime_labels': sample_regime_labels, 'validated_data': sample_features}
        sample_features['regime_label'] = sample_regime_labels
        result = await step.execute(sample_training_input, pipeline_state)
        assert result['success'] is True
        original_dist = pd.Series(sample_regime_labels).value_counts(normalize=True)
        for split_name in ['train_data', 'val_data', 'test_data']:
            split_data = result[split_name]
            if 'regime_label' in split_data.columns:
                split_dist = split_data['regime_label'].value_counts(normalize=True)
                for regime in original_dist.index:
                    if regime in split_dist.index:
                        assert abs(original_dist[regime] - split_dist[regime]) < 0.1

class TestStep05Labeling:
    """Test Step 5: Labeling."""

    @pytest.mark.asyncio
    async def test_triple_barrier_labeling(self, sample_config: Any, sample_training_input: Any, sample_features: List[Any]) -> None:
        """Test triple barrier labeling."""
        from src.training.steps.model_training.step05_labeling import LabelingStep
        step = LabelingStep(sample_config)
        pipeline_state = {'unified_data': sample_features}
        result = await step.execute(sample_training_input, pipeline_state)
        assert result['success'] is True
        assert 'labeled_data' in result
        assert 'label_statistics' in result
        labeled_data = result['labeled_data']
        assert 'triple_barrier_label' in labeled_data.columns
        assert 'label_binary' in labeled_data.columns
        stats = result['label_statistics']
        assert stats['labeled_samples'] > 0
        assert 'class_distribution' in stats

    @pytest.mark.asyncio
    async def test_regime_aware_labeling(self, sample_config: Any, sample_training_input: Any, sample_features: List[Any], sample_regime_labels: List[Any]) -> None:
        """Test regime-aware labeling."""
        from src.training.steps.model_training.step05_labeling import LabelingStep
        sample_config['labeling_config']['regime_aware'] = True
        step = LabelingStep(sample_config)
        sample_features['regime_label'] = sample_regime_labels[:len(sample_features)]
        pipeline_state = {'unified_data': sample_features, 'regime_labels': sample_regime_labels[:len(sample_features)], 'regime_characteristics': {'regime_0': {'volatility_20_mean': 0.01}, 'regime_1': {'volatility_20_mean': 0.02}, 'regime_2': {'volatility_20_mean': 0.03}}}
        result = await step.execute(sample_training_input, pipeline_state)
        assert result['success'] is True
        labeled_data = result['labeled_data']
        if 'label_regime_at_entry' in labeled_data.columns:
            assert labeled_data['label_regime_at_entry'].notna().any()

class TestStep06FeatureEngineering:
    """Test Step 6: Feature Engineering."""

    @pytest.mark.asyncio
    async def test_feature_engineering(self, sample_config: Any, sample_training_input: Any, sample_market_data: Any) -> None:
        """Test feature engineering."""
        from src.training.steps.feature_engineering.step06_feature_engineering import FeatureEngineeringStep
        step = FeatureEngineeringStep(sample_config)
        sample_market_data['label'] = np.random.choice([-1, 0, 1], size=len(sample_market_data))
        pipeline_state = {'labeled_data': sample_market_data}
        result = await step.execute(sample_training_input, pipeline_state)
        assert result['success'] is True
        assert 'engineered_data' in result
        assert 'selected_features' in result
        assert 'feature_statistics' in result
        engineered = result['engineered_data']['all']
        feature_cols = [col for col in engineered.columns if col.startswith('feature_')]
        assert len(feature_cols) > 5
        selected = result['selected_features']
        assert len(selected) <= sample_config['feature_engineering_config']['feature_selection']['max_features']

    @pytest.mark.asyncio
    async def test_feature_selection(self, sample_config: Any, sample_training_input: Any, sample_features: List[Any]) -> None:
        """Test feature selection functionality."""
        from src.training.steps.feature_engineering.step06_feature_engineering import FeatureEngineeringStep
        sample_config['feature_engineering_config']['feature_selection']['max_features'] = 3
        step = FeatureEngineeringStep(sample_config)
        for i in range(10):
            sample_features[f'feature_corr_{i}'] = sample_features['feature_returns'] * (1 + i * 0.01)
        pipeline_state = {'labeled_data': sample_features}
        result = await step.execute(sample_training_input, pipeline_state)
        assert result['success'] is True
        selected = result['selected_features']
        assert len(selected) <= 3
        stats = result['feature_statistics']['all']
        assert len(stats['high_correlation_pairs']) > 0

class TestStep07MatrixOperations:
    """Test Step 7: Enhanced Matrix Operations."""

    @pytest.mark.asyncio
    async def test_matrix_operations(self, sample_config: Any, sample_training_input: Any, sample_features: List[Any]) -> None:
        """Test matrix operations."""
        from src.training.steps.model_training.step07_enhanced_matrix_operations import EnhancedMatrixOperationsStep
        sample_config['matrix_operations_config']['use_gpu'] = False
        step = EnhancedMatrixOperationsStep(sample_config)
        feature_cols = [col for col in sample_features.columns if col.startswith('feature_')]
        pipeline_state = {'engineered_data': {'train': sample_features}, 'selected_features': feature_cols[:5]}
        result = await step.execute(sample_training_input, pipeline_state)
        assert result['success'] is True
        assert 'matrix_results' in result
        assert 'feature_importance' in result
        assert 'optimization_insights' in result
        matrices = result['matrix_results']['train']
        assert 'correlation_matrix' in matrices
        assert 'covariance_matrix' in matrices
        n_features = len(feature_cols[:5])
        assert matrices['correlation_matrix'].shape == (n_features, n_features)
        assert matrices['covariance_matrix'].shape == (n_features, n_features)
        importance = result['feature_importance']
        assert 'aggregated_importance' in importance
        assert len(importance['aggregated_importance']) > 0

    @pytest.mark.asyncio
    async def test_regime_transition_matrix(self, sample_config: Any, sample_training_input: Any, sample_features: List[Any], sample_regime_labels: List[Any]) -> None:
        """Test regime transition matrix computation."""
        from src.training.steps.model_training.step07_enhanced_matrix_operations import EnhancedMatrixOperationsStep
        sample_config['matrix_operations_config']['matrix_computations']['regime_transition_matrix'] = True
        step = EnhancedMatrixOperationsStep(sample_config)
        sample_features['regime_label'] = sample_regime_labels[:len(sample_features)]
        pipeline_state = {'engineered_data': {'train': sample_features}, 'selected_features': [], 'regime_labels': sample_regime_labels[:len(sample_features)]}
        result = await step.execute(sample_training_input, pipeline_state)
        assert result['success'] is True
        matrices = result['matrix_results']['train']
        assert 'regime_transition_matrix' in matrices
        trans_matrix = matrices['regime_transition_matrix']
        n_regimes = len(np.unique(sample_regime_labels))
        assert trans_matrix.shape == (n_regimes, n_regimes)
        row_sums = trans_matrix.sum(axis=1)
        assert np.allclose(row_sums[row_sums > 0], 1.0)