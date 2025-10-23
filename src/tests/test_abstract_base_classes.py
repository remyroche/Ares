"""
Comprehensive Tests for Abstract Base Classes

This module provides comprehensive tests for all abstract base classes
and their concrete implementations to ensure production readiness.

Test Coverage:
1. BaseValidator - Validation framework testing
2. BaseTrainingStep - Training pipeline testing
3. BaseClusteringAlgorithm - Clustering algorithm testing
4. MultiOutputModel - Multi-output model testing
5. BasePatternDiscoverer - Pattern discovery testing
6. BaseLabelingStrategy - Labeling strategy testing
"""

import pytest
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
import asyncio
import time
import os
import tempfile

# Import base classes
from src.core.abstract_base_classes import (
    BaseValidator, BaseTrainingStep, BaseClusteringAlgorithm,
    MultiOutputModel, BasePatternDiscoverer, BaseLabelingStrategy,
    ValidationResult, TrainingResult, ClusteringResult, PatternDiscoveryResult,
    PatternDefinition, LabelingResult, ValidationLevel, TrainingStatus,
    ClusteringAlgorithm, PatternType, LabelingStrategy
)

# Import concrete implementations
from src.core.concrete_implementations import (
    DataValidator, MLTrainingStep, KMeansClustering,
    MultiOutputRandomForest, MomentumPatternDiscoverer, ProfitBasedLabeling
)

# Test data generators
def generate_test_data(n_samples: int = 1000, n_features: int = 10) -> pd.DataFrame:
    """Generate test data for validation."""
    np.random.seed(42)
    # Generate more realistic data with correlations
    base_data = np.random.randn(n_samples, n_features)
    
    # Add some correlations between features
    for i in range(1, min(3, n_features)):
        base_data[:, i] = 0.7 * base_data[:, 0] + 0.3 * base_data[:, i]
    
    data = pd.DataFrame(
        base_data,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some missing values for testing
    if n_samples > 100:
        missing_indices = np.random.choice(n_samples, size=n_samples // 20, replace=False)
        data.iloc[missing_indices, 0] = np.nan
    
    return data

def generate_test_clustering_data(n_samples: int = 500, n_features: int = 2) -> np.ndarray:
    """Generate test data for clustering."""
    np.random.seed(42)
    # Generate 3 well-separated clusters with different sizes
    cluster1 = np.random.normal([0, 0], 0.5, (n_samples // 3, n_features))
    cluster2 = np.random.normal([4, 4], 0.6, (n_samples // 3, n_features))
    cluster3 = np.random.normal([-4, 4], 0.4, (n_samples - 2 * (n_samples // 3), n_features))
    
    # Add some noise points
    noise_points = np.random.uniform(-6, 6, (n_samples // 20, n_features))
    
    return np.vstack([cluster1, cluster2, cluster3, noise_points])

def generate_test_training_data(n_samples: int = 1000, n_features: int = 10) -> tuple:
    """Generate test data for training."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Create more realistic target with some relationship to features
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * np.random.randn(n_samples)
    
    return X, y

def generate_test_multi_output_data(n_samples: int = 1000, n_features: int = 10, n_outputs: int = 3) -> tuple:
    """Generate test data for multi-output models."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Create realistic multi-output targets with different relationships
    y = np.zeros((n_samples, n_outputs))
    for i in range(n_outputs):
        y[:, i] = (0.4 + 0.1 * i) * X[:, 0] + (0.3 - 0.05 * i) * X[:, 1] + 0.1 * np.random.randn(n_samples)
    
    return X, y

def generate_test_price_data(n_samples: int = 1000) -> np.ndarray:
    """Generate test price data for pattern discovery."""
    np.random.seed(42)
    # Generate more realistic price data with trends and volatility clusters
    returns = np.random.normal(0.001, 0.02, n_samples)
    
    # Add some trend periods
    trend_periods = n_samples // 4
    for i in range(0, n_samples, trend_periods):
        end_idx = min(i + trend_periods, n_samples)
        trend_strength = np.random.choice([-0.002, 0.002])
        returns[i:end_idx] += trend_strength
    
    # Add volatility clustering
    volatility = np.ones(n_samples) * 0.02
    for i in range(50, n_samples, 100):
        end_idx = min(i + 50, n_samples)
        volatility[i:end_idx] *= 2.0  # High volatility period
    
    returns = returns * volatility
    prices = np.cumsum(returns) + 100
    
    return prices

# ============================================================================
# BASE VALIDATOR TESTS
# ============================================================================

class TestBaseValidator:
    """Test cases for BaseValidator abstract class."""
    
    def test_initialization(self):
        """Test validator initialization."""
        validator = DataValidator("test_validator")
        assert validator.name == "test_validator"
        assert validator.validation_level == ValidationLevel.STANDARD
        assert validator.total_validations == 0
        assert validator.successful_validations == 0
        assert validator.failed_validations == 0

    def test_validation_history_tracking(self):
        """Test validation history tracking."""
        validator = DataValidator("test_validator")
        
        # Create mock validation results
        result1 = ValidationResult(is_valid=True, execution_time=0.1)
        result2 = ValidationResult(is_valid=False, errors=["test error"], execution_time=0.2)
        
        validator._record_validation(result1)
        validator._record_validation(result2)
        
        assert validator.total_validations == 2
        assert validator.successful_validations == 1
        assert validator.failed_validations == 1
        assert validator.get_success_rate() == 0.5

    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        validator = DataValidator("test_validator")
        
        # Record multiple validations
        for i in range(5):
            result = ValidationResult(is_valid=True, execution_time=0.1 * (i + 1))
            validator._record_validation(result)
        
        metrics = validator.get_performance_summary()
        assert metrics['total_validations'] == 5
        assert metrics['successful_validations'] == 5
        assert metrics['success_rate'] == 1.0
        assert metrics['avg_validation_time'] == 0.3  # (0.1 + 0.2 + 0.3 + 0.4 + 0.5) / 5

    def test_clear_history(self):
        """Test history clearing."""
        validator = DataValidator("test_validator")
        
        # Add some validation results
        for i in range(3):
            result = ValidationResult(is_valid=True, execution_time=0.1)
            validator._record_validation(result)
        
        assert validator.total_validations == 3
        
        # Clear history
        validator.clear_history()
        
        assert validator.total_validations == 0
        assert validator.successful_validations == 0
        assert validator.failed_validations == 0

class TestDataValidator:
    """Test cases for DataValidator concrete implementation."""
    
    def test_initialization(self):
        """Test data validator initialization."""
        validator = DataValidator("test_data_validator")
        assert validator.name == "test_data_validator"
        assert validator.required_columns == []
        assert validator.max_missing_ratio == 0.1
        assert validator.min_samples == 10

    def test_validation_with_valid_data(self):
        """Test validation with valid data."""
        validator = DataValidator("test_validator")
        data = generate_test_data(100, 5)
        
        # Test synchronous validation
        result = validator.validate_sync(data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.metrics['n_samples'] == 100
        assert result.metrics['n_features'] == 5

    def test_validation_with_invalid_data(self):
        """Test validation with invalid data."""
        validator = DataValidator(
            "test_validator",
            config={
                'required_columns': ['feature_0', 'feature_1'],
                'min_samples': 50
            }
        )
        
        # Create data with missing required columns
        data = pd.DataFrame({'feature_2': [1, 2, 3]})
        
        result = validator.validate_sync(data)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any('Missing required columns' in error for error in result.errors)

    def test_validation_with_missing_values(self):
        """Test validation with missing values."""
        validator = DataValidator(
            "test_validator",
            config={'max_missing_ratio': 0.05}
        )
        
        # Create data with many missing values
        data = generate_test_data(100, 5)
        data.iloc[:50, :3] = np.nan  # 50% missing in first 3 columns
        
        result = validator.validate_sync(data)
        
        assert not result.is_valid
        assert any('Too many missing values' in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_async_validation(self):
        """Test asynchronous validation."""
        validator = DataValidator("test_validator")
        data = generate_test_data(100, 5)
        
        result = await validator.validate(data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid

    def test_validation_summary(self):
        """Test validation summary generation."""
        validator = DataValidator("test_validator")
        
        # Add some validation results
        for i in range(3):
            data = generate_test_data(100, 5)
            validator.validate_sync(data)
        
        summary = validator.get_validation_summary()
        
        assert summary['total_validations'] == 3
        assert summary['successful_validations'] == 3
        assert summary['success_rate'] == 1.0

# ============================================================================
# BASE TRAINING STEP TESTS
# ============================================================================

class TestBaseTrainingStep:
    """Test cases for BaseTrainingStep abstract class."""
    
    def test_initialization(self):
        """Test training step initialization."""
        step = MLTrainingStep("test_step")
        assert step.name == "test_step"
        assert step.status == TrainingStatus.NOT_STARTED
        assert len(step.training_results) == 0
        assert step.current_model is None

    def test_training_summary(self):
        """Test training summary generation."""
        step = MLTrainingStep("test_step")
        
        summary = step.get_training_summary()
        
        assert summary['name'] == "test_step"
        assert summary['status'] == TrainingStatus.NOT_STARTED.value
        assert summary['number_of_training_runs'] == 0

    def test_model_save_load(self):
        """Test model saving and loading."""
        step = MLTrainingStep("test_step")
        
        # Create a real model by training it
        X, y = generate_test_training_data(100, 5)
        result = asyncio.run(step.execute_training((X, y)))
        assert result.success
        assert step.current_model is not None
        
        # Test saving
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            success = step.save_model(tmp_file.name)
            assert success
            
            # Test loading
            step.current_model = None
            success = step.load_model(tmp_file.name)
            assert success
            assert step.current_model is not None
            
            # Clean up
            os.unlink(tmp_file.name)

class TestMLTrainingStep:
    """Test cases for MLTrainingStep concrete implementation."""
    
    def test_initialization(self):
        """Test ML training step initialization."""
        step = MLTrainingStep("test_ml_step", model_type="random_forest")
        assert step.name == "test_ml_step"
        assert step.model_type == "random_forest"
        assert step.model is None

    def test_data_processing(self):
        """Test data processing."""
        step = MLTrainingStep("test_ml_step")
        data = generate_test_training_data(100, 5)
        X, y = data
        
        processed_data = step._process_data(X)
        
        assert isinstance(processed_data, np.ndarray)
        assert processed_data.shape == X.shape

    def test_artifact_generation(self):
        """Test artifact generation."""
        step = MLTrainingStep("test_ml_step")
        
        # Create a real model by training it
        X, y = generate_test_training_data(100, 5)
        result = asyncio.run(step.execute_training((X, y)))
        assert result.success
        
        artifacts = step._generate_artifacts(step.current_model, result)
        
        assert 'model_type' in artifacts
        assert 'feature_names' in artifacts
        assert 'training_timestamp' in artifacts
        assert 'model_params' in artifacts

    def test_metrics_calculation(self):
        """Test metrics calculation."""
        step = MLTrainingStep("test_ml_step")
        
        # Create a real model by training it
        X, y = generate_test_training_data(100, 5)
        result = asyncio.run(step.execute_training((X, y)))
        assert result.success
        
        # Generate test data
        X_test, y_test = generate_test_training_data(50, 5)
        test_data = (X_test, y_test)
        
        metrics = step._calculate_metrics(step.current_model, test_data)
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert isinstance(metrics['mse'], float)
        assert isinstance(metrics['mae'], float)
        assert isinstance(metrics['r2'], float)

    @pytest.mark.asyncio
    async def test_training_execution(self):
        """Test complete training execution."""
        step = MLTrainingStep("test_ml_step")
        data = generate_test_training_data(100, 5)
        X, y = data
        
        result = await step.execute_training((X, y))
        
        assert isinstance(result, TrainingResult)
        assert result.success
        assert result.model is not None
        assert result.training_time > 0

# ============================================================================
# BASE CLUSTERING ALGORITHM TESTS
# ============================================================================

class TestBaseClusteringAlgorithm:
    """Test cases for BaseClusteringAlgorithm abstract class."""
    
    def test_initialization(self):
        """Test clustering algorithm initialization."""
        algorithm = KMeansClustering("test_clustering", n_clusters=3)
        assert algorithm.name == "test_clustering"
        assert algorithm.algorithm == ClusteringAlgorithm.KMEANS
        assert algorithm.n_clusters == 3
        assert not algorithm.is_fitted

    def test_clustering_summary(self):
        """Test clustering summary generation."""
        algorithm = KMeansClustering("test_clustering")
        
        summary = algorithm.get_clustering_summary()
        
        assert summary['name'] == "test_clustering"
        assert summary['algorithm'] == ClusteringAlgorithm.KMEANS.value
        assert summary['is_fitted'] == False

    def test_silhouette_score_calculation(self):
        """Test silhouette score calculation."""
        algorithm = KMeansClustering("test_clustering")
        data = generate_test_clustering_data(100, 2)
        labels = np.random.randint(0, 3, 100)
        
        score = algorithm.get_silhouette_score(data, labels)
        
        assert isinstance(score, float)
        assert -1 <= score <= 1

    def test_inertia_calculation(self):
        """Test inertia calculation."""
        algorithm = KMeansClustering("test_clustering")
        data = generate_test_clustering_data(100, 2)
        labels = np.random.randint(0, 3, 100)
        
        inertia = algorithm.get_inertia(data, labels)
        
        assert isinstance(inertia, float)
        assert inertia >= 0

class TestKMeansClustering:
    """Test cases for KMeansClustering concrete implementation."""
    
    def test_initialization(self):
        """Test K-means clustering initialization."""
        algorithm = KMeansClustering("test_kmeans", n_clusters=3)
        assert algorithm.name == "test_kmeans"
        assert algorithm.n_clusters == 3
        assert algorithm.model is None

    def test_fit_predict(self):
        """Test fit_predict method."""
        algorithm = KMeansClustering("test_kmeans", n_clusters=3)
        data = generate_test_clustering_data(100, 2)
        
        result = algorithm.fit_predict(data)
        
        assert isinstance(result, ClusteringResult)
        assert result.n_clusters == 3
        assert len(result.labels) == len(data)
        assert result.algorithm == 'kmeans'
        assert result.silhouette_score is not None
        assert result.inertia is not None

    def test_fit_method(self):
        """Test fit method."""
        algorithm = KMeansClustering("test_kmeans", n_clusters=3)
        data = generate_test_clustering_data(100, 2)
        
        fitted_algorithm = algorithm.fit(data)
        
        assert fitted_algorithm is algorithm
        assert algorithm.is_fitted
        assert algorithm.model is not None

    def test_predict_method(self):
        """Test predict method."""
        algorithm = KMeansClustering("test_kmeans", n_clusters=3)
        data = generate_test_clustering_data(100, 2)
        
        # Fit first
        algorithm.fit(data)
        
        # Predict on new data
        new_data = generate_test_clustering_data(50, 2)
        labels = algorithm.predict(new_data)
        
        assert len(labels) == len(new_data)
        assert all(0 <= label < 3 for label in labels)

    def test_cluster_centers(self):
        """Test cluster centers retrieval."""
        algorithm = KMeansClustering("test_kmeans", n_clusters=3)
        data = generate_test_clustering_data(100, 2)
        
        algorithm.fit(data)
        centers = algorithm.get_cluster_centers()
        
        assert centers is not None
        assert centers.shape == (3, 2)

# ============================================================================
# MULTI-OUTPUT MODEL TESTS
# ============================================================================

class TestMultiOutputModel:
    """Test cases for MultiOutputModel abstract class."""
    
    def test_initialization(self):
        """Test multi-output model initialization."""
        model = MultiOutputRandomForest("test_model", n_outputs=3)
        assert model.name == "test_model"
        assert model.n_outputs == 3
        assert len(model.output_names) == 3
        assert not model.is_fitted

    def test_model_summary(self):
        """Test model summary generation."""
        model = MultiOutputRandomForest("test_model", n_outputs=2)
        
        summary = model.get_model_summary()
        
        assert summary['name'] == "test_model"
        assert summary['n_outputs'] == 2
        assert summary['is_fitted'] == False

    def test_feature_importance(self):
        """Test feature importance retrieval."""
        model = MultiOutputRandomForest("test_model", n_outputs=2)
        
        # Before fitting
        importance = model.get_feature_importance()
        assert importance is None

class TestMultiOutputRandomForest:
    """Test cases for MultiOutputRandomForest concrete implementation."""
    
    def test_initialization(self):
        """Test multi-output random forest initialization."""
        model = MultiOutputRandomForest("test_rf", n_outputs=3)
        assert model.name == "test_rf"
        assert model.n_outputs == 3
        assert len(model.output_names) == 3

    def test_fit_method(self):
        """Test fit method."""
        model = MultiOutputRandomForest("test_rf", n_outputs=2)
        X, y = generate_test_multi_output_data(100, 5, 2)
        
        fitted_model = model.fit(X, y)
        
        assert fitted_model is model
        assert model.is_fitted
        assert len(model.models) == 2

    def test_predict_method(self):
        """Test predict method."""
        model = MultiOutputRandomForest("test_rf", n_outputs=2)
        X, y = generate_test_multi_output_data(100, 5, 2)
        
        # Fit first
        model.fit(X, y)
        
        # Predict
        predictions = model.predict(X[:10])
        
        assert predictions.shape == (10, 2)

    def test_evaluate_performance(self):
        """Test performance evaluation."""
        model = MultiOutputRandomForest("test_rf", n_outputs=2)
        X, y = generate_test_multi_output_data(100, 5, 2)
        
        # Fit first
        model.fit(X, y)
        
        # Evaluate
        results = model.evaluate_performance(X, y)
        
        assert 'per_output_metrics' in results
        assert 'overall_metrics' in results
        assert 'predictions' in results
        assert 'targets' in results

    def test_model_save_load(self):
        """Test model saving and loading."""
        model = MultiOutputRandomForest("test_rf", n_outputs=2)
        X, y = generate_test_multi_output_data(100, 5, 2)
        
        # Fit first
        model.fit(X, y)
        
        # Save
        success = model.save_model("/tmp/test_multi_output_model.pkl")
        assert success
        
        # Load
        new_model = MultiOutputRandomForest("new_model", n_outputs=2)
        success = new_model.load_model("/tmp/test_multi_output_model.pkl")
        assert success
        assert new_model.is_fitted
        assert len(new_model.models) == 2

# ============================================================================
# BASE PATTERN DISCOVERER TESTS
# ============================================================================

class TestBasePatternDiscoverer:
    """Test cases for BasePatternDiscoverer abstract class."""
    
    def test_initialization(self):
        """Test pattern discoverer initialization."""
        discoverer = MomentumPatternDiscoverer("test_discoverer")
        assert discoverer.name == "test_discoverer"
        assert discoverer.pattern_type == PatternType.MOMENTUM
        assert len(discoverer.discovered_patterns) == 0

    def test_pattern_validation(self):
        """Test pattern validation."""
        discoverer = MomentumPatternDiscoverer("test_discoverer")
        
        # Create real pattern result by discovering patterns
        data = generate_test_price_data(100)
        result = discoverer.discover_pattern(data)
        
        is_valid = discoverer.validate_pattern(result)
        assert isinstance(is_valid, bool)
        
        # Test with invalid pattern (low frequency)
        definition = PatternDefinition(
            name="Test Pattern",
            pattern_type=PatternType.MOMENTUM,
            description="Test",
            mathematical_formula="test",
            parameters={},
            frequency_threshold=0.9,  # Very high threshold
            confidence_threshold=0.7
        )
        
        invalid_result = PatternDiscoveryResult(
            definition=definition,
            labels=np.array([1, 0, 1, 0]),
            confidence_scores=np.array([0.8, 0.2, 0.9, 0.1]),
            frequency=0.5  # Below threshold
        )
        
        is_invalid = discoverer.validate_pattern(invalid_result)
        assert not is_invalid

    def test_pattern_summary(self):
        """Test pattern summary generation."""
        discoverer = MomentumPatternDiscoverer("test_discoverer")
        
        summary = discoverer.get_pattern_summary()
        
        assert summary['name'] == "test_discoverer"
        assert summary['pattern_type'] == PatternType.MOMENTUM.value
        assert summary['discovered_patterns'] == 0

class TestMomentumPatternDiscoverer:
    """Test cases for MomentumPatternDiscoverer concrete implementation."""
    
    def test_initialization(self):
        """Test momentum pattern discoverer initialization."""
        discoverer = MomentumPatternDiscoverer("test_momentum")
        assert discoverer.name == "test_momentum"
        assert discoverer.pattern_type == PatternType.MOMENTUM
        assert discoverer.lookback_period == 20

    def test_pattern_discovery(self):
        """Test pattern discovery."""
        discoverer = MomentumPatternDiscoverer("test_momentum")
        data = generate_test_price_data(100)
        
        result = discoverer.discover_pattern(data)
        
        assert isinstance(result, PatternDiscoveryResult)
        assert len(result.labels) == len(data)
        assert len(result.confidence_scores) == len(data)
        assert 0 <= result.frequency <= 1

    def test_pattern_definition(self):
        """Test pattern definition generation."""
        discoverer = MomentumPatternDiscoverer("test_momentum")
        
        definition = discoverer.get_pattern_definition()
        
        assert isinstance(definition, PatternDefinition)
        assert definition.name == "Momentum Pattern"
        assert definition.pattern_type == PatternType.MOMENTUM
        assert 'momentum' in definition.mathematical_formula.lower()

    def test_momentum_calculation(self):
        """Test momentum calculation."""
        discoverer = MomentumPatternDiscoverer("test_momentum")
        data = np.array([100, 101, 102, 103, 104, 105])
        
        momentum = discoverer._calculate_momentum(data)
        
        assert len(momentum) == len(data)
        assert momentum[0] == 0  # First value should be 0
        assert momentum[-1] > 0  # Last value should be positive

    def test_confidence_calculation(self):
        """Test confidence calculation."""
        discoverer = MomentumPatternDiscoverer("test_momentum")
        momentum = np.array([0.1, 0.05, 0.2, 0.0])
        pattern_mask = np.array([1, 1, 1, 0])
        
        confidence = discoverer._calculate_confidence(momentum, pattern_mask)
        
        assert len(confidence) == len(momentum)
        assert all(0 <= c <= 1 for c in confidence)
        assert confidence[3] == 0  # No pattern, no confidence

# ============================================================================
# BASE LABELING STRATEGY TESTS
# ============================================================================

class TestBaseLabelingStrategy:
    """Test cases for BaseLabelingStrategy abstract class."""
    
    def test_initialization(self):
        """Test labeling strategy initialization."""
        strategy = ProfitBasedLabeling("test_labeling")
        assert strategy.name == "test_labeling"
        assert strategy.strategy == LabelingStrategy.PROFIT_BASED
        assert len(strategy.labeling_results) == 0

    def test_label_validation(self):
        """Test label validation."""
        strategy = ProfitBasedLabeling("test_labeling")
        
        # Valid labels
        valid_labels = np.array([0, 1, 0, 1, 1])
        assert strategy.validate_labels(valid_labels)
        
        # Invalid labels (all same)
        invalid_labels = np.array([1, 1, 1, 1, 1])
        assert not strategy.validate_labels(invalid_labels)

    def test_labeling_summary(self):
        """Test labeling summary generation."""
        strategy = ProfitBasedLabeling("test_labeling")
        
        summary = strategy.get_labeling_summary()
        
        assert summary['name'] == "test_labeling"
        assert summary['strategy'] == LabelingStrategy.PROFIT_BASED.value
        assert summary['labeling_results'] == 0

class TestProfitBasedLabeling:
    """Test cases for ProfitBasedLabeling concrete implementation."""
    
    def test_initialization(self):
        """Test profit-based labeling initialization."""
        strategy = ProfitBasedLabeling("test_profit_labeling")
        assert strategy.name == "test_profit_labeling"
        assert strategy.strategy == LabelingStrategy.PROFIT_BASED
        assert strategy.profit_threshold == 0.02

    def test_label_generation(self):
        """Test label generation."""
        strategy = ProfitBasedLabeling("test_profit_labeling")
        data = generate_test_price_data(100)
        
        result = strategy.generate_labels(data)
        
        assert isinstance(result, LabelingResult)
        assert len(result.labels) == len(data)
        assert len(result.confidence_scores) == len(data)
        assert result.strategy == LabelingStrategy.PROFIT_BASED

    def test_confidence_calculation(self):
        """Test confidence calculation."""
        strategy = ProfitBasedLabeling("test_profit_labeling")
        labels = np.array([1, 0, 1, 0])
        data = np.array([100, 101, 102, 103])
        
        confidence = strategy.calculate_confidence(labels, data)
        
        assert len(confidence) == len(labels)
        assert all(0 <= c <= 1 for c in confidence)

    def test_future_profits_calculation(self):
        """Test future profits calculation."""
        strategy = ProfitBasedLabeling("test_profit_labeling")
        prices = np.array([100, 101, 102, 103, 104, 105])
        
        profits = strategy._calculate_future_profits(prices)
        
        assert len(profits) == len(prices)
        assert profits[0] == 0  # First value should be 0
        assert profits[-1] == 0  # Last value should be 0 (no future data)

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_validation_training(self):
        """Test end-to-end validation and training pipeline."""
        # Create validator with specific requirements
        validator = DataValidator(
            "integration_validator",
            config={
                'required_columns': ['feature_0', 'feature_1'],
                'min_samples': 50,
                'max_missing_ratio': 0.1
            }
        )
        
        # Generate and validate data
        data = generate_test_data(100, 5)
        validation_result = validator.validate_sync(data)
        assert validation_result.is_valid
        assert validation_result.metrics['n_samples'] == 100
        assert validation_result.metrics['n_features'] == 5
        
        # Create training step with specific configuration
        training_step = MLTrainingStep(
            "integration_training",
            model_type="random_forest",
            config={
                'n_estimators': 50,
                'max_depth': 10,
                'scale_features': True
            }
        )
        
        # Train model
        X, y = generate_test_training_data(100, 5)
        training_result = asyncio.run(training_step.execute_training((X, y)))
        assert training_result.success
        assert training_result.model is not None
        assert training_result.training_time > 0
        
        # Verify training summary
        summary = training_step.get_training_summary()
        assert summary['successful_runs'] == 1
        assert summary['status'] == 'completed'
        assert summary['latest_model'] == True

    def test_end_to_end_clustering_labeling(self):
        """Test end-to-end clustering and labeling pipeline."""
        # Create clustering algorithm
        clustering = KMeansClustering(
            "integration_clustering", 
            n_clusters=3,
            config={'enable_optimization': True}
        )
        
        # Generate and cluster data
        data = generate_test_clustering_data(100, 2)
        clustering_result = clustering.fit_predict(data)
        assert clustering_result.n_clusters == 3
        assert clustering_result.silhouette_score is not None
        assert clustering_result.inertia is not None
        assert len(clustering_result.labels) == len(data)
        
        # Verify clustering quality
        assert clustering_result.silhouette_score > 0.3  # Reasonable clustering quality
        
        # Create labeling strategy with specific configuration
        labeling = ProfitBasedLabeling(
            "integration_labeling",
            config={
                'profit_threshold': 0.01,
                'lookforward_period': 3,
                'min_confidence': 0.6
            }
        )
        
        # Generate labels
        price_data = generate_test_price_data(100)
        labeling_result = labeling.generate_labels(price_data)
        assert len(labeling_result.labels) == len(price_data)
        assert len(labeling_result.confidence_scores) == len(price_data)
        assert labeling_result.strategy == LabelingStrategy.PROFIT_BASED
        
        # Verify label quality
        assert np.isfinite(labeling_result.labels).all()
        assert np.isfinite(labeling_result.confidence_scores).all()
        assert np.all((labeling_result.labels >= 0) & (labeling_result.labels <= 1))

    def test_end_to_end_multi_output_training(self):
        """Test end-to-end multi-output training pipeline."""
        # Create multi-output model with specific configuration
        model = MultiOutputRandomForest(
            "integration_multi_output", 
            n_outputs=2,
            output_names=['output_1', 'output_2'],
            config={
                'n_estimators': 50,
                'max_depth': 8,
                'enable_optimization': True
            }
        )
        
        # Generate and train on data
        X, y = generate_test_multi_output_data(100, 5, 2)
        model.fit(X, y)
        assert model.is_fitted
        assert len(model.models) == 2
        assert model.models['output_1'] is not None
        assert model.models['output_2'] is not None
        
        # Make predictions
        predictions = model.predict(X[:10])
        assert predictions.shape == (10, 2)
        assert np.isfinite(predictions).all()
        
        # Evaluate performance
        results = model.evaluate_performance(X, y)
        assert 'overall_metrics' in results
        assert 'per_output_metrics' in results
        assert 'predictions' in results
        assert 'targets' in results
        
        # Verify metrics are reasonable
        overall_metrics = results['overall_metrics']
        assert 'overall_mse' in overall_metrics
        assert 'overall_mae' in overall_metrics
        assert 'overall_r2' in overall_metrics
        assert overall_metrics['overall_r2'] > 0.5  # Reasonable R² score

# ============================================================================
# COMPREHENSIVE INTEGRATION TESTS
# ============================================================================

class TestComprehensiveIntegration:
    """Comprehensive integration tests without mocks."""
    
    def test_full_ml_pipeline(self):
        """Test complete ML pipeline from validation to prediction."""
        # Step 1: Data validation
        validator = DataValidator(
            "pipeline_validator",
            config={
                'min_samples': 50,
                'max_missing_ratio': 0.05
            }
        )
        
        data = generate_test_data(200, 8)
        validation_result = validator.validate_sync(data)
        assert validation_result.is_valid
        
        # Step 2: Data preprocessing and training
        training_step = MLTrainingStep(
            "pipeline_training",
            model_type="random_forest",
            config={
                'n_estimators': 100,
                'scale_features': True
            }
        )
        
        X, y = generate_test_training_data(200, 8)
        training_result = asyncio.run(training_step.execute_training((X, y)))
        assert training_result.success
        
        # Step 3: Model evaluation
        X_test, y_test = generate_test_training_data(50, 8)
        test_data = (X_test, y_test)
        metrics = training_step._calculate_metrics(training_step.current_model, test_data)
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert metrics['r2'] > 0.3  # Reasonable performance
        
        # Step 4: Model persistence
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            success = training_step.save_model(tmp_file.name)
            assert success
            
            # Load and verify
            new_step = MLTrainingStep("loaded_training")
            success = new_step.load_model(tmp_file.name)
            assert success
            assert new_step.current_model is not None
            
            # Make predictions with loaded model
            predictions = new_step.current_model.predict(X_test[:5])
            assert len(predictions) == 5
            assert np.isfinite(predictions).all()
            
            os.unlink(tmp_file.name)
    
    def test_clustering_with_validation(self):
        """Test clustering with comprehensive validation."""
        # Generate data with known structure
        data = generate_test_clustering_data(300, 3)
        
        # Test different cluster numbers
        for n_clusters in [2, 3, 4]:
            clustering = KMeansClustering(
                f"clustering_{n_clusters}",
                n_clusters=n_clusters,
                config={'enable_optimization': True}
            )
            
            result = clustering.fit_predict(data)
            
            # Validate results
            assert result.n_clusters == n_clusters
            assert len(result.labels) == len(data)
            assert len(set(result.labels)) <= n_clusters
            assert result.silhouette_score is not None
            assert result.inertia is not None
            
            # Test prediction on new data
            new_data = generate_test_clustering_data(50, 3)
            new_labels = clustering.predict(new_data)
            assert len(new_labels) == len(new_data)
            assert all(0 <= label < n_clusters for label in new_labels)
    
    def test_pattern_discovery_validation(self):
        """Test pattern discovery with comprehensive validation."""
        # Generate realistic price data
        price_data = generate_test_price_data(500)
        
        # Test momentum pattern discovery
        discoverer = MomentumPatternDiscoverer(
            "comprehensive_discoverer",
            config={
                'lookback_period': 20,
                'momentum_threshold': 0.02,
                'confidence_threshold': 0.6
            }
        )
        
        result = discoverer.discover_pattern(price_data)
        
        # Validate pattern discovery results
        assert len(result.labels) == len(price_data)
        assert len(result.confidence_scores) == len(price_data)
        assert 0 <= result.frequency <= 1
        assert np.all((result.labels >= 0) & (result.labels <= 1))
        assert np.all((result.confidence_scores >= 0) & (result.confidence_scores <= 1))
        
        # Validate pattern definition
        definition = discoverer.get_pattern_definition()
        assert definition.name == "Momentum Pattern"
        assert definition.pattern_type == PatternType.MOMENTUM
        assert 'momentum' in definition.mathematical_formula.lower()
        
        # Test pattern validation
        is_valid = discoverer.validate_pattern(result)
        assert isinstance(is_valid, bool)
        
        # Test pattern summary
        summary = discoverer.get_pattern_summary()
        assert summary['name'] == "comprehensive_discoverer"
        assert summary['pattern_type'] == PatternType.MOMENTUM.value
        assert summary['discovered_patterns'] >= 1
    
    def test_labeling_strategy_comprehensive(self):
        """Test labeling strategy with comprehensive validation."""
        # Generate realistic price data
        price_data = generate_test_price_data(300)
        
        # Test profit-based labeling
        labeling = ProfitBasedLabeling(
            "comprehensive_labeling",
            config={
                'profit_threshold': 0.015,
                'lookforward_period': 5,
                'min_confidence': 0.5
            }
        )
        
        result = labeling.generate_labels(price_data)
        
        # Validate labeling results
        assert len(result.labels) == len(price_data)
        assert len(result.confidence_scores) == len(price_data)
        assert result.strategy == LabelingStrategy.PROFIT_BASED
        assert np.all((result.labels >= 0) & (result.labels <= 1))
        assert np.all((result.confidence_scores >= 0) & (result.confidence_scores <= 1))
        
        # Test confidence calculation
        confidence = labeling.calculate_confidence(result.labels, price_data)
        assert len(confidence) == len(price_data)
        assert np.all((confidence >= 0) & (confidence <= 1))
        
        # Test label validation
        is_valid = labeling.validate_labels(result.labels)
        assert is_valid
        
        # Test labeling summary
        summary = labeling.get_labeling_summary()
        assert summary['name'] == "comprehensive_labeling"
        assert summary['strategy'] == LabelingStrategy.PROFIT_BASED.value
        assert summary['labeling_results'] >= 1

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance tests for the base classes."""
    
    def test_validation_performance(self):
        """Test validation performance with large datasets."""
        validator = DataValidator(
            "performance_validator",
            config={
                'min_samples': 1000,
                'max_missing_ratio': 0.1
            }
        )
        
        # Generate large dataset
        data = generate_test_data(10000, 50)
        
        start_time = time.time()
        result = validator.validate_sync(data)
        execution_time = time.time() - start_time
        
        assert result.is_valid
        assert result.metrics['n_samples'] == 10000
        assert result.metrics['n_features'] == 50
        assert execution_time < 10.0  # Should complete within 10 seconds
        
        # Verify performance metrics are tracked
        performance_summary = validator.get_performance_summary()
        assert performance_summary['total_validations'] == 1
        assert performance_summary['success_rate'] == 1.0

    def test_training_performance(self):
        """Test training performance with large datasets."""
        training_step = MLTrainingStep(
            "performance_training",
            model_type="random_forest",
            config={
                'n_estimators': 50,  # Reduced for performance test
                'max_depth': 10,
                'scale_features': True
            }
        )
        
        # Generate large dataset
        X, y = generate_test_training_data(1000, 20)
        
        start_time = time.time()
        result = asyncio.run(training_step.execute_training((X, y)))
        execution_time = time.time() - start_time
        
        assert result.success
        assert result.model is not None
        assert result.training_time > 0
        assert execution_time < 30.0  # Should complete within 30 seconds
        
        # Verify training summary
        summary = training_step.get_training_summary()
        assert summary['successful_runs'] == 1
        assert summary['total_training_time'] > 0

    def test_clustering_performance(self):
        """Test clustering performance with large datasets."""
        clustering = KMeansClustering(
            "performance_clustering", 
            n_clusters=5,
            config={'enable_optimization': True}
        )
        
        # Generate large dataset
        data = generate_test_clustering_data(1000, 10)
        
        start_time = time.time()
        result = clustering.fit_predict(data)
        execution_time = time.time() - start_time
        
        assert result.n_clusters == 5
        assert result.silhouette_score is not None
        assert result.inertia is not None
        assert len(result.labels) == len(data)
        assert execution_time < 15.0  # Should complete within 15 seconds
        
        # Verify clustering summary
        summary = clustering.get_clustering_summary()
        assert summary['total_samples_processed'] == 1000
        assert summary['number_of_clustering_runs'] == 1

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Error handling tests for the base classes."""
    
    def test_validation_error_handling(self):
        """Test validation error handling."""
        validator = DataValidator(
            "error_validator",
            config={
                'required_columns': ['required_col'],
                'min_samples': 100
            }
        )
        
        # Test with invalid data type
        result = validator.validate_sync("invalid_data")
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any('Validation failed with exception' in error for error in result.errors)
        
        # Test with insufficient data
        small_data = generate_test_data(50, 3)  # Below min_samples
        result = validator.validate_sync(small_data)
        assert not result.is_valid
        assert any('Insufficient samples' in error for error in result.errors)
        
        # Test with missing required columns
        data_no_required = generate_test_data(200, 3)
        result = validator.validate_sync(data_no_required)
        assert not result.is_valid
        assert any('Missing required columns' in error for error in result.errors)

    def test_training_error_handling(self):
        """Test training error handling."""
        training_step = MLTrainingStep(
            "error_training",
            model_type="random_forest",
            config={'n_estimators': 10}
        )
        
        # Test with invalid data
        result = asyncio.run(training_step.execute_training("invalid_data"))
        assert not result.success
        assert len(result.errors) > 0
        
        # Test with insufficient data
        X, y = generate_test_training_data(5, 10)  # Very small dataset
        result = asyncio.run(training_step.execute_training((X, y)))
        # This might succeed but with warnings, or fail depending on implementation
        assert isinstance(result, TrainingResult)
        
        # Test with mismatched dimensions
        X = np.random.randn(100, 5)
        y = np.random.randn(50)  # Mismatched length
        result = asyncio.run(training_step.execute_training((X, y)))
        assert not result.success
        assert len(result.errors) > 0

    def test_clustering_error_handling(self):
        """Test clustering error handling."""
        clustering = KMeansClustering("error_clustering", n_clusters=3)
        
        # Test with insufficient data
        data = np.array([[1, 2]])  # Only 1 sample for 3 clusters
        with pytest.raises(ValueError):
            clustering.fit_predict(data)

    def test_multi_output_error_handling(self):
        """Test multi-output model error handling."""
        model = MultiOutputRandomForest("error_multi_output", n_outputs=2)
        
        # Test with mismatched output dimensions
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 3)  # 3 outputs instead of 2
        
        with pytest.raises(ValueError):
            model.fit(X, y)

if __name__ == "__main__":
    pytest.main([__file__])