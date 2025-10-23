"""
Integration Tests for Base Classes Wiring

This module tests the integration and wiring of the new abstract base classes
with the existing codebase to ensure everything works correctly together.

Test Coverage:
1. Factory function integration
2. Backward compatibility with existing classes
3. Production vs existing class behavior
4. Configuration and validation integration
5. End-to-end pipeline integration
6. Error handling and edge cases
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
from typing import Dict, Any, List
import logging

# Import factory functions
from src.core.factory import (
    BaseClassFactory, create_validator, create_training_step,
    create_clustering_algorithm, create_multi_output_model,
    create_pattern_discoverer, create_labeling_strategy,
    create_complete_pipeline, ConfigurationPresets
)

# Import base classes
from src.core.abstract_base_classes import (
    ValidationLevel, TrainingStatus, ClusteringAlgorithm,
    PatternType, LabelingStrategy
)

# Import existing classes to test backward compatibility
from src.utils.base_validator import BaseValidator as ExistingBaseValidator
from src.utils.ml_common.training.base_training_step import BaseTrainingStep as ExistingBaseTrainingStep
from src.training.steps.market_analysis.components.clustering_algorithms import BaseClusteringAlgorithm as ExistingBaseClusteringAlgorithm
from src.utils.ml_common.models.multi_output_models import MultiOutputModel as ExistingMultiOutputModel
from src.research.price_patterns.pattern_discovery_framework import BasePatternDiscoverer as ExistingBasePatternDiscoverer
from src.research.profit_labeling.ensemble_labeling_system import BaseLabelingStrategy as ExistingBaseLabelingStrategy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test data generators
def generate_test_data(n_samples: int = 1000, n_features: int = 10) -> pd.DataFrame:
    """Generate test data for integration tests."""
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    return data

def generate_test_clustering_data(n_samples: int = 500, n_features: int = 2) -> np.ndarray:
    """Generate test clustering data."""
    np.random.seed(42)
    # Generate 3 clusters
    cluster1 = np.random.normal([0, 0], 0.5, (n_samples // 3, n_features))
    cluster2 = np.random.normal([3, 3], 0.5, (n_samples // 3, n_features))
    cluster3 = np.random.normal([-3, 3], 0.5, (n_samples - 2 * (n_samples // 3), n_features))
    
    return np.vstack([cluster1, cluster2, cluster3])

def generate_test_training_data(n_samples: int = 1000, n_features: int = 10) -> tuple:
    """Generate test training data."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    return X, y

def generate_test_price_data(n_samples: int = 1000) -> np.ndarray:
    """Generate test price data."""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = np.cumsum(returns) + 100
    return prices

# ============================================================================
# FACTORY FUNCTION TESTS
# ============================================================================

class TestFactoryFunctions:
    """Test factory function integration."""
    
    def test_create_validator_production(self):
        """Test creating validator with production base class."""
        validator = create_validator(
            "test_validator",
            validator_type="data",
            validation_level=ValidationLevel.PRODUCTION,
            use_production=True
        )
        
        assert validator is not None
        assert validator.name == "test_validator"
        assert validator.validation_level == ValidationLevel.PRODUCTION
        assert hasattr(validator, 'validate')
        assert hasattr(validator, 'get_validation_summary')

    def test_create_validator_existing(self):
        """Test creating validator with existing base class."""
        validator = create_validator(
            "test_validator",
            use_production=False
        )
        
        assert validator is not None
        assert isinstance(validator, ExistingBaseValidator)
        assert validator.step_name == "test_validator"
        assert hasattr(validator, 'validate')
        assert hasattr(validator, 'get_validation_summary')

    def test_create_training_step_production(self):
        """Test creating training step with production base class."""
        training_step = create_training_step(
            "test_training",
            model_type="random_forest",
            use_production=True
        )
        
        assert training_step is not None
        assert training_step.name == "test_training"
        assert hasattr(training_step, 'execute_training')
        assert hasattr(training_step, 'get_training_summary')

    def test_create_training_step_existing(self):
        """Test creating training step with existing base class."""
        training_step = create_training_step(
            "test_training",
            use_production=False
        )
        
        assert training_step is not None
        assert isinstance(training_step, ExistingBaseTrainingStep)
        assert hasattr(training_step, 'execute_training')

    def test_create_clustering_algorithm_production(self):
        """Test creating clustering algorithm with production base class."""
        clustering = create_clustering_algorithm(
            "test_clustering",
            algorithm=ClusteringAlgorithm.KMEANS,
            n_clusters=3,
            use_production=True
        )
        
        assert clustering is not None
        assert clustering.name == "test_clustering"
        assert clustering.algorithm == ClusteringAlgorithm.KMEANS
        assert hasattr(clustering, 'fit_predict')
        assert hasattr(clustering, 'get_clustering_summary')

    def test_create_clustering_algorithm_existing(self):
        """Test creating clustering algorithm with existing base class."""
        clustering = create_clustering_algorithm(
            "test_clustering",
            use_production=False
        )
        
        assert clustering is not None
        assert isinstance(clustering, ExistingBaseClusteringAlgorithm)
        assert hasattr(clustering, 'fit_predict')

    def test_create_multi_output_model_production(self):
        """Test creating multi-output model with production base class."""
        model = create_multi_output_model(
            "test_model",
            n_outputs=3,
            output_names=['output1', 'output2', 'output3'],
            use_production=True
        )
        
        assert model is not None
        assert model.name == "test_model"
        assert model.n_outputs == 3
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'get_model_summary')

    def test_create_multi_output_model_existing(self):
        """Test creating multi-output model with existing base class."""
        model = create_multi_output_model(
            "test_model",
            n_outputs=3,
            use_production=False
        )
        
        assert model is not None
        assert isinstance(model, ExistingMultiOutputModel)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_create_pattern_discoverer_production(self):
        """Test creating pattern discoverer with production base class."""
        discoverer = create_pattern_discoverer(
            "test_discoverer",
            pattern_type=PatternType.MOMENTUM,
            use_production=True
        )
        
        assert discoverer is not None
        assert discoverer.name == "test_discoverer"
        assert discoverer.pattern_type == PatternType.MOMENTUM
        assert hasattr(discoverer, 'discover_pattern')
        assert hasattr(discoverer, 'get_pattern_definition')

    def test_create_pattern_discoverer_existing(self):
        """Test creating pattern discoverer with existing base class."""
        discoverer = create_pattern_discoverer(
            "test_discoverer",
            use_production=False
        )
        
        assert discoverer is not None
        assert isinstance(discoverer, ExistingBasePatternDiscoverer)
        assert hasattr(discoverer, 'discover_pattern')
        assert hasattr(discoverer, 'get_pattern_definition')

    def test_create_labeling_strategy_production(self):
        """Test creating labeling strategy with production base class."""
        strategy = create_labeling_strategy(
            "test_labeling",
            strategy=LabelingStrategy.PROFIT_BASED,
            use_production=True
        )
        
        assert strategy is not None
        assert strategy.name == "test_labeling"
        assert strategy.strategy == LabelingStrategy.PROFIT_BASED
        assert hasattr(strategy, 'generate_labels')
        assert hasattr(strategy, 'calculate_confidence')

    def test_create_labeling_strategy_existing(self):
        """Test creating labeling strategy with existing base class."""
        strategy = create_labeling_strategy(
            "test_labeling",
            use_production=False
        )
        
        assert strategy is not None
        assert isinstance(strategy, ExistingBaseLabelingStrategy)
        assert hasattr(strategy, 'generate_labels')
        assert hasattr(strategy, 'calculate_confidence')

# ============================================================================
# CONFIGURATION PRESET TESTS
# ============================================================================

class TestConfigurationPresets:
    """Test configuration presets."""
    
    def test_production_config(self):
        """Test production configuration preset."""
        config = ConfigurationPresets.get_production_config()
        
        assert config['validation_level'] == ValidationLevel.PRODUCTION
        assert config['enable_logging'] == True
        assert config['enable_metrics'] == True
        assert config['enable_optimization'] == True

    def test_development_config(self):
        """Test development configuration preset."""
        config = ConfigurationPresets.get_development_config()
        
        assert config['validation_level'] == ValidationLevel.STANDARD
        assert config['enable_logging'] == True
        assert config['enable_metrics'] == True
        assert config['enable_optimization'] == False

    def test_testing_config(self):
        """Test testing configuration preset."""
        config = ConfigurationPresets.get_testing_config()
        
        assert config['validation_level'] == ValidationLevel.BASIC
        assert config['enable_logging'] == False
        assert config['enable_metrics'] == False
        assert config['enable_optimization'] == False

    def test_ml_pipeline_config(self):
        """Test ML pipeline configuration preset."""
        config = ConfigurationPresets.get_ml_pipeline_config()
        
        assert config['model_type'] == 'random_forest'
        assert config['n_estimators'] == 200
        assert config['max_depth'] == 10
        assert config['scale_features'] == True

    def test_clustering_config(self):
        """Test clustering configuration preset."""
        config = ConfigurationPresets.get_clustering_config()
        
        assert config['n_clusters'] == 5
        assert config['random_state'] == 42
        assert config['n_init'] == 10
        assert config['max_iter'] == 300

    def test_pattern_discovery_config(self):
        """Test pattern discovery configuration preset."""
        config = ConfigurationPresets.get_pattern_discovery_config()
        
        assert config['lookback_period'] == 20
        assert config['momentum_threshold'] == 0.03
        assert config['confidence_threshold'] == 0.7
        assert config['frequency_threshold'] == 0.1

    def test_labeling_config(self):
        """Test labeling configuration preset."""
        config = ConfigurationPresets.get_labeling_config()
        
        assert config['profit_threshold'] == 0.02
        assert config['lookforward_period'] == 5
        assert config['min_confidence'] == 0.6
        assert config['max_confidence'] == 1.0

# ============================================================================
# COMPLETE PIPELINE TESTS
# ============================================================================

class TestCompletePipeline:
    """Test complete pipeline integration."""
    
    def test_create_complete_pipeline_production(self):
        """Test creating complete pipeline with production base classes."""
        pipeline = create_complete_pipeline(
            "test_pipeline",
            config_preset="production",
            use_production=True
        )
        
        assert pipeline is not None
        assert pipeline['name'] == "test_pipeline"
        assert 'validator' in pipeline
        assert 'training_step' in pipeline
        assert 'clustering' in pipeline
        assert 'multi_output_model' in pipeline
        assert 'pattern_discoverer' in pipeline
        assert 'labeling_strategy' in pipeline
        
        # Test that all components are properly initialized
        assert pipeline['validator'].name == "test_pipeline_validator"
        assert pipeline['training_step'].name == "test_pipeline_training"
        assert pipeline['clustering'].name == "test_pipeline_clustering"
        assert pipeline['multi_output_model'].name == "test_pipeline_multi_output"
        assert pipeline['pattern_discoverer'].name == "test_pipeline_pattern_discoverer"
        assert pipeline['labeling_strategy'].name == "test_pipeline_labeling"

    def test_create_complete_pipeline_existing(self):
        """Test creating complete pipeline with existing base classes."""
        pipeline = create_complete_pipeline(
            "test_pipeline",
            config_preset="development",
            use_production=False
        )
        
        assert pipeline is not None
        assert pipeline['name'] == "test_pipeline"
        assert 'validator' in pipeline
        assert 'training_step' in pipeline
        assert 'clustering' in pipeline
        assert 'multi_output_model' in pipeline
        assert 'pattern_discoverer' in pipeline
        assert 'labeling_strategy' in pipeline

    def test_create_complete_pipeline_different_presets(self):
        """Test creating complete pipeline with different configuration presets."""
        presets = ["production", "development", "testing"]
        
        for preset in presets:
            pipeline = create_complete_pipeline(
                f"test_pipeline_{preset}",
                config_preset=preset,
                use_production=True
            )
            
            assert pipeline is not None
            assert pipeline['name'] == f"test_pipeline_{preset}"

# ============================================================================
# INTEGRATION FUNCTIONALITY TESTS
# ============================================================================

class TestIntegrationFunctionality:
    """Test integration functionality with real data."""
    
    @pytest.mark.asyncio
    async def test_validator_integration(self):
        """Test validator integration with real data."""
        validator = create_validator(
            "integration_validator",
            validation_level=ValidationLevel.PRODUCTION,
            use_production=True
        )
        
        # Test with valid data
        data = generate_test_data(100, 5)
        result = await validator.validate(data)
        
        assert result.is_valid
        assert result.execution_time > 0
        assert 'n_samples' in result.metrics
        assert 'n_features' in result.metrics

    def test_clustering_integration(self):
        """Test clustering integration with real data."""
        clustering = create_clustering_algorithm(
            "integration_clustering",
            algorithm=ClusteringAlgorithm.KMEANS,
            n_clusters=3,
            use_production=True
        )
        
        # Test with real data
        data = generate_test_clustering_data(100, 2)
        result = clustering.fit_predict(data)
        
        assert result.n_clusters == 3
        assert len(result.labels) == len(data)
        assert result.silhouette_score is not None
        assert result.inertia is not None

    def test_multi_output_model_integration(self):
        """Test multi-output model integration with real data."""
        model = create_multi_output_model(
            "integration_model",
            n_outputs=2,
            use_production=True
        )
        
        # Test with real data
        X, y = generate_test_training_data(100, 5)
        y_multi = np.column_stack([y, y * 2])  # Create 2 outputs
        
        model.fit(X, y_multi)
        predictions = model.predict(X[:10])
        
        assert predictions.shape == (10, 2)
        assert model.is_fitted

    def test_pattern_discoverer_integration(self):
        """Test pattern discoverer integration with real data."""
        discoverer = create_pattern_discoverer(
            "integration_discoverer",
            pattern_type=PatternType.MOMENTUM,
            use_production=True
        )
        
        # Test with real data
        price_data = generate_test_price_data(100)
        result = discoverer.discover_pattern(price_data)
        
        assert len(result.labels) == len(price_data)
        assert len(result.confidence_scores) == len(price_data)
        assert 0 <= result.frequency <= 1

    def test_labeling_strategy_integration(self):
        """Test labeling strategy integration with real data."""
        strategy = create_labeling_strategy(
            "integration_labeling",
            strategy=LabelingStrategy.PROFIT_BASED,
            use_production=True
        )
        
        # Test with real data
        price_data = generate_test_price_data(100)
        result = strategy.generate_labels(price_data)
        
        assert len(result.labels) == len(price_data)
        assert len(result.confidence_scores) == len(price_data)
        assert result.strategy == LabelingStrategy.PROFIT_BASED

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling in integration."""
    
    def test_factory_error_handling(self):
        """Test factory function error handling."""
        # Test with invalid parameters
        with pytest.raises(Exception):
            create_validator("", use_production=True)
        
        with pytest.raises(Exception):
            create_training_step("", use_production=True)
        
        with pytest.raises(Exception):
            create_clustering_algorithm("", use_production=True)

    def test_configuration_error_handling(self):
        """Test configuration error handling."""
        # Test with invalid configuration
        with pytest.raises(Exception):
            create_validator(
                "test_validator",
                config={'invalid_param': 'invalid_value'},
                use_production=True
            )

    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        # Test with invalid pipeline name
        with pytest.raises(Exception):
            create_complete_pipeline("", use_production=True)

# ============================================================================
# BACKWARD COMPATIBILITY TESTS
# ============================================================================

class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_existing_validator_compatibility(self):
        """Test that existing validator still works."""
        validator = create_validator(
            "compatibility_validator",
            use_production=False
        )
        
        assert isinstance(validator, ExistingBaseValidator)
        assert validator.step_name == "compatibility_validator"
        
        # Test that existing methods still work
        summary = validator.get_validation_summary()
        assert 'step_name' in summary
        assert 'validation_count' in summary

    def test_existing_training_step_compatibility(self):
        """Test that existing training step still works."""
        training_step = create_training_step(
            "compatibility_training",
            use_production=False
        )
        
        assert isinstance(training_step, ExistingBaseTrainingStep)
        
        # Test that existing methods still work
        assert hasattr(training_step, 'execute_training')

    def test_mixed_pipeline_compatibility(self):
        """Test mixed pipeline with both production and existing classes."""
        # Create some components with production classes
        validator = create_validator("prod_validator", use_production=True)
        clustering = create_clustering_algorithm("prod_clustering", use_production=True)
        
        # Create some components with existing classes
        training_step = create_training_step("existing_training", use_production=False)
        model = create_multi_output_model("existing_model", n_outputs=2, use_production=False)
        
        # Test that all components work together
        assert validator is not None
        assert clustering is not None
        assert training_step is not None
        assert model is not None

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance of integrated components."""
    
    def test_factory_performance(self):
        """Test factory function performance."""
        import time
        
        start_time = time.time()
        
        # Create multiple components
        for i in range(10):
            create_validator(f"perf_validator_{i}", use_production=True)
            create_training_step(f"perf_training_{i}", use_production=True)
            create_clustering_algorithm(f"perf_clustering_{i}", use_production=True)
        
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < 5.0

    def test_pipeline_creation_performance(self):
        """Test pipeline creation performance."""
        import time
        
        start_time = time.time()
        
        # Create multiple pipelines
        for i in range(5):
            create_complete_pipeline(f"perf_pipeline_{i}", use_production=True)
        
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < 10.0

if __name__ == "__main__":
    pytest.main([__file__])