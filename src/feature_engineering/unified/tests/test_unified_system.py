"""
Comprehensive tests for the unified feature generation system.
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from ..core import (
    FeatureGenerator, FeatureGeneratorConfig, FeatureGenerationResult,
    FeatureCategory, FeaturePriority, CompositeFeatureGenerator
)
from ..orchestrator import FeatureOrchestrator, OrchestrationConfig
from ..registry import FeatureRegistry, register_feature_generator
from ..compatibility import BackwardsCompatibilityLayer, LegacyFeatureAdapter
from ..validation import FeatureValidator, FeatureConsistencyChecker, FeatureQualityMetrics
from ..generators.technical_indicators import TechnicalIndicatorsGenerator


class TestFeatureGenerator(FeatureGenerator):
    """Test feature generator for testing purposes."""
    
    def __init__(self, name: str = "test_generator"):
        config = FeatureGeneratorConfig(
            name=name,
            category=FeatureCategory.CUSTOM,
            priority=FeaturePriority.MEDIUM,
            enabled=True
        )
        super().__init__(config)
    
    async def initialize(self) -> bool:
        self._is_initialized = True
        return True
    
    async def generate_features(
        self, 
        data: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> FeatureGenerationResult:
        if not self._is_initialized:
            return FeatureGenerationResult(success=False, errors=["Not initialized"])
        
        # Generate simple test features
        features = pd.DataFrame(index=data.index)
        features["test_feature_1"] = data["close"] * 0.1
        features["test_feature_2"] = data["volume"] * 0.01
        
        return FeatureGenerationResult(
            success=True,
            features=features,
            metadata={"test": True}
        )
    
    def get_required_columns(self) -> List[str]:
        return ["close", "volume"]
    
    def get_output_columns(self) -> List[str]:
        return ["test_feature_1", "test_feature_2"]


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
    np.random.seed(42)
    
    data = pd.DataFrame({
        "open": 100 + np.random.randn(100).cumsum(),
        "high": 100 + np.random.randn(100).cumsum() + np.random.rand(100) * 2,
        "low": 100 + np.random.randn(100).cumsum() - np.random.rand(100) * 2,
        "close": 100 + np.random.randn(100).cumsum(),
        "volume": np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Ensure high >= low and high >= close >= low
    data["high"] = np.maximum(data["high"], data["close"])
    data["low"] = np.minimum(data["low"], data["close"])
    data["high"] = np.maximum(data["high"], data["open"])
    data["low"] = np.minimum(data["low"], data["open"])
    
    return data


@pytest.fixture
async def orchestrator():
    """Create and initialize orchestrator for testing."""
    config = OrchestrationConfig(
        enable_parallel_processing=True,
        max_parallel_generators=2,
        enable_validation=True,
        enable_quality_checks=True
    )
    
    orchestrator = FeatureOrchestrator(config)
    await orchestrator.initialize()
    return orchestrator


@pytest.fixture
async def test_generator():
    """Create test generator."""
    generator = TestFeatureGenerator()
    await generator.initialize()
    return generator


class TestFeatureGenerator:
    """Test the base FeatureGenerator class."""
    
    def test_initialization(self):
        """Test generator initialization."""
        generator = TestFeatureGenerator()
        assert generator.config.name == "test_generator"
        assert generator.config.category == FeatureCategory.CUSTOM
        assert not generator.is_initialized()
    
    @pytest.mark.asyncio
    async def test_initialize(self, test_generator):
        """Test generator initialization."""
        assert test_generator.is_initialized()
    
    @pytest.mark.asyncio
    async def test_generate_features(self, test_generator, sample_data):
        """Test feature generation."""
        result = await test_generator.generate_features(sample_data)
        
        assert result.success
        assert result.features is not None
        assert len(result.features.columns) == 2
        assert "test_feature_1" in result.features.columns
        assert "test_feature_2" in result.features.columns
        assert result.metadata["test"] is True
    
    def test_required_columns(self, test_generator):
        """Test required columns."""
        required = test_generator.get_required_columns()
        assert "close" in required
        assert "volume" in required
    
    def test_output_columns(self, test_generator):
        """Test output columns."""
        output = test_generator.get_output_columns()
        assert "test_feature_1" in output
        assert "test_feature_2" in output
    
    def test_validate_input(self, test_generator, sample_data):
        """Test input validation."""
        is_valid, errors = test_generator.validate_input(sample_data)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_input_missing_columns(self, test_generator):
        """Test input validation with missing columns."""
        data = pd.DataFrame({"open": [1, 2, 3]})
        is_valid, errors = test_generator.validate_input(data)
        assert not is_valid
        assert "Missing required columns" in errors[0]
    
    def test_validate_output(self, test_generator, sample_data):
        """Test output validation."""
        features = pd.DataFrame({
            "test_feature_1": [1, 2, 3],
            "test_feature_2": [4, 5, 6]
        })
        is_valid, errors = test_generator.validate_output(features)
        assert is_valid
        assert len(errors) == 0


class TestFeatureRegistry:
    """Test the feature registry system."""
    
    @pytest.mark.asyncio
    async def test_registry_initialization(self):
        """Test registry initialization."""
        registry = FeatureRegistry()
        await registry.initialize()
        assert registry._initialized
    
    def test_register_generator(self):
        """Test manual generator registration."""
        registry = FeatureRegistry()
        generator = TestFeatureGenerator("manual_test")
        
        success = registry.register_generator(
            "manual_test",
            TestFeatureGenerator,
            generator.config
        )
        assert success
        
        info = registry.get_generator("manual_test")
        assert info is not None
        assert info.name == "manual_test"
    
    def test_get_generators_by_category(self):
        """Test getting generators by category."""
        registry = FeatureRegistry()
        generator = TestFeatureGenerator("custom_test")
        
        registry.register_generator(
            "custom_test",
            TestFeatureGenerator,
            generator.config
        )
        
        generators = registry.get_generators_by_category(FeatureCategory.CUSTOM)
        assert len(generators) == 1
        assert generators[0].name == "custom_test"


class TestFeatureOrchestrator:
    """Test the feature orchestrator."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator._initialized
    
    @pytest.mark.asyncio
    async def test_generate_features_basic(self, orchestrator, sample_data):
        """Test basic feature generation."""
        # Register test generator
        test_generator = TestFeatureGenerator("orchestrator_test")
        register_feature_generator("orchestrator_test", TestFeatureGenerator, test_generator.config)
        
        result = await orchestrator.generate_features(
            sample_data,
            generator_names=["orchestrator_test"]
        )
        
        assert result.success
        assert result.features is not None
        assert len(result.features.columns) >= 2
    
    @pytest.mark.asyncio
    async def test_generate_features_parallel(self, orchestrator, sample_data):
        """Test parallel feature generation."""
        # Register multiple test generators
        for i in range(3):
            generator = TestFeatureGenerator(f"parallel_test_{i}")
            register_feature_generator(f"parallel_test_{i}", TestFeatureGenerator, generator.config)
        
        result = await orchestrator.generate_features(
            sample_data,
            generator_names=["parallel_test_0", "parallel_test_1", "parallel_test_2"]
        )
        
        assert result.success
        assert result.features is not None
        # Should have features from all generators
        assert len(result.features.columns) >= 6
    
    @pytest.mark.asyncio
    async def test_create_custom_pipeline(self, orchestrator):
        """Test creating custom pipeline."""
        # Register test generators
        for i in range(2):
            generator = TestFeatureGenerator(f"pipeline_test_{i}")
            register_feature_generator(f"pipeline_test_{i}", TestFeatureGenerator, generator.config)
        
        success = orchestrator.create_custom_pipeline(
            "test_pipeline",
            ["pipeline_test_0", "pipeline_test_1"]
        )
        
        assert success
        assert "test_pipeline" in orchestrator.list_pipelines()
        
        info = orchestrator.get_pipeline_info("test_pipeline")
        assert info is not None
        assert info["generator_count"] == 2


class TestBackwardsCompatibility:
    """Test backwards compatibility layer."""
    
    @pytest.mark.asyncio
    async def test_compatibility_layer_initialization(self, orchestrator):
        """Test compatibility layer initialization."""
        compatibility = BackwardsCompatibilityLayer(orchestrator)
        await compatibility.initialize()
        assert compatibility._initialized
    
    @pytest.mark.asyncio
    async def test_register_legacy_function(self, orchestrator):
        """Test registering legacy function."""
        compatibility = BackwardsCompatibilityLayer(orchestrator)
        await compatibility.initialize()
        
        def legacy_function(data: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"legacy_feature": data["close"] * 0.1}, index=data.index)
        
        success = compatibility.register_legacy_function(
            "legacy_test",
            legacy_function,
            ["close"],
            ["legacy_feature"],
            FeatureCategory.CUSTOM
        )
        
        assert success
        assert "legacy_test" in compatibility.list_legacy_adapters()
    
    @pytest.mark.asyncio
    async def test_generate_features_legacy(self, orchestrator, sample_data):
        """Test legacy feature generation."""
        compatibility = BackwardsCompatibilityLayer(orchestrator)
        await compatibility.initialize()
        
        # Register test generator for orchestrator method
        test_generator = TestFeatureGenerator("legacy_orchestrator_test")
        register_feature_generator("legacy_orchestrator_test", TestFeatureGenerator, test_generator.config)
        
        result = await compatibility.generate_features_legacy(
            sample_data,
            method="orchestrator"
        )
        
        assert result.success
        assert result.features is not None


class TestFeatureValidation:
    """Test feature validation system."""
    
    @pytest.mark.asyncio
    async def test_validator_initialization(self):
        """Test validator initialization."""
        validator = FeatureValidator()
        await validator.initialize()
        assert validator._initialized
    
    @pytest.mark.asyncio
    async def test_validate_features_success(self, test_generator, sample_data):
        """Test successful feature validation."""
        validator = FeatureValidator()
        await validator.initialize()
        
        result = await test_generator.generate_features(sample_data)
        validation_result = await validator.validate_features(result, test_generator)
        
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_features_with_nan(self, test_generator, sample_data):
        """Test validation with NaN values."""
        validator = FeatureValidator()
        await validator.initialize()
        
        # Create result with NaN values
        features = pd.DataFrame({
            "test_feature_1": [1, 2, np.nan, 4],
            "test_feature_2": [5, 6, 7, 8]
        })
        
        result = FeatureGenerationResult(
            success=True,
            features=features
        )
        
        validation_result = await validator.validate_features(result, test_generator)
        
        assert not validation_result.is_valid
        assert any("NaN values" in error for error in validation_result.errors)
    
    @pytest.mark.asyncio
    async def test_quality_metrics(self, test_generator, sample_data):
        """Test quality metrics calculation."""
        quality_calculator = FeatureQualityMetrics()
        
        result = await test_generator.generate_features(sample_data)
        metrics = await quality_calculator.calculate_quality_metrics(result, test_generator)
        
        assert metrics.completeness > 0
        assert metrics.consistency > 0
        assert metrics.stability > 0
        assert metrics.performance > 0
        assert metrics.overall_score > 0
    
    @pytest.mark.asyncio
    async def test_consistency_checker(self, sample_data):
        """Test consistency checker."""
        checker = FeatureConsistencyChecker()
        
        # Create baseline features
        baseline_features = pd.DataFrame({
            "feature_1": [1, 2, 3, 4],
            "feature_2": [5, 6, 7, 8]
        })
        checker.set_baseline(baseline_features)
        
        # Test with identical features
        current_features = baseline_features.copy()
        is_consistent, details = await checker.check_consistency(current_features)
        
        assert is_consistent
        assert details["is_consistent"]
    
    @pytest.mark.asyncio
    async def test_consistency_checker_different(self, sample_data):
        """Test consistency checker with different features."""
        checker = FeatureConsistencyChecker()
        
        # Create baseline features
        baseline_features = pd.DataFrame({
            "feature_1": [1, 2, 3, 4],
            "feature_2": [5, 6, 7, 8]
        })
        checker.set_baseline(baseline_features)
        
        # Test with different features
        current_features = pd.DataFrame({
            "feature_1": [1, 2, 3, 5],  # Different last value
            "feature_2": [5, 6, 7, 8]
        })
        is_consistent, details = await checker.check_consistency(current_features)
        
        assert not is_consistent
        assert not details["is_consistent"]


class TestTechnicalIndicatorsGenerator:
    """Test the technical indicators generator."""
    
    @pytest.mark.asyncio
    async def test_technical_indicators_initialization(self):
        """Test technical indicators generator initialization."""
        generator = TechnicalIndicatorsGenerator()
        success = await generator.initialize()
        assert success
    
    @pytest.mark.asyncio
    async def test_technical_indicators_generation(self, sample_data):
        """Test technical indicators generation."""
        generator = TechnicalIndicatorsGenerator()
        await generator.initialize()
        
        result = await generator.generate_features(sample_data)
        
        assert result.success
        assert result.features is not None
        assert len(result.features.columns) > 0
        
        # Check for some expected indicators
        output_columns = result.features.columns.tolist()
        assert any("sma_" in col for col in output_columns)
        assert any("rsi_" in col for col in output_columns)
    
    def test_required_columns(self):
        """Test required columns for technical indicators."""
        generator = TechnicalIndicatorsGenerator()
        required = generator.get_required_columns()
        
        assert "open" in required
        assert "high" in required
        assert "low" in required
        assert "close" in required
        assert "volume" in required
    
    def test_output_columns(self):
        """Test output columns for technical indicators."""
        generator = TechnicalIndicatorsGenerator()
        output = generator.get_output_columns()
        
        assert len(output) > 0
        assert any("sma_" in col for col in output)
        assert any("ema_" in col for col in output)
        assert any("rsi_" in col for col in output)


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_generation(self, sample_data):
        """Test complete end-to-end feature generation."""
        # Initialize orchestrator
        config = OrchestrationConfig(
            enable_parallel_processing=True,
            enable_validation=True,
            enable_quality_checks=True
        )
        orchestrator = FeatureOrchestrator(config)
        await orchestrator.initialize()
        
        # Register technical indicators generator
        tech_generator = TechnicalIndicatorsGenerator()
        register_feature_generator("technical_indicators", TechnicalIndicatorsGenerator, tech_generator.config)
        
        # Generate features
        result = await orchestrator.generate_features(
            sample_data,
            generator_names=["technical_indicators"]
        )
        
        # Validate result
        assert result.success
        assert result.features is not None
        assert len(result.features.columns) > 0
        
        # Check performance metrics
        assert "indicators_generated" in result.performance_metrics
        
        # Validate features
        validator = FeatureValidator()
        await validator.initialize()
        validation_result = await validator.validate_features(result)
        
        assert validation_result.is_valid
    
    @pytest.mark.asyncio
    async def test_backwards_compatibility_integration(self, sample_data):
        """Test backwards compatibility integration."""
        # Initialize orchestrator
        config = OrchestrationConfig()
        orchestrator = FeatureOrchestrator(config)
        await orchestrator.initialize()
        
        # Initialize compatibility layer
        compatibility = BackwardsCompatibilityLayer(orchestrator)
        await compatibility.initialize()
        
        # Register test generator
        test_generator = TestFeatureGenerator("compatibility_test")
        register_feature_generator("compatibility_test", TestFeatureGenerator, test_generator.config)
        
        # Test legacy generation
        result = await compatibility.generate_features_legacy(
            sample_data,
            method="orchestrator"
        )
        
        assert result.success
        assert result.features is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])