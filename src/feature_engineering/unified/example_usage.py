"""
Example usage of the unified feature generation system.

This file demonstrates how to use the unified feature generation system
with various configurations and use cases.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

from .core import (
    FeatureGenerator, FeatureGeneratorConfig, FeatureGenerationResult,
    FeatureCategory, FeaturePriority
)
from .orchestrator import FeatureOrchestrator, OrchestrationConfig
from .registry import register_feature_generator
from .compatibility import BackwardsCompatibilityLayer, wrap_legacy_function
from .validation import FeatureValidator, FeatureConsistencyChecker, FeatureQualityMetrics
from .generators.technical_indicators import TechnicalIndicatorsGenerator


def create_sample_data(periods: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for examples."""
    dates = pd.date_range(start="2023-01-01", periods=periods, freq="1H")
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.randn(periods) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        "open": prices * (1 + np.random.randn(periods) * 0.001),
        "high": prices * (1 + np.abs(np.random.randn(periods)) * 0.002),
        "low": prices * (1 - np.abs(np.random.randn(periods)) * 0.002),
        "close": prices,
        "volume": np.random.randint(1000, 10000, periods)
    }, index=dates)
    
    # Ensure OHLC relationships
    data["high"] = np.maximum(data["high"], data[["open", "close"]].max(axis=1))
    data["low"] = np.minimum(data["low"], data[["open", "close"]].min(axis=1))
    
    return data


class CustomFeatureGenerator(FeatureGenerator):
    """Example custom feature generator."""
    
    def __init__(self, name: str = "custom_features"):
        config = FeatureGeneratorConfig(
            name=name,
            category=FeatureCategory.CUSTOM,
            priority=FeaturePriority.MEDIUM,
            enabled=True,
            parameters={
                "lookback_periods": [5, 10, 20],
                "custom_multiplier": 2.0
            }
        )
        super().__init__(config)
    
    async def initialize(self) -> bool:
        """Initialize the custom generator."""
        try:
            self.logger.info("Initializing custom feature generator...")
            self._is_initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Error initializing custom generator: {e}")
            return False
    
    async def generate_features(
        self, 
        data: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> FeatureGenerationResult:
        """Generate custom features."""
        try:
            if not self._is_initialized:
                return FeatureGenerationResult(
                    success=False,
                    errors=["Generator not initialized"]
                )
            
            # Validate input
            is_valid, errors = self.validate_input(data)
            if not is_valid:
                return FeatureGenerationResult(success=False, errors=errors)
            
            # Generate custom features
            features = pd.DataFrame(index=data.index)
            
            # Price momentum features
            lookback_periods = self.config.parameters.get("lookback_periods", [5, 10, 20])
            multiplier = self.config.parameters.get("custom_multiplier", 2.0)
            
            for period in lookback_periods:
                if len(data) >= period:
                    features[f"price_momentum_{period}"] = data["close"].pct_change(period) * multiplier
                    features[f"volume_momentum_{period}"] = data["volume"].pct_change(period) * multiplier
            
            # Volatility features
            features["price_volatility_5"] = data["close"].rolling(5).std()
            features["price_volatility_20"] = data["close"].rolling(20).std()
            features["volatility_ratio"] = features["price_volatility_5"] / features["price_volatility_20"]
            
            # Range features
            features["daily_range"] = (data["high"] - data["low"]) / data["close"]
            features["close_position"] = (data["close"] - data["low"]) / (data["high"] - data["low"])
            
            # Volume features
            features["volume_ma_ratio"] = data["volume"] / data["volume"].rolling(20).mean()
            features["volume_spike"] = (data["volume"] > data["volume"].rolling(20).mean() * 2).astype(int)
            
            # Validate output
            is_valid, errors = self.validate_output(features)
            if not is_valid:
                return FeatureGenerationResult(
                    success=False,
                    features=features,
                    errors=errors
                )
            
            return FeatureGenerationResult(
                success=True,
                features=features,
                metadata={
                    "generator": "custom",
                    "lookback_periods": lookback_periods,
                    "multiplier": multiplier
                },
                performance_metrics={
                    "features_generated": len(features.columns),
                    "lookback_periods_used": len(lookback_periods)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating custom features: {e}")
            return FeatureGenerationResult(
                success=False,
                errors=[f"Custom generation error: {str(e)}"]
            )
    
    def get_required_columns(self) -> List[str]:
        """Get required columns."""
        return ["open", "high", "low", "close", "volume"]
    
    def get_output_columns(self) -> List[str]:
        """Get output columns."""
        lookback_periods = self.config.parameters.get("lookback_periods", [5, 10, 20])
        
        output_columns = []
        for period in lookback_periods:
            output_columns.extend([
                f"price_momentum_{period}",
                f"volume_momentum_{period}"
            ])
        
        output_columns.extend([
            "price_volatility_5",
            "price_volatility_20", 
            "volatility_ratio",
            "daily_range",
            "close_position",
            "volume_ma_ratio",
            "volume_spike"
        ])
        
        return output_columns


def legacy_feature_function(data: pd.DataFrame) -> pd.DataFrame:
    """Example legacy feature function."""
    features = pd.DataFrame(index=data.index)
    features["legacy_sma_5"] = data["close"].rolling(5).mean()
    features["legacy_sma_20"] = data["close"].rolling(20).mean()
    features["legacy_rsi"] = 100 - (100 / (1 + data["close"].diff().rolling(14).apply(
        lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if abs(x[x < 0].sum()) > 0 else 0
    )))
    return features


async def example_basic_usage():
    """Example of basic usage."""
    print("=== Basic Usage Example ===")
    
    # Create sample data
    data = create_sample_data(100)
    print(f"Created sample data with {len(data)} rows and {len(data.columns)} columns")
    
    # Initialize orchestrator
    config = OrchestrationConfig(
        enable_parallel_processing=True,
        max_parallel_generators=2,
        enable_validation=True,
        enable_quality_checks=True
    )
    
    orchestrator = FeatureOrchestrator(config)
    await orchestrator.initialize()
    print("Orchestrator initialized")
    
    # Register technical indicators generator
    tech_generator = TechnicalIndicatorsGenerator()
    register_feature_generator("technical_indicators", TechnicalIndicatorsGenerator, tech_generator.config)
    print("Technical indicators generator registered")
    
    # Generate features
    result = await orchestrator.generate_features(
        data,
        generator_names=["technical_indicators"]
    )
    
    if result.success:
        print(f"✅ Generated {len(result.features.columns)} features")
        print(f"Features: {list(result.features.columns)[:10]}...")  # Show first 10
        print(f"Performance metrics: {result.performance_metrics}")
    else:
        print(f"❌ Feature generation failed: {result.errors}")


async def example_custom_generator():
    """Example of using custom generator."""
    print("\n=== Custom Generator Example ===")
    
    # Create sample data
    data = create_sample_data(100)
    
    # Initialize orchestrator
    orchestrator = FeatureOrchestrator(OrchestrationConfig())
    await orchestrator.initialize()
    
    # Register custom generator
    custom_generator = CustomFeatureGenerator()
    register_feature_generator("custom_features", CustomFeatureGenerator, custom_generator.config)
    print("Custom generator registered")
    
    # Generate features
    result = await orchestrator.generate_features(
        data,
        generator_names=["custom_features"]
    )
    
    if result.success:
        print(f"✅ Generated {len(result.features.columns)} custom features")
        print(f"Features: {list(result.features.columns)}")
        print(f"Metadata: {result.metadata}")
    else:
        print(f"❌ Custom feature generation failed: {result.errors}")


async def example_parallel_processing():
    """Example of parallel processing."""
    print("\n=== Parallel Processing Example ===")
    
    # Create sample data
    data = create_sample_data(200)
    
    # Initialize orchestrator with parallel processing
    config = OrchestrationConfig(
        enable_parallel_processing=True,
        max_parallel_generators=4,
        enable_validation=True
    )
    
    orchestrator = FeatureOrchestrator(config)
    await orchestrator.initialize()
    
    # Register multiple generators
    tech_generator = TechnicalIndicatorsGenerator()
    register_feature_generator("technical_indicators", TechnicalIndicatorsGenerator, tech_generator.config)
    
    custom_generator = CustomFeatureGenerator("custom_1")
    register_feature_generator("custom_1", CustomFeatureGenerator, custom_generator.config)
    
    custom_generator2 = CustomFeatureGenerator("custom_2")
    register_feature_generator("custom_2", CustomFeatureGenerator, custom_generator2.config)
    
    print("Multiple generators registered")
    
    # Generate features in parallel
    import time
    start_time = time.time()
    
    result = await orchestrator.generate_features(
        data,
        generator_names=["technical_indicators", "custom_1", "custom_2"]
    )
    
    duration = time.time() - start_time
    
    if result.success:
        print(f"✅ Generated {len(result.features.columns)} features in {duration:.2f}s")
        print(f"Parallel execution: {result.metadata.get('parallel_execution', False)}")
        print(f"Generator count: {result.metadata.get('generator_count', 0)}")
    else:
        print(f"❌ Parallel feature generation failed: {result.errors}")


async def example_backwards_compatibility():
    """Example of backwards compatibility."""
    print("\n=== Backwards Compatibility Example ===")
    
    # Create sample data
    data = create_sample_data(100)
    
    # Initialize orchestrator
    orchestrator = FeatureOrchestrator(OrchestrationConfig())
    await orchestrator.initialize()
    
    # Initialize compatibility layer
    compatibility = BackwardsCompatibilityLayer(orchestrator)
    await compatibility.initialize()
    
    # Register legacy function
    legacy_adapter = wrap_legacy_function(
        legacy_feature_function,
        required_columns=["close"],
        output_columns=["legacy_sma_5", "legacy_sma_20", "legacy_rsi"],
        name="legacy_features"
    )
    
    success = compatibility.register_legacy_function(
        "legacy_features",
        legacy_feature_function,
        ["close"],
        ["legacy_sma_5", "legacy_sma_20", "legacy_rsi"],
        FeatureCategory.CUSTOM
    )
    
    print(f"Legacy function registered: {success}")
    
    # Generate features using legacy method
    result = await compatibility.generate_features_legacy(
        data,
        method="legacy_features"
    )
    
    if result.success:
        print(f"✅ Generated {len(result.features.columns)} legacy features")
        print(f"Features: {list(result.features.columns)}")
    else:
        print(f"❌ Legacy feature generation failed: {result.errors}")


async def example_validation_and_quality():
    """Example of validation and quality checks."""
    print("\n=== Validation and Quality Example ===")
    
    # Create sample data
    data = create_sample_data(100)
    
    # Initialize orchestrator
    orchestrator = FeatureOrchestrator(OrchestrationConfig())
    await orchestrator.initialize()
    
    # Register generator
    custom_generator = CustomFeatureGenerator()
    register_feature_generator("custom_features", CustomFeatureGenerator, custom_generator.config)
    
    # Generate features
    result = await orchestrator.generate_features(
        data,
        generator_names=["custom_features"]
    )
    
    if not result.success:
        print(f"❌ Feature generation failed: {result.errors}")
        return
    
    # Initialize validator
    validator = FeatureValidator()
    await validator.initialize()
    
    # Validate features
    validation_result = await validator.validate_features(result, custom_generator)
    
    print(f"Validation result: {'✅ Valid' if validation_result.is_valid else '❌ Invalid'}")
    if validation_result.errors:
        print(f"Errors: {validation_result.errors}")
    if validation_result.warnings:
        print(f"Warnings: {validation_result.warnings}")
    
    # Calculate quality metrics
    quality_calculator = FeatureQualityMetrics()
    metrics = await quality_calculator.calculate_quality_metrics(result, custom_generator)
    
    print(f"Quality Metrics:")
    print(f"  Completeness: {metrics.completeness:.2f}")
    print(f"  Consistency: {metrics.consistency:.2f}")
    print(f"  Stability: {metrics.stability:.2f}")
    print(f"  Performance: {metrics.performance:.2f}")
    print(f"  Overall Score: {metrics.overall_score:.2f}")
    
    # Test consistency checking
    checker = FeatureConsistencyChecker()
    checker.set_baseline(result.features)
    
    # Check consistency with identical data
    is_consistent, details = await checker.check_consistency(result.features)
    print(f"Consistency check: {'✅ Consistent' if is_consistent else '❌ Inconsistent'}")


async def example_pipeline_creation():
    """Example of creating custom pipelines."""
    print("\n=== Pipeline Creation Example ===")
    
    # Create sample data
    data = create_sample_data(100)
    
    # Initialize orchestrator
    orchestrator = FeatureOrchestrator(OrchestrationConfig())
    await orchestrator.initialize()
    
    # Register generators
    tech_generator = TechnicalIndicatorsGenerator()
    register_feature_generator("technical_indicators", TechnicalIndicatorsGenerator, tech_generator.config)
    
    custom_generator = CustomFeatureGenerator()
    register_feature_generator("custom_features", CustomFeatureGenerator, custom_generator.config)
    
    # Create custom pipeline
    success = orchestrator.create_custom_pipeline(
        "comprehensive_features",
        ["technical_indicators", "custom_features"]
    )
    
    print(f"Pipeline created: {success}")
    
    if success:
        # Get pipeline info
        info = orchestrator.get_pipeline_info("comprehensive_features")
        print(f"Pipeline info: {info}")
        
        # Use pipeline
        result = await orchestrator.generate_features(
            data,
            pipeline_name="comprehensive_features"
        )
        
        if result.success:
            print(f"✅ Generated {len(result.features.columns)} features using pipeline")
        else:
            print(f"❌ Pipeline feature generation failed: {result.errors}")


async def example_error_handling():
    """Example of error handling and recovery."""
    print("\n=== Error Handling Example ===")
    
    # Create sample data
    data = create_sample_data(100)
    
    # Initialize orchestrator with retry enabled
    config = OrchestrationConfig(
        retry_failed_generators=True,
        max_retries=3,
        retry_delay_seconds=0.1
    )
    
    orchestrator = FeatureOrchestrator(config)
    await orchestrator.initialize()
    
    # Register a generator that might fail
    class FailingGenerator(FeatureGenerator):
        def __init__(self):
            config = FeatureGeneratorConfig(
                name="failing_generator",
                category=FeatureCategory.CUSTOM,
                enabled=True
            )
            super().__init__(config)
            self._call_count = 0
        
        async def initialize(self) -> bool:
            self._is_initialized = True
            return True
        
        async def generate_features(self, data: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> FeatureGenerationResult:
            self._call_count += 1
            if self._call_count < 3:  # Fail first 2 times
                return FeatureGenerationResult(
                    success=False,
                    errors=[f"Simulated failure {self._call_count}"]
                )
            
            # Succeed on 3rd try
            features = pd.DataFrame({"recovered_feature": data["close"] * 0.1}, index=data.index)
            return FeatureGenerationResult(success=True, features=features)
        
        def get_required_columns(self) -> List[str]:
            return ["close"]
        
        def get_output_columns(self) -> List[str]:
            return ["recovered_feature"]
    
    failing_generator = FailingGenerator()
    register_feature_generator("failing_generator", FailingGenerator, failing_generator.config)
    
    # Generate features (should retry and eventually succeed)
    result = await orchestrator.generate_features(
        data,
        generator_names=["failing_generator"]
    )
    
    if result.success:
        print(f"✅ Generator recovered after retries, generated {len(result.features.columns)} features")
    else:
        print(f"❌ Generator failed after retries: {result.errors}")


async def main():
    """Run all examples."""
    print("Unified Feature Generation System - Examples")
    print("=" * 50)
    
    try:
        await example_basic_usage()
        await example_custom_generator()
        await example_parallel_processing()
        await example_backwards_compatibility()
        await example_validation_and_quality()
        await example_pipeline_creation()
        await example_error_handling()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())