#!/usr/bin/env python3
"""
Integration Test for Artifact Manager Adoption in Pre-Training Steps

This script tests the complete integration of the artifact manager across all
pre-training steps to ensure:
1. Artifacts are properly stored and retrieved
2. No data re-loads occur when artifacts are cached
3. All steps work correctly with the artifact manager
4. Backward compatibility is maintained
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.training.steps.pre_training.utils.artifact_manager import (
    get_pretraining_artifact_manager,
    ArtifactKeys,
)
from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_labeling_integration_step import (
    FeatureGenerationLabelingIntegrationStep
)
from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_feature_generation_step import (
    FeatureGenerationStep
)
from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_feature_selection_step import (
    FeatureGenerationFeatureSelectionStep
)
from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_period_lookback_optimization_step import (
    FeatureGenerationPeriodLookbackOptimizationStep
)
from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_interaction_generation_step import (
    FeatureGenerationInteractionGenerationStep
)
from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_vectorization_step import (
    FeatureGenerationVectorizationStep
)
from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_final_validation_step import (
    FeatureGenerationFinalValidationStep
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test data for the integration test."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='15T')
    
    # Create realistic OHLCV data
    n = len(dates)
    base_price = 100
    returns = np.random.normal(0, 0.02, n)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # Ensure high >= low and proper OHLC relationships
    data['high'] = np.maximum(data['high'], data['low'])
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data

async def test_artifact_manager_basic_functionality():
    """Test basic artifact manager functionality."""
    logger.info("🧪 Testing basic artifact manager functionality...")
    
    artifact_manager = get_pretraining_artifact_manager()
    
    # Test storing and retrieving simple data
    test_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    artifact_manager.store_enhanced(ArtifactKeys.RAW_DATAFRAME, test_data, {
        'test': True,
        'created_at': datetime.now().isoformat()
    })
    
    retrieved_data = artifact_manager.retrieve_enhanced(ArtifactKeys.RAW_DATAFRAME)
    
    assert retrieved_data is not None, "Failed to retrieve stored data"
    assert retrieved_data.equals(test_data), "Retrieved data doesn't match stored data"
    
    logger.info("✅ Basic artifact manager functionality test passed")

async def test_labeling_integration_step():
    """Test labeling integration step with artifact manager."""
    logger.info("🧪 Testing labeling integration step...")
    
    # Clear any existing artifacts
    artifact_manager = get_pretraining_artifact_manager()
    artifact_manager.clear_cache()
    
    # Create test data
    data = create_test_data()
    
    # Create step instance
    step = FeatureGenerationLabelingIntegrationStep()
    
    # First execution - should store artifacts
    result1 = step._process_data(data)
    assert result1['success'], f"First execution failed: {result1.get('error_message')}"
    assert not result1['integration_metadata'].get('cache_hit', False), "First execution should not be cached"
    
    # Second execution - should retrieve from cache
    result2 = step._process_data(data)
    assert result2['success'], f"Second execution failed: {result2.get('error_message')}"
    assert result2['integration_metadata'].get('cache_hit', False), "Second execution should be cached"
    
    # Verify artifacts are stored
    labeled_data = artifact_manager.retrieve_enhanced(ArtifactKeys.LABELED_DATAFRAME)
    targets = artifact_manager.retrieve_enhanced(ArtifactKeys.TARGETS)
    
    assert labeled_data is not None, "Labeled data not found in artifact manager"
    assert targets is not None, "Targets not found in artifact manager"
    
    logger.info("✅ Labeling integration step test passed")

async def test_feature_generation_step():
    """Test feature generation step with artifact manager."""
    logger.info("🧪 Testing feature generation step...")
    
    # Clear any existing artifacts
    artifact_manager = get_pretraining_artifact_manager()
    artifact_manager.clear_cache()
    
    # Create test data
    data = create_test_data()
    
    # Create step instance
    step = FeatureGenerationStep()
    
    # First execution - should store artifacts
    result1 = await step.execute(data)
    assert result1.success, f"First execution failed: {result1.error_message}"
    assert not result1.cache_hit, "First execution should not be cached"
    
    # Second execution - should retrieve from cache
    result2 = await step.execute(data)
    assert result2.success, f"Second execution failed: {result2.error_message}"
    assert result2.cache_hit, "Second execution should be cached"
    
    # Verify artifacts are stored
    features = artifact_manager.retrieve_enhanced(ArtifactKeys.FEATURE_DATAFRAME)
    feature_names = artifact_manager.retrieve_enhanced(ArtifactKeys.FEATURE_NAMES)
    
    assert features is not None, "Features not found in artifact manager"
    assert feature_names is not None, "Feature names not found in artifact manager"
    
    logger.info("✅ Feature generation step test passed")

async def test_feature_selection_step():
    """Test feature selection step with artifact manager."""
    logger.info("🧪 Testing feature selection step...")
    
    # Clear any existing artifacts
    artifact_manager = get_pretraining_artifact_manager()
    artifact_manager.clear_cache()
    
    # Create test data with features
    data = create_test_data()
    # Add some simple features
    data['feature_1'] = data['close'].pct_change()
    data['feature_2'] = data['volume'].rolling(5).mean()
    data['feature_3'] = data['high'] - data['low']
    data = data.dropna()
    
    targets = data['close'].pct_change().shift(-1).fillna(0)
    
    # Create step instance
    step = FeatureGenerationFeatureSelectionStep()
    
    # First execution - should store artifacts
    result1 = await step.execute(data, targets)
    assert result1.success, f"First execution failed: {result1.error_message}"
    assert not result1.selection_metadata.get('cache_hit', False), "First execution should not be cached"
    
    # Second execution - should retrieve from cache
    result2 = await step.execute(data, targets)
    assert result2.success, f"Second execution failed: {result2.error_message}"
    assert result2.selection_metadata.get('cache_hit', False), "Second execution should be cached"
    
    # Verify artifacts are stored
    selected_features = artifact_manager.retrieve_enhanced(ArtifactKeys.SELECTED_FEATURES)
    selection_metrics = artifact_manager.retrieve_enhanced(ArtifactKeys.SELECTION_METRICS)
    
    assert selected_features is not None, "Selected features not found in artifact manager"
    assert selection_metrics is not None, "Selection metrics not found in artifact manager"
    
    logger.info("✅ Feature selection step test passed")

async def test_period_lookback_optimization_step():
    """Test period/lookback optimization step with artifact manager."""
    logger.info("🧪 Testing period/lookback optimization step...")
    
    # Clear any existing artifacts
    artifact_manager = get_pretraining_artifact_manager()
    artifact_manager.clear_cache()
    
    # Create test data
    data = create_test_data()
    
    # Create step instance
    step = FeatureGenerationPeriodLookbackOptimizationStep()
    
    # First execution - should store artifacts
    result1 = step._process_data(data)
    assert result1['success'], f"First execution failed: {result1.get('error_message')}"
    assert not result1['artifacts'].get('cache_hit', False), "First execution should not be cached"
    
    # Second execution - should retrieve from cache
    result2 = step._process_data(data)
    assert result2['success'], f"Second execution failed: {result2.get('error_message')}"
    assert result2['artifacts'].get('cache_hit', False), "Second execution should be cached"
    
    # Verify artifacts are stored
    optimized_periods = artifact_manager.retrieve_enhanced(ArtifactKeys.OPTIMIZED_PERIODS)
    optimized_lookbacks = artifact_manager.retrieve_enhanced(ArtifactKeys.OPTIMIZED_LOOKBACKS)
    
    assert optimized_periods is not None, "Optimized periods not found in artifact manager"
    assert optimized_lookbacks is not None, "Optimized lookbacks not found in artifact manager"
    
    logger.info("✅ Period/lookback optimization step test passed")

async def test_interaction_generation_step():
    """Test interaction generation step with artifact manager."""
    logger.info("🧪 Testing interaction generation step...")
    
    # Clear any existing artifacts
    artifact_manager = get_pretraining_artifact_manager()
    artifact_manager.clear_cache()
    
    # Create test data
    data = create_test_data()
    
    # Create step instance
    step = FeatureGenerationInteractionGenerationStep()
    
    training_input = {'data': data}
    pipeline_state = {}
    
    # First execution - should store artifacts
    result1 = await step.execute(training_input, pipeline_state)
    assert result1.success, f"First execution failed: {result1.error_message}"
    assert not result1.artifacts.get('cache_hit', False), "First execution should not be cached"
    
    # Second execution - should retrieve from cache
    result2 = await step.execute(training_input, pipeline_state)
    assert result2.success, f"Second execution failed: {result2.error_message}"
    assert result2.artifacts.get('cache_hit', False), "Second execution should be cached"
    
    # Verify artifacts are stored
    interaction_features = artifact_manager.retrieve_enhanced(ArtifactKeys.INTERACTION_FEATURES)
    interaction_metadata = artifact_manager.retrieve_enhanced(ArtifactKeys.INTERACTION_METADATA)
    
    assert interaction_features is not None, "Interaction features not found in artifact manager"
    assert interaction_metadata is not None, "Interaction metadata not found in artifact manager"
    
    logger.info("✅ Interaction generation step test passed")

async def test_vectorization_step():
    """Test vectorization step with artifact manager."""
    logger.info("🧪 Testing vectorization step...")
    
    # Clear any existing artifacts
    artifact_manager = get_pretraining_artifact_manager()
    artifact_manager.clear_cache()
    
    # Create test data
    data = create_test_data()
    
    # Create step instance
    step = FeatureGenerationVectorizationStep()
    
    training_input = {'data': data}
    pipeline_state = {}
    
    # First execution - should store artifacts
    result1 = await step.execute(training_input, pipeline_state)
    assert result1.success, f"First execution failed: {result1.error_message}"
    assert not result1.artifacts.get('cache_hit', False), "First execution should not be cached"
    
    # Second execution - should retrieve from cache
    result2 = await step.execute(training_input, pipeline_state)
    assert result2.success, f"Second execution failed: {result2.error_message}"
    assert result2.artifacts.get('cache_hit', False), "Second execution should be cached"
    
    # Verify artifacts are stored
    vectorized_features = artifact_manager.retrieve_enhanced(ArtifactKeys.VECTORIZED_FEATURES)
    vectorization_metadata = artifact_manager.retrieve_enhanced(ArtifactKeys.VECTORIZATION_METADATA)
    
    assert vectorized_features is not None, "Vectorized features not found in artifact manager"
    assert vectorization_metadata is not None, "Vectorization metadata not found in artifact manager"
    
    logger.info("✅ Vectorization step test passed")

async def test_final_validation_step():
    """Test final validation step with artifact manager."""
    logger.info("🧪 Testing final validation step...")
    
    # Clear any existing artifacts
    artifact_manager = get_pretraining_artifact_manager()
    artifact_manager.clear_cache()
    
    # Create test data
    data = create_test_data()
    
    # Create step instance
    step = FeatureGenerationFinalValidationStep()
    
    # First execution - should store artifacts
    result1 = await step.execute(data)
    assert result1.success, f"First execution failed: {result1.error_message}"
    assert not result1.artifacts.get('cache_hit', False), "First execution should not be cached"
    
    # Second execution - should retrieve from cache
    result2 = await step.execute(data)
    assert result2.success, f"Second execution failed: {result2.error_message}"
    assert result2.artifacts.get('cache_hit', False), "Second execution should be cached"
    
    # Verify artifacts are stored
    final_dataset = artifact_manager.retrieve_enhanced(ArtifactKeys.FINAL_DATASET)
    validation_metrics = artifact_manager.retrieve_enhanced(ArtifactKeys.FINAL_VALIDATION_METRICS)
    
    assert final_dataset is not None, "Final dataset not found in artifact manager"
    assert validation_metrics is not None, "Validation metrics not found in artifact manager"
    
    logger.info("✅ Final validation step test passed")

async def test_full_pipeline_integration():
    """Test full pipeline integration with artifact manager."""
    logger.info("🧪 Testing full pipeline integration...")
    
    # Clear any existing artifacts
    artifact_manager = get_pretraining_artifact_manager()
    artifact_manager.clear_cache()
    
    # Create test data
    data = create_test_data()
    
    # Test the complete pipeline flow
    steps = [
        ("labeling_integration", FeatureGenerationLabelingIntegrationStep()),
        ("feature_generation", FeatureGenerationStep()),
        ("feature_selection", FeatureGenerationFeatureSelectionStep()),
        ("period_lookback_optimization", FeatureGenerationPeriodLookbackOptimizationStep()),
        ("interaction_generation", FeatureGenerationInteractionGenerationStep()),
        ("vectorization", FeatureGenerationVectorizationStep()),
        ("final_validation", FeatureGenerationFinalValidationStep())
    ]
    
    current_data = data
    
    for step_name, step in steps:
        logger.info(f"🔄 Executing {step_name}...")
        
        if step_name == "labeling_integration":
            result = step._process_data(current_data)
            assert result['success'], f"{step_name} failed: {result.get('error_message')}"
            current_data = result['artifacts']['labeled_dataframe']
            
        elif step_name == "feature_generation":
            result = await step.execute(current_data)
            assert result.success, f"{step_name} failed: {result.error_message}"
            current_data = result.generated_features
            
        elif step_name == "feature_selection":
            targets = current_data.get('target', current_data['close'].pct_change().shift(-1).fillna(0))
            result = await step.execute(current_data, targets)
            assert result.success, f"{step_name} failed: {result.error_message}"
            current_data = result.selected_features
            
        elif step_name == "period_lookback_optimization":
            result = step._process_data(current_data)
            assert result['success'], f"{step_name} failed: {result.get('error_message')}"
            
        elif step_name == "interaction_generation":
            training_input = {'data': current_data}
            pipeline_state = {}
            result = await step.execute(training_input, pipeline_state)
            assert result.success, f"{step_name} failed: {result.error_message}"
            current_data = result.interaction_features
            
        elif step_name == "vectorization":
            training_input = {'data': current_data}
            pipeline_state = {}
            result = await step.execute(training_input, pipeline_state)
            assert result.success, f"{step_name} failed: {result.error_message}"
            current_data = result.vectorized_features
            
        elif step_name == "final_validation":
            result = await step.execute(current_data)
            assert result.success, f"{step_name} failed: {result.error_message}"
            current_data = result.final_dataset
        
        logger.info(f"✅ {step_name} completed successfully")
    
    # Verify all artifacts are stored
    all_artifacts = [
        ArtifactKeys.LABELED_DATAFRAME,
        ArtifactKeys.FEATURE_DATAFRAME,
        ArtifactKeys.SELECTED_FEATURES,
        ArtifactKeys.OPTIMIZED_PERIODS,
        ArtifactKeys.INTERACTION_FEATURES,
        ArtifactKeys.VECTORIZED_FEATURES,
        ArtifactKeys.FINAL_DATASET
    ]
    
    for artifact_key in all_artifacts:
        artifact = artifact_manager.retrieve_enhanced(artifact_key)
        assert artifact is not None, f"Artifact {artifact_key} not found after full pipeline"
    
    logger.info("✅ Full pipeline integration test passed")

async def test_memory_efficiency():
    """Test memory efficiency with artifact manager."""
    logger.info("🧪 Testing memory efficiency...")
    
    artifact_manager = get_pretraining_artifact_manager()
    
    # Get initial memory metrics
    initial_metrics = artifact_manager.get_performance_metrics()
    logger.info(f"Initial cache size: {initial_metrics['cache_size_mb']:.2f} MB")
    
    # Test with large dataset
    large_data = create_test_data()
    # Duplicate to make it larger
    large_data = pd.concat([large_data] * 10, ignore_index=True)
    
    # Store large dataset
    artifact_manager.store_enhanced(ArtifactKeys.RAW_DATAFRAME, large_data, {
        'test': 'memory_efficiency',
        'size': len(large_data)
    })
    
    # Get memory metrics after storage
    after_storage_metrics = artifact_manager.get_performance_metrics()
    logger.info(f"After storage cache size: {after_storage_metrics['cache_size_mb']:.2f} MB")
    
    # Test retrieval
    retrieved_data = artifact_manager.retrieve_enhanced(ArtifactKeys.RAW_DATAFRAME)
    assert retrieved_data is not None, "Failed to retrieve large dataset"
    assert len(retrieved_data) == len(large_data), "Retrieved data size mismatch"
    
    logger.info("✅ Memory efficiency test passed")

async def main():
    """Run all integration tests."""
    logger.info("🚀 Starting Artifact Manager Integration Tests")
    logger.info("=" * 60)
    
    try:
        # Basic functionality tests
        await test_artifact_manager_basic_functionality()
        
        # Individual step tests
        await test_labeling_integration_step()
        await test_feature_generation_step()
        await test_feature_selection_step()
        await test_period_lookback_optimization_step()
        await test_interaction_generation_step()
        await test_vectorization_step()
        await test_final_validation_step()
        
        # Full pipeline integration test
        await test_full_pipeline_integration()
        
        # Memory efficiency test
        await test_memory_efficiency()
        
        logger.info("=" * 60)
        logger.info("🎉 All integration tests passed successfully!")
        logger.info("✅ Artifact Manager adoption is complete and working correctly")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
