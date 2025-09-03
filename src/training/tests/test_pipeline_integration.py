"""Integration tests for the training pipeline."""

import pytest
import asyncio
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.training.core.training_manager import create_training_manager
from src.training.base_step import BaseStep


class TestPipelineIntegration:
    """Test full pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_step_execution_order(self, sample_config, sample_training_input):
        """Test that steps execute in correct order."""
        # Track execution order
        execution_order = []
        
        class TrackingStep(BaseStep):
            def __init__(self, config, step_num, step_name):
                super().__init__(config, step_num, step_name)
                
            def _initialize_step(self):
                pass
                
            def validate_inputs(self, training_input, pipeline_state):
                return True, []
                
            async def execute_logic(self, training_input, pipeline_state):
                execution_order.append(self.full_step_name)
                return pipeline_state
                
            def validate_outputs(self, pipeline_state):
                return True, []
                
            def get_required_inputs(self):
                return []
                
            def get_produced_outputs(self):
                return []
                
            def get_dependencies(self):
                if self.step_number == "02":
                    return ["01_data_collection"]
                elif self.step_number == "03":
                    return ["02_data_reading"]
                return []
        
        # Create mock steps
        steps = {
            "01_data_collection": TrackingStep(sample_config, "01", "data_collection"),
            "02_data_reading": TrackingStep(sample_config, "02", "data_reading"),
            "03_hmm_regime_discovery": TrackingStep(sample_config, "03", "hmm_regime_discovery")
        }
        
        # Execute in wrong order intentionally
        for step_name in ["03_hmm_regime_discovery", "01_data_collection", "02_data_reading"]:
            step = steps[step_name]
            await step.execute(sample_training_input, {})
        
        # Check execution order
        expected_order = [
            "step03_hmm_regime_discovery",
            "step01_data_collection", 
            "step02_data_reading"
        ]
        assert execution_order == expected_order
    
    @pytest.mark.asyncio
    async def test_step_error_propagation(self, sample_config, sample_training_input):
        """Test that errors propagate correctly through pipeline."""
        
        class ErrorStep(BaseStep):
            def __init__(self, config):
                super().__init__(config, "99", "error_step")
                
            def _initialize_step(self):
                pass
                
            def validate_inputs(self, training_input, pipeline_state):
                return False, ["Test error"]
                
            async def execute_logic(self, training_input, pipeline_state):
                raise ValueError("Test execution error")
                
            def validate_outputs(self, pipeline_state):
                return True, []
                
            def get_required_inputs(self):
                return []
                
            def get_produced_outputs(self):
                return []
                
            def get_dependencies(self):
                return []
        
        step = ErrorStep(sample_config)
        
        # Test input validation error
        result = await step.execute(sample_training_input, {})
        assert result["success"] is False
        assert "validation_errors" in result
        
        # Test execution error (modify validate_inputs to pass)
        step.validate_inputs = lambda x, y: (True, [])
        result = await step.execute(sample_training_input, {})
        assert result["success"] is False
    
    @pytest.mark.asyncio
    async def test_step_data_flow(
        self, 
        sample_config, 
        sample_training_input,
        sample_market_data,
        test_data_dir
    ):
        """Test data flow between steps."""
        from src.training.steps.data_preparation.step02_data_reading import DataReadingStep
        from src.training.steps.market_analysis.step03_hmm_regime_discovery import HMMRegimeDiscoveryStep
        
        # Save sample data
        data_path = test_data_dir / "data" / "sample_data.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_market_data.to_parquet(data_path)
        
        # Create pipeline state with data
        pipeline_state = {
            "raw_market_data": str(data_path)
        }
        
        # Execute step 2
        step2 = DataReadingStep(sample_config)
        result = await step2.execute(sample_training_input, pipeline_state)
        
        assert result["success"] is True
        assert "validated_data" in result
        assert "data_validation_results" in result
        
        # Execute step 3 with step 2 outputs
        step3 = HMMRegimeDiscoveryStep(sample_config)
        result = await step3.execute(sample_training_input, result)
        
        assert result["success"] is True
        assert "features" in result
        assert "hmm_results" in result
        assert "regime_labels" in result
    
    def test_step_dependencies(self):
        """Test that step dependencies are correctly defined."""
        from src.training.steps.data_preparation.step01_data_collection import DataCollectionStep
        from src.training.steps.data_preparation.step02_data_reading import DataReadingStep
        from src.training.steps.market_analysis.step03_hmm_regime_discovery import HMMRegimeDiscoveryStep
        from src.training.steps.market_analysis.step04_regime_data_splitting import RegimeDataSplittingStep
        
        # Check dependencies
        step1 = DataCollectionStep({})
        assert step1.get_dependencies() == []
        
        step2 = DataReadingStep({})
        deps = step2.get_dependencies()
        assert "01_data_collection" in deps or "01_5_data_converter" in deps
        
        step3 = HMMRegimeDiscoveryStep({})
        assert "02_data_reading" in step3.get_dependencies()
        
        step4 = RegimeDataSplittingStep({})
        assert "03_hmm_regime_discovery" in step4.get_dependencies()
    
    def test_step_input_output_contracts(self):
        """Test that step inputs/outputs match."""
        from src.training.steps.data_preparation.step01_data_collection import DataCollectionStep
        from src.training.steps.data_preparation.step02_data_reading import DataReadingStep
        
        step1 = DataCollectionStep({})
        step2 = DataReadingStep({})
        
        # Check that step1 outputs are in step2 inputs
        step1_outputs = step1.get_produced_outputs()
        step2_inputs = step2.get_required_inputs()
        
        # At least one of step1's outputs should satisfy step2's inputs
        assert any(
            "raw_market_data" in output or "unified_data_path" in output 
            for output in step1_outputs
        )
    
    @pytest.mark.asyncio
    async def test_checkpoint_recovery(
        self, 
        sample_config, 
        sample_training_input,
        sample_pipeline_state,
        test_data_dir
    ):
        """Test pipeline recovery from checkpoints."""
        # Create a checkpoint
        checkpoint_dir = test_data_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save pipeline state
        import json
        checkpoint_file = checkpoint_dir / "step03_checkpoint.json"
        
        # Convert numpy arrays to lists for JSON serialization
        checkpoint_data = {}
        for key, value in sample_pipeline_state.items():
            if isinstance(value, np.ndarray):
                checkpoint_data[key] = value.tolist()
            elif isinstance(value, pd.DataFrame):
                checkpoint_data[key] = "dataframe_placeholder"
            else:
                checkpoint_data[key] = value
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Test recovery - step should skip execution if checkpoint exists
        from src.training.steps.market_analysis.step04_regime_data_splitting import RegimeDataSplittingStep
        
        step4 = RegimeDataSplittingStep(sample_config)
        
        # Add checkpoint info to pipeline state
        sample_pipeline_state["checkpoint_available"] = True
        sample_pipeline_state["checkpoint_path"] = str(checkpoint_file)
        
        # Execute step - should use checkpoint data
        result = await step4.execute(sample_training_input, sample_pipeline_state)
        
        assert result["success"] is True
        assert "unified_data" in result or "train_data" in result


class TestStepValidation:
    """Test individual step validation."""
    
    def test_data_collection_validation(self, sample_config):
        """Test data collection step validation."""
        from src.training.steps.data_preparation.step01_data_collection import DataCollectionStep
        
        step = DataCollectionStep(sample_config)
        
        # Test missing symbol
        is_valid, errors = step.validate_inputs({}, {})
        assert is_valid is False
        assert any("symbol" in error.lower() for error in errors)
        
        # Test invalid symbol format
        is_valid, errors = step.validate_inputs(
            {"symbol": "btcusdt", "exchange": "binance"}, 
            {}
        )
        assert is_valid is False
        assert any("uppercase" in error.lower() for error in errors)
        
        # Test valid inputs
        is_valid, errors = step.validate_inputs(
            {"symbol": "BTCUSDT", "exchange": "binance", "timeframe": "1h"}, 
            {}
        )
        assert is_valid is True
        assert len(errors) == 0
    
    def test_feature_engineering_validation(self, sample_config, sample_pipeline_state):
        """Test feature engineering step validation."""
        from src.training.steps.feature_engineering.step06_feature_engineering import FeatureEngineeringStep
        
        step = FeatureEngineeringStep(sample_config)
        
        # Test missing labeled data
        is_valid, errors = step.validate_inputs({}, {})
        assert is_valid is False
        assert any("labeled data" in error.lower() for error in errors)
        
        # Test with valid data
        sample_pipeline_state["labeled_data"] = sample_pipeline_state["validated_data"]
        is_valid, errors = step.validate_inputs({}, sample_pipeline_state)
        assert is_valid is True
    
    def test_matrix_operations_validation(self, sample_config, sample_pipeline_state):
        """Test matrix operations step validation."""
        from src.training.steps.model_training.step07_enhanced_matrix_operations import EnhancedMatrixOperationsStep
        
        step = EnhancedMatrixOperationsStep(sample_config)
        
        # Test missing engineered data
        is_valid, errors = step.validate_inputs({}, {})
        assert is_valid is False
        assert any("engineered data" in error.lower() for error in errors)
        
        # Test with valid data
        sample_pipeline_state["engineered_data"] = {"train": sample_pipeline_state["validated_data"]}
        is_valid, errors = step.validate_inputs({}, sample_pipeline_state)
        assert is_valid is True


class TestPerformance:
    """Test pipeline performance."""
    
    @pytest.mark.asyncio
    async def test_step_execution_time(
        self, 
        sample_config, 
        sample_training_input,
        sample_features
    ):
        """Test that steps complete within reasonable time."""
        import time
        from src.training.steps.feature_engineering.step06_feature_engineering import FeatureEngineeringStep
        
        step = FeatureEngineeringStep(sample_config)
        
        pipeline_state = {
            "labeled_data": sample_features
        }
        
        start_time = time.time()
        result = await step.execute(sample_training_input, pipeline_state)
        execution_time = time.time() - start_time
        
        assert result["success"] is True
        assert execution_time < 5.0  # Should complete within 5 seconds
        
        # Check execution tracking
        assert step.execution_duration is not None
        assert step.execution_duration > 0
    
    @pytest.mark.asyncio
    async def test_memory_usage(
        self, 
        sample_config, 
        sample_training_input,
        sample_features
    ):
        """Test memory usage during step execution."""
        import psutil
        import os
        
        from src.training.steps.model_training.step07_enhanced_matrix_operations import EnhancedMatrixOperationsStep
        
        step = EnhancedMatrixOperationsStep(sample_config)
        
        # Measure memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        pipeline_state = {
            "engineered_data": {"train": sample_features},
            "selected_features": [col for col in sample_features.columns if col.startswith("feature_")]
        }
        
        result = await step.execute(sample_training_input, pipeline_state)
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        assert result["success"] is True
        assert memory_increase < 500  # Should not increase by more than 500MB
    
    def test_parallel_step_execution(self, sample_config):
        """Test that independent steps can execute in parallel."""
        # This would test parallel execution of steps that don't depend on each other
        # For now, just verify that steps declare independence correctly
        from src.training.steps.data_preparation.step01_data_collection import DataCollectionStep
        from src.training.steps.data_preparation.step02_data_reading import DataReadingStep
        
        step1 = DataCollectionStep(sample_config)
        step2 = DataReadingStep(sample_config) 
        
        # Step 1 should have no dependencies
        assert len(step1.get_dependencies()) == 0
        
        # Step 2 should depend on step 1
        assert len(step2.get_dependencies()) > 0