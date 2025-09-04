"""Unit tests for Step 11: Analyst Creation."""

import pytest
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from src.training.steps.model_training.step11_analyst_creation import (
    AnalystCreationStep,
    AnalystModelBuilder,
    MultiOutputAnalystBuilder
)
from copy import copy
import asyncio


class TestAnalystModelBuilder:
    """Test cases for AnalystModelBuilder."""
    
    @pytest.fixture
    def builder(self):
        """Create builder instance."""
        config = {
            "model_types": ["lightgbm", "xgboost", "random_forest"],
            "optimization_trials": 5,  # Reduced for testing
            "cv_folds": 3
        }
        return AnalystModelBuilder(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(0, 3, n_samples))
        
        return X, y
    
    def test_build_regime_analyst(self, builder, sample_data):
        """Test building analyst for a regime."""
        X, y = sample_data
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build analyst
        result = builder.build_regime_analyst(0, X_train, y_train, X_val, y_val)
        
        # Check structure
        assert "regime_id" in result
        assert result["regime_id"] == 0
        assert "models" in result
        assert "best_model" in result
        assert "best_score" in result
        assert "feature_importance" in result
        
        # Check models were trained
        assert len(result["models"]) > 0
        assert result["best_model"] in result["models"]
        assert result["best_score"] > 0
    
    @patch('lightgbm.train')
    def test_train_lightgbm(self, mock_lgb_train, builder, sample_data):
        """Test LightGBM training."""
        X, y = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.rand(len(X_val), 3)
        mock_model.best_iteration = 10
        mock_model.feature_importance.return_value = np.random.rand(X.shape[1])
        mock_lgb_train.return_value = mock_model
        
        # Train model
        result = builder._train_lightgbm(X_train, y_train, X_val, y_val)
        
        # Check result
        assert "model" in result
        assert "best_params" in result
        assert "validation_score" in result
        assert "feature_importance" in result
        assert isinstance(result["feature_importance"], dict)
    
    @patch('xgboost.XGBClassifier')
    def test_train_xgboost(self, mock_xgb, builder, sample_data):
        """Test XGBoost training."""
        X, y = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.randint(0, 3, len(X_val))
        mock_model.feature_importances_ = np.random.rand(X.shape[1])
        mock_xgb.return_value = mock_model
        
        # Train model
        result = builder._train_xgboost(X_train, y_train, X_val, y_val)
        
        # Check result
        assert "model" in result
        assert "best_params" in result
        assert "validation_score" in result
        assert "feature_importance" in result
    
    def test_train_random_forest(self, builder, sample_data):
        """Test Random Forest training."""
        X, y = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        
        # Train model (no mocking needed for RF)
        result = builder._train_random_forest(X_train, y_train, X_val, y_val)
        
        # Check result
        assert "model" in result
        assert "best_params" in result
        assert "validation_score" in result
        assert "feature_importance" in result
        assert result["validation_score"] >= 0


class TestMultiOutputAnalystBuilder:
    """Test cases for MultiOutputAnalystBuilder."""
    
    @pytest.fixture
    def builder(self):
        """Create builder instance."""
        config = {
            "model_types": ["random_forest"],  # Single model for faster testing
            "optimization_trials": 3,
            "cv_folds": 2
        }
        return MultiOutputAnalystBuilder(config)
    
    @pytest.fixture
    def sample_multi_output_data(self):
        """Create sample multi-output training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        
        y_dict = {
            "direction": pd.Series(np.random.randint(0, 2, n_samples)),
            "magnitude": pd.Series(np.random.randint(0, 3, n_samples)),
            "confidence": pd.Series(np.random.randint(0, 4, n_samples))
        }
        
        return X, y_dict
    
    def test_build_multi_output_analyst(self, builder, sample_multi_output_data):
        """Test building multi-output analyst."""
        X, y_dict = sample_multi_output_data
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train_dict = {k: v[:split_idx] for k, v in y_dict.items()}
        y_val_dict = {k: v[split_idx:] for k, v in y_dict.items()}
        
        # Build analyst
        result = builder.build_multi_output_analyst(
            0, X_train, y_train_dict, X_val, y_val_dict
        )
        
        # Check structure
        assert "regime_id" in result
        assert result["regime_id"] == 0
        assert "output_models" in result
        assert "aggregated_metrics" in result
        assert "feature_importance" in result
        
        # Check each output has a model
        for output_name in y_dict.keys():
            assert output_name in result["output_models"]
            assert "best_model" in result["output_models"][output_name]
    
    def test_calculate_aggregated_metrics(self, builder):
        """Test aggregated metrics calculation."""
        output_models = {
            "output1": {"best_score": 0.8},
            "output2": {"best_score": 0.7},
            "output3": {"best_score": 0.9}
        }
        
        metrics = builder._calculate_aggregated_metrics(output_models)
        
        assert "avg_validation_score" in metrics
        assert metrics["avg_validation_score"] == pytest.approx(0.8, rel=1e-3)
        assert metrics["min_validation_score"] == 0.7
        assert metrics["max_validation_score"] == 0.9
        assert len(metrics["output_scores"]) == 3


class TestAnalystCreationStep:
    """Test cases for AnalystCreationStep."""
    
    @pytest.fixture
    def step(self):
        """Create step instance."""
        config = {
            "model_types": ["random_forest"],  # Fast model for testing
            "optimization_trials": 2,
            "cv_folds": 2,
            "use_multi_output": False,
            "validation_split": 0.2,
            "random_state": 42,
            "artifacts_dir": "test_artifacts"
        }
        return AnalystCreationStep(config)
    
    @pytest.fixture
    def valid_pipeline_state(self):
        """Create valid pipeline state."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        n_regimes = 3
        
        # Create features
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='5min')
        regime_features = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
            index=dates
        )
        
        # Create labels with balanced regimes
        regime_labels = pd.Series(
            np.repeat(np.arange(n_regimes), n_samples // n_regimes + 1)[:n_samples]
        )
        np.random.shuffle(regime_labels.values)
        
        return {
            "regime_features": regime_features,
            "regime_labels": regime_labels,
            "num_regimes": n_regimes
        }
    
    @pytest.fixture
    def valid_multi_output_state(self, valid_pipeline_state):
        """Create valid multi-output pipeline state."""
        state = valid_pipeline_state.copy()
        n_samples = len(state["regime_features"])
        
        # Replace single labels with multi-output labels
        state["regime_labels"] = {
            "direction": pd.Series(np.random.randint(0, 2, n_samples)),
            "magnitude": pd.Series(np.random.randint(0, 3, n_samples))
        }
        
        return state
    
    def test_initialization(self, step):
        """Test step initialization."""
        assert step.step_number == "11"
        assert step.step_name == "analyst_creation"
        assert step.model_builder is not None
        assert step.multi_output_builder is not None
    
    def test_get_required_inputs(self, step):
        """Test required inputs."""
        inputs = step.get_required_inputs()
        assert "regime_features" in inputs
        assert "regime_labels" in inputs
        assert "num_regimes" in inputs
    
    def test_get_produced_outputs(self, step):
        """Test produced outputs."""
        outputs = step.get_produced_outputs()
        assert "regime_analysts" in outputs
        assert "analyst_metadata" in outputs
        assert "feature_importance" in outputs
        assert "analyst_performance" in outputs
    
    def test_validate_inputs_valid(self, step, valid_pipeline_state):
        """Test input validation with valid inputs."""
        is_valid, errors = step.validate_inputs({}, valid_pipeline_state)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_inputs_missing(self, step):
        """Test input validation with missing inputs."""
        incomplete_state = {"regime_features": pd.DataFrame()}
        is_valid, errors = step.validate_inputs({}, incomplete_state)
        assert not is_valid
        assert any("regime_labels" in error for error in errors)
    
    def test_validate_inputs_wrong_type(self, step, valid_pipeline_state):
        """Test input validation with wrong types."""
        invalid_state = valid_pipeline_state.copy()
        invalid_state["regime_features"] = "not a dataframe"
        
        is_valid, errors = step.validate_inputs({}, invalid_state)
        assert not is_valid
        assert any("DataFrame" in error for error in errors)
    
    def test_validate_inputs_multi_output(self, step, valid_multi_output_state):
        """Test input validation for multi-output mode."""
        step.use_multi_output = True
        is_valid, errors = step.validate_inputs({}, valid_multi_output_state)
        assert is_valid
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_execute_logic(self, step, valid_pipeline_state):
        """Test execution logic."""
        training_input = {}
        
        # Mock artifact saving
        with patch.object(step, '_save_artifacts', new_callable=AsyncMock):
            result = await step.execute_logic(training_input, valid_pipeline_state)
        
        # Check outputs
        assert "regime_analysts" in result
        assert "analyst_metadata" in result
        assert "feature_importance" in result
        assert "analyst_performance" in result
        
        # Check regime analysts were created
        assert len(result["regime_analysts"]) > 0
        
        # Check performance metrics
        assert "overall" in result["analyst_performance"]
        assert "mean_score" in result["analyst_performance"]["overall"]
    
    @pytest.mark.asyncio
    async def test_execute_logic_multi_output(self, step, valid_multi_output_state):
        """Test execution logic with multi-output."""
        step.use_multi_output = True
        training_input = {}
        
        # Mock artifact saving
        with patch.object(step, '_save_artifacts', new_callable=AsyncMock):
            result = await step.execute_logic(training_input, valid_multi_output_state)
        
        # Check outputs
        assert "regime_analysts" in result
        
        # Check multi-output structure
        for regime_key, analyst_data in result["regime_analysts"].items():
            if "output_models" in analyst_data:  # Multi-output structure
                assert isinstance(analyst_data["output_models"], dict)
    
    def test_calculate_overall_metrics(self, step):
        """Test overall metrics calculation."""
        analyst_performance = {
            "regime_0": {"validation_score": 0.85},
            "regime_1": {"validation_score": 0.80},
            "regime_2": {"validation_score": 0.90}
        }
        
        metrics = step._calculate_overall_metrics(analyst_performance)
        
        assert "mean_score" in metrics
        assert metrics["mean_score"] == pytest.approx(0.85, rel=1e-3)
        assert metrics["std_score"] > 0
        assert metrics["min_score"] == 0.80
        assert metrics["max_score"] == 0.90
        assert metrics["num_regimes"] == 3
    
    def test_validate_outputs_valid(self, step):
        """Test output validation with valid outputs."""
        valid_outputs = {
            "regime_analysts": {
                "regime_0": {"models": {"rf": {"model": Mock()}}}
            },
            "analyst_metadata": {"regime_0": {"training_samples": 100}},
            "analyst_performance": {
                "regime_0": {"validation_score": 0.8},
                "overall": {"mean_score": 0.8}
            }
        }
        
        is_valid, errors = step.validate_outputs(valid_outputs)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_outputs_missing(self, step):
        """Test output validation with missing outputs."""
        incomplete_outputs = {"regime_analysts": {}}
        
        is_valid, errors = step.validate_outputs(incomplete_outputs)
        assert not is_valid
        assert any("analyst_metadata" in error for error in errors)
    
    def test_validate_outputs_empty_analysts(self, step):
        """Test output validation with empty analysts."""
        outputs = {
            "regime_analysts": {},
            "analyst_metadata": {},
            "analyst_performance": {}
        }
        
        is_valid, errors = step.validate_outputs(outputs)
        assert not is_valid
        assert any("No regime analysts" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_save_artifacts(self, step, tmp_path):
        """Test artifact saving."""
        # Set temporary artifacts directory
        step.config["artifacts_dir"] = str(tmp_path)
        
        # Create sample results
        result = {
            "regime_analysts": {
                "regime_0": {
                    "regime_id": 0,
                    "best_model": "random_forest",
                    "best_score": 0.85,
                    "feature_importance": {"feature_0": 0.5, "feature_1": 0.3},
                    "models": {
                        "random_forest": {
                            "model": Mock()  # Mock model
                        }
                    }
                }
            },
            "analyst_metadata": {
                "regime_0": {"training_samples": 100}
            },
            "analyst_performance": {
                "regime_0": {"validation_score": 0.85},
                "overall": {"mean_score": 0.85}
            }
        }
        
        # Save artifacts
        await step._save_artifacts(result)
        
        # Check files were created
        artifacts_dir = tmp_path / step.full_step_name
        assert artifacts_dir.exists()
        assert (artifacts_dir / "analyst_metadata.json").exists()
        assert (artifacts_dir / "analyst_performance.json").exists()
        assert (artifacts_dir / "regime_analysts" / "regime_0").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])