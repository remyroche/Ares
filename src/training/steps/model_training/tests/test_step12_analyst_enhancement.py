"""Unit tests for Step 12: Analyst Enhancement."""

import pytest
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from src.training.steps.model_training.step12_analyst_enhancement import (
from copy import copy
import asyncio

    AnalystEnhancementStep,
    AnalystEnhancer,
    FeatureAugmenter,
    ModelOptimizer,
    PerformanceAnalyzer
)


class TestAnalystEnhancer:
    """Test cases for AnalystEnhancer."""
    
    @pytest.fixture
    def enhancer(self):
        """Create enhancer instance."""
        config = {
            "optimization_trials": 5,  # Reduced for testing
            "feature_selection_method": "mutual_info",
            "feature_selection_k": 10,
            "enable_shap": False  # Disable SHAP for faster tests
        }
        return AnalystEnhancer(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(0, 3, n_samples))
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_val, y_val
    
    @pytest.fixture
    def sample_analyst_data(self):
        """Create sample analyst data."""
        return {
            "regime_id": 0,
            "best_model": "lightgbm",
            "best_score": 0.75,
            "models": {
                "lightgbm": {"model": Mock()}
            },
            "feature_importance": {f"feature_{i}": np.random.rand() for i in range(20)}
        }
    
    @pytest.mark.asyncio
    async def test_enhance_analyst(self, enhancer, sample_data, sample_analyst_data):
        """Test analyst enhancement."""
        X_train, y_train, X_val, y_val = sample_data
        
        # Mock model retraining
        with patch.object(enhancer, '_retrain_model', new_callable=AsyncMock) as mock_retrain:
            mock_model = Mock()
            mock_model.predict.return_value = np.random.randint(0, 3, len(y_val))
            mock_retrain.return_value = mock_model
            
            enhanced_data = await enhancer.enhance_analyst(
                sample_analyst_data, X_train, y_train, X_val, y_val
            )
        
        # Check structure
        assert "enhancements" in enhanced_data
        assert "selected_features" in enhanced_data["enhancements"]
        assert "feature_scores" in enhanced_data["enhancements"]
        assert "optimized_params" in enhanced_data["enhancements"]
        assert "performance_metrics" in enhanced_data["enhancements"]
        assert "enhanced_model" in enhanced_data
    
    @pytest.mark.asyncio
    async def test_select_features(self, enhancer, sample_data):
        """Test feature selection."""
        X_train, y_train, X_val, y_val = sample_data
        
        selected_features, feature_scores = await enhancer._select_features(
            X_train, y_train, X_val, y_val
        )
        
        assert isinstance(selected_features, list)
        assert len(selected_features) <= enhancer.feature_selection_k
        assert isinstance(feature_scores, dict)
        assert all(feat in X_train.columns for feat in selected_features)
    
    @pytest.mark.asyncio
    async def test_optimize_hyperparameters(self, enhancer, sample_data):
        """Test hyperparameter optimization."""
        X_train, y_train, X_val, y_val = sample_data
        
        # Test for lightgbm
        params = await enhancer._optimize_hyperparameters(
            "lightgbm", X_train, y_train, X_val, y_val
        )
        
        assert isinstance(params, dict)
        assert "n_estimators" in params
        assert "learning_rate" in params
        assert "max_depth" in params
    
    def test_evaluate_performance(self, enhancer, sample_data):
        """Test performance evaluation."""
        X_train, y_train, X_val, y_val = sample_data
        
        # Create mock model
        model = Mock()
        model.predict.return_value = np.random.randint(0, 3, len(y_train))
        model.predict_proba.return_value = np.random.rand(len(y_val), 3)
        
        metrics = enhancer._evaluate_performance(
            model, X_train, y_train, X_val, y_val
        )
        
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
        assert "overfitting_score" in metrics
        assert 0 <= metrics["train_accuracy"] <= 1
        assert 0 <= metrics["val_accuracy"] <= 1


class TestFeatureAugmenter:
    """Test cases for FeatureAugmenter."""
    
    @pytest.fixture
    def augmenter(self):
        """Create augmenter instance."""
        config = {
            "create_feature_interactions": True,
            "create_polynomial_features": True,
            "polynomial_degree": 2,
            "interaction_threshold": 0.8
        }
        return FeatureAugmenter(config)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features."""
        np.random.seed(42)
        return pd.DataFrame({
            "feature_0": np.random.randn(100),
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
            "feature_4": np.random.randn(100)
        })
    
    def test_augment_features(self, augmenter, sample_features):
        """Test feature augmentation."""
        feature_importance = {col: np.random.rand() for col in sample_features.columns}
        
        augmented = augmenter.augment_features(
            sample_features, feature_importance, top_k=3
        )
        
        # Check that features were added
        assert len(augmented.columns) > len(sample_features.columns)
        
        # Check that original features are preserved
        for col in sample_features.columns:
            assert col in augmented.columns
        
        # Check for interaction features
        interaction_features = [col for col in augmented.columns if "_X_" in col]
        assert len(interaction_features) > 0
        
        # Check for polynomial features
        poly_features = [col for col in augmented.columns if "_pow" in col]
        assert len(poly_features) > 0
    
    def test_select_augmented_features(self, augmenter, sample_features):
        """Test augmented feature selection."""
        # Create augmented features
        feature_importance = {col: np.random.rand() for col in sample_features.columns}
        augmented = augmenter.augment_features(sample_features, feature_importance)
        
        # Create target
        y = pd.Series(np.random.randint(0, 2, len(augmented)))
        
        # Select features
        selected = augmenter.select_augmented_features(
            augmented, y, len(sample_features.columns), selection_ratio=0.5
        )
        
        assert isinstance(selected, list)
        # All original features should be kept
        for col in sample_features.columns:
            assert col in selected
        
        # Some but not all augmented features should be selected
        augmented_only = [col for col in augmented.columns if col not in sample_features.columns]
        selected_augmented = [col for col in selected if col not in sample_features.columns]
        assert 0 < len(selected_augmented) < len(augmented_only)


class TestModelOptimizer:
    """Test cases for ModelOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        config = {
            "enable_pruning": True,
            "enable_quantization": False,
            "pruning_ratio": 0.2
        }
        return ModelOptimizer(config)
    
    def test_optimize_random_forest(self, optimizer):
        """Test Random Forest optimization."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create a model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Fit with dummy data to create estimators
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        
        # Optimize
        optimized_model, metrics = optimizer.optimize_model(model, "random_forest")
        
        # Check that pruning was applied
        assert "optimizations_applied" in metrics
        assert "tree_pruning" in metrics["optimizations_applied"]
        assert optimized_model.n_estimators < 100  # Trees were pruned
        assert metrics["size_reduction"] > 0
    
    def test_get_model_size(self, optimizer):
        """Test model size calculation."""
        model = {"test": "model", "data": np.random.randn(100, 10)}
        size = optimizer._get_model_size(model)
        assert isinstance(size, int)
        assert size > 0


class TestPerformanceAnalyzer:
    """Test cases for PerformanceAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        config = {}
        return PerformanceAnalyzer(config)
    
    def test_analyze_enhancement_impact(self, analyzer):
        """Test enhancement impact analysis."""
        original_performance = {
            "val_accuracy": 0.75,
            "val_auc": 0.80,
            "overfitting_score": 0.05
        }
        
        enhanced_performance = {
            "val_accuracy": 0.78,
            "val_auc": 0.83,
            "overfitting_score": 0.03
        }
        
        impact = analyzer.analyze_enhancement_impact(
            original_performance, enhanced_performance
        )
        
        assert "val_accuracy_improvement" in impact
        assert "val_accuracy_absolute_gain" in impact
        assert "overfitting_reduction" in impact
        assert "overall_assessment" in impact
        
        # Check calculations
        assert impact["val_accuracy_absolute_gain"] == pytest.approx(0.03)
        assert impact["overfitting_reduction"] == pytest.approx(0.02)
        assert impact["overall_assessment"] == "significant_improvement"
    
    def test_create_performance_report(self, analyzer):
        """Test performance report creation."""
        original_data = {
            "regime_id": 0,
            "best_model": "lightgbm",
            "best_score": 0.75,
            "feature_importance": {f"feature_{i}": 0.1 for i in range(10)}
        }
        
        enhanced_data = {
            "regime_id": 0,
            "best_model": "lightgbm",
            "enhancements": {
                "selected_features": ["feature_0", "feature_1", "feature_2"],
                "optimized_params": {"n_estimators": 200},
                "performance_metrics": {
                    "val_accuracy": 0.78,
                    "val_auc": 0.83
                }
            }
        }
        
        report = analyzer.create_performance_report("regime_0", original_data, enhanced_data)
        
        assert "regime_id" in report
        assert "timestamp" in report
        assert "original_model" in report
        assert "enhancements_applied" in report
        assert "enhanced_performance" in report
        assert "enhancement_impact" in report


class TestAnalystEnhancementStep:
    """Test cases for AnalystEnhancementStep."""
    
    @pytest.fixture
    def step(self):
        """Create step instance."""
        config = {
            "optimization_trials": 2,  # Very low for fast tests
            "feature_selection_k": 5,
            "enable_shap": False,
            "enhancement": {
                "parallel_processing": False,  # Disable for simpler testing
                "max_parallel_regimes": 2
            },
            "artifacts_dir": "test_artifacts"
        }
        return AnalystEnhancementStep(config)
    
    @pytest.fixture
    def valid_pipeline_state(self):
        """Create valid pipeline state."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        # Create mock analysts
        regime_analysts = {
            "regime_0": {
                "regime_id": 0,
                "best_model": "lightgbm",
                "best_score": 0.75,
                "feature_importance": {f"feature_{i}": np.random.rand() for i in range(n_features)}
            },
            "regime_1": {
                "regime_id": 1,
                "best_model": "xgboost",
                "best_score": 0.72,
                "feature_importance": {f"feature_{i}": np.random.rand() for i in range(n_features)}
            }
        }
        
        # Create features and labels
        regime_features = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        
        regime_labels = pd.Series(np.random.randint(0, 2, n_samples))
        
        return {
            "regime_analysts": regime_analysts,
            "regime_features": regime_features,
            "regime_labels": regime_labels,
            "num_regimes": 2
        }
    
    def test_initialization(self, step):
        """Test step initialization."""
        assert step.step_number == "12"
        assert step.step_name == "analyst_enhancement"
        assert step.analyst_enhancer is not None
        assert step.feature_augmenter is not None
        assert step.model_optimizer is not None
        assert step.performance_analyzer is not None
    
    def test_get_required_inputs(self, step):
        """Test required inputs."""
        inputs = step.get_required_inputs()
        assert "regime_analysts" in inputs
        assert "regime_features" in inputs
        assert "regime_labels" in inputs
        assert "num_regimes" in inputs
    
    def test_get_produced_outputs(self, step):
        """Test produced outputs."""
        outputs = step.get_produced_outputs()
        assert "enhanced_analysts" in outputs
        assert "enhancement_reports" in outputs
        assert "performance_comparison" in outputs
        assert "optimization_summary" in outputs
    
    def test_validate_inputs_valid(self, step, valid_pipeline_state):
        """Test input validation with valid inputs."""
        is_valid, errors = step.validate_inputs({}, valid_pipeline_state)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_inputs_missing(self, step):
        """Test input validation with missing inputs."""
        incomplete_state = {"regime_analysts": {}}
        is_valid, errors = step.validate_inputs({}, incomplete_state)
        assert not is_valid
        assert any("regime_features" in error for error in errors)
    
    def test_validate_inputs_empty_analysts(self, step, valid_pipeline_state):
        """Test input validation with empty analysts."""
        invalid_state = valid_pipeline_state.copy()
        invalid_state["regime_analysts"] = {}
        
        is_valid, errors = step.validate_inputs({}, invalid_state)
        assert not is_valid
        assert any("No regime analysts" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_execute_logic(self, step, valid_pipeline_state):
        """Test execution logic."""
        training_input = {}
        
        # Mock the enhancement process
        with patch.object(step, '_enhance_regime_analyst', new_callable=AsyncMock) as mock_enhance:
            # Mock successful enhancement
            enhanced_data = {
                "regime_id": 0,
                "best_model": "lightgbm",
                "enhanced_model": Mock(),
                "enhancements": {
                    "performance_metrics": {"val_accuracy": 0.80}
                }
            }
            report = {
                "regime_id": "regime_0",
                "enhanced_performance": {"val_accuracy": 0.80},
                "enhancement_impact": {"val_accuracy_absolute_gain": 0.05}
            }
            mock_enhance.return_value = ("regime_0", enhanced_data, report)
            
            # Mock artifact saving
            with patch.object(step, '_save_artifacts', new_callable=AsyncMock):
                result = await step.execute_logic(training_input, valid_pipeline_state)
        
        # Check outputs
        assert "enhanced_analysts" in result
        assert "enhancement_reports" in result
        assert "performance_comparison" in result
        assert "optimization_summary" in result
    
    def test_create_performance_comparison(self, step):
        """Test performance comparison creation."""
        original_analysts = {
            "regime_0": {"best_score": 0.75},
            "regime_1": {"best_score": 0.72}
        }
        
        enhanced_analysts = {
            "regime_0": {"enhanced_model": Mock()},
            "regime_1": {"enhanced_model": Mock()}
        }
        
        enhancement_reports = {
            "regime_0": {
                "enhanced_performance": {"val_accuracy": 0.80},
                "enhancement_impact": {
                    "val_accuracy_absolute_gain": 0.05,
                    "overall_assessment": "significant_improvement"
                }
            },
            "regime_1": {
                "enhanced_performance": {"val_accuracy": 0.73},
                "enhancement_impact": {
                    "val_accuracy_absolute_gain": 0.01,
                    "overall_assessment": "moderate_improvement"
                }
            }
        }
        
        comparison = step._create_performance_comparison(
            original_analysts, enhanced_analysts, enhancement_reports
        )
        
        assert "summary" in comparison
        assert "regime_comparisons" in comparison
        assert comparison["summary"]["enhanced_regimes"] == 2
        assert comparison["summary"]["average_improvement"] == pytest.approx(0.03)
        assert comparison["summary"]["best_improvement"]["regime"] == "regime_0"
    
    def test_create_optimization_summary(self, step):
        """Test optimization summary creation."""
        enhancement_reports = {
            "regime_0": {
                "enhancements_applied": ["feature_selection", "hyperparameter_optimization"]
            },
            "regime_1": {
                "enhancements_applied": ["feature_selection", "shap_analysis"]
            }
        }
        
        performance_comparison = {
            "summary": {
                "average_improvement": 0.03,
                "enhanced_regimes": 2
            },
            "regime_comparisons": {}
        }
        
        summary = step._create_optimization_summary(
            enhancement_reports, performance_comparison
        )
        
        assert "timestamp" in summary
        assert "total_regimes_processed" in summary
        assert "enhancements_applied" in summary
        assert "recommendations" in summary
        
        # Check enhancement counts
        assert summary["enhancements_applied"]["feature_selection"] == 2
        assert summary["enhancements_applied"]["hyperparameter_optimization"] == 1
    
    def test_validate_outputs_valid(self, step):
        """Test output validation with valid outputs."""
        valid_outputs = {
            "enhanced_analysts": {
                "regime_0": {"enhanced_model": Mock()}
            },
            "enhancement_reports": {"regime_0": {}},
            "performance_comparison": {"summary": {}},
            "optimization_summary": {}
        }
        
        is_valid, errors = step.validate_outputs(valid_outputs)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_outputs_missing(self, step):
        """Test output validation with missing outputs."""
        incomplete_outputs = {"enhanced_analysts": {}}
        
        is_valid, errors = step.validate_outputs(incomplete_outputs)
        assert not is_valid
        assert any("enhancement_reports" in error for error in errors)
    
    def test_validate_outputs_no_models(self, step):
        """Test output validation with no enhanced models."""
        outputs = {
            "enhanced_analysts": {"regime_0": {}},  # No enhanced_model
            "enhancement_reports": {},
            "performance_comparison": {"summary": {}},
            "optimization_summary": {}
        }
        
        is_valid, errors = step.validate_outputs(outputs)
        assert not is_valid
        assert any("No enhanced models" in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])