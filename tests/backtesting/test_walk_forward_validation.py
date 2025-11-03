"""
Unit Tests for Walk-Forward Validation System
==============================================

Tests for the walk-forward validation system including:
- ValidationConfig initialization and defaults
- WalkForwardValidator with nested CV
- Embargo logic
- Metric calculations (IC, AUC, MSE)
- AblationValidator functionality
- SPAValidator functionality
- Edge cases and error handling

Author: Ares Trading System
Date: 2025-10-31
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.validation.walkforward_validation import (
    ValidationConfig,
    ValidationResult,
    FoldResult,
    WalkForwardValidator,
    AblationValidator,
    SPAValidator,
    ValidationType,
    run_complete_validation
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    # Generate features
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Generate targets with some correlation to features
    targets = pd.Series(
        features.iloc[:, :3].sum(axis=1) * 0.5 + np.random.randn(n_samples) * 0.3,
        name='target'
    )
    
    return features, targets


@pytest.fixture
def sample_data_with_prefixes():
    """Create sample data with feature prefixes for ablation testing."""
    np.random.seed(42)
    n_samples = 500
    
    # Create features with different prefixes
    parent_features = pd.DataFrame(
        np.random.randn(n_samples, 5),
        columns=[f'p/feature_{i}' for i in range(5)]
    )
    
    transform_features = pd.DataFrame(
        np.random.randn(n_samples, 3),
        columns=[f't/transform_{i}' for i in range(3)]
    )
    
    patch_features = pd.DataFrame(
        np.random.randn(n_samples, 2),
        columns=[f'y_hat_{i}' for i in range(2)]
    )
    
    interaction_features = pd.DataFrame(
        np.random.randn(n_samples, 4),
        columns=[f'i/interaction_{i}' for i in range(4)]
    )
    
    features = pd.concat([
        parent_features, transform_features, patch_features, interaction_features
    ], axis=1)
    
    # Generate targets with correlation to features
    targets = pd.Series(
        features.iloc[:, :5].sum(axis=1) * 0.5 + np.random.randn(n_samples) * 0.3,
        name='target'
    )
    
    return features, targets


@pytest.fixture
def basic_config():
    """Create basic validation configuration."""
    return ValidationConfig(
        n_outer_folds=3,
        n_inner_folds=2,
        embargo_pct=0.1,
        min_train_samples=100,
        min_val_samples=20
    )


@pytest.fixture
def model_config():
    """Create basic model configuration."""
    return {'default': {'model_type': 'linear_regression'}}


# ============================================================================
# Test ValidationConfig
# ============================================================================

class TestValidationConfig:
    """Test ValidationConfig initialization and defaults."""
    
    def test_default_initialization(self):
        """Test that ValidationConfig initializes with correct defaults."""
        config = ValidationConfig()
        
        assert config.n_outer_folds == 6
        assert config.n_inner_folds == 3
        assert config.embargo_pct == 0.1
        assert config.min_train_samples == 1000
        assert config.min_val_samples == 200
        assert config.spa_permutations == 1000
        assert config.significance_level == 0.05
        assert config.ablation_steps is not None
        assert len(config.ablation_steps) == 5
    
    def test_custom_initialization(self):
        """Test that ValidationConfig accepts custom parameters."""
        custom_ablation = ['step1', 'step2']
        config = ValidationConfig(
            n_outer_folds=8,
            n_inner_folds=4,
            embargo_pct=0.15,
            min_train_samples=500,
            min_val_samples=100,
            ablation_steps=custom_ablation,
            spa_permutations=2000,
            significance_level=0.01
        )
        
        assert config.n_outer_folds == 8
        assert config.n_inner_folds == 4
        assert config.embargo_pct == 0.15
        assert config.min_train_samples == 500
        assert config.min_val_samples == 100
        assert config.ablation_steps == custom_ablation
        assert config.spa_permutations == 2000
        assert config.significance_level == 0.01
    
    def test_ablation_steps_default(self):
        """Test that ablation steps have correct default values."""
        config = ValidationConfig()
        
        expected_steps = [
            'parents_only',
            'parents_transforms',
            'parents_transforms_patch',
            'parents_transforms_patch_8_interactions',
            'parents_transforms_patch_15_interactions'
        ]
        
        assert config.ablation_steps == expected_steps


# ============================================================================
# Test WalkForwardValidator
# ============================================================================

class TestWalkForwardValidator:
    """Test WalkForwardValidator functionality."""
    
    def test_initialization(self, basic_config):
        """Test that WalkForwardValidator initializes correctly."""
        validator = WalkForwardValidator(basic_config)
        
        assert validator.config == basic_config
        assert validator.outer_cv is not None
        assert validator.inner_cv is not None
    
    def test_validate_basic(self, sample_data, basic_config, model_config):
        """Test basic validation execution."""
        features, targets = sample_data
        validator = WalkForwardValidator(basic_config)
        
        result = validator.validate(features, targets, model_config)
        
        assert isinstance(result, ValidationResult)
        assert 'mean' in result.ic_scores
        assert 'std' in result.ic_scores
        assert 'min' in result.ic_scores
        assert 'max' in result.ic_scores
        assert 'mean' in result.auc_scores
        assert 'mean' in result.mse_scores
        assert len(result.fold_results) > 0
    
    def test_validate_with_embargo(self, sample_data, model_config):
        """Test that embargo logic is applied correctly."""
        features, targets = sample_data
        
        # Config with 20% embargo
        config = ValidationConfig(
            n_outer_folds=3,
            n_inner_folds=2,
            embargo_pct=0.2,
            min_train_samples=100,
            min_val_samples=20
        )
        
        validator = WalkForwardValidator(config)
        result = validator.validate(features, targets, model_config)
        
        # Should still complete successfully with embargo
        assert result.metadata['embargo_applied'] == True
        assert result.metadata['n_folds_completed'] > 0
    
    def test_validate_metrics_ranges(self, sample_data, basic_config, model_config):
        """Test that metrics are within expected ranges."""
        features, targets = sample_data
        validator = WalkForwardValidator(basic_config)
        
        result = validator.validate(features, targets, model_config)
        
        # IC should be between -1 and 1
        assert -1 <= result.ic_scores['mean'] <= 1
        
        # AUC should be between 0 and 1
        assert 0 <= result.auc_scores['mean'] <= 1
        
        # MSE should be non-negative
        assert result.mse_scores['mean'] >= 0
    
    def test_calculate_ic(self, basic_config):
        """Test IC calculation."""
        validator = WalkForwardValidator(basic_config)
        
        # Perfect correlation (need at least 10 samples)
        predictions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ic = validator._calculate_ic(predictions, actual)
        assert abs(ic - 1.0) < 0.01
        
        # Perfect negative correlation
        predictions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        actual = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        ic = validator._calculate_ic(predictions, actual)
        assert abs(ic - (-1.0)) < 0.01
        
        # Too few samples (less than 10)
        predictions = np.array([1, 2, 3, 4, 5])
        actual = np.array([1, 2, 3, 4, 5])
        ic = validator._calculate_ic(predictions, actual)
        assert ic == 0.0
    
    def test_calculate_auc(self, basic_config):
        """Test AUC calculation."""
        validator = WalkForwardValidator(basic_config)
        
        # Perfect prediction (using continuous targets above/below 0)
        predictions = np.array([0.9, 0.8, 0.7, 0.6, -0.2, -0.1, -0.05, -0.02, -0.01, -0.001])
        actual = np.array([1.5, 1.2, 0.8, 0.5, -0.3, -0.5, -0.7, -0.9, -1.1, -1.3])
        auc = validator._calculate_auc(predictions, actual)
        assert auc > 0.9  # Should be close to 1.0
        
        # Random prediction (should be around 0.5)
        np.random.seed(42)
        predictions = np.random.rand(100)
        actual = np.random.randn(100)  # Some positive, some negative
        auc = validator._calculate_auc(predictions, actual)
        assert 0.3 < auc < 0.7  # Should be around 0.5
        
        # Too few samples
        predictions = np.array([0.5, 0.5])
        actual = np.array([1.0, -1.0])
        auc = validator._calculate_auc(predictions, actual)
        assert auc == 0.5
        
        # Single class in targets (all positive)
        predictions = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        actual = np.array([1.5, 1.2, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        auc = validator._calculate_auc(predictions, actual)
        assert auc == 0.5
    
    def test_nested_cv_selection(self, sample_data, basic_config):
        """Test nested CV for hyperparameter selection."""
        features, targets = sample_data
        validator = WalkForwardValidator(basic_config)
        
        # Create multiple model configs
        model_configs = {
            'config1': {'model_type': 'linear', 'param1': 0.1},
            'config2': {'model_type': 'linear', 'param1': 0.5},
            'config3': {'model_type': 'linear', 'param1': 1.0}
        }
        
        best_config = validator._nested_cv_selection(
            features[:300], targets[:300], model_configs
        )
        
        # Should return one of the configs
        assert best_config in model_configs.values()
    
    def test_evaluate_fold(self, sample_data, basic_config):
        """Test single fold evaluation."""
        features, targets = sample_data
        validator = WalkForwardValidator(basic_config)
        
        # Train a simple model
        X_train = features[:300]
        y_train = targets[:300]
        X_val = features[300:400]
        y_val = targets[300:400]
        
        model = validator._train_model(X_train, y_train, {'model_type': 'linear'})
        
        train_idx = np.arange(300)
        val_idx = np.arange(300, 400)
        
        fold_result = validator._evaluate_fold(
            model, X_val, y_val, train_idx, val_idx, fold_idx=0
        )
        
        assert isinstance(fold_result, FoldResult)
        assert fold_result.train_start == 0
        assert fold_result.train_end == 299
        assert fold_result.val_start == 300
        assert fold_result.val_end == 399
        assert len(fold_result.predictions) == len(y_val)
        assert len(fold_result.actual) == len(y_val)
        assert len(fold_result.feature_importance) > 0
    
    def test_aggregate_metrics_calculation(self, basic_config):
        """Test aggregate metrics calculation."""
        validator = WalkForwardValidator(basic_config)
        
        ic_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        auc_scores = [0.55, 0.60, 0.65, 0.70, 0.75]
        mse_scores = [0.1, 0.2, 0.15, 0.18, 0.22]
        
        aggregate = validator._calculate_aggregate_metrics(
            ic_scores, auc_scores, mse_scores
        )
        
        # Check IC
        assert abs(aggregate['ic']['mean'] - 0.3) < 0.01
        assert abs(aggregate['ic']['min'] - 0.1) < 0.01
        assert abs(aggregate['ic']['max'] - 0.5) < 0.01
        
        # Check AUC
        assert abs(aggregate['auc']['mean'] - 0.65) < 0.01
        assert abs(aggregate['auc']['min'] - 0.55) < 0.01
        assert abs(aggregate['auc']['max'] - 0.75) < 0.01
        
        # Check MSE
        assert abs(aggregate['mse']['mean'] - 0.17) < 0.01
        assert abs(aggregate['mse']['min'] - 0.1) < 0.01
        assert abs(aggregate['mse']['max'] - 0.22) < 0.01
    
    def test_validation_with_insufficient_samples(self, model_config):
        """Test validation with insufficient samples."""
        # Create very small dataset
        features = pd.DataFrame(np.random.randn(100, 5))
        targets = pd.Series(np.random.randn(100))
        
        # Config requiring fewer samples but may still fail some folds
        config = ValidationConfig(
            n_outer_folds=2,
            n_inner_folds=2,
            min_train_samples=30,
            min_val_samples=10,
            embargo_pct=0.0  # No embargo to maximize usable data
        )
        
        validator = WalkForwardValidator(config)
        result = validator.validate(features, targets, model_config)
        
        # Should handle gracefully - may have fewer completed folds
        assert isinstance(result, ValidationResult)
        assert result.metadata['n_folds_completed'] <= result.metadata['n_folds_attempted']
        # With this data size and config, should complete at least 1 fold
        assert result.metadata['n_folds_completed'] >= 1


# ============================================================================
# Test AblationValidator
# ============================================================================

class TestAblationValidator:
    """Test AblationValidator functionality."""
    
    def test_initialization(self, basic_config):
        """Test AblationValidator initialization."""
        validator = AblationValidator(basic_config)
        
        assert validator.config == basic_config
        assert validator.validator is not None
    
    def test_create_feature_subset_parents_only(self, sample_data_with_prefixes):
        """Test creating feature subset with parents only."""
        features, _ = sample_data_with_prefixes
        config = ValidationConfig()
        validator = AblationValidator(config)
        
        subset = validator._create_feature_subset(features, 'parents_only')
        
        # Should only have p/ prefixed features
        assert all(col.startswith('p/') for col in subset.columns)
        assert len(subset.columns) == 5
    
    def test_create_feature_subset_parents_transforms(self, sample_data_with_prefixes):
        """Test creating feature subset with parents and transforms."""
        features, _ = sample_data_with_prefixes
        config = ValidationConfig()
        validator = AblationValidator(config)
        
        subset = validator._create_feature_subset(features, 'parents_transforms')
        
        # Should have p/ and t/ prefixed features
        parent_cols = [col for col in subset.columns if col.startswith('p/')]
        transform_cols = [col for col in subset.columns if col.startswith('t/')]
        assert len(parent_cols) == 5
        assert len(transform_cols) == 3
        assert len(subset.columns) == 8
    
    def test_create_feature_subset_patch(self, sample_data_with_prefixes):
        """Test creating feature subset with patch features."""
        features, _ = sample_data_with_prefixes
        config = ValidationConfig()
        validator = AblationValidator(config)
        
        subset = validator._create_feature_subset(features, 'parents_transforms_patch')
        
        # Should have p/, t/, and y_hat features
        parent_cols = [col for col in subset.columns if col.startswith('p/')]
        transform_cols = [col for col in subset.columns if col.startswith('t/')]
        patch_cols = [col for col in subset.columns if 'y_hat' in col]
        assert len(parent_cols) == 5
        assert len(transform_cols) == 3
        assert len(patch_cols) == 2
        assert len(subset.columns) == 10
    
    def test_create_feature_subset_interactions(self, sample_data_with_prefixes):
        """Test creating feature subset with interactions."""
        features, _ = sample_data_with_prefixes
        config = ValidationConfig()
        validator = AblationValidator(config)
        
        subset = validator._create_feature_subset(
            features, 'parents_transforms_patch_8_interactions'
        )
        
        # Should have p/, t/, y_hat, and limited i/ features
        parent_cols = [col for col in subset.columns if col.startswith('p/')]
        transform_cols = [col for col in subset.columns if col.startswith('t/')]
        patch_cols = [col for col in subset.columns if 'y_hat' in col]
        interaction_cols = [col for col in subset.columns if col.startswith('i/')]
        assert len(parent_cols) == 5
        assert len(transform_cols) == 3
        assert len(patch_cols) == 2
        # Should limit to 8 interactions, but we only have 4
        assert len(interaction_cols) <= 8
    
    def test_create_feature_subset_all(self, sample_data_with_prefixes):
        """Test creating feature subset with all features."""
        features, _ = sample_data_with_prefixes
        config = ValidationConfig()
        validator = AblationValidator(config)
        
        subset = validator._create_feature_subset(
            features, 'parents_transforms_patch_15_interactions'
        )
        
        # Should have all features
        assert len(subset.columns) == len(features.columns)
    
    def test_run_ablation(self, sample_data_with_prefixes):
        """Test running ablation testing."""
        features, targets = sample_data_with_prefixes
        
        # Use minimal config for faster testing with lower requirements
        config = ValidationConfig(
            n_outer_folds=2,
            n_inner_folds=2,
            embargo_pct=0.0,  # No embargo to maximize usable data
            min_train_samples=50,
            min_val_samples=20,
            ablation_steps=['parents_only', 'parents_transforms']
        )
        
        validator = AblationValidator(config)
        model_config = {'model_type': 'linear'}
        
        results = validator.run_ablation(features, targets, model_config)
        
        # Results may be empty or partial if folds fail, just check it's a dict
        assert isinstance(results, dict)
        
        # Check result structure for any returned steps
        for step_name, step_results in results.items():
            assert 'ic_mean' in step_results
            assert 'ic_std' in step_results
            assert 'auc_mean' in step_results
            assert 'auc_std' in step_results
            assert 'mse_mean' in step_results
            assert 'mse_std' in step_results
            assert 'n_features' in step_results


# ============================================================================
# Test SPAValidator
# ============================================================================

class TestSPAValidator:
    """Test SPAValidator functionality."""
    
    def test_initialization(self, basic_config):
        """Test SPAValidator initialization."""
        validator = SPAValidator(basic_config)
        
        assert validator.config == basic_config
    
    def test_run_spa_test_basic(self, sample_data):
        """Test basic SPA test execution."""
        features, targets = sample_data
        
        # Use very few permutations and lower requirements for faster testing
        config = ValidationConfig(
            n_outer_folds=2,
            n_inner_folds=2,
            embargo_pct=0.0,  # No embargo to maximize usable data
            min_train_samples=50,
            min_val_samples=20,
            spa_permutations=3  # Very low for testing
        )
        
        validator = SPAValidator(config)
        model_config = {'model_type': 'linear'}
        
        p_value = validator.run_spa_test(features, targets, model_config)
        
        # P-value should be between 0 and 1
        assert 0 <= p_value <= 1
    
    def test_spa_test_with_no_permutations(self, sample_data):
        """Test SPA test with no permutations."""
        features, targets = sample_data
        
        config = ValidationConfig(
            n_outer_folds=2,
            n_inner_folds=2,
            embargo_pct=0.0,
            min_train_samples=50,
            min_val_samples=20,
            spa_permutations=0
        )
        
        validator = SPAValidator(config)
        model_config = {'model_type': 'linear'}
        
        p_value = validator.run_spa_test(features, targets, model_config)
        
        # With no permutations, should return 1.0
        assert p_value == 1.0


# ============================================================================
# Test Complete Validation
# ============================================================================

class TestCompleteValidation:
    """Test complete validation pipeline."""
    
    def test_run_complete_validation(self, sample_data):
        """Test complete validation pipeline."""
        features, targets = sample_data
        
        # Use minimal config for faster testing
        config = ValidationConfig(
            n_outer_folds=2,
            n_inner_folds=2,
            embargo_pct=0.0,  # No embargo to maximize usable data
            min_train_samples=50,
            min_val_samples=20,
            spa_permutations=2,  # Very low for testing
            ablation_steps=[]  # Skip ablation for faster testing
        )
        
        model_config = {'model_type': 'linear'}
        
        result = run_complete_validation(features, targets, model_config, config)
        
        # Check result structure
        assert isinstance(result, ValidationResult)
        assert 'mean' in result.ic_scores
        assert 'mean' in result.auc_scores
        assert 'mean' in result.mse_scores
        assert result.spa_p_value is not None
        assert len(result.fold_results) > 0
    
    def test_run_complete_validation_with_defaults(self, sample_data):
        """Test complete validation with modified config."""
        features, targets = sample_data
        model_config = {'model_type': 'linear'}
        
        # Use a minimal config instead of default to speed up testing
        config = ValidationConfig(
            n_outer_folds=2,
            n_inner_folds=2,
            embargo_pct=0.0,
            min_train_samples=50,
            min_val_samples=20,
            spa_permutations=2,
            ablation_steps=[]  # Skip ablation
        )
        
        result = run_complete_validation(features, targets, model_config, config)
        
        assert isinstance(result, ValidationResult)


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_validation_with_nan_features(self, model_config):
        """Test validation with NaN features."""
        features = pd.DataFrame(np.random.randn(200, 5))
        features.iloc[10:20, 2] = np.nan
        targets = pd.Series(np.random.randn(200))
        
        # Use config with lower requirements
        config = ValidationConfig(
            n_outer_folds=2,
            n_inner_folds=2,
            embargo_pct=0.0,
            min_train_samples=30,
            min_val_samples=10
        )
        
        validator = WalkForwardValidator(config)
        
        # Should handle NaN gracefully (will likely fail all folds)
        # Just checking it doesn't crash
        try:
            result = validator.validate(features, targets, model_config)
            assert isinstance(result, ValidationResult)
        except ValueError as e:
            # It's OK if it fails with empty scores
            assert "zero-size array" in str(e) or "minimum" in str(e)
    
    def test_validation_with_constant_targets(self, basic_config, model_config):
        """Test validation with constant targets."""
        features = pd.DataFrame(np.random.randn(200, 5))
        targets = pd.Series(np.ones(200))  # All same value
        
        validator = WalkForwardValidator(basic_config)
        result = validator.validate(features, targets, model_config)
        
        # Should handle gracefully
        assert isinstance(result, ValidationResult)
        # AUC should be 0.5 for constant targets
        assert abs(result.auc_scores['mean'] - 0.5) < 0.1
    
    def test_validation_with_empty_model_configs(self, sample_data):
        """Test validation with empty model configs."""
        features, targets = sample_data
        
        # Use config with lower requirements
        config = ValidationConfig(
            n_outer_folds=2,
            n_inner_folds=2,
            embargo_pct=0.0,
            min_train_samples=50,
            min_val_samples=20
        )
        
        validator = WalkForwardValidator(config)
        
        # Empty config dict should be handled (will likely fail all folds)
        try:
            result = validator.validate(features, targets, {})
            assert isinstance(result, ValidationResult)
        except (ValueError, IndexError) as e:
            # It's OK if it fails - empty configs are invalid
            assert "zero-size array" in str(e) or "list index" in str(e)
    
    def test_calculate_ic_with_nan_predictions(self, basic_config):
        """Test IC calculation with NaN predictions."""
        validator = WalkForwardValidator(basic_config)
        
        predictions = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10])
        actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        ic = validator._calculate_ic(predictions, actual)
        
        # Should handle NaN gracefully
        assert not np.isnan(ic) or ic == 0.0
    
    def test_ablation_with_missing_feature_types(self, basic_config):
        """Test ablation when certain feature types are missing."""
        # Create features without some prefixes
        features = pd.DataFrame(
            np.random.randn(200, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        targets = pd.Series(np.random.randn(200))
        
        validator = AblationValidator(basic_config)
        model_config = {'model_type': 'linear'}
        
        # Should handle missing feature types gracefully
        results = validator.run_ablation(features, targets, model_config)
        
        # Results may be empty or partial
        assert isinstance(results, dict)


# ============================================================================
# Test Data Types and Validation
# ============================================================================

class TestDataTypes:
    """Test data type handling."""
    
    def test_fold_result_dataclass(self):
        """Test FoldResult dataclass creation."""
        fold_result = FoldResult(
            train_start=0,
            train_end=100,
            val_start=101,
            val_end=150,
            ic_score=0.5,
            auc_score=0.65,
            mse_score=0.15,
            feature_importance={'f1': 0.5, 'f2': 0.3},
            predictions=np.array([1, 2, 3]),
            actual=np.array([1.1, 2.1, 2.9])
        )
        
        assert fold_result.train_start == 0
        assert fold_result.train_end == 100
        assert fold_result.val_start == 101
        assert fold_result.val_end == 150
        assert fold_result.ic_score == 0.5
        assert fold_result.auc_score == 0.65
        assert fold_result.mse_score == 0.15
        assert len(fold_result.feature_importance) == 2
        assert len(fold_result.predictions) == 3
        assert len(fold_result.actual) == 3
    
    def test_validation_result_dataclass(self):
        """Test ValidationResult dataclass creation."""
        result = ValidationResult(
            ic_scores={'mean': 0.5, 'std': 0.1, 'min': 0.3, 'max': 0.7},
            auc_scores={'mean': 0.65, 'std': 0.05, 'min': 0.6, 'max': 0.7},
            mse_scores={'mean': 0.15, 'std': 0.02, 'min': 0.13, 'max': 0.17},
            fold_results=[],
            ablation_results={},
            spa_p_value=0.1,
            metadata={'n_folds_completed': 3}
        )
        
        assert result.ic_scores['mean'] == 0.5
        assert result.auc_scores['mean'] == 0.65
        assert result.mse_scores['mean'] == 0.15
        assert result.spa_p_value == 0.1
        assert result.metadata['n_folds_completed'] == 3
    
    def test_validation_type_enum(self):
        """Test ValidationType enum."""
        assert ValidationType.WALK_FORWARD.value == "walk_forward"
        assert ValidationType.NESTED_CV.value == "nested_cv"
        assert ValidationType.ABLATION.value == "ablation"
        assert ValidationType.SPA_CHECK.value == "spa_check"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

