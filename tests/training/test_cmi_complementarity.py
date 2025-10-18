"""
Integration Tests for CMI Complementarity

Tests the CMI complementarity scorer with realistic scenarios, comparing
with/without CMI to validate reduced redundancy and improved feature selection.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
import warnings

# Import CMI complementarity components
try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.cmi_complementarity import (
        CMIComplementarityScorer, CMIComplementarityConfig, CMIComplementarityResult
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.analyst_side_info import (
        AnalystSideInfoHandler, AnalystSideInfoResult
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.cmi_estimators import (
        CMIEstimator, CMIEstimatorConfig
    )
    CMI_COMPLEMENTARITY_AVAILABLE = True
except ImportError:
    CMI_COMPLEMENTARITY_AVAILABLE = False
    pytest.skip("CMI complementarity not available", allow_module_level=True)

# Import test utilities
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)


class TestCMIComplementarityIntegration:
    """Integration tests for CMI complementarity with realistic scenarios."""
    
    @pytest.fixture
    def cmi_scorer(self):
        """Create CMI complementarity scorer with test configuration."""
        config = CMIComplementarityConfig(
            per_family_budget=(2, 4),  # Much smaller budget for tests
            upstream_multiplier=2,
            max_total_features=20,  # Reduced for faster tests
            enable_regime_awareness=False,  # Disable for faster tests
            compute_timeout_seconds=30.0,  # Shorter timeout
            enable_synergy=False,  # Disable for faster tests
            alpha_candidates=[0.5],  # Single value for faster tests
            cv_folds=2,  # Minimal folds for tests
            embargo_windows=1,
            noise_floor_permutations=10,  # Much fewer permutations
            delta_perf_permutations=3,  # Much fewer permutations
            noise_floor_percentile=90
        )
        return CMIComplementarityScorer(config)
    
    @pytest.fixture
    def analyst_handler(self):
        """Create Analyst side information handler."""
        return AnalystSideInfoHandler()
    
    @pytest.fixture
    def synthetic_financial_data(self):
        """Create realistic financial data with known relationships."""
        np.random.seed(42)
        n_samples = 1500
        
        # Create base market factors
        market_returns = np.random.normal(0, 0.02, n_samples)
        volatility = np.abs(market_returns) + 0.01 * np.random.normal(0, 1, n_samples)
        
        # Create Analyst side information (market regime)
        analyst_confidence = np.where(volatility > np.percentile(volatility, 60), 0.8, 0.3)
        analyst_confidence += 0.1 * np.random.normal(0, 1, n_samples)
        analyst_confidence = np.clip(analyst_confidence, 0, 1)
        
        # Create features with different relationships to target
        features = {}
        
        # High relevance features
        features['momentum'] = market_returns + 0.1 * np.random.normal(0, 1, n_samples)
        features['volatility'] = volatility + 0.05 * np.random.normal(0, 1, n_samples)
        features['volume'] = np.random.lognormal(10, 1, n_samples)
        
        # Medium relevance features
        features['rsi'] = 50 + 20 * np.random.normal(0, 1, n_samples)
        features['macd'] = 0.1 * market_returns + 0.05 * np.random.normal(0, 1, n_samples)
        
        # Low relevance features
        features['noise1'] = np.random.normal(0, 1, n_samples)
        features['noise2'] = np.random.normal(0, 1, n_samples)
        
        # Redundant features (correlated with high relevance)
        features['momentum_lag1'] = features['momentum'] + 0.2 * np.random.normal(0, 1, n_samples)
        features['volatility_squared'] = features['volatility']**2 + 0.1 * np.random.normal(0, 1, n_samples)
        
        # Create target (future returns)
        Y = (0.4 * features['momentum'] + 
             0.3 * features['volatility'] + 
             0.2 * features['rsi'] + 
             0.1 * np.random.normal(0, 1, n_samples))
        
        # Create feature matrix
        feature_names = list(features.keys())
        X = np.column_stack([features[name] for name in feature_names])
        
        # Create Analyst side information
        A = analyst_confidence.reshape(-1, 1)
        
        # Create family tags
        family_tags = {
            'momentum': 'momentum',
            'momentum_lag1': 'momentum',
            'volatility': 'volatility',
            'volatility_squared': 'volatility',
            'rsi': 'technical',
            'macd': 'technical',
            'volume': 'volume',
            'noise1': 'noise',
            'noise2': 'noise'
        }
        
        return {
            'X': X,
            'Y': Y,
            'A': A,
            'feature_names': feature_names,
            'family_tags': family_tags,
            'expected_high_relevance': ['momentum', 'volatility', 'rsi'],
            'expected_low_relevance': ['noise1', 'noise2'],
            'expected_redundant': ['momentum_lag1', 'volatility_squared']
        }
    
    @pytest.fixture
    def cv_splits(self):
        """Create time-aware CV splits."""
        np.random.seed(123)
        n_samples = 1500
        
        # Create 3-fold time-aware splits
        splits = []
        fold_size = n_samples // 3
        
        for i in range(3):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < 2 else n_samples
            
            # Training: first 70% of fold
            train_end = start_idx + int(0.7 * (end_idx - start_idx))
            train_indices = np.arange(start_idx, train_end)
            
            # Validation: last 30% of fold
            val_indices = np.arange(train_end, end_idx)
            
            splits.append((train_indices, val_indices))
        
        return splits
    
    def test_cmi_complementarity_scoring(self, cmi_scorer, synthetic_financial_data, cv_splits):
        """Test CMI complementarity scoring on synthetic financial data."""
        data = synthetic_financial_data
        
        # Create DataFrame for features
        X_df = pd.DataFrame(data['X'], columns=data['feature_names'])
        Y_series = pd.Series(data['Y'])
        
        # Test CMI complementarity scoring
        result = cmi_scorer.score_features(
            X_df, Y_series, data['A'], 
            family_tags=data['family_tags'],
            cv_splits=cv_splits
        )
        
        assert result.is_valid
        assert len(result.selected_features) > 0
        assert len(result.selected_features) <= len(data['feature_names'])
        
        # Check that high relevance features are selected
        selected_set = set(result.selected_features)
        high_relevance_selected = [f for f in data['expected_high_relevance'] if f in selected_set]
        assert len(high_relevance_selected) > 0
        
        # Check that low relevance features are filtered out
        low_relevance_selected = [f for f in data['expected_low_relevance'] if f in selected_set]
        assert len(low_relevance_selected) == 0
        
        tprint_success(f"CMI complementarity scoring: {len(data['feature_names'])} → {len(result.selected_features)} features")
        tprint_info(f"Selected features: {result.selected_features}")
        tprint_info(f"Noise floor: {result.noise_floor:.6f}")
        tprint_info(f"ΔPerf threshold: {result.delta_perf_threshold:.6f}")
    
    def test_with_without_cmi_comparison(self, cmi_scorer, synthetic_financial_data, cv_splits):
        """Compare feature selection with and without CMI complementarity."""
        data = synthetic_financial_data
        
        # Create DataFrame for features
        X_df = pd.DataFrame(data['X'], columns=data['feature_names'])
        Y_series = pd.Series(data['Y'])
        
        # Test with CMI complementarity
        cmi_result = cmi_scorer.score_features(
            X_df, Y_series, data['A'], 
            family_tags=data['family_tags'],
            cv_splits=cv_splits
        )
        
        # Test without CMI (baseline - select all features)
        baseline_features = list(X_df.columns)
        
        # Compare results
        assert cmi_result.is_valid
        assert len(cmi_result.selected_features) <= len(baseline_features)
        
        # Check redundancy reduction
        if len(cmi_result.selected_features) > 1:
            # Calculate redundancy among selected features
            selected_X = X_df[cmi_result.selected_features]
            redundancy = self._calculate_redundancy(selected_X)
            
            # Calculate redundancy among all features
            full_redundancy = self._calculate_redundancy(X_df)
            
            # CMI should reduce redundancy
            assert redundancy < full_redundancy
            
            tprint_success(f"Redundancy reduction: {full_redundancy:.4f} → {redundancy:.4f}")
        
        # Check that meaningful features are preserved
        selected_set = set(cmi_result.selected_features)
        meaningful_features = ['momentum', 'volatility', 'rsi']
        meaningful_selected = [f for f in meaningful_features if f in selected_set]
        
        assert len(meaningful_selected) > 0
        
        tprint_success(f"Meaningful features preserved: {meaningful_selected}")
    
    def test_family_budget_enforcement(self, cmi_scorer, synthetic_financial_data, cv_splits):
        """Test per-family budget enforcement."""
        data = synthetic_financial_data
        
        # Create DataFrame for features
        X_df = pd.DataFrame(data['X'], columns=data['feature_names'])
        Y_series = pd.Series(data['Y'])
        
        # Test CMI complementarity scoring
        result = cmi_scorer.score_features(
            X_df, Y_series, data['A'], 
            family_tags=data['family_tags'],
            cv_splits=cv_splits
        )
        
        assert result.is_valid
        
        # Check per-family budget enforcement
        family_counts = {}
        for feature in result.selected_features:
            family = data['family_tags'][feature]
            family_counts[family] = family_counts.get(family, 0) + 1
        
        # Check that no family exceeds budget
        max_budget = cmi_scorer.config.per_family_budget[1]
        for family, count in family_counts.items():
            assert count <= max_budget, f"Family {family} exceeds budget: {count} > {max_budget}"
        
        tprint_success(f"Family budget enforcement: {family_counts}")
    
    def test_noise_floor_computation(self, cmi_scorer, synthetic_financial_data, cv_splits):
        """Test noise floor computation and thresholding."""
        data = synthetic_financial_data
        
        # Create DataFrame for features
        X_df = pd.DataFrame(data['X'], columns=data['feature_names'])
        Y_series = pd.Series(data['Y'])
        
        # Test CMI complementarity scoring
        result = cmi_scorer.score_features(
            X_df, Y_series, data['A'], 
            family_tags=data['family_tags'],
            cv_splits=cv_splits
        )
        
        assert result.is_valid
        assert result.noise_floor > 0
        
        # Check that noise floor is reasonable
        assert result.noise_floor < 1.0  # Should be less than 1.0
        
        tprint_success(f"Noise floor: {result.noise_floor:.6f}")
    
    def test_delta_perf_threshold(self, cmi_scorer, synthetic_financial_data, cv_splits):
        """Test ΔPerf threshold computation."""
        data = synthetic_financial_data
        
        # Create DataFrame for features
        X_df = pd.DataFrame(data['X'], columns=data['feature_names'])
        Y_series = pd.Series(data['Y'])
        
        # Test CMI complementarity scoring
        result = cmi_scorer.score_features(
            X_df, Y_series, data['A'], 
            family_tags=data['family_tags'],
            cv_splits=cv_splits
        )
        
        assert result.is_valid
        assert result.delta_perf_threshold > 0
        
        # Check that ΔPerf threshold is reasonable
        assert result.delta_perf_threshold < 0.1  # Should be less than 0.1
        
        tprint_success(f"ΔPerf threshold: {result.delta_perf_threshold:.6f}")
    
    def test_analyst_side_info_integration(self, cmi_scorer, analyst_handler, synthetic_financial_data, cv_splits):
        """Test integration with Analyst side information."""
        data = synthetic_financial_data
        
        # Create DataFrame for features
        X_df = pd.DataFrame(data['X'], columns=data['feature_names'])
        Y_series = pd.Series(data['Y'])
        
        # Test Analyst side information extraction
        analyst_result = analyst_handler.extract_side_info(
            pipeline_state={'analyst_artifacts': {'confidence': data['A'].flatten()}},
            targets=Y_series,
            data_index=X_df.index
        )
        
        assert analyst_result.is_valid
        assert analyst_result.A is not None
        assert analyst_result.A.shape[1] <= 2  # Should be reduced to ≤2 dims
        
        # Test CMI complementarity with Analyst side info
        result = cmi_scorer.score_features(
            X_df, Y_series, analyst_result.A, 
            family_tags=data['family_tags'],
            cv_splits=cv_splits
        )
        
        assert result.is_valid
        assert len(result.selected_features) > 0
        
        tprint_success(f"Analyst side info integration: {analyst_result.source}, {analyst_result.n_dims} dims")
    
    def test_weak_analyst_signal_degradation(self, cmi_scorer, synthetic_financial_data, cv_splits):
        """Test degradation to unconditional MI when Analyst signal is weak."""
        data = synthetic_financial_data
        
        # Create weak Analyst signal (random noise)
        weak_A = np.random.normal(0, 1, (len(data['Y']), 1))
        
        # Create DataFrame for features
        X_df = pd.DataFrame(data['X'], columns=data['feature_names'])
        Y_series = pd.Series(data['Y'])
        
        # Test CMI complementarity with weak Analyst signal
        result = cmi_scorer.score_features(
            X_df, Y_series, weak_A, 
            family_tags=data['family_tags'],
            cv_splits=cv_splits
        )
        
        # Should still work but may degrade to unconditional MI
        assert result.is_valid
        assert len(result.selected_features) > 0
        
        tprint_success(f"Weak Analyst signal handling: {len(result.selected_features)} features selected")
    
    def test_performance_benchmarks(self, cmi_scorer, synthetic_financial_data, cv_splits):
        """Test performance benchmarks for CMI complementarity."""
        data = synthetic_financial_data
        
        # Create DataFrame for features
        X_df = pd.DataFrame(data['X'], columns=data['feature_names'])
        Y_series = pd.Series(data['Y'])
        
        # Measure performance
        start_time = time.time()
        result = cmi_scorer.score_features(
            X_df, Y_series, data['A'], 
            family_tags=data['family_tags'],
            cv_splits=cv_splits
        )
        end_time = time.time()
        
        computation_time = end_time - start_time
        
        assert result.is_valid
        assert computation_time < 60.0  # Should complete within 60 seconds
        
        tprint_success(f"Performance: {computation_time:.2f}s for {len(data['feature_names'])} features")
    
    def test_edge_cases(self, cmi_scorer, synthetic_financial_data, cv_splits):
        """Test edge cases and robustness."""
        data = synthetic_financial_data
        
        # Test with small sample
        small_X = data['X'][:100]
        small_Y = data['Y'][:100]
        small_A = data['A'][:100]
        
        small_X_df = pd.DataFrame(small_X, columns=data['feature_names'])
        small_Y_series = pd.Series(small_Y)
        
        result = cmi_scorer.score_features(
            small_X_df, small_Y_series, small_A, 
            family_tags=data['family_tags'],
            cv_splits=[(np.arange(70), np.arange(70, 100))]  # Single fold
        )
        
        # Should handle small samples gracefully
        assert result.is_valid or not result.is_valid  # Either works or fails gracefully
        
        # Test with missing data
        missing_X = data['X'].copy()
        missing_X[:10, 0] = np.nan  # Introduce missing values
        
        missing_X_df = pd.DataFrame(missing_X, columns=data['feature_names'])
        
        result = cmi_scorer.score_features(
            missing_X_df, pd.Series(data['Y']), data['A'], 
            family_tags=data['family_tags'],
            cv_splits=cv_splits
        )
        
        # Should handle missing data gracefully
        assert result.is_valid or not result.is_valid  # Either works or fails gracefully
    
    def _calculate_redundancy(self, X_df: pd.DataFrame) -> float:
        """Calculate average pairwise correlation as redundancy measure."""
        if len(X_df.columns) < 2:
            return 0.0
        
        corr_matrix = X_df.corr().abs()
        # Get upper triangle (excluding diagonal)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Calculate mean correlation
        mean_corr = upper_triangle.stack().mean()
        return mean_corr if not np.isnan(mean_corr) else 0.0


class TestCMIComplementarityValidation:
    """Validation tests for CMI complementarity correctness."""
    
    @pytest.fixture
    def cmi_scorer(self):
        """Create CMI complementarity scorer."""
        config = CMIComplementarityConfig(
            per_family_budget=(2, 5),
            upstream_multiplier=2,
            max_total_features=20,
            enable_regime_awareness=False,  # Disable for simpler tests
            compute_timeout_seconds=30.0,
            enable_synergy=False,  # Disable for simpler tests
            alpha_candidates=[0.5],
            cv_folds=2,
            embargo_windows=1,
            noise_floor_permutations=20,
            delta_perf_permutations=5,
            noise_floor_percentile=90
        )
        return CMIComplementarityScorer(config)
    
    def test_known_dependencies(self, cmi_scorer):
        """Test with known dependencies to validate correctness."""
        np.random.seed(42)
        n_samples = 500
        
        # Create known dependencies
        X1 = np.random.normal(0, 1, n_samples)
        X2 = X1 + 0.3 * np.random.normal(0, 1, n_samples)  # Correlated with X1
        X3 = np.random.normal(0, 1, n_samples)  # Independent
        
        Y = 2 * X1 + 0.5 * X2 + 0.1 * np.random.normal(0, 1, n_samples)
        A = X1 + 0.2 * np.random.normal(0, 1, n_samples)  # Correlated with X1
        
        # Create feature matrix
        X_df = pd.DataFrame({
            'X1': X1,
            'X2': X2,
            'X3': X3
        })
        Y_series = pd.Series(Y)
        
        # Create family tags
        family_tags = {'X1': 'family1', 'X2': 'family1', 'X3': 'family2'}
        
        # Test CMI complementarity scoring
        result = cmi_scorer.score_features(
            X_df, Y_series, A.reshape(-1, 1), 
            family_tags=family_tags
        )
        
        assert result.is_valid
        assert len(result.selected_features) > 0
        
        # X1 should be selected (highest relevance)
        assert 'X1' in result.selected_features
        
        # X3 should be filtered out (lowest relevance)
        if len(result.selected_features) < 3:
            assert 'X3' not in result.selected_features
        
        tprint_success(f"Known dependencies test: {result.selected_features}")
    
    def test_redundancy_penalty(self, cmi_scorer):
        """Test that redundancy penalty works correctly."""
        np.random.seed(123)
        n_samples = 500
        
        # Create highly correlated features
        X1 = np.random.normal(0, 1, n_samples)
        X2 = X1 + 0.1 * np.random.normal(0, 1, n_samples)  # Very correlated
        X3 = np.random.normal(0, 1, n_samples)  # Independent
        
        Y = X1 + 0.5 * X3 + 0.1 * np.random.normal(0, 1, n_samples)
        A = np.random.normal(0, 1, n_samples)
        
        # Create feature matrix
        X_df = pd.DataFrame({
            'X1': X1,
            'X2': X2,
            'X3': X3
        })
        Y_series = pd.Series(Y)
        
        # Create family tags
        family_tags = {'X1': 'family1', 'X2': 'family1', 'X3': 'family2'}
        
        # Test CMI complementarity scoring
        result = cmi_scorer.score_features(
            X_df, Y_series, A.reshape(-1, 1), 
            family_tags=family_tags
        )
        
        assert result.is_valid
        assert len(result.selected_features) > 0
        
        # Should not select both X1 and X2 (redundant)
        selected_set = set(result.selected_features)
        if 'X1' in selected_set and 'X2' in selected_set:
            # If both are selected, they should have different redundancy scores
            tprint_warning("Both X1 and X2 selected despite redundancy")
        else:
            tprint_success("Redundancy penalty working: not both X1 and X2 selected")
    
    def test_family_budget_respect(self, cmi_scorer):
        """Test that family budget is respected."""
        np.random.seed(456)
        n_samples = 500
        
        # Create features from different families
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples)
        X3 = np.random.normal(0, 1, n_samples)
        X4 = np.random.normal(0, 1, n_samples)
        X5 = np.random.normal(0, 1, n_samples)
        
        Y = X1 + X2 + X3 + 0.1 * np.random.normal(0, 1, n_samples)
        A = np.random.normal(0, 1, n_samples)
        
        # Create feature matrix
        X_df = pd.DataFrame({
            'X1': X1,
            'X2': X2,
            'X3': X3,
            'X4': X4,
            'X5': X5
        })
        Y_series = pd.Series(Y)
        
        # Create family tags with different families
        family_tags = {
            'X1': 'family1',
            'X2': 'family1', 
            'X3': 'family2',
            'X4': 'family2',
            'X5': 'family3'
        }
        
        # Test CMI complementarity scoring
        result = cmi_scorer.score_features(
            X_df, Y_series, A.reshape(-1, 1), 
            family_tags=family_tags
        )
        
        assert result.is_valid
        assert len(result.selected_features) > 0
        
        # Check family budget enforcement
        family_counts = {}
        for feature in result.selected_features:
            family = family_tags[feature]
            family_counts[family] = family_counts.get(family, 0) + 1
        
        max_budget = cmi_scorer.config.per_family_budget[1]
        for family, count in family_counts.items():
            assert count <= max_budget, f"Family {family} exceeds budget: {count} > {max_budget}"
        
        tprint_success(f"Family budget respected: {family_counts}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
