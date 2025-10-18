"""
Test Data Leakage Detection

Tests for leakage detection utilities in the Analyst→Tactician pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import validation utilities
from src.utils.ml_common.validation.leakage import (
    detect_negative_shifts,
    analyze_feature_shifts,
    rolling_holdout_test,
    analyze_feature_label_correlation,
    assert_past_only,
    validate_leakage_prevention
)


class TestNegativeShiftDetection:
    """Test detection of negative shift operations in feature expressions."""
    
    def test_detect_negative_shifts_basic(self):
        """Test basic negative shift detection."""
        # Test with simple negative shift
        expression = "data.shift(-1)"
        shifts = detect_negative_shifts(expression)
        assert -1 in shifts
    
    def test_detect_negative_shifts_multiple(self):
        """Test detection of multiple negative shifts."""
        expression = "data.shift(-1).rolling(5).mean().shift(-2)"
        shifts = detect_negative_shifts(expression)
        assert -1 in shifts
        assert -2 in shifts
    
    def test_detect_negative_shifts_no_negative(self):
        """Test with no negative shifts."""
        expression = "data.shift(1).rolling(5).mean()"
        shifts = detect_negative_shifts(expression)
        assert len(shifts) == 0
    
    def test_detect_negative_shifts_unary_operation(self):
        """Test detection with unary minus operation."""
        expression = "data.shift(-(-1))"  # This would be shift(1)
        shifts = detect_negative_shifts(expression)
        assert len(shifts) == 0  # No negative shifts after evaluation
    
    def test_detect_negative_shifts_invalid_expression(self):
        """Test with invalid expression."""
        expression = "invalid syntax here"
        shifts = detect_negative_shifts(expression)
        assert len(shifts) == 0  # Should handle gracefully


class TestFeatureShiftAnalysis:
    """Test feature shift pattern analysis."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature DataFrame for testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
        
        # Create features with different shift patterns
        np.random.seed(42)
        
        # Feature 1: Normal lagged feature
        feature1 = pd.Series(np.random.randn(100), index=dates)
        
        # Feature 2: Feature with high forward correlation (potential leakage)
        feature2 = feature1.shift(-1) + np.random.randn(100) * 0.1
        
        # Feature 3: Constant feature
        feature3 = pd.Series([1.0] * 100, index=dates)
        
        # Feature 4: Feature with perfect changes (potential leakage)
        feature4 = pd.Series([1.0 if i % 2 == 0 else 2.0 for i in range(100)], index=dates)
        
        features = pd.DataFrame({
            'normal_feature': feature1,
            'suspicious_feature': feature2,
            'constant_feature': feature3,
            'perfect_changes': feature4
        }, index=dates)
        
        return features
    
    def test_analyze_feature_shifts_basic(self, sample_features):
        """Test basic feature shift analysis."""
        analysis = analyze_feature_shifts(sample_features)
        
        assert 'total_features' in analysis
        assert 'analyzed_features' in analysis
        assert 'suspicious_features' in analysis
        assert 'shift_patterns' in analysis
        assert 'lag_analysis' in analysis
        
        assert analysis['total_features'] == len(sample_features.columns)
        assert analysis['analyzed_features'] == len(sample_features.columns)
    
    def test_analyze_feature_shifts_suspicious_detection(self, sample_features):
        """Test detection of suspicious shift patterns."""
        analysis = analyze_feature_shifts(sample_features)
        
        # Should detect the suspicious feature with high forward correlation
        suspicious_features = analysis['suspicious_features']
        assert len(suspicious_features) > 0
        
        # Check that suspicious feature is identified
        suspicious_feature_names = [sf['feature'] for sf in suspicious_features]
        assert 'suspicious_feature' in suspicious_feature_names
    
    def test_analyze_feature_shifts_lag_analysis(self, sample_features):
        """Test lag correlation analysis."""
        analysis = analyze_feature_shifts(sample_features)
        
        lag_analysis = analysis['lag_analysis']
        
        # Each feature should have lag analysis
        for feature_name in sample_features.columns:
            assert feature_name in lag_analysis
        
        # Check that lag correlations are calculated
        for feature_name, lags in lag_analysis.items():
            assert isinstance(lags, dict)
            assert len(lags) > 0  # Should have some lag correlations


class TestRollingHoldoutTest:
    """Test rolling holdout test for leakage detection."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='15min')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'feature3': np.random.randn(50)
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def sample_target(self, sample_data):
        """Create sample target for testing."""
        # Create target with some correlation to features
        target = sample_data['feature1'] * 0.5 + np.random.randn(len(sample_data)) * 0.1
        return target
    
    def mock_feature_builder_past_only(self, data, strict_past_only=True):
        """Mock feature builder that respects past-only constraint."""
        if strict_past_only:
            # Past-only version: shift all features by 1
            features = data.shift(1)
        else:
            # Naive version: use features as-is (potential leakage)
            features = data.copy()
        return features
    
    def test_rolling_holdout_test_no_leakage(self, sample_data, sample_target):
        """Test rolling holdout test with no leakage."""
        # Use a feature builder that doesn't leak
        result = rolling_holdout_test(
            X=sample_data,
            y=sample_target,
            feature_builder_func=self.mock_feature_builder_past_only,
            holdout_size=10
        )
        
        assert 'tested_indices' in result
        assert 'mismatches' in result
        assert 'total_tests' in result
        assert 'mismatch_rate' in result
        
        # With proper past-only implementation, mismatch rate should be 0
        assert result['mismatch_rate'] == 0.0
        assert len(result['mismatches']) == 0
    
    def test_rolling_holdout_test_with_leakage(self, sample_data, sample_target):
        """Test rolling holdout test with leakage detection."""
        
        def leaking_feature_builder(data, strict_past_only=True):
            """Feature builder that leaks when strict_past_only=False."""
            if strict_past_only:
                return data.shift(1)  # Past-only
            else:
                return data  # Leaks current information
        
        result = rolling_holdout_test(
            X=sample_data,
            y=sample_target,
            feature_builder_func=leaking_feature_builder,
            holdout_size=10
        )
        
        # Should detect mismatches when there's leakage
        assert result['mismatch_rate'] > 0.0
        assert len(result['mismatches']) > 0
        
        # Check mismatch structure
        for mismatch in result['mismatches']:
            assert 'index' in mismatch
            assert 'feature' in mismatch
            assert 'past_only_value' in mismatch
            assert 'naive_value' in mismatch
            assert 'difference' in mismatch


class TestFeatureLabelCorrelation:
    """Test feature-label correlation analysis."""
    
    @pytest.fixture
    def sample_features_and_target(self):
        """Create sample features and target for correlation testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
        np.random.seed(42)
        
        # Create features with different correlation patterns
        feature1 = pd.Series(np.random.randn(100), index=dates)
        feature2 = feature1 * 0.8 + np.random.randn(100) * 0.2  # High correlation
        feature3 = pd.Series(np.random.randn(100), index=dates)  # Low correlation
        feature4 = feature1 * 0.9  # Very high correlation (suspicious)
        
        features = pd.DataFrame({
            'low_corr_feature': feature3,
            'high_corr_feature': feature2,
            'suspicious_feature': feature4,
            'random_feature': pd.Series(np.random.randn(100), index=dates)
        }, index=dates)
        
        # Create target correlated with feature1
        target = feature1 * 0.7 + np.random.randn(100) * 0.3
        
        return features, target
    
    def test_analyze_feature_label_correlation_basic(self, sample_features_and_target):
        """Test basic feature-label correlation analysis."""
        features, target = sample_features_and_target
        
        analysis = analyze_feature_label_correlation(features, target)
        
        assert 'feature_correlations' in analysis
        assert 'suspicious_correlations' in analysis
        assert 'high_correlation_features' in analysis
        assert 'correlation_statistics' in analysis
        
        # Check that correlations are calculated for all features
        assert len(analysis['feature_correlations']) == len(features.columns)
    
    def test_analyze_feature_label_correlation_suspicious_detection(self, sample_features_and_target):
        """Test detection of suspicious correlations."""
        features, target = sample_features_and_target
        
        analysis = analyze_feature_label_correlation(features, target)
        
        # Should detect the suspicious feature with very high correlation
        suspicious_features = [sc['feature'] for sc in analysis['suspicious_correlations']]
        assert 'suspicious_feature' in suspicious_features
        
        # Check suspicious correlation details
        for suspicious in analysis['suspicious_correlations']:
            assert 'feature' in suspicious
            assert 'correlation' in suspicious
            assert 'abs_correlation' in suspicious
            assert 'suspicious_reason' in suspicious
            assert abs(suspicious['correlation']) > 0.8
    
    def test_analyze_feature_label_correlation_statistics(self, sample_features_and_target):
        """Test correlation statistics calculation."""
        features, target = sample_features_and_target
        
        analysis = analyze_feature_label_correlation(features, target)
        
        stats = analysis['correlation_statistics']
        
        assert 'mean_abs_correlation' in stats
        assert 'max_abs_correlation' in stats
        assert 'min_abs_correlation' in stats
        assert 'std_abs_correlation' in stats
        assert 'features_above_0.5' in stats
        assert 'features_above_0.8' in stats
        
        # Check that statistics are reasonable
        assert 0 <= stats['mean_abs_correlation'] <= 1
        assert 0 <= stats['max_abs_correlation'] <= 1
        assert 0 <= stats['min_abs_correlation'] <= 1
        assert stats['features_above_0.5'] >= 0
        assert stats['features_above_0.8'] >= 0


class TestAssertPastOnly:
    """Test the main assert_past_only function."""
    
    @pytest.fixture
    def sample_clean_data(self):
        """Create sample data without leakage."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
        np.random.seed(42)
        
        # Create properly lagged features
        feature1 = pd.Series(np.random.randn(100), index=dates).shift(1)
        feature2 = pd.Series(np.random.randn(100), index=dates).shift(2)
        feature3 = pd.Series(np.random.randn(100), index=dates).rolling(5).mean().shift(1)
        
        X = pd.DataFrame({
            'lagged_feature1': feature1,
            'lagged_feature2': feature2,
            'rolling_feature': feature3
        }, index=dates)
        
        # Create target with some correlation but not perfect
        y = pd.Series(np.random.randn(100), index=dates)
        
        return X, y
    
    @pytest.fixture
    def sample_leaky_data(self):
        """Create sample data with potential leakage."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
        np.random.seed(42)
        
        # Create features with potential leakage
        base_feature = pd.Series(np.random.randn(100), index=dates)
        
        X = pd.DataFrame({
            'lagged_feature': base_feature.shift(1),  # Properly lagged
            'current_feature': base_feature,  # Potential leakage
            'future_feature': base_feature.shift(-1)  # Definite leakage
        }, index=dates)
        
        # Create target highly correlated with base feature
        y = base_feature * 0.9 + np.random.randn(100) * 0.1
        
        return X, y
    
    def test_assert_past_only_clean_data(self, sample_clean_data):
        """Test assert_past_only with clean data."""
        X, y = sample_clean_data
        
        result = assert_past_only(X, y, horizon_bars=1, strict_mode=False)
        
        assert isinstance(result, dict)
        assert 'has_leakage' in result
        assert 'leakage_sources' in result
        assert 'feature_analysis' in result
        assert 'shift_analysis' in result
        assert 'correlation_analysis' in result
        assert 'recommendations' in result
        assert 'warnings' in result
        
        # Clean data should not have leakage
        assert not result['has_leakage']
        assert len(result['leakage_sources']) == 0
    
    def test_assert_past_only_leaky_data(self, sample_leaky_data):
        """Test assert_past_only with leaky data."""
        X, y = sample_leaky_data
        
        result = assert_past_only(X, y, horizon_bars=1, strict_mode=False)
        
        # Leaky data should be detected
        assert result['has_leakage']
        assert len(result['leakage_sources']) > 0
        
        # Check that suspicious correlations are detected
        suspicious_features = [sf['feature'] for sf in result['shift_analysis']['suspicious_features']]
        assert 'current_feature' in suspicious_features or 'future_feature' in suspicious_features
    
    def test_assert_past_only_strict_mode(self, sample_leaky_data):
        """Test assert_past_only in strict mode."""
        X, y = sample_leaky_data
        
        # In strict mode, should raise or return failure for leaky data
        result = assert_past_only(X, y, horizon_bars=1, strict_mode=True)
        
        assert result['has_leakage']
        assert len(result['leakage_sources']) > 0
        
        # Should have high-severity leakage sources
        high_severity_sources = [ls for ls in result['leakage_sources'] if ls['severity'] == 'high']
        assert len(high_severity_sources) > 0


class TestValidateLeakagePrevention:
    """Test the validate_leakage_prevention integration function."""
    
    def test_validate_leakage_prevention_success(self):
        """Test successful leakage prevention validation."""
        # Create clean artifacts
        dates = pd.date_range(start='2023-01-01', periods=50, freq='15min')
        np.random.seed(42)
        
        X = pd.DataFrame({
            'feature1': pd.Series(np.random.randn(50), index=dates).shift(1),
            'feature2': pd.Series(np.random.randn(50), index=dates).shift(2)
        }, index=dates)
        
        y = pd.Series(np.random.randn(50), index=dates)
        
        artifacts = {
            'features': X,
            'targets': y
        }
        
        result = validate_leakage_prevention(artifacts)
        
        assert 'success' in result
        assert 'results' in result
        assert 'overall_leakage_detected' in result
        assert 'config' in result
        
        # Clean data should pass validation
        assert result['success']
        assert not result['overall_leakage_detected']
    
    def test_validate_leakage_prevention_failure(self):
        """Test leakage prevention validation failure."""
        # Create leaky artifacts
        dates = pd.date_range(start='2023-01-01', periods=50, freq='15min')
        np.random.seed(42)
        
        base_feature = pd.Series(np.random.randn(50), index=dates)
        
        X = pd.DataFrame({
            'lagged_feature': base_feature.shift(1),
            'leaky_feature': base_feature  # No lag - potential leakage
        }, index=dates)
        
        y = base_feature * 0.9 + np.random.randn(50) * 0.1  # High correlation
        
        artifacts = {
            'features': X,
            'targets': y
        }
        
        config = {
            'strict_mode': True
        }
        
        result = validate_leakage_prevention(artifacts, config)
        
        # Leaky data should fail in strict mode
        assert not result['success']
        assert result['overall_leakage_detected']
    
    def test_validate_leakage_prevention_individual_features(self):
        """Test validation of individual feature columns."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='15min')
        np.random.seed(42)
        
        base_feature = pd.Series(np.random.randn(50), index=dates)
        
        artifacts = {
            'lagged_feature': base_feature.shift(1),
            'leaky_feature': base_feature,  # Potential leakage
            'future_feature': base_feature.shift(-1)  # Definite leakage
        }
        
        config = {
            'feature_columns': ['lagged_feature', 'leaky_feature', 'future_feature'],
            'strict_mode': True
        }
        
        result = validate_leakage_prevention(artifacts, config)
        
        # Should detect leakage in individual features
        assert not result['success']
        assert result['overall_leakage_detected']


class TestLeakageDetectionEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_assert_past_only_empty_data(self):
        """Test assert_past_only with empty data."""
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=float)
        
        result = assert_past_only(empty_X, empty_y)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'has_leakage' in result
    
    def test_assert_past_only_mismatched_lengths(self):
        """Test assert_past_only with mismatched data lengths."""
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = pd.Series([1, 2])  # Different length
        
        # Should handle gracefully or raise appropriate error
        try:
            result = assert_past_only(X, y)
            assert isinstance(result, dict)
        except Exception as e:
            # Acceptable to raise error for mismatched lengths
            assert "length" in str(e).lower() or "align" in str(e).lower()
    
    def test_rolling_holdout_test_empty_data(self):
        """Test rolling holdout test with empty data."""
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=float)
        
        def mock_builder(data, strict_past_only=True):
            return data
        
        result = rolling_holdout_test(empty_X, empty_y, mock_builder)
        
        assert result['total_tests'] == 0
        assert result['mismatch_rate'] == 0.0
        assert len(result['mismatches']) == 0


if __name__ == "__main__":
    pytest.main([__file__])
