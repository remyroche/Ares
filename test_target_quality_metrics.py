"""
Test script for target quality metrics calculation.
"""

import numpy as np
import pandas as pd
from src.training.steps.pre_training.utils.target_quality_metrics import calculate_target_quality_metrics

# Import des assertions standardisées
from tests.utils.assertions import (
    assert_float_equals,
    assert_dict_structure,
    assert_list_structure
)


def test_target_quality_metrics():
    """Test target quality metrics with synthetic data."""

    print("=" * 80)
    print("Testing Target Quality Metrics")
    print("=" * 80)

    # Test Case 1: Good quality target with structure
    print("\n1. Testing with structured target (sine wave + noise)...")
    np.random.seed(42)
    n = 1000
    x = np.linspace(0, 4 * np.pi, n)
    y_structured = np.sin(x) + 0.1 * np.random.randn(n)

    labels_structured = pd.Series(y_structured, name='target_long')
    metrics1 = calculate_target_quality_metrics(labels_structured)

    print("\nResults:")
    
    # Validation avec assertions standardisées
    assert_dict_structure(
        metrics1,
        ['overall_assessment', 'variance_distribution', 'autocorrelation', 'entropy', 'baseline_predictors'],
        message="Les métriques doivent contenir les clés principales"
    )
    
    assert_dict_structure(
        metrics1['overall_assessment'],
        ['quality_grade', 'quality_score'],
        message="L'évaluation globale doit contenir 'quality_grade' et 'quality_score'"
    )
    
    quality_score = metrics1['overall_assessment']['quality_score']
    assert isinstance(quality_score, (int, float)), "Le quality score doit être numérique"
    assert 0 <= quality_score <= 100, f"Le quality score doit être entre 0 et 100, valeur: {quality_score}"
    print(f"  Quality Grade: {metrics1['overall_assessment']['quality_grade']}")
    print(f"  Quality Score: {quality_score:.1f}/100")
    
    assert_dict_structure(
        metrics1['variance_distribution'],
        ['variance'],
        message="La distribution de variance doit contenir 'variance'"
    )
    variance = metrics1['variance_distribution']['variance']
    assert isinstance(variance, (int, float)), "La variance doit être numérique"
    assert variance >= 0, f"La variance doit être non-négative, valeur: {variance}"
    print(f"  Variance: {variance:.6f}")
    
    assert_dict_structure(
        metrics1['autocorrelation'],
        ['lag1_autocorrelation'],
        message="L'autocorrélation doit contenir 'lag1_autocorrelation'"
    )
    lag1_autocorr = metrics1['autocorrelation']['lag1_autocorrelation']
    assert isinstance(lag1_autocorr, (int, float)), "L'autocorrélation lag-1 doit être numérique"
    assert -1 <= lag1_autocorr <= 1, f"L'autocorrélation lag-1 doit être entre -1 et 1, valeur: {lag1_autocorr}"
    print(f"  Lag-1 Autocorrelation: {lag1_autocorr:.4f}")
    
    assert_dict_structure(
        metrics1['entropy'],
        ['normalized_entropy'],
        message="L'entropie doit contenir 'normalized_entropy'"
    )
    entropy = metrics1['entropy']['normalized_entropy']
    assert isinstance(entropy, (int, float)), "L'entropie normalisée doit être numérique"
    assert entropy >= 0, f"L'entropie doit être non-négative, valeur: {entropy}"
    print(f"  Normalized Entropy: {entropy:.4f}")
    
    assert_dict_structure(
        metrics1['baseline_predictors'],
        ['best_baseline'],
        message="Les prédicteurs de base doivent contenir 'best_baseline'"
    )
    assert_dict_structure(
        metrics1['baseline_predictors']['best_baseline'],
        ['name'],
        message="Le meilleur baseline doit contenir 'name'"
    )
    print(f"  Best Baseline: {metrics1['baseline_predictors']['best_baseline']['name']}")

    # Test Case 2: Noisy target
    print("\n2. Testing with noisy target (random)...")
    y_noisy = np.random.randn(n)
    labels_noisy = pd.Series(y_noisy, name='target_long')
    metrics2 = calculate_target_quality_metrics(labels_noisy)

    print("\nResults:")
    print(f"  Quality Grade: {metrics2['overall_assessment']['quality_grade']}")
    print(f"  Quality Score: {metrics2['overall_assessment']['quality_score']:.1f}/100")
    print(f"  Variance: {metrics2['variance_distribution']['variance']:.6f}")
    print(f"  Lag-1 Autocorrelation: {metrics2['autocorrelation']['lag1_autocorrelation']:.4f}")
    print(f"  Normalized Entropy: {metrics2['entropy']['normalized_entropy']:.4f}")
    print(f"  Is Highly Noisy: {metrics2['autocorrelation']['is_highly_noisy']}")

    # Test Case 3: Constant target (bad quality)
    print("\n3. Testing with constant target...")
    y_constant = np.ones(n) * 5.0
    labels_constant = pd.Series(y_constant, name='target_long')
    metrics3 = calculate_target_quality_metrics(labels_constant)

    print("\nResults:")
    print(f"  Quality Grade: {metrics3['overall_assessment']['quality_grade']}")
    print(f"  Quality Score: {metrics3['overall_assessment']['quality_score']:.1f}/100")
    print(f"  Variance: {metrics3['variance_distribution']['variance']:.6f}")
    print(f"  Is Nearly Constant: {metrics3['variance_distribution']['is_nearly_constant']}")
    print(f"  Issues: {metrics3['overall_assessment']['issues_detected']}")

    # Test Case 4: Binary target (trading signals)
    print("\n4. Testing with binary target (trading signals)...")
    y_binary = np.zeros(n)
    y_binary[np.random.choice(n, size=50, replace=False)] = 1.0  # 5% opportunities
    labels_binary = pd.DataFrame({
        'target_long': y_binary,
        'target_short': np.zeros(n)
    })
    metrics4 = calculate_target_quality_metrics(labels_binary)

    print("\nResults:")
    print(f"  Quality Grade: {metrics4['overall_assessment']['quality_grade']}")
    print(f"  Quality Score: {metrics4['overall_assessment']['quality_score']:.1f}/100")
    print(f"  Variance: {metrics4['variance_distribution']['variance']:.6f}")
    print(f"  Mean: {metrics4['variance_distribution']['mean']:.6f}")
    print(f"  Best Baseline MSE: {metrics4['baseline_predictors']['best_baseline']['mse']:.6f}")

    # Test Case 5: Target with outliers
    print("\n5. Testing with target containing outliers...")
    y_outliers = np.random.randn(n)
    y_outliers[np.random.choice(n, size=10, replace=False)] = 10.0  # Add outliers
    labels_outliers = pd.Series(y_outliers, name='target_long')
    metrics5 = calculate_target_quality_metrics(labels_outliers)

    print("\nResults:")
    print(f"  Quality Grade: {metrics5['overall_assessment']['quality_grade']}")
    print(f"  Quality Score: {metrics5['overall_assessment']['quality_score']:.1f}/100")
    print(f"  Outliers: {metrics5['distribution_outliers']['n_outliers']} ({metrics5['distribution_outliers']['outlier_percentage']:.2f}%)")
    print(f"  Skewness: {metrics5['distribution_outliers']['skewness']:.4f}")
    print(f"  Kurtosis: {metrics5['distribution_outliers']['kurtosis']:.4f}")

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)

    return True


if __name__ == '__main__':
    try:
        test_target_quality_metrics()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
