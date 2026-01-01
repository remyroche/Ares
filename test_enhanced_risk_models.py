#!/usr/bin/env python3
"""
Test Enhanced Risk Models (Phases 1 & 2)

Comprehensive test script to validate all risk model enhancements:
- Phase 1: Volatility term structure, directional bias, exponential smoothing
- Phase 2: Ensemble fusion, regime-specific weights, microstructure features

Expected Results:
- Risk Score MI: 0.0131 → 0.025+ (90%+ improvement)
- Path Risk Score MI: 0.0082 → 0.018+ (120%+ improvement)
- Better regime differentiation and trading signal filtering
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, '/Users/remyroche/Documents/Ares')

from src.utils.tprint import tprint, tprint_success, tprint_error, tprint_info
from src.training.steps.market_analysis.shared_utils.enhanced_risk_integration_example import (
    EnhancedRiskIntegration, 
    create_example_config,
    run_integration_example
)


def test_phase1_enhancements():
    """Test Phase 1 enhancements: volatility term structure, directional bias, smoothing."""
    tprint("🧪 Testing Phase 1 Enhancements")
    tprint("=" * 50)
    
    config = create_example_config()
    integration = EnhancedRiskIntegration(config)
    
    # Generate test data
    n_samples = 500
    
    # Test enhanced risk_score features
    risk_features = pd.DataFrame({
        'parkinson_volatility': np.random.gamma(2, 0.01, n_samples),
        'rolling_kurtosis': np.random.normal(0, 1, n_samples),
        'rolling_skewness': np.random.normal(0, 0.5, n_samples),
        'volatility_of_volatility': np.random.gamma(1, 0.005, n_samples),
        # Phase 1 enhancements
        'volatility_1h': np.random.gamma(2, 0.008, n_samples),
        'volatility_4h': np.random.gamma(2, 0.01, n_samples),
        'volatility_24h': np.random.gamma(2, 0.015, n_samples),
        'volatility_term_spread_1h_4h': np.random.normal(0, 0.002, n_samples),
        'momentum_decay_1h': np.random.normal(0, 0.1, n_samples),
        'price_momentum_1h': np.random.normal(0, 0.02, n_samples),
        'volume_weighted_spread': np.random.exponential(0.001, n_samples),
        'order_flow_imbalance': np.random.normal(0, 0.1, n_samples),
    })
    
    # Test directional path features
    path_features = pd.DataFrame({
        'path_trend_r2': np.random.beta(2, 3, n_samples),
        'efficiency_ratio': np.random.beta(1.5, 2, n_samples),
        'impulse_quality': np.random.normal(0, 0.3, n_samples),
        'body_range_ratio': np.random.beta(2, 5, n_samples),
        'traffic_overlap_3h': np.random.beta(1, 4, n_samples),
    })
    
    # Mock HMM
    class MockHMM:
        def __init__(self):
            self.means_ = np.random.normal(0, 1, (3, len(risk_features.columns)))
            self.covars_ = np.array([np.eye(len(risk_features.columns)) * 0.1 for _ in range(3)])
            self.covariance_type = 'diag'
    
    # Test enhanced risk score calculation
    enhanced_risk = integration._simulate_enhanced_risk_score(
        risk_features, MockHMM(), 0, config
    )
    
    # Test directional path risk
    regime_labels = np.random.randint(0, 3, n_samples)
    ohlcv_data = pd.DataFrame({'close': 100 + np.cumsum(np.random.normal(0, 0.5, n_samples))})
    
    directional_risk = integration._simulate_directional_path_risk(
        path_features, regime_labels, ohlcv_data, config
    )
    
    # Validate Phase 1 enhancements
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Enhanced risk score range
    if 0 <= np.nanmean(enhanced_risk) <= 1:
        tprint_success("✓ Enhanced risk score in valid range [0, 1]")
        tests_passed += 1
    else:
        tprint_error("✗ Enhanced risk score out of range")
    
    # Test 2: Directional path risk range
    if 0 <= np.nanmean(directional_risk) <= 1:
        tprint_success("✓ Directional path risk score in valid range [0, 1]")
        tests_passed += 1
    else:
        tprint_error("✗ Directional path risk score out of range")
    
    # Test 3: Volatility term structure impact
    vol_config = config.copy()
    vol_config['volatility_term_weight'] = 0.5
    vol_enhanced = integration._simulate_enhanced_risk_score(risk_features, MockHMM(), 0, vol_config)
    
    if not np.allclose(enhanced_risk, vol_enhanced, rtol=0.1):
        tprint_success("✓ Volatility term structure changes risk scores")
        tests_passed += 1
    else:
        tprint_error("✗ Volatility term structure has no impact")
    
    # Test 4: Momentum decay impact
    mom_config = config.copy()
    mom_config['momentum_decay_weight'] = 0.4
    mom_enhanced = integration._simulate_enhanced_risk_score(risk_features, MockHMM(), 0, mom_config)
    
    if not np.allclose(enhanced_risk, mom_enhanced, rtol=0.1):
        tprint_success("✓ Momentum decay changes risk scores")
        tests_passed += 1
    else:
        tprint_error("✗ Momentum decay has no impact")
    
    # Test 5: Directional bias impact
    bias_config = config.copy()
    bias_config['trend_alignment_weight'] = 0.4
    bias_enhanced = integration._simulate_directional_path_risk(path_features, regime_labels, ohlcv_data, bias_config)
    
    if not np.allclose(directional_risk, bias_enhanced, rtol=0.1):
        tprint_success("✓ Directional bias changes path risk scores")
        tests_passed += 1
    else:
        tprint_error("✗ Directional bias has no impact")
    
    # Test 6: Exponential smoothing effect
    smooth_config = config.copy()
    smooth_config['risk_smoothing_span'] = 20
    smooth_enhanced = integration._simulate_enhanced_risk_score(risk_features, MockHMM(), 0, smooth_config)
    
    smooth_std = np.nanstd(smooth_enhanced)
    original_std = np.nanstd(enhanced_risk)
    
    if smooth_std < original_std:  # Smoothing should reduce variance
        tprint_success("✓ Exponential smoothing reduces variance")
        tests_passed += 1
    else:
        tprint_error("✗ Exponential smoothing doesn't reduce variance")
    
    tprint_info(f"Phase 1 Tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests


def test_phase2_enhancements():
    """Test Phase 2 enhancements: ensemble fusion, regime-specific weights."""
    tprint("🧪 Testing Phase 2 Enhancements")
    tprint("=" * 50)
    
    config = create_example_config()
    integration = EnhancedRiskIntegration(config)
    
    n_samples = 500
    n_regimes = 3
    
    # Generate test data
    risk_scores = np.random.beta(2, 5, n_samples)
    path_scores = np.random.beta(1.5, 4, n_samples)
    market_scores = np.random.beta(2.5, 3, n_samples)
    regime_labels = np.random.randint(0, n_regimes, n_samples)
    returns = np.random.normal(0, 0.02, n_samples)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Ensemble fusion functionality
    try:
        ensemble_scores, metadata = integration.ensemble_fusion.fuse_risk_scores(
            risk_scores=risk_scores,
            path_risk_scores=path_scores,
            market_risk_scores=market_scores,
            regime_labels=regime_labels,
            returns=returns
        )
        
        if 0 <= np.nanmean(ensemble_scores) <= 1 and 'final_weights' in metadata:
            tprint_success("✓ Ensemble fusion produces valid scores and metadata")
            tests_passed += 1
        else:
            tprint_error("✗ Ensemble fusion output invalid")
    except Exception as e:
        tprint_error(f"✗ Ensemble fusion failed: {e}")
    
    # Test 2: Adaptive weight optimization
    try:
        adaptive_config = config.copy()
        adaptive_config['enable_adaptive_weights'] = True
        adaptive_integration = EnhancedRiskIntegration(adaptive_config)
        
        adaptive_scores, adaptive_metadata = adaptive_integration.ensemble_fusion.fuse_risk_scores(
            risk_scores=risk_scores,
            path_risk_scores=path_scores,
            market_risk_scores=market_scores,
            regime_labels=regime_labels,
            returns=returns
        )
        
        if 'adaptive_weights' in adaptive_metadata:
            tprint_success("✓ Adaptive weight optimization works")
            tests_passed += 1
        else:
            tprint_error("✗ Adaptive weight optimization failed")
    except Exception as e:
        tprint_error(f"✗ Adaptive weight optimization failed: {e}")
    
    # Test 3: Regime-specific weights initialization
    try:
        feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        integration.regime_weights.initialize_regime_weights(feature_names, n_regimes)
        
        if len(integration.regime_weights.regime_weights) == n_regimes:
            tprint_success("✓ Regime-specific weights initialization works")
            tests_passed += 1
        else:
            tprint_error("✗ Regime-specific weights initialization failed")
    except Exception as e:
        tprint_error(f"✗ Regime-specific weights initialization failed: {e}")
    
    # Test 4: Different fusion methods
    try:
        methods = ['weighted_average', 'ridge_ensemble', 'quantile_fusion']
        method_results = []
        
        for method in methods:
            method_config = config.copy()
            method_config['fusion_method'] = method
            method_integration = EnhancedRiskIntegration(method_config)
            
            scores, _ = method_integration.ensemble_fusion.fuse_risk_scores(
                risk_scores=risk_scores,
                path_risk_scores=path_scores,
                market_risk_scores=market_scores
            )
            method_results.append(scores)
        
        # Different methods should produce different results
        if not all(np.allclose(method_results[0], result, rtol=0.05) for result in method_results[1:]):
            tprint_success("✓ Different fusion methods produce different results")
            tests_passed += 1
        else:
            tprint_error("✗ Fusion methods produce identical results")
    except Exception as e:
        tprint_error(f"✗ Fusion method testing failed: {e}")
    
    # Test 5: Weight stability metrics
    try:
        stability_metrics = integration.ensemble_fusion.get_weight_stability()
        
        if 'stability' in stability_metrics and 'trend_strength' in stability_metrics:
            tprint_success("✓ Weight stability metrics calculated")
            tests_passed += 1
        else:
            tprint_error("✗ Weight stability metrics missing")
    except Exception as e:
        tprint_error(f"✗ Weight stability metrics failed: {e}")
    
    tprint_info(f"Phase 2 Tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests


def test_integration_performance():
    """Test overall integration performance and expected improvements."""
    tprint("🧪 Testing Integration Performance")
    tprint("=" * 50)
    
    # Run complete integration example
    results = run_integration_example()
    
    performance = results.get('performance_metrics', {})
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Risk score MI improvement estimate
    risk_mi = performance.get('risk_score_mi_estimate', 0)
    if risk_mi >= 0.020:  # Target: 0.025+
        tprint_success(f"✓ Risk score MI estimate: {risk_mi:.4f} (target: ≥0.020)")
        tests_passed += 1
    else:
        tprint_error(f"✗ Risk score MI estimate too low: {risk_mi:.4f}")
    
    # Test 2: Path risk score MI improvement estimate
    path_mi = performance.get('path_score_mi_estimate', 0)
    if path_mi >= 0.015:  # Target: 0.018+
        tprint_success(f"✓ Path risk score MI estimate: {path_mi:.4f} (target: ≥0.015)")
        tests_passed += 1
    else:
        tprint_error(f"✗ Path risk score MI estimate too low: {path_mi:.4f}")
    
    # Test 3: Ensemble fusion gain
    ensemble_gain = performance.get('ensemble_gain', 0)
    if ensemble_gain >= 0.10:  # Target: 15%+
        tprint_success(f"✓ Ensemble fusion gain: {ensemble_gain:.1%} (target: ≥10%)")
        tests_passed += 1
    else:
        tprint_error(f"✗ Ensemble fusion gain too low: {ensemble_gain:.1%}")
    
    # Test 4: Weight stability
    weight_stability = performance.get('weight_stability', 0)
    if weight_stability >= 0.7:  # Target: high stability
        tprint_success(f"✓ Weight stability: {weight_stability:.3f} (target: ≥0.700)")
        tests_passed += 1
    else:
        tprint_error(f"✗ Weight stability too low: {weight_stability:.3f}")
    
    tprint_info(f"Performance Tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests


def main():
    """Run all enhanced risk model tests."""
    tprint("🚀 Enhanced Risk Models Test Suite")
    tprint("=" * 60)
    tprint("Testing Phases 1 & 2 Enhancements")
    tprint("Expected Improvements:")
    tprint("  • Risk Score MI: 0.0131 → 0.025+ (90%+ improvement)")
    tprint("  • Path Risk Score MI: 0.0082 → 0.018+ (120%+ improvement)")
    tprint("  • Ensemble fusion gain: 15%+")
    tprint("  • Better regime differentiation")
    tprint("=" * 60)
    
    # Run all test suites
    phase1_passed = test_phase1_enhancements()
    tprint()
    
    phase2_passed = test_phase2_enhancements()
    tprint()
    
    performance_passed = test_integration_performance()
    tprint()
    
    # Summary
    total_suites = 3
    passed_suites = sum([phase1_passed, phase2_passed, performance_passed])
    
    tprint("=" * 60)
    tprint("FINAL TEST RESULTS")
    tprint("=" * 60)
    tprint_info(f"Phase 1 Enhancements: {'✓ PASSED' if phase1_passed else '✗ FAILED'}")
    tprint_info(f"Phase 2 Enhancements: {'✓ PASSED' if phase2_passed else '✗ FAILED'}")
    tprint_info(f"Performance Tests: {'✓ PASSED' if performance_passed else '✗ FAILED'}")
    tprint()
    
    if passed_suites == total_suites:
        tprint_success(f"🎉 ALL TESTS PASSED ({passed_suites}/{total_suites})")
        tprint_success("Enhanced risk models ready for production!")
        tprint()
        tprint_info("Next steps:")
        tprint_info("  1. Run with real market data")
        tprint_info("  2. Validate MI improvements with actual labels")
        tprint_info("  3. Fine-tune configuration parameters")
        tprint_info("  4. Monitor performance in live trading")
        return 0
    else:
        tprint_error(f"❌ TESTS FAILED ({passed_suites}/{total_suites})")
        tprint_error("Some enhancements need attention before production use")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
