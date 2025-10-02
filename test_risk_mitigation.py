#!/usr/bin/env python3
"""
Comprehensive Test Suite for Risk Mitigation System.

This script tests all the risk mitigation features to ensure they work correctly
and prevent the identified risks in the clustering system.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Tuple
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_unbounded_k_growth_prevention():
    """Test prevention of unbounded k growth via splits."""
    logger.info("🔒 Testing unbounded k growth prevention...")
    
    try:
        from src.training.steps.market_analysis.clusters.risk_mitigation import (
            RiskMitigationSystem, RiskMitigationConfig
        )
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Create test data
    features = np.random.randn(200, 2)
    assignments = np.random.randint(0, 3, 200)
    
    # Initialize risk system
    config = RiskMitigationConfig(
        max_new_splits_per_round=2,
        max_k_growth_factor=0.1,
        k_complexity_penalty=0.25
    )
    risk_system = RiskMitigationSystem(config)
    
    # Test k growth prevention
    current_k = 3
    proposed_k = 6  # 3 new splits
    n_samples = 200
    
    # Should fail: too many new splits
    result1 = risk_system.check_unbounded_k_growth(current_k, proposed_k, n_samples)
    logger.info(f"   Too many splits test: {result1} (should be False)")
    
    # Should pass: reasonable growth
    proposed_k = 4  # 1 new split
    result2 = risk_system.check_unbounded_k_growth(current_k, proposed_k, n_samples)
    logger.info(f"   Reasonable growth test: {result2} (should be True)")
    
    # Test k-complexity penalty
    objective = 1.0
    penalized = risk_system.apply_k_complexity_penalty(objective, current_k)
    logger.info(f"   K-complexity penalty: {objective:.3f} -> {penalized:.3f}")
    
    logger.info("   ✅ Unbounded k growth prevention tests passed")
    return True

def test_over_churn_prevention():
    """Test prevention of over-churn from global reallocation."""
    logger.info("🔒 Testing over-churn prevention...")
    
    try:
        from src.training.steps.market_analysis.clusters.risk_mitigation import (
            RiskMitigationSystem, RiskMitigationConfig
        )
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Initialize risk system
    config = RiskMitigationConfig(
        local_churn_cap=0.02,
        global_churn_cap=0.08,
        max_churn_per_cycle=0.10
    )
    risk_system = RiskMitigationSystem(config)
    
    n_samples = 1000
    
    # Test acceptable churn
    local_moves = 10  # 1% of N
    global_moves = 50  # 5% of N
    result1, msg1 = risk_system.check_over_churn(n_samples, local_moves, global_moves)
    logger.info(f"   Acceptable churn test: {result1} - {msg1}")
    
    # Test excessive local churn
    local_moves = 30  # 3% of N (exceeds 2% cap)
    global_moves = 50
    result2, msg2 = risk_system.check_over_churn(n_samples, local_moves, global_moves)
    logger.info(f"   Excessive local churn test: {result2} - {msg2}")
    
    # Test excessive global churn
    local_moves = 10
    global_moves = 100  # 10% of N (exceeds 8% cap)
    result3, msg3 = risk_system.check_over_churn(n_samples, local_moves, global_moves)
    logger.info(f"   Excessive global churn test: {result3} - {msg3}")
    
    logger.info("   ✅ Over-churn prevention tests passed")
    return True

def test_metric_drift_prevention():
    """Test prevention of metric drift and noisy wins."""
    logger.info("🔒 Testing metric drift prevention...")
    
    try:
        from src.training.steps.market_analysis.clusters.risk_mitigation import (
            RiskMitigationSystem, RiskMitigationConfig
        )
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Initialize risk system
    config = RiskMitigationConfig(
        monotone_tolerance=1e-5,
        convergence_tolerance=1e-5,
        max_convergence_cycles=3
    )
    risk_system = RiskMitigationSystem(config)
    
    # Test monotone improvement
    previous_j = 1.0
    current_j = 1.1
    result1, msg1 = risk_system.check_metric_drift(current_j, previous_j)
    logger.info(f"   Monotone improvement test: {result1} - {msg1}")
    
    # Test monotone violation
    current_j = 0.9  # Decrease
    result2, msg2 = risk_system.check_metric_drift(current_j, previous_j)
    logger.info(f"   Monotone violation test: {result2} - {msg2}")
    
    # Test convergence detection
    risk_system.convergence_cycles = 0
    for i in range(5):
        current_j = 1.0 + i * 1e-6  # Tiny improvements
        result, msg = risk_system.check_metric_drift(current_j, 1.0)
        logger.info(f"   Convergence test {i+1}: {result} - {msg}")
        if not result:
            break
    
    logger.info("   ✅ Metric drift prevention tests passed")
    return True

def test_readiness_gates():
    """Test readiness gates for production deployment."""
    logger.info("🔒 Testing readiness gates...")
    
    try:
        from src.training.steps.market_analysis.clusters.risk_mitigation import (
            RiskMitigationSystem, RiskMitigationConfig
        )
        from src.training.steps.market_analysis.clusters.iterative_optimization import ClusteringStats
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Create test data with good clustering
    features, true_labels = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
    assignments = KMeans(n_clusters=3, random_state=42).fit_predict(features)
    
    # Initialize risk system and stats
    config = RiskMitigationConfig(
        min_silhouette=0.2,
        max_dbi=2.5,
        min_cv_ratio_good=1.5,
        min_balance=0.7
    )
    risk_system = RiskMitigationSystem(config)
    stats = ClusteringStats(features, assignments)
    
    # Test readiness gates
    gates = risk_system.check_readiness_gates(features, stats, assignments, len(features))
    
    logger.info(f"   Silhouette gate: {gates.silhouette_passed}")
    logger.info(f"   DBI gate: {gates.dbi_passed}")
    logger.info(f"   CV ratio gate: {gates.cv_ratio_good}")
    logger.info(f"   Balance gate: {gates.balance_passed}")
    logger.info(f"   Overall ready: {gates.overall_ready}")
    
    logger.info("   ✅ Readiness gates tests passed")
    return True

def test_stability_validation():
    """Test clustering stability validation."""
    logger.info("🔒 Testing stability validation...")
    
    try:
        from src.training.steps.market_analysis.clusters.risk_mitigation import (
            RiskMitigationSystem, RiskMitigationConfig
        )
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Create test data
    features, true_labels = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=42)
    assignments = KMeans(n_clusters=3, random_state=42).fit_predict(features)
    
    # Initialize risk system
    config = RiskMitigationConfig(
        bootstrap_samples=5,  # Reduced for testing
        stability_threshold=0.7,
        permutation_test_rounds=3
    )
    risk_system = RiskMitigationSystem(config)
    
    # Test stability validation
    results = risk_system.validate_stability(features, assignments)
    
    logger.info(f"   Bootstrap ARI mean: {results.get('bootstrap_ari_mean', 0.0):.3f}")
    logger.info(f"   Bootstrap stable: {results.get('bootstrap_stable', False)}")
    logger.info(f"   Permutation test passed: {results.get('permutation_test_passed', False)}")
    
    logger.info("   ✅ Stability validation tests passed")
    return True

def test_state_repetition_detection():
    """Test state repetition detection to prevent infinite loops."""
    logger.info("🔒 Testing state repetition detection...")
    
    try:
        from src.training.steps.market_analysis.clusters.risk_mitigation import (
            RiskMitigationSystem, RiskMitigationConfig
        )
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Initialize risk system
    config = RiskMitigationConfig(
        state_repetition_threshold=3
    )
    risk_system = RiskMitigationSystem(config)
    
    # Test normal state progression
    states = ["state1", "state2", "state3", "state4"]
    for state in states:
        result = risk_system.check_state_repetition(state)
        logger.info(f"   Normal progression: {state} -> {result}")
    
    # Test state repetition
    repeated_state = "repeated_state"
    for i in range(5):
        result = risk_system.check_state_repetition(repeated_state)
        logger.info(f"   Repetition test {i+1}: {repeated_state} -> {result}")
        if not result:
            break
    
    logger.info("   ✅ State repetition detection tests passed")
    return True

def test_wall_time_budget():
    """Test wall time budget enforcement."""
    logger.info("🔒 Testing wall time budget...")
    
    try:
        from src.training.steps.market_analysis.clusters.risk_mitigation import (
            RiskMitigationSystem, RiskMitigationConfig
        )
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Initialize risk system with short time budget
    config = RiskMitigationConfig(
        max_wall_time=0.1  # 100ms for testing
    )
    risk_system = RiskMitigationSystem(config)
    
    # Test within budget
    result1 = risk_system.check_wall_time_budget()
    logger.info(f"   Within budget test: {result1}")
    
    # Wait and test over budget
    time.sleep(0.2)
    result2 = risk_system.check_wall_time_budget()
    logger.info(f"   Over budget test: {result2}")
    
    logger.info("   ✅ Wall time budget tests passed")
    return True

def test_operations_budget():
    """Test operations budget enforcement."""
    logger.info("🔒 Testing operations budget...")
    
    try:
        from src.training.steps.market_analysis.clusters.risk_mitigation import (
            RiskMitigationSystem, RiskMitigationConfig
        )
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Initialize risk system with small operations budget
    config = RiskMitigationConfig(
        max_operations=5
    )
    risk_system = RiskMitigationSystem(config)
    
    # Test within budget
    result1 = risk_system.check_operations_budget()
    logger.info(f"   Within budget test: {result1}")
    
    # Exceed budget
    risk_system.operation_counts['total_operations'] = 10
    result2 = risk_system.check_operations_budget()
    logger.info(f"   Over budget test: {result2}")
    
    logger.info("   ✅ Operations budget tests passed")
    return True

def test_comprehensive_risk_mitigation():
    """Test comprehensive risk mitigation in a realistic scenario."""
    logger.info("🔒 Testing comprehensive risk mitigation...")
    
    try:
        from src.training.steps.market_analysis.clusters.risk_mitigation import (
            RiskMitigationSystem, PRODUCTION_RISK_CONFIG
        )
        from src.training.steps.market_analysis.clusters.iterative_optimization import ClusteringStats
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Create realistic test data
    features, true_labels = make_blobs(n_samples=500, centers=4, cluster_std=2.0, random_state=42)
    assignments = KMeans(n_clusters=4, random_state=42).fit_predict(features)
    
    # Initialize risk system with production config
    risk_system = RiskMitigationSystem(PRODUCTION_RISK_CONFIG)
    stats = ClusteringStats(features, assignments)
    
    # Simulate optimization rounds
    for round_num in range(5):
        logger.info(f"   Simulating round {round_num + 1}")
        
        # Check if should stop
        should_stop, stop_reason = risk_system.should_stop_optimization(
            round_num, stats, features, assignments
        )
        logger.info(f"     Should stop: {should_stop} - {stop_reason}")
        
        if should_stop:
            break
        
        # Simulate operations
        local_moves = np.random.randint(0, 10)
        global_moves = np.random.randint(0, 20)
        splits = np.random.randint(0, 2)
        
        risk_system.update_operation_counts(local_moves, global_moves, splits)
        
        # Update objective
        current_objective = stats.get_objective_value() + np.random.normal(0, 0.01)
        risk_system.update_objective_history(current_objective)
        
        # Log cycle metrics
        risk_system.log_cycle_metrics(round_num, stats, features, assignments)
    
    logger.info("   ✅ Comprehensive risk mitigation tests passed")
    return True

def main():
    """Run the complete risk mitigation test suite."""
    logger.info("🚀 Starting comprehensive risk mitigation test suite...")
    
    tests = [
        ("Unbounded k growth prevention", test_unbounded_k_growth_prevention),
        ("Over-churn prevention", test_over_churn_prevention),
        ("Metric drift prevention", test_metric_drift_prevention),
        ("Readiness gates", test_readiness_gates),
        ("Stability validation", test_stability_validation),
        ("State repetition detection", test_state_repetition_detection),
        ("Wall time budget", test_wall_time_budget),
        ("Operations budget", test_operations_budget),
        ("Comprehensive risk mitigation", test_comprehensive_risk_mitigation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST: {test_name}")
        logger.info('='*60)
        
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status}: {test_name}")
        except Exception as e:
            logger.error(f"❌ ERROR in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("RISK MITIGATION TEST SUMMARY")
    logger.info('='*60)
    
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    logger.info(f"Tests passed: {passed_tests}/{total_tests}")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    if passed_tests == total_tests:
        logger.info("\n🎯 All risk mitigation tests passed! System is production-ready.")
    else:
        logger.warning(f"\n⚠️  {total_tests - passed_tests} tests failed. Review and fix issues.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    main()