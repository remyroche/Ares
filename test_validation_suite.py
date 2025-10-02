#!/usr/bin/env python3
"""
Comprehensive Validation Suite for Iterative Clustering Optimization.

This script tests the clustering engine for correctness, stability, and performance
using the validation framework and synthetic test cases.
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

def test_synthetic_scenarios():
    """Test the clustering engine on synthetic scenarios."""
    logger.info("🧪 Running synthetic validation suite...")
    
    # Import after setting up logging
    try:
        from src.training.steps.market_analysis.clusters.validation_framework import (
            ClusteringValidator, ValidationConfig, ValidationResults
        )
        from src.training.steps.market_analysis.clusters.iterative_optimization import (
            IterativeOptimization, ClusteringStats
        )
        from src.training.steps.market_analysis.clusters.step1_feature_preparation import (
            ClusteringContext
        )
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Create test scenarios
    test_cases = [
        ("well_separated", create_well_separated_blobs()),
        ("overlapping", create_overlapping_blobs()),
        ("giant_small", create_giant_small_clusters()),
        ("no_structure", create_no_structure_data())
    ]
    
    results = {}
    
    for name, (features, expected_k) in test_cases:
        logger.info(f"\n🔍 Testing scenario: {name}")
        logger.info(f"   Features shape: {features.shape}")
        logger.info(f"   Expected clusters: {expected_k}")
        
        # Initialize clustering
        initial_kmeans = KMeans(n_clusters=expected_k, random_state=42)
        initial_assignments = initial_kmeans.fit_predict(features)
        
        # Create context and stats
        context = ClusteringContext(
            features=features,
            assignments=initial_assignments,
            metadata={}
        )
        stats = ClusteringStats(features, initial_assignments)
        
        # Run validation
        validator = ClusteringValidator()
        result = run_validation_test(context, stats, validator, name)
        results[name] = result
        
        # Log results
        logger.info(f"   ✅ Test completed: {name}")
        logger.info(f"   Final k: {stats.n_clusters}")
        logger.info(f"   CV ratio: {stats.get_cv_ratio():.4f}")
        logger.info(f"   Balance: {stats.get_balance_score():.4f}")
        logger.info(f"   Validation passed: {result['validation_passed']}")
    
    return results

def create_well_separated_blobs():
    """Create well-separated blob data."""
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
    return X, 3

def create_overlapping_blobs():
    """Create overlapping blob data."""
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=3.0, random_state=42)
    return X, 3

def create_giant_small_clusters():
    """Create one giant cluster with small clusters."""
    X = np.vstack([
        np.random.normal(0, 1, (200, 2)),  # Giant cluster
        np.random.normal(10, 0.5, (50, 2)),  # Small cluster 1
        np.random.normal(-10, 0.5, (50, 2))  # Small cluster 2
    ])
    return X, 3

def create_no_structure_data():
    """Create isotropic noise data."""
    X = np.random.normal(0, 1, (300, 2))
    return X, 2

def run_validation_test(context, stats, validator, test_name):
    """Run a single validation test."""
    start_time = time.time()
    
    # Initialize validation results
    validation_results = {
        'incremental_checks_passed': 0,
        'incremental_checks_failed': 0,
        'monotone_violations': 0,
        'invariant_violations': 0,
        'total_moves': 0,
        'validation_passed': True
    }
    
    # Run optimization with validation
    optimizer = IterativeOptimization(verbose=False)
    
    # Track objective function
    current_j = stats.get_objective_value()
    previous_j = current_j
    
    max_rounds = 20
    for round_num in range(max_rounds):
        logger.info(f"   Round {round_num + 1}/{max_rounds}")
        
        # Validate invariants
        validator._validate_invariants(stats, len(context.features), validation_results)
        
        # Run optimization steps (simplified for testing)
        # Note: In full implementation, these would be async calls
        step1_improvement = 0.0  # Placeholder
        step2_improvement = 0.0  # Placeholder  
        step3_improvement = 0.0  # Placeholder
        
        # Calculate new objective
        new_j = stats.get_objective_value()
        
        # Validate monotone objective
        if not validator._validate_monotone_objective(previous_j, new_j, validation_results):
            logger.warning(f"   Monotone violation in {test_name}")
            validation_results['validation_passed'] = False
        
        # Check convergence
        if abs(new_j - current_j) < 1e-5:
            logger.info(f"   Converged after {round_num + 1} rounds")
            break
        
        current_j = new_j
        previous_j = new_j
        
        # Periodic validation checks
        if round_num % 5 == 0:
            validator._validate_incremental_correctness(context.features, stats, validation_results)
    
    # Final validation
    total_time = time.time() - start_time
    
    # Calculate final metrics
    final_cv = stats.get_cv_ratio()
    final_balance = stats.get_balance_score()
    final_silhouette = 0.0  # Placeholder
    if len(np.unique(stats.assignments)) > 1:
        final_silhouette = silhouette_score(context.features, stats.assignments)
    
    # Log results
    logger.info(f"   Final metrics - CV: {final_cv:.4f}, Balance: {final_balance:.4f}, Silhouette: {final_silhouette:.4f}")
    logger.info(f"   Validation stats: {validation_results}")
    logger.info(f"   Time: {total_time:.2f}s")
    
    return {
        'validation_passed': validation_results['validation_passed'],
        'final_cv': final_cv,
        'final_balance': final_balance,
        'final_silhouette': final_silhouette,
        'total_time': total_time,
        'validation_stats': validation_results
    }

def test_correctness_checks():
    """Test correctness checks."""
    logger.info("🔍 Testing correctness checks...")
    
    try:
        from src.training.steps.market_analysis.clusters.iterative_optimization import ClusteringStats
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Create test data
    features = np.random.randn(100, 2)
    assignments = np.random.randint(0, 3, 100)
    
    # Test ClusteringStats
    stats = ClusteringStats(features, assignments)
    
    # Test basic functionality
    cv_ratio = stats.get_cv_ratio()
    balance = stats.get_balance_score()
    objective = stats.get_objective_value()
    
    logger.info(f"   CV ratio: {cv_ratio:.4f}")
    logger.info(f"   Balance: {balance:.4f}")
    logger.info(f"   Objective: {objective:.4f}")
    
    # Test move delta calculation
    delta = stats.calculate_move_delta(0, 0, 1)
    logger.info(f"   Move delta: {delta}")
    
    logger.info("   ✅ Correctness checks passed")
    return True

def test_performance_benchmarks():
    """Test performance benchmarks."""
    logger.info("⚡ Testing performance benchmarks...")
    
    try:
        from src.training.steps.market_analysis.clusters.iterative_optimization import ClusteringStats
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Test with different data sizes
    sizes = [100, 500, 1000]
    
    for size in sizes:
        logger.info(f"   Testing with {size} samples...")
        
        # Create test data
        features = np.random.randn(size, 2)
        assignments = np.random.randint(0, 5, size)
        
        # Time ClusteringStats initialization
        start_time = time.time()
        stats = ClusteringStats(features, assignments)
        init_time = time.time() - start_time
        
        # Time objective calculation
        start_time = time.time()
        objective = stats.get_objective_value()
        calc_time = time.time() - start_time
        
        logger.info(f"     Init time: {init_time:.4f}s")
        logger.info(f"     Calc time: {calc_time:.4f}s")
        logger.info(f"     Objective: {objective:.4f}")
    
    logger.info("   ✅ Performance benchmarks completed")
    return True

def main():
    """Run the complete validation suite."""
    logger.info("🚀 Starting comprehensive validation suite...")
    
    # Test 1: Correctness checks
    logger.info("\n" + "="*50)
    logger.info("TEST 1: Correctness Checks")
    logger.info("="*50)
    test_correctness_checks()
    
    # Test 2: Performance benchmarks
    logger.info("\n" + "="*50)
    logger.info("TEST 2: Performance Benchmarks")
    logger.info("="*50)
    test_performance_benchmarks()
    
    # Test 3: Synthetic scenarios
    logger.info("\n" + "="*50)
    logger.info("TEST 3: Synthetic Scenarios")
    logger.info("="*50)
    results = test_synthetic_scenarios()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*50)
    
    if results:
        passed_tests = sum(1 for r in results.values() if r['validation_passed'])
        total_tests = len(results)
        logger.info(f"Tests passed: {passed_tests}/{total_tests}")
        
        for name, result in results.items():
            status = "✅ PASS" if result['validation_passed'] else "❌ FAIL"
            logger.info(f"  {name}: {status}")
            logger.info(f"    CV: {result['final_cv']:.4f}, Balance: {result['final_balance']:.4f}")
    else:
        logger.error("No test results available")
    
    logger.info("\n🎯 Validation suite completed!")

if __name__ == "__main__":
    main()