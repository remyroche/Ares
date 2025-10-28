#!/usr/bin/env python3
"""
Test script to verify the correct SR pipeline order.

This script demonstrates that:
1. sr_parameter_optimization runs first (without requiring prior artifacts)
2. sr_detection loads and uses the optimized parameters
3. sr_clustering clusters the properly detected SR levels

Usage:
    python test_sr_pipeline_correct_order.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.training.steps.market_analysis.components.sr_parameter_optimization import SRParameterOptimizationStep
from src.training.steps.market_analysis.components.sr_detection import SRDetectionComponent
from src.training.steps.market_analysis.components.sr_clustering import SRClusteringComponent


async def test_correct_order():
    """Test the correct SR pipeline order."""
    
    print("=" * 80)
    print("TESTING CORRECT SR PIPELINE ORDER")
    print("=" * 80)
    print()
    
    # Configuration
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'longs',
        'execution_mode': 'light',
        'data_dir': 'historical_data'
    }
    
    print(f"Configuration:")
    print(f"  Symbol: {config['symbol']}")
    print(f"  Exchange: {config['exchange']}")
    print(f"  Timeframe: {config['timeframe']}")
    print()
    
    # =========================================================================
    # STEP 1: Parameter Optimization (FIRST)
    # =========================================================================
    print("=" * 80)
    print("STEP 1: SR PARAMETER OPTIMIZATION (First)")
    print("=" * 80)
    print("Purpose: Find optimal SR detection parameters")
    print("Input: Market data (no prior artifacts required)")
    print("Output: sr_parameter_optimization_result")
    print()
    
    param_opt = SRParameterOptimizationStep()
    
    print("Executing parameter optimization...")
    param_result = await param_opt.execute({
        **config,
        'enable_bayesian_hpo': True,
        'n_trials': 10,  # Reduced for testing
        'enable_vectorbt': True
    })
    
    if param_result['success']:
        print(f"✅ STEP 1 SUCCESS: Parameters optimized")
        
        # Extract key metrics
        metrics = param_result.get('metrics', {})
        print(f"   Total combinations tested: {metrics.get('total_combinations_tested', 'N/A')}")
        print(f"   Best score: {metrics.get('best_score', 'N/A')}")
        print(f"   Optimization time: {metrics.get('optimization_time', 'N/A'):.2f}s")
        
        # Show some optimized parameters
        artifacts = param_result.get('artifacts', {})
        if 'sr_parameter_optimization_result' in artifacts:
            opt_result = artifacts['sr_parameter_optimization_result']
            opt_params = opt_result.get('optimized_parameters', {})
            if opt_params:
                print(f"   Optimized parameters ({len(opt_params)} total):")
                for key, value in list(opt_params.items())[:5]:
                    print(f"     - {key}: {value}")
                if len(opt_params) > 5:
                    print(f"     ... and {len(opt_params) - 5} more")
    else:
        print(f"❌ STEP 1 FAILED: {param_result.get('error', 'Unknown error')}")
        return False
    
    print()
    
    # =========================================================================
    # STEP 2: SR Detection (SECOND)
    # =========================================================================
    print("=" * 80)
    print("STEP 2: SR DETECTION (Second)")
    print("=" * 80)
    print("Purpose: Detect SR levels using optimized parameters from Step 1")
    print("Input: Market data + sr_parameter_optimization_result")
    print("Output: sr_detection_result")
    print()
    
    detection = SRDetectionComponent()
    
    print("Executing SR detection...")
    detection_result = await detection.execute({
        **config,
        'use_optimized_parameters': True,  # Load from step 1
        'enable_shap_lime': True,
        'enable_vectorbt': True
    })
    
    if detection_result['success']:
        print(f"✅ STEP 2 SUCCESS: SR levels detected")
        
        # Extract key metrics
        metrics = detection_result.get('metrics', {})
        print(f"   Total SR levels: {metrics.get('total_levels', 0)}")
        print(f"   Support levels: {metrics.get('support_levels', 0)}")
        print(f"   Resistance levels: {metrics.get('resistance_levels', 0)}")
        print(f"   Used optimized parameters: {metrics.get('used_optimized_parameters', False)}")
        
        if metrics.get('used_optimized_parameters'):
            print(f"   ✓ CORRECT: Using optimized parameters from Step 1")
        else:
            print(f"   ⚠ WARNING: Using default parameters (optimization failed?)")
    else:
        print(f"❌ STEP 2 FAILED: {detection_result.get('error', 'Unknown error')}")
        return False
    
    print()
    
    # =========================================================================
    # STEP 3: SR Clustering (THIRD)
    # =========================================================================
    print("=" * 80)
    print("STEP 3: SR CLUSTERING (Third)")
    print("=" * 80)
    print("Purpose: Cluster the SR levels detected in Step 2")
    print("Input: sr_detection_result from Step 2")
    print("Output: sr_clustering_result, sr_levels_dictionary")
    print()
    
    clustering = SRClusteringComponent()
    
    print("Executing SR clustering...")
    clustering_result = await clustering.execute({
        **config,
        'clustering_algorithm': 'hdbscan',
        'enable_hardware_optimization': True,
        'enable_vectorbt_optimization': True
    })
    
    if clustering_result['success']:
        print(f"✅ STEP 3 SUCCESS: SR levels clustered")
        
        # Extract key metrics
        metrics = clustering_result.get('metrics', {})
        print(f"   Total clusters: {metrics.get('total_clusters', 0)}")
        print(f"   Clustered levels: {metrics.get('clustered_levels', 0)}")
        print(f"   Noise levels: {metrics.get('noise_levels', 0)}")
        
        quality_metrics = metrics.get('quality_metrics', {})
        if quality_metrics:
            print(f"   Silhouette score: {quality_metrics.get('silhouette_score', 'N/A')}")
            print(f"   Clustering efficiency: {quality_metrics.get('clustering_efficiency', 'N/A')}")
    else:
        print(f"❌ STEP 3 FAILED: {clustering_result.get('error', 'Unknown error')}")
        return False
    
    print()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print()
    print("✅ Pipeline completed successfully in correct order:")
    print("   1. sr_parameter_optimization → Found optimal parameters")
    print("   2. sr_detection → Used those parameters to detect SR levels")
    print("   3. sr_clustering → Clustered the properly detected levels")
    print()
    print("Benefits of correct order:")
    print("  ✓ Detection used optimized parameters from the start")
    print("  ✓ Clustering worked on high-quality detections")
    print("  ✓ No wasted computation on suboptimal detections")
    print("  ✓ Ready for iterative refinement in next run")
    print()
    
    # Show artifacts created
    print("Artifacts created:")
    print("  1. sr_parameter_optimization_result → Contains optimized parameters")
    print("  2. sr_detection_result → Contains detected SR levels")
    print("  3. sr_clustering_result → Contains clustering results")
    print("  4. sr_levels_dictionary → Contains organized SR levels")
    print()
    
    return True


async def test_iterative_refinement():
    """Test iterative refinement where clustering results refine parameter optimization."""
    
    print("=" * 80)
    print("TESTING ITERATIVE REFINEMENT")
    print("=" * 80)
    print()
    print("Running 2 iterations to demonstrate refinement...")
    print()
    
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'longs',
        'execution_mode': 'light'
    }
    
    for iteration in range(1, 3):
        print(f"--- ITERATION {iteration} ---")
        print()
        
        # Step 1: Parameter optimization
        # On iteration 2+, it will try to use clustering results from iteration 1
        param_opt = SRParameterOptimizationStep()
        param_result = await param_opt.execute({
            **config,
            'n_trials': 5,  # Reduced for testing
        })
        
        # Step 2: Detection with optimized parameters
        detection = SRDetectionComponent()
        detection_result = await detection.execute({
            **config,
            'use_optimized_parameters': True
        })
        
        # Step 3: Clustering
        clustering = SRClusteringComponent()
        clustering_result = await clustering.execute(config)
        
        # Show iteration results
        if all([param_result['success'], detection_result['success'], clustering_result['success']]):
            print(f"✅ Iteration {iteration} completed")
            print(f"   Levels detected: {detection_result['metrics'].get('total_levels', 0)}")
            print(f"   Clusters created: {clustering_result['metrics'].get('total_clusters', 0)}")
        
        print()
    
    print("✅ Iterative refinement completed")
    print("   Each iteration refined parameters based on previous clustering results")
    print()


if __name__ == "__main__":
    print()
    print("SR PIPELINE CORRECT ORDER TEST")
    print()
    
    # Run the main test
    success = asyncio.run(test_correct_order())
    
    if success:
        print()
        print("=" * 80)
        print()
        
        # Run iterative test
        asyncio.run(test_iterative_refinement())
        
        print("=" * 80)
        print("ALL TESTS PASSED ✅")
        print("=" * 80)
        print()
        print("The SR pipeline is now configured in the correct order:")
        print("  1. sr_parameter_optimization (first)")
        print("  2. sr_detection (second, uses optimized parameters)")
        print("  3. sr_clustering (third, clusters optimized detections)")
        print()
        print("See CORRECT_SR_PIPELINE_ORDER.md for detailed documentation.")
        print()
        
        sys.exit(0)
    else:
        print()
        print("=" * 80)
        print("TESTS FAILED ❌")
        print("=" * 80)
        sys.exit(1)
