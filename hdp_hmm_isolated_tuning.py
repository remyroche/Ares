#!/usr/bin/env python3
"""
HDP-HMM Smart Tuning (Top-K Local Search Approach) - REFINED v2

Stage 1: 5×5×5 = 125 tests, 60 iterations (refined after α=1.5 dominance)
Stage 2: Top-5 × 5×5×5 = 625 tests, 100 iterations (local refinement around winners)
Stage 3: Top-3 × 5×5×5 = 375 tests, 200 iterations (final precision tuning)

Total: 1125 tests (data-driven exploration with focused parameter ranges)
Time savings: Smart allocation - fewer iterations in Stage 1, more in Stages 2-3

Systematically explores parameter space to find optimal trade-off:
α ∈ [1.5, 2.5]  - REFINED: Focus on α=1.5-2.0 sweet spot (α=1.5 dominated results)
κ ∈ [25, 60]    - Controls regime persistence (focused on high-persistence zone)
γ ∈ [5, 9]      - REFINED: Focus on γ=6-9 sweet spot (10.0 underperformed)

Composite Score Weighting (OPTIMIZED):
- Temporal Smoothness: 45% (reduced from 50% to allow higher-alpha exploration)
- CV Ratio: 30-35% (increased to better reward regime separation)
- Silhouette: 10%
- Balance: 10%

Usage:
    python3 hdp_hmm_isolated_tuning.py              # Use cached features
    python3 hdp_hmm_isolated_tuning.py --clear-cache # Delete cache and regenerate features
"""

import numpy as np
import pandas as pd
import subprocess
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import tprint utilities for consistent logging
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, 
        tprint_error, tprint_progress, tprint_performance
    )
except ImportError:
    # Fallback if tprint not available
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_progress(*args, **kwargs): print("PROGRESS:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='HDP-HMM Iterative Grid Refinement (3-Stage Approach)')
parser.add_argument('--clear-cache', action='store_true', 
                    help='Delete cached features before running (forces feature regeneration)')
args = parser.parse_args()

# Handle cache deletion if requested
if args.clear_cache:
    cache_files = ['hdp_hmm_features_cache.npy', 'hdp_hmm_features_cache.pkl']
    for cache_file in cache_files:
        cache_path = Path(cache_file)
        if cache_path.exists():
            os.remove(cache_path)
            tprint(f"🗑️  Deleted cache file: {cache_file}")
        else:
            tprint(f"ℹ️  Cache file not found (already clean): {cache_file}")
    tprint("")

tprint("=" * 80)
tprint("HDP-HMM Smart Tuning (Top-K Local Search) - REFINED v2")
tprint("=" * 80)
tprint("")
tprint("Stage 1: 5×5×5 = 125 tests @ 60 iters (refined after α=1.5 dominance)")
tprint("Stage 2: Top-5 × 5×5×5 = 625 tests @ 100 iters (local refinement)")
tprint("Stage 3: Top-3 × 5×5×5 = 375 tests @ 200 iters (final precision)")
tprint("Total: 1125 tests (data-driven exploration!)")
tprint("")
tprint("Refinements (based on previous results):")
tprint("  • α ∈ [1.5, 2.5] (α=1.5 dominated top 13 configs)")
tprint("  • κ ∈ [25, 60] (all values performing well)")
tprint("  • γ ∈ [5, 9] (γ=6-9 sweet spot, 10 underperformed)")
tprint("  • Temporal: 45%, CV: 30-35% (balanced weighting)")
tprint("")

# Create outcomes directory
outcomes_dir = Path("outcomes")
outcomes_dir.mkdir(exist_ok=True)

# Global timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Parallel processing configuration (M1-optimized)
MAX_WORKERS = 4  # M1 Mac optimal: 4 workers to avoid overwhelming the system

def run_single_test(params, n_iterations=30):
    """Run a single HDP-HMM test with specified iterations"""
    alpha, kappa, gamma = params
    try:
        cmd = ['python3', 'hdp_hmm_single_test.py', str(alpha), str(kappa), str(gamma), str(n_iterations)]
        
        test_start = datetime.now()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        test_elapsed = (datetime.now() - test_start).total_seconds()
        
        # Parse output - take the LAST line matching SUCCESS/FAILED/ERROR
        output_lines = result.stdout.strip().split('\n')
        result_line = [l for l in output_lines if l.startswith(('SUCCESS|', 'FAILED|', 'ERROR|'))]
        
        if result_line:
            # Use the LAST matching line (in case there's logging noise)
            parts = result_line[-1].split('|')
            status = parts[0]
            
            if status == 'SUCCESS':
                def safe_float(s):
                    if s == 'None' or s == 'nan' or s == '':
                        return 0.0
                    try:
                        return float(s)
                    except:
                        return 0.0
                
                # FIXED: Parse new format with convergence information, cv_ratio, and per-category CVs
                # Format: SUCCESS|α|κ|γ|clusters|silhouette|temporal|balance|between_cv|within_cv|cv_ratio|economic_cv|elapsed|converged|conv_iter|order_flow_cv|microstructure_cv|momentum_cv|volatility_cv|volume_cv|trend_cv|temporal_cv
                result_dict = {
                    'alpha': safe_float(parts[1]),
                    'kappa': safe_float(parts[2]),
                    'gamma': safe_float(parts[3]),
                    'n_clusters': int(safe_float(parts[4])),
                    'silhouette_score': safe_float(parts[5]),
                    'temporal_smoothness': safe_float(parts[6]),
                    'balance_score': safe_float(parts[7]),
                    'between_regime_cv': safe_float(parts[8]),
                    'within_regime_cv': safe_float(parts[9]) if len(parts) > 9 and parts[9] and parts[9] != '0.0' else None,
                    'cv_ratio': safe_float(parts[10]) if len(parts) > 10 else 0.0,
                    'economic_cv_ratio': safe_float(parts[11]) if len(parts) > 11 else 0.0,
                    'runtime': safe_float(parts[12]) if len(parts) > 12 else 0.0,
                    'converged': bool(int(safe_float(parts[13]))) if len(parts) > 13 else False,
                    'convergence_iteration': int(safe_float(parts[14])) if len(parts) > 14 else None,
                    # Per-category CV ratios (7 categories)
                    'cv_order_flow': safe_float(parts[15]) if len(parts) > 15 else 0.0,
                    'cv_microstructure': safe_float(parts[16]) if len(parts) > 16 else 0.0,
                    'cv_momentum': safe_float(parts[17]) if len(parts) > 17 else 0.0,
                    'cv_volatility': safe_float(parts[18]) if len(parts) > 18 else 0.0,
                    'cv_volume': safe_float(parts[19]) if len(parts) > 19 else 0.0,
                    'cv_trend': safe_float(parts[20]) if len(parts) > 20 else 0.0,
                    'cv_temporal': safe_float(parts[21]) if len(parts) > 21 else 0.0,
                    'success': True,
                    'error': None
                }
                
                # Calculate composite score with BALANCED weighting
                # Use the cv_ratio from the output (already calculated)
                cv_ratio = result_dict['cv_ratio']
                temp_score = result_dict['temporal_smoothness']
                
                # OPTIMIZED: Reduced temporal penalty to allow higher-alpha exploration
                temp_contribution = (temp_score ** 1.5) * 0.45  # 45% weight (reduced from 50%)
                
                # SAFEGUARD: Ensure stable CV ratio calculation
                if cv_ratio == 0.0:
                    # Fallback: calculate if not provided
                    if result_dict['within_regime_cv'] is None or result_dict['within_regime_cv'] == 0:
                        composite = 0.0  # Penalize missing/invalid data
                    else:
                        # Use minimum threshold and cap to prevent extreme values
                        within_cv_safe = max(result_dict['within_regime_cv'], 0.01)
                        cv_ratio = min(result_dict['between_regime_cv'] / within_cv_safe, 100.0)
                        # Use log-scaled tanh to prevent CV ratio from dominating the score
                        # tanh(log(1 + cv_ratio)) spreads values more evenly across the range
                        cv_contribution = np.tanh(np.log1p(cv_ratio)) * 0.35  # 35% weight (increased from 30%)
                        composite = (result_dict['silhouette_score'] * 0.10 +      # Cluster quality (10%)
                                   result_dict['balance_score'] * 0.10 +          # Cluster balance (10%)
                                   temp_contribution +                             # Temporal stability (45%, reduced)
                                   cv_contribution)                                # Feature separation (35%, increased)
                else:
                    # Cap cv_ratio to prevent extreme values from dominating the score
                    cv_ratio_capped = min(cv_ratio, 100.0)
                    # Use log-scaled tanh to prevent CV ratio from dominating the score
                    # tanh(log(1 + cv_ratio)) spreads values more evenly across the range
                    cv_contribution = np.tanh(np.log1p(cv_ratio_capped)) * 0.30  # 30% weight (increased from 25%)
                    composite = (result_dict['silhouette_score'] * 0.10 +      # Cluster quality (10%)
                               result_dict['balance_score'] * 0.10 +          # Cluster balance (10%)
                               temp_contribution +                             # Temporal stability (45%, reduced)
                               cv_contribution)                                 # CV ratio contribution (30%, increased)
                result_dict['composite_score'] = composite
                
                return result_dict, test_elapsed, True
            else:
                error_msg = parts[4] if len(parts) > 4 else "Unknown error"
                return {'alpha': alpha, 'kappa': kappa, 'gamma': gamma, 'success': False, 'error': error_msg}, test_elapsed, False
        else:
            return {'alpha': alpha, 'kappa': kappa, 'gamma': gamma, 'success': False, 'error': 'No output'}, test_elapsed, False
            
    except Exception as e:
        return {'alpha': alpha, 'kappa': kappa, 'gamma': gamma, 'success': False, 'error': str(e)}, 0, False

def run_local_search_around_configs(stage_num, base_configs, search_radius_pct=0.10, n_iterations=100, grid_size=(5, 5, 5)):
    """
    Run local search around top-K configurations (SMARTER than full grid).
    
    Args:
        stage_num: Stage number (for logging)
        base_configs: List of configurations to refine (top-K from previous stage)
        search_radius_pct: Search radius as % of parameter range (default 10%)
        n_iterations: Gibbs iterations for this stage
        grid_size: Tuple of (alpha_steps, kappa_steps, gamma_steps) or single int for all
    
    Returns:
        (results, successful_tests, failed_tests)
    """
    # Handle both tuple and int grid_size
    if isinstance(grid_size, (list, tuple)):
        alpha_steps, kappa_steps, gamma_steps = grid_size
    else:
        alpha_steps = kappa_steps = gamma_steps = grid_size
    
    K = len(base_configs)
    tests_per_config = alpha_steps * kappa_steps * gamma_steps
    total_tests = K * tests_per_config
    
    tprint(f"\n")
    tprint(f"{'█'*80}")
    tprint(f"{'█'*80}")
    tprint(f"{'█'*80}")
    tprint(f"🔍 STAGE {stage_num}: Top-{K} Local Search ({n_iterations} Gibbs iterations)")
    tprint(f"{'█'*80}")
    tprint(f"{'█'*80}")
    tprint(f"{'█'*80}")
    tprint(f"Strategy: {alpha_steps}×{kappa_steps}×{gamma_steps} local grid around each of top-{K} configs")
    tprint(f"Total tests: {K} × {tests_per_config} = {total_tests}")
    tprint(f"Search radius: ±{search_radius_pct*100:.0f}% of original range")
    
    results = []
    successful_tests = 0
    failed_tests = 0
    start_time_stage = datetime.now()
    
    # Parameter ranges from Stage 1 (REFINED to match new search space)
    alpha_full_range = 2.5 - 1.5  # 1.0
    kappa_full_range = 60.0 - 25.0  # 35.0
    gamma_full_range = 9.0 - 5.0  # 4.0
    
    test_counter = 0
    
    for config_idx, base_config in enumerate(base_configs, 1):
        alpha_base = base_config['alpha']
        kappa_base = base_config['kappa']
        gamma_base = base_config['gamma']
        
        tprint(f"\n🎯 Refining config {config_idx}/{K}: α={alpha_base:.3f}, κ={kappa_base:.3f}, γ={gamma_base:.3f} (Score={base_config.get('composite_score', 0):.3f})")
        
        # Create local ranges (±radius% around base)
        alpha_width = alpha_full_range * search_radius_pct
        kappa_width = kappa_full_range * search_radius_pct
        gamma_width = gamma_full_range * search_radius_pct
        
        # FIX: Symmetric clamping - ensure equal exploration on both sides when possible
        # If near boundary, shift the window to maintain symmetry
        def create_symmetric_range(base, width, min_val, max_val):
            ideal_min = base - width
            ideal_max = base + width
            
            # Check if we need to clamp
            if ideal_min < min_val:
                # Shift window right to maintain width
                actual_min = min_val
                actual_max = min(max_val, min_val + 2 * width)
            elif ideal_max > max_val:
                # Shift window left to maintain width
                actual_max = max_val
                actual_min = max(min_val, max_val - 2 * width)
            else:
                # No clamping needed
                actual_min = ideal_min
                actual_max = ideal_max
            
            return actual_min, actual_max
        
        alpha_min, alpha_max = create_symmetric_range(alpha_base, alpha_width, 1.5, 2.5)
        kappa_min, kappa_max = create_symmetric_range(kappa_base, kappa_width, 25.0, 60.0)
        gamma_min, gamma_max = create_symmetric_range(gamma_base, gamma_width, 5.0, 9.0)
        
        # Warn if base is very close to boundary (asymmetric search unavoidable)
        alpha_near_edge = (alpha_base - 1.5) < alpha_width or (2.5 - alpha_base) < alpha_width
        kappa_near_edge = (kappa_base - 25.0) < kappa_width or (60.0 - kappa_base) < kappa_width
        gamma_near_edge = (gamma_base - 5.0) < gamma_width or (9.0 - gamma_base) < gamma_width
        
        if alpha_near_edge or kappa_near_edge or gamma_near_edge:
            tprint(f"   ⚠️ Base near boundary - search window adjusted for symmetry")
        
        tprint(f"   Local search: α[{alpha_min:.2f}, {alpha_max:.2f}], κ[{kappa_min:.1f}, {kappa_max:.1f}], γ[{gamma_min:.2f}, {gamma_max:.2f}]")
        
        # Generate local grid
        alpha_values = np.linspace(alpha_min, alpha_max, alpha_steps)
        kappa_values = np.linspace(kappa_min, kappa_max, kappa_steps)
        gamma_values = np.linspace(gamma_min, gamma_max, gamma_steps)
        
        local_configs = list(itertools.product(alpha_values, kappa_values, gamma_values))
        
        for alpha, kappa, gamma in local_configs:
            test_counter += 1
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            result_dict, test_elapsed, success = run_single_test((alpha, kappa, gamma), n_iterations)
            
            if success:
                results.append(result_dict)
                successful_tests += 1
                
                composite = result_dict['composite_score']
                temporal = result_dict['temporal_smoothness']
                balance = result_dict['balance_score']
                # SAFEGUARD: Ensure minimum within_cv and cap cv_ratio to prevent extreme values
                within_cv = max(result_dict.get('within_regime_cv') or 0.01, 0.01)
                cv_ratio_raw = min(result_dict['between_regime_cv'] / within_cv, 100.0)  # Cap at 100x
                cv_ratio_scaled = np.tanh(np.log1p(cv_ratio_raw))  # Same scaling as in composite score
                
                tprint(f"[{current_time}] ✅ Test {test_counter}/{total_tests} | "
                      f"α={alpha:.2f}, κ={kappa:.1f}, γ={gamma:.1f} | "
                      f"Clusters={result_dict['n_clusters']}, Score={composite:.3f} "
                      f"(Temp={temporal:.2f}, Bal={balance:.2f}, CV={cv_ratio_scaled:.2f})")
            else:
                results.append(result_dict)
                failed_tests += 1
                tprint(f"[{current_time}] ❌ Test {test_counter}/{total_tests} | "
                      f"α={alpha:.2f}, κ={kappa:.1f}, γ={gamma:.1f} | "
                      f"Error: {result_dict.get('error', 'Unknown')}")
            
            # Progress update every 10 tests
            if test_counter % 10 == 0:
                elapsed = (datetime.now() - start_time_stage).total_seconds()
                avg_time = elapsed / test_counter
                remaining = (total_tests - test_counter) * avg_time
                tprint(f"📊 Progress: {test_counter}/{total_tests} ({100*test_counter/total_tests:.1f}%) | "
                      f"Success: {successful_tests}, Failed: {failed_tests} | "
                      f"ETA: {remaining/60:.1f}m")
    
    # Stage summary
    stage_time = (datetime.now() - start_time_stage).total_seconds()
    
    # Calculate convergence statistics
    converged_count = sum(1 for r in results if r.get('success') and r.get('converged', False))
    converged_rate = (converged_count / successful_tests * 100) if successful_tests > 0 else 0
    
    # Calculate average convergence iteration for converged models
    converged_iterations = [r.get('convergence_iteration', n_iterations) 
                           for r in results if r.get('success') and r.get('converged', False)]
    avg_conv_iter = np.mean(converged_iterations) if converged_iterations else n_iterations
    
    tprint(f"\n")
    tprint(f"{'='*80}")
    tprint(f"{'='*80}")
    tprint(f"📊 STAGE {stage_num} COMPLETE")
    tprint(f"{'='*80}")
    tprint(f"{'='*80}")
    tprint(f"⏱️  Stage Time: {stage_time/60:.1f} minutes")
    tprint(f"✅ Successful: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)")
    tprint(f"❌ Failed: {failed_tests}/{total_tests}")
    tprint(f"🎯 Converged: {converged_count}/{successful_tests} ({converged_rate:.1f}% of successful)")
    if converged_count > 0:
        tprint(f"⚡ Avg Convergence: {avg_conv_iter:.0f}/{n_iterations} iterations ({avg_conv_iter/n_iterations*100:.0f}%)")
    tprint(f"{'='*80}")
    tprint(f"\n")
    
    return results, successful_tests, failed_tests


def run_grid_stage(stage_num, alpha_range, kappa_range, gamma_range, n_iterations=30, alpha_steps=4, kappa_steps=6, gamma_steps=4):
    """
    Run a single stage of grid search with specified Gibbs iterations
    """
    tprint(f"\n")
    tprint(f"{'█'*80}")
    tprint(f"{'█'*80}")
    tprint(f"{'█'*80}")
    tprint(f"🔍 STAGE {stage_num}: Grid Search ({n_iterations} Gibbs iterations)")
    tprint(f"{'█'*80}")
    tprint(f"{'█'*80}")
    tprint(f"{'█'*80}")
    tprint(f"α: {alpha_steps} steps in [{alpha_range[0]:.3f}, {alpha_range[1]:.3f}]")
    tprint(f"κ: {kappa_steps} steps in [{kappa_range[0]:.3f}, {kappa_range[1]:.3f}]")
    tprint(f"γ: {gamma_steps} steps in [{gamma_range[0]:.3f}, {gamma_range[1]:.3f}]")
    
    # Generate grid
    alpha_values = np.linspace(alpha_range[0], alpha_range[1], alpha_steps)
    kappa_values = np.linspace(kappa_range[0], kappa_range[1], kappa_steps)
    gamma_values = np.linspace(gamma_range[0], gamma_range[1], gamma_steps)
    
    test_configs = list(itertools.product(alpha_values, kappa_values, gamma_values))
    total_tests = len(test_configs)
    
    tprint(f"Total combinations: {total_tests} ({alpha_steps}×{kappa_steps}×{gamma_steps})")
    # Estimate: ~0.5s per Gibbs iteration + 15s import overhead
    estimated_per_test = (n_iterations * 0.5) + 15
    tprint(f"Estimated time: ~{total_tests*estimated_per_test/60:.1f} minutes (~{estimated_per_test:.0f}s per test)")
    
    results = []
    successful_tests = 0
    failed_tests = 0
    start_time_stage = datetime.now()
    
    # Run tests SEQUENTIALLY (pyhsmm can't handle parallel processes!)
    # But with optimizations: cached features + minimal iterations
    tprint(f"⚙️  Running tests sequentially (pyhsmm limitation)\n")
    
    for i, (alpha, kappa, gamma) in enumerate(test_configs, 1):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        result_dict, test_elapsed, success = run_single_test((alpha, kappa, gamma), n_iterations)
        
        if success:
            results.append(result_dict)
            successful_tests += 1
            
            composite = result_dict['composite_score']
            temporal = result_dict['temporal_smoothness']
            balance = result_dict['balance_score']
            cv_ratio_raw = result_dict['between_regime_cv'] / (result_dict['within_regime_cv'] + 1e-9)
            cv_ratio_scaled = np.tanh(np.log1p(cv_ratio_raw))  # Same scaling as in composite score
            
            # Enhanced feedback with key metrics
            tprint(f"[{current_time}] ✅ Test {i}/{total_tests} | "
                  f"α={alpha:.2f}, κ={kappa:.1f}, γ={gamma:.1f} | "
                  f"Clusters={result_dict['n_clusters']}, Score={composite:.3f} "
                  f"(Temp={temporal:.2f}, Bal={balance:.2f}, CV={cv_ratio_scaled:.2f})")
        else:
            results.append(result_dict)
            failed_tests += 1
            tprint(f"[{current_time}] ❌ Test {i}/{total_tests} | "
                  f"α={alpha:.2f}, κ={kappa:.1f}, γ={gamma:.1f} | "
                  f"Error: {result_dict.get('error', 'Unknown')}")
        
        # Progress update every 10 tests
        if i % 10 == 0:
            elapsed = (datetime.now() - start_time_stage).total_seconds()
            avg_time = elapsed / i
            remaining = (total_tests - i) * avg_time
            tprint(f"📊 Progress: {i}/{total_tests} ({100*i/total_tests:.1f}%) | "
                  f"Success: {successful_tests}, Failed: {failed_tests} | "
                  f"ETA: {remaining/60:.1f}m")
        
        # Checkpoint every 50 tests
        if i % 50 == 0 and results:
            try:
                checkpoint_df = pd.DataFrame(results)
                # FIX: Add timestamp to prevent overwriting checkpoints from different runs
                checkpoint_path = outcomes_dir / f"stage{stage_num}_checkpoint_{i}_{timestamp}.csv"
                checkpoint_df.to_csv(checkpoint_path, index=False)
                tprint(f"\n💾 Checkpoint: {checkpoint_path} ({len(results)} results)\n")
            except Exception as e:
                tprint(f"\n⚠️ Checkpoint failed: {e}\n")
    
    # Stage summary
    stage_time = (datetime.now() - start_time_stage).total_seconds()
    
    # Calculate convergence statistics
    converged_count = sum(1 for r in results if r.get('success') and r.get('converged', False))
    converged_rate = (converged_count / successful_tests * 100) if successful_tests > 0 else 0
    
    # Calculate average convergence iteration for converged models
    converged_iterations = [r.get('convergence_iteration', n_iterations) 
                           for r in results if r.get('success') and r.get('converged', False)]
    avg_conv_iter = np.mean(converged_iterations) if converged_iterations else n_iterations
    
    tprint(f"\n")
    tprint(f"{'='*80}")
    tprint(f"{'='*80}")
    tprint(f"📊 STAGE {stage_num} COMPLETE")
    tprint(f"{'='*80}")
    tprint(f"{'='*80}")
    tprint(f"⏱️  Stage Time: {stage_time/60:.1f} minutes")
    tprint(f"✅ Successful: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)")
    tprint(f"❌ Failed: {failed_tests}/{total_tests}")
    tprint(f"🎯 Converged: {converged_count}/{successful_tests} ({converged_rate:.1f}% of successful)")
    if converged_count > 0:
        tprint(f"⚡ Avg Convergence: {avg_conv_iter:.0f}/{n_iterations} iterations ({avg_conv_iter/n_iterations*100:.0f}%)")
    tprint(f"{'='*80}")
    tprint(f"\n")
    
    return results, successful_tests, failed_tests


# ============================================================================
# STAGE 1: REFINED Exploration (Based on α=1.5 dominance in results)
# ============================================================================
alpha_range_1 = (1.5, 2.5)   # REFINED: Focus on optimal α zone (1.5 was best)
kappa_range_1 = (25.0, 60.0) # OPTIMIZED: Focus on high-persistence zone
gamma_range_1 = (5.0, 9.0)   # REFINED: Focus on γ sweet spot (6-9 performed best)

results_stage1, success_1, fail_1 = run_grid_stage(
    1, alpha_range_1, kappa_range_1, gamma_range_1,
    n_iterations=60,  # INCREASED: Better kappa convergence (was 30)
    alpha_steps=5, kappa_steps=5, gamma_steps=5  # 5×5×5 = 125 tests
)

if not results_stage1 or success_1 == 0:
    tprint_error("Stage 1 had no successful results. Stopping.")
    sys.exit(1)

# Find best configuration from Stage 1
stage1_df = pd.DataFrame([r for r in results_stage1 if r['success']])
stage1_df = stage1_df.sort_values('composite_score', ascending=False)

# Reorder columns: composite_score first, then params, then metrics, then per-category CVs
column_order = [
    'composite_score', 'alpha', 'kappa', 'gamma', 'n_clusters',
    'silhouette_score', 'temporal_smoothness', 'balance_score',
    'between_regime_cv', 'within_regime_cv', 'cv_ratio', 'economic_cv_ratio',
    # Per-category CV ratios (shows which feature types contribute to regime separation)
    'cv_order_flow', 'cv_microstructure', 'cv_momentum', 'cv_volatility', 
    'cv_volume', 'cv_trend', 'cv_temporal',
    'runtime', 'converged', 'convergence_iteration', 'success', 'error'
]
# Only include columns that exist in the dataframe
column_order = [col for col in column_order if col in stage1_df.columns]
stage1_df = stage1_df[column_order]

best_stage1 = stage1_df.iloc[0]
tprint(f"\n🏆 Best from Stage 1:")
tprint(f"   α={best_stage1['alpha']:.4f}, κ={best_stage1['kappa']:.4f}, γ={best_stage1['gamma']:.4f}")
tprint(f"   Composite Score: {best_stage1['composite_score']:.4f}")
tprint(f"   Clusters: {int(best_stage1['n_clusters'])}")

# Save Stage 1 results
stage1_csv = outcomes_dir / f"hdp_hmm_stage1_{timestamp}.csv"
stage1_df.to_csv(stage1_csv, index=False)
tprint(f"\n💾 Stage 1 results saved: {stage1_csv}")

# ============================================================================
# ============================================================================
# ============================================================================
# STAGE 2: Top-5 Local Search (SMARTER than full grid!)
# ============================================================================
# ============================================================================
# ============================================================================
# Select top 5 configurations from Stage 1
TOP_K_STAGE2 = 5
top_k_stage1 = stage1_df.nlargest(TOP_K_STAGE2, 'composite_score')

tprint(f"\n🏆 Top {TOP_K_STAGE2} configs from Stage 1:")
for idx, row in top_k_stage1.iterrows():
    tprint(f"   #{idx+1}: α={row['alpha']:.3f}, κ={row['kappa']:.3f}, γ={row['gamma']:.3f} → Score={row['composite_score']:.3f}, Clusters={int(row['n_clusters'])}")

# ENHANCEMENT: Adaptive search radius based on score variance
# If top configs have similar scores → search wider (flat landscape)
# If top configs have diverse scores → search tighter (sharp peaks)
score_std = top_k_stage1['composite_score'].std()
score_range = top_k_stage1['composite_score'].max() - top_k_stage1['composite_score'].min()

if score_std < 0.05 or score_range < 0.10:
    # Flat landscape → search wider to escape plateau
    stage2_radius = 0.15
    tprint(f"   📊 Low score variance ({score_std:.3f}) → Wider search (±15%)")
else:
    # Sharp peaks → search tighter to refine
    stage2_radius = 0.10
    tprint(f"   📊 Good score variance ({score_std:.3f}) → Standard search (±10%)")

# Run local search (5×5×5 = 125 tests) around each of top-5
# Total: 5 × 125 = 625 tests
results_stage2, success_2, fail_2 = run_local_search_around_configs(
    stage_num=2,
    base_configs=top_k_stage1.to_dict('records'),
    search_radius_pct=stage2_radius,  # Adaptive: 10% or 15% based on variance
    n_iterations=100,                  # Higher quality for refinement
    grid_size=(5, 5, 5)                # 5×5×5 = 125 tests per config
)

if not results_stage2 or success_2 == 0:
    tprint_warning("Stage 2 had no successful results. Using Stage 1 best.")
    best_overall = best_stage1
else:
    # Find best from Stage 2
    stage2_df = pd.DataFrame([r for r in results_stage2 if r['success']])
    stage2_df = stage2_df.sort_values('composite_score', ascending=False)
    
    # Reorder columns: composite_score first
    column_order = [col for col in column_order if col in stage2_df.columns]
    stage2_df = stage2_df[column_order]
    
    best_stage2 = stage2_df.iloc[0]
    tprint(f"\n🏆 Best from Stage 2:")
    tprint(f"   α={best_stage2['alpha']:.4f}, κ={best_stage2['kappa']:.4f}, γ={best_stage2['gamma']:.4f}")
    tprint(f"   Composite Score: {best_stage2['composite_score']:.4f}")
    tprint(f"   Clusters: {int(best_stage2['n_clusters'])}")
    
    # Save Stage 2 results
    stage2_csv = outcomes_dir / f"hdp_hmm_stage2_{timestamp}.csv"
    stage2_df.to_csv(stage2_csv, index=False)
    tprint(f"\n💾 Stage 2 results saved: {stage2_csv}")
    
    # ============================================================================
    # ============================================================================
    # ============================================================================
    # STAGE 3: Top-3 Ultra-Precision (Final Refinement)
    # ============================================================================
    # ============================================================================
    # ============================================================================
    # Select top 3 configurations from Stage 2 for ultra-precise refinement
    TOP_K_STAGE3 = 3
    top_k_stage2 = stage2_df.nlargest(TOP_K_STAGE3, 'composite_score')
    
    tprint(f"\n🏆 Top {TOP_K_STAGE3} configs from Stage 2:")
    for idx, row in top_k_stage2.iterrows():
        tprint(f"   #{idx+1}: α={row['alpha']:.3f}, κ={row['kappa']:.3f}, γ={row['gamma']:.3f} → Score={row['composite_score']:.3f}, Clusters={int(row['n_clusters'])}")
    
    # ENHANCEMENT: Adaptive search radius for Stage 3
    score_std_stage2 = top_k_stage2['composite_score'].std()
    score_range_stage2 = top_k_stage2['composite_score'].max() - top_k_stage2['composite_score'].min()
    
    if score_std_stage2 < 0.03 or score_range_stage2 < 0.05:
        # Very similar scores → might need wider search
        stage3_radius = 0.08
        tprint(f"   📊 Very low score variance ({score_std_stage2:.3f}) → Wider precision search (±8%)")
    else:
        # Diverse scores → tight refinement
        stage3_radius = 0.05
        tprint(f"   📊 Good score variance ({score_std_stage2:.3f}) → Tight precision search (±5%)")
    
    # Run ultra-precise local search (5×5×5 = 125 tests) around each of top-3
    # Total: 3 × 125 = 375 tests with 200 iterations for maximum precision
    results_stage3, success_3, fail_3 = run_local_search_around_configs(
        stage_num=3,
        base_configs=top_k_stage2.to_dict('records'),
        search_radius_pct=stage3_radius,  # Adaptive: 5% or 8% based on variance
        n_iterations=200,                  # Maximum quality for final precision
        grid_size=(5, 5, 5)                # 5×5×5 = 125 tests per config
    )
    
    if not results_stage3 or success_3 == 0:
        tprint_warning("Stage 3 had no successful results. Using Stage 2 best.")
        best_overall = best_stage2
    else:
        # Find best from Stage 3
        stage3_df = pd.DataFrame([r for r in results_stage3 if r['success']])
        stage3_df = stage3_df.sort_values('composite_score', ascending=False)
        
        # Reorder columns: composite_score first
        column_order_stage3 = [col for col in column_order if col in stage3_df.columns]
        stage3_df = stage3_df[column_order_stage3]
        
        best_stage3 = stage3_df.iloc[0]
        best_overall = best_stage3
        
        tprint(f"\n🏆 Best from Stage 3:")
        tprint(f"   α={best_stage3['alpha']:.4f}, κ={best_stage3['kappa']:.4f}, γ={best_stage3['gamma']:.4f}")
        tprint(f"   Composite Score: {best_stage3['composite_score']:.4f}")
        tprint(f"   Clusters: {int(best_stage3['n_clusters'])}")
        
        # Save Stage 3 results
        stage3_csv = outcomes_dir / f"hdp_hmm_stage3_{timestamp}.csv"
        stage3_df.to_csv(stage3_csv, index=False)
        tprint(f"\n💾 Stage 3 results saved: {stage3_csv}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
# FIX: Safer conditional logic for combining results
all_results = results_stage1.copy()
all_results.extend(results_stage2)
if success_3 > 0 and 'results_stage3' in locals() and results_stage3:
    all_results.extend(results_stage3)
all_results_df = pd.DataFrame(all_results)

# Save combined results
combined_csv = outcomes_dir / f"hdp_hmm_iterative_all_results_{timestamp}.csv"
all_results_df.to_csv(combined_csv, index=False)
tprint(f"\n💾 Combined results saved: {combined_csv}")

tprint(f"\n{'='*80}")
tprint("🎉 ITERATIVE GRID REFINEMENT COMPLETE!")
tprint(f"{'='*80}")
tprint(f"\nStage 1: {success_1} successful, {fail_1} failed")
tprint(f"Stage 2: {success_2} successful, {fail_2} failed")
# FIX: Safer check for Stage 3 completion
stage3_ran = 'success_3' in locals() and success_3 is not None
if stage3_ran:
    tprint(f"Stage 3: {success_3} successful, {fail_3} failed")
tprint(f"\nTotal Tests: {len(all_results)}")
tprint(f"Total Successful: {success_1 + success_2 + (success_3 if stage3_ran else 0)}")

# Show final best
tprint(f"\n{'='*80}")
tprint("🏆 FINAL BEST CONFIGURATION")
tprint(f"{'='*80}")
tprint(f"\nParameters:")
tprint(f"   α (alpha) = {best_overall['alpha']:.4f}")
tprint(f"   κ (kappa) = {best_overall['kappa']:.4f}")
tprint(f"   γ (gamma) = {best_overall['gamma']:.4f}")
tprint(f"\nMetrics:")
tprint(f"   Composite Score: {best_overall.get('composite_score', 0.0):.4f}")
tprint(f"   Clusters: {int(best_overall.get('n_clusters', 0))}")
tprint(f"   Silhouette: {best_overall.get('silhouette_score', 0.0):.4f}")
tprint(f"   Temporal Smoothness: {best_overall.get('temporal_smoothness', 0.0):.4f}")
tprint(f"   Balance Score: {best_overall.get('balance_score', 0.0):.4f}")
tprint(f"   Between-Regime CV: {best_overall.get('between_regime_cv', 0.0):.4f}")
tprint(f"   Within-Regime CV: {best_overall.get('within_regime_cv', 0.0):.4f}")
tprint(f"   CV Ratio (Feature): {best_overall.get('between_regime_cv', 0.0) / (best_overall.get('within_regime_cv', 1.0) + 1e-9):.4f}")
tprint(f"   Economic CV Ratio: {best_overall.get('economic_cv_ratio', 0.0):.4f}")
tprint(f"   Runtime: {best_overall.get('runtime', 0.0):.1f}s")

tprint(f"\n{'='*80}")
tprint("✅ All results saved to outcomes/ directory")
tprint(f"{'='*80}\n")
