#!/usr/bin/env python3
"""
HDP-HMM Iterative Grid Refinement (3-Stage Approach)

Stage 1: 4×6×4 = 96 tests (coarse exploration)
Stage 2: 4×6×4 = 96 tests (refine around best from Stage 1)
Stage 3: 4×6×4 = 96 tests (final refinement)

Total: 288 tests instead of 810 (64% reduction)
"""

import numpy as np
import pandas as pd
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import itertools

# Import tprint utilities for consistent logging
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, 
        tprint_error, tprint_progress, tprint_performance
    )
except ImportError:
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_progress(*args, **kwargs): print("PROGRESS:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

tprint("=" * 80)
tprint("HDP-HMM Iterative Grid Refinement (3-Stage Approach)")
tprint("=" * 80)
tprint("")
tprint("Stage 1: 4×6×4 = 96 tests (coarse exploration)")
tprint("Stage 2: 4×6×4 = 96 tests (refine around best)")
tprint("Stage 3: 4×6×4 = 96 tests (final refinement)")
tprint("Total: 288 tests (much faster than 810!)")
tprint("")

# Ensure outcomes directory exists
outcomes_dir = Path("outcomes")
outcomes_dir.mkdir(exist_ok=True)

def run_grid_stage(stage_num, alpha_range, kappa_range, gamma_range, alpha_steps=4, kappa_steps=6, gamma_steps=4):
    """
    Run a single stage of grid search
    """
    tprint(f"\n{'='*80}")
    tprint(f"🔍 STAGE {stage_num}: Grid Search")
    tprint(f"{'='*80}")
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
    tprint(f"Estimated time: ~{total_tests*20/60:.1f} minutes\n")
    
    results = []
    successful_tests = 0
    failed_tests = 0
    start_time_stage = datetime.now()
    
    for i, (alpha, kappa, gamma) in enumerate(test_configs, 1):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tprint(f"\n{'='*60}")
        tprint(f"[{current_time}] Stage {stage_num}, Test {i}/{total_tests}")
        tprint_progress(i, total_tests, f"α={alpha:.3f}, κ={kappa:.3f}, γ={gamma:.3f}")
        tprint(f"{'='*60}")
        tprint(f"   Progress: {i}/{total_tests} ({100*i/total_tests:.1f}%)")
        elapsed = (datetime.now() - start_time_stage).total_seconds()
        if i > 1:
            avg_time = elapsed / (i - 1)
            remaining = (total_tests - i) * avg_time
            tprint(f"   Elapsed: {elapsed/60:.1f}m, Estimated remaining: {remaining/60:.1f}m")
        tprint(f"   Success: {successful_tests}, Failed: {failed_tests}")
        tprint("")
        
        try:
            cmd = ['python3', 'hdp_hmm_single_test.py', str(alpha), str(kappa), str(gamma)]
            
            test_start = datetime.now()
            tprint(f"   ⏱️  Started at: {test_start.strftime('%H:%M:%S')}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
            test_elapsed = (datetime.now() - test_start).total_seconds()
            
            # Parse output
            output_lines = result.stdout.strip().split('\n')
            result_line = [l for l in output_lines if l.startswith(('SUCCESS|', 'FAILED|', 'ERROR|'))]
            
            if result_line:
                parts = result_line[0].split('|')
                status = parts[0]
                
                if status == 'SUCCESS':
                    # Parse: SUCCESS|alpha|kappa|gamma|n_clusters|silhouette|temporal|balance|between_cv|within_cv|economic_cv|runtime
                    def safe_float(s):
                        if s == 'None' or s == 'nan' or s == '':
                            return 0.0
                        try:
                            return float(s)
                        except:
                            return 0.0
                    
                    result_dict = {
                        'alpha': safe_float(parts[1]),
                        'kappa': safe_float(parts[2]),
                        'gamma': safe_float(parts[3]),
                        'n_clusters': int(safe_float(parts[4])),
                        'silhouette_score': safe_float(parts[5]),
                        'temporal_smoothness': safe_float(parts[6]),
                        'balance_score': safe_float(parts[7]),
                        'between_regime_cv': safe_float(parts[8]),
                        'within_regime_cv': safe_float(parts[9]) or 1.0,
                        'economic_cv_ratio': safe_float(parts[10]) if len(parts) > 10 else 0.0,
                        'runtime': safe_float(parts[11]) if len(parts) > 11 else safe_float(parts[10]),
                        'success': True,
                        'error': None
                    }
                    
                    # Calculate composite score
                    cv_ratio = result_dict['between_regime_cv'] / (result_dict['within_regime_cv'] + 1e-9)
                    # Use log-scaled tanh to prevent CV ratio from dominating
                    cv_contribution = np.tanh(np.log1p(cv_ratio)) * 0.2
                    composite = (result_dict['silhouette_score'] * 0.3 + 
                               result_dict['balance_score'] * 0.3 + 
                               result_dict['temporal_smoothness'] * 0.2 +
                               cv_contribution)
                    result_dict['composite_score'] = composite
                    
                    results.append(result_dict)
                    successful_tests += 1
                    
                    # Quick feedback
                    completion_time = datetime.now().strftime('%H:%M:%S')
                    tprint_success(f"Completed at {completion_time} (took {test_elapsed:.1f}s)")
                    tprint(f"   Clusters: {result_dict['n_clusters']}")
                    tprint(f"   Silhouette: {result_dict['silhouette_score']:.4f}")
                    tprint(f"   Temporal: {result_dict['temporal_smoothness']:.4f} {'✅' if result_dict['temporal_smoothness'] >= 0.70 else '⚠️'}")
                    tprint(f"   Balance:  {result_dict['balance_score']:.4f} {'✅' if result_dict['balance_score'] >= 0.40 else '⚠️'}")
                    tprint(f"   CV Ratio (Feat): {cv_ratio:.4f} {'✅' if cv_ratio >= 1.0 else '⚠️'}")
                    tprint(f"   CV Ratio (Econ): {result_dict.get('economic_cv_ratio', 0.0):.4f}")
                    tprint(f"   Composite Score: {composite:.4f}")
                    tprint_performance("HMM Runtime", result_dict['runtime'])
                    
                else:
                    # FAILED or ERROR
                    error_msg = parts[4] if len(parts) > 4 else "Unknown error"
                    result_dict = {
                        'alpha': alpha,
                        'kappa': kappa,
                        'gamma': gamma,
                        'success': False,
                        'error': error_msg
                    }
                    results.append(result_dict)
                    failed_tests += 1
                    tprint_error(f"Test failed: {error_msg}")
            else:
                # No parseable output
                failed_tests += 1
                tprint_error(f"No output from subprocess (completed in {test_elapsed:.1f}s)")
                results.append({
                    'alpha': alpha,
                    'kappa': kappa,
                    'gamma': gamma,
                    'success': False,
                    'error': 'No output'
                })
                
        except Exception as e:
            failed_tests += 1
            tprint_error(f"SUBPROCESS ERROR: {e}")
            results.append({
                'alpha': alpha,
                'kappa': kappa,
                'gamma': gamma,
                'success': False,
                'error': str(e)
            })
    
    stage_time = (datetime.now() - start_time_stage).total_seconds()
    
    # Stage summary
    tprint(f"\n{'='*80}")
    tprint(f"📊 STAGE {stage_num} COMPLETE")
    tprint(f"{'='*80}")
    tprint(f"⏱️  Stage Time: {stage_time/60:.1f} minutes")
    tprint(f"✅ Successful: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)")
    tprint(f"❌ Failed: {failed_tests}/{total_tests}")
    
    return results, successful_tests, failed_tests


# ============================================================================
# STAGE 1: Coarse Exploration
# ============================================================================
alpha_range_1 = (1.0, 1.9)
kappa_range_1 = (5.0, 35.0)
gamma_range_1 = (3.0, 6.0)

results_stage1, success_1, fail_1 = run_grid_stage(
    1, alpha_range_1, kappa_range_1, gamma_range_1,
    alpha_steps=4, kappa_steps=6, gamma_steps=4
)

if not results_stage1 or success_1 == 0:
    tprint_error("Stage 1 had no successful results. Stopping.")
    sys.exit(1)

# Find best configuration from Stage 1
stage1_df = pd.DataFrame([r for r in results_stage1 if r['success']])
stage1_df = stage1_df.sort_values('composite_score', ascending=False)

best_stage1 = stage1_df.iloc[0]
tprint(f"\n🏆 Best from Stage 1:")
tprint(f"   α={best_stage1['alpha']:.4f}, κ={best_stage1['kappa']:.4f}, γ={best_stage1['gamma']:.4f}")
tprint(f"   Composite Score: {best_stage1['composite_score']:.4f}")
tprint(f"   Clusters: {int(best_stage1['n_clusters'])}")

# Save Stage 1 results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
stage1_csv = outcomes_dir / f"hdp_hmm_stage1_{timestamp}.csv"
stage1_df.to_csv(stage1_csv, index=False)
tprint(f"\n💾 Stage 1 results saved: {stage1_csv}")

# ============================================================================
# STAGE 2: Refine Around Best
# ============================================================================
# Create narrower ranges around best from Stage 1
alpha_best = best_stage1['alpha']
kappa_best = best_stage1['kappa']
gamma_best = best_stage1['gamma']

# Define refinement window (50% of Stage 1 range)
alpha_width = (alpha_range_1[1] - alpha_range_1[0]) * 0.25  # ±25% of original range
kappa_width = (kappa_range_1[1] - kappa_range_1[0]) * 0.25
gamma_width = (gamma_range_1[1] - gamma_range_1[0]) * 0.25

alpha_range_2 = (max(alpha_range_1[0], alpha_best - alpha_width), 
                 min(alpha_range_1[1], alpha_best + alpha_width))
kappa_range_2 = (max(kappa_range_1[0], kappa_best - kappa_width), 
                 min(kappa_range_1[1], kappa_best + kappa_width))
gamma_range_2 = (max(gamma_range_1[0], gamma_best - gamma_width), 
                 min(gamma_range_1[1], gamma_best + gamma_width))

tprint(f"\n📍 Stage 2 will explore:")
tprint(f"   α: [{alpha_range_2[0]:.3f}, {alpha_range_2[1]:.3f}] (centered on {alpha_best:.3f})")
tprint(f"   κ: [{kappa_range_2[0]:.3f}, {kappa_range_2[1]:.3f}] (centered on {kappa_best:.3f})")
tprint(f"   γ: [{gamma_range_2[0]:.3f}, {gamma_range_2[1]:.3f}] (centered on {gamma_best:.3f})")

results_stage2, success_2, fail_2 = run_grid_stage(
    2, alpha_range_2, kappa_range_2, gamma_range_2,
    alpha_steps=4, kappa_steps=6, gamma_steps=4
)

if not results_stage2 or success_2 == 0:
    tprint_warning("Stage 2 had no successful results. Using Stage 1 best.")
    best_overall = best_stage1
else:
    # Find best from Stage 2
    stage2_df = pd.DataFrame([r for r in results_stage2 if r['success']])
    stage2_df = stage2_df.sort_values('composite_score', ascending=False)
    
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
    # STAGE 3: Final Refinement
    # ============================================================================
    alpha_best = best_stage2['alpha']
    kappa_best = best_stage2['kappa']
    gamma_best = best_stage2['gamma']
    
    # Narrow window (25% of Stage 2 range)
    alpha_width = (alpha_range_2[1] - alpha_range_2[0]) * 0.25
    kappa_width = (kappa_range_2[1] - kappa_range_2[0]) * 0.25
    gamma_width = (gamma_range_2[1] - gamma_range_2[0]) * 0.25
    
    alpha_range_3 = (max(alpha_range_1[0], alpha_best - alpha_width), 
                     min(alpha_range_1[1], alpha_best + alpha_width))
    kappa_range_3 = (max(kappa_range_1[0], kappa_best - kappa_width), 
                     min(kappa_range_1[1], kappa_best + kappa_width))
    gamma_range_3 = (max(gamma_range_1[0], gamma_best - gamma_width), 
                     min(gamma_range_1[1], gamma_best + gamma_width))
    
    tprint(f"\n📍 Stage 3 will explore:")
    tprint(f"   α: [{alpha_range_3[0]:.3f}, {alpha_range_3[1]:.3f}] (centered on {alpha_best:.3f})")
    tprint(f"   κ: [{kappa_range_3[0]:.3f}, {kappa_range_3[1]:.3f}] (centered on {kappa_best:.3f})")
    tprint(f"   γ: [{gamma_range_3[0]:.3f}, {gamma_range_3[1]:.3f}] (centered on {gamma_best:.3f})")
    
    results_stage3, success_3, fail_3 = run_grid_stage(
        3, alpha_range_3, kappa_range_3, gamma_range_3,
        alpha_steps=4, kappa_steps=6, gamma_steps=4
    )
    
    if not results_stage3 or success_3 == 0:
        tprint_warning("Stage 3 had no successful results. Using Stage 2 best.")
        best_overall = best_stage2
    else:
        # Find best from Stage 3
        stage3_df = pd.DataFrame([r for r in results_stage3 if r['success']])
        stage3_df = stage3_df.sort_values('composite_score', ascending=False)
        
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
total_time = (datetime.now() - datetime.now()).total_seconds()  # Will calculate properly
all_results = results_stage1 + results_stage2 + (results_stage3 if 'results_stage3' in locals() else [])

tprint(f"\n{'='*80}")
tprint("🎉 ITERATIVE GRID REFINEMENT COMPLETE!")
tprint(f"{'='*80}")
tprint(f"\nStage 1: {success_1} successful, {fail_1} failed")
tprint(f"Stage 2: {success_2} successful, {fail_2} failed")
if 'success_3' in locals():
    tprint(f"Stage 3: {success_3} successful, {fail_3} failed")
tprint(f"\nTotal Tests: {len(all_results)}")
tprint(f"Total Successful: {success_1 + success_2 + (success_3 if 'success_3' in locals() else 0)}")

# Save combined results
all_results_df = pd.DataFrame(all_results)
combined_csv = outcomes_dir / f"hdp_hmm_iterative_all_results_{timestamp}.csv"
all_results_df.to_csv(combined_csv, index=False)
tprint(f"\n💾 Combined results saved: {combined_csv}")

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

tprint(f"\n{'='*80}")
tprint("✅ All results saved to outcomes/ directory")
tprint(f"{'='*80}\n")

