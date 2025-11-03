#!/usr/bin/env python3
"""
HDP-HMM Progressive Parameter Tuning (3-Stage Coarse-to-Fine)
Uses isolated subprocess execution with parallel workers (M1-optimized)

Strategy:
- Stage 1: Coarse exploration (3×5×3 = 45 tests)
- Stage 2: Refine around top 2 configs (2×45 = 90 tests if close)
- Stage 3: Fine-tune best config (3×5×3 = 45 tests)

Total: ~135-180 tests vs 810 in full grid (83% reduction)
With 2 workers + early stopping: ~15-20 minutes vs ~8 hours
"""

import numpy as np
import pandas as pd
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import itertools
from multiprocessing import Pool
from typing import List, Tuple, Dict, Any, Optional

# Import hardware optimizations
try:
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    HARDWARE_OPT_AVAILABLE = True
except ImportError:
    HARDWARE_OPT_AVAILABLE = False
    print("⚠️ Hardware optimizations not available - using default settings")

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


# ============================================================================
# Configuration
# ============================================================================

# Determine optimal worker count (2 workers or use hardware optimizer)
if HARDWARE_OPT_AVAILABLE:
    cpu_optimizer = get_m1_cpu_optimizer()
    # Force 2 workers as requested
    NUM_WORKERS = 2
    tprint_info(f"Using M1-optimized configuration: {NUM_WORKERS} workers")
    tprint_info(f"   M1 Generation: {cpu_optimizer.m1_generation}")
    tprint_info(f"   Performance cores: {cpu_optimizer.performance_cores}")
else:
    NUM_WORKERS = 2
    tprint_warning(f"Using default configuration: {NUM_WORKERS} workers")

# Early stopping thresholds
EARLY_STOP_CONFIG = {
    'min_silhouette': -0.1,        # Abort test if silhouette too low
    'min_clusters': 2,              # Minimum valid clusters
    'max_clusters': 12,             # Maximum valid clusters
    'min_composite_stage1': 0.25,  # Minimum composite to continue to Stage 2
    'improvement_threshold': 0.02,  # Minimum improvement % to continue
    'close_threshold': 0.05,        # Threshold to explore 2nd best config
}

# Create outcomes directory
outcomes_dir = Path("outcomes")
outcomes_dir.mkdir(exist_ok=True)


# ============================================================================
# Helper Functions
# ============================================================================

def safe_float(s):
    """Safely convert string to float."""
    if s in ('None', 'nan', ''):
        return 0.0
    try:
        return float(s)
    except:
        return 0.0


def run_single_test(params: Tuple[float, float, float, int]) -> Dict[str, Any]:
    """
    Run a single HDP-HMM test in isolated subprocess.
    
    Args:
        params: Tuple of (alpha, kappa, gamma, test_number)
    
    Returns:
        Dictionary with test results
    """
    alpha, kappa, gamma, test_num = params
    
    try:
        # Run test in isolated subprocess
        cmd = ['python3', 'hdp_hmm_single_test.py', str(alpha), str(kappa), str(gamma)]
        
        test_start = datetime.now()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd='.'
        )
        test_elapsed = (datetime.now() - test_start).total_seconds()
        
        # Parse output
        output_lines = result.stdout.strip().split('\n')
        result_line = [l for l in output_lines if l.startswith(('SUCCESS|', 'FAILED|', 'ERROR|'))]
        
        if result_line:
            parts = result_line[0].split('|')
            status = parts[0]
            
            if status == 'SUCCESS':
                # Parse: SUCCESS|alpha|kappa|gamma|n_clusters|silhouette|temporal|balance|between_cv|within_cv|economic_cv|runtime
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
                    'error': None,
                    'test_number': test_num
                }
                
                # Calculate derived metrics
                cv_ratio = result_dict['between_regime_cv'] / (result_dict['within_regime_cv'] + 1e-9)
                # Use log-scaled tanh to prevent CV ratio from dominating
                cv_contribution = np.tanh(np.log1p(cv_ratio)) * 0.2
                composite = (result_dict['silhouette_score'] * 0.3 + 
                           result_dict['balance_score'] * 0.3 + 
                           result_dict['temporal_smoothness'] * 0.2 +
                           cv_contribution)
                result_dict['composite_score'] = composite
                result_dict['cv_ratio'] = cv_ratio
                
                # Early stopping check
                if (result_dict['silhouette_score'] < EARLY_STOP_CONFIG['min_silhouette'] or
                    result_dict['n_clusters'] < EARLY_STOP_CONFIG['min_clusters'] or
                    result_dict['n_clusters'] > EARLY_STOP_CONFIG['max_clusters']):
                    result_dict['early_stopped'] = True
                else:
                    result_dict['early_stopped'] = False
                
                return result_dict
                
            else:
                # FAILED or ERROR
                error_msg = parts[4] if len(parts) > 4 else "Unknown error"
                return {
                    'alpha': alpha,
                    'kappa': kappa,
                    'gamma': gamma,
                    'success': False,
                    'error': error_msg,
                    'test_number': test_num
                }
        else:
            # No parseable output
            return {
                'alpha': alpha,
                'kappa': kappa,
                'gamma': gamma,
                'success': False,
                'error': 'No output from subprocess',
                'test_number': test_num
            }
            
    except Exception as e:
        return {
            'alpha': alpha,
            'kappa': kappa,
            'gamma': gamma,
            'success': False,
            'error': str(e),
            'test_number': test_num
        }


def run_grid_parallel(test_configs: List[Tuple], stage_name: str) -> pd.DataFrame:
    """
    Run grid search in parallel using multiprocessing.
    
    Args:
        test_configs: List of (alpha, kappa, gamma, test_num) tuples
        stage_name: Name of current stage for logging
    
    Returns:
        DataFrame with results
    """
    tprint_info(f"\n{stage_name}: Running {len(test_configs)} tests with {NUM_WORKERS} workers...")
    
    start_time = datetime.now()
    results = []
    
    # Run tests in parallel
    with Pool(processes=NUM_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(run_single_test, test_configs), 1):
            results.append(result)
            
            # Progress logging (every 5 tests or 10%)
            if i % 5 == 0 or i % max(1, len(test_configs) // 10) == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time = elapsed / i
                remaining = (len(test_configs) - i) * avg_time
                
                successful = sum(1 for r in results if r.get('success', False))
                
                tprint_progress(i, len(test_configs), 
                              f"{stage_name}: {i}/{len(test_configs)} tests, "
                              f"{successful} successful, ETA: {remaining/60:.1f}m")
            
            # Show result details for successful tests
            if result.get('success', False) and 'composite_score' in result:
                tprint(f"   ✅ α={result['alpha']:.3f}, κ={result['kappa']:.1f}, γ={result['gamma']:.1f} "
                      f"→ Score: {result['composite_score']:.4f}, "
                      f"Sil: {result['silhouette_score']:.3f}, "
                      f"Clusters: {result['n_clusters']}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    tprint_success(f"{stage_name} completed in {elapsed/60:.1f} minutes")
    tprint_performance(f"{stage_name} average time", elapsed / len(test_configs))
    
    return pd.DataFrame(results)


def create_refined_grid(best_config: pd.Series, 
                       alpha_range: float, 
                       kappa_range: float, 
                       gamma_range: float) -> List[Tuple]:
    """
    Create 3×5×3 grid centered on best config.
    
    Args:
        best_config: Best configuration from previous stage
        alpha_range: Total range for alpha (will be split into 3 points)
        kappa_range: Total range for kappa (will be split into 5 points)
        gamma_range: Total range for gamma (will be split into 3 points)
    
    Returns:
        List of (alpha, kappa, gamma, test_num) tuples
    """
    alphas = np.linspace(
        max(1.0, best_config['alpha'] - alpha_range/2),
        min(1.9, best_config['alpha'] + alpha_range/2),
        3
    )
    kappas = np.linspace(
        max(5.0, best_config['kappa'] - kappa_range/2),
        min(35.0, best_config['kappa'] + kappa_range/2),
        5
    )
    gammas = np.linspace(
        max(3.0, best_config['gamma'] - gamma_range/2),
        min(6.0, best_config['gamma'] + gamma_range/2),
        3
    )
    
    configs = list(itertools.product(alphas, kappas, gammas))
    return [(a, k, g, i) for i, (a, k, g) in enumerate(configs, 1)]


def print_top_results(df: pd.DataFrame, stage_name: str, n: int = 5):
    """Print top N results from stage."""
    successful = df[df['success'] == True].copy()
    
    if len(successful) == 0:
        tprint_error(f"{stage_name}: No successful results")
        return
    
    if 'composite_score' in successful.columns:
        successful = successful.sort_values('composite_score', ascending=False)
        
        tprint(f"\n{'='*80}")
        tprint(f"📊 {stage_name}: TOP {min(n, len(successful))} RESULTS")
        tprint(f"{'='*80}")
        
        for i, (_, row) in enumerate(successful.head(n).iterrows(), 1):
            tprint(f"\n{i}. α={row['alpha']:.4f}, κ={row['kappa']:.2f}, γ={row['gamma']:.3f}")
            tprint(f"   Composite: {row['composite_score']:.4f}")
            tprint(f"   Clusters: {int(row['n_clusters'])}, "
                  f"Silhouette: {row['silhouette_score']:.4f}, "
                  f"Temporal: {row['temporal_smoothness']:.4f}")
            tprint(f"   Balance: {row['balance_score']:.4f}, "
                  f"CV Ratio: {row.get('cv_ratio', 0):.4f}")


# ============================================================================
# Main Progressive Tuning Logic
# ============================================================================

def main():
    """Main progressive tuning function."""
    
    tprint("=" * 80)
    tprint("HDP-HMM Progressive Parameter Tuning (3-Stage Coarse-to-Fine)")
    tprint("=" * 80)
    tprint("")
    tprint("Strategy: Intelligent grid search with early stopping")
    tprint(f"Workers: {NUM_WORKERS} (M1-optimized)" if HARDWARE_OPT_AVAILABLE else f"Workers: {NUM_WORKERS}")
    tprint("")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    overall_start = datetime.now()
    
    # ========================================================================
    # STAGE 1: Coarse Exploration (3×5×3 = 45 tests)
    # ========================================================================
    
    tprint("\n" + "="*80)
    tprint("🔍 STAGE 1: COARSE EXPLORATION")
    tprint("="*80)
    
    alpha_s1 = [1.0, 1.45, 1.9]
    kappa_s1 = [5.0, 12.5, 20.0, 27.5, 35.0]
    gamma_s1 = [3.0, 4.5, 6.0]
    
    tprint(f"Grid: α={alpha_s1}, κ={kappa_s1}, γ={gamma_s1}")
    tprint(f"Total combinations: {len(alpha_s1) * len(kappa_s1) * len(gamma_s1)}")
    
    stage1_configs = [(a, k, g, i) for i, (a, k, g) in 
                     enumerate(itertools.product(alpha_s1, kappa_s1, gamma_s1), 1)]
    
    stage1_results = run_grid_parallel(stage1_configs, "STAGE 1")
    all_results.append(stage1_results)
    
    # Analyze Stage 1
    print_top_results(stage1_results, "STAGE 1", n=5)
    
    successful_s1 = stage1_results[stage1_results['success'] == True]
    
    if len(successful_s1) == 0:
        tprint_error("STAGE 1 failed - no successful tests. Aborting.")
        return
    
    # Get best from Stage 1
    successful_s1 = successful_s1.sort_values('composite_score', ascending=False)
    best_s1 = successful_s1.iloc[0]
    
    tprint(f"\n🏆 STAGE 1 WINNER:")
    tprint(f"   α={best_s1['alpha']:.4f}, κ={best_s1['kappa']:.2f}, γ={best_s1['gamma']:.3f}")
    tprint(f"   Composite Score: {best_s1['composite_score']:.4f}")
    
    # Early stopping check
    if best_s1['composite_score'] < EARLY_STOP_CONFIG['min_composite_stage1']:
        tprint_warning(f"STAGE 1 best score ({best_s1['composite_score']:.4f}) below threshold "
                      f"({EARLY_STOP_CONFIG['min_composite_stage1']}) - stopping early")
        save_final_results(all_results, timestamp)
        return
    
    # Check if 2nd best is close (explore both in Stage 2)
    explore_two_configs = False
    second_best_s1 = None
    
    if len(successful_s1) >= 2:
        second_best_s1 = successful_s1.iloc[1]
        score_diff = abs(best_s1['composite_score'] - second_best_s1['composite_score'])
        
        if score_diff < EARLY_STOP_CONFIG['close_threshold']:
            explore_two_configs = True
            tprint_info(f"\n🎯 2nd place is close (diff: {score_diff:.4f}) - will explore both in Stage 2")
            tprint(f"   2nd: α={second_best_s1['alpha']:.4f}, κ={second_best_s1['kappa']:.2f}, "
                  f"γ={second_best_s1['gamma']:.3f}, Score: {second_best_s1['composite_score']:.4f}")
    
    # ========================================================================
    # STAGE 2: Local Refinement (45-90 tests depending on close 2nd)
    # ========================================================================
    
    tprint("\n" + "="*80)
    tprint("🎯 STAGE 2: LOCAL REFINEMENT")
    tprint("="*80)
    
    stage2_configs = []
    
    # Always refine around best
    tprint(f"\nRefining around 1st place: α={best_s1['alpha']:.3f}, κ={best_s1['kappa']:.1f}, γ={best_s1['gamma']:.1f}")
    configs_best = create_refined_grid(best_s1, 
                                       alpha_range=0.3, 
                                       kappa_range=10.0, 
                                       gamma_range=1.5)
    stage2_configs.extend(configs_best)
    
    # Optionally refine around 2nd best
    if explore_two_configs:
        tprint(f"Refining around 2nd place: α={second_best_s1['alpha']:.3f}, κ={second_best_s1['kappa']:.1f}, γ={second_best_s1['gamma']:.1f}")
        configs_second = create_refined_grid(second_best_s1,
                                            alpha_range=0.3,
                                            kappa_range=10.0,
                                            gamma_range=1.5)
        stage2_configs.extend(configs_second)
    
    tprint(f"Total Stage 2 tests: {len(stage2_configs)}")
    
    stage2_results = run_grid_parallel(stage2_configs, "STAGE 2")
    all_results.append(stage2_results)
    
    # Analyze Stage 2
    print_top_results(stage2_results, "STAGE 2", n=5)
    
    successful_s2 = stage2_results[stage2_results['success'] == True]
    
    if len(successful_s2) == 0:
        tprint_warning("STAGE 2 had no successful tests - using Stage 1 best")
        save_final_results(all_results, timestamp)
        return
    
    successful_s2 = successful_s2.sort_values('composite_score', ascending=False)
    best_s2 = successful_s2.iloc[0]
    
    tprint(f"\n🏆 STAGE 2 WINNER:")
    tprint(f"   α={best_s2['alpha']:.4f}, κ={best_s2['kappa']:.2f}, γ={best_s2['gamma']:.3f}")
    tprint(f"   Composite Score: {best_s2['composite_score']:.4f}")
    
    # Check improvement
    improvement = (best_s2['composite_score'] - best_s1['composite_score']) / best_s1['composite_score']
    tprint(f"   Improvement over Stage 1: {improvement*100:.1f}%")
    
    if improvement < EARLY_STOP_CONFIG['improvement_threshold']:
        tprint_warning(f"STAGE 2 improvement ({improvement*100:.1f}%) below threshold - skipping Stage 3")
        save_final_results(all_results, timestamp)
        return
    
    # ========================================================================
    # STAGE 3: Fine-Tuning (3×5×3 = 45 tests)
    # ========================================================================
    
    tprint("\n" + "="*80)
    tprint("🔬 STAGE 3: FINE-TUNING")
    tprint("="*80)
    
    tprint(f"Fine-tuning around: α={best_s2['alpha']:.4f}, κ={best_s2['kappa']:.2f}, γ={best_s2['gamma']:.3f}")
    
    stage3_configs = create_refined_grid(best_s2,
                                        alpha_range=0.1,
                                        kappa_range=4.0,
                                        gamma_range=0.6)
    
    tprint(f"Total Stage 3 tests: {len(stage3_configs)}")
    
    stage3_results = run_grid_parallel(stage3_configs, "STAGE 3")
    all_results.append(stage3_results)
    
    # Analyze Stage 3
    print_top_results(stage3_results, "STAGE 3", n=5)
    
    successful_s3 = stage3_results[stage3_results['success'] == True]
    
    if len(successful_s3) > 0:
        successful_s3 = successful_s3.sort_values('composite_score', ascending=False)
        best_s3 = successful_s3.iloc[0]
        
        tprint(f"\n🏆 STAGE 3 WINNER:")
        tprint(f"   α={best_s3['alpha']:.4f}, κ={best_s3['kappa']:.2f}, γ={best_s3['gamma']:.3f}")
        tprint(f"   Composite Score: {best_s3['composite_score']:.4f}")
        
        improvement_s3 = (best_s3['composite_score'] - best_s2['composite_score']) / best_s2['composite_score']
        tprint(f"   Improvement over Stage 2: {improvement_s3*100:.1f}%")
    
    # Save results
    save_final_results(all_results, timestamp, overall_start)


def save_final_results(all_results: List[pd.DataFrame], timestamp: str, overall_start: datetime = None):
    """Save all results and generate report."""
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save CSV
    csv_path = outcomes_dir / f"hdp_hmm_progressive_results_{timestamp}.csv"
    combined_df.to_csv(csv_path, index=False)
    tprint_success(f"\n✅ Results saved to: {csv_path}")
    
    # Generate summary
    successful = combined_df[combined_df['success'] == True]
    
    if len(successful) > 0 and 'composite_score' in successful.columns:
        successful = successful.sort_values('composite_score', ascending=False)
        best_overall = successful.iloc[0]
        
        tprint("\n" + "="*80)
        tprint("🏆 FINAL BEST CONFIGURATION")
        tprint("="*80)
        tprint(f"\nParameters:")
        tprint(f"   α (alpha) = {best_overall['alpha']:.4f}")
        tprint(f"   κ (kappa) = {best_overall['kappa']:.2f}")
        tprint(f"   γ (gamma) = {best_overall['gamma']:.3f}")
        tprint(f"\nMetrics:")
        tprint(f"   Composite Score: {best_overall['composite_score']:.4f}")
        tprint(f"   Clusters: {int(best_overall['n_clusters'])}")
        tprint(f"   Silhouette: {best_overall['silhouette_score']:.4f}")
        tprint(f"   Temporal Smoothness: {best_overall['temporal_smoothness']:.4f}")
        tprint(f"   Balance Score: {best_overall['balance_score']:.4f}")
        tprint(f"   CV Ratio: {best_overall.get('cv_ratio', 0):.4f}")
        
        tprint(f"\n📊 Summary:")
        tprint(f"   Total tests run: {len(combined_df)}")
        tprint(f"   Successful: {len(successful)}")
        tprint(f"   Failed: {len(combined_df) - len(successful)}")
        
        if overall_start:
            total_time = (datetime.now() - overall_start).total_seconds()
            tprint(f"   Total time: {total_time/60:.1f} minutes")
            tprint(f"   Average per test: {total_time/len(combined_df):.1f}s")
        
        # Generate markdown report
        generate_report(combined_df, timestamp, best_overall)


def generate_report(df: pd.DataFrame, timestamp: str, best_config: pd.Series):
    """Generate markdown report."""
    
    report_path = outcomes_dir / f"hdp_hmm_progressive_report_{timestamp}.md"
    
    successful = df[df['success'] == True]
    
    report = f"""# HDP-HMM Progressive Tuning Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Strategy**: 3-Stage Coarse-to-Fine Grid Search  
**Workers**: {NUM_WORKERS} (M1-optimized)

## Results Summary

- **Total Tests**: {len(df)}
- **Successful**: {len(successful)} ({100*len(successful)/len(df):.1f}%)
- **Failed**: {len(df) - len(successful)} ({100*(len(df)-len(successful))/len(df):.1f}%)

## 🏆 Best Configuration

**Parameters:**
- α (alpha) = {best_config['alpha']:.4f}
- κ (kappa) = {best_config['kappa']:.2f}
- γ (gamma) = {best_config['gamma']:.3f}

**Metrics:**
- Composite Score: {best_config['composite_score']:.4f}
- Clusters: {int(best_config['n_clusters'])}
- Silhouette: {best_config['silhouette_score']:.4f}
- Temporal Smoothness: {best_config['temporal_smoothness']:.4f}
- Balance Score: {best_config['balance_score']:.4f}
- CV Ratio: {best_config.get('cv_ratio', 0):.4f}
- Runtime: {best_config.get('runtime', 0):.1f}s

## Top 10 Configurations

{successful.sort_values('composite_score', ascending=False).head(10)[['alpha', 'kappa', 'gamma', 'composite_score', 'n_clusters', 'silhouette_score', 'temporal_smoothness', 'balance_score']].to_markdown(index=False)}

## Progressive Search Strategy

This tuning used a **3-stage coarse-to-fine strategy**:

1. **Stage 1**: Coarse exploration (3×5×3 grid) to find promising regions
2. **Stage 2**: Local refinement around top 1-2 candidates
3. **Stage 3**: Fine-tuning for optimal parameters

### Advantages:
- ✅ 83% fewer tests than full grid search (810 → ~135-180)
- ✅ Parallel execution with {NUM_WORKERS} workers
- ✅ Early stopping for poor configurations
- ✅ Adaptive exploration of multiple promising regions

---
*Full results available in: hdp_hmm_progressive_results_{timestamp}.csv*
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    tprint_success(f"📄 Report saved to: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        tprint_warning("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        tprint_error(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

