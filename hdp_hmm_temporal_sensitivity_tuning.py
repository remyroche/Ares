#!/usr/bin/env python3
"""
HDP-HMM Temporal Sensitivity Tuning

Tests different temporal smoothness calculation modes to find parameter combinations
that create more variation in temporal metrics. Uses expanded parameter ranges
and multiple sensitivity modes to disrupt temporal stability.
"""

import os
import sys
import json
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_data_preview

def run_hdp_hmm_test(alpha, kappa, gamma, n_iterations=50, sensitivity_mode="standard"):
    """Run a single HDP-HMM test with specified parameters."""
    try:
        cmd = [
            'python3', 'hdp_hmm_single_test.py',
            str(alpha), str(kappa), str(gamma), str(n_iterations)
        ]
        if sensitivity_mode != "standard":
            cmd.extend(['--sensitivity_mode', sensitivity_mode])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=project_root
        )

        if result.returncode == 0 and result.stdout:
            # Parse the JSON output
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    try:
                        data = json.loads(line.strip())
                        data['sensitivity_mode'] = sensitivity_mode
                        return data, result.stderr, True
                    except json.JSONDecodeError:
                        continue

        return {'error': result.stderr or 'No output', 'alpha': alpha, 'kappa': kappa, 'gamma': gamma, 'sensitivity_mode': sensitivity_mode}, result.stderr, False

    except subprocess.TimeoutExpired:
        return {'error': 'Timeout', 'alpha': alpha, 'kappa': kappa, 'gamma': gamma, 'sensitivity_mode': sensitivity_mode}, '', False
    except Exception as e:
        return {'error': str(e), 'alpha': alpha, 'kappa': kappa, 'gamma': gamma, 'sensitivity_mode': sensitivity_mode}, '', False

def main():
    """Main temporal sensitivity tuning experiment."""
    tprint("🚀 HDP-HMM Temporal Sensitivity Tuning", "info")
    tprint("Testing expanded parameter ranges with different temporal sensitivity modes", "info")

    # EXPANDED parameter ranges for maximum temporal disruption
    alpha_range = [0.5, 1.0, 2.0, 3.0, 4.0]  # More extreme concentration
    kappa_range = [1.0, 10.0, 25.0, 50.0, 100.0]  # Wider stickiness range
    gamma_range = [1.0, 3.0, 6.0, 9.0, 12.0]  # More extreme discount

    # Different temporal sensitivity modes
    sensitivity_modes = [
        "standard",                    # Original calculation
        "exponential_decay",           # More aggressive transition penalty
        "weighted_transitions",        # Weight transitions by duration
        "regime_persistence_focused"   # Emphasize long regime persistence
    ]

    total_tests = len(alpha_range) * len(kappa_range) * len(gamma_range) * len(sensitivity_modes)
    tprint(f"📊 Testing {total_tests} configurations", "info")
    tprint(f"   α ∈ {alpha_range}", "info")
    tprint(f"   κ ∈ {kappa_range}", "info")
    tprint(f"   γ ∈ {gamma_range}", "info")
    tprint(f"   Sensitivity modes: {sensitivity_modes}", "info")

    results = []
    successful_tests = 0
    test_counter = 0
    start_time = datetime.now()

    for sensitivity_mode in sensitivity_modes:
        tprint(f"\n🎯 Testing sensitivity mode: {sensitivity_mode}", "info")

        for alpha in alpha_range:
            for kappa in kappa_range:
                for gamma in gamma_range:
                    test_counter += 1
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    tprint(f"[{current_time}] 🔬 Test {test_counter}/{total_tests} | "
                          f"α={alpha:.2f}, κ={kappa:.1f}, γ={gamma:.1f} | Mode: {sensitivity_mode}")

                    result_dict, stderr, success = run_hdp_hmm_test(alpha, kappa, gamma, sensitivity_mode=sensitivity_mode)

                    if success and 'temporal_smoothness' in result_dict:
                        results.append(result_dict)
                        successful_tests += 1

                        # Extract temporal metrics
                        temporal = result_dict.get('temporal_smoothness', 0.0)
                        duration_stability = result_dict.get('regime_duration_distribution', {}).get('duration_stability_score', 0.0)
                        transition_stability = result_dict.get('transition_probability_matrix', {}).get('transition_stability_score', 0.0)
                        composite = result_dict.get('composite_score', 0.0)
                        clusters = result_dict.get('n_clusters', 0)

                        tprint(f"[{current_time}] ✅ Test {test_counter}/{total_tests} | "
                              f"α={alpha:.2f}, κ={kappa:.1f}, γ={gamma:.1f} | "
                              f"Clusters={clusters}, Score={composite:.3f} "
                              f"(Temp={temporal:.2f}, DurStab={duration_stability:.2f}, TransStab={transition_stability:.2f}) | Mode: {sensitivity_mode}")
                    else:
                        results.append(result_dict)
                        tprint(f"[{current_time}] ❌ Test {test_counter}/{total_tests} | "
                              f"α={alpha:.2f}, κ={kappa:.1f}, γ={gamma:.1f} | "
                              f"Error: {result_dict.get('error', 'Unknown')} | Mode: {sensitivity_mode}")

                    # Progress update every 20 tests
                    if test_counter % 20 == 0:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        avg_time = elapsed / test_counter
                        remaining = (total_tests - test_counter) * avg_time
                        tprint(f"📊 Progress: {test_counter}/{total_tests} ({100*test_counter/total_tests:.1f}%) | "
                              f"Success: {successful_tests}, ETA: {remaining/60:.1f}m")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outcomes/hdp_hmm_temporal_sensitivity_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'experiment': 'temporal_sensitivity_tuning',
            'timestamp': timestamp,
            'parameter_ranges': {
                'alpha': alpha_range,
                'kappa': kappa_range,
                'gamma': gamma_range,
                'sensitivity_modes': sensitivity_modes
            },
            'results': results,
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0
            }
        }, f, indent=2)

    tprint(f"\n✅ Temporal sensitivity tuning completed!", "success")
    tprint(f"   Results saved to: {output_file}", "success")
    tprint(f"   Success rate: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)", "success")

    # Analyze results by sensitivity mode
    for mode in sensitivity_modes:
        mode_results = [r for r in results if r.get('sensitivity_mode') == mode and 'temporal_smoothness' in r]
        if mode_results:
            temporal_values = [r['temporal_smoothness'] for r in mode_results]
            tprint(f"   {mode}: {len(mode_results)} tests, Temp range: [{min(temporal_values):.3f}, {max(temporal_values):.3f}], Std: {np.std(temporal_values):.3f}", "info")

if __name__ == "__main__":
    main()
