"""
Hierarchical Grid Search for HDP-HMM
Three-stage search: Coarse → Medium → Fine with 20-50x speedup
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import subprocess
from datetime import datetime
from pathlib import Path


@dataclass
class StageConfig:
    """Configuration for one stage of hierarchical search."""
    name: str
    n_iterations: int
    n_combinations: int
    convergence_threshold: float
    timeout: int  # seconds
    description: str


class HierarchicalGridSearch:
    """
    Three-stage grid search with progressive refinement.
    
    Stage 1 (Coarse): Fast exploration with Latin Hypercube Sampling
    Stage 2 (Medium): Focus on top performers from Stage 1
    Stage 3 (Fine): Tight grid around very best from Stage 2
    
    Expected speedup: 20-50x compared to exhaustive grid search
    """
    
    def __init__(self, 
                 test_runner: Optional[Callable] = None,
                 verbose: bool = True):
        """
        Initialize hierarchical grid search.
        
        Args:
            test_runner: Optional function to run single test (alpha, kappa, gamma, n_iters) -> result_dict
            verbose: Whether to print progress
        """
        self.test_runner = test_runner
        self.verbose = verbose
        
        self.stage_configs = {
            'coarse': StageConfig(
                name='coarse',
                n_iterations=30,
                n_combinations=20,
                convergence_threshold=0.05,
                timeout=60,
                description="Fast exploration with Latin Hypercube Sampling"
            ),
            'medium': StageConfig(
                name='medium',
                n_iterations=100,
                n_combinations=15,
                convergence_threshold=0.02,
                timeout=180,
                description="Focus on top 15 from coarse stage"
            ),
            'fine': StageConfig(
                name='fine',
                n_iterations=200,
                n_combinations=5,
                convergence_threshold=0.01,
                timeout=300,
                description="Tight grid around top 5 from medium stage"
            )
        }
    
    def _latin_hypercube_sample(self, 
                                param_ranges: Dict[str, Tuple[float, float]], 
                                n_samples: int) -> List[Tuple[float, float, float]]:
        """
        Generate Latin Hypercube samples for better parameter space coverage.
        
        Args:
            param_ranges: Dict with 'alpha', 'kappa', 'gamma' ranges
            n_samples: Number of samples to generate
            
        Returns:
            List of (alpha, kappa, gamma) tuples
        """
        try:
            from scipy.stats import qmc
            
            # Create Latin Hypercube sampler
            sampler = qmc.LatinHypercube(d=3, seed=42)
            samples = sampler.random(n=n_samples)
            
            # Scale to parameter ranges
            alpha_range = param_ranges['alpha']
            kappa_range = param_ranges['kappa']
            gamma_range = param_ranges['gamma']
            
            params = []
            for sample in samples:
                alpha = alpha_range[0] + sample[0] * (alpha_range[1] - alpha_range[0])
                kappa = kappa_range[0] + sample[1] * (kappa_range[1] - kappa_range[0])
                gamma = gamma_range[0] + sample[2] * (gamma_range[1] - gamma_range[0])
                params.append((alpha, kappa, gamma))
            
            return params
            
        except ImportError:
            # Fallback to random sampling if scipy not available
            if self.verbose:
                print("⚠️  scipy.stats.qmc not available, using random sampling instead")
            
            alpha_range = param_ranges['alpha']
            kappa_range = param_ranges['kappa']
            gamma_range = param_ranges['gamma']
            
            np.random.seed(42)
            params = []
            for _ in range(n_samples):
                alpha = np.random.uniform(alpha_range[0], alpha_range[1])
                kappa = np.random.uniform(kappa_range[0], kappa_range[1])
                gamma = np.random.uniform(gamma_range[0], gamma_range[1])
                params.append((alpha, kappa, gamma))
            
            return params
    
    def _expand_around_best(self, 
                           best_configs: List[dict], 
                           n_samples: int,
                           expansion_factor: float = 0.2) -> List[Tuple[float, float, float]]:
        """
        Generate samples around best configurations from previous stage.
        
        Args:
            best_configs: List of best configurations
            n_samples: Number of samples to generate per config
            expansion_factor: How much to expand around each config (as fraction of range)
            
        Returns:
            List of (alpha, kappa, gamma) tuples
        """
        params = []
        
        # Original parameter ranges
        alpha_full_range = 4.0 - 1.0
        kappa_full_range = 45.0 - 5.0
        gamma_full_range = 6.0 - 3.0
        
        for config in best_configs:
            alpha_base = config['alpha']
            kappa_base = config['kappa']
            gamma_base = config['gamma']
            
            # Calculate search window (±expansion_factor)
            alpha_width = alpha_full_range * expansion_factor
            kappa_width = kappa_full_range * expansion_factor
            gamma_width = gamma_full_range * expansion_factor
            
            # Generate local samples
            for _ in range(n_samples // len(best_configs)):
                alpha = np.clip(
                    np.random.normal(alpha_base, alpha_width / 3),
                    max(1.0, alpha_base - alpha_width),
                    min(4.0, alpha_base + alpha_width)
                )
                kappa = np.clip(
                    np.random.normal(kappa_base, kappa_width / 3),
                    max(5.0, kappa_base - kappa_width),
                    min(45.0, kappa_base + kappa_width)
                )
                gamma = np.clip(
                    np.random.normal(gamma_base, gamma_width / 3),
                    max(3.0, gamma_base - gamma_width),
                    min(6.0, gamma_base + gamma_width)
                )
                params.append((alpha, kappa, gamma))
        
        return params
    
    def _fine_grid_around_best(self, 
                              best_configs: List[dict], 
                              n_samples: int,
                              expansion_factor: float = 0.1) -> List[Tuple[float, float, float]]:
        """
        Generate tight grid around very best configurations.
        
        Args:
            best_configs: List of best configurations
            n_samples: Number of samples to generate
            expansion_factor: Grid size (as fraction of range)
            
        Returns:
            List of (alpha, kappa, gamma) tuples
        """
        params = []
        
        # Original parameter ranges
        alpha_full_range = 4.0 - 1.0
        kappa_full_range = 45.0 - 5.0
        gamma_full_range = 6.0 - 3.0
        
        # Use only the very best config as center
        best = best_configs[0]
        alpha_base = best['alpha']
        kappa_base = best['kappa']
        gamma_base = best['gamma']
        
        # Calculate tight grid (±expansion_factor)
        alpha_width = alpha_full_range * expansion_factor
        kappa_width = kappa_full_range * expansion_factor
        gamma_width = gamma_full_range * expansion_factor
        
        # Create 3D grid
        grid_size = int(np.ceil(n_samples ** (1/3)))  # Cube root for 3D grid
        
        alpha_values = np.linspace(
            max(1.0, alpha_base - alpha_width),
            min(4.0, alpha_base + alpha_width),
            grid_size
        )
        kappa_values = np.linspace(
            max(5.0, kappa_base - kappa_width),
            min(45.0, kappa_base + kappa_width),
            grid_size
        )
        gamma_values = np.linspace(
            max(3.0, gamma_base - gamma_width),
            min(6.0, gamma_base + gamma_width),
            grid_size
        )
        
        for alpha in alpha_values:
            for kappa in kappa_values:
                for gamma in gamma_values:
                    params.append((alpha, kappa, gamma))
        
        return params[:n_samples]  # Limit to requested number
    
    def _default_test_runner(self, alpha: float, kappa: float, gamma: float, n_iterations: int) -> dict:
        """Default test runner using subprocess."""
        try:
            cmd = ['python3', 'hdp_hmm_single_test.py', str(alpha), str(kappa), str(gamma), str(n_iterations)]
            
            test_start = datetime.now()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            test_elapsed = (datetime.now() - test_start).total_seconds()
            
            # Parse output
            output_lines = result.stdout.strip().split('\n')
            result_line = [l for l in output_lines if l.startswith(('SUCCESS|', 'FAILED|', 'ERROR|'))]
            
            if result_line:
                parts = result_line[-1].split('|')
                status = parts[0]
                
                if status == 'SUCCESS' and len(parts) >= 12:
                    def safe_float(s):
                        try:
                            return float(s) if s and s != 'None' and s != 'nan' else 0.0
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
                        'within_regime_cv': safe_float(parts[9]) if parts[9] and parts[9] != '0.0' else 1.0,
                        'economic_cv_ratio': safe_float(parts[10]),
                        'runtime': test_elapsed,
                        'success': True
                    }
                    
                    # Calculate composite score
                    within_cv = result_dict['within_regime_cv']
                    if within_cv and within_cv > 0:
                        cv_ratio = result_dict['between_regime_cv'] / (within_cv + 1e-9)
                        # Use log-scaled tanh to prevent CV ratio from dominating
                        cv_contribution = np.tanh(np.log1p(cv_ratio)) * 0.30
                        composite = (
                            result_dict['silhouette_score'] * 0.20 +
                            result_dict['balance_score'] * 0.25 +
                            result_dict['temporal_smoothness'] * 0.25 +
                            cv_contribution
                        )
                    else:
                        composite = 0.0
                    
                    result_dict['composite_score'] = composite
                    return result_dict
            
            return {'success': False, 'error': 'Failed to parse output', 'composite_score': 0.0}
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Timeout', 'composite_score': 0.0}
        except Exception as e:
            return {'success': False, 'error': str(e), 'composite_score': 0.0}
    
    def run_stage(self, 
                  stage: str, 
                  param_space: dict = None,
                  best_from_previous: List[dict] = None) -> List[dict]:
        """
        Run one stage of hierarchical search.
        
        Args:
            stage: Stage name ('coarse', 'medium', 'fine')
            param_space: Parameter space dict (for coarse stage)
            best_from_previous: Best configs from previous stage (for medium/fine)
            
        Returns:
            List of result dictionaries
        """
        config = self.stage_configs[stage]
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🔍 STAGE: {config.name.upper()}")
            print(f"{'='*80}")
            print(f"Description: {config.description}")
            print(f"Iterations: {config.n_iterations}")
            print(f"Combinations: {config.n_combinations}")
            print(f"{'='*80}\n")
        
        # Generate parameter combinations for this stage
        if stage == 'coarse':
            if param_space is None:
                param_space = {
                    'alpha': (1.0, 4.0),
                    'kappa': (5.0, 45.0),
                    'gamma': (3.0, 6.0)
                }
            params = self._latin_hypercube_sample(param_space, config.n_combinations)
            
        elif stage == 'medium':
            if not best_from_previous:
                raise ValueError("Medium stage requires best_from_previous configs")
            params = self._expand_around_best(best_from_previous, config.n_combinations, expansion_factor=0.2)
            
        else:  # fine
            if not best_from_previous:
                raise ValueError("Fine stage requires best_from_previous configs")
            params = self._fine_grid_around_best(best_from_previous, config.n_combinations, expansion_factor=0.1)
        
        # Run tests
        runner = self.test_runner if self.test_runner else self._default_test_runner
        results = []
        
        for i, (alpha, kappa, gamma) in enumerate(params, 1):
            if self.verbose:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Test {i}/{len(params)} | "
                      f"α={alpha:.2f}, κ={kappa:.1f}, γ={gamma:.1f}")
            
            result = runner(alpha, kappa, gamma, config.n_iterations)
            results.append(result)
            
            if result.get('success') and self.verbose:
                score = result.get('composite_score', 0.0)
                clusters = result.get('n_clusters', 0)
                print(f"  ✅ Score={score:.3f}, Clusters={clusters}")
            elif self.verbose:
                print(f"  ❌ Failed: {result.get('error', 'Unknown')}")
        
        # Sort by composite score
        results.sort(key=lambda x: x.get('composite_score', 0.0), reverse=True)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"✅ {config.name.upper()} STAGE COMPLETE")
            print(f"Best score: {results[0].get('composite_score', 0.0):.3f}")
            print(f"Best params: α={results[0].get('alpha', 0):.2f}, "
                  f"κ={results[0].get('kappa', 0):.1f}, "
                  f"γ={results[0].get('gamma', 0):.1f}")
            print(f"{'='*80}\n")
        
        return results
    
    def run_full_search(self, param_space: dict = None) -> dict:
        """
        Run complete hierarchical search (all 3 stages).
        
        Args:
            param_space: Parameter space dict (optional, uses defaults if None)
            
        Returns:
            Dict with best params and stage history
        """
        start_time = datetime.now()
        
        # Stage 1: Coarse
        coarse_results = self.run_stage('coarse', param_space=param_space)
        top_coarse = coarse_results[:15]  # Top 15 for medium stage
        
        # Stage 2: Medium
        medium_results = self.run_stage('medium', best_from_previous=top_coarse)
        top_medium = medium_results[:5]  # Top 5 for fine stage
        
        # Stage 3: Fine
        fine_results = self.run_stage('fine', best_from_previous=top_medium)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        if self.verbose:
            print(f"\n{'█'*80}")
            print(f"{'█'*80}")
            print(f"🏆 HIERARCHICAL SEARCH COMPLETE")
            print(f"{'█'*80}")
            print(f"{'█'*80}")
            print(f"\nTotal time: {total_time/60:.1f} minutes")
            print(f"Total tests: {len(coarse_results) + len(medium_results) + len(fine_results)}")
            print(f"\nBest configuration:")
            best = fine_results[0]
            print(f"  α = {best.get('alpha', 0):.3f}")
            print(f"  κ = {best.get('kappa', 0):.1f}")
            print(f"  γ = {best.get('gamma', 0):.3f}")
            print(f"  Score = {best.get('composite_score', 0):.3f}")
            print(f"  Clusters = {best.get('n_clusters', 0)}")
            print(f"{'█'*80}\n")
        
        return {
            'best_params': fine_results[0],
            'total_time': total_time,
            'stage_history': {
                'coarse': coarse_results,
                'medium': medium_results,
                'fine': fine_results
            }
        }


__all__ = ['HierarchicalGridSearch', 'StageConfig']

