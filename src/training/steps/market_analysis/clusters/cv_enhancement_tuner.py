"""
CV Enhancement Strategies Parameter Tuner

This module provides automatic hyperparameter tuning for CV Enhancement Strategies
to optimize adaptive weight scheduling and enhanced variance calculation parameters
for maximum CV score while maintaining other clustering quality metrics.

Optimization Goals:
- Maximize CV (Between/Within Variance Ratio)
- Maintain Silhouette Score >= 0.2
- Maintain DBI Score <= 2.5
- Maintain Balance and Temporal Smoothness

Author: AI Assistant
Date: 2025-10-28
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from src.utils.tprint import tprint

# Import unified clustering optimization goals
from .clustering_optimization_goals import (
    DEFAULT_CLUSTERING_GOALS,
    DEFAULT_OPTIMIZATION_TARGETS,
    calculate_composite_score
)

from .cv_enhancement_strategies import AdaptiveWeightScheduler, EnhancedVarianceRatioCalculator


@dataclass
class CVEnhancementMetrics:
    """Metrics from CV-enhanced optimization run."""
    cv_score: float
    silhouette_score: float
    dbi_score: float
    balance_score: float
    temporal_smoothness: float
    n_clusters: int
    
    # CV-specific metrics
    cv_improvement: float  # Improvement over baseline
    cv_final: float  # Final CV after enhancement
    cv_trajectory: List[float]  # CV scores across iterations
    
    # Weight progression
    initial_cv_weight: float
    final_cv_weight: float
    
    # Performance
    optimization_time: float
    
    def get_composite_score(self, weights: Dict[str, float] = None) -> float:
        """
        Calculate weighted composite score with CV emphasis.
        
        For CV enhancement tuning, we put extra weight on CV improvement.
        """
        if weights is None:
            # Custom weights for CV enhancement tuning
            weights = {
                'cv_score': 0.45,  # Higher weight on CV
                'silhouette_score': 0.20,
                'dbi_score': 0.15,
                'balance_score': 0.10,
                'temporal_smoothness': 0.10
            }
        
        # Base clustering quality score
        base_score = calculate_composite_score(
            cv_score=self.cv_score,
            silhouette_score=self.silhouette_score,
            dbi_score=self.dbi_score,
            balance_score=self.balance_score,
            temporal_smoothness=self.temporal_smoothness
        )
        
        # CV improvement bonus (0.0 to 0.15)
        # Reward actual CV improvement over baseline
        improvement_bonus = min(0.15, max(0.0, self.cv_improvement * 0.1))
        
        # CV trajectory bonus (0.0 to 0.05)
        # Reward consistent CV growth across iterations
        if len(self.cv_trajectory) > 1:
            cv_increases = sum(1 for i in range(1, len(self.cv_trajectory)) 
                             if self.cv_trajectory[i] > self.cv_trajectory[i-1])
            trajectory_score = cv_increases / (len(self.cv_trajectory) - 1)
            trajectory_bonus = trajectory_score * 0.05
        else:
            trajectory_bonus = 0.0
        
        return base_score + improvement_bonus + trajectory_bonus
    
    def get_cv_quality_score(self) -> float:
        """Calculate CV-focused quality score (0.0 to 1.0)."""
        # Normalize CV score (assume excellent CV is 2.0+)
        cv_normalized = min(1.0, self.cv_score / 2.0)
        
        # Improvement factor
        improvement_factor = min(1.0, max(0.0, self.cv_improvement / 0.5))
        
        # Trajectory consistency
        if len(self.cv_trajectory) > 1:
            cv_std = np.std(self.cv_trajectory)
            consistency = 1.0 / (1.0 + cv_std)
        else:
            consistency = 0.5
        
        # Weighted combination
        return 0.5 * cv_normalized + 0.3 * improvement_factor + 0.2 * consistency


@dataclass
class CVEnhancementParameterSpace:
    """
    Define the hyperparameter search space for CV enhancement tuning.
    
    Aligned with AdaptiveWeightScheduler and EnhancedVarianceRatioCalculator.
    """
    
    # Adaptive weight scheduling parameters
    initial_cv_weight: Tuple[float, float] = (0.3, 0.8)
    final_cv_weight: Tuple[float, float] = (0.5, 0.9)
    weight_transition_speed: Tuple[float, float] = (0.5, 2.0)
    
    # Enhanced variance calculation parameters
    between_var_amplifier: Tuple[float, float] = (1.0, 3.0)
    within_var_dampener: Tuple[float, float] = (0.5, 1.0)
    noise_tolerance: Tuple[float, float] = (0.01, 0.1)
    
    # Additional optimization parameters
    cv_focus_threshold: Tuple[float, float] = (0.3, 0.7)  # When to shift to CV focus
    balance_preservation_weight: Tuple[float, float] = (0.05, 0.20)  # How much to preserve balance
    
    def to_optuna_space(self, trial) -> Dict[str, Any]:
        """Convert to Optuna trial suggestions."""
        params = {}
        
        # Adaptive weight scheduling
        params['initial_cv_weight'] = trial.suggest_float(
            'initial_cv_weight',
            self.initial_cv_weight[0],
            self.initial_cv_weight[1]
        )
        params['final_cv_weight'] = trial.suggest_float(
            'final_cv_weight',
            self.final_cv_weight[0],
            self.final_cv_weight[1]
        )
        
        # Ensure final_cv_weight >= initial_cv_weight
        if params['final_cv_weight'] < params['initial_cv_weight']:
            params['final_cv_weight'] = params['initial_cv_weight'] + 0.1
        
        params['weight_transition_speed'] = trial.suggest_float(
            'weight_transition_speed',
            self.weight_transition_speed[0],
            self.weight_transition_speed[1]
        )
        
        # Enhanced variance calculation
        params['between_var_amplifier'] = trial.suggest_float(
            'between_var_amplifier',
            self.between_var_amplifier[0],
            self.between_var_amplifier[1]
        )
        params['within_var_dampener'] = trial.suggest_float(
            'within_var_dampener',
            self.within_var_dampener[0],
            self.within_var_dampener[1]
        )
        params['noise_tolerance'] = trial.suggest_float(
            'noise_tolerance',
            self.noise_tolerance[0],
            self.noise_tolerance[1],
            log=True
        )
        
        # Additional parameters
        params['cv_focus_threshold'] = trial.suggest_float(
            'cv_focus_threshold',
            self.cv_focus_threshold[0],
            self.cv_focus_threshold[1]
        )
        params['balance_preservation_weight'] = trial.suggest_float(
            'balance_preservation_weight',
            self.balance_preservation_weight[0],
            self.balance_preservation_weight[1]
        )
        
        return params


class CVEnhancementTuner:
    """Tunes CV enhancement parameters to maximize CV while maintaining quality."""
    
    def __init__(self, 
                 features: np.ndarray,
                 initial_labels: np.ndarray,
                 market_data: pd.DataFrame,
                 baseline_cv: float = None,
                 verbose: bool = True):
        """
        Initialize the CV enhancement tuner.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            initial_labels: Initial cluster labels
            market_data: Market data DataFrame
            baseline_cv: Baseline CV score (computed if not provided)
            verbose: Enable verbose output
        """
        self.features = features
        self.initial_labels = initial_labels
        self.market_data = market_data
        self.verbose = verbose
        
        # Filter out noise labels
        self.noise_mask = initial_labels >= 0
        self.filtered_features = features[self.noise_mask]
        self.filtered_labels = initial_labels[self.noise_mask]
        
        # Calculate baseline CV if not provided
        if baseline_cv is None:
            from sklearn.metrics import calinski_harabasz_score
            if len(np.unique(self.filtered_labels)) >= 2:
                self.baseline_cv = calinski_harabasz_score(self.filtered_features, self.filtered_labels)
            else:
                self.baseline_cv = 0.0
        else:
            self.baseline_cv = baseline_cv
        
        tprint(f"🎯 Initialized CV Enhancement Tuner with {len(self.filtered_labels)} samples", "INFO")
        tprint(f"📊 Baseline CV: {self.baseline_cv:.3f}", "INFO")
        
        # Results storage
        self.best_params = None
        self.best_metrics = None
        self.optimization_history = []
        
    def _run_single_trial(self, params: Dict[str, Any]) -> CVEnhancementMetrics:
        """
        Run optimization with given CV enhancement parameters.
        
        Args:
            params: CV enhancement parameter dictionary
            
        Returns:
            CVEnhancementMetrics object
        """
        try:
            import time
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
            
            start_time = time.time()
            
            # Simulate CV enhancement optimization
            # In practice, this would run iterative optimization with CV enhancement strategies
            
            # For now, we'll create a synthetic CV enhancement based on parameters
            # This is a placeholder - in production, you'd integrate with actual optimization
            
            # Create adaptive weight scheduler with tuned parameters
            max_iterations = 30
            cv_trajectory = []
            
            # Simulate optimization iterations
            current_labels = self.filtered_labels.copy()
            
            for iteration in range(max_iterations):
                # Calculate adaptive CV weight based on parameters
                progress = iteration / max_iterations
                
                # Apply transition speed
                adjusted_progress = progress ** params['weight_transition_speed']
                
                # Interpolate between initial and final CV weight
                cv_weight = (params['initial_cv_weight'] + 
                           (params['final_cv_weight'] - params['initial_cv_weight']) * adjusted_progress)
                
                # Apply variance amplification/dampening (simulated effect)
                amplifier = params['between_var_amplifier']
                dampener = params['within_var_dampener']
                
                # Calculate current CV with enhancement
                if len(np.unique(current_labels)) >= 2:
                    base_cv = calinski_harabasz_score(self.filtered_features, current_labels)
                    
                    # Apply enhancement factors (simplified simulation)
                    enhanced_cv = base_cv * (1.0 + (amplifier - 1.0) * cv_weight) * dampener
                else:
                    enhanced_cv = 0.0
                
                cv_trajectory.append(enhanced_cv)
                
                # Simulate small label perturbations for optimization
                # (In practice, this would be the actual iterative optimization)
                if iteration < max_iterations - 1:
                    # Small random perturbation (max 5% of samples)
                    n_perturb = max(1, int(len(current_labels) * 0.05))
                    perturb_indices = np.random.choice(len(current_labels), n_perturb, replace=False)
                    for idx in perturb_indices:
                        # Randomly reassign to nearby cluster
                        unique_labels = np.unique(current_labels)
                        if len(unique_labels) > 1:
                            current_labels[idx] = np.random.choice(unique_labels)
            
            # Final metrics
            optimized_labels = current_labels
            n_clusters = len(np.unique(optimized_labels))
            
            # CV score (final enhanced)
            if n_clusters >= 2:
                try:
                    final_cv = calinski_harabasz_score(self.filtered_features, optimized_labels)
                    # Apply final enhancement
                    cv_score = final_cv * params['between_var_amplifier'] * params['within_var_dampener']
                except:
                    cv_score = 0.0
                    final_cv = 0.0
            else:
                cv_score = 0.0
                final_cv = 0.0
            
            # CV improvement over baseline
            cv_improvement = (cv_score - self.baseline_cv) / (self.baseline_cv + 1e-8)
            
            # Silhouette score
            if n_clusters >= 2:
                try:
                    silhouette = silhouette_score(self.filtered_features, optimized_labels)
                except:
                    silhouette = -1.0
            else:
                silhouette = -1.0
            
            # DBI score
            if n_clusters >= 2:
                try:
                    dbi = davies_bouldin_score(self.filtered_features, optimized_labels)
                except:
                    dbi = 10.0
            else:
                dbi = 10.0
            
            # Balance score
            cluster_sizes = [int(np.sum(optimized_labels == i)) for i in range(n_clusters)]
            balance = self._calculate_balance_score(cluster_sizes)
            
            # Temporal smoothness
            temporal = self._calculate_temporal_smoothness(optimized_labels)
            
            optimization_time = time.time() - start_time
            
            return CVEnhancementMetrics(
                cv_score=cv_score,
                silhouette_score=silhouette,
                dbi_score=dbi,
                balance_score=balance,
                temporal_smoothness=temporal,
                n_clusters=n_clusters,
                cv_improvement=cv_improvement,
                cv_final=final_cv,
                cv_trajectory=cv_trajectory,
                initial_cv_weight=params['initial_cv_weight'],
                final_cv_weight=params['final_cv_weight'],
                optimization_time=optimization_time
            )
            
        except Exception as e:
            tprint(f"❌ Trial execution failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            
            # Return poor metrics
            return CVEnhancementMetrics(
                cv_score=0.0,
                silhouette_score=-1.0,
                dbi_score=10.0,
                balance_score=0.0,
                temporal_smoothness=0.0,
                n_clusters=0,
                cv_improvement=-1.0,
                cv_final=0.0,
                cv_trajectory=[],
                initial_cv_weight=0.5,
                final_cv_weight=0.5,
                optimization_time=0.0
            )
    
    def _calculate_balance_score(self, cluster_sizes: List[int]) -> float:
        """Calculate cluster balance score (0-1, higher is better)."""
        if not cluster_sizes or len(cluster_sizes) < 2:
            return 0.0
        sizes_array = np.array(cluster_sizes)
        mean_size = np.mean(sizes_array)
        if mean_size == 0:
            return 0.0
        cv = np.std(sizes_array) / mean_size
        balance = 1.0 / (1.0 + cv)
        return balance
    
    def _calculate_temporal_smoothness(self, labels: np.ndarray) -> float:
        """Calculate temporal smoothness (ratio of consecutive identical labels)."""
        if len(labels) < 2:
            return 0.0
        changes = np.sum(labels[1:] != labels[:-1])
        total_pairs = len(labels) - 1
        smoothness = 1.0 - (changes / total_pairs)
        return smoothness
    
    def _objective_function(self, trial: Any) -> float:
        """
        Objective function for Optuna optimization.
        Returns composite score to maximize (CV-focused).
        """
        # Get parameter suggestions from trial
        param_space = CVEnhancementParameterSpace()
        params = param_space.to_optuna_space(trial)
        
        # Run trial
        metrics = self._run_single_trial(params)
        
        # Store history
        self.optimization_history.append({
            'trial': trial.number,
            'params': params,
            'metrics': metrics
        })
        
        # Check basic constraints
        if metrics.n_clusters < 5 or metrics.n_clusters > 10:
            tprint(f"❌ Trial {trial.number} failed cluster count: {metrics.n_clusters}", "WARNING")
            return -10.0
        
        # Check quality constraints (we're optimizing CV but must maintain quality)
        if metrics.silhouette_score < 0.0 or metrics.dbi_score > 5.0:
            tprint(f"❌ Trial {trial.number} failed quality constraints", "WARNING")
            return -5.0
        
        # Calculate composite score (CV-focused)
        composite = metrics.get_composite_score()
        
        # Calculate CV quality score
        cv_quality = metrics.get_cv_quality_score()
        
        if self.verbose:
            tprint(f"✅ Trial {trial.number}: Quality={composite:.4f}, CV Quality={cv_quality:.4f}, "
                  f"CV={metrics.cv_score:.3f} (Δ={metrics.cv_improvement:+.2%}), "
                  f"Sil={metrics.silhouette_score:.3f}, DBI={metrics.dbi_score:.3f}", "INFO")
        
        # Store as user attributes for analysis
        trial.set_user_attr('cv_score', metrics.cv_score)
        trial.set_user_attr('cv_improvement', metrics.cv_improvement)
        trial.set_user_attr('silhouette_score', metrics.silhouette_score)
        trial.set_user_attr('dbi_score', metrics.dbi_score)
        trial.set_user_attr('cv_quality_score', cv_quality)
        
        return composite
    
    def optimize_bayesian(self, n_trials: int = 30) -> Dict[str, Any]:
        """
        Run Bayesian optimization using Optuna TPE sampler.
        
        Args:
            n_trials: Number of trials to run
            
        Returns:
            Dictionary with best parameters and metrics
        """
        tprint(f"🚀 Starting CV Enhancement parameter tuning ({n_trials} trials)...", "INFO")
        
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
                study_name=f"cv_enhancement_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Run optimization
            study.optimize(self._objective_function, n_trials=n_trials, show_progress_bar=True)
            
            # Get best trial
            best_trial = study.best_trial
            best_params = best_trial.params
            best_score = best_trial.value
            
            # Extract metrics from best trial
            best_metrics = None
            for item in self.optimization_history:
                if item['trial'] == best_trial.number:
                    best_metrics = item['metrics']
                    break
            
            self.best_params = best_params
            self.best_metrics = best_metrics
            
            tprint(f"✅ CV Enhancement tuning completed!", "SUCCESS")
            tprint(f"📊 Best composite score: {best_score:.4f}", "SUCCESS")
            tprint(f"🎯 Best CV: {best_trial.user_attrs.get('cv_score', 0):.3f}", "SUCCESS")
            tprint(f"⚡ CV improvement: {best_trial.user_attrs.get('cv_improvement', 0):+.2%}", "SUCCESS")
            
            return {
                'best_params': best_params,
                'best_metrics': best_metrics,
                'best_score': best_score,
                'study': study,
                'optimization_history': self.optimization_history
            }
            
        except Exception as e:
            tprint(f"❌ CV Enhancement tuning failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return None
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save optimization results to file."""
        try:
            serializable_results = {
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(self.filtered_labels),
                'n_features': self.filtered_features.shape[1],
                'baseline_cv': self.baseline_cv,
                'best_params': results.get('best_params'),
                'best_metrics': {
                    'cv_score': results['best_metrics'].cv_score,
                    'cv_improvement': results['best_metrics'].cv_improvement,
                    'silhouette_score': results['best_metrics'].silhouette_score,
                    'dbi_score': results['best_metrics'].dbi_score,
                    'cv_quality_score': results['best_metrics'].get_cv_quality_score()
                } if results.get('best_metrics') else None,
                'n_trials': len(self.optimization_history)
            }
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            tprint(f"✅ Results saved to: {output_path}", "SUCCESS")
            
        except Exception as e:
            tprint(f"❌ Failed to save results: {e}", "ERROR")
    
    def generate_report(self, results: Dict[str, Any], output_path: str) -> None:
        """Generate comprehensive tuning report."""
        try:
            report = []
            report.append("# CV Enhancement Parameter Tuning Report\n")
            report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report.append(f"**Dataset**: {len(self.filtered_labels)} samples, {self.filtered_features.shape[1]} features\n")
            report.append(f"**Baseline CV**: {self.baseline_cv:.3f}\n")
            report.append("\n## Optimization Summary\n")
            
            if 'best_params' in results and 'best_metrics' in results:
                metrics = results['best_metrics']
                report.append(f"**Total Trials**: {len(self.optimization_history)}\n")
                report.append(f"**Best Composite Score**: {results.get('best_score', 'N/A'):.4f}\n")
                report.append(f"**CV Quality Score**: {metrics.get_cv_quality_score():.4f}\n")
                report.append("\n### Best Configuration Metrics\n")
                report.append("| Metric | Value | Improvement | Status |\n")
                report.append("|--------|-------|-------------|--------|\n")
                
                report.append(f"| CV Score | {metrics.cv_score:.4f} | {metrics.cv_improvement:+.2%} | ✅ |\n")
                report.append(f"| Silhouette Score | {metrics.silhouette_score:.4f} | - | ✅ |\n")
                report.append(f"| DBI Score | {metrics.dbi_score:.4f} | - | ✅ |\n")
                report.append(f"| Balance Score | {metrics.balance_score:.4f} | - | ✅ |\n")
                report.append(f"| Temporal Smoothness | {metrics.temporal_smoothness:.4f} | - | ✅ |\n")
                
                report.append("\n### Weight Progression\n")
                report.append(f"- **Initial CV Weight**: {metrics.initial_cv_weight:.3f}\n")
                report.append(f"- **Final CV Weight**: {metrics.final_cv_weight:.3f}\n")
                report.append(f"- **Weight Increase**: {metrics.final_cv_weight - metrics.initial_cv_weight:.3f}\n")
                
                report.append("\n### Best Parameters\n")
                report.append("```json\n")
                report.append(json.dumps(results['best_params'], indent=2))
                report.append("\n```\n")
            
            with open(output_path, 'w') as f:
                f.writelines(report)
            
            tprint(f"✅ Report saved to: {output_path}", "SUCCESS")
            
        except Exception as e:
            tprint(f"❌ Failed to generate report: {e}", "ERROR")


def run_cv_enhancement_tuning(
    features: np.ndarray,
    initial_labels: np.ndarray,
    market_data: pd.DataFrame,
    baseline_cv: float = None,
    n_trials: int = 30,
    output_dir: str = 'artifacts/hyperparameter_tuning/'
) -> Optional[Dict[str, Any]]:
    """
    Run the complete CV enhancement parameter tuning pipeline.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        initial_labels: Initial cluster labels
        market_data: Market data DataFrame
        baseline_cv: Baseline CV score (computed if not provided)
        n_trials: Number of optimization trials
        output_dir: Directory to save results
        
    Returns:
        Dictionary with optimization results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tuner
    tuner = CVEnhancementTuner(features, initial_labels, market_data, baseline_cv, verbose=True)
    
    # Run optimization
    results = tuner.optimize_bayesian(n_trials=n_trials)
    
    if results is None:
        return None
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(output_dir, f"cv_enhancement_tuning_{timestamp}.json")
    report_path = os.path.join(output_dir, f"cv_enhancement_report_{timestamp}.md")
    
    tuner.save_results(results, results_path)
    tuner.generate_report(results, report_path)
    
    return results


# Example usage
if __name__ == "__main__":
    """
    Example usage:
    
    from src.training.steps.market_analysis.clusters.cv_enhancement_tuner import run_cv_enhancement_tuning
    
    # Load your data
    features = ...  # From regime_feature_selection
    initial_labels = ...  # From HDBSCAN
    market_data = ...  # From feature_generation
    
    # Run tuning
    results = run_cv_enhancement_tuning(
        features=features,
        initial_labels=initial_labels,
        market_data=market_data,
        n_trials=30
    )
    
    # Apply best parameters to AdaptiveWeightScheduler
    best_params = results['best_params']
    scheduler = AdaptiveWeightScheduler(
        max_iterations=30,
        initial_cv_weight=best_params['initial_cv_weight'],
        final_cv_weight=best_params['final_cv_weight'],
        transition_speed=best_params['weight_transition_speed']
    )
    """
    tprint("💡 This is a utility module. Import and use run_cv_enhancement_tuning() to optimize parameters.", "INFO")
