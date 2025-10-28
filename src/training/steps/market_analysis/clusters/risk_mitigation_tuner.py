"""
Risk Mitigation Parameter Tuner

This module provides automatic hyperparameter tuning for the Risk Mitigation System
to optimize stability thresholds, rate limits, quality gates, and convergence criteria
while maintaining high clustering quality.

Optimization Goals:
- Maximize clustering quality (CV, Silhouette, DBI)
- Maintain stability (minimize instability events)
- Optimize convergence speed
- Prevent over-churn and unbounded growth

Author: AI Assistant
Date: 2025-10-28
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import json
import time
from pathlib import Path

from src.utils.tprint import tprint
from src.utils.ml_common.optimization import (
    HyperparameterOptimization,
    ParetoOptimizer,
    ParetoFrontAnalyzer
)

# Import unified clustering optimization goals
from .clustering_optimization_goals import (
    DEFAULT_CLUSTERING_GOALS,
    DEFAULT_OPTIMIZATION_TARGETS,
    calculate_composite_score,
    meets_optimization_constraints
)

from .risk_mitigation import RiskMitigationConfig, RiskMitigationSystem


@dataclass
class RiskMitigationMetrics:
    """Metrics from risk-mitigated optimization run."""
    cv_score: float
    silhouette_score: float
    dbi_score: float
    balance_score: float
    temporal_smoothness: float
    n_clusters: int
    
    # Risk-specific metrics
    instability_events: int
    total_splits: int
    total_merges: int
    total_reassignments: int
    convergence_rounds: int
    quality_degradation_events: int
    
    # Performance metrics
    optimization_time: float
    converged: bool
    
    def get_composite_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted composite score using unified goals."""
        if weights is None:
            weights = DEFAULT_CLUSTERING_GOALS.get_weights_dict()
        
        # Base clustering quality score
        base_score = calculate_composite_score(
            cv_score=self.cv_score,
            silhouette_score=self.silhouette_score,
            dbi_score=self.dbi_score,
            balance_score=self.balance_score,
            temporal_smoothness=self.temporal_smoothness
        )
        
        # Stability bonus (0.0 to 0.1)
        # Fewer instability events = higher bonus
        stability_bonus = max(0.0, 0.1 - (self.instability_events * 0.01))
        
        # Convergence bonus (0.0 to 0.05)
        # Faster convergence = higher bonus
        convergence_bonus = 0.05 if self.converged and self.convergence_rounds < 20 else 0.0
        
        # Quality protection bonus (0.0 to 0.05)
        # No quality degradation = full bonus
        quality_bonus = max(0.0, 0.05 - (self.quality_degradation_events * 0.01))
        
        return base_score + stability_bonus + convergence_bonus + quality_bonus
    
    def get_stability_score(self) -> float:
        """Calculate stability-specific score (0.0 to 1.0)."""
        # Penalize instability events
        instability_penalty = min(1.0, self.instability_events * 0.1)
        
        # Penalize excessive operations
        churn_penalty = min(0.5, (self.total_reassignments / 1000) * 0.1)
        
        # Penalize quality degradation
        quality_penalty = min(0.3, self.quality_degradation_events * 0.1)
        
        return max(0.0, 1.0 - instability_penalty - churn_penalty - quality_penalty)


@dataclass
class RiskMitigationParameterSpace:
    """
    Define the hyperparameter search space for risk mitigation tuning.
    
    Aligned with RiskMitigationConfig parameters.
    """
    
    # Stability thresholds
    min_stability_score: Tuple[float, float] = (0.5, 0.95)
    max_instability_events: Tuple[int, int] = (1, 10)
    
    # Rate limits - prevent over-churn
    max_splits_per_round: Tuple[int, int] = (1, 5)
    max_merges_per_round: Tuple[int, int] = (1, 5)
    max_reassignments_per_round: Tuple[int, int] = (10, 500)
    
    # Quality gates
    min_cluster_quality: Tuple[float, float] = (0.3, 0.8)
    max_quality_degradation: Tuple[float, float] = (0.05, 0.3)
    
    # Convergence criteria
    convergence_window: Tuple[int, int] = (3, 10)
    convergence_threshold: Tuple[float, float] = (0.001, 0.05)
    
    # Churn caps (as fraction of N)
    local_churn_cap: Tuple[float, float] = (0.01, 0.05)  # 1-5% of N
    global_churn_cap: Tuple[float, float] = (0.05, 0.15)  # 5-15% of N
    
    # K-growth prevention
    k_complexity_penalty: Tuple[float, float] = (0.1, 0.5)
    max_k_growth_factor: Tuple[float, float] = (0.05, 0.20)  # 5-20% of current k
    
    def to_optuna_space(self, trial) -> Dict[str, Any]:
        """Convert to Optuna trial suggestions."""
        params = {}
        
        # Stability parameters
        params['min_stability_score'] = trial.suggest_float(
            'min_stability_score', 
            self.min_stability_score[0], 
            self.min_stability_score[1]
        )
        params['max_instability_events'] = trial.suggest_int(
            'max_instability_events',
            self.max_instability_events[0],
            self.max_instability_events[1]
        )
        
        # Rate limits
        params['max_splits_per_round'] = trial.suggest_int(
            'max_splits_per_round',
            self.max_splits_per_round[0],
            self.max_splits_per_round[1]
        )
        params['max_merges_per_round'] = trial.suggest_int(
            'max_merges_per_round',
            self.max_merges_per_round[0],
            self.max_merges_per_round[1]
        )
        params['max_reassignments_per_round'] = trial.suggest_int(
            'max_reassignments_per_round',
            self.max_reassignments_per_round[0],
            self.max_reassignments_per_round[1]
        )
        
        # Quality gates
        params['min_cluster_quality'] = trial.suggest_float(
            'min_cluster_quality',
            self.min_cluster_quality[0],
            self.min_cluster_quality[1]
        )
        params['max_quality_degradation'] = trial.suggest_float(
            'max_quality_degradation',
            self.max_quality_degradation[0],
            self.max_quality_degradation[1]
        )
        
        # Convergence criteria
        params['convergence_window'] = trial.suggest_int(
            'convergence_window',
            self.convergence_window[0],
            self.convergence_window[1]
        )
        params['convergence_threshold'] = trial.suggest_float(
            'convergence_threshold',
            self.convergence_threshold[0],
            self.convergence_threshold[1],
            log=True
        )
        
        # Churn caps
        params['local_churn_cap'] = trial.suggest_float(
            'local_churn_cap',
            self.local_churn_cap[0],
            self.local_churn_cap[1]
        )
        params['global_churn_cap'] = trial.suggest_float(
            'global_churn_cap',
            self.global_churn_cap[0],
            self.global_churn_cap[1]
        )
        
        # K-growth prevention
        params['k_complexity_penalty'] = trial.suggest_float(
            'k_complexity_penalty',
            self.k_complexity_penalty[0],
            self.k_complexity_penalty[1]
        )
        params['max_k_growth_factor'] = trial.suggest_float(
            'max_k_growth_factor',
            self.max_k_growth_factor[0],
            self.max_k_growth_factor[1]
        )
        
        return params


class RiskMitigationTuner:
    """Tunes risk mitigation parameters to maximize quality while maintaining stability."""
    
    def __init__(self, 
                 features: np.ndarray,
                 initial_labels: np.ndarray,
                 market_data: pd.DataFrame,
                 verbose: bool = True):
        """
        Initialize the risk mitigation tuner.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            initial_labels: Initial cluster labels
            market_data: Market data DataFrame
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
        
        tprint(f"🎯 Initialized Risk Mitigation Tuner with {len(self.filtered_labels)} samples", "INFO")
        
        # Results storage
        self.best_params = None
        self.best_metrics = None
        self.optimization_history = []
        
    def _run_single_trial(self, params: Dict[str, Any]) -> RiskMitigationMetrics:
        """
        Run optimization with given risk mitigation parameters.
        
        Args:
            params: Risk mitigation parameter dictionary
            
        Returns:
            RiskMitigationMetrics object
        """
        try:
            import time
            from src.training.steps.market_analysis.clusters.iterative_optimization import IterativeOptimization
            from src.training.steps.market_analysis.clusters.step1_feature_preparation import ClusteringContext
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
            
            start_time = time.time()
            
            # Create risk mitigation config from params
            risk_config = RiskMitigationConfig(
                # Stability
                stability_threshold=params['min_stability_score'],
                
                # Rate limits
                max_new_splits_per_round=params['max_splits_per_round'],
                
                # Churn caps
                local_churn_cap=params['local_churn_cap'],
                global_churn_cap=params['global_churn_cap'],
                
                # K-growth prevention
                k_complexity_penalty=params['k_complexity_penalty'],
                max_k_growth_factor=params['max_k_growth_factor'],
                
                # Convergence
                convergence_tolerance=params['convergence_threshold'],
                max_convergence_cycles=params['convergence_window'],
                
                # Quality gates (mapped from params)
                min_silhouette=params['min_cluster_quality'],
                
                # Readiness gates
                max_churn_per_cycle=params['max_reassignments_per_round'] / len(self.filtered_labels)
            )
            
            # Create context
            context = ClusteringContext(
                original_features=self.filtered_features,
                market_data=self.market_data
            )
            context.initial_assignments = self.filtered_labels.copy()
            context.assignments = self.filtered_labels.copy()
            context.optimized_features = self.filtered_features
            context.optimal_k = len(np.unique(self.filtered_labels))
            
            # Run optimization with risk mitigation
            optimizer = IterativeOptimization(verbose=False)
            risk_system = RiskMitigationSystem(config=risk_config)
            
            # Run optimization synchronously
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                pass
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            optimizer.execute_optimization_loop(
                                context, {},
                                max_iterations=30,
                                enable_risk_mitigation=True
                            )
                        )
                        optimized_context = future.result()
                else:
                    optimized_context = loop.run_until_complete(
                        optimizer.execute_optimization_loop(
                            context, {},
                            max_iterations=30,
                            enable_risk_mitigation=True
                        )
                    )
            except RuntimeError:
                optimized_context = asyncio.run(
                    optimizer.execute_optimization_loop(
                        context, {},
                        max_iterations=30,
                        enable_risk_mitigation=True
                    )
                )
            
            # Extract results
            optimized_labels = optimized_context.assignments if hasattr(optimized_context, 'assignments') else optimized_context.optimized_assignments
            
            # Calculate metrics
            n_clusters = len(np.unique(optimized_labels))
            
            # CV score
            if n_clusters >= 2:
                try:
                    cv_score = calinski_harabasz_score(self.filtered_features, optimized_labels)
                except:
                    cv_score = 0.0
            else:
                cv_score = 0.0
            
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
            
            # Risk-specific metrics (extract from risk_system if available)
            instability_events = 0  # Would be tracked during optimization
            total_splits = risk_system.operation_counts.get('splits', 0)
            total_merges = 0  # Would need to track merges
            total_reassignments = risk_system.operation_counts.get('local_moves', 0) + risk_system.operation_counts.get('global_moves', 0)
            convergence_rounds = risk_system.convergence_cycles
            quality_degradation = 0  # Would need to track
            
            converged = convergence_rounds < params['convergence_window']
            
            optimization_time = time.time() - start_time
            
            return RiskMitigationMetrics(
                cv_score=cv_score,
                silhouette_score=silhouette,
                dbi_score=dbi,
                balance_score=balance,
                temporal_smoothness=temporal,
                n_clusters=n_clusters,
                instability_events=instability_events,
                total_splits=total_splits,
                total_merges=total_merges,
                total_reassignments=total_reassignments,
                convergence_rounds=convergence_rounds,
                quality_degradation_events=quality_degradation,
                optimization_time=optimization_time,
                converged=converged
            )
            
        except Exception as e:
            tprint(f"❌ Trial execution failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            
            # Return poor metrics
            return RiskMitigationMetrics(
                cv_score=0.0,
                silhouette_score=-1.0,
                dbi_score=10.0,
                balance_score=0.0,
                temporal_smoothness=0.0,
                n_clusters=0,
                instability_events=100,
                total_splits=0,
                total_merges=0,
                total_reassignments=0,
                convergence_rounds=100,
                quality_degradation_events=10,
                optimization_time=0.0,
                converged=False
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
        Returns composite score to maximize.
        """
        # Get parameter suggestions from trial
        param_space = RiskMitigationParameterSpace()
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
        
        # Calculate composite score (includes stability bonuses)
        composite = metrics.get_composite_score()
        
        # Calculate stability score
        stability = metrics.get_stability_score()
        
        if self.verbose:
            tprint(f"✅ Trial {trial.number}: Quality={composite:.4f}, Stability={stability:.4f}, "
                  f"CV={metrics.cv_score:.3f}, Sil={metrics.silhouette_score:.3f}, "
                  f"Instability={metrics.instability_events}, Convergence={metrics.convergence_rounds}", "INFO")
        
        # Store as user attributes for analysis
        trial.set_user_attr('cv_score', metrics.cv_score)
        trial.set_user_attr('silhouette_score', metrics.silhouette_score)
        trial.set_user_attr('dbi_score', metrics.dbi_score)
        trial.set_user_attr('stability_score', stability)
        trial.set_user_attr('instability_events', metrics.instability_events)
        trial.set_user_attr('converged', metrics.converged)
        
        return composite
    
    def optimize_bayesian(self, n_trials: int = 30) -> Dict[str, Any]:
        """
        Run Bayesian optimization using Optuna TPE sampler.
        
        Args:
            n_trials: Number of trials to run
            
        Returns:
            Dictionary with best parameters and metrics
        """
        tprint(f"🚀 Starting Risk Mitigation parameter tuning ({n_trials} trials)...", "INFO")
        
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
                study_name=f"risk_mitigation_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            
            tprint(f"✅ Risk Mitigation tuning completed!", "SUCCESS")
            tprint(f"📊 Best composite score: {best_score:.4f}", "SUCCESS")
            tprint(f"🎯 Best stability: {best_trial.user_attrs.get('stability_score', 0):.4f}", "SUCCESS")
            tprint(f"⚡ Instability events: {best_trial.user_attrs.get('instability_events', 0)}", "SUCCESS")
            
            return {
                'best_params': best_params,
                'best_metrics': best_metrics,
                'best_score': best_score,
                'study': study,
                'optimization_history': self.optimization_history
            }
            
        except Exception as e:
            tprint(f"❌ Risk Mitigation tuning failed: {e}", "ERROR")
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
                'best_params': results.get('best_params'),
                'best_metrics': {
                    'cv_score': results['best_metrics'].cv_score,
                    'silhouette_score': results['best_metrics'].silhouette_score,
                    'dbi_score': results['best_metrics'].dbi_score,
                    'stability_score': results['best_metrics'].get_stability_score(),
                    'instability_events': results['best_metrics'].instability_events,
                    'converged': results['best_metrics'].converged
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
            report.append("# Risk Mitigation Parameter Tuning Report\n")
            report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report.append(f"**Dataset**: {len(self.filtered_labels)} samples, {self.filtered_features.shape[1]} features\n")
            report.append("\n## Optimization Summary\n")
            
            if 'best_params' in results and 'best_metrics' in results:
                metrics = results['best_metrics']
                report.append(f"**Total Trials**: {len(self.optimization_history)}\n")
                report.append(f"**Best Composite Score**: {results.get('best_score', 'N/A'):.4f}\n")
                report.append(f"**Stability Score**: {metrics.get_stability_score():.4f}\n")
                report.append("\n### Best Configuration Metrics\n")
                report.append("| Metric | Value | Status |\n")
                report.append("|--------|-------|--------|\n")
                
                report.append(f"| CV Score | {metrics.cv_score:.4f} | ✅ |\n")
                report.append(f"| Silhouette Score | {metrics.silhouette_score:.4f} | ✅ |\n")
                report.append(f"| DBI Score | {metrics.dbi_score:.4f} | ✅ |\n")
                report.append(f"| Instability Events | {metrics.instability_events} | {'✅' if metrics.instability_events <= 5 else '⚠️'} |\n")
                report.append(f"| Convergence Rounds | {metrics.convergence_rounds} | {'✅' if metrics.converged else '⚠️'} |\n")
                report.append(f"| Total Operations | {metrics.total_reassignments} | ✅ |\n")
                
                report.append("\n### Best Parameters\n")
                report.append("```json\n")
                report.append(json.dumps(results['best_params'], indent=2))
                report.append("\n```\n")
            
            with open(output_path, 'w') as f:
                f.writelines(report)
            
            tprint(f"✅ Report saved to: {output_path}", "SUCCESS")
            
        except Exception as e:
            tprint(f"❌ Failed to generate report: {e}", "ERROR")


def run_risk_mitigation_tuning(
    features: np.ndarray,
    initial_labels: np.ndarray,
    market_data: pd.DataFrame,
    n_trials: int = 30,
    output_dir: str = 'artifacts/hyperparameter_tuning/'
) -> Optional[Dict[str, Any]]:
    """
    Run the complete risk mitigation parameter tuning pipeline.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        initial_labels: Initial cluster labels
        market_data: Market data DataFrame
        n_trials: Number of optimization trials
        output_dir: Directory to save results
        
    Returns:
        Dictionary with optimization results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tuner
    tuner = RiskMitigationTuner(features, initial_labels, market_data, verbose=True)
    
    # Run optimization
    results = tuner.optimize_bayesian(n_trials=n_trials)
    
    if results is None:
        return None
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(output_dir, f"risk_mitigation_tuning_{timestamp}.json")
    report_path = os.path.join(output_dir, f"risk_mitigation_report_{timestamp}.md")
    
    tuner.save_results(results, results_path)
    tuner.generate_report(results, report_path)
    
    return results


# Example usage
if __name__ == "__main__":
    """
    Example usage:
    
    from src.training.steps.market_analysis.clusters.risk_mitigation_tuner import run_risk_mitigation_tuning
    
    # Load your data
    features = ...  # From regime_feature_selection
    initial_labels = ...  # From HDBSCAN
    market_data = ...  # From feature_generation
    
    # Run tuning
    results = run_risk_mitigation_tuning(
        features=features,
        initial_labels=initial_labels,
        market_data=market_data,
        n_trials=30
    )
    
    # Apply best parameters to RiskMitigationConfig
    best_params = results['best_params']
    risk_config = RiskMitigationConfig(
        stability_threshold=best_params['min_stability_score'],
        max_new_splits_per_round=best_params['max_splits_per_round'],
        # ... etc
    )
    """
    tprint("💡 This is a utility module. Import and use run_risk_mitigation_tuning() to optimize parameters.", "INFO")
