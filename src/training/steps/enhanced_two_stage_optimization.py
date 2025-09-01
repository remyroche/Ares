#!/usr/bin/env python3
"""
Enhanced Two-Stage Optimization: Adaptive Grid/Random Search → TPE

This module implements a sophisticated two-stage optimization approach for DBSCAN parameters
in the enhanced clustering system. It combines fast initial exploration with intelligent
refinement using Tree-structured Parzen Estimators (TPE).
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root))

# Import pipeline standards
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards

# Standardized import management
REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "src.utils.logger",
    "src.utils.centralized_decorators"
]

# Validate environment dependencies
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

# Safe imports with fallbacks
centralized_decorators = PipelineStandards.safe_import("src.utils.centralized_decorators", None)
system_logger = PipelineStandards.safe_import("src.utils.logger", None)

# Fallback functions if imports fail
def create_fallback_logger():
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("EnhancedTwoStageOptimization")

def create_fallback_decorator():
    def decorator(func):
        return func
    return decorator

# Initialize fallbacks
if system_logger is None:
    system_logger = create_fallback_logger()

if centralized_decorators is None:
    handle_errors = create_fallback_decorator()
    monitor_step_execution = create_fallback_decorator()
    with_tracing_span = create_fallback_decorator()
    memory_efficient = create_fallback_decorator()
    resource_monitor = create_fallback_decorator()
else:
    handle_errors = centralized_decorators.handle_errors
    monitor_step_execution = centralized_decorators.monitor_step_execution
    with_tracing_span = centralized_decorators.with_tracing_span
    memory_efficient = centralized_decorators.memory_efficient
    resource_monitor = centralized_decorators.resource_monitor

logger = system_logger.getChild("EnhancedTwoStageOptimization")

class EnhancedTwoStageOptimizer:
    """
    Enhanced two-stage optimization for DBSCAN parameters.
    
    Combines fast initial exploration (adaptive grid search or random search)
    with intelligent refinement using Tree-structured Parzen Estimators (TPE).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the two-stage optimizer.
        
        Args:
            config: Configuration dictionary with optimization parameters
        """
        self.config = config
        self.logger = logger
        
        # Default configuration
        self.default_config = {
            "max_evaluations": 50,
            "stage1_ratio": 0.6,
            "min_quality_threshold": 0.3,
            "robustness_level": "medium",  # "low", "medium", "high"
            "search_space_expansion": 1.5,  # Multiplier for TPE search space
            "multiple_regions": True,  # Whether to refine multiple promising regions
            "region_threshold": 0.8,  # Threshold for considering a region promising
            "random_seed": 42
        }
        
        # Update with provided config
        self.config = {**self.default_config, **config}
        
        # Set random seed
        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])
        
        # Initialize results tracking
        self.optimization_history = []
        self.stage1_results = []
        self.stage2_results = []
        
    @handle_errors(exceptions=(Exception,), default_return={"success": False, "error": "Optimization failed"})
    @monitor_step_execution
    @with_tracing_span("enhanced_two_stage_optimization")
    @memory_efficient
    @resource_monitor
    def optimize_dbscan_parameters(
        self, 
        features: np.ndarray, 
        max_evaluations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform two-stage optimization for DBSCAN parameters.
        
        Args:
            features: Input features for clustering
            max_evaluations: Override for maximum evaluations
            
        Returns:
            Dictionary with optimization results
        """
        if max_evaluations is None:
            max_evaluations = self.config["max_evaluations"]
            
        self.logger.info(f"🚀 Starting Enhanced Two-Stage Optimization ({max_evaluations} evaluations)")
        
        # Analyze problem characteristics
        problem_analysis = self._analyze_problem_characteristics(features)
        self.logger.info(f"📊 Problem Analysis: {problem_analysis}")
        
        # Determine stage allocation
        stage1_evaluations = int(max_evaluations * self.config["stage1_ratio"])
        stage2_evaluations = max_evaluations - stage1_evaluations
        
        self.logger.info(f"🎯 Stage 1: {stage1_evaluations} evaluations")
        self.logger.info(f"🔍 Stage 2: {stage2_evaluations} evaluations")
        
        # Stage 1: Fast exploration
        stage1_results = self._stage1_fast_exploration(features, stage1_evaluations, problem_analysis)
        
        # Stage 2: TPE refinement
        stage2_results = self._stage2_tpe_refinement(
            features, stage1_results, stage2_evaluations, problem_analysis
        )
        
        # Combine results
        final_results = self._combine_optimization_results(stage1_results, stage2_results)
        
        self.logger.info(f"✅ Optimization completed successfully")
        self.logger.info(f"📈 Final Score: {final_results['best_score']:.4f}")
        self.logger.info(f"🔧 Best Parameters: {final_results['best_params']}")
        
        return final_results
    
    def _analyze_problem_characteristics(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Analyze problem characteristics to guide optimization strategy.
        
        Args:
            features: Input features
            
        Returns:
            Problem analysis dictionary
        """
        n_samples, n_features = features.shape
        
        # Calculate feature statistics
        feature_std = np.std(features, axis=0)
        feature_range = np.max(features, axis=0) - np.min(features, axis=0)
        
        # Determine problem complexity
        if n_samples < 2000:
            complexity = "small"
            stage1_method = "adaptive_grid"
            stage1_ratio = 0.7
        elif n_samples < 10000:
            complexity = "medium"
            stage1_method = "adaptive_grid"
            stage1_ratio = 0.6
        else:
            complexity = "large"
            stage1_method = "random_search"
            stage1_ratio = 0.4
        
        # Determine feature characteristics
        if np.std(feature_std) < 0.1:
            feature_type = "uniform"
        elif np.std(feature_std) > 1.0:
            feature_type = "diverse"
        else:
            feature_type = "mixed"
        
        analysis = {
            "n_samples": n_samples,
            "n_features": n_features,
            "complexity": complexity,
            "feature_type": feature_type,
            "stage1_method": stage1_method,
            "stage1_ratio": stage1_ratio,
            "feature_std_mean": np.mean(feature_std),
            "feature_std_std": np.std(feature_std),
            "feature_range_mean": np.mean(feature_range),
            "feature_range_std": np.std(feature_range)
        }
        
        return analysis
    
    def _stage1_fast_exploration(
        self, 
        features: np.ndarray, 
        n_evaluations: int, 
        problem_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Stage 1: Fast exploration using adaptive grid search or random search.
        
        Args:
            features: Input features
            n_evaluations: Number of evaluations for stage 1
            problem_analysis: Problem analysis results
            
        Returns:
            Stage 1 results dictionary
        """
        method = problem_analysis["stage1_method"]
        
        if method == "adaptive_grid":
            return self._adaptive_grid_search_stage1(features, n_evaluations)
        else:
            return self._random_search_stage1(features, n_evaluations)
    
    def _adaptive_grid_search_stage1(self, features: np.ndarray, n_evaluations: int) -> Dict[str, Any]:
        """
        Enhanced adaptive grid search for stage 1 with multiple region identification.
        
        Args:
            features: Input features
            n_evaluations: Number of evaluations
            
        Returns:
            Stage 1 results with multiple promising regions
        """
        self.logger.info("📊 Stage 1: Enhanced Adaptive Grid Search")
        
        # Phase 1: Coarse grid exploration
        eps_coarse = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        min_samples_coarse = [5, 8, 12, 18, 25, 35, 45, 50]
        
        coarse_results = []
        best_score = -float('inf')
        best_params = None
        
        # Evaluate coarse grid
        for eps in eps_coarse:
            for min_samples in min_samples_coarse:
                clustering = self._run_dbscan(features, eps, min_samples)
                if clustering is not None:
                    score = self._calculate_composite_score(features, clustering.labels_)
                    result = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'score': score,
                        'n_clusters': len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                    }
                    coarse_results.append(result)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'eps': eps, 'min_samples': min_samples}
        
        # Identify multiple promising regions
        promising_regions = self._identify_promising_regions(coarse_results)
        
        # Phase 2: Refine promising regions
        refinement_results = []
        remaining_evaluations = n_evaluations - len(coarse_results)
        
        if remaining_evaluations > 0 and promising_regions:
            evaluations_per_region = remaining_evaluations // len(promising_regions)
            
            for region in promising_regions:
                region_results = self._refine_region(
                    features, region, evaluations_per_region
                )
                refinement_results.extend(region_results)
        
        # Combine results
        all_results = coarse_results + refinement_results
        best_result = max(all_results, key=lambda x: x['score'])
        
        stage1_results = {
            'method': 'adaptive_grid',
            'best_params': best_result,
            'best_score': best_result['score'],
            'promising_regions': promising_regions,
            'all_results': all_results,
            'n_evaluations': len(all_results)
        }
        
        self.stage1_results = stage1_results
        return stage1_results
    
    def _identify_promising_regions(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify multiple promising regions from coarse grid results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            List of promising regions
        """
        if not results:
            return []
        
        # Sort results by score
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        best_score = sorted_results[0]['score']
        
        # Calculate threshold for promising regions
        threshold = best_score * self.config["region_threshold"]
        
        # Group results by proximity
        regions = []
        used_indices = set()
        
        for i, result in enumerate(sorted_results):
            if i in used_indices or result['score'] < threshold:
                continue
            
            # Start a new region
            region = {
                'center_eps': result['eps'],
                'center_min_samples': result['min_samples'],
                'center_score': result['score'],
                'members': [result],
                'bounds': {
                    'eps_min': result['eps'] * 0.5,
                    'eps_max': result['eps'] * 1.5,
                    'min_samples_min': max(5, result['min_samples'] - 10),
                    'min_samples_max': min(50, result['min_samples'] + 10)
                }
            }
            
            used_indices.add(i)
            
            # Find nearby results
            for j, other_result in enumerate(sorted_results):
                if j in used_indices or other_result['score'] < threshold:
                    continue
                
                # Check if results are nearby
                eps_distance = abs(other_result['eps'] - result['eps']) / result['eps']
                min_samples_distance = abs(other_result['min_samples'] - result['min_samples']) / result['min_samples']
                
                if eps_distance < 0.3 and min_samples_distance < 0.3:
                    region['members'].append(other_result)
                    used_indices.add(j)
                    
                    # Update bounds
                    region['bounds']['eps_min'] = min(region['bounds']['eps_min'], other_result['eps'] * 0.5)
                    region['bounds']['eps_max'] = max(region['bounds']['eps_max'], other_result['eps'] * 1.5)
                    region['bounds']['min_samples_min'] = min(region['bounds']['min_samples_min'], max(5, other_result['min_samples'] - 10))
                    region['bounds']['min_samples_max'] = max(region['bounds']['min_samples_max'], min(50, other_result['min_samples'] + 10))
            
            regions.append(region)
        
        # Limit number of regions based on robustness level
        if self.config["robustness_level"] == "low":
            regions = regions[:1]
        elif self.config["robustness_level"] == "medium":
            regions = regions[:2]
        else:  # high
            regions = regions[:3]
        
        self.logger.info(f"🎯 Identified {len(regions)} promising regions")
        for i, region in enumerate(regions):
            self.logger.info(f"   Region {i+1}: eps={region['center_eps']:.3f}, min_samples={region['center_min_samples']}, score={region['center_score']:.4f}")
        
        return regions
    
    def _refine_region(
        self, 
        features: np.ndarray, 
        region: Dict[str, Any], 
        n_evaluations: int
    ) -> List[Dict[str, Any]]:
        """
        Refine a specific region with fine-grained search.
        
        Args:
            features: Input features
            region: Region to refine
            n_evaluations: Number of evaluations for this region
            
        Returns:
            List of refinement results
        """
        bounds = region['bounds']
        center_eps = region['center_eps']
        center_min_samples = region['center_min_samples']
        
        results = []
        
        # Generate fine grid around region center
        eps_fine = np.linspace(bounds['eps_min'], bounds['eps_max'], min(10, n_evaluations // 2))
        min_samples_fine = np.linspace(bounds['min_samples_min'], bounds['min_samples_max'], min(10, n_evaluations // 2), dtype=int)
        
        # Evaluate fine grid
        for eps in eps_fine:
            for min_samples in min_samples_fine:
                clustering = self._run_dbscan(features, eps, int(min_samples))
                if clustering is not None:
                    score = self._calculate_composite_score(features, clustering.labels_)
                    result = {
                        'eps': eps,
                        'min_samples': int(min_samples),
                        'score': score,
                        'n_clusters': len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                    }
                    results.append(result)
        
        return results
    
    def _random_search_stage1(self, features: np.ndarray, n_evaluations: int) -> Dict[str, Any]:
        """
        Random search for stage 1.
        
        Args:
            features: Input features
            n_evaluations: Number of evaluations
            
        Returns:
            Stage 1 results
        """
        self.logger.info("🎲 Stage 1: Random Search")
        
        results = []
        best_score = -float('inf')
        best_params = None
        
        for i in range(n_evaluations):
            eps = random.uniform(0.1, 2.0)
            min_samples = random.randint(5, 50)
            
            clustering = self._run_dbscan(features, eps, min_samples)
            if clustering is not None:
                score = self._calculate_composite_score(features, clustering.labels_)
                result = {
                    'eps': eps,
                    'min_samples': min_samples,
                    'score': score,
                    'n_clusters': len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                }
                results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}
        
        stage1_results = {
            'method': 'random_search',
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'n_evaluations': len(results)
        }
        
        self.stage1_results = stage1_results
        return stage1_results
    
    def _stage2_tpe_refinement(
        self, 
        features: np.ndarray, 
        stage1_results: Dict[str, Any], 
        n_evaluations: int,
        problem_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Stage 2: TPE refinement with adaptive search space.
        
        Args:
            features: Input features
            stage1_results: Results from stage 1
            n_evaluations: Number of evaluations for stage 2
            problem_analysis: Problem analysis results
            
        Returns:
            Stage 2 results
        """
        self.logger.info("🔍 Stage 2: TPE Refinement")
        
        try:
            import optuna
        except ImportError:
            self.logger.warning("⚠️ Optuna not available, falling back to coordinate descent")
            return self._coordinate_descent_refinement(features, stage1_results, n_evaluations)
        
        # Define adaptive search space based on stage 1 results
        search_space = self._define_adaptive_search_space(stage1_results, problem_analysis)
        
        def objective(trial):
            eps = trial.suggest_float('eps', search_space['eps_min'], search_space['eps_max'])
            min_samples = trial.suggest_int('min_samples', search_space['min_samples_min'], search_space['min_samples_max'])
            
            clustering = self._run_dbscan(features, eps, min_samples)
            if clustering is None:
                return -float('inf')  # Penalize invalid clustering
            
            score = self._calculate_composite_score(features, clustering.labels_)
            return score
        
        # Create study with TPE sampler
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.config["random_seed"])
        )
        
        # Optimize with TPE
        study.optimize(objective, n_trials=n_evaluations)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        stage2_results = {
            'method': 'tpe',
            'best_params': best_params,
            'best_score': best_score,
            'study': study,
            'n_evaluations': n_evaluations,
            'search_space': search_space
        }
        
        self.stage2_results = stage2_results
        return stage2_results
    
    def _define_adaptive_search_space(
        self, 
        stage1_results: Dict[str, Any], 
        problem_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Define adaptive search space for TPE based on stage 1 results.
        
        Args:
            stage1_results: Results from stage 1
            problem_analysis: Problem analysis results
            
        Returns:
            Search space bounds
        """
        expansion_factor = self.config["search_space_expansion"]
        
        if stage1_results['method'] == 'adaptive_grid' and 'promising_regions' in stage1_results:
            # Use multiple regions for search space
            regions = stage1_results['promising_regions']
            
            # Combine bounds from all regions
            eps_min = min(region['bounds']['eps_min'] for region in regions)
            eps_max = max(region['bounds']['eps_max'] for region in regions)
            min_samples_min = min(region['bounds']['min_samples_min'] for region in regions)
            min_samples_max = max(region['bounds']['min_samples_max'] for region in regions)
            
            # Expand search space
            eps_center = (eps_min + eps_max) / 2
            eps_range = (eps_max - eps_min) / 2
            eps_min = max(0.1, eps_center - eps_range * expansion_factor)
            eps_max = min(2.0, eps_center + eps_range * expansion_factor)
            
            min_samples_center = (min_samples_min + min_samples_max) / 2
            min_samples_range = (min_samples_max - min_samples_min) / 2
            min_samples_min = max(5, int(min_samples_center - min_samples_range * expansion_factor))
            min_samples_max = min(50, int(min_samples_center + min_samples_range * expansion_factor))
            
        else:
            # Use single best point from stage 1
            best_params = stage1_results['best_params']
            eps_best = best_params['eps']
            min_samples_best = best_params['min_samples']
            
            # Define search space around best point
            eps_min = max(0.1, eps_best / expansion_factor)
            eps_max = min(2.0, eps_best * expansion_factor)
            min_samples_min = max(5, int(min_samples_best / expansion_factor))
            min_samples_max = min(50, int(min_samples_best * expansion_factor))
        
        # Adjust based on problem characteristics
        if problem_analysis['complexity'] == 'small':
            # Smaller search space for small problems
            eps_min = max(eps_min, eps_best * 0.7)
            eps_max = min(eps_max, eps_best * 1.3)
        elif problem_analysis['complexity'] == 'large':
            # Larger search space for large problems
            eps_min = max(0.1, eps_best * 0.3)
            eps_max = min(2.0, eps_best * 2.0)
        
        search_space = {
            'eps_min': eps_min,
            'eps_max': eps_max,
            'min_samples_min': min_samples_min,
            'min_samples_max': min_samples_max
        }
        
        self.logger.info(f"🔍 TPE Search Space: eps=[{eps_min:.3f}, {eps_max:.3f}], min_samples=[{min_samples_min}, {min_samples_max}]")
        
        return search_space
    
    def _coordinate_descent_refinement(
        self, 
        features: np.ndarray, 
        stage1_results: Dict[str, Any], 
        n_evaluations: int
    ) -> Dict[str, Any]:
        """
        Fallback coordinate descent refinement when TPE is not available.
        
        Args:
            features: Input features
            stage1_results: Results from stage 1
            n_evaluations: Number of evaluations
            
        Returns:
            Stage 2 results
        """
        self.logger.info("🔍 Stage 2: Coordinate Descent Refinement (TPE fallback)")
        
        best_params = stage1_results['best_params'].copy()
        best_score = stage1_results['best_score']
        
        for iteration in range(n_evaluations // 2):
            # Optimize eps while fixing min_samples
            eps_candidates = [best_params['eps'] * (1 + 0.1 * i) for i in [-2, -1, 0, 1, 2]]
            best_eps = best_params['eps']
            
            for eps in eps_candidates:
                if 0.1 <= eps <= 2.0:
                    clustering = self._run_dbscan(features, eps, best_params['min_samples'])
                    if clustering is not None:
                        score = self._calculate_composite_score(features, clustering.labels_)
                        if score > best_score:
                            best_score = score
                            best_eps = eps
            
            best_params['eps'] = best_eps
            
            # Optimize min_samples while fixing eps
            min_samples_candidates = [best_params['min_samples'] + i for i in [-5, -2, 0, 2, 5]]
            best_min_samples = best_params['min_samples']
            
            for min_samples in min_samples_candidates:
                if 5 <= min_samples <= 50:
                    clustering = self._run_dbscan(features, best_params['eps'], min_samples)
                    if clustering is not None:
                        score = self._calculate_composite_score(features, clustering.labels_)
                        if score > best_score:
                            best_score = score
                            best_min_samples = min_samples
            
            best_params['min_samples'] = best_min_samples
        
        stage2_results = {
            'method': 'coordinate_descent',
            'best_params': best_params,
            'best_score': best_score,
            'n_evaluations': n_evaluations
        }
        
        self.stage2_results = stage2_results
        return stage2_results
    
    def _combine_optimization_results(
        self, 
        stage1_results: Dict[str, Any], 
        stage2_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine results from both stages.
        
        Args:
            stage1_results: Results from stage 1
            stage2_results: Results from stage 2
            
        Returns:
            Combined optimization results
        """
        # Determine final best result
        if stage2_results['best_score'] > stage1_results['best_score']:
            final_params = stage2_results['best_params']
            final_score = stage2_results['best_score']
            improvement = final_score - stage1_results['best_score']
        else:
            final_params = stage1_results['best_params']
            final_score = stage1_results['best_score']
            improvement = 0.0
        
        combined_results = {
            'success': True,
            'best_params': final_params,
            'best_score': final_score,
            'improvement': improvement,
            'stage1_results': stage1_results,
            'stage2_results': stage2_results,
            'total_evaluations': stage1_results['n_evaluations'] + stage2_results['n_evaluations'],
            'optimization_history': self.optimization_history
        }
        
        return combined_results
    
    def _run_dbscan(self, features: np.ndarray, eps: float, min_samples: int):
        """
        Run DBSCAN clustering with given parameters.
        
        Args:
            features: Input features
            eps: DBSCAN eps parameter
            min_samples: DBSCAN min_samples parameter
            
        Returns:
            DBSCAN clustering result or None if failed
        """
        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=min_samples, random_state=self.config["random_seed"])
            clustering.fit(features)
            return clustering
        except Exception as e:
            self.logger.warning(f"⚠️ DBSCAN failed with eps={eps}, min_samples={min_samples}: {e}")
            return None
    
    def _calculate_composite_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate composite quality score for clustering.
        
        Args:
            features: Input features
            labels: Cluster labels
            
        Returns:
            Composite quality score
        """
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            # Filter out noise points
            valid_mask = labels != -1
            if np.sum(valid_mask) < 10:
                return -float('inf')
            
            features_valid = features[valid_mask]
            labels_valid = labels[valid_mask]
            
            # Calculate metrics
            silhouette = silhouette_score(features_valid, labels_valid)
            calinski_harabasz = calinski_harabasz_score(features_valid, labels_valid)
            davies_bouldin = davies_bouldin_score(features_valid, labels_valid)
            
            # Composite score (higher is better)
            composite_score = (
                0.4 * silhouette +
                0.2 * (calinski_harabasz / 1000) -  # Normalize
                0.2 * (davies_bouldin / 10) -       # Normalize
                0.1 * self._calculate_skew_penalty(features_valid, labels_valid) -
                0.1 * self._calculate_volatility_penalty(features_valid, labels_valid)
            )
            
            return composite_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate composite score: {e}")
            return -float('inf')
    
    def _calculate_skew_penalty(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate skew penalty for cluster size distribution."""
        cluster_sizes = [np.sum(labels == label) for label in set(labels)]
        if len(cluster_sizes) < 2:
            return 1.0
        
        mean_size = np.mean(cluster_sizes)
        std_size = np.std(cluster_sizes)
        
        if mean_size == 0:
            return 1.0
        
        coefficient_of_variation = std_size / mean_size
        return min(1.0, coefficient_of_variation)
    
    def _calculate_volatility_penalty(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate volatility penalty for cluster stability."""
        # Simple implementation: penalize clusters with high variance
        cluster_variances = []
        for label in set(labels):
            cluster_points = features[labels == label]
            if len(cluster_points) > 1:
                cluster_var = np.var(cluster_points, axis=0).mean()
                cluster_variances.append(cluster_var)
        
        if not cluster_variances:
            return 1.0
        
        return min(1.0, np.mean(cluster_variances))

# Convenience functions for backward compatibility
def optimize_dbscan_parameters(
    features: np.ndarray, 
    config: Optional[Dict[str, Any]] = None,
    max_evaluations: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function for DBSCAN parameter optimization.
    
    Args:
        features: Input features
        config: Optimization configuration
        max_evaluations: Maximum evaluations
        
    Returns:
        Optimization results
    """
    if config is None:
        config = {}
    
    optimizer = EnhancedTwoStageOptimizer(config)
    return optimizer.optimize_dbscan_parameters(features, max_evaluations)

def smart_two_stage_optimization(
    features: np.ndarray, 
    max_evaluations: int = 50
) -> Dict[str, Any]:
    """
    Smart two-stage optimization with automatic configuration.
    
    Args:
        features: Input features
        max_evaluations: Maximum evaluations
        
    Returns:
        Optimization results
    """
    # Analyze problem and choose optimal configuration
    n_samples = len(features)
    
    if n_samples < 2000:
        config = {
            "max_evaluations": max_evaluations,
            "stage1_ratio": 0.7,
            "robustness_level": "low",
            "search_space_expansion": 1.2
        }
    elif n_samples < 10000:
        config = {
            "max_evaluations": max_evaluations,
            "stage1_ratio": 0.6,
            "robustness_level": "medium",
            "search_space_expansion": 1.5
        }
    else:
        config = {
            "max_evaluations": max_evaluations,
            "stage1_ratio": 0.4,
            "robustness_level": "high",
            "search_space_expansion": 2.0
        }
    
    optimizer = EnhancedTwoStageOptimizer(config)
    return optimizer.optimize_dbscan_parameters(features)