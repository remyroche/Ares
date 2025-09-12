#!/usr/bin/env python3
"""
Ensemble Enhancement: Dynamic Weight Optimization for HMM Clustering

This module implements dynamic weight optimization for ensemble clustering:
- Performance-Based Weight Optimization
- Multi-Objective Optimization using Pareto front
- Ensemble weight adaptation and optimization

Author: AI Assistant
Date: 2024-01-XX
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import time
import logging
from pathlib import Path
import json

# Scipy imports for optimization
try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Sklearn imports
try:
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.cluster import KMeans, DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Pareto optimization imports
try:
    from src.utils.ml_common.pareto import ParetoFront, ParetoOptimizer
    PARETO_AVAILABLE = True
except ImportError:
    PARETO_AVAILABLE = False

from src.utils.logger import system_logger

@dataclass
class EnsembleResult:
    """Result of ensemble clustering"""
    predictions: np.ndarray
    weights: Dict[str, float]
    individual_scores: Dict[str, float]
    ensemble_score: float
    method: str

@dataclass
class OptimizationResult:
    """Result of weight optimization"""
    optimal_weights: Dict[str, float]
    optimization_score: float
    optimization_time: float
    method: str
    convergence_info: Dict[str, Any]

class EnsembleWeightOptimizer:
    """Dynamic weight optimization for ensemble clustering"""
    
    def __init__(self, logger=None):
        self.logger = logger or system_logger.getChild('EnsembleWeightOptimizer')
        self.optimization_history = []
    
    def performance_based_optimization(self, hmm_results: Dict[str, Any], 
                                     kmeans_results: Dict[str, Any], 
                                     dbscan_results: Dict[str, Any],
                                     validation_data: np.ndarray) -> OptimizationResult:
        """
        Optimize ensemble weights based on validation performance
        
        Args:
            hmm_results: HMM clustering results
            kmeans_results: K-means clustering results  
            dbscan_results: DBSCAN clustering results
            validation_data: Validation data for scoring
            
        Returns:
            OptimizationResult with optimal weights
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy not available")
        
        start_time = time.time()
        self.logger.info("🔍 Optimizing ensemble weights based on performance...")
        
        # Extract predictions
        hmm_pred = hmm_results['predictions']
        kmeans_pred = kmeans_results['predictions']
        dbscan_pred = dbscan_results['predictions']
        
        def ensemble_score(weights):
            """Calculate ensemble score for given weights"""
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Weighted combination of predictions
            ensemble_pred = (weights[0] * hmm_pred + 
                           weights[1] * kmeans_pred + 
                           weights[2] * dbscan_pred)
            
            # Calculate silhouette score
            try:
                score = silhouette_score(validation_data, ensemble_pred)
                return -score  # Minimize negative score
            except Exception:
                return -1.0  # Return bad score if error
        
        # Initial weights
        initial_weights = np.array([0.4, 0.3, 0.3])
        
        # Constraints: weights sum to 1, all positive
        constraints = {'type': 'eq', 'fun': lambda x: x.sum() - 1}
        bounds = [(0, 1) for _ in range(3)]
        
        # Optimize using SLSQP
        result = minimize(ensemble_score, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Normalize final weights
        optimal_weights = result.x / result.x.sum()
        
        optimization_time = time.time() - start_time
        
        # Calculate final ensemble score
        final_ensemble_pred = (optimal_weights[0] * hmm_pred + 
                              optimal_weights[1] * kmeans_pred + 
                              optimal_weights[2] * dbscan_pred)
        final_score = silhouette_score(validation_data, final_ensemble_pred)
        
        optimization_result = OptimizationResult(
            optimal_weights={
                'hmm': optimal_weights[0],
                'kmeans': optimal_weights[1],
                'dbscan': optimal_weights[2]
            },
            optimization_score=final_score,
            optimization_time=optimization_time,
            method='performance_based_optimization',
            convergence_info={
                'success': result.success,
                'message': result.message,
                'iterations': result.nit,
                'function_evaluations': result.nfev
            }
        )
        
        self.optimization_history.append(optimization_result)
        self.logger.info(f"✅ Performance-based optimization completed: {optimization_result.optimal_weights}")
        
        return optimization_result
    
    def multi_objective_optimization(self, hmm_results: Dict[str, Any], 
                                   kmeans_results: Dict[str, Any], 
                                   dbscan_results: Dict[str, Any],
                                   validation_data: np.ndarray,
                                   regime_labels: Optional[np.ndarray] = None) -> OptimizationResult:
        """
        Multi-objective optimization considering multiple metrics
        
        Args:
            hmm_results: HMM clustering results
            kmeans_results: K-means clustering results
            dbscan_results: DBSCAN clustering results
            validation_data: Validation data for scoring
            regime_labels: True regime labels if available
            
        Returns:
            OptimizationResult with optimal weights
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy not available")
        
        start_time = time.time()
        self.logger.info("🔍 Multi-objective optimization...")
        
        # Extract predictions
        hmm_pred = hmm_results['predictions']
        kmeans_pred = kmeans_results['predictions']
        dbscan_pred = dbscan_results['predictions']
        
        def multi_objective_score(weights):
            """Multi-objective function combining multiple metrics"""
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Ensemble prediction
            ensemble_pred = (weights[0] * hmm_pred + 
                           weights[1] * kmeans_pred + 
                           weights[2] * dbscan_pred)
            
            try:
                # Multiple objectives
                sil_score = silhouette_score(validation_data, ensemble_pred)
                ch_score = calinski_harabasz_score(validation_data, ensemble_pred)
                db_score = davies_bouldin_score(validation_data, ensemble_pred)
                
                # Combined score (higher is better)
                # Normalize scores to [0, 1] range
                sil_norm = max(0, min(1, sil_score))  # Silhouette: [-1, 1] -> [0, 1]
                ch_norm = max(0, min(1, ch_score / 1000))  # CH: [0, inf] -> [0, 1]
                db_norm = max(0, min(1, 1 - db_score))  # DB: [0, inf] -> [0, 1] (inverted)
                
                combined_score = 0.5 * sil_norm + 0.3 * ch_norm + 0.2 * db_norm
                
                return -combined_score  # Minimize negative score
                
            except Exception as e:
                self.logger.warning(f"Error in multi-objective scoring: {e}")
                return -1.0
        
        # Initial weights
        initial_weights = np.array([0.4, 0.3, 0.3])
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda x: x.sum() - 1}
        bounds = [(0, 1) for _ in range(3)]
        
        # Try multiple optimization methods
        best_result = None
        best_score = -np.inf
        
        methods = ['SLSQP', 'L-BFGS-B', 'TNC']
        
        for method in methods:
            try:
                result = minimize(multi_objective_score, initial_weights, 
                               method=method, bounds=bounds, constraints=constraints)
                
                if result.success and -result.fun > best_score:
                    best_score = -result.fun
                    best_result = result
                    
            except Exception as e:
                self.logger.warning(f"Optimization method {method} failed: {e}")
                continue
        
        if best_result is None:
            # Fallback to differential evolution
            try:
                result = differential_evolution(multi_objective_score, bounds, 
                                             seed=42, maxiter=100)
                best_result = result
            except Exception as e:
                self.logger.error(f"All optimization methods failed: {e}")
                # Use equal weights as fallback
                optimal_weights = np.array([0.33, 0.33, 0.34])
                optimization_time = time.time() - start_time
                
                return OptimizationResult(
                    optimal_weights={'hmm': 0.33, 'kmeans': 0.33, 'dbscan': 0.34},
                    optimization_score=0.0,
                    optimization_time=optimization_time,
                    method='multi_objective_optimization_fallback',
                    convergence_info={'success': False, 'message': 'All methods failed'}
                )
        
        # Normalize final weights
        optimal_weights = best_result.x / best_result.x.sum()
        
        optimization_time = time.time() - start_time
        
        # Calculate final ensemble score
        final_ensemble_pred = (optimal_weights[0] * hmm_pred + 
                              optimal_weights[1] * kmeans_pred + 
                              optimal_weights[2] * dbscan_pred)
        final_score = silhouette_score(validation_data, final_ensemble_pred)
        
        optimization_result = OptimizationResult(
            optimal_weights={
                'hmm': optimal_weights[0],
                'kmeans': optimal_weights[1],
                'dbscan': optimal_weights[2]
            },
            optimization_score=final_score,
            optimization_time=optimization_time,
            method='multi_objective_optimization',
            convergence_info={
                'success': best_result.success,
                'message': best_result.message,
                'iterations': getattr(best_result, 'nit', 0),
                'function_evaluations': getattr(best_result, 'nfev', 0)
            }
        )
        
        self.optimization_history.append(optimization_result)
        self.logger.info(f"✅ Multi-objective optimization completed: {optimization_result.optimal_weights}")
        
        return optimization_result
    
    def pareto_optimization(self, hmm_results: Dict[str, Any], 
                          kmeans_results: Dict[str, Any], 
                          dbscan_results: Dict[str, Any],
                          validation_data: np.ndarray) -> OptimizationResult:
        """
        Pareto optimization for multi-objective weight optimization
        
        Args:
            hmm_results: HMM clustering results
            kmeans_results: K-means clustering results
            dbscan_results: DBSCAN clustering results
            validation_data: Validation data for scoring
            
        Returns:
            OptimizationResult with optimal weights
        """
        if not PARETO_AVAILABLE:
            self.logger.warning("Pareto optimization not available, falling back to multi-objective")
            return self.multi_objective_optimization(hmm_results, kmeans_results, dbscan_results, validation_data)
        
        start_time = time.time()
        self.logger.info("🔍 Pareto optimization...")
        
        # Extract predictions
        hmm_pred = hmm_results['predictions']
        kmeans_pred = kmeans_results['predictions']
        dbscan_pred = dbscan_results['predictions']
        
        def evaluate_weights(weights):
            """Evaluate weights for Pareto optimization"""
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Ensemble prediction
            ensemble_pred = (weights[0] * hmm_pred + 
                           weights[1] * kmeans_pred + 
                           weights[2] * dbscan_pred)
            
            try:
                # Multiple objectives
                sil_score = silhouette_score(validation_data, ensemble_pred)
                ch_score = calinski_harabasz_score(validation_data, ensemble_pred)
                db_score = davies_bouldin_score(validation_data, ensemble_pred)
                
                return [sil_score, ch_score, -db_score]  # Negative DB score (lower is better)
                
            except Exception:
                return [0.0, 0.0, 0.0]
        
        # Generate weight combinations
        weight_combinations = []
        for w1 in np.linspace(0.1, 0.8, 10):
            for w2 in np.linspace(0.1, 0.8, 10):
                w3 = 1.0 - w1 - w2
                if w3 > 0:
                    weight_combinations.append([w1, w2, w3])
        
        # Evaluate all combinations
        objectives = []
        for weights in weight_combinations:
            obj_values = evaluate_weights(weights)
            objectives.append(obj_values)
        
        objectives = np.array(objectives)
        
        # Find Pareto front
        pareto_front = ParetoFront()
        pareto_indices = pareto_front.find_pareto_front(objectives)
        
        # Select best solution from Pareto front (using knee point if available)
        if len(pareto_indices) > 0:
            if hasattr(pareto_front, 'find_knee_point'):
                best_idx = pareto_front.find_knee_point(objectives[pareto_indices])
                best_weights = weight_combinations[pareto_indices[best_idx]]
            else:
                # Select solution with highest silhouette score
                best_idx = np.argmax(objectives[pareto_indices, 0])
                best_weights = weight_combinations[pareto_indices[best_idx]]
        else:
            # Fallback to equal weights
            best_weights = [0.33, 0.33, 0.34]
        
        optimization_time = time.time() - start_time
        
        # Calculate final ensemble score
        final_ensemble_pred = (best_weights[0] * hmm_pred + 
                              best_weights[1] * kmeans_pred + 
                              best_weights[2] * dbscan_pred)
        final_score = silhouette_score(validation_data, final_ensemble_pred)
        
        optimization_result = OptimizationResult(
            optimal_weights={
                'hmm': best_weights[0],
                'kmeans': best_weights[1],
                'dbscan': best_weights[2]
            },
            optimization_score=final_score,
            optimization_time=optimization_time,
            method='pareto_optimization',
            convergence_info={
                'success': True,
                'message': 'Pareto optimization completed',
                'pareto_solutions': len(pareto_indices)
            }
        )
        
        self.optimization_history.append(optimization_result)
        self.logger.info(f"✅ Pareto optimization completed: {optimization_result.optimal_weights}")
        
        return optimization_result
    
    def adaptive_weight_updates(self, hmm_results: Dict[str, Any], 
                              kmeans_results: Dict[str, Any], 
                              dbscan_results: Dict[str, Any],
                              validation_data: np.ndarray,
                              learning_rate: float = 0.01,
                              n_iterations: int = 10) -> OptimizationResult:
        """
        Adaptive weight updates based on performance feedback
        
        Args:
            hmm_results: HMM clustering results
            kmeans_results: K-means clustering results
            dbscan_results: DBSCAN clustering results
            validation_data: Validation data for scoring
            learning_rate: Learning rate for weight updates
            n_iterations: Number of adaptation iterations
            
        Returns:
            OptimizationResult with optimal weights
        """
        start_time = time.time()
        self.logger.info(f"🔍 Adaptive weight updates (lr={learning_rate}, iterations={n_iterations})...")
        
        # Extract predictions
        hmm_pred = hmm_results['predictions']
        kmeans_pred = kmeans_results['predictions']
        dbscan_pred = dbscan_results['predictions']
        
        # Initialize weights
        weights = np.array([0.4, 0.3, 0.3])
        best_weights = weights.copy()
        best_score = -np.inf
        
        for iteration in range(n_iterations):
            # Calculate current ensemble prediction
            ensemble_pred = (weights[0] * hmm_pred + 
                           weights[1] * kmeans_pred + 
                           weights[2] * dbscan_pred)
            
            # Calculate current score
            try:
                current_score = silhouette_score(validation_data, ensemble_pred)
            except Exception:
                current_score = 0.0
            
            # Update best weights if current is better
            if current_score > best_score:
                best_score = current_score
                best_weights = weights.copy()
            
            # Calculate individual algorithm scores
            individual_scores = []
            for pred in [hmm_pred, kmeans_pred, dbscan_pred]:
                try:
                    score = silhouette_score(validation_data, pred)
                    individual_scores.append(score)
                except Exception:
                    individual_scores.append(0.0)
            
            # Normalize scores
            individual_scores = np.array(individual_scores)
            if individual_scores.sum() > 0:
                individual_scores = individual_scores / individual_scores.sum()
            else:
                individual_scores = np.array([0.33, 0.33, 0.34])
            
            # Update weights based on performance
            weight_updates = learning_rate * (individual_scores - weights)
            weights = weights + weight_updates
            
            # Ensure weights are positive and sum to 1
            weights = np.maximum(weights, 0.01)  # Minimum weight
            weights = weights / weights.sum()
            
            self.logger.debug(f"Iteration {iteration+1}: weights={weights}, score={current_score:.4f}")
        
        optimization_time = time.time() - start_time
        
        # Calculate final ensemble score
        final_ensemble_pred = (best_weights[0] * hmm_pred + 
                              best_weights[1] * kmeans_pred + 
                              best_weights[2] * dbscan_pred)
        final_score = silhouette_score(validation_data, final_ensemble_pred)
        
        optimization_result = OptimizationResult(
            optimal_weights={
                'hmm': best_weights[0],
                'kmeans': best_weights[1],
                'dbscan': best_weights[2]
            },
            optimization_score=final_score,
            optimization_time=optimization_time,
            method='adaptive_weight_updates',
            convergence_info={
                'success': True,
                'message': 'Adaptive updates completed',
                'iterations': n_iterations,
                'final_score': final_score
            }
        )
        
        self.optimization_history.append(optimization_result)
        self.logger.info(f"✅ Adaptive weight updates completed: {optimization_result.optimal_weights}")
        
        return optimization_result
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization runs"""
        if not self.optimization_history:
            return {"message": "No optimization runs recorded"}
        
        summary = {
            "total_runs": len(self.optimization_history),
            "methods_used": list(set(r.method for r in self.optimization_history)),
            "best_overall_score": max(r.optimization_score for r in self.optimization_history),
            "total_optimization_time": sum(r.optimization_time for r in self.optimization_history),
            "runs": []
        }
        
        for i, result in enumerate(self.optimization_history):
            summary["runs"].append({
                "run_id": i,
                "method": result.method,
                "optimal_weights": result.optimal_weights,
                "optimization_score": result.optimization_score,
                "optimization_time": result.optimization_time,
                "convergence_info": result.convergence_info
            })
        
        return summary
    
    def save_optimization_results(self, filepath: str) -> None:
        """Save optimization results to file"""
        results = {
            "optimization_history": [
                {
                    "optimal_weights": r.optimal_weights,
                    "optimization_score": r.optimization_score,
                    "optimization_time": r.optimization_time,
                    "method": r.method,
                    "convergence_info": r.convergence_info
                }
                for r in self.optimization_history
            ],
            "summary": self.get_optimization_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"💾 Optimization results saved to {filepath}")

# Example usage and testing
def test_ensemble_optimization():
    """Test the ensemble optimization functionality"""
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    validation_data = np.random.randn(n_samples, n_features)
    
    # Generate mock clustering results
    hmm_results = {
        'predictions': np.random.randint(0, 4, n_samples),
        'score': 0.5
    }
    kmeans_results = {
        'predictions': np.random.randint(0, 4, n_samples),
        'score': 0.4
    }
    dbscan_results = {
        'predictions': np.random.randint(0, 4, n_samples),
        'score': 0.3
    }
    
    optimizer = EnsembleWeightOptimizer()
    
    # Test performance-based optimization
    print("Testing performance-based optimization...")
    result1 = optimizer.performance_based_optimization(hmm_results, kmeans_results, dbscan_results, validation_data)
    print(f"Optimal weights: {result1.optimal_weights}")
    
    # Test multi-objective optimization
    print("\nTesting multi-objective optimization...")
    result2 = optimizer.multi_objective_optimization(hmm_results, kmeans_results, dbscan_results, validation_data)
    print(f"Optimal weights: {result2.optimal_weights}")
    
    # Test adaptive weight updates
    print("\nTesting adaptive weight updates...")
    result3 = optimizer.adaptive_weight_updates(hmm_results, kmeans_results, dbscan_results, validation_data)
    print(f"Optimal weights: {result3.optimal_weights}")
    
    # Print summary
    print("\nOptimization Summary:")
    summary = optimizer.get_optimization_summary()
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    test_ensemble_optimization()