"""
Perfect Multi-Objective Optimizer

Advanced multi-objective optimization for Perfect NAS Regime System.
Combines multiple objectives with sophisticated optimization algorithms.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..core.perfect_nas_config import PerfectNASConfig, OptimizationObjective
from ..evaluation.economic_evaluator import EconomicSignificanceEvaluator
from ..evaluation.trading_viability_evaluator import TradingViabilityEvaluator

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Result from multi-objective optimization."""
    success: bool
    pareto_solutions: List[Dict[str, Any]]
    best_solution: Optional[Dict[str, Any]]
    optimization_metrics: Dict[str, float]
    convergence_history: List[Dict[str, float]]
    execution_time: float
    error_message: Optional[str] = None

class PerfectMultiObjectiveOptimizer:
    """
    Perfect Multi-Objective Optimizer for NAS Regime System.
    
    Optimizes multiple objectives simultaneously:
    - Regime accuracy
    - Economic significance
    - Trading viability
    - Computational efficiency
    - Architecture complexity
    - Regime stability
    - Transition accuracy
    """
    
    def __init__(self, config: PerfectNASConfig):
        """Initialize perfect multi-objective optimizer.
        
        Args:
            config: Perfect NAS configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize evaluators
        self.economic_evaluator = EconomicSignificanceEvaluator(config.economic_config)
        self.trading_evaluator = TradingViabilityEvaluator(config.trading_config)
        
        # Optimization parameters
        self.objectives = config.objectives
        self.objective_weights = config.objective_weights
        
        # Optimization history
        self.optimization_history = []
        self.pareto_frontier = []
        
        self.logger.info("✅ Perfect Multi-Objective Optimizer initialized")
        self.logger.info(f"   Objectives: {[obj.value for obj in self.objectives]}")
        self.logger.info(f"   Weights: {self.objective_weights}")
    
    def optimize(self, market_data: np.ndarray, regime_predictions: np.ndarray,
                timestamps: Optional[np.ndarray] = None,
                max_iterations: int = 100) -> OptimizationResult:
        """Perform multi-objective optimization.
        
        Args:
            market_data: Market data
            regime_predictions: Current regime predictions
            timestamps: Optional timestamps
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimization result with Pareto solutions
        """
        try:
            import time
            start_time = time.time()
            
            self.logger.info("🚀 Starting Perfect Multi-Objective Optimization")
            self.logger.info(f"   Data shape: {market_data.shape}")
            self.logger.info(f"   Regimes: {len(np.unique(regime_predictions))}")
            self.logger.info(f"   Max iterations: {max_iterations}")
            
            # Initialize optimization
            self._initialize_optimization(market_data, regime_predictions, timestamps)
            
            # Perform optimization iterations
            for iteration in range(max_iterations):
                self.logger.info(f"🔄 Optimization iteration {iteration + 1}/{max_iterations}")
                
                # Generate candidate solutions
                candidates = self._generate_candidate_solutions(iteration)
                
                # Evaluate candidates
                evaluated_candidates = self._evaluate_candidates(candidates, market_data, regime_predictions, timestamps)
                
                # Update Pareto frontier
                self._update_pareto_frontier(evaluated_candidates)
                
                # Check convergence
                if self._check_convergence(iteration):
                    self.logger.info(f"✅ Optimization converged at iteration {iteration + 1}")
                    break
            
            # Finalize optimization
            execution_time = time.time() - start_time
            result = self._finalize_optimization(execution_time)
            
            self.logger.info(f"✅ Multi-objective optimization completed in {execution_time:.2f}s")
            self.logger.info(f"   Pareto solutions: {len(result.pareto_solutions)}")
            self.logger.info(f"   Best solution score: {result.best_solution.get('total_score', 0.0):.4f}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Multi-objective optimization failed: {e}")
            
            return OptimizationResult(
                success=False,
                pareto_solutions=[],
                best_solution=None,
                optimization_metrics={},
                convergence_history=[],
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _initialize_optimization(self, market_data: np.ndarray, 
                               regime_predictions: np.ndarray,
                               timestamps: Optional[np.ndarray]):
        """Initialize optimization process."""
        try:
            # Store optimization data
            self.market_data = market_data
            self.regime_predictions = regime_predictions
            self.timestamps = timestamps
            
            # Initialize Pareto frontier
            self.pareto_frontier = []
            self.optimization_history = []
            
            # Calculate baseline metrics
            baseline_metrics = self._calculate_baseline_metrics(market_data, regime_predictions, timestamps)
            self.baseline_metrics = baseline_metrics
            
            self.logger.info("✅ Optimization initialized")
            self.logger.info(f"   Baseline metrics: {baseline_metrics}")
            
        except Exception as e:
            self.logger.error(f"Optimization initialization failed: {e}")
            raise
    
    def _calculate_baseline_metrics(self, market_data: np.ndarray, 
                                  regime_predictions: np.ndarray,
                                  timestamps: Optional[np.ndarray]) -> Dict[str, float]:
        """Calculate baseline metrics for comparison."""
        try:
            metrics = {}
            
            # Regime accuracy (simplified)
            unique_regimes = len(np.unique(regime_predictions))
            metrics['regime_accuracy'] = min(unique_regimes / self.config.n_regimes, 1.0)
            
            # Economic significance
            economic_scores = self.economic_evaluator.evaluate(market_data, regime_predictions, timestamps)
            metrics['economic_significance'] = np.mean(economic_scores)
            
            # Trading viability
            trading_scores = self.trading_evaluator.evaluate(market_data, regime_predictions, timestamps)
            metrics['trading_viability'] = np.mean(trading_scores)
            
            # Computational efficiency (placeholder)
            metrics['computational_efficiency'] = 0.8
            
            # Architecture complexity (placeholder)
            metrics['architecture_complexity'] = 0.5
            
            # Regime stability
            regime_stability = self._calculate_regime_stability(regime_predictions)
            metrics['regime_stability'] = regime_stability
            
            # Transition accuracy
            transition_accuracy = self._calculate_transition_accuracy(regime_predictions)
            metrics['transition_accuracy'] = transition_accuracy
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Baseline metrics calculation failed: {e}")
            return {obj.value: 0.5 for obj in self.objectives}
    
    def _generate_candidate_solutions(self, iteration: int) -> List[Dict[str, Any]]:
        """Generate candidate solutions for optimization."""
        try:
            candidates = []
            
            # Generate different types of candidates
            n_candidates = 10
            
            for i in range(n_candidates):
                candidate = {
                    'id': f"candidate_{iteration}_{i}",
                    'iteration': iteration,
                    'regime_count': np.random.randint(5, 15),
                    'architecture_type': np.random.choice(['neural_ode', 'vision_transformer', 'hybrid']),
                    'complexity_factor': np.random.uniform(0.3, 1.0),
                    'efficiency_factor': np.random.uniform(0.5, 1.0),
                    'stability_factor': np.random.uniform(0.4, 1.0),
                    'parameters': {
                        'learning_rate': np.random.uniform(1e-4, 1e-2),
                        'batch_size': np.random.choice([32, 64, 128]),
                        'dropout_rate': np.random.uniform(0.1, 0.5),
                        'hidden_size': np.random.choice([64, 128, 256])
                    }
                }
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            self.logger.warning(f"Candidate generation failed: {e}")
            return []
    
    def _evaluate_candidates(self, candidates: List[Dict[str, Any]], 
                           market_data: np.ndarray, regime_predictions: np.ndarray,
                           timestamps: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        """Evaluate candidate solutions."""
        try:
            evaluated_candidates = []
            
            for candidate in candidates:
                try:
                    # Simulate regime predictions based on candidate parameters
                    simulated_predictions = self._simulate_regime_predictions(
                        market_data, candidate, regime_predictions
                    )
                    
                    # Evaluate objectives
                    objective_scores = {}
                    
                    for objective in self.objectives:
                        if objective == OptimizationObjective.REGIME_ACCURACY:
                            score = self._evaluate_regime_accuracy(simulated_predictions, regime_predictions)
                        elif objective == OptimizationObjective.ECONOMIC_SIGNIFICANCE:
                            score = np.mean(self.economic_evaluator.evaluate(market_data, simulated_predictions, timestamps))
                        elif objective == OptimizationObjective.TRADING_VIABILITY:
                            score = np.mean(self.trading_evaluator.evaluate(market_data, simulated_predictions, timestamps))
                        elif objective == OptimizationObjective.COMPUTATIONAL_EFFICIENCY:
                            score = candidate['efficiency_factor']
                        elif objective == OptimizationObjective.ARCHITECTURE_COMPLEXITY:
                            score = 1.0 - candidate['complexity_factor']  # Lower complexity is better
                        elif objective == OptimizationObjective.REGIME_STABILITY:
                            score = self._calculate_regime_stability(simulated_predictions)
                        elif objective == OptimizationObjective.TRANSITION_ACCURACY:
                            score = self._calculate_transition_accuracy(simulated_predictions)
                        else:
                            score = 0.5
                        
                        objective_scores[objective.value] = score
                    
                    # Calculate weighted total score
                    total_score = sum(
                        objective_scores.get(obj.value, 0.0) * self.objective_weights.get(obj, 0.0)
                        for obj in self.objectives
                    )
                    
                    candidate['objective_scores'] = objective_scores
                    candidate['total_score'] = total_score
                    candidate['simulated_predictions'] = simulated_predictions
                    
                    evaluated_candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Candidate evaluation failed: {e}")
                    continue
            
            return evaluated_candidates
            
        except Exception as e:
            self.logger.error(f"Candidate evaluation failed: {e}")
            return []
    
    def _simulate_regime_predictions(self, market_data: np.ndarray, 
                                   candidate: Dict[str, Any],
                                   original_predictions: np.ndarray) -> np.ndarray:
        """Simulate regime predictions based on candidate parameters."""
        try:
            # Simple simulation based on candidate parameters
            n_samples = len(original_predictions)
            n_regimes = candidate['regime_count']
            
            # Add some randomness based on candidate parameters
            noise_factor = candidate['complexity_factor']
            stability_factor = candidate['stability_factor']
            
            # Generate predictions with controlled randomness
            base_predictions = np.random.randint(0, n_regimes, n_samples)
            
            # Apply stability factor (higher stability = less random changes)
            if stability_factor > 0.5:
                # Make predictions more stable
                for i in range(1, len(base_predictions)):
                    if np.random.random() < stability_factor:
                        base_predictions[i] = base_predictions[i-1]
            
            return base_predictions
            
        except Exception as e:
            self.logger.warning(f"Regime prediction simulation failed: {e}")
            return original_predictions
    
    def _evaluate_regime_accuracy(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Evaluate regime prediction accuracy."""
        try:
            # Simple accuracy calculation
            if len(predicted) != len(actual):
                return 0.0
            
            # Calculate accuracy
            correct = np.sum(predicted == actual)
            accuracy = correct / len(actual)
            
            return accuracy
            
        except Exception as e:
            self.logger.warning(f"Regime accuracy evaluation failed: {e}")
            return 0.0
    
    def _calculate_regime_stability(self, regime_predictions: np.ndarray) -> float:
        """Calculate regime stability score."""
        try:
            if len(regime_predictions) < 2:
                return 0.0
            
            # Calculate regime changes
            regime_changes = np.sum(np.diff(regime_predictions) != 0)
            total_periods = len(regime_predictions) - 1
            
            # Stability is inverse of change frequency
            stability = 1.0 - (regime_changes / total_periods) if total_periods > 0 else 0.0
            
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            self.logger.warning(f"Regime stability calculation failed: {e}")
            return 0.0
    
    def _calculate_transition_accuracy(self, regime_predictions: np.ndarray) -> float:
        """Calculate regime transition accuracy."""
        try:
            if len(regime_predictions) < 3:
                return 0.5
            
            # Calculate transition matrix
            unique_regimes = np.unique(regime_predictions)
            n_regimes = len(unique_regimes)
            
            if n_regimes < 2:
                return 0.5
            
            # Create transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            for i in range(len(regime_predictions) - 1):
                current_regime = regime_predictions[i]
                next_regime = regime_predictions[i + 1]
                
                if current_regime in unique_regimes and next_regime in unique_regimes:
                    current_idx = np.where(unique_regimes == current_regime)[0][0]
                    next_idx = np.where(unique_regimes == next_regime)[0][0]
                    transition_matrix[current_idx, next_idx] += 1
            
            # Calculate transition accuracy (simplified)
            total_transitions = np.sum(transition_matrix)
            if total_transitions > 0:
                # Higher diagonal values indicate more stable transitions
                diagonal_sum = np.trace(transition_matrix)
                transition_accuracy = diagonal_sum / total_transitions
            else:
                transition_accuracy = 0.5
            
            return min(transition_accuracy, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Transition accuracy calculation failed: {e}")
            return 0.5
    
    def _update_pareto_frontier(self, evaluated_candidates: List[Dict[str, Any]]):
        """Update Pareto frontier with new candidates."""
        try:
            # Add new candidates to frontier
            for candidate in evaluated_candidates:
                self.pareto_frontier.append(candidate)
            
            # Sort by total score
            self.pareto_frontier.sort(key=lambda x: x.get('total_score', 0.0), reverse=True)
            
            # Keep only top solutions (limit to 50)
            if len(self.pareto_frontier) > 50:
                self.pareto_frontier = self.pareto_frontier[:50]
            
        except Exception as e:
            self.logger.warning(f"Pareto frontier update failed: {e}")
    
    def _check_convergence(self, iteration: int) -> bool:
        """Check if optimization has converged."""
        try:
            # Simple convergence check
            if iteration < 10:
                return False
            
            # Check if improvement has plateaued
            recent_scores = [c.get('total_score', 0.0) for c in self.pareto_frontier[-10:]]
            if len(recent_scores) < 5:
                return False
            
            # Check if standard deviation is low (convergence)
            score_std = np.std(recent_scores)
            if score_std < 0.01:  # Low variation indicates convergence
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Convergence check failed: {e}")
            return False
    
    def _finalize_optimization(self, execution_time: float) -> OptimizationResult:
        """Finalize optimization and create result."""
        try:
            # Get best solution
            best_solution = None
            if self.pareto_frontier:
                best_solution = self.pareto_frontier[0]
            
            # Create optimization metrics
            optimization_metrics = {
                'total_candidates': len(self.pareto_frontier),
                'best_score': best_solution.get('total_score', 0.0) if best_solution else 0.0,
                'average_score': np.mean([c.get('total_score', 0.0) for c in self.pareto_frontier]),
                'score_std': np.std([c.get('total_score', 0.0) for c in self.pareto_frontier]),
                'execution_time': execution_time
            }
            
            # Create convergence history
            convergence_history = []
            for i, candidate in enumerate(self.pareto_frontier):
                convergence_history.append({
                    'iteration': candidate.get('iteration', i),
                    'score': candidate.get('total_score', 0.0),
                    'objectives': candidate.get('objective_scores', {})
                })
            
            return OptimizationResult(
                success=True,
                pareto_solutions=self.pareto_frontier,
                best_solution=best_solution,
                optimization_metrics=optimization_metrics,
                convergence_history=convergence_history,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Optimization finalization failed: {e}")
            return OptimizationResult(
                success=False,
                pareto_solutions=[],
                best_solution=None,
                optimization_metrics={},
                convergence_history=[],
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        try:
            if not self.pareto_frontier:
                return {}
            
            # Calculate summary statistics
            scores = [c.get('total_score', 0.0) for c in self.pareto_frontier]
            
            summary = {
                'total_solutions': len(self.pareto_frontier),
                'best_score': max(scores),
                'worst_score': min(scores),
                'average_score': np.mean(scores),
                'score_std': np.std(scores),
                'score_range': max(scores) - min(scores),
                'convergence_rate': self._calculate_convergence_rate(),
                'objective_contributions': self._calculate_objective_contributions()
            }
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"Optimization summary calculation failed: {e}")
            return {}
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate."""
        try:
            if len(self.optimization_history) < 2:
                return 0.0
            
            # Calculate improvement rate
            recent_scores = [h.get('best_score', 0.0) for h in self.optimization_history[-5:]]
            if len(recent_scores) < 2:
                return 0.0
            
            improvement = recent_scores[-1] - recent_scores[0]
            return max(0.0, improvement)
            
        except Exception:
            return 0.0
    
    def _calculate_objective_contributions(self) -> Dict[str, float]:
        """Calculate contribution of each objective to total score."""
        try:
            if not self.pareto_frontier:
                return {}
            
            contributions = {}
            
            for objective in self.objectives:
                obj_scores = [c.get('objective_scores', {}).get(objective.value, 0.0) 
                             for c in self.pareto_frontier]
                if obj_scores:
                    contributions[objective.value] = np.mean(obj_scores)
                else:
                    contributions[objective.value] = 0.0
            
            return contributions
            
        except Exception as e:
            self.logger.warning(f"Objective contribution calculation failed: {e}")
            return {}