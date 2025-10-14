"""
Robust MOEA Optimizer with Comprehensive Convergence Criteria

Implements robust multi-objective evolutionary algorithm optimization with
deterministic "anytime" stopping conditions and comprehensive convergence monitoring.

Key Features:
- Robust convergence criteria
- Anytime stopping conditions
- Hypervolume and ε-progress monitoring
- Time-boxed optimization
- Performance tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from enum import Enum
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

logger = logging.getLogger(__name__)


class ConvergenceCriterion(Enum):
    """Types of convergence criteria."""
    HYPERVOLUME_TOLERANCE = "hypervolume_tolerance"
    EPSILON_PROGRESS = "epsilon_progress"
    STAGNATION_GENERATIONS = "stagnation_generations"
    MAX_GENERATIONS = "max_generations"
    MAX_EVALUATIONS = "max_evaluations"
    MAX_TIME = "max_time"
    MIN_IMPROVEMENT = "min_improvement"
    DIVERSITY_THRESHOLD = "diversity_threshold"


class OptimizationAlgorithm(Enum):
    """Types of optimization algorithms."""
    NSGA2 = "nsga2"
    SPEA2 = "spea2"
    MOEA_D = "moea_d"
    NSGA3 = "nsga3"
    CUSTOM = "custom"


@dataclass
class ConvergenceConfig:
    """Configuration for convergence criteria."""
    
    # Basic parameters
    max_generations: int = 100
    max_evaluations: int = 10000
    max_time_seconds: float = 3600.0  # 1 hour max
    
    # Convergence criteria
    hypervolume_tolerance: float = 1e-6
    epsilon_progress_tolerance: float = 1e-6
    stagnation_generations: int = 20
    min_improvement: float = 1e-8
    diversity_threshold: float = 0.01
    
    # Anytime stopping
    enable_anytime_stop: bool = True
    min_generations: int = 10
    min_evaluations: int = 100
    min_time_seconds: float = 60.0  # 1 minute minimum
    
    # Progress monitoring
    progress_window: int = 10  # Generations to look back for progress
    progress_threshold: float = 0.01  # Minimum progress threshold
    
    # Performance constraints
    memory_limit_mb: float = 8000.0
    cpu_limit_percent: float = 90.0
    
    # Parallel processing
    enable_parallel: bool = True
    max_workers: int = 4
    
    # Logging and monitoring
    log_frequency: int = 10  # Log every N generations
    save_intermediate_results: bool = False
    intermediate_save_frequency: int = 50


@dataclass
class ConvergenceResult:
    """Result from convergence monitoring."""
    
    # Convergence status
    converged: bool
    convergence_criterion: ConvergenceCriterion
    generations_completed: int
    evaluations_completed: int
    time_elapsed: float
    
    # Convergence metrics
    final_hypervolume: float
    hypervolume_improvement: float
    epsilon_progress: float
    stagnation_count: int
    diversity_score: float
    
    # Performance metrics
    average_generation_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    
    # Optimization results
    best_solutions: List[Dict[str, Any]]
    pareto_front: List[Dict[str, Any]]
    objective_values: Dict[str, List[float]]
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class RobustMOEAOptimizer:
    """
    Robust MOEA optimizer with comprehensive convergence criteria.
    
    This class provides robust multi-objective optimization with deterministic
    "anytime" stopping conditions and comprehensive convergence monitoring.
    """
    
    def __init__(self, config: Optional[ConvergenceConfig] = None):
        """
        Initialize the robust MOEA optimizer.
        
        Args:
            config: Configuration for convergence criteria
        """
        self.config = config or ConvergenceConfig()
        self.logger = logger
        
        # Initialize performance tracking
        self.performance_history = []
        self.convergence_history = []
        
        # Set up parallel processing
        if self.config.enable_parallel:
            self.max_workers = min(self.config.max_workers, mp.cpu_count())
        else:
            self.max_workers = 1
        
        tprint_info("🧬 Robust MOEA Optimizer initialized")
        tprint_debug(f"📊 Max generations: {self.config.max_generations}")
        tprint_debug(f"📊 Max time: {self.config.max_time_seconds}s")
        tprint_debug(f"📊 Max workers: {self.max_workers}")
    
    def optimize(self, 
                problem: Callable,
                algorithm: OptimizationAlgorithm = OptimizationAlgorithm.NSGA2,
                **kwargs) -> ConvergenceResult:
        """
        Optimize using robust MOEA with comprehensive convergence criteria.
        
        Args:
            problem: Optimization problem to solve
            algorithm: Optimization algorithm to use
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            ConvergenceResult with optimization results
        """
        start_time = time.time()
        
        tprint_info("🧬 Starting robust MOEA optimization...")
        tprint_debug(f"📊 Algorithm: {algorithm.value}")
        tprint_debug(f"📊 Problem: {problem.__name__ if hasattr(problem, '__name__') else 'Unknown'}")
        
        try:
            # Initialize optimization state
            optimization_state = self._initialize_optimization_state(problem, algorithm, **kwargs)
            
            # Run optimization with convergence monitoring
            result = self._run_optimization_with_convergence(
                problem, algorithm, optimization_state, start_time
            )
            
            # Finalize results
            result = self._finalize_optimization_result(result, start_time)
            
            tprint_success(f"✅ Robust MOEA optimization completed")
            tprint_info(f"📊 Generations: {result.generations_completed}")
            tprint_info(f"📊 Evaluations: {result.evaluations_completed}")
            tprint_info(f"📊 Time: {result.time_elapsed:.2f}s")
            tprint_info(f"📊 Converged: {result.converged}")
            tprint_info(f"📊 Final hypervolume: {result.final_hypervolume:.6f}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Robust MOEA optimization failed: {e}")
            return self._create_error_result(start_time, str(e))
    
    def _initialize_optimization_state(self, 
                                     problem: Callable,
                                     algorithm: OptimizationAlgorithm,
                                     **kwargs) -> Dict[str, Any]:
        """Initialize optimization state."""
        return {
            'problem': problem,
            'algorithm': algorithm,
            'algorithm_params': kwargs,
            'generation': 0,
            'evaluation_count': 0,
            'best_solutions': [],
            'pareto_front': [],
            'hypervolume_history': [],
            'epsilon_progress_history': [],
            'diversity_history': [],
            'stagnation_count': 0,
            'last_improvement_generation': 0,
            'start_time': time.time()
        }
    
    def _run_optimization_with_convergence(self, 
                                         problem: Callable,
                                         algorithm: OptimizationAlgorithm,
                                         state: Dict[str, Any],
                                         start_time: float) -> ConvergenceResult:
        """Run optimization with comprehensive convergence monitoring."""
        
        # Main optimization loop
        for generation in range(self.config.max_generations):
            generation_start = time.time()
            
            # Check time limit
            if time.time() - start_time > self.config.max_time_seconds:
                tprint_info(f"⏰ Time limit reached at generation {generation}")
                break
            
            # Run one generation
            generation_result = self._run_single_generation(
                problem, algorithm, state, generation
            )
            
            # Update state
            state.update(generation_result)
            state['generation'] = generation + 1
            
            # Check convergence
            convergence_status = self._check_convergence(state, generation)
            
            # Log progress
            if generation % self.config.log_frequency == 0:
                self._log_optimization_progress(state, generation, generation_start)
            
            # Save intermediate results
            if (self.config.save_intermediate_results and 
                generation % self.config.intermediate_save_frequency == 0):
                self._save_intermediate_results(state, generation)
            
            # Check if converged
            if convergence_status['converged']:
                tprint_info(f"✅ Convergence achieved at generation {generation}")
                break
        
        # Create convergence result
        return self._create_convergence_result(state, convergence_status, start_time)
    
    def _run_single_generation(self, 
                             problem: Callable,
                             algorithm: OptimizationAlgorithm,
                             state: Dict[str, Any],
                             generation: int) -> Dict[str, Any]:
        """Run a single generation of optimization."""
        generation_start = time.time()
        
        try:
            # This is a simplified implementation
            # In practice, you'd integrate with actual MOEA libraries (DEAP, pymoo, etc.)
            
            # Simulate generation (replace with actual algorithm)
            if algorithm == OptimizationAlgorithm.NSGA2:
                generation_result = self._simulate_nsga2_generation(problem, state, generation)
            elif algorithm == OptimizationAlgorithm.SPEA2:
                generation_result = self._simulate_spea2_generation(problem, state, generation)
            elif algorithm == OptimizationAlgorithm.MOEA_D:
                generation_result = self._simulate_moea_d_generation(problem, state, generation)
            else:
                generation_result = self._simulate_generic_generation(problem, state, generation)
            
            # Update evaluation count
            generation_result['evaluation_count'] = state['evaluation_count'] + generation_result.get('evaluations', 0)
            
            # Calculate metrics
            generation_result.update(self._calculate_generation_metrics(state, generation_result))
            
            # Update performance history
            generation_time = time.time() - generation_start
            self.performance_history.append({
                'generation': generation,
                'time': generation_time,
                'evaluations': generation_result.get('evaluations', 0),
                'hypervolume': generation_result.get('hypervolume', 0.0),
                'diversity': generation_result.get('diversity', 0.0)
            })
            
            return generation_result
            
        except Exception as e:
            tprint_error(f"❌ Generation {generation} failed: {e}")
            return {
                'evaluations': 0,
                'hypervolume': state.get('hypervolume_history', [0.0])[-1] if state.get('hypervolume_history') else 0.0,
                'diversity': 0.0,
                'solutions': [],
                'error': str(e)
            }
    
    def _simulate_nsga2_generation(self, 
                                  problem: Callable,
                                  state: Dict[str, Any],
                                  generation: int) -> Dict[str, Any]:
        """Simulate NSGA-II generation (placeholder implementation)."""
        # This is a placeholder - in practice, you'd use actual NSGA-II implementation
        
        # Simulate some optimization progress
        base_hypervolume = 0.5 + 0.3 * np.exp(-generation / 50)  # Simulated improvement
        noise = np.random.normal(0, 0.01)
        hypervolume = max(0.1, base_hypervolume + noise)
        
        # Simulate solutions
        n_solutions = min(100, 50 + generation // 10)
        solutions = []
        for i in range(n_solutions):
            solutions.append({
                'id': f"gen_{generation}_sol_{i}",
                'objectives': [np.random.random(), np.random.random()],
                'variables': np.random.random(10).tolist(),
                'fitness': np.random.random()
            })
        
        return {
            'evaluations': n_solutions,
            'hypervolume': hypervolume,
            'diversity': np.random.random(),
            'solutions': solutions,
            'pareto_front': solutions[:10]  # Top 10 solutions
        }
    
    def _simulate_spea2_generation(self, 
                                  problem: Callable,
                                  state: Dict[str, Any],
                                  generation: int) -> Dict[str, Any]:
        """Simulate SPEA2 generation (placeholder implementation)."""
        # Similar to NSGA-II but with different characteristics
        base_hypervolume = 0.4 + 0.4 * np.exp(-generation / 60)
        noise = np.random.normal(0, 0.015)
        hypervolume = max(0.1, base_hypervolume + noise)
        
        n_solutions = min(80, 40 + generation // 8)
        solutions = []
        for i in range(n_solutions):
            solutions.append({
                'id': f"gen_{generation}_sol_{i}",
                'objectives': [np.random.random(), np.random.random()],
                'variables': np.random.random(10).tolist(),
                'fitness': np.random.random()
            })
        
        return {
            'evaluations': n_solutions,
            'hypervolume': hypervolume,
            'diversity': np.random.random(),
            'solutions': solutions,
            'pareto_front': solutions[:8]
        }
    
    def _simulate_moea_d_generation(self, 
                                   problem: Callable,
                                   state: Dict[str, Any],
                                   generation: int) -> Dict[str, Any]:
        """Simulate MOEA/D generation (placeholder implementation)."""
        # MOEA/D typically has different convergence characteristics
        base_hypervolume = 0.3 + 0.5 * (1 - np.exp(-generation / 40))
        noise = np.random.normal(0, 0.02)
        hypervolume = max(0.1, base_hypervolume + noise)
        
        n_solutions = min(120, 60 + generation // 5)
        solutions = []
        for i in range(n_solutions):
            solutions.append({
                'id': f"gen_{generation}_sol_{i}",
                'objectives': [np.random.random(), np.random.random()],
                'variables': np.random.random(10).tolist(),
                'fitness': np.random.random()
            })
        
        return {
            'evaluations': n_solutions,
            'hypervolume': hypervolume,
            'diversity': np.random.random(),
            'solutions': solutions,
            'pareto_front': solutions[:12]
        }
    
    def _simulate_generic_generation(self, 
                                   problem: Callable,
                                   state: Dict[str, Any],
                                   generation: int) -> Dict[str, Any]:
        """Simulate generic generation (placeholder implementation)."""
        # Generic simulation with moderate improvement
        base_hypervolume = 0.2 + 0.6 * (1 - np.exp(-generation / 30))
        noise = np.random.normal(0, 0.025)
        hypervolume = max(0.1, base_hypervolume + noise)
        
        n_solutions = min(90, 45 + generation // 7)
        solutions = []
        for i in range(n_solutions):
            solutions.append({
                'id': f"gen_{generation}_sol_{i}",
                'objectives': [np.random.random(), np.random.random()],
                'variables': np.random.random(10).tolist(),
                'fitness': np.random.random()
            })
        
        return {
            'evaluations': n_solutions,
            'hypervolume': hypervolume,
            'diversity': np.random.random(),
            'solutions': solutions,
            'pareto_front': solutions[:9]
        }
    
    def _calculate_generation_metrics(self, 
                                    state: Dict[str, Any],
                                    generation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for a generation."""
        try:
            # Update hypervolume history
            hypervolume = generation_result.get('hypervolume', 0.0)
            state['hypervolume_history'].append(hypervolume)
            
            # Calculate hypervolume improvement
            if len(state['hypervolume_history']) > 1:
                hypervolume_improvement = hypervolume - state['hypervolume_history'][-2]
            else:
                hypervolume_improvement = 0.0
            
            # Calculate epsilon progress
            epsilon_progress = self._calculate_epsilon_progress(state, generation_result)
            state['epsilon_progress_history'].append(epsilon_progress)
            
            # Calculate diversity
            diversity = generation_result.get('diversity', 0.0)
            state['diversity_history'].append(diversity)
            
            return {
                'hypervolume_improvement': hypervolume_improvement,
                'epsilon_progress': epsilon_progress,
                'diversity': diversity
            }
            
        except Exception as e:
            tprint_debug(f"⚠️ Metric calculation failed: {e}")
            return {
                'hypervolume_improvement': 0.0,
                'epsilon_progress': 0.0,
                'diversity': 0.0
            }
    
    def _calculate_epsilon_progress(self, 
                                  state: Dict[str, Any],
                                  generation_result: Dict[str, Any]) -> float:
        """Calculate epsilon progress metric."""
        try:
            # Simplified epsilon progress calculation
            # In practice, you'd use proper epsilon-dominance metrics
            
            if len(state['hypervolume_history']) < 2:
                return 0.0
            
            # Calculate improvement rate
            recent_hypervolumes = state['hypervolume_history'][-self.config.progress_window:]
            if len(recent_hypervolumes) < 2:
                return 0.0
            
            improvement_rate = (recent_hypervolumes[-1] - recent_hypervolumes[0]) / len(recent_hypervolumes)
            
            return max(0.0, improvement_rate)
            
        except Exception:
            return 0.0
    
    def _check_convergence(self, 
                         state: Dict[str, Any],
                         generation: int) -> Dict[str, Any]:
        """Check convergence criteria."""
        convergence_status = {
            'converged': False,
            'criterion': None,
            'reason': None
        }
        
        try:
            # Check minimum requirements
            if generation < self.config.min_generations:
                return convergence_status
            
            if state['evaluation_count'] < self.config.min_evaluations:
                return convergence_status
            
            if time.time() - state['start_time'] < self.config.min_time_seconds:
                return convergence_status
            
            # Check hypervolume tolerance
            if len(state['hypervolume_history']) >= 2:
                recent_improvement = abs(state['hypervolume_history'][-1] - state['hypervolume_history'][-2])
                if recent_improvement < self.config.hypervolume_tolerance:
                    state['stagnation_count'] += 1
                else:
                    state['stagnation_count'] = 0
                    state['last_improvement_generation'] = generation
                
                if state['stagnation_count'] >= self.config.stagnation_generations:
                    convergence_status.update({
                        'converged': True,
                        'criterion': ConvergenceCriterion.STAGNATION_GENERATIONS,
                        'reason': f"Stagnation for {state['stagnation_count']} generations"
                    })
                    return convergence_status
            
            # Check epsilon progress
            if len(state['epsilon_progress_history']) >= self.config.progress_window:
                recent_progress = state['epsilon_progress_history'][-self.config.progress_window:]
                avg_progress = np.mean(recent_progress)
                
                if avg_progress < self.config.epsilon_progress_tolerance:
                    convergence_status.update({
                        'converged': True,
                        'criterion': ConvergenceCriterion.EPSILON_PROGRESS,
                        'reason': f"Epsilon progress below threshold: {avg_progress:.2e}"
                    })
                    return convergence_status
            
            # Check diversity threshold
            if len(state['diversity_history']) >= 2:
                recent_diversity = state['diversity_history'][-1]
                if recent_diversity < self.config.diversity_threshold:
                    convergence_status.update({
                        'converged': True,
                        'criterion': ConvergenceCriterion.DIVERSITY_THRESHOLD,
                        'reason': f"Diversity below threshold: {recent_diversity:.2e}"
                    })
                    return convergence_status
            
            # Check minimum improvement
            if len(state['hypervolume_history']) >= 2:
                total_improvement = state['hypervolume_history'][-1] - state['hypervolume_history'][0]
                if total_improvement < self.config.min_improvement:
                    convergence_status.update({
                        'converged': True,
                        'criterion': ConvergenceCriterion.MIN_IMPROVEMENT,
                        'reason': f"Total improvement below threshold: {total_improvement:.2e}"
                    })
                    return convergence_status
            
        except Exception as e:
            tprint_debug(f"⚠️ Convergence check failed: {e}")
        
        return convergence_status
    
    def _log_optimization_progress(self, 
                                 state: Dict[str, Any],
                                 generation: int,
                                 generation_start: float):
        """Log optimization progress."""
        try:
            generation_time = time.time() - generation_start
            hypervolume = state['hypervolume_history'][-1] if state['hypervolume_history'] else 0.0
            diversity = state['diversity_history'][-1] if state['diversity_history'] else 0.0
            stagnation = state['stagnation_count']
            
            tprint_info(f"📊 Gen {generation:3d}: HV={hypervolume:.6f}, Div={diversity:.4f}, "
                       f"Stag={stagnation:2d}, Time={generation_time:.2f}s")
            
        except Exception as e:
            tprint_debug(f"⚠️ Progress logging failed: {e}")
    
    def _save_intermediate_results(self, state: Dict[str, Any], generation: int):
        """Save intermediate optimization results."""
        try:
            # This would save intermediate results to disk
            # Implementation depends on specific requirements
            pass
        except Exception as e:
            tprint_debug(f"⚠️ Intermediate save failed: {e}")
    
    def _create_convergence_result(self, 
                                 state: Dict[str, Any],
                                 convergence_status: Dict[str, Any],
                                 start_time: float) -> ConvergenceResult:
        """Create convergence result from optimization state."""
        try:
            # Calculate final metrics
            final_hypervolume = state['hypervolume_history'][-1] if state['hypervolume_history'] else 0.0
            hypervolume_improvement = (final_hypervolume - state['hypervolume_history'][0] 
                                     if len(state['hypervolume_history']) > 1 else 0.0)
            
            epsilon_progress = state['epsilon_progress_history'][-1] if state['epsilon_progress_history'] else 0.0
            diversity_score = state['diversity_history'][-1] if state['diversity_history'] else 0.0
            
            # Calculate performance metrics
            total_time = time.time() - start_time
            avg_generation_time = total_time / max(1, state['generation'])
            
            # Estimate memory usage
            memory_usage = self._estimate_memory_usage(state)
            
            # Get best solutions
            best_solutions = state.get('solutions', [])
            pareto_front = state.get('pareto_front', [])
            
            # Extract objective values
            objective_values = self._extract_objective_values(best_solutions)
            
            return ConvergenceResult(
                converged=convergence_status['converged'],
                convergence_criterion=convergence_status.get('criterion'),
                generations_completed=state['generation'],
                evaluations_completed=state['evaluation_count'],
                time_elapsed=total_time,
                final_hypervolume=final_hypervolume,
                hypervolume_improvement=hypervolume_improvement,
                epsilon_progress=epsilon_progress,
                stagnation_count=state['stagnation_count'],
                diversity_score=diversity_score,
                average_generation_time=avg_generation_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=0.0,  # Would need system monitoring
                best_solutions=best_solutions,
                pareto_front=pareto_front,
                objective_values=objective_values,
                metadata={
                    'algorithm': state['algorithm'].value,
                    'convergence_reason': convergence_status.get('reason'),
                    'performance_history': self.performance_history
                }
            )
            
        except Exception as e:
            tprint_error(f"❌ Failed to create convergence result: {e}")
            return self._create_error_result(start_time, str(e))
    
    def _extract_objective_values(self, solutions: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Extract objective values from solutions."""
        try:
            if not solutions:
                return {}
            
            # Extract objectives from first solution to get structure
            first_solution = solutions[0]
            objectives = first_solution.get('objectives', [])
            
            # Create objective value lists
            objective_values = {}
            for i, obj in enumerate(objectives):
                objective_values[f'objective_{i}'] = []
            
            # Extract values from all solutions
            for solution in solutions:
                solution_objectives = solution.get('objectives', [])
                for i, obj in enumerate(solution_objectives):
                    if f'objective_{i}' in objective_values:
                        objective_values[f'objective_{i}'].append(obj)
            
            return objective_values
            
        except Exception:
            return {}
    
    def _estimate_memory_usage(self, state: Dict[str, Any]) -> float:
        """Estimate memory usage in MB."""
        try:
            # Rough estimation based on state size
            memory_usage = 0.0
            
            # Add memory for solutions
            solutions = state.get('solutions', [])
            memory_usage += len(solutions) * 0.001  # Rough estimate per solution
            
            # Add memory for history lists
            history_lists = ['hypervolume_history', 'epsilon_progress_history', 'diversity_history']
            for history_name in history_lists:
                history = state.get(history_name, [])
                memory_usage += len(history) * 0.0001  # Rough estimate per entry
            
            return memory_usage
            
        except Exception:
            return 0.0
    
    def _finalize_optimization_result(self, 
                                    result: ConvergenceResult,
                                    start_time: float) -> ConvergenceResult:
        """Finalize optimization result with additional processing."""
        try:
            # Add any final processing here
            result.time_elapsed = time.time() - start_time
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Result finalization failed: {e}")
            return result
    
    def _create_error_result(self, start_time: float, error_message: str) -> ConvergenceResult:
        """Create error result for failed optimization."""
        return ConvergenceResult(
            converged=False,
            convergence_criterion=None,
            generations_completed=0,
            evaluations_completed=0,
            time_elapsed=time.time() - start_time,
            final_hypervolume=0.0,
            hypervolume_improvement=0.0,
            epsilon_progress=0.0,
            stagnation_count=0,
            diversity_score=0.0,
            average_generation_time=0.0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            best_solutions=[],
            pareto_front=[],
            objective_values={},
            metadata={'error': True, 'error_message': error_message}
        )


# Convenience functions
def optimize_with_robust_moea(problem: Callable,
                            algorithm: OptimizationAlgorithm = OptimizationAlgorithm.NSGA2,
                            config: Optional[ConvergenceConfig] = None,
                            **kwargs) -> ConvergenceResult:
    """
    Convenience function to optimize with robust MOEA.
    
    Args:
        problem: Optimization problem to solve
        algorithm: Optimization algorithm to use
        config: Convergence configuration
        **kwargs: Additional algorithm-specific parameters
        
    Returns:
        ConvergenceResult with optimization results
    """
    optimizer = RobustMOEAOptimizer(config)
    return optimizer.optimize(problem, algorithm, **kwargs)


# Export main classes and functions
__all__ = [
    'RobustMOEAOptimizer',
    'ConvergenceConfig',
    'ConvergenceResult',
    'ConvergenceCriterion',
    'OptimizationAlgorithm',
    'optimize_with_robust_moea'
]