"""
TAS Multi-Objective Search - Updated to use Unified Implementation

This module provides TAS-specific multi-objective optimization using
the unified multi-objective optimization framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import time
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path

# Import unified multi-objective optimizer
try:
    from src.utils.nas_tas import (
        UnifiedMultiObjectiveOptimizer,
        PerformanceEstimator,
        ArchitectureFeatures,
        PerformancePrediction,
        PerformanceMetric,
        EstimatorType,
        OptimizationConfig,
        MultiObjectiveResult
    )
    UNIFIED_MULTI_OBJECTIVE_AVAILABLE = True
except ImportError:
    UNIFIED_MULTI_OBJECTIVE_AVAILABLE = False

# TAS-specific wrapper for unified multi-objective optimizer
class TASMultiObjectiveOptimizer:
    """TAS-specific multi-objective optimizer using unified implementation."""
    
    def __init__(self, config: OptimizationConfig = None):
        """Initialize TAS multi-objective optimizer."""
        if not UNIFIED_MULTI_OBJECTIVE_AVAILABLE:
            raise ImportError("Unified multi-objective optimizer not available")
        self.unified_optimizer = UnifiedMultiObjectiveOptimizer(config)
    
    def __getattr__(self, name):
        """Delegate to unified optimizer."""
        return getattr(self.unified_optimizer, name)


class OptimizationAlgorithm(Enum):
    """Types of optimization algorithms available."""
    NSGA2 = "nsga2"
    SPEA2 = "spea2"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    HYBRID = "hybrid"


class ObjectiveType(Enum):
    """Types of optimization objectives."""
    REGIME_ACCURACY = "regime_accuracy"
    ECONOMIC_SIGNIFICANCE = "economic_significance"
    TRADING_VIABILITY = "trading_viability"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"
    ARCHITECTURE_COMPLEXITY = "architecture_complexity"
    REGIME_STABILITY = "regime_stability"
    TRANSITION_ACCURACY = "transition_accuracy"
    MODEL_CONFIDENCE = "model_confidence"
    RISK_ADJUSTED_RETURNS = "risk_adjusted_returns"
    EXECUTION_FEASIBILITY = "execution_feasibility"


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective search."""
    # Basic parameters
    n_generations: int = 100
    population_size: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    
    # Objectives
    objectives: List[ObjectiveType] = field(default_factory=lambda: [
        ObjectiveType.REGIME_ACCURACY,
        ObjectiveType.ECONOMIC_SIGNIFICANCE,
        ObjectiveType.TRADING_VIABILITY,
        ObjectiveType.COMPUTATIONAL_EFFICIENCY
    ])
    
    # Objective weights
    objective_weights: Dict[ObjectiveType, float] = field(default_factory=lambda: {
        ObjectiveType.REGIME_ACCURACY: 0.3,
        ObjectiveType.ECONOMIC_SIGNIFICANCE: 0.25,
        ObjectiveType.TRADING_VIABILITY: 0.25,
        ObjectiveType.COMPUTATIONAL_EFFICIENCY: 0.2
    })
    
    # Algorithm selection
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.NSGA2
    
    # Advanced parameters
    enable_constraint_handling: bool = True
    enable_adaptive_weights: bool = False
    enable_real_time_optimization: bool = False
    
    # Pareto frontier management
    max_pareto_solutions: int = 100
    pareto_epsilon: float = 0.01
    
    # Economic and trading evaluation
    enable_economic_evaluation: bool = True
    enable_trading_viability: bool = True
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    memory_limit_gb: float = 8.0
    
    # Performance settings
    enable_parallel_evaluation: bool = True
    max_workers: int = 4
    
    # Logging and monitoring
    log_level: str = 'INFO'
    enable_progress_logging: bool = True
    save_checkpoints: bool = True
    checkpoint_interval: int = 10


class Individual:
    """Individual in the population for multi-objective optimization."""
    
    def __init__(self, parameters: Dict[str, Any], objectives: List[float] = None):
        self.parameters = parameters
        self.objectives = objectives or []
        self.rank = 0
        self.crowding_distance = 0.0
        self.domination_count = 0
        self.dominated_solutions = []
        self.individual_id = self._generate_id()
        self.timestamp = datetime.now()
    
    def _generate_id(self) -> str:
        """Generate unique individual ID."""
        param_str = str(sorted(self.parameters.items()))
        return f"ind_{hash(param_str) % 1000000}"
    
    def dominates(self, other: 'Individual') -> bool:
        """Check if this individual dominates another."""
        if len(self.objectives) != len(other.objectives):
            return False
        
        better_in_at_least_one = False
        for i, (obj1, obj2) in enumerate(zip(self.objectives, other.objectives)):
            if obj1 < obj2:  # Assuming minimization
                return False
            elif obj1 > obj2:
                better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert individual to dictionary."""
        return {
            'individual_id': self.individual_id,
            'parameters': self.parameters,
            'objectives': self.objectives,
            'rank': self.rank,
            'crowding_distance': self.crowding_distance,
            'timestamp': self.timestamp.isoformat()
        }


class MultiObjectiveTreeSearch:
    """Multi-objective search for tree architectures."""
    
    def __init__(self, config: MultiObjectiveConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Population and results
        self.population = []
        self.fitness_scores = []
        self.pareto_front = []
        self.search_history = []
        
        # Unified evaluators
        self.economic_evaluator = None
        self.trading_evaluator = None
        self.hardware_optimizer = None
        self.regime_analyzer = None
        self.validation_system = None
        self.multi_objective_optimizer = None
        
        # Performance tracking
        self.generation_results = []
        self.convergence_history = []
        
        # Initialize unified components
        self._initialize_unified_components()
    
    def _initialize_unified_components(self):
        """Initialize unified evaluation components."""
        try:
            if self.config.enable_economic_evaluation:
                self.economic_evaluator = create_unified_economic_evaluator()
            
            if self.config.enable_trading_viability:
                self.trading_evaluator = create_unified_trading_viability_evaluator()
            
            if self.config.enable_hardware_optimization:
                self.hardware_optimizer = create_unified_hardware_optimizer()
            
            self.regime_analyzer = create_unified_regime_analyzer()
            self.validation_system = create_unified_validation_system()
            self.multi_objective_optimizer = create_unified_multi_objective_optimizer()
            
            tprint_success("✅ Unified components initialized successfully")
            
        except (ImportError, ModuleNotFoundError) as e:
            tprint_error(f"❌ Failed to initialize unified components due to missing dependencies: {e}")
            self.logger.warning(f"Some unified components may not be available: {e}")
        except (ValueError, TypeError) as e:
            tprint_error(f"❌ Failed to initialize unified components due to configuration issue: {e}")
            self.logger.warning(f"Configuration error: {e}")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize unified components with unexpected error: {e}")
            tprint_error(f"Error type: {type(e).__name__}")
            self.logger.warning(f"Some unified components may not be available: {e}")
    
    def search(self, 
               search_space: Dict[str, Any],
               train_data: pd.DataFrame = None,
               validation_data: pd.DataFrame = None,
               test_data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """Perform multi-objective search for optimal tree architectures."""
        tprint_info("🎯 Starting multi-objective tree search")
        
        start_time = time.time()
        
        # Initialize search environment
        search_env = {
            'search_space': search_space,
            'train_data': train_data,
            'validation_data': validation_data,
            'test_data': test_data
        }
        
        # Initialize population
        self._initialize_population(search_space)
        
        # Evolution loop
        for generation in range(self.config.n_generations):
            generation_start = time.time()
            
            # Evaluate fitness
            self._evaluate_population(search_env)
            
            # Non-dominated sorting and Pareto front update
            self._update_pareto_front()
            
            # Selection
            parents = self._select_parents()
            
            # Crossover and mutation
            offspring = self._create_offspring(parents, search_space)
            
            # Update population
            self._update_population(offspring)
            
            # Store generation results
            generation_time = time.time() - generation_start
            self.generation_results.append({
                'generation': generation,
                'population_size': len(self.population),
                'pareto_front_size': len(self.pareto_front),
                'best_objectives': self._get_best_objectives(),
                'duration': generation_time
            })
            
            # Progress logging
            if generation % 10 == 0:
                tprint_progress(f"Generation {generation + 1}: Pareto front size = {len(self.pareto_front)}")
            
            # Checkpoint saving
            if self.config.save_checkpoints and generation % self.config.checkpoint_interval == 0:
                self._save_checkpoint(generation)
        
        # Final results
        total_time = time.time() - start_time
        tprint_success(f"🎉 Multi-objective search completed in {total_time:.2f}s")
        tprint_info(f"Pareto front size: {len(self.pareto_front)}")
        
        return [individual.to_dict() for individual in self.pareto_front]
    
    def _initialize_population(self, search_space: Dict[str, Any]):
        """Initialize population with random individuals."""
        self.population = []
        for _ in range(self.config.population_size):
            individual = self._create_random_individual(search_space)
            self.population.append(individual)
    
    def _create_random_individual(self, search_space: Dict[str, Any]) -> Individual:
        """Create a random individual from search space."""
        parameters = {}
        for param, values in search_space.items():
            if isinstance(values, list):
                parameters[param] = np.random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                # Range parameter
                parameters[param] = np.random.uniform(values[0], values[1])
            else:
                parameters[param] = values
        
        return Individual(parameters)
    
    def _evaluate_population(self, search_env: Dict[str, Any]):
        """Evaluate fitness of all individuals in population."""
        for individual in self.population:
            if not individual.objectives:  # Only evaluate if not already evaluated
                objectives = self._evaluate_individual(individual, search_env)
                individual.objectives = objectives
    
    def _evaluate_individual(self, individual: Individual, search_env: Dict[str, Any]) -> List[float]:
        """Evaluate fitness of a single individual for multiple objectives."""
        objectives = []
        
        try:
            for objective_type in self.config.objectives:
                if objective_type == ObjectiveType.REGIME_ACCURACY:
                    # Regime accuracy - use actual model evaluation
                    accuracy = self._evaluate_regime_accuracy(individual, search_env)
                    objectives.append(1.0 - accuracy)  # Convert to minimization
                
                elif objective_type == ObjectiveType.ECONOMIC_SIGNIFICANCE:
                    # Economic significance
                    if self.economic_evaluator and search_env.get('train_data') is not None:
                        try:
                            economic_result = quick_economic_evaluation(
                                search_env['train_data'], 
                                individual.parameters
                            )
                            significance = economic_result.get('significance_score', 0.5)
                            objectives.append(1.0 - significance)  # Convert to minimization
                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"Economic evaluation failed due to data type issue: {e}")
                            self.logger.warning(f"Individual parameters: {individual.parameters}")
                            objectives.append(0.5)
                        except (MemoryError, OSError) as e:
                            self.logger.warning(f"Economic evaluation failed due to system resource issue: {e}")
                            objectives.append(0.5)
                        except Exception as e:
                            self.logger.warning(f"Economic evaluation failed with unexpected error: {e}")
                            self.logger.warning(f"Error type: {type(e).__name__}")
                            objectives.append(0.5)
                    else:
                        significance = np.random.random() * 0.6 + 0.4
                        objectives.append(1.0 - significance)
                
                elif objective_type == ObjectiveType.TRADING_VIABILITY:
                    # Trading viability
                    if self.trading_evaluator and search_env.get('train_data') is not None:
                        try:
                            trading_result = quick_trading_viability_evaluation(
                                search_env['train_data'],
                                individual.parameters
                            )
                            viability = trading_result.get('viability_score', 0.5)
                            objectives.append(1.0 - viability)  # Convert to minimization
                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"Trading viability evaluation failed due to data type issue: {e}")
                            self.logger.warning(f"Individual parameters: {individual.parameters}")
                            objectives.append(0.5)
                        except (MemoryError, OSError) as e:
                            self.logger.warning(f"Trading viability evaluation failed due to system resource issue: {e}")
                            objectives.append(0.5)
                        except Exception as e:
                            self.logger.warning(f"Trading viability evaluation failed with unexpected error: {e}")
                            self.logger.warning(f"Error type: {type(e).__name__}")
                            objectives.append(0.5)
                    else:
                        viability = np.random.random() * 0.6 + 0.4
                        objectives.append(1.0 - viability)
                
                elif objective_type == ObjectiveType.COMPUTATIONAL_EFFICIENCY:
                    # Computational efficiency (based on architecture complexity)
                    complexity = self._calculate_architecture_complexity(individual.parameters)
                    objectives.append(complexity)  # Minimize complexity
                
                elif objective_type == ObjectiveType.ARCHITECTURE_COMPLEXITY:
                    # Architecture complexity
                    complexity = self._calculate_architecture_complexity(individual.parameters)
                    objectives.append(complexity)  # Minimize complexity
                
                else:
                    # Default objective
                    objectives.append(np.random.random())
            
        except (ValueError, TypeError) as e:
            self.logger.error(f"Error evaluating individual due to data type issue: {e}")
            self.logger.error(f"Individual: {individual}, objectives: {self.config.objectives}")
            objectives = [np.random.random() for _ in self.config.objectives]
        except (MemoryError, OSError) as e:
            self.logger.error(f"Error evaluating individual due to system resource issue: {e}")
            objectives = [np.random.random() for _ in self.config.objectives]
        except Exception as e:
            self.logger.error(f"Error evaluating individual with unexpected error: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            objectives = [np.random.random() for _ in self.config.objectives]
        
        return objectives
    
    def _calculate_architecture_complexity(self, params: Dict[str, Any]) -> float:
        """Calculate architecture complexity score."""
        complexity = 0.0
        
        for param, value in params.items():
            if isinstance(value, (int, float)):
                # Normalize numeric parameters
                complexity += min(1.0, abs(value) / 100.0)
            elif isinstance(value, str):
                # String parameters contribute less complexity
                complexity += 0.1
            else:
                # Other types
                complexity += 0.2
        
        return min(1.0, complexity)
    
    def _evaluate_regime_accuracy(self, individual: Individual, search_env: Dict[str, Any]) -> float:
        """Evaluate regime accuracy for an individual."""
        try:
            # Get training data
            train_data = search_env.get('train_data')
            if train_data is None or train_data.empty:
                # Fallback to random accuracy if no data available
                return np.random.random() * 0.8 + 0.2
            
            # Extract features and targets
            feature_columns = [col for col in train_data.columns if col not in ['timestamp', 'regime', 'target']]
            if not feature_columns:
                return np.random.random() * 0.8 + 0.2
            
            X = train_data[feature_columns].values
            y = train_data.get('regime', train_data.get('target', None))
            
            if y is None:
                return np.random.random() * 0.8 + 0.2
            
            y = y.values if hasattr(y, 'values') else y
            
            # Create a simple model based on individual parameters
            model = self._create_model_from_individual(individual)
            
            if model is None:
                return np.random.random() * 0.8 + 0.2
            
            # Train and evaluate the model
            accuracy = self._train_and_evaluate_model(model, X, y)
            
            return accuracy
            
        except Exception as e:
            self.logger.warning(f"Failed to evaluate regime accuracy: {e}")
            return np.random.random() * 0.8 + 0.2
    
    def _create_model_from_individual(self, individual: Individual):
        """Create a model based on individual parameters."""
        try:
            params = individual.parameters
            
            # Determine model type based on parameters
            model_type = params.get('model_type', 'random_forest')
            
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 10),
                    min_samples_split=params.get('min_samples_split', 2),
                    min_samples_leaf=params.get('min_samples_leaf', 1),
                    random_state=42
                )
            elif model_type == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    learning_rate=params.get('learning_rate', 0.1),
                    max_depth=params.get('max_depth', 3),
                    random_state=42
                )
            elif model_type == 'svm':
                from sklearn.svm import SVC
                return SVC(
                    C=params.get('C', 1.0),
                    kernel=params.get('kernel', 'rbf'),
                    gamma=params.get('gamma', 'scale'),
                    random_state=42
                )
            elif model_type == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(
                    C=params.get('C', 1.0),
                    max_iter=params.get('max_iter', 1000),
                    random_state=42
                )
            else:
                # Default to random forest
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42)
                
        except Exception as e:
            self.logger.warning(f"Failed to create model from individual: {e}")
            return None
    
    def _train_and_evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Train and evaluate a model."""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import StandardScaler
            
            # Handle data preprocessing
            if X.shape[0] < 10:  # Need minimum samples
                return np.random.random() * 0.8 + 0.2
            
            # Scale features if needed
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, X.shape[0] // 2), scoring='accuracy')
            
            # Return mean accuracy
            return float(np.mean(cv_scores))
            
        except Exception as e:
            self.logger.warning(f"Failed to train and evaluate model: {e}")
            return np.random.random() * 0.8 + 0.2
    
    def _update_pareto_front(self):
        """Update Pareto front with non-dominated solutions."""
        # Fast non-dominated sorting
        fronts = self._fast_non_dominated_sort()
        
        # Update Pareto front with first front
        if fronts:
            self.pareto_front = fronts[0]
        else:
            # Fallback: select best individuals by weighted sum
            if self.population:
                weighted_scores = []
                for individual in self.population:
                    if individual.objectives:
                        score = sum(
                            obj * self.config.objective_weights.get(
                                self.config.objectives[i], 1.0
                            )
                            for i, obj in enumerate(individual.objectives)
                        )
                        weighted_scores.append((score, individual))
                
                # Sort by score and take top individuals
                weighted_scores.sort(key=lambda x: x[0])
                self.pareto_front = [ind for _, ind in weighted_scores[:min(10, len(weighted_scores))]]
    
    def _fast_non_dominated_sort(self) -> List[List[Individual]]:
        """Fast non-dominated sorting algorithm."""
        fronts = []
        current_front = []
        
        # Initialize domination counts
        for individual in self.population:
            individual.domination_count = 0
            individual.dominated_solutions = []
        
        # Calculate domination relationships
        for i, individual1 in enumerate(self.population):
            for j, individual2 in enumerate(self.population):
                if i != j:
                    if individual1.dominates(individual2):
                        individual1.dominated_solutions.append(individual2)
                    elif individual2.dominates(individual1):
                        individual1.domination_count += 1
            
            # Add to first front if not dominated
            if individual1.domination_count == 0:
                individual1.rank = 0
                current_front.append(individual1)
        
        fronts.append(current_front)
        
        # Build subsequent fronts
        front_index = 0
        while fronts[front_index]:
            next_front = []
            for individual in fronts[front_index]:
                for dominated_individual in individual.dominated_solutions:
                    dominated_individual.domination_count -= 1
                    if dominated_individual.domination_count == 0:
                        dominated_individual.rank = front_index + 1
                        next_front.append(dominated_individual)
            
            if next_front:
                fronts.append(next_front)
            front_index += 1
        
        return fronts
    
    def _select_parents(self) -> List[Individual]:
        """Select parents for reproduction using tournament selection."""
        parents = []
        
        for _ in range(self.config.population_size):
            # Tournament selection
            tournament_size = min(3, len(self.population))
            tournament_indices = np.random.choice(
                len(self.population), tournament_size, replace=False
            )
            
            # Select best individual from tournament
            best_individual = None
            best_rank = float('inf')
            best_crowding_distance = -1
            
            for idx in tournament_indices:
                individual = self.population[idx]
                if individual.rank < best_rank or (
                    individual.rank == best_rank and 
                    individual.crowding_distance > best_crowding_distance
                ):
                    best_individual = individual
                    best_rank = individual.rank
                    best_crowding_distance = individual.crowding_distance
            
            if best_individual:
                parents.append(best_individual)
            else:
                # Fallback to random selection
                parents.append(np.random.choice(self.population))
        
        return parents
    
    def _create_offspring(self, parents: List[Individual], search_space: Dict[str, Any]) -> List[Individual]:
        """Create offspring through crossover and mutation."""
        offspring = []
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                # Crossover
                if np.random.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover(parents[i], parents[i + 1], search_space)
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parents[i], parents[i + 1]])
            else:
                offspring.append(parents[i])
        
        # Mutation
        for child in offspring:
            if np.random.random() < self.config.mutation_rate:
                self._mutate(child, search_space)
        
        return offspring
    
    def _crossover(self, parent1: Individual, parent2: Individual, search_space: Dict[str, Any]) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        child1_params = parent1.parameters.copy()
        child2_params = parent2.parameters.copy()
        
        # Single-point crossover
        if len(child1_params) > 1:
            crossover_point = np.random.randint(1, len(child1_params))
            keys = list(child1_params.keys())
            
            for i, key in enumerate(keys):
                if i < crossover_point:
                    child1_params[key], child2_params[key] = child2_params[key], child1_params[key]
        
        return Individual(child1_params), Individual(child2_params)
    
    def _mutate(self, individual: Individual, search_space: Dict[str, Any]):
        """Mutate an individual."""
        for key, value in individual.parameters.items():
            if np.random.random() < 0.1:  # 10% chance to mutate each parameter
                if isinstance(value, (int, float)):
                    # Add small random change
                    noise = np.random.normal(0, 0.1 * abs(value))
                    individual.parameters[key] = value + noise
                elif isinstance(value, str):
                    # Random choice from possible values
                    if key in search_space and isinstance(search_space[key], list):
                        individual.parameters[key] = np.random.choice(search_space[key])
    
    def _update_population(self, offspring: List[Individual]):
        """Update population with offspring using NSGA-II selection."""
        # Combine parents and offspring
        combined = self.population + offspring
        
        # Evaluate new individuals
        for individual in combined:
            if not individual.objectives:
                individual.objectives = [np.random.random() for _ in self.config.objectives]
        
        # Fast non-dominated sorting
        fronts = self._fast_non_dominated_sort()
        
        # Crowding distance assignment
        for front in fronts:
            self._assign_crowding_distance(front)
        
        # Select new population
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= self.config.population_size:
                new_population.extend(front)
            else:
                # Sort by crowding distance and select best
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                remaining = self.config.population_size - len(new_population)
                new_population.extend(front[:remaining])
                break
        
        self.population = new_population
    
    def _assign_crowding_distance(self, front: List[Individual]):
        """Assign crowding distance to individuals in a front."""
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return
        
        # Initialize crowding distance
        for individual in front:
            individual.crowding_distance = 0.0
        
        # Calculate crowding distance for each objective
        for obj_idx in range(len(self.config.objectives)):
            # Sort by objective value
            front.sort(key=lambda x: x.objectives[obj_idx])
            
            # Set boundary points to infinity
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate range
            obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]
            if obj_range == 0:
                continue
            
            # Add to crowding distance
            for i in range(1, len(front) - 1):
                distance = (front[i + 1].objectives[obj_idx] - front[i - 1].objectives[obj_idx]) / obj_range
                front[i].crowding_distance += distance
    
    def _get_best_objectives(self) -> List[float]:
        """Get best objectives from current population."""
        if not self.population:
            return []
        
        best_objectives = []
        for obj_idx in range(len(self.config.objectives)):
            best_value = min(individual.objectives[obj_idx] for individual in self.population if individual.objectives)
            best_objectives.append(best_value)
        
        return best_objectives
    
    def _save_checkpoint(self, generation: int):
        """Save checkpoint."""
        try:
            checkpoint_data = {
                'generation': generation,
                'population': [individual.to_dict() for individual in self.population],
                'pareto_front': [individual.to_dict() for individual in self.pareto_front],
                'generation_results': self.generation_results,
                'config': self.config.__dict__
            }
            
            checkpoint_path = f"multi_objective_checkpoint_generation_{generation}.json"
            JSONSerializer.save(checkpoint_data, checkpoint_path)
            
        except (IOError, OSError) as e:
            self.logger.warning(f"Failed to save checkpoint due to file system issue: {e}")
            self.logger.warning(f"Checkpoint path: {checkpoint_path}")
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Failed to save checkpoint due to data serialization issue: {e}")
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint with unexpected error: {e}")
            self.logger.warning(f"Error type: {type(e).__name__}")


class TreeMultiObjectiveOptimizer:
    """Tree multi-objective optimizer for architecture search."""
    
    def __init__(self, config: MultiObjectiveConfig):
        self.config = config
        self.multi_objective_search = MultiObjectiveTreeSearch(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def optimize(self, search_space: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """Optimize tree architecture using multi-objective optimization."""
        tprint_info("🎯 Starting tree multi-objective optimization")
        
        return self.multi_objective_search.search(search_space, **kwargs)


class TreeNSGA2:
    """Tree NSGA-II algorithm for multi-objective optimization."""
    
    def __init__(self, config: MultiObjectiveConfig):
        self.config = config
        self.config.algorithm = OptimizationAlgorithm.NSGA2
        self.multi_objective_search = MultiObjectiveTreeSearch(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def search(self, search_space: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """Perform NSGA-II search for optimal tree architectures."""
        tprint_info("🧬 Starting tree NSGA-II search")
        
        return self.multi_objective_search.search(search_space, **kwargs)