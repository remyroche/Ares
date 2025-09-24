"""
Adaptive Cascade Architecture for CVLSA

This module implements an adaptive cascade architecture with:
1. Dynamic cascade depth based on data complexity and regime characteristics
2. Genetic algorithm optimization for model distribution across cascade levels
3. Cascade pruning mechanisms for inefficient levels
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
import random
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
import concurrent.futures
from threading import Lock

# Import existing utilities
from src.utils.matrix_operations.enhanced_operations import get_enhanced_matrix_operations
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from .cvlsa_architecture import EnhancedCVLSAConfig, MemoryEfficientCVLSA

logger = logging.getLogger(__name__)

@dataclass
class CascadeLevel:
    """Represents a single level in the cascade architecture."""
    level_id: int
    models: List[Any] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    is_active: bool = True
    complexity_score: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    def get_combined_prediction(self, X: np.ndarray) -> np.ndarray:
        """Get combined prediction from all models in this level."""
        if not self.models or not self.is_active:
            return np.zeros(X.shape[0])
        
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model prediction failed in level {self.level_id}: {e}")
                continue
        
        if not predictions:
            return np.zeros(X.shape[0])
        
        # Simple averaging (can be enhanced with weighted averaging)
        return np.mean(predictions, axis=0)

@dataclass
class GeneticOptimizationConfig:
    """Configuration for genetic algorithm optimization."""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    convergence_threshold: float = 0.001
    max_stagnation: int = 20

class AdaptiveCascadeArchitecture:
    """Adaptive cascade architecture with genetic optimization."""
    
    def __init__(self, config: EnhancedCVLSAConfig, 
                 genetic_config: Optional[GeneticOptimizationConfig] = None):
        self.config = config
        self.genetic_config = genetic_config or GeneticOptimizationConfig()
        
        # Cascade levels
        self.cascade_levels: List[CascadeLevel] = []
        self.max_depth = 5
        self.current_depth = 0
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.optimization_results: Dict[str, Any] = {}
        
        # Resource monitoring
        self.resource_monitor = None
        self._init_resource_monitoring()
        
        # Thread safety
        self._lock = Lock()
        
        logger.info("🏗️ Adaptive Cascade Architecture initialized")
    
    def _init_resource_monitoring(self):
        """Initialize resource monitoring."""
        try:
            self.memory_optimizer = get_m1_memory_optimizer()
            self.gpu_manager = get_m1_gpu_manager()
            self.matrix_ops = get_enhanced_matrix_operations()
        except Exception as e:
            logger.warning(f"Resource monitoring not available: {e}")
            self.memory_optimizer = None
            self.gpu_manager = None
            self.matrix_ops = None
    
    def calculate_data_complexity(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate data complexity score."""
        try:
            # Feature complexity
            feature_variance = np.var(X, axis=0)
            feature_complexity = np.mean(feature_variance) / (np.std(feature_variance) + 1e-8)
            
            # Target complexity
            target_variance = np.var(y)
            target_complexity = target_variance / (np.mean(np.abs(y)) + 1e-8)
            
            # Dimensionality complexity
            dimensionality_complexity = X.shape[1] / X.shape[0] if X.shape[0] > 0 else 0
            
            # Interaction complexity (correlation between features)
            correlation_matrix = np.corrcoef(X.T)
            interaction_complexity = np.mean(np.abs(correlation_matrix - np.eye(correlation_matrix.shape[0])))
            
            # Combined complexity score
            complexity_score = (
                0.3 * feature_complexity +
                0.3 * target_complexity +
                0.2 * dimensionality_complexity +
                0.2 * interaction_complexity
            )
            
            return min(complexity_score, 1.0)  # Normalize to [0, 1]
            
        except Exception as e:
            logger.warning(f"Complexity calculation failed: {e}")
            return 0.5  # Default moderate complexity
    
    def calculate_regime_characteristics(self, X: np.ndarray, regimes: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate regime characteristics for adaptive depth."""
        characteristics = {
            'regime_count': 1,
            'regime_stability': 1.0,
            'transition_frequency': 0.0,
            'regime_complexity': 0.5
        }
        
        if regimes is None or len(regimes) == 0:
            return characteristics
        
        try:
            unique_regimes = np.unique(regimes)
            characteristics['regime_count'] = len(unique_regimes)
            
            # Calculate regime stability (how long each regime lasts on average)
            regime_changes = np.diff(regimes, prepend=regimes[0])
            change_indices = np.where(regime_changes != 0)[0]
            
            if len(change_indices) > 1:
                regime_durations = np.diff(change_indices)
                characteristics['regime_stability'] = np.mean(regime_durations) / len(regimes)
                characteristics['transition_frequency'] = len(change_indices) / len(regimes)
            
            # Calculate regime complexity (variance within each regime)
            regime_complexities = []
            for regime in unique_regimes:
                regime_mask = regimes == regime
                regime_X = X[regime_mask]
                if len(regime_X) > 1:
                    regime_var = np.var(regime_X, axis=0)
                    regime_complexities.append(np.mean(regime_var))
            
            if regime_complexities:
                characteristics['regime_complexity'] = np.mean(regime_complexities)
            
        except Exception as e:
            logger.warning(f"Regime characteristics calculation failed: {e}")
        
        return characteristics
    
    def determine_optimal_depth(self, X: np.ndarray, y: np.ndarray, 
                               regimes: Optional[np.ndarray] = None) -> int:
        """Determine optimal cascade depth based on data characteristics."""
        data_complexity = self.calculate_data_complexity(X, y)
        regime_characteristics = self.calculate_regime_characteristics(X, regimes)
        
        # Base depth calculation
        base_depth = max(2, int(data_complexity * 5))  # 2-5 levels based on complexity
        
        # Adjust based on regime characteristics
        regime_factor = 1 + (regime_characteristics['regime_count'] - 1) * 0.2
        stability_factor = 1 + (1 - regime_characteristics['regime_stability']) * 0.5
        complexity_factor = 1 + regime_characteristics['regime_complexity'] * 0.5
        
        optimal_depth = int(base_depth * regime_factor * stability_factor * complexity_factor)
        optimal_depth = min(optimal_depth, self.max_depth)
        
        logger.info(f"📊 Optimal cascade depth determined: {optimal_depth}")
        logger.info(f"   Data complexity: {data_complexity:.3f}")
        logger.info(f"   Regime count: {regime_characteristics['regime_count']}")
        logger.info(f"   Regime stability: {regime_characteristics['regime_stability']:.3f}")
        
        return optimal_depth
    
    def create_model_distribution(self, level_id: int, total_models: int) -> List[str]:
        """Create model distribution for a cascade level using genetic algorithm."""
        # Model types available
        model_types = ['random_forest', 'extra_trees', 'linear', 'svr', 'cvlsa']
        
        # Initialize population
        population = []
        for _ in range(self.genetic_config.population_size):
            individual = {
                'model_distribution': random.choices(model_types, k=total_models),
                'fitness': 0.0
            }
            population.append(individual)
        
        # Genetic algorithm optimization
        best_individual = None
        stagnation_count = 0
        best_fitness = float('-inf')
        
        for generation in range(self.genetic_config.generations):
            # Evaluate fitness
            for individual in population:
                individual['fitness'] = self._evaluate_model_distribution_fitness(
                    individual['model_distribution'], level_id
                )
            
            # Sort by fitness
            population.sort(key=lambda x: x['fitness'], reverse=True)
            
            # Check for convergence
            current_best = population[0]['fitness']
            if current_best > best_fitness:
                best_fitness = current_best
                best_individual = population[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            if stagnation_count >= self.genetic_config.max_stagnation:
                logger.info(f"🔄 Genetic optimization converged at generation {generation}")
                break
            
            # Create next generation
            new_population = []
            
            # Elitism
            for i in range(self.genetic_config.elite_size):
                new_population.append(population[i].copy())
            
            # Crossover and mutation
            while len(new_population) < self.genetic_config.population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        if best_individual is None:
            # Fallback to random distribution
            return random.choices(model_types, k=total_models)
        
        logger.info(f"🧬 Genetic optimization completed for level {level_id}")
        logger.info(f"   Best distribution: {best_individual['model_distribution']}")
        logger.info(f"   Best fitness: {best_individual['fitness']:.4f}")
        
        return best_individual['model_distribution']
    
    def _evaluate_model_distribution_fitness(self, model_distribution: List[str], level_id: int) -> float:
        """Evaluate fitness of a model distribution."""
        # Diversity score (prefer diverse model types)
        unique_types = len(set(model_distribution))
        diversity_score = unique_types / len(model_distribution)
        
        # Complexity balance (balance between simple and complex models)
        simple_models = model_distribution.count('linear') + model_distribution.count('svr')
        complex_models = model_distribution.count('random_forest') + model_distribution.count('cvlsa')
        balance_score = 1.0 - abs(simple_models - complex_models) / len(model_distribution)
        
        # Level appropriateness (early levels should have simpler models)
        if level_id == 0:
            early_level_score = model_distribution.count('linear') / len(model_distribution)
        else:
            early_level_score = 1.0  # No penalty for later levels
        
        # Combined fitness
        fitness = 0.4 * diversity_score + 0.3 * balance_score + 0.3 * early_level_score
        
        return fitness
    
    def _tournament_selection(self, population: List[Dict], tournament_size: int = 3) -> Dict:
        """Tournament selection for genetic algorithm."""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x['fitness'])
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover operation for genetic algorithm."""
        if random.random() > self.genetic_config.crossover_rate:
            return parent1.copy()
        
        child_distribution = []
        for i in range(len(parent1['model_distribution'])):
            if random.random() < 0.5:
                child_distribution.append(parent1['model_distribution'][i])
            else:
                child_distribution.append(parent2['model_distribution'][i])
        
        return {
            'model_distribution': child_distribution,
            'fitness': 0.0
        }
    
    def _mutate(self, individual: Dict) -> Dict:
        """Mutation operation for genetic algorithm."""
        if random.random() > self.genetic_config.mutation_rate:
            return individual
        
        model_types = ['random_forest', 'extra_trees', 'linear', 'svr', 'cvlsa']
        mutated_distribution = individual['model_distribution'].copy()
        
        # Mutate a random position
        mutation_index = random.randint(0, len(mutated_distribution) - 1)
        mutated_distribution[mutation_index] = random.choice(model_types)
        
        return {
            'model_distribution': mutated_distribution,
            'fitness': 0.0
        }
    
    def create_cascade_level(self, level_id: int, X: np.ndarray, y: np.ndarray,
                           previous_predictions: Optional[np.ndarray] = None) -> CascadeLevel:
        """Create a new cascade level with optimized model distribution."""
        logger.info(f"🏗️ Creating cascade level {level_id}")
        
        # Determine number of models for this level
        base_models = 3
        if level_id == 0:
            num_models = base_models
        else:
            num_models = min(base_models + level_id, 8)  # Increase models in later levels
        
        # Get optimized model distribution
        model_distribution = self.create_model_distribution(level_id, num_models)
        
        # Create models
        models = []
        for i, model_type in enumerate(model_distribution):
            try:
                model = self._create_model(model_type, level_id)
                models.append(model)
                logger.info(f"   Created {model_type} model {i+1}/{num_models}")
            except Exception as e:
                logger.warning(f"Failed to create {model_type} model: {e}")
                continue
        
        # Create cascade level
        cascade_level = CascadeLevel(
            level_id=level_id,
            models=models,
            is_active=True
        )
        
        # Train models
        self._train_cascade_level(cascade_level, X, y, previous_predictions)
        
        # Calculate performance metrics
        self._evaluate_cascade_level(cascade_level, X, y)
        
        logger.info(f"✅ Cascade level {level_id} created with {len(models)} models")
        return cascade_level
    
    def _create_model(self, model_type: str, level_id: int) -> Any:
        """Create a specific model type."""
        if model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=50 + level_id * 10,
                max_depth=5 + level_id,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'extra_trees':
            return ExtraTreesRegressor(
                n_estimators=50 + level_id * 10,
                max_depth=5 + level_id,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'linear':
            return Ridge(alpha=1.0 + level_id * 0.1)
        elif model_type == 'svr':
            return SVR(
                kernel='rbf',
                C=1.0 + level_id * 0.5,
                gamma='scale'
            )
        elif model_type == 'cvlsa':
            # Create CVLSA model with level-specific configuration
            cvlsa_config = EnhancedCVLSAConfig(
                input_dim=self.config.input_dim,
                output_dim=self.config.output_dim,
                seq_length=max(50, self.config.seq_length - level_id * 10),
                memory_efficient=True
            )
            return MemoryEfficientCVLSA(cvlsa_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _train_cascade_level(self, cascade_level: CascadeLevel, X: np.ndarray, y: np.ndarray,
                           previous_predictions: Optional[np.ndarray] = None):
        """Train all models in a cascade level."""
        for i, model in enumerate(cascade_level.models):
            try:
                # Prepare training data
                if previous_predictions is not None and len(previous_predictions) > 0:
                    # Include previous predictions as features
                    X_enhanced = np.column_stack([X, previous_predictions])
                else:
                    X_enhanced = X
                
                # Train model
                if hasattr(model, 'fit'):
                    model.fit(X_enhanced, y)
                else:
                    # Handle CVLSA models differently
                    logger.warning(f"Model {i} in level {cascade_level.level_id} doesn't support standard fit")
                
                logger.debug(f"   Trained model {i+1} in level {cascade_level.level_id}")
                
            except Exception as e:
                logger.warning(f"Failed to train model {i} in level {cascade_level.level_id}: {e}")
                continue
    
    def _evaluate_cascade_level(self, cascade_level: CascadeLevel, X: np.ndarray, y: np.ndarray):
        """Evaluate performance of a cascade level."""
        try:
            predictions = cascade_level.get_combined_prediction(X)
            
            # Calculate performance metrics
            mse = mean_squared_error(y, predictions)
            mae = np.mean(np.abs(y - predictions))
            r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
            
            cascade_level.performance_metrics = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'prediction_std': np.std(predictions)
            }
            
            # Calculate complexity score
            cascade_level.complexity_score = self._calculate_level_complexity(cascade_level)
            
            logger.info(f"📊 Level {cascade_level.level_id} performance:")
            logger.info(f"   MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
        except Exception as e:
            logger.warning(f"Failed to evaluate cascade level {cascade_level.level_id}: {e}")
            cascade_level.performance_metrics = {'mse': float('inf'), 'mae': float('inf'), 'r2': -1.0}
    
    def _calculate_level_complexity(self, cascade_level: CascadeLevel) -> float:
        """Calculate complexity score for a cascade level."""
        complexity = 0.0
        
        for model in cascade_level.models:
            if hasattr(model, 'n_estimators'):
                complexity += model.n_estimators / 100.0
            elif hasattr(model, 'C'):
                complexity += model.C / 10.0
            elif hasattr(model, 'alpha'):
                complexity += 1.0 / (model.alpha + 1.0)
            else:
                complexity += 0.5  # Default complexity
        
        return complexity / len(cascade_level.models) if cascade_level.models else 0.0
    
    def prune_inefficient_levels(self, performance_threshold: float = 0.1):
        """Prune cascade levels that don't contribute significantly to performance."""
        logger.info("✂️ Pruning inefficient cascade levels...")
        
        if len(self.cascade_levels) < 2:
            return
        
        # Calculate performance improvement for each level
        level_contributions = []
        for i, level in enumerate(self.cascade_levels):
            if not level.is_active:
                continue
            
            # Calculate contribution (simplified)
            if i == 0:
                contribution = level.performance_metrics.get('r2', 0.0)
            else:
                # Compare with previous level
                prev_level = self.cascade_levels[i-1]
                contribution = level.performance_metrics.get('r2', 0.0) - prev_level.performance_metrics.get('r2', 0.0)
            
            level_contributions.append((i, contribution))
        
        # Sort by contribution
        level_contributions.sort(key=lambda x: x[1])
        
        # Prune levels with low contribution
        pruned_count = 0
        for level_id, contribution in level_contributions:
            if contribution < performance_threshold:
                self.cascade_levels[level_id].is_active = False
                pruned_count += 1
                logger.info(f"   Pruned level {level_id} (contribution: {contribution:.4f})")
        
        logger.info(f"✂️ Pruned {pruned_count} inefficient levels")
    
    def build_adaptive_cascade(self, X: np.ndarray, y: np.ndarray,
                             regimes: Optional[np.ndarray] = None) -> 'AdaptiveCascadeArchitecture':
        """Build the complete adaptive cascade architecture."""
        logger.info("🏗️ Building adaptive cascade architecture...")
        
        start_time = time.time()
        
        # Determine optimal depth
        optimal_depth = self.determine_optimal_depth(X, y, regimes)
        
        # Create cascade levels
        self.cascade_levels = []
        previous_predictions = None
        
        for level_id in range(optimal_depth):
            try:
                cascade_level = self.create_cascade_level(level_id, X, y, previous_predictions)
                self.cascade_levels.append(cascade_level)
                
                # Update previous predictions for next level
                previous_predictions = cascade_level.get_combined_prediction(X)
                
                # Resource monitoring
                if self.memory_optimizer:
                    memory_stats = self.memory_optimizer.get_memory_stats()
                    cascade_level.resource_usage = {
                        'memory_mb': memory_stats.get('used_memory', 0) / (1024 * 1024),
                        'memory_percent': memory_stats.get('memory_percent', 0)
                    }
                
            except Exception as e:
                logger.error(f"Failed to create cascade level {level_id}: {e}")
                break
        
        # Prune inefficient levels
        self.prune_inefficient_levels()
        
        # Store performance history
        self.performance_history.append({
            'timestamp': time.time(),
            'depth': len(self.cascade_levels),
            'active_levels': sum(1 for level in self.cascade_levels if level.is_active),
            'total_models': sum(len(level.models) for level in self.cascade_levels if level.is_active),
            'build_time': time.time() - start_time
        })
        
        logger.info(f"✅ Adaptive cascade built in {time.time() - start_time:.2f}s")
        logger.info(f"   Total levels: {len(self.cascade_levels)}")
        logger.info(f"   Active levels: {sum(1 for level in self.cascade_levels if level.is_active)}")
        logger.info(f"   Total models: {sum(len(level.models) for level in self.cascade_levels if level.is_active)}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the adaptive cascade."""
        if not self.cascade_levels:
            raise ValueError("Cascade not built. Call build_adaptive_cascade first.")
        
        predictions = None
        current_features = X.copy()
        
        for level in self.cascade_levels:
            if not level.is_active:
                continue
            
            try:
                # Get predictions from this level
                level_predictions = level.get_combined_prediction(current_features)
                
                # Combine with previous predictions
                if predictions is None:
                    predictions = level_predictions
                else:
                    # Weighted combination (can be enhanced)
                    alpha = 0.7  # Weight for current level
                    predictions = alpha * level_predictions + (1 - alpha) * predictions
                
                # Add predictions as features for next level
                current_features = np.column_stack([X, predictions])
                
            except Exception as e:
                logger.warning(f"Prediction failed at level {level.level_id}: {e}")
                continue
        
        if predictions is None:
            logger.warning("No valid predictions from cascade")
            return np.zeros(X.shape[0])
        
        return predictions
    
    def get_cascade_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about the cascade architecture."""
        active_levels = [level for level in self.cascade_levels if level.is_active]
        
        analytics = {
            'total_levels': len(self.cascade_levels),
            'active_levels': len(active_levels),
            'total_models': sum(len(level.models) for level in active_levels),
            'performance_by_level': [
                {
                    'level_id': level.level_id,
                    'models_count': len(level.models),
                    'performance': level.performance_metrics,
                    'complexity_score': level.complexity_score,
                    'resource_usage': level.resource_usage
                }
                for level in active_levels
            ],
            'optimization_results': self.optimization_results,
            'performance_history': self.performance_history
        }
        
        return analytics


# Factory functions
def create_adaptive_cascade(config: EnhancedCVLSAConfig,
                          genetic_config: Optional[GeneticOptimizationConfig] = None) -> AdaptiveCascadeArchitecture:
    """Create adaptive cascade architecture."""
    return AdaptiveCascadeArchitecture(config, genetic_config)