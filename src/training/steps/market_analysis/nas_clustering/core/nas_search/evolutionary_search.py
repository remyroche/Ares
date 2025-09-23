"""
Evolutionary Neural Architecture Search

This module implements evolutionary algorithms for neural architecture search,
specifically optimized for regime detection in financial time series.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging
import time
import random
from abc import ABC, abstractmethod
import copy

# Essential imports only

from .search_space import (
    SearchSpace, LayerConfig, ConnectionConfig, ArchitectureConstraints,
    LayerType, ActivationFunction, ConnectionType
)

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureIndividual:
    """Individual architecture in the evolutionary population."""
    layers: List[LayerConfig]
    connections: List[ConnectionConfig]
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    evaluation_time: float = 0.0
    parameters_count: int = 0
    
    def __post_init__(self):
        """Calculate derived properties after initialization."""
        self.parameters_count = self._estimate_parameters()
    
    def _estimate_parameters(self) -> int:
        """Estimate the number of parameters in this architecture."""
        try:
            total_params = 0
            for layer in self.layers:
                if layer.layer_type == LayerType.DENSE:
                    total_params += layer.units * 64  # Rough estimate
                elif layer.layer_type in [LayerType.LSTM, LayerType.GRU]:
                    total_params += 4 * layer.units * layer.units
                elif layer.layer_type == LayerType.CONV1D:
                    if layer.kernel_size:
                        total_params += layer.kernel_size * 32 * layer.units
                elif layer.layer_type in [LayerType.ATTENTION, LayerType.MULTI_HEAD_ATTENTION]:
                    total_params += layer.units * layer.units
            return total_params
        except Exception:
            return 0
    
    def copy(self) -> 'ArchitectureIndividual':
        """Create a deep copy of this individual."""
        return ArchitectureIndividual(
            layers=copy.deepcopy(self.layers),
            connections=copy.deepcopy(self.connections),
            fitness_score=self.fitness_score,
            generation=self.generation,
            parent_ids=copy.deepcopy(self.parent_ids),
            mutation_history=copy.deepcopy(self.mutation_history),
            evaluation_time=self.evaluation_time,
            parameters_count=self.parameters_count
        )


@dataclass
class ArchitecturePopulation:
    """Population of architecture individuals for evolutionary search."""
    individuals: List[ArchitectureIndividual]
    generation: int = 0
    best_fitness: float = 0.0
    average_fitness: float = 0.0
    diversity_score: float = 0.0
    
    def __post_init__(self):
        """Calculate population statistics after initialization."""
        self._update_statistics()
    
    def _update_statistics(self):
        """Update population statistics."""
        try:
            if not self.individuals:
                return
            
            fitness_scores = [ind.fitness_score for ind in self.individuals]
            self.best_fitness = max(fitness_scores)
            self.average_fitness = np.mean(fitness_scores)
            self.diversity_score = self._calculate_diversity()
            
        except Exception as e:
            logger.warning(f"Statistics update failed: {e}")
    
    def _calculate_diversity(self) -> float:
        """Calculate diversity score of the population."""
        try:
            if len(self.individuals) < 2:
                return 0.0
            
            # Calculate pairwise diversity based on layer configurations
            diversity_scores = []
            for i in range(len(self.individuals)):
                for j in range(i + 1, len(self.individuals)):
                    diversity = self._architecture_diversity(
                        self.individuals[i], self.individuals[j]
                    )
                    diversity_scores.append(diversity)
            
            return np.mean(diversity_scores) if diversity_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Diversity calculation failed: {e}")
            return 0.0
    
    def _architecture_diversity(self, ind1: ArchitectureIndividual, ind2: ArchitectureIndividual) -> float:
        """Calculate diversity between two architectures."""
        try:
            # Compare layer types
            types1 = [layer.layer_type for layer in ind1.layers]
            types2 = [layer.layer_type for layer in ind2.layers]
            
            # Calculate Jaccard similarity for layer types
            set1, set2 = set(types1), set(types2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            jaccard_similarity = intersection / union if union > 0 else 0.0
            diversity = 1.0 - jaccard_similarity
            
            return diversity
            
        except Exception as e:
            logger.warning(f"Architecture diversity calculation failed: {e}")
            return 0.5  # Default diversity score
    
    def get_best_individuals(self, n: int) -> List[ArchitectureIndividual]:
        """Get the best n individuals from the population."""
        try:
            sorted_individuals = sorted(
                self.individuals, 
                key=lambda x: x.fitness_score, 
                reverse=True
            )
            return sorted_individuals[:n]
        except Exception as e:
            logger.warning(f"Best individuals selection failed: {e}")
            return []
    
    def get_diverse_individuals(self, n: int) -> List[ArchitectureIndividual]:
        """Get diverse individuals using tournament selection with diversity."""
        try:
            if len(self.individuals) <= n:
                return self.individuals.copy()
            
            selected = []
            available = list(range(len(self.individuals)))
            
            while len(selected) < n and available:
                # Tournament selection
                tournament_size = min(3, len(available))
                tournament_indices = random.sample(available, tournament_size)
                tournament_individuals = [self.individuals[i] for i in tournament_indices]
                
                # Select best from tournament
                winner = max(tournament_individuals, key=lambda x: x.fitness_score)
                winner_index = self.individuals.index(winner)
                
                selected.append(winner)
                available.remove(winner_index)
            
            return selected
            
        except Exception as e:
            logger.warning(f"Diverse individuals selection failed: {e}")
            return self.get_best_individuals(n)


class FitnessEvaluator(ABC):
    """Abstract base class for fitness evaluation."""
    
    @abstractmethod
    def evaluate(self, individual: ArchitectureIndividual, 
                 data: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate fitness of an architecture individual."""
        pass


class RegimeDetectionFitnessEvaluator(FitnessEvaluator):
    """Essential fitness evaluator for NAS."""
    
    def __init__(self):
        """Initialize fitness evaluator."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate(self, individual: ArchitectureIndividual, 
                 data: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate fitness of an architecture for NAS."""
        try:
            start_time = time.time()
            
            # Evaluate architecture performance
            fitness_score = self._evaluate_architecture_performance(individual, data, labels)
            
            # Add complexity penalty
            complexity_penalty = self._calculate_complexity_penalty(individual)
            
            # Combined fitness score
            final_score = fitness_score - complexity_penalty
            
            # Record evaluation time
            individual.evaluation_time = time.time() - start_time
            
            return max(0.0, final_score)  # Ensure non-negative fitness
            
        except Exception as e:
            self.logger.warning(f"Fitness evaluation failed: {e}")
            return 0.0
    
    def _evaluate_architecture_performance(self, individual: ArchitectureIndividual,
                                         data: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate architecture performance based on structure."""
        try:
            score = 0.0
            
            # Reward temporal layers
            temporal_layers = sum(1 for layer in individual.layers 
                                if layer.layer_type in [LayerType.LSTM, LayerType.GRU, LayerType.CONV1D])
            score += temporal_layers * 0.1
            
            # Reward appropriate depth
            layer_count = len(individual.layers)
            if 3 <= layer_count <= 6:
                score += 0.2
            elif layer_count > 6:
                score += 0.1
            
            # Reward skip connections
            skip_connections = sum(1 for conn in individual.connections 
                                 if conn.connection_type != ConnectionType.SEQUENTIAL)
            score += skip_connections * 0.05
            
            # Reward batch normalization
            batch_norm_layers = sum(1 for layer in individual.layers if layer.batch_norm)
            score += batch_norm_layers * 0.05
            
            return min(1.0, score)  # Cap at 1.0
            
        except Exception as e:
            self.logger.warning(f"Architecture performance evaluation failed: {e}")
            return 0.0
    
    def _calculate_complexity_penalty(self, individual: ArchitectureIndividual) -> float:
        """Calculate penalty for overly complex architectures."""
        try:
            penalty = 0.0
            
            # Parameter count penalty
            param_count = individual.parameters_count
            if param_count > 500000:  # 500K parameters
                penalty += (param_count - 500000) / 1000000  # 0.1 penalty per 100K extra params
            
            # Layer count penalty
            layer_count = len(individual.layers)
            if layer_count > 8:
                penalty += (layer_count - 8) * 0.05
            
            # Connection complexity penalty
            connection_count = len(individual.connections)
            if connection_count > 15:
                penalty += (connection_count - 15) * 0.01
            
            return penalty
            
        except Exception as e:
            self.logger.warning(f"Complexity penalty calculation failed: {e}")
            return 0.0
    


class GeneticAlgorithm:
    """Essential genetic algorithm for neural architecture search."""
    
    def __init__(self, search_space: SearchSpace, fitness_evaluator: FitnessEvaluator,
                 population_size: int = 50, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8, elite_size: int = 5):
        """Initialize genetic algorithm."""
        self.search_space = search_space
        self.fitness_evaluator = fitness_evaluator
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Statistics
        self.generation_stats = []
        self.best_individual_history = []
    
    def initialize_population(self) -> ArchitecturePopulation:
        """Initialize random population of architectures."""
        try:
            individuals = []
            
            for i in range(self.population_size):
                individual = self._create_random_individual()
                individual.generation = 0
                individuals.append(individual)
            
            population = ArchitecturePopulation(individuals=individuals, generation=0)
            self.logger.info(f"✅ Initialized population of {len(individuals)} individuals")
            
            return population
            
        except Exception as e:
            self.logger.error(f"Population initialization failed: {e}")
            raise
    
    def _create_random_individual(self) -> ArchitectureIndividual:
        """Create a random architecture individual."""
        try:
            # Random number of layers
            n_layers = np.random.randint(
                self.search_space.constraints.min_layers,
                min(self.search_space.constraints.max_layers + 1, 8)  # Cap at 8 for initialization
            )
            
            layers = []
            for i in range(n_layers):
                layer_config = self.search_space.get_random_layer_config()
                layers.append(layer_config)
            
            # Create sequential connections
            connections = []
            for i in range(n_layers - 1):
                conn_config = self.search_space.get_random_connection_config(i, i + 1)
                conn_config.connection_type = ConnectionType.SEQUENTIAL  # Start with sequential
                connections.append(conn_config)
            
            # Add some random skip connections
            n_skip = np.random.randint(0, min(2, n_layers))
            for _ in range(n_skip):
                from_layer = np.random.randint(0, n_layers - 1)
                to_layer = np.random.randint(from_layer + 1, n_layers)
                if from_layer != to_layer - 1:  # Not sequential
                    skip_conn = self.search_space.get_random_connection_config(from_layer, to_layer)
                    connections.append(skip_conn)
            
            return ArchitectureIndividual(
                layers=layers,
                connections=connections,
                generation=0
            )
            
        except Exception as e:
            self.logger.warning(f"Random individual creation failed: {e}")
            # Return a simple default individual
            return ArchitectureIndividual(
                layers=[
                    LayerConfig(LayerType.DENSE, ActivationFunction.RELU, 64, 0.2, True),
                    LayerConfig(LayerType.DENSE, ActivationFunction.RELU, 32, 0.1, True)
                ],
                connections=[
                    ConnectionConfig(ConnectionType.SEQUENTIAL, 0, 1)
                ],
                generation=0
            )
    
    def evolve_generation(self, population: ArchitecturePopulation, 
                         data: np.ndarray, labels: np.ndarray) -> ArchitecturePopulation:
        """Evolve one generation of the population."""
        try:
            start_time = time.time()
            
            # Evaluate fitness for all individuals
            self.logger.info(f"Evaluating fitness for generation {population.generation}")
            for individual in population.individuals:
                if individual.fitness_score == 0.0:  # Only evaluate if not already evaluated
                    individual.fitness_score = self.fitness_evaluator.evaluate(individual, data, labels)
            
            # Update population statistics
            population._update_statistics()
            
            # Record generation statistics
            generation_stat = {
                'generation': population.generation,
                'best_fitness': population.best_fitness,
                'average_fitness': population.average_fitness,
                'diversity_score': population.diversity_score,
                'evaluation_time': time.time() - start_time
            }
            self.generation_stats.append(generation_stat)
            
            # Select parents for next generation
            parents = self._select_parents(population)
            
            # Create new population through crossover and mutation
            new_individuals = []
            
            # Elitism: keep best individuals
            elite = population.get_best_individuals(self.elite_size)
            for individual in elite:
                elite_copy = individual.copy()
                elite_copy.generation = population.generation + 1
                new_individuals.append(elite_copy)
            
            # Generate offspring
            while len(new_individuals) < self.population_size:
                if np.random.random() < self.crossover_rate:
                    # Crossover
                    parent1, parent2 = random.sample(parents, 2)
                    offspring = self._crossover(parent1, parent2)
                else:
                    # Mutation only
                    parent = random.choice(parents)
                    offspring = self._mutate(parent)
                
                offspring.generation = population.generation + 1
                new_individuals.append(offspring)
            
            # Create new population
            new_population = ArchitecturePopulation(
                individuals=new_individuals,
                generation=population.generation + 1
            )
            
            # Record best individual
            best_individual = max(new_population.individuals, key=lambda x: x.fitness_score)
            self.best_individual_history.append(best_individual.copy())
            
            self.logger.info(f"✅ Generation {population.generation + 1} evolved")
            self.logger.info(f"   Best fitness: {new_population.best_fitness:.4f}")
            self.logger.info(f"   Average fitness: {new_population.average_fitness:.4f}")
            self.logger.info(f"   Diversity: {new_population.diversity_score:.4f}")
            
            return new_population
            
        except Exception as e:
            self.logger.error(f"Generation evolution failed: {e}")
            raise
    
    def _select_parents(self, population: ArchitecturePopulation) -> List[ArchitectureIndividual]:
        """Select parents for reproduction using tournament selection."""
        try:
            parents = []
            tournament_size = 3
            
            while len(parents) < self.population_size - self.elite_size:
                # Tournament selection
                tournament = random.sample(population.individuals, 
                                         min(tournament_size, len(population.individuals)))
                winner = max(tournament, key=lambda x: x.fitness_score)
                parents.append(winner)
            
            return parents
            
        except Exception as e:
            self.logger.warning(f"Parent selection failed: {e}")
            return population.get_best_individuals(self.population_size - self.elite_size)
    
    def _crossover(self, parent1: ArchitectureIndividual, parent2: ArchitectureIndividual) -> ArchitectureIndividual:
        """Perform crossover between two parent individuals."""
        try:
            # Simple uniform crossover for layers
            child_layers = []
            max_layers = max(len(parent1.layers), len(parent2.layers))
            
            for i in range(max_layers):
                if i < len(parent1.layers) and i < len(parent2.layers):
                    # Both parents have this layer position
                    if np.random.random() < 0.5:
                        child_layers.append(copy.deepcopy(parent1.layers[i]))
                    else:
                        child_layers.append(copy.deepcopy(parent2.layers[i]))
                elif i < len(parent1.layers):
                    child_layers.append(copy.deepcopy(parent1.layers[i]))
                else:
                    child_layers.append(copy.deepcopy(parent2.layers[i]))
            
            # Crossover for connections (simplified)
            child_connections = []
            if parent1.connections and parent2.connections:
                # Take connections from both parents
                all_connections = parent1.connections + parent2.connections
                # Remove duplicates and validate
                seen_connections = set()
                for conn in all_connections:
                    conn_key = (conn.from_layer, conn.to_layer, conn.connection_type)
                    if conn_key not in seen_connections:
                        seen_connections.add(conn_key)
                        if (conn.from_layer < len(child_layers) and 
                            conn.to_layer < len(child_layers) and 
                            conn.from_layer != conn.to_layer):
                            child_connections.append(copy.deepcopy(conn))
            
            # Ensure at least sequential connections
            if not child_connections:
                for i in range(len(child_layers) - 1):
                    child_connections.append(
                        ConnectionConfig(ConnectionType.SEQUENTIAL, i, i + 1)
                    )
            
            return ArchitectureIndividual(
                layers=child_layers,
                connections=child_connections,
                parent_ids=[id(parent1), id(parent2)],
                mutation_history=['crossover']
            )
            
        except Exception as e:
            self.logger.warning(f"Crossover failed: {e}")
            # Return a copy of the first parent
            return parent1.copy()
    
    def _mutate(self, individual: ArchitectureIndividual) -> ArchitectureIndividual:
        """Apply mutations to an individual."""
        try:
            mutated = individual.copy()
            mutations = []
            
            # Layer mutations
            if np.random.random() < self.mutation_rate:
                mutation_type = np.random.choice(['add_layer', 'remove_layer', 'modify_layer'])
                
                if mutation_type == 'add_layer' and len(mutated.layers) < self.search_space.constraints.max_layers:
                    # Add a new layer
                    new_layer = self.search_space.get_random_layer_config()
                    insert_pos = np.random.randint(0, len(mutated.layers) + 1)
                    mutated.layers.insert(insert_pos, new_layer)
                    
                    # Update connection indices
                    for conn in mutated.connections:
                        if conn.from_layer >= insert_pos:
                            conn.from_layer += 1
                        if conn.to_layer >= insert_pos:
                            conn.to_layer += 1
                    
                    mutations.append('add_layer')
                
                elif mutation_type == 'remove_layer' and len(mutated.layers) > self.search_space.constraints.min_layers:
                    # Remove a layer
                    remove_pos = np.random.randint(0, len(mutated.layers))
                    mutated.layers.pop(remove_pos)
                    
                    # Update connection indices and remove invalid connections
                    valid_connections = []
                    for conn in mutated.connections:
                        if conn.from_layer == remove_pos or conn.to_layer == remove_pos:
                            continue  # Remove connections involving the deleted layer
                        
                        if conn.from_layer > remove_pos:
                            conn.from_layer -= 1
                        if conn.to_layer > remove_pos:
                            conn.to_layer -= 1
                        
                        valid_connections.append(conn)
                    
                    mutated.connections = valid_connections
                    mutations.append('remove_layer')
                
                elif mutation_type == 'modify_layer':
                    # Modify a random layer
                    layer_idx = np.random.randint(0, len(mutated.layers))
                    layer = mutated.layers[layer_idx]
                    
                    # Modify layer properties
                    if np.random.random() < 0.3:
                        layer.activation = np.random.choice(self.search_space.available_activations)
                    if np.random.random() < 0.3:
                        layer.units = np.random.choice(self.search_space.layer_size_options)
                    if np.random.random() < 0.3:
                        layer.dropout_rate = np.random.choice(self.search_space.dropout_options)
                    if np.random.random() < 0.3:
                        layer.batch_norm = np.random.choice([True, False])
                    
                    mutations.append('modify_layer')
            
            # Connection mutations
            if np.random.random() < self.mutation_rate:
                mutation_type = np.random.choice(['add_connection', 'remove_connection', 'modify_connection'])
                
                if mutation_type == 'add_connection':
                    # Add a random skip connection
                    from_layer = np.random.randint(0, len(mutated.layers) - 1)
                    to_layer = np.random.randint(from_layer + 1, len(mutated.layers))
                    
                    # Check if connection already exists
                    exists = any(conn.from_layer == from_layer and conn.to_layer == to_layer 
                               for conn in mutated.connections)
                    
                    if not exists:
                        new_conn = self.search_space.get_random_connection_config(from_layer, to_layer)
                        mutated.connections.append(new_conn)
                        mutations.append('add_connection')
                
                elif mutation_type == 'remove_connection' and len(mutated.connections) > len(mutated.layers) - 1:
                    # Remove a non-sequential connection
                    non_seq_connections = [i for i, conn in enumerate(mutated.connections)
                                         if conn.connection_type != ConnectionType.SEQUENTIAL]
                    if non_seq_connections:
                        remove_idx = np.random.choice(non_seq_connections)
                        mutated.connections.pop(remove_idx)
                        mutations.append('remove_connection')
                
                elif mutation_type == 'modify_connection':
                    # Modify a random connection
                    if mutated.connections:
                        conn_idx = np.random.randint(0, len(mutated.connections))
                        conn = mutated.connections[conn_idx]
                        conn.connection_type = np.random.choice(self.search_space.available_connections)
                        mutations.append('modify_connection')
            
            mutated.mutation_history.extend(mutations)
            return mutated
            
        except Exception as e:
            self.logger.warning(f"Mutation failed: {e}")
            return individual.copy()


class EvolutionaryArchitectureSearch:
    """Essential evolutionary architecture search class."""
    
    def __init__(self, search_space: SearchSpace, 
                 population_size: int = 50, generations: int = 100,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        """Initialize evolutionary architecture search."""
        self.search_space = search_space
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Initialize fitness evaluator
        self.fitness_evaluator = RegimeDetectionFitnessEvaluator()
        
        # Initialize genetic algorithm
        self.genetic_algorithm = GeneticAlgorithm(
            search_space=search_space,
            fitness_evaluator=self.fitness_evaluator,
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Search history
        self.search_history = []
        self.best_architecture = None
    
    def search(self, data: np.ndarray, labels: np.ndarray) -> ArchitectureIndividual:
        """Perform evolutionary architecture search."""
        try:
            self.logger.info(f"🚀 Starting evolutionary architecture search")
            self.logger.info(f"   Population size: {self.population_size}")
            self.logger.info(f"   Generations: {self.generations}")
            self.logger.info(f"   Data shape: {data.shape}")
            
            # Initialize population
            population = self.genetic_algorithm.initialize_population()
            
            # Evolve for specified generations
            for generation in range(self.generations):
                self.logger.info(f"🔄 Evolving generation {generation + 1}/{self.generations}")
                
                # Evolve generation
                population = self.genetic_algorithm.evolve_generation(population, data, labels)
                
                # Record search progress
                search_state = {
                    'generation': population.generation,
                    'best_fitness': population.best_fitness,
                    'average_fitness': population.average_fitness,
                    'diversity_score': population.diversity_score,
                    'best_architecture': population.get_best_individuals(1)[0].copy()
                }
                self.search_history.append(search_state)
                
                # Early stopping if fitness plateaus
                if len(self.search_history) > 10:
                    recent_improvements = [
                        self.search_history[i]['best_fitness'] - self.search_history[i-1]['best_fitness']
                        for i in range(-10, 0)
                    ]
                    if all(imp < 0.001 for imp in recent_improvements):
                        self.logger.info("🛑 Early stopping due to fitness plateau")
                        break
            
            # Get best architecture
            self.best_architecture = population.get_best_individuals(1)[0]
            
            self.logger.info(f"✅ Evolutionary search completed")
            self.logger.info(f"   Best fitness: {self.best_architecture.fitness_score:.4f}")
            self.logger.info(f"   Best architecture layers: {len(self.best_architecture.layers)}")
            self.logger.info(f"   Best architecture connections: {len(self.best_architecture.connections)}")
            
            return self.best_architecture
            
        except Exception as e:
            self.logger.error(f"Evolutionary search failed: {e}")
            raise
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        try:
            if not self.search_history:
                return {}
            
            final_generation = self.search_history[-1]
            first_generation = self.search_history[0]
            
            return {
                'total_generations': len(self.search_history),
                'final_best_fitness': final_generation['best_fitness'],
                'initial_best_fitness': first_generation['best_fitness'],
                'fitness_improvement': final_generation['best_fitness'] - first_generation['best_fitness'],
                'final_average_fitness': final_generation['average_fitness'],
                'final_diversity_score': final_generation['diversity_score'],
                'best_architecture': {
                    'layers': len(self.best_architecture.layers) if self.best_architecture else 0,
                    'connections': len(self.best_architecture.connections) if self.best_architecture else 0,
                    'parameters': self.best_architecture.parameters_count if self.best_architecture else 0,
                    'fitness': self.best_architecture.fitness_score if self.best_architecture else 0.0
                },
                'generation_stats': self.genetic_algorithm.generation_stats
            }
            
        except Exception as e:
            self.logger.warning(f"Search statistics calculation failed: {e}")
            return {}