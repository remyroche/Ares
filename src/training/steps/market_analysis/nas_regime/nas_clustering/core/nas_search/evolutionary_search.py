"""
Evolutionary Architecture Search

Evolutionary algorithm for neural architecture search.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class Architecture:
    """Neural architecture representation."""
    layers: List[Dict[str, Any]]
    parameters_count: int
    fitness_score: float
    complexity_score: float
    efficiency_score: float

class EvolutionaryArchitectureSearch:
    """
    Evolutionary Architecture Search for neural architecture optimization.
    """
    
    def __init__(self, population_size: int = 50, generations: int = 100):
        """
        Initialize Evolutionary Architecture Search.
        
        Args:
            population_size: Size of the population
            generations: Number of generations to evolve
        """
        self.population_size = population_size
        self.generations = generations
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Evolution parameters
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
        
        self.logger.info(f"Evolutionary Architecture Search initialized with population_size={population_size}")
    
    def search(self, search_space: Dict[str, Any], data_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Perform evolutionary architecture search.
        
        Args:
            search_space: Architecture search space definition
            data_shape: Shape of input data (n_samples, n_features)
            
        Returns:
            Dictionary containing search results
        """
        start_time = time.time()
        self.logger.info(f"Starting evolutionary architecture search for data shape {data_shape}")
        
        try:
            # Generate initial population
            population = self._generate_initial_population(search_space, data_shape)
            
            # Evolution loop
            for generation in range(self.generations):
                # Evaluate population
                self._evaluate_population(population, data_shape)
                
                # Select parents
                parents = self._select_parents(population)
                
                # Create offspring
                offspring = self._create_offspring(parents, search_space, data_shape)
                
                # Combine population
                population = population + offspring
                
                # Select next generation
                population = self._select_next_generation(population)
                
                self.logger.debug(f"Generation {generation + 1}: Best fitness = {max(ind.fitness_score for ind in population):.3f}")
            
            # Final evaluation
            self._evaluate_population(population, data_shape)
            
            # Select best architecture
            best_architecture = max(population, key=lambda x: x.fitness_score)
            
            execution_time = time.time() - start_time
            
            result = {
                'success': True,
                'best_architecture': best_architecture,
                'search_history': [arch.fitness_score for arch in population],
                'execution_time': execution_time,
                'final_population_size': len(population)
            }
            
            self.logger.info(f"Evolutionary search completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Evolutionary search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _generate_initial_population(self, search_space: Dict[str, Any], data_shape: Tuple[int, int]) -> List[Architecture]:
        """Generate initial population of architectures."""
        try:
            population = []
            
            for _ in range(self.population_size):
                architecture = self._create_random_architecture(search_space, data_shape)
                population.append(architecture)
            
            return population
            
        except Exception as e:
            self.logger.warning(f"Initial population generation failed: {e}")
            return []
    
    def _create_random_architecture(self, search_space: Dict[str, Any], data_shape: Tuple[int, int]) -> Architecture:
        """Create a random architecture."""
        try:
            n_features = data_shape[1]
            
            # Random number of layers
            n_layers = np.random.randint(2, 6)
            
            layers = []
            current_size = n_features
            
            for i in range(n_layers):
                if i == n_layers - 1:
                    # Output layer
                    layer = {
                        'type': 'linear',
                        'input_size': current_size,
                        'output_size': 3,  # Default number of regimes
                        'activation': 'none'
                    }
                else:
                    # Hidden layer
                    layer_size = np.random.randint(32, 256)
                    layer = {
                        'type': 'linear',
                        'input_size': current_size,
                        'output_size': layer_size,
                        'activation': np.random.choice(['relu', 'tanh', 'sigmoid']),
                        'dropout': np.random.uniform(0.0, 0.5)
                    }
                    current_size = layer_size
                
                layers.append(layer)
            
            # Calculate parameter count
            parameters_count = self._calculate_parameters(layers)
            
            architecture = Architecture(
                layers=layers,
                parameters_count=parameters_count,
                fitness_score=0.0,
                complexity_score=0.0,
                efficiency_score=0.0
            )
            
            return architecture
            
        except Exception as e:
            self.logger.warning(f"Random architecture creation failed: {e}")
            # Return simple fallback architecture
            return Architecture(
                layers=[
                    {'type': 'linear', 'input_size': data_shape[1], 'output_size': 64, 'activation': 'relu'},
                    {'type': 'linear', 'input_size': 64, 'output_size': 3, 'activation': 'none'}
                ],
                parameters_count=1000,
                fitness_score=0.0,
                complexity_score=0.5,
                efficiency_score=0.5
            )
    
    def _calculate_parameters(self, layers: List[Dict[str, Any]]) -> int:
        """Calculate total number of parameters in architecture."""
        try:
            total_params = 0
            
            for layer in layers:
                if layer['type'] == 'linear':
                    input_size = layer['input_size']
                    output_size = layer['output_size']
                    total_params += input_size * output_size + output_size  # weights + biases
            
            return total_params
            
        except Exception as e:
            self.logger.warning(f"Parameter calculation failed: {e}")
            return 1000
    
    def _evaluate_population(self, population: List[Architecture], data_shape: Tuple[int, int]):
        """Evaluate population fitness."""
        try:
            for architecture in population:
                # Calculate fitness based on architecture characteristics
                architecture.fitness_score = self._calculate_fitness(architecture)
                architecture.complexity_score = self._calculate_complexity(architecture)
                architecture.efficiency_score = self._calculate_efficiency(architecture)
                
        except Exception as e:
            self.logger.warning(f"Population evaluation failed: {e}")
    
    def _calculate_fitness(self, architecture: Architecture) -> float:
        """Calculate architecture fitness."""
        try:
            # Simple fitness based on parameter count and layer structure
            param_score = 1.0 / (1.0 + architecture.parameters_count / 10000)  # Prefer fewer parameters
            
            layer_score = 1.0 / (1.0 + len(architecture.layers) / 10)  # Prefer simpler architectures
            
            # Bonus for balanced architectures
            balance_score = 1.0
            if len(architecture.layers) > 1:
                layer_sizes = [layer.get('output_size', 0) for layer in architecture.layers[:-1]]
                if layer_sizes:
                    size_variance = np.var(layer_sizes)
                    balance_score = 1.0 / (1.0 + size_variance / 10000)
            
            fitness = (param_score * 0.4 + layer_score * 0.3 + balance_score * 0.3)
            return min(1.0, max(0.0, fitness))
            
        except Exception as e:
            self.logger.warning(f"Fitness calculation failed: {e}")
            return 0.5
    
    def _calculate_complexity(self, architecture: Architecture) -> float:
        """Calculate architecture complexity."""
        try:
            # Complexity based on number of layers and parameters
            layer_complexity = len(architecture.layers) / 10.0
            param_complexity = architecture.parameters_count / 10000.0
            
            complexity = (layer_complexity + param_complexity) / 2.0
            return min(1.0, complexity)
            
        except Exception as e:
            self.logger.warning(f"Complexity calculation failed: {e}")
            return 0.5
    
    def _calculate_efficiency(self, architecture: Architecture) -> float:
        """Calculate architecture efficiency."""
        try:
            # Efficiency inversely related to complexity
            complexity = self._calculate_complexity(architecture)
            efficiency = 1.0 - complexity
            
            # Bonus for architectures with good parameter utilization
            if len(architecture.layers) > 1:
                param_per_layer = architecture.parameters_count / len(architecture.layers)
                if param_per_layer > 0:
                    efficiency *= (1.0 + 1.0 / (1.0 + param_per_layer / 1000))
            
            return min(1.0, max(0.0, efficiency))
            
        except Exception as e:
            self.logger.warning(f"Efficiency calculation failed: {e}")
            return 0.5
    
    def _select_parents(self, population: List[Architecture]) -> List[Architecture]:
        """Select parents using tournament selection."""
        try:
            parents = []
            
            # Sort population by fitness
            sorted_population = sorted(population, key=lambda x: x.fitness_score, reverse=True)
            
            # Select elite individuals
            elite = sorted_population[:self.elite_size]
            parents.extend(elite)
            
            # Tournament selection for remaining parents
            while len(parents) < self.population_size:
                tournament = np.random.choice(population, size=3, replace=False)
                winner = max(tournament, key=lambda x: x.fitness_score)
                parents.append(winner)
            
            return parents[:self.population_size]
            
        except Exception as e:
            self.logger.warning(f"Parent selection failed: {e}")
            return population[:self.population_size//2]
    
    def _create_offspring(self, parents: List[Architecture], search_space: Dict[str, Any], 
                         data_shape: Tuple[int, int]) -> List[Architecture]:
        """Create offspring through crossover and mutation."""
        try:
            offspring = []
            
            while len(offspring) < self.population_size:
                # Select two parents
                parent1, parent2 = np.random.choice(parents, size=2, replace=False)
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = self._copy_architecture(parent1)
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    child = self._mutate(child, search_space, data_shape)
                
                offspring.append(child)
            
            return offspring[:self.population_size]
            
        except Exception as e:
            self.logger.warning(f"Offspring creation failed: {e}")
            return []
    
    def _crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Perform crossover between two architectures."""
        try:
            # Simple crossover: take layers from both parents
            child_layers = []
            
            max_layers = max(len(parent1.layers), len(parent2.layers))
            
            for i in range(max_layers):
                if i < len(parent1.layers) and i < len(parent2.layers):
                    # Randomly choose layer from either parent
                    if np.random.random() < 0.5:
                        child_layers.append(parent1.layers[i].copy())
                    else:
                        child_layers.append(parent2.layers[i].copy())
                elif i < len(parent1.layers):
                    child_layers.append(parent1.layers[i].copy())
                else:
                    child_layers.append(parent2.layers[i].copy())
            
            # Ensure valid architecture
            child_layers = self._validate_layers(child_layers)
            
            child = Architecture(
                layers=child_layers,
                parameters_count=self._calculate_parameters(child_layers),
                fitness_score=0.0,
                complexity_score=0.0,
                efficiency_score=0.0
            )
            
            return child
            
        except Exception as e:
            self.logger.warning(f"Crossover failed: {e}")
            return parent1
    
    def _mutate(self, architecture: Architecture, search_space: Dict[str, Any], 
                data_shape: Tuple[int, int]) -> Architecture:
        """Mutate an architecture."""
        try:
            mutated = self._copy_architecture(architecture)
            
            # Random mutations
            mutation_type = np.random.choice(['add_layer', 'remove_layer', 'modify_layer', 'change_activation'])
            
            if mutation_type == 'add_layer' and len(mutated.layers) < 6:
                self._add_layer(mutated, data_shape[1])
            elif mutation_type == 'remove_layer' and len(mutated.layers) > 2:
                self._remove_layer(mutated)
            elif mutation_type == 'modify_layer':
                self._modify_layer(mutated)
            elif mutation_type == 'change_activation':
                self._change_activation(mutated)
            
            # Recalculate parameters
            mutated.parameters_count = self._calculate_parameters(mutated.layers)
            
            return mutated
            
        except Exception as e:
            self.logger.warning(f"Mutation failed: {e}")
            return architecture
    
    def _add_layer(self, architecture: Architecture, input_size: int):
        """Add a layer to architecture."""
        try:
            if len(architecture.layers) > 0:
                last_layer = architecture.layers[-1]
                if last_layer['type'] == 'linear':
                    layer_size = np.random.randint(32, 256)
                    new_layer = {
                        'type': 'linear',
                        'input_size': last_layer['output_size'],
                        'output_size': layer_size,
                        'activation': np.random.choice(['relu', 'tanh', 'sigmoid']),
                        'dropout': np.random.uniform(0.0, 0.5)
                    }
                    architecture.layers.insert(-1, new_layer)  # Insert before output layer
        except Exception as e:
            self.logger.warning(f"Add layer failed: {e}")
    
    def _remove_layer(self, architecture: Architecture):
        """Remove a layer from architecture."""
        try:
            if len(architecture.layers) > 2:  # Keep at least input and output layers
                # Remove a hidden layer (not the last one)
                layer_to_remove = np.random.randint(0, len(architecture.layers) - 1)
                architecture.layers.pop(layer_to_remove)
                
                # Fix layer connections
                self._fix_layer_connections(architecture)
        except Exception as e:
            self.logger.warning(f"Remove layer failed: {e}")
    
    def _modify_layer(self, architecture: Architecture):
        """Modify a layer in architecture."""
        try:
            if len(architecture.layers) > 1:
                layer_idx = np.random.randint(0, len(architecture.layers) - 1)  # Don't modify output layer
                layer = architecture.layers[layer_idx]
                
                if layer['type'] == 'linear':
                    # Modify layer size
                    new_size = max(16, min(512, layer['output_size'] + np.random.randint(-32, 33)))
                    layer['output_size'] = new_size
                    
                    # Fix layer connections
                    self._fix_layer_connections(architecture)
        except Exception as e:
            self.logger.warning(f"Modify layer failed: {e}")
    
    def _change_activation(self, architecture: Architecture):
        """Change activation function in architecture."""
        try:
            activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu']
            
            for layer in architecture.layers:
                if 'activation' in layer and layer['activation'] != 'none':
                    layer['activation'] = np.random.choice(activations)
        except Exception as e:
            self.logger.warning(f"Change activation failed: {e}")
    
    def _fix_layer_connections(self, architecture: Architecture):
        """Fix layer input/output connections."""
        try:
            for i in range(len(architecture.layers)):
                layer = architecture.layers[i]
                if i == 0:
                    # First layer keeps original input size
                    pass
                else:
                    # Update input size to match previous layer output size
                    layer['input_size'] = architecture.layers[i-1]['output_size']
        except Exception as e:
            self.logger.warning(f"Fix layer connections failed: {e}")
    
    def _validate_layers(self, layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and fix layer definitions."""
        try:
            validated_layers = []
            
            for i, layer in enumerate(layers):
                if layer['type'] == 'linear':
                    # Ensure valid sizes
                    input_size = max(1, layer.get('input_size', 1))
                    output_size = max(1, layer.get('output_size', 1))
                    
                    validated_layer = layer.copy()
                    validated_layer['input_size'] = input_size
                    validated_layer['output_size'] = output_size
                    
                    validated_layers.append(validated_layer)
            
            return validated_layers
            
        except Exception as e:
            self.logger.warning(f"Layer validation failed: {e}")
            return layers
    
    def _copy_architecture(self, architecture: Architecture) -> Architecture:
        """Create a copy of an architecture."""
        try:
            copied_layers = []
            for layer in architecture.layers:
                copied_layers.append(layer.copy())
            
            return Architecture(
                layers=copied_layers,
                parameters_count=architecture.parameters_count,
                fitness_score=architecture.fitness_score,
                complexity_score=architecture.complexity_score,
                efficiency_score=architecture.efficiency_score
            )
            
        except Exception as e:
            self.logger.warning(f"Architecture copy failed: {e}")
            return architecture
    
    def _select_next_generation(self, population: List[Architecture]) -> List[Architecture]:
        """Select next generation using fitness-based selection."""
        try:
            # Sort by fitness
            sorted_population = sorted(population, key=lambda x: x.fitness_score, reverse=True)
            
            # Select top individuals
            return sorted_population[:self.population_size]
            
        except Exception as e:
            self.logger.warning(f"Next generation selection failed: {e}")
            return population[:self.population_size]
