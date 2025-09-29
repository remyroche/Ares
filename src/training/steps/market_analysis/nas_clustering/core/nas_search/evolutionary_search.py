"""
Evolutionary Architecture Search

Implementation for evolutionary architecture search in NAS clustering.
"""

import numpy as np
import random

class EvolutionaryArchitectureSearch:
    """Evolutionary Architecture Search for NAS clustering."""
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        
    def search(self, search_space, fitness_function, initial_population=None):
        """Perform evolutionary architecture search."""
        # Initialize population
        if initial_population:
            self.population = initial_population.copy()
        else:
            self.population = self._initialize_population(search_space)
        
        # Ensure population size
        while len(self.population) < self.population_size:
            individual = self._generate_random_individual(search_space)
            self.population.append(individual)
        
        self.population = self.population[:self.population_size]
        
        # Evolutionary loop
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = self._evaluate_fitness(fitness_function)
            
            # Record best individual
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_individual = self.population[best_idx].copy()
            
            # Record fitness history
            self.fitness_history.append({
                'generation': generation,
                'best_fitness': np.max(fitness_scores),
                'avg_fitness': np.mean(fitness_scores)
            })
            
            # Selection and reproduction
            self.population = self._selection_and_reproduction(fitness_scores, search_space)
        
        return {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history
        }
    
    def _initialize_population(self, search_space):
        """Initialize random population."""
        population = []
        for _ in range(self.population_size):
            individual = self._generate_random_individual(search_space)
            population.append(individual)
        return population
    
    def _generate_random_individual(self, search_space):
        """Generate random individual from search space."""
        individual = {}
        num_layers = random.randint(2, 8)
        layers = []
        
        for _ in range(num_layers):
            layer = {
                'width': random.choice(search_space.get('layer_widths', [32, 64, 128, 256, 512])),
                'activation': random.choice(search_space.get('activations', ['relu', 'tanh', 'sigmoid'])),
                'dropout': random.uniform(0.0, 0.5)
            }
            layers.append(layer)
        
        individual['layers'] = layers
        individual['fitness'] = 0.0
        return individual
    
    def _evaluate_fitness(self, fitness_function):
        """Evaluate fitness for all individuals."""
        fitness_scores = []
        for individual in self.population:
            try:
                fitness = fitness_function(individual)
                fitness_scores.append(fitness)
                individual['fitness'] = fitness
            except Exception:
                fitness_scores.append(-1.0)
                individual['fitness'] = -1.0
        return np.array(fitness_scores)
    
    def _selection_and_reproduction(self, fitness_scores, search_space):
        """Selection and reproduction operations."""
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        # Keep top 20% as elite
        elite_size = max(1, self.population_size // 5)
        new_population = [self.population[i].copy() for i in sorted_indices[:elite_size]]
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(fitness_scores)
            parent2 = self._tournament_selection(fitness_scores)
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child, search_space)
            
            new_population.append(child)
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, fitness_scores, tournament_size=3):
        """Tournament selection."""
        tournament_indices = random.sample(range(len(self.population)), 
                                          min(tournament_size, len(self.population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """Crossover operation."""
        child = {'layers': []}
        layers1 = parent1.get('layers', [])
        layers2 = parent2.get('layers', [])
        
        max_layers = max(len(layers1), len(layers2))
        
        for i in range(max_layers):
            if i < len(layers1) and i < len(layers2):
                if random.random() < 0.5:
                    child['layers'].append(layers1[i].copy())
                else:
                    child['layers'].append(layers2[i].copy())
            elif i < len(layers1):
                child['layers'].append(layers1[i].copy())
            else:
                child['layers'].append(layers2[i].copy())
        
        return child
    
    def _mutate(self, individual, search_space):
        """Mutation operation."""
        mutated = individual.copy()
        layers = mutated.get('layers', [])
        
        for layer in layers:
            if random.random() < 0.5:
                layer['width'] = random.choice(search_space.get('layer_widths', [32, 64, 128, 256, 512]))
            if random.random() < 0.3:
                layer['activation'] = random.choice(search_space.get('activations', ['relu', 'tanh', 'sigmoid']))
            if random.random() < 0.3:
                layer['dropout'] = random.uniform(0.0, 0.5)
        
        return mutated
