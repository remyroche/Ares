"""
Evolutionary Architecture Search

Implementation for evolutionary architecture search in NAS clustering.
"""

from rich.console import Console
from rich import print as tprint

tprint("🔍 [EVOLUTIONARY_SEARCH] Loading Evolutionary Architecture Search module")
tprint("🔍 [EVOLUTIONARY_SEARCH] Module path: /workspace/src/training/steps/market_analysis/nas_clustering/core/nas_search/evolutionary_search.py")
tprint("🔍 [EVOLUTIONARY_SEARCH] Purpose: Implementation for evolutionary architecture search in NAS clustering")
tprint("🔍 [EVOLUTIONARY_SEARCH] Status: Starting module import")

import numpy as np
tprint("🔍 [EVOLUTIONARY_SEARCH] ✓ NumPy imported successfully")

import random
tprint("🔍 [EVOLUTIONARY_SEARCH] ✓ Random module imported successfully")

tprint("🔍 [EVOLUTIONARY_SEARCH] All imports completed successfully")

class EvolutionaryArchitectureSearch:
    """Evolutionary Architecture Search for NAS clustering."""
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1):
        tprint("🔍 [EVOLUTIONARY_SEARCH_INIT] Initializing EvolutionaryArchitectureSearch")
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_INIT] Population size: {population_size}")
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_INIT] Generations: {generations}")
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_INIT] Mutation rate: {mutation_rate}")
        
        self.population_size = population_size
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_INIT] ✓ Population size set to: {self.population_size}")
        
        self.generations = generations
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_INIT] ✓ Generations set to: {self.generations}")
        
        self.mutation_rate = mutation_rate
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_INIT] ✓ Mutation rate set to: {self.mutation_rate}")
        
        self.population = []
        tprint("🔍 [EVOLUTIONARY_SEARCH_INIT] ✓ Population initialized as empty list")
        
        self.fitness_history = []
        tprint("🔍 [EVOLUTIONARY_SEARCH_INIT] ✓ Fitness history initialized as empty list")
        
        self.best_individual = None
        tprint("🔍 [EVOLUTIONARY_SEARCH_INIT] ✓ Best individual initialized as None")
        
        self.best_fitness = float('-inf')
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_INIT] ✓ Best fitness initialized to: {self.best_fitness}")
        
        tprint("🔍 [EVOLUTIONARY_SEARCH_INIT] Initialization complete!")
        
    def search(self, search_space, fitness_function, initial_population=None):
        """Perform evolutionary architecture search."""
        tprint("🔍 [EVOLUTIONARY_SEARCH_SEARCH] Starting evolutionary architecture search")
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Search space: {search_space}")
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Fitness function provided: {fitness_function is not None}")
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Initial population provided: {initial_population is not None}")
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Population size: {self.population_size}")
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Generations: {self.generations}")
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Mutation rate: {self.mutation_rate}")
        
        # Initialize population
        tprint("🔍 [EVOLUTIONARY_SEARCH_SEARCH] Initializing population...")
        if initial_population:
            tprint("🔍 [EVOLUTIONARY_SEARCH_SEARCH] Using provided initial population")
            self.population = initial_population.copy()
            tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] ✓ Initial population copied - size: {len(self.population)}")
        else:
            tprint("🔍 [EVOLUTIONARY_SEARCH_SEARCH] Generating random initial population")
            self.population = self._initialize_population(search_space)
            tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] ✓ Random population generated - size: {len(self.population)}")
        
        # Ensure population size
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Ensuring population size: {self.population_size}")
        while len(self.population) < self.population_size:
            tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Adding individual {len(self.population)+1}/{self.population_size}")
            individual = self._generate_random_individual(search_space)
            self.population.append(individual)
        
        self.population = self.population[:self.population_size]
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] ✓ Population size ensured: {len(self.population)}")
        
        # Evolutionary loop
        tprint("🔍 [EVOLUTIONARY_SEARCH_SEARCH] Starting evolutionary loop...")
        for generation in range(self.generations):
            if generation % 10 == 0:  # Print progress every 10 generations
                tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Generation {generation}/{self.generations}")
            
            # Evaluate fitness
            tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Evaluating fitness for generation {generation}...")
            fitness_scores = self._evaluate_fitness(fitness_function)
            tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] ✓ Fitness evaluated for generation {generation}")
            tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Best fitness: {np.max(fitness_scores):.6f}")
            tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Average fitness: {np.mean(fitness_scores):.6f}")
            tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Fitness std: {np.std(fitness_scores):.6f}")
            
            # Record best individual
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_individual = self.population[best_idx].copy()
                tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] ✓ New best individual found - fitness: {self.best_fitness:.6f}")
            
            # Record fitness history
            fitness_record = {
                'generation': generation,
                'best_fitness': np.max(fitness_scores),
                'avg_fitness': np.mean(fitness_scores)
            }
            self.fitness_history.append(fitness_record)
            tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] ✓ Fitness history recorded for generation {generation}")
            
            # Selection and reproduction
            tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Performing selection and reproduction for generation {generation}...")
            self.population = self._selection_and_reproduction(fitness_scores, search_space)
            tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] ✓ Selection and reproduction completed for generation {generation}")
        
        result = {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history
        }
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] ✓ Evolutionary search completed successfully")
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Best fitness achieved: {self.best_fitness:.6f}")
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Total generations: {len(self.fitness_history)}")
        tprint(f"🔍 [EVOLUTIONARY_SEARCH_SEARCH] Result: {result}")
        return result
    
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
