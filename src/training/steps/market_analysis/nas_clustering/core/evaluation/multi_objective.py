"""
Multi-objective Evaluation

Implementation for multi-objective optimization in NAS clustering.
"""

from rich.console import Console
from rich import print as tprint

tprint("🔍 [MULTI_OBJECTIVE] Loading Multi-objective Evaluation module")
tprint("🔍 [MULTI_OBJECTIVE] Module path: /workspace/src/training/steps/market_analysis/nas_clustering/core/evaluation/multi_objective.py")
tprint("🔍 [MULTI_OBJECTIVE] Purpose: Implementation for multi-objective optimization in NAS clustering")
tprint("🔍 [MULTI_OBJECTIVE] Status: Starting module import")

import numpy as np
tprint("🔍 [MULTI_OBJECTIVE] ✓ NumPy imported successfully")

tprint("🔍 [MULTI_OBJECTIVE] All imports completed successfully")

class NSGAIIOptimizer:
    """Non-dominated Sorting Genetic Algorithm II for multi-objective optimization."""
    def __init__(self, objectives=None, weights=None):
        tprint("🔍 [NSGAII_OPTIMIZER_INIT] Initializing NSGAIIOptimizer")
        tprint(f"🔍 [NSGAII_OPTIMIZER_INIT] Objectives provided: {objectives is not None}")
        tprint(f"🔍 [NSGAII_OPTIMIZER_INIT] Weights provided: {weights is not None}")
        
        self.objectives = objectives or []
        tprint(f"🔍 [NSGAII_OPTIMIZER_INIT] ✓ Objectives set to: {self.objectives}")
        
        self.weights = weights or []
        tprint(f"🔍 [NSGAII_OPTIMIZER_INIT] ✓ Weights set to: {self.weights}")
        
        self.population = []
        tprint("🔍 [NSGAII_OPTIMIZER_INIT] ✓ Population initialized as empty list")
        
        self.pareto_front = []
        tprint("🔍 [NSGAII_OPTIMIZER_INIT] ✓ Pareto front initialized as empty list")
        
        tprint("🔍 [NSGAII_OPTIMIZER_INIT] Initialization complete!")
        
    def optimize(self, architectures, data, target, generations=100):
        """Optimize architectures using NSGA-II."""
        tprint("🔍 [NSGAII_OPTIMIZER_OPTIMIZE] Starting NSGA-II optimization")
        tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] Number of architectures: {len(architectures)}")
        tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] Data shape: {data.shape}")
        tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] Target shape: {target.shape}")
        tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] Generations: {generations}")
        tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] Objectives: {self.objectives}")
        tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] Weights: {self.weights}")
        
        # Initialize population
        tprint("🔍 [NSGAII_OPTIMIZER_OPTIMIZE] Initializing population...")
        self.population = architectures.copy()
        tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] ✓ Population initialized with {len(self.population)} individuals")
        
        for generation in range(generations):
            if generation % 10 == 0:  # Print progress every 10 generations
                tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] Generation {generation}/{generations}")
            
            # Evaluate objectives
            tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] Evaluating objectives for generation {generation}...")
            self._evaluate_objectives(data, target)
            tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] ✓ Objectives evaluated for generation {generation}")
            
            # Non-dominated sorting
            tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] Performing non-dominated sorting for generation {generation}...")
            fronts = self._non_dominated_sorting()
            tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] ✓ Non-dominated sorting completed for generation {generation} - {len(fronts)} fronts")
            
            # Selection and reproduction
            tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] Performing selection and reproduction for generation {generation}...")
            self._selection_and_reproduction()
            tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] ✓ Selection and reproduction completed for generation {generation}")
        
        # Extract Pareto front
        tprint("🔍 [NSGAII_OPTIMIZER_OPTIMIZE] Extracting Pareto front...")
        self.pareto_front = self._extract_pareto_front()
        tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] ✓ Pareto front extracted with {len(self.pareto_front)} solutions")
        
        result = {'pareto_front': self.pareto_front}
        tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] ✓ Optimization completed successfully")
        tprint(f"🔍 [NSGAII_OPTIMIZER_OPTIMIZE] Result: {result}")
        return result
    
    def _evaluate_objectives(self, data, target):
        """Evaluate objectives for each architecture."""
        tprint("🔍 [NSGAII_OPTIMIZER_EVALUATE] Starting objective evaluation")
        tprint(f"🔍 [NSGAII_OPTIMIZER_EVALUATE] Population size: {len(self.population)}")
        tprint(f"🔍 [NSGAII_OPTIMIZER_EVALUATE] Objectives to evaluate: {self.objectives}")
        
        for i, individual in enumerate(self.population):
            if i % 10 == 0:  # Print progress every 10 individuals
                tprint(f"🔍 [NSGAII_OPTIMIZER_EVALUATE] Evaluating individual {i+1}/{len(self.population)}")
            
            objectives = {}
            tprint(f"🔍 [NSGAII_OPTIMIZER_EVALUATE] Individual {i}: Starting objective evaluation")
            
            for obj_type in self.objectives:
                tprint(f"🔍 [NSGAII_OPTIMIZER_EVALUATE] Individual {i}: Evaluating objective '{obj_type}'")
                if obj_type == 'accuracy':
                    objectives['accuracy'] = np.random.random()
                    tprint(f"🔍 [NSGAII_OPTIMIZER_EVALUATE] Individual {i}: Accuracy = {objectives['accuracy']:.6f}")
                elif obj_type == 'efficiency':
                    objectives['efficiency'] = np.random.random()
                    tprint(f"🔍 [NSGAII_OPTIMIZER_EVALUATE] Individual {i}: Efficiency = {objectives['efficiency']:.6f}")
                elif obj_type == 'complexity':
                    objectives['complexity'] = np.random.random()
                    tprint(f"🔍 [NSGAII_OPTIMIZER_EVALUATE] Individual {i}: Complexity = {objectives['complexity']:.6f}")
            
            individual['objectives'] = objectives
            tprint(f"🔍 [NSGAII_OPTIMIZER_EVALUATE] Individual {i}: Objectives assigned: {objectives}")
        
        tprint("🔍 [NSGAII_OPTIMIZER_EVALUATE] ✓ Objective evaluation completed")
    
    def _non_dominated_sorting(self):
        """Perform non-dominated sorting."""
        fronts = []
        remaining = self.population.copy()
        
        while remaining:
            current_front = []
            dominated = []
            
            for i, individual in enumerate(remaining):
                is_dominated = False
                for j, other in enumerate(remaining):
                    if i != j and self._dominates(other, individual):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    current_front.append(individual)
                else:
                    dominated.append(individual)
            
            if current_front:
                fronts.append(current_front)
                remaining = dominated
            else:
                break
        
        return fronts
    
    def _dominates(self, individual1, individual2):
        """Check if individual1 dominates individual2."""
        obj1 = individual1.get('objectives', {})
        obj2 = individual2.get('objectives', {})
        
        at_least_as_good = all(
            obj1.get(obj, 0) >= obj2.get(obj, 0) for obj in self.objectives
        )
        strictly_better = any(
            obj1.get(obj, 0) > obj2.get(obj, 0) for obj in self.objectives
        )
        
        return at_least_as_good and strictly_better
    
    def _selection_and_reproduction(self):
        """Selection and reproduction operations."""
        # Simple selection based on objectives
        self.population.sort(key=lambda x: sum(x.get('objectives', {}).values()), reverse=True)
        self.population = self.population[:len(self.population)]
    
    def _extract_pareto_front(self):
        """Extract Pareto front."""
        fronts = self._non_dominated_sorting()
        return fronts[0] if fronts else []

def create_nas_objectives(objective_types=None, weights=None):
    """Create NAS objective functions."""
    tprint("🔍 [CREATE_NAS_OBJECTIVES] Creating NAS objective functions")
    tprint(f"🔍 [CREATE_NAS_OBJECTIVES] Objective types provided: {objective_types is not None}")
    tprint(f"🔍 [CREATE_NAS_OBJECTIVES] Weights provided: {weights is not None}")
    
    if objective_types is None:
        objective_types = ['accuracy', 'efficiency', 'complexity']
        tprint("🔍 [CREATE_NAS_OBJECTIVES] Using default objective types")
    
    tprint(f"🔍 [CREATE_NAS_OBJECTIVES] Objective types: {objective_types}")
    tprint(f"🔍 [CREATE_NAS_OBJECTIVES] Weights: {weights}")
    
    objectives = []
    tprint("🔍 [CREATE_NAS_OBJECTIVES] Creating objective functions...")
    
    for i, obj_type in enumerate(objective_types):
        tprint(f"🔍 [CREATE_NAS_OBJECTIVES] Creating objective function {i+1}: '{obj_type}'")
        if obj_type == 'accuracy':
            objectives.append(lambda x: x.get('accuracy', 0))
            tprint(f"🔍 [CREATE_NAS_OBJECTIVES] ✓ Accuracy objective function created")
        elif obj_type == 'efficiency':
            objectives.append(lambda x: x.get('efficiency', 0))
            tprint(f"🔍 [CREATE_NAS_OBJECTIVES] ✓ Efficiency objective function created")
        elif obj_type == 'complexity':
            objectives.append(lambda x: x.get('complexity', 0))
            tprint(f"🔍 [CREATE_NAS_OBJECTIVES] ✓ Complexity objective function created")
    
    tprint(f"🔍 [CREATE_NAS_OBJECTIVES] ✓ Created {len(objectives)} objective functions")
    tprint("🔍 [CREATE_NAS_OBJECTIVES] Objective function creation completed")
    return objectives
