"""
Essential NAS Clusterer - True Neural Architecture Search

This module provides a streamlined implementation focusing only on essential NAS components
for dynamic neural architecture discovery and optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass

# Import essential NAS components
from .nas_search.evolutionary_search import (
    EvolutionaryArchitectureSearch, ArchitectureIndividual
)
from .nas_search.search_space import (
    SearchSpace, get_default_search_space, LayerType, ActivationFunction
)
from .evaluation.multi_objective import (
    ParetoFrontier, NSGAIIOptimizer, WeightedSumOptimizer, create_nas_objectives
)

logger = logging.getLogger(__name__)


@dataclass
class EssentialNASResult:
    """Result from essential NAS clustering operation."""
    success: bool
    best_architecture: Optional[ArchitectureIndividual]
    pareto_frontier: Optional[ParetoFrontier]
    execution_time: float
    search_statistics: Dict[str, Any]
    error_message: Optional[str] = None


class EssentialNASClusterer:
    """Essential NAS clusterer focusing only on true Neural Architecture Search."""
    
    def __init__(self, search_space: Optional[SearchSpace] = None,
                 population_size: int = 30, generations: int = 50,
                 enable_multi_objective: bool = True):
        """Initialize essential NAS clusterer."""
        self.search_space = search_space or get_default_search_space()
        self.population_size = population_size
        self.generations = generations
        self.enable_multi_objective = enable_multi_objective
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize evolutionary search
        self.evolutionary_search = EvolutionaryArchitectureSearch(
            search_space=self.search_space,
            population_size=population_size,
            generations=generations
        )
        
        # Initialize multi-objective optimizer if enabled
        self.multi_objective_optimizer = None
        if enable_multi_objective:
            objectives = create_nas_objectives()
            self.multi_objective_optimizer = NSGAIIOptimizer(
                objectives=objectives,
                population_size=min(20, population_size)
            )
        
        self.logger.info(f"✅ Essential NAS Clusterer initialized")
        self.logger.info(f"   Population size: {population_size}")
        self.logger.info(f"   Generations: {generations}")
        self.logger.info(f"   Multi-objective: {enable_multi_objective}")
    
    def search(self, data: np.ndarray, labels: np.ndarray) -> EssentialNASResult:
        """Perform essential neural architecture search."""
        try:
            start_time = time.time()
            self.logger.info(f"🚀 Starting essential NAS search")
            self.logger.info(f"   Data shape: {data.shape}")
            self.logger.info(f"   Labels shape: {labels.shape}")
            
            # Perform evolutionary architecture search
            best_architecture = self.evolutionary_search.search(data, labels)
            
            # Perform multi-objective optimization if enabled
            pareto_frontier = None
            if self.enable_multi_objective and self.multi_objective_optimizer:
                self.logger.info("🎯 Performing multi-objective optimization...")
                
                # Create candidate architectures from search
                candidate_architectures = [best_architecture]
                
                # Add some variations
                for i in range(5):
                    variant = best_architecture.copy()
                    variant.fitness_score = 0.0  # Reset fitness
                    candidate_architectures.append(variant)
                
                # Perform multi-objective optimization
                pareto_frontier = self.multi_objective_optimizer.optimize(
                    candidate_architectures, data, labels, max_iterations=15
                )
                
                # Get best solution from Pareto frontier
                best_solutions = pareto_frontier.get_best_solutions(1)
                if best_solutions:
                    best_architecture = best_solutions[0].architecture
            
            execution_time = time.time() - start_time
            
            # Get search statistics
            search_statistics = self.evolutionary_search.get_search_statistics()
            
            # Add multi-objective statistics if available
            if pareto_frontier:
                search_statistics.update({
                    'multi_objective_enabled': True,
                    'pareto_solutions': len(pareto_frontier.solutions),
                    'pareto_fronts': len(pareto_frontier.fronts) if pareto_frontier.fronts else 0
                })
            else:
                search_statistics.update({
                    'multi_objective_enabled': False,
                    'pareto_solutions': 0,
                    'pareto_fronts': 0
                })
            
            result = EssentialNASResult(
                success=True,
                best_architecture=best_architecture,
                pareto_frontier=pareto_frontier,
                execution_time=execution_time,
                search_statistics=search_statistics
            )
            
            self.logger.info(f"✅ Essential NAS search completed in {execution_time:.2f}s")
            self.logger.info(f"   Best architecture fitness: {best_architecture.fitness_score:.4f}")
            self.logger.info(f"   Architecture layers: {len(best_architecture.layers)}")
            self.logger.info(f"   Architecture connections: {len(best_architecture.connections)}")
            
            if pareto_frontier:
                self.logger.info(f"   Pareto solutions: {len(pareto_frontier.solutions)}")
                self.logger.info(f"   Pareto fronts: {len(pareto_frontier.fronts)}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Essential NAS search failed: {e}")
            
            return EssentialNASResult(
                success=False,
                best_architecture=None,
                pareto_frontier=None,
                execution_time=execution_time,
                search_statistics={},
                error_message=str(e)
            )
    
    def get_best_architecture_info(self, result: EssentialNASResult) -> Dict[str, Any]:
        """Get detailed information about the best architecture."""
        try:
            if not result.success or not result.best_architecture:
                return {}
            
            arch = result.best_architecture
            
            info = {
                'fitness_score': arch.fitness_score,
                'generation': arch.generation,
                'parameters_count': arch.parameters_count,
                'evaluation_time': arch.evaluation_time,
                'layers': [],
                'connections': []
            }
            
            # Layer information
            for i, layer in enumerate(arch.layers):
                layer_info = {
                    'index': i,
                    'type': layer.layer_type.value,
                    'activation': layer.activation.value,
                    'units': layer.units,
                    'dropout_rate': layer.dropout_rate,
                    'batch_norm': layer.batch_norm
                }
                
                if layer.kernel_size is not None:
                    layer_info['kernel_size'] = layer.kernel_size
                
                info['layers'].append(layer_info)
            
            # Connection information
            for i, conn in enumerate(arch.connections):
                conn_info = {
                    'index': i,
                    'type': conn.connection_type.value,
                    'from_layer': conn.from_layer,
                    'to_layer': conn.to_layer,
                    'weight': conn.weight
                }
                info['connections'].append(conn_info)
            
            return info
            
        except Exception as e:
            self.logger.warning(f"Best architecture info extraction failed: {e}")
            return {}
    
    def get_pareto_summary(self, result: EssentialNASResult) -> Dict[str, Any]:
        """Get summary of Pareto frontier results."""
        try:
            if not result.success or not result.pareto_frontier:
                return {}
            
            frontier = result.pareto_frontier
            summary = frontier.get_pareto_summary()
            
            # Add best solutions details
            best_solutions = frontier.get_best_solutions(5)
            summary['top_solutions'] = []
            
            for i, solution in enumerate(best_solutions):
                solution_info = {
                    'rank': i + 1,
                    'objectives': solution.objectives,
                    'rank_value': solution.rank,
                    'crowding_distance': solution.crowding_distance
                }
                summary['top_solutions'].append(solution_info)
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"Pareto summary extraction failed: {e}")
            return {}
    
    def print_search_results(self, result: EssentialNASResult):
        """Print comprehensive search results."""
        try:
            if not result.success:
                self.logger.error(f"❌ Search failed: {result.error_message}")
                return
            
            self.logger.info("📊 Essential NAS Search Results")
            self.logger.info("=" * 50)
            self.logger.info(f"Execution time: {result.execution_time:.2f}s")
            self.logger.info(f"Success: {result.success}")
            
            # Best architecture info
            arch_info = self.get_best_architecture_info(result)
            if arch_info:
                self.logger.info(f"Best architecture fitness: {arch_info['fitness_score']:.4f}")
                self.logger.info(f"Architecture layers: {len(arch_info['layers'])}")
                self.logger.info(f"Architecture connections: {len(arch_info['connections'])}")
                self.logger.info(f"Parameters count: {arch_info['parameters_count']}")
                
                # Layer details
                self.logger.info("Architecture layers:")
                for layer in arch_info['layers']:
                    self.logger.info(f"  Layer {layer['index']}: {layer['type']} "
                                   f"({layer['units']} units, {layer['activation']})")
            
            # Multi-objective results
            if result.pareto_frontier:
                pareto_summary = self.get_pareto_summary(result)
                if pareto_summary:
                    self.logger.info(f"Pareto solutions: {pareto_summary.get('total_solutions', 0)}")
                    self.logger.info(f"Pareto fronts: {pareto_summary.get('num_fronts', 0)}")
                    
                    if 'top_solutions' in pareto_summary:
                        self.logger.info("Top Pareto solutions:")
                        for solution in pareto_summary['top_solutions'][:3]:
                            self.logger.info(f"  Rank {solution['rank']}: {solution['objectives']}")
            
            # Search statistics
            stats = result.search_statistics
            if stats:
                self.logger.info("Search statistics:")
                self.logger.info(f"  Total generations: {stats.get('total_generations', 0)}")
                self.logger.info(f"  Final best fitness: {stats.get('final_best_fitness', 0.0):.4f}")
                self.logger.info(f"  Fitness improvement: {stats.get('fitness_improvement', 0.0):.4f}")
            
        except Exception as e:
            self.logger.warning(f"Results printing failed: {e}")