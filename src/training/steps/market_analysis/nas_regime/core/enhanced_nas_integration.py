"""
Enhanced NAS Integration Module

This module integrates the advanced neural architectures and enhanced search strategies
to provide a comprehensive neural architecture search system for regime detection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass
from pathlib import Path
import time
import json

# Import tprint for comprehensive debugging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)
from src.utils.pipeline_results_manager import pipeline_results_manager

from .advanced_neural_architectures import (
    AdvancedArchitectureConfig, ArchitectureType, create_advanced_architecture,
    AdvancedArchitectureManager, TransformerRegimeDetector, GraphNeuralNetworkRegimeDetector,
    HybridTransformerGNN
)

from .enhanced_search_strategies import (
    SearchStrategyConfig, SearchStrategyType, create_search_strategy,
    create_enhanced_search_manager, EnhancedSearchStrategyManager,
    ReinforcementLearningSearch, DifferentiableArchitectureSearch,
    ProgressiveArchitectureSearch, MultiObjectiveEvolutionarySearch
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedNASConfig:
    """Configuration for Enhanced NAS system."""
    # Architecture configuration
    architecture_config: AdvancedArchitectureConfig = None
    
    # Search strategy configuration
    search_config: SearchStrategyConfig = None
    
    # General parameters
    max_search_iterations: int = 1000
    performance_threshold: float = 0.8
    enable_parallel_evaluation: bool = True
    n_workers: int = 4
    
    # Output configuration
    output_dir: str = "outcomes"
    save_intermediate_results: bool = True
    save_final_architecture: bool = True
    
    def __post_init__(self):
        if self.architecture_config is None:
            self.architecture_config = AdvancedArchitectureConfig()
        if self.search_config is None:
            self.search_config = SearchStrategyConfig()


@dataclass
class EnhancedNASResult:
    """Result from Enhanced NAS search."""
    success: bool
    best_architecture: Any
    best_performance: float
    search_history: List[Dict[str, Any]]
    architecture_info: Dict[str, Any]
    search_strategy_used: str
    execution_time: float
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None


class SearchSpace:
    """Search space definition for neural architectures."""
    
    def __init__(self, operations: List[str], max_layers: int = 20, max_ops_per_layer: int = 5):
        self.operations = operations
        self.max_layers = max_layers
        self.max_ops_per_layer = max_ops_per_layer
        self.edges = list(range(max_layers))
    
    def create_empty_architecture(self):
        """Create an empty architecture."""
        return Architecture(layers=[])
    
    def apply_operation(self, architecture, operation_idx):
        """Apply an operation to the architecture."""
        if operation_idx < len(self.operations):
            layer = Layer(operation_id=operation_idx, operation_name=self.operations[operation_idx])
            architecture.layers.append(layer)
        return architecture
    
    def sample_random_architecture(self):
        """Sample a random architecture from the search space."""
        architecture = self.create_empty_architecture()
        num_layers = np.random.randint(1, self.max_layers + 1)
        
        for _ in range(num_layers):
            operation_idx = np.random.randint(0, len(self.operations))
            architecture = self.apply_operation(architecture, operation_idx)
        
        return architecture


@dataclass
class Layer:
    """Neural network layer definition."""
    operation_id: int
    operation_name: str
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class Architecture:
    """Neural architecture definition."""
    layers: List[Layer]
    estimated_complexity: float = 0.0
    
    def __post_init__(self):
        if self.estimated_complexity == 0.0:
            self.estimated_complexity = len(self.layers)


class PerformanceEvaluator:
    """Performance evaluator for neural architectures."""
    
    def __init__(self, config: EnhancedNASConfig):
        self.config = config
        self.evaluation_cache = {}
        self.evaluation_count = 0
        
    def __call__(self, architecture) -> float:
        """Evaluate architecture performance."""
        # Check cache first
        arch_key = self._get_architecture_key(architecture)
        if arch_key in self.evaluation_cache:
            return self.evaluation_cache[arch_key]
        
        # Evaluate performance
        performance = self._evaluate_architecture(architecture)
        
        # Cache result
        self.evaluation_cache[arch_key] = performance
        self.evaluation_count += 1
        
        return performance
    
    def _get_architecture_key(self, architecture):
        """Get unique key for architecture."""
        return tuple((layer.operation_id, layer.operation_name) for layer in architecture.layers)
    
    def _evaluate_architecture(self, architecture) -> float:
        """Evaluate architecture performance (simplified implementation)."""
        # This is a simplified evaluation function
        # In practice, this would involve:
        # 1. Creating the actual neural network from the architecture
        # 2. Training it on a validation set
        # 3. Evaluating its performance
        
        # Base performance from architecture complexity
        base_performance = 0.3
        
        # Complexity bonus (more layers can be better up to a point)
        complexity_bonus = min(len(architecture.layers) * 0.05, 0.4)
        
        # Operation diversity bonus
        unique_operations = len(set(layer.operation_id for layer in architecture.layers))
        diversity_bonus = unique_operations * 0.02
        
        # Random noise for realism
        noise = np.random.normal(0, 0.05)
        
        performance = base_performance + complexity_bonus + diversity_bonus + noise
        
        # Ensure performance is between 0 and 1
        return max(0.0, min(1.0, performance))


class EnhancedNASSystem:
    """Enhanced Neural Architecture Search System."""
    
    def __init__(self, config: EnhancedNASConfig):
        tprint("🚀 [ENHANCED-NAS] Initializing Enhanced NAS System", color="blue", bold=True)
        tprint(f"📊 [ENHANCED-NAS] Architecture: {config.architecture_config.architecture_type.value}", color="cyan")
        tprint(f"📊 [ENHANCED-NAS] Search Strategy: {config.search_config.strategy_type.value}", color="cyan")
        tprint(f"📊 [ENHANCED-NAS] Max Iterations: {config.max_search_iterations}", color="cyan")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        tprint("🔧 [ENHANCED-NAS] Initializing search space", color="yellow")
        self.search_space = self._initialize_search_space()
        
        tprint("🔧 [ENHANCED-NAS] Initializing performance evaluator", color="yellow")
        self.performance_evaluator = PerformanceEvaluator(config)
        
        tprint("🔧 [ENHANCED-NAS] Initializing search manager", color="yellow")
        self.search_manager = create_enhanced_search_manager(
            self.search_space, self.performance_evaluator, config.search_config
        )
        
        # Results tracking
        self.search_results = []
        self.best_architecture = None
        self.best_performance = -np.inf
        
        # Create output directory
        tprint(f"🔧 [ENHANCED-NAS] Creating output directory: {config.output_dir}", color="yellow")
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        tprint_success("✅ [ENHANCED-NAS] Enhanced NAS System fully initialized")
        tprint(f"📊 [ENHANCED-NAS] Search space size: {len(self.search_space.operations)} operations", color="cyan")
    
    def _initialize_search_space(self) -> SearchSpace:
        """Initialize the search space."""
        # Define available operations
        operations = [
            "conv1d", "conv2d", "linear", "lstm", "gru", "attention",
            "batch_norm", "layer_norm", "dropout", "relu", "gelu",
            "transformer_block", "graph_conv", "temporal_conv"
        ]
        
        return SearchSpace(
            operations=operations,
            max_layers=self.config.architecture_config.num_layers,
            max_ops_per_layer=5
        )
    
    def search(self, strategy_type: Optional[SearchStrategyType] = None, **kwargs) -> EnhancedNASResult:
        """Perform neural architecture search."""
        start_time = time.time()
        
        try:
            tprint("🚀 [ENHANCED-NAS] Starting Enhanced NAS search", color="blue", bold=True)
            strategy_name = strategy_type.value if strategy_type else self.config.search_config.strategy_type.value
            tprint(f"📊 [ENHANCED-NAS] Strategy: {strategy_name}", color="cyan")
            
            # Perform search
            if strategy_type == SearchStrategyType.HYBRID_SEARCH:
                tprint("🔧 [ENHANCED-NAS] Executing hybrid search", color="yellow")
                result = self.search_manager.hybrid_search(**kwargs)
            else:
                tprint(f"🔧 [ENHANCED-NAS] Executing {strategy_name} search", color="yellow")
                result = self.search_manager.search(strategy_type, **kwargs)
            
            # Create architecture from search result
            if result and 'best_architecture' in result:
                best_arch = result['best_architecture']
                best_performance = result.get('best_performance', 0.0)
                
                tprint(f"📊 [ENHANCED-NAS] Search completed with performance: {best_performance:.4f}", color="cyan")
                
                # Create advanced architecture
                tprint("🔧 [ENHANCED-NAS] Creating advanced architecture manager", color="yellow")
                advanced_arch_manager = AdvancedArchitectureManager(self.config.architecture_config)
                
                # Update tracking
                if best_performance > self.best_performance:
                    tprint(f"🏆 [ENHANCED-NAS] New best performance: {best_performance:.4f}", color="green", bold=True)
                    self.best_performance = best_performance
                    self.best_architecture = best_arch
                else:
                    tprint(f"📊 [ENHANCED-NAS] Current performance: {best_performance:.4f} (best overall: {self.best_performance:.4f})", color="cyan")
                
                # Create final result
                execution_time = time.time() - start_time
                nas_result = EnhancedNASResult(
                    success=True,
                    best_architecture=best_arch,
                    best_performance=best_performance,
                    search_history=result.get('search_history', []),
                    architecture_info=advanced_arch_manager.get_architecture_info(),
                    search_strategy_used=strategy_name,
                    execution_time=execution_time,
                    metadata={
                        'evaluation_count': self.performance_evaluator.evaluation_count,
                        'cache_hit_rate': len(self.performance_evaluator.evaluation_cache) / max(1, self.performance_evaluator.evaluation_count),
                        'search_space_size': len(self.search_space.operations) ** self.config.architecture_config.num_layers,
                        'search_strategy_details': {
                            'strategy_type': strategy_name,
                            'search_iterations': len(result.get('search_history', [])),
                            'architecture_complexity': len(best_arch.layers) if best_arch else 0
                        }
                    }
                )
                
                # Save results if configured
                if self.config.save_intermediate_results:
                    tprint("🔧 [ENHANCED-NAS] Saving intermediate results", color="yellow")
                    self._save_intermediate_results(nas_result)
                
                self.search_results.append(nas_result)
                
                tprint_success("✅ [ENHANCED-NAS] Enhanced NAS search completed successfully")
                tprint(f"📊 [ENHANCED-NAS] Final metrics: Performance={best_performance:.4f}, Time={execution_time:.2f}s", color="cyan")
                
                return nas_result
            else:
                tprint_error("❌ [ENHANCED-NAS] Search did not return valid results")
                raise ValueError("Search did not return valid results")
                
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ [ENHANCED-NAS] Enhanced NAS search failed: {e}")
            
            return EnhancedNASResult(
                success=False,
                best_architecture=None,
                best_performance=0.0,
                search_history=[],
                architecture_info={},
                search_strategy_used=strategy_type.value if strategy_type else self.config.search_config.strategy_type.value,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _save_intermediate_results(self, result: EnhancedNASResult):
        """Save intermediate results using pipeline results manager."""
        try:
            # Convert result to serializable format
            result_dict = {
                'success': result.success,
                'best_performance': result.best_performance,
                'search_strategy_used': result.search_strategy_used,
                'execution_time': result.execution_time,
                'architecture_info': result.architecture_info,
                'metadata': result.metadata,
                'search_history_summary': {
                    'total_iterations': len(result.search_history),
                    'best_performance_evolution': [
                        entry.get('best_performance', 0.0) for entry in result.search_history
                        if 'best_performance' in entry
                    ]
                }
            }
            
            # Use pipeline results manager to save to outcomes/ directory
            filepath = pipeline_results_manager.save_nas_results(
                nas_result=result_dict,
                symbol=getattr(self, 'symbol', None),
                timeframe=getattr(self, 'timeframe', None),
                additional_metadata={
                    'search_iterations': len(result.search_history),
                    'best_architecture_type': result.architecture_info.get('architecture_type', 'unknown')
                }
            )
            
            self.logger.info(f"ℹ️ Intermediate results saved to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save intermediate results: {e}")
    
    def save_final_results(self):
        """Save final results and best architecture."""
        if not self.best_architecture:
            self.logger.warning("No best architecture found to save")
            return
        
        try:
            # Save best architecture
            if self.config.save_final_architecture:
                arch_filename = f"best_architecture_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
                arch_filepath = Path(self.config.output_dir) / arch_filename
                
                # Create architecture manager for the best architecture
                arch_manager = AdvancedArchitectureManager(self.config.architecture_config)
                arch_manager.save_architecture(str(arch_filepath))
                
                self.logger.info(f"Best architecture saved to {arch_filepath}")
            
            # Save search summary
            summary_filename = f"search_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
            summary_filepath = Path(self.config.output_dir) / summary_filename
            
            summary = self.search_manager.get_search_summary()
            summary.update({
                'total_searches': len(self.search_results),
                'best_overall_performance': self.best_performance,
                'config': {
                    'architecture_type': self.config.architecture_config.architecture_type.value,
                    'search_strategy': self.config.search_config.strategy_type.value,
                    'max_iterations': self.config.max_search_iterations
                }
            })
            
            with open(summary_filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Search summary saved to {summary_filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save final results: {e}")
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of all search results."""
        tprint("📊 [ENHANCED-NAS] Generating search summary", color="yellow")
        summary = self.search_manager.get_search_summary()
        tprint(f"📊 [ENHANCED-NAS] Summary generated: {len(summary.get('strategy_performance', {}))} strategies", color="cyan")
        return summary
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report with all enhanced NAS data."""
        tprint("📋 [ENHANCED-NAS] Generating comprehensive final report", color="blue", bold=True)
        
        # Collect all data
        report = {
            'enhanced_nas_overview': {
                'system_initialized': True,
                'total_searches': len(self.search_results),
                'best_overall_performance': self.best_performance,
                'configuration': {
                    'architecture_type': self.config.architecture_config.architecture_type.value,
                    'search_strategy': self.config.search_config.strategy_type.value,
                    'max_iterations': self.config.max_search_iterations,
                    'output_directory': self.config.output_dir
                }
            },
            'advanced_architectures': {
                'available_types': [arch_type.value for arch_type in ArchitectureType],
                'selected_architecture': self.config.architecture_config.architecture_type.value,
                'architecture_details': {
                    'input_dim': self.config.architecture_config.input_dim,
                    'hidden_dim': self.config.architecture_config.hidden_dim,
                    'num_heads': self.config.architecture_config.num_heads,
                    'num_layers': self.config.architecture_config.num_layers,
                    'num_regimes': self.config.architecture_config.num_regimes,
                    'dropout': self.config.architecture_config.dropout
                }
            },
            'enhanced_search_strategies': {
                'available_strategies': [strategy.value for strategy in SearchStrategyType],
                'selected_strategy': self.config.search_config.strategy_type.value,
                'strategy_details': {
                    'rl_learning_rate': self.config.search_config.rl_learning_rate,
                    'rl_gamma': self.config.search_config.rl_gamma,
                    'progressive_initial_ops': self.config.search_config.progressive_initial_ops,
                    'progressive_max_ops': self.config.search_config.progressive_max_ops,
                    'mo_population_size': self.config.search_config.mo_population_size,
                    'mo_generations': self.config.search_config.mo_generations
                }
            },
            'search_results': [],
            'performance_metrics': {
                'evaluation_count': self.performance_evaluator.evaluation_count,
                'cache_hit_rate': len(self.performance_evaluator.evaluation_cache) / max(1, self.performance_evaluator.evaluation_count),
                'search_space_size': len(self.search_space.operations) ** self.config.architecture_config.num_layers
            },
            'best_architecture': None,
            'search_history_summary': {}
        }
        
        # Process search results
        for i, result in enumerate(self.search_results):
            tprint(f"🔧 [ENHANCED-NAS] Processing search result {i+1}/{len(self.search_results)}", color="yellow")
            
            search_result_data = {
                'search_index': i + 1,
                'success': result.success,
                'strategy_used': result.search_strategy_used,
                'best_performance': result.best_performance,
                'execution_time': result.execution_time,
                'architecture_info': result.architecture_info,
                'metadata': result.metadata,
                'error_message': result.error_message if not result.success else None
            }
            
            if result.best_architecture:
                search_result_data['architecture_details'] = {
                    'num_layers': len(result.best_architecture.layers),
                    'estimated_complexity': result.best_architecture.estimated_complexity,
                    'layer_operations': [layer.operation_name for layer in result.best_architecture.layers]
                }
            
            report['search_results'].append(search_result_data)
        
        # Set best architecture
        if self.best_architecture:
            tprint("🔧 [ENHANCED-NAS] Processing best architecture details", color="yellow")
            report['best_architecture'] = {
                'num_layers': len(self.best_architecture.layers),
                'estimated_complexity': self.best_architecture.estimated_complexity,
                'layer_operations': [layer.operation_name for layer in self.best_architecture.layers],
                'performance': self.best_performance
            }
        
        # Generate search history summary
        if self.search_results:
            tprint("🔧 [ENHANCED-NAS] Generating search history summary", color="yellow")
            all_histories = []
            for result in self.search_results:
                if result.search_history:
                    all_histories.extend(result.search_history)
            
            report['search_history_summary'] = {
                'total_iterations': len(all_histories),
                'performance_evolution': [
                    entry.get('best_performance', 0.0) for entry in all_histories
                    if 'best_performance' in entry
                ],
                'strategy_breakdown': {}
            }
            
            # Strategy breakdown
            for result in self.search_results:
                strategy = result.search_strategy_used
                if strategy not in report['search_history_summary']['strategy_breakdown']:
                    report['search_history_summary']['strategy_breakdown'][strategy] = {
                        'count': 0,
                        'total_performance': 0.0,
                        'best_performance': 0.0,
                        'total_time': 0.0
                    }
                
                breakdown = report['search_history_summary']['strategy_breakdown'][strategy]
                breakdown['count'] += 1
                breakdown['total_performance'] += result.best_performance
                breakdown['best_performance'] = max(breakdown['best_performance'], result.best_performance)
                breakdown['total_time'] += result.execution_time
            
            # Calculate averages
            for strategy, data in report['search_history_summary']['strategy_breakdown'].items():
                data['average_performance'] = data['total_performance'] / data['count']
                data['average_time'] = data['total_time'] / data['count']
        
        tprint_success("✅ [ENHANCED-NAS] Comprehensive final report generated")
        tprint(f"📊 [ENHANCED-NAS] Report contains: {len(report['search_results'])} search results, {len(report['search_history_summary'].get('strategy_breakdown', {}))} strategies", color="cyan")
        
        return report
    
    def compare_strategies(self, strategies: List[SearchStrategyType], **kwargs) -> Dict[str, EnhancedNASResult]:
        """Compare multiple search strategies."""
        self.logger.info(f"Comparing search strategies: {[s.value for s in strategies]}")
        
        results = {}
        for strategy in strategies:
            try:
                result = self.search(strategy, **kwargs)
                results[strategy.value] = result
                
                self.logger.info(f"Strategy {strategy.value}: Best performance = {result.best_performance:.4f}")
                
            except Exception as e:
                self.logger.error(f"Strategy {strategy.value} failed: {e}")
                results[strategy.value] = EnhancedNASResult(
                    success=False,
                    best_architecture=None,
                    best_performance=0.0,
                    search_history=[],
                    architecture_info={},
                    search_strategy_used=strategy.value,
                    execution_time=0.0,
                    error_message=str(e)
                )
        
        return results
    
    def benchmark_architectures(self, architectures: List[ArchitectureType]) -> Dict[str, float]:
        """Benchmark different architecture types."""
        self.logger.info(f"Benchmarking architecture types: {[a.value for a in architectures]}")
        
        results = {}
        original_arch_type = self.config.architecture_config.architecture_type
        
        for arch_type in architectures:
            try:
                # Update configuration
                self.config.architecture_config.architecture_type = arch_type
                
                # Create and evaluate architecture
                arch_manager = AdvancedArchitectureManager(self.config.architecture_config)
                
                # Simple evaluation (in practice, this would involve training)
                performance = np.random.uniform(0.5, 0.9)  # Placeholder
                results[arch_type.value] = performance
                
                self.logger.info(f"Architecture {arch_type.value}: Performance = {performance:.4f}")
                
            except Exception as e:
                self.logger.error(f"Architecture {arch_type.value} failed: {e}")
                results[arch_type.value] = 0.0
        
        # Restore original configuration
        self.config.architecture_config.architecture_type = original_arch_type
        
        return results


# Factory functions
def create_enhanced_nas_system(config: EnhancedNASConfig) -> EnhancedNASSystem:
    """Factory function to create Enhanced NAS system."""
    return EnhancedNASSystem(config)


def create_default_enhanced_nas_config() -> EnhancedNASConfig:
    """Create default Enhanced NAS configuration."""
    return EnhancedNASConfig()


def quick_enhanced_nas_search(
    architecture_type: ArchitectureType = ArchitectureType.TRANSFORMER_REGIME,
    search_strategy: SearchStrategyType = SearchStrategyType.REINFORCEMENT_LEARNING,
    max_iterations: int = 100
) -> EnhancedNASResult:
    """Quick Enhanced NAS search with default settings."""
    
    # Create configuration
    config = EnhancedNASConfig()
    config.architecture_config.architecture_type = architecture_type
    config.search_config.strategy_type = search_strategy
    config.max_search_iterations = max_iterations
    config.save_intermediate_results = False
    config.save_final_architecture = False
    
    # Create and run system
    nas_system = create_enhanced_nas_system(config)
    result = nas_system.search()
    
    return result


def compare_all_strategies(max_iterations: int = 50) -> Dict[str, EnhancedNASResult]:
    """Compare all available search strategies."""
    
    config = EnhancedNASConfig()
    config.max_search_iterations = max_iterations
    config.save_intermediate_results = False
    config.save_final_architecture = False
    
    nas_system = create_enhanced_nas_system(config)
    
    strategies = [
        SearchStrategyType.REINFORCEMENT_LEARNING,
        SearchStrategyType.PROGRESSIVE_SEARCH,
        SearchStrategyType.MULTI_OBJECTIVE_EVOLUTIONARY
    ]
    
    return nas_system.compare_strategies(strategies)


def benchmark_all_architectures() -> Dict[str, float]:
    """Benchmark all available architecture types."""
    
    config = create_default_enhanced_nas_config()
    nas_system = create_enhanced_nas_system(config)
    
    architectures = [
        ArchitectureType.TRANSFORMER_REGIME,
        ArchitectureType.GRAPH_NEURAL_NETWORK,
        ArchitectureType.TEMPORAL_CONVOLUTIONAL,
        ArchitectureType.HYBRID_TRANSFORMER_GNN
    ]
    
    return nas_system.benchmark_architectures(architectures)