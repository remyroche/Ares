"""
NAS Integration with Shared Utilities

This module demonstrates how to integrate the shared utilities (evolutionary search,
feature engineering, and advanced metrics) with Neural Architecture Search (NAS) systems.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
import time
import warnings
warnings.filterwarnings('ignore')

# Import centralized NAS utilities
from src.utils.nas_tas.core.nas_engine import NASEngine
from src.utils.nas_tas.optimization.architecture_search import ArchitectureSearchOptimizer, ArchitectureSearchConfig

# Import shared utilities
try:
    from src.utils.ml_common.optimization.shared_utils.evolutionary_search import (
        EvolutionaryAlgorithmManager, EvolutionaryConfig, EvolutionaryResult,
        create_evolutionary_algorithm_manager, Individual
    )
    from src.utils.ml_common.optimization.shared_utils.feature_engineering import (
        UnifiedFeatureEngineer, FeatureConfig, FeatureEngineeringResult,
        create_unified_feature_engineer
    )
    from src.utils.ml_common.optimization.shared_utils.advanced_metrics import (
        AdvancedEvaluator, AdvancedEvaluationResult,
        create_advanced_evaluator
    )
    SHARED_UTILS_AVAILABLE = True
except ImportError as e:
    SHARED_UTILS_AVAILABLE = False
    print(f"⚠️ Shared utilities not available: {e}")

# Import tprint for debugging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

@dataclass
class NASArchitecture:
    """Neural architecture representation for NAS."""
    layers: List[Dict[str, Any]]
    parameters_count: int
    fitness_score: float
    complexity_score: float
    efficiency_score: float
    regime_accuracy: float
    economic_significance: float
    trading_viability: float
    architecture_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert architecture to dictionary."""
        return {
            'layers': self.layers,
            'parameters_count': self.parameters_count,
            'fitness_score': self.fitness_score,
            'complexity_score': self.complexity_score,
            'efficiency_score': self.efficiency_score,
            'regime_accuracy': self.regime_accuracy,
            'economic_significance': self.economic_significance,
            'trading_viability': self.trading_viability,
            'architecture_id': self.architecture_id
        }

@dataclass
class NASSearchConfig:
    """Configuration for NAS search with shared utilities."""

    # Evolutionary search configuration
    population_size: int = 50
    max_generations: int = 100
    crossover_probability: float = 0.8
    mutation_probability: float = 0.1
    tournament_size: int = 3
    elitism_size: int = 10

    # Feature engineering configuration
    enable_feature_engineering: bool = True
    feature_selection_method: str = "mutual_info"
    max_features: int = 100

    # Advanced evaluation configuration
    enable_advanced_metrics: bool = True
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "risk_adjusted", "regime_aware", "economic_significance", "trading_viability"
    ])

    # NAS-specific configuration
    search_space: Dict[str, Any] = field(default_factory=dict)
    performance_evaluator: Optional[Callable] = None
    regime_labels: Optional[np.ndarray] = None

    # Multi-objective optimization
    objectives: List[str] = field(default_factory=lambda: [
        "accuracy", "efficiency", "complexity", "regime_awareness"
    ])
    objective_weights: List[float] = field(default_factory=lambda: [0.4, 0.2, 0.2, 0.2])

class NASSharedUtilsIntegration:
    """Integration of shared utilities with NAS systems."""

    def __init__(self, config: NASSearchConfig):
        """Initialize NAS integration with shared utilities."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize shared utilities
        self._initialize_shared_utilities()

        # NAS state
        self.population: List[NASArchitecture] = []
        self.search_history: List[Dict[str, Any]] = []
        self.best_architectures: List[NASArchitecture] = []

        tprint_success("✅ NAS Shared Utils Integration initialized")

    def _initialize_shared_utilities(self):
        """Initialize shared utilities."""
        if not SHARED_UTILS_AVAILABLE:
            tprint_warning("⚠️ Shared utilities not available")
            return

        try:
            # Initialize evolutionary search
            tprint_debug("🧬 Initializing evolutionary search for NAS...")
            evolutionary_config = EvolutionaryConfig(
                population_size=self.config.population_size,
                max_generations=self.config.max_generations,
                crossover_probability=self.config.crossover_probability,
                mutation_probability=self.config.mutation_probability,
                tournament_size=self.config.tournament_size,
                elitism_size=self.config.elitism_size,
                use_nsga2=True,  # Use NSGA-II for multi-objective optimization
                use_spea2=True,
                use_genetic_algorithm=True
            )
            self.evolutionary_manager = create_evolutionary_algorithm_manager(evolutionary_config)
            tprint_success("✅ Evolutionary search initialized for NAS")

            # Initialize feature engineering
            if self.config.enable_feature_engineering:
                tprint_debug("🔧 Initializing feature engineering for NAS...")
                feature_config = FeatureConfig(
                    enable_technical_indicators=True,
                    enable_feature_selection=True,
                    feature_selection_method=self.config.feature_selection_method,
                    max_features=self.config.max_features
                )
                self.feature_engineer = create_unified_feature_engineer(feature_config)
                tprint_success("✅ Feature engineering initialized for NAS")

            # Initialize advanced evaluation
            if self.config.enable_advanced_metrics:
                tprint_debug("📊 Initializing advanced evaluation for NAS...")
                self.advanced_evaluator = create_advanced_evaluator()
                tprint_success("✅ Advanced evaluation initialized for NAS")

        except Exception as e:
            tprint_error(f"❌ Shared utilities initialization failed: {e}")
            raise

    def search_architectures(self, train_data: Tuple[np.ndarray, np.ndarray],
                           validation_data: Tuple[np.ndarray, np.ndarray],
                           test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> List[NASArchitecture]:
        """Search for optimal neural architectures using centralized NAS utilities."""
        try:
            tprint_info("🔍 Starting NAS architecture search with centralized utilities...")

            # Create search configuration
            search_config = ArchitectureSearchConfig(
                max_iterations=100,
                population_size=50,
                enable_parallel_processing=True,
                max_workers=4
            )

            # Initialize centralized NAS optimizer
            nas_engine = NASEngine()
            architecture_optimizer = ArchitectureSearchOptimizer(search_config)

            # Convert data to DataFrame format for centralized optimizer
            X_train, y_train = train_data
            X_val, y_val = validation_data

            # Create DataFrame for search
            train_df = pd.DataFrame(X_train)
            train_df['target'] = y_train

            # Define search space
            search_space = {
                'complexity': [1.0, 1.5, 2.0, 2.5, 3.0],
                'depth': [1, 2, 3, 4, 5],
                'width': [8, 16, 32, 64, 128],
                'activation': ['relu', 'tanh', 'sigmoid']
            }

            # Perform search using centralized optimizer
            results = nas_engine.search_architectures(
                data=train_df,
                search_space=search_space,
                optimization_method="bayesian_tpe",
                n_trials=50
            )

            if results and 'best_architecture' in results:
                tprint_success("✅ Architecture search completed successfully")
                # Convert results to NAS architecture format
                best_arch = NASArchitecture(
                    layers=[{'type': 'dense', 'units': results['best_architecture'].get('width', 32)}],
                    activation=results['best_architecture'].get('activation', 'relu'),
                    complexity=results['best_architecture'].get('complexity', 1.0)
                )
                return [best_arch]
            else:
                tprint_warning("⚠️ No architectures found, returning empty list")
                return []

        except Exception as e:
            tprint_error(f"❌ NAS architecture search failed: {e}")
            return []

    def _create_objective_functions(self, train_data: Tuple[np.ndarray, np.ndarray],
                                  validation_data: Tuple[np.ndarray, np.ndarray],
                                  test_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> List[Callable]:
        """Create objective functions for evolutionary search."""
        try:
            objective_functions = []

            # Accuracy objective
            def accuracy_objective(params):
                try:
                    architecture = self._create_architecture_from_params(params)
                    if architecture is None:
                        return 0.0

                    # Train and evaluate architecture
                    model = self._build_model(architecture)
                    if model is None:
                        return 0.0

                    # Simple evaluation (in practice, this would be more sophisticated)
                    X_train, y_train = train_data
                    X_val, y_val = validation_data

                    # Train model (simplified)
                    model.train()
                    # ... training code would go here ...

                    # Evaluate model
                    model.eval()
                    with torch.no_grad():
                        # ... evaluation code would go here ...
                        accuracy = 0.8  # Placeholder

                    return accuracy

                except Exception as e:
                    self.logger.warning(f"⚠️ Accuracy objective evaluation failed: {e}")
                    return 0.0

            # Efficiency objective
            def efficiency_objective(params):
                try:
                    architecture = self._create_architecture_from_params(params)
                    if architecture is None:
                        return 0.0

                    # Calculate efficiency based on parameters and performance
                    param_count = architecture.parameters_count
                    efficiency = 1.0 / (1.0 + param_count / 1000000.0)  # Normalize
                    return efficiency

                except Exception as e:
                    self.logger.warning(f"⚠️ Efficiency objective evaluation failed: {e}")
                    return 0.0

            # Complexity objective (minimize)
            def complexity_objective(params):
                try:
                    architecture = self._create_architecture_from_params(params)
                    if architecture is None:
                        return 1.0  # High complexity penalty

                    # Calculate complexity score
                    complexity = architecture.complexity_score
                    return complexity

                except Exception as e:
                    self.logger.warning(f"⚠️ Complexity objective evaluation failed: {e}")
                    return 1.0

            # Regime awareness objective
            def regime_awareness_objective(params):
                try:
                    architecture = self._create_architecture_from_params(params)
                    if architecture is None:
                        return 0.0

                    # Calculate regime awareness
                    regime_awareness = architecture.regime_accuracy
                    return regime_awareness

                except Exception as e:
                    self.logger.warning(f"⚠️ Regime awareness objective evaluation failed: {e}")
                    return 0.0

            objective_functions.extend([
                accuracy_objective,
                efficiency_objective,
                complexity_objective,
                regime_awareness_objective
            ])

            return objective_functions

        except Exception as e:
            self.logger.error(f"❌ Objective functions creation failed: {e}")
            return []

    def _create_architecture_parameter_space(self) -> Dict[str, Any]:
        """Create parameter space for neural architectures."""
        return {
            'num_layers': {'type': 'integer', 'min': 2, 'max': 10},
            'hidden_size': {'type': 'integer', 'min': 32, 'max': 512},
            'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'sigmoid', 'gelu']},
            'dropout_rate': {'type': 'continuous', 'min': 0.0, 'max': 0.5},
            'learning_rate': {'type': 'continuous', 'min': 1e-5, 'max': 1e-2},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd', 'rmsprop']}
        }

    def _create_architecture_from_params(self, params: Dict[str, Any]) -> Optional[NASArchitecture]:
        """Create NAS architecture from parameters."""
        try:
            # Extract parameters
            num_layers = params.get('num_layers', 3)
            hidden_size = params.get('hidden_size', 128)
            activation = params.get('activation', 'relu')
            dropout_rate = params.get('dropout_rate', 0.1)
            learning_rate = params.get('learning_rate', 0.001)
            batch_size = params.get('batch_size', 32)
            optimizer = params.get('optimizer', 'adam')

            # Create architecture layers
            layers = []
            for i in range(num_layers):
                layer = {
                    'type': 'linear',
                    'input_size': hidden_size if i > 0 else 784,  # Assuming input size
                    'output_size': hidden_size,
                    'activation': activation,
                    'dropout': dropout_rate
                }
                layers.append(layer)

            # Calculate parameters count
            parameters_count = sum(
                layer['input_size'] * layer['output_size'] for layer in layers
            )

            # Create architecture
            architecture = NASArchitecture(
                layers=layers,
                parameters_count=parameters_count,
                fitness_score=0.0,
                complexity_score=num_layers * hidden_size / 1000.0,
                efficiency_score=1.0 / (1.0 + parameters_count / 1000000.0),
                regime_accuracy=0.0,
                economic_significance=0.0,
                trading_viability=0.0,
                architecture_id=f"arch_{len(self.population)}"
            )

            return architecture

        except Exception as e:
            self.logger.warning(f"⚠️ Architecture creation failed: {e}")
            return None

    def _build_model(self, architecture: NASArchitecture) -> Optional[nn.Module]:
        """Build PyTorch model from architecture."""
        try:
            # This is a simplified model builder
            # In practice, this would be more sophisticated
            class SimpleModel(nn.Module):
                def __init__(self, layers):
                    super().__init__()
                    self.layers = nn.ModuleList()
                    for layer in layers:
                        self.layers.append(nn.Linear(layer['input_size'], layer['output_size']))

                def forward(self, x):
                    for layer in self.layers:
                        x = layer(x)
                    return x

            model = SimpleModel(architecture.layers)
            return model

        except Exception as e:
            self.logger.warning(f"⚠️ Model building failed: {e}")
            return None

    def _convert_to_nas_architectures(self, evolutionary_result: EvolutionaryResult) -> List[NASArchitecture]:
        """Convert evolutionary search results to NAS architectures."""
        try:
            nas_architectures = []

            for individual in evolutionary_result.pareto_front:
                architecture = self._create_architecture_from_params(individual.parameters)
                if architecture is not None:
                    # Set objectives as scores
                    if len(individual.objectives) >= 4:
                        architecture.fitness_score = individual.objectives[0]  # Accuracy
                        architecture.efficiency_score = individual.objectives[1]  # Efficiency
                        architecture.complexity_score = individual.objectives[2]  # Complexity
                        architecture.regime_accuracy = individual.objectives[3]  # Regime awareness

                    nas_architectures.append(architecture)

            return nas_architectures

        except Exception as e:
            self.logger.error(f"❌ Architecture conversion failed: {e}")
            return []

    def _evaluate_architectures_with_advanced_metrics(self):
        """Evaluate architectures using advanced metrics."""
        try:
            if not self.config.enable_advanced_metrics or not SHARED_UTILS_AVAILABLE:
                return

            tprint_debug("📊 Evaluating architectures with advanced metrics...")

            for architecture in self.best_architectures:
                # Create dummy predictions and targets for evaluation
                # In practice, these would come from actual model evaluation
                predictions = np.random.randint(0, 2, 100)
                targets = np.random.randint(0, 2, 100)
                returns = np.random.normal(0.001, 0.02, 100)

                # Evaluate with advanced metrics
                eval_result = self.advanced_evaluator.evaluate(
                    predictions, targets, returns, self.config.regime_labels
                )

                if eval_result.success:
                    # Update architecture with advanced metrics
                    architecture.economic_significance = eval_result.economic_score
                    architecture.trading_viability = eval_result.trading_score

                    tprint_debug(f"   Architecture {architecture.architecture_id}:")
                    tprint_debug(f"     Economic significance: {architecture.economic_significance:.4f}")
                    tprint_debug(f"     Trading viability: {architecture.trading_viability:.4f}")

            tprint_success("✅ Advanced metrics evaluation completed")

        except Exception as e:
            tprint_error(f"❌ Advanced metrics evaluation failed: {e}")

    def _fallback_architecture_search(self, train_data: Tuple[np.ndarray, np.ndarray],
                                     validation_data: Tuple[np.ndarray, np.ndarray],
                                     test_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> List[NASArchitecture]:
        """Fallback architecture search when shared utilities are not available."""
        try:
            tprint_warning("⚠️ Using fallback architecture search")

            # Simple random architecture generation
            architectures = []
            for i in range(self.config.population_size):
                params = {
                    'num_layers': np.random.randint(2, 6),
                    'hidden_size': np.random.randint(32, 256),
                    'activation': np.random.choice(['relu', 'tanh', 'sigmoid']),
                    'dropout_rate': np.random.uniform(0.0, 0.3),
                    'learning_rate': np.random.uniform(1e-4, 1e-2),
                    'batch_size': np.random.choice([16, 32, 64]),
                    'optimizer': np.random.choice(['adam', 'sgd'])
                }

                architecture = self._create_architecture_from_params(params)
                if architecture is not None:
                    # Set random scores
                    architecture.fitness_score = np.random.random()
                    architecture.efficiency_score = np.random.random()
                    architecture.complexity_score = np.random.random()
                    architecture.regime_accuracy = np.random.random()
                    architecture.economic_significance = np.random.random()
                    architecture.trading_viability = np.random.random()

                    architectures.append(architecture)

            return architectures

        except Exception as e:
            tprint_error(f"❌ Fallback architecture search failed: {e}")
            return []

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        try:
            if not self.best_architectures:
                return {
                    'total_architectures': 0,
                    'best_fitness': 0.0,
                    'average_complexity': 0.0,
                    'search_success': False
                }

            fitness_scores = [arch.fitness_score for arch in self.best_architectures]
            complexity_scores = [arch.complexity_score for arch in self.best_architectures]

            return {
                'total_architectures': len(self.best_architectures),
                'best_fitness': max(fitness_scores),
                'average_fitness': np.mean(fitness_scores),
                'best_complexity': min(complexity_scores),
                'average_complexity': np.mean(complexity_scores),
                'search_success': True
            }

        except Exception as e:
            self.logger.error(f"❌ Search statistics calculation failed: {e}")
            return {'search_success': False, 'error': str(e)}

def demonstrate_nas_shared_utils_integration():
    """Demonstrate NAS integration with shared utilities."""
    tprint_info("🎯 Demonstrating NAS Integration with Shared Utilities")
    tprint_info("=" * 60)

    if not SHARED_UTILS_AVAILABLE:
        tprint_warning("⚠️ Shared utilities not available")
        return

    # Create sample data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create regime labels
    regime_labels = np.random.choice([0, 1, 2], size=len(y_train), p=[0.4, 0.4, 0.2])

    # Configure NAS search
    config = NASSearchConfig(
        population_size=20,
        max_generations=10,
        enable_feature_engineering=True,
        enable_advanced_metrics=True,
        regime_labels=regime_labels
    )

    # Create NAS integration
    nas_integration = NASSharedUtilsIntegration(config)

    # Run architecture search
    tprint_info("🔍 Running NAS architecture search...")
    start_time = time.time()

    best_architectures = nas_integration.search_architectures(
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    )

    search_time = time.time() - start_time

    if best_architectures:
        tprint_success(f"✅ NAS search completed in {search_time:.2f}s")
        tprint_info(f"   Found {len(best_architectures)} architectures")

        # Show best architectures
        for i, arch in enumerate(best_architectures[:3]):  # Show top 3
            tprint_info(f"   Architecture {i+1}:")
            tprint_info(f"     Fitness: {arch.fitness_score:.4f}")
            tprint_info(f"     Efficiency: {arch.efficiency_score:.4f}")
            tprint_info(f"     Complexity: {arch.complexity_score:.4f}")
            tprint_info(f"     Regime Accuracy: {arch.regime_accuracy:.4f}")
            tprint_info(f"     Economic Significance: {arch.economic_significance:.4f}")
            tprint_info(f"     Trading Viability: {arch.trading_viability:.4f}")

        # Show search statistics
        stats = nas_integration.get_search_statistics()
        tprint_info("📊 Search Statistics:")
        for key, value in stats.items():
            tprint_info(f"   {key}: {value}")
    else:
        tprint_warning("⚠️ No architectures found")

    tprint_success("🎉 NAS Shared Utils Integration demonstration completed!")

if __name__ == "__main__":
    demonstrate_nas_shared_utils_integration()
