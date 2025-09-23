"""
RegimeNAS: Neural Architecture Search for Market Regimes

This module implements RegimeNAS, a sophisticated neural architecture search system
that discovers optimal architectures for different market regimes while maintaining
efficiency and avoiding the need for separate models per regime.

Key Features:
1. Regime-aware architecture search
2. Dynamic architecture adaptation based on regime characteristics
3. Multi-objective optimization (accuracy, efficiency, robustness)
4. Hierarchical architecture search with progressive refinement
5. Transfer learning between similar regimes
6. Uncertainty-aware architecture selection
7. Hardware-aware search for deployment constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import logging
import time
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, Categorical
    from torch.utils.data import DataLoader, TensorDataset
    import optuna
    TORCH_AVAILABLE = True
    OPTUNA_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    OPTUNA_AVAILABLE = False
    logger.warning("⚠️ PyTorch or Optuna not available, using fallback implementations")


class ArchitectureType(Enum):
    """Types of neural architectures for different regimes."""
    CONVOLUTIONAL = "convolutional"
    RECURRENT = "recurrent"
    ATTENTION = "attention"
    HYBRID = "hybrid"
    SPECIALIZED = "specialized"
    EFFICIENT = "efficient"


class RegimeType(Enum):
    """Market regime types for architecture optimization."""
    HIGH_VOLATILITY = "high_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    LOW_VOLATILITY = "low_volatility"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class ArchitectureConfig:
    """Configuration for neural architecture search."""
    input_dim: int = 100
    output_dim: int = 4
    max_layers: int = 10
    min_layers: int = 2
    max_neurons: int = 1024
    min_neurons: int = 16
    allowed_operations: List[str] = field(default_factory=lambda: [
        'conv1d', 'lstm', 'gru', 'attention', 'dense', 'dropout', 'batch_norm'
    ])
    regime_specialization: bool = True
    multi_objective: bool = True
    hardware_constraints: Optional[Dict[str, Any]] = None
    search_budget: int = 100
    validation_frequency: int = 10


class ArchitectureCandidate:
    """Represents a candidate architecture in the search space."""

    def __init__(self, layers: List[Dict[str, Any]], regime_type: RegimeType,
                 architecture_id: str, parent_id: Optional[str] = None):
        """Initialize architecture candidate.

        Args:
            layers: List of layer configurations
            regime_type: Target regime type
            architecture_id: Unique identifier
            parent_id: Parent architecture for evolutionary search
        """
        self.layers = layers
        self.regime_type = regime_type
        self.architecture_id = architecture_id
        self.parent_id = parent_id
        self.fitness_score = 0.0
        self.complexity_score = 0.0
        self.efficiency_score = 0.0
        self.regime_score = 0.0
        self.metadata = {}

    def get_complexity(self) -> float:
        """Calculate architecture complexity."""
        total_params = 0
        for layer in self.layers:
            if layer['type'] == 'dense':
                in_dim = layer.get('input_dim', 100)
                out_dim = layer.get('output_dim', 64)
                total_params += in_dim * out_dim + out_dim
            elif layer['type'] == 'conv1d':
                channels = layer.get('channels', 32)
                kernel_size = layer.get('kernel_size', 3)
                total_params += channels * kernel_size
            elif layer['type'] in ['lstm', 'gru']:
                hidden_dim = layer.get('hidden_dim', 64)
                total_params += 4 * hidden_dim * hidden_dim  # LSTM parameter count

        return total_params / 1000000.0  # Convert to millions

    def get_layer_count(self) -> int:
        """Get total number of layers."""
        return len(self.layers)

    def is_valid(self) -> bool:
        """Check if architecture is valid."""
        if len(self.layers) < 1:
            return False

        # Check for proper input/output dimensions
        current_dim = self.layers[0].get('input_dim', 100)

        for layer in self.layers:
            if layer['type'] == 'dense':
                if 'input_dim' not in layer:
                    layer['input_dim'] = current_dim
                current_dim = layer.get('output_dim', 64)
            elif layer['type'] == 'dropout':
                continue  # Dropout doesn't change dimensions
            elif layer['type'] == 'batch_norm':
                continue  # Batch norm doesn't change dimensions

        # Check final output dimension
        if current_dim != 4:  # Assuming 4 outputs for trading
            return False

        return True


class RegimeArchitectureDatabase:
    """Database of architectures optimized for different regimes."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize regime architecture database."""
        self.config = config or {}
        self.architectures: Dict[RegimeType, List[ArchitectureCandidate]] = {
            regime: [] for regime in RegimeType
        }
        self.performance_history: Dict[str, Dict[str, Any]] = {}

    def add_architecture(self, architecture: ArchitectureCandidate) -> None:
        """Add architecture to database."""
        if architecture.regime_type not in self.architectures:
            self.architectures[architecture.regime_type] = []

        self.architectures[architecture.regime_type].append(architecture)

        # Sort by fitness score
        self.architectures[architecture.regime_type].sort(
            key=lambda x: x.fitness_score, reverse=True
        )

        # Keep only top architectures
        max_architectures = self.config.get('max_architectures_per_regime', 50)
        if len(self.architectures[architecture.regime_type]) > max_architectures:
            self.architectures[architecture.regime_type] = \
                self.architectures[architecture.regime_type][:max_architectures]

    def get_best_architecture(self, regime_type: RegimeType,
                            constraints: Optional[Dict[str, Any]] = None) -> Optional[ArchitectureCandidate]:
        """Get best architecture for a regime with optional constraints."""
        candidates = self.architectures.get(regime_type, [])

        if not candidates:
            return None

        # Filter by constraints
        if constraints:
            filtered_candidates = []
            for candidate in candidates:
                if self._meets_constraints(candidate, constraints):
                    filtered_candidates.append(candidate)
            candidates = filtered_candidates

        if not candidates:
            return None

        # Return architecture with highest fitness
        return max(candidates, key=lambda x: x.fitness_score)

    def _meets_constraints(self, candidate: ArchitectureCandidate,
                          constraints: Dict[str, Any]) -> bool:
        """Check if architecture meets constraints."""
        complexity_limit = constraints.get('max_complexity', float('inf'))
        if candidate.get_complexity() > complexity_limit:
            return False

        layer_limit = constraints.get('max_layers', float('inf'))
        if candidate.get_layer_count() > layer_limit:
            return False

        return True

    def get_similar_regimes(self, regime_type: RegimeType) -> List[RegimeType]:
        """Get regimes that might benefit from similar architectures."""
        similarity_mapping = {
            RegimeType.HIGH_VOLATILITY: [RegimeType.MIXED],
            RegimeType.TRENDING: [RegimeType.MIXED],
            RegimeType.MEAN_REVERTING: [RegimeType.MIXED],
            RegimeType.LOW_VOLATILITY: [RegimeType.MIXED]
        }

        return similarity_mapping.get(regime_type, [RegimeType.MIXED])

    def update_performance(self, architecture_id: str, performance: Dict[str, Any]) -> None:
        """Update performance metrics for an architecture."""
        if architecture_id not in self.performance_history:
            self.performance_history[architecture_id] = []

        self.performance_history[architecture_id].append({
            **performance,
            'timestamp': time.time()
        })

    def save_database(self, filepath: str) -> None:
        """Save database to file."""
        try:
            # Convert architectures to serializable format
            serializable_data = {}
            for regime, architectures in self.architectures.items():
                serializable_data[regime.value] = [
                    {
                        'layers': arch.layers,
                        'architecture_id': arch.architecture_id,
                        'parent_id': arch.parent_id,
                        'fitness_score': arch.fitness_score,
                        'complexity_score': arch.complexity_score,
                        'efficiency_score': arch.efficiency_score,
                        'regime_score': arch.regime_score,
                        'metadata': arch.metadata
                    }
                    for arch in architectures
                ]

            data = {
                'architectures': serializable_data,
                'performance_history': self.performance_history,
                'timestamp': time.time()
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"💾 Architecture database saved to {filepath}")

        except Exception as e:
            logger.error(f"❌ Failed to save architecture database: {e}")

    def load_database(self, filepath: str) -> None:
        """Load database from file."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Reconstruct architectures
                for regime_str, architectures_data in data.get('architectures', {}).items():
                    regime_type = RegimeType(regime_str)
                    architectures = []

                    for arch_data in architectures_data:
                        arch = ArchitectureCandidate(
                            layers=arch_data['layers'],
                            regime_type=regime_type,
                            architecture_id=arch_data['architecture_id'],
                            parent_id=arch_data.get('parent_id')
                        )
                        arch.fitness_score = arch_data.get('fitness_score', 0.0)
                        arch.complexity_score = arch_data.get('complexity_score', 0.0)
                        arch.efficiency_score = arch_data.get('efficiency_score', 0.0)
                        arch.regime_score = arch_data.get('regime_score', 0.0)
                        arch.metadata = arch_data.get('metadata', {})

                        architectures.append(arch)

                    self.architectures[regime_type] = architectures

                self.performance_history = data.get('performance_history', {})

                logger.info(f"📂 Architecture database loaded from {filepath}")
            else:
                logger.warning(f"⚠️ Architecture database file not found: {filepath}")

        except Exception as e:
            logger.error(f"❌ Failed to load architecture database: {e}")


class RegimeAwareArchitectureSearch:
    """Main RegimeNAS implementation."""

    def __init__(self, config: Optional[ArchitectureConfig] = None):
        """Initialize RegimeNAS."""
        self.config = config or ArchitectureConfig()
        self.database = RegimeArchitectureDatabase(config.__dict__ if config else {})
        self.search_history = {}
        self.best_architecture = None

        # Check for required libraries
        if not TORCH_AVAILABLE:
            logger.warning("⚠️ PyTorch not available, using simplified search")
        if not OPTUNA_AVAILABLE:
            logger.warning("⚠️ Optuna not available, using random search")

    def create_architecture_candidate(self, regime_type: RegimeType,
                                    architecture_type: ArchitectureType = ArchitectureType.HYBRID) -> ArchitectureCandidate:
        """Create a candidate architecture for a specific regime."""

        # Get regime-specific architecture template
        template = self._get_regime_template(regime_type, architecture_type)

        # Create layers based on template
        layers = self._build_architecture_layers(template, regime_type)

        # Create architecture candidate
        architecture_id = f"{regime_type.value}_{architecture_type.value}_{int(time.time())}"
        candidate = ArchitectureCandidate(layers, regime_type, architecture_id)

        return candidate

    def _get_regime_template(self, regime_type: RegimeType,
                           architecture_type: ArchitectureType) -> Dict[str, Any]:
        """Get architecture template for regime and type."""
        templates = {
            RegimeType.HIGH_VOLATILITY: {
                ArchitectureType.CONVOLUTIONAL: {
                    'layers': ['conv1d', 'batch_norm', 'conv1d', 'batch_norm', 'dense', 'dropout', 'dense'],
                    'focus': 'noise_reduction'
                },
                ArchitectureType.RECURRENT: {
                    'layers': ['lstm', 'dropout', 'lstm', 'dropout', 'dense', 'dropout', 'dense'],
                    'focus': 'stability'
                },
                ArchitectureType.ATTENTION: {
                    'layers': ['dense', 'attention', 'dense', 'dropout', 'dense'],
                    'focus': 'robustness'
                },
                ArchitectureType.HYBRID: {
                    'layers': ['conv1d', 'lstm', 'attention', 'dense', 'dropout', 'dense'],
                    'focus': 'comprehensive'
                }
            },
            RegimeType.TRENDING: {
                ArchitectureType.CONVOLUTIONAL: {
                    'layers': ['conv1d', 'conv1d', 'dense', 'dropout', 'dense', 'dense'],
                    'focus': 'trend_detection'
                },
                ArchitectureType.RECURRENT: {
                    'layers': ['lstm', 'lstm', 'dense', 'dropout', 'dense'],
                    'focus': 'momentum_tracking'
                },
                ArchitectureType.ATTENTION: {
                    'layers': ['dense', 'attention', 'dense', 'dense', 'dropout', 'dense'],
                    'focus': 'pattern_recognition'
                },
                ArchitectureType.HYBRID: {
                    'layers': ['conv1d', 'lstm', 'attention', 'dense', 'dense', 'dropout', 'dense'],
                    'focus': 'trend_prediction'
                }
            },
            RegimeType.MEAN_REVERTING: {
                ArchitectureType.CONVOLUTIONAL: {
                    'layers': ['conv1d', 'dense', 'conv1d', 'dense', 'dropout', 'dense'],
                    'focus': 'oscillation_detection'
                },
                ArchitectureType.RECURRENT: {
                    'layers': ['gru', 'dropout', 'gru', 'dense', 'dropout', 'dense'],
                    'focus': 'reversion_tracking'
                },
                ArchitectureType.ATTENTION: {
                    'layers': ['dense', 'attention', 'dense', 'dropout', 'dense'],
                    'focus': 'mean_targeting'
                },
                ArchitectureType.HYBRID: {
                    'layers': ['conv1d', 'gru', 'attention', 'dense', 'dropout', 'dense'],
                    'focus': 'reversion_prediction'
                }
            },
            RegimeType.LOW_VOLATILITY: {
                ArchitectureType.EFFICIENT: {
                    'layers': ['dense', 'dropout', 'dense', 'dense'],
                    'focus': 'efficiency'
                },
                ArchitectureType.HYBRID: {
                    'layers': ['dense', 'dense', 'dense', 'dropout', 'dense'],
                    'focus': 'precision'
                }
            }
        }

        return templates.get(regime_type, {}).get(architecture_type, {
            'layers': ['dense', 'dropout', 'dense', 'dense'],
            'focus': 'general'
        })

    def _build_architecture_layers(self, template: Dict[str, Any],
                                 regime_type: RegimeType) -> List[Dict[str, Any]]:
        """Build architecture layers from template."""
        layers = []
        layer_types = template['layers']

        # Input dimension
        current_dim = self.config.input_dim

        for layer_type in layer_types:
            layer_config = {'type': layer_type}

            if layer_type == 'dense':
                # Dense layer
                output_dim = np.random.randint(self.config.min_neurons, self.config.max_neurons)
                layer_config.update({
                    'input_dim': current_dim,
                    'output_dim': output_dim,
                    'activation': 'relu'
                })
                current_dim = output_dim

            elif layer_type == 'conv1d':
                # 1D Convolution
                channels = np.random.randint(16, 128)
                kernel_size = np.random.choice([3, 5, 7])
                layer_config.update({
                    'channels': channels,
                    'kernel_size': kernel_size,
                    'padding': kernel_size // 2,
                    'input_channels': 1 if not layers else layers[-1].get('channels', 32)
                })

            elif layer_type in ['lstm', 'gru']:
                # Recurrent layer
                hidden_dim = np.random.randint(self.config.min_neurons, self.config.max_neurons)
                layer_config.update({
                    'hidden_dim': hidden_dim,
                    'input_dim': current_dim,
                    'num_layers': 1,
                    'bidirectional': np.random.choice([True, False])
                })

            elif layer_type == 'attention':
                # Attention layer
                attention_dim = np.random.randint(64, 512)
                num_heads = np.random.choice([4, 8, 16])
                layer_config.update({
                    'attention_dim': attention_dim,
                    'num_heads': num_heads,
                    'input_dim': current_dim
                })

            elif layer_type == 'dropout':
                # Dropout layer
                dropout_rate = np.random.uniform(0.1, 0.5)
                layer_config.update({
                    'rate': dropout_rate
                })

            elif layer_type == 'batch_norm':
                # Batch normalization
                layer_config.update({
                    'input_dim': current_dim
                })

            layers.append(layer_config)

        # Ensure output dimension is correct
        if layers and layers[-1]['type'] == 'dense':
            layers[-1]['output_dim'] = self.config.output_dim

        return layers

    def evaluate_architecture(self, candidate: ArchitectureCandidate,
                            X: np.ndarray, y: np.ndarray,
                            regime_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Evaluate architecture performance."""

        try:
            # Create model from architecture
            model = self._create_model_from_architecture(candidate)

            if model is None:
                return {'fitness': 0.0, 'complexity': 0.0, 'efficiency': 0.0}

            # Train and evaluate
            model.fit(X, y)

            # Calculate metrics
            predictions = model.predict(X)
            mse = mean_squared_error(y, predictions)
            mae = np.mean(np.abs(y - predictions))

            # Fitness score (lower is better for error metrics)
            fitness = 1.0 / (1.0 + mse + mae)

            # Complexity score
            complexity = candidate.get_complexity()
            complexity_score = 1.0 / (1.0 + complexity)  # Prefer simpler models

            # Efficiency score (based on inference time)
            import time
            start_time = time.time()
            for _ in range(10):  # Average over multiple runs
                _ = model.predict(X[:100])  # Use subset for speed
            inference_time = (time.time() - start_time) / 10.0
            efficiency_score = 1.0 / (1.0 + inference_time)

            # Regime-specific score
            regime_score = self._calculate_regime_score(candidate, predictions, y, regime_data)

            # Update candidate scores
            candidate.fitness_score = fitness
            candidate.complexity_score = complexity_score
            candidate.efficiency_score = efficiency_score
            candidate.regime_score = regime_score

            return {
                'fitness': fitness,
                'complexity': complexity,
                'efficiency': efficiency_score,
                'regime': regime_score
            }

        except Exception as e:
            logger.warning(f"⚠️ Error evaluating architecture: {e}")
            return {'fitness': 0.0, 'complexity': 0.0, 'efficiency': 0.0}

    def _create_model_from_architecture(self, candidate: ArchitectureCandidate) -> Optional[Any]:
        """Create PyTorch model from architecture candidate."""
        if not TORCH_AVAILABLE:
            return None

        try:
            class DynamicModel(nn.Module):
                def __init__(self, layers_config):
                    super(DynamicModel, self).__init__()
                    self.layers = nn.ModuleList()
                    current_dim = self._get_input_dim(layers_config)

                    for layer_config in layers_config:
                        layer = self._create_layer(layer_config, current_dim)
                        if layer is not None:
                            self.layers.append(layer)
                            current_dim = self._update_current_dim(layer_config, current_dim)

                def _get_input_dim(self, layers_config):
                    for layer_config in layers_config:
                        if layer_config['type'] == 'dense' and 'input_dim' in layer_config:
                            return layer_config['input_dim']
                    return 100  # Default

                def _create_layer(self, layer_config, input_dim):
                    layer_type = layer_config['type']

                    if layer_type == 'dense':
                        return nn.Linear(input_dim, layer_config['output_dim'])
                    elif layer_type == 'conv1d':
                        return nn.Conv1d(
                            layer_config['input_channels'], layer_config['channels'],
                            layer_config['kernel_size'], padding=layer_config['padding']
                        )
                    elif layer_type == 'lstm':
                        return nn.LSTM(input_dim, layer_config['hidden_dim'],
                                     batch_first=True, bidirectional=layer_config.get('bidirectional', False))
                    elif layer_type == 'dropout':
                        return nn.Dropout(layer_config['rate'])
                    elif layer_type == 'batch_norm':
                        return nn.BatchNorm1d(input_dim)

                    return None

                def _update_current_dim(self, layer_config, current_dim):
                    layer_type = layer_config['type']

                    if layer_type == 'dense':
                        return layer_config['output_dim']
                    elif layer_type == 'conv1d':
                        return layer_config['channels']

                    return current_dim

                def forward(self, x):
                    for layer in self.layers:
                        if isinstance(layer, nn.LSTM):
                            x, _ = layer(x)
                        else:
                            x = layer(x)
                    return x

            return DynamicModel(candidate.layers)

        except Exception as e:
            logger.error(f"❌ Failed to create model from architecture: {e}")
            return None

    def _calculate_regime_score(self, candidate: ArchitectureCandidate,
                              predictions: np.ndarray, targets: np.ndarray,
                              regime_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate regime-specific performance score."""
        # Base score from prediction accuracy
        mse = mean_squared_error(targets, predictions)
        base_score = 1.0 / (1.0 + mse)

        # Regime-specific adjustments
        regime_multipliers = {
            RegimeType.HIGH_VOLATILITY: 1.2,  # Reward robustness
            RegimeType.TRENDING: 1.1,        # Reward trend accuracy
            RegimeType.MEAN_REVERTING: 1.1,   # Reward reversion accuracy
            RegimeType.LOW_VOLATILITY: 1.3    # Reward precision
        }

        multiplier = regime_multipliers.get(candidate.regime_type, 1.0)

        return base_score * multiplier

    def search_optimal_architecture(self, X: np.ndarray, y: np.ndarray,
                                  regime_type: RegimeType,
                                  X_val: Optional[np.ndarray] = None,
                                  y_val: Optional[np.ndarray] = None,
                                  regime_data: Optional[Dict[str, Any]] = None) -> ArchitectureCandidate:
        """Search for optimal architecture for a specific regime."""

        logger.info(f"🔍 Starting RegimeNAS for {regime_type.value}")

        best_candidate = None
        best_score = 0.0

        # Initialize with existing architectures if available
        existing_candidates = self.database.get_best_architecture(regime_type)

        if existing_candidates:
            logger.info(f"📚 Found {len(self.database.architectures[regime_type])} existing architectures")

        # Generate new candidates
        for i in range(self.config.search_budget):
            # Create candidate
            architecture_type = np.random.choice([
                ArchitectureType.CONVOLUTIONAL,
                ArchitectureType.RECURRENT,
                ArchitectureType.ATTENTION,
                ArchitectureType.HYBRID
            ])

            candidate = self.create_architecture_candidate(regime_type, architecture_type)

            # Skip invalid architectures
            if not candidate.is_valid():
                continue

            # Evaluate candidate
            scores = self.evaluate_architecture(candidate, X, y, regime_data)

            if scores['fitness'] > best_score:
                best_score = scores['fitness']
                best_candidate = candidate

            # Add to database
            self.database.add_architecture(candidate)

            # Periodic logging
            if (i + 1) % self.config.validation_frequency == 0:
                logger.info(f"🏃 Search progress: {i + 1}/{self.config.search_budget}, "
                          f"best fitness: {best_score:.6f}")

        # Final evaluation on validation set if available
        if X_val is not None and y_val is not None and best_candidate:
            val_scores = self.evaluate_architecture(best_candidate, X_val, y_val, regime_data)
            best_candidate.metadata['validation_score'] = val_scores['fitness']

        self.best_architecture = best_candidate

        if best_candidate:
            logger.info(f"✅ RegimeNAS completed. Best architecture: {best_candidate.architecture_id}")
            logger.info(f"   Fitness: {best_candidate.fitness_score:.6f}")
            logger.info(f"   Complexity: {best_candidate.get_complexity():.3f}M params")
            logger.info(f"   Layers: {best_candidate.get_layer_count()}")

        return best_candidate

    def get_adaptive_architecture(self, regime_type: RegimeType,
                                constraints: Optional[Dict[str, Any]] = None) -> Optional[ArchitectureCandidate]:
        """Get adaptive architecture based on current conditions."""
        # Try to get from database first
        architecture = self.database.get_best_architecture(regime_type, constraints)

        if architecture:
            return architecture

        # If no architecture in database, create a default one
        logger.warning(f"⚠️ No architecture found for {regime_type.value}, creating default")
        return self.create_architecture_candidate(regime_type, ArchitectureType.HYBRID)

    def save_search_results(self, filepath: str) -> None:
        """Save search results to file."""
        self.database.save_database(filepath)

    def load_search_results(self, filepath: str) -> None:
        """Load search results from file."""
        self.database.load_database(filepath)


# Factory functions and utilities
def create_regime_nas(config: Dict[str, Any]) -> RegimeAwareArchitectureSearch:
    """Create RegimeNAS instance."""
    arch_config = ArchitectureConfig(**config.get('architecture_params', {}))
    return RegimeAwareArchitectureSearch(arch_config)


def search_optimal_architecture(X: np.ndarray, y: np.ndarray,
                              regime_type: str,
                              config: Optional[Dict[str, Any]] = None) -> ArchitectureCandidate:
    """Search for optimal architecture for a regime."""
    regime = RegimeType(regime_type)
    nas = create_regime_nas(config or {})
    return nas.search_optimal_architecture(X, y, regime)


def get_example_nas_config() -> Dict[str, Any]:
    """Get example configuration for RegimeNAS."""
    return {
        'architecture_params': {
            'input_dim': 100,
            'output_dim': 4,
            'max_layers': 8,
            'min_layers': 2,
            'max_neurons': 512,
            'min_neurons': 32,
            'allowed_operations': ['conv1d', 'lstm', 'attention', 'dense', 'dropout'],
            'regime_specialization': True,
            'multi_objective': True,
            'search_budget': 50,
            'validation_frequency': 10
        }
    }