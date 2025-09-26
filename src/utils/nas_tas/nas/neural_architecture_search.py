from __future__ import annotations

"""
Neural Architecture Search (NAS) for ML Common

This module provides comprehensive Neural Architecture Search capabilities
specifically designed for financial time series and trading models.

Key Features:
- Evolutionary architecture search
- Reinforcement learning-based search
- Multi-objective optimization (accuracy + efficiency)
- Regime-aware architecture adaptation
- Integration with existing ML pipeline
"""

import json
import hashlib
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from sklearn.base import TransformerMixin
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - sklearn is an optional dependency in some deployments
    SKLEARN_AVAILABLE = False
    Pipeline = None  # type: ignore
    TransformerMixin = Any  # type: ignore
    PCA = None  # type: ignore
    TfidfVectorizer = None  # type: ignore
    MinMaxScaler = None  # type: ignore
    RobustScaler = None  # type: ignore
    StandardScaler = None  # type: ignore

# Neural network imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    models = None
    optimizers = None

# Optimization imports
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Import new Bayesian TPE optimizer
from src.utils.nas_tas.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    BayesianTPEConfig,
    optimize_with_bayesian_tpe
)
from src.utils.nas_tas.common_constants import (
    DATA_AWARE_PARAMETER_CAPACITY,
    ESTIMATED_INPUT_FEATURES,
    RECOMMENDED_HIDDEN_SIZE_OPTIONS,
    RECOMMENDED_MAX_LAYERS,
    RECOMMENDED_MAX_UNITS,
    RECOMMENDED_MIN_LAYERS,
    RECOMMENDED_MIN_UNITS,
)

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureConfig:
    """Configuration for neural architecture search."""
    
    # Search space
    min_layers: int = RECOMMENDED_MIN_LAYERS
    max_layers: int = RECOMMENDED_MAX_LAYERS
    min_units: int = RECOMMENDED_MIN_UNITS
    max_units: int = RECOMMENDED_MAX_UNITS
    activation_functions: List[str] = field(
        default_factory=lambda: ['relu', 'tanh', 'swish', 'gelu', 'sigmoid']
    )
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4])
    layer_types: List[str] = field(
        default_factory=lambda: [
            'dense', 'lstm', 'gru', 'conv1d', 'conv2d', 'batchnorm', 'dropout', 'self_attention'
        ]
    )
    conditional_layers: Dict[str, List[str]] = field(
        default_factory=lambda: {
            'dropout': ['dense', 'lstm', 'gru', 'conv1d', 'conv2d'],
            'batchnorm': ['dense', 'conv1d', 'conv2d'],
            'self_attention': ['dense', 'lstm', 'gru']
        }
    )

    # Search parameters
    n_trials: int = 50
    timeout_seconds: int = 3600  # 1 hour
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    max_search_time: Optional[float] = None
    enable_successive_halving: bool = True
    halving_factor: int = 3
    min_resource_epochs: int = 5

    # Multi-objective optimization
    objectives: List[str] = field(default_factory=lambda: ['accuracy', 'efficiency', 'robustness'])
    objective_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    multi_objective_strategy: str = 'pareto'

    # Regime awareness
    enable_regime_awareness: bool = True
    regime_adaptation_strength: float = 0.3

    # Performance
    n_jobs: int = 1
    memory_limit_gb: float = 8.0

    # Data handling
    random_seed: Optional[int] = None
    auto_detect_data_type: bool = True
    supported_data_types: Tuple[str, ...] = ('tabular', 'time_series', 'image', 'text')
    enable_pipeline_search: bool = True
    preprocessing_components: Dict[str, List[str]] = field(
        default_factory=lambda: {
            'tabular': ['standard_scaler', 'minmax_scaler', 'robust_scaler', 'pca'],
            'time_series': ['standard_scaler'],
            'text': ['tfidf'],
            'image': []
        }
    )
    preprocessing_combinations: int = 3
    allow_custom_preprocessors: bool = True


@dataclass
class ArchitectureCandidate:
    """A candidate neural architecture."""

    # Architecture definition
    layers: List[Dict[str, Any]]  # List of layer configurations
    total_params: int
    estimated_flops: int
    data_type: str = 'tabular'
    preprocessing_steps: Optional[List[str]] = None

    # Performance metrics
    accuracy: float = 0.0
    efficiency_score: float = 0.0
    robustness_score: float = 0.0
    overall_score: float = 0.0
    
    # Training info
    training_time: float = 0.0
    convergence_epochs: int = 0
    final_loss: float = 0.0
    
    # Regime performance
    regime_performance: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    trial_number: int = 0


class ArchitectureSearchSpace:
    """Defines the search space for neural architectures."""
    
    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.logger = logger.getChild('ArchitectureSearchSpace')
    
    def sample_architecture(
        self,
        trial_number: int = 0,
        data_type: str = 'tabular',
        preprocessing_steps: Optional[List[str]] = None
    ) -> ArchitectureCandidate:
        """Sample a random architecture from the search space."""
        try:
            # Sample number of layers
            n_layers = np.random.randint(self.config.min_layers, self.config.max_layers + 1)

            # Sample layer configurations
            layers = []
            total_params = 0
            estimated_flops = 0

            allowed_layer_types = self._get_layer_types_for_data(data_type)

            for i in range(n_layers):
                # Sample layer type
                layer_type = self._sample_layer_type(layers, allowed_layer_types)

                # Sample layer parameters
                if layer_type == 'dense':
                    units = int(np.random.choice(RECOMMENDED_HIDDEN_SIZE_OPTIONS))
                    units = int(np.clip(units, self.config.min_units, self.config.max_units))
                    activation = np.random.choice(self.config.activation_functions)
                    dropout = np.random.choice(self.config.dropout_rates)

                    layer_config = {
                        'type': 'dense',
                        'units': units,
                        'activation': activation,
                        'dropout': dropout
                    }
                    
                    prev_units = (
                        ESTIMATED_INPUT_FEATURES if not layers
                        else layers[-1].get('units', ESTIMATED_INPUT_FEATURES)
                    )
                    layer_params = prev_units * units

                    total_params += layer_params
                    estimated_flops += layer_params * 2  # Simplified FLOP estimate

                elif layer_type in ['lstm', 'gru']:
                    units = int(np.random.choice(RECOMMENDED_HIDDEN_SIZE_OPTIONS))
                    units = int(np.clip(units, self.config.min_units, self.config.max_units))
                    return_sequences = i < n_layers - 1  # Only last layer returns sequences
                    dropout = np.random.choice(self.config.dropout_rates)

                    layer_config = {
                        'type': layer_type,
                        'units': units,
                        'return_sequences': return_sequences,
                        'dropout': dropout
                    }

                    # Estimate parameters for RNN layers
                    layer_params = 4 * units * units if layer_type == 'lstm' else 3 * units * units
                    total_params += layer_params
                    estimated_flops += layer_params * 4  # RNN operations are more expensive

                elif layer_type == 'conv1d':
                    filters = int(np.random.choice(RECOMMENDED_HIDDEN_SIZE_OPTIONS))
                    filters = int(np.clip(filters, self.config.min_units, self.config.max_units))
                    kernel_size = np.random.choice([1, 3, 5, 7])
                    activation = np.random.choice(self.config.activation_functions)

                    layer_config = {
                        'type': 'conv1d',
                        'filters': int(filters),
                        'kernel_size': int(kernel_size),
                        'activation': activation
                    }

                    layer_params = max(filters * kernel_size * ESTIMATED_INPUT_FEATURES, 0)
                    total_params += layer_params
                    estimated_flops += layer_params * 2

                elif layer_type == 'conv2d':
                    filters = int(np.random.choice(RECOMMENDED_HIDDEN_SIZE_OPTIONS))
                    filters = int(np.clip(filters, self.config.min_units, self.config.max_units))
                    kernel_size = tuple(np.random.choice([3, 5], size=2))
                    strides = tuple(np.random.choice([1, 2], size=2))
                    activation = np.random.choice(self.config.activation_functions)

                    layer_config = {
                        'type': 'conv2d',
                        'filters': int(filters),
                        'kernel_size': tuple(int(k) for k in kernel_size),
                        'strides': tuple(int(s) for s in strides),
                        'activation': activation
                    }

                    layer_params = max(filters * int(np.prod(kernel_size)) * ESTIMATED_INPUT_FEATURES, 0)
                    total_params += layer_params
                    estimated_flops += layer_params * 4

                elif layer_type == 'batchnorm':
                    momentum = float(np.random.uniform(0.8, 0.99))
                    epsilon = float(np.random.uniform(1e-5, 1e-3))
                    layer_config = {
                        'type': 'batchnorm',
                        'momentum': momentum,
                        'epsilon': epsilon
                    }

                    layer_params = layers[-1].get('units', 0) if layers else 0
                    total_params += max(layer_params, 0)
                    estimated_flops += max(layer_params, 0)

                elif layer_type == 'dropout':
                    rate = float(np.random.choice(self.config.dropout_rates))
                    layer_config = {
                        'type': 'dropout',
                        'rate': max(min(rate, 0.9), 0.0)
                    }

                elif layer_type == 'self_attention':
                    heads = int(np.random.choice([2, 4, 8]))
                    key_dim = int(np.random.choice(RECOMMENDED_HIDDEN_SIZE_OPTIONS))
                    key_dim = int(np.clip(key_dim, self.config.min_units, self.config.max_units))
                    dropout = float(np.random.choice(self.config.dropout_rates))

                    layer_config = {
                        'type': 'self_attention',
                        'heads': heads,
                        'key_dim': key_dim,
                        'dropout': max(min(dropout, 0.5), 0.0)
                    }

                    layer_params = max(heads * key_dim * ESTIMATED_INPUT_FEATURES, 0)
                    total_params += layer_params
                    estimated_flops += layer_params * 4

                else:
                    layer_config = {'type': layer_type}

                self._validate_layer_config(layer_config)

                layers.append(layer_config)

            # Create architecture candidate
            candidate = ArchitectureCandidate(
                layers=layers,
                total_params=total_params,
                estimated_flops=estimated_flops,
                trial_number=trial_number,
                data_type=data_type,
                preprocessing_steps=preprocessing_steps
            )

            if total_params > DATA_AWARE_PARAMETER_CAPACITY:
                self.logger.debug(
                    "Sampled architecture above data-aware parameter budget (%s > %s)",
                    total_params,
                    DATA_AWARE_PARAMETER_CAPACITY,
                )

            self.logger.debug(f"Sampled architecture with {n_layers} layers, {total_params} parameters")
            return candidate
            
        except Exception as e:
            self.logger.error(f"Architecture sampling failed: {e}")
            # Return minimal architecture as fallback
            return ArchitectureCandidate(
                layers=[{'type': 'dense', 'units': 64, 'activation': 'relu', 'dropout': 0.0}],
                total_params=1000,
                estimated_flops=2000,
                trial_number=trial_number,
                data_type=data_type,
                preprocessing_steps=preprocessing_steps
            )

    def _get_layer_types_for_data(self, data_type: str) -> List[str]:
        """Return allowed layer types for the detected data type."""
        candidates = list(self.config.layer_types)
        if data_type == 'text':
            return [lt for lt in candidates if lt != 'conv2d']
        if data_type == 'image':
            return [lt for lt in candidates if lt not in {'lstm', 'gru'}]
        return candidates

    def _sample_layer_type(self, existing_layers: List[Dict[str, Any]], allowed: List[str]) -> str:
        """Sample a layer type respecting conditional dependencies."""
        if not allowed:
            return 'dense'

        layer_type = str(np.random.choice(allowed))
        if layer_type in self.config.conditional_layers:
            valid_predecessors = self.config.conditional_layers[layer_type]
            if existing_layers:
                previous_type = existing_layers[-1]['type']
                if previous_type not in valid_predecessors:
                    fallback_choices = [
                        lt for lt in allowed if lt not in self.config.conditional_layers
                    ]
                    if fallback_choices:
                        layer_type = str(np.random.choice(fallback_choices))
                    else:
                        layer_type = str(np.random.choice(valid_predecessors or ['dense']))
            else:
                layer_type = str(np.random.choice(valid_predecessors or ['dense']))
        return layer_type

    def _validate_layer_config(self, layer_config: Dict[str, Any]) -> None:
        """Ensure layer configuration respects sensible bounds."""
        layer_type = layer_config.get('type', '')
        if layer_type == 'dense':
            units = layer_config.get('units', self.config.min_units)
            layer_config['units'] = int(np.clip(units, self.config.min_units, self.config.max_units))
            dropout = layer_config.get('dropout', 0.0)
            layer_config['dropout'] = float(np.clip(dropout, 0.0, 0.9))
        elif layer_type in {'lstm', 'gru'}:
            units = layer_config.get('units', self.config.min_units)
            layer_config['units'] = int(np.clip(units, self.config.min_units, self.config.max_units))
            layer_config['dropout'] = float(np.clip(layer_config.get('dropout', 0.0), 0.0, 0.9))
        elif layer_type.startswith('conv'):
            filters = layer_config.get('filters', self.config.min_units)
            layer_config['filters'] = int(np.clip(filters, self.config.min_units, self.config.max_units))
        elif layer_type == 'self_attention':
            heads = layer_config.get('heads', 2)
            layer_config['heads'] = int(np.clip(heads, 1, 16))
            key_dim = layer_config.get('key_dim', self.config.min_units)
            layer_config['key_dim'] = int(np.clip(key_dim, self.config.min_units, self.config.max_units))


class NeuralArchitectureSearch:
    """Main Neural Architecture Search implementation."""
    
    def __init__(self, config: ArchitectureConfig):
        """Initialize NAS."""
        self.config = config
        self.logger = logger.getChild('NeuralArchitectureSearch')
        self.search_space = ArchitectureSearchSpace(config)
        self.candidates = []
        self.best_candidate = None
        self.pareto_front: List[ArchitectureCandidate] = []
        self.data_metadata: Dict[str, Any] = {}

        self._set_random_seed()

        # Initialize framework
        self.framework = self._detect_framework()

        self.logger.info(f"✅ Neural Architecture Search initialized with {config.n_trials} trials")
    
    def _detect_framework(self) -> str:
        """Detect available deep learning framework."""
        if TORCH_AVAILABLE:
            return 'pytorch'
        elif TF_AVAILABLE:
            return 'tensorflow'
        else:
            raise ImportError("No deep learning framework available. Install PyTorch or TensorFlow.")

    def _set_random_seed(self) -> None:
        """Set deterministic random seeds when requested."""
        if self.config.random_seed is None:
            return

        seed = int(self.config.random_seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            import torch  # type: ignore

            torch.manual_seed(seed)
            if torch.cuda.is_available():  # pragma: no cover - GPU specific
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

        if TF_AVAILABLE:
            try:
                tf.random.set_seed(seed)  # type: ignore[attr-defined]
            except Exception:
                pass

    def _prepare_data_inputs(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any],
        y_val: Optional[Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, Optional[List[str]]]:
        """Normalize incoming dataset types and build preprocessing pipelines."""

        X_train_np = self._to_numpy(X_train)
        y_train_np = self._to_numpy(y_train, is_target=True)

        X_val_np = self._to_numpy(X_val) if X_val is not None else None
        y_val_np = self._to_numpy(y_val, is_target=True) if y_val is not None else None

        data_type = self._infer_data_type(X_train_np, original=X_train)

        preprocessing_steps = self._sample_preprocessing_steps(data_type)
        pipeline = self._build_data_pipeline(preprocessing_steps, data_type)

        # Perform validation split if needed
        if X_val_np is None or y_val_np is None:
            if SKLEARN_AVAILABLE:
                X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
                    X_train_np,
                    y_train_np,
                    test_size=self.config.validation_split,
                    random_state=self.config.random_seed or 42,
                    shuffle=data_type != 'time_series'
                )
            else:  # Fallback split without sklearn
                split_index = int(len(X_train_np) * (1 - self.config.validation_split))
                X_val_np = X_train_np[split_index:]
                y_val_np = y_train_np[split_index:]
                X_train_np = X_train_np[:split_index]
                y_train_np = y_train_np[:split_index]

        if pipeline is not None:
            X_train_np, y_train_np, pipeline = self._fit_transform_pipeline(
                pipeline, X_train_np, y_train_np
            )
            X_val_np = self._transform_pipeline(pipeline, X_val_np)

        self.data_metadata = {
            'data_type': data_type,
            'preprocessing_steps': preprocessing_steps,
            'pipeline': pipeline,
        }

        return X_train_np, y_train_np, X_val_np, y_val_np, data_type, preprocessing_steps

    def _to_numpy(self, data: Any, is_target: bool = False) -> np.ndarray:
        """Convert supported data containers into numpy arrays."""
        if data is None:
            return np.array([])

        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, pd.DataFrame):
            return data.values
        if isinstance(data, pd.Series):
            return data.values.reshape(-1, 1) if not is_target else data.values
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):  # type: ignore[name-defined]
            return data.detach().cpu().numpy()
        if TF_AVAILABLE and tf.is_tensor(data):  # type: ignore[attr-defined]
            return data.numpy()
        if isinstance(data, list):
            return np.array(data)

        # Unknown type -> attempt to convert via np.array
        try:
            return np.array(data)
        except Exception as exc:
            raise TypeError(f"Unsupported data type for conversion to numpy: {type(data)}") from exc

    def _infer_data_type(self, X: np.ndarray, original: Any = None) -> str:
        """Heuristically detect the dataset type."""
        if not self.config.auto_detect_data_type:
            return 'tabular'

        if original is not None and isinstance(original, (pd.Series, list)):
            if len(X.shape) == 1 and X.dtype == object:
                return 'text'

        if len(X.shape) == 3:
            return 'time_series'
        if len(X.shape) == 4:
            return 'image'
        if X.dtype == object:
            return 'text'
        return 'tabular'

    def _sample_preprocessing_steps(self, data_type: str) -> Optional[List[str]]:
        """Sample preprocessing components for the given data type."""
        if not self.config.enable_pipeline_search:
            return None

        if data_type in {'time_series', 'image'}:
            return None

        available = self.config.preprocessing_components.get(data_type, [])
        if not available:
            return None

        n_steps = min(len(available), max(1, self.config.preprocessing_combinations))
        selected = np.random.choice(available, size=n_steps, replace=False)
        return list(dict.fromkeys(selected))  # Preserve order without duplicates

    def _build_data_pipeline(
        self, steps: Optional[List[str]], data_type: str
    ) -> Optional[Pipeline]:
        """Create a scikit-learn pipeline from requested preprocessing steps."""
        if not SKLEARN_AVAILABLE or not steps:
            return None

        transformers: List[Tuple[str, TransformerMixin]] = []
        for step in steps:
            if step == 'standard_scaler' and StandardScaler is not None:
                transformers.append((step, StandardScaler()))
            elif step == 'minmax_scaler' and MinMaxScaler is not None:
                transformers.append((step, MinMaxScaler()))
            elif step == 'robust_scaler' and RobustScaler is not None:
                transformers.append((step, RobustScaler()))
            elif step == 'pca' and PCA is not None and data_type != 'text':
                transformers.append((step, PCA(n_components=0.95)))
            elif step == 'tfidf' and TfidfVectorizer is not None:
                transformers.append((step, TfidfVectorizer(max_features=1000)))

        if not transformers:
            return None

        return Pipeline(transformers)

    def _fit_transform_pipeline(
        self,
        pipeline: Pipeline,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Pipeline]:
        """Fit and transform the dataset with the pipeline."""
        try:
            transformed = pipeline.fit_transform(X_train, y_train)
            if hasattr(transformed, 'toarray'):
                transformed = transformed.toarray()
            return transformed, y_train, pipeline
        except Exception as exc:
            self.logger.warning(f"Preprocessing pipeline failed: {exc}")
            return X_train, y_train, pipeline

    def _transform_pipeline(self, pipeline: Pipeline, X_val: np.ndarray) -> np.ndarray:
        """Apply a fitted pipeline to validation data."""
        if pipeline is None:
            return X_val

        try:
            transformed = pipeline.transform(X_val)
            if hasattr(transformed, 'toarray'):
                transformed = transformed.toarray()
            return transformed
        except Exception as exc:
            self.logger.warning(f"Validation preprocessing failed: {exc}")
            return X_val

    def search(self,
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: Optional[np.ndarray] = None,
               y_val: Optional[np.ndarray] = None,
               regime_labels: Optional[np.ndarray] = None) -> ArchitectureCandidate:
        """
        Perform neural architecture search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            regime_labels: Regime labels for regime-aware search (optional)
            
        Returns:
            Best architecture candidate
        """
        self.logger.info("🚀 Starting Neural Architecture Search...")
        start_time = time.time()

        deadline = None
        if self.config.max_search_time:
            deadline = start_time + self.config.max_search_time
        elif self.config.timeout_seconds:
            deadline = start_time + self.config.timeout_seconds

        try:
            (
                X_train_np,
                y_train_np,
                X_val_np,
                y_val_np,
                data_type,
                preprocessing_steps
            ) = self._prepare_data_inputs(X_train, y_train, X_val, y_val)

            # Search for architectures
            if OPTUNA_AVAILABLE and self.config.multi_objective_strategy == 'weighted_sum':
                best_candidate = self._optuna_search(
                    X_train_np,
                    y_train_np,
                    X_val_np,
                    y_val_np,
                    data_type,
                    preprocessing_steps,
                    regime_labels,
                    deadline
                )
            else:
                best_candidate = self._random_search(
                    X_train_np,
                    y_train_np,
                    X_val_np,
                    y_val_np,
                    data_type,
                    preprocessing_steps,
                    regime_labels,
                    deadline
                )

            search_time = time.time() - start_time
            self.logger.info(f"✅ NAS completed in {search_time:.2f}s")
            self.logger.info(f"📊 Best architecture: {best_candidate.total_params} parameters, score: {best_candidate.overall_score:.4f}")

            return best_candidate
            
        except Exception as e:
            self.logger.error(f"Neural Architecture Search failed: {e}")
            raise
    
    def _optuna_search(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val: np.ndarray,
                      y_val: np.ndarray,
                      data_type: str,
                      preprocessing_steps: Optional[List[str]],
                      regime_labels: Optional[np.ndarray] = None,
                      deadline: Optional[float] = None) -> ArchitectureCandidate:
        """Perform architecture search using Bayesian TPE optimizer."""
        self.logger.info("🔍 Starting Bayesian TPE architecture search...")

        # Create search space for architecture optimization
        search_space = self._create_architecture_search_space()
        
        # Define objective function for Bayesian TPE optimizer
        def objective_function(params: Dict[str, Any], **kwargs) -> float:
            try:
                candidate = self._sample_architecture_from_params(
                    params,
                    data_type=data_type,
                    preprocessing_steps=preprocessing_steps
                )

                # Train and evaluate
                performance = self._train_and_evaluate_architecture(
                    candidate,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    regime_labels,
                    deadline
                )

                self._update_candidate_records(candidate, performance)

                return performance['overall_score']

            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return 0.0
        
        # Configure Bayesian TPE optimizer
        tpe_config = BayesianTPEConfig(
            n_trials=self.config.n_trials,
            timeout_seconds=self.config.timeout_seconds,
            enable_grid_search=True,
            coarse_grid_points=3,
            fine_grid_points=5,
            backend='optuna',
            enable_parallel=False,  # Sequential for architecture search
            enable_early_stopping=True,
            early_stopping_patience=self.config.early_stopping_patience,
            log_level='INFO'
        )
        
        # Run optimization using new unified optimizer
        self.logger.info("🎯 Starting Bayesian TPE optimization for neural architecture search")
        optimizer = BayesianTPEOptimizer(tpe_config)
        result = optimizer.optimize(objective_function, search_space)
        
        if not result.success:
            raise RuntimeError(f"Architecture search failed: {result.error_message}")
        
        # Get best candidate
        best_candidate = self._sample_architecture_from_params(
            result.best_params,
            data_type=data_type,
            preprocessing_steps=preprocessing_steps
        )

        # Train final model
        performance = self._train_and_evaluate_architecture(
            best_candidate,
            X_train,
            y_train,
            X_val,
            y_val,
            regime_labels,
            deadline
        )
        
        best_candidate.accuracy = performance['accuracy']
        best_candidate.efficiency_score = performance['efficiency_score']
        best_candidate.robustness_score = performance['robustness_score']
        best_candidate.overall_score = performance['overall_score']

        self.best_candidate = best_candidate
        self.logger.info(f"✅ Neural Architecture Search completed")
        self.logger.info(f"📊 Best score: {result.best_score:.4f}")
        self.logger.info(f"📊 Optimization time: {result.optimization_time:.2f}s")
        self.logger.info(f"📊 Trials: {result.n_trials}")
        
        return best_candidate

    def _create_architecture_search_space(self) -> Dict[str, Dict[str, Any]]:
        """Build generic search space for Bayesian optimization."""
        search_space = {
            'n_layers': {
                'type': 'int',
                'low': self.config.min_layers,
                'high': self.config.max_layers
            },
            'dense_units': {
                'type': 'int',
                'low': self.config.min_units,
                'high': self.config.max_units
            },
            'rnn_units': {
                'type': 'int',
                'low': self.config.min_units,
                'high': self.config.max_units
            },
            'conv_filters': {
                'type': 'int',
                'low': self.config.min_units,
                'high': self.config.max_units
            },
            'activation': {
                'type': 'categorical',
                'choices': self.config.activation_functions
            },
            'dropout_rate': {
                'type': 'categorical',
                'choices': self.config.dropout_rates
            }
        }

        layer_choices = list(self.config.layer_types)
        for layer_index in range(self.config.max_layers):
            search_space[f'layer_type_{layer_index}'] = {
                'type': 'categorical',
                'choices': layer_choices
            }

        return search_space
    
    def _random_search(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val: np.ndarray,
                      y_val: np.ndarray,
                      data_type: str,
                      preprocessing_steps: Optional[List[str]],
                      regime_labels: Optional[np.ndarray] = None,
                      deadline: Optional[float] = None) -> ArchitectureCandidate:
        """Perform random architecture search."""
        self.logger.info("🔍 Starting random architecture search...")

        best_candidate = None
        best_score = -np.inf

        for trial in range(self.config.n_trials):
            if deadline and time.time() >= deadline:
                self.logger.info("⏹️ Search deadline reached; stopping early.")
                break
            try:
                # Sample random architecture
                candidate = self.search_space.sample_architecture(
                    trial,
                    data_type=data_type,
                    preprocessing_steps=preprocessing_steps
                )

                # Train and evaluate
                performance = self._train_and_evaluate_architecture(
                    candidate,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    regime_labels,
                    deadline,
                    trial
                )

                # Update best candidate
                self._update_candidate_records(candidate, performance)
                best_candidate = self._select_best_candidate(best_candidate, candidate, performance)
                best_score = best_candidate.overall_score if best_candidate else best_score

                self.logger.debug(f"Trial {trial}: Score {performance['overall_score']:.4f}")

            except Exception as e:
                self.logger.warning(f"Trial {trial} failed: {e}")
                continue

        if best_candidate is None:
            raise RuntimeError("No successful architecture found")

        self.best_candidate = best_candidate
        return best_candidate
    
    def _sample_architecture_from_trial(
        self,
        trial,
        data_type: str = 'tabular',
        preprocessing_steps: Optional[List[str]] = None
    ) -> ArchitectureCandidate:
        """Sample architecture from Optuna trial."""
        # Sample number of layers
        n_layers = trial.suggest_int('n_layers', self.config.min_layers, self.config.max_layers)

        layers = []
        total_params = 0
        estimated_flops = 0

        allowed_layer_types = self.search_space._get_layer_types_for_data(data_type)

        for i in range(n_layers):
            layer_type = trial.suggest_categorical(f'layer_type_{i}', allowed_layer_types)

            if layer_type == 'dense':
                units = trial.suggest_int(f'units_{i}', self.config.min_units, self.config.max_units)
                activation = trial.suggest_categorical(f'activation_{i}', self.config.activation_functions)
                dropout = trial.suggest_categorical(f'dropout_{i}', self.config.dropout_rates)

                layer_config = {
                    'type': 'dense',
                    'units': units,
                    'activation': activation,
                    'dropout': dropout
                }

                prev_units = (
                    ESTIMATED_INPUT_FEATURES if not layers
                    else layers[-1].get('units', ESTIMATED_INPUT_FEATURES)
                )
                layer_params = prev_units * units

                total_params += layer_params
                estimated_flops += layer_params * 2

            elif layer_type in ['lstm', 'gru']:
                units = trial.suggest_int('rnn_units', self.config.min_units, self.config.max_units)
                return_sequences = trial.suggest_categorical('return_sequences', [True, False])
                dropout = trial.suggest_categorical('rnn_dropout', self.config.dropout_rates)
                
                layer_config = {
                    'type': layer_type,
                    'units': units,
                    'return_sequences': return_sequences,
                    'dropout': dropout
                }
                
                layer_params = 4 * units * units if layer_type == 'lstm' else 3 * units * units
                total_params += layer_params
                estimated_flops += layer_params * 4

            elif layer_type == 'conv1d':
                filters = trial.suggest_int(f'filters_{i}', self.config.min_units, self.config.max_units)
                kernel_size = trial.suggest_categorical(f'kernel_{i}', [1, 3, 5, 7])
                activation = trial.suggest_categorical(f'conv_activation_{i}', self.config.activation_functions)

                layer_config = {
                    'type': 'conv1d',
                    'filters': filters,
                    'kernel_size': kernel_size,
                    'activation': activation
                }

                layer_params = filters * kernel_size * ESTIMATED_INPUT_FEATURES
                total_params += layer_params
                estimated_flops += layer_params * 2

            elif layer_type == 'conv2d':
                filters = trial.suggest_int(f'conv2d_filters_{i}', self.config.min_units, self.config.max_units)
                kernel = trial.suggest_categorical(f'conv2d_kernel_{i}', [(3, 3), (5, 5)])
                layer_config = {
                    'type': 'conv2d',
                    'filters': filters,
                    'kernel_size': kernel,
                    'strides': (1, 1),
                    'activation': trial.suggest_categorical(f'conv2d_activation_{i}', self.config.activation_functions)
                }
                layer_params = filters * int(np.prod(kernel)) * ESTIMATED_INPUT_FEATURES
                total_params += layer_params
                estimated_flops += layer_params * 4

            elif layer_type == 'batchnorm':
                layer_config = {
                    'type': 'batchnorm',
                    'momentum': trial.suggest_float(f'batchnorm_momentum_{i}', 0.8, 0.99),
                    'epsilon': trial.suggest_float(f'batchnorm_epsilon_{i}', 1e-5, 1e-3)
                }

            elif layer_type == 'dropout':
                layer_config = {
                    'type': 'dropout',
                    'rate': trial.suggest_float(f'dropout_rate_{i}', 0.0, 0.8)
                }

            elif layer_type == 'self_attention':
                heads = trial.suggest_categorical(f'attention_heads_{i}', [2, 4, 8])
                key_dim = trial.suggest_int(
                    f'attention_keydim_{i}', self.config.min_units, self.config.max_units
                )
                dropout = trial.suggest_float(f'attention_dropout_{i}', 0.0, 0.5)
                layer_config = {
                    'type': 'self_attention',
                    'heads': heads,
                    'key_dim': key_dim,
                    'dropout': dropout
                }
                layer_params = heads * key_dim * ESTIMATED_INPUT_FEATURES
                total_params += layer_params
                estimated_flops += layer_params * 4

            layers.append(layer_config)

        if total_params > DATA_AWARE_PARAMETER_CAPACITY:
            self.logger.debug(
                "Trial-sampled architecture above data-aware parameter budget (%s > %s)",
                total_params,
                DATA_AWARE_PARAMETER_CAPACITY,
            )

        return ArchitectureCandidate(
            layers=layers,
            total_params=total_params,
            estimated_flops=estimated_flops,
            trial_number=trial.number,
            data_type=data_type,
            preprocessing_steps=preprocessing_steps
        )

    def _sample_architecture_from_params(
        self,
        params: Dict[str, Any],
        data_type: str,
        preprocessing_steps: Optional[List[str]]
    ) -> ArchitectureCandidate:
        """Reconstruct architecture using deterministic parameters."""
        seed_source = json.dumps(params, sort_keys=True)
        seed = int(hashlib.md5(seed_source.encode()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.default_rng(seed)

        n_layers = int(params.get('n_layers', self.config.min_layers))
        allowed_layer_types = self.search_space._get_layer_types_for_data(data_type)

        layers = []
        total_params = 0
        estimated_flops = 0

        for i in range(n_layers):
            layer_type = params.get(f'layer_type_{i}') or rng.choice(allowed_layer_types)

            if layer_type == 'dense':
                base_units = params.get('dense_units', self.config.max_units)
                noise = rng.integers(-32, 32)
                units = int(np.clip(base_units + noise, self.config.min_units, self.config.max_units))
                activation = params.get('activation', rng.choice(self.config.activation_functions))
                dropout = params.get('dropout_rate', rng.choice(self.config.dropout_rates))
                layer_config = {
                    'type': 'dense',
                    'units': units,
                    'activation': activation,
                    'dropout': dropout
                }
                prev_units = (
                    layers[-1].get('units', ESTIMATED_INPUT_FEATURES)
                    if layers else ESTIMATED_INPUT_FEATURES
                )
                layer_params = prev_units * units
                total_params += layer_params
                estimated_flops += layer_params * 2

            elif layer_type in {'lstm', 'gru'}:
                base_units = params.get('rnn_units', self.config.max_units)
                noise = rng.integers(-16, 16)
                units = int(np.clip(base_units + noise, self.config.min_units, self.config.max_units))
                layer_config = {
                    'type': layer_type,
                    'units': units,
                    'return_sequences': i < n_layers - 1,
                    'dropout': float(params.get('dropout_rate', rng.choice(self.config.dropout_rates)))
                }
                layer_params = (4 if layer_type == 'lstm' else 3) * units * units
                total_params += layer_params
                estimated_flops += layer_params * 4

            elif layer_type == 'conv1d':
                filters = int(
                    np.clip(
                        params.get('conv_filters', self.config.max_units) + rng.integers(-8, 8),
                        self.config.min_units,
                        self.config.max_units,
                    )
                )
                kernel = int(rng.choice([1, 3, 5, 7]))
                layer_config = {
                    'type': 'conv1d',
                    'filters': filters,
                    'kernel_size': kernel,
                    'activation': params.get('activation', rng.choice(self.config.activation_functions))
                }
                layer_params = filters * kernel * ESTIMATED_INPUT_FEATURES
                total_params += layer_params
                estimated_flops += layer_params * 2

            elif layer_type == 'conv2d':
                filters = int(
                    np.clip(
                        params.get('conv_filters', self.config.max_units // 2) + rng.integers(-8, 8),
                        self.config.min_units,
                        self.config.max_units,
                    )
                )
                kernel = rng.choice([(3, 3), (5, 5)])
                layer_config = {
                    'type': 'conv2d',
                    'filters': filters,
                    'kernel_size': kernel,
                    'strides': (1, 1),
                    'activation': params.get('activation', 'relu')
                }
                layer_params = filters * int(np.prod(kernel)) * ESTIMATED_INPUT_FEATURES
                total_params += layer_params
                estimated_flops += layer_params * 4

            elif layer_type == 'batchnorm':
                layer_config = {
                    'type': 'batchnorm',
                    'momentum': float(rng.uniform(0.8, 0.99)),
                    'epsilon': float(rng.uniform(1e-5, 1e-3))
                }

            elif layer_type == 'dropout':
                layer_config = {
                    'type': 'dropout',
                    'rate': float(np.clip(params.get('dropout_rate', rng.choice(self.config.dropout_rates)), 0.0, 0.9))
                }

            elif layer_type == 'self_attention':
                heads = int(rng.choice([2, 4, 8]))
                key_dim = int(
                    np.clip(
                        rng.integers(self.config.min_units, self.config.max_units + 1),
                        self.config.min_units,
                        self.config.max_units,
                    )
                )
                layer_config = {
                    'type': 'self_attention',
                    'heads': heads,
                    'key_dim': key_dim,
                    'dropout': float(np.clip(params.get('dropout_rate', rng.choice(self.config.dropout_rates)), 0.0, 0.5))
                }
                layer_params = heads * key_dim * ESTIMATED_INPUT_FEATURES
                total_params += layer_params
                estimated_flops += layer_params * 4

            else:
                layer_config = {'type': layer_type}

            self.search_space._validate_layer_config(layer_config)
            layers.append(layer_config)

        if total_params > DATA_AWARE_PARAMETER_CAPACITY:
            self.logger.debug(
                "Parameter budget exceeded when reconstructing architecture (%s > %s)",
                total_params,
                DATA_AWARE_PARAMETER_CAPACITY,
            )

        return ArchitectureCandidate(
            layers=layers,
            total_params=total_params,
            estimated_flops=estimated_flops,
            data_type=data_type,
            preprocessing_steps=preprocessing_steps
        )
    
    def _train_and_evaluate_architecture(self,
                                       candidate: ArchitectureCandidate,
                                       X_train: np.ndarray,
                                       y_train: np.ndarray,
                                       X_val: np.ndarray,
                                       y_val: np.ndarray,
                                       regime_labels: Optional[np.ndarray] = None,
                                       deadline: Optional[float] = None,
                                       trial_index: Optional[int] = None) -> Dict[str, float]:
        """Train and evaluate an architecture candidate."""
        try:
            if deadline and time.time() >= deadline:
                raise TimeoutError("Search deadline reached before training")

            training_budget = self._get_training_budget(trial_index or 0)

            # Create model
            if self.framework == 'pytorch':
                input_shape = X_train.shape[1:] if X_train.ndim > 1 else (1,)
                model = self._create_pytorch_model(
                    candidate,
                    input_shape,
                    y_train.shape[1] if len(y_train.shape) > 1 else 1
                )
                performance = self._train_pytorch_model(
                    model,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    training_budget
                )
            else:
                input_shape = X_train.shape[1:] if X_train.ndim > 1 else (1,)
                model = self._create_tensorflow_model(
                    candidate,
                    input_shape,
                    y_train.shape[1] if len(y_train.shape) > 1 else 1
                )
                performance = self._train_tensorflow_model(
                    model,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    training_budget
                )

            # Calculate multi-objective score
            overall_score = self._calculate_overall_score(performance, candidate)
            performance['overall_score'] = overall_score

            return performance
            
        except Exception as e:
            self.logger.warning(f"Architecture training failed: {e}")
            return {
                'accuracy': 0.0,
                'efficiency_score': 0.0,
                'robustness_score': 0.0,
                'overall_score': 0.0
            }

    def _get_training_budget(self, trial_index: int) -> Dict[str, int]:
        """Determine the training budget (epochs) using successive halving."""
        base_epochs = 50
        if not self.config.enable_successive_halving:
            return {'epochs': base_epochs}

        stage = max(0, trial_index // max(1, self.config.halving_factor))
        epochs = max(
            self.config.min_resource_epochs,
            int(base_epochs / (self.config.halving_factor ** stage))
        )
        return {'epochs': epochs}
    
    def _create_pytorch_model(
        self,
        candidate: ArchitectureCandidate,
        input_shape: Tuple[int, ...],
        output_size: int
    ) -> nn.Module:
        """Create PyTorch model from architecture candidate."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        class NASModel(nn.Module):
            def __init__(self, layers_config, input_shape, output_size):
                super().__init__()
                self.layers = nn.ModuleList()
                self.flatten_layer = None
                prev_size = input_shape[-1] if input_shape else 1

                for layer_config in layers_config:
                    layer_type = layer_config['type']

                    if layer_type == 'dense':
                        self.layers.append(nn.Linear(prev_size, layer_config['units']))
                        activation = layer_config.get('activation', 'relu')
                        if activation == 'relu':
                            self.layers.append(nn.ReLU())
                        elif activation == 'tanh':
                            self.layers.append(nn.Tanh())
                        elif activation == 'swish':
                            self.layers.append(nn.SiLU())
                        elif activation == 'gelu':
                            self.layers.append(nn.GELU())
                        elif activation == 'sigmoid':
                            self.layers.append(nn.Sigmoid())

                        dropout = layer_config.get('dropout', 0.0)
                        if dropout > 0:
                            self.layers.append(nn.Dropout(dropout))
                        prev_size = layer_config['units']

                    elif layer_type == 'lstm':
                        self.layers.append(nn.LSTM(prev_size, layer_config['units'],
                                                   batch_first=True, dropout=layer_config.get('dropout', 0.0)))
                        prev_size = layer_config['units']

                    elif layer_type == 'gru':
                        self.layers.append(nn.GRU(prev_size, layer_config['units'],
                                                  batch_first=True, dropout=layer_config.get('dropout', 0.0)))
                        prev_size = layer_config['units']

                    elif layer_type == 'conv1d':
                        channels = layer_config.get('filters', 32)
                        kernel_size = layer_config.get('kernel_size', 3)
                        stride = layer_config.get('stride', 1)
                        self.layers.append(nn.Conv1d(prev_size, channels, kernel_size, stride))
                        self.layers.append(nn.AdaptiveAvgPool1d(1))
                        self.layers.append(nn.ReLU())
                        prev_size = channels

                    elif layer_type == 'conv2d':
                        channels = layer_config.get('filters', 32)
                        kernel_size = layer_config.get('kernel_size', (3, 3))
                        strides = layer_config.get('strides', (1, 1))
                        self.layers.append(nn.Conv2d(prev_size, channels, kernel_size, strides))
                        self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                        self.layers.append(nn.ReLU())
                        prev_size = channels

                    elif layer_type == 'batchnorm':
                        self.layers.append(nn.BatchNorm1d(prev_size))

                    elif layer_type == 'dropout':
                        self.layers.append(nn.Dropout(layer_config.get('rate', 0.1)))

                    elif layer_type == 'self_attention':
                        attention = nn.MultiheadAttention(
                            embed_dim=prev_size,
                            num_heads=layer_config.get('heads', 2),
                            dropout=layer_config.get('dropout', 0.1)
                        )
                        self.layers.append(attention)

                self.flatten_layer = nn.Flatten()
                self.output_layer = nn.Linear(prev_size, output_size)

            def forward(self, x):
                if x.ndim == 2:
                    pass
                elif x.ndim == 3:
                    x = x.float()
                elif x.ndim == 4:
                    x = x.permute(0, 3, 1, 2)

                for layer in self.layers:
                    if isinstance(layer, (nn.LSTM, nn.GRU)):
                        x, _ = layer(x)
                    elif isinstance(layer, nn.MultiheadAttention):
                        x = x.permute(1, 0, 2)
                        x, _ = layer(x, x, x)
                        x = x.permute(1, 0, 2)
                        x = x.mean(dim=1, keepdim=True)
                    elif isinstance(layer, nn.Conv1d):
                        if x.ndim == 2:
                            x = x.unsqueeze(1)
                        if x.shape[1] != layer.in_channels:
                            x = x.permute(0, 2, 1)
                        x = layer(x)
                    elif isinstance(layer, nn.Conv2d):
                        if x.ndim == 3:
                            x = x.unsqueeze(1)
                        if x.shape[1] != layer.in_channels:
                            x = x.permute(0, 3, 1, 2)
                        x = layer(x)
                    else:
                        x = layer(x)

                if x.ndim > 2:
                    x = self.flatten_layer(x)
                x = self.output_layer(x)
                return x

        input_shape = input_shape if isinstance(input_shape, tuple) else (input_shape,)
        return NASModel(candidate.layers, input_shape, output_size)

    def _create_tensorflow_model(
        self,
        candidate: ArchitectureCandidate,
        input_shape: Tuple[int, ...],
        output_size: int
    ) -> keras.Model:
        """Create TensorFlow model from architecture candidate."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")

        inputs = keras.Input(shape=input_shape)
        x = inputs

        for layer_config in candidate.layers:
            layer_type = layer_config['type']
            if layer_type == 'dense':
                x = layers.Dense(layer_config['units'], activation=layer_config['activation'])(x)
                if layer_config.get('dropout', 0.0) > 0:
                    x = layers.Dropout(layer_config['dropout'])(x)

            elif layer_type == 'lstm':
                x = layers.LSTM(
                    layer_config['units'],
                    return_sequences=layer_config['return_sequences'],
                    dropout=layer_config.get('dropout', 0.0)
                )(x)

            elif layer_type == 'gru':
                x = layers.GRU(
                    layer_config['units'],
                    return_sequences=layer_config['return_sequences'],
                    dropout=layer_config.get('dropout', 0.0)
                )(x)

            elif layer_type == 'conv1d':
                x = layers.Conv1D(
                    filters=layer_config.get('filters', 32),
                    kernel_size=layer_config.get('kernel_size', 3),
                    activation=layer_config.get('activation', 'relu')
                )(x)

            elif layer_type == 'conv2d':
                x = layers.Conv2D(
                    filters=layer_config.get('filters', 32),
                    kernel_size=layer_config.get('kernel_size', (3, 3)),
                    strides=layer_config.get('strides', (1, 1)),
                    activation=layer_config.get('activation', 'relu')
                )(x)

            elif layer_type == 'batchnorm':
                x = layers.BatchNormalization()(x)

            elif layer_type == 'dropout':
                x = layers.Dropout(layer_config.get('rate', 0.1))(x)

            elif layer_type == 'self_attention':
                attention = layers.MultiHeadAttention(
                    num_heads=layer_config.get('heads', 2),
                    key_dim=layer_config.get('key_dim', 32),
                    dropout=layer_config.get('dropout', 0.1)
                )
                x = attention(x, x)

        if len(x.shape) > 2:
            x = layers.Flatten()(x)

        outputs = layers.Dense(output_size, activation='linear' if output_size > 1 else 'sigmoid')(x)
        model = keras.Model(inputs, outputs)
        return model

    def _train_pytorch_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           training_budget: Dict[str, int]) -> Dict[str, float]:
        """Train PyTorch model and return performance metrics."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        criterion = nn.MSELoss() if len(y_train.shape) == 1 else nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = training_budget.get('epochs', 50)
        model.train()
        for _ in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_tensor)
            val_pred = model(X_val_tensor)

            train_loss = criterion(train_pred, y_train_tensor).item()
            val_loss = criterion(val_pred, y_val_tensor).item()

            if len(y_train.shape) == 1:
                variance = torch.var(y_train_tensor).item() or 1.0
                train_accuracy = 1 - (train_loss / variance)
                val_accuracy = 1 - (val_loss / variance)
            else:
                train_accuracy = (torch.argmax(train_pred, dim=1) == torch.argmax(y_train_tensor, dim=1)).float().mean().item()
                val_accuracy = (torch.argmax(val_pred, dim=1) == torch.argmax(y_val_tensor, dim=1)).float().mean().item()

        total_params = sum(p.numel() for p in model.parameters())
        budget = max(DATA_AWARE_PARAMETER_CAPACITY, 1)
        efficiency_score = 1.0 / (1.0 + total_params / budget)
        robustness_score = 1.0 - abs(train_accuracy - val_accuracy)

        return {
            'accuracy': val_accuracy,
            'efficiency_score': efficiency_score,
            'robustness_score': robustness_score
        }

    def _update_candidate_records(self, candidate: ArchitectureCandidate, performance: Dict[str, float]) -> None:
        """Persist evaluation metrics and maintain Pareto front."""
        candidate.accuracy = performance['accuracy']
        candidate.efficiency_score = performance['efficiency_score']
        candidate.robustness_score = performance['robustness_score']
        candidate.overall_score = performance['overall_score']
        self.candidates.append(candidate)
        self.pareto_front = self._compute_pareto_front(self.candidates)

    def _select_best_candidate(
        self,
        current_best: Optional[ArchitectureCandidate],
        candidate: ArchitectureCandidate,
        performance: Dict[str, float]
    ) -> Optional[ArchitectureCandidate]:
        """Choose the best candidate using configured multi-objective strategy."""
        if not current_best:
            return candidate

        if self.config.multi_objective_strategy == 'pareto':
            if candidate in self.pareto_front and current_best not in self.pareto_front:
                return candidate
            if candidate in self.pareto_front and current_best in self.pareto_front:
                cand_score = self._crowding_distance(candidate, self.pareto_front)
                best_score = self._crowding_distance(current_best, self.pareto_front)
                if cand_score > best_score:
                    return candidate
        else:
            if performance['overall_score'] > current_best.overall_score:
                return candidate
        return current_best

    def _compute_pareto_front(self, candidates: List[ArchitectureCandidate]) -> List[ArchitectureCandidate]:
        """Compute Pareto optimal candidates across objectives."""
        pareto = []
        for cand in candidates:
            dominated = False
            for other in candidates:
                if other is cand:
                    continue
                if self._dominates(other, cand):
                    dominated = True
                    break
            if not dominated:
                pareto.append(cand)
        return pareto

    def _dominates(self, a: ArchitectureCandidate, b: ArchitectureCandidate) -> bool:
        """Return True if candidate a dominates candidate b."""
        metrics = ['accuracy', 'efficiency_score', 'robustness_score']
        better_or_equal = all(getattr(a, m) >= getattr(b, m) for m in metrics)
        strictly_better = any(getattr(a, m) > getattr(b, m) for m in metrics)
        return better_or_equal and strictly_better

    def _crowding_distance(self, candidate: ArchitectureCandidate, pareto: List[ArchitectureCandidate]) -> float:
        """Approximate crowding distance for Pareto ranking."""
        if not pareto:
            return 0.0

        metrics = ['accuracy', 'efficiency_score', 'robustness_score']
        distances = []
        for metric in metrics:
            values = sorted(pareto, key=lambda c: getattr(c, metric))
            if values[0] is candidate or values[-1] is candidate:
                distances.append(float('inf'))
                continue
            idx = values.index(candidate)
            prev_val = getattr(values[idx - 1], metric)
            next_val = getattr(values[idx + 1], metric)
            range_val = getattr(values[-1], metric) - getattr(values[0], metric) or 1e-6
            distances.append((next_val - prev_val) / range_val)
        return float(np.sum(distances))

    def _train_tensorflow_model(self, model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              training_budget: Dict[str, int]) -> Dict[str, float]:
        """Train TensorFlow model and return performance metrics."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")

        model.compile(
            optimizer='adam',
            loss='mse' if len(y_train.shape) == 1 else 'categorical_crossentropy',
            metrics=['accuracy']
        )

        epochs = training_budget.get('epochs', 50)
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )

        val_accuracy = history.history.get('val_accuracy', [0.0])[-1]
        train_accuracy = history.history.get('accuracy', [0.0])[-1]

        total_params = model.count_params()
        budget = max(DATA_AWARE_PARAMETER_CAPACITY, 1)
        efficiency_score = 1.0 / (1.0 + total_params / budget)
        robustness_score = 1.0 - abs(train_accuracy - val_accuracy)

        return {
            'accuracy': val_accuracy,
            'efficiency_score': efficiency_score,
            'robustness_score': robustness_score
        }
    
    def _calculate_overall_score(self, performance: Dict[str, float], candidate: ArchitectureCandidate) -> float:
        """Calculate overall score from multiple objectives."""
        try:
            # Get objective weights
            weights = self.config.objective_weights
            
            # Calculate weighted score
            overall_score = (
                weights[0] * performance['accuracy'] +
                weights[1] * performance['efficiency_score'] +
                weights[2] * performance['robustness_score']
            )
            
            return float(overall_score)
            
        except Exception as e:
            self.logger.warning(f"Overall score calculation failed: {e}")
            return 0.0
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of architecture search results."""
        if not self.candidates:
            return {'message': 'No search results available'}
        
        try:
            # Calculate summary statistics
            accuracies = [c.accuracy for c in self.candidates]
            efficiency_scores = [c.efficiency_score for c in self.candidates]
            overall_scores = [c.overall_score for c in self.candidates]
            param_counts = [c.total_params for c in self.candidates]
            
            return {
                'total_candidates': len(self.candidates),
                'best_accuracy': float(np.max(accuracies)),
                'best_efficiency': float(np.max(efficiency_scores)),
                'best_overall_score': float(np.max(overall_scores)),
                'average_parameters': float(np.mean(param_counts)),
                'parameter_range': [int(np.min(param_counts)), int(np.max(param_counts))],
                'search_statistics': {
                    'accuracy_mean': float(np.mean(accuracies)),
                    'accuracy_std': float(np.std(accuracies)),
                    'efficiency_mean': float(np.mean(efficiency_scores)),
                    'efficiency_std': float(np.std(efficiency_scores)),
                    'overall_score_mean': float(np.mean(overall_scores)),
                    'overall_score_std': float(np.std(overall_scores))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Search summary generation failed: {e}")
            return {'error': str(e)}


# Convenience function
def search_neural_architecture(X_train: np.ndarray, 
                              y_train: np.ndarray,
                              X_val: Optional[np.ndarray] = None,
                              y_val: Optional[np.ndarray] = None,
                              config: Optional[ArchitectureConfig] = None,
                              regime_labels: Optional[np.ndarray] = None) -> ArchitectureCandidate:
    """
    Convenience function to perform neural architecture search.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        config: Architecture search configuration
        regime_labels: Regime labels for regime-aware search (optional)
        
    Returns:
        Best architecture candidate
    """
    if config is None:
        config = ArchitectureConfig()
    
    nas = NeuralArchitectureSearch(config)
    return nas.search(X_train, y_train, X_val, y_val, regime_labels)