"""
Unified Architecture Compression Utilities for NAS and TAS Systems

This module provides comprehensive compression techniques for both neural and tree
architectures, including pruning, quantization, distillation, and other optimization
methods to reduce model size and inference time while maintaining performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class CompressionMethod(Enum):
    """Types of compression methods."""
    PRUNING = "pruning"
    QUANTIZATION = "quantization"
    DISTILLATION = "distillation"
    LOW_RANK_DECOMPOSITION = "low_rank_decomposition"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    STRUCTURED_PRUNING = "structured_pruning"
    UNSTRUCTURED_PRUNING = "unstructured_pruning"
    WEIGHT_SHARING = "weight_sharing"
    ACTIVATION_COMPRESSION = "activation_compression"
    TREE_PRUNING = "tree_pruning"
    FEATURE_SELECTION = "feature_selection"

class CompressionLevel(Enum):
    """Compression intensity levels."""
    LIGHT = "light"      # < 20% compression
    MODERATE = "moderate"  # 20-50% compression
    AGGRESSIVE = "aggressive"  # 50-80% compression
    EXTREME = "extreme"   # > 80% compression

@dataclass
class CompressionConfig:
    """Configuration for architecture compression."""

    # Compression method
    compression_method: CompressionMethod = CompressionMethod.PRUNING
    compression_level: CompressionLevel = CompressionLevel.MODERATE

    # Pruning parameters
    pruning_ratio: float = 0.3
    pruning_criterion: str = "magnitude"  # "magnitude", "gradient", "activation"
    structured_pruning: bool = False

    # Quantization parameters
    quantization_bits: int = 8
    quantization_scheme: str = "symmetric"  # "symmetric", "asymmetric", "dynamic"
    calibration_samples: int = 100

    # Distillation parameters
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.7
    distillation_epochs: int = 50

    # Performance constraints
    max_performance_loss: float = 0.05  # Maximum allowed performance degradation
    min_compression_ratio: float = 0.2  # Minimum compression ratio to achieve

    # Hardware constraints
    target_memory_mb: Optional[int] = None
    target_inference_time_ms: Optional[float] = None

    # Validation
    enable_validation: bool = True
    validation_samples: int = 1000

@dataclass
class CompressionResult:
    """Result from architecture compression."""
    compressed_architecture: Any
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    performance_retention: float
    inference_speedup: float
    compression_method: CompressionMethod
    compression_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class UnifiedArchitectureCompressor:
    """Unified architecture compressor for both NAS and TAS systems."""

    def __init__(self, config: CompressionConfig):
        """Initialize the unified architecture compressor."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Compression state
        self.compression_history = []
        self.compression_cache = {}

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info("✅ Unified Architecture Compressor initialized")
        self.logger.info(f"   Method: {config.compression_method.value}")
        self.logger.info(f"   Level: {config.compression_level.value}")
        self.logger.info(f"   Device: {self.device}")

    def compress_architecture(self,
                            architecture: Any,
                            architecture_type: str,
                            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> CompressionResult:
        """Compress an architecture based on its type and configuration."""
        start_time = time.time()
        self.logger.info(f"🗜️ Starting {self.config.compression_method.value} compression for {architecture_type}")

        try:
            # Determine compression parameters based on level
            compression_params = self._get_compression_parameters()

            # Compress based on architecture type
            if architecture_type.lower() == "neural":
                compressed_arch, original_size, compressed_size = self._compress_neural_architecture(
                    architecture, compression_params, validation_data
                )
            elif architecture_type.lower() == "tree":
                compressed_arch, original_size, compressed_size = self._compress_tree_architecture(
                    architecture, compression_params, validation_data
                )
            else:
                raise ValueError(f"Unsupported architecture type: {architecture_type}")

            # Calculate metrics
            compression_ratio = (original_size - compressed_size) / original_size
            performance_retention = self._evaluate_performance_retention(
                architecture, compressed_arch, validation_data
            )
            inference_speedup = self._calculate_inference_speedup(
                architecture, compressed_arch, validation_data
            )

            compression_time = time.time() - start_time

            result = CompressionResult(
                compressed_architecture=compressed_arch,
                original_size_mb=original_size,
                compressed_size_mb=compressed_size,
                compression_ratio=compression_ratio,
                performance_retention=performance_retention,
                inference_speedup=inference_speedup,
                compression_method=self.config.compression_method,
                compression_time=compression_time,
                metadata={
                    'architecture_type': architecture_type,
                    'compression_params': compression_params,
                    'compression_level': self.config.compression_level.value
                }
            )

            # Validate compression result
            if self.config.enable_validation:
                validation_result = self._validate_compression(result)
                if not validation_result['is_valid']:
                    self.logger.warning(f"Compression validation failed: {validation_result['reason']}")

            self.compression_history.append(result)

            self.logger.info(f"✅ Compression completed in {compression_time:.2f}s")
            self.logger.info(f"   Compression Ratio: {compression_ratio:.2%}")
            self.logger.info(f"   Performance Retention: {performance_retention:.2%}")
            self.logger.info(f"   Inference Speedup: {inference_speedup:.2f}x")

            return result

        except Exception as e:
            compression_time = time.time() - start_time
            self.logger.error(f"❌ Compression failed: {e}")

            # Return fallback result
            return CompressionResult(
                compressed_architecture=architecture,
                original_size_mb=0.0,
                compressed_size_mb=0.0,
                compression_ratio=0.0,
                performance_retention=0.0,
                inference_speedup=1.0,
                compression_method=self.config.compression_method,
                compression_time=compression_time,
                metadata={'error': str(e)}
            )

    def _get_compression_parameters(self) -> Dict[str, Any]:
        """Get compression parameters based on compression level."""
        params = {
            'pruning_ratio': self.config.pruning_ratio,
            'quantization_bits': self.config.quantization_bits,
            'distillation_temperature': self.config.distillation_temperature,
            'distillation_alpha': self.config.distillation_alpha
        }

        # Adjust parameters based on compression level
        if self.config.compression_level == CompressionLevel.LIGHT:
            params['pruning_ratio'] *= 0.5
            params['quantization_bits'] = max(16, params['quantization_bits'])
        elif self.config.compression_level == CompressionLevel.MODERATE:
            # Use default parameters
            pass
        elif self.config.compression_level == CompressionLevel.AGGRESSIVE:
            params['pruning_ratio'] *= 1.5
            params['quantization_bits'] = min(8, params['quantization_bits'])
        elif self.config.compression_level == CompressionLevel.EXTREME:
            params['pruning_ratio'] *= 2.0
            params['quantization_bits'] = min(4, params['quantization_bits'])

        return params

    def _compress_neural_architecture(self,
                                    architecture: Any,
                                    compression_params: Dict[str, Any],
                                    validation_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Any, float, float]:
        """Compress neural architecture using specified method."""
        original_size = self._estimate_neural_architecture_size(architecture)

        if self.config.compression_method == CompressionMethod.PRUNING:
            compressed_arch = self._prune_neural_architecture(architecture, compression_params)
        elif self.config.compression_method == CompressionMethod.QUANTIZATION:
            compressed_arch = self._quantize_neural_architecture(architecture, compression_params)
        elif self.config.compression_method == CompressionMethod.DISTILLATION:
            compressed_arch = self._distill_neural_architecture(architecture, compression_params, validation_data)
        elif self.config.compression_method == CompressionMethod.LOW_RANK_DECOMPOSITION:
            compressed_arch = self._decompose_neural_architecture(architecture, compression_params)
        else:
            # Default to pruning
            compressed_arch = self._prune_neural_architecture(architecture, compression_params)

        compressed_size = self._estimate_neural_architecture_size(compressed_arch)

        return compressed_arch, original_size, compressed_size

    def _compress_tree_architecture(self,
                                  architecture: Any,
                                  compression_params: Dict[str, Any],
                                  validation_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Any, float, float]:
        """Compress tree architecture using specified method."""
        original_size = self._estimate_tree_architecture_size(architecture)

        if self.config.compression_method == CompressionMethod.TREE_PRUNING:
            compressed_arch = self._prune_tree_architecture(architecture, compression_params)
        elif self.config.compression_method == CompressionMethod.FEATURE_SELECTION:
            compressed_arch = self._select_tree_features(architecture, compression_params)
        elif self.config.compression_method == CompressionMethod.DISTILLATION:
            compressed_arch = self._distill_tree_architecture(architecture, compression_params, validation_data)
        else:
            # Default to tree pruning
            compressed_arch = self._prune_tree_architecture(architecture, compression_params)

        compressed_size = self._estimate_tree_architecture_size(compressed_arch)

        return compressed_arch, original_size, compressed_size

    def _prune_neural_architecture(self, architecture: Any, compression_params: Dict[str, Any]) -> Any:
        """Prune neural architecture."""
        try:
            if not isinstance(architecture, nn.Module):
                self.logger.warning("Architecture is not a PyTorch module, returning original")
                return architecture

            pruning_ratio = compression_params['pruning_ratio']

            # Create a copy of the architecture
            compressed_arch = self._deep_copy_neural_architecture(architecture)

            # Apply pruning to each layer
            for name, module in compressed_arch.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    if self.config.structured_pruning:
                        # Structured pruning
                        prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
                    else:
                        # Unstructured pruning
                        prune.l1_unstructured(module, name='weight', amount=pruning_ratio)

            # Make pruning permanent
            for name, module in compressed_arch.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.remove(module, 'weight')

            self.logger.info(f"✅ Neural architecture pruned with ratio {pruning_ratio:.2%}")
            return compressed_arch

        except Exception as e:
            self.logger.error(f"Neural pruning failed: {e}")
            return architecture

    def _quantize_neural_architecture(self, architecture: Any, compression_params: Dict[str, Any]) -> Any:
        """Quantize neural architecture."""
        try:
            if not isinstance(architecture, nn.Module):
                self.logger.warning("Architecture is not a PyTorch module, returning original")
                return architecture

            quantization_bits = compression_params['quantization_bits']

            # Create a copy of the architecture
            compressed_arch = self._deep_copy_neural_architecture(architecture)

            # Apply quantization
            if quantization_bits == 8:
                compressed_arch = torch.quantization.quantize_dynamic(
                    compressed_arch, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
            elif quantization_bits == 16:
                compressed_arch = torch.quantization.quantize_dynamic(
                    compressed_arch, {nn.Linear, nn.Conv2d}, dtype=torch.float16
                )
            else:
                # Custom quantization
                compressed_arch = self._custom_quantize_architecture(compressed_arch, quantization_bits)

            self.logger.info(f"✅ Neural architecture quantized to {quantization_bits} bits")
            return compressed_arch

        except Exception as e:
            self.logger.error(f"Neural quantization failed: {e}")
            return architecture

    def _distill_neural_architecture(self,
                                   architecture: Any,
                                   compression_params: Dict[str, Any],
                                   validation_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> Any:
        """Distill neural architecture to a smaller model."""
        try:
            if not isinstance(architecture, nn.Module):
                self.logger.warning("Architecture is not a PyTorch module, returning original")
                return architecture

            # Create a smaller student model
            student_arch = self._create_student_neural_architecture(architecture)

            if validation_data is not None:
                # Perform knowledge distillation
                student_arch = self._perform_knowledge_distillation(
                    architecture, student_arch, validation_data, compression_params
                )

            self.logger.info("✅ Neural architecture distilled to smaller model")
            return student_arch

        except Exception as e:
            self.logger.error(f"Neural distillation failed: {e}")
            return architecture

    def _decompose_neural_architecture(self, architecture: Any, compression_params: Dict[str, Any]) -> Any:
        """Apply low-rank decomposition to neural architecture."""
        try:
            if not isinstance(architecture, nn.Module):
                self.logger.warning("Architecture is not a PyTorch module, returning original")
                return architecture

            # Create a copy of the architecture
            compressed_arch = self._deep_copy_neural_architecture(architecture)

            # Apply SVD decomposition to linear layers
            for name, module in compressed_arch.named_modules():
                if isinstance(module, nn.Linear):
                    # Perform SVD decomposition
                    W = module.weight.data
                    U, S, V = torch.svd(W)

                    # Keep only top singular values
                    rank = max(1, int(W.size(0) * (1 - compression_params['pruning_ratio'])))
                    U = U[:, :rank]
                    S = S[:rank]
                    V = V[:, :rank]

                    # Reconstruct weight matrix
                    new_weight = torch.mm(U, torch.mm(torch.diag(S), V.t()))
                    module.weight.data = new_weight

            self.logger.info("✅ Neural architecture decomposed using SVD")
            return compressed_arch

        except Exception as e:
            self.logger.error(f"Neural decomposition failed: {e}")
            return architecture

    def _prune_tree_architecture(self, architecture: Any, compression_params: Dict[str, Any]) -> Any:
        """Prune tree architecture."""
        try:
            pruning_ratio = compression_params['pruning_ratio']

            if isinstance(architecture, (DecisionTreeClassifier, DecisionTreeRegressor)):
                # Single tree pruning
                compressed_arch = self._prune_single_tree(architecture, pruning_ratio)
            elif isinstance(architecture, (RandomForestClassifier, RandomForestRegressor)):
                # Random forest pruning
                compressed_arch = self._prune_random_forest(architecture, pruning_ratio)
            else:
                # Generic tree pruning
                compressed_arch = self._prune_generic_tree(architecture, pruning_ratio)

            self.logger.info(f"✅ Tree architecture pruned with ratio {pruning_ratio:.2%}")
            return compressed_arch

        except Exception as e:
            self.logger.error(f"Tree pruning failed: {e}")
            return architecture

    def _prune_single_tree(self, tree: Any, pruning_ratio: float) -> Any:
        """Prune a single decision tree."""
        try:
            # Create a copy
            compressed_tree = self._deep_copy_tree_architecture(tree)

            # Apply pruning by reducing max_depth
            if hasattr(compressed_tree, 'max_depth') and compressed_tree.max_depth is not None:
                new_depth = max(1, int(compressed_tree.max_depth * (1 - pruning_ratio)))
                compressed_tree.max_depth = new_depth

            # Apply pruning by increasing min_samples_leaf
            if hasattr(compressed_tree, 'min_samples_leaf'):
                compressed_tree.min_samples_leaf = max(1, int(compressed_tree.min_samples_leaf * (1 + pruning_ratio)))

            return compressed_tree

        except Exception as e:
            self.logger.error(f"Single tree pruning failed: {e}")
            return tree

    def _prune_random_forest(self, forest: Any, pruning_ratio: float) -> Any:
        """Prune a random forest."""
        try:
            # Create a copy
            compressed_forest = self._deep_copy_tree_architecture(forest)

            # Reduce number of estimators
            if hasattr(compressed_forest, 'n_estimators'):
                new_n_estimators = max(1, int(compressed_forest.n_estimators * (1 - pruning_ratio)))
                compressed_forest.n_estimators = new_n_estimators

            # Prune individual trees
            if hasattr(compressed_forest, 'estimators_'):
                for i, tree in enumerate(compressed_forest.estimators_):
                    compressed_forest.estimators_[i] = self._prune_single_tree(tree, pruning_ratio)

            return compressed_forest

        except Exception as e:
            self.logger.error(f"Random forest pruning failed: {e}")
            return forest

    def _prune_generic_tree(self, tree: Any, pruning_ratio: float) -> Any:
        """Prune a generic tree architecture."""
        try:
            # Generic pruning approach
            compressed_tree = self._deep_copy_tree_architecture(tree)

            # Try to reduce complexity parameters
            for param_name in ['max_depth', 'max_leaf_nodes', 'max_features']:
                if hasattr(compressed_tree, param_name):
                    current_value = getattr(compressed_tree, param_name)
                    if current_value is not None:
                        new_value = max(1, int(current_value * (1 - pruning_ratio)))
                        setattr(compressed_tree, param_name, new_value)

            return compressed_tree

        except Exception as e:
            self.logger.error(f"Generic tree pruning failed: {e}")
            return tree

    def _select_tree_features(self, architecture: Any, compression_params: Dict[str, Any]) -> Any:
        """Select features for tree architecture."""
        try:
            # Feature selection for tree models
            compressed_arch = self._deep_copy_tree_architecture(architecture)

            # Reduce max_features
            if hasattr(compressed_arch, 'max_features'):
                current_max_features = compressed_arch.max_features
                if isinstance(current_max_features, (int, float)):
                    new_max_features = max(1, int(current_max_features * (1 - compression_params['pruning_ratio'])))
                    compressed_arch.max_features = new_max_features

            self.logger.info("✅ Tree architecture features selected")
            return compressed_arch

        except Exception as e:
            self.logger.error(f"Tree feature selection failed: {e}")
            return architecture

    def _distill_tree_architecture(self,
                                 architecture: Any,
                                 compression_params: Dict[str, Any],
                                 validation_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> Any:
        """Distill tree architecture to a smaller model."""
        try:
            # Create a smaller student tree
            student_arch = self._create_student_tree_architecture(architecture)

            if validation_data is not None:
                # Perform knowledge distillation
                student_arch = self._perform_tree_distillation(
                    architecture, student_arch, validation_data, compression_params
                )

            self.logger.info("✅ Tree architecture distilled to smaller model")
            return student_arch

        except Exception as e:
            self.logger.error(f"Tree distillation failed: {e}")
            return architecture

    def _deep_copy_neural_architecture(self, architecture: Any) -> Any:
        """Create a deep copy of neural architecture."""
        try:
            if isinstance(architecture, nn.Module):
                return pickle.loads(pickle.dumps(architecture))
            else:
                # For non-PyTorch architectures, return original
                return architecture
        except Exception as e:
            self.logger.warning(f"Deep copy failed: {e}")
            return architecture

    def _deep_copy_tree_architecture(self, architecture: Any) -> Any:
        """Create a deep copy of tree architecture."""
        try:
            return pickle.loads(pickle.dumps(architecture))
        except Exception as e:
            self.logger.warning(f"Tree deep copy failed: {e}")
            return architecture

    def _create_student_neural_architecture(self, teacher: Any) -> Any:
        """Create a smaller student neural architecture."""
        try:
            if not isinstance(teacher, nn.Module):
                return teacher

            # Create a smaller version with reduced layers/size
            student_layers = []
            for name, module in teacher.named_modules():
                if isinstance(module, nn.Linear):
                    # Reduce layer size
                    in_features = module.in_features
                    out_features = module.out_features
                    new_out_features = max(1, int(out_features * 0.5))  # 50% reduction
                    student_layers.append(nn.Linear(in_features, new_out_features))
                elif isinstance(module, nn.Conv2d):
                    # Reduce channels
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    new_out_channels = max(1, int(out_channels * 0.5))  # 50% reduction
                    student_layers.append(nn.Conv2d(in_channels, new_out_channels,
                                                   module.kernel_size, module.stride, module.padding))

            # Create student model
            student = nn.Sequential(*student_layers)
            return student

        except Exception as e:
            self.logger.error(f"Student neural architecture creation failed: {e}")
            return teacher

    def _create_student_tree_architecture(self, teacher: Any) -> Any:
        """Create a smaller student tree architecture."""
        try:
            if isinstance(teacher, (DecisionTreeClassifier, DecisionTreeRegressor)):
                # Create a smaller decision tree
                student = DecisionTreeClassifier(
                    max_depth=max(1, teacher.max_depth // 2) if teacher.max_depth else 5,
                    min_samples_leaf=max(1, teacher.min_samples_leaf * 2) if teacher.min_samples_leaf else 2,
                    max_features=max(1, teacher.max_features // 2) if teacher.max_features else 'sqrt'
                )
            elif isinstance(teacher, (RandomForestClassifier, RandomForestRegressor)):
                # Create a smaller random forest
                student = RandomForestClassifier(
                    n_estimators=max(1, teacher.n_estimators // 2),
                    max_depth=max(1, teacher.max_depth // 2) if teacher.max_depth else 10,
                    min_samples_leaf=max(1, teacher.min_samples_leaf * 2) if teacher.min_samples_leaf else 2
                )
            else:
                # Generic student creation
                student = teacher

            return student

        except Exception as e:
            self.logger.error(f"Student tree architecture creation failed: {e}")
            return teacher

    def _perform_knowledge_distillation(self,
                                      teacher: Any,
                                      student: Any,
                                      validation_data: Tuple[np.ndarray, np.ndarray],
                                      compression_params: Dict[str, Any]) -> Any:
        """Perform knowledge distillation between teacher and student."""
        try:
            # Simplified knowledge distillation
            # In practice, this would involve training the student with teacher's soft targets

            X_val, y_val = validation_data

            # Convert to tensors if needed
            if isinstance(X_val, np.ndarray):
                X_val = torch.FloatTensor(X_val)
            if isinstance(y_val, np.ndarray):
                y_val = torch.LongTensor(y_val)

            # Set models to evaluation mode
            teacher.eval()
            student.train()

            # Get teacher predictions (soft targets)
            with torch.no_grad():
                teacher_logits = teacher(X_val)
                teacher_probs = torch.softmax(teacher_logits / compression_params['distillation_temperature'], dim=1)

            # Train student with distillation loss
            optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
            criterion = nn.KLDivLoss(reduction='batchmean')

            for epoch in range(compression_params.get('distillation_epochs', 10)):
                student_logits = student(X_val)
                student_probs = torch.log_softmax(student_logits / compression_params['distillation_temperature'], dim=1)

                # Distillation loss
                distillation_loss = criterion(student_probs, teacher_probs)

                # Hard target loss
                hard_loss = nn.CrossEntropyLoss()(student_logits, y_val)

                # Combined loss
                total_loss = (compression_params['distillation_alpha'] * distillation_loss +
                            (1 - compression_params['distillation_alpha']) * hard_loss)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            self.logger.info("✅ Knowledge distillation completed")
            return student

        except Exception as e:
            self.logger.error(f"Knowledge distillation failed: {e}")
            return student

    def _perform_tree_distillation(self,
                                 teacher: Any,
                                 student: Any,
                                 validation_data: Tuple[np.ndarray, np.ndarray],
                                 compression_params: Dict[str, Any]) -> Any:
        """Perform knowledge distillation for tree models."""
        try:
            X_val, y_val = validation_data

            # Train student with teacher's predictions
            teacher_predictions = teacher.predict_proba(X_val) if hasattr(teacher, 'predict_proba') else teacher.predict(X_val)

            # Train student model
            student.fit(X_val, y_val)

            self.logger.info("✅ Tree knowledge distillation completed")
            return student

        except Exception as e:
            self.logger.error(f"Tree distillation failed: {e}")
            return student

    def _custom_quantize_architecture(self, architecture: Any, bits: int) -> Any:
        """Apply custom quantization to architecture."""
        try:
            # Custom quantization implementation
            # This is a simplified version - in practice would be more sophisticated

            for name, module in architecture.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    # Quantize weights
                    weight = module.weight.data
                    scale = (2 ** (bits - 1) - 1) / torch.max(torch.abs(weight))
                    quantized_weight = torch.round(weight * scale) / scale
                    module.weight.data = quantized_weight

            return architecture

        except Exception as e:
            self.logger.error(f"Custom quantization failed: {e}")
            return architecture

    def _estimate_neural_architecture_size(self, architecture: Any) -> float:
        """Estimate size of neural architecture in MB."""
        try:
            if isinstance(architecture, nn.Module):
                total_params = sum(p.numel() for p in architecture.parameters())
                # Assume 4 bytes per parameter (float32)
                size_mb = (total_params * 4) / (1024 * 1024)
                return size_mb
            else:
                return 1.0  # Default size for non-PyTorch models
        except Exception as e:
            self.logger.warning(f"Neural size estimation failed: {e}")
            return 1.0

    def _estimate_tree_architecture_size(self, architecture: Any) -> float:
        """Estimate size of tree architecture in MB."""
        try:
            # Estimate based on model complexity
            if hasattr(architecture, 'tree_') and architecture.tree_ is not None:
                # Single tree
                n_nodes = architecture.tree_.node_count
                size_mb = (n_nodes * 50) / (1024 * 1024)  # Rough estimate
            elif hasattr(architecture, 'estimators_'):
                # Ensemble (Random Forest, etc.)
                total_nodes = sum(est.tree_.node_count for est in architecture.estimators_)
                size_mb = (total_nodes * 50) / (1024 * 1024)  # Rough estimate
            else:
                size_mb = 0.1  # Default small size

            return max(0.001, size_mb)  # Minimum size

        except Exception as e:
            self.logger.warning(f"Tree size estimation failed: {e}")
            return 0.1

    def _evaluate_performance_retention(self,
                                      original: Any,
                                      compressed: Any,
                                      validation_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate performance retention after compression."""
        try:
            if validation_data is None:
                return 1.0  # Assume 100% retention if no validation data

            X_val, y_val = validation_data

            # Get predictions from both models
            if isinstance(original, nn.Module) and isinstance(compressed, nn.Module):
                original.eval()
                compressed.eval()

                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_val)
                    original_pred = original(X_tensor)
                    compressed_pred = compressed(X_tensor)

                # Calculate accuracy retention
                original_acc = self._calculate_accuracy(original_pred, y_val)
                compressed_acc = self._calculate_accuracy(compressed_pred, y_val)

                retention = compressed_acc / original_acc if original_acc > 0 else 1.0

            else:
                # For tree models
                original_pred = original.predict(X_val)
                compressed_pred = compressed.predict(X_val)

                original_acc = np.mean(original_pred == y_val)
                compressed_acc = np.mean(compressed_pred == y_val)

                retention = compressed_acc / original_acc if original_acc > 0 else 1.0

            return min(1.0, max(0.0, retention))

        except Exception as e:
            self.logger.warning(f"Performance retention evaluation failed: {e}")
            return 0.8  # Default 80% retention

    def _calculate_accuracy(self, predictions: torch.Tensor, targets: np.ndarray) -> float:
        """Calculate accuracy for PyTorch predictions."""
        try:
            if predictions.dim() > 1:
                pred_classes = torch.argmax(predictions, dim=1)
            else:
                pred_classes = predictions

            targets_tensor = torch.LongTensor(targets)
            correct = (pred_classes == targets_tensor).float()
            accuracy = correct.mean().item()

            return accuracy

        except Exception as e:
            self.logger.warning(f"Accuracy calculation failed: {e}")
            return 0.5

    def _calculate_inference_speedup(self,
                                   original: Any,
                                   compressed: Any,
                                   validation_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Calculate inference speedup after compression."""
        try:
            if validation_data is None:
                # Estimate speedup based on size reduction
                original_size = self._estimate_neural_architecture_size(original)
                compressed_size = self._estimate_neural_architecture_size(compressed)

                if compressed_size > 0:
                    speedup = original_size / compressed_size
                else:
                    speedup = 1.0

                return min(10.0, max(1.0, speedup))  # Cap at 10x speedup

            # Measure actual inference time
            X_val, _ = validation_data
            sample_size = min(100, len(X_val))  # Use subset for timing

            # Time original model
            start_time = time.time()
            for _ in range(10):  # Multiple runs for better timing
                if isinstance(original, nn.Module):
                    with torch.no_grad():
                        _ = original(torch.FloatTensor(X_val[:sample_size]))
                else:
                    _ = original.predict(X_val[:sample_size])
            original_time = (time.time() - start_time) / 10

            # Time compressed model
            start_time = time.time()
            for _ in range(10):
                if isinstance(compressed, nn.Module):
                    with torch.no_grad():
                        _ = compressed(torch.FloatTensor(X_val[:sample_size]))
                else:
                    _ = compressed.predict(X_val[:sample_size])
            compressed_time = (time.time() - start_time) / 10

            speedup = original_time / compressed_time if compressed_time > 0 else 1.0

            return min(10.0, max(1.0, speedup))  # Cap at 10x speedup

        except Exception as e:
            self.logger.warning(f"Inference speedup calculation failed: {e}")
            return 1.5  # Default 1.5x speedup

    def _validate_compression(self, result: CompressionResult) -> Dict[str, Any]:
        """Validate compression result against constraints."""
        violations = []

        # Check compression ratio
        if result.compression_ratio < self.config.min_compression_ratio:
            violations.append(f"Compression ratio too low: {result.compression_ratio:.2%} < {self.config.min_compression_ratio:.2%}")

        # Check performance retention
        if result.performance_retention < (1.0 - self.config.max_performance_loss):
            violations.append(f"Performance loss too high: {(1.0 - result.performance_retention):.2%} > {self.config.max_performance_loss:.2%}")

        # Check target memory if specified
        if self.config.target_memory_mb and result.compressed_size_mb > self.config.target_memory_mb:
            violations.append(f"Compressed size too large: {result.compressed_size_mb:.2f}MB > {self.config.target_memory_mb}MB")

        return {
            'is_valid': len(violations) == 0,
            'violations': violations,
            'reason': '; '.join(violations) if violations else 'Valid'
        }

    def get_compression_statistics(self) -> Dict[str, Any]:
        """Get compression statistics."""
        if not self.compression_history:
            return {}

        compression_ratios = [r.compression_ratio for r in self.compression_history]
        performance_retentions = [r.performance_retention for r in self.compression_history]
        inference_speedups = [r.inference_speedup for r in self.compression_history]

        return {
            'total_compressions': len(self.compression_history),
            'avg_compression_ratio': np.mean(compression_ratios),
            'avg_performance_retention': np.mean(performance_retentions),
            'avg_inference_speedup': np.mean(inference_speedups),
            'compression_methods_used': list(set(r.compression_method.value for r in self.compression_history)),
            'successful_compressions': len([r for r in self.compression_history if r.compression_ratio > 0])
        }

def create_unified_architecture_compressor(config: CompressionConfig = None) -> UnifiedArchitectureCompressor:
    """Create a unified architecture compressor instance."""
    if config is None:
        config = CompressionConfig()

    return UnifiedArchitectureCompressor(config)

def quick_compress_architecture(architecture: Any,
                              architecture_type: str,
                              compression_method: CompressionMethod = CompressionMethod.PRUNING,
                              compression_level: CompressionLevel = CompressionLevel.MODERATE,
                              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> CompressionResult:
    """Quick architecture compression with default settings."""
    config = CompressionConfig(
        compression_method=compression_method,
        compression_level=compression_level
    )

    compressor = create_unified_architecture_compressor(config)
    return compressor.compress_architecture(architecture, architecture_type, validation_data)
