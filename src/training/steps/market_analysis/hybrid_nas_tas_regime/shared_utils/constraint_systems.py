"""
Architecture Constraint Systems for NAS and TAS

This module provides comprehensive constraint validation systems to ensure that
neural and tree architectures are valid, practical, and meet computational requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
import psutil
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

class ConstraintSeverity(Enum):
    """Severity levels for constraint violations."""
    ERROR = "error"  # Invalid architecture
    WARNING = "warning"  # Suboptimal but valid
    INFO = "info"  # Informational constraint

class ConstraintType(Enum):
    """Types of architectural constraints."""
    LAYER_COUNT = "layer_count"
    PARAMETER_COUNT = "parameter_count"
    MEMORY_USAGE = "memory_usage"
    TRAINING_TIME = "training_time"
    CONNECTION_VALIDITY = "connection_validity"
    GRADIENT_FLOW = "gradient_flow"
    ARCHITECTURE_COMPLEXITY = "architecture_complexity"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    NUMERICAL_STABILITY = "numerical_stability"
    PRACTICALITY = "practicality"

@dataclass
class ConstraintViolation:
    """Information about a constraint violation."""
    constraint_type: ConstraintType
    severity: ConstraintSeverity
    message: str
    value: float
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConstraintValidationResult:
    """Result of constraint validation."""
    is_valid: bool
    violations: List[ConstraintViolation]
    warnings: List[ConstraintViolation]
    info: List[ConstraintViolation]
    validation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ArchitectureConstraints:
    """Comprehensive constraints for neural and tree architectures."""
    # Basic architectural constraints
    max_layers: int = 20
    min_layers: int = 2
    max_hidden_size: int = 2048
    min_hidden_size: int = 8
    max_parameters: int = 10000000  # 10M parameters
    min_parameters: int = 100

    # Connection constraints
    max_connections_per_layer: int = 5
    min_connections_per_layer: int = 1
    allow_residual_connections: bool = True
    allow_skip_connections: bool = True
    max_residual_depth: int = 5
    enforce_gradient_flow: bool = True

    # Resource constraints
    max_memory_usage_mb: int = 4096  # 4GB
    max_training_time_seconds: int = 3600  # 1 hour
    max_inference_time_ms: int = 100  # 100ms
    max_model_size_mb: int = 100  # 100MB

    # Numerical stability constraints
    max_dropout_rate: float = 0.8
    min_dropout_rate: float = 0.0
    max_gradient_norm: float = 10.0
    min_gradient_norm: float = 1e-8

    # Complexity constraints
    max_complexity_score: float = 5.0
    max_tree_depth: int = 30
    max_trees: int = 50
    max_feature_ratio: float = 0.8  # Max features as ratio of samples

    # Financial model specific constraints
    max_lookback_periods: int = 1000
    min_lookback_periods: int = 10
    max_prediction_horizon: int = 100
    min_prediction_horizon: int = 1
    required_features: List[str] = field(default_factory=list)

    # Hardware-specific constraints
    enable_gpu_constraints: bool = False
    max_gpu_memory_gb: int = 8
    enable_cpu_constraints: bool = True
    max_cpu_cores: int = 8

    # Advanced constraints
    allow_complex_activations: bool = True
    allow_attention_mechanisms: bool = True
    allow_recurrent_layers: bool = True
    allow_conv_layers: bool = True
    enforce_reproducibility: bool = True
    allow_stochastic_layers: bool = True

class BaseConstraintValidator:
    """Base class for constraint validators."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the constraint validator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_initialized = False
        self.system_info = self._get_system_info()

    def validate(self, architecture: Any) -> ConstraintValidationResult:
        """Validate an architecture against all constraints."""
        try:
            violations = []
            is_valid = True

            # Check all constraint types
            for constraint_type in ConstraintType:
                violation = self.check_single_constraint(architecture, constraint_type)
                if violation is not None:
                    violations.append(violation)
                    is_valid = False

            # Calculate overall score
            if len(violations) == 0:
                score = 1.0
            else:
                # Calculate score based on severity of violations
                total_severity = sum(v.severity for v in violations)
                max_severity = len(violations) * 1.0  # Maximum possible severity
                score = max(0.0, 1.0 - (total_severity / max_severity))

            return ConstraintValidationResult(
                is_valid=is_valid,
                score=score,
                violations=violations,
                total_constraints=len(ConstraintType),
                passed_constraints=len(ConstraintType) - len(violations)
            )

        except Exception as e:
            tprint_error(f"Error validating architecture: {e}")
            tprint_debug(f"Error details: {type(e).__name__}: {str(e)}")
            # Return a default validation result
            return ConstraintValidationResult(
                is_valid=False,
                score=0.0,
                violations=[],
                total_constraints=len(ConstraintType),
                passed_constraints=0
            )

    def check_single_constraint(self, architecture: Any, constraint_type: ConstraintType) -> Optional[ConstraintViolation]:
        """Check a single constraint type."""
        try:
            tprint_debug(f"Checking constraint: {constraint_type}")
            if constraint_type == ConstraintType.PARAMETER_COUNT:
                tprint_debug("Checking parameter count constraint")
                return self._check_parameter_count_constraint(architecture)
            elif constraint_type == ConstraintType.MEMORY_USAGE:
                tprint_debug("Checking memory usage constraint")
                return self._check_memory_usage_constraint(architecture)
            elif constraint_type == ConstraintType.TRAINING_TIME:
                tprint_debug("Checking training time constraint")
                return self._check_training_time_constraint(architecture)
            elif constraint_type == ConstraintType.INFERENCE_TIME:
                tprint_debug("Checking inference time constraint")
                return self._check_inference_time_constraint(architecture)
            elif constraint_type == ConstraintType.ARCHITECTURE_DEPTH:
                tprint_debug("Checking architecture depth constraint")
                return self._check_architecture_depth_constraint(architecture)
            elif constraint_type == ConstraintType.ARCHITECTURE_WIDTH:
                tprint_debug("Checking architecture width constraint")
                return self._check_architecture_width_constraint(architecture)
            elif constraint_type == ConstraintType.ACTIVATION_FUNCTION:
                tprint_debug("Checking activation function constraint")
                return self._check_activation_function_constraint(architecture)
            elif constraint_type == ConstraintType.OPTIMIZER:
                tprint_debug("Checking optimizer constraint")
                return self._check_optimizer_constraint(architecture)
            elif constraint_type == ConstraintType.LEARNING_RATE:
                tprint_debug("Checking learning rate constraint")
                return self._check_learning_rate_constraint(architecture)
            elif constraint_type == ConstraintType.BATCH_SIZE:
                tprint_debug("Checking batch size constraint")
                return self._check_batch_size_constraint(architecture)
            else:
                tprint_warning(f"Unknown constraint type: {constraint_type}")
                # Unknown constraint type
                return ConstraintViolation(
                    constraint_type=constraint_type,
                    message=f"Unknown constraint type: {constraint_type}",
                    severity=0.5,
                    suggested_fix="Check constraint type definition"
                )

        except Exception as e:
            tprint_error(f"Error checking constraint {constraint_type}: {e}")
            tprint_debug(f"Error details: {type(e).__name__}: {str(e)}")
            return ConstraintViolation(
                constraint_type=constraint_type,
                message=f"Error checking constraint: {e}",
                severity=0.8,
                suggested_fix="Fix constraint checking logic"
            )

    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get a summary of all constraints."""
        try:
            summary = {
                'total_constraints': len(ConstraintType),
                'constraint_types': [ct.value for ct in ConstraintType],
                'system_info': self.system_info,
                'constraint_config': {
                    'max_parameters': self.config.max_parameters,
                    'max_memory_gb': self.config.max_memory_gb,
                    'max_training_time_hours': self.config.max_training_time_hours,
                    'max_inference_time_ms': self.config.max_inference_time_ms,
                    'max_depth': self.config.max_depth,
                    'max_width': self.config.max_width,
                    'allowed_activations': self.config.allowed_activations,
                    'allowed_optimizers': self.config.allowed_optimizers,
                    'min_learning_rate': self.config.min_learning_rate,
                    'max_learning_rate': self.config.max_learning_rate,
                    'min_batch_size': self.config.min_batch_size,
                    'max_batch_size': self.config.max_batch_size
                },
                'is_initialized': self.is_initialized
            }

            return summary

        except Exception as e:
            tprint(f"⚠️ [CONSTRAINT] Error getting constraint summary: {e}", color="yellow")
            return {
                'error': str(e),
                'total_constraints': 0,
                'constraint_types': [],
                'system_info': {},
                'constraint_config': {},
                'is_initialized': False
            }

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for resource constraints."""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'gpu_available': len(psutil.gpu_count()) > 0 if hasattr(psutil, 'gpu_count') else False
            }
        except:
            return {
                'cpu_count': 4,  # Default assumptions
                'memory_total_gb': 8,
                'memory_available_gb': 6,
                'gpu_available': False
            }

class NeuralConstraintValidator(BaseConstraintValidator):
    """Constraint validator for neural architectures."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize neural constraint validator."""
        super().__init__(config)
        self.constraints = config.get('constraints', ArchitectureConstraints())
        self.logger.info("✅ Neural Constraint Validator initialized")

    def validate(self, architecture: Any) -> ConstraintValidationResult:
        """Validate a neural architecture against all constraints."""
        from ..search_spaces import NeuralArchitecture, LayerType, ConnectionType

        if not isinstance(architecture, NeuralArchitecture):
            raise ValueError("Architecture must be a NeuralArchitecture instance")

        start_time = time.time()
        violations = []
        warnings = []
        info = []

        # Basic layer count constraints
        layer_count_violation = self._check_layer_count(architecture)
        if layer_count_violation:
            if layer_count_violation.severity == ConstraintSeverity.ERROR:
                violations.append(layer_count_violation)
            else:
                warnings.append(layer_count_violation)

        # Parameter count constraints
        param_violation = self._check_parameter_count(architecture)
        if param_violation:
            if param_violation.severity == ConstraintSeverity.ERROR:
                violations.append(param_violation)
            else:
                warnings.append(param_violation)

        # Connection validity constraints
        conn_violation = self._check_connection_validity(architecture)
        if conn_violation:
            violations.append(conn_violation)

        # Gradient flow constraints
        if self.constraints.enforce_gradient_flow:
            flow_violation = self._check_gradient_flow(architecture)
            if flow_violation:
                violations.append(flow_violation)

        # Resource constraints
        memory_violation = self._check_memory_usage(architecture)
        if memory_violation:
            violations.append(memory_violation)

        time_violation = self._check_training_time(architecture)
        if time_violation:
            warnings.append(time_violation)

        # Numerical stability constraints
        stability_violation = self._check_numerical_stability(architecture)
        if stability_violation:
            warnings.append(stability_violation)

        # Practicality constraints
        practicality_violation = self._check_practicality(architecture)
        if practicality_violation:
            info.append(practicality_violation)

        validation_time = time.time() - start_time

        return ConstraintValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            info=info,
            validation_time=validation_time,
            metadata={
                'architecture_type': 'neural',
                'n_layers': len(architecture.layers),
                'n_connections': len(architecture.connections)
            }
        )

    def check_single_constraint(self, architecture: Any, constraint_type: ConstraintType) -> Optional[ConstraintViolation]:
        """Check a single constraint type."""
        if constraint_type == ConstraintType.LAYER_COUNT:
            return self._check_layer_count(architecture)
        elif constraint_type == ConstraintType.PARAMETER_COUNT:
            return self._check_parameter_count(architecture)
        elif constraint_type == ConstraintType.CONNECTION_VALIDITY:
            return self._check_connection_validity(architecture)
        elif constraint_type == ConstraintType.GRADIENT_FLOW:
            return self._check_gradient_flow(architecture)
        elif constraint_type == ConstraintType.MEMORY_USAGE:
            return self._check_memory_usage(architecture)
        elif constraint_type == ConstraintType.TRAINING_TIME:
            return self._check_training_time(architecture)
        elif constraint_type == ConstraintType.NUMERICAL_STABILITY:
            return self._check_numerical_stability(architecture)
        else:
            return None

    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get a summary of all constraints."""
        return {
            'max_layers': self.constraints.max_layers,
            'min_layers': self.constraints.min_layers,
            'max_parameters': self.constraints.max_parameters,
            'max_memory_mb': self.constraints.max_memory_usage_mb,
            'max_training_seconds': self.constraints.max_training_time_seconds,
            'enforce_gradient_flow': self.constraints.enforce_gradient_flow,
            'system_info': self.system_info
        }

    def _check_layer_count(self, architecture: Any) -> Optional[ConstraintViolation]:
        """Check layer count constraints."""
        n_layers = len(architecture.layers)

        if n_layers < self.constraints.min_layers:
            return ConstraintViolation(
                constraint_type=ConstraintType.LAYER_COUNT,
                severity=ConstraintSeverity.ERROR,
                message=f"Too few layers: {n_layers} < {self.constraints.min_layers}",
                value=n_layers,
                threshold=self.constraints.min_layers
            )

        if n_layers > self.constraints.max_layers:
            return ConstraintViolation(
                constraint_type=ConstraintType.LAYER_COUNT,
                severity=ConstraintSeverity.WARNING,
                message=f"Too many layers: {n_layers} > {self.constraints.max_layers}",
                value=n_layers,
                threshold=self.constraints.max_layers
            )

        return None

    def _check_parameter_count(self, architecture: Any) -> Optional[ConstraintViolation]:
        """Check parameter count constraints."""
        total_params = sum(
            layer.hidden_size * (layer.hidden_size if hasattr(layer, 'hidden_size') else 1)
            for layer in architecture.layers
        )

        if total_params < self.constraints.min_parameters:
            return ConstraintViolation(
                constraint_type=ConstraintType.PARAMETER_COUNT,
                severity=ConstraintSeverity.ERROR,
                message=f"Too few parameters: {total_params} < {self.constraints.min_parameters}",
                value=total_params,
                threshold=self.constraints.min_parameters
            )

        if total_params > self.constraints.max_parameters:
            return ConstraintViolation(
                constraint_type=ConstraintType.PARAMETER_COUNT,
                severity=ConstraintSeverity.WARNING,
                message=f"Too many parameters: {total_params} > {self.constraints.max_parameters}",
                value=total_params,
                threshold=self.constraints.max_parameters
            )

        return None

    def _check_connection_validity(self, architecture: Any) -> Optional[ConstraintViolation]:
        """Check connection validity constraints."""
        n_layers = len(architecture.layers)
        connections_per_layer = {}

        for i in range(n_layers):
            connections_per_layer[i] = 0

        for conn in architecture.connections:
            from_idx, to_idx, conn_type = conn
            if 0 <= from_idx < n_layers and 0 <= to_idx < n_layers:
                connections_per_layer[from_idx] += 1

        max_connections = max(connections_per_layer.values()) if connections_per_layer else 0
        min_connections = min(connections_per_layer.values()) if connections_per_layer else 0

        if max_connections > self.constraints.max_connections_per_layer:
            return ConstraintViolation(
                constraint_type=ConstraintType.CONNECTION_VALIDITY,
                severity=ConstraintSeverity.ERROR,
                message=f"Too many connections per layer: {max_connections} > {self.constraints.max_connections_per_layer}",
                value=max_connections,
                threshold=self.constraints.max_connections_per_layer
            )

        if min_connections < self.constraints.min_connections_per_layer:
            return ConstraintViolation(
                constraint_type=ConstraintType.CONNECTION_VALIDITY,
                severity=ConstraintSeverity.WARNING,
                message=f"Too few connections per layer: {min_connections} < {self.constraints.min_connections_per_layer}",
                value=min_connections,
                threshold=self.constraints.min_connections_per_layer
            )

        return None

    def _check_gradient_flow(self, architecture: Any) -> Optional[ConstraintViolation]:
        """Check gradient flow constraints."""
        # Check for potential gradient vanishing/exploding issues
        has_skip_connections = any(
            conn[2] in [ConnectionType.RESIDUAL, ConnectionType.SKIP]
            for conn in architecture.connections
        )

        if not has_skip_connections and len(architecture.layers) > 10:
            return ConstraintViolation(
                constraint_type=ConstraintType.GRADIENT_FLOW,
                severity=ConstraintSeverity.WARNING,
                message="Deep network without skip connections may have gradient flow issues",
                value=len(architecture.layers),
                threshold=10
            )

        return None

    def _check_memory_usage(self, architecture: Any) -> Optional[ConstraintViolation]:
        """Check memory usage constraints."""
        estimated_memory = architecture.estimated_memory_usage

        if estimated_memory > self.constraints.max_memory_usage_mb:
            return ConstraintViolation(
                constraint_type=ConstraintType.MEMORY_USAGE,
                severity=ConstraintSeverity.ERROR,
                message=f"Estimated memory usage too high: {estimated_memory:.1f}MB > {self.constraints.max_memory_usage_mb}MB",
                value=estimated_memory,
                threshold=self.constraints.max_memory_usage_mb
            )

        return None

    def _check_training_time(self, architecture: Any) -> Optional[ConstraintViolation]:
        """Check training time constraints."""
        estimated_time = architecture.estimated_training_time

        if estimated_time > self.constraints.max_training_time_seconds:
            return ConstraintViolation(
                constraint_type=ConstraintType.TRAINING_TIME,
                severity=ConstraintSeverity.WARNING,
                message=f"Estimated training time too long: {estimated_time:.1f}s > {self.constraints.max_training_time_seconds}s",
                value=estimated_time,
                threshold=self.constraints.max_training_time_seconds
            )

        return None

    def _check_numerical_stability(self, architecture: Any) -> Optional[ConstraintViolation]:
        """Check numerical stability constraints."""
        max_dropout = max((layer.dropout_rate for layer in architecture.layers), default=0.0)

        if max_dropout > self.constraints.max_dropout_rate:
            return ConstraintViolation(
                constraint_type=ConstraintType.NUMERICAL_STABILITY,
                severity=ConstraintSeverity.WARNING,
                message=f"Dropout rate too high: {max_dropout:.2f} > {self.constraints.max_dropout_rate}",
                value=max_dropout,
                threshold=self.constraints.max_dropout_rate
            )

        return None

    def _check_practicality(self, architecture: Any) -> Optional[ConstraintViolation]:
        """Check practicality constraints."""
        complexity = architecture.estimated_complexity

        if complexity > self.constraints.max_complexity_score:
            return ConstraintViolation(
                constraint_type=ConstraintType.PRACTICALITY,
                severity=ConstraintSeverity.INFO,
                message=f"Architecture complexity high: {complexity:.2f} > {self.constraints.max_complexity_score}",
                value=complexity,
                threshold=self.constraints.max_complexity_score
            )

        return None

class TreeConstraintValidator(BaseConstraintValidator):
    """Constraint validator for tree architectures."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize tree constraint validator."""
        super().__init__(config)
        self.constraints = config.get('constraints', ArchitectureConstraints())
        self.logger.info("✅ Tree Constraint Validator initialized")

    def validate(self, architecture: Any) -> ConstraintValidationResult:
        """Validate a tree architecture against all constraints."""

        if not isinstance(architecture, TreeArchitecture):
            raise ValueError("Architecture must be a TreeArchitecture instance")

        start_time = time.time()
        violations = []
        warnings = []
        info = []

        # Basic tree count constraints
        tree_count_violation = self._check_tree_count(architecture)
        if tree_count_violation:
            if tree_count_violation.severity == ConstraintSeverity.ERROR:
                violations.append(tree_count_violation)
            else:
                warnings.append(tree_count_violation)

        # Tree depth constraints
        depth_violation = self._check_tree_depth(architecture)
        if depth_violation:
            violations.append(depth_violation)

        # Resource constraints
        memory_violation = self._check_memory_usage(architecture)
        if memory_violation:
            violations.append(memory_violation)

        time_violation = self._check_training_time(architecture)
        if time_violation:
            warnings.append(time_violation)

        # Complexity constraints
        complexity_violation = self._check_complexity(architecture)
        if complexity_violation:
            warnings.append(complexity_violation)

        # Ensemble method constraints
        ensemble_violation = self._check_ensemble_method(architecture)
        if ensemble_violation:
            warnings.append(ensemble_violation)

        validation_time = time.time() - start_time

        return ConstraintValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            info=info,
            validation_time=validation_time,
            metadata={
                'architecture_type': 'tree',
                'n_trees': len(architecture.trees),
                'ensemble_method': architecture.ensemble_method
            }
        )

    def check_single_constraint(self, architecture: Any, constraint_type: ConstraintType) -> Optional[ConstraintViolation]:
        """Check a single constraint type."""
        if constraint_type == ConstraintType.LAYER_COUNT:
            return self._check_tree_count(architecture)
        elif constraint_type == ConstraintType.PARAMETER_COUNT:
            return self._check_parameter_count(architecture)
        elif constraint_type == ConstraintType.MEMORY_USAGE:
            return self._check_memory_usage(architecture)
        elif constraint_type == ConstraintType.TRAINING_TIME:
            return self._check_training_time(architecture)
        elif constraint_type == ConstraintType.ARCHITECTURE_COMPLEXITY:
            return self._check_complexity(architecture)
        else:
            return None

    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get a summary of all constraints."""
        return {
            'max_trees': self.constraints.max_trees,
            'max_tree_depth': self.constraints.max_tree_depth,
            'max_memory_mb': self.constraints.max_memory_usage_mb,
            'max_training_seconds': self.constraints.max_training_time_seconds,
            'system_info': self.system_info
        }

    def _check_tree_count(self, architecture: Any) -> Optional[ConstraintViolation]:
        """Check tree count constraints."""
        n_trees = len(architecture.trees)

        if n_trees < 1:
            return ConstraintViolation(
                constraint_type=ConstraintType.LAYER_COUNT,
                severity=ConstraintSeverity.ERROR,
                message="At least one tree required",
                value=n_trees,
                threshold=1
            )

        if n_trees > self.constraints.max_trees:
            return ConstraintViolation(
                constraint_type=ConstraintType.LAYER_COUNT,
                severity=ConstraintSeverity.WARNING,
                message=f"Too many trees: {n_trees} > {self.constraints.max_trees}",
                value=n_trees,
                threshold=self.constraints.max_trees
            )

        return None

    def _check_tree_depth(self, architecture: Any) -> Optional[ConstraintViolation]:
        """Check tree depth constraints."""
        max_depth = max((tree.max_depth or 10 for tree in architecture.trees), default=10)

        if max_depth > self.constraints.max_tree_depth:
            return ConstraintViolation(
                constraint_type=ConstraintType.ARCHITECTURE_COMPLEXITY,
                severity=ConstraintSeverity.ERROR,
                message=f"Tree depth too high: {max_depth} > {self.constraints.max_tree_depth}",
                value=max_depth,
                threshold=self.constraints.max_tree_depth
            )

        return None

    def _check_memory_usage(self, architecture: Any) -> Optional[ConstraintViolation]:
        """Check memory usage constraints."""
        estimated_memory = architecture.estimated_memory_usage

        if estimated_memory > self.constraints.max_memory_usage_mb:
            return ConstraintViolation(
                constraint_type=ConstraintType.MEMORY_USAGE,
                severity=ConstraintSeverity.ERROR,
                message=f"Estimated memory usage too high: {estimated_memory:.1f}MB > {self.constraints.max_memory_usage_mb}MB",
                value=estimated_memory,
                threshold=self.constraints.max_memory_usage_mb
            )

        return None

    def _check_training_time(self, architecture: Any) -> Optional[ConstraintViolation]:
        """Check training time constraints."""
        estimated_time = architecture.estimated_training_time

        if estimated_time > self.constraints.max_training_time_seconds:
            return ConstraintViolation(
                constraint_type=ConstraintType.TRAINING_TIME,
                severity=ConstraintSeverity.WARNING,
                message=f"Estimated training time too long: {estimated_time:.1f}s > {self.constraints.max_training_time_seconds}s",
                value=estimated_time,
                threshold=self.constraints.max_training_time_seconds
            )

        return None

    def _check_complexity(self, architecture: Any) -> Optional[ConstraintViolation]:
        """Check complexity constraints."""
        complexity = architecture.estimated_complexity

        if complexity > self.constraints.max_complexity_score:
            return ConstraintViolation(
                constraint_type=ConstraintType.ARCHITECTURE_COMPLEXITY,
                severity=ConstraintSeverity.WARNING,
                message=f"Architecture complexity high: {complexity:.2f} > {self.constraints.max_complexity_score}",
                value=complexity,
                threshold=self.constraints.max_complexity_score
            )

        return None

    def _check_ensemble_method(self, architecture: Any) -> Optional[ConstraintViolation]:
        """Check ensemble method constraints."""
        ensemble_method = architecture.ensemble_method

        if ensemble_method == 'stacking':
            n_trees = len(architecture.trees)
            if n_trees < 5:
                return ConstraintViolation(
                    constraint_type=ConstraintType.PRACTICALITY,
                    severity=ConstraintSeverity.WARNING,
                    message="Stacking ensemble with few trees may overfit",
                    value=n_trees,
                    threshold=5
                )

        return None

class UnifiedConstraintValidator:
    """Unified constraint validator that handles both neural and tree architectures."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize unified constraint validator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize specialized validators
        self.neural_validator = NeuralConstraintValidator(config.get('neural_config', {}))
        self.tree_validator = TreeConstraintValidator(config.get('tree_config', {}))

        self.logger.info("✅ Unified Constraint Validator initialized")

    def validate(self, architecture: Any) -> ConstraintValidationResult:
        """Validate any architecture type."""

        if isinstance(architecture, NeuralArchitecture):
            return self.neural_validator.validate(architecture)
        elif isinstance(architecture, TreeArchitecture):
            return self.tree_validator.validate(architecture)
        else:
            raise ValueError(f"Unsupported architecture type: {type(architecture)}")

    def check_single_constraint(self, architecture: Any, constraint_type: ConstraintType) -> Optional[ConstraintViolation]:
        """Check a single constraint type for any architecture."""

        if isinstance(architecture, NeuralArchitecture):
            return self.neural_validator.check_single_constraint(architecture, constraint_type)
        elif isinstance(architecture, TreeArchitecture):
            return self.tree_validator.check_single_constraint(architecture, constraint_type)
        else:
            return None

    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get a summary of all constraints."""
        neural_summary = self.neural_validator.get_constraint_summary()
        tree_summary = self.tree_validator.get_constraint_summary()

        return {
            'neural': neural_summary,
            'tree': tree_summary,
            'unified': {
                'system_info': self.neural_validator.system_info
            }
        }

    def validate_multiple(self, architectures: List[Any]) -> List[ConstraintValidationResult]:
        """Validate multiple architectures."""
        results = []

        for architecture in architectures:
            try:
                result = self.validate(architecture)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to validate architecture: {e}")
                results.append(ConstraintValidationResult(
                    is_valid=False,
                    violations=[ConstraintViolation(
                        constraint_type=ConstraintType.PRACTICALITY,
                        severity=ConstraintSeverity.ERROR,
                        message=f"Validation failed: {e}",
                        value=0.0,
                        threshold=0.0
                    )],
                    warnings=[],
                    info=[],
                    validation_time=0.0
                ))

        return results

    def filter_valid_architectures(self, architectures: List[Any]) -> Tuple[List[Any], List[ConstraintValidationResult]]:
        """Filter architectures to keep only valid ones."""
        valid_architectures = []
        results = []

        for architecture in architectures:
            try:
                result = self.validate(architecture)
                results.append(result)

                if result.is_valid:
                    valid_architectures.append(architecture)
            except:
                results.append(ConstraintValidationResult(
                    is_valid=False,
                    violations=[],
                    warnings=[],
                    info=[],
                    validation_time=0.0
                ))

        self.logger.info(f"Filtered {len(architectures)} architectures to {len(valid_architectures)} valid ones")
        return valid_architectures, results

def create_neural_constraint_validator(config: Dict[str, Any]) -> NeuralConstraintValidator:
    """Create a neural constraint validator."""
    return NeuralConstraintValidator(config)

def create_tree_constraint_validator(config: Dict[str, Any]) -> TreeConstraintValidator:
    """Create a tree constraint validator."""
    return TreeConstraintValidator(config)

def create_unified_constraint_validator(config: Dict[str, Any]) -> UnifiedConstraintValidator:
    """Create a unified constraint validator."""
    return UnifiedConstraintValidator(config)
