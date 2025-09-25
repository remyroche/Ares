"""
Agnostic Configuration Module

Agnostic configuration and results system that can be used by both NAS and TAS
components with architecture-specific adaptations.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime

# Import shared utilities
try:
    from src.utils.common_operations import (
        memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
        safe_json_dump, safe_json_load, ensure_directory
    )
    from src.utils.math_validation import MathValidation
    from src.utils.serialization_utils import UniversalSerializer
    from src.utils.tprint import (
        tprint, tprint_info, tprint_debug, tprint_warning, tprint_error,
        tprint_success, tprint_progress, tprint_performance
    )
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args)
    def tprint_info(*args, **kwargs): print("INFO:", *args)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args)
    def tprint_error(*args, **kwargs): print("ERROR:", *args)

logger = logging.getLogger(__name__)

@dataclass
class AgnosticConfig:
    """Agnostic configuration for NAS and TAS components."""
    
    # General settings
    component_type: str = "agnostic"  # nas, tas, agnostic
    algorithm: str = "default"
    random_state: int = 42
    
    # Performance settings
    enable_optimization: bool = True
    optimization_method: str = "bayesian_tpe"  # bayesian_tpe, grid_search, random_search
    n_trials: int = 50
    cv_folds: int = 5
    
    # Hardware optimization
    enable_m1_optimization: bool = True
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    memory_limit_gb: Optional[float] = None
    
    # Monitoring and logging
    verbose: bool = True
    log_level: str = "INFO"
    save_results: bool = True
    
    # Output settings
    output_dir: str = "agnostic_results"
    results_format: str = "json"  # json, pickle
    
    # Architecture-specific parameters (to be extended by subclasses)
    architecture_params: Dict[str, Any] = field(default_factory=dict)
    
    # Performance thresholds
    performance_threshold: float = 0.0
    convergence_threshold: float = 0.01
    early_stopping_patience: int = 10
    
    # Validation settings
    enable_cross_validation: bool = True
    validation_split: float = 0.2
    enable_early_stopping: bool = True
    
    # Feature processing
    enable_feature_selection: bool = True
    enable_feature_scaling: bool = True
    enable_dimensionality_reduction: bool = True
    
    # Meta-learning settings
    enable_meta_learning: bool = False
    meta_learning_rate: float = 0.01
    meta_learning_adaptations: int = 5
    
    # Regime detection
    enable_regime_detection: bool = False
    regime_detection_threshold: float = 0.1
    regime_adaptation_factor: float = 1.2

@dataclass
class AgnosticResults:
    """Agnostic results for NAS and TAS components."""
    
    # Basic results
    success: bool
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Performance metrics
    primary_score: float = 0.0
    secondary_scores: Dict[str, float] = field(default_factory=dict)
    cross_validation_scores: Optional[List[float]] = None
    
    # Model information
    model_type: str = ""
    model_complexity: int = 0
    n_parameters: int = 0
    
    # Feature information
    n_features_original: int = 0
    n_features_used: int = 0
    feature_importance: Optional[np.ndarray] = None
    selected_features: Optional[List[int]] = None
    
    # Architecture-specific results
    architecture_results: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization results
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    best_parameters: Dict[str, Any] = field(default_factory=dict)
    convergence_achieved: bool = False
    
    # Meta-learning results
    meta_learning_applied: bool = False
    meta_learning_improvements: List[float] = field(default_factory=list)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Regime analysis
    regime_detected: bool = False
    regime_confidence: float = 0.0
    regime_adaptations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config_used: Dict[str, Any] = field(default_factory=dict)

class AgnosticConfigManager:
    """
    Agnostic Configuration Manager.
    
    Manages configuration and results for both NAS and TAS components
    with architecture-specific adaptations.
    """
    
    def __init__(self, config: Optional[AgnosticConfig] = None):
        """Initialize agnostic config manager."""
        self.config = config or AgnosticConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize utilities
        self._init_utilities()
        
        # Configuration state
        self.config_history = []
        self.results_history = []
        
        tprint_success("🚀 Agnostic Config Manager initialized")
        tprint_info(f"   → Component type: {self.config.component_type}")
        tprint_info(f"   → Optimization: {'enabled' if self.config.enable_optimization else 'disabled'}")
        tprint_info(f"   → M1 optimization: {'enabled' if self.config.enable_m1_optimization else 'disabled'}")
    
    def _init_utilities(self):
        """Initialize utility components."""
        if SHARED_UTILS_AVAILABLE:
            self.math_validator = MathValidation()
            self.serializer = UniversalSerializer()
        else:
            self.math_validator = None
            self.serializer = None
    
    def create_nas_config(self, **kwargs) -> AgnosticConfig:
        """Create NAS-specific configuration."""
        config = AgnosticConfig(**kwargs)
        config.component_type = "nas"
        config.algorithm = "neural_architecture_search"
        
        # NAS-specific parameters
        config.architecture_params.update({
            'neural_architecture': 'mlp',
            'layer_types': ['dense', 'conv', 'lstm'],
            'activation_functions': ['relu', 'tanh', 'sigmoid'],
            'optimization_algorithms': ['adam', 'sgd', 'rmsprop'],
            'learning_rates': [0.001, 0.01, 0.1],
            'batch_sizes': [16, 32, 64, 128],
            'epochs': [50, 100, 200, 500]
        })
        
        # NAS-specific settings
        config.enable_meta_learning = True
        config.meta_learning_rate = 0.001
        config.enable_regime_detection = True
        config.regime_detection_threshold = 0.05
        
        tprint_info("✅ NAS configuration created")
        return config
    
    def create_tas_config(self, **kwargs) -> AgnosticConfig:
        """Create TAS-specific configuration."""
        config = AgnosticConfig(**kwargs)
        config.component_type = "tas"
        config.algorithm = "tree_architecture_search"
        
        # TAS-specific parameters
        config.architecture_params.update({
            'tree_types': ['random_forest', 'xgboost', 'lightgbm', 'decision_tree'],
            'max_depths': [3, 5, 10, 15, 20],
            'min_samples_splits': [2, 5, 10, 20],
            'min_samples_leaves': [1, 2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.5, 0.8, 1.0],
            'n_estimators': [50, 100, 200, 500, 1000]
        })
        
        # TAS-specific settings
        config.enable_meta_learning = True
        config.meta_learning_rate = 0.01
        config.enable_regime_detection = True
        config.regime_detection_threshold = 0.1
        
        tprint_info("✅ TAS configuration created")
        return config
    
    def create_agnostic_config(self, **kwargs) -> AgnosticConfig:
        """Create agnostic configuration."""
        config = AgnosticConfig(**kwargs)
        config.component_type = "agnostic"
        config.algorithm = "agnostic"
        
        # Agnostic parameters
        config.architecture_params.update({
            'generic_parameters': ['param1', 'param2', 'param3'],
            'optimization_ranges': {'param1': (0, 1), 'param2': (0, 10), 'param3': (0, 100)}
        })
        
        tprint_info("✅ Agnostic configuration created")
        return config
    
    def validate_config(self, config: AgnosticConfig) -> Tuple[bool, List[str]]:
        """Validate configuration."""
        errors = []
        
        # Basic validation
        if not config.component_type:
            errors.append("Component type cannot be empty")
        
        if config.n_trials <= 0:
            errors.append("Number of trials must be positive")
        
        if config.cv_folds <= 0:
            errors.append("Number of CV folds must be positive")
        
        if config.validation_split <= 0 or config.validation_split >= 1:
            errors.append("Validation split must be between 0 and 1")
        
        # Architecture-specific validation
        if config.component_type == "nas":
            if 'neural_architecture' not in config.architecture_params:
                errors.append("NAS config missing neural_architecture parameter")
        
        elif config.component_type == "tas":
            if 'tree_types' not in config.architecture_params:
                errors.append("TAS config missing tree_types parameter")
        
        # Hardware validation
        if config.memory_limit_gb is not None and config.memory_limit_gb <= 0:
            errors.append("Memory limit must be positive")
        
        return len(errors) == 0, errors
    
    def save_config(self, config: AgnosticConfig, filepath: Optional[str] = None) -> str:
        """Save configuration to file."""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{config.component_type}_config_{timestamp}.json"
                filepath = Path(config.output_dir) / filename
            
            filepath = Path(filepath)
            ensure_directory(filepath.parent)
            
            # Prepare config data
            config_data = {
                'component_type': config.component_type,
                'algorithm': config.algorithm,
                'random_state': config.random_state,
                'enable_optimization': config.enable_optimization,
                'optimization_method': config.optimization_method,
                'n_trials': config.n_trials,
                'cv_folds': config.cv_folds,
                'enable_m1_optimization': config.enable_m1_optimization,
                'enable_parallel_processing': config.enable_parallel_processing,
                'n_jobs': config.n_jobs,
                'memory_limit_gb': config.memory_limit_gb,
                'verbose': config.verbose,
                'log_level': config.log_level,
                'save_results': config.save_results,
                'output_dir': config.output_dir,
                'results_format': config.results_format,
                'architecture_params': config.architecture_params,
                'performance_threshold': config.performance_threshold,
                'convergence_threshold': config.convergence_threshold,
                'early_stopping_patience': config.early_stopping_patience,
                'enable_cross_validation': config.enable_cross_validation,
                'validation_split': config.validation_split,
                'enable_early_stopping': config.enable_early_stopping,
                'enable_feature_selection': config.enable_feature_selection,
                'enable_feature_scaling': config.enable_feature_scaling,
                'enable_dimensionality_reduction': config.enable_dimensionality_reduction,
                'enable_meta_learning': config.enable_meta_learning,
                'meta_learning_rate': config.meta_learning_rate,
                'meta_learning_adaptations': config.meta_learning_adaptations,
                'enable_regime_detection': config.enable_regime_detection,
                'regime_detection_threshold': config.regime_detection_threshold,
                'regime_adaptation_factor': config.regime_adaptation_factor
            }
            
            if self.serializer:
                self.serializer.save(config_data, str(filepath))
            else:
                safe_json_dump(config_data, filepath)
            
            tprint_success(f"💾 Configuration saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            tprint_error(f"❌ Failed to save configuration: {e}")
            raise
    
    def load_config(self, filepath: str) -> AgnosticConfig:
        """Load configuration from file."""
        try:
            filepath = Path(filepath)
            
            if self.serializer:
                config_data = self.serializer.load(str(filepath))
            else:
                config_data = safe_json_load(filepath)
            
            # Create config object
            config = AgnosticConfig(
                component_type=config_data.get('component_type', 'agnostic'),
                algorithm=config_data.get('algorithm', 'default'),
                random_state=config_data.get('random_state', 42),
                enable_optimization=config_data.get('enable_optimization', True),
                optimization_method=config_data.get('optimization_method', 'bayesian_tpe'),
                n_trials=config_data.get('n_trials', 50),
                cv_folds=config_data.get('cv_folds', 5),
                enable_m1_optimization=config_data.get('enable_m1_optimization', True),
                enable_parallel_processing=config_data.get('enable_parallel_processing', True),
                n_jobs=config_data.get('n_jobs', -1),
                memory_limit_gb=config_data.get('memory_limit_gb'),
                verbose=config_data.get('verbose', True),
                log_level=config_data.get('log_level', 'INFO'),
                save_results=config_data.get('save_results', True),
                output_dir=config_data.get('output_dir', 'agnostic_results'),
                results_format=config_data.get('results_format', 'json'),
                architecture_params=config_data.get('architecture_params', {}),
                performance_threshold=config_data.get('performance_threshold', 0.0),
                convergence_threshold=config_data.get('convergence_threshold', 0.01),
                early_stopping_patience=config_data.get('early_stopping_patience', 10),
                enable_cross_validation=config_data.get('enable_cross_validation', True),
                validation_split=config_data.get('validation_split', 0.2),
                enable_early_stopping=config_data.get('enable_early_stopping', True),
                enable_feature_selection=config_data.get('enable_feature_selection', True),
                enable_feature_scaling=config_data.get('enable_feature_scaling', True),
                enable_dimensionality_reduction=config_data.get('enable_dimensionality_reduction', True),
                enable_meta_learning=config_data.get('enable_meta_learning', False),
                meta_learning_rate=config_data.get('meta_learning_rate', 0.01),
                meta_learning_adaptations=config_data.get('meta_learning_adaptations', 5),
                enable_regime_detection=config_data.get('enable_regime_detection', False),
                regime_detection_threshold=config_data.get('regime_detection_threshold', 0.1),
                regime_adaptation_factor=config_data.get('regime_adaptation_factor', 1.2)
            )
            
            tprint_success(f"✅ Configuration loaded from {filepath}")
            return config
            
        except Exception as e:
            tprint_error(f"❌ Failed to load configuration: {e}")
            raise
    
    def save_results(self, results: AgnosticResults, filepath: Optional[str] = None) -> str:
        """Save results to file."""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{results.model_type}_results_{timestamp}.{self.config.results_format}"
                filepath = Path(self.config.output_dir) / filename
            
            filepath = Path(filepath)
            ensure_directory(filepath.parent)
            
            # Prepare results data
            results_data = {
                'success': results.success,
                'execution_time': results.execution_time,
                'memory_usage_mb': results.memory_usage_mb,
                'primary_score': results.primary_score,
                'secondary_scores': results.secondary_scores,
                'cross_validation_scores': results.cross_validation_scores,
                'model_type': results.model_type,
                'model_complexity': results.model_complexity,
                'n_parameters': results.n_parameters,
                'n_features_original': results.n_features_original,
                'n_features_used': results.n_features_used,
                'feature_importance': results.feature_importance.tolist() if results.feature_importance is not None else None,
                'selected_features': results.selected_features,
                'architecture_results': results.architecture_results,
                'optimization_history': results.optimization_history,
                'best_parameters': results.best_parameters,
                'convergence_achieved': results.convergence_achieved,
                'meta_learning_applied': results.meta_learning_applied,
                'meta_learning_improvements': results.meta_learning_improvements,
                'adaptation_history': results.adaptation_history,
                'regime_detected': results.regime_detected,
                'regime_confidence': results.regime_confidence,
                'regime_adaptations': results.regime_adaptations,
                'error_message': results.error_message,
                'warnings': results.warnings,
                'timestamp': results.timestamp,
                'config_used': results.config_used
            }
            
            if self.serializer:
                self.serializer.save(results_data, str(filepath))
            else:
                safe_json_dump(results_data, filepath)
            
            tprint_success(f"💾 Results saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            tprint_error(f"❌ Failed to save results: {e}")
            raise
    
    def load_results(self, filepath: str) -> AgnosticResults:
        """Load results from file."""
        try:
            filepath = Path(filepath)
            
            if self.serializer:
                results_data = self.serializer.load(str(filepath))
            else:
                results_data = safe_json_load(filepath)
            
            # Create results object
            results = AgnosticResults(
                success=results_data.get('success', False),
                execution_time=results_data.get('execution_time', 0.0),
                memory_usage_mb=results_data.get('memory_usage_mb', 0.0),
                primary_score=results_data.get('primary_score', 0.0),
                secondary_scores=results_data.get('secondary_scores', {}),
                cross_validation_scores=results_data.get('cross_validation_scores'),
                model_type=results_data.get('model_type', ''),
                model_complexity=results_data.get('model_complexity', 0),
                n_parameters=results_data.get('n_parameters', 0),
                n_features_original=results_data.get('n_features_original', 0),
                n_features_used=results_data.get('n_features_used', 0),
                feature_importance=np.array(results_data.get('feature_importance')) if results_data.get('feature_importance') is not None else None,
                selected_features=results_data.get('selected_features'),
                architecture_results=results_data.get('architecture_results', {}),
                optimization_history=results_data.get('optimization_history', []),
                best_parameters=results_data.get('best_parameters', {}),
                convergence_achieved=results_data.get('convergence_achieved', False),
                meta_learning_applied=results_data.get('meta_learning_applied', False),
                meta_learning_improvements=results_data.get('meta_learning_improvements', []),
                adaptation_history=results_data.get('adaptation_history', []),
                regime_detected=results_data.get('regime_detected', False),
                regime_confidence=results_data.get('regime_confidence', 0.0),
                regime_adaptations=results_data.get('regime_adaptations', []),
                error_message=results_data.get('error_message'),
                warnings=results_data.get('warnings', []),
                timestamp=results_data.get('timestamp', datetime.now().isoformat()),
                config_used=results_data.get('config_used', {})
            )
            
            tprint_success(f"✅ Results loaded from {filepath}")
            return results
            
        except Exception as e:
            tprint_error(f"❌ Failed to load results: {e}")
            raise
    
    def get_config_summary(self):
        """Get configuration summary."""
        return {
            'component_type': self.config.component_type,
            'algorithm': self.config.algorithm,
            'optimization_enabled': self.config.enable_optimization,
            'm1_optimization_enabled': self.config.enable_m1_optimization,
            'meta_learning_enabled': self.config.enable_meta_learning,
            'regime_detection_enabled': self.config.enable_regime_detection,
            'architecture_params': self.config.architecture_params,
            'config_history_length': len(self.config_history),
            'results_history_length': len(self.results_history)
        }

# Factory functions for creating architecture-specific configurations
def create_nas_config(**kwargs) -> AgnosticConfig:
    """Create NAS-specific configuration."""
    manager = AgnosticConfigManager()
    return manager.create_nas_config(**kwargs)

def create_tas_config(**kwargs) -> AgnosticConfig:
    """Create TAS-specific configuration."""
    manager = AgnosticConfigManager()
    return manager.create_tas_config(**kwargs)

def create_agnostic_config(**kwargs) -> AgnosticConfig:
    """Create agnostic configuration."""
    manager = AgnosticConfigManager()
    return manager.create_agnostic_config(**kwargs)