"""
NAS Optimization Configuration with Grid Utils and Hardware Integration.

This module provides comprehensive configuration for NAS parameter optimization
using existing grid utilities, matrix operations, and hardware optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import numpy as np

from src.utils.hardware.unified_hardware_manager import WorkloadType, OptimizationLevel
from .nas_bayesian_optimizer import OptimizationStrategy


@dataclass
class NASGridOptimizationConfig:
    """Configuration for grid-based optimization phase."""
    
    # Grid search parameters
    enable_coarse_grid: bool = True
    enable_fine_grid: bool = True
    coarse_grid_points: int = 8
    fine_grid_points: int = 5
    grid_phase_trials: int = 30
    
    # Grid refinement
    fine_grid_around_best: bool = True
    fine_grid_range_factor: float = 0.2  # 20% around best parameters
    fine_grid_trials: int = 15
    
    # Grid search space
    grid_search_space: Dict[str, Any] = field(default_factory=lambda: {
        # Architecture parameters - coarse grid
        'architecture_depth': {
            'type': 'int',
            'low': 3,
            'high': 9,
            'grid_points': 4
        },
        'hidden_units': {
            'type': 'int',
            'low': 32,
            'high': 256,
            'grid_points': 5
        },
        'activation_function': {
            'type': 'categorical',
            'choices': ['relu', 'tanh', 'swish', 'gelu']
        },
        'dropout_rate': {
            'type': 'float',
            'low': 0.1,
            'high': 0.4,
            'grid_points': 4
        },
        'learning_rate': {
            'type': 'float',
            'low': 0.001,
            'high': 0.1,
            'log': True,
            'grid_points': 5
        },
        
        # Sensitivity parameters - fine grid
        'micro_regime_sensitivity': {
            'type': 'float',
            'low': 0.5,
            'high': 0.9,
            'grid_points': 6
        },
        'economic_significance_threshold': {
            'type': 'float',
            'low': 0.5,
            'high': 0.9,
            'grid_points': 6
        },
        'trading_viability_threshold': {
            'type': 'float',
            'low': 0.4,
            'high': 0.8,
            'grid_points': 6
        }
    })


@dataclass
class NASMatrixOptimizationConfig:
    """Configuration for matrix operations optimization."""
    
    # Matrix operations settings
    enable_matrix_optimization: bool = True
    enable_batch_processing: bool = True
    enable_vectorization: bool = True
    enable_parallel_processing: bool = True
    
    # Batch processing
    batch_size: int = 1000
    max_batch_size: int = 5000
    adaptive_batch_size: bool = True
    
    # Memory optimization
    enable_memory_pooling: bool = True
    memory_pool_size: int = 1000
    enable_compression: bool = True
    
    # Matrix operations
    matrix_operations_backend: str = 'numpy'  # 'numpy', 'scipy', 'cupy'
    enable_gpu_acceleration: bool = True
    enable_mkl_optimization: bool = True
    
    # Performance monitoring
    enable_matrix_profiling: bool = True
    matrix_operation_timeout: float = 30.0
    enable_operation_caching: bool = True


@dataclass
class NASHardwareOptimizationConfig:
    """Configuration for hardware optimization."""
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    workload_type: WorkloadType = WorkloadType.ML_TRAINING
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    
    # CPU optimization
    enable_cpu_optimization: bool = True
    cpu_optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    enable_core_affinity: bool = True
    enable_thermal_monitoring: bool = True
    
    # GPU optimization
    enable_gpu_optimization: bool = True
    gpu_optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    enable_mps_acceleration: bool = True
    enable_gpu_memory_pooling: bool = True
    
    # Memory optimization
    enable_memory_optimization: bool = True
    memory_optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    memory_limit_gb: float = 8.0
    enable_memory_pooling: bool = True
    enable_predictive_allocation: bool = True
    
    # Adaptive optimization
    enable_adaptive_optimization: bool = True
    learning_enabled: bool = True
    auto_tuning_enabled: bool = True
    performance_monitoring_enabled: bool = True
    
    # Monitoring
    monitoring_interval: float = 5.0
    metrics_retention_hours: int = 24
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 85.0,
        'memory_usage': 90.0,
        'gpu_usage': 80.0,
        'temperature': 85.0
    })


@dataclass
class NASBayesianOptimizationConfig:
    """Configuration for Bayesian optimization phase."""
    
    # TPE optimization
    enable_tpe_optimization: bool = True
    n_trials: int = 100
    n_startup_trials: int = 20
    n_warmup_steps: int = 5
    n_ei_candidates: int = 24
    
    # Pruning and early stopping
    enable_pruning: bool = True
    pruning_patience: int = 10
    min_trial_duration: float = 30.0
    max_trial_duration: float = 300.0
    
    # Multi-objective optimization
    objectives: List[str] = field(default_factory=lambda: [
        'regime_stability',
        'economic_significance',
        'trading_viability',
        'micro_regime_accuracy'
    ])
    objective_weights: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.2, 0.2])
    
    # TPE search space
    tpe_search_space: Dict[str, Any] = field(default_factory=lambda: {
        # Architecture parameters
        'architecture_depth': {
            'type': 'int',
            'low': 3,
            'high': 9
        },
        'hidden_units': {
            'type': 'int',
            'low': 32,
            'high': 256
        },
        'activation_function': {
            'type': 'categorical',
            'choices': ['relu', 'tanh', 'swish', 'gelu']
        },
        'dropout_rate': {
            'type': 'float',
            'low': 0.1,
            'high': 0.4
        },
        'learning_rate': {
            'type': 'float',
            'low': 0.001,
            'high': 0.1,
            'log': True
        },
        
        # Sensitivity parameters
        'micro_regime_sensitivity': {
            'type': 'float',
            'low': 0.5,
            'high': 0.9
        },
        'economic_significance_threshold': {
            'type': 'float',
            'low': 0.5,
            'high': 0.9
        },
        'trading_viability_threshold': {
            'type': 'float',
            'low': 0.4,
            'high': 0.8
        },
        'regime_transition_cost': {
            'type': 'float',
            'low': 0.01,
            'high': 0.1
        },
        
        # Performance parameters
        'batch_size': {
            'type': 'int',
            'low': 500,
            'high': 2000
        },
        'max_memory_usage': {
            'type': 'float',
            'low': 0.6,
            'high': 0.9
        },
        
        # Validation thresholds
        'min_regime_stability': {
            'type': 'float',
            'low': 0.4,
            'high': 0.8
        },
        'min_economic_significance': {
            'type': 'float',
            'low': 0.5,
            'high': 0.9
        },
        'min_trading_viability': {
            'type': 'float',
            'low': 0.4,
            'high': 0.8
        },
        'max_regime_volatility': {
            'type': 'float',
            'low': 0.1,
            'high': 0.5
        }
    })


@dataclass
class NASOptimizationConfig:
    """Main NAS optimization configuration."""
    
    # Optimization strategy
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.GRID_FIRST
    
    # Component configurations
    grid_config: NASGridOptimizationConfig = field(default_factory=NASGridOptimizationConfig)
    matrix_config: NASMatrixOptimizationConfig = field(default_factory=NASMatrixOptimizationConfig)
    hardware_config: NASHardwareOptimizationConfig = field(default_factory=NASHardwareOptimizationConfig)
    bayesian_config: NASBayesianOptimizationConfig = field(default_factory=NASBayesianOptimizationConfig)
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    monitoring_interval: float = 5.0
    save_intermediate_results: bool = True
    save_final_results: bool = True
    
    # Results management
    results_directory: str = "nas_optimization_results"
    enable_result_visualization: bool = True
    enable_convergence_analysis: bool = True
    
    # Validation
    enable_parameter_validation: bool = True
    validation_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_score_improvement': 0.01,
        'max_parameter_variance': 0.1,
        'min_trial_success_rate': 0.8
    })
    
    @classmethod
    def create_short_term_trading_config(cls) -> 'NASOptimizationConfig':
        """Create configuration optimized for short-term trading."""
        return cls(
            optimization_strategy=OptimizationStrategy.GRID_FIRST,
            grid_config=NASGridOptimizationConfig(
                enable_coarse_grid=True,
                enable_fine_grid=True,
                coarse_grid_points=8,
                fine_grid_points=5,
                grid_phase_trials=30
            ),
            matrix_config=NASMatrixOptimizationConfig(
                enable_matrix_optimization=True,
                enable_batch_processing=True,
                batch_size=1000,
                enable_gpu_acceleration=True
            ),
            hardware_config=NASHardwareOptimizationConfig(
                enable_hardware_optimization=True,
                workload_type=WorkloadType.ML_TRAINING,
                optimization_level=OptimizationLevel.BALANCED
            ),
            bayesian_config=NASBayesianOptimizationConfig(
                enable_tpe_optimization=True,
                n_trials=100,
                objectives=['regime_stability', 'economic_significance', 'trading_viability', 'micro_regime_accuracy'],
                objective_weights=[0.3, 0.3, 0.2, 0.2]
            )
        )
    
    @classmethod
    def create_high_performance_config(cls) -> 'NASOptimizationConfig':
        """Create configuration optimized for high performance."""
        return cls(
            optimization_strategy=OptimizationStrategy.HYBRID,
            grid_config=NASGridOptimizationConfig(
                enable_coarse_grid=True,
                enable_fine_grid=True,
                coarse_grid_points=12,
                fine_grid_points=8,
                grid_phase_trials=50
            ),
            matrix_config=NASMatrixOptimizationConfig(
                enable_matrix_optimization=True,
                enable_batch_processing=True,
                batch_size=2000,
                enable_gpu_acceleration=True,
                enable_parallel_processing=True
            ),
            hardware_config=NASHardwareOptimizationConfig(
                enable_hardware_optimization=True,
                workload_type=WorkloadType.ML_TRAINING,
                optimization_level=OptimizationLevel.AGGRESSIVE,
                memory_limit_gb=16.0
            ),
            bayesian_config=NASBayesianOptimizationConfig(
                enable_tpe_optimization=True,
                n_trials=200,
                n_startup_trials=30,
                objectives=['regime_stability', 'economic_significance', 'trading_viability', 'micro_regime_accuracy'],
                objective_weights=[0.25, 0.25, 0.25, 0.25]
            )
        )
    
    @classmethod
    def create_quick_test_config(cls) -> 'NASOptimizationConfig':
        """Create configuration for quick testing."""
        return cls(
            optimization_strategy=OptimizationStrategy.TPE_ONLY,
            grid_config=NASGridOptimizationConfig(
                enable_coarse_grid=False,
                enable_fine_grid=False,
                grid_phase_trials=0
            ),
            matrix_config=NASMatrixOptimizationConfig(
                enable_matrix_optimization=True,
                batch_size=500
            ),
            hardware_config=NASHardwareOptimizationConfig(
                enable_hardware_optimization=False,
                optimization_level=OptimizationLevel.MINIMAL
            ),
            bayesian_config=NASBayesianOptimizationConfig(
                enable_tpe_optimization=True,
                n_trials=20,
                n_startup_trials=5,
                objectives=['regime_stability', 'economic_significance'],
                objective_weights=[0.5, 0.5]
            )
        )
    
    def get_combined_search_space(self) -> Dict[str, Any]:
        """Get combined search space for optimization."""
        # Start with grid search space
        combined_space = self.grid_config.grid_search_space.copy()
        
        # Add TPE-specific parameters
        for param, config in self.bayesian_config.tpe_search_space.items():
            if param not in combined_space:
                combined_space[param] = config
        
        return combined_space
    
    def validate_config(self) -> bool:
        """Validate optimization configuration."""
        try:
            # Validate grid configuration
            if self.grid_config.enable_coarse_grid and self.grid_config.coarse_grid_points < 2:
                return False
            
            # Validate matrix configuration
            if self.matrix_config.batch_size < 1:
                return False
            
            # Validate hardware configuration
            if self.hardware_config.memory_limit_gb < 1.0:
                return False
            
            # Validate Bayesian configuration
            if self.bayesian_config.n_trials < 1:
                return False
            
            # Validate objective weights
            if abs(sum(self.bayesian_config.objective_weights) - 1.0) > 0.01:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization configuration summary."""
        return {
            'strategy': self.optimization_strategy.value,
            'grid_enabled': self.grid_config.enable_coarse_grid,
            'tpe_enabled': self.bayesian_config.enable_tpe_optimization,
            'hardware_optimization': self.hardware_config.enable_hardware_optimization,
            'matrix_optimization': self.matrix_config.enable_matrix_optimization,
            'total_trials': (
                (self.grid_config.grid_phase_trials if self.grid_config.enable_coarse_grid else 0) +
                self.bayesian_config.n_trials
            ),
            'objectives': self.bayesian_config.objectives,
            'objective_weights': self.bayesian_config.objective_weights
        }