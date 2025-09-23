"""
NAS Bayesian Optimizer with Grid Utils and Hardware Integration.

This module provides Bayesian optimization for NAS parameters using:
- Grid utils for coarse-to-fine optimization
- Matrix operations for efficient computations
- Hardware optimization for performance
- Multi-objective optimization for regime detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import json
from pathlib import Path

# Import existing grid utilities
from src.utils.ml_common.optimization.grid_utils import (
    build_coarse_grid_from_search_space,
    build_fine_grid_around_best
)

# Import matrix operations
from src.utils.matrix_operations import UnifiedMatrixOperations

# Import hardware optimization
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel
)

# Import NAS components
from ..core.nas_config import NASClusteringConfig, NASArchitectureType
from ..core.nas_clusterer import NASClusterer, NASClusteringResult
from ..core.nas_regime_optimizer import NASRegimeOptimizer

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategy types."""
    GRID_FIRST = "grid_first"          # Use grid search before TPE
    TPE_ONLY = "tpe_only"              # Use TPE directly
    HYBRID = "hybrid"                  # Combine grid and TPE
    ADAPTIVE = "adaptive"              # Adaptive strategy selection


@dataclass
class NASOptimizationConfig:
    """Configuration for NAS Bayesian optimization."""
    
    # Optimization strategy
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.GRID_FIRST
    n_trials: int = 100
    n_startup_trials: int = 20
    n_warmup_steps: int = 5
    n_ei_candidates: int = 24
    
    # Grid search configuration
    grid_coarse_points: int = 8
    grid_fine_points: int = 5
    grid_phase_trials: int = 30
    
    # Multi-objective optimization
    objectives: List[str] = field(default_factory=lambda: [
        'regime_stability',
        'economic_significance', 
        'trading_viability',
        'micro_regime_accuracy'
    ])
    objective_weights: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.2, 0.2])
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    hardware_workload_type: WorkloadType = WorkloadType.ML_TRAINING
    hardware_optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    
    # Matrix operations optimization
    enable_matrix_optimization: bool = True
    enable_batch_processing: bool = True
    batch_size: int = 1000
    
    # Pruning and early stopping
    enable_pruning: bool = True
    pruning_patience: int = 10
    min_trial_duration: float = 30.0
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    monitoring_interval: float = 5.0
    save_intermediate_results: bool = True


@dataclass
class NASOptimizationResult:
    """Result of NAS optimization."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    hardware_metrics: Dict[str, Any]
    matrix_operations_metrics: Dict[str, Any]
    execution_time: float
    n_trials: int
    convergence_analysis: Dict[str, Any]
    recommendations: List[str]


class NASBayesianOptimizer:
    """Bayesian optimizer for NAS parameters with grid utils and hardware integration."""
    
    def __init__(self, config: NASOptimizationConfig):
        """Initialize NAS Bayesian optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.matrix_ops = UnifiedMatrixOperations()
        self.hardware_manager = None
        self.optimization_history = []
        self.best_score = -np.inf
        self.best_params = None
        
        # Initialize hardware optimization if enabled
        if config.enable_hardware_optimization:
            hardware_config = HardwareConfig(
                cpu_optimization_level=config.hardware_optimization_level,
                gpu_optimization_level=config.hardware_optimization_level,
                memory_optimization_level=config.hardware_optimization_level,
                enable_adaptive_optimization=True,
                learning_enabled=True,
                auto_tuning_enabled=True
            )
            self.hardware_manager = UnifiedHardwareManager(hardware_config)
            self.logger.info("✅ Hardware optimization enabled")
        
        # Define NAS search space
        self.nas_search_space = self._define_nas_search_space()
        
        self.logger.info(f"✅ NAS Bayesian Optimizer initialized with {config.optimization_strategy.value} strategy")
    
    def _define_nas_search_space(self) -> Dict[str, Any]:
        """Define NAS parameter search space for optimization."""
        return {
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
        }
    
    def optimize(self, 
                 market_data: pd.DataFrame,
                 features: np.ndarray,
                 timestamps: np.ndarray,
                 nas_config: NASClusteringConfig,
                 save_path: Optional[str] = None) -> NASOptimizationResult:
        """Optimize NAS parameters using Bayesian optimization.
        
        Args:
            market_data: Market data for optimization
            features: Feature matrix
            timestamps: Timestamps array
            nas_config: Base NAS configuration
            save_path: Path to save optimization results
            
        Returns:
            NASOptimizationResult with optimization results
        """
        start_time = time.time()
        self.logger.info("🚀 Starting NAS Bayesian optimization")
        
        # Initialize hardware optimization
        if self.hardware_manager:
            self.hardware_manager.start_optimization(
                workload_type=self.config.hardware_workload_type,
                optimization_level=self.config.hardware_optimization_level
            )
        
        try:
            # Phase 1: Grid search (if enabled)
            if self.config.optimization_strategy in [OptimizationStrategy.GRID_FIRST, OptimizationStrategy.HYBRID]:
                self.logger.info("📊 Phase 1: Coarse grid search")
                grid_results = self._run_grid_search(market_data, features, timestamps, nas_config)
                self.logger.info(f"✅ Grid search completed: {len(grid_results)} trials")
            
            # Phase 2: Bayesian optimization with TPE
            self.logger.info("🧠 Phase 2: Bayesian optimization with TPE")
            tpe_results = self._run_tpe_optimization(market_data, features, timestamps, nas_config)
            
            # Combine results
            optimization_result = self._combine_optimization_results(tpe_results)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics()
            hardware_metrics = self._get_hardware_metrics()
            matrix_metrics = self._get_matrix_operations_metrics()
            
            # Generate recommendations
            recommendations = self._generate_recommendations(optimization_result)
            
            # Create final result
            result = NASOptimizationResult(
                best_params=optimization_result['best_params'],
                best_score=optimization_result['best_score'],
                optimization_history=self.optimization_history,
                performance_metrics=performance_metrics,
                hardware_metrics=hardware_metrics,
                matrix_operations_metrics=matrix_metrics,
                execution_time=time.time() - start_time,
                n_trials=len(self.optimization_history),
                convergence_analysis=self._analyze_convergence(),
                recommendations=recommendations
            )
            
            # Save results if path provided
            if save_path:
                self._save_optimization_results(result, save_path)
            
            self.logger.info(f"✅ NAS optimization completed: {result.best_score:.4f} score in {result.execution_time:.2f}s")
            return result
            
        finally:
            # Cleanup hardware optimization
            if self.hardware_manager:
                self.hardware_manager.stop_optimization()
    
    def _run_grid_search(self, 
                        market_data: pd.DataFrame,
                        features: np.ndarray, 
                        timestamps: np.ndarray,
                        nas_config: NASClusteringConfig) -> List[Dict[str, Any]]:
        """Run coarse grid search for initial parameter exploration."""
        self.logger.info("📊 Running coarse grid search")
        
        # Build coarse grid
        coarse_grid = build_coarse_grid_from_search_space(
            self.nas_search_space, 
            self.config.grid_coarse_points
        )
        
        grid_results = []
        for i, params in enumerate(coarse_grid[:self.config.grid_phase_trials]):
            try:
                # Evaluate parameters
                score = self._evaluate_parameters(
                    params, market_data, features, timestamps, nas_config
                )
                
                result = {
                    'trial': i,
                    'params': params,
                    'score': score,
                    'phase': 'grid_coarse'
                }
                grid_results.append(result)
                self.optimization_history.append(result)
                
                # Update best if improved
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                
                self.logger.info(f"Grid trial {i+1}/{len(coarse_grid)}: {score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Grid trial {i+1} failed: {e}")
                continue
        
        return grid_results
    
    def _run_tpe_optimization(self,
                             market_data: pd.DataFrame,
                             features: np.ndarray,
                             timestamps: np.ndarray, 
                             nas_config: NASClusteringConfig) -> optuna.Study:
        """Run TPE-based Bayesian optimization."""
        self.logger.info("🧠 Running TPE optimization")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_warmup_steps=self.config.n_warmup_steps,
                n_ei_candidates=self.config.n_ei_candidates
            ),
            pruner=MedianPruner() if self.config.enable_pruning else None
        )
        
        # Define objective function
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_config in self.nas_search_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'], 
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            
            # Evaluate parameters
            score = self._evaluate_parameters(
                params, market_data, features, timestamps, nas_config
            )
            
            # Record trial
            trial_result = {
                'trial': len(self.optimization_history),
                'params': params,
                'score': score,
                'phase': 'tpe'
            }
            self.optimization_history.append(trial_result)
            
            # Update best if improved
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
            
            return score
        
        # Run optimization
        study.optimize(
            objective, 
            n_trials=self.config.n_trials,
            timeout=None
        )
        
        return study
    
    def _evaluate_parameters(self,
                           params: Dict[str, Any],
                           market_data: pd.DataFrame,
                           features: np.ndarray,
                           timestamps: np.ndarray,
                           nas_config: NASClusteringConfig) -> float:
        """Evaluate NAS parameters and return multi-objective score."""
        try:
            # Update NAS configuration with parameters
            updated_config = self._update_nas_config(nas_config, params)
            
            # Create NAS clusterer
            clusterer = NASClusterer(updated_config)
            
            # Run clustering
            result = clusterer.cluster(
                data=market_data,
                timestamps=timestamps,
                optimize_parameters=False,  # Already optimizing
                generate_report=False
            )
            
            if not result.success:
                return 0.0
            
            # Calculate multi-objective score
            scores = {}
            for i, objective in enumerate(self.config.objectives):
                if objective == 'regime_stability':
                    scores[objective] = result.quality_metrics.get('regime_stability', 0.0)
                elif objective == 'economic_significance':
                    scores[objective] = result.quality_metrics.get('economic_significance', 0.0)
                elif objective == 'trading_viability':
                    scores[objective] = result.quality_metrics.get('trading_viability', 0.0)
                elif objective == 'micro_regime_accuracy':
                    scores[objective] = result.quality_metrics.get('micro_regime_accuracy', 0.0)
                else:
                    scores[objective] = 0.0
            
            # Calculate weighted score
            weighted_score = sum(
                scores[obj] * self.config.objective_weights[i] 
                for i, obj in enumerate(self.config.objectives)
            )
            
            return weighted_score
            
        except Exception as e:
            self.logger.warning(f"Parameter evaluation failed: {e}")
            return 0.0
    
    def _update_nas_config(self, 
                          base_config: NASClusteringConfig,
                          params: Dict[str, Any]) -> NASClusteringConfig:
        """Update NAS configuration with optimization parameters."""
        # Create updated config
        updated_config = NASClusteringConfig(
            timeframe=base_config.timeframe,
            micro_timeframe=base_config.micro_timeframe,
            n_regimes=base_config.n_regimes,
            min_regime_duration=base_config.min_regime_duration,
            max_regime_duration=base_config.max_regime_duration,
            data_driven_regimes=base_config.data_driven_regimes,
            nas_architecture_type=base_config.nas_architecture_type,
            enable_micro_regime_detection=base_config.enable_micro_regime_detection,
            exclude_complex_features=base_config.exclude_complex_features,
            include_technical_indicators=base_config.include_technical_indicators,
            include_volume_features=base_config.include_volume_features,
            include_volatility_features=base_config.include_volatility_features,
            include_momentum_features=base_config.include_momentum_features,
            include_trend_features=base_config.include_trend_features,
            short_term_optimization=base_config.short_term_optimization,
            enable_hardware_acceleration=base_config.enable_hardware_acceleration,
            enable_matrix_optimization=base_config.enable_matrix_optimization
        )
        
        # Update with optimization parameters
        updated_config.micro_regime_sensitivity = params.get('micro_regime_sensitivity', 0.7)
        updated_config.economic_significance_threshold = params.get('economic_significance_threshold', 0.7)
        updated_config.trading_viability_threshold = params.get('trading_viability_threshold', 0.6)
        updated_config.regime_transition_cost = params.get('regime_transition_cost', 0.05)
        
        # Update validation thresholds
        updated_config.validation_thresholds = {
            'min_regime_stability': params.get('min_regime_stability', 0.6),
            'min_economic_significance': params.get('min_economic_significance', 0.7),
            'min_trading_viability': params.get('min_trading_viability', 0.6),
            'max_regime_volatility': params.get('max_regime_volatility', 0.3)
        }
        
        # Update NAS search space with architecture parameters
        updated_config.nas_search_space = {
            'architecture_depth': [params.get('architecture_depth', 5)],
            'hidden_units': [params.get('hidden_units', 128)],
            'activation_functions': [params.get('activation_function', 'relu')],
            'dropout_rates': [params.get('dropout_rate', 0.2)],
            'learning_rates': [params.get('learning_rate', 0.01)]
        }
        
        return updated_config
    
    def _combine_optimization_results(self, tpe_study: optuna.Study) -> Dict[str, Any]:
        """Combine optimization results from different phases."""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'tpe_best_params': tpe_study.best_params,
            'tpe_best_score': tpe_study.best_value,
            'n_trials': len(self.optimization_history)
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from optimization."""
        if not self.optimization_history:
            return {}
        
        scores = [trial['score'] for trial in self.optimization_history]
        return {
            'best_score': max(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'improvement_rate': self._calculate_improvement_rate(),
            'convergence_speed': self._calculate_convergence_speed()
        }
    
    def _get_hardware_metrics(self) -> Dict[str, Any]:
        """Get hardware optimization metrics."""
        if not self.hardware_manager:
            return {}
        
        return self.hardware_manager.get_performance_metrics()
    
    def _get_matrix_operations_metrics(self) -> Dict[str, Any]:
        """Get matrix operations metrics."""
        return {
            'matrix_operations_available': self.matrix_ops is not None,
            'optimization_enabled': self.config.enable_matrix_optimization
        }
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate improvement rate over optimization."""
        if len(self.optimization_history) < 2:
            return 0.0
        
        initial_score = self.optimization_history[0]['score']
        final_score = self.optimization_history[-1]['score']
        
        if initial_score == 0:
            return 0.0
        
        return (final_score - initial_score) / initial_score
    
    def _calculate_convergence_speed(self) -> float:
        """Calculate convergence speed."""
        if len(self.optimization_history) < 10:
            return 0.0
        
        # Calculate how quickly we reach 90% of best score
        best_score = max(trial['score'] for trial in self.optimization_history)
        target_score = 0.9 * best_score
        
        for i, trial in enumerate(self.optimization_history):
            if trial['score'] >= target_score:
                return i / len(self.optimization_history)
        
        return 1.0
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence patterns."""
        if len(self.optimization_history) < 5:
            return {'converged': False, 'reason': 'insufficient_trials'}
        
        scores = [trial['score'] for trial in self.optimization_history]
        
        # Check for convergence in last 20% of trials
        last_20_percent = int(0.2 * len(scores))
        recent_scores = scores[-last_20_percent:]
        
        if len(recent_scores) < 2:
            return {'converged': False, 'reason': 'insufficient_recent_trials'}
        
        score_std = np.std(recent_scores)
        score_mean = np.mean(recent_scores)
        
        # Consider converged if std is less than 5% of mean
        converged = score_std < 0.05 * score_mean
        
        return {
            'converged': converged,
            'final_std': score_std,
            'final_mean': score_mean,
            'improvement_trend': self._calculate_improvement_trend()
        }
    
    def _calculate_improvement_trend(self) -> str:
        """Calculate improvement trend."""
        if len(self.optimization_history) < 10:
            return 'insufficient_data'
        
        scores = [trial['score'] for trial in self.optimization_history]
        
        # Split into first and second half
        mid_point = len(scores) // 2
        first_half = scores[:mid_point]
        second_half = scores[mid_point:]
        
        first_mean = np.mean(first_half)
        second_mean = np.mean(second_half)
        
        if second_mean > first_mean * 1.05:
            return 'improving'
        elif second_mean < first_mean * 0.95:
            return 'degrading'
        else:
            return 'stable'
    
    def _generate_recommendations(self, optimization_result: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Performance recommendations
        if optimization_result['best_score'] > 0.8:
            recommendations.append("Excellent optimization results achieved")
        elif optimization_result['best_score'] > 0.6:
            recommendations.append("Good optimization results, consider more trials")
        else:
            recommendations.append("Consider expanding search space or increasing trials")
        
        # Parameter recommendations
        best_params = optimization_result['best_params']
        if best_params:
            if best_params.get('architecture_depth', 5) > 7:
                recommendations.append("High architecture depth detected - consider regularization")
            if best_params.get('dropout_rate', 0.2) > 0.3:
                recommendations.append("High dropout rate - consider reducing for better learning")
            if best_params.get('learning_rate', 0.01) < 0.005:
                recommendations.append("Low learning rate - consider increasing for faster convergence")
        
        return recommendations
    
    def _save_optimization_results(self, result: NASOptimizationResult, save_path: str) -> None:
        """Save optimization results to file."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save results as JSON
        results_dict = {
            'best_params': result.best_params,
            'best_score': result.best_score,
            'execution_time': result.execution_time,
            'n_trials': result.n_trials,
            'performance_metrics': result.performance_metrics,
            'recommendations': result.recommendations
        }
        
        with open(save_path / 'nas_optimization_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save optimization history
        with open(save_path / 'optimization_history.json', 'w') as f:
            json.dump(result.optimization_history, f, indent=2)
        
        self.logger.info(f"✅ Optimization results saved to {save_path}")


def create_nas_optimizer(config: Optional[NASOptimizationConfig] = None) -> NASBayesianOptimizer:
    """Create NAS Bayesian optimizer with default configuration."""
    if config is None:
        config = NASOptimizationConfig()
    
    return NASBayesianOptimizer(config)