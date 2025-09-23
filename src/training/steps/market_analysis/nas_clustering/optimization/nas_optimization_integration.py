"""
NAS Optimization Integration with Existing Tools.

This module demonstrates how to integrate NAS Bayesian optimization with:
- Grid utilities for coarse-to-fine optimization
- Matrix operations for efficient computations
- Hardware optimization for performance
- Multi-objective optimization for regime detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from pathlib import Path
import json

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
from ..core.nas_clusterer import NASClusterer
from .nas_bayesian_optimizer import NASBayesianOptimizer, OptimizationStrategy
from .nas_optimization_config import NASOptimizationConfig

logger = logging.getLogger(__name__)


class NASOptimizationIntegration:
    """Integration class for NAS optimization with existing tools."""
    
    def __init__(self, config: Optional[NASOptimizationConfig] = None):
        """Initialize NAS optimization integration.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or NASOptimizationConfig.create_short_term_trading_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.matrix_ops = UnifiedMatrixOperations()
        self.hardware_manager = None
        self.optimization_results = {}
        
        # Initialize hardware optimization if enabled
        if self.config.hardware_config.enable_hardware_optimization:
            hardware_config = HardwareConfig(
                cpu_optimization_level=self.config.hardware_config.cpu_optimization_level,
                gpu_optimization_level=self.config.hardware_config.gpu_optimization_level,
                memory_optimization_level=self.config.hardware_config.memory_optimization_level,
                memory_limit_gb=self.config.hardware_config.memory_limit_gb,
                enable_adaptive_optimization=self.config.hardware_config.enable_adaptive_optimization,
                learning_enabled=self.config.hardware_config.learning_enabled,
                auto_tuning_enabled=self.config.hardware_config.auto_tuning_enabled
            )
            self.hardware_manager = UnifiedHardwareManager(hardware_config)
            self.logger.info("✅ Hardware optimization enabled")
        
        # Initialize matrix operations optimization
        if self.config.matrix_config.enable_matrix_optimization:
            self.logger.info("✅ Matrix operations optimization enabled")
        
        self.logger.info(f"✅ NAS Optimization Integration initialized with {self.config.optimization_strategy.value} strategy")
    
    def run_optimization(self,
                        market_data: pd.DataFrame,
                        features: np.ndarray,
                        timestamps: np.ndarray,
                        nas_config: NASClusteringConfig,
                        save_path: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive NAS optimization with all integrated tools.
        
        Args:
            market_data: Market data for optimization
            features: Feature matrix
            timestamps: Timestamps array
            nas_config: Base NAS configuration
            save_path: Path to save optimization results
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        self.logger.info("🚀 Starting comprehensive NAS optimization")
        
        # Initialize hardware optimization
        if self.hardware_manager:
            self.hardware_manager.start_optimization(
                workload_type=self.config.hardware_config.workload_type,
                optimization_level=self.config.hardware_config.optimization_level
            )
        
        try:
            # Phase 1: Grid search (if enabled)
            grid_results = {}
            if self.config.grid_config.enable_coarse_grid:
                self.logger.info("📊 Phase 1: Grid search optimization")
                grid_results = self._run_grid_optimization(market_data, features, timestamps, nas_config)
                self.logger.info(f"✅ Grid search completed: {len(grid_results.get('trials', []))} trials")
            
            # Phase 2: Bayesian optimization with TPE
            bayesian_results = {}
            if self.config.bayesian_config.enable_tpe_optimization:
                self.logger.info("🧠 Phase 2: Bayesian optimization with TPE")
                bayesian_results = self._run_bayesian_optimization(market_data, features, timestamps, nas_config)
                self.logger.info(f"✅ Bayesian optimization completed: {bayesian_results.get('n_trials', 0)} trials")
            
            # Combine results
            combined_results = self._combine_optimization_results(grid_results, bayesian_results)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics()
            hardware_metrics = self._get_hardware_metrics()
            matrix_metrics = self._get_matrix_operations_metrics()
            
            # Generate final results
            final_results = {
                'optimization_results': combined_results,
                'performance_metrics': performance_metrics,
                'hardware_metrics': hardware_metrics,
                'matrix_operations_metrics': matrix_metrics,
                'execution_time': time.time() - start_time,
                'configuration_summary': self.config.get_optimization_summary(),
                'recommendations': self._generate_recommendations(combined_results)
            }
            
            # Save results if path provided
            if save_path:
                self._save_optimization_results(final_results, save_path)
            
            self.logger.info(f"✅ Comprehensive NAS optimization completed in {final_results['execution_time']:.2f}s")
            return final_results
            
        finally:
            # Cleanup hardware optimization
            if self.hardware_manager:
                self.hardware_manager.stop_optimization()
    
    def _run_grid_optimization(self,
                             market_data: pd.DataFrame,
                             features: np.ndarray,
                             timestamps: np.ndarray,
                             nas_config: NASClusteringConfig) -> Dict[str, Any]:
        """Run grid search optimization using existing grid utilities."""
        self.logger.info("📊 Running grid search optimization")
        
        # Build coarse grid using existing grid utilities
        coarse_grid = build_coarse_grid_from_search_space(
            self.config.grid_config.grid_search_space,
            self.config.grid_config.coarse_grid_points
        )
        
        grid_results = {
            'trials': [],
            'best_score': -np.inf,
            'best_params': None,
            'phase': 'grid_coarse'
        }
        
        # Run coarse grid trials
        for i, params in enumerate(coarse_grid[:self.config.grid_config.grid_phase_trials]):
            try:
                score = self._evaluate_parameters_with_optimizations(
                    params, market_data, features, timestamps, nas_config
                )
                
                trial_result = {
                    'trial': i,
                    'params': params,
                    'score': score,
                    'phase': 'grid_coarse'
                }
                grid_results['trials'].append(trial_result)
                
                # Update best if improved
                if score > grid_results['best_score']:
                    grid_results['best_score'] = score
                    grid_results['best_params'] = params.copy()
                
                self.logger.info(f"Grid trial {i+1}/{len(coarse_grid)}: {score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Grid trial {i+1} failed: {e}")
                continue
        
        # Run fine grid around best parameters (if enabled)
        if (self.config.grid_config.enable_fine_grid and 
            self.config.grid_config.fine_grid_around_best and 
            grid_results['best_params']):
            
            self.logger.info("🔍 Running fine grid search around best parameters")
            
            # Build fine grid around best parameters
            fine_grid = build_fine_grid_around_best(
                self.config.grid_config.grid_search_space,
                grid_results['best_params'],
                self.config.grid_config.fine_grid_points
            )
            
            # Run fine grid trials
            for i, params in enumerate(fine_grid[:self.config.grid_config.fine_grid_trials]):
                try:
                    score = self._evaluate_parameters_with_optimizations(
                        params, market_data, features, timestamps, nas_config
                    )
                    
                    trial_result = {
                        'trial': len(grid_results['trials']),
                        'params': params,
                        'score': score,
                        'phase': 'grid_fine'
                    }
                    grid_results['trials'].append(trial_result)
                    
                    # Update best if improved
                    if score > grid_results['best_score']:
                        grid_results['best_score'] = score
                        grid_results['best_params'] = params.copy()
                    
                    self.logger.info(f"Fine grid trial {i+1}/{len(fine_grid)}: {score:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"Fine grid trial {i+1} failed: {e}")
                    continue
        
        return grid_results
    
    def _run_bayesian_optimization(self,
                                 market_data: pd.DataFrame,
                                 features: np.ndarray,
                                 timestamps: np.ndarray,
                                 nas_config: NASClusteringConfig) -> Dict[str, Any]:
        """Run Bayesian optimization with TPE."""
        self.logger.info("🧠 Running Bayesian optimization with TPE")
        
        # Create NAS Bayesian optimizer
        bayesian_optimizer = NASBayesianOptimizer(
            config=self.config.bayesian_config
        )
        
        # Run optimization
        optimization_result = bayesian_optimizer.optimize(
            market_data=market_data,
            features=features,
            timestamps=timestamps,
            nas_config=nas_config
        )
        
        return {
            'best_params': optimization_result.best_params,
            'best_score': optimization_result.best_score,
            'n_trials': optimization_result.n_trials,
            'optimization_history': optimization_result.optimization_history,
            'performance_metrics': optimization_result.performance_metrics,
            'convergence_analysis': optimization_result.convergence_analysis,
            'recommendations': optimization_result.recommendations
        }
    
    def _evaluate_parameters_with_optimizations(self,
                                               params: Dict[str, Any],
                                               market_data: pd.DataFrame,
                                               features: np.ndarray,
                                               timestamps: np.ndarray,
                                               nas_config: NASClusteringConfig) -> float:
        """Evaluate parameters with all optimizations enabled."""
        try:
            # Update NAS configuration with parameters
            updated_config = self._update_nas_config(nas_config, params)
            
            # Create NAS clusterer
            clusterer = NASClusterer(updated_config)
            
            # Run clustering with optimizations
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
            for i, objective in enumerate(self.config.bayesian_config.objectives):
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
                scores[obj] * self.config.bayesian_config.objective_weights[i] 
                for i, obj in enumerate(self.config.bayesian_config.objectives)
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
    
    def _combine_optimization_results(self, 
                                    grid_results: Dict[str, Any],
                                    bayesian_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from grid and Bayesian optimization."""
        combined_results = {
            'grid_results': grid_results,
            'bayesian_results': bayesian_results,
            'best_overall_score': -np.inf,
            'best_overall_params': None,
            'optimization_phases': []
        }
        
        # Compare grid and Bayesian results
        if grid_results.get('best_score', -np.inf) > combined_results['best_overall_score']:
            combined_results['best_overall_score'] = grid_results['best_score']
            combined_results['best_overall_params'] = grid_results['best_params']
            combined_results['best_phase'] = 'grid'
        
        if bayesian_results.get('best_score', -np.inf) > combined_results['best_overall_score']:
            combined_results['best_overall_score'] = bayesian_results['best_score']
            combined_results['best_overall_params'] = bayesian_results['best_params']
            combined_results['best_phase'] = 'bayesian'
        
        # Add optimization phases
        if grid_results.get('trials'):
            combined_results['optimization_phases'].append('grid_search')
        if bayesian_results.get('n_trials', 0) > 0:
            combined_results['optimization_phases'].append('bayesian_optimization')
        
        return combined_results
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from optimization."""
        metrics = {
            'hardware_optimization_enabled': self.hardware_manager is not None,
            'matrix_optimization_enabled': self.config.matrix_config.enable_matrix_optimization,
            'grid_optimization_enabled': self.config.grid_config.enable_coarse_grid,
            'bayesian_optimization_enabled': self.config.bayesian_config.enable_tpe_optimization
        }
        
        if self.hardware_manager:
            hardware_metrics = self.hardware_manager.get_performance_metrics()
            metrics.update({
                'hardware_metrics': hardware_metrics,
                'cpu_optimization_level': self.config.hardware_config.cpu_optimization_level.value,
                'gpu_optimization_level': self.config.hardware_config.gpu_optimization_level.value,
                'memory_optimization_level': self.config.hardware_config.memory_optimization_level.value
            })
        
        return metrics
    
    def _get_hardware_metrics(self) -> Dict[str, Any]:
        """Get hardware optimization metrics."""
        if not self.hardware_manager:
            return {'hardware_optimization': False}
        
        return {
            'hardware_optimization': True,
            'workload_type': self.config.hardware_config.workload_type.value,
            'optimization_level': self.config.hardware_config.optimization_level.value,
            'performance_metrics': self.hardware_manager.get_performance_metrics()
        }
    
    def _get_matrix_operations_metrics(self) -> Dict[str, Any]:
        """Get matrix operations metrics."""
        return {
            'matrix_operations_available': self.matrix_ops is not None,
            'optimization_enabled': self.config.matrix_config.enable_matrix_optimization,
            'batch_processing_enabled': self.config.matrix_config.enable_batch_processing,
            'gpu_acceleration_enabled': self.config.matrix_config.enable_gpu_acceleration,
            'batch_size': self.config.matrix_config.batch_size
        }
    
    def _generate_recommendations(self, combined_results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Performance recommendations
        best_score = combined_results.get('best_overall_score', 0.0)
        if best_score > 0.8:
            recommendations.append("Excellent optimization results achieved")
        elif best_score > 0.6:
            recommendations.append("Good optimization results, consider more trials")
        else:
            recommendations.append("Consider expanding search space or increasing trials")
        
        # Phase recommendations
        best_phase = combined_results.get('best_phase', 'unknown')
        if best_phase == 'grid':
            recommendations.append("Grid search found best parameters - consider fine-tuning with Bayesian optimization")
        elif best_phase == 'bayesian':
            recommendations.append("Bayesian optimization found best parameters - TPE was effective")
        
        # Hardware recommendations
        if self.hardware_manager:
            hardware_metrics = self.hardware_manager.get_performance_metrics()
            if hardware_metrics.get('cpu_usage', 0) > 80:
                recommendations.append("High CPU usage detected - consider reducing batch size or increasing hardware resources")
            if hardware_metrics.get('memory_usage', 0) > 85:
                recommendations.append("High memory usage detected - consider reducing batch size or enabling memory optimization")
        
        return recommendations
    
    def _save_optimization_results(self, results: Dict[str, Any], save_path: str) -> None:
        """Save optimization results to file."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        with open(save_path / 'nas_optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save configuration
        config_summary = self.config.get_optimization_summary()
        with open(save_path / 'optimization_config.json', 'w') as f:
            json.dump(config_summary, f, indent=2)
        
        self.logger.info(f"✅ Optimization results saved to {save_path}")


def create_nas_optimization_integration(config: Optional[NASOptimizationConfig] = None) -> NASOptimizationIntegration:
    """Create NAS optimization integration with default configuration."""
    if config is None:
        config = NASOptimizationConfig.create_short_term_trading_config()
    
    return NASOptimizationIntegration(config)