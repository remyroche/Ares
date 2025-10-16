"""
Automatic Timeframe Optimizer for Training Pipeline Integration

This module provides automatic timeframe optimization for Analyst and Tactician
model training by integrating the research framework with the training pipeline.

Key Features:
- Automatic discovery of optimal timeframes for each model type
- Integration with existing training pipeline
- Model-specific optimization (Analyst vs Tactician)
- Performance monitoring and validation
- Fallback to default configurations when optimization fails
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json
from pathlib import Path

from src.utils.logger import get_logger
from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonConfig

# Import optimization components
try:
    from research.profit_labeling.dynamic_target_optimizer import (
        JointTargetHorizonOptimizer,
        DynamicOptimizationConfig,
        OptimizationMethod,
        OptimizationObjective
    )
    from research.profit_labeling.heuristic_analyzer import (
        HeuristicAnalyzer,
        HeuristicAnalysisConfig
    )
    from research.profit_labeling.labeling_validator import (
        LabelingValidator,
        ValidationConfig
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    OPTIMIZATION_AVAILABLE = False
    print(f"⚠️ Optimization components not available: {e}")

class ModelType(Enum):
    """Enumeration of model types for optimization."""
    ANALYST = "analyst"
    TACTICIAN = "tactician"
    BOTH = "both"

@dataclass
class OptimizationResult:
    """Result container for timeframe optimization."""
    model_type: ModelType
    optimal_config: MultiHorizonConfig
    optimization_score: float
    validation_score: float
    performance_metrics: Dict[str, float]
    optimization_time: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_type': self.model_type.value,
            'optimal_config': {
                'time_horizons': self.optimal_config.time_horizons,
                'profit_targets': self.optimal_config.profit_targets,
                'transaction_cost': self.optimal_config.transaction_cost
            },
            'optimization_score': self.optimization_score,
            'validation_score': self.validation_score,
            'performance_metrics': self.performance_metrics,
            'optimization_time': self.optimization_time,
            'timestamp': self.timestamp.isoformat()
        }

class AutomaticTimeframeOptimizer:
    """
    Automatic timeframe optimizer for training pipeline integration.

    This class provides automatic discovery of optimal timeframes for
    Analyst and Tactician model training using the research framework.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize automatic timeframe optimizer."""
        self.logger = get_logger('AutomaticTimeframeOptimizer')
        self.config = config or {}
        self.optimization_enabled = OPTIMIZATION_AVAILABLE

        if self.optimization_enabled:
            self._initialize_optimization_components()
            self.logger.info('🎯 Automatic timeframe optimization ENABLED')
        else:
            self.logger.warning('⚠️ Automatic timeframe optimization DISABLED')

        # Results storage
        self.optimization_results: Dict[ModelType, OptimizationResult] = {}
        self.optimization_history: List[OptimizationResult] = []

        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = {}

    def _initialize_optimization_components(self):
        """Initialize optimization components."""
        try:
            # Analyst-specific optimization config (15m base timeframe, 1-16 periods)
            self.analyst_config = DynamicOptimizationConfig(
                optimization_method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
                min_horizon=1,  # 15 minutes (1 * 15m)
                max_horizon=16,  # 240 minutes (16 * 15m)
                horizon_step=1,  # Test every period from 1-16
                optimization_objective=OptimizationObjective.MULTI_OBJECTIVE,
                n_target_candidates=8,  # Increased for 1-16 periods
                target_range=(0.002, 0.010),  # 0.2% to 1.0%
                bayesian_iterations=25  # Analyst optimization
            )

            # Tactician-specific optimization config (5m base timeframe, 1-16 periods)
            self.tactician_config = DynamicOptimizationConfig(
                optimization_method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
                min_horizon=1,   # 5 minutes (1 * 5m)
                max_horizon=16,  # 80 minutes (16 * 5m)
                horizon_step=1,  # Test every period from 1-16
                optimization_objective=OptimizationObjective.MULTI_OBJECTIVE,
                n_target_candidates=8,  # Increased for 1-16 periods
                target_range=(0.005, 0.015),  # 0.5% to 1.5%
                bayesian_iterations=30  # Tactician optimization
            )

            # Initialize optimizers
            self.analyst_optimizer = JointTargetHorizonOptimizer(self.analyst_config)
            self.tactician_optimizer = JointTargetHorizonOptimizer(self.tactician_config)

            # Initialize analysis components
            self.heuristic_analyzer = HeuristicAnalyzer()
            self.labeling_validator = LabelingValidator()

            self.logger.info('✅ Optimization components initialized successfully')

        except Exception as e:
            self.logger.error(f'❌ Failed to initialize optimization components: {e}')
            self.optimization_enabled = False

    def optimize_for_model(self,
                          model_type: ModelType,
                          market_data: pd.DataFrame,
                          force_optimization: bool = False) -> OptimizationResult:
        """
        Optimize timeframes for a specific model type.

        Args:
            model_type: Type of model to optimize for
            market_data: Market data for optimization
            force_optimization: Force optimization even if cached results exist

        Returns:
            OptimizationResult with optimal configuration
        """
        if not self.optimization_enabled:
            raise RuntimeError(
                f"❌ FAST FAIL: Optimization disabled for {model_type.value} model. "
                f"Cannot proceed without optimal timeframe discovery. "
                f"Training pipeline will terminate."
            )

        # Check for cached results
        if not force_optimization and model_type in self.optimization_results:
            self.logger.info(f'📋 Using cached optimization results for {model_type.value}')
            return self.optimization_results[model_type]

        self.logger.info(f'🎯 Starting optimization for {model_type.value} model')
        start_time = datetime.now()

        try:
            # Select appropriate optimizer
            if model_type == ModelType.ANALYST:
                optimizer = self.analyst_optimizer
                config = self.analyst_config
            elif model_type == ModelType.TACTICIAN:
                optimizer = self.tactician_optimizer
                config = self.tactician_config
            else:
                # For BOTH, use a combined approach
                return self._optimize_for_both_models(market_data)

            # Run optimization
            self.logger.info(f'   → Running {config.optimization_method.value} optimization...')
            optimization_result = optimizer.optimize_target_horizon_combinations(market_data)

            if optimization_result.objective_score < 0.3:
                self.logger.error(f'❌ FAST FAIL: Low optimization score ({optimization_result.objective_score:.3f}) for {model_type.value}')
                raise RuntimeError(
                    f"❌ FAST FAIL: Optimization failed for {model_type.value} model. "
                    f"Optimization score ({optimization_result.objective_score:.3f}) below minimum threshold (0.3). "
                    f"Cannot proceed without optimal timeframe discovery."
                )

            # Create optimized configuration
            optimized_config = self._create_optimized_config(
                optimization_result, model_type
            )

            # Validate configuration
            validation_score = self._validate_optimized_config(
                optimized_config, market_data, model_type
            )

            # Fast fail if validation score is too low
            if validation_score < 0.5:
                self.logger.error(f'❌ FAST FAIL: Low validation score ({validation_score:.3f}) for optimized configuration')
                raise RuntimeError(
                    f"❌ FAST FAIL: Optimized configuration validation failed for {model_type.value} model. "
                    f"Validation score ({validation_score:.3f}) below minimum threshold (0.5). "
                    f"Cannot proceed with invalid timeframe configuration."
                )

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                optimization_result, validation_score
            )

            # Create result
            result = OptimizationResult(
                model_type=model_type,
                optimal_config=optimized_config,
                optimization_score=optimization_result.objective_score,
                validation_score=validation_score,
                performance_metrics=performance_metrics,
                optimization_time=(datetime.now() - start_time).total_seconds()
            )

            # Store results
            self.optimization_results[model_type] = result
            self.optimization_history.append(result)

            self.logger.info(f'✅ Optimization completed for {model_type.value}')
            self.logger.info(f'   → Optimization score: {result.optimization_score:.3f}')
            self.logger.info(f'   → Validation score: {result.validation_score:.3f}')
            self.logger.info(f'   → Time horizons: {optimized_config.time_horizons}')
            self.logger.info(f'   → Profit targets: {optimized_config.profit_targets}')

            return result

        except Exception as e:
            self.logger.error(f'❌ FAST FAIL: Optimization failed for {model_type.value}: {e}')
            raise RuntimeError(
                f"❌ FAST FAIL: Optimization failed for {model_type.value} model. "
                f"Error: {e}. Cannot proceed without optimal timeframe discovery."
            )

    def _optimize_for_both_models(self, market_data: pd.DataFrame) -> OptimizationResult:
        """Optimize for both Analyst and Tactician models."""
        self.logger.info('🎯 Optimizing for both Analyst and Tactician models')

        # Optimize for each model type
        analyst_result = self.optimize_for_model(ModelType.ANALYST, market_data)
        tactician_result = self.optimize_for_model(ModelType.TACTICIAN, market_data)

        # Create combined configuration
        combined_config = MultiHorizonConfig()

        # Combine time horizons (use Analyst immediate, Tactician short)
        combined_config.time_horizons = {
            'immediate': analyst_result.optimal_config.time_horizons.get('immediate', 2),
            'short': tactician_result.optimal_config.time_horizons.get('short', 4)
        }

        # Combine profit targets
        analyst_targets = analyst_result.optimal_config.profit_targets
        tactician_targets = tactician_result.optimal_config.profit_targets

        combined_config.profit_targets = {
            'micro': analyst_targets.get('micro', 0.003),
            'small': analyst_targets.get('small', 0.005),
            'medium': tactician_targets.get('medium', 0.007),
            'good': tactician_targets.get('good', 0.010)
        }

        # Create combined result
        combined_result = OptimizationResult(
            model_type=ModelType.BOTH,
            optimal_config=combined_config,
            optimization_score=(analyst_result.optimization_score + tactician_result.optimization_score) / 2,
            validation_score=(analyst_result.validation_score + tactician_result.validation_score) / 2,
            performance_metrics={
                'analyst_score': analyst_result.optimization_score,
                'tactician_score': tactician_result.optimization_score,
                'combined_score': (analyst_result.optimization_score + tactician_result.optimization_score) / 2
            },
            optimization_time=analyst_result.optimization_time + tactician_result.optimization_time
        )

        self.optimization_results[ModelType.BOTH] = combined_result
        return combined_result

    def _create_optimized_config(self,
                                optimization_result: Any,
                                model_type: ModelType) -> MultiHorizonConfig:
        """Create optimized configuration from optimization results."""
        config = MultiHorizonConfig()

        # Map discovered horizons to configuration
        if hasattr(optimization_result, 'optimal_horizons') and optimization_result.optimal_horizons:
            horizon_values = list(optimization_result.optimal_horizons.values())
            if len(horizon_values) >= 2:
                if model_type == ModelType.ANALYST:
                    # Analyst needs quick response
                    config.time_horizons = {
                        'immediate': min(horizon_values[:2]),
                        'short': max(horizon_values[:2])
                    }
                else:  # Tactician
                    # Tactician needs more time for position management
                    config.time_horizons = {
                        'immediate': max(horizon_values[:2]),
                        'short': max(horizon_values[:2]) * 2
                    }
            else:
                # Fallback
                config.time_horizons = {
                    'immediate': horizon_values[0] if horizon_values else 2,
                    'short': horizon_values[0] * 2 if horizon_values else 4
                }

        # Map discovered targets to configuration
        if hasattr(optimization_result, 'optimal_targets') and optimization_result.optimal_targets:
            target_values = list(optimization_result.optimal_targets.values())
            if len(target_values) >= 4:
                sorted_targets = sorted(target_values)
                config.profit_targets = {
                    'micro': sorted_targets[0],
                    'small': sorted_targets[1],
                    'medium': sorted_targets[2],
                    'good': sorted_targets[3]
                }

        return config

    def _validate_optimized_config(self,
                                  config: MultiHorizonConfig,
                                  market_data: pd.DataFrame,
                                  model_type: ModelType) -> float:
        """Validate optimized configuration using heuristic analysis."""
        try:
            # Generate labels with optimized config
            from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonProfitLabeler
            labeler = MultiHorizonProfitLabeler(config)
            labeled_data = labeler.generate_labels(market_data.copy())

            # Analyze effectiveness
            heuristic_results = self.heuristic_analyzer.analyze_labeling_heuristics(labeled_data)

            # Calculate overall effectiveness score
            effectiveness_scores = []
            for result in heuristic_results.values():
                if hasattr(result, 'metric_value'):
                    effectiveness_scores.append(result.metric_value)

            if effectiveness_scores:
                avg_effectiveness = np.mean(effectiveness_scores)
                return min(1.0, max(0.0, avg_effectiveness))

            return 0.5  # Neutral score if no results

        except Exception as e:
            self.logger.warning(f'⚠️ Configuration validation failed: {e}')
            return 0.3  # Low score on error

    def _calculate_performance_metrics(self,
                                     optimization_result: Any,
                                     validation_score: float) -> Dict[str, float]:
        """Calculate performance metrics for the optimization result."""
        metrics = {
            'optimization_score': getattr(optimization_result, 'objective_score', 0.0),
            'validation_score': validation_score,
            'overall_score': (getattr(optimization_result, 'objective_score', 0.0) + validation_score) / 2
        }

        # Add performance metrics from optimization result if available
        if hasattr(optimization_result, 'performance_metrics'):
            metrics.update(optimization_result.performance_metrics)

        return metrics

    def _create_fallback_result(self, model_type: ModelType) -> OptimizationResult:
        """Fast fail when optimization is not available - no fallback allowed."""
        raise RuntimeError(
            f"❌ FAST FAIL: Cannot find optimal periods for {model_type.value} model. "
            f"Optimization components not available or failed to initialize. "
            f"Training cannot proceed without optimal timeframe discovery."
        )

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization results."""
        summary = {
            'optimization_enabled': self.optimization_enabled,
            'results_count': len(self.optimization_results),
            'history_count': len(self.optimization_history),
            'results': {}
        }

        for model_type, result in self.optimization_results.items():
            summary['results'][model_type.value] = result.to_dict()

        return summary

    def save_optimization_results(self, output_dir: str = "optimization_results"):
        """Save optimization results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = output_path / 'optimization_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(self.get_optimization_summary(), f, indent=2)

        # Save individual results
        for model_type, result in self.optimization_results.items():
            result_path = output_path / f'{model_type.value}_optimization_result.json'
            with open(result_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

        self.logger.info(f'💾 Optimization results saved to {output_path}')

# Convenience functions for integration
def optimize_timeframes_for_training(market_data: pd.DataFrame,
                                   model_type: str = "both",
                                   force_optimization: bool = False) -> Dict[str, Any]:
    """
    Convenience function to optimize timeframes for training.

    Args:
        market_data: Market data for optimization
        model_type: Type of model ("analyst", "tactician", or "both")
        force_optimization: Force optimization even if cached results exist

    Returns:
        Dictionary with optimization results
    """
    optimizer = AutomaticTimeframeOptimizer()

    # Convert string to enum
    model_enum = ModelType.BOTH
    if model_type.lower() == "analyst":
        model_enum = ModelType.ANALYST
    elif model_type.lower() == "tactician":
        model_enum = ModelType.TACTICIAN

    # Run optimization
    result = optimizer.optimize_for_model(model_enum, market_data, force_optimization)

    return {
        'model_type': result.model_type.value,
        'optimal_config': result.optimal_config,
        'optimization_score': result.optimization_score,
        'validation_score': result.validation_score,
        'performance_metrics': result.performance_metrics,
        'optimization_time': result.optimization_time
    }

def get_optimal_timeframes_for_models(market_data: pd.DataFrame) -> Dict[str, MultiHorizonConfig]:
    """
    Get optimal timeframes for both Analyst and Tactician models.

    Args:
        market_data: Market data for optimization

    Returns:
        Dictionary with optimal configurations for each model type
    """
    optimizer = AutomaticTimeframeOptimizer()

    # Optimize for both models
    analyst_result = optimizer.optimize_for_model(ModelType.ANALYST, market_data)
    tactician_result = optimizer.optimize_for_model(ModelType.TACTICIAN, market_data)

    return {
        'analyst': analyst_result.optimal_config,
        'tactician': tactician_result.optimal_config
    }
