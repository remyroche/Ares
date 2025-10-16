"""
Enhanced Multi-Horizon Pipeline with Automatic Timeframe Optimization

This module provides an enhanced version of the multi-horizon pipeline that
automatically optimizes timeframes for Analyst and Tactician model training.

Key Features:
- Automatic timeframe optimization for each model type
- Integration with existing training pipeline
- Model-specific configuration optimization
- Performance monitoring and validation
- Seamless fallback to default configurations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime
from pathlib import Path

from src.utils.logger import get_logger
from src.training.steps.market_analysis.multi_horizon_sub_pipeline_adapter import (
    MultiHorizonSubPipelineAdapter,
    execute_multi_horizon_labeling_step
)
from src.training.steps.market_analysis.automatic_timeframe_optimizer import (
    AutomaticTimeframeOptimizer,
    ModelType,
    optimize_timeframes_for_training,
    get_optimal_timeframes_for_models
)

@dataclass
class EnhancedPipelineConfig:
    """Configuration for enhanced multi-horizon pipeline."""
    # Optimization settings
    enable_automatic_optimization: bool = True
    optimize_for_analyst: bool = True
    optimize_for_tactician: bool = True
    force_optimization: bool = False

    # Model-specific settings
    analyst_optimization_priority: str = "speed"  # speed, accuracy, balance
    tactician_optimization_priority: str = "risk_management"  # risk_management, profit, balance

    # Performance settings
    enable_performance_monitoring: bool = True
    save_optimization_results: bool = True
    optimization_output_dir: str = "optimization_results"

    # Fast fail settings (no fallbacks allowed)
    fast_fail_on_optimization_failure: bool = True
    log_optimization_details: bool = True

class EnhancedMultiHorizonPipeline:
    """
    Enhanced multi-horizon pipeline with automatic timeframe optimization.

    This class extends the standard multi-horizon pipeline with automatic
    optimization capabilities for Analyst and Tactician model training.
    """

    def __init__(self, config: Optional[EnhancedPipelineConfig] = None):
        """Initialize enhanced multi-horizon pipeline."""
        self.config = config or EnhancedPipelineConfig()
        self.logger = get_logger('EnhancedMultiHorizonPipeline')

        # Initialize base adapter
        self.base_adapter = MultiHorizonSubPipelineAdapter()

        # Initialize optimizer
        self.optimizer = AutomaticTimeframeOptimizer()

        # Optimization results cache
        self.optimization_cache: Dict[str, Any] = {}

        self.logger.info('🚀 Enhanced Multi-Horizon Pipeline initialized')
        self.logger.info(f'   → Automatic optimization: {"ENABLED" if self.config.enable_automatic_optimization else "DISABLED"}')
        self.logger.info(f'   → Analyst optimization: {"ENABLED" if self.config.optimize_for_analyst else "DISABLED"}')
        self.logger.info(f'   → Tactician optimization: {"ENABLED" if self.config.optimize_for_tactician else "DISABLED"}')

    def execute_enhanced_labeling_step(self,
                                     data: pd.DataFrame,
                                     regime_labels: Optional[pd.Series] = None,
                                     config: Optional[Dict[str, Any]] = None,
                                     symbol: Optional[str] = None,
                                     exchange: Optional[str] = None,
                                     timeframe: Optional[str] = None,
                                     mode: str = 'full',
                                     features: Optional[Dict[str, Any]] = None,
                                     model_type: str = "both") -> Dict[str, Any]:
        """
        Execute enhanced multi-horizon labeling with automatic optimization.

        Args:
            data: Input OHLCV data
            regime_labels: Optional regime labels
            config: Configuration dictionary
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            mode: Execution mode
            features: Optional pre-computed features
            model_type: Type of model ("analyst", "tactician", or "both")

        Returns:
            Dictionary with enhanced labeling results
        """
        self.logger.info(f'🎯 Executing enhanced labeling for {model_type} model(s)')

        # Step 1: Automatic timeframe optimization
        optimized_configs = {}
        if self.config.enable_automatic_optimization:
            optimized_configs = self._run_automatic_optimization(
                data, model_type, symbol, exchange
            )

        # Step 2: Execute labeling with optimized configurations
        results = {}

        if model_type.lower() in ["analyst", "both"] and self.config.optimize_for_analyst:
            analyst_config = optimized_configs.get('analyst', {})
            analyst_result = self._execute_model_specific_labeling(
                data, regime_labels, config, symbol, exchange, timeframe, mode, features,
                model_type="analyst", optimized_config=analyst_config
            )
            results['analyst'] = analyst_result

        if model_type.lower() in ["tactician", "both"] and self.config.optimize_for_tactician:
            tactician_config = optimized_configs.get('tactician', {})
            tactician_result = self._execute_model_specific_labeling(
                data, regime_labels, config, symbol, exchange, timeframe, mode, features,
                model_type="tactician", optimized_config=tactician_config
            )
            results['tactician'] = tactician_result

        # Step 3: Combine results if both models were processed
        if len(results) > 1:
            combined_result = self._combine_model_results(results)
            results['combined'] = combined_result

        # Step 4: Save optimization results if enabled
        if self.config.save_optimization_results and optimized_configs:
            self._save_optimization_results(optimized_configs, symbol, exchange)

        return results

    def _run_automatic_optimization(self,
                                   data: pd.DataFrame,
                                   model_type: str,
                                   symbol: Optional[str] = None,
                                   exchange: Optional[str] = None) -> Dict[str, Any]:
        """Run automatic timeframe optimization for specified model types."""
        self.logger.info('🎯 Running automatic timeframe optimization')

        optimized_configs = {}

        try:
            if model_type.lower() in ["analyst", "both"] and self.config.optimize_for_analyst:
                self.logger.info('   → Optimizing for Analyst model...')
                analyst_result = self.optimizer.optimize_for_model(
                    ModelType.ANALYST, data, self.config.force_optimization
                )
                optimized_configs['analyst'] = {
                    'config': analyst_result.optimal_config,
                    'score': analyst_result.optimization_score,
                    'validation_score': analyst_result.validation_score,
                    'performance_metrics': analyst_result.performance_metrics
                }
                self.logger.info(f'   ✅ Analyst optimization completed (score: {analyst_result.optimization_score:.3f})')

            if model_type.lower() in ["tactician", "both"] and self.config.optimize_for_tactician:
                self.logger.info('   → Optimizing for Tactician model...')
                tactician_result = self.optimizer.optimize_for_model(
                    ModelType.TACTICIAN, data, self.config.force_optimization
                )
                optimized_configs['tactician'] = {
                    'config': tactician_result.optimal_config,
                    'score': tactician_result.optimization_score,
                    'validation_score': tactician_result.validation_score,
                    'performance_metrics': tactician_result.performance_metrics
                }
                self.logger.info(f'   ✅ Tactician optimization completed (score: {tactician_result.optimization_score:.3f})')

            return optimized_configs

        except Exception as e:
            self.logger.error(f'❌ FAST FAIL: Automatic optimization failed: {e}')
            raise RuntimeError(
                f"❌ FAST FAIL: Automatic timeframe optimization failed. "
                f"Error: {e}. Cannot proceed without optimal timeframe discovery. "
                f"Training pipeline will terminate."
            )

    def _execute_model_specific_labeling(self,
                                       data: pd.DataFrame,
                                       regime_labels: Optional[pd.Series] = None,
                                       config: Optional[Dict[str, Any]] = None,
                                       symbol: Optional[str] = None,
                                       exchange: Optional[str] = None,
                                       timeframe: Optional[str] = None,
                                       mode: str = 'full',
                                       features: Optional[Dict[str, Any]] = None,
                                       model_type: str = "analyst",
                                       optimized_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute labeling for a specific model type with optimized configuration."""

        # Prepare configuration with optimization
        enhanced_config = config.copy() if config else {}

        if optimized_config and 'config' in optimized_config:
            opt_config = optimized_config['config']
            enhanced_config.update({
                'time_horizons': opt_config.time_horizons,
                'profit_targets': opt_config.profit_targets,
                'transaction_cost': opt_config.transaction_cost,
                'optimization_score': optimized_config.get('score', 0.0),
                'validation_score': optimized_config.get('validation_score', 0.0)
            })

            self.logger.info(f'🔧 Using optimized configuration for {model_type}:')
            self.logger.info(f'   → Time horizons: {opt_config.time_horizons}')
            self.logger.info(f'   → Profit targets: {opt_config.profit_targets}')

        # Execute labeling with enhanced configuration
        result = self.base_adapter.execute_multi_horizon_labeling_step(
            data=data,
            regime_labels=regime_labels,
            config=enhanced_config,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            mode=mode,
            features=features
        )

        # Add optimization metadata
        if optimized_config:
            result['optimization_metadata'] = {
                'model_type': model_type,
                'optimization_score': optimized_config.get('score', 0.0),
                'validation_score': optimized_config.get('validation_score', 0.0),
                'performance_metrics': optimized_config.get('performance_metrics', {}),
                'optimization_timestamp': datetime.now().isoformat()
            }

        return result

    def _combine_model_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from multiple model types."""
        self.logger.info('🔄 Combining results from multiple model types')

        combined_result = {
            'status': 'success',
            'model_types': list(results.keys()),
            'combined_artifacts': {},
            'optimization_summary': {}
        }

        # Combine artifacts
        for model_type, result in results.items():
            if 'artifacts' in result:
                for artifact_name, artifact_data in result['artifacts'].items():
                    combined_artifact_name = f"{model_type}_{artifact_name}"
                    combined_result['combined_artifacts'][combined_artifact_name] = artifact_data

        # Combine optimization metadata
        for model_type, result in results.items():
            if 'optimization_metadata' in result:
                combined_result['optimization_summary'][model_type] = result['optimization_metadata']

        return combined_result

    def _get_fallback_configurations(self, model_type: str) -> Dict[str, Any]:
        """Fast fail - no fallback configurations allowed."""
        raise RuntimeError(
            f"❌ FAST FAIL: Fallback configurations not allowed. "
            f"Optimization must succeed for {model_type} model. "
            f"Training cannot proceed without optimal timeframe discovery."
        )

    def _save_optimization_results(self,
                                 optimized_configs: Dict[str, Any],
                                 symbol: Optional[str] = None,
                                 exchange: Optional[str] = None):
        """Save optimization results to disk."""
        try:
            output_dir = Path(self.config.optimization_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with symbol and exchange info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol_str = symbol or "unknown"
            exchange_str = exchange or "unknown"
            filename = f"optimization_results_{symbol_str}_{exchange_str}_{timestamp}.json"

            # Save results
            import json
            results_path = output_dir / filename
            with open(results_path, 'w') as f:
                json.dump(optimized_configs, f, indent=2, default=str)

            self.logger.info(f'💾 Optimization results saved to {results_path}')

        except Exception as e:
            self.logger.warning(f'⚠️ Failed to save optimization results: {e}')

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        return self.optimizer.get_optimization_summary()

    def save_all_optimization_results(self):
        """Save all optimization results."""
        self.optimizer.save_optimization_results(self.config.optimization_output_dir)

# Enhanced convenience functions
def execute_enhanced_multi_horizon_labeling(data: pd.DataFrame,
                                           regime_labels: Optional[pd.Series] = None,
                                           config: Optional[Dict[str, Any]] = None,
                                           symbol: Optional[str] = None,
                                           exchange: Optional[str] = None,
                                           timeframe: Optional[str] = None,
                                           mode: str = 'full',
                                           features: Optional[Dict[str, Any]] = None,
                                           model_type: str = "both",
                                           enable_optimization: bool = True) -> Dict[str, Any]:
    """
    Execute enhanced multi-horizon labeling with automatic optimization.

    This is the main entry point for enhanced multi-horizon labeling with
    automatic timeframe optimization for Analyst and Tactician models.
    """
    pipeline_config = EnhancedPipelineConfig(
        enable_automatic_optimization=enable_optimization
    )

    pipeline = EnhancedMultiHorizonPipeline(pipeline_config)

    return pipeline.execute_enhanced_labeling_step(
        data=data,
        regime_labels=regime_labels,
        config=config,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        mode=mode,
        features=features,
        model_type=model_type
    )

def get_optimal_configurations_for_training(market_data: pd.DataFrame,
                                          model_types: List[str] = ["analyst", "tactician"]) -> Dict[str, Any]:
    """
    Get optimal configurations for training specific model types.

    Args:
        market_data: Market data for optimization
        model_types: List of model types to optimize for

    Returns:
        Dictionary with optimal configurations for each model type
    """
    optimizer = AutomaticTimeframeOptimizer()
    results = {}

    for model_type in model_types:
        if model_type.lower() == "analyst":
            result = optimizer.optimize_for_model(ModelType.ANALYST, market_data)
            results['analyst'] = result.optimal_config
        elif model_type.lower() == "tactician":
            result = optimizer.optimize_for_model(ModelType.TACTICIAN, market_data)
            results['tactician'] = result.optimal_config

    return results
