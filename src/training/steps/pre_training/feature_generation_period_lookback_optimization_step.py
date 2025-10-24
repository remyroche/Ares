"""
Feature Generation Period Lookback Optimization Step

This step optimizes the lookback period for feature generation based on execution mode
and performance requirements.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Import base step
from src.training.steps.base_step import BaseStep

# Import step registry
from src.training.steps.base_step import step_registry

# Import pipeline mode configuration
from src.config.pipeline_modes import get_mode_config, get_mode_lookback_days

# Import utilities
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

logger = logging.getLogger(__name__)


class FeatureGenerationPeriodLookbackOptimizationStep(BaseStep):
    """
    Step for optimizing feature generation lookback periods based on execution mode.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the step."""
        super().__init__(config)
        self.step_name = "feature_generation_period_lookback_optimization"
        
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the lookback period optimization step.
        
        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - direction: Trading direction ('longs', 'shorts', 'both')
                - execution_mode: Execution mode ('full', 'light', 'blank')
        
        Returns:
            Dictionary containing execution results
        """
        try:
            tprint_info("🚀 Starting Feature Generation Period Lookback Optimization")
            
            # Get execution mode and validate
            execution_mode = config.get('execution_mode', 'light')
            if execution_mode not in ['full', 'light', 'blank']:
                tprint_warning(f"Invalid execution mode '{execution_mode}', defaulting to 'light'")
                execution_mode = 'light'
            
            # Get mode configuration
            mode_config = get_mode_config(execution_mode)
            lookback_days = mode_config.lookback_days
            
            tprint_info(f"📊 Execution Mode: {execution_mode.upper()}")
            tprint_info(f"📅 Optimized Lookback Period: {lookback_days} days")
            tprint_info(f"⚡ Computational Intensity: {mode_config.computational_intensity}")
            
            # Set context
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                information=config.get('information'),
                direction=config.get('direction', 'longs'),
                model=config.get('model', 'LookbackOptimization'),
                execution_mode=execution_mode
            )
            
            # Create optimization results
            optimization_results = {
                'execution_mode': execution_mode,
                'lookback_days': lookback_days,
                'lookback_years': mode_config.lookback_years,
                'intensity_percentage': mode_config.intensity_percentage,
                'computational_intensity': mode_config.computational_intensity,
                'estimated_duration_minutes': mode_config.estimated_duration_minutes,
                'max_trials': mode_config.max_trials,
                'n_trials': mode_config.n_trials,
                'optuna_trials': mode_config.optuna_trials,
                'optuna_timeout': mode_config.optuna_timeout,
                'batch_size': mode_config.batch_size,
                'epochs': mode_config.epochs,
                'early_stopping_patience': mode_config.early_stopping_patience,
                'cross_validation_folds': mode_config.cross_validation_folds,
                'enable_parallelization': mode_config.enable_parallelization,
                'enable_caching': mode_config.enable_caching,
                'enable_advanced_features': mode_config.enable_advanced_features,
                'enable_ensemble_training': mode_config.enable_ensemble_training,
                'enable_multi_timeframe_training': mode_config.enable_multi_timeframe_training,
                'enable_adaptive_training': mode_config.enable_adaptive_training
            }
            
            # Save optimization results as artifact
            self._save_dataframe(
                pd.DataFrame([optimization_results]), 
                'lookback_optimization_results',
                metadata={
                    'execution_mode': execution_mode,
                    'lookback_days': lookback_days,
                    'description': f'Lookback optimization results for {execution_mode} execution mode'
                }
            )
            
            # Create summary statistics
            summary_stats = {
                'total_parameters_optimized': len(optimization_results),
                'lookback_period_days': lookback_days,
                'execution_mode': execution_mode,
                'optimization_timestamp': pd.Timestamp.now().isoformat(),
                'mode_configuration': {
                    'name': mode_config.name,
                    'description': mode_config.description,
                    'computational_intensity': mode_config.computational_intensity,
                    'estimated_duration_minutes': mode_config.estimated_duration_minutes
                }
            }
            
            # Save summary statistics
            self._save_dataframe(
                pd.DataFrame([summary_stats]), 
                'lookback_optimization_summary',
                metadata={
                    'execution_mode': execution_mode,
                    'description': f'Summary statistics for lookback optimization in {execution_mode} mode'
                }
            )
            
            tprint_success(f"✅ Lookback optimization completed for {execution_mode} mode")
            tprint_info(f"📊 Optimized lookback period: {lookback_days} days")
            tprint_info(f"⚙️ Computational intensity: {mode_config.computational_intensity}")
            
            return {
                'success': True,
                'artifacts': ['lookback_optimization_results', 'lookback_optimization_summary'],
                'metrics': {
                    'lookback_days': lookback_days,
                    'execution_mode': execution_mode,
                    'computational_intensity': mode_config.computational_intensity,
                    'estimated_duration_minutes': mode_config.estimated_duration_minutes
                },
                'optimization_results': optimization_results,
                'summary_stats': summary_stats
            }
            
        except Exception as e:
            error_msg = f"Lookback optimization failed: {str(e)}"
            tprint_error(error_msg)
            logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'error': error_msg,
                'artifacts': [],
                'metrics': {}
            }


# Register the step
step_registry.register("feature_generation_period_lookback_optimization", FeatureGenerationPeriodLookbackOptimizationStep)