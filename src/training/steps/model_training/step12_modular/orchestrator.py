"""
Step 12 Modular: Main Orchestrator

This module contains the main orchestrator for Step 12 analyst enhancement.
"""

import asyncio
from typing import Dict, Any, List

from .base.imports import (
    log_important_calls, log_all_calls, handles_errors,
    system_logger, pipeline_standards, dependency_status
)
from .base.utils import error, failed, timeout, warning
from .config.step12_config import Step12Config, DEFAULT_CONFIG
from .models.enhancement_model import RegimeAwareAnalystEnhancementModel
from .hpo.optimizer import HyperparameterOptimizer
from .feature_selection.selector import FeatureSelector

class RegimeAwareAnalystEnhancementOrchestrator:
    """Main orchestrator for Step 12 Analyst Enhancement."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the orchestrator.

        Args:
            config: Configuration dictionary for the step.
        """
        self.config = config
        self.step12_config = Step12Config(**config)
        self.logger = system_logger

        # Initialize components
        self.enhancement_model = RegimeAwareAnalystEnhancementModel(config)
        self.hpo_optimizer = HyperparameterOptimizer(config)
        self.feature_selector = FeatureSelector(
            config,
            self.step12_config.metadata_columns,
            self.step12_config.label_columns
        )

        self.logger.info('Step 12 Orchestrator initialized')

    @handles_errors(Exception, default_return={'status': 'FAILED', 'error': 'Execution failed'})
    async def execute(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the full Step 12 analyst enhancement pipeline.

        Args:
            training_input: Input parameters including symbol, exchange, and data directories.
            pipeline_state: The current state of the pipeline.

        Returns:
            Dictionary containing the results of the enhancement process.
        """
        self.logger.info('🚀 Starting Step 12: Regime-Aware Analyst Enhancement')
        self.logger.info('🔄 Executing full enhancement pipeline...')

        try:
            # Initialize the enhancement model
            init_success = await self.enhancement_model.initialize()
            if not init_success:
                return {'status': 'FAILED', 'error': 'Failed to initialize enhancement model'}

            # Execute the enhancement
            result = await self.enhancement_model.execute_enhancement(
                training_input, pipeline_state
            )

            # Add orchestrator metadata
            result['orchestrator_info'] = {
                'step12_config': self.step12_config.to_dict(),
                'components_initialized': [
                    'enhancement_model',
                    'hpo_optimizer',
                    'feature_selector'
                ]
            }

            self.logger.info('✅ Step 12 enhancement completed successfully')
            return result

        except Exception as e:
            self.logger.error(failed(f'Step 12 enhancement failed: {e}'))
            return {'status': 'FAILED', 'error': str(e)}

    async def enhance_single_model(
        self,
        model_data: Dict[str, Any],
        model_name: str,
        X_train,
        y_train,
        X_val,
        y_val,
        timeframe_name: str = 'default'
    ) -> Dict[str, Any]:
        """Enhance a single model using the modular components.

        Args:
            model_data: Model data and configuration.
            model_name: Name of the model to enhance.
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            timeframe_name: Name of the timeframe.

        Returns:
            Dictionary containing enhancement results.
        """
        self.logger.info(f'🔄 Enhancing {model_name} for {timeframe_name}...')

        try:
            # Step 1: Feature Selection
            selected_features, feature_summary = await self.feature_selector.select_optimal_features(
                model_data.get('model'), model_name, X_train, y_train, X_val, y_val
            )

            # Step 2: Hyperparameter Optimization
            best_params, hpo_score = await self.hpo_optimizer.optimize_model(
                model_name, X_train[selected_features], y_train, X_val[selected_features], y_val
            )

            # Return enhancement results
            return {
                'model': model_data.get('model'),
                'selected_features': selected_features,
                'best_params': best_params,
                'hpo_score': hpo_score,
                'feature_summary': feature_summary,
                'enhancement_metadata': {
                    'model_name': model_name,
                    'timeframe': timeframe_name,
                    'selected_feature_count': len(selected_features),
                    'enhancement_applied': True
                }
            }

        except Exception as e:
            self.logger.error(error(f'Failed to enhance {model_name}: {e}'))
            return {
                'model': model_data.get('model'),
                'selected_features': list(X_train.columns),
                'best_params': {},
                'hpo_score': 0.0,
                'error': str(e),
                'enhancement_metadata': {
                    'model_name': model_name,
                    'timeframe': timeframe_name,
                    'enhancement_applied': False
                }
            }

__all__ = ['RegimeAwareAnalystEnhancementOrchestrator']
