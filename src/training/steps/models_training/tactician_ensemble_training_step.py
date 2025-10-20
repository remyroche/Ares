"""
Tactician Ensemble Training Step - BaseStep Wrapper

This module provides a BaseStep wrapper for Tactician ensemble training
to enable integration with ares_launcher.py.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.training.steps.base_step import BaseStep
from .core.ensemble_trainer import EnsembleTrainer
from .core.base_trainer import TrainingConfig, TrainingRole, ModelType


class TacticianEnsembleTrainingStep(BaseStep):
    """
    BaseStep wrapper for Tactician Ensemble Training.
    
    This step trains Tactician ensemble models that combine multiple base models
    for enhanced entry/exit timing predictions.
    """
    
    def __init__(self, step_name: str = "tactician_ensemble_training", config: Optional[Dict[str, Any]] = None):
        """Initialize the tactician ensemble training step."""
        super().__init__(step_name, config)
        self.logger = logging.getLogger(f"ares.step.{step_name}")
        
        # Initialize the ensemble trainer
        training_config = TrainingConfig(
            role=TrainingRole.TACTICIAN,
            model_types=[ModelType.LIGHTGBM, ModelType.CATBOOST, ModelType.NEURAL_NETWORK],
            timeframe=config.get('timeframe', '15m') if config else '15m',
            symbol=config.get('symbol', 'ETHUSDT') if config else 'ETHUSDT',
            enable_ensemble=True,
            custom_params={
                'ensemble_strategy': 'stacking',
                'meta_learner_type': 'lightgbm',
                'cv_folds': 5,
                **(config or {})
            }
        )
        
        self.trainer = EnsembleTrainer(training_config, self.logger)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tactician ensemble training step.
        
        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - direction: Trading direction ('longs', 'shorts', 'both')
                - execution_mode: Execution mode ('full', 'light', 'blank')
        
        Returns:
            Execution result dictionary
        """
        try:
            self.logger.info("🚀 Starting Tactician Ensemble Training Step")
            
            # Set context for artifact management
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                information='training',
                direction=config.get('direction', 'longs'),
                model='Tactician'
            )
            
            # Load training data
            training_data = self._load_dataframe('training_data')
            if training_data is None:
                training_data = self._load_dataframe('market_data')
                if training_data is None:
                    training_data = self._load_dataframe('processed_data')
            
            if training_data is None:
                return {
                    'success': False,
                    'error': 'No training data found. Please ensure data is available in artifacts.',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Load targets if available
            targets = self._load_dataframe('tactician_targets')
            if targets is None:
                target_columns = ['target', 'y', 'label', 'tactician_target', 'entry_target', 'exit_target']
                for col in target_columns:
                    if col in training_data.columns:
                        targets = training_data[col]
                        training_data = training_data.drop(columns=[col])
                        break
                
                if targets is None:
                    return {
                        'success': False,
                        'error': 'No target data found for tactician ensemble training',
                        'artifacts': [],
                        'metrics': {}
                    }
            
            # Load analyst predictions if available (for enhanced features)
            analyst_predictions = self._load_dataframe('analyst_predictions')
            if analyst_predictions is not None:
                # Add analyst predictions as features
                for col in analyst_predictions.columns:
                    training_data[f'analyst_{col}'] = analyst_predictions[col]
                self.logger.info(f"Enhanced training data with {len(analyst_predictions.columns)} analyst features")
            
            # Load tactician base model outputs if available
            base_model_outputs = self._load_dataframe('tactician_base_predictions')
            if base_model_outputs is not None:
                # Add base model outputs as features
                for col in base_model_outputs.columns:
                    training_data[f'base_{col}'] = base_model_outputs[col]
                self.logger.info(f"Enhanced training data with {len(base_model_outputs.columns)} base model features")
            
            # Initialize trainer
            if not await self.trainer.initialize():
                return {
                    'success': False,
                    'error': 'Failed to initialize tactician ensemble trainer',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Train ensemble
            result = await self.trainer.train(training_data, targets)
            
            if result.success:
                # Save ensemble model
                if result.model:
                    self._save_model(result.model, 'tactician_ensemble_model')
                
                # Save individual model results
                if hasattr(result, 'metadata') and 'individual_results' in result.metadata:
                    individual_results = result.metadata['individual_results']
                    for model_name, model_result in individual_results.items():
                        if model_result.success and model_result.model:
                            self._save_model(model_result.model, f'tactician_ensemble_{model_name}_model')
                
                # Save metrics
                if result.metrics:
                    self._save_metadata(result.metrics, 'tactician_ensemble_metrics')
                
                # Save feature importance
                if result.feature_importance:
                    self._save_metadata(result.feature_importance, 'tactician_ensemble_feature_importance')
                
                # Save SHAP explanations if available
                if hasattr(result, 'shap_explanations') and result.shap_explanations:
                    self._save_metadata(result.shap_explanations, 'tactician_ensemble_shap_explanations')
                
                # Save training summary
                training_summary = self.trainer.get_training_summary()
                self._save_metadata(training_summary, 'tactician_ensemble_summary')
                
                self.logger.info("✅ Tactician Ensemble Training completed successfully")
                
                return {
                    'success': True,
                    'artifacts': [
                        'tactician_ensemble_model',
                        'tactician_ensemble_metrics',
                        'tactician_ensemble_feature_importance',
                        'tactician_ensemble_shap_explanations',
                        'tactician_ensemble_summary'
                    ],
                    'metrics': result.metrics,
                    'ensemble_strategy': result.metadata.get('ensemble_strategy', 'unknown') if hasattr(result, 'metadata') else 'unknown',
                    'base_models_count': result.metadata.get('individual_models', 0) if hasattr(result, 'metadata') else 0,
                    'training_time': result.training_time
                }
            else:
                return {
                    'success': False,
                    'error': f"Tactician ensemble training failed: {result.error_message}",
                    'artifacts': [],
                    'metrics': {}
                }
                
        except Exception as e:
            self.logger.error(f"❌ Tactician Ensemble Training Step failed: {e}")
            return {
                'success': False,
                'error': f"Step execution failed: {str(e)}",
                'artifacts': [],
                'metrics': {}
            }