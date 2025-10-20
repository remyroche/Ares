"""
Tactician Base Training Step - BaseStep Wrapper

This module provides a BaseStep wrapper for Tactician base models training
to enable integration with ares_launcher.py.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.training.steps.base_step import BaseStep
from .core.model_trainer import ModelTrainer
from .core.base_trainer import TrainingConfig, TrainingRole, ModelType


class TacticianBaseTrainingStep(BaseStep):
    """
    BaseStep wrapper for Tactician Base Models Training.
    
    This step trains individual Tactician base models for precise entry/exit timing.
    """
    
    def __init__(self, step_name: str = "tactician_base_training", config: Optional[Dict[str, Any]] = None):
        """Initialize the tactician base training step."""
        super().__init__(step_name, config)
        self.logger = logging.getLogger(f"ares.step.{step_name}")
        
        # Initialize the model trainer
        training_config = TrainingConfig(
            role=TrainingRole.TACTICIAN,
            model_types=[ModelType.LIGHTGBM, ModelType.CATBOOST, ModelType.NEURAL_NETWORK],
            timeframe=config.get('timeframe', '15m') if config else '15m',
            symbol=config.get('symbol', 'ETHUSDT') if config else 'ETHUSDT',
            enable_ensemble=False,  # Individual models only
            custom_params=config or {}
        )
        
        self.trainer = ModelTrainer(training_config, self.logger)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tactician base training step.
        
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
            self.logger.info("🚀 Starting Tactician Base Training Step")
            
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
                        'error': 'No target data found for tactician training',
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
            
            # Initialize trainer
            if not await self.trainer.initialize():
                return {
                    'success': False,
                    'error': 'Failed to initialize tactician trainer',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Train models
            result = await self.trainer.train(training_data, targets)
            
            if result.success:
                # Save trained models
                if result.model:
                    self._save_model(result.model, 'tactician_base_models')
                
                # Save individual model results
                if hasattr(result, 'metadata') and 'individual_results' in result.metadata:
                    individual_results = result.metadata['individual_results']
                    for model_name, model_result in individual_results.items():
                        if model_result.success and model_result.model:
                            self._save_model(model_result.model, f'tactician_{model_name}_model')
                
                # Save metrics
                if result.metrics:
                    self._save_metadata(result.metrics, 'tactician_training_metrics')
                
                # Save feature importance
                if result.feature_importance:
                    self._save_metadata(result.feature_importance, 'tactician_feature_importance')
                
                # Save training summary
                training_summary = self.trainer.get_training_summary()
                self._save_metadata(training_summary, 'tactician_training_summary')
                
                self.logger.info("✅ Tactician Base Training completed successfully")
                
                return {
                    'success': True,
                    'artifacts': [
                        'tactician_base_models',
                        'tactician_training_metrics',
                        'tactician_feature_importance',
                        'tactician_training_summary'
                    ],
                    'metrics': result.metrics,
                    'models_trained': result.metadata.get('models_trained', 0) if hasattr(result, 'metadata') else 0,
                    'training_time': result.training_time
                }
            else:
                return {
                    'success': False,
                    'error': f"Tactician training failed: {result.error_message}",
                    'artifacts': [],
                    'metrics': {}
                }
                
        except Exception as e:
            self.logger.error(f"❌ Tactician Base Training Step failed: {e}")
            return {
                'success': False,
                'error': f"Step execution failed: {str(e)}",
                'artifacts': [],
                'metrics': {}
            }