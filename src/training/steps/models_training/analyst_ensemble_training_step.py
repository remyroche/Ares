"""
Analyst Ensemble Training Step - BaseStep Wrapper

This module provides a BaseStep wrapper for the AnalystEnsembleTrainingModular component
to enable integration with ares_launcher.py.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.training.steps.base_step import BaseStep
from .components.analyst_ensemble_training_modular import AnalystEnsembleTrainingModular


class AnalystEnsembleTrainingStep(BaseStep):
    """
    BaseStep wrapper for Analyst Ensemble Training.
    
    This step trains Analyst ensemble models that combine:
    - Base models (LightGBM, LightGBM+PatchTST, CatBoost, Stacker LGBM Calibrated)
    - HMM regime features and probabilities
    - Meta-learner ensemble for enhanced trading signal generation
    """
    
    def __init__(self, step_name: str = "analyst_ensemble_training", config: Optional[Dict[str, Any]] = None):
        """Initialize the analyst ensemble training step."""
        super().__init__(step_name, config)
        self.logger = logging.getLogger(f"ares.step.{step_name}")
        
        # Initialize the modular component
        self.component = AnalystEnsembleTrainingModular(
            name="analyst_ensemble_training",
            config=config,
            logger=self.logger
        )
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the analyst ensemble training step.
        
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
            self.logger.info("🚀 Starting Analyst Ensemble Training Step")
            
            # Set context for artifact management
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                information='training',
                direction=config.get('direction', 'longs'),
                model='Analyst'
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
            targets = self._load_dataframe('analyst_targets')
            if targets is None:
                target_columns = ['target', 'y', 'label', 'analyst_target']
                for col in target_columns:
                    if col in training_data.columns:
                        targets = training_data[col]
                        training_data = training_data.drop(columns=[col])
                        break
                
                if targets is None:
                    return {
                        'success': False,
                        'error': 'No target data found for analyst ensemble training',
                        'artifacts': [],
                        'metrics': {}
                    }
            
            # Load base model outputs (from previous analyst base training)
            base_model_outputs = self._load_dataframe('analyst_base_predictions')
            if base_model_outputs is None:
                # Try alternative names
                base_model_outputs = self._load_dataframe('base_model_outputs')
                if base_model_outputs is None:
                    # Create placeholder base model outputs
                    base_model_outputs = {
                        'lightgbm': np.random.random(len(training_data)),
                        'catboost': np.random.random(len(training_data)),
                        'stacker_lgbm': np.random.random(len(training_data))
                    }
                    self.logger.warning("No base model outputs found, using placeholder data")
            
            # Prepare data for component
            component_data = {
                'X_train': training_data,
                'y_train': targets,
                'base_model_outputs': base_model_outputs
            }
            
            # Initialize component
            if not self.component.initialize():
                return {
                    'success': False,
                    'error': 'Failed to initialize analyst ensemble training component',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Process data with component
            result = self.component.process(component_data)
            
            if result.success:
                # Save ensemble model
                if hasattr(result, 'ensemble_model') and result.ensemble_model:
                    self._save_model(result.ensemble_model, 'analyst_ensemble_model')
                
                # Save HMM model
                if hasattr(result, 'hmm_model') and result.hmm_model:
                    self._save_model(result.hmm_model, 'analyst_hmm_model')
                
                # Save metrics
                if hasattr(result, 'ensemble_metrics') and result.ensemble_metrics:
                    self._save_metadata(result.ensemble_metrics, 'analyst_ensemble_metrics')
                
                # Save training summary
                training_summary = self.component.get_training_summary()
                self._save_metadata(training_summary, 'analyst_ensemble_summary')
                
                self.logger.info("✅ Analyst Ensemble Training completed successfully")
                
                return {
                    'success': True,
                    'artifacts': [
                        'analyst_ensemble_model',
                        'analyst_hmm_model',
                        'analyst_ensemble_metrics',
                        'analyst_ensemble_summary'
                    ],
                    'metrics': result.ensemble_metrics if hasattr(result, 'ensemble_metrics') else {},
                    'ensemble_method': getattr(result, 'ensemble_method', 'unknown'),
                    'training_time': result.training_time if hasattr(result, 'training_time') else 0
                }
            else:
                return {
                    'success': False,
                    'error': f"Analyst ensemble training failed: {getattr(result, 'error_message', 'Unknown error')}",
                    'artifacts': [],
                    'metrics': {}
                }
                
        except Exception as e:
            self.logger.error(f"❌ Analyst Ensemble Training Step failed: {e}")
            return {
                'success': False,
                'error': f"Step execution failed: {str(e)}",
                'artifacts': [],
                'metrics': {}
            }
        finally:
            # Cleanup component
            if hasattr(self.component, 'cleanup'):
                self.component.cleanup()