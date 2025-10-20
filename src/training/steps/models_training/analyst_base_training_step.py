"""
Analyst Base Training Step - BaseStep Wrapper

This module provides a BaseStep wrapper for the AnalystModelsTrainingModular component
to enable integration with ares_launcher.py.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.training.steps.base_step import BaseStep
from .components.analyst_models_training_modular import AnalystModelsTrainingModular


class AnalystBaseTrainingStep(BaseStep):
    """
    BaseStep wrapper for Analyst Base Models Training.
    
    This step trains individual Analyst base models:
    - LightGBM model
    - LightGBM + PatchTST features model
    - CatBoost model
    - Stacker LGBM Calibrated (meta-learner)
    """
    
    def __init__(self, step_name: str = "analyst_base_training", config: Optional[Dict[str, Any]] = None):
        """Initialize the analyst base training step."""
        super().__init__(step_name, config)
        self.logger = logging.getLogger(f"ares.step.{step_name}")
        
        # Initialize the modular component
        self.component = AnalystModelsTrainingModular(
            name="analyst_models_training",
            config=config,
            logger=self.logger
        )
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the analyst base training step.
        
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
            self.logger.info("🚀 Starting Analyst Base Training Step")
            
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
                # Try alternative names
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
                # Try to extract targets from training data
                target_columns = ['target', 'y', 'label', 'analyst_target']
                for col in target_columns:
                    if col in training_data.columns:
                        targets = training_data[col]
                        training_data = training_data.drop(columns=[col])
                        break
                
                if targets is None:
                    return {
                        'success': False,
                        'error': 'No target data found for analyst training',
                        'artifacts': [],
                        'metrics': {}
                    }
            
            # Prepare data for component
            component_data = {
                'X_train': training_data,
                'y_train': targets
            }
            
            # Initialize component
            if not self.component.initialize():
                return {
                    'success': False,
                    'error': 'Failed to initialize analyst training component',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Process data with component
            result = self.component.process(component_data)
            
            if result.success:
                # Save trained models
                if hasattr(result, 'models') and result.models:
                    self._save_model(result.models, 'analyst_base_models')
                
                # Save metrics
                if hasattr(result, 'metrics') and result.metrics:
                    self._save_metadata(result.metrics, 'analyst_training_metrics')
                
                # Save training summary
                training_summary = self.component.get_training_summary()
                self._save_metadata(training_summary, 'analyst_training_summary')
                
                self.logger.info("✅ Analyst Base Training completed successfully")
                
                return {
                    'success': True,
                    'artifacts': [
                        'analyst_base_models',
                        'analyst_training_metrics',
                        'analyst_training_summary'
                    ],
                    'metrics': result.metrics if hasattr(result, 'metrics') else {},
                    'models_trained': len(result.models) if hasattr(result, 'models') else 0,
                    'training_time': result.training_time if hasattr(result, 'training_time') else 0
                }
            else:
                return {
                    'success': False,
                    'error': f"Analyst training failed: {getattr(result, 'error_message', 'Unknown error')}",
                    'artifacts': [],
                    'metrics': {}
                }
                
        except Exception as e:
            self.logger.error(f"❌ Analyst Base Training Step failed: {e}")
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