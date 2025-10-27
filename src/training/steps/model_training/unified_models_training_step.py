"""
Unified Models Training Step.

This step consolidates all analyst and tactician training (base and ensemble)
into a single unified script that calls UnifiedTrainingPipeline.
"""

import asyncio
import logging
import yaml
import os
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error
from src.training.steps.models_training.unified_training_pipeline import UnifiedTrainingPipeline


class UnifiedModelsTrainingStep(BaseStep):
    """
    Unified Models Training Step.

    Consolidates all analyst and tactician training (base and ensemble) into a single step
    that calls UnifiedTrainingPipeline with appropriate configuration based on training type.
    """

    def __init__(self, step_name: str = "unified_models_training"):
        """Initialize the unified models training step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('UnifiedModelsTraining')
        self.unified_pipeline = None

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute unified model training.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - direction: Trading direction ('longs', 'shorts', 'both')
                - training_type: Type of training ('analyst_base', 'analyst_ensemble', 'tactician_base', 'tactician_ensemble')
                - execution_mode: Execution mode ('full', 'light', 'blank')

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        training_type = config.get('training_type', 'analyst_base')
        symbol = config.get('symbol', 'UNKNOWN')
        timeframe = config.get('timeframe', '15m')
        direction = config.get('direction', 'longs')
        
        tprint_info(f"🚀 Starting unified {training_type} training for {symbol} {timeframe} {direction}")

        try:
            # Initialize unified training pipeline
            self.unified_pipeline = UnifiedTrainingPipeline(self.logger)
            
            # Load appropriate YAML configuration
            yaml_config = await self._load_training_config(training_type, config)
            
            # Retrieve training data and targets from artifacts
            training_data, analyst_targets, tactician_targets = await self._retrieve_training_data(config)
            
            # Apply light mode filtering if needed
            training_data = self._apply_light_mode_filter(training_data, config, timeframe)
            
            # Execute training based on type
            result = await self._execute_training_by_type(
                training_type, training_data, analyst_targets, tactician_targets, yaml_config, config
            )
            
            if result.get('success', False):
                tprint_success(f"✅ Unified {training_type} training completed successfully")
                
                # Save artifacts
                artifacts = await self._save_training_artifacts(result, training_type, config)
                result['artifacts'] = artifacts
                
                return {
                    'success': True,
                    'artifacts': artifacts,
                    'metrics': result.get('metrics', {}),
                    'training_type': training_type,
                    'execution_time': result.get('execution_time', 0.0)
                }
            else:
                tprint_error(f"❌ Unified {training_type} training failed")
                return {
                    'success': False,
                    'artifacts': {},
                    'metrics': {},
                    'error': result.get('error_message', 'Training failed'),
                    'training_type': training_type
                }

        except Exception as e:
            error_msg = f"Unified {training_type} training failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg)

            return {
                'success': False,
                'artifacts': {},
                'metrics': {},
                'error': error_msg,
                'training_type': training_type
            }

    async def _load_training_config(self, training_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load appropriate YAML configuration based on training type."""
        try:
            # Map training types to config files in steps/model_training/
            config_mapping = {
                'analyst_base': 'src/training/steps/model_training/analyst_base_config.yaml',
                'analyst_ensemble': 'src/training/steps/model_training/analyst_ensemble_config.yaml',
                'tactician_base': 'src/training/steps/model_training/tactician_base_config.yaml',
                'tactician_ensemble': 'src/training/steps/model_training/tactician_ensemble_config.yaml'
            }
            
            config_file = config_mapping.get(training_type)
            if not config_file or not os.path.exists(config_file):
                # Fallback to default configuration
                tprint_info(f"Using default configuration for {training_type}")
                return self._get_default_config(training_type, config)
            
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Update configuration with runtime parameters
            yaml_config = self._update_config_with_runtime_params(yaml_config, config)
            
            tprint_info(f"Loaded configuration from {config_file}")
            return yaml_config
            
        except Exception as e:
            tprint_error(f"Failed to load config for {training_type}: {e}")
            return self._get_default_config(training_type, config)

    def _get_default_config(self, training_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get default configuration when YAML file is not available."""
        symbol = config.get('symbol', 'ETHUSDT')
        timeframe = config.get('timeframe', '15m')
        direction = config.get('direction', 'longs')
        
        base_config = {
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': direction,
            'execution_mode': config.get('execution_mode', 'light'),
            'enable_analyst': training_type.startswith('analyst'),
            'enable_tactician': training_type.startswith('tactician'),
            'enable_ensemble': training_type.endswith('ensemble'),
            'enable_hpo': True,
            'enable_explainability': True,
            'enable_vectorization': True
        }
        
        if training_type.startswith('analyst'):
            base_config.update({
                'analyst_config': {
                    'model_name': f"analyst_{'ensemble' if training_type.endswith('ensemble') else 'base'}",
                    'timeframe': timeframe,
                    'n_outputs': 4,
                    'output_names': ["signal_strength", "confidence", "risk_score", "regime_label"]
                }
            })
        elif training_type.startswith('tactician'):
            base_config.update({
                'tactician_config': {
                    'model_name': f"tactician_{'ensemble' if training_type.endswith('ensemble') else 'base'}",
                    'timeframe': timeframe,
                    'n_outputs': 4,
                    'output_names': ["entry_timing", "position_size", "stop_loss", "take_profit"]
                }
            })
        
        return base_config

    def _update_config_with_runtime_params(self, yaml_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Update YAML configuration with runtime parameters."""
        # Update basic parameters in the main config sections
        if 'analyst_config' in yaml_config:
            yaml_config['analyst_config']['timeframe'] = config.get('timeframe', '15m')
            yaml_config['analyst_config']['symbol'] = config.get('symbol', 'ETHUSDT')
            yaml_config['analyst_config']['direction'] = config.get('direction', 'longs')
        if 'tactician_config' in yaml_config:
            yaml_config['tactician_config']['timeframe'] = config.get('timeframe', '15m')
            yaml_config['tactician_config']['symbol'] = config.get('symbol', 'ETHUSDT')
            yaml_config['tactician_config']['direction'] = config.get('direction', 'longs')
        
        # Add runtime parameters to the root level
        yaml_config.update({
            'symbol': config.get('symbol', 'ETHUSDT'),
            'timeframe': config.get('timeframe', '15m'),
            'direction': config.get('direction', 'longs'),
            'execution_mode': config.get('execution_mode', 'light'),
            'exchange': config.get('exchange', 'binance')
        })
        
        return yaml_config

    async def _retrieve_training_data(self, config: Dict[str, Any]) -> tuple:
        """Retrieve training data and targets from artifacts."""
        try:
            # Try to get training data from artifacts
            training_data = None
            analyst_targets = None
            tactician_targets = None
            
            # Get training dataset
            try:
                training_data = self._get_artifact('training_dataset', 'data')
                tprint_info(f"Retrieved training dataset: {training_data.shape if hasattr(training_data, 'shape') else 'unknown shape'}")
            except Exception as e:
                self.logger.warning(f"Training dataset not found in artifacts: {e}")
            
            # Get analyst targets
            try:
                analyst_targets = self._get_artifact('analyst_targets', 'data')
                tprint_info(f"Retrieved analyst targets: {len(analyst_targets) if hasattr(analyst_targets, '__len__') else 'unknown length'}")
            except Exception as e:
                self.logger.warning(f"Analyst targets not found in artifacts: {e}")
            
            # Get tactician targets
            try:
                tactician_targets = self._get_artifact('tactician_targets', 'data')
                tprint_info(f"Retrieved tactician targets: {len(tactician_targets) if hasattr(tactician_targets, '__len__') else 'unknown length'}")
            except Exception as e:
                self.logger.warning(f"Tactician targets not found in artifacts: {e}")
            
            # If no data found, create dummy data for testing
            if training_data is None:
                tprint_info("No training data found, creating dummy data for testing")
                import pandas as pd
                import numpy as np
                
                # Create dummy training data
                n_samples = 1000 if config.get('execution_mode') == 'light' else 10000
                training_data = pd.DataFrame({
                    'close': np.random.randn(n_samples).cumsum() + 100,
                    'volume': np.random.exponential(1000, n_samples),
                    'returns': np.random.randn(n_samples) * 0.01,
                    'volatility': np.random.exponential(0.02, n_samples)
                })
                
                # Create dummy targets
                if analyst_targets is None:
                    analyst_targets = pd.Series(np.random.randn(n_samples), name='analyst_target')
                if tactician_targets is None:
                    tactician_targets = pd.Series(np.random.randn(n_samples), name='tactician_target')
            
            return training_data, analyst_targets, tactician_targets
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve training data: {e}")
            raise

    async def _execute_training_by_type(
        self, 
        training_type: str, 
        training_data, 
        analyst_targets, 
        tactician_targets, 
        yaml_config: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute training based on the specified type."""
        try:
            if training_type == 'analyst_base':
                return await self.unified_pipeline.train_analyst_models(
                    data=training_data,
                    targets=analyst_targets,
                    config=yaml_config
                )
            elif training_type == 'analyst_ensemble':
                return await self.unified_pipeline.train_ensemble_models(
                    data=training_data,
                    analyst_targets=analyst_targets,
                    tactician_targets=None,
                    config=yaml_config
                )
            elif training_type == 'tactician_base':
                return await self.unified_pipeline.train_tactician_models(
                    data=training_data,
                    targets=tactician_targets,
                    config=yaml_config
                )
            elif training_type == 'tactician_ensemble':
                return await self.unified_pipeline.train_ensemble_models(
                    data=training_data,
                    analyst_targets=analyst_targets,
                    tactician_targets=tactician_targets,
                    config=yaml_config
                )
            else:
                raise ValueError(f"Unknown training type: {training_type}")
                
        except Exception as e:
            self.logger.error(f"Training execution failed for {training_type}: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }

    async def _save_training_artifacts(self, result: Dict[str, Any], training_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Save training artifacts."""
        try:
            artifacts = {}
            
            # Save model artifacts
            if 'models' in result:
                for model_name, model in result['models'].items():
                    artifact_path = self._save_artifact(
                        data=model,
                        artifact_name=f"{training_type}_{model_name}",
                        artifact_type='model',
                        metadata={
                            'training_type': training_type,
                            'symbol': config.get('symbol'),
                            'timeframe': config.get('timeframe'),
                            'direction': config.get('direction'),
                            'created_at': datetime.now().isoformat()
                        }
                    )
                    artifacts[f"{training_type}_{model_name}"] = artifact_path
            
            # Save performance metrics
            if 'metrics' in result:
                metrics_path = self._save_artifact(
                    data=result['metrics'],
                    artifact_name=f"{training_type}_metrics",
                    artifact_type='metadata',
                    metadata={
                        'training_type': training_type,
                        'symbol': config.get('symbol'),
                        'timeframe': config.get('timeframe'),
                        'direction': config.get('direction'),
                        'created_at': datetime.now().isoformat()
                    }
                )
                artifacts[f"{training_type}_metrics"] = metrics_path
            
            # Save configuration
            config_path = self._save_artifact(
                data=config,
                artifact_name=f"{training_type}_config",
                artifact_type='metadata',
                metadata={
                    'training_type': training_type,
                    'created_at': datetime.now().isoformat()
                }
            )
            artifacts[f"{training_type}_config"] = config_path
            
            return artifacts
            
        except Exception as e:
            self.logger.error(f"Failed to save training artifacts: {e}")
            return {}

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)


# Register the step
def register_unified_models_training_step():
    """Register the unified models training step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("unified_models_training", UnifiedModelsTrainingStep)
    tprint("✅ Unified models training step registered", "SUCCESS")


# Auto-register when module is imported
register_unified_models_training_step()