"""
Unified Models Training Step.

This step consolidates all analyst and tactician training (base and ensemble)
into a single unified script that calls UnifiedTrainingPipeline.
"""

import asyncio
import logging
import yaml
import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

# --- HPO IMPORTS ---
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    ParameterGroup,
    OptimizationStage
)
from src.training.steps.model_training.hpo_config import (
    HPOOrchestrator,
    ModelParameterGroups,
    YAMLConfigUpdater,
    CustomBalancedScoreObjective
)
# --- END HPO IMPORTS ---

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning

# Import dynamic config calculator
from src.training.steps.model_training.dynamic_config_calculator import (
    DynamicConfigCalculator, DynamicTrainingConfig
)

# Try to import unified training pipeline if it exists, otherwise use placeholder
try:
    from src.training.steps.models_training.unified_training_pipeline import UnifiedTrainingPipeline
    UNIFIED_PIPELINE_AVAILABLE = True
except ImportError:
    UNIFIED_PIPELINE_AVAILABLE = False
    tprint_info("UnifiedTrainingPipeline not available, using placeholder")


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
        self.param_groups_factory = ModelParameterGroups()
        self.hpo_orchestrator = None

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute unified model training.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - direction: Trading direction ('long', 'short', 'both')
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
        direction = config.get('direction', 'long')
        
        tprint_info(f"🚀 Starting unified {training_type} training for {symbol} {timeframe} {direction}")

        try:
            # Check if unified pipeline is available
            if not UNIFIED_PIPELINE_AVAILABLE:
                tprint_error("UnifiedTrainingPipeline not available - cannot train models")
                return {
                    'success': False,
                    'artifacts': {},
                    'metrics': {},
                    'error': "UnifiedTrainingPipeline not available",
                    'training_type': training_type
                }
            
            # Initialize unified training pipeline
            self.unified_pipeline = UnifiedTrainingPipeline(self.logger)
            
            # Load appropriate YAML configuration
            yaml_config = await self._load_training_config(training_type, config)
            
            # Retrieve training data and targets from artifacts
            training_data, analyst_targets, tactician_targets = await self._retrieve_training_data(config)
            
            # --- MODIFIED: Retrieve and merge additional features for ensemble/tactician models ---
            if training_type.endswith('ensemble') or training_type == 'tactician_base':
                tprint_info(f"Retrieving additional model outputs for {training_type}...")
                additional_outputs = await self._get_additional_model_outputs(training_type, config)
                
                if additional_outputs is not None:
                    # Align indices before concatenating
                    aligned_training_data, aligned_additional_outputs = training_data.align(additional_outputs, join='inner', axis=0)
                    
                    if aligned_training_data.empty:
                        tprint_warning("Data alignment resulted in empty DataFrame. Check for index mismatches.")
                        # Fallback to original data if alignment fails
                    else:
                        training_data = pd.concat([aligned_training_data, aligned_additional_outputs], axis=1)
                        tprint_success(f"✅ Merged additional features. New training data shape: {training_data.shape}")
                else:
                    tprint_warning(f"No additional model outputs found for {training_type}. Proceeding with primary features only.")
            # --- END MODIFICATION ---

            # Apply light mode filtering if needed
            training_data = self._apply_light_mode_filter(training_data, config, timeframe)
            
            # Calculate COMPREHENSIVE dynamic configuration based on data and hardware
            if training_data is not None:
                tprint_info("🚀 Calculating comprehensive dynamic configuration...")
                
                calculator = DynamicConfigCalculator()
                dynamic_config = calculator.calculate_all_parameters(
                    total_samples=len(training_data),
                    n_features=len(training_data.columns),
                    timeframe=timeframe,
                    execution_mode=config.get('execution_mode', 'full'),
                    model_type='ensemble',  # Generic, will be refined per model
                    training_type=training_type,
                    train_percentage=config.get('train_percentage', 0.70),
                    validation_percentage=config.get('validation_percentage', 0.15),
                    test_percentage=config.get('test_percentage', 0.15)
                )
                
                # Apply dynamic configuration to YAML config
                yaml_config = self._apply_dynamic_config(yaml_config, dynamic_config, training_type)
                tprint_success(f"✅ Configured training with dynamic parameters (samples, epochs, batch size, memory, etc.)")
            else:
                tprint_warning("No training data available, using default configuration from YAML")
            
            # Perform hyperparameter optimization before training
            if config.get('enable_hpo', True) and training_data is not None:
                # Determine which targets to use for HPO
                hpo_targets = analyst_targets if training_type.startswith('analyst') else tactician_targets
                if hpo_targets is not None:
                    tprint_info("🔍 Performing hyperparameter optimization using custom_balanced_score...")
                    
                    # Get the appropriate model config
                    if training_type.startswith('analyst'):
                        model_config_key = 'analyst_config'
                    elif training_type.startswith('tactician'):
                        model_config_key = 'tactician_config'
                    else:
                        model_config_key = 'ensemble_config'
                    
                    # Get config file path for this training type
                    config_mapping = {
                        'analyst_base': 'src/training/steps/model_training/analyst_base_config.yaml',
                        'analyst_ensemble': 'src/training/steps/model_training/analyst_ensemble_config.yaml',
                        'tactician_base': 'src/training/steps/model_training/tactician_base_config.yaml',
                        'tactician_ensemble': 'src/training/steps/model_training/tactician_ensemble_config.yaml'
                    }
                    config_file = config_mapping.get(training_type)
                    
                    if model_config_key in yaml_config and config_file:
                        # Use new HPO system with custom_balanced_score
                        yaml_config[model_config_key] = await self._perform_hierarchical_hpo(
                            training_data=training_data,
                            targets=hpo_targets,
                            model_config=yaml_config[model_config_key],
                            config_file=config_file,
                            config=config,
                            training_type=training_type
                        )
                    else:
                        tprint_warning(f"No {model_config_key} found in config or config file, skipping HPO")
                else:
                    tprint_warning("No targets available for HPO, skipping optimization")
            else:
                tprint_info("Hyperparameter optimization disabled or no training data available")
            
            # Execute training based on type
            result = await self._execute_training_by_type(
                training_type, training_data, analyst_targets, tactician_targets, yaml_config, config
            )
            
            if result.get('success', False):
                tprint_success(f"✅ Unified {training_type} training completed successfully")
                
                # Save artifacts
                artifacts = await self._save_training_artifacts(result, training_type, config)
                result['artifacts'] = artifacts
                
                # Save ML-scored historical data for backtesting
                if training_data is not None and 'predictions' in result:
                    try:
                        model_type = 'analyst' if training_type.startswith('analyst') else 'tactician'
                        tprint_info(f"📊 Saving ML-scored historical data ({model_type})...")
                        
                        ml_scored_path = self._save_ml_scored_data(
                            data=training_data,
                            predictions=result['predictions'],
                            model_type=model_type,
                            config=config,
                            metadata={
                                'training_type': training_type,
                                'metrics': result.get('metrics', {}),
                                'model_names': list(result.get('models', {}).keys())
                            }
                        )
                        
                        artifacts['ml_scored_historical_data'] = ml_scored_path
                        tprint_success(f"✅ ML-scored data saved: {ml_scored_path}")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to save ML-scored data: {e}")
                        self.logger.warning(f"ML-scored data save failed: {e}")
                
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
            import traceback
            error_msg = f"Unified {training_type} training failed: {str(e)}"
            traceback_str = traceback.format_exc()
            tprint_error(f"❌ {error_msg}")
            tprint_error(f"Traceback:\n{traceback_str}")
            self.logger.error(error_msg)
            self.logger.error(traceback_str)

            return {
                'success': False,
                'artifacts': {},
                'metrics': {},
                'error': error_msg,
                'traceback': traceback_str,
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
        direction = config.get('direction', 'long')
        
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
            yaml_config['analyst_config']['direction'] = config.get('direction', 'long')
        if 'tactician_config' in yaml_config:
            yaml_config['tactician_config']['timeframe'] = config.get('timeframe', '15m')
            yaml_config['tactician_config']['symbol'] = config.get('symbol', 'ETHUSDT')
            yaml_config['tactician_config']['direction'] = config.get('direction', 'long')
        
        # Add runtime parameters to the root level
        yaml_config.update({
            'symbol': config.get('symbol', 'ETHUSDT'),
            'timeframe': config.get('timeframe', '15m'),
            'direction': config.get('direction', 'long'),
            'execution_mode': config.get('execution_mode', 'light'),
            'exchange': config.get('exchange', 'binance')
        })
        
        # Apply light mode optimizations for TCN if in light execution mode
        execution_mode = config.get('execution_mode', 'light')
        if execution_mode == 'light':
            self._apply_light_mode_tcn_optimizations(yaml_config)
        
        return yaml_config
    
    def _apply_light_mode_tcn_optimizations(self, yaml_config: Dict[str, Any]) -> None:
        """Apply aggressive model optimizations for light mode execution (10x lighter)."""
        execution_mode = yaml_config.get('execution_mode', 'light')
        
        # Check if analyst_config exists
        if 'analyst_config' in yaml_config:
            base_models = yaml_config['analyst_config'].get('base_models', {})
            
            # Optimize TCN
            if 'tcn' in base_models:
                tcn_config = base_models['tcn']
                tprint_warning(f"⚡ Applying {execution_mode.upper()} mode TCN optimizations (10x lighter)")
                
                # Drastically reduce TCN parameters for light mode
                tcn_params = tcn_config.get('params', {})
                tcn_params['num_filters'] = 32  # Reduced from 64
                tcn_params['num_layers'] = 2  # Reduced from 4
                tcn_params['epochs'] = 10  # Reduced from 50 (10x lighter)
                tcn_params['batch_size'] = 128  # Increased from 64 (fewer iterations)
                tcn_params['early_stopping_patience'] = 3  # Reduced from 7
                tcn_params['use_autoencoder'] = False  # Disabled to save 25 epochs
                tcn_params['autoencoder_epochs'] = 5  # Reduced from 25 if autoencoder is re-enabled
                
                # Disable TCN HPO in light mode
                if 'hpo' in tcn_config:
                    tcn_config['hpo']['enabled'] = False
                
                tprint_info(f"  TCN epochs: 50 → 10 (10x lighter)")
                tprint_info(f"  TCN autoencoder: DISABLED (saves 25 epochs)")
                tprint_info(f"  TCN HPO: DISABLED")
            
            # Optimize CatBoost
            if 'catboost' in base_models:
                catboost_config = base_models['catboost']
                tprint_warning(f"⚡ Applying {execution_mode.upper()} mode CatBoost optimizations (10x lighter)")
                
                # Reduce CatBoost iterations for light mode
                catboost_params = catboost_config.get('params', {})
                catboost_params['iterations'] = 50  # Reduced from 500 (10x lighter)
                catboost_params['depth'] = 4  # Reduced from 6
                catboost_params['early_stopping_rounds'] = 10  # Reduced from 50
                
                # Disable CatBoost HPO in light mode
                if 'hpo' in catboost_config:
                    catboost_config['hpo']['enabled'] = False
                
                tprint_info(f"  CatBoost iterations: 500 → 50 (10x lighter)")
                tprint_info(f"  CatBoost depth: 6 → 4")
                tprint_info(f"  CatBoost HPO: DISABLED")
            
            # Optimize LGBM
            if 'lgbm' in base_models:
                lgbm_config = base_models['lgbm']
                tprint_warning(f"⚡ Applying {execution_mode.upper()} mode LGBM optimizations (10x lighter)")
                
                # Reduce LGBM estimators for light mode
                lgbm_params = lgbm_config.get('params', {})
                lgbm_params['n_estimators'] = 100  # Reduced from 1000 (10x lighter)
                lgbm_params['max_depth'] = 6  # Reduced from 8
                
                # Disable LGBM HPO in light mode
                if 'hpo' in lgbm_config:
                    lgbm_config['hpo']['enabled'] = False
                
                tprint_info(f"  LGBM n_estimators: 1000 → 100 (10x lighter)")
                tprint_info(f"  LGBM max_depth: 8 → 6")
                tprint_info(f"  LGBM HPO: DISABLED")
        
        # Check if tactician_config has GRU model
        if 'tactician_config' in yaml_config:
            base_models = yaml_config['tactician_config'].get('base_models', [])
            for model in base_models:
                if model.get('model_name') == 'StandaloneGRU':
                    tprint_warning(f"⚡ Applying {execution_mode.upper()} mode GRU optimizations (10x lighter)")
                    params = model.get('params', {})
                    params['epochs'] = 10  # Reduce epochs (10x lighter)
                    params['batch_size'] = 128  # Increase batch size
                    if 'hpo' in model:
                        model['hpo']['enabled'] = False
                    tprint_info(f"  GRU epochs: Reduced to 10")
                    tprint_info(f"  GRU HPO: DISABLED")
    
    def _calculate_sample_allocations(self, total_samples: int, config: Dict[str, Any]) -> Dict[str, int]:
        """
        Calculate sample allocations based on percentages of total samples.
        
        Args:
            total_samples: Total number of samples available
            config: Configuration dictionary with optional percentage overrides
            
        Returns:
            Dictionary with calculated sample counts for train, validation, test, and CV
        """
        # Default percentages
        default_train_pct = config.get('train_percentage', 0.70)  # 70% for training
        default_val_pct = config.get('validation_percentage', 0.15)  # 15% for validation
        default_test_pct = config.get('test_percentage', 0.15)  # 15% for testing
        
        # Ensure percentages sum to 1.0
        total_pct = default_train_pct + default_val_pct + default_test_pct
        if not np.isclose(total_pct, 1.0):
            tprint_warning(f"Sample percentages sum to {total_pct}, normalizing to 1.0")
            default_train_pct /= total_pct
            default_val_pct /= total_pct
            default_test_pct /= total_pct
        
        # Calculate sample counts
        train_samples = int(total_samples * default_train_pct)
        val_samples = int(total_samples * default_val_pct)
        test_samples = total_samples - train_samples - val_samples  # Use remaining for test to avoid rounding issues
        
        # CV folds (default to 5)
        cv_folds = config.get('cv_folds', 5)
        
        tprint_info(f"Sample allocation for {total_samples} total samples:")
        tprint_info(f"  Training: {train_samples} ({default_train_pct*100:.1f}%)")
        tprint_info(f"  Validation: {val_samples} ({default_val_pct*100:.1f}%)")
        tprint_info(f"  Test: {test_samples} ({default_test_pct*100:.1f}%)")
        tprint_info(f"  CV Folds: {cv_folds}")
        
        return {
            'training_samples': train_samples,
            'validation_samples': val_samples,
            'test_samples': test_samples,
            'cv_folds': cv_folds
        }
    
    def _override_training_config_with_allocations(
        self, 
        yaml_config: Dict[str, Any], 
        allocations: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Override training configuration with calculated sample allocations.
        
        Args:
            yaml_config: YAML configuration dictionary
            allocations: Calculated sample allocations
            
        Returns:
            Updated configuration dictionary
        """
        # Update analyst config if present
        if 'analyst_config' in yaml_config and 'training' in yaml_config['analyst_config']:
            yaml_config['analyst_config']['training'].update(allocations)
            tprint_info("Updated analyst_config with calculated allocations")
        
        # Update tactician config if present
        if 'tactician_config' in yaml_config and 'training' in yaml_config['tactician_config']:
            yaml_config['tactician_config']['training'].update(allocations)
            tprint_info("Updated tactician_config with calculated allocations")
        
        # Update root-level training config if present
        if 'training' in yaml_config:
            yaml_config['training'].update(allocations)
            tprint_info("Updated root training config with calculated allocations")
        
        return yaml_config
    
    # --- NEW HPO METHOD USING CUSTOM_BALANCED_SCORE ---
    async def _perform_hierarchical_hpo(
        self,
        training_data: pd.DataFrame,
        targets: pd.Series,
        model_config: Dict[str, Any],
        config_file: str,
        config: Dict[str, Any],
        training_type: str
    ) -> Dict[str, Any]:
        """
        Perform hierarchical hyperparameter optimization using custom_balanced_score.
        
        This method uses the new HPO system from hpo_config.py which:
        1. Reads parameter ranges from YAML files
        2. Uses custom_balanced_score as optimization metric
        3. Performs hierarchical optimization (2 rounds by default)
        4. Saves optimal parameters back to YAML files
        
        Args:
            training_data: Training data
            targets: Target variables
            model_config: Model configuration dictionary
            config_file: Path to YAML config file
            config: General configuration dictionary
            training_type: Type of training (analyst_base, etc.)
            
        Returns:
            Updated model configuration with optimized parameters
        """
        try:
            # Check if HPO is enabled
            enable_hpo = config.get('enable_hpo', True)
            if not enable_hpo:
                tprint_info("Hyperparameter optimization disabled, using default parameters")
                return model_config
            
            tprint_info("🔍 Starting Hierarchical HPO with custom_balanced_score...")
            
            # Split data for HPO validation (80/20 split)
            hpo_train_size = int(len(training_data) * 0.8)
            X_train = training_data.iloc[:hpo_train_size]
            X_val = training_data.iloc[hpo_train_size:]
            y_train = targets.iloc[:hpo_train_size]
            y_val = targets.iloc[hpo_train_size:]
            
            tprint_info(f"HPO split: {len(X_train)} train, {len(X_val)} validation samples")
            
            # Create HPO orchestrator
            execution_mode = config.get('execution_mode', 'full')
            self.hpo_orchestrator = HPOOrchestrator(
                config_file=config_file,
                execution_mode=execution_mode
            )
            
            # Determine which models to optimize
            models_to_optimize = []
            
            if training_type.endswith('ensemble'):
                # Ensemble models: optimize the meta-learner
                if 'meta_learner' in model_config:
                    models_to_optimize.append({
                        'name': 'meta_learner',
                        'type': model_config['meta_learner'].get('model_type', 'stacker_lgbm_calibrated'),
                        'class': lgb.LGBMRegressor,
                        'is_classification': False
                    })
            else:
                # Base models: optimize each base model
                if 'base_models' in model_config:
                    # Handle both list and dict formats
                    base_models = model_config['base_models']
                    if isinstance(base_models, list):
                        # List format: iterate through list items
                        for model_item in base_models:
                            model_name = model_item.get('model_name', 'unknown')
                            model_params = model_item
                            model_type = model_params.get('model_type', '')
                            
                            # Map model types to classes
                            if 'lgbm' in model_type.lower():
                                model_class = lgb.LGBMRegressor
                                is_classification = False
                            elif 'catboost' in model_type.lower():
                                import catboost as cb
                                model_class = cb.CatBoostRegressor
                                is_classification = False
                            else:
                                # Skip models we don't support yet (TCN, GRU, etc.)
                                tprint_info(f"Skipping HPO for {model_name} ({model_type}) - not yet supported")
                                continue
                            
                            models_to_optimize.append({
                                'name': model_name,
                                'type': model_type,
                                'class': model_class,
                                'is_classification': is_classification
                            })
                    else:
                        # Dict format: use items()
                        for model_name, model_params in base_models.items():
                            model_type = model_params.get('model_type', '')
                            
                            # Map model types to classes
                            if 'lgbm' in model_type.lower():
                                model_class = lgb.LGBMRegressor
                                is_classification = False
                            elif 'catboost' in model_type.lower():
                                import catboost as cb
                                model_class = cb.CatBoostRegressor
                                is_classification = False
                            else:
                                # Skip models we don't support yet (TCN, GRU, etc.)
                                tprint_info(f"Skipping HPO for {model_name} ({model_type}) - not yet supported")
                                continue
                            
                            models_to_optimize.append({
                                'name': model_name,
                                'type': model_type,
                                'class': model_class,
                                'is_classification': is_classification
                            })
            
            # Run HPO for each model
            all_results = {}
            for model_info in models_to_optimize:
                tprint_info(f"🎯 Optimizing {model_info['name']} ({model_info['type']})...")
                
                # Run HPO in separate thread to avoid blocking event loop
                result = await asyncio.to_thread(
                    self.hpo_orchestrator.run_hpo,
                    model_name=model_info['name'],
                    model_type=model_info['type'],
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    model_class=model_info['class'],
                    is_classification=model_info['is_classification']
                )
                
                if result:
                    all_results[model_info['name']] = result
                    tprint_success(f"✅ {model_info['name']} HPO complete: score={result.best_score:.6f}")
                    tprint_info(f"   Optimal params: {result.best_params}")
                else:
                    tprint_warning(f"⚠️ HPO failed for {model_info['name']}, using default parameters")
            
            # Reload the updated YAML config (it was updated by HPOOrchestrator)
            with open(config_file, 'r') as f:
                updated_yaml = yaml.safe_load(f)
            
            # Extract the relevant config section
            if training_type.startswith('analyst'):
                updated_model_config = updated_yaml.get('analyst_config', model_config)
            elif training_type.startswith('tactician'):
                updated_model_config = updated_yaml.get('tactician_config', model_config)
            else:
                updated_model_config = model_config
            
            tprint_success(f"✅ HPO complete for {len(all_results)} models")
            tprint_info(f"   Optimal parameters saved to {config_file}")
            
            return updated_model_config
            
        except Exception as e:
            tprint_error(f"Hierarchical HPO failed: {e}")
            import traceback
            self.logger.error(f"HPO error: {e}\n{traceback.format_exc()}")
            return model_config  # Return original config on failure
    
    # Legacy method removed - search spaces now defined in YAML files
    # See hpo_config.py for parameter group definitions
    
    def _apply_dynamic_config(
        self,
        yaml_config: Dict[str, Any],
        dynamic_config: DynamicTrainingConfig,
        training_type: str
    ) -> Dict[str, Any]:
        """
        Apply dynamic configuration to YAML config.
        
        Args:
            yaml_config: YAML configuration dictionary
            dynamic_config: Dynamically calculated configuration
            training_type: Type of training (analyst_base, tactician_base, etc.)
            
        Returns:
            Updated YAML configuration
        """
        try:
            tprint_info("🔧 Applying dynamic configuration to YAML config...")
            
            # Determine which config section to update
            if training_type.startswith('analyst'):
                config_key = 'analyst_config'
            elif training_type.startswith('tactician'):
                config_key = 'tactician_config'
            else:
                config_key = 'ensemble_config'
            
            # Update the appropriate config section
            if config_key in yaml_config:
                # Update training parameters
                if 'training' in yaml_config[config_key]:
                    yaml_config[config_key]['training'].update({
                        'training_samples': dynamic_config.training_samples,
                        'validation_samples': dynamic_config.validation_samples,
                        'test_samples': dynamic_config.test_samples,
                        'cv_folds': dynamic_config.cv_folds,
                        'early_stopping_patience': dynamic_config.early_stopping_patience
                    })
                
                # Update base model parameters
                if 'base_models' in yaml_config[config_key]:
                    base_models = yaml_config[config_key]['base_models']
                    
                    # Handle both list and dict formats
                    if isinstance(base_models, list):
                        # List format: iterate through list items
                        for model_item in base_models:
                            model_name = model_item.get('model_name', 'unknown')
                            model_params = model_item
                            
                            if 'params' not in model_params:
                                model_params['params'] = {}
                            
                            # Update common parameters
                            model_type = model_params.get('model_type', '').lower()
                            
                            # Neural network models
                            if any(nn in model_type for nn in ['gru', 'lstm', 'tcn', 'transformer']):
                                model_params['params'].update({
                                    'batch_size': dynamic_config.batch_size,
                                    'epochs': dynamic_config.epochs if dynamic_config.epochs > 0 else 100,
                                    'learning_rate': dynamic_config.learning_rate,
                                    'early_stopping_patience': dynamic_config.early_stopping_patience
                                })
                                
                                # Add sequence length for time series models
                                if any(ts in model_type for ts in ['gru', 'lstm', 'tcn']):
                                    model_params['params']['sequence_length'] = dynamic_config.sequence_length
                            
                            # Tree-based models
                            elif any(tree in model_type for tree in ['lgbm', 'catboost', 'xgboost']):
                                if 'lgbm' in model_type:
                                    model_params['params']['n_estimators'] = dynamic_config.n_estimators
                                elif 'catboost' in model_type:
                                    tprint_info("Applying CatBoost GPU (Apple M1) configuration...")
                                    model_params['params']['task_type'] = 'GPU'
                                    model_params['params']['devices'] = '0' # Use '0' for Apple M1 GPU
                                    
                                    # Remove subsample if it exists, as it's not supported for GPU training
                                    if 'subsample' in model_params['params']:
                                        del model_params['params']['subsample']
                                        tprint_info("Removed 'subsample' param, not supported by CatBoost GPU.")     
                                        
                                model_params['params']['learning_rate'] = dynamic_config.learning_rate
                            
                            tprint_info(f"  Updated {model_name} with dynamic parameters")
                    else:
                        # Dict format: use items()
                        for model_name, model_params in base_models.items():
                            if 'params' not in model_params:
                                model_params['params'] = {}
                            
                            # Update common parameters
                            model_type = model_params.get('model_type', '').lower()
                            
                            # Neural network models
                            if any(nn in model_type for nn in ['gru', 'lstm', 'tcn', 'transformer']):
                                model_params['params'].update({
                                    'batch_size': dynamic_config.batch_size,
                                    'epochs': dynamic_config.epochs if dynamic_config.epochs > 0 else 100,
                                    'learning_rate': dynamic_config.learning_rate,
                                    'early_stopping_patience': dynamic_config.early_stopping_patience
                                })
                                
                                # Add sequence length for time series models
                                if any(ts in model_type for ts in ['gru', 'lstm', 'tcn']):
                                    model_params['params']['sequence_length'] = dynamic_config.sequence_length
                            
                            # Tree-based models
                            elif any(tree in model_type for tree in ['lgbm', 'catboost', 'xgboost']):
                                if 'lgbm' in model_type:
                                    model_params['params']['n_estimators'] = dynamic_config.n_estimators
                                elif 'catboost' in model_type:
                                    model_params['params']['iterations'] = dynamic_config.iterations
                                    tprint_info("Applying CatBoost GPU (Apple M1) configuration...")
                                    model_params['params']['task_type'] = 'GPU'
                                    model_params['params']['devices'] = '0' # Use '0' for Apple M1 GPU
                                    
                                    # Remove subsample if it exists, as it's not supported for GPU training
                                    if 'subsample' in model_params['params']:
                                        del model_params['params']['subsample']
                                        tprint_info("Removed 'subsample' param, not supported by CatBoost GPU.")
                                        
                                model_params['params']['learning_rate'] = dynamic_config.learning_rate
                            
                            tprint_info(f"  Updated {model_name} with dynamic parameters")
                
                # Update hardware settings
                if 'hardware' in yaml_config[config_key]:
                    yaml_config[config_key]['hardware'].update({
                        'memory_limit_gb': dynamic_config.memory_limit_gb,
                        'max_workers': dynamic_config.max_workers
                    })
                elif 'hardware' in yaml_config:
                    yaml_config['hardware'].update({
                        'memory_limit_gb': dynamic_config.memory_limit_gb,
                        'max_workers': dynamic_config.max_workers
                    })
            
            # Update root-level hardware settings if present
            if 'hardware' in yaml_config:
                yaml_config['hardware'].update({
                    'memory_limit_gb': dynamic_config.memory_limit_gb,
                    'max_workers': dynamic_config.max_workers
                })
            
            # Store HPO settings
            yaml_config['hpo_max_trials'] = dynamic_config.hpo_max_trials
            yaml_config['hpo_time_budget_seconds'] = dynamic_config.hpo_time_budget_seconds
            
            tprint_success("✅ Dynamic configuration applied successfully")
            return yaml_config
            
        except Exception as e:
            tprint_error(f"Failed to apply dynamic config: {e}")
            self.logger.error(f"Dynamic config application error: {e}")
            return yaml_config

    def _apply_light_mode_filter(self, training_data: pd.DataFrame, config: Dict[str, Any], timeframe: str) -> pd.DataFrame:
        """Apply light mode filtering to training data if needed."""
        try:
            execution_mode = config.get('execution_mode', 'light')
            
            if execution_mode == 'light' and training_data is not None:
                # Limit to 1000 samples in light mode
                if len(training_data) > 1000:
                    tprint_info(f"Light mode: Limiting training data from {len(training_data)} to 1000 samples")
                    training_data = training_data.tail(1000)
            
            return training_data
            
        except Exception as e:
            self.logger.warning(f"Error applying light mode filter: {e}")
            return training_data
    
    async def _retrieve_training_data(self, config: Dict[str, Any]) -> tuple:
        """Retrieve training data and targets from artifacts with fast-fail on missing data."""
        try:
            tprint_info("🔍 Retrieving training data from feature generation artifacts...")
            
            training_data = None
            analyst_targets = None
            tactician_targets = None
            
            # Determine feature set size to use (default to 50 features)
            feature_set_size = config.get('feature_set_size', 50)
            
            # Try to get selected features from feature_generation_final_feature_selection_step
            feature_artifact_names = [
                f'selected_feature_dataframe_{feature_set_size}',  # Specific size
                f'selected_features_{feature_set_size}',           # Alternative name
                'selected_feature_dataframe_50',                   # Fallback to 50
                'selected_feature_dataframe_60',                   # Fallback to 60
                'selected_feature_dataframe_40',                   # Fallback to 40
            ]
            
            for artifact_name in feature_artifact_names:
                try:
                    training_data = self._get_artifact(artifact_name, 'data')
                    if training_data is not None:
                        tprint_success(f"✅ Retrieved training features from '{artifact_name}': {training_data.shape}")
                        break
                except Exception as e:
                    self.logger.debug(f"Artifact '{artifact_name}' not found: {e}")
                    continue
            
            # If still no data, try alternative artifact names
            if training_data is None:
                alternative_names = [
                    'final_dataset',           # From final validation step
                    'labeled_data',            # From labeling integration step
                    'training_dataset',        # Generic name
                ]
                
                for artifact_name in alternative_names:
                    try:
                        training_data = self._get_artifact(artifact_name, 'data')
                        if training_data is not None:
                            tprint_success(f"✅ Retrieved training data from '{artifact_name}': {training_data.shape}")
                            break
                    except Exception as e:
                        self.logger.debug(f"Artifact '{artifact_name}' not found: {e}")
                        continue
            
            # FAIL FAST: If no training data found, raise error
            if training_data is None:
                error_msg = (
                    "❌ CRITICAL: No training data found in artifacts!\n"
                    f"   Expected artifacts from 'feature_generation_final_feature_selection_step':\n"
                    f"   - selected_feature_dataframe_{feature_set_size}\n"
                    f"   - selected_feature_dataframe_50/60/40\n"
                    f"   OR from other steps: final_dataset, labeled_data\n"
                    f"   \n"
                    f"   Please ensure feature_generation_final_feature_selection_step has run successfully.\n"
                    f"   Check artifacts directory for available artifacts."
                )
                tprint_error(error_msg)
                raise ValueError(error_msg)
            
            # Get targets from labeling integration step
            target_artifact_names = [
                'analyst_targets',             # Specific analyst targets
                'tactician_targets',           # Specific tactician targets
                'targets',                     # Generic targets
                'labeling_metadata',           # From labeling step
            ]
            
            # Try to get analyst targets
            for artifact_name in ['analyst_targets', 'targets']:
                try:
                    analyst_targets = self._get_artifact(artifact_name, 'data')
                    if analyst_targets is not None:
                        tprint_success(f"✅ Retrieved analyst targets from '{artifact_name}': {len(analyst_targets)} samples")
                        break
                except Exception as e:
                    self.logger.debug(f"Artifact '{artifact_name}' not found: {e}")
                    continue
            
            # Try to get tactician targets
            for artifact_name in ['tactician_targets', 'targets']:
                try:
                    tactician_targets = self._get_artifact(artifact_name, 'data')
                    if tactician_targets is not None:
                        tprint_success(f"✅ Retrieved tactician targets from '{artifact_name}': {len(tactician_targets)} samples")
                        break
                except Exception as e:
                    self.logger.debug(f"Artifact '{artifact_name}' not found: {e}")
                    continue
            
            # FALLBACK: Try to extract targets from labeled_data if separate targets not found
            if analyst_targets is None and tactician_targets is None:
                tprint_warning("⚠️ No separate target artifacts found, trying to extract from labeled_data...")
                
                # Try to get labeled_data artifact
                for artifact_name in ['labeled_data', 'labeled_features']:
                    try:
                        labeled_data = self._get_artifact(artifact_name, 'data')
                        if labeled_data is not None and isinstance(labeled_data, pd.DataFrame):
                            tprint_info(f"✅ Found labeled_data artifact: {labeled_data.shape}")
                            
                            # Extract target columns
                            target_cols = [col for col in labeled_data.columns if 'target' in col.lower()]
                            if target_cols:
                                tprint_success(f"✅ Extracting targets from labeled_data: {target_cols}")
                                # Use first target column as analyst targets
                                analyst_targets = labeled_data[target_cols[0]]
                                tprint_success(f"✅ Extracted analyst targets: {len(analyst_targets)} samples")
                                
                                # CRITICAL: Ensure training_data and targets are aligned
                                if len(training_data) != len(analyst_targets):
                                    tprint_warning(f"⚠️ Shape mismatch detected! Features: {len(training_data)}, Targets: {len(analyst_targets)}")
                                    tprint_warning(f"⚠️ Attempting to align by using labeled_data as both features and targets...")
                                    
                                    # Use labeled_data for features (drop target columns)
                                    feature_cols = [col for col in labeled_data.columns if col not in target_cols]
                                    training_data = labeled_data[feature_cols]
                                    analyst_targets = labeled_data[target_cols[0]]
                                    
                                    tprint_success(f"✅ Aligned data - Features: {training_data.shape}, Targets: {len(analyst_targets)} samples")
                                
                                break
                    except Exception as e:
                        self.logger.debug(f"Could not extract targets from '{artifact_name}': {e}")
                        continue
            
            # FAIL FAST: If still no targets found, raise error
            if analyst_targets is None and tactician_targets is None:
                error_msg = (
                    "❌ CRITICAL: No training targets found in artifacts!\n"
                    f"   Expected artifacts from labeling steps:\n"
                    f"   - analyst_targets\n"
                    f"   - tactician_targets\n"
                    f"   - targets (generic)\n"
                    f"   - labeled_data (with target columns)\n"
                    f"   \n"
                    f"   Please ensure labeling integration steps have run successfully."
                )
                tprint_error(error_msg)
                raise ValueError(error_msg)
            
            # Log summary
            tprint_info("📊 Training Data Summary:")
            tprint_info(f"   Features: {training_data.shape[0]} samples × {training_data.shape[1]} features")
            if analyst_targets is not None:
                tprint_info(f"   Analyst Targets: {len(analyst_targets)} samples")
            if tactician_targets is not None:
                tprint_info(f"   Tactician Targets: {len(tactician_targets)} samples")
            
            return training_data, analyst_targets, tactician_targets
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve training data: {e}")
            raise

    async def _get_primary_features(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Get primary features from feature generation step."""
        try:
            # Determine artifact name based on training type and config
            training_type = config.get('training_type', 'analyst_base')
            direction = config.get('direction', 'long')
            exchange = config.get('exchange', 'binance')
            symbol = config.get('symbol', 'ETHUSDT')
            
            if training_type.startswith('analyst'):
                artifact_name = f"analyst_features_{direction}_{exchange}_{symbol}"
            else:
                artifact_name = f"tactician_features_{direction}_{exchange}_{symbol}"
            
            features = await self._get_artifact(artifact_name, config)
            if features is None:
                # Fallback to generic artifact name
                features = await self._get_artifact('selected_features', config)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error retrieving primary features: {e}")
            return None

    async def _get_regime_features(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Get regime probability features."""
        try:
            regime_features = await self._get_artifact('regime_probabilities', config)
            if regime_features is None:
                # Try alternative artifact names
                regime_features = await self._get_artifact('regime_ml_outputs', config)
            
            return regime_features
            
        except Exception as e:
            self.logger.error(f"Error retrieving regime features: {e}")
            return None

    # --- MODIFIED: Added statistical meta-feature generation ---
    async def _get_additional_model_outputs(self, training_type: str, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Get additional model outputs based on training type."""
        try:
            additional_features_list = []
            base_outputs_for_stats = None # Store the specific DataFrame to calculate stats on

            if training_type == 'analyst_ensemble':
                # Base models for analyst_ensemble are the analyst_base outputs
                base_outputs = await self._get_artifact('analyst_base_outputs', config)
                if base_outputs is not None:
                    additional_features_list.append(base_outputs)
                    base_outputs_for_stats = base_outputs # Calculate stats on these
            
            elif training_type == 'tactician_base':
                # Base model for tactician_base is the analyst_ensemble output
                analyst_outputs = await self._get_artifact('analyst_ensemble_outputs', config)
                if analyst_outputs is not None:
                    additional_features_list.append(analyst_outputs)
                # No stats needed here, this is for a base model
            
            elif training_type == 'tactician_ensemble':
                # Base models for tactician_ensemble are the tactician_base outputs
                # Analyst_ensemble outputs are also included as features.
                analyst_outputs = await self._get_artifact('analyst_ensemble_outputs', config)
                tactician_base_outputs = await self._get_artifact('tactician_base_outputs', config)
                
                if analyst_outputs is not None:
                    additional_features_list.append(analyst_outputs)
                if tactician_base_outputs is not None:
                    additional_features_list.append(tactician_base_outputs)
                    base_outputs_for_stats = tactician_base_outputs # Calculate stats on these

            # --- NEW: Calculate ensemble meta-features ---
            if base_outputs_for_stats is not None and not base_outputs_for_stats.empty:
                tprint_info(f"Calculating ensemble meta-features from {base_outputs_for_stats.shape[1]} base model outputs...")
                meta_features = pd.DataFrame(index=base_outputs_for_stats.index)
                
                meta_features['ens_pred_variance'] = base_outputs_for_stats.var(axis=1)
                meta_features['ens_avg_confidence'] = base_outputs_for_stats.mean(axis=1)
                meta_features['ens_conf_spread'] = base_outputs_for_stats.max(axis=1) - base_outputs_for_stats.min(axis=1)
                
                # Calculate normalized spread with protection against division by zero
                mean_abs = base_outputs_for_stats.mean(axis=1).abs().replace(0, 1e-6)
                std_dev = base_outputs_for_stats.std(axis=1)
                meta_features['ens_norm_pred_spread'] = (std_dev / mean_abs).fillna(0) # fillna if std is 0 and mean is 0
                
                meta_features['ens_quartile_spread_iqr'] = base_outputs_for_stats.quantile(0.75, axis=1) - base_outputs_for_stats.quantile(0.25, axis=1)
                meta_features['ens_skewness'] = base_outputs_for_stats.skew(axis=1).fillna(0) # Skew is NaN if variance is 0
                meta_features['ens_kurtosis'] = base_outputs_for_stats.kurt(axis=1).fillna(0) # Kurtosis is NaN if variance is 0
                
                # Add these new features to the list
                additional_features_list.append(meta_features)
                tprint_success(f"✅ Added {len(meta_features.columns)} statistical meta-features for ensemble.")
            
            if additional_features_list:
                # Concatenate all features (base outputs + meta-features)
                return pd.concat(additional_features_list, axis=1)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error retrieving additional model outputs: {e}")
            return None
    # --- END MODIFICATION ---

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
