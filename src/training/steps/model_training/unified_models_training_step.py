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

# --- ADDED IMPORTS ---
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    ParameterGroup,
    OptimizationStage
)
# --- END ADDED IMPORTS ---

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
                    tprint_info("🔍 Performing hyperparameter optimization before training...")
                    
                    # Get the appropriate model config
                    if training_type.startswith('analyst'):
                        model_config_key = 'analyst_config'
                    elif training_type.startswith('tactician'):
                        model_config_key = 'tactician_config'
                    else:
                        model_config_key = 'ensemble_config'
                    
                    if model_config_key in yaml_config:
                        yaml_config[model_config_key] = await self._perform_hyperparameter_optimization(
                            training_data, hpo_targets, yaml_config[model_config_key], config
                        )
                    else:
                        tprint_warning(f"No {model_config_key} found in config, skipping HPO")
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
        
        return yaml_config
    
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
    
    # --- Define the LGBM Parameter Groups ---
    
    # 1st Layer: Core structure and learning rate
    lgbm_group_1 = ParameterGroup(
        name="structure_learning_rate",
        params={
            "max_depth": {"type": "int", "low": 3, "high": 6},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True}
        },
        priority=1,
        description="Optimize core structure (max_depth) and learning_rate first."
    )
    
    # 2nd Layer: Regularization and subsampling, dependent on Layer 1
    lgbm_group_2 = ParameterGroup(
        name="regularization_subsampling",
        params={
            # Per your guideline: num_leaves ≈ 2^max_depth ± 2
            # Since max_depth range is [3, 6], 2^max_depth is [8, 64].
            # This static range [6, 66] covers all possibilities.
            "num_leaves": {"type": "int", "low": 6, "high": 66},
            
            "reg_alpha": {"type": "float", "low": 0.1, "high": 5.0},
            "reg_lambda": {"type": "float", "low": 0.1, "high": 5.0},
            
            "subsample": {"type": "float", "low": 0.8, "high": 0.9},
            "colsample_bytree": {"type": "float", "low": 0.8, "high": 0.9},
            
            "min_child_samples": {"type": "int", "low": 20, "high": 50},
        },
        priority=2,
        depends_on=["structure_learning_rate"], # Ensures this group runs second
        description="Optimize regularization and subsampling parameters."
    )
    
    # List of all groups to pass to the optimizer
    lgbm_parameter_groups: List[ParameterGroup] = [lgbm_group_1, lgbm_group_2]

    # --- MODIFIED HPO METHOD ---
    async def _perform_hyperparameter_optimization(
        self,
        training_data: pd.DataFrame,
        targets: pd.Series,
        model_config: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform hierarchical hyperparameter optimization for model parameters.
        
        Args:
            training_data: Training data
            targets: Target variables
            model_config: Model configuration dictionary
            config: General configuration dictionary
            
        Returns:
            Optimized hyperparameters dictionary
        """
        try:
            # Check if HPO is enabled
            enable_hpo = config.get('enable_hpo', True)
            if not enable_hpo:
                tprint_info("Hyperparameter optimization disabled, using default parameters")
                return model_config
            
            tprint_info("🔍 Starting Hierarchical Hyperparameter Optimization (LGBM)...")
            
            # 1. Split data for HPO validation (80/20 split for HPO)
            hpo_train_size = int(len(training_data) * 0.8)
            hpo_train_data = training_data.iloc[:hpo_train_size]
            hpo_val_data = training_data.iloc[hpo_train_size:]
            hpo_train_targets = targets.iloc[:hpo_train_size]
            hpo_val_targets = targets.iloc[hpo_train_size:]
            
            tprint_info(f"Running HPO with {len(hpo_train_data)} train samples and {len(hpo_val_data)} validation samples")

            # 2. Define the synchronous objective function
            # This is defined *inside* the method to close over the data variables
            def lgbm_objective_function(params: Dict[str, Any], **kwargs) -> float:
                """Synchronous objective function for HPO."""
                try:
                    # Get data from closure
                    X_train_hpo = kwargs.get('X_train')
                    y_train_hpo = kwargs.get('y_train')
                    X_val_hpo = kwargs.get('X_val')
                    y_val_hpo = kwargs.get('y_val')

                    # Add fixed params for LGBM (assuming regression as per dummy data)
                    fixed_params = {
                        'objective': 'regression_l1', # MAE is robust to outliers
                        'metric': 'l1',
                        'n_estimators': 1000, # High number, will use early stopping
                        'verbose': -1,
                        'n_jobs': -1,
                    }
                    
                    # Combine trial params with fixed params
                    model_params = {**fixed_params, **params}

                    # --- Handle potential int/float type mismatches from optimizer ---
                    for int_param in ['max_depth', 'num_leaves', 'min_child_samples']:
                        if int_param in model_params:
                            model_params[int_param] = int(model_params[int_param])
                    # ---

                    model = lgb.LGBMRegressor(**model_params)
                    
                    model.fit(
                        X_train_hpo, y_train_hpo,
                        eval_set=[(X_val_hpo, y_val_hpo)],
                        eval_metric='l1',
                        callbacks=[lgb.early_stopping(patience=50, verbose=False)]
                    )
                    
                    preds = model.predict(X_val_hpo)
                    score = mean_squared_error(y_val_hpo, preds) # Minimize MSE
                    
                    return score

                except Exception as e:
                    self.logger.warning(f"HPO trial failed: {e}")
                    # Return a very high score to penalize this trial
                    return float('inf')

            # 3. Initialize the Hierarchical Optimizer
            hpo_param_groups = self.lgbm_parameter_groups
            
            execution_mode = config.get('execution_mode', 'full')
            if execution_mode == 'light':
                tprint_info("HPO Light Mode: Using Coarse Grid only")
                hpo_stages = [OptimizationStage.COARSE_GRID]
                n_rounds = 1
                final_refinement = False
            else:
                tprint_info("HPO Full Mode: Using Coarse Grid -> TPE")
                hpo_stages = [
                    OptimizationStage.COARSE_GRID,
                    OptimizationStage.TPE
                ]
                n_rounds = 1
                final_refinement = True
                
            optimizer = HierarchicalParameterOptimizer(
                param_groups=hpo_param_groups,
                objective_func=lgbm_objective_function,
                stages=hpo_stages,
                direction='minimize', # We are minimizing mean_squared_error
                n_rounds=n_rounds,
                enable_final_refinement=final_refinement,
                verbose=True # Will use the logger
            )

            # 4. Run the optimization in a separate thread
            tprint_info("🚀 Starting hierarchical HPO for LGBM stacker...")
            
            # Run the synchronous, CPU-bound HPO in a separate thread
            # to avoid blocking the asyncio event loop.
            opt_result = await asyncio.to_thread(
                optimizer.optimize,
                X_train=hpo_train_data,
                y_train=hpo_train_targets,
                X_val=hpo_val_data,
                y_val=hpo_val_targets,
                model=None # Objective func creates the model
            )
            
            best_hyperparams = opt_result.best_params
            tprint_success(f"✅ Hierarchical HPO complete. Best score: {opt_result.best_score:.6f}")

            # 5. Update the model_config
            # This ensures the HPO parameters are passed to the ensemble config
            if 'params' not in model_config:
                model_config['params'] = {}
            
            model_config['params'].update(best_hyperparams)
            tprint_info(f"Updated model config with {len(best_hyperparams)} optimized parameters.")
            
            # Also, update any 'base_models' that are LGBM (if they exist in the config)
            if 'base_models' in model_config:
                for model_name, model_params in model_config['base_models'].items():
                    if 'lgbm' in model_params.get('model_type', '').lower():
                        if 'params' not in model_params:
                            model_params['params'] = {}
                        # Update with keys that are relevant
                        relevant_params = {k: v for k, v in best_hyperparams.items() if k in model_params['params']}
                        model_params['params'].update(relevant_params)
                        tprint_info(f"Updated base_model '{model_name}' with relevant HPO params.")

            return model_config
            
        except Exception as e:
            tprint_error(f"Hyperparameter optimization failed: {e}")
            import traceback
            self.logger.error(f"HPO error: {e}\n{traceback.format_exc()}")
            return model_config # Return original config on failure
    
    def _get_hpo_search_space(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get hyperparameter search space based on model configuration.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Search space dictionary for HPO
        """
        # This method is now superseded by the hierarchical optimizer,
        # but we keep it for any legacy logic that might still call it.
        # The new HPO logic in _perform_hyperparameter_optimization
        # does NOT use this method.
        
        search_space = {}
        
        # Define search spaces for different model types
        if 'base_models' in model_config:
            for model_name, model_params in model_config['base_models'].items():
                model_type = model_params.get('model_type', '').lower()
                
                if 'lgbm' in model_type.lower():
                    # Use the hierarchical definitions as a default
                    search_space[model_name] = {
                        **self.lgbm_group_1.params,
                        **self.lgbm_group_2.params
                    }
                
                elif 'catboost' in model_type.lower():
                    search_space[model_name] = {
                        'iterations': [1000, 1500, 2000],
                        'learning_rate': [0.03, 0.05, 0.08],
                        'depth': [6, 8, 10],
                        'l2_leaf_reg': [1.0, 3.0, 5.0]
                    }
                
                elif 'tcn' in model_type.lower() or 'temporal' in model_type.lower():
                    search_space[model_name] = {
                        'hidden_size': [32, 64, 128],
                        'num_layers': [2, 3, 4],
                        'kernel_size': [2, 3, 4],
                        'dropout': [0.1, 0.2, 0.3],
                        'learning_rate': [0.0001, 0.001, 0.01]
                    }
                
                elif 'gru' in model_type.lower() or 'lstm' in model_type.lower():
                    search_space[model_name] = {
                        'hidden_units': [32, 64, 128],
                        'num_layers': [1, 2, 3],
                        'dropout': [0.1, 0.2, 0.3],
                        'learning_rate': [0.0001, 0.001, 0.01]
                    }
        
        return search_space
    
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
                    for model_name, model_params in yaml_config[config_key]['base_models'].items():
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

    async def _get_primary_features(self, config: Dict[str, Any]) -> pd.DataFrame:
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

    async def _get_regime_features(self, config: Dict[str, Any]) -> pd.DataFrame:
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
    async def _get_additional_model_outputs(self, training_type: str, config: Dict[str, Any]) -> pd.DataFrame:
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
                artifacts[f"{training_type}_metrics}"] = metrics_path
            
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
            artifacts[f"{training_type}_config}"] = config_path
            
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
