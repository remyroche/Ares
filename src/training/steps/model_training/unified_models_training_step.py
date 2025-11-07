"""
Unified Models Training Step.

This step consolidates all analyst and tactician training (base and ensemble)
into a single unified script that calls UnifiedTrainingPipeline.
"""

import asyncio
import yaml
import os
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from datetime import datetime

# HPO imports (only those actually used)
import lightgbm as lgb
from src.training.steps.model_training.hpo_config import (
    HPOOrchestrator,
    ModelParameterGroups
)

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning

# Import dynamic config calculator
from src.training.steps.model_training.dynamic_config_calculator import (
    DynamicConfigCalculator, DynamicTrainingConfig
)

# Import temporal splitting for proper train/val/test separation
from src.utils.versioned_artifacts import (
    create_temporal_split_config_for_pipeline,
    get_data_for_purpose,
    TemporalSplitConfig
)

# Try to import unified training pipeline if it exists, otherwise use placeholder
try:
    from src.training.steps.models_training.unified_training_pipeline import UnifiedTrainingPipeline
    unified_pipeline_available = True
except ImportError:
    unified_pipeline_available = False
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
            if not unified_pipeline_available:
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
            training_data, analyst_targets, tactician_targets = await self._retrieve_training_data(config, yaml_config)

            # ========================================================================
            # TEMPORAL SPLITTING: Enforce train/val/test boundaries
            # ========================================================================
            tprint_info("=" * 80)
            tprint_info("🔐 TEMPORAL DATA SPLITTING - Preventing Data Leakage")
            tprint_info("=" * 80)

            # Store full datasets before filtering (needed for validation period in HPO)
            self._full_training_data = training_data.copy() if training_data is not None else None
            self._full_analyst_targets = analyst_targets.copy() if analyst_targets is not None else None
            self._full_tactician_targets = tactician_targets.copy() if tactician_targets is not None else None

            if training_data is not None and len(training_data) > 0:
                # Create or load temporal split configuration
                tprint_info(f"📅 Creating temporal split configuration for {symbol} {exchange}")

                # Determine data boundaries from actual data
                data_start = training_data.index.min()
                data_end = training_data.index.max()

                tprint_info(f"   Data range: {data_start} to {data_end}")
                tprint_info(f"   Total samples: {len(training_data)}")

                # Create temporal split config
                temporal_config = create_temporal_split_config_for_pipeline(
                    symbol=symbol,
                    exchange=config.get('exchange', 'binance'),
                    timeframe=timeframe,
                    data_start=data_start,
                    data_end=data_end,
                    train_pct=config.get('train_percentage', 0.60),  # 60% for training
                    val_pct=config.get('validation_percentage', 0.20),  # 20% for validation (HPO)
                    test_pct=config.get('test_percentage', 0.20),  # 20% for test
                    embargo_days=1  # 1-day embargo between periods
                )

                # Store temporal config for later use
                self._temporal_config = temporal_config
                config['temporal_config'] = temporal_config

                # Log temporal split boundaries
                tprint_info("📊 Temporal Split Configuration:")
                tprint_info(f"   Training:   {temporal_config.training.start} → {temporal_config.training.end} "
                          f"({len(training_data.loc[temporal_config.training.start:temporal_config.training.effective_end])} samples)")
                tprint_info(f"   Validation: {temporal_config.validation.start} → {temporal_config.validation.end} "
                          f"({len(training_data.loc[temporal_config.validation.start:temporal_config.validation.effective_end])} samples)")
                tprint_info(f"   Test:       {temporal_config.test.start} → {temporal_config.test.end} "
                          f"({len(training_data.loc[temporal_config.test.start:temporal_config.test.end])} samples)")
                tprint_info(f"   Embargo:    {temporal_config.embargo_days} day(s)")

                # Filter training data to TRAINING PERIOD ONLY
                tprint_info("🔒 Filtering data to TRAINING period (preventing future data leakage)...")
                original_len = len(training_data)

                # Use temporal filtering
                training_data = get_data_for_purpose(
                    training_data,
                    purpose='training',
                    config=temporal_config
                )

                filtered_len = len(training_data)
                tprint_success(f"✅ Filtered to training period: {original_len} → {filtered_len} samples "
                             f"({filtered_len/original_len*100:.1f}% of full dataset)")

                # Filter targets to match training period
                if analyst_targets is not None:
                    analyst_targets = analyst_targets.loc[training_data.index]
                    tprint_info(f"   ↪ Analyst targets filtered to {len(analyst_targets)} samples")

                if tactician_targets is not None:
                    tactician_targets = tactician_targets.loc[training_data.index]
                    tprint_info(f"   ↪ Tactician targets filtered to {len(tactician_targets)} samples")

                tprint_info("=" * 80)
            else:
                tprint_warning("⚠️ No training data available for temporal splitting")
                self._temporal_config = None

            # --- MODIFIED: Retrieve and merge additional features for ensemble/tactician models ---
            if training_type.endswith('ensemble') or training_type == 'tactician_base':
                tprint_info(f"Retrieving additional model outputs for {training_type}...")
                # --- FIX 5: Pass training_data for index alignment ---
                additional_outputs = await self._get_additional_model_outputs(training_type, config, training_data)

                if additional_outputs is not None:
                    # Align indices before concatenating
                    # This alignment is still necessary AFTER resampling, just in case.
                    tprint_info(
                        "🔄 Aligning primary training data with additional outputs before concatenation"
                    )
                    tprint_info(
                        f"   ↪ training_data shape={training_data.shape}, columns={len(training_data.columns)}"
                    )
                    tprint_info(
                        f"   ↪ additional_outputs shape={additional_outputs.shape}, columns={len(additional_outputs.columns)}"
                    )
                    aligned_training_data, aligned_additional_outputs = training_data.align(additional_outputs, join='inner', axis=0)                    
                    tprint_info(
                        f"   ↪ aligned_training_data shape={aligned_training_data.shape}, aligned_additional_outputs shape={aligned_additional_outputs.shape}"
                    )
                    if aligned_training_data.empty:
                        tprint_warning("Data alignment resulted in empty DataFrame. Check for index mismatches.")
                        # Fallback to original data if alignment fails
                    else:
                        merged_columns = len(aligned_training_data.columns) + len(aligned_additional_outputs.columns)
                        tprint_info(
                            f"   ↪ Concatenating columns -> expected merged column count ≈ {merged_columns}"
                        )
                        # Use safe concatenation with temporal alignment validation
                        training_data = self._safe_concat(
                            [aligned_training_data, aligned_additional_outputs],
                            axis=1,
                            operation_name="merge_training_and_additional_features",
                            validate_alignment=True
                        )
                        tprint_success(f"✅ Merged additional features. New training data shape: {training_data.shape}")
                else:
                    tprint_warning(f"No additional model outputs found for {training_type}. Proceeding with primary features only.")
            # --- END MODIFICATION ---

            # Apply light mode filtering if needed
            training_data = self._apply_light_mode_filter(training_data, config, timeframe)
            
            # Align targets to match filtered training data
            if training_data is not None and analyst_targets is not None:
                if len(analyst_targets) != len(training_data):
                    tprint_warning(f"⚠️ Aligning analyst targets from {len(analyst_targets)} to {len(training_data)} samples")
                    analyst_targets = analyst_targets.loc[training_data.index]
            if training_data is not None and tactician_targets is not None:
                if len(tactician_targets) != len(training_data):
                    tprint_warning(f"⚠️ Aligning tactician targets from {len(tactician_targets)} to {len(training_data)} samples")
                    tactician_targets = tactician_targets.loc[training_data.index]
            
            # Calculate COMPREHENSIVE dynamic configuration based on data and hardware
            if training_data is not None:
                non_feature_cols = [col for col in training_data.columns if 'timestamp' in col.lower() or 'datetime' in col.lower()]
                if non_feature_cols:
                    tprint_info(f"Dropping non-feature timestamp columns: {non_feature_cols}")
                    training_data = training_data.drop(columns=non_feature_cols)   
                    
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
            
            # Optimize DepthwiseCNN (replaces TCN)
            if 'depthwise_cnn' in base_models:
                depthwise_config = base_models['depthwise_cnn']
                tprint_warning(f"⚡ Applying {execution_mode.upper()} mode DepthwiseCNN optimizations (10x lighter)")

                # Drastically reduce DepthwiseCNN parameters for light mode
                depthwise_params = depthwise_config.get('params', {})
                depthwise_params['filters'] = 32  # Reduced from 64
                depthwise_params['epochs'] = 10  # Reduced from 50 (10x lighter)
                depthwise_params['batch_size'] = 128  # Increased from 64 (fewer iterations)
                depthwise_params['early_stopping_patience'] = 3  # Reduced from 7

                # Disable DepthwiseCNN HPO in light mode
                if 'hpo' in depthwise_config:
                    depthwise_config['hpo']['enabled'] = False

                tprint_info(f"  DepthwiseCNN epochs: 50 → 10 (10x lighter)")
                tprint_info(f"  DepthwiseCNN HPO: DISABLED")
            
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

            # ========================================================================
            # OPTION A: Use VALIDATION PERIOD for HPO (not 80/20 split)
            # ========================================================================
            tprint_info("=" * 80)
            tprint_info("🔐 HPO DATA SPLITTING - Using Validation Period")
            tprint_info("=" * 80)

            # Get temporal config
            temporal_config = getattr(self, '_temporal_config', None)

            if temporal_config is not None and hasattr(self, '_full_training_data'):
                # Use validation period for HPO
                tprint_info("✅ Using temporal split: Training period for training, Validation period for HPO")

                # Training data is already filtered to training period
                X_train = training_data
                y_train = targets

                # Get validation period data from full dataset
                tprint_info("📅 Retrieving validation period data for HPO...")
                validation_data = get_data_for_purpose(
                    self._full_training_data,
                    purpose='validation',
                    config=temporal_config
                )

                # Get validation targets
                if training_type.startswith('analyst') and hasattr(self, '_full_analyst_targets'):
                    validation_targets = self._full_analyst_targets.loc[validation_data.index]
                elif training_type.startswith('tactician') and hasattr(self, '_full_tactician_targets'):
                    validation_targets = self._full_tactician_targets.loc[validation_data.index]
                else:
                    validation_targets = None

                if validation_targets is not None:
                    X_val = validation_data
                    y_val = validation_targets

                    tprint_success(f"✅ HPO data split:")
                    tprint_info(f"   Training:   {len(X_train)} samples ({temporal_config.training.start} → {temporal_config.training.effective_end})")
                    tprint_info(f"   Validation: {len(X_val)} samples ({temporal_config.validation.start} → {temporal_config.validation.effective_end})")
                    tprint_info(f"   🔒 No data leakage: Validation period completely separate from training")
                else:
                    tprint_warning("⚠️ Validation targets not found, falling back to 80/20 split within training period")
                    hpo_train_size = int(len(training_data) * 0.8)
                    X_train = training_data.iloc[:hpo_train_size]
                    X_val = training_data.iloc[hpo_train_size:]
                    y_train = targets.iloc[:hpo_train_size]
                    y_val = targets.iloc[hpo_train_size:]
                    tprint_info(f"   Training: {len(X_train)} samples (80%)")
                    tprint_info(f"   Validation: {len(X_val)} samples (20%)")
            else:
                # Fallback to 80/20 split if temporal config not available
                tprint_warning("⚠️ Temporal config not available, using 80/20 split within training data")
                hpo_train_size = int(len(training_data) * 0.8)
                X_train = training_data.iloc[:hpo_train_size]
                X_val = training_data.iloc[hpo_train_size:]
                y_train = targets.iloc[:hpo_train_size]
                y_val = targets.iloc[hpo_train_size:]
                tprint_info(f"   Training: {len(X_train)} samples (80%)")
                tprint_info(f"   Validation: {len(X_val)} samples (20%)")

            tprint_info("=" * 80)
            
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
    
    def _apply_light_mode_nn_optimizations(self, yaml_config: Dict[str, Any]) -> None:
        """Apply aggressive model optimizations for light mode execution (10x lighter)."""
        execution_mode = yaml_config.get('execution_mode', 'light')
        
        # Check if analyst_config exists
        if 'analyst_config' in yaml_config:
            base_models = yaml_config['analyst_config'].get('base_models', {})
            
            # Optimize DepthwiseCNN (replaces TCN)
            if 'depthwise_cnn' in base_models:
                depthwise_config = base_models['depthwise_cnn']
                tprint_warning(f"⚡ Applying {execution_mode.upper()} mode DepthwiseCNN optimizations (10x lighter)")

                # Drastically reduce DepthwiseCNN parameters for light mode
                depthwise_params = depthwise_config.get('params', {})
                depthwise_params['filters'] = 32  # Reduced from 64
                depthwise_params['epochs'] = 10  # Reduced from 50 (10x lighter)
                depthwise_params['batch_size'] = 128  # Increased from 64 (fewer iterations)
                depthwise_params['early_stopping_patience'] = 3  # Reduced from 7

                # Disable DepthwiseCNN HPO in light mode
                if 'hpo' in depthwise_config:
                    depthwise_config['hpo']['enabled'] = False

                tprint_info(f"  DepthwiseCNN epochs: 50 → 10 (10x lighter)")
                tprint_info(f"  DepthwiseCNN HPO: DISABLED")
            
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
        
        # Check if tactician_config has GRU or DepthwiseCNN models
        if 'tactician_config' in yaml_config:
            base_models = yaml_config['tactician_config'].get('base_models', [])
            for model in base_models:
                model_name = model.get('model_name', 'Unknown')
                
                if model_name == 'StandaloneGRU':
                    tprint_warning(f"⚡ Applying {execution_mode.upper()} mode GRU optimizations (10x lighter)")
                    params = model.get('params', {})
                    if 'training_params' in params:
                        params['training_params']['epochs'] = 10  # Reduce epochs
                        params['training_params']['batch_size'] = 128 # Increase batch size
                    else:
                        params['epochs'] = 10
                        params['batch_size'] = 128
                        
                    if 'hpo' in model:
                        model['hpo']['enabled'] = False
                    tprint_info(f"  GRU epochs: Reduced to 10")
                    tprint_info(f"  GRU HPO: DISABLED")


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
                            # Use model_name for matching, as model_type is for HPO
                            model_type_key = model_name.lower() # Use model_name for key
                            
                            # Neural network models
                            if any(nn in model_type_key for nn in ['gru', 'lstm', 'tcn', 'transformer', 'depthwisecnn', 'cnn']):
                                nn_params_to_update = {
                                    'batch_size': dynamic_config.batch_size,
                                    'epochs': dynamic_config.epochs if dynamic_config.epochs > 0 else 100,
                                    'learning_rate': dynamic_config.learning_rate,
                                    'early_stopping_patience': dynamic_config.early_stopping_patience
                                }
                                
                                # Handle GRU's nested training_params
                                if 'gru' in model_type_key and 'training_params' in model_params['params']:
                                     model_params['params']['training_params'].update(nn_params_to_update)
                                else:
                                     model_params['params'].update(nn_params_to_update)
                                
                                # Add sequence length for time series models
                                if any(ts in model_type_key for ts in ['gru', 'lstm', 'tcn', 'depthwisecnn', 'cnn']):
                                    model_params['params']['sequence_length'] = dynamic_config.sequence_length
                            
                            # Tree-based models
                            elif any(tree in model_type_key for tree in ['lgbm', 'catboost', 'xgboost', 'extratrees']):
                                if 'lgbm' in model_type_key:
                                    model_params['params']['n_estimators'] = dynamic_config.n_estimators
                                elif 'catboost' in model_type_key:
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
                            model_type_key = model_name.lower() # Use model_name for key
                            
                            # Neural network models
                            if any(nn in model_type_key for nn in ['gru', 'lstm', 'tcn', 'transformer', 'depthwisecnn', 'cnn']):
                                model_params['params'].update({
                                    'batch_size': dynamic_config.batch_size,
                                    'epochs': dynamic_config.epochs if dynamic_config.epochs > 0 else 100,
                                    'learning_rate': dynamic_config.learning_rate,
                                    'early_stopping_patience': dynamic_config.early_stopping_patience
                                })
                                
                                # Add sequence length for time series models
                                if any(ts in model_type_key for ts in ['gru', 'lstm', 'tcn', 'depthwisecnn', 'cnn']):
                                    model_params['params']['sequence_length'] = dynamic_config.sequence_length
                            
                            # Tree-based models
                            elif any(tree in model_type_key for tree in ['lgbm', 'catboost', 'xgboost', 'extratrees']):
                                if 'lgbm' in model_type_key:
                                    model_params['params']['n_estimators'] = dynamic_config.n_estimators
                                elif 'catboost' in model_type_key:
                                    model_params['params']['iterations'] = dynamic_config.iterations
                                    tprint_info("Applying CatBoost GPU (Apple M1) configuration...")
                                    model_params['params']['task_type'] = 'GPU'
                                    model_params['params']['devices'] = '0' # Use '0' for Apple M1 GPU
                                    
                                    # Remove subsample if it exists, as it's not supported by CatBoost GPU
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
    
    async def _retrieve_training_data(self, config: Dict[str, Any], yaml_config: Dict[str, Any]) -> tuple:
        """Retrieve training data and targets from artifacts with fast-fail on missing data."""
        try:
            tprint_info("🔍 Retrieving training data from feature generation artifacts...")
            
            training_data = None
            analyst_targets = None
            tactician_targets = None
            
            # Determine feature set size to use (default to 50 features)
            feature_set_size = config.get('feature_set_size', 50)

            # Try to get selected features from feature_generation_final_feature_selection_step
            # IMPORTANT: Fallback order prioritizes larger feature sets (60 > 50 > 40) for better model performance
            feature_artifact_names = [
                f'selected_feature_dataframe_{feature_set_size}',  # Specific size
                f'selected_features_{feature_set_size}',           # Alternative name
                f'final_dataset_{feature_set_size}',               # Validation step generic alias
                f'final_analyst_dataset_{feature_set_size}',       # Analyst-specific validation alias
                'selected_feature_dataframe_60',                   # Fallback to 60 (try largest first)
                'selected_feature_dataframe_50',                   # Fallback to 50
                'selected_feature_dataframe_40',                   # Fallback to 40
                'final_dataset_60',                                # Validation step 60
                'final_dataset_50',                                # Validation step 50
                'final_dataset_40',                                # Validation step 40
            ]

            tprint_info(f"🔎 Attempting to load training features from artifacts: {feature_artifact_names}")
            feature_source_name = None

            for artifact_name in feature_artifact_names:
                try:
                    tprint_info(f"   ↪ Trying '{artifact_name}'")
                    training_data = self._get_artifact(artifact_name, 'data')
                    if training_data is not None:
                        feature_source_name = artifact_name
                        tprint_success(f"✅ Retrieved training features from '{artifact_name}': {training_data.shape if hasattr(training_data, 'shape') else type(training_data)}")
                        break
                    else:
                        tprint_warning(f"⚠️ Artifact '{artifact_name}' returned None (metadata exists but data file missing?)")
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
                        tprint_info(f"   ↪ Trying alternative '{artifact_name}'")
                        training_data = self._get_artifact(artifact_name, 'data')
                        if training_data is not None:
                            feature_source_name = artifact_name
                            tprint_success(f"✅ Retrieved training data from '{artifact_name}': {training_data.shape if hasattr(training_data, 'shape') else type(training_data)}")
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
            
            # Normalize training_data to DataFrame for downstream processing
            if training_data is not None and not isinstance(training_data, pd.DataFrame):
                try:
                    training_data = pd.DataFrame(training_data)
                    tprint_warning(f"⚠️ Converted training data from type '{type(training_data)}' to DataFrame")
                except Exception as e:
                    tprint_error(f"❌ Failed to convert training data to DataFrame: {e}")
                    raise

            if training_data is not None and isinstance(training_data, pd.DataFrame):
                # Comprehensive feature loading verification and logging
                tprint_info("=" * 80)
                tprint_info("📊 COMPREHENSIVE FEATURE LOADING VERIFICATION")
                tprint_info("=" * 80)
                tprint_success(f"✅ HDF5 Access Verified: Successfully loaded from '{feature_source_name or 'unknown_source'}'")
                tprint_info(f"📦 Feature DataFrame Shape: {training_data.shape} ({training_data.shape[0]:,} samples × {training_data.shape[1]:,} features)")
                tprint_info(f"💾 Memory Usage: {training_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                tprint_info(f"🔢 Data Types Distribution: {dict(training_data.dtypes.value_counts())}")
                tprint_info(f"📋 Column Names (first 20): {list(training_data.columns[:20])}")
                if len(training_data.columns) > 20:
                    tprint_info(f"    ... and {len(training_data.columns) - 20} more columns")

                # Check for missing data
                null_counts = training_data.isnull().sum()
                cols_with_nulls = null_counts[null_counts > 0]
                if len(cols_with_nulls) > 0:
                    tprint_warning(f"⚠️ Found {len(cols_with_nulls)} columns with missing values")
                    tprint_info(f"   Top 10 columns with most nulls:")
                    for col, count in cols_with_nulls.nlargest(10).items():
                        pct = (count / len(training_data)) * 100
                        tprint_info(f"      - {col}: {count:,} ({pct:.1f}%)")
                else:
                    tprint_success("✅ No missing values detected")

                # Check for constant/zero-variance columns
                numeric_cols = training_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    std_devs = training_data[numeric_cols].std()
                    zero_var_cols = std_devs[std_devs == 0].index.tolist()
                    if len(zero_var_cols) > 0:
                        tprint_warning(f"⚠️ Found {len(zero_var_cols)} zero-variance columns (will be removed during cleaning)")
                    else:
                        tprint_success("✅ All numeric columns have variance")

                tprint_info("=" * 80)

                self._log_feature_snapshot(training_data, feature_source_name or 'unknown_source', prefix='📥 Raw load ')
                tprint_info(
                    f"🧪 Raw feature frame -> shape={training_data.shape}, columns={len(training_data.columns)}, "
                    f"dtypes={training_data.dtypes.value_counts().to_dict()}"
                )

                # Drop duplicate columns to avoid shape mismatches later
                if training_data.columns.duplicated().any():
                    duplicate_cols = training_data.columns[training_data.columns.duplicated()].unique().tolist()
                    tprint_warning(f"🧹 Dropping duplicate columns ({len(duplicate_cols)}): {duplicate_cols}")
                    training_data = training_data.loc[:, ~training_data.columns.duplicated()].copy()

                # Drop columns that are entirely NaN or have insufficient valid data
                min_valid_threshold = 0.01  # Require at least 1% valid data
                empty_cols = []
                for col in training_data.columns:
                    valid_count = training_data[col].notna().sum()
                    valid_ratio = valid_count / len(training_data)
                    if valid_ratio < min_valid_threshold:
                        empty_cols.append(col)

                if empty_cols:
                    tprint_warning(f"⚠️ Dropping {len(empty_cols)} columns with insufficient valid data (<{min_valid_threshold*100}%): {empty_cols[:10]}{'...' if len(empty_cols) > 10 else ''}")
                    training_data = training_data.drop(columns=empty_cols)
                else:
                    tprint_info("✅ All columns have sufficient valid data")

                # Remove non-numeric columns (they break model training/HPO)
                non_numeric_cols = training_data.select_dtypes(exclude=[np.number, 'bool']).columns.tolist()
                if non_numeric_cols:
                    preview = non_numeric_cols[:10]
                    suffix = '...' if len(non_numeric_cols) > 10 else ''
                    tprint_warning(f"⚠️ Dropping {len(non_numeric_cols)} non-numeric columns: {preview}{suffix}")
                    training_data = training_data.drop(columns=non_numeric_cols)
                else:
                    tprint_info("✅ No non-numeric columns detected during cleaning")

                # Convert boolean columns to numeric floats for model compatibility
                bool_cols = training_data.select_dtypes(include=['bool']).columns.tolist()
                if bool_cols:
                    training_data[bool_cols] = training_data[bool_cols].astype(np.float32)
                    tprint_info(f"ℹ️ Converted boolean columns to float: {bool_cols}")
                else:
                    tprint_info("✅ No boolean columns required conversion")

                # Remove obvious target columns that might have slipped into the feature frame
                potential_target_cols = [
                    col for col in training_data.columns
                    if col.lower() in {'target', 'label'}
                    or col.lower().endswith('_target')
                    or col.lower().endswith('_label')
                ]
                if potential_target_cols:
                    tprint_warning(f"⚠️ Dropping target-like columns from features: {potential_target_cols}")
                    training_data = training_data.drop(columns=potential_target_cols)
                else:
                    tprint_info("✅ No target-like columns detected in feature frame")

                # Remove metadata columns that are not features
                metadata_col_patterns = [
                    'labeling_method', 'labeling_timestamp', 'base_threshold',
                    'lookahead_periods', 'optimization_iteration', 'quality_acceptance_rate'
                ]
                metadata_cols_to_drop = [
                    col for col in training_data.columns
                    if any(pattern in col.lower() for pattern in metadata_col_patterns)
                ]
                if metadata_cols_to_drop:
                    tprint_warning(f"⚠️ Dropping metadata columns from features: {metadata_cols_to_drop}")
                    training_data = training_data.drop(columns=metadata_cols_to_drop)
                else:
                    tprint_info("✅ No metadata columns detected in feature frame")

                if training_data.empty:
                    raise ValueError("All feature columns were removed during cleaning; check upstream artifacts.")

                self._log_feature_snapshot(training_data, feature_source_name or 'unknown_source', prefix='🧹 Cleaned ')
                tprint_info(
                    f"🧼 Post-cleaning feature frame -> shape={training_data.shape}, columns={len(training_data.columns)}, "
                    f"dtypes={training_data.dtypes.value_counts().to_dict()}"
                )

            # Get targets from labeling integration step (direction-aware)
            direction = config.get('direction', 'long')
            tprint_info(f"📍 Loading labels for direction: {direction}")

            # Try to get analyst targets (direction-specific first, then generic)
            analyst_artifact_names = [
                f'analyst_targets_{direction}',  # Direction-specific (e.g., analyst_targets_long)
                f'{direction}_analyst_targets',  # Alternative naming
                f'{direction}_targets',           # Generic direction-specific
                'analyst_targets',                # Generic analyst targets
                'targets'                         # Fallback to generic targets
            ]

            for artifact_name in analyst_artifact_names:
                try:
                    analyst_targets = self._get_artifact(artifact_name, 'data')
                    if analyst_targets is not None:
                        tprint_success(f"✅ Retrieved analyst targets from '{artifact_name}': {len(analyst_targets)} samples")
                        break
                except Exception as e:
                    self.logger.debug(f"Artifact '{artifact_name}' not found: {e}")
                    continue

            # Try to get tactician targets (direction-specific first, then generic)
            tactician_artifact_names = [
                f'tactician_targets_{direction}',  # Direction-specific
                f'{direction}_tactician_targets',  # Alternative naming
                f'{direction}_targets',             # Generic direction-specific
                'tactician_targets',                # Generic tactician targets
                'targets'                           # Fallback to generic targets
            ]

            for artifact_name in tactician_artifact_names:
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

                            # Extract target columns (direction-aware)
                            target_cols = [col for col in labeled_data.columns if 'target' in col.lower()]
                            if target_cols:
                                tprint_info(f"📋 Available target columns: {target_cols}")

                                # Prioritize direction-specific target columns
                                direction_specific_cols = [
                                    col for col in target_cols
                                    if direction in col.lower()
                                ]

                                if direction_specific_cols:
                                    target_col = direction_specific_cols[0]
                                    tprint_success(f"✅ Using direction-specific target column: {target_col}")
                                else:
                                    target_col = target_cols[0]
                                    tprint_warning(f"⚠️ No direction-specific column found, using: {target_col}")

                                analyst_targets = labeled_data[target_col]
                                tprint_success(f"✅ Extracted analyst targets: {len(analyst_targets)} samples")
                                
                                # CRITICAL: Ensure training_data and targets are aligned
                                if len(training_data) != len(analyst_targets):
                                    tprint_warning(f"⚠️ Shape mismatch detected! Features: {len(training_data)}, Targets: {len(analyst_targets)}")

                                    # Try to align by common indices if both have indices
                                    if hasattr(training_data, 'index') and hasattr(labeled_data, 'index'):
                                        common_idx = training_data.index.intersection(labeled_data.index)

                                        if len(common_idx) > 0:
                                            tprint_info(f"✅ Found {len(common_idx)} common samples, aligning by index...")
                                            training_data = training_data.loc[common_idx]
                                            analyst_targets = labeled_data.loc[common_idx, target_cols[0]]
                                            tprint_success(f"✅ Aligned by index - Features: {training_data.shape}, Targets: {len(analyst_targets)} samples")
                                        else:
                                            tprint_warning(f"⚠️ No common indices found, using first N samples for alignment...")
                                            # Align by position: use the smaller dataset size
                                            min_len = min(len(training_data), len(analyst_targets))
                                            training_data = training_data.iloc[:min_len]
                                            analyst_targets = analyst_targets.iloc[:min_len]
                                            tprint_success(f"✅ Aligned by position - Features: {training_data.shape}, Targets: {len(analyst_targets)} samples")
                                    else:
                                        tprint_warning(f"⚠️ DataFrames lack indices, using first N samples for alignment...")
                                        min_len = min(len(training_data), len(analyst_targets))
                                        training_data = training_data.iloc[:min_len] if hasattr(training_data, 'iloc') else training_data[:min_len]
                                        analyst_targets = analyst_targets.iloc[:min_len] if hasattr(analyst_targets, 'iloc') else analyst_targets[:min_len]
                                        tprint_success(f"✅ Aligned by position - Features: {len(training_data)}, Targets: {len(analyst_targets)} samples")
                                
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

            tprint_info("Aligning features and targets...")
            if training_data is not None and analyst_targets is not None:
                common_index = training_data.index.intersection(analyst_targets.index)
                if len(common_index) == 0:
                    raise ValueError("Data alignment failed: No common index between features and analyst targets.")
                if len(common_index) < len(training_data) or len(common_index) < len(analyst_targets):
                    tprint_warning(f"Index mismatch: Aligning {len(training_data)} features and {len(analyst_targets)} analyst targets to {len(common_index)} common rows.")
                
                training_data = training_data.loc[common_index]
                analyst_targets = analyst_targets.loc[common_index]
                tprint_success(f"✅ Aligned features and analyst targets. New shape: {training_data.shape}")

            if training_data is not None and tactician_targets is not None:
                common_index = training_data.index.intersection(tactician_targets.index)
                if len(common_index) == 0:
                    raise ValueError("Data alignment failed: No common index between features and tactician targets.")
                if len(common_index) < len(training_data) or len(common_index) < len(tactician_targets):
                    tprint_warning(f"Index mismatch: Aligning {len(training_data)} features and {len(tactician_targets)} tactician targets to {len(common_index)} common rows.")
                
                training_data = training_data.loc[common_index]
                tactician_targets = tactician_targets.loc[common_index]
                if analyst_targets is not None: # Re-align analyst targets if tactician targets caused a change
                    analyst_targets = analyst_targets.loc[common_index]
                tprint_success(f"✅ Aligned features and tactician targets. New shape: {training_data.shape}")
                
            # Exclude raw OHLCV features from training data as requested
            if training_data is not None:
                excluded_ohlcv_features = []
                
                # Get OHLCV exclusion configuration from YAML config
                ohlcv_config = yaml_config.get('feature_engineering', {}).get('exclude_raw_ohlcv', {})
                if ohlcv_config.get('enabled', True):
                    ohlcv_patterns = ohlcv_config.get('excluded_patterns', ['volume', 'close', 'high', 'open', 'low'])
                    technical_terms = ohlcv_config.get('technical_indicators', ['rsi', 'sma', 'ema', 'bb_', 'macd', 'atr', 'roc', 'mom', 
                                                                             'return', 'pct', 'ratio', 'std', 'volatility', 'trend',
                                                                             'momentum', 'oscillator', 'signal', 'cross', 'divergence'])
                else:
                    # Fallback to hardcoded values if config is disabled
                    ohlcv_patterns = ['volume', 'close', 'high', 'open', 'low']
                    technical_terms = ['rsi', 'sma', 'ema', 'bb_', 'macd', 'atr', 'roc', 'mom', 
                                     'return', 'pct', 'ratio', 'std', 'volatility', 'trend',
                                     'momentum', 'oscillator', 'signal', 'cross', 'divergence']
                
                for col in training_data.columns:
                    col_lower = col.lower()
                    if any(pattern in col_lower for pattern in ohlcv_patterns):
                        # Only exclude if it's a raw OHLCV column, not derived features
                        # Raw OHLCV columns are typically named exactly with these patterns
                        # and don't contain additional technical indicator terms
                        is_raw_ohlcv = not any(term in col_lower for term in technical_terms)
                        
                        if is_raw_ohlcv:
                            excluded_ohlcv_features.append(col)
                
                if excluded_ohlcv_features:
                    tprint_warning(f"🚨 Excluding raw OHLCV features from training: {excluded_ohlcv_features}")
                    training_data = training_data.drop(columns=excluded_ohlcv_features)
                    tprint_success(f"✅ Removed {len(excluded_ohlcv_features)} raw OHLCV features. New shape: {training_data.shape}")
            
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

    async def _get_additional_model_outputs(self, training_type: str, config: Dict[str, Any], training_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get additional model outputs based on training type."""
        try:
            additional_features_list = []
            base_outputs_for_stats = None # Store the specific DataFrame to calculate stats on

            # --- START: Load and Resample Regime Features (FAST-FAIL) ---
            try:
                # Load regime ensemble predictions (from regime_ensemble_training) - REQUIRED
                regime_features = self._get_artifact('regime_ensemble_predictions', 'data')
                if regime_features is None:
                    error_msg = (
                        "❌ CRITICAL: regime_ensemble_predictions artifact not found!\n"
                        "   This artifact is REQUIRED for model training.\n"
                        "   Please ensure regime_ensemble_training step has run successfully."
                    )
                    tprint_error(error_msg)
                    raise ValueError(error_msg)

                tprint_info(f"   ↪ Retrieved regime_ensemble_predictions: shape={regime_features.shape}, columns={len(regime_features.columns)}")
                tprint_success(f"✅ Loaded regime ensemble predictions: {regime_features.shape}")

                # Resample regime features if needed to match training data (should already be 15m)
                if not regime_features.index.equals(training_data.index):
                    tprint_warning(f"Regime features index mismatch. Resampling {len(regime_features)} rows to match {len(training_data)} rows.")
                    # Use ffill and bfill for any alignment issues
                    regime_features_resampled = regime_features.reindex(training_data.index, method='ffill').fillna(method='bfill')
                    tprint_info(
                        f"   ↪ Resampled regime features -> shape={regime_features_resampled.shape}, columns={len(regime_features_resampled.columns)}"
                    )
                    additional_features_list.append(regime_features_resampled)
                    tprint_success("✅ Resampled and added regime ensemble features.")
                else:
                    tprint_info("   ↪ Regime features already aligned with training index")
                    additional_features_list.append(regime_features)
            except ValueError:
                # Re-raise ValueError for fast-fail
                raise
            except Exception as e:
                # Unexpected errors should also fast-fail
                error_msg = f"❌ CRITICAL: Failed to load regime_ensemble_predictions: {e}"
                tprint_error(error_msg)
                raise ValueError(error_msg) from e
            # --- END: Load and Resample Regime Features (FAST-FAIL) ---


            if training_type == 'analyst_ensemble':
                # Base models for analyst_ensemble are the analyst_base outputs
                base_outputs = self._get_artifact('analyst_base_outputs', 'data') # Changed to sync
                if base_outputs is not None:
                    # --- FIX 5: Resample/Reindex ---
                    if not base_outputs.index.equals(training_data.index):
                        tprint_warning(f"Aligning 'analyst_base_outputs' index to training data.")
                        base_outputs = base_outputs.reindex(training_data.index, method='ffill').fillna(method='bfill')
                        tprint_info(
                            f"   ↪ Resampled analyst_base_outputs -> shape={base_outputs.shape}, columns={len(base_outputs.columns)}"
                        )
                    additional_features_list.append(base_outputs)
                    base_outputs_for_stats = base_outputs # Calculate stats on these
                    tprint_info(
                        f"   ↪ Added analyst_base_outputs features, cumulative={sum(df.shape[1] for df in additional_features_list)} columns"
                    )

            elif training_type == 'tactician_base':
                # Base model for tactician_base is the analyst_ensemble output
                analyst_outputs = self._get_artifact('analyst_ensemble_outputs', 'data') # Changed to sync
                if analyst_outputs is not None:
                    # --- FIX 5: Resample/Reindex ---
                    if not analyst_outputs.index.equals(training_data.index):
                        tprint_warning(f"Aligning 'analyst_ensemble_outputs' index to training data.")
                        analyst_outputs = analyst_outputs.reindex(training_data.index, method='ffill').fillna(method='bfill')
                        tprint_info(
                            f"   ↪ Resampled analyst_ensemble_outputs -> shape={analyst_outputs.shape}, columns={len(analyst_outputs.columns)}"
                        )
                    additional_features_list.append(analyst_outputs)
                    tprint_info(
                        f"   ↪ Added analyst_ensemble_outputs features, cumulative={sum(df.shape[1] for df in additional_features_list)} columns"
                    )
                # No stats needed here, this is for a base model

            elif training_type == 'tactician_ensemble':
                # Base models for tactician_ensemble are the tactician_base outputs
                # Analyst_ensemble outputs are also included as features.
                analyst_outputs = self._get_artifact('analyst_ensemble_outputs', 'data') # Changed to sync
                tactician_base_outputs = self._get_artifact('tactician_base_outputs', 'data') # Changed to sync

                if analyst_outputs is not None:
                    # --- FIX 5: Resample/Reindex ---
                    if not analyst_outputs.index.equals(training_data.index):
                        tprint_warning(f"Aligning 'analyst_ensemble_outputs' index to training data.")
                        analyst_outputs = analyst_outputs.reindex(training_data.index, method='ffill').fillna(method='bfill')
                        tprint_info(
                            f"   ↪ Resampled analyst_ensemble_outputs -> shape={analyst_outputs.shape}, columns={len(analyst_outputs.columns)}"
                        )
                    additional_features_list.append(analyst_outputs)
                    tprint_info(
                        f"   ↪ Added analyst_ensemble_outputs features, cumulative={sum(df.shape[1] for df in additional_features_list)} columns"
                    )

                if tactician_base_outputs is not None:
                    # --- FIX 5: Resample/Reindex ---
                    if not tactician_base_outputs.index.equals(training_data.index):
                        tprint_warning(f"Aligning 'tactician_base_outputs' index to training data.")
                        tactician_base_outputs = tactician_base_outputs.reindex(training_data.index, method='ffill').fillna(method='bfill')
                        tprint_info(
                            f"   ↪ Resampled tactician_base_outputs -> shape={tactician_base_outputs.shape}, columns={len(tactician_base_outputs.columns)}"
                        )
                    additional_features_list.append(tactician_base_outputs)
                    tprint_info(
                        f"   ↪ Added tactician_base_outputs features, cumulative={sum(df.shape[1] for df in additional_features_list)} columns"
                    )
                    base_outputs_for_stats = tactician_base_outputs # Calculate stats on these

            # --- NEW: Calculate ensemble meta-features ---
            if base_outputs_for_stats is not None and not base_outputs_for_stats.empty:
                # Calculate meta-features from base model outputs
                meta_features = pd.DataFrame(index=base_outputs_for_stats.index)
                # Add these new features to the list
                additional_features_list.append(meta_features)
                tprint_success(f"✅ Added {len(meta_features.columns)} statistical meta-features for ensemble.")

            if additional_features_list:
                # Concatenate all features (base outputs + meta-features)
                # Use safe concatenation with temporal alignment validation
                try:
                    final_additional_features = self._safe_concat(
                        additional_features_list,
                        axis=1,
                        operation_name="concatenate_additional_features",
                        validate_alignment=True
                    )
                    return final_additional_features
                except ValueError as e:
                    tprint_error(f"❌ Temporal alignment error in additional features: {e}")
                    return None
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error retrieving additional model outputs: {e}")
            return None

    def _log_feature_snapshot(self, df: pd.DataFrame, source_name: str, prefix: str = "") -> None:
        """Log concise diagnostics about the feature dataframe."""
        try:
            n_samples, n_features = df.shape
            dtypes_summary = df.dtypes.value_counts().to_dict()
            sample_columns = df.columns[:10].tolist()
            tprint_info(f"{prefix}source={source_name}, samples={n_samples}, features={n_features}, dtypes={dtypes_summary}")
            tprint_info(f"{prefix}sample columns: {sample_columns}{'...' if n_features > len(sample_columns) else ''}")
        except Exception as exc:
            self.logger.debug(f"Failed to log feature snapshot for {source_name}: {exc}")

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
