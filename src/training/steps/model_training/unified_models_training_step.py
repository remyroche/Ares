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
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

# HPO imports (only those actually used)
import lightgbm as lgb
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
import psutil
import gc

from src.training.steps.model_training.hpo_config import (
    HPOOrchestrator,
    ModelParameterGroups
)

# Import disagreement features calculator
from src.feature_engineering_roadmap.disagreement_meta_features import DisagreementMetaFeatures

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning

# Import model training report generator
from src.training.steps.model_training.model_training_report_generator import create_model_training_report

# Import dynamic config calculator
from src.training.steps.model_training.dynamic_config_calculator import (
    DynamicConfigCalculator, DynamicTrainingConfig
)

# Import temporal splitting for proper train/val/test separation
from src.utils.versioned_artifacts import (
    create_temporal_split_config_for_pipeline,
    create_walkforward_split_config_for_pipeline,
    get_data_for_purpose,
    TemporalSplitConfig,
    WalkForwardSplitConfig
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
                # Create or load WALK-FORWARD temporal split configuration
                tprint_info(f"📅 Creating WALK-FORWARD split configuration for {symbol} {config.get('exchange', 'binance')}")

                # Determine data boundaries from actual data
                data_start = training_data.index.min()
                data_end = training_data.index.max()

                tprint_info(f"   Data range: {data_start} to {data_end}")
                tprint_info(f"   Total samples: {len(training_data)}")

                # Create walk-forward split config with expanding window
                walkforward_config = create_walkforward_split_config_for_pipeline(
                    symbol=symbol,
                    exchange=config.get('exchange', 'binance'),
                    timeframe=timeframe,
                    data_start=data_start,
                    data_end=data_end,
                    n_folds=3,  # 3 train/val pairs
                    val_pct_per_fold=0.10,  # 10% validation per fold
                    final_test_pct=0.15,  # 15% for final test
                    min_train_pct=0.55,  # Start with 55% training
                    embargo_days=1  # 1-day embargo
                )

                # Store walk-forward config for later use
                self._walkforward_config = walkforward_config
                config['walkforward_config'] = walkforward_config

                # Log walk-forward split boundaries
                tprint_info("=" * 80)
                tprint_info("📊 WALK-FORWARD EXPANDING WINDOW CONFIGURATION:")
                tprint_info("=" * 80)
                for fold in walkforward_config.folds:
                    train_samples = len(training_data.loc[fold.training.start:fold.training.effective_end])
                    val_samples = len(training_data.loc[fold.validation.start:fold.validation.effective_end])
                    tprint_info(f"   Fold {fold.fold_num}:")
                    tprint_info(f"      Train: {fold.training.start} → {fold.training.effective_end} ({train_samples} samples)")
                    tprint_info(f"      Val:   {fold.validation.start} → {fold.validation.effective_end} ({val_samples} samples)")

                test_samples = len(training_data.loc[walkforward_config.test.start:walkforward_config.test.end])
                tprint_info(f"   Test:  {walkforward_config.test.start} → {walkforward_config.test.end} ({test_samples} samples)")
                tprint_info(f"   Strategy: {walkforward_config.strategy}")
                tprint_info("=" * 80)

                # For walk-forward, we keep data for the LARGEST fold (Fold 3's training period)
                # This ensures we have all data needed for HPO across all folds
                last_fold = walkforward_config.folds[-1]
                tprint_info(f"🔒 Filtering data to LARGEST training period (Fold {last_fold.fold_num}) for final model training...")
                original_len = len(training_data)

                # Filter to largest training period
                training_data_filtered = training_data.loc[
                    (training_data.index >= last_fold.training.start) &
                    (training_data.index <= last_fold.training.effective_end)
                ].copy()

                filtered_len = len(training_data_filtered)
                tprint_success(f"✅ Filtered to largest training period: {original_len} → {filtered_len} samples "
                             f"({filtered_len/original_len*100:.1f}% of full dataset)")

                # Store filtered data for final model training
                training_data = training_data_filtered

                # Filter targets to match largest training period
                if analyst_targets is not None:
                    analyst_targets = analyst_targets.loc[training_data.index]
                    tprint_info(f"   ↪ Analyst targets filtered to {len(analyst_targets)} samples")

                if tactician_targets is not None:
                    tactician_targets = tactician_targets.loc[training_data.index]
                    tprint_info(f"   ↪ Tactician targets filtered to {len(tactician_targets)} samples")

                tprint_info("=" * 80)
            else:
                tprint_warning("⚠️ No training data available for temporal splitting")
                self._walkforward_config = None

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

                # Generate markdown and JSON training reports
                try:
                    tprint_info("📝 Generating training reports (Markdown + JSON)...")

                    # Prepare feature info
                    feature_info = {
                        'feature_count': training_data.shape[1] if training_data is not None else 0,
                        'feature_source': 'feature_generation_final_feature_selection_step',
                        'feature_names': list(training_data.columns) if training_data is not None else [],
                        'regime_features_included': True
                    }

                    # Generate reports
                    markdown_path, json_path = create_model_training_report(
                        training_type=training_type,
                        symbol=symbol,
                        exchange=config.get('exchange', 'binance'),
                        timeframe=timeframe,
                        direction=direction,
                        models_trained=result.get('models', {}),
                        metrics=result.get('metrics', {}),
                        hpo_results=result.get('hpo_results'),
                        regime_performance=result.get('regime_performance'),
                        training_config=config,
                        feature_info=feature_info,
                        execution_time=result.get('execution_time', 0.0),
                        outcomes_dir='outcomes'
                    )

                    if markdown_path:
                        artifacts['training_report_markdown'] = markdown_path
                        tprint_success(f"✅ Markdown report saved: {markdown_path}")

                    if json_path:
                        artifacts['training_report_json'] = json_path
                        tprint_success(f"✅ JSON metrics report saved: {json_path}")

                except Exception as e:
                    tprint_warning(f"⚠️ Failed to generate training reports: {e}")
                    self.logger.warning(f"Training report generation failed: {e}")

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

                # Generate comprehensive training reports (markdown + JSON)
                tprint_info("📝 Generating comprehensive training reports...")
                report_paths = self._generate_training_reports(result, training_type, config)
                if report_paths:
                    artifacts.update(report_paths)
                    tprint_success(f"✅ Training reports generated: {len(report_paths)} files")

                return {
                    'success': True,
                    'artifacts': artifacts,
                    'metrics': result.get('metrics', {}),
                    'training_type': training_type,
                    'execution_time': result.get('execution_time', 0.0),
                    'reports': report_paths
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
    
    def _apply_feature_selection_before_hpo(
        self,
        training_data: pd.DataFrame,
        targets: pd.Series,
        config: Dict[str, Any],
        training_type: str
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Apply feature selection before HPO to reduce dimensionality and improve performance.
        
        This method implements:
        1. Variance threshold to remove zero-variance features
        2. SelectFromModel to reduce from 1,637 to ~80 features
        3. Detailed logging of removed features
        4. Memory usage monitoring
        
        Args:
            training_data: Original training data with all features
            targets: Target variables
            config: Configuration dictionary
            training_type: Type of training (analyst_base, etc.)
            
        Returns:
            Tuple of (selected_features, targets, feature_selection_info)
        """
        try:
            tprint_info("=" * 80)
            tprint_info("🔍 FEATURE SELECTION BEFORE HPO - Reducing Dimensionality")
            tprint_info("=" * 80)
            
            original_shape = training_data.shape
            original_memory = training_data.memory_usage(deep=True).sum() / 1024**2
            sample_to_feature_ratio = len(training_data) / len(training_data.columns)
            
            tprint_info(f"📊 Original Data:")
            tprint_info(f"   Samples: {len(training_data):,}")
            tprint_info(f"   Features: {len(training_data.columns):,}")
            tprint_info(f"   Sample-to-Feature Ratio: {sample_to_feature_ratio:.3f}")
            tprint_info(f"   Memory Usage: {original_memory:.2f} MB")
            
            feature_selection_info = {
                'original_shape': original_shape,
                'original_memory_mb': original_memory,
                'sample_to_feature_ratio': sample_to_feature_ratio,
                'removed_features': {
                    'zero_variance': [],
                    'low_importance': []
                },
                'feature_importance_stats': {}
            }
            
            # ========================================================================
            # STEP 1: Remove Zero-Variance Features
            # ========================================================================
            tprint_info("\n🔧 STEP 1: Removing Zero-Variance Features...")
            
            # Calculate variance for each feature
            numeric_cols = training_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                tprint_warning("⚠️ No numeric columns found for variance threshold")
                return training_data, targets, feature_selection_info
            
            # Calculate variance
            variances = training_data[numeric_cols].var()
            zero_var_features = variances[variances == 0].index.tolist()
            
            if zero_var_features:
                tprint_warning(f"⚠️ Found {len(zero_var_features)} zero-variance features:")
                for feature in zero_var_features[:10]:  # Show first 10
                    tprint_info(f"      - {feature}")
                if len(zero_var_features) > 10:
                    tprint_info(f"      ... and {len(zero_var_features) - 10} more")
                
                # Remove zero-variance features
                training_data = training_data.drop(columns=zero_var_features)
                feature_selection_info['removed_features']['zero_variance'] = zero_var_features
                
                tprint_success(f"✅ Removed {len(zero_var_features)} zero-variance features")
            else:
                tprint_success("✅ No zero-variance features found")
            
            # ========================================================================
            # STEP 2: Apply SelectFromModel for Feature Importance Selection
            # ========================================================================
            tprint_info("\n🎯 STEP 2: SelectFromModel Feature Importance Selection...")
            
            current_shape = training_data.shape
            current_memory = training_data.memory_usage(deep=True).sum() / 1024**2
            
            tprint_info(f"📊 After Zero-Variance Removal:")
            tprint_info(f"   Features: {current_shape[1]:,}")
            tprint_info(f"   Memory Usage: {current_memory:.2f} MB")
            
            # Determine target number of features based on dataset size
            n_samples = len(training_data)
            if n_samples < 500:
                target_features = min(80, max(50, n_samples // 3))
            elif n_samples < 1000:
                target_features = min(100, max(60, n_samples // 5))
            else:
                target_features = min(150, max(80, n_samples // 10))
            
            tprint_info(f"🎯 Target feature count: {target_features}")
            
            # Use RandomForest for feature importance
            rf_selector = SelectFromModel(
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                max_features=target_features
            )
            
            # Fit selector
            tprint_info("🔄 Training RandomForest for feature importance...")
            rf_selector.fit(training_data, targets)
            
            # Get selected features
            selected_features_mask = rf_selector.get_support()
            selected_feature_names = training_data.columns[selected_features_mask].tolist()
            removed_feature_names = training_data.columns[~selected_features_mask].tolist()
            
            # Calculate feature importances
            feature_importances = rf_selector.estimator_.feature_importances_
            importance_df = pd.DataFrame({
                'feature': training_data.columns,
                'importance': feature_importances
            }).sort_values('importance', ascending=False)
            
            # Log feature importance statistics
            feature_selection_info['feature_importance_stats'] = {
                'top_10_features': importance_df.head(10).to_dict('records'),
                'importance_distribution': {
                    'mean': float(importance_df['importance'].mean()),
                    'std': float(importance_df['importance'].std()),
                    'min': float(importance_df['importance'].min()),
                    'max': float(importance_df['importance'].max()),
                    'median': float(importance_df['importance'].median())
                },
                'high_importance_count': int((importance_df['importance'] > 0.01).sum()),
                'medium_importance_count': int((importance_df['importance'] > 0.001).sum()),
                'low_importance_count': int((importance_df['importance'] <= 0.001).sum())
            }
            
            # Apply feature selection
            training_data_selected = training_data[selected_feature_names]
            
            feature_selection_info['removed_features']['low_importance'] = removed_feature_names
            
            # ========================================================================
            # STEP 3: Log Results and Memory Savings
            # ========================================================================
            final_shape = training_data_selected.shape
            final_memory = training_data_selected.memory_usage(deep=True).sum() / 1024**2
            memory_reduction = (original_memory - final_memory) / original_memory * 100
            feature_reduction = (original_shape[1] - final_shape[1]) / original_shape[1] * 100
            final_sample_to_feature_ratio = len(training_data_selected) / len(training_data_selected.columns)
            
            tprint_info("\n📊 FEATURE SELECTION SUMMARY:")
            tprint_info("=" * 60)
            tprint_info(f"Original:  {original_shape[0]:,} samples × {original_shape[1]:,} features")
            tprint_info(f"Final:     {final_shape[0]:,} samples × {final_shape[1]:,} features")
            tprint_info(f"Reduction: {feature_reduction:.1f}% features removed")
            tprint_info(f"Memory:    {original_memory:.2f} MB → {final_memory:.2f} MB ({memory_reduction:.1f}% reduction)")
            tprint_info(f"Ratio:     {sample_to_feature_ratio:.3f} → {final_sample_to_feature_ratio:.3f}")
            
            tprint_info(f"\n🎯 Top 10 Selected Features:")
            for i, row in importance_df.head(10).iterrows():
                tprint_info(f"   {i+1:2d}. {row['feature']:<40} (importance: {row['importance']:.6f})")
            
            # Check for suspicious HPO scores potential
            if final_sample_to_feature_ratio < 1.0:
                tprint_warning(f"⚠️ Low sample-to-feature ratio ({final_sample_to_feature_ratio:.3f}) - risk of overfitting")
            elif final_sample_to_feature_ratio < 2.0:
                tprint_warning(f"⚠️ Moderate sample-to-feature ratio ({final_sample_to_feature_ratio:.3f}) - monitor for overfitting")
            else:
                tprint_success(f"✅ Good sample-to-feature ratio ({final_sample_to_feature_ratio:.3f})")
            
            # Log removed features for debugging
            if removed_feature_names:
                tprint_info(f"\n🗑️ Removed {len(removed_feature_names)} low-importance features:")
                low_importance_preview = removed_feature_names[:5]
                for feature in low_importance_preview:
                    importance = importance_df[importance_df['feature'] == feature]['importance'].iloc[0]
                    tprint_info(f"      - {feature} (importance: {importance:.6f})")
                if len(removed_feature_names) > 5:
                    tprint_info(f"      ... and {len(removed_feature_names) - 5} more")
            
            # Update feature selection info
            feature_selection_info.update({
                'final_shape': final_shape,
                'final_memory_mb': final_memory,
                'memory_reduction_pct': memory_reduction,
                'feature_reduction_pct': feature_reduction,
                'final_sample_to_feature_ratio': final_sample_to_feature_ratio,
                'target_features': target_features,
                'actual_features': len(selected_feature_names)
            })
            
            tprint_info("=" * 80)
            tprint_success(f"✅ Feature selection complete: {original_shape[1]:,} → {final_shape[1]:,} features")
            tprint_info("=" * 80)
            
            return training_data_selected, targets, feature_selection_info
            
        except Exception as e:
            tprint_error(f"❌ Feature selection failed: {e}")
            import traceback
            self.logger.error(f"Feature selection error: {e}\n{traceback.format_exc()}")
            # Return original data if feature selection fails
            return training_data, targets, {'error': str(e)}

    def _monitor_memory_usage(self, operation_name: str = "Unknown") -> Dict[str, float]:
        """
        Monitor current memory usage and return statistics.
        
        Args:
            operation_name: Name of the current operation for logging
            
        Returns:
            Dictionary with memory usage statistics
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # System memory
            system_memory = psutil.virtual_memory()
            
            memory_stats = {
                'process_memory_mb': memory_info.rss / 1024 / 1024,
                'process_memory_percent': memory_percent,
                'system_memory_gb': system_memory.total / 1024 / 1024 / 1024,
                'system_memory_used_gb': system_memory.used / 1024 / 1024 / 1024,
                'system_memory_percent': system_memory.percent,
                'available_memory_gb': system_memory.available / 1024 / 1024 / 1024
            }
            
            # Log warnings if memory usage is high
            if memory_percent > 85:
                tprint_warning(f"⚠️ HIGH MEMORY USAGE during {operation_name}: {memory_percent:.1f}%")
            elif memory_percent > 70:
                tprint_info(f"ℹ️ Memory usage during {operation_name}: {memory_percent:.1f}%")
            
            return memory_stats
            
        except Exception as e:
            self.logger.warning(f"Memory monitoring failed: {e}")
            return {}

    def _check_for_data_leakage(
        self,
        hpo_score: float,
        feature_selection_info: Dict[str, Any],
        training_type: str
    ) -> bool:
        """
        Check for potential data leakage based on suspiciously high HPO scores.
        
        Args:
            hpo_score: The HPO score achieved
            feature_selection_info: Information about feature selection
            training_type: Type of training
            
        Returns:
            True if data leakage is suspected, False otherwise
        """
        try:
            leakage_suspected = False
            
            # Check for suspiciously high scores
            if hpo_score > 0.85:
                tprint_warning(f"⚠️ SUSPICIOUSLY HIGH HPO SCORE: {hpo_score:.6f}")
                tprint_warning(f"   This may indicate data leakage or overfitting")
                leakage_suspected = True
            
            # Check sample-to-feature ratio
            ratio = feature_selection_info.get('final_sample_to_feature_ratio', 0)
            if ratio < 1.0 and hpo_score > 0.75:
                tprint_warning(f"⚠️ HIGH SCORE WITH LOW SAMPLE-TO-FEATURE RATIO")
                tprint_warning(f"   Score: {hpo_score:.6f}, Ratio: {ratio:.3f}")
                tprint_warning(f"   This combination often indicates data leakage")
                leakage_suspected = True
            
            # Check for extreme feature reduction with high score
            feature_reduction = feature_selection_info.get('feature_reduction_pct', 0)
            if feature_reduction > 90 and hpo_score > 0.80:
                tprint_warning(f"⚠️ EXTREME FEATURE REDUCTION WITH HIGH SCORE")
                tprint_warning(f"   Feature reduction: {feature_reduction:.1f}%, Score: {hpo_score:.6f}")
                leakage_suspected = True
            
            if leakage_suspected:
                tprint_error("🚨 DATA LEAKAGE INVESTIGATION RECOMMENDED:")
                tprint_error("   1. Check temporal ordering of features and targets")
                tprint_error("   2. Verify no future information is in features")
                tprint_error("   3. Review feature engineering for look-ahead bias")
                tprint_error("   4. Consider using stricter temporal validation")
            
            return leakage_suspected
            
        except Exception as e:
            self.logger.warning(f"Data leakage check failed: {e}")
            return False

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
            
            tprint_info("🔍 Starting Hierarchical HPO with Walk-Forward Cross-Validation...")

            # ========================================================================
            # FEATURE SELECTION: Apply before HPO to reduce dimensionality
            # ========================================================================
            tprint_info("=" * 80)
            tprint_info("🎯 FEATURE SELECTION BEFORE HPO")
            tprint_info("=" * 80)
            
            # Monitor memory before feature selection
            memory_before = self._monitor_memory_usage("Before Feature Selection")
            
            # Apply feature selection
            training_data_selected, targets_selected, feature_selection_info = self._apply_feature_selection_before_hpo(
                training_data=training_data,
                targets=targets,
                config=config,
                training_type=training_type
            )
            
            # Monitor memory after feature selection
            memory_after = self._monitor_memory_usage("After Feature Selection")
            
            # Update training_data and targets for HPO
            training_data = training_data_selected
            targets = targets_selected
            
            # Store feature selection info for later use
            self._feature_selection_info = feature_selection_info

            # ========================================================================
            # WALK-FORWARD: HPO across multiple train/val folds with score aggregation
            # ========================================================================
            tprint_info("=" * 80)
            tprint_info("🔐 WALK-FORWARD HPO - Multiple Validation Windows")
            tprint_info("=" * 80)

            # Get walk-forward config
            walkforward_config = getattr(self, '_walkforward_config', None)

            # This will store fold data for iteration
            fold_data_list = []

            if walkforward_config is not None and hasattr(self, '_full_training_data'):
                # Walk-forward mode: iterate through all folds
                tprint_info(f"✅ Using WALK-FORWARD with {len(walkforward_config.folds)} folds")
                tprint_info("   Each fold provides independent train/val split for robust HPO")
                tprint_info("")

                for fold in walkforward_config.folds:
                    # Get training data for this fold (apply feature selection)
                    fold_training_data_full = self._full_training_data.loc[
                        (self._full_training_data.index >= fold.training.start) &
                        (self._full_training_data.index <= fold.training.effective_end)
                    ].copy()

                    # Get validation data for this fold (apply feature selection)
                    fold_validation_data_full = self._full_training_data.loc[
                        (self._full_training_data.index >= fold.validation.start) &
                        (self._full_training_data.index <= fold.validation.effective_end)
                    ].copy()
                    
                    # Apply the same feature selection to fold data
                    # Use only the selected features from the main feature selection
                    selected_features = training_data.columns.tolist()
                    fold_training_data = fold_training_data_full[selected_features].copy()
                    fold_validation_data = fold_validation_data_full[selected_features].copy()

                    # Get targets for training and validation
                    if training_type.startswith('analyst') and hasattr(self, '_full_analyst_targets'):
                        fold_train_targets = self._full_analyst_targets.loc[fold_training_data.index]
                        fold_val_targets = self._full_analyst_targets.loc[fold_validation_data.index]
                    elif training_type.startswith('tactician') and hasattr(self, '_full_tactician_targets'):
                        fold_train_targets = self._full_tactician_targets.loc[fold_training_data.index]
                        fold_val_targets = self._full_tactician_targets.loc[fold_validation_data.index]
                    else:
                        fold_train_targets = None
                        fold_val_targets = None

                    if fold_train_targets is not None and fold_val_targets is not None:
                        fold_data_list.append({
                            'fold_num': fold.fold_num,
                            'X_train': fold_training_data,
                            'y_train': fold_train_targets,
                            'X_val': fold_validation_data,
                            'y_val': fold_val_targets
                        })

                        tprint_info(f"   Fold {fold.fold_num}:")
                        tprint_info(f"      Train: {len(fold_training_data)} samples ({fold.training.start} → {fold.training.effective_end})")
                        tprint_info(f"      Val:   {len(fold_validation_data)} samples ({fold.validation.start} → {fold.validation.effective_end})")

                tprint_success(f"✅ Prepared {len(fold_data_list)} folds for walk-forward HPO")
                tprint_info("   🔒 No data leakage: Each validation window is completely separate")
                tprint_info("")
            else:
                # Fallback to 80/20 split if walk-forward config not available
                tprint_warning("⚠️ Walk-forward config not available, falling back to 80/20 split")
                hpo_train_size = int(len(training_data) * 0.8)
                X_train = training_data.iloc[:hpo_train_size]
                X_val = training_data.iloc[hpo_train_size:]
                y_train = targets.iloc[:hpo_train_size]
                y_val = targets.iloc[hpo_train_size:]

                fold_data_list.append({
                    'fold_num': 1,
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val
                })
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
            
            # Run HPO for each model across all folds
            all_results = {}
            for model_info in models_to_optimize:
                tprint_info(f"🎯 Optimizing {model_info['name']} ({model_info['type']}) across {len(fold_data_list)} folds...")
                tprint_info("")

                # Store results for each fold
                fold_results = []

                for fold_data in fold_data_list:
                    fold_num = fold_data['fold_num']
                    tprint_info(f"   Fold {fold_num}/{len(fold_data_list)}...")

                    # Run HPO on this fold
                    result = await asyncio.to_thread(
                        self.hpo_orchestrator.run_hpo,
                        model_name=model_info['name'],
                        model_type=model_info['type'],
                        X_train=fold_data['X_train'],
                        y_train=fold_data['y_train'],
                        X_val=fold_data['X_val'],
                        y_val=fold_data['y_val'],
                        model_class=model_info['class'],
                        is_classification=model_info['is_classification']
                    )

                    if result:
                        fold_results.append({
                            'fold_num': fold_num,
                            'result': result,
                            'score': result.best_score
                        })
                        tprint_info(f"      ✓ Fold {fold_num} score: {result.best_score:.6f}")
                    else:
                        tprint_warning(f"      ⚠️ Fold {fold_num} HPO failed")

                # Aggregate scores across folds
                if fold_results:
                    scores = [fr['score'] for fr in fold_results]
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)

                    # Use the result from the best fold (highest score)
                    best_fold = max(fold_results, key=lambda x: x['score'])
                    best_result = best_fold['result']

                    # Check for data leakage
                    self._check_for_data_leakage(
                        hpo_score=best_fold['score'],
                        feature_selection_info=feature_selection_info,
                        training_type=training_type
                    )

                    all_results[model_info['name']] = best_result

                    tprint_success(f"✅ {model_info['name']} Walk-Forward HPO Complete:")
                    tprint_info(f"   Average score: {mean_score:.6f} ± {std_score:.6f}")
                    tprint_info(f"   Best fold: {best_fold['fold_num']} (score: {best_fold['score']:.6f})")
                    tprint_info(f"   Optimal params (from best fold): {best_result.best_params}")
                    
                    # Memory cleanup after each model HPO
                    gc.collect()
                    memory_cleanup = self._monitor_memory_usage(f"After {model_info['name']} HPO")
                    tprint_info("")
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
            
            # ========================================================================
            # FEATURE LOADING FROM HDF5 VERSIONED ARTIFACTS
            # ========================================================================
            # Determine feature set size to use (default to 60 features for analyst base)
            # The 60-feature set is the recommended size for optimal model performance
            feature_set_size = config.get('feature_set_size', 60)

            tprint_info("=" * 80)
            tprint_info("📦 LOADING FEATURES FROM HDF5 VERSIONED ARTIFACTS")
            tprint_info("=" * 80)
            tprint_info(f"   Source Step: feature_generation_final_feature_selection_step")
            tprint_info(f"   Target Feature Set Size: {feature_set_size} features")
            tprint_info(f"   Storage Format: HDF5 (via versioned_artifacts)")

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

            tprint_info(f"🔎 Attempting to load training features from HDF5 artifacts...")
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

            # ========================================================================
            # LABELS/TARGETS LOADING FROM HDF5 VERSIONED ARTIFACTS
            # ========================================================================
            # Get targets from labeling integration step (direction-aware)
            direction = config.get('direction', 'long')
            tprint_info("=" * 80)
            tprint_info("🎯 LOADING LABELS/TARGETS FROM HDF5 VERSIONED ARTIFACTS")
            tprint_info("=" * 80)
            tprint_info(f"   Source Step: feature_generation_labeling_integration_step")
            tprint_info(f"   Direction: {direction}")
            tprint_info(f"   Storage Format: HDF5 (via versioned_artifacts)")

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

                                # Check for new simplified target structure first (highest priority)
                                if 'target_long' in labeled_data.columns and 'target_short' in labeled_data.columns:
                                    # New simplified target structure from labeling integration step
                                    # Use direction-specific target based on the training direction
                                    if direction == 'long':
                                        target_col = 'target_long'
                                        tprint_success(f"✅ Using new simplified target structure: target_long for {direction} direction")
                                    elif direction == 'short':
                                        target_col = 'target_short'
                                        tprint_success(f"✅ Using new simplified target structure: target_short for {direction} direction")
                                    else:
                                        # For 'both' direction, use target_long as default for analyst
                                        target_col = 'target_long'
                                        tprint_success(f"✅ Using new simplified target structure: target_long (default for analyst)")
                                    
                                    # Log target statistics for the new simplified structure
                                    long_signals = (labeled_data['target_long'] > 0).sum()
                                    short_signals = (labeled_data['target_short'] > 0).sum()
                                    total_signals = long_signals + short_signals
                                    tprint_info(f"📊 Simplified target statistics:")
                                    tprint_info(f"   • Long signals: {long_signals} ({long_signals/len(labeled_data)*100:.1f}%)")
                                    tprint_info(f"   • Short signals: {short_signals} ({short_signals/len(labeled_data)*100:.1f}%)")
                                    tprint_info(f"   • Total signals: {total_signals} ({total_signals/len(labeled_data)*100:.1f}%)")
                                    
                                else:
                                    # Fallback to legacy target detection
                                    # Prioritize direction-specific target columns
                                    direction_specific_cols = [
                                        col for col in target_cols
                                        if direction in col.lower()
                                    ]

                                    if direction_specific_cols:
                                        target_col = direction_specific_cols[0]
                                        tprint_success(f"✅ Using legacy direction-specific target column: {target_col}")
                                    else:
                                        target_col = target_cols[0]
                                        tprint_warning(f"⚠️ No direction-specific column found, using legacy: {target_col}")

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
            
            # Also extract tactician targets from same labeled_data if not found separately
            if tactician_targets is None and 'labeled_data' in locals():
                try:
                    labeled_data = locals()['labeled_data']
                    if isinstance(labeled_data, pd.DataFrame):
                        # For tactician, use the appropriate target based on direction
                        if 'target_long' in labeled_data.columns and 'target_short' in labeled_data.columns:
                            # New simplified target structure
                            if direction == 'long':
                                # For long direction tactician, use target_short (exit signals for long positions)
                                tactician_targets = labeled_data['target_short']
                                tprint_success(f"✅ Using target_short for tactician {direction} direction (exit signals)")
                            elif direction == 'short':
                                # For short direction tactician, use target_long (exit signals for short positions)
                                tactician_targets = labeled_data['target_long']
                                tprint_success(f"✅ Using target_long for tactician {direction} direction (exit signals)")
                            else:
                                # Default: use target_long for tactician (both directions)
                                tactician_targets = labeled_data['target_long']
                                tprint_success(f"✅ Using target_long for tactician (default for both directions)")
                        else:
                            # Fallback to legacy structure - use same targets as analyst
                            tactician_targets = analyst_targets.copy() if analyst_targets is not None else None
                            tprint_warning(f"⚠️ Using legacy targets for tactician (same as analyst)")
                except Exception as e:
                    self.logger.warning(f"Failed to extract tactician targets: {e}")
            
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
            # ========================================================================
            # REGIME PROBABILITY LOADING FROM HDF5 VERSIONED ARTIFACTS
            # ========================================================================
            try:
                tprint_info("=" * 80)
                tprint_info("🌍 LOADING REGIME PROBABILITIES FROM HDF5 VERSIONED ARTIFACTS")
                tprint_info("=" * 80)
                tprint_info(f"   Source Step: regime_ensemble_training")
                tprint_info(f"   Artifact Name: regime_ensemble_predictions")
                tprint_info(f"   Storage Format: HDF5 (via versioned_artifacts)")

                # Load regime ensemble predictions (from regime_ensemble_training) - REQUIRED
                regime_features = self._get_artifact('regime_ensemble_predictions', 'data')
                if regime_features is None:
                    error_msg = (
                        "❌ CRITICAL: regime_ensemble_predictions artifact not found!\n"
                        "   This artifact is REQUIRED for model training.\n"
                        "   Source: regime_ensemble_training step\n"
                        "   Format: HDF5 (versioned_artifacts)\n"
                        "   Please ensure regime_ensemble_training step has run successfully."
                    )
                    tprint_error(error_msg)
                    raise ValueError(error_msg)

                tprint_info(f"   ↪ Retrieved regime_ensemble_predictions from HDF5: shape={regime_features.shape}, columns={len(regime_features.columns)}")
                tprint_success(f"✅ Loaded regime ensemble predictions from HDF5: {regime_features.shape}")

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

            # --- NEW: Calculate ensemble meta-features (disagreement features) ---
            if base_outputs_for_stats is not None and not base_outputs_for_stats.empty:
                try:
                    tprint_info("🔍 Calculating disagreement meta-features from base model outputs...")

                    # Initialize disagreement features calculator
                    disagreement_calc = DisagreementMetaFeatures(logger=self.logger)

                    # Prepare model outputs as dict for disagreement calculator
                    # Assume columns are named like: model1_prediction, model2_prediction, etc.
                    # or model1_prob_0, model1_prob_1, model2_prob_0, model2_prob_1, etc.

                    model_predictions = {}
                    model_probabilities = {}
                    model_confidences = {}

                    # Parse column names to identify model outputs
                    for col in base_outputs_for_stats.columns:
                        col_lower = col.lower()

                        # Extract model predictions (columns ending with _prediction or _pred)
                        if '_prediction' in col_lower or '_pred' in col_lower:
                            model_name = col.split('_prediction')[0].split('_pred')[0]
                            model_predictions[model_name] = base_outputs_for_stats[col].values

                        # Extract model probabilities (columns with _prob or _probability)
                        elif '_prob' in col_lower or '_probability' in col_lower:
                            # Group multi-class probabilities by model
                            parts = col.split('_')
                            for i, part in enumerate(parts):
                                if 'prob' in part.lower():
                                    model_name = '_'.join(parts[:i])
                                    if model_name not in model_probabilities:
                                        model_probabilities[model_name] = []
                                    model_probabilities[model_name].append(base_outputs_for_stats[col].values)
                                    break

                        # Extract model confidence scores
                        elif '_confidence' in col_lower or '_conf' in col_lower:
                            model_name = col.split('_confidence')[0].split('_conf')[0]
                            model_confidences[model_name] = base_outputs_for_stats[col].values

                    # Convert probability lists to arrays
                    for model_name in model_probabilities:
                        if isinstance(model_probabilities[model_name], list):
                            model_probabilities[model_name] = np.column_stack(model_probabilities[model_name])

                    tprint_info(f"   ↪ Parsed {len(model_predictions)} prediction outputs")
                    tprint_info(f"   ↪ Parsed {len(model_probabilities)} probability outputs")
                    tprint_info(f"   ↪ Parsed {len(model_confidences)} confidence outputs")

                    # Calculate disagreement features
                    if model_predictions or model_probabilities:
                        # If we only have predictions, create dummy probabilities
                        if not model_probabilities and model_predictions:
                            tprint_warning("⚠️ No probabilities found, using predictions only for disagreement features")
                            # Convert predictions to simple binary probabilities
                            for model_name, preds in model_predictions.items():
                                probs = np.column_stack([
                                    np.where(preds > 0, 0, 1),  # Prob of class 0 (negative)
                                    np.where(preds > 0, 1, 0)   # Prob of class 1 (positive)
                                ])
                                model_probabilities[model_name] = probs

                        disagreement_features_dict = disagreement_calc.calculate_all_disagreement_features(
                            model_predictions=model_predictions,
                            model_probabilities=model_probabilities,
                            model_confidences=model_confidences if model_confidences else None
                        )

                        # Convert dict of Series to DataFrame
                        all_meta_features = pd.DataFrame(disagreement_features_dict, index=base_outputs_for_stats.index)

                        # Filter to keep only the 6 most important disagreement features
                        # These are the most informative features for ensemble learning
                        core_features = [
                            'prediction_dispersion',    # 1. Variance of predictions across models
                            'confidence_gap',           # 2. Margin between top predictions
                            'uncertainty',              # 3. Normalized entropy (uncertainty measure)
                            'prediction_range',         # 4. Range of predictions (max - min)
                            'avg_divergence',           # 5. Average pairwise model divergence
                            'max_confidence'            # 6. Highest confidence among models
                        ]

                        # Select only core features that exist
                        available_core_features = [f for f in core_features if f in all_meta_features.columns]
                        meta_features = all_meta_features[available_core_features].copy()

                        # Normalize prediction_range and avg_divergence by standard deviation
                        features_to_normalize = ['prediction_range', 'avg_divergence']
                        for feature in features_to_normalize:
                            if feature in meta_features.columns:
                                feature_std = meta_features[feature].std()
                                if feature_std > 0:
                                    meta_features[feature] = meta_features[feature] / feature_std
                                    tprint_info(f"   ↪ Normalized '{feature}' by std={feature_std:.6f}")
                                else:
                                    tprint_warning(f"   ⚠️ Cannot normalize '{feature}' (std=0)")

                        tprint_success(f"✅ Calculated {len(meta_features.columns)} core disagreement meta-features:")
                        tprint_info(f"   Feature columns: {list(meta_features.columns)}")

                        if len(available_core_features) < len(core_features):
                            missing = set(core_features) - set(available_core_features)
                            tprint_warning(f"   ⚠️ Missing features: {missing}")

                        # Add these new features to the list
                        additional_features_list.append(meta_features)
                    else:
                        tprint_warning("⚠️ Could not parse model outputs for disagreement features, creating empty meta-features")
                        meta_features = pd.DataFrame(index=base_outputs_for_stats.index)
                        # Don't add empty DataFrame to avoid errors

                except Exception as e:
                    tprint_error(f"❌ Failed to calculate disagreement features: {e}")
                    import traceback
                    tprint_error(traceback.format_exc())
                    self.logger.error(f"Disagreement feature calculation failed: {e}")
                    # Continue without disagreement features rather than failing

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

            # Save performance metrics (JSON + Markdown report)
            if 'metrics' in result:
                # Save as JSON
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

                # Save as Markdown report
                try:
                    md_report_path = self._generate_metrics_markdown_report(
                        metrics=result['metrics'],
                        training_type=training_type,
                        config=config,
                        hpo_results=result.get('hpo_results'),
                        execution_time=result.get('execution_time', 0.0)
                    )
                    if md_report_path:
                        artifacts[f"{training_type}_metrics_report"] = md_report_path
                        tprint_success(f"✅ Saved metrics markdown report: {md_report_path}")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to save markdown report: {e}")
                    self.logger.warning(f"Markdown report generation failed: {e}")

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

    def _generate_metrics_markdown_report(
        self,
        metrics: Dict[str, Any],
        training_type: str,
        config: Dict[str, Any],
        hpo_results: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0
    ) -> Optional[str]:
        """
        Generate a comprehensive markdown report for training metrics.

        Args:
            metrics: Training metrics dictionary
            training_type: Type of training (tactician_ensemble, etc.)
            config: Training configuration
            hpo_results: HPO optimization results (optional)
            execution_time: Total execution time in seconds

        Returns:
            Path to saved markdown report, or None if failed
        """
        try:
            import os
            from datetime import datetime

            # Generate report content
            report_lines = []

            # Header
            report_lines.append(f"# {training_type.replace('_', ' ').title()} Training Report")
            report_lines.append("")
            report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"**Symbol:** {config.get('symbol', 'UNKNOWN')}")
            report_lines.append(f"**Exchange:** {config.get('exchange', 'binance')}")
            report_lines.append(f"**Timeframe:** {config.get('timeframe', '15m')}")
            report_lines.append(f"**Direction:** {config.get('direction', 'long')}")
            report_lines.append(f"**Execution Time:** {execution_time:.2f}s")
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")

            # HPO Results Section
            if hpo_results:
                report_lines.append("## Hyperparameter Optimization (HPO)")
                report_lines.append("")
                report_lines.append("### Best HPO Scores")
                report_lines.append("")

                if 'best_scores' in hpo_results:
                    report_lines.append("| Model | Score | Parameters |")
                    report_lines.append("|-------|-------|------------|")
                    for model_name, score_data in hpo_results['best_scores'].items():
                        score = score_data.get('score', 'N/A') if isinstance(score_data, dict) else score_data
                        params = score_data.get('params', {}) if isinstance(score_data, dict) else {}
                        params_str = ', '.join([f"{k}={v}" for k, v in list(params.items())[:3]])
                        if len(params) > 3:
                            params_str += ', ...'
                        report_lines.append(f"| {model_name} | {score:.6f} | {params_str} |")

                report_lines.append("")
                report_lines.append("### HPO Details")
                report_lines.append("")
                report_lines.append(f"- **Optimization Rounds:** {hpo_results.get('optimization_rounds', 'N/A')}")
                report_lines.append(f"- **Total Trials:** {hpo_results.get('total_trials', 'N/A')}")
                report_lines.append(f"- **Best Overall Score:** {hpo_results.get('best_overall_score', 'N/A')}")
                report_lines.append("")
                report_lines.append("---")
                report_lines.append("")

            # Training Metrics Section
            report_lines.append("## Training Metrics")
            report_lines.append("")

            # Accuracy metrics
            if 'accuracy' in metrics or 'train_accuracy' in metrics:
                report_lines.append("### Accuracy Metrics")
                report_lines.append("")
                report_lines.append("| Split | Accuracy |")
                report_lines.append("|-------|----------|")
                for split in ['train', 'val', 'test']:
                    key = f"{split}_accuracy"
                    if key in metrics:
                        report_lines.append(f"| {split.capitalize()} | {metrics[key]:.4f} |")
                    elif split == 'train' and 'accuracy' in metrics:
                        report_lines.append(f"| Train | {metrics['accuracy']:.4f} |")
                report_lines.append("")

            # R² metrics
            if any('r2' in k.lower() for k in metrics.keys()):
                report_lines.append("### R² Score Metrics")
                report_lines.append("")
                report_lines.append("| Split | R² Score |")
                report_lines.append("|-------|----------|")
                for split in ['train', 'val', 'test']:
                    for key in [f"{split}_r2", f"{split}_r2_score", f"r2_{split}"]:
                        if key in metrics:
                            report_lines.append(f"| {split.capitalize()} | {metrics[key]:.4f} |")
                            break
                report_lines.append("")

            # Loss metrics
            if any('loss' in k.lower() for k in metrics.keys()):
                report_lines.append("### Loss Metrics")
                report_lines.append("")
                report_lines.append("| Split | Loss |")
                report_lines.append("|-------|------|")
                for split in ['train', 'val', 'test']:
                    for key in [f"{split}_loss", f"loss_{split}"]:
                        if key in metrics:
                            report_lines.append(f"| {split.capitalize()} | {metrics[key]:.6f} |")
                            break
                report_lines.append("")

            # Other metrics
            report_lines.append("### Additional Metrics")
            report_lines.append("")

            # Filter out already-displayed metrics
            displayed_keys = set()
            for key in metrics.keys():
                if any(x in key.lower() for x in ['accuracy', 'r2', 'loss']):
                    displayed_keys.add(key)

            remaining_metrics = {k: v for k, v in metrics.items() if k not in displayed_keys}

            if remaining_metrics:
                report_lines.append("| Metric | Value |")
                report_lines.append("|--------|-------|")
                for key, value in remaining_metrics.items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"| {key} | {value:.6f} |")
                    else:
                        report_lines.append(f"| {key} | {value} |")
                report_lines.append("")

            # Model Information
            report_lines.append("---")
            report_lines.append("")
            report_lines.append("## Model Information")
            report_lines.append("")
            report_lines.append(f"- **Training Type:** {training_type}")
            report_lines.append(f"- **Execution Mode:** {config.get('execution_mode', 'unknown')}")
            report_lines.append(f"- **Enable HPO:** {config.get('enable_hpo', False)}")
            report_lines.append("")

            # Save report
            report_content = '\n'.join(report_lines)

            # Determine output directory
            output_dir = os.path.join('outcomes', 'training_reports')
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol = config.get('symbol', 'UNKNOWN')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            filename = f"{training_type}_{symbol}_{timeframe}_{direction}_{timestamp}.md"
            filepath = os.path.join(output_dir, filename)

            # Write file
            with open(filepath, 'w') as f:
                f.write(report_content)

            tprint_success(f"✅ Generated markdown metrics report: {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Failed to generate markdown report: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
          
    def _extract_comprehensive_metrics(
        self,
        result: Dict[str, Any],
        training_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract comprehensive metrics from training results.

        This method centralizes metric extraction for all model types,
        ensuring consistency and completeness across all reports.

        Args:
            result: Training result dictionary
            training_type: Type of training (analyst_base, analyst_ensemble, tactician_base, tactician_ensemble)
            config: Configuration dictionary

        Returns:
            Comprehensive metrics dictionary with all available metrics organized by category
        """
        metrics = result.get('metrics', {})
        models = result.get('models', {})

        comprehensive_metrics = {
            'model_type': training_type,
            'timestamp': datetime.now().isoformat(),
            'execution_summary': {},
            'overall_performance': {},
            'per_model_metrics': {},
            'training_metrics': {},
            'validation_metrics': {},
            'test_metrics': {},
            'hpo_results': {},
            'feature_importance': {},
            'data_quality': {},
            'model_complexity': {},
            'prediction_statistics': {},
            'ensemble_specific': {} if 'ensemble' in training_type else None,
            'walkforward_results': {},
            'error_analysis': {},
            'data_drift_checks': {},
            'uncertainty_calibration': {},
            'shap_explanations': {},
            'threshold_optimization': {}
        }

        # ===== EXECUTION SUMMARY =====
        comprehensive_metrics['execution_summary'] = {
            'success': result.get('success', False),
            'execution_time_seconds': result.get('execution_time', 0.0),
            'training_type': training_type,
            'models_trained_count': len(models),
            'model_names': list(models.keys()) if models else [],
            'error': result.get('error', None),
            'warnings': result.get('warnings', [])
        }

        # ===== OVERALL PERFORMANCE METRICS =====
        overall_keys = [
            'overall_accuracy', 'overall_precision', 'overall_recall', 'overall_f1_score',
            'overall_r2_score', 'overall_mse', 'overall_mae', 'overall_rmse',
            'overall_mape', 'overall_sharpe_ratio', 'overall_sortino_ratio',
            'best_model', 'best_model_score', 'model_count'
        ]
        for key in overall_keys:
            if key in metrics:
                comprehensive_metrics['overall_performance'][key] = metrics[key]

        # ===== PER-MODEL METRICS =====
        for model_name in models.keys():
            model_metrics = {}

            # Standard metrics per model
            metric_types = [
                'accuracy', 'precision', 'recall', 'f1_score',
                'r2_score', 'mse', 'mae', 'rmse', 'mape',
                'train_accuracy', 'train_r2', 'train_loss', 'train_mse',
                'val_accuracy', 'val_r2', 'val_loss', 'val_mse',
                'test_accuracy', 'test_r2', 'test_loss', 'test_mse',
                'cv_score_mean', 'cv_score_std', 'cv_scores',
                'training_time_seconds', 'prediction_time_seconds',
                'n_estimators', 'max_depth', 'learning_rate', 'num_leaves',
                'iterations', 'depth', 'l2_leaf_reg'
            ]

            for metric_type in metric_types:
                # Check both prefixed and non-prefixed versions
                for key_variant in [f"{model_name}_{metric_type}", f"{metric_type}_{model_name}"]:
                    if key_variant in metrics:
                        model_metrics[metric_type] = metrics[key_variant]
                        break

            # Add if any metrics found
            if model_metrics:
                comprehensive_metrics['per_model_metrics'][model_name] = model_metrics

        # ===== SPLIT-BASED METRICS (Train/Val/Test) =====
        for split in ['train', 'val', 'test']:
            split_metrics = {}
            split_keys = [
                f'{split}_accuracy', f'{split}_precision', f'{split}_recall', f'{split}_f1_score',
                f'{split}_r2', f'{split}_r2_score', f'{split}_mse', f'{split}_mae', f'{split}_rmse',
                f'{split}_loss', f'{split}_samples', f'{split}_time_seconds'
            ]

            for key in split_keys:
                if key in metrics:
                    metric_name = key.replace(f'{split}_', '')
                    split_metrics[metric_name] = metrics[key]

            if split_metrics:
                if split == 'train':
                    comprehensive_metrics['training_metrics'] = split_metrics
                elif split == 'val':
                    comprehensive_metrics['validation_metrics'] = split_metrics
                elif split == 'test':
                    comprehensive_metrics['test_metrics'] = split_metrics

        # ===== HPO RESULTS =====
        hpo_data = result.get('hpo_results') or metrics.get('hpo_results')
        if hpo_data:
            comprehensive_metrics['hpo_results'] = {
                'method': hpo_data.get('method', 'unknown'),
                'optimization_rounds': hpo_data.get('optimization_rounds', 0),
                'total_trials': hpo_data.get('total_trials', 0),
                'best_overall_score': hpo_data.get('best_overall_score', None),
                'best_params': hpo_data.get('best_params', {}),
                'best_scores': hpo_data.get('best_scores', {}),
                'optimization_time': hpo_data.get('optimization_time', 0),
                'per_model_trials': hpo_data.get('per_model_trials', {})
            }

        # ===== FEATURE IMPORTANCE =====
        feature_importance_data = result.get('feature_importance') or metrics.get('feature_importance')
        if feature_importance_data:
            comprehensive_metrics['feature_importance'] = feature_importance_data

        # ===== DATA QUALITY METRICS =====
        data_quality = metrics.get('data_quality', {})
        if data_quality:
            comprehensive_metrics['data_quality'] = data_quality

        # Add basic data stats if available
        # Note: Top-level metrics override nested data_quality values if duplicates exist
        if 'feature_count' in metrics:
            comprehensive_metrics['data_quality']['feature_count'] = metrics['feature_count']
        if 'sample_count' in metrics:
            comprehensive_metrics['data_quality']['sample_count'] = metrics['sample_count']
        if 'missing_values_pct' in metrics:
            comprehensive_metrics['data_quality']['missing_values_pct'] = metrics['missing_values_pct']

        # ===== MODEL COMPLEXITY =====
        complexity_keys = [
            'total_parameters', 'trainable_parameters', 'model_size_mb',
            'inference_time_ms', 'memory_usage_mb'
        ]
        for key in complexity_keys:
            if key in metrics:
                comprehensive_metrics['model_complexity'][key] = metrics[key]

        # ===== PREDICTION STATISTICS =====
        pred_stats_keys = [
            'prediction_mean', 'prediction_std', 'prediction_min', 'prediction_max',
            'prediction_skewness', 'prediction_kurtosis',
            'true_positive_rate', 'false_positive_rate', 'true_negative_rate', 'false_negative_rate',
            'confusion_matrix'
        ]
        for key in pred_stats_keys:
            if key in metrics:
                comprehensive_metrics['prediction_statistics'][key] = metrics[key]

        # ===== ENSEMBLE-SPECIFIC METRICS =====
        if 'ensemble' in training_type and comprehensive_metrics['ensemble_specific'] is not None:
            ensemble_keys = [
                'ensemble_diversity', 'ensemble_agreement', 'stacking_improvement',
                'base_models_count', 'meta_model_type', 'meta_model_accuracy',
                'weighted_voting_accuracy', 'simple_voting_accuracy'
            ]
            for key in ensemble_keys:
                if key in metrics:
                    comprehensive_metrics['ensemble_specific'][key] = metrics[key]

        # ===== WALK-FORWARD VALIDATION RESULTS =====
        if hasattr(self, '_walkforward_config') and self._walkforward_config:
            wf_metrics = {
                'n_folds': len(self._walkforward_config.folds),
                'strategy': self._walkforward_config.strategy,
                'embargo_days': self._walkforward_config.embargo_days,
                'per_fold_metrics': {}
            }

            # Extract per-fold metrics if available
            for i, fold in enumerate(self._walkforward_config.folds, 1):
                fold_key = f'fold_{i}'
                fold_metrics = {}
                for metric_name in ['accuracy', 'r2', 'mse', 'mae', 'loss']:
                    key = f'{fold_key}_{metric_name}'
                    if key in metrics:
                        fold_metrics[metric_name] = metrics[key]

                if fold_metrics:
                    wf_metrics['per_fold_metrics'][fold_key] = fold_metrics

            comprehensive_metrics['walkforward_results'] = wf_metrics

        # ===== ERROR ANALYSIS =====
        error_keys = [
            'max_error', 'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error',
            'median_absolute_error', 'explained_variance_score',
            'directional_accuracy', 'sign_accuracy'
        ]
        for key in error_keys:
            if key in metrics:
                comprehensive_metrics['error_analysis'][key] = metrics[key]

        # ===== DATA DRIFT / DISTRIBUTION SHIFT CHECKS =====
        # Detects if train/val/test distributions differ significantly
        # Critical for ensuring models aren't learning time leakage or anomalies
        drift_keys = [
            'ks_test_train_val', 'ks_test_train_test', 'ks_test_val_test',
            'ks_statistic_train_val', 'ks_statistic_train_test', 'ks_statistic_val_test',
            'ks_pvalue_train_val', 'ks_pvalue_train_test', 'ks_pvalue_val_test',
            'psi_train_val', 'psi_train_test', 'psi_val_test',
            'psi_score', 'psi_interpretation',
            'chi_square_train_val', 'chi_square_train_test', 'chi_square_val_test',
            'chi_square_statistic', 'chi_square_pvalue',
            'wasserstein_distance_train_val', 'wasserstein_distance_train_test',
            'jensen_shannon_divergence_train_val', 'jensen_shannon_divergence_train_test',
            'drift_detected', 'drift_score', 'drift_features',
            'covariate_shift_detected', 'concept_drift_detected'
        ]
        for key in drift_keys:
            if key in metrics:
                comprehensive_metrics['data_drift_checks'][key] = metrics[key]

        # Also check per-model drift metrics
        for model_name in models.keys():
            for drift_metric in ['ks_test', 'psi', 'chi_square', 'drift_score']:
                key = f"{model_name}_{drift_metric}"
                if key in metrics:
                    if f'{model_name}_drift' not in comprehensive_metrics['data_drift_checks']:
                        comprehensive_metrics['data_drift_checks'][f'{model_name}_drift'] = {}
                    comprehensive_metrics['data_drift_checks'][f'{model_name}_drift'][drift_metric] = metrics[key]

        # ===== UNCERTAINTY / CONFIDENCE CALIBRATION =====
        # Measures how well predicted probabilities match actual outcomes
        # Critical for decision-making confidence in production
        calibration_keys = [
            'brier_score', 'brier_score_loss',
            'expected_calibration_error', 'ece', 'ece_score',
            'maximum_calibration_error', 'mce', 'mce_score',
            'calibration_curve', 'reliability_diagram',
            'calibration_slope', 'calibration_intercept',
            'log_loss', 'cross_entropy_loss',
            'prediction_confidence_mean', 'prediction_confidence_std',
            'prediction_confidence_median',
            'overconfidence_ratio', 'underconfidence_ratio',
            'confidence_histogram', 'reliability_bins',
            'sharpness', 'refinement',
            'calibration_in_the_large', 'calibration_in_the_small'
        ]
        for key in calibration_keys:
            if key in metrics:
                comprehensive_metrics['uncertainty_calibration'][key] = metrics[key]

        # Per-model calibration metrics
        for model_name in models.keys():
            for calib_metric in ['brier_score', 'ece', 'mce', 'log_loss', 'calibration_slope']:
                key = f"{model_name}_{calib_metric}"
                if key in metrics:
                    if f'{model_name}_calibration' not in comprehensive_metrics['uncertainty_calibration']:
                        comprehensive_metrics['uncertainty_calibration'][f'{model_name}_calibration'] = {}
                    comprehensive_metrics['uncertainty_calibration'][f'{model_name}_calibration'][calib_metric] = metrics[key]

        # ===== SHAPLEY-BASED EXPLANATIONS (SHAP) =====
        # Model interpretability and feature attribution
        # Note: Complex objects like plots are stored separately, only metadata here
        shap_keys = [
            'shap_values_available', 'shap_summary_plot_path', 'shap_dependence_plot_path',
            'shap_force_plot_path', 'shap_waterfall_plot_path',
            'shap_feature_importance', 'shap_interaction_values',
            'shap_top_features', 'shap_top_10_features', 'shap_top_20_features',
            'pdp_plots_path', 'ice_plots_path',
            'pdp_features', 'ice_features',
            'partial_dependence_available', 'individual_conditional_expectation_available',
            'lime_explanations_available', 'lime_top_features',
            'global_feature_importance', 'local_feature_importance'
        ]
        for key in shap_keys:
            if key in metrics:
                comprehensive_metrics['shap_explanations'][key] = metrics[key]

        # Per-model SHAP data
        for model_name in models.keys():
            for shap_metric in ['shap_values', 'shap_feature_importance', 'shap_summary_plot_path']:
                key = f"{model_name}_{shap_metric}"
                if key in metrics:
                    if f'{model_name}_shap' not in comprehensive_metrics['shap_explanations']:
                        comprehensive_metrics['shap_explanations'][f'{model_name}_shap'] = {}
                    comprehensive_metrics['shap_explanations'][f'{model_name}_shap'][shap_metric] = metrics[key]

        # ===== DECISION THRESHOLD OPTIMIZATION =====
        # Optimizing classification thresholds for business objectives
        threshold_keys = [
            'optimal_threshold', 'optimal_threshold_roc', 'optimal_threshold_pr',
            'optimal_threshold_f1', 'optimal_threshold_fbeta',
            'roc_auc_score', 'roc_curve', 'roc_curve_path',
            'pr_auc_score', 'precision_recall_curve', 'pr_curve_path',
            'f1_threshold_curve', 'fbeta_threshold_curve',
            'fbeta_score', 'fbeta_optimal', 'beta_value',
            'cost_matrix', 'cost_weighted_threshold', 'expected_cost',
            'profit_curve', 'profit_optimal_threshold',
            'youden_index', 'youden_threshold',
            'sensitivity_specificity_curve',
            'Matthews_correlation_coefficient', 'mcc', 'mcc_threshold',
            'threshold_metrics', 'threshold_analysis',
            'business_metric_optimal_threshold', 'custom_metric_threshold'
        ]
        for key in threshold_keys:
            if key in metrics:
                comprehensive_metrics['threshold_optimization'][key] = metrics[key]

        # Per-model threshold optimization
        for model_name in models.keys():
            for thresh_metric in ['optimal_threshold', 'roc_auc', 'pr_auc', 'f1_threshold', 'fbeta_optimal']:
                key = f"{model_name}_{thresh_metric}"
                if key in metrics:
                    if f'{model_name}_threshold' not in comprehensive_metrics['threshold_optimization']:
                        comprehensive_metrics['threshold_optimization'][f'{model_name}_threshold'] = {}
                    comprehensive_metrics['threshold_optimization'][f'{model_name}_threshold'][thresh_metric] = metrics[key]

        return comprehensive_metrics

    def _generate_csv_metrics_report(
        self,
        comprehensive_metrics: Dict[str, Any],
        result: Dict[str, Any],
        training_type: str,
        config: Dict[str, Any],
        reports_dir: str
    ) -> Optional[str]:
        """
        Generate a CSV file with one line per model containing all metrics.

        This CSV excludes complex objects (plots, curves, matrices) but includes
        all numeric and string metrics for easy analysis in spreadsheets/data tools.

        Args:
            comprehensive_metrics: Extracted comprehensive metrics
            result: Training result dictionary
            training_type: Type of training
            config: Configuration dictionary
            reports_dir: Directory to save the CSV

        Returns:
            Path to the generated CSV file, or None if failed
        """
        try:
            import csv

            csv_path = os.path.join(reports_dir, f'{training_type}_metrics.csv')
            models = result.get('models', {})

            # If no models, create a single row for the training run
            if not models:
                models = {training_type: None}

            # Collect all possible column names from comprehensive metrics
            csv_columns = []
            csv_rows = []

            # Fixed columns (metadata)
            # NOTE: timestamp is ISO format (YYYY-MM-DDTHH:MM:SS) for sorting
            #       training_date is YYYY-MM-DD for daily grouping
            fixed_columns = [
                'timestamp',                # ISO format, sortable, identifies training run
                'training_date',            # Date only (YYYY-MM-DD) for grouping by day
                'model_name',
                'training_type',
                'symbol',
                'timeframe',
                'direction',
                'execution_time_seconds',
                'success',
                'models_trained_count'
            ]
            csv_columns.extend(fixed_columns)

            # Helper function to flatten nested dicts and filter out complex objects
            def flatten_metrics(metrics_dict: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
                """Flatten nested dict, exclude complex objects."""
                flat = {}
                for key, value in metrics_dict.items():
                    full_key = f"{prefix}{key}" if prefix else key

                    # Skip complex objects that can't be represented in one CSV cell
                    if key in ['shap_values', 'shap_summary_plot_path', 'shap_dependence_plot_path',
                               'shap_force_plot_path', 'shap_waterfall_plot_path',
                               'pdp_plots_path', 'ice_plots_path',
                               'roc_curve', 'pr_curve', 'calibration_curve',
                               'reliability_diagram', 'confusion_matrix',
                               'cost_matrix', 'reliability_bins', 'confidence_histogram',
                               'cv_scores', 'threshold_metrics']:
                        continue

                    # Skip plot/curve paths (too long for CSV)
                    if 'plot_path' in key or 'curve_path' in key or 'plots_path' in key:
                        continue

                    if isinstance(value, dict):
                        # Recursively flatten nested dicts
                        flat.update(flatten_metrics(value, f"{full_key}_"))
                    elif isinstance(value, (list, tuple)):
                        # Skip lists/arrays (can't represent in single cell easily)
                        if len(value) <= 5 and all(isinstance(x, (int, float, str, bool)) for x in value):
                            # Only include short lists of simple types
                            flat[full_key] = str(value)
                    elif isinstance(value, (int, float, str, bool, type(None))):
                        flat[full_key] = value
                    else:
                        # Skip complex types
                        continue

                return flat

            # Build rows for each model
            for model_name in models.keys():
                row = {}

                # Add fixed metadata (timestamp fields first for easy sorting)
                timestamp_str = comprehensive_metrics.get('timestamp', datetime.now().isoformat())
                row['timestamp'] = timestamp_str

                # Extract date part for daily grouping (YYYY-MM-DD)
                try:
                    if 'T' in timestamp_str:
                        row['training_date'] = timestamp_str.split('T')[0]
                    else:
                        row['training_date'] = datetime.now().strftime('%Y-%m-%d')
                except Exception:
                    row['training_date'] = datetime.now().strftime('%Y-%m-%d')

                row['model_name'] = model_name
                row['training_type'] = training_type
                row['symbol'] = config.get('symbol', 'UNKNOWN')
                row['timeframe'] = config.get('timeframe', '15m')
                row['direction'] = config.get('direction', 'long')
                row['execution_time_seconds'] = comprehensive_metrics['execution_summary'].get('execution_time_seconds', 0)
                row['success'] = comprehensive_metrics['execution_summary'].get('success', False)
                row['models_trained_count'] = comprehensive_metrics['execution_summary'].get('models_trained_count', 0)

                # Add flattened metrics from all categories
                categories = [
                    'overall_performance',
                    'training_metrics',
                    'validation_metrics',
                    'test_metrics',
                    'data_quality',
                    'model_complexity',
                    'prediction_statistics',
                    'error_analysis',
                    'data_drift_checks',
                    'uncertainty_calibration',
                    'threshold_optimization'
                ]

                for category in categories:
                    if category in comprehensive_metrics:
                        flat_cat = flatten_metrics(comprehensive_metrics[category], f"{category}_")
                        row.update(flat_cat)

                # Add per-model metrics if available
                if model_name in comprehensive_metrics.get('per_model_metrics', {}):
                    model_specific = flatten_metrics(
                        comprehensive_metrics['per_model_metrics'][model_name],
                        'model_specific_'
                    )
                    row.update(model_specific)

                # Add HPO results (flattened)
                if comprehensive_metrics.get('hpo_results'):
                    hpo_flat = flatten_metrics(comprehensive_metrics['hpo_results'], 'hpo_')
                    # Exclude large nested structures
                    hpo_flat = {k: v for k, v in hpo_flat.items()
                                if not k.endswith('_best_params') and not k.endswith('_best_scores')}
                    row.update(hpo_flat)

                # Add walk-forward metrics (simplified)
                if comprehensive_metrics.get('walkforward_results'):
                    wf = comprehensive_metrics['walkforward_results']
                    if 'n_folds' in wf:
                        row['walkforward_n_folds'] = wf['n_folds']
                    if 'strategy' in wf:
                        row['walkforward_strategy'] = wf['strategy']
                    if 'embargo_days' in wf:
                        row['walkforward_embargo_days'] = wf['embargo_days']

                # Add ensemble-specific metrics if applicable
                if comprehensive_metrics.get('ensemble_specific'):
                    ensemble_flat = flatten_metrics(comprehensive_metrics['ensemble_specific'], 'ensemble_')
                    row.update(ensemble_flat)

                # Collect all column names
                for key in row.keys():
                    if key not in csv_columns:
                        csv_columns.append(key)

                csv_rows.append(row)

            # ========================================================================
            # 1. Write Per-Run CSV (timestamped, one file per training run)
            # ========================================================================
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns, extrasaction='ignore')
                writer.writeheader()

                for row in csv_rows:
                    # Ensure all columns exist in row (fill missing with None)
                    complete_row = {col: row.get(col, None) for col in csv_columns}
                    writer.writerow(complete_row)

            tprint_success(f"✅ CSV metrics report saved: {csv_path}")

            # ========================================================================
            # 2. Append to Consolidated CSV (aggregates ALL models across ALL runs)
            # ========================================================================
            # This allows aggregation when running analyst/tactician base/ensemble separately
            symbol = config.get('symbol', 'UNKNOWN')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')

            # Consolidated CSV path (at symbol level, not timestamped)
            consolidated_dir = os.path.join('reports', f"{symbol}_{timeframe}_{direction}")
            os.makedirs(consolidated_dir, exist_ok=True)
            consolidated_csv_path = os.path.join(consolidated_dir, 'all_models_metrics.csv')

            # Check if consolidated CSV exists to determine if we need headers
            file_exists = os.path.exists(consolidated_csv_path)

            # If file exists, read existing headers to ensure compatibility
            existing_columns = []
            if file_exists:
                try:
                    with open(consolidated_csv_path, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        existing_columns = reader.fieldnames if reader.fieldnames else []
                except Exception as e:
                    self.logger.warning(f"Could not read existing consolidated CSV headers: {e}")
                    existing_columns = []

            # Merge column sets (existing + new) to ensure all columns are present
            if existing_columns:
                # Create union of column sets, preserving order
                all_columns = list(existing_columns)
                for col in csv_columns:
                    if col not in all_columns:
                        all_columns.append(col)
            else:
                all_columns = csv_columns

            # If columns were added, we need to rewrite the file with expanded headers
            if file_exists and existing_columns and set(all_columns) != set(existing_columns):
                tprint_info(f"📊 Expanding consolidated CSV with new columns...")

                # Read existing data
                existing_data = []
                try:
                    with open(consolidated_csv_path, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        existing_data = list(reader)
                except Exception as e:
                    self.logger.error(f"Failed to read existing CSV for column expansion: {e}")
                    existing_data = []

                # Rewrite with expanded columns
                if existing_data:
                    with open(consolidated_csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction='ignore')
                        writer.writeheader()
                        for old_row in existing_data:
                            complete_row = {col: old_row.get(col, None) for col in all_columns}
                            writer.writerow(complete_row)
                    file_exists = True  # File has been rewritten

            # Append new rows to consolidated CSV
            try:
                # Use file locking to prevent concurrent write issues
                import fcntl
                has_fcntl = True
            except ImportError:
                # fcntl not available on Windows
                has_fcntl = False

            mode = 'a' if file_exists else 'w'
            with open(consolidated_csv_path, mode, newline='', encoding='utf-8') as csvfile:
                # Apply file lock if available (Unix-like systems)
                if has_fcntl:
                    try:
                        fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)
                    except Exception as e:
                        self.logger.warning(f"Could not acquire file lock: {e}")

                writer = csv.DictWriter(csvfile, fieldnames=all_columns, extrasaction='ignore')

                # Write header only if file is new
                if not file_exists:
                    writer.writeheader()

                # Append all rows from this training run
                for row in csv_rows:
                    complete_row = {col: row.get(col, None) for col in all_columns}
                    writer.writerow(complete_row)

                # Release lock (automatic when file closes, but explicit for clarity)
                if has_fcntl:
                    try:
                        fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass

            tprint_success(f"✅ Consolidated CSV updated: {consolidated_csv_path}")
            tprint_info(f"   ↪ Added {len(csv_rows)} model(s) to consolidated metrics")

            return csv_path

        except Exception as e:
            self.logger.error(f"Failed to generate CSV metrics report: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _generate_training_reports(
        self,
        result: Dict[str, Any],
        training_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate comprehensive markdown and JSON reports for training metrics.

        This is the centralized reporting hub that creates detailed reports
        with as many metrics as possible for each of the 4 model types:
        - Analyst Base
        - Analyst Ensemble
        - Tactician Base
        - Tactician Ensemble

        Args:
            result: Training result dictionary containing metrics and models
            training_type: Type of training (analyst_base, tactician_base, etc.)
            config: Configuration dictionary

        Returns:
            Dictionary with paths to generated reports
        """
        try:
            import json

            report_paths = {}

            # Extract comprehensive metrics using centralized extractor
            comprehensive_metrics = self._extract_comprehensive_metrics(result, training_type, config)

            # Create reports directory
            symbol = config.get('symbol', 'UNKNOWN')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            reports_dir = os.path.join(
                'reports',
                training_type,
                f"{symbol}_{timeframe}_{direction}",
                timestamp
            )
            os.makedirs(reports_dir, exist_ok=True)

            # ========================================================================
            # COMPREHENSIVE MARKDOWN REPORT WITH ALL METRICS
            # ========================================================================
            markdown_path = os.path.join(reports_dir, f'{training_type}_comprehensive_report.md')

            with open(markdown_path, 'w') as f:
                # ===== HEADER =====
                f.write(f"# {training_type.replace('_', ' ').title()} - Comprehensive Training Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")

                # ===== EXECUTION SUMMARY =====
                f.write("## 📋 Execution Summary\n\n")
                exec_summary = comprehensive_metrics['execution_summary']
                f.write(f"- **Training Type:** {exec_summary.get('training_type', 'N/A')}\n")
                f.write(f"- **Success:** {'✅ Yes' if exec_summary.get('success', False) else '❌ No'}\n")
                f.write(f"- **Execution Time:** {exec_summary.get('execution_time_seconds', 0):.2f} seconds\n")
                f.write(f"- **Models Trained:** {exec_summary.get('models_trained_count', 0)}\n")
                if exec_summary.get('model_names'):
                    f.write(f"- **Model Names:** {', '.join(exec_summary['model_names'])}\n")
                if exec_summary.get('error'):
                    f.write(f"- **Error:** {exec_summary['error']}\n")
                if exec_summary.get('warnings'):
                    f.write(f"- **Warnings:** {len(exec_summary['warnings'])} warning(s)\n")
                f.write("\n---\n\n")

                # ===== CONFIGURATION =====
                f.write("## ⚙️ Configuration\n\n")
                f.write(f"- **Symbol:** {symbol}\n")
                f.write(f"- **Exchange:** {config.get('exchange', 'binance')}\n")
                f.write(f"- **Timeframe:** {timeframe}\n")
                f.write(f"- **Direction:** {direction}\n")
                f.write(f"- **Execution Mode:** {config.get('execution_mode', 'full')}\n")
                f.write(f"- **HPO Enabled:** {config.get('enable_hpo', True)}\n")
                f.write("\n---\n\n")

                # ===== OVERALL PERFORMANCE METRICS =====
                f.write("## 📊 Overall Performance Metrics\n\n")
                overall_perf = comprehensive_metrics['overall_performance']
                if overall_perf:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(overall_perf.items()):
                        label = key.replace('overall_', '').replace('_', ' ').title()
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")
                else:
                    f.write("*No overall performance metrics available.*\n\n")
                f.write("---\n\n")

                # ===== TRAINING/VALIDATION/TEST SPLIT METRICS =====
                f.write("## 📈 Split-Based Performance Metrics\n\n")

                # Training Metrics
                f.write("### Training Set Metrics\n\n")
                train_metrics = comprehensive_metrics['training_metrics']
                if train_metrics:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(train_metrics.items()):
                        label = key.replace('_', ' ').title()
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")
                else:
                    f.write("*No training metrics available.*\n\n")

                # Validation Metrics
                f.write("### Validation Set Metrics\n\n")
                val_metrics = comprehensive_metrics['validation_metrics']
                if val_metrics:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(val_metrics.items()):
                        label = key.replace('_', ' ').title()
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")
                else:
                    f.write("*No validation metrics available.*\n\n")

                # Test Metrics
                f.write("### Test Set Metrics\n\n")
                test_metrics = comprehensive_metrics['test_metrics']
                if test_metrics:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(test_metrics.items()):
                        label = key.replace('_', ' ').title()
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")
                else:
                    f.write("*No test metrics available.*\n\n")
                f.write("---\n\n")

                # ===== PER-MODEL DETAILED METRICS =====
                f.write("## 🤖 Per-Model Detailed Metrics\n\n")
                per_model = comprehensive_metrics['per_model_metrics']
                if per_model:
                    f.write(f"**Total Models:** {len(per_model)}\n\n")
                    for model_name, model_metrics in per_model.items():
                        f.write(f"### {model_name.upper()}\n\n")
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        for key, value in sorted(model_metrics.items()):
                            label = key.replace('_', ' ').title()
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                f.write(f"| {label} | {value:.6f} |\n")
                            else:
                                f.write(f"| {label} | {value} |\n")
                        f.write("\n")
                else:
                    f.write("*No per-model metrics available.*\n\n")
                f.write("---\n\n")

                # ===== HPO RESULTS =====
                f.write("## 🔍 Hyperparameter Optimization (HPO) Results\n\n")
                hpo_results = comprehensive_metrics['hpo_results']
                if hpo_results and hpo_results.get('method'):
                    f.write(f"**Method:** {hpo_results.get('method', 'N/A')}\n")
                    f.write(f"**Optimization Rounds:** {hpo_results.get('optimization_rounds', 0)}\n")
                    f.write(f"**Total Trials:** {hpo_results.get('total_trials', 0)}\n")
                    f.write(f"**Best Overall Score:** {hpo_results.get('best_overall_score', 'N/A')}\n")
                    f.write(f"**Optimization Time:** {hpo_results.get('optimization_time', 0):.2f}s\n\n")

                    if hpo_results.get('best_params'):
                        f.write("### Best Parameters by Model\n\n")
                        for model_name, params in hpo_results['best_params'].items():
                            f.write(f"#### {model_name.upper()}\n\n")
                            if isinstance(params, dict):
                                f.write("```json\n")
                                f.write(json.dumps(params, indent=2))
                                f.write("\n```\n\n")

                    if hpo_results.get('best_scores'):
                        f.write("### Best Scores by Model\n\n")
                        f.write("| Model | Score |\n")
                        f.write("|-------|-------|\n")
                        for model_name, score_data in hpo_results['best_scores'].items():
                            if isinstance(score_data, dict):
                                score = score_data.get('score', 'N/A')
                            else:
                                score = score_data
                            f.write(f"| {model_name} | {score if isinstance(score, str) else f'{score:.6f}'} |\n")
                        f.write("\n")
                else:
                    f.write("*No HPO results available or HPO was disabled.*\n\n")
                f.write("---\n\n")

                # ===== WALK-FORWARD VALIDATION RESULTS =====
                f.write("## 📅 Walk-Forward Validation Results\n\n")
                wf_results = comprehensive_metrics['walkforward_results']
                if wf_results and wf_results.get('n_folds'):
                    f.write(f"**Number of Folds:** {wf_results.get('n_folds', 0)}\n")
                    f.write(f"**Strategy:** {wf_results.get('strategy', 'N/A')}\n")
                    f.write(f"**Embargo Days:** {wf_results.get('embargo_days', 0)}\n\n")

                    per_fold = wf_results.get('per_fold_metrics', {})
                    if per_fold:
                        f.write("### Per-Fold Metrics\n\n")
                        for fold_name, fold_metrics in sorted(per_fold.items()):
                            f.write(f"#### {fold_name.upper().replace('_', ' ')}\n\n")
                            f.write("| Metric | Value |\n")
                            f.write("|--------|-------|\n")
                            for key, value in sorted(fold_metrics.items()):
                                label = key.replace('_', ' ').title()
                                if isinstance(value, (int, float)):
                                    f.write(f"| {label} | {value:.6f} |\n")
                                else:
                                    f.write(f"| {label} | {value} |\n")
                            f.write("\n")
                else:
                    f.write("*No walk-forward validation results available.*\n\n")
                f.write("---\n\n")

                # ===== ENSEMBLE-SPECIFIC METRICS =====
                if comprehensive_metrics['ensemble_specific'] is not None:
                    f.write("## 🎯 Ensemble-Specific Metrics\n\n")
                    ensemble_metrics = comprehensive_metrics['ensemble_specific']
                    if ensemble_metrics:
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        for key, value in sorted(ensemble_metrics.items()):
                            label = key.replace('_', ' ').title()
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                f.write(f"| {label} | {value:.6f} |\n")
                            else:
                                f.write(f"| {label} | {value} |\n")
                        f.write("\n")
                    else:
                        f.write("*No ensemble-specific metrics available.*\n\n")
                    f.write("---\n\n")

                # ===== FEATURE IMPORTANCE =====
                f.write("## 📋 Feature Importance\n\n")
                feat_importance = comprehensive_metrics['feature_importance']
                if feat_importance:
                    if isinstance(feat_importance, dict):
                        f.write("### Top 20 Most Important Features\n\n")
                        # Sort by importance value (descending)
                        sorted_features = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)[:20]
                        f.write("| Rank | Feature | Importance |\n")
                        f.write("|------|---------|------------|\n")
                        for i, (feature, importance) in enumerate(sorted_features, 1):
                            if isinstance(importance, (int, float)):
                                f.write(f"| {i} | {feature} | {importance:.6f} |\n")
                            else:
                                f.write(f"| {i} | {feature} | {importance} |\n")
                        f.write("\n")
                else:
                    f.write("*No feature importance data available.*\n\n")
                f.write("---\n\n")

                # ===== DATA QUALITY METRICS =====
                f.write("## 📊 Data Quality Metrics\n\n")
                data_quality = comprehensive_metrics['data_quality']
                if data_quality:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(data_quality.items()):
                        label = key.replace('_', ' ').title()
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f if isinstance(value, float) else value} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")
                else:
                    f.write("*No data quality metrics available.*\n\n")
                f.write("---\n\n")

                # ===== MODEL COMPLEXITY =====
                f.write("## 🧮 Model Complexity Metrics\n\n")
                complexity = comprehensive_metrics['model_complexity']
                if complexity:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(complexity.items()):
                        label = key.replace('_', ' ').title()
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f if isinstance(value, float) else value} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")
                else:
                    f.write("*No model complexity metrics available.*\n\n")
                f.write("---\n\n")

                # ===== PREDICTION STATISTICS =====
                f.write("## 📊 Prediction Statistics\n\n")
                pred_stats = comprehensive_metrics['prediction_statistics']
                if pred_stats:
                    f.write("| Statistic | Value |\n")
                    f.write("|-----------|-------|\n")
                    for key, value in sorted(pred_stats.items()):
                        label = key.replace('_', ' ').title()
                        if key == 'confusion_matrix':
                            f.write(f"| {label} | See detailed analysis |\n")
                        elif isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")
                else:
                    f.write("*No prediction statistics available.*\n\n")
                f.write("---\n\n")

                # ===== ERROR ANALYSIS =====
                f.write("## ⚠️ Error Analysis\n\n")
                error_analysis = comprehensive_metrics['error_analysis']
                if error_analysis:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(error_analysis.items()):
                        label = key.replace('_', ' ').title()
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")
                else:
                    f.write("*No error analysis metrics available.*\n\n")
                f.write("---\n\n")

                # ===== DATA DRIFT / DISTRIBUTION SHIFT CHECKS =====
                f.write("## 📊 Data Drift & Distribution Shift Checks\n\n")
                f.write("*Detects if train/val/test distributions differ significantly (KS tests, PSI, chi-square)*\n\n")
                drift_checks = comprehensive_metrics['data_drift_checks']
                if drift_checks:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(drift_checks.items()):
                        label = key.replace('_', ' ').title()
                        if isinstance(value, dict):
                            # Per-model drift metrics
                            f.write(f"| **{label}** | (see details below) |\n")
                        elif isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")

                    # Detailed per-model drift if available
                    per_model_drift = {k: v for k, v in drift_checks.items() if isinstance(v, dict)}
                    if per_model_drift:
                        f.write("### Per-Model Drift Metrics\n\n")
                        for model_name, drift_data in per_model_drift.items():
                            f.write(f"#### {model_name.replace('_', ' ').upper()}\n\n")
                            f.write("| Metric | Value |\n")
                            f.write("|--------|-------|\n")
                            for key, value in sorted(drift_data.items()):
                                label = key.replace('_', ' ').title()
                                if isinstance(value, (int, float)) and not isinstance(value, bool):
                                    f.write(f"| {label} | {value:.6f} |\n")
                                else:
                                    f.write(f"| {label} | {value} |\n")
                            f.write("\n")
                else:
                    f.write("*No data drift checks available.*\n\n")
                f.write("---\n\n")

                # ===== UNCERTAINTY / CONFIDENCE CALIBRATION =====
                f.write("## 🎯 Uncertainty & Confidence Calibration\n\n")
                f.write("*Measures how well predicted probabilities match actual outcomes (Brier Score, ECE)*\n\n")
                calibration = comprehensive_metrics['uncertainty_calibration']
                if calibration:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for key, value in sorted(calibration.items()):
                        label = key.replace('_', ' ').title()
                        if isinstance(value, dict):
                            # Per-model calibration metrics
                            f.write(f"| **{label}** | (see details below) |\n")
                        elif isinstance(value, (int, float)) and not isinstance(value, bool):
                            f.write(f"| {label} | {value:.6f} |\n")
                        else:
                            f.write(f"| {label} | {value} |\n")
                    f.write("\n")

                    # Detailed per-model calibration if available
                    per_model_calib = {k: v for k, v in calibration.items() if isinstance(v, dict)}
                    if per_model_calib:
                        f.write("### Per-Model Calibration Metrics\n\n")
                        for model_name, calib_data in per_model_calib.items():
                            f.write(f"#### {model_name.replace('_', ' ').upper()}\n\n")
                            f.write("| Metric | Value |\n")
                            f.write("|--------|-------|\n")
                            for key, value in sorted(calib_data.items()):
                                label = key.replace('_', ' ').title()
                                if isinstance(value, (int, float)) and not isinstance(value, bool):
                                    f.write(f"| {label} | {value:.6f} |\n")
                                else:
                                    f.write(f"| {label} | {value} |\n")
                            f.write("\n")
                else:
                    f.write("*No uncertainty/calibration metrics available.*\n\n")
                f.write("---\n\n")

                # ===== SHAPLEY-BASED EXPLANATIONS (SHAP) =====
                f.write("## 🔍 SHAP Explanations & Model Interpretability\n\n")
                f.write("*Shapley values, PDP/ICE curves, and feature attribution*\n\n")
                shap_exp = comprehensive_metrics['shap_explanations']
                if shap_exp:
                    # Filter out plot paths and complex objects for main table
                    simple_shap = {k: v for k, v in shap_exp.items()
                                   if not isinstance(v, dict) and 'plot' not in k.lower() and 'curve' not in k.lower()}

                    if simple_shap:
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        for key, value in sorted(simple_shap.items()):
                            label = key.replace('_', ' ').title()
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                f.write(f"| {label} | {value:.6f} |\n")
                            else:
                                f.write(f"| {label} | {value} |\n")
                        f.write("\n")

                    # Plot paths if available
                    plot_paths = {k: v for k, v in shap_exp.items() if 'path' in k.lower()}
                    if plot_paths:
                        f.write("### Generated Explanation Plots\n\n")
                        for key, path in sorted(plot_paths.items()):
                            label = key.replace('_', ' ').replace('path', '').strip().title()
                            f.write(f"- **{label}:** `{path}`\n")
                        f.write("\n")

                    # Per-model SHAP data
                    per_model_shap = {k: v for k, v in shap_exp.items() if isinstance(v, dict)}
                    if per_model_shap:
                        f.write("### Per-Model SHAP Data\n\n")
                        for model_name, shap_data in per_model_shap.items():
                            f.write(f"#### {model_name.replace('_', ' ').upper()}\n\n")
                            for key, value in sorted(shap_data.items()):
                                label = key.replace('_', ' ').title()
                                if 'path' in key.lower():
                                    f.write(f"- **{label}:** `{value}`\n")
                                elif isinstance(value, (int, float)):
                                    f.write(f"- **{label}:** {value:.6f}\n")
                                else:
                                    f.write(f"- **{label}:** {value}\n")
                            f.write("\n")
                else:
                    f.write("*No SHAP explanations available.*\n\n")
                f.write("---\n\n")

                # ===== DECISION THRESHOLD OPTIMIZATION =====
                f.write("## ⚖️ Decision Threshold Optimization\n\n")
                f.write("*ROC/PR curves, F-beta optimization, cost-weighted thresholds*\n\n")
                threshold_opt = comprehensive_metrics['threshold_optimization']
                if threshold_opt:
                    # Filter out complex objects
                    simple_threshold = {k: v for k, v in threshold_opt.items()
                                        if not isinstance(v, dict) and 'curve' not in k.lower() and 'matrix' not in k.lower()}

                    if simple_threshold:
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        for key, value in sorted(simple_threshold.items()):
                            label = key.replace('_', ' ').title()
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                f.write(f"| {label} | {value:.6f} |\n")
                            else:
                                f.write(f"| {label} | {value} |\n")
                        f.write("\n")

                    # Per-model threshold optimization
                    per_model_threshold = {k: v for k, v in threshold_opt.items() if isinstance(v, dict)}
                    if per_model_threshold:
                        f.write("### Per-Model Threshold Optimization\n\n")
                        for model_name, thresh_data in per_model_threshold.items():
                            f.write(f"#### {model_name.replace('_', ' ').upper()}\n\n")
                            f.write("| Metric | Value |\n")
                            f.write("|--------|-------|\n")
                            for key, value in sorted(thresh_data.items()):
                                label = key.replace('_', ' ').title()
                                if isinstance(value, (int, float)) and not isinstance(value, bool):
                                    f.write(f"| {label} | {value:.6f} |\n")
                                else:
                                    f.write(f"| {label} | {value} |\n")
                            f.write("\n")
                else:
                    f.write("*No threshold optimization metrics available.*\n\n")
                f.write("---\n\n")

                # ===== ARTIFACTS =====
                f.write("## 💾 Generated Artifacts\n\n")
                if 'artifacts' in result:
                    artifacts = result['artifacts']
                    f.write("| Artifact Name | Path |\n")
                    f.write("|---------------|------|\n")
                    for artifact_name, artifact_path in sorted(artifacts.items()):
                        f.write(f"| {artifact_name} | `{artifact_path}` |\n")
                    f.write("\n")
                else:
                    f.write("*No artifacts information available.*\n\n")

                # ===== FOOTER =====
                f.write("---\n\n")
                f.write(f"*Comprehensive report generated by Ares Unified Training Pipeline v3.0 on {timestamp}*\n")
                f.write(f"*Training Type: {training_type.upper()} | Symbol: {symbol} | Timeframe: {timeframe} | Direction: {direction}*\n")

            report_paths['markdown'] = markdown_path
            tprint_success(f"✅ Markdown report saved: {markdown_path}")

            # ========================================================================
            # COMPREHENSIVE JSON REPORT WITH ALL METRICS
            # ========================================================================
            json_path = os.path.join(reports_dir, f'{training_type}_comprehensive_metrics.json')

            # Build comprehensive JSON report using the extracted metrics
            json_report = {
                'report_version': '3.0',
                'metadata': {
                    'training_type': training_type,
                    'symbol': symbol,
                    'exchange': config.get('exchange', 'binance'),
                    'timeframe': timeframe,
                    'direction': direction,
                    'execution_mode': config.get('execution_mode', 'full'),
                    'timestamp': timestamp,
                    'generated_at': datetime.now().isoformat(),
                    'report_type': 'comprehensive_training_metrics'
                },
                'configuration': {
                    'symbol': symbol,
                    'exchange': config.get('exchange', 'binance'),
                    'timeframe': timeframe,
                    'direction': direction,
                    'execution_mode': config.get('execution_mode', 'full'),
                    'enable_hpo': config.get('enable_hpo', True),
                    'train_percentage': config.get('train_percentage', 0.70),
                    'validation_percentage': config.get('validation_percentage', 0.15),
                    'test_percentage': config.get('test_percentage', 0.15),
                    'walkforward_config': str(config.get('walkforward_config', 'N/A'))
                },

                # ===== COMPREHENSIVE METRICS =====
                'execution_summary': comprehensive_metrics['execution_summary'],
                'overall_performance': comprehensive_metrics['overall_performance'],
                'per_model_metrics': comprehensive_metrics['per_model_metrics'],
                'training_metrics': comprehensive_metrics['training_metrics'],
                'validation_metrics': comprehensive_metrics['validation_metrics'],
                'test_metrics': comprehensive_metrics['test_metrics'],
                'hpo_results': comprehensive_metrics['hpo_results'],
                'walkforward_results': comprehensive_metrics['walkforward_results'],
                'feature_importance': comprehensive_metrics['feature_importance'],
                'data_quality': comprehensive_metrics['data_quality'],
                'model_complexity': comprehensive_metrics['model_complexity'],
                'prediction_statistics': comprehensive_metrics['prediction_statistics'],
                'error_analysis': comprehensive_metrics['error_analysis'],
                'data_drift_checks': comprehensive_metrics['data_drift_checks'],
                'uncertainty_calibration': comprehensive_metrics['uncertainty_calibration'],
                'shap_explanations': comprehensive_metrics['shap_explanations'],
                'threshold_optimization': comprehensive_metrics['threshold_optimization'],

                # Add ensemble-specific metrics if applicable
                'ensemble_specific': comprehensive_metrics['ensemble_specific'] if comprehensive_metrics['ensemble_specific'] is not None else None,

                # ===== RAW METRICS (for backward compatibility) =====
                'raw_metrics': result.get('metrics', {}),

                # ===== ARTIFACTS =====
                'artifacts': result.get('artifacts', {}),

                # ===== MODELS INFO =====
                'models': {
                    'count': len(result.get('models', {})),
                    'names': list(result.get('models', {}).keys()),
                    'details': {
                        model_name: {
                            'type': str(type(model).__name__) if hasattr(model, '__class__') else 'unknown'
                        }
                        for model_name, model in result.get('models', {}).items()
                    }
                }
            }

            # Save JSON report
            with open(json_path, 'w') as f:
                json.dump(json_report, f, indent=2, default=str)

            report_paths['json'] = json_path
            tprint_success(f"✅ Comprehensive JSON metrics saved: {json_path}")

            # ========================================================================
            # CSV METRICS REPORT (One line per model)
            # ========================================================================
            csv_path = self._generate_csv_metrics_report(
                comprehensive_metrics=comprehensive_metrics,
                result=result,
                training_type=training_type,
                config=config,
                reports_dir=reports_dir
            )
            if csv_path:
                report_paths['csv'] = csv_path

            # ========================================================================
            # SUMMARY LOG OUTPUT
            # ========================================================================
            tprint_info("=" * 80)
            tprint_info(f"📊 TRAINING REPORT SUMMARY - {training_type.upper()}")
            tprint_info("=" * 80)
            tprint_info(f"✅ Success: {comprehensive_metrics['execution_summary']['success']}")
            tprint_info(f"⏱️  Execution Time: {comprehensive_metrics['execution_summary']['execution_time_seconds']:.2f}s")
            tprint_info(f"🤖 Models Trained: {comprehensive_metrics['execution_summary']['models_trained_count']}")

            if comprehensive_metrics['overall_performance']:
                tprint_info("📈 Overall Performance:")
                for key, value in list(comprehensive_metrics['overall_performance'].items())[:5]:
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        tprint_info(f"   • {key}: {value:.4f}")
                    else:
                        tprint_info(f"   • {key}: {value}")

            tprint_info(f"📄 Markdown Report: {markdown_path}")
            tprint_info(f"📊 JSON Report: {json_path}")
            if csv_path:
                tprint_info(f"📊 CSV Report: {csv_path}")
            tprint_info("=" * 80)

            return report_paths

        except Exception as e:
            self.logger.error(f"Failed to generate training reports: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
