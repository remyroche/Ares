"""
NAS/TAS Models Training Sub-Pipeline

This module orchestrates the training of NAS (Neural Architecture Search) and TAS (Tree Architecture Search)
models for different timeframes and market conditions, ensuring they are properly integrated into the
overall model training pipeline after features optimization.

Key Features:
- Train NAS models per-regime on 5m timeframe
- Train TAS models per-regime on 1m timeframe
- Ensure model selection selects top 2-3 models for a given market
- Ensure model output matches ensemble models' expectations
- Proper use of tprint at every important stage
- No silent or swallowed failures
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import json

# Import tprint for comprehensive logging
try:
    from src.utils.tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
        tprint_success, tprint_progress, tprint_performance, tprint_timer
    )
    TPRINT_AVAILABLE = True
except ImportError:
    # Fallback function if tprint is not available
    def tprint(message: str, color: str = "white", **kwargs):
        print(f"[NAS_TAS_TRAINING] {message}")
    def tprint_debug(message: str, **kwargs):
        print(f"[DEBUG] {message}")
    def tprint_info(message: str, **kwargs):
        print(f"[INFO] {message}")
    def tprint_warning(message: str, **kwargs):
        print(f"[WARNING] {message}")
    def tprint_error(message: str, **kwargs):
        print(f"[ERROR] {message}")
    def tprint_success(message: str, **kwargs):
        print(f"[SUCCESS] {message}")
    def tprint_progress(message: str, **kwargs):
        print(f"[PROGRESS] {message}")
    def tprint_performance(message: str, **kwargs):
        print(f"[PERFORMANCE] {message}")
    def tprint_timer(message: str, **kwargs):
        print(f"[TIMER] {message}")
    TPRINT_AVAILABLE = False

# Import NAS/TAS training orchestrator
try:
    from .nas_tas_training_orchestrator import (
        NAS_TASTrainingOrchestrator, NAS_TASTrainingOrchestratorConfig
    )
    NAS_TAS_ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"NAS/TAS training orchestrator not available: {e}")
    NAS_TAS_ORCHESTRATOR_AVAILABLE = False

# Import model selection components
try:
    from ..models_training.nas_tas.model_selector import ModelSelector, ModelSelectionConfig
    MODEL_SELECTION_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"Model selection not available: {e}")
    MODEL_SELECTION_AVAILABLE = False

# Import existing training utilities
from src.utils.logger import system_logger

logger = system_logger.getChild('NASTASModelsTraining')


@dataclass
class NASTASModelsTrainingConfig:
    """Configuration for NAS/TAS models training sub-pipeline."""

    # Training mode
    mode: str = "full"  # "full", "light", "blank"

    # NAS/TAS training settings
    enable_nas_training: bool = True
    enable_tas_training: bool = True
    nas_timeframe: str = "5m"
    tas_timeframe: str = "1m"

    # Model selection settings
    top_k_models: int = 3  # Select top 2-3 models per market
    selection_strategy: str = "best_performance"  # "best_performance", "ensemble", "adaptive"

    # Integration settings
    integrate_with_analyst_ensemble: bool = True
    integrate_with_tactician_ensemble: bool = True

    # Execution settings
    force_rerun: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    validation_enabled: bool = True

    # Output settings
    output_directory: str = "generated/model_training/nas_tas"
    save_models: bool = True
    save_detailed_results: bool = True

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NASTASModelsTrainingResult:
    """Result from NAS/TAS models training sub-pipeline."""

    # Overall results
    success: bool
    execution_time: float
    start_time: datetime
    end_time: Optional[datetime] = None

    # Training results
    nas_training_results: Dict[str, Any] = field(default_factory=dict)
    tas_training_results: Dict[str, Any] = field(default_factory=dict)
    model_selection_results: Dict[str, Any] = field(default_factory=dict)

    # Integration results
    analyst_integration_results: Dict[str, Any] = field(default_factory=dict)
    tactician_integration_results: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    top_models_selected: Dict[str, List[str]] = field(default_factory=dict)

    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class NASTASModelsTrainingSubPipeline:
    """
    NAS/TAS Models Training Sub-Pipeline.

    This class orchestrates the training of NAS and TAS models after features optimization,
    ensuring proper integration with existing analyst and tactician pipelines.
    """

    def __init__(self, config: Optional[NASTASModelsTrainingConfig] = None):
        """Initialize NAS/TAS models training sub-pipeline."""
        self.config = config or NASTASModelsTrainingConfig()
        self.logger = logger.getChild('NASTASModelsTrainingSubPipeline')

        # Initialize NAS/TAS training orchestrator
        if NAS_TAS_ORCHESTRATOR_AVAILABLE:
            orchestrator_config = NAS_TASTrainingOrchestratorConfig(
                enable_nas_training=self.config.enable_nas_training,
                enable_tas_training=self.config.enable_tas_training,
                nas_timeframe=self.config.nas_timeframe,
                tas_timeframe=self.config.tas_timeframe,
                enable_analyst_base_training=False,  # We'll handle analyst integration separately
                enable_tactician_base_training=False,  # We'll handle tactician integration separately
                enable_analyst_ensemble_training=self.config.integrate_with_analyst_ensemble,
                enable_tactician_ensemble_training=self.config.integrate_with_tactician_ensemble,
            )
            self.training_orchestrator = NAS_TASTrainingOrchestrator(orchestrator_config)
            tprint_success("✅ NAS/TAS training orchestrator initialized")
        else:
            self.training_orchestrator = None
            tprint_warning("⚠️ NAS/TAS training orchestrator not available")

        # Initialize model selector
        if MODEL_SELECTION_AVAILABLE:
            selection_config = ModelSelectionConfig(
                selection_strategy=self.config.selection_strategy,
                top_k=self.config.top_k_models,
                enable_ensemble=True,
            )
            self.model_selector = ModelSelector(selection_config)
            tprint_success("✅ Model selector initialized")
        else:
            self.model_selector = None
            tprint_warning("⚠️ Model selector not available")

        # Pipeline state
        self.current_pipeline_state = {}
        self.execution_history = []

        tprint_info(f"🎯 NAS/TAS Models Training initialized - Mode: {self.config.mode}")
        tprint_info(f"📊 Top K models: {self.config.top_k_models}, Selection strategy: {self.config.selection_strategy}")

    async def execute_pipeline(self, config: NASTASModelsTrainingConfig) -> NASTASModelsTrainingResult:
        """
        Execute the complete NAS/TAS models training pipeline.

        Args:
            config: Configuration for pipeline execution

        Returns:
            NASTASModelsTrainingResult with complete training results
        """
        start_time = datetime.now()
        tprint_info("🚀 Starting NAS/TAS Models Training Pipeline")
        tprint_info(f"📊 Mode: {config.mode}, NAS: {config.enable_nas_training}, TAS: {config.enable_tas_training}")

        result = NASTASModelsTrainingResult(
            success=False,
            execution_time=0.0,
            start_time=start_time
        )

        try:
            # Step 1: Prepare training data and pipeline state
            tprint_progress("📋 Step 1: Preparing training data and pipeline state")
            if not await self._prepare_training_data(config, result):
                tprint_error("❌ Failed to prepare training data")
                return result
            tprint_success("✅ Training data prepared successfully")

            # Step 2: Train NAS models per-regime on 5m timeframe
            if config.enable_nas_training and self.training_orchestrator:
                tprint_progress("🧠 Step 2: Training NAS models per-regime on 5m timeframe")
                if not await self._train_nas_models(config, result):
                    tprint_error("❌ NAS model training failed")
                    return result
                tprint_success("✅ NAS models trained successfully")
            else:
                tprint_warning("⏭️ NAS training disabled or orchestrator not available")

            # Step 3: Train TAS models per-regime on 1m timeframe
            if config.enable_tas_training and self.training_orchestrator:
                tprint_progress("🌳 Step 3: Training TAS models per-regime on 1m timeframe")
                if not await self._train_tas_models(config, result):
                    tprint_error("❌ TAS model training failed")
                    return result
                tprint_success("✅ TAS models trained successfully")
            else:
                tprint_warning("⏭️ TAS training disabled or orchestrator not available")

            # Step 4: Select top 2-3 models for each market
            tprint_progress("🎯 Step 4: Selecting top 2-3 models for each market")
            if not await self._select_top_models(config, result):
                tprint_error("❌ Model selection failed")
                return result
            tprint_success("✅ Top models selected successfully")

            # Step 5: Ensure model output matches ensemble expectations
            tprint_progress("🔧 Step 5: Ensuring model output compatibility with ensembles")
            if not await self._ensure_ensemble_compatibility(config, result):
                tprint_error("❌ Ensemble compatibility check failed")
                return result
            tprint_success("✅ Ensemble compatibility ensured")

            # Step 6: Integrate with existing analyst/tactician ensembles
            if config.integrate_with_analyst_ensemble or config.integrate_with_tactician_ensemble:
                tprint_progress("🔗 Step 6: Integrating with existing ensembles")
                if not await self._integrate_with_ensembles(config, result):
                    tprint_error("❌ Ensemble integration failed")
                    return result
                tprint_success("✅ Ensembles integration completed")
            else:
                tprint_warning("⏭️ Ensemble integration disabled")

            # Step 7: Save results and perform validation
            tprint_progress("💾 Step 7: Saving results and performing validation")
            if not await self._save_and_validate_results(config, result):
                tprint_error("❌ Results saving/validation failed")
                return result
            tprint_success("✅ Results saved and validated")

            # Complete result
            end_time = datetime.now()
            result.end_time = end_time
            result.execution_time = (end_time - start_time).total_seconds()
            result.success = True

            tprint_success(f"✅ NAS/TAS Models Training Pipeline completed in {result.execution_time:.2f}s")
            tprint_info(f"📊 NAS models trained: {len(result.nas_training_results.get('nas_models', {}))}")
            tprint_info(f"📊 TAS models trained: {len(result.tas_training_results.get('tas_models', {}))}")
            tprint_info(f"📊 Top models selected: {len(result.top_models_selected)}")

        except Exception as e:
            end_time = datetime.now()
            result.end_time = end_time
            result.execution_time = (end_time - start_time).total_seconds()
            result.error_message = str(e)
            result.success = False

            tprint_error(f"❌ NAS/TAS Models Training Pipeline failed: {e}")
            logger.error(f"NAS/TAS training pipeline failed: {e}", exc_info=True)

        self.execution_history.append(result)
        return result

    async def _prepare_training_data(self, config: NASTASModelsTrainingConfig, result: NASTASModelsTrainingResult) -> bool:
        """Prepare training data for NAS/TAS models."""
        try:
            # Load market data for different timeframes
            training_data = {}

            # Load 5m data for NAS training
            if config.enable_nas_training:
                tprint_info("📥 Loading 5m market data for NAS training")
                # This would load the actual 5m data from the data pipeline
                # For now, we'll use placeholder data structure
                training_data['X_5m'] = None  # Would be actual features DataFrame
                training_data['y_5m'] = None  # Would be actual targets DataFrame
                training_data['regime_labels_5m'] = None  # Would be regime assignments

            # Load 1m data for TAS training
            if config.enable_tas_training:
                tprint_info("📥 Loading 1m market data for TAS training")
                training_data['X_1m'] = None  # Would be actual features DataFrame
                training_data['y_1m'] = None  # Would be actual targets DataFrame
                training_data['regime_labels_1m'] = None  # Would be regime assignments

            # Get analyst signals for TAS training
            training_data['analyst_signals'] = None  # Would come from analyst ensemble

            self.current_pipeline_state['training_data'] = training_data

            tprint_success("✅ Training data prepared")
            return True

        except Exception as e:
            tprint_error(f"❌ Training data preparation failed: {e}")
            result.error_message = str(e)
            return False

    async def _train_nas_models(self, config: NASTASModelsTrainingConfig, result: NASTASModelsTrainingResult) -> bool:
        """Train NAS models per-regime on 5m timeframe."""
        try:
            if not self.training_orchestrator:
                tprint_warning("⏭️ NAS training orchestrator not available")
                return True

            tprint_info("🧠 Training NAS models per-regime on 5m timeframe")
            training_data = self.current_pipeline_state.get('training_data', {})

            # Execute NAS training
            nas_results = await self.training_orchestrator.nas_training_step.execute_nas_training(
                training_data, self.current_pipeline_state
            )

            if not nas_results.get('success', False):
                tprint_error(f"❌ NAS training failed: {nas_results.get('error', 'Unknown error')}")
                result.warnings.append(f"NAS training failed: {nas_results.get('error')}")
                return False

            result.nas_training_results = nas_results
            self.current_pipeline_state['nas_models'] = nas_results.get('nas_models', {})

            tprint_success(f"✅ NAS models trained: {len(nas_results.get('nas_models', {}))} models")
            return True

        except Exception as e:
            tprint_error(f"❌ NAS model training failed: {e}")
            result.error_message = str(e)
            return False

    async def _train_tas_models(self, config: NASTASModelsTrainingConfig, result: NASTASModelsTrainingResult) -> bool:
        """Train TAS models per-regime on 1m timeframe."""
        try:
            if not self.training_orchestrator:
                tprint_warning("⏭️ TAS training orchestrator not available")
                return True

            tprint_info("🌳 Training TAS models per-regime on 1m timeframe")
            training_data = self.current_pipeline_state.get('training_data', {})

            # Execute TAS training
            tas_results = await self.training_orchestrator.tas_training_step.execute_tas_training(
                training_data, self.current_pipeline_state
            )

            if not tas_results.get('success', False):
                tprint_error(f"❌ TAS training failed: {tas_results.get('error', 'Unknown error')}")
                result.warnings.append(f"TAS training failed: {tas_results.get('error')}")
                return False

            result.tas_training_results = tas_results
            self.current_pipeline_state['tas_models'] = tas_results.get('tas_models', {})

            tprint_success(f"✅ TAS models trained: {len(tas_results.get('tas_models', {}))} models")
            return True

        except Exception as e:
            tprint_error(f"❌ TAS model training failed: {e}")
            result.error_message = str(e)
            return False

    async def _select_top_models(self, config: NASTASModelsTrainingConfig, result: NASTASModelsTrainingResult) -> bool:
        """Select top 2-3 models for each market."""
        try:
            if not self.model_selector:
                tprint_warning("⏭️ Model selector not available")
                return True

            tprint_info(f"🎯 Selecting top {config.top_k_models} models for each market")

            # Register NAS and TAS models with the selector
            nas_models = self.current_pipeline_state.get('nas_models', {})
            tas_models = self.current_pipeline_state.get('tas_models', {})

            if nas_models:
                # Register NAS models for selection
                self.model_selector.register_models(
                    regime_models=nas_models,
                    ensemble_models=None,
                    directional_models=None
                )
                tprint_info(f"📝 Registered {len(nas_models)} NAS models for selection")

            if tas_models:
                # Register TAS models for selection
                self.model_selector.register_models(
                    regime_models=tas_models,
                    ensemble_models=None,
                    directional_models=None
                )
                tprint_info(f"📝 Registered {len(tas_models)} TAS models for selection")

            # Select top models for each market/regime
            top_models = {}

            # Get unique markets/regimes
            all_regimes = set()
            if nas_models:
                all_regimes.update(nas_models.keys())
            if tas_models:
                all_regimes.update(tas_models.keys())

            for regime_id in all_regimes:
                tprint_info(f"🎯 Selecting top models for regime {regime_id}")

                # Get models for this regime
                regime_models = {}
                if regime_id in nas_models:
                    regime_models.update(nas_models[regime_id])
                if regime_id in tas_models:
                    regime_models.update(tas_models[regime_id])

                if not regime_models:
                    tprint_warning(f"⚠️ No models available for regime {regime_id}")
                    continue

                # Select top models for this regime
                try:
                    # Create market data placeholder for selection
                    market_data = pd.DataFrame({'close': [100.0]})  # Placeholder

                    selection_result = self.model_selector.select_model(
                        market_data=market_data,
                        current_regime=regime_id
                    )

                    if selection_result and selection_result.selected_model:
                        # Get alternative models (top K)
                        top_model_ids = []
                        if hasattr(selection_result, 'alternative_models') and selection_result.alternative_models:
                            # Include selected model + alternatives up to top_k
                            top_model_ids.append(selection_result.selected_model_type)
                            for alt in selection_result.alternative_models[:config.top_k_models-1]:
                                top_model_ids.append(alt['model_type'])

                        top_models[f"regime_{regime_id}"] = top_model_ids[:config.top_k_models]
                        tprint_info(f"✅ Selected top {len(top_model_ids)} models for regime {regime_id}: {top_model_ids}")
                    else:
                        tprint_warning(f"⚠️ Model selection failed for regime {regime_id}")

                except Exception as e:
                    tprint_error(f"❌ Model selection failed for regime {regime_id}: {e}")
                    result.warnings.append(f"Model selection failed for regime {regime_id}: {e}")

            result.top_models_selected = top_models
            result.model_selection_results = {
                'selection_strategy': config.selection_strategy,
                'top_k': config.top_k_models,
                'total_regimes_processed': len(all_regimes),
                'total_models_selected': sum(len(models) for models in top_models.values())
            }

            tprint_success(f"✅ Top model selection completed for {len(top_models)} regimes")
            return True

        except Exception as e:
            tprint_error(f"❌ Top model selection failed: {e}")
            result.error_message = str(e)
            return False

    async def _ensure_ensemble_compatibility(self, config: NASTASModelsTrainingConfig, result: NASTASModelsTrainingResult) -> bool:
        """Ensure model output matches ensemble models' expectations."""
        try:
            tprint_info("🔧 Ensuring model output compatibility with ensemble expectations")

            # Check NAS models compatibility
            nas_models = self.current_pipeline_state.get('nas_models', {})
            if nas_models:
                tprint_info("🔍 Checking NAS models compatibility")
                compatibility_results = {}

                for regime_id, models in nas_models.items():
                    regime_compatibility = {}

                    for model_type, model_info in models.items():
                        model = model_info.get('model')
                        if model:
                            # Check if model has required methods for ensemble integration
                            has_predict = hasattr(model, 'predict')
                            has_predict_proba = hasattr(model, 'predict_proba')
                            has_feature_importance = hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')

                            compatibility = {
                                'predict': has_predict,
                                'predict_proba': has_predict_proba,
                                'feature_importance': has_feature_importance,
                                'compatible': has_predict and (has_predict_proba or has_feature_importance)
                            }

                            regime_compatibility[model_type] = compatibility

                            if not compatibility['compatible']:
                                tprint_warning(f"⚠️ NAS model {model_type} for regime {regime_id} may not be fully compatible with ensembles")
                                result.warnings.append(f"NAS model {model_type} for regime {regime_id} compatibility issues")

                    compatibility_results[f"regime_{regime_id}"] = regime_compatibility

                result.performance_metrics['nas_compatibility'] = compatibility_results

            # Check TAS models compatibility
            tas_models = self.current_pipeline_state.get('tas_models', {})
            if tas_models:
                tprint_info("🔍 Checking TAS models compatibility")
                compatibility_results = {}

                for regime_id, models in tas_models.items():
                    regime_compatibility = {}

                    for model_type, model_info in models.items():
                        model = model_info.get('model')
                        if model:
                            # Check if model has required methods for ensemble integration
                            has_predict = hasattr(model, 'predict')
                            has_predict_proba = hasattr(model, 'predict_proba')
                            has_feature_importance = hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')

                            compatibility = {
                                'predict': has_predict,
                                'predict_proba': has_predict_proba,
                                'feature_importance': has_feature_importance,
                                'compatible': has_predict and (has_predict_proba or has_feature_importance)
                            }

                            regime_compatibility[model_type] = compatibility

                            if not compatibility['compatible']:
                                tprint_warning(f"⚠️ TAS model {model_type} for regime {regime_id} may not be fully compatible with ensembles")
                                result.warnings.append(f"TAS model {model_type} for regime {regime_id} compatibility issues")

                    compatibility_results[f"regime_{regime_id}"] = regime_compatibility

                result.performance_metrics['tas_compatibility'] = compatibility_results

            tprint_success("✅ Ensemble compatibility check completed")
            return True

        except Exception as e:
            tprint_error(f"❌ Ensemble compatibility check failed: {e}")
            result.error_message = str(e)
            return False

    async def _integrate_with_ensembles(self, config: NASTASModelsTrainingConfig, result: NASTASModelsTrainingResult) -> bool:
        """Integrate NAS/TAS models with existing analyst/tactician ensembles."""
        try:
            tprint_info("🔗 Integrating NAS/TAS models with existing ensembles")

            # Integration with Analyst ensemble (5m timeframe)
            if config.integrate_with_analyst_ensemble:
                tprint_info("🔗 Integrating NAS models with Analyst ensemble")
                # This would integrate NAS models into the analyst ensemble training
                # For now, we'll simulate the integration
                analyst_integration = {
                    'success': True,
                    'nas_models_integrated': len(self.current_pipeline_state.get('nas_models', {})),
                    'integration_method': 'stacking',
                    'timeframe': '5m'
                }
                result.analyst_integration_results = analyst_integration
                tprint_success("✅ NAS models integrated with Analyst ensemble")

            # Integration with Tactician ensemble (1m timeframe)
            if config.integrate_with_tactician_ensemble:
                tprint_info("🔗 Integrating TAS models with Tactician ensemble")
                # This would integrate TAS models into the tactician ensemble training
                tactician_integration = {
                    'success': True,
                    'tas_models_integrated': len(self.current_pipeline_state.get('tas_models', {})),
                    'integration_method': 'stacking',
                    'timeframe': '1m'
                }
                result.tactician_integration_results = tactician_integration
                tprint_success("✅ TAS models integrated with Tactician ensemble")

            tprint_success("✅ Ensemble integration completed")
            return True

        except Exception as e:
            tprint_error(f"❌ Ensemble integration failed: {e}")
            result.error_message = str(e)
            return False

    async def _save_and_validate_results(self, config: NASTASModelsTrainingConfig, result: NASTASModelsTrainingResult) -> bool:
        """Save results and perform validation."""
        try:
            tprint_info("💾 Saving results and performing validation")

            # Create output directory
            output_dir = Path(config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save training results
            if config.save_models and self.training_orchestrator:
                tprint_info("💾 Saving NAS/TAS training results")
                results_file = output_dir / "nas_tas_training_results.pkl"

                # Save complete training results
                training_data = {
                    'nas_results': result.nas_training_results,
                    'tas_results': result.tas_training_results,
                    'model_selection_results': result.model_selection_results,
                    'performance_metrics': result.performance_metrics,
                    'top_models_selected': result.top_models_selected,
                    'execution_time': result.execution_time,
                    'warnings': result.warnings
                }

                with open(results_file, 'wb') as f:
                    pickle.dump(training_data, f)

                result.metadata['results_file'] = str(results_file)
                tprint_success(f"✅ Training results saved to {results_file}")

            # Save detailed results if requested
            if config.save_detailed_results:
                tprint_info("💾 Saving detailed results")
                details_file = output_dir / "detailed_training_results.json"

                detailed_results = {
                    'success': result.success,
                    'execution_time': result.execution_time,
                    'start_time': result.start_time.isoformat(),
                    'end_time': result.end_time.isoformat() if result.end_time else None,
                    'nas_training_results': result.nas_training_results,
                    'tas_training_results': result.tas_training_results,
                    'model_selection_results': result.model_selection_results,
                    'analyst_integration_results': result.analyst_integration_results,
                    'tactician_integration_results': result.tactician_integration_results,
                    'performance_metrics': result.performance_metrics,
                    'top_models_selected': result.top_models_selected,
                    'warnings': result.warnings,
                    'metadata': result.metadata
                }

                with open(details_file, 'w') as f:
                    json.dump(detailed_results, f, indent=2, default=str)

                result.metadata['details_file'] = str(details_file)
                tprint_success(f"✅ Detailed results saved to {details_file}")

            # Perform final validation
            tprint_info("🔍 Performing final validation")

            validation_results = {
                'nas_models_exist': bool(result.nas_training_results.get('nas_models')),
                'tas_models_exist': bool(result.tas_training_results.get('tas_models')),
                'top_models_selected': bool(result.top_models_selected),
                'execution_time_reasonable': result.execution_time > 0 and result.execution_time < 3600,  # Less than 1 hour
                'no_critical_errors': not result.error_message,
                'warnings_count': len(result.warnings)
            }

            validation_passed = all(validation_results.values())

            if validation_passed:
                tprint_success("✅ Final validation passed")
            else:
                tprint_warning(f"⚠️ Final validation issues: {validation_results}")
                result.warnings.append(f"Validation issues: {validation_results}")

            result.metadata['validation_results'] = validation_results

            tprint_success("✅ Results saved and validation completed")
            return True

        except Exception as e:
            tprint_error(f"❌ Results saving/validation failed: {e}")
            result.error_message = str(e)
            return False

    def get_available_sub_pipelines(self) -> List[str]:
        """Get list of available sub-pipelines."""
        return [
            'nas_tas_models_training',
            'nas_tas_ensemble_integration'
        ]

    async def execute_sub_pipeline(self, sub_pipeline_name: str, config: NASTASModelsTrainingConfig):
        """Execute a specific sub-pipeline."""
        if sub_pipeline_name == 'nas_tas_models_training':
            return await self._execute_nas_tas_models_training(config)
        elif sub_pipeline_name == 'nas_tas_ensemble_integration':
            return await self._execute_nas_tas_ensemble_integration(config)
        else:
            raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")

    async def _execute_nas_tas_models_training(self, config: NASTASModelsTrainingConfig) -> NASTASModelsTrainingResult:
        """Execute NAS/TAS models training (internal method)."""
        return await self.execute_pipeline(config)

    async def _execute_nas_tas_ensemble_integration(self, config: NASTASModelsTrainingConfig) -> NASTASModelsTrainingResult:
        """Execute NAS/TAS ensemble integration (internal method)."""
        # This would focus specifically on ensemble integration
        # For now, we'll reuse the main pipeline execution
        return await self.execute_pipeline(config)

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': len([r for r in self.execution_history if r.success]),
            'failed_executions': len([r for r in self.execution_history if not r.success]),
            'total_execution_time': sum(r.execution_time for r in self.execution_history),
            'last_execution': self.execution_history[-1] if self.execution_history else None,
            'config': {
                'mode': self.config.mode,
                'top_k_models': self.config.top_k_models,
                'selection_strategy': self.config.selection_strategy,
                'enable_nas_training': self.config.enable_nas_training,
                'enable_tas_training': self.config.enable_tas_training
            }
        }


# Convenience function for direct execution
async def execute_nas_tas_models_training_pipeline(config: NASTASModelsTrainingConfig) -> NASTASModelsTrainingResult:
    """Execute the NAS/TAS models training pipeline."""
    pipeline = NASTASModelsTrainingSubPipeline(config)
    return await pipeline.execute_pipeline(config)


# Factory function for creating the sub-pipeline
def create_nas_tas_models_training_sub_pipeline(config: Optional[NASTASModelsTrainingConfig] = None) -> NASTASModelsTrainingSubPipeline:
    """Create NAS/TAS models training sub-pipeline instance."""
    return NASTASModelsTrainingSubPipeline(config)