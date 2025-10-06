"""
Enhanced Analyst Models Training - A1-A4 Models Integration

This module handles training of the new Analyst models:
- A1: PatchTST-Embed + LightGBM (tree, baseline upgrade)
- A2: PatchTST-Embed + XGBoost (tree, diversified booster bias)
- A3: FT-Transformer (tabular transformer)
- A4: PatchTST-Embed + CatBoost (tree, different bias)
- Stacker: LGBM Calibrated Meta-Learner (per-regime)

The Analyst operates on the dedicated 15m timeframe and decides IF we trade by
screening market conditions and producing the green-signal gating that the
Tactician consumes.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum
import os
import asyncio

# Enhanced imports with comprehensive error handling
try:
    from src.utils.logger import system_logger
    from src.utils.ml_common.config import PerRegimeTrainingConfig
    from src.utils.ml_common.training import PerRegimeTrainingStep
    ANALYST_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import core ML utilities: {e}")
    ANALYST_TRAINING_AVAILABLE = False

# Import enhanced logging and utilities - CRITICAL: Fast fail if not available
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_progress, tprint_performance, tprint_structured,
        tprint_timer, LogLevel
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL ERROR: tprint is required but not available: {e}")
    TPRINT_AVAILABLE = False

# Import common utilities - CRITICAL: Fast fail if not available
try:
    from src.utils.common_operations import (
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        cleanup_m1_optimizers, integrate_with_m1_optimizers
    )
    COMMON_OPS_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Common operations utilities are required but not available: {e}")
    COMMON_OPS_AVAILABLE = False

try:
    from src.utils.common_utilities import (
        safe_dataframe_operation, validate_dataframe_columns, calculate_data_quality_metrics,
        safe_merge_dataframes, create_summary_statistics
    )
    COMMON_UTILITIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Common utilities are required but not available: {e}")
    COMMON_UTILITIES_AVAILABLE = False

try:
    from src.utils.math_validation import (
        safe_divide, validate_finite, validate_positive, validate_range,
        safe_correlation, safe_percentage_change
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Math validation utilities are required but not available: {e}")
    MATH_VALIDATION_AVAILABLE = False

# Import the new analyst models
try:
    from src.analyst.models.analyst_models_orchestrator import (
        AnalystModelsOrchestrator,
        AnalystModelsConfig,
        create_analyst_models_orchestrator
    )
    ANALYST_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import new analyst models: {e}")
    ANALYST_MODELS_AVAILABLE = False


class AnalystModelType(Enum):
    """Enhanced Analyst model types."""
    A1_PATCHTST_LIGHTGBM = "A1_PATCHTST_LIGHTGBM"
    A2_PATCHTST_XGBOOST = "A2_PATCHTST_XGBOOST"
    A3_FT_TRANSFORMER = "A3_FT_TRANSFORMER"
    A4_PATCHTST_CATBOOST = "A4_PATCHTST_CATBOOST"
    STACKER_LGBM_CALIBRATED = "STACKER_LGBM_CALIBRATED"
    # Legacy models for backward compatibility
    TCN = "TCN"
    LIGHTGBM = "LIGHTGBM"
    RIDGE = "RIDGE"
    ELASTIC_NET = "ELASTIC_NET"
    RANDOM_FOREST = "RANDOM_FOREST"
    NAS = "NAS"
    TAS = "TAS"


@dataclass
class EnhancedAnalystModelsTrainingConfig:
    """Configuration for Enhanced Analyst models training."""
    # Model selection
    model_types: List[AnalystModelType] = None
    enable_legacy_models: bool = False
    
    # New A1-A4 models configuration
    enable_a1: bool = True
    enable_a2: bool = True
    enable_a3: bool = True
    enable_a4: bool = True
    enable_stacker: bool = True
    
    # Training parameters
    save_models: bool = True
    output_directory: str = "generated/analyst_models_enhanced"
    
    # Hardware optimization
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    max_workers: int = 4
    
    # Validation parameters
    validation_split: float = 0.2
    min_training_samples: int = 100
    
    # Calibration settings
    enable_calibration: bool = True
    calibration_method: str = 'isotonic'
    enable_venn_abers: bool = True
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.model_types is None:
            if self.enable_legacy_models:
                self.model_types = [
                    AnalystModelType.A1_PATCHTST_LIGHTGBM,
                    AnalystModelType.A2_PATCHTST_XGBOOST,
                    AnalystModelType.A3_FT_TRANSFORMER,
                    AnalystModelType.A4_PATCHTST_CATBOOST,
                    AnalystModelType.STACKER_LGBM_CALIBRATED,
                    AnalystModelType.TCN,
                    AnalystModelType.LIGHTGBM,
                    AnalystModelType.RIDGE,
                    AnalystModelType.ELASTIC_NET,
                    AnalystModelType.RANDOM_FOREST,
                    AnalystModelType.NAS,
                    AnalystModelType.TAS
                ]
            else:
                self.model_types = [
                    AnalystModelType.A1_PATCHTST_LIGHTGBM,
                    AnalystModelType.A2_PATCHTST_XGBOOST,
                    AnalystModelType.A3_FT_TRANSFORMER,
                    AnalystModelType.A4_PATCHTST_CATBOOST,
                    AnalystModelType.STACKER_LGBM_CALIBRATED
                ]


@dataclass
class EnhancedAnalystModelsTrainingResult:
    """Result of Enhanced Analyst models training."""
    # Training results
    models: Dict[str, Any] = None
    training_metrics: Dict[str, Any] = None
    
    # New A1-A4 models results
    a1_a4_results: Dict[str, Any] = None
    stacker_results: Dict[str, Any] = None
    
    # Metadata
    execution_time: float = 0.0
    total_samples: int = 0
    features_used: List[str] = None
    model_types_trained: List[str] = None
    models_per_type: int = 0
    
    # Status
    training_completed: bool = False
    error: Optional[str] = None


class EnhancedAnalystModelsTrainingStep:
    """
    Enhanced Analyst Models Training Step.
    
    Handles training of the new A1-A4 Analyst models with orchestrator.
    """
    
    def __init__(self, config: Optional[EnhancedAnalystModelsTrainingConfig] = None):
        """Initialize the Enhanced Analyst models training step."""
        try:
            self.config = config or EnhancedAnalystModelsTrainingConfig()
            self.logger = system_logger.getChild('EnhancedAnalystModelsTrainingStep')
            
            # Initialize orchestrator if new models are available
            self.orchestrator = None
            if ANALYST_MODELS_AVAILABLE:
                orchestrator_config = AnalystModelsConfig(
                    enable_a1=self.config.enable_a1,
                    enable_a2=self.config.enable_a2,
                    enable_a3=self.config.enable_a3,
                    enable_a4=self.config.enable_a4,
                    enable_stacker=self.config.enable_stacker,
                    enable_parallel_training=self.config.enable_parallel_processing,
                    max_workers=self.config.max_workers,
                    save_models=self.config.save_models,
                    output_directory=self.config.output_directory
                )
                self.orchestrator = create_analyst_models_orchestrator(orchestrator_config)
                tprint_success("✅ Enhanced Analyst Models Orchestrator initialized")
            else:
                tprint_warning("⚠️ New analyst models not available, falling back to legacy models")
            
            # Initialize hardware optimizers
            if COMMON_OPS_AVAILABLE:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                tprint_success("✅ Hardware optimizers initialized")
            else:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
            
            tprint_success("✅ EnhancedAnalystModelsTrainingStep initialized successfully")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize EnhancedAnalystModelsTrainingStep: {e}")
            raise
    
    async def train_analyst_models(
        self,
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        sample_weight: Optional[np.ndarray] = None,
        regime_assignments: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train Enhanced Analyst models (A1-A4 + Stacker).
        
        Args:
            training_data: DataFrame with features and targets
            feature_columns: List of feature column names
            target_columns: List of target column names
            sample_weight: Optional sample weights
            regime_assignments: Optional regime assignments
            **kwargs: Additional parameters
            
        Returns:
            Dict with trained models and metrics
        """
        start_time = tprint_timer()
        tprint_info("🚀 Starting Enhanced Analyst models training...")
        
        try:
            # Validate inputs
            if training_data.empty or not feature_columns or not target_columns:
                raise ValueError("Insufficient training data or missing columns")
            
            # Prepare training data
            X = training_data[feature_columns].values
            y = training_data[target_columns].values
            
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            if sample_weight is None:
                sample_weight = np.ones(len(training_data))
            
            # Train new A1-A4 models if available
            a1_a4_results = {}
            if ANALYST_MODELS_AVAILABLE and self.orchestrator is not None:
                tprint_info("🤖 Training A1-A4 models with orchestrator...")
                
                try:
                    # Train orchestrator
                    await self.orchestrator.fit(
                        X=X,
                        y=y.ravel(),
                        regimes=regime_assignments,
                        sample_weight=sample_weight
                    )
                    
                    # Get performance metrics
                    performance = self.orchestrator.get_model_performance()
                    a1_a4_results = {
                        'orchestrator': self.orchestrator,
                        'performance': performance,
                        'models_trained': list(self.orchestrator.training_results.keys()),
                        'stacker_trained': self.orchestrator.stacker is not None
                    }
                    
                    tprint_success("✅ A1-A4 models trained successfully")
                    
                except Exception as e:
                    tprint_error(f"❌ A1-A4 models training failed: {e}")
                    a1_a4_results = {'error': str(e)}
            
            # Train legacy models if enabled
            legacy_results = {}
            if self.config.enable_legacy_models:
                tprint_info("🔧 Training legacy models...")
                legacy_results = await self._train_legacy_models(
                    training_data, feature_columns, target_columns, 
                    sample_weight, regime_assignments, **kwargs
                )
            
            # Combine results
            all_models = {}
            all_metrics = {}
            
            # Add A1-A4 models
            if a1_a4_results.get('orchestrator'):
                orchestrator = a1_a4_results['orchestrator']
                for model_name, result in orchestrator.training_results.items():
                    if result.get('model') is not None:
                        all_models[f"enhanced_{model_name}"] = result['model']
                        all_metrics[f"enhanced_{model_name}"] = {
                            'accuracy': result.get('accuracy', 0.0),
                            'logloss': result.get('logloss', float('inf')),
                            'model_type': type(result['model']).__name__
                        }
                
                # Add stacker if available
                if orchestrator.stacker is not None:
                    all_models['enhanced_stacker'] = orchestrator.stacker
                    all_metrics['enhanced_stacker'] = {
                        'model_type': 'StackerLGBMCalibrated',
                        'feature_importance': orchestrator.stacker.get_feature_importance()
                    }
            
            # Add legacy models
            if legacy_results.get('models'):
                all_models.update(legacy_results['models'])
                all_metrics.update(legacy_results['metrics'])
            
            if not all_models:
                raise ValueError("Failed to train any models")
            
            execution_time = tprint_timer(start_time)
            tprint_success(f"✅ Enhanced Analyst models training completed in {execution_time:.2f}s")
            
            return {
                'models': all_models,
                'metrics': all_metrics,
                'a1_a4_results': a1_a4_results,
                'legacy_results': legacy_results,
                'training_time': execution_time,
                'features_used': feature_columns,
                'samples_used': len(training_data),
                'model_types_trained': [mt.value for mt in self.config.model_types],
                'models_per_type': len(self.config.model_types)
            }
            
        except Exception as e:
            execution_time = tprint_timer(start_time)
            tprint_error(f"❌ Enhanced Analyst models training failed: {e}")
            return {
                'models': {},
                'metrics': {},
                'a1_a4_results': {},
                'legacy_results': {},
                'training_time': execution_time,
                'error': str(e)
            }
    
    async def _train_legacy_models(
        self,
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        sample_weight: np.ndarray,
        regime_assignments: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train legacy models for backward compatibility."""
        try:
            # Import legacy training step
            from .analyst_models_training import AnalystModelsTrainingStep
            
            legacy_config = AnalystModelsTrainingConfig(
                model_types=[AnalystModelType.TCN, AnalystModelType.LIGHTGBM, 
                           AnalystModelType.RIDGE, AnalystModelType.ELASTIC_NET,
                           AnalystModelType.RANDOM_FOREST, AnalystModelType.NAS, AnalystModelType.TAS],
                save_models=self.config.save_models,
                output_directory=f"{self.config.output_directory}/legacy",
                enable_parallel_processing=self.config.enable_parallel_processing,
                enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                memory_limit_gb=self.config.memory_limit_gb,
                validation_split=self.config.validation_split,
                min_training_samples=self.config.min_training_samples
            )
            
            legacy_trainer = AnalystModelsTrainingStep(legacy_config)
            
            # Train legacy models
            legacy_result = await legacy_trainer.train_analyst_models(
                training_data=training_data,
                feature_columns=feature_columns,
                target_columns=target_columns,
                sample_weight=sample_weight,
                regime_assignments=regime_assignments,
                **kwargs
            )
            
            tprint_success("✅ Legacy models trained successfully")
            return legacy_result
            
        except Exception as e:
            tprint_error(f"❌ Legacy models training failed: {e}")
            return {'models': {}, 'metrics': {}, 'error': str(e)}
    
    def predict_green_light(
        self,
        market_data: pd.DataFrame,
        feature_columns: List[str],
        regime_assignments: Optional[np.ndarray] = None,
        use_enhanced_models: bool = True
    ) -> Dict[str, Any]:
        """Predict green light using trained models."""
        if not self.orchestrator or not self.orchestrator.is_fitted:
            raise ValueError("Enhanced models must be trained before prediction")
        
        try:
            # Prepare data
            X = market_data[feature_columns].values
            
            # Get predictions from orchestrator
            probabilities = self.orchestrator.predict_proba(X, regime_assignments)
            uncertainty = self.orchestrator.predict_uncertainty(X, regime_assignments)
            
            # Determine green light decision
            green_light_threshold = 0.6  # Configurable threshold
            green_light_decisions = (probabilities > green_light_threshold).astype(int)
            
            return {
                'probabilities': probabilities,
                'green_light_decisions': green_light_decisions,
                'uncertainty': uncertainty,
                'threshold': green_light_threshold,
                'confidence_levels': uncertainty.get('confidence_intervals', {}),
                'margin_stats': uncertainty.get('margin_stats', {}),
                'regime_stats': uncertainty.get('regime_stats', {})
            }
            
        except Exception as e:
            tprint_error(f"❌ Green light prediction failed: {e}")
            return {
                'error': str(e),
                'probabilities': np.zeros(len(market_data)),
                'green_light_decisions': np.zeros(len(market_data), dtype=int)
            }
    
    def get_model_insights(self) -> Dict[str, Any]:
        """Get insights from trained models."""
        if not self.orchestrator or not self.orchestrator.is_fitted:
            return {}
        
        insights = {
            'performance': self.orchestrator.get_model_performance(),
            'feature_importance': {},
            'model_info': {}
        }
        
        # Get feature importance from each model
        for model_name, result in self.orchestrator.training_results.items():
            if result.get('model') is not None and hasattr(result['model'], 'get_feature_importance'):
                insights['feature_importance'][model_name] = result['model'].get_feature_importance()
            
            # Get model-specific info
            if hasattr(result['model'], 'get_booster_info'):
                insights['model_info'][model_name] = result['model'].get_booster_info()
            elif hasattr(result['model'], 'get_catboost_info'):
                insights['model_info'][model_name] = result['model'].get_catboost_info()
        
        # Get stacker info
        if self.orchestrator.stacker is not None:
            insights['stacker_info'] = {
                'feature_importance': self.orchestrator.stacker.get_feature_importance(),
                'regime_calibration': self.orchestrator.stacker.get_regime_calibration_info()
            }
        
        return insights
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the models training step."""
        metrics = {
            'config': {
                'model_types': [mt.value for mt in self.config.model_types],
                'enable_a1': self.config.enable_a1,
                'enable_a2': self.config.enable_a2,
                'enable_a3': self.config.enable_a3,
                'enable_a4': self.config.enable_a4,
                'enable_stacker': self.config.enable_stacker,
                'enable_legacy_models': self.config.enable_legacy_models,
                'save_models': self.config.save_models,
                'output_directory': self.config.output_directory,
                'enable_parallel_processing': self.config.enable_parallel_processing,
                'enable_gpu_acceleration': self.config.enable_gpu_acceleration
            },
            'hardware_optimization': {
                'gpu_manager': self.gpu_manager is not None,
                'memory_optimizer': self.memory_optimizer is not None,
                'cpu_optimizer': self.cpu_optimizer is not None
            },
            'model_availability': {
                'analyst_models_available': ANALYST_MODELS_AVAILABLE,
                'orchestrator_initialized': self.orchestrator is not None
            }
        }
        
        return metrics


# Convenience function for external usage
async def execute_enhanced_analyst_models_training(
    training_data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    sample_weight: Optional[np.ndarray] = None,
    regime_assignments: Optional[np.ndarray] = None,
    config: Optional[EnhancedAnalystModelsTrainingConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute Enhanced Analyst models training.
    
    Args:
        training_data: DataFrame with features and targets
        feature_columns: List of feature column names
        target_columns: List of target column names
        sample_weight: Optional sample weights
        regime_assignments: Optional regime assignments
        config: Optional configuration
        **kwargs: Additional parameters
        
    Returns:
        Dict with trained models and metrics
    """
    trainer = EnhancedAnalystModelsTrainingStep(config)
    return await trainer.train_analyst_models(
        training_data, feature_columns, target_columns, sample_weight, regime_assignments, **kwargs
    )