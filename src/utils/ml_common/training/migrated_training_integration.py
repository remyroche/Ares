"""
Migrated Training Integration

This module provides comprehensive training integration for all migrated ML models
across HMM, Analyst, and Tactician components with support for:
- Regime-aware training and parameter optimization
- Comprehensive regularization and overfitting prevention
- Integration with existing ML training pipeline
- Support for validation and HPO
- Multi-timeframe training coordination
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
import joblib

# Enhanced dependency management
try:
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args)
    def tprint_info(*args, **kwargs): print(f"INFO: {args[0] if args else ''}")
    def tprint_warning(*args, **kwargs): print(f"WARNING: {args[0] if args else ''}")
    def tprint_error(*args, **kwargs): print(f"ERROR: {args[0] if args else ''}")
    def tprint_success(*args, **kwargs): print(f"SUCCESS: {args[0] if args else ''}")

# Import migrated model components
try:
    from ..models.migrated_model_configs import (
        MigratedModelConfigs, ModelConfig, ModelArchitecture,
        RegimeCharacteristics, RegimeAwareParameterOptimizer
    )
    from ..models.enhanced_migrated_factory import EnhancedMigratedModelFactory
    MIGRATED_MODELS_AVAILABLE = True
except ImportError:
    MIGRATED_MODELS_AVAILABLE = False
    tprint_warning("⚠️ Migrated model components not available")

# Import training utilities
try:
    from .enhanced_training_utils import (
        EarlyStoppingConfig, PurgedCVConfig, OverfittingMonitorConfig,
        EnhancedTrainingUtils, ValidationResult
    )
    TRAINING_UTILS_AVAILABLE = True
except ImportError:
    TRAINING_UTILS_AVAILABLE = False
    tprint_warning("⚠️ Enhanced training utils not available")

# Import base training components
try:
    from .base_training_step import BaseTrainingStep
    from .enhanced_early_stopping import EnhancedEarlyStopping
    BASE_TRAINING_AVAILABLE = True
except ImportError:
    BASE_TRAINING_AVAILABLE = False
    tprint_warning("⚠️ Base training components not available")

# Import common utilities
try:
    from src.utils.common_operations import safe_json_dump, safe_json_load, ensure_directory
    from src.utils.math_validation import validate_finite, validate_positive
    COMMON_UTILS_AVAILABLE = True
except ImportError:
    COMMON_UTILS_AVAILABLE = False
    tprint_warning("⚠️ Common utilities not available")

logger = logging.getLogger(__name__)


@dataclass
class MigratedTrainingConfig:
    """Configuration for migrated model training."""
    # Component configurations
    hmm_config: Dict[str, Any] = field(default_factory=dict)
    analyst_config: Dict[str, Any] = field(default_factory=dict)
    tactician_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training parameters
    enable_regime_aware_training: bool = True
    enable_overfitting_prevention: bool = True
    enable_regularization: bool = True
    enable_early_stopping: bool = True
    
    # Validation parameters
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    enable_purged_cv: bool = True
    enable_walk_forward: bool = True
    
    # HPO parameters
    enable_hpo: bool = True
    hpo_trials: int = 100
    hpo_timeout: int = 3600
    
    # Multi-timeframe coordination
    enable_multi_timeframe_coordination: bool = True
    timeframe_hierarchy: List[str] = field(default_factory=lambda: ["15m", "5m", "1m"])
    
    # Output configuration
    output_dir: str = "./migrated_models"
    enable_model_persistence: bool = True
    enable_performance_tracking: bool = True


class MigratedTrainingIntegration:
    """Comprehensive training integration for migrated ML models."""
    
    def __init__(self, config: Optional[MigratedTrainingConfig] = None):
        """Initialize migrated training integration."""
        self.logger = logger.getChild('MigratedTrainingIntegration')
        self.logger.info("🚀 Initializing Migrated Training Integration...")
        start_time = time.time()
        
        self.config = config or MigratedTrainingConfig()
        
        # Initialize model factory
        self.model_factory = None
        if MIGRATED_MODELS_AVAILABLE:
            self.model_factory = EnhancedMigratedModelFactory()
            self.logger.info("✅ Migrated model factory initialized")
        
        # Initialize training utilities
        self.training_utils = None
        if TRAINING_UTILS_AVAILABLE:
            self.training_utils = EnhancedTrainingUtils()
            self.logger.info("✅ Enhanced training utilities initialized")
        
        # Initialize early stopping
        self.early_stopping = None
        if BASE_TRAINING_AVAILABLE:
            self.early_stopping = EnhancedEarlyStopping()
            self.logger.info("✅ Enhanced early stopping initialized")
        
        # Training results storage
        self.training_results: Dict[str, Dict[str, Any]] = {
            "hmm_models": {},
            "analyst_models": {},
            "tactician_models": {}
        }
        
        # Regime characteristics cache
        self.regime_cache: Dict[str, RegimeCharacteristics] = {}
        
        # Ensure output directory
        if COMMON_UTILS_AVAILABLE:
            ensure_directory(Path(self.config.output_dir))
        
        init_time = time.time() - start_time
        self.logger.info(f"✅ Migrated Training Integration initialized in {init_time:.3f}s")
        self.logger.info(f"📊 Regime-aware training: {self.config.enable_regime_aware_training}")
        self.logger.info(f"📊 Overfitting prevention: {self.config.enable_overfitting_prevention}")
        self.logger.info(f"📊 Multi-timeframe coordination: {self.config.enable_multi_timeframe_coordination}")
    
    def train_hmm_models(self, X: np.ndarray, y: np.ndarray, 
                        regime_data: Optional[Dict[str, Any]] = None,
                        feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train all HMM models for regime detection on 15m timeframe."""
        self.logger.info("🔄 Starting HMM models training...")
        start_time = time.time()
        
        if not MIGRATED_MODELS_AVAILABLE:
            raise ImportError("Migrated model components not available")
        
        # Extract regime characteristics if available
        regime_characteristics = None
        if regime_data and self.config.enable_regime_aware_training:
            regime_characteristics = self._extract_regime_characteristics(regime_data)
            self.regime_cache["hmm"] = regime_characteristics
        
        # Get HMM model configurations
        hmm_models = MigratedModelConfigs.get_hmm_models()
        
        # Train each model
        results = {}
        for model_name, model_config in hmm_models.items():
            try:
                self.logger.info(f"🔄 Training HMM model: {model_name}")
                
                # Create model
                model = self.model_factory.create_hmm_model(
                    model_name, X.shape[1], len(np.unique(y)), regime_characteristics
                )
                
                # Train model with enhanced utilities
                training_result = self._train_single_model(
                    model, X, y, model_config, "hmm", model_name
                )
                
                results[model_name] = training_result
                self.training_results["hmm_models"][model_name] = training_result
                
                self.logger.info(f"✅ HMM model {model_name} trained successfully")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to train HMM model {model_name}: {e}")
                results[model_name] = {"error": str(e), "success": False}
        
        training_time = time.time() - start_time
        self.logger.info(f"✅ HMM models training completed in {training_time:.3f}s")
        self.logger.info(f"📊 Successfully trained {len([r for r in results.values() if r.get('success', False)])} models")
        
        return results
    
    def train_analyst_models(self, X: np.ndarray, y: np.ndarray,
                           hmm_predictions: Optional[np.ndarray] = None,
                           regime_data: Optional[Dict[str, Any]] = None,
                           feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train all Analyst models for trading opportunities on 5m timeframe."""
        self.logger.info("🔄 Starting Analyst models training...")
        start_time = time.time()
        
        if not MIGRATED_MODELS_AVAILABLE:
            raise ImportError("Migrated model components not available")
        
        # Combine features with HMM predictions if available
        if hmm_predictions is not None:
            X_enhanced = np.column_stack([X, hmm_predictions])
            self.logger.info(f"📊 Enhanced features with HMM predictions: {X_enhanced.shape[1]} total features")
        else:
            X_enhanced = X
        
        # Extract regime characteristics if available
        regime_characteristics = None
        if regime_data and self.config.enable_regime_aware_training:
            regime_characteristics = self._extract_regime_characteristics(regime_data)
            self.regime_cache["analyst"] = regime_characteristics
        
        # Get Analyst model configurations
        analyst_models = MigratedModelConfigs.get_analyst_models()
        
        # Train each model
        results = {}
        for model_name, model_config in analyst_models.items():
            try:
                self.logger.info(f"🔄 Training Analyst model: {model_name}")
                
                # Create model
                model = self.model_factory.create_analyst_model(
                    model_name, X_enhanced.shape[1], 1, regime_characteristics  # Assuming regression
                )
                
                # Train model with enhanced utilities
                training_result = self._train_single_model(
                    model, X_enhanced, y, model_config, "analyst", model_name
                )
                
                results[model_name] = training_result
                self.training_results["analyst_models"][model_name] = training_result
                
                self.logger.info(f"✅ Analyst model {model_name} trained successfully")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to train Analyst model {model_name}: {e}")
                results[model_name] = {"error": str(e), "success": False}
        
        training_time = time.time() - start_time
        self.logger.info(f"✅ Analyst models training completed in {training_time:.3f}s")
        self.logger.info(f"📊 Successfully trained {len([r for r in results.values() if r.get('success', False)])} models")
        
        return results
    
    def train_tactician_models(self, X: np.ndarray, y: np.ndarray,
                             hmm_predictions: Optional[np.ndarray] = None,
                             analyst_predictions: Optional[np.ndarray] = None,
                             regime_data: Optional[Dict[str, Any]] = None,
                             feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train all Tactician models for entry timing on 1m timeframe."""
        self.logger.info("🔄 Starting Tactician models training...")
        start_time = time.time()
        
        if not MIGRATED_MODELS_AVAILABLE:
            raise ImportError("Migrated model components not available")
        
        # Combine features with HMM and Analyst predictions if available
        X_enhanced = X
        if hmm_predictions is not None:
            X_enhanced = np.column_stack([X_enhanced, hmm_predictions])
            self.logger.info(f"📊 Enhanced features with HMM predictions")
        
        if analyst_predictions is not None:
            X_enhanced = np.column_stack([X_enhanced, analyst_predictions])
            self.logger.info(f"📊 Enhanced features with Analyst predictions")
        
        self.logger.info(f"📊 Final feature count: {X_enhanced.shape[1]}")
        
        # Extract regime characteristics if available
        regime_characteristics = None
        if regime_data and self.config.enable_regime_aware_training:
            regime_characteristics = self._extract_regime_characteristics(regime_data)
            self.regime_cache["tactician"] = regime_characteristics
        
        # Get Tactician model configurations
        tactician_models = MigratedModelConfigs.get_tactician_models()
        
        # Train each model
        results = {}
        for model_name, model_config in tactician_models.items():
            try:
                self.logger.info(f"🔄 Training Tactician model: {model_name}")
                
                # Create model
                model = self.model_factory.create_tactician_model(
                    model_name, X_enhanced.shape[1], 1, regime_characteristics  # Assuming regression
                )
                
                # Train model with enhanced utilities
                training_result = self._train_single_model(
                    model, X_enhanced, y, model_config, "tactician", model_name
                )
                
                results[model_name] = training_result
                self.training_results["tactician_models"][model_name] = training_result
                
                self.logger.info(f"✅ Tactician model {model_name} trained successfully")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to train Tactician model {model_name}: {e}")
                results[model_name] = {"error": str(e), "success": False}
        
        training_time = time.time() - start_time
        self.logger.info(f"✅ Tactician models training completed in {training_time:.3f}s")
        self.logger.info(f"📊 Successfully trained {len([r for r in results.values() if r.get('success', False)])} models")
        
        return results
    
    def train_all_models(self, data_config: Dict[str, Dict[str, Any]],
                        regime_data: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
        """Train all models across all components with multi-timeframe coordination."""
        self.logger.info("🔄 Starting comprehensive migrated models training...")
        start_time = time.time()
        
        all_results = {}
        
        try:
            # Train HMM models first (15m timeframe)
            if "hmm" in data_config:
                hmm_data = data_config["hmm"]
                hmm_regime_data = regime_data.get("hmm") if regime_data else None
                
                hmm_results = self.train_hmm_models(
                    hmm_data["X"], hmm_data["y"], hmm_regime_data, hmm_data.get("feature_names")
                )
                all_results["hmm_models"] = hmm_results
                
                # Extract HMM predictions for downstream models
                hmm_predictions = self._extract_model_predictions(hmm_results, hmm_data["X"])
            
            # Train Analyst models (5m timeframe) with HMM inputs
            if "analyst" in data_config:
                analyst_data = data_config["analyst"]
                analyst_regime_data = regime_data.get("analyst") if regime_data else None
                
                analyst_results = self.train_analyst_models(
                    analyst_data["X"], analyst_data["y"], hmm_predictions, 
                    analyst_regime_data, analyst_data.get("feature_names")
                )
                all_results["analyst_models"] = analyst_results
                
                # Extract Analyst predictions for downstream models
                analyst_predictions = self._extract_model_predictions(analyst_results, analyst_data["X"])
            
            # Train Tactician models (1m timeframe) with HMM and Analyst inputs
            if "tactician" in data_config:
                tactician_data = data_config["tactician"]
                tactician_regime_data = regime_data.get("tactician") if regime_data else None
                
                tactician_results = self.train_tactician_models(
                    tactician_data["X"], tactician_data["y"], hmm_predictions, analyst_predictions,
                    tactician_regime_data, tactician_data.get("feature_names")
                )
                all_results["tactician_models"] = tactician_results
            
            # Save training results
            if self.config.enable_model_persistence:
                self._save_training_results(all_results)
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive training failed: {e}")
            all_results["error"] = str(e)
        
        training_time = time.time() - start_time
        self.logger.info(f"✅ Comprehensive migrated models training completed in {training_time:.3f}s")
        
        return all_results
    
    def _train_single_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                           model_config: ModelConfig, component: str, model_name: str) -> Dict[str, Any]:
        """Train a single model with enhanced utilities."""
        try:
            # Prepare training configuration
            training_config = {
                "enable_early_stopping": self.config.enable_early_stopping,
                "enable_overfitting_prevention": self.config.enable_overfitting_prevention,
                "enable_regularization": self.config.enable_regularization,
                "validation_split": self.config.validation_split,
                "cross_validation_folds": self.config.cross_validation_folds,
                "enable_purged_cv": self.config.enable_purged_cv
            }
            
            # Train model
            if hasattr(model, 'fit'):
                model.fit(X, y)
                training_success = True
            else:
                raise ValueError(f"Model {model_name} does not have fit method")
            
            # Validate model
            validation_result = None
            if TRAINING_UTILS_AVAILABLE:
                validation_result = self.training_utils.validate_model(model, X, y, training_config)
            
            # Extract predictions for evaluation
            if hasattr(model, 'predict'):
                predictions = model.predict(X)
            else:
                predictions = None
            
            result = {
                "model": model,
                "success": training_success,
                "validation": validation_result,
                "predictions": predictions,
                "model_config": model_config,
                "component": component,
                "model_name": model_name,
                "training_time": time.time()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to train model {model_name}: {e}")
            return {
                "error": str(e),
                "success": False,
                "component": component,
                "model_name": model_name,
                "training_time": time.time()
            }
    
    def _extract_regime_characteristics(self, regime_data: Dict[str, Any]) -> RegimeCharacteristics:
        """Extract regime characteristics from regime data."""
        try:
            # Extract 4D regime characteristics
            volume = regime_data.get("volume", 0.5)
            volatility = regime_data.get("volatility", 0.5)
            momentum = regime_data.get("momentum", 0.5)
            trend = regime_data.get("trend", 0.5)
            
            # Validate values
            volume = max(0.0, min(1.0, float(volume)))
            volatility = max(0.0, min(1.0, float(volatility)))
            momentum = max(0.0, min(1.0, float(momentum)))
            trend = max(0.0, min(1.0, float(trend)))
            
            return RegimeCharacteristics(
                volume=volume,
                volatility=volatility,
                momentum=momentum,
                trend=trend
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract regime characteristics: {e}")
            # Return default characteristics
            return RegimeCharacteristics(volume=0.5, volatility=0.5, momentum=0.5, trend=0.5)
    
    def _extract_model_predictions(self, model_results: Dict[str, Any], X: np.ndarray) -> Optional[np.ndarray]:
        """Extract predictions from trained models."""
        try:
            # Find the best performing model
            best_model = None
            best_score = -np.inf
            
            for model_name, result in model_results.items():
                if result.get("success", False) and "validation" in result:
                    validation = result["validation"]
                    if validation and "score" in validation:
                        score = validation["score"]
                        if score > best_score:
                            best_score = score
                            best_model = result.get("model")
            
            # Extract predictions from best model
            if best_model and hasattr(best_model, 'predict'):
                predictions = best_model.predict(X)
                return predictions
            
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract model predictions: {e}")
            return None
    
    def _save_training_results(self, results: Dict[str, Any]) -> None:
        """Save training results to disk."""
        try:
            if not COMMON_UTILS_AVAILABLE:
                return
            
            output_path = Path(self.config.output_dir) / "training_results.json"
            
            # Prepare results for serialization (remove non-serializable objects)
            serializable_results = {}
            for component, component_results in results.items():
                if component == "error":
                    serializable_results[component] = component_results
                    continue
                
                serializable_results[component] = {}
                for model_name, model_result in component_results.items():
                    serializable_result = {
                        "success": model_result.get("success", False),
                        "component": model_result.get("component"),
                        "model_name": model_result.get("model_name"),
                        "training_time": model_result.get("training_time"),
                        "validation": model_result.get("validation")
                    }
                    
                    if "error" in model_result:
                        serializable_result["error"] = model_result["error"]
                    
                    serializable_results[component][model_name] = serializable_result
            
            # Save results
            safe_json_dump(serializable_results, output_path)
            self.logger.info(f"✅ Training results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save training results: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        summary = {
            "total_models": 0,
            "successful_models": 0,
            "failed_models": 0,
            "components": {},
            "regime_aware_models": 0,
            "training_time": time.time()
        }
        
        for component, models in self.training_results.items():
            component_summary = {
                "total": len(models),
                "successful": 0,
                "failed": 0,
                "models": {}
            }
            
            for model_name, result in models.items():
                component_summary["total"] += 1
                
                if result.get("success", False):
                    component_summary["successful"] += 1
                    summary["successful_models"] += 1
                else:
                    component_summary["failed"] += 1
                    summary["failed_models"] += 1
                
                # Check if model is regime-aware
                if result.get("model_config", {}).get("regime_aware", False):
                    summary["regime_aware_models"] += 1
                
                component_summary["models"][model_name] = {
                    "success": result.get("success", False),
                    "has_validation": "validation" in result,
                    "regime_aware": result.get("model_config", {}).get("regime_aware", False)
                }
            
            summary["components"][component] = component_summary
            summary["total_models"] += component_summary["total"]
        
        return summary


def create_migrated_training_integration(config: Optional[MigratedTrainingConfig] = None) -> MigratedTrainingIntegration:
    """Create a migrated training integration instance."""
    return MigratedTrainingIntegration(config)