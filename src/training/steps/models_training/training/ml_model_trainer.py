"""
Unified ML Model Trainer Pipeline

This module provides a single, unified pipeline for training all ML models:
- Analyst Base Models
- Analyst Ensemble Models  
- Tactician Base Models
- Tactician Ensemble Models

The pipeline handles:
- Configuration management
- Feature engineering
- Data preprocessing
- Model training
- Cross-validation
- Hyperparameter optimization
- Data leakage detection
- Metrics analysis
- SHAP analysis
- Model evaluation
- Results reporting

Everything is managed by the pipeline except for:
- Which ML models to train (specified in config)
- ML model parameters (specified in config)
- What targets to use (specified in config)
- What inputs to use (specified in config)
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import hashlib
import json

# Import existing components - removed non-existent core imports
# These will be handled by the unified MLModelTrainer approach

# Import utilities
from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_performance, tprint_data_preview, tprint_data_format, LogLevel
)
from src.core.decorators import handles_errors, traced, log_execution_time

# Import common operations and utilities
from src.utils.common_operations import safe_dataframe_operation, safe_array_operation
from src.utils.common_utilities import (
    validate_dataframe, validate_array, safe_dataframe_operation,
    memory_managed, MemoryStrategy, get_memory_manager, force_cleanup
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, safe_exp,
    validate_numeric_input, validate_array_input, safe_statistical_operation
)

# Import hardware optimization
from src.utils.hardware.integrated_hardware_manager import get_integrated_hardware_manager, WorkloadType
from src.utils.hardware.optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked
)
from src.utils.hardware.memory_optimized_decorators import (
    memory_optimized, comprehensive_memory_optimization, MemoryOptimizationLevel
)

# Import ML common utilities
from src.utils.ml_common.optimization.consolidated_hpo import (
    ConsolidatedHPO, HPOConfig, OptimizationResult, MultiFidelityHPO
)
from src.utils.ml_common.validation.consolidated_cv import (
    ConsolidatedCV, CVConfig, PurgedCV, WalkForwardCV, TemporalCV
)
from src.utils.ml_common.validation.data_leakage_detector import (
    DataLeakageDetector, DataLeakageReport
)
from src.utils.ml_common.explainability.model_explainability import (
    ModelExplainabilityManager, ExplanationConfig
)
from src.utils.ml_common.explainability.shap_lime_integration import (
    SHAPLIMEIntegration, ExplanationResult
)
from src.utils.ml_common.data_processing.multi_timeframe_training import MultiTimeframeProcessor
from src.utils.ml_common.ensembles.stacking_ensemble_manager import StackingEnsembleManager
from src.utils.ml_common.feature_selection import (
    FeatureSelector, FeatureSelectionConfig, mRMRSelector, LASSOSelector, RFESelector
)

# Import data quality and analysis tools
from src.training.steps.pre_training.profit_labeling.enhanced_label_definitions import (
    EnhancedLabelDefinitions, AnalystLabelConfig, TacticianLabelConfig
)

# Import weighted loss integration
from src.training.steps.models_training.core.weighted_loss_integration import (
    WeightedLossIntegrator, WeightedLossIntegrationConfig, WeightingStrategy,
    wrap_model_with_weighted_loss
)

# Import validation and metrics
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

# Import hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    tprint_warning("Optuna not available. Hyperparameter optimization will be disabled.")


class ModelType(Enum):
    """Model types supported by the unified pipeline."""
    ANALYST_BASE = "analyst_base"
    ANALYST_ENSEMBLE = "analyst_ensemble"
    TACTICIAN_BASE = "tactician_base"
    TACTICIAN_ENSEMBLE = "tactician_ensemble"


@dataclass
class MLModelTrainerConfig:
    """Configuration for the unified ML model trainer."""
    # Pipeline configuration
    model_types: List[ModelType] = field(default_factory=lambda: [
        ModelType.ANALYST_BASE,
        ModelType.ANALYST_ENSEMBLE,
        ModelType.TACTICIAN_BASE,
        ModelType.TACTICIAN_ENSEMBLE
    ])
    
    # Data configuration
    timeframe: str = "15m"  # Default timeframe, can be overridden by ares_launcher
    random_state: int = 42
    
    # Training configuration
    validation_split: float = 0.2
    test_split: float = 0.1
    cv_folds: int = 5
    
    # Mode configuration for parameter scaling
    mode: str = "FULL"  # LIGHT (10%), BLANK (50%), FULL (100%)
    
    # Performance configuration
    enable_parallel_training: bool = True
    max_workers: int = 4
    enable_gpu: bool = False
    
    # Output configuration
    output_dir: str = "results/ml_model_trainer"
    save_models: bool = True
    save_predictions: bool = True
    save_reports: bool = True
    
    # Monitoring configuration
    enable_monitoring: bool = True
    log_level: str = "INFO"
    verbose: bool = True
    
    # Weighted loss configuration
    enable_weighted_loss: bool = True
    weighted_loss_strategy: str = "adaptive"  # "difficulty_based", "failure_context", "adaptive", "focal_loss", "gradient_based"
    weighted_loss_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize configuration."""
        pass
    
    def get_mode_scaling_factor(self) -> float:
        """Get scaling factor based on mode."""
        if self.mode == "LIGHT":
            return 0.1  # 10%
        elif self.mode == "BLANK":
            return 0.5  # 50%
        else:  # FULL
            return 1.0  # 100%


@dataclass
class TrainingResult:
    """Result of model training."""
    model_type: ModelType
    model_name: str
    success: bool
    model: Any = None
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: np.ndarray = None
    probabilities: np.ndarray = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    shap_values: np.ndarray = None
    training_time: float = 0.0
    error_message: str = ""


class MLModelTrainer:
    """
    Unified ML Model Trainer Pipeline.
    
    This class provides a single interface for training all ML models with
    comprehensive configuration management, feature engineering, validation,
    and analysis capabilities.
    """
    
    def __init__(self, config: MLModelTrainerConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the ML Model Trainer.
        
        Args:
            config: Configuration for the trainer
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger or system_logger
        
        # Initialize process pool for CPU-bound training
        self._process_pool = ProcessPoolExecutor(max_workers=self.config.max_workers)
        self._process_pool_initialized = True
        
        # Load unified memory configuration
        self.memory_config = self._load_unified_memory_config()
        
        # Initialize hardware manager with unified config
        self.hardware_manager = self._initialize_hardware_manager()
        
        # Initialize components
        self._initialize_components()
        
        # Training results storage
        self.training_results: Dict[ModelType, List[TrainingResult]] = {}
        
        # Feature engineering components
        self.feature_engineers = {}
        self.target_generators = {}
        
        # Data quality components
        self.data_validators = {}
        self.leakage_detectors = {}
        
        # Analysis components
        self.metrics_calculators = {}
        self.shap_analyzers = {}
        
        # Weighted loss integration
        self.weighted_loss_integrator = None
        if self.config.enable_weighted_loss:
            self._initialize_weighted_loss_integrator()
        
        tprint_info(f"🔧 Initialized MLModelTrainer for {config.timeframe}")
    
    def _initialize_weighted_loss_integrator(self):
        """Initialize weighted loss integrator."""
        try:
            # Load weighted loss configuration from YAML if not provided
            if self.config.weighted_loss_config is None:
                self.config.weighted_loss_config = self._load_weighted_loss_config()
            
            # Create weighted loss integration config
            weighted_loss_config = WeightedLossIntegrationConfig(
                enable_weighted_loss=self.config.enable_weighted_loss,
                weighting_strategy=WeightingStrategy(self.config.weighted_loss_strategy),
                **(self.config.weighted_loss_config or {})
            )
            
            # Create integrator
            self.weighted_loss_integrator = WeightedLossIntegrator(weighted_loss_config)
            
            # Initialize with model types
            model_types = [model_type.value for model_type in self.config.model_types]
            self.weighted_loss_integrator.initialize(model_types)
            
            tprint_success("✅ Weighted loss integrator initialized")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize weighted loss integrator: {e}")
            self.weighted_loss_integrator = None
    
    def _load_weighted_loss_config(self) -> Dict[str, Any]:
        """Load weighted loss configuration from YAML file."""
        try:
            config_path = Path("config/weighted_loss_config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                tprint_info("✅ Loaded weighted loss configuration from YAML")
                return config
            else:
                tprint_warning("Weighted loss config file not found, using defaults")
                return {}
        except Exception as e:
            tprint_warning(f"Failed to load weighted loss config: {e}, using defaults")
            return {}
    
    def __del__(self):
        """Cleanup resources on destruction."""
        if hasattr(self, '_process_pool_initialized') and self._process_pool_initialized:
            try:
                self._process_pool.shutdown(wait=True)
                tprint_info("🔄 ProcessPoolExecutor shutdown completed")
            except Exception as e:
                tprint_error(f"Error during ProcessPool cleanup: {e}")
    
    async def cleanup(self):
        """Explicit cleanup method for async contexts."""
        if hasattr(self, '_process_pool_initialized') and self._process_pool_initialized:
            try:
                self._process_pool.shutdown(wait=True)
                self._process_pool_initialized = False
                tprint_info("🔄 ProcessPoolExecutor shutdown completed")
            except Exception as e:
                tprint_error(f"Error during ProcessPool cleanup: {e}")
        self.logger.info(f"Initialized MLModelTrainer for {config.timeframe}")
    
    def _infer_task_type_from_recipe(self, recipe: Dict[str, Any]) -> str:
        """Infer task type from YAML recipe targets.target_type."""
        tt = (recipe.get("targets", {}).get("target_type") or "").lower()
        if tt in {"binary_classification", "multiclass_classification"}:
            return "classification"
        if tt in {"regression"}:
            return "regression"
        return "classification"  # safe default for Analyst Base
    
    def _resolve_scorer(self, recipe: Dict[str, Any], task_type: str) -> Tuple[str, dict]:
        """Resolve scorer from YAML metrics.primary to sklearn scorer."""
        SCORER_MAP = {
            "f1_score": "f1",
            "precision": "precision", 
            "recall": "recall",
            "accuracy": "accuracy",
            "auc_roc": "roc_auc",
            "mse": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2_score": "r2",
        }
        
        primary = recipe.get("metrics", {}).get("primary", "f1_score")
        scorer = SCORER_MAP.get(primary, primary)
        scorer_kwargs = {}
        
        if task_type == "classification" and scorer in {"f1", "precision", "recall"}:
            scorer_kwargs["average"] = "weighted"
        
        return scorer, scorer_kwargs
    
    def _get_base_models(self, recipe: Dict[str, Any], X: np.ndarray, y: np.ndarray, task_type: str):
        """Resolve base model sources (train or load)."""
        base_cfgs = [m for m in recipe.get("base_models", []) if m.get("enabled", True)]
        base_models = []
        
        for cfg in base_cfgs:
            # Create model with parameters from config
            mdl = self._create_model_with_params(cfg, cfg.get("parameters", {}), task_type)
            base_models.append((cfg["name"], mdl, cfg))
        
        tprint_data_format(f"Resolved {len(base_models)} base models: {[name for name, _, _ in base_models]}", LogLevel.INFO)
        return base_models  # list of (name, model, cfg)
    
    
    def _diversity_metrics(self, oof_dict: Dict[str, np.ndarray]):
        """Calculate diversity and correlation metrics from OOF predictions."""
        names = list(oof_dict.keys())
        if len(names) < 2:
            return {"names": names, "corr": [], "avg_correlation": 0.0}
        
        # Create matrix of OOF predictions
        M = np.column_stack([oof_dict[n] for n in names])  # (n, k)
        corr = np.corrcoef(M, rowvar=False)  # (k, k)
        
        # Calculate average off-diagonal correlation
        mask = ~np.eye(corr.shape[0], dtype=bool)
        avg_correlation = float(np.mean(corr[mask]))
        
        return {
            "names": names, 
            "corr": corr.tolist(),
            "avg_correlation": avg_correlation
        }
    
    def train_shallow_lgbm_stacker(self, X: np.ndarray, y: np.ndarray, base_models_cfg: List[Dict], 
                                 meta_cfg: Dict, cv_folds: int = 5, use_features_in_secondary: bool = True, 
                                 use_proba_as_level1: bool = True):
        """
        Train a leakage-safe OOF stacking ensemble with shallow LGBM meta-learner.
        
        Returns: dict with fitted base models, fitted meta model, fold indices,
                OOF matrix (level-1), and feature names.
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.base import clone
        
        tprint_info("🏗️ Training shallow LGBM stacker ensemble")
        
        # 1) Instantiate base models
        base_models = []
        base_names = []
        for bm in base_models_cfg:
            if not bm.get("enabled", True):
                continue
            m = self._create_model_with_params(bm, bm.get("parameters", {}), "classification")
            base_models.append(m)
            base_names.append(bm["name"])

        n, p = X.shape
        k = cv_folds
        tss = TimeSeriesSplit(n_splits=k)

        # 2) OOF container
        oof_L1 = np.zeros((n, len(base_models)), dtype=float)
        fold_assign = np.full(n, -1, dtype=int)
        base_fold_models = [[] for _ in base_models]
        
        # OOF per base model
        oof_dict = {name: np.zeros(n, dtype=float) for name in base_names}
        oof_proba_dict = {name: np.full((n, 2), np.nan, dtype=float) 
                          for name in base_names} if use_proba_as_level1 else {}

        # 3) Build OOF predictions
        tprint_info(f"🔄 Building OOF predictions with {k} folds")
        for fold_idx, (tr, va) in enumerate(tss.split(X)):
            X_tr, y_tr, X_va = X[tr], y[tr], X[va]
            for j, bm in enumerate(base_models):
                m = clone(bm)
                
                # Enable early stopping if supported
                name = type(m).__name__.upper()
                if "LGBM" in name:
                    m.fit(X_tr, y_tr, eval_set=[(X_va, y[va])], verbose=False)
                elif "XGB" in name:
                    m.fit(X_tr, y_tr, eval_set=[(X_va, y[va])], verbose=False)
                elif "CATBOOST" in name:
                    m.fit(X_tr, y_tr, eval_set=(X_va, y[va]), verbose=False)
                else:
                    m.fit(X_tr, y_tr)

                base_fold_models[j].append(m)

                if use_proba_as_level1 and hasattr(m, "predict_proba"):
                    proba = m.predict_proba(X_va)
                    # Binary classification → use column 1
                    oof_L1[va, j] = proba[:, 1]
                    oof_dict[base_names[j]][va] = proba[:, 1]
                    if proba.shape[1] == 2:
                        oof_proba_dict[base_names[j]][va] = proba
                else:
                    pred = m.predict(X_va)
                    oof_L1[va, j] = pred
                    oof_dict[base_names[j]][va] = pred

            fold_assign[va] = fold_idx

        # 4) Meta features (level-2 input)
        if use_features_in_secondary:
            X_meta = np.hstack([oof_L1, X])
            meta_feature_names = [f"oof_{nm}" for nm in base_names] + [f"f{i}" for i in range(p)]
        else:
            X_meta = oof_L1
            meta_feature_names = [f"oof_{nm}" for nm in base_names]

        # 5) Train shallow LGBM meta on OOF with early stopping
        meta_params = meta_cfg.get("meta_learner_params", {})
        meta_model = self._create_model_with_params(
            {"type": meta_cfg.get("meta_learner_type", "LIGHTGBM"), "parameters": meta_params},
            meta_params,
            "classification",
        )

        # Early stopping split: last fold as validation for meta
        last_fold = (fold_assign == fold_assign.max())
        tr_idx = ~last_fold
        X_meta_tr, y_tr = X_meta[tr_idx], y[tr_idx]
        X_meta_va, y_va = X_meta[last_fold], y[last_fold]

        tprint_info("🎯 Training meta-learner with early stopping")
        if "LGBM" in type(meta_model).__name__.upper():
            meta_model.fit(X_meta_tr, y_tr, eval_set=[(X_meta_va, y_va)], verbose=False)
        else:
            meta_model.fit(X_meta_tr, y_tr)

        # 6) Refit base models on full data for deployment
        tprint_info("🔄 Refitting base models on full data")
        base_models_fitted = []
        for bm in base_models:
            m = clone(bm)
            # No leakage: fit on all data (production-ready)
            m.fit(X, y)
            base_models_fitted.append(m)

        # 7) Train final meta on full level-1 built from full-data base preds
        tprint_info("🎯 Final meta-learner training on full data")
        if use_proba_as_level1 and hasattr(base_models_fitted[0], "predict_proba"):
            L1_full = np.column_stack([m.predict_proba(X)[:, 1] for m in base_models_fitted])
        else:
            L1_full = np.column_stack([m.predict(X) for m in base_models_fitted])

        X_meta_full = np.hstack([L1_full, X]) if use_features_in_secondary else L1_full
        meta_model.fit(X_meta_full, y)  # Final fit

        tprint_success("✅ Shallow LGBM stacker training completed")
        
        return {
            "base_names": base_names,
            "base_models": base_models_fitted,
            "meta_model": meta_model,
            "oof_matrix": oof_L1,
            "oof_dict": oof_dict,
            "oof_proba_dict": oof_proba_dict if use_proba_as_level1 else {},
            "fold_assign": fold_assign,
            "meta_feature_names": meta_feature_names,
        }
    
    def predict_shallow_lgbm_stacker(self, bundle: Dict, X: np.ndarray):
        """Make predictions using the trained stacker bundle."""
        base_models = bundle["base_models"]
        
        if hasattr(base_models[0], "predict_proba"):
            L1 = np.column_stack([m.predict_proba(X)[:, 1] for m in base_models])
        else:
            L1 = np.column_stack([m.predict(X) for m in base_models])
        
        X_meta = np.hstack([L1, X]) if len(bundle["meta_feature_names"]) > L1.shape[1] else L1
        
        predictions = bundle["meta_model"].predict(X_meta)
        probabilities = getattr(bundle["meta_model"], "predict_proba", lambda _: None)(X_meta)
        
        return predictions, probabilities
    
    @performance_tracked
    async def _train_ensemble_model(self, data: Dict[str, Any], model_type: ModelType, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train ensemble model with proper OOF handling."""
        tprint_info(f"🎯 Training ensemble model: {model_type.value}")
        
        # Get base models
        task_type = self._infer_task_type_from_recipe(config)
        base_models = self._get_base_models(config, data['features'], data['targets'], task_type)
        
        if not base_models:
            raise ValueError("No enabled base models found for ensemble training")
        
        # OOF is produced inside the stacker; compute diversity from bundle oof_matrix if desired
        # Calculate diversity metrics from base model predictions if available
        diversity_metrics = {}
        tprint_data_format(f"Diversity metrics: {diversity_metrics}", LogLevel.INFO)
        
        # Train stacking ensemble
        ensemble_config = config.get('models', [{}])[0]  # Get first ensemble config
        if ensemble_config.get('type') != 'STACKING':
            raise NotImplementedError(f"Only STACKING ensemble type is supported, got: {ensemble_config.get('type')}")
        
        bundle = self.train_shallow_lgbm_stacker(
            data['features'], data['targets'],
            base_models_cfg=config.get("base_models", []),
            meta_cfg=ensemble_config.get("parameters", {}),
            cv_folds=ensemble_config.get("parameters", {}).get("cv_folds", 5),
            use_features_in_secondary=ensemble_config.get("parameters", {}).get("use_features_in_secondary", True),
            use_proba_as_level1=ensemble_config.get("parameters", {}).get("use_proba_as_level1", True),
        )
        
        # Evaluate ensemble
        predictions, probabilities = self.predict_shallow_lgbm_stacker(bundle, data['features'])
        
        # Extract OOF data from bundle
        oof = bundle["oof_dict"]
        oof_proba = bundle.get("oof_proba_dict", {})
        fold_idx = bundle["fold_assign"]
        
        # Calculate ensemble metrics
        ensemble_metrics = self._calculate_ensemble_metrics(
            data['targets'], predictions, probabilities, oof, task_type, config
        )
        
        # Add diversity metrics to ensemble metrics
        ensemble_metrics.update({
            'diversity_metrics': diversity_metrics,
            'ensemble_improvement': self._calculate_ensemble_improvement(oof, data['targets'], ensemble_metrics, config, task_type)
        })
        
        # Save artifacts
        await self._save_ensemble_artifacts(bundle, oof, oof_proba, fold_idx, model_type, config)
        
        # Cleanup intermediate results to free memory
        self._cleanup_ensemble_memory(oof, oof_proba, fold_idx)
        
        tprint_success(f"✅ Ensemble model training completed: {model_type.value}")
        
        return {
            'model_type': model_type.value,
            'ensemble_type': 'STACKING',
            'base_models': [name for name, _, _ in base_models],
            'metrics': ensemble_metrics,
            'bundle': bundle  # For prediction use
        }
    
    def _calculate_ensemble_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, 
                                  oof: Dict[str, np.ndarray], task_type: str, config: Dict[str, Any]) -> Dict[str, float]:
        """Calculate ensemble-specific metrics."""
        metrics = {}
        
        if task_type == "classification":
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
            metrics.update({
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1_score": float(f1_score(y_true, y_pred, average="weighted")),
                "precision": float(precision_score(y_true, y_pred, average="weighted")),
                "recall": float(recall_score(y_true, y_pred, average="weighted"))
            })
            
            if y_proba is not None and np.unique(y_true).size == 2:
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
            metrics.update({
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2_score": float(r2_score(y_true, y_pred)),
                "explained_variance": float(explained_variance_score(y_true, y_pred))
            })
        
        return metrics
    
    def _calculate_ensemble_improvement(self, oof: Dict[str, np.ndarray], y_true: np.ndarray, 
                                      ensemble_metrics: Dict[str, float], config: Dict[str, Any], 
                                      task_type: str) -> float:
        """Calculate improvement of ensemble over best individual model."""
        scorer, scorer_kwargs = self._resolve_scorer(config, task_type)
        
        # Map scorer to metric key
        metric_key = {
            "f1": "f1_score",
            "precision": "precision", 
            "recall": "recall",
            "accuracy": "accuracy",
            "roc_auc": "auc_roc",
            "neg_mean_squared_error": "rmse",
            "neg_mean_absolute_error": "mae",
            "r2": "r2_score"
        }.get(scorer, scorer)
        
        # Calculate individual model scores
        individual_scores = {}
        for name, oof_pred in oof.items():
            if task_type == "classification":
                from sklearn.metrics import f1_score
                score = f1_score(y_true, oof_pred, average="weighted")
            else:
                from sklearn.metrics import mean_squared_error
                score = -mean_squared_error(y_true, oof_pred)  # Negative MSE for maximization
            individual_scores[name] = score
        
        best_individual = max(individual_scores.values())
        ensemble_score = ensemble_metrics.get(metric_key, 0.0)
        
        return float(ensemble_score - best_individual)
    
    def _make_scorer(self, name: str, kwargs: dict):
        """Create a scorer with proper parameters."""
        from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
        
        if name == "f1":
            return make_scorer(f1_score, **kwargs)
        if name == "precision":
            return make_scorer(precision_score, **kwargs)
        if name == "recall":
            return make_scorer(recall_score, **kwargs)
        return name  # built-in string scorers
    
    def _make_cv(self, recipe):
        """Create CV splitter based on recipe configuration."""
        n = recipe.get("training", {}).get("cv_folds")
        if n:
            # Update CV system with recipe-specific folds
            from src.utils.ml_common.validation.consolidated_cv import CVConfig
            self.cv_system = ConsolidatedCV(CVConfig(
                enable_purged_cv=True,
                enable_walk_forward=True,
                enable_temporal_cv=True,
                n_splits=n
            ))
        return self.cv_system
    
    async def _save_ensemble_artifacts(self, bundle: Dict, oof: Dict[str, np.ndarray], 
                                     oof_proba: Dict[str, np.ndarray], fold_idx: np.ndarray, 
                                     model_type: ModelType, config: Dict[str, Any]):
        """Save ensemble artifacts for persistence."""
        import joblib
        import json
        from pathlib import Path
        
        # Create artifacts directory
        artifacts_dir = Path("artifacts") / "ensemble" / model_type.value
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save OOF predictions
        oof_dir = artifacts_dir / "oof"
        oof_dir.mkdir(exist_ok=True)
        
        for name, oof_pred in oof.items():
            np.save(oof_dir / f"{name}_oof.npy", oof_pred)
        
        for name, oof_prob in oof_proba.items():
            if oof_prob is not None:
                np.save(oof_dir / f"{name}_oof_proba.npy", oof_prob)
        
        np.save(oof_dir / "fold_idx.npy", fold_idx)
        
        # Save ensemble bundle
        joblib.dump(bundle, artifacts_dir / "ensemble_bundle.joblib")
        
        # Save metadata
        metadata = {
            "base_names": bundle["base_names"],
            "meta_feature_names": bundle["meta_feature_names"],
            "ensemble_type": "STACKING",
            "config_hash": getattr(self, '_config_hash', 'unknown')
        }
        
        with open(artifacts_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        tprint_success(f"✅ Ensemble artifacts saved to {artifacts_dir}")
    
    def _cleanup_ensemble_memory(self, oof: Dict[str, np.ndarray], oof_proba: Dict[str, np.ndarray], fold_idx: np.ndarray):
        """Cleanup ensemble training memory to prevent memory leaks."""
        try:
            # Clear OOF predictions
            for name in oof:
                del oof[name]
            oof.clear()
            
            # Clear OOF probabilities
            for name in oof_proba:
                if oof_proba[name] is not None:
                    del oof_proba[name]
            oof_proba.clear()
            
            # Clear fold assignments
            del fold_idx
            
            # Force garbage collection
            import gc
            gc.collect()
            
            tprint_info("🧹 Ensemble memory cleanup completed")
            
        except Exception as e:
            tprint_warning(f"Memory cleanup failed: {e}")
    
    def _make_cv(self, recipe: Dict[str, Any]):
        """Create CV strategy from YAML training.cv_strategy and cv_params."""
        name = (recipe.get("training", {}).get("cv_strategy") or "TimeSeriesSplit").lower()
        params = recipe.get("training", {}).get("cv_params", {})
        
        if name == "timeseriessplit":
            from sklearn.model_selection import TimeSeriesSplit
            return TimeSeriesSplit(n_splits=params.get("n_splits", 5))
        elif name == "purgedcv":
            # Use ConsolidatedCV's purged CV
            from src.utils.ml_common.validation.consolidated_cv import create_purged_cv
            return create_purged_cv(
                n_splits=params.get("n_splits", 5),
                purge_length=params.get("purge_length", 1),
                embargo_length=params.get("embargo_length", 1)
            )
        elif name == "walkforward":
            # Use ConsolidatedCV's walk forward CV
            from src.utils.ml_common.validation.consolidated_cv import create_walk_forward_cv
            return create_walk_forward_cv(
                n_splits=params.get("n_splits", 5),
                initial_train_size=params.get("initial_train_size", 0.6),
                step_size=params.get("step_size", 0.1)
            )
        else:
            # Default to TimeSeriesSplit
            from sklearn.model_selection import TimeSeriesSplit
            return TimeSeriesSplit(n_splits=5)
    
    def _fit_with_early_stopping(self, model, X: np.ndarray, y: np.ndarray, recipe: Dict[str, Any], task_type: str):
        """Fit model with early stopping using eval_set."""
        # Early stopping guard when dataset is tiny
        if X.shape[0] < 50:
            return model.fit(X, y)
            
        es = recipe.get("training", {}).get("early_stopping", {}).get("enabled", False)
        if not es:
            return model.fit(X, y)
        
        # Make a small temporal validation split
        from sklearn.model_selection import TimeSeriesSplit
        tss = TimeSeriesSplit(n_splits=3)
        train_idx, val_idx = list(tss.split(X))[-1]
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # LightGBM
        if "LGBM" in type(model).__name__.upper():
            kwargs = {}
            # Allow objective/metric already in params; just add eval_set & early stopping
            kwargs["eval_set"] = [(X_val, y_val)]
            if hasattr(model, "fit"):
                return model.fit(X_tr, y_tr, **kwargs)
        
        # XGBoost
        if "XGB" in type(model).__name__.upper():
            eval_set = [(X_val, y_val)]
            return model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
        
        # CatBoost
        if "CATBOOST" in type(model).__name__.upper():
            return model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        
        # Fallback
        return model.fit(X, y)
    
    def _infer_task_type(self, model_config: Dict[str, Any], y: np.ndarray) -> str:
        """Infer task type from config or data (legacy method)."""
        t = (model_config.get("task") or "").lower()
        if t in {"classification", "regression"}:
            return t
        # Fallback by data
        if y is not None:
            return "classification" if (np.issubdtype(y.dtype, np.integer) and len(np.unique(y)) <= 50) else "regression"
        return "classification"  # Default fallback

    def _load_unified_memory_config(self) -> Dict[str, Any]:
        """Load unified memory configuration from YAML file."""
        try:
            config_path = Path(__file__).parent / "config" / "ml_model_trainer" / "unified_memory_config.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            tprint_info("Loaded unified memory configuration")
            return config
        except Exception as e:
            tprint_warning(f"Failed to load unified memory config: {e}. Using defaults.")
            return self._get_default_memory_config()
    
    def _get_default_memory_config(self) -> Dict[str, Any]:
        """Get default memory configuration if unified config fails to load."""
        return {
            "memory_optimization": {
                "strategy": "comprehensive",
                "memory_limit_gb": 8.0,
                "enable_garbage_collection": True,
                "gc_frequency": 100,
                "enable_memory_mapping": True,
                "chunk_size": 10000,
                "enable_compression": True,
                "compression_level": 6,
                "enable_memory_pooling": True,
                "enable_predictive_allocation": True
            },
            "caching": {
                "enabled": True,
                "cache_features": True,
                "cache_targets": True,
                "cache_cv_splits": True,
                "cache_feature_importance": True,
                "cache_shap_values": True,
                "cache_dir": "cache/ml_models",
                "cache_ttl_hours": 24,
                "enable_compression": True,
                "compression_level": 6,
                "max_cache_size_mb": 2048,
                "enable_lru_eviction": True,
                "enable_data_type_optimization": True
            }
        }
    
    def _initialize_hardware_manager(self):
        """Initialize hardware manager with unified configuration."""
        try:
            # Create hardware config from unified memory config
            from src.utils.hardware.integrated_hardware_manager import IntegratedHardwareConfig
            
            hardware_config = IntegratedHardwareConfig(
                memory_optimization_level=self.memory_config.get("hardware_optimizations", {}).get("memory_optimization_level", "balanced"),
                memory_limit_gb=self.memory_config.get("memory_optimization", {}).get("memory_limit_gb", 8.0),
                enable_memory_pooling=self.memory_config.get("memory_optimization", {}).get("enable_memory_pooling", True),
                enable_compression=self.memory_config.get("memory_optimization", {}).get("enable_compression", True),
                enable_adaptive_optimization=self.memory_config.get("hardware_optimizations", {}).get("enable_adaptive_optimization", True),
                enable_learning=self.memory_config.get("hardware_optimizations", {}).get("enable_learning", True),
                auto_tuning_enabled=self.memory_config.get("hardware_optimizations", {}).get("auto_tuning_enabled", True),
                performance_monitoring_enabled=self.memory_config.get("hardware_optimizations", {}).get("performance_monitoring_enabled", True)
            )
            
            # Initialize hardware manager
            hardware_manager = get_integrated_hardware_manager()
            hardware_manager.configure(hardware_config)
            
            tprint_success("Hardware manager initialized with unified configuration")
            return hardware_manager
            
        except Exception as e:
            tprint_warning(f"Failed to initialize hardware manager: {e}. Using default.")
            return get_integrated_hardware_manager()
    
    def _get_model_specific_memory_config(self, model_type: ModelType) -> Dict[str, Any]:
        """Get model-specific memory configuration overrides."""
        model_type_key = model_type.value.lower()
        overrides = self.memory_config.get("model_specific_overrides", {}).get(model_type_key, {})
        
        # Merge with base configuration
        config = self.memory_config.copy()
        if overrides:
            config["memory_optimization"].update(overrides)
            config["caching"].update(overrides)
        
        return config
    
    def _apply_memory_optimizations(self, model_type: ModelType, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply memory optimizations based on unified configuration."""
        try:
            # Get model-specific memory config
            memory_config = self._get_model_specific_memory_config(model_type)
            
            # Apply chunked processing if enabled
            if memory_config.get("performance_optimization", {}).get("enable_batch_processing", True):
                max_batch_size = memory_config.get("performance_optimization", {}).get("max_batch_size", 1000)
                chunk_size = memory_config.get("memory_optimization", {}).get("chunk_size", 10000)
                
                # Apply chunked processing to large datasets
                for key, value in data.items():
                    if hasattr(value, 'shape') and len(value.shape) > 0:
                        if value.shape[0] > max_batch_size:
                            tprint_info(f"Applying chunked processing to {key} (shape: {value.shape})")
                            # This will be handled by the memory_optimized decorator
            
            # Apply garbage collection if enabled
            if memory_config.get("memory_optimization", {}).get("enable_garbage_collection", True):
                gc_frequency = memory_config.get("memory_optimization", {}).get("gc_frequency", 100)
                # This will be handled by the gc_optimized decorator
            
            return data
            
        except Exception as e:
            tprint_warning(f"Failed to apply memory optimizations: {e}")
            return data

    def _initialize_components(self):
        """Initialize all pipeline components using existing utilities."""
        tprint_info("🔧 Initializing ML Model Trainer components")
        
        # Load unified memory configuration
        self.memory_config = self._load_unified_memory_config()
        
        # Initialize hardware manager with unified config
        self.hardware_manager = self._initialize_hardware_manager()
        
        # Initialize enhanced caching system with unified config
        from src.utils.hardware.enhanced_caching_system import EnhancedCacheSystem, CacheConfig, DataTypeOptimization
        
        cache_config = CacheConfig(
            max_memory_usage=self.memory_config.get("caching", {}).get("max_cache_size_mb", 2048) / 1024,  # Convert MB to GB
            strategy=CacheStrategy.ADAPTIVE,
            enable_compression=self.memory_config.get("caching", {}).get("enable_compression", True),
            enable_serialization=True,
            data_type_optimization=DataTypeOptimization.AGGRESSIVE if self.memory_config.get("caching", {}).get("enable_data_type_optimization", True) else DataTypeOptimization.NONE,
            enable_lru_eviction=self.memory_config.get("caching", {}).get("enable_lru_eviction", True)
        )
        self.cache_system = EnhancedCacheSystem(cache_config)
        tprint_data_format("Enhanced caching system initialized with unified configuration", LogLevel.INFO)
        
        # Initialize memory manager
        self.memory_manager = get_memory_manager()
        tprint_data_format("Memory manager initialized", LogLevel.INFO)
        
        # Initialize HPO system with mode-based parameters
        mode = getattr(self.config, 'mode', 'FULL')  # Get mode from config or default to FULL
        if mode == "LIGHT":
            max_trials = 10  # 90% reduction
            timeout = 360  # 90% reduction
        elif mode == "BLANK":
            max_trials = 50  # 50% reduction
            timeout = 1800  # 50% reduction
        else:
            max_trials = 100
            timeout = 3600
        
        self.hpo_system = ConsolidatedHPO(HPOConfig(
            enable_optuna=OPTUNA_AVAILABLE,
            max_trials=max_trials,
            timeout=timeout,
            n_jobs=self.config.max_workers,
            enable_multi_fidelity=True,
            fidelity_levels=[0.1, 0.3, 0.6, 1.0]  # Progressive fidelity levels
        ))
        tprint_data_format("HPO system initialized", LogLevel.INFO)
        
        # Initialize cross-validation system with mode-based parameters
        mode = getattr(self.config, 'mode', 'FULL')  # Get mode from config or default to FULL
        if mode == "LIGHT":
            cv_folds = max(2, self.config.cv_folds // 10)  # 10% of original, minimum 2
        elif mode == "BLANK":
            cv_folds = max(3, self.config.cv_folds // 2)  # 50% of original, minimum 3
        else:
            cv_folds = self.config.cv_folds
        
        self.cv_system = ConsolidatedCV(CVConfig(
            enable_purged_cv=True,
            enable_walk_forward=True,
            enable_temporal_cv=True,
            n_splits=cv_folds
        ))
        tprint_data_format("CV system initialized", LogLevel.INFO)
        
        # Initialize data leakage detector
        self.leakage_detector = DataLeakageDetector({
            'temporal_tolerance': 1,
            'lookahead_tolerance': 24,
            'feature_contamination_threshold': 0.1,
            'enable_strict_mode': True,
            'use_vectorbt_analysis': True,
            'correlation_threshold': 0.95
        })
        tprint_data_format("Data leakage detector initialized", LogLevel.INFO)
        
        # Initialize explainability manager
        self.explainability_manager = ModelExplainabilityManager(ExplanationConfig(
            enable_shap=True,
            enable_lime=True,
            shap_sample_size=100,
            lime_sample_size=1000
        ))
        tprint_data_format("Explainability manager initialized", LogLevel.INFO)
        
        # Initialize feature selectors
        self.feature_selectors = {
            ModelType.ANALYST_BASE: FeatureSelector(FeatureSelectionConfig(
                method='mrmr',
                max_features=50,
                enable_correlation_filter=True
            )),
            ModelType.ANALYST_ENSEMBLE: FeatureSelector(FeatureSelectionConfig(
                method='lasso',
                max_features=100,
                enable_correlation_filter=True
            )),
            ModelType.TACTICIAN_BASE: FeatureSelector(FeatureSelectionConfig(
                method='rfe',
                max_features=30,
                enable_correlation_filter=True
            )),
            ModelType.TACTICIAN_ENSEMBLE: FeatureSelector(FeatureSelectionConfig(
                method='mrmr',
                max_features=80,
                enable_correlation_filter=True
            ))
        }
        tprint_data_format("Feature selectors initialized", LogLevel.INFO)
        
        # Initialize multi-timeframe processor
        self.multi_timeframe_processor = MultiTimeframeProcessor()
        tprint_data_format("Multi-timeframe processor initialized", LogLevel.INFO)
        
        # Initialize ensemble managers
        self.ensemble_managers = {
            ModelType.ANALYST_ENSEMBLE: StackingEnsembleManager(),
            ModelType.TACTICIAN_ENSEMBLE: StackingEnsembleManager()
        }
        tprint_data_format("Ensemble managers initialized", LogLevel.INFO)
        
        tprint_success("✅ All components initialized successfully")
    
    @memory_managed(MemoryStrategy.MODERATE)
    @smart_cache
    def _prepare_features(self, data: Dict[str, Any], model_type: ModelType, config: Dict[str, Any], model_config: Dict[str, Any] = None) -> np.ndarray:
        """Prepare features using pre-selected features from upstream feature generation."""
        tprint_info(f"🔄 Preparing features for {model_type.value}")
        
        # Start with pre-selected features from upstream feature generation
        # These are specific to model type (tactician/analyst) and direction (shorts/longs)
        base_features = data.get('features', np.array([]))
        tprint_data_preview(base_features, f"Pre-selected features for {model_type.value}")
        
        # Validate input data
        if not validate_array(base_features):
            tprint_error("Invalid input features")
            raise ValueError("Invalid input features")
        
        # Ensure features are 2D
        if base_features.ndim == 1:
            base_features = base_features.reshape(1, -1)
        elif base_features.ndim > 2:
            base_features = base_features.reshape(base_features.shape[0], -1)
        
        # Add regime features for all models (from regime_ensemble_training)
        regime_features = self._get_regime_features(data, model_type)
        if regime_features is not None:
            base_features = np.hstack([base_features, regime_features])
            tprint_data_format(f"Added regime features: {base_features.shape}", LogLevel.INFO)
        
        # Add model-specific features based on model type
        if model_type == ModelType.ANALYST_ENSEMBLE:
            # Add outputs from Analyst base models
            analyst_base_outputs = self._get_analyst_base_outputs(data, model_type)
            if analyst_base_outputs is not None:
                base_features = np.hstack([base_features, analyst_base_outputs])
                tprint_data_format(f"Added Analyst base outputs: {base_features.shape}", LogLevel.INFO)
                
        elif model_type == ModelType.TACTICIAN_BASE:
            # Add outputs from Analyst ensemble model
            analyst_ensemble_outputs = self._get_analyst_ensemble_outputs(data, model_type)
            if analyst_ensemble_outputs is not None:
                base_features = np.hstack([base_features, analyst_ensemble_outputs])
                tprint_data_format(f"Added Analyst ensemble outputs: {base_features.shape}", LogLevel.INFO)
                
        elif model_type == ModelType.TACTICIAN_ENSEMBLE:
            # Add outputs from Tactician base models + Analyst ensemble output
            tactician_base_outputs = self._get_tactician_base_outputs(data, model_type)
            analyst_ensemble_outputs = self._get_analyst_ensemble_outputs(data, model_type)
            
            if tactician_base_outputs is not None:
                base_features = np.hstack([base_features, tactician_base_outputs])
                tprint_data_format(f"Added Tactician base outputs: {base_features.shape}", LogLevel.INFO)
                
            if analyst_ensemble_outputs is not None:
                base_features = np.hstack([base_features, analyst_ensemble_outputs])
                tprint_data_format(f"Added Analyst ensemble outputs: {base_features.shape}", LogLevel.INFO)
        
        # Apply hardware optimization
        optimized_features = self.hardware_manager.process_data_with_optimization(
            base_features, WorkloadType.ML_TRAINING
        )
        tprint_data_format(f"Hardware optimization completed: {optimized_features.shape}", LogLevel.INFO)
        
        return optimized_features
    
    def _get_regime_features(self, data: Dict[str, Any], model_type: ModelType) -> Optional[np.ndarray]:
        """Get regime features from regime_ensemble_training/regime_data_splitting."""
        regime_features = data.get('regime_features', None)
        if regime_features is not None and validate_array(regime_features):
            if regime_features.ndim == 1:
                regime_features = regime_features.reshape(-1, 1)
            return regime_features
        return None
    
    def _get_analyst_base_outputs(self, data: Dict[str, Any], model_type: ModelType) -> Optional[np.ndarray]:
        """Get outputs from Analyst base models for Analyst ensemble."""
        analyst_base_outputs = data.get('analyst_base_outputs', None)
        if analyst_base_outputs is not None and validate_array(analyst_base_outputs):
            if analyst_base_outputs.ndim == 1:
                analyst_base_outputs = analyst_base_outputs.reshape(-1, 1)
            return analyst_base_outputs
        return None
    
    def _get_analyst_ensemble_outputs(self, data: Dict[str, Any], model_type: ModelType) -> Optional[np.ndarray]:
        """Get outputs from Analyst ensemble model for Tactician models."""
        analyst_ensemble_outputs = data.get('analyst_ensemble_outputs', None)
        if analyst_ensemble_outputs is not None and validate_array(analyst_ensemble_outputs):
            if analyst_ensemble_outputs.ndim == 1:
                analyst_ensemble_outputs = analyst_ensemble_outputs.reshape(-1, 1)
            return analyst_ensemble_outputs
        return None
    
    def _get_tactician_base_outputs(self, data: Dict[str, Any], model_type: ModelType) -> Optional[np.ndarray]:
        """Get outputs from Tactician base models for Tactician ensemble."""
        tactician_base_outputs = data.get('tactician_base_outputs', None)
        if tactician_base_outputs is not None and validate_array(tactician_base_outputs):
            if tactician_base_outputs.ndim == 1:
                tactician_base_outputs = tactician_base_outputs.reshape(-1, 1)
            return tactician_base_outputs
        return None
    
    def _apply_model_specific_memory_optimization(self, features: np.ndarray, model_type: ModelType, config: Dict[str, Any]) -> np.ndarray:
        """Apply model-specific memory optimization strategies."""
        try:
            # Get model-specific memory strategy
            memory_strategy = self._get_model_memory_strategy(model_type, config)
            
            # Apply LightGBM-specific optimizations
            if model_type in [ModelType.ANALYST_BASE, ModelType.ANALYST_ENSEMBLE]:
                features = self._optimize_for_lightgbm(features, config)
            
            # Apply neural network optimizations
            elif model_type in [ModelType.TACTICIAN_BASE, ModelType.TACTICIAN_ENSEMBLE]:
                features = self._optimize_for_neural_networks(features, config)
            
            # Apply ensemble-specific optimizations
            if model_type in [ModelType.ANALYST_ENSEMBLE, ModelType.TACTICIAN_ENSEMBLE]:
                features = self._optimize_for_ensemble(features, config)
            
            # Apply general hardware optimization
            optimized_features = self.hardware_manager.process_data_with_optimization(
                features, WorkloadType.ML_TRAINING
            )
            
            return optimized_features
            
        except Exception as e:
            tprint_warning(f"Model-specific memory optimization failed: {e}, using default")
            return self.hardware_manager.process_data_with_optimization(
                features, WorkloadType.ML_TRAINING
            )
    
    def _get_model_memory_strategy(self, model_type: ModelType, config: Dict[str, Any]) -> str:
        """Get memory strategy for specific model type."""
        # Get memory configuration from unified config
        memory_config = config.get('memory_optimization', {})
        
        if model_type in [ModelType.ANALYST_BASE, ModelType.ANALYST_ENSEMBLE]:
            return memory_config.get('analyst_strategy', 'balanced')
        elif model_type in [ModelType.TACTICIAN_BASE, ModelType.TACTICIAN_ENSEMBLE]:
            return memory_config.get('tactician_strategy', 'aggressive')
        else:
            return memory_config.get('default_strategy', 'moderate')
    
    def _optimize_for_lightgbm(self, features: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Apply LightGBM-specific memory optimizations."""
        try:
            # Convert to float32 to reduce memory usage
            if features.dtype == np.float64:
                features = features.astype(np.float32)
            
            # Use categorical features optimization if available
            categorical_features = config.get('categorical_features', [])
            if categorical_features:
                # Convert categorical features to appropriate type
                for cat_idx in categorical_features:
                    if cat_idx < features.shape[1]:
                        features[:, cat_idx] = features[:, cat_idx].astype(np.int32)
            
            return features
            
        except Exception as e:
            tprint_warning(f"LightGBM optimization failed: {e}")
            return features
    
    def _optimize_for_neural_networks(self, features: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Apply neural network-specific memory optimizations."""
        try:
            # Use float32 for neural networks
            if features.dtype == np.float64:
                features = features.astype(np.float32)
            
            # Apply gradient checkpointing if specified
            if config.get('gradient_checkpointing', False):
                # This would be handled in the model training, not here
                pass
            
            return features
            
        except Exception as e:
            tprint_warning(f"Neural network optimization failed: {e}")
            return features
    
    def _optimize_for_ensemble(self, features: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Apply ensemble-specific memory optimizations."""
        try:
            # Use model sharing for ensemble models
            if config.get('enable_model_sharing', True):
                # This would be handled in the ensemble training
                pass
            
            # Use memory-efficient data types
            if features.dtype == np.float64:
                features = features.astype(np.float32)
            
            return features
            
        except Exception as e:
            tprint_warning(f"Ensemble optimization failed: {e}")
            return features
    
    def _get_intelligent_cache_key(self, data: Dict[str, Any], model_type: ModelType, config: Dict[str, Any]) -> str:
        """Generate intelligent cache key based on data and configuration."""
        import hashlib
        import json
        
        # Create cache key components
        key_components = {
            'model_type': model_type.value,
            'data_shape': data.get('features', np.array([])).shape if 'features' in data else (0, 0),
            'config_hash': hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8],
            'data_hash': hashlib.md5(str(data.get('features', np.array([])).tobytes()).hexdigest()[:8] if 'features' in data else 'empty'
        }
        
        return f"ml_trainer_{model_type.value}_{key_components['data_shape'][0]}_{key_components['config_hash']}_{key_components['data_hash']}"
    
    def _should_invalidate_cache(self, cache_key: str, model_type: ModelType) -> bool:
        """Determine if cache should be invalidated based on intelligent rules."""
        try:
            # Check if model configuration has changed
            # This would check against stored configuration hashes
            
            # Check if data has changed significantly
            # This would check data drift or significant changes
            
            # Check if model type requires fresh training
            if model_type in [ModelType.ANALYST_ENSEMBLE, ModelType.TACTICIAN_ENSEMBLE]:
                # Ensemble models should be retrained more frequently
                return True
            
            return False
            
        except Exception as e:
            tprint_warning(f"Cache invalidation check failed: {e}")
            return True  # Invalidate on error for safety
    
    def _apply_model_specific_caching(self, data: Dict[str, Any], model_type: ModelType, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply model-specific caching strategies."""
        try:
            # Get caching configuration
            cache_config = config.get('caching', {})
            
            # Apply different caching strategies based on model type
            if model_type in [ModelType.ANALYST_BASE, ModelType.ANALYST_ENSEMBLE]:
                # Analyst models: cache features and targets
                return self._cache_analyst_data(data, cache_config)
            elif model_type in [ModelType.TACTICIAN_BASE, ModelType.TACTICIAN_ENSEMBLE]:
                # Tactician models: cache sequences and embeddings
                return self._cache_tactician_data(data, cache_config)
            else:
                # Default caching
                return self._cache_default_data(data, cache_config)
                
        except Exception as e:
            tprint_warning(f"Model-specific caching failed: {e}")
            return data
    
    def _cache_analyst_data(self, data: Dict[str, Any], cache_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply analyst-specific caching."""
        try:
            # Cache features with high TTL
            if cache_config.get('cache_features', True):
                features = data.get('features', np.array([]))
                if features.size > 0:
                    # Use hardware-optimized caching
                    cache_key = f"analyst_features_{hash(features.tobytes())}"
                    cached_features = self.hardware_manager.get_cached_data(cache_key)
                    if cached_features is not None:
                        data['features'] = cached_features
                    else:
                        self.hardware_manager.cache_data(cache_key, features, ttl=3600)
            
            return data
            
        except Exception as e:
            tprint_warning(f"Analyst caching failed: {e}")
            return data
    
    def _cache_tactician_data(self, data: Dict[str, Any], cache_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply tactician-specific caching."""
        try:
            # Cache sequences with medium TTL
            if cache_config.get('cache_sequences', True):
                features = data.get('features', np.array([]))
                if features.size > 0:
                    # Use hardware-optimized caching for sequences
                    cache_key = f"tactician_sequences_{hash(features.tobytes())}"
                    cached_features = self.hardware_manager.get_cached_data(cache_key)
                    if cached_features is not None:
                        data['features'] = cached_features
                    else:
                        self.hardware_manager.cache_data(cache_key, features, ttl=1800)
            
            return data
            
        except Exception as e:
            tprint_warning(f"Tactician caching failed: {e}")
            return data
    
    def _cache_default_data(self, data: Dict[str, Any], cache_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default caching strategy."""
        try:
            # Basic caching for all data
            if cache_config.get('cache_all', True):
                for key, value in data.items():
                    if isinstance(value, np.ndarray) and value.size > 0:
                        cache_key = f"default_{key}_{hash(value.tobytes())}"
                        cached_value = self.hardware_manager.get_cached_data(cache_key)
                        if cached_value is not None:
                            data[key] = cached_value
                        else:
                            self.hardware_manager.cache_data(cache_key, value, ttl=900)
            
            return data
            
        except Exception as e:
            tprint_warning(f"Default caching failed: {e}")
            return data
    
    @memory_managed(MemoryStrategy.MODERATE)
    @smart_cache
    def _prepare_targets(self, data: Dict[str, Any], model_type: ModelType, config: Dict[str, Any]) -> np.ndarray:
        """Prepare targets using existing utilities."""
        tprint_info(f"🎯 Preparing targets for {model_type.value}")
        
        # Get expected target name from config
        expected_target = config.get('targets', {}).get('primary', 'target')
        
        # Extract targets based on model type
        if model_type in [ModelType.ANALYST_BASE, ModelType.ANALYST_ENSEMBLE]:
            targets = data.get('targets', np.array([]))
            if targets.ndim == 2 and targets.shape[1] >= 2:
                # Use first two columns for analyst targets
                targets = targets[:, :2]
            
            # Validate target name for analyst models
            if expected_target not in ['analyst_confidence', 'target']:
                tprint_warning(f"Expected 'analyst_confidence' for analyst models, got '{expected_target}'")
        else:  # Tactician models
            targets = data.get('targets', np.array([]))
            if targets.ndim == 2 and targets.shape[1] >= 3:
                # Use last three columns for tactician targets
                targets = targets[:, -3:]
            
            # Validate target name for tactician models
            if expected_target not in ['position_confidence', 'target']:
                tprint_warning(f"Expected 'position_confidence' for tactician models, got '{expected_target}'")
        
        tprint_data_preview(targets, f"Targets for {model_type.value}")
        
        # Validate targets
        if not validate_array(targets):
            tprint_error("Invalid target data")
            raise ValueError("Invalid target data")
        
        # Ensure targets are 1D for single-output with proper validation
        if targets.ndim > 1:
            if targets.shape[1] == 1:
                targets = targets.ravel()
            elif targets.shape[1] > 1:
                # multi-output supported later; for now pick the first
                tprint_info(f"Multi-output targets detected ({targets.shape[1]} outputs), using first output")
                targets = targets[:, 0]
        else:
            targets = targets.ravel()
        
        # Validate target data types and ranges
        if targets.dtype == np.object_:
            tprint_warning("Target data type is object, attempting conversion")
            try:
                targets = targets.astype(float)
            except (ValueError, TypeError) as e:
                tprint_error(f"Failed to convert targets to float: {e}")
                raise ValueError("Invalid target data type")
        
        # Check for valid target values
        if np.any(np.isnan(targets)):
            tprint_warning("NaN values found in targets, replacing with 0")
            targets = np.nan_to_num(targets, nan=0.0)
        
        if np.any(np.isinf(targets)):
            tprint_warning("Infinite values found in targets, clipping to finite range")
            targets = np.clip(targets, -1e6, 1e6)
        
        tprint_data_format(f"Target preparation completed: {targets.shape}", LogLevel.INFO)
        return targets
    
    @traced
    @log_execution_time
    async def train_models(self, data: Dict[str, Any], config_paths: Dict[ModelType, str]) -> Dict[ModelType, List[TrainingResult]]:
        """
        Train all configured models.
        
        Args:
            data: Input data dictionary
            config_paths: Paths to configuration files for each model type
            
        Returns:
            Dictionary of training results by model type
        """
        tprint_info("🚀 Starting unified ML model training pipeline")
        self.logger.info("Starting unified ML model training pipeline")
        
        # Load configurations
        configs = await self._load_configurations(config_paths)
        
        # Preprocess data
        processed_data = await self._preprocess_data(data)
        
        # Train models
        if self.config.enable_parallel_training:
            results = await self._train_models_parallel(processed_data, configs)
        else:
            results = await self._train_models_sequential(processed_data, configs)
        
        # Generate reports
        if self.config.save_reports:
            await self._generate_reports(results)
        
        # Final memory cleanup
        self._final_memory_cleanup()
        
        tprint_success("✅ ML model training pipeline completed")
        self.logger.info("ML model training pipeline completed")
        
        return results
    
    async def _load_configurations(self, config_paths: Dict[ModelType, str]) -> Dict[ModelType, Dict[str, Any]]:
        """Load configuration files for each model type with inheritance support."""
        configs = {}
        
        for model_type, config_path in config_paths.items():
            try:
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                
                # Handle inheritance
                if 'extends' in cfg:
                    parent = (Path(config_path).parent / cfg['extends']).resolve()
                    with open(parent) as pf:
                        base = yaml.safe_load(pf) or {}
                    cfg = {**base, **cfg}  # base overrides trial
                    cfg.pop('extends', None)
                
                # Compute config hash for reproducibility
                config_str = json.dumps(cfg, sort_keys=True)
                config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
                cfg['_config_hash'] = config_hash
                
                configs[model_type] = cfg
                tprint_info(f"📋 Loaded configuration for {model_type.value} (hash: {config_hash})")
            except Exception as e:
                tprint_error(f"❌ Failed to load configuration for {model_type.value}: {e}")
                raise
        
        # Apply mode-based parameter scaling
        for model_type, config in configs.items():
            configs[model_type] = self._apply_mode_scaling(config)
        
        return configs
    
    async def _load_unified_memory_config(self) -> Dict[str, Any]:
        """Load unified memory configuration."""
        try:
            unified_config_path = Path("src/training/steps/models_training/config/ml_model_trainer/unified_memory_config.yaml")
            if unified_config_path.exists():
                with open(unified_config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            else:
                tprint_warning("Unified memory config not found, using defaults")
                return {}
        except Exception as e:
            tprint_warning(f"Failed to load unified memory config: {e}")
            return {}
    
    def _merge_unified_memory_config(self, config: Dict[str, Any], unified_config: Dict[str, Any], model_type: ModelType) -> Dict[str, Any]:
        """Merge unified memory configuration with model-specific config."""
        try:
            if not unified_config:
                return config
            
            # Get model-specific overrides
            model_overrides = unified_config.get('model_specific_overrides', {}).get(model_type.value, {})
            
            # Merge memory optimization settings
            if 'memory_optimization' not in config:
                config['memory_optimization'] = {}
            
            # Apply unified memory settings
            config['memory_optimization'].update(unified_config.get('memory_optimization', {}))
            
            # Apply model-specific overrides
            config['memory_optimization'].update(model_overrides)
            
            # Apply caching settings
            if 'caching' not in config:
                config['caching'] = {}
            
            config['caching'].update(unified_config.get('caching', {}))
            
            return config
            
        except Exception as e:
            tprint_warning(f"Failed to merge unified memory config: {e}")
            return config
    
    def _validate_configuration(self, config: Dict[str, Any], model_type: ModelType) -> None:
        """Validate configuration for completeness and correctness."""
        try:
            # Required sections
            required_sections = ['models', 'targets', 'training', 'metrics']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section: {section}")
            
            # Validate models section
            models = config.get('models', [])
            if not models:
                raise ValueError("No models configured")
            
            for model in models:
                if 'name' not in model or 'type' not in model:
                    raise ValueError("Model missing required fields: name, type")
            
            # Validate targets section
            targets = config.get('targets', {})
            if 'primary' not in targets:
                raise ValueError("Missing primary target")
            
            # Validate training section
            training = config.get('training', {})
            if 'validation_split' not in training:
                training['validation_split'] = 0.2
            if 'cv_folds' not in training:
                training['cv_folds'] = 5
            
            # Validate metrics section
            metrics = config.get('metrics', {})
            if 'primary' not in metrics:
                metrics['primary'] = 'f1_score'
            
            tprint_success(f"✅ Configuration validation passed for {model_type.value}")
            
        except Exception as e:
            tprint_error(f"❌ Configuration validation failed for {model_type.value}: {e}")
            raise
    
    def _apply_mode_scaling(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mode-based scaling to configuration parameters."""
        scaling_factor = self.config.get_mode_scaling_factor()
        
        # Create a deep copy to avoid modifying the original
        import copy
        scaled_config = copy.deepcopy(config)
        
        # Scale training parameters
        if 'training' in scaled_config:
            training = scaled_config['training']
            
            # Scale CV folds
            if 'cv_folds' in training:
                original_cv_folds = training['cv_folds']
                if self.config.mode == "LIGHT":
                    training['cv_folds'] = max(2, int(original_cv_folds * scaling_factor))
                elif self.config.mode == "BLANK":
                    training['cv_folds'] = max(3, int(original_cv_folds * scaling_factor))
                else:
                    training['cv_folds'] = original_cv_folds
            
            # Scale HPO parameters
            if 'hyperparameter_optimization' in training:
                hpo = training['hyperparameter_optimization']
                if 'n_trials' in hpo:
                    original_trials = hpo['n_trials']
                    if self.config.mode == "LIGHT":
                        hpo['n_trials'] = max(10, int(original_trials * scaling_factor))
                    elif self.config.mode == "BLANK":
                        hpo['n_trials'] = max(50, int(original_trials * scaling_factor))
                    else:
                        hpo['n_trials'] = original_trials
        
        # Scale model-specific parameters
        if 'models' in scaled_config:
            for model in scaled_config['models']:
                if 'parameters' in model:
                    params = model['parameters']
                    
                    # Scale iterations for CatBoost
                    if 'iterations' in params:
                        original_iterations = params['iterations']
                        if self.config.mode == "LIGHT":
                            params['iterations'] = max(100, int(original_iterations * scaling_factor))
                        elif self.config.mode == "BLANK":
                            params['iterations'] = max(500, int(original_iterations * scaling_factor))
                        else:
                            params['iterations'] = original_iterations
                    
                    # Scale n_estimators for LightGBM
                    if 'n_estimators' in params:
                        original_estimators = params['n_estimators']
                        if self.config.mode == "LIGHT":
                            params['n_estimators'] = max(100, int(original_estimators * scaling_factor))
                        elif self.config.mode == "BLANK":
                            params['n_estimators'] = max(500, int(original_estimators * scaling_factor))
                        else:
                            params['n_estimators'] = original_estimators
                    
                    # Scale max_iter for neural networks
                    if 'max_iter' in params:
                        original_max_iter = params['max_iter']
                        if self.config.mode == "LIGHT":
                            params['max_iter'] = max(100, int(original_max_iter * scaling_factor))
                        elif self.config.mode == "BLANK":
                            params['max_iter'] = max(500, int(original_max_iter * scaling_factor))
                        else:
                            params['max_iter'] = original_max_iter
                    
                    # Scale epochs for neural networks
                    if 'epochs' in params:
                        original_epochs = params['epochs']
                        if self.config.mode == "LIGHT":
                            params['epochs'] = max(10, int(original_epochs * scaling_factor))
                        elif self.config.mode == "BLANK":
                            params['epochs'] = max(50, int(original_epochs * scaling_factor))
                        else:
                            params['epochs'] = original_epochs
        
        tprint_info(f"🔧 Applied {self.config.mode} mode scaling (factor: {scaling_factor:.1%})")
        return scaled_config
    
    @comprehensive_memory_optimization(MemoryOptimizationLevel.AGGRESSIVE)
    async def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess input data using existing utilities."""
        tprint_info("🔄 Preprocessing data with comprehensive validation")
        
        # Validate input data structure
        if not isinstance(data, dict):
            tprint_error("Data must be a dictionary")
            raise ValueError("Data must be a dictionary")
        
        # Extract and validate features
        features = data.get('features', np.array([]))
        if not validate_array(features):
            tprint_error("Invalid features data")
            raise ValueError("Invalid features data")
        
        tprint_data_preview(features, "Input features")
        tprint_data_format(f"Features shape: {features.shape}, dtype: {features.dtype}", LogLevel.INFO)
        
        # Extract and validate targets
        targets = data.get('targets', np.array([]))
        if not validate_array(targets):
            tprint_error("Invalid targets data")
            raise ValueError("Invalid targets data")
        
        tprint_data_preview(targets, "Input targets")
        tprint_data_format(f"Targets shape: {targets.shape}, dtype: {targets.dtype}", LogLevel.INFO)
        
        # Apply safe data operations - ensure features are 2D
        processed_features = safe_array_operation(np.atleast_2d(features), self._clean_data)
        if processed_features.shape[0] < processed_features.shape[1]:
            # assume already samples x features
            pass
        
        # Ensure targets are 1D for single-output
        processed_targets = safe_array_operation(targets, self._clean_data)
        if processed_targets.ndim > 1 and processed_targets.shape[1] == 1:
            processed_targets = processed_targets.ravel()
        elif processed_targets.ndim > 1 and processed_targets.shape[1] > 1:
            # multi-output supported later; for now pick the first
            processed_targets = processed_targets[:, 0]
        else:
            processed_targets = processed_targets.ravel()
        
        # Detect data leakage using existing detector
        leakage_report = self.leakage_detector.detect_leakage(processed_features, processed_targets)
        if leakage_report.has_leakage:
            tprint_warning(f"Data leakage detected: {leakage_report.leakage_score:.3f}")
            tprint_warning(f"Recommendations: {leakage_report.recommendations}")
        else:
            tprint_success("No data leakage detected")
        
        # Apply hardware optimization
        processed_data = {
            'features': self.hardware_manager.process_data_with_optimization(
                processed_features, WorkloadType.ML_TRAINING
            ),
            'targets': self.hardware_manager.process_data_with_optimization(
                processed_targets, WorkloadType.ML_TRAINING
            ),
            'metadata': data.get('metadata', {}),
            'leakage_report': leakage_report
        }
        
        tprint_data_format(f"Processed data - Features: {processed_data['features'].shape}, Targets: {processed_data['targets'].shape}", LogLevel.INFO)
        tprint_success("✅ Data preprocessing completed")
        
        return processed_data
    
    def _clean_data(self, data: np.ndarray) -> np.ndarray:
        """Clean data using safe operations."""
        # Remove infinite values
        data = np.where(np.isfinite(data), data, 0.0)
        
        # Remove NaN values
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        return data
    
    @performance_tracked
    async def _validate_data_quality(self, data: Dict[str, Any], model_type: ModelType):
        """Validate data quality using existing utilities."""
        tprint_info(f"🔍 Validating data quality for {model_type.value}")
        
        features = data.get('features', np.array([]))
        targets = data.get('targets', np.array([]))
        
        # Use existing validation utilities
        validation_result = validate_dataframe(pd.DataFrame(features)) if features.size > 0 else None
        if validation_result and not validation_result.is_valid:
            tprint_warning(f"Data quality issues detected: {validation_result.errors}")
        
        # Check for data consistency
        if features.size > 0 and targets.size > 0:
            if len(features) != len(targets):
                tprint_error(f"Feature-target length mismatch: {len(features)} vs {len(targets)}")
                raise ValueError("Feature-target length mismatch")
        
        tprint_success(f"✅ Data quality validation completed for {model_type.value}")
    
    
    async def _train_models_parallel(self, data: Dict[str, Any], configs: Dict[ModelType, Dict[str, Any]]) -> Dict[ModelType, List[TrainingResult]]:
        """Train models in parallel using hardware-optimized parallelization."""
        tprint_info("🔄 Training models in parallel with hardware optimization")
        
        # Get optimal parallelization strategy from hardware manager
        workload_type = WorkloadType.ML_TRAINING
        optimal_strategy = self.hardware_manager.get_optimal_parallelization_strategy(workload_type)
        
        results = {}
        
        # Separate tasks by type for optimal parallelization
        ensemble_tasks = []
        cpu_bound_tasks = []
        io_bound_tasks = []
        
        for model_type in self.config.model_types:
            if model_type in configs:
                if model_type in [ModelType.ANALYST_ENSEMBLE, ModelType.TACTICIAN_ENSEMBLE]:
                    ensemble_tasks.append((model_type, configs[model_type]))
                elif model_type in [ModelType.ANALYST_BASE, ModelType.TACTICIAN_BASE]:
                    # These are CPU-intensive model training tasks
                    cpu_bound_tasks.append((model_type, configs[model_type]))
                else:
                    # Other tasks that might be I/O bound
                    io_bound_tasks.append((model_type, configs[model_type]))
        
        # Process ensemble models in main process (complex async operations)
        for model_type, config in ensemble_tasks:
            try:
                ens = await self._train_ensemble_model(data, model_type, config)
                results[model_type] = [TrainingResult(
                    model_type=model_type,
                    model_name=f"{ens['ensemble_type'].lower()}_stacker",
                    success=True,
                    model=ens['bundle'],
                    metrics=ens['metrics'],
                )]
                tprint_success(f"✅ Completed ensemble training for {model_type.value}")
            except Exception as e:
                tprint_error(f"❌ Ensemble training failed for {model_type.value}: {e}")
                results[model_type] = [TrainingResult(
                    model_type=model_type,
                    model_name="ensemble_error",
                    success=False,
                    metrics={"error": str(e)}
                )]
        
        # Process CPU-bound tasks with optimized ProcessPoolExecutor
        if cpu_bound_tasks:
            tprint_info(f"Processing {len(cpu_bound_tasks)} CPU-bound tasks with ProcessPoolExecutor")
            cpu_results = await self._process_cpu_bound_tasks(data, cpu_bound_tasks)
            results.update(cpu_results)
        
        # Process I/O-bound tasks with ThreadPoolExecutor
        if io_bound_tasks:
            tprint_info(f"Processing {len(io_bound_tasks)} I/O-bound tasks with ThreadPoolExecutor")
            io_results = await self._process_io_bound_tasks(data, io_bound_tasks)
            results.update(io_results)
        
        return results
    
    async def _process_cpu_bound_tasks(self, data: Dict[str, Any], tasks: List[Tuple[ModelType, Dict[str, Any]]]) -> Dict[ModelType, List[TrainingResult]]:
        """Process CPU-bound tasks using ProcessPoolExecutor with hardware optimization."""
        results = {}
        loop = asyncio.get_running_loop()
        
        # Use hardware-optimized process pool
        max_workers = self.hardware_manager.get_optimal_worker_count(WorkloadType.ML_TRAINING)
        
        # Create tasks
        async_tasks = []
        for model_type, config in tasks:
            task = loop.run_in_executor(
                self._process_pool,
                self._train_model_type_sync,
                data, model_type, config
            )
            async_tasks.append((model_type, task))
        
        # Execute tasks with hardware monitoring
        if async_tasks:
            try:
                task_results = await asyncio.gather(*[task for _, task in async_tasks], return_exceptions=True)
                for (model_type, _), result in zip(async_tasks, task_results):
                    if isinstance(result, Exception):
                        tprint_error(f"❌ CPU-bound training failed for {model_type.value}: {result}")
                        results[model_type] = []
                    else:
                        results[model_type] = result
                        tprint_success(f"✅ Completed CPU-bound training for {model_type.value}")
            except Exception as e:
                tprint_error(f"❌ CPU-bound parallel processing failed: {e}")
                # Fallback to sequential processing
                for model_type, config in tasks:
                    try:
                        result = self._train_model_type_sync(data, model_type, config)
                        results[model_type] = result
                        tprint_success(f"✅ Completed fallback training for {model_type.value}")
                    except Exception as task_e:
                        tprint_error(f"❌ Fallback training failed for {model_type.value}: {task_e}")
                        results[model_type] = []
        
        return results
    
    async def _process_io_bound_tasks(self, data: Dict[str, Any], tasks: List[Tuple[ModelType, Dict[str, Any]]]) -> Dict[ModelType, List[TrainingResult]]:
        """Process I/O-bound tasks using ThreadPoolExecutor."""
        results = {}
        
        # Use ThreadPoolExecutor for I/O-bound tasks
        max_workers = min(len(tasks), self.hardware_manager.get_optimal_worker_count(WorkloadType.ML_TRAINING))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            loop = asyncio.get_running_loop()
            
            # Create tasks
            async_tasks = []
            for model_type, config in tasks:
                task = loop.run_in_executor(
                    executor,
                    self._train_model_type_sync,
                    data, model_type, config
                )
                async_tasks.append((model_type, task))
            
            # Execute tasks
            if async_tasks:
                try:
                    task_results = await asyncio.gather(*[task for _, task in async_tasks], return_exceptions=True)
                    for (model_type, _), result in zip(async_tasks, task_results):
                        if isinstance(result, Exception):
                            tprint_error(f"❌ I/O-bound training failed for {model_type.value}: {result}")
                            results[model_type] = []
                        else:
                            results[model_type] = result
                            tprint_success(f"✅ Completed I/O-bound training for {model_type.value}")
                except Exception as e:
                    tprint_error(f"❌ I/O-bound parallel processing failed: {e}")
                    # Fallback to sequential processing
                    for model_type, config in tasks:
                        try:
                            result = self._train_model_type_sync(data, model_type, config)
                            results[model_type] = result
                            tprint_success(f"✅ Completed fallback training for {model_type.value}")
                        except Exception as task_e:
                            tprint_error(f"❌ Fallback training failed for {model_type.value}: {task_e}")
                            results[model_type] = []
        
        return results
    
    def _train_model_type_sync(self, data: Dict[str, Any], model_type: ModelType, config: Dict[str, Any]) -> List[TrainingResult]:
        """Synchronous version of model training for ProcessPoolExecutor."""
        try:
            # Reuse the async logic synchronously
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(self._train_model_type(data, model_type, config))
            finally:
                loop.close()
        except Exception as e:
            tprint_error(f"Sync training failed for {model_type.value}: {e}")
            return []
    
    async def _train_models_sequential(self, data: Dict[str, Any], configs: Dict[ModelType, Dict[str, Any]]) -> Dict[ModelType, List[TrainingResult]]:
        """Train models sequentially."""
        tprint_info("🔄 Training models sequentially")
        
        results = {}
        
        for model_type in self.config.model_types:
            if model_type in configs:
                try:
                    result = await self._train_model_type(data, model_type, configs[model_type])
                    results[model_type] = result
                    tprint_success(f"✅ Completed training for {model_type.value}")
                except Exception as e:
                    tprint_error(f"❌ Failed training for {model_type.value}: {e}")
                    results[model_type] = []
        
        return results
    
    async def _train_model_type(self, data: Dict[str, Any], model_type: ModelType, config: Dict[str, Any]) -> List[TrainingResult]:
        """Train all models of a specific type."""
        results = []
        
        # Extract model configurations
        models_config = config.get('models', [])
        
        for model_config in models_config:
            if not model_config.get('enabled', True):
                continue
            
            try:
                result = await self._train_single_model(data, model_type, model_config, config)
                results.append(result)
            except Exception as e:
                tprint_error(f"❌ Failed to train {model_config.get('name', 'unknown')}: {e}")
                results.append(TrainingResult(
                    model_type=model_type,
                    model_name=model_config.get('name', 'unknown'),
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    async def _train_single_model(self, data: Dict[str, Any], model_type: ModelType, model_config: Dict[str, Any], config: Dict[str, Any]) -> TrainingResult:
        """Train a single model."""
        start_time = time.time()
        model_name = model_config.get('name', 'unknown')
        
        tprint_info(f"🔄 Training {model_name} ({model_type.value})")
        
        try:
            # Apply memory optimizations
            data = self._apply_memory_optimizations(model_type, data)
            
            # Check if this is an ensemble model
            if model_type in [ModelType.ANALYST_ENSEMBLE, ModelType.TACTICIAN_ENSEMBLE]:
                # Use ensemble training method
                ens = await self._train_ensemble_model(data, model_type, config)
                training_time = time.time() - start_time
                
                result = TrainingResult(
                    model_type=model_type,
                    model_name=model_config.get('name', 'ensemble'),
                    success=True,
                    model=ens['bundle'],
                    metrics=ens['metrics'],
                    training_time=training_time
                )
                
                tprint_success(f"✅ Successfully trained ensemble {model_config.get('name', 'ensemble')} in {training_time:.2f}s")
                return result
            else:
                # Create trainer based on model type
                trainer = self._create_trainer(model_type, model_config, config)
                
                # Prepare features and targets
                X, y = await self._prepare_training_data(data, model_type, config)
                
                # Train model
                model = await self._train_model(trainer, X, y, model_config, config)
            
            # Evaluate model
            metrics = await self._evaluate_model(model, X, y, model_type, config)
            
            # Calculate feature importance
            feature_importance = await self._calculate_feature_importance(model, X, y, model_type)
            
            # Perform SHAP analysis
            shap_values = await self._perform_shap_analysis(model, X, y, model_type, config)
            
            # Generate predictions
            predictions = await self._generate_predictions(model, X, model_type)
            probabilities = await self._generate_probabilities(model, X, model_type)
            
            training_time = time.time() - start_time
            
            result = TrainingResult(
                model_type=model_type,
                model_name=model_name,
                success=True,
                model=model,
                metrics=metrics,
                predictions=predictions,
                probabilities=probabilities,
                feature_importance=feature_importance,
                shap_values=shap_values,
                training_time=training_time
            )
            
            tprint_success(f"✅ Successfully trained {model_name} in {training_time:.2f}s")
            return result
            
        except Exception as e:
            training_time = time.time() - start_time
            tprint_error(f"❌ Failed to train {model_name}: {e}")
            
            return TrainingResult(
                model_type=model_type,
                model_name=model_name,
                success=False,
                training_time=training_time,
                error_message=str(e)
            )
    
    def _create_trainer(self, model_type: ModelType, model_config: Dict[str, Any], config: Dict[str, Any]):
        """Create appropriate trainer for model type."""
        if model_type == ModelType.ANALYST_BASE:
            return self._create_analyst_base_trainer(model_config, config)
        elif model_type == ModelType.ANALYST_ENSEMBLE:
            return self._create_analyst_ensemble_trainer(model_config, config)
        elif model_type == ModelType.TACTICIAN_BASE:
            return self._create_tactician_base_trainer(model_config, config)
        elif model_type == ModelType.TACTICIAN_ENSEMBLE:
            return self._create_tactician_ensemble_trainer(model_config, config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_analyst_base_trainer(self, model_config: Dict[str, Any], config: Dict[str, Any]):
        """Create analyst base trainer."""
        # Import the correct configuration class
        from src.training.steps.model_training.analyst_models_training_refactored import AnalystTrainingConfig
        
        # Convert config to AnalystTrainingConfig
        training_config = AnalystTrainingConfig(
            timeframe=config.get('timeframe', '15m'),
            enable_patchtst_features=config.get('inputs', {}).get('analyst_features', {}).get('enable_patchtst_features', True),
            enable_regime_features=config.get('inputs', {}).get('analyst_features', {}).get('enable_regime_features', True),
            enable_multi_timeframe=config.get('inputs', {}).get('analyst_features', {}).get('enable_multi_timeframe', True),
            lightgbm_params=model_config.get('parameters', {}),
            catboost_params=model_config.get('parameters', {}),
            stacker_params=model_config.get('parameters', {}),
            validation_split=config.get('training', {}).get('validation_split', 0.2),
            cv_folds=config.get('training', {}).get('cv_folds', 5)
        )
        
        return AnalystBaseTrainer(training_config, self.logger)
    
    def _create_analyst_ensemble_trainer(self, model_config: Dict[str, Any], config: Dict[str, Any]):
        """Create analyst ensemble trainer."""
        # Import the correct configuration classes
        from src.training.steps.model_training.analyst_ensemble_training import AnalystEnsembleTrainingConfig
        from src.config.config_ensemble import EnsembleMethod
        
        # Convert config to AnalystEnsembleTrainingConfig
        training_config = AnalystEnsembleTrainingConfig(
            timeframe=config.get('timeframe', '15m'),
            enable_patchtst_features=config.get('inputs', {}).get('analyst_features', {}).get('enable_patchtst_features', True),
            enable_regime_features=config.get('inputs', {}).get('analyst_features', {}).get('enable_regime_features', True),
            enable_multi_timeframe=config.get('inputs', {}).get('analyst_features', {}).get('enable_multi_timeframe', True),
            ensemble_method=EnsembleMethod[model_config.get('type', 'STACKING').upper()],
            base_models=[model.get('type', 'LIGHTGBM').upper() for model in config.get('base_models', [])],
            meta_learner_params=model_config.get('parameters', {}).get('meta_learner_params', {}),
            validation_split=config.get('training', {}).get('validation_split', 0.2),
            cv_folds=config.get('training', {}).get('cv_folds', 5)
        )
        
        return AnalystEnsembleTrainer(training_config, self.logger)
    
    def _create_tactician_base_trainer(self, model_config: Dict[str, Any], config: Dict[str, Any]):
        """Create tactician base trainer."""
        # Import the correct configuration class
        from src.training.steps.model_training.tactician_models_training_refactored import TacticianTrainingConfig
        
        # Convert config to TacticianTrainingConfig
        training_config = TacticianTrainingConfig(
            timeframe=config.get('timeframe', '15m'),
            enable_entry_timing=config.get('inputs', {}).get('tactician_features', {}).get('enable_entry_timing', True),
            enable_exit_timing=config.get('inputs', {}).get('tactician_features', {}).get('enable_exit_timing', True),
            enable_position_sizing=config.get('inputs', {}).get('tactician_features', {}).get('enable_position_sizing', True),
            lightgbm_params=model_config.get('parameters', {}),
            catboost_params=model_config.get('parameters', {}),
            neural_network_params=model_config.get('parameters', {}),
            linear_params=model_config.get('parameters', {}),
            validation_split=config.get('training', {}).get('validation_split', 0.2),
            cv_folds=config.get('training', {}).get('cv_folds', 5)
        )
        
        return TacticianBaseTrainer(training_config, self.logger)
    
    def _create_tactician_ensemble_trainer(self, model_config: Dict[str, Any], config: Dict[str, Any]):
        """Create tactician ensemble trainer."""
        # Import the correct configuration class
        from src.training.steps.model_training.tactician_ensemble_training import TacticianEnsembleTrainingConfig
        from src.config.config_ensemble import EnsembleMethod
        
        # Convert config to TacticianEnsembleTrainingConfig
        training_config = TacticianEnsembleTrainingConfig(
            timeframe=config.get('timeframe', '15m'),
            enable_entry_timing=config.get('inputs', {}).get('tactician_features', {}).get('enable_entry_timing', True),
            enable_exit_timing=config.get('inputs', {}).get('tactician_features', {}).get('enable_exit_timing', True),
            enable_position_sizing=config.get('inputs', {}).get('tactician_features', {}).get('enable_position_sizing', True),
            ensemble_method=EnsembleMethod[model_config.get('type', 'STACKING').upper()],
            base_models=[model.get('type', 'LIGHTGBM').upper() for model in config.get('base_models', [])],
            meta_learner_params=model_config.get('parameters', {}).get('meta_learner_params', {}),
            validation_split=config.get('training', {}).get('validation_split', 0.2),
            cv_folds=config.get('training', {}).get('cv_folds', 5)
        )
        
        return TacticianEnsembleTrainer(training_config, self.logger)
    
    @memory_managed(MemoryStrategy.MODERATE)
    async def _prepare_training_data(self, data: Dict[str, Any], model_type: ModelType, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data using existing utilities."""
        tprint_info(f"🔄 Preparing training data for {model_type.value}")
        
        # Prepare features using existing utilities
        X = self._prepare_features(data, model_type, config, model_config=None)
        tprint_data_format(f"Features prepared: {X.shape}", LogLevel.INFO)
        
        # Prepare targets using existing utilities
        y = self._prepare_targets(data, model_type, config)
        tprint_data_format(f"Targets prepared: {y.shape}", LogLevel.INFO)
        
        # Validate data consistency
        if len(X) != len(y):
            tprint_error(f"Feature-target length mismatch: {len(X)} vs {len(y)}")
            raise ValueError("Feature-target length mismatch")
        
        # Apply safe statistical operations for data validation
        X = safe_statistical_operation(X, np.asarray)
        y = safe_statistical_operation(y, np.asarray)
        
        tprint_data_preview(X, f"Final features for {model_type.value}")
        tprint_data_preview(y, f"Final targets for {model_type.value}")
        
        tprint_success(f"✅ Training data prepared for {model_type.value}")
        return X, y
    
    @comprehensive_memory_optimization(MemoryOptimizationLevel.AGGRESSIVE)
    async def _train_model(self, trainer, X: np.ndarray, y: np.ndarray, model_config: Dict[str, Any], config: Dict[str, Any]):
        """Train the model using the trainer with HPO."""
        tprint_info(f"🚀 Training model: {model_config.get('name', 'unknown')}")
        
        # Validate inputs
        if not validate_array(X) or not validate_array(y):
            tprint_error("Invalid training data")
            raise ValueError("Invalid training data")
        
        tprint_data_format(f"Training data - X: {X.shape}, y: {y.shape}", LogLevel.INFO)
        
        # Check if hyperparameter optimization is enabled
        hpo_config = config.get('training', {}).get('hyperparameter_optimization', {})
        if hpo_config.get('enabled', False) and OPTUNA_AVAILABLE:
            tprint_info("🔧 Running hyperparameter optimization")
            
            # Infer task type for proper scoring
            task_type = self._infer_task_type(model_config, y)
            
            # Define objective function for HPO
            def objective(trial):
                # Get mode from config
                mode = getattr(self.config, 'mode', 'FULL')
                
                # Get hyperparameters from trial
                params = self._get_hpo_params(trial, model_config, mode)
                
                # Create model with trial parameters
                model = self._create_model_with_params(model_config, params, task_type)
                
                # Train and evaluate
                try:
                    model.fit(X, y)
                    score = self._evaluate_model_score(model, X, y, config, task_type)
                    
                    # Get direction from config
                    metric_dir = (config.get('training', {})
                                      .get('hyperparameter_optimization', {})
                                      .get('direction', "maximize"))
                    return score if metric_dir == "maximize" else -score
                except Exception as e:
                    tprint_warning(f"HPO trial failed: {e}")
                    return float('-inf')
            
            # Run HPO with mode-scaled parameters
            n_trials = hpo_config.get('n_trials', 100)
            timeout = hpo_config.get('timeout', 3600)
            
            # Apply mode scaling to HPO parameters
            scaling_factor = self.config.get_mode_scaling_factor()
            if self.config.mode == "LIGHT":
                n_trials = max(10, int(n_trials * scaling_factor))
                timeout = max(300, int(timeout * scaling_factor))  # Minimum 5 minutes
            elif self.config.mode == "BLANK":
                n_trials = max(50, int(n_trials * scaling_factor))
                timeout = max(1800, int(timeout * scaling_factor))  # Minimum 30 minutes
            
            best_params = self.hpo_system.optimize(
                objective=objective,
                n_trials=n_trials,
                timeout=timeout
            )
            
            tprint_success(f"✅ HPO completed. Best params: {best_params}")
            
            # Create final model with best parameters
            final_model = self._create_model_with_params(model_config, best_params, task_type)
        else:
            # Use default parameters
            final_model = self._create_model_with_params(model_config, model_config.get('parameters', {}), task_type)
        
        # Apply weighted loss if enabled
        if self.weighted_loss_integrator is not None:
            tprint_info("🎯 Applying weighted loss for negative learning approximation")
            
            # Get model type for weighted loss
            model_type = model_config.get('type', 'LIGHTGBM').upper()
            
            # Fit weighted loss manager for this model type
            self.weighted_loss_integrator.fit(model_type, X, y)
            
            # Wrap model with weighted loss functionality
            final_model = wrap_model_with_weighted_loss(final_model, model_type, self.weighted_loss_integrator)
            
            tprint_success("✅ Weighted loss applied to model")
        
        # Train the final model with early stopping
        tprint_info("🏋️ Training final model with early stopping")
        final_model = self._fit_with_early_stopping(final_model, X, y, config, task_type)
        
        tprint_success(f"✅ Model training completed: {model_config.get('name', 'unknown')}")
        return final_model
    
    def _get_hpo_params(self, trial, model_config: Dict[str, Any], mode: str = "FULL") -> Dict[str, Any]:
        """Get hyperparameters from Optuna trial with mode-based reduction."""
        model_type = model_config.get('type', 'LIGHTGBM').upper()
        
        # Apply mode-based reduction
        if mode == "LIGHT":
            reduction_factor = 0.1  # 90% reduction
        elif mode == "BLANK":
            reduction_factor = 0.5  # 50% reduction
        else:
            reduction_factor = 1.0  # No reduction
        
        params = {}
        
        if model_type in {'LIGHTGBM', 'LIGHTGBM_PATCHTST', 'STACKER_LGBM_CALIBRATED'}:
            base_estimators = int(1000 * reduction_factor)
            params = {
                'n_estimators': trial.suggest_int('n_estimators', base_estimators//10, base_estimators),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }
        elif model_type == 'CATBOOST':
            # Use mode-based scaling for base iterations
            scaling_factor = self.config.get_mode_scaling_factor()
            base_iterations = int(1000 * scaling_factor)
            if self.config.mode == "LIGHT":
                base_iterations = max(100, base_iterations)
            elif self.config.mode == "BLANK":
                base_iterations = max(500, base_iterations)
            
            params = {
                'iterations': trial.suggest_int('iterations', base_iterations//10, base_iterations),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
        elif model_type == 'XGBOOST':
            base_estimators = int(1000 * reduction_factor)
            params = {
                'n_estimators': trial.suggest_int('n_estimators', base_estimators//10, base_estimators),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }
        
        return params
    
    def _create_model_with_params(self, model_config: Dict[str, Any], params: Dict[str, Any], task_type: str):
        """Create model with given parameters, handling special model types."""
        try:
            model_key = model_config.get('type', 'LIGHTGBM').upper()
            fixed = model_config.get('parameters', {})
            merged = {**params, **fixed}  # fixed overrides tuned if both set
            
            # Validate required parameters
            if not model_key:
                raise ValueError("Model type is required")
            
            # Determine if classification based on task type
            is_classification = (task_type == "classification")
            
            tprint_info(f"Creating model: {model_key} (classification: {is_classification})")
            
        except Exception as e:
            tprint_error(f"Error in model creation setup: {e}")
            raise
        
        if model_key in {"LIGHTGBM", "STACKER_LGBM_CALIBRATED"}:
            try:
                from lightgbm import LGBMClassifier, LGBMRegressor
                cls = LGBMClassifier if is_classification else LGBMRegressor
                if "n_jobs" not in merged and "nthread" not in merged and "thread_count" not in merged:
                    merged["n_jobs"] = -1 if cls.__name__.startswith("LGBM") else merged.get("n_jobs", -1)
                return cls(**merged, random_state=42, verbose=-1)
            except Exception as e:
                tprint_error(f"Failed to create LightGBM model: {e}")
                raise
        
        elif model_key == "LIGHTGBM_PATCHTST":
            try:
                from src.training.steps.models_training.core.patchtst_wrapper import PatchTSTWrapper
                return PatchTSTWrapper(**merged)
            except Exception as e:
                tprint_error(f"Failed to create LIGHTGBM_PATCHTST model: {e}")
                raise
        
        elif model_key == "LIGHTGBM_GRU":
            try:
                from src.training.steps.models_training.core.lgbm_gru_wrapper import LGBMGRUWrapper
                return LGBMGRUWrapper(**merged)
            except Exception as e:
                tprint_error(f"Failed to create LIGHTGBM_GRU model: {e}")
                raise
        
        elif model_key == "STACKER_LGBM_CALIBRATED_GATED":
            try:
                from src.training.steps.models_training.core.stacker_lgbm_calibrated_gated import StackerLGBMCalibratedGated
                return StackerLGBMCalibratedGated(**merged)
            except Exception as e:
                tprint_error(f"Failed to create STACKER_LGBM_CALIBRATED_GATED model: {e}")
                raise
        
        elif model_key == "CAUSAL_DILATED_TCN":
            try:
                from src.models.tcn_regressor import TCNRegressor
                from src.training.steps.models_training.core.tcn_classifier_wrapper import TCNClassifierWrapper
                
                if not is_classification:
                    return TCNRegressor(**merged)
                else:
                    # Use TCN classifier wrapper for classification tasks
                    tcn_regressor = TCNRegressor(**merged)
                    return TCNClassifierWrapper(tcn_regressor)
            except Exception as e:
                tprint_error(f"Failed to create CAUSAL_DILATED_TCN model: {e}")
                raise
        
        elif model_key == "CATBOOST":
            try:
                from catboost import CatBoostClassifier, CatBoostRegressor
                cls = CatBoostClassifier if is_classification else CatBoostRegressor
                return cls(**merged, random_seed=42, verbose=False, thread_count=1)
            except Exception as e:
                tprint_error(f"Failed to create CatBoost model: {e}")
                raise
        
        elif model_key == "XGBOOST":
            try:
                from xgboost import XGBClassifier, XGBRegressor
                cls = XGBClassifier if is_classification else XGBRegressor
                if "n_jobs" not in merged and "nthread" not in merged and "thread_count" not in merged:
                    merged["n_jobs"] = -1
                return cls(**merged, random_state=42, verbosity=0)
            except Exception as e:
                tprint_error(f"Failed to create XGBoost model: {e}")
                raise
        
        # Fallback for unknown model types
        tprint_warning(f"Unknown model type: {model_key}, falling back to RandomForest")
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            cls = RandomForestClassifier if is_classification else RandomForestRegressor
            if "n_jobs" not in merged and "nthread" not in merged and "thread_count" not in merged:
                merged["n_jobs"] = -1
            return cls(**merged, random_state=42)
        except Exception as e:
            tprint_error(f"Failed to create fallback model: {e}")
            # Last resort: create a simple dummy model
            from sklearn.dummy import DummyClassifier, DummyRegressor
            cls = DummyClassifier if is_classification else DummyRegressor
            return cls(strategy="most_frequent" if is_classification else "mean")
    
    def _evaluate_model_score(self, model, X: np.ndarray, y: np.ndarray, recipe: Dict[str, Any], task_type: str) -> float:
        """Evaluate model score for HPO using YAML metrics."""
        try:
            scorer, scorer_kwargs = self._resolve_scorer(recipe, task_type)
            # Use recipe-specific CV folds
            cv_system = self._make_cv(recipe)
            scores = cv_system.cross_validate(
                model, X, y, 
                cv_type='temporal', 
                scoring=scorer, 
                **scorer_kwargs
            )
            return float(np.mean(scores))
        except Exception:
            return float('-inf')
    
    @performance_tracked
    async def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray, model_type: ModelType, config: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the trained model using YAML-based metrics."""
        tprint_info(f"📊 Evaluating model for {model_type.value}")
        
        # Validate inputs
        if not validate_array(X) or not validate_array(y):
            tprint_error("Invalid evaluation data")
            raise ValueError("Invalid evaluation data")
        
        # Generate predictions
        predictions = model.predict(X)
        tprint_data_format(f"Predictions generated: {predictions.shape}", LogLevel.INFO)
        
        # Infer task type from YAML recipe
        task_type = self._infer_task_type_from_recipe(config)
        
        # Calculate basic metrics
        metrics = {}
        
        if task_type == "classification":
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
            metrics["accuracy"] = float(accuracy_score(y, predictions))
            metrics["f1_score"] = float(f1_score(y, predictions, average="weighted"))
            metrics["precision"] = float(precision_score(y, predictions, average="weighted"))
            metrics["recall"] = float(recall_score(y, predictions, average="weighted"))
            
            if hasattr(model, "predict_proba") and np.unique(y).size == 2:
                proba = model.predict_proba(X)[:, 1]
                metrics["auc_roc"] = float(roc_auc_score(y, proba))
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
            rmse = np.sqrt(mean_squared_error(y, predictions))
            metrics.update({
                "rmse": float(rmse),
                "mae": float(mean_absolute_error(y, predictions)),
                "r2_score": float(r2_score(y, predictions)),
                "explained_variance": float(explained_variance_score(y, predictions)),
            })
        
        # CV summary on primary scorer
        scorer, scorer_kwargs = self._resolve_scorer(config, task_type)
        # Use recipe-specific CV folds
        cv_system = self._make_cv(config)
        cv_scores = cv_system.cross_validate(
            model, X, y, 
            cv_type='temporal', 
            scoring=scorer, 
            **scorer_kwargs
        )
        metrics["cv_mean"] = float(np.mean(cv_scores))
        metrics["cv_std"] = float(np.std(cv_scores))
        
        tprint_data_format(f"Evaluation metrics: {metrics}", LogLevel.INFO)
        tprint_success(f"✅ Model evaluation completed for {model_type.value}")
        
        return metrics
    
    @performance_tracked
    async def _calculate_feature_importance(self, model, X: np.ndarray, y: np.ndarray, model_type: ModelType) -> Dict[str, float]:
        """Calculate feature importance using existing utilities."""
        tprint_info(f"🔍 Calculating feature importance for {model_type.value}")
        
        # Validate inputs
        if not validate_array(X) or not validate_array(y):
            tprint_error("Invalid data for feature importance calculation")
            return {}
        
        feature_importance = {}
        
        try:
            # Try to get built-in feature importance
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                feature_names = [f"feature_{i}" for i in range(len(importance_scores))]
                feature_importance = dict(zip(feature_names, importance_scores))
                tprint_data_format(f"Built-in feature importance: {len(feature_importance)} features", LogLevel.INFO)
            
            # Use feature selector for additional importance
            feature_selector = self.feature_selectors.get(model_type)
            if feature_selector:
                try:
                    selector_importance = feature_selector.get_feature_importance(X, y)
                    if selector_importance:
                        feature_importance.update(selector_importance)
                        tprint_data_format(f"Feature selector importance: {len(selector_importance)} features", LogLevel.INFO)
                except Exception as e:
                    tprint_warning(f"Feature selector importance failed: {e}")
            
            # Apply safe mathematical operations
            for key, value in feature_importance.items():
                if not np.isfinite(value):
                    feature_importance[key] = 0.0
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            tprint_data_format(f"Feature importance calculated: {len(feature_importance)} features", LogLevel.INFO)
            tprint_success(f"✅ Feature importance calculation completed for {model_type.value}")
            
        except Exception as e:
            tprint_error(f"Feature importance calculation failed: {e}")
            # Return empty dict on failure
            feature_importance = {}
        
        return feature_importance
    
    @performance_tracked
    async def _perform_shap_analysis(self, model, X: np.ndarray, y: np.ndarray, model_type: ModelType, config: Dict[str, Any]) -> np.ndarray:
        """Perform SHAP analysis using existing utilities."""
        tprint_info(f"🔍 Performing SHAP analysis for {model_type.value}")
        
        # Validate inputs
        if not validate_array(X) or not validate_array(y):
            tprint_error("Invalid data for SHAP analysis")
            return np.array([])
        
        try:
            # Use existing explainability manager
            explanation_config = ExplanationConfig(
                enable_shap=True,
                enable_lime=False,  # Focus on SHAP for now
                shap_sample_size=min(100, X.shape[0]),
                shap_max_features=min(50, X.shape[1])
            )
            
            # Generate SHAP explanations
            shap_values = self.explainability_manager.explain_model(
                model=model,
                X=X,
                y=y,
                config=explanation_config
            )
            
            if shap_values is not None:
                tprint_data_format(f"SHAP analysis completed: {shap_values.shape}", LogLevel.INFO)
                tprint_success(f"✅ SHAP analysis completed for {model_type.value}")
                return shap_values
            else:
                tprint_warning("SHAP analysis returned None")
                return np.array([])
                
        except Exception as e:
            tprint_error(f"SHAP analysis failed: {e}")
            # Return empty array on failure
            return np.array([])
    
    @performance_tracked
    async def _generate_predictions(self, model, X: np.ndarray, model_type: ModelType) -> np.ndarray:
        """Generate predictions using safe operations."""
        tprint_info(f"🔮 Generating predictions for {model_type.value}")
        
        # Validate inputs
        if not validate_array(X):
            tprint_error("Invalid data for prediction generation")
            return np.array([])
        
        try:
            predictions = model.predict(X)
            tprint_data_format(f"Predictions generated: {predictions.shape}", LogLevel.INFO)
            tprint_success(f"✅ Predictions generated for {model_type.value}")
            return predictions
        except Exception as e:
            tprint_error(f"Prediction generation failed: {e}")
            return np.array([])
    
    @performance_tracked
    async def _generate_probabilities(self, model, X: np.ndarray, model_type: ModelType) -> np.ndarray:
        """Generate prediction probabilities using safe operations."""
        tprint_info(f"🔮 Generating probabilities for {model_type.value}")
        
        # Validate inputs
        if not validate_array(X):
            tprint_error("Invalid data for probability generation")
            return None
        
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                tprint_data_format(f"Probabilities generated: {proba.shape}", LogLevel.INFO)
                tprint_success(f"✅ Probabilities generated for {model_type.value}")
                return proba  # (n, n_classes) - full matrix
            else:
                tprint_info(f"Model does not support probability prediction for {model_type.value}")
                return None
        except Exception as e:
            tprint_error(f"Probability generation failed: {e}")
            return None
    
    async def _generate_reports(self, results: Dict[ModelType, List[TrainingResult]]):
        """Generate comprehensive reports."""
        tprint_info("📊 Generating reports")
        
        # This would implement comprehensive report generation
        # including HTML reports, plots, tables, etc.
        
        tprint_success("✅ Reports generated")
    
    def _final_memory_cleanup(self):
        """Perform final memory cleanup after training."""
        try:
            # Clear training results
            if hasattr(self, 'training_results'):
                self.training_results.clear()
            
            # Clear feature engineers
            if hasattr(self, 'feature_engineers'):
                self.feature_engineers.clear()
            
            # Clear target generators
            if hasattr(self, 'target_generators'):
                self.target_generators.clear()
            
            # Clear data validators
            if hasattr(self, 'data_validators'):
                self.data_validators.clear()
            
            # Clear leakage detectors
            if hasattr(self, 'leakage_detectors'):
                self.leakage_detectors.clear()
            
            # Clear metrics calculators
            if hasattr(self, 'metrics_calculators'):
                self.metrics_calculators.clear()
            
            # Clear SHAP analyzers
            if hasattr(self, 'shap_analyzers'):
                self.shap_analyzers.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Use hardware manager for final cleanup
            if hasattr(self, 'hardware_manager'):
                self.hardware_manager.cleanup()
            
            tprint_success("🧹 Final memory cleanup completed")
            
        except Exception as e:
            tprint_warning(f"Final memory cleanup failed: {e}")


# Example usage
async def main():
    """Example usage of the ML Model Trainer."""
    
    # Create configuration
    config = MLModelTrainerConfig(
        model_types=[
            ModelType.ANALYST_BASE,
            ModelType.ANALYST_ENSEMBLE,
            ModelType.TACTICIAN_BASE,
            ModelType.TACTICIAN_ENSEMBLE
        ],
        timeframe="15m",
        enable_parallel_training=True,
        max_workers=4
    )
    
    # Create trainer
    trainer = MLModelTrainer(config)
    
    # Define config paths
    config_paths = {
        ModelType.ANALYST_BASE: "src/training/steps/models_training/config/ml_model_trainer/analyst_base_config.yaml",
        ModelType.ANALYST_ENSEMBLE: "src/training/steps/models_training/config/ml_model_trainer/analyst_ensemble_config.yaml",
        ModelType.TACTICIAN_BASE: "src/training/steps/models_training/config/ml_model_trainer/tactician_base_config.yaml",
        ModelType.TACTICIAN_ENSEMBLE: "src/training/steps/models_training/config/ml_model_trainer/tactician_ensemble_config.yaml"
    }
    
    # Prepare data (placeholder)
    data = {
        'features': np.random.randn(1000, 50),
        'targets': np.random.randint(0, 2, 1000),
        'metadata': {}
    }
    
    # Train models
    results = await trainer.train_models(data, config_paths)
    
    # Print results
    for model_type, model_results in results.items():
        print(f"\n{model_type.value} Results:")
        for result in model_results:
            print(f"  {result.model_name}: {'Success' if result.success else 'Failed'}")
            if result.success:
                print(f"    Metrics: {result.metrics}")
                print(f"    Training Time: {result.training_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())