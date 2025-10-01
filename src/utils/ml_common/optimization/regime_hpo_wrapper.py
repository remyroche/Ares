"""
Regime-Specific Hyperparameter Optimization Wrapper

This module provides HPO integration for regime base training and meta-model training
configurations with enhanced meta-features for improved regime detection and prediction.

Key Features:
- Regime-specific search spaces for CatBoost, ExtraTrees, LightGBM, Bayesian Rules
- Integration with existing HPO infrastructure
- Support for OOF validation and time-series CV
- Meta-feature optimization
- Hierarchical optimization (base → meta models)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
import yaml
from pathlib import Path

# Import existing HPO infrastructure
from .hpo_utils import HyperparameterOptimization, optimize_hyperparameters
from .hierarchical_hpo import HierarchicalHPO, HPOPhaseConfig, HPOPhaseResult
from ..validation.unified_cv import perform_cross_validation
from ..ensembles.oof_stacking_ensemble_manager import OOFStackingEnsembleManager

# Import regime configurations
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from config.regime_base_training_config import load_regime_base_config
from config.regime_metamodel_training_config import load_regime_metamodel_config

logger = logging.getLogger(__name__)

@dataclass
class RegimeHPOConfig:
    """Configuration for regime-specific HPO."""
    
    # Base model HPO settings
    base_model_n_trials: int = 100
    base_model_timeout: Optional[int] = 3600  # 1 hour
    base_model_cv_folds: int = 5
    
    # Meta model HPO settings
    meta_model_n_trials: int = 50
    meta_model_timeout: Optional[int] = 1800  # 30 minutes
    meta_model_cv_folds: int = 3
    
    # Optimization strategy
    optimization_strategy: str = 'hierarchical'  # 'hierarchical', 'staged', 'bayesian'
    enable_early_stopping: bool = True
    enable_pruning: bool = True
    
    # Meta-feature optimization
    enable_meta_feature_optimization: bool = True
    meta_feature_n_trials: int = 30
    
    # Parallel processing
    enable_parallel: bool = True
    max_workers: int = 4
    
    # Validation settings
    enable_oof_validation: bool = True
    enable_time_series_cv: bool = True
    validation_split: float = 0.2
    
    # Scoring metrics
    base_model_scoring: str = 'balanced_accuracy'
    meta_model_scoring: str = 'f1_macro'
    meta_feature_scoring: str = 'regime_stability'

@dataclass
class RegimeHPOResult:
    """Result of regime-specific HPO."""
    
    # Base model results
    base_model_results: Dict[str, Dict[str, Any]]
    base_model_best_params: Dict[str, Dict[str, Any]]
    base_model_best_scores: Dict[str, float]
    
    # Meta model results
    meta_model_results: Dict[str, Any]
    meta_model_best_params: Dict[str, Any]
    meta_model_best_score: float
    
    # Meta-feature results
    meta_feature_results: Optional[Dict[str, Any]] = None
    meta_feature_best_params: Optional[Dict[str, Any]] = None
    
    # Optimization metadata
    total_optimization_time: float
    optimization_strategy: str
    n_total_trials: int
    convergence_info: Dict[str, Any]

class RegimeHPOWrapper:
    """
    Wrapper class for regime-specific hyperparameter optimization.
    
    Integrates the new regime training configurations with the existing
    HPO infrastructure for optimal regime detection and prediction.
    """
    
    def __init__(self, 
                 regime_base_config_path: str = "src/config/regime_base_training_config.yaml",
                 regime_metamodel_config_path: str = "src/config/regime_metamodel_training_config.yaml",
                 hpo_config: Optional[RegimeHPOConfig] = None):
        """
        Initialize the RegimeHPOWrapper.
        
        Args:
            regime_base_config_path: Path to regime base training config
            regime_metamodel_config_path: Path to regime meta-model training config
            hpo_config: HPO configuration
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Initializing RegimeHPOWrapper...")
        
        # Load configurations
        self.regime_base_config = self._load_regime_base_config(regime_base_config_path)
        self.regime_metamodel_config = self._load_regime_metamodel_config(regime_metamodel_config_path)
        self.hpo_config = hpo_config or RegimeHPOConfig()
        
        # Initialize HPO infrastructure
        self.hpo_utils = HyperparameterOptimization(
            config={
                'enable_parallel': self.hpo_config.enable_parallel,
                'max_workers': self.hpo_config.max_workers,
                'enable_monitoring': True,
                'use_nonlinear_optimization': True
            }
        )
        
        # Initialize hierarchical HPO
        self.hierarchical_hpo = HierarchicalHPO()
        
        # Initialize OOF ensemble manager
        self.oof_manager = OOFStackingEnsembleManager()
        
        # Create regime-specific search spaces
        self.regime_search_spaces = self._create_regime_search_spaces()
        
        self.logger.info("✅ RegimeHPOWrapper initialized successfully")
    
    def _load_regime_base_config(self, config_path: str) -> Dict[str, Any]:
        """Load regime base training configuration."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"📊 Loaded regime base config from {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load regime base config: {e}")
            return {}
    
    def _load_regime_metamodel_config(self, config_path: str) -> Dict[str, Any]:
        """Load regime meta-model training configuration."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"📊 Loaded regime meta-model config from {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load regime meta-model config: {e}")
            return {}
    
    def _create_regime_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Create regime-specific search spaces from configurations."""
        search_spaces = {}
        
        # CatBoost search space
        if 'catboost' in self.regime_base_config:
            catboost_config = self.regime_base_config['catboost']
            # Resolve bootstrap type choices
            bootstrap_type_config = catboost_config.get('bootstrap_type', ['Bayesian', 'Bernoulli'])
            if bootstrap_type_config is None:
                bootstrap_type_choices = ['Bayesian', 'Bernoulli']
            elif isinstance(bootstrap_type_config, (str, bytes)):
                bootstrap_type_choices = [bootstrap_type_config]
            elif isinstance(bootstrap_type_config, (list, tuple, set)):
                bootstrap_type_choices = list(bootstrap_type_config)
            else:
                bootstrap_type_choices = [bootstrap_type_config]

            # Resolve subsample range
            subsample_config = catboost_config.get('subsample')
            if isinstance(subsample_config, dict):
                subsample_low = subsample_config.get('low', 0.5)
                subsample_high = subsample_config.get('high', 0.9)
            elif isinstance(subsample_config, (list, tuple, set)):
                subsample_low = min(subsample_config)
                subsample_high = max(subsample_config)
            elif subsample_config is not None:
                subsample_low = subsample_high = float(subsample_config)
            else:
                subsample_low = 0.5
                subsample_high = 0.9

            # Resolve column sample by level range
            colsample_config = catboost_config.get('colsample_bylevel')
            if isinstance(colsample_config, dict):
                colsample_low = colsample_config.get('low', 0.5)
                colsample_high = colsample_config.get('high', 0.9)
            elif isinstance(colsample_config, (list, tuple, set)):
                colsample_low = min(colsample_config)
                colsample_high = max(colsample_config)
            elif colsample_config is not None:
                colsample_low = colsample_high = float(colsample_config)
            else:
                colsample_low = 0.5
                colsample_high = 0.9

            search_spaces['catboost'] = {
                'depth': {
                    'type': 'int',
                    'low': min(catboost_config.get('depth', [4, 5, 6])),
                    'high': max(catboost_config.get('depth', [4, 5, 6]))
                },
                'learning_rate': {
                    'type': 'float',
                    'low': min(catboost_config.get('learning_rate', [0.03, 0.04, 0.05, 0.06])),
                    'high': max(catboost_config.get('learning_rate', [0.03, 0.04, 0.05, 0.06]))
                },
                'l2_leaf_reg': {
                    'type': 'float',
                    'low': min(catboost_config.get('l2_leaf_reg', [6, 8, 10, 12])),
                    'high': max(catboost_config.get('l2_leaf_reg', [6, 8, 10, 12]))
                },
                'iterations': {
                    'type': 'int',
                    'low': min(catboost_config.get('iterations', [500, 800, 1200])),
                    'high': max(catboost_config.get('iterations', [500, 800, 1200]))
                },
                'bootstrap_type': {
                    'type': 'categorical',
                    'choices': bootstrap_type_choices or ['Bayesian', 'Bernoulli']
                },
                'subsample': {
                    'type': 'float',
                    'low': subsample_low,
                    'high': subsample_high
                },
                'colsample_bylevel': {
                    'type': 'float',
                    'low': colsample_low,
                    'high': colsample_high
                }
            }
        
        # ExtraTrees search space
        if 'extratrees' in self.regime_base_config:
            extratrees_config = self.regime_base_config['extratrees']
            search_spaces['extratrees'] = {
                'n_estimators': {
                    'type': 'int',
                    'low': min(extratrees_config.get('n_estimators', [300, 500, 800])),
                    'high': max(extratrees_config.get('n_estimators', [300, 500, 800]))
                },
                'max_depth': {
                    'type': 'categorical',
                    'choices': extratrees_config.get('max_depth', [None, 10, 15])
                },
                'min_samples_split': {
                    'type': 'int',
                    'low': min(extratrees_config.get('min_samples_split', [5, 10, 20])),
                    'high': max(extratrees_config.get('min_samples_split', [5, 10, 20]))
                },
                'min_samples_leaf': {
                    'type': 'int',
                    'low': min(extratrees_config.get('min_samples_leaf', [2, 5, 10])),
                    'high': max(extratrees_config.get('min_samples_leaf', [2, 5, 10]))
                },
                'max_features': {
                    'type': 'categorical',
                    'choices': extratrees_config.get('max_features', ['sqrt', 0.3, 0.5])
                }
            }
        
        # LightGBM Meta search space
        if 'lightgbm_meta' in self.regime_metamodel_config:
            lightgbm_config = self.regime_metamodel_config['lightgbm_meta']
            search_spaces['lightgbm_meta'] = {
                'num_leaves': {
                    'type': 'int',
                    'low': min(lightgbm_config.get('num_leaves', [15, 23, 31])),
                    'high': max(lightgbm_config.get('num_leaves', [15, 23, 31]))
                },
                'max_depth': {
                    'type': 'int',
                    'low': min(lightgbm_config.get('max_depth', [3, 4, 5])),
                    'high': max(lightgbm_config.get('max_depth', [3, 4, 5]))
                },
                'learning_rate': {
                    'type': 'float',
                    'low': min(lightgbm_config.get('learning_rate', [0.03, 0.04, 0.05])),
                    'high': max(lightgbm_config.get('learning_rate', [0.03, 0.04, 0.05]))
                },
                'min_data_in_leaf': {
                    'type': 'int',
                    'low': min(lightgbm_config.get('min_data_in_leaf', [50, 100, 150])),
                    'high': max(lightgbm_config.get('min_data_in_leaf', [50, 100, 150]))
                },
                'feature_fraction': {
                    'type': 'float',
                    'low': min(lightgbm_config.get('feature_fraction', [0.6, 0.75, 0.9])),
                    'high': max(lightgbm_config.get('feature_fraction', [0.6, 0.75, 0.9]))
                },
                'lambda_l1': {
                    'type': 'float',
                    'low': min(lightgbm_config.get('lambda_l1', [0, 1e-2, 1e-1])),
                    'high': max(lightgbm_config.get('lambda_l1', [0, 1e-2, 1e-1]))
                },
                'lambda_l2': {
                    'type': 'float',
                    'low': min(lightgbm_config.get('lambda_l2', [0, 1e-2, 1e-1])),
                    'high': max(lightgbm_config.get('lambda_l2', [0, 1e-2, 1e-1]))
                },
                'n_estimators': {
                    'type': 'int',
                    'low': min(lightgbm_config.get('n_estimators', [200, 400, 600])),
                    'high': max(lightgbm_config.get('n_estimators', [200, 400, 600]))
                }
            }
        
        # Bayesian Rule Lists search space
        if 'bayesian_rule_lists' in self.regime_base_config:
            brl_config = self.regime_base_config['bayesian_rule_lists']
            search_spaces['bayesian_rule_lists'] = {
                'listlengthprior': {
                    'type': 'int',
                    'low': 2,
                    'high': 5
                },
                'maxcardinality': {
                    'type': 'int',
                    'low': 2,
                    'high': 3
                },
                'minsupport': {
                    'type': 'float',
                    'low': 0.02,
                    'high': 0.05
                },
                'alpha': {
                    'type': 'float',
                    'low': 0.5,
                    'high': 2.0
                },
                'beta': {
                    'type': 'float',
                    'low': 0.5,
                    'high': 2.0
                },
                'list_length_lambda': {
                    'type': 'int',
                    'low': 3,
                    'high': 5
                },
                'rule_length_penalty': {
                    'type': 'float',
                    'low': 0.8,
                    'high': 1.2
                },
                'n_chains': {
                    'type': 'int',
                    'low': 2,
                    'high': 3
                },
                'n_iter': {
                    'type': 'int',
                    'low': 6000,
                    'high': 14000
                },
                'burn_in': {
                    'type': 'int',
                    'low': 1000,
                    'high': 2000
                },
                'thin': {
                    'type': 'int',
                    'low': 1,
                    'high': 5
                },
                'max_candidates': {
                    'type': 'int',
                    'low': 1000,
                    'high': 4000
                }
            }
        
        self.logger.info(f"📊 Created search spaces for {list(search_spaces.keys())}")
        return search_spaces
    
    def optimize_regime_base_models(self, 
                                  X: np.ndarray, 
                                  y: np.ndarray,
                                  regime_type: str = 'all') -> Dict[str, Any]:
        """
        Optimize hyperparameters for regime base models.
        
        Args:
            X: Feature matrix
            y: Target array (regime labels)
            regime_type: Type of regime ('all', 'trend', 'volatility', etc.)
            
        Returns:
            Optimization results for base models
        """
        self.logger.info(f"🎯 Starting regime base model optimization for {regime_type}")
        start_time = time.time()
        
        results = {}
        
        # Optimize each base model type
        for model_type, search_space in self.regime_search_spaces.items():
            if model_type == 'lightgbm_meta':  # Skip meta models in base optimization
                continue
                
            self.logger.info(f"🔧 Optimizing {model_type}...")
            
            try:
                # Create model factory
                model_factory = self._create_model_factory(model_type)
                
                # Determine CV strategy based on configuration
                cv_strategy = None
                if self.hpo_config.enable_time_series_cv:
                    try:
                        from sklearn.model_selection import TimeSeriesSplit
                        # Use TimeSeriesSplit for temporal data to prevent data leakage
                        cv_strategy = TimeSeriesSplit(n_splits=self.hpo_config.base_model_cv_folds)
                        self.logger.info(f"📅 Using TimeSeriesSplit with {self.hpo_config.base_model_cv_folds} folds for {model_type}")
                    except ImportError:
                        self.logger.warning("⚠️ TimeSeriesSplit not available, falling back to StratifiedKFold WITHOUT shuffle to prevent data leakage")
                        from sklearn.model_selection import StratifiedKFold
                        # CRITICAL: Do NOT shuffle time series data - causes severe data leakage!
                        cv_strategy = StratifiedKFold(n_splits=self.hpo_config.base_model_cv_folds, shuffle=False)

                # Perform optimization
                if self.hpo_config.optimization_strategy == 'staged':
                    optimization_result = self.hpo_utils.staged_hpo(
                        model_factory=model_factory,
                        X=X,
                        y=y,
                        search_space=search_space,
                        n_trials=self.hpo_config.base_model_n_trials,
                        scoring=self.hpo_config.base_model_scoring,
                        cv=cv_strategy,
                        timeout=self.hpo_config.base_model_timeout
                    )
                elif self.hpo_config.optimization_strategy == 'bayesian':
                    optimization_result = self.hpo_utils.bayesian_optimization(
                        model_factory=model_factory,
                        X=X,
                        y=y,
                        search_space=search_space,
                        n_trials=self.hpo_config.base_model_n_trials,
                        scoring=self.hpo_config.base_model_scoring,
                        cv=cv_strategy,
                        timeout=self.hpo_config.base_model_timeout,
                        optimization_context=f"Base Model Optimization - {model_name} hyperparameter tuning for {regime} market regime using ensemble learning with cross-validation and regime-specific feature engineering",
                        study_name=f"base_model_{model_name}_{regime}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                else:
                    # Default to staged HPO
                    optimization_result = self.hpo_utils.staged_hpo(
                        model_factory=model_factory,
                        X=X,
                        y=y,
                        search_space=search_space,
                        n_trials=self.hpo_config.base_model_n_trials,
                        scoring=self.hpo_config.base_model_scoring,
                        cv=cv_strategy
                    )
                
                results[model_type] = optimization_result
                self.logger.info(f"✅ {model_type} optimization completed - Best score: {optimization_result.get('best_score', 0):.4f}")
                
            except Exception as e:
                self.logger.error(f"❌ {model_type} optimization failed: {e}")
                results[model_type] = {'error': str(e)}
        
        optimization_time = time.time() - start_time
        self.logger.info(f"🏆 Base model optimization completed in {optimization_time:.2f}s")
        
        return {
            'results': results,
            'optimization_time': optimization_time,
            'regime_type': regime_type,
            'strategy': self.hpo_config.optimization_strategy
        }
    
    def optimize_regime_meta_model(self, 
                                  X: np.ndarray, 
                                  y: np.ndarray,
                                  base_model_predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters for regime meta-model.
        
        Args:
            X: Feature matrix
            y: Target array (regime labels)
            base_model_predictions: Predictions from base models (optional)
            
        Returns:
            Optimization results for meta-model
        """
        self.logger.info("🎯 Starting regime meta-model optimization")
        start_time = time.time()
        
        # Use LightGBM meta search space
        if 'lightgbm_meta' not in self.regime_search_spaces:
            self.logger.error("❌ LightGBM meta search space not found")
            return {'error': 'LightGBM meta search space not found'}
        
        search_space = self.regime_search_spaces['lightgbm_meta']
        
        try:
            # Create LightGBM meta model factory
            model_factory = self._create_lightgbm_meta_factory()

            # Determine CV strategy based on configuration
            cv_strategy = None
            if self.hpo_config.enable_time_series_cv:
                try:
                    from sklearn.model_selection import TimeSeriesSplit
                    # Use TimeSeriesSplit for temporal data to prevent data leakage
                    cv_strategy = TimeSeriesSplit(n_splits=self.hpo_config.meta_model_cv_folds)
                    self.logger.info(f"📅 Using TimeSeriesSplit with {self.hpo_config.meta_model_cv_folds} folds for meta-model")
                except ImportError:
                    self.logger.warning("⚠️ TimeSeriesSplit not available, falling back to StratifiedKFold WITHOUT shuffle to prevent data leakage")
                    from sklearn.model_selection import StratifiedKFold
                    # CRITICAL: Do NOT shuffle time series data - causes severe data leakage!
                    cv_strategy = StratifiedKFold(n_splits=self.hpo_config.meta_model_cv_folds, shuffle=False)

            # Perform optimization
            if self.hpo_config.optimization_strategy == 'staged':
                optimization_result = self.hpo_utils.staged_hpo(
                    model_factory=model_factory,
                    X=X,
                    y=y,
                    search_space=search_space,
                    n_trials=self.hpo_config.meta_model_n_trials,
                    scoring=self.hpo_config.meta_model_scoring,
                    cv=cv_strategy,
                    timeout=self.hpo_config.meta_model_timeout
                )
            else:
                optimization_result = self.hpo_utils.bayesian_optimization(
                    model_factory=model_factory,
                    X=X,
                    y=y,
                    search_space=search_space,
                    n_trials=self.hpo_config.meta_model_n_trials,
                    scoring=self.hpo_config.meta_model_scoring,
                    cv=cv_strategy,
                    timeout=self.hpo_config.meta_model_timeout,
                    optimization_context=f"Meta Model Optimization - {model_name} hyperparameter tuning for regime ensemble meta-learning with stacked generalization and cross-validated predictions",
                    study_name=f"meta_model_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            
            optimization_time = time.time() - start_time
            self.logger.info(f"✅ Meta-model optimization completed in {optimization_time:.2f}s - Best score: {optimization_result.get('best_score', 0):.4f}")
            
            return {
                'result': optimization_result,
                'optimization_time': optimization_time,
                'strategy': self.hpo_config.optimization_strategy
            }
            
        except Exception as e:
            self.logger.error(f"❌ Meta-model optimization failed: {e}")
            return {'error': str(e)}
    
    def optimize_meta_features(self, 
                              X: np.ndarray, 
                              y: np.ndarray,
                              base_model_predictions: np.ndarray) -> Dict[str, Any]:
        """
        Optimize meta-feature configurations for regime detection.
        
        Args:
            X: Feature matrix
            y: Target array (regime labels)
            base_model_predictions: Predictions from base models
            
        Returns:
            Optimization results for meta-features
        """
        if not self.hpo_config.enable_meta_feature_optimization:
            self.logger.info("ℹ️ Meta-feature optimization disabled")
            return {'disabled': True}
        
        self.logger.info("🎯 Starting meta-feature optimization")
        start_time = time.time()
        
        # Create meta-feature search space
        meta_feature_search_space = self._create_meta_feature_search_space()
        
        try:
            # Create meta-feature factory
            meta_feature_factory = self._create_meta_feature_factory(base_model_predictions)
            
            # Perform optimization
            optimization_result = self.hpo_utils.bayesian_optimization(
                model_factory=meta_feature_factory,
                X=X,
                y=y,
                search_space=meta_feature_search_space,
                n_trials=self.hpo_config.meta_feature_n_trials,
                scoring=self.hpo_config.meta_feature_scoring,
                optimization_context="Meta Feature Optimization - Ensemble feature selection and weighting optimization for regime detection using advanced feature engineering and ensemble stacking",
                study_name=f"meta_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            optimization_time = time.time() - start_time
            self.logger.info(f"✅ Meta-feature optimization completed in {optimization_time:.2f}s")
            
            return {
                'result': optimization_result,
                'optimization_time': optimization_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ Meta-feature optimization failed: {e}")
            return {'error': str(e)}
    
    def hierarchical_optimization(self, 
                                 X: np.ndarray, 
                                 y: np.ndarray) -> RegimeHPOResult:
        """
        Perform hierarchical optimization: base models → meta model → meta features.
        
        Args:
            X: Feature matrix
            y: Target array (regime labels)
            
        Returns:
            Complete hierarchical optimization results
        """
        self.logger.info("🏗️ Starting hierarchical regime optimization")
        total_start_time = time.time()
        
        # Phase 1: Optimize base models
        self.logger.info("📊 Phase 1: Optimizing base models...")
        base_results = self.optimize_regime_base_models(X, y)
        
        # Phase 2: Optimize meta model
        self.logger.info("📊 Phase 2: Optimizing meta model...")
        meta_results = self.optimize_regime_meta_model(X, y)
        
        # Phase 3: Optimize meta features (if enabled)
        meta_feature_results = None
        if self.hpo_config.enable_meta_feature_optimization:
            self.logger.info("📊 Phase 3: Optimizing meta features...")
            # Generate base model predictions for meta-feature optimization
            base_predictions = self._generate_base_model_predictions(X, y, base_results)
            meta_feature_results = self.optimize_meta_features(X, y, base_predictions)
        
        total_time = time.time() - total_start_time
        
        # Compile results
        result = RegimeHPOResult(
            base_model_results=base_results.get('results', {}),
            base_model_best_params={model: res.get('best_params', {}) for model, res in base_results.get('results', {}).items()},
            base_model_best_scores={model: res.get('best_score', 0) for model, res in base_results.get('results', {}).items()},
            meta_model_results=meta_results.get('result', {}),
            meta_model_best_params=meta_results.get('result', {}).get('best_params', {}),
            meta_model_best_score=meta_results.get('result', {}).get('best_score', 0),
            meta_feature_results=meta_feature_results,
            meta_feature_best_params=meta_feature_results.get('result', {}).get('best_params', {}) if meta_feature_results else None,
            total_optimization_time=total_time,
            optimization_strategy='hierarchical',
            n_total_trials=sum(len(res.get('optimization_history', [])) for res in base_results.get('results', {}).values()) + 
                          len(meta_results.get('result', {}).get('optimization_history', [])),
            convergence_info={
                'base_models_converged': all('error' not in res for res in base_results.get('results', {}).values()),
                'meta_model_converged': 'error' not in meta_results,
                'meta_features_converged': meta_feature_results is None or 'error' not in meta_feature_results
            }
        )
        
        self.logger.info(f"🏆 Hierarchical optimization completed in {total_time:.2f}s")
        return result
    
    def _create_model_factory(self, model_type: str) -> Callable:
        """Create model factory for given model type."""
        if model_type == 'catboost':
            return self._create_catboost_factory()
        elif model_type == 'extratrees':
            return self._create_extratrees_factory()
        elif model_type == 'bayesian_rule_lists':
            return self._create_bayesian_rule_lists_factory()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_catboost_factory(self) -> Callable:
        """Create CatBoost model factory."""
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            self.logger.error("❌ CatBoost not available")
            raise ImportError("CatBoost not available")
        
        def factory(**params):
            return CatBoostClassifier(
                task_type='CPU',
                loss_function='MultiClass',
                grow_policy='SymmetricTree',
                bootstrap_type='Bayesian',
                eval_metric='MultiClass',
                verbose=False,
                **params
            )
        return factory
    
    def _create_extratrees_factory(self) -> Callable:
        """Create ExtraTrees model factory."""
        try:
            from sklearn.ensemble import ExtraTreesClassifier
        except ImportError:
            self.logger.error("❌ Scikit-learn not available")
            raise ImportError("Scikit-learn not available")
        
        def factory(**params):
            processed_params = params.copy()
            max_depth = processed_params.get('max_depth')

            if isinstance(max_depth, str):
                if max_depth.strip().lower() in {'none', 'null', ''}:
                    processed_params['max_depth'] = None
            elif max_depth is not None and isinstance(max_depth, float) and np.isnan(max_depth):
                processed_params['max_depth'] = None

            return ExtraTreesClassifier(
                bootstrap=False,
                criterion='gini',
                random_state=42,
                **processed_params
            )
        return factory
    
    def _create_bayesian_rule_lists_factory(self) -> Callable:
        """Create Bayesian Rule Lists model factory."""
        try:
            from imodels import BayesianRuleListClassifier
        except ImportError:
            self.logger.error("❌ imodels not available")
            raise ImportError("imodels not available")
        
        def factory(**params):
            return BayesianRuleListClassifier(**params)
        return factory
    
    def _create_lightgbm_meta_factory(self) -> Callable:
        """Create LightGBM meta model factory."""
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            self.logger.error("❌ LightGBM not available")
            raise ImportError("LightGBM not available")
        
        def factory(**params):
            return LGBMClassifier(
                objective='multiclass',
                boosting='gbdt',
                metric='multi_logloss',
                verbose=-1,
                **params
            )
        return factory
    
    def _create_meta_feature_search_space(self) -> Dict[str, Any]:
        """Create search space for meta-feature optimization."""
        return {
            'margin_logit': {'type': 'float', 'low': -3.0, 'high': 3.0},
            'entropy_logit': {'type': 'float', 'low': -3.0, 'high': 3.0},
            'gini_logit': {'type': 'float', 'low': -3.0, 'high': 3.0},
            'variance_logit': {'type': 'float', 'low': -3.0, 'high': 3.0},
            'disagreement_logit': {'type': 'float', 'low': -3.0, 'high': 3.0},
            'js_divergence_logit': {'type': 'float', 'low': -3.0, 'high': 3.0},
            'temporal_logit': {'type': 'float', 'low': -3.0, 'high': 3.0},
            'regime_persistence_logit': {'type': 'float', 'low': -3.0, 'high': 3.0}
        }
    
    def _create_meta_feature_factory(self, base_predictions: np.ndarray) -> Callable:
        """Create meta-feature factory."""
        def factory(**params):
            # Convert sampled logits into a normalized weight dictionary
            weight_pairs = [
                ('margin_logit', 'margin_weight'),
                ('entropy_logit', 'entropy_weight'),
                ('gini_logit', 'gini_weight'),
                ('variance_logit', 'variance_weight'),
                ('disagreement_logit', 'disagreement_weight'),
                ('js_divergence_logit', 'js_divergence_weight'),
                ('temporal_logit', 'temporal_weight'),
                ('regime_persistence_logit', 'regime_persistence_weight')
            ]
            logits = np.array([params.get(key, 0.0) for key, _ in weight_pairs], dtype=float)
            # Stabilize exponentiation to avoid overflow
            logits = logits - np.max(logits)
            exp_logits = np.exp(logits)
            denom = exp_logits.sum()
            if denom == 0 or not np.isfinite(denom):
                normalized = np.full_like(exp_logits, 1.0 / len(exp_logits))
            else:
                normalized = exp_logits / denom
            normalized_weights = {
                weight_key: float(weight_value)
                for weight_value, (_, weight_key) in zip(normalized, weight_pairs)
            }

            # This would create a meta-feature extractor with optimized weights
            # For now, return a placeholder
            class MetaFeatureExtractor:
                def __init__(self, weights):
                    self.weights = weights
                
                def fit(self, X, y):
                    return self
                
                def transform(self, X):
                    # Implement meta-feature extraction logic
                    return X
                
                def fit_transform(self, X, y):
                    return self.fit(X, y).transform(X)
            
            return MetaFeatureExtractor(normalized_weights)
        return factory
    
    def _generate_base_model_predictions(self, X: np.ndarray, y: np.ndarray, base_results: Dict[str, Any]) -> np.ndarray:
        """Generate predictions from optimized base models."""
        # This would train the best base models and generate predictions
        # For now, return random predictions as placeholder
        return np.random.rand(len(X), len(np.unique(y)))
    
    def save_optimization_results(self, results: RegimeHPOResult, filepath: str):
        """Save optimization results to file."""
        try:
            # Convert results to serializable format
            serializable_results = {
                'base_model_best_params': results.base_model_best_params,
                'base_model_best_scores': results.base_model_best_scores,
                'meta_model_best_params': results.meta_model_best_params,
                'meta_model_best_score': results.meta_model_best_score,
                'meta_feature_best_params': results.meta_feature_best_params,
                'total_optimization_time': results.total_optimization_time,
                'optimization_strategy': results.optimization_strategy,
                'n_total_trials': results.n_total_trials,
                'convergence_info': results.convergence_info,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                yaml.dump(serializable_results, f, default_flow_style=False)
            
            self.logger.info(f"💾 Optimization results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save optimization results: {e}")


# Convenience functions for easy integration
def optimize_regime_models(X: np.ndarray, 
                          y: np.ndarray,
                          regime_type: str = 'all',
                          config: Optional[RegimeHPOConfig] = None) -> RegimeHPOResult:
    """
    Convenience function for regime model optimization.
    
    Args:
        X: Feature matrix
        y: Target array (regime labels)
        regime_type: Type of regime
        config: HPO configuration
        
    Returns:
        Optimization results
    """
    wrapper = RegimeHPOWrapper(hpo_config=config)
    return wrapper.hierarchical_optimization(X, y)


def create_regime_hpo_config(**kwargs) -> RegimeHPOConfig:
    """Create regime HPO configuration with custom parameters."""
    return RegimeHPOConfig(**kwargs)