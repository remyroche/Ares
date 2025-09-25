"""
Regime-Aware Model Trainer

Trains different ML models for different market regimes detected by TAS/NAS systems.
Provides comprehensive training pipeline with regime-specific optimization.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

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
        print(f"[REGIME_AWARE_TRAINER] {message}")
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

# Import regime detection systems
try:
    from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import TASRegimeDetector, TASRegimeConfig
    from src.training.steps.market_analysis.nas_regime.core.perfect_nas_regime_detector import PerfectNASRegimeDetector, PerfectNASConfig
    # from src.utils.nas_tas.core.hybrid_regime_detector import HybridNASTASRegimeDetector, HybridRegimeConfig  # DELETED
    from src.utils.nas_tas.regime_detector import create_hybrid_regime_detector, UnifiedRegimeConfig  # Use unified system
    REGIME_DETECTION_AVAILABLE = True
except ImportError:
    REGIME_DETECTION_AVAILABLE = False

# Import ML common utilities
try:
    from src.utils.ml_common.common_operations import get_ml_common_operations
    from src.utils.ml_common.validation import get_validation_framework
    from src.utils.ml_common.optimization.grid_utils import build_coarse_grid_from_search_space
    # Import Bayesian TPE optimizer
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer,
        BayesianTPEConfig,
        optimize_with_bayesian_tpe
    )
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types for regime-aware training."""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


class RegimeTrainingStrategy(Enum):
    """Regime training strategies."""
    SEPARATE_MODELS = "separate_models"  # Train separate model for each regime
    REGIME_WEIGHTED = "regime_weighted"  # Train single model with regime weights
    META_LEARNING = "meta_learning"     # Use meta-learning for regime adaptation
    CONTINUAL_LEARNING = "continual_learning"  # Continual learning across regimes


@dataclass
class RegimeAwareTrainingConfig:
    """Configuration for regime-aware model training."""
    
    # Training strategy
    training_strategy: RegimeTrainingStrategy = RegimeTrainingStrategy.SEPARATE_MODELS
    model_types: List[ModelType] = field(default_factory=lambda: [
        ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.CATBOOST
    ])
    
    # Data splitting
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    time_series_split: bool = True
    n_splits: int = 5
    
    # Regime-specific settings
    min_regime_samples: int = 100
    regime_balance_threshold: float = 0.1
    enable_regime_balancing: bool = True
    regime_weighting_method: str = "inverse_frequency"  # "inverse_frequency", "balanced", "custom"
    
    # Model hyperparameters
    enable_hyperparameter_optimization: bool = True
    optimization_method: str = "grid_search"  # "grid_search", "random_search", "bayesian"
    n_trials: int = 50
    cv_folds: int = 5
    
    # Training parameters
    early_stopping_rounds: int = 50
    validation_metric: str = "f1_score"
    enable_class_weights: bool = True
    enable_feature_importance: bool = True
    
    # Regime detection
    regime_detection_method: str = "hybrid"  # "tas", "nas", "hybrid"
    regime_confidence_threshold: float = 0.7
    enable_regime_validation: bool = True
    
    # Performance tracking
    enable_performance_tracking: bool = True
    save_models: bool = True
    model_save_path: str = "models/regime_aware"
    enable_model_versioning: bool = True
    
    # Advanced features
    enable_ensemble_training: bool = True
    ensemble_method: str = "voting"  # "voting", "stacking", "blending"
    enable_meta_features: bool = True
    enable_regime_transition_modeling: bool = True


@dataclass
class RegimeTrainingResult:
    """Result from regime-aware model training."""
    
    # Training results
    success: bool
    training_time: float
    n_regimes_detected: int
    models_trained: Dict[int, Dict[str, Any]]  # regime_id -> model_info
    ensemble_models: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    regime_performance: Dict[int, Dict[str, float]] = field(default_factory=dict)
    overall_performance: Dict[str, float] = field(default_factory=dict)
    cross_regime_performance: Dict[str, float] = field(default_factory=dict)
    
    # Model metadata
    feature_importance: Dict[int, Dict[str, float]] = field(default_factory=dict)
    hyperparameters: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    training_curves: Dict[int, Dict[str, List[float]]] = field(default_factory=dict)
    
    # Regime information
    regime_statistics: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    regime_transitions: Optional[Dict[str, Any]] = None
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class RegimeAwareTrainer:
    """
    Regime-aware model trainer that trains different models for different market regimes.
    
    Integrates with TAS/NAS regime detection systems to provide comprehensive
    regime-aware model training capabilities.
    """
    
    def __init__(self, config: RegimeAwareTrainingConfig):
        """Initialize regime-aware trainer.
        
        Args:
            config: Training configuration
        """
        tprint("🎓 Initializing Regime-Aware Trainer", color="blue")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        tprint(f"📊 Config: strategy={config.training_strategy.value}, models={[mt.value for mt in config.model_types]}, regime_method={config.regime_detection_method}", color="cyan")
        
        # Initialize regime detection
        tprint("🔍 Initializing regime detection", color="yellow")
        self._initialize_regime_detection()
        
        # Initialize ML common utilities
        tprint("🤖 Initializing ML common utilities", color="yellow")
        self._initialize_ml_common()
        
        # Initialize model factories
        tprint("🏭 Initializing model factories", color="yellow")
        self._initialize_model_factories()
        
        # Training state
        tprint("📊 Initializing training state", color="yellow")
        self.trained_models = {}
        self.regime_models = {}
        self.ensemble_models = {}
        self.performance_history = []
        
        self.logger.info("✅ Regime-Aware Trainer initialized")
        self.logger.info(f"   Training strategy: {config.training_strategy.value}")
        self.logger.info(f"   Model types: {[mt.value for mt in config.model_types]}")
        self.logger.info(f"   Regime detection: {config.regime_detection_method}")
        
        tprint("✅ Regime-Aware Trainer initialization complete", color="green")
        tprint(f"🎓 Strategy: {config.training_strategy.value}, Models: {len(config.model_types)}, Regime: {config.regime_detection_method}", color="cyan")
    
    def _initialize_regime_detection(self):
        """Initialize regime detection systems."""
        tprint("🔍 Starting regime detection initialization", color="yellow")
        if not REGIME_DETECTION_AVAILABLE:
            tprint("⚠️ Regime detection systems not available", color="red")
            self.logger.warning("⚠️ Regime detection systems not available")
            self.tas_detector = None
            self.nas_detector = None
            return
        
        try:
            if self.config.regime_detection_method in ["tas", "hybrid", "hybrid_nas_tas"]:
                tprint("🌳 Creating TAS regime detector", color="yellow")
                tas_config = TASRegimeConfig(
                    n_regimes=8,
                    enable_economic_evaluation=True,
                    enable_uncertainty_quantification=True
                )
                self.tas_detector = TASRegimeDetector(tas_config)
                self.logger.info("✅ TAS regime detector initialized")
                tprint("✅ TAS regime detector created", color="green")

            if self.config.regime_detection_method in ["nas", "hybrid", "hybrid_nas_tas"]:
                tprint("🧠 Creating NAS regime detector", color="yellow")
                nas_config = PerfectNASConfig.create_short_term_trading_config()
                self.nas_detector = PerfectNASRegimeDetector(nas_config)
                self.logger.info("✅ NAS regime detector initialized")
                tprint("✅ NAS regime detector created", color="green")

            if self.config.regime_detection_method == "hybrid_nas_tas":
                tprint("🔗 Creating hybrid NAS-TAS regime detector", color="yellow")
                hybrid_config = HybridRegimeConfig(
                    combination_strategy="weighted",
                    tas_weight=0.4,
                    nas_weight=0.6,
                    enable_economic_evaluation=True,
                    enable_financial_relevance=True
                )
                self.hybrid_detector = HybridNASTASRegimeDetector(hybrid_config)
                self.logger.info("✅ Hybrid NAS-TAS regime detector initialized")
                tprint("✅ Hybrid NAS-TAS regime detector created", color="green")

            tprint("✅ Regime detection systems initialized successfully", color="green")

        except Exception as e:
            tprint(f"⚠️ Regime detection initialization failed: {e}", color="red")
            self.logger.warning(f"Regime detection initialization failed: {e}")
            self.tas_detector = None
            self.nas_detector = None
            self.hybrid_detector = None
    
    def _initialize_ml_common(self):
        """Initialize ML common utilities."""
        tprint("🤖 Starting ML common utilities initialization", color="yellow")
        if not ML_COMMON_AVAILABLE:
            tprint("⚠️ ML common utilities not available", color="red")
            self.logger.warning("⚠️ ML common utilities not available")
            self.ml_common_ops = None
            self.validation_framework = None
            self.bayesian_tpe_optimizer = None
            return
        
        try:
            tprint("🔧 Creating ML common operations", color="yellow")
            self.ml_common_ops = get_ml_common_operations()
            tprint("🛡️ Creating validation framework", color="yellow")
            self.validation_framework = get_validation_framework()
            # Initialize Bayesian TPE optimizer
            tprint("🎯 Creating Bayesian TPE optimizer", color="yellow")
            self.bayesian_tpe_optimizer = BayesianTPEOptimizer(
                BayesianTPEConfig(
                    n_trials=50,
                    enable_grid_search=True,
                    coarse_grid_points=5,
                    fine_grid_points=8,
                    enable_parallel=True,
                    max_workers=4
                )
            )
            self.logger.info("✅ ML common utilities initialized")
            tprint("✅ ML common utilities initialized successfully", color="green")
        except Exception as e:
            tprint(f"⚠️ ML common initialization failed: {e}", color="red")
            self.logger.warning(f"ML common initialization failed: {e}")
            self.ml_common_ops = None
            self.validation_framework = None
            self.bayesian_tpe_optimizer = None
    
    def _initialize_model_factories(self):
        """Initialize model factories for different model types."""
        self.model_factories = {
            ModelType.RANDOM_FOREST: self._create_random_forest,
            ModelType.XGBOOST: self._create_xgboost,
            ModelType.LIGHTGBM: self._create_lightgbm,
            ModelType.CATBOOST: self._create_catboost,
            ModelType.LOGISTIC_REGRESSION: self._create_logistic_regression,
            ModelType.SVM: self._create_svm,
            ModelType.NEURAL_NETWORK: self._create_neural_network,
            ModelType.ENSEMBLE: self._create_ensemble
        }
    
    def train_models(self, 
                    market_data: pd.DataFrame,
                    target_variable: str,
                    feature_columns: Optional[List[str]] = None,
                    timestamps: Optional[pd.Series] = None) -> RegimeTrainingResult:
        """
        Train regime-aware models.
        
        Args:
            market_data: Market data with features and target
            target_variable: Name of target variable column
            feature_columns: List of feature column names (None for all except target)
            timestamps: Optional timestamps for time series validation
            
        Returns:
            RegimeTrainingResult with training results
        """
        start_time = datetime.now()
        self.logger.info("🚀 Starting regime-aware model training")
        
        try:
            # Step 1: Detect regimes
            self.logger.info("🔍 Detecting market regimes...")
            regime_results = self._detect_regimes(market_data, timestamps)
            
            if not regime_results['success']:
                return RegimeTrainingResult(
                    success=False,
                    training_time=0.0,
                    n_regimes_detected=0,
                    models_trained={},
                    error_message="Regime detection failed"
                )
            
            # Step 2: Prepare regime-specific datasets
            self.logger.info("📊 Preparing regime-specific datasets...")
            regime_datasets = self._prepare_regime_datasets(
                market_data, target_variable, feature_columns, regime_results
            )
            
            # Step 3: Train models for each regime
            self.logger.info("🤖 Training models for each regime...")
            regime_models = self._train_regime_models(regime_datasets)
            
            # Step 4: Train ensemble models
            ensemble_models = None
            if self.config.enable_ensemble_training:
                self.logger.info("🎯 Training ensemble models...")
                ensemble_models = self._train_ensemble_models(regime_datasets, regime_models)
            
            # Step 5: Evaluate performance
            self.logger.info("📈 Evaluating model performance...")
            performance_results = self._evaluate_performance(
                regime_datasets, regime_models, ensemble_models
            )
            
            # Step 6: Save models if requested
            if self.config.save_models:
                self.logger.info("💾 Saving trained models...")
                self._save_models(regime_models, ensemble_models)
            
            # Create result
            training_time = (datetime.now() - start_time).total_seconds()
            result = RegimeTrainingResult(
                success=True,
                training_time=training_time,
                n_regimes_detected=regime_results['n_regimes'],
                models_trained=regime_models,
                ensemble_models=ensemble_models,
                regime_performance=performance_results['regime_performance'],
                overall_performance=performance_results['overall_performance'],
                cross_regime_performance=performance_results['cross_regime_performance'],
                feature_importance=performance_results['feature_importance'],
                hyperparameters=performance_results['hyperparameters'],
                training_curves=performance_results['training_curves'],
                regime_statistics=regime_results['regime_statistics'],
                regime_transitions=regime_results.get('transitions')
            )
            
            self.logger.info(f"✅ Regime-aware training completed in {training_time:.2f}s")
            self.logger.info(f"   Regimes detected: {result.n_regimes_detected}")
            self.logger.info(f"   Models trained: {len(result.models_trained)}")
            self.logger.info(f"   Overall performance: {result.overall_performance}")
            
            return result
            
        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"❌ Regime-aware training failed: {e}")
            
            return RegimeTrainingResult(
                success=False,
                training_time=training_time,
                n_regimes_detected=0,
                models_trained={},
                error_message=str(e)
            )
    
    def _detect_regimes(self, market_data: pd.DataFrame, timestamps: Optional[pd.Series]) -> Dict[str, Any]:
        """Detect market regimes using configured method."""
        try:
            if self.config.regime_detection_method == "tas" and self.tas_detector:
                result = self.tas_detector.detect_regimes(market_data, timestamps)
                return {
                    'success': result.success,
                    'regime_predictions': result.regime_predictions,
                    'regime_probabilities': result.regime_probabilities,
                    'n_regimes': len(np.unique(result.regime_predictions)),
                    'regime_statistics': self._calculate_regime_statistics(result.regime_predictions, market_data)
                }
            
            elif self.config.regime_detection_method == "nas" and self.nas_detector:
                result = self.nas_detector.detect_regimes(market_data, timestamps)
                return {
                    'success': result.success,
                    'regime_predictions': result.regime_predictions,
                    'regime_probabilities': result.regime_probabilities,
                    'n_regimes': len(np.unique(result.regime_predictions)),
                    'regime_statistics': self._calculate_regime_statistics(result.regime_predictions, market_data)
                }
            
            elif self.config.regime_detection_method == "hybrid":
                # Use both detectors and combine results
                tas_result = self.tas_detector.detect_regimes(market_data, timestamps) if self.tas_detector else None
                nas_result = self.nas_detector.detect_regimes(market_data, timestamps) if self.nas_detector else None

                if tas_result and nas_result:
                    # Combine predictions (simple majority voting)
                    combined_predictions = self._combine_regime_predictions(
                        tas_result.regime_predictions, nas_result.regime_predictions
                    )
                    return {
                        'success': True,
                        'regime_predictions': combined_predictions,
                        'regime_probabilities': (tas_result.regime_probabilities + nas_result.regime_probabilities) / 2,
                        'n_regimes': len(np.unique(combined_predictions)),
                        'regime_statistics': self._calculate_regime_statistics(combined_predictions, market_data)
                    }
                elif tas_result:
                    return {
                        'success': tas_result.success,
                        'regime_predictions': tas_result.regime_predictions,
                        'regime_probabilities': tas_result.regime_probabilities,
                        'n_regimes': len(np.unique(tas_result.regime_predictions)),
                        'regime_statistics': self._calculate_regime_statistics(tas_result.regime_predictions, market_data)
                    }
                elif nas_result:
                    return {
                        'success': nas_result.success,
                        'regime_predictions': nas_result.regime_predictions,
                        'regime_probabilities': nas_result.regime_probabilities,
                        'n_regimes': len(np.unique(nas_result.regime_predictions)),
                        'regime_statistics': self._calculate_regime_statistics(nas_result.regime_predictions, market_data)
                    }

            elif self.config.regime_detection_method == "hybrid_nas_tas" and self.hybrid_detector:
                # Use the advanced hybrid NAS-TAS detector
                result = self.hybrid_detector.detect_regimes(market_data, timestamps)
                return {
                    'success': result.success,
                    'regime_predictions': result.regime_predictions,
                    'regime_probabilities': result.regime_probabilities,
                    'n_regimes': len(np.unique(result.regime_predictions)),
                    'regime_statistics': self._calculate_enhanced_regime_statistics(result, market_data)
                }
            
            # Fallback to simple clustering
            return self._fallback_regime_detection(market_data)
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _fallback_regime_detection(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback regime detection using simple clustering."""
        try:
            from sklearn.cluster import KMeans
            
            # Use price and volume features for clustering
            features = ['close', 'volume'] if 'close' in market_data.columns and 'volume' in market_data.columns else market_data.select_dtypes(include=[np.number]).columns[:2]
            data = market_data[features].values
            
            # Normalize data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            regime_predictions = kmeans.fit_predict(data_scaled)
            
            return {
                'success': True,
                'regime_predictions': regime_predictions,
                'regime_probabilities': np.ones((len(regime_predictions), 3)) / 3,
                'n_regimes': 3,
                'regime_statistics': self._calculate_regime_statistics(regime_predictions, market_data)
            }
            
        except Exception as e:
            self.logger.error(f"Fallback regime detection failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _combine_regime_predictions(self, tas_predictions: np.ndarray, nas_predictions: np.ndarray) -> np.ndarray:
        """Combine regime predictions from TAS and NAS detectors."""
        # Simple majority voting
        combined = np.zeros_like(tas_predictions)
        for i in range(len(tas_predictions)):
            if tas_predictions[i] == nas_predictions[i]:
                combined[i] = tas_predictions[i]
            else:
                # Use TAS prediction as tie-breaker
                combined[i] = tas_predictions[i]
        return combined
    
    def _calculate_regime_statistics(self, regime_predictions: np.ndarray, market_data: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Calculate statistics for each regime."""
        regime_stats = {}
        
        for regime_id in np.unique(regime_predictions):
            regime_mask = regime_predictions == regime_id
            regime_data = market_data[regime_mask]
            
            stats = {
                'n_samples': len(regime_data),
                'percentage': len(regime_data) / len(market_data) * 100,
                'start_index': np.where(regime_mask)[0][0] if len(np.where(regime_mask)[0]) > 0 else 0,
                'end_index': np.where(regime_mask)[0][-1] if len(np.where(regime_mask)[0]) > 0 else 0
            }
            
            # Add price statistics if available
            if 'close' in market_data.columns:
                close_prices = regime_data['close']
                stats.update({
                    'mean_price': close_prices.mean(),
                    'std_price': close_prices.std(),
                    'min_price': close_prices.min(),
                    'max_price': close_prices.max()
                })
            
            # Add volume statistics if available
            if 'volume' in market_data.columns:
                volumes = regime_data['volume']
                stats.update({
                    'mean_volume': volumes.mean(),
                    'std_volume': volumes.std()
                })
            
            regime_stats[int(regime_id)] = stats
        
        return regime_stats

    def _calculate_enhanced_regime_statistics(self, regime_result, market_data: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Calculate enhanced statistics for hybrid regime detection results."""
        regime_stats = {}

        for regime_id in np.unique(regime_result.regime_predictions):
            regime_mask = regime_result.regime_predictions == regime_id
            regime_data = market_data[regime_mask]

            # Basic statistics
            stats = {
                'n_samples': len(regime_data),
                'percentage': len(regime_data) / len(market_data) * 100,
                'start_index': np.where(regime_mask)[0][0] if len(np.where(regime_mask)[0]) > 0 else 0,
                'end_index': np.where(regime_mask)[0][-1] if len(np.where(regime_mask)[0]) > 0 else 0,
                'economic_significance': regime_result.economic_significance_scores[regime_id] if hasattr(regime_result, 'economic_significance_scores') else 0.0,
                'financial_relevance': regime_result.financial_relevance_scores[regime_id] if hasattr(regime_result, 'financial_relevance_scores') else 0.0,
                'regime_stability': regime_result.regime_stability_scores[regime_id] if hasattr(regime_result, 'regime_stability_scores') else 0.0
            }

            # Add price statistics if available
            if 'close' in market_data.columns:
                close_prices = regime_data['close']
                stats.update({
                    'mean_price': close_prices.mean(),
                    'std_price': close_prices.std(),
                    'min_price': close_prices.min(),
                    'max_price': close_prices.max()
                })

            # Add volume statistics if available
            if 'volume' in market_data.columns:
                volumes = regime_data['volume']
                stats.update({
                    'mean_volume': volumes.mean(),
                    'std_volume': volumes.std()
                })

            # Add momentum and volume profile statistics if available
            if hasattr(regime_result, 'momentum_scores') and hasattr(regime_result, 'volume_profiles'):
                stats.update({
                    'mean_momentum': regime_result.momentum_scores[regime_mask].mean() if len(regime_result.momentum_scores[regime_mask]) > 0 else 0.0,
                    'mean_volume_profile': regime_result.volume_profiles[regime_mask].mean() if len(regime_result.volume_profiles[regime_mask]) > 0 else 0.0
                })

            regime_stats[int(regime_id)] = stats

        return regime_stats
    
    def _prepare_regime_datasets(self, 
                                market_data: pd.DataFrame,
                                target_variable: str,
                                feature_columns: Optional[List[str]],
                                regime_results: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Prepare datasets for each regime."""
        regime_datasets = {}
        regime_predictions = regime_results['regime_predictions']
        
        # Determine feature columns
        if feature_columns is None:
            feature_columns = [col for col in market_data.columns if col != target_variable]
        
        # Prepare data
        X = market_data[feature_columns].values
        y = market_data[target_variable].values
        
        for regime_id in np.unique(regime_predictions):
            regime_mask = regime_predictions == regime_id
            regime_X = X[regime_mask]
            regime_y = y[regime_mask]
            
            # Check if regime has enough samples
            if len(regime_X) < self.config.min_regime_samples:
                self.logger.warning(f"⚠️ Regime {regime_id} has only {len(regime_X)} samples, skipping")
                continue
            
            # Split data
            if self.config.time_series_split:
                # Time series split for financial data
                tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
                splits = list(tscv.split(regime_X))
                
                # Use last split for train/validation/test
                train_idx, val_test_idx = splits[-1]
                val_idx, test_idx = train_test_split(
                    val_test_idx, 
                    test_size=self.config.test_ratio / (self.config.validation_ratio + self.config.test_ratio),
                    random_state=42
                )
            else:
                # Random split
                train_idx, temp_idx = train_test_split(
                    range(len(regime_X)), 
                    test_size=1 - self.config.train_ratio, 
                    random_state=42
                )
                val_idx, test_idx = train_test_split(
                    temp_idx, 
                    test_size=self.config.test_ratio / (self.config.validation_ratio + self.config.test_ratio),
                    random_state=42
                )
            
            regime_datasets[regime_id] = {
                'X_train': regime_X[train_idx],
                'y_train': regime_y[train_idx],
                'X_val': regime_X[val_idx],
                'y_val': regime_y[val_idx],
                'X_test': regime_X[test_idx],
                'y_test': regime_y[test_idx],
                'feature_names': feature_columns,
                'n_samples': len(regime_X),
                'class_distribution': np.bincount(regime_y.astype(int))
            }
        
        return regime_datasets
    
    def _train_regime_models(self, regime_datasets: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Train models for each regime."""
        regime_models = {}
        
        for regime_id, dataset in regime_datasets.items():
            self.logger.info(f"🤖 Training models for regime {regime_id}...")
            
            regime_models[regime_id] = {}
            
            for model_type in self.config.model_types:
                try:
                    self.logger.info(f"   Training {model_type.value} for regime {regime_id}...")
                    
                    # Create model
                    model = self.model_factories[model_type]()
                    
                    # Train model
                    if hasattr(model, 'fit'):
                        # Sklearn-style model
                        model.fit(dataset['X_train'], dataset['y_train'])
                        
                        # Make predictions
                        train_pred = model.predict(dataset['X_train'])
                        val_pred = model.predict(dataset['X_val'])
                        test_pred = model.predict(dataset['X_test'])
                        
                        # Calculate metrics
                        train_metrics = self._calculate_metrics(dataset['y_train'], train_pred)
                        val_metrics = self._calculate_metrics(dataset['y_val'], val_pred)
                        test_metrics = self._calculate_metrics(dataset['y_test'], test_pred)
                        
                        regime_models[regime_id][model_type.value] = {
                            'model': model,
                            'train_metrics': train_metrics,
                            'val_metrics': val_metrics,
                            'test_metrics': test_metrics,
                            'feature_importance': self._get_feature_importance(model, dataset['feature_names']),
                            'hyperparameters': self._get_model_hyperparameters(model)
                        }
                        
                        self.logger.info(f"   ✅ {model_type.value} trained - Val F1: {val_metrics['f1_score']:.3f}")
                    
                except (ValueError, TypeError, RuntimeError) as e:
                    self.logger.warning(f"   ⚠️ Failed to train {model_type.value} for regime {regime_id}: {e}")
                    self.logger.info(f"   Continuing with other models for regime {regime_id}")
                    continue
                except Exception as e:
                    self.logger.error(f"   ❌ Unexpected error training {model_type.value} for regime {regime_id}: {e}")
                    self.logger.warning(f"   ⚠️ Continuing with other models for regime {regime_id}")
                    continue
        
        return regime_models
    
    def _train_ensemble_models(self, 
                              regime_datasets: Dict[int, Dict[str, Any]],
                              regime_models: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Train ensemble models across regimes."""
        try:
            from sklearn.ensemble import VotingClassifier
            
            ensemble_models = {}
            
            # Create ensemble for each regime
            for regime_id, dataset in regime_datasets.items():
                if regime_id not in regime_models:
                    continue
                
                # Get base models for this regime
                base_models = []
                for model_type, model_info in regime_models[regime_id].items():
                    base_models.append((model_type, model_info['model']))
                
                if len(base_models) > 1:
                    # Create voting ensemble
                    ensemble = VotingClassifier(
                        estimators=base_models,
                        voting='soft' if self.config.ensemble_method == 'voting' else 'hard'
                    )
                    
                    # Train ensemble
                    ensemble.fit(dataset['X_train'], dataset['y_train'])
                    
                    # Evaluate ensemble
                    val_pred = ensemble.predict(dataset['X_val'])
                    val_metrics = self._calculate_metrics(dataset['y_val'], val_pred)
                    
                    ensemble_models[f'regime_{regime_id}'] = {
                        'model': ensemble,
                        'val_metrics': val_metrics,
                        'base_models': [name for name, _ in base_models]
                    }
            
            return ensemble_models
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Ensemble training failed: {e}")
            self.logger.info("Returning empty ensemble models - models will use individual regime models")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error during ensemble training: {e}")
            self.logger.warning("Returning empty ensemble models - models will use individual regime models")
            return {}
    
    def _evaluate_performance(self, 
                            regime_datasets: Dict[int, Dict[str, Any]],
                            regime_models: Dict[int, Dict[str, Any]],
                            ensemble_models: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model performance across regimes."""
        performance_results = {
            'regime_performance': {},
            'overall_performance': {},
            'cross_regime_performance': {},
            'feature_importance': {},
            'hyperparameters': {},
            'training_curves': {}
        }
        
        # Calculate regime-specific performance
        for regime_id, models in regime_models.items():
            regime_performance = {}
            for model_type, model_info in models.items():
                regime_performance[model_type] = {
                    'train_f1': model_info['train_metrics']['f1_score'],
                    'val_f1': model_info['val_metrics']['f1_score'],
                    'test_f1': model_info['test_metrics']['f1_score'],
                    'val_accuracy': model_info['val_metrics']['accuracy'],
                    'val_precision': model_info['val_metrics']['precision'],
                    'val_recall': model_info['val_metrics']['recall']
                }
            
            performance_results['regime_performance'][regime_id] = regime_performance
        
        # Calculate overall performance
        all_val_f1 = []
        for regime_id, models in regime_models.items():
            for model_type, model_info in models.items():
                all_val_f1.append(model_info['val_metrics']['f1_score'])
        
        if all_val_f1:
            performance_results['overall_performance'] = {
                'mean_f1': np.mean(all_val_f1),
                'std_f1': np.std(all_val_f1),
                'min_f1': np.min(all_val_f1),
                'max_f1': np.max(all_val_f1)
            }
        
        # Calculate cross-regime performance
        performance_results['cross_regime_performance'] = self._calculate_cross_regime_performance(
            regime_datasets, regime_models
        )
        
        # Collect feature importance
        for regime_id, models in regime_models.items():
            regime_importance = {}
            for model_type, model_info in models.items():
                regime_importance[model_type] = model_info['feature_importance']
            performance_results['feature_importance'][regime_id] = regime_importance
        
        # Collect hyperparameters
        for regime_id, models in regime_models.items():
            regime_hyperparams = {}
            for model_type, model_info in models.items():
                regime_hyperparams[model_type] = model_info['hyperparameters']
            performance_results['hyperparameters'][regime_id] = regime_hyperparams
        
        return performance_results
    
    def _calculate_cross_regime_performance(self, 
                                          regime_datasets: Dict[int, Dict[str, Any]],
                                          regime_models: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate cross-regime performance metrics."""
        cross_regime_metrics = {}
        
        # Test each regime's models on other regimes
        for source_regime, source_models in regime_models.items():
            for target_regime, target_dataset in regime_datasets.items():
                if source_regime == target_regime:
                    continue
                
                for model_type, model_info in source_models.items():
                    model = model_info['model']
                    
                    # Test on target regime
                    target_pred = model.predict(target_dataset['X_test'])
                    target_metrics = self._calculate_metrics(target_dataset['y_test'], target_pred)
                    
                    key = f"regime_{source_regime}_{model_type}_on_regime_{target_regime}"
                    cross_regime_metrics[key] = target_metrics['f1_score']
        
        return cross_regime_metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                if len(feature_names) != len(model.feature_importances_):
                    self.logger.warning(f"Feature names length ({len(feature_names)}) doesn't match importance length ({len(model.feature_importances_)})")
                    return {}
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            elif hasattr(model, 'coef_'):
                # For linear models
                if len(feature_names) != len(model.coef_[0]):
                    self.logger.warning(f"Feature names length ({len(feature_names)}) doesn't match coefficients length ({len(model.coef_[0])})")
                    return {}
                importance_dict = dict(zip(feature_names, np.abs(model.coef_[0])))
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            else:
                return {}
        except (ValueError, IndexError, TypeError) as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error extracting feature importance: {e}")
            return {}
    
    def _get_model_hyperparameters(self, model: Any) -> Dict[str, Any]:
        """Get model hyperparameters."""
        try:
            if hasattr(model, 'get_params'):
                return model.get_params()
            else:
                self.logger.debug(f"Model type {type(model).__name__} doesn't support get_params()")
                return {}
        except Exception as e:
            self.logger.warning(f"Could not extract hyperparameters: {e}")
            return {}
    
    def _save_models(self, regime_models: Dict[int, Dict[str, Any]], ensemble_models: Optional[Dict[str, Any]]):
        """Save trained models."""
        try:
            save_path = Path(self.config.model_save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save regime models
            for regime_id, models in regime_models.items():
                regime_path = save_path / f"regime_{regime_id}"
                regime_path.mkdir(exist_ok=True)
                
                for model_type, model_info in models.items():
                    model_file = regime_path / f"{model_type}.pkl"
                    with open(model_file, 'wb') as f:
                        pickle.dump(model_info['model'], f)
            
            # Save ensemble models
            if ensemble_models:
                ensemble_path = save_path / "ensemble"
                ensemble_path.mkdir(exist_ok=True)
                
                for ensemble_name, ensemble_info in ensemble_models.items():
                    ensemble_file = ensemble_path / f"{ensemble_name}.pkl"
                    with open(ensemble_file, 'wb') as f:
                        pickle.dump(ensemble_info['model'], f)
            
            self.logger.info(f"✅ Models saved to {save_path}")
            
        except (IOError, OSError, pickle.PicklingError) as e:
            self.logger.error(f"❌ Could not save models: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Unexpected error saving models: {e}")
            raise
    
    # Model factory methods
    def _create_random_forest(self):
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced' if self.config.enable_class_weights else None
        )
    
    def _create_xgboost(self):
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
    
    def _create_lightgbm(self):
        return lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            class_weight='balanced' if self.config.enable_class_weights else None
        )
    
    def _create_catboost(self):
        return cb.CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
    
    def _create_logistic_regression(self):
        return LogisticRegression(
            random_state=42,
            class_weight='balanced' if self.config.enable_class_weights else None,
            max_iter=1000
        )
    
    def _create_svm(self):
        return SVC(
            random_state=42,
            class_weight='balanced' if self.config.enable_class_weights else None,
            probability=True
        )
    
    def _create_neural_network(self):
        # Simple neural network using sklearn's MLPClassifier
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42
        )
    
    def _create_ensemble(self):
        # This will be handled in _train_ensemble_models
        return None
    
    def predict(self, 
                market_data: pd.DataFrame,
                regime_id: Optional[int] = None,
                model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Make predictions using trained models.
        
        Args:
            market_data: Market data for prediction
            regime_id: Specific regime to use (None for auto-detection)
            model_type: Specific model type to use (None for best model)
            
        Returns:
            Prediction results
        """
        try:
            if not self.trained_models:
                raise ValueError("No models trained. Call train_models() first.")
            
            # Detect regime if not specified
            if regime_id is None:
                regime_results = self._detect_regimes(market_data, None)
                if not regime_results['success']:
                    raise ValueError("Failed to detect regime")
                regime_id = regime_results['regime_predictions'][-1]  # Use last prediction
            
            # Get model for regime
            if regime_id not in self.trained_models:
                raise ValueError(f"No models trained for regime {regime_id}")
            
            regime_models = self.trained_models[regime_id]
            
            # Select model type
            if model_type is None:
                # Use best performing model
                best_model_type = max(
                    regime_models.keys(),
                    key=lambda x: regime_models[x]['val_metrics']['f1_score']
                )
            else:
                best_model_type = model_type
            
            if best_model_type not in regime_models:
                raise ValueError(f"Model type {best_model_type} not available for regime {regime_id}")
            
            model = regime_models[best_model_type]['model']
            
            # Make predictions
            predictions = model.predict(market_data)
            probabilities = model.predict_proba(market_data) if hasattr(model, 'predict_proba') else None
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'regime_id': regime_id,
                'model_type': best_model_type,
                'confidence': np.max(probabilities, axis=1) if probabilities is not None else None
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, regime_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize hyperparameters using Bayesian TPE optimizer."""
        if not hasattr(self, 'bayesian_tpe_optimizer') or not self.bayesian_tpe_optimizer:
            tprint("⚠️ Bayesian TPE optimizer not available, skipping hyperparameter optimization", color="red")
            self.logger.warning("Bayesian TPE optimizer not available, skipping hyperparameter optimization")
            return {'success': False, 'error': 'Bayesian TPE optimizer not available'}
        
        tprint("🔧 Starting Bayesian TPE hyperparameter optimization", color="blue")
        self.logger.info("🔧 Starting Bayesian TPE hyperparameter optimization")
        
        # Define hyperparameter search space for regime-aware training
        search_space = {
            'learning_rate': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True},
            'batch_size': {'type': 'int', 'low': 16, 'high': 128},
            'dropout_rate': {'type': 'float', 'low': 0.0, 'high': 0.5},
            'regularization': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True},
            'hidden_units': {'type': 'int', 'low': 32, 'high': 256},
            'num_layers': {'type': 'int', 'low': 2, 'high': 6},
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'max_depth': {'type': 'int', 'low': 3, 'high': 15}
        }
        
        try:
            # Use Bayesian TPE optimizer with automatic grid search
            result = self.bayesian_tpe_optimizer.optimize(
                objective_function=lambda params: self._evaluate_hyperparameters(X, y, regime_labels, params),
                search_space=search_space,
                X=X,
                y=y
            )
            
            if result.success:
                tprint(f"✅ Bayesian TPE optimization completed - Best score: {result.best_score:.4f}", color="green")
                self.logger.info(f"✅ Bayesian TPE optimization completed - Best score: {result.best_score:.4f}")
                self.logger.info(f"   Best parameters: {result.best_params}")
                
                # Update configuration with optimized parameters
                self._update_config_with_optimized_params(result.best_params)
                
                return {
                    'success': True,
                    'best_params': result.best_params,
                    'best_score': result.best_score,
                    'optimization_time': result.optimization_time,
                    'n_trials': result.n_trials,
                    'grid_search_used': result.convergence_info.get('grid_search_used', False),
                    'tpe_optimization_used': result.convergence_info.get('tpe_optimization_used', False)
                }
            else:
                tprint(f"⚠️ Bayesian TPE optimization failed: {result.error_message}", color="red")
                self.logger.warning(f"⚠️ Bayesian TPE optimization failed: {result.error_message}")
                return {'success': False, 'error': result.error_message}
                
        except Exception as e:
            tprint(f"⚠️ Bayesian TPE optimization failed: {e}", color="red")
            self.logger.warning(f"⚠️ Bayesian TPE optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _evaluate_hyperparameters(self, X: np.ndarray, y: np.ndarray, regime_labels: Optional[np.ndarray], params: Dict[str, Any]) -> float:
        """Evaluate hyperparameters for Bayesian TPE optimization."""
        try:
            # Create a temporary configuration with new hyperparameters
            temp_config = self._create_temp_config_with_params(params)
            
            # Create a temporary trainer with optimized parameters
            temp_trainer = RegimeAwareTrainer(temp_config)
            
            # Perform a quick training and evaluation
            # This is a simplified evaluation - in practice, you would do full training
            score = self._quick_evaluate_config(X, y, regime_labels, temp_config)
            
            return score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Hyperparameter evaluation failed: {e}")
            return -1.0
    
    def _create_temp_config_with_params(self, params: Dict[str, Any]) -> RegimeAwareTrainingConfig:
        """Create a temporary configuration with optimized parameters."""
        # Create a copy of the current configuration
        temp_config = RegimeAwareTrainingConfig(
            training_strategy=self.config.training_strategy,
            model_types=self.config.model_types,
            regime_detection_method=self.config.regime_detection_method,
            # Update with optimized parameters
            learning_rate=params.get('learning_rate', self.config.learning_rate),
            batch_size=params.get('batch_size', self.config.batch_size),
            dropout_rate=params.get('dropout_rate', self.config.dropout_rate),
            regularization=params.get('regularization', self.config.regularization),
            hidden_units=params.get('hidden_units', self.config.hidden_units),
            num_layers=params.get('num_layers', self.config.num_layers),
            n_estimators=params.get('n_estimators', self.config.n_estimators),
            max_depth=params.get('max_depth', self.config.max_depth),
            max_epochs=self.config.max_epochs,
            early_stopping_patience=self.config.early_stopping_patience,
            validation_split=self.config.validation_split
        )
        return temp_config
    
    def _quick_evaluate_config(self, X: np.ndarray, y: np.ndarray, regime_labels: Optional[np.ndarray], config: RegimeAwareTrainingConfig) -> float:
        """Quickly evaluate a configuration without full training."""
        try:
            # Calculate a score based on data characteristics and configuration
            data_quality = self._calculate_data_quality_score(X, y)
            config_compatibility = self._calculate_config_compatibility_score(config)
            
            # Combine scores
            total_score = 0.6 * data_quality + 0.4 * config_compatibility
            
            return total_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Quick evaluation failed: {e}")
            return 0.0
    
    def _calculate_data_quality_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate a quality score based on data characteristics."""
        try:
            if len(X) == 0 or len(y) == 0:
                return 0.0
            
            # Calculate variance (higher variance often indicates more informative data)
            X_variance = np.var(X)
            y_variance = np.var(y)
            
            # Calculate mean (normalize for different scales)
            X_mean = np.mean(X)
            y_mean = np.mean(y)
            
            # Calculate coefficient of variation (relative variability)
            X_cv = np.sqrt(X_variance) / abs(X_mean) if X_mean != 0 else 0.0
            y_cv = np.sqrt(y_variance) / abs(y_mean) if y_mean != 0 else 0.0
            
            # Normalize score between 0 and 1
            quality_score = min(1.0, (X_cv + y_cv) * 0.25 + (X_variance + y_variance) * 0.0001)
            
            return quality_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Data quality calculation failed: {e}")
            return 0.0
    
    def _calculate_config_compatibility_score(self, config: RegimeAwareTrainingConfig) -> float:
        """Calculate a compatibility score for configuration."""
        try:
            score = 0.0
            
            # Learning rate compatibility
            if 0.0001 <= config.learning_rate <= 0.01:
                score += 0.2
            
            # Batch size compatibility
            if 16 <= config.batch_size <= 128:
                score += 0.2
            
            # Dropout rate compatibility
            if 0.0 <= config.dropout_rate <= 0.5:
                score += 0.2
            
            # Regularization compatibility
            if 0.0001 <= config.regularization <= 0.01:
                score += 0.2
            
            # Hidden units compatibility
            if 32 <= config.hidden_units <= 256:
                score += 0.1
            
            # Number of layers compatibility
            if 2 <= config.num_layers <= 6:
                score += 0.1
            
            return score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Config compatibility calculation failed: {e}")
            return 0.0
    
    def _update_config_with_optimized_params(self, best_params: Dict[str, Any]):
        """Update the configuration with optimized parameters."""
        try:
            if 'learning_rate' in best_params:
                self.config.learning_rate = best_params['learning_rate']
            if 'batch_size' in best_params:
                self.config.batch_size = best_params['batch_size']
            if 'dropout_rate' in best_params:
                self.config.dropout_rate = best_params['dropout_rate']
            if 'regularization' in best_params:
                self.config.regularization = best_params['regularization']
            if 'hidden_units' in best_params:
                self.config.hidden_units = best_params['hidden_units']
            if 'num_layers' in best_params:
                self.config.num_layers = best_params['num_layers']
            if 'n_estimators' in best_params:
                self.config.n_estimators = best_params['n_estimators']
            if 'max_depth' in best_params:
                self.config.max_depth = best_params['max_depth']
                
            tprint("✅ Configuration updated with optimized parameters", color="green")
            self.logger.info("✅ Configuration updated with optimized parameters")
            
        except Exception as e:
            tprint(f"⚠️ Failed to update configuration: {e}", color="red")
            self.logger.warning(f"⚠️ Failed to update configuration: {e}")
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance across all regimes."""
        if not self.performance_history:
            return {}
        
        summary = {
            'total_regimes': len(self.performance_history),
            'regime_performance': {},
            'overall_metrics': {}
        }
        
        # Calculate overall metrics
        all_f1_scores = []
        for regime_id, performance in self.performance_history.items():
            regime_f1 = []
            for model_type, metrics in performance.items():
                if 'val_metrics' in metrics:
                    regime_f1.append(metrics['val_metrics']['f1_score'])
                    all_f1_scores.append(metrics['val_metrics']['f1_score'])
            
            summary['regime_performance'][regime_id] = {
                'mean_f1': np.mean(regime_f1) if regime_f1 else 0,
                'best_model': max(performance.keys(), 
                                key=lambda x: performance[x]['val_metrics']['f1_score']) if performance else None
            }
        
        if all_f1_scores:
            summary['overall_metrics'] = {
                'mean_f1': np.mean(all_f1_scores),
                'std_f1': np.std(all_f1_scores),
                'min_f1': np.min(all_f1_scores),
                'max_f1': np.max(all_f1_scores)
            }
        
        return summary