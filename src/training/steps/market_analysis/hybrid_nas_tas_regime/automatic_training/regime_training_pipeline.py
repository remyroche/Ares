"""
Automatic Training Pipeline for Regime-Specific Models

This module implements an automatic training pipeline that:
1. Automatically trains ML models for each detected regime
2. Performs regime-specific data extraction and preprocessing
3. Automates model validation and deployment after training
4. Supports continuous learning and model updates
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import regime detection and model selection
from ..regime_model_mapping import DataDrivenModelSelector, ModelSelectorConfig
from ..core.hybrid_regime_detector import HybridNASTASRegimeDetector
from ..config.hybrid_regime_config import HybridRegimeConfig

# Import HPO integration
from .regime_hpo_integration import RegimeHPOIntegration, create_regime_hpo_integration_config

# Import advanced tree models
from ...tas_regime.components.advanced_tree_models import (
    AdvancedTreeModelFactory, AdvancedTreeConfig,
    MetaLearningTreeModel, ContinualLearningTreeModel, CLVSAEnhancedTreeModel
)

logger = logging.getLogger(__name__)


@dataclass
class RegimeTrainingConfig:
    """Configuration for automatic regime training pipeline."""
    
    # Training parameters
    min_regime_samples: int = 100  # Minimum samples required for training
    max_regime_samples: int = 10000  # Maximum samples to use for training
    train_test_split: float = 0.8  # Training/test split ratio
    validation_split: float = 0.2  # Validation split from training data
    
    # Model training
    enable_hyperparameter_tuning: bool = True
    hyperparameter_trials: int = 50
    enable_cross_validation: bool = True
    cv_folds: int = 5
    
    # HPO Integration
    enable_hpo_optimization: bool = True
    hpo_strategy: str = 'hierarchical'  # 'hierarchical', 'staged', 'bayesian'
    hpo_base_model_trials: int = 100
    hpo_meta_model_trials: int = 50
    enable_meta_feature_optimization: bool = True
    hpo_timeout: Optional[int] = 3600  # 1 hour timeout
    
    # Model types to train
    model_types: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "catboost", "random_forest", "gradient_boosting"
    ])
    
    # Advanced features
    enable_meta_learning: bool = True
    enable_continual_learning: bool = True
    enable_patchtst_enhancement: bool = True
    enable_ensemble_training: bool = True
    
    # Data preprocessing
    enable_feature_engineering: bool = True
    enable_feature_selection: bool = True
    enable_data_augmentation: bool = True
    augmentation_factor: float = 1.5
    
    # Model validation
    validation_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1_score", "roc_auc"
    ])
    min_validation_score: float = 0.6
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Deployment
    enable_automatic_deployment: bool = True
    deployment_threshold: float = 0.8
    enable_model_versioning: bool = True
    max_model_versions: int = 5
    
    # Continuous learning
    enable_continuous_learning: bool = True
    retraining_frequency: int = 1000  # Retrain every N new samples
    performance_drift_threshold: float = 0.1
    enable_incremental_learning: bool = True


@dataclass
class RegimeTrainingResult:
    """Result from regime-specific model training."""
    regime_id: int
    model_name: str
    success: bool
    training_time: float
    validation_scores: Dict[str, float]
    model_performance: Dict[str, Any]
    model_path: Optional[str] = None
    deployment_status: str = "pending"  # "pending", "deployed", "failed"
    error_message: Optional[str] = None


class RegimeDataExtractor:
    """
    Extracts and preprocesses data for specific regimes.
    """
    
    def __init__(self, config: RegimeTrainingConfig):
        """Initialize regime data extractor."""
        tprint("🔧 [REGIME_EXTRACTOR] Initializing Regime Data Extractor", color="blue")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        tprint("✅ [REGIME_EXTRACTOR] Regime Data Extractor initialized", color="green")
        self.logger.info("✅ Regime Data Extractor initialized")
    
    def extract_regime_data(self, 
                           market_data: np.ndarray,
                           regime_predictions: np.ndarray,
                           regime_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract data for a specific regime.
        
        Args:
            market_data: Full market data
            regime_predictions: Regime predictions for all data
            regime_id: ID of the regime to extract
            
        Returns:
            Tuple of (regime_data, regime_labels, regime_features)
        """
        tprint(f"🔍 [REGIME_EXTRACTOR] extract_regime_data() called for regime {regime_id}", color="blue")
        tprint(f"📊 [REGIME_EXTRACTOR] Input: market_data={market_data.shape}, predictions={regime_predictions.shape}", color="cyan")
        try:
            tprint(f"🔍 [REGIME_EXTRACTOR] Extracting data for regime {regime_id}", color="blue")
            
            # Get regime mask
            regime_mask = regime_predictions == regime_id
            regime_data = market_data[regime_mask]
            tprint(f"📊 [REGIME_EXTRACTOR] Found {len(regime_data)} samples for regime {regime_id}", color="cyan")
            
            if len(regime_data) < self.config.min_regime_samples:
                tprint(f"⚠️ [REGIME_EXTRACTOR] Regime {regime_id} has insufficient samples: {len(regime_data)} < {self.config.min_regime_samples}", color="yellow")
                self.logger.warning(f"Regime {regime_id} has insufficient samples: {len(regime_data)}")
                return np.array([]), np.array([]), np.array([])
            
            # Limit data size if too large
            if len(regime_data) > self.config.max_regime_samples:
                tprint(f"📏 [REGIME_EXTRACTOR] Limiting regime {regime_id} data from {len(regime_data)} to {self.config.max_regime_samples} samples", color="yellow")
                indices = np.random.choice(len(regime_data), self.config.max_regime_samples, replace=False)
                regime_data = regime_data[indices]
            
            # Create labels (simplified - in practice would be more complex)
            tprint(f"🏷️ [REGIME_EXTRACTOR] Creating labels for regime {regime_id}", color="blue")
            regime_labels = self._create_regime_labels(regime_data)
            
            # Extract features
            tprint(f"🔧 [REGIME_EXTRACTOR] Extracting features for regime {regime_id}", color="blue")
            regime_features = self._extract_regime_features(regime_data)
            
            tprint(f"✅ [REGIME_EXTRACTOR] Extracted {len(regime_data)} samples for regime {regime_id}", color="green")
            self.logger.info(f"Extracted {len(regime_data)} samples for regime {regime_id}")
            
            tprint_success(f"✅ [REGIME_EXTRACTOR] extract_regime_data() completed successfully for regime {regime_id}")
            tprint(f"📊 [REGIME_EXTRACTOR] extract_regime_data() outcome: {len(regime_data)} samples, {regime_features.shape[1]} features", color="green")
            return regime_data, regime_labels, regime_features
            
        except Exception as e:
            tprint_error(f"❌ [REGIME_EXTRACTOR] extract_regime_data() failed for regime {regime_id}: {e}")
            self.logger.error(f"Failed to extract data for regime {regime_id}: {e}")
            tprint(f"📊 [REGIME_EXTRACTOR] extract_regime_data() outcome: FAILED", color="red")
            return np.array([]), np.array([]), np.array([])
    
    def _create_regime_labels(self, regime_data: np.ndarray) -> np.ndarray:
        """Create labels for regime data."""
        tprint(f"🏷️ [REGIME_EXTRACTOR] _create_regime_labels() called for {len(regime_data)} samples", color="blue")
        try:
            # Simplified labeling - in practice would be more sophisticated
            # For now, create binary labels based on price movement
            if regime_data.shape[1] > 0:
                price_changes = np.diff(regime_data[:, 0])  # Assuming first column is price
                labels = (price_changes > 0).astype(int)
                # Pad with last label to match data length
                labels = np.append(labels, labels[-1] if len(labels) > 0 else 0)
            else:
                labels = np.zeros(len(regime_data))
            
            tprint(f"📊 [REGIME_EXTRACTOR] _create_regime_labels() outcome: {len(labels)} labels created", color="green")
            return labels
            
        except Exception as e:
            tprint_error(f"❌ [REGIME_EXTRACTOR] _create_regime_labels() failed: {e}")
            self.logger.error(f"Failed to create regime labels: {e}")
            tprint(f"📊 [REGIME_EXTRACTOR] _create_regime_labels() outcome: FAILED", color="red")
            return np.zeros(len(regime_data))
    
    def _extract_regime_features(self, regime_data: np.ndarray) -> np.ndarray:
        """Extract features from regime data."""
        tprint(f"🔧 [REGIME_EXTRACTOR] _extract_regime_features() called for {len(regime_data)} samples", color="blue")
        try:
            if self.config.enable_feature_engineering:
                tprint("🔧 [REGIME_EXTRACTOR] Feature engineering enabled", color="cyan")
                features = self._engineer_features(regime_data)
            else:
                tprint("🔧 [REGIME_EXTRACTOR] Using raw features", color="cyan")
                features = regime_data
            
            if self.config.enable_feature_selection:
                tprint("🔧 [REGIME_EXTRACTOR] Feature selection enabled", color="cyan")
                features = self._select_features(features)
            
            tprint(f"📊 [REGIME_EXTRACTOR] _extract_regime_features() outcome: {features.shape[1]} features", color="green")
            return features
            
        except Exception as e:
            tprint_error(f"❌ [REGIME_EXTRACTOR] _extract_regime_features() failed: {e}")
            self.logger.error(f"Failed to extract regime features: {e}")
            tprint(f"📊 [REGIME_EXTRACTOR] _extract_regime_features() outcome: FAILED", color="red")
            return regime_data
    
    def _engineer_features(self, data: np.ndarray) -> np.ndarray:
        """Engineer additional features."""
        try:
            features = [data]  # Start with original data
            
            # Add technical indicators (simplified)
            if data.shape[1] >= 4:  # OHLCV data
                # Price-based features
                price_features = np.column_stack([
                    data[:, 0],  # Open
                    data[:, 1],  # High
                    data[:, 2],  # Low
                    data[:, 3],  # Close
                    data[:, 0] / data[:, 3],  # Open/Close ratio
                    data[:, 1] / data[:, 2],  # High/Low ratio
                    np.diff(data[:, 0], prepend=data[0, 0]),  # Price change
                ])
                features.append(price_features)
            
            # Add volume features if available
            if data.shape[1] > 4:
                volume_features = np.column_stack([
                    data[:, 4],  # Volume
                    np.diff(data[:, 4], prepend=data[0, 4]),  # Volume change
                ])
                features.append(volume_features)
            
            # Combine all features
            combined_features = np.hstack(features)
            
            return combined_features
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return data
    
    def _select_features(self, features: np.ndarray) -> np.ndarray:
        """Select most important features."""
        try:
            # Simplified feature selection - in practice would use more sophisticated methods
            if features.shape[1] > 20:  # Only select if we have many features
                # Select first 20 features (in practice would use feature importance)
                selected_features = features[:, :20]
            else:
                selected_features = features
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            return features


class RegimeModelTrainer:
    """
    Trains models for specific regimes.
    """
    
    def __init__(self, config: RegimeTrainingConfig):
        """Initialize regime model trainer."""
        tprint("🤖 [REGIME_TRAINER] Initializing Regime Model Trainer", color="blue")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize advanced tree model factory
        tprint("🔧 [REGIME_TRAINER] Setting up advanced tree model factory", color="blue")
        tree_config = AdvancedTreeConfig(
            enable_patchtst_enhancement=config.enable_patchtst_enhancement,
            enable_meta_learning=config.enable_meta_learning,
            enable_continual_learning=config.enable_continual_learning
        )
        self.tree_factory = AdvancedTreeModelFactory(tree_config)
        
        # Training state
        self.trained_models: Dict[Tuple[int, str], Any] = {}
        self.training_history: List[Dict[str, Any]] = []
        
        tprint("✅ [REGIME_TRAINER] Regime Model Trainer initialized", color="green")
        self.logger.info("✅ Regime Model Trainer initialized")
    
    def train_regime_models(self, 
                           regime_id: int,
                           regime_data: np.ndarray,
                           regime_labels: np.ndarray,
                           regime_features: np.ndarray) -> List[RegimeTrainingResult]:
        """
        Train models for a specific regime.
        
        Args:
            regime_id: ID of the regime
            regime_data: Regime data
            regime_labels: Regime labels
            regime_features: Regime features
            
        Returns:
            List of training results for each model
        """
        tprint(f"🤖 [REGIME_TRAINER] train_regime_models() called for regime {regime_id}", color="blue")
        tprint(f"📊 [REGIME_TRAINER] Input: data={regime_data.shape}, labels={regime_labels.shape}, features={regime_features.shape}", color="cyan")
        try:
            tprint(f"🚀 [REGIME_TRAINER] Starting model training for regime {regime_id}", color="cyan", bold=True)
            tprint(f"📊 [REGIME_TRAINER] Training data: {len(regime_data)} samples, {regime_features.shape[1]} features", color="blue")
            
            results = []
            
            # Split data
            tprint(f"✂️ [REGIME_TRAINER] Splitting data for regime {regime_id}", color="blue")
            train_data, test_data, train_labels, test_labels = self._split_data(
                regime_features, regime_labels
            )
            tprint(f"📊 [REGIME_TRAINER] Data split: {len(train_data)} train, {len(test_data)} test", color="cyan")
            
            # Train each model type
            tprint(f"🤖 [REGIME_TRAINER] Training {len(self.config.model_types)} model types for regime {regime_id}", color="yellow")
            for model_type in self.config.model_types:
                tprint(f"🔧 [REGIME_TRAINER] Training {model_type} for regime {regime_id}...", color="blue")
                self.logger.info(f"Training {model_type} for regime {regime_id}...")
                
                result = self._train_single_model(
                    regime_id, model_type, train_data, train_labels, test_data, test_labels
                )
                results.append(result)
                
                if result.success:
                    # Store trained model
                    key = (regime_id, model_type)
                    self.trained_models[key] = result
                    
                    tprint(f"✅ [REGIME_TRAINER] {model_type} trained successfully for regime {regime_id}", color="green")
                    self.logger.info(f"✅ {model_type} trained successfully for regime {regime_id}")
                else:
                    tprint(f"❌ [REGIME_TRAINER] {model_type} training failed for regime {regime_id}: {result.error_message}", color="red")
                    self.logger.error(f"❌ {model_type} training failed for regime {regime_id}: {result.error_message}")
            
            # Train ensemble if enabled
            if self.config.enable_ensemble_training:
                tprint(f"🎯 [REGIME_TRAINER] Training ensemble model for regime {regime_id}", color="magenta")
                ensemble_result = self._train_ensemble_model(
                    regime_id, train_data, train_labels, test_data, test_labels, results
                )
                if ensemble_result:
                    results.append(ensemble_result)
                    tprint(f"✅ [REGIME_TRAINER] Ensemble model trained for regime {regime_id}", color="green")
            
            tprint(f"🎉 [REGIME_TRAINER] Completed training for regime {regime_id}: {len(results)} models", color="green", bold=True)
            tprint_success(f"✅ [REGIME_TRAINER] train_regime_models() completed successfully for regime {regime_id}")
            tprint(f"📊 [REGIME_TRAINER] train_regime_models() outcome: {len(results)} models trained", color="green")
            return results
            
        except Exception as e:
            tprint_error(f"❌ [REGIME_TRAINER] train_regime_models() failed for regime {regime_id}: {e}")
            self.logger.error(f"Failed to train models for regime {regime_id}: {e}")
            tprint(f"📊 [REGIME_TRAINER] train_regime_models() outcome: FAILED", color="red")
            return []
    
    def _split_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/test sets."""
        try:
            from sklearn.model_selection import train_test_split
            
            train_features, test_features, train_labels, test_labels = train_test_split(
                features, labels,
                test_size=1 - self.config.train_test_split,
                random_state=42,
                stratify=labels if len(np.unique(labels)) > 1 else None
            )
            
            return train_features, test_features, train_labels, test_labels
            
        except Exception as e:
            self.logger.error(f"Data splitting failed: {e}")
            # Fallback to simple split
            split_idx = int(len(features) * self.config.train_test_split)
            return features[:split_idx], features[split_idx:], labels[:split_idx], labels[split_idx:]
    
    def _train_single_model(self, 
                           regime_id: int,
                           model_type: str,
                           train_data: np.ndarray,
                           train_labels: np.ndarray,
                           test_data: np.ndarray,
                           test_labels: np.ndarray) -> RegimeTrainingResult:
        """Train a single model for a regime."""
        try:
            start_time = time.time()
            tprint(f"🔧 [MODEL_TRAINER] Training {model_type} for regime {regime_id}", color="blue")
            tprint(f"📊 [MODEL_TRAINER] Training data: {len(train_data)} samples, {train_data.shape[1]} features", color="cyan")
            
            # Create model
            tprint(f"🏗️ [MODEL_TRAINER] Creating {model_type} model", color="yellow")
            model = self.tree_factory.create_model(
                model_type,
                enable_meta_learning=self.config.enable_meta_learning,
                enable_continual_learning=self.config.enable_continual_learning,
                enable_patchtst_enhancement=self.config.enable_patchtst_enhancement
            )
            tprint(f"✅ [MODEL_TRAINER] {model_type} model created", color="green")
            
            # Train model
            tprint(f"🚀 [MODEL_TRAINER] Training {model_type} model", color="blue")
            model.fit(train_data, train_labels)
            tprint(f"✅ [MODEL_TRAINER] {model_type} model training completed", color="green")
            
            # Make predictions
            tprint(f"🔮 [MODEL_TRAINER] Making predictions with {model_type}", color="blue")
            predictions = model.predict(test_data)
            probabilities = model.predict_proba(test_data) if hasattr(model, 'predict_proba') else None
            tprint(f"✅ [MODEL_TRAINER] Predictions generated: {len(predictions)} predictions", color="green")
            
            # Calculate validation scores
            tprint(f"📊 [MODEL_TRAINER] Calculating validation scores for {model_type}", color="blue")
            validation_scores = self._calculate_validation_scores(test_labels, predictions, probabilities)
            tprint(f"📈 [MODEL_TRAINER] Validation scores: {validation_scores}", color="cyan")
            
            # Check if model meets deployment threshold
            deployment_status = "deployed" if validation_scores.get('f1_score', 0) >= self.config.deployment_threshold else "pending"
            tprint(f"🎯 [MODEL_TRAINER] Deployment status: {deployment_status} (F1: {validation_scores.get('f1_score', 0):.3f})", color="green" if deployment_status == "deployed" else "yellow")
            
            training_time = time.time() - start_time
            
            # Create result
            result = RegimeTrainingResult(
                regime_id=regime_id,
                model_name=model_type,
                success=True,
                training_time=training_time,
                validation_scores=validation_scores,
                model_performance={
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'model_type': model_type
                },
                deployment_status=deployment_status
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to train {model_type} for regime {regime_id}: {e}")
            return RegimeTrainingResult(
                regime_id=regime_id,
                model_name=model_type,
                success=False,
                training_time=0.0,
                validation_scores={},
                model_performance={},
                error_message=str(e)
            )
    
    def _train_ensemble_model(self, 
                            regime_id: int,
                            train_data: np.ndarray,
                            train_labels: np.ndarray,
                            test_data: np.ndarray,
                            test_labels: np.ndarray,
                            individual_results: List[RegimeTrainingResult]) -> Optional[RegimeTrainingResult]:
        """Train ensemble model for a regime."""
        try:
            tprint(f"🎯 [ENSEMBLE_TRAINER] Creating ensemble model for regime {regime_id}", color="magenta")
            
            # Get successful models
            successful_models = [r for r in individual_results if r.success]
            tprint(f"📊 [ENSEMBLE_TRAINER] Found {len(successful_models)} successful models out of {len(individual_results)}", color="cyan")
            
            if len(successful_models) < 2:
                tprint(f"⚠️ [ENSEMBLE_TRAINER] Not enough successful models for ensemble in regime {regime_id}", color="yellow")
                self.logger.warning(f"Not enough successful models for ensemble in regime {regime_id}")
                return None
            
            start_time = time.time()
            
            # Create ensemble predictions
            tprint(f"🔮 [ENSEMBLE_TRAINER] Creating ensemble predictions", color="blue")
            ensemble_predictions = []
            ensemble_probabilities = []
            
            for result in successful_models:
                if result.model_performance.get('predictions') is not None:
                    ensemble_predictions.append(result.model_performance['predictions'])
                if result.model_performance.get('probabilities') is not None:
                    ensemble_probabilities.append(result.model_performance['probabilities'])
            
            if not ensemble_predictions:
                tprint(f"❌ [ENSEMBLE_TRAINER] No ensemble predictions available", color="red")
                return None
            
            tprint(f"📊 [ENSEMBLE_TRAINER] Combining {len(ensemble_predictions)} model predictions", color="blue")
            
            # Combine predictions (simple averaging)
            ensemble_pred = np.mean(ensemble_predictions, axis=0)
            ensemble_pred = np.round(ensemble_pred).astype(int)
            
            ensemble_prob = np.mean(ensemble_probabilities, axis=0) if ensemble_probabilities else None
            tprint(f"✅ [ENSEMBLE_TRAINER] Ensemble predictions created: {len(ensemble_pred)} predictions", color="green")
            
            # Calculate validation scores
            tprint(f"📊 [ENSEMBLE_TRAINER] Calculating ensemble validation scores", color="blue")
            validation_scores = self._calculate_validation_scores(test_labels, ensemble_pred, ensemble_prob)
            tprint(f"📈 [ENSEMBLE_TRAINER] Ensemble validation scores: {validation_scores}", color="cyan")
            
            training_time = time.time() - start_time
            
            result = RegimeTrainingResult(
                regime_id=regime_id,
                model_name="ensemble",
                success=True,
                training_time=training_time,
                validation_scores=validation_scores,
                model_performance={
                    'predictions': ensemble_pred,
                    'probabilities': ensemble_prob,
                    'model_type': 'ensemble',
                    'component_models': [r.model_name for r in successful_models]
                },
                deployment_status="deployed" if validation_scores.get('f1_score', 0) >= self.config.deployment_threshold else "pending"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to train ensemble for regime {regime_id}: {e}")
            return None
    
    def _calculate_validation_scores(self, 
                                   true_labels: np.ndarray,
                                   predictions: np.ndarray,
                                   probabilities: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate validation scores."""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            scores = {}
            
            # Basic classification metrics
            scores['accuracy'] = accuracy_score(true_labels, predictions)
            scores['precision'] = precision_score(true_labels, predictions, average='weighted', zero_division=0)
            scores['recall'] = recall_score(true_labels, predictions, average='weighted', zero_division=0)
            scores['f1_score'] = f1_score(true_labels, predictions, average='weighted', zero_division=0)
            
            # ROC AUC if probabilities available
            if probabilities is not None and len(np.unique(true_labels)) > 1:
                try:
                    scores['roc_auc'] = roc_auc_score(true_labels, probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0])
                except:
                    scores['roc_auc'] = 0.5
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Failed to calculate validation scores: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}


class AutomaticRegimeTrainingPipeline:
    """
    Automatic training pipeline for regime-specific models.
    """
    
    def __init__(self, 
                 hybrid_config: HybridRegimeConfig,
                 training_config: RegimeTrainingConfig):
        """Initialize automatic training pipeline."""
        tprint("🚀 [AUTO_PIPELINE] Initializing Automatic Regime Training Pipeline", color="blue")
        self.hybrid_config = hybrid_config
        self.training_config = training_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        tprint("🔧 [AUTO_PIPELINE] Setting up hybrid regime detector", color="blue")
        self.hybrid_detector = HybridNASTASRegimeDetector(hybrid_config)
        
        tprint("🔧 [AUTO_PIPELINE] Setting up regime data extractor", color="blue")
        self.data_extractor = RegimeDataExtractor(training_config)
        
        tprint("🔧 [AUTO_PIPELINE] Setting up regime model trainer", color="blue")
        self.model_trainer = RegimeModelTrainer(training_config)
        
        # Initialize HPO integration
        if self.training_config.enable_hpo_optimization:
            tprint("🔧 [AUTO_PIPELINE] Setting up HPO integration", color="blue")
            hpo_config = create_regime_hpo_integration_config(
                optimization_strategy=self.training_config.hpo_strategy,
                base_model_n_trials=self.training_config.hpo_base_model_trials,
                meta_model_n_trials=self.training_config.hpo_meta_model_trials,
                enable_meta_feature_optimization=self.training_config.enable_meta_feature_optimization
            )
            self.hpo_integration = RegimeHPOIntegration(
                regime_config=hybrid_config,
                hpo_config=hpo_config
            )
        else:
            self.hpo_integration = None
        
        # Training state
        self.training_results: Dict[int, List[RegimeTrainingResult]] = {}
        self.deployed_models: Dict[int, str] = {}  # regime_id -> best_model_name
        
        tprint("✅ [AUTO_PIPELINE] Automatic Regime Training Pipeline initialized", color="green")
        self.logger.info("✅ Automatic Regime Training Pipeline initialized")
    
    def run_automatic_training(self,
                             market_data: Union[pd.DataFrame, np.ndarray],
                             timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run automatic training pipeline.
        
        Args:
            market_data: Market data for training
            timestamps: Optional timestamps
            
        Returns:
            Training pipeline results
        """
        try:
            start_time = time.time()
            tprint("🚀 [AUTO_PIPELINE] Starting automatic regime training pipeline", color="cyan", bold=True)
            tprint(f"📊 [AUTO_PIPELINE] Input data: {len(market_data)} samples", color="blue")
            
            # Step 1: Detect regimes
            tprint("🔍 [AUTO_PIPELINE] Step 1: Detecting regimes...", color="yellow")
            tprint("🔍 [AUTO_PIPELINE] Running hybrid regime detection with economic validation", color="blue")
            self.logger.info("🔍 Detecting regimes...")
            hybrid_result = self.hybrid_detector.detect_regimes(
                market_data=market_data,
                timestamps=timestamps,
                validate_economic_significance=True,
                validate_financial_relevance=True
            )
            
            if not hybrid_result.success:
                tprint(f"❌ [AUTO_PIPELINE] Regime detection failed: {hybrid_result.error_message}", color="red", bold=True)
                raise ValueError(f"Regime detection failed: {hybrid_result.error_message}")
            
            unique_regimes = np.unique(hybrid_result.regime_predictions)
            tprint(f"✅ [AUTO_PIPELINE] Detected {len(unique_regimes)} regimes: {unique_regimes}", color="green")
            
            # Step 2: Optimize hyperparameters (if enabled)
            if self.training_config.enable_hpo_optimization and self.hpo_integration is not None:
                tprint("🎯 [AUTO_PIPELINE] Step 2a: Optimizing hyperparameters...", color="yellow")
                tprint("🎯 [AUTO_PIPELINE] Running HPO optimization for regime models", color="blue")
                self.logger.info("🎯 Optimizing hyperparameters...")
                hpo_results = self.hpo_integration.run_complete_optimization(
                    market_data=market_data,
                    regime_labels=hybrid_result.regime_predictions,
                    save_results=True
                )
                tprint(f"✅ [AUTO_PIPELINE] HPO optimization completed in {hpo_results['total_optimization_time']:.2f}s", color="green")
            
            # Step 2b: Extract data for each regime
            tprint("📊 [AUTO_PIPELINE] Step 2b: Extracting regime data...", color="yellow")
            tprint(f"📊 [AUTO_PIPELINE] Processing {len(unique_regimes)} unique regimes", color="blue")
            self.logger.info("📊 Extracting regime data...")
            regime_data_dict = {}
            
            for regime_id in unique_regimes:
                tprint(f"🔍 [AUTO_PIPELINE] Extracting data for regime {regime_id}", color="blue")
                tprint(f"🔍 [AUTO_PIPELINE] Running regime data extraction with feature engineering", color="cyan")
                regime_data, regime_labels, regime_features = self.data_extractor.extract_regime_data(
                    market_data, hybrid_result.regime_predictions, regime_id
                )
                
                if len(regime_data) > 0:
                    regime_data_dict[regime_id] = {
                        'data': regime_data,
                        'labels': regime_labels,
                        'features': regime_features
                    }
                    tprint(f"✅ [AUTO_PIPELINE] Extracted {len(regime_data)} samples for regime {regime_id}", color="green")
                    self.logger.info(f"Extracted {len(regime_data)} samples for regime {regime_id}")
                else:
                    tprint(f"⚠️ [AUTO_PIPELINE] No data extracted for regime {regime_id}", color="yellow")
                    self.logger.warning(f"No data extracted for regime {regime_id}")
            
            # Step 3: Train models for each regime
            tprint(f"🤖 [AUTO_PIPELINE] Step 3: Training models for {len(regime_data_dict)} regimes...", color="yellow")
            tprint(f"🤖 [AUTO_PIPELINE] Training {len(self.training_config.model_types)} model types per regime", color="blue")
            self.logger.info("🤖 Training models for each regime...")
            all_training_results = {}
            
            for regime_id, regime_info in regime_data_dict.items():
                tprint(f"🚀 [AUTO_PIPELINE] Training models for regime {regime_id}...", color="cyan")
                tprint(f"🚀 [AUTO_PIPELINE] Training with {len(regime_info['data'])} samples and {regime_info['features'].shape[1]} features", color="blue")
                self.logger.info(f"Training models for regime {regime_id}...")
                
                training_results = self.model_trainer.train_regime_models(
                    regime_id=regime_id,
                    regime_data=regime_info['data'],
                    regime_labels=regime_info['labels'],
                    regime_features=regime_info['features']
                )
                
                all_training_results[regime_id] = training_results
                
                # Select best model for deployment
                best_model = self._select_best_model(training_results)
                if best_model:
                    self.deployed_models[regime_id] = best_model.model_name
                    tprint(f"🎯 [AUTO_PIPELINE] Deployed {best_model.model_name} for regime {regime_id}", color="green")
                    self.logger.info(f"Deployed {best_model.model_name} for regime {regime_id}")
            
            execution_time = time.time() - start_time
            
            # Create comprehensive result
            tprint("📊 [AUTO_PIPELINE] Creating comprehensive results", color="blue")
            result = {
                'success': True,
                'execution_time': execution_time,
                'regimes_detected': len(unique_regimes),
                'regimes_trained': len(regime_data_dict),
                'training_results': all_training_results,
                'deployed_models': self.deployed_models,
                'pipeline_summary': self._create_pipeline_summary(all_training_results),
                'metadata': {
                    'system': 'Automatic Regime Training Pipeline',
                    'timestamp': datetime.now().isoformat(),
                    'config': {
                        'model_types': self.training_config.model_types,
                        'enable_meta_learning': self.training_config.enable_meta_learning,
                        'enable_patchtst_enhancement': self.training_config.enable_patchtst_enhancement,
                        'enable_ensemble_training': self.training_config.enable_ensemble_training
                    }
                }
            }
            
            tprint(f"🎉 [AUTO_PIPELINE] SUCCESS: Automatic training pipeline completed in {execution_time:.2f}s", color="green", bold=True)
            tprint(f"📊 [AUTO_PIPELINE] Results: {len(unique_regimes)} regimes detected, {len(regime_data_dict)} trained, {len(self.deployed_models)} deployed", color="cyan")
            self.logger.info(f"✅ Automatic training pipeline completed in {execution_time:.2f}s")
            self.logger.info(f"   Regimes detected: {len(unique_regimes)}")
            self.logger.info(f"   Regimes trained: {len(regime_data_dict)}")
            self.logger.info(f"   Models deployed: {len(self.deployed_models)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Automatic training pipeline failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'execution_time': time.time() - start_time if 'start_time' in locals() else 0.0
            }
    
    def _select_best_model(self, training_results: List[RegimeTrainingResult]) -> Optional[RegimeTrainingResult]:
        """Select the best model from training results."""
        try:
            successful_models = [r for r in training_results if r.success]
            if not successful_models:
                return None
            
            # Select based on F1 score
            best_model = max(successful_models, key=lambda x: x.validation_scores.get('f1_score', 0))
            return best_model
            
        except Exception as e:
            self.logger.error(f"Failed to select best model: {e}")
            return None
    
    def _create_pipeline_summary(self, all_training_results: Dict[int, List[RegimeTrainingResult]]) -> Dict[str, Any]:
        """Create summary of the training pipeline."""
        try:
            summary = {
                'total_regimes': len(all_training_results),
                'total_models_trained': sum(len(results) for results in all_training_results.values()),
                'successful_models': 0,
                'deployed_models': len(self.deployed_models),
                'average_training_time': 0.0,
                'model_performance': {}
            }
            
            total_training_time = 0.0
            successful_count = 0
            
            for regime_id, results in all_training_results.items():
                regime_summary = {
                    'models_trained': len(results),
                    'successful_models': 0,
                    'best_model': None,
                    'best_f1_score': 0.0
                }
                
                for result in results:
                    if result.success:
                        successful_count += 1
                        regime_summary['successful_models'] += 1
                        total_training_time += result.training_time
                        
                        f1_score = result.validation_scores.get('f1_score', 0)
                        if f1_score > regime_summary['best_f1_score']:
                            regime_summary['best_f1_score'] = f1_score
                            regime_summary['best_model'] = result.model_name
                
                summary['model_performance'][regime_id] = regime_summary
            
            summary['successful_models'] = successful_count
            summary['average_training_time'] = total_training_time / successful_count if successful_count > 0 else 0.0
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to create pipeline summary: {e}")
            return {'error': str(e)}
    
    def get_trained_models(self) -> Dict[int, str]:
        """Get deployed models for each regime."""
        return self.deployed_models.copy()
    
    def get_training_results(self, regime_id: int) -> List[RegimeTrainingResult]:
        """Get training results for a specific regime."""
        return self.training_results.get(regime_id, [])
    
    def save_training_results(self, filepath: str):
        """Save training results to file."""
        try:
            data = {
                'training_results': self.training_results,
                'deployed_models': self.deployed_models,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"Training results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save training results: {e}")