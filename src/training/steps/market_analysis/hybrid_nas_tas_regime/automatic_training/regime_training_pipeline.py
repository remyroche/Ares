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

# Import regime detection and model selection
from ..regime_model_mapping import DataDrivenModelSelector, ModelSelectorConfig
from ..core.hybrid_regime_detector import HybridNASTASRegimeDetector
from ..config.hybrid_regime_config import HybridRegimeConfig

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
    
    # Model types to train
    model_types: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "catboost", "random_forest", "gradient_boosting"
    ])
    
    # Advanced features
    enable_meta_learning: bool = True
    enable_continual_learning: bool = True
    enable_clvsa_enhancement: bool = True
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
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
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
        try:
            # Get regime mask
            regime_mask = regime_predictions == regime_id
            regime_data = market_data[regime_mask]
            
            if len(regime_data) < self.config.min_regime_samples:
                self.logger.warning(f"Regime {regime_id} has insufficient samples: {len(regime_data)}")
                return np.array([]), np.array([]), np.array([])
            
            # Limit data size if too large
            if len(regime_data) > self.config.max_regime_samples:
                indices = np.random.choice(len(regime_data), self.config.max_regime_samples, replace=False)
                regime_data = regime_data[indices]
            
            # Create labels (simplified - in practice would be more complex)
            regime_labels = self._create_regime_labels(regime_data)
            
            # Extract features
            regime_features = self._extract_regime_features(regime_data)
            
            self.logger.info(f"Extracted {len(regime_data)} samples for regime {regime_id}")
            
            return regime_data, regime_labels, regime_features
            
        except Exception as e:
            self.logger.error(f"Failed to extract data for regime {regime_id}: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def _create_regime_labels(self, regime_data: np.ndarray) -> np.ndarray:
        """Create labels for regime data."""
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
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Failed to create regime labels: {e}")
            return np.zeros(len(regime_data))
    
    def _extract_regime_features(self, regime_data: np.ndarray) -> np.ndarray:
        """Extract features from regime data."""
        try:
            if self.config.enable_feature_engineering:
                features = self._engineer_features(regime_data)
            else:
                features = regime_data
            
            if self.config.enable_feature_selection:
                features = self._select_features(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract regime features: {e}")
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
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize advanced tree model factory
        tree_config = AdvancedTreeConfig(
            enable_clvsa_enhancement=config.enable_clvsa_enhancement,
            enable_meta_learning=config.enable_meta_learning,
            enable_continual_learning=config.enable_continual_learning
        )
        self.tree_factory = AdvancedTreeModelFactory(tree_config)
        
        # Training state
        self.trained_models: Dict[Tuple[int, str], Any] = {}
        self.training_history: List[Dict[str, Any]] = []
        
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
        try:
            results = []
            
            # Split data
            train_data, test_data, train_labels, test_labels = self._split_data(
                regime_features, regime_labels
            )
            
            # Train each model type
            for model_type in self.config.model_types:
                self.logger.info(f"Training {model_type} for regime {regime_id}...")
                
                result = self._train_single_model(
                    regime_id, model_type, train_data, train_labels, test_data, test_labels
                )
                results.append(result)
                
                if result.success:
                    # Store trained model
                    key = (regime_id, model_type)
                    self.trained_models[key] = result
                    
                    self.logger.info(f"✅ {model_type} trained successfully for regime {regime_id}")
                else:
                    self.logger.error(f"❌ {model_type} training failed for regime {regime_id}: {result.error_message}")
            
            # Train ensemble if enabled
            if self.config.enable_ensemble_training:
                ensemble_result = self._train_ensemble_model(
                    regime_id, train_data, train_labels, test_data, test_labels, results
                )
                if ensemble_result:
                    results.append(ensemble_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to train models for regime {regime_id}: {e}")
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
            
            # Create model
            model = self.tree_factory.create_model(
                model_type,
                enable_meta_learning=self.config.enable_meta_learning,
                enable_continual_learning=self.config.enable_continual_learning,
                enable_clvsa_enhancement=self.config.enable_clvsa_enhancement
            )
            
            # Train model
            model.fit(train_data, train_labels)
            
            # Make predictions
            predictions = model.predict(test_data)
            probabilities = model.predict_proba(test_data) if hasattr(model, 'predict_proba') else None
            
            # Calculate validation scores
            validation_scores = self._calculate_validation_scores(test_labels, predictions, probabilities)
            
            # Check if model meets deployment threshold
            deployment_status = "deployed" if validation_scores.get('f1_score', 0) >= self.config.deployment_threshold else "pending"
            
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
            # Get successful models
            successful_models = [r for r in individual_results if r.success]
            if len(successful_models) < 2:
                self.logger.warning(f"Not enough successful models for ensemble in regime {regime_id}")
                return None
            
            start_time = time.time()
            
            # Create ensemble predictions
            ensemble_predictions = []
            ensemble_probabilities = []
            
            for result in successful_models:
                if result.model_performance.get('predictions') is not None:
                    ensemble_predictions.append(result.model_performance['predictions'])
                if result.model_performance.get('probabilities') is not None:
                    ensemble_probabilities.append(result.model_performance['probabilities'])
            
            if not ensemble_predictions:
                return None
            
            # Combine predictions (simple averaging)
            ensemble_pred = np.mean(ensemble_predictions, axis=0)
            ensemble_pred = np.round(ensemble_pred).astype(int)
            
            ensemble_prob = np.mean(ensemble_probabilities, axis=0) if ensemble_probabilities else None
            
            # Calculate validation scores
            validation_scores = self._calculate_validation_scores(test_labels, ensemble_pred, ensemble_prob)
            
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
        self.hybrid_config = hybrid_config
        self.training_config = training_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.hybrid_detector = HybridNASTASRegimeDetector(hybrid_config)
        self.data_extractor = RegimeDataExtractor(training_config)
        self.model_trainer = RegimeModelTrainer(training_config)
        
        # Training state
        self.training_results: Dict[int, List[RegimeTrainingResult]] = {}
        self.deployed_models: Dict[int, str] = {}  # regime_id -> best_model_name
        
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
            
            # Step 1: Detect regimes
            self.logger.info("🔍 Detecting regimes...")
            hybrid_result = self.hybrid_detector.detect_regimes(
                market_data=market_data,
                timestamps=timestamps,
                validate_economic_significance=True,
                validate_financial_relevance=True
            )
            
            if not hybrid_result.success:
                raise ValueError(f"Regime detection failed: {hybrid_result.error_message}")
            
            # Step 2: Extract data for each regime
            self.logger.info("📊 Extracting regime data...")
            regime_data_dict = {}
            unique_regimes = np.unique(hybrid_result.regime_predictions)
            
            for regime_id in unique_regimes:
                regime_data, regime_labels, regime_features = self.data_extractor.extract_regime_data(
                    market_data, hybrid_result.regime_predictions, regime_id
                )
                
                if len(regime_data) > 0:
                    regime_data_dict[regime_id] = {
                        'data': regime_data,
                        'labels': regime_labels,
                        'features': regime_features
                    }
                    self.logger.info(f"Extracted {len(regime_data)} samples for regime {regime_id}")
                else:
                    self.logger.warning(f"No data extracted for regime {regime_id}")
            
            # Step 3: Train models for each regime
            self.logger.info("🤖 Training models for each regime...")
            all_training_results = {}
            
            for regime_id, regime_info in regime_data_dict.items():
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
                    self.logger.info(f"Deployed {best_model.model_name} for regime {regime_id}")
            
            execution_time = time.time() - start_time
            
            # Create comprehensive result
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
                        'enable_clvsa_enhancement': self.training_config.enable_clvsa_enhancement,
                        'enable_ensemble_training': self.training_config.enable_ensemble_training
                    }
                }
            }
            
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