"""
Enhanced Tactician Models Training with MSM and Attention Mechanisms

This module provides enhanced tactician model training with:
- MSM-based regime discovery (replacing HMM)
- Attention mechanisms for tree-based models
- Bayesian optimization for all components
- Full pipeline integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger

# Enhanced imports with comprehensive error handling
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

# Import MSM clustering
try:
    from src.training.steps.market_analysis.msm_clustering import (
        MSMOptimizedClusterer, MSMClusteringConfig, MSMClusteringResult
    )
    MSM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ MSM clustering not available: {e}")
    MSM_AVAILABLE = False

# Import attention mechanisms
try:
    from src.training.steps.model_training.attention_mechanisms import (
        CatBoostAttentionWrapper, LightGBMAttentionWrapper, XGBoostAttentionWrapper,
        AttentionConfig
    )
    ATTENTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Attention mechanisms not available: {e}")
    ATTENTION_AVAILABLE = False

# Import Bayesian optimization
try:
    from src.training.steps.model_training.bayesian_optimization import (
        UnifiedBayesianOptimizer, UnifiedOptimizationConfig,
        MSMBayesianOptimizer, AttentionBayesianOptimizer
    )
    BAYESIAN_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Bayesian optimization not available: {e}")
    BAYESIAN_OPTIMIZATION_AVAILABLE = False

# Import existing ML utilities
try:
    from src.utils.ml_common.config import PerRegimeTrainingConfig
    from src.utils.ml_common.training import PerRegimeTrainingStep
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML common utilities not available: {e}")
    ML_COMMON_AVAILABLE = False


class TacticianModelType(Enum):
    """Supported tactician model types."""
    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST_ATTENTION = "catboost_attention"
    LIGHTGBM_ATTENTION = "lightgbm_attention"
    XGBOOST_ATTENTION = "xgboost_attention"


@dataclass
class EnhancedTacticianConfig:
    """Configuration for enhanced tactician training."""
    
    # Model configuration
    model_types: List[TacticianModelType] = None
    use_msm_clustering: bool = True
    use_attention_mechanisms: bool = True
    use_bayesian_optimization: bool = True
    
    # MSM configuration
    msm_config: MSMClusteringConfig = None
    
    # Attention configuration
    attention_config: AttentionConfig = None
    
    # Bayesian optimization configuration
    optimization_config: UnifiedOptimizationConfig = None
    
    # Tactician-specific configuration
    decision_threshold: float = 0.5
    confidence_threshold: float = 0.7
    risk_tolerance: float = 0.1
    
    # Training configuration
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Performance configuration
    n_jobs: int = -1
    verbose: bool = True
    
    def __post_init__(self):
        if self.model_types is None:
            self.model_types = [
                TacticianModelType.CATBOOST_ATTENTION,
                TacticianModelType.LIGHTGBM_ATTENTION,
                TacticianModelType.XGBOOST_ATTENTION
            ]
        
        if self.msm_config is None:
            self.msm_config = MSMClusteringConfig.create_default()
        
        if self.attention_config is None:
            self.attention_config = AttentionConfig()
        
        if self.optimization_config is None:
            self.optimization_config = UnifiedOptimizationConfig()


class EnhancedTacticianTrainer:
    """Enhanced tactician trainer with MSM and attention mechanisms."""
    
    def __init__(self, config: EnhancedTacticianConfig):
        """Initialize enhanced tactician trainer."""
        self.config = config
        self.logger = system_logger.getChild('EnhancedTacticianTrainer')
        
        # Initialize components
        self.msm_clusterer = None
        self.attention_models = {}
        self.bayesian_optimizer = None
        self.training_results = {}
        
        # Initialize MSM clusterer if available
        if MSM_AVAILABLE and self.config.use_msm_clustering:
            self.msm_clusterer = MSMOptimizedClusterer(self.config.msm_config)
            tprint_info("✅ MSM clusterer initialized")
        
        # Initialize Bayesian optimizer if available
        if BAYESIAN_OPTIMIZATION_AVAILABLE and self.config.use_bayesian_optimization:
            self.bayesian_optimizer = UnifiedBayesianOptimizer(self.config.optimization_config)
            tprint_info("✅ Bayesian optimizer initialized")
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train enhanced tactician models."""
        tprint_info("🚀 Starting enhanced tactician training")
        
        try:
            # Step 1: MSM-based regime discovery
            regime_labels = self._discover_regimes(X, y)
            
            # Step 2: Train attention-based models
            attention_models = self._train_attention_models(X, y, regime_labels)
            
            # Step 3: Bayesian optimization
            if self.bayesian_optimizer:
                optimization_results = self._optimize_hyperparameters(X, y, attention_models)
            else:
                optimization_results = {}
            
            # Step 4: Create ensemble
            ensemble_model = self._create_ensemble(attention_models, optimization_results)
            
            # Step 5: Evaluate models
            evaluation_results = self._evaluate_models(attention_models, ensemble_model, X, y)
            
            # Step 6: Train decision logic
            decision_logic = self._train_decision_logic(attention_models, X, y)
            
            # Store results
            self.training_results = {
                'regime_labels': regime_labels,
                'attention_models': attention_models,
                'ensemble_model': ensemble_model,
                'optimization_results': optimization_results,
                'evaluation_results': evaluation_results,
                'decision_logic': decision_logic
            }
            
            tprint_success("✅ Enhanced tactician training completed successfully")
            return self.training_results
            
        except Exception as e:
            tprint_error(f"❌ Enhanced tactician training failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _discover_regimes(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Discover regimes using MSM clustering."""
        if not MSM_AVAILABLE or not self.msm_clusterer:
            tprint_warning("⚠️ MSM clustering not available, using simple clustering")
            return self._simple_regime_discovery(X, y)
        
        try:
            tprint_info("🔍 Discovering regimes using MSM clustering")
            
            # Fit MSM clusterer
            msm_result = self.msm_clusterer.fit(X)
            
            tprint_info(f"✅ Discovered {msm_result.n_regimes} regimes")
            tprint_info(f"📊 Regime statistics: {msm_result.regime_statistics}")
            
            return msm_result.regime_labels
            
        except Exception as e:
            tprint_warning(f"⚠️ MSM clustering failed: {e}, falling back to simple clustering")
            return self._simple_regime_discovery(X, y)
    
    def _simple_regime_discovery(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Simple regime discovery fallback."""
        try:
            from sklearn.cluster import KMeans
            
            # Simple K-means clustering
            n_regimes = min(3, len(X) // 100)  # Adaptive number of regimes
            kmeans = KMeans(n_clusters=n_regimes, random_state=42)
            regime_labels = kmeans.fit_predict(X)
            
            tprint_info(f"✅ Simple regime discovery completed with {n_regimes} regimes")
            return regime_labels
            
        except Exception as e:
            tprint_warning(f"⚠️ Simple regime discovery failed: {e}")
            # Return uniform regime labels
            return np.zeros(len(X), dtype=int)
    
    def _train_attention_models(self, X: np.ndarray, y: np.ndarray, 
                               regime_labels: np.ndarray) -> Dict[str, Any]:
        """Train attention-based models."""
        if not ATTENTION_AVAILABLE:
            tprint_warning("⚠️ Attention mechanisms not available, using standard models")
            return self._train_standard_models(X, y, regime_labels)
        
        attention_models = {}
        
        for model_type in self.config.model_types:
            try:
                tprint_info(f"🤖 Training {model_type.value} model")
                
                # Create attention model
                if model_type == TacticianModelType.CATBOOST_ATTENTION:
                    model = CatBoostAttentionWrapper(self.config.attention_config, 'regression')
                elif model_type == TacticianModelType.LIGHTGBM_ATTENTION:
                    model = LightGBMAttentionWrapper(self.config.attention_config, 'regression')
                elif model_type == TacticianModelType.XGBOOST_ATTENTION:
                    model = XGBoostAttentionWrapper(self.config.attention_config, 'regression')
                else:
                    tprint_warning(f"⚠️ Unsupported model type: {model_type}")
                    continue
                
                # Train model
                model.fit(X, y)
                
                # Store model
                attention_models[model_type.value] = model
                
                tprint_success(f"✅ {model_type.value} model trained successfully")
                
            except Exception as e:
                tprint_error(f"❌ Failed to train {model_type.value}: {e}")
                continue
        
        return attention_models
    
    def _train_standard_models(self, X: np.ndarray, y: np.ndarray, 
                              regime_labels: np.ndarray) -> Dict[str, Any]:
        """Train standard models without attention."""
        tprint_warning("⚠️ Training standard models without attention mechanisms")
        
        # This would implement standard model training
        # For now, return empty dict
        return {}
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                 attention_models: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hyperparameters using Bayesian optimization."""
        if not self.bayesian_optimizer:
            tprint_warning("⚠️ Bayesian optimizer not available")
            return {}
        
        try:
            tprint_info("🔧 Starting Bayesian hyperparameter optimization")
            
            # Run optimization
            optimization_results = self.bayesian_optimizer.optimize(X, y)
            
            tprint_success(f"✅ Optimization completed with score: {optimization_results['best_score']:.4f}")
            
            return optimization_results
            
        except Exception as e:
            tprint_error(f"❌ Bayesian optimization failed: {e}")
            return {}
    
    def _create_ensemble(self, attention_models: Dict[str, Any], 
                        optimization_results: Dict[str, Any]) -> Any:
        """Create ensemble of attention models."""
        if not attention_models:
            tprint_warning("⚠️ No attention models available for ensemble")
            return None
        
        try:
            tprint_info("🎯 Creating ensemble of attention models")
            
            # Simple ensemble implementation
            # In practice, you would implement a more sophisticated ensemble
            
            ensemble_model = {
                'models': attention_models,
                'weights': self._calculate_ensemble_weights(attention_models),
                'optimization_results': optimization_results
            }
            
            tprint_success("✅ Ensemble created successfully")
            return ensemble_model
            
        except Exception as e:
            tprint_error(f"❌ Ensemble creation failed: {e}")
            return None
    
    def _calculate_ensemble_weights(self, attention_models: Dict[str, Any]) -> Dict[str, float]:
        """Calculate ensemble weights for models."""
        # Simple equal weighting
        n_models = len(attention_models)
        weight = 1.0 / n_models if n_models > 0 else 0.0
        
        return {model_name: weight for model_name in attention_models.keys()}
    
    def _train_decision_logic(self, attention_models: Dict[str, Any], 
                              X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train decision logic for tactician."""
        try:
            tprint_info("🎯 Training decision logic")
            
            # Get predictions from all models
            model_predictions = {}
            for model_name, model in attention_models.items():
                try:
                    predictions = model.predict(X)
                    model_predictions[model_name] = predictions
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to get predictions from {model_name}: {e}")
                    continue
            
            if not model_predictions:
                tprint_warning("⚠️ No model predictions available for decision logic")
                return {}
            
            # Calculate decision thresholds
            decision_thresholds = {}
            for model_name, predictions in model_predictions.items():
                # Calculate threshold based on prediction distribution
                threshold = np.percentile(predictions, 50)  # Median as threshold
                decision_thresholds[model_name] = threshold
            
            # Calculate confidence thresholds
            confidence_thresholds = {}
            for model_name, predictions in model_predictions.items():
                # Calculate confidence based on prediction variance
                confidence = 1.0 / (1.0 + np.std(predictions))
                confidence_thresholds[model_name] = confidence
            
            decision_logic = {
                'decision_thresholds': decision_thresholds,
                'confidence_thresholds': confidence_thresholds,
                'model_predictions': model_predictions
            }
            
            tprint_success("✅ Decision logic trained successfully")
            return decision_logic
            
        except Exception as e:
            tprint_error(f"❌ Decision logic training failed: {e}")
            return {}
    
    def _evaluate_models(self, attention_models: Dict[str, Any], 
                        ensemble_model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate all models."""
        evaluation_results = {}
        
        try:
            tprint_info("📊 Evaluating models")
            
            # Evaluate individual attention models
            for model_name, model in attention_models.items():
                try:
                    # Get predictions
                    predictions = model.predict(X)
                    
                    # Calculate metrics
                    mse = np.mean((predictions - y) ** 2)
                    mae = np.mean(np.abs(predictions - y))
                    r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
                    
                    # Calculate tactician-specific metrics
                    decision_accuracy = self._calculate_decision_accuracy(predictions, y)
                    risk_metrics = self._calculate_risk_metrics(predictions, y)
                    
                    evaluation_results[model_name] = {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'decision_accuracy': decision_accuracy,
                        'risk_metrics': risk_metrics,
                        'predictions': predictions
                    }
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to evaluate {model_name}: {e}")
                    continue
            
            # Evaluate ensemble if available
            if ensemble_model:
                try:
                    # Simple ensemble prediction
                    ensemble_predictions = self._ensemble_predict(ensemble_model, X)
                    
                    # Calculate ensemble metrics
                    mse = np.mean((ensemble_predictions - y) ** 2)
                    mae = np.mean(np.abs(ensemble_predictions - y))
                    r2 = 1 - (np.sum((y - ensemble_predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
                    
                    # Calculate tactician-specific metrics
                    decision_accuracy = self._calculate_decision_accuracy(ensemble_predictions, y)
                    risk_metrics = self._calculate_risk_metrics(ensemble_predictions, y)
                    
                    evaluation_results['ensemble'] = {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'decision_accuracy': decision_accuracy,
                        'risk_metrics': risk_metrics,
                        'predictions': ensemble_predictions
                    }
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to evaluate ensemble: {e}")
            
            tprint_success("✅ Model evaluation completed")
            return evaluation_results
            
        except Exception as e:
            tprint_error(f"❌ Model evaluation failed: {e}")
            return {}
    
    def _calculate_decision_accuracy(self, predictions: np.ndarray, y: np.ndarray) -> float:
        """Calculate decision accuracy for tactician."""
        try:
            # Convert predictions to binary decisions
            decisions = (predictions > self.config.decision_threshold).astype(int)
            actual_decisions = (y > self.config.decision_threshold).astype(int)
            
            # Calculate accuracy
            accuracy = np.mean(decisions == actual_decisions)
            return accuracy
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to calculate decision accuracy: {e}")
            return 0.0
    
    def _calculate_risk_metrics(self, predictions: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate risk metrics for tactician."""
        try:
            # Calculate prediction errors
            errors = np.abs(predictions - y)
            
            # Calculate risk metrics
            max_error = np.max(errors)
            mean_error = np.mean(errors)
            error_std = np.std(errors)
            
            # Calculate risk-adjusted metrics
            risk_adjusted_score = np.mean(predictions) / (1.0 + error_std)
            
            return {
                'max_error': max_error,
                'mean_error': mean_error,
                'error_std': error_std,
                'risk_adjusted_score': risk_adjusted_score
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to calculate risk metrics: {e}")
            return {}
    
    def _ensemble_predict(self, ensemble_model: Any, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not ensemble_model or 'models' not in ensemble_model:
            return np.zeros(len(X))
        
        models = ensemble_model['models']
        weights = ensemble_model.get('weights', {})
        
        # Get predictions from all models
        predictions = []
        for model_name, model in models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except:
                continue
        
        if not predictions:
            return np.zeros(len(X))
        
        # Weighted average of predictions
        predictions = np.array(predictions)
        weights_array = np.array([weights.get(name, 1.0) for name in models.keys()])
        weights_array = weights_array / np.sum(weights_array)
        
        ensemble_predictions = np.average(predictions, axis=0, weights=weights_array)
        
        return ensemble_predictions
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Make predictions using the trained ensemble."""
        if not self.training_results or 'ensemble_model' not in self.training_results:
            raise ValueError("Model must be trained before prediction")
        
        ensemble_model = self.training_results['ensemble_model']
        decision_logic = self.training_results.get('decision_logic', {})
        
        # Get ensemble predictions
        predictions = self._ensemble_predict(ensemble_model, X)
        
        # Apply decision logic
        decisions = self._apply_decision_logic(predictions, decision_logic)
        
        return {
            'predictions': predictions,
            'decisions': decisions,
            'confidence': self._calculate_confidence(predictions, decision_logic)
        }
    
    def _apply_decision_logic(self, predictions: np.ndarray, decision_logic: Dict[str, Any]) -> np.ndarray:
        """Apply decision logic to predictions."""
        try:
            # Get decision thresholds
            decision_thresholds = decision_logic.get('decision_thresholds', {})
            
            if not decision_thresholds:
                # Use default threshold
                threshold = self.config.decision_threshold
            else:
                # Use average threshold
                threshold = np.mean(list(decision_thresholds.values()))
            
            # Make decisions
            decisions = (predictions > threshold).astype(int)
            
            return decisions
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to apply decision logic: {e}")
            return np.zeros(len(predictions), dtype=int)
    
    def _calculate_confidence(self, predictions: np.ndarray, decision_logic: Dict[str, Any]) -> np.ndarray:
        """Calculate confidence scores for predictions."""
        try:
            # Get confidence thresholds
            confidence_thresholds = decision_logic.get('confidence_thresholds', {})
            
            if not confidence_thresholds:
                # Use default confidence
                confidence = np.ones(len(predictions)) * self.config.confidence_threshold
            else:
                # Use average confidence
                avg_confidence = np.mean(list(confidence_thresholds.values()))
                confidence = np.ones(len(predictions)) * avg_confidence
            
            return confidence
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to calculate confidence: {e}")
            return np.ones(len(predictions)) * self.config.confidence_threshold
    
    def get_feature_importance(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get feature importance from all models."""
        if not self.training_results or 'attention_models' not in self.training_results:
            raise ValueError("Model must be trained before getting feature importance")
        
        feature_importance = {}
        
        for model_name, model in self.training_results['attention_models'].items():
            try:
                importance = model.get_feature_importance(X)
                feature_importance[model_name] = importance
            except Exception as e:
                tprint_warning(f"⚠️ Failed to get feature importance for {model_name}: {e}")
                continue
        
        return feature_importance
    
    def save_models(self, filepath: str) -> None:
        """Save all trained models."""
        if not self.training_results:
            tprint_warning("⚠️ No training results to save")
            return
        
        try:
            import joblib
            
            # Prepare data for saving
            save_data = {
                'config': self.config,
                'training_results': self.training_results,
                'msm_clusterer': self.msm_clusterer,
                'bayesian_optimizer': self.bayesian_optimizer
            }
            
            joblib.dump(save_data, filepath)
            tprint_success(f"✅ Models saved to {filepath}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to save models: {e}")
    
    def load_models(self, filepath: str) -> None:
        """Load trained models."""
        try:
            import joblib
            
            save_data = joblib.load(filepath)
            
            self.config = save_data['config']
            self.training_results = save_data['training_results']
            self.msm_clusterer = save_data.get('msm_clusterer')
            self.bayesian_optimizer = save_data.get('bayesian_optimizer')
            
            tprint_success(f"✅ Models loaded from {filepath}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to load models: {e}")
            raise