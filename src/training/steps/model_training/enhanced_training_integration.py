"""
Enhanced Training Integration

This module provides integration between the new MSM clustering, attention mechanisms,
and Bayesian optimization with the existing training pipeline.
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

# Enhanced imports
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

# Import enhanced components
try:
    from .enhanced_analyst_training import EnhancedAnalystTrainer, EnhancedAnalystConfig
    ENHANCED_ANALYST_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced analyst training not available: {e}")
    ENHANCED_ANALYST_AVAILABLE = False

try:
    from .enhanced_tactician_training import EnhancedTacticianTrainer, EnhancedTacticianConfig
    ENHANCED_TACTICIAN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced tactician training not available: {e}")
    ENHANCED_TACTICIAN_AVAILABLE = False

# Import MSM clustering
try:
    from src.training.steps.market_analysis.msm_clustering import (
        MSMOptimizedClusterer, MSMClusteringConfig
    )
    MSM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ MSM clustering not available: {e}")
    MSM_AVAILABLE = False

# Import attention mechanisms
try:
    from .attention_mechanisms import (
        CatBoostAttentionWrapper, LightGBMAttentionWrapper, XGBoostAttentionWrapper
    )
    ATTENTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Attention mechanisms not available: {e}")
    ATTENTION_AVAILABLE = False

# Import Bayesian optimization
try:
    from .bayesian_optimization import (
        UnifiedBayesianOptimizer, UnifiedOptimizationConfig
    )
    BAYESIAN_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Bayesian optimization not available: {e}")
    BAYESIAN_OPTIMIZATION_AVAILABLE = False


class TrainingMode(Enum):
    """Training modes for enhanced training."""
    ANALYST_ONLY = "analyst_only"
    TACTICIAN_ONLY = "tactician_only"
    BOTH = "both"
    ENSEMBLE = "ensemble"


@dataclass
class EnhancedTrainingConfig:
    """Configuration for enhanced training integration."""
    
    # Training mode
    training_mode: TrainingMode = TrainingMode.BOTH
    
    # Component availability
    use_msm_clustering: bool = True
    use_attention_mechanisms: bool = True
    use_bayesian_optimization: bool = True
    
    # Analyst configuration
    analyst_config: EnhancedAnalystConfig = None
    
    # Tactician configuration
    tactician_config: EnhancedTacticianConfig = None
    
    # MSM configuration
    msm_config: MSMClusteringConfig = None
    
    # Bayesian optimization configuration
    optimization_config: UnifiedOptimizationConfig = None
    
    # Training parameters
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Performance parameters
    n_jobs: int = -1
    verbose: bool = True
    
    def __post_init__(self):
        if self.analyst_config is None:
            self.analyst_config = EnhancedAnalystConfig()
        
        if self.tactician_config is None:
            self.tactician_config = EnhancedTacticianConfig()
        
        if self.msm_config is None:
            self.msm_config = MSMClusteringConfig.create_default()
        
        if self.optimization_config is None:
            self.optimization_config = UnifiedOptimizationConfig()


class EnhancedTrainingIntegration:
    """Enhanced training integration coordinator."""
    
    def __init__(self, config: EnhancedTrainingConfig):
        """Initialize enhanced training integration."""
        self.config = config
        self.logger = system_logger.getChild('EnhancedTrainingIntegration')
        
        # Initialize components
        self.analyst_trainer = None
        self.tactician_trainer = None
        self.msm_clusterer = None
        self.bayesian_optimizer = None
        
        # Initialize analyst trainer
        if ENHANCED_ANALYST_AVAILABLE and self.config.training_mode in [TrainingMode.ANALYST_ONLY, TrainingMode.BOTH, TrainingMode.ENSEMBLE]:
            self.analyst_trainer = EnhancedAnalystTrainer(self.config.analyst_config)
            tprint_info("✅ Enhanced analyst trainer initialized")
        
        # Initialize tactician trainer
        if ENHANCED_TACTICIAN_AVAILABLE and self.config.training_mode in [TrainingMode.TACTICIAN_ONLY, TrainingMode.BOTH, TrainingMode.ENSEMBLE]:
            self.tactician_trainer = EnhancedTacticianTrainer(self.config.tactician_config)
            tprint_info("✅ Enhanced tactician trainer initialized")
        
        # Initialize MSM clusterer
        if MSM_AVAILABLE and self.config.use_msm_clustering:
            self.msm_clusterer = MSMOptimizedClusterer(self.config.msm_config)
            tprint_info("✅ MSM clusterer initialized")
        
        # Initialize Bayesian optimizer
        if BAYESIAN_OPTIMIZATION_AVAILABLE and self.config.use_bayesian_optimization:
            self.bayesian_optimizer = UnifiedBayesianOptimizer(self.config.optimization_config)
            tprint_info("✅ Bayesian optimizer initialized")
    
    def train_analyst(self, X: np.ndarray, y: np.ndarray, 
                      feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train enhanced analyst models."""
        if not self.analyst_trainer:
            raise ValueError("Analyst trainer not available")
        
        tprint_info("🚀 Starting enhanced analyst training")
        
        try:
            # Train analyst models
            analyst_results = self.analyst_trainer.train(X, y, feature_names)
            
            tprint_success("✅ Enhanced analyst training completed")
            return analyst_results
            
        except Exception as e:
            tprint_error(f"❌ Enhanced analyst training failed: {e}")
            raise
    
    def train_tactician(self, X: np.ndarray, y: np.ndarray, 
                        feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train enhanced tactician models."""
        if not self.tactician_trainer:
            raise ValueError("Tactician trainer not available")
        
        tprint_info("🚀 Starting enhanced tactician training")
        
        try:
            # Train tactician models
            tactician_results = self.tactician_trainer.train(X, y, feature_names)
            
            tprint_success("✅ Enhanced tactician training completed")
            return tactician_results
            
        except Exception as e:
            tprint_error(f"❌ Enhanced tactician training failed: {e}")
            raise
    
    def train_both(self, X: np.ndarray, y: np.ndarray, 
                   feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train both analyst and tactician models."""
        tprint_info("🚀 Starting enhanced training for both analyst and tactician")
        
        try:
            results = {}
            
            # Train analyst models
            if self.analyst_trainer:
                tprint_info("📊 Training analyst models...")
                analyst_results = self.analyst_trainer.train(X, y, feature_names)
                results['analyst'] = analyst_results
                tprint_success("✅ Analyst training completed")
            
            # Train tactician models
            if self.tactician_trainer:
                tprint_info("🎯 Training tactician models...")
                tactician_results = self.tactician_trainer.train(X, y, feature_names)
                results['tactician'] = tactician_results
                tprint_success("✅ Tactician training completed")
            
            # Create ensemble if both are available
            if self.analyst_trainer and self.tactician_trainer:
                tprint_info("🎯 Creating ensemble of analyst and tactician models...")
                ensemble_results = self._create_ensemble(results)
                results['ensemble'] = ensemble_results
                tprint_success("✅ Ensemble created")
            
            tprint_success("✅ Enhanced training for both models completed")
            return results
            
        except Exception as e:
            tprint_error(f"❌ Enhanced training failed: {e}")
            raise
    
    def _create_ensemble(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble of analyst and tactician models."""
        try:
            ensemble_results = {
                'analyst_models': results.get('analyst', {}).get('attention_models', {}),
                'tactician_models': results.get('tactician', {}).get('attention_models', {}),
                'analyst_ensemble': results.get('analyst', {}).get('ensemble_model'),
                'tactician_ensemble': results.get('tactician', {}).get('ensemble_model'),
                'ensemble_weights': self._calculate_ensemble_weights(results)
            }
            
            return ensemble_results
            
        except Exception as e:
            tprint_error(f"❌ Ensemble creation failed: {e}")
            return {}
    
    def _calculate_ensemble_weights(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate ensemble weights for analyst and tactician models."""
        try:
            # Simple equal weighting for now
            # In practice, you would calculate weights based on performance
            weights = {
                'analyst': 0.5,
                'tactician': 0.5
            }
            
            return weights
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to calculate ensemble weights: {e}")
            return {'analyst': 0.5, 'tactician': 0.5}
    
    def predict(self, X: np.ndarray, model_type: str = 'ensemble') -> Dict[str, Any]:
        """Make predictions using the trained models."""
        try:
            predictions = {}
            
            if model_type == 'analyst' and self.analyst_trainer:
                analyst_predictions = self.analyst_trainer.predict(X)
                predictions['analyst'] = analyst_predictions
            
            elif model_type == 'tactician' and self.tactician_trainer:
                tactician_predictions = self.tactician_trainer.predict(X)
                predictions['tactician'] = tactician_predictions
            
            elif model_type == 'ensemble':
                # Get predictions from both models
                if self.analyst_trainer:
                    analyst_predictions = self.analyst_trainer.predict(X)
                    predictions['analyst'] = analyst_predictions
                
                if self.tactician_trainer:
                    tactician_predictions = self.tactician_trainer.predict(X)
                    predictions['tactician'] = tactician_predictions
                
                # Create ensemble prediction
                if 'analyst' in predictions and 'tactician' in predictions:
                    ensemble_prediction = self._ensemble_predict(predictions)
                    predictions['ensemble'] = ensemble_prediction
            
            return predictions
            
        except Exception as e:
            tprint_error(f"❌ Prediction failed: {e}")
            return {}
    
    def _ensemble_predict(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble prediction from analyst and tactician predictions."""
        try:
            analyst_pred = predictions.get('analyst', np.zeros(len(predictions.get('tactician', [0]))))
            tactician_pred = predictions.get('tactician', np.zeros(len(predictions.get('analyst', [0]))))
            
            # Simple weighted average
            ensemble_prediction = 0.5 * analyst_pred + 0.5 * tactician_pred
            
            return {
                'prediction': ensemble_prediction,
                'analyst_weight': 0.5,
                'tactician_weight': 0.5
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Ensemble prediction failed: {e}")
            return {}
    
    def get_feature_importance(self, X: np.ndarray, model_type: str = 'ensemble') -> Dict[str, Any]:
        """Get feature importance from trained models."""
        try:
            feature_importance = {}
            
            if model_type == 'analyst' and self.analyst_trainer:
                analyst_importance = self.analyst_trainer.get_feature_importance(X)
                feature_importance['analyst'] = analyst_importance
            
            elif model_type == 'tactician' and self.tactician_trainer:
                tactician_importance = self.tactician_trainer.get_feature_importance(X)
                feature_importance['tactician'] = tactician_importance
            
            elif model_type == 'ensemble':
                # Get importance from both models
                if self.analyst_trainer:
                    analyst_importance = self.analyst_trainer.get_feature_importance(X)
                    feature_importance['analyst'] = analyst_importance
                
                if self.tactician_trainer:
                    tactician_importance = self.tactician_trainer.get_feature_importance(X)
                    feature_importance['tactician'] = tactician_importance
                
                # Combine importance scores
                if 'analyst' in feature_importance and 'tactician' in feature_importance:
                    combined_importance = self._combine_feature_importance(feature_importance)
                    feature_importance['ensemble'] = combined_importance
            
            return feature_importance
            
        except Exception as e:
            tprint_error(f"❌ Feature importance extraction failed: {e}")
            return {}
    
    def _combine_feature_importance(self, feature_importance: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Combine feature importance from analyst and tactician models."""
        try:
            analyst_importance = feature_importance.get('analyst', {})
            tactician_importance = feature_importance.get('tactician', {})
            
            combined_importance = {}
            
            # Combine importance from all models
            for model_name, importance in analyst_importance.items():
                if model_name in tactician_importance:
                    # Average importance from both models
                    combined_importance[model_name] = 0.5 * importance + 0.5 * tactician_importance[model_name]
                else:
                    combined_importance[model_name] = importance
            
            return combined_importance
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature importance combination failed: {e}")
            return {}
    
    def save_models(self, filepath: str) -> None:
        """Save all trained models."""
        try:
            import joblib
            
            # Prepare data for saving
            save_data = {
                'config': self.config,
                'analyst_trainer': self.analyst_trainer,
                'tactician_trainer': self.tactician_trainer,
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
            self.analyst_trainer = save_data.get('analyst_trainer')
            self.tactician_trainer = save_data.get('tactician_trainer')
            self.msm_clusterer = save_data.get('msm_clusterer')
            self.bayesian_optimizer = save_data.get('bayesian_optimizer')
            
            tprint_success(f"✅ Models loaded from {filepath}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to load models: {e}")
            raise
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results."""
        try:
            summary = {
                'config': self.config,
                'component_availability': {
                    'msm_clustering': MSM_AVAILABLE,
                    'attention_mechanisms': ATTENTION_AVAILABLE,
                    'bayesian_optimization': BAYESIAN_OPTIMIZATION_AVAILABLE,
                    'enhanced_analyst': ENHANCED_ANALYST_AVAILABLE,
                    'enhanced_tactician': ENHANCED_TACTICIAN_AVAILABLE
                },
                'initialized_components': {
                    'analyst_trainer': self.analyst_trainer is not None,
                    'tactician_trainer': self.tactician_trainer is not None,
                    'msm_clusterer': self.msm_clusterer is not None,
                    'bayesian_optimizer': self.bayesian_optimizer is not None
                }
            }
            
            return summary
            
        except Exception as e:
            tprint_error(f"❌ Failed to get training summary: {e}")
            return {}