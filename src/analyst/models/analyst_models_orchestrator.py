"""
Analyst Models Orchestrator

Orchestrates all Analyst models (A1-A4) and the stacker meta-learner
for comprehensive "green light" binary classification with uncertainty estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import os

# Import all analyst models
from .analyst_a1_patchtst_lightgbm import (
    AnalystA1Model, PatchTSTConfig as A1PatchTSTConfig, 
    LightGBMConfig, CalibrationConfig as A1CalibrationConfig
)
from .analyst_a2_patchtst_xgboost import (
    AnalystA2Model, XGBoostConfig, CalibrationConfig as A2CalibrationConfig
)
from .analyst_a3_ft_transformer import (
    AnalystA3Model, FTTransformerConfig, CalibrationConfig as A3CalibrationConfig
)
from .analyst_a4_patchtst_catboost import (
    AnalystA4Model, PatchTSTConfig as A4PatchTSTConfig, 
    CatBoostConfig, CalibrationConfig as A4CalibrationConfig
)
from .stacker_lgbm_calibrated import (
    StackerLGBMCalibrated, StackerConfig, CalibrationConfig as StackerCalibrationConfig
)

logger = logging.getLogger(__name__)


@dataclass
class AnalystModelsConfig:
    """Configuration for all Analyst models."""
    # Model enablement
    enable_a1: bool = True
    enable_a2: bool = True
    enable_a3: bool = True
    enable_a4: bool = True
    enable_stacker: bool = True
    
    # Parallel training
    enable_parallel_training: bool = True
    max_workers: int = 4
    
    # Model-specific configs
    a1_config: Optional[Dict[str, Any]] = None
    a2_config: Optional[Dict[str, Any]] = None
    a3_config: Optional[Dict[str, Any]] = None
    a4_config: Optional[Dict[str, Any]] = None
    stacker_config: Optional[Dict[str, Any]] = None
    
    # Output settings
    save_models: bool = True
    output_directory: str = "generated/analyst_models"
    model_prefix: str = "analyst"


class AnalystModelsOrchestrator:
    """Orchestrates all Analyst models for ensemble prediction."""
    
    def __init__(self, config: Optional[AnalystModelsConfig] = None):
        self.config = config or AnalystModelsConfig()
        
        # Model instances
        self.models = {}
        self.stacker = None
        self.is_fitted = False
        
        # Results storage
        self.training_results = {}
        self.prediction_cache = {}
        
        logger.info("Initialized Analyst Models Orchestrator")
    
    def _create_model_configs(self) -> Dict[str, Any]:
        """Create model-specific configurations."""
        configs = {}
        
        # A1 Model (PatchTST + LightGBM)
        if self.config.enable_a1:
            a1_patchtst_config = A1PatchTSTConfig(
                patch_len=24, stride=6, d_model=128, n_layers=3,
                causal_pooling=True, dropout=0.1, attention_heads=4
            )
            a1_lightgbm_config = LightGBMConfig(
                max_depth=7, learning_rate=0.05, n_estimators=1000,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1
            )
            a1_calibration_config = A1CalibrationConfig(
                method='isotonic', cv_folds=5, enable_venn_abers=True
            )
            configs['a1'] = {
                'patchtst_config': a1_patchtst_config,
                'lightgbm_config': a1_lightgbm_config,
                'calibration_config': a1_calibration_config
            }
        
        # A2 Model (PatchTST + XGBoost)
        if self.config.enable_a2:
            a2_patchtst_config = A1PatchTSTConfig(
                patch_len=24, stride=6, d_model=128, n_layers=3,
                causal_pooling=True, dropout=0.1, attention_heads=4
            )
            a2_xgboost_config = XGBoostConfig(
                max_depth=6, learning_rate=0.05, n_estimators=1000,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1
            )
            a2_calibration_config = A2CalibrationConfig(
                method='isotonic', cv_folds=5, enable_venn_abers=True
            )
            configs['a2'] = {
                'patchtst_config': a2_patchtst_config,
                'xgboost_config': a2_xgboost_config,
                'calibration_config': a2_calibration_config
            }
        
        # A3 Model (FT-Transformer)
        if self.config.enable_a3:
            a3_transformer_config = FTTransformerConfig(
                d_model=128, n_blocks=3, n_heads=2, dropout=0.1,
                d_ff=512, learning_rate=1e-4, num_epochs=100
            )
            a3_calibration_config = A3CalibrationConfig(
                method='isotonic', cv_folds=5, enable_venn_abers=True
            )
            configs['a3'] = {
                'transformer_config': a3_transformer_config,
                'calibration_config': a3_calibration_config
            }
        
        # A4 Model (PatchTST + CatBoost)
        if self.config.enable_a4:
            a4_patchtst_config = A4PatchTSTConfig(
                patch_len=24, stride=6, d_model=128, n_layers=3,
                causal_pooling=True, dropout=0.1, attention_heads=4, frozen=True
            )
            a4_catboost_config = CatBoostConfig(
                depth=6, iterations=1000, l2_leaf_reg=10.0,
                learning_rate=0.1, subsample=0.8, colsample_bylevel=0.8
            )
            a4_calibration_config = A4CalibrationConfig(
                method='isotonic', cv_folds=5, enable_venn_abers=True
            )
            configs['a4'] = {
                'patchtst_config': a4_patchtst_config,
                'catboost_config': a4_catboost_config,
                'calibration_config': a4_calibration_config
            }
        
        # Stacker Model
        if self.config.enable_stacker:
            stacker_config = StackerConfig(
                max_depth=6, learning_rate=0.05, n_estimators=500,
                subsample=0.8, colsample_bytree=0.8
            )
            stacker_calibration_config = StackerCalibrationConfig(
                method='isotonic', cv_folds=5, enable_venn_abers=True,
                per_regime_calibration=True
            )
            configs['stacker'] = {
                'stacker_config': stacker_config,
                'calibration_config': stacker_calibration_config
            }
        
        return configs
    
    def _create_models(self) -> None:
        """Create model instances."""
        configs = self._create_model_configs()
        
        # Create A1 model
        if self.config.enable_a1 and 'a1' in configs:
            self.models['a1'] = AnalystA1Model(**configs['a1'])
            logger.info("Created A1 model (PatchTST + LightGBM)")
        
        # Create A2 model
        if self.config.enable_a2 and 'a2' in configs:
            self.models['a2'] = AnalystA2Model(**configs['a2'])
            logger.info("Created A2 model (PatchTST + XGBoost)")
        
        # Create A3 model
        if self.config.enable_a3 and 'a3' in configs:
            self.models['a3'] = AnalystA3Model(**configs['a3'])
            logger.info("Created A3 model (FT-Transformer)")
        
        # Create A4 model
        if self.config.enable_a4 and 'a4' in configs:
            self.models['a4'] = AnalystA4Model(**configs['a4'])
            logger.info("Created A4 model (PatchTST + CatBoost)")
        
        # Create stacker model
        if self.config.enable_stacker and 'stacker' in configs:
            self.stacker = StackerLGBMCalibrated(**configs['stacker'])
            logger.info("Created Stacker LGBM Calibrated meta-learner")
    
    def _train_single_model(self, model_name: str, model, X: np.ndarray, y: np.ndarray,
                           regimes: Optional[np.ndarray] = None,
                           sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train a single model."""
        try:
            logger.info(f"Training {model_name} model...")
            
            # Fit model
            model.fit(X, y, regimes=regimes, sample_weight=sample_weight)
            
            # Get predictions for evaluation
            y_prob = model.predict_proba(X, regimes)
            y_pred = (y_prob > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y)
            logloss = -np.mean(y * np.log(y_prob + 1e-15) + (1 - y) * np.log(1 - y_prob + 1e-15))
            
            result = {
                'model': model,
                'accuracy': accuracy,
                'logloss': logloss,
                'predictions': y_prob,
                'uncertainty': model.predict_uncertainty(X, regimes) if hasattr(model, 'predict_uncertainty') else {}
            }
            
            logger.info(f"✅ {model_name} model trained - Accuracy: {accuracy:.4f}, LogLoss: {logloss:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to train {model_name} model: {e}")
            return {'model': None, 'error': str(e)}
    
    async def fit(self, X: np.ndarray, y: np.ndarray, 
                  regimes: Optional[np.ndarray] = None,
                  sample_weight: Optional[np.ndarray] = None) -> 'AnalystModelsOrchestrator':
        """Fit all Analyst models."""
        logger.info("Fitting Analyst Models Orchestrator...")
        
        # Create models
        self._create_models()
        
        if not self.models:
            raise ValueError("No models enabled for training")
        
        # Train base models
        if self.config.enable_parallel_training:
            # Parallel training
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {}
                for model_name, model in self.models.items():
                    future = executor.submit(
                        self._train_single_model, 
                        model_name, model, X, y, regimes, sample_weight
                    )
                    futures[future] = model_name
                
                # Collect results
                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        result = future.result()
                        self.training_results[model_name] = result
                    except Exception as e:
                        logger.error(f"❌ Error training {model_name}: {e}")
                        self.training_results[model_name] = {'model': None, 'error': str(e)}
        else:
            # Sequential training
            for model_name, model in self.models.items():
                result = self._train_single_model(model_name, model, X, y, regimes, sample_weight)
                self.training_results[model_name] = result
        
        # Train stacker if enabled and we have successful base models
        if self.config.enable_stacker and self.stacker is not None:
            successful_models = {k: v for k, v in self.training_results.items() 
                               if v.get('model') is not None}
            
            if len(successful_models) >= 2:  # Need at least 2 models for stacking
                logger.info("Training stacker meta-learner...")
                
                # Prepare stacking data
                model_predictions = {}
                uncertainty_estimates = {}
                
                for model_name, result in successful_models.items():
                    model_predictions[model_name] = result['predictions']
                    uncertainty_estimates[model_name] = result['uncertainty']
                
                # Train stacker
                try:
                    self.stacker.fit(
                        model_predictions=model_predictions,
                        uncertainty_estimates=uncertainty_estimates,
                        y=y,
                        regimes=regimes,
                        sample_weight=sample_weight
                    )
                    logger.info("✅ Stacker meta-learner trained successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to train stacker: {e}")
                    self.stacker = None
            else:
                logger.warning("⚠️ Not enough successful base models for stacking")
                self.stacker = None
        
        self.is_fitted = True
        logger.info("✅ Analyst Models Orchestrator fitted successfully")
        return self
    
    def predict_proba(self, X: np.ndarray, regimes: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Orchestrator must be fitted before prediction")
        
        # Get predictions from base models
        model_predictions = {}
        uncertainty_estimates = {}
        
        for model_name, result in self.training_results.items():
            if result.get('model') is not None:
                model = result['model']
                predictions = model.predict_proba(X, regimes)
                uncertainty = model.predict_uncertainty(X, regimes) if hasattr(model, 'predict_uncertainty') else {}
                
                model_predictions[model_name] = predictions
                uncertainty_estimates[model_name] = uncertainty
        
        # Use stacker if available
        if self.stacker is not None and len(model_predictions) >= 2:
            return self.stacker.predict_proba(model_predictions, uncertainty_estimates, regimes)
        else:
            # Fallback: average predictions
            if model_predictions:
                predictions_array = np.array(list(model_predictions.values()))
                return np.mean(predictions_array, axis=0)
            else:
                raise ValueError("No trained models available for prediction")
    
    def predict_uncertainty(self, X: np.ndarray, regimes: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Predict uncertainty estimates."""
        if not self.is_fitted:
            raise ValueError("Orchestrator must be fitted before prediction")
        
        # Get predictions from base models
        model_predictions = {}
        uncertainty_estimates = {}
        
        for model_name, result in self.training_results.items():
            if result.get('model') is not None:
                model = result['model']
                predictions = model.predict_proba(X, regimes)
                uncertainty = model.predict_uncertainty(X, regimes) if hasattr(model, 'predict_uncertainty') else {}
                
                model_predictions[model_name] = predictions
                uncertainty_estimates[model_name] = uncertainty
        
        # Use stacker if available
        if self.stacker is not None and len(model_predictions) >= 2:
            return self.stacker.predict_uncertainty(model_predictions, uncertainty_estimates, regimes)
        else:
            # Fallback: combine uncertainties
            if uncertainty_estimates:
                # Simple combination of uncertainties
                all_probs = [est.get('probability', np.zeros(len(X))) for est in uncertainty_estimates.values()]
                combined_probs = np.mean(all_probs, axis=0)
                
                return {
                    'probability': combined_probs,
                    'confidence_intervals': {},
                    'margin_stats': {
                        'mean_probability': np.mean(combined_probs),
                        'std_probability': np.std(combined_probs),
                        'min_probability': np.min(combined_probs),
                        'max_probability': np.max(combined_probs),
                        'confidence_range': np.max(combined_probs) - np.min(combined_probs)
                    }
                }
            else:
                raise ValueError("No trained models available for prediction")
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models."""
        if not self.is_fitted:
            return {}
        
        performance = {}
        for model_name, result in self.training_results.items():
            if result.get('model') is not None:
                performance[model_name] = {
                    'accuracy': result.get('accuracy', 0.0),
                    'logloss': result.get('logloss', float('inf')),
                    'model_type': type(result['model']).__name__
                }
        
        if self.stacker is not None:
            performance['stacker'] = {
                'model_type': 'StackerLGBMCalibrated',
                'feature_importance': self.stacker.get_feature_importance()
            }
        
        return performance
    
    def save_models(self, base_path: Optional[str] = None) -> None:
        """Save all models to disk."""
        if not self.is_fitted:
            raise ValueError("Orchestrator must be fitted before saving")
        
        if not self.config.save_models:
            return
        
        base_path = base_path or self.config.output_directory
        os.makedirs(base_path, exist_ok=True)
        
        # Save base models
        for model_name, result in self.training_results.items():
            if result.get('model') is not None:
                model_path = os.path.join(base_path, f"{self.config.model_prefix}_{model_name}.joblib")
                result['model'].save_model(model_path)
                logger.info(f"✅ Saved {model_name} model to {model_path}")
        
        # Save stacker
        if self.stacker is not None:
            stacker_path = os.path.join(base_path, f"{self.config.model_prefix}_stacker.joblib")
            self.stacker.save_model(stacker_path)
            logger.info(f"✅ Saved stacker model to {stacker_path}")
        
        # Save orchestrator metadata
        metadata = {
            'config': self.config,
            'training_results': {k: {key: val for key, val in v.items() if key != 'model'} 
                               for k, v in self.training_results.items()},
            'is_fitted': self.is_fitted
        }
        metadata_path = os.path.join(base_path, f"{self.config.model_prefix}_orchestrator_metadata.joblib")
        joblib.dump(metadata, metadata_path)
        logger.info(f"✅ Saved orchestrator metadata to {metadata_path}")
    
    @classmethod
    def load_models(cls, base_path: str, config: Optional[AnalystModelsConfig] = None) -> 'AnalystModelsOrchestrator':
        """Load all models from disk."""
        # Load metadata
        metadata_path = os.path.join(base_path, "analyst_orchestrator_metadata.joblib")
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            config = config or metadata.get('config', AnalystModelsConfig())
        else:
            config = config or AnalystModelsConfig()
        
        # Create orchestrator
        orchestrator = cls(config)
        
        # Load base models
        for model_name in ['a1', 'a2', 'a3', 'a4']:
            model_path = os.path.join(base_path, f"analyst_{model_name}.joblib")
            if os.path.exists(model_path):
                try:
                    if model_name == 'a1':
                        model = AnalystA1Model.load_model(model_path)
                    elif model_name == 'a2':
                        model = AnalystA2Model.load_model(model_path)
                    elif model_name == 'a3':
                        model = AnalystA3Model.load_model(model_path)
                    elif model_name == 'a4':
                        model = AnalystA4Model.load_model(model_path)
                    
                    orchestrator.training_results[model_name] = {'model': model}
                    logger.info(f"✅ Loaded {model_name} model from {model_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to load {model_name} model: {e}")
        
        # Load stacker
        stacker_path = os.path.join(base_path, "analyst_stacker.joblib")
        if os.path.exists(stacker_path):
            try:
                orchestrator.stacker = StackerLGBMCalibrated.load_model(stacker_path)
                logger.info(f"✅ Loaded stacker model from {stacker_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load stacker model: {e}")
        
        orchestrator.is_fitted = True
        logger.info("✅ Analyst Models Orchestrator loaded successfully")
        return orchestrator


# Factory function for easy orchestrator creation
def create_analyst_models_orchestrator(config: Optional[AnalystModelsConfig] = None) -> AnalystModelsOrchestrator:
    """Create an Analyst Models Orchestrator with the specified configuration."""
    return AnalystModelsOrchestrator(config)