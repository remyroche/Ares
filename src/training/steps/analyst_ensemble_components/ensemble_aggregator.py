"""Ensemble aggregation component for analyst ensemble creation."""

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger


class EnsembleAggregator:
    """Handles ensemble aggregation strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the ensemble aggregator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get("ensemble_creation", {})
        self.logger = system_logger.getChild("ensemble_aggregator")
        
        # Aggregation configuration
        self.stacking_cv_folds = self.config.get("stacking_cv_folds", 5)
        self.blending_holdout_ratio = self.config.get("blending_holdout_ratio", 0.2)
        self.use_proba_features = self.config.get("use_proba_features", True)
        
    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="weighted ensemble creation"
    )
    async def create_weighted_ensemble(
        self,
        models: Dict[str, Dict[str, Any]],
        features: pd.DataFrame
    ) -> Any:
        """Create a weighted ensemble based on model performance.
        
        Args:
            models: Dictionary of models with metadata
            features: Feature data for validation
            
        Returns:
            Weighted ensemble model
        """
        self.logger.info("Creating weighted ensemble...")
        
        # Extract models and their performance scores
        model_list = []
        weights = []
        
        for model_name, model_info in models.items():
            model = model_info["model"]
            performance = model_info.get("performance", 0.5)
            
            model_list.append((model_name, model))
            weights.append(performance)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Create weighted voting classifier
        weighted_ensemble = VotingClassifier(
            estimators=model_list,
            voting='soft',
            weights=weights,
            n_jobs=-1
        )
        
        self.logger.info(
            f"Created weighted ensemble with {len(model_list)} models, "
            f"weights: {dict(zip([m[0] for m in model_list], weights))}"
        )
        
        return weighted_ensemble
    
    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="stacking ensemble creation"
    )
    async def create_stacking_ensemble(
        self,
        models: Dict[str, Dict[str, Any]],
        features: pd.DataFrame,
        meta_learner_type: str = "logistic_regression"
    ) -> Any:
        """Create a stacking ensemble with meta-learner.
        
        Args:
            models: Dictionary of models with metadata
            features: Feature data for validation
            meta_learner_type: Type of meta-learner to use
            
        Returns:
            Stacking ensemble model
        """
        self.logger.info(f"Creating stacking ensemble with {meta_learner_type} meta-learner...")
        
        # Extract base models
        base_models = []
        for model_name, model_info in models.items():
            model = model_info["model"]
            base_models.append((model_name, model))
        
        # Create meta-learner
        meta_learner = self._create_meta_learner(meta_learner_type)
        
        # Create stacking classifier
        stacking_ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=self.stacking_cv_folds,
            stack_method='predict_proba' if self.use_proba_features else 'predict',
            n_jobs=-1
        )
        
        self.logger.info(
            f"Created stacking ensemble with {len(base_models)} base models "
            f"and {meta_learner_type} meta-learner"
        )
        
        return stacking_ensemble
    
    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="blending ensemble creation"
    )
    async def create_blending_ensemble(
        self,
        models: Dict[str, Dict[str, Any]],
        features: pd.DataFrame
    ) -> Any:
        """Create a blending ensemble.
        
        Args:
            models: Dictionary of models with metadata
            features: Feature data for validation
            
        Returns:
            Blending ensemble model
        """
        self.logger.info("Creating blending ensemble...")
        
        # For blending, we create a custom ensemble that uses a holdout set
        # This is a simplified implementation
        class BlendingEnsemble:
            def __init__(self, base_models, meta_learner, holdout_ratio=0.2):
                self.base_models = base_models
                self.meta_learner = meta_learner
                self.holdout_ratio = holdout_ratio
                self.is_fitted = False
                
            def fit(self, X, y):
                # Split data into blend and holdout
                n_samples = len(X)
                n_holdout = int(n_samples * self.holdout_ratio)
                
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                
                holdout_indices = indices[:n_holdout]
                blend_indices = indices[n_holdout:]
                
                # Fit base models on blend data
                X_blend = X.iloc[blend_indices] if hasattr(X, 'iloc') else X[blend_indices]
                y_blend = y.iloc[blend_indices] if hasattr(y, 'iloc') else y[blend_indices]
                
                X_holdout = X.iloc[holdout_indices] if hasattr(X, 'iloc') else X[holdout_indices]
                y_holdout = y.iloc[holdout_indices] if hasattr(y, 'iloc') else y[holdout_indices]
                
                # Fit base models
                for name, model in self.base_models:
                    model.fit(X_blend, y_blend)
                
                # Generate predictions on holdout
                blend_features = []
                for name, model in self.base_models:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X_holdout)[:, 1]
                    else:
                        pred = model.predict(X_holdout)
                    blend_features.append(pred)
                
                # Stack predictions
                blend_features = np.column_stack(blend_features)
                
                # Fit meta-learner on blend features
                self.meta_learner.fit(blend_features, y_holdout)
                
                # Refit all models on full data
                for name, model in self.base_models:
                    model.fit(X, y)
                
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                
                # Generate base predictions
                blend_features = []
                for name, model in self.base_models:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)[:, 1]
                    else:
                        pred = model.predict(X)
                    blend_features.append(pred)
                
                # Stack and predict
                blend_features = np.column_stack(blend_features)
                return self.meta_learner.predict(blend_features)
            
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                
                # Generate base predictions
                blend_features = []
                for name, model in self.base_models:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)[:, 1]
                    else:
                        pred = model.predict(X)
                    blend_features.append(pred)
                
                # Stack and predict
                blend_features = np.column_stack(blend_features)
                
                if hasattr(self.meta_learner, 'predict_proba'):
                    return self.meta_learner.predict_proba(blend_features)
                else:
                    # Create pseudo-probabilities
                    predictions = self.meta_learner.predict(blend_features)
                    return np.column_stack([1 - predictions, predictions])
        
        # Extract base models
        base_models = [(name, info["model"]) for name, info in models.items()]
        
        # Create meta-learner
        meta_learner = self._create_meta_learner("logistic_regression")
        
        # Create blending ensemble
        blending_ensemble = BlendingEnsemble(
            base_models,
            meta_learner,
            self.blending_holdout_ratio
        )
        
        self.logger.info(
            f"Created blending ensemble with {len(base_models)} base models"
        )
        
        return blending_ensemble
    
    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="meta ensemble creation"
    )
    async def create_meta_ensemble(
        self,
        regime_models: Dict[str, Any],
        features: pd.DataFrame
    ) -> Any:
        """Create a meta-ensemble from regime-specific models.
        
        Args:
            regime_models: Dictionary of models from different regimes
            features: Feature data
            
        Returns:
            Meta-ensemble model
        """
        self.logger.info("Creating meta-ensemble across regimes...")
        
        # Convert to list format for VotingClassifier
        model_list = list(regime_models.items())
        
        # Create voting ensemble with equal weights
        meta_ensemble = VotingClassifier(
            estimators=model_list,
            voting='soft',
            n_jobs=-1
        )
        
        self.logger.info(
            f"Created meta-ensemble with {len(model_list)} regime models"
        )
        
        return meta_ensemble
    
    def _create_meta_learner(self, meta_learner_type: str) -> Any:
        """Create a meta-learner model.
        
        Args:
            meta_learner_type: Type of meta-learner
            
        Returns:
            Meta-learner model instance
        """
        if meta_learner_type == "logistic_regression":
            return LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        else:
            # Default to logistic regression
            self.logger.warning(
                f"Unknown meta-learner type: {meta_learner_type}, "
                f"using logistic regression"
            )
            return LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )