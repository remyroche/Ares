"""Voting mechanism component for analyst ensemble creation."""

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier

from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger


class VotingMechanism:
    """Handles different voting mechanisms for ensemble creation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the voting mechanism.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get("voting_mechanism", {})
        self.logger = system_logger.getChild("voting_mechanism")
        
        # Voting configuration
        self.default_voting_type = self.config.get("default_type", "soft")
        self.use_weights = self.config.get("use_weights", True)
        self.weight_by_performance = self.config.get("weight_by_performance", True)
        
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="voting ensemble creation"
    )
    async def create_voting_ensemble(
        self,
        models: Dict[str, Dict[str, Any]],
        voting_types: List[str]
    ) -> Dict[str, Any]:
        """Create voting ensembles with different voting mechanisms.
        
        Args:
            models: Dictionary of models with metadata
            voting_types: List of voting types to create
            
        Returns:
            Dictionary of voting ensembles
        """
        self.logger.info(f"Creating voting ensembles with types: {voting_types}")
        
        voting_ensembles = {}
        
        # Extract base models
        base_models = [(name, info["model"]) for name, info in models.items()]
        
        for voting_type in voting_types:
            if voting_type == "hard":
                ensemble = await self._create_hard_voting_ensemble(base_models)
            elif voting_type == "soft":
                ensemble = await self._create_soft_voting_ensemble(base_models)
            elif voting_type == "weighted_soft":
                ensemble = await self._create_weighted_soft_voting_ensemble(
                    base_models, models
                )
            else:
                self.logger.warning(f"Unknown voting type: {voting_type}")
                continue
            
            if ensemble is not None:
                voting_ensembles[voting_type] = ensemble
        
        return voting_ensembles
    
    async def _create_hard_voting_ensemble(
        self,
        base_models: List[tuple]
    ) -> Optional[VotingClassifier]:
        """Create a hard voting ensemble.
        
        Args:
            base_models: List of (name, model) tuples
            
        Returns:
            Hard voting ensemble
        """
        try:
            ensemble = VotingClassifier(
                estimators=base_models,
                voting='hard',
                n_jobs=-1
            )
            
            self.logger.info(f"Created hard voting ensemble with {len(base_models)} models")
            return ensemble
            
        except Exception as e:
            self.logger.error(f"Failed to create hard voting ensemble: {str(e)}")
            return None
    
    async def _create_soft_voting_ensemble(
        self,
        base_models: List[tuple]
    ) -> Optional[VotingClassifier]:
        """Create a soft voting ensemble.
        
        Args:
            base_models: List of (name, model) tuples
            
        Returns:
            Soft voting ensemble
        """
        try:
            # Filter models that support probability predictions
            proba_models = []
            for name, model in base_models:
                if hasattr(model, 'predict_proba'):
                    proba_models.append((name, model))
                else:
                    self.logger.warning(
                        f"Model {name} doesn't support predict_proba, "
                        f"excluding from soft voting"
                    )
            
            if not proba_models:
                self.logger.warning("No models support soft voting")
                return None
            
            ensemble = VotingClassifier(
                estimators=proba_models,
                voting='soft',
                n_jobs=-1
            )
            
            self.logger.info(f"Created soft voting ensemble with {len(proba_models)} models")
            return ensemble
            
        except Exception as e:
            self.logger.error(f"Failed to create soft voting ensemble: {str(e)}")
            return None
    
    async def _create_weighted_soft_voting_ensemble(
        self,
        base_models: List[tuple],
        models_info: Dict[str, Dict[str, Any]]
    ) -> Optional[VotingClassifier]:
        """Create a weighted soft voting ensemble.
        
        Args:
            base_models: List of (name, model) tuples
            models_info: Dictionary with model metadata including performance
            
        Returns:
            Weighted soft voting ensemble
        """
        try:
            # Filter models that support probability predictions
            proba_models = []
            weights = []
            
            for name, model in base_models:
                if hasattr(model, 'predict_proba'):
                    proba_models.append((name, model))
                    
                    # Get weight based on performance
                    if self.weight_by_performance and name in models_info:
                        performance = models_info[name].get("performance", 0.5)
                        weights.append(performance)
                    else:
                        weights.append(1.0)
            
            if not proba_models:
                self.logger.warning("No models support weighted soft voting")
                return None
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            ensemble = VotingClassifier(
                estimators=proba_models,
                voting='soft',
                weights=weights,
                n_jobs=-1
            )
            
            self.logger.info(
                f"Created weighted soft voting ensemble with {len(proba_models)} models, "
                f"weights: {dict(zip([m[0] for m in proba_models], weights))}"
            )
            return ensemble
            
        except Exception as e:
            self.logger.error(f"Failed to create weighted soft voting ensemble: {str(e)}")
            return None
    
    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="dynamic voting creation"
    )
    async def create_dynamic_voting_ensemble(
        self,
        models: Dict[str, Dict[str, Any]],
        features: pd.DataFrame
    ) -> Optional[Any]:
        """Create a dynamic voting ensemble that adjusts weights based on input.
        
        Args:
            models: Dictionary of models with metadata
            features: Feature data for validation
            
        Returns:
            Dynamic voting ensemble
        """
        self.logger.info("Creating dynamic voting ensemble...")
        
        # This is a placeholder for a more sophisticated dynamic voting mechanism
        # In practice, this could use techniques like:
        # - Dynamic classifier selection (DCS)
        # - Dynamic ensemble selection (DES)
        # - Meta-learning for weight prediction
        
        class DynamicVotingEnsemble:
            def __init__(self, base_models, initial_weights=None):
                self.base_models = base_models
                self.initial_weights = initial_weights or np.ones(len(base_models))
                self.is_fitted = False
                
            def fit(self, X, y):
                # Fit all base models
                for name, model in self.base_models:
                    model.fit(X, y)
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                
                # Get predictions from all models
                predictions = []
                for name, model in self.base_models:
                    pred = model.predict(X)
                    predictions.append(pred)
                
                predictions = np.array(predictions)
                
                # Simple majority voting for now
                # In practice, weights could be adjusted based on X
                return np.apply_along_axis(
                    lambda x: np.bincount(x.astype(int)).argmax(),
                    axis=0,
                    arr=predictions
                )
            
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                
                # Get probability predictions from models that support it
                proba_predictions = []
                weights = []
                
                for i, (name, model) in enumerate(self.base_models):
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        proba_predictions.append(proba)
                        weights.append(self.initial_weights[i])
                
                if not proba_predictions:
                    # Fallback to hard predictions
                    predictions = self.predict(X)
                    n_classes = 2  # Assuming binary classification
                    proba = np.zeros((len(X), n_classes))
                    proba[np.arange(len(X)), predictions] = 1.0
                    return proba
                
                # Weighted average of probabilities
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                weighted_proba = np.zeros_like(proba_predictions[0])
                for proba, weight in zip(proba_predictions, weights):
                    weighted_proba += weight * proba
                
                return weighted_proba
        
        # Extract base models
        base_models = [(name, info["model"]) for name, info in models.items()]
        
        # Get initial weights based on performance
        if self.weight_by_performance:
            weights = [info.get("performance", 0.5) for info in models.values()]
        else:
            weights = None
        
        dynamic_ensemble = DynamicVotingEnsemble(base_models, weights)
        
        self.logger.info(
            f"Created dynamic voting ensemble with {len(base_models)} models"
        )
        
        return dynamic_ensemble