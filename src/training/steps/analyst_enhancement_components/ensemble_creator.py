"""Ensemble creation component for analyst enhancement."""

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger


class EnsembleCreator:
    """Handles ensemble creation from enhanced analyst models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the ensemble creator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get("ensemble", {})
        self.logger = system_logger.getChild("ensemble_creator")
        
        # Ensemble configuration
        self.ensemble_methods = self.config.get("methods", ["voting", "stacking"])
        self.voting_type = self.config.get("voting_type", "soft")
        self.meta_learner = self.config.get("meta_learner", "logistic_regression")
        self.min_models_for_ensemble = self.config.get("min_models", 3)
        
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="ensemble creation"
    )
    async def create_ensembles(
        self,
        enhanced_models: Dict[str, Dict[str, Any]],
        regime_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create ensemble models from enhanced models.
        
        Args:
            enhanced_models: Dictionary of enhanced models by regime
            regime_data: Regime metadata
            
        Returns:
            Dictionary of ensemble models
        """
        self.logger.info("Creating ensemble models...")
        
        ensemble_models = {}
        
        # Create ensembles for each regime
        for regime_id, regime_models in enhanced_models.items():
            if len(regime_models) < self.min_models_for_ensemble:
                self.logger.warning(
                    f"Regime {regime_id} has only {len(regime_models)} models, "
                    f"skipping ensemble creation"
                )
                continue
            
            regime_ensembles = await self._create_regime_ensembles(
                regime_id,
                regime_models,
                regime_data.get(regime_id, {})
            )
            
            ensemble_models[regime_id] = regime_ensembles
        
        # Create cross-regime ensemble
        if len(enhanced_models) > 1:
            cross_regime_ensemble = await self._create_cross_regime_ensemble(
                enhanced_models,
                regime_data
            )
            ensemble_models["cross_regime"] = cross_regime_ensemble
        
        self.logger.info(f"Created {len(ensemble_models)} ensemble models")
        
        return ensemble_models
    
    async def _create_regime_ensembles(
        self,
        regime_id: str,
        regime_models: Dict[str, Any],
        regime_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create ensembles for a specific regime.
        
        Args:
            regime_id: Regime identifier
            regime_models: Dictionary of models for this regime
            regime_info: Regime metadata
            
        Returns:
            Dictionary of ensemble models for this regime
        """
        ensembles = {}
        
        # Extract base models
        base_models = []
        model_names = []
        
        for model_name, model_info in regime_models.items():
            if isinstance(model_info, dict) and "model" in model_info:
                base_models.append((model_name, model_info["model"]))
                model_names.append(model_name)
        
        if len(base_models) < self.min_models_for_ensemble:
            return ensembles
        
        # Create voting ensemble
        if "voting" in self.ensemble_methods:
            voting_ensemble = await self._create_voting_ensemble(
                base_models,
                regime_id
            )
            if voting_ensemble is not None:
                ensembles["voting"] = {
                    "model": voting_ensemble,
                    "base_models": model_names,
                    "method": "voting",
                    "voting_type": self.voting_type
                }
        
        # Create stacking ensemble
        if "stacking" in self.ensemble_methods:
            stacking_ensemble = await self._create_stacking_ensemble(
                base_models,
                regime_id
            )
            if stacking_ensemble is not None:
                ensembles["stacking"] = {
                    "model": stacking_ensemble,
                    "base_models": model_names,
                    "method": "stacking",
                    "meta_learner": self.meta_learner
                }
        
        # Create weighted ensemble based on validation scores
        if regime_models:
            weighted_ensemble = await self._create_weighted_ensemble(
                regime_models,
                regime_id
            )
            if weighted_ensemble is not None:
                ensembles["weighted"] = weighted_ensemble
        
        return ensembles
    
    async def _create_voting_ensemble(
        self,
        base_models: List[tuple],
        regime_id: str
    ) -> Optional[VotingClassifier]:
        """Create a voting ensemble."""
        try:
            # Create voting classifier
            voting_ensemble = VotingClassifier(
                estimators=base_models,
                voting=self.voting_type,
                n_jobs=-1
            )
            
            self.logger.info(
                f"Created {self.voting_type} voting ensemble "
                f"with {len(base_models)} models for regime {regime_id}"
            )
            
            return voting_ensemble
            
        except Exception as e:
            self.logger.error(f"Failed to create voting ensemble: {str(e)}")
            return None
    
    async def _create_stacking_ensemble(
        self,
        base_models: List[tuple],
        regime_id: str
    ) -> Optional[StackingClassifier]:
        """Create a stacking ensemble."""
        try:
            # Create meta-learner
            if self.meta_learner == "logistic_regression":
                meta_model = LogisticRegression(random_state=42, max_iter=1000)
            else:
                # Default to logistic regression
                meta_model = LogisticRegression(random_state=42, max_iter=1000)
            
            # Create stacking classifier
            stacking_ensemble = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=3,  # Use 3-fold cross-validation for training meta-learner
                n_jobs=-1
            )
            
            self.logger.info(
                f"Created stacking ensemble with {len(base_models)} models "
                f"and {self.meta_learner} meta-learner for regime {regime_id}"
            )
            
            return stacking_ensemble
            
        except Exception as e:
            self.logger.error(f"Failed to create stacking ensemble: {str(e)}")
            return None
    
    async def _create_weighted_ensemble(
        self,
        regime_models: Dict[str, Any],
        regime_id: str
    ) -> Optional[Dict[str, Any]]:
        """Create a weighted ensemble based on validation scores."""
        try:
            # Extract models and their validation scores
            models_with_scores = []
            
            for model_name, model_info in regime_models.items():
                if isinstance(model_info, dict):
                    model = model_info.get("model")
                    score = model_info.get("validation_accuracy", 0.5)
                    if model is not None:
                        models_with_scores.append((model_name, model, score))
            
            if not models_with_scores:
                return None
            
            # Calculate weights based on scores
            scores = np.array([s for _, _, s in models_with_scores])
            
            # Use softmax to convert scores to weights
            exp_scores = np.exp(scores * 10)  # Scale factor of 10
            weights = exp_scores / exp_scores.sum()
            
            # Create weighted ensemble info
            weighted_ensemble = {
                "models": [(name, model) for name, model, _ in models_with_scores],
                "weights": weights.tolist(),
                "method": "weighted",
                "base_scores": scores.tolist()
            }
            
            self.logger.info(
                f"Created weighted ensemble with weights: "
                f"{dict(zip([m[0] for m in models_with_scores], weights))}"
            )
            
            return weighted_ensemble
            
        except Exception as e:
            self.logger.error(f"Failed to create weighted ensemble: {str(e)}")
            return None
    
    async def _create_cross_regime_ensemble(
        self,
        enhanced_models: Dict[str, Dict[str, Any]],
        regime_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an ensemble across different regimes."""
        try:
            # Collect best model from each regime
            best_models = []
            
            for regime_id, regime_models in enhanced_models.items():
                best_model = None
                best_score = -np.inf
                
                for model_name, model_info in regime_models.items():
                    if isinstance(model_info, dict):
                        score = model_info.get("validation_accuracy", 0)
                        if score > best_score:
                            best_score = score
                            best_model = (f"{regime_id}_{model_name}", model_info.get("model"))
                
                if best_model is not None and best_model[1] is not None:
                    best_models.append(best_model)
            
            if len(best_models) < 2:
                return {}
            
            # Create voting ensemble from best models
            cross_regime_voting = VotingClassifier(
                estimators=best_models,
                voting=self.voting_type,
                n_jobs=-1
            )
            
            return {
                "voting": {
                    "model": cross_regime_voting,
                    "base_models": [name for name, _ in best_models],
                    "method": "cross_regime_voting",
                    "voting_type": self.voting_type
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create cross-regime ensemble: {str(e)}")
            return {}