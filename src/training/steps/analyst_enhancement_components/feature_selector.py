"""Feature selection component for analyst enhancement."""

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger


class FeatureSelector:
    """Handles feature selection for analyst models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the feature selector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get("feature_selection", {})
        self.logger = system_logger.getChild("feature_selector")
        
        # Feature selection configuration
        self.methods = self.config.get("methods", ["mutual_info", "importance", "rfe"])
        self.selection_threshold = self.config.get("selection_threshold", 0.8)
        self.min_features = self.config.get("min_features", 10)
        self.max_features = self.config.get("max_features", None)
        
        # Method-specific configuration
        self.method_config = {
            "mutual_info": {
                "n_neighbors": 3,
                "random_state": 42
            },
            "importance": {
                "n_estimators": 100,
                "random_state": 42
            },
            "rfe": {
                "step": 0.1,
                "n_features_to_select": None
            }
        }
    
    @handles_errors(
        exceptions=(Exception,),
        default_return=[],
        context="feature selection"
    )
    async def select_features(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        regime_id: str
    ) -> List[str]:
        """Select optimal features for a model.
        
        Args:
            model: Model to use for selection
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            regime_id: Regime identifier
            
        Returns:
            List of selected feature names
        """
        self.logger.info(f"Starting feature selection for regime {regime_id}")
        
        # Run multiple selection methods
        feature_scores = {}
        
        for method in self.methods:
            if method == "mutual_info":
                scores = await self._mutual_info_selection(X_train, y_train)
            elif method == "importance":
                scores = await self._importance_selection(model, X_train, y_train)
            elif method == "rfe":
                scores = await self._rfe_selection(model, X_train, y_train)
            else:
                continue
                
            feature_scores[method] = scores
        
        # Combine scores from different methods
        combined_scores = self._combine_feature_scores(feature_scores)
        
        # Select features based on combined scores
        selected_features = self._select_top_features(combined_scores)
        
        # Validate selection
        selected_features = await self._validate_selection(
            model, X_train, y_train, X_val, y_val, selected_features
        )
        
        self.logger.info(
            f"Selected {len(selected_features)} features from {X_train.shape[1]} "
            f"for regime {regime_id}"
        )
        
        return selected_features
    
    async def _mutual_info_selection(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, float]:
        """Perform mutual information feature selection."""
        try:
            # Calculate mutual information scores
            mi_scores = mutual_info_classif(
                X_train,
                y_train,
                n_neighbors=self.method_config["mutual_info"]["n_neighbors"],
                random_state=self.method_config["mutual_info"]["random_state"]
            )
            
            # Create feature score dictionary
            scores = {
                feature: score 
                for feature, score in zip(X_train.columns, mi_scores)
            }
            
            return scores
            
        except Exception as e:
            self.logger.warning(f"Mutual info selection failed: {str(e)}")
            return {}
    
    async def _importance_selection(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, float]:
        """Perform feature importance selection."""
        try:
            # Use RandomForest for feature importance if model doesn't support it
            if hasattr(model, "feature_importances_"):
                # Fit model and get importances
                model.fit(X_train, y_train)
                importances = model.feature_importances_
            else:
                # Use RandomForest as proxy
                rf = RandomForestClassifier(
                    n_estimators=self.method_config["importance"]["n_estimators"],
                    random_state=self.method_config["importance"]["random_state"],
                    n_jobs=-1
                )
                rf.fit(X_train, y_train)
                importances = rf.feature_importances_
            
            # Create feature score dictionary
            scores = {
                feature: importance 
                for feature, importance in zip(X_train.columns, importances)
            }
            
            return scores
            
        except Exception as e:
            self.logger.warning(f"Importance selection failed: {str(e)}")
            return {}
    
    async def _rfe_selection(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, float]:
        """Perform recursive feature elimination."""
        try:
            # Determine number of features to select
            n_features = self.method_config["rfe"]["n_features_to_select"]
            if n_features is None:
                n_features = max(
                    self.min_features,
                    int(X_train.shape[1] * self.selection_threshold)
                )
            
            # Create RFE selector
            selector = RFE(
                estimator=model,
                n_features_to_select=n_features,
                step=self.method_config["rfe"]["step"]
            )
            
            # Fit selector
            selector.fit(X_train, y_train)
            
            # Create feature score dictionary (1 for selected, 0 for not selected)
            scores = {
                feature: 1.0 if selected else 0.0
                for feature, selected in zip(X_train.columns, selector.support_)
            }
            
            return scores
            
        except Exception as e:
            self.logger.warning(f"RFE selection failed: {str(e)}")
            return {}
    
    def _combine_feature_scores(
        self,
        feature_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Combine scores from different selection methods."""
        if not feature_scores:
            return {}
        
        # Get all features
        all_features = set()
        for scores in feature_scores.values():
            all_features.update(scores.keys())
        
        # Calculate combined scores (average)
        combined = {}
        for feature in all_features:
            scores = []
            for method_scores in feature_scores.values():
                if feature in method_scores:
                    # Normalize score to [0, 1]
                    score = method_scores[feature]
                    if score > 0:
                        scores.append(score)
            
            if scores:
                # Normalize scores within each method
                combined[feature] = np.mean(scores)
            else:
                combined[feature] = 0.0
        
        # Normalize combined scores
        max_score = max(combined.values()) if combined else 1.0
        if max_score > 0:
            combined = {f: s / max_score for f, s in combined.items()}
        
        return combined
    
    def _select_top_features(
        self,
        feature_scores: Dict[str, float]
    ) -> List[str]:
        """Select top features based on scores."""
        if not feature_scores:
            return []
        
        # Sort features by score
        sorted_features = sorted(
            feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Determine number of features to select
        n_features = len(sorted_features)
        if self.max_features is not None:
            n_features = min(n_features, self.max_features)
        
        # Apply threshold
        threshold_features = [
            f for f, s in sorted_features 
            if s >= self.selection_threshold
        ]
        
        # Ensure minimum features
        if len(threshold_features) < self.min_features:
            selected = [f for f, _ in sorted_features[:self.min_features]]
        else:
            selected = threshold_features[:n_features]
        
        return selected
    
    async def _validate_selection(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        selected_features: List[str]
    ) -> List[str]:
        """Validate feature selection by comparing performance."""
        try:
            # Train with all features
            model_all = model.__class__(**model.get_params())
            model_all.fit(X_train, y_train)
            pred_all = model_all.predict(X_val)
            score_all = accuracy_score(y_val, pred_all)
            
            # Train with selected features
            if selected_features:
                X_train_selected = X_train[selected_features]
                X_val_selected = X_val[selected_features]
                
                model_selected = model.__class__(**model.get_params())
                model_selected.fit(X_train_selected, y_train)
                pred_selected = model_selected.predict(X_val_selected)
                score_selected = accuracy_score(y_val, pred_selected)
                
                # If selected features perform significantly worse, use all features
                if score_selected < score_all * 0.95:  # 5% tolerance
                    self.logger.warning(
                        f"Selected features underperform "
                        f"({score_selected:.4f} vs {score_all:.4f}), "
                        f"using all features"
                    )
                    return X_train.columns.tolist()
                else:
                    self.logger.info(
                        f"Feature selection validated: "
                        f"{score_selected:.4f} (selected) vs {score_all:.4f} (all)"
                    )
                    return selected_features
            else:
                return X_train.columns.tolist()
                
        except Exception as e:
            self.logger.warning(f"Feature validation failed: {str(e)}")
            return selected_features if selected_features else X_train.columns.tolist()