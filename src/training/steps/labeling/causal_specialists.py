"""
Causal Specialists Module

Implements causal specialists as causal parents in the structural causal model.
Manages specialist predictions, surprise detection, and coordination.

Key Features:
1. Causal specialists as causal parents
2. Specialist prediction tracking
3. Surprise detection and management
4. Specialist coordination and consensus
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from dataclasses import dataclass

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class CausalSpecialist:
    """
    Individual causal specialist representing a causal parent.
    
    Each specialist focuses on a specific causal relationship and
    provides predictions for that domain.
    """
    
    def __init__(
        self,
        name: str,
        causal_parent: str,
        causal_child: str,
        model_type: str = "linear",
        prediction_window: int = 1,
        confidence_threshold: float = 0.5,
        verbose: bool = True
    ):
        """
        Initialize causal specialist.
        
        Args:
            name: Specialist name
            causal_parent: Parent variable in causal relationship
            causal_child: Child variable in causal relationship
            model_type: Type of model to use
            prediction_window: Window for predictions
            confidence_threshold: Threshold for confidence
            verbose: Whether to print progress information
        """
        self.name = name
        self.causal_parent = causal_parent
        self.causal_child = causal_child
        self.model_type = model_type
        self.prediction_window = prediction_window
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        
        # Model and data storage
        self.model = None
        self.predictions_ = None
        self.prediction_errors_ = None
        self.confidence_scores_ = None
        self.surprise_events_ = None
        
        # Performance metrics
        self.performance_metrics_ = {}
        
    def _create_model(self) -> Any:
        """
        Create prediction model based on type.
        
        Returns:
            Model instance
        """
        if self.model_type == "linear":
            return LinearRegression()
        elif self.model_type == "ridge":
            return Ridge(alpha=1.0)
        elif self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=50, random_state=42, max_depth=5
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Fit the specialist model.
        
        Args:
            X: Feature matrix (parent variable)
            y: Target vector (child variable)
            sample_weights: Optional sample weights
            
        Returns:
            Training metrics
        """
        try:
            if self.verbose:
                tprint_info(f"🧠 Training Specialist: {self.name}")
            
            # Create and fit model
            self.model = self._create_model()
            
            # Check if y is numeric
            if y.dtype == 'object' or str(y.dtype) == 'string':
                tprint_warning(f"   ⚠️ Skipping {self.name}: target is non-numeric")
                self.is_fitted_ = False
                return {"error": "non-numeric target", "skipped": True}
            
            # Prepare data - filter to numeric columns only
            if self.causal_parent in X.columns:
                parent_col = X[self.causal_parent]
                # Skip if parent column is non-numeric
                if parent_col.dtype == 'object' or str(parent_col.dtype) == 'string':
                    tprint_warning(f"   ⚠️ Skipping {self.name}: parent '{self.causal_parent}' is non-numeric")
                    self.is_fitted_ = False
                    return {"error": "non-numeric parent", "skipped": True}
                X_train = X[[self.causal_parent]].fillna(0)
            else:
                # Filter to numeric columns only
                X_train = X.select_dtypes(include=[np.number]).fillna(0)
                if X_train.empty:
                    tprint_warning(f"   ⚠️ Skipping {self.name}: no numeric columns")
                    self.is_fitted_ = False
                    return {"error": "no numeric columns", "skipped": True}
            
            # Handle y fillna for numeric only
            if pd.api.types.is_numeric_dtype(y):
                y_train = y.fillna(y.mean())
            else:
                tprint_warning(f"   ⚠️ Skipping {self.name}: y is not numeric")
                self.is_fitted_ = False
                return {"error": "y not numeric", "skipped": True}
            
            # Fit model
            self.model.fit(X_train, y_train)
            
            # Generate in-sample predictions
            predictions = self.model.predict(X_train)
            
            # Compute prediction errors
            errors = y_train - predictions
            
            # Compute confidence scores (simplified)
            if hasattr(self.model, 'predict'):
                # Use prediction variance as confidence proxy
                confidence = 1.0 / (1.0 + np.var(errors))
            else:
                confidence = 0.5
            
            # Store results
            self.predictions_ = pd.Series(predictions, index=X_train.index)
            self.prediction_errors_ = pd.Series(errors, index=X_train.index)
            self.confidence_scores_ = pd.Series(
                np.full(len(predictions), confidence),
                index=X_train.index
            )
            
            # Compute performance metrics
            mse = mean_squared_error(y_train, predictions)
            mae = mean_absolute_error(y_train, predictions)
            r2 = self.model.score(X_train, y_train)
            
            self.performance_metrics_ = {
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "n_samples": len(X_train),
                "mean_error": errors.mean(),
                "std_error": errors.std()
            }
            
            if self.verbose:
                tprint_success(f"   ✅ Training complete:")
                tprint_info(f"      - MSE: {mse:.6f}")
                tprint_info(f"      - R²: {r2:.4f}")
                tprint_info(f"      - Samples: {len(X_train)}")
            
            return self.performance_metrics_
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Specialist training failed: {e}")
            raise
    
    def predict(
        self,
        X: pd.DataFrame,
        return_confidence: bool = True
    ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Generate predictions.
        
        Args:
            X: Feature matrix
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predictions and optionally confidence scores
        """
        try:
            if self.model is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            # Check our own fitted flag first
            if hasattr(self, 'is_fitted_') and not self.is_fitted_:
                return pd.Series(np.nan, index=X.index)
            
            # Prepare data - handle non-numeric parent columns
            if self.causal_parent in X.columns:
                parent_col = X[self.causal_parent]
                if parent_col.dtype == 'object' or str(parent_col.dtype) == 'string':
                    # Non-numeric parent, cannot predict
                    return pd.Series(np.nan, index=X.index)
                X_pred = X[[self.causal_parent]].fillna(0)
            else:
                X_pred = X.select_dtypes(include=[np.number]).fillna(0)
                if X_pred.empty:
                    return pd.Series(np.nan, index=X.index)
            
            # Check if model is fitted before predicting
            from sklearn.utils.validation import check_is_fitted
            try:
                check_is_fitted(self.model)
            except Exception:
                tprint_warning(f"⚠️ Model for {self.name} is not fitted, returning empty predictions")
                return pd.Series(np.nan, index=X_pred.index)
            
            # Generate predictions
            predictions = self.model.predict(X_pred)
            
            # Store predictions
            self.predictions_ = pd.Series(predictions, index=X_pred.index)
            
            if not return_confidence:
                return self.predictions_
            
            # Generate confidence scores (simplified)
            if hasattr(self.model, 'predict'):
                # Use historical error variance
                if self.prediction_errors_ is not None:
                    error_var = self.prediction_errors_.var()
                    confidence = 1.0 / (1.0 + error_var)
                else:
                    confidence = 0.5
            else:
                confidence = 0.5
            
            confidence_scores = pd.Series(
                np.full(len(predictions), confidence),
                index=X_pred.index
            )
            
            self.confidence_scores_ = confidence_scores
            
            return self.predictions_, self.confidence_scores_
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Prediction failed: {e}")
            raise
    
    def detect_surprise_events(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        surprise_threshold: float = 1.8
    ) -> pd.Series:
        """
        Detect surprise events for this specialist.
        
        Args:
            X: Feature matrix
            y: True values
            surprise_threshold: Threshold for surprise detection
            
        Returns:
            Surprise event indicators
        """
        try:
            if self.verbose:
                tprint_info(f"🔍 Detecting Surprise Events: {self.name}")
            
            # Generate predictions
            predictions, confidence = self.predict(X, return_confidence=True)
            
            # Compute prediction errors
            errors = y - predictions
            
            # Compute rolling statistics
            rolling_mean = errors.rolling(window=20, min_periods=5).mean()
            rolling_std = errors.rolling(window=20, min_periods=5).std()
            
            # Compute z-scores
            z_scores = np.abs(errors - rolling_mean) / (rolling_std + 1e-8)
            
            # Detect surprise events
            surprise_events = z_scores > surprise_threshold
            
            # Store results
            self.surprise_events_ = surprise_events
            
            if self.verbose:
                n_surprises = surprise_events.sum()
                tprint_success(f"   ✅ Surprise detection complete:")
                tprint_info(f"      - Surprise events: {n_surprises}")
                tprint_info(f"      - Surprise rate: {n_surprises/len(surprise_events):.2%}")
            
            return surprise_events
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Surprise detection failed: {e}")
            return pd.Series(False, index=X.index)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of specialist performance.
        
        Returns:
            Summary dictionary
        """
        summary = {
            "name": self.name,
            "causal_parent": self.causal_parent,
            "causal_child": self.causal_child,
            "model_type": self.model_type,
            "is_fitted": self.model is not None,
            "has_predictions": self.predictions_ is not None,
            "has_surprise_events": self.surprise_events_ is not None
        }
        
        if self.performance_metrics_:
            summary.update(self.performance_metrics_)
        
        if self.surprise_events_ is not None:
            summary["surprise_events_count"] = self.surprise_events_.sum()
            summary["surprise_rate"] = self.surprise_events_.mean()
        
        return summary

class CausalSpecialistManager:
    """
    Manages a collection of causal specialists.
    
    Coordinates multiple specialists, tracks their performance,
    and manages consensus and surprise detection.
    """
    
    def __init__(
        self,
        specialists: Optional[List[CausalSpecialist]] = None,
        consensus_threshold: float = 0.6,
        surprise_aggregation: str = "majority",
        verbose: bool = True
    ):
        """
        Initialize Causal Specialist Manager.
        
        Args:
            specialists: List of specialists
            consensus_threshold: Threshold for consensus
            surprise_aggregation: Method for aggregating surprises
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.consensus_threshold = consensus_threshold
        self.surprise_aggregation = surprise_aggregation
        
        # Specialist storage
        self.specialists = specialists or []
        self.specialist_dict_ = {}
        
        # Performance tracking
        self.consensus_predictions_ = None
        self.aggregated_surprises_ = None
        self.manager_metrics_ = {}
        
        # Initialize specialist dictionary
        for specialist in self.specialists:
            self.specialist_dict_[specialist.name] = specialist
    
    def add_specialist(self, specialist: CausalSpecialist) -> None:
        """
        Add a specialist to the manager.
        
        Args:
            specialist: Specialist to add
        """
        self.specialists.append(specialist)
        self.specialist_dict_[specialist.name] = specialist
        
        if self.verbose:
            tprint_info(f"📝 Added specialist: {specialist.name}")
    
    def fit_all_specialists(
        self,
        X: pd.DataFrame,
        y_dict: Dict[str, pd.Series],
        sample_weights: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fit all specialists.
        
        Args:
            X: Feature matrix
            y_dict: Dictionary of targets for each specialist
            sample_weights: Optional sample weights
            
        Returns:
            Training metrics for all specialists
        """
        try:
            if self.verbose:
                tprint_info("🚀 Training All Specialists...")
            
            training_metrics = {}
            
            for specialist in self.specialists:
                if specialist.name in y_dict:
                    y_target = y_dict[specialist.name]
                    weights = sample_weights.get(specialist.name) if sample_weights else None
                    
                    try:
                        metrics = specialist.fit(X, y_target, weights)
                        training_metrics[specialist.name] = metrics
                    except Exception as e:
                        if self.verbose:
                            tprint_warning(f"⚠️ Training failed for {specialist.name}: {e}")
                        training_metrics[specialist.name] = {"error": str(e)}
                else:
                    if self.verbose:
                        tprint_warning(f"⚠️ No target data for {specialist.name}")
            
            self.manager_metrics_["training_metrics"] = training_metrics
            
            if self.verbose:
                n_successful = sum(1 for m in training_metrics.values() if "error" not in m)
                tprint_success(f"✅ Training complete: {n_successful}/{len(self.specialists)} specialists")
            
            return training_metrics
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Specialist training failed: {e}")
            raise
    
    def predict_all_specialists(
        self,
        X: pd.DataFrame,
        return_confidence: bool = True
    ) -> Dict[str, Union[pd.Series, Tuple[pd.Series, pd.Series]]]:
        """
        Generate predictions from all specialists.
        
        Args:
            X: Feature matrix
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary of predictions from all specialists
        """
        try:
            if self.verbose:
                tprint_info("🔮 Generating Predictions from All Specialists...")
            
            predictions = {}
            
            for specialist in self.specialists:
                try:
                    pred = specialist.predict(X, return_confidence)
                    predictions[specialist.name] = pred
                except Exception as e:
                    if self.verbose:
                        tprint_warning(f"⚠️ Prediction failed for {specialist.name}: {e}")
            
            # Compute consensus predictions
            if predictions:
                self.consensus_predictions_ = self._compute_consensus(predictions)
            
            if self.verbose:
                tprint_success(f"✅ Predictions complete: {len(predictions)} specialists")
            
            return predictions
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Specialist prediction failed: {e}")
            raise
    
    def detect_all_surprises(
        self,
        X: pd.DataFrame,
        y_dict: Dict[str, pd.Series],
        surprise_threshold: float = 1.8
    ) -> Dict[str, pd.Series]:
        """
        Detect surprise events from all specialists.
        
        Args:
            X: Feature matrix
            y_dict: Dictionary of true values
            surprise_threshold: Threshold for surprise detection
            
        Returns:
            Dictionary of surprise events from all specialists
        """
        try:
            if self.verbose:
                tprint_info("🚨 Detecting Surprise Events from All Specialists...")
            
            surprise_events = {}
            
            for specialist in self.specialists:
                if specialist.name in y_dict:
                    try:
                        surprises = specialist.detect_surprise_events(
                            X, y_dict[specialist.name], surprise_threshold
                        )
                        surprise_events[specialist.name] = surprises
                    except Exception as e:
                        if self.verbose:
                            tprint_warning(f"⚠️ Surprise detection failed for {specialist.name}: {e}")
            
            # Aggregate surprise events
            if surprise_events:
                self.aggregated_surprises_ = self._aggregate_surprises(surprise_events)
            
            if self.verbose:
                tprint_success(f"✅ Surprise detection complete: {len(surprise_events)} specialists")
            
            return surprise_events
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Surprise detection failed: {e}")
            raise
    
    def _compute_consensus(
        self,
        predictions: Dict[str, Union[pd.Series, Tuple[pd.Series, pd.Series]]]
    ) -> pd.DataFrame:
        """
        Compute consensus predictions from all specialists.
        
        Args:
            predictions: Dictionary of predictions
            
        Returns:
            Consensus predictions DataFrame
        """
        try:
            # Extract predictions (ignore confidence for consensus)
            pred_values = {}
            
            for name, pred in predictions.items():
                if isinstance(pred, tuple):
                    pred_values[name] = pred[0]
                else:
                    pred_values[name] = pred
            
            # Create DataFrame
            consensus_df = pd.DataFrame(pred_values)
            
            # Compute consensus statistics
            consensus_df["consensus_mean"] = consensus_df.mean(axis=1)
            consensus_df["consensus_std"] = consensus_df.std(axis=1)
            consensus_df["n_agreeing"] = (consensus_df.iloc[:, :-2] > 0).sum(axis=1)  # Assuming binary
            
            return consensus_df
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Consensus computation failed: {e}")
            return pd.DataFrame()
    
    def _aggregate_surprises(
        self,
        surprise_events: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Aggregate surprise events from all specialists.
        
        Args:
            surprise_events: Dictionary of surprise events
            
        Returns:
            Aggregated surprise events DataFrame
        """
        try:
            # Create DataFrame
            surprise_df = pd.DataFrame(surprise_events)
            
            # Aggregate based on method
            if self.surprise_aggregation == "majority":
                surprise_df["majority_surprise"] = (surprise_df.sum(axis=1) > len(surprise_df) / 2).astype(int)
            elif self.surprise_aggregation == "any":
                surprise_df["any_surprise"] = (surprise_df.sum(axis=1) > 0).astype(int)
            elif self.surprise_aggregation == "weighted":
                # Weight by specialist performance (simplified)
                weights = np.ones(len(surprise_df.columns))
                surprise_df["weighted_surprise"] = (surprise_df * weights).sum(axis=1) / weights.sum()
            else:
                surprise_df["sum_surprise"] = surprise_df.sum(axis=1)
            
            return surprise_df
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Surprise aggregation failed: {e}")
            return pd.DataFrame()
    
    def get_specialist_consensus(
        self,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Get specialist consensus information.
        
        Args:
            threshold: Consensus threshold
            
        Returns:
            Consensus information
        """
        if self.consensus_predictions_ is None:
            return pd.DataFrame()
        
        consensus_df = self.consensus_predictions_.copy()
        
        if threshold is not None:
            consensus_df["high_consensus"] = (
                consensus_df["n_agreeing"] / len(self.specialists) >= threshold
            ).astype(int)
        
        return consensus_df
    
    def get_manager_summary(self) -> Dict[str, Any]:
        """
        Get summary of specialist manager.
        
        Returns:
            Manager summary
        """
        summary = {
            "n_specialists": len(self.specialists),
            "specialist_names": [s.name for s in self.specialists],
            "consensus_threshold": self.consensus_threshold,
            "surprise_aggregation": self.surprise_aggregation,
            "has_consensus_predictions": self.consensus_predictions_ is not None,
            "has_aggregated_surprises": self.aggregated_surprises_ is not None
        }
        
        # Add specialist summaries
        specialist_summaries = {}
        for specialist in self.specialists:
            specialist_summaries[specialist.name] = specialist.get_summary()
        
        summary["specialist_summaries"] = specialist_summaries
        
        return summary

# Convenience functions
def create_causal_specialists(
    causal_graph: Dict[str, List[str]],
    model_type: str = "linear",
    **kwargs
) -> CausalSpecialistManager:
    """
    Create causal specialists from causal graph.
    
    Args:
        causal_graph: Causal graph
        model_type: Model type for specialists
        **kwargs: Additional parameters
        
    Returns:
        CausalSpecialistManager instance
    """
    specialists = []
    
    for child, parents in causal_graph.items():
        for parent in parents:
            specialist_name = f"{parent}_to_{child}"
            specialist = CausalSpecialist(
                name=specialist_name,
                causal_parent=parent,
                causal_child=child,
                model_type=model_type,
                **kwargs
            )
            specialists.append(specialist)
    
    return CausalSpecialistManager(specialists, **kwargs)

def quick_specialist_training(
    X: pd.DataFrame,
    y_dict: Dict[str, pd.Series],
    causal_graph: Optional[Dict[str, List[str]]] = None,
    **kwargs
) -> CausalSpecialistManager:
    """
    Quick specialist training.
    
    Args:
        X: Feature matrix
        y_dict: Target dictionary
        causal_graph: Causal graph
        **kwargs: Additional parameters
        
    Returns:
        Trained CausalSpecialistManager
    """
    if causal_graph:
        manager = create_causal_specialists(causal_graph, **kwargs)
    else:
        manager = CausalSpecialistManager(**kwargs)
    
    manager.fit_all_specialists(X, y_dict)
    return manager


@dataclass
class LiquidationSpecialist(CausalSpecialist):
    """
    Specialist focused on Liquidation Squeezes.
    
    Logic:
    - Identifies proximity to Liquidation Proxy Levels (Pivot + ATR).
    - Checks for Volume Stress (RVS > Threshold).
    - Triggers 'Squeeze' signal if Price dips into Proxy Zone with High Stress.
    """
    
    def __init__(
        self,
        name: str = "Liquidation_Specialist",
        rvs_threshold: float = 2.0,
        **kwargs
    ):
        super().__init__(name, causal_parent="liquidation_risk_long", causal_child="returns", **kwargs)
        self.rvs_threshold = rvs_threshold
        self.confidence_scores_ = None # Initialize 
        
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Fit method (Standard Linear Model on composite feature).
        
        We primarily use the 'liquidation_risk_long' composite feature created in feature generation,
        but we can also look at individual components if available.
        """
        if self.verbose:
            tprint_info(f"🌊 Training Liquidation Specialist: {self.name}")
        
        # We use the standard fit from parent, but ensure we target the composite risk score
        # passing 'liquidation_risk_long' as the parent feature.
        return super().fit(X, y, sample_weights)

    def detect_squeeze(self, X: pd.DataFrame) -> pd.Series:
        """
        Detect Squeeze Events based on Hard Logic rules.
        
        Rule:
        - Price < Long_Proxy (implied by positive dist_to_long_proxy if normalized correctly, 
          OR check 'in_long_liq_zone' flag if available).
        - RVS > Threshold.
        """
        # We can reconstruct logic from raw features or use the composite
        # Composite 'liquidation_risk_long' = in_zone * high_stress
        if 'liquidation_risk_long' in X.columns:
            # If feature gen did the heavy lifting
            return (X['liquidation_risk_long'] > 0).astype(int)
            
        # Fallback reconstruction
        squeeze_signal = pd.Series(0, index=X.index)
        
        if 'dist_to_long_proxy' in X.columns and 'relative_volume_stress' in X.columns:
            # dist_to_long_proxy = (Price - Proxy). 
            # If Price < Proxy, dist is negative.
            # Wait, feature gen said: "For Long Squeeze (price drops to proxy): Distance is positive if Price > Proxy"
            # mtf_feature_generation.py: dist_to_long_proxy = (close - proxy_long) / atr
            # So if Price < Proxy, dist is NEGATIVE.
            
            # BUT, the user prompt said: "Price < Long_Proxy_Level".
            # So we look for dist_to_long_proxy < 0.
            
            in_zone = X['dist_to_long_proxy'] < 0
            high_stress = X['relative_volume_stress'] > self.rvs_threshold
            
            squeeze_signal = (in_zone & high_stress).astype(int)
            
        return squeeze_signal

    def predict(self, X: pd.DataFrame, return_confidence: bool = True) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Predict returns, boosting them if a Squeeze is detected.
        """
        # Base prediction from linear model
        base_pred = super().predict(X, return_confidence=False)
        
        # Add Squeeze Boost
        squeeze = self.detect_squeeze(X)
        
        if return_confidence:
            confidence = pd.Series(0.5, index=X.index)
            # Boost confidence where squeeze is detected
            confidence[squeeze == 1] = 0.9
            # Use base confidence for others?
            if self.confidence_scores_ is not None and not self.confidence_scores_.empty:
                 # Align base confidence
                 base_conf = self.confidence_scores_.reindex(X.index).fillna(0.5)
                 confidence = np.maximum(confidence, base_conf)
                 
            return base_pred, confidence
        
        return base_pred

