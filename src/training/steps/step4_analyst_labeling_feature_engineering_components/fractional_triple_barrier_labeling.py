# src/training/steps/step4_analyst_labeling_feature_engineering_components/fractional_triple_barrier_labeling.py

"""Fractional Triple Barrier Labeling for enhanced model training.
Implements continuous labeling instead of binary classification for better
gradient flow and more nuanced risk management.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple

from src.utils.centralized_decorators import (
    guard_dataframe_nulls,
    handle_errors,
    with_tracing_span,
)
from src.utils.logger import get_logger
from .optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
import copy
from src.core.decorators import handles_errors

class FractionalTripleBarrierLabeling:
    """Enhanced triple barrier labeling with fractional (continuous) labels.
    
    Instead of binary labels (1/-1), generates continuous values based on:
    - Distance to barriers
    - Time decay factors
    - Volatility normalization
    - Regime-specific scaling
    """

    def __init__(
        self,
        profit_take_multiplier: float = 0.002,
        stop_loss_multiplier: float = 0.001,
        time_barrier_minutes: int = 30,
        max_lookahead: int = 100,
        fractional_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize fractional triple barrier labeling.
        
        Args:
            profit_take_multiplier: Multiplier for profit take barrier
            stop_loss_multiplier: Multiplier for stop loss barrier
            time_barrier_minutes: Time barrier in minutes
            max_lookahead: Maximum lookahead points
            fractional_config: Configuration for fractional labeling
        """
        self.base_labeler = OptimizedTripleBarrierLabeling(
            profit_take_multiplier=profit_take_multiplier,
            stop_loss_multiplier=stop_loss_multiplier,
            time_barrier_minutes=time_barrier_minutes,
            max_lookahead=max_lookahead,
            binary_classification=False,  # We want all samples for fractional processing
        )
        
        # Default fractional configuration
        self.fractional_config = fractional_config or {
            "enable_distance_scaling": True,
            "enable_time_decay": True,
            "enable_volatility_normalization": True,
            "enable_regime_scaling": False,
            "distance_weight": 0.4,
            "time_weight": 0.3,
            "volatility_weight": 0.3,
            "min_confidence_threshold": 0.1,
            "max_confidence_threshold": 0.95,
        }
        
        self.logger = get_logger("FractionalTripleBarrierLabeling")

    @handles_errors(fallback=pd.DataFrame())
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    @with_tracing_span("FractionalTripleBarrier.apply", log_args=False)
    def apply_fractional_triple_barrier_labeling(
        self,
        data: pd.DataFrame,
        regime_labels: Optional[np.ndarray] = None,
        volatility_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Apply fractional triple barrier labeling.
        
        Args:
            data: OHLCV data
            regime_labels: Optional regime labels for regime-specific scaling
            volatility_series: Optional volatility series for normalization
            
        Returns:
            DataFrame with fractional labels and confidence scores
        """
        self.logger.info("Applying fractional triple barrier labeling")
        
        # Step 1: Get base binary labels
        labeled_data = self.base_labeler.apply_triple_barrier_labeling_vectorized(data)
        
        # Step 2: Calculate fractional components
        fractional_components = self._calculate_fractional_components(
            labeled_data, regime_labels, volatility_series
        )
        
        # Step 3: Combine into final fractional labels
        final_labels = self._combine_fractional_components(fractional_components)
        
        # Step 4: Add confidence scores
        confidence_scores = self._calculate_confidence_scores(
            labeled_data, fractional_components
        )
        
        # Step 5: Update the dataframe
        labeled_data["fractional_label"] = final_labels
        labeled_data["confidence_score"] = confidence_scores
        labeled_data["barrier_distance"] = fractional_components["distance_score"]
        labeled_data["time_decay_score"] = fractional_components["time_score"]
        labeled_data["volatility_score"] = fractional_components["volatility_score"]
        
        # Step 6: Filter by confidence threshold
        min_confidence = self.fractional_config["min_confidence_threshold"]
        filtered_data = labeled_data[confidence_scores >= min_confidence].copy()
        
        self.logger.info(f"Fractional labeling complete: {len(filtered_data)}/{len(labeled_data)} samples retained")
        
        return filtered_data

    def _calculate_fractional_components(
        self,
        labeled_data: pd.DataFrame,
        regime_labels: Optional[np.ndarray] = None,
        volatility_series: Optional[pd.Series] = None,
    ) -> Dict[str, np.ndarray]:
        """Calculate individual fractional components."""
        n = len(labeled_data)
        components = {
            "distance_score": np.zeros(n),
            "time_score": np.zeros(n),
            "volatility_score": np.zeros(n),
        }
        
        # Distance-based scoring
        if self.fractional_config["enable_distance_scaling"]:
            components["distance_score"] = self._calculate_distance_scores(labeled_data)
        
        # Time decay scoring
        if self.fractional_config["enable_time_decay"]:
            components["time_score"] = self._calculate_time_decay_scores(labeled_data)
        
        # Volatility normalization
        if self.fractional_config["enable_volatility_normalization"]:
            components["volatility_score"] = self._calculate_volatility_scores(
                labeled_data, volatility_series
            )
        
        return components

    def _calculate_distance_scores(self, labeled_data: pd.DataFrame) -> np.ndarray:
        """Calculate distance-based fractional scores."""
        scores = np.zeros(len(labeled_data))
        
        # For profit hits: score based on how quickly profit was achieved
        profit_hits = labeled_data["label"] == 1
        if profit_hits.any():
            profit_pcts = labeled_data.loc[profit_hits, "potential_profit_pct"]
            # Normalize by target profit
            target_profit = self.base_labeler.profit_take_multiplier
            scores[profit_hits] = np.clip(profit_pcts / target_profit, 0, 1)
        
        # For stop loss hits: score based on how quickly stop was hit
        stop_hits = labeled_data["label"] == -1
        if stop_hits.any():
            stop_pcts = labeled_data.loc[stop_hits, "potential_profit_pct"]
            # Normalize by target loss (negative)
            target_loss = -self.base_labeler.stop_loss_multiplier
            scores[stop_hits] = np.clip(stop_pcts / target_loss, 0, 1)
        
        return scores

    def _calculate_time_decay_scores(self, labeled_data: pd.DataFrame) -> np.ndarray:
        """Calculate time decay scores based on how quickly barriers were hit."""
        scores = np.zeros(len(labeled_data))
        
        # This would require tracking the actual time/bar index when barriers were hit
        # For now, use a simplified approach based on profit percentages
        # Higher profit percentages (achieved quickly) get higher scores
        
        profit_hits = labeled_data["label"] == 1
        stop_hits = labeled_data["label"] == -1
        
        if profit_hits.any():
            # Quick profit hits get higher scores
            profit_pcts = labeled_data.loc[profit_hits, "potential_profit_pct"]
            scores[profit_hits] = np.clip(profit_pcts / self.base_labeler.profit_take_multiplier, 0, 1)
        
        if stop_hits.any():
            # Quick stop hits get lower scores (worse performance)
            stop_pcts = labeled_data.loc[stop_hits, "potential_profit_pct"]
            scores[stop_hits] = np.clip(stop_pcts / self.base_labeler.stop_loss_multiplier, 0, 1)
        
        return scores

    def _calculate_volatility_scores(
        self, 
        labeled_data: pd.DataFrame, 
        volatility_series: Optional[pd.Series] = None
    ) -> np.ndarray:
        """Calculate volatility-normalized scores."""
        scores = np.zeros(len(labeled_data))
        
        if volatility_series is not None:
            # Normalize by volatility - higher volatility periods get adjusted scores
            volatility_norm = volatility_series / volatility_series.rolling(20).mean()
            scores = np.clip(1 / volatility_norm, 0.5, 2.0)  # Bounded normalization
        else:
            # Use simple rolling volatility from price data
            returns = labeled_data["close"].pct_change()
            rolling_vol = returns.rolling(20).std()
            vol_norm = rolling_vol / rolling_vol.rolling(100).mean()
            scores = np.clip(1 / vol_norm, 0.5, 2.0)
        
        return scores

    def _combine_fractional_components(
        self, 
        components: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Combine fractional components into final labels."""
        weights = {
            "distance": self.fractional_config["distance_weight"],
            "time": self.fractional_config["time_weight"],
            "volatility": self.fractional_config["volatility_weight"],
        }
        
        # Weighted combination
        final_labels = (
            weights["distance"] * components["distance_score"] +
            weights["time"] * components["time_score"] +
            weights["volatility"] * components["volatility_score"]
        )
        
        # Scale to [-1, 1] range
        final_labels = np.clip(final_labels, -1, 1)
        
        return final_labels

    def _calculate_confidence_scores(
        self, 
        labeled_data: pd.DataFrame, 
        components: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Calculate confidence scores for fractional labels."""
        # Base confidence from barrier hit certainty
        base_confidence = np.abs(labeled_data["label"])
        
        # Additional confidence from component consistency
        component_std = np.std([
            components["distance_score"],
            components["time_score"],
            components["volatility_score"]
        ], axis=0)
        
        # Higher consistency (lower std) means higher confidence
        consistency_confidence = 1 - component_std
        
        # Combine base and consistency confidence
        final_confidence = 0.7 * base_confidence + 0.3 * consistency_confidence
        
        # Apply thresholds
        min_conf = self.fractional_config["min_confidence_threshold"]
        max_conf = self.fractional_config["max_confidence_threshold"]
        final_confidence = np.clip(final_confidence, min_conf, max_conf)
        
        return final_confidence

    def get_fractional_label_statistics(self, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about fractional labels."""
        stats = {
            "total_samples": len(labeled_data),
            "fractional_label_mean": labeled_data["fractional_label"].mean(),
            "fractional_label_std": labeled_data["fractional_label"].std(),
            "confidence_mean": labeled_data["confidence_score"].mean(),
            "confidence_std": labeled_data["confidence_score"].std(),
            "positive_labels": (labeled_data["fractional_label"] > 0).sum(),
            "negative_labels": (labeled_data["fractional_label"] < 0).sum(),
            "neutral_labels": (labeled_data["fractional_label"] == 0).sum(),
        }
        
        return stats