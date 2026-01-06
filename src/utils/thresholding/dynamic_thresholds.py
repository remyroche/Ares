"""
Dynamic Thresholding System for Trading Models

This module implements adaptive threshold calculation to address zero trade execution
issues by adjusting confidence thresholds based on prediction distribution,
volatility, and market conditions.

Key Features:
- Adaptive threshold calculation based on prediction statistics
- Volatility-adjusted thresholds
- Confidence interval-based thresholding
- Minimum trade frequency enforcement
- Market regime-aware thresholds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

from src.utils.logger import system_logger

logger = system_logger.getChild("DynamicThresholds")


class DynamicThresholdCalculator:
    """
    Dynamic threshold calculator for trading models.
    
    Addresses zero trade execution by adapting thresholds based on:
    - Prediction distribution characteristics
    - Recent volatility levels
    - Historical trade frequency
    - Market regime conditions
    """
    
    def __init__(
        self,
        base_threshold: float = 0.6,
        min_threshold: float = 0.5,
        max_threshold: float = 0.85,
        lookback_window: int = 100,
        volatility_window: int = 20,
        min_trades_per_period: int = 5,
        adjustment_factor: float = 0.1,
    ):
        """
        Initialize dynamic threshold calculator.
        
        Args:
            base_threshold: Starting threshold value
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
            lookback_window: Window for prediction statistics
            volatility_window: Window for volatility calculation
            min_trades_per_period: Minimum trades to maintain frequency
            adjustment_factor: Step size for threshold adjustments
        """
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.lookback_window = lookback_window
        self.volatility_window = volatility_window
        self.min_trades_per_period = min_trades_per_period
        self.adjustment_factor = adjustment_factor
        
        # State tracking
        self.current_threshold = base_threshold
        self.trade_history = []
        self.prediction_history = []
        self.volatility_history = []
        
    def calculate_prediction_statistics(self, predictions: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive prediction statistics."""
        if len(predictions) < 10:
            return {"mean": 0.5, "std": 0.1, "skew": 0.0, "kurt": 0.0}
        
        clean_preds = predictions.dropna()
        if len(clean_preds) == 0:
            return {"mean": 0.5, "std": 0.1, "skew": 0.0, "kurt": 0.0}
        
        return {
            "mean": float(clean_preds.mean()),
            "std": float(clean_preds.std()),
            "skew": float(stats.skew(clean_preds)),
            "kurt": float(stats.kurtosis(clean_preds)),
            "range": float(clean_preds.max() - clean_preds.min()),
            "percentile_75": float(clean_preds.quantile(0.75)),
            "percentile_25": float(clean_preds.quantile(0.25)),
        }
    
    def calculate_volatility_adjustment(self, returns: pd.Series) -> float:
        """Calculate volatility-based threshold adjustment."""
        if len(returns) < self.volatility_window:
            return 0.0
        
        recent_vol = returns.tail(self.volatility_window).std()
        if len(self.volatility_history) > 0:
            avg_vol = np.mean(self.volatility_history[-50:])  # Long-term average
            if recent_vol > avg_vol * 1.5:  # High volatility
                return -self.adjustment_factor * 0.5  # Lower threshold
            elif recent_vol < avg_vol * 0.5:  # Low volatility
                return self.adjustment_factor * 0.5  # Raise threshold
        
        self.volatility_history.append(recent_vol)
        return 0.0
    
    def calculate_distribution_adjustment(self, pred_stats: Dict[str, float]) -> float:
        """Calculate adjustment based on prediction distribution."""
        adjustment = 0.0
        
        # If predictions are clustered around 0.5, lower threshold
        if pred_stats["std"] < 0.05:
            adjustment -= self.adjustment_factor * 2
        elif pred_stats["std"] < 0.1:
            adjustment -= self.adjustment_factor
        
        # If predictions are skewed, adjust accordingly
        if abs(pred_stats["skew"]) > 0.5:
            adjustment -= self.adjustment_factor * 0.5
        
        # If range is too narrow, lower threshold
        if pred_stats["range"] < 0.3:
            adjustment -= self.adjustment_factor
        
        return adjustment
    
    def calculate_frequency_adjustment(self, recent_trades: int) -> float:
        """Calculate adjustment based on trade frequency."""
        if recent_trades < self.min_trades_per_period:
            return -self.adjustment_factor * 2  # Aggressively lower threshold
        elif recent_trades > self.min_trades_per_period * 3:
            return self.adjustment_factor  # Raise threshold if too many trades
        return 0.0
    
    def calculate_confidence_interval_threshold(
        self, predictions: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """Calculate threshold based on confidence intervals."""
        clean_preds = predictions.dropna()
        if len(clean_preds) < 30:
            return self.current_threshold
        
        # Calculate confidence interval for predictions
        ci_lower, ci_upper = stats.norm.interval(
            confidence_level, 
            loc=clean_preds.mean(), 
            scale=clean_preds.std()
        )
        
        # Use upper bound of CI as threshold if predictions are above 0.5
        if ci_upper > 0.5:
            return max(self.min_threshold, min(self.max_threshold, ci_upper))
        
        return self.current_threshold
    
    def update_threshold(
        self,
        predictions: pd.Series,
        returns: Optional[pd.Series] = None,
        recent_trades: int = 0,
        market_regime: str = "neutral",
    ) -> float:
        """
        Update threshold based on current conditions.
        
        Args:
            predictions: Recent model predictions
            returns: Recent returns for volatility calculation
            recent_trades: Number of trades in recent period
            market_regime: Current market regime (trending/ranging/volatile)
        
        Returns:
            Updated threshold value
        """
        # Calculate prediction statistics
        pred_stats = self.calculate_prediction_statistics(predictions)
        
        # Calculate adjustments
        vol_adjustment = 0.0
        if returns is not None:
            vol_adjustment = self.calculate_volatility_adjustment(returns)
        
        dist_adjustment = self.calculate_distribution_adjustment(pred_stats)
        freq_adjustment = self.calculate_frequency_adjustment(recent_trades)
        
        # Regime-based adjustments
        regime_adjustment = 0.0
        if market_regime == "trending":
            regime_adjustment = -self.adjustment_factor * 0.3
        elif market_regime == "volatile":
            regime_adjustment = -self.adjustment_factor * 0.5
        elif market_regime == "ranging":
            regime_adjustment = self.adjustment_factor * 0.2
        
        # Calculate new threshold
        total_adjustment = vol_adjustment + dist_adjustment + freq_adjustment + regime_adjustment
        new_threshold = self.current_threshold + total_adjustment
        
        # Apply bounds
        new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))
        
        # Update state
        self.prediction_history.append(pred_stats)
        self.current_threshold = new_threshold
        
        logger.info(
            f"Threshold updated: {new_threshold:.3f} "
            f"(vol: {vol_adjustment:+.3f}, dist: {dist_adjustment:+.3f}, "
            f"freq: {freq_adjustment:+.3f}, regime: {regime_adjustment:+.3f})"
        )
        
        return new_threshold
    
    def get_dynamic_thresholds(
        self,
        predictions: pd.Series,
        returns: Optional[pd.Series] = None,
        method: str = "adaptive",
        n_thresholds: int = 4,
    ) -> List[float]:
        """
        Generate multiple dynamic thresholds.
        
        Args:
            predictions: Model predictions
            returns: Returns for volatility calculation
            method: Threshold generation method
            n_thresholds: Number of thresholds to generate
        
        Returns:
            List of threshold values
        """
        if method == "adaptive":
            base_thresh = self.update_threshold(predictions, returns)
            # Generate thresholds around the adaptive base
            return [
                max(self.min_threshold, base_thresh - 0.1),
                base_thresh,
                min(self.max_threshold, base_thresh + 0.05),
                min(self.max_threshold, base_thresh + 0.1),
            ]
        
        elif method == "percentile":
            clean_preds = predictions.dropna()
            if len(clean_preds) < 30:
                return [0.5, 0.6, 0.7, 0.8]
            
            # Use percentiles of prediction distribution
            percentiles = [60, 70, 80, 90]
            thresholds = []
            for p in percentiles:
                thresh = clean_preds.quantile(p / 100)
                thresholds.append(max(self.min_threshold, min(self.max_threshold, thresh)))
            return thresholds
        
        elif method == "confidence_interval":
            ci_thresh = self.calculate_confidence_interval_threshold(predictions)
            return [
                max(self.min_threshold, ci_thresh - 0.1),
                ci_thresh,
                min(self.max_threshold, ci_thresh + 0.05),
                min(self.max_threshold, ci_thresh + 0.1),
            ]
        
        else:  # fallback to static
            return [0.5, 0.6, 0.7, 0.8]


def calculate_dynamic_thresholds_batch(
    predictions: pd.Series,
    returns: Optional[pd.Series] = None,
    method: str = "adaptive",
    base_threshold: float = 0.6,
    min_trades_target: int = 5,
) -> List[float]:
    """
    Convenience function for batch threshold calculation.
    
    Args:
        predictions: Model predictions
        returns: Optional returns for volatility adjustment
        method: Threshold calculation method
        base_threshold: Base threshold for adaptive methods
        min_trades_target: Target minimum trades per period
    
    Returns:
        List of dynamic thresholds
    """
    calculator = DynamicThresholdCalculator(
        base_threshold=base_threshold,
        min_trades_per_period=min_trades_target
    )
    
    return calculator.get_dynamic_thresholds(
        predictions=predictions,
        returns=returns,
        method=method
    )


def analyze_prediction_distribution(predictions: pd.Series) -> Dict[str, Union[float, str]]:
    """
    Analyze prediction distribution for threshold optimization.
    
    Args:
        predictions: Model predictions
    
    Returns:
        Analysis results with recommendations
    """
    clean_preds = predictions.dropna()
    if len(clean_preds) < 10:
        return {"error": "Insufficient data for analysis"}
    
    stats_dict = {
        "count": len(clean_preds),
        "mean": float(clean_preds.mean()),
        "std": float(clean_preds.std()),
        "min": float(clean_preds.min()),
        "max": float(clean_preds.max()),
        "range": float(clean_preds.max() - clean_preds.min()),
        "skew": float(stats.skew(clean_preds)),
        "kurt": float(stats.kurtosis(clean_preds)),
    }
    
    # Distribution analysis
    above_60 = (clean_preds >= 0.6).sum()
    above_70 = (clean_preds >= 0.7).sum()
    above_80 = (clean_preds >= 0.8).sum()
    
    stats_dict.update({
        "above_60_pct": above_60 / len(clean_preds),
        "above_70_pct": above_70 / len(clean_preds),
        "above_80_pct": above_80 / len(clean_preds),
    })
    
    # Recommendations
    recommendations = []
    
    if stats_dict["std"] < 0.05:
        recommendations.append("Low prediction variance - consider lowering thresholds")
    
    if stats_dict["above_60_pct"] < 0.05:
        recommendations.append("Few predictions above 60% - dynamic thresholding needed")
    
    if stats_dict["range"] < 0.3:
        recommendations.append("Narrow prediction range - check model calibration")
    
    if abs(stats_dict["skew"]) > 1.0:
        recommendations.append("Highly skewed predictions - consider rebalancing")
    
    stats_dict["recommendations"] = recommendations
    
    return stats_dict
