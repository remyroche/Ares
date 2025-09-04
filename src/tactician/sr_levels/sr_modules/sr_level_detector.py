"""Support/Resistance Level Detection Module."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.core.decorators import handles_errors
from src.utils.logger import system_logger

# DBSCAN clustering for S/R level analysis
try:
    from sklearn.cluster import DBSCAN
    DBSCAN_AVAILABLE = True
except ImportError:
    DBSCAN_AVAILABLE = False
    print("Warning: sklearn not available, DBSCAN clustering will be disabled")


class SRLevelDetector:
    """Detects support and resistance levels using various methods."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize SR level detector."""
        self.config = config
        self.logger = system_logger.getChild("SRLevelDetector")
        
        # Configuration
        self.sr_config = config.get("sr_breakout_predictor", {})
        self.sr_detection_method = self.sr_config.get("sr_detection_method", "fractal")
        self.min_sr_strength = self.sr_config.get("min_sr_strength", 0.3)
        self.max_sr_levels = self.sr_config.get("max_sr_levels", 10)
        self.sr_lookback_periods = self.sr_config.get("sr_lookback_periods", 100)
        self.volume_weight = self.sr_config.get("volume_weight", 0.7)
        self.price_weight = self.sr_config.get("price_weight", 0.3)
        self.atr_multiplier = self.sr_config.get("atr_multiplier", 1.5)
        
        # Clustering parameters
        self.enable_clustering = self.sr_config.get("enable_clustering", DBSCAN_AVAILABLE)
        self.clustering_eps = self.sr_config.get("clustering_eps", 0.001)
        self.clustering_min_samples = self.sr_config.get("clustering_min_samples", 3)
        
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=[],
        context="detect support/resistance levels"
    )
    def detect_sr_levels(
        self, 
        market_data: pd.DataFrame, 
        current_price: float
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect support and resistance levels using configured method.
        
        Args:
            market_data: Market data DataFrame
            current_price: Current price
            
        Returns:
            Dictionary containing support and resistance levels
        """
        try:
            # Validate data
            if market_data.empty or len(market_data) < self.sr_lookback_periods:
                self.logger.warning("Insufficient data for S/R detection")
                return {"support": [], "resistance": []}
            
            # Use configured detection method
            if self.sr_detection_method == "fractal":
                levels = self._detect_fractal_levels(market_data)
            elif self.sr_detection_method == "volume_weighted":
                levels = self._detect_volume_weighted_levels(market_data)
            elif self.sr_detection_method == "pivot":
                levels = self._detect_pivot_levels(market_data)
            else:
                # Default to fractal method
                levels = self._detect_fractal_levels(market_data)
                
            # Separate into support and resistance
            support_levels = [
                level for level in levels 
                if level["price"] < current_price
            ]
            resistance_levels = [
                level for level in levels 
                if level["price"] >= current_price
            ]
            
            # Sort and limit levels
            support_levels = sorted(
                support_levels, 
                key=lambda x: x["price"], 
                reverse=True
            )[:self.max_sr_levels]
            
            resistance_levels = sorted(
                resistance_levels, 
                key=lambda x: x["price"]
            )[:self.max_sr_levels]
            
            return {
                "support": support_levels,
                "resistance": resistance_levels
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting S/R levels: {e}")
            return {"support": [], "resistance": []}
    
    def _detect_fractal_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect S/R levels using fractal analysis."""
        levels = []
        
        # Get data
        highs = market_data["high"].values
        lows = market_data["low"].values
        volumes = market_data["volume"].values
        
        # Fractal window
        window = 5
        
        # Find fractal highs (resistance)
        for i in range(window, len(highs) - window):
            is_fractal_high = True
            for j in range(1, window + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_fractal_high = False
                    break
                    
            if is_fractal_high:
                strength = self._calculate_level_strength(
                    market_data, i, "resistance"
                )
                if strength >= self.min_sr_strength:
                    levels.append({
                        "price": float(highs[i]),
                        "type": "resistance",
                        "strength": strength,
                        "volume": float(volumes[i]),
                        "index": i,
                        "method": "fractal"
                    })
        
        # Find fractal lows (support)
        for i in range(window, len(lows) - window):
            is_fractal_low = True
            for j in range(1, window + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_fractal_low = False
                    break
                    
            if is_fractal_low:
                strength = self._calculate_level_strength(
                    market_data, i, "support"
                )
                if strength >= self.min_sr_strength:
                    levels.append({
                        "price": float(lows[i]),
                        "type": "support",
                        "strength": strength,
                        "volume": float(volumes[i]),
                        "index": i,
                        "method": "fractal"
                    })
        
        # Apply clustering if enabled
        if self.enable_clustering and levels:
            levels = self._cluster_levels(levels)
            
        return levels
    
    def _detect_volume_weighted_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect S/R levels using volume-weighted analysis."""
        levels = []
        
        # Calculate VWAP levels
        typical_price = (market_data["high"] + market_data["low"] + market_data["close"]) / 3
        cumulative_volume = market_data["volume"].cumsum()
        cumulative_tp_volume = (typical_price * market_data["volume"]).cumsum()
        vwap = cumulative_tp_volume / cumulative_volume
        
        # Find significant volume spikes
        volume_mean = market_data["volume"].rolling(20).mean()
        volume_std = market_data["volume"].rolling(20).std()
        volume_threshold = volume_mean + 2 * volume_std
        
        significant_indices = market_data[market_data["volume"] > volume_threshold].index
        
        for idx in significant_indices:
            if idx < len(market_data) - 1:
                price = float(typical_price.iloc[idx])
                strength = self._calculate_volume_strength(market_data, idx)
                
                if strength >= self.min_sr_strength:
                    levels.append({
                        "price": price,
                        "type": "both",  # Volume levels can act as both S/R
                        "strength": strength,
                        "volume": float(market_data["volume"].iloc[idx]),
                        "index": idx,
                        "method": "volume_weighted"
                    })
        
        return levels
    
    def _detect_pivot_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect S/R levels using pivot points."""
        levels = []
        
        # Get latest OHLC
        high = market_data["high"].iloc[-1]
        low = market_data["low"].iloc[-1]
        close = market_data["close"].iloc[-1]
        
        # Calculate pivot point
        pivot = (high + low + close) / 3
        
        # Calculate support and resistance levels
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        # Add levels with default strength
        pivot_levels = [
            {"price": float(r3), "type": "resistance", "strength": 0.5, "level": "R3"},
            {"price": float(r2), "type": "resistance", "strength": 0.7, "level": "R2"},
            {"price": float(r1), "type": "resistance", "strength": 0.9, "level": "R1"},
            {"price": float(pivot), "type": "both", "strength": 1.0, "level": "Pivot"},
            {"price": float(s1), "type": "support", "strength": 0.9, "level": "S1"},
            {"price": float(s2), "type": "support", "strength": 0.7, "level": "S2"},
            {"price": float(s3), "type": "support", "strength": 0.5, "level": "S3"},
        ]
        
        for level in pivot_levels:
            level.update({
                "volume": 0,  # Pivot levels don't have associated volume
                "index": len(market_data) - 1,
                "method": "pivot"
            })
            levels.append(level)
            
        return levels
    
    def _calculate_level_strength(
        self, 
        market_data: pd.DataFrame, 
        index: int, 
        level_type: str
    ) -> float:
        """Calculate strength of a S/R level."""
        try:
            # Get relevant data
            volumes = market_data["volume"].values
            prices = market_data["close"].values
            
            # Volume strength
            volume_strength = volumes[index] / np.max(volumes[-self.sr_lookback_periods:])
            
            # Touch count (how many times price touched this level)
            level_price = prices[index]
            price_range = np.abs(prices - level_price) / level_price
            touches = np.sum(price_range < 0.001)  # Within 0.1% of level
            touch_strength = min(touches / 10, 1.0)  # Normalize to max 1.0
            
            # Recency factor (more recent levels are stronger)
            recency = (index - (len(market_data) - self.sr_lookback_periods)) / self.sr_lookback_periods
            recency_strength = 0.5 + 0.5 * recency  # 0.5 to 1.0
            
            # Combine strengths
            strength = (
                self.volume_weight * volume_strength +
                self.price_weight * touch_strength +
                0.2 * recency_strength
            )
            
            return float(min(strength, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error calculating level strength: {e}")
            return 0.5
    
    def _calculate_volume_strength(self, market_data: pd.DataFrame, index: int) -> float:
        """Calculate strength based on volume profile."""
        try:
            volumes = market_data["volume"].values
            
            # Get volume at this level
            level_volume = volumes[index]
            
            # Compare to average volume
            avg_volume = np.mean(volumes[-self.sr_lookback_periods:])
            std_volume = np.std(volumes[-self.sr_lookback_periods:])
            
            # Calculate z-score
            if std_volume > 0:
                z_score = (level_volume - avg_volume) / std_volume
                # Normalize to 0-1 range
                strength = 1 / (1 + np.exp(-z_score))
            else:
                strength = 0.5
                
            return float(strength)
            
        except Exception as e:
            self.logger.error(f"Error calculating volume strength: {e}")
            return 0.5
    
    def _cluster_levels(self, levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster nearby S/R levels."""
        if not DBSCAN_AVAILABLE or len(levels) < 2:
            return levels
            
        try:
            # Extract prices for clustering
            prices = np.array([[level["price"]] for level in levels])
            
            # Normalize prices for clustering
            price_range = prices.max() - prices.min()
            if price_range > 0:
                normalized_prices = (prices - prices.min()) / price_range
            else:
                return levels
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(
                eps=self.clustering_eps,
                min_samples=self.clustering_min_samples
            ).fit(normalized_prices)
            
            # Group levels by cluster
            clustered_levels = []
            unique_labels = set(clustering.labels_)
            
            for label in unique_labels:
                if label == -1:  # Noise points (not clustered)
                    # Keep individual levels
                    indices = np.where(clustering.labels_ == label)[0]
                    for idx in indices:
                        clustered_levels.append(levels[idx])
                else:
                    # Merge clustered levels
                    indices = np.where(clustering.labels_ == label)[0]
                    cluster_levels = [levels[idx] for idx in indices]
                    
                    # Calculate merged level
                    avg_price = np.mean([level["price"] for level in cluster_levels])
                    max_strength = max([level["strength"] for level in cluster_levels])
                    total_volume = sum([level["volume"] for level in cluster_levels])
                    
                    clustered_levels.append({
                        "price": float(avg_price),
                        "type": cluster_levels[0]["type"],
                        "strength": float(max_strength),
                        "volume": float(total_volume),
                        "index": cluster_levels[0]["index"],
                        "method": cluster_levels[0]["method"],
                        "cluster_size": len(cluster_levels)
                    })
            
            return clustered_levels
            
        except Exception as e:
            self.logger.error(f"Error clustering levels: {e}")
            return levels