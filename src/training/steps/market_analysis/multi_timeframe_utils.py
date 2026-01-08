"""
Multi-Timeframe Data Processing Utilities for GMM Enhanced Features

This module provides utilities for handling multiple timeframes (15m, 60m, 4h)
with efficient resampling, alignment, and fusion capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
from src.utils.tprint import tprint_info, tprint_warning, tprint_success


class MultiTimeframeProcessor:
    """
    Efficient multi-timeframe data processor with memory-conscious design.
    
    Supports 15m (base), 60m, and 4h timeframes with intelligent alignment
    and feature fusion capabilities.
    """
    
    def __init__(self, base_timeframe: str = "15m"):
        self.base_timeframe = base_timeframe
        self.target_timeframes = ["15m", "60m", "4h"]
        self.timeframe_multipliers = {
            "15m": 1,
            "60m": 4,    # 4x 15m bars
            "4h": 16     # 16x 15m bars
        }
        
        # Resampling configurations
        self.resample_config = {
            "60m": {
                "agg_rules": {
                    "open": "first",
                    "high": "max", 
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                },
                "feature_windows": [3, 6, 12, 24]  # hours
            },
            "4h": {
                "agg_rules": {
                    "open": "first",
                    "high": "max",
                    "low": "min", 
                    "close": "last",
                    "volume": "sum"
                },
                "feature_windows": [1, 2, 4, 8]  # 4h bars
            }
        }
        
        # Memory management
        self.max_memory_mb = 1024  # 1GB limit
        self.chunk_size = 10000     # Default chunk size
        
    def estimate_memory_usage(self, data_shape: Tuple[int, int]) -> float:
        """Estimate memory usage in MB for given data shape."""
        # Assume float32 (4 bytes) per value
        bytes_per_value = 4
        total_bytes = data_shape[0] * data_shape[1] * bytes_per_value
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def calculate_optimal_chunk_size(self, n_rows: int, n_features: int) -> int:
        """Calculate optimal chunk size based on memory constraints."""
        available_memory = self.max_memory_mb * 0.7  # Use 70% of available memory
        
        # Estimate memory per row
        memory_per_row = (n_features * 4) / (1024 * 1024)  # MB per row
        
        if memory_per_row > 0:
            max_rows = int(available_memory / memory_per_row)
            return min(max_rows, n_rows, self.chunk_size)
        
        return min(self.chunk_size, n_rows)
    
    def resample_ohlcv(self, data: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """
        Resample OHLCV data to target timeframe with efficient aggregation.
        
        Args:
            data: OHLCV data with datetime index
            target_tf: Target timeframe ("60m" or "4h")
            
        Returns:
            Resampled DataFrame
        """
        if target_tf not in self.resample_config:
            raise ValueError(f"Unsupported timeframe: {target_tf}")
        
        config = self.resample_config[target_tf]
        
        # Convert to pandas if not already
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Resample using optimized rules
        resampled = data.resample(target_tf).agg(config['agg_rules'])
        
        # Remove any rows with NaN values (gaps in data)
        resampled = resampled.dropna()
        
        # Ensure column names are lowercase for consistency
        resampled.columns = [col.lower() for col in resampled.columns]
        
        # Add derived features
        resampled['returns'] = resampled['close'].pct_change()
        resampled['volatility'] = resampled['returns'].rolling(
            window=self.timeframe_multipliers[target_tf]
        ).std()
        
        return resampled
    
    def align_timeframes(self, 
                        base_data: pd.DataFrame, 
                        higher_tf_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align higher timeframe data to base timeframe (15m) using forward fill.
        
        Args:
            base_data: 15m data
            higher_tf_data: Dictionary of resampled data for 60m and 4h
            
        Returns:
            Dictionary with aligned data for each timeframe
        """
        aligned_data = {"15m": base_data}
        
        for tf, data in higher_tf_data.items():
            if tf == "15m":
                continue
                
            # Reindex to base timeframe and forward fill
            aligned = data.reindex(base_data.index, method='ffill')
            
            # Add prefix to columns to avoid conflicts
            aligned.columns = [f"{tf}_{col}" for col in aligned.columns]
            
            aligned_data[tf] = aligned
            
        tprint_info(f"✅ Aligned {tf} data to {len(base_data)} 15m bars")
        
        return aligned_data
    
    def calculate_timeframe_weights(self, 
                                  volatility_regime: str = "normal",
                                  custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate dynamic weights for timeframe fusion based on market conditions.
        
        Args:
            volatility_regime: Current volatility regime (low/normal/high)
            custom_weights: Optional custom weight overrides
            
        Returns:
            Dictionary of weights for each timeframe
        """
        if custom_weights:
            return custom_weights
        
        # Base weights depend on volatility regime
        base_weights = {
            "low": {"15m": 0.4, "60m": 0.35, "4h": 0.25},
            "normal": {"15m": 0.5, "60m": 0.3, "4h": 0.2},
            "high": {"15m": 0.6, "60m": 0.25, "4h": 0.15}
        }
        
        weights = base_weights.get(volatility_regime, base_weights["normal"])
        
        # Normalize weights
        total_weight = sum(weights.values())
        return {tf: w/total_weight for tf, w in weights.items()}
    
    def fuse_multi_timeframe_features(self,
                                     features_15m: pd.DataFrame,
                                     features_60m: pd.DataFrame, 
                                     features_4h: pd.DataFrame,
                                     weights: Optional[Dict[str, float]] = None,
                                     method: str = "weighted_average") -> pd.DataFrame:
        """
        Fuse features from multiple timeframes using specified method.
        
        Args:
            features_15m: 15m timeframe features
            features_60m: 60m timeframe features (aligned to 15m)
            features_4h: 4h timeframe features (aligned to 15m)
            weights: Optional weights for fusion
            method: Fusion method ("weighted_average", "ensemble", "adaptive")
            
        Returns:
            Fused features DataFrame
        """
        if weights is None:
            weights = self.calculate_timeframe_weights()
        
        if method == "weighted_average":
            return self._weighted_average_fusion(
                features_15m, features_60m, features_4h, weights
            )
        elif method == "ensemble":
            return self._ensemble_fusion(
                features_15m, features_60m, features_4h, weights
            )
        elif method == "adaptive":
            return self._adaptive_fusion(
                features_15m, features_60m, features_4h, weights
            )
        else:
            raise ValueError(f"Unknown fusion method: {method}")
    
    def _weighted_average_fusion(self,
                               features_15m: pd.DataFrame,
                               features_60m: pd.DataFrame,
                               features_4h: pd.DataFrame,
                               weights: Dict[str, float]) -> pd.DataFrame:
        """Simple weighted average fusion."""
        # Ensure all DataFrames have the same index
        common_index = features_15m.index
        features_60m = features_60m.reindex(common_index)
        features_4h = features_4h.reindex(common_index)
        
        # Start with base timeframe
        fused = features_15m.copy()
        
        # Add weighted contributions from higher timeframes
        for tf, features, weight in [
            ("60m", features_60m, weights["60m"]),
            ("4h", features_4h, weights["4h"])
        ]:
            # Align features by name (strip timeframe prefix)
            aligned_features = features.copy()
            aligned_features.columns = [
                col.replace(f"{tf}_", "") for col in aligned_features.columns
            ]
            
            # Only add features that exist in base timeframe
            common_cols = set(fused.columns) & set(aligned_features.columns)
            for col in common_cols:
                fused[col] = (
                    weights["15m"] * fused[col] + 
                    weight * aligned_features[col]
                ) / (weights["15m"] + weight)
        
        return fused
    
    def _ensemble_fusion(self,
                        features_15m: pd.DataFrame,
                        features_60m: pd.DataFrame,
                        features_4h: pd.DataFrame,
                        weights: Dict[str, float]) -> pd.DataFrame:
        """Ensemble fusion with feature selection."""
        # Calculate feature importance scores for each timeframe
        importance_scores = {
            "15m": self._calculate_feature_importance(features_15m),
            "60m": self._calculate_feature_importance(features_60m),
            "4h": self._calculate_feature_importance(features_4h)
        }
        
        # Select top features from each timeframe
        top_features = {}
        for tf, scores in importance_scores.items():
            # Select top 70% of features
            threshold = np.percentile(list(scores.values()), 30)
            top_features[tf] = {
                col: score for col, score in scores.items() 
                if score >= threshold
            }
        
        # Fuse selected features
        fused = pd.DataFrame(index=features_15m.index)
        
        for tf, features_dict in top_features.items():
            tf_features = {
                "15m": features_15m,
                "60m": features_60m, 
                "4h": features_4h
            }[tf]
            
            for col in features_dict.keys():
                # Handle timeframe prefixes
                clean_col = col.replace(f"{tf}_", "")
                fused_col = f"{clean_col}_{tf}"
                
                if clean_col in tf_features.columns:
                    fused[fused_col] = tf_features[clean_col] * weights[tf]
        
        return fused
    
    def _adaptive_fusion(self,
                         features_15m: pd.DataFrame,
                         features_60m: pd.DataFrame,
                         features_4h: pd.DataFrame,
                         weights: Dict[str, float]) -> pd.DataFrame:
        """Adaptive fusion based on recent performance."""
        # Calculate recent volatility for each timeframe
        volatilities = {
            "15m": self._calculate_recent_volatility(features_15m),
            "60m": self._calculate_recent_volatility(features_60m),
            "4h": self._calculate_recent_volatility(features_4h)
        }
        
        # Adjust weights based on volatility (inverse relationship)
        adaptive_weights = {}
        total_inverse_vol = sum(1/vol for vol in volatilities.values() if vol > 0)
        
        for tf, vol in volatilities.items():
            if vol > 0 and total_inverse_vol > 0:
                adaptive_weights[tf] = (1/vol) / total_inverse_vol
            else:
                adaptive_weights[tf] = weights[tf]
        
        # Blend with original weights
        blended_weights = {
            tf: 0.7 * weights[tf] + 0.3 * adaptive_weights[tf]
            for tf in weights.keys()
        }
        
        return self._weighted_average_fusion(
            features_15m, features_60m, features_4h, blended_weights
        )
    
    def _calculate_feature_importance(self, features: pd.DataFrame) -> Dict[str, float]:
        """Calculate simple feature importance based on variance."""
        importance = {}
        for col in features.columns:
            # Use variance as importance proxy
            variance = features[col].var()
            if variance > 0:
                importance[col] = variance
            else:
                importance[col] = 0.0
        return importance
    
    def _calculate_recent_volatility(self, features: pd.DataFrame, window: int = 20) -> float:
        """Calculate recent volatility of features."""
        if len(features) < window:
            return 0.0
        
        # Use mean of rolling standard deviations across all features
        rolling_stds = features.rolling(window=window).std()
        mean_std = rolling_stds.mean().mean()
        
        return mean_std if not np.isnan(mean_std) else 0.0
    
    def process_multi_timeframe_streaming(self,
                                        data_15m: pd.DataFrame,
                                        feature_generator,
                                        chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Process multi-timeframe data in streaming fashion to handle large datasets.
        
        Args:
            data_15m: 15m OHLCV data
            feature_generator: Function to generate features for a chunk
            chunk_size: Optional custom chunk size
            
        Returns:
            Combined features from all timeframes
        """
        if chunk_size is None:
            chunk_size = self.calculate_optimal_chunk_size(
                len(data_15m), 
                len(data_15m.columns) * 10  # Estimate feature count
            )
        
        tprint_info(f"🚀 Processing {len(data_15m)} rows in chunks of {chunk_size}")
        
        # Resample to higher timeframes once
        data_60m = self.resample_ohlcv(data_15m, "60m")
        data_4h = self.resample_ohlcv(data_15m, "4h")
        
        # Align all timeframes
        aligned_data = self.align_timeframes(data_15m, {"60m": data_60m, "4h": data_4h})
        
        # Process in chunks
        all_features = []
        
        for i in range(0, len(data_15m), chunk_size):
            end_idx = min(i + chunk_size, len(data_15m))
            chunk_15m = aligned_data["15m"].iloc[i:end_idx]
            chunk_60m = aligned_data["60m"].iloc[i:end_idx]
            chunk_4h = aligned_data["4h"].iloc[i:end_idx]
            
            # Generate features for this chunk
            chunk_features = feature_generator(chunk_15m, chunk_60m, chunk_4h)
            
            all_features.append(chunk_features)
            
            # Memory cleanup
            del chunk_15m, chunk_60m, chunk_4h, chunk_features
            
            if i % (chunk_size * 5) == 0:  # Progress update every 5 chunks
                tprint_info(f"📊 Processed {min(end_idx, len(data_15m))}/{len(data_15m)} rows")
        
        # Combine all chunks
        final_features = pd.concat(all_features, ignore_index=False)
        
        tprint_success(f"✅ Streaming processing complete: {len(final_features)} features")
        
        return final_features


def create_multi_timeframe_features(chunk_15m: pd.DataFrame,
                                   chunk_60m: pd.DataFrame, 
                                   chunk_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Example feature generator for multi-timeframe processing.
    
    This function demonstrates how to create features from multiple timeframes
    and will be used as a template for the GMM enhanced features.
    """
    features = pd.DataFrame(index=chunk_15m.index)
    
    # 15m features (base timeframe)
    features['returns_15m'] = chunk_15m['close'].pct_change()
    features['volatility_15m'] = features['returns_15m'].rolling(20).std()
    features['rsi_15m'] = calculate_rsi(chunk_15m['close'])
    
    # 60m features (aligned to 15m)
    if '60m_returns' in chunk_60m.columns:
        features['returns_60m'] = chunk_60m['60m_returns']
        features['volatility_60m'] = chunk_60m['60m_volatility']
        features['trend_60m'] = chunk_60m['60m_close'].rolling(6).mean()
    
    # 4h features (aligned to 15m)
    if '4h_returns' in chunk_4h.columns:
        features['returns_4h'] = chunk_4h['4h_returns']
        features['volatility_4h'] = chunk_4h['4h_volatility']
        features['trend_4h'] = chunk_4h['4h_close'].rolling(4).mean()
    
    # Cross-timeframe features
    if 'returns_60m' in features.columns and 'returns_15m' in features.columns:
        features['momentum_alignment'] = (
            np.sign(features['returns_60m']) == np.sign(features['returns_15m'])
        ).astype(int)
    
    return features


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Fill NaN with neutral RSI
