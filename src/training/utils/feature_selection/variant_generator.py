"""
Variant Generator for Feature Engineering

Generates normalized variants of features using optimal lookbacks with:
- Volatility normalization
- VWAP weighting
- Trend adjustment
- RobustScaler bounding to prevent extreme values
- Causality enforcement via shift(1)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler
import logging

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning
from src.utils.logger import system_logger


@dataclass
class VariantConfig:
    """Configuration for variant generation."""
    feature_name: str
    category: str
    optimal_lookback: int
    enable_vol_norm: bool = True
    enable_vwap: bool = True
    enable_trend_adj: bool = True
    robust_scaler_quantile_range: Tuple[float, float] = (1.0, 99.0)


class VariantGenerator:
    """
    Generates normalized variants of features with causality enforcement.
    
    Produces 3-4 variants per feature depending on category:
    1. Base variant (original)
    2. Volatility-normalized (except volatility features)
    3. VWAP-weighted (only price-based features, not volume)
    4. Trend-adjusted (only oscillators/momentum)
    """
    
    def __init__(self):
        """Initialize variant generator."""
        self.logger = system_logger.getChild('VariantGenerator')
        self.scaler = RobustScaler(quantile_range=(1.0, 99.0))
        
        # Track statistics
        self.stats = {
            'total_variants_generated': 0,
            'failed_variants': [],
            'clipping_stats': {},
            'variants_by_type': {
                'base': 0,
                'volnorm': 0,
                'vwap': 0,
                'trend_adj': 0
            }
        }
    
    def generate_variants(
        self,
        data: pd.DataFrame,
        feature_name: str,
        category: str,
        optimal_lookback: int,
        ohlcv_data: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Generate all applicable variants for a feature.
        
        Args:
            data: DataFrame containing the feature
            feature_name: Name of the feature to generate variants for
            category: Feature category (trend, oscillator, momentum, return, volatility, volume, acceleration)
            optimal_lookback: Optimal lookback period from optimization
            ohlcv_data: DataFrame with OHLCV columns (close, high, low, open, volume)
            
        Returns:
            Dictionary mapping variant names to Series
        """
        variants = {}
        
        try:
            # 1. Base variant (original feature)
            base_variant = data[feature_name].copy()
            variants[f"{feature_name}_base"] = self._apply_causality_shift(base_variant)
            self.stats['variants_by_type']['base'] += 1
            
            # 2. Volatility-normalized (skip if volatility feature)
            if category.lower() != 'volatility':
                try:
                    vol_norm = self._generate_volatility_normalized(
                        data[feature_name],
                        ohlcv_data['close'],
                        optimal_lookback
                    )
                    if vol_norm is not None:
                        variants[f"{feature_name}_volnorm"] = self._apply_causality_shift(vol_norm)
                        self.stats['variants_by_type']['volnorm'] += 1
                except Exception as e:
                    self.logger.warning(f"Failed to generate vol-norm variant for {feature_name}: {e}")
                    self.stats['failed_variants'].append(f"{feature_name}_volnorm")
            
            # 3. VWAP-weighted (only for price-based features, skip volume)
            if category.lower() not in ['volume'] and self._is_price_based_feature(feature_name, category):
                try:
                    vwap_weighted = self._generate_vwap_weighted(
                        data[feature_name],
                        ohlcv_data['volume'],
                        optimal_lookback
                    )
                    if vwap_weighted is not None:
                        variants[f"{feature_name}_vwap"] = self._apply_causality_shift(vwap_weighted)
                        self.stats['variants_by_type']['vwap'] += 1
                except Exception as e:
                    self.logger.warning(f"Failed to generate VWAP variant for {feature_name}: {e}")
                    self.stats['failed_variants'].append(f"{feature_name}_vwap")
            
            # 4. Trend-adjusted (only for oscillators/momentum)
            if category.lower() in ['oscillator', 'momentum']:
                try:
                    trend_adj = self._generate_trend_adjusted(
                        data[feature_name],
                        ohlcv_data,
                        optimal_lookback
                    )
                    if trend_adj is not None:
                        variants[f"{feature_name}_trend_adj"] = self._apply_causality_shift(trend_adj)
                        self.stats['variants_by_type']['trend_adj'] += 1
                except Exception as e:
                    self.logger.warning(f"Failed to generate trend-adj variant for {feature_name}: {e}")
                    self.stats['failed_variants'].append(f"{feature_name}_trend_adj")
            
            self.stats['total_variants_generated'] += len(variants)
            
        except Exception as e:
            self.logger.error(f"Failed to generate variants for {feature_name}: {e}")
            # Return at least base variant
            if f"{feature_name}_base" in variants:
                return {f"{feature_name}_base": variants[f"{feature_name}_base"]}
            return {}
        
        return variants
    
    def _generate_volatility_normalized(
        self,
        feature: pd.Series,
        close_prices: pd.Series,
        lookback: int
    ) -> Optional[pd.Series]:
        """
        Generate volatility-normalized variant.
        
        Formula: feature / rolling_std(returns, window=lookback)
        """
        try:
            # Calculate returns
            returns = close_prices.pct_change()
            
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=lookback, min_periods=max(1, lookback // 2)).std()
            
            # Avoid division by zero
            rolling_vol = rolling_vol.replace(0, np.nan)
            rolling_vol = rolling_vol.fillna(rolling_vol.mean())
            
            # Normalize
            vol_normalized = feature / rolling_vol
            
            # Apply robust scaling to bound extreme values
            vol_normalized = self._apply_robust_scaling(vol_normalized, f"volnorm")
            
            return vol_normalized
            
        except Exception as e:
            self.logger.error(f"Volatility normalization failed: {e}")
            return None
    
    def _generate_vwap_weighted(
        self,
        feature: pd.Series,
        volume: pd.Series,
        lookback: int
    ) -> Optional[pd.Series]:
        """
        Generate VWAP-weighted variant.
        
        Formula: feature * (volume / rolling_mean(volume, window=lookback))
        """
        try:
            # Calculate rolling mean volume
            rolling_vol_mean = volume.rolling(window=lookback, min_periods=max(1, lookback // 2)).mean()
            
            # Avoid division by zero
            rolling_vol_mean = rolling_vol_mean.replace(0, np.nan)
            rolling_vol_mean = rolling_vol_mean.fillna(volume.mean())
            
            # Calculate volume ratio
            volume_ratio = volume / rolling_vol_mean
            
            # Weight feature by volume ratio
            vwap_weighted = feature * volume_ratio
            
            # Apply robust scaling to bound extreme values
            vwap_weighted = self._apply_robust_scaling(vwap_weighted, f"vwap")
            
            return vwap_weighted
            
        except Exception as e:
            self.logger.error(f"VWAP weighting failed: {e}")
            return None
    
    def _generate_trend_adjusted(
        self,
        feature: pd.Series,
        ohlcv_data: pd.DataFrame,
        lookback: int
    ) -> Optional[pd.Series]:
        """
        Generate trend-adjusted variant.
        
        Formula: feature * EMA(ADX, span=14) * sign(SMA(close, lookback) - close.shift(1))
        """
        try:
            close = ohlcv_data['close']
            high = ohlcv_data['high']
            low = ohlcv_data['low']
            
            # Calculate ADX (Average Directional Index)
            adx = self._calculate_adx(high, low, close, period=14)
            
            # Smooth ADX with EMA
            adx_ema = adx.ewm(span=14, adjust=False).mean()
            
            # Calculate trend direction
            sma = close.rolling(window=lookback, min_periods=max(1, lookback // 2)).mean()
            trend_direction = np.sign(sma - close.shift(1))
            
            # Combine: feature * smoothed_ADX * trend_direction
            trend_adjusted = feature * (adx_ema / 100.0) * trend_direction  # Normalize ADX to 0-1 range
            
            # Apply robust scaling
            trend_adjusted = self._apply_robust_scaling(trend_adjusted, f"trend_adj")
            
            return trend_adjusted
            
        except Exception as e:
            self.logger.error(f"Trend adjustment failed: {e}")
            return None
    
    def _calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        Simplified implementation for trend strength measurement.
        """
        try:
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            plus_dm = pd.Series(plus_dm, index=high.index)
            minus_dm = pd.Series(minus_dm, index=high.index)
            
            # Smooth with EMA
            atr = tr.ewm(span=period, adjust=False).mean()
            plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
            
            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = dx.ewm(span=period, adjust=False).mean()
            
            return adx.fillna(25)  # Default ADX value
            
        except Exception as e:
            self.logger.error(f"ADX calculation failed: {e}")
            # Return default ADX values
            return pd.Series(25, index=high.index)
    
    def _apply_robust_scaling(self, series: pd.Series, variant_type: str) -> pd.Series:
        """
        Apply RobustScaler to bound extreme values using percentile clipping.
        
        Uses 1st and 99th percentiles to avoid discontinuities from z-score clipping.
        """
        try:
            # Remove NaN values for scaling
            valid_mask = ~series.isna()
            valid_data = series[valid_mask].values.reshape(-1, 1)
            
            if len(valid_data) == 0:
                return series
            
            # Fit and transform
            scaled_data = self.scaler.fit_transform(valid_data)
            
            # Track clipping statistics
            original_range = (valid_data.min(), valid_data.max())
            scaled_range = (scaled_data.min(), scaled_data.max())
            
            if variant_type not in self.stats['clipping_stats']:
                self.stats['clipping_stats'][variant_type] = {
                    'count': 0,
                    'original_ranges': [],
                    'scaled_ranges': []
                }
            
            self.stats['clipping_stats'][variant_type]['count'] += 1
            self.stats['clipping_stats'][variant_type]['original_ranges'].append(original_range)
            self.stats['clipping_stats'][variant_type]['scaled_ranges'].append(scaled_range)
            
            # Create result series
            result = series.copy()
            result[valid_mask] = scaled_data.flatten()
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Robust scaling failed for {variant_type}: {e}")
            return series
    
    def _apply_causality_shift(self, series: pd.Series) -> pd.Series:
        """
        Apply shift(1) to enforce causality and prevent lookahead bias.
        """
        return series.shift(1)
    
    def _is_price_based_feature(self, feature_name: str, category: str) -> bool:
        """
        Determine if a feature is price-based (suitable for VWAP weighting).
        
        Price-based features include: returns, momentum, oscillators, trends
        Not suitable: volume, volatility (already normalized)
        """
        price_based_categories = ['return', 'returns', 'momentum', 'oscillator', 'trend']
        
        if category.lower() in price_based_categories:
            return True
        
        # Check feature name for price-related keywords
        price_keywords = ['price', 'return', 'rsi', 'macd', 'sma', 'ema', 'momentum', 'roc']
        return any(keyword in feature_name.lower() for keyword in price_keywords)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_variants_generated': 0,
            'failed_variants': [],
            'clipping_stats': {},
            'variants_by_type': {
                'base': 0,
                'volnorm': 0,
                'vwap': 0,
                'trend_adj': 0
            }
        }


def generate_all_variants(
    features_df: pd.DataFrame,
    selected_features: List[Dict[str, Any]],
    ohlcv_data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate variants for all selected features.
    
    Args:
        features_df: DataFrame containing all features
        selected_features: List of dicts with keys: feature_name, category, optimal_lookback
        ohlcv_data: DataFrame with OHLCV columns
        
    Returns:
        Tuple of (variants_df, statistics)
    """
    generator = VariantGenerator()
    all_variants = {}
    
    tprint_info(f"🔄 Generating variants for {len(selected_features)} features...")
    
    for i, feature_info in enumerate(selected_features):
        feature_name = feature_info['feature_name']
        category = feature_info['category']
        optimal_lookback = feature_info['optimal_lookback']
        
        if feature_name not in features_df.columns:
            tprint_warning(f"⚠️ Feature {feature_name} not found in DataFrame, skipping...")
            continue
        
        try:
            variants = generator.generate_variants(
                features_df,
                feature_name,
                category,
                optimal_lookback,
                ohlcv_data
            )
            
            all_variants.update(variants)
            
            if (i + 1) % 10 == 0:
                tprint_info(f"  Progress: {i+1}/{len(selected_features)} features processed")
                
        except Exception as e:
            tprint_error(f"❌ Failed to generate variants for {feature_name}: {e}")
            generator.stats['failed_variants'].append(feature_name)
    
    # Create DataFrame from variants
    variants_df = pd.DataFrame(all_variants, index=features_df.index)
    
    stats = generator.get_stats()
    tprint_success(f"✅ Generated {len(variants_df.columns)} total variants from {len(selected_features)} features")
    tprint_info(f"  Breakdown: {stats['variants_by_type']}")
    
    if stats['failed_variants']:
        tprint_warning(f"⚠️ Failed variants: {len(stats['failed_variants'])}")
    
    return variants_df, stats

