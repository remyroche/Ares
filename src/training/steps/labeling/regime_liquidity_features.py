"""
Regime and Liquidity Features for Modern De Prado Framework

This module generates market regime indicators and liquidity features
from OHLCV data to enhance the causal framework's feature set.

Key Features:
- Market regime indicators (volatility, trend, volume, price level)
- Liquidity dynamics (spread proxy, VWAP, price impact, depth)
- Cross-asset correlation features
- Temporal pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")


class RegimeLiquidityFeatureGenerator:
    """
    Generate regime and liquidity features from OHLCV data.
    
    This class creates market state indicators and liquidity dynamics
    features that capture non-causal alpha patterns for the Chaser system
    and enhance the overall causal framework.
    """
    
    def __init__(
        self,
        volatility_windows: List[int] = [10, 20, 50],
        trend_windows: List[int] = [20, 50, 100],
        volume_windows: List[int] = [10, 20, 50],
        liquidity_windows: List[int] = [5, 10, 20],
        verbose: bool = True
    ):
        """
        Initialize Regime and Liquidity Feature Generator.
        
        Args:
            volatility_windows: Windows for volatility regime calculation
            trend_windows: Windows for trend regime calculation
            volume_windows: Windows for volume regime calculation
            liquidity_windows: Windows for liquidity feature calculation
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.volatility_windows = volatility_windows
        self.trend_windows = trend_windows
        self.volume_windows = volume_windows
        self.liquidity_windows = liquidity_windows
        
        # Feature cache for efficiency
        self._feature_cache = {}
        
        if self.verbose:
            tprint_info("🔧 Regime & Liquidity Feature Generator: Initializing...")
            tprint_info(f"   ⚙️ Volatility windows: {volatility_windows}")
            tprint_info(f"   ⚙️ Trend windows: {trend_windows}")
            tprint_info(f"   ⚙️ Volume windows: {volume_windows}")
            tprint_info(f"   ⚙️ Liquidity windows: {liquidity_windows}")
            tprint_success("   ✅ Regime & Liquidity Feature Generator: Initialization complete")
    
    def generate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate market regime indicators from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data (open, high, low, close, volume)
            
        Returns:
            DataFrame with regime features
        """
        try:
            if self.verbose:
                tprint_info("🎯 Generating Regime Features...")
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                tprint_error(f"   ❌ Missing required columns: {missing_cols}")
                return pd.DataFrame(index=df.index)
            
            features = pd.DataFrame(index=df.index)
            feature_count = 0
            
            # 1. Volatility Regime Features
            if self.verbose:
                tprint_info("   📊 Computing volatility regime features...")
            
            for window in self.volatility_windows:
                try:
                    # Calculate returns volatility
                    returns = df['close'].pct_change()
                    vol_short = returns.rolling(window).std()
                    vol_long = returns.rolling(window * 2).std() if window * 2 <= len(df) else vol_short
                    
                    # Volatility regime ratio
                    vol_regime = vol_short / (vol_long + 1e-9)
                    features[f'vol_regime_{window}'] = vol_regime
                    feature_count += 1
                    
                    # Volatility level (normalized)
                    vol_normalized = vol_short / vol_short.rolling(100).mean()
                    features[f'vol_level_{window}'] = vol_normalized
                    feature_count += 1
                    
                    # Volatility trend (is volatility increasing/decreasing?)
                    vol_trend = vol_short.pct_change(window)
                    features[f'vol_trend_{window}'] = vol_trend
                    feature_count += 1
                    
                except Exception as e:
                    tprint_warning(f"      ⚠️ Volatility regime {window} failed: {e}")
            
            # 2. Trend Regime Features
            if self.verbose:
                tprint_info("   📈 Computing trend regime features...")
            
            for window in self.trend_windows:
                try:
                    # Price trend using linear regression
                    price_trend = df['close'].rolling(window).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
                    )
                    
                    # Trend direction
                    trend_direction = np.sign(price_trend)
                    features[f'trend_direction_{window}'] = trend_direction
                    feature_count += 1
                    
                    # Trend strength (normalized)
                    trend_strength = np.abs(price_trend) / (df['close'].rolling(window).std() + 1e-9)
                    features[f'trend_strength_{window}'] = trend_strength
                    feature_count += 1
                    
                    # Price momentum
                    price_momentum = (df['close'] / df['close'].shift(window) - 1)
                    features[f'price_momentum_{window}'] = price_momentum
                    feature_count += 1
                    
                except Exception as e:
                    tprint_warning(f"      ⚠️ Trend regime {window} failed: {e}")
            
            # 3. Volume Regime Features
            if self.verbose:
                tprint_info("   📊 Computing volume regime features...")
            
            for window in self.volume_windows:
                try:
                    # Volume moving average
                    vol_ma = df['volume'].rolling(window).mean()
                    
                    # Volume regime ratio
                    volume_regime = df['volume'] / (vol_ma + 1e-9)
                    features[f'volume_regime_{window}'] = volume_regime
                    feature_count += 1
                    
                    # Volume trend
                    volume_trend = vol_ma.pct_change(window)
                    features[f'volume_trend_{window}'] = volume_trend
                    feature_count += 1
                    
                    # Volume-price trend (divergence)
                    price_trend = df['close'].pct_change(window)
                    volume_price_divergence = volume_trend - price_trend
                    features[f'volume_price_divergence_{window}'] = volume_price_divergence
                    feature_count += 1
                    
                except Exception as e:
                    tprint_warning(f"      ⚠️ Volume regime {window} failed: {e}")
            
            # 4. Price Level Regime Features
            if self.verbose:
                tprint_info("   📍 Computing price level regime features...")
            
            for window in [20, 50, 100]:  # Fixed windows for price levels
                try:
                    if window <= len(df):
                        # Price position in range
                        price_high = df['high'].rolling(window).max()
                        price_low = df['low'].rolling(window).min()
                        price_position = (df['close'] - price_low) / (price_high - price_low + 1e-9)
                        
                        features[f'price_position_{window}'] = price_position
                        feature_count += 1
                        
                        # Price range expansion/contraction
                        price_range = price_high - price_low
                        range_trend = price_range.pct_change(window)
                        features[f'range_trend_{window}'] = range_trend
                        feature_count += 1
                        
                except Exception as e:
                    tprint_warning(f"      ⚠️ Price level regime {window} failed: {e}")
            
            if self.verbose:
                tprint_success(f"   ✅ Regime features generated: {feature_count} features")
            
            return features
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Regime feature generation failed: {e}")
            return pd.DataFrame(index=df.index)
    
    def generate_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate liquidity dynamics features from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with liquidity features
        """
        try:
            if self.verbose:
                tprint_info("💧 Generating Liquidity Features...")
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                tprint_error(f"   ❌ Missing required columns: {missing_cols}")
                return pd.DataFrame(index=df.index)
            
            features = pd.DataFrame(index=df.index)
            feature_count = 0
            
            # 1. Spread Proxy Features
            if self.verbose:
                tprint_info("   📏 Computing spread proxy features...")
            
            for window in self.liquidity_windows:
                try:
                    # Bid-ask spread proxy (high-low range)
                    spread_proxy = (df['high'] - df['low']) / df['close']
                    features[f'spread_proxy_{window}'] = spread_proxy.rolling(window).mean()
                    feature_count += 1
                    
                    # Spread volatility
                    spread_volatility = spread_proxy.rolling(window).std()
                    features[f'spread_volatility_{window}'] = spread_volatility
                    feature_count += 1
                    
                    # Spread trend
                    spread_trend = spread_proxy.rolling(window).mean().pct_change(window)
                    features[f'spread_trend_{window}'] = spread_trend
                    feature_count += 1
                    
                except Exception as e:
                    tprint_warning(f"      ⚠️ Spread proxy {window} failed: {e}")
            
            # 2. Volume-Weighted Average Price (VWAP) Features
            if self.verbose:
                tprint_info("   ⚖️ Computing VWAP features...")
            
            for window in self.liquidity_windows:
                try:
                    # Typical price (HLC/3)
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    
                    # VWAP
                    vwap = (typical_price * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
                    features[f'vwap_{window}'] = vwap
                    feature_count += 1
                    
                    # Price deviation from VWAP
                    vwap_deviation = (df['close'] - vwap) / vwap
                    features[f'vwap_deviation_{window}'] = vwap_deviation
                    feature_count += 1
                    
                    # VWAP trend
                    vwap_trend = vwap.pct_change(window)
                    features[f'vwap_trend_{window}'] = vwap_trend
                    feature_count += 1
                    
                except Exception as e:
                    tprint_warning(f"      ⚠️ VWAP {window} failed: {e}")
            
            # 3. Price Impact Features
            if self.verbose:
                tprint_info("   💥 Computing price impact features...")
            
            for window in self.liquidity_windows:
                try:
                    # Price change
                    price_change = df['close'].pct_change()
                    
                    # Volume change
                    volume_change = df['volume'].pct_change()
                    
                    # Price impact (abs(price_change) / volume_change)
                    price_impact = abs(price_change) / (abs(volume_change) + 1e-9)
                    features[f'price_impact_{window}'] = price_impact.rolling(window).mean()
                    feature_count += 1
                    
                    # Impact volatility
                    impact_volatility = price_impact.rolling(window).std()
                    features[f'impact_volatility_{window}'] = impact_volatility
                    feature_count += 1
                    
                    # Impact trend
                    impact_trend = price_impact.rolling(window).mean().pct_change(window)
                    features[f'impact_trend_{window}'] = impact_trend
                    feature_count += 1
                    
                except Exception as e:
                    tprint_warning(f"      ⚠️ Price impact {window} failed: {e}")
            
            # 4. Liquidity Ratio Features
            if self.verbose:
                tprint_info("   📊 Computing liquidity ratio features...")
            
            for window in self.liquidity_windows:
                try:
                    # Liquidity ratio (volume / price range)
                    price_range = df['high'] - df['low']
                    liquidity_ratio = df['volume'] / (price_range + 1e-9)
                    features[f'liquidity_ratio_{window}'] = liquidity_ratio.rolling(window).mean()
                    feature_count += 1
                    
                    # Liquidity trend
                    liquidity_trend = liquidity_ratio.rolling(window).mean().pct_change(window)
                    features[f'liquidity_trend_{window}'] = liquidity_trend
                    feature_count += 1
                    
                    # Liquidity volatility
                    liquidity_volatility = liquidity_ratio.rolling(window).std()
                    features[f'liquidity_volatility_{window}'] = liquidity_volatility
                    feature_count += 1
                    
                except Exception as e:
                    tprint_warning(f"      ⚠️ Liquidity ratio {window} failed: {e}")
            
            # 5. Depth Indicator Features
            if self.verbose:
                tprint_info("   🏊 Computing depth indicator features...")
            
            for window in self.liquidity_windows:
                try:
                    # Depth indicator (volume * price / price range)
                    price_range = df['high'] - df['low']
                    depth_indicator = (df['volume'] * df['close']) / (price_range + 1e-9)
                    features[f'depth_indicator_{window}'] = depth_indicator.rolling(window).mean()
                    feature_count += 1
                    
                    # Depth trend
                    depth_trend = depth_indicator.rolling(window).mean().pct_change(window)
                    features[f'depth_trend_{window}'] = depth_trend
                    feature_count += 1
                    
                    # Depth volatility
                    depth_volatility = depth_indicator.rolling(window).std()
                    features[f'depth_volatility_{window}'] = depth_volatility
                    feature_count += 1
                    
                except Exception as e:
                    tprint_warning(f"      ⚠️ Depth indicator {window} failed: {e}")
            
            if self.verbose:
                tprint_success(f"   ✅ Liquidity features generated: {feature_count} features")
            
            return features
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Liquidity feature generation failed: {e}")
            return pd.DataFrame(index=df.index)
    
    def generate_cross_asset_features(self, df: pd.DataFrame, reference_assets: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Generate cross-asset correlation features.
        
        Args:
            df: Primary asset OHLCV data
            reference_assets: Dictionary of reference asset DataFrames
            
        Returns:
            DataFrame with cross-asset features
        """
        try:
            if self.verbose:
                tprint_info("🔗 Generating Cross-Asset Features...")
            
            features = pd.DataFrame(index=df.index)
            feature_count = 0
            
            if reference_assets is None or len(reference_assets) == 0:
                if self.verbose:
                    tprint_warning("   ⚠️ No reference assets provided, generating internal correlations")
                
                # Generate internal correlation features
                for window in [10, 20, 50]:
                    try:
                        # Price-volume correlation
                        price_volume_corr = df['close'].pct_change().rolling(window).corr(
                            df['volume'].pct_change()
                        )
                        features[f'price_volume_corr_{window}'] = price_volume_corr
                        feature_count += 1
                        
                        # High-low correlation (range consistency)
                        high_low_corr = df['high'].pct_change().rolling(window).corr(
                            df['low'].pct_change()
                        )
                        features[f'high_low_corr_{window}'] = high_low_corr
                        feature_count += 1
                        
                    except Exception as e:
                        tprint_warning(f"      ⚠️ Internal correlation {window} failed: {e}")
            else:
                # Generate cross-asset correlation features
                for asset_name, asset_data in reference_assets.items():
                    try:
                        if 'close' in asset_data.columns and len(asset_data) == len(df):
                            # Price correlation
                            price_corr = df['close'].pct_change().rolling(20).corr(
                                asset_data['close'].pct_change()
                            )
                            features[f'price_corr_{asset_name}'] = price_corr
                            feature_count += 1
                            
                            # Beta estimation
                            asset_returns = asset_data['close'].pct_change()
                            market_returns = df['close'].pct_change()
                            
                            beta = asset_returns.rolling(20).cov(market_returns) / (
                                market_returns.rolling(20).var() + 1e-9
                            )
                            features[f'beta_{asset_name}'] = beta
                            feature_count += 1
                            
                            # Relative strength
                            relative_strength = (df['close'] / asset_data['close']).pct_change()
                            features[f'relative_strength_{asset_name}'] = relative_strength
                            feature_count += 1
                            
                    except Exception as e:
                        tprint_warning(f"      ⚠️ Cross-asset {asset_name} failed: {e}")
            
            if self.verbose:
                tprint_success(f"   ✅ Cross-asset features generated: {feature_count} features")
            
            return features
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Cross-asset feature generation failed: {e}")
            return pd.DataFrame(index=df.index)
    
    def generate_all_features(
        self, 
        df: pd.DataFrame, 
        reference_assets: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Generate all regime and liquidity features.
        
        Args:
            df: Primary asset OHLCV data
            reference_assets: Dictionary of reference asset DataFrames
            
        Returns:
            DataFrame with all regime and liquidity features
        """
        try:
            if self.verbose:
                tprint_info("🚀 Regime & Liquidity Feature Generation: Starting complete pipeline...")
            
            start_time = pd.Timestamp.now()
            
            # Generate regime features
            regime_features = self.generate_regime_features(df)
            
            # Generate liquidity features
            liquidity_features = self.generate_liquidity_features(df)
            
            # Generate cross-asset features
            cross_asset_features = self.generate_cross_asset_features(df, reference_assets)
            
            # Combine all features
            all_features = pd.concat([regime_features, liquidity_features, cross_asset_features], axis=1)
            
            # Clean up infinite values and fill NaN
            all_features = all_features.replace([np.inf, -np.inf], np.nan)
            all_features = all_features.fillna(method='ffill').fillna(0)
            
            end_time = pd.Timestamp.now()
            generation_time = (end_time - start_time).total_seconds()
            
            if self.verbose:
                tprint_success("✅ Regime & Liquidity Feature Generation Complete:")
                tprint_info(f"   - Total features: {len(all_features.columns)}")
                tprint_info(f"   - Regime features: {len(regime_features.columns)}")
                tprint_info(f"   - Liquidity features: {len(liquidity_features.columns)}")
                tprint_info(f"   - Cross-asset features: {len(cross_asset_features.columns)}")
                tprint_info(f"   - Generation time: {generation_time:.3f}s")
                tprint_info(f"   - Data points: {len(all_features)}")
            
            return all_features
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Complete feature generation failed: {e}")
            return pd.DataFrame(index=df.index)


# Convenience functions for quick usage
def quick_regime_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Quick generation of regime features."""
    generator = RegimeLiquidityFeatureGenerator(verbose=verbose)
    return generator.generate_regime_features(df)


def quick_liquidity_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Quick generation of liquidity features."""
    generator = RegimeLiquidityFeatureGenerator(verbose=verbose)
    return generator.generate_liquidity_features(df)


def quick_all_regime_liquidity_features(
    df: pd.DataFrame, 
    reference_assets: Optional[Dict[str, pd.DataFrame]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Quick generation of all regime and liquidity features."""
    generator = RegimeLiquidityFeatureGenerator(verbose=verbose)
    return generator.generate_all_features(df, reference_assets)


if __name__ == "__main__":
    # Example usage
    print("Regime & Liquidity Features Module")
    print("Use quick_regime_features(), quick_liquidity_features(), or quick_all_regime_liquidity_features()")
