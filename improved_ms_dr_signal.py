#!/usr/bin/env python3
"""
Improved MS-DR Composite Signal Construction

Addresses the three main problems:
1. Degenerate clustering (all samples -> Regime 0)
2. Burn-in detection not triggering correctly
3. Composite signal too uniform

Solutions:
- Enhance signal separation using non-linear transformations
- Add diversity metrics to validate signal quality
- Use adaptive component selection based on correlation analysis
- Implement multi-scale regime indicators
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy import stats
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, 'src')

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error


class ImprovedRegimeSignalBuilder:
    """
    Enhanced regime signal builder with better separation properties.
    
    Key improvements:
    1. Non-linear transformations for better regime separation
    2. Multi-scale indicators (different lookback periods)
    3. Adaptive weighting based on component correlation
    4. Signal diversity validation
    """
    
    def __init__(self, 
                 use_nonlinear_transforms: bool = True,
                 use_multiscale: bool = True,
                 use_adaptive_weighting: bool = True,
                 correlation_threshold: float = 0.7):
        """
        Initialize signal builder.
        
        Args:
            use_nonlinear_transforms: Apply non-linear transformations for better separation
            use_multiscale: Use multiple timeframe indicators
            use_adaptive_weighting: Adjust weights based on component correlation
            correlation_threshold: Max correlation between components
        """
        self.use_nonlinear_transforms = use_nonlinear_transforms
        self.use_multiscale = use_multiscale
        self.use_adaptive_weighting = use_adaptive_weighting
        self.correlation_threshold = correlation_threshold
        
        tprint_info("🔧 Initialized Improved Regime Signal Builder")
        tprint_info(f"   Non-linear transforms: {use_nonlinear_transforms}")
        tprint_info(f"   Multi-scale: {use_multiscale}")
        tprint_info(f"   Adaptive weighting: {use_adaptive_weighting}")
    
    def build_regime_signal(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        Build improved composite regime signal.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (regime_signal, diagnostics_dict)
        """
        tprint_info("🔨 Building improved regime signal...")
        
        # 1. Create multi-scale regime indicators
        regime_indicators = self._create_multiscale_indicators(df)
        
        # 2. Validate component diversity
        diversity_metrics = self._validate_component_diversity(regime_indicators)
        
        # 3. Apply non-linear transformations if enabled
        if self.use_nonlinear_transforms:
            regime_indicators = self._apply_nonlinear_transforms(regime_indicators)
        
        # 4. Determine weights (adaptive or fixed)
        if self.use_adaptive_weighting:
            weights = self._compute_adaptive_weights(regime_indicators)
        else:
            # Fixed weights with more balanced distribution
            weights = self._get_fixed_weights(regime_indicators)
        
        # 5. Construct composite signal
        regime_signal = self._construct_composite(regime_indicators, weights)
        
        # 6. Validate signal quality
        signal_quality = self._validate_signal_quality(regime_signal, regime_indicators)
        
        # 7. Prepare diagnostics
        diagnostics = {
            'diversity_metrics': diversity_metrics,
            'weights': weights,
            'signal_quality': signal_quality,
            'component_names': list(regime_indicators.columns),
            'n_components': len(regime_indicators.columns)
        }
        
        tprint_success(f"✅ Built regime signal with {len(regime_indicators.columns)} components")
        tprint_info(f"   Signal diversity score: {signal_quality['diversity_score']:.3f}")
        tprint_info(f"   Signal range: [{regime_signal.min():.3f}, {regime_signal.max():.3f}]")
        
        return regime_signal, diagnostics
    
    def _create_multiscale_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create regime indicators at multiple time scales."""
        tprint_info("📊 Creating multi-scale regime indicators...")
        
        regime_indicators = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change()
        
        # === VOLATILITY REGIMES (Multi-scale) ===
        if self.use_multiscale:
            # Short-term volatility (20-period)
            vol_short = returns.rolling(20).std()
            vol_short_z = (vol_short - vol_short.rolling(100).mean()) / (vol_short.rolling(100).std() + 1e-8)
            regime_indicators['vol_short'] = vol_short_z
            
            # Medium-term volatility (50-period)
            vol_med = returns.rolling(50).std()
            vol_med_z = (vol_med - vol_med.rolling(200).mean()) / (vol_med.rolling(200).std() + 1e-8)
            regime_indicators['vol_med'] = vol_med_z
            
            # Volatility acceleration (change in volatility)
            vol_accel = vol_short.diff(10) / (vol_short.rolling(20).mean() + 1e-8)
            regime_indicators['vol_accel'] = vol_accel
        else:
            # Single scale
            vol_20 = returns.rolling(20).std()
            vol_z = (vol_20 - vol_20.rolling(252).mean()) / (vol_20.rolling(252).std() + 1e-8)
            regime_indicators['vol_regime'] = vol_z
        
        # === TREND REGIMES (Multi-scale) ===
        if self.use_multiscale:
            # Short-term trend (20-period SMA)
            sma_short = df['close'].rolling(20).mean()
            trend_short = (df['close'] - sma_short) / (sma_short + 1e-8)
            regime_indicators['trend_short'] = trend_short
            
            # Long-term trend (100-period SMA)
            sma_long = df['close'].rolling(100).mean()
            trend_long = (df['close'] - sma_long) / (sma_long + 1e-8)
            regime_indicators['trend_long'] = trend_long
            
            # Trend strength (ADX-style indicator)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(14).mean()
            trend_strength = np.abs(trend_short) / (atr / df['close'] + 1e-8)
            regime_indicators['trend_strength'] = trend_strength
        else:
            # Single scale
            sma_50 = df['close'].rolling(50).mean()
            trend = (df['close'] - sma_50) / (sma_50 + 1e-8)
            regime_indicators['trend_regime'] = trend
        
        # === VOLUME REGIMES (Multi-scale) ===
        if self.use_multiscale:
            # Volume z-score (short)
            volume_ma_short = df['volume'].rolling(20).mean()
            volume_std_short = df['volume'].rolling(20).std()
            volume_z_short = (df['volume'] - volume_ma_short) / (volume_std_short + 1e-8)
            regime_indicators['volume_short'] = volume_z_short
            
            # Volume trend (increasing/decreasing)
            volume_trend = df['volume'].rolling(20).mean() / (df['volume'].rolling(100).mean() + 1e-8)
            regime_indicators['volume_trend'] = np.log1p(volume_trend)
            
            # Volume-price correlation (money flow)
            price_change = returns
            volume_change = df['volume'].pct_change()
            volume_price_corr = price_change.rolling(20).corr(volume_change)
            regime_indicators['volume_price_corr'] = volume_price_corr.fillna(0)
        else:
            # Single scale
            volume_ma = df['volume'].rolling(252).mean()
            volume_std = df['volume'].rolling(252).std()
            volume_z = (df['volume'] - volume_ma) / (volume_std + 1e-8)
            regime_indicators['volume_regime'] = volume_z
        
        # === MOMENTUM REGIMES (Multi-scale) ===
        if self.use_multiscale:
            # RSI (14-period)
            price_diff = df['close'].diff()
            gains = price_diff.where(price_diff > 0, 0).rolling(14).mean()
            losses = -price_diff.where(price_diff < 0, 0).rolling(14).mean()
            rs = gains / (losses + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            rsi_normalized = (rsi - 50) / 50  # Normalize to [-1, 1]
            regime_indicators['momentum_rsi'] = rsi_normalized
            
            # Rate of change (10-period)
            roc = (df['close'] / df['close'].shift(10) - 1) * 100
            regime_indicators['momentum_roc'] = roc / 10  # Normalize
            
            # Momentum acceleration
            mom = df['close'].diff(10)
            mom_accel = mom.diff(5) / (mom.rolling(10).std() + 1e-8)
            regime_indicators['momentum_accel'] = mom_accel
        else:
            # Single scale
            price_diff = df['close'].diff(14)
            avg_gain = price_diff[price_diff > 0].rolling(14).mean().fillna(0)
            avg_loss = -price_diff[price_diff < 0].rolling(14).mean().fillna(0)
            rs = avg_gain / (avg_loss + 1e-8)
            rsi_style = (rs / (1 + rs)) * 2 - 1
            regime_indicators['momentum_regime'] = rsi_style
        
        # === ADDITIONAL DISCRIMINATIVE INDICATORS ===
        # Range expansion/contraction (regime transitions)
        price_range = (df['high'] - df['low']) / df['close']
        range_ma = price_range.rolling(50).mean()
        range_z = (price_range - range_ma) / (price_range.rolling(50).std() + 1e-8)
        regime_indicators['range_regime'] = range_z
        
        # Bid-ask spread proxy (high-low as % of close)
        spread_proxy = (df['high'] - df['low']) / df['close'] * 100
        spread_z = (spread_proxy - spread_proxy.rolling(100).mean()) / (spread_proxy.rolling(100).std() + 1e-8)
        regime_indicators['spread_regime'] = spread_z
        
        # Fill NaN values properly
        regime_indicators = regime_indicators.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        # Clip extreme values (beyond ±5 std devs)
        for col in regime_indicators.columns:
            regime_indicators[col] = regime_indicators[col].clip(-5, 5)
        
        tprint_success(f"✅ Created {len(regime_indicators.columns)} regime indicators")
        
        return regime_indicators
    
    def _validate_component_diversity(self, regime_indicators: pd.DataFrame) -> Dict:
        """Validate that components are sufficiently diverse (not too correlated)."""
        tprint_info("🔍 Validating component diversity...")
        
        # Calculate correlation matrix
        corr_matrix = regime_indicators.corr().abs()
        
        # Find high correlations (excluding diagonal)
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr = corr_matrix.max().max()
        mean_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        # Count high correlations
        high_corr_count = (corr_matrix > self.correlation_threshold).sum().sum() / 2
        
        # Assess diversity
        diversity_score = 1.0 - mean_corr
        
        if max_corr > 0.9:
            tprint_warning(f"⚠️ Very high correlation detected: {max_corr:.3f}")
        if high_corr_count > len(regime_indicators.columns) / 2:
            tprint_warning(f"⚠️ Many components highly correlated: {high_corr_count} pairs")
        
        metrics = {
            'max_correlation': float(max_corr),
            'mean_correlation': float(mean_corr),
            'high_corr_pairs': int(high_corr_count),
            'diversity_score': float(diversity_score),
            'correlation_matrix': corr_matrix
        }
        
        tprint_info(f"   Max correlation: {max_corr:.3f}")
        tprint_info(f"   Mean correlation: {mean_corr:.3f}")
        tprint_info(f"   Diversity score: {diversity_score:.3f}")
        
        return metrics
    
    def _apply_nonlinear_transforms(self, regime_indicators: pd.DataFrame) -> pd.DataFrame:
        """Apply non-linear transformations to enhance regime separation."""
        tprint_info("🔄 Applying non-linear transformations...")
        
        transformed = regime_indicators.copy()
        
        for col in regime_indicators.columns:
            # Hyperbolic tangent (squashes extremes, enhances mid-range)
            transformed[f'{col}_tanh'] = np.tanh(regime_indicators[col])
            
            # Sign-preserving square root (enhances small values, compresses large)
            sign = np.sign(regime_indicators[col])
            transformed[f'{col}_sqrt'] = sign * np.sqrt(np.abs(regime_indicators[col]))
        
        tprint_info(f"   Added {len(transformed.columns) - len(regime_indicators.columns)} transformed features")
        
        return transformed
    
    def _compute_adaptive_weights(self, regime_indicators: pd.DataFrame) -> Dict[str, float]:
        """Compute adaptive weights based on component independence."""
        tprint_info("⚖️ Computing adaptive weights...")
        
        # Calculate correlation matrix
        corr_matrix = regime_indicators.corr().abs()
        
        # For each component, calculate average correlation with others
        avg_corr_with_others = {}
        for col in regime_indicators.columns:
            other_cols = [c for c in regime_indicators.columns if c != col]
            avg_corr = corr_matrix.loc[col, other_cols].mean()
            avg_corr_with_others[col] = avg_corr
        
        # Weight inversely proportional to correlation
        # Lower correlation = higher weight (more unique information)
        raw_weights = {col: 1.0 / (1.0 + corr) for col, corr in avg_corr_with_others.items()}
        
        # Normalize to sum to 1
        total_weight = sum(raw_weights.values())
        weights = {col: w / total_weight for col, w in raw_weights.items()}
        
        # Show top 5 weighted components
        top_components = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        tprint_info("   Top weighted components:")
        for comp, weight in top_components:
            tprint_info(f"      {comp}: {weight:.3f}")
        
        return weights
    
    def _get_fixed_weights(self, regime_indicators: pd.DataFrame) -> Dict[str, float]:
        """Get fixed weights for components."""
        n_components = len(regime_indicators.columns)
        
        # More balanced weights than original
        if self.use_multiscale:
            # Equal weighting for multi-scale (let data speak)
            weights = {col: 1.0 / n_components for col in regime_indicators.columns}
        else:
            # Original-style weights
            weights = {
                'vol_regime': 0.30,
                'trend_regime': 0.30,
                'volume_regime': 0.20,
                'momentum_regime': 0.20
            }
        
        return weights
    
    def _construct_composite(self, regime_indicators: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """Construct weighted composite signal."""
        tprint_info("🔨 Constructing composite signal...")
        
        # Weight each component
        regime_signal = pd.Series(0.0, index=regime_indicators.index)
        for col, weight in weights.items():
            if col in regime_indicators.columns:
                regime_signal += weight * regime_indicators[col]
        
        # Standardize final signal
        regime_signal = (regime_signal - regime_signal.mean()) / (regime_signal.std() + 1e-8)
        
        # Replace inf/nan
        regime_signal = regime_signal.replace([np.inf, -np.inf], 0).fillna(0)
        
        return regime_signal
    
    def _validate_signal_quality(self, regime_signal: pd.Series, regime_indicators: pd.DataFrame) -> Dict:
        """Validate the quality of the constructed signal."""
        tprint_info("✅ Validating signal quality...")
        
        # 1. Check for sufficient variance
        signal_std = regime_signal.std()
        signal_range = regime_signal.max() - regime_signal.min()
        
        # 2. Check for temporal structure (autocorrelation)
        autocorr_lag1 = regime_signal.autocorr(lag=1)
        autocorr_lag10 = regime_signal.autocorr(lag=10)
        
        # 3. Check for non-uniformity (test if normally distributed)
        _, normality_pvalue = stats.normaltest(regime_signal.dropna())
        
        # 4. Check for regime switches (look for transitions)
        signal_diff = regime_signal.diff().abs()
        transition_rate = (signal_diff > signal_diff.quantile(0.75)).sum() / len(regime_signal)
        
        # 5. Diversity score
        diversity_score = min(1.0, signal_range / 10.0)  # Ideally spans ±5 std devs
        
        quality = {
            'std': float(signal_std),
            'range': float(signal_range),
            'autocorr_lag1': float(autocorr_lag1),
            'autocorr_lag10': float(autocorr_lag10),
            'normality_pvalue': float(normality_pvalue),
            'transition_rate': float(transition_rate),
            'diversity_score': float(diversity_score)
        }
        
        # Warnings
        if signal_std < 0.5:
            tprint_warning(f"⚠️ Low signal variance: {signal_std:.3f}")
        if signal_range < 3.0:
            tprint_warning(f"⚠️ Narrow signal range: {signal_range:.3f}")
        if transition_rate < 0.1:
            tprint_warning(f"⚠️ Low transition rate: {transition_rate:.3f}")
        
        return quality


def create_improved_regime_signal(df: pd.DataFrame, 
                                   use_nonlinear: bool = True,
                                   use_multiscale: bool = True,
                                   use_adaptive_weights: bool = True) -> Tuple[pd.Series, Dict]:
    """
    Convenience function to create improved regime signal.
    
    Args:
        df: DataFrame with OHLCV data
        use_nonlinear: Apply non-linear transformations
        use_multiscale: Use multi-scale indicators
        use_adaptive_weights: Use adaptive weighting
        
    Returns:
        Tuple of (regime_signal, diagnostics)
    """
    builder = ImprovedRegimeSignalBuilder(
        use_nonlinear_transforms=use_nonlinear,
        use_multiscale=use_multiscale,
        use_adaptive_weighting=use_adaptive_weights
    )
    
    return builder.build_regime_signal(df)


if __name__ == "__main__":
    # Demo with synthetic data
    tprint("=" * 80)
    tprint("🎯 IMPROVED MS-DR SIGNAL CONSTRUCTION DEMO")
    tprint("=" * 80)
    
    # Create synthetic market data
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.cumsum(np.random.randn(n_samples) * 0.01) + 3000,
        'high': np.cumsum(np.random.randn(n_samples) * 0.01) + 3010,
        'low': np.cumsum(np.random.randn(n_samples) * 0.01) + 2990,
        'close': np.cumsum(np.random.randn(n_samples) * 0.01) + 3000,
        'volume': np.random.uniform(500, 2000, n_samples)
    })
    df.set_index('timestamp', inplace=True)
    
    # Build signal with all enhancements
    regime_signal, diagnostics = create_improved_regime_signal(
        df,
        use_nonlinear=True,
        use_multiscale=True,
        use_adaptive_weights=True
    )
    
    tprint("\n" + "=" * 80)
    tprint("📊 SIGNAL DIAGNOSTICS")
    tprint("=" * 80)
    
    tprint(f"\n✅ Composite Signal Stats:")
    tprint(f"   Shape: {regime_signal.shape}")
    tprint(f"   Range: [{regime_signal.min():.3f}, {regime_signal.max():.3f}]")
    tprint(f"   Mean: {regime_signal.mean():.3f}, Std: {regime_signal.std():.3f}")
    
    tprint(f"\n✅ Quality Metrics:")
    for key, value in diagnostics['signal_quality'].items():
        tprint(f"   {key}: {value:.4f}")
    
    tprint(f"\n✅ Diversity Metrics:")
    for key, value in diagnostics['diversity_metrics'].items():
        if key != 'correlation_matrix':
            tprint(f"   {key}: {value}")
    
    tprint("\n" + "=" * 80)
    tprint("✅ DEMO COMPLETE")
    tprint("=" * 80)

