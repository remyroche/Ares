"""
CV Ratio Enhancement Strategies - Practical Implementation.

This module implements the strategies documented in CV_RATIO_IMPROVEMENT_STRATEGIES.md
to improve clustering quality, particularly the CV ratio (variance ratio).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import calinski_harabasz_score

from src.utils.tprint import tprint

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class RegimeDiscriminativeFeatures:
    """
    Add regime-discriminative features designed to maximize between-regime variance.
    These features are specifically engineered to capture regime transitions and characteristics.
    """
    
    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime-discriminative features to enhance clustering quality.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV and basic features
            
        Returns:
        --------
        df : pd.DataFrame
            DataFrame with additional regime-discriminative features
        """
        try:
            tprint("🔧 Adding regime-discriminative features...", "INFO")
            
            # 1. Volatility Regime Features (HIGH IMPACT)
            if 'volatility_20' in df.columns and 'volatility_5' in df.columns:
                # Volatility regime Z-score
                vol_mean_60 = df['volatility_20'].rolling(60).mean()
                vol_std_60 = df['volatility_20'].rolling(60).std()
                df['vol_regime_zscore'] = (df['volatility_20'] - vol_mean_60) / (vol_std_60 + 1e-8)
                
                # Volatility percentile rank (0-1 scale)
                df['vol_regime_percentile'] = df['volatility_20'].rolling(252).rank(pct=True)
                
                # Volatility transition indicator (short-term vs long-term)
                df['vol_regime_transition'] = np.log((df['volatility_5'] + 1e-8) / (df['volatility_20'] + 1e-8))
            
            # 2. Return Distribution Features (HIGH IMPACT)
            if 'close_return' in df.columns:
                # Return skewness (asymmetry indicator)
                df['return_skew_20'] = df['close_return'].rolling(20).skew()
                
                # Return kurtosis (tail risk indicator)  
                df['return_kurt_20'] = df['close_return'].rolling(20).kurt()
                
                # Return regime Z-score
                ret_mean_60 = df['close_return'].rolling(60).mean()
                ret_std_60 = df['close_return'].rolling(60).std()
                df['return_regime_zscore'] = (df['close_return'] - ret_mean_60) / (ret_std_60 + 1e-8)
            
            # 3. Trend Strength Features (MEDIUM IMPACT)
            if 'close_sma_20' in df.columns and 'close_sma_5' in df.columns and 'atr' in df.columns:
                # Trend strength (normalized by volatility)
                df['trend_strength'] = np.abs(df['close_sma_20'] - df['close_sma_5']) / (df['atr'] + 1e-8)
                
                # Trend consistency (directional bias)
                df['trend_consistency'] = (df['close_return'] > 0).rolling(20).mean() - 0.5
                
                # Trend acceleration
                ma_diff_5 = df['close_sma_20'].diff(5)
                ma_diff_20 = df['close_sma_20'].diff(20)
                df['trend_acceleration'] = ma_diff_5 / (ma_diff_20 + 1e-8)
            
            # 4. Multi-Timeframe Volatility Features (MEDIUM IMPACT)
            if 'close_return' in df.columns:
                timeframes = [5, 10, 20, 40, 60]
                vols = []
                for tf in timeframes:
                    vol = df['close_return'].rolling(tf).std() * np.sqrt(252)
                    df[f'vol_{tf}d'] = vol
                    vols.append(vol)
                
                # Timeframe alignment (regime convergence indicator)
                # Low std across timeframes = consistent regime
                vol_matrix = pd.concat(vols, axis=1)
                df['vol_timeframe_alignment'] = vol_matrix.std(axis=1)
            
            # 5. Volume Regime Features (LOW-MEDIUM IMPACT)
            if 'volume_sma_20' in df.columns and 'volume' in df.columns:
                # Volume regime (short vs long term)
                df['volume_regime'] = df['volume'] / (df['volume_sma_20'] + 1e-8)
                
                # Volume regime percentile
                df['volume_regime_percentile'] = df['volume'].rolling(252).rank(pct=True)
            
            # 6. Price-Volume Divergence (MEDIUM IMPACT)
            if 'close_return' in df.columns and 'volume_return' in df.columns:
                # Rolling correlation between price and volume
                df['price_volume_corr_20'] = df['close_return'].rolling(20).corr(df['volume_return'])
                
                # Divergence indicator (uncorrelated price-volume = regime change)
                df['price_volume_divergence'] = 1.0 - np.abs(df['price_volume_corr_20'])
            
            # 7. Regime Persistence Indicator (HIGH IMPACT)
            if 'vol_regime_percentile' in df.columns:
                # How long has current regime persisted?
                df['regime_persistence'] = RegimeDiscriminativeFeatures._calculate_regime_persistence(
                    df['vol_regime_percentile']
                )
            
            tprint(f"✅ Added regime-discriminative features: {len([c for c in df.columns if 'regime' in c or 'trend_' in c])} features", "SUCCESS")
            
            return df
            
        except Exception as e:
            tprint(f"⚠️ Failed to add some regime features: {e}", "WARNING")
            return df
    
    @staticmethod
    def _calculate_regime_persistence(regime_signal: pd.Series, threshold: float = 0.5) -> pd.Series:
        """Calculate how long the current regime has persisted."""
        # Binary regime indicator (above/below median)
        regime_binary = (regime_signal > threshold).astype(int)
        
        # Count consecutive periods in same regime
        persistence = pd.Series(0, index=regime_signal.index)
        count = 0
        prev_regime = -1
        
        for i, regime in enumerate(regime_binary):
            if regime == prev_regime:
                count += 1
            else:
                count = 1
            persistence.iloc[i] = count
            prev_regime = regime
        
        return persistence


class AdaptiveWeightScheduler:
    """
    Adaptive weight scheduler that adjusts optimization weights based on iteration progress.
    Early iterations: balanced exploration
    Late iterations: aggressive CV optimization
    """
    
    def __init__(self, max_iterations: int = 30):
        """
        Initialize adaptive weight scheduler.
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of optimization iterations
        """
        self.max_iterations = max_iterations
    
    def get_weights(self, iteration: int) -> Dict[str, float]:
        """
        Get adaptive weights for current iteration.
        
        Parameters:
        -----------
        iteration : int
            Current iteration number (0-based)
            
        Returns:
        --------
        weights : dict
            Dictionary with weight values for each component
        """
        progress = min(1.0, iteration / self.max_iterations)
        
        # Gradually increase CV weight (0.45 → 0.55)
        # As we progress, focus more on CV ratio
        w_cv = 0.45 + 0.10 * progress
        
        # Gradually decrease balance weight (0.05 → 0.02)
        # Balance becomes less important as we optimize
        w_bal = 0.05 * (1 - 0.6 * progress)
        
        # Keep temporal and silhouette relatively stable with slight emphasis
        # Temporal: 0.35 → 0.32 (slight decrease to accommodate CV increase)
        w_temp = 0.35 - 0.03 * progress
        
        # Silhouette: 0.15 → 0.16 (slight increase for quality)
        w_sil = 0.15 + 0.01 * progress
        
        # Normalize to sum to 1.0
        total = w_cv + w_temp + w_sil + w_bal
        
        weights = {
            'w_cv': w_cv / total,
            'w_temp': w_temp / total,
            'w_sil': w_sil / total,
            'w_bal': w_bal / total
        }
        
        # Log adaptive weights at key milestones
        if iteration % 5 == 0 or iteration == 0:
            tprint(f"📊 Iteration {iteration}: Adaptive weights - "
                  f"CV={weights['w_cv']:.3f}, Temp={weights['w_temp']:.3f}, "
                  f"Sil={weights['w_sil']:.3f}, Bal={weights['w_bal']:.3f}", "INFO")
        
        return weights


class EnhancedVarianceRatioCalculator:
    """
    Enhanced variance ratio calculator that combines multiple variance-based metrics
    for more robust CV ratio estimation.
    """
    
    @staticmethod
    def calculate_enhanced_cv(features: np.ndarray, 
                             assignments: np.ndarray,
                             include_calinski_harabasz: bool = True) -> Dict[str, float]:
        """
        Calculate enhanced CV ratio incorporating multiple variance metrics.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix (n_samples, n_features)
        assignments : np.ndarray
            Cluster assignments (n_samples,)
        include_calinski_harabasz : bool
            Whether to include Calinski-Harabasz score
            
        Returns:
        --------
        metrics : dict
            Dictionary with variance ratio metrics
        """
        try:
            # 1. Standard variance ratio (between / within)
            cv_ratio = EnhancedVarianceRatioCalculator._calculate_standard_cv(
                features, assignments
            )
            
            # 2. Calinski-Harabasz score (another variance ratio metric)
            ch_score = 0.0
            ch_normalized = 0.0
            if include_calinski_harabasz and len(np.unique(assignments)) > 1:
                try:
                    ch_score = calinski_harabasz_score(features, assignments)
                    # Normalize CH score to [0, 1] range using sigmoid-like function
                    ch_normalized = ch_score / (ch_score + 100)
                except:
                    ch_normalized = 0.0
            
            # 3. Combined variance ratio (weighted average)
            # 70% standard CV, 30% Calinski-Harabasz
            combined_cv = 0.7 * cv_ratio + 0.3 * ch_normalized
            
            return {
                'cv_ratio': float(cv_ratio),
                'calinski_harabasz': float(ch_score),
                'ch_normalized': float(ch_normalized),
                'combined_cv': float(combined_cv)
            }
            
        except Exception as e:
            tprint(f"⚠️ Enhanced CV calculation failed: {e}", "WARNING")
            return {
                'cv_ratio': 0.0,
                'calinski_harabasz': 0.0,
                'ch_normalized': 0.0,
                'combined_cv': 0.0
            }
    
    @staticmethod
    def _calculate_standard_cv(features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate standard variance ratio (between / within)."""
        try:
            unique_clusters = np.unique(assignments)
            n_clusters = len(unique_clusters)
            
            if n_clusters < 2:
                return 0.0
            
            # Global mean
            global_mean = np.mean(features, axis=0)
            
            # Within-cluster variance
            within_var = 0.0
            for cluster_id in unique_clusters:
                cluster_mask = assignments == cluster_id
                cluster_features = features[cluster_mask]
                if len(cluster_features) > 0:
                    cluster_var = np.var(cluster_features, axis=0).mean()
                    within_var += cluster_var * len(cluster_features)
            within_var /= len(features)
            
            # Between-cluster variance
            between_var = 0.0
            for cluster_id in unique_clusters:
                cluster_mask = assignments == cluster_id
                cluster_features = features[cluster_mask]
                if len(cluster_features) > 0:
                    cluster_mean = np.mean(cluster_features, axis=0)
                    cluster_diff = cluster_mean - global_mean
                    between_var += np.sum(cluster_diff ** 2) * len(cluster_features)
            between_var /= len(features)
            
            # Variance ratio
            if within_var > 0:
                cv_ratio = between_var / within_var
            else:
                cv_ratio = 0.0
            
            return float(cv_ratio)
            
        except:
            return 0.0


def apply_cv_enhancement_strategies(df: pd.DataFrame,
                                    add_regime_features: bool = True) -> pd.DataFrame:
    """
    Apply CV ratio enhancement strategies to feature dataframe.
    
    This is the main entry point for applying enhancement strategies.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with basic features
    add_regime_features : bool
        Whether to add regime-discriminative features
        
    Returns:
    --------
    df : pd.DataFrame
        Enhanced dataframe with additional features
    """
    try:
        tprint("\n🚀 Applying CV Enhancement Strategies...", "SUCCESS")
        tprint("="*80, "INFO")
        
        initial_features = len(df.columns)
        
        # Strategy 1: Add regime-discriminative features
        if add_regime_features:
            df = RegimeDiscriminativeFeatures.add_features(df)
        
        final_features = len(df.columns)
        added_features = final_features - initial_features
        
        tprint(f"✅ CV Enhancement Complete: Added {added_features} discriminative features", "SUCCESS")
        tprint(f"   Total features: {initial_features} → {final_features}", "INFO")
        tprint("="*80, "INFO")
        
        return df
        
    except Exception as e:
        tprint(f"❌ CV enhancement failed: {e}", "ERROR")
        return df


    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
