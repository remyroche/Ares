"""
Production Feature Integration for Advanced Markov Models

This module integrates advanced Markov model features with the existing
feature engineering infrastructure, focusing on 1h timeframe with
leakage-safe multi-horizon features.

Key Features:
1. Integration with existing feature_engineer infrastructure
2. Focus on 1h timeframe as primary resolution
3. Multi-horizon windows (5, 20, 60 bars) on 1h data
4. Leakage-safe feature generation
5. Advanced Markov model specific features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
import pandas_ta as ta

from src.utils.logger import system_logger
from src.utils.data.feature_engineer import FeatureEngineer
from src.analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator

# Import existing feature components
try:
    from src.feature_generation.utils.step06_enhanced_feature_engineering import EnhancedFeatureEngineeringStep
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    warnings.warn("Enhanced feature engineering not available")

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    warnings.warn("ruptures not available - structural break detection limited")


@dataclass
class ProductionFeatureConfig:
    """Configuration for production feature integration."""
    # Primary timeframe (focus on 1h)
    primary_timeframe: str = "1h"
    
    # Multi-horizon windows (in bars of primary timeframe)
    horizons: List[int] = None  # [5, 20, 60] hours
    
    # Integration with existing features
    use_existing_orchestrator: bool = True
    use_existing_feature_engineer: bool = True
    
    # Advanced Markov features
    enable_structural_break_features: bool = True
    enable_duration_features: bool = True
    enable_regime_transition_features: bool = True
    
    # Feature filtering
    variance_threshold: float = 1e-6
    correlation_threshold: float = 0.90
    enable_pca_compression: bool = False  # Disable for production clarity
    
    # Rolling statistics (in hours)
    rolling_window_hours: int = 500  # ~3 weeks of 1h data
    min_periods_hours: int = 50      # ~2 days minimum
    
    # Leakage safety
    strict_no_lookahead: bool = True
    current_time_aware: bool = True
    
    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [1, 2, 4]  # 1h, 2h, 4h windows


class ProductionLeakageSafeFeatures:
    """
    Leakage-safe feature calculator integrated with existing infrastructure.
    Focuses on 1h timeframe with multi-horizon analysis.
    """
    
    def __init__(self, config: ProductionFeatureConfig):
        self.config = config
        self.logger = system_logger.getChild('ProductionLeakageSafeFeatures')
        
        # Initialize existing components
        self.feature_engineer = None
        self.orchestrator = None
        
        if self.config.use_existing_feature_engineer:
            try:
                self.feature_engineer = FeatureEngineer()
                self.logger.info("✅ Integrated with existing FeatureEngineer")
            except Exception as e:
                self.logger.warning(f"Could not initialize FeatureEngineer: {e}")
        
        if self.config.use_existing_orchestrator:
            try:
                # Basic config for orchestrator
                orchestrator_config = {
                    'feature_engineering_orchestrator': {},
                    'enable_multi_timeframe': True
                }
                self.orchestrator = FeatureEngineeringOrchestrator(orchestrator_config)
                self.logger.info("✅ Integrated with existing FeatureEngineeringOrchestrator")
            except Exception as e:
                self.logger.warning(f"Could not initialize orchestrator: {e}")
    
    def generate_production_features(self, 
                                   data: pd.DataFrame,
                                   symbol: str = "ETHUSDT",
                                   current_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Generate production-ready features for 1h timeframe.
        
        Args:
            data: 1h OHLCV data
            symbol: Trading symbol
            current_time: Current time for leakage-safe calculation
            
        Returns:
            Feature matrix with all generated features
        """
        self.logger.info(f"🔧 Generating production features for {symbol} on {self.config.primary_timeframe}")
        self.logger.info(f"📊 Data shape: {data.shape}, timespan: {data.index[0]} to {data.index[-1]}")
        
        # Validate 1h timeframe
        if not self._validate_1h_timeframe(data):
            self.logger.warning("Data may not be 1h timeframe - proceeding with caution")
        
        # Determine current index for leakage safety
        current_idx = self._get_current_index(data, current_time)
        
        features = pd.DataFrame(index=data.index)
        
        # Step 1: Generate existing features using orchestrator
        if self.orchestrator is not None:
            try:
                existing_features = self._generate_existing_features(data, current_idx)
                if not existing_features.empty:
                    features = pd.concat([features, existing_features], axis=1)
                    self.logger.info(f"✅ Added {len(existing_features.columns)} existing features")
            except Exception as e:
                self.logger.warning(f"Failed to generate existing features: {e}")
        
        # Step 2: Generate multi-horizon base features
        base_features = self._generate_multi_horizon_base_features(data, current_idx)
        features = pd.concat([features, base_features], axis=1)
        self.logger.info(f"✅ Added {len(base_features.columns)} multi-horizon base features")
        
        # Step 3: Generate advanced Markov features
        if self.config.enable_structural_break_features:
            break_features = self._generate_structural_break_features(data, current_idx)
            features = pd.concat([features, break_features], axis=1)
            self.logger.info(f"✅ Added {len(break_features.columns)} structural break features")
        
        if self.config.enable_duration_features:
            duration_features = self._generate_duration_features(data, current_idx)
            features = pd.concat([features, duration_features], axis=1)
            self.logger.info(f"✅ Added {len(duration_features.columns)} duration persistence features")
        
        if self.config.enable_regime_transition_features:
            transition_features = self._generate_transition_features(data, current_idx)
            features = pd.concat([features, transition_features], axis=1)
            self.logger.info(f"✅ Added {len(transition_features.columns)} regime transition features")
        
        # Step 4: Apply leakage-safe normalization
        features = self._apply_leakage_safe_normalization(features, current_idx)
        
        # Step 5: Apply feature filtering
        features = self._apply_production_filtering(features)
        
        self.logger.info(f"🎯 Production features generated: {len(features.columns)} features for {len(features)} observations")
        
        return features
    
    def _validate_1h_timeframe(self, data: pd.DataFrame) -> bool:
        """Validate that data is approximately 1h timeframe."""
        if len(data) < 2:
            return True  # Can't validate with insufficient data
        
        # Check time differences
        time_diffs = data.index.to_series().diff().dropna()
        median_diff = time_diffs.median()
        
        # Should be approximately 1 hour
        expected_diff = pd.Timedelta(hours=1)
        tolerance = pd.Timedelta(minutes=5)  # 5-minute tolerance
        
        is_valid = abs(median_diff - expected_diff) <= tolerance
        
        if not is_valid:
            self.logger.warning(f"Expected 1h intervals, got median: {median_diff}")
        
        return is_valid
    
    def _get_current_index(self, data: pd.DataFrame, current_time: Optional[pd.Timestamp]) -> Optional[int]:
        """Get current index for leakage-safe calculations."""
        if not self.config.strict_no_lookahead or current_time is None:
            return None  # Use all data
        
        # Find index closest to current time
        try:
            current_idx = data.index.get_indexer([current_time], method='ffill')[0]
            if current_idx == -1:
                current_idx = len(data) - 1
            return current_idx
        except Exception:
            return None
    
    async def _generate_existing_features(self, data: pd.DataFrame, current_idx: Optional[int]) -> pd.DataFrame:
        """Generate features using existing orchestrator."""
        try:
            # Prepare data in expected format
            klines_df = data.copy()
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in klines_df.columns:
                    self.logger.warning(f"Missing column {col}, using close price as fallback")
                    klines_df[col] = klines_df['close'] if 'close' in klines_df.columns else 100.0
            
            # Limit data if current_idx specified (leakage safety)
            if current_idx is not None:
                klines_df = klines_df.iloc[:current_idx + 1]
            
            # Generate features using orchestrator
            features_df = await self.orchestrator.generate_all_features(
                klines_df=klines_df,
                agg_trades_df=None,  # Not available in this context
                futures_df=None,     # Not available in this context
                sr_levels=None       # Could be added later
            )
            
            # Remove original OHLCV columns to avoid duplication
            feature_cols = [col for col in features_df.columns if col not in required_cols]
            
            if feature_cols:
                return features_df[feature_cols]
            else:
                return pd.DataFrame(index=data.index)
                
        except Exception as e:
            self.logger.error(f"Failed to generate existing features: {e}")
            return pd.DataFrame(index=data.index)
    
    def _generate_multi_horizon_base_features(self, data: pd.DataFrame, current_idx: Optional[int]) -> pd.DataFrame:
        """Generate multi-horizon base features for 1h timeframe."""
        features = pd.DataFrame(index=data.index)
        
        returns = data['close'].pct_change()
        
        for horizon in self.config.horizons:
            horizon_suffix = f"_{horizon}h"
            
            # Price-based features
            features[f'momentum{horizon_suffix}'] = data['close'].pct_change(horizon)
            features[f'volatility{horizon_suffix}'] = returns.rolling(
                horizon, min_periods=max(1, horizon//4)
            ).std()
            
            # Volume features (if available)
            if 'volume' in data.columns:
                vol_ma = data['volume'].rolling(horizon, min_periods=max(1, horizon//4)).mean()
                features[f'rel_volume{horizon_suffix}'] = data['volume'] / (vol_ma + 1e-8)
            
            # Technical indicators adapted for hourly data
            if horizon >= 5:  # Only for longer horizons
                # RSI adapted for hourly
                try:
                    rsi_values = ta.rsi(data['close'], length=horizon)
                    features[f'rsi{horizon_suffix}'] = rsi_values
                except:
                    features[f'rsi{horizon_suffix}'] = 50.0
                
                # Bollinger Band position
                bb = ta.bbands(data['close'], length=horizon)
                if bb is not None and not bb.empty:
                    bb_cols = [col for col in bb.columns if 'BBP' in col]
                    if bb_cols:
                        features[f'bb_position{horizon_suffix}'] = bb[bb_cols[0]]
            
            # Range-based features
            if all(col in data.columns for col in ['high', 'low']):
                rolling_high = data['high'].rolling(horizon, min_periods=max(1, horizon//4)).max()
                rolling_low = data['low'].rolling(horizon, min_periods=max(1, horizon//4)).min()
                features[f'price_position{horizon_suffix}'] = (
                    (data['close'] - rolling_low) / (rolling_high - rolling_low + 1e-8)
                )
                
                # True Range adapted for hourly
                tr = ta.true_range(data['high'], data['low'], data['close'])
                features[f'atr{horizon_suffix}'] = tr.rolling(horizon, min_periods=max(1, horizon//4)).mean()
        
        return features.fillna(0.0)
    
    def _generate_structural_break_features(self, data: pd.DataFrame, current_idx: Optional[int]) -> pd.DataFrame:
        """Generate structural break detection features for MSM."""
        features = pd.DataFrame(index=data.index)
        returns = data['close'].pct_change()
        
        for horizon in self.config.horizons:
            horizon_suffix = f"_{horizon}h"
            
            # Variance ratio test for parameter stability
            features[f'variance_ratio{horizon_suffix}'] = self._rolling_variance_ratio(
                returns, horizon, current_idx
            )
            
            # Parameter drift indicator
            features[f'param_drift{horizon_suffix}'] = self._rolling_parameter_drift(
                returns, horizon, current_idx
            )
            
            # Structural change detector (CUSUM-like)
            features[f'cusum_stat{horizon_suffix}'] = self._rolling_cusum_statistic(
                returns, horizon, current_idx
            )
            
            # Correlation stability (price-volume if available)
            if 'volume' in data.columns:
                features[f'corr_stability{horizon_suffix}'] = self._rolling_correlation_stability(
                    returns, data['volume'], horizon, current_idx
                )
            
            # Regime entropy proxy
            features[f'regime_entropy{horizon_suffix}'] = self._rolling_regime_entropy(
                returns, horizon, current_idx
            )
        
        return features.fillna(0.0)
    
    def _generate_duration_features(self, data: pd.DataFrame, current_idx: Optional[int]) -> pd.DataFrame:
        """Generate duration persistence features for HSMM."""
        features = pd.DataFrame(index=data.index)
        returns = data['close'].pct_change()
        
        for horizon in self.config.horizons:
            horizon_suffix = f"_{horizon}h"
            
            # Autocorrelation of volatility (regime persistence proxy)
            vol = returns.rolling(max(2, horizon//4), min_periods=2).std()
            features[f'vol_autocorr{horizon_suffix}'] = self._rolling_autocorr(
                vol, 1, horizon, current_idx
            )
            
            # Trend persistence
            momentum = data['close'].pct_change(max(1, horizon//4))
            features[f'trend_persistence{horizon_suffix}'] = self._rolling_autocorr(
                momentum, 1, horizon, current_idx
            )
            
            # Mean reversion speed
            features[f'mean_reversion{horizon_suffix}'] = self._rolling_mean_reversion_speed(
                returns, horizon, current_idx
            )
            
            # Volatility clustering intensity
            vol_changes = vol.diff().abs()
            features[f'vol_clustering{horizon_suffix}'] = vol_changes.rolling(
                horizon, min_periods=max(1, horizon//4)
            ).mean()
            
            # State duration proxy (run lengths)
            features[f'state_duration_proxy{horizon_suffix}'] = self._rolling_state_duration_proxy(
                returns, horizon, current_idx
            )
        
        return features.fillna(0.0)
    
    def _generate_transition_features(self, data: pd.DataFrame, current_idx: Optional[int]) -> pd.DataFrame:
        """Generate regime transition features."""
        features = pd.DataFrame(index=data.index)
        returns = data['close'].pct_change()
        
        for horizon in self.config.horizons:
            horizon_suffix = f"_{horizon}h"
            
            # Transition volatility indicator
            vol = returns.rolling(max(2, horizon//8), min_periods=2).std()
            vol_changes = vol.diff().abs()
            features[f'transition_vol{horizon_suffix}'] = vol_changes.rolling(
                horizon, min_periods=max(1, horizon//4)
            ).mean()
            
            # Regime switching probability proxy
            features[f'regime_switch_prob{horizon_suffix}'] = self._rolling_regime_switch_prob(
                returns, horizon, current_idx
            )
            
            # Transition timing indicator
            features[f'transition_timing{horizon_suffix}'] = self._rolling_transition_timing(
                vol, horizon, current_idx
            )
            
            # Market stress indicator (for transition detection)
            if 'volume' in data.columns:
                price_vol_corr = self._rolling_correlation(
                    returns.abs(), data['volume'], horizon//2, current_idx
                )
                features[f'market_stress{horizon_suffix}'] = price_vol_corr.abs()
        
        return features.fillna(0.0)
    
    def _rolling_variance_ratio(self, series: pd.Series, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate rolling variance ratio test statistic."""
        result = []
        
        for i in range(len(series)):
            if current_idx is not None and i > current_idx:
                result.append(np.nan)
                continue
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                result.append(1.0)
                continue
            
            data_window = series.iloc[start_idx:end_idx].dropna()
            
            if len(data_window) < 10:
                result.append(1.0)
                continue
            
            # Split window and compare variances
            mid = len(data_window) // 2
            var1 = data_window.iloc[:mid].var()
            var2 = data_window.iloc[mid:].var()
            
            if var1 > 0 and var2 > 0:
                ratio = max(var1, var2) / min(var1, var2)
            else:
                ratio = 1.0
            
            result.append(ratio)
        
        return pd.Series(result, index=series.index)
    
    def _rolling_parameter_drift(self, series: pd.Series, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate rolling parameter drift indicator."""
        result = []
        
        for i in range(len(series)):
            if current_idx is not None and i > current_idx:
                result.append(np.nan)
                continue
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                result.append(0.0)
                continue
            
            data_window = series.iloc[start_idx:end_idx].dropna()
            
            if len(data_window) < 10:
                result.append(0.0)
                continue
            
            # Compare first and second half statistics
            mid = len(data_window) // 2
            first_half = data_window.iloc[:mid]
            second_half = data_window.iloc[mid:]
            
            mean_drift = abs(first_half.mean() - second_half.mean())
            var_drift = abs(first_half.var() - second_half.var())
            
            pooled_std = np.sqrt((first_half.var() + second_half.var()) / 2)
            
            if pooled_std > 0:
                drift = (mean_drift + var_drift) / pooled_std
            else:
                drift = 0.0
            
            result.append(drift)
        
        return pd.Series(result, index=series.index)
    
    def _rolling_cusum_statistic(self, series: pd.Series, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate rolling CUSUM statistic."""
        result = []
        
        for i in range(len(series)):
            if current_idx is not None and i > current_idx:
                result.append(np.nan)
                continue
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                result.append(0.0)
                continue
            
            data_window = series.iloc[start_idx:end_idx].dropna()
            
            if len(data_window) < 5:
                result.append(0.0)
                continue
            
            # CUSUM statistic
            mean_val = data_window.mean()
            cumsum = np.cumsum(data_window - mean_val)
            std_val = data_window.std()
            
            if std_val > 0:
                cusum_stat = np.max(np.abs(cumsum)) / std_val
            else:
                cusum_stat = 0.0
            
            result.append(cusum_stat)
        
        return pd.Series(result, index=series.index)
    
    def _rolling_correlation_stability(self, series1: pd.Series, series2: pd.Series, 
                                     window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate rolling correlation stability."""
        result = []
        
        for i in range(len(series1)):
            if current_idx is not None and i > current_idx:
                result.append(np.nan)
                continue
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                result.append(1.0)
                continue
            
            data1 = series1.iloc[start_idx:end_idx].dropna()
            data2 = series2.iloc[start_idx:end_idx].dropna()
            
            # Align series
            common_idx = data1.index.intersection(data2.index)
            if len(common_idx) < window // 4:
                result.append(1.0)
                continue
            
            data1_aligned = data1[common_idx]
            data2_aligned = data2[common_idx]
            
            # Calculate correlation in first and second half
            mid = len(data1_aligned) // 2
            try:
                corr1 = data1_aligned.iloc[:mid].corr(data2_aligned.iloc[:mid])
                corr2 = data1_aligned.iloc[mid:].corr(data2_aligned.iloc[mid:])
                
                if not (np.isnan(corr1) or np.isnan(corr2)):
                    stability = 1.0 - abs(corr1 - corr2)
                else:
                    stability = 1.0
            except:
                stability = 1.0
            
            result.append(stability)
        
        return pd.Series(result, index=series1.index)
    
    def _rolling_regime_entropy(self, series: pd.Series, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate rolling regime entropy proxy."""
        result = []
        
        for i in range(len(series)):
            if current_idx is not None and i > current_idx:
                result.append(np.nan)
                continue
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                result.append(0.5)
                continue
            
            data_window = series.iloc[start_idx:end_idx].dropna()
            
            if len(data_window) < 10:
                result.append(0.5)
                continue
            
            # Create regime proxy based on volatility
            vol = data_window.rolling(max(2, window//10), min_periods=2).std()
            vol_quantiles = vol.quantile([0.33, 0.67])
            
            # Assign regime labels
            regimes = np.zeros(len(vol))
            regimes[vol <= vol_quantiles.iloc[0]] = 0
            regimes[(vol > vol_quantiles.iloc[0]) & (vol <= vol_quantiles.iloc[1])] = 1
            regimes[vol > vol_quantiles.iloc[1]] = 2
            
            # Calculate entropy
            unique, counts = np.unique(regimes, return_counts=True)
            probs = counts / len(regimes)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            
            # Normalize by max entropy
            max_entropy = np.log(3)
            normalized_entropy = entropy / max_entropy
            
            result.append(normalized_entropy)
        
        return pd.Series(result, index=series.index)
    
    def _rolling_autocorr(self, series: pd.Series, lag: int, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate rolling autocorrelation."""
        result = []
        
        for i in range(len(series)):
            if current_idx is not None and i > current_idx:
                result.append(np.nan)
                continue
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                result.append(0.0)
                continue
            
            data_window = series.iloc[start_idx:end_idx].dropna()
            
            if len(data_window) > lag:
                try:
                    autocorr = data_window.autocorr(lag=lag)
                    result.append(autocorr if not np.isnan(autocorr) else 0.0)
                except:
                    result.append(0.0)
            else:
                result.append(0.0)
        
        return pd.Series(result, index=series.index)
    
    def _rolling_mean_reversion_speed(self, series: pd.Series, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate rolling mean reversion speed."""
        result = []
        
        for i in range(len(series)):
            if current_idx is not None and i > current_idx:
                result.append(np.nan)
                continue
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                result.append(0.0)
                continue
            
            data_window = series.iloc[start_idx:end_idx].dropna()
            
            if len(data_window) < 3:
                result.append(0.0)
                continue
            
            # AR(1) coefficient as mean reversion proxy
            y = data_window.iloc[1:].values
            x = data_window.iloc[:-1].values
            
            if len(x) > 0 and np.std(x) > 0:
                corr = np.corrcoef(x, y)[0, 1]
                reversion_speed = max(0.0, -corr)
            else:
                reversion_speed = 0.0
            
            result.append(reversion_speed)
        
        return pd.Series(result, index=series.index)
    
    def _rolling_state_duration_proxy(self, series: pd.Series, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate rolling state duration proxy."""
        result = []
        
        for i in range(len(series)):
            if current_idx is not None and i > current_idx:
                result.append(np.nan)
                continue
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                result.append(1.0)
                continue
            
            data_window = series.iloc[start_idx:end_idx].dropna()
            
            if len(data_window) < 10:
                result.append(1.0)
                continue
            
            # Calculate volatility regime proxy
            vol = data_window.rolling(max(2, window//10), min_periods=2).std()
            vol_median = vol.median()
            
            # Binary high/low volatility
            high_vol = (vol > vol_median).astype(int)
            
            # Count run lengths
            runs = []
            current_run = 1
            
            for j in range(1, len(high_vol)):
                if high_vol.iloc[j] == high_vol.iloc[j-1]:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_run = 1
            runs.append(current_run)
            
            # Average run length as duration proxy
            avg_duration = np.mean(runs) if runs else 1.0
            result.append(avg_duration)
        
        return pd.Series(result, index=series.index)
    
    def _rolling_regime_switch_prob(self, series: pd.Series, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate rolling regime switching probability."""
        result = []
        
        for i in range(len(series)):
            if current_idx is not None and i > current_idx:
                result.append(np.nan)
                continue
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                result.append(0.1)
                continue
            
            data_window = series.iloc[start_idx:end_idx].dropna()
            
            if len(data_window) < 10:
                result.append(0.1)
                continue
            
            # Volatility-based switching probability
            vol = data_window.rolling(max(2, window//10), min_periods=2).std()
            vol_changes = vol.diff().abs()
            
            # Probability based on recent volatility changes
            vol_change_threshold = vol_changes.quantile(0.8)
            recent_change = vol_changes.iloc[-1] if len(vol_changes) > 0 else 0
            
            switch_prob = min(0.5, recent_change / (vol_change_threshold + 1e-8) * 0.2)
            result.append(switch_prob)
        
        return pd.Series(result, index=series.index)
    
    def _rolling_transition_timing(self, series: pd.Series, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate rolling transition timing indicator."""
        result = []
        
        for i in range(len(series)):
            if current_idx is not None and i > current_idx:
                result.append(np.nan)
                continue
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                result.append(0.0)
                continue
            
            data_window = series.iloc[start_idx:end_idx].dropna()
            
            if len(data_window) < 10:
                result.append(0.0)
                continue
            
            # Time since last significant change
            changes = data_window.diff().abs()
            threshold = changes.quantile(0.9)
            
            significant_changes = changes > threshold
            if significant_changes.any():
                last_change_idx = significant_changes[::-1].idxmax()
                time_since_change = len(data_window) - (data_window.index.get_loc(last_change_idx) + 1)
                timing_indicator = time_since_change / len(data_window)
            else:
                timing_indicator = 1.0
            
            result.append(timing_indicator)
        
        return pd.Series(result, index=series.index)
    
    def _rolling_correlation(self, series1: pd.Series, series2: pd.Series, 
                           window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate rolling correlation."""
        result = []
        
        for i in range(len(series1)):
            if current_idx is not None and i > current_idx:
                result.append(np.nan)
                continue
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                result.append(0.0)
                continue
            
            data1 = series1.iloc[start_idx:end_idx].dropna()
            data2 = series2.iloc[start_idx:end_idx].dropna()
            
            # Align series
            common_idx = data1.index.intersection(data2.index)
            if len(common_idx) < window // 4:
                result.append(0.0)
                continue
            
            try:
                corr = data1[common_idx].corr(data2[common_idx])
                result.append(corr if not np.isnan(corr) else 0.0)
            except:
                result.append(0.0)
        
        return pd.Series(result, index=series1.index)
    
    def _apply_leakage_safe_normalization(self, features: pd.DataFrame, current_idx: Optional[int]) -> pd.DataFrame:
        """Apply leakage-safe feature normalization."""
        normalized_features = features.copy()
        
        for col in features.columns:
            # Calculate rolling z-score with no lookahead
            rolling_mean = features[col].rolling(
                window=self.config.rolling_window_hours,
                min_periods=self.config.min_periods_hours
            ).mean()
            
            rolling_std = features[col].rolling(
                window=self.config.rolling_window_hours,
                min_periods=self.config.min_periods_hours
            ).std()
            
            # Apply z-score normalization
            normalized_features[col] = (features[col] - rolling_mean) / (rolling_std + 1e-8)
            
            # Clip extreme values
            normalized_features[col] = normalized_features[col].clip(-10, 10)
        
        return normalized_features.fillna(0.0)
    
    def _apply_production_filtering(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply production-ready feature filtering."""
        # Remove near-zero variance features
        if self.config.variance_threshold > 0:
            selector = VarianceThreshold(threshold=self.config.variance_threshold)
            selected_features = selector.fit_transform(features.fillna(0))
            selected_mask = selector.get_support()
            selected_columns = features.columns[selected_mask]
            
            features = pd.DataFrame(
                selected_features,
                index=features.index,
                columns=selected_columns
            )
        
        # Remove highly correlated features
        if self.config.correlation_threshold < 1.0:
            corr_matrix = features.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [
                column for column in upper_triangle.columns
                if any(upper_triangle[column] > self.config.correlation_threshold)
            ]
            
            features = features.drop(columns=to_drop)
        
        return features
    
    def get_feature_metadata(self) -> Dict[str, Any]:
        """Get metadata about generated features."""
        return {
            'config': self.config.__dict__,
            'primary_timeframe': self.config.primary_timeframe,
            'horizons': self.config.horizons,
            'advanced_features_enabled': {
                'structural_breaks': self.config.enable_structural_break_features,
                'duration_persistence': self.config.enable_duration_features,
                'regime_transitions': self.config.enable_regime_transition_features
            },
            'integration_status': {
                'existing_orchestrator': self.orchestrator is not None,
                'existing_feature_engineer': self.feature_engineer is not None
            }
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Generate synthetic 1h market data
    np.random.seed(42)
    
    # Create 1h timestamps for 3 months
    dates = pd.date_range('2023-01-01', '2023-04-01', freq='1H')
    n_obs = len(dates)
    
    # Create realistic 1h market data with regime structure
    prices = np.zeros(n_obs)
    prices[0] = 100.0
    
    for i in range(1, n_obs):
        # Regime-switching volatility (hourly appropriate)
        if i < n_obs // 3:
            vol = 0.008  # Low vol regime (hourly)
        elif i < 2 * n_obs // 3:
            vol = 0.020  # High vol regime
        else:
            vol = 0.012  # Medium vol regime
        
        ret = np.random.normal(0, vol)
        prices[i] = prices[i-1] * (1 + ret)
    
    # Create 1h OHLCV data
    test_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0005, n_obs)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_obs))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_obs))),
        'close': prices,
        'volume': np.random.lognormal(12, 0.3, n_obs)  # Adjusted for hourly
    }, index=dates)
    
    print("🧪 Testing Production Feature Integration")
    print(f"📊 Test data: {len(test_data)} observations (1h timeframe)")
    print(f"⏰ Timespan: {test_data.index[0]} to {test_data.index[-1]}")
    
    # Initialize production feature generator
    config = ProductionFeatureConfig(
        primary_timeframe="1h",
        horizons=[1, 2, 4],  # 1h, 2h, 4h windows
        enable_structural_break_features=True,
        enable_duration_features=True,
        enable_regime_transition_features=True,
        use_existing_orchestrator=False,  # Disable for testing
        use_existing_feature_engineer=False
    )
    
    feature_generator = ProductionLeakageSafeFeatures(config)
    
    async def test_feature_generation():
        # Generate features
        features = feature_generator.generate_production_features(
            data=test_data,
            symbol="ETHUSDT",
            current_time=test_data.index[-100]  # Simulate current time 100 hours ago
        )
        
        print(f"\n✅ Generated {len(features.columns)} production features")
        
        # Show feature breakdown by type
        feature_types = {}
        for col in features.columns:
            feature_type = col.split('_')[0] if '_' in col else 'other'
            if feature_type not in feature_types:
                feature_types[feature_type] = 0
            feature_types[feature_type] += 1
        
        print(f"\n📊 Feature breakdown by type:")
        for feature_type, count in sorted(feature_types.items()):
            print(f"  {feature_type}: {count} features")
        
        # Show sample advanced features
        print(f"\n🔬 Sample advanced Markov features:")
        advanced_features = [col for col in features.columns if any(
            keyword in col for keyword in ['variance_ratio', 'param_drift', 'vol_autocorr', 'transition_vol']
        )]
        
        for feature in advanced_features[:5]:
            print(f"  {feature}")
        
        # Show metadata
        metadata = feature_generator.get_feature_metadata()
        print(f"\n📋 Feature metadata:")
        print(f"  Primary timeframe: {metadata['primary_timeframe']}")
        print(f"  Horizons: {metadata['horizons']}")
        print(f"  Advanced features enabled: {metadata['advanced_features_enabled']}")
        
        return features
    
    # Run test
    features = asyncio.run(test_feature_generation())
    
    print(f"\n🎯 Production feature integration test completed!")
    print(f"   Ready for advanced Markov model integration")