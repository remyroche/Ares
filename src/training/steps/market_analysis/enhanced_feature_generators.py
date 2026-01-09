"""
Enhanced Feature Generators for MI Improvement

This module provides enhanced feature generation capabilities to improve
Mutual Information (MI) scores through non-linear transformations,
market regime indicators, and target-specific features.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class RankBasedFeatureGenerator:
    """Generate rank-based features to handle magnitude safely without explosive values."""
    
    @staticmethod
    def add_rank_features(df: pd.DataFrame, columns: List[str], windows: List[int] = [100]) -> pd.DataFrame:
        """
        Add rolling rank-based features (percentiles) for specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to transform
            windows: Rolling windows for rank calculation
            
        Returns:
            DataFrame with rank features
        """
        features = pd.DataFrame(index=df.index)
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    # Rolling percentile rank (0 to 1)
                    features[f'{col}_rank_{window}'] = df[col].rolling(window).rank(pct=True)

                    # Also add a "distance from median" rank feature
                    rolling_median = df[col].rolling(window).median()
                    features[f'{col}_dist_median_{window}'] = (df[col] - rolling_median) / (df[col].rolling(window).std() + 1e-8)
        
        logger.info(f"Added {len(features.columns)} rank-based features")
        return features

    @staticmethod
    def add_interaction_features(df: pd.DataFrame, column_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Add interaction features between column pairs using safe operations.
        """
        features = pd.DataFrame(index=df.index)
        
        for col1, col2 in column_pairs:
            if col1 in df.columns and col2 in df.columns:
                # Rank-based interaction (more stable)
                rank1 = df[col1].rolling(100).rank(pct=True)
                rank2 = df[col2].rolling(100).rank(pct=True)
                
                features[f'{col1}_x_{col2}_rank'] = rank1 * rank2
                features[f'{col1}_div_{col2}_rank'] = rank1 / (rank2 + 0.01)
                
        logger.info(f"Added {len(features.columns)} interaction features")
        return features


class MarketRegimeFeatureGenerator:
    """Generate market regime indicators for context awareness."""
    
    @staticmethod
    def add_volatility_regime_features(df: pd.DataFrame, price_col: str = 'close', 
                                     windows: List[int] = [20, 50]) -> pd.DataFrame:
        """
        Add volatility regime features.
        
        Args:
            df: Input DataFrame
            price_col: Price column name
            windows: List of rolling windows
            
        Returns:
            DataFrame with volatility regime features
        """
        features = pd.DataFrame(index=df.index)
        
        if price_col in df.columns:
            returns = df[price_col].pct_change()
            
            for window in windows:
                rolling_vol = returns.rolling(window).std()
                rolling_vol_mean = rolling_vol.rolling(window * 2).mean()
                
                # Volatility regime flags
                features[f'vol_regime_high_{window}'] = (rolling_vol > rolling_vol_mean * 1.5).astype(int)
                features[f'vol_regime_low_{window}'] = (rolling_vol < rolling_vol_mean * 0.5).astype(int)
                features[f'vol_regime_normal_{window}'] = (
                    (rolling_vol >= rolling_vol_mean * 0.5) & 
                    (rolling_vol <= rolling_vol_mean * 1.5)
                ).astype(int)
                
                # Volatility quantiles
                features[f'vol_quantile_{window}'] = rolling_vol.rolling(window * 4).rank(pct=True)
                
                # Volatility momentum
                features[f'vol_momentum_{window}'] = rolling_vol - rolling_vol.rolling(10).mean()
        
        logger.info(f"Added {len(features.columns)} volatility regime features")
        return features
    
    @staticmethod
    def add_trend_regime_features(df: pd.DataFrame, price_col: str = 'close',
                                short_windows: List[int] = [10, 20],
                                long_windows: List[int] = [50, 100]) -> pd.DataFrame:
        """
        Add trend regime features.
        
        Args:
            df: Input DataFrame
            price_col: Price column name
            short_windows: Short-term moving average windows
            long_windows: Long-term moving average windows
            
        Returns:
            DataFrame with trend regime features
        """
        features = pd.DataFrame(index=df.index)
        
        if price_col in df.columns:
            price = df[price_col]
            
            for short in short_windows:
                for long in long_windows:
                    if short < long:
                        sma_short = price.rolling(short).mean()
                        sma_long = price.rolling(long).mean()
                        
                        # Trend regime flags
                        features[f'trend_regime_uptrend_{short}_{long}'] = (sma_short > sma_long).astype(int)
                        features[f'trend_regime_downtrend_{short}_{long}'] = (sma_short < sma_long).astype(int)
                        
                        # Trend strength
                        features[f'trend_strength_{short}_{long}'] = (sma_short - sma_long) / sma_long
                        
                        # Trend momentum
                        features[f'trend_momentum_{short}_{long}'] = features[f'trend_strength_{short}_{long}'].diff()
        
        logger.info(f"Added {len(features.columns)} trend regime features")
        return features
    
    @staticmethod
    def add_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features for intraday patterns.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with time-based features
        """
        features = pd.DataFrame(index=df.index)
        
        if isinstance(df.index, pd.DatetimeIndex):
            # Hour of day
            features['hour_of_day'] = df.index.hour
            features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            
            # Day of week
            features['day_of_week'] = df.index.dayofweek
            features['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            features['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            
            # Session indicators
            features['is_london_session'] = ((df.index.hour >= 8) & (df.index.hour <= 16)).astype(int)
            features['is_ny_session'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(int)
            features['is_asia_session'] = ((df.index.hour >= 0) & (df.index.hour <= 8)).astype(int)
            
            # Session overlap
            features['is_london_ny_overlap'] = ((df.index.hour >= 13) & (df.index.hour <= 16)).astype(int)
            
            # Weekend/holiday indicators
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            # Month indicators
            features['month_of_year'] = df.index.month
            features['is_month_end'] = (df.index.day >= 25).astype(int)
            features['is_month_start'] = (df.index.day <= 5).astype(int)
        
        logger.info(f"Added {len(features.columns)} time-based features")
        return features


class TargetSpecificFeatureGenerator:
    """Generate target-specific features for different specialist types."""
    
    @staticmethod
    def add_breakout_features(df: pd.DataFrame, price_cols: List[str] = ['high', 'low', 'close'],
                            volume_col: str = 'volume', windows: List[int] = [20, 50]) -> pd.DataFrame:
        """
        Add features specifically for breakout prediction.
        
        Args:
            df: Input DataFrame
            price_cols: Price column names
            volume_col: Volume column name
            windows: Rolling windows
            
        Returns:
            DataFrame with breakout features
        """
        features = pd.DataFrame(index=df.index)
        
        if 'high' in df.columns and 'low' in df.columns:
            high_low_range = df['high'] - df['low']
            
            for window in windows:
                # Range expansion/contraction
                range_mean = high_low_range.rolling(window).mean()
                features[f'range_expansion_{window}'] = high_low_range / range_mean
                features[f'range_contraction_{window}'] = (high_low_range < range_mean * 0.7).astype(int)
                
                # Range breakouts
                range_high = high_low_range.rolling(window).max()
                range_low = high_low_range.rolling(window).min()
                features[f'range_breakout_up_{window}'] = (high_low_range > range_high.shift(1)).astype(int)
                features[f'range_breakout_down_{window}'] = (high_low_range < range_low.shift(1)).astype(int)
        
        if volume_col in df.columns:
            volume = df[volume_col]
            
            for window in windows:
                volume_mean = volume.rolling(window).mean()
                
                # Volume patterns
                features[f'volume_surge_{window}'] = (volume > volume_mean * 1.5).astype(int)
                features[f'volume_dry_up_{window}'] = (volume < volume_mean * 0.5).astype(int)
                features[f'volume_ratio_{window}'] = volume / volume_mean
                
                # Volume-price relationship
                if 'close' in df.columns:
                    price_change = df['close'].pct_change()
                    features[f'volume_price_correlation_{window}'] = (
                        volume.rolling(window).corr(price_change)
                    )
        
        logger.info(f"Added {len(features.columns)} breakout features")
        return features
    
    @staticmethod
    def add_volume_force_features(df: pd.DataFrame, price_col: str = 'close',
                                volume_col: str = 'volume', windows: List[int] = [10, 20]) -> pd.DataFrame:
        """
        Add enhanced volume force features.
        
        Args:
            df: Input DataFrame
            price_col: Price column name
            volume_col: Volume column name
            windows: Rolling windows
            
        Returns:
            DataFrame with enhanced volume force features
        """
        features = pd.DataFrame(index=df.index)
        
        if price_col in df.columns and volume_col in df.columns:
            price = df[price_col]
            volume = df[volume_col]
            
            # Price change
            price_change = price.pct_change()
            
            for window in windows:
                # Volume-weighted price change
                features[f'volume_weighted_change_{window}'] = (price_change * volume).rolling(window).sum()
                
                # Volume momentum
                volume_change = volume.pct_change()
                features[f'volume_momentum_{window}'] = volume_change.rolling(window).sum()
                
                # Volume acceleration
                features[f'volume_acceleration_{window}'] = volume_change.rolling(window).sum() - volume_change.rolling(window*2).sum()
                
                # Force index variations
                force_index = price_change * volume
                features[f'force_index_{window}'] = force_index.rolling(window).sum()
                features[f'force_index_ma_{window}'] = force_index.rolling(window).mean()
                
                # Volume efficiency
                features[f'volume_efficiency_{window}'] = price_change.rolling(window).sum() / (volume.rolling(window).sum() + 1e-8)
                
                # Money flow indicators
                if 'high' in df.columns and 'low' in df.columns:
                    typical_price = (df['high'] + df['low'] + price) / 3
                    money_flow = typical_price * volume
                    features[f'money_flow_{window}'] = money_flow.rolling(window).sum()
                    features[f'money_flow_momentum_{window}'] = money_flow.rolling(window).sum() - money_flow.rolling(window*2).sum()
        
        logger.info(f"Added {len(features.columns)} enhanced volume force features")
        return features
    
    @staticmethod
    def add_momentum_features(df: pd.DataFrame, price_col: str = 'close',
                            windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Add enhanced momentum features.
        
        Args:
            df: Input DataFrame
            price_col: Price column name
            windows: Momentum windows
            
        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=df.index)
        
        if price_col in df.columns:
            price = df[price_col]
            returns = price.pct_change()
            
            for window in windows:
                # Momentum strength
                features[f'momentum_strength_{window}'] = returns.rolling(window).sum()
                
                # Momentum acceleration
                features[f'momentum_acceleration_{window}'] = (
                    returns.rolling(window).sum() - returns.rolling(window*2).sum()
                )
                
                # Momentum volatility
                features[f'momentum_volatility_{window}'] = returns.rolling(window).std()
                
                # Momentum persistence
                features[f'momentum_persistence_{window}'] = (
                    (returns.rolling(window).sum() > 0) == 
                    (returns.rolling(window//2).sum() > 0)
                ).astype(int)
                
                # RSI-like momentum
                gains = returns.clip(lower=0)
                losses = -returns.clip(upper=0)
                avg_gains = gains.rolling(window).mean()
                avg_losses = losses.rolling(window).mean()
                rs = avg_gains / (avg_losses + 1e-8)
                features[f'momentum_rsi_{window}'] = 100 - (100 / (1 + rs))
        
        logger.info(f"Added {len(features.columns)} momentum features")
        return features


class EnhancedFeaturePipeline:
    """Main pipeline for enhanced feature generation."""
    
    def __init__(self):
        self.rank_gen = RankBasedFeatureGenerator()
        self.regime_gen = MarketRegimeFeatureGenerator()
        self.target_gen = TargetSpecificFeatureGenerator()
    
    def generate_enhanced_features(self, df: pd.DataFrame, specialist_type: str,
                                 config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Generate enhanced features for a specific specialist type.
        
        Args:
            df: Input market data
            specialist_type: Type of specialist ('volume_force', 'breakout', 'momentum', etc.)
            config: Configuration dictionary
            
        Returns:
            DataFrame with enhanced features
        """
        all_features = []
        
        # 1. Rank-based transformations (Replacing Polynomials)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            rank_features = self.rank_gen.add_rank_features(df, numeric_cols[:5], windows=[50, 200])
            all_features.append(rank_features)
        
        # 2. Market regime features
        regime_features = self.regime_gen.add_volatility_regime_features(df)
        all_features.append(regime_features)
        
        regime_features_trend = self.regime_gen.add_trend_regime_features(df)
        all_features.append(regime_features_trend)
        
        if isinstance(df.index, pd.DatetimeIndex):
            time_features = self.regime_gen.add_time_based_features(df)
            all_features.append(time_features)
        
        # 3. Target-specific features
        if specialist_type == 'volume_force':
            target_features = self.target_gen.add_volume_force_features(df)
            all_features.append(target_features)
        elif specialist_type == 'breakout':
            target_features = self.target_gen.add_breakout_features(df)
            all_features.append(target_features)
        elif specialist_type == 'momentum':
            target_features = self.target_gen.add_momentum_features(df)
            all_features.append(target_features)
        
        # Combine all features
        if all_features:
            enhanced_df = pd.concat(all_features, axis=1)
            # Remove any infinite or NaN values
            enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            logger.info(f"Generated {len(enhanced_df.columns)} enhanced features for {specialist_type}")
            return enhanced_df
        else:
            return pd.DataFrame(index=df.index)

class MIOptimizedFeatureGenerator:
    """Generate features specifically optimized for MI improvement."""
    
    @staticmethod
    def add_target_aligned_features(df: pd.DataFrame, target_col: str = 'target_long', time_scales: List[int] = [1, 4, 16]) -> pd.DataFrame:
        """
        Add features specifically aligned with target prediction, across multiple timeframes.
        
        Args:
            df: Input DataFrame with OHLCV data
            target_col: Target column name
            time_scales: List of time scales (multipliers) to generate features for
            
        Returns:
            DataFrame with target-aligned features
        """
        features = pd.DataFrame(index=df.index)
        
        if 'close' in df.columns and 'high' in df.columns and 'low' in df.columns:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            for scale in time_scales:
                # Adjust windows based on scale
                w5 = 5 * scale
                w10 = 10 * scale
                w20 = 20 * scale

                suffix = f"_x{scale}" if scale > 1 else ""

                # Price momentum features
                features[f'momentum_{w5}{suffix}'] = returns.rolling(w5).mean()
                features[f'momentum_{w10}{suffix}'] = returns.rolling(w10).mean()
                features[f'momentum_{w20}{suffix}'] = returns.rolling(w20).mean()

                # Volatility features
                features[f'volatility_{w5}{suffix}'] = returns.rolling(w5).std()
                features[f'volatility_{w10}{suffix}'] = returns.rolling(w10).std()
                features[f'volatility_ratio_{suffix}'] = features[f'volatility_{w5}{suffix}'] / (features[f'volatility_{w10}{suffix}'] + 1e-8)

                # Volume features (rolling averages)
                features[f'volume_ma_ratio_{suffix}'] = volume / (volume.rolling(w20).mean() + 1e-8)
                features[f'volume_price_trend_{suffix}'] = (volume * returns).rolling(w5).sum()

                # Trend features
                features[f'trend_strength_{suffix}'] = abs(returns.rolling(w20).mean())
                features[f'trend_consistency_{suffix}'] = (returns > 0).rolling(w10).mean()

                # Mean reversion features
                features[f'mean_reversion_signal_{suffix}'] = (close - close.rolling(w20).mean()) / (close.rolling(w20).std() + 1e-8)
            
            # Price position features (instantaneous, no scaling needed unless smoothing)
            features['high_low_ratio'] = high / (low + 1e-8)
            features['close_position'] = (close - low) / (high - low + 1e-8)
            
        return features
    
    @staticmethod
    def add_regime_features(df: pd.DataFrame, time_scales: List[int] = [1, 4, 16]) -> pd.DataFrame:
        """
        Add market regime detection features across multiple timeframes.
        """
        features = pd.DataFrame(index=df.index)
        
        if 'close' in df.columns and 'volume' in df.columns:
            close = df['close']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            for scale in time_scales:
                w20 = 20 * scale
                w100 = 100 * scale
                suffix = f"_x{scale}" if scale > 1 else ""

                # Trend regime features
                features[f'trend_regime_{suffix}'] = (returns.rolling(w20).mean() > 0).astype(int)
                features[f'trend_strength_{suffix}'] = abs(returns.rolling(w20).mean())

                # Volatility regime features
                volatility = returns.rolling(w20).std()
                vol_baseline = returns.rolling(w100).std() # Approximation for baseline
                features[f'vol_regime_{suffix}'] = (volatility > vol_baseline).astype(int)
                features[f'vol_regime_strength_{suffix}'] = volatility / (vol_baseline + 1e-8)

                # Volume regime features
                volume_ma = volume.rolling(w20).mean()
                features[f'volume_regime_{suffix}'] = (volume > volume_ma).astype(int)
                features[f'volume_regime_strength_{suffix}'] = volume / (volume_ma + 1e-8)

                # Combined regime features
                features[f'regime_consistency_{suffix}'] = (
                    features[f'trend_regime_{suffix}'] +
                    features[f'vol_regime_{suffix}'] +
                    features[f'volume_regime_{suffix}']
                ) / 3
            
        return features
    
    @staticmethod
    def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features.
        """
        features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Price microstructure features
            features['price_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
            features['intraday_range'] = (df['high'] - df['low']) / df['close']
            features['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
            
            # Wick features
            features['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-8)
            features['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-8)
            
            # Price action features
            features['price_action'] = np.sign(df['close'] - df['open'])
            features['gap_ratio'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            
        return features

class MIOptimizedFeaturePipeline:
    """Main pipeline for MI-optimized feature generation."""
    
    def __init__(self):
        self.rank_gen = RankBasedFeatureGenerator()
        self.regime_gen = MarketRegimeFeatureGenerator()
        self.target_gen = TargetSpecificFeatureGenerator()
        self.mi_gen = MIOptimizedFeatureGenerator()
        self.time_scales = [1, 4, 16]
    
    def generate_enhanced_features(self, df: pd.DataFrame, specialist_type: str,
                                 config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Generate MI-optimized enhanced features for a specific specialist type.
        
        Args:
            df: Input market data
            specialist_type: Type of specialist ('volume_force', 'breakout', 'momentum', etc.)
            config: Configuration dictionary
            
        Returns:
            DataFrame with MI-optimized enhanced features
        """
        all_features = []
        
        # 1. Target-aligned features (MI-focused) on multiple timeframes
        target_features = self.mi_gen.add_target_aligned_features(df, time_scales=self.time_scales)
        all_features.append(target_features)
        
        # 2. Regime features (MI-focused) on multiple timeframes
        regime_features = self.mi_gen.add_regime_features(df, time_scales=self.time_scales)
        all_features.append(regime_features)
        
        # 3. Ensemble diversity features
        diversity_features = EnsembleDiversityFeatureGenerator().add_diversity_features(df)
        all_features.append(diversity_features)

        # 4. Microstructure features (MI-focused)
        microstructure_features = self.mi_gen.add_microstructure_features(df)
        all_features.append(microstructure_features)
        
        # 5. Rank-based transformations (Replacing Polynomials)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # Use RankBasedFeatureGenerator instead of polynomials
            rank_features = self.rank_gen.add_rank_features(df, numeric_cols[:5], windows=[50, 200])
            all_features.append(rank_features)
        
        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            # Remove duplicates and clean
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            combined_features = combined_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return combined_features
        
        return pd.DataFrame(index=df.index)

class EnsembleDiversityFeatureGenerator:
    """Generate ensemble diversity features to reduce correlation."""
    
    @staticmethod
    def add_diversity_features(df: pd.DataFrame, existing_features: pd.DataFrame = None) -> pd.DataFrame:
        """Add ensemble diversity features."""
        features = pd.DataFrame(index=df.index)
        
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            
            # 1. Cross-timeframe momentum diversity
            for window in [5, 15, 25, 35]:
                momentum = returns.rolling(window).sum()
                features[f'momentum_diversity_{window}'] = momentum / momentum.rolling(60).std()
            
            # 2. Volatility regime diversity
            vol_short = returns.rolling(15).std()
            vol_long = returns.rolling(60).std()
            features['volatility_regime_diversity'] = vol_short / vol_long
            
            # 3. Trend strength diversity
            trend_short = returns.rolling(15).mean()
            trend_long = returns.rolling(35).mean()
            features['trend_strength_diversity'] = (trend_short - trend_long) / vol_long
            
            # 4. Mean reversion diversity
            mean_reversion = -returns.rolling(25).mean() / returns.rolling(25).std()
            features['mean_reversion_diversity'] = mean_reversion
            
            # 5. Volume-price diversity
            if 'volume' in df.columns:
                volume_change = df['volume'].pct_change()
                volume_price_corr = returns.rolling(25).corr(volume_change)
                features['volume_price_diversity'] = volume_price_corr
        
        return features
