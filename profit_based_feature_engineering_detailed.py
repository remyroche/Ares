#!/usr/bin/env python3
"""
Detailed Implementation of Profit-Based Feature Engineering.

This module shows how to create new features based on profit patterns and relationships
to enhance ML model training with the profit tracking data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from src.utils.logger import get_logger

@dataclass
class ProfitFeatureConfig:
    """Configuration for profit-based feature engineering."""
    # Basic profit features
    include_profit_magnitude: bool = True
    include_profit_direction: bool = True
    include_profit_squared: bool = True
    
    # Interaction features
    include_profit_interactions: bool = True
    interaction_features: List[str] = None  # Will be auto-detected
    
    # Categorical features
    include_profit_categories: bool = True
    profit_category_bins: List[float] = None  # Will use defaults
    
    # Risk-reward features
    include_risk_reward_features: bool = True
    volatility_features: List[str] = None  # Will be auto-detected
    
    # Advanced features
    include_profit_momentum: bool = True
    include_profit_volatility: bool = True
    include_profit_regime_features: bool = True
    
    # Rolling window features
    include_rolling_profit_features: bool = True
    rolling_windows: List[int] = None  # Will use defaults

class ProfitBasedFeatureEngineer:
    """Creates profit-based features for enhanced ML training."""
    
    def __init__(self, config: ProfitFeatureConfig):
        self.config = config
        self.logger = get_logger("ProfitBasedFeatureEngineer")
        
        # Set default values
        if self.config.interaction_features is None:
            self.config.interaction_features = [
                'rsi', 'macd', 'bollinger_upper', 'bollinger_lower', 
                'sma_20', 'ema_12', 'stoch_k', 'stoch_d', 'atr'
            ]
        
        if self.config.profit_category_bins is None:
            self.config.profit_category_bins = [-np.inf, -0.02, -0.01, 0, 0.01, 0.02, np.inf]
        
        if self.config.volatility_features is None:
            self.config.volatility_features = ['atr', 'volatility_20', 'bb_width']
        
        if self.config.rolling_windows is None:
            self.config.rolling_windows = [5, 10, 20, 50]
    
    def create_basic_profit_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic profit-based features.
        
        Args:
            data: DataFrame with 'potential_profit_pct' column
            
        Returns:
            DataFrame with basic profit features added
        """
        enhanced = data.copy()
        
        if self.config.include_profit_magnitude:
            enhanced['profit_abs'] = np.abs(data['potential_profit_pct'])
            enhanced['profit_log_abs'] = np.log(np.abs(data['potential_profit_pct']) + 1e-8)
        
        if self.config.include_profit_direction:
            enhanced['profit_sign'] = np.sign(data['potential_profit_pct'])
            enhanced['profit_positive'] = (data['potential_profit_pct'] > 0).astype(int)
            enhanced['profit_negative'] = (data['potential_profit_pct'] < 0).astype(int)
        
        if self.config.include_profit_squared:
            enhanced['profit_squared'] = data['potential_profit_pct'] ** 2
            enhanced['profit_cubed'] = data['potential_profit_pct'] ** 3
        
        return enhanced
    
    def create_profit_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between profit and technical indicators.
        
        Args:
            data: DataFrame with profit and technical indicator columns
            
        Returns:
            DataFrame with interaction features added
        """
        enhanced = data.copy()
        
        if not self.config.include_profit_interactions:
            return enhanced
        
        profit_col = 'potential_profit_pct'
        
        # Create interaction features with available technical indicators
        for feature in self.config.interaction_features:
            if feature in data.columns:
                # Linear interaction
                enhanced[f'{feature}_profit_interaction'] = data[feature] * data[profit_col]
                
                # Quadratic interaction
                enhanced[f'{feature}_profit_squared_interaction'] = data[feature] * (data[profit_col] ** 2)
                
                # Conditional interaction (only for positive profits)
                enhanced[f'{feature}_positive_profit_interaction'] = (
                    data[feature] * data[profit_col] * (data[profit_col] > 0)
                )
                
                # Conditional interaction (only for negative profits)
                enhanced[f'{feature}_negative_profit_interaction'] = (
                    data[feature] * data[profit_col] * (data[profit_col] < 0)
                )
        
        return enhanced
    
    def create_profit_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create categorical features based on profit ranges.
        
        Args:
            data: DataFrame with 'potential_profit_pct' column
            
        Returns:
            DataFrame with categorical profit features added
        """
        enhanced = data.copy()
        
        if not self.config.include_profit_categories:
            return enhanced
        
        # Create profit categories
        labels = ['high_loss', 'medium_loss', 'small_loss', 'small_profit', 'medium_profit', 'high_profit']
        
        enhanced['profit_category'] = pd.cut(
            data['potential_profit_pct'],
            bins=self.config.profit_category_bins,
            labels=labels
        )
        
        # One-hot encode profit categories
        profit_dummies = pd.get_dummies(enhanced['profit_category'], prefix='profit_cat')
        enhanced = pd.concat([enhanced, profit_dummies], axis=1)
        
        # Create binary features for extreme profits/losses
        enhanced['extreme_profit'] = (data['potential_profit_pct'] > 0.03).astype(int)
        enhanced['extreme_loss'] = (data['potential_profit_pct'] < -0.02).astype(int)
        
        return enhanced
    
    def create_risk_reward_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk-reward ratio features.
        
        Args:
            data: DataFrame with profit and volatility columns
            
        Returns:
            DataFrame with risk-reward features added
        """
        enhanced = data.copy()
        
        if not self.config.include_risk_reward_features:
            return enhanced
        
        profit_col = 'potential_profit_pct'
        
        # Basic risk-reward ratio
        enhanced['risk_reward_ratio'] = np.abs(data[profit_col]) / (1 + np.abs(data[profit_col]))
        
        # Volatility-adjusted risk-reward features
        for vol_feature in self.config.volatility_features:
            if vol_feature in data.columns:
                # Avoid division by zero
                volatility = data[vol_feature].replace(0, 1e-8)
                
                # Volatility-adjusted profit
                enhanced[f'vol_adj_profit_{vol_feature}'] = data[profit_col] / volatility
                
                # Risk-reward ratio with volatility
                enhanced[f'risk_reward_{vol_feature}'] = np.abs(data[profit_col]) / volatility
                
                # Sharpe-like ratio (profit per unit of volatility)
                enhanced[f'sharpe_like_{vol_feature}'] = data[profit_col] / volatility
        
        # Kelly criterion inspired features
        win_rate = (data[profit_col] > 0).rolling(window=50, min_periods=1).mean()
        avg_win = data[data[profit_col] > 0][profit_col].rolling(window=50, min_periods=1).mean()
        avg_loss = abs(data[data[profit_col] < 0][profit_col].rolling(window=50, min_periods=1).mean())
        
        enhanced['kelly_fraction'] = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        enhanced['kelly_fraction'] = enhanced['kelly_fraction'].fillna(0).clip(-1, 1)
        
        return enhanced
    
    def create_profit_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create profit momentum and trend features.
        
        Args:
            data: DataFrame with 'potential_profit_pct' column
            
        Returns:
            DataFrame with momentum features added
        """
        enhanced = data.copy()
        
        if not self.config.include_profit_momentum:
            return enhanced
        
        profit_col = 'potential_profit_pct'
        
        # Profit momentum (change in profit potential)
        enhanced['profit_momentum_1'] = data[profit_col].diff(1)
        enhanced['profit_momentum_3'] = data[profit_col].diff(3)
        enhanced['profit_momentum_5'] = data[profit_col].diff(5)
        
        # Profit acceleration (change in momentum)
        enhanced['profit_acceleration'] = enhanced['profit_momentum_1'].diff(1)
        
        # Profit trend (rolling mean)
        for window in [5, 10, 20]:
            enhanced[f'profit_trend_{window}'] = data[profit_col].rolling(window=window, min_periods=1).mean()
        
        # Profit momentum indicators
        enhanced['profit_rsi'] = self._calculate_rsi(data[profit_col], window=14)
        enhanced['profit_macd'] = self._calculate_macd(data[profit_col])
        
        return enhanced
    
    def create_profit_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create profit volatility features.
        
        Args:
            data: DataFrame with 'potential_profit_pct' column
            
        Returns:
            DataFrame with volatility features added
        """
        enhanced = data.copy()
        
        if not self.config.include_profit_volatility:
            return enhanced
        
        profit_col = 'potential_profit_pct'
        
        # Rolling volatility of profit potential
        for window in [5, 10, 20, 50]:
            enhanced[f'profit_volatility_{window}'] = (
                data[profit_col].rolling(window=window, min_periods=1).std()
            )
        
        # Profit volatility ratio (current vs historical)
        enhanced['profit_vol_ratio_10'] = (
            enhanced['profit_volatility_5'] / enhanced['profit_volatility_20']
        )
        
        # Profit volatility percentile
        enhanced['profit_vol_percentile'] = (
            data[profit_col].rolling(window=50, min_periods=1)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        )
        
        return enhanced
    
    def create_profit_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create profit regime and market condition features.
        
        Args:
            data: DataFrame with profit and market data
            
        Returns:
            DataFrame with regime features added
        """
        enhanced = data.copy()
        
        if not self.config.include_profit_regime_features:
            return enhanced
        
        profit_col = 'potential_profit_pct'
        
        # Profit regime detection
        profit_ma_20 = data[profit_col].rolling(window=20, min_periods=1).mean()
        profit_ma_50 = data[profit_col].rolling(window=50, min_periods=1).mean()
        
        # Regime indicators
        enhanced['profit_regime_bullish'] = (profit_ma_20 > profit_ma_50).astype(int)
        enhanced['profit_regime_bearish'] = (profit_ma_20 < profit_ma_50).astype(int)
        
        # Regime strength
        enhanced['profit_regime_strength'] = abs(profit_ma_20 - profit_ma_50)
        
        # Profit consistency (how often profits are positive)
        enhanced['profit_consistency_20'] = (
            (data[profit_col] > 0).rolling(window=20, min_periods=1).mean()
        )
        
        # Profit stability (inverse of volatility)
        enhanced['profit_stability_20'] = 1 / (enhanced['profit_volatility_20'] + 1e-8)
        
        return enhanced
    
    def create_rolling_profit_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window profit features.
        
        Args:
            data: DataFrame with 'potential_profit_pct' column
            
        Returns:
            DataFrame with rolling features added
        """
        enhanced = data.copy()
        
        if not self.config.include_rolling_profit_features:
            return enhanced
        
        profit_col = 'potential_profit_pct'
        
        for window in self.config.rolling_windows:
            # Rolling statistics
            enhanced[f'profit_mean_{window}'] = data[profit_col].rolling(window=window, min_periods=1).mean()
            enhanced[f'profit_std_{window}'] = data[profit_col].rolling(window=window, min_periods=1).std()
            enhanced[f'profit_min_{window}'] = data[profit_col].rolling(window=window, min_periods=1).min()
            enhanced[f'profit_max_{window}'] = data[profit_col].rolling(window=window, min_periods=1).max()
            
            # Rolling percentiles
            enhanced[f'profit_median_{window}'] = data[profit_col].rolling(window=window, min_periods=1).median()
            enhanced[f'profit_q75_{window}'] = data[profit_col].rolling(window=window, min_periods=1).quantile(0.75)
            enhanced[f'profit_q25_{window}'] = data[profit_col].rolling(window=window, min_periods=1).quantile(0.25)
            
            # Rolling ratios
            enhanced[f'profit_range_{window}'] = enhanced[f'profit_max_{window}'] - enhanced[f'profit_min_{window}']
            enhanced[f'profit_cv_{window}'] = enhanced[f'profit_std_{window}'] / (enhanced[f'profit_mean_{window}'] + 1e-8)
        
        return enhanced
    
    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI for a series."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD for a series."""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def create_all_profit_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all profit-based features.
        
        Args:
            data: DataFrame with 'potential_profit_pct' column
            
        Returns:
            DataFrame with all profit features added
        """
        self.logger.info("🔧 Creating profit-based features...")
        
        # Check if profit tracking data exists
        if 'potential_profit_pct' not in data.columns:
            self.logger.error("❌ 'potential_profit_pct' column not found. Run triple barrier with include_profit_tracking=True")
            return data
        
        # Create features in sequence
        enhanced = data.copy()
        
        enhanced = self.create_basic_profit_features(enhanced)
        enhanced = self.create_profit_interaction_features(enhanced)
        enhanced = self.create_profit_categorical_features(enhanced)
        enhanced = self.create_risk_reward_features(enhanced)
        enhanced = self.create_profit_momentum_features(enhanced)
        enhanced = self.create_profit_volatility_features(enhanced)
        enhanced = self.create_profit_regime_features(enhanced)
        enhanced = self.create_rolling_profit_features(enhanced)
        
        # Count new features
        original_cols = set(data.columns)
        new_cols = set(enhanced.columns) - original_cols
        
        self.logger.info(f"✅ Created {len(new_cols)} profit-based features")
        self.logger.info(f"   Original features: {len(original_cols)}")
        self.logger.info(f"   Total features: {len(enhanced.columns)}")
        
        return enhanced
    
    def get_profit_feature_summary(self, enhanced_data: pd.DataFrame) -> Dict:
        """
        Generate summary of profit-based features.
        
        Args:
            enhanced_data: DataFrame with profit features
            
        Returns:
            Dictionary with feature summary
        """
        # Identify profit-based features
        profit_features = [col for col in enhanced_data.columns if any(keyword in col.lower() for keyword in 
                        ['profit', 'risk_reward', 'kelly', 'regime', 'momentum', 'volatility'])]
        
        # Calculate feature statistics
        feature_stats = {}
        for feature in profit_features:
            if feature in enhanced_data.columns:
                feature_stats[feature] = {
                    'mean': enhanced_data[feature].mean(),
                    'std': enhanced_data[feature].std(),
                    'min': enhanced_data[feature].min(),
                    'max': enhanced_data[feature].max(),
                    'null_count': enhanced_data[feature].isnull().sum()
                }
        
        return {
            'total_profit_features': len(profit_features),
            'profit_feature_names': profit_features,
            'feature_statistics': feature_stats
        }

# Example usage
def demonstrate_profit_feature_engineering():
    """Demonstrate profit-based feature engineering."""
    
    print("🔧 Profit-Based Feature Engineering Demonstration")
    print("=" * 60)
    
    # Configuration
    config = ProfitFeatureConfig(
        include_profit_magnitude=True,
        include_profit_direction=True,
        include_profit_interactions=True,
        include_profit_categories=True,
        include_risk_reward_features=True,
        include_profit_momentum=True,
        include_profit_volatility=True,
        include_profit_regime_features=True,
        include_rolling_profit_features=True
    )
    
    # Create feature engineer
    engineer = ProfitBasedFeatureEngineer(config)
    
    print("✅ Profit-based feature engineering ready for use")
    print("\n📋 Feature Categories:")
    print("1. Basic profit features (magnitude, direction, squared)")
    print("2. Interaction features (profit × technical indicators)")
    print("3. Categorical features (profit ranges)")
    print("4. Risk-reward features (volatility-adjusted)")
    print("5. Momentum features (profit trends)")
    print("6. Volatility features (profit stability)")
    print("7. Regime features (market conditions)")
    print("8. Rolling features (statistical summaries)")
    
    return engineer

if __name__ == "__main__":
    demonstrate_profit_feature_engineering()