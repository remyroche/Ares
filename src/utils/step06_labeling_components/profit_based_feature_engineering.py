"""
Profit-Based Feature Engineering - Moved to Utilities

This module contains the original step06 profit-based feature engineering functionality
now available as utilities. All functionality has been preserved from the original step06.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import warnings

# Import validation and safety utilities
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, 
    validate_positive, validate_range, MathValidationError
)

logger = logging.getLogger(__name__)

class ProfitBasedFeatureEngineering:
    """
    Profit-based feature engineering for enhanced trading signal generation.
    This is the original step06 functionality now available as utilities.
    """
    
    def __init__(self, profit_threshold: float = 0.002,
                 risk_reward_ratio: float = 2.0,
                 min_profit_margin: float = 0.001,
                 max_profit_margin: float = 0.01):
        """
        Initialize profit-based feature engineering.
        
        Args:
            profit_threshold: Minimum profit threshold for signal generation
            risk_reward_ratio: Risk-reward ratio for position sizing
            min_profit_margin: Minimum profit margin
            max_profit_margin: Maximum profit margin
        """
        self.profit_threshold = profit_threshold
        self.risk_reward_ratio = risk_reward_ratio
        self.min_profit_margin = min_profit_margin
        self.max_profit_margin = max_profit_margin
        self.logger = logger
        
        # Validate parameters
        self._validate_parameters()
        
        self.logger.info("💰 Profit-Based Feature Engineering (Step06 Utilities) initialized")
        self.logger.info(f"   Profit threshold: {profit_threshold:.4f}")
        self.logger.info(f"   Risk-reward ratio: {risk_reward_ratio}")
        self.logger.info(f"   Profit margin range: {min_profit_margin:.4f} - {max_profit_margin:.4f}")

    def _validate_parameters(self) -> None:
        """Validate profit-based feature engineering parameters."""
        if not validate_positive(self.profit_threshold):
            raise MathValidationError("Profit threshold must be positive")
        if not validate_positive(self.risk_reward_ratio):
            raise MathValidationError("Risk-reward ratio must be positive")
        if not validate_positive(self.min_profit_margin):
            raise MathValidationError("Minimum profit margin must be positive")
        if not validate_positive(self.max_profit_margin):
            raise MathValidationError("Maximum profit margin must be positive")
        if self.min_profit_margin >= self.max_profit_margin:
            raise MathValidationError("Minimum profit margin must be less than maximum")

    def create_profit_based_features(self, market_data: pd.DataFrame,
                                   returns: pd.Series,
                                   labels: pd.Series) -> pd.DataFrame:
        """
        Create profit-based features for enhanced trading signals.
        
        Args:
            market_data: OHLCV market data
            returns: Price returns
            labels: Trading labels
            
        Returns:
            DataFrame with profit-based features
        """
        self.logger.info("💰 Creating profit-based features...")
        
        try:
            # Initialize result DataFrame
            result = pd.DataFrame(index=market_data.index)
            
            # 1. Profit potential features
            profit_features = self._create_profit_potential_features(market_data, returns)
            result = pd.concat([result, profit_features], axis=1)
            
            # 2. Risk-adjusted features
            risk_features = self._create_risk_adjusted_features(market_data, returns, labels)
            result = pd.concat([result, risk_features], axis=1)
            
            # 3. Position sizing features
            position_features = self._create_position_sizing_features(market_data, returns, labels)
            result = pd.concat([result, position_features], axis=1)
            
            # 4. Profit margin features
            margin_features = self._create_profit_margin_features(market_data, returns, labels)
            result = pd.concat([result, margin_features], axis=1)
            
            # 5. Transaction cost features
            cost_features = self._create_transaction_cost_features(market_data, returns, labels)
            result = pd.concat([result, cost_features], axis=1)
            
            self.logger.info(f"✅ Profit-based features created: {len(result.columns)} features")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Profit-based feature creation failed: {e}")
            raise

    def _create_profit_potential_features(self, market_data: pd.DataFrame,
                                        returns: pd.Series) -> pd.DataFrame:
        """Create profit potential features."""
        features = pd.DataFrame(index=market_data.index)
        
        # Calculate potential profit based on price movements
        high_low_range = (market_data['high'] - market_data['low']) / market_data['close']
        open_close_range = abs(market_data['close'] - market_data['open']) / market_data['close']
        
        # Profit potential indicators
        features['profit_potential_high_low'] = high_low_range
        features['profit_potential_open_close'] = open_close_range
        features['profit_potential_ratio'] = safe_divide(open_close_range, high_low_range, default=0.5)
        
        # Rolling profit potential
        for window in [5, 10, 20]:
            features[f'profit_potential_ma_{window}'] = high_low_range.rolling(window).mean()
            features[f'profit_potential_std_{window}'] = high_low_range.rolling(window).std()
        
        # Profit threshold indicators
        features['above_profit_threshold'] = (high_low_range > self.profit_threshold).astype(float)
        features['profit_threshold_ratio'] = safe_divide(high_low_range, self.profit_threshold, default=0.0)
        
        return features

    def _create_risk_adjusted_features(self, market_data: pd.DataFrame,
                                     returns: pd.Series,
                                     labels: pd.Series) -> pd.DataFrame:
        """Create risk-adjusted features."""
        features = pd.DataFrame(index=market_data.index)
        
        # Calculate volatility as risk measure
        volatility = returns.rolling(20).std()
        
        # Risk-adjusted returns
        features['risk_adjusted_returns'] = safe_divide(returns, volatility, default=0.0)
        features['volatility'] = volatility
        
        # Risk-reward ratio features
        for window in [5, 10, 20]:
            rolling_returns = returns.rolling(window)
            rolling_vol = volatility.rolling(window)
            
            features[f'risk_reward_ratio_{window}'] = safe_divide(
                rolling_returns.mean(), rolling_vol.mean(), default=0.0
            )
        
        # Risk-adjusted profit potential
        high_low_range = (market_data['high'] - market_data['low']) / market_data['close']
        features['risk_adjusted_profit_potential'] = safe_divide(
            high_low_range, volatility, default=0.0
        )
        
        # Risk threshold indicators
        features['above_risk_threshold'] = (volatility > volatility.quantile(0.7)).astype(float)
        features['below_risk_threshold'] = (volatility < volatility.quantile(0.3)).astype(float)
        
        return features

    def _create_position_sizing_features(self, market_data: pd.DataFrame,
                                       returns: pd.Series,
                                       labels: pd.Series) -> pd.DataFrame:
        """Create position sizing features."""
        features = pd.DataFrame(index=market_data.index)
        
        # Calculate position sizing based on risk-reward ratio
        volatility = returns.rolling(20).std()
        high_low_range = (market_data['high'] - market_data['low']) / market_data['close']
        
        # Kelly criterion-based position sizing
        win_rate = (labels == 1.0).rolling(50).mean()
        avg_win = returns[labels == 1.0].rolling(50).mean()
        avg_loss = returns[labels == -1.0].rolling(50).mean()
        
        kelly_fraction = safe_divide(
            win_rate * avg_win - (1 - win_rate) * abs(avg_loss),
            avg_win, default=0.0
        )
        features['kelly_fraction'] = np.clip(kelly_fraction, 0.0, 0.25)  # Cap at 25%
        
        # Risk-based position sizing
        risk_based_size = safe_divide(0.02, volatility, default=0.0)  # 2% risk per trade
        features['risk_based_position_size'] = np.clip(risk_based_size, 0.0, 0.1)  # Cap at 10%
        
        # Profit-based position sizing
        profit_based_size = safe_divide(high_low_range, self.profit_threshold, default=0.0)
        features['profit_based_position_size'] = np.clip(profit_based_size, 0.0, 0.2)  # Cap at 20%
        
        # Combined position sizing
        features['combined_position_size'] = (
            features['kelly_fraction'] * 0.4 +
            features['risk_based_position_size'] * 0.3 +
            features['profit_based_position_size'] * 0.3
        )
        
        return features

    def _create_profit_margin_features(self, market_data: pd.DataFrame,
                                     returns: pd.Series,
                                     labels: pd.Series) -> pd.DataFrame:
        """Create profit margin features."""
        features = pd.DataFrame(index=market_data.index)
        
        # Calculate profit margins
        high_low_range = (market_data['high'] - market_data['low']) / market_data['close']
        open_close_range = abs(market_data['close'] - market_data['open']) / market_data['close']
        
        # Profit margin indicators
        features['profit_margin_high_low'] = high_low_range
        features['profit_margin_open_close'] = open_close_range
        
        # Rolling profit margins
        for window in [5, 10, 20]:
            features[f'profit_margin_ma_{window}'] = open_close_range.rolling(window).mean()
            features[f'profit_margin_std_{window}'] = open_close_range.rolling(window).std()
        
        # Profit margin thresholds
        features['above_min_margin'] = (open_close_range > self.min_profit_margin).astype(float)
        features['below_max_margin'] = (open_close_range < self.max_profit_margin).astype(float)
        features['optimal_margin_range'] = (
            (open_close_range > self.min_profit_margin) & 
            (open_close_range < self.max_profit_margin)
        ).astype(float)
        
        # Profit margin efficiency
        features['margin_efficiency'] = safe_divide(
            open_close_range, high_low_range, default=0.0
        )
        
        return features

    def _create_transaction_cost_features(self, market_data: pd.DataFrame,
                                        returns: pd.Series,
                                        labels: pd.Series) -> pd.DataFrame:
        """Create transaction cost features."""
        features = pd.DataFrame(index=market_data.index)
        
        # Calculate transaction costs
        high_low_range = (market_data['high'] - market_data['low']) / market_data['close']
        open_close_range = abs(market_data['close'] - market_data['open']) / market_data['close']
        
        # Transaction cost ratios
        features['transaction_cost_ratio'] = safe_divide(
            0.0008, high_low_range, default=0.0  # 0.08% transaction cost
        )
        features['net_profit_potential'] = high_low_range - 0.0008
        
        # Cost-adjusted profit potential
        features['cost_adjusted_profit'] = np.maximum(0.0, high_low_range - 0.0008)
        features['cost_efficiency'] = safe_divide(
            features['cost_adjusted_profit'], high_low_range, default=0.0
        )
        
        # Cost threshold indicators
        features['above_cost_threshold'] = (high_low_range > 0.0008).astype(float)
        features['cost_threshold_ratio'] = safe_divide(high_low_range, 0.0008, default=0.0)
        
        # Rolling cost metrics
        for window in [5, 10, 20]:
            features[f'cost_efficiency_ma_{window}'] = features['cost_efficiency'].rolling(window).mean()
            features[f'net_profit_ma_{window}'] = features['net_profit_potential'].rolling(window).mean()
        
        return features

    def get_profit_based_statistics(self, market_data: pd.DataFrame,
                                  returns: pd.Series,
                                  labels: pd.Series) -> Dict[str, Any]:
        """
        Get comprehensive profit-based feature statistics.
        
        Args:
            market_data: OHLCV market data
            returns: Price returns
            labels: Trading labels
            
        Returns:
            Dictionary with profit-based feature statistics
        """
        try:
            # Create profit-based features
            profit_features = self.create_profit_based_features(market_data, returns, labels)
            
            # Calculate statistics
            feature_stats = {}
            for col in profit_features.columns:
                feature_stats[col] = {
                    'mean': profit_features[col].mean(),
                    'std': profit_features[col].std(),
                    'min': profit_features[col].min(),
                    'max': profit_features[col].max(),
                    'median': profit_features[col].median()
                }
            
            # Calculate profit potential statistics
            high_low_range = (market_data['high'] - market_data['low']) / market_data['close']
            profit_potential_stats = {
                'mean_profit_potential': high_low_range.mean(),
                'std_profit_potential': high_low_range.std(),
                'above_threshold_ratio': (high_low_range > self.profit_threshold).mean(),
                'optimal_margin_ratio': (
                    (high_low_range > self.min_profit_margin) & 
                    (high_low_range < self.max_profit_margin)
                ).mean()
            }
            
            return {
                'feature_statistics': feature_stats,
                'profit_potential_statistics': profit_potential_stats,
                'total_features': len(profit_features.columns),
                'parameters': {
                    'profit_threshold': self.profit_threshold,
                    'risk_reward_ratio': self.risk_reward_ratio,
                    'min_profit_margin': self.min_profit_margin,
                    'max_profit_margin': self.max_profit_margin
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Profit-based statistics calculation failed: {e}")
            raise