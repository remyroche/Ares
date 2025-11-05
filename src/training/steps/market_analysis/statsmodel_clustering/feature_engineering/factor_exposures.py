"""
Factor Exposure Calculator for Market Features

This module calculates factor exposures including market, size, value,
and momentum factors for comprehensive feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import warnings
from scipy import stats

# Import utilities
try:
    from src.utils.tprint import (
        tprint_info, tprint_success, tprint_warning, tprint_error
    )
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')


@dataclass
class FactorConfig:
    """Configuration for factor exposure calculation."""
    factor_types: List[str] = None
    market_proxy: str = 'close'  # 'close', 'equal_weighted', 'value_weighted'
    size_proxy: str = 'market_cap'  # 'market_cap', 'volume', 'price'
    value_proxy: str = 'book_value'  # 'book_value', 'earnings', 'sales'
    momentum_periods: List[int] = None
    
    # Calculation settings
    use_log_returns: bool = True
    use_robust_scaling: bool = True
    neutralize_factors: bool = True
    
    def __post_init__(self):
        if self.factor_types is None:
            self.factor_types = ['market', 'size', 'value', 'momentum']
        if self.momentum_periods is None:
            self.momentum_periods = [1, 3, 6, 12]


class FactorExposureCalculator:
    """
    Factor exposure calculator for market features.
    
    This class calculates various factor exposures including market beta,
    size, value, and momentum factors for comprehensive analysis.
    """
    
    def __init__(self, config: Optional[FactorConfig] = None):
        """
        Initialize factor exposure calculator.
        
        Args:
            config: Configuration for factor calculation
        """
        self.config = config or FactorConfig()
        
        tprint_info("🔧 Initialized Factor Exposure Calculator")
        tprint_info(f"📊 Factor types: {self.config.factor_types}")
    
    def calculate_factor_exposures(self, 
                               price_data: pd.DataFrame,
                               volume_data: Optional[pd.DataFrame] = None,
                               market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate factor exposures for all configured factor types.
        
        Args:
            price_data: DataFrame with OHLC price data
            volume_data: Optional volume data
            market_data: Optional market data for factor calculations
            
        Returns:
            DataFrame with factor exposures
        """
        tprint_info("🔍 Calculating factor exposures")
        
        try:
            # Initialize factor DataFrame
            factor_features = pd.DataFrame(index=price_data.index)
            
            # Calculate returns for factor calculations
            returns = self._calculate_returns(price_data)
            
            # Calculate each factor type
            if 'market' in self.config.factor_types:
                market_factors = self._calculate_market_factors(returns, market_data)
                factor_features = pd.concat([factor_features, market_factors], axis=1)
                tprint_info("✅ Added market factors")
            
            if 'size' in self.config.factor_types:
                size_factors = self._calculate_size_factors(price_data, volume_data, market_data)
                factor_features = pd.concat([factor_features, size_factors], axis=1)
                tprint_info("✅ Added size factors")
            
            if 'value' in self.config.factor_types:
                value_factors = self._calculate_value_factors(price_data, market_data)
                factor_features = pd.concat([factor_features, value_factors], axis=1)
                tprint_info("✅ Added value factors")
            
            if 'momentum' in self.config.factor_types:
                momentum_factors = self._calculate_momentum_factors(returns)
                factor_features = pd.concat([factor_features, momentum_factors], axis=1)
                tprint_info("✅ Added momentum factors")
            
            # Apply neutralization if requested
            if self.config.neutralize_factors:
                factor_features = self._neutralize_factors(factor_features)
                tprint_info("✅ Applied factor neutralization")
            
            tprint_success(f"✅ Factor exposure calculation complete: {factor_features.shape[1]} factors")
            return factor_features
            
        except Exception as e:
            tprint_error(f"❌ Factor exposure calculation failed: {e}")
            raise
    
    def _calculate_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns for factor calculations."""
        tprint_info("📈 Calculating returns for factor calculations")
        
        close_prices = price_data['close']
        
        if self.config.use_log_returns:
            tprint_info("📊 Using log returns")
            returns = np.log(close_prices / close_prices.shift(1))
        else:
            tprint_info("📊 Using simple returns")
            returns = close_prices.pct_change()
        
        tprint_success("✅ Returns calculated successfully")
        return pd.DataFrame({'returns': returns})
    
    def _calculate_market_factors(self,
                              returns: pd.DataFrame,
                              market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate market factor exposures."""
        tprint_info("📊 Calculating market factor exposures")
        
        market_factors = pd.DataFrame(index=returns.index)
        
        if market_data is not None:
            tprint_info(f"📈 Using provided market data with proxy: {self.config.market_proxy}")
            # Use provided market data
            if self.config.market_proxy == 'close':
                market_return = market_data['close'].pct_change()
            elif self.config.market_proxy == 'equal_weighted':
                # Equal-weighted market return
                tprint_info("📊 Using equal-weighted market return")
                market_return = market_data.mean(axis=1).pct_change(axis=1)
            elif self.config.market_proxy == 'value_weighted':
                # Value-weighted market return
                tprint_info("📊 Using value-weighted market return")
                weights = market_data['market_cap'] if 'market_cap' in market_data.columns else market_data.iloc[:, 0]
                market_return = (market_data * weights).sum(axis=1).pct_change()
        else:
            tprint_info("📈 Using price data as market proxy")
            # Use price data as market proxy
            if self.config.market_proxy == 'close':
                market_return = returns['returns']
            else:
                # Simple market return from price data
                market_return = returns['returns']
        
        # Calculate market beta (rolling)
        tprint_info("📊 Calculating rolling market betas")
        for window in [20, 60, 252]:  # 1 month, 3 months, 1 year
            if len(returns) >= window:
                market_beta = self._calculate_rolling_beta(
                    returns['returns'], market_return, window
                )
                market_factors[f'market_beta_{window}d'] = market_beta
        
        # Calculate market correlation
        tprint_info("📊 Calculating rolling market correlations")
        for window in [20, 60, 252]:
            if len(returns) >= window:
                market_corr = returns['returns'].rolling(window).corr(market_return.rolling(window))
                market_factors[f'market_corr_{window}d'] = market_corr
        
        # Calculate market volatility exposure
        tprint_info("📊 Calculating market volatility exposure")
        market_vol = returns['returns'].rolling(20).std()
        market_factors['market_volatility_exposure'] = market_vol
        
        tprint_success("✅ Market factors calculated successfully")
        return market_factors.add_prefix('market_')
    
    def _calculate_size_factors(self,
                             price_data: pd.DataFrame,
                             volume_data: Optional[pd.DataFrame] = None,
                             market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate size factor exposures."""
        tprint_info(f"📊 Calculating size factors using proxy: {self.config.size_proxy}")
        
        size_factors = pd.DataFrame(index=price_data.index)
        
        if self.config.size_proxy == 'market_cap' and market_data is not None:
            tprint_info("📈 Using market capitalization as size proxy")
            # Use market capitalization
            if 'market_cap' in market_data.columns:
                market_cap = market_data['market_cap']
                
                # Calculate size deciles
                tprint_info("📊 Calculating size deciles")
                size_deciles = market_cap.rank(pct=True) // 0.1 + 1
                size_factors['size_decile'] = size_deciles
                
                # Calculate size z-scores
                tprint_info("📊 Calculating size z-scores")
                size_z = (market_cap - market_cap.mean()) / market_cap.std()
                size_factors['size_zscore'] = size_z
                
                # Calculate size factors (long-short)
                tprint_info("📊 Creating size factor long/short positions")
                size_factors['size_factor_long'] = np.where(size_deciles >= 8, 1, 0)  # Large cap
                size_factors['size_factor_short'] = np.where(size_deciles <= 3, 1, 0)  # Small cap
                size_factors['size_factor'] = size_factors['size_factor_long'] - size_factors['size_factor_short']
        
        elif self.config.size_proxy == 'volume' and volume_data is not None:
            tprint_info("📊 Using volume as size proxy")
            # Use volume as size proxy
            if 'volume' in volume_data.columns:
                volume = volume_data['volume']
                
                # Calculate volume deciles
                tprint_info("📊 Calculating volume deciles")
                volume_deciles = volume.rank(pct=True) // 0.1 + 1
                size_factors['volume_decile'] = volume_deciles
                
                # Calculate volume z-scores
                tprint_info("📊 Calculating volume z-scores")
                volume_z = (volume - volume.mean()) / volume.std()
                size_factors['volume_zscore'] = volume_z
                
                # Calculate volume factors
                tprint_info("📊 Creating volume factor long/short positions")
                size_factors['volume_factor_long'] = np.where(volume_deciles >= 8, 1, 0)  # High volume
                size_factors['volume_factor_short'] = np.where(volume_deciles <= 3, 1, 0)  # Low volume
                size_factors['volume_factor'] = size_factors['volume_factor_long'] - size_factors['volume_factor_short']
        
        elif self.config.size_proxy == 'price':
            tprint_info("📊 Using price as size proxy (inverted)")
            # Use price as size proxy (inverse - low price = large cap)
            close_prices = price_data['close']
            
            # Calculate price deciles (inverted for size)
            tprint_info("📊 Calculating price deciles (inverted for size)")
            price_deciles = close_prices.rank(pct=True, ascending=False) // 0.1 + 1
            size_factors['price_decile'] = price_deciles
            
            # Calculate price factors (inverted)
            tprint_info("📊 Creating price factor long/short positions (inverted)")
            size_factors['price_factor_long'] = np.where(price_deciles <= 3, 1, 0)  # High price = small cap
            size_factors['price_factor_short'] = np.where(price_deciles >= 8, 1, 0)  # Low price = large cap
            size_factors['price_factor'] = size_factors['price_factor_long'] - size_factors['price_factor_short']
        
        # Calculate size momentum
        tprint_info("📈 Calculating size momentum factors")
        returns = price_data['close'].pct_change()
        for period in [1, 3, 6, 12]:
            if len(returns) >= period:
                size_momentum = returns.rolling(period).mean()
                size_factors[f'size_momentum_{period}m'] = size_momentum
        
        tprint_success("✅ Size factors calculated successfully")
        return size_factors.add_prefix('size_')
    
    def _calculate_value_factors(self,
                             price_data: pd.DataFrame,
                             market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate value factor exposures."""
        tprint_info(f"📊 Calculating value factors using proxy: {self.config.value_proxy}")
        
        value_factors = pd.DataFrame(index=price_data.index)
        
        if market_data is not None:
            tprint_info("📈 Using fundamental data for value factors")
            # Use fundamental data if available
            if self.config.value_proxy == 'book_value' and 'book_value' in market_data.columns:
                tprint_info("📊 Using book-to-market ratio")
                book_value = market_data['book_value']
                
                # Calculate book-to-market ratio
                market_cap = market_data.get('market_cap', price_data['close'] * 1e6)  # Fallback
                btm_ratio = book_value / market_cap
                
                # Calculate value deciles
                tprint_info("📊 Calculating value deciles")
                value_deciles = btm_ratio.rank(pct=True) // 0.1 + 1
                value_factors['btm_decile'] = value_deciles
                
                # Calculate value factors (high BTM = value stock)
                tprint_info("📊 Creating value factor long/short positions")
                value_factors['value_factor_long'] = np.where(value_deciles >= 8, 1, 0)  # High BTM
                value_factors['value_factor_short'] = np.where(value_deciles <= 3, 1, 0)  # Low BTM
                value_factors['value_factor'] = value_factors['value_factor_long'] - value_factors['value_factor_short']
                
                # Calculate value z-scores
                tprint_info("📊 Calculating value z-scores")
                value_z = (btm_ratio - btm_ratio.mean()) / btm_ratio.std()
                value_factors['value_zscore'] = value_z
            
            elif self.config.value_proxy == 'earnings' and 'earnings' in market_data.columns:
                tprint_info("📊 Using P/E ratio")
                earnings = market_data['earnings']
                market_cap = market_data.get('market_cap', price_data['close'] * 1e6)
                
                # Calculate P/E ratio
                pe_ratio = market_cap / earnings
                
                # Calculate earnings yield (inverse P/E)
                earnings_yield = earnings / market_cap
                
                # Calculate value deciles
                tprint_info("📊 Calculating P/E deciles")
                value_deciles = pe_ratio.rank(pct=True) // 0.1 + 1
                value_factors['pe_decile'] = value_deciles
                
                # Calculate value factors (low P/E = value stock)
                tprint_info("📊 Creating value factor long/short positions")
                value_factors['value_factor_long'] = np.where(value_deciles <= 3, 1, 0)  # Low P/E
                value_factors['value_factor_short'] = np.where(value_deciles >= 8, 1, 0)  # High P/E
                value_factors['value_factor'] = value_factors['value_factor_long'] - value_factors['value_factor_short']
                
                # Calculate earnings yield z-scores
                tprint_info("📊 Calculating earnings yield z-scores")
                yield_z = (earnings_yield - earnings_yield.mean()) / earnings_yield.std()
                value_factors['earnings_yield_zscore'] = yield_z
        
        else:
            tprint_info("📈 Using price-based value factors as fallback")
            # Use price-based value factors as fallback
            close_prices = price_data['close']
            
            # Calculate price-to-earnings proxy using price momentum
            for period in [1, 3, 6, 12]:
                if len(close_prices) >= period:
                    price_momentum = close_prices.pct_change(period).rolling(period).mean()
                    
                    # Value factor based on price momentum (inverse relationship)
                    value_factors[f'value_momentum_{period}m'] = -price_momentum
        
        tprint_success("✅ Value factors calculated successfully")
        return value_factors.add_prefix('value_')
    
    def _calculate_momentum_factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum factor exposures."""
        tprint_info(f"📈 Calculating momentum factors for periods: {self.config.momentum_periods}")
        
        momentum_factors = pd.DataFrame(index=returns.index)
        
        for period in self.config.momentum_periods:
            if len(returns) >= period:
                tprint_info(f"📊 Calculating {period}-period momentum factors")
                
                # Price momentum
                price_momentum = returns['returns'].rolling(period).mean()
                momentum_factors[f'price_momentum_{period}m'] = price_momentum
                
                # Cumulative returns
                cumulative_returns = (1 + returns['returns']).rolling(period).prod() - 1
                momentum_factors[f'cumulative_return_{period}m'] = cumulative_returns
                
                # Momentum reversal factor
                short_momentum = returns['returns'].rolling(period//2).mean()
                long_momentum = returns['returns'].rolling(period).mean()
                momentum_reversal = short_momentum - long_momentum
                momentum_factors[f'momentum_reversal_{period}m'] = momentum_reversal
                
                # Momentum strength
                momentum_strength = abs(price_momentum)
                momentum_factors[f'momentum_strength_{period}m'] = momentum_strength
                
                # Momentum consistency
                momentum_consistency = (returns['returns'].rolling(period).apply(
                    lambda x: np.sum(np.sign(x)) / period
                ))
                momentum_factors[f'momentum_consistency_{period}m'] = momentum_consistency
        
        tprint_success("✅ Momentum factors calculated successfully")
        return momentum_factors.add_prefix('momentum_')
    
    def _calculate_rolling_beta(self,
                              asset_returns: pd.Series,
                              market_returns: pd.Series,
                              window: int) -> pd.Series:
        """Calculate rolling beta coefficient."""
        tprint_info(f"📊 Calculating {window}-day rolling beta")
        
        # Calculate covariance and variance
        tprint_info("📈 Calculating rolling covariance and variance")
        covariance = asset_returns.rolling(window).cov(market_returns.rolling(window))
        market_variance = market_returns.rolling(window).var()
        
        # Calculate beta (avoid division by zero)
        tprint_info("🔄 Computing beta coefficient")
        beta = covariance / (market_variance + 1e-10)
        
        tprint_success("✅ Rolling beta calculated successfully")
        return beta
    
    def _neutralize_factors(self, factor_features: pd.DataFrame) -> pd.DataFrame:
        """Neutralize factors to have zero mean and unit variance."""
        tprint_info(f"🔄 Neutralizing factors (robust_scaling={self.config.use_robust_scaling})")
        
        neutralized_features = factor_features.copy()
        
        for col in factor_features.columns:
            if self.config.use_robust_scaling:
                # Use median and MAD for robust scaling
                tprint_info(f"📊 Applying robust scaling to {col}")
                median = factor_features[col].median()
                mad = (factor_features[col] - median).abs().median()
                
                if mad > 0:
                    neutralized_features[col] = (factor_features[col] - median) / mad
                else:
                    neutralized_features[col] = 0.0
            else:
                # Use mean and std for standard scaling
                tprint_info(f"📊 Applying standard scaling to {col}")
                mean = factor_features[col].mean()
                std = factor_features[col].std()
                
                if std > 0:
                    neutralized_features[col] = (factor_features[col] - mean) / std
                else:
                    neutralized_features[col] = 0.0
        
        tprint_success("✅ Factor neutralization complete")
        return neutralized_features


def create_factor_exposure_calculator(
    factor_types: Optional[List[str]] = None,
    market_proxy: str = 'close',
    size_proxy: str = 'market_cap',
    value_proxy: str = 'book_value',
    momentum_periods: Optional[List[int]] = None,
    use_log_returns: bool = True,
    neutralize_factors: bool = True
) -> FactorExposureCalculator:
    """
    Factory function to create factor exposure calculator.
    
    Args:
        factor_types: List of factor types to calculate
        market_proxy: Market proxy for calculations
        size_proxy: Size proxy for calculations
        value_proxy: Value proxy for calculations
        momentum_periods: List of momentum periods
        use_log_returns: Use log returns for calculations
        neutralize_factors: Whether to neutralize factors
        
    Returns:
        FactorExposureCalculator instance
    """
    tprint_info("🏭 Creating Factor Exposure Calculator with factory function")
    
    config = FactorConfig(
        factor_types=factor_types,
        market_proxy=market_proxy,
        size_proxy=size_proxy,
        value_proxy=value_proxy,
        momentum_periods=momentum_periods,
        use_log_returns=use_log_returns,
        neutralize_factors=neutralize_factors
    )
    
    tprint_info(f"📊 Configuration: factor_types={factor_types}, market_proxy={market_proxy}")
    tprint_info(f"📊 Configuration: size_proxy={size_proxy}, value_proxy={value_proxy}")
    tprint_info(f"📊 Configuration: momentum_periods={momentum_periods}, use_log_returns={use_log_returns}")
    
    calculator = FactorExposureCalculator(config)
    tprint_success("✅ Factor Exposure Calculator created successfully")
    return calculator