"""
Enhanced Label Design for Pre-Training Pipeline.

This module addresses critical issues in profit-based labeling:
1. Non-overlapping sample windows to ensure independence
2. Explicit volatility lookback with frozen windows
3. Transaction cost adjustment for realistic profitability
4. Regime-dependent labeling logic
5. Triple-barrier method (à la López de Prado)

References:
- Advances in Financial Machine Learning (López de Prado, 2018)
- Machine Learning for Asset Managers (López de Prado, 2020)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import system_logger


class LabelingMethod(Enum):
    """Available labeling methods."""
    
    FIXED_HORIZON = "fixed_horizon"  # Simple fixed horizon returns
    TRIPLE_BARRIER = "triple_barrier"  # Triple-barrier method
    META_LABELING = "meta_labeling"  # Meta-labeling for bet sizing


@dataclass
class TransactionCostModel:
    """Model for transaction costs."""
    
    maker_fee_bps: float = 2.0  # Maker fee in basis points (0.02%)
    taker_fee_bps: float = 4.0  # Taker fee in basis points (0.04%)
    slippage_bps: float = 2.0  # Slippage in basis points
    market_impact_coeff: float = 0.0  # Market impact coefficient
    
    def total_cost_one_way(self, is_aggressive: bool = True, trade_size_usd: float = 10000) -> float:
        """
        Calculate one-way transaction cost.
        
        Args:
            is_aggressive: True for taker (market order), False for maker (limit)
            trade_size_usd: Trade size in USD for market impact calculation
        
        Returns:
            Total cost as a fraction (e.g., 0.0006 = 6 basis points)
        """
        # Base fee
        fee = self.taker_fee_bps if is_aggressive else self.maker_fee_bps
        
        # Add slippage
        cost = (fee + self.slippage_bps) / 10000.0
        
        # Add market impact (proportional to sqrt of trade size)
        if self.market_impact_coeff > 0:
            impact = self.market_impact_coeff * np.sqrt(trade_size_usd / 10000)
            cost += impact / 10000.0
        
        return cost
    
    def roundtrip_cost(self, is_aggressive: bool = True, trade_size_usd: float = 10000) -> float:
        """Calculate round-trip transaction cost."""
        return 2 * self.total_cost_one_way(is_aggressive, trade_size_usd)


@dataclass
class VolatilityConfig:
    """Configuration for volatility estimation."""
    
    lookback_window: int = 48  # Number of bars for volatility calculation
    method: str = "std"  # 'std', 'ewm', 'parkinson', 'garman_klass'
    min_periods: int = 20  # Minimum periods required
    clip_sigma_range: Tuple[float, float] = (0.5, 3.0)  # Clip to [0.5σ, 3σ]
    
    # For EWMA
    ewm_halflife: int = 24  # Half-life for exponential weighting
    
    # For range-based estimators
    use_ohlc: bool = False  # Whether to use OHLC for estimators


@dataclass
class TripleBarrierConfig:
    """Configuration for triple-barrier labeling."""
    
    profit_target_sigma: float = 2.0  # Upper barrier in units of σ
    stop_loss_sigma: float = 2.0  # Lower barrier in units of σ
    max_horizon_bars: int = 48  # Maximum holding period
    
    # Barrier asymmetry for trend-following
    asymmetric_barriers: bool = False
    trend_adjustment: float = 0.0  # Adjust barriers based on trend
    
    # Transaction cost adjustment
    adjust_for_costs: bool = True
    transaction_cost: Optional[TransactionCostModel] = None
    
    def __post_init__(self):
        """Initialize transaction cost model if needed."""
        if self.adjust_for_costs and self.transaction_cost is None:
            self.transaction_cost = TransactionCostModel()


@dataclass
class RegimeLabelingConfig:
    """Configuration for regime-dependent labeling."""
    
    enable_regime_adaptation: bool = True
    regime_column: Optional[str] = None
    
    # Regime-specific parameters
    regime_params: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        """Initialize default regime parameters."""
        if self.regime_params is None:
            self.regime_params = {
                'trending': {
                    'profit_target_sigma': 2.5,
                    'stop_loss_sigma': 1.5,
                    'volatility_scale': 1.0
                },
                'mean_reverting': {
                    'profit_target_sigma': 1.5,
                    'stop_loss_sigma': 2.0,
                    'volatility_scale': 0.8
                },
                'volatile': {
                    'profit_target_sigma': 3.0,
                    'stop_loss_sigma': 2.5,
                    'volatility_scale': 1.2
                }
            }


class EnhancedLabeler:
    """
    Enhanced labeling system with proper sampling and cost adjustment.
    
    Key features:
    - Non-overlapping sample windows for independence
    - Frozen volatility windows (no future data leakage)
    - Transaction cost adjustment
    - Regime-adaptive labeling
    - Triple-barrier method
    """
    
    def __init__(
        self,
        volatility_config: Optional[VolatilityConfig] = None,
        barrier_config: Optional[TripleBarrierConfig] = None,
        regime_config: Optional[RegimeLabelingConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the enhanced labeler.
        
        Args:
            volatility_config: Configuration for volatility estimation
            barrier_config: Configuration for triple-barrier method
            regime_config: Configuration for regime-adaptive labeling
            logger: Optional logger instance
        """
        self.volatility_config = volatility_config or VolatilityConfig()
        self.barrier_config = barrier_config or TripleBarrierConfig()
        self.regime_config = regime_config or RegimeLabelingConfig()
        self.logger = logger or system_logger.getChild('EnhancedLabeler')
    
    def compute_volatility(
        self,
        prices: pd.Series,
        ohlc: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Compute rolling volatility using specified method.
        
        Args:
            prices: Price series (typically close prices)
            ohlc: Optional DataFrame with OHLC data for advanced estimators
        
        Returns:
            Series with volatility estimates
        """
        if self.volatility_config.method == "std":
            # Simple rolling standard deviation of returns
            returns = prices.pct_change()
            volatility = returns.rolling(
                window=self.volatility_config.lookback_window,
                min_periods=self.volatility_config.min_periods
            ).std()
            
        elif self.volatility_config.method == "ewm":
            # Exponentially weighted moving average
            returns = prices.pct_change()
            volatility = returns.ewm(
                halflife=self.volatility_config.ewm_halflife,
                min_periods=self.volatility_config.min_periods
            ).std()
            
        elif self.volatility_config.method == "parkinson" and ohlc is not None:
            # Parkinson's high-low estimator
            hl = np.log(ohlc['high'] / ohlc['low'])
            volatility = hl.rolling(
                window=self.volatility_config.lookback_window,
                min_periods=self.volatility_config.min_periods
            ).apply(lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2))))
            
        elif self.volatility_config.method == "garman_klass" and ohlc is not None:
            # Garman-Klass estimator
            hl = np.log(ohlc['high'] / ohlc['low'])**2
            co = np.log(ohlc['close'] / ohlc['open'])**2
            volatility = (0.5 * hl - (2 * np.log(2) - 1) * co).rolling(
                window=self.volatility_config.lookback_window,
                min_periods=self.volatility_config.min_periods
            ).apply(lambda x: np.sqrt(np.mean(x)))
            
        else:
            # Fallback to simple std
            self.logger.warning(f"Unknown volatility method: {self.volatility_config.method}, using 'std'")
            returns = prices.pct_change()
            volatility = returns.rolling(
                window=self.volatility_config.lookback_window,
                min_periods=self.volatility_config.min_periods
            ).std()
        
        # Forward fill initial NaNs
        volatility = volatility.fillna(method='bfill')
        
        return volatility
    
    def create_non_overlapping_labels(
        self,
        prices: pd.Series,
        horizon_bars: int,
        ohlc: Optional[pd.DataFrame] = None,
        regime_labels: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Create labels with non-overlapping sampling windows.
        
        This ensures sample independence by sampling only once per horizon period.
        
        Args:
            prices: Price series
            horizon_bars: Horizon length in bars
            ohlc: Optional OHLC data for advanced volatility estimators
            regime_labels: Optional regime classifications
        
        Returns:
            DataFrame with labels sampled at non-overlapping intervals
        """
        # Compute volatility
        volatility = self.compute_volatility(prices, ohlc)
        
        # Sample non-overlapping points
        sample_indices = list(range(0, len(prices), horizon_bars))
        
        labels_list = []
        
        for i in sample_indices:
            if i + horizon_bars >= len(prices):
                break  # Skip if we don't have full horizon
            
            current_price = prices.iloc[i]
            future_price = prices.iloc[i + horizon_bars]
            current_vol = volatility.iloc[i]
            
            if pd.isna(current_vol) or current_vol <= 1e-8:
                continue
            
            # Get regime for this sample if available
            regime = None
            if regime_labels is not None and self.regime_config.enable_regime_adaptation:
                regime = regime_labels.iloc[i]
            
            # Compute return
            ret = (future_price - current_price) / current_price
            
            # Adjust for transaction costs
            if self.barrier_config.adjust_for_costs:
                cost = self.barrier_config.transaction_cost.roundtrip_cost()
                ret_adjusted = ret - cost
            else:
                ret_adjusted = ret
            
            # Normalize by volatility (sigma-scaling)
            sigma_scaled_return = ret_adjusted / (current_vol * np.sqrt(horizon_bars))
            
            # Adapt thresholds based on regime
            profit_threshold = self.barrier_config.profit_target_sigma
            loss_threshold = -self.barrier_config.stop_loss_sigma
            
            if regime and regime in self.regime_config.regime_params:
                params = self.regime_config.regime_params[regime]
                profit_threshold = params.get('profit_target_sigma', profit_threshold)
                loss_threshold = -params.get('stop_loss_sigma', abs(loss_threshold))
            
            # Create label
            if sigma_scaled_return > profit_threshold:
                label = 1  # Long signal
            elif sigma_scaled_return < loss_threshold:
                label = -1  # Short signal
            else:
                label = 0  # No signal
            
            labels_list.append({
                'timestamp': prices.index[i],
                'label': label,
                'raw_return': ret,
                'adjusted_return': ret_adjusted,
                'sigma_scaled_return': sigma_scaled_return,
                'volatility': current_vol,
                'regime': regime,
                'horizon_bars': horizon_bars
            })
        
        if not labels_list:
            self.logger.warning("No valid labels created")
            return pd.DataFrame()
        
        labels_df = pd.DataFrame(labels_list)
        labels_df.set_index('timestamp', inplace=True)
        
        self.logger.info(
            f"Created {len(labels_df)} non-overlapping labels "
            f"(sampling every {horizon_bars} bars)"
        )
        
        return labels_df
    
    def create_triple_barrier_labels(
        self,
        prices: pd.Series,
        ohlc: Optional[pd.DataFrame] = None,
        regime_labels: Optional[pd.Series] = None,
        sample_every_n_bars: int = 1
    ) -> pd.DataFrame:
        """
        Create labels using the triple-barrier method.
        
        Args:
            prices: Price series
            ohlc: Optional OHLC data
            regime_labels: Optional regime classifications
            sample_every_n_bars: Sample frequency to avoid overlaps
        
        Returns:
            DataFrame with triple-barrier labels
        """
        # Compute volatility
        volatility = self.compute_volatility(prices, ohlc)
        
        labels_list = []
        
        for i in range(0, len(prices) - self.barrier_config.max_horizon_bars, sample_every_n_bars):
            entry_price = prices.iloc[i]
            entry_vol = volatility.iloc[i]
            
            if pd.isna(entry_vol) or entry_vol <= 1e-8:
                continue
            
            # Get regime
            regime = None
            if regime_labels is not None and self.regime_config.enable_regime_adaptation:
                regime = regime_labels.iloc[i]
            
            # Adapt barriers based on regime
            profit_sigma = self.barrier_config.profit_target_sigma
            loss_sigma = self.barrier_config.stop_loss_sigma
            
            if regime and regime in self.regime_config.regime_params:
                params = self.regime_config.regime_params[regime]
                profit_sigma = params.get('profit_target_sigma', profit_sigma)
                loss_sigma = params.get('stop_loss_sigma', loss_sigma)
                vol_scale = params.get('volatility_scale', 1.0)
                entry_vol *= vol_scale
            
            # Set barriers (in price units)
            upper_barrier = entry_price * (1 + profit_sigma * entry_vol)
            lower_barrier = entry_price * (1 - loss_sigma * entry_vol)
            
            # Adjust for transaction costs
            if self.barrier_config.adjust_for_costs:
                cost = self.barrier_config.transaction_cost.roundtrip_cost()
                upper_barrier = entry_price * (1 + profit_sigma * entry_vol - cost)
                lower_barrier = entry_price * (1 - loss_sigma * entry_vol - cost)
            
            # Scan forward to find barrier touch
            label = 0
            exit_bar = self.barrier_config.max_horizon_bars
            exit_return = 0.0
            
            for j in range(1, self.barrier_config.max_horizon_bars + 1):
                if i + j >= len(prices):
                    break
                
                current_price = prices.iloc[i + j]
                
                # Check upper barrier (profit)
                if current_price >= upper_barrier:
                    label = 1
                    exit_bar = j
                    exit_return = (current_price - entry_price) / entry_price
                    break
                
                # Check lower barrier (loss)
                if current_price <= lower_barrier:
                    label = -1
                    exit_bar = j
                    exit_return = (current_price - entry_price) / entry_price
                    break
            
            # If no barrier touched, use final return
            if label == 0 and i + self.barrier_config.max_horizon_bars < len(prices):
                final_price = prices.iloc[i + self.barrier_config.max_horizon_bars]
                exit_return = (final_price - entry_price) / entry_price
                
                # Determine label based on final return
                sigma_scaled = exit_return / (entry_vol * np.sqrt(self.barrier_config.max_horizon_bars))
                if sigma_scaled > 0.5:
                    label = 1
                elif sigma_scaled < -0.5:
                    label = -1
            
            labels_list.append({
                'timestamp': prices.index[i],
                'label': label,
                'exit_return': exit_return,
                'exit_bar': exit_bar,
                'volatility': entry_vol,
                'regime': regime,
                'upper_barrier': upper_barrier,
                'lower_barrier': lower_barrier
            })
        
        if not labels_list:
            self.logger.warning("No valid triple-barrier labels created")
            return pd.DataFrame()
        
        labels_df = pd.DataFrame(labels_list)
        labels_df.set_index('timestamp', inplace=True)
        
        self.logger.info(
            f"Created {len(labels_df)} triple-barrier labels "
            f"(max horizon: {self.barrier_config.max_horizon_bars} bars)"
        )
        
        return labels_df


def create_enhanced_labels(
    prices: pd.Series,
    horizon_bars: int = 48,
    method: LabelingMethod = LabelingMethod.TRIPLE_BARRIER,
    ohlc: Optional[pd.DataFrame] = None,
    regime_labels: Optional[pd.Series] = None,
    volatility_config: Optional[VolatilityConfig] = None,
    barrier_config: Optional[TripleBarrierConfig] = None,
    regime_config: Optional[RegimeLabelingConfig] = None,
    sample_every_n_bars: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Convenience function to create enhanced labels.
    
    Args:
        prices: Price series
        horizon_bars: Horizon length (for fixed horizon method)
        method: Labeling method to use
        ohlc: Optional OHLC data
        regime_labels: Optional regime classifications
        volatility_config: Volatility configuration
        barrier_config: Triple-barrier configuration
        regime_config: Regime labeling configuration
        sample_every_n_bars: Sampling frequency (None = use horizon_bars)
        logger: Optional logger
    
    Returns:
        DataFrame with labels
    """
    labeler = EnhancedLabeler(
        volatility_config=volatility_config,
        barrier_config=barrier_config,
        regime_config=regime_config,
        logger=logger
    )
    
    if method == LabelingMethod.FIXED_HORIZON:
        return labeler.create_non_overlapping_labels(
            prices=prices,
            horizon_bars=horizon_bars,
            ohlc=ohlc,
            regime_labels=regime_labels
        )
    elif method == LabelingMethod.TRIPLE_BARRIER:
        sample_freq = sample_every_n_bars or horizon_bars
        return labeler.create_triple_barrier_labels(
            prices=prices,
            ohlc=ohlc,
            regime_labels=regime_labels,
            sample_every_n_bars=sample_freq
        )
    else:
        raise ValueError(f"Unsupported labeling method: {method}")


__all__ = [
    'EnhancedLabeler',
    'LabelingMethod',
    'TransactionCostModel',
    'VolatilityConfig',
    'TripleBarrierConfig',
    'RegimeLabelingConfig',
    'create_enhanced_labels',
]