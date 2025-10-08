"""
Enhanced Label Design Module

This module provides advanced labeling techniques to address:
1. Transaction cost adjustment in profit-based labels
2. Regime-dependent labeling logic
3. Meta-labeling and triple-barrier methods
4. Non-overlapping window sampling
5. Volatility window definition and freezing

Implements recommendations from Section 2: Label Design & Target Quality
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success


logger = logging.getLogger(__name__)


@dataclass
class TransactionCostConfig:
    """Configuration for transaction cost modeling."""
    
    # Cost components
    maker_fee: float = 0.0002  # 0.02% maker fee
    taker_fee: float = 0.0004  # 0.04% taker fee
    slippage_bps: float = 2.0  # 2 basis points slippage
    
    # Market impact (for larger orders)
    enable_market_impact: bool = False
    market_impact_coefficient: float = 0.0001  # Per 1% of volume
    
    # Total estimated cost
    def total_cost(self, is_aggressive: bool = True) -> float:
        """Calculate total transaction cost."""
        fee = self.taker_fee if is_aggressive else self.maker_fee
        slippage = self.slippage_bps / 10000.0
        return fee + slippage


@dataclass
class VolatilityConfig:
    """Configuration for volatility estimation."""
    
    # Window settings
    lookback_window: int = 48  # 48 periods for volatility
    min_periods: int = 20  # Minimum periods required
    
    # Method
    method: str = "ewm"  # 'ewm' or 'rolling'
    ewm_halflife: int = 24  # Half-life for EWM
    
    # Scaling
    annualization_factor: float = np.sqrt(252 * 24)  # For hourly data
    
    # Freezing (to prevent lookahead)
    freeze_during_training: bool = True


@dataclass
class TripleBarrierConfig:
    """Configuration for triple-barrier method."""
    
    # Barriers (in volatility units)
    profit_barrier_sigma: float = 2.0  # 2σ profit barrier
    stop_loss_barrier_sigma: float = 2.0  # 2σ stop loss barrier
    
    # Time-based barrier
    max_holding_period: int = 24  # Maximum holding period
    
    # Label generation
    label_on_first_touch: bool = True  # Label when first barrier is touched
    
    # Minimum return threshold (in σ)
    min_return_sigma: float = 0.5  # Minimum 0.5σ return to generate label


class EnhancedLabelDesigner:
    """
    Enhanced label designer with transaction costs and regime awareness.
    
    Key Features:
    1. Transaction cost-adjusted profit labels
    2. Regime-dependent thresholds
    3. Triple-barrier method
    4. Non-overlapping sampling
    5. Proper volatility estimation with freezing
    """
    
    def __init__(
        self,
        cost_config: Optional[TransactionCostConfig] = None,
        volatility_config: Optional[VolatilityConfig] = None,
        barrier_config: Optional[TripleBarrierConfig] = None
    ):
        """
        Initialize enhanced label designer.
        
        Args:
            cost_config: Transaction cost configuration
            volatility_config: Volatility estimation configuration
            barrier_config: Triple-barrier configuration
        """
        self.cost_config = cost_config or TransactionCostConfig()
        self.vol_config = volatility_config or VolatilityConfig()
        self.barrier_config = barrier_config or TripleBarrierConfig()
        
        tprint_success("✅ EnhancedLabelDesigner initialized")
        tprint_info(f"   → Transaction cost: {self.cost_config.total_cost():.4%}")
        tprint_info(f"   → Volatility window: {self.vol_config.lookback_window}")
        tprint_info(f"   → Barrier method: enabled")
    
    def calculate_volatility(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
        freeze_at: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Calculate volatility with proper windowing and optional freezing.
        
        Args:
            prices: Price series
            returns: Optional pre-computed returns
            freeze_at: Optional timestamp to freeze volatility (prevent lookahead)
        
        Returns:
            Volatility series
        """
        if returns is None:
            returns = prices.pct_change()
        
        # Calculate volatility based on method
        if self.vol_config.method == "ewm":
            vol = returns.ewm(
                halflife=self.vol_config.ewm_halflife,
                min_periods=self.vol_config.min_periods
            ).std()
        else:  # rolling
            vol = returns.rolling(
                window=self.vol_config.lookback_window,
                min_periods=self.vol_config.min_periods
            ).std()
        
        # Annualize volatility
        vol = vol * self.vol_config.annualization_factor
        
        # Freeze volatility if requested (prevent lookahead)
        if freeze_at is not None and self.vol_config.freeze_during_training:
            future_mask = vol.index > freeze_at
            if future_mask.any():
                last_valid_vol = vol.loc[freeze_at] if freeze_at in vol.index else vol[~future_mask].iloc[-1]
                vol.loc[future_mask] = last_valid_vol
                tprint_info(f"🔒 Frozen volatility at {freeze_at} for {future_mask.sum()} future samples")
        
        return vol
    
    def adjust_returns_for_costs(
        self,
        forward_returns: pd.DataFrame,
        is_aggressive: bool = True
    ) -> pd.DataFrame:
        """
        Adjust forward returns for transaction costs.
        
        Args:
            forward_returns: Forward return DataFrame
            is_aggressive: Whether to use aggressive (taker) fees
        
        Returns:
            Cost-adjusted forward returns
        """
        tprint_info("💰 Adjusting returns for transaction costs...")
        
        total_cost = self.cost_config.total_cost(is_aggressive)
        
        # Subtract round-trip costs (entry + exit)
        adjusted_returns = forward_returns - (2 * total_cost)
        
        tprint_info(f"   → Round-trip cost: {2 * total_cost:.4%}")
        tprint_info(f"   → Adjusted {len(adjusted_returns.columns)} return horizons")
        
        return adjusted_returns
    
    def create_triple_barrier_labels(
        self,
        prices: pd.Series,
        volatility: pd.Series,
        horizons: List[int] = [1, 3, 6, 12, 24]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create labels using triple-barrier method.
        
        Args:
            prices: Price series
            volatility: Volatility series
            horizons: List of horizons to test
        
        Returns:
            Tuple of (labels, touch_times, returns)
        """
        tprint_info(f"🎯 Creating triple-barrier labels for {len(horizons)} horizons...")
        
        labels_dict = {}
        touch_times_dict = {}
        returns_dict = {}
        
        for horizon in horizons:
            tprint_info(f"   → Processing horizon {horizon}...")
            
            barrier_labels = []
            barrier_touch_times = []
            barrier_returns = []
            
            for idx in range(len(prices) - horizon):
                current_price = prices.iloc[idx]
                current_vol = volatility.iloc[idx]
                
                if pd.isna(current_vol) or current_vol == 0:
                    barrier_labels.append(0)
                    barrier_touch_times.append(np.nan)
                    barrier_returns.append(0.0)
                    continue
                
                # Define barriers
                profit_barrier = current_price * (1 + self.barrier_config.profit_barrier_sigma * current_vol)
                stop_barrier = current_price * (1 - self.barrier_config.stop_loss_barrier_sigma * current_vol)
                
                # Check price path over horizon
                future_prices = prices.iloc[idx+1:idx+1+horizon]
                
                # Find first barrier touch
                profit_touches = future_prices >= profit_barrier
                stop_touches = future_prices <= stop_barrier
                
                profit_touch_idx = profit_touches.idxmax() if profit_touches.any() else None
                stop_touch_idx = stop_touches.idxmax() if stop_touches.any() else None
                
                # Determine label
                if profit_touch_idx is not None and (stop_touch_idx is None or profit_touch_idx <= stop_touch_idx):
                    # Profit barrier hit first
                    label = 1
                    touch_time = (profit_touch_idx - prices.index[idx]).total_seconds() / 3600.0
                    final_return = (future_prices.loc[profit_touch_idx] / current_price - 1)
                elif stop_touch_idx is not None:
                    # Stop loss hit first
                    label = -1
                    touch_time = (stop_touch_idx - prices.index[idx]).total_seconds() / 3600.0
                    final_return = (future_prices.loc[stop_touch_idx] / current_price - 1)
                else:
                    # No barrier touched, use end of horizon
                    final_price = prices.iloc[idx+horizon]
                    final_return = (final_price / current_price - 1)
                    
                    # Classify based on minimum threshold
                    if final_return > self.barrier_config.min_return_sigma * current_vol:
                        label = 1
                    elif final_return < -self.barrier_config.min_return_sigma * current_vol:
                        label = -1
                    else:
                        label = 0
                    
                    touch_time = horizon
                
                barrier_labels.append(label)
                barrier_touch_times.append(touch_time)
                barrier_returns.append(final_return)
            
            # Pad to match original length
            barrier_labels.extend([0] * horizon)
            barrier_touch_times.extend([np.nan] * horizon)
            barrier_returns.extend([0.0] * horizon)
            
            labels_dict[f'barrier_label_h{horizon}'] = barrier_labels
            touch_times_dict[f'touch_time_h{horizon}'] = barrier_touch_times
            returns_dict[f'return_h{horizon}'] = barrier_returns
        
        labels_df = pd.DataFrame(labels_dict, index=prices.index)
        touch_times_df = pd.DataFrame(touch_times_dict, index=prices.index)
        returns_df = pd.DataFrame(returns_dict, index=prices.index)
        
        tprint_success(f"✅ Triple-barrier labels created: {len(labels_df)} samples")
        
        return labels_df, touch_times_df, returns_df
    
    def create_non_overlapping_samples(
        self,
        labels: pd.DataFrame,
        horizon: int,
        touch_times: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create non-overlapping samples to avoid label dependence.
        
        Args:
            labels: Label DataFrame
            horizon: Prediction horizon
            touch_times: Optional touch times for adaptive sampling
        
        Returns:
            Non-overlapping sample indices
        """
        tprint_info(f"📊 Creating non-overlapping samples (horizon={horizon})...")
        
        # Simple approach: sample every 'horizon' steps
        if touch_times is None:
            sampled_indices = labels.index[::horizon]
            sampled_labels = labels.loc[sampled_indices]
        else:
            # Adaptive approach: use actual touch times
            sampled_indices = []
            last_sample_idx = 0
            
            for idx in range(len(labels)):
                if idx < last_sample_idx:
                    continue
                
                # Check if this sample is valid
                touch_time_col = f'touch_time_h{horizon}'
                if touch_time_col in touch_times.columns:
                    touch_time = touch_times.iloc[idx][touch_time_col]
                    if not pd.isna(touch_time):
                        sampled_indices.append(labels.index[idx])
                        last_sample_idx = idx + int(touch_time)
                    else:
                        # Use default horizon
                        sampled_indices.append(labels.index[idx])
                        last_sample_idx = idx + horizon
                else:
                    # Fallback to horizon-based sampling
                    sampled_indices.append(labels.index[idx])
                    last_sample_idx = idx + horizon
            
            sampled_labels = labels.loc[sampled_indices]
        
        tprint_info(f"   → Original samples: {len(labels)}")
        tprint_info(f"   → Non-overlapping samples: {len(sampled_labels)}")
        tprint_info(f"   → Reduction: {(1 - len(sampled_labels)/len(labels)):.1%}")
        
        return sampled_labels
    
    def create_regime_dependent_labels(
        self,
        forward_returns: pd.DataFrame,
        volatility: pd.Series,
        regimes: pd.Series,
        regime_thresholds: Optional[Dict[int, float]] = None
    ) -> pd.DataFrame:
        """
        Create regime-dependent labels with adaptive thresholds.
        
        Args:
            forward_returns: Forward returns
            volatility: Volatility series
            regimes: Regime assignments
            regime_thresholds: Optional custom thresholds per regime
        
        Returns:
            Regime-adjusted labels
        """
        tprint_info("🎭 Creating regime-dependent labels...")
        
        # If no custom thresholds, calculate from data
        if regime_thresholds is None:
            regime_thresholds = {}
            
            unique_regimes = regimes.unique()
            for regime in unique_regimes:
                regime_mask = regimes == regime
                regime_vol = volatility[regime_mask].mean()
                
                # Higher threshold for high-vol regimes, lower for low-vol
                regime_thresholds[regime] = regime_vol
            
            tprint_info(f"   → Calculated thresholds for {len(regime_thresholds)} regimes")
        
        # Create labels for each return column
        labels_dict = {}
        
        for col in forward_returns.columns:
            labels = []
            
            for idx, ret in forward_returns[col].items():
                if idx not in regimes.index or idx not in volatility.index:
                    labels.append(0)
                    continue
                
                regime = regimes.loc[idx]
                vol = volatility.loc[idx]
                threshold = regime_thresholds.get(regime, vol)
                
                # Normalize return by regime-specific threshold
                if not pd.isna(ret) and not pd.isna(threshold) and threshold > 0:
                    normalized_ret = ret / threshold
                    
                    # Classify based on normalized return
                    if normalized_ret > 1.0:
                        label = 1
                    elif normalized_ret < -1.0:
                        label = -1
                    else:
                        label = 0
                else:
                    label = 0
                
                labels.append(label)
            
            labels_dict[f'{col}_regime_adj'] = labels
        
        labels_df = pd.DataFrame(labels_dict, index=forward_returns.index)
        
        tprint_success(f"✅ Regime-dependent labels created: {len(labels_df.columns)} columns")
        
        return labels_df
    
    def create_meta_labels(
        self,
        primary_labels: pd.Series,
        features: pd.DataFrame,
        forward_returns: pd.Series,
        threshold: float = 0.0
    ) -> pd.Series:
        """
        Create meta-labels for sizing model (bet size, not direction).
        
        Meta-labeling: Given a primary prediction, predict whether to trade.
        
        Args:
            primary_labels: Primary model predictions (direction)
            features: Features for meta-model
            forward_returns: Actual forward returns
            threshold: Return threshold for positive meta-label
        
        Returns:
            Meta-labels (1 = trade, 0 = don't trade)
        """
        tprint_info("🎯 Creating meta-labels...")
        
        meta_labels = []
        
        for idx in primary_labels.index:
            if idx not in forward_returns.index:
                meta_labels.append(0)
                continue
            
            primary_pred = primary_labels.loc[idx]
            actual_return = forward_returns.loc[idx]
            
            # Meta-label is 1 if primary prediction would be profitable
            if pd.isna(actual_return) or pd.isna(primary_pred):
                meta_label = 0
            else:
                profitable = (primary_pred * actual_return) > threshold
                meta_label = 1 if profitable else 0
            
            meta_labels.append(meta_label)
        
        meta_labels_series = pd.Series(meta_labels, index=primary_labels.index, name='meta_label')
        
        positive_ratio = meta_labels_series.mean()
        tprint_success(f"✅ Meta-labels created: {positive_ratio:.1%} positive")
        
        return meta_labels_series
    
    def validate_label_quality(
        self,
        labels: pd.DataFrame,
        returns: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Validate label quality metrics.
        
        Args:
            labels: Label DataFrame
            returns: Optional returns for validation
        
        Returns:
            Dictionary of quality metrics
        """
        tprint_info("🔍 Validating label quality...")
        
        quality_metrics = {}
        
        for col in labels.columns:
            label_series = labels[col].dropna()
            
            # Class balance
            value_counts = label_series.value_counts()
            if len(value_counts) > 0:
                balance = value_counts.min() / value_counts.max() if value_counts.max() > 0 else 0
            else:
                balance = 0
            
            # Label autocorrelation
            autocorr_lag1 = label_series.autocorr(lag=1) if len(label_series) > 1 else 0
            
            quality_metrics[col] = {
                'n_samples': len(label_series),
                'n_positive': int((label_series > 0).sum()),
                'n_negative': int((label_series < 0).sum()),
                'n_neutral': int((label_series == 0).sum()),
                'class_balance': float(balance),
                'autocorr_lag1': float(autocorr_lag1)
            }
            
            tprint_info(f"   → {col}: balance={balance:.2f}, autocorr={autocorr_lag1:.3f}")
        
        return quality_metrics


def create_enhanced_label_designer(
    cost_config: Optional[TransactionCostConfig] = None,
    volatility_config: Optional[VolatilityConfig] = None,
    barrier_config: Optional[TripleBarrierConfig] = None
) -> EnhancedLabelDesigner:
    """
    Factory function to create EnhancedLabelDesigner.
    
    Args:
        cost_config: Transaction cost configuration
        volatility_config: Volatility configuration
        barrier_config: Barrier configuration
    
    Returns:
        EnhancedLabelDesigner instance
    """
    return EnhancedLabelDesigner(cost_config, volatility_config, barrier_config)