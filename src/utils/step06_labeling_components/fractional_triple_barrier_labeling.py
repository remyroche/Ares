"""
Fractional Triple Barrier Labeling - Moved to Utilities

This module contains the original step06 fractional triple barrier labeling functionality
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

class FractionalTripleBarrierLabeling:
    """
    Fractional triple barrier labeling with fractional differentiation.
    This is the original step06 functionality now available as utilities.
    """
    
    def __init__(self, d: float = 0.5, threshold: float = 0.01,
                 profit_take_multiplier: float = 0.004,
                 stop_loss_multiplier: float = 0.003,
                 transaction_cost: float = 0.0008):
        """
        Initialize fractional triple barrier labeling.
        
        Args:
            d: Fractional differentiation order
            threshold: Threshold for stationarity
            profit_take_multiplier: Profit take threshold
            stop_loss_multiplier: Stop loss threshold
            transaction_cost: Transaction cost
        """
        self.d = d
        self.threshold = threshold
        self.profit_take_multiplier = profit_take_multiplier
        self.stop_loss_multiplier = stop_loss_multiplier
        self.transaction_cost = transaction_cost
        self.logger = logger
        
        # Validate parameters
        self._validate_parameters()
        
        self.logger.info("🔢 Fractional Triple Barrier Labeling (Step06 Utilities) initialized")
        self.logger.info(f"   Differentiation order: {d}")
        self.logger.info(f"   Stationarity threshold: {threshold}")
        self.logger.info(f"   Profit take: {profit_take_multiplier:.4f}")
        self.logger.info(f"   Stop loss: {stop_loss_multiplier:.4f}")

    def _validate_parameters(self) -> None:
        """Validate fractional triple barrier labeling parameters."""
        if not validate_range(self.d, 0.0, 1.0):
            raise MathValidationError("Differentiation order must be between 0 and 1")
        if not validate_positive(self.threshold):
            raise MathValidationError("Stationarity threshold must be positive")
        if not validate_positive(self.profit_take_multiplier):
            raise MathValidationError("Profit take multiplier must be positive")
        if not validate_positive(self.stop_loss_multiplier):
            raise MathValidationError("Stop loss multiplier must be positive")
        if not validate_range(self.transaction_cost, 0.0, 0.01):
            raise MathValidationError("Transaction cost must be between 0 and 0.01")

    def apply_fractional_differentiation(self, series: pd.Series) -> pd.Series:
        """
        Apply fractional differentiation to a time series.
        
        Args:
            series: Input time series
            
        Returns:
            Fractionally differentiated series
        """
        self.logger.info("🔢 Applying fractional differentiation...")
        
        try:
            # Calculate fractional differentiation weights
            weights = self._calculate_weights(len(series))
            
            # Apply fractional differentiation
            diff_series = pd.Series(index=series.index, dtype='float64')
            
            for i in range(len(series)):
                if i == 0:
                    diff_series.iloc[i] = series.iloc[i]
                else:
                    # Calculate weighted sum
                    weighted_sum = 0.0
                    for j in range(min(i + 1, len(weights))):
                        if i - j >= 0:
                            weighted_sum += weights[j] * series.iloc[i - j]
                    diff_series.iloc[i] = weighted_sum
            
            self.logger.info(f"✅ Fractional differentiation applied")
            return diff_series
            
        except Exception as e:
            self.logger.error(f"❌ Fractional differentiation failed: {e}")
            raise

    def _calculate_weights(self, length: int) -> np.ndarray:
        """Calculate fractional differentiation weights."""
        weights = np.zeros(length)
        weights[0] = 1.0
        
        for i in range(1, length):
            weights[i] = weights[i-1] * (i - 1 - self.d) / i
        
        return weights

    def create_fractional_labels(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create fractional triple barrier labels.
        
        Args:
            market_data: OHLCV market data
            
        Returns:
            DataFrame with fractional labels and metadata
        """
        self.logger.info("🏷️ Creating fractional triple barrier labels...")
        
        try:
            # Apply fractional differentiation to close prices
            frac_close = self.apply_fractional_differentiation(market_data['close'])
            
            # Calculate fractional returns
            frac_returns = frac_close.pct_change()
            
            # Create labels based on fractional returns
            labels = pd.Series(index=market_data.index, dtype='float64')
            
            # Apply thresholds
            pos_mask = frac_returns > self.profit_take_multiplier
            neg_mask = frac_returns < -self.stop_loss_multiplier
            mid_mask = (~pos_mask & ~neg_mask) & frac_returns.notna()
            
            labels[pos_mask] = 1.0  # Long signal
            labels[neg_mask] = -1.0  # Short signal
            labels[mid_mask] = 0.0   # No signal
            
            # Apply transaction cost adjustment
            if self.transaction_cost > 0:
                net_returns = frac_returns - self.transaction_cost
                pos_mask_net = net_returns > self.profit_take_multiplier
                neg_mask_net = net_returns < -self.stop_loss_multiplier
                
                labels[pos_mask_net] = 1.0
                labels[neg_mask_net] = -1.0
                labels[~pos_mask_net & ~neg_mask_net] = 0.0
            
            # Create result DataFrame
            result = pd.DataFrame({
                'label': labels,
                'fractional_close': frac_close,
                'fractional_returns': frac_returns,
                'original_close': market_data['close'],
                'original_returns': market_data['close'].pct_change(),
                'transaction_cost': self.transaction_cost,
                'profit_take_threshold': self.profit_take_multiplier,
                'stop_loss_threshold': self.stop_loss_multiplier,
                'differentiation_order': self.d
            }, index=market_data.index)
            
            self.logger.info(f"✅ Fractional triple barrier labels created: {len(labels.dropna())} valid labels")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Fractional triple barrier labeling failed: {e}")
            raise

    def get_fractional_statistics(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive fractional labeling statistics.
        
        Args:
            market_data: OHLCV market data
            
        Returns:
            Dictionary with fractional labeling statistics
        """
        try:
            # Create fractional labels
            fractional_data = self.create_fractional_labels(market_data)
            
            # Calculate statistics
            label_distribution = fractional_data['label'].value_counts().to_dict()
            
            # Compare original vs fractional returns
            original_returns = fractional_data['original_returns']
            fractional_returns = fractional_data['fractional_returns']
            
            comparison_stats = {
                'original_returns_mean': original_returns.mean(),
                'original_returns_std': original_returns.std(),
                'fractional_returns_mean': fractional_returns.mean(),
                'fractional_returns_std': fractional_returns.std(),
                'correlation': original_returns.corr(fractional_returns)
            }
            
            return {
                'label_distribution': label_distribution,
                'comparison_statistics': comparison_stats,
                'total_labels': len(fractional_data),
                'parameters': {
                    'differentiation_order': self.d,
                    'stationarity_threshold': self.threshold,
                    'profit_take_multiplier': self.profit_take_multiplier,
                    'stop_loss_multiplier': self.stop_loss_multiplier,
                    'transaction_cost': self.transaction_cost
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Fractional statistics calculation failed: {e}")
            raise