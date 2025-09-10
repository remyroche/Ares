"""
Optimized Triple Barrier Labeling - Moved to Utilities

This module contains the original step06 optimized triple barrier labeling functionality
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

class OptimizedTripleBarrierLabeling:
    """
    Optimized triple barrier labeling with enhanced financial parameters and transaction cost modeling.
    This is the original step06 functionality now available as utilities.
    """
    
    def __init__(self, profit_take_multiplier: float = 0.004,
                 stop_loss_multiplier: float = 0.003,
                 transaction_cost: float = 0.0008,
                 time_barrier_minutes: int = 30,
                 max_lookahead: int = 100):
        """
        Initialize optimized triple barrier labeling.
        
        Args:
            profit_take_multiplier: Profit take threshold
            stop_loss_multiplier: Stop loss threshold
            transaction_cost: Transaction cost
            time_barrier_minutes: Time barrier in minutes
            max_lookahead: Maximum lookahead for barrier calculation
        """
        self.profit_take_multiplier = profit_take_multiplier
        self.stop_loss_multiplier = stop_loss_multiplier
        self.transaction_cost = transaction_cost
        self.time_barrier_minutes = time_barrier_minutes
        self.max_lookahead = max_lookahead
        self.logger = logger
        
        # Validate parameters
        self._validate_parameters()
        
        self.logger.info("🏷️ Optimized Triple Barrier Labeling (Step06 Utilities) initialized")
        self.logger.info(f"   Profit take: {profit_take_multiplier:.4f}")
        self.logger.info(f"   Stop loss: {stop_loss_multiplier:.4f}")
        self.logger.info(f"   Transaction cost: {transaction_cost:.4f}")
        self.logger.info(f"   Time barrier: {time_barrier_minutes} minutes")
        self.logger.info(f"   Max lookahead: {max_lookahead}")

    def _validate_parameters(self) -> None:
        """Validate labeling parameters."""
        if not validate_positive(self.profit_take_multiplier):
            raise MathValidationError("Profit take multiplier must be positive")
        if not validate_positive(self.stop_loss_multiplier):
            raise MathValidationError("Stop loss multiplier must be positive")
        if not validate_range(self.transaction_cost, 0.0, 0.01):
            raise MathValidationError("Transaction cost must be between 0 and 0.01")
        if not validate_positive(self.time_barrier_minutes):
            raise MathValidationError("Time barrier must be positive")
        if not validate_positive(self.max_lookahead):
            raise MathValidationError("Max lookahead must be positive")

    def create_labels(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Create triple barrier labels for trading signals.
        
        Args:
            market_data: OHLCV market data
            
        Returns:
            Series with trading labels
        """
        self.logger.info("🏷️ Creating optimized triple barrier labels...")
        
        try:
            # Calculate returns
            returns = market_data['close'].pct_change()
            
            # Create labels based on returns
            labels = pd.Series(index=market_data.index, dtype='float64')
            
            # Apply thresholds
            pos_mask = returns > self.profit_take_multiplier
            neg_mask = returns < -self.stop_loss_multiplier
            mid_mask = (~pos_mask & ~neg_mask) & returns.notna()
            
            labels[pos_mask] = 1.0  # Long signal
            labels[neg_mask] = -1.0  # Short signal
            labels[mid_mask] = 0.0   # No signal
            
            # Apply transaction cost adjustment
            if self.transaction_cost > 0:
                # Adjust labels based on transaction costs
                net_returns = returns - self.transaction_cost
                pos_mask_net = net_returns > self.profit_take_multiplier
                neg_mask_net = net_returns < -self.stop_loss_multiplier
                
                # Update labels with transaction cost consideration
                labels[pos_mask_net] = 1.0
                labels[neg_mask_net] = -1.0
                labels[~pos_mask_net & ~neg_mask_net] = 0.0
            
            self.logger.info(f"✅ Optimized triple barrier labels created: {len(labels.dropna())} valid labels")
            return labels
            
        except Exception as e:
            self.logger.error(f"❌ Optimized triple barrier labeling failed: {e}")
            raise

    def apply_triple_barrier_labeling_vectorized(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply triple barrier labeling using vectorized operations for better performance.
        
        Args:
            market_data: OHLCV market data
            
        Returns:
            DataFrame with labels and metadata
        """
        self.logger.info("🏷️ Applying vectorized triple barrier labeling...")
        
        try:
            # Calculate returns
            returns = market_data['close'].pct_change()
            
            # Vectorized label creation
            labels = pd.Series(index=market_data.index, dtype='float64')
            
            # Apply thresholds vectorized
            pos_mask = returns > self.profit_take_multiplier
            neg_mask = returns < -self.stop_loss_multiplier
            mid_mask = (~pos_mask & ~neg_mask) & returns.notna()
            
            labels[pos_mask] = 1.0
            labels[neg_mask] = -1.0
            labels[mid_mask] = 0.0
            
            # Calculate profit metrics
            potential_profit = returns.abs()
            net_profit = returns - self.transaction_cost
            
            # Create result DataFrame
            result = pd.DataFrame({
                'label': labels,
                'returns': returns,
                'potential_profit_pct': potential_profit,
                'net_profit_pct': net_profit,
                'transaction_cost': self.transaction_cost,
                'profit_take_threshold': self.profit_take_multiplier,
                'stop_loss_threshold': self.stop_loss_multiplier
            }, index=market_data.index)
            
            # Calculate statistics
            label_distribution = labels.value_counts().to_dict()
            profit_stats = {
                'mean_profit': potential_profit.mean(),
                'std_profit': potential_profit.std(),
                'min_profit': potential_profit.min(),
                'max_profit': potential_profit.max()
            }
            
            self.logger.info(f"✅ Vectorized triple barrier labeling completed")
            self.logger.info(f"   Labels generated: {len(result)}")
            self.logger.info(f"   Label distribution: {label_distribution}")
            self.logger.info(f"   Mean profit: {profit_stats['mean_profit']:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Vectorized triple barrier labeling failed: {e}")
            raise

    def create_labels_with_metadata(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create triple barrier labels with additional metadata.
        
        Args:
            market_data: OHLCV market data
            
        Returns:
            DataFrame with labels and metadata
        """
        self.logger.info("🏷️ Creating triple barrier labels with metadata...")
        
        try:
            # Calculate returns
            returns = market_data['close'].pct_change()
            
            # Create labels
            labels = self.create_labels(market_data)
            
            # Calculate additional metadata
            potential_profit = returns.abs()
            net_profit = returns - self.transaction_cost
            
            # Create result DataFrame
            result = pd.DataFrame({
                'label': labels,
                'returns': returns,
                'potential_profit_pct': potential_profit,
                'net_profit_pct': net_profit,
                'transaction_cost': self.transaction_cost,
                'profit_take_threshold': self.profit_take_multiplier,
                'stop_loss_threshold': self.stop_loss_multiplier
            }, index=market_data.index)
            
            self.logger.info(f"✅ Triple barrier labels with metadata created")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Triple barrier labeling with metadata failed: {e}")
            raise

    def calculate_net_profit_after_costs(self, returns: pd.Series, labels: pd.Series) -> pd.Series:
        """
        Calculate net profit after transaction costs.
        
        Args:
            returns: Price returns
            labels: Trading labels
            
        Returns:
            Series with net profit after costs
        """
        try:
            # Calculate gross profit
            gross_profit = returns * labels
            
            # Apply transaction costs
            net_profit = gross_profit - self.transaction_cost
            
            return net_profit
            
        except Exception as e:
            self.logger.error(f"❌ Net profit calculation failed: {e}")
            raise

    def get_labeling_statistics(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive labeling statistics.
        
        Args:
            market_data: OHLCV market data
            
        Returns:
            Dictionary with labeling statistics
        """
        try:
            # Create labels
            labeled_data = self.apply_triple_barrier_labeling_vectorized(market_data)
            
            # Calculate statistics
            label_distribution = labeled_data['label'].value_counts().to_dict()
            profit_stats = {
                'mean_profit': labeled_data['potential_profit_pct'].mean(),
                'std_profit': labeled_data['potential_profit_pct'].std(),
                'min_profit': labeled_data['potential_profit_pct'].min(),
                'max_profit': labeled_data['potential_profit_pct'].max()
            }
            
            # Calculate net profit after transaction costs
            long_profits = labeled_data[labeled_data['label'] == 1]['potential_profit_pct']
            short_profits = labeled_data[labeled_data['label'] == -1]['potential_profit_pct']
            
            net_profit_stats = {
                'long_mean_net_profit': long_profits.mean() if len(long_profits) > 0 else 0.0,
                'short_mean_net_profit': short_profits.mean() if len(short_profits) > 0 else 0.0,
                'overall_net_profit': labeled_data['potential_profit_pct'].mean()
            }
            
            return {
                'label_distribution': label_distribution,
                'profit_statistics': profit_stats,
                'net_profit_statistics': net_profit_stats,
                'total_labels': len(labeled_data),
                'parameters': {
                    'profit_take_multiplier': self.profit_take_multiplier,
                    'stop_loss_multiplier': self.stop_loss_multiplier,
                    'transaction_cost': self.transaction_cost,
                    'time_barrier_minutes': self.time_barrier_minutes
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Labeling statistics calculation failed: {e}")
            raise