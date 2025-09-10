"""
Labeling Utilities

This module provides comprehensive labeling utilities that were previously
part of step06. These utilities can be used by any step in the pipeline that needs
advanced labeling capabilities.

Features include:
- Triple barrier labeling
- Meta-labeling
- Regime-aware labeling
- Fractional differentiation
- Profit-based feature engineering
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

class TripleBarrierLabeling:
    """
    Triple barrier labeling utility for creating trading signals.
    """
    
    def __init__(self, profit_take_multiplier: float = 0.004,
                 stop_loss_multiplier: float = 0.003,
                 transaction_cost: float = 0.0008,
                 time_barrier_minutes: int = 30):
        """
        Initialize triple barrier labeling.
        
        Args:
            profit_take_multiplier: Profit take threshold
            stop_loss_multiplier: Stop loss threshold
            transaction_cost: Transaction cost
            time_barrier_minutes: Time barrier in minutes
        """
        self.profit_take_multiplier = profit_take_multiplier
        self.stop_loss_multiplier = stop_loss_multiplier
        self.transaction_cost = transaction_cost
        self.time_barrier_minutes = time_barrier_minutes
        self.logger = logger
        
        # Validate parameters
        self._validate_parameters()
        
        self.logger.info("🏷️ Triple Barrier Labeling initialized")
        self.logger.info(f"   Profit take: {profit_take_multiplier:.4f}")
        self.logger.info(f"   Stop loss: {stop_loss_multiplier:.4f}")
        self.logger.info(f"   Transaction cost: {transaction_cost:.4f}")
        self.logger.info(f"   Time barrier: {time_barrier_minutes} minutes")

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

    def create_labels(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Create triple barrier labels for trading signals.
        
        Args:
            market_data: OHLCV market data
            
        Returns:
            Series with trading labels
        """
        self.logger.info("🏷️ Creating triple barrier labels...")
        
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
            
            self.logger.info(f"✅ Triple barrier labels created: {len(labels.dropna())} valid labels")
            return labels
            
        except Exception as e:
            self.logger.error(f"❌ Triple barrier labeling failed: {e}")
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


class MetaLabeling:
    """
    Meta-labeling utility for creating secondary labels based on primary signals.
    """
    
    def __init__(self, confidence_threshold: float = 0.6,
                 min_samples_per_class: int = 100):
        """
        Initialize meta-labeling.
        
        Args:
            confidence_threshold: Minimum confidence threshold
            min_samples_per_class: Minimum samples per class
        """
        self.confidence_threshold = confidence_threshold
        self.min_samples_per_class = min_samples_per_class
        self.logger = logger
        
        # Validate parameters
        self._validate_parameters()
        
        self.logger.info("🏷️ Meta-Labeling initialized")
        self.logger.info(f"   Confidence threshold: {confidence_threshold}")
        self.logger.info(f"   Min samples per class: {min_samples_per_class}")

    def _validate_parameters(self) -> None:
        """Validate meta-labeling parameters."""
        if not validate_range(self.confidence_threshold, 0.0, 1.0):
            raise MathValidationError("Confidence threshold must be between 0 and 1")
        if not validate_positive(self.min_samples_per_class):
            raise MathValidationError("Min samples per class must be positive")

    def create_meta_labels(self, primary_labels: pd.Series, 
                          confidence_scores: pd.Series) -> pd.Series:
        """
        Create meta-labels based on primary labels and confidence scores.
        
        Args:
            primary_labels: Primary trading labels
            confidence_scores: Confidence scores for each label
            
        Returns:
            Series with meta-labels
        """
        self.logger.info("🏷️ Creating meta-labels...")
        
        try:
            # Initialize meta-labels
            meta_labels = pd.Series(index=primary_labels.index, dtype='float64')
            
            # Apply confidence threshold
            high_confidence_mask = confidence_scores >= self.confidence_threshold
            
            # Create meta-labels
            meta_labels[high_confidence_mask] = primary_labels[high_confidence_mask]
            meta_labels[~high_confidence_mask] = 0.0  # No signal for low confidence
            
            # Check class balance
            label_counts = meta_labels.value_counts()
            for label, count in label_counts.items():
                if count < self.min_samples_per_class:
                    self.logger.warning(f"⚠️ Class {label} has only {count} samples (minimum {self.min_samples_per_class})")
            
            self.logger.info(f"✅ Meta-labels created: {len(meta_labels.dropna())} valid labels")
            return meta_labels
            
        except Exception as e:
            self.logger.error(f"❌ Meta-labeling failed: {e}")
            raise


class RegimeAwareLabeling:
    """
    Regime-aware labeling utility that adjusts labels based on market regimes.
    """
    
    def __init__(self, regime_threshold: float = 0.7,
                 regime_specific_thresholds: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize regime-aware labeling.
        
        Args:
            regime_threshold: Minimum regime confidence threshold
            regime_specific_thresholds: Regime-specific labeling thresholds
        """
        self.regime_threshold = regime_threshold
        self.regime_specific_thresholds = regime_specific_thresholds or {}
        self.logger = logger
        
        # Validate parameters
        self._validate_parameters()
        
        self.logger.info("🏷️ Regime-Aware Labeling initialized")
        self.logger.info(f"   Regime threshold: {regime_threshold}")
        self.logger.info(f"   Regime-specific thresholds: {len(self.regime_specific_thresholds)} regimes")

    def _validate_parameters(self) -> None:
        """Validate regime-aware labeling parameters."""
        if not validate_range(self.regime_threshold, 0.0, 1.0):
            raise MathValidationError("Regime threshold must be between 0 and 1")

    def create_regime_aware_labels(self, market_data: pd.DataFrame,
                                 regime_labels: pd.Series,
                                 regime_confidence: pd.Series,
                                 base_labeling: TripleBarrierLabeling) -> pd.DataFrame:
        """
        Create regime-aware labels.
        
        Args:
            market_data: OHLCV market data
            regime_labels: Market regime labels
            regime_confidence: Regime confidence scores
            base_labeling: Base triple barrier labeling instance
            
        Returns:
            DataFrame with regime-aware labels
        """
        self.logger.info("🏷️ Creating regime-aware labels...")
        
        try:
            # Create base labels
            base_labels = base_labeling.create_labels_with_metadata(market_data)
            
            # Initialize regime-aware labels
            regime_aware_labels = base_labels.copy()
            
            # Apply regime-specific adjustments
            for regime, thresholds in self.regime_specific_thresholds.items():
                regime_mask = (regime_labels == regime) & (regime_confidence >= self.regime_threshold)
                
                if regime_mask.any():
                    # Adjust thresholds for this regime
                    profit_take = thresholds.get('profit_take_multiplier', base_labeling.profit_take_multiplier)
                    stop_loss = thresholds.get('stop_loss_multiplier', base_labeling.stop_loss_multiplier)
                    
                    # Recalculate labels for this regime
                    regime_returns = market_data.loc[regime_mask, 'close'].pct_change()
                    
                    # Apply regime-specific thresholds
                    pos_mask = regime_returns > profit_take
                    neg_mask = regime_returns < -stop_loss
                    mid_mask = (~pos_mask & ~neg_mask) & regime_returns.notna()
                    
                    # Update labels
                    regime_aware_labels.loc[regime_mask & pos_mask, 'label'] = 1.0
                    regime_aware_labels.loc[regime_mask & neg_mask, 'label'] = -1.0
                    regime_aware_labels.loc[regime_mask & mid_mask, 'label'] = 0.0
                    
                    # Update metadata
                    regime_aware_labels.loc[regime_mask, 'profit_take_threshold'] = profit_take
                    regime_aware_labels.loc[regime_mask, 'stop_loss_threshold'] = stop_loss
            
            # Add regime information
            regime_aware_labels['regime'] = regime_labels
            regime_aware_labels['regime_confidence'] = regime_confidence
            
            self.logger.info(f"✅ Regime-aware labels created")
            return regime_aware_labels
            
        except Exception as e:
            self.logger.error(f"❌ Regime-aware labeling failed: {e}")
            raise


class FractionalDifferentiation:
    """
    Fractional differentiation utility for creating stationary features.
    """
    
    def __init__(self, d: float = 0.5, threshold: float = 0.01):
        """
        Initialize fractional differentiation.
        
        Args:
            d: Fractional differentiation order
            threshold: Threshold for stationarity
        """
        self.d = d
        self.threshold = threshold
        self.logger = logger
        
        # Validate parameters
        self._validate_parameters()
        
        self.logger.info("🔢 Fractional Differentiation initialized")
        self.logger.info(f"   Differentiation order: {d}")
        self.logger.info(f"   Stationarity threshold: {threshold}")

    def _validate_parameters(self) -> None:
        """Validate fractional differentiation parameters."""
        if not validate_range(self.d, 0.0, 1.0):
            raise MathValidationError("Differentiation order must be between 0 and 1")
        if not validate_positive(self.threshold):
            raise MathValidationError("Stationarity threshold must be positive")

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


# Convenience functions for easy access
def create_triple_barrier_labeling(profit_take_multiplier: float = 0.004,
                                 stop_loss_multiplier: float = 0.003,
                                 transaction_cost: float = 0.0008,
                                 time_barrier_minutes: int = 30) -> TripleBarrierLabeling:
    """Create a new instance of TripleBarrierLabeling."""
    return TripleBarrierLabeling(profit_take_multiplier, stop_loss_multiplier, 
                                transaction_cost, time_barrier_minutes)

def create_meta_labeling(confidence_threshold: float = 0.6,
                        min_samples_per_class: int = 100) -> MetaLabeling:
    """Create a new instance of MetaLabeling."""
    return MetaLabeling(confidence_threshold, min_samples_per_class)

def create_regime_aware_labeling(regime_threshold: float = 0.7,
                               regime_specific_thresholds: Optional[Dict[str, Dict[str, float]]] = None) -> RegimeAwareLabeling:
    """Create a new instance of RegimeAwareLabeling."""
    return RegimeAwareLabeling(regime_threshold, regime_specific_thresholds)

def create_fractional_differentiation(d: float = 0.5, threshold: float = 0.01) -> FractionalDifferentiation:
    """Create a new instance of FractionalDifferentiation."""
    return FractionalDifferentiation(d, threshold)

def create_triple_barrier_labels(market_data: pd.DataFrame, 
                               profit_take_multiplier: float = 0.004,
                               stop_loss_multiplier: float = 0.003,
                               transaction_cost: float = 0.0008) -> pd.Series:
    """Convenience function to create triple barrier labels."""
    labeling = TripleBarrierLabeling(profit_take_multiplier, stop_loss_multiplier, transaction_cost)
    return labeling.create_labels(market_data)