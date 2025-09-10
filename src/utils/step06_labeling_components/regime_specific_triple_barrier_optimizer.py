"""
Regime Specific Triple Barrier Optimizer - Moved to Utilities

This module contains the original step06 regime-specific triple barrier optimization functionality
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

class RegimeSpecificTripleBarrierOptimizer:
    """
    Regime-specific triple barrier optimizer for different market conditions.
    This is the original step06 functionality now available as utilities.
    """
    
    def __init__(self, regime_threshold: float = 0.7,
                 regime_specific_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
                 base_profit_take: float = 0.004,
                 base_stop_loss: float = 0.003,
                 base_transaction_cost: float = 0.0008):
        """
        Initialize regime-specific triple barrier optimizer.
        
        Args:
            regime_threshold: Minimum regime confidence threshold
            regime_specific_thresholds: Regime-specific labeling thresholds
            base_profit_take: Base profit take threshold
            base_stop_loss: Base stop loss threshold
            base_transaction_cost: Base transaction cost
        """
        self.regime_threshold = regime_threshold
        self.regime_specific_thresholds = regime_specific_thresholds or {}
        self.base_profit_take = base_profit_take
        self.base_stop_loss = base_stop_loss
        self.base_transaction_cost = base_transaction_cost
        self.logger = logger
        
        # Validate parameters
        self._validate_parameters()
        
        self.logger.info("🏛️ Regime-Specific Triple Barrier Optimizer (Step06 Utilities) initialized")
        self.logger.info(f"   Regime threshold: {regime_threshold}")
        self.logger.info(f"   Regime-specific thresholds: {len(self.regime_specific_thresholds)} regimes")
        self.logger.info(f"   Base profit take: {base_profit_take:.4f}")
        self.logger.info(f"   Base stop loss: {base_stop_loss:.4f}")

    def _validate_parameters(self) -> None:
        """Validate regime-specific optimizer parameters."""
        if not validate_range(self.regime_threshold, 0.0, 1.0):
            raise MathValidationError("Regime threshold must be between 0 and 1")
        if not validate_positive(self.base_profit_take):
            raise MathValidationError("Base profit take must be positive")
        if not validate_positive(self.base_stop_loss):
            raise MathValidationError("Base stop loss must be positive")
        if not validate_range(self.base_transaction_cost, 0.0, 0.01):
            raise MathValidationError("Base transaction cost must be between 0 and 0.01")

    def optimize_regime_thresholds(self, market_data: pd.DataFrame,
                                 regime_labels: pd.Series,
                                 regime_confidence: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Optimize thresholds for each regime based on historical performance.
        
        Args:
            market_data: OHLCV market data
            regime_labels: Market regime labels
            regime_confidence: Regime confidence scores
            
        Returns:
            Dictionary with optimized thresholds for each regime
        """
        self.logger.info("🔧 Optimizing regime-specific thresholds...")
        
        try:
            optimized_thresholds = {}
            
            # Get unique regimes
            unique_regimes = regime_labels.unique()
            
            for regime in unique_regimes:
                if pd.isna(regime):
                    continue
                    
                # Filter data for this regime
                regime_mask = (regime_labels == regime) & (regime_confidence >= self.regime_threshold)
                regime_data = market_data[regime_mask]
                
                if len(regime_data) < 50:  # Need minimum data points
                    self.logger.warning(f"⚠️ Insufficient data for regime {regime}: {len(regime_data)} points")
                    continue
                
                # Calculate returns for this regime
                regime_returns = regime_data['close'].pct_change()
                
                # Optimize thresholds based on regime characteristics
                regime_thresholds = self._optimize_thresholds_for_regime(regime_returns, regime)
                optimized_thresholds[str(regime)] = regime_thresholds
                
                self.logger.info(f"✅ Optimized thresholds for regime {regime}")
                self.logger.info(f"   Profit take: {regime_thresholds['profit_take_multiplier']:.4f}")
                self.logger.info(f"   Stop loss: {regime_thresholds['stop_loss_multiplier']:.4f}")
            
            return optimized_thresholds
            
        except Exception as e:
            self.logger.error(f"❌ Regime threshold optimization failed: {e}")
            raise

    def _optimize_thresholds_for_regime(self, returns: pd.Series, regime: str) -> Dict[str, float]:
        """
        Optimize thresholds for a specific regime.
        
        Args:
            returns: Returns for the regime
            regime: Regime identifier
            
        Returns:
            Dictionary with optimized thresholds
        """
        try:
            # Calculate regime statistics
            regime_volatility = returns.std()
            regime_mean = returns.mean()
            regime_skewness = returns.skew()
            
            # Adjust thresholds based on regime characteristics
            if regime_volatility > 0.02:  # High volatility regime
                profit_take = self.base_profit_take * 1.5
                stop_loss = self.base_stop_loss * 1.2
            elif regime_volatility < 0.005:  # Low volatility regime
                profit_take = self.base_profit_take * 0.7
                stop_loss = self.base_stop_loss * 0.8
            else:  # Normal volatility regime
                profit_take = self.base_profit_take
                stop_loss = self.base_stop_loss
            
            # Adjust for regime skewness
            if regime_skewness > 0.5:  # Positive skew (more upside potential)
                profit_take *= 1.2
            elif regime_skewness < -0.5:  # Negative skew (more downside risk)
                stop_loss *= 1.2
            
            # Ensure thresholds are within reasonable bounds
            profit_take = np.clip(profit_take, 0.001, 0.01)
            stop_loss = np.clip(stop_loss, 0.001, 0.01)
            
            return {
                'profit_take_multiplier': profit_take,
                'stop_loss_multiplier': stop_loss,
                'transaction_cost': self.base_transaction_cost,
                'regime_volatility': regime_volatility,
                'regime_mean': regime_mean,
                'regime_skewness': regime_skewness
            }
            
        except Exception as e:
            self.logger.error(f"❌ Threshold optimization for regime {regime} failed: {e}")
            # Return base thresholds as fallback
            return {
                'profit_take_multiplier': self.base_profit_take,
                'stop_loss_multiplier': self.base_stop_loss,
                'transaction_cost': self.base_transaction_cost,
                'regime_volatility': 0.0,
                'regime_mean': 0.0,
                'regime_skewness': 0.0
            }

    def create_regime_aware_labels(self, market_data: pd.DataFrame,
                                 regime_labels: pd.Series,
                                 regime_confidence: pd.Series,
                                 optimized_thresholds: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
        """
        Create regime-aware labels using optimized thresholds.
        
        Args:
            market_data: OHLCV market data
            regime_labels: Market regime labels
            regime_confidence: Regime confidence scores
            optimized_thresholds: Optimized thresholds for each regime
            
        Returns:
            DataFrame with regime-aware labels
        """
        self.logger.info("🏷️ Creating regime-aware labels...")
        
        try:
            # Use provided thresholds or optimize new ones
            if optimized_thresholds is None:
                optimized_thresholds = self.optimize_regime_thresholds(
                    market_data, regime_labels, regime_confidence
                )
            
            # Initialize result DataFrame
            result = pd.DataFrame(index=market_data.index)
            result['regime'] = regime_labels
            result['regime_confidence'] = regime_confidence
            
            # Create labels for each regime
            all_labels = []
            
            for regime, thresholds in optimized_thresholds.items():
                regime_mask = (regime_labels == regime) & (regime_confidence >= self.regime_threshold)
                
                if regime_mask.any():
                    # Get regime data
                    regime_data = market_data[regime_mask]
                    regime_returns = regime_data['close'].pct_change()
                    
                    # Create labels for this regime
                    regime_labels_series = pd.Series(index=regime_data.index, dtype='float64')
                    
                    # Apply regime-specific thresholds
                    profit_take = thresholds['profit_take_multiplier']
                    stop_loss = thresholds['stop_loss_multiplier']
                    transaction_cost = thresholds['transaction_cost']
                    
                    pos_mask = regime_returns > profit_take
                    neg_mask = regime_returns < -stop_loss
                    mid_mask = (~pos_mask & ~neg_mask) & regime_returns.notna()
                    
                    regime_labels_series[pos_mask] = 1.0
                    regime_labels_series[neg_mask] = -1.0
                    regime_labels_series[mid_mask] = 0.0
                    
                    # Apply transaction cost adjustment
                    if transaction_cost > 0:
                        net_returns = regime_returns - transaction_cost
                        pos_mask_net = net_returns > profit_take
                        neg_mask_net = net_returns < -stop_loss
                        
                        regime_labels_series[pos_mask_net] = 1.0
                        regime_labels_series[neg_mask_net] = -1.0
                        regime_labels_series[~pos_mask_net & ~neg_mask_net] = 0.0
                    
                    all_labels.append(regime_labels_series)
                    
                    # Add regime-specific metadata
                    result.loc[regime_mask, 'profit_take_threshold'] = profit_take
                    result.loc[regime_mask, 'stop_loss_threshold'] = stop_loss
                    result.loc[regime_mask, 'transaction_cost'] = transaction_cost
                    result.loc[regime_mask, 'regime_volatility'] = thresholds.get('regime_volatility', 0.0)
            
            # Combine all labels
            if all_labels:
                combined_labels = pd.concat(all_labels).sort_index()
                result['label'] = combined_labels
            else:
                result['label'] = 0.0
            
            # Fill missing values
            result = result.fillna(0.0)
            
            self.logger.info(f"✅ Regime-aware labels created")
            self.logger.info(f"   Total labels: {len(result)}")
            self.logger.info(f"   Regimes used: {len(optimized_thresholds)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Regime-aware labeling failed: {e}")
            raise

    def get_regime_optimization_statistics(self, market_data: pd.DataFrame,
                                         regime_labels: pd.Series,
                                         regime_confidence: pd.Series) -> Dict[str, Any]:
        """
        Get comprehensive regime optimization statistics.
        
        Args:
            market_data: OHLCV market data
            regime_labels: Market regime labels
            regime_confidence: Regime confidence scores
            
        Returns:
            Dictionary with regime optimization statistics
        """
        try:
            # Optimize thresholds
            optimized_thresholds = self.optimize_regime_thresholds(
                market_data, regime_labels, regime_confidence
            )
            
            # Create regime-aware labels
            regime_aware_data = self.create_regime_aware_labels(
                market_data, regime_labels, regime_confidence, optimized_thresholds
            )
            
            # Calculate statistics
            label_distribution = regime_aware_data['label'].value_counts().to_dict()
            regime_distribution = regime_aware_data['regime'].value_counts().to_dict()
            
            # Calculate performance by regime
            regime_performance = {}
            for regime in optimized_thresholds.keys():
                regime_mask = regime_aware_data['regime'] == regime
                regime_data = regime_aware_data[regime_mask]
                
                if len(regime_data) > 0:
                    regime_performance[regime] = {
                        'total_samples': len(regime_data),
                        'positive_labels': (regime_data['label'] == 1.0).sum(),
                        'negative_labels': (regime_data['label'] == -1.0).sum(),
                        'neutral_labels': (regime_data['label'] == 0.0).sum(),
                        'avg_confidence': regime_data['regime_confidence'].mean(),
                        'profit_take_threshold': regime_data['profit_take_threshold'].iloc[0],
                        'stop_loss_threshold': regime_data['stop_loss_threshold'].iloc[0]
                    }
            
            return {
                'label_distribution': label_distribution,
                'regime_distribution': regime_distribution,
                'regime_performance': regime_performance,
                'optimized_thresholds': optimized_thresholds,
                'total_samples': len(regime_aware_data),
                'parameters': {
                    'regime_threshold': self.regime_threshold,
                    'base_profit_take': self.base_profit_take,
                    'base_stop_loss': self.base_stop_loss,
                    'base_transaction_cost': self.base_transaction_cost
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Regime optimization statistics calculation failed: {e}")
            raise