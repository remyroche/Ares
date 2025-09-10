"""
Regime-Aware Triple Barrier Labeling - Moved to Utilities

This module contains the original step06 regime-aware triple barrier labeling functionality
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

class RegimeAwareTripleBarrierLabeling:
    """
    Regime-aware triple barrier labeling with regime-specific adjustments.
    This is the original step06 functionality now available as utilities.
    """
    
    def __init__(self, regime_threshold: float = 0.7,
                 regime_specific_params: Optional[Dict[str, Dict[str, float]]] = None,
                 base_profit_take: float = 0.004,
                 base_stop_loss: float = 0.003,
                 base_transaction_cost: float = 0.0008):
        """
        Initialize regime-aware triple barrier labeling.
        
        Args:
            regime_threshold: Minimum regime confidence threshold
            regime_specific_params: Regime-specific parameters
            base_profit_take: Base profit take threshold
            base_stop_loss: Base stop loss threshold
            base_transaction_cost: Base transaction cost
        """
        self.regime_threshold = regime_threshold
        self.regime_specific_params = regime_specific_params or {}
        self.base_profit_take = base_profit_take
        self.base_stop_loss = base_stop_loss
        self.base_transaction_cost = base_transaction_cost
        self.logger = logger
        
        # Validate parameters
        self._validate_parameters()
        
        self.logger.info("🏛️ Regime-Aware Triple Barrier Labeling (Step06 Utilities) initialized")
        self.logger.info(f"   Regime threshold: {regime_threshold}")
        self.logger.info(f"   Regime-specific params: {len(self.regime_specific_params)} regimes")
        self.logger.info(f"   Base profit take: {base_profit_take:.4f}")
        self.logger.info(f"   Base stop loss: {base_stop_loss:.4f}")

    def _validate_parameters(self) -> None:
        """Validate regime-aware labeling parameters."""
        if not validate_range(self.regime_threshold, 0.0, 1.0):
            raise MathValidationError("Regime threshold must be between 0 and 1")
        if not validate_positive(self.base_profit_take):
            raise MathValidationError("Base profit take must be positive")
        if not validate_positive(self.base_stop_loss):
            raise MathValidationError("Base stop loss must be positive")
        if not validate_range(self.base_transaction_cost, 0.0, 0.01):
            raise MathValidationError("Base transaction cost must be between 0 and 0.01")

    def create_regime_aware_labels(self, market_data: pd.DataFrame,
                                 regime_labels: pd.Series,
                                 regime_confidence: pd.Series) -> pd.DataFrame:
        """
        Create regime-aware triple barrier labels.
        
        Args:
            market_data: OHLCV market data
            regime_labels: Market regime labels
            regime_confidence: Regime confidence scores
            
        Returns:
            DataFrame with regime-aware labels
        """
        self.logger.info("🏷️ Creating regime-aware triple barrier labels...")
        
        try:
            # Initialize result DataFrame
            result = pd.DataFrame(index=market_data.index)
            result['regime'] = regime_labels
            result['regime_confidence'] = regime_confidence
            
            # Create labels for each regime
            all_labels = []
            
            for regime in regime_labels.unique():
                if pd.isna(regime):
                    continue
                    
                # Filter data for this regime
                regime_mask = (regime_labels == regime) & (regime_confidence >= self.regime_threshold)
                regime_data = market_data[regime_mask]
                
                if len(regime_data) < 10:  # Need minimum data points
                    self.logger.warning(f"⚠️ Insufficient data for regime {regime}: {len(regime_data)} points")
                    continue
                
                # Get regime-specific parameters
                regime_params = self.regime_specific_params.get(str(regime), {})
                
                # Create labels for this regime
                regime_labels_series = self._create_labels_for_regime(
                    regime_data, regime, regime_params
                )
                
                all_labels.append(regime_labels_series)
                
                # Add regime-specific metadata
                result.loc[regime_mask, 'profit_take_threshold'] = regime_params.get(
                    'profit_take_multiplier', self.base_profit_take
                )
                result.loc[regime_mask, 'stop_loss_threshold'] = regime_params.get(
                    'stop_loss_multiplier', self.base_stop_loss
                )
                result.loc[regime_mask, 'transaction_cost'] = regime_params.get(
                    'transaction_cost', self.base_transaction_cost
                )
            
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
            self.logger.info(f"   Regimes used: {len(regime_labels.unique())}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Regime-aware labeling failed: {e}")
            raise

    def _create_labels_for_regime(self, regime_data: pd.DataFrame,
                                regime: str,
                                regime_params: Dict[str, float]) -> pd.Series:
        """
        Create labels for a specific regime.
        
        Args:
            regime_data: Market data for the regime
            regime: Regime identifier
            regime_params: Regime-specific parameters
            
        Returns:
            Series with labels for the regime
        """
        try:
            # Get regime-specific parameters
            profit_take = regime_params.get('profit_take_multiplier', self.base_profit_take)
            stop_loss = regime_params.get('stop_loss_multiplier', self.base_stop_loss)
            transaction_cost = regime_params.get('transaction_cost', self.base_transaction_cost)
            
            # Calculate returns for this regime
            regime_returns = regime_data['close'].pct_change()
            
            # Create labels based on regime-specific thresholds
            labels = pd.Series(index=regime_data.index, dtype='float64')
            
            # Apply thresholds
            pos_mask = regime_returns > profit_take
            neg_mask = regime_returns < -stop_loss
            mid_mask = (~pos_mask & ~neg_mask) & regime_returns.notna()
            
            labels[pos_mask] = 1.0  # Long signal
            labels[neg_mask] = -1.0  # Short signal
            labels[mid_mask] = 0.0   # No signal
            
            # Apply transaction cost adjustment
            if transaction_cost > 0:
                net_returns = regime_returns - transaction_cost
                pos_mask_net = net_returns > profit_take
                neg_mask_net = net_returns < -stop_loss
                
                labels[pos_mask_net] = 1.0
                labels[neg_mask_net] = -1.0
                labels[~pos_mask_net & ~neg_mask_net] = 0.0
            
            return labels
            
        except Exception as e:
            self.logger.error(f"❌ Label creation for regime {regime} failed: {e}")
            # Return neutral labels as fallback
            return pd.Series(0.0, index=regime_data.index)

    def optimize_regime_parameters(self, market_data: pd.DataFrame,
                                 regime_labels: pd.Series,
                                 regime_confidence: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Optimize parameters for each regime based on historical performance.
        
        Args:
            market_data: OHLCV market data
            regime_labels: Market regime labels
            regime_confidence: Regime confidence scores
            
        Returns:
            Dictionary with optimized parameters for each regime
        """
        self.logger.info("🔧 Optimizing regime-specific parameters...")
        
        try:
            optimized_params = {}
            
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
                
                # Optimize parameters for this regime
                regime_params = self._optimize_parameters_for_regime(regime_data, regime)
                optimized_params[str(regime)] = regime_params
                
                self.logger.info(f"✅ Optimized parameters for regime {regime}")
                self.logger.info(f"   Profit take: {regime_params['profit_take_multiplier']:.4f}")
                self.logger.info(f"   Stop loss: {regime_params['stop_loss_multiplier']:.4f}")
            
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"❌ Regime parameter optimization failed: {e}")
            raise

    def _optimize_parameters_for_regime(self, regime_data: pd.DataFrame, regime: str) -> Dict[str, float]:
        """
        Optimize parameters for a specific regime.
        
        Args:
            regime_data: Market data for the regime
            regime: Regime identifier
            
        Returns:
            Dictionary with optimized parameters
        """
        try:
            # Calculate regime statistics
            regime_returns = regime_data['close'].pct_change()
            regime_volatility = regime_returns.std()
            regime_mean = regime_returns.mean()
            regime_skewness = regime_returns.skew()
            
            # Adjust parameters based on regime characteristics
            if regime_volatility > 0.02:  # High volatility regime
                profit_take = self.base_profit_take * 1.5
                stop_loss = self.base_stop_loss * 1.2
                transaction_cost = self.base_transaction_cost * 1.1
            elif regime_volatility < 0.005:  # Low volatility regime
                profit_take = self.base_profit_take * 0.7
                stop_loss = self.base_stop_loss * 0.8
                transaction_cost = self.base_transaction_cost * 0.9
            else:  # Normal volatility regime
                profit_take = self.base_profit_take
                stop_loss = self.base_stop_loss
                transaction_cost = self.base_transaction_cost
            
            # Adjust for regime skewness
            if regime_skewness > 0.5:  # Positive skew (more upside potential)
                profit_take *= 1.2
            elif regime_skewness < -0.5:  # Negative skew (more downside risk)
                stop_loss *= 1.2
            
            # Ensure parameters are within reasonable bounds
            profit_take = np.clip(profit_take, 0.001, 0.01)
            stop_loss = np.clip(stop_loss, 0.001, 0.01)
            transaction_cost = np.clip(transaction_cost, 0.0001, 0.005)
            
            return {
                'profit_take_multiplier': profit_take,
                'stop_loss_multiplier': stop_loss,
                'transaction_cost': transaction_cost,
                'regime_volatility': regime_volatility,
                'regime_mean': regime_mean,
                'regime_skewness': regime_skewness
            }
            
        except Exception as e:
            self.logger.error(f"❌ Parameter optimization for regime {regime} failed: {e}")
            # Return base parameters as fallback
            return {
                'profit_take_multiplier': self.base_profit_take,
                'stop_loss_multiplier': self.base_stop_loss,
                'transaction_cost': self.base_transaction_cost,
                'regime_volatility': 0.0,
                'regime_mean': 0.0,
                'regime_skewness': 0.0
            }

    def get_regime_aware_statistics(self, market_data: pd.DataFrame,
                                  regime_labels: pd.Series,
                                  regime_confidence: pd.Series) -> Dict[str, Any]:
        """
        Get comprehensive regime-aware labeling statistics.
        
        Args:
            market_data: OHLCV market data
            regime_labels: Market regime labels
            regime_confidence: Regime confidence scores
            
        Returns:
            Dictionary with regime-aware labeling statistics
        """
        try:
            # Optimize parameters
            optimized_params = self.optimize_regime_parameters(
                market_data, regime_labels, regime_confidence
            )
            
            # Create regime-aware labels
            regime_aware_data = self.create_regime_aware_labels(
                market_data, regime_labels, regime_confidence
            )
            
            # Calculate statistics
            label_distribution = regime_aware_data['label'].value_counts().to_dict()
            regime_distribution = regime_aware_data['regime'].value_counts().to_dict()
            
            # Calculate performance by regime
            regime_performance = {}
            for regime in optimized_params.keys():
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
                'optimized_parameters': optimized_params,
                'total_samples': len(regime_aware_data),
                'parameters': {
                    'regime_threshold': self.regime_threshold,
                    'base_profit_take': self.base_profit_take,
                    'base_stop_loss': self.base_stop_loss,
                    'base_transaction_cost': self.base_transaction_cost
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Regime-aware statistics calculation failed: {e}")
            raise