"""Labeling strategy component for tactician labeling."""
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger
from copy import copy

class LabelingStrategy:
    """Handles different labeling strategies for tactician models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the labeling strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('tactician_labeling', {})
        self.logger = system_logger.getChild('labeling_strategy')
        self.max_lookahead = self.config.get('max_lookahead', 50)
        self.binary_classification = self.config.get('binary_classification', True)
        self.multi_outcome_classes = self.config.get('multi_outcome_classes', 3)
        self.require_analyst_signal = self.config.get('require_analyst_signal', True)
        self.analyst_weight = self.config.get('analyst_weight', 0.3)
        self.direction_agreement_boost = self.config.get('direction_agreement_boost', 0.2)

    @handles_errors(exceptions=(Exception,), default_return=pd.DataFrame(), context='triple barrier labeling')
    async def apply_triple_barrier_labeling(self, data: pd.DataFrame, barriers: Dict[str, Tuple[float, float]], analyst_predictions: pd.DataFrame, precision_thresholds: Dict[str, float], regime_id: str) -> pd.DataFrame:
        """Apply triple barrier labeling strategy.
        
        Args:
            data: Market data
            barriers: Dictionary of barrier configurations
            analyst_predictions: Predictions from analyst models
            precision_thresholds: Precision thresholds for filtering
            regime_id: Regime identifier
            
        Returns:
            DataFrame with labels
        """
        self.logger.info(f'Applying triple barrier labeling for regime {regime_id}')
        labeled_data = data.copy()
        n_samples = len(labeled_data)
        labeled_data['label'] = 0
        labeled_data['barrier_type'] = 'none'
        labeled_data['exit_time'] = 0
        labeled_data['potential_profit_pct'] = 0.0
        labeled_data['signal_strength'] = 0.0
        if not analyst_predictions.empty:
            labeled_data = labeled_data.join(analyst_predictions, how='left')
            labeled_data['analyst_signal'] = labeled_data['analyst_signal'].fillna(0.5)
            labeled_data['analyst_prediction'] = labeled_data['analyst_prediction'].fillna(0)
        else:
            labeled_data['analyst_signal'] = 0.5
            labeled_data['analyst_prediction'] = 0
        all_labels = []
        for barrier_name, (upper_barrier, lower_barrier) in barriers.items():
            self.logger.info(f"  Processing barrier '{barrier_name}': upper={upper_barrier:.4f}, lower={lower_barrier:.4f}")
            barrier_labels = await self._apply_single_barrier(labeled_data, upper_barrier, lower_barrier, barrier_name, precision_thresholds)
            all_labels.append(barrier_labels)
        if all_labels:
            combined_labels = self._combine_barrier_labels(all_labels, labeled_data)
            for col in ['label', 'barrier_type', 'exit_time', 'potential_profit_pct', 'signal_strength']:
                if col in combined_labels.columns:
                    labeled_data[col] = combined_labels[col]
        labeled_data = self._apply_precision_filtering(labeled_data, precision_thresholds)
        if self.binary_classification and 'label' in labeled_data.columns:
            labeled_data['label'] = (labeled_data['label'] > 0).astype(int)
        return labeled_data

    async def _apply_single_barrier(self, data: pd.DataFrame, upper_barrier: float, lower_barrier: float, barrier_name: str, precision_thresholds: Dict[str, float]) -> pd.DataFrame:
        """Apply a single barrier configuration.
        
        Args:
            data: Market data with analyst signals
            upper_barrier: Upper barrier threshold
            lower_barrier: Lower barrier threshold
            barrier_name: Name of the barrier configuration
            precision_thresholds: Precision thresholds
            
        Returns:
            DataFrame with barrier labels
        """
        labels = pd.DataFrame(index=data.index)
        labels['label'] = 0
        labels['barrier_type'] = barrier_name
        labels['exit_time'] = 0
        labels['potential_profit_pct'] = 0.0
        labels['signal_strength'] = 0.0
        if 'close' not in data.columns:
            self.logger.warning("No 'close' price column found")
            return labels
        prices = data['close'].values
        n_samples = len(prices)
        for i in range(n_samples - 1):
            if self.require_analyst_signal and data.iloc[i]['analyst_signal'] < 0.5:
                continue
            signal_strength = self._calculate_signal_strength(data.iloc[i], precision_thresholds)
            if signal_strength < precision_thresholds.get('min_signal_strength', 0.8):
                continue
            current_price = prices[i]
            max_lookahead = min(self.max_lookahead, n_samples - i - 1)
            for j in range(1, max_lookahead + 1):
                future_price = prices[i + j]
                price_change = (future_price - current_price) / current_price
                if price_change >= upper_barrier:
                    labels.iloc[i] = {'label': 1, 'barrier_type': barrier_name, 'exit_time': j, 'potential_profit_pct': price_change * 100, 'signal_strength': signal_strength}
                    break
                elif price_change <= -lower_barrier:
                    labels.iloc[i] = {'label': -1, 'barrier_type': barrier_name, 'exit_time': j, 'potential_profit_pct': price_change * 100, 'signal_strength': signal_strength}
                    break
                elif j == max_lookahead:
                    if price_change > 0:
                        label = 1 if price_change > upper_barrier * 0.5 else 0
                    else:
                        label = -1 if price_change < -lower_barrier * 0.5 else 0
                    labels.iloc[i] = {'label': label, 'barrier_type': barrier_name, 'exit_time': j, 'potential_profit_pct': price_change * 100, 'signal_strength': signal_strength}
        return labels

    def _calculate_signal_strength(self, row: pd.Series, precision_thresholds: Dict[str, float]) -> float:
        """Calculate signal strength based on analyst predictions and other factors.
        
        Args:
            row: Data row
            precision_thresholds: Precision thresholds
            
        Returns:
            Signal strength score
        """
        strength = 0.0
        if 'analyst_signal' in row:
            strength = float(row['analyst_signal'])
        if 'analyst_prediction' in row and row['analyst_prediction'] == 1:
            strength += self.direction_agreement_boost
        strength = min(1.0, max(0.0, strength))
        return strength

    def _combine_barrier_labels(self, all_labels: List[pd.DataFrame], original_data: pd.DataFrame) -> pd.DataFrame:
        """Combine labels from multiple barrier configurations.
        
        Args:
            all_labels: List of label DataFrames from different barriers
            original_data: Original market data
            
        Returns:
            Combined labels DataFrame
        """
        if not all_labels:
            return pd.DataFrame(index=original_data.index)
        combined = all_labels[0].copy()
        for labels in all_labels[1:]:
            no_signal_mask = combined['label'] == 0
            for col in ['label', 'barrier_type', 'exit_time', 'potential_profit_pct', 'signal_strength']:
                if col in labels.columns:
                    combined.loc[no_signal_mask, col] = labels.loc[no_signal_mask, col]
            has_signal_mask = ~no_signal_mask & (labels['label'] != 0)
            better_signal_mask = has_signal_mask & (labels['signal_strength'] > combined['signal_strength'])
            for col in ['label', 'barrier_type', 'exit_time', 'potential_profit_pct', 'signal_strength']:
                if col in labels.columns:
                    combined.loc[better_signal_mask, col] = labels.loc[better_signal_mask, col]
        return combined

    def _apply_precision_filtering(self, labeled_data: pd.DataFrame, precision_thresholds: Dict[str, float]) -> pd.DataFrame:
        """Apply precision-based filtering to labels.
        
        Args:
            labeled_data: Data with labels
            precision_thresholds: Precision thresholds
            
        Returns:
            Filtered labeled data
        """
        min_strength = precision_thresholds.get('min_signal_strength', 0.8)
        low_strength_mask = labeled_data['signal_strength'] < min_strength
        labeled_data.loc[low_strength_mask, 'label'] = 0
        labeled_data.loc[low_strength_mask, 'barrier_type'] = 'filtered'
        return labeled_data

    @handles_errors(exceptions=(Exception,), default_return=pd.DataFrame(), context='multi-outcome labeling')
    async def apply_multi_outcome_labeling(self, data: pd.DataFrame, barriers: Dict[str, Tuple[float, float]], analyst_predictions: pd.DataFrame, regime_id: str) -> pd.DataFrame:
        """Apply multi-outcome labeling strategy.
        
        Args:
            data: Market data
            barriers: Dictionary of barrier configurations
            analyst_predictions: Predictions from analyst models
            regime_id: Regime identifier
            
        Returns:
            DataFrame with multi-class labels
        """
        self.logger.info(f'Applying multi-outcome labeling for regime {regime_id}')
        labeled_data = await self.apply_triple_barrier_labeling(data, barriers, analyst_predictions, {}, regime_id)
        if 'potential_profit_pct' in labeled_data.columns:
            profit_bins = [-np.inf, -2.0, -0.5, 0.5, 2.0, np.inf]
            profit_labels = [0, 1, 2, 3, 4]
            labeled_data['multi_label'] = pd.cut(labeled_data['potential_profit_pct'], bins=profit_bins, labels=profit_labels)
        return labeled_data