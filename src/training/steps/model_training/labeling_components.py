"""Labeling components for the labeling step.

This module contains specialized labeling components including
triple barrier labeling with regime awareness.
"""
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from src.utils.logger import system_logger
from copy import copy
import asyncio

class TripleBarrierLabeler:
    """Implements triple barrier labeling method."""

    def __init__(self, barrier_config: Dict[str, Any]) -> None:
        """Initialize triple barrier labeler.
        
        Args:
            barrier_config: Configuration for barriers
        """
        self.barrier_config = barrier_config
        self.logger = system_logger.getChild('TripleBarrierLabeler')
        self.profit_taking = barrier_config.get('profit_taking', 0.02)
        self.stop_loss = barrier_config.get('stop_loss', 0.01)
        self.max_holding_period = barrier_config.get('max_holding_period', 100)
        self.min_holding_period = barrier_config.get('min_holding_period', 1)

    async def label(self, data: pd.DataFrame, regime_info: Optional[Dict[str, Any]]=None) -> pd.DataFrame:
        """Apply triple barrier labeling to data.
        
        Args:
            data: Market data to label
            regime_info: Optional regime information for regime-aware labeling
            
        Returns:
            Labeled DataFrame
        """
        self.logger.info('🎯 Applying triple barrier labeling...')
        if regime_info and regime_info.get('labels') is not None:
            labeled_data = await self._regime_aware_labeling(data, regime_info)
        else:
            labeled_data = self._standard_labeling(data)
        labeled_data = self._add_label_features(labeled_data)
        self._log_labeling_summary(labeled_data)
        return labeled_data

    def _standard_labeling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply standard triple barrier labeling.
        
        Args:
            data: Market data
            
        Returns:
            Labeled data
        """
        n_samples = len(data)
        labels = np.zeros(n_samples)
        label_metadata = {'exit_index': np.full(n_samples, -1, dtype=int), 'holding_period': np.zeros(n_samples, dtype=int), 'exit_return': np.zeros(n_samples), 'exit_reason': [''] * n_samples, 'max_profit': np.zeros(n_samples), 'max_loss': np.zeros(n_samples)}
        close_prices = data['close'].values
        for i in range(n_samples - self.min_holding_period):
            entry_price = close_prices[i]
            max_j = min(i + self.max_holding_period + 1, n_samples)
            max_profit = 0
            max_loss = 0
            for j in range(i + self.min_holding_period, max_j):
                exit_price = close_prices[j]
                return_pct = (exit_price - entry_price) / entry_price
                max_profit = max(max_profit, return_pct)
                max_loss = min(max_loss, return_pct)
                if return_pct >= self.profit_taking:
                    labels[i] = 1
                    label_metadata['exit_index'][i] = j
                    label_metadata['holding_period'][i] = j - i
                    label_metadata['exit_return'][i] = return_pct
                    label_metadata['exit_reason'][i] = 'profit_target'
                    break
                elif return_pct <= -self.stop_loss:
                    labels[i] = -1
                    label_metadata['exit_index'][i] = j
                    label_metadata['holding_period'][i] = j - i
                    label_metadata['exit_return'][i] = return_pct
                    label_metadata['exit_reason'][i] = 'stop_loss'
                    break
                elif j == max_j - 1:
                    if return_pct > 0:
                        labels[i] = 1
                    else:
                        labels[i] = -1
                    label_metadata['exit_index'][i] = j
                    label_metadata['holding_period'][i] = j - i
                    label_metadata['exit_return'][i] = return_pct
                    label_metadata['exit_reason'][i] = 'max_holding'
                    break
            label_metadata['max_profit'][i] = max_profit
            label_metadata['max_loss'][i] = max_loss
        labeled_data = data.copy()
        labeled_data['label'] = labels
        labeled_data['label_binary'] = (labels > 0).astype(int)
        for key, values in label_metadata.items():
            labeled_data[f'label_{key}'] = values
        return labeled_data

    async def _regime_aware_labeling(self, data: pd.DataFrame, regime_info: Dict[str, Any]) -> pd.DataFrame:
        """Apply regime-aware triple barrier labeling.
        
        Args:
            data: Market data
            regime_info: Regime information
            
        Returns:
            Labeled data with regime-specific barriers
        """
        self.logger.info('🔄 Using regime-aware barrier parameters...')
        regime_labels = regime_info['labels']
        regime_characteristics = regime_info.get('characteristics', {})
        regime_barriers = self._calculate_regime_barriers(regime_characteristics)
        n_samples = len(data)
        labels = np.zeros(n_samples)
        label_metadata = {'exit_index': np.full(n_samples, -1, dtype=int), 'holding_period': np.zeros(n_samples, dtype=int), 'exit_return': np.zeros(n_samples), 'exit_reason': [''] * n_samples, 'regime_at_entry': np.zeros(n_samples, dtype=int), 'regime_at_exit': np.zeros(n_samples, dtype=int), 'used_profit_barrier': np.zeros(n_samples), 'used_loss_barrier': np.zeros(n_samples)}
        close_prices = data['close'].values
        for i in range(n_samples - self.min_holding_period):
            entry_regime = regime_labels[i] if i < len(regime_labels) else 0
            entry_price = close_prices[i]
            barriers = regime_barriers.get(entry_regime, {'profit_taking': self.profit_taking, 'stop_loss': self.stop_loss, 'max_holding': self.max_holding_period})
            max_j = min(i + barriers['max_holding'] + 1, n_samples)
            for j in range(i + self.min_holding_period, max_j):
                exit_price = close_prices[j]
                return_pct = (exit_price - entry_price) / entry_price
                exit_regime = regime_labels[j] if j < len(regime_labels) else 0
                if return_pct >= barriers['profit_taking']:
                    labels[i] = 1
                    label_metadata['exit_index'][i] = j
                    label_metadata['holding_period'][i] = j - i
                    label_metadata['exit_return'][i] = return_pct
                    label_metadata['exit_reason'][i] = 'profit_target'
                    label_metadata['regime_at_entry'][i] = entry_regime
                    label_metadata['regime_at_exit'][i] = exit_regime
                    label_metadata['used_profit_barrier'][i] = barriers['profit_taking']
                    label_metadata['used_loss_barrier'][i] = barriers['stop_loss']
                    break
                elif return_pct <= -barriers['stop_loss']:
                    labels[i] = -1
                    label_metadata['exit_index'][i] = j
                    label_metadata['holding_period'][i] = j - i
                    label_metadata['exit_return'][i] = return_pct
                    label_metadata['exit_reason'][i] = 'stop_loss'
                    label_metadata['regime_at_entry'][i] = entry_regime
                    label_metadata['regime_at_exit'][i] = exit_regime
                    label_metadata['used_profit_barrier'][i] = barriers['profit_taking']
                    label_metadata['used_loss_barrier'][i] = barriers['stop_loss']
                    break
                elif j == max_j - 1:
                    if return_pct > 0:
                        labels[i] = 1
                    else:
                        labels[i] = -1
                    label_metadata['exit_index'][i] = j
                    label_metadata['holding_period'][i] = j - i
                    label_metadata['exit_return'][i] = return_pct
                    label_metadata['exit_reason'][i] = 'max_holding'
                    label_metadata['regime_at_entry'][i] = entry_regime
                    label_metadata['regime_at_exit'][i] = exit_regime
                    label_metadata['used_profit_barrier'][i] = barriers['profit_taking']
                    label_metadata['used_loss_barrier'][i] = barriers['stop_loss']
                    break
        labeled_data = data.copy()
        labeled_data['label'] = labels
        labeled_data['label_binary'] = (labels > 0).astype(int)
        for key, values in label_metadata.items():
            labeled_data[f'label_{key}'] = values
        if 'regime_label' not in labeled_data.columns and len(regime_labels) == len(labeled_data):
            labeled_data['regime_label'] = regime_labels
        return labeled_data

    def _calculate_regime_barriers(self, regime_characteristics: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
        """Calculate regime-specific barrier parameters.
        
        Args:
            regime_characteristics: Characteristics of each regime
            
        Returns:
            Dictionary mapping regime ID to barrier parameters
        """
        regime_barriers = {}
        for regime_key, chars in regime_characteristics.items():
            if isinstance(chars, dict) and regime_key.startswith('regime_'):
                regime_id = int(regime_key.split('_')[1])
                volatility = chars.get('volatility_20_mean', chars.get('volatility_10_mean', 0.01))
                volatility_multiplier = max(0.5, min(2.0, volatility / 0.01))
                regime_barriers[regime_id] = {'profit_taking': self.profit_taking * volatility_multiplier, 'stop_loss': self.stop_loss * volatility_multiplier, 'max_holding': int(self.max_holding_period / volatility_multiplier)}
                self.logger.info(f"Regime {regime_id} barriers: PT={regime_barriers[regime_id]['profit_taking']:.3f}, SL={regime_barriers[regime_id]['stop_loss']:.3f}, MH={regime_barriers[regime_id]['max_holding']}")
        return regime_barriers

    def _add_label_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add additional features derived from labels.
        
        Args:
            data: Labeled data
            
        Returns:
            Data with additional label features
        """
        if 'label' in data.columns:
            if 'label_exit_return' in data.columns:
                data['label_confidence'] = np.abs(data['label_exit_return'])
            if 'label_holding_period' in data.columns:
                max_holding = data['label_holding_period'].max()
                if max_holding > 0:
                    data['label_quality'] = 1 - data['label_holding_period'] / max_holding
                else:
                    data['label_quality'] = 1.0
            if 'label_max_profit' in data.columns and 'label_max_loss' in data.columns:
                data['label_profit_loss_ratio'] = np.where(data['label_max_loss'] < -0.0001, -data['label_max_profit'] / data['label_max_loss'], np.inf)
        return data

    def _log_labeling_summary(self, data: pd.DataFrame) -> None:
        """Log summary of labeling results.
        
        Args:
            data: Labeled data
        """
        if 'label' in data.columns:
            label_counts = data['label'].value_counts().sort_index()
            total_labeled = (data['label'] != 0).sum()
            self.logger.info('📊 Triple Barrier Labeling Summary:')
            self.logger.info(f'   Total samples: {len(data):,}')
            self.logger.info(f'   Labeled samples: {total_labeled:,} ({total_labeled / len(data) * 100:.1f}%)')
            for label_value, count in label_counts.items():
                pct = count / len(data) * 100
                label_name = 'Positive' if label_value > 0 else 'Negative' if label_value < 0 else 'Neutral'
                self.logger.info(f'   {label_name} ({label_value}): {count:,} ({pct:.1f}%)')
            if 'label_exit_reason' in data.columns:
                exit_reasons = data[data['label'] != 0]['label_exit_reason'].value_counts()
                self.logger.info('\n   Exit reasons:')
                for reason, count in exit_reasons.items():
                    if reason:
                        pct = count / total_labeled * 100
                        self.logger.info(f'     {reason}: {count:,} ({pct:.1f}%)')
            if 'label_holding_period' in data.columns:
                avg_holding = data[data['label'] != 0]['label_holding_period'].mean()
                self.logger.info(f'\n   Average holding period: {avg_holding:.1f} bars')