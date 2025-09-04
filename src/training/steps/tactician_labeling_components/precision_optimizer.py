"""Precision optimization component for tactician labeling."""
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger

class PrecisionOptimizer:
    """Handles precision optimization for tactician labeling."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the precision optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('precision_optimization', {})
        self.logger = system_logger.getChild('precision_optimizer')
        self.target_precision = self.config.get('target_precision', 0.85)
        self.min_recall = self.config.get('min_recall', 0.1)
        self.optimization_metric = self.config.get('optimization_metric', 'f1')
        self.threshold_search_steps = self.config.get('threshold_search_steps', 20)
        self.signal_strength_range = self.config.get('signal_strength_range', (0.5, 0.95))
        self.precision_threshold_range = self.config.get('precision_threshold_range', (0.7, 0.95))
        self.regime_specific_optimization = self.config.get('regime_specific_optimization', True)
        self.cross_regime_validation = self.config.get('cross_regime_validation', True)

    @handles_errors(exceptions=(Exception,), default_return={}, context='regime threshold calculation')
    async def get_regime_thresholds(self, regime_id: str, regime_data: pd.DataFrame, analyst_predictions: pd.DataFrame) -> Dict[str, float]:
        """Get optimized thresholds for a specific regime.
        
        Args:
            regime_id: Regime identifier
            regime_data: Market data for the regime
            analyst_predictions: Analyst predictions
            
        Returns:
            Dictionary of optimized thresholds
        """
        self.logger.info(f'Calculating precision thresholds for regime {regime_id}')
        thresholds = {'min_signal_strength': 0.8, 'precision_threshold': self.target_precision, 'confidence_threshold': 0.7, 'min_samples_for_signal': 10}
        if analyst_predictions.empty or 'analyst_signal' not in analyst_predictions.columns:
            return thresholds
        signal_stats = self._analyze_signal_distribution(analyst_predictions)
        if self.regime_specific_optimization:
            if 'close' in regime_data.columns:
                volatility = regime_data['close'].pct_change().std()
                if volatility > 0.02:
                    thresholds['min_signal_strength'] = min(0.9, thresholds['min_signal_strength'] + 0.1)
                    thresholds['precision_threshold'] = min(0.9, thresholds['precision_threshold'] + 0.05)
            if signal_stats['mean_signal'] < 0.6:
                thresholds['min_signal_strength'] = max(0.85, signal_stats['percentile_75'])
            elif signal_stats['mean_signal'] > 0.8:
                thresholds['min_signal_strength'] = max(0.7, signal_stats['percentile_50'])
        return thresholds

    @handles_errors(exceptions=(Exception,), default_return={}, context='threshold optimization')
    async def optimize_thresholds(self, labeled_data: pd.DataFrame, labeling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize precision thresholds across all data.
        
        Args:
            labeled_data: Combined labeled data
            labeling_results: Results from labeling process
            
        Returns:
            Dictionary of optimization results
        """
        self.logger.info('Optimizing precision thresholds globally')
        optimization_results = {'global_thresholds': {}, 'regime_thresholds': {}, 'optimization_metrics': {}, 'validation_results': {}}
        if not labeled_data.empty and 'label' in labeled_data.columns and ('signal_strength' in labeled_data.columns):
            global_thresholds = await self._optimize_global_thresholds(labeled_data)
            optimization_results['global_thresholds'] = global_thresholds
            validation = await self._validate_thresholds(labeled_data, global_thresholds)
            optimization_results['validation_results'] = validation
        if self.regime_specific_optimization and 'regime_id' in labeled_data.columns:
            for regime_id in labeled_data['regime_id'].unique():
                regime_mask = labeled_data['regime_id'] == regime_id
                regime_labeled = labeled_data[regime_mask]
                if len(regime_labeled) > 100:
                    regime_thresholds = await self._optimize_regime_specific_thresholds(regime_labeled, regime_id)
                    optimization_results['regime_thresholds'][regime_id] = regime_thresholds
        optimization_results['optimization_metrics'] = self._calculate_optimization_metrics(labeled_data, optimization_results)
        return optimization_results

    async def _optimize_global_thresholds(self, labeled_data: pd.DataFrame) -> Dict[str, float]:
        """Optimize thresholds globally across all data.
        
        Args:
            labeled_data: Labeled data with signals
            
        Returns:
            Dictionary of optimized thresholds
        """
        if 'label' not in labeled_data.columns or 'signal_strength' not in labeled_data.columns:
            return {}
        labels = labeled_data['label'].values
        signals = labeled_data['signal_strength'].values
        binary_labels = (labels > 0).astype(int)
        best_threshold = self.signal_strength_range[0]
        best_metric = -np.inf
        thresholds_to_test = np.linspace(self.signal_strength_range[0], self.signal_strength_range[1], self.threshold_search_steps)
        for threshold in thresholds_to_test:
            predictions = (signals >= threshold).astype(int)
            if predictions.sum() == 0:
                continue
            precision = precision_score(binary_labels, predictions, zero_division=0)
            recall = recall_score(binary_labels, predictions, zero_division=0)
            f1 = f1_score(binary_labels, predictions, zero_division=0)
            if precision < self.target_precision or recall < self.min_recall:
                continue
            if self.optimization_metric == 'precision':
                metric = precision
            elif self.optimization_metric == 'recall':
                metric = recall
            elif self.optimization_metric == 'f1':
                metric = f1
            else:
                metric = f1
            if metric > best_metric:
                best_metric = metric
                best_threshold = threshold
        return {'min_signal_strength': best_threshold, 'achieved_precision': best_metric if self.optimization_metric == 'precision' else 0, 'optimization_metric': self.optimization_metric, 'metric_value': best_metric}

    async def _optimize_regime_specific_thresholds(self, regime_data: pd.DataFrame, regime_id: str) -> Dict[str, float]:
        """Optimize thresholds for a specific regime.
        
        Args:
            regime_data: Labeled data for the regime
            regime_id: Regime identifier
            
        Returns:
            Dictionary of regime-specific thresholds
        """
        thresholds = await self._optimize_global_thresholds(regime_data)
        thresholds['regime_id'] = regime_id
        thresholds['sample_size'] = len(regime_data)
        return thresholds

    async def _validate_thresholds(self, labeled_data: pd.DataFrame, thresholds: Dict[str, float]) -> Dict[str, float]:
        """Validate optimized thresholds.
        
        Args:
            labeled_data: Labeled data
            thresholds: Optimized thresholds
            
        Returns:
            Validation metrics
        """
        if 'min_signal_strength' not in thresholds:
            return {}
        threshold = thresholds['min_signal_strength']
        signals = labeled_data['signal_strength'].values
        labels = labeled_data['label'].values
        binary_labels = (labels > 0).astype(int)
        predictions = (signals >= threshold).astype(int)
        validation = {'precision': precision_score(binary_labels, predictions, zero_division=0), 'recall': recall_score(binary_labels, predictions, zero_division=0), 'f1_score': f1_score(binary_labels, predictions, zero_division=0), 'n_signals': predictions.sum(), 'signal_rate': predictions.mean(), 'true_positive_rate': (predictions & binary_labels).sum() / max(1, binary_labels.sum())}
        if self.cross_regime_validation and 'regime_id' in labeled_data.columns:
            regime_metrics = {}
            for regime_id in labeled_data['regime_id'].unique():
                regime_mask = labeled_data['regime_id'] == regime_id
                regime_predictions = predictions[regime_mask]
                regime_labels = binary_labels[regime_mask]
                if len(regime_labels) > 0:
                    regime_metrics[regime_id] = {'precision': precision_score(regime_labels, regime_predictions, zero_division=0), 'recall': recall_score(regime_labels, regime_predictions, zero_division=0), 'n_signals': regime_predictions.sum()}
            validation['regime_metrics'] = regime_metrics
        return validation

    def _analyze_signal_distribution(self, predictions: pd.DataFrame) -> Dict[str, float]:
        """Analyze the distribution of analyst signals.
        
        Args:
            predictions: Analyst predictions
            
        Returns:
            Signal distribution statistics
        """
        if 'analyst_signal' not in predictions.columns:
            return {}
        signals = predictions['analyst_signal'].dropna()
        if len(signals) == 0:
            return {}
        return {'mean_signal': signals.mean(), 'std_signal': signals.std(), 'min_signal': signals.min(), 'max_signal': signals.max(), 'percentile_25': signals.quantile(0.25), 'percentile_50': signals.quantile(0.5), 'percentile_75': signals.quantile(0.75), 'percentile_90': signals.quantile(0.9), 'n_strong_signals': (signals > 0.8).sum(), 'strong_signal_rate': (signals > 0.8).mean()}

    def _calculate_optimization_metrics(self, labeled_data: pd.DataFrame, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for the optimization process.
        
        Args:
            labeled_data: Labeled data
            optimization_results: Optimization results
            
        Returns:
            Dictionary of optimization metrics
        """
        metrics = {'total_samples': len(labeled_data), 'labeled_samples': (labeled_data['label'] != 0).sum() if 'label' in labeled_data.columns else 0, 'optimization_performed': len(optimization_results['global_thresholds']) > 0, 'regime_specific_optimizations': len(optimization_results['regime_thresholds'])}
        if 'validation_results' in optimization_results:
            validation = optimization_results['validation_results']
            metrics.update({'final_precision': validation.get('precision', 0), 'final_recall': validation.get('recall', 0), 'final_f1': validation.get('f1_score', 0), 'final_signal_rate': validation.get('signal_rate', 0)})
        return metrics