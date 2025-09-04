"""Step 14: Tactician Labeling - Migrated to use BaseStep pattern.

This step applies regime-aware triple barrier labeling for tactician multi-outcome predictions.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.core.decorators import handles_errors, log_execution_time, validates
from src.training.base_step import BaseStep
from src.utils.logger import system_logger
from .labeling_strategy import LabelingStrategy
from .barrier_calculator import BarrierCalculator
from .quality_filter import QualityFilter
from .precision_optimizer import PrecisionOptimizer
from copy import copy

class TacticianLabelingStep(BaseStep):
    """Step 14: Tactician Labeling with regime-aware strategies."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the Tactician Labeling step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, '14', 'tactician_labeling')

    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        self.labeling_strategy = LabelingStrategy(self.config)
        self.barrier_calculator = BarrierCalculator(self.config)
        self.quality_filter = QualityFilter(self.config)
        self.precision_optimizer = PrecisionOptimizer(self.config)
        self.tactician_config = self._initialize_tactician_config()
        self.regime_barrier_results: Dict[str, Any] = {}
        self.regime_labeling_results: Dict[str, Any] = {}
        self.regime_validation_results: Dict[str, Any] = {}

    def _initialize_tactician_config(self) -> Dict[str, Any]:
        """Initialize tactician-specific configuration."""
        return {'regime_specific_barriers': True, 'regime_specific_precision': True, 'regime_specific_quality_filters': True, 'min_regime_samples': 100, 'barrier_combinations': 4, 'max_lookahead': 50, 'dynamic_barriers': True, 'enable_high_precision_mode': True, 'precision_threshold': 0.85, 'min_signal_strength': 0.8, 'enable_quality_filters': True, 'min_volume_threshold': 1000, 'min_spread_threshold': 0.0001, 'volatility_filter': True, 'analyst_signal_requirement': True, 'direction_agreement_required': True, 'confidence_boost_threshold': 0.9, 'timeframes': ['1m', '5m'], 'primary_timeframe': '1m', 'secondary_timeframe': '5m', 'binary_classification': True, 'multi_outcome_prediction': True}

    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        if 'step13_analyst_ensemble_creation_completed' not in pipeline_state:
            errors.append('Step 13 (Analyst Ensemble Creation) must be completed before tactician labeling')
        if 'analyst_ensembles' not in pipeline_state:
            errors.append('No analyst ensembles found in pipeline state')
        if 'market_data' not in pipeline_state:
            errors.append('No market data found for labeling')
        if 'regime_data' not in pipeline_state:
            errors.append('No regime data found for regime-specific labeling')
        required_config = ['tactician_labeling', 'barrier_calculation', 'quality_filters']
        for key in required_config:
            if key not in self.config:
                errors.append(f'Missing required configuration: {key}')
        return (len(errors) == 0, errors)

    @handles_errors(exceptions=(Exception,), default_return={'success': False}, context='tactician labeling logic')
    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the main tactician labeling logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with labeled data
        """
        self.logger.info('🎯 Starting tactician labeling...')
        analyst_ensembles = pipeline_state['analyst_ensembles']
        market_data = pipeline_state['market_data']
        regime_data = pipeline_state['regime_data']
        labeled_data = {}
        labeling_results = {}
        barrier_results = {}
        for regime_id, regime_info in regime_data.items():
            self.logger.info(f'📊 Processing regime {regime_id} for tactician labeling...')
            regime_market_data = self._get_regime_market_data(market_data, regime_info)
            if len(regime_market_data) < self.tactician_config['min_regime_samples']:
                self.logger.warning(f"Regime {regime_id} has insufficient samples ({len(regime_market_data)} < {self.tactician_config['min_regime_samples']})")
                continue
            regime_analyst_predictions = await self._get_regime_analyst_predictions(regime_id, analyst_ensembles, regime_market_data)
            regime_labeled = await self._apply_regime_labeling(regime_id, regime_market_data, regime_analyst_predictions, regime_info)
            if regime_labeled is not None:
                labeled_data[regime_id] = regime_labeled['data']
                labeling_results[regime_id] = regime_labeled['results']
                barrier_results[regime_id] = regime_labeled['barriers']
        combined_labeled_data = self._combine_regime_labeled_data(labeled_data)
        if self.tactician_config['enable_quality_filters']:
            self.logger.info('🔍 Applying global quality filters...')
            combined_labeled_data = await self.quality_filter.apply_global_filters(combined_labeled_data)
        if self.tactician_config['enable_high_precision_mode']:
            self.logger.info('🎯 Optimizing precision thresholds...')
            precision_results = await self.precision_optimizer.optimize_thresholds(combined_labeled_data, labeling_results)
        else:
            precision_results = {}
        result = pipeline_state.copy()
        result['tactician_labeled_data'] = combined_labeled_data
        result['tactician_labeling_results'] = labeling_results
        result['tactician_barrier_results'] = barrier_results
        result['tactician_precision_results'] = precision_results
        result['tactician_labeling_summary'] = self._create_labeling_summary(labeled_data, labeling_results, barrier_results)
        return result

    async def _apply_regime_labeling(self, regime_id: str, regime_data: pd.DataFrame, analyst_predictions: pd.DataFrame, regime_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply labeling for a specific regime.
        
        Args:
            regime_id: Regime identifier
            regime_data: Market data for this regime
            analyst_predictions: Analyst predictions for this regime
            regime_info: Regime metadata
            
        Returns:
            Dictionary with labeled data and results
        """
        try:
            self.logger.info(f'  📏 Calculating barriers for regime {regime_id}...')
            regime_barriers = await self.barrier_calculator.calculate_regime_barriers(regime_data, regime_info, self.tactician_config['barrier_combinations'])
            if self.tactician_config['regime_specific_quality_filters']:
                quality_filters = await self.quality_filter.get_regime_filters(regime_id, regime_data)
                filtered_data = await self.quality_filter.apply_regime_filters(regime_data, quality_filters)
            else:
                filtered_data = regime_data
                quality_filters = {}
            if self.tactician_config['regime_specific_precision']:
                precision_thresholds = await self.precision_optimizer.get_regime_thresholds(regime_id, filtered_data, analyst_predictions)
            else:
                precision_thresholds = {'min_signal_strength': self.tactician_config['min_signal_strength'], 'precision_threshold': self.tactician_config['precision_threshold']}
            labeled_data = await self.labeling_strategy.apply_triple_barrier_labeling(filtered_data, regime_barriers, analyst_predictions, precision_thresholds, regime_id)
            labeling_stats = self._calculate_labeling_statistics(labeled_data, filtered_data, len(regime_data))
            self.logger.info(f"  ✅ Regime {regime_id} labeling complete: {labeling_stats['labeled_samples']} labeled, {labeling_stats['positive_rate']:.2%} positive rate")
            return {'data': labeled_data, 'results': {'statistics': labeling_stats, 'precision_thresholds': precision_thresholds, 'quality_filters': quality_filters}, 'barriers': regime_barriers}
        except Exception as e:
            self.logger.error(f'Failed to label regime {regime_id}: {str(e)}')
            return None

    async def _get_regime_analyst_predictions(self, regime_id: str, analyst_ensembles: Dict[str, Any], regime_data: pd.DataFrame) -> pd.DataFrame:
        """Get analyst predictions for a specific regime.
        
        Args:
            regime_id: Regime identifier
            analyst_ensembles: All analyst ensemble models
            regime_data: Market data for this regime
            
        Returns:
            DataFrame with analyst predictions
        """
        predictions = pd.DataFrame(index=regime_data.index)
        if regime_id in analyst_ensembles:
            regime_ensemble = analyst_ensembles[regime_id]
            best_type = regime_ensemble.get('best_type', 'voting')
            ensemble_models = regime_ensemble.get('ensemble', {})
            if best_type in ensemble_models:
                model = ensemble_models[best_type]
                try:
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(regime_data)
                        predictions['analyst_signal'] = pred_proba[:, 1]
                        predictions['analyst_prediction'] = (pred_proba[:, 1] > 0.5).astype(int)
                    else:
                        predictions['analyst_prediction'] = model.predict(regime_data)
                        predictions['analyst_signal'] = predictions['analyst_prediction']
                except Exception as e:
                    self.logger.warning(f'Failed to get predictions for regime {regime_id}: {str(e)}')
                    predictions['analyst_signal'] = 0.5
                    predictions['analyst_prediction'] = 0
        elif 'cross_regime' in analyst_ensembles:
            cross_ensemble = analyst_ensembles['cross_regime']
            predictions['analyst_signal'] = 0.5
            predictions['analyst_prediction'] = 0
        else:
            predictions['analyst_signal'] = 0.5
            predictions['analyst_prediction'] = 0
        return predictions

    def _get_regime_market_data(self, market_data: pd.DataFrame, regime_info: Dict[str, Any]) -> pd.DataFrame:
        """Extract market data for a specific regime."""
        regime_mask = regime_info.get('mask', [])
        if isinstance(regime_mask, list) and regime_mask:
            regime_mask = np.array(regime_mask)
            return market_data[regime_mask]
        return market_data

    def _combine_regime_labeled_data(self, labeled_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine labeled data from all regimes."""
        if not labeled_data:
            return pd.DataFrame()
        combined = pd.concat(labeled_data.values(), axis=0)
        combined = combined.sort_index()
        for regime_id, regime_df in labeled_data.items():
            combined.loc[regime_df.index, 'regime_id'] = regime_id
        return combined

    def _calculate_labeling_statistics(self, labeled_data: pd.DataFrame, filtered_data: pd.DataFrame, original_size: int) -> Dict[str, Any]:
        """Calculate statistics for labeled data."""
        stats = {'original_samples': original_size, 'filtered_samples': len(filtered_data), 'labeled_samples': len(labeled_data), 'filtering_rate': 1 - len(filtered_data) / original_size if original_size > 0 else 0, 'labeling_rate': len(labeled_data) / len(filtered_data) if len(filtered_data) > 0 else 0}
        if 'label' in labeled_data.columns:
            label_counts = labeled_data['label'].value_counts()
            total_labeled = label_counts.sum()
            stats.update({'positive_samples': label_counts.get(1, 0), 'negative_samples': label_counts.get(0, 0), 'positive_rate': label_counts.get(1, 0) / total_labeled if total_labeled > 0 else 0, 'negative_rate': label_counts.get(0, 0) / total_labeled if total_labeled > 0 else 0})
            if not self.tactician_config['binary_classification']:
                for label_value in label_counts.index:
                    if label_value not in [0, 1]:
                        stats[f'class_{label_value}_samples'] = label_counts[label_value]
                        stats[f'class_{label_value}_rate'] = label_counts[label_value] / total_labeled
        return stats

    def _create_labeling_summary(self, labeled_data: Dict[str, pd.DataFrame], labeling_results: Dict[str, Any], barrier_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the labeling process."""
        summary = {'total_regimes': len(labeled_data), 'successfully_labeled_regimes': len(labeling_results), 'total_samples': sum((len(df) for df in labeled_data.values())), 'regime_statistics': {}, 'average_metrics': {'filtering_rate': [], 'positive_rate': [], 'labeling_rate': []}}
        for regime_id, results in labeling_results.items():
            if 'statistics' in results:
                stats = results['statistics']
                summary['regime_statistics'][regime_id] = {'samples': stats.get('labeled_samples', 0), 'positive_rate': stats.get('positive_rate', 0), 'filtering_rate': stats.get('filtering_rate', 0)}
                summary['average_metrics']['filtering_rate'].append(stats.get('filtering_rate', 0))
                summary['average_metrics']['positive_rate'].append(stats.get('positive_rate', 0))
                summary['average_metrics']['labeling_rate'].append(stats.get('labeling_rate', 0))
        for metric, values in summary['average_metrics'].items():
            if values:
                summary['average_metrics'][metric] = np.mean(values)
            else:
                summary['average_metrics'][metric] = 0.0
        summary['barrier_configurations'] = len(barrier_results)
        return summary

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        required_outputs = ['tactician_labeled_data', 'tactician_labeling_results', 'tactician_barrier_results', 'tactician_labeling_summary']
        for output in required_outputs:
            if output not in pipeline_state:
                errors.append(f'Missing required output: {output}')
        if 'tactician_labeled_data' in pipeline_state:
            labeled_data = pipeline_state['tactician_labeled_data']
            if not isinstance(labeled_data, pd.DataFrame):
                errors.append('Labeled data must be a DataFrame')
            elif len(labeled_data) == 0:
                errors.append('No data was labeled')
            elif 'label' not in labeled_data.columns:
                errors.append("Labeled data missing 'label' column")
        if 'tactician_labeling_summary' in pipeline_state:
            summary = pipeline_state['tactician_labeling_summary']
            if summary.get('total_samples', 0) == 0:
                errors.append('No samples were processed')
        return (len(errors) == 0, errors)

    def get_required_inputs(self) -> List[str]:
        """Get list of required inputs for this step."""
        return ['analyst_ensembles', 'market_data', 'regime_data', 'step13_analyst_ensemble_creation_completed']

    def get_produced_outputs(self) -> List[str]:
        """Get list of outputs produced by this step."""
        return ['tactician_labeled_data', 'tactician_labeling_results', 'tactician_barrier_results', 'tactician_precision_results', 'tactician_labeling_summary']

    def get_dependencies(self) -> List[str]:
        """Get list of step dependencies."""
        return ['step13_analyst_ensemble_creation']