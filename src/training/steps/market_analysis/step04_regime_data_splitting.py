"""Step 4: Regime Data Splitting - Refactored to use BaseStep.

This module creates a unified dataset with regime labels for regime-aware processing.
Uses labels to differentiate regimes instead of creating separate files per regime.
"""
from typing import Any, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import json
from src.training.base_step import BaseStep
from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from copy import copy
import asyncio

class RegimeDataSplittingStep(BaseStep):
    """Step 4: Regime Data Splitting using standardized base class."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize regime data splitting step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, '04', 'regime_data_splitting')
        self.min_regime_samples = config.get('min_regime_samples', 100)
        self.train_ratio = config.get('train_ratio', 0.8)
        self.val_ratio = config.get('val_ratio', 0.1)
        self.test_ratio = config.get('test_ratio', 0.1)
        self.stratify_by_regime = config.get('stratify_by_regime', True)

    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            self.logger.warning(f'Split ratios sum to {total_ratio}, normalizing to 1.0')
            self.train_ratio /= total_ratio
            self.val_ratio /= total_ratio
            self.test_ratio /= total_ratio
        self.logger.info('✅ Regime data splitting step initialized')

    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        if 'features' not in pipeline_state:
            errors.append('No features from step 3')
        if 'regime_labels' not in pipeline_state:
            errors.append('No regime labels from step 3')
        if 'validated_data' not in pipeline_state and 'dataframe' not in pipeline_state:
            errors.append('No validated data from step 2')
        if 'features' in pipeline_state and 'regime_labels' in pipeline_state:
            features = pipeline_state['features']
            labels = pipeline_state['regime_labels']
            if len(features) != len(labels):
                errors.append(f'Feature/label length mismatch: {len(features)} vs {len(labels)}')
        return (len(errors) == 0, errors)

    @handles_errors(exceptions=(Exception,), default_return={'success': False}, context='regime data splitting execution')
    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute regime data splitting logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state
        """
        features = pipeline_state['features']
        regime_labels = pipeline_state['regime_labels']
        original_data = pipeline_state.get('validated_data') or pipeline_state.get('dataframe')
        self.logger.info('🔀 Starting regime data splitting...')
        unified_data = self._create_unified_dataset(original_data, features, regime_labels)
        regime_stats = self._analyze_regime_distribution(regime_labels)
        self._log_regime_stats(regime_stats)
        split_data = self._split_data(unified_data, regime_labels, stratify=self.stratify_by_regime)
        split_reports = self._generate_split_reports(split_data, regime_stats)
        pipeline_state.update({'unified_data': unified_data, 'train_data': split_data['train'], 'val_data': split_data['val'], 'test_data': split_data['test'], 'split_indices': split_data['indices'], 'regime_statistics': regime_stats, 'split_reports': split_reports})
        await self._save_outputs(training_input, pipeline_state)
        return pipeline_state

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        required_outputs = ['unified_data', 'train_data', 'val_data', 'test_data']
        for output in required_outputs:
            if output not in pipeline_state:
                errors.append(f'Missing required output: {output}')
        if all((key in pipeline_state for key in ['train_data', 'val_data', 'test_data'])):
            train_len = len(pipeline_state['train_data'])
            val_len = len(pipeline_state['val_data'])
            test_len = len(pipeline_state['test_data'])
            total_len = train_len + val_len + test_len
            if total_len == 0:
                errors.append('No data in splits')
            else:
                actual_train_ratio = train_len / total_len
                actual_val_ratio = val_len / total_len
                actual_test_ratio = test_len / total_len
                tolerance = 0.05
                if abs(actual_train_ratio - self.train_ratio) > tolerance:
                    errors.append(f'Train split ratio mismatch: expected {self.train_ratio:.2f}, got {actual_train_ratio:.2f}')
        if 'regime_statistics' in pipeline_state:
            regime_stats = pipeline_state['regime_statistics']
            for regime_id, stats in regime_stats.items():
                if stats['count'] < self.min_regime_samples:
                    self.logger.warning(f"Regime {regime_id} has only {stats['count']} samples (minimum: {self.min_regime_samples})")
        return (len(errors) == 0, errors)

    def _create_unified_dataset(self, original_data: pd.DataFrame, features: pd.DataFrame, regime_labels: np.ndarray) -> pd.DataFrame:
        """Create unified dataset with all data and regime labels.
        
        Args:
            original_data: Original market data
            features: Engineered features
            regime_labels: Regime labels array
            
        Returns:
            Unified DataFrame
        """
        unified = original_data.copy()
        common_index = unified.index.intersection(features.index)
        unified = unified.loc[common_index]
        for col in features.columns:
            unified[f'feature_{col}'] = features.loc[common_index, col]
        if len(regime_labels) == len(features):
            label_series = pd.Series(regime_labels, index=features.index)
            unified['regime_label'] = label_series.loc[common_index]
        else:
            self.logger.warning(f'Regime label length mismatch: {len(regime_labels)} vs {len(features)}')
            unified['regime_label'] = regime_labels[:len(unified)]
        unified['regime'] = unified['regime_label'].astype('category')
        self.logger.info(f'✅ Created unified dataset with {len(unified)} rows and {len(unified.columns)} columns')
        return unified

    def _analyze_regime_distribution(self, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze the distribution of regimes.
        
        Args:
            regime_labels: Array of regime labels
            
        Returns:
            Dictionary with regime statistics
        """
        unique_regimes, counts = np.unique(regime_labels, return_counts=True)
        total_samples = len(regime_labels)
        regime_stats = {}
        for regime, count in zip(unique_regimes, counts):
            regime_stats[int(regime)] = {'count': int(count), 'percentage': float(count / total_samples * 100), 'sufficient_samples': count >= self.min_regime_samples}
        transitions = 0
        for i in range(1, len(regime_labels)):
            if regime_labels[i] != regime_labels[i - 1]:
                transitions += 1
        regime_stats['total_transitions'] = transitions
        regime_stats['avg_regime_duration'] = total_samples / (transitions + 1)
        return regime_stats

    def _split_data(self, data: pd.DataFrame, regime_labels: np.ndarray, stratify: bool=True) -> Dict[str, Any]:
        """Split data into train/validation/test sets.
        
        Args:
            data: Unified dataset
            regime_labels: Regime labels
            stratify: Whether to stratify by regime
            
        Returns:
            Dictionary with split data and indices
        """
        n_samples = len(data)
        indices = np.arange(n_samples)
        if stratify and 'regime_label' in data.columns:
            train_indices = []
            val_indices = []
            test_indices = []
            for regime in np.unique(data['regime_label']):
                regime_mask = data['regime_label'] == regime
                regime_indices = indices[regime_mask]
                n_regime = len(regime_indices)
                n_train = int(n_regime * self.train_ratio)
                n_val = int(n_regime * self.val_ratio)
                np.random.shuffle(regime_indices)
                train_indices.extend(regime_indices[:n_train])
                val_indices.extend(regime_indices[n_train:n_train + n_val])
                test_indices.extend(regime_indices[n_train + n_val:])
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
            test_indices = np.array(test_indices)
        else:
            n_train = int(n_samples * self.train_ratio)
            n_val = int(n_samples * self.val_ratio)
            np.random.shuffle(indices)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
        split_data = {'train': data.iloc[train_indices], 'val': data.iloc[val_indices], 'test': data.iloc[test_indices], 'indices': {'train': train_indices.tolist(), 'val': val_indices.tolist(), 'test': test_indices.tolist()}}
        self.logger.info(f'✅ Split data: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}')
        return split_data

    def _log_regime_stats(self, regime_stats: Dict[str, Any]) -> None:
        """Log regime distribution statistics.
        
        Args:
            regime_stats: Regime statistics dictionary
        """
        self.logger.info('📊 Regime Distribution:')
        for regime_id, stats in regime_stats.items():
            if isinstance(stats, dict):
                self.logger.info(f"   Regime {regime_id}: {stats['count']} samples ({stats['percentage']:.1f}%) - {('✅' if stats['sufficient_samples'] else '⚠️ Low samples')}")
        if 'total_transitions' in regime_stats:
            self.logger.info(f"   Total transitions: {regime_stats['total_transitions']}")
            self.logger.info(f"   Avg regime duration: {regime_stats['avg_regime_duration']:.1f} samples")

    def _generate_split_reports(self, split_data: Dict[str, pd.DataFrame], regime_stats: Dict[str, Any]) -> Dict[str, str]:
        """Generate reports for data splits.
        
        Args:
            split_data: Dictionary with split dataframes
            regime_stats: Regime statistics
            
        Returns:
            Dictionary of reports
        """
        reports = {}
        summary_lines = ['Data Split Summary', '=' * 40, f'Total samples: {sum((len(df) for df in split_data.values() if isinstance(df, pd.DataFrame)))}', '', 'Split sizes:']
        for split_name in ['train', 'val', 'test']:
            if split_name in split_data:
                df = split_data[split_name]
                summary_lines.append(f'  {split_name.capitalize()}: {len(df)} samples')
                if 'regime_label' in df.columns:
                    regime_dist = df['regime_label'].value_counts().sort_index()
                    for regime, count in regime_dist.items():
                        pct = count / len(df) * 100
                        summary_lines.append(f'    Regime {regime}: {count} ({pct:.1f}%)')
        reports['summary'] = '\n'.join(summary_lines)
        detail_lines = ['Detailed Split Analysis', '=' * 40]
        for split_name in ['train', 'val', 'test']:
            if split_name in split_data:
                df = split_data[split_name]
                detail_lines.extend(['', f'{split_name.upper()} SET:', f'  Shape: {df.shape}', f'  Date range: {df.index.min()} to {df.index.max()}' if hasattr(df.index, 'min') else '', f"  Features: {sum((1 for col in df.columns if col.startswith('feature_')))}", f"  Original columns: {sum((1 for col in df.columns if not col.startswith('feature_')))}"])
        reports['detailed'] = '\n'.join(detail_lines)
        return reports

    async def _save_outputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> None:
        """Save step outputs to disk.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Pipeline state with results
        """
        output_dir = Path(training_input.get('output_dir', 'output')) / 'step04_regime_split'
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ['train', 'val', 'test']:
            if f'{split_name}_data' in pipeline_state:
                data = pipeline_state[f'{split_name}_data']
                file_path = output_dir / f'{split_name}_data.parquet'
                data.to_parquet(file_path)
                self.logger.info(f'💾 Saved {split_name} data to {file_path}')
        if 'split_indices' in pipeline_state:
            indices_path = output_dir / 'split_indices.json'
            with open(indices_path, 'w') as f:
                json.dump(pipeline_state['split_indices'], f, indent=2)
            self.logger.info(f'💾 Saved split indices to {indices_path}')
        if 'regime_statistics' in pipeline_state:
            stats_path = output_dir / 'regime_statistics.json'
            with open(stats_path, 'w') as f:
                json.dump(pipeline_state['regime_statistics'], f, indent=2)
            self.logger.info(f'💾 Saved regime statistics to {stats_path}')
        if 'split_reports' in pipeline_state:
            for report_name, content in pipeline_state['split_reports'].items():
                report_path = output_dir / f'{report_name}_report.txt'
                with open(report_path, 'w') as f:
                    f.write(content)
                self.logger.info(f'💾 Saved {report_name} report')

    def get_required_inputs(self) -> list:
        """Get list of required inputs for this step."""
        return ['features', 'regime_labels', 'validated_data or dataframe']

    def get_produced_outputs(self) -> list:
        """Get list of outputs produced by this step."""
        return ['unified_data', 'train_data', 'val_data', 'test_data', 'split_indices', 'regime_statistics', 'split_reports']

    def get_dependencies(self) -> list:
        """Get list of step dependencies."""
        return ['03_hmm_regime_discovery']