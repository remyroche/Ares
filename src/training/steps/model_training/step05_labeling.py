"""Step 5: Labeling - Refactored to use BaseStep.

This module creates comprehensive labels for the training data, combining triple barrier
labels with additional labeling strategies and meta-labeling features.
"""
from typing import Any, Dict, Tuple, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
from src.training.base_step import BaseStep
from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from copy import copy
import asyncio

class LabelingStep(BaseStep):
    """Step 5: Labeling using standardized base class."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize labeling step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, '05', 'labeling')
        self.labeling_config = config.get('labeling_config', {'use_triple_barrier': True, 'use_meta_labeling': True, 'barrier_config': {'profit_taking': 0.02, 'stop_loss': 0.01, 'max_holding_period': 100}, 'regime_aware': True})
        self.triple_barrier_labeler = None
        self.meta_labeler = None

    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        try:
            if self.labeling_config.get('use_triple_barrier', True):
                from src.training.steps.model_training.labeling_components import TripleBarrierLabeler
                self.triple_barrier_labeler = TripleBarrierLabeler(self.labeling_config.get('barrier_config', {}))
            if self.labeling_config.get('use_meta_labeling', True):
                from src.analyst.meta_labeling_system import MetaLabelingSystem
from src.core.decorators.errors import handles_errors
                self.meta_labeler = MetaLabelingSystem()
            self.logger.info('✅ Labeling components initialized')
        except ImportError as e:
            self.logger.warning(f'⚠️ Some labeling components not available: {e}')

    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check for required data
        if "unified_data" not in pipeline_state:
            # Check for individual splits
            if not all(f"{split}_data" in pipeline_state for split in ["train", "val", "test"]):
                errors.append("No unified data or split data from step 4")
        
        # Check for regime information if regime-aware labeling is enabled
        if self.labeling_config.get("regime_aware", True):
            if "regime_labels" not in pipeline_state and "regime_characteristics" not in pipeline_state:
                self.logger.warning("Regime information not available, will use standard labeling")
        
        # Validate barrier configuration
        barrier_config = self.labeling_config.get("barrier_config", {})
        if barrier_config.get("profit_taking", 0) <= 0:
            errors.append("Invalid profit_taking threshold (must be > 0)")
        if barrier_config.get("stop_loss", 0) <= 0:
            errors.append("Invalid stop_loss threshold (must be > 0)")

        # Basic schema checks on input data
        df = pipeline_state.get("unified_data") or pipeline_state.get("train_data") or pipeline_state.get("dataframe")
        if isinstance(df, pd.DataFrame):
            for col in ["open", "high", "low", "close"]:
                if col not in df.columns:
                    self.logger.warning(f"Missing expected price column: {col}")
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.warning("Input index is not DatetimeIndex")
        
        return len(errors) == 0, errors
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={"success": False},
        context="labeling execution"
    )
    async def execute_logic(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:

        """Execute labeling logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state
        """
        self.logger.info('🏷️ Starting labeling process...')
        data_to_label = self._get_data_to_label(pipeline_state)
        if self.labeling_config.get('use_triple_barrier', True):
            self.logger.info('🎯 Applying triple barrier labeling...')
            labeled_data = await self._apply_triple_barrier_labels(data_to_label, pipeline_state)
        else:
            labeled_data = data_to_label
        if self.labeling_config.get('use_meta_labeling', True) and self.meta_labeler:
            self.logger.info('🔍 Applying meta-labeling...')
            labeled_data = await self._apply_meta_labels(labeled_data)
        label_stats = self._calculate_label_statistics(labeled_data)
        self._log_label_stats(label_stats)
        reports = self._generate_labeling_reports(labeled_data, label_stats)
        pipeline_state.update({'labeled_data': labeled_data, 'label_statistics': label_stats, 'labeling_reports': reports, 'labeling_config': self.labeling_config})
        if all((f'{split}_data' in pipeline_state for split in ['train', 'val', 'test'])):
            pipeline_state = self._update_split_labels(pipeline_state, labeled_data)
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
        if 'labeled_data' not in pipeline_state:
            errors.append('No labeled data in pipeline state')
            return (False, errors)
        labeled_data = pipeline_state['labeled_data']
        label_columns = [col for col in labeled_data.columns if 'label' in col.lower()]
        if len(label_columns) == 0:
            errors.append("No label columns found in labeled data")
        else:
            # Sanity checks on labels
            try:
                for col in label_columns:
                    if labeled_data[col].isna().any():
                        errors.append(f"NaN values found in {col}")
            except Exception:
                pass
        
        # Check label statistics
        if "label_statistics" in pipeline_state:
            stats = pipeline_state["label_statistics"]
            
            # Check class balance
            if "class_distribution" in stats:
                for label_col, dist in stats["class_distribution"].items():

                    if isinstance(dist, dict):
                        values = list(dist.values())
                        if values and max(values) / min(values) > 10:
                            self.logger.warning(f'⚠️ Severe class imbalance detected in {label_col}')
        return (len(errors) == 0, errors)

    def _get_data_to_label(self, pipeline_state: Dict[str, Any]) -> pd.DataFrame:
        """Get the appropriate data to label.
        
        Args:
            pipeline_state: Current pipeline state
            
        Returns:
            DataFrame to label
        """
        if 'unified_data' in pipeline_state:
            return pipeline_state['unified_data'].copy()
        data_parts = []
        for split in ['train', 'val', 'test']:
            if f'{split}_data' in pipeline_state:
                df = pipeline_state[f'{split}_data'].copy()
                df['data_split'] = split
                data_parts.append(df)
        if data_parts:
            return pd.concat(data_parts, axis=0)
        if 'dataframe' in pipeline_state:
            return pipeline_state['dataframe'].copy()
        raise ValueError('No data available for labeling')

    async def _apply_triple_barrier_labels(self, data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> pd.DataFrame:
        """Apply triple barrier labeling.
        
        Args:
            data: Data to label
            pipeline_state: Pipeline state for regime information
            
        Returns:
            Labeled DataFrame
        """
        if self.triple_barrier_labeler:
            regime_info = None
            if self.labeling_config.get('regime_aware', True):
                regime_info = {'labels': pipeline_state.get('regime_labels'), 'characteristics': pipeline_state.get('regime_characteristics')}
            return await self.triple_barrier_labeler.label(data, regime_info)
        else:
            return self._simple_triple_barrier(data)

    def _simple_triple_barrier(self, data: pd.DataFrame) -> pd.DataFrame:
        """Simple triple barrier labeling implementation.
        
        Args:
            data: Data to label
            
        Returns:
            Labeled DataFrame
        """
        barrier_config = self.labeling_config.get('barrier_config', {})
        profit_taking = barrier_config.get('profit_taking', 0.02)
        stop_loss = barrier_config.get('stop_loss', 0.01)
        max_holding = barrier_config.get('max_holding_period', 100)
        labels = np.zeros(len(data))
        label_info = []
        close_prices = data['close'].values
        for i in range(len(data) - max_holding):
            entry_price = close_prices[i]
            for j in range(i + 1, min(i + max_holding + 1, len(data))):
                exit_price = close_prices[j]
                return_pct = (exit_price - entry_price) / entry_price
                if return_pct >= profit_taking:
                    labels[i] = 1
                    label_info.append({'index': i, 'exit_index': j, 'return': return_pct, 'reason': 'profit_target'})
                    break
                elif return_pct <= -stop_loss:
                    labels[i] = -1
                    label_info.append({'index': i, 'exit_index': j, 'return': return_pct, 'reason': 'stop_loss'})
                    break
                elif j == min(i + max_holding, len(data) - 1):
                    if return_pct > 0:
                        labels[i] = 1
                    else:
                        labels[i] = -1
                    label_info.append({'index': i, 'exit_index': j, 'return': return_pct, 'reason': 'max_holding'})
                    break
        data['triple_barrier_label'] = labels
        data['label_binary'] = (labels > 0).astype(int)
        self.logger.info(f'✅ Applied triple barrier labels: {np.sum(labels == 1)} positive, {np.sum(labels == -1)} negative, {np.sum(labels == 0)} neutral')
        return data

    async def _apply_meta_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply meta-labeling features.
        
        Args:
            data: Labeled data
            
        Returns:
            Data with meta-labels
        """
        if self.meta_labeler:
            try:
                meta_features = await self.meta_labeler.generate_meta_labels(data)
                for col in meta_features.columns:
                    data[f'meta_{col}'] = meta_features[col]
                self.logger.info(f'✅ Added {len(meta_features.columns)} meta-labeling features')
            except Exception as e:
                self.logger.warning(f'⚠️ Meta-labeling failed: {e}')
        return data

    def _calculate_label_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics about the labels.
        
        Args:
            data: Labeled data
            
        Returns:
            Label statistics dictionary
        """
        stats = {'total_samples': len(data), 'labeled_samples': 0, 'class_distribution': {}, 'label_columns': []}
        label_columns = [col for col in data.columns if 'label' in col.lower()]
        stats['label_columns'] = label_columns
        for label_col in label_columns:
            if label_col in data.columns:
                labeled_mask = data[label_col] != 0
                stats['labeled_samples'] = max(stats['labeled_samples'], labeled_mask.sum())
                value_counts = data[label_col].value_counts()
                stats['class_distribution'][label_col] = value_counts.to_dict()
                if len(value_counts) > 0:
                    stats[f'{label_col}_metrics'] = {'unique_values': len(value_counts), 'most_common': value_counts.index[0], 'most_common_pct': value_counts.iloc[0] / len(data) * 100, 'entropy': -np.sum(value_counts / len(data) * np.log2(value_counts / len(data)))}
        return stats

    def _log_label_stats(self, stats: Dict[str, Any]) -> None:
        """Log label statistics.
        
        Args:
            stats: Label statistics dictionary
        """
        self.logger.info('📊 Label Statistics:')
        self.logger.info(f"   Total samples: {stats['total_samples']:,}")
        self.logger.info(f"   Labeled samples: {stats['labeled_samples']:,}")
        self.logger.info(f"   Label columns: {', '.join(stats['label_columns'])}")
        for label_col, dist in stats['class_distribution'].items():
            if isinstance(dist, dict):
                self.logger.info(f'\n   {label_col} distribution:')
                for value, count in sorted(dist.items()):
                    pct = count / stats['total_samples'] * 100
                    self.logger.info(f'     Class {value}: {count:,} ({pct:.1f}%)')

    def _update_split_labels(self, pipeline_state: Dict[str, Any], labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Update split data with labels.
        
        Args:
            pipeline_state: Current pipeline state
            labeled_data: Data with labels
            
        Returns:
            Updated pipeline state
        """
        label_columns = [col for col in labeled_data.columns if 'label' in col.lower()]
        for split in ['train', 'val', 'test']:
            if f'{split}_data' in pipeline_state:
                split_data = pipeline_state[f'{split}_data']
                for label_col in label_columns:
                    if label_col in labeled_data.columns:
                        common_idx = split_data.index.intersection(labeled_data.index)
                        split_data.loc[common_idx, label_col] = labeled_data.loc[common_idx, label_col]
                pipeline_state[f'{split}_data'] = split_data
        return pipeline_state

    def _generate_labeling_reports(self, data: pd.DataFrame, stats: Dict[str, Any]) -> Dict[str, str]:
        """Generate labeling reports.
        
        Args:
            data: Labeled data
            stats: Label statistics
            
        Returns:
            Dictionary of reports
        """
        reports = {}
        summary_lines = ['Labeling Summary', '=' * 40, f"Total samples: {stats['total_samples']:,}", f"Labeled samples: {stats['labeled_samples']:,}", f"Labeling rate: {stats['labeled_samples'] / stats['total_samples'] * 100:.1f}%", '', 'Label Columns:']
        for label_col in stats['label_columns']:
            if f'{label_col}_metrics' in stats:
                metrics = stats[f'{label_col}_metrics']
                summary_lines.extend([f'\n{label_col}:', f"  Unique values: {metrics['unique_values']}", f"  Most common: {metrics['most_common']} ({metrics['most_common_pct']:.1f}%)", f"  Entropy: {metrics['entropy']:.3f}"])
        reports['summary'] = '\n'.join(summary_lines)
        config_lines = ['Labeling Configuration', '=' * 40, json.dumps(self.labeling_config, indent=2)]
        reports['configuration'] = '\n'.join(config_lines)
        return reports

    async def _save_outputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> None:
        """Save step outputs to disk.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Pipeline state with results
        """
        output_dir = Path(training_input.get('output_dir', 'output')) / 'step05_labeling'
        output_dir.mkdir(parents=True, exist_ok=True)
        if 'labeled_data' in pipeline_state:
            file_path = output_dir / 'labeled_data.parquet'
            pipeline_state['labeled_data'].to_parquet(file_path)
            self.logger.info(f'💾 Saved labeled data to {file_path}')
        if 'label_statistics' in pipeline_state:
            stats_path = output_dir / 'label_statistics.json'
            with open(stats_path, 'w') as f:
                json.dump(pipeline_state['label_statistics'], f, indent=2)
            self.logger.info(f'💾 Saved label statistics to {stats_path}')
        if 'labeling_reports' in pipeline_state:
            for report_name, content in pipeline_state['labeling_reports'].items():
                report_path = output_dir / f'{report_name}_report.txt'
                with open(report_path, 'w') as f:
                    f.write(content)
                self.logger.info(f'💾 Saved {report_name} report')

    def get_required_inputs(self) -> list:
        """Get list of required inputs for this step."""
        return ['unified_data or split data (train_data, val_data, test_data)']

    def get_produced_outputs(self) -> list:
        """Get list of outputs produced by this step."""
        return ['labeled_data', 'label_statistics', 'labeling_reports', 'updated split data with labels']

    def get_dependencies(self) -> list:
        """Get list of step dependencies."""
        return ['04_regime_data_splitting']