"""Data quality fixer for handling common data issues."""
import pandas as pd

from typing import Dict, Any, Optional, Tuple
from .logger import system_logger

from .logger import system_logger
import logging
import numpy as np

class DataQualityFixer:
    """Handles common data quality issues in trading data."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild('DataQualityFixer')

    def fix_data_quality_issues(self, data: pd.DataFrame, timestamp_column: str='timestamp') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fix common data quality issues in the DataFrame.
        
        Args:
            data: DataFrame to fix
            timestamp_column: Name of the timestamp column
            
        Returns:
            Tuple of (fixed_data, fix_report)
        """
        fix_report = {'original_rows': len(data), 'duplicates_removed': 0, 'index_fixed': False, 'timestamp_converted': False, 'final_rows': 0, 'issues_fixed': []}
        self.logger.info(f'🔧 Starting data quality fixes for {len(data)} rows')
        fixed_data = data.copy()
        if timestamp_column in fixed_data.columns:
            fixed_data, timestamp_fixed = self._fix_timestamp_column(fixed_data, timestamp_column)
            if timestamp_fixed:
                fix_report['timestamp_converted'] = True
                fix_report['issues_fixed'].append('timestamp_converted')
        if timestamp_column in fixed_data.columns:
            original_count = len(fixed_data)
            fixed_data = self._remove_duplicate_timestamps(fixed_data, timestamp_column)
            duplicates_removed = original_count - len(fixed_data)
            fix_report['duplicates_removed'] = duplicates_removed
            if duplicates_removed > 0:
                fix_report['issues_fixed'].append(f'removed_{duplicates_removed}_duplicates')
                self.logger.info(f'🗑️ Removed {duplicates_removed} duplicate timestamps')
        if timestamp_column in fixed_data.columns:
            fixed_data, index_fixed = self._fix_non_monotonic_index(fixed_data, timestamp_column)
            if index_fixed:
                fix_report['index_fixed'] = True
                fix_report['issues_fixed'].append('index_sorted')
                self.logger.info('📈 Fixed non-monotonic timestamp index')
        if timestamp_column in fixed_data.columns:
            fixed_data = self._set_datetime_index(fixed_data, timestamp_column)
        fix_report['final_rows'] = len(fixed_data)
        self.logger.info(f"✅ Data quality fixes completed: {fix_report['original_rows']} → {fix_report['final_rows']} rows")
        self.logger.info(f"🔧 Issues fixed: {', '.join(fix_report['issues_fixed'])}")
        return (fixed_data, fix_report)

    def _fix_timestamp_column(self, data: pd.DataFrame, timestamp_column: str) -> Tuple[pd.DataFrame, bool]:
        """Fix timestamp column format."""
        if not pd.api.types.is_datetime64_any_dtype(data[timestamp_column]):
            try:
                data[timestamp_column] = pd.to_datetime(data[timestamp_column])
                self.logger.info('🕒 Converted timestamp column to datetime')
                return (data, True)
            except Exception as e:
                self.logger.warning(f'⚠️ Could not convert timestamp column: {e}')
                return (data, False)
        return (data, False)

    def _remove_duplicate_timestamps(self, data: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
        """Remove duplicate timestamps, keeping the last occurrence."""
        original_count = len(data)
        data = data.drop_duplicates(subset=[timestamp_column], keep='last')
        removed_count = original_count - len(data)
        if removed_count > 0:
            self.logger.info(f'🗑️ Removed {removed_count} duplicate timestamps')
        return data

    def _fix_non_monotonic_index(self, data: pd.DataFrame, timestamp_column: str) -> Tuple[pd.DataFrame, bool]:
        """Fix non-monotonic timestamp index by sorting."""
        if not data[timestamp_column].is_monotonic_increasing:
            data = data.sort_values(timestamp_column).reset_index(drop = True)
            self.logger.info('📈 Sorted data by timestamp to fix non-monotonic index')
            return (data, True)
        return (data, False)

    def _set_datetime_index(self, data: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
        """Set datetime index and remove the timestamp column."""
        if timestamp_column in data.columns:
            data = data.set_index(timestamp_column)
            self.logger.info('📅 Set datetime index')
        return data

    def validate_fixed_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate the fixed data quality."""
        validation_results = {'total_rows': len(data), 'duplicate_timestamps': 0, 'monotonic_index': True, 'datetime_index': isinstance(data.index, pd.DatetimeIndex), 'quality_score': 100.0}
        if isinstance(data.index, pd.DatetimeIndex):
            validation_results['duplicate_timestamps'] = data.index.duplicated().sum()
            validation_results['monotonic_index'] = data.index.is_monotonic_increasing
        quality_score = 100.0
        if validation_results['duplicate_timestamps'] > 0:
            quality_score -= min(20, validation_results['duplicate_timestamps'] / len(data) * 100)
        if not validation_results['monotonic_index']:
            quality_score -= 10
        if not validation_results['datetime_index']:
            quality_score -= 5
        validation_results['quality_score'] = max(0, quality_score)
        return validation_results