"""
Comprehensive Duplicate Analyzer

This module provides advanced duplicate detection and analysis capabilities
for time series data, specifically designed for financial data quality validation.

Features:
- True vs False duplicate detection
- Mixed duplicate analysis
- Comprehensive reporting and recommendations
- Safe duplicate resolution strategies
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter

@dataclass
class DuplicateAnalysisResult:
    """Result of comprehensive duplicate analysis."""
    total_duplicates: int = 0
    duplicate_groups: int = 0
    true_duplicate_groups: int = 0
    false_duplicate_groups: int = 0
    mixed_duplicate_groups: int = 0
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    duplicate_details: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    analysis_timestamp: datetime = field(default_factory=datetime.now)

class ComprehensiveDuplicateAnalyzer:
    """
    Comprehensive duplicate analyzer for time series data.

    This analyzer distinguishes between different types of duplicates:
    - True duplicates: Identical records (safe to remove)
    - False duplicates: Same timestamp, different values (requires investigation)
    - Mixed duplicates: Combination of true and false duplicates
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the comprehensive duplicate analyzer."""
        self.logger = logger or logging.getLogger(__name__)
        self.analysis_config = {
            'timestamp_tolerance_ms': 1,  # 1ms tolerance for timestamp comparison
            'value_tolerance': 1e-10,     # Tolerance for numeric value comparison
            'max_duplicate_groups_to_analyze': 1000,  # Limit for performance
            'enable_detailed_analysis': True
        }

    def analyze_duplicates(self, df: pd.DataFrame, timestamp_column: str = 'timestamp') -> DuplicateAnalysisResult:
        """
        Perform comprehensive duplicate analysis on a DataFrame.

        Args:
            df: DataFrame to analyze
            timestamp_column: Name of the timestamp column

        Returns:
            DuplicateAnalysisResult with detailed analysis
        """
        if df is None or df.empty:
            return DuplicateAnalysisResult()

        if timestamp_column not in df.columns:
            self.logger.warning(f"Timestamp column '{timestamp_column}' not found in DataFrame")
            return DuplicateAnalysisResult()

        self.logger.info(f"🔍 Starting comprehensive duplicate analysis on {len(df)} records")

        # Find duplicate timestamps
        duplicate_timestamps = self._find_duplicate_timestamps(df, timestamp_column)

        if duplicate_timestamps.empty:
            self.logger.info("✅ No duplicate timestamps found")
            return DuplicateAnalysisResult()

        # Analyze duplicate groups
        duplicate_groups = self._group_duplicates_by_timestamp(df, duplicate_timestamps, timestamp_column)

        # Classify duplicate groups
        classified_groups = self._classify_duplicate_groups(duplicate_groups)

        # Generate comprehensive results
        result = self._generate_analysis_result(classified_groups, df, timestamp_column)

        self.logger.info(f"✅ Duplicate analysis completed: {result.total_duplicates} duplicates in {result.duplicate_groups} groups")

        return result

    def _find_duplicate_timestamps(self, df: pd.DataFrame, timestamp_column: str) -> pd.Series:
        """Find duplicate timestamps in the DataFrame."""
        timestamps = df[timestamp_column]

        # Handle different timestamp formats
        if pd.api.types.is_datetime64_any_dtype(timestamps):
            # Convert to milliseconds for consistent comparison
            timestamps_ms = timestamps.astype('int64') // 10**6
        else:
            # Assume already in milliseconds
            timestamps_ms = timestamps

        # Find duplicates
        duplicate_mask = timestamps_ms.duplicated(keep=False)
        return timestamps_ms[duplicate_mask]

    def _group_duplicates_by_timestamp(self, df: pd.DataFrame, duplicate_timestamps: pd.Series,
                                     timestamp_column: str) -> Dict[Any, pd.DataFrame]:
        """Group duplicate records by their timestamp efficiently."""
        # 1. Convert duplicate_timestamps (which are in ms if datetime) to the original format if needed
        unique_timestamps = duplicate_timestamps.unique()
        
        # 2. Extract only the rows that have duplicates to minimize search space
        if pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            # Use milliseconds for matching if the source was converted
            source_timestamps_ms = df[timestamp_column].astype('int64') // 10**6
            duplicate_mask = source_timestamps_ms.isin(unique_timestamps)
            temp_df = df[duplicate_mask].copy()
            temp_df['_match_ts'] = source_timestamps_ms
            match_col = '_match_ts'
        else:
            duplicate_mask = df[timestamp_column].isin(unique_timestamps)
            temp_df = df[duplicate_mask].copy()
            match_col = timestamp_column

        # 3. Group by the match column
        groups = {}
        for ts, group_df in temp_df.groupby(match_col):
            groups[ts] = group_df.copy()
            if match_col == '_match_ts':
                groups[ts] = groups[ts].drop(columns=['_match_ts'])

        return groups

    def _classify_duplicate_groups(self, duplicate_groups: Dict[Any, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
        """Classify duplicate groups into true, false, and mixed duplicates."""
        classified = {
            'true_duplicates': [],
            'false_duplicates': [],
            'mixed_duplicates': []
        }

        for timestamp, group_df in duplicate_groups.items():
            group_info = {
                'timestamp': timestamp,
                'record_count': len(group_df),
                'records': group_df.to_dict('records')
            }

            # Check if all records are identical (excluding timestamp)
            data_columns = [col for col in group_df.columns if col != 'timestamp']
            if not data_columns:
                # Only timestamp column - treat as true duplicate
                classified['true_duplicates'].append(group_info)
                continue

            # Compare records
            first_record = group_df.iloc[0][data_columns]
            all_identical = True

            # Pre-filter numeric columns to avoid TypeError in np.isclose
            numeric_cols = [col for col in data_columns if pd.api.types.is_numeric_dtype(group_df[col])]
            non_numeric_cols = [col for col in data_columns if col not in numeric_cols]

            for idx in range(1, len(group_df)):
                current_record = group_df.iloc[idx][data_columns]

                # Compare numeric columns with tolerance
                for col in numeric_cols:
                    val1 = first_record[col]
                    val2 = current_record[col]
                    
                    # Handle None/NaN explicitly
                    if pd.isna(val1) and pd.isna(val2):
                        continue
                    if pd.isna(val1) or pd.isna(val2):
                        all_identical = False
                        break
                        
                    if not np.isclose(float(val1), float(val2),
                                    rtol=self.analysis_config['value_tolerance']):
                        all_identical = False
                        break
                
                if not all_identical:
                    break

                # Compare non-numeric columns
                for col in non_numeric_cols:
                    if first_record[col] != current_record[col]:
                        all_identical = False
                        break
                
                if not all_identical:
                    break

            if all_identical:
                classified['true_duplicates'].append(group_info)
            else:
                # Optimized mixed check: skip O(K^2) for very large groups
                if len(group_df) > 50:
                    classified['false_duplicates'].append(group_info)
                    continue

                # Check if it's mixed (some identical, some different)
                identical_pairs = 0
                total_pairs = 0

                for i in range(len(group_df)):
                    for j in range(i + 1, len(group_df)):
                        total_pairs += 1
                        record1 = group_df.iloc[i][data_columns]
                        record2 = group_df.iloc[j][data_columns]

                        for col in data_columns:
                            val1 = record1[col]
                            val2 = record2[col]
                            
                            # Universal NaN/None check first to prevent "boolean value of NA is ambiguous"
                            # This handles pd.NA, np.nan, None safely
                            is_val1_na = pd.isna(val1)
                            is_val2_na = pd.isna(val2)
                            
                            if is_val1_na and is_val2_na:
                                # Both are NaN/None -> they are equal
                                continue
                            
                            if is_val1_na or is_val2_na:
                                # One is NaN/None, other is not -> they are different
                                pair_identical = False
                                break

                            # Now we know both are not NaN/None
                            is_numeric_col = pd.api.types.is_numeric_dtype(group_df[col])
                            is_val1_num = isinstance(val1, (int, float, np.number))
                            is_val2_num = isinstance(val2, (int, float, np.number))

                            if is_numeric_col and is_val1_num and is_val2_num:
                                try:
                                    if not np.isclose(val1, val2, rtol=self.analysis_config['value_tolerance']):
                                        pair_identical = False
                                        break
                                except (TypeError, ValueError):
                                    # Fallback to direct comparison if isclose fails
                                    if val1 != val2:
                                        pair_identical = False
                                        break
                            else:
                                # Non-numeric comparison or mixed types
                                # Since we checked for NA above, standard equality should be safe(r)
                                # but strictly speaking pd.NA could still sneak in if pd.isna() false neg?
                                # (unlikely for pd.NA, but safe equality is better)
                                if val1 != val2:
                                    pair_identical = False
                                    break

                        if pair_identical:
                            identical_pairs += 1

                if identical_pairs > 0 and identical_pairs < total_pairs:
                    classified['mixed_duplicates'].append(group_info)
                else:
                    classified['false_duplicates'].append(group_info)

        return classified

    def _generate_analysis_result(self, classified_groups: Dict[str, List[Dict[str, Any]]],
                                df: pd.DataFrame, timestamp_column: str) -> DuplicateAnalysisResult:
        """Generate comprehensive analysis result."""
        result = DuplicateAnalysisResult()

        # Count duplicates
        result.true_duplicate_groups = len(classified_groups['true_duplicates'])
        result.false_duplicate_groups = len(classified_groups['false_duplicates'])
        result.mixed_duplicate_groups = len(classified_groups['mixed_duplicates'])
        result.duplicate_groups = (result.true_duplicate_groups +
                                 result.false_duplicate_groups +
                                 result.mixed_duplicate_groups)

        # Calculate total duplicate records
        total_duplicates = 0
        for group_type in classified_groups.values():
            for group in group_type:
                total_duplicates += group['record_count'] - 1  # -1 because one record is not a duplicate

        result.total_duplicates = total_duplicates

        # Generate summary statistics
        result.summary_stats = {
            'total_records': len(df),
            'duplicate_percentage': (total_duplicates / len(df)) * 100 if len(df) > 0 else 0,
            'duplicate_type_distribution': {
                'true_duplicates': result.true_duplicate_groups,
                'false_duplicates': result.false_duplicate_groups,
                'mixed_duplicates': result.mixed_duplicate_groups
            },
            'average_group_size': {
                'true_duplicates': self._calculate_average_group_size(classified_groups['true_duplicates']),
                'false_duplicates': self._calculate_average_group_size(classified_groups['false_duplicates']),
                'mixed_duplicates': self._calculate_average_group_size(classified_groups['mixed_duplicates'])
            }
        }

        # Generate detailed analysis (limited for performance)
        if self.analysis_config['enable_detailed_analysis']:
            result.duplicate_details = self._generate_detailed_analysis(classified_groups)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(classified_groups, result.summary_stats)

        return result

    def _calculate_average_group_size(self, groups: List[Dict[str, Any]]) -> float:
        """Calculate average group size for a type of duplicate groups."""
        if not groups:
            return 0.0

        total_records = sum(group['record_count'] for group in groups)
        return total_records / len(groups)

    def _generate_detailed_analysis(self, classified_groups: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Generate detailed analysis of duplicate groups (limited for performance)."""
        details = []
        max_groups = self.analysis_config['max_duplicate_groups_to_analyze']

        for group_type, groups in classified_groups.items():
            for i, group in enumerate(groups[:max_groups]):
                # Safely get sample records
                records = group.get('records', [])
                sample_records = records[:3] if isinstance(records, list) else []

                detail = {
                    'type': group_type,
                    'timestamp': group['timestamp'],
                    'record_count': group['record_count'],
                    'sample_records': sample_records
                }
                details.append(detail)

        return details

    def _generate_recommendations(self, classified_groups: Dict[str, List[Dict[str, Any]]],
                                summary_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on duplicate analysis."""
        recommendations = []

        total_groups = sum(len(groups) for groups in classified_groups.values())
        duplicate_percentage = summary_stats.get('duplicate_percentage', 0)

        if duplicate_percentage > 10:
            recommendations.append("High duplicate percentage detected - investigate data collection process")
        elif duplicate_percentage > 5:
            recommendations.append("Moderate duplicate percentage - consider implementing deduplication")

        if classified_groups['false_duplicates']:
            recommendations.append("False duplicates detected - same timestamp with different values requires investigation")

        if classified_groups['mixed_duplicates']:
            recommendations.append("Mixed duplicates detected - combination of true and false duplicates requires careful analysis")

        if classified_groups['true_duplicates']:
            recommendations.append("True duplicates detected - safe to remove duplicate records")

        if not recommendations:
            recommendations.append("No significant duplicate issues detected")

        return recommendations

    def resolve_duplicates(self, df: pd.DataFrame, strategy: str = 'manual_review',
                         timestamp_column: str = 'timestamp') -> pd.DataFrame:
        """
        Resolve duplicates using specified strategy.

        Note: This method only supports 'manual_review' strategy for safety.
        Automatic resolution is disabled to prevent data loss.

        Args:
            df: DataFrame to process
            strategy: Resolution strategy (only 'manual_review' supported)
            timestamp_column: Name of the timestamp column

        Returns:
            DataFrame with duplicates resolved (or original if manual_review)
        """
        if strategy != 'manual_review':
            raise ValueError("Only 'manual_review' strategy is supported. Automatic resolution is disabled for safety.")

        self.logger.warning("Manual review strategy selected - no automatic resolution performed")
        self.logger.info("Please review duplicate analysis results and implement appropriate resolution manually")

        return df.copy()

# Convenience functions for easy usage
def analyze_duplicates_comprehensive(df: pd.DataFrame, timestamp_column: str = 'timestamp') -> DuplicateAnalysisResult:
    """Convenience function for comprehensive duplicate analysis."""
    analyzer = ComprehensiveDuplicateAnalyzer()
    return analyzer.analyze_duplicates(df, timestamp_column)

def get_duplicate_summary(df: pd.DataFrame, timestamp_column: str = 'timestamp') -> Dict[str, Any]:
    """Get a quick summary of duplicates in the DataFrame."""
    result = analyze_duplicates_comprehensive(df, timestamp_column)

    return {
        'has_duplicates': result.total_duplicates > 0,
        'total_duplicates': result.total_duplicates,
        'duplicate_groups': result.duplicate_groups,
        'true_duplicates': result.true_duplicate_groups,
        'false_duplicates': result.false_duplicate_groups,
        'mixed_duplicates': result.mixed_duplicate_groups,
        'duplicate_percentage': result.summary_stats.get('duplicate_percentage', 0),
        'recommendations': result.recommendations
    }
