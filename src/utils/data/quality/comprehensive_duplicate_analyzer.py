"""
Comprehensive Duplicate Timestamp Analyzer

This module provides advanced duplicate timestamp detection and analysis that distinguishes
between true duplicates (identical records) and false duplicates (same timestamp, different data).

Features:
- True duplicate detection: identical timestamp + identical values across columns
- False duplicate detection: identical timestamp + different values across columns
- Conflict resolution strategies
- Detailed reporting and analytics
- Integration with existing data quality pipelines
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

from src.utils.logger import system_logger


@dataclass
class DuplicateGroup:
    """Represents a group of duplicate records for a single timestamp."""
    timestamp: Any
    record_count: int
    records: pd.DataFrame
    duplicate_type: str  # 'true_duplicates', 'false_duplicates', 'mixed'
    conflict_columns: List[str] = field(default_factory=list)
    conflict_summary: Dict[str, Any] = field(default_factory=dict)
    resolution_strategy: Optional[str] = None
    resolved_record: Optional[pd.Series] = None


@dataclass
class DuplicateAnalysisResult:
    """Comprehensive result of duplicate analysis."""
    total_duplicates: int
    true_duplicate_groups: int
    false_duplicate_groups: int
    mixed_duplicate_groups: int
    duplicate_groups: List[DuplicateGroup]
    summary_stats: Dict[str, Any]
    recommendations: List[str]
    analysis_timestamp: datetime = field(default_factory=datetime.now)


class ComprehensiveDuplicateAnalyzer:
    """Advanced duplicate timestamp analyzer with true/false duplicate distinction."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the duplicate analyzer."""
        self.logger = logger or system_logger.getChild('ComprehensiveDuplicateAnalyzer')
        self.analysis_results: Optional[DuplicateAnalysisResult] = None

        # Define conflict resolution strategies - MANUAL REVIEW ONLY
        self.resolution_strategies = {
            'manual_review': self._resolve_by_manual_review
        }

        # Define columns that should be identical for true duplicates
        self.comparison_columns = [
            'open', 'high', 'low', 'close', 'volume', 'quote_volume',
            'trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        ]

        # Define metadata columns (these can legitimately differ)
        self.metadata_columns = [
            'timestamp', 'exchange', 'symbol', 'timeframe', 'source',
            'collection_method', 'version', 'fetch_timestamp'
        ]

    def analyze_duplicates(self, df: pd.DataFrame, timestamp_column: str = 'timestamp') -> DuplicateAnalysisResult:
        """
        Perform comprehensive duplicate analysis.

        Args:
            df: DataFrame to analyze
            timestamp_column: Name of timestamp column

        Returns:
            Comprehensive analysis results
        """
        self.logger.info(f"🔍 Starting comprehensive duplicate analysis on {len(df)} records")

        # Check if timestamp is available as column or index
        if timestamp_column not in df.columns:
            # Check if timestamp is the index
            if hasattr(df.index, 'name') and df.index.name == timestamp_column:
                self.logger.info(f"🔄 Using timestamp from index for duplicate analysis")
                # Reset index to make timestamp a column for analysis
                df_for_analysis = df.reset_index()
                timestamp_column_for_analysis = timestamp_column
            else:
                raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame columns or index")
        else:
            df_for_analysis = df
            timestamp_column_for_analysis = timestamp_column

        # Find duplicate timestamps
        duplicate_mask = df_for_analysis[timestamp_column_for_analysis].duplicated(keep=False)
        duplicate_records = df_for_analysis[duplicate_mask]

        if duplicate_records.empty:
            self.logger.info("✅ No duplicate timestamps found")
            return DuplicateAnalysisResult(
                total_duplicates=0,
                true_duplicate_groups=0,
                false_duplicate_groups=0,
                mixed_duplicate_groups=0,
                duplicate_groups=[],
                summary_stats={},
                recommendations=[]
            )

        # Group by timestamp and analyze each group
        duplicate_groups = []
        true_duplicate_groups = 0
        false_duplicate_groups = 0
        mixed_duplicate_groups = 0

        for timestamp, group in duplicate_records.groupby(timestamp_column_for_analysis):
            duplicate_group = self._analyze_duplicate_group(timestamp, group)
            duplicate_groups.append(duplicate_group)

            if duplicate_group.duplicate_type == 'true_duplicates':
                true_duplicate_groups += 1
            elif duplicate_group.duplicate_type == 'false_duplicates':
                false_duplicate_groups += 1
            else:  # mixed
                mixed_duplicate_groups += 1

        # Generate summary statistics
        summary_stats = self._generate_summary_stats(duplicate_groups, duplicate_records)

        # Generate recommendations
        recommendations = self._generate_recommendations(duplicate_groups, summary_stats)

        result = DuplicateAnalysisResult(
            total_duplicates=len(duplicate_records),
            true_duplicate_groups=true_duplicate_groups,
            false_duplicate_groups=false_duplicate_groups,
            mixed_duplicate_groups=mixed_duplicate_groups,
            duplicate_groups=duplicate_groups,
            summary_stats=summary_stats,
            recommendations=recommendations
        )

        self.analysis_results = result
        self.logger.info(f"✅ Duplicate analysis completed: {result.total_duplicates} duplicates in {len(duplicate_groups)} groups")
        return result

    def _analyze_duplicate_group(self, timestamp: Any, group: pd.DataFrame) -> DuplicateGroup:
        """
        Analyze a single group of duplicate records.

        Args:
            timestamp: The duplicate timestamp
            group: DataFrame containing all records for this timestamp

        Returns:
            Analyzed DuplicateGroup
        """
        record_count = len(group)

        # Check for conflicts in comparison columns
        conflict_columns = []
        conflict_summary = {}

        # Compare all records to the first one
        first_record = group.iloc[0]
        all_identical = True

        for col in self.comparison_columns:
            if col in group.columns:
                values = group[col].values
                if not np.all(values == values[0]):  # Check if all values are identical
                    conflict_columns.append(col)
                    conflict_summary[col] = {
                        'unique_values': len(np.unique(values)),
                        'value_range': [float(np.min(values)), float(np.max(values))],
                        'std_dev': float(np.std(values)) if len(values) > 1 else 0.0
                    }
                    all_identical = False

        # Determine duplicate type
        if all_identical:
            duplicate_type = 'true_duplicates'
        elif conflict_columns:
            duplicate_type = 'false_duplicates'
        else:
            duplicate_type = 'mixed'  # Conflicts in non-comparison columns only

        return DuplicateGroup(
            timestamp=timestamp,
            record_count=record_count,
            records=group.copy(),
            duplicate_type=duplicate_type,
            conflict_columns=conflict_columns,
            conflict_summary=conflict_summary
        )

    def _generate_summary_stats(self, duplicate_groups: List[DuplicateGroup],
                               duplicate_records: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""

        if not duplicate_groups:
            return {}

        stats = {
            'total_duplicate_groups': len(duplicate_groups),
            'total_duplicate_records': sum(g.record_count for g in duplicate_groups),
            'avg_records_per_group': np.mean([g.record_count for g in duplicate_groups]),
            'max_records_in_group': max(g.record_count for g in duplicate_groups),
            'duplicate_type_distribution': {
                'true_duplicates': sum(1 for g in duplicate_groups if g.duplicate_type == 'true_duplicates'),
                'false_duplicates': sum(1 for g in duplicate_groups if g.duplicate_type == 'false_duplicates'),
                'mixed': sum(1 for g in duplicate_groups if g.duplicate_type == 'mixed')
            }
        }

        # Conflict analysis
        all_conflict_columns = []
        for group in duplicate_groups:
            all_conflict_columns.extend(group.conflict_columns)

        if all_conflict_columns:
            from collections import Counter
            conflict_counts = Counter(all_conflict_columns)
            stats['most_conflicted_columns'] = dict(conflict_counts.most_common(5))
            stats['total_conflict_columns'] = len(set(all_conflict_columns))
        else:
            stats['most_conflicted_columns'] = {}
            stats['total_conflict_columns'] = 0

        # Volume analysis for resolution insights
        if 'volume' in duplicate_records.columns:
            volume_stats = duplicate_records['volume'].describe()
            stats['volume_statistics'] = {
                'mean': float(volume_stats['mean']),
                'std': float(volume_stats['std']),
                'min': float(volume_stats['min']),
                'max': float(volume_stats['max'])
            }

        return stats

    def _generate_recommendations(self, duplicate_groups: List[DuplicateGroup],
                                summary_stats: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""

        recommendations = []

        if not duplicate_groups:
            return recommendations

        # True duplicates - safe to remove
        true_dup_count = sum(1 for g in duplicate_groups if g.duplicate_type == 'true_duplicates')
        if true_dup_count > 0:
            recommendations.append(
                f"SAFE: Remove {true_dup_count} groups of true duplicates "
                f"({sum(g.record_count for g in duplicate_groups if g.duplicate_type == 'true_duplicates')} total records)"
            )

        # False duplicates - need investigation
        false_dup_count = sum(1 for g in duplicate_groups if g.duplicate_type == 'false_duplicates')
        if false_dup_count > 0:
            conflict_cols = set()
            for g in duplicate_groups:
                if g.duplicate_type == 'false_duplicates':
                    conflict_cols.update(g.conflict_columns)

            recommendations.append(
                f"INVESTIGATE: Review {false_dup_count} groups of false duplicates. "
                f"Conflicts in columns: {', '.join(conflict_cols)}. "
                "Consider data source quality and collection methods."
            )

        # High volume groups
        high_volume_groups = [g for g in duplicate_groups if g.record_count > 5]
        if high_volume_groups:
            recommendations.append(
                f"PRIORITY: {len(high_volume_groups)} duplicate groups have >5 records each. "
                "Manual review recommended for these high-frequency duplicates."
            )

        # Volume-based resolution suggestion
        if 'volume_statistics' in summary_stats:
            vol_std = summary_stats['volume_statistics']['std']
            vol_mean = summary_stats['volume_statistics']['mean']
            cv = vol_std / vol_mean if vol_mean > 0 else 0

            if cv > 0.5:  # High volume variation
                recommendations.append(
                    f"VOLUME-BASED: High volume variation detected (CV={cv:.2f}). "
                    "Consider 'highest_volume' resolution strategy for conflicting duplicates."
                )

        return recommendations

    def resolve_duplicates(self, df: pd.DataFrame, strategy: str = 'manual_review',
                          timestamp_column: str = 'timestamp') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Resolve duplicates using manual review strategy only.

        Args:
            df: DataFrame with duplicates
            strategy: Resolution strategy (only 'manual_review' supported)
            timestamp_column: Name of timestamp column

        Returns:
            Tuple of (original_df, resolution_summary) - NO automatic resolution
        """
        if not self.analysis_results:
            self.analyze_duplicates(df, timestamp_column)

        self.logger.info(f"🔧 Processing duplicates using strategy: {strategy}")

        if strategy not in self.resolution_strategies:
            raise ValueError(f"Only 'manual_review' strategy is supported. Got: {strategy}")

        resolver = self.resolution_strategies[strategy]
        resolution_summary = {
            'strategy': strategy,
            'groups_processed': 0,
            'records_flagged': 0,
            'manual_review_needed': [],
            'duplicate_analysis': {
                'total_duplicates': self.analysis_results.total_duplicates,
                'duplicate_groups': len(self.analysis_results.duplicate_groups),
                'true_duplicates': self.analysis_results.true_duplicate_groups,
                'false_duplicates': self.analysis_results.false_duplicate_groups,
                'mixed_duplicates': self.analysis_results.mixed_duplicate_groups
            }
        }

        # Process each duplicate group - NO automatic resolution
        for group in self.analysis_results.duplicate_groups:
            _, resolution_info = resolver(group)

            # Always flag for manual review
            resolution_summary['manual_review_needed'].append({
                'timestamp': group.timestamp,
                'record_count': group.record_count,
                'duplicate_type': group.duplicate_type,
                'conflict_columns': group.conflict_columns,
                'reason': resolution_info or f'{group.duplicate_type} requires manual review'
            })

            resolution_summary['groups_processed'] += 1
            resolution_summary['records_flagged'] += group.record_count

        # Return ORIGINAL dataframe - NO changes made
        self.logger.info(f"✅ Duplicate processing completed: {resolution_summary['records_flagged']} records flagged for manual review")
        self.logger.warning("⚠️ MANUAL REVIEW REQUIRED: All duplicate records have been flagged for manual inspection")
        self.logger.warning("⚠️ No automatic resolution was performed - original data preserved")

        return df, resolution_summary

    # Resolution Strategies - ONLY MANUAL REVIEW IS ACTIVE

    # COMMENTED OUT: Automatic resolution strategies disabled
    # def _resolve_by_highest_volume(self, group: DuplicateGroup) -> Tuple[Optional[pd.Series], str]:
    #     """Resolve by keeping record with highest volume."""
    #     if 'volume' not in group.records.columns:
    #         return None, "Volume column not available for resolution"
    #
    #     max_volume_idx = group.records['volume'].idxmax()
    #     resolved_record = group.records.loc[max_volume_idx]
    #
    #     return resolved_record, f"Kept record with highest volume ({resolved_record['volume']})"
    #
    # def _resolve_by_latest_record(self, group: DuplicateGroup) -> Tuple[Optional[pd.Series], str]:
    #     """Resolve by keeping the most recently collected record."""
    #     # Look for collection timestamp or use last record as proxy
    #     if 'fetch_timestamp' in group.records.columns:
    #         latest_idx = group.records['fetch_timestamp'].idxmax()
    #         resolved_record = group.records.loc[latest_idx]
    #         return resolved_record, f"Kept most recent record (fetch_timestamp: {resolved_record['fetch_timestamp']})"
    #     else:
    #         # Use last record in group as proxy for most recent
    #         resolved_record = group.records.iloc[-1]
    #         return resolved_record, "Kept last record in group (proxy for most recent)"
    #
    # def _resolve_by_most_complete(self, group: DuplicateGroup) -> Tuple[Optional[pd.Series], str]:
    #     """Resolve by keeping record with fewest NaN values."""
    #     nan_counts = group.records.isnull().sum(axis=1)
    #     min_nan_idx = nan_counts.idxmin()
    #     resolved_record = group.records.loc[min_nan_idx]
    #
    #     nan_count = nan_counts.loc[min_nan_idx]
    #     total_cols = len(group.records.columns)
    #
    #     return resolved_record, f"Kept most complete record ({total_cols - nan_count}/{total_cols} non-null values)"
    #
    # def _resolve_by_average_values(self, group: DuplicateGroup) -> Tuple[Optional[pd.Series], str]:
    #     """Resolve by averaging numeric values across duplicate records."""
    #     resolved_record = group.records.iloc[0].copy()
    #
    #     # Average numeric columns
    #     numeric_columns = group.records.select_dtypes(include=[np.number]).columns
    #     for col in numeric_columns:
    #         if col != 'timestamp':  # Don't average timestamps
    #         resolved_record[col] = group.records[col].mean()
    #
    #     return resolved_record, f"Averaged values across {group.record_count} records"

    def _resolve_by_manual_review(self, group: DuplicateGroup) -> Tuple[Optional[pd.Series], str]:
        """Flag for manual review - don't resolve automatically."""
        return None, f"Flagged for manual review: {group.duplicate_type} with {len(group.conflict_columns)} conflicting columns"

    def generate_report(self, analysis_result: Optional[DuplicateAnalysisResult] = None) -> str:
        """Generate a comprehensive text report of duplicate analysis."""
        if analysis_result is None:
            analysis_result = self.analysis_results

        if not analysis_result:
            return "No duplicate analysis results available"

        report_lines = [
            "=" * 80,
            "COMPREHENSIVE DUPLICATE TIMESTAMP ANALYSIS REPORT",
            "=" * 80,
            f"Analysis Date: {analysis_result.analysis_timestamp}",
            "",
            "SUMMARY STATISTICS:",
            f"  Total Duplicate Records: {analysis_result.total_duplicates:,}",
            f"  Duplicate Groups: {len(analysis_result.duplicate_groups):,}",
            f"  True Duplicate Groups: {analysis_result.true_duplicate_groups:,}",
            f"  False Duplicate Groups: {analysis_result.false_duplicate_groups:,}",
            f"  Mixed Duplicate Groups: {analysis_result.mixed_duplicate_groups:,}",
            ""
        ]

        if analysis_result.summary_stats:
            stats = analysis_result.summary_stats
            report_lines.extend([
                "DETAILED STATISTICS:",
                f"  Average Records per Group: {stats.get('avg_records_per_group', 0):.1f}",
                f"  Maximum Records in Group: {stats.get('max_records_in_group', 0)}",
                f"  Total Conflict Columns: {stats.get('total_conflict_columns', 0)}",
                ""
            ])

            if stats.get('most_conflicted_columns'):
                report_lines.append("MOST CONFLICTED COLUMNS:")
                for col, count in stats['most_conflicted_columns'].items():
                    report_lines.append(f"  {col}: {count} conflicts")
                report_lines.append("")

        if analysis_result.recommendations:
            report_lines.extend([
                "RECOMMENDATIONS:",
                *[f"  • {rec}" for rec in analysis_result.recommendations],
                ""
            ])

        # Sample duplicate groups
        if analysis_result.duplicate_groups:
            report_lines.extend([
                "SAMPLE DUPLICATE GROUPS:",
                f"Showing first {min(5, len(analysis_result.duplicate_groups))} groups:",
                ""
            ])

            for i, group in enumerate(analysis_result.duplicate_groups[:5]):
                report_lines.extend([
                    f"Group {i+1}:",
                    f"  Timestamp: {group.timestamp}",
                    f"  Records: {group.record_count}",
                    f"  Type: {group.duplicate_type}",
                    f"  Conflicts: {', '.join(group.conflict_columns) if group.conflict_columns else 'None'}",
                    ""
                ])

        report_lines.extend(["=" * 80])
        return "\n".join(report_lines)


# Convenience functions for easy integration

def analyze_duplicates_comprehensive(df: pd.DataFrame,
                                   timestamp_column: str = 'timestamp') -> DuplicateAnalysisResult:
    """Convenience function for comprehensive duplicate analysis."""
    analyzer = ComprehensiveDuplicateAnalyzer()
    return analyzer.analyze_duplicates(df, timestamp_column)


def resolve_duplicates_comprehensive(df: pd.DataFrame,
                                   strategy: str = 'highest_volume',
                                   timestamp_column: str = 'timestamp') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience function for duplicate resolution."""
    analyzer = ComprehensiveDuplicateAnalyzer()
    return analyzer.resolve_duplicates(df, strategy, timestamp_column)


def generate_duplicate_report(df: pd.DataFrame,
                            timestamp_column: str = 'timestamp') -> str:
    """Convenience function to generate duplicate analysis report."""
    analyzer = ComprehensiveDuplicateAnalyzer()
    analysis = analyzer.analyze_duplicates(df, timestamp_column)
    return analyzer.generate_report(analysis)


if __name__ == "__main__":
    # Example usage
    print("Comprehensive Duplicate Timestamp Analyzer")
    print("Run with: python -m src.utils.data.quality.comprehensive_duplicate_analyzer")
