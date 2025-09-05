#!/usr/bin/env python3
"""
Step04 Data Merging Fixes

This module provides improved data merging strategies to address:
1. Timestamp alignment issues in regime data merging
2. Retention ratio thresholds for high-frequency data
3. Data quality validation during merging process
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

class ImprovedDataMerger:
    """Enhanced data merging with better timestamp alignment and retention validation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configurable retention thresholds based on data frequency
        self.retention_thresholds = {
            '1m': 0.95,    # 95% retention for 1-minute data
            '5m': 0.90,    # 90% retention for 5-minute data
            '15m': 0.85,   # 85% retention for 15-minute data
            '30m': 0.80,   # 80% retention for 30-minute data
            '1h': 0.75,    # 75% retention for 1-hour data
            '4h': 0.70,    # 70% retention for 4-hour data
            '1d': 0.65     # 65% retention for daily data
        }
        
        # Timestamp alignment strategies
        self.alignment_strategies = {
            'strict': 'exact_match',
            'tolerant': 'nearest_within_threshold',
            'interpolate': 'linear_interpolation',
            'forward_fill': 'forward_fill_gaps'
        }
    
    def merge_regime_data_improved(
        self, 
        unified_data: pd.DataFrame, 
        regime_data: pd.DataFrame, 
        timeframe: str,
        alignment_strategy: str = 'tolerant'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Improved regime data merging with better timestamp alignment.
        
        Args:
            unified_data: Main market data
            regime_data: Regime labels data
            timeframe: Data timeframe for retention threshold
            alignment_strategy: How to handle timestamp mismatches
            
        Returns:
            Tuple of (merged_data, merge_metadata)
        """
        self.logger.info(f"🔄 Starting improved regime data merge for {timeframe} data")
        self.logger.info(f"   Unified data: {len(unified_data)} rows")
        self.logger.info(f"   Regime data: {len(regime_data)} rows")
        self.logger.info(f"   Alignment strategy: {alignment_strategy}")
        
        # Ensure timestamp columns are datetime
        unified_data = self._standardize_timestamps(unified_data, 'timestamp')
        regime_data = self._standardize_timestamps(regime_data, 'timestamp')
        
        # Pre-merge validation
        validation_result = self._validate_pre_merge_data(unified_data, regime_data)
        if not validation_result['valid']:
            raise ValueError(f"Pre-merge validation failed: {validation_result['errors']}")
        
        # Apply alignment strategy
        if alignment_strategy == 'strict':
            merged_data = self._strict_merge(unified_data, regime_data)
        elif alignment_strategy == 'tolerant':
            merged_data = self._tolerant_merge(unified_data, regime_data, timeframe)
        elif alignment_strategy == 'interpolate':
            merged_data = self._interpolated_merge(unified_data, regime_data)
        elif alignment_strategy == 'forward_fill':
            merged_data = self._forward_fill_merge(unified_data, regime_data)
        else:
            raise ValueError(f"Unknown alignment strategy: {alignment_strategy}")
        
        # Post-merge validation
        merge_metadata = self._validate_merge_result(
            unified_data, regime_data, merged_data, timeframe
        )
        
        self.logger.info(f"✅ Merge completed with {merge_metadata['retention_ratio']:.3f} retention")
        return merged_data, merge_metadata
    
    def _standardize_timestamps(self, data: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Standardize timestamp column to datetime."""
        if timestamp_col not in data.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found")
        
        if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
            data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        
        # Sort by timestamp
        data = data.sort_values(timestamp_col).reset_index(drop=True)
        return data
    
    def _validate_pre_merge_data(
        self, 
        unified_data: pd.DataFrame, 
        regime_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate data before merging."""
        errors = []
        warnings = []
        
        # Check for required columns
        if 'composite_cluster_id' not in regime_data.columns:
            errors.append("Missing 'composite_cluster_id' in regime data")
        
        # Check for duplicate timestamps
        if unified_data['timestamp'].duplicated().any():
            warnings.append("Duplicate timestamps found in unified data")
        
        if regime_data['timestamp'].duplicated().any():
            warnings.append("Duplicate timestamps found in regime data")
        
        # Check timestamp ranges
        unified_range = (unified_data['timestamp'].min(), unified_data['timestamp'].max())
        regime_range = (regime_data['timestamp'].min(), regime_data['timestamp'].max())
        
        overlap_start = max(unified_range[0], regime_range[0])
        overlap_end = min(unified_range[1], regime_range[1])
        
        if overlap_start >= overlap_end:
            errors.append("No timestamp overlap between unified and regime data")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'unified_range': unified_range,
            'regime_range': regime_range,
            'overlap_range': (overlap_start, overlap_end)
        }
    
    def _strict_merge(
        self, 
        unified_data: pd.DataFrame, 
        regime_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Strict merge with exact timestamp matching."""
        self.logger.info("🔍 Performing strict timestamp merge")
        
        # Inner join on exact timestamp match
        merged = pd.merge(
            unified_data, 
            regime_data[['timestamp', 'composite_cluster_id']], 
            on='timestamp', 
            how='inner'
        )
        
        return merged
    
    def _tolerant_merge(
        self, 
        unified_data: pd.DataFrame, 
        regime_data: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """Tolerant merge with nearest timestamp matching within threshold."""
        self.logger.info("🔍 Performing tolerant timestamp merge")
        
        # Define tolerance based on timeframe
        tolerance_map = {
            '1m': pd.Timedelta(minutes=1),
            '5m': pd.Timedelta(minutes=5),
            '15m': pd.Timedelta(minutes=15),
            '30m': pd.Timedelta(minutes=30),
            '1h': pd.Timedelta(hours=1),
            '4h': pd.Timedelta(hours=4),
            '1d': pd.Timedelta(days=1)
        }
        
        tolerance = tolerance_map.get(timeframe, pd.Timedelta(minutes=5))
        
        # Use merge_asof for nearest timestamp matching
        merged = pd.merge_asof(
            unified_data.sort_values('timestamp'),
            regime_data[['timestamp', 'composite_cluster_id']].sort_values('timestamp'),
            on='timestamp',
            tolerance=tolerance,
            direction='nearest'
        )
        
        # Remove rows where no regime match was found
        merged = merged.dropna(subset=['composite_cluster_id'])
        
        return merged
    
    def _interpolated_merge(
        self, 
        unified_data: pd.DataFrame, 
        regime_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge with linear interpolation for regime labels."""
        self.logger.info("🔍 Performing interpolated merge")
        
        # Create a unified timestamp index
        all_timestamps = pd.concat([
            unified_data['timestamp'],
            regime_data['timestamp']
        ]).drop_duplicates().sort_values()
        
        # Reindex regime data with all timestamps
        regime_interpolated = regime_data.set_index('timestamp').reindex(all_timestamps)
        
        # Interpolate regime labels (use forward fill for categorical data)
        regime_interpolated['composite_cluster_id'] = regime_interpolated['composite_cluster_id'].fillna(method='ffill')
        
        # Merge with unified data
        merged = pd.merge(
            unified_data,
            regime_interpolated.reset_index(),
            on='timestamp',
            how='inner'
        )
        
        return merged
    
    def _forward_fill_merge(
        self, 
        unified_data: pd.DataFrame, 
        regime_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge with forward fill for regime labels."""
        self.logger.info("🔍 Performing forward fill merge")
        
        # Use merge_asof with forward direction
        merged = pd.merge_asof(
            unified_data.sort_values('timestamp'),
            regime_data[['timestamp', 'composite_cluster_id']].sort_values('timestamp'),
            on='timestamp',
            direction='forward'
        )
        
        # Remove rows where no regime match was found
        merged = merged.dropna(subset=['composite_cluster_id'])
        
        return merged
    
    def _validate_merge_result(
        self,
        original_unified: pd.DataFrame,
        original_regime: pd.DataFrame,
        merged_data: pd.DataFrame,
        timeframe: str
    ) -> Dict[str, Any]:
        """Validate merge result and calculate retention metrics."""
        
        # Calculate retention ratio
        retention_ratio = len(merged_data) / len(original_unified)
        
        # Get threshold for timeframe
        threshold = self.retention_thresholds.get(timeframe, 0.80)
        
        # Check if retention meets threshold
        meets_threshold = retention_ratio >= threshold
        
        # Calculate regime distribution
        regime_distribution = merged_data['composite_cluster_id'].value_counts().to_dict()
        
        # Calculate data quality metrics
        quality_metrics = {
            'missing_regime_labels': merged_data['composite_cluster_id'].isna().sum(),
            'duplicate_timestamps': merged_data['timestamp'].duplicated().sum(),
            'regime_consistency': self._calculate_regime_consistency(merged_data)
        }
        
        # Generate warnings if needed
        warnings = []
        if not meets_threshold:
            warnings.append(
                f"Low retention ratio {retention_ratio:.3f} < {threshold:.3f} for {timeframe} data"
            )
        
        if quality_metrics['missing_regime_labels'] > 0:
            warnings.append(f"Found {quality_metrics['missing_regime_labels']} missing regime labels")
        
        if quality_metrics['duplicate_timestamps'] > 0:
            warnings.append(f"Found {quality_metrics['duplicate_timestamps']} duplicate timestamps")
        
        return {
            'retention_ratio': retention_ratio,
            'meets_threshold': meets_threshold,
            'threshold': threshold,
            'regime_distribution': regime_distribution,
            'quality_metrics': quality_metrics,
            'warnings': warnings,
            'merge_success': meets_threshold and len(warnings) == 0
        }
    
    def _calculate_regime_consistency(self, data: pd.DataFrame) -> float:
        """Calculate regime consistency score."""
        if len(data) < 2:
            return 1.0
        
        # Calculate regime transitions
        regime_changes = (data['composite_cluster_id'].diff() != 0).sum()
        total_periods = len(data) - 1
        
        # Consistency score (lower changes = higher consistency)
        consistency = 1.0 - (regime_changes / total_periods)
        return max(0.0, consistency)
    
    def get_retention_recommendations(
        self, 
        timeframe: str, 
        actual_retention: float
    ) -> List[str]:
        """Get recommendations for improving retention ratio."""
        recommendations = []
        threshold = self.retention_thresholds.get(timeframe, 0.80)
        
        if actual_retention < threshold:
            recommendations.append(
                f"Consider using 'tolerant' or 'interpolate' alignment strategy "
                f"instead of 'strict' for {timeframe} data"
            )
            
            if timeframe in ['1m', '5m']:
                recommendations.append(
                    "For high-frequency data, consider data preprocessing to "
                    "align timestamps before merging"
                )
            
            recommendations.append(
                "Check for data gaps in regime discovery step that might "
                "cause timestamp misalignment"
            )
        
        return recommendations


# Example usage and testing
def test_improved_data_merger():
    """Test the improved data merger with sample data."""
    
    # Create sample data
    timestamps = pd.date_range('2024-01-01', periods=1000, freq='1min')
    unified_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Create regime data with some timestamp misalignment
    regime_timestamps = timestamps[::2]  # Every other timestamp
    regime_data = pd.DataFrame({
        'timestamp': regime_timestamps,
        'composite_cluster_id': np.random.randint(0, 5, len(regime_timestamps))
    })
    
    # Test different alignment strategies
    config = {}
    merger = ImprovedDataMerger(config)
    
    strategies = ['strict', 'tolerant', 'interpolate', 'forward_fill']
    
    for strategy in strategies:
        print(f"\n=== Testing {strategy} strategy ===")
        try:
            merged_data, metadata = merger.merge_regime_data_improved(
                unified_data, regime_data, '1m', strategy
            )
            print(f"Retention ratio: {metadata['retention_ratio']:.3f}")
            print(f"Meets threshold: {metadata['meets_threshold']}")
            print(f"Regime distribution: {metadata['regime_distribution']}")
            if metadata['warnings']:
                print(f"Warnings: {metadata['warnings']}")
        except Exception as e:
            print(f"Error with {strategy}: {e}")


if __name__ == "__main__":
    test_improved_data_merger()