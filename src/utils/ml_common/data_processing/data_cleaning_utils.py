"""
Data Cleaning Utilities for ML Training

This module provides utilities to clean and preprocess financial data
for machine learning training, including handling of corrupted periods.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)
from src.utils.tprint import tprint

def exclude_corrupted_periods(df: pd.DataFrame,
                             corrupted_periods: Optional[List[Tuple[str, str]]] = None,
                             datetime_col: str = 'datetime') -> pd.DataFrame:
    """
    Exclude known corrupted data periods from the dataset.

    This function removes periods where the data source returned artificial/corrupted data
    that could negatively impact ML training quality.

    Args:
        df: DataFrame with datetime column
        corrupted_periods: List of (start, end) datetime strings for corrupted periods.
                          If None, uses default periods for ETHUSDT.
        datetime_col: Name of datetime column

    Returns:
        DataFrame with corrupted periods excluded

    Example:
        >>> df_clean = exclude_corrupted_periods(df)
        >>> print(f"Excluded {len(df) - len(df_clean)} corrupted rows")
    """
    if corrupted_periods is None:
        # Default corrupted periods for ETHUSDT data
        # These periods contain artificial data (zero volumes + constant prices)
        corrupted_periods = [
            ('2023-03-24 11:28:00', '2023-03-24 12:39:00'),  # Binance API corruption: 72 min zero volume
        ]

    if datetime_col not in df.columns:
        logger.warning(f"Datetime column '{datetime_col}' not found. Available columns: {list(df.columns)}")
        return df

    df_clean = df.copy()
    total_excluded = 0

    logger.info("🧹 Starting corrupted data exclusion...")
    logger.info(f"📊 Processing {len(corrupted_periods)} corrupted periods")

    for i, (start_str, end_str) in enumerate(corrupted_periods, 1):
        start_dt = pd.Timestamp(start_str)
        end_dt = pd.Timestamp(end_str)

        mask = (df_clean[datetime_col] >= start_dt) & (df_clean[datetime_col] <= end_dt)
        excluded_count = mask.sum()

        if excluded_count > 0:
            df_clean = df_clean[~mask]
            total_excluded += excluded_count

            duration_minutes = (end_dt - start_dt).total_seconds() / 60
            logger.info(f"  {i}. Excluded {excluded_count:,} rows from {start_str} to {end_str} ({duration_minutes:.0f} min)")
        else:
            logger.warning(f"  {i}. No rows found for period {start_str} to {end_str}")

    # Summary
    original_count = len(df)
    clean_count = len(df_clean)
    loss_percentage = 100 * (original_count - clean_count) / original_count

    logger.info("✅ Corrupted data exclusion completed:")
    logger.info(f"   📊 Original dataset: {original_count:,} rows")
    logger.info(f"   🧹 Clean dataset: {clean_count:,} rows")
    logger.info(f"   ❌ Excluded: {total_excluded:,} rows ({loss_percentage:.4f}%)")

    if total_excluded == 0:
        logger.info("   ✅ No corrupted periods found - dataset appears clean")
    elif loss_percentage < 0.01:
        logger.info("   ✅ Minimal data loss - safe for ML training")
    elif loss_percentage < 1.0:
        logger.info("   ⚠️  Moderate data loss - monitor model performance")
    else:
        logger.warning("   🚨 Significant data loss - consider alternative data sources")

    return df_clean

def validate_data_quality_after_cleaning(df: pd.DataFrame,
                                        volume_col: str = 'volume',
                                        datetime_col: str = 'datetime') -> dict:
    """
    Validate data quality after cleaning to ensure corrupted data was properly removed.

    Args:
        df: Cleaned DataFrame
        volume_col: Name of volume column
        datetime_col: Name of datetime column

    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'zero_volumes': 0,
        'constant_price_periods': 0,
        'time_gaps': 0,
        'quality_score': 100.0,
        'recommendations': []
    }

    if volume_col in df.columns:
        zero_volumes = (df[volume_col] == 0).sum()
        quality_report['zero_volumes'] = zero_volumes

        if zero_volumes > 0:
            quality_report['quality_score'] -= min(50, zero_volumes * 0.1)
            quality_report['recommendations'].append(f"Found {zero_volumes} zero volume entries - investigate data source")

    if datetime_col in df.columns:
        # Check for time gaps
        df_sorted = df.sort_values(datetime_col)
        time_diffs = df_sorted[datetime_col].diff().dt.total_seconds()
        large_gaps = (time_diffs > 300).sum()  # Gaps > 5 minutes
        quality_report['time_gaps'] = large_gaps

        if large_gaps > 0:
            quality_report['quality_score'] -= min(20, large_gaps * 2)
            quality_report['recommendations'].append(f"Found {large_gaps} large time gaps - may affect time series analysis")

    # Check for constant price periods (potential corruption indicator)
    if 'close' in df.columns:
        price_diffs = df['close'].diff()
        constant_periods = 0
        current_constant = 0

        for diff in price_diffs:
            if diff == 0:
                current_constant += 1
            else:
                if current_constant > 30:  # >30 minutes of constant price
                    constant_periods += 1
                current_constant = 0

        quality_report['constant_price_periods'] = constant_periods

        if constant_periods > 0:
            quality_report['quality_score'] -= min(30, constant_periods * 10)
            quality_report['recommendations'].append(f"Found {constant_periods} long constant price periods - potential data quality issues")

    # Overall assessment
    if quality_report['quality_score'] >= 95:
        quality_report['assessment'] = 'EXCELLENT'
    elif quality_report['quality_score'] >= 85:
        quality_report['assessment'] = 'GOOD'
    elif quality_report['quality_score'] >= 70:
        quality_report['assessment'] = 'FAIR'
    else:
        quality_report['assessment'] = 'POOR'
        quality_report['recommendations'].append("Data quality is poor - consider alternative data sources")

    logger.info(f"📊 Data quality assessment: {quality_report['assessment']} ({quality_report['quality_score']:.1f}%)")

    return quality_report

def create_clean_dataset_pipeline(input_file: str, output_file: str) -> dict:
    """
    Complete pipeline to create a clean dataset by excluding corrupted periods.

    Args:
        input_file: Path to input parquet file
        output_file: Path to output clean parquet file

    Returns:
        Dictionary with processing results
    """
    logger.info(f"🚀 Starting clean dataset creation pipeline")
    logger.info(f"📁 Input: {input_file}")
    logger.info(f"📁 Output: {output_file}")

    try:
        # Load data
        logger.info("📖 Loading input data...")
        df = pd.read_parquet(input_file)

        # Ensure datetime column exists
        if 'timestamp' in df.columns and df['timestamp'].dtype == 'int64':
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        elif 'datetime' not in df.columns:
            logger.error("❌ No datetime column found in dataset")
            return {'success': False, 'error': 'No datetime column'}

        # Clean corrupted periods
        logger.info("🧹 Cleaning corrupted data periods...")
        df_clean = exclude_corrupted_periods(df)

        # Validate quality
        logger.info("🔍 Validating data quality...")
        quality_report = validate_data_quality_after_cleaning(df_clean)

        # Save clean dataset
        logger.info(f"💾 Saving clean dataset to {output_file}")
        df_clean.to_parquet(output_file, index=False)

        # Results
        result = {
            'success': True,
            'original_rows': len(df),
            'clean_rows': len(df_clean),
            'excluded_rows': len(df) - len(df_clean),
            'data_loss_percentage': 100 * (len(df) - len(df_clean)) / len(df),
            'output_file': output_file,
            'quality_report': quality_report
        }

        logger.info("✅ Clean dataset pipeline completed successfully")
        logger.info(f"📊 Results: {result['excluded_rows']:,} rows excluded ({result['data_loss_percentage']:.4f}%)")
        logger.info(f"🎯 Quality: {quality_report['assessment']} ({quality_report['quality_score']:.1f}%)")

        return result

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return {'success': False, 'error': str(e)}

# Example usage
if __name__ == "__main__":
    # Example: Clean the consolidated ETHUSDT dataset
    input_file = "/Users/remyroche/Documents/Ares/historical_data/features_binance_ETHUSDT_consolidated.parquet"
    output_file = "/Users/remyroche/Documents/Ares/historical_data/features_binance_ETHUSDT_clean.parquet"

    result = create_clean_dataset_pipeline(input_file, output_file)

    if result['success']:
        tprint("🎉 Clean dataset created successfully!")
        tprint(f"Excluded: {result['excluded_rows']:,} rows ({result['data_loss_percentage']:.4f}%)")
        tprint(f"Quality: {result['quality_report']['assessment']}")
    else:
        tprint(f"❌ Failed to create clean dataset: {result['error']}")
