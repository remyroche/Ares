"""
Memory-efficient DataFrame operations for feature lookback optimization.

This module provides optimized DataFrame operations that minimize memory usage
by using views instead of copies where possible.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List, Dict, Any
from dataclasses import dataclass
import gc

from ..error_handling.error_handler import safe_operation, DataValidationError
from .tprint_utils import tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success


@dataclass
class SplitResult:
    """Result of memory-efficient DataFrame splitting."""
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    split_index: int
    memory_saved_mb: float


class MemoryEfficientOps:
    """Memory-efficient DataFrame operations."""
    
    def __init__(self, enable_gc: bool = True):
        tprint("🔧 Initializing MemoryEfficientOps...")
        self.enable_gc = enable_gc
        self._memory_stats = {
            'copies_avoided': 0,
            'memory_saved_mb': 0.0,
            'views_created': 0
        }
        tprint("✅ MemoryEfficientOps initialized successfully")
    
    @safe_operation("memory-efficient data split", default_value=None)
    def split_dataframe_efficiently(
        self, 
        data: pd.DataFrame, 
        split_ratio: float = 0.8,
        force_copy: bool = False
    ) -> Optional[SplitResult]:
        """
        Split DataFrame efficiently using views when possible.
        
        Args:
            data: Input DataFrame
            split_ratio: Ratio for train/test split (0.0 to 1.0)
            force_copy: Force copying even when views would work
            
        Returns:
            SplitResult with train/test data and memory stats
        """
        tprint(f"🔧 Starting memory-efficient data split (ratio={split_ratio}, force_copy={force_copy})...")
        
        if not isinstance(data, pd.DataFrame):
            tprint_error(f"❌ Input must be DataFrame, got {type(data)}")
            raise DataValidationError(f"Input must be DataFrame, got {type(data)}")
        
        if data.empty:
            tprint_error("❌ Cannot split empty DataFrame")
            raise DataValidationError("Cannot split empty DataFrame")
        
        split_index = int(len(data) * split_ratio)
        tprint(f"📊 Split index calculated: {split_index} (total rows: {len(data)})")
        
        if force_copy or self._requires_copy(data):
            # Use copies when necessary
            tprint("📋 Using data copies for split")
            train_data = data.iloc[:split_index].copy()
            test_data = data.iloc[split_index:].copy()
            memory_saved = 0.0
        else:
            # Use views for memory efficiency
            tprint("👁️ Using data views for memory efficiency")
            train_data = data.iloc[:split_index]
            test_data = data.iloc[split_index:]
            memory_saved = self._estimate_memory_saved(data, split_index)
            self._memory_stats['views_created'] += 1
            self._memory_stats['memory_saved_mb'] += memory_saved
            tprint(f"💾 Memory saved: {memory_saved:.2f} MB")
        
        self._memory_stats['copies_avoided'] += 1
        
        if self.enable_gc:
            tprint("🧹 Running garbage collection...")
            gc.collect()
        
        tprint(f"✅ Data split completed successfully (train: {len(train_data)}, test: {len(test_data)})")
        return SplitResult(
            train_data=train_data,
            test_data=test_data,
            split_index=split_index,
            memory_saved_mb=memory_saved
        )
    
    def _requires_copy(self, data: pd.DataFrame) -> bool:
        """Determine if DataFrame requires copying."""
        tprint_debug("🔍 Checking if DataFrame requires copying...")
        
        # Check if DataFrame has complex dtypes that might cause issues with views
        complex_dtypes = ['object', 'category', 'datetime64[ns]']
        has_complex_dtypes = any(str(dtype) in complex_dtypes for dtype in data.dtypes)
        
        # Check if DataFrame is already a view
        is_view = data._is_view
        
        # Check memory usage - if small, copying is fine
        memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        is_small = memory_mb < 10  # Less than 10MB
        
        requires_copy = has_complex_dtypes or is_view or is_small
        tprint_debug(f"📊 Copy required: {requires_copy} (complex_dtypes: {has_complex_dtypes}, is_view: {is_view}, is_small: {is_small}, memory: {memory_mb:.2f}MB)")
        
        return requires_copy
    
    def _estimate_memory_saved(self, data: pd.DataFrame, split_index: int) -> float:
        """Estimate memory saved by using views instead of copies."""
        tprint_debug("💾 Estimating memory saved by using views...")
        total_memory = data.memory_usage(deep=True).sum() / 1024 / 1024
        estimated_saved = total_memory * 0.5  # Rough estimate for split data
        tprint_debug(f"📊 Estimated memory saved: {estimated_saved:.2f} MB (total: {total_memory:.2f} MB)")
        return estimated_saved
    
    @safe_operation("memory-efficient column selection", default_value=None)
    def select_columns_efficiently(
        self, 
        data: pd.DataFrame, 
        columns: List[str],
        force_copy: bool = False
    ) -> pd.DataFrame:
        """
        Select columns efficiently using views when possible.
        
        Args:
            data: Input DataFrame
            columns: List of column names to select
            force_copy: Force copying even when views would work
            
        Returns:
            DataFrame with selected columns
        """
        tprint(f"🔧 Starting memory-efficient column selection (columns={len(columns)}, force_copy={force_copy})...")
        
        if not columns:
            tprint("⚠️ No columns specified, returning empty DataFrame")
            return data.iloc[:, :0]  # Empty DataFrame with same index
        
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols:
            tprint_error(f"❌ Missing columns: {missing_cols}")
            raise DataValidationError(f"Missing columns: {missing_cols}")
        
        if force_copy or self._requires_copy(data):
            tprint("📋 Using column copy for selection")
            result = data[columns].copy()
        else:
            tprint("👁️ Using column view for selection")
            result = data[columns]
        
        tprint(f"✅ Column selection completed successfully ({len(result.columns)} columns, {len(result)} rows)")
        return result
    
    @safe_operation("memory-efficient data alignment", default_value=None)
    def align_dataframes_efficiently(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame,
        on: Optional[str] = None,
        how: str = 'inner'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align DataFrames efficiently without unnecessary copying.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            on: Column to align on (uses index if None)
            how: Join type ('inner', 'left', 'right', 'outer')
            
        Returns:
            Tuple of aligned DataFrames
        """
        tprint(f"🔧 Starting memory-efficient DataFrame alignment (on={on}, how={how})...")
        tprint(f"📊 DataFrame 1: {len(df1)} rows, {len(df1.columns)} columns")
        tprint(f"📊 DataFrame 2: {len(df2)} rows, {len(df2.columns)} columns")
        
        if on is None:
            # Align by index
            tprint("🔍 Aligning by index...")
            common_index = df1.index.intersection(df2.index)
            if len(common_index) == 0:
                tprint_error("❌ No common index values found")
                raise DataValidationError("No common index values found")
            
            tprint(f"📊 Found {len(common_index)} common index values")
            aligned_df1 = df1.loc[common_index]
            aligned_df2 = df2.loc[common_index]
        else:
            # Align by column
            tprint(f"🔍 Aligning by column '{on}'...")
            if on not in df1.columns or on not in df2.columns:
                tprint_error(f"❌ Column '{on}' not found in both DataFrames")
                raise DataValidationError(f"Column '{on}' not found in both DataFrames")
            
            merged = pd.merge(df1, df2, on=on, how=how, suffixes=('_1', '_2'))
            aligned_df1 = merged[[col for col in df1.columns if col != on]]
            aligned_df2 = merged[[col for col in df2.columns if col != on]]
            tprint(f"📊 Merge result: {len(merged)} rows")
        
        tprint(f"✅ DataFrame alignment completed successfully (aligned: {len(aligned_df1)} rows)")
        return aligned_df1, aligned_df2
    
    @safe_operation("memory-efficient data preparation", default_value=None)
    def prepare_data_efficiently(
        self, 
        data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        remove_nans: bool = True
    ) -> pd.DataFrame:
        """
        Prepare data efficiently for optimization.
        
        Args:
            data: Input DataFrame
            feature_columns: List of feature column names
            target_columns: List of target column names
            remove_nans: Whether to remove rows with NaN values
            
        Returns:
            Prepared DataFrame
        """
        tprint(f"🔧 Starting memory-efficient data preparation (features={len(feature_columns)}, targets={len(target_columns)}, remove_nans={remove_nans})...")
        
        all_columns = feature_columns + target_columns
        missing_cols = [col for col in all_columns if col not in data.columns]
        if missing_cols:
            tprint_error(f"❌ Missing columns: {missing_cols}")
            raise DataValidationError(f"Missing columns: {missing_cols}")
        
        # Select only required columns efficiently
        tprint("🔍 Selecting required columns efficiently...")
        prepared_data = self.select_columns_efficiently(data, all_columns)
        
        if remove_nans:
            tprint("🧹 Removing rows with NaN values...")
            # Remove rows with NaN values in any required column
            valid_mask = prepared_data[all_columns].notna().all(axis=1)
            if not valid_mask.any():
                tprint_error("❌ No valid rows after removing NaN values")
                raise DataValidationError("No valid rows after removing NaN values")
            
            initial_rows = len(prepared_data)
            prepared_data = prepared_data[valid_mask]
            removed_rows = initial_rows - len(prepared_data)
            tprint(f"📊 Removed {removed_rows} rows with NaN values (remaining: {len(prepared_data)})")
        
        tprint(f"✅ Data preparation completed successfully ({len(prepared_data)} rows, {len(prepared_data.columns)} columns)")
        return prepared_data
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory optimization statistics."""
        tprint_debug("📊 Retrieving memory optimization statistics...")
        stats = self._memory_stats.copy()
        tprint_debug(f"📊 Memory stats: {stats}")
        return stats
    
    def reset_memory_stats(self) -> None:
        """Reset memory optimization statistics."""
        tprint("🔄 Resetting memory optimization statistics...")
        self._memory_stats = {
            'copies_avoided': 0,
            'memory_saved_mb': 0.0,
            'views_created': 0
        }
        tprint("✅ Memory statistics reset successfully")


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by converting data types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Memory-optimized DataFrame
    """
    tprint("🔧 Starting DataFrame memory optimization...")
    
    if df.empty:
        tprint("⚠️ DataFrame is empty, returning as-is")
        return df
    
    initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    tprint(f"📊 Initial memory usage: {initial_memory:.2f} MB")
    
    optimized_df = df.copy()
    optimizations_applied = 0
    
    # Convert object columns to category if they have low cardinality
    tprint("🔍 Checking object columns for category conversion...")
    for col in optimized_df.select_dtypes(include=['object']).columns:
        if optimized_df[col].nunique() / len(optimized_df) < 0.5:  # Less than 50% unique values
            optimized_df[col] = optimized_df[col].astype('category')
            optimizations_applied += 1
            tprint(f"📊 Converted column '{col}' to category")
    
    # Convert int64 to int32 if values fit
    tprint("🔍 Checking int64 columns for int32 conversion...")
    for col in optimized_df.select_dtypes(include=['int64']).columns:
        if optimized_df[col].min() >= np.iinfo(np.int32).min and optimized_df[col].max() <= np.iinfo(np.int32).max:
            optimized_df[col] = optimized_df[col].astype('int32')
            optimizations_applied += 1
            tprint(f"📊 Converted column '{col}' from int64 to int32")
    
    # Convert float64 to float32 if precision allows
    tprint("🔍 Checking float64 columns for float32 conversion...")
    for col in optimized_df.select_dtypes(include=['float64']).columns:
        if not optimized_df[col].isna().any():  # Only if no NaN values
            if (optimized_df[col].min() >= np.finfo(np.float32).min and 
                optimized_df[col].max() <= np.finfo(np.float32).max):
                optimized_df[col] = optimized_df[col].astype('float32')
                optimizations_applied += 1
                tprint(f"📊 Converted column '{col}' from float64 to float32")
    
    final_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024
    memory_saved = initial_memory - final_memory
    tprint(f"✅ Memory optimization completed: {optimizations_applied} optimizations applied, {memory_saved:.2f} MB saved ({final_memory:.2f} MB final)")
    
    return optimized_df


def create_dataframe_view(df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    """
    Create a memory-efficient view of a DataFrame slice.
    
    Args:
        df: Input DataFrame
        start_idx: Start index
        end_idx: End index (exclusive)
        
    Returns:
        DataFrame view (not a copy)
    """
    tprint_debug(f"👁️ Creating DataFrame view (indices {start_idx}:{end_idx})...")
    view = df.iloc[start_idx:end_idx]
    tprint_debug(f"📊 View created: {len(view)} rows, {len(view.columns)} columns")
    return view


def batch_process_dataframe(
    df: pd.DataFrame, 
    batch_size: int,
    processor_func,
    *args, **kwargs
) -> List[Any]:
    """
    Process DataFrame in batches to manage memory usage.
    
    Args:
        df: Input DataFrame
        batch_size: Size of each batch
        processor_func: Function to process each batch
        *args, **kwargs: Arguments for processor function
        
    Returns:
        List of results from each batch
    """
    tprint(f"🔧 Starting batch processing (batch_size={batch_size}, total_rows={len(df)})...")
    
    results = []
    total_rows = len(df)
    total_batches = (total_rows + batch_size - 1) // batch_size
    tprint(f"📊 Will process {total_batches} batches")
    
    for batch_num, start_idx in enumerate(range(0, total_rows, batch_size), 1):
        end_idx = min(start_idx + batch_size, total_rows)
        tprint(f"🔄 Processing batch {batch_num}/{total_batches} (rows {start_idx}-{end_idx-1})...")
        
        batch_df = create_dataframe_view(df, start_idx, end_idx)
        
        try:
            result = processor_func(batch_df, *args, **kwargs)
            results.append(result)
            tprint(f"✅ Batch {batch_num} processed successfully")
        except Exception as e:
            # Fast fail on batch processing errors
            tprint_error(f"❌ Batch processing failed for rows {start_idx}-{end_idx}: {e}")
            raise  # Fast fail instead of silent continuation
        
        # Force garbage collection after each batch
        tprint_debug("🧹 Running garbage collection after batch...")
        gc.collect()
    
    tprint(f"✅ Batch processing completed successfully ({len(results)} results)")
    return results