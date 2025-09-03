"""Data Cleaner Component
Handles data cleaning operations including duplicate removal, missing value handling, and outlier detection.
Extracted from step01_5_data_converter.py
"""
from typing import Any, Optional, Union
import pandas as pd
import numpy as np
from src.utils.logger import system_logger


class DataCleaner:
    """Handles data cleaning operations for market data.
    
    This class provides functionality for:
    - Removing duplicates
    - Handling missing values
    - Detecting and removing outliers
    - Data normalization and standardization
    - Time series specific cleaning
    """
    
    def __init__(self, logger=None):
        self.logger = logger or system_logger.getChild("DataCleaner")
        
    def remove_duplicates(
        self, 
        df: pd.DataFrame, 
        subset: Optional[list[str]] = None,
        keep: str = "first"
    ) -> pd.DataFrame:
        """
        Remove duplicate rows from DataFrame.
        
        Args:
            df: DataFrame to clean
            subset: Columns to consider for identifying duplicates
            keep: Which duplicate to keep ('first', 'last', False)
            
        Returns:
            DataFrame with duplicates removed
        """
        try:
            initial_rows = len(df)
            
            if subset:
                df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
            else:
                df_cleaned = df.drop_duplicates(keep=keep)
                
            removed_rows = initial_rows - len(df_cleaned)
            
            if removed_rows > 0:
                self.logger.info(f"✅ Removed {removed_rows} duplicate rows")
            else:
                self.logger.info("✅ No duplicate rows found")
                
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"❌ Error removing duplicates: {e}")
            return df
            
    def fill_missing_values(
        self, 
        df: pd.DataFrame,
        method: str = "auto",
        numeric_fill: Union[str, float] = 0,
        string_fill: str = "",
        custom_fills: Optional[dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Fill missing values in DataFrame.
        
        Args:
            df: DataFrame to clean
            method: Method to use ('auto', 'forward', 'backward', 'interpolate')
            numeric_fill: Value to use for numeric columns
            string_fill: Value to use for string columns
            custom_fills: Dictionary of column-specific fill values
            
        Returns:
            DataFrame with missing values filled
        """
        try:
            filled_columns: list[str] = []
            df_cleaned = df.copy()
            
            # Handle custom fills first
            if custom_fills:
                for col, fill_value in custom_fills.items():
                    if col in df_cleaned.columns:
                        missing_count = int(df_cleaned[col].isna().sum())
                        if missing_count > 0:
                            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                            filled_columns.append(f"{col} ({missing_count} values)")
            
            if method == "auto":
                # Fill numeric columns
                numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    if col in ("timestamp", "year", "month", "day"):
                        continue
                    if custom_fills and col in custom_fills:
                        continue
                        
                    missing_count = int(df_cleaned[col].isna().sum())
                    if missing_count > 0:
                        df_cleaned[col] = df_cleaned[col].fillna(numeric_fill)
                        filled_columns.append(f"{col} ({missing_count} values)")
                
                # Fill string columns
                string_columns = df_cleaned.select_dtypes(include=["object", "string"]).columns
                for col in string_columns:
                    if custom_fills and col in custom_fills:
                        continue
                        
                    missing_count = int(df_cleaned[col].isna().sum())
                    if missing_count > 0:
                        df_cleaned[col] = df_cleaned[col].fillna(string_fill)
                        filled_columns.append(f"{col} ({missing_count} values)")
                        
            elif method == "forward":
                df_cleaned = df_cleaned.fillna(method="ffill")
                
            elif method == "backward":
                df_cleaned = df_cleaned.fillna(method="bfill")
                
            elif method == "interpolate":
                numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
                df_cleaned[numeric_columns] = df_cleaned[numeric_columns].interpolate(method="linear")
            
            if filled_columns:
                self.logger.info(
                    f"✅ Filled missing values in: {', '.join(filled_columns)}"
                )
                
            return df_cleaned
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to fill missing values: {e}")
            return df
            
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[list[str]] = None,
        method: str = "zscore",
        threshold: float = 3.0
    ) -> tuple[pd.DataFrame, dict[str, list[int]]]:
        """
        Detect outliers in specified columns.
        
        Args:
            df: DataFrame to analyze
            columns: Columns to check for outliers (None = all numeric)
            method: Detection method ('zscore', 'iqr', 'isolation_forest')
            threshold: Threshold for outlier detection
            
        Returns:
            Tuple of (DataFrame with outlier flags, dict of outlier indices by column)
        """
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
                columns = [col for col in columns if col not in ["timestamp", "year", "month", "day"]]
            
            outliers = {}
            df_with_flags = df.copy()
            
            for col in columns:
                if col not in df.columns:
                    continue
                    
                if method == "zscore":
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outlier_mask = z_scores > threshold
                    
                elif method == "iqr":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    
                else:
                    self.logger.warning(f"Unknown outlier detection method: {method}")
                    continue
                
                outlier_indices = df.index[outlier_mask].tolist()
                if outlier_indices:
                    outliers[col] = outlier_indices
                    df_with_flags[f"{col}_outlier"] = outlier_mask
                    self.logger.info(
                        f"📊 Found {len(outlier_indices)} outliers in column '{col}'"
                    )
                    
            return df_with_flags, outliers
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting outliers: {e}")
            return df, {}
            
    def remove_outliers(
        self,
        df: pd.DataFrame,
        outliers: dict[str, list[int]],
        method: str = "remove"
    ) -> pd.DataFrame:
        """
        Remove or handle outliers based on detection results.
        
        Args:
            df: DataFrame with data
            outliers: Dictionary of outlier indices by column
            method: How to handle outliers ('remove', 'cap', 'nan')
            
        Returns:
            DataFrame with outliers handled
        """
        try:
            df_cleaned = df.copy()
            
            if method == "remove":
                # Get all unique outlier indices
                all_outlier_indices = set()
                for indices in outliers.values():
                    all_outlier_indices.update(indices)
                    
                df_cleaned = df_cleaned.drop(index=list(all_outlier_indices))
                self.logger.info(
                    f"✅ Removed {len(all_outlier_indices)} rows containing outliers"
                )
                
            elif method == "cap":
                for col, indices in outliers.items():
                    if col in df_cleaned.columns:
                        # Cap at 99th and 1st percentile
                        upper_cap = df_cleaned[col].quantile(0.99)
                        lower_cap = df_cleaned[col].quantile(0.01)
                        df_cleaned.loc[indices, col] = df_cleaned.loc[indices, col].clip(
                            lower=lower_cap, upper=upper_cap
                        )
                self.logger.info("✅ Capped outlier values to percentile bounds")
                
            elif method == "nan":
                for col, indices in outliers.items():
                    if col in df_cleaned.columns:
                        df_cleaned.loc[indices, col] = np.nan
                self.logger.info("✅ Replaced outlier values with NaN")
                
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"❌ Error removing outliers: {e}")
            return df
            
    def clean_time_series(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        remove_weekends: bool = False,
        remove_holidays: bool = False,
        ensure_regular_intervals: bool = True
    ) -> pd.DataFrame:
        """
        Perform time series specific cleaning operations.
        
        Args:
            df: DataFrame with time series data
            timestamp_col: Name of timestamp column
            remove_weekends: Whether to remove weekend data
            remove_holidays: Whether to remove holiday data
            ensure_regular_intervals: Whether to ensure regular time intervals
            
        Returns:
            Cleaned time series DataFrame
        """
        try:
            df_cleaned = df.copy()
            
            # Ensure timestamp is datetime
            if timestamp_col in df_cleaned.columns:
                if not pd.api.types.is_datetime64_any_dtype(df_cleaned[timestamp_col]):
                    df_cleaned[timestamp_col] = pd.to_datetime(
                        df_cleaned[timestamp_col], unit="ms", utc=True
                    )
                
                # Sort by timestamp
                df_cleaned = df_cleaned.sort_values(timestamp_col)
                
                # Remove weekends if requested
                if remove_weekends:
                    weekday_mask = df_cleaned[timestamp_col].dt.dayofweek < 5
                    removed_count = len(df_cleaned) - weekday_mask.sum()
                    df_cleaned = df_cleaned[weekday_mask]
                    if removed_count > 0:
                        self.logger.info(f"✅ Removed {removed_count} weekend rows")
                
                # Ensure regular intervals if requested
                if ensure_regular_intervals:
                    # Detect the most common interval
                    time_diffs = df_cleaned[timestamp_col].diff().dropna()
                    mode_interval = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
                    
                    # Find gaps larger than expected
                    gap_threshold = mode_interval * 2
                    gaps = time_diffs > gap_threshold
                    gap_count = gaps.sum()
                    
                    if gap_count > 0:
                        self.logger.warning(
                            f"⚠️ Found {gap_count} time gaps larger than expected interval"
                        )
                        
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning time series: {e}")
            return df
            
    def validate_cleaned_data(
        self,
        df: pd.DataFrame,
        original_df: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Validate the cleaned data against the original.
        
        Args:
            df: Cleaned DataFrame
            original_df: Original DataFrame
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {
                "original_rows": len(original_df),
                "cleaned_rows": len(df),
                "rows_removed": len(original_df) - len(df),
                "columns_unchanged": list(df.columns) == list(original_df.columns),
                "missing_values_before": original_df.isna().sum().sum(),
                "missing_values_after": df.isna().sum().sum(),
                "data_types_preserved": True,
                "warnings": []
            }
            
            # Check data types
            for col in df.columns:
                if col in original_df.columns:
                    if df[col].dtype != original_df[col].dtype:
                        validation_results["data_types_preserved"] = False
                        validation_results["warnings"].append(
                            f"Data type changed for column '{col}': "
                            f"{original_df[col].dtype} -> {df[col].dtype}"
                        )
            
            # Check for significant data loss
            row_loss_pct = validation_results["rows_removed"] / validation_results["original_rows"] * 100
            if row_loss_pct > 10:
                validation_results["warnings"].append(
                    f"Significant data loss: {row_loss_pct:.1f}% of rows removed"
                )
                
            # Log validation results
            self.logger.info(
                f"📊 Validation complete: "
                f"{validation_results['cleaned_rows']}/{validation_results['original_rows']} rows retained, "
                f"{validation_results['missing_values_after']} missing values remaining"
            )
            
            if validation_results["warnings"]:
                for warning in validation_results["warnings"]:
                    self.logger.warning(f"⚠️ {warning}")
                    
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Error validating cleaned data: {e}")
            return {"error": str(e)}