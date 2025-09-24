"""
Data Preprocessor for Regime Detection Systems.

This module provides standardized data preprocessing utilities that can be
used by both NAS and TAS regime detection systems.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from src.utils.logger import system_logger


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    remove_outliers: bool = True
    outlier_threshold: float = 3.0
    fill_missing_values: bool = True
    fill_method: str = 'forward'  # 'forward', 'backward', 'interpolate', 'median'
    normalize_features: bool = False
    normalization_method: str = 'standard'  # 'standard', 'minmax', 'robust'
    remove_duplicate_timestamps: bool = True
    validate_data_quality: bool = True
    min_data_quality_score: float = 0.7


class DataPreprocessor:
    """
    Data preprocessor for regime detection systems.

    This class provides standardized preprocessing that follows the same
    patterns as hmm_regime_discovery.py for data cleaning and preparation.
    """

    def __init__(self):
        """Initialize the data preprocessor."""
        self.logger = system_logger.getChild('DataPreprocessor')

    def preprocess_data(self,
                       data: pd.DataFrame,
                       config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Preprocess market data for regime detection.

        Args:
            data: Raw market data
            config: Preprocessing configuration

        Returns:
            Preprocessed data
        """
        try:
            self.logger.info(f"🔄 Preprocessing {len(data)} data points")

            # Parse configuration
            if config is None:
                config = {}
            preprocessing_config = PreprocessingConfig(**config)

            # Work on a copy to avoid modifying original data
            processed_data = data.copy()

            # Apply preprocessing steps
            if preprocessing_config.validate_data_quality:
                self._validate_data_quality(processed_data, preprocessing_config)

            if preprocessing_config.remove_duplicate_timestamps:
                processed_data = self._remove_duplicate_timestamps(processed_data)

            if preprocessing_config.fill_missing_values:
                processed_data = self._fill_missing_values(processed_data, preprocessing_config)

            if preprocessing_config.remove_outliers:
                processed_data = self._remove_outliers(processed_data, preprocessing_config)

            if preprocessing_config.normalize_features:
                processed_data = self._normalize_features(processed_data, preprocessing_config)

            # Ensure required columns exist and are properly formatted
            processed_data = self._ensure_required_columns(processed_data)

            # Final validation
            processed_data = self._final_validation(processed_data)

            self.logger.info(f"✅ Data preprocessing completed: {len(processed_data)} samples")
            return processed_data

        except Exception as e:
            self.logger.error(f"❌ Data preprocessing failed: {e}")
            return data

    def _validate_data_quality(self, data: pd.DataFrame, config: PreprocessingConfig):
        """
        Validate data quality and raise warnings for issues.

        Args:
            data: Data to validate
            config: Preprocessing configuration
        """
        try:
            quality_score = self._calculate_data_quality_score(data)

            if quality_score < config.min_data_quality_score:
                self.logger.warning(f"⚠️ Low data quality score: {quality_score".2f"}")

            # Check for common issues
            self._check_common_data_issues(data)

        except Exception as e:
            self.logger.warning(f"⚠️ Data quality validation failed: {e}")

    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """
        Calculate a data quality score.

        Args:
            data: Data to assess

        Returns:
            Quality score (0-1)
        """
        try:
            score = 1.0

            # Check for missing values
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            score -= missing_ratio * 0.3

            # Check for infinite values
            inf_count = np.isinf(data.values).sum()
            if inf_count > 0:
                score -= 0.2

            # Check for reasonable data ranges
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                # Check for constant columns
                constant_columns = (numeric_data.std() == 0).sum()
                score -= (constant_columns / len(numeric_data.columns)) * 0.2

            # Check for temporal consistency if datetime index
            if isinstance(data.index, pd.DatetimeIndex):
                time_diffs = data.index.to_series().diff().dropna()
                unique_diffs = time_diffs.unique()
                if len(unique_diffs) > 5:  # Too many different intervals
                    score -= 0.2

            return max(0.0, min(1.0, score))

        except Exception as e:
            self.logger.warning(f"⚠️ Data quality score calculation failed: {e}")
            return 0.5

    def _check_common_data_issues(self, data: pd.DataFrame):
        """
        Check for common data issues and log warnings.

        Args:
            data: Data to check
        """
        try:
            # Check price relationships
            if all(col in data.columns for col in ['high', 'low', 'close', 'open']):
                invalid_high = (data['high'] < data[['low', 'open', 'close']].max(axis=1)).sum()
                invalid_low = (data['low'] > data[['high', 'open', 'close']].min(axis=1)).sum()

                if invalid_high > 0:
                    self.logger.warning(f"⚠️ Found {invalid_high} invalid high prices")
                if invalid_low > 0:
                    self.logger.warning(f"⚠️ Found {invalid_low} invalid low prices")

            # Check for extreme outliers
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
                extreme_outliers = (z_scores > 5).sum().sum()
                if extreme_outliers > 0:
                    self.logger.warning(f"⚠️ Found {extreme_outliers} extreme outliers (>5σ)")

        except Exception as e:
            self.logger.warning(f"⚠️ Common data issues check failed: {e}")

    def _remove_duplicate_timestamps(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate timestamps from data.

        Args:
            data: Data with potential duplicate timestamps

        Returns:
            Data without duplicate timestamps
        """
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                return data

            # Remove duplicate timestamps, keeping the last occurrence
            duplicates_before = data.index.duplicated().sum()
            if duplicates_before > 0:
                data = data[~data.index.duplicated(keep='last')]
                self.logger.info(f"✅ Removed {duplicates_before} duplicate timestamps")

            return data

        except Exception as e:
            self.logger.warning(f"⚠️ Duplicate timestamp removal failed: {e}")
            return data

    def _fill_missing_values(self, data: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
        """
        Fill missing values using specified method.

        Args:
            data: Data with missing values
            config: Preprocessing configuration

        Returns:
            Data with missing values filled
        """
        try:
            filled_data = data.copy()
            missing_before = filled_data.isnull().sum().sum()

            if missing_before == 0:
                return filled_data

            for column in filled_data.columns:
                if filled_data[column].isnull().any():
                    if config.fill_method == 'forward':
                        filled_data[column] = filled_data[column].fillna(method='ffill')
                    elif config.fill_method == 'backward':
                        filled_data[column] = filled_data[column].fillna(method='bfill')
                    elif config.fill_method == 'interpolate':
                        filled_data[column] = filled_data[column].interpolate(method='linear')
                    elif config.fill_method == 'median':
                        median_val = filled_data[column].median()
                        filled_data[column] = filled_data[column].fillna(median_val)

                    # Final fallback for any remaining NaN
                    if filled_data[column].isnull().any():
                        if filled_data[column].dtype in ['int64', 'float64']:
                            # For numeric columns, use 0 for price/volume data, median otherwise
                            if 'price' in column.lower() or 'volume' in column.lower():
                                filled_data[column] = filled_data[column].fillna(0.0)
                            else:
                                filled_data[column] = filled_data[column].fillna(filled_data[column].median())
                        else:
                            filled_data[column] = filled_data[column].fillna('unknown')

            missing_after = filled_data.isnull().sum().sum()
            filled_count = missing_before - missing_after

            if filled_count > 0:
                self.logger.info(f"✅ Filled {filled_count} missing values using {config.fill_method} method")

            return filled_data

        except Exception as e:
            self.logger.warning(f"⚠️ Missing value filling failed: {e}")
            return data

    def _remove_outliers(self, data: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
        """
        Remove outliers using statistical methods.

        Args:
            data: Data with potential outliers
            config: Preprocessing configuration

        Returns:
            Data with outliers removed
        """
        try:
            cleaned_data = data.copy()
            outliers_removed = 0

            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns

            for column in numeric_columns:
                if column in cleaned_data.columns:
                    # Calculate z-scores
                    column_data = cleaned_data[column].dropna()
                    if len(column_data) < 10:  # Need sufficient data
                        continue

                    mean_val = column_data.mean()
                    std_val = column_data.std()

                    if std_val == 0:
                        continue  # No variation, skip

                    z_scores = np.abs((cleaned_data[column] - mean_val) / std_val)

                    # Mark outliers
                    outlier_mask = z_scores > config.outlier_threshold

                    if outlier_mask.any():
                        # Replace outliers with median (less aggressive than removal)
                        median_val = cleaned_data[column].median()
                        cleaned_data.loc[outlier_mask, column] = median_val
                        outliers_removed += outlier_mask.sum()

            if outliers_removed > 0:
                self.logger.info(f"✅ Handled {outliers_removed} outliers (threshold: {config.outlier_threshold}σ)")

            return cleaned_data

        except Exception as e:
            self.logger.warning(f"⚠️ Outlier removal failed: {e}")
            return data

    def _normalize_features(self, data: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
        """
        Normalize features using specified method.

        Args:
            data: Data to normalize
            config: Preprocessing configuration

        Returns:
            Normalized data
        """
        try:
            normalized_data = data.copy()
            numeric_columns = normalized_data.select_dtypes(include=[np.number]).columns

            for column in numeric_columns:
                if column in normalized_data.columns:
                    column_data = normalized_data[column].dropna()

                    if len(column_data) < 2:
                        continue

                    if config.normalization_method == 'standard':
                        # Z-score normalization
                        mean_val = column_data.mean()
                        std_val = column_data.std()
                        if std_val > 0:
                            normalized_data[column] = (normalized_data[column] - mean_val) / std_val

                    elif config.normalization_method == 'minmax':
                        # Min-max normalization
                        min_val = column_data.min()
                        max_val = column_data.max()
                        if max_val > min_val:
                            normalized_data[column] = (normalized_data[column] - min_val) / (max_val - min_val)

                    elif config.normalization_method == 'robust':
                        # Robust normalization using median and IQR
                        median_val = column_data.median()
                        q75 = column_data.quantile(0.75)
                        q25 = column_data.quantile(0.25)
                        iqr = q75 - q25
                        if iqr > 0:
                            normalized_data[column] = (normalized_data[column] - median_val) / iqr

            self.logger.info(f"✅ Normalized features using {config.normalization_method} method")
            return normalized_data

        except Exception as e:
            self.logger.warning(f"⚠️ Feature normalization failed: {e}")
            return data

    def _ensure_required_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure required columns exist and are properly formatted.

        Args:
            data: Data to check

        Returns:
            Data with required columns
        """
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']

            for col in required_columns:
                if col not in data.columns:
                    self.logger.warning(f"⚠️ Required column '{col}' missing, creating fallback")
                    if col == 'volume':
                        data[col] = 1000  # Default volume
                    else:
                        # Use close price as fallback for OHLC
                        data[col] = data.get('close', 1000)

            # Ensure proper data types
            for col in required_columns:
                if col in data.columns:
                    if data[col].dtype in ['object', 'string']:
                        # Try to convert to numeric
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                        # Fill any conversion failures
                        if data[col].isnull().any():
                            if col == 'volume':
                                data[col] = data[col].fillna(1000)
                            else:
                                data[col] = data[col].fillna(data[col].median())

            return data

        except Exception as e:
            self.logger.warning(f"⚠️ Required columns check failed: {e}")
            return data

    def _final_validation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform final validation on processed data.

        Args:
            data: Data to validate

        Returns:
            Validated data
        """
        try:
            # Check for any remaining issues
            if data.isnull().sum().sum() > 0:
                self.logger.warning(f"⚠️ Still have {data.isnull().sum().sum()} missing values after preprocessing")

            if np.isinf(data.values).sum() > 0:
                self.logger.warning(f"⚠️ Still have {np.isinf(data.values).sum()} infinite values after preprocessing")

            # Ensure we have minimum required data
            if len(data) < 100:
                self.logger.warning(f"⚠️ Very small dataset: {len(data)} samples (minimum recommended: 100)")

            return data

        except Exception as e:
            self.logger.warning(f"⚠️ Final validation failed: {e}")
            return data