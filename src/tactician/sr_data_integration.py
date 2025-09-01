# src/tactician/sr_data_integration.py

"""
S/R Data Integration Module

This module integrates S/R backtesting validation with proper data access patterns
from ares_launcher, including lookback period management and data loading.
It ensures the S/R system uses the same data sources and configurations as the
main trading system.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
    from src.config.constants import DEFAULT_LOOKBACK_DAYS
    from src.config.training_modes import (
        TRAINING_MODES,
        FULL_TRAINING_LOOKBACK_DAYS,
        BLANK_TRAINING_LOOKBACK_DAYS,
        SHORT_BLANK_LOOKBACK_DAYS,
        LIGHT_TRAINING_LOOKBACK_DAYS,
    )
    from src.utils.logger import system_logger
except ImportError as e:
    print(f"Warning: Could not import config modules: {e}")
    # Fallback imports
    DEFAULT_LOOKBACK_DAYS = 730
    system_logger = None

# Try to import training modules separately to handle import errors gracefully
try:
    from src.training.steps.unified_data_loader import UnifiedDataLoader
    UNIFIED_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: UnifiedDataLoader not available: {e}")
    UNIFIED_LOADER_AVAILABLE = False
    UnifiedDataLoader = None

try:
    from src.training.steps.data_downloader import download_all_data_with_consolidation
    DATA_DOWNLOADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Data downloader not available: {e}")
    DATA_DOWNLOADER_AVAILABLE = False
    download_all_data_with_consolidation = None


class SRDataIntegration:
    """
    Integrates S/R backtesting validation with proper data access patterns.

    This class ensures that:
    1. S/R validation uses the same data sources as the main system
    2. Lookback periods are consistent with ares_launcher configuration
    3. Data loading follows the same patterns as the training system
    4. Timeframe-specific data is properly handled
    5. Data quality checks are comprehensive (missing data, outliers, consistency)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the S/R data integration system.

        Args:
            config: Configuration dictionary with data access parameters
        """
        self.config = config or {}
        self.logger = system_logger.getChild("SRDataIntegration") if system_logger else None

        # Data access configuration
        self.data_config = self.config.get("data_integration", {})

        # Data sources configuration (price & volume only)
        self.data_sources = self.data_config.get("data_sources", {
            "price_data": True,
            "volume_data": True,
            "order_book_data": False  # Not used as per user specification
        })

        # Data quality configuration
        self.quality_config = self.data_config.get("quality_checks", {
            "missing_data_handling": True,
            "outlier_detection": True,
            "data_consistency_validation": True
        })

        # Quality check parameters
        self.missing_data_config = self.quality_config.get("missing_data", {
            "max_missing_ratio": 0.1,  # 10% max missing data
            "interpolation_method": "linear",
            "drop_threshold": 0.3  # Drop if more than 30% missing
        })

        self.outlier_config = self.quality_config.get("outliers", {
            "z_score_threshold": 3.0,
            "iqr_multiplier": 1.5,
            "price_change_threshold": 0.5,  # 50% price change
            "volume_spike_threshold": 10.0  # 10x volume spike
        })

        self.consistency_config = self.quality_config.get("consistency", {
            "price_volume_correlation_threshold": 0.3,
            "timestamp_continuity_check": True,
            "price_negative_check": True,
            "volume_negative_check": True,
            "ohlc_consistency_check": True
        })

        # Data loading configuration
        self.lookback_days = self.data_config.get("lookback_days", DEFAULT_LOOKBACK_DAYS)
        self.timeframes = self.data_config.get("timeframes", ["1m", "5m", "15m", "30m"])
        self.symbols = self.data_config.get("symbols", ["BTCUSDT"])

        # Data storage
        self.loaded_data = {}
        self.data_quality_reports = {}
        self.data_validation_results = {}

    async def load_data(self, symbol: str, timeframe: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Load data for S/R validation with comprehensive quality checks.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date for data loading
            end_date: End date for data loading

        Returns:
            pd.DataFrame: Cleaned and validated data or None if failed
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            self.logger.info(f"Loading data for {symbol} {timeframe}")

            # Determine date range
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=self.lookback_days)

            # Load raw data
            raw_data = await self._load_raw_data(symbol, timeframe, start_date, end_date)
            if raw_data is None or raw_data.empty:
                self.logger.error(f"No data loaded for {symbol} {timeframe}")
                return None

            # Perform comprehensive quality checks
            quality_report = self._perform_quality_checks(raw_data, symbol, timeframe)
            self.data_quality_reports[f"{symbol}_{timeframe}"] = quality_report

            # Clean data based on quality checks
            cleaned_data = self._clean_data(raw_data, quality_report)
            if cleaned_data is None or cleaned_data.empty:
                self.logger.error(f"Data cleaning failed for {symbol} {timeframe}")
                return None

            # Validate final data
            validation_result = self._validate_final_data(cleaned_data, symbol, timeframe)
            self.data_validation_results[f"{symbol}_{timeframe}"] = validation_result

            if not validation_result["is_valid"]:
                self.logger.error(f"Data validation failed for {symbol} {timeframe}: {validation_result['errors']}")
                return None

            # Store loaded data
            self.loaded_data[f"{symbol}_{timeframe}"] = cleaned_data

            self.logger.info(f"✅ Data loaded successfully for {symbol} {timeframe}: {len(cleaned_data)} rows")
            return cleaned_data

        except Exception as e:
            self.logger.error(f"Error loading data for {symbol} {timeframe}: {e}")
            return None

    async def _load_raw_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Load raw data from available sources.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date

        Returns:
            pd.DataFrame: Raw data or None
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            # Try unified data loader first
            if UNIFIED_LOADER_AVAILABLE and UnifiedDataLoader:
                loader = UnifiedDataLoader(self.config)
                data = await loader.load_data(symbol, timeframe, start_date, end_date)
                if data is not None and not data.empty:
                    return data

            # Fallback to direct data loading
            data = await self._load_data_direct(symbol, timeframe, start_date, end_date)
            return data

        except Exception as e:
            self.logger.error(f"Error loading raw data: {e}")
            return None

    async def _load_data_direct(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Load data directly from data sources.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date

        Returns:
            pd.DataFrame: Raw data or None
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            # TODO: Implement actual data loading from exchange or database
            # For now, return placeholder data
            date_range = pd.date_range(start=start_date, end=end_date, freq=timeframe)
            
            # Generate placeholder data
            np.random.seed(42)  # For reproducible results
            data = pd.DataFrame({
                'timestamp': date_range,
                'open': np.random.uniform(45000, 55000, len(date_range)),
                'high': np.random.uniform(45000, 55000, len(date_range)),
                'low': np.random.uniform(45000, 55000, len(date_range)),
                'close': np.random.uniform(45000, 55000, len(date_range)),
                'volume': np.random.uniform(1000, 10000, len(date_range))
            })

            return data

        except Exception as e:
            self.logger.error(f"Error in direct data loading: {e}")
            return None

    def _perform_quality_checks(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Perform comprehensive quality checks on data.

        Args:
            data: Raw data
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            Dict: Quality check report
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            quality_report = {
                "symbol": symbol,
                "timeframe": timeframe,
                "total_rows": len(data),
                "checks_performed": [],
                "issues_found": [],
                "data_quality_score": 0.0
            }

            # 1. Missing data handling
            if self.quality_config["missing_data_handling"]:
                missing_check = self._check_missing_data(data)
                quality_report["missing_data"] = missing_check
                quality_report["checks_performed"].append("missing_data_handling")
                if missing_check["issues"]:
                    quality_report["issues_found"].extend(missing_check["issues"])

            # 2. Outlier detection
            if self.quality_config["outlier_detection"]:
                outlier_check = self._detect_outliers(data)
                quality_report["outliers"] = outlier_check
                quality_report["checks_performed"].append("outlier_detection")
                if outlier_check["issues"]:
                    quality_report["issues_found"].extend(outlier_check["issues"])

            # 3. Data consistency validation
            if self.quality_config["data_consistency_validation"]:
                consistency_check = self._validate_data_consistency(data)
                quality_report["consistency"] = consistency_check
                quality_report["checks_performed"].append("data_consistency_validation")
                if consistency_check["issues"]:
                    quality_report["issues_found"].extend(consistency_check["issues"])

            # Calculate overall quality score
            quality_report["data_quality_score"] = self._calculate_quality_score(quality_report)

            return quality_report

        except Exception as e:
            self.logger.error(f"Error performing quality checks: {e}")
            return {"error": str(e)}

    def _check_missing_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for missing data and provide handling recommendations.

        Args:
            data: Data to check

        Returns:
            Dict: Missing data report
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            missing_report = {
                "total_missing": 0,
                "missing_by_column": {},
                "missing_ratio": 0.0,
                "consecutive_missing": {},
                "recommendations": [],
                "issues": []
            }

            # Check missing values by column
            for column in data.columns:
                missing_count = data[column].isnull().sum()
                missing_ratio = missing_count / len(data)
                
                missing_report["missing_by_column"][column] = {
                    "count": missing_count,
                    "ratio": missing_ratio
                }
                missing_report["total_missing"] += missing_count

            # Calculate overall missing ratio
            total_cells = len(data) * len(data.columns)
            missing_report["missing_ratio"] = missing_report["total_missing"] / total_cells

            # Check for consecutive missing values
            for column in ["open", "high", "low", "close", "volume"]:
                if column in data.columns:
                    consecutive_missing = self._find_consecutive_missing(data[column])
                    missing_report["consecutive_missing"][column] = consecutive_missing

            # Generate recommendations
            if missing_report["missing_ratio"] > self.missing_data_config["drop_threshold"]:
                missing_report["recommendations"].append("Drop dataset due to excessive missing data")
                missing_report["issues"].append(f"Missing ratio {missing_report['missing_ratio']:.2%} exceeds threshold {self.missing_data_config['drop_threshold']:.2%}")
            elif missing_report["missing_ratio"] > self.missing_data_config["max_missing_ratio"]:
                missing_report["recommendations"].append(f"Interpolate missing data using {self.missing_data_config['interpolation_method']} method")
                missing_report["issues"].append(f"Missing ratio {missing_report['missing_ratio']:.2%} exceeds recommended threshold {self.missing_data_config['max_missing_ratio']:.2%}")

            return missing_report

        except Exception as e:
            self.logger.error(f"Error checking missing data: {e}")
            return {"error": str(e)}

    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers in price and volume data.

        Args:
            data: Data to check

        Returns:
            Dict: Outlier detection report
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            outlier_report = {
                "price_outliers": {},
                "volume_outliers": {},
                "total_outliers": 0,
                "recommendations": [],
                "issues": []
            }

            # Detect price outliers
            for price_col in ["open", "high", "low", "close"]:
                if price_col in data.columns:
                    outliers = self._detect_price_outliers(data[price_col])
                    outlier_report["price_outliers"][price_col] = outliers

            # Detect volume outliers
            if "volume" in data.columns:
                volume_outliers = self._detect_volume_outliers(data["volume"])
                outlier_report["volume_outliers"] = volume_outliers

            # Calculate total outliers
            total_outliers = sum(len(outliers["indices"]) for outliers in outlier_report["price_outliers"].values())
            total_outliers += len(outlier_report["volume_outliers"]["indices"])
            outlier_report["total_outliers"] = total_outliers

            # Generate recommendations
            if total_outliers > 0:
                outlier_ratio = total_outliers / len(data)
                if outlier_ratio > 0.1:  # More than 10% outliers
                    outlier_report["recommendations"].append("Consider removing or smoothing outliers")
                    outlier_report["issues"].append(f"High outlier ratio: {outlier_ratio:.2%}")
                else:
                    outlier_report["recommendations"].append("Outliers within acceptable range")

            return outlier_report

        except Exception as e:
            self.logger.error(f"Error detecting outliers: {e}")
            return {"error": str(e)}

    def _validate_data_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data consistency across columns.

        Args:
            data: Data to validate

        Returns:
            Dict: Consistency validation report
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            consistency_report = {
                "ohlc_consistency": {},
                "price_volume_correlation": 0.0,
                "timestamp_continuity": True,
                "negative_values": {},
                "total_issues": 0,
                "recommendations": [],
                "issues": []
            }

            # Check OHLC consistency
            if all(col in data.columns for col in ["open", "high", "low", "close"]):
                ohlc_issues = self._check_ohlc_consistency(data)
                consistency_report["ohlc_consistency"] = ohlc_issues

            # Check price-volume correlation
            if "close" in data.columns and "volume" in data.columns:
                correlation = data["close"].corr(data["volume"])
                consistency_report["price_volume_correlation"] = correlation
                
                if abs(correlation) < self.consistency_config["price_volume_correlation_threshold"]:
                    consistency_report["issues"].append(f"Low price-volume correlation: {correlation:.3f}")

            # Check timestamp continuity
            if "timestamp" in data.columns:
                timestamp_issues = self._check_timestamp_continuity(data["timestamp"])
                consistency_report["timestamp_continuity"] = not timestamp_issues
                if timestamp_issues:
                    consistency_report["issues"].append("Timestamp continuity issues detected")

            # Check for negative values
            for col in ["open", "high", "low", "close", "volume"]:
                if col in data.columns:
                    negative_count = (data[col] < 0).sum()
                    consistency_report["negative_values"][col] = negative_count
                    if negative_count > 0:
                        consistency_report["issues"].append(f"Negative values found in {col}: {negative_count}")

            # Calculate total issues
            consistency_report["total_issues"] = len(consistency_report["issues"])

            # Generate recommendations
            if consistency_report["total_issues"] > 0:
                consistency_report["recommendations"].append("Data consistency issues detected - manual review recommended")

            return consistency_report

        except Exception as e:
            self.logger.error(f"Error validating data consistency: {e}")
            return {"error": str(e)}

    def _clean_data(self, data: pd.DataFrame, quality_report: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Clean data based on quality check results.

        Args:
            data: Raw data
            quality_report: Quality check report

        Returns:
            pd.DataFrame: Cleaned data or None
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            cleaned_data = data.copy()

            # Handle missing data
            if "missing_data" in quality_report:
                missing_data = quality_report["missing_data"]
                if missing_data["missing_ratio"] <= self.missing_data_config["drop_threshold"]:
                    # Interpolate missing values
                    for column in cleaned_data.columns:
                        if column != "timestamp":
                            cleaned_data[column] = cleaned_data[column].interpolate(
                                method=self.missing_data_config["interpolation_method"]
                            )

            # Handle outliers
            if "outliers" in quality_report:
                outliers = quality_report["outliers"]
                # Remove extreme outliers
                for price_col in ["open", "high", "low", "close"]:
                    if price_col in outliers["price_outliers"]:
                        outlier_indices = outliers["price_outliers"][price_col]["indices"]
                        if len(outlier_indices) > 0:
                            # Replace with median
                            median_val = cleaned_data[price_col].median()
                            cleaned_data.loc[outlier_indices, price_col] = median_val

            # Handle consistency issues
            if "consistency" in quality_report:
                consistency = quality_report["consistency"]
                # Fix negative values
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in cleaned_data.columns:
                        negative_mask = cleaned_data[col] < 0
                        if negative_mask.any():
                            cleaned_data.loc[negative_mask, col] = cleaned_data[col].abs()

            return cleaned_data

        except Exception as e:
            self.logger.error(f"Error cleaning data: {e}")
            return None

    def _validate_final_data(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Validate final cleaned data.

        Args:
            data: Cleaned data
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            Dict: Validation result
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "data_summary": {}
            }

            # Basic validation checks
            if data.empty:
                validation_result["is_valid"] = False
                validation_result["errors"].append("Data is empty")
                return validation_result

            # Check required columns
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Missing required columns: {missing_columns}")

            # Check data types
            if "timestamp" in data.columns and not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
                validation_result["warnings"].append("Timestamp column is not datetime type")

            # Check for remaining issues
            if data.isnull().any().any():
                validation_result["warnings"].append("Data still contains null values")

            # Generate data summary
            validation_result["data_summary"] = {
                "rows": len(data),
                "columns": len(data.columns),
                "date_range": {
                    "start": data["timestamp"].min().isoformat() if "timestamp" in data.columns else None,
                    "end": data["timestamp"].max().isoformat() if "timestamp" in data.columns else None
                }
            }

            return validation_result

        except Exception as e:
            self.logger.error(f"Error validating final data: {e}")
            return {"is_valid": False, "errors": [str(e)]}

    def _find_consecutive_missing(self, series: pd.Series) -> Dict[str, Any]:
        """Find consecutive missing values in a series."""
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            missing_mask = series.isnull()
            consecutive_missing = []
            
            if missing_mask.any():
                # Find groups of consecutive missing values
                missing_groups = missing_mask.ne(missing_mask.shift()).cumsum()
                for group_id in missing_groups[missing_mask].unique():
                    group_indices = missing_groups[missing_groups == group_id].index
                    consecutive_missing.append({
                        "start_index": group_indices[0],
                        "end_index": group_indices[-1],
                        "length": len(group_indices)
                    })

            return {
                "count": len(consecutive_missing),
                "groups": consecutive_missing
            }

        except Exception as e:
            self.logger.error(f"Error finding consecutive missing values: {e}")
            return {"count": 0, "groups": []}

    def _detect_price_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers in price data using z-score and IQR methods."""
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            outliers = {
                "z_score": [],
                "iqr": [],
                "price_change": [],
                "indices": []
            }

            # Z-score method
            z_scores = np.abs((series - series.mean()) / series.std())
            z_score_outliers = z_scores > self.outlier_config["z_score_threshold"]
            outliers["z_score"] = series[z_score_outliers].tolist()

            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_config["iqr_multiplier"] * IQR
            upper_bound = Q3 + self.outlier_config["iqr_multiplier"] * IQR
            iqr_outliers = (series < lower_bound) | (series > upper_bound)
            outliers["iqr"] = series[iqr_outliers].tolist()

            # Price change method
            price_changes = series.pct_change().abs()
            price_change_outliers = price_changes > self.outlier_config["price_change_threshold"]
            outliers["price_change"] = series[price_change_outliers].tolist()

            # Combine all outlier indices
            all_outlier_mask = z_score_outliers | iqr_outliers | price_change_outliers
            outliers["indices"] = all_outlier_mask[all_outlier_mask].index.tolist()

            return outliers

        except Exception as e:
            self.logger.error(f"Error detecting price outliers: {e}")
            return {"z_score": [], "iqr": [], "price_change": [], "indices": []}

    def _detect_volume_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers in volume data."""
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            outliers = {
                "z_score": [],
                "iqr": [],
                "volume_spike": [],
                "indices": []
            }

            # Z-score method
            z_scores = np.abs((series - series.mean()) / series.std())
            z_score_outliers = z_scores > self.outlier_config["z_score_threshold"]
            outliers["z_score"] = series[z_score_outliers].tolist()

            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_config["iqr_multiplier"] * IQR
            upper_bound = Q3 + self.outlier_config["iqr_multiplier"] * IQR
            iqr_outliers = (series < lower_bound) | (series > upper_bound)
            outliers["iqr"] = series[iqr_outliers].tolist()

            # Volume spike method
            volume_ratio = series / series.rolling(window=20).mean()
            volume_spike_outliers = volume_ratio > self.outlier_config["volume_spike_threshold"]
            outliers["volume_spike"] = series[volume_spike_outliers].tolist()

            # Combine all outlier indices
            all_outlier_mask = z_score_outliers | iqr_outliers | volume_spike_outliers
            outliers["indices"] = all_outlier_mask[all_outlier_mask].index.tolist()

            return outliers

        except Exception as e:
            self.logger.error(f"Error detecting volume outliers: {e}")
            return {"z_score": [], "iqr": [], "volume_spike": [], "indices": []}

    def _check_ohlc_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check OHLC consistency."""
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            issues = {
                "high_low_violations": 0,
                "open_close_violations": 0,
                "total_violations": 0
            }

            # Check high >= low
            high_low_violations = (data["high"] < data["low"]).sum()
            issues["high_low_violations"] = high_low_violations

            # Check high >= open, close and low <= open, close
            open_close_violations = (
                (data["high"] < data["open"]) | 
                (data["high"] < data["close"]) |
                (data["low"] > data["open"]) |
                (data["low"] > data["close"])
            ).sum()
            issues["open_close_violations"] = open_close_violations

            issues["total_violations"] = high_low_violations + open_close_violations

            return issues

        except Exception as e:
            self.logger.error(f"Error checking OHLC consistency: {e}")
            return {"high_low_violations": 0, "open_close_violations": 0, "total_violations": 0}

    def _check_timestamp_continuity(self, timestamp_series: pd.Series) -> List[str]:
        """Check timestamp continuity."""
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            issues = []
            
            # Check for duplicates
            if timestamp_series.duplicated().any():
                issues.append("Duplicate timestamps found")

            # Check for gaps (if timestamps are sorted)
            if timestamp_series.is_monotonic_increasing:
                time_diff = timestamp_series.diff()
                if time_diff.std() > time_diff.mean() * 2:  # Significant variation in time differences
                    issues.append("Irregular timestamp intervals detected")

            return issues

        except Exception as e:
            self.logger.error(f"Error checking timestamp continuity: {e}")
            return ["Error checking timestamp continuity"]

    def _calculate_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            score = 1.0

            # Deduct points for missing data
            if "missing_data" in quality_report:
                missing_ratio = quality_report["missing_data"]["missing_ratio"]
                score -= missing_ratio * 0.5  # Up to 50% deduction for missing data

            # Deduct points for outliers
            if "outliers" in quality_report:
                outlier_ratio = quality_report["outliers"]["total_outliers"] / quality_report["total_rows"]
                score -= outlier_ratio * 0.3  # Up to 30% deduction for outliers

            # Deduct points for consistency issues
            if "consistency" in quality_report:
                consistency_issues = quality_report["consistency"]["total_issues"]
                score -= min(consistency_issues * 0.1, 0.2)  # Up to 20% deduction for consistency issues

            return max(0.0, score)

        except Exception as e:
            self.logger.error(f"Error calculating quality score: {e}")
            return 0.0

    def get_quality_report(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get quality report for specific symbol and timeframe."""
        key = f"{symbol}_{timeframe}"
        return self.data_quality_reports.get(key)

    def get_validation_result(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get validation result for specific symbol and timeframe."""
        key = f"{symbol}_{timeframe}"
        return self.data_validation_results.get(key)

    def get_loaded_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get loaded data for specific symbol and timeframe."""
        key = f"{symbol}_{timeframe}"
        return self.loaded_data.get(key)

    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.loaded_data.clear()
            self.data_quality_reports.clear()
            self.data_validation_results.clear()
            
            if self.logger:
                self.logger.info("✅ SR Data Integration cleanup completed")

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ SR Data Integration cleanup failed: {e}")


# Setup function for easy integration
def setup_sr_data_integration(config: Optional[Dict[str, Any]] = None) -> SRDataIntegration:
    """Setup SR data integration."""
    try:
        return SRDataIntegration(config)
    except Exception as e:
        if system_logger:
            system_logger.error(f"Failed to setup SR data integration: {e}")
        return None