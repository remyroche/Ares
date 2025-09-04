"""Refactored Raw Data Quality Checker
This is a refactored version of raw_data_quality_checker.py that uses the extracted components.
"""
import asyncio
import functools
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Optional, Tuple

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from src.utils.logger import system_logger
from src.utils.warning_symbols import critical

# Import the extracted components
from src.training.steps.data_quality_components import (
    QualityMetricsCalculator,
    DataIntegrityChecker,
    AnomalyDetector
)


class RawDataQualityChecker:
    """Refactored raw data quality checker using extracted components.
    
    This class now delegates specific quality checking tasks to specialized components
    while maintaining the same interface for backward compatibility.
    """
    
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.logger = system_logger.getChild("RawDataQualityChecker")
        self.config = config or self._get_default_config()
        
        # Initialize extracted components
        self.metrics_calculator = QualityMetricsCalculator(config=self.config)
        self.integrity_checker = DataIntegrityChecker(config=self.config)
        self.anomaly_detector = AnomalyDetector(config=self.config)
        
    @staticmethod
    def ensure_datetime_index(func):
        """Decorator to ensure DataFrame has datetime index before processing."""
        @functools.wraps(func)
        def wrapper(self, data: pd.DataFrame, *args, **kwargs):
            if not isinstance(data.index, pd.DatetimeIndex):
                self.logger.warning(f"⚠️ {func.__name__}: Data does not have datetime index, attempting to fix...")
                
                # Create a mock results dict for the fix_datetime_index method
                mock_results = {"warnings": [], "critical_issues": []}
                fixed_data = self._fix_datetime_index(data, mock_results)
                
                if fixed_data is not None:
                    self.logger.info(f"✅ {func.__name__}: Successfully created datetime index")
                    data = fixed_data
                else:
                    self.logger.error(f"❌ {func.__name__}: Failed to create datetime index")
                    # Return a safe fallback result
                    if func.__name__ == "validate_raw_data":
                        return {
                            "validation_passed": False,
                            "critical_issues": ["Failed to create datetime index"],
                            "warnings": [],
                            "data_quality_score": 0.0,
                            "symbol": kwargs.get("symbol", "UNKNOWN"),
                            "exchange": kwargs.get("exchange", "UNKNOWN"),
                            "timestamp": datetime.now().isoformat(),
                            "data_shape": data.shape,
                        }, data
                    return None
                    
            return func(self, data, *args, **kwargs)
        return wrapper
        
    @staticmethod
    def validate_data_structure(func):
        """Decorator to validate basic data structure before processing."""
        @functools.wraps(func)
        def wrapper(self, data: pd.DataFrame, *args, **kwargs):
            # Check if data is empty
            if data is None or data.empty:
                self.logger.error(f"❌ {func.__name__}: Empty or None data provided")
                if func.__name__ == "validate_raw_data":
                    return {
                        "validation_passed": False,
                        "critical_issues": ["Empty or None data provided"],
                        "warnings": [],
                        "data_quality_score": 0.0,
                        "symbol": kwargs.get("symbol", "UNKNOWN"),
                        "exchange": kwargs.get("exchange", "UNKNOWN"),
                        "timestamp": datetime.now().isoformat(),
                        "data_shape": (0, 0) if data is None else data.shape,
                    }, data if data is not None else pd.DataFrame()
                return None
                
            # Check for required columns
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"❌ {func.__name__}: Missing required columns: {missing_columns}")
                if func.__name__ == "validate_raw_data":
                    return {
                        "validation_passed": False,
                        "critical_issues": [f"Missing required columns: {missing_columns}"],
                        "warnings": [],
                        "data_quality_score": 0.0,
                        "symbol": kwargs.get("symbol", "UNKNOWN"),
                        "exchange": kwargs.get("exchange", "UNKNOWN"),
                        "timestamp": datetime.now().isoformat(),
                        "data_shape": data.shape,
                    }, data
                return None
                
            return func(self, data, *args, **kwargs)
        return wrapper
        
    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for quality checks."""
        return {
            # Critical thresholds that will fail validation
            "critical_thresholds": {
                "min_rows": 100,
                "max_missing_ratio": 0.5,
                "max_duplicate_ratio": 0.1,
                "max_negative_prices": 0.0,
                "max_zero_volume_ratio": 0.9,
            },
            # Warning thresholds that won't fail but indicate issues
            "warning_thresholds": {
                "high_missing_ratio": 0.1,
                "high_duplicate_ratio": 0.01,
                "irregular_interval_ratio": 0.05,
                "high_zero_volume_ratio": 0.5,
            },
            # Checks to perform
            "integrity_checks": {
                "check_ohlc_consistency": True,
                "check_negative_values": True,
                "check_extreme_movements": True,
                "check_time_gaps": True,
                "check_for_market_gaps": True,
            },
            # Quality metrics weights
            "score_weights": {
                "completeness": 0.25,
                "consistency": 0.25,
                "timeliness": 0.20,
                "validity": 0.20,
                "accuracy": 0.10
            }
        }
        
    @ensure_datetime_index
    @validate_data_structure
    def validate_raw_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        auto_fix: bool = False,
        timeframe: str | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """
        Validate raw market data using extracted components.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Trading symbol
            exchange: Exchange name
            auto_fix: Whether to attempt automatic fixes
            timeframe: Expected timeframe
            
        Returns:
            Tuple of (validation_results, processed_data)
        """
        self.logger.info(
            f"🔍 Starting raw data validation for {symbol} on {exchange}..."
        )
        
        # Initialize results dictionary
        results = {
            "validation_passed": True,
            "critical_issues": [],
            "warnings": [],
            "data_quality_score": 0.0,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now().isoformat(),
            "data_shape": data.shape,
            "detailed_analysis": {},
        }
        
        # Determine timeframe if not provided
        if timeframe is None:
            timeframe = self._determine_timeframe_from_data(data)
            results["detected_timeframe"] = timeframe
            
        # Step 1: Data Integrity Checks using DataIntegrityChecker
        self.logger.info("📊 Step 1: Checking data integrity...")
        integrity_valid, integrity_results = self.integrity_checker.validate_data_integrity(data)
        results["critical_issues"].extend(integrity_results.get("critical_issues", []))
        results["warnings"].extend(integrity_results.get("warnings", []))
        results["detailed_analysis"]["integrity"] = integrity_results.get("detailed_analysis", {})
        
        if not integrity_valid:
            results["validation_passed"] = False
            
        # Step 2: Market-specific validation
        market_valid, market_results = self.integrity_checker.validate_market_specific_issues(data)
        results["warnings"].extend(market_results.get("warnings", []))
        results["detailed_analysis"]["market_specific"] = market_results.get("detailed_analysis", {})
        
        # Step 3: Anomaly Detection using AnomalyDetector
        self.logger.info("🔍 Step 3: Detecting anomalies...")
        anomaly_results = self.anomaly_detector.detect_anomalies(data)
        
        if anomaly_results["summary"].get("total_anomalies", 0) > 0:
            results["warnings"].append(
                f"Detected {anomaly_results['summary']['total_anomalies']} anomalies "
                f"in columns: {', '.join(anomaly_results['summary']['columns_with_anomalies'])}"
            )
        results["detailed_analysis"]["anomalies"] = anomaly_results
        
        # Step 4: Calculate Quality Metrics using QualityMetricsCalculator
        self.logger.info("📈 Step 4: Calculating quality metrics...")
        quality_report = self.metrics_calculator.generate_quality_report(
            data, symbol, exchange, include_recommendations=True
        )
        
        results["detailed_analysis"]["quality_metrics"] = quality_report["metrics"]
        results["recommendations"] = quality_report["recommendations"]
        
        # Calculate final quality score
        results["data_quality_score"] = self.metrics_calculator.calculate_quality_score(results)
        
        # Determine if validation passed
        if results["critical_issues"]:
            results["validation_passed"] = False
            
        # Auto-fix if requested and validation failed
        processed_data = data
        if auto_fix and not results["validation_passed"]:
            self.logger.info("🔧 Attempting to auto-fix data quality issues...")
            processed_data, fix_results = self.validate_and_fix_data_quality_issues(
                data, symbol, exchange
            )
            results["auto_fix_applied"] = True
            results["fix_results"] = fix_results
        
        # Log final results
        self._log_validation_summary(results)
        
        return results, processed_data
        
    def validate_and_fix_data_quality_issues(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Validate and attempt to fix data quality issues.
        
        Args:
            data: DataFrame to validate and fix
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            Tuple of (fixed_data, fix_results)
        """
        fix_results = {
            "fixes_applied": [],
            "fixes_failed": [],
            "data_modified": False
        }
        
        fixed_data = data.copy()
        
        try:
            # Fix datetime index if needed
            if not isinstance(fixed_data.index, pd.DatetimeIndex):
                temp_results = {"warnings": [], "critical_issues": []}
                fixed_index_data = self._fix_datetime_index(fixed_data, temp_results)
                if fixed_index_data is not None:
                    fixed_data = fixed_index_data
                    fix_results["fixes_applied"].append("datetime_index")
                    fix_results["data_modified"] = True
                else:
                    fix_results["fixes_failed"].append("datetime_index")
                    
            # Remove duplicates
            initial_len = len(fixed_data)
            fixed_data = fixed_data[~fixed_data.index.duplicated(keep="first")]
            if len(fixed_data) < initial_len:
                fix_results["fixes_applied"].append(
                    f"removed_{initial_len - len(fixed_data)}_duplicates"
                )
                fix_results["data_modified"] = True
                
            # Handle missing values
            missing_before = fixed_data.isna().sum().sum()
            if missing_before > 0:
                # Forward fill for time series continuity
                fixed_data = fixed_data.fillna(method="ffill", limit=5)
                # Fill remaining with 0 for numeric columns
                numeric_cols = fixed_data.select_dtypes(include=[np.number]).columns
                fixed_data[numeric_cols] = fixed_data[numeric_cols].fillna(0)
                
                missing_after = fixed_data.isna().sum().sum()
                if missing_after < missing_before:
                    fix_results["fixes_applied"].append(
                        f"filled_{missing_before - missing_after}_missing_values"
                    )
                    fix_results["data_modified"] = True
                    
            # Fix OHLC inconsistencies
            ohlc_cols = ["open", "high", "low", "close"]
            if all(col in fixed_data.columns for col in ohlc_cols):
                # Ensure high is max of OHLC
                fixed_data["high"] = fixed_data[ohlc_cols].max(axis=1)
                # Ensure low is min of OHLC
                fixed_data["low"] = fixed_data[ohlc_cols].min(axis=1)
                fix_results["fixes_applied"].append("ohlc_consistency")
                fix_results["data_modified"] = True
                
            # Fix negative prices by setting to previous value
            for col in ohlc_cols:
                if col in fixed_data.columns:
                    negative_mask = fixed_data[col] < 0
                    if negative_mask.any():
                        fixed_data.loc[negative_mask, col] = fixed_data[col].shift(1)
                        fix_results["fixes_applied"].append(f"fixed_negative_{col}")
                        fix_results["data_modified"] = True
                        
        except Exception as e:
            self.logger.error(f"Error during auto-fix: {e}")
            fix_results["fixes_failed"].append(str(e))
            
        return fixed_data, fix_results
        
    def _fix_datetime_index(
        self,
        data: pd.DataFrame,
        results: dict[str, Any]
    ) -> pd.DataFrame | None:
        """Attempt to create datetime index from timestamp column."""
        try:
            # Check for timestamp column
            timestamp_cols = [col for col in data.columns if "timestamp" in col.lower()]
            
            if not timestamp_cols:
                results["critical_issues"].append("No timestamp column found")
                return None
                
            timestamp_col = timestamp_cols[0]
            
            # Try to convert to datetime
            if pd.api.types.is_numeric_dtype(data[timestamp_col]):
                # Assume milliseconds if large number
                if data[timestamp_col].max() > 1e10:
                    data.index = pd.to_datetime(data[timestamp_col], unit="ms", utc=True)
                else:
                    data.index = pd.to_datetime(data[timestamp_col], unit="s", utc=True)
            else:
                data.index = pd.to_datetime(data[timestamp_col], utc=True)
                
            # Sort by index
            data = data.sort_index()
            
            return data
            
        except Exception as e:
            results["critical_issues"].append(f"Failed to create datetime index: {e}")
            return None
            
    def _determine_timeframe_from_data(self, data: pd.DataFrame) -> str:
        """Determine timeframe from data intervals."""
        if not isinstance(data.index, pd.DatetimeIndex) or len(data) < 2:
            return "unknown"
            
        # Calculate time differences
        time_diffs = data.index.to_series().diff().dropna()
        
        # Get the mode (most common interval)
        if len(time_diffs) == 0:
            return "unknown"
            
        mode_interval = time_diffs.mode()
        if len(mode_interval) == 0:
            mode_interval = time_diffs.median()
        else:
            mode_interval = mode_interval[0]
            
        # Convert to minutes
        minutes = mode_interval.total_seconds() / 60
        
        # Map to standard timeframes
        timeframe_map = {
            1: "1m",
            3: "3m",
            5: "5m",
            15: "15m",
            30: "30m",
            60: "1h",
            240: "4h",
            1440: "1d",
        }
        
        # Find closest match
        closest_tf = min(timeframe_map.keys(), key=lambda x: abs(x - minutes))
        
        # Check if it's close enough (within 10%)
        if abs(closest_tf - minutes) / closest_tf < 0.1:
            return timeframe_map[closest_tf]
        else:
            return f"{int(minutes)}m"
            
    def _log_validation_summary(self, results: dict[str, Any]) -> None:
        """Log a summary of validation results."""
        status = "✅ PASSED" if results["validation_passed"] else "❌ FAILED"
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 Validation Summary: {status}")
        self.logger.info(f"Symbol: {results['symbol']}, Exchange: {results['exchange']}")
        self.logger.info(f"Data Shape: {results['data_shape']}")
        self.logger.info(f"Quality Score: {results['data_quality_score']:.2f}")
        
        if results["critical_issues"]:
            self.logger.error(f"Critical Issues ({len(results['critical_issues'])}):")
            for issue in results["critical_issues"]:
                self.logger.error(f"  - {issue}")
                
        if results["warnings"]:
            self.logger.warning(f"Warnings ({len(results['warnings'])}):")
            for warning in results["warnings"][:5]:  # Show first 5
                self.logger.warning(f"  - {warning}")
            if len(results["warnings"]) > 5:
                self.logger.warning(f"  ... and {len(results['warnings']) - 5} more")
                
        if results.get("recommendations"):
            self.logger.info(f"Recommendations:")
            for rec in results["recommendations"]:
                self.logger.info(f"  - {rec}")
                
        self.logger.info(f"{'='*60}\n")
        
    def get_data_quality_report(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str
    ) -> dict[str, Any]:
        """
        Generate a comprehensive data quality report.
        
        This method provides backward compatibility by delegating to
        the QualityMetricsCalculator component.
        """
        return self.metrics_calculator.generate_quality_report(
            data, symbol, exchange, include_recommendations=True
        )


# Convenience functions for backward compatibility
def validate_raw_data_quality(
    data: pd.DataFrame,
    symbol: str,
    exchange: str,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Convenience function to validate raw data quality."""
    checker = RawDataQualityChecker(config)
    return checker.validate_raw_data(data, symbol, exchange)


def validate_and_fix_data_quality_issues(
    data: pd.DataFrame,
    symbol: str,
    exchange: str,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convenience function to validate and fix data quality issues."""
    checker = RawDataQualityChecker(config)
    return checker.validate_and_fix_data_quality_issues(data, symbol, exchange)