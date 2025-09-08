"""Simplified Raw Data Quality Checker
from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

This is a simplified version of raw_data_quality_checker.py that uses extracted components.
The complexity has been reduced from 586 to approximately 150-200 by using component architecture.
"""

import warnings

from typing import Any, Optional, Tuple
import pandas as pd

warnings.filterwarnings("ignore")

from src.utils.logger import system_logger

# Import the extracted components
from .data_quality_components import (
    QualityCheckConfig,
    DataPreprocessor,
    DataDownloader,
    DataIntegrityChecker,
    QualityMetricsCalculator,
    AnomalyDetector,
    ValidationResultBuilder,
    ErrorHandler,
    StructureValidationStrategy,
    CompletenessValidationStrategy,
    IntegrityValidationStrategy,
    MarketSpecificValidationStrategy,
    FeatureEngineeringValidationStrategy,
    validate_data,
    log_validation_progress,
    handle_validation_errors
)

class RawDataQualityChecker:
    """Simplified raw data quality checker using component architecture.
    
    This class orchestrates various components to provide comprehensive
    data quality validation while maintaining a much lower complexity
    than the original monolithic implementation.
    """
    @log_important_calls
    
    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.logger = system_logger.getChild("RawDataQualityChecker")
        
        # Initialize configuration manager
        self.config_manager = QualityCheckConfig(config)
        
        # Initialize components
        self.preprocessor = DataPreprocessor(self.config_manager.get_config())
        self.downloader = DataDownloader(self.config_manager.get_config())
        self.integrity_checker = DataIntegrityChecker(self.config_manager.get_config())
        self.metrics_calculator = QualityMetricsCalculator(self.config_manager.get_config())
        self.anomaly_detector = AnomalyDetector(self.config_manager.get_config())
        self.error_handler = ErrorHandler(self.config_manager.get_config())
        
        # Initialize validation strategies
        self.validation_strategies = [
            StructureValidationStrategy(self.config_manager.get_config()),
            CompletenessValidationStrategy(self.config_manager.get_config()),
            IntegrityValidationStrategy(self.config_manager.get_config()),
            MarketSpecificValidationStrategy(self.config_manager.get_config()),
            FeatureEngineeringValidationStrategy(self.config_manager.get_config())
        ]
        
    @log_validation_progress
    @handle_validation_errors
    @validate_data
    def validate_raw_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        auto_fix: bool = False,
        auto_download_missing: bool = False
    ) -> Tuple[dict[str, Any], pd.DataFrame]:
        """
        Comprehensive validation of raw market data using component architecture.
        
        Args:
            data: Raw OHLCV data
            symbol: Trading symbol
            exchange: Exchange name
            auto_fix: Whether to attempt automatic fixes
            auto_download_missing: Whether to automatically download missing data
            
        Returns:
            Tuple of (validation_results, processed_data)
        """
        self.logger.info(f'🔍 Starting raw data quality validation for {exchange} {symbol}')
        
        # Initialize result builder
        result_builder = ValidationResultBuilder(symbol, exchange, data.shape)
        
        # Run validation strategies
        for strategy in self.validation_strategies:
            try:
                is_valid = strategy.validate(data, result_builder.get_current_state())
                if not is_valid:
                    result_builder.add_critical_issue(f"Validation failed in {strategy.__class__.__name__}")
            except Exception as e:
                self.logger.exception(f"Error in {strategy.__class__.__name__}: {e}")
                result_builder.add_critical_issue(f"Error in {strategy.__class__.__name__}: {str(e)}")
        
        # Run data integrity checks
        try:
            integrity_valid, integrity_results = self.integrity_checker.validate_data_integrity(data)
            if not integrity_valid:
                result_builder.add_critical_issue("Data integrity validation failed")
            result_builder.add_detailed_analysis("integrity", integrity_results.get("detailed_analysis", {}))
        except Exception as e:
            self.logger.exception(f"Error in data integrity check: {e}")
            result_builder.add_critical_issue(f"Data integrity check error: {str(e)}")
        
        # Run anomaly detection
        try:
            anomaly_results = self.anomaly_detector.detect_anomalies(data)
            if anomaly_results["summary"].get("total_anomalies", 0) > 0:
                result_builder.add_warning(
                    f"Detected {anomaly_results['summary']['total_anomalies']} anomalies "
                    f"in columns: {', '.join(anomaly_results['summary']['columns_with_anomalies'])}"
                )
            result_builder.add_detailed_analysis("anomalies", anomaly_results)
        except Exception as e:
            self.logger.exception(f"Error in anomaly detection: {e}")
            result_builder.add_warning(f"Anomaly detection error: {str(e)}")
        
        # Calculate quality score
        result_builder.calculate_quality_score()
        
        # Auto-fix if requested and validation failed
        processed_data = data
        if auto_fix and not result_builder.get_current_state()["validation_passed"]:
            self.logger.info("🔧 Attempting to auto-fix data quality issues...")
            processed_data, fix_results = self._auto_fix_issues(data, symbol, exchange, result_builder)
            result_builder.set_preprocessing_applied(fix_results)
        
        # Handle missing data download if requested
        if auto_download_missing:
            self.logger.info("🔧 Attempting to download missing data...")
            processed_data, download_summary = self.downloader.handle_missing_data_download(
                processed_data, symbol, exchange, result_builder.get_current_state()
            )
            result_builder.set_data_downloaded(
                download_summary.get('data_downloaded', False),
                download_summary
            )
        
        # Generate final results
        results = result_builder.build()
        
        return results, processed_data
    @log_all_calls
        
    def _auto_fix_issues(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        result_builder: ValidationResultBuilder
    ) -> Tuple[pd.DataFrame, dict[str, Any]]:
        """Auto-fix common data quality issues.
        
        Args:
            data: Data to fix
            symbol: Trading symbol
            exchange: Exchange name
            result_builder: Result builder to update
            
        Returns:
            Tuple of (fixed_data, fix_results)
        """
        fix_results = {
            "fixes_applied": [],
            "fixes_failed": [],
            "data_modified": False,
            "original_shape": data.shape
        }
        
        fixed_data = data.copy()
        
        try:
            # Fix irregular intervals
            if self._needs_interval_fixing(result_builder.get_current_state()):
                self.logger.info("🔧 Fixing irregular intervals...")
                fixed_data = self.preprocessor.fix_irregular_intervals_automatically(fixed_data, symbol, exchange)
                fix_results["fixes_applied"].append("irregular_intervals")
                fix_results["data_modified"] = True
            
            # Remove duplicates
            initial_len = len(fixed_data)
            fixed_data = fixed_data[~fixed_data.index.duplicated(keep="first")]
            if len(fixed_data) < initial_len:
                fix_results["fixes_applied"].append(f"removed_{initial_len - len(fixed_data)}_duplicates")
                fix_results["data_modified"] = True
            
            # Handle missing values
            missing_before = fixed_data.isna().sum().sum()
            if missing_before > 0:
                # Forward fill for time series continuity
                fixed_data = fixed_data.fillna(method="ffill", limit = 5)
                # Fill remaining with 0 for numeric columns
                numeric_cols = fixed_data.select_dtypes(include=[pd.api.types.is_numeric_dtype]).columns
                fixed_data[numeric_cols] = fixed_data[numeric_cols].fillna(0)
                
                missing_after = fixed_data.isna().sum().sum()
                if missing_after < missing_before:
                    fix_results["fixes_applied"].append(f"filled_{missing_before - missing_after}_missing_values")
                    fix_results["data_modified"] = True
            
            # Fix OHLC inconsistencies
            ohlc_cols = ["open", "high", "low", "close"]
            if all(col in fixed_data.columns for col in ohlc_cols):
                # Ensure high is max of OHLC
                fixed_data["high"] = fixed_data[ohlc_cols].max(axis = 1)
                # Ensure low is min of OHLC
                fixed_data["low"] = fixed_data[ohlc_cols].min(axis = 1)
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
            
            fix_results["final_shape"] = fixed_data.shape
            
        except Exception as e:
            self.logger.error(f"Error during auto-fix: {e}")
            fix_results["fixes_failed"].append(str(e))
        
        return fixed_data, fix_results
    @log_all_calls
        
    def _needs_interval_fixing(self, results: dict[str, Any]) -> bool:
        """Check if interval fixing is needed based on validation results.
        
        Args:
            results: Validation results
            
        Returns:
            True if interval fixing is needed
        """
        # Check for irregular interval warnings
        for warning in results.get("warnings", []):
            if "irregular" in warning.lower() or "interval" in warning.lower():
                return True
        
        # Check detailed analysis
        if "feature_engineering" in results.get("detailed_analysis", {}):
            fe_analysis = results["detailed_analysis"]["feature_engineering"]
            irregular_ratio = fe_analysis.get("irregular_interval_ratio", 0)
            if irregular_ratio > 0.01:
                return True
        
        return False
        
    def validate_and_fix_data_quality_issues(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str
    ) -> Tuple[pd.DataFrame, dict[str, Any]]:
        """
        Validate and attempt to fix data quality issues.
        
        Args:
            data: DataFrame to validate and fix
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            Tuple of (fixed_data, validation_results)
        """
        self.logger.info(f'🔍 Comprehensive data quality validation and fixing for {exchange} {symbol}')
        
        # Run initial validation
        initial_results, _ = self.validate_raw_data(data, symbol, exchange, auto_fix = False)
        
        # Check if fixing is needed
        needs_fixing = (
            not initial_results["validation_passed"] or
            len(initial_results["warnings"]) > 0 or
            initial_results["data_quality_score"] < 0.8
        )
        
        if needs_fixing:
            self.logger.info("🔧 Auto-fixing data quality issues...")
            fixed_data, fix_results = self._auto_fix_issues(
                data, symbol, exchange, 
                ValidationResultBuilder(symbol, exchange, data.shape)
            )
            
            # Re-validate fixed data
            final_results, final_data = self.validate_raw_data(fixed_data, symbol, exchange, auto_fix = False)
            
            # Calculate improvement
            quality_improvement = final_results.get('data_quality_score', 0) - initial_results.get('data_quality_score', 0)
            self.logger.info(f'✅ Quality improvement: {quality_improvement:.3f}')
            
            final_results['preprocessing_summary'] = {
                'irregular_ratio_before': initial_results.get('detailed_analysis', {}).get('feature_engineering', {}).get('irregular_interval_ratio', 0),
                'quality_improvement': quality_improvement,
                'fixes_applied': fix_results.get('fixes_applied', []),
                'original_shape': data.shape,
                'fixed_shape': fixed_data.shape
            }
            
            return final_data, final_results
        else:
            self.logger.info('✅ No data quality issues detected')
            initial_results['preprocessing_summary'] = {
                'irregular_ratio': 0.0,
                'quality_improvement': 0.0,
                'fixes_applied': [],
                'original_shape': data.shape,
                'fixed_shape': data.shape
            }
            return data, initial_results
    
    def get_data_quality_report(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str
    ) -> dict[str, Any]:
        """
        Generate a comprehensive data quality report.
        
        Args:
            data: Market data to analyze
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            Comprehensive data quality report
        """
        validation_results, _ = self.validate_raw_data(data, symbol, exchange)
        
        # Add interval analysis
        from .data_quality_components import calculate_interval_statistics
        interval_stats = calculate_interval_statistics(data)
        validation_results['interval_analysis'] = interval_stats
        
        return validation_results
    @log_all_calls
    
    def _create_error_result(self, message: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Create a standardized error result.
        
        Args:
            message: Error message
            kwargs: Function keyword arguments
            
        Returns:
            Error result dictionary
        """
        return self.error_handler.create_error_result(
            message,
            {"function": "validate_raw_data", "kwargs": str(kwargs)[:100]},
            "ValidationError"
        )

# Convenience functions for backward compatibility
def validate_raw_data_quality(
    data: pd.DataFrame,
    symbol: str,
    exchange: str,
    config: Optional[dict[str, Any]] = None,
    auto_download_missing: bool = False
) -> Tuple[dict[str, Any], pd.DataFrame]:
    """Convenience function to validate raw data quality."""
    checker = RawDataQualityChecker(config)
    return checker.validate_raw_data(data, symbol, exchange, auto_download_missing = auto_download_missing)

def validate_and_fix_data_quality_issues(
    data: pd.DataFrame,
    symbol: str,
    exchange: str,
    config: Optional[dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """Convenience function to validate and fix data quality issues."""
    checker = RawDataQualityChecker(config)
    return checker.validate_and_fix_data_quality_issues(data, symbol, exchange)

def fix_irregular_intervals_automatically(
    data: pd.DataFrame,
    symbol: str,
    exchange: str,
    config: Optional[dict[str, Any]] = None
) -> pd.DataFrame:
    """Convenience function to automatically fix irregular intervals."""
    checker = RawDataQualityChecker(config)
    return checker.preprocessor.fix_irregular_intervals_automatically(data, symbol, exchange)

def enhanced_preprocess_market_data(
    data: pd.DataFrame,
    symbol: str,
    exchange: str,
    expected_interval_seconds: int = 60,
    max_forward_fill_seconds: int = 10,
    download_missing_data: bool = True,
    config: Optional[dict[str, Any]] = None
) -> pd.DataFrame:
    """Convenience function for enhanced preprocessing with intelligent gap handling."""
    checker = RawDataQualityChecker(config)
    return checker.preprocessor.enhanced_preprocess_market_data(
        data = data,
        symbol = symbol,
        exchange = exchange,
        expected_interval_seconds = expected_interval_seconds,
        max_forward_fill_seconds = max_forward_fill_seconds,
        download_missing_data = download_missing_data
    )