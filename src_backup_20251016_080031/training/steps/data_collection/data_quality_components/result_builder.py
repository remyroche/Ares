"""Result Builder Component

Builder pattern for creating consistent validation results.
Extracted from raw_data_quality_checker.py
"""

from datetime import datetime
from typing import Any, List, Optional
import pandas as pd
import logging
import numpy as np
import time

from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

class ValidationResultBuilder:
    """Builder for validation results using the builder pattern.
    
    This class provides functionality for:
    - Building consistent validation results
    - Managing validation state
    - Adding issues, warnings, and recommendations
    - Calculating quality scores
    - Generating final results
    """
    @log_important_calls
    def __init__(self, symbol: str, exchange: str, data_shape: tuple[int, int]):
        self.logger = system_logger.getChild("ValidationResultBuilder")
        self.result = {
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now().isoformat(),
            "data_shape": data_shape,
            "validation_passed": True,
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "data_quality_score": 0.0,
            "detailed_analysis": {},
            "preprocessing_applied": {},
            "data_downloaded": False,
            "download_summary": {}
        }
        
    def add_critical_issue(self, issue: str) -> 'ValidationResultBuilder':
        """Add a critical issue that will cause validation to fail.
        
        Args:
            issue: Critical issue description
            
        Returns:
            Self for method chaining
        """
        self.result["critical_issues"].append(issue)
        self.result["validation_passed"] = False
        self.logger.error(f"❌ Critical issue: {issue}")
        return self
        
    def add_warning(self, warning: str) -> 'ValidationResultBuilder':
        """Add a warning that won't cause validation to fail.
        
        Args:
            warning: Warning description
            
        Returns:
            Self for method chaining
        """
        self.result["warnings"].append(warning)
        self.logger.warning(f"⚠️ Warning: {warning}")
        return self
        
    def add_recommendation(self, recommendation: str) -> 'ValidationResultBuilder':
        """Add a recommendation for improving data quality.
        
        Args:
            recommendation: Recommendation description
            
        Returns:
            Self for method chaining
        """
        self.result["recommendations"].append(recommendation)
        self.logger.info(f"💡 Recommendation: {recommendation}")
        return self
        
    def set_quality_score(self, score: float) -> 'ValidationResultBuilder':
        """Set the overall data quality score.
        
        Args:
            score: Quality score between 0.0 and 1.0
            
        Returns:
            Self for method chaining
        """
        self.result["data_quality_score"] = max(0.0, min(1.0, score))
        return self
        
    def add_detailed_analysis(self, category: str, analysis: dict[str, Any]) -> 'ValidationResultBuilder':
        """Add detailed analysis for a specific category.
        
        Args:
            category: Analysis category (e.g., 'structure', 'completeness')
            analysis: Analysis results
            
        Returns:
            Self for method chaining
        """
        self.result["detailed_analysis"][category] = analysis
        return self
        
    def set_preprocessing_applied(self, preprocessing_info: dict[str, Any]) -> 'ValidationResultBuilder':
        """Set information about preprocessing that was applied.
        
        Args:
            preprocessing_info: Preprocessing information
            
        Returns:
            Self for method chaining
        """
        self.result["preprocessing_applied"] = preprocessing_info
        return self
        
    def set_data_downloaded(self, downloaded: bool, download_summary: Optional[dict[str, Any]] = None) -> 'ValidationResultBuilder':
        """Set information about data downloading.
        
        Args:
            downloaded: Whether data was downloaded
            download_summary: Download summary information
            
        Returns:
            Self for method chaining
        """
        self.result["data_downloaded"] = downloaded
        if download_summary:
            self.result["download_summary"] = download_summary
        return self
        
    def add_interval_analysis(self, data: pd.DataFrame) -> 'ValidationResultBuilder':
        """Add interval analysis to the results.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Self for method chaining
        """
        try:
            from .data_utils import calculate_interval_statistics
            
            interval_stats = calculate_interval_statistics(data)
            self.add_detailed_analysis("interval_analysis", interval_stats)
            
            # Add warnings based on interval analysis
            if interval_stats["irregular_ratio"] > 0.01:
                self.add_warning(
                    f"High irregular interval ratio: {interval_stats['irregular_ratio']:.3f} "
                    f"({interval_stats['irregular_intervals']} irregular intervals)"
                )
                
            if interval_stats["coefficient_of_variation"] > 0.3:
                self.add_warning(
                    f"High interval variability (CV: {interval_stats['coefficient_of_variation']:.3f}) "
                    "may affect multi-timeframe features"
                )
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to add interval analysis: {e}")
            
        return self
        
    def add_gap_analysis(self, data: pd.DataFrame, max_gap_hours: float = 1.0) -> 'ValidationResultBuilder':
        """Add gap analysis to the results.
        
        Args:
            data: DataFrame to analyze
            max_gap_hours: Maximum acceptable gap in hours
            
        Returns:
            Self for method chaining
        """
        try:
            from .data_utils import detect_data_gaps
            
            gap_analysis = detect_data_gaps(data, max_gap_hours)
            self.add_detailed_analysis("gap_analysis", gap_analysis)
            
            # Add warnings based on gap analysis
            if gap_analysis["large_gaps"] > 0:
                self.add_warning(
                    f"Found {gap_analysis['large_gaps']} large gaps (> {max_gap_hours}h) "
                    f"out of {gap_analysis['total_gaps']} total gaps"
                )
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to add gap analysis: {e}")
            
        return self
        
    def add_volume_analysis(self, data: pd.DataFrame) -> 'ValidationResultBuilder':
        """Add volume analysis to the results.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Self for method chaining
        """
        try:
            from .data_utils import calculate_volume_statistics
            
            volume_stats = calculate_volume_statistics(data)
            self.add_detailed_analysis("volume_statistics", volume_stats)
            
            # Add warnings based on volume analysis
            if volume_stats["zero_volume_ratio"] > 0.05:
                self.add_warning(
                    f"High zero volume ratio: {volume_stats['zero_volume_ratio']:.3f} "
                    f"({volume_stats['zero_volume_ratio'] * 100:.1f}% of records)"
                )
                
            if volume_stats["volume_spikes"] > 0:
                self.add_warning(
                    f"Detected {volume_stats['volume_spikes']} volume spikes "
                    f"and {volume_stats['volume_drops']} volume drops"
                )
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to add volume analysis: {e}")
            
        return self
        
    def calculate_quality_score(self) -> 'ValidationResultBuilder':
        """Calculate and set the quality score based on current issues and warnings.
        
        Returns:
            Self for method chaining
        """
        # Base score starts at 1.0
        score = 1.0
        
        # Deduct points for critical issues (more severe)
        critical_penalty = 0.3
        score -= len(self.result["critical_issues"]) * critical_penalty
        
        # Deduct points for warnings (less severe)
        warning_penalty = 0.05
        score -= len(self.result["warnings"]) * warning_penalty
        
        # Additional penalties from detailed analysis
        if "detailed_analysis" in self.result:
            # Missing data penalty
            if "completeness" in self.result["detailed_analysis"]:
                missing_ratio = self.result["detailed_analysis"]["completeness"].get("missing_ohlc_ratio", 0)
                score -= missing_ratio * 0.1
                
            # Interval irregularity penalty
            if "interval_analysis" in self.result["detailed_analysis"]:
                irregular_ratio = self.result["detailed_analysis"]["interval_analysis"].get("irregular_ratio", 0)
                score -= irregular_ratio * 0.2
                
            # Volume issues penalty
            if "volume_statistics" in self.result["detailed_analysis"]:
                zero_volume_ratio = self.result["detailed_analysis"]["volume_statistics"].get("zero_volume_ratio", 0)
                score -= zero_volume_ratio * 0.1
                
        # Ensure score doesn't go below 0
        score = max(0.0, score)
        
        self.set_quality_score(score)
        return self
        
    def generate_recommendations(self) -> 'ValidationResultBuilder':
        """Generate recommendations based on current issues and analysis.
        
        Returns:
            Self for method chaining
        """
        # Don't add recommendations if validation failed
        if not self.result["validation_passed"]:
            return self
            
        # Generate recommendations based on quality score
        quality_score = self.result["data_quality_score"]
        
        if quality_score < 0.8:
            self.add_recommendation("Consider re-downloading data due to quality issues")
            
        if self.result["warnings"]:
            self.add_recommendation("Review warnings before proceeding with feature engineering")
            
        # Generate recommendations based on detailed analysis
        if "detailed_analysis" in self.result:
            # Missing data recommendations
            if "completeness" in self.result["detailed_analysis"]:
                missing_ratio = self.result["detailed_analysis"]["completeness"].get("missing_ohlc_ratio", 0)
                if missing_ratio > 0.001:
                    self.add_recommendation("Consider data interpolation for missing values")
                    
            # Interval recommendations
            if "interval_analysis" in self.result["detailed_analysis"]:
                irregular_ratio = self.result["detailed_analysis"]["interval_analysis"].get("irregular_ratio", 0)
                if irregular_ratio > 0.05:
                    self.add_recommendation(
                        f"High irregular interval ratio ({irregular_ratio:.3f}) - "
                        "consider data resampling for multi-timeframe features"
                    )
                    
            # Volume recommendations
            if "volume_statistics" in self.result["detailed_analysis"]:
                zero_volume_ratio = self.result["detailed_analysis"]["volume_statistics"].get("zero_volume_ratio", 0)
                if zero_volume_ratio > 0.05:
                    self.add_recommendation("High zero volume may indicate data quality issues")
                    
            # Feature engineering recommendations
            if "feature_engineering" in self.result["detailed_analysis"]:
                fe_issues = self.result["detailed_analysis"]["feature_engineering"]
                
                if not fe_issues.get("rolling_window_compatible", True):
                    self.add_recommendation("Insufficient data for rolling windows - consider longer lookback period")
                    
                wavelet_gaps = fe_issues.get("wavelet_gaps_count", 0)
                if wavelet_gaps > 0:
                    self.add_recommendation("Large gaps detected - consider data interpolation for wavelet features")
                    
                volume_price_corr = fe_issues.get("volume_price_correlation")
                if volume_price_corr and abs(volume_price_corr) > 0.95:
                    self.add_recommendation("Unusually high volume-price correlation - verify data source integrity")
                    
                irregular_ratio = fe_issues.get("irregular_interval_ratio", 0)
                if irregular_ratio > 0.01:
                    self.add_recommendation("Irregular time intervals detected - may affect multi-timeframe features")
                    
                spike_ratio = fe_issues.get("volume_spike_ratio", 0)
                if spike_ratio > 0.05:
                    self.add_recommendation("High volume spikes detected - consider outlier detection for microstructure features")
                    
                trend_strength = fe_issues.get("price_trend_strength", 0)
                if trend_strength > 0.01:
                    self.add_recommendation("Strong price trend detected - consider detrending for stationarity-based features")
                    
        return self
        
    def log_summary(self) -> 'ValidationResultBuilder':
        """Log a summary of the validation results.
        
        Returns:
            Self for method chaining
        """
        status = "✅ PASSED" if self.result["validation_passed"] else "❌ FAILED"
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 Validation Summary: {status}")
        self.logger.info(f"Symbol: {self.result['symbol']}, Exchange: {self.result['exchange']}")
        self.logger.info(f"Data Shape: {self.result['data_shape']}")
        self.logger.info(f"Quality Score: {self.result['data_quality_score']:.2f}")
        
        if self.result["critical_issues"]:
            self.logger.error(f"Critical Issues ({len(self.result['critical_issues'])}):")
            for issue in self.result["critical_issues"]:
                self.logger.error(f"  - {issue}")
                
        if self.result["warnings"]:
            self.logger.warning(f"Warnings ({len(self.result['warnings'])}):")
            for warning in self.result["warnings"][:5]:  # Show first 5
                self.logger.warning(f"  - {warning}")
            if len(self.result["warnings"]) > 5:
                self.logger.warning(f"  ... and {len(self.result['warnings']) - 5} more")
                
        if self.result["recommendations"]:
            self.logger.info(f"Recommendations:")
            for rec in self.result["recommendations"]:
                self.logger.info(f"  - {rec}")
                
        self.logger.info(f"{'='*60}\n")
        return self
        
    def build(self) -> dict[str, Any]:
        """Build the final validation result.
        
        Returns:
            Complete validation result dictionary
        """
        # Ensure quality score is calculated
        if self.result["data_quality_score"] == 0.0:
            self.calculate_quality_score()
            
        # Generate recommendations if not already done
        if not self.result["recommendations"]:
            self.generate_recommendations()
            
        # Log summary
        self.log_summary()
        
        return self.result.copy()
        
    def get_current_state(self) -> dict[str, Any]:
        """Get the current state of the result builder.
        
        Returns:
            Current result state
        """
        return self.result.copy()
        
    def reset(self) -> 'ValidationResultBuilder':
        """Reset the result builder to initial state.
        
        Returns:
            Self for method chaining
        """
        symbol = self.result["symbol"]
        exchange = self.result["exchange"]
        data_shape = self.result["data_shape"]
        
        self.__init__(symbol, exchange, data_shape)
        return self