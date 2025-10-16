"""Configuration Manager Component

Manages configuration for quality checks and data processing.
Extracted from raw_data_quality_checker.py
"""

from typing import Any, Optional, List
import numpy as np

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

class QualityCheckConfig:
    """Manages configuration for quality checks and data processing.
    
    This class provides functionality for:
    - Managing validation thresholds
    - Configuring preprocessing options
    - Setting up integrity checks
    - Managing feature engineering requirements
    """

    @log_important_calls
    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or self._get_default_config()

    @log_all_calls
    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for quality checks optimized for feature engineering."""
        return {
            # Critical thresholds that will fail validation
            "critical_thresholds": {
                "min_records": 1000,
                "max_missing_ohlc": 0.005,
                "max_price_anomalies": 0.0005,
                "max_volume_anomalies": 0.02,
                "min_data_span_days": 7,
                "min_continuous_data_hours": 48,
                "max_ohlc_inconsistency": 0.0,
                "max_negative_prices": 0.0,
                "max_zero_volume_ratio": 0.05
            },
            
            # Warning thresholds that won't fail but indicate issues
            "warning_thresholds": {
                "max_gap_hours": 1,
                "max_duplicate_timestamps": 0.0005,
                "max_extreme_price_moves": 0.001,
                "max_volume_spikes": 0.01,
                "max_timestamp_discontinuity": 0.02
            },
            
            # Feature engineering specific checks
            "feature_engineering_checks": {
                "check_rolling_window_compatibility": True,
                "check_wavelet_data_requirements": True,
                "check_microstructure_feature_requirements": True,
                "check_multi_timeframe_alignment": True,
                "check_volume_price_relationship": True,
                "check_timestamp_regularity": True,
                "check_data_stationarity_preconditions": True
            },
            
            # Data integrity checks
            "integrity_checks": {
                "check_ohlc_consistency": True,
                "check_timestamp_continuity": True,
                "check_price_logical_consistency": True,
                "check_volume_sanity": True,
                "check_for_market_gaps": True,
                "check_data_type_consistency": True,
                "check_index_alignment": True
            },
            
            # Preprocessing configuration
            "preprocessing": {
                "max_forward_fill_seconds": 10,
                "auto_fix_irregular_intervals": True,
                "download_missing_data": True,
                "preserve_original_data": True
            },
            
            # Quality metrics weights
            "score_weights": {
                "completeness": 0.25,
                "consistency": 0.25,
                "timeliness": 0.20,
                "validity": 0.20,
                "accuracy": 0.10
            },
            
            # Multi-timeframe configuration
            "multi_timeframe": {
                "large_change_threshold": 0.1,
                "large_change_ratio_threshold": 0.01
            }
        }
        
    def get_threshold(self, category: str, key: str) -> Any:
        """Get specific threshold value.
        
        Args:
            category: Configuration category (e.g., 'critical_thresholds')
            key: Threshold key (e.g., 'min_records')
            
        Returns:
            Threshold value or None if not found
        """
        return self.config.get(category, {}).get(key)
        
    def get_critical_threshold(self, key: str) -> Any:
        """Get critical threshold value.
        
        Args:
            key: Threshold key
            
        Returns:
            Critical threshold value or None if not found
        """
        return self.get_threshold("critical_thresholds", key)
        
    def get_warning_threshold(self, key: str) -> Any:
        """Get warning threshold value.
        
        Args:
            key: Threshold key
            
        Returns:
            Warning threshold value or None if not found
        """
        return self.get_threshold("warning_thresholds", key)
        
    def is_check_enabled(self, category: str, key: str) -> bool:
        """Check if a specific check is enabled.
        
        Args:
            category: Configuration category
            key: Check key
            
        Returns:
            True if check is enabled, False otherwise
        """
        return self.config.get(category, {}).get(key, False)
        
    def is_integrity_check_enabled(self, key: str) -> bool:
        """Check if an integrity check is enabled.
        
        Args:
            key: Integrity check key
            
        Returns:
            True if check is enabled, False otherwise
        """
        return self.is_check_enabled("integrity_checks", key)
        
    def is_feature_engineering_check_enabled(self, key: str) -> bool:
        """Check if a feature engineering check is enabled.
        
        Args:
            key: Feature engineering check key
            
        Returns:
            True if check is enabled, False otherwise
        """
        return self.is_check_enabled("feature_engineering_checks", key)
        
    def get_preprocessing_config(self, key: str) -> Any:
        """Get preprocessing configuration value.
        
        Args:
            key: Preprocessing configuration key
            
        Returns:
            Preprocessing configuration value or None if not found
        """
        return self.get_threshold("preprocessing", key)
        
    def get_score_weight(self, key: str) -> float:
        """Get quality score weight.
        
        Args:
            key: Score weight key
            
        Returns:
            Score weight or 0.0 if not found
        """
        return self.config.get("score_weights", {}).get(key, 0.0)
        
    def update_config(self, updates: dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        for category, values in updates.items():
            if category in self.config:
                if isinstance(values, dict):
                    self.config[category].update(values)
                else:
                    self.config[category] = values
            else:
                self.config[category] = values
                
    def get_config(self) -> dict[str, Any]:
        """Get the complete configuration.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()
        
    def validate_config(self) -> list[str]:
        """Validate the configuration for consistency.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check critical thresholds
        critical_thresholds = self.config.get("critical_thresholds", {})
        if not critical_thresholds:
            errors.append("Missing critical_thresholds configuration")
            
        # Check warning thresholds
        warning_thresholds = self.config.get("warning_thresholds", {})
        if not warning_thresholds:
            errors.append("Missing warning_thresholds configuration")
            
        # Validate threshold values
        for category in ["critical_thresholds", "warning_thresholds"]:
            thresholds = self.config.get(category, {})
            for key, value in thresholds.items():
                if not isinstance(value, (int, float)):
                    errors.append(f"Invalid threshold value for {category}.{key}: {value}")
                elif value < 0:
                    errors.append(f"Negative threshold value for {category}.{key}: {value}")
                    
        # Validate score weights
        score_weights = self.config.get("score_weights", {})
        total_weight = sum(score_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Score weights don't sum to 1.0: {total_weight}")
            
        return errors
        
    def get_validation_summary(self) -> dict[str, Any]:
        """Get a summary of the current configuration.
        
        Returns:
            Configuration summary
        """
        return {
            "critical_thresholds_count": len(self.config.get("critical_thresholds", {})),
            "warning_thresholds_count": len(self.config.get("warning_thresholds", {})),
            "integrity_checks_enabled": sum(self.config.get("integrity_checks", {}).values()),
            "feature_engineering_checks_enabled": sum(self.config.get("feature_engineering_checks", {}).values()),
            "preprocessing_enabled": self.config.get("preprocessing", {}).get("auto_fix_irregular_intervals", False),
            "download_missing_data_enabled": self.config.get("preprocessing", {}).get("download_missing_data", False),
            "config_validation_errors": len(self.validate_config())
        }