"""Data Integrity Checker Component
Validates data integrity and logical consistency for market data.
Extracted from raw_data_quality_checker.py
"""
from typing import Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import timedelta
from src.utils.logger import system_logger


class DataIntegrityChecker:
    """Validates data integrity and logical consistency of market data.
    
    This class provides functionality for:
    - OHLC consistency validation
    - Price and volume integrity checks
    - Market-specific validation
    - Time series integrity validation
    - Cross-validation between related data points
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.logger = system_logger.getChild("DataIntegrityChecker")
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for integrity checks."""
        return {
            "integrity_checks": {
                "check_ohlc_consistency": True,
                "check_negative_values": True,
                "check_extreme_movements": True,
                "check_time_gaps": True,
                "check_for_market_gaps": True
            },
            "critical_thresholds": {
                "max_negative_prices": 0.0,
                "max_zero_volume_ratio": 0.1,
                "max_extreme_move_ratio": 0.001,
                "max_price_change": 0.5,  # 50%
                "max_acceptable_gap_hours": 72
            },
            "warning_thresholds": {
                "high_zero_volume_ratio": 0.05,
                "high_extreme_move_ratio": 0.0005,
                "high_volume_spike_ratio": 0.02,
                "low_volume_threshold_ratio": 0.1
            }
        }
        
    def validate_data_integrity(
        self, 
        data: pd.DataFrame, 
        results: Optional[dict[str, Any]] = None
    ) -> Tuple[bool, dict[str, Any]]:
        """
        Validate data integrity and logical consistency.
        
        Args:
            data: DataFrame with OHLCV data
            results: Optional results dictionary to append to
            
        Returns:
            Tuple of (is_valid, detailed_results)
        """
        if results is None:
            results = {
                "critical_issues": [],
                "warnings": [],
                "detailed_analysis": {}
            }
            
        is_valid = True
        
        # Run various integrity checks
        if self.config["integrity_checks"]["check_ohlc_consistency"]:
            ohlc_valid = self._check_ohlc_consistency(data, results)
            is_valid &= ohlc_valid
            
        if self.config["integrity_checks"]["check_negative_values"]:
            values_valid = self._check_negative_values(data, results)
            is_valid &= values_valid
            
        if self.config["integrity_checks"]["check_extreme_movements"]:
            movements_valid = self._check_extreme_movements(data, results)
            is_valid &= movements_valid
            
        if self.config["integrity_checks"]["check_time_gaps"]:
            time_valid = self._check_time_integrity(data, results)
            is_valid &= time_valid
            
        return is_valid, results
        
    def _check_ohlc_consistency(self, data: pd.DataFrame, results: dict[str, Any]) -> bool:
        """Check OHLC data consistency."""
        self.logger.info("Checking OHLC consistency...")
        
        if not all(col in data.columns for col in ["open", "high", "low", "close"]):
            results["critical_issues"].append("Missing required OHLC columns")
            return False
            
        # Check logical OHLC relationships
        ohlc_inconsistent = (
            (data["high"] < data["low"]) |
            (data["open"] > data["high"]) |
            (data["close"] > data["high"]) |
            (data["open"] < data["low"]) |
            (data["close"] < data["low"])
        )
        
        ohlc_inconsistent_count = ohlc_inconsistent.sum()
        ohlc_inconsistent_ratio = ohlc_inconsistent_count / len(data) if len(data) > 0 else 0
        
        if ohlc_inconsistent_ratio > 0:
            results["critical_issues"].append(
                f"OHLC inconsistency found: {ohlc_inconsistent_ratio:.3%} of records ({ohlc_inconsistent_count} rows)"
            )
            
            # Store detailed information
            if "integrity" not in results["detailed_analysis"]:
                results["detailed_analysis"]["integrity"] = {}
            results["detailed_analysis"]["integrity"]["ohlc_inconsistent_ratio"] = float(ohlc_inconsistent_ratio)
            results["detailed_analysis"]["integrity"]["ohlc_inconsistent_indices"] = data.index[ohlc_inconsistent].tolist()[:10]  # First 10
            
            return False
            
        self.logger.info("✅ OHLC consistency check passed")
        return True
        
    def _check_negative_values(self, data: pd.DataFrame, results: dict[str, Any]) -> bool:
        """Check for negative prices and volumes."""
        self.logger.info("Checking for negative values...")
        
        is_valid = True
        
        # Check for negative prices
        price_columns = [col for col in ["open", "high", "low", "close"] if col in data.columns]
        if price_columns:
            negative_prices = (data[price_columns] < 0).any(axis=1)
            negative_price_count = negative_prices.sum()
            negative_price_ratio = negative_price_count / len(data) if len(data) > 0 else 0
            
            max_negative = self.config["critical_thresholds"]["max_negative_prices"]
            
            if negative_price_ratio > max_negative:
                results["critical_issues"].append(
                    f"Negative prices found: {negative_price_ratio:.3%} of records ({negative_price_count} rows)"
                )
                is_valid = False
                
        # Check for zero or negative volume
        if "volume" in data.columns:
            zero_volume = data["volume"] <= 0
            zero_volume_count = zero_volume.sum()
            zero_volume_ratio = zero_volume_count / len(data) if len(data) > 0 else 0
            
            max_zero_volume = self.config["critical_thresholds"]["max_zero_volume_ratio"]
            high_zero_volume = self.config["warning_thresholds"]["high_zero_volume_ratio"]
            
            if zero_volume_ratio > max_zero_volume:
                results["critical_issues"].append(
                    f"High zero/negative volume: {zero_volume_ratio:.3%} (threshold: {max_zero_volume:.1%})"
                )
                is_valid = False
            elif zero_volume_ratio > high_zero_volume:
                results["warnings"].append(
                    f"Elevated zero/negative volume: {zero_volume_ratio:.3%} of records"
                )
                
            # Store detailed information
            if "integrity" not in results["detailed_analysis"]:
                results["detailed_analysis"]["integrity"] = {}
            results["detailed_analysis"]["integrity"]["zero_volume_ratio"] = float(zero_volume_ratio)
            
        return is_valid
        
    def _check_extreme_movements(self, data: pd.DataFrame, results: dict[str, Any]) -> bool:
        """Check for extreme price movements."""
        self.logger.info("Checking for extreme movements...")
        
        if "close" not in data.columns:
            return True
            
        # Calculate price changes
        price_changes = data["close"].pct_change().abs()
        max_change_threshold = self.config["critical_thresholds"]["max_price_change"]
        
        extreme_moves = price_changes > max_change_threshold
        extreme_move_count = extreme_moves.sum()
        extreme_move_ratio = extreme_move_count / len(price_changes.dropna()) if len(price_changes.dropna()) > 0 else 0
        
        max_extreme_ratio = self.config["critical_thresholds"]["max_extreme_move_ratio"]
        warning_extreme_ratio = self.config["warning_thresholds"]["high_extreme_move_ratio"]
        
        if extreme_move_ratio > max_extreme_ratio:
            results["critical_issues"].append(
                f"Too many extreme price movements: {extreme_move_ratio:.3%} of records "
                f"(>{max_change_threshold:.0%} change)"
            )
            return False
        elif extreme_move_ratio > warning_extreme_ratio:
            results["warnings"].append(
                f"Extreme price movements detected: {extreme_move_ratio:.3%} of records"
            )
            
        # Store detailed information
        if "integrity" not in results["detailed_analysis"]:
            results["detailed_analysis"]["integrity"] = {}
        results["detailed_analysis"]["integrity"]["extreme_move_ratio"] = float(extreme_move_ratio)
        
        # Find largest movements
        if extreme_move_count > 0:
            largest_moves = price_changes.nlargest(min(5, extreme_move_count))
            results["detailed_analysis"]["integrity"]["largest_price_changes"] = {
                str(idx): float(val) for idx, val in largest_moves.items()
            }
            
        return True
        
    def _check_time_integrity(self, data: pd.DataFrame, results: dict[str, Any]) -> bool:
        """Check time series integrity."""
        self.logger.info("Checking time series integrity...")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            results["warnings"].append("Data does not have datetime index")
            return True
            
        # Check if timestamps are sorted
        if not data.index.is_monotonic_increasing:
            results["critical_issues"].append("Timestamps are not sorted in ascending order")
            return False
            
        # Check for duplicate timestamps
        duplicate_timestamps = data.index.duplicated()
        if duplicate_timestamps.any():
            duplicate_count = duplicate_timestamps.sum()
            results["critical_issues"].append(
                f"Duplicate timestamps found: {duplicate_count} duplicates"
            )
            return False
            
        # Check for time gaps
        time_diffs = data.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            max_gap_hours = self.config["critical_thresholds"]["max_acceptable_gap_hours"]
            large_gaps = time_diffs > timedelta(hours=max_gap_hours)
            
            if large_gaps.any():
                gap_count = large_gaps.sum()
                max_gap = time_diffs.max()
                results["warnings"].append(
                    f"Large time gaps found: {gap_count} gaps > {max_gap_hours} hours, "
                    f"largest gap: {max_gap.total_seconds()/3600:.1f} hours"
                )
                
        return True
        
    def validate_market_specific_issues(
        self, 
        data: pd.DataFrame, 
        results: Optional[dict[str, Any]] = None
    ) -> Tuple[bool, dict[str, Any]]:
        """
        Validate market-specific issues and anomalies.
        
        Args:
            data: DataFrame with market data
            results: Optional results dictionary
            
        Returns:
            Tuple of (is_valid, detailed_results)
        """
        if results is None:
            results = {
                "critical_issues": [],
                "warnings": [],
                "detailed_analysis": {}
            }
            
        self.logger.info("Validating market-specific issues...")
        
        # Check for market gaps (weekends, holidays)
        if self.config["integrity_checks"]["check_for_market_gaps"] and isinstance(data.index, pd.DatetimeIndex):
            time_diffs = data.index.to_series().diff().dropna()
            weekend_gaps = time_diffs[time_diffs > timedelta(hours=48)]
            
            if len(weekend_gaps) > 0:
                results["warnings"].append(
                    f"Detected {len(weekend_gaps)} potential market gaps (weekends/holidays)"
                )
                
        # Check for suspicious volume patterns
        if "volume" in data.columns:
            volume_mean = data["volume"].mean()
            volume_std = data["volume"].std()
            
            if volume_std > 0:
                # High volume spikes
                high_volume = data["volume"] > (volume_mean + 3 * volume_std)
                high_volume_ratio = high_volume.sum() / len(data) if len(data) > 0 else 0
                
                if high_volume_ratio > self.config["warning_thresholds"]["high_volume_spike_ratio"]:
                    results["warnings"].append(
                        f"Unusual high volume periods: {high_volume_ratio:.3%} of records"
                    )
                    
                # Low volume periods
                low_volume = data["volume"] < (volume_mean * 0.1)
                low_volume_ratio = low_volume.sum() / len(data) if len(data) > 0 else 0
                
                if low_volume_ratio > self.config["warning_thresholds"]["low_volume_threshold_ratio"]:
                    results["warnings"].append(
                        f"Unusual low volume periods: {low_volume_ratio:.3%} of records"
                    )
                    
            # Store volume statistics
            if "market_specific" not in results["detailed_analysis"]:
                results["detailed_analysis"]["market_specific"] = {}
                
            results["detailed_analysis"]["market_specific"]["volume_statistics"] = {
                "mean": float(volume_mean),
                "std": float(volume_std),
                "min": float(data["volume"].min()),
                "max": float(data["volume"].max()),
                "coefficient_of_variation": float(volume_std / volume_mean) if volume_mean > 0 else None
            }
            
        return True, results
        
    def check_cross_validation(
        self,
        data: pd.DataFrame,
        validation_rules: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Perform cross-validation between related data points.
        
        Args:
            data: DataFrame to validate
            validation_rules: Custom validation rules
            
        Returns:
            Validation results
        """
        results = {
            "passed": True,
            "failed_rules": [],
            "warnings": []
        }
        
        default_rules = {
            "high_low_spread": {
                "max_ratio": 0.5,  # High should not be > 50% above low
                "type": "warning"
            },
            "volume_price_correlation": {
                "check": True,
                "type": "info"
            }
        }
        
        rules = validation_rules or default_rules
        
        # High-Low spread check
        if "high_low_spread" in rules and all(col in data.columns for col in ["high", "low"]):
            spread_ratio = (data["high"] - data["low"]) / data["low"]
            excessive_spread = spread_ratio > rules["high_low_spread"]["max_ratio"]
            
            if excessive_spread.any():
                message = f"Excessive high-low spread found in {excessive_spread.sum()} records"
                if rules["high_low_spread"]["type"] == "critical":
                    results["failed_rules"].append(message)
                    results["passed"] = False
                else:
                    results["warnings"].append(message)
                    
        return results