# src/training/steps/data_preparation_components/data_integrity_checker.py

"""Data Integrity Checker - Validates data integrity and relationships."""

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.pipeline_standards import pipeline_standards


class DataIntegrityChecker:
    """Checks data integrity including relationships, constraints, and consistency."""
    
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize DataIntegrityChecker with configuration."""
        self.config = config or self._get_default_config()
        self.logger = system_logger.getChild("DataIntegrityChecker")
        self.standards = pipeline_standards
        
    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for integrity checking."""
        return {
            "check_referential_integrity": True,
            "check_business_rules": True,
            "check_temporal_integrity": True,
            "check_cross_field_integrity": True,
            "tolerance": 1e-8,
            "max_time_gap_minutes": 60,
            "expected_frequency": "1T",  # 1 minute
            "business_rules": {
                "price_relationships": True,
                "volume_constraints": True,
                "time_constraints": True
            }
        }
    
    async def check_integrity(
        self,
        data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Perform comprehensive integrity checks.
        
        Args:
            data: Primary DataFrame to check
            reference_data: Optional reference data for comparison
            metadata: Optional metadata about the data
            
        Returns:
            dict: Integrity check results
        """
        results = {
            "is_valid": True,
            "integrity_score": 100.0,
            "checks_performed": [],
            "violations": [],
            "warnings": [],
            "summary": {}
        }
        
        try:
            # Check data structure integrity
            structure_results = await self._check_structure_integrity(data)
            results["checks_performed"].append("structure_integrity")
            results["summary"]["structure"] = structure_results
            
            # Check temporal integrity
            if self.config["check_temporal_integrity"]:
                temporal_results = await self._check_temporal_integrity(data)
                results["checks_performed"].append("temporal_integrity")
                results["summary"]["temporal"] = temporal_results
            
            # Check cross-field integrity
            if self.config["check_cross_field_integrity"]:
                cross_field_results = await self._check_cross_field_integrity(data)
                results["checks_performed"].append("cross_field_integrity")
                results["summary"]["cross_field"] = cross_field_results
            
            # Check business rules
            if self.config["check_business_rules"]:
                business_rules_results = await self._check_business_rules(data)
                results["checks_performed"].append("business_rules")
                results["summary"]["business_rules"] = business_rules_results
            
            # Check referential integrity if reference data provided
            if reference_data is not None and self.config["check_referential_integrity"]:
                referential_results = await self._check_referential_integrity(data, reference_data)
                results["checks_performed"].append("referential_integrity")
                results["summary"]["referential"] = referential_results
            
            # Aggregate results
            for check_name, check_results in results["summary"].items():
                if isinstance(check_results, dict):
                    if not check_results.get("is_valid", True):
                        results["is_valid"] = False
                    results["violations"].extend(check_results.get("violations", []))
                    results["warnings"].extend(check_results.get("warnings", []))
            
            # Calculate integrity score
            results["integrity_score"] = self._calculate_integrity_score(results)
            
            # Add data fingerprint
            results["data_fingerprint"] = self._calculate_data_fingerprint(data)
            
        except Exception as e:
            self.logger.error(f"Integrity check failed: {e}")
            results["is_valid"] = False
            results["error"] = str(e)
            
        return results
    
    async def _check_structure_integrity(self, data: pd.DataFrame) -> dict[str, Any]:
        """Check structural integrity of the data."""
        results = {
            "is_valid": True,
            "violations": [],
            "warnings": [],
            "structure_info": {}
        }
        
        try:
            # Check for required columns
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                results["is_valid"] = False
                results["violations"].append(f"Missing required columns: {missing_columns}")
            
            # Check data types
            expected_types = {
                "open": "numeric",
                "high": "numeric",
                "low": "numeric",
                "close": "numeric",
                "volume": "numeric"
            }
            
            for col, expected_type in expected_types.items():
                if col in data.columns:
                    if expected_type == "numeric" and not pd.api.types.is_numeric_dtype(data[col]):
                        results["violations"].append(
                            f"Column '{col}' is not numeric: {data[col].dtype}"
                        )
            
            # Check index integrity
            if isinstance(data.index, pd.DatetimeIndex):
                results["structure_info"]["has_datetime_index"] = True
                results["structure_info"]["index_is_unique"] = data.index.is_unique
                results["structure_info"]["index_is_sorted"] = data.index.is_monotonic_increasing
                
                if not data.index.is_unique:
                    results["is_valid"] = False
                    results["violations"].append("Index has duplicate values")
                
                if not data.index.is_monotonic_increasing:
                    results["warnings"].append("Index is not sorted in ascending order")
            else:
                results["structure_info"]["has_datetime_index"] = False
                results["warnings"].append("Data does not have a datetime index")
            
            # Record structure info
            results["structure_info"]["shape"] = data.shape
            results["structure_info"]["columns"] = list(data.columns)
            results["structure_info"]["dtypes"] = {col: str(dtype) for col, dtype in data.dtypes.items()}
            
        except Exception as e:
            results["is_valid"] = False
            results["violations"].append(f"Structure integrity check failed: {e}")
            
        return results
    
    async def _check_temporal_integrity(self, data: pd.DataFrame) -> dict[str, Any]:
        """Check temporal integrity of time series data."""
        results = {
            "is_valid": True,
            "violations": [],
            "warnings": [],
            "temporal_info": {}
        }
        
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                results["warnings"].append("Cannot check temporal integrity without datetime index")
                return results
            
            # Check for time gaps
            time_diffs = data.index.to_series().diff()[1:]  # Skip first NaN
            
            if len(time_diffs) > 0:
                # Detect expected frequency
                mode_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
                results["temporal_info"]["detected_frequency"] = str(mode_diff)
                
                # Check for gaps
                max_gap = timedelta(minutes=self.config["max_time_gap_minutes"])
                large_gaps = time_diffs[time_diffs > max_gap]
                
                if len(large_gaps) > 0:
                    results["warnings"].append(
                        f"Found {len(large_gaps)} time gaps larger than {self.config['max_time_gap_minutes']} minutes"
                    )
                    results["temporal_info"]["large_gaps"] = [
                        {
                            "timestamp": str(idx),
                            "gap_minutes": gap.total_seconds() / 60
                        }
                        for idx, gap in large_gaps.head(10).items()  # Limit to first 10
                    ]
                
                # Check for backwards time
                backwards_time = time_diffs[time_diffs < timedelta(0)]
                if len(backwards_time) > 0:
                    results["is_valid"] = False
                    results["violations"].append(
                        f"Found {len(backwards_time)} instances of backwards time progression"
                    )
                
                # Check frequency consistency
                freq_std = time_diffs.std()
                if freq_std > mode_diff:
                    results["warnings"].append(
                        f"Irregular time intervals detected (std: {freq_std})"
                    )
            
            # Check time range
            results["temporal_info"]["start_time"] = str(data.index.min())
            results["temporal_info"]["end_time"] = str(data.index.max())
            results["temporal_info"]["duration"] = str(data.index.max() - data.index.min())
            results["temporal_info"]["total_periods"] = len(data)
            
            # Check for future timestamps
            current_time = datetime.now(data.index[0].tzinfo) if data.index[0].tzinfo else datetime.now()
            future_timestamps = data.index[data.index > current_time]
            
            if len(future_timestamps) > 0:
                results["warnings"].append(
                    f"Found {len(future_timestamps)} timestamps in the future"
                )
                
        except Exception as e:
            results["is_valid"] = False
            results["violations"].append(f"Temporal integrity check failed: {e}")
            
        return results
    
    async def _check_cross_field_integrity(self, data: pd.DataFrame) -> dict[str, Any]:
        """Check integrity across multiple fields."""
        results = {
            "is_valid": True,
            "violations": [],
            "warnings": [],
            "relationships": {}
        }
        
        try:
            # Check OHLC relationships
            ohlc_cols = ["open", "high", "low", "close"]
            if all(col in data.columns for col in ohlc_cols):
                # High >= Low
                high_low_violations = data[data["high"] < data["low"]]
                if len(high_low_violations) > 0:
                    results["is_valid"] = False
                    results["violations"].append(
                        f"Found {len(high_low_violations)} rows where high < low"
                    )
                    results["relationships"]["high_low_violations"] = len(high_low_violations)
                
                # Low <= Open, Close <= High
                price_range_violations = data[
                    (data["low"] > data["open"]) |
                    (data["low"] > data["close"]) |
                    (data["high"] < data["open"]) |
                    (data["high"] < data["close"])
                ]
                if len(price_range_violations) > 0:
                    results["is_valid"] = False
                    results["violations"].append(
                        f"Found {len(price_range_violations)} rows with prices outside high-low range"
                    )
                    results["relationships"]["price_range_violations"] = len(price_range_violations)
                
                # Check for identical OHLC (potential data issue)
                identical_ohlc = data[
                    (data["open"] == data["high"]) &
                    (data["high"] == data["low"]) &
                    (data["low"] == data["close"]) &
                    (data["volume"] > 0)  # Should have some price movement if volume > 0
                ]
                if len(identical_ohlc) > 10:  # Allow some, but flag if many
                    results["warnings"].append(
                        f"Found {len(identical_ohlc)} rows with identical OHLC values but positive volume"
                    )
                    results["relationships"]["identical_ohlc_count"] = len(identical_ohlc)
            
            # Check price-volume relationships
            if "close" in data.columns and "volume" in data.columns:
                # Large price changes should often accompany large volume
                price_changes = data["close"].pct_change().abs()
                large_price_changes = price_changes > 0.05  # 5% change
                
                if large_price_changes.any():
                    avg_volume_on_large_moves = data.loc[large_price_changes, "volume"].mean()
                    avg_volume_normal = data.loc[~large_price_changes, "volume"].mean()
                    
                    if avg_volume_normal > 0:
                        volume_ratio = avg_volume_on_large_moves / avg_volume_normal
                        results["relationships"]["volume_ratio_on_large_moves"] = round(volume_ratio, 2)
                        
                        if volume_ratio < 1.5:  # Expect at least 50% more volume on large moves
                            results["warnings"].append(
                                "Large price movements not accompanied by increased volume"
                            )
            
            # Check for correlated anomalies
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                # Calculate correlation matrix
                corr_matrix = data[numeric_cols].corr()
                
                # Check for unexpected correlations
                high_corr_threshold = 0.95
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > high_corr_threshold:
                            col1, col2 = numeric_cols[i], numeric_cols[j]
                            if not (col1 in ohlc_cols and col2 in ohlc_cols):  # OHLC correlation is expected
                                results["warnings"].append(
                                    f"High correlation ({corr_value:.3f}) between {col1} and {col2}"
                                )
                                
        except Exception as e:
            results["is_valid"] = False
            results["violations"].append(f"Cross-field integrity check failed: {e}")
            
        return results
    
    async def _check_business_rules(self, data: pd.DataFrame) -> dict[str, Any]:
        """Check business rule integrity."""
        results = {
            "is_valid": True,
            "violations": [],
            "warnings": [],
            "rules_checked": []
        }
        
        try:
            rules = self.config["business_rules"]
            
            # Price relationship rules
            if rules.get("price_relationships", True):
                results["rules_checked"].append("price_relationships")
                
                # Check for zero prices (usually invalid for financial data)
                price_cols = ["open", "high", "low", "close"]
                for col in price_cols:
                    if col in data.columns:
                        zero_prices = (data[col] == 0).sum()
                        if zero_prices > 0:
                            results["violations"].append(
                                f"Found {zero_prices} zero values in {col} prices"
                            )
                            results["is_valid"] = False
                
                # Check for extreme price movements (circuit breaker levels)
                if "close" in data.columns:
                    daily_returns = data["close"].pct_change()
                    extreme_moves = daily_returns.abs() > 0.5  # 50% move
                    if extreme_moves.any():
                        results["warnings"].append(
                            f"Found {extreme_moves.sum()} extreme price movements (>50%)"
                        )
            
            # Volume constraint rules
            if rules.get("volume_constraints", True):
                results["rules_checked"].append("volume_constraints")
                
                if "volume" in data.columns:
                    # Check for negative volume
                    negative_volume = (data["volume"] < 0).sum()
                    if negative_volume > 0:
                        results["is_valid"] = False
                        results["violations"].append(
                            f"Found {negative_volume} negative volume values"
                        )
                    
                    # Check for unrealistic volume spikes
                    volume_mean = data["volume"].mean()
                    volume_std = data["volume"].std()
                    if volume_std > 0:
                        volume_zscore = (data["volume"] - volume_mean) / volume_std
                        extreme_volume = (volume_zscore > 10).sum()
                        if extreme_volume > 0:
                            results["warnings"].append(
                                f"Found {extreme_volume} extreme volume spikes (>10 std dev)"
                            )
            
            # Time constraint rules
            if rules.get("time_constraints", True) and isinstance(data.index, pd.DatetimeIndex):
                results["rules_checked"].append("time_constraints")
                
                # Check for weekend data (if not expected)
                weekend_data = data[data.index.weekday >= 5]
                if len(weekend_data) > 0:
                    weekend_pct = len(weekend_data) / len(data) * 100
                    if weekend_pct > 5:  # More than 5% weekend data is unusual
                        results["warnings"].append(
                            f"Found {len(weekend_data)} weekend data points ({weekend_pct:.1f}%)"
                        )
                
                # Check for data outside market hours (simplified check)
                # This would need to be adjusted based on the specific market
                off_hours_data = data[
                    (data.index.hour < 9) | (data.index.hour >= 17)
                ]
                if len(off_hours_data) > 0:
                    off_hours_pct = len(off_hours_data) / len(data) * 100
                    if off_hours_pct < 20:  # If most data is during hours, flag off-hours data
                        results["warnings"].append(
                            f"Found {len(off_hours_data)} data points outside typical market hours"
                        )
                        
        except Exception as e:
            results["is_valid"] = False
            results["violations"].append(f"Business rules check failed: {e}")
            
        return results
    
    async def _check_referential_integrity(
        self,
        data: pd.DataFrame,
        reference_data: pd.DataFrame
    ) -> dict[str, Any]:
        """Check referential integrity against reference data."""
        results = {
            "is_valid": True,
            "violations": [],
            "warnings": [],
            "comparison": {}
        }
        
        try:
            # Check overlapping time periods
            if isinstance(data.index, pd.DatetimeIndex) and isinstance(reference_data.index, pd.DatetimeIndex):
                data_start, data_end = data.index.min(), data.index.max()
                ref_start, ref_end = reference_data.index.min(), reference_data.index.max()
                
                # Find overlapping period
                overlap_start = max(data_start, ref_start)
                overlap_end = min(data_end, ref_end)
                
                if overlap_start <= overlap_end:
                    # Extract overlapping data
                    data_overlap = data.loc[overlap_start:overlap_end]
                    ref_overlap = reference_data.loc[overlap_start:overlap_end]
                    
                    results["comparison"]["overlap_period"] = {
                        "start": str(overlap_start),
                        "end": str(overlap_end),
                        "data_points": len(data_overlap),
                        "reference_points": len(ref_overlap)
                    }
                    
                    # Compare values for common timestamps
                    common_idx = data_overlap.index.intersection(ref_overlap.index)
                    if len(common_idx) > 0:
                        # Compare common columns
                        common_cols = list(set(data.columns).intersection(set(reference_data.columns)))
                        
                        for col in common_cols:
                            if col in ["open", "high", "low", "close", "volume"]:
                                data_vals = data_overlap.loc[common_idx, col]
                                ref_vals = ref_overlap.loc[common_idx, col]
                                
                                # Calculate differences
                                abs_diff = (data_vals - ref_vals).abs()
                                rel_diff = abs_diff / ref_vals.abs().replace(0, np.nan)
                                
                                # Check for significant differences
                                large_diffs = rel_diff > 0.01  # 1% threshold
                                if large_diffs.any():
                                    diff_count = large_diffs.sum()
                                    max_diff = rel_diff.max()
                                    
                                    if diff_count > len(common_idx) * 0.05:  # More than 5% differ
                                        results["violations"].append(
                                            f"Column '{col}' differs significantly from reference "
                                            f"({diff_count} points, max diff: {max_diff:.2%})"
                                        )
                                        results["is_valid"] = False
                                    else:
                                        results["warnings"].append(
                                            f"Column '{col}' has {diff_count} points differing from reference"
                                        )
                else:
                    results["warnings"].append("No overlapping time period with reference data")
            
            # Check data consistency metrics
            if len(data) > 100 and len(reference_data) > 100:
                # Compare statistical properties
                for col in ["close", "volume"]:
                    if col in data.columns and col in reference_data.columns:
                        data_mean = data[col].mean()
                        ref_mean = reference_data[col].mean()
                        
                        if ref_mean > 0:
                            mean_diff_pct = abs(data_mean - ref_mean) / ref_mean * 100
                            
                            if mean_diff_pct > 20:  # 20% difference in means
                                results["warnings"].append(
                                    f"Column '{col}' mean differs by {mean_diff_pct:.1f}% from reference"
                                )
                                
        except Exception as e:
            results["is_valid"] = False
            results["violations"].append(f"Referential integrity check failed: {e}")
            
        return results
    
    def _calculate_integrity_score(self, results: dict[str, Any]) -> float:
        """Calculate overall integrity score."""
        score = 100.0
        
        # Deduct for violations (major issues)
        score -= len(results.get("violations", [])) * 10
        
        # Deduct for warnings (minor issues)
        score -= len(results.get("warnings", [])) * 2
        
        # Ensure score is within bounds
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
    
    def _calculate_data_fingerprint(self, data: pd.DataFrame) -> str:
        """Calculate a fingerprint/hash of the data for verification."""
        try:
            # Create a string representation of key data properties
            fingerprint_data = f"{data.shape}_{data.index.min()}_{data.index.max()}"
            
            # Add column statistics
            for col in ["open", "high", "low", "close", "volume"]:
                if col in data.columns:
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        fingerprint_data += f"_{col}:{col_data.mean():.6f}:{col_data.std():.6f}"
            
            # Generate hash
            return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
            
        except Exception as e:
            self.logger.error(f"Failed to calculate data fingerprint: {e}")
            return "error"
    
    async def verify_data_consistency(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        check_type: str = "exact"
    ) -> dict[str, Any]:
        """Verify consistency between two datasets.
        
        Args:
            data1: First dataset
            data2: Second dataset
            check_type: Type of check ('exact', 'statistical', 'structural')
            
        Returns:
            dict: Consistency verification results
        """
        results = {
            "is_consistent": True,
            "check_type": check_type,
            "differences": [],
            "similarity_score": 0.0
        }
        
        try:
            if check_type == "exact":
                # Check for exact match
                if data1.shape != data2.shape:
                    results["is_consistent"] = False
                    results["differences"].append(
                        f"Shape mismatch: {data1.shape} vs {data2.shape}"
                    )
                else:
                    # Compare values
                    if not data1.equals(data2):
                        results["is_consistent"] = False
                        # Find differences
                        diff_mask = (data1 != data2).any(axis=1)
                        diff_count = diff_mask.sum()
                        results["differences"].append(
                            f"Found {diff_count} rows with differences"
                        )
                        results["similarity_score"] = (1 - diff_count / len(data1)) * 100
                    else:
                        results["similarity_score"] = 100.0
                        
            elif check_type == "statistical":
                # Compare statistical properties
                results["similarity_score"] = 100.0
                
                for col in data1.columns:
                    if col in data2.columns:
                        # Compare means
                        mean1, mean2 = data1[col].mean(), data2[col].mean()
                        if mean1 != 0:
                            mean_diff = abs(mean1 - mean2) / abs(mean1) * 100
                            if mean_diff > 5:  # 5% threshold
                                results["differences"].append(
                                    f"Column '{col}' mean differs by {mean_diff:.2f}%"
                                )
                                results["similarity_score"] -= mean_diff / 10
                        
                        # Compare standard deviations
                        std1, std2 = data1[col].std(), data2[col].std()
                        if std1 != 0:
                            std_diff = abs(std1 - std2) / std1 * 100
                            if std_diff > 10:  # 10% threshold
                                results["differences"].append(
                                    f"Column '{col}' std dev differs by {std_diff:.2f}%"
                                )
                                results["similarity_score"] -= std_diff / 20
                
                results["similarity_score"] = max(0, results["similarity_score"])
                results["is_consistent"] = len(results["differences"]) == 0
                
            elif check_type == "structural":
                # Compare structure only
                if set(data1.columns) != set(data2.columns):
                    results["is_consistent"] = False
                    missing_in_2 = set(data1.columns) - set(data2.columns)
                    missing_in_1 = set(data2.columns) - set(data1.columns)
                    
                    if missing_in_2:
                        results["differences"].append(f"Columns in data1 but not data2: {missing_in_2}")
                    if missing_in_1:
                        results["differences"].append(f"Columns in data2 but not data1: {missing_in_1}")
                    
                    common_cols = len(set(data1.columns).intersection(set(data2.columns)))
                    total_cols = len(set(data1.columns).union(set(data2.columns)))
                    results["similarity_score"] = (common_cols / total_cols) * 100 if total_cols > 0 else 0
                else:
                    results["similarity_score"] = 100.0
                    
        except Exception as e:
            results["is_consistent"] = False
            results["error"] = str(e)
            
        return results