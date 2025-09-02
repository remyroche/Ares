"""
Cross-Step Data Consistency Validation Module

This module provides validation for data consistency between pipeline steps,
ensuring data integrity is maintained throughout transformations.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import system_logger
from src.utils.pipeline_standards import ValidationIssue, ValidationResult, DataQualityLevel


class CrossStepValidator:
    """Validates data consistency between pipeline steps."""
    
    def __init__(self, logger=None):
        self.logger = logger or system_logger.getChild("CrossStepValidator")
        self.validation_history = {}
        self.step_metadata = {}
        
    def validate_step_transition(
        self,
        previous_step_output: pd.DataFrame,
        current_step_input: pd.DataFrame,
        previous_step_name: str,
        current_step_name: str,
        tolerance: Dict[str, float] = None
    ) -> ValidationResult:
        """
        Validate data consistency between consecutive steps.
        
        Args:
            previous_step_output: Output data from previous step
            current_step_input: Input data to current step
            previous_step_name: Name of previous step
            current_step_name: Name of current step
            tolerance: Tolerance levels for various checks
            
        Returns:
            ValidationResult with detailed findings
        """
        self.logger.info(f"🔍 Validating transition: {previous_step_name} → {current_step_name}")
        
        result = ValidationResult(passed=True)
        tolerance = tolerance or {
            "row_count_change": 0.01,  # 1% tolerance
            "timestamp_drift": 1000,   # 1 second in milliseconds
            "column_preservation": 1.0, # 100% required columns must be preserved
            "value_drift": 0.001       # 0.1% value drift tolerance
        }
        
        # Check if dataframes are valid
        if previous_step_output is None or current_step_input is None:
            result.passed = False
            result.issues.append(ValidationIssue(
                severity=DataQualityLevel.CRITICAL,
                message="One or both dataframes are None",
                details={
                    "previous_step": previous_step_name,
                    "current_step": current_step_name
                }
            ))
            return result
        
        # 1. Row Count Consistency
        row_count_prev = len(previous_step_output)
        row_count_curr = len(current_step_input)
        row_count_change = abs(row_count_curr - row_count_prev) / row_count_prev if row_count_prev > 0 else float('inf')
        
        if row_count_change > tolerance["row_count_change"]:
            severity = DataQualityLevel.CRITICAL if row_count_change > 0.1 else DataQualityLevel.WARNING
            result.issues.append(ValidationIssue(
                severity=severity,
                message=f"Significant row count change detected: {row_count_prev} → {row_count_curr}",
                details={
                    "previous_rows": row_count_prev,
                    "current_rows": row_count_curr,
                    "change_percentage": row_count_change * 100
                }
            ))
            if severity == DataQualityLevel.CRITICAL:
                result.passed = False
        
        # 2. Column Preservation
        prev_columns = set(previous_step_output.columns)
        curr_columns = set(current_step_input.columns)
        
        # Identify critical columns that should be preserved
        critical_columns = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        critical_preserved = critical_columns.intersection(prev_columns).intersection(curr_columns)
        critical_lost = critical_columns.intersection(prev_columns) - curr_columns
        
        if critical_lost:
            result.passed = False
            result.issues.append(ValidationIssue(
                severity=DataQualityLevel.CRITICAL,
                message=f"Critical columns lost in transition: {critical_lost}",
                details={
                    "lost_columns": list(critical_lost),
                    "previous_columns": list(prev_columns),
                    "current_columns": list(curr_columns)
                }
            ))
        
        # Check for unexpected column additions
        new_columns = curr_columns - prev_columns
        if len(new_columns) > 20:  # Arbitrary threshold
            result.warnings.append(ValidationIssue(
                severity=DataQualityLevel.WARNING,
                message=f"Large number of new columns added: {len(new_columns)}",
                details={"new_columns": list(new_columns)[:10] + ["..."]}  # Show first 10
            ))
        
        # 3. Timestamp Continuity
        if 'timestamp' in critical_preserved:
            # Check timestamp alignment
            try:
                prev_timestamps = previous_step_output['timestamp'].sort_values()
                curr_timestamps = current_step_input['timestamp'].sort_values()
                
                # Check range preservation
                prev_min, prev_max = prev_timestamps.min(), prev_timestamps.max()
                curr_min, curr_max = curr_timestamps.min(), curr_timestamps.max()
                
                if abs(prev_min - curr_min) > tolerance["timestamp_drift"]:
                    result.warnings.append(ValidationIssue(
                        severity=DataQualityLevel.WARNING,
                        message="Timestamp range start has drifted",
                        details={
                            "previous_min": prev_min,
                            "current_min": curr_min,
                            "drift_ms": abs(prev_min - curr_min)
                        }
                    ))
                
                if abs(prev_max - curr_max) > tolerance["timestamp_drift"]:
                    result.warnings.append(ValidationIssue(
                        severity=DataQualityLevel.WARNING,
                        message="Timestamp range end has drifted",
                        details={
                            "previous_max": prev_max,
                            "current_max": curr_max,
                            "drift_ms": abs(prev_max - curr_max)
                        }
                    ))
                
                # Check for gaps
                self._check_timestamp_gaps(prev_timestamps, curr_timestamps, result)
                
            except Exception as e:
                result.warnings.append(ValidationIssue(
                    severity=DataQualityLevel.WARNING,
                    message=f"Failed to validate timestamp continuity: {str(e)}"
                ))
        
        # 4. Data Integrity Checks
        for column in critical_preserved:
            if column == 'timestamp':
                continue
                
            try:
                # Check for value preservation in overlapping rows
                if row_count_curr > 0 and row_count_prev > 0:
                    # Sample check on first 1000 rows
                    sample_size = min(1000, row_count_prev, row_count_curr)
                    
                    prev_sample = previous_step_output[column].iloc[:sample_size]
                    curr_sample = current_step_input[column].iloc[:sample_size]
                    
                    # Check for unexpected value changes
                    value_diff = np.abs(prev_sample.values - curr_sample.values)
                    max_diff = np.nanmax(value_diff)
                    
                    if max_diff > tolerance["value_drift"] * np.nanmax(prev_sample.values):
                        result.warnings.append(ValidationIssue(
                            severity=DataQualityLevel.WARNING,
                            message=f"Unexpected value changes in column '{column}'",
                            details={
                                "max_difference": float(max_diff),
                                "sample_size": sample_size
                            }
                        ))
                        
            except Exception as e:
                self.logger.debug(f"Could not compare values for column {column}: {e}")
        
        # 5. Statistical Fingerprint
        fingerprint_match = self._validate_statistical_fingerprint(
            previous_step_output, current_step_input, critical_preserved
        )
        if not fingerprint_match["matches"]:
            result.warnings.append(ValidationIssue(
                severity=DataQualityLevel.WARNING,
                message="Statistical fingerprint mismatch detected",
                details=fingerprint_match["details"]
            ))
        
        # Calculate quality score
        total_checks = 5
        critical_issues = len([i for i in result.issues if i.severity == DataQualityLevel.CRITICAL])
        warning_issues = len([i for i in result.issues if i.severity == DataQualityLevel.WARNING])
        
        result.quality_score = max(0, 1 - (critical_issues * 0.3 + warning_issues * 0.1))
        result.passed = result.passed and critical_issues == 0
        
        # Store validation metadata
        self._store_validation_metadata(
            previous_step_name, current_step_name, result, 
            row_count_prev, row_count_curr
        )
        
        return result
    
    def _check_timestamp_gaps(
        self, 
        prev_timestamps: pd.Series, 
        curr_timestamps: pd.Series,
        result: ValidationResult
    ) -> None:
        """Check for unexpected gaps in timestamps."""
        try:
            # Calculate expected frequency
            prev_freq = pd.Series(prev_timestamps).diff().median()
            curr_freq = pd.Series(curr_timestamps).diff().median()
            
            if abs(prev_freq - curr_freq) > 1000:  # 1 second tolerance
                result.warnings.append(ValidationIssue(
                    severity=DataQualityLevel.WARNING,
                    message="Timestamp frequency has changed",
                    details={
                        "previous_freq_ms": float(prev_freq),
                        "current_freq_ms": float(curr_freq)
                    }
                ))
            
            # Check for large gaps
            curr_gaps = pd.Series(curr_timestamps).diff()
            large_gaps = curr_gaps[curr_gaps > prev_freq * 10]  # Gaps 10x larger than normal
            
            if len(large_gaps) > 0:
                result.warnings.append(ValidationIssue(
                    severity=DataQualityLevel.WARNING,
                    message=f"Found {len(large_gaps)} large timestamp gaps",
                    details={
                        "gap_count": len(large_gaps),
                        "max_gap_ms": float(large_gaps.max()),
                        "gap_locations": large_gaps.index.tolist()[:5]  # First 5 locations
                    }
                ))
                
        except Exception as e:
            self.logger.debug(f"Gap analysis failed: {e}")
    
    def _validate_statistical_fingerprint(
        self,
        prev_df: pd.DataFrame,
        curr_df: pd.DataFrame,
        columns: set
    ) -> Dict[str, Any]:
        """Create and compare statistical fingerprints of the data."""
        fingerprint_match = {"matches": True, "details": {}}
        
        for column in columns:
            if column == 'timestamp':
                continue
                
            try:
                if column in prev_df.columns and column in curr_df.columns:
                    prev_stats = {
                        "mean": float(prev_df[column].mean()),
                        "std": float(prev_df[column].std()),
                        "skew": float(prev_df[column].skew()),
                        "kurtosis": float(prev_df[column].kurtosis())
                    }
                    
                    curr_stats = {
                        "mean": float(curr_df[column].mean()),
                        "std": float(curr_df[column].std()),
                        "skew": float(curr_df[column].skew()),
                        "kurtosis": float(curr_df[column].kurtosis())
                    }
                    
                    # Check for significant changes
                    mean_change = abs(curr_stats["mean"] - prev_stats["mean"]) / (abs(prev_stats["mean"]) + 1e-10)
                    std_change = abs(curr_stats["std"] - prev_stats["std"]) / (abs(prev_stats["std"]) + 1e-10)
                    
                    if mean_change > 0.1 or std_change > 0.2:  # 10% mean change or 20% std change
                        fingerprint_match["matches"] = False
                        fingerprint_match["details"][column] = {
                            "mean_change_pct": mean_change * 100,
                            "std_change_pct": std_change * 100,
                            "previous_stats": prev_stats,
                            "current_stats": curr_stats
                        }
                        
            except Exception as e:
                self.logger.debug(f"Statistical fingerprint failed for {column}: {e}")
        
        return fingerprint_match
    
    def _store_validation_metadata(
        self,
        prev_step: str,
        curr_step: str,
        result: ValidationResult,
        prev_rows: int,
        curr_rows: int
    ) -> None:
        """Store metadata about the validation for tracking."""
        transition_key = f"{prev_step}→{curr_step}"
        
        self.validation_history[transition_key] = {
            "timestamp": datetime.now().isoformat(),
            "validation_passed": result.passed,
            "quality_score": result.quality_score,
            "row_count_change": curr_rows - prev_rows,
            "issue_count": len(result.issues),
            "warning_count": len(result.warnings)
        }
        
        self.step_metadata[curr_step] = {
            "input_rows": curr_rows,
            "validation_timestamp": datetime.now().isoformat(),
            "from_step": prev_step
        }
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed."""
        return {
            "total_validations": len(self.validation_history),
            "passed_validations": sum(1 for v in self.validation_history.values() if v["validation_passed"]),
            "average_quality_score": np.mean([v["quality_score"] for v in self.validation_history.values()]),
            "validation_history": self.validation_history,
            "step_metadata": self.step_metadata
        }
    
    def validate_pipeline_continuity(
        self,
        step_outputs: Dict[str, pd.DataFrame],
        expected_flow: List[str]
    ) -> ValidationResult:
        """
        Validate the entire pipeline flow for data continuity.
        
        Args:
            step_outputs: Dictionary mapping step names to their output dataframes
            expected_flow: Ordered list of step names in expected execution order
            
        Returns:
            Comprehensive validation result for the pipeline
        """
        result = ValidationResult(passed=True)
        
        for i in range(len(expected_flow) - 1):
            prev_step = expected_flow[i]
            curr_step = expected_flow[i + 1]
            
            if prev_step not in step_outputs or curr_step not in step_outputs:
                result.warnings.append(ValidationIssue(
                    severity=DataQualityLevel.WARNING,
                    message=f"Missing data for steps: {prev_step} or {curr_step}"
                ))
                continue
            
            # Validate each transition
            transition_result = self.validate_step_transition(
                step_outputs[prev_step],
                step_outputs[curr_step],
                prev_step,
                curr_step
            )
            
            # Aggregate results
            result.issues.extend(transition_result.issues)
            result.warnings.extend(transition_result.warnings)
            result.passed = result.passed and transition_result.passed
        
        # Calculate overall quality score
        if len(expected_flow) > 1:
            transition_scores = [
                self.validation_history.get(f"{expected_flow[i]}→{expected_flow[i+1]}", {}).get("quality_score", 0)
                for i in range(len(expected_flow) - 1)
            ]
            result.quality_score = np.mean(transition_scores) if transition_scores else 0
        
        return result