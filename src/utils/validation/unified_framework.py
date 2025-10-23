"""
Unified Validation Framework

This module consolidates multiple validation implementations into a single, extensible
framework with a consistent API. It provides:

- Common validation primitives (levels, status, rule/result types)
- A rule-based engine to register and execute validations by category
- Preconfigured frameworks for:
  - Comprehensive function I/O + data quality + performance/business logic checks
  - Feature lookback optimization validations (data/results/pipeline)
- Optional HMM validation integration (when available)

Backwards-compatibility shims are provided by having legacy modules re-export
the preconfigured frameworks defined here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    psutil = None  # type: ignore

# ---------- Core types ----------

class ValidationLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ValidationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

@dataclass
class ValidationRule:
    name: str
    description: str
    level: ValidationLevel
    validator_func: Callable[[Any, Dict[str, Any]], Tuple[bool, str, Optional[Dict[str, Any]]]]
    required: bool = True
    auto_fix: bool = False
    fix_func: Optional[Callable[[Any], Any]] = None

@dataclass
class ValidationResult:
    rule_name: str
    status: ValidationStatus
    level: ValidationLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    auto_fixed: bool = False
    fix_applied: Optional[str] = None

@dataclass
class ValidationSummary:
    total_rules: int
    passed: int
    failed: int
    warnings: int
    skipped: int
    critical_failures: int
    overall_status: ValidationStatus
    quality_score: float
    recommendations: List[str]

class UnifiedValidationFramework:
    """Rule-based validation framework with pluggable rule sets by category."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.validation_history: List[Dict[str, Any]] = []

    # ----- Rule registration -----

    def register_rule(self, category: str, rule: ValidationRule) -> None:
        if category not in self.validation_rules:
            self.validation_rules[category] = []
        self.validation_rules[category].append(rule)

    def register_rules(self, category: str, rules: List[ValidationRule]) -> None:
        if not rules:
            return
        if category not in self.validation_rules:
            self.validation_rules[category] = []
        self.validation_rules[category].extend(rules)

    # ----- Execution -----

    def _execute_rule(self, rule: ValidationRule, data: Any, context: Dict[str, Any]) -> ValidationResult:
        try:
            is_valid, message, details = rule.validator_func(data, context)
            status = (
                ValidationStatus.PASSED
                if is_valid
                else (ValidationStatus.FAILED if rule.level == ValidationLevel.CRITICAL else ValidationStatus.WARNING)
            )
            return ValidationResult(
                rule_name=rule.name,
                status=status,
                level=rule.level,
                message=message,
                details=details,
            )
        except Exception as e:  # Defensive: single rule must not break whole validation
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.FAILED,
                level=rule.level,
                message=f"Validation error: {e}",
            )

    def validate_category(self, category: str, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        results: List[ValidationResult] = []
        fixed_data = data

        for rule in self.validation_rules.get(category, []):
            result = self._execute_rule(rule, fixed_data, context)
            results.append(result)

            if (
                not result.status == ValidationStatus.PASSED
                and rule.auto_fix
                and rule.fix_func is not None
                and fixed_data is not None
            ):
                try:
                    fixed_data = rule.fix_func(fixed_data)
                    result.auto_fixed = True
                    result.fix_applied = f"Applied auto-fix for {rule.name}"
                    result.status = ValidationStatus.PASSED
                except Exception as e:
                    # Keep original result; note auto-fix failure as warning
                    results.append(
                        ValidationResult(
                            rule_name=f"auto_fix::{rule.name}",
                            status=ValidationStatus.WARNING,
                            level=rule.level,
                            message=f"Auto-fix failed: {e}",
                        )
                    )

        critical_failures = [r for r in results if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL]
        overall_valid = len(critical_failures) == 0

        payload = {
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "overall_valid": overall_valid,
            "results": results,
        }

        self.validation_history.append({
            "category": category,
            "timestamp": payload["timestamp"],
            "overall_valid": overall_valid,
            "failed": len([r for r in results if r.status == ValidationStatus.FAILED]),
            "warnings": len([r for r in results if r.status == ValidationStatus.WARNING]),
        })

        return payload

    # ---------- Default rule packs (drawn from legacy implementations) ----------

    # --- Input validation rules ---
    @staticmethod
    def _rule_validate_dataframe_input(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if not isinstance(data, pd.DataFrame):
            return False, f"Expected DataFrame, got {type(data).__name__}", None
        if len(data) == 0:
            return False, "DataFrame is empty", None
        required_columns = context.get("required_columns", [])
        if required_columns:
            missing = [c for c in required_columns if c not in data.columns]
            if missing:
                return False, f"Missing required columns: {missing}", {"missing_columns": missing}
        critical_columns = context.get("critical_columns", [])
        warnings: List[str] = []
        for col in critical_columns:
            if col in data.columns and data[col].isna().any():
                warnings.append(f"Column '{col}' contains NaN values")
        if warnings:
            return True, "; ".join(warnings), {"warnings": warnings}
        return True, "Valid DataFrame input", None

    @staticmethod
    def _rule_validate_string_input(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if not isinstance(data, str):
            return False, f"Expected string, got {type(data).__name__}", None
        if not data.strip():
            return False, "String is empty or whitespace only", None
        min_length = context.get("min_length", 0)
        max_length = context.get("max_length", float("inf"))
        if len(data) < min_length:
            return False, f"String too short (min: {min_length})", None
        if len(data) > max_length:
            return False, f"String too long (max: {max_length})", None
        pattern = context.get("pattern")
        if pattern and not re.match(pattern, data):
            return False, f"String doesn't match required pattern: {pattern}", None
        return True, "Valid string input", None

    @staticmethod
    def _rule_validate_numeric_input(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if not isinstance(data, (int, float, np.number)):
            return False, f"Expected numeric, got {type(data).__name__}", None
        if np.isnan(data) or np.isinf(data):  # type: ignore[arg-type]
            return False, "Value is NaN or infinite", None
        min_value = context.get("min_value", float("-inf"))
        max_value = context.get("max_value", float("inf"))
        if data < min_value:
            return False, f"Value too small (min: {min_value})", None
        if data > max_value:
            return False, f"Value too large (max: {max_value})", None
        return True, "Valid numeric input", None

    @staticmethod
    def _rule_validate_path_input(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        path = Path(data) if not isinstance(data, Path) else data
        must_exist = context.get("must_exist", True)
        expected_type = context.get("expected_type", "file")
        expected_extensions = context.get("expected_extensions", [])
        if must_exist and not path.exists():
            return False, f"Path does not exist: {path}", None
        if path.exists():
            if expected_type == "file" and not path.is_file():
                return False, f"Expected file, got directory: {path}", None
            if expected_type == "directory" and not path.is_dir():
                return False, f"Expected directory, got file: {path}", None
        if expected_extensions and path.suffix.lower() not in expected_extensions:
            return False, f"Invalid file extension. Expected: {expected_extensions}", None
        return True, "Valid path input", None

    # --- Output validation rules ---
    @staticmethod
    def _rule_validate_dataframe_output(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if data is None:
            return False, "Output is None", None
        if not isinstance(data, pd.DataFrame):
            return False, f"Expected DataFrame output, got {type(data).__name__}", None
        required_columns = context.get("required_columns", [])
        if required_columns:
            missing = [c for c in required_columns if c not in data.columns]
            if missing:
                return False, f"Missing required output columns: {missing}", {"missing_columns": missing}
        if "label" in data.columns:
            counts = data["label"].value_counts()
            if len(counts) == 0:
                return True, "No labels generated", {"warnings": ["No labels generated"]}
            if len(counts) == 1:
                return True, "Only one label class generated", {"warnings": ["Only one label class generated"]}
        return True, "Valid DataFrame output", None

    @staticmethod
    def _rule_validate_boolean_output(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if not isinstance(data, bool):
            return False, f"Expected boolean output, got {type(data).__name__}", None
        expected_value = context.get("expected_value")
        if expected_value is not None and data != expected_value:
            return True, f"Expected {expected_value}, got {data}", {"warnings": ["Unexpected boolean value"]}
        return True, "Valid boolean output", None

    @staticmethod
    def _rule_validate_numeric_output(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if not isinstance(data, (int, float, np.number)):
            return False, f"Expected numeric output, got {type(data).__name__}", None
        if np.isnan(data) or np.isinf(data):  # type: ignore[arg-type]
            return False, "Output is NaN or infinite", None
        min_value = context.get("min_value", float("-inf"))
        max_value = context.get("max_value", float("inf"))
        if data < min_value or data > max_value:
            return True, f"Output value {data} outside expected range [{min_value}, {max_value}]", {"warnings": ["Out of range"]}
        return True, "Valid numeric output", None

    @staticmethod
    def _rule_validate_series_output(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if data is None:
            return False, "Output is None", None
        if not isinstance(data, pd.Series):
            return False, f"Expected Series output, got {type(data).__name__}", None
        warnings: List[str] = []
        if len(data) == 0:
            warnings.append("Output Series is empty")
        if data.isna().any():
            warnings.append(f"Output Series contains {int(data.isna().sum())} NaN values")
        if warnings:
            return True, "; ".join(warnings), {"warnings": warnings}
        return True, "Valid Series output", None

    # --- Data quality rules (OHLC/finance aware) ---
    @staticmethod
    def _rule_data_completeness(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if not isinstance(data, pd.DataFrame):
            return False, "Cannot validate completeness on non-DataFrame data", None
        total_cells = data.shape[0] * data.shape[1]
        non_null_cells = data.count().sum()
        completeness = non_null_cells / total_cells if total_cells > 0 else 0.0
        min_completeness = context.get("min_completeness", 0.95)
        if completeness < min_completeness:
            return False, f"Data completeness {completeness:.2%} below threshold {min_completeness:.2%}", {
                "completeness": completeness,
                "threshold": min_completeness,
            }
        return True, f"Data completeness acceptable: {completeness:.2%}", {"completeness": completeness}

    @staticmethod
    def _rule_no_infinite_values(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if not isinstance(data, pd.DataFrame):
            return False, "Cannot validate infinite values on non-DataFrame data", None
        numeric = data.select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric).sum().sum()
        if inf_count > 0:
            cols = numeric.columns[np.isinf(numeric).any()].tolist()
            return False, f"Found {int(inf_count)} infinite values", {"infinite_count": int(inf_count), "columns": cols}
        return True, "No infinite values found", None

    @staticmethod
    def _rule_price_consistency(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if not isinstance(data, pd.DataFrame):
            return False, "Cannot validate price consistency on non-DataFrame data", None
        issues: List[str] = []
        if all(c in data.columns for c in ["high", "open", "close"]):
            bad = (data["high"] < np.maximum(data["open"], data["close"])).sum()
            if bad > 0:
                issues.append(f"High < max(open, close): {int(bad)} rows")
        if all(c in data.columns for c in ["low", "open", "close"]):
            bad = (data["low"] > np.minimum(data["open"], data["close"])).sum()
            if bad > 0:
                issues.append(f"Low > min(open, close): {int(bad)} rows")
        if issues:
            return False, f"Price consistency issues: {'; '.join(issues)}", {"issues": issues}
        return True, "Price relationships are consistent", None

    @staticmethod
    def _rule_volume_positive(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if not isinstance(data, pd.DataFrame) or "volume" not in data.columns:
            return True, "Volume column not present, skipping", None
        neg = (data["volume"] < 0).sum()
        if neg > 0:
            return False, f"Found {int(neg)} negative volume values", {"negative_count": int(neg)}
        return True, "All volume values are non-negative", None

    @staticmethod
    def _rule_data_freshness(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if not isinstance(data, pd.DataFrame):
            return True, "Cannot validate freshness on non-DataFrame data", None
        ts_cols = [c for c in data.columns if "time" in c.lower() or "date" in c.lower()]
        if not ts_cols:
            return True, "No timestamp column found, skipping freshness validation", None
        col = ts_cols[0]
        try:
            timestamps = pd.to_datetime(data[col])
            latest = timestamps.max()
            days_old = (pd.Timestamp.now() - latest).days
            if days_old > context.get("max_days_old", 30):
                return False, f"Data is {days_old} days old", {"days_old": int(days_old), "latest_timestamp": str(latest)}
            return True, f"Data freshness OK: {days_old} days old", {"days_old": int(days_old), "latest_timestamp": str(latest)}
        except Exception:
            return True, "Could not parse timestamps, skipping freshness validation", None

    # --- Performance/resource rules ---
    @staticmethod
    def _rule_execution_time(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        exec_time = context.get("execution_time", 0.0)
        max_time = context.get("max_execution_time", 300.0)
        if exec_time > max_time:
            return False, f"Execution time {exec_time:.2f}s exceeds threshold {max_time:.2f}s", {
                "execution_time": exec_time,
                "max_execution_time": max_time,
            }
        return True, "Execution time within threshold", {"execution_time": exec_time}

    @staticmethod
    def _rule_memory_usage(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if psutil is None:
            return True, "psutil not available - skipping memory usage validation", None
        memory_usage_mb = context.get("memory_usage_mb")
        max_memory_mb = context.get("max_memory_mb", 1000.0)
        if memory_usage_mb is None:
            # Fallback to system available memory check only
            available = psutil.virtual_memory().available / (1024 * 1024)
            return True, f"System available memory: {available:.1f}MB", {"available_memory_mb": available}
        if memory_usage_mb > max_memory_mb:
            return False, f"Memory usage {memory_usage_mb:.1f}MB exceeds {max_memory_mb:.1f}MB", {
                "memory_usage_mb": memory_usage_mb,
                "max_memory_mb": max_memory_mb,
            }
        return True, "Memory usage within threshold", {"memory_usage_mb": memory_usage_mb}

    @staticmethod
    def _rule_cpu_usage(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if psutil is None:
            return True, "psutil not available - skipping CPU usage validation", None
        cpu_usage = context.get("cpu_usage_percent")
        max_cpu = context.get("max_cpu_percent", 80.0)
        if cpu_usage is None:
            cpu_usage = psutil.cpu_percent(interval=0.1)
        if cpu_usage > max_cpu:
            return False, f"CPU usage {cpu_usage:.1f}% exceeds threshold {max_cpu:.1f}%", {
                "cpu_usage_percent": cpu_usage,
                "max_cpu_percent": max_cpu,
            }
        return True, "CPU usage within threshold", {"cpu_usage_percent": cpu_usage}

    # --- Business logic rules ---
    @staticmethod
    def _rule_labeling_logic(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if isinstance(data, pd.DataFrame) and "label" in data.columns:
            labels = data["label"].dropna()
            if len(labels) == 0:
                return True, "No labels generated", {"warnings": ["No labels generated"]}
            counts = labels.value_counts()
            if len(counts) > 1:
                max_count = counts.max()
                min_count = counts.min()
                ratio = max_count / max(1, min_count)
                if ratio > 10:
                    return True, f"Severe class imbalance (ratio: {ratio:.1f})", {"warnings": ["Severe class imbalance"]}
        return True, "Labeling logic validation OK", None

    @staticmethod
    def _rule_regime_logic(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if not isinstance(data, pd.DataFrame):
            return True, "Not a DataFrame - skipping regime logic", None
        regime_cols = [c for c in data.columns if "regime" in c.lower()]
        warnings: List[str] = []
        for col in regime_cols:
            regimes = data[col].dropna()
            if len(regimes) > 1:
                counts = regimes.value_counts()
                if len(counts) < 2:
                    warnings.append(f"Only {len(counts)} regime(s) in {col}")
                elif len(counts) > 10:
                    warnings.append(f"Too many regimes ({len(counts)}) in {col}")
                if len(counts) > 1:
                    ratio = counts.max() / max(1, counts.min())
                    if ratio > 5:
                        warnings.append(f"Unbalanced regimes in {col} (ratio: {ratio:.1f})")
        if warnings:
            return True, "; ".join(warnings), {"warnings": warnings}
        return True, "Regime logic validation OK", None

    @staticmethod
    def _rule_triple_barrier_logic(data: Any, context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if not isinstance(data, pd.DataFrame):
            return True, "Not a DataFrame - skipping triple barrier logic", None
        tb_cols = [c for c in data.columns if "triple_barrier" in c.lower()]
        warnings: List[str] = []
        for col in tb_cols:
            labels = data[col].dropna()
            if len(labels) == 0:
                continue
            valid = labels.isin([-1, 0, 1])
            if not valid.all():
                invalid = labels[~valid].unique().tolist()
                warnings.append(f"Invalid triple barrier labels in {col}: {invalid}")
            counts = labels.value_counts()
            total = len(labels)
            neutral_ratio = counts.get(0, 0) / max(1, total)
            if neutral_ratio > 0.8:
                warnings.append(f"Too many neutral labels in {col} ({neutral_ratio:.1%})")
        if warnings:
            return True, "; ".join(warnings), {"warnings": warnings}
        return True, "Triple barrier logic validation OK", None

    # ----- Helper: build common default packs -----

    def register_default_io_rules(self) -> None:
        self.register_rules(
            "input_validation",
            [
                ValidationRule("dataframe_input", "Validate DataFrame input", ValidationLevel.CRITICAL, self._rule_validate_dataframe_input),
                ValidationRule("string_input", "Validate string input", ValidationLevel.HIGH, self._rule_validate_string_input, required=False),
                ValidationRule("numeric_input", "Validate numeric input", ValidationLevel.HIGH, self._rule_validate_numeric_input, required=False),
                ValidationRule("path_input", "Validate path input", ValidationLevel.HIGH, self._rule_validate_path_input, required=False),
            ],
        )
        self.register_rules(
            "output_validation",
            [
                ValidationRule("dataframe_output", "Validate DataFrame output", ValidationLevel.CRITICAL, self._rule_validate_dataframe_output),
                ValidationRule("boolean_output", "Validate boolean output", ValidationLevel.HIGH, self._rule_validate_boolean_output, required=False),
                ValidationRule("numeric_output", "Validate numeric output", ValidationLevel.HIGH, self._rule_validate_numeric_output, required=False),
                ValidationRule("series_output", "Validate Series output", ValidationLevel.HIGH, self._rule_validate_series_output, required=False),
            ],
        )

    def register_default_data_quality_rules(self) -> None:
        self.register_rules(
            "data_quality",
            [
                ValidationRule("data_completeness", "Data completeness is above threshold", ValidationLevel.HIGH, self._rule_data_completeness),
                ValidationRule("no_infinite_values", "No infinite values in numeric columns", ValidationLevel.HIGH, self._rule_no_infinite_values, auto_fix=True, fix_func=self._fix_infinite_values),
                ValidationRule("price_consistency", "OHLC price relationships are consistent", ValidationLevel.MEDIUM, self._rule_price_consistency, auto_fix=True, fix_func=self._fix_price_consistency, required=False),
                ValidationRule("volume_positive", "Volume values are non-negative", ValidationLevel.MEDIUM, self._rule_volume_positive, auto_fix=True, fix_func=self._fix_negative_volume, required=False),
                ValidationRule("data_freshness", "Data is not too old", ValidationLevel.LOW, self._rule_data_freshness, required=False),
            ],
        )

    def register_default_performance_rules(self) -> None:
        self.register_rules(
            "performance_validation",
            [
                ValidationRule("execution_time", "Execution time within threshold", ValidationLevel.LOW, self._rule_execution_time, required=False),
                ValidationRule("memory_usage", "Memory usage within threshold", ValidationLevel.LOW, self._rule_memory_usage, required=False),
                ValidationRule("cpu_usage", "CPU usage within threshold", ValidationLevel.LOW, self._rule_cpu_usage, required=False),
            ],
        )

    def register_default_business_rules(self) -> None:
        self.register_rules(
            "business_logic",
            [
                ValidationRule("labeling_logic", "Validate labeling logic", ValidationLevel.MEDIUM, self._rule_labeling_logic, required=False),
                ValidationRule("regime_logic", "Validate regime logic", ValidationLevel.MEDIUM, self._rule_regime_logic, required=False),
                ValidationRule("triple_barrier_logic", "Validate triple barrier logic", ValidationLevel.MEDIUM, self._rule_triple_barrier_logic, required=False),
            ],
        )

    # ----- Auto-fix helpers -----
    @staticmethod
    def _fix_infinite_values(data: pd.DataFrame) -> pd.DataFrame:
        fixed = data.copy()
        numeric = fixed.select_dtypes(include=[np.number])
        fixed[numeric.columns] = numeric.replace([np.inf, -np.inf], np.nan)
        fixed = fixed.ffill().bfill()
        return fixed

    @staticmethod
    def _fix_price_consistency(data: pd.DataFrame) -> pd.DataFrame:
        fixed = data.copy()
        if all(c in fixed.columns for c in ["open", "high", "low", "close"]):
            fixed["high"] = np.maximum(fixed["high"], np.maximum(fixed["open"], fixed["close"]))
            fixed["low"] = np.minimum(fixed["low"], np.minimum(fixed["open"], fixed["close"]))
        return fixed

    @staticmethod
    def _fix_negative_volume(data: pd.DataFrame) -> pd.DataFrame:
        fixed = data.copy()
        if "volume" in fixed.columns:
            fixed["volume"] = np.maximum(fixed["volume"], 0)
        return fixed

# ---------- Preconfigured frameworks (backward-compatibility) ----------

class ComprehensiveValidationFramework(UnifiedValidationFramework):
    """Preconfigured framework for function input/output + data/performance/business rules."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        super().__init__(logger=logger)
        self.register_default_io_rules()
        self.register_default_data_quality_rules()
        self.register_default_performance_rules()
        self.register_default_business_rules()

    def validate_function_input(self, function_name: str, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = dict(context or {})
        context.setdefault("function_name", function_name)
        result = self.validate_category("input_validation", input_data, context)
        # Also run data quality on inputs when DataFrame
        if isinstance(input_data, pd.DataFrame):
            dq = self.validate_category("data_quality", input_data, context)
            # Merge summaries
            result["data_quality"] = dq
            result["overall_valid"] = result["overall_valid"] and dq["overall_valid"]
        return result

    def validate_function_output(self, function_name: str, output_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = dict(context or {})
        context.setdefault("function_name", function_name)
        result = self.validate_category("output_validation", output_data, context)
        # Also run quality/business logic on outputs when DataFrame
        if isinstance(output_data, pd.DataFrame):
            dq = self.validate_category("data_quality", output_data, context)
            bl = self.validate_category("business_logic", output_data, context)
            result["data_quality"] = dq
            result["business_logic"] = bl
            result["overall_valid"] = result["overall_valid"] and dq["overall_valid"] and bl["overall_valid"]
        return result

    def generate_validation_report(self) -> Dict[str, Any]:
        if not self.validation_history:
            return {"total_validations": 0, "message": "No validation data recorded"}
        total = len(self.validation_history)
        successes = len([v for v in self.validation_history if v.get("overall_valid")])
        failures = total - successes
        # Simple pattern aggregation
        return {
            "total_validations": total,
            "successful_validations": successes,
            "failed_validations": failures,
            "success_rate": (successes / total * 100.0) if total > 0 else 0.0,
        }

    def log_validation_report(self, report: Dict[str, Any]) -> None:
        try:
            self.logger.info("\n📋 COMPREHENSIVE VALIDATION REPORT")
            self.logger.info("=" * 50)
            self.logger.info(f"Total Validations: {report.get('total_validations', 0)}")
            self.logger.info(f"Successful Validations: {report.get('successful_validations', 0)}")
            self.logger.info(f"Failed Validations: {report.get('failed_validations', 0)}")
            self.logger.info(f"Success Rate: {report.get('success_rate', 0.0):.1f}%")
        except Exception as e:
            self.logger.error(f"Failed to log validation report: {e}")

def comprehensive_validation(validator: ComprehensiveValidationFramework) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to validate function input/output using the comprehensive framework."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            input_context = {"function_name": func.__name__, "args_count": len(args), "kwargs_keys": list(kwargs.keys())}
            if args and isinstance(args[0], pd.DataFrame):
                _ = validator.validate_function_input(func.__name__, args[0], input_context)
            result = func(*args, **kwargs)
            output_context = {"function_name": func.__name__}
            _ = validator.validate_function_output(func.__name__, result, output_context)
            return result

        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            input_context = {"function_name": func.__name__, "args_count": len(args), "kwargs_keys": list(kwargs.keys())}
            if args and isinstance(args[0], pd.DataFrame):
                _ = validator.validate_function_input(func.__name__, args[0], input_context)
            result = await func(*args, **kwargs)
            output_context = {"function_name": func.__name__}
            _ = validator.validate_function_output(func.__name__, result, output_context)
            return result

        # Return appropriate wrapper
        import asyncio as _asyncio  # Local import to avoid global dependency

        return async_wrapper if _asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

class FeatureLookbackValidationFramework(UnifiedValidationFramework):
    """Preconfigured framework mirroring Feature Lookback Optimization validation sets."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        super().__init__(logger=logger)
        # Reuse data quality and add optimization/pipeline specific rule sets
        self.register_default_data_quality_rules()

        # Optimization rules
        self.register_rules(
            "optimization",
            [
                ValidationRule(
                    name="optimization_results_present",
                    description="Optimization results are present and non-empty",
                    level=ValidationLevel.CRITICAL,
                    validator_func=self._rule_optimization_results_present,
                ),
                ValidationRule(
                    name="optimized_features_present",
                    description="Optimized features are present and non-empty",
                    level=ValidationLevel.CRITICAL,
                    validator_func=self._rule_optimized_features_present,
                ),
                ValidationRule(
                    name="best_score_valid",
                    description="Best optimization score is valid",
                    level=ValidationLevel.HIGH,
                    validator_func=self._rule_best_score_valid,
                ),
                ValidationRule(
                    name="lookback_periods_valid",
                    description="Lookback periods are within valid range",
                    level=ValidationLevel.HIGH,
                    validator_func=self._rule_lookback_periods_valid,
                ),
            ],
        )

        # Pipeline rules
        self.register_rules(
            "pipeline",
            [
                ValidationRule(
                    name="labeling_results_present",
                    description="Labeling results are present (multi-horizon or triple barrier)",
                    level=ValidationLevel.CRITICAL,
                    validator_func=self._rule_labeling_results_present,
                ),
            ],
        )

    # --- Public API (mirrors legacy) ---
    def validate_data(self, data: Any) -> Tuple[bool, List[ValidationResult], Optional[Any]]:
        dq = self.validate_category("data_quality", data)
        results: List[ValidationResult] = list(dq["results"])  # type: ignore[index]
        critical_failures = [r for r in results if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL]
        return len(critical_failures) == 0, results, data

    def validate_optimization_results(self, optimization_result: Dict[str, Any]) -> Tuple[bool, List[ValidationResult]]:
        res = self.validate_category("optimization", optimization_result)
        results: List[ValidationResult] = list(res["results"])  # type: ignore[index]
        critical_failures = [r for r in results if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL]
        return len(critical_failures) == 0, results

    def validate_pipeline_state(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, List[ValidationResult]]:
        res = self.validate_category("pipeline", pipeline_state)
        results: List[ValidationResult] = list(res["results"])  # type: ignore[index]
        critical_failures = [r for r in results if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL]
        return len(critical_failures) == 0, results

    def generate_validation_summary(self, validation_results: List[ValidationResult]) -> ValidationSummary:
        total = len(validation_results)
        passed = len([r for r in validation_results if r.status == ValidationStatus.PASSED])
        failed = len([r for r in validation_results if r.status == ValidationStatus.FAILED])
        warnings = len([r for r in validation_results if r.status == ValidationStatus.WARNING])
        skipped = len([r for r in validation_results if r.status == ValidationStatus.SKIPPED])
        critical_failures = len([r for r in validation_results if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL])
        quality_score = passed / total if total > 0 else 0.0
        if critical_failures > 0:
            overall = ValidationStatus.FAILED
        elif failed > 0:
            overall = ValidationStatus.WARNING
        else:
            overall = ValidationStatus.PASSED
        return ValidationSummary(
            total_rules=total,
            passed=passed,
            failed=failed,
            warnings=warnings,
            skipped=skipped,
            critical_failures=critical_failures,
            overall_status=overall,
            quality_score=quality_score,
            recommendations=self._generate_recommendations(validation_results),
        )

    # --- Optimization rule implementations ---
    @staticmethod
    def _rule_optimization_results_present(optimization_result: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        results = optimization_result.get("optimization_results", {})
        if not results:
            return False, "No optimization results found", None
        required = ["best_lookback_period", "best_score", "optimization_method"]
        missing = [k for k in required if k not in results]
        if missing:
            return False, f"Missing optimization result keys: {missing}", {"missing_keys": missing}
        return True, "Optimization results present", None

    @staticmethod
    def _rule_optimized_features_present(optimization_result: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        features = optimization_result.get("optimized_features", {})
        if not features:
            return False, "No optimized features found", None
        return True, f"Found {len(features)} optimized features", {"feature_count": len(features)}

    @staticmethod
    def _rule_best_score_valid(optimization_result: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        results = optimization_result.get("optimization_results", {})
        best_score = results.get("best_score", 0.0)
        if best_score <= 0:
            return False, f"Best score is invalid: {best_score}", {"best_score": best_score}
        return True, f"Best score valid: {best_score}", {"best_score": best_score}

    @staticmethod
    def _rule_lookback_periods_valid(optimization_result: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        features = optimization_result.get("optimized_features", {})
        if not features:
            return False, "No optimized features to validate", None
        invalid: List[str] = []
        for name, data in features.items():
            lookback = data.get("lookback", 0)
            if lookback <= 0 or lookback > 1000:
                invalid.append(f"{name}: {lookback}")
        if invalid:
            return False, f"Invalid lookback periods: {'; '.join(invalid)}", {"invalid_periods": invalid}
        return True, "All lookback periods valid", None

    # --- Pipeline rule implementations ---
    @staticmethod
    def _rule_labeling_results_present(pipeline_state: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        mh = pipeline_state.get("multi_horizon_labeling_result", {})
        if mh:
            required = ["labeled_data", "labeling_metrics", "method"]
            missing = [k for k in required if k not in mh]
            if missing:
                return False, f"Missing multi-horizon labeling keys: {missing}", {"missing_keys": missing}
            return True, "Multi-horizon labeling results present", None
        tb = pipeline_state.get("triple_barrier_labeling_result", {})
        if tb:
            required = ["labels", "barriers", "metadata"]
            missing = [k for k in required if k not in tb]
            if missing:
                return False, f"Missing triple barrier labeling keys: {missing}", {"missing_keys": missing}
            return True, "Triple barrier labeling results present", None
        return False, "No labeling results found (neither multi-horizon nor triple barrier)", None

    @staticmethod
    def _generate_recommendations(validation_results: List[ValidationResult]) -> List[str]:
        recs: List[str] = []
        for r in validation_results:
            if r.status == ValidationStatus.FAILED:
                if r.rule_name == "data_completeness":
                    recs.append("Improve data quality by addressing missing values")
                elif r.rule_name == "no_infinite_values":
                    recs.append("Clean data to remove infinite values")
                elif r.rule_name == "price_consistency":
                    recs.append("Fix OHLC price relationship inconsistencies")
                elif r.rule_name == "best_score_valid":
                    recs.append("Review optimization parameters and data quality")
        return recs or ["All validations passed - no recommendations needed"]

# ---------- Optional HMM integration ----------

class HMMValidatorAdapter:
    """Adapter that exposes a simple API and defers to HMMValidationFramework if available."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._hmm = None
        try:
            from src.utils.ml_common.hmm_validation_metrics import HMMValidationFramework  # noqa: WPS433

            self._hmm = HMMValidationFramework()
        except Exception as e:  # pragma: no cover - optional dependency
            self.logger.warning(f"HMM validation metrics not available: {e}")

    def validate_hmm_regimes(self, regime_data: pd.DataFrame, original_data: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> Any:
        if not self._hmm:
            return {
                "hmm_validation_metrics": {"hmm_quality_score": 0.0, "validation_passed": False},
                "error": "HMMValidationFramework not available",
            }
        return self._hmm.validate_hmm_regimes(regime_data, original_data, feature_columns)

__all__ = [
    "ValidationLevel",
    "ValidationStatus",
    "ValidationRule",
    "ValidationResult",
    "ValidationSummary",
    "UnifiedValidationFramework",
    "ComprehensiveValidationFramework",
    "FeatureLookbackValidationFramework",
    "HMMValidatorAdapter",
    "comprehensive_validation",
]
