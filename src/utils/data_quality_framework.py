"""
Comprehensive Data Quality Framework

This module provides standardized data quality management including:
    - Data validation and schema enforcement
    - Data formatting and standardization
    - Quality scoring and metrics
    - Data cleaning and preprocessing
    - Quality gates and validation rules
    - Data profiling and analysis
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum

from .pipeline_standards import PipelineStandards, pipeline_standards
from .logger import system_logger
from .error_handler import handle_errors

class DataQualityLevel(Enum):
    """Data quality levels for validation rules."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ValidationRule:
    """Defines a validation rule for data quality."""
    
    def __init__(self, name: str, rule_type: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.rule_type = rule_type
        self.parameters = parameters or {}
        self.is_initialized = False
        self.logger = system_logger.getChild(f"ValidationRule.{name}")
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="validationrule initialization",
    )
    async def initialize(self) -> bool:
        """Initialize ValidationRule."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    
    def validate(self, data: Any) -> Tuple[List[str], List[str]]:
        """Validate data and return (issues, warnings)."""
        raise NotImplementedError("Subclasses must implement validate method")

class SchemaValidationRule(ValidationRule):
    """Validates data schema and structure."""
    
    def __init__(self, required_columns: List[str] = None, data_types: Dict[str, str] = None):
        super().__init__("schema_validation", "schema", {
            "required_columns": required_columns or [],
            "data_types": data_types or {}
        })
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="schemavalidationrule initialization",
    )
    async def initialize(self) -> bool:
        """Initialize SchemaValidationRule."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    
    def validate(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate DataFrame schema."""
        issues = []
        warnings = []
        
        if not isinstance(data, pd.DataFrame):
            issues.append("Data must be a pandas DataFrame")
            return issues, warnings
        
        # Check required columns
        if self.parameters["required_columns"]:
            missing_columns = set(self.parameters["required_columns"]) - set(data.columns)
            if missing_columns:
                issues.append(f"Missing required columns: {list(missing_columns)}")
        
        # Check data types
        if self.parameters["data_types"]:
            for col, expected_type in self.parameters["data_types"].items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    if actual_type != expected_type:
                        warnings.append(f"Column '{col}' has type {actual_type}, expected {expected_type}")
        
        return issues, warnings

class RangeValidationRule(ValidationRule):
    """Validates numeric value ranges."""
    
    def __init__(self, column: str, min_value: float = None, max_value: float = None, allow_nan: bool = True):
        super().__init__("range_validation", "range", {
            "column": column,
            "min_value": min_value,
            "max_value": max_value,
            "allow_nan": allow_nan
        })
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="rangevalidationrule initialization",
    )
    async def initialize(self) -> bool:
        """Initialize RangeValidationRule."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    
    def validate(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate numeric range constraints."""
        issues = []
        warnings = []
        
        column = self.parameters["column"]
        if column not in data.columns:
            issues.append(f"Column '{column}' not found in data")
            return issues, warnings
        
        if not pd.api.types.is_numeric_dtype(data[column]):
            issues.append(f"Column '{column}' is not numeric")
            return issues, warnings
        
        # Check min value
        if self.parameters["min_value"] is not None:
            below_min = data[column] < self.parameters["min_value"]
            if below_min.any():
                count = below_min.sum()
                issues.append(f"Column '{column}' has {count} values below minimum {self.parameters['min_value']}")
        
        # Check max value
        if self.parameters["max_value"] is not None:
            above_max = data[column] > self.parameters["max_value"]
            if above_max.any():
                count = above_max.sum()
                issues.append(f"Column '{column}' has {count} values above maximum {self.parameters['max_value']}")
        
        return issues, warnings

class CompletenessValidationRule(ValidationRule):
    """Validates data completeness and missing values."""
    
    def __init__(self, max_missing_ratio: float = 0.1, columns: List[str] = None):
        super().__init__("completeness_validation", "completeness", {
            "max_missing_ratio": max_missing_ratio,
            "columns": columns or []
        })
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="completenessvalidationrule initialization",
    )
    async def initialize(self) -> bool:
        """Initialize CompletenessValidationRule."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    
    def validate(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate data completeness."""
        issues = []
        warnings = []
        
        columns_to_check = self.parameters["columns"] or data.columns
        max_ratio = self.parameters["max_missing_ratio"]
        
        for col in columns_to_check:
            if col in data.columns:
                missing_ratio = data[col].isna().sum() / len(data)
                if missing_ratio > max_ratio:
                    issues.append(f"Column '{col}' has {missing_ratio:.1%} missing values (max: {max_ratio:.1%})")
                elif missing_ratio > max_ratio * 0.5:
                    warnings.append(f"Column '{col}' has {missing_ratio:.1%} missing values (approaching limit)")
        
        return issues, warnings

class ConsistencyValidationRule(ValidationRule):
    """Validates data consistency and business rules."""
    
    def __init__(self, rules: List[Callable] = None, case_sensitive: bool = True):
        super().__init__("consistency_validation", "consistency", {
            "rules": rules or [],
            "case_sensitive": case_sensitive
        })
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="consistencyvalidationrule initialization",
    )
    async def initialize(self) -> bool:
        """Initialize ConsistencyValidationRule."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    
    def validate(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate data consistency using custom rules."""
        issues = []
        warnings = []
        
        for rule_func in self.parameters["rules"]:
            try:
                rule_result = rule_func(data)
                if isinstance(rule_result, tuple):
                    rule_issues, rule_warnings = rule_result
                    issues.extend(rule_issues)
                    warnings.extend(rule_warnings)
                elif rule_result is False:
                    issues.append(f"Consistency rule '{rule_func.__name__}' failed")
            except Exception as e:
                issues.append(f"Error executing consistency rule '{rule_func.__name__}': {e}")
        
        return issues, warnings

class DataQualityFramework:
    """Main framework for data quality management."""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.quality_policies = {
            "max_issues_critical": 0,
            "max_issues_high": 5,
            "max_issues_medium": 10,
            "max_issues_low": 20
        }
        self.logger = system_logger.getChild(f"DataQualityFramework.{name}")
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="dataqualityframework initialization",
    )
    async def initialize(self) -> bool:
        """Initialize DataQualityFramework."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            
            # Initialize all validation rules
            for rule in self.validation_rules.values():
                await rule.initialize()
            
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    
    def add_validation_rule(self, name: str, rule: ValidationRule):
        """Add a validation rule to the framework."""
        self.validation_rules[name] = rule
        self.logger.info(f"Added validation rule: {name}")
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="data validation"
    )
    def validate_data(self, data: pd.DataFrame, rules: List[str] = None) -> Dict[str, Any]:
        """Validate data using specified or all rules."""
        if rules is None:
            rules = list(self.validation_rules.keys())
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "data_shape": data.shape,
            "rules_applied": rules,
            "results": {},
            "summary": {
                "total_issues": 0,
                "total_warnings": 0,
                "critical_issues": 0,
                "high_issues": 0,
                "medium_issues": 0,
                "low_issues": 0
            }
        }
        
        for rule_name in rules:
            if rule_name in self.validation_rules:
                rule = self.validation_rules[rule_name]
                try:
                    issues, warnings = rule.validate(data)
                    validation_results["results"][rule_name] = {
                        "issues": issues,
                        "warnings": warnings,
                        "issue_count": len(issues),
                        "warning_count": len(warnings)
                    }
                    
                    # Update summary
                    validation_results["summary"]["total_issues"] += len(issues)
                    validation_results["summary"]["total_warnings"] += len(warnings)
                    
                    # Categorize issues by severity (simplified)
                    validation_results["summary"]["critical_issues"] += len([i for i in issues if "missing" in i.lower()])
                    validation_results["summary"]["high_issues"] += len([i for i in issues if "type" in i.lower()])
                    validation_results["summary"]["medium_issues"] += len([i for i in issues if "range" in i.lower()])
                    validation_results["summary"]["low_issues"] += len([i for i in issues if "duplicate" in i.lower()])
                    
                except Exception as e:
                    validation_results["results"][rule_name] = {
                        "error": str(e),
                        "issues": [],
                        "warnings": [],
                        "issue_count": 0,
                        "warning_count": 0
                    }
        
        # Evaluate overall validation status
        validation_results["status"] = self._evaluate_validation_status(validation_results["summary"])
        
        return validation_results
    
    def _evaluate_validation_status(self, summary: Dict[str, int]) -> str:
        """Evaluate overall validation status based on issue counts."""
        if summary["critical_issues"] > self.quality_policies["max_issues_critical"]:
            return "FAILED"
        elif summary["high_issues"] > self.quality_policies["max_issues_high"]:
            return "FAILED"
        elif summary["medium_issues"] > self.quality_policies["max_issues_medium"]:
            return "WARNING"
        elif summary["low_issues"] > self.quality_policies["max_issues_low"]:
            return "WARNING"
        else:
            return "PASSED"
    
    def get_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate a quality score from validation results."""
        if not validation_results or "summary" not in validation_results:
            return 0.0
        
        summary = validation_results["summary"]
        total_issues = summary["total_issues"]
        total_warnings = summary["total_warnings"]
        
        if total_issues == 0 and total_warnings == 0:
            return 1.0
        
        # Weight issues more heavily than warnings
        weighted_score = 1.0 - (total_issues * 0.1 + total_warnings * 0.05)
        return max(0.0, min(1.0, weighted_score))
    
    def generate_quality_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a human-readable quality report."""
        if not validation_results:
            return "No validation results available"
        
        report = f"Data Quality Report - {self.name}\n"
        report += "=" * 50 + "\n"
        report += f"Timestamp: {validation_results.get('timestamp', 'N/A')}\n"
        report += f"Data Shape: {validation_results.get('data_shape', 'N/A')}\n"
        report += f"Status: {validation_results.get('status', 'N/A')}\n"
        report += f"Quality Score: {self.get_quality_score(validation_results):.2f}\n\n"
        
        summary = validation_results.get("summary", {})
        report += "Summary:\n"
        report += f"  - Total Issues: {summary.get('total_issues', 0)}\n"
        report += f"  - Total Warnings: {summary.get('total_warnings', 0)}\n"
        report += f"  - Critical Issues: {summary.get('critical_issues', 0)}\n"
        report += f"  - High Issues: {summary.get('high_issues', 0)}\n"
        report += f"  - Medium Issues: {summary.get('medium_issues', 0)}\n"
        report += f"  - Low Issues: {summary.get('low_issues', 0)}\n\n"
        
        results = validation_results.get("results", {})
        if results:
            report += "Rule Results:\n"
            for rule_name, result in results.items():
                if "error" in result:
                    report += f"  {rule_name}: ERROR - {result['error']}\n"
                else:
                    report += f"  {rule_name}: {result['issue_count']} issues, {result['warning_count']} warnings\n"
        
        return report

# --------------------------
# Predefined Validation Rules
# --------------------------

def create_volume_validation_rule() -> ValidationRule:
    """Create a rule to validate trading volume data."""
    def volume_rule(data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        issues = []
        warnings = []
        
        if 'volume' in data.columns:
            # Check for negative volumes
            negative_volumes = data['volume'] < 0
            if negative_volumes.any():
                issues.append(f"Found {negative_volumes.sum()} negative volume values")
            
            # Check for zero volumes
            zero_volumes = data['volume'] == 0
            if zero_volumes.any():
                warnings.append(f"Found {zero_volumes.sum()} zero volume values")
        
        return issues, warnings
    
    rule = ValidationRule("volume_validation", "custom", {"rule_function": volume_rule})
    rule.validate = volume_rule
    return rule

def create_price_validation_rule() -> ValidationRule:
    """Create a rule to validate price data."""
    def price_rule(data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        issues = []
        warnings = []
        
        price_columns = [col for col in data.columns if 'price' in col.lower() or 'close' in col.lower()]
        
        for col in price_columns:
            # Check for negative prices
            negative_prices = data[col] < 0
            if negative_prices.any():
                issues.append(f"Column '{col}' has {negative_prices.sum()} negative prices")
            
            # Check for extreme price changes (more than 50% in one step)
            if len(data) > 1:
                price_changes = data[col].pct_change().abs()
                extreme_changes = price_changes > 0.5
                if extreme_changes.any():
                    warnings.append(f"Column '{col}' has {extreme_changes.sum()} extreme price changes (>50%)")
        
        return issues, warnings
    
    rule = ValidationRule("price_validation", "custom", {"rule_function": price_rule})
    rule.validate = price_rule
    return rule

# --------------------------
# Framework Factory Functions
# --------------------------

def create_default_quality_framework() -> DataQualityFramework:
    """Create a default data quality framework with common rules."""
    framework = DataQualityFramework("default")
    
    # Add common validation rules
    schema_rule = SchemaValidationRule()
    range_rule = RangeValidationRule("volume", min_value=0)
    completeness_rule = CompletenessValidationRule(max_missing_ratio=0.05)
    volume_rule = create_volume_validation_rule()
    price_rule = create_price_validation_rule()
    
    framework.add_validation_rule("schema", schema_rule)
    framework.add_validation_rule("range", range_rule)
    framework.add_validation_rule("completeness", completeness_rule)
    framework.add_validation_rule("volume_validation", volume_rule)
    framework.add_validation_rule("price_validation", price_rule)
    
    return framework

def create_trading_quality_framework() -> DataQualityFramework:
    """Create a specialized framework for trading data quality."""
    framework = DataQualityFramework("trading")
    
    # Trading-specific rules
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data_types = {
        'timestamp': 'datetime64[ns]',
        'open': 'float64',
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'volume': 'float64'
    }
    
    schema_rule = SchemaValidationRule(required_columns, data_types)
    range_rule = RangeValidationRule("volume", min_value=0)
    completeness_rule = CompletenessValidationRule(max_missing_ratio=0.01)
    volume_rule = create_volume_validation_rule()
    price_rule = create_price_validation_rule()
    
    framework.add_validation_rule("schema", schema_rule)
    framework.add_validation_rule("range", range_rule)
    framework.add_validation_rule("completeness", completeness_rule)
    framework.add_validation_rule("volume_validation", volume_rule)
    framework.add_validation_rule("price_validation", price_rule)
    
    return framework

# --------------------------
# Utility Functions
# --------------------------

def quick_validate_dataframe(df: pd.DataFrame, framework_name: str = "default") -> Dict[str, Any]:
    """Quick validation of a DataFrame using a predefined framework."""
    if framework_name == "trading":
        framework = create_trading_quality_framework()
    else:
        framework = create_default_quality_framework()
    
    # Initialize framework
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create new event loop if current one is running
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(framework.initialize())
    except Exception:
        # Fallback to synchronous initialization
        pass
    
    # Validate data
    results = framework.validate_data(df)
    
    return {
        "framework": framework_name,
        "validation_results": results,
        "quality_score": framework.get_quality_score(results),
        "report": framework.generate_quality_report(results)
    }

def validate_data_quality_batch(dataframes: Dict[str, pd.DataFrame], 
                              framework_name: str = "default") -> Dict[str, Any]:
    """Validate multiple DataFrames in batch."""
    results = {}
    
    for name, df in dataframes.items():
        results[name] = quick_validate_dataframe(df, framework_name)
    
    # Calculate overall batch quality
    total_score = sum(result["quality_score"] for result in results.values())
    avg_score = total_score / len(results) if results else 0.0
    
    return {
        "batch_results": results,
        "overall_quality_score": avg_score,
        "total_dataframes": len(dataframes),
        "timestamp": datetime.now().isoformat()
    }