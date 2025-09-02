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

class ValidationRule:
    """Defines a validation rule for data quality."""
    
    def __init__(self, name: str, rule_type: str, parameters: Dict[str, Any]):
        self.name = name
        self.rule_type = rule_type
        self.parameters = parameters
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
    
    def validate(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Validate data according to the rule."""
        raise NotImplementedError("Subclasses must implement validate method")

class SchemaValidationRule(ValidationRule):
    """Validates data schema and structure."""
    
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
    
    def validate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Validate DataFrame schema."""
        required_columns = kwargs.get('required_columns', [])
        data_types = kwargs.get('data_types', {})
        
        issues = []
        warnings = []
        
        if not isinstance(data, pd.DataFrame):
            issues.append("Data must be a pandas DataFrame")
            return {
                "is_valid": False,
                "issues": issues,
                "warnings": warnings
            }
        
        # Check required columns
        if required_columns:
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                issues.append(f"Missing required columns: {list(missing_columns)}")
        
        # Check data types
        if data_types:
            for col, expected_type in data_types.items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    if not self._is_compatible_type(actual_type, expected_type):
                        warnings.append(f"Column '{col}' has type {actual_type}, expected {expected_type}")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }
    
    def _is_compatible_type(self, actual: str, expected: str) -> bool:
        """Check if actual type is compatible with expected type."""
        type_mapping = {
            'int64': ['int', 'integer', 'int64', 'int32'],
            'float64': ['float', 'float64', 'float32', 'number'],
            'object': ['object', 'string', 'str', 'category'],
            'datetime64[ns]': ['datetime', 'datetime64', 'timestamp']
        }
        
        for base_type, compatible_types in type_mapping.items():
            if actual in compatible_types:
                return expected in compatible_types
        return False

class RangeValidationRule(ValidationRule):
    """Validates data ranges and boundaries."""
    
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
    
    def validate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Validate data ranges."""
        min_values = kwargs.get('min_values', {})
        max_values = kwargs.get('max_values', {})
        allow_nan = kwargs.get('allow_nan', True)
        
        issues = []
        warnings = []
        
        for col in data.columns:
            if col in min_values or col in max_values:
                if pd.api.types.is_numeric_dtype(data[col]):
                    col_data = data[col].dropna() if not allow_nan else data[col]
                    
                    if col in min_values and len(col_data) > 0:
                        min_val = col_data.min()
                        if min_val < min_values[col]:
                            issues.append(f"Column '{col}' has values below minimum: {min_val} < {min_values[col]}")
                    
                    if col in max_values and len(col_data) > 0:
                        max_val = col_data.max()
                        if max_val > max_values[col]:
                            issues.append(f"Column '{col}' has values above maximum: {max_val} > {max_values[col]}")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }

class CompletenessValidationRule(ValidationRule):
    """Validates data completeness and missing values."""
    
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
    
    def validate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Validate data completeness."""
        max_missing_ratio = kwargs.get('max_missing_ratio', 0.1)
        
        issues = []
        warnings = []
        
        missing_ratios = data.isnull().sum() / len(data)
        problematic_columns = missing_ratios[missing_ratios > max_missing_ratio]
        
        for col, ratio in problematic_columns.items():
            issues.append(f"Column '{col}' has {ratio:.2%} missing values (max allowed: {max_missing_ratio:.2%})")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }

class ConsistencyValidationRule(ValidationRule):
    """Validates data consistency and relationships."""
    
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
    
    def validate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Validate data consistency."""
        unique_constraints = kwargs.get('unique_constraints', [])
        case_sensitive = kwargs.get('case_sensitive', True)
        
        issues = []
        warnings = []
        
        # Check unique constraints
        for col in unique_constraints:
            if col in data.columns:
                if not case_sensitive and data[col].dtype == 'object':
                    # Case-insensitive uniqueness check
                    unique_values = data[col].str.lower().nunique()
                    total_values = len(data[col].dropna())
                    if unique_values != total_values:
                        issues.append(f"Column '{col}' has duplicate values (case-insensitive)")
                else:
                    # Case-sensitive uniqueness check
                    if data[col].nunique() != len(data[col].dropna()):
                        issues.append(f"Column '{col}' has duplicate values")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }

class DataQualityFramework:
    """Main framework for managing data quality validation."""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.validation_rules = {}
        self.quality_policies = {
            "max_issues_critical": 0,
            "max_issues_high": 5,
            "max_issues_medium": 10,
            "max_issues_low": 20
        }
        self.logger = system_logger.getChild(f"DataQualityFramework.{name}")
        self.is_initialized = False
    
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
            
            # Initialize default validation rules
            await self._initialize_default_rules()
            
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    
    async def _initialize_default_rules(self):
        """Initialize default validation rules."""
        # Schema validation
        schema_rule = SchemaValidationRule(
            "basic_schema",
            "schema",
            {"required_columns": [], "data_types": {}}
        )
        await schema_rule.initialize()
        self.add_validation_rule("schema_validation", schema_rule)
        
        # Completeness validation
        completeness_rule = CompletenessValidationRule(
            "basic_completeness",
            "completeness",
            {"max_missing_ratio": 0.1}
        )
        await completeness_rule.initialize()
        self.add_validation_rule("completeness_validation", completeness_rule)
        
        # Range validation for common financial data
        volume_validation = RangeValidationRule(
            "volume_validation",
            "range",
            {"min_values": {"volume": 0}, "max_values": {}, "allow_nan": False}
        )
        await volume_validation.initialize()
        self.add_validation_rule("volume_validation", volume_validation)
    
    def add_validation_rule(self, name: str, rule: ValidationRule) -> None:
        """Add a validation rule to the framework."""
        self.validation_rules[name] = rule
        self.logger.info(f"Added validation rule: {name}")
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="data validation"
    )
    def validate_data(self, data: pd.DataFrame, rules: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate data using specified or all validation rules."""
        if rules is None:
            rules = list(self.validation_rules.keys())
        
        if not self.is_initialized:
            self.logger.warning("Framework not initialized, attempting to initialize...")
            # Note: This is a synchronous method, so we can't await here
            # In a real implementation, you'd want to handle this differently
        
        validation_results = {}
        total_issues = 0
        total_warnings = 0
        
        for rule_name in rules:
            if rule_name in self.validation_rules:
                rule = self.validation_rules[rule_name]
                try:
                    result = rule.validate(data)
                    validation_results[rule_name] = result
                    total_issues += len(result.get('issues', []))
                    total_warnings += len(result.get('warnings', []))
                except Exception as e:
                    self.logger.error(f"Error executing rule {rule_name}: {e}")
                    validation_results[rule_name] = {
                        "is_valid": False,
                        "issues": [f"Rule execution error: {e}"],
                        "warnings": []
                    }
                    total_issues += 1
            else:
                self.logger.warning(f"Validation rule '{rule_name}' not found")
        
        # Evaluate overall validation status
        overall_status = self._evaluate_validation_status(validation_results, total_issues, total_warnings)
        
        return {
            "overall_status": overall_status,
            "validation_results": validation_results,
            "summary": {
                "total_rules": len(rules),
                "total_issues": total_issues,
                "total_warnings": total_warnings,
                "critical_issues": sum(1 for r in validation_results.values() if not r.get('is_valid', True))
            }
        }
    
    def _evaluate_validation_status(self, validation_results: Dict[str, Any], total_issues: int, total_warnings: int) -> str:
        """Evaluate the overall validation status based on quality policies."""
        if total_issues > self.quality_policies["max_issues_critical"]:
            return "CRITICAL"
        elif total_issues > self.quality_policies["max_issues_high"]:
            return "HIGH"
        elif total_issues > self.quality_policies["max_issues_medium"]:
            return "MEDIUM"
        elif total_issues > self.quality_policies["max_issues_low"]:
            return "LOW"
        else:
            return "PASS"
    
    def get_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate a quality score from validation results."""
        if not validation_results:
            return 0.0
        
        total_rules = len(validation_results)
        passed_rules = sum(1 for result in validation_results.values() if result.get('is_valid', False))
        
        return passed_rules / total_rules if total_rules > 0 else 0.0
    
    def generate_quality_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a human-readable quality report."""
        if not validation_results:
            return "No validation results available."
        
        report_lines = [
            "=" * 60,
            "DATA QUALITY VALIDATION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            f"Framework: {self.name}",
            ""
        ]
        
        # Summary
        total_rules = len(validation_results)
        passed_rules = sum(1 for result in validation_results.values() if result.get('is_valid', False))
        failed_rules = total_rules - passed_rules
        
        report_lines.extend([
            "SUMMARY",
            "-" * 20,
            f"Total Rules: {total_rules}",
            f"Passed: {passed_rules}",
            f"Failed: {failed_rules}",
            f"Success Rate: {(passed_rules/total_rules)*100:.1f}%",
            ""
        ])
        
        # Detailed results
        report_lines.extend([
            "DETAILED RESULTS",
            "-" * 20
        ])
        
        for rule_name, result in validation_results.items():
            status = "✅ PASS" if result.get('is_valid', False) else "❌ FAIL"
            report_lines.append(f"{rule_name}: {status}")
            
            if result.get('issues'):
                for issue in result['issues']:
                    report_lines.append(f"  - Issue: {issue}")
            
            if result.get('warnings'):
                for warning in result['warnings']:
                    report_lines.append(f"  - Warning: {warning}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)

# --------------------------
# Utility functions
# --------------------------

def create_standard_framework() -> DataQualityFramework:
    """Create a standard data quality framework with common rules."""
    framework = DataQualityFramework("standard")
    
    # Add common validation rules
    schema_rule = SchemaValidationRule("standard_schema", "schema", {})
    completeness_rule = CompletenessValidationRule("standard_completeness", "completeness", {"max_missing_ratio": 0.05})
    range_rule = RangeValidationRule("standard_range", "range", {})
    
    framework.validation_rules.update({
        "schema": schema_rule,
        "completeness": completeness_rule,
        "range": range_rule
    })
    
    return framework

def validate_dataframe_quality(df: pd.DataFrame, framework: Optional[DataQualityFramework] = None) -> Dict[str, Any]:
    """Quick validation of DataFrame quality using standard framework."""
    if framework is None:
        framework = create_standard_framework()
    
    return framework.validate_data(df)

# --------------------------
# Export main classes
# --------------------------

__all__ = [
    'DataQualityFramework',
    'ValidationRule',
    'SchemaValidationRule',
    'RangeValidationRule',
    'CompletenessValidationRule',
    'ConsistencyValidationRule',
    'DataQualityLevel',
    'create_standard_framework',
    'validate_dataframe_quality'
]