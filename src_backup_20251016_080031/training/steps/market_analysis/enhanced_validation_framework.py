"""
Enhanced Validation Framework for Market Analysis Steps

This module provides comprehensive validation capabilities using the proper
data quality tools from src/utils/data/quality/.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from src.utils.logger import system_logger

class ValidationLevel(Enum):
    """Validation levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Validation result."""
    passed: bool
    message: str
    level: ValidationLevel
    details: Dict[str, Any] = None
    quality_score: float = None
    issues: List[str] = None
    warnings: List[str] = None

class EnhancedValidator:
    """Enhanced validator using comprehensive data quality tools."""
    
    def __init__(self):
        self.logger = system_logger.getChild('EnhancedValidator')
        self.logger.info("🔧 Enhanced Validator initialized with comprehensive data quality tools")
    
    async def validate_data_quality(self, 
                                  data: pd.DataFrame, 
                                  level: ValidationLevel = ValidationLevel.WARNING,
                                  context: str = "general") -> ValidationResult:
        """
        Validate data quality using comprehensive quality assessment tools.
        
        Args:
            data: DataFrame to validate
            level: Validation level threshold
            context: Validation context
            
        Returns:
            Validation result with comprehensive quality assessment
        """
        try:
            # Use comprehensive quality assessment
            from src.utils.data.quality.comprehensive_quality_scorer import get_quality_scorer
            from src.utils.data.quality.data_quality import DataQualityFramework
            from src.utils.data.quality.data_cleaning import get_data_cleaner
            
            # Initialize quality tools
            quality_scorer = get_quality_scorer()
            quality_framework = DataQualityFramework()
            data_cleaner = get_data_cleaner(data_type='klines')
            
            # Perform comprehensive quality assessment
            self.logger.info(f"📊 Performing comprehensive data quality assessment for {context}")
            quality_assessment = quality_scorer.assess_data_quality(
                data,
                context="market_analysis",
                step_name=context,
                data_type="klines"
            )
            
            # Determine if validation passes based on level
            passed = True
            message = f"Data quality assessment completed: {quality_assessment.overall_score:.2f} ({quality_assessment.level.value})"
            
            # Check against validation level
            if level == ValidationLevel.CRITICAL:
                passed = quality_assessment.level.value not in ['critical']
                if not passed:
                    message = f"CRITICAL: Data quality too low for processing: {quality_assessment.overall_score:.2f}"
            elif level == ValidationLevel.ERROR:
                passed = quality_assessment.level.value not in ['critical', 'poor']
                if not passed:
                    message = f"ERROR: Data quality below acceptable threshold: {quality_assessment.overall_score:.2f}"
            elif level == ValidationLevel.WARNING:
                passed = quality_assessment.level.value not in ['critical']
                if not passed:
                    message = f"WARNING: Data quality issues detected: {quality_assessment.overall_score:.2f}"
            
            # Log quality assessment results
            self.logger.info(f"📈 Quality assessment: {quality_assessment.overall_score:.2f} ({quality_assessment.level.value})")
            if quality_assessment.issues:
                self.logger.warning(f"⚠️ Quality issues: {quality_assessment.issues}")
            if quality_assessment.warnings:
                self.logger.info(f"ℹ️ Quality warnings: {quality_assessment.warnings}")
            
            return ValidationResult(
                passed=passed,
                message=message,
                level=level,
                details={
                    'quality_assessment': quality_assessment,
                    'component_scores': quality_assessment.component_scores,
                    'recommendations': quality_assessment.recommendations
                },
                quality_score=quality_assessment.overall_score,
                issues=quality_assessment.issues,
                warnings=quality_assessment.warnings
            )
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Comprehensive quality tools not available, using fallback: {e}")
            # Fallback to basic validation
            return await self._fallback_validation(data, level, context)
        except Exception as e:
            self.logger.error(f"❌ Error in data quality validation: {e}")
            return ValidationResult(
                passed=False,
                message=f"Validation error: {str(e)}",
                level=ValidationLevel.ERROR,
                details={'error': str(e)}
            )
    
    async def _fallback_validation(self, 
                                 data: pd.DataFrame, 
                                 level: ValidationLevel,
                                 context: str) -> ValidationResult:
        """Fallback validation using basic checks."""
        try:
            # Basic validation checks
            issues = []
            warnings = []
            
            # Check if data is empty
            if data.empty:
                issues.append("DataFrame is empty")
            
            # Check for missing values
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) if data.shape[0] > 0 and data.shape[1] > 0 else 0
            if missing_ratio > 0.1:
                warnings.append(f"High missing value ratio: {missing_ratio:.2%}")
            
            # Check for duplicates
            duplicate_ratio = data.duplicated().sum() / len(data) if len(data) > 0 else 0
            if duplicate_ratio > 0.05:
                warnings.append(f"High duplicate ratio: {duplicate_ratio:.2%}")
            
            # Determine if validation passes
            passed = len(issues) == 0
            if level == ValidationLevel.CRITICAL:
                passed = len(issues) == 0 and len(warnings) == 0
            elif level == ValidationLevel.ERROR:
                passed = len(issues) == 0
            
            message = f"Fallback validation: {len(issues)} issues, {len(warnings)} warnings"
            
            return ValidationResult(
                passed=passed,
                message=message,
                level=level,
                details={'fallback': True, 'missing_ratio': missing_ratio, 'duplicate_ratio': duplicate_ratio},
                issues=issues,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Fallback validation error: {str(e)}",
                level=ValidationLevel.ERROR,
                details={'error': str(e)}
            )
    
    async def validate_process_completion(self, 
                                        process_name: str,
                                        expected_outputs: List[str],
                                        output_dir: str,
                                        level: ValidationLevel = ValidationLevel.WARNING) -> ValidationResult:
        """
        Validate that a process has completed successfully by checking expected outputs.
        
        Args:
            process_name: Name of the process
            expected_outputs: List of expected output files
            output_dir: Directory where outputs should be located
            level: Validation level
            
        Returns:
            Validation result
        """
        try:
            from pathlib import Path
            
            missing_files = []
            existing_files = []
            
            for output_file in expected_outputs:
                file_path = Path(output_dir) / output_file
                if file_path.exists():
                    existing_files.append(str(file_path))
                else:
                    missing_files.append(str(file_path))
            
            # Determine if validation passes
            passed = len(missing_files) == 0
            message = f"Process completion validation: {len(existing_files)}/{len(expected_outputs)} files found"
            
            if missing_files:
                message += f", missing: {missing_files}"
            
            return ValidationResult(
                passed=passed,
                message=message,
                level=level,
                details={
                    'process_name': process_name,
                    'expected_outputs': expected_outputs,
                    'existing_files': existing_files,
                    'missing_files': missing_files,
                    'output_dir': output_dir
                }
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Process completion validation error: {str(e)}",
                level=ValidationLevel.ERROR,
                details={'error': str(e)}
            )
    
    async def validate_dataframe_schema(self, 
                                      data: pd.DataFrame,
                                      required_columns: List[str],
                                      level: ValidationLevel = ValidationLevel.WARNING) -> ValidationResult:
        """
        Validate DataFrame schema against required columns.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            level: Validation level
            
        Returns:
            Validation result
        """
        try:
            missing_columns = [col for col in required_columns if col not in data.columns]
            extra_columns = [col for col in data.columns if col not in required_columns]
            
            passed = len(missing_columns) == 0
            message = f"Schema validation: {len(data.columns)} columns, {len(missing_columns)} missing"
            
            if missing_columns:
                message += f", missing: {missing_columns}"
            if extra_columns:
                message += f", extra: {extra_columns}"
            
            return ValidationResult(
                passed=passed,
                message=message,
                level=level,
                details={
                    'required_columns': required_columns,
                    'actual_columns': list(data.columns),
                    'missing_columns': missing_columns,
                    'extra_columns': extra_columns
                }
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Schema validation error: {str(e)}",
                level=ValidationLevel.ERROR,
                details={'error': str(e)}
            )