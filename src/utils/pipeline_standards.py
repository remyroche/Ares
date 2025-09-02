"""
Pipeline Standards Module

This module defines the standards and configurations for the Ares trading system pipeline,
including data quality levels, validation rules, and pipeline configurations.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.logger import system_logger

class DataQualityLevel(Enum):
    """Enumeration for data quality levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: DataQualityLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    column: Optional[str] = None
    row_count: Optional[int] = None

@dataclass
class ValidationResult:
    """Represents a validation result."""
    is_valid: bool
    issues: List[ValidationIssue]
    summary: Dict[str, Any]
    timestamp: float

class PipelineStandards:
    """Pipeline standards and configuration manager."""
    
    def __init__(self):
        """Initialize the PipelineStandards."""
        self.logger = system_logger.getChild("PipelineStandards")
        self.is_initialized = False
        self.standards = {}
        self.configurations = {}
        
    async def initialize(self) -> bool:
        """Initialize the PipelineStandards."""
        try:
            self.logger.info("🚀 Initializing PipelineStandards...")
            
            # Load pipeline standards
            await self._load_standards()
            
            # Load pipeline configurations
            await self._load_configurations()
            
            self.is_initialized = True
            self.logger.info("✅ PipelineStandards initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing PipelineStandards: {e}")
            return False
    
    async def _load_standards(self) -> None:
        """Load pipeline standards."""
        try:
            self.standards = {
                "data_quality": {
                    "min_rows": 100,
                    "max_null_ratio": 0.3,
                    "max_duplicate_ratio": 0.05,
                    "required_columns": ["timestamp"],
                    "data_types": {
                        "timestamp": ["int64", "datetime64[ns]"],
                        "price": ["float64"],
                        "volume": ["float64", "int64"]
                    }
                },
                "file_formats": {
                    "preferred": ["parquet", "csv"],
                    "supported": ["parquet", "csv", "json", "pkl"],
                    "max_file_size_mb": 1000
                },
                "validation": {
                    "levels": ["BASIC", "STANDARD", "COMPREHENSIVE", "CRITICAL"],
                    "default_level": "STANDARD",
                    "auto_fix": False
                }
            }
            
            self.logger.info(f"Loaded {len(self.standards)} pipeline standards")
        except Exception as e:
            self.logger.error(f"Error loading pipeline standards: {e}")
    
    async def _load_configurations(self) -> None:
        """Load pipeline configurations."""
        try:
            self.configurations = {
                "steps": {
                    "step01_data_collection": {
                        "description": "Data collection from exchanges",
                        "required_inputs": ["symbol", "exchange", "timeframe"],
                        "expected_outputs": ["klines_data", "trades_data"],
                        "validation_level": "STANDARD"
                    },
                    "step01_5_data_preprocessing": {
                        "description": "Data preprocessing and cleaning",
                        "required_inputs": ["raw_data"],
                        "expected_outputs": ["cleaned_data", "preprocessing_config"],
                        "validation_level": "COMPREHENSIVE"
                    },
                    "step02_feature_engineering": {
                        "description": "Feature engineering and selection",
                        "required_inputs": ["cleaned_data"],
                        "expected_outputs": ["features", "feature_config"],
                        "validation_level": "COMPREHENSIVE"
                    },
                    "step03_model_training": {
                        "description": "Model training and optimization",
                        "required_inputs": ["features", "labels"],
                        "expected_outputs": ["model", "training_metrics"],
                        "validation_level": "CRITICAL"
                    },
                    "step04_model_evaluation": {
                        "description": "Model evaluation and validation",
                        "required_inputs": ["model", "test_data"],
                        "expected_outputs": ["evaluation_metrics", "validation_report"],
                        "validation_level": "CRITICAL"
                    }
                },
                "data_quality": {
                    "thresholds": {
                        "missing_data": 0.1,
                        "outlier_ratio": 0.05,
                        "data_consistency": 0.95
                    },
                    "checks": {
                        "duplicate_detection": True,
                        "outlier_detection": True,
                        "data_type_validation": True,
                        "range_validation": True
                    }
                }
            }
            
            self.logger.info(f"Loaded {len(self.configurations)} pipeline configurations")
        except Exception as e:
            self.logger.error(f"Error loading pipeline configurations: {e}")
    
    def get_step_config(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific step."""
        try:
            return self.configurations.get("steps", {}).get(step_name)
        except Exception as e:
            self.logger.error(f"Error getting step config for {step_name}: {e}")
            return None
    
    def get_data_quality_thresholds(self) -> Dict[str, Any]:
        """Get data quality thresholds."""
        try:
            return self.configurations.get("data_quality", {}).get("thresholds", {})
        except Exception as e:
            self.logger.error(f"Error getting data quality thresholds: {e}")
            return {}
    
    def get_validation_level(self, step_name: str) -> str:
        """Get validation level for a specific step."""
        try:
            step_config = self.get_step_config(step_name)
            if step_config:
                return step_config.get("validation_level", "STANDARD")
            return "STANDARD"
        except Exception as e:
            self.logger.error(f"Error getting validation level for {step_name}: {e}")
            return "STANDARD"
    
    def validate_step_inputs(self, step_name: str, inputs: Dict[str, Any]) -> ValidationResult:
        """Validate inputs for a specific step."""
        try:
            step_config = self.get_step_config(step_name)
            if not step_config:
                return ValidationResult(
                    is_valid=False,
                    issues=[ValidationIssue(
                        severity=DataQualityLevel.ERROR,
                        message=f"No configuration found for step: {step_name}"
                    )],
                    summary={"step_name": step_name, "validation_type": "inputs"},
                    timestamp=asyncio.get_event_loop().time()
                )
            
            required_inputs = step_config.get("required_inputs", [])
            issues = []
            
            for required_input in required_inputs:
                if required_input not in inputs:
                    issues.append(ValidationIssue(
                        severity=DataQualityLevel.ERROR,
                        message=f"Missing required input: {required_input}",
                        details={"required_input": required_input}
                    ))
            
            is_valid = len([i for i in issues if i.severity == DataQualityLevel.ERROR]) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                issues=issues,
                summary={
                    "step_name": step_name,
                    "validation_type": "inputs",
                    "required_inputs": required_inputs,
                    "provided_inputs": list(inputs.keys())
                },
                timestamp=asyncio.get_event_loop().time()
            )
            
        except Exception as e:
            self.logger.error(f"Error validating inputs for {step_name}: {e}")
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    severity=DataQualityLevel.ERROR,
                    message=f"Input validation error: {str(e)}"
                )],
                summary={"step_name": step_name, "validation_type": "inputs"},
                timestamp=asyncio.get_event_loop().time()
            )
    
    def validate_step_outputs(self, step_name: str, outputs: Dict[str, Any]) -> ValidationResult:
        """Validate outputs for a specific step."""
        try:
            step_config = self.get_step_config(step_name)
            if not step_config:
                return ValidationResult(
                    is_valid=False,
                    issues=[ValidationIssue(
                        severity=DataQualityLevel.ERROR,
                        message=f"No configuration found for step: {step_name}"
                    )],
                    summary={"step_name": step_name, "validation_type": "outputs"},
                    timestamp=asyncio.get_event_loop().time()
                )
            
            expected_outputs = step_config.get("expected_outputs", [])
            issues = []
            
            for expected_output in expected_outputs:
                if expected_output not in outputs:
                    issues.append(ValidationIssue(
                        severity=DataQualityLevel.WARNING,
                        message=f"Missing expected output: {expected_output}",
                        details={"expected_output": expected_output}
                    ))
            
            is_valid = len([i for i in issues if i.severity == DataQualityLevel.ERROR]) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                issues=issues,
                summary={
                    "step_name": step_name,
                    "validation_type": "outputs",
                    "expected_outputs": expected_outputs,
                    "provided_outputs": list(outputs.keys())
                },
                timestamp=asyncio.get_event_loop().time()
            )
            
        except Exception as e:
            self.logger.error(f"Error validating outputs for {step_name}: {e}")
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    severity=DataQualityLevel.ERROR,
                    message=f"Output validation error: {str(e)}"
                )],
                summary={"step_name": step_name, "validation_type": "outputs"},
                timestamp=asyncio.get_event_loop().time()
            )
    
    def get_all_step_names(self) -> List[str]:
        """Get all available step names."""
        try:
            return list(self.configurations.get("steps", {}).keys())
        except Exception as e:
            self.logger.error(f"Error getting step names: {e}")
            return []
    
    def get_standards_summary(self) -> Dict[str, Any]:
        """Get a summary of all pipeline standards."""
        try:
            return {
                "total_standards": len(self.standards),
                "total_configurations": len(self.configurations),
                "total_steps": len(self.configurations.get("steps", {})),
                "data_quality_levels": [level.value for level in DataQualityLevel],
                "supported_file_formats": self.standards.get("file_formats", {}).get("supported", []),
                "validation_levels": self.standards.get("validation", {}).get("levels", [])
            }
        except Exception as e:
            self.logger.error(f"Error getting standards summary: {e}")
            return {}
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.logger.info("🧹 Cleaning up PipelineStandards...")
            self.standards.clear()
            self.configurations.clear()
            self.is_initialized = False
            self.logger.info("✅ PipelineStandards cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# Global pipeline standards instance
pipeline_standards = PipelineStandards()

# Convenience function for creating standards instance
async def create_pipeline_standards() -> PipelineStandards:
    """Create and initialize a PipelineStandards instance."""
    standards = PipelineStandards()
    success = await standards.initialize()
    if not success:
        raise RuntimeError("Failed to initialize PipelineStandards")
    return standards