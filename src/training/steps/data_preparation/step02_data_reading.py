"""Step 2: Data Reading - Refactored to use BaseStep.

This module handles reading the unified data from step1_5 and performs comprehensive
data quality validation before proceeding to HMM regime discovery.
"""

from typing import Any, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from src.training.base_step import BaseStep
from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.utils.common_operations import (
    safe_read_parquet, validate_dataframe_schema, validate_data_quality
)


class DataReadingStep(BaseStep):
    """Step 2: Data Reading and Validation using standardized base class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data reading step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "02", "data_reading")
        
        # Step-specific configuration
        self.data_quality_thresholds = config.get("data_quality_thresholds", {
            "min_rows": 1000,
            "max_missing_pct": 0.05,
            "min_unique_timestamps": 500
        })
        
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        self.logger.info("✅ Data reading step initialized")
    
    def validate_inputs(
        self, 
        training_input: Dict[str, Any], 
        pipeline_state: Dict[str, Any]
    ) -> Tuple[bool, list]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check if unified data path exists from step 1.5
        if "unified_data_path" not in pipeline_state:
            # Fall back to raw market data from step 1
            if "raw_market_data" not in pipeline_state:
                errors.append("No unified_data_path or raw_market_data in pipeline state")
            else:
                self.logger.info("Using raw_market_data as fallback")
        
        # Validate data file exists
        data_path = pipeline_state.get("unified_data_path") or pipeline_state.get("raw_market_data")
        if data_path and not Path(data_path).exists():
            errors.append(f"Data file does not exist: {data_path}")
        
        return len(errors) == 0, errors
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={"success": False},
        context="data reading execution"
    )
    async def execute_logic(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute data reading and validation logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state
        """
        # Get data path
        data_path = pipeline_state.get("unified_data_path") or pipeline_state.get("raw_market_data")
        self.logger.info(f"📖 Reading data from: {data_path}")
        
        # Read data
        try:
            data = safe_read_parquet(data_path)
            if data is None:
                raise ValueError(f"Failed to read data from {data_path}")
            
            self.logger.info(f"✅ Loaded {len(data)} rows with {len(data.columns)} columns")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to read data: {e}")
            raise
        
        # Perform data quality validation
        validation_results = self._validate_data_quality(data)
        
        # Update pipeline state
        pipeline_state["validated_data"] = data
        pipeline_state["data_validation_results"] = validation_results
        pipeline_state["data_info"] = {
            "shape": data.shape,
            "columns": list(data.columns),
            "index_type": str(type(data.index)),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
            "date_range": {
                "start": str(data.index.min()) if hasattr(data.index, 'min') else None,
                "end": str(data.index.max()) if hasattr(data.index, 'max') else None
            }
        }
        
        # Log validation summary
        self._log_validation_summary(validation_results)
        
        # Store in-memory for next steps
        pipeline_state["dataframe"] = data
        
        return pipeline_state
    
    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check if validated data exists
        if "validated_data" not in pipeline_state and "dataframe" not in pipeline_state:
            errors.append("No validated data in pipeline state")
            return False, errors
        
        # Check data validation results
        if "data_validation_results" not in pipeline_state:
            errors.append("No data validation results in pipeline state")
        else:
            validation_results = pipeline_state["data_validation_results"]
            
            # Check for critical issues
            if not validation_results.get("has_required_columns", True):
                errors.append("Missing required columns")
            
            if validation_results.get("missing_data_pct", 100) > self.data_quality_thresholds["max_missing_pct"] * 100:
                errors.append(f"Too much missing data: {validation_results.get('missing_data_pct', 0):.2f}%")
            
            if validation_results.get("total_rows", 0) < self.data_quality_thresholds["min_rows"]:
                errors.append(f"Insufficient data rows: {validation_results.get('total_rows', 0)}")
        
        return len(errors) == 0, errors
    
    def _validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality validation.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Validation results dictionary
        """
        results = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "has_required_columns": True,
            "missing_data_pct": 0,
            "duplicate_rows": 0,
            "data_quality_score": 100,
            "issues": []
        }
        
        # Check required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            results["has_required_columns"] = False
            results["issues"].append(f"Missing required columns: {missing_columns}")
            results["data_quality_score"] -= 20
        
        # Check for missing data
        missing_count = data.isnull().sum().sum()
        total_cells = data.shape[0] * data.shape[1]
        if total_cells > 0:
            results["missing_data_pct"] = (missing_count / total_cells) * 100
            if results["missing_data_pct"] > 0:
                results["issues"].append(f"Missing data: {results['missing_data_pct']:.2f}%")
                results["data_quality_score"] -= min(20, results["missing_data_pct"] * 4)
        
        # Check for duplicates
        if hasattr(data.index, 'duplicated'):
            duplicate_count = data.index.duplicated().sum()
            if duplicate_count > 0:
                results["duplicate_rows"] = duplicate_count
                results["issues"].append(f"Duplicate timestamps: {duplicate_count}")
                results["data_quality_score"] -= 10
        
        # Check data consistency
        if all(col in data.columns for col in ["high", "low", "open", "close"]):
            # High should be >= max(open, close, low)
            invalid_high = (data["high"] < data[["open", "close", "low"]].max(axis=1)).sum()
            if invalid_high > 0:
                results["issues"].append(f"Invalid high values: {invalid_high} rows")
                results["data_quality_score"] -= 5
            
            # Low should be <= min(open, close, high)
            invalid_low = (data["low"] > data[["open", "close", "high"]].min(axis=1)).sum()
            if invalid_low > 0:
                results["issues"].append(f"Invalid low values: {invalid_low} rows")
                results["data_quality_score"] -= 5
        
        # Check for zero or negative prices
        price_columns = ["open", "high", "low", "close"]
        for col in price_columns:
            if col in data.columns:
                invalid_prices = (data[col] <= 0).sum()
                if invalid_prices > 0:
                    results["issues"].append(f"Invalid {col} prices: {invalid_prices} rows")
                    results["data_quality_score"] -= 5
        
        # Check for zero volume
        if "volume" in data.columns:
            zero_volume = (data["volume"] == 0).sum()
            if zero_volume > 0:
                zero_volume_pct = (zero_volume / len(data)) * 100
                if zero_volume_pct > 10:
                    results["issues"].append(f"Zero volume: {zero_volume} rows ({zero_volume_pct:.1f}%)")
                    results["data_quality_score"] -= min(10, zero_volume_pct)
        
        # Ensure score doesn't go below 0
        results["data_quality_score"] = max(0, results["data_quality_score"])
        
        return results
    
    def _log_validation_summary(self, validation_results: Dict[str, Any]) -> None:
        """Log a summary of validation results.
        
        Args:
            validation_results: Validation results dictionary
        """
        self.logger.info("📊 Data Validation Summary:")
        self.logger.info(f"   - Total rows: {validation_results['total_rows']:,}")
        self.logger.info(f"   - Total columns: {validation_results['total_columns']}")
        self.logger.info(f"   - Missing data: {validation_results['missing_data_pct']:.2f}%")
        self.logger.info(f"   - Duplicate rows: {validation_results['duplicate_rows']}")
        self.logger.info(f"   - Quality score: {validation_results['data_quality_score']}/100")
        
        if validation_results["issues"]:
            self.logger.warning("⚠️ Data quality issues found:")
            for issue in validation_results["issues"]:
                self.logger.warning(f"   - {issue}")
        else:
            self.logger.info("✅ No data quality issues found")
    
    def get_required_inputs(self) -> list:
        """Get list of required inputs for this step."""
        return ["unified_data_path or raw_market_data"]
    
    def get_produced_outputs(self) -> list:
        """Get list of outputs produced by this step."""
        return ["validated_data", "data_validation_results", "data_info", "dataframe"]
    
    def get_dependencies(self) -> list:
        """Get list of step dependencies."""
        return ["01_data_collection", "01_5_data_converter"]  # Can work with either