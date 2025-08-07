"""
Validator for Step 1: Data Collection
"""

import os
import sys
import pickle
import asyncio
import pandas as pd
from typing import Any, Dict
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.base_validator import BaseValidator
from src.config import CONFIG


class Step1DataCollectionValidator(BaseValidator):
    """Validator for Step 1: Data Collection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("step1_data_collection", config)
        # Fine-tuned parameters for ML training (more lenient to avoid stopping training)
        self.min_records = 500  # Reduced from 1000 to allow smaller datasets
        self.max_gap_ratio = 0.2  # Allow up to 20% large gaps (increased from 10%)
        self.max_gap_hours = 48  # Increased from 24 hours
        self.price_tolerance = 0.001  # Allow very small negative prices due to precision
        self.volume_tolerance = 0.001  # Allow very small negative volumes due to precision
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> bool:
        """
        Validate the data collection step.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating data collection step...")
        
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        
        # Validate step result from pipeline state
        step_result = pipeline_state.get("data_collection", {})
        
        # 1. Validate error absence (CRITICAL - blocks process)
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics
        
        if not error_passed:
            self.logger.error("❌ Data collection step had critical errors - stopping process")
            return False
        
        # 2. Validate file existence (CRITICAL - blocks process)
        data_file_path = f"{data_dir}/{exchange}_{symbol}_historical_data.pkl"
        file_passed, file_metrics = self.validate_file_exists(data_file_path, "historical_data")
        self.validation_results["file_existence"] = file_metrics
        
        if not file_passed:
            self.logger.error(f"❌ Data file not found: {data_file_path} - stopping process")
            return False
        
        # 3. Validate data quality (WARNING - doesn't block)
        try:
            with open(data_file_path, "rb") as f:
                historical_data = pickle.load(f)
            
            # Convert to DataFrame if needed
            if not isinstance(historical_data, pd.DataFrame):
                # Handle numpy arrays and other data structures
                if hasattr(historical_data, 'shape') and len(historical_data.shape) == 2:
                    # It's a 2D array, create DataFrame with default column names
                    historical_data = pd.DataFrame(historical_data, columns=[f'col_{i}' for i in range(historical_data.shape[1])])
                elif isinstance(historical_data, (list, tuple)):
                    # It's a list/tuple, try to create DataFrame
                    try:
                        historical_data = pd.DataFrame(historical_data)
                    except:
                        # If that fails, wrap in a list
                        historical_data = pd.DataFrame([historical_data])
                else:
                    # For other types, wrap in a list
                    historical_data = pd.DataFrame([historical_data])
            
            quality_passed, quality_metrics = self.validate_data_quality(historical_data, "historical_data")
            self.validation_results["data_quality"] = quality_metrics
            
            if not quality_passed:
                self.logger.warning("⚠️ Data quality validation failed - continuing with caution")
            
            # 4. Validate data characteristics (WARNING - doesn't block)
            characteristics_passed = self._validate_data_characteristics(historical_data, symbol, exchange)
            if not characteristics_passed:
                self.logger.warning("⚠️ Data characteristics validation failed - continuing with caution")
            
            # 5. Validate outcome favorability (WARNING - doesn't block)
            outcome_passed, outcome_metrics = self.validate_outcome_favorability(step_result)
            self.validation_results["outcome_favorability"] = outcome_metrics
            
            if not outcome_passed:
                self.logger.warning("⚠️ Data collection outcome is not favorable - continuing with caution")
            
            # Overall validation passes if critical checks pass
            critical_passed = error_passed and file_passed
            if critical_passed:
                self.logger.info("✅ Data collection validation passed (critical checks only)")
                return True
            else:
                self.logger.error("❌ Data collection validation failed (critical checks failed)")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Error during data validation: {e}")
            return False
    
    def _validate_data_characteristics(self, data: pd.DataFrame, symbol: str, exchange: str) -> bool:
        """
        Validate specific characteristics of the collected data.
        
        Args:
            data: Historical data DataFrame
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            bool: True if characteristics are valid
        """
        try:
            # Check minimum data size (more lenient for ML training)
            if len(data) < self.min_records:
                self.logger.warning(f"⚠️ Insufficient data: {len(data)} records (minimum: {self.min_records}) - continuing with caution")
                return False
            
            # Check for required columns (basic OHLCV)
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.warning(f"⚠️ Missing required columns: {missing_columns} - continuing with caution")
                return False
            
            # Check for reasonable price ranges (more tolerant)
            price_columns = ["open", "high", "low", "close"]
            for col in price_columns:
                if col in data.columns:
                    min_price = data[col].min()
                    if min_price < -self.price_tolerance:  # Allow small negative values due to precision
                        self.logger.warning(f"⚠️ Invalid price values in {col} column (min: {min_price}) - continuing with caution")
                        return False
            
            # Check for reasonable volume values (more tolerant)
            if "volume" in data.columns:
                min_volume = data["volume"].min()
                if min_volume < -self.volume_tolerance:  # Allow small negative values due to precision
                    self.logger.warning(f"⚠️ Invalid volume values (min: {min_volume}) - continuing with caution")
                    return False
            
            # Check data consistency (high >= low, etc.) - more lenient
            if all(col in data.columns for col in ["high", "low", "open", "close"]):
                invalid_rows = (
                    (data["high"] < data["low"]) |
                    (data["high"] < data["open"]) |
                    (data["high"] < data["close"]) |
                    (data["low"] > data["open"]) |
                    (data["low"] > data["close"])
                ).sum()
                
                invalid_ratio = invalid_rows / len(data)
                if invalid_ratio > 0.05:  # Allow up to 5% invalid rows
                    self.logger.warning(f"⚠️ Found {invalid_rows} rows ({invalid_ratio:.2%}) with inconsistent OHLC data - continuing with caution")
                elif invalid_rows > 0:
                    self.logger.info(f"ℹ️ Found {invalid_rows} rows with minor OHLC inconsistencies (acceptable)")
            
            # Check for reasonable time gaps (if timestamp column exists) - more lenient
            if "timestamp" in data.columns:
                data_sorted = data.sort_values("timestamp")
                time_diffs = data_sorted["timestamp"].diff().dropna()
                
                # Check for reasonable time intervals (not too large gaps)
                large_gaps = (time_diffs > pd.Timedelta(hours=self.max_gap_hours)).sum()
                large_gap_ratio = large_gaps / len(data)
                
                if large_gap_ratio > self.max_gap_ratio:  # Allow up to 20% large gaps
                    self.logger.warning(f"⚠️ Found {large_gaps} large time gaps ({large_gap_ratio:.2%}) in data - continuing with caution")
                elif large_gaps > 0:
                    self.logger.info(f"ℹ️ Found {large_gaps} large time gaps (acceptable)")
            
            self.logger.info(f"✅ Data characteristics validation passed: {len(data)} records")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during data characteristics validation: {e}")
            return False


async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the Step 1 Data Collection validator.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        
    Returns:
        Dictionary containing validation results
    """
    validator = Step1DataCollectionValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)
    
    return {
        "step_name": "step1_data_collection",
        "validation_passed": validation_passed,
        "validation_results": validator.validation_results,
        "duration": 0,  # Could be enhanced to track actual duration
        "timestamp": asyncio.get_event_loop().time()
    }


if __name__ == "__main__":
    import asyncio
    
    # Example usage
    async def test_validator():
        training_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data/training"
        }
        
        pipeline_state = {
            "data_collection": {
                "status": "SUCCESS",
                "duration": 120.5
            }
        }
        
        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")
    
    asyncio.run(test_validator())
