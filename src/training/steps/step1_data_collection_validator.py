"""
Validator for Step 1: Data Collection
"""

import os
import sys
import pickle
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
        
        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics
        
        if not error_passed:
            self.logger.error("❌ Data collection step had errors")
            return False
        
        # 2. Validate file existence
        data_file_path = f"{data_dir}/{exchange}_{symbol}_historical_data.pkl"
        file_passed, file_metrics = self.validate_file_exists(data_file_path, "historical_data")
        self.validation_results["file_existence"] = file_metrics
        
        if not file_passed:
            self.logger.error(f"❌ Data file not found: {data_file_path}")
            return False
        
        # 3. Validate data quality
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
                self.logger.error("❌ Data quality validation failed")
                return False
            
            # 4. Validate data characteristics
            characteristics_passed = self._validate_data_characteristics(historical_data, symbol, exchange)
            if not characteristics_passed:
                self.logger.error("❌ Data characteristics validation failed")
                return False
            
            # 5. Validate outcome favorability
            outcome_passed, outcome_metrics = self.validate_outcome_favorability(step_result)
            self.validation_results["outcome_favorability"] = outcome_metrics
            
            if not outcome_passed:
                self.logger.warning("⚠️ Data collection outcome is not favorable")
                return False
            
            self.logger.info("✅ Data collection validation passed")
            return True
            
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
            # Check minimum data size
            min_records = 1000
            if len(data) < min_records:
                self.logger.error(f"❌ Insufficient data: {len(data)} records (minimum: {min_records})")
                return False
            
            # Check for required columns (basic OHLCV)
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"❌ Missing required columns: {missing_columns}")
                return False
            
            # Check for reasonable price ranges
            price_columns = ["open", "high", "low", "close"]
            for col in price_columns:
                if col in data.columns:
                    if data[col].min() <= 0:
                        self.logger.error(f"❌ Invalid price values in {col} column")
                        return False
            
            # Check for reasonable volume values
            if "volume" in data.columns:
                if data["volume"].min() < 0:
                    self.logger.error("❌ Invalid volume values (negative)")
                    return False
            
            # Check data consistency (high >= low, etc.)
            if all(col in data.columns for col in ["high", "low", "open", "close"]):
                invalid_rows = (
                    (data["high"] < data["low"]) |
                    (data["high"] < data["open"]) |
                    (data["high"] < data["close"]) |
                    (data["low"] > data["open"]) |
                    (data["low"] > data["close"])
                ).sum()
                
                if invalid_rows > 0:
                    self.logger.warning(f"⚠️ Found {invalid_rows} rows with inconsistent OHLC data")
            
            # Check for reasonable time gaps (if timestamp column exists)
            if "timestamp" in data.columns:
                data_sorted = data.sort_values("timestamp")
                time_diffs = data_sorted["timestamp"].diff().dropna()
                
                # Check for reasonable time intervals (not too large gaps)
                max_gap_hours = 24  # 1 day
                large_gaps = (time_diffs > pd.Timedelta(hours=max_gap_hours)).sum()
                
                if large_gaps > len(data) * 0.1:  # More than 10% large gaps
                    self.logger.warning(f"⚠️ Found {large_gaps} large time gaps in data")
            
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
    return await validator.run_validation(training_input, pipeline_state)


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
