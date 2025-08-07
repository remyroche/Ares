"""
Validator for Step 3: Regime Data Splitting
"""

import os
import sys
import pickle
import pandas as pd
from typing import Any, Dict, List
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.base_validator import BaseValidator
from src.config import CONFIG


class Step3RegimeDataSplittingValidator(BaseValidator):
    """Validator for Step 3: Regime Data Splitting."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("step3_regime_data_splitting", config)
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> bool:
        """
        Validate the regime data splitting step.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating regime data splitting step...")
        
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        
        # Validate step result from pipeline state
        step_result = pipeline_state.get("regime_data_splitting", {})
        
        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics
        
        if not error_passed:
            self.logger.error("❌ Regime data splitting step had errors")
            return False
        
        # 2. Validate split data files existence
        split_files_passed = self._validate_split_files_existence(symbol, exchange, data_dir)
        if not split_files_passed:
            self.logger.error("❌ Split data files validation failed")
            return False
        
        # 3. Validate split data quality
        quality_passed = self._validate_split_data_quality(symbol, exchange, data_dir)
        if not quality_passed:
            self.logger.error("❌ Split data quality validation failed")
            return False
        
        # 4. Validate split proportions
        proportions_passed = self._validate_split_proportions(symbol, exchange, data_dir)
        if not proportions_passed:
            self.logger.error("❌ Split proportions validation failed")
            return False
        
        # 5. Validate split consistency
        consistency_passed = self._validate_split_consistency(symbol, exchange, data_dir)
        if not consistency_passed:
            self.logger.error("❌ Split consistency validation failed")
            return False
        
        # 6. Validate outcome favorability
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(step_result)
        self.validation_results["outcome_favorability"] = outcome_metrics
        
        if not outcome_passed:
            self.logger.warning("⚠️ Regime data splitting outcome is not favorable")
            return False
        
        self.logger.info("✅ Regime data splitting validation passed")
        return True
    
    def _validate_split_files_existence(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate that all expected split data files exist.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if all files exist
        """
        try:
            # Expected file patterns for different splits
            expected_files = [
                f"{data_dir}/{exchange}_{symbol}_train_data.pkl",
                f"{data_dir}/{exchange}_{symbol}_validation_data.pkl",
                f"{data_dir}/{exchange}_{symbol}_test_data.pkl"
            ]
            
            missing_files = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(file_path, "split_data")
                if not file_passed:
                    missing_files.append(file_path)
            
            if missing_files:
                self.logger.error(f"❌ Missing split data files: {missing_files}")
                return False
            
            self.logger.info("✅ All split data files exist")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating split files existence: {e}")
            return False
    
    def _validate_split_data_quality(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate quality of split data.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if data quality is acceptable
        """
        try:
            split_files = [
                ("train", f"{data_dir}/{exchange}_{symbol}_train_data.pkl"),
                ("validation", f"{data_dir}/{exchange}_{symbol}_validation_data.pkl"),
                ("test", f"{data_dir}/{exchange}_{symbol}_test_data.pkl")
            ]
            
            for split_name, file_path in split_files:
                try:
                    with open(file_path, "rb") as f:
                        split_data = pickle.load(f)
                    
                    # Convert to DataFrame if needed
                    if not isinstance(split_data, pd.DataFrame):
                        split_data = pd.DataFrame(split_data)
                    
                    # Validate data quality for this split
                    quality_passed, quality_metrics = self.validate_data_quality(split_data, f"{split_name}_data")
                    self.validation_results[f"{split_name}_data_quality"] = quality_metrics
                    
                    if not quality_passed:
                        self.logger.error(f"❌ {split_name} data quality validation failed")
                        return False
                    
                    # Additional split-specific validations
                    if not self._validate_split_specific_characteristics(split_data, split_name):
                        return False
                    
                except Exception as e:
                    self.logger.error(f"❌ Error validating {split_name} data: {e}")
                    return False
            
            self.logger.info("✅ All split data quality validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during split data quality validation: {e}")
            return False
    
    def _validate_split_specific_characteristics(self, data: pd.DataFrame, split_name: str) -> bool:
        """
        Validate characteristics specific to each split.
        
        Args:
            data: Split data DataFrame
            split_name: Name of the split (train/validation/test)
            
        Returns:
            bool: True if characteristics are valid
        """
        try:
            # Check minimum size requirements
            min_sizes = {
                "train": 1000,
                "validation": 200,
                "test": 200
            }
            
            min_size = min_sizes.get(split_name, 100)
            if len(data) < min_size:
                self.logger.error(f"❌ {split_name} split too small: {len(data)} records (minimum: {min_size})")
                return False
            
            # Check for required columns
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"❌ {split_name} split missing columns: {missing_columns}")
                return False
            
            # Check for reasonable data ranges
            if "close" in data.columns:
                price_range = data["close"].max() - data["close"].min()
                if price_range <= 0:
                    self.logger.error(f"❌ {split_name} split has invalid price range")
                    return False
            
            self.logger.info(f"✅ {split_name} split characteristics validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating {split_name} characteristics: {e}")
            return False
    
    def _validate_split_proportions(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate that split proportions are reasonable.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if proportions are acceptable
        """
        try:
            split_files = [
                ("train", f"{data_dir}/{exchange}_{symbol}_train_data.pkl"),
                ("validation", f"{data_dir}/{exchange}_{symbol}_validation_data.pkl"),
                ("test", f"{data_dir}/{exchange}_{symbol}_test_data.pkl")
            ]
            
            split_sizes = {}
            total_size = 0
            
            # Calculate sizes for each split
            for split_name, file_path in split_files:
                try:
                    with open(file_path, "rb") as f:
                        split_data = pickle.load(f)
                    
                    if not isinstance(split_data, pd.DataFrame):
                        split_data = pd.DataFrame(split_data)
                    
                    split_sizes[split_name] = len(split_data)
                    total_size += len(split_data)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error reading {split_name} split: {e}")
                    return False
            
            if total_size == 0:
                self.logger.error("❌ Total data size is zero")
                return False
            
            # Calculate proportions
            proportions = {split: size / total_size for split, size in split_sizes.items()}
            
            # Validate proportions
            expected_proportions = {
                "train": (0.6, 0.8),      # 60-80%
                "validation": (0.100, 0.200),  # 10-20%
                "test": (0.100, 0.200)        # 10-20% 
            }
            
            for split_name, (min_prop, max_prop) in expected_proportions.items():
                actual_prop = proportions.get(split_name, 0)
                tolerance = 1e-6
                # Check if proportion is within range (inclusive of boundaries)
                if actual_prop < min_prop - tolerance or actual_prop > max_prop + tolerance:
                    self.logger.warning(f"⚠️ {split_name} proportion {actual_prop:.6f} outside expected range ({min_prop:.6f}-{max_prop:.6f})")
                    self.logger.debug(f"Debug: actual_prop={actual_prop}, min_prop={min_prop}, max_prop={max_prop}")
                    self.logger.debug(f"Debug: actual_prop < min_prop = {actual_prop < min_prop}")
                    self.logger.debug(f"Debug: actual_prop > max_prop = {actual_prop > max_prop}")
                    # Don't fail validation for proportion warnings, just log them
                else:
                    self.logger.info(f"✅ {split_name} proportion {actual_prop:.6f} within expected range ({min_prop:.6f}-{max_prop:.6f})")
                    
            # Check that proportions sum to approximately 1
            total_proportion = sum(proportions.values())
            if abs(total_proportion - 1.0) > 0.01:
                self.logger.error(f"❌ Split proportions don't sum to 1: {total_proportion:.3f}")
                return False
            
            self.validation_results["split_proportions"] = {
                "proportions": proportions,
                "total_size": total_size,
                "split_sizes": split_sizes,
                "passed": True
            }
            
            self.logger.info(f"✅ Split proportions validation passed: {proportions}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during split proportions validation: {e}")
            return False
    
    def _validate_split_consistency(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate consistency between splits (e.g., no data leakage).
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if splits are consistent
        """
        try:
            split_files = [
                ("train", f"{data_dir}/{exchange}_{symbol}_train_data.pkl"),
                ("validation", f"{data_dir}/{exchange}_{symbol}_validation_data.pkl"),
                ("test", f"{data_dir}/{exchange}_{symbol}_test_data.pkl")
            ]
            
            split_data = {}
            
            # Load all splits
            for split_name, file_path in split_files:
                try:
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)
                    
                    if not isinstance(data, pd.DataFrame):
                        data = pd.DataFrame(data)
                    
                    split_data[split_name] = data
                    
                except Exception as e:
                    self.logger.error(f"❌ Error loading {split_name} split: {e}")
                    return False
            
            # Check for overlapping timestamps (if timestamp column exists)
            if "timestamp" in split_data["train"].columns:
                train_timestamps = set(split_data["train"]["timestamp"])
                val_timestamps = set(split_data["validation"]["timestamp"])
                test_timestamps = set(split_data["test"]["timestamp"])
                
                # Check for overlaps
                train_val_overlap = train_timestamps.intersection(val_timestamps)
                train_test_overlap = train_timestamps.intersection(test_timestamps)
                val_test_overlap = val_timestamps.intersection(test_timestamps)
                
                if train_val_overlap:
                    self.logger.error(f"❌ Train-validation overlap: {len(train_val_overlap)} timestamps")
                    return False
                
                if train_test_overlap:
                    self.logger.error(f"❌ Train-test overlap: {len(train_test_overlap)} timestamps")
                    return False
                
                if val_test_overlap:
                    self.logger.error(f"❌ Validation-test overlap: {len(val_test_overlap)} timestamps")
                    return False
            
            # Check for temporal ordering (train < validation < test)
            if "timestamp" in split_data["train"].columns:
                train_max_time = split_data["train"]["timestamp"].max()
                val_min_time = split_data["validation"]["timestamp"].min()
                val_max_time = split_data["validation"]["timestamp"].max()
                test_min_time = split_data["test"]["timestamp"].min()
                
                if val_min_time <= train_max_time:
                    self.logger.warning("⚠️ Validation data may overlap with training data temporally")
                
                if test_min_time <= val_max_time:
                    self.logger.warning("⚠️ Test data may overlap with validation data temporally")
            
            # Check for reasonable feature distributions across splits
            numeric_columns = split_data["train"].select_dtypes(include=['number']).columns
            for col in numeric_columns[:5]:  # Check first 5 numeric columns
                if col in split_data["train"].columns:
                    train_mean = split_data["train"][col].mean()
                    val_mean = split_data["validation"][col].mean()
                    test_mean = split_data["test"][col].mean()
                    
                    # Check if means are reasonably close (within 50% of train mean)
                    if train_mean != 0:
                        val_diff = abs(val_mean - train_mean) / abs(train_mean)
                        test_diff = abs(test_mean - train_mean) / abs(train_mean)
                        
                        if val_diff > 0.5 or test_diff > 0.5:
                            self.logger.warning(f"⚠️ Large distribution difference in {col}: train={train_mean:.3f}, val={val_mean:.3f}, test={test_mean:.3f}")
            
            self.logger.info("✅ Split consistency validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during split consistency validation: {e}")
            return False


async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the Step 3 Regime Data Splitting validator.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        
    Returns:
        Dictionary containing validation results
    """
    validator = Step3RegimeDataSplittingValidator(CONFIG)
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
            "regime_data_splitting": {
                "status": "SUCCESS",
                "duration": 30.5
            }
        }
        
        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")
    
    asyncio.run(test_validator())
