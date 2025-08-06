"""
Validator for Step 4: Analyst Labeling and Feature Engineering
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from typing import Any, Dict, List
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.base_validator import BaseValidator
from src.config import CONFIG


class Step4AnalystLabelingFeatureEngineeringValidator(BaseValidator):
    """Validator for Step 4: Analyst Labeling and Feature Engineering."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("step4_analyst_labeling_feature_engineering", config)
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> bool:
        """
        Validate the analyst labeling and feature engineering step.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating analyst labeling and feature engineering step...")
        
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        
        # Validate step result from pipeline state
        step_result = pipeline_state.get("analyst_labeling_feature_engineering", {})
        
        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics
        
        if not error_passed:
            self.logger.error("❌ Analyst labeling and feature engineering step had errors")
            return False
        
        # 2. Validate feature engineering outputs
        features_passed = self._validate_feature_engineering_outputs(symbol, exchange, data_dir)
        if not features_passed:
            self.logger.error("❌ Feature engineering outputs validation failed")
            return False
        
        # 3. Validate labeling quality
        labeling_passed = self._validate_labeling_quality(symbol, exchange, data_dir)
        if not labeling_passed:
            self.logger.error("❌ Labeling quality validation failed")
            return False
        
        # 4. Validate feature quality
        feature_quality_passed = self._validate_feature_quality(symbol, exchange, data_dir)
        if not feature_quality_passed:
            self.logger.error("❌ Feature quality validation failed")
            return False
        
        # 5. Validate data balance
        balance_passed = self._validate_data_balance(symbol, exchange, data_dir)
        if not balance_passed:
            self.logger.error("❌ Data balance validation failed")
            return False
        
        # 6. Validate outcome favorability
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(step_result)
        self.validation_results["outcome_favorability"] = outcome_metrics
        
        if not outcome_passed:
            self.logger.warning("⚠️ Analyst labeling and feature engineering outcome is not favorable")
            return False
        
        self.logger.info("✅ Analyst labeling and feature engineering validation passed")
        return True
    
    def _validate_feature_engineering_outputs(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate feature engineering outputs.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if outputs are valid
        """
        try:
            # Expected feature engineering output files
            expected_files = [
                f"{data_dir}/{exchange}_{symbol}_features_train.pkl",
                f"{data_dir}/{exchange}_{symbol}_features_validation.pkl",
                f"{data_dir}/{exchange}_{symbol}_features_test.pkl"
            ]
            
            missing_files = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(file_path, "feature_engineering")
                if not file_passed:
                    missing_files.append(file_path)
            
            if missing_files:
                self.logger.error(f"❌ Missing feature engineering files: {missing_files}")
                return False
            
            # Validate feature data quality
            for file_path in expected_files:
                try:
                    with open(file_path, "rb") as f:
                        feature_data = pickle.load(f)
                    
                    if not isinstance(feature_data, pd.DataFrame):
                        feature_data = pd.DataFrame(feature_data)
                    
                    # Validate feature data quality
                    quality_passed, quality_metrics = self.validate_data_quality(feature_data, "feature_data")
                    if not quality_passed:
                        self.logger.error(f"❌ Feature data quality validation failed for {file_path}")
                        return False
                    
                except Exception as e:
                    self.logger.error(f"❌ Error validating feature file {file_path}: {e}")
                    return False
            
            self.logger.info("✅ Feature engineering outputs validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during feature engineering outputs validation: {e}")
            return False
    
    def _validate_labeling_quality(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate labeling quality.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if labeling quality is acceptable
        """
        try:
            # Load labeled data files
            labeled_files = [
                f"{data_dir}/{exchange}_{symbol}_labeled_train.pkl",
                f"{data_dir}/{exchange}_{symbol}_labeled_validation.pkl",
                f"{data_dir}/{exchange}_{symbol}_labeled_test.pkl"
            ]
            
            for file_path in labeled_files:
                if not os.path.exists(file_path):
                    self.logger.warning(f"⚠️ Labeled data file not found: {file_path}")
                    continue
                
                try:
                    with open(file_path, "rb") as f:
                        labeled_data = pickle.load(f)
                    
                    if not isinstance(labeled_data, pd.DataFrame):
                        labeled_data = pd.DataFrame(labeled_data)
                    
                    # Check for label column
                    if "label" not in labeled_data.columns:
                        self.logger.error(f"❌ No label column found in {file_path}")
                        return False
                    
                    # Validate label values
                    labels = labeled_data["label"]
                    unique_labels = labels.unique()
                    
                    # Check for reasonable number of classes
                    if len(unique_labels) < 2:
                        self.logger.error(f"❌ Insufficient label classes: {len(unique_labels)}")
                        return False
                    
                    if len(unique_labels) > 10:
                        self.logger.warning(f"⚠️ Many label classes: {len(unique_labels)}")
                    
                    # Check for label balance
                    label_counts = labels.value_counts()
                    min_count = label_counts.min()
                    max_count = label_counts.max()
                    balance_ratio = min_count / max_count if max_count > 0 else 0
                    
                    if balance_ratio < 0.1:  # Imbalanced if minority class is less than 10% of majority
                        self.logger.warning(f"⚠️ Label imbalance detected: {balance_ratio:.3f}")
                    
                    # Check for missing labels
                    null_labels = labels.isnull().sum()
                    if null_labels > 0:
                        self.logger.warning(f"⚠️ Found {null_labels} missing labels")
                    
                    # Check for reasonable label values
                    if labels.dtype in ['int64', 'float64']:
                        if labels.min() < 0:
                            self.logger.warning("⚠️ Negative label values found")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error validating labeled data {file_path}: {e}")
                    return False
            
            self.logger.info("✅ Labeling quality validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during labeling quality validation: {e}")
            return False
    
    def _validate_feature_quality(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate feature quality characteristics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if feature quality is acceptable
        """
        try:
            feature_files = [
                f"{data_dir}/{exchange}_{symbol}_features_train.pkl",
                f"{data_dir}/{exchange}_{symbol}_features_validation.pkl",
                f"{data_dir}/{exchange}_{symbol}_features_test.pkl"
            ]
            
            for file_path in feature_files:
                try:
                    with open(file_path, "rb") as f:
                        feature_data = pickle.load(f)
                    
                    if not isinstance(feature_data, pd.DataFrame):
                        feature_data = pd.DataFrame(feature_data)
                    
                    # Check for sufficient features
                    if len(feature_data.columns) < 5:
                        self.logger.warning(f"⚠️ Few features: {len(feature_data.columns)}")
                    
                    # Check for feature diversity
                    numeric_features = feature_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_features) < 3:
                        self.logger.warning(f"⚠️ Few numeric features: {len(numeric_features)}")
                    
                    # Check for feature correlation
                    if len(numeric_features) > 1:
                        corr_matrix = feature_data[numeric_features].corr()
                        high_corr_pairs = []
                        
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                corr_val = abs(corr_matrix.iloc[i, j])
                                if corr_val > 0.95:  # Very high correlation
                                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                        
                        if high_corr_pairs:
                            self.logger.warning(f"⚠️ Found {len(high_corr_pairs)} highly correlated feature pairs")
                    
                    # Check for feature variance
                    for col in numeric_features:
                        variance = feature_data[col].var()
                        if variance == 0:
                            self.logger.warning(f"⚠️ Zero variance feature: {col}")
                        elif variance < 1e-6:
                            self.logger.warning(f"⚠️ Very low variance feature: {col} (var={variance})")
                    
                    # Check for reasonable feature ranges
                    for col in numeric_features:
                        feature_range = feature_data[col].max() - feature_data[col].min()
                        if feature_range == 0:
                            self.logger.warning(f"⚠️ Constant feature: {col}")
                        elif feature_range > 1e6:
                            self.logger.warning(f"⚠️ Very large range feature: {col} (range={feature_range})")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error validating feature quality for {file_path}: {e}")
                    return False
            
            self.logger.info("✅ Feature quality validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during feature quality validation: {e}")
            return False
    
    def _validate_data_balance(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate data balance across splits.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            bool: True if data is balanced
        """
        try:
            # Load labeled data from all splits
            split_files = [
                ("train", f"{data_dir}/{exchange}_{symbol}_labeled_train.pkl"),
                ("validation", f"{data_dir}/{exchange}_{symbol}_labeled_validation.pkl"),
                ("test", f"{data_dir}/{exchange}_{symbol}_labeled_test.pkl")
            ]
            
            split_data = {}
            
            for split_name, file_path in split_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "rb") as f:
                            data = pickle.load(f)
                        
                        if not isinstance(data, pd.DataFrame):
                            data = pd.DataFrame(data)
                        
                        split_data[split_name] = data
                        
                    except Exception as e:
                        self.logger.error(f"❌ Error loading {split_name} split: {e}")
                        return False
            
            if not split_data:
                self.logger.error("❌ No labeled data files found")
                return False
            
            # Check label distribution across splits
            if "label" in split_data.get("train", pd.DataFrame()).columns:
                train_labels = split_data["train"]["label"].value_counts()
                
                for split_name, data in split_data.items():
                    if split_name != "train" and "label" in data.columns:
                        split_labels = data["label"].value_counts()
                        
                        # Check if label distribution is similar
                        common_labels = set(train_labels.index) & set(split_labels.index)
                        
                        for label in common_labels:
                            train_ratio = train_labels[label] / len(split_data["train"])
                            split_ratio = split_labels[label] / len(data)
                            
                            # Check if ratios are reasonably similar (within 20%)
                            if abs(train_ratio - split_ratio) > 0.2:
                                self.logger.warning(f"⚠️ Label distribution mismatch for {label} in {split_name}: train={train_ratio:.3f}, {split_name}={split_ratio:.3f}")
            
            # Check feature distribution across splits
            feature_files = [
                ("train", f"{data_dir}/{exchange}_{symbol}_features_train.pkl"),
                ("validation", f"{data_dir}/{exchange}_{symbol}_features_validation.pkl"),
                ("test", f"{data_dir}/{exchange}_{symbol}_features_test.pkl")
            ]
            
            feature_data = {}
            for split_name, file_path in feature_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "rb") as f:
                            data = pickle.load(f)
                        
                        if not isinstance(data, pd.DataFrame):
                            data = pd.DataFrame(data)
                        
                        feature_data[split_name] = data
                        
                    except Exception as e:
                        self.logger.error(f"❌ Error loading {split_name} features: {e}")
                        return False
            
            if feature_data:
                # Check feature statistics across splits
                train_features = feature_data.get("train", pd.DataFrame())
                if not train_features.empty:
                    numeric_cols = train_features.select_dtypes(include=[np.number]).columns
                    
                    for split_name, data in feature_data.items():
                        if split_name != "train" and not data.empty:
                            for col in numeric_cols[:5]:  # Check first 5 features
                                if col in data.columns:
                                    train_mean = train_features[col].mean()
                                    split_mean = data[col].mean()
                                    
                                    if train_mean != 0:
                                        diff_ratio = abs(split_mean - train_mean) / abs(train_mean)
                                        if diff_ratio > 0.5:
                                            self.logger.warning(f"⚠️ Large feature distribution difference in {col} for {split_name}: train={train_mean:.3f}, {split_name}={split_mean:.3f}")
            
            self.logger.info("✅ Data balance validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during data balance validation: {e}")
            return False


async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the Step 4 Analyst Labeling and Feature Engineering validator.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        
    Returns:
        Dictionary containing validation results
    """
    validator = Step4AnalystLabelingFeatureEngineeringValidator(CONFIG)
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
            "analyst_labeling_feature_engineering": {
                "status": "SUCCESS",
                "duration": 180.5
            }
        }
        
        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")
    
    asyncio.run(test_validator())
