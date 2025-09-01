"""
Comprehensive Data Quality Validator

This module provides comprehensive data quality validation for all pipeline steps,
with special attention to NaN, infinite, and constant values in Step2.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
except ImportError:
    system_logger = logging.getLogger("ComprehensiveDataQualityValidator")


class ComprehensiveDataQualityValidator:
    """
    Comprehensive data quality validator for all pipeline steps.
    
    Features:
    - File structure validation
    - Data quality checks (NaN, infinite, constant values)
    - Feature-specific validation for Step2
    - Detailed logging and reporting
    - Configurable thresholds
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = system_logger.getChild("ComprehensiveDataQualityValidator")
        
        # Quality thresholds - Updated with zero tolerance for NaN and infinite values
        self.max_nan_ratio = self.config.get("max_nan_ratio", 0.0)  # 0% NaN (zero tolerance)
        self.max_infinite_count = self.config.get("max_infinite_count", 0)  # 0 infinite values (zero tolerance)
        self.min_unique_values = self.config.get("min_unique_values", 2)  # 2+ unique values
        self.max_constant_ratio = self.config.get("max_constant_ratio", 0.95)
        self.min_feature_count = self.config.get("min_feature_count", 40)
        self.max_correlation_threshold = self.config.get("max_correlation_threshold", 0.95)
        
        # Validation results storage
        self.validation_results = {}
        self.quality_issues = {}
        
    def validate_step1_data_quality(self, symbol: str, exchange: str, data_dir: str = "data_cache") -> Dict[str, Any]:
        """
        Validate Step1 data collection quality.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            Dict with validation results
        """
        self.logger.info("🔍 Validating Step1 data collection quality...")
        
        results = {
            "step": "step01_data_collection",
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            "validation_passed": False,
            "issues": [],
            "file_checks": {},
            "data_quality": {}
        }
        
        try:
            # Check required files exist
            required_files = [
                f"klines_{exchange}_{symbol}_1m_consolidated.parquet",
                f"aggtrades_{exchange}_{symbol}_consolidated.parquet"
            ]
            
            for file_name in required_files:
                file_path = os.path.join(data_dir, file_name)
                file_check = self._validate_file_structure(file_path, file_name)
                results["file_checks"][file_name] = file_check
                
                if file_check["exists"]:
                    # Validate data quality for existing files
                    data_quality = self._validate_dataframe_quality(file_path, file_name)
                    results["data_quality"][file_name] = data_quality
                    
                    if not data_quality["passed"]:
                        results["issues"].extend(data_quality["issues"])
                else:
                    results["issues"].append(f"Required file missing: {file_name}")
            
            # Check if validation passed
            results["validation_passed"] = len(results["issues"]) == 0
            
            if results["validation_passed"]:
                self.logger.info("✅ Step1 data quality validation passed")
            else:
                self.logger.warning(f"⚠️ Step1 data quality validation found {len(results['issues'])} issues")
                
        except Exception as e:
            self.logger.exception(f"❌ Error during Step1 validation: {e}")
            results["issues"].append(f"Validation error: {str(e)}")
            results["validation_passed"] = False
            
        self.validation_results["step1"] = results
        return results
    
    def validate_step1_5_data_quality(self, symbol: str, exchange: str, data_dir: str = "data_cache") -> Dict[str, Any]:
        """
        Validate Step1.5 data converter quality.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            Dict with validation results
        """
        self.logger.info("🔍 Validating Step1.5 data converter quality...")
        
        results = {
            "step": "step01_5_data_converter",
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            "validation_passed": False,
            "issues": [],
            "file_checks": {},
            "data_quality": {}
        }
        
        try:
            # Check unified data directory structure
            unified_dir = os.path.join(data_dir, "unified", exchange.lower(), symbol, "1m")
            
            if not os.path.exists(unified_dir):
                results["issues"].append(f"Unified data directory not found: {unified_dir}")
            else:
                # Check for parquet files in unified directory
                parquet_files = list(Path(unified_dir).rglob("*.parquet"))
                
                if not parquet_files:
                    results["issues"].append("No parquet files found in unified directory")
                else:
                    results["file_checks"]["unified_files"] = {
                        "exists": True,
                        "count": len(parquet_files),
                        "files": [str(f) for f in parquet_files[:5]]  # Show first 5 files
                    }
                    
                    # Validate sample of unified data files
                    for i, file_path in enumerate(parquet_files[:3]):  # Check first 3 files
                        data_quality = self._validate_dataframe_quality(str(file_path), f"unified_sample_{i}")
                        results["data_quality"][f"unified_sample_{i}"] = data_quality
                        
                        if not data_quality["passed"]:
                            results["issues"].extend(data_quality["issues"])
            
            # Check if validation passed
            results["validation_passed"] = len(results["issues"]) == 0
            
            if results["validation_passed"]:
                self.logger.info("✅ Step1.5 data quality validation passed")
            else:
                self.logger.warning(f"⚠️ Step1.5 data quality validation found {len(results['issues'])} issues")
                
        except Exception as e:
            self.logger.exception(f"❌ Error during Step1.5 validation: {e}")
            results["issues"].append(f"Validation error: {str(e)}")
            results["validation_passed"] = False
            
        self.validation_results["step01_5"] = results
        return results
    
    def validate_step2_data_quality(self, symbol: str, exchange: str, data_dir: str = "data/training") -> Dict[str, Any]:
        """
        Validate Step2 feature engineering quality with special attention to NaN, infinite, and constant values.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            
        Returns:
            Dict with validation results
        """
        self.logger.info("🔍 Validating Step2 feature engineering quality...")
        
        results = {
            "step": "step02_feature_engineering",
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            "validation_passed": False,
            "issues": [],
            "file_checks": {},
            "data_quality": {},
            "feature_quality": {},
            "problematic_features": {
                "nan_features": [],
                "infinite_features": [],
                "constant_features": [],
                "high_correlation_pairs": []
            }
        }
        
        try:
            # Check feature files exist
            feature_files = [
                f"{exchange}_{symbol}_features_train.parquet",
                f"{exchange}_{symbol}_features_validation.parquet",
                f"{exchange}_{symbol}_features_test.parquet"
            ]
            
            for file_name in feature_files:
                file_path = os.path.join(data_dir, file_name)
                file_check = self._validate_file_structure(file_path, file_name)
                results["file_checks"][file_name] = file_check
                
                if file_check["exists"]:
                    # Comprehensive feature quality validation
                    feature_quality = self._validate_feature_quality(file_path, file_name)
                    results["feature_quality"][file_name] = feature_quality
                    
                    if not feature_quality["passed"]:
                        results["issues"].extend(feature_quality["issues"])
                        
                    # Collect problematic features
                    for issue_type in ["nan_features", "infinite_features", "constant_features", "high_correlation_pairs"]:
                        if issue_type in feature_quality:
                            results["problematic_features"][issue_type].extend(feature_quality[issue_type])
                else:
                    results["issues"].append(f"Required feature file missing: {file_name}")
            
            # Check if validation passed
            results["validation_passed"] = len(results["issues"]) == 0
            
            # Log detailed feature quality report
            self._log_feature_quality_report(results)
            
            if results["validation_passed"]:
                self.logger.info("✅ Step2 feature quality validation passed")
            else:
                self.logger.warning(f"⚠️ Step2 feature quality validation found {len(results['issues'])} issues")
                
        except Exception as e:
            self.logger.exception(f"❌ Error during Step2 validation: {e}")
            results["issues"].append(f"Validation error: {str(e)}")
            results["validation_passed"] = False
            
        self.validation_results["step2"] = results
        return results
    
    def _validate_file_structure(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """Validate file structure and basic properties."""
        result = {
            "exists": False,
            "size_bytes": 0,
            "size_mb": 0,
            "last_modified": None,
            "issues": []
        }
        
        try:
            if os.path.exists(file_path):
                result["exists"] = True
                result["size_bytes"] = os.path.getsize(file_path)
                result["size_mb"] = result["size_bytes"] / (1024 * 1024)
                result["last_modified"] = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                
                # Check if file is empty
                if result["size_bytes"] == 0:
                    result["issues"].append("File is empty")
                    
            else:
                result["issues"].append("File does not exist")
                
        except Exception as e:
            result["issues"].append(f"Error checking file: {str(e)}")
            
        return result
    
    def _validate_dataframe_quality(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """Validate DataFrame quality including NaN, infinite, and constant values."""
        result = {
            "passed": False,
            "shape": None,
            "memory_usage_mb": 0,
            "dtypes": {},
            "nan_counts": {},
            "infinite_counts": {},
            "constant_features": [],
            "structure_issues": [],
            "issues": []
        }
        
        try:
            # Load DataFrame
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    df = pickle.load(f)
                    if not isinstance(df, pd.DataFrame):
                        df = pd.DataFrame(df)
            else:
                result["issues"].append("Unsupported file format")
                return result
            
            result["shape"] = df.shape
            result["memory_usage_mb"] = df.memory_usage(deep=True).sum() / (1024 * 1024)
            result["dtypes"] = df.dtypes.value_counts().to_dict()
            
            # Validate data structure first
            structure_issues = self._validate_data_structure(df, file_name)
            result["structure_issues"] = structure_issues
            result["issues"].extend(structure_issues)
            
            # Check for NaN values (zero tolerance)
            nan_counts = df.isnull().sum()
            nan_features = nan_counts[nan_counts > 0].index.tolist()  # Any NaN values
            result["nan_counts"] = nan_counts.to_dict()
            
            if nan_features:
                # Detailed NaN logging
                nan_details = []
                for feature in nan_features:
                    nan_count = nan_counts[feature]
                    nan_ratio = nan_count / len(df) * 100
                    nan_details.append(f"{feature}({nan_count} NaN, {nan_ratio:.3f}%)")
                result["issues"].append(f"Features with NaN values (zero tolerance): {', '.join(nan_details)}")
            
            # Check for infinite values (zero tolerance)
            infinite_counts = {}
            infinite_features = []
            infinite_details = []
            
            for col in df.select_dtypes(include=[np.number]).columns:
                infinite_count = np.isinf(df[col]).sum()
                infinite_counts[col] = infinite_count
                
                if infinite_count > 0:  # Any infinite values
                    infinite_features.append(col)
                    infinite_ratio = infinite_count / len(df) * 100
                    infinite_details.append(f"{col}({infinite_count} infinite, {infinite_ratio:.3f}%)")
            
            result["infinite_counts"] = infinite_counts
            
            if infinite_features:
                result["issues"].append(f"Features with infinite values (zero tolerance): {', '.join(infinite_details)}")
            
            # Check for constant features (2+ unique values, except boolean)
            constant_features = []
            for col in df.columns:
                unique_count = df[col].nunique()
                # Allow boolean features (2 unique values) and binary features
                if unique_count < self.min_unique_values and not self._is_boolean_feature(df[col]):
                    constant_features.append(col)
            
            result["constant_features"] = constant_features
            
            if constant_features:
                result["issues"].append(f"Constant features found: {constant_features}")
            
            # Check if validation passed
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Error loading file: {str(e)}")
            
        return result
    
    def _validate_data_structure(self, df: pd.DataFrame, file_name: str) -> List[str]:
        """
        Validate data structure including columns, format, index, and data types.
        
        Args:
            df: DataFrame to validate
            file_name: Name of the file for logging
            
        Returns:
            List of structure issues found
        """
        issues = []
        
        # Check if DataFrame is empty
        if df.empty:
            issues.append(f"{file_name}: DataFrame is empty")
            return issues
        
        # Check for required columns based on file type
        if "klines" in file_name.lower():
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                issues.append(f"{file_name}: Missing required klines columns: {missing_columns}")
        
        elif "aggtrades" in file_name.lower():
            required_columns = ["timestamp", "price", "quantity"]
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                issues.append(f"{file_name}: Missing required aggtrades columns: {missing_columns}")
        
        elif "features" in file_name.lower():
            # Features should have timestamp and at least some feature columns
            if "timestamp" not in df.columns:
                issues.append(f"{file_name}: Missing timestamp column in features")
            
            feature_columns = [col for col in df.columns if col != "timestamp"]
            if len(feature_columns) < 5:  # Minimum feature count
                issues.append(f"{file_name}: Insufficient feature columns: {len(feature_columns)}")
        
        # Check for duplicate columns
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            issues.append(f"{file_name}: Duplicate columns found: {duplicate_columns}")
        
        # Check for empty columns
        empty_columns = []
        for col in df.columns:
            if df[col].isnull().all():
                empty_columns.append(col)
        if empty_columns:
            issues.append(f"{file_name}: Empty columns found: {empty_columns}")
        
        # Check for proper data types
        type_issues = []
        for col in df.columns:
            if col == "timestamp":
                # Timestamp should be datetime
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    type_issues.append(f"timestamp column should be datetime, got {df[col].dtype}")
            elif col in ["open", "high", "low", "close", "volume", "price", "quantity"]:
                # OHLCV and trade data should be numeric
                if not pd.api.types.is_numeric_dtype(df[col]):
                    type_issues.append(f"{col} column should be numeric, got {df[col].dtype}")
        
        if type_issues:
            issues.append(f"{file_name}: Data type issues: {type_issues}")
        
        # Check for proper index
        if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
            # If timestamp column exists, it should be properly formatted
            try:
                pd.to_datetime(df["timestamp"])
            except Exception:
                issues.append(f"{file_name}: Timestamp column contains invalid datetime values")
        
        # Check for reasonable data ranges
        range_issues = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check for extreme values
                if col in ["open", "high", "low", "close", "price"]:
                    if (df[col] <= 0).any():
                        range_issues.append(f"{col} contains non-positive values")
                elif col == "volume":
                    if (df[col] < 0).any():
                        range_issues.append(f"{col} contains negative values")
        
        if range_issues:
            issues.append(f"{file_name}: Data range issues: {range_issues}")
        
        return issues
    
    def _is_boolean_feature(self, series: pd.Series) -> bool:
        """
        Check if a series represents a boolean feature.
        
        Args:
            series: Pandas series to check
            
        Returns:
            True if the series represents a boolean feature
        """
        # Check if it's already boolean dtype
        if pd.api.types.is_bool_dtype(series):
            return True
        
        # Check if it has exactly 2 unique values that could be boolean
        unique_values = series.dropna().unique()
        if len(unique_values) == 2:
            # Check if values are typical boolean patterns
            unique_set = set(unique_values)
            boolean_patterns = [
                {True, False},
                {1, 0},
                {1.0, 0.0},
                {'True', 'False'},
                {'true', 'false'},
                {'1', '0'},
                {'yes', 'no'},
                {'Y', 'N'},
                {'y', 'n'}
            ]
            
            for pattern in boolean_patterns:
                if unique_set == pattern:
                    return True
        
        return False
    
    def _validate_feature_quality(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """Comprehensive feature quality validation for Step2."""
        result = {
            "passed": False,
            "shape": None,
            "feature_count": 0,
            "relevant_feature_count": 0,
            "nan_features": [],
            "infinite_features": [],
            "constant_features": [],
            "high_correlation_pairs": [],
            "issues": []
        }
        
        try:
            # Load feature DataFrame
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    df = pickle.load(f)
                    if not isinstance(df, pd.DataFrame):
                        df = pd.DataFrame(df)
            else:
                result["issues"].append("Unsupported file format")
                return result
            
            result["shape"] = df.shape
            result["feature_count"] = len(df.columns)
            
            # Remove raw OHLCV columns if present
            forbidden = {"open", "high", "low", "close", "volume"}
            present_forbidden = [c for c in df.columns if c in forbidden]
            if present_forbidden:
                df = df.drop(columns=present_forbidden)
                result["issues"].append(f"Removed raw OHLCV columns: {present_forbidden}")
            
            # Validate data structure first
            structure_issues = self._validate_data_structure(df, file_name)
            result["structure_issues"] = structure_issues
            result["issues"].extend(structure_issues)
            
            # Check for NaN values (zero tolerance)
            nan_counts = df.isnull().sum()
            nan_features = nan_counts[nan_counts > 0].index.tolist()  # Any NaN values
            result["nan_features"] = nan_features
            
            if nan_features:
                # Detailed NaN logging with counts and percentages
                nan_details = []
                for feature in nan_features:
                    nan_count = nan_counts[feature]
                    nan_ratio = nan_count / len(df) * 100
                    nan_details.append(f"{feature}({nan_count} NaN, {nan_ratio:.3f}%)")
                result["issues"].append(f"Features with NaN values (zero tolerance): {', '.join(nan_details)}")
            
            # Check for infinite values (zero tolerance)
            infinite_features = []
            infinite_details = []
            for col in df.select_dtypes(include=[np.number]).columns:
                infinite_count = np.isinf(df[col]).sum()
                if infinite_count > 0:  # Any infinite values
                    infinite_features.append(col)
                    infinite_ratio = infinite_count / len(df) * 100
                    infinite_details.append(f"{col}({infinite_count} infinite, {infinite_ratio:.3f}%)")
            
            result["infinite_features"] = infinite_features
            
            if infinite_features:
                result["issues"].append(f"Features with infinite values (zero tolerance): {', '.join(infinite_details)}")
            
            # Check for constant features (2+ unique values, except boolean)
            constant_features = []
            constant_details = []
            for col in df.columns:
                unique_count = df[col].nunique()
                # Allow boolean features (2 unique values) and binary features
                if unique_count < self.min_unique_values and not self._is_boolean_feature(df[col]):
                    constant_features.append(col)
                    unique_values = df[col].dropna().unique()
                    constant_details.append(f"{col}({unique_count} unique: {unique_values})")
            
            result["constant_features"] = constant_features
            
            if constant_features:
                result["issues"].append(f"Constant features found: {', '.join(constant_details)}")
            
            # Check for high correlation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr().abs()
                high_corr_pairs = []
                high_corr_details = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if corr_value > self.max_correlation_threshold:
                            pair = (corr_matrix.columns[i], corr_matrix.columns[j])
                            high_corr_pairs.append(pair)
                            high_corr_details.append(f"{pair[0]}↔{pair[1]}({corr_value:.3f})")
                
                result["high_correlation_pairs"] = high_corr_pairs
                
                if high_corr_pairs:
                    result["issues"].append(f"Highly correlated feature pairs: {', '.join(high_corr_details)}")
            
            # Calculate relevant features
            problematic_features = set(nan_features + infinite_features + constant_features)
            relevant_features = [col for col in df.columns if col not in problematic_features]
            result["relevant_feature_count"] = len(relevant_features)
            
            # Check minimum feature requirement
            if result["relevant_feature_count"] < self.min_feature_count:
                result["issues"].append(
                    f"Insufficient relevant features: {result['relevant_feature_count']} "
                    f"(minimum required: {self.min_feature_count})"
                )
            
            # Check if validation passed
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Error validating features: {str(e)}")
            
        return result
    
    def _log_feature_quality_report(self, results: Dict[str, Any]) -> None:
        """Log detailed feature quality report with comprehensive information about problematic values."""
        self.logger.info("=" * 80)
        self.logger.info("📊 STEP2 FEATURE QUALITY REPORT")
        self.logger.info("=" * 80)
        
        problematic = results["problematic_features"]
        
        # Log NaN features with detailed information
        if problematic["nan_features"]:
            self.logger.warning(f"⚠️ Features with NaN values ({len(problematic['nan_features'])}):")
            for feature in problematic["nan_features"][:10]:  # Show first 10
                self.logger.warning(f"   - {feature}")
            if len(problematic["nan_features"]) > 10:
                self.logger.warning(f"   ... and {len(problematic['nan_features']) - 10} more")
            
            # Log detailed NaN statistics
            self.logger.warning("📊 NaN Statistics:")
            for feature in problematic["nan_features"][:5]:  # Show detailed info for first 5
                self.logger.warning(f"   • {feature}: NaN count and percentage details available in validation results")
        
        # Log infinite features with detailed information
        if problematic["infinite_features"]:
            self.logger.warning(f"⚠️ Features with infinite values ({len(problematic['infinite_features'])}):")
            for feature in problematic["infinite_features"][:10]:  # Show first 10
                self.logger.warning(f"   - {feature}")
            if len(problematic["infinite_features"]) > 10:
                self.logger.warning(f"   ... and {len(problematic['infinite_features']) - 10} more")
            
            # Log detailed infinite statistics
            self.logger.warning("📊 Infinite Value Statistics:")
            for feature in problematic["infinite_features"][:5]:  # Show detailed info for first 5
                self.logger.warning(f"   • {feature}: Infinite count and percentage details available in validation results")
        
        # Log constant features with detailed information
        if problematic["constant_features"]:
            self.logger.warning(f"⚠️ Constant features ({len(problematic['constant_features'])}):")
            for feature in problematic["constant_features"][:10]:  # Show first 10
                self.logger.warning(f"   - {feature}")
            if len(problematic["constant_features"]) > 10:
                self.logger.warning(f"   ... and {len(problematic['constant_features']) - 10} more")
            
            # Log detailed constant feature information
            self.logger.warning("📊 Constant Feature Details:")
            for feature in problematic["constant_features"][:5]:  # Show detailed info for first 5
                self.logger.warning(f"   • {feature}: Unique values and counts available in validation results")
        
        # Log high correlation pairs with detailed information
        if problematic["high_correlation_pairs"]:
            self.logger.warning(f"⚠️ Highly correlated feature pairs ({len(problematic['high_correlation_pairs'])}):")
            for pair in problematic["high_correlation_pairs"][:5]:  # Show first 5
                self.logger.warning(f"   - {pair[0]} ↔ {pair[1]}")
            if len(problematic["high_correlation_pairs"]) > 5:
                self.logger.warning(f"   ... and {len(problematic['high_correlation_pairs']) - 5} more")
            
            # Log detailed correlation information
            self.logger.warning("📊 Correlation Details:")
            for pair in problematic["high_correlation_pairs"][:3]:  # Show detailed info for first 3
                self.logger.warning(f"   • {pair[0]} ↔ {pair[1]}: Correlation coefficient available in validation results")
        
        # Summary with detailed breakdown
        total_issues = (
            len(problematic["nan_features"]) +
            len(problematic["infinite_features"]) +
            len(problematic["constant_features"]) +
            len(problematic["high_correlation_pairs"])
        )
        
        if total_issues == 0:
            self.logger.info("✅ No feature quality issues detected")
        else:
            self.logger.warning(f"⚠️ Total feature quality issues: {total_issues}")
            self.logger.warning("📋 Issue Breakdown:")
            self.logger.warning(f"   • NaN features: {len(problematic['nan_features'])}")
            self.logger.warning(f"   • Infinite features: {len(problematic['infinite_features'])}")
            self.logger.warning(f"   • Constant features: {len(problematic['constant_features'])}")
            self.logger.warning(f"   • High correlation pairs: {len(problematic['high_correlation_pairs'])}")
            self.logger.warning("💡 For detailed information about each problematic value, check the validation results")
        
        self.logger.info("=" * 80)
    
    def save_validation_report(self, output_path: str) -> None:
        """Save comprehensive validation report to file."""
        try:
            report = {
                "validation_timestamp": datetime.now().isoformat(),
                "config": self.config,
                "results": self.validation_results,
                "summary": {
                    "total_steps": len(self.validation_results),
                    "passed_steps": sum(1 for r in self.validation_results.values() if r["validation_passed"]),
                    "failed_steps": sum(1 for r in self.validation_results.values() if not r["validation_passed"])
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"✅ Validation report saved to: {output_path}")
            
        except Exception as e:
            self.logger.exception(f"❌ Error saving validation report: {e}")


# Convenience functions for easy integration
def validate_step1_quality(symbol: str, exchange: str, data_dir: str = "data_cache") -> Dict[str, Any]:
    """Convenience function to validate Step1 data quality."""
    validator = ComprehensiveDataQualityValidator()
    return validator.validate_step1_data_quality(symbol, exchange, data_dir)

def validate_step1_5_quality(symbol: str, exchange: str, data_dir: str = "data_cache") -> Dict[str, Any]:
    """Convenience function to validate Step1.5 data quality."""
    validator = ComprehensiveDataQualityValidator()
    return validator.validate_step1_5_data_quality(symbol, exchange, data_dir)

def validate_step2_quality(symbol: str, exchange: str, data_dir: str = "data/training") -> Dict[str, Any]:
    """Convenience function to validate Step2 data quality."""
    validator = ComprehensiveDataQualityValidator()
    return validator.validate_step2_data_quality(symbol, exchange, data_dir)