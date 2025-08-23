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
        
        # Quality thresholds
        self.max_nan_ratio = self.config.get("max_nan_ratio", 0.5)
        self.max_infinite_ratio = self.config.get("max_infinite_ratio", 0.1)
        self.min_unique_values = self.config.get("min_unique_values", 2)
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
            "step": "step1_data_collection",
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
            "step": "step1_5_data_converter",
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
            
        self.validation_results["step1_5"] = results
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
            "step": "step2_feature_engineering",
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
            
            # Check for NaN values
            nan_counts = df.isnull().sum()
            high_nan_features = nan_counts[nan_counts > len(df) * self.max_nan_ratio].index.tolist()
            result["nan_counts"] = nan_counts.to_dict()
            
            if high_nan_features:
                result["issues"].append(f"Features with >{self.max_nan_ratio*100}% NaN values: {high_nan_features}")
            
            # Check for infinite values
            infinite_counts = {}
            infinite_features = []
            
            for col in df.select_dtypes(include=[np.number]).columns:
                infinite_count = np.isinf(df[col]).sum()
                infinite_counts[col] = infinite_count
                
                if infinite_count > len(df) * self.max_infinite_ratio:
                    infinite_features.append(col)
            
            result["infinite_counts"] = infinite_counts
            
            if infinite_features:
                result["issues"].append(f"Features with >{self.max_infinite_ratio*100}% infinite values: {infinite_features}")
            
            # Check for constant features
            constant_features = []
            for col in df.columns:
                if df[col].nunique() <= self.min_unique_values:
                    constant_features.append(col)
            
            result["constant_features"] = constant_features
            
            if constant_features:
                result["issues"].append(f"Constant features found: {constant_features}")
            
            # Check if validation passed
            result["passed"] = len(result["issues"]) == 0
            
        except Exception as e:
            result["issues"].append(f"Error loading file: {str(e)}")
            
        return result
    
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
            
            # Check for NaN values
            nan_counts = df.isnull().sum()
            nan_features = nan_counts[nan_counts > 0].index.tolist()
            result["nan_features"] = nan_features
            
            if nan_features:
                result["issues"].append(f"Features with NaN values: {nan_features}")
            
            # Check for infinite values
            infinite_features = []
            for col in df.select_dtypes(include=[np.number]).columns:
                if np.isinf(df[col]).any():
                    infinite_features.append(col)
            
            result["infinite_features"] = infinite_features
            
            if infinite_features:
                result["issues"].append(f"Features with infinite values: {infinite_features}")
            
            # Check for constant features
            constant_features = []
            for col in df.columns:
                if df[col].nunique() <= self.min_unique_values:
                    constant_features.append(col)
            
            result["constant_features"] = constant_features
            
            if constant_features:
                result["issues"].append(f"Constant features found: {constant_features}")
            
            # Check for high correlation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr().abs()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > self.max_correlation_threshold:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                
                result["high_correlation_pairs"] = high_corr_pairs
                
                if high_corr_pairs:
                    result["issues"].append(f"Highly correlated feature pairs: {high_corr_pairs}")
            
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
        """Log detailed feature quality report."""
        self.logger.info("=" * 80)
        self.logger.info("📊 STEP2 FEATURE QUALITY REPORT")
        self.logger.info("=" * 80)
        
        problematic = results["problematic_features"]
        
        # Log NaN features
        if problematic["nan_features"]:
            self.logger.warning(f"⚠️ Features with NaN values ({len(problematic['nan_features'])}):")
            for feature in problematic["nan_features"][:10]:  # Show first 10
                self.logger.warning(f"   - {feature}")
            if len(problematic["nan_features"]) > 10:
                self.logger.warning(f"   ... and {len(problematic['nan_features']) - 10} more")
        
        # Log infinite features
        if problematic["infinite_features"]:
            self.logger.warning(f"⚠️ Features with infinite values ({len(problematic['infinite_features'])}):")
            for feature in problematic["infinite_features"][:10]:  # Show first 10
                self.logger.warning(f"   - {feature}")
            if len(problematic["infinite_features"]) > 10:
                self.logger.warning(f"   ... and {len(problematic['infinite_features']) - 10} more")
        
        # Log constant features
        if problematic["constant_features"]:
            self.logger.warning(f"⚠️ Constant features ({len(problematic['constant_features'])}):")
            for feature in problematic["constant_features"][:10]:  # Show first 10
                self.logger.warning(f"   - {feature}")
            if len(problematic["constant_features"]) > 10:
                self.logger.warning(f"   ... and {len(problematic['constant_features']) - 10} more")
        
        # Log high correlation pairs
        if problematic["high_correlation_pairs"]:
            self.logger.warning(f"⚠️ Highly correlated feature pairs ({len(problematic['high_correlation_pairs'])}):")
            for pair in problematic["high_correlation_pairs"][:5]:  # Show first 5
                self.logger.warning(f"   - {pair[0]} ↔ {pair[1]}")
            if len(problematic["high_correlation_pairs"]) > 5:
                self.logger.warning(f"   ... and {len(problematic['high_correlation_pairs']) - 5} more")
        
        # Summary
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