"""
Data Quality Framework

This module provides a comprehensive data quality framework that integrates with
the enhanced outlier handler and other quality validation tools.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

from .enhanced_outlier_handler import enhanced_outlier_handler, OutlierSeverity
from .logger import system_logger


class DataQualityFramework:
    """Comprehensive data quality framework with outlier handling integration."""
    
    def __init__(self):
        """Initialize data quality framework."""
        self.logger = system_logger.getChild("DataQualityFramework")
        self.outlier_handler = enhanced_outlier_handler
        
        # Default cleaning rules
        self.default_cleaning_rules = {
            "outlier_handling": "detect_only",
            "outlier_config": {
                "method": "iqr",
                "threshold": 1.5,
                "severity_threshold": "medium",
                "raise_errors": False
            },
            "null_handling": "drop",
            "duplicate_handling": "drop_first",
            "data_type_validation": True,
            "schema_validation": True
        }
        
        self.logger.info("🔧 Data Quality Framework initialized")
    
    def clean_data(self, data: pd.DataFrame, cleaning_rules: Dict[str, Any] = None) -> pd.DataFrame:
        """Clean data according to specified rules.
        
        Args:
            data: Data to clean
            cleaning_rules: Cleaning configuration (uses defaults if None)
            
        Returns:
            Cleaned data
        """
        if cleaning_rules is None:
            cleaning_rules = self.default_cleaning_rules.copy()
        
        self.logger.info(f"🧹 Starting data cleaning for {len(data)} rows")
        original_shape = data.shape
        
        # Apply cleaning steps
        cleaned_data = data.copy()
        
        # 1. Schema validation
        if cleaning_rules.get("schema_validation", True):
            cleaned_data = self._validate_schema(cleaned_data, cleaning_rules)
        
        # 2. Data type validation
        if cleaning_rules.get("data_type_validation", True):
            cleaned_data = self._validate_data_types(cleaned_data, cleaning_rules)
        
        # 3. Null handling
        cleaned_data = self._handle_nulls(cleaned_data, cleaning_rules)
        
        # 4. Duplicate handling
        cleaned_data = self._handle_duplicates(cleaned_data, cleaning_rules)
        
        # 5. Outlier handling
        cleaned_data = self._handle_outliers(cleaned_data, cleaning_rules)
        
        # Log cleaning results
        final_shape = cleaned_data.shape
        rows_removed = original_shape[0] - final_shape[0]
        cols_removed = original_shape[1] - final_shape[1]
        
        self.logger.info(f"✅ Data cleaning completed")
        self.logger.info(f"   Original shape: {original_shape}")
        self.logger.info(f"   Final shape: {final_shape}")
        self.logger.info(f"   Rows removed: {rows_removed}")
        self.logger.info(f"   Columns removed: {cols_removed}")
        
        return cleaned_data
    
    def _validate_schema(self, data: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Validate data schema."""
        try:
            # Try to validate against klines schema first
            validation_result = self.outlier_handler.validate_data_schema(data, "klines")
            
            if not validation_result["valid"]:
                self.logger.warning(f"Schema validation issues: {validation_result['errors']}")
                
                # Try features schema
                validation_result = self.outlier_handler.validate_data_schema(data, "features")
                if not validation_result["valid"]:
                    self.logger.warning(f"Features schema validation issues: {validation_result['errors']}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Schema validation error: {e}")
            return data
    
    def _validate_data_types(self, data: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Validate and fix data types."""
        try:
            # Check for common data type issues
            for col in data.columns:
                if col == "timestamp" and data[col].dtype != "int64":
                    try:
                        # Try to convert to int64
                        data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
                        self.logger.info(f"Converted {col} to int64")
                    except:
                        self.logger.warning(f"Could not convert {col} to int64")
                
                elif col in ["open", "high", "low", "close", "volume"]:
                    if data[col].dtype not in ["float64", "float32"]:
                        try:
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                            self.logger.info(f"Converted {col} to numeric")
                        except:
                            self.logger.warning(f"Could not convert {col} to numeric")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Data type validation error: {e}")
            return data
    
    def _handle_nulls(self, data: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Handle null values according to rules."""
        try:
            null_handling = rules.get("null_handling", "drop")
            
            if null_handling == "drop":
                original_rows = len(data)
                data = data.dropna()
                rows_removed = original_rows - len(data)
                if rows_removed > 0:
                    self.logger.info(f"Removed {rows_removed} rows with null values")
            
            elif null_handling == "fill":
                # Fill numeric columns with median, categorical with mode
                for col in data.columns:
                    if data[col].dtype in ["float64", "float32", "int64"]:
                        data[col] = data[col].fillna(data[col].median())
                    else:
                        data[col] = data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else "unknown")
                
                self.logger.info("Filled null values with appropriate defaults")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Null handling error: {e}")
            return data
    
    def _handle_duplicates(self, data: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Handle duplicate values according to rules."""
        try:
            duplicate_handling = rules.get("duplicate_handling", "drop_first")
            
            if duplicate_handling == "drop_first":
                original_rows = len(data)
                data = data.drop_duplicates()
                rows_removed = original_rows - len(data)
                if rows_removed > 0:
                    self.logger.info(f"Removed {rows_removed} duplicate rows")
            
            elif duplicate_handling == "drop_last":
                original_rows = len(data)
                data = data.drop_duplicates(keep='first')
                rows_removed = original_rows - len(data)
                if rows_removed > 0:
                    self.logger.info(f"Removed {rows_removed} duplicate rows")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Duplicate handling error: {e}")
            return data
    
    def _handle_outliers(self, data: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Handle outliers according to rules."""
        try:
            outlier_handling = rules.get("outlier_handling", "detect_only")
            outlier_config = rules.get("outlier_config", {})
            
            if outlier_handling == "detect_only":
                # Just detect and log outliers
                outliers = self.outlier_handler.detect_outliers(
                    data, 
                    method=outlier_config.get("method", "iqr"),
                    threshold=outlier_config.get("threshold", 1.5),
                    raise_errors=outlier_config.get("raise_errors", False)
                )
                
                if outliers:
                    self.logger.info(f"Detected {len(outliers)} outlier groups")
                    for outlier in outliers:
                        self.logger.warning(f"  {outlier.column}: {len(outlier.indices)} values, severity={outlier.severity.value}")
            
            elif outlier_handling == "remove":
                # Remove outliers based on severity threshold
                severity_threshold = outlier_config.get("severity_threshold", "medium")
                severity_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
                threshold_level = severity_map.get(severity_threshold, 1)
                
                outliers = self.outlier_handler.detect_outliers(
                    data, 
                    method=outlier_config.get("method", "iqr"),
                    threshold=outlier_config.get("threshold", 1.5),
                    raise_errors=False
                )
                
                # Remove outliers above threshold
                high_severity_outliers = [o for o in outliers if o.severity.value >= threshold_level]
                
                if high_severity_outliers:
                    outlier_indices = set()
                    for outlier in high_severity_outliers:
                        outlier_indices.update(outlier.indices)
                    
                    original_rows = len(data)
                    data = data.drop(data.index[list(outlier_indices)])
                    rows_removed = original_rows - len(data)
                    
                    self.logger.info(f"Removed {rows_removed} rows with {severity_threshold}+ severity outliers")
            
            elif outlier_handling == "cap":
                # Cap outliers at threshold boundaries
                outlier_config = rules.get("outlier_config", {})
                method = outlier_config.get("method", "iqr")
                threshold = outlier_config.get("threshold", 1.5)
                
                for col in data.select_dtypes(include=[np.number]).columns:
                    if method == "iqr":
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        
                        # Cap outliers
                        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                        
                        # Count capped values
                        capped_lower = (data[col] == lower_bound).sum()
                        capped_upper = (data[col] == upper_bound).sum()
                        
                        if capped_lower > 0 or capped_upper > 0:
                            self.logger.info(f"Capped {capped_lower + capped_upper} outliers in {col}")
                    
                    elif method == "zscore":
                        mean_val = data[col].mean()
                        std_val = data[col].std()
                        lower_bound = mean_val - threshold * std_val
                        upper_bound = mean_val + threshold * std_val
                        
                        # Cap outliers
                        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                        
                        # Count capped values
                        capped_lower = (data[col] == lower_bound).sum()
                        capped_upper = (data[col] == upper_bound).sum()
                        
                        if capped_lower > 0 or capped_upper > 0:
                            self.logger.info(f"Capped {capped_lower + capped_upper} outliers in {col}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Outlier handling error: {e}")
            return data
    
    def generate_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality report.
        
        Args:
            data: Data to analyze
            
        Returns:
            Quality report
        """
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "data_shape": data.shape,
                "data_types": data.dtypes.to_dict(),
                "null_analysis": self._analyze_nulls(data),
                "duplicate_analysis": self._analyze_duplicates(data),
                "outlier_analysis": self._analyze_outliers(data),
                "data_quality_score": self._calculate_quality_score(data),
                "recommendations": self._generate_recommendations(data)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating quality report: {e}")
            return {"error": str(e)}
    
    def _analyze_nulls(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze null values in data."""
        try:
            null_counts = data.isnull().sum()
            null_percentages = (null_counts / len(data)) * 100
            
            return {
                "total_null_values": null_counts.sum(),
                "columns_with_nulls": null_counts[null_counts > 0].to_dict(),
                "null_percentages": null_percentages[null_percentages > 0].to_dict(),
                "worst_column": null_counts.idxmax() if null_counts.max() > 0 else None,
                "worst_percentage": null_percentages.max() if null_percentages.max() > 0 else 0
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate values in data."""
        try:
            duplicate_rows = data.duplicated().sum()
            duplicate_percentage = (duplicate_rows / len(data)) * 100
            
            return {
                "duplicate_rows": duplicate_rows,
                "duplicate_percentage": duplicate_percentage,
                "has_duplicates": duplicate_rows > 0
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in data."""
        try:
            # Use default outlier detection
            outliers = self.outlier_handler.detect_outliers(
                data, 
                method="iqr", 
                threshold=1.5,
                raise_errors=False
            )
            
            if not outliers:
                return {"total_outlier_groups": 0, "severity_distribution": {}}
            
            # Analyze severity distribution
            severity_counts = {}
            column_counts = {}
            
            for outlier in outliers:
                severity = outlier.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                column = outlier.column
                if column not in column_counts:
                    column_counts[column] = {"count": 0, "total_values": 0}
                column_counts[column]["count"] += 1
                column_counts[column]["total_values"] += len(outlier.indices)
            
            return {
                "total_outlier_groups": len(outliers),
                "severity_distribution": severity_counts,
                "column_distribution": column_counts,
                "worst_column": max(column_counts.items(), key=lambda x: x[1]["total_values"])[0] if column_counts else None
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)."""
        try:
            score = 100.0
            
            # Deduct for null values
            null_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            score -= null_percentage * 0.5  # 0.5 penalty per percentage point
            
            # Deduct for duplicates
            duplicate_percentage = (data.duplicated().sum() / len(data)) * 100
            score -= duplicate_percentage * 0.3  # 0.3 penalty per percentage point
            
            # Deduct for outliers
            outliers = self.outlier_handler.detect_outliers(data, method="iqr", threshold=1.5, raise_errors=False)
            if outliers:
                critical_outliers = len([o for o in outliers if o.severity == OutlierSeverity.CRITICAL])
                high_outliers = len([o for o in outliers if o.severity == OutlierSeverity.HIGH])
                
                score -= critical_outliers * 5.0  # 5 points per critical outlier group
                score -= high_outliers * 2.0      # 2 points per high outlier group
            
            return max(0.0, score)
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def _generate_recommendations(self, data: pd.DataFrame) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        try:
            # Null value recommendations
            null_analysis = self._analyze_nulls(data)
            if null_analysis.get("worst_percentage", 0) > 10:
                recommendations.append(f"High null percentage in {null_analysis['worst_column']}: {null_analysis['worst_percentage']:.1f}%")
            
            # Duplicate recommendations
            duplicate_analysis = self._analyze_duplicates(data)
            if duplicate_analysis.get("has_duplicates", False):
                recommendations.append(f"Remove {duplicate_analysis['duplicate_rows']} duplicate rows")
            
            # Outlier recommendations
            outlier_analysis = self._analyze_outliers(data)
            if outlier_analysis.get("total_outlier_groups", 0) > 0:
                severity_dist = outlier_analysis.get("severity_distribution", {})
                if severity_dist.get("critical", 0) > 0:
                    recommendations.append("Critical outliers detected - investigate data source")
                if severity_dist.get("high", 0) > 5:
                    recommendations.append("Many high-severity outliers - consider outlier removal")
            
            # Data type recommendations
            for col, dtype in data.dtypes.items():
                if col == "timestamp" and dtype != "int64":
                    recommendations.append(f"Convert {col} to int64 for timestamp consistency")
                elif col in ["open", "high", "low", "close", "volume"] and dtype not in ["float64", "float32"]:
                    recommendations.append(f"Convert {col} to numeric type for calculations")
            
            # Size recommendations
            if len(data) < 1000:
                recommendations.append("Small dataset - consider collecting more data")
            if len(data.columns) > 100:
                recommendations.append("High-dimensional data - consider feature selection")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]


# Global data quality framework instance
data_quality_framework = DataQualityFramework()