# src/training/steps/data_preparation_components/quality_metrics_calculator.py

"""Quality Metrics Calculator - Calculates comprehensive data quality metrics."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import system_logger
from src.utils.pipeline_standards import pipeline_standards


class QualityMetricsCalculator:
    """Calculates comprehensive quality metrics for market data."""
    
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize QualityMetricsCalculator with configuration."""
        self.config = config or self._get_default_config()
        self.logger = system_logger.getChild("QualityMetricsCalculator")
        self.standards = pipeline_standards
        
    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for quality metrics."""
        return {
            "completeness_threshold": 0.95,
            "consistency_threshold": 0.98,
            "timeliness_threshold_hours": 24,
            "accuracy_checks": ["price_range", "volume_validity", "timestamp_order"],
            "uniqueness_columns": ["timestamp"],
            "validity_rules": {
                "price_positive": True,
                "volume_non_negative": True,
                "high_low_consistency": True
            }
        }
    
    async def calculate_all_metrics(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        exchange: str = "UNKNOWN"
    ) -> dict[str, Any]:
        """Calculate all quality metrics for the dataset.
        
        Args:
            data: DataFrame to analyze
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            dict: Comprehensive quality metrics
        """
        metrics = {
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now().isoformat(),
            "data_shape": data.shape if data is not None else (0, 0),
            "dimensions": {},
            "overall_score": 0.0,
            "recommendations": []
        }
        
        try:
            if data is None or data.empty:
                metrics["overall_score"] = 0.0
                metrics["recommendations"].append("No data available for analysis")
                return metrics
            
            # Calculate each dimension
            metrics["dimensions"]["completeness"] = await self.calculate_completeness(data)
            metrics["dimensions"]["consistency"] = await self.calculate_consistency(data)
            metrics["dimensions"]["accuracy"] = await self.calculate_accuracy(data)
            metrics["dimensions"]["timeliness"] = await self.calculate_timeliness(data)
            metrics["dimensions"]["uniqueness"] = await self.calculate_uniqueness(data)
            metrics["dimensions"]["validity"] = await self.calculate_validity(data)
            
            # Calculate overall score
            dimension_scores = [
                dim["score"] for dim in metrics["dimensions"].values()
                if isinstance(dim, dict) and "score" in dim
            ]
            metrics["overall_score"] = np.mean(dimension_scores) if dimension_scores else 0.0
            
            # Generate recommendations
            metrics["recommendations"] = self._generate_recommendations(metrics["dimensions"])
            
            # Add statistical summary
            metrics["statistical_summary"] = self._calculate_statistical_summary(data)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate quality metrics: {e}")
            metrics["error"] = str(e)
            
        return metrics
    
    async def calculate_completeness(self, data: pd.DataFrame) -> dict[str, Any]:
        """Calculate data completeness metrics."""
        completeness = {
            "score": 100.0,
            "missing_values": {},
            "missing_rows": 0,
            "completeness_by_column": {},
            "issues": []
        }
        
        try:
            # Calculate missing values per column
            total_cells = len(data)
            
            for col in data.columns:
                missing_count = data[col].isna().sum()
                missing_pct = (missing_count / total_cells) * 100 if total_cells > 0 else 0
                
                completeness["missing_values"][col] = {
                    "count": int(missing_count),
                    "percentage": round(missing_pct, 2)
                }
                
                col_completeness = 100 - missing_pct
                completeness["completeness_by_column"][col] = round(col_completeness, 2)
                
                if missing_pct > 0:
                    completeness["issues"].append(
                        f"Column '{col}' has {missing_pct:.2f}% missing values"
                    )
            
            # Calculate rows with any missing values
            rows_with_missing = data.isna().any(axis=1).sum()
            completeness["missing_rows"] = int(rows_with_missing)
            
            # Calculate overall completeness score
            total_missing = sum(item["count"] for item in completeness["missing_values"].values())
            total_possible = len(data) * len(data.columns)
            completeness["score"] = round(
                (1 - total_missing / total_possible) * 100 if total_possible > 0 else 100,
                2
            )
            
            # Check against threshold
            if completeness["score"] < self.config["completeness_threshold"] * 100:
                completeness["issues"].append(
                    f"Overall completeness {completeness['score']:.2f}% is below "
                    f"threshold {self.config['completeness_threshold'] * 100}%"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to calculate completeness: {e}")
            completeness["error"] = str(e)
            
        return completeness
    
    async def calculate_consistency(self, data: pd.DataFrame) -> dict[str, Any]:
        """Calculate data consistency metrics."""
        consistency = {
            "score": 100.0,
            "inconsistencies": {},
            "format_issues": {},
            "value_range_issues": {},
            "issues": []
        }
        
        try:
            inconsistency_count = 0
            total_checks = 0
            
            # Check OHLC consistency
            if all(col in data.columns for col in ["open", "high", "low", "close"]):
                total_checks += len(data)
                
                # Low should not exceed high
                low_high_issues = (data["low"] > data["high"]).sum()
                if low_high_issues > 0:
                    inconsistency_count += low_high_issues
                    consistency["inconsistencies"]["low_exceeds_high"] = int(low_high_issues)
                    consistency["issues"].append(
                        f"Found {low_high_issues} rows where low > high"
                    )
                
                # Low should be <= all prices
                low_price_issues = (
                    (data["low"] > data["open"]) |
                    (data["low"] > data["close"])
                ).sum()
                if low_price_issues > 0:
                    inconsistency_count += low_price_issues
                    consistency["inconsistencies"]["low_exceeds_prices"] = int(low_price_issues)
                
                # High should be >= all prices
                high_price_issues = (
                    (data["high"] < data["open"]) |
                    (data["high"] < data["close"])
                ).sum()
                if high_price_issues > 0:
                    inconsistency_count += high_price_issues
                    consistency["inconsistencies"]["high_below_prices"] = int(high_price_issues)
            
            # Check data type consistency
            for col in data.columns:
                if col in ["open", "high", "low", "close", "volume"]:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        consistency["format_issues"][col] = f"Non-numeric type: {data[col].dtype}"
                        consistency["issues"].append(
                            f"Column '{col}' has non-numeric type: {data[col].dtype}"
                        )
            
            # Check timestamp consistency
            if isinstance(data.index, pd.DatetimeIndex):
                # Check for duplicate timestamps
                duplicates = data.index.duplicated().sum()
                if duplicates > 0:
                    consistency["inconsistencies"]["duplicate_timestamps"] = int(duplicates)
                    consistency["issues"].append(f"Found {duplicates} duplicate timestamps")
                    inconsistency_count += duplicates
                    total_checks += len(data)
                
                # Check for non-monotonic timestamps
                if not data.index.is_monotonic_increasing:
                    consistency["issues"].append("Timestamps are not monotonic increasing")
                    inconsistency_count += 1
                    total_checks += 1
            
            # Calculate consistency score
            if total_checks > 0:
                consistency["score"] = round(
                    (1 - inconsistency_count / total_checks) * 100,
                    2
                )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate consistency: {e}")
            consistency["error"] = str(e)
            
        return consistency
    
    async def calculate_accuracy(self, data: pd.DataFrame) -> dict[str, Any]:
        """Calculate data accuracy metrics."""
        accuracy = {
            "score": 100.0,
            "outliers": {},
            "suspicious_values": {},
            "statistical_anomalies": {},
            "issues": []
        }
        
        try:
            accuracy_scores = []
            
            # Check for outliers using IQR method
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                if col not in data.columns:
                    continue
                
                col_data = data[col].dropna()
                if len(col_data) == 0:
                    continue
                
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                outlier_pct = (outliers / len(col_data)) * 100
                
                if outliers > 0:
                    accuracy["outliers"][col] = {
                        "count": int(outliers),
                        "percentage": round(outlier_pct, 2),
                        "bounds": {
                            "lower": float(lower_bound),
                            "upper": float(upper_bound)
                        }
                    }
                    
                    if outlier_pct > 5:  # More than 5% outliers is suspicious
                        accuracy["issues"].append(
                            f"Column '{col}' has {outlier_pct:.2f}% outliers"
                        )
                
                # Calculate column accuracy score (fewer outliers = higher accuracy)
                col_accuracy = max(0, 100 - outlier_pct * 2)  # 2x penalty for outliers
                accuracy_scores.append(col_accuracy)
            
            # Check for suspicious patterns
            if "volume" in data.columns:
                zero_volume_pct = (data["volume"] == 0).sum() / len(data) * 100
                if zero_volume_pct > 10:
                    accuracy["suspicious_values"]["zero_volume_percentage"] = round(zero_volume_pct, 2)
                    accuracy["issues"].append(
                        f"High percentage of zero volume: {zero_volume_pct:.2f}%"
                    )
                    accuracy_scores.append(max(0, 100 - zero_volume_pct))
            
            # Check for price spikes
            if "close" in data.columns:
                price_changes = data["close"].pct_change().abs()
                extreme_changes = (price_changes > 0.2).sum()  # 20% changes
                if extreme_changes > 0:
                    accuracy["suspicious_values"]["extreme_price_changes"] = int(extreme_changes)
                    accuracy["issues"].append(
                        f"Found {extreme_changes} extreme price changes (>20%)"
                    )
            
            # Calculate overall accuracy score
            if accuracy_scores:
                accuracy["score"] = round(np.mean(accuracy_scores), 2)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate accuracy: {e}")
            accuracy["error"] = str(e)
            
        return accuracy
    
    async def calculate_timeliness(self, data: pd.DataFrame) -> dict[str, Any]:
        """Calculate data timeliness metrics."""
        timeliness = {
            "score": 100.0,
            "data_age_hours": None,
            "last_update": None,
            "update_frequency": None,
            "gaps": [],
            "issues": []
        }
        
        try:
            if isinstance(data.index, pd.DatetimeIndex) and len(data) > 0:
                # Calculate data age
                last_timestamp = data.index.max()
                current_time = datetime.now(last_timestamp.tzinfo) if last_timestamp.tzinfo else datetime.now()
                data_age = current_time - last_timestamp
                timeliness["data_age_hours"] = round(data_age.total_seconds() / 3600, 2)
                timeliness["last_update"] = last_timestamp.isoformat()
                
                # Score based on age
                if timeliness["data_age_hours"] <= self.config["timeliness_threshold_hours"]:
                    timeliness["score"] = 100.0
                else:
                    # Decay score based on how old the data is
                    overtime = timeliness["data_age_hours"] - self.config["timeliness_threshold_hours"]
                    timeliness["score"] = max(0, 100 - (overtime / 24) * 10)  # -10 points per day
                
                if timeliness["data_age_hours"] > self.config["timeliness_threshold_hours"]:
                    timeliness["issues"].append(
                        f"Data is {timeliness['data_age_hours']:.2f} hours old, "
                        f"exceeds threshold of {self.config['timeliness_threshold_hours']} hours"
                    )
                
                # Calculate update frequency
                if len(data) > 1:
                    time_diffs = data.index.to_series().diff().dropna()
                    mode_interval = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
                    timeliness["update_frequency"] = str(mode_interval)
                    
                    # Find significant gaps
                    significant_gaps = time_diffs[time_diffs > mode_interval * 3]
                    if len(significant_gaps) > 0:
                        for idx, gap in significant_gaps.items():
                            gap_info = {
                                "timestamp": idx.isoformat(),
                                "gap_duration": str(gap),
                                "gap_hours": round(gap.total_seconds() / 3600, 2)
                            }
                            timeliness["gaps"].append(gap_info)
                        
                        timeliness["issues"].append(
                            f"Found {len(significant_gaps)} significant time gaps"
                        )
                        # Penalize score for gaps
                        timeliness["score"] *= (1 - len(significant_gaps) / len(data))
                        
            else:
                timeliness["score"] = 0.0
                timeliness["issues"].append("No datetime index available for timeliness check")
                
        except Exception as e:
            self.logger.error(f"Failed to calculate timeliness: {e}")
            timeliness["error"] = str(e)
            
        return timeliness
    
    async def calculate_uniqueness(self, data: pd.DataFrame) -> dict[str, Any]:
        """Calculate data uniqueness metrics."""
        uniqueness = {
            "score": 100.0,
            "duplicate_rows": 0,
            "duplicate_indices": 0,
            "uniqueness_by_column": {},
            "issues": []
        }
        
        try:
            # Check for duplicate rows
            duplicates = data.duplicated()
            uniqueness["duplicate_rows"] = int(duplicates.sum())
            
            if uniqueness["duplicate_rows"] > 0:
                dup_pct = (uniqueness["duplicate_rows"] / len(data)) * 100
                uniqueness["issues"].append(
                    f"Found {uniqueness['duplicate_rows']} duplicate rows ({dup_pct:.2f}%)"
                )
                uniqueness["score"] *= (1 - dup_pct / 100)
            
            # Check for duplicate indices
            if hasattr(data.index, 'duplicated'):
                index_duplicates = data.index.duplicated()
                uniqueness["duplicate_indices"] = int(index_duplicates.sum())
                
                if uniqueness["duplicate_indices"] > 0:
                    uniqueness["issues"].append(
                        f"Found {uniqueness['duplicate_indices']} duplicate index values"
                    )
                    uniqueness["score"] *= 0.8  # Significant penalty for duplicate indices
            
            # Calculate uniqueness for each column
            for col in data.columns:
                unique_values = data[col].nunique()
                total_values = len(data[col].dropna())
                uniqueness_ratio = unique_values / total_values if total_values > 0 else 0
                
                uniqueness["uniqueness_by_column"][col] = {
                    "unique_values": int(unique_values),
                    "total_values": int(total_values),
                    "uniqueness_ratio": round(uniqueness_ratio, 4)
                }
                
                # Check if column should be unique
                if col in self.config.get("uniqueness_columns", []):
                    if uniqueness_ratio < 1.0:
                        uniqueness["issues"].append(
                            f"Column '{col}' should be unique but has duplicates"
                        )
                        uniqueness["score"] *= uniqueness_ratio
                        
        except Exception as e:
            self.logger.error(f"Failed to calculate uniqueness: {e}")
            uniqueness["error"] = str(e)
            
        return uniqueness
    
    async def calculate_validity(self, data: pd.DataFrame) -> dict[str, Any]:
        """Calculate data validity metrics."""
        validity = {
            "score": 100.0,
            "invalid_values": {},
            "validation_results": {},
            "issues": []
        }
        
        try:
            validity_checks_passed = 0
            validity_checks_total = 0
            
            # Apply validity rules from config
            for rule_name, rule_enabled in self.config["validity_rules"].items():
                if not rule_enabled:
                    continue
                
                validity_checks_total += 1
                
                if rule_name == "price_positive":
                    # Check that prices are positive
                    price_cols = ["open", "high", "low", "close"]
                    for col in price_cols:
                        if col in data.columns:
                            negative_count = (data[col] < 0).sum()
                            if negative_count > 0:
                                validity["invalid_values"][f"{col}_negative"] = int(negative_count)
                                validity["issues"].append(
                                    f"Column '{col}' has {negative_count} negative values"
                                )
                            else:
                                validity_checks_passed += 0.25  # Each price column counts as 1/4
                    else:
                        validity_checks_passed += 1
                        
                elif rule_name == "volume_non_negative":
                    # Check that volume is non-negative
                    if "volume" in data.columns:
                        negative_volume = (data["volume"] < 0).sum()
                        if negative_volume > 0:
                            validity["invalid_values"]["negative_volume"] = int(negative_volume)
                            validity["issues"].append(
                                f"Volume has {negative_volume} negative values"
                            )
                        else:
                            validity_checks_passed += 1
                    else:
                        validity_checks_passed += 1
                        
                elif rule_name == "high_low_consistency":
                    # Already checked in consistency, but include for completeness
                    if all(col in data.columns for col in ["high", "low"]):
                        invalid = (data["high"] < data["low"]).sum()
                        if invalid > 0:
                            validity["invalid_values"]["high_less_than_low"] = int(invalid)
                            validity["issues"].append(
                                f"Found {invalid} rows where high < low"
                            )
                        else:
                            validity_checks_passed += 1
                    else:
                        validity_checks_passed += 1
            
            # Calculate validity score
            if validity_checks_total > 0:
                validity["score"] = round(
                    (validity_checks_passed / validity_checks_total) * 100,
                    2
                )
            
            # Add validation summary
            validity["validation_results"]["total_checks"] = validity_checks_total
            validity["validation_results"]["passed_checks"] = validity_checks_passed
            validity["validation_results"]["failed_checks"] = validity_checks_total - validity_checks_passed
            
        except Exception as e:
            self.logger.error(f"Failed to calculate validity: {e}")
            validity["error"] = str(e)
            
        return validity
    
    def _calculate_statistical_summary(self, data: pd.DataFrame) -> dict[str, Any]:
        """Calculate statistical summary of the data."""
        summary = {}
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) == 0:
                    continue
                
                summary[col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "median": float(col_data.median()),
                    "q1": float(col_data.quantile(0.25)),
                    "q3": float(col_data.quantile(0.75)),
                    "skewness": float(stats.skew(col_data)),
                    "kurtosis": float(stats.kurtosis(col_data))
                }
                
        except Exception as e:
            self.logger.error(f"Failed to calculate statistical summary: {e}")
            
        return summary
    
    def _generate_recommendations(self, dimensions: dict[str, Any]) -> list[str]:
        """Generate recommendations based on quality metrics."""
        recommendations = []
        
        try:
            # Check each dimension for issues
            for dim_name, dim_data in dimensions.items():
                if isinstance(dim_data, dict):
                    score = dim_data.get("score", 100)
                    issues = dim_data.get("issues", [])
                    
                    if score < 80:
                        recommendations.append(
                            f"Improve {dim_name}: score is {score:.1f}% (target: 80%+)"
                        )
                    
                    # Add specific recommendations based on issues
                    if dim_name == "completeness" and issues:
                        recommendations.append(
                            "Consider using forward-fill or interpolation for missing values"
                        )
                    elif dim_name == "consistency" and issues:
                        recommendations.append(
                            "Fix OHLC price inconsistencies and ensure data types are correct"
                        )
                    elif dim_name == "accuracy" and "outliers" in dim_data:
                        recommendations.append(
                            "Review and handle outliers using statistical methods"
                        )
                    elif dim_name == "timeliness" and dim_data.get("data_age_hours", 0) > 24:
                        recommendations.append(
                            "Update data source to get more recent data"
                        )
                    elif dim_name == "uniqueness" and dim_data.get("duplicate_rows", 0) > 0:
                        recommendations.append(
                            "Remove duplicate rows to improve data quality"
                        )
                    elif dim_name == "validity" and issues:
                        recommendations.append(
                            "Fix invalid values according to business rules"
                        )
                        
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            
        return recommendations
    
    async def generate_quality_report(
        self,
        data: pd.DataFrame,
        output_format: str = "dict"
    ) -> Any:
        """Generate a comprehensive quality report.
        
        Args:
            data: DataFrame to analyze
            output_format: Output format ('dict', 'json', 'html')
            
        Returns:
            Quality report in specified format
        """
        try:
            # Calculate all metrics
            metrics = await self.calculate_all_metrics(data)
            
            if output_format == "dict":
                return metrics
            
            elif output_format == "json":
                import json
                return json.dumps(metrics, indent=2, default=str)
            
            elif output_format == "html":
                # Generate HTML report
                html = self._generate_html_report(metrics)
                return html
            
            else:
                self.logger.warning(f"Unknown output format: {output_format}, returning dict")
                return metrics
                
        except Exception as e:
            self.logger.error(f"Failed to generate quality report: {e}")
            return {"error": str(e)}
    
    def _generate_html_report(self, metrics: dict[str, Any]) -> str:
        """Generate HTML quality report."""
        html = f"""
        <html>
        <head>
            <title>Data Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin-bottom: 20px; }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .bad {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Data Quality Report</h1>
            <p>Symbol: {metrics.get('symbol', 'N/A')} | Exchange: {metrics.get('exchange', 'N/A')}</p>
            <p>Generated: {metrics.get('timestamp', 'N/A')}</p>
            
            <div class="metric">
                <h2>Overall Quality Score</h2>
                <div class="score {self._get_score_class(metrics.get('overall_score', 0))}">
                    {metrics.get('overall_score', 0):.1f}%
                </div>
            </div>
        """
        
        # Add dimension scores
        if "dimensions" in metrics:
            html += "<h2>Quality Dimensions</h2><table>"
            html += "<tr><th>Dimension</th><th>Score</th><th>Issues</th></tr>"
            
            for dim_name, dim_data in metrics["dimensions"].items():
                if isinstance(dim_data, dict):
                    score = dim_data.get("score", 0)
                    issues = len(dim_data.get("issues", []))
                    score_class = self._get_score_class(score)
                    
                    html += f"""
                    <tr>
                        <td>{dim_name.capitalize()}</td>
                        <td class="{score_class}">{score:.1f}%</td>
                        <td>{issues} issues</td>
                    </tr>
                    """
            
            html += "</table>"
        
        # Add recommendations
        if "recommendations" in metrics and metrics["recommendations"]:
            html += "<h2>Recommendations</h2><ul>"
            for rec in metrics["recommendations"]:
                html += f"<li>{rec}</li>"
            html += "</ul>"
        
        html += "</body></html>"
        return html
    
    def _get_score_class(self, score: float) -> str:
        """Get CSS class based on score."""
        if score >= 80:
            return "good"
        elif score >= 60:
            return "warning"
        else:
            return "bad"