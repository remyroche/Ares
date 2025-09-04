"""Quality Metrics Calculator Component
Calculates various data quality metrics and scores for market data.
Extracted from raw_data_quality_checker.py
"""
from typing import Any, Optional
import pandas as pd
import numpy as np
from datetime import timedelta
from src.utils.logger import system_logger


class QualityMetricsCalculator:
    """Calculates comprehensive quality metrics for market data.
    
    This class provides functionality for:
    - Calculating overall data quality scores
    - Computing detailed metrics for different quality aspects
    - Generating quality reports
    - Tracking quality trends over time
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.logger = system_logger.getChild("QualityMetricsCalculator")
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for quality metrics."""
        return {
            "score_weights": {
                "completeness": 0.25,
                "consistency": 0.25,
                "timeliness": 0.20,
                "validity": 0.20,
                "accuracy": 0.10
            },
            "thresholds": {
                "critical_issue_penalty": 0.3,
                "warning_penalty": 0.05,
                "missing_data_penalty": 0.1,
                "outlier_penalty": 0.02
            }
        }
        
    def calculate_quality_score(self, results: dict[str, Any]) -> float:
        """
        Calculate overall data quality score based on various metrics.
        
        Args:
            results: Dictionary containing quality check results
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        # Base score starts at 1.0
        score = 1.0
        
        # Deduct points for critical issues
        critical_penalty = self.config["thresholds"]["critical_issue_penalty"]
        score -= len(results.get("critical_issues", [])) * critical_penalty
        
        # Deduct points for warnings (less severe)
        warning_penalty = self.config["thresholds"]["warning_penalty"]
        score -= len(results.get("warnings", [])) * warning_penalty
        
        # Additional penalties from detailed analysis
        if "detailed_analysis" in results:
            # Missing data penalty
            if "missing_ratio" in results["detailed_analysis"]:
                missing_ratio = results["detailed_analysis"]["missing_ratio"]
                score -= missing_ratio * self.config["thresholds"]["missing_data_penalty"]
                
            # Outlier penalty
            if "outlier_ratio" in results["detailed_analysis"]:
                outlier_ratio = results["detailed_analysis"]["outlier_ratio"]
                score -= outlier_ratio * self.config["thresholds"]["outlier_penalty"]
        
        # Ensure score doesn't go below 0
        return max(0.0, score)
        
    def calculate_completeness_metrics(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Calculate data completeness metrics.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with completeness metrics
        """
        metrics = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "missing_values": {},
            "completeness_by_column": {},
            "overall_completeness": 0.0
        }
        
        # Calculate missing values per column
        for col in data.columns:
            missing_count = data[col].isna().sum()
            missing_ratio = missing_count / len(data) if len(data) > 0 else 0
            
            metrics["missing_values"][col] = {
                "count": int(missing_count),
                "ratio": float(missing_ratio)
            }
            metrics["completeness_by_column"][col] = float(1 - missing_ratio)
        
        # Overall completeness
        total_cells = len(data) * len(data.columns)
        total_missing = sum(m["count"] for m in metrics["missing_values"].values())
        metrics["overall_completeness"] = float(1 - (total_missing / total_cells)) if total_cells > 0 else 0
        
        return metrics
        
    def calculate_consistency_metrics(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Calculate data consistency metrics.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with consistency metrics
        """
        metrics = {
            "ohlc_consistency": {},
            "volume_consistency": {},
            "price_consistency": {},
            "overall_consistency": 0.0
        }
        
        # Check OHLC consistency
        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            ohlc_inconsistent = (
                (data["high"] < data["low"]) |
                (data["open"] > data["high"]) |
                (data["close"] > data["high"]) |
                (data["open"] < data["low"]) |
                (data["close"] < data["low"])
            )
            
            metrics["ohlc_consistency"] = {
                "inconsistent_count": int(ohlc_inconsistent.sum()),
                "inconsistent_ratio": float(ohlc_inconsistent.sum() / len(data)) if len(data) > 0 else 0,
                "consistent_ratio": float(1 - (ohlc_inconsistent.sum() / len(data))) if len(data) > 0 else 1
            }
            
        # Check volume consistency (non-negative)
        if "volume" in data.columns:
            negative_volume = data["volume"] < 0
            zero_volume = data["volume"] == 0
            
            metrics["volume_consistency"] = {
                "negative_count": int(negative_volume.sum()),
                "zero_count": int(zero_volume.sum()),
                "negative_ratio": float(negative_volume.sum() / len(data)) if len(data) > 0 else 0,
                "zero_ratio": float(zero_volume.sum() / len(data)) if len(data) > 0 else 0
            }
            
        # Check price consistency (positive prices)
        price_cols = [col for col in ["open", "high", "low", "close"] if col in data.columns]
        if price_cols:
            negative_prices = (data[price_cols] < 0).any(axis=1)
            zero_prices = (data[price_cols] == 0).any(axis=1)
            
            metrics["price_consistency"] = {
                "negative_count": int(negative_prices.sum()),
                "zero_count": int(zero_prices.sum()),
                "negative_ratio": float(negative_prices.sum() / len(data)) if len(data) > 0 else 0,
                "zero_ratio": float(zero_prices.sum() / len(data)) if len(data) > 0 else 0
            }
            
        # Calculate overall consistency score
        consistency_scores = []
        if metrics["ohlc_consistency"]:
            consistency_scores.append(metrics["ohlc_consistency"]["consistent_ratio"])
        if metrics["volume_consistency"]:
            consistency_scores.append(1 - metrics["volume_consistency"]["negative_ratio"])
        if metrics["price_consistency"]:
            consistency_scores.append(1 - metrics["price_consistency"]["negative_ratio"])
            
        metrics["overall_consistency"] = float(np.mean(consistency_scores)) if consistency_scores else 0.0
        
        return metrics
        
    def calculate_timeliness_metrics(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Calculate data timeliness metrics.
        
        Args:
            data: DataFrame with datetime index
            
        Returns:
            Dictionary with timeliness metrics
        """
        metrics = {
            "data_age": {},
            "update_frequency": {},
            "gap_analysis": {},
            "overall_timeliness": 0.0
        }
        
        if isinstance(data.index, pd.DatetimeIndex):
            # Data age
            now = pd.Timestamp.now(tz=data.index.tz)
            latest_timestamp = data.index.max()
            oldest_timestamp = data.index.min()
            
            metrics["data_age"] = {
                "latest_data_age_hours": float((now - latest_timestamp).total_seconds() / 3600),
                "oldest_data_age_days": float((now - oldest_timestamp).total_seconds() / 86400),
                "data_span_days": float((latest_timestamp - oldest_timestamp).total_seconds() / 86400)
            }
            
            # Update frequency
            time_diffs = data.index.to_series().diff().dropna()
            
            metrics["update_frequency"] = {
                "mean_interval_seconds": float(time_diffs.mean().total_seconds()),
                "median_interval_seconds": float(time_diffs.median().total_seconds()),
                "mode_interval_seconds": float(time_diffs.mode()[0].total_seconds()) if len(time_diffs.mode()) > 0 else None,
                "std_interval_seconds": float(time_diffs.std().total_seconds())
            }
            
            # Gap analysis
            expected_interval = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
            gaps = time_diffs[time_diffs > expected_interval * 2]
            
            metrics["gap_analysis"] = {
                "gap_count": int(len(gaps)),
                "gap_ratio": float(len(gaps) / len(time_diffs)) if len(time_diffs) > 0 else 0,
                "max_gap_hours": float(gaps.max().total_seconds() / 3600) if len(gaps) > 0 else 0,
                "total_gap_hours": float(gaps.sum().total_seconds() / 3600) if len(gaps) > 0 else 0
            }
            
            # Calculate timeliness score
            age_penalty = min(metrics["data_age"]["latest_data_age_hours"] / 24, 1.0)  # Penalty for old data
            gap_penalty = metrics["gap_analysis"]["gap_ratio"]
            metrics["overall_timeliness"] = float(max(0, 1 - (age_penalty * 0.5 + gap_penalty * 0.5)))
            
        return metrics
        
    def calculate_validity_metrics(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Calculate data validity metrics.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with validity metrics
        """
        metrics = {
            "data_type_validity": {},
            "range_validity": {},
            "format_validity": {},
            "overall_validity": 0.0
        }
        
        validity_scores = []
        
        # Check data types
        expected_types = {
            "open": [np.number],
            "high": [np.number],
            "low": [np.number],
            "close": [np.number],
            "volume": [np.number]
        }
        
        for col, expected in expected_types.items():
            if col in data.columns:
                is_valid_type = any(np.issubdtype(data[col].dtype, t) for t in expected)
                metrics["data_type_validity"][col] = is_valid_type
                validity_scores.append(1.0 if is_valid_type else 0.0)
                
        # Check value ranges
        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            # Prices should be positive
            positive_prices = (data[["open", "high", "low", "close"]] > 0).all(axis=1)
            metrics["range_validity"]["positive_prices_ratio"] = float(positive_prices.sum() / len(data)) if len(data) > 0 else 0
            validity_scores.append(metrics["range_validity"]["positive_prices_ratio"])
            
        if "volume" in data.columns:
            # Volume should be non-negative
            non_negative_volume = data["volume"] >= 0
            metrics["range_validity"]["non_negative_volume_ratio"] = float(non_negative_volume.sum() / len(data)) if len(data) > 0 else 0
            validity_scores.append(metrics["range_validity"]["non_negative_volume_ratio"])
            
        # Overall validity
        metrics["overall_validity"] = float(np.mean(validity_scores)) if validity_scores else 0.0
        
        return metrics
        
    def generate_quality_report(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        exchange: str,
        include_recommendations: bool = True
    ) -> dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Args:
            data: DataFrame to analyze
            symbol: Trading symbol
            exchange: Exchange name
            include_recommendations: Whether to include improvement recommendations
            
        Returns:
            Comprehensive quality report
        """
        report = {
            "metadata": {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": pd.Timestamp.now().isoformat(),
                "data_shape": data.shape,
                "date_range": {
                    "start": str(data.index.min()) if isinstance(data.index, pd.DatetimeIndex) else None,
                    "end": str(data.index.max()) if isinstance(data.index, pd.DatetimeIndex) else None
                }
            },
            "metrics": {},
            "overall_score": 0.0,
            "recommendations": []
        }
        
        # Calculate all metrics
        report["metrics"]["completeness"] = self.calculate_completeness_metrics(data)
        report["metrics"]["consistency"] = self.calculate_consistency_metrics(data)
        report["metrics"]["timeliness"] = self.calculate_timeliness_metrics(data)
        report["metrics"]["validity"] = self.calculate_validity_metrics(data)
        
        # Calculate weighted overall score
        weights = self.config["score_weights"]
        weighted_scores = []
        
        for metric_type, weight in weights.items():
            if metric_type in report["metrics"] and f"overall_{metric_type}" in report["metrics"][metric_type]:
                score = report["metrics"][metric_type][f"overall_{metric_type}"]
                weighted_scores.append(score * weight)
                
        report["overall_score"] = float(sum(weighted_scores))
        
        # Generate recommendations
        if include_recommendations:
            report["recommendations"] = self._generate_recommendations(report["metrics"])
            
        return report
        
    def _generate_recommendations(self, metrics: dict[str, Any]) -> list[str]:
        """Generate recommendations based on quality metrics."""
        recommendations = []
        
        # Completeness recommendations
        if "completeness" in metrics:
            if metrics["completeness"]["overall_completeness"] < 0.95:
                recommendations.append(
                    f"Data completeness is {metrics['completeness']['overall_completeness']:.1%}. "
                    "Consider filling missing values or investigating data collection issues."
                )
                
        # Consistency recommendations
        if "consistency" in metrics:
            if metrics["consistency"].get("ohlc_consistency", {}).get("inconsistent_ratio", 0) > 0.001:
                recommendations.append(
                    "OHLC consistency issues detected. Review data source and validation logic."
                )
                
        # Timeliness recommendations
        if "timeliness" in metrics:
            if metrics["timeliness"].get("data_age", {}).get("latest_data_age_hours", 0) > 24:
                recommendations.append(
                    "Data is more than 24 hours old. Consider updating to more recent data."
                )
                
        # Validity recommendations
        if "validity" in metrics:
            if metrics["validity"]["overall_validity"] < 0.95:
                recommendations.append(
                    "Data validity issues detected. Review data types and value ranges."
                )
                
        return recommendations