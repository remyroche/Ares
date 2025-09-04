"""Anomaly Detector Component
Detects various types of anomalies in market data.
Extracted from raw_data_quality_checker.py
"""
from typing import Any, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
from datetime import timedelta
from src.utils.logger import system_logger


class AnomalyDetector:
    """Detects anomalies in market data using multiple detection methods.
    
    This class provides functionality for:
    - Statistical anomaly detection
    - Pattern-based anomaly detection
    - Time-based anomaly detection
    - Volume anomaly detection
    - Multi-dimensional anomaly detection
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.logger = system_logger.getChild("AnomalyDetector")
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for anomaly detection."""
        return {
            "detection_methods": {
                "statistical": True,
                "isolation_forest": False,  # Requires sklearn
                "local_outlier_factor": False,  # Requires sklearn
                "pattern_based": True,
                "time_based": True
            },
            "statistical_params": {
                "zscore_threshold": 3.0,
                "iqr_multiplier": 1.5,
                "mad_threshold": 3.0,  # Median Absolute Deviation
                "rolling_window": 20
            },
            "pattern_params": {
                "min_pattern_length": 3,
                "similarity_threshold": 0.95
            },
            "volume_params": {
                "spike_threshold": 5.0,  # Times the average
                "drop_threshold": 0.1,  # Times the average
                "rolling_window": 20
            }
        }
        
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        methods: Optional[List[str]] = None
    ) -> dict[str, Any]:
        """
        Detect anomalies using multiple methods.
        
        Args:
            data: DataFrame with market data
            columns: Columns to check for anomalies (None = all numeric)
            methods: Detection methods to use (None = use config defaults)
            
        Returns:
            Dictionary with anomaly detection results
        """
        results = {
            "anomalies": {},
            "summary": {},
            "detailed_analysis": {},
            "recommendations": []
        }
        
        # Determine columns to analyze
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            columns = [col for col in columns if col not in ["timestamp", "year", "month", "day"]]
            
        # Determine methods to use
        if methods is None:
            methods = [method for method, enabled in self.config["detection_methods"].items() if enabled]
            
        # Apply each detection method
        for method in methods:
            if method == "statistical":
                self._detect_statistical_anomalies(data, columns, results)
            elif method == "pattern_based":
                self._detect_pattern_anomalies(data, columns, results)
            elif method == "time_based":
                self._detect_time_based_anomalies(data, results)
                
        # Generate summary
        self._generate_anomaly_summary(results)
        
        # Generate recommendations
        results["recommendations"] = self._generate_anomaly_recommendations(results)
        
        return results
        
    def _detect_statistical_anomalies(
        self,
        data: pd.DataFrame,
        columns: List[str],
        results: dict[str, Any]
    ) -> None:
        """Detect statistical anomalies using z-score, IQR, and MAD methods."""
        self.logger.info("Detecting statistical anomalies...")
        
        for col in columns:
            if col not in data.columns:
                continue
                
            col_data = data[col].dropna()
            if len(col_data) < 10:  # Need minimum data points
                continue
                
            anomalies = {
                "zscore": [],
                "iqr": [],
                "mad": [],
                "indices": []
            }
            
            # Z-score method
            if self.config["statistical_params"]["zscore_threshold"] > 0:
                mean = col_data.mean()
                std = col_data.std()
                if std > 0:
                    z_scores = np.abs((col_data - mean) / std)
                    zscore_anomalies = z_scores > self.config["statistical_params"]["zscore_threshold"]
                    anomalies["zscore"] = col_data.index[zscore_anomalies].tolist()
                    
            # IQR method
            if self.config["statistical_params"]["iqr_multiplier"] > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                multiplier = self.config["statistical_params"]["iqr_multiplier"]
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                iqr_anomalies = (col_data < lower_bound) | (col_data > upper_bound)
                anomalies["iqr"] = col_data.index[iqr_anomalies].tolist()
                
            # MAD (Median Absolute Deviation) method
            if self.config["statistical_params"]["mad_threshold"] > 0:
                median = col_data.median()
                mad = np.median(np.abs(col_data - median))
                if mad > 0:
                    modified_z_scores = 0.6745 * (col_data - median) / mad
                    mad_anomalies = np.abs(modified_z_scores) > self.config["statistical_params"]["mad_threshold"]
                    anomalies["mad"] = col_data.index[mad_anomalies].tolist()
                    
            # Combine all anomaly indices
            all_indices = set()
            for method_indices in anomalies.values():
                if isinstance(method_indices, list):
                    all_indices.update(method_indices)
                    
            anomalies["indices"] = sorted(list(all_indices))
            
            if anomalies["indices"]:
                results["anomalies"][col] = anomalies
                
    def _detect_pattern_anomalies(
        self,
        data: pd.DataFrame,
        columns: List[str],
        results: dict[str, Any]
    ) -> None:
        """Detect anomalies based on unusual patterns."""
        self.logger.info("Detecting pattern-based anomalies...")
        
        pattern_anomalies = {}
        
        for col in columns:
            if col not in data.columns:
                continue
                
            # Detect sudden spikes or drops
            if col in ["volume", "close"]:
                rolling_mean = data[col].rolling(
                    window=self.config["statistical_params"]["rolling_window"]
                ).mean()
                rolling_std = data[col].rolling(
                    window=self.config["statistical_params"]["rolling_window"]
                ).std()
                
                # Detect values far from rolling average
                if rolling_mean is not None and rolling_std is not None:
                    deviations = np.abs(data[col] - rolling_mean) / rolling_std
                    pattern_anomaly_mask = deviations > 3  # 3 standard deviations
                    pattern_anomaly_indices = data.index[pattern_anomaly_mask.fillna(False)].tolist()
                    
                    if pattern_anomaly_indices:
                        pattern_anomalies[col] = {
                            "type": "sudden_change",
                            "indices": pattern_anomaly_indices,
                            "count": len(pattern_anomaly_indices)
                        }
                        
        if pattern_anomalies:
            results["detailed_analysis"]["pattern_anomalies"] = pattern_anomalies
            
    def _detect_time_based_anomalies(
        self,
        data: pd.DataFrame,
        results: dict[str, Any]
    ) -> None:
        """Detect time-based anomalies like unusual trading hours or gaps."""
        self.logger.info("Detecting time-based anomalies...")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            return
            
        time_anomalies = {}
        
        # Detect unusual trading hours
        trading_hours = data.index.hour
        unusual_hours = (trading_hours < 6) | (trading_hours > 22)  # Assuming normal hours 6 AM - 10 PM
        
        if unusual_hours.any():
            time_anomalies["unusual_hours"] = {
                "count": unusual_hours.sum(),
                "percentage": float(unusual_hours.sum() / len(data) * 100),
                "sample_times": data.index[unusual_hours][:10].tolist()
            }
            
        # Detect time gaps
        time_diffs = data.index.to_series().diff()
        expected_interval = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
        
        if pd.notna(expected_interval):
            significant_gaps = time_diffs > expected_interval * 10
            
            if significant_gaps.any():
                gap_indices = data.index[significant_gaps]
                time_anomalies["time_gaps"] = {
                    "count": len(gap_indices),
                    "gap_times": [
                        {
                            "start": str(data.index[i-1]),
                            "end": str(data.index[i]),
                            "duration_hours": float(time_diffs.iloc[i].total_seconds() / 3600)
                        }
                        for i in range(len(data))
                        if significant_gaps.iloc[i] and i > 0
                    ][:10]  # First 10 gaps
                }
                
        if time_anomalies:
            results["detailed_analysis"]["time_anomalies"] = time_anomalies
            
    def detect_volume_anomalies(
        self,
        data: pd.DataFrame,
        volume_col: str = "volume"
    ) -> dict[str, Any]:
        """
        Detect volume-specific anomalies.
        
        Args:
            data: DataFrame with volume data
            volume_col: Name of volume column
            
        Returns:
            Volume anomaly detection results
        """
        results = {
            "volume_spikes": [],
            "volume_drops": [],
            "zero_volume_periods": [],
            "statistics": {}
        }
        
        if volume_col not in data.columns:
            return results
            
        volume = data[volume_col]
        
        # Calculate rolling statistics
        rolling_mean = volume.rolling(
            window=self.config["volume_params"]["rolling_window"]
        ).mean()
        rolling_std = volume.rolling(
            window=self.config["volume_params"]["rolling_window"]
        ).std()
        
        # Detect volume spikes
        spike_threshold = self.config["volume_params"]["spike_threshold"]
        volume_spikes = volume > (rolling_mean * spike_threshold)
        results["volume_spikes"] = data.index[volume_spikes.fillna(False)].tolist()
        
        # Detect volume drops
        drop_threshold = self.config["volume_params"]["drop_threshold"]
        volume_drops = (volume < (rolling_mean * drop_threshold)) & (volume > 0)
        results["volume_drops"] = data.index[volume_drops.fillna(False)].tolist()
        
        # Detect zero volume periods
        zero_volume = volume == 0
        results["zero_volume_periods"] = data.index[zero_volume].tolist()
        
        # Calculate statistics
        results["statistics"] = {
            "mean_volume": float(volume.mean()),
            "std_volume": float(volume.std()),
            "min_volume": float(volume.min()),
            "max_volume": float(volume.max()),
            "spike_count": len(results["volume_spikes"]),
            "drop_count": len(results["volume_drops"]),
            "zero_count": len(results["zero_volume_periods"]),
            "anomaly_rate": float(
                (len(results["volume_spikes"]) + len(results["volume_drops"]) + 
                 len(results["zero_volume_periods"])) / len(data)
            )
        }
        
        return results
        
    def detect_price_anomalies(
        self,
        data: pd.DataFrame,
        price_cols: Optional[List[str]] = None
    ) -> dict[str, Any]:
        """
        Detect price-specific anomalies.
        
        Args:
            data: DataFrame with price data
            price_cols: Price columns to analyze
            
        Returns:
            Price anomaly detection results
        """
        if price_cols is None:
            price_cols = ["open", "high", "low", "close"]
            
        results = {
            "price_spikes": {},
            "price_drops": {},
            "price_reversals": {},
            "statistics": {}
        }
        
        for col in price_cols:
            if col not in data.columns:
                continue
                
            prices = data[col]
            returns = prices.pct_change()
            
            # Detect extreme price movements
            extreme_threshold = 0.1  # 10% movement
            price_spikes = returns > extreme_threshold
            price_drops = returns < -extreme_threshold
            
            results["price_spikes"][col] = data.index[price_spikes.fillna(False)].tolist()
            results["price_drops"][col] = data.index[price_drops.fillna(False)].tolist()
            
            # Detect price reversals (large movement followed by opposite movement)
            reversals = []
            for i in range(1, len(returns) - 1):
                if (returns.iloc[i] > extreme_threshold and 
                    returns.iloc[i + 1] < -extreme_threshold * 0.5):
                    reversals.append(data.index[i])
                elif (returns.iloc[i] < -extreme_threshold and 
                      returns.iloc[i + 1] > extreme_threshold * 0.5):
                    reversals.append(data.index[i])
                    
            results["price_reversals"][col] = reversals
            
            # Calculate statistics
            results["statistics"][col] = {
                "mean_return": float(returns.mean()),
                "std_return": float(returns.std()),
                "max_return": float(returns.max()),
                "min_return": float(returns.min()),
                "spike_count": len(results["price_spikes"][col]),
                "drop_count": len(results["price_drops"][col]),
                "reversal_count": len(results["price_reversals"][col])
            }
            
        return results
        
    def _generate_anomaly_summary(self, results: dict[str, Any]) -> None:
        """Generate summary of all detected anomalies."""
        total_anomalies = 0
        anomaly_columns = []
        
        # Count anomalies from main detection
        if "anomalies" in results:
            for col, anomaly_data in results["anomalies"].items():
                if "indices" in anomaly_data:
                    count = len(anomaly_data["indices"])
                    total_anomalies += count
                    if count > 0:
                        anomaly_columns.append(col)
                        
        # Count pattern anomalies
        if "pattern_anomalies" in results.get("detailed_analysis", {}):
            for col, pattern_data in results["detailed_analysis"]["pattern_anomalies"].items():
                total_anomalies += pattern_data.get("count", 0)
                
        # Count time anomalies
        if "time_anomalies" in results.get("detailed_analysis", {}):
            for anomaly_type, time_data in results["detailed_analysis"]["time_anomalies"].items():
                total_anomalies += time_data.get("count", 0)
                
        results["summary"] = {
            "total_anomalies": total_anomalies,
            "columns_with_anomalies": anomaly_columns,
            "anomaly_types_detected": list(results.get("detailed_analysis", {}).keys())
        }
        
    def _generate_anomaly_recommendations(self, results: dict[str, Any]) -> List[str]:
        """Generate recommendations based on detected anomalies."""
        recommendations = []
        
        # Check for high anomaly rate
        if results["summary"].get("total_anomalies", 0) > 100:
            recommendations.append(
                "High number of anomalies detected. Consider reviewing data source and collection process."
            )
            
        # Check for specific anomaly types
        if "time_anomalies" in results.get("detailed_analysis", {}):
            if "time_gaps" in results["detailed_analysis"]["time_anomalies"]:
                recommendations.append(
                    "Significant time gaps detected. Verify data completeness and consider gap-filling strategies."
                )
                
        # Check for volume anomalies
        for col in results.get("anomalies", {}):
            if "volume" in col.lower():
                recommendations.append(
                    "Volume anomalies detected. Review for potential data errors or unusual market conditions."
                )
                break
                
        # Check for price anomalies
        price_cols = ["open", "high", "low", "close"]
        if any(col in results.get("anomalies", {}) for col in price_cols):
            recommendations.append(
                "Price anomalies detected. Validate against external sources and check for data feed issues."
            )
            
        return recommendations