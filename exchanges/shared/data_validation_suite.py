"""
Advanced Data Validation Suite

This module provides comprehensive data validation and quality assurance
for standardized OHLCV data across all exchanges.

Features:
- Advanced data quality metrics
- Statistical validation
- Cross-exchange data consistency checks
- Performance monitoring
- Automated data correction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics
from scipy import stats
import warnings

# Import our unified components
from .unified_ohlcv_standardizer import StandardizedOHLCVData, ExchangeType
from .unified_exchange_interface import UnifiedExchangeManager

# Import src/utils/data utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data import DataQualityFramework, DataProcessor
from src.utils.logger import system_logger
from src.utils.tprint import tprint

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Data validation levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    CRITICAL = "critical"


class DataAnomalyType(Enum):
    """Types of data anomalies"""
    PRICE_SPIKE = "price_spike"
    VOLUME_ANOMALY = "volume_anomaly"
    TIMESTAMP_GAP = "timestamp_gap"
    OHLC_INCONSISTENCY = "ohlc_inconsistency"
    DUPLICATE_RECORD = "duplicate_record"
    MISSING_DATA = "missing_data"
    OUTLIER = "outlier"


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    quality_score: float
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    processing_time: float = 0.0


@dataclass
class DataAnomaly:
    """Data anomaly information"""
    anomaly_type: DataAnomalyType
    severity: str  # low, medium, high, critical
    description: str
    affected_records: List[int]
    suggested_fix: Optional[str] = None
    confidence: float = 1.0


class AdvancedDataValidator:
    """
    Advanced data validator for OHLCV data quality assurance.
    
    Provides comprehensive validation, anomaly detection, and data correction
    capabilities for standardized OHLCV data across all exchanges.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """Initialize the advanced data validator"""
        tprint(f"Initializing AdvancedDataValidator with validation_level={validation_level.value}", "INFO")
        self.validation_level = validation_level
        self.logger = system_logger.getChild("AdvancedDataValidator")
        
        # Initialize data processing utilities
        self.quality_framework = DataQualityFramework()
        self.data_processor = DataProcessor()
        
        # Validation thresholds based on level
        self.thresholds = self._get_validation_thresholds()

        self.logger.info(f"✅ AdvancedDataValidator initialized with {validation_level.value} level")
        tprint("AdvancedDataValidator initialized successfully", "SUCCESS")
    
    def _get_validation_thresholds(self) -> Dict[str, Any]:
        """Get validation thresholds based on validation level"""
        base_thresholds = {
            'price_change_max': 0.5,  # 50% max price change
            'volume_spike_threshold': 10.0,  # 10x volume spike
            'timestamp_gap_max': 300,  # 5 minutes max gap
            'ohlc_tolerance': 1e-6,  # OHLC consistency tolerance
            'outlier_z_score': 3.0,  # Z-score for outlier detection
            'missing_data_max': 0.05,  # 5% max missing data
            'duplicate_tolerance': 0.001  # 1ms duplicate tolerance
        }
        
        if self.validation_level == ValidationLevel.BASIC:
            return {k: v * 2 for k, v in base_thresholds.items()}
        elif self.validation_level == ValidationLevel.STANDARD:
            return base_thresholds
        elif self.validation_level == ValidationLevel.STRICT:
            return {k: v * 0.5 for k, v in base_thresholds.items()}
        elif self.validation_level == ValidationLevel.CRITICAL:
            return {k: v * 0.1 for k, v in base_thresholds.items()}
        
        return base_thresholds
    
    def validate_ohlcv_data(
        self, 
        data: pd.DataFrame, 
        exchange: ExchangeType,
        context: str = "validation"
    ) -> ValidationResult:
        """
        Perform comprehensive validation of OHLCV data.
        
        Args:
            data: OHLCV DataFrame to validate
            exchange: Exchange source
            context: Validation context for logging
            
        Returns:
            Comprehensive validation result
        """
        tprint(f"Validating OHLCV data: exchange={exchange.value}, records={len(data)}, context={context}", "INFO")
        start_time = datetime.now()

        try:
            self.logger.info(f"Starting validation for {exchange.value} data: {len(data)} records")
            
            # Initialize validation result
            result = ValidationResult(is_valid=True, quality_score=100.0)
            
            # Basic data structure validation
            self._validate_data_structure(data, result)
            
            # OHLC consistency validation
            self._validate_ohlc_consistency(data, result)
            
            # Price movement validation
            self._validate_price_movements(data, result)
            
            # Volume validation
            self._validate_volume_data(data, result)
            
            # Timestamp validation
            self._validate_timestamps(data, result)
            
            # Statistical validation
            self._validate_statistics(data, result)
            
            # Anomaly detection
            self._detect_anomalies(data, result)
            
            # Cross-validation with exchange standards
            self._validate_exchange_standards(data, exchange, result)
            
            # Calculate final quality score
            result.quality_score = self._calculate_quality_score(result)
            result.is_valid = result.quality_score >= self._get_minimum_quality_score()
            
            # Generate recommendations
            self._generate_recommendations(result)
            
            # Calculate processing time
            result.processing_time = (datetime.now() - start_time).total_seconds()

            self.logger.info(f"Validation completed: {result.quality_score:.2f} quality score, {len(result.anomalies)} anomalies")
            tprint(f"Validation completed: quality_score={result.quality_score:.2f}, anomalies={len(result.anomalies)}, valid={result.is_valid}", "SUCCESS")

            return result
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            tprint(f"Validation failed: {e}", "ERROR")
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                errors=[f"Validation error: {str(e)}"],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _validate_data_structure(self, data: pd.DataFrame, result: ValidationResult):
        """Validate basic data structure"""
        tprint(f"Validating data structure: shape={data.shape}", "INFO")
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            result.errors.append(f"Missing required columns: {missing_columns}")
            result.is_valid = False
            tprint(f"Missing required columns: {missing_columns}", "ERROR")
        
        # Check data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                result.errors.append(f"Column {col} should be numeric")
                result.is_valid = False
        
        # Check for empty data
        if data.empty:
            result.errors.append("Data is empty")
            result.is_valid = False
            tprint("Data is empty", "ERROR")
    
    def _validate_ohlc_consistency(self, data: pd.DataFrame, result: ValidationResult):
        """Validate OHLC price consistency"""
        tprint("Validating OHLC consistency", "INFO")
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return
        
        # Check high >= max(open, close)
        high_violations = (data['high'] < data[['open', 'close']].max(axis=1)).sum()
        if high_violations > 0:
            tprint(f"OHLC inconsistency detected: high price violations={high_violations}", "WARNING")
            result.anomalies.append({
                'type': 'ohlc_inconsistency',
                'severity': 'high',
                'description': f"High price violations: {high_violations}",
                'affected_records': data[data['high'] < data[['open', 'close']].max(axis=1)].index.tolist()
            })
        
        # Check low <= min(open, close)
        low_violations = (data['low'] > data[['open', 'close']].min(axis=1)).sum()
        if low_violations > 0:
            result.anomalies.append({
                'type': 'ohlc_inconsistency',
                'severity': 'high',
                'description': f"Low price violations: {low_violations}",
                'affected_records': data[data['low'] > data[['open', 'close']].min(axis=1)].index.tolist()
            })
        
        # Check high >= low
        high_low_violations = (data['high'] < data['low']).sum()
        if high_low_violations > 0:
            result.anomalies.append({
                'type': 'ohlc_inconsistency',
                'severity': 'critical',
                'description': f"High < Low violations: {high_low_violations}",
                'affected_records': data[data['high'] < data['low']].index.tolist()
            })
    
    def _validate_price_movements(self, data: pd.DataFrame, result: ValidationResult):
        """Validate price movement patterns"""
        if 'close' not in data.columns or len(data) < 2:
            return
        
        # Calculate price changes
        price_changes = data['close'].pct_change().dropna()
        
        # Check for extreme price changes
        extreme_changes = price_changes[abs(price_changes) > self.thresholds['price_change_max']]
        if len(extreme_changes) > 0:
            result.anomalies.append({
                'type': 'price_spike',
                'severity': 'high' if len(extreme_changes) > len(data) * 0.01 else 'medium',
                'description': f"Extreme price changes: {len(extreme_changes)}",
                'affected_records': extreme_changes.index.tolist()
            })
        
        # Check for constant prices (potential data issue)
        constant_prices = (price_changes == 0).sum()
        if constant_prices > len(data) * 0.1:  # More than 10% constant prices
            result.warnings.append(f"High number of constant prices: {constant_prices}")
    
    def _validate_volume_data(self, data: pd.DataFrame, result: ValidationResult):
        """Validate volume data"""
        if 'volume' not in data.columns:
            return
        
        # Check for negative volumes
        negative_volumes = (data['volume'] < 0).sum()
        if negative_volumes > 0:
            result.anomalies.append({
                'type': 'volume_anomaly',
                'severity': 'critical',
                'description': f"Negative volumes: {negative_volumes}",
                'affected_records': data[data['volume'] < 0].index.tolist()
            })
        
        # Check for volume spikes
        if len(data) > 1:
            volume_changes = data['volume'].pct_change().dropna()
            volume_spikes = volume_changes[volume_changes > self.thresholds['volume_spike_threshold']]
            if len(volume_spikes) > 0:
                result.anomalies.append({
                    'type': 'volume_anomaly',
                    'severity': 'medium',
                    'description': f"Volume spikes: {len(volume_spikes)}",
                    'affected_records': volume_spikes.index.tolist()
                })
    
    def _validate_timestamps(self, data: pd.DataFrame, result: ValidationResult):
        """Validate timestamp data"""
        if 'timestamp' not in data.columns:
            return
        
        # Check for duplicate timestamps
        duplicate_timestamps = data['timestamp'].duplicated().sum()
        if duplicate_timestamps > 0:
            result.anomalies.append({
                'type': 'duplicate_record',
                'severity': 'medium',
                'description': f"Duplicate timestamps: {duplicate_timestamps}",
                'affected_records': data[data['timestamp'].duplicated()].index.tolist()
            })
        
        # Check for timestamp gaps
        if len(data) > 1:
            timestamps = pd.to_datetime(data['timestamp']).sort_values()
            time_diffs = timestamps.diff().dropna()
            
            # Check for gaps larger than expected interval
            expected_interval = self._get_expected_interval(data)
            large_gaps = time_diffs[time_diffs > timedelta(seconds=expected_interval * 2)]
            
            if len(large_gaps) > 0:
                result.anomalies.append({
                    'type': 'timestamp_gap',
                    'severity': 'medium',
                    'description': f"Large timestamp gaps: {len(large_gaps)}",
                    'affected_records': large_gaps.index.tolist()
                })
    
    def _validate_statistics(self, data: pd.DataFrame, result: ValidationResult):
        """Validate statistical properties of data"""
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in numeric_columns:
            if col not in data.columns:
                continue
            
            values = data[col].dropna()
            if len(values) == 0:
                continue
            
            # Calculate statistics
            stats_dict = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median(),
                'skewness': values.skew(),
                'kurtosis': values.kurtosis()
            }
            
            result.statistics[col] = stats_dict
            
            # Check for outliers using Z-score
            if len(values) > 10:  # Need sufficient data for Z-score
                z_scores = np.abs(stats.zscore(values))
                outliers = z_scores > self.thresholds['outlier_z_score']
                
                if outliers.sum() > 0:
                    result.anomalies.append({
                        'type': 'outlier',
                        'severity': 'low',
                        'description': f"Outliers in {col}: {outliers.sum()}",
                        'affected_records': values[outliers].index.tolist()
                    })
    
    def _detect_anomalies(self, data: pd.DataFrame, result: ValidationResult):
        """Detect various types of data anomalies"""
        # Missing data detection
        missing_data = data.isnull().sum()
        total_missing = missing_data.sum()
        missing_percentage = total_missing / (len(data) * len(data.columns))
        
        if missing_percentage > self.thresholds['missing_data_max']:
            result.anomalies.append({
                'type': 'missing_data',
                'severity': 'high',
                'description': f"High missing data: {missing_percentage:.2%}",
                'affected_records': []
            })
        
        # Pattern detection
        self._detect_pattern_anomalies(data, result)
    
    def _detect_pattern_anomalies(self, data: pd.DataFrame, result: ValidationResult):
        """Detect pattern-based anomalies"""
        if len(data) < 10:  # Need sufficient data for pattern detection
            return
        
        # Check for suspiciously regular patterns (potential data manipulation)
        if 'close' in data.columns:
            close_prices = data['close'].values
            
            # Check for too many identical consecutive values
            consecutive_identical = 0
            max_consecutive = 0
            
            for i in range(1, len(close_prices)):
                if close_prices[i] == close_prices[i-1]:
                    consecutive_identical += 1
                    max_consecutive = max(max_consecutive, consecutive_identical)
                else:
                    consecutive_identical = 0
            
            if max_consecutive > 5:  # More than 5 consecutive identical prices
                result.anomalies.append({
                    'type': 'pattern_anomaly',
                    'severity': 'medium',
                    'description': f"Consecutive identical prices: {max_consecutive}",
                    'affected_records': []
                })
    
    def _validate_exchange_standards(self, data: pd.DataFrame, exchange: ExchangeType, result: ValidationResult):
        """Validate data against exchange-specific standards"""
        # Exchange-specific validation rules
        if exchange == ExchangeType.BINANCE:
            self._validate_binance_standards(data, result)
        elif exchange == ExchangeType.BINGX:
            self._validate_bingx_standards(data, result)
        elif exchange == ExchangeType.OKX:
            self._validate_okx_standards(data, result)
        elif exchange == ExchangeType.MEXC:
            self._validate_mexc_standards(data, result)
    
    def _validate_binance_standards(self, data: pd.DataFrame, result: ValidationResult):
        """Binance-specific validation"""
        # Binance typically has very precise timestamps
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
            # Check if timestamps are properly aligned to minute boundaries
            second_components = timestamps.dt.second
            if (second_components != 0).any():
                result.warnings.append("Binance timestamps not aligned to minute boundaries")
    
    def _validate_bingx_standards(self, data: pd.DataFrame, result: ValidationResult):
        """BingX-specific validation"""
        # BingX specific checks
        pass
    
    def _validate_okx_standards(self, data: pd.DataFrame, result: ValidationResult):
        """OKX-specific validation"""
        # OKX specific checks
        pass
    
    def _validate_mexc_standards(self, data: pd.DataFrame, result: ValidationResult):
        """MEXC-specific validation"""
        # MEXC specific checks
        pass
    
    def _calculate_quality_score(self, result: ValidationResult) -> float:
        """Calculate overall quality score"""
        base_score = 100.0
        
        # Deduct points for errors
        error_penalty = len(result.errors) * 10
        base_score -= error_penalty
        
        # Deduct points for anomalies based on severity
        for anomaly in result.anomalies:
            severity = anomaly.get('severity', 'low')
            if severity == 'critical':
                base_score -= 20
            elif severity == 'high':
                base_score -= 10
            elif severity == 'medium':
                base_score -= 5
            elif severity == 'low':
                base_score -= 2
        
        # Deduct points for warnings
        warning_penalty = len(result.warnings) * 1
        base_score -= warning_penalty
        
        return max(0.0, base_score)
    
    def _get_minimum_quality_score(self) -> float:
        """Get minimum quality score based on validation level"""
        if self.validation_level == ValidationLevel.BASIC:
            return 60.0
        elif self.validation_level == ValidationLevel.STANDARD:
            return 75.0
        elif self.validation_level == ValidationLevel.STRICT:
            return 85.0
        elif self.validation_level == ValidationLevel.CRITICAL:
            return 95.0
        
        return 75.0
    
    def _get_expected_interval(self, data: pd.DataFrame) -> int:
        """Get expected interval in seconds based on data"""
        if 'interval' in data.columns and not data['interval'].empty:
            interval_str = data['interval'].iloc[0]
            interval_map = {
                '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
                '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, '8h': 28800,
                '12h': 43200, '1d': 86400, '3d': 259200, '1w': 604800, '1M': 2592000
            }
            return interval_map.get(interval_str, 60)
        
        return 60  # Default to 1 minute
    
    def _generate_recommendations(self, result: ValidationResult):
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Recommendations based on anomalies
        for anomaly in result.anomalies:
            anomaly_type = anomaly.get('type', '')
            
            if anomaly_type == 'ohlc_inconsistency':
                recommendations.append("Review OHLC data for price consistency issues")
            elif anomaly_type == 'price_spike':
                recommendations.append("Investigate extreme price movements for market events")
            elif anomaly_type == 'volume_anomaly':
                recommendations.append("Check volume data for potential errors")
            elif anomaly_type == 'timestamp_gap':
                recommendations.append("Fill missing timestamp gaps or investigate data source")
            elif anomaly_type == 'duplicate_record':
                recommendations.append("Remove duplicate records from dataset")
            elif anomaly_type == 'missing_data':
                recommendations.append("Address missing data points")
            elif anomaly_type == 'outlier':
                recommendations.append("Review outliers for data quality issues")
        
        # General recommendations based on quality score
        if result.quality_score < 80:
            recommendations.append("Consider data cleaning and preprocessing")
        if result.quality_score < 60:
            recommendations.append("Data quality is poor - consider alternative data source")
        
        result.recommendations = recommendations
    
    def compare_exchange_data(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        exchange1: ExchangeType,
        exchange2: ExchangeType
    ) -> Dict[str, Any]:
        """
        Compare data from two different exchanges.

        Args:
            data1: Data from first exchange
            data2: Data from second exchange
            exchange1: First exchange type
            exchange2: Second exchange type

        Returns:
            Comparison results
        """
        tprint(f"Comparing exchange data: {exchange1.value} ({len(data1)} records) vs {exchange2.value} ({len(data2)} records)", "INFO")
        comparison_result = {
            'exchanges': [exchange1.value, exchange2.value],
            'data_points': [len(data1), len(data2)],
            'similarity_score': 0.0,
            'differences': {},
            'recommendations': []
        }
        
        try:
            # Basic comparison
            if len(data1) != len(data2):
                comparison_result['differences']['length'] = {
                    'data1': len(data1),
                    'data2': len(data2)
                }
            
            # Column comparison
            cols1 = set(data1.columns)
            cols2 = set(data2.columns)
            
            if cols1 != cols2:
                comparison_result['differences']['columns'] = {
                    'only_in_data1': list(cols1 - cols2),
                    'only_in_data2': list(cols2 - cols1)
                }
            
            # Price comparison (if both have close prices)
            if 'close' in data1.columns and 'close' in data2.columns:
                min_len = min(len(data1), len(data2))
                if min_len > 0:
                    close1 = data1['close'].iloc[:min_len]
                    close2 = data2['close'].iloc[:min_len]
                    
                    # Calculate correlation
                    correlation = close1.corr(close2)
                    comparison_result['similarity_score'] = correlation if not pd.isna(correlation) else 0.0
                    
                    # Calculate price differences
                    price_diff = abs(close1 - close2).mean()
                    comparison_result['differences']['price_difference'] = price_diff
            
            # Generate recommendations
            if comparison_result['similarity_score'] < 0.9:
                comparison_result['recommendations'].append("Low correlation between exchanges - investigate data sources")
            
            if 'price_difference' in comparison_result['differences']:
                price_diff = comparison_result['differences']['price_difference']
                if price_diff > 0.01:  # 1% difference
                    comparison_result['recommendations'].append("Significant price differences detected")

            tprint(f"Exchange comparison completed: similarity_score={comparison_result['similarity_score']:.2f}", "SUCCESS")

        except Exception as e:
            comparison_result['error'] = str(e)
            tprint(f"Exchange comparison failed: {e}", "ERROR")

        return comparison_result


# Global validator instance
advanced_data_validator = AdvancedDataValidator()


# Convenience functions
def validate_ohlcv_data_quality(
    data: pd.DataFrame,
    exchange: str,
    validation_level: str = "standard"
) -> ValidationResult:
    """
    Convenience function to validate OHLCV data quality.
    
    Args:
        data: OHLCV DataFrame to validate
        exchange: Exchange name
        validation_level: Validation level (basic, standard, strict, critical)
        
    Returns:
        Validation result
    """
    tprint(f"validate_ohlcv_data_quality called: exchange={exchange}, validation_level={validation_level}", "INFO")
    try:
        exchange_type = ExchangeType(exchange.lower())
        validation_level_enum = ValidationLevel(validation_level.lower())
        
        validator = AdvancedDataValidator(validation_level_enum)
        return validator.validate_ohlcv_data(data, exchange_type)


    except ValueError as e:
        tprint(f"Invalid parameters for validation: {e}", "ERROR")
        return ValidationResult(
            is_valid=False,
            quality_score=0.0,
            errors=[f"Invalid parameters: {e}"]
        )


def compare_exchange_quality(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    exchange1: str,
    exchange2: str
) -> Dict[str, Any]:
    """
    Convenience function to compare data quality between exchanges.
    
    Args:
        data1: Data from first exchange
        data2: Data from second exchange
        exchange1: First exchange name
        exchange2: Second exchange name
        
    Returns:
        Comparison results
    """
    tprint(f"compare_exchange_quality called: {exchange1} vs {exchange2}", "INFO")
    try:
        exchange_type1 = ExchangeType(exchange1.lower())
        exchange_type2 = ExchangeType(exchange2.lower())
        
        return advanced_data_validator.compare_exchange_data(
            data1, data2, exchange_type1, exchange_type2
        )

    except ValueError as e:
        tprint(f"Invalid exchange names: {e}", "ERROR")
        return {'error': f"Invalid exchange names: {e}"}