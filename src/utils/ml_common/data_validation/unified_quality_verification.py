"""
Unified Data Quality Verification Orchestrator

This module provides a unified interface for data quality verification across different data types
(aggtrades, klines, futures, etc.) with automatic detection and appropriate verification strategies.

Key Features:
- Automatic data type detection
- Unified interface for all data types
- Pipeline integration hooks
- Stage beginning and data collection completion verification
- Comprehensive reporting and monitoring
- Integration with existing validation framework

Built on existing utilities:
- Uses aggtrades_quality_verification.py for aggtrades data
- Uses klines_quality_verification.py for klines data
- Leverages validation_utils.py for validation framework
- Integrates with structured_logging.py for comprehensive logging
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

from ...math_validation import safe_divide, MathValidationError
from ...validation_utils import ValidationError
from .aggtrades_quality_verification import AggtradesQualityVerifier, QualityReport as AggtradesQualityReport
from .klines_quality_verification import KlinesQualityVerifier, QualityReport as KlinesQualityReport
from ...structured_logging import StructuredLogger

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Supported data types for quality verification."""
    AGGRADES = "aggtrades"
    KLINES = "klines"
    FUTURES = "futures"
    UNIFIED = "unified"
    UNKNOWN = "unknown"


class VerificationStage(Enum):
    """Pipeline stages where quality verification is performed."""
    DATA_COLLECTION_END = "data_collection_end"
    STAGE_BEGINNING = "stage_beginning"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    CUSTOM = "custom"


@dataclass
class UnifiedQualityReport:
    """Unified quality report combining all data types."""
    timestamp: datetime
    data_type: DataType
    stage: VerificationStage
    total_rows: int
    quality_score: float
    issues: List[Dict[str, Any]]
    summary: Dict[str, Any]
    recommendations: List[str]
    verification_details: Dict[str, Any]


class UnifiedQualityVerifier:
    """Unified data quality verification orchestrator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize unified quality verifier.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(f"{__name__}.UnifiedQualityVerifier")
        self.structured_logger = StructuredLogger(self.logger)

        # Initialize data type verifiers
        self.aggtrades_verifier = AggtradesQualityVerifier(self.config.get('aggtrades', {}), self.logger)
        self.klines_verifier = KlinesQualityVerifier(self.config.get('klines', {}), self.logger)

        # Pipeline integration settings
        self.enable_pipeline_integration = self.config.get('enable_pipeline_integration', True)
        self.auto_fix_enabled = self.config.get('auto_fix_enabled', True)
        self.export_reports = self.config.get('export_reports', True)
        self.reports_directory = self.config.get('reports_directory', 'reports/quality')

        # Stage-specific configurations
        self.stage_configs = self.config.get('stage_configs', {})
        self._setup_default_stage_configs()

    def _setup_default_stage_configs(self):
        """Setup default configurations for different pipeline stages."""
        default_configs = {
            VerificationStage.DATA_COLLECTION_END: {
                'auto_fix': True,
                'strict_mode': False,
                'export_report': True,
                'alert_on_issues': True
            },
            VerificationStage.STAGE_BEGINNING: {
                'auto_fix': True,
                'strict_mode': True,
                'export_report': True,
                'alert_on_issues': True
            },
            VerificationStage.PREPROCESSING: {
                'auto_fix': True,
                'strict_mode': False,
                'export_report': True,
                'alert_on_issues': False
            },
            VerificationStage.FEATURE_ENGINEERING: {
                'auto_fix': True,
                'strict_mode': True,
                'export_report': True,
                'alert_on_issues': True
            },
            VerificationStage.MODEL_TRAINING: {
                'auto_fix': False,
                'strict_mode': True,
                'export_report': True,
                'alert_on_issues': True
            }
        }

        for stage, config in default_configs.items():
            if stage not in self.stage_configs:
                self.stage_configs[stage] = config

    def detect_data_type(self, data: pd.DataFrame) -> DataType:
        """
        Automatically detect data type based on column structure.

        Args:
            data: DataFrame to analyze

        Returns:
            Detected data type
        """
        columns = set(data.columns)
        
        # Aggtrades detection
        aggtrades_required = {'timestamp', 'price', 'quantity'}
        if aggtrades_required.issubset(columns):
            return DataType.AGGRADES
        
        # Klines detection
        klines_required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        if klines_required.issubset(columns):
            return DataType.KLINES
        
        # Futures detection
        futures_required = {'timestamp', 'fundingRate'}
        if futures_required.issubset(columns):
            return DataType.FUTURES
        
        # Unified detection (contains both aggtrades and klines columns)
        if aggtrades_required.issubset(columns) and klines_required.issubset(columns):
            return DataType.UNIFIED
        
        return DataType.UNKNOWN

    def verify_data_quality(self, data: pd.DataFrame, 
                          stage: VerificationStage = VerificationStage.CUSTOM,
                          data_type: Optional[DataType] = None,
                          custom_config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, UnifiedQualityReport]:
        """
        Unified data quality verification.

        Args:
            data: DataFrame to verify
            stage: Pipeline stage
            data_type: Data type (auto-detected if None)
            custom_config: Custom configuration overrides

        Returns:
            Tuple of (cleaned_data, unified_quality_report)

        Raises:
            ValidationError: For critical quality issues
        """
        self.logger.info(f"🔍 Starting unified quality verification for stage: {stage.value}")
        
        # Auto-detect data type if not provided
        if data_type is None:
            data_type = self.detect_data_type(data)
            self.logger.info(f"📊 Auto-detected data type: {data_type.value}")

        # Get stage configuration
        stage_config = self.stage_configs.get(stage, {})
        
        # Merge with custom config
        if custom_config:
            stage_config = {**stage_config, **custom_config}

        # Verify based on data type
        if data_type == DataType.AGGRADES:
            cleaned_data, report = self._verify_aggtrades(data, stage_config)
        elif data_type == DataType.KLINES:
            cleaned_data, report = self._verify_klines(data, stage_config)
        elif data_type == DataType.FUTURES:
            cleaned_data, report = self._verify_futures(data, stage_config)
        elif data_type == DataType.UNIFIED:
            cleaned_data, report = self._verify_unified(data, stage_config)
        else:
            self.logger.warning(f"⚠️ Unknown data type: {data_type.value}, using basic validation")
            cleaned_data, report = self._verify_unknown(data, stage_config)

        # Convert to unified report
        unified_report = self._create_unified_report(report, data_type, stage, data, cleaned_data)

        # Export report if enabled
        if stage_config.get('export_report', self.export_reports):
            self._export_unified_report(unified_report, stage)

        # Handle alerts
        if stage_config.get('alert_on_issues', False):
            self._handle_quality_alerts(unified_report, stage)

        self.logger.info(f"✅ Unified quality verification completed - Quality score: {unified_report.quality_score:.3f}")
        return cleaned_data, unified_report

    def _verify_aggtrades(self, data: pd.DataFrame, stage_config: Dict[str, Any]) -> Tuple[pd.DataFrame, AggtradesQualityReport]:
        """Verify aggtrades data quality."""
        auto_fix = stage_config.get('auto_fix', self.auto_fix_enabled)
        return self.aggtrades_verifier.verify_aggtrades_quality(data, auto_fix=auto_fix)

    def _verify_klines(self, data: pd.DataFrame, stage_config: Dict[str, Any]) -> Tuple[pd.DataFrame, KlinesQualityReport]:
        """Verify klines data quality."""
        auto_fix = stage_config.get('auto_fix', self.auto_fix_enabled)
        return self.klines_verifier.verify_klines_quality(data, auto_fix=auto_fix)

    def _verify_futures(self, data: pd.DataFrame, stage_config: Dict[str, Any]) -> Tuple[pd.DataFrame, UnifiedQualityReport]:
        """Verify futures data quality (basic validation)."""
        self.logger.info("📊 Verifying futures data quality (basic validation)")
        
        # Basic validation for futures data
        cleaned_data = data.copy()
        issues = []
        
        # Check required columns
        required_columns = ['timestamp', 'fundingRate']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append({
                'issue_type': 'missing_columns',
                'severity': 'critical',
                'message': f"Missing required columns: {missing_columns}",
                'affected_rows': []
            })
        
        # Check for invalid funding rates
        if 'fundingRate' in data.columns:
            invalid_rates = data['fundingRate'].isna()
            invalid_count = invalid_rates.sum()
            if invalid_count > 0:
                issues.append({
                    'issue_type': 'invalid_funding_rate',
                    'severity': 'warning',
                    'message': f"Found {invalid_count} invalid funding rates",
                    'affected_rows': data.index[invalid_rates].tolist()
                })
        
        # Create basic report
        report = UnifiedQualityReport(
            timestamp=datetime.now(),
            data_type=DataType.FUTURES,
            stage=VerificationStage.CUSTOM,
            total_rows=len(data),
            quality_score=1.0 - (len(issues) * 0.1),
            issues=issues,
            summary={'basic_validation': True},
            recommendations=[],
            verification_details={'validation_type': 'basic'}
        )
        
        return cleaned_data, report

    def _verify_unified(self, data: pd.DataFrame, stage_config: Dict[str, Any]) -> Tuple[pd.DataFrame, UnifiedQualityReport]:
        """Verify unified data quality (both aggtrades and klines)."""
        self.logger.info("📊 Verifying unified data quality")
        
        # Verify both aggtrades and klines components
        auto_fix = stage_config.get('auto_fix', self.auto_fix_enabled)
        
        # First verify as aggtrades
        aggtrades_data = data[['timestamp', 'price', 'quantity'] + 
                             [col for col in data.columns if col in ['first_trade_id', 'last_trade_id', 'trade_time', 'is_buyer_maker']]]
        _, aggtrades_report = self.aggtrades_verifier.verify_aggtrades_quality(aggtrades_data, auto_fix=auto_fix)
        
        # Then verify as klines
        klines_data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume'] + 
                          [col for col in data.columns if col in ['quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']]]
        _, klines_report = self.klines_verifier.verify_klines_quality(klines_data, auto_fix=auto_fix)
        
        # Combine results
        combined_issues = []
        combined_issues.extend([{
            'issue_type': f"aggtrades_{issue.issue_type}",
            'severity': issue.severity.value,
            'message': f"Aggtrades: {issue.message}",
            'affected_rows': issue.affected_rows,
            'details': issue.details
        } for issue in aggtrades_report.issues])
        
        combined_issues.extend([{
            'issue_type': f"klines_{issue.issue_type}",
            'severity': issue.severity.value,
            'message': f"Klines: {issue.message}",
            'affected_rows': issue.affected_rows,
            'details': issue.details
        } for issue in klines_report.issues])
        
        # Calculate combined quality score
        combined_score = (aggtrades_report.quality_score + klines_report.quality_score) / 2
        
        # Create unified report
        report = UnifiedQualityReport(
            timestamp=datetime.now(),
            data_type=DataType.UNIFIED,
            stage=VerificationStage.CUSTOM,
            total_rows=len(data),
            quality_score=combined_score,
            issues=combined_issues,
            summary={
                'aggtrades_score': aggtrades_report.quality_score,
                'klines_score': klines_report.quality_score,
                'combined_score': combined_score
            },
            recommendations=aggtrades_report.recommendations + klines_report.recommendations,
            verification_details={
                'aggtrades_issues': len(aggtrades_report.issues),
                'klines_issues': len(klines_report.issues),
                'total_issues': len(combined_issues)
            }
        )
        
        return data, report

    def _verify_unknown(self, data: pd.DataFrame, stage_config: Dict[str, Any]) -> Tuple[pd.DataFrame, UnifiedQualityReport]:
        """Verify unknown data type with basic validation."""
        self.logger.info("📊 Verifying unknown data type with basic validation")
        
        # Basic validation
        cleaned_data = data.copy()
        issues = []
        
        # Check for empty data
        if len(data) == 0:
            issues.append({
                'issue_type': 'empty_data',
                'severity': 'critical',
                'message': "Data is empty",
                'affected_rows': []
            })
        
        # Check for all NaN columns
        all_nan_columns = data.columns[data.isnull().all()].tolist()
        if all_nan_columns:
            issues.append({
                'issue_type': 'all_nan_columns',
                'severity': 'warning',
                'message': f"Columns with all NaN values: {all_nan_columns}",
                'affected_rows': []
            })
        
        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            issues.append({
                'issue_type': 'duplicate_rows',
                'severity': 'warning',
                'message': f"Found {duplicate_count} duplicate rows",
                'affected_rows': data.index[data.duplicated()].tolist()
            })
        
        # Create basic report
        report = UnifiedQualityReport(
            timestamp=datetime.now(),
            data_type=DataType.UNKNOWN,
            stage=VerificationStage.CUSTOM,
            total_rows=len(data),
            quality_score=1.0 - (len(issues) * 0.1),
            issues=issues,
            summary={'basic_validation': True, 'unknown_data_type': True},
            recommendations=[],
            verification_details={'validation_type': 'basic_unknown'}
        )
        
        return cleaned_data, report

    def _create_unified_report(self, report: Union[AggtradesQualityReport, KlinesQualityReport, UnifiedQualityReport], 
                             data_type: DataType, stage: VerificationStage, 
                             original_data: pd.DataFrame, cleaned_data: pd.DataFrame) -> UnifiedQualityReport:
        """Create unified quality report from specific data type report."""
        
        if isinstance(report, UnifiedQualityReport):
            return report
        
        # Convert specific report to unified format
        unified_issues = []
        for issue in report.issues:
            unified_issues.append({
                'issue_type': issue.issue_type,
                'severity': issue.severity.value,
                'message': issue.message,
                'affected_rows': issue.affected_rows,
                'details': issue.details,
                'action': issue.action.value
            })
        
        return UnifiedQualityReport(
            timestamp=report.timestamp,
            data_type=data_type,
            stage=stage,
            total_rows=report.total_rows,
            quality_score=report.quality_score,
            issues=unified_issues,
            summary=report.summary,
            recommendations=report.recommendations,
            verification_details={
                'data_type': data_type.value,
                'original_rows': len(original_data),
                'cleaned_rows': len(cleaned_data),
                'rows_removed': len(original_data) - len(cleaned_data)
            }
        )

    def _export_unified_report(self, report: UnifiedQualityReport, stage: VerificationStage) -> None:
        """Export unified quality report."""
        import json
        import os
        
        # Ensure reports directory exists
        os.makedirs(self.reports_directory, exist_ok=True)
        
        # Create filename
        timestamp_str = report.timestamp.strftime('%Y%m%d_%H%M%S')
        filename = f"quality_report_{report.data_type.value}_{stage.value}_{timestamp_str}.json"
        filepath = os.path.join(self.reports_directory, filename)
        
        # Convert report to dictionary
        report_dict = {
            "timestamp": report.timestamp.isoformat(),
            "data_type": report.data_type.value,
            "stage": report.stage.value,
            "total_rows": report.total_rows,
            "quality_score": report.quality_score,
            "issues": report.issues,
            "summary": report.summary,
            "recommendations": report.recommendations,
            "verification_details": report.verification_details
        }
        
        # Export
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        self.logger.info(f"📄 Quality report exported to: {filepath}")

    def _handle_quality_alerts(self, report: UnifiedQualityReport, stage: VerificationStage) -> None:
        """Handle quality alerts based on report."""
        critical_issues = [issue for issue in report.issues if issue['severity'] == 'critical']
        error_issues = [issue for issue in report.issues if issue['severity'] == 'error']
        
        if critical_issues or error_issues:
            alert_message = f"Quality issues in {stage.value}: {len(critical_issues)} critical, {len(error_issues)} errors"
            self.logger.error(f"🚨 {alert_message}")
            
            # Here you could integrate with your alerting system
            # send_alert(alert_message, report)

    # Pipeline integration methods

    def verify_data_collection_completion(self, data: pd.DataFrame, 
                                        exchange: str, symbol: str, 
                                        data_type: Optional[DataType] = None) -> Tuple[pd.DataFrame, UnifiedQualityReport]:
        """
        Verify data quality at the end of data collection.

        Args:
            data: Collected data
            exchange: Exchange name
            symbol: Symbol name
            data_type: Data type (auto-detected if None)

        Returns:
            Tuple of (cleaned_data, quality_report)
        """
        self.logger.info(f"🔍 Verifying data collection completion for {exchange}_{symbol}")
        
        # Add exchange and symbol to config
        stage_config = self.stage_configs.get(VerificationStage.DATA_COLLECTION_END, {}).copy()
        stage_config.update({
            'exchange': exchange,
            'symbol': symbol,
            'context': 'data_collection_completion'
        })
        
        return self.verify_data_quality(data, VerificationStage.DATA_COLLECTION_END, data_type, stage_config)

    def verify_stage_beginning(self, data: pd.DataFrame, stage_name: str, 
                             data_type: Optional[DataType] = None) -> Tuple[pd.DataFrame, UnifiedQualityReport]:
        """
        Verify data quality at the beginning of a pipeline stage.

        Args:
            data: Input data for the stage
            stage_name: Name of the stage
            data_type: Data type (auto-detected if None)

        Returns:
            Tuple of (cleaned_data, quality_report)
        """
        self.logger.info(f"🔍 Verifying stage beginning for: {stage_name}")
        
        # Map stage name to verification stage
        stage_mapping = {
            'preprocessing': VerificationStage.PREPROCESSING,
            'feature_engineering': VerificationStage.FEATURE_ENGINEERING,
            'model_training': VerificationStage.MODEL_TRAINING
        }
        
        verification_stage = stage_mapping.get(stage_name, VerificationStage.STAGE_BEGINNING)
        
        # Add stage name to config
        stage_config = self.stage_configs.get(verification_stage, {}).copy()
        stage_config.update({
            'stage_name': stage_name,
            'context': 'stage_beginning'
        })
        
        return self.verify_data_quality(data, verification_stage, data_type, stage_config)

    def create_pipeline_integration_hooks(self) -> Dict[str, Callable]:
        """
        Create pipeline integration hooks for easy integration.

        Returns:
            Dictionary of hook functions
        """
        return {
            'data_collection_end': self.verify_data_collection_completion,
            'stage_beginning': self.verify_stage_beginning,
            'verify_quality': self.verify_data_quality
        }

    def get_quality_summary(self, reports: List[UnifiedQualityReport]) -> Dict[str, Any]:
        """
        Generate quality summary from multiple reports.

        Args:
            reports: List of quality reports

        Returns:
            Quality summary dictionary
        """
        if not reports:
            return {'error': 'No reports provided'}
        
        # Calculate overall statistics
        total_rows = sum(report.total_rows for report in reports)
        avg_quality_score = sum(report.quality_score for report in reports) / len(reports)
        min_quality_score = min(report.quality_score for report in reports)
        max_quality_score = max(report.quality_score for report in reports)
        
        # Count issues by severity
        severity_counts = {'critical': 0, 'error': 0, 'warning': 0, 'info': 0}
        for report in reports:
            for issue in report.issues:
                severity_counts[issue['severity']] += 1
        
        # Count issues by data type
        data_type_counts = {}
        for report in reports:
            data_type = report.data_type.value
            if data_type not in data_type_counts:
                data_type_counts[data_type] = 0
            data_type_counts[data_type] += len(report.issues)
        
        return {
            'total_reports': len(reports),
            'total_rows_processed': total_rows,
            'quality_scores': {
                'average': avg_quality_score,
                'minimum': min_quality_score,
                'maximum': max_quality_score,
                'std': np.std([report.quality_score for report in reports])
            },
            'issue_summary': {
                'by_severity': severity_counts,
                'by_data_type': data_type_counts,
                'total_issues': sum(severity_counts.values())
            },
            'recommendations': list(set([
                rec for report in reports for rec in report.recommendations
            ]))
        }


# Convenience functions
def create_unified_quality_verifier(config: Optional[Dict[str, Any]] = None, 
                                  logger: Optional[logging.Logger] = None) -> UnifiedQualityVerifier:
    """Create a unified quality verifier instance."""
    return UnifiedQualityVerifier(config, logger)


def verify_data_quality_unified(data: pd.DataFrame, 
                              stage: VerificationStage = VerificationStage.CUSTOM,
                              data_type: Optional[DataType] = None,
                              config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, UnifiedQualityReport]:
    """
    Convenience function for unified data quality verification.

    Args:
        data: DataFrame to verify
        stage: Pipeline stage
        data_type: Data type (auto-detected if None)
        config: Configuration dictionary

    Returns:
        Tuple of (cleaned_data, unified_quality_report)
    """
    verifier = UnifiedQualityVerifier(config)
    return verifier.verify_data_quality(data, stage, data_type)


def create_pipeline_quality_config(**kwargs) -> Dict[str, Any]:
    """
    Create pipeline quality verification configuration.

    Args:
        **kwargs: Configuration overrides

    Returns:
        Configuration dictionary
    """
    return {
        'enable_pipeline_integration': True,
        'auto_fix_enabled': True,
        'export_reports': True,
        'reports_directory': 'reports/quality',
        'aggtrades': {
            'max_timestamp_gap_seconds': 0.5,
            'max_duplicate_ratio': 0.001,
            'duplicate_action': 'remove',
            'price_negative_action': 'fail',
            'volume_negative_action': 'fail'
        },
        'klines': {
            'timeframe': '1m',
            'max_timestamp_gap_multiplier': 2.0,
            'max_duplicate_ratio': 0.001,
            'duplicate_action': 'remove',
            'ohlc_negative_action': 'fail',
            'volume_negative_action': 'fail'
        },
        **kwargs
    }