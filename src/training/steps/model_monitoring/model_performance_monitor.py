"""
Model Performance Monitor

This module provides comprehensive model performance monitoring and tracking
for all trained models, ensuring continuous model health and performance.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import joblib
from pathlib import Path

# Core utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.utils.parquet_utils import get_parquet_utils, ParquetUtils
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.utils.enhanced_error_handler import handle_errors_with_tracking
from src.utils.logger import system_logger

class PerformanceMetric(Enum):
    """Performance metric types."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC = "auc"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2_SCORE = "r2_score"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"

class ModelStatus(Enum):
    """Model status types."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    FAILED = "failed"

@dataclass
class PerformanceRecord:
    """Performance record for a model."""
    model_id: str
    model_name: str
    metric_name: str
    metric_value: float
    threshold: float
    status: ModelStatus
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelHealthReport:
    """Model health report."""
    model_id: str
    model_name: str
    overall_status: ModelStatus
    performance_metrics: List[PerformanceRecord]
    health_score: float
    issues: List[str]
    recommendations: List[str]
    last_updated: datetime
    monitoring_duration: float

class ModelPerformanceMonitor:
    """
    Comprehensive model performance monitor.
    
    This monitor tracks model performance over time, detects performance
    degradation, and provides health reports for all trained models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model performance monitor."""
        self.config = config
        self.logger = system_logger.getChild('ModelPerformanceMonitor')
        self.parquet_utils = get_parquet_utils()
        
        # Monitoring configuration
        self.monitoring_config = config.get('model_monitoring', {})
        self.performance_thresholds = self.monitoring_config.get('performance_thresholds', {})
        self.health_check_interval = self.monitoring_config.get('health_check_interval', 3600)  # 1 hour
        self.performance_history_limit = self.monitoring_config.get('performance_history_limit', 1000)
        
        # Model registry
        self.registered_models: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, List[PerformanceRecord]] = {}
        self.health_reports: Dict[str, ModelHealthReport] = {}
        
        # Performance tracking
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None

    @handles_errors(Exception, fallback=False, log_level='ERROR')
    @validates(strict=True)
    @traced("register_model")
    @log_execution_time
    async def register_model(
        self, 
        model_id: str, 
        model_name: str, 
        model_path: str,
        model_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a model for performance monitoring.
        
        Args:
            model_id: Unique model identifier
            model_name: Human-readable model name
            model_path: Path to the model file
            model_type: Type of model (classification, regression, etc.)
            metadata: Additional model metadata
            
        Returns:
            bool: True if registration successful
        """
        try:
            self.logger.info(f"📝 Registering model: {model_name} ({model_id})")
            
            # Validate model file exists
            if not safe_file_exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Register model
            self.registered_models[model_id] = {
                'model_name': model_name,
                'model_path': model_path,
                'model_type': model_type,
                'metadata': metadata or {},
                'registered_at': datetime.now(),
                'last_checked': None,
                'status': ModelStatus.HEALTHY
            }
            
            # Initialize performance history
            self.performance_history[model_id] = []
            self.health_reports[model_id] = None
            
            self.logger.info(f"✅ Model registered successfully: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register model {model_id}: {e}")
            return False

    @handles_errors(Exception, fallback=False, log_level='ERROR')
    @validates(strict=True)
    @traced("monitor_model_performance")
    @log_execution_time
    async def monitor_model_performance(
        self, 
        model_id: str, 
        test_data: pd.DataFrame,
        test_labels: pd.Series,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> PerformanceRecord:
        """
        Monitor model performance on test data.
        
        Args:
            model_id: Model identifier
            test_data: Test features
            test_labels: Test labels
            additional_metrics: Additional performance metrics
            
        Returns:
            PerformanceRecord: Performance record
        """
        try:
            if model_id not in self.registered_models:
                raise ValueError(f"Model {model_id} not registered")
            
            model_info = self.registered_models[model_id]
            self.logger.info(f"🔍 Monitoring performance for model: {model_info['model_name']}")
            
            # Load model
            model = joblib.load(model_info['model_path'])
            
            # Make predictions
            predictions = model.predict(test_data)
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(
                model_info['model_type'],
                test_labels,
                predictions,
                additional_metrics
            )
            
            # Create performance records
            performance_records = []
            for metric_name, metric_value in performance_metrics.items():
                threshold = self.performance_thresholds.get(metric_name, 0.0)
                status = self._determine_status(metric_value, threshold)
                
                record = PerformanceRecord(
                    model_id=model_id,
                    model_name=model_info['model_name'],
                    metric_name=metric_name,
                    metric_value=metric_value,
                    threshold=threshold,
                    status=status,
                    timestamp=datetime.now(),
                    metadata={'test_samples': len(test_data)}
                )
                
                performance_records.append(record)
                self.performance_history[model_id].append(record)
            
            # Update model status
            overall_status = self._determine_overall_status(performance_records)
            self.registered_models[model_id]['status'] = overall_status
            self.registered_models[model_id]['last_checked'] = datetime.now()
            
            # Limit history size
            if len(self.performance_history[model_id]) > self.performance_history_limit:
                self.performance_history[model_id] = self.performance_history[model_id][-self.performance_history_limit:]
            
            self.logger.info(f"✅ Performance monitoring completed for {model_info['model_name']}")
            return performance_records[0] if performance_records else None
            
        except Exception as e:
            self.logger.error(f"❌ Performance monitoring failed for model {model_id}: {e}")
            raise

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("calculate_performance_metrics")
    async def _calculate_performance_metrics(
        self,
        model_type: str,
        test_labels: pd.Series,
        predictions: np.ndarray,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calculate performance metrics based on model type."""
        metrics = {}
        
        try:
            if model_type.lower() in ['classification', 'classifier']:
                # Classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                metrics['accuracy'] = accuracy_score(test_labels, predictions)
                metrics['precision'] = precision_score(test_labels, predictions, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(test_labels, predictions, average='weighted', zero_division=0)
                metrics['f1_score'] = f1_score(test_labels, predictions, average='weighted', zero_division=0)
                
                # AUC for binary classification
                if len(np.unique(test_labels)) == 2:
                    try:
                        metrics['auc'] = roc_auc_score(test_labels, predictions)
                    except ValueError:
                        metrics['auc'] = 0.0
                
            elif model_type.lower() in ['regression', 'regressor']:
                # Regression metrics
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                metrics['mae'] = mean_absolute_error(test_labels, predictions)
                metrics['mse'] = mean_squared_error(test_labels, predictions)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['r2_score'] = r2_score(test_labels, predictions)
            
            # Add additional metrics if provided
            if additional_metrics:
                metrics.update(additional_metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating performance metrics: {e}")
            return {}

    def _determine_status(self, metric_value: float, threshold: float) -> ModelStatus:
        """Determine model status based on metric value and threshold."""
        if metric_value >= threshold:
            return ModelStatus.HEALTHY
        elif metric_value >= threshold * 0.8:
            return ModelStatus.WARNING
        elif metric_value >= threshold * 0.6:
            return ModelStatus.DEGRADED
        else:
            return ModelStatus.CRITICAL

    def _determine_overall_status(self, performance_records: List[PerformanceRecord]) -> ModelStatus:
        """Determine overall model status from performance records."""
        if not performance_records:
            return ModelStatus.HEALTHY
        
        statuses = [record.status for record in performance_records]
        
        if ModelStatus.CRITICAL in statuses:
            return ModelStatus.CRITICAL
        elif ModelStatus.DEGRADED in statuses:
            return ModelStatus.DEGRADED
        elif ModelStatus.WARNING in statuses:
            return ModelStatus.WARNING
        else:
            return ModelStatus.HEALTHY

    @handles_errors(Exception, fallback=False, log_level='ERROR')
    @traced("generate_health_report")
    @log_execution_time
    async def generate_health_report(self, model_id: str) -> ModelHealthReport:
        """
        Generate comprehensive health report for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelHealthReport: Comprehensive health report
        """
        try:
            if model_id not in self.registered_models:
                raise ValueError(f"Model {model_id} not registered")
            
            model_info = self.registered_models[model_id]
            self.logger.info(f"📊 Generating health report for: {model_info['model_name']}")
            
            # Get recent performance records
            recent_records = self.performance_history.get(model_id, [])
            if not recent_records:
                return ModelHealthReport(
                    model_id=model_id,
                    model_name=model_info['model_name'],
                    overall_status=ModelStatus.HEALTHY,
                    performance_metrics=[],
                    health_score=1.0,
                    issues=[],
                    recommendations=["No performance data available"],
                    last_updated=datetime.now(),
                    monitoring_duration=0.0
                )
            
            # Calculate health score
            health_score = self._calculate_health_score(recent_records)
            
            # Identify issues
            issues = self._identify_issues(recent_records)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(recent_records, issues)
            
            # Create health report
            health_report = ModelHealthReport(
                model_id=model_id,
                model_name=model_info['model_name'],
                overall_status=model_info['status'],
                performance_metrics=recent_records[-10:],  # Last 10 records
                health_score=health_score,
                issues=issues,
                recommendations=recommendations,
                last_updated=datetime.now(),
                monitoring_duration=time.time() - model_info['registered_at'].timestamp()
            )
            
            self.health_reports[model_id] = health_report
            self.logger.info(f"✅ Health report generated for {model_info['model_name']}")
            
            return health_report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate health report for model {model_id}: {e}")
            raise

    def _calculate_health_score(self, performance_records: List[PerformanceRecord]) -> float:
        """Calculate overall health score from performance records."""
        if not performance_records:
            return 1.0
        
        # Weight recent records more heavily
        weights = np.linspace(0.5, 1.0, len(performance_records))
        scores = []
        
        for i, record in enumerate(performance_records):
            if record.status == ModelStatus.HEALTHY:
                score = 1.0
            elif record.status == ModelStatus.WARNING:
                score = 0.8
            elif record.status == ModelStatus.DEGRADED:
                score = 0.6
            else:
                score = 0.3
            
            scores.append(score * weights[i])
        
        return safe_mean(scores) if scores else 1.0

    def _identify_issues(self, performance_records: List[PerformanceRecord]) -> List[str]:
        """Identify issues from performance records."""
        issues = []
        
        if not performance_records:
            return issues
        
        # Check for performance degradation
        recent_records = performance_records[-5:] if len(performance_records) >= 5 else performance_records
        critical_records = [r for r in recent_records if r.status == ModelStatus.CRITICAL]
        degraded_records = [r for r in recent_records if r.status == ModelStatus.DEGRADED]
        
        if critical_records:
            issues.append(f"Critical performance issues detected: {len(critical_records)} metrics")
        
        if degraded_records:
            issues.append(f"Performance degradation detected: {len(degraded_records)} metrics")
        
        # Check for consistent failures
        failed_metrics = {}
        for record in recent_records:
            if record.status in [ModelStatus.CRITICAL, ModelStatus.DEGRADED]:
                if record.metric_name not in failed_metrics:
                    failed_metrics[record.metric_name] = 0
                failed_metrics[record.metric_name] += 1
        
        for metric_name, count in failed_metrics.items():
            if count >= 3:  # Consistent failures
                issues.append(f"Consistent failures in {metric_name}: {count} occurrences")
        
        return issues

    def _generate_recommendations(self, performance_records: List[PerformanceRecord], issues: List[str]) -> List[str]:
        """Generate recommendations based on performance records and issues."""
        recommendations = []
        
        if not issues:
            recommendations.append("Model performance is healthy - continue monitoring")
            return recommendations
        
        # Generate specific recommendations based on issues
        if any("Critical performance issues" in issue for issue in issues):
            recommendations.append("Consider retraining the model with fresh data")
            recommendations.append("Review model architecture and hyperparameters")
        
        if any("Performance degradation" in issue for issue in issues):
            recommendations.append("Monitor model performance more frequently")
            recommendations.append("Consider model fine-tuning or incremental learning")
        
        if any("Consistent failures" in issue for issue in issues):
            recommendations.append("Investigate specific metric failures")
            recommendations.append("Consider feature engineering improvements")
        
        # General recommendations
        recommendations.append("Ensure data quality and consistency")
        recommendations.append("Monitor for concept drift and data distribution changes")
        
        return recommendations

    @handles_errors(Exception, fallback=False, log_level='ERROR')
    @traced("start_monitoring")
    async def start_monitoring(self) -> None:
        """Start continuous model monitoring."""
        if self.monitoring_active:
            self.logger.warning("⚠️ Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("🚀 Model performance monitoring started")

    @handles_errors(Exception, fallback=False, log_level='ERROR')
    @traced("stop_monitoring")
    async def stop_monitoring(self) -> None:
        """Stop continuous model monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("🛑 Model performance monitoring stopped")

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("monitoring_loop")
    async def _monitoring_loop(self) -> None:
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                # Check all registered models
                for model_id in self.registered_models:
                    try:
                        # Generate health report
                        await self.generate_health_report(model_id)
                        
                        # Check if model needs attention
                        model_info = self.registered_models[model_id]
                        if model_info['status'] in [ModelStatus.CRITICAL, ModelStatus.DEGRADED]:
                            self.logger.warning(f"⚠️ Model {model_info['model_name']} needs attention: {model_info['status'].value}")
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error monitoring model {model_id}: {e}")
                
                # Wait for next check
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    @handles_errors(Exception, fallback=False, log_level='ERROR')
    @traced("save_monitoring_data")
    async def save_monitoring_data(self, output_dir: str) -> str:
        """Save monitoring data to file."""
        ensure_directory(output_dir)
        
        monitoring_data = {
            'registered_models': {
                model_id: {
                    'model_name': info['model_name'],
                    'model_path': info['model_path'],
                    'model_type': info['model_type'],
                    'metadata': info['metadata'],
                    'registered_at': info['registered_at'].isoformat(),
                    'last_checked': info['last_checked'].isoformat() if info['last_checked'] else None,
                    'status': info['status'].value
                }
                for model_id, info in self.registered_models.items()
            },
            'performance_history': {
                model_id: [
                    {
                        'model_id': record.model_id,
                        'model_name': record.model_name,
                        'metric_name': record.metric_name,
                        'metric_value': record.metric_value,
                        'threshold': record.threshold,
                        'status': record.status.value,
                        'timestamp': record.timestamp.isoformat(),
                        'metadata': record.metadata
                    }
                    for record in records
                ]
                for model_id, records in self.performance_history.items()
            },
            'health_reports': {
                model_id: {
                    'model_id': report.model_id,
                    'model_name': report.model_name,
                    'overall_status': report.overall_status.value,
                    'health_score': report.health_score,
                    'issues': report.issues,
                    'recommendations': report.recommendations,
                    'last_updated': report.last_updated.isoformat(),
                    'monitoring_duration': report.monitoring_duration
                }
                for model_id, report in self.health_reports.items()
                if report is not None
            },
            'monitoring_config': self.monitoring_config,
            'timestamp': datetime.now().isoformat()
        }
        
        output_file = f"{output_dir}/model_monitoring_data.json"
        safe_json_dump(monitoring_data, output_file, indent=2)
        
        self.logger.info(f"💾 Monitoring data saved to: {output_file}")
        return output_file