"""
Real-Time Optimization Performance Monitoring

This module provides real-time monitoring of optimization performance
and adaptive optimization based on performance feedback.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import threading
import time
from collections import deque

from src.utils.logger import get_logger


@dataclass
class PerformanceMetric:
    """Performance metric with timestamp."""
    metric_name: str
    value: float
    timestamp: datetime
    model_type: str
    optimization_id: str


@dataclass
class OptimizationPerformance:
    """Optimization performance summary."""
    optimization_id: str
    model_type: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    objective_score: float
    validation_score: float
    performance_metrics: Dict[str, float]
    status: str  # running, completed, failed
    error_message: Optional[str] = None


class OptimizationMonitor:
    """
    Real-time optimization performance monitor.
    """
    
    def __init__(self, monitoring_interval: int = 60, max_history: int = 1000):
        """Initialize optimization monitor."""
        self.monitoring_interval = monitoring_interval
        self.max_history = max_history
        self.logger = get_logger('OptimizationMonitor')
        
        # Performance tracking
        self.performance_history = deque(maxlen=max_history)
        self.active_optimizations = {}
        self.completed_optimizations = {}
        
        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Performance thresholds
        self.thresholds = {
            'objective_score': 0.3,
            'validation_score': 0.5,
            'optimization_time': 300,  # 5 minutes
            'performance_degradation': 0.1
        }
        
        self.logger.info('🔧 Optimization monitor initialized')
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info('📊 Started real-time monitoring')
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info('📊 Stopped real-time monitoring')
    
    def start_optimization(self, optimization_id: str, model_type: str) -> None:
        """Start tracking an optimization."""
        self.active_optimizations[optimization_id] = OptimizationPerformance(
            optimization_id=optimization_id,
            model_type=model_type,
            start_time=datetime.now(),
            end_time=None,
            duration_seconds=0.0,
            objective_score=0.0,
            validation_score=0.0,
            performance_metrics={},
            status='running'
        )
        
        self.logger.info(f'🎯 Started tracking optimization {optimization_id} for {model_type}')
    
    def update_optimization_progress(self, optimization_id: str, 
                                   objective_score: float,
                                   validation_score: float,
                                   performance_metrics: Dict[str, float]) -> None:
        """Update optimization progress."""
        if optimization_id in self.active_optimizations:
            optimization = self.active_optimizations[optimization_id]
            optimization.objective_score = objective_score
            optimization.validation_score = validation_score
            optimization.performance_metrics = performance_metrics
            
            # Record performance metric
            self._record_performance_metric(
                optimization_id, 'objective_score', objective_score, optimization.model_type
            )
            self._record_performance_metric(
                optimization_id, 'validation_score', validation_score, optimization.model_type
            )
    
    def complete_optimization(self, optimization_id: str, 
                            objective_score: float,
                            validation_score: float,
                            performance_metrics: Dict[str, float]) -> None:
        """Complete optimization tracking."""
        if optimization_id in self.active_optimizations:
            optimization = self.active_optimizations[optimization_id]
            optimization.end_time = datetime.now()
            optimization.duration_seconds = (optimization.end_time - optimization.start_time).total_seconds()
            optimization.objective_score = objective_score
            optimization.validation_score = validation_score
            optimization.performance_metrics = performance_metrics
            optimization.status = 'completed'
            
            # Move to completed optimizations
            self.completed_optimizations[optimization_id] = optimization
            del self.active_optimizations[optimization_id]
            
            self.logger.info(f'✅ Completed optimization {optimization_id} in {optimization.duration_seconds:.2f}s')
    
    def fail_optimization(self, optimization_id: str, error_message: str) -> None:
        """Mark optimization as failed."""
        if optimization_id in self.active_optimizations:
            optimization = self.active_optimizations[optimization_id]
            optimization.end_time = datetime.now()
            optimization.duration_seconds = (optimization.end_time - optimization.start_time).total_seconds()
            optimization.status = 'failed'
            optimization.error_message = error_message
            
            # Move to completed optimizations
            self.completed_optimizations[optimization_id] = optimization
            del self.active_optimizations[optimization_id]
            
            self.logger.error(f'❌ Failed optimization {optimization_id}: {error_message}')
    
    def get_performance_summary(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary."""
        # Filter by model type if specified
        if model_type:
            completed_ops = {k: v for k, v in self.completed_optimizations.items() 
                           if v.model_type == model_type}
        else:
            completed_ops = self.completed_optimizations
        
        if not completed_ops:
            return {'status': 'no_data'}
        
        # Calculate summary statistics
        objective_scores = [op.objective_score for op in completed_ops.values()]
        validation_scores = [op.validation_score for op in completed_ops.values()]
        durations = [op.duration_seconds for op in completed_ops.values()]
        
        return {
            'total_optimizations': len(completed_ops),
            'active_optimizations': len(self.active_optimizations),
            'avg_objective_score': np.mean(objective_scores),
            'avg_validation_score': np.mean(validation_scores),
            'avg_duration_seconds': np.mean(durations),
            'success_rate': sum(1 for op in completed_ops.values() if op.status == 'completed') / len(completed_ops),
            'performance_trend': self._calculate_performance_trend(completed_ops)
        }
    
    def get_optimization_recommendations(self, model_type: str) -> List[str]:
        """Get optimization recommendations based on performance history."""
        recommendations = []
        
        # Get recent performance
        recent_ops = self._get_recent_optimizations(model_type, days=7)
        
        if not recent_ops:
            return ["No recent optimization data available"]
        
        # Analyze performance trends
        objective_scores = [op.objective_score for op in recent_ops]
        validation_scores = [op.validation_score for op in recent_ops]
        durations = [op.duration_seconds for op in recent_ops]
        
        # Performance recommendations
        if np.mean(objective_scores) < self.thresholds['objective_score']:
            recommendations.append("Consider adjusting optimization parameters to improve objective scores")
        
        if np.mean(validation_scores) < self.thresholds['validation_score']:
            recommendations.append("Improve validation framework to increase validation scores")
        
        if np.mean(durations) > self.thresholds['optimization_time']:
            recommendations.append("Optimization taking too long, consider reducing search space")
        
        # Performance degradation detection
        if len(objective_scores) > 5:
            recent_avg = np.mean(objective_scores[-3:])
            older_avg = np.mean(objective_scores[:-3])
            if recent_avg < older_avg - self.thresholds['performance_degradation']:
                recommendations.append("Performance degradation detected, consider re-optimization")
        
        return recommendations
    
    def export_performance_data(self, output_file: str) -> None:
        """Export performance data to file."""
        try:
            export_data = {
                'performance_history': [
                    {
                        'metric_name': metric.metric_name,
                        'value': metric.value,
                        'timestamp': metric.timestamp.isoformat(),
                        'model_type': metric.model_type,
                        'optimization_id': metric.optimization_id
                    }
                    for metric in self.performance_history
                ],
                'completed_optimizations': {
                    opt_id: {
                        'model_type': opt.model_type,
                        'start_time': opt.start_time.isoformat(),
                        'end_time': opt.end_time.isoformat() if opt.end_time else None,
                        'duration_seconds': opt.duration_seconds,
                        'objective_score': opt.objective_score,
                        'validation_score': opt.validation_score,
                        'performance_metrics': opt.performance_metrics,
                        'status': opt.status,
                        'error_message': opt.error_message
                    }
                    for opt_id, opt in self.completed_optimizations.items()
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f'📊 Exported performance data to {output_file}')
            
        except Exception as e:
            self.logger.error(f'❌ Error exporting performance data: {e}')
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check for stuck optimizations
                self._check_stuck_optimizations()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f'❌ Error in monitoring loop: {e}')
                time.sleep(self.monitoring_interval)
    
    def _check_stuck_optimizations(self) -> None:
        """Check for stuck optimizations."""
        current_time = datetime.now()
        stuck_threshold = timedelta(minutes=30)  # 30 minutes
        
        stuck_optimizations = []
        for opt_id, optimization in self.active_optimizations.items():
            if current_time - optimization.start_time > stuck_threshold:
                stuck_optimizations.append(opt_id)
        
        for opt_id in stuck_optimizations:
            self.fail_optimization(opt_id, "Optimization stuck - timeout")
            self.logger.warning(f'⚠️ Optimization {opt_id} marked as stuck and failed')
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics."""
        # Calculate performance trends
        for model_type in ['analyst', 'tactician']:
            recent_ops = self._get_recent_optimizations(model_type, days=1)
            if recent_ops:
                avg_score = np.mean([op.objective_score for op in recent_ops])
                self._record_performance_metric(
                    f"monitor_{model_type}", 'daily_avg_score', avg_score, model_type
                )
    
    def _record_performance_metric(self, optimization_id: str, metric_name: str, 
                                 value: float, model_type: str) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            model_type=model_type,
            optimization_id=optimization_id
        )
        
        self.performance_history.append(metric)
    
    def _get_recent_optimizations(self, model_type: str, days: int) -> List[OptimizationPerformance]:
        """Get recent optimizations for a model type."""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        recent_ops = []
        for optimization in self.completed_optimizations.values():
            if (optimization.model_type == model_type and 
                optimization.start_time >= cutoff_time):
                recent_ops.append(optimization)
        
        return sorted(recent_ops, key=lambda x: x.start_time)
    
    def _calculate_performance_trend(self, optimizations: Dict[str, OptimizationPerformance]) -> str:
        """Calculate performance trend."""
        if len(optimizations) < 3:
            return "insufficient_data"
        
        # Sort by start time
        sorted_ops = sorted(optimizations.values(), key=lambda x: x.start_time)
        objective_scores = [op.objective_score for op in sorted_ops]
        
        # Calculate trend
        if len(objective_scores) >= 3:
            recent_avg = np.mean(objective_scores[-3:])
            older_avg = np.mean(objective_scores[:-3])
            
            if recent_avg > older_avg + 0.05:
                return "improving"
            elif recent_avg < older_avg - 0.05:
                return "declining"
            else:
                return "stable"
        
        return "insufficient_data"
