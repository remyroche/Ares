"""
Optimization monitoring system.

This module provides monitoring and diagnostics for the optimization process.
"""

from typing import Any, Dict, List, Optional
import time
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from ..exceptions import MonitoringError
from ..results import HPOResult

# Import tprint functions
try:
    from ...tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
        tprint_success, tprint_performance, tprint_timer, tprint_data_preview,
        tprint_data_format, LogLevel, TPrintConfig
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_info(*args, **kwargs): pass
    def tprint_warning(*args, **kwargs): pass
    def tprint_success(*args, **kwargs): pass
    def tprint_error(*args, **kwargs): pass
    def tprint_debug(*args, **kwargs): pass
    def tprint_performance(*args, **kwargs): pass
    def tprint_data_preview(*args, **kwargs): pass
    def tprint_data_format(*args, **kwargs): pass


@dataclass
class OptimizationMetrics:
    """Metrics for a single optimization run."""
    model_name: str
    strategy: str
    start_time: float
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    end_time: Optional[float] = None
    duration: Optional[float] = None
    n_trials: int = 0
    best_score: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    system_metrics: Dict[str, Any] = field(default_factory=dict)


class OptimizationMonitor:
    """Monitor optimization processes and collect metrics."""
    
    def __init__(self, enable_detailed_logging: bool = False):
        """
        Initialize monitoring system.
        
        Args:
            enable_detailed_logging: Enable detailed logging of optimization progress
        """
        self.enable_detailed_logging = enable_detailed_logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.metrics_history: List[OptimizationMetrics] = []
        self.active_optimizations: Dict[str, OptimizationMetrics] = {}
        
        if TPRINT_AVAILABLE:
            tprint_success(f"📊 OptimizationMonitor initialized (detailed_logging: {enable_detailed_logging})")
    
    def start_optimization(self, model_name: str, strategy: str) -> str:
        """Start monitoring an optimization process."""
        try:
            run_id = str(uuid.uuid4())
            metrics = OptimizationMetrics(
                model_name=model_name,
                strategy=strategy,
                start_time=time.time(),
                run_id=run_id
            )
            
            # Use run_id as key to prevent collisions
            self.active_optimizations[run_id] = metrics
            
            if TPRINT_AVAILABLE:
                tprint_info(f"📊 Started monitoring {model_name} using {strategy} (run_id: {run_id[:8]}...)")
                tprint_data_format({
                    'model_name': model_name,
                    'strategy': strategy,
                    'run_id': run_id,
                    'start_time': metrics.start_time
                }, f"monitoring_start_{model_name}")
            elif self.enable_detailed_logging:
                self.logger.info(f"Started monitoring optimization for {model_name} using {strategy} (run_id: {run_id})")
            
            return run_id
                
        except Exception as e:
            raise MonitoringError(f"Failed to start monitoring: {e}") from e
    
    def stop_optimization(self, run_id: str, result: Optional[HPOResult] = None, 
                         error: Optional[str] = None) -> None:
        """Stop monitoring an optimization process."""
        try:
            if run_id not in self.active_optimizations:
                self.logger.warning(f"No active optimization found for run_id: {run_id}")
                return
            
            metrics = self.active_optimizations[run_id]
            metrics.end_time = time.time()
            metrics.duration = metrics.end_time - metrics.start_time
            
            if result:
                metrics.n_trials = result.n_trials
                metrics.best_score = result.best_score
                metrics.success = True
            elif error:
                metrics.error = error
                metrics.success = False
            
            # Collect system metrics
            metrics.system_metrics = self._collect_system_metrics()
            
            # Move to history
            self.metrics_history.append(metrics)
            del self.active_optimizations[run_id]
            
            if TPRINT_AVAILABLE:
                status = "✅ successful" if metrics.success else "❌ failed"
                tprint_info(f"📊 Stopped monitoring {metrics.model_name} (run_id: {run_id[:8]}...): {status} in {metrics.duration:.2f}s")
                if metrics.success:
                    tprint_success(f"🏆 Best score: {metrics.best_score:.4f}, Trials: {metrics.n_trials}")
                else:
                    tprint_error(f"💥 Error: {metrics.error}")
            elif self.enable_detailed_logging:
                status = "successful" if metrics.success else "failed"
                self.logger.info(f"Stopped monitoring {metrics.model_name} (run_id: {run_id}): {status} in {metrics.duration:.2f}s")
                
        except Exception as e:
            raise MonitoringError(f"Failed to stop monitoring: {e}") from e
    
    def update_trial_progress(self, model_name: str, trial_number: int, 
                            score: Optional[float] = None) -> None:
        """Update trial progress for active optimization."""
        try:
            if model_name in self.active_optimizations:
                metrics = self.active_optimizations[model_name]
                metrics.n_trials = max(metrics.n_trials, trial_number + 1)
                
                if score is not None:
                    if metrics.best_score is None or score > metrics.best_score:
                        metrics.best_score = score
                
                if TPRINT_AVAILABLE and trial_number % 10 == 0:
                    tprint_info(f"🔄 Trial {trial_number} for {model_name}: score={score:.4f}")
                elif self.enable_detailed_logging and trial_number % 10 == 0:
                    self.logger.info(f"Trial {trial_number} for {model_name}: score={score}")
                    
        except Exception as e:
            raise MonitoringError(f"Failed to update trial progress: {e}") from e
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            import psutil
            
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "timestamp": datetime.now().isoformat()
            }
        except ImportError:
            return {"timestamp": datetime.now().isoformat()}
        except Exception:
            return {"timestamp": datetime.now().isoformat()}
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization metrics."""
        if not self.metrics_history:
            return {"total_optimizations": 0}
        
        successful = [m for m in self.metrics_history if m.success]
        failed = [m for m in self.metrics_history if not m.success]
        
        total_duration = sum(m.duration for m in self.metrics_history if m.duration)
        avg_duration = total_duration / len(self.metrics_history) if self.metrics_history else 0
        
        return {
            "total_optimizations": len(self.metrics_history),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.metrics_history) if self.metrics_history else 0,
            "average_duration": avg_duration,
            "total_trials": sum(m.n_trials for m in self.metrics_history),
            "best_scores": [m.best_score for m in successful if m.best_score is not None]
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            "active_optimizations": len(self.active_optimizations),
            "total_completed": len(self.metrics_history),
            "active_models": list(self.active_optimizations.keys())
        }
    
    def clear_history(self) -> None:
        """Clear monitoring history."""
        self.metrics_history.clear()
        self.active_optimizations.clear()
        if TPRINT_AVAILABLE:
            tprint_success("🧹 Cleared monitoring history")
        else:
            self.logger.info("Cleared monitoring history")
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to file."""
        try:
            import json
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "metrics": [
                    {
                        "model_name": m.model_name,
                        "strategy": m.strategy,
                        "start_time": m.start_time,
                        "end_time": m.end_time,
                        "duration": m.duration,
                        "n_trials": m.n_trials,
                        "best_score": m.best_score,
                        "success": m.success,
                        "error": m.error,
                        "system_metrics": m.system_metrics
                    }
                    for m in self.metrics_history
                ],
                "summary": self.get_metrics_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"📁 Exported metrics to {filepath}")
            else:
                self.logger.info(f"Exported metrics to {filepath}")
            
        except Exception as e:
            raise MonitoringError(f"Failed to export metrics: {e}") from e