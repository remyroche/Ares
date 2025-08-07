# src/training/memory_profiler.py

import gc
import os
import psutil
import sys
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class MemoryProfiler:
    """
    Comprehensive memory profiler for detecting memory leaks and optimizing memory usage
    during training processes.
    """
    
    def __init__(self, enable_tracemalloc: bool = True, enable_continuous_monitoring: bool = True):
        self.logger = system_logger.getChild("MemoryProfiler")
        self.enable_tracemalloc = enable_tracemalloc
        self.enable_continuous_monitoring = enable_continuous_monitoring
        
        # Memory tracking
        self.memory_snapshots = deque(maxlen=1000)  # Circular buffer for memory snapshots
        self.object_counts = defaultdict(int)
        self.object_sizes = defaultdict(int)
        self.allocation_traces = []
        
        # Memory leak detection
        self.baseline_memory = None
        self.leak_threshold_mb = 100  # MB
        self.leak_detection_window = 10  # Number of snapshots to analyze
        
        # Continuous monitoring
        self.monitoring_thread = None
        self.monitoring_interval = 30  # seconds
        self.is_monitoring = False
        
        # Process information
        self.process = psutil.Process(os.getpid())
        
        # Start tracemalloc if enabled
        if self.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
            self.logger.info("Memory tracing started")
        
        # Start continuous monitoring if enabled
        if self.enable_continuous_monitoring:
            self.start_continuous_monitoring()
    
    def take_snapshot(self, label: str = None) -> Dict[str, Any]:
        """Take a comprehensive memory snapshot."""
        timestamp = datetime.now()
        
        # System memory info
        system_memory = psutil.virtual_memory()
        
        # Process memory info
        process_memory = self.process.memory_info()
        process_memory_percent = self.process.memory_percent()
        
        # Python object counts
        object_counts = self._get_object_counts()
        
        # Tracemalloc snapshot if enabled
        tracemalloc_info = None
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc_info = self._get_tracemalloc_info()
        
        # Garbage collector info
        gc_info = self._get_gc_info()
        
        snapshot = {
            "timestamp": timestamp,
            "label": label or f"snapshot_{len(self.memory_snapshots)}",
            "system_memory": {
                "total_gb": system_memory.total / (1024**3),
                "available_gb": system_memory.available / (1024**3),
                "used_gb": system_memory.used / (1024**3),
                "percent": system_memory.percent
            },
            "process_memory": {
                "rss_mb": process_memory.rss / (1024**2),
                "vms_mb": process_memory.vms / (1024**2),
                "percent": process_memory_percent,
                "num_fds": self.process.num_fds() if hasattr(self.process, 'num_fds') else None
            },
            "object_counts": object_counts,
            "tracemalloc_info": tracemalloc_info,
            "gc_info": gc_info,
            "thread_count": threading.active_count()
        }
        
        self.memory_snapshots.append(snapshot)
        
        # Check for memory leaks
        if len(self.memory_snapshots) >= self.leak_detection_window:
            leak_info = self._detect_memory_leaks()
            if leak_info:
                snapshot["leak_detection"] = leak_info
        
        self.logger.info(f"Memory snapshot taken: {label}, Process RSS: {process_memory.rss / (1024**2):.1f}MB")
        return snapshot
    
    def _get_object_counts(self) -> Dict[str, int]:
        """Get counts of Python objects by type."""
        object_counts = defaultdict(int)
        
        # Count objects by type
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            object_counts[obj_type] += 1
        
        # Convert to regular dict and sort by count
        sorted_counts = dict(sorted(object_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Return top 20 most common object types
        return dict(list(sorted_counts.items())[:20])
    
    def _get_tracemalloc_info(self) -> Dict[str, Any]:
        """Get tracemalloc memory allocation information."""
        if not tracemalloc.is_tracing():
            return None
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        # Get top 10 memory allocations
        top_allocations = []
        for index, stat in enumerate(top_stats[:10]):
            top_allocations.append({
                "filename": stat.traceback.format()[0] if stat.traceback.format() else "unknown",
                "size_mb": stat.size / (1024**2),
                "count": stat.count
            })
        
        # Get total memory usage
        total_size = sum(stat.size for stat in top_stats)
        
        return {
            "total_size_mb": total_size / (1024**2),
            "total_count": len(top_stats),
            "top_allocations": top_allocations
        }
    
    def _get_gc_info(self) -> Dict[str, Any]:
        """Get garbage collector information."""
        return {
            "counts": gc.get_count(),
            "stats": gc.get_stats() if hasattr(gc, 'get_stats') else None,
            "flags": gc.get_debug(),
            "threshold": gc.get_threshold()
        }
    
    def _detect_memory_leaks(self) -> Optional[Dict[str, Any]]:
        """Detect potential memory leaks by analyzing memory growth trends."""
        if len(self.memory_snapshots) < self.leak_detection_window:
            return None
        
        recent_snapshots = list(self.memory_snapshots)[-self.leak_detection_window:]
        
        # Analyze RSS memory growth
        rss_values = [s["process_memory"]["rss_mb"] for s in recent_snapshots]
        rss_growth = rss_values[-1] - rss_values[0]
        
        # Analyze object count growth
        object_growth = {}
        first_snapshot = recent_snapshots[0]
        last_snapshot = recent_snapshots[-1]
        
        for obj_type in last_snapshot["object_counts"]:
            first_count = first_snapshot["object_counts"].get(obj_type, 0)
            last_count = last_snapshot["object_counts"][obj_type]
            growth = last_count - first_count
            if growth > 100:  # Only track significant growth
                object_growth[obj_type] = growth
        
        # Determine if there's a potential leak
        potential_leak = False
        leak_indicators = []
        
        if rss_growth > self.leak_threshold_mb:
            potential_leak = True
            leak_indicators.append(f"RSS memory growth: {rss_growth:.1f}MB")
        
        if object_growth:
            top_growth = sorted(object_growth.items(), key=lambda x: x[1], reverse=True)[:5]
            for obj_type, growth in top_growth:
                if growth > 1000:  # Significant object growth
                    potential_leak = True
                    leak_indicators.append(f"{obj_type} objects increased by {growth}")
        
        if potential_leak:
            return {
                "detected": True,
                "rss_growth_mb": rss_growth,
                "object_growth": object_growth,
                "indicators": leak_indicators,
                "window_size": self.leak_detection_window,
                "recommendation": self._generate_leak_recommendations(rss_growth, object_growth)
            }
        
        return None
    
    def _generate_leak_recommendations(self, rss_growth: float, object_growth: Dict[str, int]) -> List[str]:
        """Generate recommendations for addressing memory leaks."""
        recommendations = []
        
        if rss_growth > self.leak_threshold_mb:
            recommendations.append("Consider calling gc.collect() more frequently")
            recommendations.append("Review large data structures and clear them when no longer needed")
        
        if "DataFrame" in object_growth and object_growth["DataFrame"] > 100:
            recommendations.append("Clear unused pandas DataFrames explicitly")
            recommendations.append("Use del statements or set DataFrames to None")
        
        if "ndarray" in object_growth and object_growth["ndarray"] > 100:
            recommendations.append("Clear unused numpy arrays")
            recommendations.append("Consider using numpy.memmap for large arrays")
        
        if "dict" in object_growth and object_growth["dict"] > 1000:
            recommendations.append("Review dictionary caches and implement size limits")
            recommendations.append("Clear unused dictionaries or use weak references")
        
        if not recommendations:
            recommendations.append("Monitor for specific object types causing growth")
            recommendations.append("Use memory profiling tools for detailed analysis")
        
        return recommendations
    
    def start_continuous_monitoring(self):
        """Start continuous memory monitoring in a background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Continuous memory monitoring started")
    
    def stop_continuous_monitoring(self):
        """Stop continuous memory monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1)
        self.logger.info("Continuous memory monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                snapshot = self.take_snapshot("continuous_monitoring")
                
                # Check for high memory usage
                if snapshot["process_memory"]["percent"] > 80:
                    self.logger.warning(f"High memory usage detected: {snapshot['process_memory']['percent']:.1f}%")
                
                # Check for potential leaks
                if "leak_detection" in snapshot and snapshot["leak_detection"]["detected"]:
                    self.logger.warning("Potential memory leak detected!")
                    for indicator in snapshot["leak_detection"]["indicators"]:
                        self.logger.warning(f"  - {indicator}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def analyze_memory_trends(self, window_size: int = 50) -> Dict[str, Any]:
        """Analyze memory usage trends over time."""
        if len(self.memory_snapshots) < 2:
            return {"status": "insufficient_data"}
        
        recent_snapshots = list(self.memory_snapshots)[-window_size:]
        
        # Extract time series data
        timestamps = [s["timestamp"] for s in recent_snapshots]
        rss_values = [s["process_memory"]["rss_mb"] for s in recent_snapshots]
        system_memory_percent = [s["system_memory"]["percent"] for s in recent_snapshots]
        
        # Calculate trends
        if len(rss_values) > 1:
            rss_trend = np.polyfit(range(len(rss_values)), rss_values, 1)[0]  # Slope
            system_trend = np.polyfit(range(len(system_memory_percent)), system_memory_percent, 1)[0]
        else:
            rss_trend = 0
            system_trend = 0
        
        # Calculate statistics
        rss_stats = {
            "mean": np.mean(rss_values),
            "std": np.std(rss_values),
            "min": np.min(rss_values),
            "max": np.max(rss_values),
            "trend_mb_per_snapshot": rss_trend
        }
        
        system_stats = {
            "mean_percent": np.mean(system_memory_percent),
            "max_percent": np.max(system_memory_percent),
            "trend_percent_per_snapshot": system_trend
        }
        
        return {
            "status": "success",
            "window_size": len(recent_snapshots),
            "time_range": {
                "start": timestamps[0].isoformat(),
                "end": timestamps[-1].isoformat()
            },
            "rss_stats": rss_stats,
            "system_stats": system_stats,
            "growing_objects": self._find_growing_object_types(recent_snapshots)
        }
    
    def _find_growing_object_types(self, snapshots: List[Dict[str, Any]]) -> Dict[str, float]:
        """Find object types that are consistently growing."""
        if len(snapshots) < 2:
            return {}
        
        growing_objects = {}
        first_snapshot = snapshots[0]
        last_snapshot = snapshots[-1]
        
        # Compare object counts between first and last snapshot
        for obj_type in last_snapshot["object_counts"]:
            first_count = first_snapshot["object_counts"].get(obj_type, 0)
            last_count = last_snapshot["object_counts"][obj_type]
            
            if first_count > 0:
                growth_rate = (last_count - first_count) / first_count
                if growth_rate > 0.1:  # 10% growth threshold
                    growing_objects[obj_type] = growth_rate
        
        return dict(sorted(growing_objects.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection and report on freed memory."""
        before_snapshot = self.take_snapshot("before_gc")
        
        # Force garbage collection
        collected = []
        for generation in range(3):
            collected.append(gc.collect(generation))
        
        after_snapshot = self.take_snapshot("after_gc")
        
        # Calculate memory freed
        memory_freed = before_snapshot["process_memory"]["rss_mb"] - after_snapshot["process_memory"]["rss_mb"]
        
        return {
            "memory_freed_mb": memory_freed,
            "objects_collected": collected,
            "before_rss_mb": before_snapshot["process_memory"]["rss_mb"],
            "after_rss_mb": after_snapshot["process_memory"]["rss_mb"]
        }
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Perform comprehensive memory optimization."""
        self.logger.info("Starting memory optimization...")
        
        optimization_results = {}
        
        # 1. Force garbage collection
        gc_results = self.force_garbage_collection()
        optimization_results["garbage_collection"] = gc_results
        
        # 2. Clear internal caches
        optimization_results["cache_clearing"] = self._clear_internal_caches()
        
        # 3. Analyze current state
        current_snapshot = self.take_snapshot("post_optimization")
        optimization_results["final_state"] = current_snapshot
        
        self.logger.info(f"Memory optimization completed. Freed {gc_results['memory_freed_mb']:.1f}MB")
        return optimization_results
    
    def _clear_internal_caches(self) -> Dict[str, Any]:
        """Clear internal profiler caches to free memory."""
        initial_snapshots = len(self.memory_snapshots)
        initial_traces = len(self.allocation_traces)
        
        # Keep only recent snapshots
        if len(self.memory_snapshots) > 100:
            # Convert to list, slice, and convert back to deque
            recent_snapshots = list(self.memory_snapshots)[-50:]
            self.memory_snapshots.clear()
            self.memory_snapshots.extend(recent_snapshots)
        
        # Clear allocation traces
        self.allocation_traces.clear()
        
        # Clear object tracking
        self.object_counts.clear()
        self.object_sizes.clear()
        
        return {
            "snapshots_cleared": initial_snapshots - len(self.memory_snapshots),
            "traces_cleared": initial_traces,
            "snapshots_retained": len(self.memory_snapshots)
        }
    
    def generate_memory_report(self) -> Dict[str, Any]:
        """Generate a comprehensive memory usage report."""
        if not self.memory_snapshots:
            return {"status": "no_data"}
        
        # Analyze trends
        trends = self.analyze_memory_trends()
        
        # Get current state
        current_snapshot = self.take_snapshot("report_generation")
        
        # Generate recommendations
        recommendations = self._generate_memory_recommendations(current_snapshot, trends)
        
        return {
            "status": "success",
            "generated_at": datetime.now().isoformat(),
            "current_state": current_snapshot,
            "trends": trends,
            "recommendations": recommendations,
            "total_snapshots": len(self.memory_snapshots),
            "monitoring_active": self.is_monitoring
        }
    
    def _generate_memory_recommendations(self, current_snapshot: Dict[str, Any], 
                                       trends: Dict[str, Any]) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        # High memory usage
        if current_snapshot["process_memory"]["percent"] > 80:
            recommendations.append("Process memory usage is high (>80%). Consider optimizing data structures.")
        
        # Growing memory trend
        if trends.get("rss_stats", {}).get("trend_mb_per_snapshot", 0) > 1:
            recommendations.append("Memory usage is growing over time. Check for memory leaks.")
        
        # High object counts
        object_counts = current_snapshot.get("object_counts", {})
        if object_counts.get("DataFrame", 0) > 100:
            recommendations.append("High number of DataFrames. Consider clearing unused ones.")
        if object_counts.get("ndarray", 0) > 500:
            recommendations.append("High number of numpy arrays. Consider memory-efficient alternatives.")
        if object_counts.get("dict", 0) > 1000:
            recommendations.append("High number of dictionaries. Review caching strategies.")
        
        # Growing object types
        growing_objects = trends.get("growing_objects", {})
        if growing_objects:
            top_growing = list(growing_objects.keys())[:3]
            recommendations.append(f"Object types showing growth: {', '.join(top_growing)}")
        
        if not recommendations:
            recommendations.append("Memory usage appears to be within normal parameters.")
        
        return recommendations
    
    def __del__(self):
        """Cleanup when profiler is destroyed."""
        self.stop_continuous_monitoring()
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()


class MemoryLeakDetector:
    """
    Specialized memory leak detector for long-running training processes.
    """
    
    def __init__(self, profiler: MemoryProfiler):
        self.profiler = profiler
        self.logger = system_logger.getChild("MemoryLeakDetector")
        self.leak_alerts = []
        
    def check_for_leaks(self, threshold_mb: float = 100, window_size: int = 10) -> Dict[str, Any]:
        """Check for memory leaks using multiple detection methods."""
        results = {
            "leak_detected": False,
            "detection_methods": {},
            "recommendations": []
        }
        
        # Method 1: RSS growth analysis
        rss_analysis = self._analyze_rss_growth(threshold_mb, window_size)
        results["detection_methods"]["rss_growth"] = rss_analysis
        
        # Method 2: Object count analysis
        object_analysis = self._analyze_object_growth(window_size)
        results["detection_methods"]["object_growth"] = object_analysis
        
        # Method 3: Tracemalloc analysis
        if self.profiler.enable_tracemalloc:
            tracemalloc_analysis = self._analyze_tracemalloc_growth()
            results["detection_methods"]["tracemalloc"] = tracemalloc_analysis
        
        # Determine overall leak status
        leak_indicators = sum([
            rss_analysis.get("leak_detected", False),
            object_analysis.get("leak_detected", False),
            results["detection_methods"].get("tracemalloc", {}).get("leak_detected", False)
        ])
        
        if leak_indicators >= 2:  # Require multiple indicators
            results["leak_detected"] = True
            results["confidence"] = "high" if leak_indicators == 3 else "medium"
            results["recommendations"] = self._generate_leak_recommendations(results)
            
            # Log alert
            self.leak_alerts.append({
                "timestamp": datetime.now(),
                "detection_results": results
            })
            
            self.logger.warning("Memory leak detected with multiple indicators!")
        
        return results
    
    def _analyze_rss_growth(self, threshold_mb: float, window_size: int) -> Dict[str, Any]:
        """Analyze RSS memory growth for leak detection."""
        if len(self.profiler.memory_snapshots) < window_size:
            return {"status": "insufficient_data"}
        
        recent_snapshots = list(self.profiler.memory_snapshots)[-window_size:]
        rss_values = [s["process_memory"]["rss_mb"] for s in recent_snapshots]
        
        # Calculate growth
        initial_rss = rss_values[0]
        final_rss = rss_values[-1]
        growth = final_rss - initial_rss
        growth_rate = growth / initial_rss if initial_rss > 0 else 0
        
        return {
            "status": "success",
            "initial_rss_mb": initial_rss,
            "final_rss_mb": final_rss,
            "growth_mb": growth,
            "growth_rate": growth_rate,
            "leak_detected": growth > threshold_mb and growth_rate > 0.1
        }
    
    def _analyze_object_growth(self, window_size: int) -> Dict[str, Any]:
        """Analyze object count growth for leak detection."""
        if len(self.profiler.memory_snapshots) < window_size:
            return {"status": "insufficient_data"}
        
        recent_snapshots = list(self.profiler.memory_snapshots)[-window_size:]
        first_snapshot = recent_snapshots[0]
        last_snapshot = recent_snapshots[-1]
        
        suspicious_growth = {}
        leak_detected = False
        
        for obj_type in last_snapshot.get("object_counts", {}):
            first_count = first_snapshot.get("object_counts", {}).get(obj_type, 0)
            last_count = last_snapshot["object_counts"][obj_type]
            
            if first_count > 0:
                growth = last_count - first_count
                growth_rate = growth / first_count
                
                # Flag suspicious growth
                if growth > 1000 and growth_rate > 1.0:  # Doubled and significant absolute growth
                    suspicious_growth[obj_type] = {
                        "growth": growth,
                        "growth_rate": growth_rate,
                        "initial_count": first_count,
                        "final_count": last_count
                    }
                    leak_detected = True
        
        return {
            "status": "success",
            "suspicious_growth": suspicious_growth,
            "leak_detected": leak_detected,
            "analysis_window": window_size
        }
    
    def _analyze_tracemalloc_growth(self) -> Dict[str, Any]:
        """Analyze tracemalloc data for leak detection."""
        if not self.profiler.enable_tracemalloc or not tracemalloc.is_tracing():
            return {"status": "tracemalloc_disabled"}
        
        # This is a simplified analysis - in practice, you'd want to compare
        # snapshots over time and look for consistently growing allocations
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        # Look for large allocations that might indicate leaks
        large_allocations = [
            stat for stat in top_stats 
            if stat.size > 10 * 1024 * 1024  # > 10MB
        ]
        
        return {
            "status": "success",
            "large_allocations": len(large_allocations),
            "total_allocations": len(top_stats),
            "leak_detected": len(large_allocations) > 5  # Arbitrary threshold
        }
    
    def _generate_leak_recommendations(self, detection_results: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on leak detection results."""
        recommendations = []
        
        rss_growth = detection_results["detection_methods"].get("rss_growth", {})
        if rss_growth.get("leak_detected"):
            recommendations.append("RSS memory growth detected. Investigate large data structures.")
        
        object_growth = detection_results["detection_methods"].get("object_growth", {})
        if object_growth.get("leak_detected"):
            suspicious = object_growth.get("suspicious_growth", {})
            for obj_type in suspicious:
                recommendations.append(f"Investigate {obj_type} object growth")
        
        tracemalloc_data = detection_results["detection_methods"].get("tracemalloc", {})
        if tracemalloc_data.get("leak_detected"):
            recommendations.append("Large memory allocations detected. Use tracemalloc for detailed analysis.")
        
        # General recommendations
        recommendations.extend([
            "Force garbage collection: gc.collect()",
            "Clear unused variables: del variable_name",
            "Review caching strategies and implement size limits",
            "Monitor for circular references"
        ])
        
        return recommendations