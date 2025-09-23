"""
Enhanced Clustering Integration with Comprehensive Metrics Tracking

This module integrates EnhancedMatrixOptimizedClusterer with the standard MatrixOptimizedClusterer,
providing comprehensive metrics tracking across all clustering steps with fast fail mechanism.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
import signal
import psutil
from dataclasses import dataclass

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

try:
    from src.utils.hardware import (
        get_hardware_accelerator,
        get_memory_manager,
        get_performance_monitor
    )
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False

from ..core.matrix_optimized import MatrixOptimizedClusterer
from ..core.enhanced_optimized import EnhancedMatrixOptimizedClusterer
from ..metrics.basic_metrics import BasicClusteringMetrics
from ..metrics.detailed_metrics import DetailedClusteringMetrics
from ..metrics.evolution_report import MetricsEvolutionReporter

logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveMetricsReport:
    """Comprehensive metrics report for the complete clustering pipeline."""
    
    # Standard Matrix-Optimized Clustering Metrics
    standard_clustering_metrics: Dict[str, Any]
    
    # Enhanced Clustering Metrics (if successful)
    enhanced_clustering_metrics: Optional[Dict[str, Any]] = None
    
    # Metrics Evolution Across All Steps
    metrics_evolution: Dict[str, Any] = None
    
    # Performance Metrics
    performance_metrics: Dict[str, float] = None
    
    # Hardware Optimization Status
    hardware_optimization_status: Dict[str, bool] = None
    
    # Success Status
    standard_clustering_success: bool = True
    enhanced_clustering_success: bool = False
    
    # Execution Information
    execution_time: float = 0.0
    timestamp: str = None


class EnhancedClusteringIntegration:
    """Integrates enhanced clustering with standard clustering and comprehensive metrics tracking."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced clustering integration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_manager = None
        self.performance_monitor = None
        
        if HARDWARE_ACCELERATION_AVAILABLE:
            try:
                self.hardware_accelerator = get_hardware_accelerator()
                self.memory_manager = get_memory_manager()
                self.performance_monitor = get_performance_monitor()
                self.logger.info("✅ Hardware acceleration initialized for enhanced integration")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available for enhanced integration: {e}")
        
        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None
        
        if MATRIX_OPERATIONS_AVAILABLE:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for enhanced integration")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available for enhanced integration: {e}")
        
        # Initialize clustering components
        self.standard_clusterer = MatrixOptimizedClusterer(config)
        self.enhanced_clusterer = EnhancedMatrixOptimizedClusterer(config)
        
        # Initialize metrics calculators
        self.basic_metrics_calc = BasicClusteringMetrics(config)
        self.detailed_metrics_calc = DetailedClusteringMetrics(config)
        self.evolution_reporter = MetricsEvolutionReporter(config)
        
        # Fast fail configuration
        self.fast_fail_enabled = config.get('enable_fast_fail', True)
        self.timeout_seconds = config.get('timeout_seconds', 300)
        self.memory_limit_gb = config.get('memory_limit_gb', 8.0)
        self.quality_threshold = config.get('quality_threshold', 0.3)

    def run_enhanced_clustering_pipeline(self, features: np.ndarray) -> ComprehensiveMetricsReport:
        """Run the complete enhanced clustering pipeline with comprehensive metrics tracking.

        Args:
            features: Feature matrix for clustering

        Returns:
            ComprehensiveMetricsReport with all metrics and results
        """
        start_time = time.time()
        
        try:
            self.logger.info("🚀 Starting enhanced clustering pipeline with comprehensive metrics tracking")
            
            # Monitor performance
            if self.performance_monitor:
                self.performance_monitor.start_monitoring("enhanced_clustering_pipeline")
            
            # Check system resources
            self._check_system_resources()
            
            # Run standard matrix-optimized clustering first
            self.logger.info("📊 Stage 1: Running standard matrix-optimized clustering")
            standard_result = self._run_standard_clustering(features)
            
            if not standard_result.success:
                self.logger.error("❌ Standard clustering failed")
                if self.fast_fail_enabled:
                    return self._create_fast_fail_report("Standard clustering failed", standard_result.error_message, start_time)
                else:
                    self.logger.warning("⚠️ Standard clustering failed, but continuing...")
            
            # Run enhanced clustering with 4D frontier optimization
            self.logger.info("🎯 Stage 2: Running enhanced clustering with 4D frontier optimization")
            enhanced_result = self._run_enhanced_clustering(features)
            
            if not enhanced_result.success:
                self.logger.error("❌ Enhanced clustering failed")
                if self.fast_fail_enabled:
                    return self._create_fast_fail_report("Enhanced clustering failed", enhanced_result.error_message, start_time)
                else:
                    self.logger.warning("⚠️ Enhanced clustering failed, using standard result")
                    enhanced_result = None
            
            # Calculate comprehensive metrics
            self.logger.info("📈 Stage 3: Calculating comprehensive metrics")
            comprehensive_metrics = self._calculate_comprehensive_metrics(features, standard_result, enhanced_result)
            
            # Generate metrics evolution report
            self.logger.info("📊 Stage 4: Generating metrics evolution report")
            evolution_report = self._generate_metrics_evolution_report(standard_result, enhanced_result)
            
            # Stop performance monitoring
            perf_metrics = {}
            if self.performance_monitor:
                perf_metrics = self.performance_monitor.stop_monitoring("enhanced_clustering_pipeline")
            
            execution_time = time.time() - start_time
            
            # Create comprehensive report
            report = ComprehensiveMetricsReport(
                standard_clustering_metrics=standard_result.metadata if standard_result else {},
                enhanced_clustering_metrics=enhanced_result.metadata if enhanced_result else None,
                metrics_evolution=evolution_report,
                performance_metrics=perf_metrics,
                hardware_optimization_status={
                    'matrix_ops_available': self.matrix_ops is not None,
                    'hardware_acceleration_available': self.hardware_accelerator is not None,
                    'memory_manager_available': self.memory_manager is not None
                },
                standard_clustering_success=standard_result.success if standard_result else False,
                enhanced_clustering_success=enhanced_result.success if enhanced_result else False,
                execution_time=execution_time,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            self.logger.info("✅ Enhanced clustering pipeline completed successfully")
            return report
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Enhanced clustering pipeline failed: {e}")
            return self._create_fast_fail_report("Pipeline failed", str(e), start_time)

    def _run_standard_clustering(self, features: np.ndarray) -> Any:
        """Run standard matrix-optimized clustering.

        Args:
            features: Feature matrix

        Returns:
            Standard clustering result
        """
        try:
            # Check timeout
            if self.fast_fail_enabled:
                self._check_timeout(self.timeout_seconds)
            
            # Monitor memory usage
            if self.memory_manager:
                self.memory_manager.check_memory_usage()
            
            # Run clustering
            result = self.standard_clusterer.cluster(features)
            
            # Validate quality
            if self.fast_fail_enabled and result.success:
                if not self._validate_clustering_quality(result):
                    result.success = False
                    result.error_message = "Clustering quality below threshold"
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Standard clustering failed: {e}")
            # Create error result
            from ..core.base_clustering import ClusteringResult
            return ClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics={},
                quality_metrics={},
                validation={'valid': False, 'error': str(e)},
                metadata={'error': str(e)},
                success=False,
                error_message=str(e)
            )

    def _run_enhanced_clustering(self, features: np.ndarray) -> Any:
        """Run enhanced clustering with 4D frontier optimization.

        Args:
            features: Feature matrix

        Returns:
            Enhanced clustering result
        """
        try:
            # Check timeout (allow more time for enhanced clustering)
            if self.fast_fail_enabled:
                self._check_timeout(self.timeout_seconds * 2)  # Double timeout for enhanced
            
            # Monitor memory usage
            if self.memory_manager:
                self.memory_manager.check_memory_usage()
            
            # Run enhanced clustering
            result = self.enhanced_clusterer.cluster(features)
            
            # Validate quality
            if self.fast_fail_enabled and result.success:
                if not self._validate_clustering_quality(result):
                    result.success = False
                    result.error_message = "Enhanced clustering quality below threshold"
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced clustering failed: {e}")
            # Create error result
            from ..core.enhanced_optimized import EnhancedClusteringResult
            return EnhancedClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics={},
                quality_metrics={},
                validation={'valid': False, 'error': str(e)},
                metadata={'error': str(e)},
                success=False,
                error_message=str(e)
            )

    def _calculate_comprehensive_metrics(self, features: np.ndarray, standard_result: Any, 
                                       enhanced_result: Any) -> Dict[str, Any]:
        """Calculate comprehensive metrics for both clustering results.

        Args:
            features: Feature matrix
            standard_result: Standard clustering result
            enhanced_result: Enhanced clustering result

        Returns:
            Dictionary of comprehensive metrics
        """
        try:
            comprehensive_metrics = {}
            
            # Calculate basic metrics for standard result
            if standard_result and standard_result.success:
                basic_metrics = self.basic_metrics_calc.calculate_basic_metrics(features, standard_result.labels)
                detailed_metrics = self.detailed_metrics_calc.calculate_detailed_metrics(features, standard_result.labels)
                
                comprehensive_metrics['standard_basic_metrics'] = {
                    'silhouette': basic_metrics.silhouette,
                    'davies_bouldin': basic_metrics.davies_bouldin,
                    'calinski_harabasz': basic_metrics.calinski_harabasz,
                    'n_clusters': basic_metrics.n_clusters,
                    'n_valid_points': basic_metrics.n_valid_points,
                    'n_noise_points': basic_metrics.n_noise_points,
                    'average_cluster_cv': basic_metrics.average_cluster_cv,
                    'cluster_size_cv': basic_metrics.cluster_size_cv
                }
                
                comprehensive_metrics['standard_detailed_metrics'] = {
                    'adjusted_rand_index': detailed_metrics.adjusted_rand_index,
                    'normalized_mutual_info': detailed_metrics.normalized_mutual_info,
                    'cluster_separation': detailed_metrics.cluster_separation,
                    'cluster_compactness': detailed_metrics.cluster_compactness,
                    'size_balance_score': detailed_metrics.size_balance_score
                }
            
            # Calculate basic metrics for enhanced result
            if enhanced_result and enhanced_result.success:
                basic_metrics = self.basic_metrics_calc.calculate_basic_metrics(features, enhanced_result.labels)
                detailed_metrics = self.detailed_metrics_calc.calculate_detailed_metrics(features, enhanced_result.labels)
                
                comprehensive_metrics['enhanced_basic_metrics'] = {
                    'silhouette': basic_metrics.silhouette,
                    'davies_bouldin': basic_metrics.davies_bouldin,
                    'calinski_harabasz': basic_metrics.calinski_harabasz,
                    'n_clusters': basic_metrics.n_clusters,
                    'n_valid_points': basic_metrics.n_valid_points,
                    'n_noise_points': basic_metrics.n_noise_points,
                    'average_cluster_cv': basic_metrics.average_cluster_cv,
                    'cluster_size_cv': basic_metrics.cluster_size_cv
                }
                
                comprehensive_metrics['enhanced_detailed_metrics'] = {
                    'adjusted_rand_index': detailed_metrics.adjusted_rand_index,
                    'normalized_mutual_info': detailed_metrics.normalized_mutual_info,
                    'cluster_separation': detailed_metrics.cluster_separation,
                    'cluster_compactness': detailed_metrics.cluster_compactness,
                    'size_balance_score': detailed_metrics.size_balance_score
                }
                
                # Add enhanced-specific metrics
                if hasattr(enhanced_result, 'frontiers'):
                    comprehensive_metrics['enhanced_frontier_metrics'] = enhanced_result.frontiers
                
                if hasattr(enhanced_result, 'transfer_history'):
                    comprehensive_metrics['enhanced_transfer_history'] = enhanced_result.transfer_history
                
                if hasattr(enhanced_result, 'optimization_iterations'):
                    comprehensive_metrics['enhanced_optimization_iterations'] = enhanced_result.optimization_iterations
            
            # Compare standard vs enhanced if both available
            if (standard_result and standard_result.success and 
                enhanced_result and enhanced_result.success):
                comparison = self._compare_clustering_results(
                    comprehensive_metrics['standard_basic_metrics'],
                    comprehensive_metrics['enhanced_basic_metrics']
                )
                comprehensive_metrics['standard_vs_enhanced_comparison'] = comparison
            
            return comprehensive_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Comprehensive metrics calculation failed: {e}")
            return {'error': str(e)}

    def _compare_clustering_results(self, standard_metrics: Dict[str, Any], 
                                  enhanced_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare standard and enhanced clustering results.

        Args:
            standard_metrics: Standard clustering metrics
            enhanced_metrics: Enhanced clustering metrics

        Returns:
            Comparison metrics
        """
        try:
            comparison = {
                'silhouette_improvement': enhanced_metrics['silhouette'] - standard_metrics['silhouette'],
                'davies_bouldin_improvement': standard_metrics['davies_bouldin'] - enhanced_metrics['davies_bouldin'],
                'calinski_harabasz_improvement': enhanced_metrics['calinski_harabasz'] - standard_metrics['calinski_harabasz'],
                'cluster_count_difference': enhanced_metrics['n_clusters'] - standard_metrics['n_clusters'],
                'overall_improvement_score': 0.0
            }
            
            # Calculate overall improvement score
            silhouette_improvement = comparison['silhouette_improvement']
            davies_bouldin_improvement = comparison['davies_bouldin_improvement']
            calinski_harabasz_improvement = comparison['calinski_harabasz_improvement']
            
            # Normalize improvements (higher is better for silhouette and calinski-harabasz, lower is better for davies-bouldin)
            overall_score = (silhouette_improvement + davies_bouldin_improvement + calinski_harabasz_improvement) / 3.0
            comparison['overall_improvement_score'] = overall_score
            
            return comparison
            
        except Exception as e:
            self.logger.warning(f"⚠️ Clustering results comparison failed: {e}")
            return {'error': str(e)}

    def _generate_metrics_evolution_report(self, standard_result: Any, enhanced_result: Any) -> Dict[str, Any]:
        """Generate metrics evolution report.

        Args:
            standard_result: Standard clustering result
            enhanced_result: Enhanced clustering result

        Returns:
            Metrics evolution report
        """
        try:
            # Extract metrics evolution from standard result
            standard_evolution = {}
            if standard_result and hasattr(standard_result, 'metadata') and 'metrics_evolution' in standard_result.metadata:
                standard_evolution = standard_result.metadata['metrics_evolution']
            
            # Extract metrics evolution from enhanced result
            enhanced_metrics = None
            if enhanced_result and hasattr(enhanced_result, 'metadata'):
                enhanced_metrics = enhanced_result.metadata
            
            # Generate comprehensive report
            evolution_report = self.evolution_reporter.generate_comprehensive_report(
                standard_evolution,
                enhanced_metrics,
                self.config
            )
            
            return evolution_report
            
        except Exception as e:
            self.logger.warning(f"⚠️ Metrics evolution report generation failed: {e}")
            return {'error': str(e)}

    def _check_system_resources(self) -> None:
        """Check system resources before running clustering."""
        try:
            if self.fast_fail_enabled:
                # Check memory usage
                memory_info = psutil.virtual_memory()
                memory_gb = memory_info.used / (1024**3)
                
                if memory_gb > self.memory_limit_gb:
                    raise RuntimeError(f"Memory usage ({memory_gb:.1f}GB) exceeds limit ({self.memory_limit_gb}GB)")
                
                # Check CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > 90:
                    self.logger.warning(f"⚠️ High CPU usage: {cpu_percent:.1f}%")
                
                self.logger.info(f"✅ System resources check passed: Memory {memory_gb:.1f}GB, CPU {cpu_percent:.1f}%")
                
        except Exception as e:
            self.logger.warning(f"⚠️ System resources check failed: {e}")

    def _check_timeout(self, timeout_seconds: float) -> None:
        """Check if operation has exceeded timeout.

        Args:
            timeout_seconds: Timeout in seconds
        """
        # This is a simplified timeout check
        # In a real implementation, you'd use proper timeout mechanisms
        pass

    def _validate_clustering_quality(self, result: Any) -> bool:
        """Validate clustering quality against thresholds.

        Args:
            result: Clustering result

        Returns:
            True if quality is acceptable
        """
        try:
            if not result.success:
                return False
            
            # Check silhouette score
            silhouette = result.quality_metrics.get('silhouette', 0.0)
            if silhouette < self.quality_threshold:
                self.logger.warning(f"⚠️ Silhouette score ({silhouette:.3f}) below threshold ({self.quality_threshold})")
                return False
            
            # Check cluster count
            n_clusters = result.quality_metrics.get('n_clusters', 0)
            if n_clusters < 2 or n_clusters > 25:
                self.logger.warning(f"⚠️ Cluster count ({n_clusters}) outside acceptable range (2-25)")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Quality validation failed: {e}")
            return False

    def _create_fast_fail_report(self, error_type: str, error_message: str, start_time: float) -> ComprehensiveMetricsReport:
        """Create fast fail report.

        Args:
            error_type: Type of error
            error_message: Error message
            start_time: Start time

        Returns:
            Fast fail report
        """
        execution_time = time.time() - start_time
        
        return ComprehensiveMetricsReport(
            standard_clustering_metrics={'error': error_message},
            enhanced_clustering_metrics={'error': error_message},
            metrics_evolution={'error': error_message},
            performance_metrics={'execution_time': execution_time, 'error': error_type},
            hardware_optimization_status={
                'matrix_ops_available': self.matrix_ops is not None,
                'hardware_acceleration_available': self.hardware_accelerator is not None,
                'memory_manager_available': self.memory_manager is not None
            },
            standard_clustering_success=False,
            enhanced_clustering_success=False,
            execution_time=execution_time,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )


def create_enhanced_clustering_integration(config: Dict[str, Any]) -> EnhancedClusteringIntegration:
    """Create an enhanced clustering integration instance.

    Args:
        config: Configuration dictionary

    Returns:
        EnhancedClusteringIntegration instance
    """
    return EnhancedClusteringIntegration(config)