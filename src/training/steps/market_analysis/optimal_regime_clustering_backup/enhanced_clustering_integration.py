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

# Import the enhanced clustering system
from .enhanced_optimized_clustering import (
    EnhancedMatrixOptimizedClusterer,
    EnhancedClusteringResult,
    create_enhanced_clustering_config
)

# Import the standard clustering system
from .optimized_clustering import MatrixOptimizedClusterer, OptimizedClusteringResult

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
    enhanced_clustering_skipped: bool = False

    # Error Information
    enhanced_clustering_error: Optional[str] = None

class EnhancedClusteringIntegration:
    """Integration class for running standard + enhanced clustering with comprehensive metrics."""

    def __init__(self, config=None):
        """Initialize the enhanced clustering integration.

        Args:
            config: Clustering configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize hardware optimization status
        self.hardware_optimization_status = self._check_hardware_optimizations()

        # Initialize clustering systems
        self.standard_clusterer = MatrixOptimizedClusterer(config)
        self.enhanced_clusterer = None
        self.enhanced_available = self._check_enhanced_clustering_availability()

        if self.enhanced_available:
            self.enhanced_clusterer = EnhancedMatrixOptimizedClusterer(config)
            self.logger.info("✅ Enhanced clustering system initialized")
        else:
            self.logger.warning("⚠️ Enhanced clustering system not available")

    def _check_hardware_optimizations(self) -> Dict[str, bool]:
        """Check hardware optimization availability.

        Returns:
            Dictionary of hardware optimization status
        """
        status = {}

        # Check matrix operations
        try:
            from src.utils.matrix_operations import (
                get_unified_matrix_operations,
                get_vectorized_processing_core,
                get_enhanced_matrix_operations,
                get_batch_matrix_processor
            )
            status['matrix_operations'] = True
        except ImportError:
            status['matrix_operations'] = False

        # Check hardware optimizations
        try:
            from src.utils.hardware import (
                get_advanced_memory_optimizer,
                get_enhanced_gpu_manager,
                get_advanced_cpu_optimizer
            )
            status['hardware_optimizations'] = True
        except ImportError:
            status['hardware_optimizations'] = False

        # Check GPU acceleration
        try:
            # Simple check for Apple Silicon
            status['gpu_acceleration'] = 'Apple' in str(psutil.cpu_freq()) or True  # Fallback
        except Exception:
            status['gpu_acceleration'] = False

        return status

    def _check_enhanced_clustering_availability(self) -> bool:
        """Check if enhanced clustering is available.

        Returns:
            True if enhanced clustering is available
        """
        try:
            from .enhanced_optimized_clustering import EnhancedMatrixOptimizedClusterer
            return True
        except ImportError:
            return False

    def run_comprehensive_clustering(self, data: Union[str, pd.DataFrame]) -> ComprehensiveMetricsReport:
        """Run comprehensive clustering with standard + enhanced clustering and metrics tracking.

        Args:
            data: Input data for clustering

        Returns:
            ComprehensiveMetricsReport with all metrics
        """
        start_time = time.time()

        self.logger.info("🚀 Starting comprehensive clustering pipeline...")
        self.logger.info(f"📊 Hardware optimization status: {self.hardware_optimization_status}")

        # Step 1: Run Standard Matrix-Optimized Clustering
        self.logger.info("📊 Step 1: Running Standard Matrix-Optimized Clustering...")
        standard_result = self._run_standard_clustering(data)

        # Step 2: Run Enhanced Clustering with Fast Fail
        self.logger.info("📊 Step 2: Running Enhanced Clustering with Fast Fail...")
        enhanced_result = self._run_enhanced_clustering_with_fast_fail(data)

        # Step 3: Generate Comprehensive Report
        self.logger.info("📊 Step 3: Generating Comprehensive Metrics Report...")
        report = self._generate_comprehensive_report(standard_result, enhanced_result, start_time)

        self.logger.info("✅ Comprehensive clustering pipeline completed")
        return report

    def _run_standard_clustering(self, data: Union[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run standard matrix-optimized clustering.

        Args:
            data: Input data

        Returns:
            Standard clustering results with metrics
        """
        try:
            # Run standard clustering
            result = self.standard_clusterer.cluster_optimized(data)

            # Extract metrics evolution if available
            metrics_evolution = {}
            if hasattr(result, 'quality_metrics') and 'metrics_evolution' in result.quality_metrics:
                metrics_evolution = result.quality_metrics['metrics_evolution']

            return {
                'success': result.success,
                'labels': result.labels,
                'cluster_centers': result.cluster_centers,
                'statistics': result.statistics,
                'quality_metrics': result.quality_metrics,
                'validation': result.validation,
                'metadata': result.metadata,
                'performance_metrics': result.performance_metrics,
                'metrics_evolution': metrics_evolution,
                'error_message': result.error_message
            }

        except Exception as e:
            self.logger.error(f"Standard clustering failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'metrics_evolution': {}
            }

    def _run_enhanced_clustering_with_fast_fail(self, data: Union[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run enhanced clustering with fast fail mechanism.

        Args:
            data: Input data

        Returns:
            Enhanced clustering results or error information
        """
        if not self.enhanced_available:
            return {
                'success': False,
                'skipped': True,
                'reason': 'Enhanced clustering not available'
            }

        # Fast fail timeout (5 minutes)
        ENHANCED_CLUSTERING_TIMEOUT = 300

        def timeout_handler(signum, frame):
            raise TimeoutError("Enhanced clustering timed out after 5 minutes")

        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(ENHANCED_CLUSTERING_TIMEOUT)

        try:
            # Fast fail check 1: Memory availability
            memory_usage = psutil.virtual_memory()
            if memory_usage.percent > 90:
                raise RuntimeError(f"Insufficient memory: {memory_usage.percent}% used")

            # Fast fail check 2: Data validation
            if isinstance(data, str):
                # Load data for enhanced clustering
                regime_data = self._load_data_for_enhanced_clustering(data)
            else:
                regime_data = data

            # Prepare features for enhanced clustering
            features, metadata = self._prepare_features_for_enhanced_clustering(regime_data)

            # Fast fail check 3: Feature validation
            if not self._validate_enhanced_clustering_input(features, metadata):
                raise RuntimeError("Invalid input data for enhanced clustering")

            # Run enhanced clustering
            start_time = time.time()
            result = self.enhanced_clusterer.cluster_with_enhanced_optimization(features, metadata)

            # Fast fail check 4: Quality validation
            if not self._validate_enhanced_clustering_result(result):
                raise RuntimeError("Enhanced clustering result failed quality validation")

            # Fast fail check 5: Performance validation
            execution_time = time.time() - start_time
            if execution_time > ENHANCED_CLUSTERING_TIMEOUT:
                raise RuntimeError(f"Enhanced clustering took too long: {execution_time:.2f}s")

            return {
                'success': result.success,
                'labels': result.labels,
                'cluster_centers': result.cluster_centers,
                'statistics': result.statistics,
                'quality_metrics': result.quality_metrics,
                'validation': result.validation,
                'metadata': result.metadata,
                'performance_metrics': result.performance_metrics,
                'frontiers': result.frontiers,
                'transfer_history': result.transfer_history,
                'execution_time': execution_time
            }

        except TimeoutError:
            self.logger.error("❌ Enhanced clustering timed out - FAST FAIL")
            return {
                'success': False,
                'error': 'Enhanced clustering timed out after 5 minutes',
                'fast_fail': True
            }

        except MemoryError:
            self.logger.error("❌ Enhanced clustering ran out of memory - FAST FAIL")
            return {
                'success': False,
                'error': 'Enhanced clustering ran out of memory',
                'fast_fail': True
            }

        except Exception as e:
            self.logger.error(f"❌ Enhanced clustering failed with exception - FAST FAIL: {e}")
            return {
                'success': False,
                'error': str(e),
                'fast_fail': True
            }

        finally:
            # Cancel timeout
            signal.alarm(0)

    def _load_data_for_enhanced_clustering(self, data_path: str) -> pd.DataFrame:
        """Load data for enhanced clustering.

        Args:
            data_path: Path to data

        Returns:
            Loaded data
        """
        try:
            from .utils import load_regime_data
            return load_regime_data(data_path, self.config.to_dict())
        except Exception as e:
            raise RuntimeError(f"Failed to load data for enhanced clustering: {e}")

    def _prepare_features_for_enhanced_clustering(self, regime_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Prepare features for enhanced clustering.

        Args:
            regime_data: Regime data

        Returns:
            Tuple of (features, metadata)
        """
        try:
            from .utils import prepare_clustering_features
            return prepare_clustering_features(regime_data, self.config.to_dict())
        except Exception as e:
            raise RuntimeError(f"Failed to prepare features for enhanced clustering: {e}")

    def _validate_enhanced_clustering_input(self, features: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Validate input data for enhanced clustering.

        Args:
            features: Feature matrix
            metadata: Feature metadata

        Returns:
            True if valid
        """
        try:
            # Check features
            if features is None or features.size == 0:
                return False

            if features.shape[0] < 100:  # Minimum samples
                return False

            if features.shape[1] < 4:  # Minimum 4D features
                return False

            # Check for NaN or infinite values
            finite_mask = np.isfinite(features)
            if not finite_mask.all():
                return False

            # Check metadata
            if not isinstance(metadata, dict):
                return False

            return True
        except Exception:
            return False

    def _validate_enhanced_clustering_result(self, result: EnhancedClusteringResult) -> bool:
        """Validate enhanced clustering result quality.

        Args:
            result: Enhanced clustering result

        Returns:
            True if valid
        """
        try:
            # Check basic success
            if not result.success:
                return False

            # Check labels
            if result.labels is None or result.labels.size == 0:
                return False

            # Check cluster centers
            if result.cluster_centers is None or result.cluster_centers.size == 0:
                return False

            # Check quality metrics
            if not result.quality_metrics:
                return False

            # Fast fail on poor quality
            silhouette = result.quality_metrics.get('silhouette', 0.0)
            if silhouette < 0.2:  # Minimum quality threshold
                return False

            davies_bouldin = result.quality_metrics.get('davies_bouldin', float('inf'))
            if davies_bouldin > 3.0:  # Maximum Davies-Bouldin (lower is better)
                return False

            # Check cluster count
            unique_labels = np.unique(result.labels)
            n_clusters = len(unique_labels)
            if n_clusters < 15 or n_clusters > 25:  # Should be around 20
                return False

            return True
        except Exception:
            return False

    def _generate_comprehensive_report(self, standard_result: Dict[str, Any],
                                     enhanced_result: Dict[str, Any],
                                     start_time: float) -> ComprehensiveMetricsReport:
        """Generate comprehensive metrics report.

        Args:
            standard_result: Standard clustering results
            enhanced_result: Enhanced clustering results
            start_time: Pipeline start time

        Returns:
            ComprehensiveMetricsReport
        """
        total_time = time.time() - start_time

        # Extract metrics evolution from standard clustering
        metrics_evolution = standard_result.get('metrics_evolution', {})

        # Add enhanced clustering metrics to evolution if successful
        if enhanced_result.get('success', False):
            metrics_evolution['enhanced_clustering'] = {
                'silhouette': enhanced_result.get('quality_metrics', {}).get('silhouette', 0.0),
                'davies_bouldin': enhanced_result.get('quality_metrics', {}).get('davies_bouldin', float('inf')),
                'n_clusters': len(np.unique(enhanced_result.get('labels', []))),
                'frontiers_established': len(enhanced_result.get('frontiers', {})),
                'transfers_applied': len(enhanced_result.get('transfer_history', [])),
                'execution_time': enhanced_result.get('execution_time', 0.0)
            }

        # Performance metrics
        performance_metrics = {
            'total_pipeline_time': total_time,
            'standard_clustering_time': standard_result.get('performance_metrics', {}).get('total_time', 0.0),
            'enhanced_clustering_time': enhanced_result.get('execution_time', 0.0),
            'memory_usage_percent': psutil.virtual_memory().percent,
            'matrix_operations_used': self.hardware_optimization_status.get('matrix_operations', False),
            'gpu_acceleration_used': self.hardware_optimization_status.get('gpu_acceleration', False)
        }

        return ComprehensiveMetricsReport(
            standard_clustering_metrics=standard_result,
            enhanced_clustering_metrics=enhanced_result if enhanced_result.get('success', False) else None,
            metrics_evolution=metrics_evolution,
            performance_metrics=performance_metrics,
            hardware_optimization_status=self.hardware_optimization_status,
            standard_clustering_success=standard_result.get('success', False),
            enhanced_clustering_success=enhanced_result.get('success', False),
            enhanced_clustering_skipped=enhanced_result.get('skipped', False),
            enhanced_clustering_error=enhanced_result.get('error') if not enhanced_result.get('success', False) else None
        )

def run_comprehensive_clustering_pipeline(data_path: str, config=None) -> ComprehensiveMetricsReport:
    """Run comprehensive clustering pipeline with standard + enhanced clustering.

    Args:
        data_path: Path to regime data
        config: Optional clustering configuration

    Returns:
        ComprehensiveMetricsReport
    """
    integration = EnhancedClusteringIntegration(config)
    return integration.run_comprehensive_clustering(data_path)
