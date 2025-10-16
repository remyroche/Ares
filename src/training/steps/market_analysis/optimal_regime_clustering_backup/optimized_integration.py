"""
Optimized HMM Clustering Integration

This module demonstrates how to integrate the optimized clustering algorithms
with the existing HMM regime discovery pipeline.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Import optimized clustering components
from .optimized_clustering import (
    OptimizedRegimeClusterer,
    OptimizedClusteringResult,
    create_optimized_clusterer,
    cluster_hmm_regimes_optimized
)
from .vectorized_operations import VectorizedClusteringOperations
from .config import OptimalClusteringConfig
from .performance_benchmark import ClusteringPerformanceBenchmark

logger = logging.getLogger(__name__)

class OptimizedHMMClusteringIntegration:
    """
    Integration layer for optimized HMM clustering with existing pipeline.
    """

    def __init__(self, config: Optional[OptimalClusteringConfig] = None):
        """Initialize optimized clustering integration."""
        self.config = config or OptimalClusteringConfig()
        self.clusterer = create_optimized_clusterer(self.config)
        self.vectorized_ops = VectorizedClusteringOperations()
        self.performance_metrics = {}

    def process_hmm_regime_data(self, hmm_data: pd.DataFrame,
                               feature_columns: Optional[list] = None) -> OptimizedClusteringResult:
        """
        Process HMM regime data with optimized clustering.

        Args:
            hmm_data: DataFrame containing HMM regime data
            feature_columns: List of feature columns to use

        Returns:
            Optimized clustering result
        """
        try:
            logger.info("🚀 Processing HMM regime data with optimized clustering...")

            # Prepare features
            if feature_columns is None:
                feature_columns = [col for col in hmm_data.columns
                                 if col not in ['timestamp', 'regime', 'probability']]

            features_df = hmm_data[feature_columns]

            # Run optimized clustering
            result = self.clusterer.cluster(features_df)

            # Log performance metrics
            if result.success:
                logger.info("✅ Optimized clustering completed successfully")
                logger.info(f"   Clusters: {len(np.unique(result.labels))}")
                logger.info(f"   Silhouette: {result.quality_metrics.get('silhouette', 0.0):.3f}")
                logger.info(f"   Performance: {result.performance_metrics}")
            else:
                logger.error(f"❌ Clustering failed: {result.error_message}")

            return result

        except Exception as e:
            logger.error(f"Error processing HMM regime data: {e}")
            raise

    def compare_with_original(self, hmm_data: pd.DataFrame,
                            feature_columns: Optional[list] = None) -> Dict[str, Any]:
        """
        Compare optimized clustering with original implementation.

        Args:
            hmm_data: DataFrame containing HMM regime data
            feature_columns: List of feature columns to use

        Returns:
            Comparison results
        """
        try:
            logger.info("🔍 Comparing optimized vs original clustering...")

            # Prepare features
            if feature_columns is None:
                feature_columns = [col for col in hmm_data.columns
                                 if col not in ['timestamp', 'regime', 'probability']]

            features_df = hmm_data[feature_columns]

            # Run optimized clustering
            start_time = pd.Timestamp.now()
            optimized_result = self.clusterer.cluster(features_df)
            optimized_time = (pd.Timestamp.now() - start_time).total_seconds()

            # Run original clustering for comparison
            try:
                from .clustering import OptimalRegimeClusterer
                original_clusterer = OptimalRegimeClusterer(self.config)
                start_time = pd.Timestamp.now()
                original_result = original_clusterer.cluster(features_df)
                original_time = (pd.Timestamp.now() - start_time).total_seconds()
            except Exception as e:
                logger.warning(f"Original clustering failed: {e}")
                original_result = None
                original_time = float('inf')

            # Compare results
            comparison = {
                'optimized': {
                    'time': optimized_time,
                    'success': optimized_result.success,
                    'n_clusters': len(np.unique(optimized_result.labels)) if optimized_result.success else 0,
                    'silhouette': optimized_result.quality_metrics.get('silhouette', 0.0),
                    'performance_metrics': optimized_result.performance_metrics
                },
                'original': {
                    'time': original_time,
                    'success': original_result.success if original_result else False,
                    'n_clusters': len(np.unique(original_result.labels)) if original_result and original_result.success else 0,
                    'silhouette': original_result.quality_metrics.get('silhouette', 0.0) if original_result else 0.0
                }
            }

            # Calculate speedup
            if original_time != float('inf') and optimized_time > 0:
                speedup = original_time / optimized_time
                comparison['speedup'] = speedup
                logger.info(f"🚀 Speedup: {speedup:.2f}x")

            return comparison

        except Exception as e:
            logger.error(f"Error comparing clustering implementations: {e}")
            raise

    def benchmark_performance(self, dataset_sizes: list = None) -> Dict[str, Any]:
        """
        Benchmark performance across different dataset sizes.

        Args:
            dataset_sizes: List of dataset sizes to test

        Returns:
            Benchmark results
        """
        if dataset_sizes is None:
            dataset_sizes = [500, 1000, 2000, 5000]

        logger.info(f"🔍 Benchmarking performance for sizes: {dataset_sizes}")

        benchmark = ClusteringPerformanceBenchmark(self.config)
        results = benchmark.run_comprehensive_benchmark(dataset_sizes)

        return results

    def optimize_for_dataset(self, hmm_data: pd.DataFrame) -> OptimalClusteringConfig:
        """
        Optimize configuration for specific dataset characteristics.

        Args:
            hmm_data: Dataset to optimize for

        Returns:
            Optimized configuration
        """
        try:
            logger.info("🔧 Optimizing configuration for dataset...")

            n_samples = len(hmm_data)
            n_features = len([col for col in hmm_data.columns
                            if col not in ['timestamp', 'regime', 'probability']])

            # Create optimized configuration
            optimized_config = OptimalClusteringConfig()

            # Adjust parameters based on dataset size
            if n_samples < 1000:
                optimized_config.chunk_size = 500
                optimized_config.kmeans_num_seeds = 5
            elif n_samples < 5000:
                optimized_config.chunk_size = 1000
                optimized_config.kmeans_num_seeds = 10
            else:
                optimized_config.chunk_size = 2000
                optimized_config.kmeans_num_seeds = 15

            # Adjust for feature count
            if n_features > 20:
                optimized_config.clustering_method = "hybrid"  # Better for high dimensions
            else:
                optimized_config.clustering_method = "kmeans"

            # Enable memory optimization for large datasets
            if n_samples > 5000:
                optimized_config.use_memory_optimization = True
                optimized_config.multi_stage_clustering = True

            logger.info(f"✅ Configuration optimized for {n_samples} samples, {n_features} features")

            return optimized_config

        except Exception as e:
            logger.error(f"Error optimizing configuration: {e}")
            return self.config

    def create_clustering_report(self, result: OptimizedClusteringResult,
                                output_path: str = "clustering_report.html") -> str:
        """
        Create comprehensive clustering report.

        Args:
            result: Clustering result
            output_path: Path to save report

        Returns:
            Report content
        """
        try:
            logger.info("📊 Creating clustering report...")

            # Generate HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>HMM Clustering Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; }}
                    .metric {{ margin: 10px 0; }}
                    .success {{ color: green; }}
                    .error {{ color: red; }}
                    .performance {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>HMM Clustering Report</h1>
                    <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <div class="section">
                    <h2>Clustering Results</h2>
                    <div class="metric">
                        <strong>Status:</strong>
                        <span class="{'success' if result.success else 'error'}">
                            {'Success' if result.success else 'Failed'}
                        </span>
                    </div>
                    <div class="metric">
                        <strong>Number of Clusters:</strong> {len(np.unique(result.labels)) if result.success else 'N/A'}
                    </div>
                    <div class="metric">
                        <strong>Silhouette Score:</strong> {result.quality_metrics.get('silhouette', 0.0):.3f}
                    </div>
                    <div class="metric">
                        <strong>Calinski-Harabasz Score:</strong> {result.quality_metrics.get('calinski_harabasz', 0.0):.1f}
                    </div>
                    <div class="metric">
                        <strong>Davies-Bouldin Score:</strong> {result.quality_metrics.get('davies_bouldin', 0.0):.3f}
                    </div>
                </div>

                <div class="section">
                    <h2>Performance Metrics</h2>
                    <div class="performance">
                        {self._format_performance_metrics(result.performance_metrics)}
                    </div>
                </div>

                <div class="section">
                    <h2>Hardware Optimizations</h2>
                    <div class="metric">
                        <strong>Memory Optimizer:</strong> {'Available' if result.metadata.get('hardware_optimizations', {}).get('memory_optimizer_available') else 'Not Available'}
                    </div>
                    <div class="metric">
                        <strong>GPU Manager:</strong> {'Available' if result.metadata.get('hardware_optimizations', {}).get('gpu_manager_available') else 'Not Available'}
                    </div>
                    <div class="metric">
                        <strong>Matrix Operations:</strong> {'Available' if result.metadata.get('hardware_optimizations', {}).get('matrix_ops_available') else 'Not Available'}
                    </div>
                </div>

                {f'<div class="section"><h2>Error Details</h2><p class="error">{result.error_message}</p></div>' if not result.success else ''}
            </body>
            </html>
            """

            # Save report
            with open(output_path, 'w') as f:
                f.write(html_content)

            logger.info(f"📊 Report saved to {output_path}")
            return html_content

        except Exception as e:
            logger.error(f"Error creating report: {e}")
            return ""

    def _format_performance_metrics(self, metrics: Dict[str, float]) -> str:
        """Format performance metrics for HTML display."""
        if not metrics:
            return "No performance metrics available"

        html = "<ul>"
        for operation, time_taken in metrics.items():
            html += f"<li><strong>{operation}:</strong> {time_taken:.4f}s</li>"
        html += "</ul>"

        return html

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'clusterer'):
            self.clusterer.cleanup()
        if hasattr(self, 'vectorized_ops'):
            self.vectorized_ops.cleanup()

# Convenience functions
def process_hmm_data_optimized(hmm_data: pd.DataFrame,
                              config: Optional[OptimalClusteringConfig] = None,
                              feature_columns: Optional[list] = None) -> OptimizedClusteringResult:
    """Process HMM data with optimized clustering."""
    integration = OptimizedHMMClusteringIntegration(config)
    try:
        result = integration.process_hmm_regime_data(hmm_data, feature_columns)
        return result
    finally:
        integration.cleanup()

def benchmark_clustering_performance(dataset_sizes: list = None) -> Dict[str, Any]:
    """Benchmark clustering performance."""
    integration = OptimizedHMMClusteringIntegration()
    try:
        return integration.benchmark_performance(dataset_sizes)
    finally:
        integration.cleanup()

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Generate sample data
    n_samples = 1000
    n_features = 10
    sample_data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Process with optimized clustering
    result = process_hmm_data_optimized(sample_data)

    if result.success:
        print(f"✅ Clustering completed successfully!")
        print(f"   Clusters: {len(np.unique(result.labels))}")
        print(f"   Silhouette: {result.quality_metrics.get('silhouette', 0.0):.3f}")
    else:
        print(f"❌ Clustering failed: {result.error_message}")
