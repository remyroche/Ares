"""
Optimal Regime Clustering Orchestrator

This module orchestrates the entire optimal regime clustering pipeline from HMM discovery
to ML-ready cluster outputs with comprehensive validation and reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import warnings
import logging
from datetime import datetime
import time

from .config import OptimalClusteringConfig, get_clustering_config
from .clustering import OptimalRegimeClusterer, ClusteringResult, create_optimal_clusterer
from .utils import (
    create_cluster_summary_report, bootstrap_cluster_stability,
    optimize_cluster_parameters, ClusterStatistics
)

logger = logging.getLogger(__name__)

class OptimalRegimeClusteringOrchestrator:
    """Orchestrates the optimal regime clustering pipeline."""

    def __init__(self, config: Optional[OptimalClusteringConfig] = None):
        """Initialize the orchestrator.

        Args:
            config: Clustering configuration
        """
        self.config = config or OptimalClusteringConfig()
        self.clusterer = create_optimal_clusterer(self.config)
        self.logger = logging.getLogger(__name__)
        self.start_time = None

    def run_clustering_pipeline(self, data_path: str, output_dir: str,
                              symbol: str = "UNKNOWN", exchange: str = "UNKNOWN",
                              timeframe: str = "1h", **kwargs) -> Dict[str, Any]:
        """Run the complete clustering pipeline.

        Args:
            data_path: Path to HMM regime data
            output_dir: Directory for output files
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            **kwargs: Additional parameters

        Returns:
            Pipeline results dictionary
        """
        try:
            self.start_time = time.time()
            self.logger.info("🚀 Starting optimal regime clustering pipeline...")

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Step 1: Load and validate data
            self.logger.info("📊 Step 1: Loading and validating data...")
            regime_data = self._load_and_validate_data(data_path)

            # Step 2: Perform clustering
            self.logger.info("🎯 Step 2: Performing optimal clustering...")
            clustering_result = self.clusterer.cluster(regime_data)

            if not clustering_result.success:
                raise RuntimeError(f"Clustering failed: {clustering_result.error_message}")

            # Step 3: Validate results
            self.logger.info("✅ Step 3: Validating clustering results...")
            validation_passed = self._validate_clustering_results(clustering_result)

            # Step 4: Optimize if needed
            if not validation_passed and self.config.adaptive_clustering:
                self.logger.info("🔧 Step 4: Optimizing clustering parameters...")
                clustering_result = self._optimize_clustering(regime_data, clustering_result)

            # Step 5: Generate comprehensive reports
            self.logger.info("📋 Step 5: Generating comprehensive reports...")
            reports = self._generate_comprehensive_reports(
                clustering_result, symbol, exchange, timeframe
            )

            # Step 6: Save results
            self.logger.info("💾 Step 6: Saving results...")
            saved_files = self._save_results(clustering_result, reports, output_path)

            # Step 7: Create ML-ready datasets
            self.logger.info("🤖 Step 7: Creating ML-ready datasets...")
            ml_datasets = self._create_ml_datasets(clustering_result, regime_data, output_path)

            # Compile final results
            pipeline_results = {
                'success': True,
                'execution_time': time.time() - self.start_time,
                'clustering_result': clustering_result,
                'reports': reports,
                'saved_files': saved_files,
                'ml_datasets': ml_datasets,
                'metadata': {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'pipeline_version': '2.0',
                    'config': self.config.to_dict()
                }
            }

            self.logger.info("🎉 Optimal regime clustering pipeline completed successfully!")
            return pipeline_results

        except Exception as e:
            self.logger.error(f"❌ Error in clustering pipeline: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - self.start_time if self.start_time else 0,
                'metadata': {
                    'symbol': kwargs.get('symbol', 'UNKNOWN'),
                    'exchange': kwargs.get('exchange', 'UNKNOWN'),
                    'timeframe': kwargs.get('timeframe', '1h')
                }
            }

    def _load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate HMM regime data.

        Args:
            data_path: Path to data file

        Returns:
            Validated DataFrame
        """
        try:
            # Try different data sources
            if Path(data_path).exists():
                data = pd.read_parquet(data_path)
            else:
                # Try to find HMM cluster data
                possible_paths = [
                    "historical_data/binance/ethusdt/hmm_clusters/hmm_composite_clusters_binance_ETHUSDT_1h.parquet",
                    "artifacts/hmm_regime_unified_artifacts.json",
                    "data/hmm_clusters/*.parquet"
                ]

                data = None
                for path in possible_paths:
                    try:
                        if path.endswith('.parquet'):
                            data = pd.read_parquet(path)
                            break
                        elif path.endswith('.json'):
                            with open(path, 'r') as f:
                                json_data = json.load(f)
                                # Extract cluster data from JSON
                                if 'regime_statistics' in json_data:
                                    data = pd.DataFrame(json_data['regime_statistics'])
                                    break
                    except Exception:
                        continue

                if data is None:
                    raise FileNotFoundError(f"Could not load data from {data_path} or default locations")

            # Validate data has required features
            required_features = self.config.feature_dimensions
            available_features = []

            for feature in required_features:
                feature_cols = [col for col in data.columns if feature.lower() in col.lower()]
                available_features.extend(feature_cols)

            if not available_features:
                raise ValueError(f"No features found for dimensions: {required_features}")

            self.logger.info(f"✅ Data loaded: {data.shape[0]} rows, {len(available_features)} features")
            return data

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def _validate_clustering_results(self, clustering_result: ClusteringResult) -> bool:
        """Validate clustering results.

        Args:
            clustering_result: Clustering result to validate

        Returns:
            True if validation passes
        """
        try:
            # Check basic requirements
            if clustering_result.statistics.n_clusters == 0:
                self.logger.warning("❌ No clusters found")
                return False

            if clustering_result.statistics.noise_percentage > self.config.max_noise_pct:
                self.logger.warning(f"❌ Noise percentage {clustering_result.statistics.noise_percentage".3f"} exceeds limit")
                return False

            if clustering_result.statistics.coverage_percentage < self.config.target_coverage_pct:
                self.logger.warning(f"❌ Coverage {clustering_result.statistics.coverage_percentage".3f"} below target")
                return False

            # Check cluster size distribution
            cluster_sizes = clustering_result.statistics.cluster_sizes
            if len(cluster_sizes) > 0:
                min_size_pct = np.min(cluster_sizes) / (cluster_sizes.sum() + len(clustering_result.labels) * clustering_result.statistics.noise_percentage)
                max_size_pct = np.max(cluster_sizes) / (cluster_sizes.sum() + len(clustering_result.labels) * clustering_result.statistics.noise_percentage)

                if min_size_pct < self.config.min_cluster_size_pct:
                    self.logger.warning(f"❌ Smallest cluster {min_size_pct".3f"} below minimum")
                    return False

                if max_size_pct > self.config.max_cluster_size_pct:
                    self.logger.warning(f"❌ Largest cluster {max_size_pct".3f"} exceeds maximum")
                    return False

            # Check quality metrics
            quality = clustering_result.quality_metrics
            if quality.get('silhouette', 0.0) < self.config.min_silhouette_score:
                self.logger.warning(f"❌ Silhouette score {quality.get('silhouette', 0.0)".3f"} below threshold")
                return False

            self.logger.info("✅ Clustering validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Error validating clustering results: {e}")
            return False

    def _optimize_clustering(self, data: pd.DataFrame, current_result: ClusteringResult) -> ClusteringResult:
        """Optimize clustering parameters and re-run.

        Args:
            data: Input data
            current_result: Current clustering result

        Returns:
            Optimized clustering result
        """
        try:
            self.logger.info("🔧 Optimizing clustering parameters...")

            # Optimize parameters based on current results
            features, _ = self._prepare_features_for_optimization(data)
            optimized_params = optimize_cluster_parameters(features, self.config.to_dict())

            # Update configuration with optimized parameters
            self.config.min_cluster_size = optimized_params.get('min_cluster_size', self.config.min_cluster_size)
            self.config.min_samples = optimized_params.get('min_samples', self.config.min_samples)
            self.config.cluster_selection_epsilon = optimized_params.get('cluster_selection_epsilon', self.config.cluster_selection_epsilon)

            # Re-run clustering with optimized parameters
            new_result = self.clusterer.cluster(data)

            self.logger.info("✅ Clustering optimization completed")
            return new_result

        except Exception as e:
            self.logger.warning(f"Error optimizing clustering: {e}")
            return current_result

    def _prepare_features_for_optimization(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Prepare features for parameter optimization.

        Args:
            data: Input data

        Returns:
            Tuple of (features, metadata)
        """
        try:
            from .utils import prepare_clustering_features
            return prepare_clustering_features(data, self.config.to_dict())
        except Exception as e:
            self.logger.error(f"Error preparing features for optimization: {e}")
            raise

    def _generate_comprehensive_reports(self, clustering_result: ClusteringResult,
                                     symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Generate comprehensive reports.

        Args:
            clustering_result: Clustering result
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe

        Returns:
            Dictionary of reports
        """
        try:
            # Create summary report
            summary_report = create_cluster_summary_report(
                clustering_result.statistics,
                clustering_result.quality_metrics,
                clustering_result.validation
            )

            # Add metadata
            summary_report.update({
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'execution_time': time.time() - self.start_time if self.start_time else 0,
                'timestamp': datetime.now().isoformat(),
                'config': self.config.to_dict()
            })

            # Calculate bootstrap stability
            if clustering_result.labels.size > 0:
                features = clustering_result.metadata.get('features', np.array([]))
                if features.size > 0:
                    stability_score = bootstrap_cluster_stability(
                        features, clustering_result.labels,
                        n_iterations=self.config.bootstrap_iterations
                    )
                    summary_report['stability_score'] = stability_score

            # Create detailed cluster analysis
            detailed_analysis = self._create_detailed_cluster_analysis(clustering_result)

            reports = {
                'summary': summary_report,
                'detailed_analysis': detailed_analysis,
                'cluster_characteristics': self._analyze_cluster_characteristics(clustering_result)
            }

            self.logger.info("✅ Comprehensive reports generated")
            return reports

        except Exception as e:
            self.logger.error(f"Error generating comprehensive reports: {e}")
            return {'error': str(e)}

    def _create_detailed_cluster_analysis(self, clustering_result: ClusteringResult) -> Dict[str, Any]:
        """Create detailed cluster analysis.

        Args:
            clustering_result: Clustering result

        Returns:
            Detailed analysis dictionary
        """
        try:
            analysis = {
                'cluster_distribution': {
                    'size_distribution': clustering_result.statistics.cluster_sizes.tolist(),
                    'percentage_distribution': clustering_result.statistics.cluster_percentages.tolist(),
                    'size_statistics': {
                        'mean': float(clustering_result.statistics.mean_cluster_size),
                        'std': float(clustering_result.statistics.std_cluster_size),
                        'min': float(clustering_result.statistics.min_cluster_size),
                        'max': float(clustering_result.statistics.max_cluster_size)
                    }
                },
                'quality_metrics_breakdown': clustering_result.quality_metrics,
                'validation_details': {
                    'warnings': clustering_result.validation.warnings,
                    'recommendations': clustering_result.validation.recommendations,
                    'is_valid': clustering_result.validation.is_valid
                },
                'clustering_metadata': clustering_result.metadata
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Error creating detailed cluster analysis: {e}")
            return {'error': str(e)}

    def _analyze_cluster_characteristics(self, clustering_result: ClusteringResult) -> Dict[str, Any]:
        """Analyze characteristics of each cluster.

        Args:
            clustering_result: Clustering result

        Returns:
            Cluster characteristics dictionary
        """
        try:
            characteristics = {}

            unique_labels = np.unique(clustering_result.labels)
            if -1 in unique_labels:
                unique_labels = unique_labels[unique_labels != -1]

            for label in unique_labels:
                mask = clustering_result.labels == label
                characteristics[f'cluster_{label}'] = {
                    'size': int(mask.sum()),
                    'percentage': float(mask.sum() / len(clustering_result.labels)),
                    'center': clustering_result.cluster_centers[label].tolist() if clustering_result.cluster_centers.size > 0 else []
                }

            return characteristics

        except Exception as e:
            self.logger.error(f"Error analyzing cluster characteristics: {e}")
            return {'error': str(e)}

    def _save_results(self, clustering_result: ClusteringResult, reports: Dict[str, Any],
                     output_path: Path) -> List[str]:
        """Save clustering results to files.

        Args:
            clustering_result: Clustering result
            reports: Generated reports
            output_path: Output directory path

        Returns:
            List of saved file paths
        """
        try:
            saved_files = []

            # Save cluster labels
            labels_file = output_path / "optimal_cluster_labels.parquet"
            labels_df = pd.DataFrame({
                'cluster_id': clustering_result.labels,
                'timestamp': pd.date_range(start='2020-01-01', periods=len(clustering_result.labels), freq='H')
            })
            labels_df.to_parquet(labels_file)
            saved_files.append(str(labels_file))

            # Save summary report
            summary_file = output_path / "clustering_summary_report.json"
            with open(summary_file, 'w') as f:
                json.dump(reports['summary'], f, indent=2, default=str)
            saved_files.append(str(summary_file))

            # Save detailed analysis
            detailed_file = output_path / "detailed_cluster_analysis.json"
            with open(detailed_file, 'w') as f:
                json.dump(reports['detailed_analysis'], f, indent=2, default=str)
            saved_files.append(str(detailed_file))

            # Save cluster characteristics
            characteristics_file = output_path / "cluster_characteristics.json"
            with open(characteristics_file, 'w') as f:
                json.dump(reports['cluster_characteristics'], f, indent=2, default=str)
            saved_files.append(str(characteristics_file))

            self.logger.info(f"✅ Saved {len(saved_files)} result files")
            return saved_files

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return []

    def _create_ml_datasets(self, clustering_result: ClusteringResult, original_data: pd.DataFrame,
                           output_path: Path) -> Dict[str, str]:
        """Create ML-ready datasets for each cluster.

        Args:
            clustering_result: Clustering result
            original_data: Original regime data
            output_path: Output directory path

        Returns:
            Dictionary of dataset paths
        """
        try:
            datasets = {}

            unique_labels = np.unique(clustering_result.labels)
            if -1 in unique_labels:
                unique_labels = unique_labels[unique_labels != -1]

            for label in unique_labels:
                mask = clustering_result.labels == label

                if mask.sum() > 10:  # Only create datasets for clusters with sufficient data
                    cluster_data = original_data[mask].copy()
                    cluster_data['cluster_id'] = label

                    # Save cluster dataset
                    cluster_file = output_path / f"cluster_{label}_dataset.parquet"
                    cluster_data.to_parquet(cluster_file)
                    datasets[f'cluster_{label}'] = str(cluster_file)

            # Create combined dataset with cluster assignments
            combined_file = output_path / "all_clusters_dataset.parquet"
            all_data = original_data.copy()
            all_data['optimal_cluster_id'] = clustering_result.labels
            all_data.to_parquet(combined_file)
            datasets['combined'] = str(combined_file)

            self.logger.info(f"✅ Created {len(datasets)} ML-ready datasets")
            return datasets

        except Exception as e:
            self.logger.error(f"Error creating ML datasets: {e}")
            return {}

def run_optimal_clustering(data_path: str, output_dir: str, config: Optional[OptimalClusteringConfig] = None,
                          symbol: str = "ETHUSDT", exchange: str = "binance", timeframe: str = "1h",
                          **kwargs) -> Dict[str, Any]:
    """Convenience function to run optimal regime clustering.

    Args:
        data_path: Path to HMM regime data
        output_dir: Output directory
        config: Clustering configuration
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        **kwargs: Additional parameters

    Returns:
        Pipeline results
    """
    orchestrator = OptimalRegimeClusteringOrchestrator(config)
    return orchestrator.run_clustering_pipeline(data_path, output_dir, symbol, exchange, timeframe, **kwargs)

def run_high_quality_clustering(data_path: str, output_dir: str,
                               symbol: str = "ETHUSDT", exchange: str = "binance", timeframe: str = "1h",
                               **kwargs) -> Dict[str, Any]:
    """Run high-quality clustering with enhanced parameters.

    Args:
        data_path: Path to HMM regime data
        output_dir: Output directory
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        **kwargs: Additional parameters

    Returns:
        Pipeline results
    """
    config = get_clustering_config("high_quality")
    orchestrator = OptimalRegimeClusteringOrchestrator(config)
    return orchestrator.run_clustering_pipeline(data_path, output_dir, symbol, exchange, timeframe, **kwargs)

def run_fast_clustering(data_path: str, output_dir: str,
                       symbol: str = "ETHUSDT", exchange: str = "binance", timeframe: str = "1h",
                       **kwargs) -> Dict[str, Any]:
    """Run fast clustering for quick results.

    Args:
        data_path: Path to HMM regime data
        output_dir: Output directory
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        **kwargs: Additional parameters

    Returns:
        Pipeline results
    """
    config = get_clustering_config("fast_processing")
    orchestrator = OptimalRegimeClusteringOrchestrator(config)
    return orchestrator.run_clustering_pipeline(data_path, output_dir, symbol, exchange, timeframe, **kwargs)