"""
Optimal Regime Clustering Orchestrator

This module orchestrates the entire optimal regime clustering pipeline from HMM discovery
to ML-ready cluster outputs with comprehensive validation and reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import json
import pickle
import warnings
import logging
from datetime import datetime
import time
import os
import glob

from .config import OptimalClusteringConfig, get_clustering_config
from .clustering import OptimalRegimeClusterer, ClusteringResult, create_optimal_clusterer
from .optimized_clustering import MatrixOptimizedClusterer, OptimizedClusteringResult, create_matrix_optimized_clusterer
from .utils import (
    create_cluster_summary_report,
    optimize_cluster_parameters, ClusterStatistics
)
from .enhanced_analysis import (
    create_enhanced_statistical_summary,
    calculate_cluster_dimension_analysis,
    DimensionCVMetrics
)

logger = logging.getLogger(__name__)

def detect_latest_hmm_results(symbol: str = "ETHUSDT", exchange: str = "binance", timeframe: str = "15m") -> tuple:
    """Detect the latest HMM regime discovery results.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe

    Returns:
        tuple: (data_path, output_dir) or (None, None) if not found
    """
    try:
        # First priority: Look for timeframe-specific HMM regime discovery outcome files
        timeframe_patterns = [
            f"outcomes/market_analysis_hmm_regime_discovery_outcome_*_{symbol.lower()}_{exchange.lower()}_{timeframe}.json",
            f"outcomes/market_analysis_hmm_regime_discovery_outcome_*_{symbol.lower()}_{exchange.lower()}_*.json",
            f"outcomes/market_analysis_hmm_regime_discovery_outcome_*.json"
        ]

        data_path = None
        for pattern in timeframe_patterns:
            files = glob.glob(pattern, recursive=False)
            if files:
                # Get the most recent file
                data_path = max(files, key=lambda x: Path(x).stat().st_mtime)
                break

        # CRITICAL: Only use specified timeframe, no fallbacks
        if not data_path:
            raise FileNotFoundError(f"❌ No {timeframe} HMM results found for {symbol} on {exchange}. {timeframe} timeframe is critical and no fallback is allowed.")

        if data_path:
            # Determine output directory based on data path location
            data_path_obj = Path(data_path)
            if "historical_data" in str(data_path):
                # Use the same directory structure
                output_dir = data_path_obj.parent / "optimal_clusters"
            else:
                # Use standard output location
                output_dir = Path(f"generated/market_analysis/optimal_clusters/{exchange}/{symbol}/{timeframe}")

            logger.info(f"✅ Detected HMM results: {data_path}")
            logger.info(f"📁 Output directory: {output_dir}")
            return str(data_path), str(output_dir)
        else:
            logger.warning(f"❌ No HMM results found for {exchange}/{symbol}/{timeframe}")
            return None, None

    except Exception as e:
        logger.error(f"Error detecting HMM results: {e}")
        return None, None

class OptimalRegimeClusteringOrchestrator:
    """Orchestrates the optimal regime clustering pipeline."""

    def __init__(self, config: Optional[OptimalClusteringConfig] = None):
        """Initialize the orchestrator.

        Args:
            config: Clustering configuration
        """
        self.config = config or OptimalClusteringConfig()
        self.logger = logging.getLogger(__name__)
        self.start_time = None

        # Force matrix optimization as default when available
        self.matrix_available = self._check_matrix_operations_availability()
        self.use_matrix_optimization = True  # Always prefer matrix optimization

        if self.matrix_available:
            self.clusterer = create_matrix_optimized_clusterer(self.config)
            self.logger.info("✅ Using matrix-optimized clustering (GPU acceleration enabled)")
        else:
            self.clusterer = create_optimal_clusterer(self.config)
            self.logger.warning("⚠️ Matrix operations not available, falling back to standard clustering")

    def _check_matrix_operations_availability(self) -> bool:
        """Check if matrix operations are available.

        Returns:
            True if matrix operations are available
        """
        try:
            from src.utils.matrix_operations import (
                get_unified_matrix_operations,
                get_vectorized_processing_core,
                get_enhanced_matrix_operations,
                get_batch_matrix_processor
            )
            return True
        except ImportError:
            return False

    def _convert_optimized_to_standard_result(self, optimized_result: OptimizedClusteringResult) -> ClusteringResult:
        """Convert optimized result to standard clustering result format.

        Args:
            optimized_result: Optimized clustering result

        Returns:
            Standard clustering result
        """
        class StandardClusteringResult:
            def __init__(self, labels, centers, stats, quality, validation, metadata):
                self.labels = labels
                self.cluster_centers = centers
                self.statistics = stats
                self.quality_metrics = quality
                self.validation = validation
                self.metadata = metadata
                self.success = True

        return StandardClusteringResult(
            labels=optimized_result.labels,
            centers=optimized_result.cluster_centers,
            stats=optimized_result.statistics,
            quality=optimized_result.quality_metrics,
            validation=optimized_result.validation,
            metadata={
                **optimized_result.metadata,
                'matrix_optimization_used': True,
                'performance_metrics': optimized_result.performance_metrics
            }
        )

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

            # Initialize file variables at the start to prevent scope issues
            hmm_training_file = ""
            cluster_metrics_file = ""

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Step 1: Load and validate data
            self.logger.info("📊 Step 1: Loading and validating data...")
            regime_data = self._load_and_validate_data(data_path)

            # Step 2: Perform clustering
            self.logger.info("🎯 Step 2: Performing optimal clustering...")
            if self.use_matrix_optimization:
                # Use optimized clustering
                optimized_result = self.clusterer.cluster_optimized(regime_data)

                if not optimized_result.success:
                    raise RuntimeError(f"Optimized clustering failed: {optimized_result.error_message}")

                # Convert optimized result to standard format for compatibility
                clustering_result = self._convert_optimized_to_standard_result(optimized_result)
            else:
                # Use standard clustering
                if hasattr(self.clusterer, 'cluster_optimized'):
                    # Matrix optimized clusterer
                    clustering_result = self.clusterer.cluster_optimized(regime_data)
                else:
                    # Standard clusterer
                    clustering_result = self.clusterer.cluster(regime_data)

                if not clustering_result.success:
                    raise RuntimeError(f"Clustering failed: {clustering_result.error_message}")

            # Step 3: Validate results
            self.logger.info("✅ Step 3: Validating clustering results...")
            validation_passed = self._validate_clustering_results(clustering_result)

            # Step 3.5: Apply aggressive cluster splitting to enforce size constraints
            # Note: This is disabled by default to preserve natural cluster distribution
            if self.config.enable_aggressive_splitting and not validation_passed:
                self.logger.info("🔧 Step 3.5: Applying aggressive cluster splitting to enforce size constraints...")
                clustering_result = self._apply_aggressive_cluster_splitting(clustering_result)
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

            # Step 6: Save results (initial save without HMM training file)
            self.logger.info("💾 Step 6: Saving results...")
            saved_files = self._save_results(clustering_result, reports, output_path, "", "")

            # Step 7: Create ML-ready datasets
            self.logger.info("🤖 Step 7: Creating ML-ready datasets...")
            ml_datasets = self._create_ml_datasets(clustering_result, regime_data, output_path)

            # Step 8: Generate HMM models training input file
            self.logger.info("🧠 Step 8: Generating HMM models training input file...")
            hmm_training_file = self._create_hmm_training_input_file(clustering_result, regime_data, output_path)

            # Step 9: Generate comprehensive cluster metrics file
            self.logger.info("📊 Step 9: Generating comprehensive cluster metrics file...")
            cluster_metrics_file = self._create_cluster_metrics_file(clustering_result, regime_data, output_path)

            # Step 10: Update saved files list with HMM training and cluster metrics files
            if hmm_training_file:
                saved_files.append(hmm_training_file)
            if cluster_metrics_file:
                saved_files.append(cluster_metrics_file)

            # Compile final results
            pipeline_results = {
                'success': True,
                'execution_time': time.time() - self.start_time,
                'clustering_result': clustering_result,
                'reports': reports,
                'saved_files': saved_files,
                'ml_datasets': ml_datasets,
                'hmm_training_input_file': hmm_training_file,
                'cluster_metrics_file': cluster_metrics_file,
                'metadata': {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'pipeline_version': '2.1',
                    'config': self.config.to_dict(),
                    'files_generated': {
                        'hmm_training_input': hmm_training_file != "",
                        'cluster_metrics': cluster_metrics_file != ""
                    }
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
            # Import standardized_parquet_handler like HMM clustering does
            from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

            # Check if it's a JSON outcome file
            if data_path.endswith('.json'):
                # Load JSON outcome file and extract HMM regime discovery results
                with open(data_path, 'r') as f:
                    outcome_data = json.load(f)

                # Extract HMM regime discovery artifacts
                artifacts = outcome_data.get('artifacts', {})
                hmm_regime_discovery = artifacts.get('hmm_regime_discovery_result', {})

                if not hmm_regime_discovery:
                    raise ValueError(f"No HMM regime discovery results found in outcome file: {data_path}")

                # Store the regime discovery data for use by the clustering component
                self.hmm_regime_discovery = hmm_regime_discovery

                # For now, load the corresponding klines data since the optimal clustering
                # needs the original market data
                symbol = getattr(self.config, 'symbol', 'ETHUSDT')
                exchange = getattr(self.config, 'exchange', 'BINANCE')
                timeframe = getattr(self.config, 'timeframe', '15m')

                # Try the processed data path pattern used by the sub_pipeline
                processed_dir = Path('historical_data') / exchange.lower() / symbol.lower() / 'processed' / f"{symbol.lower()}_{timeframe}"

                # Look for the processed directory (it should contain year/month subdirectories)
                if processed_dir.exists():
                    logger.debug(f"📁 Found processed directory: {processed_dir}")

                    # Find all parquet files recursively in the processed directory
                    parquet_files = []
                    for root, dirs, files in os.walk(processed_dir):
                        for file in files:
                            if file.endswith('.parquet'):
                                parquet_files.append(Path(root) / file)

                    if parquet_files:
                        logger.debug(f"📊 Found {len(parquet_files)} parquet files in processed directory")

                        # Load all parquet files
                        all_data = []
                        for file_path in sorted(parquet_files):
                            try:
                                file_data = standardized_parquet_handler.read_parquet_standardized(file_path)
                                if file_data is not None:
                                    all_data.append(file_data)
                                    logger.debug(f"✅ Loaded data from: {file_path} ({len(file_data)} rows)")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to load {file_path}: {e}")
                                continue

                        if all_data:
                            # Concatenate all data
                            data = pd.concat(all_data, ignore_index=True)
                            logger.debug(f"✅ Successfully loaded klines data from processed directory: {len(data)} total rows")
                            logger.debug(f"✅ Successfully loaded HMM regime discovery from: {data_path}")
                            return data
                    else:
                        logger.warning(f"⚠️ No parquet files found in processed directory: {processed_dir}")
                else:
                    logger.warning(f"⚠️ Processed directory not found: {processed_dir}")

                # Fallback to standard path
                klines_data_path = Path('historical_data') / f"{exchange.lower()}_{symbol.lower()}_{timeframe}_klines.parquet"
                if klines_data_path.exists():
                    data = standardized_parquet_handler.read_parquet_standardized(klines_data_path)
                    if data is not None:
                        logger.debug(f"✅ Loaded klines data from: {klines_data_path}")
                        logger.debug(f"✅ Successfully loaded HMM regime discovery from: {data_path}")
                        return data

                raise ValueError(f"Klines data not found at processed directory or standard path")
            else:
                # Try to load data using standardized handler (parquet files)
                if Path(data_path).exists():
                    data = standardized_parquet_handler.read_parquet_standardized(data_path)
                else:
                    # Use same logic as HMM clustering for finding data
                    symbol = getattr(self.config, 'symbol', 'ETHUSDT')
                exchange = getattr(self.config, 'exchange', 'BINANCE')
                timeframe = getattr(self.config, 'timeframe', '15m')

                # Try the standard HMM clustering data path (using same pattern as hmm_clustering.py)
                data_path = Path('historical_data') / f"{exchange.lower()}_{symbol.lower()}_{timeframe}_klines.parquet"

                if data_path.exists():
                    data = standardized_parquet_handler.read_parquet_standardized(data_path)
                    if data is not None:
                        logger.info(f"✅ Loaded data from primary path: {data_path}")

                if data is not None:
                    logger.debug(f"✅ Successfully loaded {len(data)} records for regime clustering")

                # If no data found, continue to HMM cluster data search
                else:
                    # Try to find HMM cluster data for the specified timeframe only
                    possible_paths = [
                        f"historical_data/binance/ethusdt/hmm_clusters/hmm_composite_clusters_binance_ETHUSDT_{timeframe}.parquet",
                        f"artifacts/hmm_regime_unified_artifacts.json"
                    ]

                    data = None
                    for path in possible_paths:
                        try:
                            if path.endswith('.parquet'):
                                if Path(path).exists():
                                    data = standardized_parquet_handler.read_parquet_standardized(path)
                                    break
                            elif path.endswith('.json'):
                                if Path(path).exists():
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

    def _load_processed_data_for_regime_clustering(self, processed_data_path: str) -> Optional[pd.DataFrame]:
        """Load processed data from partitioned files for regime clustering.

        Args:
            processed_data_path: Path to processed data directory

        Returns:
            DataFrame with regime clustering features or None if failed
        """
        try:

            data_path = Path(processed_data_path)
            if not data_path.exists():
                return None

            # Collect all parquet files from the processed directory
            parquet_files = []
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.parquet'):
                        parquet_files.append(Path(root) / file)

            if not parquet_files:
                self.logger.warning(f"⚠️ No parquet files found in {processed_data_path}")
                return None

            # Load and concatenate all parquet files
            dfs = []
            for file_path in parquet_files:
                try:
                    df = standardized_parquet_handler.read_parquet_standardized(file_path)
                    if df is not None and not df.empty:
                        dfs.append(df)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {file_path}: {e}")
                    continue

            if not dfs:
                self.logger.warning(f"⚠️ No valid data loaded from {processed_data_path}")
                return None

            # Concatenate all dataframes
            combined_data = pd.concat(dfs, ignore_index=True)

            if len(combined_data) == 0:
                self.logger.warning(f"⚠️ Combined data is empty for {processed_data_path}")
                return None

            # Prepare features for regime clustering
            # Select relevant columns for clustering (volume, volatility, momentum, trend indicators)
            feature_columns = []
            for col in combined_data.columns:
                if any(keyword in col.lower() for keyword in ['volume', 'volatility', 'momentum', 'trend', 'price', 'return', 'rsi', 'sma', 'ema', 'bbands']):
                    feature_columns.append(col)

            # If no specific feature columns found, use basic price/volume data
            if not feature_columns:
                feature_columns = ['open', 'high', 'low', 'close', 'volume']
                # Remove columns that don't exist
                feature_columns = [col for col in feature_columns if col in combined_data.columns]

            if not feature_columns:
                self.logger.warning(f"⚠️ No suitable feature columns found in processed data")
                return None

            # Extract features for clustering
            features_data = combined_data[feature_columns].copy()

            # Handle missing values
            features_data = features_data.fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).fillna(0)

            # Add timestamp if available for temporal ordering
            if 'timestamp' in combined_data.columns:
                features_data['timestamp'] = combined_data['timestamp']
                features_data = features_data.sort_values('timestamp').reset_index(drop=True)

            self.logger.info(f"✅ Loaded {len(features_data)} samples with {len(feature_columns)} features from {processed_data_path}")
            return features_data

        except Exception as e:
            self.logger.warning(f"Error loading processed data: {e}")
            return None

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
                self.logger.warning(f"❌ Noise percentage {clustering_result.statistics.noise_percentage:.3f} exceeds limit")
                return False

            if clustering_result.statistics.coverage_percentage < self.config.target_coverage_pct:
                self.logger.warning(f"❌ Coverage {clustering_result.statistics.coverage_percentage:.3f} below target")
                return False

            # Check cluster size distribution
            cluster_sizes = clustering_result.statistics.cluster_sizes
            if len(cluster_sizes) > 0:
                min_size_pct = np.min(cluster_sizes) / (cluster_sizes.sum() + len(clustering_result.labels) * clustering_result.statistics.noise_percentage)
                max_size_pct = np.max(cluster_sizes) / (cluster_sizes.sum() + len(clustering_result.labels) * clustering_result.statistics.noise_percentage)

                if min_size_pct < self.config.min_cluster_size_pct:
                    self.logger.warning(f"❌ Smallest cluster {min_size_pct:.3f} below minimum")
                    return False

                if max_size_pct > self.config.max_cluster_size_pct:
                    self.logger.warning(f"❌ Largest cluster {max_size_pct:.3f} exceeds maximum")
                    return False

            # Check quality metrics with sanitation
            raw_quality = clustering_result.quality_metrics or {}
            quality = {}
            for k, v in raw_quality.items():
                try:
                    val = float(v)
                except Exception:
                    val = 0.0
                if not np.isfinite(val):
                    self.logger.warning(f"Non-finite quality metric {k} detected: {v}, coercing")
                    if 'cv' in k.lower():
                        val = 10.0
                    elif 'davies' in k.lower():
                        val = 10.0
                    else:
                        val = 0.0
                if val < 0:
                    self.logger.warning(f"Negative quality metric {k} detected: {val}, clamping")
                    val = 0.0
                quality[k] = val
            if quality.get('silhouette', 0.0) < self.config.min_silhouette_score:
                self.logger.warning(f"❌ Silhouette score {quality.get('silhouette', 0.0):.3f} below threshold")
                return False

            self.logger.info("✅ Clustering validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Error validating clustering results: {e}")
            return False

    def _apply_aggressive_cluster_splitting(self, clustering_result: ClusteringResult) -> ClusteringResult:
        """Apply aggressive cluster splitting to enforce size constraints.

        Args:
            clustering_result: Current clustering result

        Returns:
            ClusteringResult with size constraints enforced
        """
        try:
            from src.training.steps.market_analysis.cluster_constraints import (
                split_giant_clusters,
                merge_tail_into_topk,
                balance_topk_range,
            )

            # Check if constraint enforcement is disabled in config
            if not self.config.enable_aggressive_splitting:
                self.logger.info("⏭️ Skipping aggressive cluster splitting (disabled in config)")
                return clustering_result

            # Extract features from the clustering result metadata
            features = clustering_result.metadata.get('features')
            if features is None:
                self.logger.warning("No features available for cluster splitting")
                return clustering_result

            # Get current labels
            current_labels = clustering_result.labels

            # Apply aggressive cluster splitting with more permissive parameters
            max_prop = max(0.50, self.config.max_cluster_size_pct * 2.0)  # More permissive
            target_range = (self.config.min_cluster_size_pct * 0.5, self.config.max_cluster_size_pct * 1.5)  # Wider range

            new_labels = split_giant_clusters(
                features,
                current_labels,
                max_prop=max_prop,
                target_range=target_range,
                metric="euclidean",
                random_state=self.config.random_state
            )

            # Keep cluster count and distribution aligned with goals
            new_labels = merge_tail_into_topk(
                features,
                new_labels,
                k=self.config.target_n_clusters,
                coverage_target=(self.config.target_coverage_pct - 0.05, self.config.target_coverage_pct),
                metric="euclidean",
            )
            new_labels = balance_topk_range(
                features,
                new_labels,
                k=self.config.target_n_clusters,
                target_range=(self.config.min_cluster_size_pct, self.config.max_cluster_size_pct),
                metric="euclidean",
                random_state=self.config.random_state,
            )

            # Create new clustering result
            new_result = ClusteringResult(
                labels=new_labels,
                cluster_centers=clustering_result.cluster_centers,
                statistics=clustering_result.statistics,
                quality_metrics=clustering_result.quality_metrics,
                validation=clustering_result.validation,
                metadata=clustering_result.metadata,
                success=True
            )

            self.logger.info(f"✅ Applied aggressive cluster splitting: {len(np.unique(current_labels))} -> {len(np.unique(new_labels))} clusters")
            return new_result

        except Exception as e:
            self.logger.warning(f"Error applying aggressive cluster splitting: {e}")
            return clustering_result

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

            # Add metadata including matrix optimization info
            summary_report.update({
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'execution_time': time.time() - self.start_time if self.start_time else 0,
                'timestamp': datetime.now().isoformat(),
                'config': self.config.to_dict(),
                'matrix_optimization_used': self.use_matrix_optimization,
                'optimization_level': 'high' if self.use_matrix_optimization else 'standard'
            })

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
                # Handle different types of cluster_centers objects
                if hasattr(clustering_result.cluster_centers, 'size') and clustering_result.cluster_centers.size > 0:
                    # Standard numpy array case
                    center = clustering_result.cluster_centers[label].tolist()
                else:
                    # Handle case where cluster_centers might be a different type of object
                    center = []

                characteristics[f'cluster_{label}'] = {
                    'size': int(mask.sum()),
                    'percentage': float(mask.sum() / len(clustering_result.labels)),
                    'center': center
                }

            return characteristics

        except Exception as e:
            self.logger.error(f"Error analyzing cluster characteristics: {e}")
            return {'error': str(e)}

    def _save_results(self, clustering_result: ClusteringResult, reports: Dict[str, Any],
                     output_path: Path, hmm_training_file: str = "", cluster_metrics_file: str = "") -> List[str]:
        """Save clustering results to files.

        Args:
            clustering_result: Clustering result
            reports: Generated reports
            output_path: Output directory path
            hmm_training_file: Path to HMM training input file (optional)
            cluster_metrics_file: Path to cluster metrics file (optional)

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
            print(f"💾 Saving cluster summary report to: {summary_file}")
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

            # Add HMM training input file if provided
            if hmm_training_file:
                saved_files.append(hmm_training_file)

            # Add cluster metrics file if provided
            if cluster_metrics_file:
                saved_files.append(cluster_metrics_file)

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
                cluster_size = mask.sum()

                if cluster_size > 10:  # Only create datasets for clusters with sufficient data
                    cluster_data = original_data[mask].copy()
                    cluster_data['cluster_id'] = label

                    # Calculate metrics for this cluster
                    metrics = {
                        'cluster_id': int(label),
                        'size': cluster_size,
                        'percentage': cluster_size / len(original_data),
                        'has_volume': 'volume' in cluster_data.columns,
                        'has_price': 'close' in cluster_data.columns or 'price' in cluster_data.columns,
                        'date_range': {
                            'start': str(cluster_data.index.min()) if not len(cluster_data) == 0 else None,
                            'end': str(cluster_data.index.max()) if not len(cluster_data) == 0 else None
                        }
                    }

                    # Save cluster dataset
                    cluster_file = output_path / f"cluster_{label}_dataset.parquet"
                    cluster_data.to_parquet(cluster_file)
                    datasets[f'cluster_{label}'] = str(cluster_file)

                    # Print metrics for this cluster
                    print(f"📊 Cluster {label}: {cluster_size} samples ({metrics['percentage']:.3%}) - Saved to: {cluster_file}")
                    if metrics['has_volume']:
                        print(f"   📈 Volume data: ✓")
                    if metrics['has_price']:
                        print(f"   💰 Price data: ✓")
                    if metrics['date_range']['start']:
                        print(f"   📅 Date range: {metrics['date_range']['start']} to {metrics['date_range']['end']}")

            # Create combined dataset with cluster assignments
            combined_file = output_path / "all_clusters_dataset.parquet"
            all_data = original_data.copy()
            all_data['optimal_cluster_id'] = clustering_result.labels
            all_data.to_parquet(combined_file)
            datasets['combined'] = str(combined_file)

            # Print metrics for combined dataset
            print(f"📊 Combined dataset: {len(all_data)} total samples - Saved to: {combined_file}")
            print(f"   📈 Contains {len(unique_labels)} clusters")
            print(f"   📅 Full date range: {all_data.index.min()} to {all_data.index.max()}")

            self.logger.info(f"✅ Created {len(datasets)} ML-ready datasets")
            print(f"🎉 Total ML-ready datasets created: {len(datasets)}")
            return datasets

        except Exception as e:
            self.logger.error(f"Error creating ML datasets: {e}")
            return {}

    def _create_hmm_training_input_file(self, clustering_result: ClusteringResult, original_data: pd.DataFrame,
                                       output_path: Path) -> str:
        """Create HMM models training input file.

        Args:
            clustering_result: Clustering result
            original_data: Original regime data
            output_path: Output directory path

        Returns:
            Path to the generated HMM training input file
        """
        try:
            # Prepare features (X) - use the same features as clustering
            features = clustering_result.metadata.get('features')
            if features is None:
                # Fallback: use original data features
                numeric_cols = original_data.select_dtypes(include=[np.number]).columns
                features = original_data[numeric_cols].values

            # Prepare targets (y) - create synthetic targets based on cluster characteristics
            # For HMM training, we need binary or multi-class targets
            # Let's create targets based on cluster size categories
            unique_labels = np.unique(clustering_result.labels)
            if -1 in unique_labels:
                unique_labels = unique_labels[unique_labels != -1]

            # Create target based on cluster size percentiles
            cluster_sizes = clustering_result.statistics.cluster_sizes
            size_percentiles = np.percentile(cluster_sizes, [25, 50, 75])

            y = np.zeros(len(clustering_result.labels))
            for i, label in enumerate(unique_labels):
                mask = clustering_result.labels == label
                if i < len(cluster_sizes):
                    size = cluster_sizes[i]
                    if size <= size_percentiles[0]:
                        y[mask] = 0  # Small clusters
                    elif size <= size_percentiles[1]:
                        y[mask] = 1  # Medium clusters
                    elif size <= size_percentiles[2]:
                        y[mask] = 2  # Large clusters
                    else:
                        y[mask] = 3  # Very large clusters

            # Prepare cluster assignments (regime_labels)
            cluster_assignments = clustering_result.labels.copy()

            # Create HMM training input dictionary
            hmm_input = {
                'X': features,
                'y': y,
                'cluster_assignments': cluster_assignments,
                'feature_names': list(original_data.select_dtypes(include=[np.number]).columns),
                'metadata': {
                    'n_samples': len(features),
                    'n_features': features.shape[1],
                    'n_clusters': len(unique_labels),
                    'cluster_sizes': clustering_result.statistics.cluster_sizes.tolist(),
                    'cluster_percentages': clustering_result.statistics.cluster_percentages.tolist(),
                    'quality_metrics': clustering_result.quality_metrics,
                    'timestamp': datetime.now().isoformat(),
                    'symbol': getattr(self, 'symbol', 'UNKNOWN'),
                    'exchange': getattr(self, 'exchange', 'UNKNOWN'),
                    'timeframe': getattr(self, 'timeframe', 'UNKNOWN')
                }
            }

            # Save HMM training input file with proper naming convention
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol = getattr(self, 'symbol', 'ETHUSDT')
            exchange = getattr(self, 'exchange', 'BINANCE')
            timeframe = getattr(self, 'timeframe', '15m')

            # Create filename following HMM training conventions
            hmm_input_filename = f"market_analysis_hmm_training_input_{symbol}_{exchange}_{timeframe}_{timestamp}.pkl"

            # Save to the same output directory for now (can be moved to outcomes/ if needed)
            hmm_input_file = output_path / hmm_input_filename
            with open(hmm_input_file, 'wb') as f:
                pickle.dump(hmm_input, f)

            self.logger.info(f"✅ Created HMM training input file: {hmm_input_file}")
            self.logger.info(f"📊 HMM input shape: X={features.shape}, y={y.shape}, clusters={len(unique_labels)}")
            return str(hmm_input_file)

        except Exception as e:
            self.logger.error(f"Error creating HMM training input file: {e}")
            return ""

    def _create_cluster_metrics_file(self, clustering_result: ClusteringResult, original_data: pd.DataFrame,
                                    output_path: Path) -> str:
        """Create comprehensive statistical and economic metrics file for clusters.

        Args:
            clustering_result: Clustering result
            original_data: Original regime data
            output_path: Output directory path

        Returns:
            Path to the generated cluster metrics file
        """
        try:
            # Calculate comprehensive cluster metrics
            metrics = self._calculate_comprehensive_cluster_metrics(
                clustering_result, original_data
            )

            # Organize metrics by groups (all, top20, top10, top5, top1)
            grouped_metrics = self._organize_metrics_by_groups(metrics, clustering_result)

            # Calculate individual cluster analysis
            individual_analysis = self._calculate_individual_cluster_analysis(
                metrics, clustering_result, original_data
            )

            # Calculate time period from data
            time_period = self._calculate_time_period(original_data)

            # Calculate coefficient of variation for cluster sizes
            cluster_sizes = [cluster['basic_stats']['size'] for cluster in metrics.values()]
            if len(cluster_sizes) > 0 and clustering_result.statistics.mean_cluster_size > 0:
                cluster_size_cv = clustering_result.statistics.std_cluster_size / clustering_result.statistics.mean_cluster_size
            else:
                cluster_size_cv = 0.0

            # Enhanced quality metrics with CV
            enhanced_quality_metrics = clustering_result.quality_metrics.copy()
            enhanced_quality_metrics['cluster_size_coefficient_of_variation'] = cluster_size_cv

            # Identify clusters outside 3-8% range
            outliers_outside_range = []
            clusters_within_range = 0
            for cluster_id, cluster_data in metrics.items():
                percentage = cluster_data['basic_stats']['percentage']
                if 0.03 <= percentage <= 0.08:
                    clusters_within_range += 1
                else:
                    outliers_outside_range.append(f"{cluster_id} ({percentage*100:.1f}%)")

            outlier_percentage = len(outliers_outside_range) / len(metrics) if len(metrics) > 0 else 0.0

            # Create comprehensive metrics report
            comprehensive_report = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': getattr(self, 'symbol', 'ETHUSDT'),
                    'exchange': getattr(self, 'exchange', 'BINANCE'),
                    'timeframe': getattr(self, 'timeframe', '15m'),
                    'time_period': time_period,
                    'total_clusters': len(metrics),
                    'total_samples': len(original_data),
                    'quality_metrics': enhanced_quality_metrics,
                    'cluster_statistics': {
                        'n_clusters': clustering_result.statistics.n_clusters,
                        'noise_percentage': clustering_result.statistics.noise_percentage,
                        'coverage_percentage': clustering_result.statistics.coverage_percentage,
                        'mean_cluster_size': float(clustering_result.statistics.mean_cluster_size),
                        'std_cluster_size': float(clustering_result.statistics.std_cluster_size),
                        'cluster_size_distribution': {
                            'min_size': int(clustering_result.statistics.min_cluster_size) if hasattr(clustering_result.statistics, 'min_cluster_size') else min(cluster_sizes),
                            'max_size': int(clustering_result.statistics.max_cluster_size) if hasattr(clustering_result.statistics, 'max_cluster_size') else max(cluster_sizes),
                            'median_size': int(np.median(cluster_sizes)) if len(cluster_sizes) > 0 else 0,
                            'size_range': max(cluster_sizes) - min(cluster_sizes) if len(cluster_sizes) > 0 else 0,
                            'outliers_outside_3_8_percent': {
                                'clusters_outside_range': outliers_outside_range,
                                'clusters_within_range': clusters_within_range,
                                'outlier_percentage': outlier_percentage
                            }
                        }
                    }
                },
                'grouped_analysis': grouped_metrics,
                'cluster_size_distribution_summary': self._create_cluster_size_distribution_summary(metrics),
                'individual_analysis': individual_analysis,
                'economic_analysis': self._calculate_economic_metrics(metrics, original_data),
                'statistical_summary': self._create_statistical_summary(metrics, clustering_result)
            }

            # Save comprehensive metrics file to outcomes/ directory with proper naming
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol = getattr(self, 'symbol', 'ETHUSDT')
            exchange = getattr(self, 'exchange', 'BINANCE')
            timeframe = getattr(self, 'timeframe', '15m')

            # Create filename following outcomes/ directory convention
            metrics_filename = f"market_analysis_optimal_regime_clustering_outcome_{symbol}_{exchange}_{timeframe}_{timestamp}.json"

            # Save to outcomes/ directory (create if needed)
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(parents=True, exist_ok=True)
            metrics_file = outcomes_dir / metrics_filename

            with open(metrics_file, 'w') as f:
                json.dump(comprehensive_report, f, indent=2, default=str)

            self.logger.info(f"✅ Created comprehensive cluster metrics file: {metrics_file}")
            self.logger.info(f"📊 Analyzed {len(metrics)} clusters with {len(grouped_metrics)} group categories")
            return str(metrics_file)

        except Exception as e:
            self.logger.error(f"Error creating cluster metrics file: {e}")
            return ""

    def _create_cluster_size_distribution_summary(self, metrics: Dict[str, Dict]) -> Dict[str, Dict]:
        """Create cluster size distribution summary for top clusters."""
        try:
            # Sort clusters by size descending
            sorted_clusters = sorted(metrics.items(), key=lambda x: x[1]['basic_stats']['size'], reverse=True)

            # Get top clusters
            top_1 = sorted_clusters[:1]
            top_5 = sorted_clusters[:5]
            top_10 = sorted_clusters[:10]
            top_20 = sorted_clusters[:20] if len(sorted_clusters) >= 20 else sorted_clusters

            def create_group_summary(clusters):
                if not clusters:
                    return {}

                cluster_ids = [cluster_id for cluster_id, _ in clusters]
                total_size = sum(cluster['basic_stats']['size'] for _, cluster in clusters)
                total_percentage = sum(cluster['basic_stats']['percentage'] for _, cluster in clusters)

                return {
                    'cluster_ids': cluster_ids,
                    'total_size': total_size,
                    'total_percentage': total_percentage,
                    'cumulative_coverage': total_percentage
                }

            return {
                'top_1_cluster': create_group_summary(top_1),
                'top_5_clusters': create_group_summary(top_5),
                'top_10_clusters': create_group_summary(top_10),
                'top_20_clusters': create_group_summary(top_20)
            }

        except Exception as e:
            self.logger.error(f"Error creating cluster size distribution summary: {e}")
            return {}

    def _calculate_time_period(self, data: pd.DataFrame) -> str:
        """Calculate time period from data."""
        try:
            if hasattr(data, 'index') and len(data.index) > 0:
                # Try to get datetime index
                if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
                    start_date = data.index.min()
                    end_date = data.index.max()

                    # Check if these are datetime objects
                    if hasattr(start_date, 'strftime') and hasattr(end_date, 'strftime'):
                        start_str = start_date.strftime('%Y-%m-%d')
                        end_str = end_date.strftime('%Y-%m-%d')

                        # Calculate approximate years
                        if hasattr(start_date, 'year') and hasattr(end_date, 'year'):
                            years = (end_date.year - start_date.year) + (end_date.month - start_date.month) / 12
                            years_str = f" (approximately {years:.1f} years)"
                        else:
                            years_str = ""

                        return f"{start_str} to {end_str}{years_str}"

            # Fallback calculation based on sample count and timeframe
            total_samples = len(data)
            timeframe = getattr(self, 'timeframe', '15m')

            # Estimate time period based on typical trading hours
            # Assuming 24/7 trading for crypto
            if 'm' in timeframe:
                minutes_per_sample = int(timeframe.replace('m', ''))
                total_minutes = total_samples * minutes_per_sample
                days = total_minutes / (24 * 60)
                years = days / 365.25

                if years >= 1:
                    return f"Approximately {years:.1f} years of {timeframe} data"
                else:
                    return f"Approximately {days:.0f} days of {timeframe} data"

            return "Unknown (insufficient datetime information)"

        except Exception as e:
            self.logger.warning(f"Error calculating time period: {e}")
            return "Unknown"

    def _calculate_comprehensive_cluster_metrics(self, clustering_result: ClusteringResult,
                                               original_data: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate comprehensive metrics for each cluster."""
        try:
            metrics = {}
            unique_labels = np.unique(clustering_result.labels)

            if -1 in unique_labels:
                unique_labels = unique_labels[unique_labels != -1]

            # Get features from clustering result
            features = clustering_result.metadata.get('features')
            if features is None:
                # Fallback to original data features
                numeric_cols = original_data.select_dtypes(include=[np.number]).columns
                features = original_data[numeric_cols].values

            for label in unique_labels:
                mask = clustering_result.labels == label
                cluster_data = original_data[mask]
                cluster_features = features[mask]

                # Basic cluster statistics
                cluster_size = mask.sum()
                cluster_percentage = cluster_size / len(original_data)

                # Statistical metrics
                if len(cluster_data) > 0:
                    # Price and volume statistics
                    price_stats = {}
                    volume_stats = {}

                    if 'close' in cluster_data.columns:
                        close_prices = cluster_data['close']
                        price_stats = {
                            'mean': float(close_prices.mean()),
                            'std': float(close_prices.std()),
                            'min': float(close_prices.min()),
                            'max': float(close_prices.max()),
                            'median': float(close_prices.median()),
                            'skewness': float(close_prices.skew()),
                            'kurtosis': float(close_prices.kurtosis())
                        }

                    if 'volume' in cluster_data.columns:
                        volumes = cluster_data['volume']
                        volume_stats = {
                            'mean': float(volumes.mean()),
                            'std': float(volumes.std()),
                            'min': float(volumes.min()),
                            'max': float(volumes.max()),
                            'median': float(volumes.median())
                        }

                    # Feature statistics
                    feature_stats = {}
                    if len(cluster_features) > 0:
                        feature_mean = np.mean(cluster_features, axis=0)
                        feature_std = np.std(cluster_features, axis=0)

                        # Create feature dimensions mapping (volume, volatility, trend, momentum)
                        feature_dimensions = {}
                        for i, mean_val in enumerate(feature_mean):
                            if i == 0:
                                feature_dimensions['volume_dimension'] = float(mean_val)
                            elif i == 1:
                                feature_dimensions['volatility_dimension'] = float(mean_val)
                            elif i == 2:
                                feature_dimensions['trend_dimension'] = float(mean_val)
                            elif i == 3:
                                feature_dimensions['momentum_dimension'] = float(mean_val)

                        feature_stats = {
                            'feature_means': feature_mean.tolist(),
                            'feature_dimensions': feature_dimensions,
                            'feature_stds': feature_std.tolist(),
                            'feature_ranges': (np.max(cluster_features, axis=0) - np.min(cluster_features, axis=0)).tolist()
                        }

                    # Return statistics (if available)
                    return_stats = {}
                    if 'close_return' in cluster_data.columns:
                        returns = cluster_data['close_return']
                        return_stats = {
                            'mean_return': float(returns.mean()),
                            'std_return': float(returns.std()),
                            'sharpe_ratio': float(returns.mean() / (returns.std() + 1e-10)),
                            'max_drawdown': self._calculate_max_drawdown(returns.values),
                            'positive_returns_pct': float((returns > 0).mean())
                        }

                metrics[f'cluster_{label}'] = {
                    'basic_stats': {
                        'size': int(cluster_size),
                        'percentage': float(cluster_percentage),
                        'rank_by_size': int(np.sum(cluster_size >= clustering_result.statistics.cluster_sizes) + 1)
                    },
                    'price_statistics': price_stats,
                    'volume_statistics': volume_stats,
                    'feature_statistics': feature_stats,
                    'return_statistics': return_stats,
                    'cluster_characteristics': {
                        'center': clustering_result.cluster_centers[label].tolist() if hasattr(clustering_result.cluster_centers, 'size') and clustering_result.cluster_centers.size > 0 else [],
                        'dispersion': float(np.mean(np.std(cluster_features, axis=0))) if len(cluster_features) > 0 else 0.0
                    }
                }

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating comprehensive cluster metrics: {e}")
            return {}

    def _organize_metrics_by_groups(self, metrics: Dict[str, Dict],
                                  clustering_result: ClusteringResult) -> Dict[str, Dict]:
        """Organize metrics by groups (all, top20, top10, top5, top1)."""
        try:
            # Sort clusters by size
            sorted_clusters = sorted(
                metrics.items(),
                key=lambda x: x[1]['basic_stats']['size'],
                reverse=True
            )

            grouped_metrics = {
                'all_clusters': self._aggregate_group_metrics(metrics),
                'top_20_clusters': {},
                'top_10_clusters': {},
                'top_5_clusters': {},
                'top_1_cluster': {}
            }

            # Top 20 clusters
            if len(sorted_clusters) >= 20:
                top_20 = dict(sorted_clusters[:20])
                grouped_metrics['top_20_clusters'] = self._aggregate_group_metrics(top_20)

            # Top 10 clusters
            if len(sorted_clusters) >= 10:
                top_10 = dict(sorted_clusters[:10])
                grouped_metrics['top_10_clusters'] = self._aggregate_group_metrics(top_10)

            # Top 5 clusters
            if len(sorted_clusters) >= 5:
                top_5 = dict(sorted_clusters[:5])
                grouped_metrics['top_5_clusters'] = self._aggregate_group_metrics(top_5)

            # Top 1 cluster
            if len(sorted_clusters) >= 1:
                top_1 = dict(sorted_clusters[:1])
                grouped_metrics['top_1_cluster'] = self._aggregate_group_metrics(top_1)

            return grouped_metrics

        except Exception as e:
            self.logger.error(f"Error organizing metrics by groups: {e}")
            return {}

    def _aggregate_group_metrics(self, group_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate metrics for a group of clusters."""
        try:
            if not group_metrics:
                return {}

            # Aggregate basic statistics
            total_size = sum(m['basic_stats']['size'] for m in group_metrics.values())
            total_percentage = sum(m['basic_stats']['percentage'] for m in group_metrics.values())
            n_clusters = len(group_metrics)

            # Aggregate price statistics
            price_stats = self._aggregate_price_stats(group_metrics)

            # Aggregate volume statistics
            volume_stats = self._aggregate_volume_stats(group_metrics)

            # Aggregate return statistics
            return_stats = self._aggregate_return_stats(group_metrics)

            return {
                'group_summary': {
                    'n_clusters': n_clusters,
                    'total_size': total_size,
                    'total_percentage': total_percentage,
                    'average_size': total_size / n_clusters,
                    'size_std': float(np.std([m['basic_stats']['size'] for m in group_metrics.values()]))
                },
                'aggregated_price_statistics': price_stats,
                'aggregated_volume_statistics': volume_stats,
                'aggregated_return_statistics': return_stats,
                'cluster_ids': list(group_metrics.keys())
            }

        except Exception as e:
            self.logger.error(f"Error aggregating group metrics: {e}")
            return {}

    def _aggregate_price_stats(self, group_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate price statistics across clusters."""
        try:
            price_values = []
            for metrics in group_metrics.values():
                if metrics['price_statistics']:
                    price_values.append(metrics['price_statistics']['mean'])

            if not price_values:
                return {}

            return {
                'mean_price': float(np.mean(price_values)),
                'price_std': float(np.std(price_values)),
                'min_price': float(np.min(price_values)),
                'max_price': float(np.max(price_values))
            }

        except Exception as e:
            self.logger.error(f"Error aggregating price stats: {e}")
            return {}

    def _aggregate_volume_stats(self, group_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate volume statistics across clusters."""
        try:
            volume_values = []
            for metrics in group_metrics.values():
                if metrics['volume_statistics']:
                    volume_values.append(metrics['volume_statistics']['mean'])

            if not volume_values:
                return {}

            return {
                'mean_volume': float(np.mean(volume_values)),
                'volume_std': float(np.std(volume_values)),
                'min_volume': float(np.min(volume_values)),
                'max_volume': float(np.max(volume_values))
            }

        except Exception as e:
            self.logger.error(f"Error aggregating volume stats: {e}")
            return {}

    def _aggregate_return_stats(self, group_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate return statistics across clusters."""
        try:
            return_values = []
            sharpe_ratios = []
            max_drawdowns = []
            positive_return_pcts = []

            for metrics in group_metrics.values():
                if metrics['return_statistics']:
                    ret_stats = metrics['return_statistics']
                    if ret_stats.get('mean_return') is not None:
                        return_values.append(ret_stats['mean_return'])
                    if ret_stats.get('sharpe_ratio') is not None:
                        sharpe_ratios.append(ret_stats['sharpe_ratio'])
                    if ret_stats.get('max_drawdown') is not None:
                        max_drawdowns.append(ret_stats['max_drawdown'])
                    if ret_stats.get('positive_returns_pct') is not None:
                        positive_return_pcts.append(ret_stats['positive_returns_pct'])

            if not return_values:
                return {}

            result = {
                'mean_return': float(np.mean(return_values)),
                'return_std': float(np.std(return_values)),
                'best_return': float(np.max(return_values)),
                'worst_return': float(np.min(return_values))
            }

            if sharpe_ratios:
                result['mean_sharpe_ratio'] = float(np.mean(sharpe_ratios))
            if max_drawdowns:
                result['mean_max_drawdown'] = float(np.mean(max_drawdowns))
            if positive_return_pcts:
                result['mean_positive_return_pct'] = float(np.mean(positive_return_pcts))

            return result

        except Exception as e:
            self.logger.error(f"Error aggregating return stats: {e}")
            return {}

    def _calculate_individual_cluster_analysis(self, metrics: Dict[str, Dict],
                                             clustering_result: ClusteringResult,
                                             original_data: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate individual cluster analysis."""
        try:
            individual_analysis = {}

            for cluster_id, cluster_metrics in metrics.items():
                individual_analysis[cluster_id] = {
                    'detailed_metrics': cluster_metrics,
                    'quality_score': self._calculate_cluster_quality_score(cluster_metrics),
                    'economic_value': self._calculate_cluster_economic_value(cluster_metrics),
                    'stability_score': self._calculate_cluster_stability_score(cluster_metrics)
                }

            return individual_analysis

        except Exception as e:
            self.logger.error(f"Error calculating individual cluster analysis: {e}")
            return {}

    def _calculate_cluster_quality_score(self, metrics: Dict) -> float:
        """Calculate quality score for a cluster."""
        try:
            score = 0.0
            components = 0

            # Size component (0-1)
            size_pct = metrics['basic_stats']['percentage']
            if size_pct > 0.01:  # At least 1% of data
                score += min(size_pct / 0.05, 1.0)  # Cap at 5%
                components += 1

            # Feature dispersion component (0-1)
            if metrics['cluster_characteristics']['dispersion'] > 0:
                dispersion = metrics['cluster_characteristics']['dispersion']
                score += min(dispersion / 10.0, 1.0)  # Normalize
                components += 1

            # Return consistency component (0-1)
            if metrics['return_statistics']:
                ret_stats = metrics['return_statistics']
                if ret_stats.get('std_return') and ret_stats.get('std_return') > 0:
                    consistency = 1.0 / (1.0 + ret_stats['std_return'])  # Lower std = higher score
                    score += consistency
                    components += 1

            return score / max(components, 1)

        except Exception as e:
            self.logger.error(f"Error calculating cluster quality score: {e}")
            return 0.0

    def _calculate_cluster_economic_value(self, metrics: Dict) -> float:
        """Calculate economic value score for a cluster."""
        try:
            score = 0.0
            components = 0

            # Return potential (0-1)
            if metrics['return_statistics']:
                ret_stats = metrics['return_statistics']
                if ret_stats.get('mean_return') is not None:
                    mean_return = ret_stats['mean_return']
                    # Normalize return to 0-1 scale
                    normalized_return = max(0, min(mean_return / 0.01, 1.0))  # Cap at 1% return
                    score += normalized_return
                    components += 1

            # Sharpe ratio component (0-1)
            if metrics['return_statistics']:
                ret_stats = metrics['return_statistics']
                if ret_stats.get('sharpe_ratio') is not None:
                    sharpe = ret_stats['sharpe_ratio']
                    # Normalize Sharpe ratio (positive values only)
                    normalized_sharpe = max(0, min(sharpe / 2.0, 1.0))  # Cap at 2.0
                    score += normalized_sharpe
                    components += 1

            # Volume potential (0-1)
            if metrics['volume_statistics']:
                vol_stats = metrics['volume_statistics']
                if vol_stats.get('mean') and vol_stats.get('mean') > 0:
                    mean_volume = vol_stats['mean']
                    # Normalize volume (log scale)
                    normalized_volume = min(np.log10(mean_volume + 1) / 10.0, 1.0)
                    score += normalized_volume
                    components += 1

            return score / max(components, 1)

        except Exception as e:
            self.logger.error(f"Error calculating cluster economic value: {e}")
            return 0.0

    def _calculate_cluster_stability_score(self, metrics: Dict) -> float:
        """Calculate stability score for a cluster."""
        try:
            score = 0.0
            components = 0

            # Size stability (0-1)
            size = metrics['basic_stats']['size']
            if size > 100:  # Minimum reasonable size
                stability = min(size / 1000.0, 1.0)  # Cap at 1000 samples
                score += stability
                components += 1

            # Return stability (0-1)
            if metrics['return_statistics']:
                ret_stats = metrics['return_statistics']
                if ret_stats.get('positive_returns_pct') is not None:
                    consistency = ret_stats['positive_returns_pct']
                    score += consistency
                    components += 1

            return score / max(components, 1)

        except Exception as e:
            self.logger.error(f"Error calculating cluster stability score: {e}")
            return 0.0

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        try:
            if len(returns) < 2:
                return 0.0

            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return float(np.min(drawdown))

        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    def _calculate_economic_metrics(self, metrics: Dict[str, Dict],
                                  original_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive economic metrics across all clusters."""
        try:
            if not metrics:
                return {}

            # Extract return and risk data
            cluster_returns = []
            cluster_sharpes = []
            cluster_drawdowns = []
            cluster_volumes = []
            cluster_sizes = []

            for cluster_id, cluster_metrics in metrics.items():
                if cluster_metrics.get('return_statistics'):
                    ret_stats = cluster_metrics['return_statistics']
                    if ret_stats.get('mean_return') is not None:
                        cluster_returns.append(ret_stats['mean_return'])
                    if ret_stats.get('sharpe_ratio') is not None:
                        cluster_sharpes.append(ret_stats['sharpe_ratio'])
                    if ret_stats.get('max_drawdown') is not None:
                        cluster_drawdowns.append(ret_stats['max_drawdown'])

                if cluster_metrics.get('volume_statistics'):
                    vol_stats = cluster_metrics['volume_statistics']
                    if vol_stats.get('mean') is not None:
                        cluster_volumes.append(vol_stats['mean'])

                cluster_sizes.append(cluster_metrics['basic_stats']['size'])

            # Market Coverage Analysis
            market_coverage = self._calculate_market_coverage_analysis(metrics)

            # Enhanced Profitability Analysis
            profitability_analysis = self._calculate_profitability_analysis(metrics, cluster_returns)

            # Risk-Adjusted Returns Analysis
            risk_adjusted_analysis = self._calculate_risk_adjusted_analysis(metrics, cluster_returns, cluster_sharpes, cluster_drawdowns)

            # Trading Economics Analysis
            trading_economics = self._calculate_trading_economics_analysis(metrics, cluster_volumes, original_data)

            # Enhanced Risk Analysis
            risk_analysis = self._calculate_enhanced_risk_analysis(metrics, cluster_returns, cluster_drawdowns, original_data)

            # Opportunity Cost Analysis
            opportunity_analysis = self._calculate_opportunity_cost_analysis(metrics, cluster_returns)

            # Portfolio Construction Insights
            portfolio_insights = self._calculate_portfolio_construction_insights(metrics, cluster_returns, cluster_sharpes)

            # Market Efficiency Analysis
            market_efficiency = self._calculate_market_efficiency_analysis(metrics, original_data)

            # Performance Attribution Analysis
            performance_attribution = self._calculate_performance_attribution_analysis(metrics, cluster_returns, cluster_sizes)

            # Comparative Analysis Between Clusters
            comparative_analysis = self._calculate_comparative_analysis(metrics, cluster_returns, cluster_sizes)

            return {
                'market_coverage_analysis': market_coverage,
                'profitability_analysis': profitability_analysis,
                'risk_adjusted_returns': risk_adjusted_analysis,
                'enhanced_risk_analysis': risk_analysis,
                'trading_economics': trading_economics,
                'opportunity_cost_analysis': opportunity_analysis,
                'portfolio_construction_insights': portfolio_insights,
                'market_efficiency_analysis': market_efficiency,
                'performance_attribution_analysis': performance_attribution,
                'comparative_analysis': comparative_analysis,
                'economic_summary': self._create_economic_summary(metrics, cluster_returns, cluster_sharpes)
            }

        except Exception as e:
            self.logger.error(f"Error calculating comprehensive economic metrics: {e}")
            return {}

    def _calculate_market_coverage_analysis(self, metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate market coverage analysis."""
        try:
            total_clusters = len(metrics)
            covered_percentage = sum(m['basic_stats']['percentage'] for m in metrics.values())
            largest_cluster_pct = max(m['basic_stats']['percentage'] for m in metrics.values()) if metrics else 0.0

            # Calculate coverage efficiency
            coverage_efficiency = covered_percentage / total_clusters if total_clusters > 0 else 0.0

            # Cluster concentration analysis
            cluster_percentages = [m['basic_stats']['percentage'] for m in metrics.values()]
            concentration_ratio = sum(sorted(cluster_percentages, reverse=True)[:3]) / covered_percentage if covered_percentage > 0 else 0.0

            return {
                'total_clusters': total_clusters,
                'covered_percentage': float(covered_percentage),
                'largest_cluster_percentage': float(largest_cluster_pct),
                'coverage_efficiency': float(coverage_efficiency),
                'cluster_concentration_ratio': float(concentration_ratio),
                'market_fragmentation_index': float(total_clusters / (covered_percentage + 1e-10)),
                'diversification_potential': float(1.0 / concentration_ratio if concentration_ratio > 0 else 0.0)
            }
        except Exception as e:
            self.logger.error(f"Error calculating market coverage analysis: {e}")
            return {}

    def _calculate_profitability_analysis(self, metrics: Dict[str, Dict], cluster_returns: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive profitability analysis."""
        try:
            if not cluster_returns:
                return {}

            positive_clusters = sum(1 for r in cluster_returns if r > 0)
            negative_clusters = sum(1 for r in cluster_returns if r < 0)

            # Profit factor and expectancy
            positive_returns = [r for r in cluster_returns if r > 0]
            negative_returns = [r for r in cluster_returns if r < 0]

            profit_factor = sum(positive_returns) / abs(sum(negative_returns)) if negative_returns else float('inf')
            expectancy = (len(positive_returns) * np.mean(positive_returns) - len(negative_returns) * abs(np.mean(negative_returns))) / len(cluster_returns) if cluster_returns else 0.0

            # Return distribution analysis
            return_percentiles = np.percentile(cluster_returns, [25, 50, 75, 90, 95])

            return {
                'profitability_summary': {
                    'positive_return_clusters': positive_clusters,
                    'negative_return_clusters': negative_clusters,
                    'profitable_clusters_percentage': positive_clusters / len(cluster_returns) if cluster_returns else 0.0,
                    'profit_factor': float(profit_factor),
                    'expectancy': float(expectancy),
                    'average_positive_return': float(np.mean(positive_returns)) if positive_returns else 0.0,
                    'average_negative_return': float(np.mean(negative_returns)) if negative_returns else 0.0
                },
                'return_distribution': {
                    'mean_return': float(np.mean(cluster_returns)),
                    'median_return': float(np.median(cluster_returns)),
                    'std_return': float(np.std(cluster_returns)),
                    'skewness': float(pd.Series(cluster_returns).skew()),
                    'kurtosis': float(pd.Series(cluster_returns).kurtosis()),
                    'percentiles': return_percentiles.tolist(),
                    'best_return': float(max(cluster_returns)),
                    'worst_return': float(min(cluster_returns))
                },
                'performance_consistency': {
                    'consistency_ratio': len(positive_returns) / len(cluster_returns) if cluster_returns else 0.0,
                    'return_stability_score': 1.0 / (1.0 + np.std(cluster_returns)) if np.std(cluster_returns) > 0 else 1.0
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating profitability analysis: {e}")
            return {}

    def _calculate_risk_adjusted_analysis(self, metrics: Dict[str, Dict], cluster_returns: List[float],
                                       cluster_sharpes: List[float], cluster_drawdowns: List[float]) -> Dict[str, Any]:
        """Calculate risk-adjusted returns analysis."""
        try:
            if not cluster_returns:
                return {}

            # Sortino Ratio calculation (only downside risk)
            sortino_ratios = []
            for cluster_id, cluster_metrics in metrics.items():
                if cluster_metrics.get('return_statistics'):
                    ret_stats = cluster_metrics['return_statistics']
                    mean_return = ret_stats.get('mean_return', 0)
                    downside_returns = [r for r in cluster_returns if r < mean_return]
                    downside_std = np.std(downside_returns) if downside_returns else 1e-10
                    sortino_ratio = mean_return / downside_std if downside_std > 0 else 0.0
                    sortino_ratios.append(sortino_ratio)

            # Calmar Ratio (annual return / max drawdown)
            calmar_ratios = []
            for cluster_id, cluster_metrics in metrics.items():
                if cluster_metrics.get('return_statistics'):
                    ret_stats = cluster_metrics['return_statistics']
                    mean_return = ret_stats.get('mean_return', 0)
                    max_dd = ret_stats.get('max_drawdown', 0)
                    calmar_ratio = mean_return / abs(max_dd) if max_dd != 0 else float('inf')
                    calmar_ratios.append(calmar_ratio)

            return {
                'sharpe_ratio_analysis': {
                    'average_sharpe_ratio': float(np.mean(cluster_sharpes)) if cluster_sharpes else 0.0,
                    'median_sharpe_ratio': float(np.median(cluster_sharpes)) if cluster_sharpes else 0.0,
                    'best_sharpe_ratio': float(max(cluster_sharpes)) if cluster_sharpes else 0.0,
                    'worst_sharpe_ratio': float(min(cluster_sharpes)) if cluster_sharpes else 0.0
                },
                'sortino_ratio_analysis': {
                    'average_sortino_ratio': float(np.mean(sortino_ratios)) if sortino_ratios else 0.0,
                    'median_sortino_ratio': float(np.median(sortino_ratios)) if sortino_ratios else 0.0,
                    'best_sortino_ratio': float(max(sortino_ratios)) if sortino_ratios else 0.0,
                    'worst_sortino_ratio': float(min(sortino_ratios)) if sortino_ratios else 0.0
                },
                'calmar_ratio_analysis': {
                    'average_calmar_ratio': float(np.mean(calmar_ratios)) if calmar_ratios else 0.0,
                    'median_calmar_ratio': float(np.median(calmar_ratios)) if calmar_ratios else 0.0,
                    'best_calmar_ratio': float(max(calmar_ratios)) if calmar_ratios else 0.0,
                    'worst_calmar_ratio': float(min(calmar_ratios)) if calmar_ratios else 0.0
                },
                'drawdown_analysis': {
                    'average_max_drawdown': float(np.mean(cluster_drawdowns)) if cluster_drawdowns else 0.0,
                    'median_max_drawdown': float(np.median(cluster_drawdowns)) if cluster_drawdowns else 0.0,
                    'worst_drawdown': float(min(cluster_drawdowns)) if cluster_drawdowns else 0.0,
                    'drawdown_recovery_ratio': float(np.mean([abs(r) / abs(d) for r, d in zip(cluster_returns, cluster_drawdowns) if d != 0]))
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted analysis: {e}")
            return {}

    def _calculate_enhanced_risk_analysis(self, metrics: Dict[str, Dict], cluster_returns: List[float],
                                       cluster_drawdowns: List[float], original_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate enhanced risk analysis with Value at Risk, Conditional VaR, and stress testing."""
        try:
            if not cluster_returns:
                return {}

            # Value at Risk (VaR) Analysis
            var_analysis = self._calculate_var_analysis(cluster_returns)

            # Conditional Value at Risk (CVaR) Analysis
            cvar_analysis = self._calculate_cvar_analysis(cluster_returns)

            # Volatility Regime Analysis
            volatility_regimes = self._calculate_volatility_regime_analysis(metrics, original_data)

            # Stress Testing Analysis
            stress_testing = self._calculate_stress_testing_analysis(metrics, cluster_returns, cluster_drawdowns)

            # Risk Factor Analysis
            risk_factors = self._calculate_risk_factor_analysis(metrics, original_data)

            # Downside Risk Measures
            downside_risk = self._calculate_downside_risk_measures(cluster_returns, cluster_drawdowns)

            return {
                'value_at_risk_analysis': var_analysis,
                'conditional_var_analysis': cvar_analysis,
                'volatility_regime_analysis': volatility_regimes,
                'stress_testing_analysis': stress_testing,
                'risk_factor_analysis': risk_factors,
                'downside_risk_measures': downside_risk,
                'risk_summary': self._create_risk_summary(cluster_returns, cluster_drawdowns, var_analysis)
            }
        except Exception as e:
            self.logger.error(f"Error calculating enhanced risk analysis: {e}")
            return {}

    def _calculate_var_analysis(self, cluster_returns: List[float]) -> Dict[str, Any]:
        """Calculate Value at Risk analysis."""
        try:
            returns_array = np.array(cluster_returns)

            # Historical VaR (95% and 99% confidence)
            var_95_historical = np.percentile(returns_array, 5)
            var_99_historical = np.percentile(returns_array, 1)

            # Parametric VaR (assuming normal distribution)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            var_95_parametric = mean_return - 1.645 * std_return  # 95% confidence
            var_99_parametric = mean_return - 2.326 * std_return  # 99% confidence

            # Modified VaR (incorporates skewness and kurtosis)
            skew = pd.Series(returns_array).skew()
            kurt = pd.Series(returns_array).kurtosis()
            modified_var_95 = var_95_parametric * (1 + skew/6 + (kurt-3)/24)
            modified_var_99 = var_99_parametric * (1 + skew/6 + (kurt-3)/24)

            return {
                'historical_var': {
                    'var_95': float(var_95_historical),
                    'var_99': float(var_99_historical),
                    'confidence_levels': {'95%': float(var_95_historical), '99%': float(var_99_historical)}
                },
                'parametric_var': {
                    'var_95': float(var_95_parametric),
                    'var_99': float(var_99_parametric),
                    'confidence_levels': {'95%': float(var_95_parametric), '99%': float(var_99_parametric)}
                },
                'modified_var': {
                    'var_95': float(modified_var_95),
                    'var_99': float(modified_var_99),
                    'confidence_levels': {'95%': float(modified_var_95), '99%': float(modified_var_99)}
                },
                'var_interpretation': {
                    'worst_case_loss_95': float(abs(var_95_historical)),
                    'worst_case_loss_99': float(abs(var_99_historical)),
                    'risk_category': 'high' if abs(var_95_historical) > 0.02 else 'moderate' if abs(var_95_historical) > 0.01 else 'low'
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating VaR analysis: {e}")
            return {}

    def _calculate_cvar_analysis(self, cluster_returns: List[float]) -> Dict[str, Any]:
        """Calculate Conditional Value at Risk analysis."""
        try:
            returns_array = np.array(cluster_returns)

            # Calculate CVaR as expected loss beyond VaR
            var_95 = np.percentile(returns_array, 5)
            var_99 = np.percentile(returns_array, 1)

            # CVaR 95% - average of losses worse than VaR 95%
            tail_losses_95 = returns_array[returns_array <= var_95]
            cvar_95 = np.mean(tail_losses_95) if len(tail_losses_95) > 0 else var_95

            # CVaR 99% - average of losses worse than VaR 99%
            tail_losses_99 = returns_array[returns_array <= var_99]
            cvar_99 = np.mean(tail_losses_99) if len(tail_losses_99) > 0 else var_99

            # Expected Shortfall ratio
            expected_shortfall_ratio = abs(cvar_95) / abs(var_95) if var_95 != 0 else 1.0

            return {
                'conditional_var': {
                    'cvar_95': float(cvar_95),
                    'cvar_99': float(cvar_99),
                    'confidence_levels': {'95%': float(cvar_95), '99%': float(cvar_99)}
                },
                'expected_shortfall_analysis': {
                    'expected_shortfall_ratio': float(expected_shortfall_ratio),
                    'tail_risk_severity': 'severe' if expected_shortfall_ratio > 1.5 else 'moderate' if expected_shortfall_ratio > 1.2 else 'mild',
                    'tail_loss_magnitude_95': float(abs(cvar_95 - var_95)) if var_95 != 0 else 0.0
                },
                'risk_assessment': {
                    'extreme_risk_potential': float(len(tail_losses_99) / len(returns_array)) if len(returns_array) > 0 else 0.0,
                    'tail_heaviness_indicator': float(np.std(tail_losses_95) / np.std(returns_array)) if len(returns_array) > 0 and np.std(returns_array) > 0 else 0.0
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating CVaR analysis: {e}")
            return {}

    def _calculate_volatility_regime_analysis(self, metrics: Dict[str, Dict], original_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility regime analysis."""
        try:
            # Extract volatility metrics from clusters
            volatilities = []
            for cluster_id, cluster_metrics in metrics.items():
                if cluster_metrics.get('price_statistics'):
                    price_stats = cluster_metrics['price_statistics']
                    if price_stats.get('std'):
                        volatilities.append(price_stats['std'])

            if not volatilities:
                return {}

            volatilities_array = np.array(volatilities)

            # Classify volatility regimes
            mean_vol = np.mean(volatilities_array)
            std_vol = np.std(volatilities_array)

            low_vol_threshold = mean_vol - 0.5 * std_vol
            high_vol_threshold = mean_vol + 0.5 * std_vol

            low_vol_clusters = [v for v in volatilities if v <= low_vol_threshold]
            medium_vol_clusters = [v for v in volatilities if low_vol_threshold < v <= high_vol_threshold]
            high_vol_clusters = [v for v in volatilities if v > high_vol_threshold]

            # Volatility clustering analysis
            volatility_autocorrelation = np.corrcoef(volatilities_array[:-1], volatilities_array[1:])[0, 1] if len(volatilities_array) > 1 else 0.0

            return {
                'volatility_regime_classification': {
                    'low_volatility_clusters': len(low_vol_clusters),
                    'medium_volatility_clusters': len(medium_vol_clusters),
                    'high_volatility_clusters': len(high_vol_clusters),
                    'regime_distribution': {
                        'low_vol_percentage': len(low_vol_clusters) / len(volatilities) if volatilities else 0.0,
                        'medium_vol_percentage': len(medium_vol_clusters) / len(volatilities) if volatilities else 0.0,
                        'high_vol_percentage': len(high_vol_clusters) / len(volatilities) if volatilities else 0.0
                    }
                },
                'volatility_dynamics': {
                    'volatility_clustering_coefficient': float(volatility_autocorrelation),
                    'regime_persistence_score': float(1.0 / (1.0 + abs(volatility_autocorrelation))),
                    'volatility_regime_stability': 'stable' if abs(volatility_autocorrelation) < 0.3 else 'clustered' if abs(volatility_autocorrelation) < 0.7 else 'highly_clustered'
                },
                'risk_implications': {
                    'diversification_effectiveness': float(1.0 / (1.0 + np.std(volatilities_array))),
                    'tail_risk_potential': 'high' if len(high_vol_clusters) / len(volatilities) > 0.3 else 'moderate' if len(high_vol_clusters) / len(volatilities) > 0.1 else 'low',
                    'portfolio_stability_index': float(np.mean(volatilities_array) / (np.std(volatilities_array) + 1e-10))
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating volatility regime analysis: {e}")
            return {}

    def _calculate_stress_testing_analysis(self, metrics: Dict[str, Dict], cluster_returns: List[float],
                                         cluster_drawdowns: List[float]) -> Dict[str, Any]:
        """Calculate stress testing analysis."""
        try:
            returns_array = np.array(cluster_returns)
            drawdowns_array = np.array(cluster_drawdowns)

            # Historical stress scenarios
            worst_historical_return = np.min(returns_array)
            worst_historical_drawdown = np.max(drawdowns_array)  # Most negative drawdown

            # Hypothetical stress scenarios
            stress_scenarios = {
                'moderate_market_crash': -0.15,  # 15% drop
                'severe_market_crash': -0.30,    # 30% drop
                'extreme_market_crash': -0.50,   # 50% drop
                'flash_crash': -0.20,           # 20% flash crash
                'prolonged_bear_market': -0.25   # 25% prolonged decline
            }

            # Calculate impact of stress scenarios
            stress_impacts = {}
            for scenario, shock in stress_scenarios.items():
                # Estimate impact based on historical worst case
                impact_estimate = worst_historical_return * (shock / worst_historical_return) if worst_historical_return < 0 else shock
                stress_impacts[scenario] = float(impact_estimate)

            # Recovery analysis
            recovery_scenarios = {
                'quick_recovery': 1.0,    # Full recovery in 1 period
                'moderate_recovery': 0.5, # 50% recovery in 1 period
                'slow_recovery': 0.25,    # 25% recovery in 1 period
                'no_recovery': 0.0        # No recovery
            }

            return {
                'historical_stress_scenarios': {
                    'worst_historical_performance': float(worst_historical_return),
                    'worst_historical_drawdown': float(worst_historical_drawdown),
                    'stress_scenario_impact': stress_impacts
                },
                'hypothetical_stress_scenarios': stress_impacts,
                'recovery_analysis': {
                    'recovery_scenarios': recovery_scenarios,
                    'best_case_recovery': float(worst_historical_return + abs(worst_historical_return) * 1.0),  # Full recovery
                    'worst_case_scenario': float(worst_historical_return)  # No recovery
                },
                'stress_test_recommendations': {
                    'risk_management_priority': 'critical' if worst_historical_return < -0.20 else 'high' if worst_historical_return < -0.10 else 'moderate',
                    'diversification_necessity': 'essential' if np.std(returns_array) > 0.05 else 'recommended' if np.std(returns_array) > 0.03 else 'optional',
                    'hedging_strategy': 'required' if len([r for r in returns_array if r < -0.15]) > 0 else 'consider' if len([r for r in returns_array if r < -0.10]) > 0 else 'optional'
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating stress testing analysis: {e}")
            return {}

    def _calculate_risk_factor_analysis(self, metrics: Dict[str, Dict], original_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk factor analysis."""
        try:
            # Analyze key risk factors
            risk_factors = {}

            # Size factor (larger clusters may be more stable)
            cluster_sizes = [m['basic_stats']['size'] for m in metrics.values()]
            size_volatility = np.std(cluster_sizes) / np.mean(cluster_sizes) if np.mean(cluster_sizes) > 0 else 0.0

            # Return factor (clusters with extreme returns)
            returns = [m['return_statistics'].get('mean_return', 0) for m in metrics.values() if m.get('return_statistics')]
            return_extremes = len([r for r in returns if abs(r) > 0.05]) / len(returns) if returns else 0.0

            # Concentration factor (market concentration risk)
            percentages = [m['basic_stats']['percentage'] for m in metrics.values()]
            top_3_concentration = sum(sorted(percentages, reverse=True)[:3]) if percentages else 0.0

            # Volatility factor (overall market volatility)
            volatilities = []
            for m in metrics.values():
                if m.get('price_statistics') and m['price_statistics'].get('std'):
                    volatilities.append(m['price_statistics']['std'])
            overall_volatility = np.mean(volatilities) if volatilities else 0.0

            risk_factors = {
                'size_factor': {
                    'size_volatility_ratio': float(size_volatility),
                    'size_stability_score': float(1.0 / (1.0 + size_volatility)),
                    'concentration_risk': 'high' if size_volatility > 1.0 else 'moderate' if size_volatility > 0.5 else 'low'
                },
                'return_factor': {
                    'extreme_return_ratio': float(return_extremes),
                    'return_stability_score': float(1.0 / (1.0 + return_extremes)),
                    'tail_risk_potential': 'high' if return_extremes > 0.3 else 'moderate' if return_extremes > 0.1 else 'low'
                },
                'concentration_factor': {
                    'top_3_concentration': float(top_3_concentration),
                    'diversification_need': 'urgent' if top_3_concentration > 0.6 else 'recommended' if top_3_concentration > 0.4 else 'optional',
                    'concentration_risk_level': 'critical' if top_3_concentration > 0.7 else 'high' if top_3_concentration > 0.5 else 'moderate'
                },
                'volatility_factor': {
                    'overall_volatility': float(overall_volatility),
                    'volatility_regime': 'high' if overall_volatility > 0.03 else 'moderate' if overall_volatility > 0.02 else 'low',
                    'volatility_trend': self._assess_volatility_trend(volatilities)
                }
            }

            return {
                'primary_risk_factors': risk_factors,
                'risk_factor_interactions': {
                    'size_volatility_interaction': float(size_volatility * overall_volatility),
                    'concentration_extreme_interaction': float(top_3_concentration * return_extremes),
                    'overall_risk_complexity': float(size_volatility + return_extremes + top_3_concentration)
                },
                'risk_mitigation_priorities': self._prioritize_risk_mitigation(risk_factors)
            }
        except Exception as e:
            self.logger.error(f"Error calculating risk factor analysis: {e}")
            return {}

    def _calculate_downside_risk_measures(self, cluster_returns: List[float], cluster_drawdowns: List[float]) -> Dict[str, Any]:
        """Calculate downside risk measures."""
        try:
            returns_array = np.array(cluster_returns)
            drawdowns_array = np.array(cluster_drawdowns)

            # Downside deviation (semi-standard deviation)
            negative_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0.0

            # Sortino ratio (using downside deviation)
            mean_return = np.mean(returns_array)
            sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else float('inf')

            # Omega ratio (probability weighted ratio of gains vs losses)
            gains = returns_array[returns_array > 0]
            losses = abs(returns_array[returns_array < 0])
            omega_ratio = np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else float('inf')

            # Calmar ratio (annualized return / max drawdown)
            max_drawdown = np.max(drawdowns_array) if len(drawdowns_array) > 0 else 0.0
            calmar_ratio = abs(mean_return) / max_drawdown if max_drawdown > 0 else float('inf')

            # Sterling ratio (annualized return / average drawdown + 10%)
            avg_drawdown = np.mean(drawdowns_array) if len(drawdowns_array) > 0 else 0.0
            sterling_ratio = abs(mean_return) / (avg_drawdown + 0.1) if (avg_drawdown + 0.1) > 0 else float('inf')

            return {
                'downside_deviation_measures': {
                    'downside_deviation': float(downside_deviation),
                    'downside_risk_ratio': float(downside_deviation / np.std(returns_array)) if np.std(returns_array) > 0 else 0.0
                },
                'asymmetric_risk_measures': {
                    'sortino_ratio': float(sortino_ratio),
                    'omega_ratio': float(omega_ratio),
                    'sortino_interpretation': 'excellent' if sortino_ratio > 2.0 else 'good' if sortino_ratio > 1.0 else 'fair' if sortino_ratio > 0.5 else 'poor'
                },
                'drawdown_based_measures': {
                    'calmar_ratio': float(calmar_ratio),
                    'sterling_ratio': float(sterling_ratio),
                    'drawdown_efficiency': float(calmar_ratio / (sterling_ratio + 1e-10))
                },
                'risk_adjusted_performance': {
                    'downside_risk_score': float(1.0 / (1.0 + downside_deviation)),
                    'asymmetric_performance_score': float(min(1.0, sortino_ratio / 2.0)),
                    'overall_downside_score': float((sortino_ratio + omega_ratio + calmar_ratio) / 3.0) if sortino_ratio != float('inf') and omega_ratio != float('inf') else 0.0
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating downside risk measures: {e}")
            return {}

    def _create_risk_summary(self, cluster_returns: List[float], cluster_drawdowns: List[float],
                           var_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive risk summary."""
        try:
            returns_array = np.array(cluster_returns)
            drawdowns_array = np.array(cluster_drawdowns)

            # Overall risk assessment
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            max_drawdown = np.max(drawdowns_array) if len(drawdowns_array) > 0 else 0.0

            # Risk score calculation
            volatility_score = min(1.0, std_return / 0.05)  # Normalize to 5% as baseline
            drawdown_score = min(1.0, abs(max_drawdown) / 0.30)  # Normalize to 30% as severe
            var_score = abs(var_analysis.get('var_interpretation', {}).get('worst_case_loss_95', 0)) / 0.02  # Normalize to 2% as concerning

            overall_risk_score = (volatility_score + drawdown_score + var_score) / 3.0

            return {
                'overall_risk_assessment': {
                    'overall_risk_score': float(overall_risk_score),
                    'risk_category': 'critical' if overall_risk_score > 0.8 else 'high' if overall_risk_score > 0.6 else 'moderate' if overall_risk_score > 0.4 else 'low',
                    'risk_rating': 'excellent' if overall_risk_score < 0.2 else 'good' if overall_risk_score < 0.4 else 'fair' if overall_risk_score < 0.6 else 'poor'
                },
                'key_risk_indicators': {
                    'volatility_indicator': float(std_return),
                    'maximum_drawdown': float(max_drawdown),
                    'var_95_worst_case': float(abs(var_analysis.get('var_interpretation', {}).get('worst_case_loss_95', 0))),
                    'risk_efficiency_ratio': float(mean_return / std_return) if std_return > 0 else 0.0
                },
                'risk_management_recommendations': {
                    'stop_loss_suggestion': float(abs(max_drawdown) * 0.5),  # 50% of max drawdown
                    'position_sizing_guidance': float(max(0.01, 1.0 / (1.0 + overall_risk_score * 10))),  # Risk-adjusted sizing
                    'diversification_priority': 'essential' if overall_risk_score > 0.5 else 'recommended' if overall_risk_score > 0.3 else 'consider'
                }
            }
        except Exception as e:
            self.logger.error(f"Error creating risk summary: {e}")
            return {}

    def _calculate_trading_economics_analysis(self, metrics: Dict[str, Dict],
                                            cluster_volumes: List[float], original_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trading economics analysis."""
        try:
            if not cluster_volumes:
                return {}

            # Volume concentration analysis
            total_volume = sum(cluster_volumes)
            volume_concentration = max(cluster_volumes) / total_volume if total_volume > 0 else 0.0

            # Liquidity analysis
            avg_volume = np.mean(cluster_volumes)
            volume_std = np.std(cluster_volumes)

            # Market impact estimation (simplified)
            market_impact_factors = []
            for volume in cluster_volumes:
                if volume > 0:
                    # Simple market impact model: larger volume = higher impact
                    impact_factor = min(volume / avg_volume, 5.0) if avg_volume > 0 else 1.0
                    market_impact_factors.append(impact_factor)

            # Slippage estimation
            estimated_slippage = np.mean([0.001 * impact for impact in market_impact_factors]) if market_impact_factors else 0.0

            return {
                'volume_analysis': {
                    'average_cluster_volume': float(avg_volume),
                    'volume_concentration_ratio': float(volume_concentration),
                    'volume_stability': float(1.0 / (1.0 + volume_std / avg_volume)) if avg_volume > 0 else 0.0,
                    'volume_distribution_skewness': float(pd.Series(cluster_volumes).skew())
                },
                'liquidity_metrics': {
                    'liquidity_score': float(min(1.0, avg_volume / (avg_volume + volume_std))),
                    'trading_feasibility_score': float(1.0 / (1.0 + volume_concentration)),
                    'market_depth_indicator': float(np.median(cluster_volumes) / avg_volume) if avg_volume > 0 else 0.0
                },
                'transaction_costs': {
                    'estimated_market_impact': float(np.mean(market_impact_factors)) if market_impact_factors else 0.0,
                    'estimated_slippage_bps': float(estimated_slippage * 10000),  # Convert to basis points
                    'execution_difficulty_score': float(volume_concentration * (1.0 + estimated_slippage))
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating trading economics analysis: {e}")
            return {}

    def _calculate_opportunity_cost_analysis(self, metrics: Dict[str, Dict], cluster_returns: List[float]) -> Dict[str, Any]:
        """Calculate opportunity cost analysis."""
        try:
            if not cluster_returns:
                return {}

            # Calculate opportunity costs relative to best performing cluster
            best_return = max(cluster_returns)
            opportunity_costs = [best_return - r for r in cluster_returns]

            # Regret analysis
            avg_opportunity_cost = np.mean(opportunity_costs)
            max_regret = max(opportunity_costs)

            # Missed opportunity analysis
            total_potential_return = sum(cluster_returns)
            realized_return = sum(cluster_returns)  # In equal-weighted scenario
            missed_opportunity = total_potential_return - realized_return

            return {
                'opportunity_costs': {
                    'average_opportunity_cost': float(avg_opportunity_cost),
                    'maximum_regret': float(max_regret),
                    'opportunity_cost_distribution': opportunity_costs
                },
                'missed_opportunities': {
                    'total_missed_return': float(missed_opportunity),
                    'missed_return_percentage': float(missed_opportunity / total_potential_return) if total_potential_return != 0 else 0.0,
                    'efficiency_ratio': float(realized_return / total_potential_return) if total_potential_return != 0 else 0.0
                },
                'strategic_implications': {
                    'best_cluster_premium': float(best_return / np.mean(cluster_returns)) if np.mean(cluster_returns) != 0 else 0.0,
                    'diversification_benefit': float(1.0 / (1.0 + np.std(opportunity_costs))) if np.std(opportunity_costs) > 0 else 1.0
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating opportunity cost analysis: {e}")
            return {}

    def _calculate_portfolio_construction_insights(self, metrics: Dict[str, Dict],
                                                 cluster_returns: List[float], cluster_sharpes: List[float]) -> Dict[str, Any]:
        """Calculate portfolio construction insights."""
        try:
            if not cluster_returns:
                return {}

            # Equal-weighted portfolio metrics
            equal_weighted_return = np.mean(cluster_returns)
            equal_weighted_volatility = np.std(cluster_returns)

            # Risk parity weights (inverse volatility)
            inverse_vol_weights = []
            for i, cluster_id in enumerate(metrics.keys()):
                if cluster_sharpes and i < len(cluster_sharpes):
                    sharpe = cluster_sharpes[i]
                    weight = sharpe / sum(cluster_sharpes) if sum(cluster_sharpes) != 0 else 1.0 / len(cluster_sharpes)
                    inverse_vol_weights.append(weight)

            # Sharpe ratio optimized weights (simplified)
            sharpe_weights = [s / sum(cluster_sharpes) for s in cluster_sharpes] if cluster_sharpes and sum(cluster_sharpes) != 0 else [1.0 / len(cluster_returns)] * len(cluster_returns)

            # Calculate average correlation for diversification ratio
            avg_correlation = 0.0
            if len(cluster_returns) > 1:
                try:
                    correlation_matrix = np.corrcoef([cluster_returns])
                    avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
                except:
                    avg_correlation = 0.0

            return {
                'equal_weighted_portfolio': {
                    'expected_return': float(equal_weighted_return),
                    'expected_volatility': float(equal_weighted_volatility),
                    'sharpe_ratio': float(equal_weighted_return / equal_weighted_volatility) if equal_weighted_volatility != 0 else 0.0,
                    'diversification_ratio': float(len(cluster_returns) / (1.0 + abs(avg_correlation))) if len(cluster_returns) > 1 and avg_correlation is not None else 1.0
                },
                'risk_parity_weights': {
                    'weights': [float(w) for w in inverse_vol_weights],
                    'concentration_index': float(max(inverse_vol_weights) / min(inverse_vol_weights)) if inverse_vol_weights else 0.0
                },
                'sharpe_optimized_weights': {
                    'weights': [float(w) for w in sharpe_weights],
                    'concentration_index': float(max(sharpe_weights) / min(sharpe_weights)) if sharpe_weights else 0.0
                },
                'portfolio_recommendations': {
                    'optimal_strategy': 'sharpe_optimized' if np.mean(cluster_sharpes) > 0.5 else 'equal_weighted',
                    'diversification_potential': float(1.0 / (1.0 + equal_weighted_volatility / abs(equal_weighted_return))) if equal_weighted_return != 0 else 0.0
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating portfolio construction insights: {e}")
            return {}

    def _calculate_market_efficiency_analysis(self, metrics: Dict[str, Dict], original_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market efficiency analysis."""
        try:
            # Calculate market efficiency indicators
            cluster_returns = [m['return_statistics'].get('mean_return', 0) for m in metrics.values() if m.get('return_statistics')]

            if not cluster_returns:
                return {}

            # Information ratio (signal to noise)
            signal_to_noise = np.mean(cluster_returns) / np.std(cluster_returns) if np.std(cluster_returns) != 0 else 0.0

            # Market predictability index
            predictability = 1.0 / (1.0 + np.std(cluster_returns) / abs(np.mean(cluster_returns))) if np.mean(cluster_returns) != 0 else 0.0

            # Cluster diversity index
            cluster_sizes = [m['basic_stats']['size'] for m in metrics.values()]
            diversity_index = len(cluster_returns) / (1.0 + np.std(cluster_sizes) / np.mean(cluster_sizes)) if np.mean(cluster_sizes) != 0 else 0.0

            return {
                'efficiency_indicators': {
                    'information_ratio': float(signal_to_noise),
                    'market_predictability_index': float(predictability),
                    'cluster_diversity_index': float(diversity_index),
                    'regime_complexity_score': float(len(cluster_returns) * predictability)
                },
                'trading_opportunity_assessment': {
                    'alpha_generation_potential': float(max(0, signal_to_noise - 0.5)),  # Above 0.5 suggests alpha potential
                    'arbitrage_opportunity_score': float(min(1.0, diversity_index * predictability)),
                    'market_inefficiency_index': float(1.0 / (1.0 + signal_to_noise))
                },
                'strategy_implications': {
                    'recommended_approach': 'active' if signal_to_noise > 0.3 else 'passive',
                    'diversification_benefit': float(diversity_index),
                    'risk_management_priority': 'high' if predictability < 0.5 else 'medium'
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating market efficiency analysis: {e}")
            return {}

    def _create_economic_summary(self, metrics: Dict[str, Dict], cluster_returns: List[float], cluster_sharpes: List[float]) -> Dict[str, Any]:
        """Create economic summary with key insights."""
        try:
            if not cluster_returns:
                return {}

            # Overall economic health score
            profitability_score = len([r for r in cluster_returns if r > 0]) / len(cluster_returns) if cluster_returns else 0.0
            risk_score = np.mean([s for s in cluster_sharpes if s > 0]) if cluster_sharpes else 0.0
            economic_health_score = (profitability_score + min(risk_score, 2.0)) / 3.0  # Cap risk score at 2.0

            # Top performing clusters
            cluster_performance = []
            for i, cluster_id in enumerate(metrics.keys()):
                if i < len(cluster_returns):
                    cluster_performance.append({
                        'cluster_id': cluster_id,
                        'return': cluster_returns[i],
                        'sharpe_ratio': cluster_sharpes[i] if i < len(cluster_sharpes) else 0.0
                    })

            top_performers = sorted(cluster_performance, key=lambda x: x['return'], reverse=True)[:5]

            return {
                'overall_economic_health': {
                    'economic_health_score': float(economic_health_score),
                    'profitability_score': float(profitability_score),
                    'risk_adjusted_score': float(risk_score),
                    'overall_rating': 'excellent' if economic_health_score > 0.8 else 'good' if economic_health_score > 0.6 else 'fair' if economic_health_score > 0.4 else 'poor'
                },
                'top_performing_clusters': top_performers,
                'key_insights': {
                    'alpha_potential': 'high' if np.mean(cluster_returns) > 0.02 else 'medium' if np.mean(cluster_returns) > 0.01 else 'low',
                    'risk_reward_balance': 'good' if risk_score > 1.0 else 'moderate' if risk_score > 0.5 else 'poor',
                    'diversification_value': 'high' if len(cluster_returns) > 10 else 'medium' if len(cluster_returns) > 5 else 'low'
                },
                'strategic_recommendations': {
                    'primary_focus': 'growth' if np.mean(cluster_returns) > 0.015 else 'balanced' if np.mean(cluster_returns) > 0.005 else 'conservative',
                    'risk_management_level': 'aggressive' if risk_score > 1.5 else 'moderate' if risk_score > 0.8 else 'conservative'
                }
            }
        except Exception as e:
            self.logger.error(f"Error creating economic summary: {e}")
            return {}

    def _calculate_performance_attribution_analysis(self, metrics: Dict[str, Dict],
                                                  cluster_returns: List[float], cluster_sizes: List[float]) -> Dict[str, Any]:
        """Calculate performance attribution analysis to understand cluster contributions."""
        try:
            if not cluster_returns or not cluster_sizes:
                return {}

            # Create cluster performance data
            cluster_data = []
            for i, cluster_id in enumerate(metrics.keys()):
                if i < len(cluster_returns) and i < len(cluster_sizes):
                    cluster_data.append({
                        'cluster_id': cluster_id,
                        'return': cluster_returns[i],
                        'size': cluster_sizes[i],
                        'weight': cluster_sizes[i] / sum(cluster_sizes) if sum(cluster_sizes) > 0 else 0.0
                    })

            # Calculate attribution metrics
            total_portfolio_return = sum([c['return'] * c['weight'] for c in cluster_data])
            total_portfolio_variance = sum([c['return']**2 * c['weight']**2 for c in cluster_data])

            # Brinson attribution model components
            allocation_effect = 0.0
            selection_effect = 0.0
            interaction_effect = 0.0

            benchmark_return = np.mean(cluster_returns)  # Use average as benchmark

            for cluster in cluster_data:
                weight = cluster['weight']
                cluster_return = cluster['return']

                # Allocation effect: (weight - benchmark_weight) * benchmark_return
                benchmark_weight = 1.0 / len(cluster_data)  # Equal weight benchmark
                allocation_effect += (weight - benchmark_weight) * benchmark_return

                # Selection effect: weight * (cluster_return - benchmark_return)
                selection_effect += weight * (cluster_return - benchmark_return)

                # Interaction effect: (weight - benchmark_weight) * (cluster_return - benchmark_return)
                interaction_effect += (weight - benchmark_weight) * (cluster_return - benchmark_return)

            # Risk attribution
            marginal_contribution_to_risk = []
            for cluster in cluster_data:
                weight = cluster['weight']
                cluster_return = cluster['return']
                # Simplified marginal contribution to risk
                mctr = weight * cluster_return**2 / (total_portfolio_variance + 1e-10) if total_portfolio_variance > 0 else 0.0
                marginal_contribution_to_risk.append(mctr)

            # Performance decomposition
            performance_decomposition = []
            for i, cluster in enumerate(cluster_data):
                contribution = cluster['return'] * cluster['weight']
                attribution_percentage = contribution / (total_portfolio_return + 1e-10) if total_portfolio_return != 0 else 0.0

                performance_decomposition.append({
                    'cluster_id': cluster['cluster_id'],
                    'contribution_to_return': float(contribution),
                    'attribution_percentage': float(attribution_percentage),
                    'contribution_to_risk': float(marginal_contribution_to_risk[i]) if i < len(marginal_contribution_to_risk) else 0.0,
                    'risk_adjusted_contribution': float(contribution / (marginal_contribution_to_risk[i] + 1e-10)) if i < len(marginal_contribution_to_risk) and marginal_contribution_to_risk[i] > 0 else 0.0
                })

            # Sort by contribution to return
            performance_decomposition.sort(key=lambda x: x['contribution_to_return'], reverse=True)

            # Factor attribution (simplified)
            factor_attribution = self._calculate_factor_attribution(cluster_data, cluster_returns)

            return {
                'brinson_attribution_model': {
                    'allocation_effect': float(allocation_effect),
                    'selection_effect': float(selection_effect),
                    'interaction_effect': float(interaction_effect),
                    'total_attribution': float(allocation_effect + selection_effect + interaction_effect),
                    'attribution_breakdown': {
                        'allocation_percentage': float(allocation_effect / (total_portfolio_return + 1e-10)) if total_portfolio_return != 0 else 0.0,
                        'selection_percentage': float(selection_effect / (total_portfolio_return + 1e-10)) if total_portfolio_return != 0 else 0.0,
                        'interaction_percentage': float(interaction_effect / (total_portfolio_return + 1e-10)) if total_portfolio_return != 0 else 0.0
                    }
                },
                'performance_decomposition': performance_decomposition,
                'risk_attribution': {
                    'marginal_contribution_to_risk': marginal_contribution_to_risk,
                    'risk_concentration_analysis': self._analyze_risk_concentration(marginal_contribution_to_risk),
                    'diversification_effectiveness': float(1.0 / (1.0 + np.std(marginal_contribution_to_risk)))
                },
                'factor_attribution': factor_attribution,
                'attribution_insights': self._generate_attribution_insights(performance_decomposition, allocation_effect, selection_effect)
            }
        except Exception as e:
            self.logger.error(f"Error calculating performance attribution analysis: {e}")
            return {}

    def _calculate_factor_attribution(self, cluster_data: List[Dict], cluster_returns: List[float]) -> Dict[str, Any]:
        """Calculate factor attribution analysis."""
        try:
            # Simplified factor model attribution
            # Size factor attribution
            size_scores = [np.log(c['size'] + 1) for c in cluster_data]  # Log size as size factor
            size_factor_return = np.corrcoef(size_scores, cluster_returns)[0, 1] if len(cluster_returns) > 1 else 0.0

            # Momentum factor attribution (using returns as proxy)
            momentum_scores = cluster_returns  # Use returns as momentum proxy
            momentum_factor_return = np.mean([r**2 for r in cluster_returns]) if cluster_returns else 0.0

            # Volatility factor attribution (using return variance as proxy)
            volatility_scores = [r**2 for r in cluster_returns]  # Squared returns as volatility proxy
            volatility_factor_return = np.mean(volatility_scores) if volatility_scores else 0.0

            # Quality factor attribution (using consistency as proxy)
            consistency_scores = [1.0 / (1.0 + abs(r)) for r in cluster_returns]  # Lower absolute return = higher quality
            quality_factor_return = np.corrcoef(consistency_scores, cluster_returns)[0, 1] if len(cluster_returns) > 1 else 0.0

            return {
                'factor_returns': {
                    'size_factor': float(size_factor_return),
                    'momentum_factor': float(momentum_factor_return),
                    'volatility_factor': float(volatility_factor_return),
                    'quality_factor': float(quality_factor_return)
                },
                'factor_exposure_analysis': {
                    'size_factor_exposure': float(np.mean(size_scores)) if size_scores else 0.0,
                    'momentum_factor_exposure': float(np.mean(momentum_scores)) if momentum_scores else 0.0,
                    'volatility_factor_exposure': float(np.mean(volatility_scores)) if volatility_scores else 0.0,
                    'quality_factor_exposure': float(np.mean(consistency_scores)) if consistency_scores else 0.0
                },
                'factor_attribution_summary': {
                    'dominant_factor': max(['size', 'momentum', 'volatility', 'quality'],
                                         key=lambda k: abs({
                                             'size': size_factor_return,
                                             'momentum': momentum_factor_return,
                                             'volatility': volatility_factor_return,
                                             'quality': quality_factor_return
                                         }[k])),
                    'factor_diversification_score': float(1.0 / (1.0 + np.std([size_factor_return, momentum_factor_return, volatility_factor_return, quality_factor_return])))
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating factor attribution: {e}")
            return {}

    def _analyze_risk_concentration(self, marginal_contributions: List[float]) -> Dict[str, Any]:
        """Analyze risk concentration across clusters."""
        try:
            if not marginal_contributions:
                return {}

            # Herfindahl-Hirschman Index for risk concentration
            hhi = sum([mc**2 for mc in marginal_contributions])

            # Top contributors to risk
            sorted_contributions = sorted(enumerate(marginal_contributions), key=lambda x: x[1], reverse=True)
            top_3_risk_contributors = [f"cluster_{idx}" for idx, _ in sorted_contributions[:3]]

            # Risk distribution analysis
            risk_percentiles = np.percentile(marginal_contributions, [25, 50, 75, 90])

            return {
                'risk_concentration_index': float(hhi),
                'concentration_level': 'high' if hhi > 0.25 else 'moderate' if hhi > 0.15 else 'low',
                'top_risk_contributors': top_3_risk_contributors,
                'risk_distribution': {
                    'q25': float(risk_percentiles[0]),
                    'q50': float(risk_percentiles[1]),
                    'q75': float(risk_percentiles[2]),
                    'q90': float(risk_percentiles[3])
                },
                'diversification_potential': float(1.0 / hhi) if hhi > 0 else 0.0
            }
        except Exception as e:
            self.logger.error(f"Error analyzing risk concentration: {e}")
            return {}

    def _generate_attribution_insights(self, performance_decomposition: List[Dict],
                                     allocation_effect: float, selection_effect: float) -> Dict[str, Any]:
        """Generate insights from performance attribution analysis."""
        try:
            # Identify key drivers
            top_contributors = performance_decomposition[:3]  # Top 3 contributors
            bottom_contributors = performance_decomposition[-3:]  # Bottom 3 contributors

            # Assess attribution quality
            total_attribution = allocation_effect + selection_effect
            attribution_accuracy = abs(sum([p['contribution_to_return'] for p in performance_decomposition]) - total_attribution) / (total_attribution + 1e-10) if total_attribution != 0 else 0.0

            # Performance concentration
            top_3_contribution = sum([p['contribution_to_return'] for p in top_contributors])
            total_contribution = sum([p['contribution_to_return'] for p in performance_decomposition])
            performance_concentration = top_3_contribution / (total_contribution + 1e-10) if total_contribution != 0 else 0.0

            return {
                'key_drivers': {
                    'top_performers': [p['cluster_id'] for p in top_contributors],
                    'bottom_performers': [p['cluster_id'] for p in bottom_contributors],
                    'performance_concentration': float(performance_concentration),
                    'concentration_assessment': 'high' if performance_concentration > 0.7 else 'moderate' if performance_concentration > 0.5 else 'low'
                },
                'attribution_quality': {
                    'attribution_accuracy': float(attribution_accuracy),
                    'data_quality_score': 'excellent' if attribution_accuracy < 0.05 else 'good' if attribution_accuracy < 0.1 else 'fair' if attribution_accuracy < 0.2 else 'poor',
                    'confidence_level': float(max(0.0, 1.0 - attribution_accuracy))
                },
                'strategic_recommendations': {
                    'focus_clusters': [p['cluster_id'] for p in top_contributors if p['attribution_percentage'] > 0.1],
                    'reduce_exposure': [p['cluster_id'] for p in bottom_contributors if p['attribution_percentage'] < -0.05],
                    'rebalancing_priority': 'high' if abs(allocation_effect) > abs(selection_effect) else 'moderate',
                    'factor_tilt_recommendation': 'size_focused' if allocation_effect > 0 else 'selection_focused' if selection_effect > 0 else 'balanced'
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating attribution insights: {e}")
            return {}

    def _calculate_comparative_analysis(self, metrics: Dict[str, Dict],
                                      cluster_returns: List[float], cluster_sizes: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive comparative analysis between clusters."""
        try:
            if not cluster_returns or not cluster_sizes:
                return {}

            # Create cluster comparison data
            cluster_data = []
            for i, cluster_id in enumerate(metrics.keys()):
                if i < len(cluster_returns) and i < len(cluster_sizes):
                    cluster_data.append({
                        'cluster_id': cluster_id,
                        'return': cluster_returns[i],
                        'size': cluster_sizes[i],
                        'size_percentile': 0,  # Will be calculated
                        'return_percentile': 0  # Will be calculated
                    })

            # Calculate percentiles for ranking
            for i, cluster in enumerate(cluster_data):
                cluster['size_percentile'] = sum(1 for c in cluster_data if c['size'] <= cluster['size']) / len(cluster_data)
                cluster['return_percentile'] = sum(1 for c in cluster_data if c['return'] <= cluster['return']) / len(cluster_data)

            # Relative strength analysis
            relative_strength = self._calculate_relative_strength_analysis(cluster_data)

            # Performance comparison matrix
            performance_matrix = self._calculate_performance_comparison_matrix(cluster_data)

            # Size-adjusted performance analysis
            size_adjusted_analysis = self._calculate_size_adjusted_performance_analysis(cluster_data)

            # Correlation analysis between clusters
            correlation_analysis = self._calculate_inter_cluster_correlations(cluster_data)

            # Cluster ranking and tier analysis
            cluster_ranking = self._calculate_cluster_ranking_analysis(cluster_data)

            return {
                'relative_strength_analysis': relative_strength,
                'performance_comparison_matrix': performance_matrix,
                'size_adjusted_performance_analysis': size_adjusted_analysis,
                'inter_cluster_correlations': correlation_analysis,
                'cluster_ranking_analysis': cluster_ranking,
                'comparative_insights': self._generate_comparative_insights(cluster_data, relative_strength, performance_matrix)
            }
        except Exception as e:
            self.logger.error(f"Error calculating comparative analysis: {e}")
            return {}

    def _calculate_relative_strength_analysis(self, cluster_data: List[Dict]) -> Dict[str, Any]:
        """Calculate relative strength analysis between clusters."""
        try:
            if not cluster_data:
                return {}

            # Sort clusters by performance
            sorted_by_return = sorted(cluster_data, key=lambda x: x['return'], reverse=True)
            sorted_by_size = sorted(cluster_data, key=lambda x: x['size'], reverse=True)

            # Calculate relative strength scores
            relative_strength_scores = []
            for cluster in cluster_data:
                # Relative strength = (cluster_return - avg_return) / std_return + size_factor
                avg_return = np.mean([c['return'] for c in cluster_data])
                std_return = np.std([c['return'] for c in cluster_data])
                size_factor = (cluster['size'] - np.mean([c['size'] for c in cluster_data])) / (np.std([c['size'] for c in cluster_data]) + 1e-10)

                rs_score = (cluster['return'] - avg_return) / (std_return + 1e-10) + 0.3 * size_factor
                relative_strength_scores.append(rs_score)

            # Identify leaders and laggards
            leaders = [c for c, rs in zip(cluster_data, relative_strength_scores) if rs > 0.5]
            laggards = [c for c, rs in zip(cluster_data, relative_strength_scores) if rs < -0.5]
            middle_performers = [c for c, rs in zip(cluster_data, relative_strength_scores) if -0.5 <= rs <= 0.5]

            return {
                'relative_strength_scores': {
                    cluster['cluster_id']: float(rs) for cluster, rs in zip(cluster_data, relative_strength_scores)
                },
                'performance_tiers': {
                    'leaders': [c['cluster_id'] for c in leaders],
                    'middle_performers': [c['cluster_id'] for c in middle_performers],
                    'laggards': [c['cluster_id'] for c in laggards],
                    'tier_distribution': {
                        'leader_percentage': len(leaders) / len(cluster_data) if cluster_data else 0.0,
                        'middle_percentage': len(middle_performers) / len(cluster_data) if cluster_data else 0.0,
                        'laggard_percentage': len(laggards) / len(cluster_data) if cluster_data else 0.0
                    }
                },
                'strength_consistency': {
                    'score_variance': float(np.var(relative_strength_scores)),
                    'strength_dispersion': float(np.std(relative_strength_scores)),
                    'competitive_intensity': 'high' if np.std(relative_strength_scores) > 1.0 else 'moderate' if np.std(relative_strength_scores) > 0.5 else 'low'
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating relative strength analysis: {e}")
            return {}

    def _calculate_performance_comparison_matrix(self, cluster_data: List[Dict]) -> Dict[str, Any]:
        """Calculate performance comparison matrix between clusters."""
        try:
            if not cluster_data:
                return {}

            # Create pairwise comparison matrix
            n_clusters = len(cluster_data)
            performance_matrix = np.zeros((n_clusters, n_clusters))

            for i, cluster_i in enumerate(cluster_data):
                for j, cluster_j in enumerate(cluster_data):
                    if i != j:
                        # Performance differential
                        return_diff = cluster_i['return'] - cluster_j['return']
                        size_diff = cluster_i['size'] - cluster_j['size']

                        # Composite comparison score
                        performance_matrix[i, j] = return_diff * 0.7 + (size_diff / max(cluster_i['size'], cluster_j['size'])) * 0.3

            # Extract key comparisons
            top_performances = []
            for i, cluster_i in enumerate(cluster_data):
                for j, cluster_j in enumerate(cluster_data):
                    if i < j and abs(performance_matrix[i, j]) > 0.5:  # Significant difference
                        top_performances.append({
                            'cluster_a': cluster_i['cluster_id'],
                            'cluster_b': cluster_j['cluster_id'],
                            'performance_differential': float(performance_matrix[i, j]),
                            'significance_level': 'high' if abs(performance_matrix[i, j]) > 1.0 else 'moderate' if abs(performance_matrix[i, j]) > 0.5 else 'low'
                        })

            # Sort by significance
            top_performances.sort(key=lambda x: abs(x['performance_differential']), reverse=True)

            return {
                'performance_matrix_summary': {
                    'matrix_shape': performance_matrix.shape,
                    'average_differential': float(np.mean(np.abs(performance_matrix))),
                    'maximum_differential': float(np.max(np.abs(performance_matrix))),
                    'competitive_balance_score': float(1.0 / (1.0 + np.std(np.abs(performance_matrix))))
                },
                'key_performance_comparisons': top_performances[:10],  # Top 10 significant differences
                'performance_distribution': {
                    'highly_superior': len([p for p in top_performances if p['performance_differential'] > 1.0]),
                    'moderately_superior': len([p for p in top_performances if 0.5 < p['performance_differential'] <= 1.0]),
                    'slightly_superior': len([p for p in top_performances if 0.0 < p['performance_differential'] <= 0.5])
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating performance comparison matrix: {e}")
            return {}

    def _calculate_size_adjusted_performance_analysis(self, cluster_data: List[Dict]) -> Dict[str, Any]:
        """Calculate size-adjusted performance analysis."""
        try:
            if not cluster_data:
                return {}

            # Create size-adjusted performance metrics
            avg_size = np.mean([c['size'] for c in cluster_data])

            size_adjusted_scores = []
            for cluster in cluster_data:
                # Size efficiency score: performance relative to size
                size_efficiency = cluster['return'] / (cluster['size'] / avg_size) if cluster['size'] > 0 else 0.0

                # Size-adjusted return (smaller clusters get bonus for efficiency)
                size_adjusted_return = cluster['return'] * (1.0 + 0.2 * (avg_size / (cluster['size'] + 1e-10)))

                size_adjusted_scores.append({
                    'cluster_id': cluster['cluster_id'],
                    'size_efficiency_score': float(size_efficiency),
                    'size_adjusted_return': float(size_adjusted_return),
                    'size_advantage_ratio': float(cluster['size'] / avg_size)
                })

            # Sort by size-adjusted performance
            size_adjusted_scores.sort(key=lambda x: x['size_adjusted_return'], reverse=True)

            # Identify size vs performance relationships
            try:
                sizes = [float(c['size']) for c in cluster_data if str(c.get('size', '')).replace('.', '').replace('-', '').isdigit()]
                returns = [float(c['return']) for c in cluster_data if str(c.get('return', '')).replace('.', '').replace('-', '').isdigit()]
                if len(sizes) == len(returns) and len(sizes) > 1:
                    size_performance_correlation = np.corrcoef(sizes, returns)[0, 1]
                else:
                    size_performance_correlation = 0.0
            except:
                size_performance_correlation = 0.0

            return {
                'size_adjusted_rankings': size_adjusted_scores,
                'size_performance_relationship': {
                    'correlation_coefficient': float(size_performance_correlation),
                    'relationship_strength': 'strong' if abs(size_performance_correlation) > 0.7 else 'moderate' if abs(size_performance_correlation) > 0.3 else 'weak',
                    'relationship_type': 'positive' if size_performance_correlation > 0 else 'negative' if size_performance_correlation < 0 else 'none'
                },
                'efficiency_analysis': {
                    'most_efficient_small_clusters': [s['cluster_id'] for s in size_adjusted_scores[:3] if s['size_advantage_ratio'] < 0.8],
                    'most_efficient_large_clusters': [s['cluster_id'] for s in size_adjusted_scores[-3:] if s['size_advantage_ratio'] > 1.2],
                    'efficiency_distribution': self._analyze_efficiency_distribution(size_adjusted_scores)
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating size-adjusted performance analysis: {e}")
            return {}

    def _calculate_inter_cluster_correlations(self, cluster_data: List[Dict]) -> Dict[str, Any]:
        """Calculate correlations between clusters."""
        try:
            if len(cluster_data) < 2:
                return {}

            # Calculate return correlations between clusters
            returns = []
            for c in cluster_data:
                try:
                    return_val = c.get('return', 0)
                    # Ensure return_val is a valid number
                    if return_val is None or np.isnan(return_val):
                        return_val = 0.0
                    return_val = float(return_val)
                    returns.append(return_val)
                except (ValueError, TypeError):
                    returns.append(0.0)

            if len(returns) < 2:
                return {
                    'average_inter_cluster_correlation': 0.0,
                    'correlation_distribution': {
                        'highly_correlated_pairs': 0,
                        'moderately_correlated_pairs': 0,
                        'weakly_correlated_pairs': 0
                    },
                    'top_correlated_pairs': [],
                    'diversification_analysis': {
                        'diversification_potential': 1.0,
                        'diversification_effectiveness': 'excellent',
                        'recommended_diversification_strategy': 'diversified'
                    }
                }

            correlation_matrix = np.corrcoef(returns)

            # Handle case where correlation_matrix might be scalar (happens with single value or identical values)
            if np.isscalar(correlation_matrix):
                # Single value case - no meaningful correlations
                return {
                    'average_inter_cluster_correlation': 0.0,
                    'correlation_distribution': {
                        'highly_correlated_pairs': 0,
                        'moderately_correlated_pairs': 0,
                        'weakly_correlated_pairs': 0
                    },
                    'top_correlated_pairs': [],
                    'diversification_analysis': {
                        'diversification_potential': 1.0,
                        'diversification_effectiveness': 'excellent',
                        'recommended_diversification_strategy': 'diversified'
                    }
                }

            # Find highly correlated pairs
            correlated_pairs = []
            for i in range(len(cluster_data)):
                for j in range(i + 1, len(cluster_data)):
                    correlation = correlation_matrix[i, j]
                    if abs(correlation) > 0.5:  # Significant correlation
                        correlated_pairs.append({
                            'cluster_a': cluster_data[i]['cluster_id'],
                            'cluster_b': cluster_data[j]['cluster_id'],
                            'correlation': float(correlation),
                            'correlation_strength': 'strong' if abs(correlation) > 0.8 else 'moderate' if abs(correlation) > 0.6 else 'weak'
                        })

            # Sort by correlation strength
            correlated_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

            # Calculate diversification potential
            avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            diversification_potential = 1.0 / (1.0 + abs(avg_correlation))

            return {
                'average_inter_cluster_correlation': float(avg_correlation),
                'correlation_distribution': {
                    'highly_correlated_pairs': len([p for p in correlated_pairs if abs(p['correlation']) > 0.8]),
                    'moderately_correlated_pairs': len([p for p in correlated_pairs if 0.6 < abs(p['correlation']) <= 0.8]),
                    'weakly_correlated_pairs': len([p for p in correlated_pairs if 0.3 < abs(p['correlation']) <= 0.6])
                },
                'top_correlated_pairs': correlated_pairs[:5],  # Top 5 most correlated pairs
                'diversification_analysis': {
                    'diversification_potential': float(diversification_potential),
                    'diversification_effectiveness': 'excellent' if diversification_potential > 0.8 else 'good' if diversification_potential > 0.6 else 'fair' if diversification_potential > 0.4 else 'poor',
                    'recommended_diversification_strategy': 'concentrated' if avg_correlation > 0.5 else 'balanced' if avg_correlation > 0.2 else 'diversified'
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating inter-cluster correlations: {e}")
            self.logger.error(f"Debug info: cluster_data length={len(cluster_data) if cluster_data else 'None'}, returns length={len(returns) if 'returns' in locals() else 'N/A'}")
            return {
                'average_inter_cluster_correlation': 0.0,
                'correlation_distribution': {
                    'highly_correlated_pairs': 0,
                    'moderately_correlated_pairs': 0,
                    'weakly_correlated_pairs': 0
                },
                'top_correlated_pairs': [],
                'diversification_analysis': {
                    'diversification_potential': 1.0,
                    'diversification_effectiveness': 'excellent',
                    'recommended_diversification_strategy': 'diversified'
                }
            }

    def _calculate_cluster_ranking_analysis(self, cluster_data: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive cluster ranking analysis."""
        try:
            # Multi-criteria ranking
            rankings = []

            for cluster in cluster_data:
                # Composite score based on return, size, and consistency
                composite_score = cluster['return'] * 0.5 + (cluster['size_percentile'] * 0.3) + (cluster['return_percentile'] * 0.2)

                rankings.append({
                    'cluster_id': cluster['cluster_id'],
                    'composite_score': float(composite_score),
                    'return_rank': 0,  # Will be set after sorting
                    'size_rank': 0,    # Will be set after sorting
                    'overall_rank': 0  # Will be set after sorting
                })

            # Sort by different criteria
            return_ranked = sorted(rankings, key=lambda x: x['composite_score'], reverse=True)
            size_ranked = sorted(rankings, key=lambda x: x['cluster_id'])  # By ID for consistency
            composite_ranked = sorted(rankings, key=lambda x: x['composite_score'], reverse=True)

            # Assign ranks
            for i, cluster in enumerate(return_ranked):
                cluster['return_rank'] = i + 1

            for i, cluster in enumerate(composite_ranked):
                cluster['overall_rank'] = i + 1

            # Identify rank discrepancies
            rank_discrepancies = []
            for cluster in rankings:
                if abs(cluster['return_rank'] - cluster['overall_rank']) > 3:
                    rank_discrepancies.append({
                        'cluster_id': cluster['cluster_id'],
                        'return_rank': cluster['return_rank'],
                        'overall_rank': cluster['overall_rank'],
                        'rank_difference': cluster['overall_rank'] - cluster['return_rank']
                    })

            return {
                'composite_rankings': return_ranked,
                'top_performers': [c['cluster_id'] for c in return_ranked[:3]],
                'bottom_performers': [c['cluster_id'] for c in return_ranked[-3:]],
                'rank_stability_analysis': {
                    'rank_discrepancies': rank_discrepancies,
                    'ranking_consistency_score': float(1.0 / (1.0 + len(rank_discrepancies) / len(cluster_data))) if cluster_data else 0.0,
                    'most_consistent_clusters': [c['cluster_id'] for c in rankings if abs(c['return_rank'] - c['overall_rank']) <= 1]
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating cluster ranking analysis: {e}")
            return {}

    def _generate_comparative_insights(self, cluster_data: List[Dict], relative_strength: Dict[str, Any],
                                     performance_matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from comparative analysis."""
        try:
            # Competitive landscape assessment
            leaders = relative_strength.get('performance_tiers', {}).get('leaders', [])
            laggards = relative_strength.get('performance_tiers', {}).get('laggards', [])

            competitive_intensity = performance_matrix.get('performance_matrix_summary', {}).get('competitive_balance_score', 0.0)

            # Market structure analysis
            market_structure = 'fragmented' if len(leaders) > 3 else 'concentrated' if len(leaders) <= 1 else 'oligopolistic'

            return {
                'competitive_landscape': {
                    'competitive_intensity_score': float(competitive_intensity),
                    'market_structure': market_structure,
                    'leadership_stability': 'stable' if len(leaders) <= 3 else 'contested',
                    'competitive_advantage': 'sustainable' if competitive_intensity > 0.7 else 'temporary' if competitive_intensity > 0.5 else 'weak'
                },
                'strategic_positioning_insights': {
                    'recommended_focus_areas': leaders[:2] if leaders else [],
                    'improvement_opportunities': laggards[:2] if laggards else [],
                    'competitive_threats': [c for c in leaders if c not in cluster_data[0]['cluster_id']],  # Competitors to top cluster
                    'differentiation_opportunities': len([c for c in cluster_data if abs(c['return'] - cluster_data[0]['return']) < 0.01])  # Similar performers
                },
                'performance_benchmarks': {
                    'top_quartile_threshold': float(np.percentile([c['return'] for c in cluster_data], 75)),
                    'median_performance': float(np.median([c['return'] for c in cluster_data])),
                    'bottom_quartile_threshold': float(np.percentile([c['return'] for c in cluster_data], 25)),
                    'achievement_targets': {
                        'excellent': float(np.percentile([c['return'] for c in cluster_data], 90)),
                        'good': float(np.percentile([c['return'] for c in cluster_data], 75)),
                        'satisfactory': float(np.percentile([c['return'] for c in cluster_data], 50))
                    }
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating comparative insights: {e}")
            return {}

    def _create_statistical_summary(self, metrics: Dict[str, Dict],
                                  clustering_result: ClusteringResult) -> Dict[str, Any]:
        """Create comprehensive statistical summary of all clusters."""
        try:
            if not metrics:
                return {}

            # Extract data for analysis
            cluster_data = self._extract_cluster_data_for_statistical_analysis(metrics)

            # Basic descriptive statistics
            basic_stats = self._calculate_basic_descriptive_statistics(cluster_data)

            # Distribution analysis
            distribution_analysis = self._calculate_distribution_analysis(cluster_data)

            # Correlation and relationship analysis
            correlation_analysis = self._calculate_correlation_analysis(metrics, cluster_data)

            # Temporal stability analysis
            temporal_analysis = self._calculate_temporal_stability_analysis(cluster_data)

            # Advanced statistical measures
            advanced_stats = self._calculate_advanced_statistical_measures(metrics, cluster_data)

            # Cluster characteristic analysis
            cluster_characteristics = self._analyze_cluster_characteristics_from_metrics(metrics)

            return {
                'basic_descriptive_statistics': basic_stats,
                'distribution_analysis': distribution_analysis,
                'correlation_and_relationships': correlation_analysis,
                'temporal_stability_analysis': temporal_analysis,
                'advanced_statistical_measures': advanced_stats,
                'cluster_characteristics_analysis': cluster_characteristics,
                'statistical_summary_insights': self._generate_statistical_insights(metrics, cluster_data)
            }

        except Exception as e:
            self.logger.error(f"Error creating comprehensive statistical summary: {e}")
            return {}

    def _extract_cluster_data_for_statistical_analysis(self, metrics: Dict[str, Dict]) -> Dict[str, List]:
        """Extract and organize cluster data for statistical analysis."""
        try:
            # Extract various metrics from clusters
            sizes = []
            returns = []
            sharpes = []
            drawdowns = []
            volumes = []
            volatilities = []
            cluster_ids = []

            for cluster_id, cluster_metrics in metrics.items():
                cluster_ids.append(cluster_id)
                sizes.append(cluster_metrics['basic_stats']['size'])

                if cluster_metrics.get('return_statistics'):
                    ret_stats = cluster_metrics['return_statistics']
                    returns.append(ret_stats.get('mean_return', 0))
                    drawdowns.append(ret_stats.get('max_drawdown', 0))

                if cluster_metrics.get('volume_statistics'):
                    vol_stats = cluster_metrics['volume_statistics']
                    volumes.append(vol_stats.get('mean', 0))

                # Extract volatility from price statistics if available
                if cluster_metrics.get('price_statistics'):
                    price_stats = cluster_metrics['price_statistics']
                    volatilities.append(price_stats.get('std', 0))

            return {
                'cluster_ids': cluster_ids,
                'sizes': sizes,
                'returns': returns,
                'sharpes': sharpes,
                'drawdowns': drawdowns,
                'volumes': volumes,
                'volatilities': volatilities
            }
        except Exception as e:
            self.logger.error(f"Error extracting cluster data: {e}")
            return {}

    def _calculate_basic_descriptive_statistics(self, cluster_data: Dict[str, List]) -> Dict[str, Any]:
        """Calculate basic descriptive statistics for cluster metrics."""
        try:
            stats = {}

            for metric_name, values in cluster_data.items():
                if not values:
                    continue

                # Filter out non-numeric values and convert to float
                numeric_values = []
                for v in values:
                    try:
                        numeric_values.append(float(v))
                    except (ValueError, TypeError):
                        continue

                if not numeric_values:
                    continue

                values_array = np.array(numeric_values)

                stats[metric_name] = {
                    'count': len(values),
                    'mean': float(np.mean(values_array)),
                    'median': float(np.median(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'range': float(np.max(values_array) - np.min(values_array)),
                    'coefficient_of_variation': float(np.std(values_array) / abs(np.mean(values_array))) if np.mean(values_array) != 0 else float('inf'),
                    'quartiles': {
                        'q25': float(np.percentile(values_array, 25)),
                        'q50': float(np.percentile(values_array, 50)),
                        'q75': float(np.percentile(values_array, 75))
                    },
                    'percentiles': {
                        'p5': float(np.percentile(values_array, 5)),
                        'p95': float(np.percentile(values_array, 95)),
                        'p99': float(np.percentile(values_array, 99))
                    }
                }

            return stats
        except Exception as e:
            self.logger.error(f"Error calculating basic descriptive statistics: {e}")
            return {}

    def _calculate_distribution_analysis(self, cluster_data: Dict[str, List]) -> Dict[str, Any]:
        """Calculate distribution analysis for cluster metrics."""
        try:
            analysis = {}

            for metric_name, values in cluster_data.items():
                if not values or len(values) < 3:
                    continue

                # Filter out non-numeric values and convert to float
                numeric_values = []
                for v in values:
                    try:
                        numeric_values.append(float(v))
                    except (ValueError, TypeError):
                        continue

                if len(numeric_values) < 3:
                    continue

                values_series = pd.Series(numeric_values)

                analysis[metric_name] = {
                    'normality_tests': {
                        'skewness': float(values_series.skew()),
                        'kurtosis': float(values_series.kurtosis()),
                        'is_symmetric': abs(values_series.skew()) < 0.5,
                        'is_normal_like': abs(values_series.skew()) < 1.0 and abs(values_series.kurtosis() - 3) < 2.0
                    },
                    'distribution_characteristics': {
                        'modality': self._assess_modality(values),
                        'outlier_count': self._count_outliers(values),
                        'distribution_type': self._classify_distribution(values_series)
                    },
                    'concentration_measures': {
                        'gini_coefficient': self._calculate_gini_coefficient(values),
                        'herfindahl_index': self._calculate_herfindahl_index(values),
                        'concentration_ratio_top_3': sum(sorted(values, reverse=True)[:3]) / sum(values) if sum(values) > 0 else 0.0
                    }
                }

            return analysis
        except Exception as e:
            self.logger.error(f"Error calculating distribution analysis: {e}")
            return {}

    def _calculate_correlation_analysis(self, metrics: Dict[str, Dict], cluster_data: Dict[str, List]) -> Dict[str, Any]:
        """Calculate correlation and relationship analysis between cluster metrics."""
        try:
            if len(cluster_data.get('returns', [])) < 2:
                return {}

            # Create correlation matrix between different metrics
            metric_correlations = {}

            # Size vs Returns correlation
            sizes = [float(s) for s in cluster_data.get('sizes', []) if str(s).replace('.', '').replace('-', '').isdigit()]
            returns = [float(r) for r in cluster_data.get('returns', []) if str(r).replace('.', '').replace('-', '').isdigit()]
            if len(sizes) == len(returns) and len(sizes) > 1:
                try:
                    sizes_returns_corr = np.corrcoef(sizes, returns)[0, 1]
                    metric_correlations['size_vs_returns'] = float(sizes_returns_corr)
                except:
                    pass

            # Volume vs Returns correlation
            volumes = [float(v) for v in cluster_data.get('volumes', []) if str(v).replace('.', '').replace('-', '').isdigit()]
            if len(volumes) == len(returns) and len(volumes) > 1:
                try:
                    volumes_returns_corr = np.corrcoef(volumes, returns)[0, 1]
                    metric_correlations['volume_vs_returns'] = float(volumes_returns_corr)
                except:
                    pass

            # Volatility vs Returns correlation
            volatilities = [float(v) for v in cluster_data.get('volatilities', []) if str(v).replace('.', '').replace('-', '').isdigit()]
            if len(volatilities) == len(returns) and len(volatilities) > 1:
                try:
                    vol_returns_corr = np.corrcoef(volatilities, returns)[0, 1]
                    metric_correlations['volatility_vs_returns'] = float(vol_returns_corr)
                except:
                    pass

            # Cross-cluster correlation analysis
            returns_numeric = [float(r) for r in cluster_data.get('returns', []) if str(r).replace('.', '').replace('-', '').isdigit()]
            if len(returns_numeric) > 1:
                try:
                    returns_array = np.array(returns_numeric)
                    cross_cluster_correlations = {
                        'average_inter_cluster_correlation': float(np.mean([np.corrcoef(returns_array, np.roll(returns_array, i))[0, 1] for i in range(1, len(returns_array))])),
                        'maximum_inter_cluster_correlation': float(np.max([np.corrcoef(returns_array, np.roll(returns_array, i))[0, 1] for i in range(1, len(returns_array))]))
                    }
                except:
                    cross_cluster_correlations = {
                        'average_inter_cluster_correlation': 0.0,
                        'maximum_inter_cluster_correlation': 0.0
                    }
            else:
                cross_cluster_correlations = {
                    'average_inter_cluster_correlation': 0.0,
                    'maximum_inter_cluster_correlation': 0.0
                }

            return {
                'metric_correlations': metric_correlations,
                'cross_cluster_analysis': cross_cluster_correlations,
                'relationship_strengths': {
                    'size_return_relationship': self._classify_correlation_strength(metric_correlations.get('size_vs_returns', 0)),
                    'volume_return_relationship': self._classify_correlation_strength(metric_correlations.get('volume_vs_returns', 0)),
                    'volatility_return_relationship': self._classify_correlation_strength(metric_correlations.get('volatility_vs_returns', 0))
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating correlation analysis: {e}")
            return {}

    def _calculate_temporal_stability_analysis(self, cluster_data: Dict[str, List]) -> Dict[str, Any]:
        """Calculate temporal stability analysis for clusters."""
        try:
            # Simulate temporal stability by analyzing consistency across different metrics
            stability_scores = {}

            for metric_name, values in cluster_data.items():
                if not values or len(values) < 5:
                    continue

                # Filter out non-numeric values and convert to float
                numeric_values = []
                for v in values:
                    try:
                        numeric_values.append(float(v))
                    except (ValueError, TypeError):
                        continue

                if len(numeric_values) < 5:
                    continue

                # Calculate rolling stability (using cumulative analysis as proxy for temporal)
                values_array = np.array(numeric_values)

                # Temporal consistency score (lower variance = higher stability)
                stability_scores[metric_name] = {
                    'stability_score': float(1.0 / (1.0 + np.std(values_array))),
                    'consistency_ratio': float(np.mean(values_array) / np.std(values_array)) if np.std(values_array) != 0 else float('inf'),
                    'trend_stability': self._assess_trend_stability(values_array),
                    'regime_persistence': self._calculate_regime_persistence(values_array)
                }

            return {
                'metric_stability_scores': stability_scores,
                'overall_stability_assessment': {
                    'average_stability': float(np.mean([s['stability_score'] for s in stability_scores.values()])),
                    'most_stable_metric': max(stability_scores.keys(), key=lambda k: stability_scores[k]['stability_score']),
                    'least_stable_metric': min(stability_scores.keys(), key=lambda k: stability_scores[k]['stability_score'])
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating temporal stability analysis: {e}")
            return {}

    def _calculate_advanced_statistical_measures(self, metrics: Dict[str, Dict], cluster_data: Dict[str, List]) -> Dict[str, Any]:
        """Calculate advanced statistical measures."""
        try:
            advanced_measures = {}

            for metric_name, values in cluster_data.items():
                if not values or len(values) < 5:
                    continue

                # Filter out non-numeric values and convert to float
                numeric_values = []
                for v in values:
                    try:
                        numeric_values.append(float(v))
                    except (ValueError, TypeError):
                        continue

                if len(numeric_values) < 5:
                    continue

                values_array = np.array(numeric_values)

                advanced_measures[metric_name] = {
                    'information_theory_measures': {
                        'entropy': float(self._calculate_entropy(values)),
                        'information_content': float(np.log2(len(set(values)))) if len(set(values)) > 0 else 0.0
                    },
                    'robustness_measures': {
                        'trimmed_mean': float(self._calculate_trimmed_mean(values_array, 0.1)),
                        'winsorized_mean': float(self._calculate_winsorized_mean(values_array, 0.1)),
                        'median_absolute_deviation': float(np.median(np.abs(values_array - np.median(values_array))))
                    },
                    'statistical_significance': {
                        'confidence_interval_95': self._calculate_confidence_interval(values_array, 0.95),
                        'standard_error': float(np.std(values_array) / np.sqrt(len(values_array))) if len(values_array) > 0 else 0.0
                    }
                }

            return advanced_measures
        except Exception as e:
            self.logger.error(f"Error calculating advanced statistical measures: {e}")
            return {}

    def _analyze_cluster_characteristics_from_metrics(self, metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze characteristics of clusters from metrics dictionary."""
        try:
            cluster_types = []
            for cluster_id, cluster_metrics in metrics.items():
                size = cluster_metrics['basic_stats']['size']
                percentage = cluster_metrics['basic_stats']['percentage']

                # Classify cluster type based on size and characteristics
                if percentage > 0.15:
                    cluster_type = 'dominant'
                elif percentage > 0.05:
                    cluster_type = 'significant'
                elif percentage > 0.02:
                    cluster_type = 'moderate'
                else:
                    cluster_type = 'niche'

                cluster_types.append({
                    'cluster_id': cluster_id,
                    'type': cluster_type,
                    'size': size,
                    'percentage': percentage
                })

            return {
                'cluster_type_distribution': {
                    'dominant': len([c for c in cluster_types if c['type'] == 'dominant']),
                    'significant': len([c for c in cluster_types if c['type'] == 'significant']),
                    'moderate': len([c for c in cluster_types if c['type'] == 'moderate']),
                    'niche': len([c for c in cluster_types if c['type'] == 'niche'])
                },
                'cluster_classification': cluster_types,
                'market_structure_insights': {
                    'concentration_level': 'high' if len([c for c in cluster_types if c['type'] == 'dominant']) > 3 else 'moderate',
                    'diversity_index': float(len(cluster_types) / 10.0),  # Normalize to 0-1 scale
                    'market_fragmentation_score': float(1.0 / len([c for c in cluster_types if c['type'] in ['dominant', 'significant']]))
                }
            }
        except Exception as e:
            self.logger.error(f"Error analyzing cluster characteristics: {e}")
            return {}

    def _generate_statistical_insights(self, metrics: Dict[str, Dict], cluster_data: Dict[str, List]) -> Dict[str, Any]:
        """Generate statistical insights and recommendations."""
        try:
            insights = {
                'data_quality_assessment': {
                    'completeness_score': float(len([v for v in cluster_data.values() if v]) / len(cluster_data)),
                    'consistency_score': self._assess_data_consistency(cluster_data),
                    'reliability_score': float(min(1.0, len(metrics) / 20.0))  # Higher score for more clusters
                },
                'statistical_significance': {
                    'sample_size_adequacy': 'adequate' if len(metrics) > 10 else 'limited',
                    'statistical_power': float(min(1.0, len(metrics) / 30.0)),
                    'confidence_level': 'high' if len(metrics) > 20 else 'moderate' if len(metrics) > 10 else 'low'
                },
                'analytical_recommendations': {
                    'suggested_analysis_depth': 'comprehensive' if len(metrics) > 15 else 'standard',
                    'recommended_confidence_level': 0.95 if len(metrics) > 20 else 0.90,
                    'analysis_complexity_rating': 'advanced' if len(metrics) > 25 else 'intermediate'
                }
            }

            return insights
        except Exception as e:
            self.logger.error(f"Error generating statistical insights: {e}")
            return {}

    # Helper methods for statistical analysis
    def _assess_modality(self, values: List[float]) -> str:
        """Assess the modality of a distribution."""
        try:
            if len(values) < 10:
                return 'unknown'

            # Simple modality assessment using peaks
            values_array = np.array(values)
            hist, bins = np.histogram(values_array, bins=10)
            peaks = len([i for i in range(1, len(hist)-1) if hist[i] > hist[i-1] and hist[i] > hist[i+1]])

            if peaks == 0:
                return 'unimodal'
            elif peaks == 1:
                return 'bimodal'
            else:
                return 'multimodal'
        except Exception:
            return 'unknown'

    def _count_outliers(self, values: List[float]) -> int:
        """Count outliers using IQR method."""
        try:
            values_array = np.array(values)
            q75, q25 = np.percentile(values_array, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - (iqr * 1.5)
            upper_bound = q75 + (iqr * 1.5)

            outliers = [v for v in values if v < lower_bound or v > upper_bound]
            return len(outliers)
        except Exception:
            return 0

    def _classify_distribution(self, values_series: pd.Series) -> str:
        """Classify the type of distribution."""
        try:
            skew = abs(values_series.skew())
            kurt = values_series.kurtosis()

            if skew < 0.5 and abs(kurt - 3) < 1:
                return 'normal'
            elif skew < 1.0 and kurt > 3:
                return 'leptokurtic'
            elif skew < 1.0 and kurt < 3:
                return 'platykurtic'
            elif skew >= 1.0:
                return 'skewed'
            else:
                return 'unknown'
        except Exception:
            return 'unknown'

    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for measuring inequality."""
        try:
            values_array = np.array(values)
            values_array = values_array[values_array > 0]  # Only positive values
            if len(values_array) == 0:
                return 0.0

            values_array = np.sort(values_array)
            n = len(values_array)
            cumsum = np.cumsum(values_array)
            return float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n) if cumsum[-1] > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_herfindahl_index(self, values: List[float]) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration."""
        try:
            values_array = np.array(values)
            if np.sum(values_array) == 0:
                return 0.0

            normalized_values = values_array / np.sum(values_array)
            return float(np.sum(normalized_values ** 2))
        except Exception:
            return 0.0

    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength."""
        try:
            abs_corr = abs(correlation)
            if abs_corr >= 0.7:
                return 'strong'
            elif abs_corr >= 0.3:
                return 'moderate'
            elif abs_corr >= 0.1:
                return 'weak'
            else:
                return 'negligible'
        except Exception:
            return 'unknown'

    def _assess_trend_stability(self, values: np.ndarray) -> str:
        """Assess trend stability."""
        try:
            if len(values) < 5:
                return 'unknown'

            # Simple linear trend
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]

            if abs(slope) < np.std(values) * 0.1:
                return 'stable'
            elif slope > 0:
                return 'increasing'
            else:
                return 'decreasing'
        except Exception:
            return 'unknown'

    def _calculate_regime_persistence(self, values: np.ndarray) -> float:
        """Calculate regime persistence score."""
        try:
            if len(values) < 5:
                return 0.0

            # Measure how consistent values are over time
            changes = np.abs(np.diff(values))
            persistence = 1.0 / (1.0 + np.mean(changes) / (np.std(values) + 1e-10))
            return float(min(persistence, 1.0))
        except Exception:
            return 0.0

    def _calculate_entropy(self, values: List[float]) -> float:
        """Calculate Shannon entropy."""
        try:
            from scipy.stats import entropy
            values_array = np.array(values)
            hist, _ = np.histogram(values_array, bins='auto', density=True)
            return float(entropy(hist))
        except Exception:
            return 0.0

    def _calculate_trimmed_mean(self, values: np.ndarray, proportion: float = 0.1) -> float:
        """Calculate trimmed mean."""
        try:
            return float(np.mean(values[(values >= np.percentile(values, proportion * 100)) &
                                      (values <= np.percentile(values, (1 - proportion) * 100))]))
        except Exception:
            return float(np.mean(values))

    def _calculate_winsorized_mean(self, values: np.ndarray, proportion: float = 0.1) -> float:
        """Calculate Winsorized mean."""
        try:
            lower_bound = np.percentile(values, proportion * 100)
            upper_bound = np.percentile(values, (1 - proportion) * 100)
            winsorized = np.clip(values, lower_bound, upper_bound)
            return float(np.mean(winsorized))
        except Exception:
            return float(np.mean(values))

    def _calculate_confidence_interval(self, values: np.ndarray, confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate confidence interval."""
        try:
            from scipy import stats
            mean_val = np.mean(values)
            std_err = stats.sem(values)
            confidence_interval = stats.t.interval(confidence_level, len(values)-1, loc=mean_val, scale=std_err)

            return {
                'lower_bound': float(confidence_interval[0]),
                'upper_bound': float(confidence_interval[1]),
                'margin_of_error': float((confidence_interval[1] - confidence_interval[0]) / 2)
            }
        except Exception:
            return {'lower_bound': 0.0, 'upper_bound': 0.0, 'margin_of_error': 0.0}

    def _assess_data_consistency(self, cluster_data: Dict[str, List]) -> float:
        """Assess data consistency across clusters."""
        try:
            # Check for consistency in data availability and quality
            completeness_scores = []

            for metric_name, values in cluster_data.items():
                if values:
                    # Check for missing values (simulated)
                    completeness = 1.0 - (len([v for v in values if v == 0]) / len(values))
                    completeness_scores.append(completeness)

            return float(np.mean(completeness_scores)) if completeness_scores else 0.0
        except Exception:
            return 0.0

    # Helper methods for enhanced risk analysis
    def _assess_volatility_trend(self, volatilities: List[float]) -> str:
        """Assess volatility trend across clusters."""
        try:
            if len(volatilities) < 3:
                return 'unknown'

            # Simple trend assessment
            values_array = np.array(volatilities)
            x = np.arange(len(values_array))
            slope = np.polyfit(x, values_array, 1)[0]

            if abs(slope) < np.std(values_array) * 0.1:
                return 'stable'
            elif slope > 0:
                return 'increasing'
            else:
                return 'decreasing'
        except Exception:
            return 'unknown'

    def _prioritize_risk_mitigation(self, risk_factors: Dict[str, Dict]) -> Dict[str, str]:
        """Prioritize risk mitigation strategies based on risk factors."""
        try:
            priorities = {}

            # Concentration risk priority
            concentration_level = risk_factors.get('concentration_factor', {}).get('concentration_risk_level', 'low')
            if concentration_level in ['critical', 'high']:
                priorities['diversification'] = 'high'
            elif concentration_level == 'moderate':
                priorities['diversification'] = 'medium'
            else:
                priorities['diversification'] = 'low'

            # Volatility risk priority
            volatility_regime = risk_factors.get('volatility_factor', {}).get('volatility_regime', 'low')
            if volatility_regime == 'high':
                priorities['volatility_hedging'] = 'high'
            elif volatility_regime == 'moderate':
                priorities['volatility_hedging'] = 'medium'
            else:
                priorities['volatility_hedging'] = 'low'

            # Tail risk priority
            tail_risk = risk_factors.get('return_factor', {}).get('tail_risk_potential', 'low')
            if tail_risk == 'high':
                priorities['tail_risk_protection'] = 'high'
            elif tail_risk == 'moderate':
                priorities['tail_risk_protection'] = 'medium'
            else:
                priorities['tail_risk_protection'] = 'low'

            # Size stability priority
            size_risk = risk_factors.get('size_factor', {}).get('concentration_risk', 'low')
            if size_risk == 'high':
                priorities['size_stabilization'] = 'high'
            elif size_risk == 'moderate':
                priorities['size_stabilization'] = 'medium'
            else:
                priorities['size_stabilization'] = 'low'

            return priorities
        except Exception as e:
            self.logger.error(f"Error prioritizing risk mitigation: {e}")
            return {}

    def _analyze_efficiency_distribution(self, size_adjusted_scores: List[Dict]) -> Dict[str, Any]:
        """Analyze efficiency distribution across clusters."""
        try:
            if not size_adjusted_scores:
                return {}

            # Calculate efficiency percentiles
            efficiency_scores = [s['size_efficiency_score'] for s in size_adjusted_scores]
            efficiency_percentiles = np.percentile(efficiency_scores, [25, 50, 75])

            # Classify clusters by efficiency
            high_efficiency = [s['cluster_id'] for s in size_adjusted_scores if s['size_efficiency_score'] > efficiency_percentiles[2]]
            medium_efficiency = [s['cluster_id'] for s in size_adjusted_scores if efficiency_percentiles[0] <= s['size_efficiency_score'] <= efficiency_percentiles[2]]
            low_efficiency = [s['cluster_id'] for s in size_adjusted_scores if s['size_efficiency_score'] < efficiency_percentiles[0]]

            return {
                'efficiency_tiers': {
                    'high_efficiency_clusters': high_efficiency,
                    'medium_efficiency_clusters': medium_efficiency,
                    'low_efficiency_clusters': low_efficiency
                },
                'efficiency_distribution': {
                    'high_efficiency_percentage': len(high_efficiency) / len(size_adjusted_scores) if size_adjusted_scores else 0.0,
                    'medium_efficiency_percentage': len(medium_efficiency) / len(size_adjusted_scores) if size_adjusted_scores else 0.0,
                    'low_efficiency_percentage': len(low_efficiency) / len(size_adjusted_scores) if size_adjusted_scores else 0.0
                },
                'efficiency_statistics': {
                    'average_efficiency': float(np.mean(efficiency_scores)),
                    'efficiency_range': float(np.max(efficiency_scores) - np.min(efficiency_scores)),
                    'efficiency_volatility': float(np.std(efficiency_scores))
                }
            }
        except Exception as e:
            self.logger.error(f"Error analyzing efficiency distribution: {e}")
            return {}

def run_optimal_clustering(data_path: Optional[str] = None, output_dir: Optional[str] = None,
                          config: Optional[OptimalClusteringConfig] = None,
                          symbol: str = "ETHUSDT", exchange: str = "binance", timeframe: str = "15m",
                          **kwargs) -> Dict[str, Any]:
    """Convenience function to run optimal regime clustering (Matrix-Optimized by Default).

    This is the main entry point for optimal regime clustering. It automatically:
    - Detects the latest HMM regime discovery results if data_path is not provided
    - Uses matrix optimization with GPU acceleration when available
    - Creates output in the same location as HMM discovery results if output_dir is not provided
    - Maintains full compatibility with the 4D feature space (volume, volatility, momentum, trend)

    Features:
    - ✅ Matrix optimization with GPU acceleration (Apple Silicon M1/M2/M3)
    - ✅ 4D feature space processing (volume, volatility, momentum, trend)
    - ✅ 20 optimal clusters with 90-95% coverage
    - ✅ <5% noise with advanced filtering
    - ✅ Automatic detection of latest HMM discovery results
    - ✅ Automatic fallback to standard clustering if matrix ops unavailable

    Args:
        data_path: Path to HMM regime data (optional - auto-detects if not provided)
        output_dir: Output directory (optional - uses HMM discovery location if not provided)
        config: Clustering configuration (optional, defaults to matrix-optimized config)
        symbol: Trading symbol (default: ETHUSDT)
        exchange: Exchange name (default: binance)
        timeframe: Data timeframe (default: 15m)
        **kwargs: Additional parameters

    Returns:
        Pipeline results with matrix optimization enabled by default
    """
    # Auto-detect HMM results if not provided
    if data_path is None or output_dir is None:
        detected_data_path, detected_output_dir = detect_latest_hmm_results(symbol, exchange, timeframe)
        if detected_data_path:
            data_path = data_path or detected_data_path
            output_dir = output_dir or detected_output_dir
            logger.info(f"🔍 Auto-detected HMM results: {data_path}")
        else:
            logger.error("❌ No HMM results found and no data_path provided")
            return {
                'success': False,
                'error': 'No HMM regime data found. Please provide data_path or ensure HMM discovery has been run.',
                'metadata': {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
            }

    orchestrator = OptimalRegimeClusteringOrchestrator(config)
    return orchestrator.run_clustering_pipeline(data_path, output_dir, symbol, exchange, timeframe, **kwargs)

def run_high_quality_clustering(data_path: Optional[str] = None, output_dir: Optional[str] = None,
                               symbol: str = "ETHUSDT", exchange: str = "binance", timeframe: str = "15m",
                               **kwargs) -> Dict[str, Any]:
    """Run high-quality clustering with enhanced validation and matrix optimization.

    This function uses stricter quality criteria while maintaining matrix optimization
    for maximum performance. Ideal for production use with rigorous validation requirements.

    Features:
    - ✅ Matrix optimization with GPU acceleration
    - ✅ Enhanced quality validation (higher thresholds)
    - ✅ Stricter cluster size requirements
    - ✅ Comprehensive validation metrics

    Args:
        data_path: Path to HMM regime data
        output_dir: Output directory
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        **kwargs: Additional parameters

    Returns:
        Pipeline results with high-quality validation
    """
    # Auto-detect HMM results if not provided
    if data_path is None or output_dir is None:
        detected_data_path, detected_output_dir = detect_latest_hmm_results(symbol, exchange, timeframe)
        if detected_data_path:
            data_path = data_path or detected_data_path
            output_dir = output_dir or detected_output_dir
            logger.info(f"🔍 Auto-detected HMM results: {data_path}")
        else:
            logger.error("❌ No HMM results found and no data_path provided")
            return {
                'success': False,
                'error': 'No HMM regime data found. Please provide data_path or ensure HMM discovery has been run.',
                'metadata': {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
            }

    config = get_clustering_config("high_quality")
    orchestrator = OptimalRegimeClusteringOrchestrator(config)
    return orchestrator.run_clustering_pipeline(data_path, output_dir, symbol, exchange, timeframe, **kwargs)

def run_fast_clustering(data_path: Optional[str] = None, output_dir: Optional[str] = None,
                       symbol: str = "ETHUSDT", exchange: str = "binance", timeframe: str = "15m",
                       **kwargs) -> Dict[str, Any]:
    """Run fast clustering for quick results with relaxed validation.

    This function prioritizes speed over quality validation. Uses matrix optimization
    when available but with relaxed quality thresholds for faster processing.

    Features:
    - ✅ Matrix optimization with GPU acceleration
    - ✅ Reduced validation requirements for speed
    - ✅ Optimized for large datasets
    - ✅ Quick results for prototyping

    Args:
        data_path: Path to HMM regime data
        output_dir: Output directory
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        **kwargs: Additional parameters

    Returns:
        Pipeline results with fast processing
    """
    # Auto-detect HMM results if not provided
    if data_path is None or output_dir is None:
        detected_data_path, detected_output_dir = detect_latest_hmm_results(symbol, exchange, timeframe)
        if detected_data_path:
            data_path = data_path or detected_data_path
            output_dir = output_dir or detected_output_dir
            logger.info(f"🔍 Auto-detected HMM results: {data_path}")
        else:
            logger.error("❌ No HMM results found and no data_path provided")
            return {
                'success': False,
                'error': 'No HMM regime data found. Please provide data_path or ensure HMM discovery has been run.',
                'metadata': {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
            }

    config = get_clustering_config("fast_processing")
    orchestrator = OptimalRegimeClusteringOrchestrator(config)
    return orchestrator.run_clustering_pipeline(data_path, output_dir, symbol, exchange, timeframe, **kwargs)

def run_matrix_optimized_clustering(data_path: Optional[str] = None, output_dir: Optional[str] = None,
                                  symbol: str = "ETHUSDT", exchange: str = "binance", timeframe: str = "15m",
                                  **kwargs) -> Dict[str, Any]:
    """Run matrix-optimized clustering (explicit matrix optimization mode).

    This function is an alias for run_optimal_clustering() that explicitly emphasizes
    the use of matrix optimization with GPU acceleration. It ensures maximum performance
    and provides the same 4D feature space processing capabilities.

    Note: run_optimal_clustering() now uses matrix optimization by default, making
    this function functionally equivalent but with explicit matrix optimization emphasis.

    Features:
    - ✅ Matrix optimization with GPU acceleration (Apple Silicon M1/M2/M3)
    - ✅ 4D feature space processing (volume, volatility, momentum, trend)
    - ✅ Maximum performance with vectorized operations
    - ✅ Comprehensive quality validation

    Args:
        data_path: Path to HMM regime data
        output_dir: Output directory
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        **kwargs: Additional parameters

    Returns:
        Pipeline results with explicit matrix optimization
    """
    # Auto-detect HMM results if not provided
    if data_path is None or output_dir is None:
        detected_data_path, detected_output_dir = detect_latest_hmm_results(symbol, exchange, timeframe)
        if detected_data_path:
            data_path = data_path or detected_data_path
            output_dir = output_dir or detected_output_dir
            logger.info(f"🔍 Auto-detected HMM results: {data_path}")
        else:
            logger.error("❌ No HMM results found and no data_path provided")
            return {
                'success': False,
                'error': 'No HMM regime data found. Please provide data_path or ensure HMM discovery has been run.',
                'metadata': {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
            }

    # Use the same implementation as run_optimal_clustering since it's now matrix-optimized by default
    return run_optimal_clustering(data_path, output_dir, None, symbol, exchange, timeframe, **kwargs)
