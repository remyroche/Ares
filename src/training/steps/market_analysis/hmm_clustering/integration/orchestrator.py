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
from ..config import HMMClusteringConfig

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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the orchestrator.

        Args:
            config: Clustering configuration
        """
        self.config = config or HMMClusteringConfig.create_default()
        self.logger = logging.getLogger(__name__)
        
        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_manager = None
        self.performance_monitor = None
        
        if HARDWARE_ACCELERATION_AVAILABLE:
            try:
                self.hardware_accelerator = get_hardware_accelerator()
                self.memory_manager = get_memory_manager()
                self.performance_monitor = get_performance_monitor()
                self.logger.info("✅ Hardware acceleration initialized for orchestrator")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available for orchestrator: {e}")
        
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
                self.logger.info("✅ Matrix operations initialized for orchestrator")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available for orchestrator: {e}")
        
        # Initialize clustering components
        self.use_matrix_optimization = self.config.clustering_config.get('use_matrix_optimization', True)
        self.use_enhanced_clustering = self.config.clustering_config.get('use_enhanced_clustering', True)
        
        # Initialize clusterers
        self.clusterer = None
        self.enhanced_clusterer = None
        
        if self.use_matrix_optimization:
            self.clusterer = MatrixOptimizedClusterer(self.config.clustering_config)
            self.logger.info("✅ Matrix optimized clusterer initialized")
        else:
            # Fallback to standard clusterer if needed
            self.logger.warning("⚠️ Matrix optimization disabled, using fallback clusterer")
        
        if self.use_enhanced_clustering:
            self.enhanced_clusterer = EnhancedMatrixOptimizedClusterer(self.config.clustering_config)
            self.logger.info("✅ Enhanced clusterer initialized")
        
        # Initialize metrics calculators
        self.basic_metrics_calc = BasicClusteringMetrics(self.config.metrics_config)
        self.detailed_metrics_calc = DetailedClusteringMetrics(self.config.metrics_config)
        self.evolution_reporter = MetricsEvolutionReporter(self.config.metrics_config)

    def run_optimal_clustering(self, data: Union[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """Run the complete optimal clustering pipeline.

        Args:
            data: Path to HMM regime data or DataFrame
            **kwargs: Additional parameters

        Returns:
            Dictionary with clustering results
        """
        start_time = time.time()
        
        try:
            self.logger.info("🚀 Starting optimal regime clustering pipeline")
            
            # Monitor performance
            if self.performance_monitor:
                self.performance_monitor.start_monitoring("optimal_clustering_pipeline")
            
            # Load and prepare data
            regime_data = self._load_regime_data(data)
            features = self._prepare_features(regime_data)
            
            # Run standard matrix-optimized clustering
            standard_result = None
            if self.clusterer is not None:
                self.logger.info("📊 Running standard matrix-optimized clustering")
                standard_result = self.clusterer.cluster(features)
                
                if not standard_result.success:
                    self.logger.error(f"❌ Standard clustering failed: {standard_result.error_message}")
                    return self._create_error_result("Standard clustering failed", standard_result.error_message, start_time)
                
                self.logger.info(f"✅ Standard clustering completed: {standard_result.quality_metrics.get('n_clusters', 0)} clusters")
            
            # Run enhanced clustering if available
            enhanced_result = None
            if self.enhanced_clusterer is not None and standard_result is not None:
                self.logger.info("🎯 Running enhanced clustering with 4D frontier optimization")
                enhanced_result = self.enhanced_clusterer.cluster(features)
                
                if not enhanced_result.success:
                    self.logger.error(f"❌ Enhanced clustering failed: {enhanced_result.error_message}")
                    # Use fast fail if configured
                    if self.config.integration_config.get('enable_fast_fail', True):
                        return self._create_error_result("Enhanced clustering failed - fast fail", enhanced_result.error_message, start_time)
                    else:
                        self.logger.warning("⚠️ Enhanced clustering failed, continuing with standard result")
                        enhanced_result = None
                else:
                    self.logger.info(f"✅ Enhanced clustering completed: {enhanced_result.quality_metrics.get('n_clusters', 0)} clusters")
            
            # Calculate comprehensive metrics
            comprehensive_metrics = self._calculate_comprehensive_metrics(features, standard_result, enhanced_result)
            
            # Generate metrics evolution report
            evolution_report = None
            if standard_result and hasattr(standard_result, 'metadata') and 'metrics_evolution' in standard_result.metadata:
                evolution_report = self.evolution_reporter.generate_comprehensive_report(
                    standard_result.metadata['metrics_evolution'],
                    enhanced_result.metadata if enhanced_result else None,
                    self.config
                )
            
            # Create final result
            final_result = self._create_final_result(
                standard_result, enhanced_result, comprehensive_metrics, evolution_report, start_time
            )
            
            # Stop performance monitoring
            if self.performance_monitor:
                perf_metrics = self.performance_monitor.stop_monitoring("optimal_clustering_pipeline")
                final_result['performance_metrics'] = perf_metrics
            
            self.logger.info("✅ Optimal regime clustering pipeline completed successfully")
            return final_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Optimal clustering pipeline failed: {e}")
            return self._create_error_result("Pipeline failed", str(e), start_time)

    def _load_regime_data(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Load regime data from file or DataFrame.

        Args:
            data: Path to data file or DataFrame

        Returns:
            Loaded regime data
        """
        try:
            if isinstance(data, str):
                if data.endswith('.json'):
                    with open(data, 'r') as f:
                        data_dict = json.load(f)
                    regime_data = pd.DataFrame(data_dict.get('regime_data', []))
                elif data.endswith('.parquet'):
                    regime_data = pd.read_parquet(data)
                elif data.endswith('.csv'):
                    regime_data = pd.read_csv(data)
                else:
                    raise ValueError(f"Unsupported file format: {data}")
            else:
                regime_data = data.copy()
            
            if regime_data.empty:
                raise ValueError("Regime data is empty")
            
            self.logger.info(f"✅ Loaded regime data: {regime_data.shape}")
            return regime_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load regime data: {e}")
            raise

    def _prepare_features(self, regime_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for clustering.

        Args:
            regime_data: Regime data DataFrame

        Returns:
            Feature matrix
        """
        try:
            # Extract features from regime data
            # This is a simplified version - in practice, you'd extract relevant features
            feature_columns = [col for col in regime_data.columns if col not in ['timestamp', 'regime']]
            
            if not feature_columns:
                # Fallback to numeric columns
                feature_columns = regime_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not feature_columns:
                raise ValueError("No suitable feature columns found in regime data")
            
            features = regime_data[feature_columns].values
            
            # Handle NaN values
            features = np.nan_to_num(features, nan=0.0)
            
            self.logger.info(f"✅ Prepared features: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare features: {e}")
            raise

    def _calculate_comprehensive_metrics(self, features: np.ndarray, standard_result: Any, 
                                       enhanced_result: Any) -> Dict[str, Any]:
        """Calculate comprehensive metrics for clustering results.

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
            if standard_result is not None:
                basic_metrics = self.basic_metrics_calc.calculate_basic_metrics(features, standard_result.labels)
                comprehensive_metrics['standard_basic_metrics'] = basic_metrics
                
                # Calculate detailed metrics for standard result
                detailed_metrics = self.detailed_metrics_calc.calculate_detailed_metrics(features, standard_result.labels)
                comprehensive_metrics['standard_detailed_metrics'] = detailed_metrics
            
            # Calculate basic metrics for enhanced result
            if enhanced_result is not None:
                basic_metrics = self.basic_metrics_calc.calculate_basic_metrics(features, enhanced_result.labels)
                comprehensive_metrics['enhanced_basic_metrics'] = basic_metrics
                
                # Calculate detailed metrics for enhanced result
                detailed_metrics = self.detailed_metrics_calc.calculate_detailed_metrics(features, enhanced_result.labels)
                comprehensive_metrics['enhanced_detailed_metrics'] = detailed_metrics
            
            # Compare standard vs enhanced if both available
            if standard_result is not None and enhanced_result is not None:
                comparison_metrics = self._compare_clustering_results(
                    standard_result, enhanced_result, comprehensive_metrics
                )
                comprehensive_metrics['standard_vs_enhanced'] = comparison_metrics
            
            self.logger.info("✅ Comprehensive metrics calculated")
            return comprehensive_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Comprehensive metrics calculation failed: {e}")
            return {'error': str(e)}

    def _compare_clustering_results(self, standard_result: Any, enhanced_result: Any, 
                                  comprehensive_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare standard and enhanced clustering results.

        Args:
            standard_result: Standard clustering result
            enhanced_result: Enhanced clustering result
            comprehensive_metrics: Comprehensive metrics

        Returns:
            Comparison metrics
        """
        try:
            comparison = {
                'silhouette_improvement': 0.0,
                'cluster_count_difference': 0,
                'quality_improvement': 'none',
                'enhancement_benefit': 0.0
            }
            
            # Compare silhouette scores
            if 'standard_basic_metrics' in comprehensive_metrics and 'enhanced_basic_metrics' in comprehensive_metrics:
                standard_silhouette = comprehensive_metrics['standard_basic_metrics'].silhouette
                enhanced_silhouette = comprehensive_metrics['enhanced_basic_metrics'].silhouette
                
                comparison['silhouette_improvement'] = enhanced_silhouette - standard_silhouette
            
            # Compare cluster counts
            if 'standard_basic_metrics' in comprehensive_metrics and 'enhanced_basic_metrics' in comprehensive_metrics:
                standard_clusters = comprehensive_metrics['standard_basic_metrics'].n_clusters
                enhanced_clusters = comprehensive_metrics['enhanced_basic_metrics'].n_clusters
                
                comparison['cluster_count_difference'] = enhanced_clusters - standard_clusters
            
            # Determine quality improvement
            silhouette_improvement = comparison['silhouette_improvement']
            if silhouette_improvement > 0.05:
                comparison['quality_improvement'] = 'significant'
            elif silhouette_improvement > 0.02:
                comparison['quality_improvement'] = 'moderate'
            elif silhouette_improvement > 0:
                comparison['quality_improvement'] = 'minor'
            else:
                comparison['quality_improvement'] = 'none'
            
            # Calculate overall enhancement benefit
            comparison['enhancement_benefit'] = max(0.0, silhouette_improvement)
            
            return comparison
            
        except Exception as e:
            self.logger.warning(f"⚠️ Clustering results comparison failed: {e}")
            return {'error': str(e)}

    def _create_final_result(self, standard_result: Any, enhanced_result: Any, 
                           comprehensive_metrics: Dict[str, Any], evolution_report: Any, 
                           start_time: float) -> Dict[str, Any]:
        """Create final clustering result.

        Args:
            standard_result: Standard clustering result
            enhanced_result: Enhanced clustering result
            comprehensive_metrics: Comprehensive metrics
            evolution_report: Metrics evolution report
            start_time: Start time

        Returns:
            Final result dictionary
        """
        execution_time = time.time() - start_time
        
        result = {
            'success': True,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat(),
            'standard_clustering': {
                'success': standard_result.success if standard_result else False,
                'labels': standard_result.labels.tolist() if standard_result else [],
                'n_clusters': standard_result.quality_metrics.get('n_clusters', 0) if standard_result else 0,
                'silhouette_score': standard_result.quality_metrics.get('silhouette', 0.0) if standard_result else 0.0
            },
            'enhanced_clustering': {
                'success': enhanced_result.success if enhanced_result else False,
                'labels': enhanced_result.labels.tolist() if enhanced_result else [],
                'n_clusters': enhanced_result.quality_metrics.get('n_clusters', 0) if enhanced_result else 0,
                'silhouette_score': enhanced_result.quality_metrics.get('silhouette', 0.0) if enhanced_result else 0.0,
                'frontiers': enhanced_result.frontiers if hasattr(enhanced_result, 'frontiers') else {},
                'transfer_history': enhanced_result.transfer_history if hasattr(enhanced_result, 'transfer_history') else [],
                'optimization_iterations': enhanced_result.optimization_iterations if hasattr(enhanced_result, 'optimization_iterations') else 0
            },
            'comprehensive_metrics': comprehensive_metrics,
            'metrics_evolution_report': evolution_report,
            'configuration': {
                'use_matrix_optimization': self.use_matrix_optimization,
                'use_enhanced_clustering': self.use_enhanced_clustering,
                'matrix_ops_available': self.matrix_ops is not None,
                'hardware_acceleration_available': self.hardware_accelerator is not None
            }
        }
        
        return result

    def _create_error_result(self, error_type: str, error_message: str, start_time: float) -> Dict[str, Any]:
        """Create error result.

        Args:
            error_type: Type of error
            error_message: Error message
            start_time: Start time

        Returns:
            Error result dictionary
        """
        execution_time = time.time() - start_time
        
        return {
            'success': False,
            'error_type': error_type,
            'error_message': error_message,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat(),
            'standard_clustering': {'success': False, 'labels': [], 'n_clusters': 0, 'silhouette_score': 0.0},
            'enhanced_clustering': {'success': False, 'labels': [], 'n_clusters': 0, 'silhouette_score': 0.0},
            'comprehensive_metrics': {'error': error_message},
            'metrics_evolution_report': None,
            'configuration': {
                'use_matrix_optimization': self.use_matrix_optimization,
                'use_enhanced_clustering': self.use_enhanced_clustering,
                'matrix_ops_available': self.matrix_ops is not None,
                'hardware_acceleration_available': self.hardware_accelerator is not None
            }
        }


def create_optimal_regime_clustering_orchestrator(config: Optional[Dict[str, Any]] = None) -> OptimalRegimeClusteringOrchestrator:
    """Create an optimal regime clustering orchestrator instance.

    Args:
        config: Configuration dictionary

    Returns:
        OptimalRegimeClusteringOrchestrator instance
    """
    return OptimalRegimeClusteringOrchestrator(config)


def run_optimal_clustering(data: Union[str, pd.DataFrame], config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Run optimal clustering with the orchestrator.

    Args:
        data: Path to HMM regime data or DataFrame
        config: Configuration dictionary
        **kwargs: Additional parameters

    Returns:
        Dictionary with clustering results
    """
    orchestrator = create_optimal_regime_clustering_orchestrator(config)
    return orchestrator.run_optimal_clustering(data, **kwargs)