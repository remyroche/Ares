"""
Main HDBSCAN Regime Discovery Orchestrator

Coordinates all components of the HDBSCAN-based regime discovery system:
- Feature extraction
- Preprocessing
- Dimensionality reduction
- HDBSCAN clustering
- Post-clustering optimization
- Validation and profiling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from dataclasses import dataclass, field
import time
import gc
import psutil

# Import existing utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, tprint_data_format, LogLevel
)
from src.utils.common_operations import safe_dataframe_operation
from src.utils.common_utilities import safe_dataframe_operation as safe_df_op
from src.utils.math_validation import validate_finite, safe_divide, safe_log, safe_sqrt
from src.utils.serialization_utils import save_pickle, load_pickle

# Import enhanced ML common utilities
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer, OptimizationConfig, TPEConfig
)
from src.utils.ml_common.unified_vectorization_manager import (
    UnifiedVectorizationManager, OperationType, OptimizationStrategy
)
from src.utils.ml_common.evaluation.unified_evaluator import UnifiedEvaluator
from src.utils.ml_common.explainability.model_explainability import ModelExplainability

# Import enhanced hardware optimization tools
from src.utils.hardware import (
    get_unified_hardware_manager, get_comprehensive_optimizer,
    smart_cache, auto_optimize, memory_efficient, performance_tracked,
    optimize_dataframe_default, optimize_numpy_array_default,
    WorkloadType, OptimizationLevel, ComprehensiveConfig, OptimizationStrategy,
    get_memory_usage
)

# Import optimized regime discovery components (default)
from .optimization.optimized_hdbscan_regime_discovery import (
    OptimizedHDBSCANRegimeDiscovery,
    OptimizedRegimeResult,
    OptimizedHDBSCANRegimeDiscoveryConfig,
    create_optimized_hdbscan_regime_discovery
)

# Import legacy regime discovery components (fallback)
from .config.regime_discovery_config import RegimeDiscoveryConfig
from .regime_feature_extractor import RegimeFeatureExtractor
from .feature_processor import FeatureProcessor
from .dimensionality_reducer import DimensionalityReducer
from .preprocessing.temporal_window_handler import TemporalWindowHandler
from .hdbscan_clusterer import HDBSCANClusterer
from .clustering.noise_handler import NoiseHandler
from .sample_reallocator import SampleReallocator
from .economic_validator import EconomicValidator
from .temporal_stabilizer import TemporalStabilizer
from .similarity_merger import SimilarityMerger

logger = logging.getLogger(__name__)


@dataclass
class RegimeResult:
    """Result of regime discovery."""
    labels: np.ndarray
    probabilities: np.ndarray
    cluster_persistence: np.ndarray
    economic_profiles: List[Any]
    validation_metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class HDBSCANRegimeDiscovery:
    """
    Main orchestrator for HDBSCAN-based regime discovery.
    
    Coordinates all components:
    1. Feature extraction (5 families)
    2. Preprocessing (winsorization, transformation, pruning)
    3. Dimensionality reduction (PCA/UMAP/densMAP)
    4. HDBSCAN clustering (with tree export)
    5. Noise handling (causal/acausal smoothing)
    6. Post-clustering optimization (reallocation, merging)
    7. Economic validation and profiling
    8. Temporal stabilization
    """
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def __init__(self, config: RegimeDiscoveryConfig, use_optimized: bool = True):
        """Initialize the regime discovery system."""
        start_time = time.perf_counter()
        initial_memory = get_memory_usage()
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.use_optimized = use_optimized
        
        # Initialize enhanced hardware optimization
        with tprint_timer("Enhanced hardware optimization initialization"):
            self._initialize_enhanced_hardware_optimization()
        
        # Initialize ML common utilities
        with tprint_timer("ML common utilities initialization"):
            self._initialize_ml_common_utilities()
        
        # Performance tracking
        self.performance_stats = {
            'initialization_time': 0.0,
            'memory_usage_mb': 0.0,
            'total_discoveries': 0,
            'total_processing_time': 0.0
        }
        
        tprint_info(f"Initializing HDBSCAN regime discovery (optimized={use_optimized})")
        
        if self.use_optimized:
            tprint_debug("Using optimized implementation")
            # Use optimized implementation by default
            with tprint_timer("Optimized discovery creation"):
                self.optimized_discovery = create_optimized_hdbscan_regime_discovery(
                    min_cluster_size=getattr(config, 'min_cluster_size', 10),
                    min_samples=getattr(config, 'min_samples', 5),
                    cluster_selection_epsilon=getattr(config, 'cluster_selection_epsilon', 0.0),
                    cluster_selection_method=getattr(config, 'cluster_selection_method', 'eom'),
                    metric=getattr(config, 'metric', 'euclidean'),
                    enable_hyperparameter_optimization=True,
                    enable_memory_optimization=True,
                    enable_vectorized_processing=True,
                    enable_features_common=True,
                    enable_feature_selection=True,
                    max_features=getattr(config, 'max_features', 50),
                    max_memory_gb=getattr(config, 'max_memory_gb', 8.0)
                )
            tprint_success("🚀 Optimized HDBSCAN regime discovery initialized")
        else:
            tprint_debug("Using legacy implementation")
            # Initialize components
            with tprint_timer("Components initialization"):
                self._initialize_components()
            
            # State tracking
            self.last_result = None
            self.discovery_history = []
            
            tprint_success("🚀 Legacy HDBSCAN regime discovery initialized")
        
        # Track initialization performance
        init_time = time.perf_counter() - start_time
        final_memory = get_memory_usage()
        self.performance_stats['initialization_time'] = init_time
        self.performance_stats['memory_usage_mb'] = final_memory
        
        tprint_performance("HDBSCAN regime discovery initialization", init_time)
        tprint_debug(f"Memory usage: {initial_memory:.2f}MB -> {final_memory:.2f}MB (delta: {final_memory - initial_memory:+.2f}MB)")
        
        self.logger.info("HDBSCANRegimeDiscovery initialized successfully")
    
    async def _discover_regimes_optimized(self, data: pd.DataFrame, 
                                        fit: bool = True, 
                                        is_live: bool = False,
                                        returns: Optional[np.ndarray] = None) -> RegimeResult:
        """Discover regimes using optimized implementation with enhanced hardware optimization."""
        try:
            # Use hardware optimization context for the entire discovery process
            with self.hardware_manager.optimization_context(WorkloadType.ML_TRAINING, OptimizationLevel.MAXIMUM):
                # Use optimized regime discovery
                optimized_result = self.optimized_discovery.discover_regimes(data)
                
                # Convert to legacy format
                return self._convert_optimized_result_to_legacy(optimized_result, data, fit, is_live, returns)
            
        except Exception as e:
            tprint(f"❌ Optimized regime discovery failed: {e}", "ERROR")
            # Return error result
            return RegimeResult(
                labels=np.array([]),
                probabilities=np.array([]),
                cluster_persistence=np.array([]),
                economic_profiles=[],
                validation_metrics={},
                metadata={},
                processing_time=0.0,
                success=False,
                error_message=f"Optimized regime discovery failed: {e}"
            )
    
    def _convert_optimized_result_to_legacy(self, optimized_result: OptimizedRegimeResult, 
                                         data: pd.DataFrame, fit: bool, is_live: bool, 
                                         returns: Optional[np.ndarray]) -> RegimeResult:
        """Convert optimized result to legacy format."""
        try:
            # Create economic profiles using economic validator
            from .economic_validator import EconomicValidator
            
            economic_validator = EconomicValidator()
            economic_result = economic_validator.validate_and_profile(
                cluster_labels=optimized_result.cluster_labels,
                market_data=data,
                returns=returns
            )
            
            # Convert to legacy format
            economic_profiles = []
            for profile in economic_result.profiles:
                legacy_profile = {
                    'regime_id': profile.regime_id,
                    'name': profile.name,
                    'key_stats': profile.key_stats,
                    'confidence_intervals': profile.confidence_intervals,
                    'avg_duration': profile.avg_duration,
                    'transitions': profile.transitions,
                    'works_best_for': profile.works_best_for,
                    'risk_caveats': profile.risk_caveats,
                    'radar_plot_data': profile.radar_plot_data
                }
                economic_profiles.append(legacy_profile)
            
            # Create validation metrics
            validation_metrics = {
                'n_regimes': optimized_result.n_clusters,
                'noise_ratio': optimized_result.noise_ratio,
                'silhouette_score': optimized_result.silhouette_score,
                'calinski_harabasz_score': optimized_result.calinski_harabasz_score,
                'davies_bouldin_score': optimized_result.davies_bouldin_score
            }
            
            # Create metadata
            metadata = {
                'processing_time': optimized_result.processing_time,
                'memory_usage_mb': optimized_result.memory_usage_mb,
                'optimization_stats': optimized_result.optimization_stats,
                'feature_importance': optimized_result.feature_importance,
                'cluster_persistence': optimized_result.cluster_persistence,
                'condensed_tree': optimized_result.condensed_tree,
                'mst': optimized_result.mst,
                'glosh_scores': optimized_result.glosh_scores,
                'cluster_centers': optimized_result.cluster_centers,
                'cluster_sizes': optimized_result.cluster_sizes
            }
            
            return RegimeResult(
                labels=optimized_result.cluster_labels,
                probabilities=optimized_result.cluster_probabilities,
                cluster_persistence=optimized_result.cluster_persistence or np.array([]),
                economic_profiles=economic_profiles,
                validation_metrics=validation_metrics,
                metadata=metadata,
                processing_time=optimized_result.processing_time,
                success=True,
                error_message=None
            )
            
        except Exception as e:
            tprint(f"⚠️ Failed to convert optimized result: {e}", "WARNING")
            # Return basic success result
            return RegimeResult(
                labels=optimized_result.cluster_labels,
                probabilities=optimized_result.cluster_probabilities,
                cluster_persistence=np.array([]),
                economic_profiles=[],
                validation_metrics={'n_regimes': optimized_result.n_clusters, 'noise_ratio': optimized_result.noise_ratio},
                metadata={'processing_time': optimized_result.processing_time},
                processing_time=optimized_result.processing_time,
                success=True,
                error_message=None
            )
    
    @tprint_logged(LogLevel.INFO, include_result=True)
    def _initialize_enhanced_hardware_optimization(self):
        """Initialize enhanced hardware optimization utilities."""
        try:
            tprint_info("Initializing enhanced hardware optimization utilities")
            
            # Initialize unified hardware manager with enhanced configuration
            with tprint_timer("Unified hardware manager initialization"):
                self.hardware_manager = get_unified_hardware_manager()
                
                # Configure for machine learning workload with aggressive optimization
                self.hardware_manager.configure_workload(WorkloadType.ML_TRAINING, OptimizationLevel.AGGRESSIVE)
                
                # Enable M1-specific optimizations if available
                if hasattr(self.hardware_manager, 'enable_m1_optimizations'):
                    self.hardware_manager.enable_m1_optimizations(True)
                    tprint_info("🍎 M1-specific optimizations enabled")
                
                # Configure memory optimization
                if hasattr(self.hardware_manager, 'configure_memory_optimization'):
                    self.hardware_manager.configure_memory_optimization(
                        enable_compression=True,
                        enable_caching=True,
                        memory_threshold_mb=500.0
                    )
                    tprint_info("💾 Memory optimization configured")
                
                tprint_success("✅ Unified hardware manager initialized")
            
            # Initialize comprehensive optimizer with enhanced configuration
            with tprint_timer("Comprehensive optimizer initialization"):
                comprehensive_config = ComprehensiveConfig(
                    optimization_strategy=OptimizationStrategy.MAXIMUM_PERFORMANCE,
                    workload_category=WorkloadType.MACHINE_LEARNING,
                    enable_adaptive_optimization=True,
                    enable_cross_component_optimization=True,
                    enable_thermal_management=True,
                    enable_power_management=True,
                    # Enhanced ML-specific optimizations
                    enable_gpu_acceleration=True,
                    enable_vectorization=True,
                    enable_parallel_processing=True,
                    max_memory_usage_gb=8.0,
                    enable_memory_compression=True
                )
                self.comprehensive_optimizer = get_comprehensive_optimizer(comprehensive_config)
                
                # Configure ML-specific optimizations
                if hasattr(self.comprehensive_optimizer, 'configure_ml_optimizations'):
                    self.comprehensive_optimizer.configure_ml_optimizations(
                        enable_batch_processing=True,
                        enable_feature_caching=True,
                        enable_model_caching=True,
                        enable_gradient_accumulation=True
                    )
                    tprint_info("🤖 ML-specific optimizations configured")
                
                tprint_success("✅ Comprehensive optimizer initialized")
            
            # Initialize caching system
            with tprint_timer("Enhanced caching system initialization"):
                self.cache_system = self.hardware_manager.cache_system
                tprint_success("✅ Enhanced caching system initialized")
            
            tprint_success("🚀 Enhanced hardware optimization complete")
                
        except Exception as e:
            tprint_error(f"❌ Enhanced hardware optimization initialization failed: {e}")
            # Fallback to basic hardware manager
            self.hardware_manager = get_unified_hardware_manager(conservative_mode=True)
            self.comprehensive_optimizer = None
            self.cache_system = None
    
    @tprint_logged(LogLevel.INFO, include_result=True)
    def _initialize_ml_common_utilities(self):
        """Initialize ML common utilities for enhanced regime discovery."""
        try:
            tprint_info("Initializing ML common utilities")
            
            # Initialize Bayesian TPE optimizer for hyperparameter optimization
            with tprint_timer("Bayesian TPE optimizer initialization"):
                tpe_config = TPEConfig(
                    n_trials=100,
                    timeout_seconds=3600,
                    enable_pruning=True,
                    enable_memory_optimization=True
                )
                self.bayesian_optimizer = BayesianTPEOptimizer(tpe_config)
                tprint_success("✅ Bayesian TPE optimizer initialized")
            
            # Initialize unified vectorization manager
            with tprint_timer("Unified vectorization manager initialization"):
                self.vectorization_manager = UnifiedVectorizationManager()
                self.vectorization_manager.configure_operation(
                    OperationType.MACHINE_LEARNING,
                    OptimizationStrategy.BALANCED
                )
                tprint_success("✅ Unified vectorization manager initialized")
            
            # Initialize unified evaluator
            with tprint_timer("Unified evaluator initialization"):
                self.evaluator = UnifiedEvaluator()
                tprint_success("✅ Unified evaluator initialized")
            
            # Initialize model explainability
            with tprint_timer("Model explainability initialization"):
                self.explainability = ModelExplainability()
                tprint_success("✅ Model explainability initialized")
            
            tprint_success("🚀 ML common utilities initialization complete")
                
        except Exception as e:
            tprint_error(f"❌ ML common utilities initialization failed: {e}")
            # Fallback to basic functionality
            self.bayesian_optimizer = None
            self.vectorization_manager = None
            self.evaluator = None
            self.explainability = None
    
    def _initialize_components(self):
        """Initialize all regime discovery components."""
        try:
            # Feature extraction
            self.feature_extractor = RegimeFeatureExtractor(self.config)
            
            # Preprocessing
            self.feature_processor = FeatureProcessor(self.config)
            self.dimensionality_reducer = DimensionalityReducer(self.config)
            self.temporal_window_handler = TemporalWindowHandler(self.config)
            
            # Clustering
            self.hdbscan_clusterer = HDBSCANClusterer(self.config)
            self.noise_handler = NoiseHandler(self.config)
            
            # Optimization
            self.sample_reallocator = SampleReallocator(self.config)
            self.economic_validator = EconomicValidator(self.config)
            self.temporal_stabilizer = TemporalStabilizer(self.config)
            self.similarity_merger = SimilarityMerger(self.config)
            
            tprint("✅ All regime discovery components initialized", "SUCCESS")
            
        except Exception as e:
            tprint(f"❌ Component initialization failed: {e}", "ERROR")
            raise
    
    @smart_cache(ttl=3600)  # Cache results for 1 hour
    @auto_optimize(optimize_inputs=True, optimize_outputs=True)
    @memory_efficient(memory_threshold_mb=200.0, auto_cleanup=True)
    @performance_tracked(log_performance=True, track_memory=True)
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    async def discover_regimes(self, data: pd.DataFrame, 
                              fit: bool = True, 
                              is_live: bool = False,
                              returns: Optional[np.ndarray] = None) -> RegimeResult:
        """
        Discover regimes using HDBSCAN-based approach with enhanced hardware optimization.
        
        Args:
            data: Market data with datetime index
            fit: Whether to fit models (train mode)
            is_live: Whether this is live trading (affects smoothing modes)
            returns: Optional returns data for economic validation
            
        Returns:
            RegimeResult with labels, profiles, and metadata
        """
        start_time = datetime.now()
        perf_start = time.perf_counter()
        initial_memory = get_memory_usage()
        
        try:
            tprint_info(f"🔍 Starting regime discovery: fit={fit}, live={is_live}")
            tprint_debug(f"Data shape: {data.shape}, Memory usage: {initial_memory:.2f}MB")
            
            # Enhanced data format analysis for troubleshooting
            tprint_data_format(data, "input_market_data", level=LogLevel.INFO)
            
            # Enhanced memory optimization using hardware tools
            with self.hardware_manager.optimization_context(WorkloadType.ML_TRAINING, OptimizationLevel.AGGRESSIVE):
                data = optimize_dataframe_default(data)
                optimized_memory = get_memory_usage()
                tprint_debug(f"Memory after enhanced optimization: {optimized_memory:.2f}MB (saved {initial_memory - optimized_memory:.2f}MB)")
                
                # Data format analysis after optimization
                tprint_data_format(data, "optimized_market_data", level=LogLevel.DEBUG)
            
            # Use optimized implementation if enabled
            if self.use_optimized:
                tprint_debug("Using optimized implementation")
                return await self._discover_regimes_optimized(data, fit, is_live, returns)
            
            # Validate input
            self._validate_input(data)
            
            # Step 1: Feature extraction with hardware optimization
            tprint("📊 Step 1: Feature extraction", "INFO")
            with self.hardware_manager.optimization_context(WorkloadType.FEATURE_ENGINEERING, OptimizationLevel.AGGRESSIVE):
                feature_result = self.feature_extractor.extract_features(data)
                
                # Data format analysis for feature extraction results
                if feature_result.success and hasattr(feature_result, 'features_df'):
                    tprint_data_format(feature_result.features_df, "extracted_features", level=LogLevel.INFO)
                else:
                    tprint_data_format(feature_result, "feature_extraction_result", level=LogLevel.ERROR)
                
                if not feature_result.success:
                    return RegimeResult(
                        labels=np.array([]),
                        probabilities=np.array([]),
                        cluster_persistence=np.array([]),
                        economic_profiles=[],
                        validation_metrics={},
                        metadata={},
                        processing_time=(datetime.now() - start_time).total_seconds(),
                        success=False,
                        error_message=f"Feature extraction failed: {feature_result.error_message}"
                    )
            
            # Step 2: Preprocessing with enhanced optimization
            tprint("⚙️ Step 2: Preprocessing", "INFO")
            with self.hardware_manager.optimization_context(WorkloadType.DATA_PROCESSING, OptimizationLevel.AGGRESSIVE):
                processed_result = self.feature_processor.process(
                    feature_result.features_df, fit=fit
                )
                
                # Data format analysis for preprocessing results
                if processed_result.success and hasattr(processed_result, 'processed_features_df'):
                    tprint_data_format(processed_result.processed_features_df, "processed_features", level=LogLevel.INFO)
                else:
                    tprint_data_format(processed_result, "preprocessing_result", level=LogLevel.ERROR)
                
                if not processed_result.success:
                    return RegimeResult(
                        labels=np.array([]),
                        probabilities=np.array([]),
                        cluster_persistence=np.array([]),
                        economic_profiles=[],
                        validation_metrics={},
                        metadata={},
                        processing_time=(datetime.now() - start_time).total_seconds(),
                        success=False,
                        error_message=f"Preprocessing failed: {processed_result.error_message}"
                    )
            
            # Step 3: Dimensionality reduction with hardware acceleration
            tprint("📉 Step 3: Dimensionality reduction", "INFO")
            with self.hardware_manager.optimization_context(WorkloadType.ML_TRAINING, OptimizationLevel.AGGRESSIVE):
                reduced_features, dr_metadata = self.dimensionality_reducer.reduce(
                    processed_result.processed_features_df.values, fit=fit
                )
                
                # Data format analysis for dimensionality reduction results
                tprint_data_format(reduced_features, "reduced_features", level=LogLevel.INFO)
                tprint_data_format(dr_metadata, "dimensionality_reduction_metadata", level=LogLevel.DEBUG)
            
            # Step 4: Temporal windowing
            tprint("🔄 Step 4: Temporal windowing", "INFO")
            windowed_features, window_metadata = self.temporal_window_handler.create_windows(
                data, processed_result.feature_names
            )
            
            # Step 5: HDBSCAN clustering with comprehensive optimization
            tprint("🔍 Step 5: HDBSCAN clustering", "INFO")
            with self.hardware_manager.optimization_context(WorkloadType.ML_TRAINING, OptimizationLevel.MAXIMUM):
                cluster_result = self.hdbscan_clusterer.fit_predict(
                    reduced_features, window_metadata['n_effective_samples']
                )
                
                # Data format analysis for clustering results
                if hasattr(cluster_result, 'labels'):
                    tprint_data_format(cluster_result.labels, "cluster_labels", level=LogLevel.INFO)
                if hasattr(cluster_result, 'probabilities'):
                    tprint_data_format(cluster_result.probabilities, "cluster_probabilities", level=LogLevel.INFO)
                if hasattr(cluster_result, 'metadata'):
                    tprint_data_format(cluster_result.metadata, "clustering_metadata", level=LogLevel.DEBUG)
            
            # Step 6: Noise handling
            tprint("🔧 Step 6: Noise handling", "INFO")
            processed_labels, noise_metadata = self.noise_handler.handle_noise(
                cluster_result.labels, cluster_result.probabilities, is_live=is_live
            )
            
            # Step 7: Post-clustering optimization
            tprint("🔄 Step 7: Post-clustering optimization", "INFO")
            reallocation_result = await self.sample_reallocator.optimize(
                reduced_features, processed_labels, processed_result.processed_features_df.values,
                condensed_tree=cluster_result.condensed_tree
            )
            
            # Step 8: Similarity-based merging
            tprint("🔗 Step 8: Similarity-based merging", "INFO")
            merge_result = self.similarity_merger.merge_similar_regimes(
                reallocation_result.optimized_labels, reduced_features,
                processed_result.processed_features_df.values, cluster_result.condensed_tree
            )
            
            # Step 9: Economic validation and profiling
            tprint("💰 Step 9: Economic validation and profiling", "INFO")
            economic_result = self.economic_validator.validate_and_profile(
                merge_result.merged_labels, processed_result.processed_features_df.values, returns
            )
            
            # Step 10: Temporal stabilization
            tprint("⏰ Step 10: Temporal stabilization", "INFO")
            stabilization_result = self.temporal_stabilizer.smooth_regimes(
                merge_result.merged_labels, cluster_result.probabilities,
                mode='causal' if is_live else 'acausal', is_live=is_live
            )
            
            # Create final result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = RegimeResult(
                labels=stabilization_result.stabilized_labels,
                probabilities=cluster_result.probabilities,
                cluster_persistence=cluster_result.cluster_persistence,
                economic_profiles=economic_result.profiles,
                validation_metrics={
                    'economic_separation': economic_result.economic_separation_pct,
                    'validation_passed': economic_result.validation_passed,
                    'n_regimes': len(np.unique(stabilization_result.stabilized_labels[stabilization_result.stabilized_labels != -1])),
                    'noise_ratio': np.sum(stabilization_result.stabilized_labels == -1) / len(stabilization_result.stabilized_labels),
                    'reallocation_moves': reallocation_result.moves_made,
                    'merges_performed': merge_result.total_merges,
                    'stabilization_changes': stabilization_result.changes_made
                },
                metadata={
                    'feature_extraction': feature_result.feature_metadata,
                    'preprocessing': processed_result.metadata,
                    'dimensionality_reduction': dr_metadata,
                    'windowing': window_metadata,
                    'clustering': cluster_result.metadata,
                    'noise_handling': noise_metadata,
                    'reallocation': reallocation_result.metadata,
                    'merging': merge_result.metadata,
                    'economic_validation': economic_result.metadata,
                    'temporal_stabilization': stabilization_result.metadata,
                    'processing_time': processing_time,
                    'fit_mode': fit,
                    'live_mode': is_live,
                    'timestamp': datetime.now().isoformat()
                },
                processing_time=processing_time,
                success=True
            )
            
            # Store result
            self.last_result = result
            self.discovery_history.append(result)
            
            tprint(f"✅ Regime discovery completed in {processing_time:.2f} seconds", "SUCCESS")
            tprint(f"📊 Final result: {result.validation_metrics['n_regimes']} regimes, {result.validation_metrics['noise_ratio']:.1%} noise", "INFO")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            tprint(f"❌ Regime discovery failed: {e}", "ERROR")
            
            return RegimeResult(
                labels=np.array([]),
                probabilities=np.array([]),
                cluster_persistence=np.array([]),
                economic_profiles=[],
                validation_metrics={},
                metadata={},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    @smart_cache(ttl=1800)  # Cache predictions for 30 minutes
    @auto_optimize(optimize_inputs=True, optimize_outputs=True)
    @performance_tracked(log_performance=True)
    def predict_regimes(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Predict regimes for new data using fitted models with enhanced hardware optimization.
        
        Args:
            data: New market data
            
        Returns:
            Tuple of (labels, probabilities, method_used)
        """
        try:
            tprint("🔮 Predicting regimes for new data", "INFO")
            
            if self.last_result is None:
                raise ValueError("No fitted models available. Call discover_regimes with fit=True first.")
            
            # Data format validation for prediction input
            tprint_data_format(data, "prediction_input_data", LogLevel.DEBUG)
            
            # Use hardware optimization context for prediction
            with self.hardware_manager.optimization_context(WorkloadType.ML_TRAINING, OptimizationLevel.BALANCED):
                # Extract features
                feature_result = self.feature_extractor.extract_features(data)
                if not feature_result.success:
                    raise ValueError(f"Feature extraction failed: {feature_result.error_message}")
                
                # Data format validation for extracted features
                tprint_data_format(feature_result.features_df, "extracted_features", LogLevel.DEBUG)
                
                # Preprocess features
                processed_result = self.feature_processor.process(feature_result.features_df, fit=False)
                if not processed_result.success:
                    raise ValueError(f"Preprocessing failed: {processed_result.error_message}")
                
                # Data format validation for processed features
                tprint_data_format(processed_result.features_df, "processed_features", LogLevel.DEBUG)
                
                # Reduce dimensionality
                reduced_features, _ = self.dimensionality_reducer.reduce(
                    processed_result.processed_features_df.values, fit=False
                )
                
                # Predict using HDBSCAN
                labels, probabilities, method_used = self.hdbscan_clusterer.approximate_predict_with_fallback(
                    reduced_features
                )
            
            tprint(f"✅ Regime prediction completed using {method_used}", "SUCCESS")
            
            return labels, probabilities, method_used
            
        except Exception as e:
            tprint(f"❌ Regime prediction failed: {e}", "ERROR")
            raise
    
    def save_models(self, path: str) -> Dict[str, Any]:
        """Save all fitted models."""
        try:
            models_data = {
                'config': self.config,
                'feature_extractor': self.feature_extractor,
                'feature_processor': self.feature_processor,
                'dimensionality_reducer': self.dimensionality_reducer,
                'hdbscan_clusterer': self.hdbscan_clusterer,
                'noise_handler': self.noise_handler,
                'sample_reallocator': self.sample_reallocator,
                'economic_validator': self.economic_validator,
                'temporal_stabilizer': self.temporal_stabilizer,
                'similarity_merger': self.similarity_merger,
                'last_result': self.last_result,
                'created_at': datetime.now().isoformat()
            }
            
            save_pickle(models_data, path)
            tprint(f"✅ Models saved to {path}", "SUCCESS")
            
            return {
                'path': path,
                'models_saved': list(models_data.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            tprint(f"❌ Model saving failed: {e}", "ERROR")
            raise
    
    def load_models(self, path: str) -> bool:
        """Load all fitted models."""
        try:
            models_data = load_pickle(path)
            
            # Restore components
            self.config = models_data.get('config', self.config)
            self.feature_extractor = models_data.get('feature_extractor', self.feature_extractor)
            self.feature_processor = models_data.get('feature_processor', self.feature_processor)
            self.dimensionality_reducer = models_data.get('dimensionality_reducer', self.dimensionality_reducer)
            self.hdbscan_clusterer = models_data.get('hdbscan_clusterer', self.hdbscan_clusterer)
            self.noise_handler = models_data.get('noise_handler', self.noise_handler)
            self.sample_reallocator = models_data.get('sample_reallocator', self.sample_reallocator)
            self.economic_validator = models_data.get('economic_validator', self.economic_validator)
            self.temporal_stabilizer = models_data.get('temporal_stabilizer', self.temporal_stabilizer)
            self.similarity_merger = models_data.get('similarity_merger', self.similarity_merger)
            self.last_result = models_data.get('last_result', self.last_result)
            
            tprint(f"✅ Models loaded from {path}", "SUCCESS")
            
            return True
            
        except Exception as e:
            tprint(f"❌ Model loading failed: {e}", "ERROR")
            return False
    
    def _validate_input(self, data: pd.DataFrame):
        """Validate input data."""
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be None or empty")
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if len(data) < self.config.window_size:
            raise ValueError(f"Insufficient data: {len(data)} < {self.config.window_size}")
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get summary of regime discovery system with enhanced hardware optimization stats."""
        summary = {
            'last_result_available': self.last_result is not None,
            'discovery_history_count': len(self.discovery_history),
            'components_initialized': {
                'feature_extractor': hasattr(self, 'feature_extractor'),
                'feature_processor': hasattr(self, 'feature_processor'),
                'dimensionality_reducer': hasattr(self, 'dimensionality_reducer'),
                'temporal_window_handler': hasattr(self, 'temporal_window_handler'),
                'hdbscan_clusterer': hasattr(self, 'hdbscan_clusterer'),
                'noise_handler': hasattr(self, 'noise_handler'),
                'sample_reallocator': hasattr(self, 'sample_reallocator'),
                'economic_validator': hasattr(self, 'economic_validator'),
                'temporal_stabilizer': hasattr(self, 'temporal_stabilizer'),
                'similarity_merger': hasattr(self, 'similarity_merger')
            },
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        }
        
        # Add enhanced hardware optimization statistics
        if hasattr(self, 'hardware_manager'):
            summary['hardware_optimization'] = {
                'hardware_manager_status': self.hardware_manager.get_system_status(),
                'cache_statistics': self.cache_system.get_statistics() if self.cache_system else None,
                'comprehensive_optimizer_metrics': self.comprehensive_optimizer.get_comprehensive_metrics() if self.comprehensive_optimizer else None
            }
        
        return summary
    
    def get_hardware_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive hardware optimization statistics."""
        if not hasattr(self, 'hardware_manager'):
            return {'error': 'Hardware manager not initialized'}
        
        stats = {
            'hardware_manager': self.hardware_manager.get_system_status(),
            'cache_system': self.cache_system.get_statistics() if self.cache_system else None,
            'comprehensive_optimizer': self.comprehensive_optimizer.get_comprehensive_metrics() if self.comprehensive_optimizer else None,
            'performance_stats': self.performance_stats
        }
        
        # Add enhanced hardware metrics if available
        if hasattr(self.hardware_manager, 'get_enhanced_metrics'):
            stats['enhanced_metrics'] = self.hardware_manager.get_enhanced_metrics()
        
        if hasattr(self.hardware_manager, 'get_m1_metrics'):
            stats['m1_metrics'] = self.hardware_manager.get_m1_metrics()
        
        if hasattr(self.hardware_manager, 'get_memory_optimization_stats'):
            stats['memory_optimization'] = self.hardware_manager.get_memory_optimization_stats()
        
        if hasattr(self.hardware_manager, 'get_gpu_metrics'):
            stats['gpu_metrics'] = self.hardware_manager.get_gpu_metrics()
        
        return stats
    
    def optimize_for_workload(self, workload_type: WorkloadType, optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE):
        """Optimize hardware for specific workload type."""
        if hasattr(self, 'hardware_manager'):
            self.hardware_manager.configure_workload(workload_type, optimization_level)
            tprint_info(f"🔧 Hardware optimized for {workload_type.value} workload ({optimization_level.value})")
        else:
            tprint_warning("⚠️ Hardware manager not available for optimization")
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def optimize_for_regime_discovery(self, data_size: int, feature_count: int) -> Dict[str, Any]:
        """
        Optimize hardware specifically for regime discovery operations.
        
        Args:
            data_size: Number of data points
            feature_count: Number of features
            
        Returns:
            Dictionary with optimization results
        """
        try:
            if not hasattr(self, 'hardware_manager'):
                return {'success': False, 'error': 'Hardware manager not available'}
            
            tprint_info(f"🔧 Optimizing hardware for regime discovery: {data_size} samples, {feature_count} features")
            
            # Configure workload based on data size
            if data_size > 100000:
                workload_type = WorkloadType.ML_TRAINING
                optimization_level = OptimizationLevel.MAXIMUM
            elif data_size > 10000:
                workload_type = WorkloadType.ML_TRAINING
                optimization_level = OptimizationLevel.AGGRESSIVE
            else:
                workload_type = WorkloadType.ML_TRAINING
                optimization_level = OptimizationLevel.BALANCED
            
            # Configure hardware for regime discovery
            self.hardware_manager.configure_workload(workload_type, optimization_level)
            
            # Configure memory optimization based on data size
            if hasattr(self.hardware_manager, 'configure_memory_optimization'):
                memory_threshold = min(1000.0, data_size * feature_count * 8 / 1024 / 1024)  # Estimate memory usage
                self.hardware_manager.configure_memory_optimization(
                    enable_compression=True,
                    enable_caching=True,
                    memory_threshold_mb=memory_threshold
                )
                tprint_info(f"💾 Memory optimization configured: {memory_threshold:.1f}MB threshold")
            
            # Configure vectorization for regime discovery
            if hasattr(self.hardware_manager, 'configure_vectorization'):
                self.hardware_manager.configure_vectorization(
                    enable_vectorization=True,
                    vectorization_threshold=1000,
                    enable_parallel_processing=True
                )
                tprint_info("⚡ Vectorization configured for regime discovery")
            
            # Get optimization recommendations
            recommendations = self.hardware_manager.get_optimization_recommendations(
                workload_type=workload_type,
                data_size=data_size,
                feature_count=feature_count
            )
            
            tprint_success(f"✅ Hardware optimized for regime discovery")
            
            return {
                'workload_type': workload_type.value,
                'optimization_level': optimization_level.value,
                'recommendations': recommendations,
                'success': True
            }
            
        except Exception as e:
            tprint_error(f"❌ Hardware optimization for regime discovery failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def optimize_hyperparameters(self, data: pd.DataFrame, 
                                search_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Bayesian TPE optimizer.
        
        Args:
            data: Market data for optimization
            search_space: Optional custom search space
            
        Returns:
            Dictionary with optimized parameters and results
        """
        try:
            if not hasattr(self, 'bayesian_optimizer') or self.bayesian_optimizer is None:
                tprint_warning("⚠️ Bayesian optimizer not available, using default parameters")
                return self._get_default_hyperparameters()
            
            tprint_info("🔧 Starting hyperparameter optimization with Bayesian TPE")
            
            # Define search space if not provided
            if search_space is None:
                search_space = {
                    'min_cluster_size': (10, 50),
                    'min_samples': (3, 20),
                    'cluster_selection_epsilon': (0.0, 0.5),
                    'metric': ['euclidean', 'manhattan', 'cosine'],
                    'alpha': (0.5, 2.0)
                }
            
            # Define objective function
            def objective(trial):
                try:
                    # Sample parameters
                    params = {
                        'min_cluster_size': trial.suggest_int('min_cluster_size', *search_space['min_cluster_size']),
                        'min_samples': trial.suggest_int('min_samples', *search_space['min_samples']),
                        'cluster_selection_epsilon': trial.suggest_float('cluster_selection_epsilon', *search_space['cluster_selection_epsilon']),
                        'metric': trial.suggest_categorical('metric', search_space['metric']),
                        'alpha': trial.suggest_float('alpha', *search_space['alpha'])
                    }
                    
                    # Create temporary clusterer with sampled parameters
                    temp_config = HDBSCANClustererConfig(**params)
                    temp_clusterer = HDBSCANClusterer(temp_config)
                    
                    # Perform clustering
                    cluster_labels, clustering_info = temp_clusterer.cluster_data(data)
                    
                    # Calculate validation score
                    if len(cluster_labels) > 0:
                        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                        if n_clusters >= 2:
                            valid_mask = cluster_labels != -1
                            if valid_mask.sum() >= 2:
                                valid_labels = cluster_labels[valid_mask]
                                valid_features = data.values[valid_mask]
                                
                                if len(set(valid_labels)) >= 2:
                                    score = silhouette_score(valid_features, valid_labels)
                                    return score
                    
                    return -1.0  # Return low score for invalid clustering
                    
                except Exception as e:
                    tprint_debug(f"Trial failed: {e}")
                    return -1.0
            
            # Run optimization
            with tprint_timer("Bayesian TPE optimization"):
                optimization_result = self.bayesian_optimizer.optimize(
                    objective_function=objective,
                    search_space=search_space,
                    n_trials=50,
                    timeout_seconds=1800
                )
            
            if optimization_result and optimization_result.best_params:
                tprint_success(f"✅ Hyperparameter optimization completed. Best score: {optimization_result.best_value:.3f}")
                return {
                    'best_params': optimization_result.best_params,
                    'best_score': optimization_result.best_value,
                    'n_trials': optimization_result.n_trials,
                    'optimization_time': optimization_result.optimization_time,
                    'success': True
                }
            else:
                tprint_warning("⚠️ Hyperparameter optimization failed, using default parameters")
                return self._get_default_hyperparameters()
                
        except Exception as e:
            tprint_error(f"❌ Hyperparameter optimization failed: {e}")
            return self._get_default_hyperparameters()
    
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters as fallback."""
        return {
            'best_params': {
                'min_cluster_size': 20,
                'min_samples': 5,
                'cluster_selection_epsilon': 0.0,
                'metric': 'euclidean',
                'alpha': 1.0
            },
            'best_score': 0.0,
            'n_trials': 0,
            'optimization_time': 0.0,
            'success': False
        }
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def optimize_vectorization(self, data: pd.DataFrame, 
                             operation_type: OperationType = OperationType.MACHINE_LEARNING) -> Dict[str, Any]:
        """
        Optimize vectorization for specific operation type.
        
        Args:
            data: Data to optimize vectorization for
            operation_type: Type of operation to optimize for
            
        Returns:
            Dictionary with optimization results
        """
        try:
            if not hasattr(self, 'vectorization_manager') or self.vectorization_manager is None:
                tprint_warning("⚠️ Vectorization manager not available")
                return {'success': False, 'error': 'Vectorization manager not available'}
            
            tprint_info(f"🔧 Optimizing vectorization for {operation_type.value}")
            
            # Configure vectorization for the operation type
            self.vectorization_manager.configure_operation(
                operation_type, 
                OptimizationStrategy.BALANCED
            )
            
            # Get optimization recommendations
            recommendations = self.vectorization_manager.get_optimization_recommendations(
                data_shape=data.shape,
                operation_type=operation_type
            )
            
            tprint_success(f"✅ Vectorization optimization completed")
            return {
                'recommendations': recommendations,
                'operation_type': operation_type.value,
                'data_shape': data.shape,
                'success': True
            }
            
        except Exception as e:
            tprint_error(f"❌ Vectorization optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def evaluate_regime_quality(self, cluster_labels: np.ndarray, 
                               features: np.ndarray, 
                               market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate regime quality using unified evaluator.
        
        Args:
            cluster_labels: Cluster labels
            features: Feature matrix
            market_data: Market data
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            if not hasattr(self, 'evaluator') or self.evaluator is None:
                tprint_warning("⚠️ Evaluator not available")
                return {'success': False, 'error': 'Evaluator not available'}
            
            tprint_info("📊 Evaluating regime quality")
            
            # Prepare data for evaluation
            evaluation_data = {
                'cluster_labels': cluster_labels,
                'features': features,
                'market_data': market_data
            }
            
            # Run comprehensive evaluation
            with tprint_timer("Regime quality evaluation"):
                evaluation_result = self.evaluator.evaluate_clustering(
                    data=evaluation_data,
                    metrics=['silhouette', 'calinski_harabasz', 'davies_bouldin', 'inertia']
                )
            
            tprint_success(f"✅ Regime quality evaluation completed")
            return {
                'evaluation_result': evaluation_result,
                'success': True
            }
            
        except Exception as e:
            tprint_error(f"❌ Regime quality evaluation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def explain_regime_discovery(self, cluster_labels: np.ndarray, 
                                features: np.ndarray, 
                                feature_names: List[str]) -> Dict[str, Any]:
        """
        Explain regime discovery using model explainability.
        
        Args:
            cluster_labels: Cluster labels
            features: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary with explanation results
        """
        try:
            if not hasattr(self, 'explainability') or self.explainability is None:
                tprint_warning("⚠️ Explainability not available")
                return {'success': False, 'error': 'Explainability not available'}
            
            tprint_info("🔍 Explaining regime discovery")
            
            # Prepare data for explanation
            explanation_data = {
                'cluster_labels': cluster_labels,
                'features': features,
                'feature_names': feature_names
            }
            
            # Run explanation
            with tprint_timer("Regime discovery explanation"):
                explanation_result = self.explainability.explain_clustering(
                    data=explanation_data,
                    methods=['shap', 'lime', 'permutation_importance']
                )
            
            tprint_success(f"✅ Regime discovery explanation completed")
            return {
                'explanation_result': explanation_result,
                'success': True
            }
            
        except Exception as e:
            tprint_error(f"❌ Regime discovery explanation failed: {e}")
            return {'success': False, 'error': str(e)}
