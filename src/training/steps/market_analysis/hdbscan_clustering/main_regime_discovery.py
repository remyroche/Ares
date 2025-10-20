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
    tprint_logged, LogLevel
)
from src.utils.common_operations import safe_dataframe_operation
from src.utils.common_utilities import safe_dataframe_operation as safe_df_op
from src.utils.math_validation import validate_finite, safe_divide, safe_log, safe_sqrt
from src.utils.serialization_utils import save_pickle, load_pickle

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
            
            # Enhanced validation if enabled
            if self.config.enable_validation:
                validation_result = self._perform_enhanced_validation(
                    data, optimized_result.cluster_labels, economic_validator
                )
                validation_metrics.update(validation_result)
            
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
            
            # Initialize unified hardware manager
            with tprint_timer("Unified hardware manager initialization"):
                self.hardware_manager = get_unified_hardware_manager()
                self.hardware_manager.configure_workload(WorkloadType.ML_TRAINING, OptimizationLevel.AGGRESSIVE)
                tprint_success("✅ Unified hardware manager initialized")
            
            # Initialize comprehensive optimizer
            with tprint_timer("Comprehensive optimizer initialization"):
                comprehensive_config = ComprehensiveConfig(
                    optimization_strategy=OptimizationStrategy.MAXIMUM_PERFORMANCE,
                    workload_category=WorkloadType.MACHINE_LEARNING,
                    enable_adaptive_optimization=True,
                    enable_cross_component_optimization=True,
                    enable_thermal_management=True,
                    enable_power_management=True
                )
                self.comprehensive_optimizer = get_comprehensive_optimizer(comprehensive_config)
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
            
            # Enhanced memory optimization using hardware tools
            with self.hardware_manager.optimization_context(WorkloadType.ML_TRAINING, OptimizationLevel.AGGRESSIVE):
                data = optimize_dataframe_default(data)
                optimized_memory = get_memory_usage()
                tprint_debug(f"Memory after enhanced optimization: {optimized_memory:.2f}MB (saved {initial_memory - optimized_memory:.2f}MB)")
            
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
            
            # Use hardware optimization context for prediction
            with self.hardware_manager.optimization_context(WorkloadType.ML_TRAINING, OptimizationLevel.BALANCED):
                # Extract features
                feature_result = self.feature_extractor.extract_features(data)
                if not feature_result.success:
                    raise ValueError(f"Feature extraction failed: {feature_result.error_message}")
                
                # Preprocess features
                processed_result = self.feature_processor.process(feature_result.features_df, fit=False)
                if not processed_result.success:
                    raise ValueError(f"Preprocessing failed: {processed_result.error_message}")
                
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
        
        return {
            'hardware_manager': self.hardware_manager.get_system_status(),
            'cache_system': self.cache_system.get_statistics() if self.cache_system else None,
            'comprehensive_optimizer': self.comprehensive_optimizer.get_comprehensive_metrics() if self.comprehensive_optimizer else None,
            'performance_stats': self.performance_stats
        }
    
    def optimize_for_workload(self, workload_type: WorkloadType, optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE):
        """Optimize hardware for specific workload type."""
        if hasattr(self, 'hardware_manager'):
            self.hardware_manager.configure_workload(workload_type, optimization_level)
            tprint_info(f"🔧 Hardware optimized for {workload_type.value} workload ({optimization_level.value})")
        else:
            tprint_warning("⚠️ Hardware manager not available for optimization")
    
    def _perform_enhanced_validation(self, 
                                   market_data: pd.DataFrame,
                                   regime_labels: np.ndarray,
                                   economic_validator: Any) -> Dict[str, Any]:
        """Perform enhanced validation of regime discovery results."""
        try:
            logger.info("🔍 Performing enhanced validation")
            
            # Use the enhanced validation from economic_validator
            validation_result = economic_validator.validate_regime_quality(
                market_data, regime_labels
            )
            
            # Extract key metrics
            enhanced_metrics = {
                'regime_quality_score': validation_result.get('overall_score', 0.0),
                'regime_profiling_valid': validation_result.get('regime_profiling', {}).get('is_valid', False),
                'statistical_analysis_valid': validation_result.get('statistical_analysis', {}).get('is_valid', False),
                'economic_validation_valid': validation_result.get('economic_validation', {}).get('is_valid', False),
                'cross_validation_score': validation_result.get('cross_validation', {}).get('mean_cv_score', 0.0),
                'regime_stability': validation_result.get('regime_profiling', {}).get('regime_stability', 0.0),
                'regime_transitions': validation_result.get('regime_profiling', {}).get('regime_transitions', 0),
                'validation_issues': validation_result.get('regime_profiling', {}).get('issues', [])
            }
            
            logger.info(f"✅ Enhanced validation completed. Quality score: {enhanced_metrics['regime_quality_score']:.3f}")
            return enhanced_metrics
            
        except Exception as e:
            logger.error(f"❌ Enhanced validation failed: {e}")
            return {
                'regime_quality_score': 0.0,
                'validation_error': str(e)
            }
    
    def enhanced_predict_with_uncertainty(self, 
                                        market_data: pd.DataFrame,
                                        use_optimized: bool = True) -> Dict[str, Any]:
        """
        Enhanced prediction with uncertainty quantification.
        
        Args:
            market_data: Market data for prediction
            use_optimized: Whether to use optimized components
            
        Returns:
            Dictionary with predictions and uncertainty measures
        """
        try:
            logger.info("🔮 Starting enhanced prediction with uncertainty")
            
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
            # Extract features
            features = self.regime_feature_extractor.extract_features(market_data)
            
            # Process features
            processed_result = self.feature_processor.process_features(features)
            processed_features = processed_result.processed_features
            
            # Reduce dimensionality
            dr_result = self.dimensionality_reducer.reduce(processed_features)
            reduced_features = dr_result.reduced_features
            
            # Get enhanced prediction
            if use_optimized and hasattr(self.optimized_discovery, 'enhanced_predict_with_uncertainty'):
                prediction_result = self.optimized_discovery.enhanced_predict_with_uncertainty(reduced_features)
            else:
                # Use legacy clusterer with enhanced prediction
                prediction_result = self.hdbscan_clusterer.enhanced_predict_with_uncertainty(reduced_features)
            
            if prediction_result.get('success', False):
                # Economic analysis for predictions
                economic_result = self.economic_validator.validate_and_profile(
                    market_data, prediction_result['labels']
                )
                
                return {
                    'labels': prediction_result['labels'],
                    'probabilities': prediction_result['probabilities'],
                    'uncertainty_measures': prediction_result.get('uncertainty_measures', {}),
                    'method_breakdown': prediction_result.get('method_breakdown', {}),
                    'economic_profiles': economic_result.get('regime_profiles', []),
                    'trading_recommendations': economic_result.get('trading_recommendations', {}),
                    'success': True
                }
            else:
                raise Exception(f"Enhanced prediction failed: {prediction_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Enhanced prediction failed: {e}")
            return {'error': str(e), 'success': False}
    
    def save_model(self, filepath: str) -> bool:
        """Save the complete regime discovery model."""
        try:
            logger.info(f"💾 Saving model to {filepath}")
            
            # Save individual components
            component_paths = {}
            
            # Save HDBSCAN clusterer
            if hasattr(self.hdbscan_clusterer, 'save_model'):
                clusterer_path = f"{filepath}_clusterer.pkl"
                if self.hdbscan_clusterer.save_model(clusterer_path):
                    component_paths['clusterer'] = clusterer_path
            
            # Save other components if they have save methods
            for name, component in [
                ('feature_extractor', self.regime_feature_extractor),
                ('feature_processor', self.feature_processor),
                ('dimensionality_reducer', self.dimensionality_reducer),
                ('economic_validator', self.economic_validator),
                ('temporal_stabilizer', self.temporal_stabilizer)
            ]:
                if hasattr(component, 'save_model'):
                    comp_path = f"{filepath}_{name}.pkl"
                    if component.save_model(comp_path):
                        component_paths[name] = comp_path
            
            # Save main model metadata
            model_data = {
                'config': self.config,
                'component_paths': component_paths,
                'is_fitted': self.is_fitted,
                'model_metadata': {
                    'created_at': time.time(),
                    'version': '1.0.0'
                }
            }
            
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("✅ Model saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model saving failed: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a complete regime discovery model."""
        try:
            logger.info(f"📂 Loading model from {filepath}")
            
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load configuration
            self.config = model_data['config']
            
            # Load individual components
            component_paths = model_data.get('component_paths', {})
            
            for name, comp_path in component_paths.items():
                try:
                    if name == 'clusterer' and hasattr(self.hdbscan_clusterer, 'load_model'):
                        self.hdbscan_clusterer.load_model(comp_path)
                    elif name == 'feature_extractor' and hasattr(self.regime_feature_extractor, 'load_model'):
                        self.regime_feature_extractor.load_model(comp_path)
                    elif name == 'feature_processor' and hasattr(self.feature_processor, 'load_model'):
                        self.feature_processor.load_model(comp_path)
                    elif name == 'dimensionality_reducer' and hasattr(self.dimensionality_reducer, 'load_model'):
                        self.dimensionality_reducer.load_model(comp_path)
                    elif name == 'economic_validator' and hasattr(self.economic_validator, 'load_model'):
                        self.economic_validator.load_model(comp_path)
                    elif name == 'temporal_stabilizer' and hasattr(self.temporal_stabilizer, 'load_model'):
                        self.temporal_stabilizer.load_model(comp_path)
                except Exception as e:
                    logger.warning(f"Failed to load component {name}: {e}")
            
            self.is_fitted = model_data.get('is_fitted', False)
            
            logger.info("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def generate_enhanced_report(self) -> str:
        """Generate an enhanced comprehensive report."""
        try:
            report = []
            report.append("=" * 100)
            report.append("ENHANCED HDBSCAN REGIME DISCOVERY SYSTEM - COMPREHENSIVE REPORT")
            report.append("=" * 100)
            report.append("")
            
            # System status
            report.append("SYSTEM STATUS:")
            report.append(f"  Fitted: {self.is_fitted}")
            report.append(f"  Use Optimized: {self.use_optimized}")
            report.append(f"  Enable Validation: {self.config.enable_validation}")
            report.append("")
            
            # Configuration
            report.append("CONFIGURATION:")
            report.append(f"  Feature Extraction: {self.config.feature_extraction_config.__class__.__name__}")
            report.append(f"  Feature Processing: {self.config.feature_processing_config.__class__.__name__}")
            report.append(f"  Dimensionality Reduction: {self.config.dimensionality_reduction_config.__class__.__name__}")
            report.append(f"  HDBSCAN Clustering: {self.config.hdbscan_config.__class__.__name__}")
            report.append(f"  Economic Validation: {self.config.economic_validation_config.__class__.__name__}")
            report.append(f"  Temporal Stabilization: {self.config.temporal_stabilization_config.__class__.__name__}")
            report.append("")
            
            # Enhanced features
            report.append("ENHANCED FEATURES:")
            report.append("  ✅ Advanced Probability Calculation")
            report.append("  ✅ Model Persistence")
            report.append("  ✅ Uncertainty Quantification")
            report.append("  ✅ Enhanced Validation")
            report.append("  ✅ Improved Error Handling")
            report.append("")
            
            # Recommendations
            report.append("RECOMMENDATIONS:")
            if self.is_fitted:
                report.append("  - System is ready for production use")
                report.append("  - Consider real-time implementation")
                report.append("  - Add monitoring and alerting")
            else:
                report.append("  - Run fit() method to train the system")
                report.append("  - Use enhanced_predict_with_uncertainty() for predictions")
                report.append("  - Save model after training for future use")
            
            report.append("")
            report.append("=" * 100)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            return f"Report generation failed: {e}"
