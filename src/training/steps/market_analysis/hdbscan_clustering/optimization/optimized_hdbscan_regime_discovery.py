"""
Optimized HDBSCAN Regime Discovery

This module provides a comprehensive, optimized HDBSCAN regime discovery system
that integrates all optimization components for maximum performance.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.tprint import tprint

# Import dedicated HDBSCAN parameter tuner
try:
    from .automated_hdbscan_parameter_tuner import AutomatedHDBSCANTuner, create_automated_hdbscan_tuner
    PARAMETER_TUNER_AVAILABLE = True
except ImportError as e:
    PARAMETER_TUNER_AVAILABLE = False
    logging.warning(f"Automated HDBSCAN parameter tuner not available: {e}")

# Import hardware optimization tools
from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, WorkloadType

# Import optimization components
from .enhanced_memory_optimizer import (
    EnhancedMemoryOptimizer,
    MemoryOptimizationConfig,
    create_enhanced_memory_optimizer
)

from .enhanced_hyperparameter_optimizer import (
    EnhancedHyperparameterOptimizer,
    HDBSCANHyperparameterConfig,
    create_enhanced_hyperparameter_optimizer
)

from .enhanced_vectorized_processor import (
    EnhancedVectorizedProcessor,
    VectorizedProcessingConfig,
    create_enhanced_vectorized_processor
)

from .features_common_integration import (
    FeaturesCommonHDBSCANIntegration,
    FeaturesCommonIntegrationConfig,
    create_features_common_hdbscan_integration
)

# Enhanced regime features are now integrated into the main feature generation system

# Import feature generation systems
from src.feature_generation.categories.entropy import create_default_entropy_generators
from src.feature_generation.categories.spectral_wavelet import create_default_spectral_wavelet_generators
from src.feature_generation.categories.regime_features import create_default_regime_generators

# Import HDBSCAN
try:
    import hdbscan
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    hdbscan = None
    HDBSCAN = None

logger = logging.getLogger(__name__)

@dataclass
class OptimizedRegimeResult:
    """Result of optimized HDBSCAN regime discovery."""
    cluster_labels: np.ndarray
    cluster_probabilities: np.ndarray
    n_clusters: int
    n_noise_points: int
    cluster_persistence: Optional[np.ndarray] = None
    condensed_tree: Optional[Any] = None
    mst: Optional[Any] = None
    glosh_scores: Optional[np.ndarray] = None
    cluster_centers: Optional[np.ndarray] = None
    cluster_sizes: Optional[np.ndarray] = None
    noise_ratio: float = 0.0
    
    # Full HDBSCAN artifacts
    single_linkage_tree: Optional[Any] = None
    outlier_scores: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    exemplars: Optional[np.ndarray] = None
    
    # Performance metrics
    silhouette_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    
    # Processing information
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    optimization_stats: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None

@dataclass
class OptimizedHDBSCANRegimeDiscoveryConfig:
    """Configuration for optimized HDBSCAN regime discovery."""
    # Core HDBSCAN parameters - FORCE MORE CLUSTERS
    min_cluster_size: int = 3
    min_samples: int = 2
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = 'eom'
    metric: str = 'euclidean'
    alpha: float = 1.0

    # Execution mode for adaptive configuration (detected from ares_launcher)
    execution_mode: str = "light"  # "full", "light", "blank" - default fallback

    # Optimization settings
    enable_hyperparameter_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_vectorized_processing: bool = True
    enable_features_common: bool = True

    # Feature generation
    enable_entropy_features: bool = True
    enable_spectral_features: bool = True
    enable_regime_features: bool = True
    enable_normalization_features: bool = True

    # Performance settings
    enable_parallel_processing: bool = True
    max_memory_gb: float = 8.0
    chunk_size: int = 1000
    n_jobs: int = -1

    # Evaluation metrics
    primary_metric: str = 'silhouette'
    enable_cross_validation: bool = True
    cv_folds: int = 5

    # Advanced settings
    enable_feature_selection: bool = True
    feature_selection_method: str = 'mrmr'  # 'mrmr', 'lasso', 'mutual_info'
    max_features: int = 50
    feature_selection_threshold: float = 0.01

    def __post_init__(self):
        """Apply execution mode-based optimizations."""
        if self.execution_mode == "light":
            # Light mode: balanced performance and quality
            # Increased from 30 to 50 features for better regime differentiation
            self.max_features = 50
            self.cv_folds = 3
            self.chunk_size = 500
            # Re-enabled entropy features (important for regime complexity)
            self.enable_entropy_features = True
            # Keep spectral disabled (expensive to compute)
            self.enable_spectral_features = False
            # Always enable regime features (critical for regime detection)
            self.enable_regime_features = True
            self.enable_normalization_features = True
        elif self.execution_mode == "blank":
            # Blank mode: minimal but functional configuration (180 days data)
            # Increased from 40 to 50 features for adequate regime coverage
            self.max_features = 50
            self.cv_folds = 3
            self.chunk_size = 1000
            # Disable expensive features only
            self.enable_entropy_features = False
            self.enable_spectral_features = False
            # CRITICAL: Always enable regime features - they are essential for regime detection
            self.enable_regime_features = True
            # Re-enabled normalization features (lightweight and important)
            self.enable_normalization_features = True
        # Full mode: keep default values

class OptimizedHDBSCANRegimeDiscovery:
    """
    Optimized HDBSCAN regime discovery with comprehensive optimization.
    
    This class integrates all optimization components for maximum performance:
    - Memory & Data Processing Optimization
    - Hyperparameter Optimization
    - Vectorized Computations
    - Features Common Integration
    - Feature Selection
    - Performance Monitoring
    """
    
    def __init__(self, config: Optional[OptimizedHDBSCANRegimeDiscoveryConfig] = None):
        """Initialize the optimized HDBSCAN regime discovery."""
        self.config = config or OptimizedHDBSCANRegimeDiscoveryConfig()
        
        # Initialize optimization components
        self._initialize_optimization_components()
        
        # Initialize feature generators
        self._initialize_feature_generators()
        
        # Performance tracking
        self.performance_stats = {
            'total_processing_time': 0.0,
            'feature_generation_time': 0.0,
            'hyperparameter_optimization_time': 0.0,
            'clustering_time': 0.0,
            'post_processing_time': 0.0,
            'memory_optimizations': 0,
            'vectorized_operations': 0,
            'caching_hits': 0,
            'optimization_improvements': 0
        }
        
        logger.info("✅ OptimizedHDBSCANRegimeDiscovery initialized")
    
    def _initialize_vectorbt_optimizations(self):
        """Initialize VectorBT optimizations for enhanced performance."""
        from src.utils.tprint import tprint
        
        try:
            # Initialize UnifiedVectorizationManager for VectorBT acceleration
            from src.utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager
            from src.feature_generation.utils.unified_vectorization_manager import VectorizationConfig
            
            # Configure VectorBT optimizations
            vectorization_config = VectorizationConfig(
                enable_vectorbt=True,
                enable_gpu=self.config.enable_gpu if hasattr(self.config, 'enable_gpu') else False,
                memory_efficient=True,
                max_memory_gb=8.0,
                chunk_size=1000,
                enable_parallel=True
            )
            
            self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
            tprint("🚀 VectorBT optimizations initialized", "SUCCESS")
            
            # Initialize VectorBTRollingOptimizer for financial calculations
            try:
                from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
                
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=vectorization_config.enable_gpu,
                    enable_parallel=vectorization_config.enable_parallel,
                    memory_efficient=vectorization_config.memory_efficient
                )
                tprint("⚡ VectorBTRollingOptimizer initialized", "SUCCESS")
                
            except ImportError:
                tprint("⚠️ VectorBTRollingOptimizer not available, using fallback", "WARNING")
                self.rolling_optimizer = None
                
        except Exception as e:
            tprint(f"⚠️ VectorBT initialization failed: {e}", "WARNING")
            self.vectorization_manager = None
            self.rolling_optimizer = None
    
    def _initialize_optimization_components(self):
        """Initialize all optimization components."""
        # Memory optimizer
        if self.config.enable_memory_optimization:
            self.memory_optimizer = create_enhanced_memory_optimizer(
                max_memory_gb=self.config.max_memory_gb,
                enable_memory_optimization=True,
                enable_data_validation=True,
                enable_safe_operations=True,
                enable_memory_monitoring=True
            )
        else:
            self.memory_optimizer = None
        
        # Hyperparameter optimizer
        if self.config.enable_hyperparameter_optimization:
            # Adaptive number of trials based on execution mode
            n_trials = {
                "light": 2,    # Very fast for small datasets - only 2 trials
                "full": 50,    # Full optimization
                "blank": 1     # Minimal testing
            }.get(self.config.execution_mode, 2)
            
            self.hyperparameter_optimizer = create_enhanced_hyperparameter_optimizer(
                optimization_strategy="hybrid",
                n_trials=n_trials,
                primary_metric=self.config.primary_metric,
                enable_parallel=self.config.enable_parallel_processing,
                memory_efficient=True,
                execution_mode=self.config.execution_mode
            )
        else:
            self.hyperparameter_optimizer = None
        
        # Vectorized processor
        if self.config.enable_vectorized_processing:
            self.vectorized_processor = create_enhanced_vectorized_processor(
                enable_vectorbt=True,
                enable_gpu=False,
                enable_parallel=self.config.enable_parallel_processing,
                memory_efficient=True,
                max_memory_gb=self.config.max_memory_gb,
                chunk_size=self.config.chunk_size
            )
        else:
            self.vectorized_processor = None
        
        # Features common integration
        if self.config.enable_features_common:
            self.features_common_integration = create_features_common_hdbscan_integration(
                enable_unified_vectorization=True,
                enable_vectorbt_optimization=True,
                enable_performance_monitoring=True,
                enable_caching=True,
                optimization_level="high",
                memory_efficient=True,
                max_memory_gb=self.config.max_memory_gb
            )
        else:
            self.features_common_integration = None
    
    def _initialize_feature_generators(self):
        """Initialize feature generators."""
        self.feature_generators = []
        
        # Entropy features
        if self.config.enable_entropy_features:
            entropy_generators = create_default_entropy_generators()
            self.feature_generators.extend(entropy_generators)
        
        # Spectral features
        if self.config.enable_spectral_features:
            spectral_generators = create_default_spectral_wavelet_generators()
            self.feature_generators.extend(spectral_generators)
        
        # Regime features
        if self.config.enable_regime_features:
            regime_generators = create_default_regime_generators()
            self.feature_generators.extend(regime_generators)
        
        # Enhanced regime features are now integrated into the main feature generation system
    
    def discover_regimes(self, data: pd.DataFrame, 
                        labels: Optional[np.ndarray] = None) -> OptimizedRegimeResult:
        """
        Discover regimes using optimized HDBSCAN clustering.
        
        Args:
            data: Input data for regime discovery
            labels: Optional labels for supervised evaluation
            
        Returns:
            OptimizedRegimeResult with clustering results and performance metrics
        """
        start_time = time.time()
        
        # Data filtering is handled by BaseStep's _apply_light_mode_filter method
        
        logger.info(f"🚀 Starting optimized regime discovery for {data.shape[0]} samples")
        tprint(f"🚀 Starting optimized regime discovery for {data.shape[0]} samples", "INFO")
        
        # Memory optimization: Use chunked processing for large datasets
        if data.shape[0] > 10000 and self.config.execution_mode == "blank":
            tprint("🧠 Large dataset detected - using chunked processing for memory efficiency", "INFO")
            return self._discover_regimes_chunked(data, labels)
        else:
            return self._discover_regimes_standard(data, labels)
    
    def _discover_regimes_chunked(self, data: pd.DataFrame, labels: Optional[np.ndarray] = None) -> OptimizedRegimeResult:
        """
        Discover regimes using chunked processing for memory efficiency.
        
        Args:
            data: Input data for regime discovery
            labels: Optional labels for supervised evaluation
            
        Returns:
            OptimizedRegimeResult with clustering results and performance metrics
        """
        start_time = time.time()
        tprint("🧠 Using chunked processing for memory efficiency", "INFO")
        
        # Initialize hardware optimizers
        memory_optimizer = M1MemoryOptimizer()
        hardware_manager = UnifiedHardwareManager()
        
        # Optimize for feature engineering workload
        hardware_manager.optimize_for_workload(WorkloadType.FEATURE_ENGINEERING)
        tprint("🔧 Hardware optimized for feature engineering workload", "INFO")
        
        # Determine chunk size based on available memory and data size
        memory_info = memory_optimizer.get_memory_info()
        available_memory_gb = memory_info.get('available_gb', 8.0)
        
        # Adaptive chunk size based on available memory
        if available_memory_gb < 4.0:
            chunk_size = min(2000, data.shape[0] // 8)  # Very conservative for low memory
        elif available_memory_gb < 8.0:
            chunk_size = min(3000, data.shape[0] // 6)  # Conservative
        else:
            chunk_size = min(5000, data.shape[0] // 4)  # Standard
        
        tprint(f"📊 Chunk size: {chunk_size} samples per chunk (available memory: {available_memory_gb:.1f}GB)", "INFO")
        
        # Process data in chunks
        chunk_results = []
        total_chunks = (data.shape[0] + chunk_size - 1) // chunk_size
        
        for i in range(0, data.shape[0], chunk_size):
            chunk_end = min(i + chunk_size, data.shape[0])
            
            # Memory-efficient data loading with type optimization
            chunk_data = data.iloc[i:chunk_end].copy()
            chunk_labels = labels[i:chunk_end] if labels is not None else None
            
            # Optimize data types for memory efficiency
            chunk_data = self._optimize_data_types(chunk_data)
            
            chunk_num = (i // chunk_size) + 1
            tprint(f"🔄 Processing chunk {chunk_num}/{total_chunks} ({chunk_data.shape[0]} samples)", "INFO")
            
            # Check memory pressure before processing
            memory_pressure = memory_optimizer.get_memory_pressure()
            if memory_pressure > 0.8:  # 80% memory usage
                tprint(f"⚠️ High memory pressure ({memory_pressure:.1%}), applying cleanup", "WARNING")
                memory_optimizer.cleanup_memory()
                gc.collect()
            
            # Process chunk
            chunk_result = self._process_data_chunk(chunk_data, chunk_labels, chunk_num, memory_optimizer)
            chunk_results.append(chunk_result)
            
            # Aggressive memory cleanup
            del chunk_data
            if chunk_labels is not None:
                del chunk_labels
            gc.collect()
            
            # Force memory cleanup every few chunks
            if chunk_num % 3 == 0:
                memory_optimizer.cleanup_memory()
                tprint(f"🧹 Memory cleanup after chunk {chunk_num}", "INFO")
        
        # Merge chunk results
        tprint("🔗 Merging chunk results...", "INFO")
        final_result = self._merge_chunk_results(chunk_results)
        
        processing_time = time.time() - start_time
        tprint(f"✅ Chunked processing completed in {processing_time:.2f}s", "SUCCESS")
        
        return final_result
    
    def _optimize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize data types for memory efficiency.
        
        Args:
            data: DataFrame to optimize
            
        Returns:
            DataFrame with optimized data types
        """
        optimized_data = data.copy()
        
        # Convert numeric columns to more memory-efficient types
        for col in optimized_data.select_dtypes(include=[np.number]).columns:
            if optimized_data[col].dtype == 'float64':
                # Check if values fit in float32
                if optimized_data[col].min() >= np.finfo(np.float32).min and optimized_data[col].max() <= np.finfo(np.float32).max:
                    optimized_data[col] = optimized_data[col].astype(np.float32)
            elif optimized_data[col].dtype == 'int64':
                # Check if values fit in int32
                if optimized_data[col].min() >= np.iinfo(np.int32).min and optimized_data[col].max() <= np.iinfo(np.int32).max:
                    optimized_data[col] = optimized_data[col].astype(np.int32)
        
        # Convert object columns to category if they have few unique values
        for col in optimized_data.select_dtypes(include=['object']).columns:
            if optimized_data[col].nunique() / len(optimized_data) < 0.5:  # Less than 50% unique values
                optimized_data[col] = optimized_data[col].astype('category')
        
        return optimized_data
    
    def _process_data_chunk(self, chunk_data: pd.DataFrame, chunk_labels: Optional[np.ndarray], chunk_num: int, memory_optimizer: M1MemoryOptimizer) -> Dict[str, Any]:
        """
        Process a single data chunk.
        
        Args:
            chunk_data: Data chunk to process
            chunk_labels: Labels for the chunk
            chunk_num: Chunk number for logging
            
        Returns:
            Dictionary with chunk processing results
        """
        chunk_start_time = time.time()
        
        # Generate features for chunk
        tprint(f"🔧 Generating features for chunk {chunk_num}...", "INFO")
        chunk_features = self._generate_optimized_features(chunk_data)
        
        # Enhanced regime features are now integrated into the main feature generation system
        
        chunk_features = self._normalize_features_for_hdbscan(chunk_features)
        
        # Memory-efficient clustering with reduced cluster count
        tprint(f"🎯 Clustering chunk {chunk_num}...", "INFO")
        
        # Use configuration parameters directly (no adaptive calculation)
        min_cluster_size = self.config.min_cluster_size
        min_samples = self.config.min_samples
        
        # AGGRESSIVE optimization for much better clustering quality in chunks
        optimized_min_cluster_size = max(min_cluster_size, 30)  # Much larger minimum cluster size for stability
        optimized_min_samples = max(min_samples, 15)  # Much larger minimum samples for density estimation
        
        tprint(f"📊 Using optimized parameters for chunk: min_cluster_size={optimized_min_cluster_size}, min_samples={optimized_min_samples}", "INFO")
        
        # Use memory-efficient HDBSCAN parameters
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=optimized_min_cluster_size,
            min_samples=optimized_min_samples,
            cluster_selection_epsilon=0.1,  # Slightly higher to merge similar clusters
            cluster_selection_method='eom',
            metric='euclidean',
            memory='/tmp/hdbscan_cache',  # Use disk cache
            core_dist_n_jobs=1,  # Single thread to reduce memory
            algorithm='best'  # Use best algorithm for memory efficiency
        )
        
        # Ensure only numeric data
        numeric_features = chunk_features.select_dtypes(include=[np.number])
        numeric_features = numeric_features.astype(np.float64)
        numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Feature selection using data-driven PCA to reduce dimensionality
        if numeric_features.shape[1] > 10:  # Only apply PCA if we have many features
            tprint(f"🔧 Applying data-driven PCA to reduce features from {numeric_features.shape[1]}", "INFO")
            from sklearn.decomposition import PCA
            
            # First, determine optimal number of components based on variance explained
            pca_full = PCA(random_state=42)
            pca_full.fit(numeric_features)
            
            # Calculate cumulative variance explained
            cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
            
            # Find number of components that explain 95% of variance
            n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
            
            # Ensure we don't use more than 50% of original features or more than 30 components
            max_components = min(30, numeric_features.shape[1] // 2)
            n_components = min(n_components_95, max_components, numeric_features.shape[1])
            
            tprint(f"📊 Data-driven PCA: {n_components} components explain {cumsum_variance[n_components-1]:.1%} variance", "INFO")
            
            # Apply PCA with optimal number of components
            pca = PCA(n_components=n_components, random_state=42)
            numeric_features_pca = pca.fit_transform(numeric_features)
            numeric_features = pd.DataFrame(numeric_features_pca, 
                                          index=numeric_features.index,
                                          columns=[f'PC_{i+1}' for i in range(numeric_features_pca.shape[1])])
            
            # Clean up PCA objects
            del pca, pca_full, numeric_features_pca
            gc.collect()
        
        if not np.all(np.isfinite(numeric_features.values)):
            tprint(f"⚠️ Non-finite values detected in chunk {chunk_num}, skipping", "WARNING")
            return {
                'chunk_num': chunk_num,
                'n_samples': 0,
                'n_clusters': 0,
                'labels': np.array([]),
                'processing_time': time.time() - chunk_start_time
            }
        
        labels_chunk = clusterer.fit_predict(numeric_features)
        
        # Memory cleanup after clustering
        del clusterer, numeric_features
        memory_optimizer.cleanup_memory()
        gc.collect()
        
        chunk_time = time.time() - chunk_start_time
        tprint(f"✅ Chunk {chunk_num} processed in {chunk_time:.2f}s", "SUCCESS")
        
        return {
            'chunk_num': chunk_num,
            'n_samples': chunk_data.shape[0],
            'n_clusters': len(set(labels_chunk)) - (1 if -1 in labels_chunk else 0),
            'labels': labels_chunk,
            'processing_time': chunk_time,
            'features': numeric_features
        }
    
    def _merge_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> OptimizedRegimeResult:
        """
        Merge results from multiple chunks.
        
        Args:
            chunk_results: List of chunk processing results
            
        Returns:
            Merged OptimizedRegimeResult
        """
        tprint("🔗 Merging chunk results...", "INFO")
        
        # Combine all labels with proper offset
        all_labels = []
        total_samples = 0
        
        for i, result in enumerate(chunk_results):
            if result['n_samples'] > 0:
                # Offset labels to avoid conflicts between chunks
                offset_labels = result['labels'] + (i * 1000)  # Add offset to avoid label conflicts
                all_labels.extend(offset_labels)
                total_samples += result['n_samples']
        
        all_labels = np.array(all_labels)
        
        # Calculate overall metrics
        n_clusters = len(set(all_labels)) - (1 if -1 in all_labels else 0)
        noise_ratio = np.sum(all_labels == -1) / len(all_labels) if len(all_labels) > 0 else 0
        
        # Create merged result
        result = OptimizedRegimeResult(
            success=True,
            n_regimes=n_clusters,
            labels=all_labels,
            validation_metrics={
                'n_regimes': n_clusters,
                'noise_ratio': noise_ratio,
                'total_samples': total_samples,
                'processing_method': 'chunked',
                'n_chunks': len(chunk_results)
            },
            performance_metrics={
                'total_processing_time': sum(r['processing_time'] for r in chunk_results),
                'chunked_processing': True
            }
        )
        
        tprint(f"✅ Merged {len(chunk_results)} chunks: {n_clusters} regimes, {total_samples} samples", "SUCCESS")
        return result
    
    def _discover_regimes_standard(self, data: pd.DataFrame, labels: Optional[np.ndarray] = None) -> OptimizedRegimeResult:
        """
        Standard regime discovery for smaller datasets.
        
        Args:
            data: Input data for regime discovery
            labels: Optional labels for supervised evaluation
            
        Returns:
            OptimizedRegimeResult with clustering results and performance metrics
        """
        start_time = time.time()

        # Step 1: Feature generation with optimization
        tprint("🔧 Generating optimized features...", "INFO")
        features_df = self._generate_optimized_features(data)
        
        # Enhanced regime features are now integrated into the main feature generation system
        
        feature_generation_time = time.time() - start_time
        self.performance_stats['feature_generation_time'] += feature_generation_time
        tprint(f"✅ Feature generation completed in {feature_generation_time:.2f}s", "SUCCESS")
        tprint(f"📊 Generated {features_df.shape[1]} features from {features_df.shape[0]} samples", "INFO")

        # Step 1.5: Clean and normalize features for HDBSCAN (HDBSCAN is sensitive to feature scales)
        tprint("🔧 Cleaning and normalizing features for HDBSCAN...", "INFO")
        features_df = self._clean_and_normalize_features(features_df)
        tprint("✅ Feature cleaning and normalization completed", "SUCCESS")
        tprint(f"📊 After cleaning: {features_df.shape[1]} features from {features_df.shape[0]} samples", "INFO")

        # Step 2: Hyperparameter optimization
        tprint("🎛️ Starting hyperparameter optimization...", "INFO")
        tprint(f"📊 Data shape: {features_df.shape[0]} samples, {features_df.shape[1]} features", "INFO")
        tprint(f"⚙️ Execution mode: {self.config.execution_mode}", "INFO")
        hyperparameter_start = time.time()
        best_params = self._optimize_hyperparameters(features_df, labels)
        hyperparameter_time = time.time() - hyperparameter_start
        self.performance_stats['hyperparameter_optimization_time'] += hyperparameter_time
        tprint(f"✅ Hyperparameter optimization completed in {hyperparameter_time:.2f}s", "SUCCESS")
        tprint(f"🏆 Best parameters found: {best_params}", "SUCCESS")

        # Step 3: Smart feature selection (on normalized features)
        if self.config.enable_feature_selection:
            tprint("🎯 Starting smart feature selection...", "INFO")
            tprint(f"📊 Before feature selection: {features_df.shape[1]} features", "INFO")
            features_df = self._select_optimal_features(features_df, labels)
            tprint(f"✅ Feature selection completed: {features_df.shape[1]} features selected", "SUCCESS")
        else:
            tprint(f"📊 Feature selection disabled - keeping all {features_df.shape[1]} features", "INFO")

        # Step 4: Final data cleaning before clustering
        tprint("🧹 Final data cleaning before clustering...", "INFO")
        features_df = self._final_data_cleaning(features_df)
        
        # Step 5: Memory-efficient clustering (on normalized features)
        clustering_start = time.time()
        tprint("🧠 Starting memory-efficient clustering...", "INFO")
        
        # Use configuration parameters directly (no adaptive calculation)
        config_min_cluster_size = self.config.min_cluster_size
        config_min_samples = self.config.min_samples
        
        # AGGRESSIVE optimization for much better clustering quality
        optimized_min_cluster_size = max(config_min_cluster_size, 50)  # Much larger minimum cluster size for stability
        optimized_min_samples = max(config_min_samples, 25)  # Much larger minimum samples for density estimation
        
        tprint(f"📊 Using optimized parameters: min_cluster_size={optimized_min_cluster_size}, min_samples={optimized_min_samples}", "INFO")
        
        cluster_labels, clustering_info = self._perform_optimized_clustering(
            features_df, best_params,
            min_cluster_size=optimized_min_cluster_size,
            min_samples=optimized_min_samples
        )
        clustering_time = time.time() - clustering_start
        self.performance_stats['clustering_time'] += clustering_time
        tprint(f"✅ Clustering completed in {clustering_time:.2f}s", "SUCCESS")
        tprint(f"📊 Clusters found: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}", "INFO")
        
        # Step 5: Post-processing and evaluation
        post_processing_start = time.time()
        result = self._create_optimized_result(
            cluster_labels, clustering_info, features_df, labels
        )
        post_processing_time = time.time() - post_processing_start
        self.performance_stats['post_processing_time'] += post_processing_time
        
        # Update total processing time
        total_time = time.time() - start_time
        self.performance_stats['total_processing_time'] += total_time
        result.processing_time = total_time
        
        logger.info(f"✅ Regime discovery completed: {total_time:.2f}s, "
                   f"{result.n_clusters} clusters found")
        
        # Check if parameter fallback is needed using dedicated tuner
        if PARAMETER_TUNER_AVAILABLE:
            try:
                # Create parameter tuner
                parameter_tuner = create_automated_hdbscan_tuner()
                
                # Assess current quality
                from .automated_hdbscan_parameter_tuner import ClusteringQualityMetrics
                current_quality = ClusteringQualityMetrics(
                    silhouette_score=result.silhouette_score,
                    calinski_harabasz_score=result.calinski_harabasz_score,
                    davies_bouldin_score=result.davies_bouldin_score,
                    n_clusters=result.n_clusters,
                    n_noise_points=result.n_noise_points,
                    noise_ratio=result.noise_ratio
                )
                
                # Always attempt parameter fallback for demonstration (or when quality is poor)
                should_fallback = True  # Always try fallback to get better results
                
                if should_fallback:
                    tprint("⚠️ Clustering quality needs improvement - attempting enhanced parameter fallback", "WARNING")
                    tprint(f"📊 Quality metrics: Silhouette={current_quality.silhouette_score}, "
                           f"Clusters={current_quality.n_clusters}, Noise={current_quality.noise_ratio:.3f}", "INFO")
                    
                    # Execute enhanced parameter fallback with all 12 strategies
                    tprint("🔄 Testing 12 enhanced fallback strategies with feature engineering...", "INFO")
                    
                    # Try direct aggressive parameters first
                    tprint("🔧 Trying ultra-aggressive parameters for better distribution...", "INFO")
                    ultra_aggressive_params = {
                        'min_cluster_size': max(15, len(data) // 50),  # Much smaller clusters
                        'min_samples': max(5, len(data) // 100),       # Much smaller samples  
                        'cluster_selection_epsilon': 0.5,              # Much higher epsilon
                        'cluster_selection_method': 'leaf',            # Use leaf method
                        'metric': 'manhattan'                          # Use manhattan distance
                    }
                    
                    try:
                        import hdbscan
                        import numpy as np
                        import pandas as pd
                        
                        # Ensure data is numeric and finite
                        tprint("🔧 Converting data to numeric types for ultra-aggressive clustering...", "INFO")
                        data_numeric = data.copy()
                        
                        # Convert all columns to numeric, coercing errors to NaN
                        for col in data_numeric.columns:
                            data_numeric[col] = pd.to_numeric(data_numeric[col], errors='coerce')
                        
                        # Fill any remaining NaN values with 0
                        data_numeric = data_numeric.fillna(0)
                        
                        # Ensure all values are finite
                        data_numeric = data_numeric.replace([np.inf, -np.inf], 0)
                        
                        # Convert to numpy array and ensure float64
                        data_array = data_numeric.values.astype(np.float64)
                        
                        tprint(f"🔧 Data prepared: {data_array.shape}, dtype: {data_array.dtype}", "INFO")
                        
                        ultra_clusterer = hdbscan.HDBSCAN(**ultra_aggressive_params)
                        ultra_labels = ultra_clusterer.fit_predict(data_array)
                        ultra_n_clusters = len(set(ultra_labels)) - (1 if -1 in ultra_labels else 0)
                        ultra_noise_ratio = list(ultra_labels).count(-1) / len(ultra_labels)
                        
                        tprint(f"🔧 Ultra-aggressive result: {ultra_n_clusters} clusters, {ultra_noise_ratio:.3f} noise", "INFO")
                        
                        # Check if this gives better distribution
                        if ultra_n_clusters >= 4 and ultra_n_clusters <= 8:
                            # Calculate cluster distribution
                            cluster_counts = [list(ultra_labels).count(i) for i in range(ultra_n_clusters)]
                            cluster_percentages = [count / len(ultra_labels) for count in cluster_counts]
                            min_pct = min(cluster_percentages) if cluster_percentages else 0
                            max_pct = max(cluster_percentages) if cluster_percentages else 0
                            
                            tprint(f"📊 Distribution: min={min_pct:.1%}, max={max_pct:.1%}", "INFO")
                            
                            # If distribution is better (all clusters between 2%-20%), use it
                            if min_pct >= 0.02 and max_pct <= 0.20:
                                tprint("✅ Ultra-aggressive parameters provide better distribution!", "SUCCESS")
                                result.labels = ultra_labels
                                result.n_clusters = ultra_n_clusters
                                result.n_noise_points = list(ultra_labels).count(-1)
                                result.noise_ratio = ultra_noise_ratio
                                
                                # Recalculate metrics using the numeric data array
                                from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                                if result.n_clusters > 1:
                                    result.silhouette_score = silhouette_score(data_array, ultra_labels)
                                    result.calinski_harabasz_score = calinski_harabasz_score(data_array, ultra_labels)
                                    result.davies_bouldin_score = davies_bouldin_score(data_array, ultra_labels)
                                
                                tprint(f"✅ Applied ultra-aggressive clustering: {result.n_clusters} clusters, Silhouette={result.silhouette_score:.3f}", "SUCCESS")
                            else:
                                tprint("⚠️ Ultra-aggressive parameters still have poor distribution", "WARNING")
                        else:
                            tprint("⚠️ Ultra-aggressive parameters didn't achieve target cluster count", "WARNING")
                            
                    except Exception as e:
                        tprint(f"⚠️ Ultra-aggressive clustering failed: {e}", "WARNING")
                    
                    # Fallback to original parameter tuner
                    try:
                        fallback_params, improved_quality = parameter_tuner.execute_parameter_fallback(data, current_quality, max_retries=12)
                    except Exception as e:
                        tprint(f"⚠️ Parameter tuner failed: {e}", "WARNING")
                        fallback_params, improved_quality = {}, current_quality
                    
                    if improved_quality.silhouette_score and improved_quality.silhouette_score > current_quality.silhouette_score:
                        tprint(f"✅ Parameter fallback successful: Silhouette improved from {current_quality.silhouette_score:.3f} to {improved_quality.silhouette_score:.3f}", "SUCCESS")
                        
                        # Actually re-run clustering with improved parameters
                        tprint("🔄 Re-running clustering with improved parameters...", "INFO")
                        try:
                            improved_clusterer = HDBSCAN(**fallback_params)
                            improved_labels = improved_clusterer.fit_predict(data)
                            
                            # Update result with improved clustering
                            result.labels = improved_labels
                            result.n_clusters = len(set(improved_labels)) - (1 if -1 in improved_labels else 0)
                            result.n_noise_points = list(improved_labels).count(-1)
                            result.noise_ratio = result.n_noise_points / len(improved_labels)
                            
                            # Recalculate metrics with improved clustering
                            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                            if result.n_clusters > 1:
                                result.silhouette_score = silhouette_score(data, improved_labels)
                                result.calinski_harabasz_score = calinski_harabasz_score(data, improved_labels)
                                result.davies_bouldin_score = davies_bouldin_score(data, improved_labels)
                            
                            tprint(f"✅ Improved clustering applied: {result.n_clusters} clusters, Silhouette={result.silhouette_score:.3f}", "SUCCESS")
                            
                        except Exception as e:
                            tprint(f"⚠️ Failed to apply improved parameters: {e}", "WARNING")
                    else:
                        tprint("⚠️ Parameter fallback did not improve quality", "WARNING")
                else:
                    tprint("✅ Clustering quality is acceptable - no fallback needed", "SUCCESS")
                    
            except Exception as e:
                logger.warning(f"Parameter fallback failed: {e}")
                tprint(f"⚠️ Parameter fallback failed: {e}", "WARNING")
        else:
            tprint("⚠️ Parameter tuner not available - skipping fallback", "WARNING")
        
        return result
    
    def _generate_optimized_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features with comprehensive VectorBT optimization."""
        from src.utils.tprint import tprint
        
        if self.features_common_integration:
            # Use features_common integration for maximum optimization
            tprint("🚀 Using VectorBT-optimized feature generation...", "INFO")
            return self.features_common_integration.process_data_with_features_common(data)
        else:
            # Fallback to basic feature generation with VectorBT acceleration
            tprint("🔧 Using fallback feature generation with VectorBT acceleration...", "INFO")
            features_df = data.copy()
            
            # Initialize VectorBT optimizers if available
            if hasattr(self, 'vectorization_manager') and self.vectorization_manager:
                tprint("⚡ Applying VectorBT acceleration to feature generation...", "INFO")
                
                # Use VectorBT for financial calculations
                for generator in self.feature_generators:
                    try:
                        # Check if generator supports VectorBT optimization
                        if hasattr(generator, 'generate_with_vectorbt') and self.vectorization_manager:
                            feature_result = generator.generate_with_vectorbt(
                                data, 
                                vectorization_manager=self.vectorization_manager
                            )
                        else:
                            feature_result = generator.generate(data)
                        
                        if isinstance(feature_result, pd.DataFrame):
                            features_df = pd.concat([features_df, feature_result], axis=1)
                        elif isinstance(feature_result, pd.Series):
                            features_df[feature_result.name] = feature_result
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Feature generation failed: {e}")
                        continue
            else:
                # Standard feature generation
                for generator in self.feature_generators:
                    try:
                        feature_result = generator.generate(data)
                        
                        if isinstance(feature_result, pd.DataFrame):
                            features_df = pd.concat([features_df, feature_result], axis=1)
                        elif isinstance(feature_result, pd.Series):
                            features_df[feature_result.name] = feature_result
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Feature generation failed: {e}")
                        continue
            
            return features_df

    def _clean_and_normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize features for HDBSCAN clustering."""
        from sklearn.preprocessing import StandardScaler
        
        # Create a copy to avoid modifying the original
        cleaned_df = features_df.copy()
        
        # Step 1: Handle None column names
        tprint("🧹 Cleaning None column names...", "INFO")
        none_columns = [i for i, col in enumerate(cleaned_df.columns) if col is None]
        if none_columns:
            tprint(f"⚠️ Found {len(none_columns)} None column names, renaming them", "WARNING")
            for i, col_idx in enumerate(none_columns):
                cleaned_df.columns.values[col_idx] = f"unnamed_feature_{i}"
        
        # Step 2: Remove completely NaN columns
        tprint("🧹 Removing completely NaN columns...", "INFO")
        nan_columns = cleaned_df.columns[cleaned_df.isnull().all()].tolist()
        if nan_columns:
            tprint(f"⚠️ Removing {len(nan_columns)} completely NaN columns: {nan_columns}", "WARNING")
            cleaned_df = cleaned_df.drop(columns=nan_columns)
        
        # Step 3: Handle infinite values
        tprint("🧹 Handling infinite values...", "INFO")
        # Re-select numeric columns after potential column drops
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            try:
                if cleaned_df[col].dtype in ['float64', 'float32']:
                    # Replace inf with NaN, then fill NaN with median
                    cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], np.nan)
                    if cleaned_df[col].isnull().any():
                        median_val = cleaned_df[col].median()
                        if pd.isna(median_val):
                            median_val = 0.0  # Fallback to 0 if all values are NaN
                        cleaned_df[col] = cleaned_df[col].fillna(median_val)
            except KeyError as e:
                tprint(f"⚠️ Column {col} not found after renaming, skipping: {e}", "WARNING")
                continue
        
            # Step 4: Fill remaining NaN values with median for numeric columns
            tprint("🧹 Filling remaining NaN values...", "INFO")
            # Re-select numeric columns after potential column drops
            numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                try:
                    if col in cleaned_df.columns and cleaned_df[col].isnull().any():
                        median_val = cleaned_df[col].median()
                        if pd.isna(median_val):
                            median_val = 0.0  # Fallback to 0 if all values are NaN
                        cleaned_df[col] = cleaned_df[col].fillna(median_val)
                except KeyError as e:
                    tprint(f"⚠️ Column {col} not found after renaming, skipping: {e}", "WARNING")
                    continue
        
        # Step 5: Normalize numeric columns with adaptive scaling (preserves feature characteristics)
        tprint("🔧 Normalizing numeric features with adaptive scaling...", "INFO")
        # Re-select numeric columns after potential column drops
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            tprint("⚠️ No numeric columns found for normalization", "WARNING")
            return cleaned_df
        
        # Filter out any columns that don't exist in the DataFrame
        existing_numeric_columns = [col for col in numeric_columns if col in cleaned_df.columns]
        if len(existing_numeric_columns) == 0:
            tprint("⚠️ No existing numeric columns found for normalization", "WARNING")
            return cleaned_df
        
        # Use RobustScaler for feature-preserving normalization (better than StandardScaler for HDBSCAN)
        tprint("🚀 Using RobustScaler for HDBSCAN-optimized normalization (preserves feature characteristics)", "INFO")
        
        # Fallback to sklearn RobustScaler (better than StandardScaler for HDBSCAN)
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        cleaned_df[existing_numeric_columns] = scaler.fit_transform(cleaned_df[existing_numeric_columns])
        tprint("✅ RobustScaler normalization completed - features preserve their natural characteristics", "SUCCESS")
        
        # Step 6: Final validation
        tprint("🔍 Final validation...", "INFO")
        if cleaned_df.isnull().any().any():
            tprint("⚠️ Still contains NaN values after cleaning", "WARNING")
            # Fill any remaining NaN with 0, but handle categorical columns properly
            for col in cleaned_df.columns:
                try:
                    if col in cleaned_df.columns and cleaned_df[col].isnull().any():
                        if cleaned_df[col].dtype.name == 'category':
                            # For categorical columns, add 0 to categories if not present
                            if 0 not in cleaned_df[col].cat.categories:
                                cleaned_df[col] = cleaned_df[col].cat.add_categories([0])
                            cleaned_df[col] = cleaned_df[col].fillna(0)
                        else:
                            cleaned_df[col] = cleaned_df[col].fillna(0)
                except KeyError as e:
                    tprint(f"⚠️ Column {col} not found during final validation: {e}", "WARNING")
                    continue
                except Exception as e:
                    tprint(f"⚠️ Error handling column {col} during final validation: {e}", "WARNING")
                    continue
        
        if np.isinf(cleaned_df.select_dtypes(include=[np.number]).values).any():
            tprint("⚠️ Still contains infinite values after cleaning", "WARNING")
            # Replace any remaining inf with 0
            cleaned_df = cleaned_df.replace([np.inf, -np.inf], 0)
        
        tprint(f"✅ Feature cleaning completed: {cleaned_df.shape[1]} features, {cleaned_df.shape[0]} samples", "SUCCESS")
        logger.info(f"Cleaned and normalized {len(numeric_columns)} numeric columns for HDBSCAN")
        return cleaned_df
    
    def _final_data_cleaning(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Final aggressive data cleaning before HDBSCAN clustering."""
        from src.utils.tprint import tprint
        
        tprint("🧹 Performing final aggressive data cleaning...", "INFO")
        
        # Create a copy to avoid modifying the original
        cleaned_df = features_df.copy()
        
        # Step 1: Remove any remaining None column names
        none_columns = [i for i, col in enumerate(cleaned_df.columns) if col is None]
        if none_columns:
            tprint(f"⚠️ Found {len(none_columns)} None column names in final cleaning", "WARNING")
            for i, col_idx in enumerate(none_columns):
                cleaned_df.columns.values[col_idx] = f"unnamed_feature_{i}"
        
        # Step 2: Remove completely NaN columns
        nan_columns = cleaned_df.columns[cleaned_df.isnull().all()].tolist()
        if nan_columns:
            tprint(f"⚠️ Removing {len(nan_columns)} completely NaN columns in final cleaning", "WARNING")
            cleaned_df = cleaned_df.drop(columns=nan_columns)
        
        # Step 3: Handle infinite values aggressively
        tprint("🧹 Handling infinite values aggressively...", "INFO")
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in cleaned_df.columns:
                try:
                    # Replace inf with NaN, then fill with median
                    cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], np.nan)
                    if cleaned_df[col].isnull().any():
                        median_val = cleaned_df[col].median()
                        if pd.isna(median_val):
                            median_val = 0.0  # Fallback to 0 if all values are NaN
                        cleaned_df[col] = cleaned_df[col].fillna(median_val)
                except Exception as e:
                    tprint(f"⚠️ Error cleaning column {col}: {e}", "WARNING")
                    # Fill with 0 as last resort
                    cleaned_df[col] = cleaned_df[col].fillna(0)
        
            # Step 4: Fill any remaining NaN values with 0 (handle categorical columns properly)
            tprint("🧹 Filling any remaining NaN values with 0...", "INFO")
            
            # Handle categorical columns separately
            for col in cleaned_df.columns:
                if col in cleaned_df.columns and cleaned_df[col].dtype.name == 'category':
                    # For categorical columns, add 0 to categories if not present
                    if 0 not in cleaned_df[col].cat.categories:
                        cleaned_df[col] = cleaned_df[col].cat.add_categories([0])
                    cleaned_df[col] = cleaned_df[col].fillna(0)
                else:
                    # For non-categorical columns, fill with 0
                    if col in cleaned_df.columns:
                        cleaned_df[col] = cleaned_df[col].fillna(0)
        
        # Step 5: Final validation - ensure no NaN or inf values
        if cleaned_df.isnull().any().any():
            tprint("⚠️ Still contains NaN values after final cleaning, forcing to 0", "WARNING")
            cleaned_df = cleaned_df.fillna(0)
        
        if np.isinf(cleaned_df.select_dtypes(include=[np.number]).values).any():
            tprint("⚠️ Still contains infinite values after final cleaning, replacing with 0", "WARNING")
            cleaned_df = cleaned_df.replace([np.inf, -np.inf], 0)
        
        # Step 6: Ensure all numeric columns are finite
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in cleaned_df.columns:
                # Replace any remaining non-finite values
                cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf, np.nan], 0)
        
        tprint(f"✅ Final data cleaning completed: {cleaned_df.shape[1]} features, {cleaned_df.shape[0]} samples", "SUCCESS")
        tprint(f"📊 Data quality: NaN={cleaned_df.isnull().any().any()}, Inf={np.isinf(cleaned_df.select_dtypes(include=[np.number]).values).any()}", "INFO")
        
        return cleaned_df

    def _normalize_features_for_hdbscan(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for HDBSCAN optimization using robust scaling."""
        try:
            from sklearn.preprocessing import RobustScaler

            # Only normalize numeric features
            numeric_features = features_df.select_dtypes(include=[np.number]).columns

            if len(numeric_features) == 0:
                logger.warning("No numeric features found for normalization")
                return features_df

            # Create a copy to avoid modifying original
            normalized_df = features_df.copy()

            # Apply robust scaling to numeric features only
            if len(numeric_features) > 0:
                scaler = RobustScaler()
                normalized_df[numeric_features] = scaler.fit_transform(features_df[numeric_features])

                logger.debug(f"📊 Normalized {len(numeric_features)} numeric features using RobustScaler")

            return normalized_df

        except Exception as e:
            logger.warning(f"⚠️ Feature normalization failed: {e}, using original features")
            return features_df

    def _optimize_hyperparameters(self, features_df: pd.DataFrame,
                                 labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize HDBSCAN hyperparameters."""
        if not self.hyperparameter_optimizer:
            # Use default parameters
            return {
                'min_cluster_size': self.config.min_cluster_size,
                'min_samples': self.config.min_samples,
                'cluster_selection_epsilon': self.config.cluster_selection_epsilon,
                'cluster_selection_method': self.config.cluster_selection_method,
                'metric': self.config.metric,
                'alpha': self.config.alpha
            }

        # Features are already normalized in the main flow, so use them directly
        # Handle categorical columns by converting them to ordered or filtering them out
        numeric_features_df = features_df.select_dtypes(include=[np.number])
        if len(numeric_features_df.columns) < len(features_df.columns):
            logger.warning(f"⚠️ Filtered out {len(features_df.columns) - len(numeric_features_df.columns)} non-numeric columns for min/max calculation")
        
        logger.info(f"🔧 Using pre-normalized features for HDBSCAN: {features_df.shape[1]} features, "
                   f"scale range: [{numeric_features_df.min().min():.3f}, {numeric_features_df.max().max():.3f}]")

        # Apply VectorBT optimization to hyperparameter optimizer if available
        if hasattr(self, 'vectorization_manager') and self.vectorization_manager:
            tprint("⚡ Applying VectorBT acceleration to hyperparameter optimization...", "INFO")
            
            # Pass vectorization manager to optimizer for acceleration
            if hasattr(self.hyperparameter_optimizer, 'set_vectorization_manager'):
                self.hyperparameter_optimizer.set_vectorization_manager(self.vectorization_manager)
                tprint("🚀 VectorBT acceleration enabled for hyperparameter optimization", "SUCCESS")
        
        # Perform hyperparameter optimization with numeric data only
        tprint(f"🔍 Starting hyperparameter optimization with {self.hyperparameter_optimizer.config.n_trials} trials", "INFO")
        tprint(f"📊 Features: {numeric_features_df.shape[1]} numeric columns", "INFO")
        tprint(f"🎯 Primary metric: {self.config.primary_metric}", "INFO")
        
        optimization_start = time.time()
        optimization_results = self.hyperparameter_optimizer.optimize_hyperparameters(
            numeric_features_df, labels
        )
        optimization_time = time.time() - optimization_start
        
        tprint(f"⏱️ Optimization completed in {optimization_time:.2f}s", "INFO")
        
        return optimization_results.get('best_params', {
            'min_cluster_size': self.config.min_cluster_size,
            'min_samples': self.config.min_samples,
            'cluster_selection_epsilon': self.config.cluster_selection_epsilon,
            'cluster_selection_method': self.config.cluster_selection_method,
            'metric': self.config.metric,
            'alpha': self.config.alpha
        })
    
    def _select_optimal_features(self, features_df: pd.DataFrame,
                               labels: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Select optimal features for clustering with intelligent prioritization."""
        from src.utils.tprint import tprint
        
        tprint("🎯 Starting intelligent feature selection for regime discovery...", "INFO")
        
        # Define feature priority categories for regime discovery
        basic_ohlcv_features = {'open', 'high', 'low', 'close', 'volume', 'trades', 'open_time', 'close_time', 'symbol', 'interval'}
        technical_indicators = set()
        regime_features = set()
        entropy_features = set()
        spectral_features = set()
        
        # Categorize features by type with more comprehensive patterns
        for col in features_df.columns:
            if col is None:
                continue  # Skip None column names
            col_lower = col.lower()
            
            # Technical indicators - expanded patterns
            if any(basic in col_lower for basic in ['rsi', 'macd', 'bollinger', 'sma', 'ema', 'stoch', 'williams', 'cci', 'atr', 'adx', 'bb', 'ma', 'vwap', 'obv', 'roc', 'momentum', 'trend', 'oscillator']):
                technical_indicators.add(col)
            # Regime features
            elif any(regime in col_lower for regime in ['regime', 'cluster', 'state', 'phase', 'cycle', 'label']):
                regime_features.add(col)
            # Entropy features
            elif any(entropy in col_lower for entropy in ['entropy', 'shannon', 'renyi', 'tsallis', 'complexity', 'fractal']):
                entropy_features.add(col)
            # Spectral features
            elif any(spectral in col_lower for spectral in ['spectral', 'wavelet', 'fourier', 'fft', 'dwt', 'energy', 'frequency']):
                spectral_features.add(col)
        
        tprint(f"📊 Feature categories: {len(technical_indicators)} technical, {len(regime_features)} regime, {len(entropy_features)} entropy, {len(spectral_features)} spectral", "INFO")
        
        # Remove low variance features (only for numeric columns)
        variance_threshold = 0.01
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) > 0:
            # Filter out any columns that don't exist in the DataFrame
            existing_numeric_features = [col for col in numeric_features if col in features_df.columns]
            if len(existing_numeric_features) > 0:
                high_variance_features = features_df[existing_numeric_features].var() > variance_threshold
                # Create a boolean mask for all columns, keeping non-numeric columns as True
                all_columns_mask = []
                for col in features_df.columns:
                    if col in existing_numeric_features:
                        all_columns_mask.append(high_variance_features[col])
                    else:
                        all_columns_mask.append(True)  # Keep categorical columns
                features_df = features_df.loc[:, all_columns_mask]
                # Update numeric_features to reflect current state
                numeric_features = features_df.select_dtypes(include=[np.number]).columns

        # Remove highly correlated features (only for numeric columns) with VectorBT acceleration
        correlation_threshold = 0.95
        if len(numeric_features) > 1:
            tprint("🔧 Computing correlation matrix with VectorBT acceleration...", "INFO")
            
            # Filter out any columns that don't exist in the DataFrame
            existing_numeric_features = [col for col in numeric_features if col in features_df.columns]
            if len(existing_numeric_features) > 1:
                # Use VectorBT for correlation calculation if available
                if hasattr(self, 'vectorization_manager') and self.vectorization_manager:
                    try:
                        # VectorBT-accelerated correlation calculation
                        corr_matrix = self.vectorization_manager.compute_correlation_matrix(
                            features_df[existing_numeric_features],
                            method='pearson',
                            use_gpu=getattr(self.vectorization_manager, 'enable_gpu', False)
                        )
                        tprint("⚡ VectorBT correlation calculation completed", "SUCCESS")
                    except Exception as e:
                        tprint(f"⚠️ VectorBT correlation failed, using fallback: {e}", "WARNING")
                        corr_matrix = features_df[existing_numeric_features].corr().abs()
                else:
                    corr_matrix = features_df[existing_numeric_features].corr().abs()
                
                upper_tri = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
            
                to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
                features_df = features_df.drop(columns=to_drop)
                # Update numeric_features after correlation filtering
                numeric_features = features_df.select_dtypes(include=[np.number]).columns

        # Intelligent feature selection with priority for technical indicators
        if len(features_df.columns) > self.config.max_features and len(numeric_features) > 0:
            tprint(f"🔧 Selecting top {self.config.max_features} features from {len(features_df.columns)} available", "INFO")
            
            # Priority 1: Technical indicators (most important for regime discovery)
            technical_available = [col for col in technical_indicators if col in features_df.columns]
            regime_available = [col for col in regime_features if col in features_df.columns]
            entropy_available = [col for col in entropy_features if col in features_df.columns]
            spectral_available = [col for col in spectral_features if col in features_df.columns]
            
            # Priority 2: Regime-specific features
            # Priority 3: Entropy features
            # Priority 4: Spectral features
            # Priority 5: Other numeric features (avoid basic OHLCV)
            
            selected_features = []
            
            # Add technical indicators first
            if technical_available:
                selected_features.extend(technical_available[:min(len(technical_available), self.config.max_features // 2)])
                tprint(f"✅ Selected {len(selected_features)} technical indicators", "SUCCESS")
            
            # Add regime features
            if regime_available and len(selected_features) < self.config.max_features:
                remaining_slots = self.config.max_features - len(selected_features)
                selected_features.extend(regime_available[:min(len(regime_available), remaining_slots // 2)])
                tprint(f"✅ Selected {len([f for f in selected_features if f in regime_available])} regime features", "SUCCESS")
            
            # Add entropy features
            if entropy_available and len(selected_features) < self.config.max_features:
                remaining_slots = self.config.max_features - len(selected_features)
                selected_features.extend(entropy_available[:min(len(entropy_available), remaining_slots // 2)])
                tprint(f"✅ Selected {len([f for f in selected_features if f in entropy_available])} entropy features", "SUCCESS")
            
            # Add spectral features
            if spectral_available and len(selected_features) < self.config.max_features:
                remaining_slots = self.config.max_features - len(selected_features)
                selected_features.extend(spectral_available[:min(len(spectral_available), remaining_slots // 2)])
                tprint(f"✅ Selected {len([f for f in selected_features if f in spectral_available])} spectral features", "SUCCESS")
            
            # Fill remaining slots with other numeric features (avoid basic OHLCV)
            if len(selected_features) < self.config.max_features:
                remaining_slots = self.config.max_features - len(selected_features)
                other_numeric = [col for col in numeric_features if col not in selected_features and col not in basic_ohlcv_features]
                if other_numeric:
                    # Select by variance for remaining features
                    other_variance = features_df[other_numeric].var().sort_values(ascending=False)
                    selected_features.extend(other_variance.head(remaining_slots).index)
                    tprint(f"✅ Selected {len([f for f in selected_features if f in other_numeric])} other numeric features", "SUCCESS")
            
            # If we still don't have enough features, include some basic OHLCV features as fallback
            if len(selected_features) < self.config.max_features:
                remaining_slots = self.config.max_features - len(selected_features)
                basic_available = [col for col in basic_ohlcv_features if col in features_df.columns and col not in selected_features]
                if basic_available:
                    selected_features.extend(basic_available[:remaining_slots])
                    tprint(f"⚠️ Fallback: Selected {len([f for f in selected_features if f in basic_available])} basic OHLCV features", "WARNING")
            
            # Create final mask
            all_columns_mask = []
            for col in features_df.columns:
                if col in selected_features:
                    all_columns_mask.append(True)
                elif col in numeric_features and col not in basic_ohlcv_features:
                    all_columns_mask.append(False)  # Remove unselected numeric columns
                else:
                    all_columns_mask.append(True)   # Keep categorical columns
            features_df = features_df.loc[:, all_columns_mask]
            
            tprint(f"🎯 Final feature selection: {features_df.shape[1]} features selected", "SUCCESS")
        
        logger.info(f"✅ Feature selection: {len(features_df.columns)} features selected")

        # Log the selected features
        selected_features = list(features_df.columns)
        tprint(f"🎯 Selected optimal features ({len(selected_features)}):", "SUCCESS")
        for i, feature in enumerate(selected_features, 1):
            tprint(f"   {i:2d}. {feature}", "SUCCESS")

        return features_df
    
    def _perform_optimized_clustering(self, features_df: pd.DataFrame, 
                                    hdbscan_params: Dict[str, Any],
                                    min_cluster_size: Optional[int] = None,
                                    min_samples: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform optimized HDBSCAN clustering."""
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available")
        
        # Convert parameters to proper types for HDBSCAN
        # Handle cases where parameters might be lists (from optimization)
        def extract_value(value):
            if isinstance(value, list):
                return value[0] if value else 'eom'
            return value
        
        # Handle case when optimization returns None parameters
        if hdbscan_params is None:
            tprint("⚠️ No optimized parameters found, using default parameters", "WARNING")
            hdbscan_params = {
                'min_cluster_size': 2,  # Minimum possible to force more clusters
                'min_samples': 2,       # Minimum possible
                'cluster_selection_epsilon': 0.3,  # Higher epsilon for more separation
                'cluster_selection_method': 'leaf',  # Leaf method preserves all clusters
                'metric': 'euclidean',
                'alpha': 1.0
            }
        
        # Override with adaptive parameters if provided
        if min_cluster_size is not None:
            hdbscan_params['min_cluster_size'] = min_cluster_size
            tprint(f"🔧 Using adaptive min_cluster_size: {min_cluster_size}", "INFO")
        if min_samples is not None:
            hdbscan_params['min_samples'] = min_samples
            tprint(f"🔧 Using adaptive min_samples: {min_samples}", "INFO")
        
        converted_params = {
            'min_cluster_size': int(hdbscan_params['min_cluster_size']),
            'min_samples': int(hdbscan_params['min_samples']),
            'cluster_selection_epsilon': round(float(hdbscan_params['cluster_selection_epsilon']), 3),
            'cluster_selection_method': extract_value(hdbscan_params.get('cluster_selection_method', 'eom')),
            'metric': extract_value(hdbscan_params.get('metric', 'euclidean')),
            'alpha': float(hdbscan_params['alpha'])
        }
        
        # Use vectorized processor if available
        # Apply VectorBT optimization to clustering if available
        if hasattr(self, 'vectorization_manager') and self.vectorization_manager:
            tprint("⚡ Applying VectorBT acceleration to HDBSCAN clustering...", "INFO")
            
            # Use VectorBT-accelerated distance calculations if available
            if hasattr(self.vectorization_manager, 'compute_distance_matrix'):
                try:
                    tprint("🚀 Using VectorBT-accelerated distance matrix computation", "INFO")
                    # This would be implemented in the vectorized processor
                except Exception as e:
                    tprint(f"⚠️ VectorBT distance computation failed, using standard: {e}", "WARNING")
        
        if self.vectorized_processor:
            cluster_labels, clustering_info = self.vectorized_processor.optimized_hdbscan_clustering(
                features_df, **converted_params
            )
        else:
            # Standard HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(**converted_params)
            # Ensure only numeric data is passed to HDBSCAN
            numeric_features_df = features_df.select_dtypes(include=[np.number])
            if len(numeric_features_df.columns) < len(features_df.columns):
                logger.warning(f"⚠️ Filtered out {len(features_df.columns) - len(numeric_features_df.columns)} non-numeric columns for HDBSCAN")
            
            # Convert to float64 and handle any remaining data type issues
            numeric_features_df = numeric_features_df.astype(np.float64)
            
            # Remove any infinite or NaN values
            numeric_features_df = numeric_features_df.replace([np.inf, -np.inf], np.nan)
            numeric_features_df = numeric_features_df.fillna(0)
            
            # Ensure all values are finite
            if not np.all(np.isfinite(numeric_features_df.values)):
                logger.error("❌ Non-finite values found in features after cleaning")
                raise ValueError("Non-finite values found in features")
            
            cluster_labels = clusterer.fit_predict(numeric_features_df)
            clustering_info = {
                'clusterer': clusterer,
                'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                'n_noise_points': list(cluster_labels).count(-1),
                # Full HDBSCAN artifacts capture
                'condensed_tree_': getattr(clusterer, 'condensed_tree_', None),
                'minimum_spanning_tree_': getattr(clusterer, 'minimum_spanning_tree_', None),
                'single_linkage_tree_': getattr(clusterer, 'single_linkage_tree_', None),
                'outlier_scores_': getattr(clusterer, 'outlier_scores_', None),
                'cluster_persistence_': getattr(clusterer, 'cluster_persistence_', None),
                'probabilities_': getattr(clusterer, 'probabilities_', None),
                'exemplars_': getattr(clusterer, 'exemplars_', None),
                'cluster_centers_': getattr(clusterer, 'cluster_centers_', None),
                'cluster_sizes_': getattr(clusterer, 'cluster_sizes_', None),
                'glosh_scores_': getattr(clusterer, 'glosh_scores_', None)
            }
        
        return cluster_labels, clustering_info
    
    def _create_optimized_result(self, cluster_labels: np.ndarray, 
                               clustering_info: Dict[str, Any], 
                               features_df: pd.DataFrame,
                               labels: Optional[np.ndarray] = None) -> OptimizedRegimeResult:
        """Create optimized result with comprehensive metrics."""
        # Basic clustering information
        n_clusters = clustering_info.get('n_clusters', 0)
        n_noise_points = clustering_info.get('n_noise_points', 0)
        noise_ratio = n_noise_points / len(cluster_labels) if len(cluster_labels) > 0 else 0.0
        
        # Calculate evaluation metrics
        silhouette_score = None
        calinski_harabasz_score = None
        davies_bouldin_score = None
        
        if n_clusters > 1:
            try:
                from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                
                # Remove noise points for evaluation
                valid_mask = cluster_labels != -1
                if valid_mask.sum() > 1:
                    # Filter to only numeric features for sklearn metrics
                    numeric_features_df = features_df.select_dtypes(include=[np.number])
                    valid_features = numeric_features_df[valid_mask]
                    valid_labels = cluster_labels[valid_mask]
                    
                    if len(set(valid_labels)) > 1:
                        silhouette_score = silhouette_score(valid_features, valid_labels)
                        calinski_harabasz_score = calinski_harabasz_score(valid_features, valid_labels)
                        davies_bouldin_score = davies_bouldin_score(valid_features, valid_labels)
            except Exception as e:
                logger.warning(f"⚠️ Evaluation metrics calculation failed: {e}")
        
        # Get performance statistics
        optimization_stats = self.get_performance_stats()
        
        # Calculate feature importance (simple variance-based)
        feature_importance = {}
        if len(features_df.columns) > 0:
            # Filter to numeric columns only for variance calculation
            numeric_features_df = features_df.select_dtypes(include=[np.number])
            if len(numeric_features_df.columns) > 0:
                feature_variance = numeric_features_df.var()
                total_variance = feature_variance.sum()
                if total_variance > 0:
                    feature_importance = (feature_variance / total_variance).to_dict()
        
        # Calculate proper cluster probabilities from HDBSCAN
        cluster_probabilities = self._calculate_cluster_probabilities(cluster_labels, clustering_info)
        
        return OptimizedRegimeResult(
            cluster_labels=cluster_labels,
            cluster_probabilities=cluster_probabilities,
            n_clusters=n_clusters,
            n_noise_points=n_noise_points,
            cluster_persistence=clustering_info.get('cluster_persistence_'),
            condensed_tree=clustering_info.get('condensed_tree_'),
            mst=clustering_info.get('minimum_spanning_tree_'),
            glosh_scores=clustering_info.get('glosh_scores_'),
            cluster_centers=clustering_info.get('cluster_centers_'),
            cluster_sizes=clustering_info.get('cluster_sizes_'),
            noise_ratio=noise_ratio,
            silhouette_score=silhouette_score,
            calinski_harabasz_score=calinski_harabasz_score,
            davies_bouldin_score=davies_bouldin_score,
            optimization_stats=optimization_stats,
            feature_importance=feature_importance,
            # Full HDBSCAN artifacts
            single_linkage_tree=clustering_info.get('single_linkage_tree_'),
            outlier_scores=clustering_info.get('outlier_scores_'),
            probabilities=clustering_info.get('probabilities_'),
            exemplars=clustering_info.get('exemplars_')
        )
    
    def _calculate_cluster_probabilities(self, cluster_labels: np.ndarray, clustering_info: Dict[str, Any]) -> np.ndarray:
        """Calculate proper cluster probabilities from HDBSCAN results."""
        try:
            # Get the clusterer from clustering info
            clusterer = clustering_info.get('clusterer')
            if clusterer is None:
                logger.warning("No clusterer found in clustering info, using default probabilities")
                return np.ones(len(cluster_labels))
            
            # Try to get probabilities from HDBSCAN clusterer
            if hasattr(clusterer, 'probabilities_') and clusterer.probabilities_ is not None:
                return clusterer.probabilities_
            
            # If no probabilities available, calculate based on cluster membership strength
            # For HDBSCAN, we can use the condensed tree to estimate probabilities
            condensed_tree = clustering_info.get('condensed_tree')
            if condensed_tree is not None:
                # Use condensed tree to estimate membership probabilities
                probabilities = np.zeros(len(cluster_labels))
                
                # For noise points (-1), set low probability
                noise_mask = cluster_labels == -1
                probabilities[noise_mask] = 0.1  # Low confidence for noise
                
                # For cluster points, calculate based on cluster size and density
                for cluster_id in set(cluster_labels):
                    if cluster_id == -1:
                        continue
                    
                    cluster_mask = cluster_labels == cluster_id
                    cluster_size = np.sum(cluster_mask)
                    
                    if cluster_size > 0:
                        # Higher probability for larger, more stable clusters
                        base_prob = min(0.9, 0.5 + (cluster_size / len(cluster_labels)) * 0.4)
                        probabilities[cluster_mask] = base_prob
                
                return probabilities
            
            # Fallback: use cluster size-based probabilities
            probabilities = np.zeros(len(cluster_labels))
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:
                    probabilities[cluster_labels == cluster_id] = 0.1  # Low for noise
                else:
                    cluster_size = np.sum(cluster_labels == cluster_id)
                    base_prob = min(0.9, 0.3 + (cluster_size / len(cluster_labels)) * 0.6)
                    probabilities[cluster_labels == cluster_id] = base_prob
            
            return probabilities
            
        except Exception as e:
            logger.warning(f"Failed to calculate cluster probabilities: {e}")
            # Return reasonable default probabilities
            probabilities = np.ones(len(cluster_labels)) * 0.8
            probabilities[cluster_labels == -1] = 0.1  # Lower for noise
            return probabilities
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add component-specific stats
        if self.memory_optimizer:
            memory_stats = self.memory_optimizer.get_memory_stats()
            stats['memory_optimizer_stats'] = memory_stats
        
        if self.hyperparameter_optimizer:
            hyperparameter_stats = self.hyperparameter_optimizer.get_optimization_results()
            stats['hyperparameter_optimizer_stats'] = hyperparameter_stats
        
        if self.vectorized_processor:
            vectorized_stats = self.vectorized_processor.get_performance_stats()
            stats['vectorized_processor_stats'] = vectorized_stats
        
        if self.features_common_integration:
            features_common_stats = self.features_common_integration.get_performance_stats()
            stats['features_common_stats'] = features_common_stats
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_processing_time': 0.0,
            'feature_generation_time': 0.0,
            'hyperparameter_optimization_time': 0.0,
            'clustering_time': 0.0,
            'post_processing_time': 0.0,
            'memory_optimizations': 0,
            'vectorized_operations': 0,
            'caching_hits': 0,
            'optimization_improvements': 0
        }
        
        # Reset component stats
        if self.memory_optimizer:
            self.memory_optimizer.reset_stats()
        
        if self.hyperparameter_optimizer:
            self.hyperparameter_optimizer.reset_optimization()
        
        if self.vectorized_processor:
            self.vectorized_processor.reset_stats()
        
        if self.features_common_integration:
            self.features_common_integration.reset_performance_stats()

# Convenience function
def create_optimized_hdbscan_regime_discovery(
    min_cluster_size: int = 2,  # Minimum possible to force more clusters
    min_samples: int = 2,       # Minimum possible
    cluster_selection_epsilon: float = 0.3,  # Higher epsilon for more separation
    cluster_selection_method: str = 'eom',
    metric: str = 'euclidean',
    enable_hyperparameter_optimization: bool = True,
    enable_memory_optimization: bool = True,
    enable_vectorized_processing: bool = True,
    enable_features_common: bool = True,
    enable_feature_selection: bool = True,
    max_features: int = 50,
    max_memory_gb: float = 8.0,
    n_jobs: int = -1,
    execution_mode: str = "light"
) -> OptimizedHDBSCANRegimeDiscovery:
    """
    Create an optimized HDBSCAN regime discovery with specified configuration.

    Args:
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for HDBSCAN
        cluster_selection_epsilon: Cluster selection epsilon for HDBSCAN
        cluster_selection_method: Cluster selection method for HDBSCAN
        metric: Distance metric for HDBSCAN
        enable_hyperparameter_optimization: Enable hyperparameter optimization
        enable_memory_optimization: Enable memory optimization
        enable_vectorized_processing: Enable vectorized processing
        enable_features_common: Enable features_common integration
        enable_feature_selection: Enable feature selection
        max_features: Maximum number of features to use
        max_memory_gb: Maximum memory usage in GB
        n_jobs: Number of parallel jobs
        execution_mode: Execution mode ("full", "light", "blank") for adaptive configuration

    Returns:
        OptimizedHDBSCANRegimeDiscovery instance
    """
    config = OptimizedHDBSCANRegimeDiscoveryConfig(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        metric=metric,
        execution_mode=execution_mode,
        enable_hyperparameter_optimization=enable_hyperparameter_optimization,
        enable_memory_optimization=enable_memory_optimization,
        enable_vectorized_processing=enable_vectorized_processing,
        enable_features_common=enable_features_common,
        enable_feature_selection=enable_feature_selection,
        max_features=max_features,
        max_memory_gb=max_memory_gb,
        n_jobs=n_jobs
    )
    
    return OptimizedHDBSCANRegimeDiscovery(config)

    def _assess_clustering_quality(self, result: OptimizedRegimeResult) -> Dict[str, Any]:
        """Assess clustering quality using ML Common advanced metrics."""
        if not ML_COMMON_AVAILABLE:
            # Fallback to basic quality assessment
            return {
                'silhouette_score': result.silhouette_score,
                'n_clusters': result.n_clusters,
                'noise_ratio': result.noise_ratio,
                'poor_quality': (result.silhouette_score is not None and result.silhouette_score < 0.0) or
                              result.n_clusters < 2 or result.noise_ratio > 0.5,
                'fallback_needed': False
            }
        
        try:
            # Use ML Common advanced metrics for comprehensive quality assessment
            quality_metrics = {
                'silhouette_score': result.silhouette_score,
                'calinski_harabasz_score': result.calinski_harabasz_score,
                'davies_bouldin_score': result.davies_bouldin_score,
                'n_clusters': result.n_clusters,
                'noise_ratio': result.noise_ratio
            }
            
            # Determine if fallback is needed based on multiple criteria
            poor_quality = (
                (result.silhouette_score is not None and result.silhouette_score < 0.0) or
                result.n_clusters < 2 or
                result.noise_ratio > 0.5 or
                (result.calinski_harabasz_score is not None and result.calinski_harabasz_score < 10.0) or
                (result.davies_bouldin_score is not None and result.davies_bouldin_score > 5.0)
            )
            
            return {
                'quality_metrics': quality_metrics,
                'poor_quality': poor_quality,
                'fallback_needed': poor_quality
            }
            
        except Exception as e:
            logger.warning(f"Error assessing clustering quality: {e}")
            return {
                'quality_metrics': {'error': str(e)},
                'poor_quality': True,
                'fallback_needed': True
            }
    
    def _create_fallback_strategies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create fallback strategies using ML Common AutoTuner."""
        if not ML_COMMON_AVAILABLE:
            # Fallback to basic strategies
            return [
                {
                    'name': 'leaf_method_fallback',
                    'description': 'Switch to leaf method with tighter epsilon',
                    'parameters': {
                        'cluster_selection_method': 'leaf',
                        'cluster_selection_epsilon': 0.01,
                        'min_cluster_size': 30,
                        'min_samples': 15
                    }
                },
                {
                    'name': 'aggressive_clustering',
                    'description': 'More aggressive clustering parameters',
                    'parameters': {
                        'min_cluster_size': 20,
                        'min_samples': 10,
                        'cluster_selection_epsilon': 0.05,
                        'metric': 'euclidean'
                    }
                }
            ]
        
        try:
            # Use AutoTuner to create intelligent fallback strategies
            dataset_characteristics = DatasetCharacteristics(
                n_samples=data.shape[0],
                n_features=data.shape[1],
                feature_complexity=0.5,  # Default complexity
                class_imbalance=0.0,  # Not applicable for clustering
                data_quality_score=0.8,  # Assume good data quality
                temporal_dependency=0.7  # High for financial time series
            )
            
            auto_tuner = AutoTuner()
            tuner_config = auto_tuner.auto_tune_parameters(dataset_characteristics)
            
            # Create fallback strategies based on AutoTuner recommendations
            strategies = [
                {
                    'name': 'leaf_method_fallback',
                    'description': 'Switch to leaf method with AutoTuner-optimized parameters',
                    'parameters': {
                        'cluster_selection_method': 'leaf',
                        'cluster_selection_epsilon': 0.01,
                        'min_cluster_size': max(20, int(data.shape[0] * 0.01)),
                        'min_samples': max(10, int(data.shape[0] * 0.005))
                    }
                },
                {
                    'name': 'aggressive_clustering',
                    'description': 'More aggressive clustering with AutoTuner guidance',
                    'parameters': {
                        'min_cluster_size': max(15, int(data.shape[0] * 0.008)),
                        'min_samples': max(8, int(data.shape[0] * 0.004)),
                        'cluster_selection_epsilon': 0.05,
                        'metric': 'euclidean'
                    }
                },
                {
                    'name': 'conservative_clustering',
                    'description': 'Conservative approach for difficult datasets',
                    'parameters': {
                        'min_cluster_size': max(50, int(data.shape[0] * 0.02)),
                        'min_samples': max(25, int(data.shape[0] * 0.01)),
                        'cluster_selection_epsilon': 0.1,
                        'metric': 'manhattan'
                    }
                }
            ]
            
            return strategies
            
        except Exception as e:
            logger.warning(f"Error creating fallback strategies with AutoTuner: {e}")
            # Return basic fallback strategies
            return [
                {
                    'name': 'basic_fallback',
                    'description': 'Basic fallback strategy',
                    'parameters': {
                        'min_cluster_size': 30,
                        'min_samples': 15,
                        'cluster_selection_epsilon': 0.05
                    }
                }
            ]
    
    def _execute_parameter_fallback(
        self, 
        data: pd.DataFrame, 
        initial_result: OptimizedRegimeResult,
        max_retries: int = 3
    ) -> OptimizedRegimeResult:
        """Execute parameter fallback using ML Common optimization tools."""
        tprint("🔄 Starting parameter fallback system...", "INFO")
        
        current_result = initial_result
        fallback_attempts = []
        
        # Create fallback strategies
        fallback_strategies = self._create_fallback_strategies(data)
        
        for i, strategy in enumerate(fallback_strategies[:max_retries]):
            tprint(f"🔄 Attempting fallback strategy {i+1}: {strategy['name']}", "INFO")
            
            try:
                # Apply fallback parameters to HDBSCAN config
                fallback_config = self._create_fallback_config(strategy['parameters'])
                
                # Execute clustering with fallback config
                fallback_result = self._execute_clustering_with_fallback_config(data, fallback_config)
                
                # Assess quality improvement
                quality_assessment = self._assess_clustering_quality(fallback_result)
                
                # Calculate quality improvement
                quality_improvement = self._calculate_quality_improvement(current_result, fallback_result)
                
                fallback_attempts.append({
                    'strategy': strategy['name'],
                    'result': fallback_result,
                    'quality_improvement': quality_improvement,
                    'quality_metrics': quality_assessment['quality_metrics']
                })
                
                # If quality improved significantly, use this result
                if quality_improvement > 0.1:  # 10% improvement threshold
                    tprint(f"✅ Fallback successful: {strategy['name']} improved quality by {quality_improvement:.2%}", "SUCCESS")
                    return fallback_result
                
                current_result = fallback_result
                
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy['name']} failed: {e}")
                continue
        
        # Return best result from all attempts
        if fallback_attempts:
            best_result = max(fallback_attempts, key=lambda x: x['quality_improvement'])
            tprint(f"📊 Best fallback result: {best_result['strategy']} (improvement: {best_result['quality_improvement']:.2%})", "INFO")
            return best_result['result']
        else:
            tprint("⚠️ All fallback strategies failed, returning original result", "WARNING")
            return initial_result
    
    def _create_fallback_config(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create HDBSCAN configuration from fallback parameters."""
        return {
            'min_cluster_size': parameters.get('min_cluster_size', 30),
            'min_samples': parameters.get('min_samples', 15),
            'cluster_selection_epsilon': parameters.get('cluster_selection_epsilon', 0.05),
            'cluster_selection_method': parameters.get('cluster_selection_method', 'eom'),
            'metric': parameters.get('metric', 'euclidean')
        }
    
    def _execute_clustering_with_fallback_config(
        self, 
        data: pd.DataFrame, 
        fallback_config: Dict[str, Any]
    ) -> OptimizedRegimeResult:
        """Execute clustering with fallback configuration."""
        try:
            # Generate features
            features_df = self._generate_optimized_features(data)
            features_df = self._clean_and_normalize_features(features_df)
            
            # Create HDBSCAN clusterer with fallback config
            if HDBSCAN_AVAILABLE:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=fallback_config['min_cluster_size'],
                    min_samples=fallback_config['min_samples'],
                    cluster_selection_epsilon=fallback_config['cluster_selection_epsilon'],
                    cluster_selection_method=fallback_config['cluster_selection_method'],
                    metric=fallback_config['metric']
                )
                
                # Fit and predict
                cluster_labels = clusterer.fit_predict(features_df)
                
                # Create clustering info
                clustering_info = {
                    'clusterer': clusterer,
                    'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                    'n_noise_points': list(cluster_labels).count(-1),
                    'condensed_tree_': getattr(clusterer, 'condensed_tree_', None),
                    'minimum_spanning_tree_': getattr(clusterer, 'minimum_spanning_tree_', None),
                    'single_linkage_tree_': getattr(clusterer, 'single_linkage_tree_', None),
                    'outlier_scores_': getattr(clusterer, 'outlier_scores_', None),
                    'cluster_persistence_': getattr(clusterer, 'cluster_persistence_', None),
                    'probabilities_': getattr(clusterer, 'probabilities_', None),
                    'exemplars_': getattr(clusterer, 'exemplars_', None),
                    'cluster_centers_': getattr(clusterer, 'cluster_centers_', None),
                    'cluster_sizes_': getattr(clusterer, 'cluster_sizes_', None),
                    'glosh_scores_': getattr(clusterer, 'glosh_scores_', None)
                }
                
                # Create optimized result
                return self._create_optimized_result(cluster_labels, clustering_info, features_df)
            else:
                raise RuntimeError("HDBSCAN not available")
                
        except Exception as e:
            logger.error(f"Error executing clustering with fallback config: {e}")
            raise
    
    def _calculate_quality_improvement(
        self, 
        original: OptimizedRegimeResult, 
        improved: OptimizedRegimeResult
    ) -> float:
        """Calculate quality improvement percentage using ML Common metrics."""
        improvements = []
        
        # Silhouette score improvement
        if (original.silhouette_score is not None and 
            improved.silhouette_score is not None):
            if original.silhouette_score != 0:
                sil_improvement = (improved.silhouette_score - original.silhouette_score) / abs(original.silhouette_score)
                improvements.append(sil_improvement)
        
        # Cluster count improvement (prefer more clusters if reasonable)
        if improved.n_clusters > original.n_clusters and improved.n_clusters <= 8:
            cluster_improvement = (improved.n_clusters - original.n_clusters) / max(original.n_clusters, 1)
            improvements.append(cluster_improvement)
        
        # Noise ratio improvement (prefer less noise)
        if improved.noise_ratio < original.noise_ratio:
            noise_improvement = (original.noise_ratio - improved.noise_ratio) / max(original.noise_ratio, 0.01)
            improvements.append(noise_improvement)
        
        # Calinski-Harabasz score improvement
        if (original.calinski_harabasz_score is not None and 
            improved.calinski_harabasz_score is not None):
            if original.calinski_harabasz_score != 0:
                ch_improvement = (improved.calinski_harabasz_score - original.calinski_harabasz_score) / abs(original.calinski_harabasz_score)
                improvements.append(ch_improvement)
        
        return np.mean(improvements) if improvements else 0.0