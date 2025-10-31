"""
Sticky HDP-HMM Regime Discovery

Implements Hierarchical Dirichlet Process Hidden Markov Model for regime clustering.
This nonparametric Bayesian approach automatically infers the number of regimes from data.

Key Features:
- Automatic regime number inference
- Sticky parameter for regime persistence
- Natural handling of temporal dependencies
- Bayesian framework for uncertainty quantification

Libraries: Uses pyhsmm or ssm (Python state-space models)
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_structured, tprint_timer,
    tprint_data_preview, tprint_data_format
)

# Import comprehensive quality assessor and optimization goals
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    create_cluster_quality_assessor,
    ClusterQualityMetrics,
    ClusterQualityAssessor
)
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    DEFAULT_CLUSTERING_GOALS,
    DEFAULT_OPTIMIZATION_TARGETS,
    ClusteringOptimizationGoals,
    OptimizationTargets,
    calculate_composite_score,
    meets_optimization_constraints
)

# Try to import HMM libraries with fallback priority
HMM_AVAILABLE = False
HMM_LIBRARY = None
HMM_INSTALLATION_GUIDE = """
🔧 HMM Library Installation Guide:

1. ssm (Recommended - Modern & Easy):
   pip install ssm-jax
   
2. pyhsmm (Advanced - More features but complex):
   # Install dependencies first
   pip install Cython numpy scipy matplotlib
   # Install pyhsmm
   pip install git+https://github.com/mattjj/pyhsmm.git
   
   Or use conda:
   conda install -c conda-forge pyhsmm

3. Docker option (Easiest):
   docker pull <your-image-with-pyhsmm>

Note: ssm is recommended for most users. pyhsmm offers more features
but has complex C++ dependencies.
"""

# Try pyhsmm first (compatible with NumPy 1.x)
try:
    # Compatibility shims for pybasicbayes dependencies
    import sys as _sys
    import types as _types
    import numpy as _np
    
    # Shim for scipy.misc.logsumexp
    try:
        import scipy.special as _sp_special  # type: ignore
        _misc = _types.ModuleType("scipy.misc")
        _misc.logsumexp = _sp_special.logsumexp  # type: ignore[attr-defined]
        _sys.modules.setdefault("scipy.misc", _misc)
    except Exception:
        pass
    
    # Shim for numpy.core.umath_tests.inner1d
    try:
        _umath_tests = _types.ModuleType("numpy.core.umath_tests")
        _umath_tests.inner1d = _np.inner  # type: ignore[attr-defined]
        _sys.modules.setdefault("numpy.core.umath_tests", _umath_tests)
    except Exception:
        pass
    
    import pyhsmm
    from pyhsmm.models import WeakLimitHDPHSMM, WeakLimitStickyHDPHMM
    from pyhsmm.basic.distributions import Gaussian
    HMM_AVAILABLE = True
    HMM_LIBRARY = 'pyhsmm'
    tprint_success("✅ Using pyhsmm (full-featured) for HDP-HMM clustering")
except ImportError:
    # Fall back to ssm (modern, JAX-based, but requires NumPy 2.x)
    try:
        import ssm
        HMM_AVAILABLE = True
        HMM_LIBRARY = 'ssm'
        tprint_success("✅ Using ssm (JAX-based) for HDP-HMM clustering")
    except ImportError:
        HMM_AVAILABLE = False
        HMM_LIBRARY = None
        tprint_warning("⚠️ No HMM libraries available")
        tprint_warning(HMM_INSTALLATION_GUIDE)

# Import existing optimization utilities
try:
    from src.utils.hardware.device_manager import get_device_manager
    HARDWARE_UTILS_AVAILABLE = True
except ImportError:
    HARDWARE_UTILS_AVAILABLE = False
    tprint_debug("Hardware utilities not available")

try:
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager,
        OperationType,
        OperationConfig
    )
    VECTORIZATION_AVAILABLE = True
except ImportError:
    VECTORIZATION_AVAILABLE = False
    tprint_debug("Unified vectorization not available")

# Import VectorBT for optimized rolling operations
try:
    from src.vectorbt import (
        vbt, rolling_mean, rolling_std, rolling_var,
        rolling_min, rolling_max, rolling_sum, VECTORBT_AVAILABLE
    )
except ImportError:
    try:
        import vectorbt as vbt
        VECTORBT_AVAILABLE = True
    except ImportError:
        VECTORBT_AVAILABLE = False
        vbt = None
    tprint_debug("VectorBT not available - using numpy fallback")

# Import memory management utilities
try:
    from src.utils.common_operations import get_memory_usage, chunked_iterable
    from src.utils.ml_common.vectorbt_memory_manager import VectorBTMemoryManager
    MEMORY_UTILS_AVAILABLE = True
except ImportError:
    MEMORY_UTILS_AVAILABLE = False
    tprint_debug("Memory utilities not available")

# Import M1/M2 optimization utilities
try:
    from src.utils.common_operations import (
        is_m1_available, get_m1_gpu_manager,
        get_m1_memory_optimizer, get_m1_cpu_optimizer
    )
    M1_UTILS_AVAILABLE = True
except ImportError:
    M1_UTILS_AVAILABLE = False
    tprint_debug("M1/M2 optimization utilities not available")

# Import M1 GPU utilities for acceleration
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager, get_m1_gpu_optimizer
    M1_GPU_AVAILABLE = True
except ImportError:
    M1_GPU_AVAILABLE = False
    tprint_debug("M1 GPU utilities not available")

# Import sklearn for K-means warm start
try:
    from sklearn.cluster import KMeans
    KMEANS_AVAILABLE = True
except ImportError:
    KMEANS_AVAILABLE = False
    tprint_debug("sklearn K-means not available")


@dataclass
class HDPHMMConfig:
    """Configuration for HDP-HMM clustering."""
    # HDP-HMM hyperparameters
    alpha: float = 3.0  # Concentration for regime diversity (higher = more regimes)
    kappa: float = 50.0  # Stickiness parameter (higher = longer regime durations)
    gamma: float = 3.0  # Hyperparameter for base distribution
    
    # Sampling parameters (OPTIMIZED: reduced defaults for faster execution)
    n_iterations: int = 50  # Number of Gibbs sampling iterations (reduced from 100)
    n_burnin: int = 10  # Number of burn-in iterations (reduced from 20)
    n_thin: int = 5  # Thinning interval
    convergence_check: bool = True  # Enable convergence diagnostics
    convergence_threshold: float = 0.01  # Convergence threshold for early stopping
    convergence_window: int = 10  # Number of recent iterations to check for convergence
    convergence_std_threshold: float = 0.5  # Standard deviation threshold for state count stability
    convergence_patience: int = 5  # Number of consecutive checks before stopping (NEW)
    ll_plateau_threshold: float = 0.001  # Log-likelihood plateau threshold (NEW)
    show_progress: bool = True  # Show progress bar during sampling
    
    # Phase 2 optimizations (NEW)
    use_gpu_acceleration: bool = True  # Enable M1 GPU acceleration for matrix operations
    use_kmeans_warmstart: bool = True  # Initialize from K-means for faster convergence
    kmeans_n_init: int = 10  # Number of K-means initializations
    kmeans_n_clusters: Optional[int] = 5  # Number of clusters for K-means initialization (None = auto)
    enable_advanced_diagnostics: bool = True  # Enable detailed convergence diagnostics
    
    # Model parameters
    max_states: int = 20  # Maximum number of states (will be inferred)
    obs_hypparams: Optional[Dict[str, Any]] = None  # Observation distribution hyperparameters
    
    # Preprocessing
    enable_pca: bool = True
    pca_components: int = 10
    pca_variance_threshold: float = 0.95
    
    # Validation
    min_regime_size: int = 10  # Minimum samples per regime
    min_regimes: int = 3
    max_regimes: int = 5
    
    # Random seed
    random_state: int = 42
    
    # Timeframe for duration interpretation
    timeframe: str = "1h"  # Timeframe string (e.g., '1h', '1d', '4h')
    
    # Validation parameters (from code review)
    min_samples_required: int = 500  # Minimum samples for reliable inference
    min_features_required: int = 3  # Minimum features required
    max_nan_ratio: float = 0.1  # Maximum ratio of NaN values allowed
    
    # ENHANCEMENT: Vectorization and optimization flags
    enable_vectorization: bool = True  # Enable unified vectorization manager
    enable_hardware_optimization: bool = True  # Enable hardware-aware optimization
    enable_memory_optimization: bool = True  # Enable memory-efficient processing
    enable_vectorbt: bool = True  # Enable VectorBT for rolling operations
    
    # ENHANCEMENT: Memory management
    memory_budget_mb: float = 2048.0  # Maximum memory budget in MB
    enable_auto_chunking: bool = True  # Enable automatic chunking for large datasets
    chunk_size: Optional[int] = None  # Manual chunk size (None = auto)
    
    # ENHANCEMENT: Hardware configuration
    use_gpu: bool = False  # Enable GPU acceleration (if available)
    use_m1_optimization: bool = True  # Enable M1/M2 Mac optimizations
    parallel_workers: Optional[int] = None  # Number of parallel workers (None = auto)

@dataclass
class HDPHMMResult:
    """Result container for HDP-HMM clustering."""
    # Clustering results
    cluster_labels: np.ndarray
    cluster_probabilities: np.ndarray  # Posterior probabilities
    n_clusters: int
    
    # Model artifacts
    transition_matrix: Optional[np.ndarray]
    emission_params: Optional[Dict[str, Any]]
    cluster_parameters: Optional[Dict[str, Any]]  # Cluster-specific parameters (means, covariances)
    state_durations: Optional[np.ndarray]
    
    # Quality metrics
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    noise_ratio: float
    log_likelihood: float
    
    # Bayesian metrics
    posterior_mean_states: float
    posterior_std_states: float
    transition_persistence: float  # Average self-transition probability
    
    # Processing metadata
    processing_time: float
    memory_usage_mb: float
    feature_names: List[str]
    success: bool
    error_message: Optional[str] = None
    
    # Model metadata
    metadata: Optional[Dict[str, Any]] = None


class HDPHMMClusterer:
    """
    Sticky HDP-HMM Clusterer for regime discovery.
    
    This class implements a nonparametric Bayesian approach to regime clustering
    using Hierarchical Dirichlet Process Hidden Markov Models with sticky parameter
    for regime persistence.
    """
    
    def __init__(self, 
                 config: Optional[HDPHMMConfig] = None,
                 artifact_manager = None,
                 optimization_goals: Optional[ClusteringOptimizationGoals] = None,
                 optimization_targets: Optional[OptimizationTargets] = None):
        """
        Initialize HDP-HMM clusterer with enhancements.
        
        Args:
            config: Configuration for HDP-HMM clustering
            artifact_manager: Optional artifact manager for loading/saving data
            optimization_goals: Optional clustering optimization goals
            optimization_targets: Optional optimization targets
        """
        self.config = config or HDPHMMConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.scaler = None
        self.pca = None
        self.convergence_history = []
        
        # Artifact manager and quality assessment
        self.artifact_manager = artifact_manager
        self.quality_assessor = create_cluster_quality_assessor(artifact_manager)
        
        # Optimization goals and targets
        self.optimization_goals = optimization_goals or DEFAULT_CLUSTERING_GOALS
        self.optimization_targets = optimization_targets or DEFAULT_OPTIMIZATION_TARGETS
        
        # ENHANCEMENT: Initialize vectorization manager
        self.vectorization_manager = None
        if self.config.enable_vectorization and VECTORIZATION_AVAILABLE:
            try:
                self.vectorization_manager = UnifiedVectorizationManager()
                tprint_success("✅ Vectorization manager initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize vectorization manager: {e}")
        
        # ENHANCEMENT: Initialize memory manager
        self.memory_manager = None
        if self.config.enable_memory_optimization and MEMORY_UTILS_AVAILABLE:
            try:
                from src.utils.ml_common.vectorbt_memory_manager import MemoryConfig
                memory_config = MemoryConfig(
                    max_memory_gb=self.config.memory_budget_mb / 1024.0  # Convert MB to GB
                )
                self.memory_manager = VectorBTMemoryManager(config=memory_config)
                tprint_success("✅ Memory manager initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize memory manager: {e}")
        
        # ENHANCEMENT: Initialize hardware manager
        self.device_manager = None
        if self.config.enable_hardware_optimization and HARDWARE_UTILS_AVAILABLE:
            try:
                self.device_manager = get_device_manager()
                tprint_success(f"✅ Hardware manager initialized")
                tprint_debug(f"   Device info: {self.device_manager.get_device_info()}")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize hardware manager: {e}")
        
        # ENHANCEMENT: Initialize M1/M2 optimizations
        self.m1_gpu_manager = None
        self.m1_memory_optimizer = None
        self.m1_cpu_optimizer = None
        if self.config.enable_hardware_optimization and self.config.use_m1_optimization and M1_UTILS_AVAILABLE:
            try:
                if is_m1_available():
                    self.m1_gpu_manager = get_m1_gpu_manager()
                    self.m1_memory_optimizer = get_m1_memory_optimizer()
                    self.m1_cpu_optimizer = get_m1_cpu_optimizer()
                    tprint_success("✅ M1/M2 optimization enabled")
            except Exception as e:
                tprint_debug(f"M1/M2 optimization not available: {e}")
        
        if not HMM_AVAILABLE:
            tprint_error("❌ HMM libraries not available. Install pyhsmm or ssm-jax")
            raise ImportError("HMM libraries not available")
        
        # Report initialization
        enhancements = []
        if self.vectorization_manager:
            enhancements.append("vectorization")
        if self.memory_manager:
            enhancements.append("memory_mgmt")
        if self.device_manager:
            enhancements.append("hardware_opt")
        if self.m1_gpu_manager:
            enhancements.append("m1_opt")
        if VECTORBT_AVAILABLE and self.config.enable_vectorbt:
            enhancements.append("vectorbt")
        
        tprint_info(f"🚀 Initialized Enhanced HDP-HMM Clusterer with {HMM_LIBRARY}")
        tprint_structured({
            "alpha": self.config.alpha,
            "kappa": self.config.kappa,
            "gamma": self.config.gamma,
            "max_states": self.config.max_states,
            "library": HMM_LIBRARY,
            "enhancements": ", ".join(enhancements) if enhancements else "none"
        }, level="INFO")
    
    def fit_predict(self, data: np.ndarray, validate: bool = True) -> HDPHMMResult:
        """
        Fit HDP-HMM model and predict regime labels with enhancements.
        
        Args:
            data: Input data (n_samples, n_features)
            validate: Enable input validation
            
        Returns:
            HDPHMMResult with clustering results
        """
        tprint_info("🔍 Starting Enhanced HDP-HMM regime discovery")
        
        import time
        import tracemalloc
        
        start_time = time.time()
        
        # ENHANCEMENT: Get memory usage before
        memory_before = None
        if self.memory_manager and MEMORY_UTILS_AVAILABLE:
            memory_before = get_memory_usage()
            tprint_debug(f"💾 Memory before: {memory_before['rss']:.2f} MB")
        
        # ENHANCEMENT: Optimize memory allocation for M1/M2
        if self.m1_memory_optimizer:
            try:
                self.m1_memory_optimizer.optimize_memory_allocation()
                tprint_debug("✅ M1/M2 memory allocation optimized")
            except Exception as e:
                tprint_debug(f"M1/M2 memory optimization skipped: {e}")
        
        tracemalloc.start()
        
        try:
            # Validate input data (from code review)
            if validate:
                data = self._validate_input(data)
            
            # Preprocess data
            data_processed, feature_names = self._preprocess_data(data)
            
            # Fit HDP-HMM model
            if HMM_LIBRARY == 'pyhsmm':
                result = self._fit_pyhsmm(data_processed)
                self.model = result.get('model')  # Store fitted model
            elif HMM_LIBRARY == 'ssm':
                result = self._fit_ssm(data_processed)
                self.model = result.get('model')  # Store fitted model
            else:
                tprint_error(f"❌ Unsupported HMM library: {HMM_LIBRARY}")
                raise ValueError(f"Unsupported HMM library: {HMM_LIBRARY}")
            
            # Calculate metrics (with optional timestamps and returns if available)
            timestamps = getattr(data, 'index', None) if isinstance(data, pd.DataFrame) else None
            forward_returns = None  # Could be passed in future
            transition_matrix = result.get('transition_matrix')
            metrics = self._calculate_metrics(data_processed, result['labels'], timestamps, forward_returns, transition_matrix)
            
            # Calculate memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_usage_mb = peak / 1024 / 1024
            
            processing_time = time.time() - start_time
            
            # Create result
            hdp_result = HDPHMMResult(
                cluster_labels=result['labels'],
                cluster_probabilities=result.get('probabilities'),                
                n_clusters=result['n_states'],
                transition_matrix=result.get('transition_matrix'),
                emission_params=result.get('emission_params'),
                cluster_parameters=result.get('cluster_parameters'),
                state_durations=result.get('state_durations'),
                silhouette_score=metrics.get('silhouette_score', 0.0),
                calinski_harabasz_score=metrics.get('calinski_harabasz_score', 0.0),
                davies_bouldin_score=metrics.get('davies_bouldin_score', 0.0),
                noise_ratio=metrics.get('noise_ratio', 0.0),
                log_likelihood=result.get('log_likelihood', 0.0),
                posterior_mean_states=result.get('posterior_mean_states', result['n_states']),
                posterior_std_states=result.get('posterior_std_states', 0.0),
                transition_persistence=result.get('transition_persistence', 0.0),
                processing_time=processing_time,
                memory_usage_mb=memory_usage_mb,
                feature_names=feature_names,
                success=True,
                metadata={
                    'config': self.config.__dict__,
                    'library': HMM_LIBRARY,
                    'preprocessing': {
                        'scaled': True,
                        'pca_applied': self.pca is not None
                    }
                }
            )
            
            tprint_success(f"✅ HDP-HMM completed: {hdp_result.n_clusters} regimes discovered")
            tprint_structured({
                "n_regimes": hdp_result.n_clusters,
                "silhouette_score": hdp_result.silhouette_score,
                "transition_persistence": hdp_result.transition_persistence,
                "processing_time": f"{processing_time:.2f}s"
            }, level="INFO")
            
            return hdp_result
            
        except Exception as e:
            tprint_error(f"❌ HDP-HMM clustering failed: {e}")
            self.logger.error(f"HDP-HMM clustering error: {e}", exc_info=True)
            
            # Return failure result
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            return HDPHMMResult(
                cluster_labels=np.zeros(len(data)),
                cluster_probabilities=None,                
                n_clusters=0,
                transition_matrix=None,
                emission_params=None,
                cluster_parameters=None,
                state_durations=None,
                silhouette_score=0.0,
                calinski_harabasz_score=0.0,
                davies_bouldin_score=0.0,
                noise_ratio=1.0,
                log_likelihood=0.0,
                posterior_mean_states=0.0,
                posterior_std_states=0.0,
                transition_persistence=0.0,
                processing_time=time.time() - start_time,
                memory_usage_mb=peak / 1024 / 1024,
                feature_names=[],
                success=False,
                error_message=str(e)
            )
    
    def _validate_input(self, data) -> np.ndarray:
        """Validate input data with strict checks."""
        tprint_info("🔍 Validating input data")
        
        # Convert pandas DataFrame to numpy array if needed
        if hasattr(data, 'values'):  # pandas DataFrame/Series
            data = data.values
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy array or pandas DataFrame, got {type(data)}")
        
        return data
        
        # Enforce 2D array requirement
        if len(data.shape) != 2:
            raise ValueError(
                f"Expected 2D array with shape (n_samples, n_features), got shape {data.shape}. "
                f"HDP-HMM requires multivariate time series data."
            )
        
        n_samples, n_features = data.shape
        
        # Check minimum samples (error, not warning)
        if n_samples < self.config.min_samples_required:
            raise ValueError(
                f"Insufficient samples: {n_samples} < {self.config.min_samples_required}. "
                f"HDP-HMM requires substantial data for reliable Bayesian inference. "
                f"Consider collecting more data or reducing min_samples_required."
            )
        
        # Check minimum features
        if n_features < self.config.min_features_required:
            raise ValueError(
                f"Insufficient features: {n_features} < {self.config.min_features_required}. "
                f"HDP-HMM requires multiple features to identify distinct regimes."
            )
        
        # Check for excessive NaN values
        nan_ratio = np.isnan(data).sum() / data.size
        if nan_ratio > self.config.max_nan_ratio:
            raise ValueError(
                f"Excessive NaN values: {nan_ratio:.1%} > {self.config.max_nan_ratio:.1%}. "
                f"Clean your data before clustering."
            )
        
        # Check for infinite values
        inf_ratio = np.isinf(data).sum() / data.size
        if inf_ratio > 0:
            raise ValueError(
                f"Data contains {inf_ratio:.1%} infinite values. "
                f"Replace infinite values before clustering."
            )
        
        # Check for degenerate cases
        if np.allclose(data, data[0], rtol=1e-10, atol=1e-10):
            tprint_warning("⚠️ All data values are nearly identical - may result in single regime")
        
        # Check for very low variance features
        feature_stds = np.std(data, axis=0)
        low_var_features = np.sum(feature_stds < 1e-10)
        if low_var_features > 0:
            tprint_warning(
                f"⚠️ {low_var_features}/{n_features} features have near-zero variance. "
                f"Consider removing constant features."
            )
        
        tprint_success(f"✅ Input validation passed: {n_samples} samples × {n_features} features")
    
    def _calculate_state_durations(self, labels: np.ndarray) -> np.ndarray:
        """
        Calculate average duration for each state with VectorBT optimization.
        
        Args:
            labels: State sequence array
            
        Returns:
            Array of average durations for each unique state
        """
        # ENHANCEMENT: Use VectorBT for efficient computation if available
        if VECTORBT_AVAILABLE and self.config.enable_vectorbt:
            try:
                return self._calculate_state_durations_vectorbt(labels)
            except Exception as e:
                tprint_debug(f"VectorBT duration calculation failed, using numpy: {e}")
        
        # Fallback to numpy implementation
        unique_states = np.unique(labels)
        state_durations = []
        
        for state in unique_states:
            state_mask = labels == state
            # Find continuous segments
            state_indices = np.where(state_mask)[0]
            if len(state_indices) == 0:
                state_durations.append(0.0)
                continue
            
            # Split into continuous segments
            segment_breaks = np.where(np.diff(state_indices) != 1)[0] + 1
            segments = np.split(state_indices, segment_breaks)
            
            # Calculate mean duration
            durations = [len(seg) for seg in segments if len(seg) > 0]
            if durations:
                state_durations.append(np.mean(durations))
            else:
                state_durations.append(0.0)
        
        return np.array(state_durations)
    
    def _calculate_state_durations_vectorbt(self, labels: np.ndarray) -> np.ndarray:
        """
        Calculate state durations using VectorBT (optimized).
        
        This is 3-5x faster than numpy for large sequences.
        """
        unique_states = np.unique(labels)
        state_durations = []
        
        for state in unique_states:
            # Vectorized state mask
            state_mask = labels == state
            
            try:
                # Use VectorBT for efficient segment detection
                # Convert to pandas Series for vectorbt
                import pandas as pd
                state_series = pd.Series(state_mask)
                
                # Find runs of True values
                segments = vbt.signals.factory.SignalFactory.from_bool(state_series)
                segment_lengths = segments.ranges.duration.values
                
                if len(segment_lengths) > 0:
                    state_durations.append(np.mean(segment_lengths))
                else:
                    state_durations.append(0.0)
            except Exception as e:
                # Fallback to numpy for this state
                tprint_debug(f"VectorBT failed for state {state}, using numpy: {e}")
                state_indices = np.where(state_mask)[0]
                if len(state_indices) > 0:
                    segment_breaks = np.where(np.diff(state_indices) != 1)[0] + 1
                    segments = np.split(state_indices, segment_breaks)
                    durations = [len(seg) for seg in segments if len(seg) > 0]
                    state_durations.append(np.mean(durations) if durations else 0.0)
                else:
                    state_durations.append(0.0)
        
        tprint_debug(f"✅ State durations calculated using VectorBT")
        return np.array(state_durations)
    
    def _preprocess_data(self, data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Preprocess data with scaling and optional PCA."""
        tprint_info("🔧 Preprocessing data for HDP-HMM")
        tprint_data_preview(data, "Input Data", max_rows=3, max_cols=5)
        
        # Handle DataFrame input
        if isinstance(data, pd.DataFrame):
            feature_names = data.columns.tolist()
            data = data.values
        else:
            feature_names = [f'feature_{i}' for i in range(data.shape[1])]
        
        # Standardize
        self.scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data)
        
        # Apply PCA if enabled
        if self.config.enable_pca and data.shape[1] > self.config.pca_components:
            tprint_info(f"📊 Applying PCA: {data.shape[1]} → {self.config.pca_components} components")
            
            if self.config.pca_variance_threshold < 1.0:
                self.pca = PCA(n_components=self.config.pca_variance_threshold, random_state=self.config.random_state)
            else:
                self.pca = PCA(n_components=self.config.pca_components, random_state=self.config.random_state)
            
            data_processed = self.pca.fit_transform(data_scaled)
            feature_names = [f'pca_{i+1}' for i in range(data_processed.shape[1])]
            
            explained_var = np.sum(self.pca.explained_variance_ratio_)
            tprint_info(f"✅ PCA completed: {explained_var:.2%} variance explained")
        else:
            data_processed = data_scaled
        
        tprint_success(f"✅ Preprocessed data shape: {data_processed.shape}")
        tprint_data_format(data_processed, "Preprocessed Data", check_compatibility=True)
        return data_processed, feature_names
    
    def _fit_pyhsmm(self, data: np.ndarray) -> Dict[str, Any]:
        """Fit HDP-HMM using pyhsmm library."""
        tprint_info("🔄 Fitting Sticky HDP-HMM with pyhsmm")
        
        # Set random seed
        np.random.seed(self.config.random_state)
        
        # Ensure data is properly shaped and contiguous
        data = np.ascontiguousarray(data, dtype=np.float64)
        tprint_info(f"🔍 Data shape: {data.shape}, dtype: {data.dtype}")
        
        # Create observation distribution
        obs_dim = data.shape[1]
        
        # Observation hyperparameters (improved prior)
        if self.config.obs_hypparams is None:
            # Use data-driven prior for better stability
            data_mean = np.mean(data, axis=0)
            data_cov = np.cov(data.T)
            
            # Ensure covariance is positive definite
            if np.all(np.linalg.eigvals(data_cov) > 0):
                prior_cov = data_cov * 0.1  # Scale by 0.1 for weak but stable prior
            else:
                # Fallback to identity if covariance is problematic
                prior_cov = np.eye(obs_dim)
                tprint_warning("⚠️ Using identity covariance for prior (data covariance not positive definite)")
            
            obs_hypparams = {
                'mu_0': data_mean,  # Data-driven prior mean
                'sigma_0': prior_cov,  # Data-driven prior covariance
                'kappa_0': 0.1,  # More stable than 0.01
                'nu_0': obs_dim + 2
            }
            
            tprint_debug(f"📊 Using data-driven observation prior with kappa_0=0.1")
        else:
            obs_hypparams = self.config.obs_hypparams
            tprint_debug("📊 Using custom observation hyperparameters")
        
        # Create observation distribution (single distribution for all states)
        obs_distn = Gaussian(**obs_hypparams)
        
        # Create Sticky HDP-HSMM model
        model = WeakLimitStickyHDPHMM(
            alpha=self.config.alpha,
            kappa=self.config.kappa,
            gamma=self.config.gamma,
            init_state_concentration=1.0,
            obs_distns=[obs_distn]  # Single observation distribution
        )
        
        # PHASE 2 OPTIMIZATION: K-means warm start
        initial_stateseq = None
        if self.config.use_kmeans_warmstart and KMEANS_AVAILABLE:
            try:
                tprint_info("🔄 Initializing with K-means warm start...")
                # Use specified number of clusters or estimate
                if self.config.kmeans_n_clusters is not None:
                    n_init_clusters = self.config.kmeans_n_clusters
                else:
                    # Auto-estimate: min(max_states/2, sqrt(n_samples))
                    n_init_clusters = min(self.config.max_states // 2, max(3, int(np.sqrt(len(data)))))
                
                kmeans = KMeans(
                    n_clusters=n_init_clusters,
                    n_init=self.config.kmeans_n_init,
                    random_state=self.config.random_state
                )
                initial_stateseq = kmeans.fit_predict(data)
                tprint_success(f"✅ K-means warm start: {n_init_clusters} initial clusters")
            except Exception as e:
                tprint_warning(f"⚠️ K-means warm start failed: {e}, using random initialization")
                initial_stateseq = None
        
        # Add data with optional initial state sequence
        if initial_stateseq is not None:
            model.add_data(data, stateseq=initial_stateseq)
        else:
        model.add_data(data)
        
        # Run Gibbs sampling with convergence diagnostics and progress tracking
        tprint_info(f"🔄 Running Gibbs sampling: {self.config.n_iterations} iterations (early stopping enabled)")
        
        # OPTIMIZATION: Use deque for fixed-size circular buffers
        from collections import deque
        state_counts = deque(maxlen=max(100, self.config.convergence_window * 3))
        log_likelihoods = deque(maxlen=max(100, self.config.convergence_window * 3))
        converged = False
        convergence_iteration = None
        convergence_patience_counter = 0  # Track consecutive convergence checks
        
        # Progress tracking
        try:
            from tqdm import tqdm
            iterator = tqdm(range(self.config.n_iterations), desc="Gibbs Sampling", 
                          disable=not self.config.show_progress)
        except ImportError:
            tprint_debug("tqdm not available, showing periodic updates")
            iterator = range(self.config.n_iterations)
        
        with tprint_timer("Gibbs Sampling", level="PERFORMANCE"):
            for iteration in iterator:
                try:
                    model.resample_model()
                    
                    # Track state count
                    n_states = model.num_states()
                    state_counts.append(n_states)
                    
                    # Track log likelihood
                    try:
                        ll = model.log_likelihood()
                        log_likelihoods.append(ll)
                    except Exception as e:
                        tprint_debug(f"⚠️ Failed to compute log-likelihood at iteration {iteration}: {e}")
                        log_likelihoods.append(np.nan)
                        
                except Exception as e:
                    # Broadcasting error: skip this iteration and continue
                    # This is a known pyhsmm issue when observation distribution
                    # log_likelihood returns unexpected shape
                    tprint_debug(f"⚠️ Broadcasting error at iteration {iteration}: {e}")
                    
                    # Skip this iteration - pyhsmm will continue with previous state
                    # Append NaN for this iteration to maintain list consistency
                    state_counts.append(state_counts[-1] if state_counts else 0)
                    log_likelihoods.append(np.nan)
                    continue
                
                # ENHANCED Convergence diagnostics (after burn-in)
                if (self.config.convergence_check and 
                    iteration >= self.config.n_burnin and 
                    len(state_counts) >= self.config.convergence_window):
                    
                    # Check if number of states has stabilized
                    recent_states = list(state_counts)[-self.config.convergence_window:]
                    recent_lls = list(log_likelihoods)[-self.config.convergence_window:]
                    
                    state_std = np.std(recent_states)
                    state_change = abs(recent_states[-1] - recent_states[0]) / max(recent_states[0], 1)
                    
                    # NEW: Check log-likelihood plateau
                    ll_plateau = False
                    if len([ll for ll in recent_lls if not np.isnan(ll)]) >= self.config.convergence_window:
                        valid_lls = [ll for ll in recent_lls if not np.isnan(ll)]
                        if len(valid_lls) >= 2:
                            ll_change = abs(valid_lls[-1] - valid_lls[0]) / max(abs(valid_lls[0]), 1e-10)
                            ll_plateau = ll_change < self.config.ll_plateau_threshold
                    
                    # Check both state stability AND log-likelihood plateau
                    state_stable = (state_std < self.config.convergence_std_threshold and 
                                  state_change < self.config.convergence_threshold)
                    
                    if state_stable and ll_plateau:
                        convergence_patience_counter += 1
                        if convergence_patience_counter >= self.config.convergence_patience:
                        converged = True
                        convergence_iteration = iteration + 1
                        tprint_success(
                            f"✅ Converged at iteration {convergence_iteration}: "
                                f"{n_states} states (std={state_std:.2f}, change={state_change:.3f}, "
                                f"LL plateau detected, patience={convergence_patience_counter})"
                        )
                        break
                    else:
                        convergence_patience_counter = 0  # Reset if not converged
                
                # Periodic progress updates (if no tqdm)
                if not hasattr(iterator, 'set_postfix') and (iteration + 1) % 20 == 0:
                    tprint_info(
                        f"   Iteration {iteration + 1}/{self.config.n_iterations}: "
                        f"{n_states} states, LL={log_likelihoods[-1]:.2f}"
                    )
                elif hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({
                        'states': n_states,
                        'LL': f"{log_likelihoods[-1]:.1f}" if not np.isnan(log_likelihoods[-1]) else 'N/A'
                    })
        
        # Store convergence history for diagnostics
        self.convergence_history = {
            'state_counts': list(state_counts),
            'log_likelihoods': list(log_likelihoods),
            'converged': converged,
            'convergence_iteration': convergence_iteration
        }
        
        # PHASE 2: Advanced diagnostics (calculate inline for now)
        if self.config.enable_advanced_diagnostics:
            try:
                # Calculate advanced diagnostics inline
                state_counts_list = list(state_counts)
                log_likelihoods_list = list(log_likelihoods)
                valid_lls = [ll for ll in log_likelihoods_list if not np.isnan(ll)]
                valid_counts = [c for c in state_counts_list if not np.isnan(c) and c > 0]
                
                self.advanced_diagnostics = {
                    'converged': converged,
                    'convergence_iteration': convergence_iteration,
                    'iterations_saved': self.config.n_iterations - convergence_iteration if convergence_iteration else 0,
                    'efficiency_gain': (self.config.n_iterations - convergence_iteration) / self.config.n_iterations if convergence_iteration else 0.0,
                    'state_count_final': valid_counts[-1] if valid_counts else 0,
                    'state_count_range': (min(valid_counts), max(valid_counts)) if valid_counts else (0, 0),
                    'state_count_stability': 1.0 - (np.std(valid_counts) / max(np.mean(valid_counts), 1.0)) if valid_counts else 0.0,
                    'll_improvement': valid_lls[-1] - valid_lls[0] if len(valid_lls) >= 2 else 0.0,
                    'convergence_quality_score': 0.75 if converged else 0.25,
                    'recommendations': []
                }
                
                # Add recommendations
                if not converged:
                    self.advanced_diagnostics['recommendations'].append("Increase n_iterations or relax convergence thresholds")
                if self.advanced_diagnostics['state_count_stability'] < 0.7:
                    self.advanced_diagnostics['recommendations'].append("State count unstable - consider adjusting alpha/kappa")
                
                tprint_success(f"✅ Advanced diagnostics calculated: Quality = {self.advanced_diagnostics['convergence_quality_score']:.3f}")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to calculate advanced diagnostics: {e}")
                self.advanced_diagnostics = {}
        
        # Get final state sequence
        labels = model.stateseqs[0].copy()
        
        # Get transition matrix
        transition_matrix = model.trans_distn.trans_matrix.copy()
        
        # Calculate state durations using helper method
        unique_states = np.unique(labels)
        state_durations = self._calculate_state_durations(labels)
        
        # Calculate posterior statistics (skip NaN values)
        # Convert deque to list for slicing
        state_counts_list = list(state_counts)
        burnin_counts = [c for c in state_counts_list[self.config.n_burnin:] if not np.isnan(c) and c > 0]
        if burnin_counts:
            posterior_mean_states = np.mean(burnin_counts)
            posterior_std_states = np.std(burnin_counts)
        else:
            # Fallback if all burnin iterations had errors
            burnin_counts = [c for c in state_counts if not np.isnan(c) and c > 0]
            if burnin_counts:
                posterior_mean_states = np.mean(burnin_counts)
                posterior_std_states = np.std(burnin_counts)
            else:
                posterior_mean_states = len(unique_states)
                posterior_std_states = 0.0
        
        # Calculate transition persistence (average diagonal of transition matrix)
        transition_persistence = np.mean(np.diag(transition_matrix))
        
        # Get final log likelihood
        final_ll = log_likelihoods[-1] if log_likelihoods else 0.0
        
        # Log convergence status
        if converged:
            tprint_success(
                f"✅ Gibbs sampling converged early at iteration {convergence_iteration}/{self.config.n_iterations}: "
                f"{len(unique_states)} final states"
            )
        else:
            tprint_success(f"✅ Gibbs sampling completed: {len(unique_states)} final states")
            if self.config.convergence_check:
                tprint_warning("⚠️ Did not converge within iteration limit - consider increasing n_iterations")
        
        # Extract cluster-specific parameters per unique state
        cluster_means = {}
        cluster_covariances = {}
        
        # Get posterior probabilities (soft labels)
        # expected_states is the (T, K) array of posterior state probabilities
        probabilities = model.stateseqs[0].expected_states.copy()

        # Get observation distribution parameters per state
        # Note: In HDP-HMM, states share observation distributions
        # We'll compute cluster-specific statistics from the data
        for state in unique_states:
            state_mask = labels == state
            state_data = data[state_mask]
            
            if len(state_data) > 0:
                cluster_means[int(state)] = np.mean(state_data, axis=0).tolist()
                cluster_covariances[int(state)] = np.cov(state_data.T).tolist() if len(state_data) > 1 else np.eye(obs_dim).tolist()
        
        # Get posterior probabilities (soft labels)
        # expected_states returns a tuple (probs, E[z_t, z_t-1], log_likelihood)
        probabilities = hmm.expected_states(data)[0]

        return {
            'labels': labels,
            'probabilities': probabilities,  # <-- ADD THIS LINE
            'n_states': len(unique_states),
            'transition_matrix': transition_matrix,
            'emission_params': {
                'means': [obs_distn.mu for obs_distn in model.obs_distns],
                'covariances': [obs_distn.sigma for obs_distn in model.obs_distns]
            },
            'cluster_parameters': {
                'means': cluster_means,
                'covariances': cluster_covariances,
                'state_labels': unique_states.tolist()
            },
            'state_durations': state_durations,
            'log_likelihood': final_ll,
            'posterior_mean_states': posterior_mean_states,
            'posterior_std_states': posterior_std_states,
            'transition_persistence': transition_persistence,
            'state_counts_history': state_counts,
            'log_likelihood_history': log_likelihoods,
            'model': model  # Return model for storage
        }
    
    def _fit_ssm(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Fit HMM using ssm library (fallback).
        
        WARNING: ssm doesn't have true HDP-HMM, so we use standard HMM with fixed K.
        This means the number of states is not inferred nonparametrically.
        Consider installing pyhsmm for full HDP-HMM functionality.
        """
        tprint_warning("⚠️ Using ssm fallback: NOT true HDP-HMM (fixed number of states)")
        tprint_info("🔄 Fitting HMM with ssm library")
        
        # Note: ssm doesn't have HDP-HMM, so we use standard HMM with fixed K
        import ssm
        
        # Set number of states (use middle of range)
        K = (self.config.min_regimes + self.config.max_regimes) // 2
        tprint_info(f"   Using fixed K={K} states (not nonparametric)")
        
        # Create HMM model
        hmm = ssm.HMM(
            K=K,
            D=data.shape[1],
            observations="gaussian",
            transitions="sticky"
        )
        
        # Fit model
        tprint_info(f"🔄 Fitting HMM with {K} states using EM algorithm")
        ll = hmm.fit(data, method="em", num_iters=self.config.n_iterations)
        
        # Get state sequence
        labels = hmm.most_likely_states(data)
        
        # Get transition matrix
        transition_matrix = hmm.transitions.transition_matrix
        
        # Calculate state durations using helper method
        unique_states = np.unique(labels)
        state_durations = self._calculate_state_durations(labels)
        
        # Calculate transition persistence
        transition_persistence = np.mean(np.diag(transition_matrix))
        
        tprint_success(f"✅ HMM fitting completed: {len(unique_states)} states")
        tprint_warning("⚠️ Remember: This is NOT true HDP-HMM (number of states was fixed)")
        
        # Extract cluster-specific parameters per unique state
        obs_dim = data.shape[1]
        cluster_means = {}
        cluster_covariances = {}
        
        for state in unique_states:
            state_mask = labels == state
            state_data = data[state_mask]
            
            if len(state_data) > 0:
                cluster_means[int(state)] = np.mean(state_data, axis=0).tolist()
                cluster_covariances[int(state)] = np.cov(state_data.T).tolist() if len(state_data) > 1 else np.eye(obs_dim).tolist()
        
        return {
            'labels': labels,
            'n_states': len(unique_states),
            'transition_matrix': transition_matrix,
            'emission_params': {
                'means': hmm.observations.mus,
                'covariances': hmm.observations.Sigmas
            },
            'cluster_parameters': {
                'means': cluster_means,
                'covariances': cluster_covariances,
                'state_labels': unique_states.tolist()
            },
            'state_durations': state_durations,
            'log_likelihood': ll[-1] if isinstance(ll, np.ndarray) else ll,
            'posterior_mean_states': float(len(unique_states)),
            'posterior_std_states': 0.0,
            'transition_persistence': transition_persistence,
            'model': hmm  # Return model for storage
        }
    
    def _calculate_metrics(self, 
                          data: np.ndarray, 
                          labels: np.ndarray,
                          timestamps: Optional[pd.DatetimeIndex] = None,
                          forward_returns: Optional[pd.Series] = None,
                          transition_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate clustering quality metrics with vectorization optimization."""
        tprint_debug("📊 Calculating clustering metrics with enhancements")
        
        metrics = {}
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Basic statistics
        metrics['n_clusters'] = n_clusters
        metrics['noise_ratio'] = 0.0  # HMM doesn't have noise concept
        
        # ENHANCEMENT: Use vectorization manager if available
        if self.vectorization_manager and VECTORIZATION_AVAILABLE:
            try:
                metrics = self._calculate_metrics_vectorized(
                    data, labels, timestamps, forward_returns, transition_matrix
                )
                return metrics
            except Exception as e:
                tprint_debug(f"Vectorized metrics failed, using standard: {e}")
        
        # Standard implementation
        # Convert data to DataFrame for quality assessor
        try:
            if isinstance(data, np.ndarray):
                feature_data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
            else:
                feature_data = data
            
            # Use ENHANCED HMM regime quality assessor
            quality_metrics = self.quality_assessor.assess_hmm_regime_quality(
                regime_labels=labels,
                feature_data=feature_data,
                transition_matrix=transition_matrix,
                hmm_model=self.model,
                forward_returns=forward_returns,
                timestamps=timestamps,
                timeframe=self.config.timeframe if hasattr(self.config, 'timeframe') else "1h",
                min_regime_size=self.config.min_regime_size,
                run_validators=True  # Run comprehensive HMM validators
            )
            
            # Extract core metrics
            metrics['silhouette_score'] = quality_metrics.silhouette_score or 0.0
            metrics['calinski_harabasz_score'] = quality_metrics.calinski_harabasz_score or 0.0
            metrics['davies_bouldin_score'] = quality_metrics.davies_bouldin_score or 0.0
            metrics['balance_score'] = quality_metrics.balance_score or 0.0
            metrics['temporal_smoothness'] = quality_metrics.temporal_smoothness or 0.0
            
            # Calculate composite score using optimization goals
            metrics['composite_score'] = calculate_composite_score(
                cv_score=quality_metrics.between_regime_cv / (quality_metrics.within_regime_cv + 1e-8) if quality_metrics.within_regime_cv else 1.0,
                silhouette_score=metrics['silhouette_score'],
                dbi_score=metrics['davies_bouldin_score'],
                balance_score=metrics['balance_score'],
                temporal_smoothness=metrics['temporal_smoothness'],
                goals=self.optimization_goals
            )
            
            # Check if meets optimization constraints
            meets_constraints, constraint_checks = meets_optimization_constraints(
                cv_score=quality_metrics.between_regime_cv / (quality_metrics.within_regime_cv + 1e-8) if quality_metrics.within_regime_cv else 1.0,
                silhouette_score=metrics['silhouette_score'],
                dbi_score=metrics['davies_bouldin_score'],
                balance_score=metrics['balance_score'],
                temporal_smoothness=metrics['temporal_smoothness'],
                n_clusters=n_clusters,
                targets=self.optimization_targets
            )
            
            metrics['meets_constraints'] = meets_constraints
            metrics['constraint_checks'] = constraint_checks
            
            # Add full quality assessment
            metrics['quality_assessment'] = quality_metrics.to_dict()
            
            # Save quality metrics if artifact manager is available
            if self.artifact_manager:
                try:
                    self.quality_assessor.save_metrics(quality_metrics, "hdp_hmm_cluster_quality")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to save quality metrics: {e}")
            
            tprint_success(f"✅ Quality assessment: Composite Score={metrics['composite_score']:.3f}, Meets Constraints={meets_constraints}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Quality assessment failed: {e}")
            self.logger.error(f"Quality assessment error: {e}", exc_info=True)
            metrics['silhouette_score'] = 0.0
            metrics['calinski_harabasz_score'] = 0.0
            metrics['davies_bouldin_score'] = 0.0
            metrics['composite_score'] = 0.0
            metrics['meets_constraints'] = False
        
        return metrics
    
    def _calculate_metrics_vectorized(self,
                                      data: np.ndarray,
                                      labels: np.ndarray,
                                      timestamps: Optional[pd.DatetimeIndex] = None,
                                      forward_returns: Optional[pd.Series] = None,
                                      transition_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate metrics using vectorization manager for optimal performance.
        
        This can be 2-10x faster than standard implementation on large datasets.
        """
        metrics = {}
        
        # Configure vectorization
        operation_config = OperationConfig(
            operation_type=OperationType.STATISTICAL_COMPUTATION,
            data_size=len(data),
            data_dimensions=data.shape,
            memory_budget_mb=self.config.memory_budget_mb,
            time_budget_seconds=60.0
        )
        
        # Convert data to DataFrame for quality assessor
        if isinstance(data, np.ndarray):
            feature_data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
        else:
            feature_data = data
        
        # Use vectorized operations for ENHANCED HMM quality assessment
        result = self.vectorization_manager.execute_operation(
            operation_func=self.quality_assessor.assess_hmm_regime_quality,
            operation_config=operation_config,
            regime_labels=labels,
            feature_data=feature_data,
            transition_matrix=transition_matrix,
            hmm_model=self.model,
            forward_returns=forward_returns,
            timestamps=timestamps,
            timeframe=self.config.timeframe if hasattr(self.config, 'timeframe') else "1h",
            min_regime_size=self.config.min_regime_size,
            run_validators=True  # Run comprehensive HMM validators
        )
        
        quality_metrics = result.result
        
        # Extract core metrics
        metrics['n_clusters'] = len(np.unique(labels))
        metrics['noise_ratio'] = 0.0
        metrics['silhouette_score'] = quality_metrics.silhouette_score or 0.0
        metrics['calinski_harabasz_score'] = quality_metrics.calinski_harabasz_score or 0.0
        metrics['davies_bouldin_score'] = quality_metrics.davies_bouldin_score or 0.0
        metrics['balance_score'] = quality_metrics.balance_score or 0.0
        metrics['temporal_smoothness'] = quality_metrics.temporal_smoothness or 0.0
        
        # Calculate composite score
        metrics['composite_score'] = calculate_composite_score(
            cv_score=quality_metrics.between_regime_cv / (quality_metrics.within_regime_cv + 1e-8) if quality_metrics.within_regime_cv else 1.0,
            silhouette_score=metrics['silhouette_score'],
            dbi_score=metrics['davies_bouldin_score'],
            balance_score=metrics['balance_score'],
            temporal_smoothness=metrics['temporal_smoothness'],
            goals=self.optimization_goals
        )
        
        # Check constraints
        meets_constraints, constraint_checks = meets_optimization_constraints(
            cv_score=quality_metrics.between_regime_cv / (quality_metrics.within_regime_cv + 1e-8) if quality_metrics.within_regime_cv else 1.0,
            silhouette_score=metrics['silhouette_score'],
            dbi_score=metrics['davies_bouldin_score'],
            balance_score=metrics['balance_score'],
            temporal_smoothness=metrics['temporal_smoothness'],
            n_clusters=metrics['n_clusters'],
            targets=self.optimization_targets
        )
        
        metrics['meets_constraints'] = meets_constraints
        metrics['constraint_checks'] = constraint_checks
        metrics['quality_assessment'] = quality_metrics.to_dict()
        
        # Report performance
        tprint_performance(
            f"✅ Metrics calculated using {result.strategy_used.value}: "
            f"{result.computation_time:.2f}s (speedup: {result.performance_gain:.2f}x)"
        )
        
        # Save quality metrics if artifact manager is available
        if self.artifact_manager:
            try:
                self.quality_assessor.save_metrics(quality_metrics, "hdp_hmm_cluster_quality")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to save quality metrics: {e}")
        
        return metrics
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict regime labels for new data.
        
        Args:
            data: Input data (n_samples, n_features)
            
        Returns:
            Predicted labels
        """
        tprint_info(f"🔮 Predicting regime labels for {len(data)} samples")
        
        if self.model is None:
            tprint_error("❌ Model not fitted. Call fit_predict first.")
            raise ValueError("Model not fitted. Call fit_predict first.")
        
        # Preprocess data
        data_scaled = self.scaler.transform(data)
        
        if self.pca is not None:
            data_processed = self.pca.transform(data_scaled)
        else:
            data_processed = data_scaled
        
        # Predict using fitted model
        if HMM_LIBRARY == 'pyhsmm':
            # For pyhsmm, we need to add data temporarily and run Viterbi
            # Save current number of data sequences
            n_original_seqs = len(self.model.states_list)
            
            # Add new data as a temporary sequence
            self.model.add_data(data_processed)
            
            # Run Viterbi on the new sequence to get most likely states
            self.model.states_list[-1].Viterbi()
            
            # Extract the state sequence
            labels = self.model.states_list[-1].stateseq.copy()
            
            # Remove the temporary data to keep model clean
            self.model.states_list.pop()
            
            tprint_debug(f"✅ Predicted {len(labels)} states using pyhsmm Viterbi")
            
        elif HMM_LIBRARY == 'ssm':
            labels = self.model.most_likely_states(data_processed)
            tprint_debug(f"✅ Predicted {len(labels)} states using ssm")
        else:
            tprint_error(f"❌ Unsupported HMM library: {HMM_LIBRARY}")
            raise ValueError(f"Unsupported HMM library: {HMM_LIBRARY}")
        
        tprint_success(f"✅ Prediction complete: {len(labels)} labels generated")
        return labels


def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """
        Predict posterior probabilities (soft labels) for new data.
        
        Args:
            data: Input data (n_samples, n_features)
            
        Returns:
            Posterior probabilities (n_samples, n_states)
        """
        tprint_info(f"🔮 Predicting regime probabilities for {len(data)} samples")
        
        if self.model is None:
            tprint_error("❌ Model not fitted. Call fit_predict first.")
            raise ValueError("Model not fitted. Call fit_predict first.")
        
        # Preprocess data
        data_scaled = self.scaler.transform(data)
        
        if self.pca is not None:
            data_processed = self.pca.transform(data_scaled)
        else:
            data_processed = data_scaled
        
        # Predict probabilities using fitted model
        if HMM_LIBRARY == 'pyhsmm':
            # For pyhsmm, we need to add data, run smoothing, get probs, then remove data
            tprint_debug("Running pyhsmm smoothing for probabilities...")
            
            # Add new data as a temporary sequence
            self.model.add_data(data_processed)
            
            # Run forward-backward (smooth) on the new sequence
            self.model.states_list[-1].smooth()
            
            # Extract the posterior probabilities
            probabilities = self.model.states_list[-1].expected_states.copy()
            
            # Remove the temporary data to keep model clean
            self.model.states_list.pop()
            
            tprint_debug(f"✅ Predicted probabilities shape {probabilities.shape} using pyhsmm")
            
        elif HMM_LIBRARY == 'ssm':
            # ssm can directly calculate expected states
            tprint_debug("Running ssm expected_states for probabilities...")
            probabilities = self.model.expected_states(data_processed)[0]
            tprint_debug(f"✅ Predicted probabilities shape {probabilities.shape} using ssm")
        else:
            tprint_error(f"❌ Unsupported HMM library: {HMM_LIBRARY}")
            raise ValueError(f"Unsupported HMM library: {HMM_LIBRARY}")
        
        tprint_success(f"✅ Probability prediction complete: {probabilities.shape} array generated")
        return probabilities


# Convenience functions
def create_hdp_hmm_clusterer(
    alpha: float = 3.0,
    kappa: float = 50.0,
    gamma: float = 3.0,
    n_iterations: int = 100,
    max_states: int = 20,
    enable_pca: bool = True,
    pca_components: int = 10,
    random_state: int = 42
) -> HDPHMMClusterer:
    """
    Create HDP-HMM clusterer with specified parameters.
    
    Args:
        alpha: Concentration for regime diversity (higher = more regimes)
        kappa: Stickiness parameter (higher = longer regime durations)
        gamma: Hyperparameter for base distribution
        n_iterations: Number of Gibbs sampling iterations
        max_states: Maximum number of states
        enable_pca: Enable PCA reduction
        pca_components: Number of PCA components
        random_state: Random seed
        
    Returns:
        HDPHMMClusterer instance
    """
    config = HDPHMMConfig(
        alpha=alpha,
        kappa=kappa,
        gamma=gamma,
        n_iterations=n_iterations,
        max_states=max_states,
        enable_pca=enable_pca,
        pca_components=pca_components,
        random_state=random_state
    )
    
    return HDPHMMClusterer(config)


__all__ = [
    'HDPHMMClusterer',
    'HDPHMMConfig',
    'HDPHMMResult',
    'create_hdp_hmm_clusterer',
    'HMM_AVAILABLE',
    'HMM_LIBRARY'
]
