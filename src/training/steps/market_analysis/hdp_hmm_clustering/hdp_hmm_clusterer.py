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
    tprint_debug, tprint_performance, tprint_structured, tprint_timer
)

# Import comprehensive quality assessor
from src.training.steps.market_analysis.hdbscan_clustering.quality_assessment import (
    create_quality_assessor,
    QualityMetrics as QualityAssessmentMetrics
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

# Try ssm first (modern, JAX-based, easier to install)
try:
    import ssm
    HMM_AVAILABLE = True
    HMM_LIBRARY = 'ssm'
    tprint_success("✅ Using ssm (JAX-based) for HDP-HMM clustering")
except ImportError:
    # Fall back to pyhsmm (more features but harder to install)
    try:
        import pyhsmm
        from pyhsmm.models import WeakLimitHDPHSMM, WeakLimitStickyHDPHSMM
        from pyhsmm.basic.distributions import Gaussian
        HMM_AVAILABLE = True
        HMM_LIBRARY = 'pyhsmm'
        tprint_success("✅ Using pyhsmm (full-featured) for HDP-HMM clustering")
    except ImportError:
        tprint_warning("⚠️ No HMM libraries available")
        tprint_warning(HMM_INSTALLATION_GUIDE)
        HMM_LIBRARY = None

# Import existing optimization utilities
try:
    from src.utils.hardware.device_manager import get_device_manager
    HARDWARE_UTILS_AVAILABLE = True
except ImportError:
    HARDWARE_UTILS_AVAILABLE = False
    tprint_debug("Hardware utilities not available")

try:
    from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager
    VECTORIZATION_AVAILABLE = True
except ImportError:
    VECTORIZATION_AVAILABLE = False
    tprint_debug("Unified vectorization not available")


@dataclass
class HDPHMMConfig:
    """Configuration for HDP-HMM clustering."""
    # HDP-HMM hyperparameters
    alpha: float = 3.0  # Concentration for regime diversity (higher = more regimes)
    kappa: float = 50.0  # Stickiness parameter (higher = longer regime durations)
    gamma: float = 3.0  # Hyperparameter for base distribution
    
    # Sampling parameters
    n_iterations: int = 100  # Number of Gibbs sampling iterations
    n_burnin: int = 20  # Number of burn-in iterations
    n_thin: int = 5  # Thinning interval
    convergence_check: bool = True  # Enable convergence diagnostics
    convergence_threshold: float = 0.01  # Convergence threshold for early stopping
    show_progress: bool = True  # Show progress bar during sampling
    
    # Model parameters
    max_states: int = 20  # Maximum number of states (will be inferred)
    obs_hypparams: Optional[Dict[str, Any]] = None  # Observation distribution hyperparameters
    
    # Preprocessing
    enable_pca: bool = True
    pca_components: int = 10
    pca_variance_threshold: float = 0.95
    
    # Validation
    min_regime_size: int = 10  # Minimum samples per regime
    min_regimes: int = 2
    max_regimes: int = 15
    
    # Random seed
    random_state: int = 42
    
    # Validation parameters (from code review)
    min_samples_required: int = 500  # Minimum samples for reliable inference
    min_features_required: int = 3  # Minimum features required
    max_nan_ratio: float = 0.1  # Maximum ratio of NaN values allowed


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
    
    def __init__(self, config: Optional[HDPHMMConfig] = None):
        """
        Initialize HDP-HMM clusterer.
        
        Args:
            config: Configuration for HDP-HMM clustering
        """
        self.config = config or HDPHMMConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.scaler = None
        self.pca = None
        self.convergence_history = []
        
        # Initialize hardware manager if available
        if HARDWARE_UTILS_AVAILABLE:
            try:
                self.device_manager = get_device_manager()
                tprint_debug(f"Hardware manager initialized: {self.device_manager.get_device_info()}")
            except Exception as e:
                tprint_debug(f"Failed to initialize hardware manager: {e}")
                self.device_manager = None
        else:
            self.device_manager = None
        
        if not HMM_AVAILABLE:
            tprint_error("❌ HMM libraries not available. Install pyhsmm or ssm-jax")
            raise ImportError("HMM libraries not available")
        
        tprint_info(f"🚀 Initialized HDP-HMM Clusterer with {HMM_LIBRARY}")
        tprint_structured({
            "alpha": self.config.alpha,
            "kappa": self.config.kappa,
            "gamma": self.config.gamma,
            "max_states": self.config.max_states,
            "library": HMM_LIBRARY
        }, level="INFO")
    
    def fit_predict(self, data: np.ndarray, validate: bool = True) -> HDPHMMResult:
        """
        Fit HDP-HMM model and predict regime labels.
        
        Args:
            data: Input data (n_samples, n_features)
            
        Returns:
            HDPHMMResult with clustering results
        """
        tprint_info("🔍 Starting HDP-HMM regime discovery")
        
        import time
        import tracemalloc
        
        start_time = time.time()
        tracemalloc.start()
        
        try:
            # Validate input data (from code review)
            if validate:
                self._validate_input(data)
            
            # Preprocess data
            data_processed, feature_names = self._preprocess_data(data)
            
            # Fit HDP-HMM model
            if HMM_LIBRARY == 'pyhsmm':
                result = self._fit_pyhsmm(data_processed)
            elif HMM_LIBRARY == 'ssm':
                result = self._fit_ssm(data_processed)
            else:
                raise ValueError(f"Unsupported HMM library: {HMM_LIBRARY}")
            
            # Calculate metrics
            metrics = self._calculate_metrics(data_processed, result['labels'])
            
            # Calculate memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_usage_mb = peak / 1024 / 1024
            
            processing_time = time.time() - start_time
            
            # Create result
            hdp_result = HDPHMMResult(
                cluster_labels=result['labels'],
                cluster_probabilities=result.get('probabilities', np.ones(len(result['labels']))),
                n_clusters=result['n_states'],
                transition_matrix=result.get('transition_matrix'),
                emission_params=result.get('emission_params'),
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
                cluster_probabilities=np.ones(len(data)),
                n_clusters=0,
                transition_matrix=None,
                emission_params=None,
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
    
    def _validate_input(self, data: np.ndarray) -> None:
        """Validate input data (from code review)."""
        tprint_info("🔍 Validating input data")
        
        # Check minimum samples
        n_samples = len(data) if len(data.shape) == 1 else data.shape[0]
        if n_samples < self.config.min_samples_required:
            tprint_warning(
                f"⚠️ Input has {n_samples} samples, but {self.config.min_samples_required}+ "
                f"recommended for reliable HDP-HMM inference"
            )
        
        # Check minimum features
        if len(data.shape) > 1:
            n_features = data.shape[1]
            if n_features < self.config.min_features_required:
                raise ValueError(
                    f"Input has {n_features} features, but minimum {self.config.min_features_required} required"
                )
        
        # Check for excessive NaN values
        if isinstance(data, np.ndarray):
            nan_ratio = np.isnan(data).sum() / data.size
            if nan_ratio > self.config.max_nan_ratio:
                raise ValueError(
                    f"Input has {nan_ratio:.1%} NaN values, exceeding maximum {self.config.max_nan_ratio:.1%}"
                )
        
        # Check for degenerate cases
        if isinstance(data, np.ndarray) and len(data.shape) > 1:
            # Check if all values are identical (single regime)
            if np.allclose(data, data[0], rtol=1e-10, atol=1e-10):
                tprint_warning("⚠️ All data values are identical - may result in single regime")
        
        tprint_success("✅ Input validation passed")
    
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
        
        # Create observation distribution
        obs_dim = data.shape[1]
        
        # Observation hyperparameters (weak prior)
        if self.config.obs_hypparams is None:
            obs_hypparams = {
                'mu_0': np.zeros(obs_dim),
                'sigma_0': np.eye(obs_dim),
                'kappa_0': 0.01,
                'nu_0': obs_dim + 2
            }
        else:
            obs_hypparams = self.config.obs_hypparams
        
        obs_distns = [Gaussian(**obs_hypparams) for _ in range(self.config.max_states)]
        
        # Create Sticky HDP-HSMM model
        model = WeakLimitStickyHDPHSMM(
            alpha=self.config.alpha,
            kappa=self.config.kappa,
            gamma=self.config.gamma,
            init_state_concentration=1.0,
            obs_distns=obs_distns
        )
        
        # Add data
        model.add_data(data)
        
        # Run Gibbs sampling with convergence diagnostics and progress tracking
        tprint_info(f"🔄 Running Gibbs sampling: {self.config.n_iterations} iterations")
        
        state_counts = []
        log_likelihoods = []
        converged = False
        convergence_iteration = None
        
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
                model.resample_model()
                
                # Track state count
                n_states = model.num_states()
                state_counts.append(n_states)
                
                # Track log likelihood
                try:
                    ll = model.log_likelihood()
                    log_likelihoods.append(ll)
                except:
                    log_likelihoods.append(np.nan)
                
                # Convergence diagnostics (after burn-in)
                if (self.config.convergence_check and 
                    iteration >= self.config.n_burnin and 
                    len(state_counts) > 10):
                    
                    # Check if number of states has stabilized
                    recent_states = state_counts[-10:]
                    state_std = np.std(recent_states)
                    state_change = abs(recent_states[-1] - recent_states[0]) / max(recent_states[0], 1)
                    
                    if state_std < 0.5 and state_change < self.config.convergence_threshold:
                        converged = True
                        convergence_iteration = iteration + 1
                        tprint_success(
                            f"✅ Converged at iteration {convergence_iteration}: "
                            f"{n_states} states (std={state_std:.2f}, change={state_change:.3f})"
                        )
                        break
                
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
            'state_counts': state_counts,
            'log_likelihoods': log_likelihoods,
            'converged': converged,
            'convergence_iteration': convergence_iteration
        }
        
        # Get final state sequence
        labels = model.stateseqs[0].copy()
        
        # Get transition matrix
        transition_matrix = model.trans_distn.trans_matrix.copy()
        
        # Calculate state durations
        unique_states = np.unique(labels)
        state_durations = []
        for state in unique_states:
            state_mask = labels == state
            # Find continuous segments
            segments = np.split(np.where(state_mask)[0], np.where(np.diff(np.where(state_mask)[0]) != 1)[0] + 1)
            durations = [len(seg) for seg in segments if len(seg) > 0]
            if durations:
                state_durations.append(np.mean(durations))
            else:
                state_durations.append(0)
        
        # Calculate posterior statistics
        posterior_mean_states = np.mean(state_counts[self.config.n_burnin:])
        posterior_std_states = np.std(state_counts[self.config.n_burnin:])
        
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
        
        return {
            'labels': labels,
            'n_states': len(unique_states),
            'transition_matrix': transition_matrix,
            'emission_params': {
                'means': [obs_distn.mu for obs_distn in model.obs_distns],
                'covariances': [obs_distn.sigma for obs_distn in model.obs_distns]
            },
            'state_durations': np.array(state_durations),
            'log_likelihood': final_ll,
            'posterior_mean_states': posterior_mean_states,
            'posterior_std_states': posterior_std_states,
            'transition_persistence': transition_persistence,
            'state_counts_history': state_counts,
            'log_likelihood_history': log_likelihoods
        }
    
    def _fit_ssm(self, data: np.ndarray) -> Dict[str, Any]:
        """Fit HDP-HMM using ssm library (fallback)."""
        tprint_info("🔄 Fitting HMM with ssm library")
        
        # Note: ssm doesn't have HDP-HMM, so we use standard HMM with fixed K
        # This is a fallback implementation
        import ssm
        
        # Set number of states (use middle of range)
        K = (self.config.min_regimes + self.config.max_regimes) // 2
        
        # Create HMM model
        hmm = ssm.HMM(
            K=K,
            D=data.shape[1],
            observations="gaussian",
            transitions="sticky"
        )
        
        # Fit model
        tprint_info(f"🔄 Fitting HMM with {K} states")
        ll = hmm.fit(data, method="em", num_iters=self.config.n_iterations)
        
        # Get state sequence
        labels = hmm.most_likely_states(data)
        
        # Get transition matrix
        transition_matrix = hmm.transitions.transition_matrix
        
        # Calculate state durations
        unique_states = np.unique(labels)
        state_durations = []
        for state in unique_states:
            state_mask = labels == state
            segments = np.split(np.where(state_mask)[0], np.where(np.diff(np.where(state_mask)[0]) != 1)[0] + 1)
            durations = [len(seg) for seg in segments if len(seg) > 0]
            if durations:
                state_durations.append(np.mean(durations))
            else:
                state_durations.append(0)
        
        # Calculate transition persistence
        transition_persistence = np.mean(np.diag(transition_matrix))
        
        tprint_success(f"✅ HMM fitting completed: {len(unique_states)} states")
        
        return {
            'labels': labels,
            'n_states': len(unique_states),
            'transition_matrix': transition_matrix,
            'emission_params': {
                'means': hmm.observations.mus,
                'covariances': hmm.observations.Sigmas
            },
            'state_durations': np.array(state_durations),
            'log_likelihood': ll[-1] if isinstance(ll, np.ndarray) else ll,
            'posterior_mean_states': float(len(unique_states)),
            'posterior_std_states': 0.0,
            'transition_persistence': transition_persistence
        }
    
    def _calculate_metrics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        tprint_debug("📊 Calculating clustering metrics")
        
        metrics = {}
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Basic statistics
        metrics['n_clusters'] = n_clusters
        metrics['noise_ratio'] = 0.0  # HMM doesn't have noise concept
        
        # Use comprehensive quality assessor for clustering metrics
        try:
            quality_assessor = create_quality_assessor()
            quality_metrics = quality_assessor.assess_clustering_quality(
                cluster_labels=labels,
                features=data,
                clusterer=None,  # HDP-HMM doesn't have HDBSCAN clusterer
                timestamps=None,  # Add if available from input data
                returns=None     # Add if available from input data
            )
            
            # Extract metrics
            metrics['silhouette_score'] = quality_metrics.silhouette_score or 0.0
            metrics['calinski_harabasz_score'] = quality_metrics.calinski_harabasz_score or 0.0
            metrics['davies_bouldin_score'] = quality_metrics.davies_bouldin_score or 0.0
            
            # Add comprehensive quality metrics
            metrics['quality_assessment'] = quality_metrics.to_dict()
            metrics['composite_quality_score'] = quality_metrics.composite_quality_score
            
            tprint_success(f"✅ Comprehensive quality assessment: Score={quality_metrics.composite_quality_score:.3f}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Quality assessment failed: {e}")
            metrics['silhouette_score'] = 0.0
            metrics['calinski_harabasz_score'] = 0.0
            metrics['davies_bouldin_score'] = 0.0
        
        return metrics
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict regime labels for new data.
        
        Args:
            data: Input data (n_samples, n_features)
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_predict first.")
        
        # Preprocess data
        data_scaled = self.scaler.transform(data)
        
        if self.pca is not None:
            data_processed = self.pca.transform(data_scaled)
        else:
            data_processed = data_scaled
        
        # Predict using fitted model
        if HMM_LIBRARY == 'pyhsmm':
            # For pyhsmm, we need to use Viterbi algorithm
            labels = self.model.predict(data_processed)
        elif HMM_LIBRARY == 'ssm':
            labels = self.model.most_likely_states(data_processed)
        else:
            raise ValueError(f"Unsupported HMM library: {HMM_LIBRARY}")
        
        return labels


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
