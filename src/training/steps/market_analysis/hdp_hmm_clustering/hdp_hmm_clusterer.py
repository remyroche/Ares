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

# Try to import HMM libraries
HMM_AVAILABLE = False
HMM_LIBRARY = None

try:
    import pyhsmm
    from pyhsmm.models import WeakLimitHDPHSMM, WeakLimitStickyHDPHSMM
    from pyhsmm.basic.distributions import Gaussian
    HMM_AVAILABLE = True
    HMM_LIBRARY = 'pyhsmm'
    tprint_info("✅ Using pyhsmm for HDP-HMM clustering")
except ImportError:
    try:
        import ssm
        HMM_AVAILABLE = True
        HMM_LIBRARY = 'ssm'
        tprint_info("✅ Using ssm for HDP-HMM clustering")
    except ImportError:
        warnings.warn("Neither pyhsmm nor ssm available. Install with: pip install pyhsmm or pip install ssm-jax")
        HMM_LIBRARY = None


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
    
    def fit_predict(self, data: np.ndarray) -> HDPHMMResult:
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
    
    def _preprocess_data(self, data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Preprocess data with scaling and optional PCA."""
        tprint_info("🔧 Preprocessing data for HDP-HMM")
        
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
        
        # Run Gibbs sampling
        tprint_info(f"🔄 Running Gibbs sampling: {self.config.n_iterations} iterations")
        
        state_counts = []
        log_likelihoods = []
        
        for iteration in range(self.config.n_iterations):
            model.resample_model()
            
            # Track state count
            state_counts.append(model.num_states())
            
            # Track log likelihood
            try:
                ll = model.log_likelihood()
                log_likelihoods.append(ll)
            except:
                log_likelihoods.append(np.nan)
            
            if (iteration + 1) % 20 == 0:
                tprint_info(f"   Iteration {iteration + 1}/{self.config.n_iterations}: {model.num_states()} states")
        
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
        
        tprint_success(f"✅ Gibbs sampling completed: {len(unique_states)} final states")
        
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
        
        # Clustering quality metrics
        if n_clusters > 1:
            try:
                metrics['silhouette_score'] = silhouette_score(data, labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(data, labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(data, labels)
            except Exception as e:
                tprint_warning(f"⚠️ Could not calculate some metrics: {e}")
                metrics['silhouette_score'] = 0.0
                metrics['calinski_harabasz_score'] = 0.0
                metrics['davies_bouldin_score'] = 0.0
        else:
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
