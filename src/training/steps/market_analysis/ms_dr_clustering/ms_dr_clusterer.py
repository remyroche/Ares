"""
Markov-Switching Dynamic Regression (MS-DR) Regime Discovery

Implements Markov-Switching models with regime-dependent dynamics for regime clustering.
This approach explicitly models regime-dependent relationships and transitions.

Key Features:
- Regime-dependent dynamics modeling
- Explicit transition probabilities
- Handles heteroskedasticity across regimes
- Economic interpretability

Libraries: Uses statsmodels.tsa.regime_switching
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

# Try to import Markov-Switching models
MS_AVAILABLE = False
MS_LIBRARY = None
MS_INSTALLATION_GUIDE = """
🔧 Markov-Switching Library Installation Guide:

1. statsmodels (Recommended):
   pip install statsmodels>=0.13.0
   
2. With conda:
   conda install -c conda-forge statsmodels

Note: statsmodels is well-maintained and easy to install.
No complex dependencies required.
"""

try:
    from statsmodels.tsa.regime_switching import markov_switching, markov_autoregression, markov_regression
    from statsmodels.tsa.regime_switching.markov_switching import MarkovSwitching
    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    MS_AVAILABLE = True
    MS_LIBRARY = 'statsmodels'
    tprint_success("✅ Using statsmodels for Markov-Switching clustering")
except ImportError:
    tprint_warning("⚠️ statsmodels not available")
    tprint_warning(MS_INSTALLATION_GUIDE)
    MS_LIBRARY = None

# Import existing optimization utilities (from code review)
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
class MSDRConfig:
    """Configuration for Markov-Switching Dynamic Regression."""
    # Model parameters
    n_regimes: int = 5  # Number of regimes (can be optimized)
    switching_variance: bool = True  # Allow variance to switch across regimes
    switching_trend: bool = True  # Allow trend to switch across regimes
    
    # Model type
    model_type: str = 'autoregression'  # 'autoregression', 'regression', 'dynamic_factor'
    order: int = 1  # Autoregression order (for AR models)
    
    # Optimization parameters
    max_iter: int = 1000
    method: str = 'powell'  # Optimization method: 'powell', 'bfgs', 'nm'
    
    # Preprocessing
    enable_pca: bool = True
    pca_components: int = 10
    pca_variance_threshold: float = 0.95
    
    # Model selection
    auto_select_regimes: bool = True  # Auto-select number of regimes using IC
    min_regimes: int = 2
    max_regimes: int = 10
    ic_criterion: str = 'aic'  # Information criterion: 'aic', 'bic', 'hqic'
    
    # Validation (enhanced from code review)
    min_regime_size: int = 10
    min_samples_required: int = 200  # Minimum samples for reliable estimation
    min_features_required: int = 1  # Minimum features required
    max_nan_ratio: float = 0.1  # Maximum ratio of NaN values allowed
    show_progress: bool = True  # Show progress during optimization
    
    # Random seed
    random_state: int = 42


@dataclass
class MSDRResult:
    """Result container for Markov-Switching Dynamic Regression."""
    # Clustering results
    cluster_labels: np.ndarray
    cluster_probabilities: np.ndarray  # Smoothed probabilities
    n_clusters: int
    
    # Model artifacts
    transition_matrix: Optional[np.ndarray]
    regime_params: Optional[Dict[str, Any]]
    regime_variances: Optional[np.ndarray]
    
    # Quality metrics
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    noise_ratio: float
    log_likelihood: float
    
    # Model selection metrics
    aic: float
    bic: float
    hqic: float
    
    # Regime statistics
    regime_durations: Optional[np.ndarray]
    transition_persistence: float  # Average self-transition probability
    
    # Processing metadata
    processing_time: float
    memory_usage_mb: float
    feature_names: List[str]
    success: bool
    error_message: Optional[str] = None
    
    # Model metadata
    metadata: Optional[Dict[str, Any]] = None


class MSDRClusterer:
    """
    Markov-Switching Dynamic Regression Clusterer for regime discovery.
    
    This class implements regime clustering using Markov-Switching models
    that explicitly model regime-dependent dynamics and transitions.
    """
    
    def __init__(self, config: Optional[MSDRConfig] = None):
        """
        Initialize MS-DR clusterer.
        
        Args:
            config: Configuration for MS-DR clustering
        """
        self.config = config or MSDRConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.scaler = None
        self.pca = None
        self.fitted_models = {}  # Store models for different regime counts
        
        # Initialize hardware manager if available (from code review)
        if HARDWARE_UTILS_AVAILABLE:
            try:
                self.device_manager = get_device_manager()
                tprint_debug(f"Hardware manager initialized: {self.device_manager.get_device_info()}")
            except Exception as e:
                tprint_debug(f"Failed to initialize hardware manager: {e}")
                self.device_manager = None
        else:
            self.device_manager = None
        
        if not MS_AVAILABLE:
            tprint_error("❌ Markov-Switching models not available. Install statsmodels")
            raise ImportError("statsmodels.tsa.regime_switching not available")
        
        tprint_info(f"🚀 Initialized MS-DR Clusterer with {MS_LIBRARY}")
        tprint_structured({
            "n_regimes": self.config.n_regimes,
            "model_type": self.config.model_type,
            "switching_variance": self.config.switching_variance,
            "auto_select_regimes": self.config.auto_select_regimes,
            "library": MS_LIBRARY
        }, level="INFO")
    
    def fit_predict(self, data: np.ndarray, validate: bool = True) -> MSDRResult:
        """
        Fit MS-DR model and predict regime labels.
        
        Args:
            data: Input data (n_samples, n_features) or time series
            
        Returns:
            MSDRResult with clustering results
        """
        tprint_info("🔍 Starting Markov-Switching regime discovery")
        
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
            
            # Auto-select number of regimes if enabled
            if self.config.auto_select_regimes:
                n_regimes = self._select_optimal_regimes(data_processed)
                tprint_info(f"📊 Optimal number of regimes: {n_regimes}")
            else:
                n_regimes = self.config.n_regimes
            
            # Fit MS-DR model
            result = self._fit_ms_model(data_processed, n_regimes)
            
            # Calculate metrics
            metrics = self._calculate_metrics(data_processed, result['labels'])
            
            # Calculate memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_usage_mb = peak / 1024 / 1024
            
            processing_time = time.time() - start_time
            
            # Create result
            ms_result = MSDRResult(
                cluster_labels=result['labels'],
                cluster_probabilities=result.get('probabilities', np.ones((len(result['labels']), n_regimes))),
                n_clusters=result['n_regimes'],
                transition_matrix=result.get('transition_matrix'),
                regime_params=result.get('regime_params'),
                regime_variances=result.get('regime_variances'),
                silhouette_score=metrics.get('silhouette_score', 0.0),
                calinski_harabasz_score=metrics.get('calinski_harabasz_score', 0.0),
                davies_bouldin_score=metrics.get('davies_bouldin_score', 0.0),
                noise_ratio=metrics.get('noise_ratio', 0.0),
                log_likelihood=result.get('log_likelihood', 0.0),
                aic=result.get('aic', 0.0),
                bic=result.get('bic', 0.0),
                hqic=result.get('hqic', 0.0),
                regime_durations=result.get('regime_durations'),
                transition_persistence=result.get('transition_persistence', 0.0),
                processing_time=processing_time,
                memory_usage_mb=memory_usage_mb,
                feature_names=feature_names,
                success=True,
                metadata={
                    'config': self.config.__dict__,
                    'library': MS_LIBRARY,
                    'preprocessing': {
                        'scaled': True,
                        'pca_applied': self.pca is not None
                    }
                }
            )
            
            tprint_success(f"✅ MS-DR completed: {ms_result.n_clusters} regimes discovered")
            tprint_structured({
                "n_regimes": ms_result.n_clusters,
                "silhouette_score": ms_result.silhouette_score,
                "aic": ms_result.aic,
                "bic": ms_result.bic,
                "transition_persistence": ms_result.transition_persistence,
                "processing_time": f"{processing_time:.2f}s"
            }, level="INFO")
            
            return ms_result
            
        except Exception as e:
            tprint_error(f"❌ MS-DR clustering failed: {e}")
            self.logger.error(f"MS-DR clustering error: {e}", exc_info=True)
            
            # Return failure result
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            return MSDRResult(
                cluster_labels=np.zeros(len(data)),
                cluster_probabilities=np.ones((len(data), 1)),
                n_clusters=0,
                transition_matrix=None,
                regime_params=None,
                regime_variances=None,
                silhouette_score=0.0,
                calinski_harabasz_score=0.0,
                davies_bouldin_score=0.0,
                noise_ratio=1.0,
                log_likelihood=0.0,
                aic=np.inf,
                bic=np.inf,
                hqic=np.inf,
                regime_durations=None,
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
                f"recommended for reliable MS-DR estimation"
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
        
        # Check for degenerate cases (all identical values)
        if isinstance(data, np.ndarray):
            data_flat = data.flatten()
            if len(np.unique(data_flat[~np.isnan(data_flat)])) == 1:
                raise ValueError("All data values are identical - cannot fit MS-DR model")
        
        tprint_success("✅ Input validation passed")
    
    def _preprocess_data(self, data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Preprocess data with scaling and optional PCA."""
        tprint_info("🔧 Preprocessing data for MS-DR")
        tprint_data_preview(data, "Input Data", max_rows=3, max_cols=5)
        
        # Handle DataFrame input
        if isinstance(data, pd.DataFrame):
            feature_names = data.columns.tolist()
            data = data.values
        else:
            feature_names = [f'feature_{i}' for i in range(data.shape[1]) if len(data.shape) > 1] or ['target']
        
        # If data is 1D, use as time series
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Standardize
        self.scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data)
        
        # Apply PCA if enabled and data has multiple features
        if self.config.enable_pca and data.shape[1] > 1 and data.shape[1] > self.config.pca_components:
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
        
        # For MS models, we typically use the first principal component or average
        if data_processed.shape[1] > 1:
            tprint_info("📊 Using first principal component for MS model")
            data_processed = data_processed[:, 0].reshape(-1, 1)
            feature_names = ['pc1']
        
        tprint_success(f"✅ Preprocessed data shape: {data_processed.shape}")
        tprint_data_format(data_processed, "Preprocessed Data", check_compatibility=True)
        return data_processed, feature_names
    
    def _select_optimal_regimes(self, data: np.ndarray) -> int:
        """Select optimal number of regimes using information criteria."""
        tprint_info("🔍 Selecting optimal number of regimes")
        
        ic_values = {}
        n_candidates = self.config.max_regimes - self.config.min_regimes + 1
        
        # Progress tracking (from code review)
        try:
            from tqdm import tqdm
            iterator = tqdm(
                range(self.config.min_regimes, self.config.max_regimes + 1),
                desc="Model Selection",
                disable=not self.config.show_progress
            )
        except ImportError:
            tprint_debug("tqdm not available, showing periodic updates")
            iterator = range(self.config.min_regimes, self.config.max_regimes + 1)
        
        with tprint_timer("Model Selection", level="PERFORMANCE"):
            for k in iterator:
                try:
                    result = self._fit_ms_model(data, k, store_model=True)
                    
                    ic_value = result.get(self.config.ic_criterion, np.inf)
                    ic_values[k] = ic_value
                    
                    # Update progress
                    if hasattr(iterator, 'set_postfix'):
                        iterator.set_postfix({
                            'k': k,
                            self.config.ic_criterion.upper(): f"{ic_value:.1f}"
                        })
                    else:
                        tprint_info(f"   k={k}: {self.config.ic_criterion.upper()}={ic_value:.2f}")
                    
                except Exception as e:
                    tprint_warning(f"   k={k}: failed ({e})")
                    ic_values[k] = np.inf
        
        # Select k with minimum IC
        optimal_k = min(ic_values, key=ic_values.get)
        
        # Log selection results
        tprint_structured({
            'optimal_k': optimal_k,
            'criterion': self.config.ic_criterion.upper(),
            'optimal_value': ic_values[optimal_k],
            'all_values': ic_values
        }, level="INFO")
        
        tprint_success(f"✅ Optimal regimes selected: {optimal_k}")
        
        return optimal_k
    
    def _fit_ms_model(self, data: np.ndarray, n_regimes: int, store_model: bool = False) -> Dict[str, Any]:
        """Fit Markov-Switching model."""
        tprint_info(f"🔄 Fitting MS model with {n_regimes} regimes")
        
        # Ensure data is 1D for MS models
        if len(data.shape) > 1 and data.shape[1] == 1:
            data_series = data.flatten()
        else:
            data_series = data.flatten()
        
        # Create pandas Series for statsmodels
        ts_data = pd.Series(data_series)
        
        # Fit Markov-Switching model based on type
        if self.config.model_type == 'autoregression':
            model = MarkovAutoregression(
                ts_data,
                k_regimes=n_regimes,
                order=self.config.order,
                switching_variance=self.config.switching_variance
            )
        elif self.config.model_type == 'regression':
            # For regression, we need exogenous variables
            # Use lagged values as predictors
            exog = pd.DataFrame({
                f'lag_{i+1}': ts_data.shift(i+1) 
                for i in range(self.config.order)
            }).dropna()
            ts_data_aligned = ts_data.iloc[self.config.order:]
            
            model = MarkovRegression(
                ts_data_aligned,
                k_regimes=n_regimes,
                exog=exog,
                switching_variance=self.config.switching_variance
            )
        else:
            # Default to autoregression
            model = MarkovAutoregression(
                ts_data,
                k_regimes=n_regimes,
                order=self.config.order,
                switching_variance=self.config.switching_variance
            )
        
        # Fit model
        try:
            fitted_model = model.fit(
                maxiter=self.config.max_iter,
                method=self.config.method,
                disp=False
            )
            
            if store_model:
                self.fitted_models[n_regimes] = fitted_model
                self.model = fitted_model
            
            # Get regime probabilities (smoothed)
            smoothed_probs = fitted_model.smoothed_marginal_probabilities
            
            # Get most likely regime sequence
            labels = np.argmax(smoothed_probs.values, axis=1)
            
            # Get transition matrix
            transition_matrix = fitted_model.regime_transition
            
            # Get regime parameters
            regime_params = {}
            for i in range(n_regimes):
                regime_params[f'regime_{i}'] = {
                    'mean': fitted_model.params.get(f'const[{i}]', 0.0),
                    'ar_coefs': [
                        fitted_model.params.get(f'ar.L{j+1}[{i}]', 0.0) 
                        for j in range(self.config.order)
                    ] if self.config.model_type == 'autoregression' else []
                }
            
            # Get regime variances if switching
            if self.config.switching_variance:
                regime_variances = np.array([
                    fitted_model.params.get(f'sigma2[{i}]', 1.0)
                    for i in range(n_regimes)
                ])
            else:
                regime_variances = np.array([fitted_model.params.get('sigma2', 1.0)] * n_regimes)
            
            # Calculate regime durations
            regime_durations = []
            for i in range(n_regimes):
                regime_mask = labels == i
                segments = np.split(np.where(regime_mask)[0], 
                                  np.where(np.diff(np.where(regime_mask)[0]) != 1)[0] + 1)
                durations = [len(seg) for seg in segments if len(seg) > 0]
                if durations:
                    regime_durations.append(np.mean(durations))
                else:
                    regime_durations.append(0)
            
            # Calculate transition persistence
            transition_persistence = np.mean(np.diag(transition_matrix))
            
            tprint_success(f"✅ MS model fitted: {n_regimes} regimes")
            
            return {
                'labels': labels,
                'probabilities': smoothed_probs.values,
                'n_regimes': n_regimes,
                'transition_matrix': transition_matrix,
                'regime_params': regime_params,
                'regime_variances': regime_variances,
                'regime_durations': np.array(regime_durations),
                'log_likelihood': fitted_model.llf,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'hqic': fitted_model.hqic,
                'transition_persistence': transition_persistence,
                'model': fitted_model
            }
            
        except Exception as e:
            tprint_error(f"❌ MS model fitting failed: {e}")
            raise
    
    def _calculate_metrics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        tprint_debug("📊 Calculating clustering metrics")
        
        metrics = {}
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Basic statistics
        metrics['n_clusters'] = n_clusters
        metrics['noise_ratio'] = 0.0  # MS models don't have noise concept
        
        # For metrics, we need multi-dimensional data
        # Use the regime probabilities as features
        if data.shape[1] == 1 and hasattr(self, 'model') and self.model is not None:
            # Use smoothed probabilities as feature space
            data_for_metrics = self.model.smoothed_marginal_probabilities.values
        else:
            data_for_metrics = data
        
        # Use comprehensive quality assessor for clustering metrics
        try:
            quality_assessor = create_quality_assessor()
            quality_metrics = quality_assessor.assess_clustering_quality(
                cluster_labels=labels,
                features=data_for_metrics,
                clusterer=None,  # MS-DR doesn't have HDBSCAN clusterer
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
        
        # For MS models, use first component
        if data_processed.shape[1] > 1:
            data_processed = data_processed[:, 0].reshape(-1, 1)
        
        # Create time series
        ts_data = pd.Series(data_processed.flatten())
        
        # Get filtered probabilities (one-step ahead)
        filtered_probs = self.model.filtered_marginal_probabilities(ts_data)
        
        # Get most likely regime
        labels = np.argmax(filtered_probs, axis=1)
        
        return labels


# Convenience functions
def create_ms_dr_clusterer(
    n_regimes: int = 5,
    model_type: str = 'autoregression',
    order: int = 1,
    switching_variance: bool = True,
    auto_select_regimes: bool = True,
    min_regimes: int = 2,
    max_regimes: int = 10,
    enable_pca: bool = True,
    pca_components: int = 10,
    random_state: int = 42
) -> MSDRClusterer:
    """
    Create MS-DR clusterer with specified parameters.
    
    Args:
        n_regimes: Number of regimes (if not auto-selecting)
        model_type: Model type ('autoregression', 'regression')
        order: Autoregression order
        switching_variance: Allow variance to switch across regimes
        auto_select_regimes: Auto-select number of regimes using IC
        min_regimes: Minimum number of regimes
        max_regimes: Maximum number of regimes
        enable_pca: Enable PCA reduction
        pca_components: Number of PCA components
        random_state: Random seed
        
    Returns:
        MSDRClusterer instance
    """
    config = MSDRConfig(
        n_regimes=n_regimes,
        model_type=model_type,
        order=order,
        switching_variance=switching_variance,
        auto_select_regimes=auto_select_regimes,
        min_regimes=min_regimes,
        max_regimes=max_regimes,
        enable_pca=enable_pca,
        pca_components=pca_components,
        random_state=random_state
    )
    
    return MSDRClusterer(config)


__all__ = [
    'MSDRClusterer',
    'MSDRConfig',
    'MSDRResult',
    'create_ms_dr_clusterer',
    'MS_AVAILABLE',
    'MS_LIBRARY'
]
