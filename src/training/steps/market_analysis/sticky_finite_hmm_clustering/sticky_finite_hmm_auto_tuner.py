"""
Sticky Finite HMM Auto-Tuner

This module provides automatic hyperparameter tuning for Sticky Finite HMM clustering using
a hierarchical multi-stage optimization approach:
1. Coarse Grid Search - Broad exploration of parameter space
2. Fine Grid Search - Refinement around best coarse results
3. TPE (Bayesian Optimization) - Final optimization with Optuna

The objective is to maximize the composite_score from cluster_quality_assessor.py
which combines multiple quality metrics (silhouette, DBI, CV ratio, balance, temporal).

Uses HierarchicalParameterOptimizer for efficient tuning of 13 parameters organized in 6 groups.

Usage:
    ```python
    from src.training.steps.market_analysis.sticky_finite_hmm_clustering import run_sticky_finite_hmm_auto_tuning
    
    best_params, best_score, tuning_results = run_sticky_finite_hmm_auto_tuning(
        market_data=df,
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h",
        use_hierarchical=True,  # ✅ 3-5x faster
        tpe_trials=100,
        timeout=3600
    )
    ```
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import logging

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)

# Import optimization utilities
try:
    from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
        HierarchicalParameterOptimizer,
        ParameterGroup,
        OptimizationStage,
        OptimizationBackend,
        StageConfig,
        create_param_group
    )
    _hierarchical_hpo_available = True
except ImportError as e:
    tprint_warning(f"⚠️ Hierarchical optimizer not available: {e}")
    _hierarchical_hpo_available = False

# Import Pareto optimization
try:
    from src.utils.ml_common.optimization.pareto import (
        ParetoOptimizer,
        Solution,
        compute_pareto_front,
        select_knee_point,
        ObjectiveDirection,
        _dominates as dominates
    )
    _pareto_available = True
except ImportError as e:
    tprint_warning(f"⚠️ Pareto optimizer not available: {e}")
    _pareto_available = False

# Import artifact manager
try:
    from src.utils.artifact_manager import ArtifactManager
    _artifact_manager_available = True
except ImportError as e:
    tprint_warning(f"⚠️ Artifact manager not available: {e}")
    _artifact_manager_available = False

# Import Bayesian TPE optimizer
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer,
        OptimizationConfig,
        OPTUNA_AVAILABLE
    )
    _optuna_available = OPTUNA_AVAILABLE
except ImportError as e:
    tprint_warning(f"⚠️ Bayesian TPE optimizer not available: {e}")
    _optuna_available = False

# Import grid search utilities
try:
    from src.utils.ml_common.optimization.grid_utils import (
        build_coarse_grid_from_search_space,
        build_fine_grid_around_best
    )
    _grid_utils_available = True
except ImportError as e:
    tprint_warning(f"⚠️ Grid utilities not available: {e}")
    _grid_utils_available = False

# Import ClusterQualityAssessor for comprehensive quality assessment
try:
    from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
        ClusterQualityAssessor,
        ClusterQualityMetrics
    )
    _quality_assessor_available = True
except ImportError as e:
    tprint_warning(f"⚠️ ClusterQualityAssessor not available: {e}")
    _quality_assessor_available = False

# Check overall optimization availability
_optimization_available = (
    _hierarchical_hpo_available or 
    _pareto_available or 
    _optuna_available or 
    _grid_utils_available
)

# Import Sticky Finite HMM components
from .sticky_finite_hmm_clusterer import StickyFiniteHMMClusterer, StickyFiniteHMMConfig
from .standalone_runner import run_sticky_finite_hmm_clustering

# Import quality assessment
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    ClusterQualityAssessor,
    create_cluster_quality_assessor
)
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    DEFAULT_CLUSTERING_GOALS,
    DEFAULT_OPTIMIZATION_TARGETS
)

# Import artifact manager
try:
    from src.utils.artifact_manager import ArtifactManager
    artifact_manager_available = True
except ImportError:
    artifact_manager_available = False
    tprint_warning("⚠️ Artifact manager not available")


@dataclass
class StickyFiniteHMMSearchSpace:
    """
    Focused search space for Sticky Finite HMM hyperparameters.
    
    OPTIMIZED PARAMETERS (5 most important):
    ==========================================
    
    I. Model Structure:
        K: Number of fixed states (4-7)
           - Determines model capacity and regime count
           - K=4: Simple, fast, may underfit
           - K=7: Complex, slower, may overfit
        
        n_mixtures: Gaussian mixtures per state (1-3)
           - 1: Single Gaussian (fast, ~30-40s, simple regimes)
           - 2: Two components (moderate, ~50-70s, bimodal regimes)
           - 3: Three components (slow, ~80-120s, complex distributions)
        
        pca_components: Dimensionality reduction (10-14)
           - Too low: lose information, poor separation
           - Too high: noise, overfitting
    
    II. Transition Dynamics:
        kappa: Stickiness parameter (5-50)
           - Controls regime persistence/duration
           - kappa=10 → ~11 timesteps
           - kappa=30 → ~28 timesteps
           - kappa=50 → ~44 timesteps
        
        base_alpha: Off-diagonal concentration (0.1-1.0)
           - Controls transition sparsity
           - Low (0.1): sparse transitions, infrequent regime changes
           - High (1.0): more uniform transitions, frequent changes
    
    III. Training:
        lr: Learning rate (1e-4 to 1e-1, log scale)
           - Controls SVI optimization speed/stability
           - Too high: unstable ELBO, poor convergence
           - Too low: very slow convergence, may not reach optimum
    
    
    FIXED PARAMETERS (automatically set):
    =====================================
    
    num_iters: 1000 (sufficient for convergence with early stopping)
    
    num_particles: 10
       - Particles for gradient estimation in SVI
       - More particles = better gradient estimates but slower
       - 10 is good balance for this problem size
    
    prior_mean_scale: 10.0
       - Prior std for emission means: Normal(0, prior_mean_scale)
       - Controls how far state means can be from zero
       - 10.0 works well for standardized PCA features
    
    prior_cov_scale: 1.0
       - Prior std for log emission scales: LogNormal(0, prior_cov_scale)
       - Controls emission variance
       - 1.0 is reasonable for standardized features
    
    patience: 50
       - Iterations to wait without ELBO improvement before stopping
       - Prevents overfitting and saves time
       - 50 is robust (balances speed vs thoroughness)
    
    elbo_improvement_threshold: 1e-3
       - Minimum ELBO improvement over convergence_window (10 iters)
       - Lower = stricter convergence criteria
       - 1e-3 is good balance (not too strict, not too loose)
    
    min_features: 50
    max_features: 100
       - Feature selection from Feature Bank (~140 total)
       - min_features=50: ensures adequate signal
       - max_features=100: prevents overfitting
       - Range matched to HDP-HMM for fair comparison
    """
    
    # OPTIMIZED PARAMETERS
    # I. MODEL STRUCTURE
    K_min: int = 4
    K_max: int = 7
    
    n_mixtures_min: int = 1
    n_mixtures_max: int = 3
    
    pca_components_min: int = 10
    pca_components_max: int = 14
    pca_components_valid: List[int] = field(default_factory=lambda: [10, 12, 14])  # Only these values are cached
    
    # II. TRANSITION PARAMETERS
    base_alpha_min: float = 0.1
    base_alpha_max: float = 1.0
    
    kappa_min: float = 5.0
    kappa_max: float = 50.0
    
    # III. TRAINING PARAMETERS
    lr_min: float = 1e-4
    lr_max: float = 1e-1
    
    # FIXED PARAMETERS (not optimized, using sensible defaults)
    num_iters_fixed: int = 1000
    num_particles_fixed: int = 10
    prior_mean_scale_fixed: float = 10.0
    prior_cov_scale_fixed: float = 1.0
    patience_fixed: int = 50
    elbo_improvement_threshold_fixed: float = 1e-3
    min_features_fixed: int = 50
    max_features_fixed: int = 100
    
    def to_search_space(self) -> Dict[str, Dict[str, Any]]:
        """Convert to focused optimization search space (6 parameters)."""
        return {
            # I. MODEL STRUCTURE (3 params)
            'K': {
                'type': 'categorical',
                'choices': [4, 5, 6, 7]  # Discrete values only (4 choices)
            },
            'n_mixtures': {
                'type': 'int',
                'low': self.n_mixtures_min,
                'high': self.n_mixtures_max
            },
            'pca_components': {
                'type': 'int',
                'low': self.pca_components_min,
                'high': self.pca_components_max
            },
            
            # II. TRANSITION PARAMETERS (2 params)
            'base_alpha': {
                'type': 'float',
                'low': self.base_alpha_min,
                'high': self.base_alpha_max,
                'log': False
            },
            'kappa': {
                'type': 'float',
                'low': self.kappa_min,
                'high': self.kappa_max,
                'log': False
            },
            
            # III. TRAINING PARAMETERS (1 param)
            'lr': {
                'type': 'float',
                'low': self.lr_min,
                'high': self.lr_max,
                'log': True  # Log scale for learning rate
            }
        }
    
    def to_hierarchical_param_groups(self) -> List[ParameterGroup]:
        """
        Convert search space to hierarchical parameter groups for efficient optimization.
        
        3 groups for 6 focused parameters:
        - Group 1: Model structure (K, n_mixtures, pca_components)
        - Group 2: Transition dynamics (base_alpha, kappa) 
        - Group 3: Training (lr)
        
        Returns:
            List of ParameterGroup objects organized by priority and dependencies
        """
        groups = [
            # Group 1 (Priority 1): Model Structure
            # These fundamentally determine the model capacity
            create_param_group(
                name="structure",
                params={
                    'K': {
                        'type': 'int',
                        'low': self.K_min,
                        'high': self.K_max
                    },
                    'n_mixtures': {
                        'type': 'int',
                        'low': self.n_mixtures_min,
                        'high': self.n_mixtures_max
                    },
                    'pca_components': {
                        'type': 'categorical',
                        'choices': self.pca_components_valid  # Only valid PCA components
                    }
                },
                priority=1,
                description="Model structure: states, mixtures, and dimensionality"
            ),
            
            # Group 2 (Priority 2): Transition Parameters
            # Depends on K being set
            create_param_group(
                name="transitions",
                params={
                    'base_alpha': {
                        'type': 'float',
                        'low': self.base_alpha_min,
                        'high': self.base_alpha_max,
                        'log': False
                    },
                    'kappa': {
                        'type': 'float',
                        'low': self.kappa_min,
                        'high': self.kappa_max,
                        'log': False
                    }
                },
                priority=2,
                depends_on=["structure"],
                description="Transition matrix priors and stickiness"
            ),
            
            # Group 3 (Priority 3): Training Parameters
            # Learning rate optimization depends on structure and transitions
            create_param_group(
                name="training",
                params={
                    'lr': {
                        'type': 'float',
                        'low': self.lr_min,
                        'high': self.lr_max,
                        'log': True
                    }
                },
                priority=3,
                depends_on=["structure", "transitions"],
                description="SVI learning rate"
            )
        ]
        
        return groups


def create_default_search_space() -> StickyFiniteHMMSearchSpace:
    """
    Create default search space for Sticky Finite HMM optimization.
    
    Focuses on 5 key parameters, others are fixed at sensible defaults.
    
    Returns:
        StickyFiniteHMMSearchSpace with sensible defaults
    """
    return StickyFiniteHMMSearchSpace(
        # OPTIMIZED: Model structure
        K_min=4,
        K_max=7,
        pca_components_min=10,
        pca_components_max=14,
        
        # OPTIMIZED: Transitions
        base_alpha_min=0.1,
        base_alpha_max=1.0,
        kappa_min=5.0,
        kappa_max=50.0,
        
        # OPTIMIZED: Training (learning rate only)
        lr_min=1e-4,
        lr_max=1e-1,
        
        # FIXED: Other parameters at sensible defaults
        num_iters_fixed=1000,
        num_particles_fixed=10,
        prior_mean_scale_fixed=10.0,
        prior_cov_scale_fixed=1.0,
        patience_fixed=50,
        elbo_improvement_threshold_fixed=1e-3,
    )


# Global trial results cache for CSV export
_trial_results_cache: list = []
# Global trial counter for progress tracking
_trial_counter: dict = {'count': 0}

def _create_basic_features_for_tuning(market_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic features from market data for quality assessment during tuning.
    
    Args:
        market_data: OHLCV market data
        
    Returns:
        DataFrame with basic features
    """
    features = pd.DataFrame({
        'returns': market_data['close'].pct_change(),
        'volume': market_data['volume'],
        'high_low_ratio': market_data['high'] / market_data['low'],
        'open_close_ratio': market_data['open'] / market_data['close'],
        'price_change': market_data['close'] - market_data['open'],
        'volatility': market_data['high'] - market_data['low'],
        'volume_price_trend': market_data['volume'] * market_data['close'].pct_change(),
        'price_momentum': market_data['close'].pct_change(5),
        'volume_sma': market_data['volume'].rolling(20).mean(),
        'price_position': (market_data['close'] - market_data['low']) / (market_data['high'] - market_data['low'])
    }).fillna(0)
    
    return features

def sticky_finite_hmm_objective_function(
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    model: Optional[Any] = None,
    cv_folds: int = 5,
    scoring_metric: str = 'composite_score',
    market_data: Optional[pd.DataFrame] = None,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "1h",
    optimization_goals = None,
    optimization_targets = None,
    logger: Optional[logging.Logger] = None,
    return_multi_objective: bool = False
) -> float:
    """
    Enhanced objective function for Sticky Finite HMM optimization with variance reduction.
    
    Now leverages:
    - Structured variational inference with forward-backward
    - Natural gradient updates for reduced variance
    - Rao-Blackwellization for parameter marginalization
    - Vectorized computations for speed
    
    Args:
        params: Parameters to evaluate
        X_train, y_train: Training data (not used directly, market_data is used)
        X_val, y_val: Validation data (not used directly)
        model: Model instance (not used directly)
        cv_folds: CV folds (not used directly)
        scoring_metric: Metric to optimize (should be 'composite_score')
        market_data: Market data DataFrame
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        optimization_goals: Clustering optimization goals
        optimization_targets: Optimization targets
        logger: Logger instance
        return_multi_objective: Return multiple objectives for Pareto optimization
    
    Returns:
        Composite score (higher is better) or multi-objective scores
    """
    if logger:
        logger.debug(f"Evaluating enhanced params with variance reduction: {params}")
    
    try:
        # Extract OPTIMIZED parameters (6 key params)
        K = int(params.get('K', 5))
        n_mixtures = int(params.get('n_mixtures', 1))
        base_alpha = float(params.get('base_alpha', 0.5))
        kappa = float(params.get('kappa', 10.0))
        lr = float(params.get('lr', 1e-2))
        pca_components = int(params.get('pca_components', 12))
        
        # OPTIMIZED: Reduced iterations for faster training with early stopping
        # 150 iterations sufficient with early stopping and natural gradients
        num_iters = 100  # Reduced from 150 for faster training
        min_features = 50  # Adequate signal
        max_features = 100  # Prevent overfitting
        
        # Ensure min_features <= max_features
        if min_features > max_features:
            min_features, max_features = max_features, min_features
        
        # ENHANCED: Run with structured variational inference and natural gradients
        tprint_info(f"🧪 Testing Parameter Set:")
        tprint_info(f"   📊 Model Structure: K={K} regimes, n_mixtures={n_mixtures}, pca_components={pca_components}")
        tprint_info(f"   ⚙️ HMM Parameters: κ={kappa:.1f} (stickiness), α={base_alpha:.2f} (transition prior)")
        tprint_info(f"   🚀 Optimization: lr={lr:.1e}, num_iters={num_iters}")
        tprint_info(f"   🎯 Feature Range: {min_features}-{max_features} features")
        
        # Ensure market_data is available
        if market_data is None:
            tprint_error("❌ Market data is required for enhanced objective function")
            return 0.0
        
        result = run_sticky_finite_hmm_clustering(
            market_data=market_data,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            min_features=min_features,
            max_features=max_features,
            K=K,
            n_mixtures=n_mixtures,
            base_alpha=base_alpha,
            kappa=kappa,
            num_iters=num_iters,
            lr=lr,
            pca_components=pca_components
        )
        
        # Extract comprehensive metrics
        quality_metrics = result.get('quality_metrics', {})
        
        # Run comprehensive quality assessment using ClusterQualityAssessor if available
        if _quality_assessor_available and 'cluster_labels' in result:
            try:
                # Initialize quality assessor
                quality_assessor = ClusterQualityAssessor(
                    artifact_manager=None,
                    enable_hardware_optimization=True,
                    enable_vectorization=True
                )
                
                # Extract data for quality assessment
                cluster_labels = np.array(result['cluster_labels'])
                feature_matrix = result.get('feature_matrix')
                
                if feature_matrix is None:
                    # Create basic features from market data
                    feature_matrix = _create_basic_features_for_tuning(market_data)
                
                # Ensure data alignment
                min_length = min(len(cluster_labels), len(feature_matrix))
                cluster_labels = cluster_labels[:min_length]
                feature_matrix = feature_matrix.iloc[:min_length].reset_index(drop=True)
                timestamps = market_data.index[:min_length]
                
                # Calculate forward returns for economic validation
                forward_returns = market_data['close'].pct_change().shift(-1).iloc[:min_length]
                
                # Run comprehensive quality assessment
                comprehensive_quality = quality_assessor.assess_quality(
                    regime_labels=cluster_labels,
                    feature_data=feature_matrix,
                    forward_returns=forward_returns,
                    timestamps=timestamps,
                    min_regime_size=5,  # Lower threshold for tuning
                    temporal_sensitivity_mode="standard"
                )
                
                # Use comprehensive quality score
                composite_score = comprehensive_quality.quality_score or 0.0
                
                # Extract additional metrics from comprehensive assessment
                balance_score = comprehensive_quality.balance_score or 0.0
                temporal_smoothness = comprehensive_quality.temporal_smoothness or 0.0
                between_regime_cv = comprehensive_quality.between_regime_cv or 0.0
                within_regime_cv = comprehensive_quality.within_regime_cv or 1e-10
                cv_ratio = between_regime_cv / max(within_regime_cv, 1e-10)
                transition_persistence = comprehensive_quality.regime_persistence or 0.0
                
                # Extract economic metrics from comprehensive assessment
                per_regime_metrics = comprehensive_quality.per_regime_metrics or {}
                regime_sharpes = [v.get('sharpe', 0) for v in per_regime_metrics.values() if isinstance(v, dict)]
                regime_returns = [v.get('mean_return', 0) for v in per_regime_metrics.values() if isinstance(v, dict)]
                avg_sharpe = np.mean(regime_sharpes) if regime_sharpes else 0.0
                avg_return = np.mean(regime_returns) if regime_returns else 0.0
                
                tprint_success(f"✅ Comprehensive quality assessment: {composite_score:.4f}")
                
            except Exception as e:
                tprint_warning(f"⚠️ Comprehensive quality assessment failed: {e}")
                # Fallback to basic metrics
                composite_score = quality_metrics.get('composite_score', 0.0)
                balance_score = 0.0
                temporal_smoothness = 0.0
                between_regime_cv = 0.0
                within_regime_cv = 1e-10
                cv_ratio = 0.0
                transition_persistence = quality_metrics.get('transition_persistence', 0.0)
                avg_sharpe = 0.0
                avg_return = 0.0
        else:
            # Fallback to basic quality metrics
            quality_assessment = quality_metrics.get('quality_assessment', {})
            
            # Convert ClusterQualityMetrics to dict if needed
            if hasattr(quality_assessment, '__dict__'):
                qa_dict = {}
                for key, value in quality_assessment.__dict__.items():
                    if not key.startswith('_'):  # Skip private attributes
                        qa_dict[key] = value
            elif isinstance(quality_assessment, dict):
                qa_dict = quality_assessment
            else:
                qa_dict = {}
            
            # Core metrics
            composite_score = quality_metrics.get('composite_score', 0.0)
            
            # Balance metrics
            balance_score = qa_dict.get('balance_score', 0.0)
            min_cluster_size_pct = qa_dict.get('min_cluster_size_pct', 0.0)
            max_cluster_size_pct = qa_dict.get('max_cluster_size_pct', 0.0)
            
            # CV metrics
            between_regime_cv = qa_dict.get('between_regime_cv', 0.0)
            within_regime_cv = qa_dict.get('within_regime_cv', 1e-10)
            cv_ratio = between_regime_cv / max(within_regime_cv, 1e-10)
            
            # Temporal smoothness
            temporal_smoothness = qa_dict.get('temporal_smoothness', 0.0)
            
            # Regime persistence
            transition_persistence = quality_metrics.get('transition_persistence', 0.0)
            
            # Extract economic metrics
            per_regime_metrics = qa_dict.get('per_regime_metrics', {})
            regime_sharpes = [v.get('sharpe', 0) for v in per_regime_metrics.values() if isinstance(v, dict)]
            regime_returns = [v.get('mean_return', 0) for v in per_regime_metrics.values() if isinstance(v, dict)]
            avg_sharpe = np.mean(regime_sharpes) if regime_sharpes else 0.0
            avg_return = np.mean(regime_returns) if regime_returns else 0.0
        
        # Enhanced trial logging with variance reduction indicators
        # Get trial number from global counter if available
        global _trial_counter
        current_trial = _trial_counter.get('count', 0) + 1
        _trial_counter['count'] = current_trial
        
        # Estimate total trials based on optimization method
        if use_hierarchical:
            estimated_total = n_rounds * 50  # Approximate based on hierarchical config
        else:
            estimated_total = tpe_trials + coarse_grid_points**6 + fine_grid_points**6
        
        # OPTIMIZED: Adaptive iterations for faster training
        total_trials = estimated_total
        if current_trial <= total_trials * 0.3:
            # Early exploration: use fewer iterations for speed
            adaptive_num_iters = 50  # Reduced from 200
            iteration_mode = "Exploration"
            adaptive_n_mixtures = 1  # Single Gaussian for speed
        elif current_trial <= total_trials * 0.7:
            # Middle phase: moderate iterations
            adaptive_num_iters = 100  # Reduced from 300
            iteration_mode = "Development"
            adaptive_n_mixtures = 1  # Still single Gaussian
        else:
            # Refinement phase: more iterations for quality
            adaptive_num_iters = 150  # Reduced from 400
            iteration_mode = "Refinement"
            adaptive_n_mixtures = 2  # Allow mixtures for top 30% (more aggressive than 20%)
        
        # Override parameters in params for this trial
        params['num_iters'] = adaptive_num_iters
        params['n_mixtures'] = adaptive_n_mixtures
        
        tprint_success(
            f"✅ TRIAL {current_trial}/{estimated_total} ({iteration_mode}): Score={composite_score:.4f} | "
            f"K={K} κ={kappa:.1f} α={base_alpha:.2f} pca={pca_components} lr={lr:.1e} iters={adaptive_num_iters}"
        )
        tprint_info(
            f"   📊 Quality Metrics: "
            f"Balance={balance_score:.3f} | "
            f"CV Ratio={cv_ratio:.3f} | "
            f"Temporal Smoothness={temporal_smoothness:.3f} | "
            f"Transition Persistence={transition_persistence:.3f}"
        )
        tprint_info(
            f"   💰 Economic Metrics: "
            f"Avg Sharpe={avg_sharpe:.3f} | "
            f"Avg Return={avg_return:+.4f} | "
            f"Cluster Sizes: {min_cluster_size_pct:.0f}%-{max_cluster_size_pct:.0f}%"
        )
        tprint_info(
            f"   🧠 Methods: Structured Variational Inference + Natural Gradients + Rao-Blackwellization"
        )
        
        # Store trial results in cache for comprehensive CSV export
        global _trial_results_cache
        _trial_results_cache.append({
            'params': params,
            'score': composite_score,
            'metrics': quality_metrics,
            'quality_assessment': quality_assessment,
            'enhanced_methods': True  # Flag for variance reduction methods
        })
        
        # Multi-objective return for Pareto optimization
        if return_multi_objective:
            multi_objective_result = {
                'composite_score': composite_score,
                'silhouette_score': qa_dict.get('silhouette_score', 0.0),
                'temporal_smoothness': temporal_smoothness,
                'balance_score': balance_score,
                'economic_sharpe': avg_sharpe
            }
            # Return composite_score for single-objective optimization compatibility
            return float(composite_score)
        
        return float(composite_score)
        
    except Exception as e:
        tprint_error(
            f"❌ ENHANCED TRIAL FAILED: K={K} κ={kappa:.1f} α={base_alpha:.2f} pca={pca_components} lr={lr:.1e} | "
            f"Error: {str(e)[:80]}"
        )
        
        if logger:
            logger.warning(f"Enhanced objective evaluation failed: {e}")
        return 0.0  # Return poor score on failure


def run_sticky_finite_hmm_auto_tuning(
    market_data: pd.DataFrame,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "1h",
    search_space: Optional[StickyFiniteHMMSearchSpace] = None,
    use_hierarchical: bool = True,
    use_multi_objective: bool = False,
    n_rounds: int = 2,
    coarse_grid_points: int = 5,
    fine_grid_points: int = 5,
    tpe_trials: int = 100,
    timeout: Optional[int] = None,
    cv_folds: int = 3,
    optimization_goals = None,
    optimization_targets = None,
    random_state: int = 42,
    cache_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    """
    Run automatic hyperparameter tuning for Sticky Finite HMM clustering.
    
    Args:
        market_data: Market data DataFrame
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        search_space: Custom search space (uses default if None)
        use_hierarchical: Whether to use hierarchical optimization (recommended)
        use_multi_objective: Whether to use multi-objective Pareto optimization
        n_rounds: Number of optimization rounds (for hierarchical)
        coarse_grid_points: Grid points for coarse search
        fine_grid_points: Grid points for fine search
        tpe_trials: Number of TPE trials
        timeout: Timeout in seconds (optional)
        cv_folds: Cross-validation folds (not used directly)
        optimization_goals: Clustering optimization goals
        optimization_targets: Optimization targets
        random_state: Random seed
        cache_dir: Directory to cache results
        verbose: Whether to print progress
    
    Returns:
        Tuple of (best_params, best_score, tuning_results)
    """
    global _trial_results_cache, _trial_counter
    
    # Clear trial results cache for fresh run
    _trial_results_cache = []
    _trial_counter = {'count': 0}
    
    tprint("=" * 80, "INFO")
    tprint("🎯 Enhanced Sticky Finite HMM Auto-Tuning with Variance Reduction", "INFO")
    tprint("=" * 80, "INFO")
    tprint("🧠 Structured Variational Inference: Forward-backward message passing", "INFO")
    tprint("🔄 Natural Gradient Updates: Closed-form parameter updates", "INFO")
    tprint("📊 Rao-Blackwellization: Zero MC variance for sufficient statistics", "INFO")
    tprint("⚡ Vectorized Computations: Optimized NumPy operations", "INFO")
    tprint("=" * 80, "INFO")
    
    start_time = time.time()
    
    # Create search space
    if search_space is None:
        search_space = create_default_search_space()
        tprint("✅ Using default search space", "SUCCESS")
    
    # Set optimization goals
    if optimization_goals is None:
        optimization_goals = DEFAULT_CLUSTERING_GOALS
    if optimization_targets is None:
        optimization_targets = DEFAULT_OPTIMIZATION_TARGETS
    
    # Create logger
    logger = logging.getLogger('StickyFiniteHMM_AutoTuner')
    
    # Multi-objective optimization
    if use_multi_objective and _pareto_available:
        tprint("🎯 Multi-Objective Optimization Mode Enabled", "INFO")
        tprint("   - Objectives: composite_score, silhouette, temporal_smoothness, balance, economic_sharpe", "INFO")
        tprint("   - Will return Pareto front of non-dominated solutions", "INFO")
        tprint("", "INFO")
    
    # Choose optimization method
    if use_hierarchical and _hierarchical_hpo_available:
        tprint("🔧 Using Hierarchical Parameter Optimization (Focused)", "INFO")
        tprint(f"   - Optimizing: 6 key parameters (K, n_mixtures, kappa, base_alpha, lr, pca_components)", "INFO")
        tprint(f"   - Fixed: 7 parameters at sensible defaults", "INFO")
        tprint(f"   - Optimization rounds: {n_rounds}", "INFO")
        tprint(f"   - Parameter groups: 3 (structure, transitions, training)", "INFO")
        tprint(f"   - Stages per group: Coarse Grid → Fine Grid → TPE", "INFO")
        tprint(f"   - Expected trials: ~50-150 (much faster than full search)", "INFO")
        if use_multi_objective:
            tprint(f"   - Will build Pareto front from all trials", "INFO")
        
        # Create parameter groups
        param_groups = search_space.to_hierarchical_param_groups()
        
        # Create objective function with market data bound
        def objective_func(params, X_train, y_train, X_val, y_val, model, cv_folds, scoring_metric):
            return sticky_finite_hmm_objective_function(
                params=params,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                model=model,
                cv_folds=cv_folds,
                scoring_metric=scoring_metric,
                market_data=market_data,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                optimization_goals=optimization_goals,
                optimization_targets=optimization_targets,
                logger=logger
            )
        
        # Create hierarchical optimizer with coarse -> fine -> finer grid stages
        # Ensure at least 5 config values tested per parameter per grid stage
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=objective_func,
            stages=[
                OptimizationStage.COARSE_GRID,
                OptimizationStage.FINE_GRID,
                OptimizationStage.FINE_GRID,  # Additional finer grid stage
                OptimizationStage.TPE
            ],
            stage_configs={
                OptimizationStage.COARSE_GRID: StageConfig(
                    stage=OptimizationStage.COARSE_GRID,
                    n_trials=50,
                    grid_points=5,  # At least 5 config values per parameter
                    enable_pruning=False
                ),
                OptimizationStage.FINE_GRID: StageConfig(
                    stage=OptimizationStage.FINE_GRID,
                    n_trials=50,
                    grid_points=5,  # At least 5 config values per parameter
                    enable_pruning=False
                ),
                OptimizationStage.TPE: StageConfig(
                    stage=OptimizationStage.TPE,
                    n_trials=tpe_trials,
                    enable_pruning=True
                )
            },
            cv_folds=cv_folds,
            scoring_metric='composite_score',
            direction='maximize',
            n_rounds=n_rounds,
            enable_final_refinement=True,
            final_refinement_trials=tpe_trials,
            cache_dir=cache_dir,
            random_state=random_state,
            verbose=verbose
        )
        
        # Run optimization (dummy data since we use market_data in objective)
        X_dummy = np.random.randn(100, 10)
        y_dummy = np.random.randn(100)
        
        result = optimizer.optimize(
            X_train=X_dummy,
            y_train=y_dummy,
            model=None
        )
        
        best_params = result.best_params
        best_score = result.best_score
        tuning_results = {
            'method': 'hierarchical',
            'total_time': result.total_time,
            'total_trials': result.total_trials,
            'n_rounds': n_rounds,
            'group_results': [
                {
                    'group_name': gr.group_name,
                    'best_score': gr.best_score,
                    'n_trials': gr.n_trials,
                    'optimization_time': gr.optimization_time
                }
                for gr in result.group_results
            ],
            'final_refinement': {
                'best_score': result.final_refinement_result.best_score,
                'n_trials': result.final_refinement_result.n_trials
            } if result.final_refinement_result else None
        }
        
        # Multi-objective Pareto front construction
        if use_multi_objective and _pareto_available:
            tprint("", "INFO")
            tprint("🎯 Constructing Pareto Front from All Trials", "INFO")
            tprint("=" * 80, "INFO")
            
            # Collect all trials with multi-objective scores
            all_trials_mo = []
            for gr in result.group_results:
                for trial in gr.all_trials:
                    trial_params = trial.get('params', {})
                    # Re-evaluate with multi-objective to get all scores
                    mo_scores = sticky_finite_hmm_objective_function(
                        params={**result.best_params, **trial_params},  # Merge with best params
                        X_train=X_dummy,
                        y_train=y_dummy,
                        market_data=market_data,
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        optimization_goals=optimization_goals,
                        optimization_targets=optimization_targets,
                        logger=logger,
                        return_multi_objective=True
                    )
                    
                    all_trials_mo.append({
                        'params': {**result.best_params, **trial_params},
                        'objectives': mo_scores
                    })
            
            # Build Pareto front using proper Pareto utils
            # Create Solution objects for each trial
            solutions = []
            for trial in all_trials_mo:
                metrics = {
                    'composite_score': trial['objectives']['composite_score'],
                    'silhouette_score': trial['objectives']['silhouette_score'],
                    'temporal_smoothness': trial['objectives']['temporal_smoothness'],
                    'balance_score': trial['objectives']['balance_score'],
                    'economic_sharpe': trial['objectives']['economic_sharpe']
                }
                solution = Solution(metrics=metrics, params=trial['params'])
                solutions.append(solution)
            
            # Define objectives (all to maximize)
            objectives = ObjectiveDirection({
                'composite_score': 'maximize',
                'silhouette_score': 'maximize',
                'temporal_smoothness': 'maximize',
                'balance_score': 'maximize',
                'economic_sharpe': 'maximize'
            })
            
            # Compute Pareto front using the proper function
            pareto_solutions = compute_pareto_front(
                solutions=solutions,
                objectives=objectives,
                use_gpu=True,
                use_vectorbt=True
            )
            
            # Select knee point as the recommended solution
            knee_solution = select_knee_point(
                pareto_solutions=pareto_solutions,
                objectives=objectives,
                weights={'composite_score': 0.4, 'silhouette_score': 0.2, 'temporal_smoothness': 0.2, 'balance_score': 0.1, 'economic_sharpe': 0.1}
            )
            
            tprint(f"✅ Pareto Front: {len(pareto_solutions)} non-dominated solutions", "SUCCESS")
            tprint(f"   Total trials evaluated: {len(all_trials_mo)}", "INFO")
            tprint(f"   Dominated solutions removed: {len(all_trials_mo) - len(pareto_solutions)}", "INFO")
            
            # Add Pareto front to results
            tuning_results['pareto_front'] = {
                'n_solutions': len(pareto_solutions),
                'solutions': [
                    {
                        'params': sol.params,
                        'objectives': {
                            'composite_score': sol.metrics['composite_score'],
                            'silhouette_score': sol.metrics['silhouette_score'],
                            'temporal_smoothness': sol.metrics['temporal_smoothness'],
                            'balance_score': sol.metrics['balance_score'],
                            'economic_sharpe': sol.metrics['economic_sharpe']
                        }
                    }
                    for sol in pareto_solutions[:10]  # Top 10 Pareto solutions
                ],
                'knee_point': {
                    'params': knee_solution.params if knee_solution else None,
                    'objectives': knee_solution.metrics if knee_solution else None
                }
            }
            
            tprint("", "INFO")
            tprint("Top 3 Pareto Solutions:", "INFO")
            for i, sol in enumerate(pareto_solutions[:3]):
                tprint(f"  {i+1}. Composite={sol.metrics['composite_score']:.4f}, "
                      f"Silhouette={sol.metrics['silhouette_score']:.4f}, "
                      f"Temporal={sol.metrics['temporal_smoothness']:.4f}", "INFO")
            
            if knee_solution:
                tprint("", "INFO")
                tprint("🎯 Recommended Knee Point Solution:", "INFO")
                tprint(f"   Composite={knee_solution.metrics['composite_score']:.4f}, "
                      f"Silhouette={knee_solution.metrics['silhouette_score']:.4f}, "
                      f"Temporal={knee_solution.metrics['temporal_smoothness']:.4f}", "INFO")
        
    else:
        if use_hierarchical and not _hierarchical_hpo_available:
            tprint_warning("⚠️ Hierarchical optimization not available, falling back to standard method")
        
        tprint("🔧 Using Standard Multi-Stage Optimization", "INFO")
        tprint(f"   - Stage 1: Coarse Grid ({coarse_grid_points} points per param)", "INFO")
        tprint(f"   - Stage 2: Fine Grid ({fine_grid_points} points per param)", "INFO")
        tprint(f"   - Stage 3: TPE ({tpe_trials} trials)", "INFO")
        
        # Fallback to standard optimization
        # This would be implemented similarly to HDP-HMM's standard method
        # For now, return default params
        tprint_warning("⚠️ Standard optimization not yet fully implemented")
        best_params = {
            'K': 5,
            'base_alpha': 0.5,
            'kappa': 10.0,
            'num_iters': 800,
            'lr': 1e-2,
            'num_particles': 10,
            'prior_mean_scale': 10.0,
            'prior_cov_scale': 1.0,
            'patience': 50,
            'elbo_improvement_threshold': 1e-3,
            'pca_components': 12,
            'min_features': 50,
            'max_features': 100
        }
        best_score = 0.0
        tuning_results = {'method': 'none', 'note': 'Using default parameters'}
    
    total_time = time.time() - start_time
    
    # Print summary
    tprint("=" * 80, "INFO")
    tprint("✅ Sticky Finite HMM Auto-Tuning Complete!", "SUCCESS")
    tprint("=" * 80, "INFO")
    tprint(f"Best Score: {best_score:.6f}", "SUCCESS")
    tprint(f"Total Time: {total_time:.2f}s", "INFO")
    if 'total_trials' in tuning_results:
        tprint(f"Total Trials: {tuning_results['total_trials']}", "INFO")
    tprint("", "INFO")
    tprint("Best Parameters:", "INFO")
    for param_name, param_value in best_params.items():
        if isinstance(param_value, float):
            tprint(f"  - {param_name}: {param_value:.6f}", "INFO")
        else:
            tprint(f"  - {param_name}: {param_value}", "INFO")
    tprint("=" * 80, "INFO")
    
    # Export comprehensive CSV with all trial results (20+ metrics per trial)
    try:
        from pathlib import Path
        import datetime
        import pandas as pd
        
        outcomes_dir = Path("outcomes") / "sticky_finite_hmm_auto_tuning" / symbol / exchange / timeframe
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = outcomes_dir / f"auto_tuning_all_trials_{timestamp}.csv"
        
        # Export trial results to CSV if any trials were run
        if _trial_results_cache:
            tprint_info("📊 Exporting trial results to CSV...")
            
            # Create DataFrame from trial cache
            trial_data = []
            for i, trial in enumerate(_trial_results_cache):
                params = trial.get('params', {})
                metrics = trial.get('metrics', {})
                qa = trial.get('quality_assessment', {})
                
                # Convert ClusterQualityMetrics to dict if needed
                if hasattr(qa, '__dict__'):
                    qa_dict = {}
                    for key, value in qa.__dict__.items():
                        if not key.startswith('_'):  # Skip private attributes
                            qa_dict[key] = value
                elif isinstance(qa, dict):
                    qa_dict = qa
                else:
                    qa_dict = {}
                
                per_regime = qa_dict.get('per_regime_metrics', {})
                
                regime_sharpes = [v.get('sharpe', 0) for v in per_regime.values() if isinstance(v, dict)]
                regime_returns = [v.get('mean_return', 0) for v in per_regime.values() if isinstance(v, dict)]
                regime_sizes = [v.get('size', 0) for v in per_regime.values() if isinstance(v, dict)]
                
                trial_dict = {
                    # Trial info
                    'trial_number': i + 1,
                    'composite_score': trial.get('score', 0.0),
                    
                    # Parameters
                    'K': params.get('K', 0),
                    'kappa': params.get('kappa', 0.0),
                    'base_alpha': params.get('base_alpha', 0.0),
                    'pca_components': params.get('pca_components', 0),
                    'lr': params.get('lr', 0.0),
                    'n_mixtures': params.get('n_mixtures', 1),
                    
                    # Quality metrics
                    'silhouette_score': qa_dict.get('silhouette_score', 0.0),
                    'davies_bouldin_score': qa_dict.get('davies_bouldin_score', 0.0),
                    'calinski_harabasz_score': qa_dict.get('calinski_harabasz_score', 0.0),
                    'balance_score': qa_dict.get('balance_score', 0.0),
                    'temporal_smoothness': qa_dict.get('temporal_smoothness', 0.0),
                    
                    # CV metrics
                    'between_regime_cv': qa_dict.get('between_regime_cv', 0.0),
                    'within_regime_cv': qa_dict.get('within_regime_cv', 0.0),
                    'cv_ratio': qa_dict.get('between_regime_cv', 0.0) / max(qa_dict.get('within_regime_cv', 1e-10), 1e-10),
                    
                    # Cluster distribution
                    'min_cluster_size_pct': qa_dict.get('min_cluster_size_pct', 0.0),
                    'max_cluster_size_pct': qa_dict.get('max_cluster_size_pct', 0.0),
                    
                    # Transition metrics
                    'transition_persistence': metrics.get('transition_persistence', 0.0),
                    'flip_flop_ratio': qa_dict.get('flip_flop_ratio', 0.0),
                    
                    # Economic metrics (aggregated)
                    'avg_sharpe_ratio': np.mean(regime_sharpes) if regime_sharpes else 0.0,
                    'avg_mean_return': np.mean(regime_returns) if regime_returns else 0.0,
                    'max_sharpe': max(regime_sharpes) if regime_sharpes else 0.0,
                    'min_sharpe': min(regime_sharpes) if regime_sharpes else 0.0,
                    
                    # Regime sizes
                    'n_regimes_active': len([s for s in regime_sizes if s > 0]),
                    'regime_size_std': np.std(regime_sizes) if regime_sizes else 0.0,
                }
                trial_data.append(trial_dict)
            
            df = pd.DataFrame(trial_data)
            df = df.sort_values('composite_score', ascending=False)
            df.to_csv(csv_path, index=False)
            tprint_success(f"📊 Auto-tuning trials exported: {csv_path}")
            tprint_info(f"   Total trials saved: {len(df)} with {len(df.columns)} metrics per trial")
            
            # Clear cache after export
            _trial_results_cache = []
        else:
            tprint_warning("⚠️ No trial data available for CSV export")
            
    except Exception as e:
        tprint_warning(f"⚠️ Could not export auto-tuning CSV: {e}")
    
    return best_params, best_score, tuning_results


__all__ = [
    'StickyFiniteHMMSearchSpace',
    'create_default_search_space',
    'sticky_finite_hmm_objective_function',
    'run_sticky_finite_hmm_auto_tuning'
]

