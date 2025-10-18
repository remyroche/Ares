"""
CMI Complementarity Scorer

This module implements conditional mutual information complementarity scoring
for feature selection. Computes R(X|A), D(X,S|A), and greedy selection with
noise floor detection and regime-aware aggregation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import time
import logging
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
import warnings

# Import CMI estimators and Analyst side info
from .cmi_estimators import CMIEstimator, CMIEstimatorConfig
from .analyst_side_info import AnalystSideInfoHandler, AnalystSideInfoConfig

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import hardware optimizations
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUOptimizer
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    HARDWARE_OPTIMIZATIONS_AVAILABLE = True
    tprint_info("✅ Hardware optimizations available for CMI complementarity")
except ImportError:
    HARDWARE_OPTIMIZATIONS_AVAILABLE = False
    tprint_warning("⚠️ Hardware optimizations not available, using standard computations")

# Import VectorBT optimizations
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    from src.feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager
    VECTORBT_OPTIMIZATIONS_AVAILABLE = True
    tprint_info("✅ VectorBT optimizations available for CMI complementarity")
except ImportError:
    VECTORBT_OPTIMIZATIONS_AVAILABLE = False
    tprint_warning("⚠️ VectorBT optimizations not available, using standard computations")

# Import ML utilities
try:
    from src.utils.purged_kfold import PurgedKFoldTime
    from src.utils.ml_common.validation.data_leakage_detector import DataLeakageDetector
    from src.utils.ml_common.utils.lookahead_protection import LookaheadValidator
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    ML_UTILITIES_AVAILABLE = True
    tprint_info("✅ ML utilities available for CMI complementarity")
except ImportError:
    ML_UTILITIES_AVAILABLE = False
    tprint_warning("⚠️ ML utilities not available, using standard implementations")

# Import common utilities
try:
    from src.utils.common_operations import safe_divide, safe_log
    from src.utils.common_utilities import validate_inputs, handle_missing_data
    from src.utils.math_validation import validate_numerical, check_finite
    COMMON_UTILITIES_AVAILABLE = True
    tprint_info("✅ Common utilities available for CMI complementarity")
except ImportError:
    COMMON_UTILITIES_AVAILABLE = False
    tprint_warning("⚠️ Common utilities not available, using standard implementations")

logger = logging.getLogger(__name__)

@dataclass
class CMIComplementarityResult:
    """Result from CMI complementarity scoring."""
    selected_features: List[str]  # Selected feature names
    relevance_scores: Dict[str, float]  # R(X|A) scores
    redundancy_scores: Dict[str, float]  # D(X,S|A) scores
    greedy_scores: Dict[str, float]  # Final greedy scores
    noise_floor: float  # Computed noise floor
    delta_perf_threshold: float  # Data-driven ΔPerf threshold
    selection_metadata: Dict[str, Any]
    is_valid: bool = True

@dataclass
class CMIComplementarityConfig:
    """Configuration for CMI complementarity scoring."""
    # Greedy selection parameters
    alpha_candidates: List[float] = None  # Redundancy penalty weights (CV-tuned)
    beta_synergy: float = 0.25  # Synergy bonus weight (optional)
    enable_synergy: bool = False  # Enable synergy computation
    
    # Noise floor computation
    noise_floor_permutations: int = 150  # Label shuffles for noise floor
    noise_floor_percentile: int = 95  # Threshold percentile
    delta_perf_permutations: int = 25  # Null permutations for ΔPerf threshold
    
    # CV parameters
    cv_folds: int = 5  # Purged K-fold splits
    embargo_windows: int = 1  # Time embargo for CV (1-3 windows)
    
    # Selection parameters
    per_family_budget: Tuple[int, int] = (5, 15)  # Min/max features per family
    upstream_multiplier: int = 3  # Total budget to RFE = 3× per-family
    max_total_features: int = 60  # Maximum total features to select
    
    # Stopping criteria
    min_relevance_threshold: float = 0.001  # Minimum R(X|A) to consider
    delta_perf_plateau_threshold: float = 0.002  # ΔPerf plateau threshold
    max_steps_without_improvement: int = 3  # Stop after N steps without improvement
    
    # Regime awareness
    enable_regime_awareness: bool = True  # Compute R(X|A) per regime
    regime_occupancy_threshold: float = 0.05  # Minimum regime occupancy
    
    # Performance limits
    compute_timeout_seconds: float = 300.0  # 5 min hard limit
    enable_timeout_fallback: bool = True

class CMIComplementarityScorer:
    """
    CMI Complementarity Scorer for feature selection.
    
    Implements:
    - Relevance: R(Xi) = I(Y; Xi | A)
    - Redundancy: D(Xi, S) = mean_j I(Xi; Xj | A)
    - Greedy scoring: Score(Xi) = R(Xi) - α·D(Xi, S) + β·Synergy(Xi, S)
    - Noise floor detection via label permutations
    - Regime-aware aggregation
    """
    
    def __init__(self, config: Optional[CMIComplementarityConfig] = None):
        """Initialize CMI complementarity scorer."""
        self.config = config or CMIComplementarityConfig()
        if self.config.alpha_candidates is None:
            self.config.alpha_candidates = [0.3, 0.5, 0.7]
        
        # Initialize components
        self.cmi_estimator = CMIEstimator()
        self.analyst_handler = AnalystSideInfoHandler()
        
        # Statistics tracking
        self._scoring_stats = {
            'total_scorings': 0,
            'features_evaluated': 0,
            'noise_floor_computations': 0,
            'delta_perf_computations': 0,
            'regime_aggregations': 0,
            'timeout_events': 0,
            'degraded_to_unconditional': 0
        }
        
        tprint_info("🎯 CMI Complementarity Scorer initialized")
        
        # Initialize hardware optimizations
        self._init_hardware_optimizations()
        
        # Initialize VectorBT optimizations
        self._init_vectorbt_optimizations()
        
        # Initialize ML utilities
        self._init_ml_utilities()
    
    def _init_hardware_optimizations(self):
        """Initialize hardware optimizations for M1 chip."""
        if HARDWARE_OPTIMIZATIONS_AVAILABLE:
            try:
                self.gpu_optimizer = M1GPUOptimizer()
                self.memory_optimizer = M1MemoryOptimizer()
                self.cpu_optimizer = M1CPUOptimizer()
                tprint_success("✅ Hardware optimizations initialized for CMI complementarity")
            except Exception as e:
                tprint_warning(f"⚠️ Hardware optimization initialization failed: {e}")
                self.gpu_optimizer = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
        else:
            self.gpu_optimizer = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _init_vectorbt_optimizations(self):
        """Initialize VectorBT optimizations for efficient rolling computations."""
        if VECTORBT_OPTIMIZATIONS_AVAILABLE:
            try:
                self.vectorbt_optimizer = VectorBTRollingOptimizer()
                self.vectorization_manager = UnifiedVectorizationManager()
                tprint_success("✅ VectorBT optimizations initialized for CMI complementarity")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT optimization initialization failed: {e}")
                self.vectorbt_optimizer = None
                self.vectorization_manager = None
        else:
            self.vectorbt_optimizer = None
            self.vectorization_manager = None
    
    def _init_ml_utilities(self):
        """Initialize ML utilities for cross-validation and data leakage detection."""
        if ML_UTILITIES_AVAILABLE:
            try:
                self.purged_kfold = PurgedKFold
                self.data_leakage_detector = DataLeakageDetector()
                self.lookahead_validator = LookaheadValidator()
                self.bayesian_optimizer = BayesianTPEOptimizer()
                tprint_success("✅ ML utilities initialized for CMI complementarity")
            except Exception as e:
                tprint_warning(f"⚠️ ML utility initialization failed: {e}")
                self.purged_kfold = None
                self.data_leakage_detector = None
                self.lookahead_validator = None
                self.bayesian_optimizer = None
        else:
            self.purged_kfold = None
            self.data_leakage_detector = None
            self.lookahead_validator = None
            self.bayesian_optimizer = None
    
    def score_features(self, X: pd.DataFrame, y: pd.Series, A: np.ndarray,
                      family_tags: Optional[Dict[str, str]] = None,
                      cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                      pipeline_state: Optional[Dict[str, Any]] = None) -> CMIComplementarityResult:
        """
        Score features using CMI complementarity.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target series
            A: Analyst side information (n_samples, n_A_dims)
            family_tags: Feature family assignments
            cv_splits: Pre-computed CV splits
            pipeline_state: Pipeline state for regime information
            
        Returns:
            CMIComplementarityResult with selected features and scores
        """
        try:
            start_time = time.time()
            self._scoring_stats['total_scorings'] += 1
            
            # Input validation
            if not self._validate_inputs(X, y, A):
                return self._create_invalid_result("Invalid inputs")
            
            # Check timeout
            if time.time() - start_time > self.config.compute_timeout_seconds:
                self._scoring_stats['timeout_events'] += 1
                tprint_warning("⚠️ CMI complementarity scoring timeout")
                return self._create_invalid_result("Timeout")
            
            # Create CV splits if not provided
            if cv_splits is None:
                cv_splits = self._create_cv_splits(X, y)
            
            # Compute noise floor
            noise_floor = self._compute_noise_floor(X, y, A, cv_splits)
            self._scoring_stats['noise_floor_computations'] += 1
            
            # Compute ΔPerf threshold
            delta_perf_threshold = self._compute_delta_perf_threshold(X, y, A, cv_splits)
            self._scoring_stats['delta_perf_computations'] += 1
            
            # Compute relevance scores R(X|A)
            relevance_scores = self._compute_relevance_scores(X, y, A, cv_splits, pipeline_state)
            
            # Filter by noise floor
            valid_features = [f for f, score in relevance_scores.items() 
                            if score > noise_floor]
            
            if not valid_features:
                tprint_warning("⚠️ No features pass noise floor threshold")
                return self._create_invalid_result("No valid features")
            
            # Apply per-family budget
            if family_tags:
                valid_features = self._apply_family_budget(valid_features, family_tags)
            
            # Greedy selection with redundancy penalty
            selected_features, redundancy_scores, greedy_scores = self._greedy_selection(
                X[valid_features], y, A, cv_splits, pipeline_state
            )
            
            # Apply ΔPerf tie-breaker
            if len(selected_features) > self.config.max_total_features:
                selected_features = self._apply_delta_perf_tiebreaker(
                    X[selected_features], y, A, selected_features, delta_perf_threshold
                )
            
            computation_time = time.time() - start_time
            
            return CMIComplementarityResult(
                selected_features=selected_features,
                relevance_scores=relevance_scores,
                redundancy_scores=redundancy_scores,
                greedy_scores=greedy_scores,
                noise_floor=noise_floor,
                delta_perf_threshold=delta_perf_threshold,
                selection_metadata={
                    'computation_time': computation_time,
                    'n_candidates': len(X.columns),
                    'n_valid': len(valid_features),
                    'n_selected': len(selected_features),
                    'family_tags_used': family_tags is not None,
                    'regime_aware': self.config.enable_regime_awareness
                },
                is_valid=True
            )
            
        except Exception as e:
            tprint_error(f"❌ CMI complementarity scoring failed: {e}")
            return self._create_invalid_result(f"Scoring failed: {e}")
    
    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series, A: np.ndarray) -> bool:
        """Validate input arrays."""
        try:
            if X is None or y is None or A is None:
                return False
            
            if len(X) != len(y) or len(X) != len(A):
                return False
            
            if len(X) < 10:  # Minimum samples
                return False
            
            return True
            
        except Exception:
            return False
    
    def _create_cv_splits(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create purged K-fold CV splits."""
        try:
            n_samples = len(X)
            kf = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
            
            splits = []
            for train_idx, val_idx in kf.split(X):
                # Apply embargo (remove samples within embargo_windows of validation)
                if self.config.embargo_windows > 0:
                    # Simple embargo: remove samples around validation indices
                    embargo_mask = np.ones(len(X), dtype=bool)
                    for val_i in val_idx:
                        start = max(0, val_i - self.config.embargo_windows)
                        end = min(len(X), val_i + self.config.embargo_windows + 1)
                        embargo_mask[start:end] = False
                    
                    train_idx = train_idx[embargo_mask[train_idx]]
                
                splits.append((train_idx, val_idx))
            
            return splits
            
        except Exception as e:
            tprint_warning(f"⚠️ CV split creation failed: {e}")
            # Fallback to simple K-fold
            kf = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
            return list(kf.split(X))
    
    def _compute_noise_floor(self, X: pd.DataFrame, y: pd.Series, A: np.ndarray,
                            cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Compute noise floor using label permutations."""
        try:
            null_scores = []
            
            for _ in range(self.config.noise_floor_permutations):
                # Shuffle y within A-strata
                y_shuffled = self._shuffle_within_strata(y, A)
                
                # Compute R(X|A) with shuffled y for a subset of features
                sample_features = np.random.choice(X.columns, 
                                                 min(50, len(X.columns)), 
                                                 replace=False)
                
                for feature in sample_features:
                    try:
                        # Use GCMI for noise floor (faster)
                        result = self.cmi_estimator.estimate_cmi(
                            X[[feature]].values, y_shuffled.values, A,
                            estimator='gcmi', stage='prefilter'
                        )
                        if result.is_valid:
                            null_scores.append(result.mi_value)
                    except Exception:
                        continue
                
                if len(null_scores) >= 100:  # Enough samples
                    break
            
            if not null_scores:
                return 0.001  # Default fallback
            
            noise_floor = np.percentile(null_scores, self.config.noise_floor_percentile)
            tprint_info(f"✅ Computed noise floor: {noise_floor:.6f}")
            return noise_floor
            
        except Exception as e:
            tprint_warning(f"⚠️ Noise floor computation failed: {e}")
            return 0.001  # Default fallback
    
    def _compute_delta_perf_threshold(self, X: pd.DataFrame, y: pd.Series, A: np.ndarray,
                                    cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Compute data-driven ΔPerf threshold using null permutations."""
        try:
            null_delta_perfs = []
            
            for _ in range(self.config.delta_perf_permutations):
                # Shuffle y
                y_shuffled = y.sample(frac=1.0, random_state=np.random.randint(0, 10000))
                y_shuffled.index = y.index
                
                # Compute ΔPerf with shuffled y
                delta_perf = self._compute_delta_perf(X, y_shuffled, A, cv_splits)
                if not np.isnan(delta_perf):
                    null_delta_perfs.append(delta_perf)
                
                if len(null_delta_perfs) >= 20:  # Enough samples
                    break
            
            if not null_delta_perfs:
                return self.config.delta_perf_plateau_threshold  # Default fallback
            
            delta_perf_threshold = np.percentile(null_delta_perfs, 75)  # 75th percentile
            tprint_info(f"✅ Computed ΔPerf threshold: {delta_perf_threshold:.6f}")
            return delta_perf_threshold
            
        except Exception as e:
            tprint_warning(f"⚠️ ΔPerf threshold computation failed: {e}")
            return self.config.delta_perf_plateau_threshold  # Default fallback
    
    def _compute_relevance_scores(self, X: pd.DataFrame, y: pd.Series, A: np.ndarray,
                                 cv_splits: List[Tuple[np.ndarray, np.ndarray]],
                                 pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Compute relevance scores R(X|A) for all features."""
        try:
            relevance_scores = {}
            
            # Check for regime awareness
            if (self.config.enable_regime_awareness and pipeline_state and 
                'regime_analysis_result' in pipeline_state):
                # Regime-aware computation
                relevance_scores = self._compute_regime_aware_relevance(
                    X, y, A, cv_splits, pipeline_state
                )
                self._scoring_stats['regime_aggregations'] += 1
            else:
                # Standard computation
                for feature in X.columns:
                    try:
                        # Compute R(X|A) across CV folds
                        fold_scores = []
                        
                        for train_idx, val_idx in cv_splits:
                            X_train = X.iloc[train_idx][[feature]]
                            y_train = y.iloc[train_idx]
                            A_train = A[train_idx]
                            
                            result = self.cmi_estimator.estimate_cmi(
                                X_train.values, y_train.values, A_train,
                                stage='prefilter'
                            )
                            
                            if result.is_valid:
                                fold_scores.append(result.mi_value)
                        
                        if fold_scores:
                            relevance_scores[feature] = np.median(fold_scores)
                        else:
                            relevance_scores[feature] = 0.0
                            
                    except Exception as e:
                        tprint_debug(f"⚠️ Relevance computation failed for {feature}: {e}")
                        relevance_scores[feature] = 0.0
            
            self._scoring_stats['features_evaluated'] += len(X.columns)
            tprint_info(f"✅ Computed relevance scores for {len(relevance_scores)} features")
            return relevance_scores
            
        except Exception as e:
            tprint_error(f"❌ Relevance score computation failed: {e}")
            return {}
    
    def _compute_regime_aware_relevance(self, X: pd.DataFrame, y: pd.Series, A: np.ndarray,
                                       cv_splits: List[Tuple[np.ndarray, np.ndarray]],
                                       pipeline_state: Dict[str, Any]) -> Dict[str, float]:
        """Compute regime-aware relevance scores."""
        try:
            regime_result = pipeline_state['regime_analysis_result']
            regime_labels = regime_result.get('regime_labels', [])
            
            if not regime_labels or len(regime_labels) != len(X):
                tprint_warning("⚠️ Invalid regime labels, using standard computation")
                return self._compute_relevance_scores(X, y, A, cv_splits, None)
            
            relevance_scores = {}
            
            for feature in X.columns:
                try:
                    # Compute R(X|A) per regime
                    regime_scores = {}
                    regime_occupancies = {}
                    
                    for regime_id in np.unique(regime_labels):
                        regime_mask = np.array(regime_labels) == regime_id
                        regime_occupancy = np.sum(regime_mask) / len(regime_mask)
                        
                        if regime_occupancy < self.config.regime_occupancy_threshold:
                            continue  # Skip rare regimes
                        
                        # Get regime data
                        X_regime = X[regime_mask][[feature]]
                        y_regime = y[regime_mask]
                        A_regime = A[regime_mask]
                        
                        if len(X_regime) < 10:  # Need minimum samples
                            continue
                        
                        # Compute R(X|A) for this regime
                        result = self.cmi_estimator.estimate_cmi(
                            X_regime.values, y_regime.values, A_regime,
                            stage='prefilter'
                        )
                        
                        if result.is_valid:
                            regime_scores[regime_id] = result.mi_value
                            regime_occupancies[regime_id] = regime_occupancy
                    
                    if regime_scores:
                        # Aggregate with occupancy weights
                        weighted_score = sum(
                            score * occupancy for score, occupancy in 
                            zip(regime_scores.values(), regime_occupancies.values())
                        ) / sum(regime_occupancies.values())
                        relevance_scores[feature] = weighted_score
                    else:
                        relevance_scores[feature] = 0.0
                        
                except Exception as e:
                    tprint_debug(f"⚠️ Regime-aware relevance failed for {feature}: {e}")
                    relevance_scores[feature] = 0.0
            
            tprint_info(f"✅ Computed regime-aware relevance for {len(relevance_scores)} features")
            return relevance_scores
            
        except Exception as e:
            tprint_warning(f"⚠️ Regime-aware relevance computation failed: {e}")
            return self._compute_relevance_scores(X, y, A, cv_splits, None)
    
    def _apply_family_budget(self, features: List[str], 
                           family_tags: Dict[str, str]) -> List[str]:
        """Apply per-family budget constraints."""
        try:
            # Group features by family
            family_features = {}
            for feature in features:
                family = family_tags.get(feature, 'unknown')
                if family not in family_features:
                    family_features[family] = []
                family_features[family].append(feature)
            
            # Apply budget per family
            selected_features = []
            for family, family_feature_list in family_features.items():
                # Sort by relevance (assuming they're already sorted)
                n_select = min(len(family_feature_list), self.config.per_family_budget[1])
                n_select = max(n_select, self.config.per_family_budget[0])
                selected_features.extend(family_feature_list[:n_select])
            
            tprint_info(f"✅ Applied family budget: {len(selected_features)} features selected")
            return selected_features
            
        except Exception as e:
            tprint_warning(f"⚠️ Family budget application failed: {e}")
            return features
    
    def _greedy_selection(self, X: pd.DataFrame, y: pd.Series, A: np.ndarray,
                        cv_splits: List[Tuple[np.ndarray, np.ndarray]],
                        pipeline_state: Optional[Dict[str, Any]] = None) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
        """Perform greedy selection with redundancy penalty."""
        try:
            selected_features = []
            redundancy_scores = {}
            greedy_scores = {}
            
            # Tune alpha via CV
            best_alpha = self._tune_alpha(X, y, A, cv_splits)
            
            remaining_features = list(X.columns)
            
            for step in range(min(len(remaining_features), self.config.max_total_features)):
                if not remaining_features:
                    break
                
                # Compute scores for remaining features
                feature_scores = {}
                
                for feature in remaining_features:
                    try:
                        # Compute relevance R(X|A)
                        relevance = self._compute_feature_relevance(
                            X[[feature]], y, A, cv_splits
                        )
                        
                        # Compute redundancy D(X,S|A)
                        redundancy = self._compute_feature_redundancy(
                            X[[feature]], X[selected_features], A, cv_splits
                        ) if selected_features else 0.0
                        
                        # Compute synergy (optional)
                        synergy = 0.0
                        if self.config.enable_synergy and selected_features:
                            synergy = self._compute_feature_synergy(
                                X[[feature]], X[selected_features], y, A, cv_splits
                            )
                        
                        # Greedy score
                        score = (relevance - 
                                best_alpha * redundancy + 
                                self.config.beta_synergy * synergy)
                        
                        feature_scores[feature] = score
                        redundancy_scores[feature] = redundancy
                        greedy_scores[feature] = score
                        
                    except Exception as e:
                        tprint_debug(f"⚠️ Score computation failed for {feature}: {e}")
                        feature_scores[feature] = -np.inf
                
                # Select best feature
                if not feature_scores or all(score == -np.inf for score in feature_scores.values()):
                    break
                
                best_feature = max(feature_scores, key=feature_scores.get)
                if feature_scores[best_feature] <= 0:
                    break  # Stop if no positive scores
                
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                
                tprint_debug(f"✅ Selected feature {step+1}: {best_feature} (score: {feature_scores[best_feature]:.6f})")
            
            tprint_info(f"✅ Greedy selection completed: {len(selected_features)} features selected")
            return selected_features, redundancy_scores, greedy_scores
            
        except Exception as e:
            tprint_error(f"❌ Greedy selection failed: {e}")
            return [], {}, {}
    
    def _tune_alpha(self, X: pd.DataFrame, y: pd.Series, A: np.ndarray,
                   cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Tune alpha parameter via CV."""
        try:
            best_alpha = self.config.alpha_candidates[0]
            best_score = -np.inf
            
            for alpha in self.config.alpha_candidates:
                # Quick evaluation on subset of features
                sample_features = np.random.choice(X.columns, 
                                                 min(20, len(X.columns)), 
                                                 replace=False)
                
                score = self._evaluate_alpha(X[sample_features], y, A, cv_splits, alpha)
                
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
            
            tprint_info(f"✅ Tuned alpha: {best_alpha} (score: {best_score:.6f})")
            return best_alpha
            
        except Exception as e:
            tprint_warning(f"⚠️ Alpha tuning failed: {e}")
            return self.config.alpha_candidates[0]
    
    def _evaluate_alpha(self, X: pd.DataFrame, y: pd.Series, A: np.ndarray,
                       cv_splits: List[Tuple[np.ndarray, np.ndarray]], alpha: float) -> float:
        """Evaluate alpha parameter."""
        try:
            scores = []
            
            for train_idx, val_idx in cv_splits:
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                A_train = A[train_idx]
                
                # Simple greedy selection on training set
                selected = []
                remaining = list(X_train.columns)
                
                for _ in range(min(5, len(remaining))):  # Select top 5
                    if not remaining:
                        break
                    
                    feature_scores = {}
                    for feature in remaining:
                        relevance = self._compute_feature_relevance(
                            X_train[[feature]], y_train, A_train, [(np.arange(len(X_train)), np.arange(len(X_train)))]
                        )
                        redundancy = self._compute_feature_redundancy(
                            X_train[[feature]], X_train[selected], A_train, [(np.arange(len(X_train)), np.arange(len(X_train)))]
                        ) if selected else 0.0
                        
                        score = relevance - alpha * redundancy
                        feature_scores[feature] = score
                    
                    if not feature_scores or all(score == -np.inf for score in feature_scores.values()):
                        break
                    
                    best_feature = max(feature_scores, key=feature_scores.get)
                    if feature_scores[best_feature] <= 0:
                        break
                    
                    selected.append(best_feature)
                    remaining.remove(best_feature)
                
                # Evaluate on validation set
                if selected:
                    val_score = self._compute_validation_score(
                        X.iloc[val_idx][selected], y.iloc[val_idx], A[val_idx]
                    )
                    scores.append(val_score)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            tprint_debug(f"⚠️ Alpha evaluation failed: {e}")
            return 0.0
    
    def _compute_feature_relevance(self, X: pd.DataFrame, y: pd.Series, A: np.ndarray,
                                  cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Compute relevance R(X|A) for a single feature."""
        try:
            fold_scores = []
            
            for train_idx, val_idx in cv_splits:
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                A_train = A[train_idx]
                
                result = self.cmi_estimator.estimate_cmi(
                    X_train.values, y_train.values, A_train,
                    stage='prefilter'
                )
                
                if result.is_valid:
                    fold_scores.append(result.mi_value)
            
            return np.median(fold_scores) if fold_scores else 0.0
            
        except Exception as e:
            tprint_debug(f"⚠️ Feature relevance computation failed: {e}")
            return 0.0
    
    def _compute_feature_redundancy(self, X: pd.DataFrame, X_selected: pd.DataFrame, A: np.ndarray,
                                   cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Compute redundancy D(X,S|A) for a single feature."""
        try:
            if X_selected.empty:
                return 0.0
            
            redundancy_scores = []
            
            for selected_feature in X_selected.columns:
                try:
                    # Compute I(X; X_selected | A)
                    X_combined = pd.concat([X, X_selected[[selected_feature]]], axis=1)
                    
                    fold_scores = []
                    for train_idx, val_idx in cv_splits:
                        X_train = X_combined.iloc[train_idx]
                        A_train = A[train_idx]
                        
                        result = self.cmi_estimator.estimate_cmi(
                            X_train.values, A_train, A_train,  # I(X; X_selected | A)
                            stage='prefilter'
                        )
                        
                        if result.is_valid:
                            fold_scores.append(result.mi_value)
                    
                    if fold_scores:
                        redundancy_scores.append(np.median(fold_scores))
                        
                except Exception as e:
                    tprint_debug(f"⚠️ Redundancy computation failed for {selected_feature}: {e}")
                    continue
            
            return np.mean(redundancy_scores) if redundancy_scores else 0.0
            
        except Exception as e:
            tprint_debug(f"⚠️ Feature redundancy computation failed: {e}")
            return 0.0
    
    def _compute_feature_synergy(self, X: pd.DataFrame, X_selected: pd.DataFrame, y: pd.Series, A: np.ndarray,
                                cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Compute synergy for a single feature with selected features."""
        try:
            if X_selected.empty:
                return 0.0
            
            synergy_scores = []
            
            for selected_feature in X_selected.columns:
                try:
                    # Compute I(Y; X, X_selected | A) - I(Y; X | A) - I(Y; X_selected | A)
                    X_combined = pd.concat([X, X_selected[[selected_feature]]], axis=1)
                    
                    # Joint MI
                    joint_mi = self._compute_feature_relevance(X_combined, y, A, cv_splits)
                    
                    # Individual MIs
                    X_mi = self._compute_feature_relevance(X, y, A, cv_splits)
                    X_selected_mi = self._compute_feature_relevance(X_selected[[selected_feature]], y, A, cv_splits)
                    
                    # Synergy
                    synergy = joint_mi - X_mi - X_selected_mi
                    synergy_scores.append(max(0.0, synergy))  # Synergy is non-negative
                    
                except Exception as e:
                    tprint_debug(f"⚠️ Synergy computation failed for {selected_feature}: {e}")
                    continue
            
            return np.mean(synergy_scores) if synergy_scores else 0.0
            
        except Exception as e:
            tprint_debug(f"⚠️ Feature synergy computation failed: {e}")
            return 0.0
    
    def _apply_delta_perf_tiebreaker(self, X: pd.DataFrame, y: pd.Series, A: np.ndarray,
                                    selected_features: List[str], delta_perf_threshold: float) -> List[str]:
        """Apply ΔPerf tie-breaker to select final features."""
        try:
            if len(selected_features) <= self.config.max_total_features:
                return selected_features
            
            # Compute ΔPerf for each feature
            feature_delta_perfs = {}
            
            for feature in selected_features:
                try:
                    delta_perf = self._compute_delta_perf(X[[feature]], y, A)
                    feature_delta_perfs[feature] = delta_perf
                except Exception as e:
                    tprint_debug(f"⚠️ ΔPerf computation failed for {feature}: {e}")
                    feature_delta_perfs[feature] = 0.0
            
            # Sort by ΔPerf and select top features
            sorted_features = sorted(feature_delta_perfs.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            final_features = []
            for feature, delta_perf in sorted_features:
                if delta_perf >= delta_perf_threshold:
                    final_features.append(feature)
                    if len(final_features) >= self.config.max_total_features:
                        break
            
            tprint_info(f"✅ Applied ΔPerf tie-breaker: {len(final_features)} features selected")
            return final_features
            
        except Exception as e:
            tprint_warning(f"⚠️ ΔPerf tie-breaker failed: {e}")
            return selected_features[:self.config.max_total_features]
    
    def _compute_delta_perf(self, X: pd.DataFrame, y: pd.Series, A: np.ndarray,
                           cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None) -> float:
        """Compute ΔPerf = Perf(Analyst ∪ features) - Perf(Analyst)."""
        try:
            if cv_splits is None:
                cv_splits = self._create_cv_splits(X, y)
            
            # Baseline: Analyst-only model
            baseline_scores = []
            # Enhanced: Analyst + features model
            enhanced_scores = []
            
            for train_idx, val_idx in cv_splits:
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                A_train = A[train_idx]
                
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]
                A_val = A[val_idx]
                
                # Baseline model (Analyst only)
                baseline_model = RandomForestRegressor(n_estimators=50, random_state=42)
                baseline_model.fit(A_train, y_train)
                baseline_pred = baseline_model.predict(A_val)
                baseline_score = self._compute_performance_score(y_val, baseline_pred)
                baseline_scores.append(baseline_score)
                
                # Enhanced model (Analyst + features)
                enhanced_features = np.column_stack([A_val, X_val.values])
                enhanced_model = RandomForestRegressor(n_estimators=50, random_state=42)
                enhanced_model.fit(np.column_stack([A_train, X_train.values]), y_train)
                enhanced_pred = enhanced_model.predict(enhanced_features)
                enhanced_score = self._compute_performance_score(y_val, enhanced_pred)
                enhanced_scores.append(enhanced_score)
            
            delta_perf = np.mean(enhanced_scores) - np.mean(baseline_scores)
            return delta_perf
            
        except Exception as e:
            tprint_debug(f"⚠️ ΔPerf computation failed: {e}")
            return 0.0
    
    def _compute_performance_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Compute performance score (AUC for binary, R² for continuous)."""
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                return roc_auc_score(y_true, y_pred)
            else:  # Regression
                return 1.0 - mean_squared_error(y_true, y_pred) / np.var(y_true)
        except Exception:
            return 0.0
    
    def _compute_validation_score(self, X: pd.DataFrame, y: pd.Series, A: np.ndarray) -> float:
        """Compute validation score for feature set."""
        try:
            # Simple model for validation
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            features = np.column_stack([A, X.values])
            model.fit(features, y)
            pred = model.predict(features)
            return self._compute_performance_score(y, pred)
        except Exception:
            return 0.0
    
    def _shuffle_within_strata(self, y: pd.Series, A: np.ndarray) -> pd.Series:
        """Shuffle y within A-strata (quantile bins)."""
        try:
            # Create A-strata
            if A.shape[1] == 1:
                strata = pd.qcut(A.flatten(), q=5, labels=False, duplicates='drop')
            else:
                # Use first dimension for strata
                strata = pd.qcut(A[:, 0], q=5, labels=False, duplicates='drop')
            
            y_shuffled = y.copy()
            
            # Shuffle within each stratum
            for stratum_id in np.unique(strata):
                stratum_mask = strata == stratum_id
                stratum_indices = np.where(stratum_mask)[0]
                
                if len(stratum_indices) > 1:
                    # Shuffle within stratum
                    stratum_values = y.iloc[stratum_indices].values
                    np.random.shuffle(stratum_values)
                    y_shuffled.iloc[stratum_indices] = stratum_values
            
            return y_shuffled
            
        except Exception as e:
            tprint_debug(f"⚠️ Stratified shuffling failed: {e}")
            return y.sample(frac=1.0, random_state=np.random.randint(0, 10000))
    
    def _create_invalid_result(self, error_message: str) -> CMIComplementarityResult:
        """Create an invalid result with error message."""
        return CMIComplementarityResult(
            selected_features=[],
            relevance_scores={},
            redundancy_scores={},
            greedy_scores={},
            noise_floor=0.0,
            delta_perf_threshold=0.0,
            selection_metadata={'error': error_message},
            is_valid=False
        )
    
    def get_scoring_stats(self) -> Dict[str, Any]:
        """Get scoring statistics."""
        return self._scoring_stats.copy()

def create_cmi_complementarity_scorer(config: Optional[CMIComplementarityConfig] = None) -> CMIComplementarityScorer:
    """Create a CMI complementarity scorer with default configuration."""
    return CMIComplementarityScorer(config)
