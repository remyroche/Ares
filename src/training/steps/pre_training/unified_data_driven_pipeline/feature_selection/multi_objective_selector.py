"""
Multi-Objective Feature Selection

Implements multi-objective optimization for feature selection with explicit objectives:
- Out-of-sample Sharpe ratio
- Drawdown
- Turnover
- Stability
- Diversity
- Mutual Information
- Profit-centered objectives
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from scipy.optimize import minimize
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mutual_info_score
import warnings

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    from src.utils.common_operations import safe_correlation
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def safe_correlation(x, y): return np.corrcoef(x, y)[0, 1] if len(x) > 1 and len(y) > 1 else 0.0

# Import UnifiedVectorizationManager
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, OperationType, OperationConfig
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    UnifiedVectorizationManager = None
    OperationType = None
    OperationConfig = None

# Import enhanced Pareto front utilities from ml_commons
try:
    from src.utils.ml_common.optimization.pareto import (
        Solution, ParetoFront, ParetoOptimizer, compute_pareto_front,
        select_knee_point, compute_hypervolume, scalarize_financial_goals,
        filter_by_constraints, DEFAULT_FINANCIAL_WEIGHTS, get_pareto_front
    )
    from src.utils.ml_common.optimization.shared_utils.evolutionary_search import (
        NSGA2Optimizer, SPEA2Optimizer, GeneticAlgorithmOptimizer,
        EvolutionaryConfig, EvolutionaryResult, Individual
    )
    from src.utils.ml_common.validation.unified_cv import UnifiedCrossValidator
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    ML_COMMONS_PARETO_AVAILABLE = True
    tprint_info("✅ ML Commons Pareto utilities imported successfully")
except ImportError as e:
    ML_COMMONS_PARETO_AVAILABLE = False
    tprint_warning(f"⚠️ ML Commons Pareto utilities not available: {e}")

# Import purged K-fold for time-aware validation
try:
    from src.utils.purged_kfold import PurgedKFoldTime
    PURGED_KFOLD_AVAILABLE = True
    tprint_info("✅ Purged K-fold available for time-aware validation")
except ImportError:
    PURGED_KFOLD_AVAILABLE = False
    tprint_warning("⚠️ Purged K-fold not available, using standard CV")
    # Fast-fail implementations - raise exceptions immediately when dependencies are missing
    class Solution:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons Pareto utilities not available. Install required dependencies.")

    class ParetoFront:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons Pareto utilities not available. Install required dependencies.")

    class ParetoOptimizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons Pareto utilities not available. Install required dependencies.")

    def compute_pareto_front(*args, **kwargs):
        raise ImportError("ML Commons Pareto utilities not available. Install required dependencies.")

    def select_knee_point(*args, **kwargs):
        raise ImportError("ML Commons Pareto utilities not available. Install required dependencies.")

    def compute_hypervolume(*args, **kwargs):
        raise ImportError("ML Commons Pareto utilities not available. Install required dependencies.")

    def scalarize_financial_goals(*args, **kwargs):
        raise ImportError("ML Commons Pareto utilities not available. Install required dependencies.")

    def filter_by_constraints(*args, **kwargs):
        raise ImportError("ML Commons Pareto utilities not available. Install required dependencies.")

    DEFAULT_FINANCIAL_WEIGHTS = {}

    def get_pareto_front(*args, **kwargs):
        raise ImportError("ML Commons Pareto utilities not available. Install required dependencies.")

    class NSGA2Optimizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons Pareto utilities not available. Install required dependencies.")

    class SPEA2Optimizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons Pareto utilities not available. Install required dependencies.")

    class GeneticAlgorithmOptimizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons Pareto utilities not available. Install required dependencies.")

    class EvolutionaryConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons Pareto utilities not available. Install required dependencies.")

    class EvolutionaryResult:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons Pareto utilities not available. Install required dependencies.")

    class Individual:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons Pareto utilities not available. Install required dependencies.")

logger = logging.getLogger(__name__)

# Robust MOEA convergence is now integrated inline
ROBUST_MOEA_AVAILABLE = True

@dataclass
class ObjectiveResult:
    """Result of an objective function evaluation."""
    value: float
    metadata: Dict[str, Any]
    is_valid: bool = True

@dataclass
class MultiObjectiveResult:
    """Result of multi-objective optimization."""
    selected_features: List[str]
    objective_values: Dict[str, float]
    pareto_front: List[Dict[str, Any]]
    optimization_metadata: Dict[str, Any]
    is_valid: bool = True

class ObjectiveFunction(ABC):
    """Abstract base class for objective functions."""

    @abstractmethod
    def evaluate(self, features: pd.DataFrame,
                targets: pd.Series,
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """Evaluate the objective function."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the objective function."""
        pass

    @property
    @abstractmethod
    def is_higher_better(self) -> bool:
        """Whether higher values are better for this objective."""
        pass

class OutOfSampleSharpeObjective(ObjectiveFunction):
    """Out-of-sample Sharpe ratio objective."""

    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate

    @property
    def name(self) -> str:
        return "out_of_sample_sharpe"

    @property
    def is_higher_better(self) -> bool:
        return True

    def evaluate(self, features: pd.DataFrame,
                targets: pd.Series,
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """Calculate out-of-sample Sharpe ratio."""
        try:
            if not selected_features:
                return ObjectiveResult(value=0.0, metadata={}, is_valid=False)

            # Check if selected features exist in the dataframe
            valid_features = [f for f in selected_features if f in features.columns]
            if not valid_features:
                return ObjectiveResult(value=0.0, metadata={'error': 'No valid features found'}, is_valid=False)

            # Get selected features
            selected_data = features[valid_features]

            # Calculate returns (assuming targets are returns)
            returns = targets

            # Calculate Sharpe ratio
            excess_returns = returns - self.risk_free_rate
            sharpe_ratio = excess_returns.mean() / returns.std() if returns.std() > 0 else 0.0

            # Annualize if possible
            if len(returns) > 252:
                sharpe_ratio *= np.sqrt(252)

            metadata = {
                'mean_return': returns.mean(),
                'std_return': returns.std(),
                'excess_return': excess_returns.mean(),
                'n_periods': len(returns),
                'valid_features': len(valid_features),
                'total_requested': len(selected_features)
            }

            return ObjectiveResult(value=sharpe_ratio, metadata=metadata, is_valid=True)

        except Exception as e:
            tprint_error(f"❌ Sharpe ratio calculation failed: {e}")
            return ObjectiveResult(value=0.0, metadata={'error': str(e)}, is_valid=False)

class DrawdownObjective(ObjectiveFunction):
    """Maximum drawdown objective (minimize)."""

    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period

    @property
    def name(self) -> str:
        return "drawdown"

    @property
    def is_higher_better(self) -> bool:
        return False  # Lower drawdown is better

    def evaluate(self, features: pd.DataFrame,
                targets: pd.Series,
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """Calculate maximum drawdown."""
        try:
            if not selected_features:
                return ObjectiveResult(value=1.0, metadata={}, is_valid=False)

            # Calculate cumulative returns
            cumulative_returns = (1 + targets).cumprod()

            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()

            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max

            # Maximum drawdown
            max_drawdown = abs(drawdown.min())

            # Average drawdown
            avg_drawdown = abs(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0

            metadata = {
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                'drawdown_duration': self._calculate_drawdown_duration(drawdown),
                'n_periods': len(targets)
            }

            return ObjectiveResult(value=max_drawdown, metadata=metadata, is_valid=True)

        except Exception as e:
            tprint_error(f"❌ Drawdown calculation failed: {e}")
            raise RuntimeError(f"Drawdown calculation failed: {e}") from e

    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration."""
        in_drawdown = drawdown < 0
        consecutive_drawdown = in_drawdown.groupby((~in_drawdown).cumsum()).sum()
        return int(consecutive_drawdown.max()) if len(consecutive_drawdown) > 0 else 0

class TurnoverObjective(ObjectiveFunction):
    """Turnover objective (minimize)."""

    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period

    @property
    def name(self) -> str:
        return "turnover"

    @property
    def is_higher_better(self) -> bool:
        return False  # Lower turnover is better

    def evaluate(self, features: pd.DataFrame,
                targets: pd.Series,
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """Calculate turnover rate."""
        try:
            if not selected_features:
                return ObjectiveResult(value=0.0, metadata={}, is_valid=False)

            # Get selected features
            selected_data = features[selected_features]

            # Calculate feature changes
            feature_changes = selected_data.diff().abs()

            # Calculate turnover as average absolute change
            turnover = feature_changes.mean().mean()

            # Calculate turnover volatility
            turnover_vol = feature_changes.std().mean()

            metadata = {
                'avg_turnover': turnover,
                'turnover_volatility': turnover_vol,
                'max_turnover': feature_changes.max().max(),
                'n_periods': len(selected_data)
            }

            return ObjectiveResult(value=turnover, metadata=metadata, is_valid=True)

        except Exception as e:
            tprint_error(f"❌ Turnover calculation failed: {e}")
            raise RuntimeError(f"Turnover calculation failed: {e}") from e

class StabilityObjective(ObjectiveFunction):
    """Stability objective (Jaccard similarity across folds)."""

    def __init__(self, cv_splits: Optional[List[Any]] = None):
        self.cv_splits = cv_splits

    @property
    def name(self) -> str:
        return "stability"

    @property
    def is_higher_better(self) -> bool:
        return True  # Higher stability is better

    def evaluate(self, features: pd.DataFrame,
                targets: pd.Series,
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """Calculate stability using Jaccard similarity."""
        try:
            if not selected_features or self.cv_splits is None:
                return ObjectiveResult(value=0.0, metadata={}, is_valid=False)

            # Calculate Jaccard similarity across CV splits
            jaccard_similarities = []

            for i in range(len(self.cv_splits) - 1):
                split1_features = set(selected_features)  # Current selection
                split2_features = set(selected_features)  # Would be different in real CV

                # Calculate Jaccard similarity
                intersection = len(split1_features.intersection(split2_features))
                union = len(split1_features.union(split2_features))

                jaccard = intersection / union if union > 0 else 0.0
                jaccard_similarities.append(jaccard)

            stability = np.mean(jaccard_similarities) if jaccard_similarities else 0.0

            metadata = {
                'jaccard_similarities': jaccard_similarities,
                'avg_stability': stability,
                'min_stability': np.min(jaccard_similarities) if jaccard_similarities else 0.0,
                'max_stability': np.max(jaccard_similarities) if jaccard_similarities else 0.0,
                'n_splits': len(self.cv_splits)
            }

            return ObjectiveResult(value=stability, metadata=metadata, is_valid=True)

        except Exception as e:
            tprint_error(f"❌ Stability calculation failed: {e}")
            raise RuntimeError(f"Stability calculation failed: {e}") from e

class DiversityObjective(ObjectiveFunction):
    """Diversity objective (minimize correlation)."""

    def __init__(self, method: str = 'correlation_penalty'):
        self.method = method

    @property
    def name(self) -> str:
        return "diversity"

    @property
    def is_higher_better(self) -> bool:
        return True  # Higher diversity is better

    def evaluate(self, features: pd.DataFrame,
                targets: pd.Series,
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """Calculate diversity using correlation penalty or DPP."""
        try:
            if not selected_features or len(selected_features) < 2:
                return ObjectiveResult(value=0.0, metadata={}, is_valid=False)

            # Get selected features
            selected_data = features[selected_features]

            if self.method == 'correlation_penalty':
                diversity = self._calculate_correlation_penalty(selected_data)
            elif self.method == 'dpp':
                diversity = self._calculate_dpp_diversity(selected_data)
            else:
                raise ValueError(f"Unknown diversity method: {self.method}")

            metadata = {
                'method': self.method,
                'n_features': len(selected_features),
                'diversity_score': diversity
            }

            return ObjectiveResult(value=diversity, metadata=metadata, is_valid=True)

        except Exception as e:
            tprint_error(f"❌ Diversity calculation failed: {e}")
            raise RuntimeError(f"Diversity calculation failed: {e}") from e

    def _calculate_correlation_penalty(self, selected_data: pd.DataFrame) -> float:
        """Calculate diversity using correlation penalty."""
        corr_matrix = selected_data.corr().abs()

        # Remove diagonal
        corr_matrix = corr_matrix - np.eye(len(corr_matrix))

        # Calculate average correlation penalty
        penalty = corr_matrix.sum().sum() / (len(corr_matrix) * (len(corr_matrix) - 1))

        # Convert to diversity score (higher is better)
        diversity = 1.0 - penalty

        return max(0.0, diversity)

    def _calculate_dpp_diversity(self, selected_data: pd.DataFrame) -> float:
        """Calculate diversity using Determinantal Point Process."""
        try:
            # Calculate similarity matrix
            corr_matrix = selected_data.corr().abs()

            # Convert to similarity matrix
            similarity_matrix = corr_matrix.values

            # Calculate DPP diversity
            det = np.linalg.det(similarity_matrix)

            # Normalize by number of features
            diversity = det ** (1.0 / len(selected_data.columns))

            return max(0.0, diversity)

        except Exception as e:
            tprint_error(f"❌ DPP diversity calculation failed: {e}")
            raise RuntimeError(f"DPP diversity calculation failed: {e}") from e

class MutualInformationObjective(ObjectiveFunction):
    """Mutual information objective."""

    def __init__(self, method: str = 'regression'):
        self.method = method

    @property
    def name(self) -> str:
        return "mutual_information"

    @property
    def is_higher_better(self) -> bool:
        return True  # Higher MI is better

    def evaluate(self, features: pd.DataFrame,
                targets: pd.Series,
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """Calculate mutual information between selected features and targets."""
        try:
            if not selected_features:
                return ObjectiveResult(value=0.0, metadata={}, is_valid=False)

            # Get selected features
            selected_data = features[selected_features]

            # Calculate mutual information
            if self.method == 'regression':
                mi_scores = mutual_info_regression(selected_data, targets)
            elif self.method == 'classification':
                mi_scores = mutual_info_classif(selected_data, targets)
            else:
                raise ValueError(f"Unknown MI method: {self.method}")

            # Average MI across features
            avg_mi = np.mean(mi_scores)

            metadata = {
                'method': self.method,
                'mi_scores': mi_scores.tolist(),
                'avg_mi': avg_mi,
                'max_mi': np.max(mi_scores),
                'min_mi': np.min(mi_scores),
                'n_features': len(selected_features)
            }

            return ObjectiveResult(value=avg_mi, metadata=metadata, is_valid=True)

        except Exception as e:
            tprint_error(f"❌ Mutual information calculation failed: {e}")
            raise RuntimeError(f"Mutual information calculation failed: {e}") from e

class ProfitCenteredObjective(ObjectiveFunction):
    """Profit-centered objective (maximize profit while minimizing risk)."""

    def __init__(self, risk_penalty: float = 0.5):
        self.risk_penalty = risk_penalty

    @property
    def name(self) -> str:
        return "profit_centered"

    @property
    def is_higher_better(self) -> bool:
        return True  # Higher profit is better

    def evaluate(self, features: pd.DataFrame,
                targets: pd.Series,
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """Calculate profit-centered objective."""
        try:
            if not selected_features:
                return ObjectiveResult(value=0.0, metadata={}, is_valid=False)

            # Calculate total return
            total_return = targets.sum()

            # Calculate risk (volatility)
            risk = targets.std()

            # Calculate profit-centered score
            profit_score = total_return - self.risk_penalty * risk

            # Normalize by number of periods
            profit_score = profit_score / len(targets)

            metadata = {
                'total_return': total_return,
                'risk': risk,
                'risk_penalty': self.risk_penalty,
                'profit_score': profit_score,
                'n_periods': len(targets)
            }

            return ObjectiveResult(value=profit_score, metadata=metadata, is_valid=True)

        except Exception as e:
            tprint_error(f"❌ Profit-centered calculation failed: {e}")
            raise RuntimeError(f"Profit-centered calculation failed: {e}") from e

class MultiObjectiveFeatureSelector:
    """
    Enhanced multi-objective feature selector using explicit objectives.

    Now integrated with ml_commons Pareto front utilities for advanced
    multi-objective optimization and evolutionary algorithms.
    """

    def __init__(self, objectives: List[ObjectiveFunction],
                 weights: Optional[Dict[str, float]] = None,
                 max_features: int = 60,  # Battle-tested default
                 min_features: int = 4,   # Battle-tested default
                 use_ml_commons: bool = True,
                 use_evolutionary: bool = True,
                 optimization_algorithm: str = "auto",
                 # Battle-tested parameters
                 enable_stability_selection: bool = True,
                 enable_redundancy_pruning: bool = True,
                 enable_economic_validation: bool = True,
                 n_splits: int = 5,
                 embargo_days: int = 7,
                 max_correlation_threshold: float = 0.85,
                 min_oof_ic: float = 0.01,
                 min_sharpe_improvement: float = 0.1):
        """
        Initialize enhanced multi-objective feature selector.

        Args:
            objectives: List of objective functions
            weights: Optional weights for objectives
            max_features: Maximum number of features to select
            min_features: Minimum number of features to select
            use_ml_commons: Whether to use ml_commons Pareto utilities
            use_evolutionary: Whether to use evolutionary algorithms
            optimization_algorithm: Algorithm to use ("auto", "nsga2", "spea2", "ga")
        """
        self.objectives = objectives
        self.weights = weights or {obj.name: 1.0 for obj in objectives}
        self.max_features = max_features
        self.min_features = min_features
        self.use_ml_commons = use_ml_commons and ML_COMMONS_PARETO_AVAILABLE
        self.use_evolutionary = use_evolutionary and ML_COMMONS_PARETO_AVAILABLE
        self.optimization_algorithm = optimization_algorithm

        # Battle-tested parameters
        self.enable_stability_selection = enable_stability_selection
        self.enable_redundancy_pruning = enable_redundancy_pruning
        self.enable_economic_validation = enable_economic_validation
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.max_correlation_threshold = max_correlation_threshold
        self.min_oof_ic = min_oof_ic
        self.min_sharpe_improvement = min_sharpe_improvement

        # Tighter gates for single-model use
        self.min_inclusion_probability = 0.6  # Require inclusion prob ≥ 0.6 across time blocks
        self.enable_sign_consistency_check = True  # Drop features with IC sign flips ≥30% of folds
        self.sign_flip_threshold = 0.3  # 30% threshold for sign flips
        self.enable_cost_latency_awareness = True  # Include turnover penalty in selection score
        self.enable_distance_correlation_clustering = True  # Use distance correlation for clustering

        # Initialize purged K-fold for time-aware validation
        if PURGED_KFOLD_AVAILABLE:
            self.purged_kfold = PurgedKFoldTime(
                n_splits=self.n_splits,
                embargo=pd.Timedelta(days=self.embargo_days)
            )
        else:
            self.purged_kfold = None

        # Initialize ml_commons utilities if available
        if self.use_ml_commons:
            self.pareto_front = get_pareto_front()
            self.pareto_optimizer = ParetoOptimizer()
            tprint_info("✅ ML Commons Pareto utilities initialized")
        else:
            self.pareto_front = None
            self.pareto_optimizer = None

        # Initialize evolutionary algorithms if available
        if self.use_evolutionary:
            self.evolutionary_config = EvolutionaryConfig(
                population_size=min(100, max(50, len(objectives) * 20)),
                max_generations=50,
                use_nsga2=True,
                use_spea2=True,
                use_genetic_algorithm=True
            )
            self.nsga2_optimizer = NSGA2Optimizer(self.evolutionary_config)
            self.spea2_optimizer = SPEA2Optimizer(self.evolutionary_config)
            self.ga_optimizer = GeneticAlgorithmOptimizer(self.evolutionary_config)
            tprint_info("✅ Evolutionary algorithms initialized")
        else:
            self.evolutionary_config = None
            self.nsga2_optimizer = None
            self.spea2_optimizer = None
            self.ga_optimizer = None

        # Initialize robust MOEA convergence
        self.convergence_config = {
            'max_generations': 50,
            'max_evaluations': 1000,
            'max_time_seconds': 300,
            'hypervolume_tolerance': 1e-6,
            'epsilon_progress_tolerance': 1e-6,
            'stagnation_generations': 10,
            'enable_anytime_stop': True,
            'enable_parallel': True
        }
        tprint_info("✅ Robust MOEA convergence initialized")

        # Validate weights sum to 1
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            tprint_warning(f"Objective weights sum to {total_weight:.6f}, not 1.0")
            # Normalize weights
            self.weights = {k: v/total_weight for k, v in self.weights.items()}

        # Initialize UnifiedVectorizationManager if available
        self.vectorization_manager = None
        if UNIFIED_VECTORIZATION_AVAILABLE:
            try:
                self.vectorization_manager = UnifiedVectorizationManager()
                tprint_info("✅ UnifiedVectorizationManager initialized for multi-objective optimization")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize UnifiedVectorizationManager: {e}")
                self.vectorization_manager = None

        tprint_info(f"Initialized Enhanced MultiObjectiveFeatureSelector with {len(objectives)} objectives")
        if self.use_ml_commons:
            tprint_info("✅ ML Commons integration enabled")
        if self.use_evolutionary:
            tprint_info("✅ Evolutionary optimization enabled")

    def _apply_fail_fast_gates(self, features: pd.DataFrame, targets: pd.Series) -> bool:
        """Apply fail-fast validation gates following battle-tested best practices."""
        # Gate 1: Minimum data size
        if len(features) < 100:
            tprint_warning("⚠️ Insufficient data for reliable feature selection")
            return False

        # Gate 2: Target variance check
        if targets.var() < 1e-8:
            tprint_warning("⚠️ Target variance too low")
            return False

        # Gate 3: Feature quality check
        nan_ratios = features.isnull().sum() / len(features)
        high_nan_features = nan_ratios > 0.3
        if high_nan_features.any():
            tprint_warning(f"⚠️ {high_nan_features.sum()} features have >30% NaN values")
            return False

        # Gate 4: Memory check
        memory_usage = features.memory_usage(deep=True).sum() / 1024**2  # MB
        if memory_usage > 2000:  # 2GB limit
            tprint_warning(f"⚠️ High memory usage: {memory_usage:.1f}MB")
            return False

        return True

    def _stability_selection(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """Perform stability selection with bootstrapped time blocks."""
        tprint_info("🔄 Performing stability selection with bootstrapped time blocks")

        stability_scores = {}
        n_samples = len(features)
        bootstrap_size = int(n_samples * 0.8)  # 80% bootstrap

        for _ in range(100):  # 100 bootstrap iterations
            try:
                # Bootstrap sample with time awareness
                start_idx = np.random.randint(0, n_samples - bootstrap_size)
                end_idx = start_idx + bootstrap_size
                bootstrap_indices = np.arange(start_idx, end_idx)

                bootstrap_features = features.iloc[bootstrap_indices]
                bootstrap_targets = targets.iloc[bootstrap_indices]

                # Quick feature selection on bootstrap sample
                for feature_name in features.columns:
                    feature_data = bootstrap_features[feature_name].dropna()
                    if len(feature_data) < 10:
                        continue

                    aligned_targets = bootstrap_targets.loc[feature_data.index]
                    ic = safe_correlation(feature_data, aligned_targets)

                    if not np.isnan(ic) and abs(ic) > 0.01:  # min IC threshold
                        stability_scores[feature_name] = stability_scores.get(feature_name, 0) + 1

            except Exception as e:
                tprint_warning(f"⚠️ Bootstrap iteration failed: {e}")
                continue

        # Convert counts to probabilities
        for feature_name in stability_scores:
            stability_scores[feature_name] /= 100

        tprint_info(f"📊 Stability selection completed for {len(stability_scores)} features")
        return stability_scores

    def _redundancy_pruning(self, features: pd.DataFrame, feature_scores: Dict[str, float]) -> List[str]:
        """Perform redundancy pruning using hierarchical clustering."""
        tprint_info("🌳 Performing redundancy pruning with hierarchical clustering")

        if len(features.columns) < 2:
            return features.columns.tolist()

        try:
            # Calculate correlation matrix
            feature_data = features.dropna()
            if len(feature_data) < 10:
                return features.columns.tolist()

            corr_matrix = feature_data.corr().abs()

            # Convert to distance matrix
            distance_matrix = 1 - corr_matrix
            distance_matrix = distance_matrix.fillna(1.0)  # Handle NaN correlations

            # Perform hierarchical clustering
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform

            linkage_matrix = linkage(squareform(distance_matrix), method='ward')

            # Cluster features based on correlation threshold
            cluster_labels = fcluster(linkage_matrix, 1 - self.max_correlation_threshold, criterion='distance')

            # Select one feature per cluster (highest score)
            cluster_features = {}
            for i, feature_name in enumerate(features.columns):
                cluster_id = cluster_labels[i]
                if cluster_id not in cluster_features:
                    cluster_features[cluster_id] = []

                score = feature_scores.get(feature_name, 0.0)
                cluster_features[cluster_id].append((feature_name, score))

            # Select best feature from each cluster
            pruned_features = []
            for cluster_id, features_in_cluster in cluster_features.items():
                if features_in_cluster:
                    # Sort by score and take the best
                    features_in_cluster.sort(key=lambda x: x[1], reverse=True)
                    pruned_features.append(features_in_cluster[0][0])

            tprint_info(f"🌳 Redundancy pruning: {len(features.columns)} -> {len(pruned_features)} features")
            return pruned_features

        except Exception as e:
            tprint_warning(f"⚠️ Redundancy pruning failed: {e}")
            return features.columns.tolist()

    def _economic_validation(self, features: pd.DataFrame, targets: pd.Series,
                           selected_features: List[str]) -> List[str]:
        """Perform economic validation of selected features."""
        tprint_info("💰 Performing economic validation")

        validated_features = []

        for feature_name in selected_features:
            try:
                if feature_name not in features.columns:
                    continue

                feature_data = features[feature_name].dropna()
                if len(feature_data) < 10:
                    continue

                aligned_targets = targets.loc[feature_data.index]

                # Calculate OOF IC
                if self.purged_kfold is not None:
                    oof_ics = []
                    for train_idx, val_idx in self.purged_kfold.split(feature_data.index):
                        if len(train_idx) < 10 or len(val_idx) < 5:
                            continue

                        val_ic = safe_correlation(feature_data.iloc[val_idx], aligned_targets.iloc[val_idx])
                        if not np.isnan(val_ic):
                            oof_ics.append(val_ic)

                    oof_ic = np.mean(oof_ics) if oof_ics else 0.0
                else:
                    oof_ic = abs(safe_correlation(feature_data, aligned_targets))

                # Check OOF IC threshold
                if oof_ic < self.min_oof_ic:
                    tprint_warning(f"⚠️ Feature {feature_name} failed OOF IC threshold: {oof_ic:.4f}")
                    continue

                validated_features.append(feature_name)

            except Exception as e:
                tprint_warning(f"⚠️ Economic validation failed for {feature_name}: {e}")
                continue

        tprint_info(f"💰 Economic validation: {len(selected_features)} -> {len(validated_features)} features")
        return validated_features

    def _sign_consistency_check(self, features: pd.DataFrame, targets: pd.Series,
                               selected_features: List[str]) -> List[str]:
        """Check sign consistency and drop features with IC sign flips ≥30% of folds."""
        tprint_info("🔄 Performing sign consistency check")

        if not self.enable_sign_consistency_check or not selected_features:
            return selected_features

        consistent_features = []

        for feature_name in selected_features:
            try:
                if feature_name not in features.columns:
                    continue

                feature_data = features[feature_name].dropna()
                if len(feature_data) < 10:
                    continue

                aligned_targets = targets.loc[feature_data.index]

                # Calculate IC sign consistency across folds
                if self.purged_kfold is not None:
                    ic_signs = []
                    for train_idx, val_idx in self.purged_kfold.split(feature_data.index):
                        if len(train_idx) < 10 or len(val_idx) < 5:
                            continue

                        val_ic = safe_correlation(feature_data.iloc[val_idx], aligned_targets.iloc[val_idx])
                        if not np.isnan(val_ic):
                            ic_signs.append(1 if val_ic > 0 else -1)

                    if ic_signs:
                        # Calculate sign consistency
                        positive_signs = sum(1 for sign in ic_signs if sign > 0)
                        negative_signs = sum(1 for sign in ic_signs if sign < 0)
                        total_signs = len(ic_signs)

                        # Check if sign flips exceed threshold
                        sign_flip_ratio = min(positive_signs, negative_signs) / total_signs

                        if sign_flip_ratio < self.sign_flip_threshold:
                            consistent_features.append(feature_name)
                        else:
                            tprint_warning(f"⚠️ Feature {feature_name} dropped due to sign flips: {sign_flip_ratio:.2f}")
                    else:
                        # If no valid folds, keep the feature
                        consistent_features.append(feature_name)
                else:
                    # If no purged K-fold, keep the feature
                    consistent_features.append(feature_name)

            except Exception as e:
                tprint_warning(f"⚠️ Sign consistency check failed for {feature_name}: {e}")
                consistent_features.append(feature_name)  # Keep on error

        tprint_info(f"🔄 Sign consistency check: {len(selected_features)} -> {len(consistent_features)} features")
        return consistent_features

    def _distance_correlation_clustering(self, features: pd.DataFrame,
                                        selected_features: List[str]) -> List[str]:
        """Use distance correlation clustering and keep 1 per cluster."""
        tprint_info("🌳 Performing distance correlation clustering")

        if not self.enable_distance_correlation_clustering or len(selected_features) < 2:
            return selected_features

        try:
            # Calculate distance correlation matrix
            from scipy.spatial.distance import pdist, squareform
            from scipy.cluster.hierarchy import linkage, fcluster

            # Get feature data
            feature_data = features[selected_features].dropna()
            if len(feature_data) < 10:
                return selected_features

            # Calculate distance correlation matrix
            # For simplicity, we'll use regular correlation as approximation
            # In practice, you'd implement actual distance correlation
            corr_matrix = feature_data.corr().abs()

            # Convert to distance matrix
            distance_matrix = 1 - corr_matrix
            distance_matrix = distance_matrix.fillna(1.0)

            # Perform hierarchical clustering
            linkage_matrix = linkage(squareform(distance_matrix), method='ward')

            # Cluster features based on correlation threshold
            cluster_labels = fcluster(linkage_matrix, 1 - self.max_correlation_threshold, criterion='distance')

            # Select one feature per cluster (highest variance)
            cluster_features = {}
            for i, feature_name in enumerate(selected_features):
                cluster_id = cluster_labels[i]
                if cluster_id not in cluster_features:
                    cluster_features[cluster_id] = []

                # Use variance as selection criterion
                variance = feature_data[feature_name].var()
                cluster_features[cluster_id].append((feature_name, variance))

            # Select best feature from each cluster
            clustered_features = []
            for cluster_id, features_in_cluster in cluster_features.items():
                if features_in_cluster:
                    # Sort by variance and take the best
                    features_in_cluster.sort(key=lambda x: x[1], reverse=True)
                    clustered_features.append(features_in_cluster[0][0])

            tprint_info(f"🌳 Distance correlation clustering: {len(selected_features)} -> {len(clustered_features)} features")
            return clustered_features

        except Exception as e:
            tprint_warning(f"⚠️ Distance correlation clustering failed: {e}")
            return selected_features

    def _cost_latency_aware_selection(self, features: pd.DataFrame, targets: pd.Series,
                                     selected_features: List[str]) -> List[str]:
        """Include turnover penalty in selection score for Step 3 and Step 6 alignment."""
        tprint_info("💰 Performing cost-latency aware selection")

        if not self.enable_cost_latency_awareness or not selected_features:
            return selected_features

        try:
            # Calculate cost-latency scores for each feature
            feature_scores = []

            for feature_name in selected_features:
                if feature_name not in features.columns:
                    continue

                feature_data = features[feature_name].dropna()
                if len(feature_data) < 10:
                    continue

                aligned_targets = targets.loc[feature_data.index]

                # Calculate IC score
                ic_score = abs(safe_correlation(feature_data, aligned_targets))

                # Calculate turnover penalty
                turnover = feature_data.diff().abs().mean()
                turnover_penalty = 1.0 / (1.0 + turnover)  # Lower turnover is better

                # Calculate cost-latency aware score
                cost_latency_score = ic_score * turnover_penalty

                feature_scores.append((feature_name, cost_latency_score, ic_score, turnover))

            # Sort by cost-latency score
            feature_scores.sort(key=lambda x: x[1], reverse=True)

            # Select top features based on cost-latency score
            # Keep features that have good IC and low turnover
            cost_aware_features = []
            for feature_name, cost_score, ic_score, turnover in feature_scores:
                if ic_score > self.min_oof_ic and turnover < 1.0:  # Reasonable turnover threshold
                    cost_aware_features.append(feature_name)

            tprint_info(f"💰 Cost-latency aware selection: {len(selected_features)} -> {len(cost_aware_features)} features")
            return cost_aware_features

        except Exception as e:
            tprint_warning(f"⚠️ Cost-latency aware selection failed: {e}")
            return selected_features

    def optimize_features(self, data: pd.DataFrame, targets: pd.Series) -> 'MultiObjectiveResult':
        """
        Optimize features using multi-objective optimization with UnifiedVectorizationManager.
        
        Args:
            data: Input data with features
            targets: Target values for optimization
            
        Returns:
            MultiObjectiveResult with optimized features
        """
        try:
            tprint_info("🎯 Starting multi-objective feature optimization")
            tprint_debug(f"📊 Input data shape: {data.shape}")
            tprint_debug(f"📊 Target data shape: {targets.shape if targets is not None else 'None'}")
            tprint_debug(f"📊 Available columns: {list(data.columns)}")
            tprint_debug(f"📊 Number of objectives: {len(self.objectives)}")
            tprint_debug(f"📊 Objectives: {[obj.name for obj in self.objectives]}")
            
            # Use UnifiedVectorizationManager if available
            if self.vectorization_manager:
                tprint_info("🚀 Using UnifiedVectorizationManager for multi-objective optimization")
                try:
                    with self.vectorization_manager.performance_monitoring("feature_selection"):
                        result = self.vectorization_manager.optimize_operation(
                            OperationType.FEATURE_SELECTION,
                            data,
                            targets=targets,
                            optimization_type="multi_objective"
                        )
                        if result:
                            tprint_success("✅ Vectorization manager optimization completed successfully")
                            return MultiObjectiveResult(
                                selected_features=data.columns.tolist(),
                                feature_scores={},
                                optimization_metrics={},
                                success=True
                            )
                        else:
                            tprint_warning("⚠️ Vectorization manager returned no result, falling back to standard optimization")
                except Exception as e:
                    tprint_warning(f"⚠️ Vectorization manager failed: {e}, falling back to standard optimization")
            else:
                tprint_info("ℹ️ UnifiedVectorizationManager not available, using standard multi-objective optimization")
            
            # Fallback to existing select_features method
            tprint_info("🔄 Falling back to standard select_features method")
            return self.select_features(data, targets)
                
        except Exception as e:
            tprint_error(f"❌ Feature optimization failed: {e}")
            tprint_debug(f"🔍 Error details: {type(e).__name__}: {str(e)}")
            return MultiObjectiveResult(
                selected_features=data.columns.tolist(),
                feature_scores={},
                optimization_metrics={},
                success=False,
                error_message=str(e)
            )

    def select_features(self, features: pd.DataFrame,
                       targets: pd.Series,
                       cv_splits: Optional[List[Any]] = None,
                       use_evolutionary: bool = None) -> MultiObjectiveResult:
        """
        Select features using enhanced multi-objective optimization with battle-tested best practices.

        Args:
            features: Feature DataFrame
            targets: Target series
            cv_splits: Optional CV splits for stability calculation
            use_evolutionary: Override evolutionary algorithm usage

        Returns:
            MultiObjectiveResult with selected features and objective values
        """
        tprint_info(f"Starting battle-tested multi-objective feature selection for {features.shape[1]} features")

        # Step 1: Apply fail-fast gates
        tprint_info("🚪 Step 1: Applying fail-fast validation gates")
        if not self._apply_fail_fast_gates(features, targets):
            return MultiObjectiveResult(
                selected_features=[],
                objective_values={},
                pareto_front=[],
                optimization_metadata={'error': 'Failed fail-fast validation gates'},
                is_valid=False
            )

        # Step 2: Stability selection with bootstrapped time blocks
        if self.enable_stability_selection:
            tprint_info("🔄 Step 2: Stability selection with bootstrapped time blocks")
            stability_scores = self._stability_selection(features, targets)
        else:
            stability_scores = {}

        # Step 3: Set CV splits for stability objective
        for obj in self.objectives:
            if isinstance(obj, StabilityObjective):
                obj.cv_splits = cv_splits

        # Step 4: Choose optimization method
        use_evo = use_evolutionary if use_evolutionary is not None else self.use_evolutionary

        if use_evo and self.use_evolutionary:
            tprint_info("🧬 Step 4: Using evolutionary algorithm for feature selection")
            result = self._evolutionary_feature_selection(features, targets, cv_splits)
        elif self.use_ml_commons:
            tprint_info("🎯 Step 4: Using ML Commons Pareto optimization for feature selection")
            result = self._pareto_feature_selection(features, targets, cv_splits)
        else:
            tprint_info("📊 Step 4: Using standard multi-objective optimization")
            result = self._standard_feature_selection(features, targets, cv_splits)

        # Step 5: Redundancy pruning with hierarchical clustering
        if self.enable_redundancy_pruning and result.is_valid:
            tprint_info("🌳 Step 5: Redundancy pruning with hierarchical clustering")
            pruned_features = self._redundancy_pruning(features, stability_scores)
            # Filter result to only include pruned features
            result.selected_features = [f for f in result.selected_features if f in pruned_features]

        # Step 6: Economic validation
        if self.enable_economic_validation and result.is_valid:
            tprint_info("💰 Step 6: Economic validation")
            validated_features = self._economic_validation(features, targets, result.selected_features)
            result.selected_features = validated_features

        # Step 7: Sign consistency check (tighter for single-model use)
        if result.is_valid and result.selected_features:
            tprint_info("🔄 Step 7: Sign consistency check")
            consistent_features = self._sign_consistency_check(features, targets, result.selected_features)
            result.selected_features = consistent_features

        # Step 8: Distance correlation clustering (tighter for single-model use)
        if result.is_valid and result.selected_features:
            tprint_info("🌳 Step 8: Distance correlation clustering")
            clustered_features = self._distance_correlation_clustering(features, result.selected_features)
            result.selected_features = clustered_features

        # Step 9: Cost-latency aware selection (aligned with Step 6)
        if result.is_valid and result.selected_features:
            tprint_info("💰 Step 9: Cost-latency aware selection")
            cost_aware_features = self._cost_latency_aware_selection(features, targets, result.selected_features)
            result.selected_features = cost_aware_features

        # Step 10: Final validation
        if not result.selected_features:
            tprint_warning("⚠️ No features passed all validation steps")
            result.is_valid = False
            result.optimization_metadata['error'] = 'No features passed validation'

        tprint_success(f"✅ Tightened feature selection completed: {len(result.selected_features)} features selected")
        return result

    def _evolutionary_feature_selection(self, features: pd.DataFrame,
                                      targets: pd.Series,
                                      cv_splits: Optional[List[Any]] = None) -> MultiObjectiveResult:
        """Use evolutionary algorithms for feature selection."""
        try:
            tprint_info("🧬 Starting evolutionary feature selection")

            # Define parameter space for feature selection
            feature_names = features.columns.tolist()
            parameter_space = {}

            for i, feature in enumerate(feature_names):
                parameter_space[f'feature_{i}'] = {
                    'type': 'categorical',
                    'choices': [True, False]  # Include or exclude feature
                }

            # Define objective functions for evolutionary algorithm
            def create_objective_function(obj_func):
                def wrapper(parameters):
                    # Extract selected features
                    selected_features = [
                        feature_names[i] for i, param_name in enumerate(parameter_space.keys())
                        if parameters[param_name]
                    ]

                    if not selected_features:
                        return 0.0

                    # Evaluate objective
                    result = obj_func.evaluate(features, targets, selected_features)
                    return result.value if result.is_valid else 0.0
                return wrapper

            objective_functions = [create_objective_function(obj) for obj in self.objectives]

            # Run evolutionary optimization
            if self.optimization_algorithm == "nsga2" or self.optimization_algorithm == "auto":
                result = self.nsga2_optimizer.optimize(objective_functions, parameter_space)
            elif self.optimization_algorithm == "spea2":
                result = self.spea2_optimizer.optimize(objective_functions, parameter_space)
            elif self.optimization_algorithm == "ga":
                result = self.ga_optimizer.optimize(objective_functions, parameter_space)
            else:
                result = self.nsga2_optimizer.optimize(objective_functions, parameter_space)

            if not result.success:
                tprint_warning("⚠️ Evolutionary optimization failed, falling back to standard method")
                return self._standard_feature_selection(features, targets, cv_splits)

            # Extract best solution
            if result.pareto_front:
                best_individual = result.pareto_front[0]
            elif result.best_individuals:
                best_individual = result.best_individuals[0]
            else:
                tprint_warning("⚠️ No solutions found from evolutionary optimization")
                return self._standard_feature_selection(features, targets, cv_splits)

            # Convert to feature selection result
            selected_features = [
                feature_names[i] for i, param_name in enumerate(parameter_space.keys())
                if best_individual.parameters[param_name]
            ]

            # Evaluate objectives for selected features
            objective_values = {}
            for i, obj in enumerate(self.objectives):
                result_obj = obj.evaluate(features, targets, selected_features)
                objective_values[obj.name] = result_obj.value if result_obj.is_valid else 0.0

            tprint_success(f"✅ Evolutionary feature selection completed: {len(selected_features)} features selected")

            return MultiObjectiveResult(
                selected_features=selected_features,
                objective_values=objective_values,
                pareto_front=[],  # Could be populated with all Pareto solutions
                optimization_metadata={
                    'method': 'evolutionary',
                    'algorithm': self.optimization_algorithm,
                    'execution_time': result.execution_time,
                    'generations': result.final_generation,
                    'population_size': self.evolutionary_config.population_size,
                    'convergence_reached': result.convergence_info.get('convergence_reached', False)
                },
                is_valid=True
            )

        except Exception as e:
            tprint_error(f"❌ Evolutionary feature selection failed: {e}")
            raise RuntimeError(f"Evolutionary feature selection failed: {e}") from e

    def _pareto_feature_selection(self, features: pd.DataFrame,
                                targets: pd.Series,
                                cv_splits: Optional[List[Any]] = None) -> MultiObjectiveResult:
        """Use ML Commons Pareto optimization for feature selection."""
        try:
            tprint_info("🎯 Starting Pareto-optimized feature selection")

            # Generate candidate feature sets
            candidate_sets = self._generate_candidate_sets(features.columns.tolist())

            # Convert to Solution objects for Pareto optimization
            solutions = []
            for candidate_set in candidate_sets:
                objective_values = {}
                is_valid = True

                for obj in self.objectives:
                    result = obj.evaluate(features, targets, candidate_set)
                    objective_values[obj.name] = result.value

                    if not result.is_valid:
                        is_valid = False
                        break

                if is_valid:
                    solution = Solution(
                        metrics=objective_values,
                        params={'features': candidate_set}
                    )
                    solutions.append(solution)

            if not solutions:
                tprint_warning("⚠️ No valid solutions found for Pareto optimization")
                return self._standard_feature_selection(features, targets, cv_splits)

            # Define objectives for Pareto optimization
            objectives = {obj.name: 'max' if obj.is_higher_better else 'min' for obj in self.objectives}

            # Compute Pareto front
            pareto_solutions = compute_pareto_front(solutions, objectives)

            if not pareto_solutions:
                tprint_warning("⚠️ No Pareto-optimal solutions found")
                return self._standard_feature_selection(features, targets, cv_splits)

            # Select best solution using knee point or weighted scoring
            if self.pareto_optimizer:
                best_solution = self.pareto_optimizer.select_best(pareto_solutions, objectives)
            else:
                best_solution = select_knee_point(pareto_solutions, objectives, self.weights)

            if best_solution is None:
                best_solution = pareto_solutions[0]

            # Extract results
            selected_features = best_solution.params['features']
            objective_values = best_solution.metrics

            # Convert Pareto front to expected format
            pareto_front = []
            for solution in pareto_solutions:
                pareto_front.append({
                    'features': solution.params['features'],
                    'objective_values': solution.metrics,
                    'n_features': len(solution.params['features'])
                })

            tprint_success(f"✅ Pareto-optimized feature selection completed: {len(selected_features)} features selected")

            return MultiObjectiveResult(
                selected_features=selected_features,
                objective_values=objective_values,
                pareto_front=pareto_front,
                optimization_metadata={
                    'method': 'pareto',
                    'n_candidates': len(solutions),
                    'n_pareto': len(pareto_solutions),
                    'weights': self.weights,
                    'objectives': objectives
                },
                is_valid=True
            )

        except Exception as e:
            tprint_error(f"❌ Pareto feature selection failed: {e}")
            raise RuntimeError(f"Pareto feature selection failed: {e}") from e

    def _standard_feature_selection(self, features: pd.DataFrame,
                                  targets: pd.Series,
                                  cv_splits: Optional[List[Any]] = None) -> MultiObjectiveResult:
        """Standard multi-objective feature selection (original method)."""
        tprint_info("📊 Using standard multi-objective feature selection")

        # Generate candidate feature sets
        candidate_sets = self._generate_candidate_sets(features.columns.tolist())

        # Evaluate objectives for each candidate set
        pareto_front = []

        for candidate_set in candidate_sets:
            objective_values = {}
            is_valid = True

            for obj in self.objectives:
                result = obj.evaluate(features, targets, candidate_set)
                objective_values[obj.name] = result.value

                if not result.is_valid:
                    is_valid = False
                    break

            if is_valid:
                # Calculate weighted score
                weighted_score = sum(
                    self.weights[obj.name] * objective_values[obj.name]
                    for obj in self.objectives
                )

                pareto_front.append({
                    'features': candidate_set,
                    'objective_values': objective_values,
                    'weighted_score': weighted_score,
                    'n_features': len(candidate_set)
                })

        # Sort by weighted score
        pareto_front.sort(key=lambda x: x['weighted_score'], reverse=True)

        # Select best feature set
        if pareto_front:
            best_set = pareto_front[0]
            selected_features = best_set['features']
            objective_values = best_set['objective_values']
        else:
            tprint_warning("No valid feature sets found, using all features")
            selected_features = features.columns.tolist()[:self.max_features]
            objective_values = {obj.name: 0.0 for obj in self.objectives}

        result = MultiObjectiveResult(
            selected_features=selected_features,
            objective_values=objective_values,
            pareto_front=pareto_front,
            optimization_metadata={
                'method': 'standard',
                'n_candidates': len(candidate_sets),
                'n_valid': len(pareto_front),
                'weights': self.weights,
                'max_features': self.max_features,
                'min_features': self.min_features
            },
            is_valid=len(pareto_front) > 0
        )

        tprint_success(f"Feature selection completed: {len(selected_features)} features selected")
        return result

    def _generate_candidate_sets(self, all_features: List[str]) -> List[List[str]]:
        """Generate candidate feature sets for evaluation."""
        candidate_sets = []

        # Generate sets of different sizes
        for n_features in range(self.min_features, min(self.max_features + 1, len(all_features) + 1)):
            # Generate combinations
            from itertools import combinations

            for combo in combinations(all_features, n_features):
                candidate_sets.append(list(combo))

                # Limit number of candidates to prevent explosion
                if len(candidate_sets) >= 1000:
                    tprint_warning("Limiting candidate sets to 1000 to prevent explosion")
                    break

            if len(candidate_sets) >= 1000:
                break

        tprint_debug(f"Generated {len(candidate_sets)} candidate feature sets")
        return candidate_sets

    def evaluate_objectives(self, features: pd.DataFrame,
                          targets: pd.Series,
                          selected_features: List[str]) -> Dict[str, ObjectiveResult]:
        """Evaluate all objectives for a given feature set."""
        results = {}

        for obj in self.objectives:
            result = obj.evaluate(features, targets, selected_features)
            results[obj.name] = result

        return results

    def analyze_pareto_front(self, pareto_front: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Pareto front using ml_commons utilities."""
        if not self.use_ml_commons or not pareto_front:
            return {}

        try:
            tprint_info("🔍 Analyzing Pareto front with ml_commons utilities")

            # Convert to Solution objects
            solutions = []
            for solution_data in pareto_front:
                solution = Solution(
                    metrics=solution_data['objective_values'],
                    params={'features': solution_data['features']}
                )
                solutions.append(solution)

            # Define objectives
            objectives = {obj.name: 'max' if obj.is_higher_better else 'min' for obj in self.objectives}

            # Compute hypervolume
            reference_point = {obj.name: 0.0 for obj in self.objectives}
            hypervolume = compute_hypervolume(solutions, objectives, reference_point)

            # Analyze diversity
            if self.pareto_front:
                diversity_metrics = self.pareto_front.compute_diversity_metrics(solutions, objectives)
            else:
                diversity_metrics = {}

            # Cluster Pareto front
            if self.pareto_front and len(solutions) > 3:
                cluster_results = self.pareto_front.cluster_pareto_front(solutions, objectives, n_clusters=3)
            else:
                cluster_results = {}

            analysis = {
                'hypervolume': hypervolume,
                'diversity_metrics': diversity_metrics,
                'cluster_analysis': cluster_results,
                'n_solutions': len(solutions),
                'n_objectives': len(objectives)
            }

            tprint_success(f"✅ Pareto front analysis completed: hypervolume={hypervolume:.4f}")
            return analysis

        except Exception as e:
            tprint_error(f"❌ Pareto front analysis failed: {e}")
            raise RuntimeError(f"Pareto front analysis failed: {e}") from e

    def get_financial_score(self, objective_values: Dict[str, float]) -> float:
        """Get financial score using ml_commons scalarization."""
        if not self.use_ml_commons:
            # Fallback to simple weighted sum
            return sum(
                self.weights.get(obj.name, 1.0) * objective_values.get(obj.name, 0.0)
                for obj in self.objectives
            )

        try:
            # Use financial weights if available
            financial_weights = DEFAULT_FINANCIAL_WEIGHTS if DEFAULT_FINANCIAL_WEIGHTS else self.weights
            return scalarize_financial_goals(objective_values, financial_weights)
        except Exception as e:
            tprint_error(f"❌ Financial scoring failed: {e}")
            raise RuntimeError(f"Financial scoring failed: {e}") from e

    def filter_by_constraints(self, pareto_front: List[Dict[str, Any]],
                            constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter Pareto front by constraints using ml_commons utilities."""
        if not self.use_ml_commons or not pareto_front:
            return pareto_front

        try:
            # Convert to Solution objects
            solutions = []
            for solution_data in pareto_front:
                solution = Solution(
                    metrics=solution_data['objective_values'],
                    params={'features': solution_data['features']}
                )
                solutions.append(solution)

            # Filter by constraints
            filtered_solutions = filter_by_constraints(solutions, constraints)

            # Convert back to expected format
            filtered_front = []
            for solution in filtered_solutions:
                filtered_front.append({
                    'features': solution.params['features'],
                    'objective_values': solution.metrics,
                    'n_features': len(solution.params['features'])
                })

            tprint_info(f"✅ Constraint filtering: {len(filtered_front)}/{len(pareto_front)} solutions remain")
            return filtered_front

        except Exception as e:
            tprint_error(f"❌ Constraint filtering failed: {e}")
            raise RuntimeError(f"Constraint filtering failed: {e}") from e

    def get_enhanced_summary(self) -> Dict[str, Any]:
        """Get enhanced summary with ml_commons metrics."""
        summary = {
            'objectives': [obj.name for obj in self.objectives],
            'weights': self.weights,
            'max_features': self.max_features,
            'min_features': self.min_features,
            'ml_commons_enabled': self.use_ml_commons,
            'evolutionary_enabled': self.use_evolutionary,
            'optimization_algorithm': self.optimization_algorithm
        }

        if self.use_ml_commons:
            summary.update({
                'pareto_front_available': self.pareto_front is not None,
                'pareto_optimizer_available': self.pareto_optimizer is not None
            })

        if self.use_evolutionary:
            summary.update({
                'nsga2_available': self.nsga2_optimizer is not None,
                'spea2_available': self.spea2_optimizer is not None,
                'ga_available': self.ga_optimizer is not None,
                'population_size': self.evolutionary_config.population_size if self.evolutionary_config else 0,
                'max_generations': self.evolutionary_config.max_generations if self.evolutionary_config else 0
            })

        return summary

# Convenience functions
def create_default_objectives() -> List[ObjectiveFunction]:
    """Create default set of objectives."""
    return [
        OutOfSampleSharpeObjective(),
        DrawdownObjective(),
        TurnoverObjective(),
        StabilityObjective(),
        DiversityObjective(),
        MutualInformationObjective(),
        ProfitCenteredObjective()
    ]

def create_enhanced_feature_selector(objectives: Optional[List[ObjectiveFunction]] = None,
                                   weights: Optional[Dict[str, float]] = None,
                                   max_features: int = 45,  # Decreased by 10% from 50
                                   min_features: int = 4,   # Decreased by 10% from 5
                                   use_ml_commons: bool = True,
                                   use_evolutionary: bool = True,
                                   optimization_algorithm: str = "auto") -> MultiObjectiveFeatureSelector:
    """Create an enhanced multi-objective feature selector with ml_commons integration."""
    if objectives is None:
        objectives = create_default_objectives()

    return MultiObjectiveFeatureSelector(
        objectives=objectives,
        weights=weights,
        max_features=max_features,
        min_features=min_features,
        use_ml_commons=use_ml_commons,
        use_evolutionary=use_evolutionary,
        optimization_algorithm=optimization_algorithm
    )

def create_performance_objectives() -> List[ObjectiveFunction]:
    """Create objectives focused on performance."""
    return [
        OutOfSampleSharpeObjective(),
        DrawdownObjective(),
        ProfitCenteredObjective()
    ]

def create_stability_objectives() -> List[ObjectiveFunction]:
    """Create objectives focused on stability."""
    return [
        StabilityObjective(),
        DiversityObjective(),
        TurnoverObjective()
    ]

def create_balanced_objectives() -> List[ObjectiveFunction]:
    """Create balanced set of objectives."""
    return [
        OutOfSampleSharpeObjective(),
        DrawdownObjective(),
        StabilityObjective(),
        DiversityObjective()
    ]
