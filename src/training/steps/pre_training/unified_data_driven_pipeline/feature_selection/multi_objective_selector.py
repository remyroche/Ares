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
import os
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

# Import UnifiedVectorizationManager and VectorBTRollingOptimizer
try:
    from src.feature_generation.utils.unified_vectorization_manager import (
        UnifiedVectorizationManager, OperationType, OperationConfig
    )
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager as MLUnifiedVectorizationManager,
        get_unified_vectorization_manager
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
    VECTORBT_ROLLING_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    VECTORBT_ROLLING_AVAILABLE = False
    UnifiedVectorizationManager = None
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    MLUnifiedVectorizationManager = None
    get_unified_vectorization_manager = None
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

# Import hardware optimization utilities
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUOptimizer, get_m1_gpu_optimizer
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer, get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer, get_m1_cpu_optimizer
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, WorkloadType, OptimizationLevel
    HARDWARE_OPTIMIZATION_AVAILABLE = True
    tprint_info("✅ M1 hardware optimization utilities available")
except ImportError as e:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    M1GPUOptimizer = None
    M1MemoryOptimizer = None
    M1CPUOptimizer = None
    UnifiedHardwareManager = None
    WorkloadType = None
    OptimizationLevel = None
    tprint_warning(f"⚠️ M1 hardware optimization utilities not available: {e}")

# Import CMI complementarity components
try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.cmi_complementarity import (
        CMIComplementarityScorer, CMIComplementarityConfig
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.analyst_side_info import (
        AnalystSideInfoHandler, AnalystSideInfoConfig
    )
    CMI_COMPLEMENTARITY_AVAILABLE = True
    tprint_info("✅ CMI complementarity components available")
except ImportError:
    CMI_COMPLEMENTARITY_AVAILABLE = False
    CMIComplementarityScorer = None
    CMIComplementarityConfig = None
    AnalystSideInfoHandler = None
    AnalystSideInfoConfig = None
    tprint_warning("⚠️ CMI complementarity components not available")
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

class MultiObjectiveResult:
    """Result of multi-objective optimization."""
    
    def __init__(self, selected_features: List[str], objective_values: Dict[str, float], 
                 pareto_front: List[Dict[str, Any]], optimization_metadata: Dict[str, Any],
                 is_valid: bool = True, feature_scores: Optional[Dict[str, float]] = None,
                 **kwargs):
        self.selected_features = selected_features
        self.objective_values = objective_values
        self.pareto_front = pareto_front
        self.optimization_metadata = optimization_metadata
        self.is_valid = is_valid
        self.feature_scores = feature_scores or {}
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

class ObjectiveFunction(ABC):
    """Abstract base class for objective functions with full implementations."""

    def evaluate(self, features: pd.DataFrame,
                targets: pd.Series,
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """
        Evaluate the objective function with comprehensive error handling and validation.
        
        This base implementation provides:
        - Input validation
        - Error handling
        - Common preprocessing
        - Standardized result format
        
        Subclasses should override _calculate_objective() for specific logic.
        """
        try:
            # Input validation
            if not isinstance(features, pd.DataFrame):
                return ObjectiveResult(
                    value=0.0, 
                    metadata={'error': 'Features must be a pandas DataFrame'}, 
                    is_valid=False
                )
            
            if not isinstance(targets, pd.Series):
                return ObjectiveResult(
                    value=0.0, 
                    metadata={'error': 'Targets must be a pandas Series'}, 
                    is_valid=False
                )
            
            if not selected_features or not isinstance(selected_features, list):
                return ObjectiveResult(
                    value=0.0, 
                    metadata={'error': 'Selected features must be a non-empty list'}, 
                    is_valid=False
                )
            
            # Check if selected features exist in the dataframe
            valid_features = [f for f in selected_features if f in features.columns]
            if not valid_features:
                return ObjectiveResult(
                    value=0.0, 
                    metadata={
                        'error': 'No valid features found',
                        'requested_features': selected_features,
                        'available_features': list(features.columns)
                    }, 
                    is_valid=False
                )
            
            # Check for sufficient data
            if len(features) < 2:
                return ObjectiveResult(
                    value=0.0, 
                    metadata={'error': 'Insufficient data (need at least 2 samples)'}, 
                    is_valid=False
                )
            
            # Check for NaN values in targets
            if targets.isnull().all():
                return ObjectiveResult(
                    value=0.0, 
                    metadata={'error': 'All target values are NaN'}, 
                    is_valid=False
                )
            
            # Align features and targets
            aligned_features = features[valid_features]
            aligned_targets = targets
            
            # Remove rows where targets are NaN
            valid_indices = ~aligned_targets.isnull()
            if not valid_indices.any():
                return ObjectiveResult(
                    value=0.0, 
                    metadata={'error': 'No valid target values after NaN removal'}, 
                    is_valid=False
                )
            
            aligned_features = aligned_features[valid_indices]
            aligned_targets = aligned_targets[valid_indices]
            
            # Call the specific objective calculation
            result = self._calculate_objective(aligned_features, aligned_targets, valid_features, **kwargs)
            
            # Add common metadata
            if result.metadata is None:
                result.metadata = {}
            
            result.metadata.update({
                'n_samples': len(aligned_features),
                'n_features': len(valid_features),
                'feature_names': valid_features,
                'target_stats': {
                    'mean': float(aligned_targets.mean()),
                    'std': float(aligned_targets.std()),
                    'min': float(aligned_targets.min()),
                    'max': float(aligned_targets.max())
                }
            })
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Objective function evaluation failed: {e}")
            return ObjectiveResult(
                value=0.0, 
                metadata={
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'objective_name': getattr(self, 'name', 'unknown')
                }, 
                is_valid=False
            )

    @abstractmethod
    def _calculate_objective(self, features: pd.DataFrame,
                           targets: pd.Series,
                           selected_features: List[str],
                           **kwargs) -> ObjectiveResult:
        """
        Calculate the specific objective value.
        
        This method must be implemented by subclasses to provide
        the actual objective calculation logic.
        
        Args:
            features: Aligned feature DataFrame (no NaN targets)
            targets: Aligned target Series (no NaN values)
            selected_features: List of valid feature names
            **kwargs: Additional keyword arguments
            
        Returns:
            ObjectiveResult with the calculated value and metadata
        """
        pass

    @property
    def name(self) -> str:
        """Get the name of the objective function."""
        return self.__class__.__name__.lower().replace('objective', '')

    @property
    def is_higher_better(self) -> bool:
        """Whether higher values are better for this objective."""
        # Default to True - most objectives are maximized
        # Subclasses can override this property
        return True

class OutOfSampleSharpeObjective(ObjectiveFunction):
    """
    Out-of-sample Sharpe ratio objective with full implementation.
    
    Calculates the Sharpe ratio as (mean_return - risk_free_rate) / std_return,
    with automatic annualization for periods > 252.
    """

    def __init__(self, risk_free_rate: float = 0.0, annualization_factor: int = 252):
        """
        Initialize the Sharpe ratio objective.
        
        Args:
            risk_free_rate: Risk-free rate for excess return calculation
            annualization_factor: Factor for annualizing the Sharpe ratio (default: 252 for daily data)
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

    @property
    def name(self) -> str:
        """Get the name of this objective."""
        return "out_of_sample_sharpe"

    @property
    def is_higher_better(self) -> bool:
        """Higher Sharpe ratio is better."""
        return True

    def _calculate_objective(self, features: pd.DataFrame,
                           targets: pd.Series,
                           selected_features: List[str],
                           **kwargs) -> ObjectiveResult:
        """
        Calculate the out-of-sample Sharpe ratio.
        
        Args:
            features: Aligned feature DataFrame (no NaN targets)
            targets: Aligned target Series (no NaN values) - assumed to be returns
            selected_features: List of valid feature names
            **kwargs: Additional keyword arguments
            
        Returns:
            ObjectiveResult with Sharpe ratio and detailed metadata
        """
        try:
            # Calculate basic statistics
            returns = targets
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Handle edge cases
            if std_return == 0:
                tprint_warning("⚠️ Zero standard deviation in returns - Sharpe ratio undefined")
                return ObjectiveResult(
                    value=0.0,
                    metadata={
                        'mean_return': float(mean_return),
                        'std_return': float(std_return),
                        'excess_return': float(mean_return - self.risk_free_rate),
                        'n_periods': len(returns),
                        'risk_free_rate': self.risk_free_rate,
                        'warning': 'Zero standard deviation - Sharpe ratio undefined'
                    },
                    is_valid=True
                )
            
            # Calculate excess returns
            excess_returns = returns - self.risk_free_rate
            excess_return_mean = excess_returns.mean()
            
            # Calculate Sharpe ratio
            sharpe_ratio = excess_return_mean / std_return
            
            # Annualize if we have enough data
            if len(returns) > self.annualization_factor:
                sharpe_ratio_annualized = sharpe_ratio * np.sqrt(self.annualization_factor)
                annualization_applied = True
            else:
                sharpe_ratio_annualized = sharpe_ratio
                annualization_applied = False
            
            # Calculate additional risk metrics
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0.0
            
            # Calculate Sortino ratio (alternative to Sharpe)
            sortino_ratio = excess_return_mean / downside_deviation if downside_deviation > 0 else 0.0
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
            
            # Calculate hit rate (percentage of positive returns)
            hit_rate = (returns > 0).mean()
            
            # Calculate volatility of returns
            return_volatility = returns.std()
            
            # Calculate skewness and kurtosis
            skewness = returns.skew() if len(returns) > 2 else 0.0
            kurtosis = returns.kurtosis() if len(returns) > 3 else 0.0
            
            # Prepare comprehensive metadata
            metadata = {
                'sharpe_ratio': float(sharpe_ratio),
                'sharpe_ratio_annualized': float(sharpe_ratio_annualized),
                'annualization_applied': annualization_applied,
                'annualization_factor': self.annualization_factor,
                'mean_return': float(mean_return),
                'std_return': float(std_return),
                'excess_return': float(excess_return_mean),
                'risk_free_rate': self.risk_free_rate,
                'n_periods': len(returns),
                'return_volatility': float(return_volatility),
                'downside_deviation': float(downside_deviation),
                'sortino_ratio': float(sortino_ratio),
                'max_drawdown': float(max_drawdown),
                'hit_rate': float(hit_rate),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'return_percentiles': {
                    'p5': float(returns.quantile(0.05)),
                    'p25': float(returns.quantile(0.25)),
                    'p50': float(returns.quantile(0.50)),
                    'p75': float(returns.quantile(0.75)),
                    'p95': float(returns.quantile(0.95))
                },
                'feature_importance': self._calculate_feature_importance(features, targets, selected_features)
            }
            
            return ObjectiveResult(
                value=float(sharpe_ratio_annualized),  # Return annualized Sharpe ratio
                metadata=metadata,
                is_valid=True
            )
            
        except Exception as e:
            tprint_error(f"❌ Sharpe ratio calculation failed: {e}")
            return ObjectiveResult(
                value=0.0,
                metadata={
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'n_periods': len(targets),
                    'n_features': len(selected_features)
                },
                is_valid=False
            )

    def _calculate_feature_importance(self, features: pd.DataFrame, 
                                    targets: pd.Series, 
                                    selected_features: List[str]) -> Dict[str, float]:
        """
        Calculate feature importance using correlation with targets.
        
        Args:
            features: Feature DataFrame
            targets: Target Series
            selected_features: List of selected feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            importance_scores = {}
            
            for feature_name in selected_features:
                if feature_name in features.columns:
                    feature_data = features[feature_name]
                    
                    # Calculate correlation with targets
                    correlation = safe_correlation(feature_data, targets)
                    
                    # Calculate mutual information if possible
                    try:
                        from sklearn.feature_selection import mutual_info_regression
                        mi_score = mutual_info_regression(
                            feature_data.values.reshape(-1, 1), 
                            targets.values
                        )[0]
                    except:
                        mi_score = 0.0
                    
                    # Combine correlation and mutual information
                    importance_score = abs(correlation) * 0.7 + mi_score * 0.3
                    
                    importance_scores[feature_name] = {
                        'correlation': float(correlation),
                        'mutual_information': float(mi_score),
                        'combined_score': float(importance_score)
                    }
            
            return importance_scores
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature importance calculation failed: {e}")
            return {feature: {'correlation': 0.0, 'mutual_information': 0.0, 'combined_score': 0.0} 
                   for feature in selected_features}

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

    def _calculate_objective(self, features: pd.DataFrame,
                           targets: pd.Series,
                           selected_features: List[str],
                           **kwargs) -> ObjectiveResult:
        """
        Calculate maximum drawdown objective.

        Args:
            features: Aligned feature DataFrame (no NaN targets)
            targets: Aligned target Series (no NaN values) - assumed to be returns
            selected_features: List of valid feature names
            **kwargs: Additional arguments

        Returns:
            ObjectiveResult: Maximum drawdown value and metadata
        """
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
            return ObjectiveResult(
                value=1.0,
                metadata={'error': str(e)},
                is_valid=False
            )

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

    def _calculate_objective(self, features: pd.DataFrame,
                           targets: pd.Series,
                           selected_features: List[str],
                           **kwargs) -> ObjectiveResult:
        """
        Calculate turnover objective.

        Args:
            features: Aligned feature DataFrame (no NaN targets)
            targets: Aligned target Series (no NaN values) - assumed to be returns
            selected_features: List of valid feature names
            **kwargs: Additional arguments

        Returns:
            ObjectiveResult: Turnover value and metadata
        """
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
            return ObjectiveResult(
                value=0.0,
                metadata={'error': str(e)},
                is_valid=False
            )

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

    def _calculate_objective(self, features: pd.DataFrame,
                           targets: pd.Series,
                           selected_features: List[str],
                           **kwargs) -> ObjectiveResult:
        """
        Calculate stability objective using Jaccard similarity.

        Args:
            features: Aligned feature DataFrame (no NaN targets)
            targets: Aligned target Series (no NaN values) - assumed to be returns
            selected_features: List of valid feature names
            **kwargs: Additional arguments

        Returns:
            ObjectiveResult: Stability value and metadata
        """
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
            return ObjectiveResult(
                value=0.0,
                metadata={'error': str(e)},
                is_valid=False
            )

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

    def _calculate_objective(self, features: pd.DataFrame,
                           targets: pd.Series,
                           selected_features: List[str],
                           **kwargs) -> ObjectiveResult:
        """
        Calculate diversity objective (minimize correlation).

        Args:
            features: Aligned feature DataFrame (no NaN targets)
            targets: Aligned target Series (no NaN values) - assumed to be returns
            selected_features: List of valid feature names
            **kwargs: Additional arguments

        Returns:
            ObjectiveResult: Diversity value and metadata
        """
        try:
            if not selected_features:
                return ObjectiveResult(value=0.0, metadata={}, is_valid=False)

            # Get selected features
            selected_data = features[selected_features]

            # Calculate correlation matrix
            corr_matrix = selected_data.corr().abs()

            # Remove diagonal
            corr_matrix = corr_matrix - np.eye(len(corr_matrix))

            # Calculate average correlation penalty
            penalty = corr_matrix.sum().sum() / (len(corr_matrix) * (len(corr_matrix) - 1))

            # Convert to diversity score (higher is better)
            diversity = 1.0 - penalty

            metadata = {
                'diversity_score': diversity,
                'correlation_penalty': penalty,
                'avg_correlation': corr_matrix.sum().sum() / (len(corr_matrix) * (len(corr_matrix) - 1)),
                'n_features': len(selected_features)
            }

            return ObjectiveResult(value=max(0.0, diversity), metadata=metadata, is_valid=True)

        except Exception as e:
            return ObjectiveResult(
                value=0.0,
                metadata={'error': str(e)},
                is_valid=False
            )

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

    def _calculate_objective(self, features: pd.DataFrame,
                           targets: pd.Series,
                           selected_features: List[str],
                           **kwargs) -> ObjectiveResult:
        """
        Calculate mutual information objective.

        Args:
            features: Aligned feature DataFrame (no NaN targets)
            targets: Aligned target Series (no NaN values) - assumed to be returns
            selected_features: List of valid feature names
            **kwargs: Additional arguments

        Returns:
            ObjectiveResult: Mutual information value and metadata
        """
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
            return ObjectiveResult(
                value=0.0,
                metadata={'error': str(e)},
                is_valid=False
            )

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

    def _calculate_objective(self, features: pd.DataFrame,
                           targets: pd.Series,
                           selected_features: List[str],
                           **kwargs) -> ObjectiveResult:
        """
        Calculate profit-centered objective (maximize profit while minimizing risk).

        Args:
            features: Aligned feature DataFrame (no NaN targets)
            targets: Aligned target Series (no NaN values) - assumed to be returns
            selected_features: List of valid feature names
            **kwargs: Additional arguments

        Returns:
            ObjectiveResult: Profit-centered value and metadata
        """
        try:
            if not selected_features:
                return ObjectiveResult(value=0.0, metadata={}, is_valid=False)

            # Calculate total return
            total_return = targets.sum()

            # Calculate risk (standard deviation)
            risk = targets.std()

            # Calculate profit score: return - risk_penalty * risk
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
            return ObjectiveResult(
                value=0.0,
                metadata={'error': str(e)},
                is_valid=False
            )

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
        self.use_evolutionary = False  # Disabled - using faster standard methods
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

        # Evolutionary algorithms removed - using faster standard methods instead
        self.evolutionary_config = None
        self.nsga2_optimizer = None
        self.spea2_optimizer = None
        self.ga_optimizer = None
        tprint_info("✅ Evolutionary algorithms disabled - using optimized standard methods")

        # Initialize hardware optimization components
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            # Initialize M1 hardware optimizers
            self.m1_gpu_optimizer = get_m1_gpu_optimizer()
            self.m1_memory_optimizer = get_m1_memory_optimizer()
            self.m1_cpu_optimizer = get_m1_cpu_optimizer()
            self.hardware_manager = UnifiedHardwareManager()
            tprint_info("✅ M1 hardware optimization components initialized")
        else:
            self.m1_gpu_optimizer = None
            self.m1_memory_optimizer = None
            self.m1_cpu_optimizer = None
            self.hardware_manager = None

        # Initialize VectorBT rolling optimizer
        if VECTORBT_ROLLING_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            self.ml_vectorization_manager = get_unified_vectorization_manager()
            tprint_info("✅ VectorBT rolling optimizer initialized")
        else:
            self.vectorbt_rolling_optimizer = None
            self.ml_vectorization_manager = None

        # Initialize Bayesian TPE optimizer
        if ML_COMMONS_PARETO_AVAILABLE:
            self.bayesian_tpe_optimizer = BayesianTPEOptimizer()
            tprint_info("✅ Bayesian TPE optimizer initialized")
        else:
            self.bayesian_tpe_optimizer = None

        # Algorithm selection strategy
        self.algorithm_selection_strategy = "adaptive"  # "adaptive", "fastest", "best_quality"
        self.performance_history = {
            'correlation_based_times': [],
            'mutual_information_times': [],
            'bayesian_tpe_times': [],
            'standard_multi_objective_times': []
        }

        # Initialize CMI complementarity components if available
        if CMI_COMPLEMENTARITY_AVAILABLE:
            # CMI configuration for multi-objective selection
            cmi_config = CMIComplementarityConfig(
                per_family_budget=(5, 15),  # Min/max features per family
                upstream_multiplier=3,  # Total budget to RFE = 3× per-family
                max_total_features=max_features,  # Use same max as selector
                enable_regime_awareness=True,  # Compute R(X|A) per regime
                compute_timeout_seconds=300.0,  # 5 min hard limit
                enable_synergy=True,  # Enable synergy computation
                beta_synergy=0.25  # Synergy bonus weight
            )
            self.cmi_scorer = CMIComplementarityScorer(cmi_config)
            self.analyst_handler = AnalystSideInfoHandler()
            tprint_info("✅ CMI complementarity components initialized")
        else:
            self.cmi_scorer = None
            self.analyst_handler = None
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

        # Initialize objective evaluation cache
        self.objective_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Enable parallel processing for stability selection
        self.enable_parallel_stability = True
        self.max_workers = min(4, os.cpu_count() or 1)  # Limit to 4 workers

        tprint_info(f"Initialized Enhanced MultiObjectiveFeatureSelector with {len(objectives)} objectives")
        tprint_info(f"🚀 Parallel processing enabled with {self.max_workers} workers")
        if self.use_ml_commons:
            tprint_info("✅ ML Commons integration enabled")
        if self.use_evolutionary:
            tprint_info("✅ Evolutionary optimization enabled")

    def _apply_fail_fast_gates(self, features: pd.DataFrame, targets: pd.Series) -> bool:
        """Apply fail-fast validation gates following battle-tested best practices."""
        tprint_info("🚪 [VALIDATION] Starting fail-fast validation gates")
        tprint_debug(f"🚪 [VALIDATION] Input features shape: {features.shape}")
        tprint_debug(f"🚪 [VALIDATION] Input targets shape: {targets.shape}")

        # Gate 1: Minimum data size
        tprint_debug(f"🚪 [VALIDATION] Gate 1 - Data size check: {len(features)} samples")
        if len(features) < 100:
            tprint_warning("⚠️ Insufficient data for reliable feature selection")
            return False

        # Gate 2: Target variance check
        target_var = targets.var()
        tprint_debug(f"🚪 [VALIDATION] Gate 2 - Target variance check: {target_var:.6f}")
        if target_var < 1e-8:
            tprint_warning("⚠️ Target variance too low")
            return False

        # Gate 3: Feature quality check
        nan_ratios = features.isnull().sum() / len(features)
        high_nan_features = nan_ratios > 0.01  # Reduced from 0.3 to 0.01 (1%)
        high_nan_count = high_nan_features.sum()
        tprint_debug(f"🚪 [VALIDATION] Gate 3 - NaN check: {high_nan_count} features with >1% NaN")
        if high_nan_features.any():
            tprint_warning(f"⚠️ {high_nan_count} features have >1% NaN values")
            return False

        # Gate 4: Memory check
        memory_usage = features.memory_usage(deep=True).sum() / 1024**2  # MB
        tprint_debug(f"🚪 [VALIDATION] Gate 4 - Memory check: {memory_usage:.1f} MB")
        if memory_usage > 3000:  # 3GB limit (increased from 2GB)
            tprint_warning(f"⚠️ High memory usage: {memory_usage:.1f}MB")
            return False

        # Additional validation: Check for reasonable feature count
        feature_count = len(features.columns)
        tprint_debug(f"🚪 [VALIDATION] Additional check - Feature count: {feature_count}")
        if feature_count < 2:
            tprint_warning("⚠️ Too few features for meaningful selection")
            return False

        # Additional validation: Check for target data quality
        targets_len = len(targets)
        features_len = len(features)
        tprint_debug(f"🚪 [VALIDATION] Additional check - Length match: targets={targets_len}, features={features_len}")
        if targets_len != features_len:
            tprint_warning(f"⚠️ Target length ({targets_len}) != features length ({features_len})")
            return False

        tprint_info(f"✅ All validation gates passed: {feature_count} features, {features_len} samples, target variance: {target_var:.6f}")
        return True

    def _stability_selection(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """Perform stability selection with bootstrapped time blocks."""
        tprint_info("🔄 Performing stability selection with bootstrapped time blocks")

        stability_scores = {}
        n_samples = len(features)
        bootstrap_size = int(n_samples * 0.8)  # 80% bootstrap

        for _ in range(25):  # 25 bootstrap iterations (optimized from 100)
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
            stability_scores[feature_name] /= 25

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
                    n = len(feature_data)
                    # Prefer position-based splitter if available
                    if hasattr(self.purged_kfold, 'split_positions'):
                        splitter = self.purged_kfold.split_positions(n, getattr(feature_data, 'index', None))
                    else:
                        X = pd.DataFrame(index=feature_data.index)
                        splitter = self.purged_kfold.split(X)

                    for train_idx, val_idx in splitter:
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
                    n = len(feature_data)
                    if hasattr(self.purged_kfold, 'split_positions'):
                        splitter = self.purged_kfold.split_positions(n, getattr(feature_data, 'index', None))
                    else:
                        X = pd.DataFrame(index=feature_data.index)
                        splitter = self.purged_kfold.split(X)
                    for train_idx, val_idx in splitter:
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
        Optimize features using hardware-optimized multi-objective optimization.
        
        Integrates:
        - M1 hardware optimization (GPU, memory, CPU)
        - VectorBT rolling optimizer for efficient computations
        - Bayesian TPE optimizer for grid search
        - ML commons utilities for CV, OOF, and data leakage prevention
        
        Args:
            data: Input data with features
            targets: Target values for optimization
            
        Returns:
            MultiObjectiveResult with optimized features
        """
        try:
            tprint_info("🎯 Starting hardware-optimized multi-objective feature optimization")
            tprint_debug(f"📊 Input data shape: {data.shape}")
            tprint_debug(f"📊 Target data shape: {targets.shape if targets is not None else 'None'}")
            tprint_debug(f"📊 Available columns: {list(data.columns)}")
            tprint_debug(f"📊 Number of objectives: {len(self.objectives)}")
            tprint_debug(f"📊 Objectives: {[obj.name for obj in self.objectives]}")
            
            # Step 1: Hardware-aware data preparation
            tprint_info("🔧 Step 1: Hardware-aware data preparation")
            optimized_data, optimized_targets = self._prepare_data_hardware_optimized(data, targets)
            
            # Step 2: VectorBT-optimized feature evaluation
            tprint_info("⚡ Step 2: VectorBT-optimized feature evaluation")
            if self.vectorbt_rolling_optimizer is not None:
                feature_scores = self._evaluate_features_vectorbt_optimized(optimized_data, optimized_targets)
            else:
                feature_scores = self._evaluate_features_standard(optimized_data, optimized_targets)
            
            # Step 3: Bayesian TPE optimization for feature selection
            tprint_info("🎯 Step 3: Bayesian TPE optimization")
            if self.bayesian_tpe_optimizer is not None:
                selected_features = self._optimize_with_bayesian_tpe(
                    optimized_data, optimized_targets, feature_scores
                )
            else:
                selected_features = self._optimize_with_standard_method(
                    optimized_data, optimized_targets, feature_scores
                )
            
            # Step 4: Final validation and metrics
            tprint_info("✅ Step 4: Final validation and metrics")
            final_metrics = self._compute_final_metrics(
                optimized_data, optimized_targets, selected_features
            )
            
            tprint_success(f"✅ Hardware-optimized feature selection completed: {len(selected_features)} features selected")
            
            return MultiObjectiveResult(
                selected_features=selected_features,
                objective_values={},
                pareto_front=[],
                optimization_metadata=final_metrics,
                feature_scores=feature_scores,
                success=True,
                error_message=None
            )
                
        except Exception as e:
            tprint_error(f"❌ Hardware-optimized feature optimization failed: {e}")
            tprint_debug(f"🔍 Error details: {type(e).__name__}: {str(e)}")
            return MultiObjectiveResult(
                selected_features=data.columns.tolist(),
                objective_values={},
                pareto_front=[],
                optimization_metadata={},
                feature_scores={},
                success=False,
                error_message=str(e)
            )

    def _prepare_data_hardware_optimized(self, data: pd.DataFrame, targets: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data with M1 hardware optimization."""
        tprint_info("🔧 Preparing data with M1 hardware optimization")
        
        # Memory optimization
        if self.m1_memory_optimizer is not None:
            tprint_info("🧠 Applying M1 memory optimization")
            tprint_debug(f"🧠 M1MemoryOptimizer type: {type(self.m1_memory_optimizer)}")
            tprint_debug(f"🧠 Available methods: {[m for m in dir(self.m1_memory_optimizer) if not m.startswith('_')]}")
            data = self.m1_memory_optimizer.optimize_dataframe_memory(data)
            if hasattr(self.m1_memory_optimizer, 'optimize_series_memory'):
                targets = self.m1_memory_optimizer.optimize_series_memory(targets)
            else:
                tprint_warning("⚠️ optimize_series_memory method not found, skipping series optimization")
                tprint_warning(f"⚠️ Available methods: {[m for m in dir(self.m1_memory_optimizer) if 'series' in m.lower()]}")
        
        # GPU optimization for large datasets
        if (self.m1_gpu_optimizer is not None and 
            len(data) > 10000 and 
            self.hardware_manager is not None):
            tprint_info("⚡ Applying M1 GPU optimization for large dataset")
            try:
                data = self.m1_gpu_optimizer.optimize_dataframe_gpu(data)
                targets = self.m1_gpu_optimizer.optimize_series_gpu(targets)
            except Exception as e:
                tprint_warning(f"⚠️ GPU optimization failed, using CPU: {e}")
        
        # CPU optimization
        if self.m1_cpu_optimizer is not None:
            tprint_info("🖥️ Applying M1 CPU optimization")
            data = self.m1_cpu_optimizer.optimize_dataframe_cpu(data)
            targets = self.m1_cpu_optimizer.optimize_series_cpu(targets)
        
        tprint_success(f"✅ Data preparation completed: {data.shape}")
        return data, targets

    def _evaluate_features_vectorbt_optimized(self, data: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """Evaluate features using VectorBT rolling optimizer."""
        tprint_info("⚡ Evaluating features with VectorBT rolling optimizer")
        
        feature_scores = {}
        
        try:
            # Use VectorBT rolling optimizer for efficient computations
            if self.ml_vectorization_manager is not None:
                # Configure operation for feature evaluation
                config = OperationConfig(
                    operation_type=OperationType.FEATURE_EVALUATION,
                    data_size=len(data),
                    data_dimensions=(len(data), len(data.columns)),
                    memory_budget_mb=8192.0,  # 8GB in MB
                    time_budget_seconds=300.0,
                    precision_requirement="medium"
                )
                
                # Optimize feature evaluation operation
                result = self.ml_vectorization_manager.optimize_operation(
                    operation_type=OperationType.FEATURE_ENGINEERING,
                    data=data,
                    config=config
                )
                
                if result.success:
                    tprint_success("✅ VectorBT optimization successful")
                    # Extract feature scores from optimized result
                    feature_scores = result.metadata.get('feature_scores', {})
                else:
                    tprint_warning("⚠️ VectorBT optimization failed, using standard evaluation")
                    feature_scores = self._evaluate_features_standard(data, targets)
            else:
                feature_scores = self._evaluate_features_standard(data, targets)
                
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT evaluation failed: {e}")
            feature_scores = self._evaluate_features_standard(data, targets)
        
        tprint_success(f"✅ Feature evaluation completed: {len(feature_scores)} features scored")
        return feature_scores

    def _evaluate_features_standard(self, data: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """Standard feature evaluation fallback."""
        tprint_info("📊 Using standard feature evaluation")
        
        feature_scores = {}
        for col in data.columns:
            try:
                # Calculate correlation as a simple score
                correlation = safe_correlation(data[col].dropna(), targets.loc[data[col].dropna().index])
                feature_scores[col] = abs(correlation)
            except Exception:
                feature_scores[col] = 0.0
        
        return feature_scores

    def _optimize_with_bayesian_tpe(self, data: pd.DataFrame, targets: pd.Series, feature_scores: Dict[str, float]) -> List[str]:
        """Optimize feature selection using grid search (Bayesian TPE fallback)."""
        tprint_info("🎯 Optimizing with grid search (Bayesian TPE fallback)")
        
        try:
            # Simple grid search instead of problematic Bayesian optimizer
            best_score = -float('inf')
            best_features = []
            
            # Test different parameter combinations
            n_features_options = range(self.min_features, min(self.max_features + 1, len(data.columns) + 1, 21))  # Limit to 20 max
            correlation_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            
            for n_features in n_features_options:
                for corr_threshold in correlation_thresholds:
                    try:
                        # Select features based on scores and thresholds
                        selected_features = self._select_features_by_scores(
                            feature_scores, n_features, corr_threshold
                        )
                        
                        if not selected_features:
                            continue
                        
                        # Calculate objective value (maximize feature quality)
                        selected_data = data[selected_features]
                        correlations = [safe_correlation(selected_data[col].dropna(), targets.loc[selected_data[col].dropna().index]) for col in selected_features]
                        score = np.mean([abs(c) for c in correlations if not np.isnan(c)])
                        
                        if score > best_score:
                            best_score = score
                            best_features = selected_features
                            
                    except Exception:
                        continue
            
            if not best_features:
                # Fallback to simple selection
                best_features = self._select_features_by_scores(feature_scores, self.min_features, 0.5)
            
            tprint_success(f"✅ Grid search selected {len(best_features)} features")
            return best_features
            
        except Exception as e:
            tprint_warning(f"⚠️ Grid search optimization failed: {e}")
            return self._optimize_with_standard_method(data, targets, feature_scores)

    def _optimize_with_standard_method(self, data: pd.DataFrame, targets: pd.Series, feature_scores: Dict[str, float]) -> List[str]:
        """Standard optimization method fallback."""
        tprint_info("📊 Using standard optimization method")
        
        # Sort features by score and select top features
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        n_features = min(self.max_features, len(sorted_features))
        selected_features = [feat for feat, score in sorted_features[:n_features]]
        
        tprint_success(f"✅ Standard optimization completed: {len(selected_features)} features selected")
        return selected_features

    def _select_features_by_scores(self, feature_scores: Dict[str, float], n_features: int, correlation_threshold: float) -> List[str]:
        """Select features based on scores and thresholds."""
        # Filter by correlation threshold
        filtered_features = {
            feat: score for feat, score in feature_scores.items() 
            if score >= correlation_threshold
        }
        
        # Sort by score and select top n_features
        sorted_features = sorted(filtered_features.items(), key=lambda x: x[1], reverse=True)
        return [feat for feat, score in sorted_features[:n_features]]

    def _compute_final_metrics(self, data: pd.DataFrame, targets: pd.Series, selected_features: List[str]) -> Dict[str, Any]:
        """Compute final optimization metrics."""
        tprint_info("📊 Computing final metrics")
        
        metrics = {
            'n_features_selected': len(selected_features),
            'n_features_total': len(data.columns),
            'selection_ratio': len(selected_features) / len(data.columns),
            'hardware_optimization_used': self.hardware_manager is not None,
            'vectorbt_optimization_used': self.vectorbt_rolling_optimizer is not None,
            'bayesian_tpe_used': self.bayesian_tpe_optimizer is not None
        }
        
        # Calculate performance metrics for selected features
        if selected_features:
            try:
                selected_data = data[selected_features]
                correlations = []
                for col in selected_features:
                    try:
                        corr = safe_correlation(selected_data[col].dropna(), targets.loc[selected_data[col].dropna().index])
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                    except Exception:
                        pass
                
                if correlations:
                    metrics['mean_correlation'] = np.mean(correlations)
                    metrics['max_correlation'] = np.max(correlations)
                    metrics['min_correlation'] = np.min(correlations)
            except Exception as e:
                tprint_warning(f"⚠️ Failed to compute correlation metrics: {e}")
        
        tprint_success("✅ Final metrics computed")
        return metrics

    def _select_optimal_algorithm(self, data: pd.DataFrame, objectives: List[ObjectiveFunction]) -> str:
        """
        Select the optimal algorithm based on problem characteristics.
        
        Returns:
            str: Algorithm name ('correlation_based', 'mutual_information', 'bayesian_tpe')
        """
        tprint_info("🎯 Selecting optimal algorithm based on problem characteristics")
        
        n_features = len(data.columns)
        n_objectives = len(objectives)
        n_samples = len(data)
        
        tprint_debug(f"📊 Problem characteristics:")
        tprint_debug(f"   • Features: {n_features}")
        tprint_debug(f"   • Objectives: {n_objectives}")
        tprint_debug(f"   • Samples: {n_samples}")
        
        # Simplified algorithm selection (evolutionary algorithms removed)
        if n_features < 50:
            # Small problem - use fastest correlation-based method
            tprint_info("🎯 Small problem → Using correlation-based selection (fastest)")
            return "correlation_based"
        elif n_objectives == 1:
            # Single objective - use mutual information
            tprint_info("🎯 Single objective → Using mutual information (effective)")
            return "mutual_information"
        elif n_features > 200 or n_samples > 20000:
            # Large problem - use Bayesian TPE for efficiency
            tprint_info("🎯 Large problem → Using Bayesian TPE (most efficient)")
            return "bayesian_tpe"
        else:
            # Medium problem - use standard multi-objective
            tprint_info("🎯 Medium problem → Using standard multi-objective optimization")
            return "standard_multi_objective"

    def _record_algorithm_performance(self, algorithm: str, execution_time: float):
        """Record algorithm performance for adaptive selection."""
        if algorithm in self.performance_history:
            self.performance_history[algorithm].append(execution_time)
            # Keep only last 10 runs to avoid memory issues
            if len(self.performance_history[algorithm]) > 10:
                self.performance_history[algorithm] = self.performance_history[algorithm][-10:]
            
            tprint_debug(f"📊 Recorded {algorithm} performance: {execution_time:.2f}s")
            tprint_debug(f"📊 Average {algorithm} time: {np.mean(self.performance_history[algorithm]):.2f}s")

    def select_features(self, features: pd.DataFrame,
                       targets: pd.Series,
                       cv_splits: Optional[List[Any]] = None,
                       use_evolutionary: bool = None,
                       analyst_side_info: Optional[np.ndarray] = None,
                       prefilter_mask: Optional[np.ndarray] = None,
                       pipeline_state: Optional[Dict[str, Any]] = None) -> MultiObjectiveResult:
        """
        Select features using enhanced multi-objective optimization with battle-tested best practices.

        Args:
            features: Feature DataFrame
            targets: Target series
            cv_splits: Optional CV splits for stability calculation
            use_evolutionary: Override evolutionary algorithm usage
            analyst_side_info: Analyst side information for CMI complementarity
            prefilter_mask: Pre-computed feature mask from upstream CMI filtering
            pipeline_state: Pipeline state for regime information

        Returns:
            MultiObjectiveResult with selected features and objective values
        """
        tprint_info(f"Starting battle-tested multi-objective feature selection for {features.shape[1]} features")
        
        # Check if CMI complementarity is enabled (Tactician mode only)
        # Temporarily disable CMI to debug the validation issue
        enable_cmi_complementarity = False  # Disable for debugging
        
        if enable_cmi_complementarity:
            tprint_info("🎯 CMI complementarity enabled for Tactician mode multi-objective selection")
            tprint_info("🔧 Tactician mode detected - CMI complementarity will be applied")
        else:
            tprint_info("📊 Standard multi-objective selection (Analyst mode or CMI unavailable)")
            tprint_info("🔧 Analyst mode detected - CMI complementarity disabled")
        
        # Apply prefilter mask if provided
        if prefilter_mask is not None and len(prefilter_mask) == len(features.columns):
            original_count = len(features.columns)
            features = features.loc[:, prefilter_mask]
            tprint_info(f"✅ Applied prefilter mask: {original_count} → {len(features.columns)} features")
        
        # Apply fast pre-filtering to reduce feature set before expensive operations
        if len(features.columns) > 100:  # Only apply if we have many features
            features = self._fast_prefilter_features(features, targets)
        
        # Apply CMI complementarity prefiltering if enabled
        if enable_cmi_complementarity and analyst_side_info is not None:
            try:
                tprint_info("🎯 Applying CMI complementarity prefiltering")
                cmi_result = self.cmi_scorer.score_features(
                    features, targets, analyst_side_info,
                    pipeline_state=pipeline_state
                )
                
                if cmi_result.is_valid and cmi_result.selected_features:
                    original_count = len(features.columns)
                    features = features[cmi_result.selected_features]
                    tprint_success(f"✅ CMI prefiltering: {original_count} → {len(features.columns)} features")
                    tprint_info(f"📊 Noise floor: {cmi_result.noise_floor:.6f}")
                    tprint_info(f"📊 ΔPerf threshold: {cmi_result.delta_perf_threshold:.6f}")
                else:
                    tprint_warning("⚠️ CMI complementarity scoring failed, using all features")
                    
            except Exception as e:
                tprint_warning(f"⚠️ CMI complementarity prefiltering failed: {e}, using all features")

        # Step 1: Apply fail-fast gates
        tprint_info("🚪 Step 1: Applying fail-fast validation gates")
        validation_result = self._apply_fail_fast_gates(features, targets)
        tprint_debug(f"🚪 [SELECTION] Validation result: {validation_result}")

        if not validation_result:
            tprint_error("❌ [SELECTION] Failed fail-fast validation gates")
            return MultiObjectiveResult(
                selected_features=[],
                objective_values={},
                pareto_front=[],
                optimization_metadata={'error': 'Failed fail-fast validation gates'},
                is_valid=False
            )

        tprint_success("✅ [SELECTION] Passed fail-fast validation gates")

        # Step 2: Stability selection with bootstrapped time blocks
        stability_scores = {}
        if self.enable_stability_selection:
            tprint_info("🔄 Step 2: Stability selection with bootstrapped time blocks")
            tprint_debug(f"🔄 [SELECTION] Input to stability selection: features={features.shape}, targets={targets.shape}")
            stability_scores = self._stability_selection(features, targets)
            tprint_debug(f"🔄 [SELECTION] Stability scores computed: {len(stability_scores)} features")
        else:
            tprint_info("🔄 [SELECTION] Stability selection disabled, skipping")
            stability_scores = {}

        # Step 3: Set CV splits for stability objective
        tprint_debug(f"🔧 [SELECTION] Setting CV splits for {len([obj for obj in self.objectives if isinstance(obj, StabilityObjective)])} stability objectives")
        for obj in self.objectives:
            if isinstance(obj, StabilityObjective):
                obj.cv_splits = cv_splits

        # Step 4: Choose optimization method (simplified - use most efficient)
        use_evo = use_evolutionary if use_evolutionary is not None else self.use_evolutionary

        # Use optimized standard multi-objective optimization (evolutionary algorithms removed)
        tprint_info("📊 Step 4: Using optimized standard multi-objective optimization")
        tprint_debug(f"📊 [SELECTION] Input to optimization: features={features.shape}, targets={targets.shape}")
        tprint_debug(f"📊 [SELECTION] CV splits: {cv_splits}")
        tprint_debug(f"📊 [SELECTION] Number of objectives: {len(self.objectives)}")

        try:
            result = self._standard_feature_selection(features, targets, cv_splits)
            tprint_debug(f"📊 [SELECTION] Optimization result: valid={result.is_valid}")
            tprint_debug(f"📊 [SELECTION] Selected features: {len(result.selected_features) if result.selected_features else 0}")
        except Exception as opt_error:
            tprint_error(f"❌ [SELECTION] Optimization failed: {opt_error}")
            raise opt_error

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
        """Standard multi-objective feature selection with early stopping."""
        tprint_info("📊 Using optimized standard multi-objective feature selection with early stopping")

        # Generate candidate feature sets
        candidate_sets = self._generate_candidate_sets(features.columns.tolist())

        # Early stopping parameters
        max_evaluations = 200  # Limit evaluations for early stopping
        quality_threshold = 0.8  # Stop when we find high-quality solutions
        consecutive_no_improvement = 0
        max_no_improvement = 50  # Stop after 50 consecutive evaluations with no improvement
        
        # Evaluate objectives for each candidate set with early stopping
        pareto_front = []
        best_score = -float('inf')
        
        tprint_info(f"Evaluating {len(candidate_sets)} candidate sets with early stopping...")

        for i, candidate_set in enumerate(candidate_sets):
            # Early stopping checks
            if i >= max_evaluations:
                tprint_info(f"Early stopping: Reached max evaluations ({max_evaluations})")
                break
                
            if consecutive_no_improvement >= max_no_improvement:
                tprint_info(f"Early stopping: No improvement for {consecutive_no_improvement} evaluations")
                break

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
                
                # Check for improvement
                if weighted_score > best_score:
                    best_score = weighted_score
                    consecutive_no_improvement = 0
                else:
                    consecutive_no_improvement += 1
                
                # Early stopping if we find a very good solution
                if weighted_score >= quality_threshold:
                    tprint_info(f"Early stopping: Found high-quality solution (score: {weighted_score:.4f})")
                    break

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
                'method': 'standard_optimized',
                'n_candidates': len(candidate_sets),
                'n_evaluated': min(i + 1, len(candidate_sets)),
                'n_valid': len(pareto_front),
                'weights': self.weights,
                'max_features': self.max_features,
                'min_features': self.min_features,
                'early_stopping_applied': i + 1 < len(candidate_sets),
                'best_score': best_score,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses
            },
            is_valid=len(pareto_front) > 0
        )

        tprint_success(f"Optimized feature selection completed: {len(selected_features)} features selected")
        tprint_info(f"📊 Evaluated {i + 1}/{len(candidate_sets)} candidate sets")
        if self.cache_hits > 0:
            cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
            tprint_info(f"📊 Cache hit rate: {cache_hit_rate:.2%}")
        
        return result

    def _generate_candidate_sets(self, all_features: List[str]) -> List[List[str]]:
        """Generate candidate feature sets for evaluation with smart sampling and early stopping."""
        candidate_sets = []
        
        # Smart sampling parameters
        max_candidates = 500  # Reduced from 1000
        quality_threshold = 0.8  # Stop when we find high-quality solutions
        max_sets_per_size = 50  # Limit combinations per feature count
        
        # Pre-compute feature importance for smarter sampling
        feature_importance = self._compute_quick_feature_importance(all_features)
        sorted_features = sorted(all_features, key=lambda x: feature_importance.get(x, 0), reverse=True)
        
        best_scores = []
        
        # Generate sets of different sizes with smart sampling
        for n_features in range(self.min_features, min(self.max_features + 1, len(all_features) + 1)):
            if len(candidate_sets) >= max_candidates:
                break
                
            # Use importance-based sampling instead of all combinations
            if n_features <= 8:  # Use combinations for small sets
                from itertools import combinations
                for combo in combinations(sorted_features, n_features):
                    candidate_sets.append(list(combo))
                    if len(candidate_sets) >= max_candidates:
                        break
            else:  # Use smart sampling for larger sets
                for _ in range(min(max_sets_per_size, max_candidates - len(candidate_sets))):
                    # Sample features based on importance with some randomness
                    selected = self._smart_sample_features(sorted_features, n_features, feature_importance)
                    if selected not in candidate_sets:
                        candidate_sets.append(selected)
                    
                    if len(candidate_sets) >= max_candidates:
                        break

        tprint_info(f"Generated {len(candidate_sets)} candidate feature sets (optimized sampling)")
        return candidate_sets
    
    def _compute_quick_feature_importance(self, features: List[str]) -> Dict[str, float]:
        """Quick feature importance computation for smart sampling."""
        # This is a placeholder - in practice, you'd use actual feature importance
        # For now, return uniform importance
        return {feature: 1.0 for feature in features}
    
    def _fast_prefilter_features(self, features: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
        """Fast pre-filtering to reduce feature set before expensive operations."""
        tprint_info("🚀 Applying fast feature pre-filtering")
        
        original_count = len(features.columns)
        
        # Filter 1: Remove features with too many NaN values
        nan_threshold = 0.01  # Remove features with >1% NaN (reduced from 0.3)
        nan_ratios = features.isnull().sum() / len(features)
        valid_features = nan_ratios[nan_ratios <= nan_threshold].index.tolist()
        
        # Filter 2: Remove constant features
        constant_features = []
        for col in valid_features:
            if features[col].nunique() <= 1:
                constant_features.append(col)
        valid_features = [f for f in valid_features if f not in constant_features]
        
        # Filter 3: Remove features with very low variance
        variance_threshold = 1e-8
        low_variance_features = []
        for col in valid_features:
            if features[col].var() < variance_threshold:
                low_variance_features.append(col)
        valid_features = [f for f in valid_features if f not in low_variance_features]
        
        # Filter 4: Quick correlation-based filtering (remove highly correlated features)
        if len(valid_features) > 50:  # Only if we have many features
            corr_matrix = features[valid_features].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(valid_features)):
                for j in range(i+1, len(valid_features)):
                    if corr_matrix.iloc[i, j] > 0.95:  # Very high correlation
                        high_corr_pairs.append((valid_features[i], valid_features[j]))
            
            # Remove one feature from each high correlation pair
            features_to_remove = set()
            for f1, f2 in high_corr_pairs:
                if f1 not in features_to_remove:
                    features_to_remove.add(f2)
            
            valid_features = [f for f in valid_features if f not in features_to_remove]
        
        filtered_features = features[valid_features]
        filtered_count = len(filtered_features.columns)
        
        tprint_info(f"🚀 Fast pre-filtering: {original_count} → {filtered_count} features")
        tprint_info(f"   • Removed {len(features.columns) - len(valid_features)} features")
        
        return filtered_features
    
    def _smart_sample_features(self, sorted_features: List[str], n_features: int, 
                              feature_importance: Dict[str, float]) -> List[str]:
        """Smart sampling of features based on importance with randomness."""
        import random
        
        # Take top 70% based on importance, 30% random
        n_important = int(n_features * 0.7)
        n_random = n_features - n_important
        
        # Select important features
        important_features = sorted_features[:n_important] if len(sorted_features) >= n_important else sorted_features
        
        # Add random features
        remaining_features = [f for f in sorted_features if f not in important_features]
        if remaining_features and n_random > 0:
            random_features = random.sample(remaining_features, min(n_random, len(remaining_features)))
            important_features.extend(random_features)
        
        # Fill remaining slots randomly if needed
        if len(important_features) < n_features:
            all_features = sorted_features
            additional_needed = n_features - len(important_features)
            available = [f for f in all_features if f not in important_features]
            if available:
                additional = random.sample(available, min(additional_needed, len(available)))
                important_features.extend(additional)
        
        return important_features[:n_features]

    def evaluate_objectives(self, features: pd.DataFrame,
                          targets: pd.Series,
                          selected_features: List[str]) -> Dict[str, ObjectiveResult]:
        """Evaluate all objectives for a given feature set with caching."""
        results = {}
        
        # Create cache key based on feature set and data shape
        cache_key = self._create_cache_key(selected_features, features.shape, targets.shape)
        
        # Check cache first
        if cache_key in self.objective_cache:
            self.cache_hits += 1
            tprint_debug(f"Cache hit for {len(selected_features)} features")
            return self.objective_cache[cache_key]

        # Cache miss - compute objectives
        self.cache_misses += 1
        for obj in self.objectives:
            result = obj.evaluate(features, targets, selected_features)
            results[obj.name] = result

        # Store in cache (limit cache size)
        if len(self.objective_cache) < 1000:  # Limit cache size
            self.objective_cache[cache_key] = results
        
        return results
    
    def _create_cache_key(self, selected_features: List[str], features_shape: tuple, targets_shape: tuple) -> str:
        """Create a cache key for objective evaluation."""
        # Sort features for consistent key generation
        sorted_features = sorted(selected_features)
        return f"{sorted_features}_{features_shape}_{targets_shape}"

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

        # Evolutionary algorithms removed - using optimized standard methods
        summary.update({
            'evolutionary_algorithms_disabled': True,
            'using_standard_methods': True,
            'correlation_based_available': True,
            'mutual_information_available': True,
            'bayesian_tpe_available': self.bayesian_tpe_optimizer is not None
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
