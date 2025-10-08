"""
Knapsack Selection with Correlation Constraints

Implements integer programming for feature selection with:
- Utility maximization subject to cost and cardinality constraints
- Correlation-aware selection using partial correlations
- Family coverage requirements
- Fallback to greedy algorithm if solver fails
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from itertools import combinations
from types import SimpleNamespace
import warnings
warnings.filterwarnings('ignore')

# Local imports
from .htf_materialization import HTFFeatureGenerator, UpdateStyle
from .config import SelectionConfig

# Import tprint for enhanced logging
try:
    from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_debug
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Module level logger for helper functions
MODULE_LOGGER = logging.getLogger(__name__)


def log_info(logger: logging.Logger, message: str) -> None:
    formatted = message if isinstance(message, str) else str(message)
    logger.info(formatted)
    tprint(f"ℹ️ {formatted}")


def log_warning(logger: logging.Logger, message: str) -> None:
    formatted = message if isinstance(message, str) else str(message)
    logger.warning(formatted)
    tprint_warning(f"⚠️ {formatted}")


def log_error(logger: logging.Logger, message: str) -> None:
    formatted = message if isinstance(message, str) else str(message)
    logger.error(formatted)
    tprint_error(f"❌ {formatted}")


def log_debug(logger: logging.Logger, message: str) -> None:
    formatted = message if isinstance(message, str) else str(message)
    logger.debug(formatted)
    tprint_debug(f"🐞 {formatted}")


# Try to import optimization solvers
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    dummy_class = type('DummyCP', (), {'__init__': lambda self, *args, **kwargs: None})
    cp = SimpleNamespace(
        Variable=dummy_class,
        Constraint=dummy_class,
        sum=lambda *args, **kwargs: 0,
        CBC='CBC',
        OPTIMAL='optimal',
        Maximize=lambda *args, **kwargs: None,
    )
    CVXPY_AVAILABLE = False
    log_warning(MODULE_LOGGER, "CVXPY not available, using greedy algorithm")

try:
    from scipy.optimize import linprog
    SCIPY_OPTIMIZE_AVAILABLE = True
except ImportError:
    SCIPY_OPTIMIZE_AVAILABLE = False


@dataclass
class FeatureCandidate:
    """Represents a feature candidate for selection."""
    feature_id: str
    feature_name: str
    family: str
    utility: float
    cost: float
    lookback: int
    update_style: str
    metadata: Dict[str, Any]


@dataclass
class CrossTimeframeKnapsackSelectionResult:
    """Result of the knapsack-based resource allocation stage."""
    selected_features: List[FeatureCandidate]
    total_utility: float
    total_cost: float
    family_coverage: Dict[str, int]
    correlation_matrix: pd.DataFrame
    selection_method: str
    metadata: Dict[str, Any]


class CorrelationCalculator:
    """Calculates partial correlations between features."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        log_debug(self.logger, "Initialized CorrelationCalculator")

    def calculate_partial_correlations(self,
                                     features: List[FeatureCandidate],
                                     feature_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate partial correlations between features.
        
        Args:
            features: List of feature candidates
            feature_data: DataFrame with feature values
            
        Returns:
            Correlation matrix
        """
        log_info(
            self.logger,
            f"Calculating partial correlations for {len(features)} features with {feature_data.shape[0]} samples",
        )
        if len(features) < 2:
            log_warning(self.logger, "Insufficient features provided for correlation calculation")
            return pd.DataFrame()

        # Extract feature columns
        feature_columns = [f.feature_name for f in features if f.feature_name in feature_data.columns]

        if len(feature_columns) < 2:
            log_warning(self.logger, "Insufficient feature columns available in data for correlation calculation")
            return pd.DataFrame()

        # Calculate partial correlations
        partial_corr_matrix = self._calculate_partial_correlations_matrix(
            feature_data[feature_columns]
        )

        # Create DataFrame with feature names
        corr_df = pd.DataFrame(
            partial_corr_matrix,
            index=feature_columns,
            columns=feature_columns
        )

        log_info(self.logger, f"Generated partial correlation matrix with shape {corr_df.shape}")

        return corr_df

    def _calculate_partial_correlations_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate partial correlation matrix."""
        # Remove rows with any NaN values
        clean_data = data.dropna()

        if len(clean_data) < 10:
            # Not enough data, return identity matrix
            log_warning(self.logger, "Not enough samples for reliable partial correlations; returning identity matrix")
            return np.eye(len(data.columns))

        # Calculate correlation matrix
        corr_matrix = clean_data.corr().values

        # Calculate partial correlations
        # For partial correlation, we need to control for other variables
        n_features = len(data.columns)
        partial_corr = np.eye(n_features)
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Calculate partial correlation between i and j
                # controlling for all other variables
                other_vars = [k for k in range(n_features) if k != i and k != j]
                
                if len(other_vars) == 0:
                    # No other variables to control for
                    partial_corr[i, j] = corr_matrix[i, j]
                    partial_corr[j, i] = corr_matrix[i, j]
                else:
                    # Calculate partial correlation
                    try:
                        partial_corr[i, j] = self._partial_correlation(
                            clean_data.iloc[:, i].values,
                            clean_data.iloc[:, j].values,
                            clean_data.iloc[:, other_vars].values
                        )
                        partial_corr[j, i] = partial_corr[i, j]
                    except Exception as exc:
                        # Fallback to regular correlation
                        log_warning(
                            self.logger,
                            f"Partial correlation fallback to Pearson for feature pair ({i}, {j})",
                            exc_info=True,
                        )
                        partial_corr[i, j] = corr_matrix[i, j]
                        partial_corr[j, i] = corr_matrix[i, j]

        return partial_corr
    
    def _partial_correlation(self, 
                           x: np.ndarray, 
                           y: np.ndarray, 
                           z: np.ndarray) -> float:
        """Calculate partial correlation between x and y controlling for z."""
        try:
            # Standardize variables
            x_std = (x - np.mean(x)) / np.std(x)
            y_std = (y - np.mean(y)) / np.std(y)
            
            if z.shape[1] == 0:
                # No control variables
                return np.corrcoef(x_std, y_std)[0, 1]
            
            # Regress x and y on z
            from sklearn.linear_model import LinearRegression
            
            reg_x = LinearRegression().fit(z, x_std)
            reg_y = LinearRegression().fit(z, y_std)
            
            # Get residuals
            x_resid = x_std - reg_x.predict(z)
            y_resid = y_std - reg_y.predict(z)
            
            # Calculate correlation of residuals
            return np.corrcoef(x_resid, y_resid)[0, 1]

        except Exception as e:
            log_warning(self.logger, f"Partial correlation calculation failed: {e}")
            return 0.0


class IntegerProgramSolver:
    """Solves the knapsack problem using integer programming."""

    def __init__(self, config: SelectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        log_debug(self.logger, "Initialized IntegerProgramSolver")

    def solve_knapsack(self,
                      features: List[FeatureCandidate],
                      correlation_matrix: pd.DataFrame) -> List[FeatureCandidate]:
        """
        Solve knapsack problem with correlation constraints.
        
        Args:
            features: List of feature candidates
            correlation_matrix: Partial correlation matrix

        Returns:
            List of selected features
        """
        log_info(
            self.logger,
            f"Solving knapsack for {len(features)} features with correlation matrix shape {correlation_matrix.shape}",
        )
        if not features:
            log_warning(self.logger, "No features provided to solver; returning empty selection")
            return []

        n_features = len(features)
        if n_features == 0:
            log_warning(self.logger, "Zero-length feature list encountered; returning empty selection")
            return []

        if CVXPY_AVAILABLE:
            log_info(self.logger, "Using CVXPY solver for knapsack selection")
            return self._solve_with_cvxpy(features, correlation_matrix)
        elif SCIPY_OPTIMIZE_AVAILABLE:
            log_info(self.logger, "Using SciPy linear programming solver for knapsack selection")
            return self._solve_with_scipy(features, correlation_matrix)
        else:
            log_warning(self.logger, "No optimization solver available, using greedy algorithm")
            return self._solve_greedy(features, correlation_matrix)

    def _solve_with_cvxpy(self,
                         features: List[FeatureCandidate],
                         correlation_matrix: pd.DataFrame) -> List[FeatureCandidate]:
        """Solve using CVXPY."""
        try:
            n_features = len(features)
            
            # Decision variables
            x = cp.Variable(n_features, boolean=True)
            
            # Objective: maximize utility
            utilities = np.array([f.utility for f in features])
            objective = cp.Maximize(utilities @ x)
            
            # Constraints
            constraints = []
            
            # Cost constraint
            costs = np.array([f.cost for f in features])
            constraints.append(costs @ x <= self.config.max_cost_ms)
            
            # Cardinality constraint
            constraints.append(cp.sum(x) <= self.config.max_features)
            
            # Family coverage constraints
            family_constraints = self._create_family_constraints(features, x)
            constraints.extend(family_constraints)
            
            # Correlation constraints
            correlation_constraints = self._create_correlation_constraints(
                features, correlation_matrix, x
            )
            constraints.extend(correlation_constraints)
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.CBC, verbose=False)

            if problem.status == cp.OPTIMAL:
                selected_indices = np.where(x.value > 0.5)[0]
                log_success_message = (
                    f"CVXPY solver optimal with {len(selected_indices)} features selected"
                )
                log_info(self.logger, log_success_message)
                tprint_success(f"✅ {log_success_message}")
                return [features[i] for i in selected_indices]
            else:
                log_warning(self.logger, f"CVXPY optimization failed: {problem.status}")
                return self._solve_greedy(features, correlation_matrix)

        except Exception as e:
            log_warning(self.logger, f"CVXPY solver failed: {e}")
            return self._solve_greedy(features, correlation_matrix)

    def _solve_with_scipy(self,
                         features: List[FeatureCandidate],
                         correlation_matrix: pd.DataFrame) -> List[FeatureCandidate]:
        """Solve using SciPy linear programming (relaxed)."""
        try:
            n_features = len(features)
            
            # Objective: maximize utility (minimize negative utility)
            c = -np.array([f.utility for f in features])
            
            # Constraints: A_ub @ x <= b_ub
            A_ub = []
            b_ub = []
            
            # Cost constraint
            A_ub.append([f.cost for f in features])
            b_ub.append(self.config.max_cost_ms)
            
            # Cardinality constraint
            A_ub.append([1] * n_features)
            b_ub.append(self.config.max_features)
            
            # Bounds: 0 <= x <= 1
            bounds = [(0, 1)] * n_features
            
            # Solve relaxed problem
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

            if result.success:
                # Round solution to integers
                x_values = result.x
                selected_indices = np.where(x_values > 0.5)[0]
                log_info(
                    self.logger,
                    f"SciPy solver produced relaxed solution with {len(selected_indices)} selected features",
                )
                return [features[i] for i in selected_indices]
            else:
                log_warning(self.logger, f"SciPy optimization failed: {result.message}")
                return self._solve_greedy(features, correlation_matrix)

        except Exception as e:
            log_warning(self.logger, f"SciPy solver failed: {e}")
            return self._solve_greedy(features, correlation_matrix)

    def _create_family_constraints(self,
                                 features: List[FeatureCandidate],
                                 x: cp.Variable) -> List[cp.Constraint]:
        """Create family coverage constraints."""
        constraints = []

        # Group features by family
        family_groups = {}
        for i, feature in enumerate(features):
            family = feature.family
            if family not in family_groups:
                family_groups[family] = []
            family_groups[family].append(i)

        # Require at least one feature from each family
        for family, indices in family_groups.items():
            if indices:  # Only if family has features
                family_vars = [x[i] for i in indices]
                log_debug(
                    self.logger,
                    f"Adding family coverage constraint for {family} with {len(indices)} candidates",
                )
                constraints.append(cp.sum(family_vars) >= 1)

        return constraints
    
    def _create_correlation_constraints(self, 
                                      features: List[FeatureCandidate],
                                      correlation_matrix: pd.DataFrame,
                                      x: cp.Variable) -> List[cp.Constraint]:
        """Create correlation constraints."""
        constraints = []
        
        # Find highly correlated pairs
        feature_names = [f.feature_name for f in features]

        for i, name1 in enumerate(feature_names):
            for j, name2 in enumerate(feature_names):
                if i < j and name1 in correlation_matrix.index and name2 in correlation_matrix.columns:
                    corr = abs(correlation_matrix.loc[name1, name2])
                    if corr > self.config.max_correlation:
                        # Add constraint: x[i] + x[j] <= 1
                        log_info(
                            self.logger,
                            f"Applying correlation constraint between {name1} and {name2} (|ρ|={corr:.3f} > {self.config.max_correlation:.2f})",
                        )
                        constraints.append(x[i] + x[j] <= 1)

        return constraints
    
    def _solve_greedy(self, 
                     features: List[FeatureCandidate],
                     correlation_matrix: pd.DataFrame) -> List[FeatureCandidate]:
        """Solve using greedy algorithm."""
        # Sort features by utility/cost ratio
        features_with_ratio = []
        for feature in features:
            ratio = feature.utility / max(feature.cost, 0.001)
            features_with_ratio.append((feature, ratio))
        
        features_with_ratio.sort(key=lambda x: x[1], reverse=True)

        selected_features = []
        total_cost = 0.0
        selected_names = set()
        
        for feature, ratio in features_with_ratio:
            # Check cost constraint
            if total_cost + feature.cost > self.config.max_cost_ms:
                continue
            
            # Check cardinality constraint
            if len(selected_features) >= self.config.max_features:
                break

            # Check correlation constraint
            if self._violates_correlation_constraint(feature, selected_features, correlation_matrix):
                continue

            # Add feature
            selected_features.append(feature)
            total_cost += feature.cost
            selected_names.add(feature.feature_name)

        log_info(
            self.logger,
            f"Greedy solver selected {len(selected_features)} features with total cost {total_cost:.2f} ms",
        )
        return selected_features

    def _violates_correlation_constraint(self,
                                       feature: FeatureCandidate,
                                       selected_features: List[FeatureCandidate],
                                       correlation_matrix: pd.DataFrame) -> bool:
        """Check if adding feature violates correlation constraint."""
        if feature.feature_name not in correlation_matrix.index:
            return False

        for selected in selected_features:
            if selected.feature_name in correlation_matrix.columns:
                corr = abs(correlation_matrix.loc[feature.feature_name, selected.feature_name])
                if corr > self.config.max_correlation:
                    log_info(
                        self.logger,
                        f"Skipping {feature.feature_name} due to correlation {corr:.3f} with {selected.feature_name} (threshold {self.config.max_correlation:.2f})",
                    )
                    return True

        return False


class KnapsackSelection:
    """Main knapsack selection system."""

    def __init__(self, config: SelectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        log_debug(self.logger, "Initialized KnapsackSelection")

        self.correlation_calculator = CorrelationCalculator()
        self.solver = IntegerProgramSolver(config)
        self.htf_generator = HTFFeatureGenerator(config)
        self._series_cache: Dict[str, pd.Series] = {}

    def select_features(self,
                       phase2_results: Dict[str, Any],
                       ehu_rih_assignments: List[Any],
                       sessionized_data: Optional[Dict[str, Any]] = None) -> CrossTimeframeKnapsackSelectionResult:
        """
        Select features using knapsack optimization.
        
        Args:
            phase2_results: Phase-2 optimization results
            ehu_rih_assignments: EHU/RIH assignment decisions

        Returns:
            Selection result with selected features
        """
        log_info(self.logger, "Starting knapsack selection workflow")
        tprint("🎯 Starting knapsack feature selection")
        tprint(f"   → Phase2 results: {len(phase2_results)} items")
        tprint(f"   → EHU/RIH assignments: {len(ehu_rih_assignments)} items")

        # Create feature candidates
        candidates = self._create_feature_candidates(phase2_results, ehu_rih_assignments)
        tprint(f"   → Created {len(candidates)} feature candidates")

        if not candidates:
            log_warning(self.logger, "No feature candidates available")
            tprint_warning("⚠️ No feature candidates available for selection")
            return CrossTimeframeKnapsackSelectionResult(
                selected_features=[],
                total_utility=0.0,
                total_cost=0.0,
                family_coverage={},
                correlation_matrix=pd.DataFrame(),
                selection_method="none",
                metadata={}
            )
        
        # Calculate correlations (if data available)
        correlation_matrix = self._calculate_correlations(
            candidates,
            phase2_results,
            sessionized_data,
        )
        
        # Solve knapsack problem
        selected_features = self.solver.solve_knapsack(candidates, correlation_matrix)
        
        # Calculate results
        total_utility = sum(f.utility for f in selected_features)
        total_cost = sum(f.cost for f in selected_features)
        
        # Calculate family coverage
        family_coverage = {}
        for feature in selected_features:
            family = feature.family
            family_coverage[family] = family_coverage.get(family, 0) + 1
        
        # Create selection result
        result = CrossTimeframeKnapsackSelectionResult(
            selected_features=selected_features,
            total_utility=total_utility,
            total_cost=total_cost,
            family_coverage=family_coverage,
            correlation_matrix=correlation_matrix,
            selection_method="integer_programming" if CVXPY_AVAILABLE else "greedy",
            metadata={
                'total_candidates': len(candidates),
                'selection_ratio': len(selected_features) / len(candidates),
                'constraints': {
                    'max_cost': self.config.max_cost_ms,
                    'max_features': self.config.max_features,
                    'max_correlation': self.config.max_correlation
                }
            }
        )

        log_info(
            self.logger,
            f"Knapsack selection completed with {len(selected_features)} features (utility={total_utility:.4f}, cost={total_cost:.2f})",
        )
        tprint_success(
            f"🏁 Selection complete → {len(selected_features)} features | utility={total_utility:.4f} | cost={total_cost:.2f}"
        )
        return result

    def _create_feature_candidates(self,
                                 phase2_results: Dict[str, Any],
                                 ehu_rih_assignments: List[Any]) -> List[FeatureCandidate]:
        """Create feature candidates from phase2 results and assignments."""
        candidates = []

        optimized_features = phase2_results.get('optimized_features', [])
        assignment_map = {a.feature_name: a for a in ehu_rih_assignments}
        log_info(
            self.logger,
            f"Building feature candidates from {len(optimized_features)} optimized features and {len(assignment_map)} assignments",
        )

        for feature in optimized_features:
            assignment = assignment_map.get(feature.feature_name)
            if not assignment:
                log_debug(self.logger, f"Skipping {feature.feature_name} with no assignment mapping")
                continue

            adaptive_score = getattr(feature, 'adaptive_score', None)
            if isinstance(adaptive_score, dict):
                utility = adaptive_score.get('utility_score', feature.optimal_ic)
            else:
                adaptive_score = None
                utility = feature.optimal_ic

            # Calculate cost
            cost = assignment.cost_per_ms * feature.optimal_lookback

            candidate = FeatureCandidate(
                feature_id=f"{feature.feature_name}_{feature.optimal_lookback}",
                feature_name=feature.feature_name,
                family=feature.family,
                utility=utility,
                cost=cost,
                lookback=feature.optimal_lookback,
                update_style=assignment.update_style.value,
                metadata={
                    'optimal_ic': feature.optimal_ic,
                    'confidence_interval': feature.confidence_interval,
                    'export_type': feature.export_type,
                    'blend_weights': feature.blend_weights,
                    'adaptive_score': adaptive_score,
                }
            )

            candidates.append(candidate)

        return candidates

    def _calculate_correlations(
        self,
        candidates: List[FeatureCandidate],
        phase2_results: Optional[Dict[str, Any]] = None,
        sessionized_data: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Calculate partial correlations between candidates."""

        if len(candidates) < 2:
            log_warning(self.logger, "Less than two candidates available; skipping correlation calculation")
            return pd.DataFrame()

        aligned_data = self._extract_aligned_data(sessionized_data)
        feature_series: Dict[str, pd.Series] = {}
        missing_series: List[str] = []

        for candidate in candidates:
            series = self._get_candidate_series(candidate, phase2_results, aligned_data)
            if isinstance(series, pd.Series) and not series.empty:
                feature_series[candidate.feature_name] = series.sort_index()
            else:
                missing_series.append(candidate.feature_name)

        if len(feature_series) < 2:
            if missing_series:
                log_warning(
                    self.logger,
                    f"Insufficient feature series for correlation calculation; missing {len(missing_series)}/{len(candidates)} candidates",
                )
            return pd.DataFrame()

        feature_df = pd.DataFrame(feature_series)
        feature_df = feature_df.dropna(how='all')

        if feature_df.empty or feature_df.shape[1] < 2:
            log_warning(self.logger, "Feature DataFrame is empty or lacks sufficient columns after cleaning")
            return pd.DataFrame()

        min_samples = getattr(self.config, "min_samples_for_correlation", 30)
        valid_columns = [
            column
            for column in feature_df.columns
            if feature_df[column].dropna().shape[0] >= min_samples
        ]

        if len(valid_columns) < 2:
            log_warning(
                self.logger,
                f"Not enough overlapping samples to compute correlations (required={min_samples})",
            )
            return pd.DataFrame()

        feature_df = feature_df[valid_columns].sort_index()

        available_candidates = [
            candidate for candidate in candidates if candidate.feature_name in feature_df.columns
        ]

        log_info(
            self.logger,
            f"Computing correlations across {len(available_candidates)} candidates with {feature_df.shape[0]} samples",
        )
        correlation_matrix = self.correlation_calculator.calculate_partial_correlations(
            available_candidates,
            feature_df,
        )

        if correlation_matrix.empty:
            log_warning(self.logger, "Correlation matrix calculation returned empty result")
            return correlation_matrix

        # Ensure ordering matches the subset of candidates with data
        ordered_names = [candidate.feature_name for candidate in available_candidates]
        correlation_matrix = correlation_matrix.reindex(index=ordered_names, columns=ordered_names)
        log_info(self.logger, f"Correlation matrix aligned to candidate order with shape {correlation_matrix.shape}")

        return correlation_matrix

    def _extract_aligned_data(
        self, sessionized_data: Optional[Dict[str, Any]]
    ) -> Optional[pd.DataFrame]:
        """Extract aligned base timeframe data if available."""

        if sessionized_data is None:
            log_warning(self.logger, "No sessionized data provided for correlation alignment")
            return None

        if isinstance(sessionized_data, pd.DataFrame):
            log_info(self.logger, "Using provided DataFrame for aligned data")
            return sessionized_data

        if isinstance(sessionized_data, dict):
            aligned = sessionized_data.get('aligned_data')
            if isinstance(aligned, pd.DataFrame):
                log_info(self.logger, "Extracted aligned DataFrame from sessionized data dictionary")
                return aligned

        log_warning(self.logger, "Unable to extract aligned data from sessionized input")
        return None

    def _get_candidate_series(
        self,
        candidate: FeatureCandidate,
        phase2_results: Optional[Dict[str, Any]],
        aligned_data: Optional[pd.DataFrame],
    ) -> Optional[pd.Series]:
        """Fetch or materialize the time series for a candidate."""

        cache_key = candidate.feature_id
        if cache_key in self._series_cache:
            log_debug(self.logger, f"Using cached series for {candidate.feature_name}")
            return self._series_cache[cache_key]

        metadata_series = candidate.metadata.get('feature_series') if candidate.metadata else None
        if isinstance(metadata_series, pd.Series):
            self._series_cache[cache_key] = metadata_series
            log_debug(self.logger, f"Loaded series for {candidate.feature_name} from metadata cache")
            return metadata_series

        if phase2_results:
            for cache_key_name in ('feature_series', 'feature_series_cache', 'series_cache'):
                cache = phase2_results.get(cache_key_name)
                if isinstance(cache, dict):
                    series = cache.get(candidate.feature_name)
                    if isinstance(series, pd.Series):
                        self._series_cache[cache_key] = series
                        log_debug(self.logger, f"Loaded series for {candidate.feature_name} from phase2 cache {cache_key_name}")
                        return series

        if aligned_data is not None:
            try:
                update_style = UpdateStyle(candidate.update_style.lower()) if isinstance(candidate.update_style, str) else UpdateStyle.EHU
            except ValueError:
                update_style = UpdateStyle.EHU

            try:
                materialized = self.htf_generator.generate_htf_feature(
                    aligned_data,
                    candidate.feature_name,
                    candidate.family,
                    candidate.lookback,
                    update_style,
                )
                series = materialized.feature_series
                if isinstance(series, pd.Series):
                    self._series_cache[cache_key] = series
                    log_debug(self.logger, f"Materialized series for {candidate.feature_name} via HTF generator")
                    return series
            except Exception as exc:  # pragma: no cover - defensive logging
                log_warning(
                    self.logger,
                    f"Failed to materialize series for {candidate.feature_name} (lookback={candidate.lookback}): {exc}",
                )

        log_debug(self.logger, f"Series unavailable for {candidate.feature_name}")
        return None

    def get_selection_summary(self, result: CrossTimeframeKnapsackSelectionResult) -> Dict[str, Any]:
        """Get summary of selection results."""
        summary = {
            'total_selected': len(result.selected_features),
            'total_utility': result.total_utility,
            'total_cost': result.total_cost,
            'avg_utility': result.total_utility / len(result.selected_features) if result.selected_features else 0,
            'avg_cost': result.total_cost / len(result.selected_features) if result.selected_features else 0,
            'family_coverage': result.family_coverage,
            'selection_method': result.selection_method,
            'constraint_utilization': {
                'cost_utilization': result.total_cost / self.config.max_cost_ms,
                'feature_utilization': len(result.selected_features) / self.config.max_features
            }
        }
        log_info(
            self.logger,
            "Selection summary prepared: "
            f"{summary['total_selected']} features, cost utilization {summary['constraint_utilization']['cost_utilization']:.2%}, "
            f"feature utilization {summary['constraint_utilization']['feature_utilization']:.2%}",
        )
        tprint(
            "📊 Selection summary → "
            f"features={summary['total_selected']}, "
            f"utility={summary['total_utility']:.4f}, "
            f"cost={summary['total_cost']:.2f}, "
            f"method={summary['selection_method']}"
        )
        return summary
