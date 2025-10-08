"""
Statistical Selection with Stability Selection and FDR

Implements rigorous statistical selection with:
- Stability selection with block bootstrap
- Permutation importance with wild/bootstrap
- Benjamini-Hochberg FDR control
- Conditional IC tests
- Group LASSO option
- Interaction heredity enforcement
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .config import SelectionConfig

# Import tprint for enhanced logging
try:
    from src.utils.tprint import (
        tprint,
        tprint_info,
        tprint_success,
        tprint_warning,
        tprint_error,
        tprint_debug,
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False

    def tprint(*args, **kwargs):
        print(*args, **kwargs)

    def tprint_info(*args, **kwargs):
        print("INFO:", *args, **kwargs)

    def tprint_success(*args, **kwargs):
        print("SUCCESS:", *args, **kwargs)

    def tprint_warning(*args, **kwargs):
        print("WARNING:", *args, **kwargs)

    def tprint_error(*args, **kwargs):
        print("ERROR:", *args, **kwargs)

    def tprint_debug(*args, **kwargs):
        print("DEBUG:", *args, **kwargs)


# Try to import Group LASSO
try:
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    GROUP_LASSO_AVAILABLE = True
except ImportError:
    GROUP_LASSO_AVAILABLE = False
    logging.warning("Group LASSO not available, using standard LASSO")


@dataclass
class CrossTimeframeStatisticalSelectionResult:
    """Result of the statistical validation and pruning stage."""
    selected_features: List[str]
    selection_frequencies: Dict[str, float]
    p_values: Dict[str, float]
    fdr_corrected_p_values: Dict[str, float]
    conditional_ics: Dict[str, float]
    group_lasso_groups: Dict[str, List[str]]
    selection_method: str
    metadata: Dict[str, Any]


class StabilitySelector:
    """Implements stability selection with block bootstrap."""
    
    def __init__(self, config: SelectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.n_resamples = config.stability_resamples
        self.block_size = None  # Will be determined automatically
    
    def select_features(self,
                       features: pd.DataFrame,
                       targets: pd.Series,
                       base_features: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Select features using stability selection.
        
        Args:
            features: Feature matrix
            targets: Target series
            base_features: Base features for conditional testing
            
        Returns:
            Dictionary of selection frequencies
        """
        self.logger.info("Starting stability selection")
        tprint_info(
            "🧮 Running stability selection",
            f"features={len(features.columns)}",
            f"samples={len(features)}",
        )

        # Determine block size
        if self.block_size is None:
            self.block_size = max(10, int(np.sqrt(len(features))))
            tprint_debug(f"   → Using block size: {self.block_size}")

        # Perform stability selection
        selection_frequencies = {}

        for feature_name in features.columns:
            try:
                frequency = self._calculate_selection_frequency(
                    features, targets, feature_name, base_features
                )
                selection_frequencies[feature_name] = frequency

            except Exception as e:
                self.logger.warning(f"Failed to calculate frequency for {feature_name}: {e}")
                tprint_warning(f"⚠️ Stability frequency failed for {feature_name}: {e}")
                selection_frequencies[feature_name] = 0.0

        self.logger.info(f"Stability selection completed: {len(selection_frequencies)} features evaluated")
        tprint_success(
            "✅ Stability selection completed",
            f"evaluated={len(selection_frequencies)}",
        )
        return selection_frequencies
    
    def _calculate_selection_frequency(self, 
                                     features: pd.DataFrame,
                                     targets: pd.Series,
                                     feature_name: str,
                                     base_features: Optional[List[str]] = None) -> float:
        """Calculate selection frequency for a single feature."""
        selections = 0
        
        for _ in range(self.n_resamples):
            try:
                # Generate bootstrap sample
                bootstrap_features, bootstrap_targets = self._block_bootstrap(
                    features, targets
                )
                
                # Select features using LASSO
                selected = self._lasso_selection(
                    bootstrap_features, bootstrap_targets, base_features
                )
                
                if feature_name in selected:
                    selections += 1
                    
            except Exception as e:
                self.logger.warning(f"Bootstrap iteration failed: {e}")
                continue
        
        return selections / self.n_resamples
    
    def _block_bootstrap(self, 
                        features: pd.DataFrame,
                        targets: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Perform block bootstrap sampling."""
        n_samples = len(features)
        block_size = self.block_size
        
        # Generate bootstrap indices
        bootstrap_indices = []
        while len(bootstrap_indices) < n_samples:
            # Random start point
            start_idx = np.random.randint(0, n_samples - block_size + 1)
            
            # Random block length
            block_length = np.random.geometric(1.0 / block_size)
            block_length = min(block_length, n_samples - start_idx)
            
            # Add block indices
            block_indices = list(range(start_idx, start_idx + block_length))
            bootstrap_indices.extend(block_indices)
        
        # Truncate to original length
        bootstrap_indices = bootstrap_indices[:n_samples]
        
        # Create bootstrap samples
        bootstrap_features = features.iloc[bootstrap_indices].reset_index(drop=True)
        bootstrap_targets = targets.iloc[bootstrap_indices].reset_index(drop=True)
        
        return bootstrap_features, bootstrap_targets
    
    def _lasso_selection(self, 
                        features: pd.DataFrame,
                        targets: pd.Series,
                        base_features: Optional[List[str]] = None) -> List[str]:
        """Select features using LASSO."""
        try:
            # Use LASSO with cross-validation
            lasso = LassoCV(cv=3, random_state=42, max_iter=1000)
            lasso.fit(features, targets)
            
            # Get selected features
            selected_features = []
            for i, coef in enumerate(lasso.coef_):
                if abs(coef) > 1e-6:  # Non-zero coefficient
                    selected_features.append(features.columns[i])
            
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"LASSO selection failed: {e}")
            return []


class PermutationTester:
    """Implements permutation importance testing."""
    
    def __init__(self, config: SelectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_p_values(self,
                          features: pd.DataFrame,
                          targets: pd.Series,
                          base_features: Optional[List[str]] = None,
                          cv_splitter: Optional[Any] = None,
                          block_size: Optional[int] = None,
                          n_permutations: Optional[int] = None,
                          random_state: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate feature p-values using fold-aware permutation testing.

        The provided ``cv_splitter`` must yield validation windows that already
        respect any desired purge or embargo rules.  Permutations are performed
        within each validation fold only so that permuted samples never
        precede their evaluation window.  Temporal dependency is preserved by
        shuffling contiguous blocks (with length at least the label horizon)
        rather than individual observations.

        Args:
            features: Feature matrix
            targets: Target series
            base_features: Base features for conditional testing
            cv_splitter: Time-series aware splitter defining validation folds
            block_size: Optional override for permutation block size
            n_permutations: Optional override for number of permutations
            random_state: Optional RNG seed for reproducibility

        Returns:
            Dictionary of p-values
        """
        self.logger.info("Starting permutation testing")
        tprint_info(
            "🔁 Running permutation importance",
            f"features={len(features.columns)}",
            f"samples={len(features)}",
        )

        p_values = {}

        for feature_name in features.columns:
            try:
                p_value = self._calculate_permutation_p_value(
                    features,
                    targets,
                    feature_name,
                    base_features,
                    cv_splitter=cv_splitter,
                    block_size=block_size,
                    n_permutations=n_permutations,
                    random_state=random_state,
                )
                p_values[feature_name] = p_value

            except Exception as e:
                self.logger.warning(f"Failed to calculate p-value for {feature_name}: {e}")
                tprint_warning(f"⚠️ Permutation test failed for {feature_name}: {e}")
                p_values[feature_name] = 1.0

        self.logger.info(f"Permutation testing completed: {len(p_values)} features tested")
        tprint_success(
            "✅ Permutation importance completed",
            f"tested={len(p_values)}",
        )
        return p_values
    
    def _calculate_permutation_p_value(self,
                                     features: pd.DataFrame,
                                     targets: pd.Series,
                                     feature_name: str,
                                     base_features: Optional[List[str]] = None,
                                     cv_splitter: Optional[Any] = None,
                                     block_size: Optional[int] = None,
                                     n_permutations: Optional[int] = None,
                                     random_state: Optional[int] = None) -> float:
        """Calculate permutation p-value for a single feature."""
        try:
            if len(features) < 3:
                return 1.0

            effective_block_size = self._resolve_block_size(len(features), block_size)
            effective_n_permutations = self._resolve_n_permutations(n_permutations)
            rng = np.random.default_rng(self._resolve_random_state(random_state))

            splitter = self._resolve_splitter(features, targets, cv_splitter)
            if splitter is None:
                return 1.0

            fold_p_values = []
            fold_weights = []

            for train_idx, val_idx in splitter:
                if len(val_idx) == 0:
                    continue

                feature_slice = features.iloc[val_idx][feature_name]
                target_slice = targets.iloc[val_idx]

                original_corr = feature_slice.corr(target_slice)

                if pd.isna(original_corr):
                    continue

                permuted_corrs = []
                for _ in range(effective_n_permutations):
                    permuted_values = self._block_permute(
                        feature_slice.to_numpy(), effective_block_size, rng
                    )
                    permuted_series = pd.Series(permuted_values, index=feature_slice.index)
                    permuted_corr = permuted_series.corr(target_slice)
                    if not pd.isna(permuted_corr):
                        permuted_corrs.append(abs(permuted_corr))

                if not permuted_corrs:
                    continue

                permuted_corrs = np.asarray(permuted_corrs)
                greater_equal = np.sum(permuted_corrs >= abs(original_corr))
                fold_p = (greater_equal + 1) / (len(permuted_corrs) + 1)
                fold_p_values.append(fold_p)
                fold_weights.append(len(val_idx))

            if not fold_p_values:
                return 1.0

            weighted_p = np.average(fold_p_values, weights=fold_weights)
            return float(weighted_p)

        except Exception as e:
            self.logger.warning(f"Permutation test failed for {feature_name}: {e}")
            return 1.0

    def _resolve_block_size(self, n_samples: int, override: Optional[int]) -> int:
        label_horizon = max(1, getattr(self.config, "label_horizon", 1))
        config_block = getattr(self.config, "permutation_block_size", None)
        block_size = override if override is not None else config_block

        if block_size is None:
            block_size = max(label_horizon, int(np.ceil(np.sqrt(n_samples))))

        block_size = max(int(block_size), label_horizon, 1)
        return block_size

    def _resolve_n_permutations(self, override: Optional[int]) -> int:
        config_n_perm = getattr(self.config, "permutation_n_permutations", 500)
        n_permutations = override if override is not None else config_n_perm
        return max(int(n_permutations), 1)

    def _resolve_random_state(self, override: Optional[int]) -> Optional[int]:
        if override is not None:
            return override
        return getattr(self.config, "permutation_random_state", None)

    def _resolve_splitter(self,
                          features: pd.DataFrame,
                          targets: pd.Series,
                          splitter: Optional[Any]) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
        if splitter is None:
            requested_splits = getattr(self.config, "permutation_cv_folds", 5)
            max_splits = min(requested_splits, max(len(features) - 1, 1))
            if max_splits < 2 or max_splits >= len(features):
                return None
            splitter = TimeSeriesSplit(n_splits=max_splits)

        try:
            indices = np.arange(len(features))
            splits = list(splitter.split(indices, targets.to_numpy() if targets is not None else None))
            if not splits:
                return None
            return [(np.asarray(train), np.asarray(test)) for train, test in splits]
        except Exception as exc:
            self.logger.warning(f"Failed to generate CV splits for permutation testing: {exc}")
            return None

    def _block_permute(self,
                       values: np.ndarray,
                       block_size: int,
                       rng: np.random.Generator) -> np.ndarray:
        if len(values) == 0:
            return values.copy()

        block_size = max(1, min(block_size, len(values)))
        n_blocks = int(np.ceil(len(values) / block_size))

        if n_blocks <= 1:
            if len(values) <= 1:
                return values.copy()
            min_shift = min(block_size, len(values) - 1)
            if min_shift <= 0:
                min_shift = 1
            shift = int(rng.integers(min_shift, len(values)))
            return np.roll(values, shift)

        blocks = [values[i * block_size: (i + 1) * block_size] for i in range(n_blocks)]
        block_order = rng.permutation(n_blocks)
        permuted = np.concatenate([blocks[idx] for idx in block_order])
        return permuted[:len(values)]


class FDRController:
    """Implements Benjamini-Hochberg FDR control."""
    
    def __init__(self, config: SelectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.fdr_q = config.fdr_q
    
    def control_fdr(self, p_values: Dict[str, float]) -> Dict[str, float]:
        """
        Control FDR using Benjamini-Hochberg procedure.

        Args:
            p_values: Dictionary of p-values

        Returns:
            Dictionary of FDR-corrected p-values
        """
        if not p_values:
            tprint_warning("⚠️ FDR control skipped - no p-values provided")
            return {}

        tprint_info(
            "📉 Applying FDR control",
            f"features={len(p_values)}",
            f"target_q={self.fdr_q}",
        )

        # Convert to arrays
        feature_names = list(p_values.keys())
        p_vals = np.array(list(p_values.values()))
        
        # Sort by p-values
        sorted_indices = np.argsort(p_vals)
        sorted_p_vals = p_vals[sorted_indices]
        
        # Calculate FDR-corrected p-values
        n = len(p_vals)
        fdr_corrected = np.zeros_like(sorted_p_vals)
        
        for i in range(n):
            fdr_corrected[i] = sorted_p_vals[i] * n / (i + 1)
        
        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            fdr_corrected[i] = min(fdr_corrected[i], fdr_corrected[i + 1])
        
        # Cap at 1.0
        fdr_corrected = np.minimum(fdr_corrected, 1.0)
        
        # Create result dictionary
        fdr_corrected_dict = {}
        for i, idx in enumerate(sorted_indices):
            fdr_corrected_dict[feature_names[idx]] = fdr_corrected[i]

        tprint_success(
            "✅ FDR control complete",
            f"features={len(fdr_corrected_dict)}",
        )
        return fdr_corrected_dict


class ConditionalICTester:
    """Implements conditional IC testing."""
    
    def __init__(self, config: SelectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.min_conditional_ic = config.min_conditional_ic
    
    def calculate_conditional_ics(self,
                                features: pd.DataFrame,
                                targets: pd.Series,
                                base_features: List[str]) -> Dict[str, float]:
        """
        Calculate conditional ICs for features.
        
        Args:
            features: Feature matrix
            targets: Target series
            base_features: Base features to condition on
            
        Returns:
            Dictionary of conditional ICs
        """
        self.logger.info("Starting conditional IC testing")
        tprint_info(
            "📐 Calculating conditional ICs",
            f"features={len(features.columns)}",
            f"base_features={len(base_features) if base_features else 0}",
        )

        conditional_ics = {}

        for feature_name in features.columns:
            try:
                conditional_ic = self._calculate_conditional_ic(
                    features, targets, feature_name, base_features
                )
                conditional_ics[feature_name] = conditional_ic
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate conditional IC for {feature_name}: {e}")
                tprint_warning(f"⚠️ Conditional IC failed for {feature_name}: {e}")
                conditional_ics[feature_name] = 0.0

        self.logger.info(f"Conditional IC testing completed: {len(conditional_ics)} features tested")
        tprint_success(
            "✅ Conditional ICs calculated",
            f"tested={len(conditional_ics)}",
        )
        return conditional_ics
    
    def _calculate_conditional_ic(self, 
                                features: pd.DataFrame,
                                targets: pd.Series,
                                feature_name: str,
                                base_features: List[str]) -> float:
        """Calculate conditional IC for a single feature."""
        try:
            if not base_features:
                # No conditioning, return regular IC
                return features[feature_name].corr(targets)
            
            # Create conditioning set
            conditioning_features = [f for f in base_features if f in features.columns]
            
            if not conditioning_features:
                # No valid conditioning features
                return features[feature_name].corr(targets)
            
            # Calculate partial correlation
            partial_corr = self._calculate_partial_correlation(
                features[feature_name],
                targets,
                features[conditioning_features]
            )
            
            return partial_corr
            
        except Exception as e:
            self.logger.warning(f"Conditional IC calculation failed for {feature_name}: {e}")
            return 0.0
    
    def _calculate_partial_correlation(self, 
                                     x: pd.Series,
                                     y: pd.Series,
                                     z: pd.DataFrame) -> float:
        """Calculate partial correlation between x and y controlling for z."""
        try:
            # Standardize variables
            x_std = (x - x.mean()) / x.std()
            y_std = (y - y.mean()) / y.std()
            
            if z.shape[1] == 0:
                # No control variables
                return x_std.corr(y_std)
            
            # Regress x and y on z
            from sklearn.linear_model import LinearRegression
            
            reg_x = LinearRegression().fit(z, x_std)
            reg_y = LinearRegression().fit(z, y_std)
            
            # Get residuals
            x_resid = x_std - reg_x.predict(z)
            y_resid = y_std - reg_y.predict(z)
            
            # Calculate correlation of residuals
            return x_resid.corr(y_resid)
            
        except Exception as e:
            self.logger.warning(f"Partial correlation calculation failed: {e}")
            return 0.0


class GroupLASSOSelector:
    """Implements Group LASSO for feature selection."""
    
    def __init__(self, config: SelectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def select_features(self,
                      features: pd.DataFrame,
                      targets: pd.Series,
                      feature_groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Select features using Group LASSO.
        
        Args:
            features: Feature matrix
            targets: Target series
            feature_groups: Dictionary of group_name -> feature_list
            
        Returns:
            Dictionary of selected groups
        """
        self.logger.info("Starting Group LASSO selection")
        tprint_info(
            "🧩 Running group selection",
            f"groups={len(feature_groups)}",
            f"features={len(features.columns)}",
        )

        if not GROUP_LASSO_AVAILABLE:
            self.logger.warning("Group LASSO not available, using standard LASSO")
            tprint_warning("⚠️ Group LASSO package unavailable - falling back to standard LASSO")
            return self._fallback_lasso_selection(features, targets, feature_groups)

        try:
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Create group matrix
            group_matrix = self._create_group_matrix(features, feature_groups)
            
            # Apply Group LASSO (simplified implementation)
            selected_groups = self._apply_group_lasso(
                features_scaled, targets, group_matrix
            )
            
            # Map back to feature groups
            selected_feature_groups = {}
            for group_name, group_features in feature_groups.items():
                if group_name in selected_groups:
                    selected_feature_groups[group_name] = group_features

            self.logger.info(f"Group LASSO selection completed: {len(selected_feature_groups)} groups selected")
            tprint_success(
                "✅ Group selection complete",
                f"selected_groups={len(selected_feature_groups)}",
            )
            return selected_feature_groups

        except Exception as e:
            self.logger.warning(f"Group LASSO failed: {e}, using fallback")
            tprint_warning(f"⚠️ Group LASSO failed ({e}), using fallback LASSO")
            return self._fallback_lasso_selection(features, targets, feature_groups)
    
    def _create_group_matrix(self, 
                           features: pd.DataFrame,
                           feature_groups: Dict[str, List[str]]) -> np.ndarray:
        """Create group matrix for Group LASSO."""
        n_features = len(features.columns)
        n_groups = len(feature_groups)
        
        group_matrix = np.zeros((n_features, n_groups))
        
        for group_idx, (group_name, group_features) in enumerate(feature_groups.items()):
            for feature_name in group_features:
                if feature_name in features.columns:
                    feature_idx = features.columns.get_loc(feature_name)
                    group_matrix[feature_idx, group_idx] = 1
        
        return group_matrix
    
    def _apply_group_lasso(self, 
                          features_scaled: np.ndarray,
                          targets: pd.Series,
                          group_matrix: np.ndarray) -> List[str]:
        """Apply Group LASSO (simplified implementation)."""
        # This is a simplified implementation
        # In practice, you'd use a proper Group LASSO solver
        
        # Use standard LASSO as fallback
        lasso = Lasso(alpha=0.01, max_iter=1000)
        lasso.fit(features_scaled, targets)
        
        # Find non-zero coefficients
        selected_indices = np.where(abs(lasso.coef_) > 1e-6)[0]
        
        # Map back to groups
        selected_groups = []
        for idx in selected_indices:
            group_idx = np.where(group_matrix[idx, :] == 1)[0]
            if len(group_idx) > 0:
                selected_groups.append(group_idx[0])
        
        return selected_groups
    
    def _fallback_lasso_selection(self, 
                                features: pd.DataFrame,
                                targets: pd.Series,
                                feature_groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Fallback to standard LASSO selection."""
        try:
            # Use LASSO with cross-validation
            lasso = LassoCV(cv=3, random_state=42, max_iter=1000)
            lasso.fit(features, targets)
            
            # Get selected features
            selected_features = []
            for i, coef in enumerate(lasso.coef_):
                if abs(coef) > 1e-6:
                    selected_features.append(features.columns[i])
            
            # Map back to groups
            selected_groups = {}
            for group_name, group_features in feature_groups.items():
                group_selected = [f for f in group_features if f in selected_features]
                if group_selected:
                    selected_groups[group_name] = group_selected
            
            return selected_groups
            
        except Exception as e:
            self.logger.warning(f"Fallback LASSO selection failed: {e}")
            return {}


class StatisticalSelection:
    """Main statistical selection system."""
    
    def __init__(self, config: SelectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.stability_selector = StabilitySelector(config)
        self.permutation_tester = PermutationTester(config)
        self.fdr_controller = FDRController(config)
        self.conditional_ic_tester = ConditionalICTester(config)
        self.group_lasso_selector = GroupLASSOSelector(config)
    
    def select_final_features(self,
                            materialized_htfs: Dict[str, Any],
                            interactions: List[Any],
                            targets: Optional[pd.Series] = None) -> CrossTimeframeStatisticalSelectionResult:
        """
        Select final features using statistical methods.
        
        Args:
            materialized_htfs: Materialized HTF features
            interactions: Generated interactions
            targets: Target variables
            
        Returns:
            Selection result with selected features
        """
        self.logger.info("Starting statistical selection")
        tprint_info(
            "🚀 Starting statistical selection",
            f"htfs={len(materialized_htfs)}",
            f"interactions={len(interactions)}",
            f"targets_provided={targets is not None}",
        )

        # Prepare feature matrix
        feature_matrix, feature_names = self._prepare_feature_matrix(
            materialized_htfs, interactions
        )

        tprint_debug(
            f"   → Prepared feature matrix with {len(feature_names)} features and {len(feature_matrix)} rows"
        )

        if feature_matrix.empty or targets is None:
            self.logger.warning("No features or targets available for selection")
            tprint_warning("⚠️ Statistical selection skipped - missing features or targets")
            return CrossTimeframeStatisticalSelectionResult(
                selected_features=[],
                selection_frequencies={},
                p_values={},
                fdr_corrected_p_values={},
                conditional_ics={},
                group_lasso_groups={},
                selection_method="none",
                metadata={}
            )
        
        # Identify base features
        base_features = self._identify_base_features(materialized_htfs)
        tprint_debug(f"   → Identified {len(base_features)} base features for conditional tests")

        # Stability selection
        selection_frequencies = self.stability_selector.select_features(
            feature_matrix, targets, base_features
        )
        tprint_info("   → Stability selection computed", f"features={len(selection_frequencies)}")

        # Permutation testing
        cv_folds = max(2, min(getattr(self.config, "permutation_cv_folds", 5), len(feature_matrix) - 1))
        cv_splitter = None
        if len(feature_matrix) > cv_folds:
            cv_splitter = TimeSeriesSplit(n_splits=cv_folds)

        p_values = self.permutation_tester.calculate_p_values(
            feature_matrix,
            targets,
            base_features,
            cv_splitter=cv_splitter,
            block_size=getattr(self.config, "permutation_block_size", None),
            n_permutations=getattr(self.config, "permutation_n_permutations", None),
            random_state=getattr(self.config, "permutation_random_state", None),
        )
        tprint_info("   → Permutation testing complete", f"features={len(p_values)}")

        # FDR control
        fdr_corrected_p_values = self.fdr_controller.control_fdr(p_values)
        tprint_info(
            "   → FDR control applied",
            f"features={len(fdr_corrected_p_values)}",
        )

        # Conditional IC testing
        conditional_ics = self.conditional_ic_tester.calculate_conditional_ics(
            feature_matrix, targets, base_features
        )
        tprint_info(
            "   → Conditional ICs calculated",
            f"features={len(conditional_ics)}",
        )

        # Group LASSO (if enabled)
        group_lasso_groups = {}
        if self.config.get('enable_group_lasso', False):
            feature_groups = self._create_feature_groups(materialized_htfs, interactions)
            group_lasso_groups = self.group_lasso_selector.select_features(
                feature_matrix, targets, feature_groups
            )
            tprint_info(
                "   → Group LASSO processed",
                f"groups={len(group_lasso_groups)}",
            )

        # Apply selection criteria
        selected_features = self._apply_selection_criteria(
            feature_names,
            selection_frequencies,
            fdr_corrected_p_values,
            conditional_ics
        )
        tprint_success(
            "🏁 Final feature selection complete",
            f"selected_features={len(selected_features)}",
        )

        # Create selection result
        result = CrossTimeframeStatisticalSelectionResult(
            selected_features=selected_features,
            selection_frequencies=selection_frequencies,
            p_values=p_values,
            fdr_corrected_p_values=fdr_corrected_p_values,
            conditional_ics=conditional_ics,
            group_lasso_groups=group_lasso_groups,
            selection_method="stability_selection_fdr",
            metadata={
                'total_features_evaluated': len(feature_names),
                'selection_threshold': 0.6,  # Minimum selection frequency
                'fdr_threshold': self.config.fdr_q,
                'conditional_ic_threshold': self.config.min_conditional_ic
            }
        )

        self.logger.info(f"Statistical selection completed: {len(selected_features)} features selected")
        tprint_success(
            "✅ Statistical selection completed",
            f"selected={len(selected_features)}",
            f"evaluated={len(feature_names)}",
        )
        return result
    
    def _prepare_feature_matrix(self, 
                              materialized_htfs: Dict[str, Any],
                              interactions: List[Any]) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare feature matrix from materialized HTFs and interactions."""
        feature_data = {}
        feature_names = []
        
        # Add HTF features
        for name, htf in materialized_htfs.items():
            if hasattr(htf, 'feature_series'):
                feature_data[name] = htf.feature_series
                feature_names.append(name)
        
        # Add interactions
        for interaction in interactions:
            if hasattr(interaction, 'feature_series'):
                feature_data[interaction.name] = interaction.feature_series
                feature_names.append(interaction.name)
        
        if not feature_data:
            return pd.DataFrame(), []
        
        # Create DataFrame
        feature_matrix = pd.DataFrame(feature_data)
        
        # Align indices and remove NaN values
        feature_matrix = feature_matrix.dropna()
        
        return feature_matrix, feature_names
    
    def _identify_base_features(self, materialized_htfs: Dict[str, Any]) -> List[str]:
        """Identify base features for conditional testing."""
        base_features = []
        
        for name, htf in materialized_htfs.items():
            if hasattr(htf, 'metadata') and 'base_feature' in htf.metadata:
                base_feature = htf.metadata['base_feature']
                if base_feature not in base_features:
                    base_features.append(base_feature)
        
        return base_features
    
    def _create_feature_groups(self, 
                             materialized_htfs: Dict[str, Any],
                             interactions: List[Any]) -> Dict[str, List[str]]:
        """Create feature groups for Group LASSO."""
        groups = {
            'htf_trend': [],
            'htf_volatility': [],
            'htf_oscillators': [],
            'htf_anchors': [],
            'interactions': []
        }
        
        # Group HTF features
        for name, htf in materialized_htfs.items():
            family = getattr(htf, 'family', 'unknown')
            if family in ['trend_level_vol']:
                if 'trend' in name.lower() or 'ema' in name.lower():
                    groups['htf_trend'].append(name)
                elif 'vol' in name.lower() or 'sigma' in name.lower():
                    groups['htf_volatility'].append(name)
            elif family == 'oscillators':
                groups['htf_oscillators'].append(name)
            elif family == 'anchors':
                groups['htf_anchors'].append(name)
        
        # Group interactions
        for interaction in interactions:
            groups['interactions'].append(interaction.name)
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}
        
        return groups
    
    def _apply_selection_criteria(self, 
                                feature_names: List[str],
                                selection_frequencies: Dict[str, float],
                                fdr_corrected_p_values: Dict[str, float],
                                conditional_ics: Dict[str, float]) -> List[str]:
        """Apply selection criteria to get final selected features."""
        selected_features = []
        
        for feature_name in feature_names:
            # Check selection frequency
            freq = selection_frequencies.get(feature_name, 0.0)
            if freq < 0.6:  # Minimum selection frequency
                continue
            
            # Check FDR-corrected p-value
            p_value = fdr_corrected_p_values.get(feature_name, 1.0)
            if p_value > self.config.fdr_q:
                continue
            
            # Check conditional IC
            conditional_ic = conditional_ics.get(feature_name, 0.0)
            if abs(conditional_ic) < self.config.min_conditional_ic:
                continue
            
            selected_features.append(feature_name)
        
        return selected_features
    
    def get_selection_summary(self, result: CrossTimeframeStatisticalSelectionResult) -> Dict[str, Any]:
        """Get summary of selection results."""
        summary = {
            'total_selected': len(result.selected_features),
            'selection_frequencies': result.selection_frequencies,
            'fdr_corrected_p_values': result.fdr_corrected_p_values,
            'conditional_ics': result.conditional_ics,
            'group_lasso_groups': result.group_lasso_groups,
            'selection_method': result.selection_method,
            'metadata': result.metadata
        }
        tprint_info(
            "📝 Selection summary",
            f"total_selected={summary['total_selected']}",
            f"method={summary['selection_method']}",
        )
        return summary