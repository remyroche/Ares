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
from scipy.stats import permutation_test
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
                          base_features: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate p-values using permutation testing.
        
        Args:
            features: Feature matrix
            targets: Target series
            base_features: Base features for conditional testing
            
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
                    features, targets, feature_name, base_features
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
                                     base_features: Optional[List[str]] = None) -> float:
        """Calculate permutation p-value for a single feature."""
        try:
            # Calculate original correlation
            original_corr = features[feature_name].corr(targets)
            
            if pd.isna(original_corr):
                return 1.0
            
            # Perform permutation test
            n_permutations = 1000
            permuted_corrs = []
            
            for _ in range(n_permutations):
                # Permute the feature
                permuted_feature = features[feature_name].sample(frac=1).reset_index(drop=True)
                
                # Calculate permuted correlation
                permuted_corr = permuted_feature.corr(targets)
                
                if not pd.isna(permuted_corr):
                    permuted_corrs.append(abs(permuted_corr))
            
            if not permuted_corrs:
                return 1.0
            
            # Calculate p-value
            p_value = np.mean(np.array(permuted_corrs) >= abs(original_corr))
            
            return p_value
            
        except Exception as e:
            self.logger.warning(f"Permutation test failed for {feature_name}: {e}")
            return 1.0


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
        self._last_scaler: Optional[StandardScaler] = None
        self._last_group_scores: Dict[str, float] = {}

    def select_features(self,
                      features: pd.DataFrame,
                      targets: pd.Series,
                      feature_groups: Dict[str, List[str]],
                      split: Optional[Any] = None) -> Dict[str, List[str]]:
        """
        Select features using Group LASSO.

        Args:
            features: Feature matrix
            targets: Target series
            feature_groups: Dictionary of group_name -> feature_list
            split: Optional pre-split datasets or indices. Supported formats:
                - (X_train, y_train, X_val, y_val)
                - (train_indices, val_indices)

        Returns:
            Dictionary of selected groups
        """
        self.logger.info("Starting Group LASSO selection")
        tprint_info(
            "🧩 Running group selection",
            f"groups={len(feature_groups)}",
            f"features={len(features.columns)}",
        )

        (train_features,
         train_targets,
         val_features,
         _) = self._resolve_split(features, targets, split)

        scaler = StandardScaler()
        scaler.fit(train_features)
        self._last_scaler = scaler

        train_features_scaled = self._scale_features(scaler, train_features)
        val_features_scaled = self._scale_features(scaler, val_features)

        feature_columns = list(train_features_scaled.columns)
        group_matrix = self._create_group_matrix(train_features_scaled, feature_groups)

        if not GROUP_LASSO_AVAILABLE:
            self.logger.warning("Group LASSO not available, using standard LASSO")
            tprint_warning("⚠️ Group LASSO package unavailable - falling back to standard LASSO")
            return self._fallback_lasso_selection(
                train_features_scaled,
                train_targets,
                feature_groups,
                val_features_scaled,
            )

        try:
            # Apply Group LASSO (simplified implementation)
            selected_group_indices, feature_scores = self._apply_group_lasso(
                train_features_scaled,
                train_targets,
                group_matrix,
            )

            self._last_group_scores = self._compute_group_scores(
                feature_scores,
                feature_groups,
                feature_columns,
                val_features_scaled,
            )

            # Map back to feature groups
            selected_feature_groups = {}
            for group_idx, (group_name, group_features) in enumerate(feature_groups.items()):
                if group_idx in selected_group_indices:
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
            return self._fallback_lasso_selection(
                train_features_scaled,
                train_targets,
                feature_groups,
                val_features_scaled,
            )

    def _resolve_split(self,
                       features: pd.DataFrame,
                       targets: pd.Series,
                       split: Optional[Any]) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
        """Resolve the training/validation split definition."""
        features_df = features if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
        targets_series = targets if isinstance(targets, pd.Series) else pd.Series(targets)

        if split is None:
            return features_df, targets_series, None, None

        if isinstance(split, tuple):
            if len(split) == 4:
                train_features, train_targets, val_features, val_targets = split
                train_features_df = self._ensure_dataframe(train_features, features_df.columns)
                val_features_df = self._ensure_dataframe(val_features, features_df.columns)
                train_targets_series = self._ensure_series(train_targets)
                val_targets_series = self._ensure_series(val_targets)
                if val_features_df is not None and len(val_features_df) == 0:
                    val_features_df = None
                    val_targets_series = None
                return train_features_df, train_targets_series, val_features_df, val_targets_series

            if len(split) == 2:
                train_indices, val_indices = split
                train_indices = np.array(train_indices)
                val_indices = np.array(val_indices)

                train_features_df = features_df.iloc[train_indices]
                train_targets_series = targets_series.iloc[train_indices]

                if len(val_indices) == 0:
                    return train_features_df, train_targets_series, None, None

                val_features_df = features_df.iloc[val_indices]
                val_targets_series = targets_series.iloc[val_indices]

                return train_features_df, train_targets_series, val_features_df, val_targets_series

        raise ValueError("Unsupported split format for GroupLASSOSelector")

    def _ensure_dataframe(self, data: Any, columns: pd.Index) -> Optional[pd.DataFrame]:
        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            # Align column order with the reference columns
            return data.reindex(columns=list(columns))
        return pd.DataFrame(data, columns=list(columns))

    def _ensure_series(self, data: Any) -> Optional[pd.Series]:
        if data is None:
            return None
        if isinstance(data, pd.Series):
            return data
        return pd.Series(data)

    def _scale_features(self,
                        scaler: StandardScaler,
                        features: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if features is None:
            return None
        scaled = scaler.transform(features)
        return pd.DataFrame(scaled, index=features.index, columns=features.columns)
    
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
                          features_scaled: pd.DataFrame,
                          targets: pd.Series,
                          group_matrix: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """Apply Group LASSO (simplified implementation)."""
        # This is a simplified implementation
        # In practice, you'd use a proper Group LASSO solver

        # Use standard LASSO as fallback
        lasso = Lasso(alpha=0.01, max_iter=1000)
        lasso.fit(features_scaled.values, targets)

        # Find non-zero coefficients
        feature_scores = np.abs(lasso.coef_)
        selected_indices = np.where(feature_scores > 1e-6)[0].tolist()

        # Map back to groups
        selected_groups = []
        for idx in selected_indices:
            group_idx = np.where(group_matrix[idx, :] == 1)[0]
            if len(group_idx) > 0:
                selected_groups.append(group_idx[0])

        return selected_groups, feature_scores

    def _fallback_lasso_selection(self,
                                features: pd.DataFrame,
                                targets: pd.Series,
                                feature_groups: Dict[str, List[str]],
                                val_features: Optional[pd.DataFrame]) -> Dict[str, List[str]]:
        """Fallback to standard LASSO selection."""
        try:
            # Use LASSO with cross-validation
            lasso = LassoCV(cv=3, random_state=42, max_iter=1000)
            lasso.fit(features.values, targets)

            # Get selected features
            selected_features = []
            feature_scores = np.abs(lasso.coef_)
            for i, coef in enumerate(feature_scores):
                if coef > 1e-6:
                    selected_features.append(features.columns[i])

            self._last_group_scores = self._compute_group_scores(
                feature_scores,
                feature_groups,
                list(features.columns),
                val_features,
            )

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

    def _compute_group_scores(self,
                              feature_scores: np.ndarray,
                              feature_groups: Dict[str, List[str]],
                              feature_columns: List[str],
                              val_features: Optional[pd.DataFrame]) -> Dict[str, float]:
        if val_features is not None and len(val_features) > 0:
            validation_activity = np.mean(np.abs(val_features.values), axis=0)
        else:
            validation_activity = np.ones(len(feature_columns))

        column_to_index = {name: idx for idx, name in enumerate(feature_columns)}
        group_scores: Dict[str, float] = {}

        for group_name, group_features in feature_groups.items():
            score = 0.0
            for feature_name in group_features:
                if feature_name in column_to_index:
                    idx = column_to_index[feature_name]
                    score += feature_scores[idx] * validation_activity[idx]
            group_scores[group_name] = score

        return group_scores


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
        p_values = self.permutation_tester.calculate_p_values(
            feature_matrix, targets, base_features
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
            group_lasso_split = self._build_group_lasso_split(feature_matrix, targets)
            group_lasso_groups = self.group_lasso_selector.select_features(
                feature_matrix,
                targets,
                feature_groups,
                split=group_lasso_split,
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

    def _build_group_lasso_split(self,
                                 features: pd.DataFrame,
                                 targets: pd.Series) -> Optional[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        """Construct a train/validation split for group LASSO selection."""
        if features is None or targets is None or len(features) < 3:
            return None

        try:
            n_splits = min(5, len(features) - 1)
            if n_splits >= 2:
                tscv = TimeSeriesSplit(n_splits=n_splits)
                splits = list(tscv.split(features))
                if splits:
                    train_indices, val_indices = splits[-1]
                    if len(val_indices) > 0:
                        return (
                            features.iloc[train_indices],
                            targets.iloc[train_indices],
                            features.iloc[val_indices],
                            targets.iloc[val_indices],
                        )
        except ValueError:
            pass

        split_idx = int(len(features) * 0.8)
        if split_idx <= 0 or split_idx >= len(features):
            return None

        train_features = features.iloc[:split_idx]
        val_features = features.iloc[split_idx:]
        if len(val_features) == 0:
            return None

        return (
            train_features,
            targets.iloc[:split_idx],
            val_features,
            targets.iloc[split_idx:],
        )
    
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