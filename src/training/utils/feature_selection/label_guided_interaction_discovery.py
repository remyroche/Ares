"""
Label-Guided Interaction Discovery

This module implements label-guided interaction discovery that:
1. Uses MI (Mutual Information) or SHAP interaction strength vs target
2. Applies regularized models (L1, group LASSO) on dense interaction grid
3. Only selects interactions with meaningful R²/MI increase vs base features

Key Features:
- Interaction-specific MI calculation (not just feature MI)
- SHAP interaction values for tree-based models
- L1/group LASSO regularization for interaction selection
- R²/MI lift requirement: interaction must beat base features
- Category-aware selection to prevent over-representation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import logging
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import warnings

from src.utils.tprint import tprint_info

try:
    import lightgbm as lgb
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class InteractionCandidate:
    """Represents a candidate interaction feature."""
    name: str
    feature1: str
    feature2: str
    operation: str  # 'multiply', 'divide', 'subtract', 'log_ratio', etc.
    mi_score: float = 0.0
    shap_interaction_score: float = 0.0
    r2_lift: float = 0.0  # R² improvement over base features
    mi_lift: float = 0.0  # MI improvement over base features
    category1: str = 'unknown'
    category2: str = 'unknown'
    selected: bool = False


@dataclass
class LabelGuidedInteractionConfig:
    """Configuration for label-guided interaction discovery."""

    # MI/SHAP scoring
    use_mi_scoring: bool = True
    use_shap_scoring: bool = True
    mi_weight: float = 0.5
    shap_weight: float = 0.5

    # Lift requirements
    min_r2_lift: float = 0.02  # Interaction must improve R² by at least 2% (tightened from 1%)
    min_mi_lift: float = 0.15  # Interaction must improve MI by at least 15% (tightened from 5%)
    require_r2_lift: bool = True
    require_mi_lift: bool = True

    # Regularization
    use_lasso: bool = True
    use_group_lasso: bool = False  # Group LASSO for category-based selection
    lasso_alpha: Optional[float] = None  # None = use CV
    lasso_cv_folds: int = 5
    lasso_max_iter: int = 1000

    # Interaction generation
    max_pairs_to_test: int = 100  # Limit pair combinations
    operations: List[str] = None  # Operations to test

    # Category controls
    max_interactions_per_category_pair: int = 7  # Max interactions per (cat1, cat2) pair
    banned_category_pairs: Set[Tuple[str, str]] = None  # Pairs to exclude

    # Performance
    n_jobs: int = -1
    random_state: int = 42

    def __post_init__(self):
        """Initialize default values."""
        if self.operations is None:
            self.operations = ['multiply', 'divide', 'subtract', 'add', 'log_ratio']
        if self.banned_category_pairs is None:
            self.banned_category_pairs = set()


class LabelGuidedInteractionDiscovery:
    """
    Label-guided interaction discovery using MI/SHAP and regularized selection.

    This class implements a sophisticated approach to interaction feature discovery that:
    1. Restricts interactions to pairs showing MI or SHAP interaction strength vs target
    2. Uses regularized models (L1, group LASSO) to pick meaningful interactions
    3. Ensures interactions provide meaningful R²/MI lift over base features
    """

    def __init__(self, config: Optional[LabelGuidedInteractionConfig] = None):
        """
        Initialize label-guided interaction discovery.

        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or LabelGuidedInteractionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Storage for candidates
        self.candidates: List[InteractionCandidate] = []
        self.selected_interactions: List[InteractionCandidate] = []

        # Cache for base feature scores
        self._base_mi_scores: Dict[str, float] = {}
        self._base_r2_scores: Dict[str, float] = {}

    def discover_interactions(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        feature_categories: Dict[str, str],
        feature_pairs: List[Tuple[str, str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Discover label-guided interactions.

        Args:
            features: Feature dataframe (n_samples, n_features)
            target: Target variable (n_samples,)
            feature_categories: Mapping from feature name to category
            feature_pairs: Optional list of feature pairs to test. If None, generates pairs.

        Returns:
            interaction_df: DataFrame of selected interaction features
            metadata: Dictionary with discovery statistics
        """
        self.logger.info("🔍 Starting label-guided interaction discovery...")

        # Clean inputs
        features_clean, target_clean = self._clean_inputs(features, target)

        # Generate or validate feature pairs
        if feature_pairs is None:
            feature_pairs = self._generate_feature_pairs(features_clean, feature_categories)
        else:
            feature_pairs = self._filter_feature_pairs(feature_pairs, feature_categories)

        # Ensure that feature pairs only reference columns that are actually
        # present in the cleaned feature matrix. This prevents the subsequent
        # candidate-generation step from silently producing zero candidates
        # because of stale or mismatched feature names.
        valid_cols = set(features_clean.columns)
        original_pair_count = len(feature_pairs)
        feature_pairs = [
            (f1, f2)
            for (f1, f2) in feature_pairs
            if f1 in valid_cols and f2 in valid_cols
        ]

        removed_pairs = original_pair_count - len(feature_pairs)
        if removed_pairs > 0:
            self.logger.info(
                "  📊 Filtered %d feature pairs that referenced missing columns; %d remain",
                removed_pairs,
                len(feature_pairs),
            )

        # If all externally-provided pairs were filtered out, fall back to
        # automatic pair generation based on the cleaned feature matrix so we
        # still generate interaction candidates.
        if len(feature_pairs) == 0:
            self.logger.warning(
                "  ⚠️ No valid feature pairs remain after alignment; "
                "falling back to automatic pair generation"
            )
            feature_pairs = self._generate_feature_pairs(features_clean, feature_categories)

        self.logger.info(f"  📊 Testing {len(feature_pairs)} feature pairs")

        # Calculate base feature scores for lift comparison
        self._calculate_base_scores(features_clean, target_clean)

        # Generate interaction candidates
        self._generate_candidates(features_clean, target_clean, feature_pairs, feature_categories)

        self.logger.info(f"  📊 Generated {len(self.candidates)} interaction candidates")

        # Score candidates using MI and/or SHAP
        self._score_candidates(features_clean, target_clean)

        # Filter candidates by lift requirements
        self._filter_by_lift()

        # Apply regularized selection (LASSO)
        if self.config.use_lasso:
            self._apply_lasso_selection(features_clean, target_clean)

        # Apply category-based limits
        self._apply_category_limits()

        # Build final interaction dataframe
        interaction_df = self._build_interaction_dataframe(features)

        # Build metadata
        metadata = self._build_metadata()

        self.logger.info(f"✅ Selected {len(self.selected_interactions)} interactions")

        return interaction_df, metadata

    def _clean_inputs(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean and align features and target.

        Primary alignment strategy:
        - Use index intersection when there is at least some overlap.
        - If there are *no* common indices (e.g. datetime index vs RangeIndex),
          fall back to positional alignment on the last min(len(features), len(target))
          rows and reset both indices to a shared RangeIndex. This mirrors the
          robustness used in the higher-level Ares steps and prevents the
          interaction discovery from operating on an empty sample set.
        """

        # Align indices with robust fallback when labels do not overlap
        common_idx = features.index.intersection(target.index)

        if len(common_idx) == 0:
            # Fall back to positional alignment, preserving recent data
            min_len = min(len(features), len(target))
            if min_len == 0:
                # Nothing to align; return empty samples
                return features.iloc[0:0].copy(), target.iloc[0:0].copy()

            features_aligned = features.iloc[-min_len:].copy()
            target_aligned = target.iloc[-min_len:].copy()

            # Reset to a shared RangeIndex to guarantee alignment downstream
            features_aligned.index = pd.RangeIndex(min_len)
            target_aligned.index = pd.RangeIndex(min_len)
        else:
            features_aligned = features.loc[common_idx].copy()
            target_aligned = target.loc[common_idx].copy()

        # Remove NaN/inf
        finite_mask = np.isfinite(target_aligned) & np.all(np.isfinite(features_aligned), axis=1)
        features_clean = features_aligned[finite_mask]
        target_clean = target_aligned[finite_mask]

        if features_clean.empty:
            self.logger.warning(
                "  \u26a0\ufe0f After NaN/inf filtering, no samples remain; applying fallback alignment."
            )

            fallback_mask = np.isfinite(target_aligned)
            features_fallback = features_aligned[fallback_mask].copy()
            target_fallback = target_aligned[fallback_mask].copy()

            if len(features_fallback) == 0:
                return features_aligned.iloc[0:0].copy(), target_aligned.iloc[0:0].copy()

            features_fallback = features_fallback.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            target_fallback = pd.Series(
                np.nan_to_num(target_fallback.values, nan=0.0, posinf=0.0, neginf=0.0),
                index=target_fallback.index,
                name=target_fallback.name,
            )

            features_clean = features_fallback
            target_clean = target_fallback

        return features_clean, target_clean

    def _generate_feature_pairs(
        self,
        features: pd.DataFrame,
        feature_categories: Dict[str, str]
    ) -> List[Tuple[str, str]]:
        """
        Generate candidate feature pairs using MI-guided selection.

        Prioritizes pairs from different categories to encourage diversity.
        """
        feature_names = list(features.columns)
        n_features = len(feature_names)

        # Limit total pairs to test
        max_pairs = min(self.config.max_pairs_to_test, n_features * (n_features - 1) // 2)

        # Generate all possible pairs
        all_pairs = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                f1, f2 = feature_names[i], feature_names[j]
                cat1 = feature_categories.get(f1, 'unknown')
                cat2 = feature_categories.get(f2, 'unknown')

                # Skip banned category pairs
                if (cat1, cat2) in self.config.banned_category_pairs:
                    continue
                if (cat2, cat1) in self.config.banned_category_pairs:
                    continue

                # Prioritize cross-category pairs
                priority = 2 if cat1 != cat2 else 1
                all_pairs.append((f1, f2, priority, cat1, cat2))

        # Sort by priority (cross-category first) and take top N
        all_pairs.sort(key=lambda x: x[2], reverse=True)
        selected_pairs = [(f1, f2) for f1, f2, _, _, _ in all_pairs[:max_pairs]]

        return selected_pairs

    def _filter_feature_pairs(
        self,
        feature_pairs: List[Tuple[str, str]],
        feature_categories: Dict[str, str]
    ) -> List[Tuple[str, str]]:
        """Filter feature pairs based on category restrictions."""
        filtered_pairs = []
        for f1, f2 in feature_pairs:
            cat1 = feature_categories.get(f1, 'unknown')
            cat2 = feature_categories.get(f2, 'unknown')

            # Skip banned category pairs
            if (cat1, cat2) in self.config.banned_category_pairs:
                continue
            if (cat2, cat1) in self.config.banned_category_pairs:
                continue

            filtered_pairs.append((f1, f2))

        return filtered_pairs

    def _calculate_base_scores(self, features: pd.DataFrame, target: pd.Series):
        """Calculate MI and R² scores for base features."""
        self.logger.info("  📊 Calculating base feature scores...")

        for col in features.columns:
            # MI score
            if self.config.use_mi_scoring:
                try:
                    mi = mutual_info_regression(
                        features[[col]].values,
                        target.values,
                        random_state=self.config.random_state,
                        n_neighbors=3
                    )[0]
                    self._base_mi_scores[col] = float(mi)
                except Exception as e:
                    self.logger.warning(f"  ⚠️ MI calculation failed for {col}: {e}")
                    self._base_mi_scores[col] = 0.0

            # R² score (using simple linear regression)
            if self.config.require_r2_lift:
                try:
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression()
                    lr.fit(features[[col]].values, target.values)
                    r2 = r2_score(target.values, lr.predict(features[[col]].values))
                    self._base_r2_scores[col] = max(0.0, r2)  # Clip negative R²
                except Exception as e:
                    self.logger.warning(f"  ⚠️ R² calculation failed for {col}: {e}")
                    self._base_r2_scores[col] = 0.0

        # Log summary statistics for base feature MI scores so we can
        # understand how strong the individual features are and how much
        # headroom interactions realistically have for MI lift.
        if self._base_mi_scores:
            mi_values = np.array(list(self._base_mi_scores.values()), dtype=float)
            self.logger.info(
                "  📊 Base feature MI stats: min=%.4f, median=%.4f, max=%.4f",
                float(np.min(mi_values)),
                float(np.median(mi_values)),
                float(np.max(mi_values)),
            )

            # Also log the top-K base features by MI so we can inspect
            # which concrete features (including vol-normalized / VWAP
            # variants) are actually carrying most of the signal.
            mi_items = sorted(
                self._base_mi_scores.items(), key=lambda kv: kv[1], reverse=True
            )
            top_k = min(30, len(mi_items))
            self.logger.info("  📊 Top base features by MI (top %d):", top_k)
            for name, mi in mi_items[:top_k]:
                self.logger.info("    • %s: MI=%.4f", name, float(mi))

    def _generate_candidates(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        feature_pairs: List[Tuple[str, str]],
        feature_categories: Dict[str, str]
    ):
        """Generate interaction candidates from feature pairs."""
        self.logger.info("  🔧 Generating interaction candidates...")

        for f1, f2 in feature_pairs:
            if f1 not in features.columns or f2 not in features.columns:
                continue

            cat1 = feature_categories.get(f1, 'unknown')
            cat2 = feature_categories.get(f2, 'unknown')

            v1 = features[f1].values
            v2 = features[f2].values

            # Generate interactions for each operation
            for op in self.config.operations:
                interaction_series, name = self._apply_operation(v1, v2, f1, f2, op, features.index)

                if interaction_series is not None:
                    candidate = InteractionCandidate(
                        name=name,
                        feature1=f1,
                        feature2=f2,
                        operation=op,
                        category1=cat1,
                        category2=cat2
                    )
                    self.candidates.append(candidate)

    def _apply_operation(
        self,
        v1: np.ndarray,
        v2: np.ndarray,
        f1: str,
        f2: str,
        operation: str,
        index: pd.Index
    ) -> Tuple[Optional[pd.Series], str]:
        """Apply an operation to create interaction feature."""
        eps = 1e-8

        try:
            if operation == 'multiply':
                result = v1 * v2
                name = f"{f1}_x_{f2}"
            elif operation == 'divide':
                result = v1 / (v2 + eps)
                name = f"{f1}_div_{f2}"
            elif operation == 'subtract':
                result = v1 - v2
                name = f"{f1}_minus_{f2}"
            elif operation == 'add':
                result = v1 + v2
                name = f"{f1}_plus_{f2}"
            elif operation == 'log_ratio':
                result = np.log(np.abs(v1) + eps) / (np.log(np.abs(v2) + eps) + eps)
                name = f"{f1}_log_ratio_{f2}"
            else:
                return None, ""

            # Replace inf/nan
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

            return pd.Series(result, index=index, name=name), name

        except Exception as e:
            self.logger.warning(f"  ⚠️ Operation {operation} failed for {f1}, {f2}: {e}")
            return None, ""

    def _score_candidates(self, features: pd.DataFrame, target: pd.Series):
        """Score interaction candidates using MI and/or SHAP."""
        self.logger.info("  📊 Scoring interaction candidates...")

        # Build candidate dataframe
        candidate_features = {}
        for i, cand in enumerate(self.candidates):
            v1 = features[cand.feature1].values
            v2 = features[cand.feature2].values
            interaction_series, _ = self._apply_operation(
                v1, v2, cand.feature1, cand.feature2, cand.operation, features.index
            )
            if interaction_series is not None:
                candidate_features[cand.name] = interaction_series

        candidate_df = pd.DataFrame(candidate_features)

        # Score using MI
        if self.config.use_mi_scoring and len(candidate_df.columns) > 0:
            self._score_mi(candidate_df, target)

        # Score using SHAP (interaction values)
        if self.config.use_shap_scoring and SHAP_AVAILABLE and len(candidate_df.columns) > 0:
            self._score_shap_interactions(features, candidate_df, target)

    def _score_mi(self, candidate_df: pd.DataFrame, target: pd.Series):
        """Score candidates using Mutual Information."""
        try:
            mi_scores = mutual_info_regression(
                candidate_df.values,
                target.values,
                random_state=self.config.random_state,
                n_neighbors=3
            )

            for i, cand in enumerate(self.candidates):
                if cand.name in candidate_df.columns:
                    idx = candidate_df.columns.get_loc(cand.name)
                    cand.mi_score = float(mi_scores[idx])

        except Exception as e:
            self.logger.warning(f"  ⚠️ MI scoring failed: {e}")

    def _score_shap_interactions(
        self,
        base_features: pd.DataFrame,
        candidate_df: pd.DataFrame,
        target: pd.Series
    ):
        """Score candidates using SHAP interaction values."""
        if not SHAP_AVAILABLE:
            return

        try:
            # Combine base features with candidates
            combined = pd.concat([base_features, candidate_df], axis=1)

            # Train a lightweight LGBM model
            model = lgb.LGBMRegressor(
                n_estimators=50,
                max_depth=3,
                num_leaves=10,
                learning_rate=0.1,
                random_state=self.config.random_state,
                verbose=-1
            )
            model.fit(combined, target)

            # Calculate SHAP interaction values
            explainer = shap.TreeExplainer(model)
            shap_interaction_values = explainer.shap_interaction_values(combined)

            # Extract interaction strengths for each candidate
            base_feature_names = list(base_features.columns)
            for cand in self.candidates:
                if cand.name not in candidate_df.columns:
                    continue

                try:
                    # Find indices
                    interaction_idx = combined.columns.get_loc(cand.name)
                    f1_idx = combined.columns.get_loc(cand.feature1)
                    f2_idx = combined.columns.get_loc(cand.feature2)

                    # SHAP interaction value between f1 and f2
                    # shap_interaction_values has shape (n_samples, n_features, n_features)
                    interaction_strength = np.abs(shap_interaction_values[:, f1_idx, f2_idx]).mean()

                    cand.shap_interaction_score = float(interaction_strength)

                except Exception as e:
                    self.logger.warning(f"  ⚠️ SHAP extraction failed for {cand.name}: {e}")

        except Exception as e:
            self.logger.warning(f"  ⚠️ SHAP interaction scoring failed: {e}")

    def _filter_by_lift(self):
        """Filter candidates that don't provide sufficient lift over base features."""
        self.logger.info("  🔍 Filtering by R²/MI lift requirements...")
        tprint_info("  🔍 [LGID] Filtering candidates by R²/MI lift requirements...")

        filtered_candidates = []
        mi_lifts_before_filter = []
        total_before = len(self.candidates)

        for cand in self.candidates:
            # Calculate MI lift
            base_mi_f1 = self._base_mi_scores.get(cand.feature1, 0.0)
            base_mi_f2 = self._base_mi_scores.get(cand.feature2, 0.0)
            max_base_mi = max(base_mi_f1, base_mi_f2)

            if max_base_mi > 0:
                cand.mi_lift = (cand.mi_score - max_base_mi) / max_base_mi
            else:
                cand.mi_lift = 0.0

            # Track MI lift values for diagnostics before any filtering
            mi_lifts_before_filter.append(cand.mi_lift)

            # Check MI lift requirement
            if self.config.require_mi_lift:
                if cand.mi_lift < self.config.min_mi_lift:
                    continue  # Skip this candidate

            # For R² lift, would need to calculate but that's expensive
            # For now, we rely on MI lift as primary filter

            filtered_candidates.append(cand)

        # Log MI-lift distribution across all candidates (before
        # filtering) so we can see whether the current thresholds are
        # realistic and whether interactions provide any incremental MI
        # at all over the best base feature in each pair.
        if mi_lifts_before_filter:
            lifts_arr = np.array(mi_lifts_before_filter, dtype=float)

            self.logger.info(
                "  📊 MI-lift distribution (before filtering): min=%.4f, p25=%.4f, "
                "median=%.4f, p75=%.4f, max=%.4f",
                float(np.min(lifts_arr)),
                float(np.percentile(lifts_arr, 25)),
                float(np.percentile(lifts_arr, 50)),
                float(np.percentile(lifts_arr, 75)),
                float(np.max(lifts_arr)),
            )

            num_pos = int(np.sum(lifts_arr > 0.0))
            num_ge_001 = int(np.sum(lifts_arr >= 0.01))
            num_ge_005 = int(np.sum(lifts_arr >= 0.05))
            num_ge_010 = int(np.sum(lifts_arr >= 0.10))

            self.logger.info(
                "  📊 MI-lift counts (before filtering): >0: %d, "+
                ">=0.01: %d, >=0.05: %d, >=0.10: %d",
                num_pos,
                num_ge_001,
                num_ge_005,
                num_ge_010,
            )

            # Log the top few candidates by MI lift for easier manual
            # inspection in logs.
            top_k = min(5, len(self.candidates))
            if top_k > 0:
                top_by_lift = sorted(
                    self.candidates, key=lambda c: c.mi_lift, reverse=True
                )[:top_k]
                for c in top_by_lift:
                    base_mi_f1 = self._base_mi_scores.get(c.feature1, 0.0)
                    base_mi_f2 = self._base_mi_scores.get(c.feature2, 0.0)
                    max_base_mi = max(base_mi_f1, base_mi_f2)
                    self.logger.info(
                        "  🔝 MI-lift candidate: %s (%s, %s) "
                        "mi=%.4f, base_max_mi=%.4f, mi_lift=%.4f",
                        c.name,
                        c.feature1,
                        c.feature2,
                        float(c.mi_score),
                        float(max_base_mi),
                        float(c.mi_lift),
                    )
        else:
            self.logger.info(
                "  📊 No MI-lift values computed for candidates (empty candidate set?)"
            )

        n_filtered = total_before - len(filtered_candidates)
        self.candidates = filtered_candidates

        self.logger.info(f"  📊 Filtered {n_filtered} candidates by lift requirements")
        tprint_info(
            f"  📊 [LGID] After lift filter: kept {len(self.candidates)}/{total_before}, "
            f"filtered {n_filtered} by MI/R² lift"
        )

    def _apply_lasso_selection(self, features: pd.DataFrame, target: pd.Series):
        """Apply LASSO regularization for interaction selection."""
        self.logger.info("  🔧 Applying LASSO regularization...")
        tprint_info("  🔧 [LGID] Applying LASSO regularization to interaction candidates...")

        # Build candidate dataframe
        candidate_features = {}
        for cand in self.candidates:
            v1 = features[cand.feature1].values
            v2 = features[cand.feature2].values
            interaction_series, _ = self._apply_operation(
                v1, v2, cand.feature1, cand.feature2, cand.operation, features.index
            )
            if interaction_series is not None:
                candidate_features[cand.name] = interaction_series

        if len(candidate_features) == 0:
            self.logger.warning("  ⚠️ No candidates to apply LASSO to")
            return

        candidate_df = pd.DataFrame(candidate_features)

        # Apply LASSO
        try:
            if self.config.lasso_alpha is None:
                # Use CV to find optimal alpha
                lasso = LassoCV(
                    cv=self.config.lasso_cv_folds,
                    max_iter=self.config.lasso_max_iter,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                )
            else:
                lasso = Lasso(
                    alpha=self.config.lasso_alpha,
                    max_iter=self.config.lasso_max_iter,
                    random_state=self.config.random_state
                )

            lasso.fit(candidate_df, target)

            # Select features with non-zero coefficients
            selected_mask = np.abs(lasso.coef_) > 1e-6
            n_selected = int(np.sum(selected_mask))

            # If LASSO converged to an all-zero solution, fall back to using
            # all MI-lift-filtered candidates so that downstream category
            # limits can still pick a sparse, high-signal subset instead of
            # returning an empty interaction set.
            if n_selected == 0:
                self.logger.warning(
                    "  ⚠️ LASSO selected 0 interactions; "
                    "falling back to MI/SHAP-ranked candidates before category limits",
                )
                tprint_info(
                    "  📊 [LGID] LASSO produced an all-zero solution; "
                    "marking all lift-filtered candidates as selected for "
                    "category-aware limiting"
                )
                for cand in self.candidates:
                    cand.selected = True
                return

            selected_names = set(candidate_df.columns[selected_mask])

            # Mark candidates as selected
            for cand in self.candidates:
                if cand.name in selected_names:
                    cand.selected = True

            self.logger.info(f"  ✅ LASSO selected {n_selected}/{len(self.candidates)} interactions")
            tprint_info(
                f"  📊 [LGID] After LASSO: selected {n_selected}/{len(self.candidates)} candidates "
                f"with non-zero coefficients"
            )

        except Exception as e:
            self.logger.warning(f"  ⚠️ LASSO selection failed: {e}")
            # Fallback: select all candidates
            for cand in self.candidates:
                cand.selected = True

    def _apply_category_limits(self):
        """Apply per-category-pair limits to prevent over-representation."""
        self.logger.info("  🔧 Applying category-pair limits...")
        tprint_info("  🔧 [LGID] Applying category-pair limits to selected interactions...")

        # Group by category pair
        category_pair_groups: Dict[Tuple[str, str], List[InteractionCandidate]] = {}

        for cand in self.candidates:
            if not cand.selected:
                continue

            # Normalize category pair (always sort to get consistent key)
            cat_pair = tuple(sorted([cand.category1, cand.category2]))

            if cat_pair not in category_pair_groups:
                category_pair_groups[cat_pair] = []
            category_pair_groups[cat_pair].append(cand)

        # Apply limits per category pair
        self.selected_interactions = []

        for cat_pair, cands in category_pair_groups.items():
            # Sort by composite score (MI + SHAP)
            cands.sort(
                key=lambda c: (
                    self.config.mi_weight * c.mi_score +
                    self.config.shap_weight * c.shap_interaction_score
                ),
                reverse=True
            )

            # Take top N per category pair
            n_to_select = min(len(cands), self.config.max_interactions_per_category_pair)
            self.selected_interactions.extend(cands[:n_to_select])

        self.logger.info(f"  ✅ Selected {len(self.selected_interactions)} interactions after category limits")
        tprint_info(
            f"  📊 [LGID] After category limits: {len(self.selected_interactions)} final interactions "
            f"across {len(category_pair_groups)} category pairs"
        )

    def _build_interaction_dataframe(self, features: pd.DataFrame) -> pd.DataFrame:
        """Build final interaction dataframe from selected interactions."""
        interaction_dict = {}

        for cand in self.selected_interactions:
            v1 = features[cand.feature1].values
            v2 = features[cand.feature2].values
            interaction_series, _ = self._apply_operation(
                v1, v2, cand.feature1, cand.feature2, cand.operation, features.index
            )

            if interaction_series is not None:
                interaction_dict[cand.name] = interaction_series

        return pd.DataFrame(interaction_dict, index=features.index)

    def _build_metadata(self) -> Dict[str, Any]:
        """Build metadata dictionary."""
        metadata = {
            'total_candidates': len(self.candidates),
            'selected_interactions': len(self.selected_interactions),
            'config': {
                'min_r2_lift': self.config.min_r2_lift,
                'min_mi_lift': self.config.min_mi_lift,
                'use_lasso': self.config.use_lasso,
                'use_mi_scoring': self.config.use_mi_scoring,
                'use_shap_scoring': self.config.use_shap_scoring,
            },
            'selected_interaction_details': [
                {
                    'name': cand.name,
                    'feature1': cand.feature1,
                    'feature2': cand.feature2,
                    'operation': cand.operation,
                    'mi_score': cand.mi_score,
                    'shap_interaction_score': cand.shap_interaction_score,
                    'mi_lift': cand.mi_lift,
                    'category1': cand.category1,
                    'category2': cand.category2,
                }
                for cand in self.selected_interactions
            ],
            'category_pair_distribution': self._get_category_pair_distribution(),
        }

        return metadata

    def _get_category_pair_distribution(self) -> Dict[str, int]:
        """Get distribution of selected interactions by category pair."""
        distribution = {}

        for cand in self.selected_interactions:
            cat_pair = tuple(sorted([cand.category1, cand.category2]))
            key = f"{cat_pair[0]}_x_{cat_pair[1]}"
            distribution[key] = distribution.get(key, 0) + 1

        return distribution
