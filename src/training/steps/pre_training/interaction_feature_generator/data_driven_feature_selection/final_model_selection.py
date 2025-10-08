"""
Final Model Selection

This module implements the final model-level selection using stability selection,
FDR control, and group heredity to select the final set of features for the model.

Key Features:
- Stability selection with block bootstrap
- FDR control for multiple testing
- Group heredity for interactions
- LightGBM with depth constraints
- Final feature count targeting
"""

import logging
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats

# Robust variance estimation utilities
try:
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hac
    STATS_MODELS_AVAILABLE = True
except ImportError:
    STATS_MODELS_AVAILABLE = False
    sm = None
    cov_hac = None
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor

# Import utilities
from .utils import FeatureGeneratorWrapper
from feature_engineering_roadmap.feature_registry import FeatureRegistry
from .interaction_generator import InteractionFeature
from .config import FinalSelectionConfig
from src.training.steps.pre_training.validation.schemas import SplitAwareScaler

# Import matrix operations for efficient computation
try:
    from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
    from src.utils.matrix_operations.batch_operations import batch_correlation_analysis
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

# Import LightGBM if available
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Import utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning, tprint_success, tprint_performance
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class FinalSelectionResult:
    """Result of final model selection."""
    final_feature_names: List[str]
    final_feature_matrix: Optional[np.ndarray]
    selection_frequencies: Dict[str, float]
    importance_scores: Dict[str, float]
    fdr_controlled_features: List[str]
    group_heredity_features: List[str]
    execution_time: float
    n_features_selected: int
    target_achieved: bool

    family_contributions: Dict[str, float] = field(default_factory=dict)
    dropped_families: List[str] = field(default_factory=list)
    retained_families: List[str] = field(default_factory=list)
    split_metadata: Optional[Dict[str, np.ndarray]] = None

    # Performance metrics
    matrix_ops_used: int = 0
    vectorized_ops: int = 0
    stability_selections: int = 0
    fdr_controls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        summary = {
            'final_feature_names': self.final_feature_names,
            'final_feature_matrix_shape': self.final_feature_matrix.shape if self.final_feature_matrix is not None else None,
            'selection_frequencies': self.selection_frequencies,
            'importance_scores': self.importance_scores,
            'fdr_controlled_features': self.fdr_controlled_features,
            'group_heredity_features': self.group_heredity_features,
            'execution_time': self.execution_time,
            'n_features_selected': self.n_features_selected,
            'target_achieved': self.target_achieved,
            'family_contributions': self.family_contributions,
            'dropped_families': self.dropped_families,
            'retained_families': self.retained_families,
            'matrix_ops_used': self.matrix_ops_used,
            'vectorized_ops': self.vectorized_ops,
            'stability_selections': self.stability_selections,
            'fdr_controls': self.fdr_controls,
            'split_metadata': {
                key: indices.tolist()
                for key, indices in (self.split_metadata or {}).items()
            }
        }
        tprint_info(
            "🧾 Final selection summary prepared",
            f"features={len(self.final_feature_names)}",
            f"families={len(self.family_contributions)}"
        )
        return summary


class FinalModelSelection:
    """Final model-level selection with stability selection and FDR control."""
    
    def __init__(self, config: FinalSelectionConfig, matrix_ops=None):
        self.config = config
        self.matrix_ops = matrix_ops
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        try:
            self.feature_registry = FeatureRegistry()
        except Exception as exc:
            self.logger.warning(f"Failed to initialize FeatureRegistry: {exc}")
            self.feature_registry = None
        else:
            tprint_info("📚 Feature registry initialised successfully")

        # Performance tracking
        self.performance_metrics = {
            'matrix_ops_used': 0,
            'vectorized_ops': 0,
            'stability_selections': 0,
            'fdr_controls': 0,
            'bootstrap_samples': 0
        }
        self._active_split_metadata: Optional[Dict[str, np.ndarray]] = None
        tprint_info(
            "⚙️ FinalModelSelection configured",
            f"target={self.config.target_feature_count}",
            f"model={self.config.model_type}"
        )
    
    def select_final_features(
        self,
        selected_wrappers: List[FeatureGeneratorWrapper],
        selected_interactions: List[InteractionFeature],
        data: pd.DataFrame,
        target: np.ndarray,
        split_metadata: Optional[Mapping[str, Sequence[int]]] = None,
    ) -> FinalSelectionResult:
        """Select final features using stability selection and FDR control."""
        start_time = time.time()
        
        try:
            tprint_info("🚀 Starting Final Model Selection")
            tprint_info(f"📊 Selecting from {len(selected_wrappers)} base features and {len(selected_interactions)} interactions")

            resolved_split_metadata = self._resolve_split_metadata(split_metadata, len(data))
            self._active_split_metadata = resolved_split_metadata

            # Generate feature matrix
            feature_matrix, feature_names, feature_metadata = self._generate_feature_matrix(
                selected_wrappers, selected_interactions, data
            )

            if feature_matrix is None or feature_matrix.shape[1] == 0:
                tprint_warning("⚠️ No features available for final selection")
                return self._create_empty_result(time.time() - start_time, resolved_split_metadata)
            
            # Stability selection
            if self.config.enable_stability_selection:
                tprint_info("🔍 Running stability selection...")
                selection_frequencies = self._run_stability_selection(feature_matrix, target, feature_names)
            else:
                selection_frequencies = {name: 1.0 for name in feature_names}
            
            # FDR control
            if self.config.enable_fdr_control:
                tprint_info("📊 Applying FDR control...")
                fdr_controlled_features = self._apply_fdr_control(feature_matrix, target, feature_names)
            else:
                fdr_controlled_features = feature_names
            
            # Group heredity
            if self.config.enable_group_heredity:
                tprint_info("🔗 Applying group heredity...")
                group_heredity_features = self._apply_group_heredity(
                    fdr_controlled_features, selected_interactions
                )
            else:
                group_heredity_features = fdr_controlled_features

            group_regularized_features = list(group_heredity_features)
            group_contributions: Dict[str, float] = {}
            dropped_families: List[str] = []
            retained_families: List[str] = []

            if getattr(self.config, 'enable_group_regularization', False):
                tprint_info("🧮 Evaluating feature families...")
                (group_regularized_features,
                 group_contributions,
                 dropped_families) = self._apply_group_regularization(
                    feature_matrix,
                    target,
                    feature_names,
                    group_regularized_features,
                    feature_metadata,
                    resolved_split_metadata,
                )

                if not group_regularized_features and group_heredity_features:
                    warning_msg = (
                        "Group regularization removed all candidates; "
                        "reverting to pre-regularization set."
                    )
                    tprint_warning(f"⚠️ {warning_msg}")
                    self.logger.warning(warning_msg)
                    group_regularized_features = list(group_heredity_features)
                    dropped_families = []
                    group_contributions = {}

                if group_contributions:
                    retained_families = sorted(
                        [family for family in group_contributions.keys() if family not in dropped_families]
                    )
                else:
                    retained_families = sorted({
                        self._normalize_family_name(feature_metadata.get(name, {}).get('family')) or 'unassigned'
                        for name in group_regularized_features
                    }) if group_regularized_features else []
            else:
                retained_families = sorted({
                    self._normalize_family_name(feature_metadata.get(name, {}).get('family')) or 'unassigned'
                    for name in group_regularized_features
                }) if group_regularized_features else []

            # Final feature selection
            final_features = self._select_final_features(
                feature_matrix,
                target,
                feature_names,
                group_regularized_features,
                feature_metadata
            )

            final_family_snapshot = sorted({
                fam
                for feature in final_features
                for fam in self._extract_feature_families(feature_metadata.get(feature, {}))
                if fam
            })

            family_message = (
                f"Final feature families -> {', '.join(final_family_snapshot)}"
                if final_family_snapshot else
                "Final feature families -> none"
            )
            self.logger.info(family_message)
            tprint_info(family_message)

            # Generate final feature matrix
            final_matrix = self._generate_final_matrix(feature_matrix, feature_names, final_features)
            
            # Compute importance scores
            importance_scores = self._compute_importance_scores(final_matrix, target, final_features)
            
            execution_time = time.time() - start_time
            
            # Create result
            result = FinalSelectionResult(
                final_feature_names=final_features,
                final_feature_matrix=final_matrix,
                selection_frequencies=selection_frequencies,
                importance_scores=importance_scores,
                fdr_controlled_features=fdr_controlled_features,
                group_heredity_features=group_heredity_features,
                execution_time=execution_time,
                n_features_selected=len(final_features),
                target_achieved=self._check_target_achievement(len(final_features)),
                family_contributions=group_contributions,
                dropped_families=dropped_families,
                retained_families=retained_families,
                matrix_ops_used=self.performance_metrics['matrix_ops_used'],
                vectorized_ops=self.performance_metrics['vectorized_ops'],
                stability_selections=self.performance_metrics['stability_selections'],
                fdr_controls=self.performance_metrics['fdr_controls'],
                split_metadata=resolved_split_metadata,
            )

            self._log_group_regularization_summary(
                result.family_contributions,
                result.dropped_families,
                result.retained_families
            )

            tprint_success(f"✅ Final selection completed in {execution_time:.3f}s")
            tprint_success(f"📊 Selected {len(final_features)} final features")
            tprint_success(f"🎯 Target achieved: {result.target_achieved}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Final selection failed: {e}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            tprint_error(f"❌ Final selection failed after {execution_time:.3f}s: {e}")

            return self._create_empty_result(execution_time, self._active_split_metadata)

    def _lookup_feature_family(self, feature_name: str, fallback: Optional[str] = None) -> Optional[str]:
        """Resolve the canonical feature family for a given feature name."""
        if self.feature_registry is None:
            return fallback

        try:
            metadata = self.feature_registry.get_feature_metadata(feature_name)
            family = getattr(metadata, 'family', None)
            if family is None:
                return fallback
            return getattr(family, 'value', str(family))
        except Exception:
            return fallback

    @staticmethod
    def _normalize_family_name(family: Optional[Union[str, Any]]) -> Optional[str]:
        """Normalize family identifiers to lowercase strings."""
        if family is None:
            return None
        if hasattr(family, 'value'):
            family = family.value
        if isinstance(family, str):
            return family.lower()
        return str(family).lower()

    def _generate_feature_matrix(self, selected_wrappers: List[FeatureGeneratorWrapper],
                               selected_interactions: List[InteractionFeature],
                               data: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str], Dict[str, Dict[str, Any]]]:
        """Generate feature matrix from selected wrappers and interactions."""
        try:
            features = []
            feature_names = []
            feature_metadata: Dict[str, Dict[str, Any]] = {}
            wrapper_category_map: Dict[str, str] = {}
            wrapper_family_map: Dict[str, List[str]] = {}
            tprint_info("🧱 Building base feature matrix components…")

            # Generate base features
            for wrapper in selected_wrappers:
                try:
                    feature_values = self._generate_wrapper_feature(wrapper, data)
                    if feature_values is not None and len(feature_values) > 10:
                        features.append(feature_values)
                        feature_names.append(wrapper.name)
                        feature_type = self._determine_feature_bucket(wrapper)
                        resolved_family = self._lookup_feature_family(
                            wrapper.name,
                            fallback=self._normalize_family_name(wrapper.family)
                        )
                        normalized_family = self._normalize_family_name(resolved_family)
                        families = [normalized_family] if normalized_family else []

                        feature_metadata[wrapper.name] = {
                            'family': normalized_family,
                            'families': families if families else ['unassigned'],
                            'category': wrapper.category,
                            'feature_type': feature_type,
                            'source': 'base'
                        }
                        wrapper_category_map[wrapper.name] = feature_type
                        wrapper_family_map[wrapper.name] = families if families else ['unassigned']
                except Exception as e:
                    self.logger.debug(f"Failed to generate feature for {wrapper.name}: {e}")
                    continue

            # Generate interaction features
            for interaction in selected_interactions:
                try:
                    feature_values = self._generate_interaction_feature(interaction, data, selected_wrappers)
                    if feature_values is not None and len(feature_values) > 10:
                        features.append(feature_values)
                        feature_names.append(interaction.name)
                        feature_type = self._determine_interaction_bucket([
                            wrapper_category_map.get(interaction.parent1),
                            wrapper_category_map.get(interaction.parent2)
                        ])
                        parent_families = list({
                            fam
                            for parent in [interaction.parent1, interaction.parent2]
                            for fam in wrapper_family_map.get(parent, [])
                            if fam
                        })

                        if not parent_families:
                            parent_families = ['interaction']

                        primary_family = parent_families[0] if parent_families else 'interaction'

                        feature_metadata[interaction.name] = {
                            'family': primary_family,
                            'families': parent_families,
                            'category': interaction.interaction_type,
                            'feature_type': feature_type,
                            'parents': [interaction.parent1, interaction.parent2],
                            'source': 'interaction'
                        }
                        wrapper_family_map[interaction.name] = parent_families
                except Exception as e:
                    self.logger.debug(f"Failed to generate interaction {interaction.name}: {e}")
                    continue

            if not features:
                tprint_warning("⚠️ Feature generation produced no usable columns")
                return None, [], {}

            # Align all features to same length
            min_length = min(len(f) for f in features)
            aligned_features = [f[:min_length] for f in features]

            # Create feature matrix
            feature_matrix = np.column_stack(aligned_features)

            tprint_info(
                "📊 Generated feature matrix",
                f"shape={feature_matrix.shape}",
                f"features={len(feature_names)}"
            )
            return feature_matrix, feature_names, feature_metadata

        except Exception as e:
            self.logger.error(f"Failed to generate feature matrix: {e}")
            tprint_error(f"❌ Feature matrix generation failed: {e}")
            return None, [], {}
    
    def _generate_wrapper_feature(self, wrapper: FeatureGeneratorWrapper, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate feature values for a wrapper."""
        try:
            if hasattr(wrapper.generator, 'generate'):
                result = wrapper.generator.generate(data, lookback=20)  # Use default lookback
                
                if hasattr(result, 'data'):
                    return result.data.values
                elif isinstance(result, pd.Series):
                    return result.values
                elif isinstance(result, np.ndarray):
                    return result
                else:
                    return None
            else:
                return None
                
        except Exception as e:
            self.logger.debug(f"Failed to generate wrapper feature {wrapper.name}: {e}")
            return None
    
    def _generate_interaction_feature(self, interaction: InteractionFeature, data: pd.DataFrame,
                                    selected_wrappers: List[FeatureGeneratorWrapper]) -> Optional[np.ndarray]:
        """Generate feature values for an interaction."""
        try:
            # Find parent wrappers
            parent1_wrapper = next((w for w in selected_wrappers if w.name == interaction.parent1), None)
            parent2_wrapper = next((w for w in selected_wrappers if w.name == interaction.parent2), None)
            
            if parent1_wrapper is None or parent2_wrapper is None:
                return None
            
            # Generate parent features
            parent1_values = self._generate_wrapper_feature(parent1_wrapper, data)
            parent2_values = self._generate_wrapper_feature(parent2_wrapper, data)
            
            if parent1_values is None or parent2_values is None:
                return None
            
            # Align arrays
            min_length = min(len(parent1_values), len(parent2_values))
            p1 = parent1_values[:min_length]
            p2 = parent2_values[:min_length]
            
            # Compute interaction values
            if interaction.interaction_type == "multiplication":
                return p1 * p2
            elif interaction.interaction_type == "division":
                return np.where(np.abs(p2) > 1e-8, p1 / p2, np.zeros_like(p1))
            elif interaction.interaction_type == "addition":
                return p1 + p2
            elif interaction.interaction_type == "subtraction":
                return p1 - p2
            else:
                return None
                
        except Exception as e:
            self.logger.debug(f"Failed to generate interaction feature {interaction.name}: {e}")
            return None

    def _determine_feature_bucket(self, wrapper: FeatureGeneratorWrapper) -> str:
        """Infer the high-level category bucket for a wrapper."""
        name = (wrapper.name or '').lower()
        category = (wrapper.category or '').lower()
        family = (wrapper.family or '').lower()

        regime_terms = ['regime', 'state', 'market_state']
        embedding_terms = ['embedding', 'representation', 'autoencoder', 'encoder', 'ae']
        htf_terms = ['htf', 'higher_time', 'higher_tf', 'multi_time', 'multitime', 'mtf', 'higher timeframe']

        if any(term in category for term in regime_terms) or any(term in name for term in regime_terms):
            return 'regime'
        if any(term in category for term in embedding_terms) or any(term in name for term in embedding_terms):
            return 'embedding'
        if any(term in category for term in htf_terms) or any(term in name for term in htf_terms) or any(term in family for term in htf_terms):
            return 'htf'

        return 'engineered'

    def _determine_interaction_bucket(self, parent_buckets: List[Optional[str]]) -> str:
        """Infer category bucket for an interaction based on parent buckets."""
        valid_buckets = [bucket for bucket in parent_buckets if bucket]
        if not valid_buckets:
            return 'engineered'

        non_engineered = [bucket for bucket in valid_buckets if bucket != 'engineered']
        if not non_engineered:
            return 'engineered'

        if len(set(non_engineered)) == 1:
            return non_engineered[0]

        # Mixed parents – default to engineered to avoid double counting specialised quotas
        return 'engineered'

    def _apply_group_regularization(
        self,
        feature_matrix: np.ndarray,
        target: np.ndarray,
        feature_names: List[str],
        candidate_features: List[str],
        feature_metadata: Dict[str, Dict[str, Any]],
        split_metadata: Mapping[str, Sequence[int]],
    ) -> Tuple[List[str], Dict[str, float], List[str]]:
        """Evaluate family-level contributions and drop low-signal families."""
        try:
            if not candidate_features:
                tprint_warning("⚠️ No candidate features available for group regularization")
                return candidate_features, {}, []

            candidate_indices = [i for i, name in enumerate(feature_names) if name in candidate_features]

            if not candidate_indices:
                tprint_warning("⚠️ Candidate indices missing during group regularization")
                return candidate_features, {}, []

            candidate_names = [feature_names[i] for i in candidate_indices]
            X_candidate = feature_matrix[:, candidate_indices]
            tprint_info(
                "🧮 Evaluating group contributions",
                f"candidates={len(candidate_names)}"
            )

            feature_groups = {
                name: self._extract_feature_families(feature_metadata.get(name, {}))
                for name in candidate_names
            }

            contributions = self._estimate_group_contributions(
                X_candidate,
                target,
                candidate_names,
                feature_groups,
                split_metadata,
            )

            if not contributions:
                tprint_warning("⚠️ Unable to estimate family contributions")
                return candidate_features, contributions, []

            threshold = float(getattr(self.config, 'group_contribution_threshold', 0.0) or 0.0)
            threshold = max(0.0, threshold)

            dropped_families = [
                family for family, value in contributions.items()
                if family not in {'unassigned', 'interaction'} and value < threshold
            ]

            if dropped_families:
                tprint_warning(
                    "🧹 Dropping underperforming families",
                    ", ".join(sorted(dropped_families))
                )
            else:
                tprint_info("✅ No families dropped during regularization")

            retained_set = {
                name for name in candidate_names
                if not any(family in dropped_families for family in feature_groups.get(name, []))
            }

            retained_ordered = [name for name in candidate_features if name in retained_set]

            tprint_info(
                "📦 Group regularization retained",
                f"features={len(retained_ordered)}"
            )

            return retained_ordered, contributions, dropped_families

        except Exception as exc:
            self.logger.warning(f"Group regularization failed: {exc}")
            tprint_error(f"❌ Group regularization failed: {exc}")
            return candidate_features, {}, []

    def _extract_feature_families(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract normalized family assignments for a feature."""
        families: List[str] = []

        raw_families = metadata.get('families')
        if isinstance(raw_families, (list, tuple, set)):
            families.extend(
                [fam for fam in (self._normalize_family_name(f) for f in raw_families) if fam]
            )

        if not families:
            fallback = self._normalize_family_name(metadata.get('family'))
            if fallback:
                families.append(fallback)

        if not families:
            families.append('unassigned')

        return families

    def _estimate_group_contributions(
        self,
        X_candidate: np.ndarray,
        target: np.ndarray,
        candidate_names: List[str],
        feature_groups: Dict[str, List[str]],
        split_metadata: Mapping[str, Sequence[int]],
    ) -> Dict[str, float]:
        """Estimate contribution of each family using configured method."""
        method = str(getattr(self.config, 'group_regularization_method', 'shap') or 'shap').lower()

        try:
            if method == 'shap' and LIGHTGBM_AVAILABLE:
                tprint_info("📐 Estimating family contributions with SHAP")
                feature_scores = self._estimate_contributions_with_shap(
                    X_candidate,
                    target,
                    candidate_names
                )
            elif method == 'lasso':
                tprint_info("📐 Estimating family contributions with Lasso coefficients")
                feature_scores = self._estimate_contributions_with_lasso(
                    X_candidate,
                    target,
                    candidate_names,
                    split_metadata,
                )
            elif method == 'shap':
                raise RuntimeError('LightGBM not available for SHAP contributions')
            else:
                tprint_info("📐 Defaulting to Lasso-based family contributions")
                feature_scores = self._estimate_contributions_with_lasso(
                    X_candidate,
                    target,
                    candidate_names,
                    split_metadata,
                )
        except Exception as exc:
            self.logger.warning(
                f"Primary group contribution method '{method}' failed: {exc}"
            )
            tprint_warning(
                f"⚠️ Primary group contribution method '{method}' failed; falling back to Lasso"
            )
            feature_scores = self._estimate_contributions_with_lasso(
                X_candidate,
                target,
                candidate_names,
                split_metadata,
            )

        return self._aggregate_group_scores(feature_scores, feature_groups)

    @staticmethod
    def build_default_split_metadata(n_samples: int) -> Dict[str, np.ndarray]:
        """Create a simple chronological train/val/test split."""

        if n_samples < 3:
            raise ValueError("At least three samples are required to create default splits")

        indices = np.arange(n_samples, dtype=int)
        train_end = max(int(n_samples * 0.6), 1)
        val_end = max(train_end + max(int(n_samples * 0.2), 1), train_end + 1)
        if val_end >= n_samples:
            val_end = n_samples - 1

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        if val_indices.size == 0:
            val_indices = indices[train_end:train_end + 1]
        if test_indices.size == 0:
            test_indices = indices[-1:]
            if val_indices.size == 0 and train_indices.size > 1:
                val_indices = indices[train_end - 1:train_end]

        return {
            'train': np.array(train_indices, dtype=int, copy=True),
            'val': np.array(val_indices, dtype=int, copy=True),
            'test': np.array(test_indices, dtype=int, copy=True),
        }

    def _resolve_split_metadata(
        self,
        split_metadata: Optional[Mapping[str, Sequence[int]]],
        n_samples: int,
    ) -> Dict[str, np.ndarray]:
        """Validate provided split metadata or build default splits."""

        if split_metadata is None:
            default_splits = self.build_default_split_metadata(n_samples)
            self.logger.info("No split metadata provided; generated default splits")
            return default_splits

        normalized = SplitAwareScaler.normalize_split_indices(split_metadata)
        required = {'train', 'val', 'test'}
        missing = required.difference(normalized.keys())
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Split metadata missing required keys: {missing_list}")

        validated: Dict[str, np.ndarray] = {}
        for split_name, indices in normalized.items():
            if indices.size == 0:
                raise ValueError(f"Split '{split_name}' must contain at least one index")
            if np.any(indices < 0) or np.any(indices >= n_samples):
                raise ValueError(
                    f"Split '{split_name}' indices must be within the range [0, {n_samples})"
                )
            validated[split_name] = np.array(indices, dtype=int, copy=True)

        return validated

    def _estimate_contributions_with_shap(self,
                                          X_candidate: np.ndarray,
                                          target: np.ndarray,
                                          candidate_names: List[str]) -> Dict[str, float]:
        """Approximate SHAP contributions via LightGBM."""
        model = lgb.LGBMRegressor(
            n_estimators=min(256, max(50, getattr(self.config, 'n_estimators', 100))),
            learning_rate=getattr(self.config, 'learning_rate', 0.1),
            max_depth=getattr(self.config, 'max_depth', -1),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        tprint_info("🔍 Training LightGBM surrogate for SHAP contributions")
        model.fit(X_candidate, target)

        try:
            shap_values = model.predict(X_candidate, pred_contrib=True)
        except TypeError:
            shap_values = model.booster_.predict(X_candidate, pred_contrib=True)

        if shap_values.ndim != 2:
            raise ValueError('Unexpected SHAP value shape')

        feature_values = shap_values[:, :len(candidate_names)]
        mean_abs = np.mean(np.abs(feature_values), axis=0)

        return {
            name: float(value)
            for name, value in zip(candidate_names, mean_abs.tolist())
        }

    def _estimate_contributions_with_lasso(
        self,
        X_candidate: np.ndarray,
        target: np.ndarray,
        candidate_names: List[str],
        split_metadata: Mapping[str, Sequence[int]],
    ) -> Dict[str, float]:
        """Estimate contributions using Lasso coefficients as proxy."""

        normalized_splits = SplitAwareScaler.normalize_split_indices(split_metadata)
        if 'train' not in normalized_splits:
            raise ValueError("Split metadata must include a 'train' split")

        train_indices = normalized_splits['train']
        if train_indices.size < 2:
            raise ValueError("At least two training samples are required for LassoCV")

        try:
            scaler = SplitAwareScaler(StandardScaler(), normalized_splits)
            scaler.fit(X_candidate, split='train')
            scaled_views = {
                split: scaler.transform(X_candidate, split=split)
                for split in normalized_splits.keys()
            }
            X_train_scaled = scaled_views['train']
            y_array = np.asarray(target)
            if y_array.ndim > 1:
                y_array = y_array.reshape(-1)
            y_train = y_array[train_indices]

            n_train = train_indices.size
            cv_folds = min(5, max(2, n_train - 1))
            if cv_folds < 2 or cv_folds >= n_train:
                raise ValueError("Insufficient training samples for LassoCV")

            lasso = LassoCV(cv=cv_folds, random_state=42)
            tprint_info("🧷 Fitting Lasso model for contribution estimates")
            lasso.fit(X_train_scaled, y_train)
            coefs = np.abs(lasso.coef_)
        except Exception as exc:
            self.logger.debug(f"Lasso contribution estimation failed: {exc}")
            tprint_warning("⚠️ Lasso contribution estimation failed; using correlations")
            coefs = []
            X_train_raw = X_candidate[train_indices, :]
            y_array = np.asarray(target)
            if y_array.ndim > 1:
                y_array = y_array.reshape(-1)
            y_train = y_array[train_indices]
            for i in range(X_train_raw.shape[1]):
                try:
                    corr = np.corrcoef(X_train_raw[:, i], y_train)[0, 1]
                    coefs.append(abs(corr) if not np.isnan(corr) else 0.0)
                except Exception:
                    coefs.append(0.0)

        coefs = np.nan_to_num(np.asarray(coefs, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

        return {
            name: float(value)
            for name, value in zip(candidate_names, coefs.tolist())
        }

    def _aggregate_group_scores(self,
                                feature_scores: Dict[str, float],
                                feature_groups: Dict[str, List[str]]) -> Dict[str, float]:
        """Aggregate feature-level scores into family contributions."""
        contributions: Dict[str, float] = {}

        for feature_name, score in feature_scores.items():
            families = feature_groups.get(feature_name, ['unassigned'])

            if not families:
                families = ['unassigned']

            share = score / len(families) if families else score

            for family in families:
                contributions[family] = contributions.get(family, 0.0) + share

        total = sum(contributions.values())
        if total > 0:
            contributions = {
                family: value / total
                for family, value in contributions.items()
            }

        return dict(sorted(contributions.items(), key=lambda item: item[1], reverse=True))

    def _log_group_regularization_summary(self,
                                          contributions: Dict[str, float],
                                          dropped: List[str],
                                          retained: List[str]) -> None:
        """Log the outcome of the group regularization stage."""
        if not contributions:
            if retained:
                summary = ", ".join(retained)
                message = f"Group regularization not applied; families observed -> {summary}"
            else:
                message = "Group regularization contributions unavailable."
            self.logger.info(message)
            tprint_info(message)
            return

        contribution_summary = ", ".join(
            f"{family}: {value:.3f}" for family, value in contributions.items()
        )
        self.logger.info(f"Family contribution summary -> {contribution_summary}")
        tprint_info(f"Family contribution summary -> {contribution_summary}")

        if dropped:
            dropped_summary = ", ".join(sorted(dropped))
            message = f"Families removed by regularization -> {dropped_summary}"
            self.logger.info(message)
            tprint_warning(message)
        else:
            message = "No families removed by regularization."
            self.logger.info(message)
            tprint_info(message)

        if retained:
            retained_summary = ", ".join(retained)
            message = f"Families retained after regularization -> {retained_summary}"
            self.logger.info(message)
            tprint_success(message)
        else:
            message = "No families retained after regularization."
            self.logger.warning(message)
            tprint_warning(message)

    def _resolve_feature_bucket(self, feature_name: str, metadata: Dict[str, Dict[str, Any]]) -> str:
        """Resolve the bucket for a feature from metadata."""
        bucket = metadata.get(feature_name, {}).get('feature_type')
        if bucket in {'engineered', 'htf', 'regime', 'embedding'}:
            return bucket
        return 'engineered'
    
    def _run_stability_selection(self, feature_matrix: np.ndarray, target: np.ndarray,
                               feature_names: List[str]) -> Dict[str, float]:
        """Run stability selection with block bootstrap."""
        try:
            n_features = feature_matrix.shape[1]
            selection_counts = np.zeros(n_features)
            tprint_info(
                "🔁 Running stability selection",
                f"bootstrap_samples={self.config.n_bootstrap_samples}",
                f"features={n_features}"
            )

            # Run bootstrap samples
            for i in range(self.config.n_bootstrap_samples):
                try:
                    # Create bootstrap sample
                    bootstrap_indices = self._create_bootstrap_sample(len(target))
                    X_boot = feature_matrix[bootstrap_indices]
                    y_boot = target[bootstrap_indices]
                    
                    # Select features using the configured method
                    selected_indices = self._select_features_single_sample(X_boot, y_boot, feature_names)
                    
                    # Update selection counts
                    for idx in selected_indices:
                        selection_counts[idx] += 1
                    
                    self.performance_metrics['bootstrap_samples'] += 1

                except Exception as e:
                    self.logger.debug(f"Bootstrap sample {i} failed: {e}")
                    tprint_warning(f"⚠️ Bootstrap sample {i} failed: {e}")
                    continue
            
            # Convert to frequencies
            selection_frequencies = {}
            for i, name in enumerate(feature_names):
                frequency = selection_counts[i] / self.config.n_bootstrap_samples
                selection_frequencies[name] = frequency

            self.performance_metrics['stability_selections'] += 1
            tprint_info("📈 Stability selection frequencies computed")
            return selection_frequencies

        except Exception as e:
            self.logger.warning(f"Stability selection failed: {e}")
            tprint_error(f"❌ Stability selection failed: {e}")
            return {name: 1.0 for name in feature_names}

    def _create_bootstrap_sample(self, n_samples: int) -> np.ndarray:
        """Create bootstrap sample with block structure."""
        try:
            # Use block bootstrap for time series
            block_size = max(1, n_samples // 20)  # 20 blocks
            n_blocks = n_samples // block_size
            tprint_info(
                "📦 Creating bootstrap sample",
                f"block_size={block_size}",
                f"blocks={n_blocks}"
            )

            # Sample blocks with replacement
            block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
            
            # Create bootstrap indices
            bootstrap_indices = []
            for block_idx in block_indices:
                start_idx = block_idx * block_size
                end_idx = min(start_idx + block_size, n_samples)
                bootstrap_indices.extend(range(start_idx, end_idx))
            
            # Ensure we have the right number of samples
            bootstrap_indices = np.array(bootstrap_indices[:n_samples])

            return bootstrap_indices

        except Exception as e:
            self.logger.debug(f"Failed to create bootstrap sample: {e}")
            # Fallback to simple bootstrap
            tprint_warning("⚠️ Block bootstrap failed; using simple bootstrap")
            return np.random.choice(n_samples, size=n_samples, replace=True)
    
    def _select_features_single_sample(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str]) -> List[int]:
        """Select features for a single bootstrap sample."""
        try:
            if self.config.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
                tprint_info("🌳 Selecting features with LightGBM")
                return self._select_features_lightgbm(X, y, feature_names)
            elif self.config.model_type == "lasso":
                tprint_info("🧷 Selecting features with Lasso")
                return self._select_features_lasso(X, y, feature_names)
            elif self.config.model_type == "random_forest":
                tprint_info("🌲 Selecting features with Random Forest")
                return self._select_features_random_forest(X, y, feature_names)
            else:
                tprint_info("📊 Selecting features with univariate method")
                return self._select_features_univariate(X, y, feature_names)

        except Exception as e:
            self.logger.debug(f"Feature selection failed for single sample: {e}")
            tprint_warning(f"⚠️ Feature selection fallback triggered: {e}")
            return []
    
    def _select_features_lightgbm(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[int]:
        """Select features using LightGBM."""
        try:
            # Create LightGBM dataset
            train_data = lgb.Dataset(X, label=y)

            # Train model
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate,
                'n_estimators': self.config.n_estimators,
                'verbose': -1
            }
            
            model = lgb.train(params, train_data, num_boost_round=self.config.n_estimators)

            # Get feature importance
            importance = model.feature_importance(importance_type='gain')

            # Select top features
            n_select = min(len(feature_names), self.config.target_feature_count)
            top_indices = np.argsort(importance)[-n_select:]

            tprint_info(
                "🌳 LightGBM selected features",
                f"count={len(top_indices)}"
            )

            return top_indices.tolist()

        except Exception as e:
            self.logger.debug(f"LightGBM feature selection failed: {e}")
            tprint_warning("⚠️ LightGBM selection failed; using univariate scores")
            return self._select_features_univariate(X, y, feature_names)
    
    def _select_features_lasso(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[int]:
        """Select features using Lasso."""
        try:
            # Use LassoCV for automatic alpha selection
            lasso = LassoCV(cv=3, random_state=42)
            tprint_info("🧷 Fitting LassoCV for feature selection")
            lasso.fit(X, y)

            # Get non-zero coefficients
            non_zero_indices = np.where(np.abs(lasso.coef_) > 1e-6)[0]

            tprint_info(
                "🧷 Lasso selected features",
                f"count={len(non_zero_indices)}"
            )

            return non_zero_indices.tolist()

        except Exception as e:
            self.logger.debug(f"Lasso feature selection failed: {e}")
            tprint_warning("⚠️ Lasso selection failed; using univariate scores")
            return self._select_features_univariate(X, y, feature_names)
    
    def _select_features_random_forest(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[int]:
        """Select features using Random Forest."""
        try:
            # Train Random Forest
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=self.config.max_depth,
                random_state=42
            )
            tprint_info("🌲 Training Random Forest for feature selection")
            rf.fit(X, y)
            
            # Get feature importance
            importance = rf.feature_importances_
            
            # Select top features
            n_select = min(len(feature_names), self.config.target_feature_count)
            top_indices = np.argsort(importance)[-n_select:]

            tprint_info(
                "🌲 Random Forest selected features",
                f"count={len(top_indices)}"
            )

            return top_indices.tolist()

        except Exception as e:
            self.logger.debug(f"Random Forest feature selection failed: {e}")
            tprint_warning("⚠️ Random Forest selection failed; using univariate scores")
            return self._select_features_univariate(X, y, feature_names)
    
    def _select_features_univariate(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[int]:
        """Select features using univariate selection."""
        try:
            # Use F-test
            selector = SelectKBest(f_regression, k=min(len(feature_names), self.config.target_feature_count))
            selector.fit(X, y)

            indices = selector.get_support(indices=True).tolist()
            tprint_info(
                "📊 Univariate selection completed",
                f"count={len(indices)}"
            )

            return indices

        except Exception as e:
            self.logger.debug(f"Univariate feature selection failed: {e}")
            tprint_warning("⚠️ Univariate selection failed; returning default indices")
            return list(range(min(len(feature_names), self.config.target_feature_count)))

    def _compute_feature_p_value_hac(self, feature: np.ndarray, target: np.ndarray) -> float:
        """Compute a robust HAC-based p-value for a single feature."""
        if not STATS_MODELS_AVAILABLE:
            raise RuntimeError("statsmodels is not available for HAC estimation")

        mask = np.isfinite(feature) & np.isfinite(target)
        if mask.sum() <= 2:
            raise ValueError("Insufficient observations for HAC estimation")

        y = target[mask]
        x = feature[mask]

        if np.allclose(x, x[0]):
            raise ValueError("Feature has no variation")

        X = sm.add_constant(x, has_constant='add')
        model = sm.OLS(y, X)
        results = model.fit()

        # Newey-West lag selection following common sqrt(n) heuristic
        n_obs = len(y)
        max_lags = int(np.floor(np.sqrt(n_obs)))
        max_lags = max(1, min(max_lags, n_obs - 1))

        hac_cov = cov_hac(results, nlags=max_lags)

        # Coefficient for the feature is at index 1 (after the intercept)
        coef = results.params[1]
        variance = hac_cov[1, 1]
        if variance <= 0:
            raise ValueError("Non-positive HAC variance estimate")

        robust_se = np.sqrt(variance)
        t_stat = coef / robust_se
        df_resid = results.df_resid
        return 2 * (1 - stats.t.cdf(abs(t_stat), df_resid))

    @staticmethod
    def _compute_feature_p_value_iid(feature: np.ndarray, target: np.ndarray) -> float:
        """Compute an IID-based p-value using correlation."""
        mask = np.isfinite(feature) & np.isfinite(target)
        if mask.sum() <= 2:
            return 1.0

        x = feature[mask]
        y = target[mask]

        if np.allclose(x, x[0]):
            return 1.0

        correlation_matrix = np.corrcoef(x, y)
        if correlation_matrix.shape != (2, 2):
            return 1.0

        correlation = correlation_matrix[0, 1]
        if np.isnan(correlation):
            return 1.0

        n = len(y)
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation ** 2))
        return 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

    def _apply_fdr_control(self, feature_matrix: np.ndarray, target: np.ndarray,
                         feature_names: List[str]) -> List[str]:
        """Apply FDR control for multiple testing."""
        try:
            # Compute p-values for all features using HAC when possible
            p_values: List[float] = []
            hac_failures = 0
            tprint_info("🧪 Computing HAC-robust p-values for FDR control")
            for i in range(feature_matrix.shape[1]):
                column = feature_matrix[:, i]
                try:
                    p_value = self._compute_feature_p_value_hac(column, target)
                except Exception as exc:
                    hac_failures += 1
                    self.logger.debug(
                        "Falling back to IID p-value for feature %s due to HAC failure: %s",
                        feature_names[i],
                        exc
                    )
                    p_value = self._compute_feature_p_value_iid(column, target)

                if not np.isfinite(p_value) or np.isnan(p_value):
                    p_value = 1.0

                p_values.append(float(p_value))

            if hac_failures:
                tprint_warning(
                    "⚠️ HAC estimation fallback",
                    f"features_with_fallback={hac_failures}"
                )

            if not p_values:
                tprint_warning("⚠️ No p-values computed; retaining all features")
                return feature_names

            # Log summary statistics for downstream reporting
            p_values_array = np.array(p_values)
            tprint_info(
                "🧮 HAC-based p-values summary",
                f"min={np.nanmin(p_values_array):.4g}",
                f"median={np.nanmedian(p_values_array):.4g}",
                f"max={np.nanmax(p_values_array):.4g}"
            )

            # Apply Benjamini-Hochberg procedure
            sorted_indices = np.argsort(p_values_array)
            sorted_p_values = p_values_array[sorted_indices]

            # Compute critical values
            m = len(sorted_p_values)
            critical_values = np.arange(1, m + 1) * self.config.fdr_q_value / m

            # Find largest k such that p(k) <= critical_value(k)
            significant_indices = []
            for i in range(m):
                if sorted_p_values[i] <= critical_values[i]:
                    significant_indices.append(sorted_indices[i])
                else:
                    break

            # Log adjusted p-values for reporting
            adjusted_p_values = {
                feature_names[idx]: float(sorted_p_values[i])
                for i, idx in enumerate(sorted_indices)
            }
            self.logger.info(
                "Computed HAC-based p-values for FDR control",
                extra={"hac_p_values": adjusted_p_values}
            )

            # Return significant feature names
            fdr_controlled_features = [feature_names[i] for i in significant_indices]

            self.performance_metrics['fdr_controls'] += 1
            tprint_info(
                "✅ FDR control applied",
                f"features_retained={len(fdr_controlled_features)}"
            )
            return fdr_controlled_features

        except Exception as e:
            self.logger.warning(f"FDR control failed: {e}")
            tprint_error(f"❌ FDR control failed: {e}")
            return feature_names
    
    def _apply_group_heredity(self, fdr_controlled_features: List[str], 
                            selected_interactions: List[InteractionFeature]) -> List[str]:
        """Apply group heredity for interactions."""
        try:
            if not self.config.enable_group_heredity:
                tprint_info("🔗 Group heredity disabled; retaining FDR features")
                return fdr_controlled_features
            
            # Get parent features
            parent_features = set()
            for interaction in selected_interactions:
                parent_features.add(interaction.parent1)
                parent_features.add(interaction.parent2)

            tprint_info(
                "🔗 Evaluating group heredity",
                f"parent_features={len(parent_features)}"
            )
            
            # Check heredity requirements
            final_features = []
            
            for feature in fdr_controlled_features:
                # Check if this is a parent feature
                if feature in parent_features:
                    final_features.append(feature)
                # Check if this is an interaction and at least one parent is selected
                elif self._is_interaction_feature(feature, selected_interactions):
                    if self._check_interaction_heredity(feature, final_features, selected_interactions):
                        final_features.append(feature)
                else:
                    # Regular feature, add it
                    final_features.append(feature)

            tprint_info(
                "✅ Group heredity applied",
                f"features_retained={len(final_features)}"
            )
            return final_features

        except Exception as e:
            self.logger.warning(f"Group heredity failed: {e}")
            tprint_error(f"❌ Group heredity failed: {e}")
            return fdr_controlled_features
    
    def _is_interaction_feature(self, feature_name: str, selected_interactions: List[InteractionFeature]) -> bool:
        """Check if a feature is an interaction feature."""
        return any(interaction.name == feature_name for interaction in selected_interactions)
    
    def _check_interaction_heredity(self, feature_name: str, selected_features: List[str], 
                                  selected_interactions: List[InteractionFeature]) -> bool:
        """Check if interaction feature meets heredity requirements."""
        try:
            # Find the interaction
            interaction = next((i for i in selected_interactions if i.name == feature_name), None)
            if interaction is None:
                return True  # Not an interaction, allow it
            
            # Check if at least one parent is selected
            parent1_selected = interaction.parent1 in selected_features
            parent2_selected = interaction.parent2 in selected_features
            
            if self.config.min_parents_required == 1:
                return parent1_selected or parent2_selected
            elif self.config.min_parents_required == 2:
                return parent1_selected and parent2_selected
            else:
                return True

        except Exception as e:
            self.logger.debug(f"Failed to check heredity for {feature_name}: {e}")
            tprint_warning(f"⚠️ Heredity check failed for {feature_name}: {e}")
            return True
    
    def _select_final_features(self, feature_matrix: np.ndarray, target: np.ndarray,
                             feature_names: List[str], candidate_features: List[str],
                             feature_metadata: Dict[str, Dict[str, Any]]) -> List[str]:
        """Select final features while satisfying category quotas."""
        try:
            if len(candidate_features) <= self.config.target_feature_count:
                return candidate_features

            candidate_indices = [i for i, name in enumerate(feature_names) if name in candidate_features]
            if not candidate_indices:
                return candidate_features

            candidate_names = [feature_names[i] for i in candidate_indices]
            metadata = {name: feature_metadata.get(name, {}) for name in candidate_names}

            quotas = {}
            if getattr(self.config, 'category_quotas', None) is not None:
                quotas = {k: int(v) for k, v in self.config.category_quotas.to_dict().items() if int(v) > 0}

            desired_total = min(
                self.config.target_feature_count,
                self.config.max_feature_count,
                len(candidate_names)
            )

            if quotas:
                self._validate_quota_configuration(quotas, desired_total)
                self._validate_category_supply(candidate_names, metadata, quotas)

            X_candidate = feature_matrix[:, candidate_indices]
            ranking = self._rank_candidate_features(X_candidate, target, candidate_names)

            if not quotas:
                return ranking[:desired_total]

            selected: List[str] = []
            used: Set[str] = set()

            for category, quota in quotas.items():
                category_features = [
                    name for name in ranking
                    if self._resolve_feature_bucket(name, metadata) == category
                ]

                if len(category_features) < quota:
                    shortfall = quota - len(category_features)
                    self._log_quota_shortfall(category, quota, len(category_features))
                    raise ValueError(
                        f"Insufficient features for category '{category}' (missing {shortfall})."
                    )

                for name in category_features[:quota]:
                    if name not in used:
                        selected.append(name)
                        used.add(name)

            waitlist = [name for name in ranking if name not in used]

            for name in waitlist:
                if len(selected) >= desired_total:
                    break
                selected.append(name)
                used.add(name)

            if len(selected) > desired_total:
                selected = selected[:desired_total]

            self._log_final_allocation(selected, metadata)

            return selected

        except Exception as e:
            self.logger.warning(f"Final feature selection failed: {e}")
            return candidate_features[:self.config.target_feature_count]

    def _validate_quota_configuration(self, quotas: Dict[str, int], desired_total: int) -> None:
        """Validate that quota configuration is internally consistent."""
        total_quota = sum(max(0, quota) for quota in quotas.values())

        if total_quota <= 0:
            message = "Category quotas must reserve at least one feature."
            self.logger.error(message)
            tprint_error(message)
            raise ValueError(message)

        if total_quota > self.config.max_feature_count:
            message = (
                f"Category quotas ({total_quota}) exceed max feature count "
                f"({self.config.max_feature_count})."
            )
            self.logger.error(message)
            tprint_error(message)
            raise ValueError(message)

        if total_quota > desired_total:
            message = (
                f"Category quotas ({total_quota}) exceed desired total ({desired_total})."
            )
            self.logger.error(message)
            tprint_error(message)
            raise ValueError(message)

    def _validate_category_supply(self, candidate_names: List[str], metadata: Dict[str, Dict[str, Any]],
                                  quotas: Dict[str, int]) -> None:
        """Validate that candidate features can satisfy the configured quotas."""
        deficits = {}
        for category, quota in quotas.items():
            available = sum(
                1 for name in candidate_names
                if self._resolve_feature_bucket(name, metadata) == category
            )
            if available < quota:
                deficits[category] = quota - available

        if deficits:
            details = ", ".join(f"{cat}: short {shortfall}" for cat, shortfall in deficits.items())
            message = f"Insufficient supply to satisfy category quotas ({details})."
            self.logger.error(message)
            tprint_error(message)
            raise ValueError(message)

    def _rank_candidate_features(self, X_candidate: np.ndarray, target: np.ndarray,
                                 candidate_names: List[str]) -> List[str]:
        """Rank candidate features using the configured selection model."""
        scores: Optional[np.ndarray] = None

        try:
            if self.config.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
                tprint_info("📈 Ranking candidates with LightGBM importance")
                train_data = lgb.Dataset(X_candidate, label=target)
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'max_depth': self.config.max_depth,
                    'learning_rate': self.config.learning_rate,
                    'n_estimators': self.config.n_estimators,
                    'verbose': -1
                }
                model = lgb.train(params, train_data, num_boost_round=self.config.n_estimators)
                scores = model.feature_importance(importance_type='gain').astype(float)
            elif self.config.model_type == "lasso":
                tprint_info("📈 Ranking candidates with Lasso coefficients")
                lasso = LassoCV(cv=3, random_state=42)
                lasso.fit(X_candidate, target)
                scores = np.abs(lasso.coef_)
            elif self.config.model_type == "random_forest":
                tprint_info("📈 Ranking candidates with Random Forest importance")
                rf = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=self.config.max_depth,
                    random_state=42
                )
                rf.fit(X_candidate, target)
                scores = rf.feature_importances_.astype(float)
            else:
                tprint_info("📈 Ranking candidates with univariate F-test")
                scores, _ = f_regression(X_candidate, target)
        except Exception as e:
            self.logger.debug(f"Primary ranking method failed: {e}")
            tprint_warning(f"⚠️ Primary ranking method failed: {e}")

        if scores is None or len(scores) != len(candidate_names):
            try:
                tprint_info("🔁 Falling back to univariate F-test ranking")
                scores, _ = f_regression(X_candidate, target)
            except Exception as e:
                self.logger.warning(f"Fallback ranking failed: {e}")
                tprint_error(f"❌ Ranking fallback failed: {e}")
                return candidate_names

        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        ranking_pairs = sorted(
            zip(scores.tolist(), candidate_names),
            key=lambda item: item[0],
            reverse=True
        )

        tprint_info("🏁 Candidate ranking complete")

        return [name for _, name in ranking_pairs]

    def _log_quota_shortfall(self, category: str, requested: int, available: int) -> None:
        """Log a shortfall for a specific category."""
        message = (
            f"Category '{category}' quota not met: requested {requested}, available {available}."
        )
        self.logger.error(message)
        tprint_error(message)

    def _log_final_allocation(self, selected: List[str], metadata: Dict[str, Dict[str, Any]]) -> None:
        """Log the final allocation across categories for observability."""
        counts: Dict[str, int] = {}
        for name in selected:
            bucket = self._resolve_feature_bucket(name, metadata)
            counts[bucket] = counts.get(bucket, 0) + 1

        summary = ", ".join(f"{category}: {count}" for category, count in sorted(counts.items()))
        message = f"Final feature allocation by category -> {summary}"
        self.logger.info(message)
        tprint_info(message)

    def _generate_final_matrix(self, feature_matrix: np.ndarray, feature_names: List[str],
                             final_features: List[str]) -> Optional[np.ndarray]:
        """Generate final feature matrix with selected features."""
        try:
            if not final_features:
                tprint_warning("⚠️ No final features to assemble into matrix")
                return None

            # Get indices of final features
            final_indices = [i for i, name in enumerate(feature_names) if name in final_features]

            if not final_indices:
                tprint_warning("⚠️ Final feature indices could not be resolved")
                return None

            # Extract final features
            final_matrix = feature_matrix[:, final_indices]

            tprint_info(
                "🧾 Final matrix generated",
                f"shape={final_matrix.shape}"
            )
            return final_matrix

        except Exception as e:
            self.logger.warning(f"Failed to generate final matrix: {e}")
            tprint_error(f"❌ Final matrix generation failed: {e}")
            return None
    
    def _compute_importance_scores(self, final_matrix: np.ndarray, target: np.ndarray, 
                                 final_features: List[str]) -> Dict[str, float]:
        """Compute importance scores for final features."""
        try:
            if final_matrix is None or len(final_features) == 0:
                tprint_warning("⚠️ Skipping importance scores due to empty matrix")
                return {}

            importance_scores = {}

            for i, feature_name in enumerate(final_features):
                try:
                    # Compute correlation as importance score
                    correlation = np.corrcoef(final_matrix[:, i], target)[0, 1]
                    if not np.isnan(correlation):
                        importance_scores[feature_name] = abs(correlation)
                    else:
                        importance_scores[feature_name] = 0.0
                except Exception as e:
                    self.logger.debug(f"Failed to compute importance for {feature_name}: {e}")
                    importance_scores[feature_name] = 0.0

            tprint_info(
                "⭐ Computed importance scores",
                f"features={len(importance_scores)}"
            )
            return importance_scores

        except Exception as e:
            self.logger.warning(f"Failed to compute importance scores: {e}")
            tprint_error(f"❌ Importance score computation failed: {e}")
            return {}

    def _check_target_achievement(self, n_features: int) -> bool:
        """Check if target feature count is achieved."""
        achieved = (self.config.min_feature_count <= n_features <= self.config.max_feature_count)
        tprint_info(
            "🎯 Target feature range check",
            f"count={n_features}",
            f"min={self.config.min_feature_count}",
            f"max={self.config.max_feature_count}",
            f"achieved={achieved}"
        )
        return achieved

    def _create_empty_result(
        self,
        execution_time: float,
        split_metadata: Optional[Dict[str, np.ndarray]] = None,
    ) -> FinalSelectionResult:
        """Create empty result for error cases."""
        tprint_warning(
            "⚠️ Returning empty FinalSelectionResult",
            f"execution_time={execution_time:.3f}s"
        )
        return FinalSelectionResult(
            final_feature_names=[],
            final_feature_matrix=None,
            selection_frequencies={},
            importance_scores={},
            fdr_controlled_features=[],
            group_heredity_features=[],
            execution_time=execution_time,
            n_features_selected=0,
            target_achieved=False,
            family_contributions={},
            dropped_families=[],
            retained_families=[],
            split_metadata=split_metadata,
        )
