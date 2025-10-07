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
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor

# Import utilities
from .utils import FeatureGeneratorWrapper
from .interaction_generator import InteractionFeature
from .config import FinalSelectionConfig

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
    
    # Performance metrics
    matrix_ops_used: int = 0
    vectorized_ops: int = 0
    stability_selections: int = 0
    fdr_controls: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'final_feature_names': self.final_feature_names,
            'final_feature_matrix_shape': self.final_feature_matrix.shape if self.final_feature_matrix is not None else None,
            'selection_frequencies': self.selection_frequencies,
            'importance_scores': self.importance_scores,
            'fdr_controlled_features': self.fdr_controlled_features,
            'group_heredity_features': self.group_heredity_features,
            'execution_time': self.execution_time,
            'n_features_selected': self.n_features_selected,
            'target_achieved': self.target_achieved,
            'matrix_ops_used': self.matrix_ops_used,
            'vectorized_ops': self.vectorized_ops,
            'stability_selections': self.stability_selections,
            'fdr_controls': self.fdr_controls
        }


class FinalModelSelection:
    """Final model-level selection with stability selection and FDR control."""
    
    def __init__(self, config: FinalSelectionConfig, matrix_ops=None):
        self.config = config
        self.matrix_ops = matrix_ops
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance tracking
        self.performance_metrics = {
            'matrix_ops_used': 0,
            'vectorized_ops': 0,
            'stability_selections': 0,
            'fdr_controls': 0,
            'bootstrap_samples': 0
        }
    
    def select_final_features(self, selected_wrappers: List[FeatureGeneratorWrapper], 
                            selected_interactions: List[InteractionFeature],
                            data: pd.DataFrame, target: np.ndarray) -> FinalSelectionResult:
        """Select final features using stability selection and FDR control."""
        start_time = time.time()
        
        try:
            tprint_info("🚀 Starting Final Model Selection")
            tprint_info(f"📊 Selecting from {len(selected_wrappers)} base features and {len(selected_interactions)} interactions")
            
            # Generate feature matrix
            feature_matrix, feature_names = self._generate_feature_matrix(
                selected_wrappers, selected_interactions, data
            )
            
            if feature_matrix is None or feature_matrix.shape[1] == 0:
                tprint_warning("⚠️ No features available for final selection")
                return self._create_empty_result(time.time() - start_time)
            
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
            
            # Final feature selection
            final_features = self._select_final_features(
                feature_matrix, target, feature_names, group_heredity_features
            )
            
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
                matrix_ops_used=self.performance_metrics['matrix_ops_used'],
                vectorized_ops=self.performance_metrics['vectorized_ops'],
                stability_selections=self.performance_metrics['stability_selections'],
                fdr_controls=self.performance_metrics['fdr_controls']
            )
            
            tprint_success(f"✅ Final selection completed in {execution_time:.3f}s")
            tprint_success(f"📊 Selected {len(final_features)} final features")
            tprint_success(f"🎯 Target achieved: {result.target_achieved}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Final selection failed: {e}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            
            return self._create_empty_result(execution_time)
    
    def _generate_feature_matrix(self, selected_wrappers: List[FeatureGeneratorWrapper], 
                               selected_interactions: List[InteractionFeature], 
                               data: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Generate feature matrix from selected wrappers and interactions."""
        try:
            features = []
            feature_names = []
            
            # Generate base features
            for wrapper in selected_wrappers:
                try:
                    feature_values = self._generate_wrapper_feature(wrapper, data)
                    if feature_values is not None and len(feature_values) > 10:
                        features.append(feature_values)
                        feature_names.append(wrapper.name)
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
                except Exception as e:
                    self.logger.debug(f"Failed to generate interaction {interaction.name}: {e}")
                    continue
            
            if not features:
                return None, []
            
            # Align all features to same length
            min_length = min(len(f) for f in features)
            aligned_features = [f[:min_length] for f in features]
            
            # Create feature matrix
            feature_matrix = np.column_stack(aligned_features)
            
            tprint_info(f"📊 Generated feature matrix: {feature_matrix.shape}")
            return feature_matrix, feature_names
            
        except Exception as e:
            self.logger.error(f"Failed to generate feature matrix: {e}")
            return None, []
    
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
    
    def _run_stability_selection(self, feature_matrix: np.ndarray, target: np.ndarray, 
                               feature_names: List[str]) -> Dict[str, float]:
        """Run stability selection with block bootstrap."""
        try:
            n_features = feature_matrix.shape[1]
            selection_counts = np.zeros(n_features)
            
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
                    continue
            
            # Convert to frequencies
            selection_frequencies = {}
            for i, name in enumerate(feature_names):
                frequency = selection_counts[i] / self.config.n_bootstrap_samples
                selection_frequencies[name] = frequency
            
            self.performance_metrics['stability_selections'] += 1
            return selection_frequencies
            
        except Exception as e:
            self.logger.warning(f"Stability selection failed: {e}")
            return {name: 1.0 for name in feature_names}
    
    def _create_bootstrap_sample(self, n_samples: int) -> np.ndarray:
        """Create bootstrap sample with block structure."""
        try:
            # Use block bootstrap for time series
            block_size = max(1, n_samples // 20)  # 20 blocks
            n_blocks = n_samples // block_size
            
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
            return np.random.choice(n_samples, size=n_samples, replace=True)
    
    def _select_features_single_sample(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str]) -> List[int]:
        """Select features for a single bootstrap sample."""
        try:
            if self.config.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
                return self._select_features_lightgbm(X, y, feature_names)
            elif self.config.model_type == "lasso":
                return self._select_features_lasso(X, y, feature_names)
            elif self.config.model_type == "random_forest":
                return self._select_features_random_forest(X, y, feature_names)
            else:
                return self._select_features_univariate(X, y, feature_names)
                
        except Exception as e:
            self.logger.debug(f"Feature selection failed for single sample: {e}")
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
            
            return top_indices.tolist()
            
        except Exception as e:
            self.logger.debug(f"LightGBM feature selection failed: {e}")
            return self._select_features_univariate(X, y, feature_names)
    
    def _select_features_lasso(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[int]:
        """Select features using Lasso."""
        try:
            # Use LassoCV for automatic alpha selection
            lasso = LassoCV(cv=3, random_state=42)
            lasso.fit(X, y)
            
            # Get non-zero coefficients
            non_zero_indices = np.where(np.abs(lasso.coef_) > 1e-6)[0]
            
            return non_zero_indices.tolist()
            
        except Exception as e:
            self.logger.debug(f"Lasso feature selection failed: {e}")
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
            rf.fit(X, y)
            
            # Get feature importance
            importance = rf.feature_importances_
            
            # Select top features
            n_select = min(len(feature_names), self.config.target_feature_count)
            top_indices = np.argsort(importance)[-n_select:]
            
            return top_indices.tolist()
            
        except Exception as e:
            self.logger.debug(f"Random Forest feature selection failed: {e}")
            return self._select_features_univariate(X, y, feature_names)
    
    def _select_features_univariate(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[int]:
        """Select features using univariate selection."""
        try:
            # Use F-test
            selector = SelectKBest(f_regression, k=min(len(feature_names), self.config.target_feature_count))
            selector.fit(X, y)
            
            return selector.get_support(indices=True).tolist()
            
        except Exception as e:
            self.logger.debug(f"Univariate feature selection failed: {e}")
            return list(range(min(len(feature_names), self.config.target_feature_count)))
    
    def _apply_fdr_control(self, feature_matrix: np.ndarray, target: np.ndarray, 
                         feature_names: List[str]) -> List[str]:
        """Apply FDR control for multiple testing."""
        try:
            # Compute p-values for all features
            p_values = []
            for i in range(feature_matrix.shape[1]):
                try:
                    # Compute correlation and p-value
                    correlation = np.corrcoef(feature_matrix[:, i], target)[0, 1]
                    if not np.isnan(correlation):
                        # Approximate p-value from correlation
                        n = len(target)
                        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                        p_values.append(p_value)
                    else:
                        p_values.append(1.0)
                except Exception as e:
                    self.logger.debug(f"Failed to compute p-value for feature {i}: {e}")
                    p_values.append(1.0)
            
            # Apply Benjamini-Hochberg procedure
            sorted_indices = np.argsort(p_values)
            sorted_p_values = np.array(p_values)[sorted_indices]
            
            # Compute critical values
            m = len(p_values)
            critical_values = np.arange(1, m + 1) * self.config.fdr_q_value / m
            
            # Find largest k such that p(k) <= critical_value(k)
            significant_indices = []
            for i in range(m):
                if sorted_p_values[i] <= critical_values[i]:
                    significant_indices.append(sorted_indices[i])
                else:
                    break
            
            # Return significant feature names
            fdr_controlled_features = [feature_names[i] for i in significant_indices]
            
            self.performance_metrics['fdr_controls'] += 1
            return fdr_controlled_features
            
        except Exception as e:
            self.logger.warning(f"FDR control failed: {e}")
            return feature_names
    
    def _apply_group_heredity(self, fdr_controlled_features: List[str], 
                            selected_interactions: List[InteractionFeature]) -> List[str]:
        """Apply group heredity for interactions."""
        try:
            if not self.config.enable_group_heredity:
                return fdr_controlled_features
            
            # Get parent features
            parent_features = set()
            for interaction in selected_interactions:
                parent_features.add(interaction.parent1)
                parent_features.add(interaction.parent2)
            
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
            
            return final_features
            
        except Exception as e:
            self.logger.warning(f"Group heredity failed: {e}")
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
            return True
    
    def _select_final_features(self, feature_matrix: np.ndarray, target: np.ndarray, 
                             feature_names: List[str], candidate_features: List[str]) -> List[str]:
        """Select final features to meet target count."""
        try:
            if len(candidate_features) <= self.config.target_feature_count:
                return candidate_features
            
            # Get indices of candidate features
            candidate_indices = [i for i, name in enumerate(feature_names) if name in candidate_features]
            
            if not candidate_indices:
                return candidate_features
            
            # Select features using the configured method
            X_candidate = feature_matrix[:, candidate_indices]
            selected_indices = self._select_features_single_sample(X_candidate, target, candidate_features)
            
            # Map back to feature names
            final_features = [candidate_features[i] for i in selected_indices]
            
            return final_features
            
        except Exception as e:
            self.logger.warning(f"Final feature selection failed: {e}")
            return candidate_features[:self.config.target_feature_count]
    
    def _generate_final_matrix(self, feature_matrix: np.ndarray, feature_names: List[str], 
                             final_features: List[str]) -> Optional[np.ndarray]:
        """Generate final feature matrix with selected features."""
        try:
            if not final_features:
                return None
            
            # Get indices of final features
            final_indices = [i for i, name in enumerate(feature_names) if name in final_features]
            
            if not final_indices:
                return None
            
            # Extract final features
            final_matrix = feature_matrix[:, final_indices]
            
            return final_matrix
            
        except Exception as e:
            self.logger.warning(f"Failed to generate final matrix: {e}")
            return None
    
    def _compute_importance_scores(self, final_matrix: np.ndarray, target: np.ndarray, 
                                 final_features: List[str]) -> Dict[str, float]:
        """Compute importance scores for final features."""
        try:
            if final_matrix is None or len(final_features) == 0:
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
            
            return importance_scores
            
        except Exception as e:
            self.logger.warning(f"Failed to compute importance scores: {e}")
            return {}
    
    def _check_target_achievement(self, n_features: int) -> bool:
        """Check if target feature count is achieved."""
        return (self.config.min_feature_count <= n_features <= self.config.max_feature_count)
    
    def _create_empty_result(self, execution_time: float) -> FinalSelectionResult:
        """Create empty result for error cases."""
        return FinalSelectionResult(
            final_feature_names=[],
            final_feature_matrix=None,
            selection_frequencies={},
            importance_scores={},
            fdr_controlled_features=[],
            group_heredity_features=[],
            execution_time=execution_time,
            n_features_selected=0,
            target_achieved=False
        )