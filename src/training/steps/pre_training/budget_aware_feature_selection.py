"""
Budget-Aware Feature Selection

This module extends the final feature selection to include budget constraints
for interaction features and cross-timeframe features generated during
interactive_feature_generation.

Key Features:
- Separate budget allocation for different feature types
- Knapsack-style optimization for feature selection
- Integration with existing final_feature_selection pipeline
- Support for min/max constraints per feature type
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import time
from pathlib import Path

# Import existing budgeted selection logic
try:
    from src.training.steps.pre_training.interaction_feature_generator.data_driven_feature_selection.budgeted_selection import (
        BudgetedFeatureSelection, BudgetedSelectionResult, BudgetConfig
    )
    from src.training.steps.pre_training.interaction_feature_generator.data_driven_feature_selection.config import (
        FeatureFamily
    )
    BUDGETED_SELECTION_AVAILABLE = True
except ImportError:
    BUDGETED_SELECTION_AVAILABLE = False

# Import matrix operations
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        correlation_matrix_gpu,
        matrix_correlation_analysis
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
        tprint_debug, tprint_performance
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class FeatureTypeBudget:
    """Budget configuration for a specific feature type."""
    feature_type: str  # 'base', 'interaction', 'cross_timeframe'
    min_features: int = 0
    max_features: int = 100
    target_features: int = 60  # Target number of features
    budget_ms: float = 50.0  # Budget in milliseconds
    priority_weight: float = 1.0  # Priority weight for this feature type
    cost_per_feature_ms: float = 1.0  # Estimated cost per feature in ms
    
    def __post_init__(self):
        """Validate budget configuration."""
        if self.min_features < 0:
            raise ValueError(f"min_features must be non-negative, got {self.min_features}")
        if self.max_features < self.min_features:
            raise ValueError(f"max_features ({self.max_features}) must be >= min_features ({self.min_features})")
        if self.budget_ms <= 0:
            raise ValueError(f"budget_ms must be positive, got {self.budget_ms}")
        if self.priority_weight <= 0:
            raise ValueError(f"priority_weight must be positive, got {self.priority_weight}")


@dataclass
class BudgetAwareSelectionConfig:
    """Configuration for budget-aware feature selection."""
    # Feature type budgets - Updated with new targets
    base_features: FeatureTypeBudget = field(default_factory=lambda: FeatureTypeBudget(
        feature_type='base',
        min_features=40,
        max_features=80,
        target_features=60,  # Main target for base features
        budget_ms=30.0,
        priority_weight=1.0,
        cost_per_feature_ms=0.5
    ))
    
    interaction_features: FeatureTypeBudget = field(default_factory=lambda: FeatureTypeBudget(
        feature_type='interaction',
        min_features=5,
        max_features=15,
        target_features=10,  # Target 10 interaction features
        budget_ms=15.0,
        priority_weight=0.8,
        cost_per_feature_ms=1.0
    ))
    
    cross_timeframe_features: FeatureTypeBudget = field(default_factory=lambda: FeatureTypeBudget(
        feature_type='cross_timeframe',
        min_features=3,
        max_features=10,
        target_features=6,  # Target 6 cross-timeframe features
        budget_ms=10.0,
        priority_weight=0.7,
        cost_per_feature_ms=1.2
    ))
    
    # Global settings
    total_budget_ms: float = 100.0
    enable_diversification: bool = True
    diversification_penalty: float = 0.1
    correlation_threshold: float = 0.8
    
    # Quality thresholds
    min_importance_threshold: float = 0.001
    min_correlation_with_target: float = 0.01
    
    # Fallback settings
    fallback_to_uniform: bool = True
    uniform_allocation_ratio: float = 0.6  # 60% of total budget for base features


@dataclass
class FeatureTypeSelectionResult:
    """Result of feature selection for a specific feature type."""
    feature_type: str
    selected_features: List[str]
    rejected_features: List[str]
    selection_score: float
    budget_utilization: float
    execution_time: float
    selection_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetAwareSelectionResult:
    """Result of budget-aware feature selection."""
    # Overall results
    total_selected_features: List[str]
    total_rejected_features: List[str]
    total_budget_utilization: float
    total_execution_time: float
    
    # Per-type results
    base_features_result: FeatureTypeSelectionResult
    interaction_features_result: FeatureTypeSelectionResult
    cross_timeframe_features_result: FeatureTypeSelectionResult
    
    # Selection metrics
    selection_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_selected_features': self.total_selected_features,
            'total_rejected_features': self.total_rejected_features,
            'total_budget_utilization': self.total_budget_utilization,
            'total_execution_time': self.total_execution_time,
            'base_features_result': {
                'feature_type': self.base_features_result.feature_type,
                'selected_features': self.base_features_result.selected_features,
                'rejected_features': self.base_features_result.rejected_features,
                'selection_score': self.base_features_result.selection_score,
                'budget_utilization': self.base_features_result.budget_utilization,
                'execution_time': self.base_features_result.execution_time,
                'selection_metrics': self.base_features_result.selection_metrics
            },
            'interaction_features_result': {
                'feature_type': self.interaction_features_result.feature_type,
                'selected_features': self.interaction_features_result.selected_features,
                'rejected_features': self.interaction_features_result.rejected_features,
                'selection_score': self.interaction_features_result.selection_score,
                'budget_utilization': self.interaction_features_result.budget_utilization,
                'execution_time': self.interaction_features_result.execution_time,
                'selection_metrics': self.interaction_features_result.selection_metrics
            },
            'cross_timeframe_features_result': {
                'feature_type': self.cross_timeframe_features_result.feature_type,
                'selected_features': self.cross_timeframe_features_result.selected_features,
                'rejected_features': self.cross_timeframe_features_result.rejected_features,
                'selection_score': self.cross_timeframe_features_result.selection_score,
                'budget_utilization': self.cross_timeframe_features_result.budget_utilization,
                'execution_time': self.cross_timeframe_features_result.execution_time,
                'selection_metrics': self.cross_timeframe_features_result.selection_metrics
            },
            'selection_metrics': self.selection_metrics
        }


class BudgetAwareFeatureSelector:
    """Budget-aware feature selector for different feature types."""
    
    def __init__(self, config: BudgetAwareSelectionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize matrix operations if available
        self.matrix_ops = None
        if MATRIX_OPS_AVAILABLE:
            try:
                self.matrix_ops = get_unified_matrix_operations()
            except Exception as e:
                tprint_warning(f"Failed to initialize matrix operations: {e}")
        
        # Initialize budgeted selection if available
        self.budgeted_selector = None
        if BUDGETED_SELECTION_AVAILABLE:
            try:
                budget_config = BudgetConfig(
                    feature_compute_p99_budget_ms=config.total_budget_ms,
                    max_features_pre_selection=sum([
                        config.base_features.max_features,
                        config.interaction_features.max_features,
                        config.cross_timeframe_features.max_features
                    ]),
                    required_families=[FeatureFamily.MOMENTUM, FeatureFamily.VOLATILITY],
                    diversification_penalty=config.diversification_penalty,
                    correlation_threshold=config.correlation_threshold
                )
                self.budgeted_selector = BudgetedFeatureSelection(budget_config, self.matrix_ops)
            except Exception as e:
                tprint_warning(f"Failed to initialize budgeted selector: {e}")
    
    def select_features(
        self,
        base_features: pd.DataFrame,
        interaction_features: pd.DataFrame,
        cross_timeframe_features: pd.DataFrame,
        target: Optional[pd.Series] = None,
        feature_importance_scores: Optional[Dict[str, float]] = None
    ) -> BudgetAwareSelectionResult:
        """
        Select features with budget constraints for each feature type.
        
        Args:
            base_features: Base features DataFrame
            interaction_features: Interaction features DataFrame
            cross_timeframe_features: Cross-timeframe features DataFrame
            target: Target variable for supervised selection
            feature_importance_scores: Pre-computed feature importance scores
            
        Returns:
            BudgetAwareSelectionResult with selected features for each type
        """
        start_time = time.time()
        
        tprint_info("🚀 Starting Budget-Aware Feature Selection")
        tprint_info(f"📊 Base features: {base_features.shape[1]} columns")
        tprint_info(f"🔗 Interaction features: {interaction_features.shape[1]} columns")
        tprint_info(f"⏰ Cross-timeframe features: {cross_timeframe_features.shape[1]} columns")
        tprint_info(f"💰 Total budget: {self.config.total_budget_ms}ms")
        
        try:
            # Select features for each type
            base_result = self._select_features_for_type(
                base_features, 
                self.config.base_features, 
                target, 
                feature_importance_scores,
                'base'
            )
            
            interaction_result = self._select_features_for_type(
                interaction_features, 
                self.config.interaction_features, 
                target, 
                feature_importance_scores,
                'interaction'
            )
            
            cross_timeframe_result = self._select_features_for_type(
                cross_timeframe_features, 
                self.config.cross_timeframe_features, 
                target, 
                feature_importance_scores,
                'cross_timeframe'
            )
            
            # Combine results
            total_selected = (
                base_result.selected_features + 
                interaction_result.selected_features + 
                cross_timeframe_result.selected_features
            )
            
            total_rejected = (
                base_result.rejected_features + 
                interaction_result.rejected_features + 
                cross_timeframe_result.rejected_features
            )
            
            total_budget_utilization = (
                base_result.budget_utilization + 
                interaction_result.budget_utilization + 
                cross_timeframe_result.budget_utilization
            ) / 3.0  # Average utilization
            
            execution_time = time.time() - start_time
            
            # Create selection metrics
            selection_metrics = {
                'total_features_considered': len(total_selected) + len(total_rejected),
                'total_features_selected': len(total_selected),
                'selection_rate': len(total_selected) / max(1, len(total_selected) + len(total_rejected)),
                'budget_efficiency': total_budget_utilization,
                'feature_type_distribution': {
                    'base': len(base_result.selected_features),
                    'interaction': len(interaction_result.selected_features),
                    'cross_timeframe': len(cross_timeframe_result.selected_features)
                }
            }
            
            result = BudgetAwareSelectionResult(
                total_selected_features=total_selected,
                total_rejected_features=total_rejected,
                total_budget_utilization=total_budget_utilization,
                total_execution_time=execution_time,
                base_features_result=base_result,
                interaction_features_result=interaction_result,
                cross_timeframe_features_result=cross_timeframe_result,
                selection_metrics=selection_metrics
            )
            
            tprint_success(f"✅ Budget-aware selection completed in {execution_time:.3f}s")
            tprint_success(f"📊 Selected {len(total_selected)} total features")
            tprint_success(f"💰 Budget utilization: {total_budget_utilization:.1%}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ Budget-aware selection failed: {e}")
            self.logger.error(f"Budget-aware selection failed: {e}")
            
            # Return empty result
            empty_result = FeatureTypeSelectionResult(
                feature_type='empty',
                selected_features=[],
                rejected_features=[],
                selection_score=0.0,
                budget_utilization=0.0,
                execution_time=execution_time
            )
            
            return BudgetAwareSelectionResult(
                total_selected_features=[],
                total_rejected_features=[],
                total_budget_utilization=0.0,
                total_execution_time=execution_time,
                base_features_result=empty_result,
                interaction_features_result=empty_result,
                cross_timeframe_features_result=empty_result,
                selection_metrics={'error': str(e)}
            )
    
    def _select_features_for_type(
        self,
        features: pd.DataFrame,
        budget_config: FeatureTypeBudget,
        target: Optional[pd.Series],
        feature_importance_scores: Optional[Dict[str, float]],
        feature_type: str
    ) -> FeatureTypeSelectionResult:
        """Select features for a specific feature type with budget constraints."""
        
        start_time = time.time()
        tprint_debug(f"🔍 Selecting {feature_type} features with budget {budget_config.budget_ms}ms")
        
        if features.empty or features.shape[1] == 0:
            tprint_warning(f"⚠️ No {feature_type} features available for selection")
            return FeatureTypeSelectionResult(
                feature_type=feature_type,
                selected_features=[],
                rejected_features=[],
                selection_score=0.0,
                budget_utilization=0.0,
                execution_time=time.time() - start_time
            )
        
        try:
            # Calculate feature scores
            feature_scores = self._calculate_feature_scores(
                features, target, feature_importance_scores, feature_type
            )
            
            # Apply budget constraints
            selected_features, rejected_features = self._apply_budget_constraints(
                features.columns.tolist(),
                feature_scores,
                budget_config
            )
            
            # Calculate selection metrics
            selection_score = np.mean([feature_scores.get(f, 0.0) for f in selected_features]) if selected_features else 0.0
            budget_utilization = len(selected_features) * budget_config.cost_per_feature_ms / budget_config.budget_ms
            
            execution_time = time.time() - start_time
            
            tprint_success(f"✅ {feature_type} selection: {len(selected_features)} selected, {len(rejected_features)} rejected")
            
            return FeatureTypeSelectionResult(
                feature_type=feature_type,
                selected_features=selected_features,
                rejected_features=rejected_features,
                selection_score=selection_score,
                budget_utilization=min(1.0, budget_utilization),
                execution_time=execution_time,
                selection_metrics={
                    'n_candidates': len(features.columns),
                    'n_selected': len(selected_features),
                    'n_rejected': len(rejected_features),
                    'selection_rate': len(selected_features) / max(1, len(features.columns)),
                    'avg_score': selection_score,
                    'budget_utilization': budget_utilization
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ {feature_type} feature selection failed: {e}")
            self.logger.error(f"{feature_type} feature selection failed: {e}")
            
            return FeatureTypeSelectionResult(
                feature_type=feature_type,
                selected_features=[],
                rejected_features=features.columns.tolist(),
                selection_score=0.0,
                budget_utilization=0.0,
                execution_time=execution_time,
                selection_metrics={'error': str(e)}
            )
    
    def _calculate_feature_scores(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series],
        feature_importance_scores: Optional[Dict[str, float]],
        feature_type: str
    ) -> Dict[str, float]:
        """Calculate feature importance scores."""
        
        scores = {}
        
        # Use pre-computed scores if available
        if feature_importance_scores:
            for col in features.columns:
                if col in feature_importance_scores:
                    scores[col] = feature_importance_scores[col]
                else:
                    scores[col] = 0.0
            return scores
        
        # Calculate scores based on feature type and available data
        if target is not None and len(target) > 0:
            # Supervised scoring
            scores = self._calculate_supervised_scores(features, target)
        else:
            # Unsupervised scoring
            scores = self._calculate_unsupervised_scores(features)
        
        # Apply feature type specific adjustments
        if feature_type == 'interaction':
            # Interaction features might have different scoring
            scores = {k: v * 0.9 for k, v in scores.items()}  # Slight penalty
        elif feature_type == 'cross_timeframe':
            # Cross-timeframe features might have different scoring
            scores = {k: v * 0.8 for k, v in scores.items()}  # Slight penalty
        
        return scores
    
    def _calculate_supervised_scores(
        self, 
        features: pd.DataFrame, 
        target: pd.Series
    ) -> Dict[str, float]:
        """Calculate supervised feature importance scores."""
        
        scores = {}
        
        try:
            # Align features and target
            common_idx = features.index.intersection(target.index)
            if len(common_idx) == 0:
                tprint_warning("⚠️ No common indices between features and target")
                return {col: 0.0 for col in features.columns}
            
            X_aligned = features.loc[common_idx]
            y_aligned = target.loc[common_idx]
            
            # Calculate correlation with target
            for col in X_aligned.columns:
                try:
                    # Remove NaN values
                    valid_idx = ~(X_aligned[col].isna() | y_aligned.isna())
                    if valid_idx.sum() < 10:  # Need at least 10 valid samples
                        scores[col] = 0.0
                        continue
                    
                    x_clean = X_aligned[col][valid_idx]
                    y_clean = y_aligned[valid_idx]
                    
                    # Calculate correlation
                    correlation = abs(np.corrcoef(x_clean, y_clean)[0, 1])
                    if np.isnan(correlation):
                        correlation = 0.0
                    
                    scores[col] = correlation
                    
                except Exception as e:
                    tprint_debug(f"Failed to calculate correlation for {col}: {e}")
                    scores[col] = 0.0
            
            return scores
            
        except Exception as e:
            tprint_warning(f"Supervised scoring failed: {e}")
            return {col: 0.0 for col in features.columns}
    
    def _calculate_unsupervised_scores(
        self, 
        features: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate unsupervised feature importance scores."""
        
        scores = {}
        
        try:
            for col in features.columns:
                try:
                    # Calculate variance as a proxy for importance
                    variance = features[col].var()
                    if np.isnan(variance):
                        variance = 0.0
                    
                    # Normalize by feature range
                    feature_range = features[col].max() - features[col].min()
                    if feature_range > 0:
                        normalized_variance = variance / feature_range
                    else:
                        normalized_variance = variance
                    
                    scores[col] = normalized_variance
                    
                except Exception as e:
                    tprint_debug(f"Failed to calculate variance for {col}: {e}")
                    scores[col] = 0.0
            
            return scores
            
        except Exception as e:
            tprint_warning(f"Unsupervised scoring failed: {e}")
            return {col: 0.0 for col in features.columns}
    
    def _apply_budget_constraints(
        self,
        feature_names: List[str],
        feature_scores: Dict[str, float],
        budget_config: FeatureTypeBudget
    ) -> Tuple[List[str], List[str]]:
        """Apply budget constraints using mRMR/Spearman → Ensemble → Ensemble+RFE pipeline."""
        
        if not feature_names:
            return [], []
        
        tprint_debug(f"🔍 Applying budget constraints for {budget_config.feature_type} features")
        tprint_debug(f"   📊 Target: {budget_config.target_features}, Min: {budget_config.min_features}, Max: {budget_config.max_features}")
        
        try:
            # Step 1: mRMR/Spearman pre-selection
            preselected_features = self._mrmr_spearman_selection(
                feature_names, feature_scores, budget_config
            )
            
            # Step 2: Ensemble selection
            ensemble_selected = self._ensemble_selection(
                preselected_features, feature_scores, budget_config
            )
            
            # Step 3: Ensemble + RFE final selection
            final_selected = self._ensemble_rfe_selection(
                ensemble_selected, feature_scores, budget_config
            )
            
            # Ensure we meet target and constraints
            selected, rejected = self._enforce_target_constraints(
                final_selected, feature_names, budget_config
            )
            
            tprint_debug(f"✅ {budget_config.feature_type} selection: {len(selected)} selected, {len(rejected)} rejected")
            return selected, rejected
            
        except Exception as e:
            tprint_warning(f"⚠️ Budget constraint application failed for {budget_config.feature_type}: {e}")
            # Fallback to simple selection
            return self._simple_fallback_selection(feature_names, feature_scores, budget_config)
    
    def _mrmr_spearman_selection(
        self,
        feature_names: List[str],
        feature_scores: Dict[str, float],
        budget_config: FeatureTypeBudget
    ) -> List[str]:
        """Step 1: mRMR/Spearman pre-selection."""
        
        # Sort by Spearman correlation (feature scores)
        sorted_features = sorted(
            feature_names,
            key=lambda f: feature_scores.get(f, 0.0),
            reverse=True
        )
        
        # Select top features for mRMR (2x target to allow for further filtering)
        mrmr_candidates = sorted_features[:min(len(sorted_features), budget_config.target_features * 2)]
        
        tprint_debug(f"   🎯 mRMR pre-selection: {len(mrmr_candidates)} candidates from {len(feature_names)}")
        return mrmr_candidates
    
    def _ensemble_selection(
        self,
        candidate_features: List[str],
        feature_scores: Dict[str, float],
        budget_config: FeatureTypeBudget
    ) -> List[str]:
        """Step 2: Ensemble selection using multiple criteria."""
        
        if not candidate_features:
            return []
        
        # Calculate ensemble scores using multiple criteria
        ensemble_scores = {}
        
        for feature in candidate_features:
            base_score = feature_scores.get(feature, 0.0)
            
            # Apply feature type specific adjustments
            if budget_config.feature_type == 'interaction':
                # Interaction features get slight boost for diversity
                ensemble_scores[feature] = base_score * 1.1
            elif budget_config.feature_type == 'cross_timeframe':
                # Cross-timeframe features get boost for temporal relevance
                ensemble_scores[feature] = base_score * 1.05
            else:
                ensemble_scores[feature] = base_score
        
        # Sort by ensemble score
        sorted_features = sorted(
            candidate_features,
            key=lambda f: ensemble_scores.get(f, 0.0),
            reverse=True
        )
        
        # Select top features (1.5x target for RFE)
        ensemble_selected = sorted_features[:min(len(sorted_features), int(budget_config.target_features * 1.5))]
        
        tprint_debug(f"   🎯 Ensemble selection: {len(ensemble_selected)} from {len(candidate_features)}")
        return ensemble_selected
    
    def _ensemble_rfe_selection(
        self,
        candidate_features: List[str],
        feature_scores: Dict[str, float],
        budget_config: FeatureTypeBudget
    ) -> List[str]:
        """Step 3: Ensemble + RFE final selection."""
        
        if not candidate_features:
            return []
        
        # Apply RFE-style elimination based on budget constraints
        selected = []
        current_cost = 0.0
        
        # Sort by ensemble score
        sorted_features = sorted(
            candidate_features,
            key=lambda f: feature_scores.get(f, 0.0),
            reverse=True
        )
        
        for feature in sorted_features:
            feature_cost = budget_config.cost_per_feature_ms
            
            # Check budget and target constraints
            if (current_cost + feature_cost <= budget_config.budget_ms and 
                len(selected) < budget_config.target_features and
                feature_scores.get(feature, 0.0) >= self.config.min_importance_threshold):
                
                selected.append(feature)
                current_cost += feature_cost
                
                # Stop if we've reached target
                if len(selected) >= budget_config.target_features:
                    break
        
        tprint_debug(f"   🎯 Ensemble+RFE: {len(selected)} final features")
        return selected
    
    def _enforce_target_constraints(
        self,
        selected_features: List[str],
        all_features: List[str],
        budget_config: FeatureTypeBudget
    ) -> Tuple[List[str], List[str]]:
        """Enforce target constraints and ensure minimum features."""
        
        selected = selected_features.copy()
        rejected = [f for f in all_features if f not in selected]
        
        # Ensure minimum features
        if len(selected) < budget_config.min_features:
            # Add more features from rejected list
            remaining_features = [f for f in rejected if f not in selected]
            needed = budget_config.min_features - len(selected)
            
            for feature in remaining_features[:needed]:
                selected.append(feature)
                rejected.remove(feature)
        
        # Ensure we don't exceed maximum
        if len(selected) > budget_config.max_features:
            # Remove lowest scoring features
            selected = selected[:budget_config.max_features]
        
        return selected, rejected
    
    def _simple_fallback_selection(
        self,
        feature_names: List[str],
        feature_scores: Dict[str, float],
        budget_config: FeatureTypeBudget
    ) -> Tuple[List[str], List[str]]:
        """Simple fallback selection when advanced methods fail."""
        
        # Sort by score
        sorted_features = sorted(
            feature_names,
            key=lambda f: feature_scores.get(f, 0.0),
            reverse=True
        )
        
        # Select up to target features
        selected = sorted_features[:budget_config.target_features]
        rejected = [f for f in feature_names if f not in selected]
        
        return selected, rejected


def create_budget_aware_selector(
    base_features_budget: Optional[FeatureTypeBudget] = None,
    interaction_features_budget: Optional[FeatureTypeBudget] = None,
    cross_timeframe_features_budget: Optional[FeatureTypeBudget] = None,
    total_budget_ms: float = 100.0
) -> BudgetAwareFeatureSelector:
    """
    Create a budget-aware feature selector with custom budget configurations.
    
    Args:
        base_features_budget: Budget config for base features
        interaction_features_budget: Budget config for interaction features
        cross_timeframe_features_budget: Budget config for cross-timeframe features
        total_budget_ms: Total budget in milliseconds
        
    Returns:
        BudgetAwareFeatureSelector instance
    """
    
    config = BudgetAwareSelectionConfig(
        base_features=base_features_budget or FeatureTypeBudget(
            feature_type='base',
            min_features=40,
            max_features=80,
            target_features=60,  # Target 60 base features
            budget_ms=total_budget_ms * 0.6,  # 60% of total budget
            priority_weight=1.0,
            cost_per_feature_ms=0.5
        ),
        interaction_features=interaction_features_budget or FeatureTypeBudget(
            feature_type='interaction',
            min_features=5,
            max_features=15,
            target_features=10,  # Target 10 interaction features
            budget_ms=total_budget_ms * 0.25,  # 25% of total budget
            priority_weight=0.8,
            cost_per_feature_ms=1.0
        ),
        cross_timeframe_features=cross_timeframe_features_budget or FeatureTypeBudget(
            feature_type='cross_timeframe',
            min_features=3,
            max_features=10,
            target_features=6,  # Target 6 cross-timeframe features
            budget_ms=total_budget_ms * 0.15,  # 15% of total budget
            priority_weight=0.7,
            cost_per_feature_ms=1.2
        ),
        total_budget_ms=total_budget_ms
    )
    
    return BudgetAwareFeatureSelector(config)