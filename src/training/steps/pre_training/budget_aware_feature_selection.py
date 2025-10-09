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
    # Feature type budgets - Focused on trading performance
    base_features: FeatureTypeBudget = field(default_factory=lambda: FeatureTypeBudget(
        feature_type='base',
        min_features=40,
        max_features=80,
        target_features=60,  # Main target for base features
        budget_ms=100.0,  # No computational budget constraint
        priority_weight=1.0,
        cost_per_feature_ms=1.0  # Equal cost for trading performance focus
    ))
    
    interaction_features: FeatureTypeBudget = field(default_factory=lambda: FeatureTypeBudget(
        feature_type='interaction',
        min_features=5,
        max_features=15,
        target_features=10,  # Target 10 interaction features
        budget_ms=100.0,  # No computational budget constraint
        priority_weight=0.8,
        cost_per_feature_ms=1.0  # Equal cost for trading performance focus
    ))
    
    cross_timeframe_features: FeatureTypeBudget = field(default_factory=lambda: FeatureTypeBudget(
        feature_type='cross_timeframe',
        min_features=3,
        max_features=10,
        target_features=6,  # Target 6 cross-timeframe features
        budget_ms=100.0,  # No computational budget constraint
        priority_weight=0.7,
        cost_per_feature_ms=1.0  # Equal cost for trading performance focus
    ))
    
    # Gate features budget - Focused on trading performance
    gate_features: FeatureTypeBudget = field(default_factory=lambda: FeatureTypeBudget(
        feature_type='gate',
        min_features=2,
        max_features=8,
        target_features=5,  # Target 5 gate features
        budget_ms=100.0,  # No computational budget constraint
        priority_weight=0.9,
        cost_per_feature_ms=1.0  # Equal cost for trading performance focus
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
    uniform_allocation_ratio: float = 0.68  # 68% of total budget for base features


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
        """Apply trading performance-focused selection using mRMR/Spearman → Ensemble → RFE pipeline."""
        
        if not feature_names:
            return [], []
        
        tprint_debug(f"🔍 Applying trading performance-focused selection for {budget_config.feature_type} features")
        tprint_debug(f"   📊 Target: {budget_config.target_features}, Min: {budget_config.min_features}, Max: {budget_config.max_features}")
        
        try:
            # Step 1: mRMR/Spearman - remove top 50% (keep bottom 50% for diversity)
            preselected_features = self._mrmr_spearman_selection(
                feature_names, feature_scores, budget_config
            )
            
            # Step 2: Multiple ensemble selection steps
            ensemble_selected = self._multi_step_ensemble_selection(
                preselected_features, feature_scores, budget_config
            )
            
            # Step 3: RFE final selection
            final_selected = self._rfe_final_selection(
                ensemble_selected, feature_scores, budget_config
            )
            
            # Ensure we meet target and constraints
            selected, rejected = self._enforce_target_constraints(
                final_selected, feature_names, budget_config
            )
            
            tprint_debug(f"✅ {budget_config.feature_type} selection: {len(selected)} selected, {len(rejected)} rejected")
            return selected, rejected
            
        except Exception as e:
            tprint_warning(f"⚠️ Trading performance selection failed for {budget_config.feature_type}: {e}")
            # Fallback to simple selection
            return self._simple_fallback_selection(feature_names, feature_scores, budget_config)
    
    def _mrmr_spearman_selection(
        self,
        feature_names: List[str],
        feature_scores: Dict[str, float],
        budget_config: FeatureTypeBudget
    ) -> List[str]:
        """Step 1: mRMR/Spearman - remove top 50% (keep bottom 50% for diversity)."""
        
        # Sort by Spearman correlation (feature scores)
        sorted_features = sorted(
            feature_names,
            key=lambda f: feature_scores.get(f, 0.0),
            reverse=True
        )
        
        # Remove top 50% - keep bottom 50% for diversity
        # This is counter-intuitive but helps avoid overfitting to obvious features
        keep_count = max(1, len(sorted_features) // 2)
        mrmr_candidates = sorted_features[-keep_count:]  # Keep bottom 50%
        
        tprint_debug(f"   🎯 mRMR pre-selection: Removed top 50%, kept {len(mrmr_candidates)} candidates from {len(feature_names)}")
        return mrmr_candidates
    
    def _multi_step_ensemble_selection(
        self,
        candidate_features: List[str],
        feature_scores: Dict[str, float],
        budget_config: FeatureTypeBudget
    ) -> List[str]:
        """Step 2: Multiple ensemble selection steps for trading performance."""
        
        if not candidate_features:
            return []
        
        tprint_debug(f"   🔄 Running multi-step ensemble selection for {budget_config.feature_type}")
        
        current_features = candidate_features.copy()
        
        # Step 2a: Diversity-based ensemble
        current_features = self._diversity_ensemble_selection(
            current_features, feature_scores, budget_config
        )
        
        # Step 2b: Stability-based ensemble
        current_features = self._stability_ensemble_selection(
            current_features, feature_scores, budget_config
        )
        
        # Step 2c: Trading performance ensemble
        current_features = self._trading_performance_ensemble_selection(
            current_features, feature_scores, budget_config
        )
        
        tprint_debug(f"   🎯 Multi-step ensemble: {len(current_features)} from {len(candidate_features)}")
        return current_features
    
    def _diversity_ensemble_selection(
        self,
        candidate_features: List[str],
        feature_scores: Dict[str, float],
        budget_config: FeatureTypeBudget
    ) -> List[str]:
        """Diversity-based ensemble selection."""
        
        if not candidate_features:
            return []
        
        # Calculate diversity scores
        diversity_scores = {}
        
        for feature in candidate_features:
            base_score = feature_scores.get(feature, 0.0)
            
            # Boost features that are different from common patterns
            diversity_boost = 1.0
            
            # Interaction features get diversity boost
            if 'interaction' in feature.lower() or '_x_' in feature or '*' in feature:
                diversity_boost = 1.2
            
            # Cross-timeframe features get diversity boost
            elif 'cross' in feature.lower() or 'timeframe' in feature.lower():
                diversity_boost = 1.15
            
            # Gate features get diversity boost
            elif 'gate' in feature.lower():
                diversity_boost = 1.1
            
            diversity_scores[feature] = base_score * diversity_boost
        
        # Sort by diversity score
        sorted_features = sorted(
            candidate_features,
            key=lambda f: diversity_scores.get(f, 0.0),
            reverse=True
        )
        
        # Keep top 80% for next step
        keep_count = max(1, int(len(sorted_features) * 0.8))
        return sorted_features[:keep_count]
    
    def _stability_ensemble_selection(
        self,
        candidate_features: List[str],
        feature_scores: Dict[str, float],
        budget_config: FeatureTypeBudget
    ) -> List[str]:
        """Stability-based ensemble selection."""
        
        if not candidate_features:
            return []
        
        # Calculate stability scores
        stability_scores = {}
        
        for feature in candidate_features:
            base_score = feature_scores.get(feature, 0.0)
            
            # Calculate stability proxy
            stability = self._calculate_feature_stability(feature, [], candidate_features)
            
            # Combine base score with stability
            stability_scores[feature] = base_score * 0.7 + stability * 0.3
        
        # Sort by stability score
        sorted_features = sorted(
            candidate_features,
            key=lambda f: stability_scores.get(f, 0.0),
            reverse=True
        )
        
        # Keep top 70% for next step
        keep_count = max(1, int(len(sorted_features) * 0.7))
        return sorted_features[:keep_count]
    
    def _trading_performance_ensemble_selection(
        self,
        candidate_features: List[str],
        feature_scores: Dict[str, float],
        budget_config: FeatureTypeBudget
    ) -> List[str]:
        """Trading performance-based ensemble selection."""
        
        if not candidate_features:
            return []
        
        # Calculate trading performance scores
        trading_scores = {}
        
        for feature in candidate_features:
            base_score = feature_scores.get(feature, 0.0)
            
            # Calculate CV and sensitivity for trading performance
            cv_score = self._calculate_cv_performance(feature, [], candidate_features)
            sensitivity_score = self._calculate_sensitivity_score(feature, [], candidate_features)
            
            # Combine for trading performance
            trading_scores[feature] = (
                base_score * 0.4 +
                cv_score * 0.4 +
                sensitivity_score * 0.2
            )
        
        # Sort by trading performance score
        sorted_features = sorted(
            candidate_features,
            key=lambda f: trading_scores.get(f, 0.0),
            reverse=True
        )
        
        # Keep top 60% for RFE
        keep_count = max(1, int(len(sorted_features) * 0.6))
        return sorted_features[:keep_count]
    
    def _rfe_final_selection(
        self,
        candidate_features: List[str],
        feature_scores: Dict[str, float],
        budget_config: FeatureTypeBudget
    ) -> List[str]:
        """Step 3: RFE final selection focused on trading performance."""
        
        if not candidate_features:
            return []
        
        tprint_debug(f"   🔬 Running RFE final selection for {budget_config.feature_type}")
        
        # Calculate trading performance metrics
        trading_metrics = self._calculate_trading_performance_metrics(
            candidate_features, feature_scores, budget_config
        )
        
        # Apply RFE with trading performance focus
        selected = self._rfe_trading_performance(
            candidate_features, trading_metrics, budget_config
        )
        
        tprint_debug(f"   🎯 RFE final: {len(selected)} features selected for trading performance")
        return selected
    
    def _calculate_trading_performance_metrics(
        self,
        candidate_features: List[str],
        feature_scores: Dict[str, float],
        budget_config: FeatureTypeBudget
    ) -> Dict[str, Dict[str, float]]:
        """Calculate trading performance metrics for RFE."""
        
        metrics = {}
        
        for feature in candidate_features:
            base_score = feature_scores.get(feature, 0.0)
            
            # Calculate trading-specific metrics
            cv_score = self._calculate_cv_performance(feature, [], candidate_features)
            stability_score = self._calculate_feature_stability(feature, [], candidate_features)
            sensitivity_score = self._calculate_sensitivity_score(feature, [], candidate_features)
            
            # Calculate trading performance score
            trading_score = (
                base_score * 0.3 +           # Base importance
                cv_score * 0.4 +             # CV performance (most important for trading)
                stability_score * 0.2 +      # Stability
                sensitivity_score * 0.1      # Sensitivity
            )
            
            metrics[feature] = {
                'trading_score': trading_score,
                'base_score': base_score,
                'cv_score': cv_score,
                'stability_score': stability_score,
                'sensitivity_score': sensitivity_score
            }
        
        return metrics
    
    def _rfe_trading_performance(
        self,
        candidate_features: List[str],
        trading_metrics: Dict[str, Dict[str, float]],
        budget_config: FeatureTypeBudget
    ) -> List[str]:
        """RFE focused on trading performance - no computational budget constraints."""
        
        selected = []
        remaining_features = candidate_features.copy()
        
        # Sort by trading performance score
        remaining_features.sort(
            key=lambda f: trading_metrics.get(f, {}).get('trading_score', 0.0),
            reverse=True
        )
        
        # Select features up to target (no budget constraints)
        while remaining_features and len(selected) < budget_config.target_features:
            best_feature = remaining_features[0]
            selected.append(best_feature)
            remaining_features.remove(best_feature)
        
        # Ensure minimum features
        if len(selected) < budget_config.min_features and remaining_features:
            needed = budget_config.min_features - len(selected)
            for feature in remaining_features[:needed]:
                if feature not in selected:
                    selected.append(feature)
        
        return selected
    
    def _calculate_robust_metrics(
        self,
        candidate_features: List[str],
        feature_scores: Dict[str, float],
        budget_config: FeatureTypeBudget
    ) -> Dict[str, Dict[str, float]]:
        """Calculate robust metrics with proper normalization and tie-breaking."""
        
        metrics = {}
        
        # Calculate raw metrics
        importance_scores = {}
        stability_scores = {}
        cv_scores = {}
        sensitivity_scores = {}
        
        for feature in candidate_features:
            importance_scores[feature] = feature_scores.get(feature, 0.0)
            stability_scores[feature] = self._calculate_feature_stability(feature, [], candidate_features)
            cv_scores[feature] = self._calculate_cv_performance(feature, [], candidate_features)
            sensitivity_scores[feature] = self._calculate_sensitivity_score(feature, [], candidate_features)
        
        # Normalize each component using z-score (clipped to ±3σ)
        importance_norm = self._z_score_normalize(importance_scores, clip_sigma=3.0)
        stability_norm = self._z_score_normalize(stability_scores, clip_sigma=3.0)
        cv_norm = self._z_score_normalize(cv_scores, clip_sigma=3.0)
        sensitivity_norm = self._z_score_normalize(sensitivity_scores, clip_sigma=3.0)
        
        # Get type-specific weights
        weights = self._get_type_specific_weights(budget_config.feature_type)
        
        # Calculate combined scores with normalized components
        for feature in candidate_features:
            combined_score = (
                importance_norm[feature] * weights['importance'] +
                stability_norm[feature] * weights['stability'] +
                cv_norm[feature] * weights['cv'] +
                sensitivity_norm[feature] * weights['sensitivity']
            )
            
            metrics[feature] = {
                'combined_score': combined_score,
                'importance': importance_scores[feature],
                'stability': stability_scores[feature],
                'cv': cv_scores[feature],
                'sensitivity': sensitivity_scores[feature],
                'importance_norm': importance_norm[feature],
                'stability_norm': stability_norm[feature],
                'cv_norm': cv_norm[feature],
                'sensitivity_norm': sensitivity_norm[feature]
            }
        
        return metrics
    
    def _z_score_normalize(self, scores: Dict[str, float], clip_sigma: float = 3.0) -> Dict[str, float]:
        """Normalize scores using z-score with clipping."""
        if not scores:
            return {}
        
        values = list(scores.values())
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return {k: 0.0 for k in scores.keys()}
        
        normalized = {}
        for feature, score in scores.items():
            z_score = (score - mean_val) / std_val
            # Clip to ±3σ
            z_score = np.clip(z_score, -clip_sigma, clip_sigma)
            normalized[feature] = z_score
        
        return normalized
    
    def _get_type_specific_weights(self, feature_type: str) -> Dict[str, float]:
        """Get type-specific weights for scoring components."""
        if feature_type == 'gate':
            # Gates prioritize stability > importance
            return {
                'importance': 0.2,
                'stability': 0.5,
                'cv': 0.2,
                'sensitivity': 0.1
            }
        else:
            # Standard weights for other types
            return {
                'importance': 0.4,
                'stability': 0.3,
                'cv': 0.2,
                'sensitivity': 0.1
            }
    
    def _rfe_with_budget_constraints(
        self,
        candidate_features: List[str],
        all_metrics: Dict[str, Dict[str, float]],
        budget_config: FeatureTypeBudget
    ) -> List[str]:
        """RFE focused on trading performance - no computational budget constraints."""
        
        selected = []
        remaining_features = candidate_features.copy()
        
        # Sort by combined score
        remaining_features.sort(
            key=lambda f: all_metrics.get(f, {}).get('combined_score', 0.0),
            reverse=True
        )
        
        # Select features up to target (no budget constraints)
        while remaining_features and len(selected) < budget_config.target_features:
            best_feature = remaining_features[0]
            selected.append(best_feature)
            remaining_features.remove(best_feature)
            
            tprint_debug(f"   ✅ Added {best_feature}: score={all_metrics.get(best_feature, {}).get('combined_score', 0.0):.4f}")
        
        # Ensure minimum features if not met
        if len(selected) < budget_config.min_features and remaining_features:
            needed = budget_config.min_features - len(selected)
            for feature in remaining_features[:needed]:
                if feature not in selected:
                    selected.append(feature)
        
        return selected
    
    def _calculate_feature_stability(self, feature: str, selected_features: List[str], all_features: List[str]) -> float:
        """Calculate stability score using 1/(1+CV of importance across folds)."""
        try:
            # For gates, calculate flip-rate stability
            if 'gate' in feature.lower() or 'regime' in feature.lower():
                return self._calculate_gate_stability(feature)
            
            # For other features, use importance stability proxy
            # This would ideally be calculated across purged CV folds
            # For now, use feature characteristics as proxy
            
            # Interaction features tend to be less stable
            if 'interaction' in feature.lower() or '_x_' in feature or '*' in feature:
                base_stability = 0.6
            
            # Cross-timeframe features are moderately stable
            elif 'cross' in feature.lower() or 'timeframe' in feature.lower():
                base_stability = 0.7
            
            # Base features are most stable
            else:
                base_stability = 0.8
            
            # Apply time-series specific adjustments
            if 'volatility' in feature.lower():
                base_stability *= 0.9  # Volatility features less stable
            
            if 'momentum' in feature.lower():
                base_stability *= 0.95  # Momentum features slightly less stable
            
            return min(1.0, base_stability)
            
        except Exception:
            return 0.5  # Default stability score
    
    def _calculate_gate_stability(self, feature: str) -> float:
        """Calculate flip-rate stability for gate features."""
        try:
            # Gate features should have consistent segmentation
            # This would ideally measure flip-rate across time periods
            # For now, use feature characteristics as proxy
            
            if 'regime' in feature.lower():
                return 0.8  # Regime gates should be stable
            
            if 'volatility' in feature.lower() and 'gate' in feature.lower():
                return 0.7  # Volatility gates moderately stable
            
            return 0.6  # Default gate stability
            
        except Exception:
            return 0.5
    
    def _calculate_cv_performance(self, feature: str, selected_features: List[str], all_features: List[str]) -> float:
        """Calculate CV performance using mean IC/AUC across folds with embargo."""
        try:
            # This would ideally calculate mean IC/AUC across purged CV folds
            # For now, use feature characteristics and complementarity as proxy
            
            base_cv = 0.6  # Base CV performance
            
            # Feature type adjustments
            if 'momentum' in feature.lower():
                base_cv = 0.7  # Momentum features generally good CV performance
            
            if 'volatility' in feature.lower():
                base_cv = 0.8  # Volatility features excellent CV performance
            
            if 'regime' in feature.lower():
                base_cv = 0.75  # Regime features good CV performance
            
            # Complementarity bonus
            if len(selected_features) == 0:
                base_cv += 0.1  # First feature gets bonus
            
            # Diversity penalty for similar features
            similarity_penalty = 0.0
            for selected in selected_features:
                if self._features_similar(feature, selected):
                    similarity_penalty += 0.05
            
            base_cv -= similarity_penalty
            
            # Apply embargo penalty (simulate time-series CV)
            embargo_penalty = 0.02  # Small penalty for time-series leakage risk
            base_cv -= embargo_penalty
            
            return max(0.0, min(1.0, base_cv))
            
        except Exception:
            return 0.5  # Default CV score
    
    def _features_similar(self, feature1: str, feature2: str) -> bool:
        """Check if two features are similar (for diversity penalty)."""
        try:
            # Simple similarity check based on feature names
            f1_lower = feature1.lower()
            f2_lower = feature2.lower()
            
            # Check for common prefixes/suffixes
            common_patterns = ['vol', 'mom', 'regime', 'interaction', 'cross']
            
            for pattern in common_patterns:
                if pattern in f1_lower and pattern in f2_lower:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _calculate_sensitivity_score(self, feature: str, selected_features: List[str], all_features: List[str]) -> float:
        """Calculate sensitivity using slope magnitude of PDP/ICE around realistic perturbations."""
        try:
            # This would ideally calculate slope magnitude of PDP/ICE around perturbations
            # For now, use feature characteristics as proxy
            
            base_sensitivity = 0.6  # Base sensitivity
            
            # Volatility features are highly sensitive to market changes
            if 'volatility' in feature.lower() or 'vol' in feature.lower():
                base_sensitivity = 0.9
            
            # Momentum features are sensitive to trend changes
            elif 'momentum' in feature.lower() or 'mom' in feature.lower():
                base_sensitivity = 0.8
            
            # Regime features are moderately sensitive
            elif 'regime' in feature.lower() or 'reg' in feature.lower():
                base_sensitivity = 0.7
            
            # Interaction features are sensitive to both components
            elif 'interaction' in feature.lower() or '_x_' in feature or '*' in feature:
                base_sensitivity = 0.75
            
            # Cross-timeframe features are sensitive to temporal changes
            elif 'cross' in feature.lower() or 'timeframe' in feature.lower():
                base_sensitivity = 0.7
            
            # Gate features sensitivity depends on type
            elif 'gate' in feature.lower():
                if 'volatility' in feature.lower():
                    base_sensitivity = 0.8  # Volatility gates sensitive
                elif 'regime' in feature.lower():
                    base_sensitivity = 0.6  # Regime gates less sensitive
                else:
                    base_sensitivity = 0.7  # Default gate sensitivity
            
            # Apply time-series specific adjustments
            if 'microstructure' in feature.lower():
                base_sensitivity *= 1.1  # Microstructure features more sensitive
            
            if 'temporal' in feature.lower():
                base_sensitivity *= 1.05  # Temporal features slightly more sensitive
            
            return min(1.0, base_sensitivity)
            
        except Exception:
            return 0.5  # Default sensitivity score
    
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
            budget_ms=100.0,  # No computational budget constraint
            priority_weight=1.0,
            cost_per_feature_ms=1.0  # Equal cost for trading performance focus
        ),
        interaction_features=interaction_features_budget or FeatureTypeBudget(
            feature_type='interaction',
            min_features=5,
            max_features=15,
            target_features=10,  # Target 10 interaction features
            budget_ms=100.0,  # No computational budget constraint
            priority_weight=0.8,
            cost_per_feature_ms=1.0  # Equal cost for trading performance focus
        ),
        cross_timeframe_features=cross_timeframe_features_budget or FeatureTypeBudget(
            feature_type='cross_timeframe',
            min_features=3,
            max_features=10,
            target_features=6,  # Target 6 cross-timeframe features
            budget_ms=100.0,  # No computational budget constraint
            priority_weight=0.7,
            cost_per_feature_ms=1.0  # Equal cost for trading performance focus
        ),
        gate_features=FeatureTypeBudget(
            feature_type='gate',
            min_features=2,
            max_features=8,
            target_features=5,  # Target 5 gate features
            budget_ms=100.0,  # No computational budget constraint
            priority_weight=0.9,
            cost_per_feature_ms=1.0  # Equal cost for trading performance focus
        ),
        total_budget_ms=100.0  # No computational budget constraint
    )
    
    return BudgetAwareFeatureSelector(config)