"""
Budget-Aware Feature Selection System

This module implements a sophisticated budget-aware feature selection system that
optimizes feature selection based on computational budget constraints and trading
performance metrics. It provides a 3-stage pipeline with mRMR, ensemble selection,
and RFE methods.

Key Features:
- Budget allocation across feature types (base, interaction, cross-timeframe, gate)
- Trading performance optimization (CV performance, base importance, stability, sensitivity)
- No computational budget constraints - focus on trading performance
- Equal cost for all features (1.0)
- Comprehensive logging and error handling
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress
)

# Import logging utilities
from src.training.steps.pre_training.market_analysis.logging_standards import (
    get_logger, log_info, log_warning, log_error, log_success, log_debug
)

# Import validation utilities
from src.training.steps.pre_training.validation.schemas import (
    SchemaValidationException,
    enforce_feature_temporal_alignment,
    schema_metadata,
    validate_engineered_features,
)

# Import common operations
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    optimize_memory_usage, parallel_processing_optimizer
)

# Import matrix operations
from src.utils.matrix_operations import (
    get_unified_matrix_operations, get_vectorized_processing_core,
    get_batch_matrix_processor, safe_matrix_multiply,
    vectorized_rolling_features, parallel_feature_engineering,
    optimize_dataframe, get_hardware_performance_report
)

# Import ML common utilities
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer, OptimizationConfig
)
from src.utils.purged_kfold import PurgedKFoldTime as PurgedKFold
from src.feature_selection import select_features as FeatureSelector

# Setup logging
logger = get_logger(__name__)


@dataclass
class FeatureTypeBudget:
    """Configuration for each feature type budget allocation."""
    # Budget allocation (in milliseconds)
    budget_ms: float = 0.0
    
    # Feature count constraints
    min_features: int = 0
    max_features: int = 1000
    target_features: int = 50
    
    # Performance weights
    cv_performance_weight: float = 0.4  # Most important for trading
    base_importance_weight: float = 0.3  # Raw predictive power
    stability_weight: float = 0.2  # Consistency over time
    sensitivity_weight: float = 0.1  # Market response
    
    # Feature cost (equal for all features)
    feature_cost: float = 1.0
    
    # Selection criteria
    min_correlation_threshold: float = 0.1
    max_correlation_threshold: float = 0.95
    min_importance_score: float = 0.01
    min_stability_score: float = 0.5


@dataclass
class BudgetAwareSelectionConfig:
    """Main configuration for budget-aware feature selection."""
    # Total budget allocation
    total_budget_ms: float = 100.0
    
    # Feature type budgets
    base_features: FeatureTypeBudget = field(default_factory=lambda: FeatureTypeBudget(
        budget_ms=68.0,  # 68% of total budget
        min_features=40,
        max_features=80,
        target_features=60
    ))
    
    interaction_features: FeatureTypeBudget = field(default_factory=lambda: FeatureTypeBudget(
        budget_ms=15.0,  # 15% of total budget
        min_features=5,
        max_features=15,
        target_features=10
    ))
    
    cross_timeframe_features: FeatureTypeBudget = field(default_factory=lambda: FeatureTypeBudget(
        budget_ms=10.0,  # 10% of total budget
        min_features=3,
        max_features=10,
        target_features=6
    ))
    
    gate_features: FeatureTypeBudget = field(default_factory=lambda: FeatureTypeBudget(
        budget_ms=7.0,  # 7% of total budget
        min_features=2,
        max_features=8,
        target_features=5
    ))
    
    # Pipeline configuration
    enable_mrmr_selection: bool = True
    enable_ensemble_selection: bool = True
    enable_rfe_selection: bool = True
    
    # Performance optimization
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    enable_hardware_acceleration: bool = True
    
    # Validation settings
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Logging
    verbose: bool = True
    log_performance: bool = True


@dataclass
class FeatureTypeSelectionResult:
    """Results for individual feature type selection."""
    feature_type: str
    selected_features: List[str]
    feature_scores: Dict[str, float]
    selection_time: float
    budget_used_ms: float
    performance_metrics: Dict[str, float]
    success: bool
    error_message: Optional[str] = None


@dataclass
class BudgetAwareSelectionResult:
    """Overall budget-aware selection results."""
    # Core results
    all_selected_features: List[str]
    feature_type_results: Dict[str, FeatureTypeSelectionResult]
    
    # Performance metrics
    total_selection_time: float
    total_budget_used_ms: float
    overall_performance_score: float
    
    # Feature breakdown
    base_features: List[str]
    interaction_features: List[str]
    cross_timeframe_features: List[str]
    gate_features: List[str]
    
    # Success indicators
    success: bool
    error_message: Optional[str] = None
    
    # Additional metadata
    config_used: BudgetAwareSelectionConfig
    performance_breakdown: Dict[str, Any] = field(default_factory=dict)


class BudgetAwareFeatureSelector:
    """
    Budget-aware feature selector that optimizes feature selection based on
    computational budget constraints and trading performance metrics.
    """
    
    def __init__(self, config: Optional[BudgetAwareSelectionConfig] = None):
        """Initialize the budget-aware feature selector."""
        self.config = config or BudgetAwareSelectionConfig()
        self.logger = get_logger(f"{__name__}.BudgetAwareFeatureSelector")
        
        # Initialize hardware optimization tools
        self.memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=8.0)
        self.gpu_manager = get_m1_gpu_manager()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        
        # Initialize matrix operations
        self.matrix_ops = get_unified_matrix_operations()
        self.vectorized_core = get_vectorized_processing_core()
        self.batch_processor = get_batch_matrix_processor()
        
        # Initialize ML utilities
        self.bayesian_optimizer = BayesianTPEOptimizer(
            OptimizationConfig(
                n_trials=50,
                timeout_minutes=10,
                enable_parallel=True,
                max_workers=self.config.max_workers
            )
        )
        
        tprint_success("🚀 BudgetAwareFeatureSelector initialized")
        tprint_info(f"   📊 Total budget: {self.config.total_budget_ms}ms")
        tprint_info(f"   🎯 Base features: {self.config.base_features.target_features}")
        tprint_info(f"   🔗 Interaction features: {self.config.interaction_features.target_features}")
        tprint_info(f"   ⏰ Cross-timeframe features: {self.config.cross_timeframe_features.target_features}")
        tprint_info(f"   🚪 Gate features: {self.config.gate_features.target_features}")
    
    async def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_types: Optional[Dict[str, List[str]]] = None
    ) -> BudgetAwareSelectionResult:
        """
        Main entry point for budget-aware feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_types: Optional mapping of feature types to feature names
            
        Returns:
            BudgetAwareSelectionResult with selected features and performance metrics
        """
        start_time = time.time()
        tprint_success("🎯 Starting budget-aware feature selection")
        tprint_info(f"   📊 Input: {X.shape[0]} samples, {X.shape[1]} features")
        tprint_info(f"   🎯 Target: {len(y)} samples")
        
        try:
            # Validate inputs
            self._validate_inputs(X, y)
            
            # Categorize features by type if not provided
            if feature_types is None:
                feature_types = self._categorize_features_by_type(X.columns)
            
            # Apply budget constraints
            selection_results = await self._apply_budget_constraints(X, y, feature_types)
            
            # Combine results
            result = self._combine_selection_results(selection_results, start_time)
            
            tprint_success("✅ Budget-aware feature selection completed")
            tprint_info(f"   📊 Selected {len(result.all_selected_features)} features")
            tprint_info(f"   ⏱️ Total time: {result.total_selection_time:.3f}s")
            tprint_info(f"   💰 Budget used: {result.total_budget_used_ms:.1f}ms")
            
            return result
            
        except Exception as e:
            error_msg = f"Budget-aware feature selection failed: {e}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg, exc_info=True)
            
            return BudgetAwareSelectionResult(
                all_selected_features=[],
                feature_type_results={},
                total_selection_time=time.time() - start_time,
                total_budget_used_ms=0.0,
                overall_performance_score=0.0,
                base_features=[],
                interaction_features=[],
                cross_timeframe_features=[],
                gate_features=[],
                success=False,
                error_message=error_msg,
                config_used=self.config
            )
    
    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate input data."""
        if X.empty:
            raise ValueError("Feature matrix is empty")
        
        if y.empty:
            raise ValueError("Target variable is empty")
        
        if len(X) != len(y):
            raise ValueError(f"Feature matrix and target have different lengths: {len(X)} vs {len(y)}")
        
        # Check for missing values
        if X.isnull().any().any():
            tprint_warning("⚠️ Feature matrix contains missing values")
        
        if y.isnull().any():
            tprint_warning("⚠️ Target variable contains missing values")
    
    def _categorize_features_by_type(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features by type based on naming patterns."""
        feature_types = {
            'base': [],
            'interaction': [],
            'cross_timeframe': [],
            'gate': []
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            
            # Categorize based on naming patterns
            if any(pattern in feature_lower for pattern in ['_x_', '*', '_mul_', '_mult_']):
                feature_types['interaction'].append(feature)
            elif any(pattern in feature_lower for pattern in ['_ctf_', '_cross_', '_tf_']):
                feature_types['cross_timeframe'].append(feature)
            elif any(pattern in feature_lower for pattern in ['_gate_', '_gating_', '_switch_']):
                feature_types['gate'].append(feature)
            else:
                feature_types['base'].append(feature)
        
        tprint_info("📊 Feature categorization:")
        for ftype, features in feature_types.items():
            tprint_info(f"   {ftype}: {len(features)} features")
        
        return feature_types
    
    async def _apply_budget_constraints(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_types: Dict[str, List[str]]
    ) -> Dict[str, FeatureTypeSelectionResult]:
        """Apply budget constraints using the 3-stage pipeline."""
        tprint_info("🔄 Applying budget constraints with 3-stage pipeline")
        
        selection_results = {}
        
        # Process each feature type
        for ftype, features in feature_types.items():
            if not features:
                continue
            
            tprint_info(f"   🎯 Processing {ftype} features: {len(features)} candidates")
            
            # Get budget configuration for this feature type
            budget_config = getattr(self.config, f"{ftype}_features")
            
            # Select features for this type
            result = await self._select_features_for_type(
                X[features], y, ftype, budget_config
            )
            
            selection_results[ftype] = result
            
            if result.success:
                tprint_success(f"   ✅ {ftype}: {len(result.selected_features)} features selected")
            else:
                tprint_error(f"   ❌ {ftype}: {result.error_message}")
        
        return selection_results
    
    async def _select_features_for_type(
        self,
        X_type: pd.DataFrame,
        y: pd.Series,
        feature_type: str,
        budget_config: FeatureTypeBudget
    ) -> FeatureTypeSelectionResult:
        """Select features for a specific type using the 3-stage pipeline."""
        start_time = time.time()
        
        try:
            # Stage 1: mRMR with Spearman correlation (remove top 50% for diversity)
            if self.config.enable_mrmr_selection:
                tprint_debug(f"   🔍 Stage 1: mRMR selection for {feature_type}")
                X_stage1 = await self._mrmr_spearman_selection(X_type, y, budget_config)
            else:
                X_stage1 = X_type
            
            # Stage 2: Multi-step ensemble selection (LASSO + SHAP/LGBM + Random Forest)
            if self.config.enable_ensemble_selection:
                tprint_debug(f"   🔍 Stage 2: Ensemble selection for {feature_type}")
                X_stage2 = await self._multi_step_ensemble_selection(X_stage1, y, budget_config)
            else:
                X_stage2 = X_stage1
            
            # Stage 3: RFE with trading performance focus
            if self.config.enable_rfe_selection:
                tprint_debug(f"   🔍 Stage 3: RFE selection for {feature_type}")
                X_final = await self._rfe_final_selection(X_stage2, y, budget_config)
            else:
                X_final = X_stage2
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(X_final, y)
            
            selection_time = time.time() - start_time
            budget_used = selection_time * 1000  # Convert to milliseconds
            
            return FeatureTypeSelectionResult(
                feature_type=feature_type,
                selected_features=list(X_final.columns),
                feature_scores={},
                selection_time=selection_time,
                budget_used_ms=budget_used,
                performance_metrics=performance_metrics,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Feature selection failed for {feature_type}: {e}"
            tprint_error(f"   ❌ {error_msg}")
            
            return FeatureTypeSelectionResult(
                feature_type=feature_type,
                selected_features=[],
                feature_scores={},
                selection_time=time.time() - start_time,
                budget_used_ms=0.0,
                performance_metrics={},
                success=False,
                error_message=error_msg
            )
    
    async def _mrmr_spearman_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        budget_config: FeatureTypeBudget
    ) -> pd.DataFrame:
        """Stage 1: mRMR with Spearman correlation (remove top 50% for diversity)."""
        tprint_debug("   🔍 Running mRMR with Spearman correlation")
        
        # Calculate Spearman correlation with target
        correlations = X.corrwith(y, method='spearman').abs()
        
        # Remove top 50% most correlated features for diversity
        n_remove = len(correlations) // 2
        if n_remove > 0:
            top_correlated = correlations.nlargest(n_remove).index
            X_filtered = X.drop(columns=top_correlated)
            tprint_debug(f"   🗑️ Removed {n_remove} most correlated features for diversity")
        else:
            X_filtered = X
        
        return X_filtered
    
    async def _multi_step_ensemble_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        budget_config: FeatureTypeBudget
    ) -> pd.DataFrame:
        """Stage 2: Multi-step ensemble selection (LASSO + SHAP/LGBM + Random Forest)."""
        tprint_debug("   🔍 Running multi-step ensemble selection")
        
        # Step 1: LASSO selection
        lasso = LassoCV(cv=self.config.cv_folds, random_state=self.config.random_state)
        lasso.fit(X, y)
        
        # Get LASSO coefficients
        lasso_coefs = pd.Series(lasso.coef_, index=X.columns)
        lasso_features = lasso_coefs[lasso_coefs != 0].index.tolist()
        
        if not lasso_features:
            tprint_warning("   ⚠️ LASSO selected no features, using all features")
            return X
        
        X_lasso = X[lasso_features]
        tprint_debug(f"   📊 LASSO selected {len(lasso_features)} features")
        
        # Step 2: Random Forest importance
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        rf.fit(X_lasso, y)
        
        # Get feature importance
        rf_importance = pd.Series(rf.feature_importances_, index=X_lasso.columns)
        
        # Step 3: Cross-validation performance
        cv_scores = {}
        for feature in X_lasso.columns:
            try:
                # Single feature CV score
                single_feature = X_lasso[[feature]]
                scores = cross_val_score(
                    RandomForestRegressor(n_estimators=50, random_state=self.config.random_state),
                    single_feature, y, cv=self.config.cv_folds, scoring='r2'
                )
                cv_scores[feature] = scores.mean()
            except Exception:
                cv_scores[feature] = 0.0
        
        # Combine scores with weights
        combined_scores = {}
        for feature in X_lasso.columns:
            cv_score = cv_scores.get(feature, 0.0)
            importance_score = rf_importance.get(feature, 0.0)
            
            # Weighted combination
            combined_score = (
                budget_config.cv_performance_weight * cv_score +
                budget_config.base_importance_weight * importance_score
            )
            combined_scores[feature] = combined_score
        
        # Select top features based on target count
        target_count = min(budget_config.target_features, len(combined_scores))
        if target_count > 0:
            top_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:target_count]
            selected_features = [feature for feature, score in top_features]
            X_selected = X_lasso[selected_features]
            tprint_debug(f"   📊 Ensemble selected {len(selected_features)} features")
        else:
            X_selected = X_lasso
        
        return X_selected
    
    async def _rfe_final_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        budget_config: FeatureTypeBudget
    ) -> pd.DataFrame:
        """Stage 3: RFE with trading performance focus."""
        tprint_debug("   🔍 Running RFE with trading performance focus")
        
        # Use Random Forest as the estimator for RFE
        estimator = RandomForestRegressor(
            n_estimators=100,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        # Determine number of features to select
        target_count = min(budget_config.target_features, len(X.columns))
        if target_count <= 0:
            return X
        
        # Run RFE
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=target_count,
            step=1
        )
        
        try:
            rfe.fit(X, y)
            selected_features = X.columns[rfe.support_].tolist()
            X_selected = X[selected_features]
            tprint_debug(f"   📊 RFE selected {len(selected_features)} features")
        except Exception as e:
            tprint_warning(f"   ⚠️ RFE failed: {e}, using all features")
            X_selected = X
        
        return X_selected
    
    def _calculate_performance_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics for selected features."""
        try:
            # Cross-validation R² score
            rf = RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)
            cv_scores = cross_val_score(rf, X, y, cv=self.config.cv_folds, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Feature importance
            rf.fit(X, y)
            importance_scores = rf.feature_importances_
            avg_importance = np.mean(importance_scores)
            
            # Stability (variance of importance across CV folds)
            stability_scores = []
            for i in range(self.config.cv_folds):
                try:
                    fold_scores = cross_val_score(rf, X, y, cv=2, scoring='r2')
                    stability_scores.append(fold_scores.mean())
                except Exception:
                    stability_scores.append(0.0)
            
            stability = 1.0 - np.var(stability_scores) if stability_scores else 0.0
            
            # Sensitivity (response to small changes)
            sensitivity = self._calculate_sensitivity(X, y)
            
            return {
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'avg_importance': avg_importance,
                'stability': stability,
                'sensitivity': sensitivity,
                'n_features': len(X.columns)
            }
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Performance calculation failed: {e}")
            return {
                'cv_mean': 0.0,
                'cv_std': 0.0,
                'avg_importance': 0.0,
                'stability': 0.0,
                'sensitivity': 0.0,
                'n_features': len(X.columns)
            }
    
    def _calculate_sensitivity(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate feature sensitivity to small changes."""
        try:
            # Add small noise and measure performance change
            X_noisy = X + np.random.normal(0, 0.01, X.shape)
            
            # Original performance
            rf_orig = RandomForestRegressor(n_estimators=50, random_state=self.config.random_state)
            orig_score = cross_val_score(rf_orig, X, y, cv=3, scoring='r2').mean()
            
            # Noisy performance
            rf_noisy = RandomForestRegressor(n_estimators=50, random_state=self.config.random_state)
            noisy_score = cross_val_score(rf_noisy, X_noisy, y, cv=3, scoring='r2').mean()
            
            # Sensitivity is the absolute difference
            sensitivity = abs(orig_score - noisy_score)
            return sensitivity
            
        except Exception:
            return 0.0
    
    def _combine_selection_results(
        self,
        selection_results: Dict[str, FeatureTypeSelectionResult],
        start_time: float
    ) -> BudgetAwareSelectionResult:
        """Combine individual feature type results into overall result."""
        total_time = time.time() - start_time
        
        # Extract selected features by type
        base_features = selection_results.get('base', FeatureTypeSelectionResult('base', [], {}, 0, 0, {}, False)).selected_features
        interaction_features = selection_results.get('interaction', FeatureTypeSelectionResult('interaction', [], {}, 0, 0, {}, False)).selected_features
        cross_timeframe_features = selection_results.get('cross_timeframe', FeatureTypeSelectionResult('cross_timeframe', [], {}, 0, 0, {}, False)).selected_features
        gate_features = selection_results.get('gate', FeatureTypeSelectionResult('gate', [], {}, 0, 0, {}, False)).selected_features
        
        # Combine all selected features
        all_selected_features = (
            base_features + interaction_features + 
            cross_timeframe_features + gate_features
        )
        
        # Calculate total budget used
        total_budget_used = sum(
            result.budget_used_ms for result in selection_results.values()
        )
        
        # Calculate overall performance score
        overall_performance = self._calculate_overall_performance(selection_results)
        
        # Check overall success
        overall_success = all(result.success for result in selection_results.values())
        
        return BudgetAwareSelectionResult(
            all_selected_features=all_selected_features,
            feature_type_results=selection_results,
            total_selection_time=total_time,
            total_budget_used_ms=total_budget_used,
            overall_performance_score=overall_performance,
            base_features=base_features,
            interaction_features=interaction_features,
            cross_timeframe_features=cross_timeframe_features,
            gate_features=gate_features,
            success=overall_success,
            config_used=self.config
        )
    
    def _calculate_overall_performance(
        self,
        selection_results: Dict[str, FeatureTypeSelectionResult]
    ) -> float:
        """Calculate overall performance score from individual results."""
        if not selection_results:
            return 0.0
        
        # Weight by feature type importance
        weights = {
            'base': 0.4,
            'interaction': 0.3,
            'cross_timeframe': 0.2,
            'gate': 0.1
        }
        
        weighted_scores = []
        for ftype, result in selection_results.items():
            if result.success and result.performance_metrics:
                cv_score = result.performance_metrics.get('cv_mean', 0.0)
                weight = weights.get(ftype, 0.1)
                weighted_scores.append(cv_score * weight)
        
        return sum(weighted_scores) if weighted_scores else 0.0


def create_budget_aware_selector(
    config: Optional[BudgetAwareSelectionConfig] = None
) -> BudgetAwareFeatureSelector:
    """Create a budget-aware feature selector with the given configuration."""
    return BudgetAwareFeatureSelector(config)


# Convenience function for direct usage
async def select_features_budget_aware(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[BudgetAwareSelectionConfig] = None,
    feature_types: Optional[Dict[str, List[str]]] = None
) -> BudgetAwareSelectionResult:
    """
    Convenience function for budget-aware feature selection.
    
    Args:
        X: Feature matrix
        y: Target variable
        config: Optional configuration
        feature_types: Optional feature type mapping
        
    Returns:
        BudgetAwareSelectionResult with selected features
    """
    selector = create_budget_aware_selector(config)
    return await selector.select_features(X, y, feature_types)