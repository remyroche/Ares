"""
Advanced Feature Selection Methods

This module implements advanced feature selection using LASSO, RandomForest,
and LightGBM with permutation importance and comprehensive validation.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# Import required libraries
try:
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig

logger = logging.getLogger(__name__)

@dataclass
class AdvancedSelectionConfig:
    """Configuration for advanced feature selection methods."""
    # General settings
    random_state: int = 42
    n_jobs: int = -1
    enable_hardware_optimization: bool = True

    # LASSO settings
    lasso_cv_folds: int = 5
    lasso_alphas: Tuple[float, float] = (0.001, 1.0)
    lasso_n_alphas: int = 100
    lasso_max_iter: int = 2000

    # RandomForest settings
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_max_features: str = 'sqrt'

    # LightGBM settings
    lgb_n_estimators: int = 100
    lgb_max_depth: int = 6
    lgb_learning_rate: float = 0.1
    lgb_num_leaves: int = 31
    lgb_feature_fraction: float = 0.9
    lgb_bagging_fraction: float = 0.8

    # Validation settings
    cv_folds: int = 5
    test_size: float = 0.2
    enable_time_series_cv: bool = True

    # Performance settings
    enable_permutation_importance: bool = True
    permutation_n_repeats: int = 10
    enable_feature_interaction: bool = True

class BaseAdvancedSelector(ABC):
    """Base class for advanced feature selectors."""

    def __init__(self, config: Optional[AdvancedSelectionConfig] = None):
        """Initialize base selector."""
        self.config = config or AdvancedSelectionConfig()
        self.logger = logger.getChild(self.__class__.__name__)

        # Initialize hardware optimization
        if self.config.enable_hardware_optimization:
            self.cpu_optimizer = M1CPUOptimizer()
            hw_config = HardwareConfig(
                cpu_optimization_level='aggressive',
                enable_adaptive_optimization=True
            )
            self.hardware_manager = UnifiedHardwareManager(hw_config)
        else:
            self.cpu_optimizer = None
            self.hardware_manager = None

        # State
        self.fitted = False
        self.feature_importance_ = None
        self.selected_features_ = None
        self.feature_names_ = None
        self.model_ = None

    @abstractmethod
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Fit the underlying model."""
        pass

    @abstractmethod
    def _get_feature_importance(self) -> np.ndarray:
        """Get feature importance from the fitted model."""
        pass

    def _calculate_permutation_importance(self, X: np.ndarray, y: np.ndarray,
                                        scoring: str = 'neg_mean_squared_error') -> np.ndarray:
        """Calculate permutation importance."""
        if not self.config.enable_permutation_importance:
            return self._get_feature_importance()

        try:
            from sklearn.inspection import permutation_importance

            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.model_, X, y,
                n_repeats=self.config.permutation_n_repeats,
                random_state=self.config.random_state,
                scoring=scoring,
                n_jobs=self.config.n_jobs
            )

            return perm_importance.importances_mean

        except Exception as e:
            self.logger.warning(f"Permutation importance failed: {e}")
            return self._get_feature_importance()

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> 'BaseAdvancedSelector':
        """Fit the feature selector."""
        tprint_info(f"🔧 Fitting {self.__class__.__name__}")

        start_time = time.time()

        try:
            # Store feature names
            self.feature_names_ = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

            # Fit model
            self.model_ = self._fit_model(X, y)

            # Get feature importance
            self.feature_importance_ = self._get_feature_importance()

            # Calculate permutation importance if enabled
            if self.config.enable_permutation_importance:
                perm_importance = self._calculate_permutation_importance(X, y)
                # Combine with regular importance
                self.feature_importance_ = 0.7 * self.feature_importance_ + 0.3 * perm_importance

            self.fitted = True

            end_time = time.time()
            tprint_success(f"✅ {self.__class__.__name__} fitted in {end_time - start_time:.3f}s")

            return self

        except Exception as e:
            self.logger.error(f"Fitting failed: {e}")
            raise

    def select_features(self, X: np.ndarray, y: np.ndarray,
                       n_features: Optional[int] = None,
                       threshold: Optional[float] = None) -> Dict[str, Any]:
        """Select features based on importance."""
        if not self.fitted:
            self.fit(X, y)

        if n_features is None and threshold is None:
            # Select top 50% of features
            n_features = max(1, X.shape[1] // 2)

        # Get feature importance
        importance = self.feature_importance_

        if threshold is not None:
            # Select features above threshold
            selected_mask = importance >= threshold
        else:
            # Select top n_features
            n_select = min(n_features, X.shape[1])
            selected_indices = np.argsort(importance)[-n_select:]
            selected_mask = np.zeros(X.shape[1], dtype=bool)
            selected_mask[selected_indices] = True

        # Get selected features
        selected_indices = np.where(selected_mask)[0]
        selected_features = [self.feature_names_[i] for i in selected_indices]

        # Create feature scores
        feature_scores = {
            self.feature_names_[i]: float(importance[i])
            for i in selected_indices
        }

        return {
            'success': True,
            'selected_features': selected_features,
            'selected_indices': selected_indices.tolist(),
            'feature_scores': feature_scores,
            'n_selected': len(selected_features),
            'n_total': X.shape[1],
            'method': self.__class__.__name__.lower(),
            'importance_scores': importance.tolist()
        }

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.fitted:
            raise ValueError("Selector must be fitted first")
        return self.feature_importance_

    def get_selected_features(self) -> List[str]:
        """Get names of selected features."""
        if not self.fitted:
            raise ValueError("Selector must be fitted first")
        return self.selected_features_ or []

class LASSOFeatureSelector(BaseAdvancedSelector):
    """LASSO-based feature selection with cross-validation."""

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> LassoCV:
        """Fit LASSO model with cross-validation."""
        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Create LASSO CV model
        lasso_cv = LassoCV(
            alphas=np.logspace(
                np.log10(self.config.lasso_alphas[0]),
                np.log10(self.config.lasso_alphas[1]),
                self.config.lasso_n_alphas
            ),
            cv=self.config.lasso_cv_folds,
            max_iter=self.config.lasso_max_iter,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )

        # Fit model
        lasso_cv.fit(X_scaled, y)

        tprint_debug(f"🔧 LASSO optimal alpha: {lasso_cv.alpha_:.6f}")

        return lasso_cv

    def _get_feature_importance(self) -> np.ndarray:
        """Get LASSO coefficients as feature importance."""
        return np.abs(self.model_.coef_)

class RandomForestFeatureSelector(BaseAdvancedSelector):
    """RandomForest-based feature selection."""

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> Union[RandomForestRegressor, RandomForestClassifier]:
        """Fit RandomForest model."""
        # Determine if classification or regression
        is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))

        if is_classification:
            model = RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                min_samples_leaf=self.config.rf_min_samples_leaf,
                max_features=self.config.rf_max_features,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
        else:
            model = RandomForestRegressor(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                min_samples_leaf=self.config.rf_min_samples_leaf,
                max_features=self.config.rf_max_features,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )

        # Fit model
        model.fit(X, y)

        return model

    def _get_feature_importance(self) -> np.ndarray:
        """Get RandomForest feature importance."""
        return self.model_.feature_importances_

class LightGBMFeatureSelector(BaseAdvancedSelector):
    """LightGBM-based feature selection."""

    def __init__(self, config: Optional[AdvancedSelectionConfig] = None):
        """Initialize LightGBM selector."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for LightGBMFeatureSelector")
        super().__init__(config)

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> lgb.LGBMRegressor:
        """Fit LightGBM model."""
        # Determine if classification or regression
        is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))

        if is_classification:
            model = lgb.LGBMClassifier(
                n_estimators=self.config.lgb_n_estimators,
                max_depth=self.config.lgb_max_depth,
                learning_rate=self.config.lgb_learning_rate,
                num_leaves=self.config.lgb_num_leaves,
                feature_fraction=self.config.lgb_feature_fraction,
                bagging_fraction=self.config.lgb_bagging_fraction,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbose=-1
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=self.config.lgb_n_estimators,
                max_depth=self.config.lgb_max_depth,
                learning_rate=self.config.lgb_learning_rate,
                num_leaves=self.config.lgb_num_leaves,
                feature_fraction=self.config.lgb_feature_fraction,
                bagging_fraction=self.config.lgb_bagging_fraction,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbose=-1
            )

        # Fit model
        model.fit(X, y)

        return model

    def _get_feature_importance(self) -> np.ndarray:
        """Get LightGBM feature importance."""
        return self.model_.feature_importances_

class EnsembleAdvancedSelector:
    """Ensemble selector combining multiple advanced methods."""

    def __init__(self, config: Optional[AdvancedSelectionConfig] = None):
        """Initialize ensemble selector."""
        self.config = config or AdvancedSelectionConfig()
        self.logger = logger.getChild('EnsembleAdvancedSelector')

        # Initialize individual selectors
        self.selectors = {
            'lasso': LASSOFeatureSelector(self.config),
            'random_forest': RandomForestFeatureSelector(self.config),
            'lightgbm': LightGBMFeatureSelector(self.config) if LIGHTGBM_AVAILABLE else None
        }

        # Remove None selectors
        self.selectors = {k: v for k, v in self.selectors.items() if v is not None}

        tprint_success(f"🔧 EnsembleAdvancedSelector initialized with {len(self.selectors)} methods")

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> 'EnsembleAdvancedSelector':
        """Fit all selectors."""
        tprint_info("🔧 Fitting ensemble selectors")

        start_time = time.time()

        for name, selector in self.selectors.items():
            tprint_debug(f"🔧 Fitting {name}")
            selector.fit(X, y, feature_names)

        end_time = time.time()
        tprint_success(f"✅ Ensemble fitted in {end_time - start_time:.3f}s")

        return self

    def select_features(self, X: np.ndarray, y: np.ndarray,
                       n_features: Optional[int] = None,
                       method: str = 'ensemble',
                       weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Select features using ensemble method."""
        if not all(selector.fitted for selector in self.selectors.values()):
            self.fit(X, y)

        if method == 'ensemble':
            return self._ensemble_selection(X, y, n_features, weights)
        else:
            # Use specific method
            if method in self.selectors:
                return self.selectors[method].select_features(X, y, n_features)
            else:
                raise ValueError(f"Unknown method: {method}")

    def _ensemble_selection(self, X: np.ndarray, y: np.ndarray,
                           n_features: Optional[int] = None,
                           weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Perform ensemble feature selection."""
        if weights is None:
            weights = {name: 1.0 for name in self.selectors.keys()}

        # Get importance from all selectors
        all_importance = {}
        for name, selector in self.selectors.items():
            importance = selector.get_feature_importance()
            all_importance[name] = importance * weights.get(name, 1.0)

        # Combine importance scores
        combined_importance = np.mean(list(all_importance.values()), axis=0)

        # Select features
        if n_features is None:
            n_features = max(1, X.shape[1] // 2)

        n_select = min(n_features, X.shape[1])
        selected_indices = np.argsort(combined_importance)[-n_select:]

        # Get feature names
        feature_names = self.selectors[list(self.selectors.keys())[0]].feature_names_
        selected_features = [feature_names[i] for i in selected_indices]

        # Create feature scores
        feature_scores = {
            feature_names[i]: float(combined_importance[i])
            for i in selected_indices
        }

        return {
            'success': True,
            'selected_features': selected_features,
            'selected_indices': selected_indices.tolist(),
            'feature_scores': feature_scores,
            'n_selected': len(selected_features),
            'n_total': X.shape[1],
            'method': 'ensemble',
            'individual_importance': all_importance,
            'combined_importance': combined_importance.tolist()
        }

    def get_individual_results(self, X: np.ndarray, y: np.ndarray,
                              n_features: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Get results from individual selectors."""
        results = {}

        for name, selector in self.selectors.items():
            results[name] = selector.select_features(X, y, n_features)

        return results

class AdvancedFeatureSelector:
    """Main advanced feature selector with all methods."""

    def __init__(self, config: Optional[AdvancedSelectionConfig] = None):
        """Initialize advanced selector."""
        self.config = config or AdvancedSelectionConfig()
        self.ensemble_selector = EnsembleAdvancedSelector(self.config)

        tprint_success("🚀 AdvancedFeatureSelector initialized")

    def select_features(self, X: np.ndarray, y: np.ndarray,
                       method: str = 'ensemble',
                       n_features: Optional[int] = None,
                       **kwargs) -> Dict[str, Any]:
        """Select features using advanced methods."""
        tprint_info(f"🔍 Advanced selection: {method}")

        return self.ensemble_selector.select_features(X, y, n_features, method, **kwargs)

    def compare_methods(self, X: np.ndarray, y: np.ndarray,
                       n_features: Optional[int] = None) -> Dict[str, Any]:
        """Compare all advanced methods."""
        tprint_info("🔍 Comparing advanced methods")

        results = self.ensemble_selector.get_individual_results(X, y, n_features)

        # Add ensemble result
        results['ensemble'] = self.ensemble_selector._ensemble_selection(X, y, n_features)

        return {
            'success': True,
            'results': results,
            'comparison_summary': self._create_comparison_summary(results)
        }

    def _create_comparison_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create comparison summary."""
        summary = {}

        for method, result in results.items():
            if result.get('success', False):
                summary[method] = {
                    'n_selected': result['n_selected'],
                    'n_total': result['n_total'],
                    'selection_ratio': result['n_selected'] / result['n_total'],
                    'top_features': result['selected_features'][:5]  # Top 5 features
                }

        return summary

def create_advanced_selector(config: Optional[AdvancedSelectionConfig] = None) -> AdvancedFeatureSelector:
    """Create an advanced feature selector."""
    return AdvancedFeatureSelector(config)
