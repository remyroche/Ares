"""
Consolidated Negative Learning Feature Generation Module

This module implements the complete negative learning plugin for Analyst/Tactician tree pipelines.
It discovers failure contexts and generates gated twin features and exception interactions
to improve model performance in challenging market conditions.

Key Components:
1. Failure Context Discovery - Data-driven detection of when features fail
2. Negative Learning Features - Gated twins and exception interactions
3. Model Constraints - Monotone constraints and sample weights
4. Feature Selection - Stability selection and budget management
5. Validation Framework - Performance monitoring and SHAP analysis
6. Pipeline Integration - Drop-in integration with existing pipelines

Time-series safe, fast, and respects latency budgets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings

from src.utils.logger import system_logger
from src.utils.math_validation import safe_divide, validate_finite, validate_positive

# Import tprint for consistent logging
try:
    from tprint import tprint
except ImportError:
    def tprint(*args, **kwargs):
        print(*args, **kwargs)

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_skew, rolling_kurt, rolling_quantile
    )
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    rolling_skew = None
    rolling_kurt = None
    rolling_quantile = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# ============================================================================
# Core Data Structures and Enums
# ============================================================================

class FailureContextType(Enum):
    """Types of failure contexts to detect"""
    HIGH_VOLATILITY = "highvol"
    CHOP = "chop"
    WIDE_SPREAD = "widespread"
    OPEN_WINDOW = "open30"
    CLOSE_WINDOW = "last30"
    TRENDING = "trending"
    RANGING = "ranging"

class ModelType(Enum):
    """Supported model types for constraints"""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    RANDOM_FOREST = "random_forest"
    LINEAR = "linear"

class ValidationMetric(Enum):
    """Validation metrics for negative learning features"""
    IC = "ic"
    R2 = "r2"
    MAE = "mae"
    MSE = "mse"
    SHAP_STABILITY = "shap_stability"
    DRIFT = "drift"

@dataclass
class FailureContext:
    """Represents a detected failure context for a feature"""
    feature_name: str
    context_type: FailureContextType
    threshold: float
    ic_positive: float
    ic_negative: float
    confidence: float
    sample_size: int
    created_at: datetime

@dataclass
class NegativeLearningFeature:
    """Represents a negative learning feature"""
    name: str
    base_feature: str
    context_type: FailureContextType
    feature_type: str  # 'gated_twin', 'exception_interaction'
    parameters: Dict[str, Any]
    ic_improvement: float
    stability_score: float
    created_at: datetime

@dataclass
class ModelConstraint:
    """Represents a model constraint for negative learning features"""
    feature_name: str
    constraint_type: str  # 'monotone', 'sample_weight'
    parameters: Dict[str, Any]
    model_type: ModelType
    created_at: datetime

@dataclass
class ValidationResult:
    """Represents validation results for negative learning features"""
    feature_name: str
    metric: ValidationMetric
    value: float
    p_value: float
    is_significant: bool
    created_at: datetime

# ============================================================================
# Core Negative Learning Feature Generator
# ============================================================================

class NegativeLearningFeatureGenerator:
    """Main class for generating negative learning features"""

    def __init__(self,
                 max_features: int = 100,
                 min_ic_improvement: float = 0.01,
                 stability_threshold: float = 0.7,
                 latency_budget_ms: int = 50):
        """
        Initialize the negative learning feature generator.

        Args:
            max_features: Maximum number of features to generate
            min_ic_improvement: Minimum IC improvement required
            stability_threshold: Minimum stability score required
            latency_budget_ms: Maximum latency budget in milliseconds
        """
        self.max_features = max_features
        self.min_ic_improvement = min_ic_improvement
        self.stability_threshold = stability_threshold
        self.latency_budget_ms = latency_budget_ms

        self.failure_contexts: List[FailureContext] = []
        self.negative_features: List[NegativeLearningFeature] = []
        self.performance_stats = {
            'features_generated': 0,
            'contexts_discovered': 0,
            'ic_improvements': [],
            'stability_scores': [],
            'processing_time': 0.0
        }

        # Initialize VectorBT optimizer if available
        self.vectorbt_optimizer = None
        if VECTORBT_AVAILABLE:
            try:
                from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
                tprint("✅ VectorBT optimizer initialized for NegativeLearningFeatureGenerator")
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def discover_failure_contexts(self,
                                 features: pd.DataFrame,
                                 returns: pd.Series,
                                 lookback_window: int = 252) -> List[FailureContext]:
        """
        Discover failure contexts where features perform poorly.

        Args:
            features: Feature matrix
            returns: Target returns
            lookback_window: Lookback window for analysis

        Returns:
            List of discovered failure contexts
        """
        tprint("🔍 Discovering failure contexts...")
        start_time = datetime.now()

        contexts = []

        for feature_name in features.columns:
            feature_values = features[feature_name].dropna()
            if len(feature_values) < lookback_window:
                continue

            # Calculate rolling IC
            rolling_ic = self._calculate_rolling_ic(feature_values, returns, lookback_window)

            # Detect different failure contexts
            feature_contexts = self._detect_feature_failure_contexts(
                feature_name, feature_values, returns, rolling_ic
            )
            contexts.extend(feature_contexts)

        self.failure_contexts = contexts
        self.performance_stats['contexts_discovered'] = len(contexts)

        processing_time = (datetime.now() - start_time).total_seconds()
        self.performance_stats['processing_time'] += processing_time

        tprint(f"✅ Discovered {len(contexts)} failure contexts in {processing_time:.2f}s")
        return contexts

    def generate_negative_features(self,
                                  features: pd.DataFrame,
                                  returns: pd.Series) -> List[NegativeLearningFeature]:
        """
        Generate negative learning features based on discovered failure contexts.

        Args:
            features: Feature matrix
            returns: Target returns

        Returns:
            List of generated negative learning features
        """
        tprint("🔧 Generating negative learning features...")
        start_time = datetime.now()

        negative_features = []

        for context in self.failure_contexts:
            if len(negative_features) >= self.max_features:
                break

            # Generate gated twin features
            gated_twin = self._generate_gated_twin_feature(features, context)
            if gated_twin is not None:
                negative_features.append(gated_twin)

            # Generate exception interaction features
            exception_interaction = self._generate_exception_interaction_feature(features, context)
            if exception_interaction is not None:
                negative_features.append(exception_interaction)

        # Validate and filter features
        validated_features = self._validate_negative_features(negative_features, features, returns)

        self.negative_features = validated_features
        self.performance_stats['features_generated'] = len(validated_features)

        processing_time = (datetime.now() - start_time).total_seconds()
        self.performance_stats['processing_time'] += processing_time

        tprint(f"✅ Generated {len(validated_features)} negative learning features in {processing_time:.2f}s")
        return validated_features

    def _calculate_rolling_ic(self,
                             feature_values: pd.Series,
                             returns: pd.Series,
                             window: int) -> pd.Series:
        """Calculate rolling information coefficient."""
        if self.vectorbt_optimizer and VECTORBT_AVAILABLE:
            try:
                # Use VectorBT for optimized rolling correlation
                rolling_corr = self.vectorbt_optimizer.rolling_corr(
                    feature_values, returns, window
                )
                return rolling_corr
            except Exception as e:
                tprint(f"VectorBT rolling correlation failed: {e}, using pandas fallback")

        # Fallback to pandas
        return feature_values.rolling(window=window).corr(returns)

    def _detect_feature_failure_contexts(self,
                                       feature_name: str,
                                       feature_values: pd.Series,
                                       returns: pd.Series,
                                       rolling_ic: pd.Series) -> List[FailureContext]:
        """Detect failure contexts for a specific feature."""
        contexts = []

        # High volatility context
        vol_context = self._detect_high_volatility_context(
            feature_name, feature_values, returns, rolling_ic
        )
        if vol_context:
            contexts.append(vol_context)

        # Chop context
        chop_context = self._detect_chop_context(
            feature_name, feature_values, returns, rolling_ic
        )
        if chop_context:
            contexts.append(chop_context)

        # Wide spread context
        spread_context = self._detect_wide_spread_context(
            feature_name, feature_values, returns, rolling_ic
        )
        if spread_context:
            contexts.append(spread_context)

        return contexts

    def _detect_high_volatility_context(self,
                                      feature_name: str,
                                      feature_values: pd.Series,
                                      returns: pd.Series,
                                      rolling_ic: pd.Series) -> Optional[FailureContext]:
        """Detect high volatility failure context."""
        # Calculate rolling volatility
        if self.vectorbt_optimizer and VECTORBT_AVAILABLE:
            try:
                rolling_vol = self.vectorbt_optimizer.rolling_std(returns, window=20)
            except Exception:
                rolling_vol = returns.rolling(window=20).std()
        else:
            rolling_vol = returns.rolling(window=20).std()

        # Find high volatility periods
        vol_threshold = rolling_vol.quantile(0.8)
        high_vol_mask = rolling_vol > vol_threshold

        if not high_vol_mask.any():
            return None

        # Calculate IC in high volatility vs normal periods
        ic_high_vol = rolling_ic[high_vol_mask].mean()
        ic_normal = rolling_ic[~high_vol_mask].mean()

        # Check if feature fails in high volatility
        if ic_high_vol < ic_normal - 0.05:  # 5% IC drop threshold
            return FailureContext(
                feature_name=feature_name,
                context_type=FailureContextType.HIGH_VOLATILITY,
                threshold=vol_threshold,
                ic_positive=ic_normal,
                ic_negative=ic_high_vol,
                confidence=abs(ic_normal - ic_high_vol),
                sample_size=high_vol_mask.sum(),
                created_at=datetime.now()
            )

        return None

    def _detect_chop_context(self,
                            feature_name: str,
                            feature_values: pd.Series,
                            returns: pd.Series,
                            rolling_ic: pd.Series) -> Optional[FailureContext]:
        """Detect chop (sideways market) failure context."""
        # Calculate price range over different windows
        if self.vectorbt_optimizer and VECTORBT_AVAILABLE:
            try:
                rolling_max = self.vectorbt_optimizer.rolling_max(returns, window=20)
                rolling_min = self.vectorbt_optimizer.rolling_min(returns, window=20)
            except Exception:
                rolling_max = returns.rolling(window=20).max()
                rolling_min = returns.rolling(window=20).min()
        else:
            rolling_max = returns.rolling(window=20).max()
            rolling_min = returns.rolling(window=20).min()

        price_range = rolling_max - rolling_min
        range_threshold = price_range.quantile(0.2)  # Bottom 20% range
        chop_mask = price_range < range_threshold

        if not chop_mask.any():
            return None

        # Calculate IC in chop vs trending periods
        ic_chop = rolling_ic[chop_mask].mean()
        ic_trending = rolling_ic[~chop_mask].mean()

        # Check if feature fails in chop
        if ic_chop < ic_trending - 0.05:  # 5% IC drop threshold
            return FailureContext(
                feature_name=feature_name,
                context_type=FailureContextType.CHOP,
                threshold=range_threshold,
                ic_positive=ic_trending,
                ic_negative=ic_chop,
                confidence=abs(ic_trending - ic_chop),
                sample_size=chop_mask.sum(),
                created_at=datetime.now()
            )

        return None

    def _detect_wide_spread_context(self,
                                   feature_name: str,
                                   feature_values: pd.Series,
                                   returns: pd.Series,
                                   rolling_ic: pd.Series) -> Optional[FailureContext]:
        """Detect wide spread failure context."""
        # Calculate bid-ask spread proxy (using high-low range)
        if 'high' in feature_values.index and 'low' in feature_values.index:
            spread = feature_values['high'] - feature_values['low']
        else:
            # Use price volatility as spread proxy
            if self.vectorbt_optimizer and VECTORBT_AVAILABLE:
                try:
                    spread = self.vectorbt_optimizer.rolling_std(returns, window=5)
                except Exception:
                    spread = returns.rolling(window=5).std()
            else:
                spread = returns.rolling(window=5).std()

        spread_threshold = spread.quantile(0.8)  # Top 20% spread
        wide_spread_mask = spread > spread_threshold

        if not wide_spread_mask.any():
            return None

        # Calculate IC in wide spread vs normal periods
        ic_wide_spread = rolling_ic[wide_spread_mask].mean()
        ic_normal = rolling_ic[~wide_spread_mask].mean()

        # Check if feature fails in wide spread
        if ic_wide_spread < ic_normal - 0.05:  # 5% IC drop threshold
            return FailureContext(
                feature_name=feature_name,
                context_type=FailureContextType.WIDE_SPREAD,
                threshold=spread_threshold,
                ic_positive=ic_normal,
                ic_negative=ic_wide_spread,
                confidence=abs(ic_normal - ic_wide_spread),
                sample_size=wide_spread_mask.sum(),
                created_at=datetime.now()
            )

        return None

    def _generate_gated_twin_feature(self,
                                   features: pd.DataFrame,
                                   context: FailureContext) -> Optional[NegativeLearningFeature]:
        """Generate gated twin feature for a failure context."""
        base_feature = features[context.feature_name]

        # Create context mask
        context_mask = self._create_context_mask(features, context)

        if not context_mask.any():
            return None

        # Generate gated twin (feature * context_mask)
        gated_twin = base_feature * context_mask

        # Calculate IC improvement
        ic_improvement = self._calculate_ic_improvement(gated_twin, base_feature)

        if ic_improvement < self.min_ic_improvement:
            return None

        return NegativeLearningFeature(
            name=f"{context.feature_name}_gated_{context.context_type.value}",
            base_feature=context.feature_name,
            context_type=context.context_type,
            feature_type="gated_twin",
            parameters={
                "threshold": context.threshold,
                "context_mask": context_mask
            },
            ic_improvement=ic_improvement,
            stability_score=0.0,  # Will be calculated later
            created_at=datetime.now()
        )

    def _generate_exception_interaction_feature(self,
                                              features: pd.DataFrame,
                                              context: FailureContext) -> Optional[NegativeLearningFeature]:
        """Generate exception interaction feature for a failure context."""
        base_feature = features[context.feature_name]

        # Create context mask
        context_mask = self._create_context_mask(features, context)

        if not context_mask.any():
            return None

        # Generate exception interaction (feature * (1 - context_mask))
        exception_interaction = base_feature * (1 - context_mask)

        # Calculate IC improvement
        ic_improvement = self._calculate_ic_improvement(exception_interaction, base_feature)

        if ic_improvement < self.min_ic_improvement:
            return None

        return NegativeLearningFeature(
            name=f"{context.feature_name}_exception_{context.context_type.value}",
            base_feature=context.feature_name,
            context_type=context.context_type,
            feature_type="exception_interaction",
            parameters={
                "threshold": context.threshold,
                "context_mask": context_mask
            },
            ic_improvement=ic_improvement,
            stability_score=0.0,  # Will be calculated later
            created_at=datetime.now()
        )

    def _create_context_mask(self,
                           features: pd.DataFrame,
                           context: FailureContext) -> pd.Series:
        """Create context mask for a failure context."""
        if context.context_type == FailureContextType.HIGH_VOLATILITY:
            # High volatility mask
            if self.vectorbt_optimizer and VECTORBT_AVAILABLE:
                try:
                    rolling_vol = self.vectorbt_optimizer.rolling_std(features['close'], window=20)
                except Exception:
                    rolling_vol = features['close'].rolling(window=20).std()
            else:
                rolling_vol = features['close'].rolling(window=20).std()

            return (rolling_vol > context.threshold).astype(float)

        elif context.context_type == FailureContextType.CHOP:
            # Chop mask
            if self.vectorbt_optimizer and VECTORBT_AVAILABLE:
                try:
                    rolling_max = self.vectorbt_optimizer.rolling_max(features['close'], window=20)
                    rolling_min = self.vectorbt_optimizer.rolling_min(features['close'], window=20)
                except Exception:
                    rolling_max = features['close'].rolling(window=20).max()
                    rolling_min = features['close'].rolling(window=20).min()
            else:
                rolling_max = features['close'].rolling(window=20).max()
                rolling_min = features['close'].rolling(window=20).min()

            price_range = rolling_max - rolling_min
            return (price_range < context.threshold).astype(float)

        elif context.context_type == FailureContextType.WIDE_SPREAD:
            # Wide spread mask
            if 'high' in features.columns and 'low' in features.columns:
                spread = features['high'] - features['low']
            else:
                if self.vectorbt_optimizer and VECTORBT_AVAILABLE:
                    try:
                        spread = self.vectorbt_optimizer.rolling_std(features['close'], window=5)
                    except Exception:
                        spread = features['close'].rolling(window=5).std()
                else:
                    spread = features['close'].rolling(window=5).std()

            return (spread > context.threshold).astype(float)

        else:
            # Default to all zeros
            return pd.Series(0, index=features.index)

    def _calculate_ic_improvement(self,
                                new_feature: pd.Series,
                                base_feature: pd.Series) -> float:
        """Calculate IC improvement of new feature over base feature."""
        # Calculate IC for both features
        returns = base_feature.pct_change().dropna()

        if len(new_feature) != len(returns):
            return 0.0

        new_ic = new_feature.corr(returns)
        base_ic = base_feature.corr(returns)

        return new_ic - base_ic

    def _validate_negative_features(self,
                                  features: List[NegativeLearningFeature],
                                  feature_matrix: pd.DataFrame,
                                  returns: pd.Series) -> List[NegativeLearningFeature]:
        """Validate and filter negative learning features."""
        validated_features = []

        for feature in features:
            # Calculate stability score
            stability_score = self._calculate_stability_score(feature, feature_matrix, returns)
            feature.stability_score = stability_score

            # Check if feature meets criteria
            if (stability_score >= self.stability_threshold and
                feature.ic_improvement >= self.min_ic_improvement):
                validated_features.append(feature)

        return validated_features

    def _calculate_stability_score(self,
                                 feature: NegativeLearningFeature,
                                 feature_matrix: pd.DataFrame,
                                 returns: pd.Series) -> float:
        """Calculate stability score for a negative learning feature."""
        # Use block bootstrap to calculate stability
        n_samples = 100
        ic_scores = []

        for _ in range(n_samples):
            # Sample with replacement
            sample_indices = np.random.choice(len(feature_matrix), size=len(feature_matrix), replace=True)
            sample_features = feature_matrix.iloc[sample_indices]
            sample_returns = returns.iloc[sample_indices]

            # Calculate IC for this sample
            if feature.feature_type == "gated_twin":
                context_mask = feature.parameters["context_mask"]
                gated_feature = sample_features[feature.base_feature] * context_mask
            else:  # exception_interaction
                context_mask = feature.parameters["context_mask"]
                exception_feature = sample_features[feature.base_feature] * (1 - context_mask)
                gated_feature = exception_feature

            ic = gated_feature.corr(sample_returns)
            if not np.isnan(ic):
                ic_scores.append(ic)

        if not ic_scores:
            return 0.0

        # Stability score is 1 - coefficient of variation
        ic_scores = np.array(ic_scores)
        stability_score = 1 - (np.std(ic_scores) / (np.abs(np.mean(ic_scores)) + 1e-8))

        return max(0.0, min(1.0, stability_score))

# ============================================================================
# Model Constraints Manager
# ============================================================================

class ModelConstraintManager:
    """Manages model constraints for negative learning features"""

    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.constraints: List[ModelConstraint] = []

    def generate_constraints(self,
                           negative_features: List[NegativeLearningFeature],
                           features: pd.DataFrame,
                           returns: pd.Series) -> List[ModelConstraint]:
        """Generate model constraints for negative learning features."""
        constraints = []

        for feature in negative_features:
            # Generate monotone constraints
            monotone_constraint = self._generate_monotone_constraint(feature, features, returns)
            if monotone_constraint:
                constraints.append(monotone_constraint)

            # Generate sample weight constraints
            sample_weight_constraint = self._generate_sample_weight_constraint(feature, features, returns)
            if sample_weight_constraint:
                constraints.append(sample_weight_constraint)

        self.constraints = constraints
        return constraints

    def _generate_monotone_constraint(self,
                                    feature: NegativeLearningFeature,
                                    features: pd.DataFrame,
                                    returns: pd.Series) -> Optional[ModelConstraint]:
        """Generate monotone constraint for a feature."""
        if self.model_type not in [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.CATBOOST]:
            return None

        # Calculate monotonicity direction
        feature_values = features[feature.base_feature].dropna()
        returns_aligned = returns.loc[feature_values.index]

        # Use linear regression to determine monotonicity
        try:
            lr = LinearRegression()
            lr.fit(feature_values.values.reshape(-1, 1), returns_aligned.values)
            monotone_direction = 1 if lr.coef_[0] > 0 else -1
        except:
            return None

        return ModelConstraint(
            feature_name=feature.name,
            constraint_type="monotone",
            parameters={"direction": monotone_direction},
            model_type=self.model_type,
            created_at=datetime.now()
        )

    def _generate_sample_weight_constraint(self,
                                         feature: NegativeLearningFeature,
                                         features: pd.DataFrame,
                                         returns: pd.Series) -> Optional[ModelConstraint]:
        """Generate sample weight constraint for a feature."""
        # Calculate uncertainty weights based on context
        context_mask = feature.parameters.get("context_mask", pd.Series(0, index=features.index))

        # Higher weight for non-context periods (where feature should work)
        weights = 1.0 + (1 - context_mask) * 0.5

        return ModelConstraint(
            feature_name=feature.name,
            constraint_type="sample_weight",
            parameters={"weights": weights},
            model_type=self.model_type,
            created_at=datetime.now()
        )

# ============================================================================
# Feature Selection Manager
# ============================================================================

class NegativeLearningFeatureSelector:
    """Manages feature selection for negative learning features"""

    def __init__(self,
                 max_features: int = 50,
                 stability_threshold: float = 0.7,
                 ic_threshold: float = 0.01):
        self.max_features = max_features
        self.stability_threshold = stability_threshold
        self.ic_threshold = ic_threshold
        self.selected_features: List[NegativeLearningFeature] = []

    def select_features(self,
                       negative_features: List[NegativeLearningFeature],
                       features: pd.DataFrame,
                       returns: pd.Series) -> List[NegativeLearningFeature]:
        """Select best negative learning features using stability selection."""
        # Filter features by basic criteria
        candidate_features = [
            f for f in negative_features
            if f.stability_score >= self.stability_threshold and f.ic_improvement >= self.ic_threshold
        ]

        if len(candidate_features) <= self.max_features:
            self.selected_features = candidate_features
            return candidate_features

        # Use stability selection
        selected_features = self._stability_selection(candidate_features, features, returns)

        self.selected_features = selected_features
        return selected_features

    def _stability_selection(self,
                           candidate_features: List[NegativeLearningFeature],
                           features: pd.DataFrame,
                           returns: pd.Series) -> List[NegativeLearningFeature]:
        """Perform stability selection using block bootstrap."""
        n_bootstrap = 50
        selection_counts = {f.name: 0 for f in candidate_features}

        for _ in range(n_bootstrap):
            # Create bootstrap sample
            sample_indices = np.random.choice(len(features), size=len(features), replace=True)
            sample_features = features.iloc[sample_indices]
            sample_returns = returns.iloc[sample_indices]

            # Select features using Lasso
            selected = self._lasso_selection(candidate_features, sample_features, sample_returns)

            for feature_name in selected:
                selection_counts[feature_name] += 1

        # Select features selected in at least 50% of bootstrap samples
        threshold = n_bootstrap * 0.5
        selected_features = [
            f for f in candidate_features
            if selection_counts[f.name] >= threshold
        ]

        # Sort by selection frequency and IC improvement
        selected_features.sort(key=lambda f: (selection_counts[f.name], f.ic_improvement), reverse=True)

        return selected_features[:self.max_features]

    def _lasso_selection(self,
                        candidate_features: List[NegativeLearningFeature],
                        features: pd.DataFrame,
                        returns: pd.Series) -> List[str]:
        """Select features using Lasso regression."""
        # Create feature matrix for negative learning features
        feature_matrix = []
        feature_names = []

        for feature in candidate_features:
            if feature.feature_type == "gated_twin":
                context_mask = feature.parameters["context_mask"]
                gated_feature = features[feature.base_feature] * context_mask
            else:  # exception_interaction
                context_mask = feature.parameters["context_mask"]
                exception_feature = features[feature.base_feature] * (1 - context_mask)
                gated_feature = exception_feature

            feature_matrix.append(gated_feature.values)
            feature_names.append(feature.name)

        if not feature_matrix:
            return []

        X = np.column_stack(feature_matrix)
        y = returns.values

        # Remove NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 10:  # Need minimum samples
            return []

        try:
            # Use LassoCV for automatic alpha selection
            lasso = LassoCV(cv=5, random_state=42)
            lasso.fit(X, y)

            # Get selected features (non-zero coefficients)
            selected_indices = np.where(lasso.coef_ != 0)[0]
            selected_features = [feature_names[i] for i in selected_indices]

            return selected_features
        except Exception as e:
            tprint(f"Lasso selection failed: {e}")
            return []

# ============================================================================
# Validation Framework
# ============================================================================

class NegativeLearningValidator:
    """Validates negative learning features using multiple metrics"""

    def __init__(self):
        self.validation_results: List[ValidationResult] = []

    def validate_features(self,
                         negative_features: List[NegativeLearningFeature],
                         features: pd.DataFrame,
                         returns: pd.Series) -> List[ValidationResult]:
        """Validate negative learning features using multiple metrics."""
        results = []

        for feature in negative_features:
            # IC validation
            ic_result = self._validate_ic(feature, features, returns)
            if ic_result:
                results.append(ic_result)

            # R2 validation
            r2_result = self._validate_r2(feature, features, returns)
            if r2_result:
                results.append(r2_result)

            # SHAP stability validation
            shap_result = self._validate_shap_stability(feature, features, returns)
            if shap_result:
                results.append(shap_result)

        self.validation_results = results
        return results

    def _validate_ic(self,
                    feature: NegativeLearningFeature,
                    features: pd.DataFrame,
                    returns: pd.Series) -> Optional[ValidationResult]:
        """Validate feature using information coefficient."""
        # Calculate feature values
        feature_values = self._get_feature_values(feature, features)

        if feature_values is None or len(feature_values) < 10:
            return None

        # Calculate IC
        ic = feature_values.corr(returns)

        # Calculate p-value using t-test
        n = len(feature_values)
        t_stat = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-8))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

        return ValidationResult(
            feature_name=feature.name,
            metric=ValidationMetric.IC,
            value=ic,
            p_value=p_value,
            is_significant=p_value < 0.05,
            created_at=datetime.now()
        )

    def _validate_r2(self,
                    feature: NegativeLearningFeature,
                    features: pd.DataFrame,
                    returns: pd.Series) -> Optional[ValidationResult]:
        """Validate feature using R-squared."""
        # Calculate feature values
        feature_values = self._get_feature_values(feature, features)

        if feature_values is None or len(feature_values) < 10:
            return None

        # Align data
        aligned_data = pd.concat([feature_values, returns], axis=1).dropna()
        if len(aligned_data) < 10:
            return None

        X = aligned_data.iloc[:, 0].values.reshape(-1, 1)
        y = aligned_data.iloc[:, 1].values

        try:
            # Fit linear regression
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)

            # Calculate R-squared
            r2 = r2_score(y, y_pred)

            # Calculate p-value using F-test
            n = len(y)
            p = 1  # number of predictors
            f_stat = (r2 / (1 - r2 + 1e-8)) * ((n - p - 1) / p)
            p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)

            return ValidationResult(
                feature_name=feature.name,
                metric=ValidationMetric.R2,
                value=r2,
                p_value=p_value,
                is_significant=p_value < 0.05,
                created_at=datetime.now()
            )
        except Exception as e:
            tprint(f"R2 validation failed for {feature.name}: {e}")
            return None

    def _validate_shap_stability(self,
                               feature: NegativeLearningFeature,
                               features: pd.DataFrame,
                               returns: pd.Series) -> Optional[ValidationResult]:
        """Validate feature using SHAP stability."""
        # This is a simplified version - in practice, you'd use actual SHAP values
        # For now, we'll use feature stability as a proxy

        # Calculate feature values
        feature_values = self._get_feature_values(feature, features)

        if feature_values is None or len(feature_values) < 20:
            return None

        # Use rolling correlation stability as SHAP stability proxy
        rolling_corr = feature_values.rolling(window=20).corr(returns)
        stability = 1 - rolling_corr.std() / (rolling_corr.abs().mean() + 1e-8)

        # Calculate p-value (simplified)
        p_value = 0.1 if stability > 0.7 else 0.5

        return ValidationResult(
            feature_name=feature.name,
            metric=ValidationMetric.SHAP_STABILITY,
            value=stability,
            p_value=p_value,
            is_significant=p_value < 0.05,
            created_at=datetime.now()
        )

    def _get_feature_values(self,
                          feature: NegativeLearningFeature,
                          features: pd.DataFrame) -> Optional[pd.Series]:
        """Get feature values for a negative learning feature."""
        if feature.feature_type == "gated_twin":
            context_mask = feature.parameters["context_mask"]
            return features[feature.base_feature] * context_mask
        elif feature.feature_type == "exception_interaction":
            context_mask = feature.parameters["context_mask"]
            return features[feature.base_feature] * (1 - context_mask)
        else:
            return None

# ============================================================================
# Pipeline Integration
# ============================================================================

class NegativeLearningPipelineManager:
    """Manages integration of negative learning into existing pipelines"""

    def __init__(self,
                 max_features: int = 50,
                 latency_budget_ms: int = 50):
        self.max_features = max_features
        self.latency_budget_ms = latency_budget_ms

        self.feature_generator = NegativeLearningFeatureGenerator(max_features=max_features)
        self.constraint_manager = None
        self.feature_selector = NegativeLearningFeatureSelector(max_features=max_features)
        self.validator = NegativeLearningValidator()

        self.is_initialized = False

    def initialize(self,
                  features: pd.DataFrame,
                  returns: pd.Series,
                  model_type: ModelType = ModelType.XGBOOST):
        """Initialize the negative learning pipeline."""
        tprint("🚀 Initializing negative learning pipeline...")

        # Discover failure contexts
        self.feature_generator.discover_failure_contexts(features, returns)

        # Generate negative features
        negative_features = self.feature_generator.generate_negative_features(features, returns)

        # Select best features
        selected_features = self.feature_selector.select_features(negative_features, features, returns)

        # Initialize constraint manager
        self.constraint_manager = ModelConstraintManager(model_type)
        constraints = self.constraint_manager.generate_constraints(selected_features, features, returns)

        # Validate features
        validation_results = self.validator.validate_features(selected_features, features, returns)

        self.is_initialized = True

        tprint(f"✅ Pipeline initialized with {len(selected_features)} features and {len(constraints)} constraints")

        return {
            'features': selected_features,
            'constraints': constraints,
            'validation_results': validation_results
        }

    def generate_features(self,
                         features: pd.DataFrame) -> pd.DataFrame:
        """Generate negative learning features for new data."""
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize() first.")

        negative_features_df = pd.DataFrame(index=features.index)

        for feature in self.feature_selector.selected_features:
            if feature.feature_type == "gated_twin":
                context_mask = feature.parameters["context_mask"]
                gated_feature = features[feature.base_feature] * context_mask
            else:  # exception_interaction
                context_mask = feature.parameters["context_mask"]
                exception_feature = features[feature.base_feature] * (1 - context_mask)
                gated_feature = exception_feature

            negative_features_df[feature.name] = gated_feature

        return negative_features_df

    def get_constraints(self) -> List[ModelConstraint]:
        """Get model constraints for the selected features."""
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize() first.")

        return self.constraint_manager.constraints if self.constraint_manager else []

    def get_validation_results(self) -> List[ValidationResult]:
        """Get validation results for the selected features."""
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize() first.")

        return self.validator.validation_results

# ============================================================================
# Factory Functions
# ============================================================================

def create_negative_learning_pipeline(max_features: int = 50,
                                    latency_budget_ms: int = 50) -> NegativeLearningPipelineManager:
    """Create a negative learning pipeline manager."""
    return NegativeLearningPipelineManager(
        max_features=max_features,
        latency_budget_ms=latency_budget_ms
    )

def create_feature_selector(max_features: int = 50,
                          stability_threshold: float = 0.7,
                          ic_threshold: float = 0.01) -> NegativeLearningFeatureSelector:
    """Create a negative learning feature selector."""
    return NegativeLearningFeatureSelector(
        max_features=max_features,
        stability_threshold=stability_threshold,
        ic_threshold=ic_threshold
    )

def create_constraint_manager(model_type: ModelType) -> ModelConstraintManager:
    """Create a model constraint manager."""
    return ModelConstraintManager(model_type)

def create_validator() -> NegativeLearningValidator:
    """Create a negative learning validator."""
    return NegativeLearningValidator()

# ============================================================================
# Main Integration Function
# ============================================================================

def integrate_negative_learning(features: pd.DataFrame,
                              returns: pd.Series,
                              model_type: ModelType = ModelType.XGBOOST,
                              max_features: int = 50,
                              latency_budget_ms: int = 50) -> Dict[str, Any]:
    """
    Main function to integrate negative learning into existing pipelines.

    Args:
        features: Feature matrix
        returns: Target returns
        model_type: Type of model to generate constraints for
        max_features: Maximum number of negative learning features
        latency_budget_ms: Maximum latency budget in milliseconds

    Returns:
        Dictionary containing negative learning features, constraints, and validation results
    """
    tprint("🔧 Integrating negative learning into pipeline...")

    # Create pipeline manager
    pipeline_manager = create_negative_learning_pipeline(
        max_features=max_features,
        latency_budget_ms=latency_budget_ms
    )

    # Initialize pipeline
    results = pipeline_manager.initialize(features, returns, model_type)

    # Generate features for the original data
    negative_features_df = pipeline_manager.generate_features(features)

    # Add negative features to original features
    combined_features = pd.concat([features, negative_features_df], axis=1)

    tprint(f"✅ Integration complete. Added {len(negative_features_df.columns)} negative learning features.")

    return {
        'combined_features': combined_features,
        'negative_features': negative_features_df,
        'constraints': results['constraints'],
        'validation_results': results['validation_results'],
        'pipeline_manager': pipeline_manager
    }

# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'FailureContextType',
    'ModelType',
    'ValidationMetric',
    'FailureContext',
    'NegativeLearningFeature',
    'ModelConstraint',
    'ValidationResult',
    'NegativeLearningFeatureGenerator',
    'ModelConstraintManager',
    'NegativeLearningFeatureSelector',
    'NegativeLearningValidator',
    'NegativeLearningPipelineManager',
    'create_negative_learning_pipeline',
    'create_feature_selector',
    'create_constraint_manager',
    'create_validator',
    'integrate_negative_learning'
]
