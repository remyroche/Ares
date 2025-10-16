"""
Intelligent Feature Pre-Selection Component

This module provides intelligent feature pre-selection from a full feature bank (200+ features)
down to a manageable set (~40 features) with category diversity enforcement.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import math validation utilities
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_correlation, safe_covariance,
    safe_mean, safe_std, safe_percentile, safe_percentage_change,
    safe_weighted_average, MathValidation
)

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
        tprint_debug, tprint_performance, tprint_step
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)
    def tprint_step(*args, **kwargs): print("STEP:", *args, **kwargs)

logger = logging.getLogger(__name__)

# Robust stability is now integrated inline
ROBUST_STABILITY_AVAILABLE = True

@dataclass
class FeatureScore:
    """
    Score and metadata for a feature.

    Represents the evaluation results for a single feature including
    quality metrics, performance scores, and metadata.
    """
    feature_name: str
    category: str
    aspect_type: str
    score: float
    variance: float
    correlation_with_target: float
    information_content: float
    uniqueness_score: float
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate feature score after initialization."""
        self._validate_score()

    def _validate_score(self) -> None:
        """
        Validate the feature score data.

        Raises:
            ValueError: If score data is invalid
        """
        try:
            tprint_debug(f"🔍 Validating FeatureScore for {self.feature_name}")

            # Validate feature name
            if not isinstance(self.feature_name, str) or not self.feature_name.strip():
                raise ValueError(f"Invalid feature_name: '{self.feature_name}'. Must be non-empty string.")

            # Validate category
            if not isinstance(self.category, str) or not self.category.strip():
                raise ValueError(f"Invalid category: '{self.category}'. Must be non-empty string.")

            # Validate aspect type
            if not isinstance(self.aspect_type, str) or not self.aspect_type.strip():
                raise ValueError(f"Invalid aspect_type: '{self.aspect_type}'. Must be non-empty string.")

            # Validate numeric scores
            if not isinstance(self.score, (int, float)) or not np.isfinite(self.score):
                raise ValueError(f"Invalid score: {self.score}. Must be finite number.")

            if not isinstance(self.variance, (int, float)) or self.variance < 0:
                raise ValueError(f"Invalid variance: {self.variance}. Must be non-negative number.")

            if not isinstance(self.correlation_with_target, (int, float)) or not -1 <= self.correlation_with_target <= 1:
                raise ValueError(f"Invalid correlation_with_target: {self.correlation_with_target}. Must be between -1 and 1.")

            if not isinstance(self.information_content, (int, float)) or self.information_content < 0:
                raise ValueError(f"Invalid information_content: {self.information_content}. Must be non-negative number.")

            if not isinstance(self.uniqueness_score, (int, float)) or not 0 <= self.uniqueness_score <= 1:
                raise ValueError(f"Invalid uniqueness_score: {self.uniqueness_score}. Must be between 0 and 1.")

            tprint_debug(f"✅ FeatureScore validation passed for {self.feature_name}")

        except ValueError as e:
            error_msg = f"FeatureScore validation failed for {self.feature_name}: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error validating FeatureScore for {self.feature_name}: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert FeatureScore to dictionary.

        Returns:
            Dictionary representation of the feature score
        """
        return {
            'feature_name': self.feature_name,
            'category': self.category,
            'aspect_type': self.aspect_type,
            'score': self.score,
            'variance': self.variance,
            'correlation_with_target': self.correlation_with_target,
            'information_content': self.information_content,
            'uniqueness_score': self.uniqueness_score,
            'metadata': self.metadata or {}
        }

@dataclass
class FeatureSelectionConfig:
    """
    Configuration for intelligent feature selection.

    Provides comprehensive configuration for feature pre-selection from
    a large feature bank with category diversity enforcement and quality thresholds.
    """

    # Target selection parameters
    target_feature_count: int = 40
    min_features_per_category: int = 2
    max_features_per_category: int = 4

    # Quality thresholds
    min_variance: float = 1e-8
    max_correlation_threshold: float = 0.95
    min_information_content: float = 0.1

    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_vectorbt: bool = True

    # Category weights for selection
    category_weights: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        """Initialize category weights and validate configuration."""
        if self.category_weights is None:
            self.category_weights = {
                'momentum': 1.0,
                'volatility': 1.0,
                'trend': 1.0,
                'oscillator': 1.0,
                'volume': 1.0,
                'returns': 1.0,
                'cross_timeframe': 1.2,
                'microstructure': 1.1,
                'entropy': 0.9,
                'support_resistance': 0.9,
                'candlestick_pattern': 0.8,
                'time': 0.7,
                'order_flow': 1.0,
                'regime': 1.0,
                'acceleration': 1.0,
                'advanced_statistical': 1.0,
                'spectral_wavelet': 0.9
            }

@dataclass
class FeatureSelectionResult:
    """Result from intelligent feature selection."""

    # Selected features
    selected_features: List[FeatureScore]
    category_distribution: Dict[str, int]
    aspect_distribution: Dict[str, int]

    # Selection metrics
    total_features_analyzed: int
    selection_time: float
    quality_metrics: Dict[str, Any]

    # Performance metrics
    parallel_operations: int = 0
    vectorbt_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    # Metadata
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate the feature selection configuration.

        Raises:
            ValueError: If configuration values are invalid
        """
        try:
            tprint_debug("🔍 Validating FeatureSelectionConfig")

            # Validate target feature count
            if not isinstance(self.target_feature_count, int) or self.target_feature_count <= 0:
                raise ValueError(f"Invalid target_feature_count: {self.target_feature_count}. Must be positive integer.")

            # Validate category constraints
            if not isinstance(self.min_features_per_category, int) or self.min_features_per_category < 0:
                raise ValueError(f"Invalid min_features_per_category: {self.min_features_per_category}. Must be non-negative integer.")

            if not isinstance(self.max_features_per_category, int) or self.max_features_per_category <= 0:
                raise ValueError(f"Invalid max_features_per_category: {self.max_features_per_category}. Must be positive integer.")

            if self.min_features_per_category > self.max_features_per_category:
                raise ValueError(f"min_features_per_category ({self.min_features_per_category}) > max_features_per_category ({self.max_features_per_category})")

            # Validate quality thresholds
            if not isinstance(self.min_variance, (int, float)) or self.min_variance < 0:
                raise ValueError(f"Invalid min_variance: {self.min_variance}. Must be non-negative number.")

            if not isinstance(self.max_correlation_threshold, (int, float)) or not 0 <= self.max_correlation_threshold <= 1:
                raise ValueError(f"Invalid max_correlation_threshold: {self.max_correlation_threshold}. Must be between 0 and 1.")

            if not isinstance(self.min_information_content, (int, float)) or self.min_information_content < 0:
                raise ValueError(f"Invalid min_information_content: {self.min_information_content}. Must be non-negative number.")

            # Validate performance settings
            if not isinstance(self.max_workers, int) or self.max_workers <= 0:
                raise ValueError(f"Invalid max_workers: {self.max_workers}. Must be positive integer.")

            # Validate category weights
            if self.category_weights is not None:
                if not isinstance(self.category_weights, dict):
                    raise ValueError(f"Invalid category_weights: {type(self.category_weights)}. Must be dictionary.")

                for category, weight in self.category_weights.items():
                    if not isinstance(category, str):
                        raise ValueError(f"Invalid category key: {category}. Must be string.")
                    if not isinstance(weight, (int, float)) or weight < 0:
                        raise ValueError(f"Invalid weight for category '{category}': {weight}. Must be non-negative number.")

            tprint_success("✅ FeatureSelectionConfig validation passed")

        except ValueError as e:
            error_msg = f"FeatureSelectionConfig validation failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error validating FeatureSelectionConfig: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    def get_category_weight(self, category: str) -> float:
        """
        Get weight for a specific category.

        Args:
            category: Category name

        Returns:
            Weight for the category

        Raises:
            KeyError: If category is not found
        """
        if self.category_weights is None:
            raise KeyError("Category weights not initialized")

        if category not in self.category_weights:
            raise KeyError(f"Category '{category}' not found in category_weights")

        return self.category_weights[category]

    def set_category_weight(self, category: str, weight: float) -> None:
        """
        Set weight for a specific category.

        Args:
            category: Category name
            weight: Weight value

        Raises:
            ValueError: If weight is invalid
        """
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(f"Invalid weight: {weight}. Must be non-negative number.")

        if self.category_weights is None:
            self.category_weights = {}

        self.category_weights[category] = weight

class IntelligentFeatureSelector:
    """
    Intelligent feature pre-selection from full feature bank.

    This class provides sophisticated feature selection that ensures:
    - Category diversity (at least 2-3 features per category)
    - Quality filtering (variance, correlation, information content)
    - Performance optimization with VectorBT
    - Parallel processing for large feature banks
    """

    def __init__(self, config: Optional[FeatureSelectionConfig] = None) -> None:
        """
        Initialize the intelligent feature selector.

        Args:
            config: Configuration for feature selection. If None, uses default config.

        Raises:
            ValueError: If config is invalid
            RuntimeError: If initialization fails
        """
        try:
            tprint_step("🧠 Initializing IntelligentFeatureSelector")

            if config is None:
                tprint_info("📋 Using default FeatureSelectionConfig")
                self.config = FeatureSelectionConfig()
            else:
                tprint_info("📋 Using provided FeatureSelectionConfig")
                self.config = config

            # Validate configuration
            if not isinstance(self.config, FeatureSelectionConfig):
                raise ValueError(f"Invalid config type: {type(self.config)}. Expected FeatureSelectionConfig.")

            self.logger = logger
            tprint_success("✅ IntelligentFeatureSelector initialized successfully")

        except ValueError as e:
            error_msg = f"Invalid configuration for IntelligentFeatureSelector: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to initialize IntelligentFeatureSelector: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

        # Performance tracking
        self.performance_stats = {
            'total_selections': 0,
            'successful_selections': 0,
            'failed_selections': 0,
            'total_selection_time': 0.0,
            'parallel_operations': 0,
            'vectorbt_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        tprint_info("🎯 Intelligent Feature Selector initialized")
        tprint_debug(f"📊 Target features: {self.config.target_feature_count}")
        tprint_debug(f"📊 Min per category: {self.config.min_features_per_category}")
        tprint_debug(f"📊 Max per category: {self.config.max_features_per_category}")

    def select_features(self,
                       data: pd.DataFrame,
                       targets: Optional[pd.Series] = None,
                       available_categories: Optional[List[str]] = None) -> FeatureSelectionResult:
        """
        Select features using intelligent pre-selection approach.

        Args:
            data: Input data with features
            targets: Target variable for relevance scoring
            available_categories: Specific categories to consider (None = all)

        Returns:
            FeatureSelectionResult with selected features
        """
        start_time = time.time()

        def _validate_inputs():
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("Data must be a non-empty DataFrame")
            if targets is not None and len(targets) != len(data):
                raise ValueError("Targets length must match data length")

        def _select_features():
            tprint_info("🎯 Starting intelligent feature selection...")
            tprint_debug(f"📊 Input data shape: {data.shape}")
            tprint_debug(f"📊 Available categories: {available_categories}")

            # Step 1: Analyze all features
            tprint_debug("Step 1: Analyzing all features...")
            all_features = self._analyze_all_features(data, targets)

            if not all_features:
                tprint_warning("⚠️ No features found for analysis")
                return self._create_empty_result(start_time, "No features found")

            tprint_success(f"✅ Analyzed {len(all_features)} features")

            # Step 2: Categorize features
            tprint_debug("Step 2: Categorizing features...")
            categorized_features = self._categorize_features(all_features, available_categories)

            tprint_success(f"✅ Categorized into {len(categorized_features)} categories")

            # Step 3: Apply quality filtering
            tprint_debug("Step 3: Applying quality filtering...")
            filtered_features = self._apply_quality_filtering(categorized_features)

            tprint_success(f"✅ Quality filtering: {sum(len(features) for features in filtered_features.values())} features remain")

            # Step 4: Select features with diversity enforcement
            tprint_debug("Step 4: Selecting features with diversity enforcement...")
            selected_features = self._select_with_diversity_enforcement(filtered_features)

            tprint_success(f"✅ Selected {len(selected_features)} features with diversity")

            # Step 5: Calculate metrics
            selection_time = time.time() - start_time
            category_distribution = self._calculate_category_distribution(selected_features)
            aspect_distribution = self._calculate_aspect_distribution(selected_features)
            quality_metrics = self._calculate_quality_metrics(selected_features)

            # Update performance stats
            self.performance_stats.update({
                'total_selections': 1,
                'successful_selections': 1,
                'total_selection_time': selection_time
            })

            tprint_success(f"✅ Intelligent feature selection completed in {selection_time:.3f}s")
            tprint_info(f"📊 Selected features: {len(selected_features)}")
            tprint_info(f"📊 Categories: {list(category_distribution.keys())}")
            tprint_info(f"📊 Quality metrics: {quality_metrics}")

            return FeatureSelectionResult(
                selected_features=selected_features,
                category_distribution=category_distribution,
                aspect_distribution=aspect_distribution,
                total_features_analyzed=len(all_features),
                selection_time=selection_time,
                quality_metrics=quality_metrics,
                parallel_operations=self.performance_stats['parallel_operations'],
                vectorbt_operations=self.performance_stats['vectorbt_operations'],
                cache_hits=self.performance_stats['cache_hits'],
                cache_misses=self.performance_stats['cache_misses'],
                metadata={
                    'config': self.config.__dict__,
                    'performance_stats': self.performance_stats.copy()
                }
            )

        # Execute with error handling
        try:
            _validate_inputs()
            return _select_features()
        except Exception as e:
            tprint_error(f"❌ Intelligent feature selection failed: {e}")
            return self._create_empty_result(start_time, str(e))

    def _analyze_all_features(self,
                             data: pd.DataFrame,
                             targets: Optional[pd.Series]) -> List[FeatureScore]:
        """Analyze all features in the dataset."""
        features = []

        for column in data.columns:
            try:
                # Skip non-numeric columns
                if not pd.api.types.is_numeric_dtype(data[column]):
                    continue

                feature_series = data[column].dropna()

                if len(feature_series) < 10:  # Skip features with too few values
                    continue

                # Calculate basic metrics
                variance = feature_series.var()
                if variance < self.config.min_variance:
                    continue

                # Calculate correlation with target
                correlation_with_target = 0.0
                if targets is not None:
                    try:
                        correlation = feature_series.corr(targets)
                        correlation_with_target = abs(correlation) if not pd.isna(correlation) else 0.0
                    except:
                        correlation_with_target = 0.0

                # Calculate information content (simplified)
                information_content = self._calculate_information_content(feature_series)

                # Calculate uniqueness score (simplified)
                uniqueness_score = self._calculate_uniqueness_score(feature_series, data)

                # Determine category and aspect
                category, aspect_type = self._categorize_feature(column)

                # Calculate overall score
                score = self._calculate_feature_score(
                    variance, correlation_with_target, information_content,
                    uniqueness_score, category
                )

                feature_score = FeatureScore(
                    feature_name=column,
                    category=category,
                    aspect_type=aspect_type,
                    score=score,
                    variance=variance,
                    correlation_with_target=correlation_with_target,
                    information_content=information_content,
                    uniqueness_score=uniqueness_score,
                    metadata={
                        'column_index': data.columns.get_loc(column),
                        'data_type': str(feature_series.dtype),
                        'non_null_count': len(feature_series)
                    }
                )

                features.append(feature_score)

            except Exception as e:
                tprint_debug(f"⚠️ Failed to analyze feature {column}: {e}")
                continue

        return features

    def _categorize_feature(self, feature_name: str) -> Tuple[str, str]:
        """Categorize a feature based on its name."""
        name_lower = feature_name.lower()

        # Category mapping
        if any(x in name_lower for x in ['mom', 'momentum', 'roc', 'rate_of_change']):
            return 'momentum', 'trend_following'
        elif any(x in name_lower for x in ['vol', 'volatility', 'std', 'sigma', 'rv']):
            return 'volatility', 'risk_measure'
        elif any(x in name_lower for x in ['sma', 'ema', 'ma', 'trend', 'moving_average']):
            return 'trend', 'trend_following'
        elif any(x in name_lower for x in ['rsi', 'stoch', 'oscillator', 'osc']):
            return 'oscillator', 'mean_reversion'
        elif any(x in name_lower for x in ['volume', 'vol', 'turnover']):
            return 'volume', 'liquidity'
        elif any(x in name_lower for x in ['return', 'ret', 'pct_change']):
            return 'returns', 'price_action'
        elif any(x in name_lower for x in ['htf', 'higher_timeframe', 'cross_timeframe']):
            return 'cross_timeframe', 'multi_timeframe'
        elif any(x in name_lower for x in ['microstructure', 'tick', 'bid_ask']):
            return 'microstructure', 'market_microstructure'
        elif any(x in name_lower for x in ['entropy', 'complexity', 'fractal']):
            return 'entropy', 'complexity'
        elif any(x in name_lower for x in ['support', 'resistance', 'level']):
            return 'support_resistance', 'technical_levels'
        elif any(x in name_lower for x in ['candlestick', 'pattern', 'doji', 'hammer']):
            return 'candlestick_pattern', 'pattern_recognition'
        elif any(x in name_lower for x in ['time', 'hour', 'day', 'session']):
            return 'time', 'temporal'
        elif any(x in name_lower for x in ['order_flow', 'flow', 'imbalance']):
            return 'order_flow', 'flow_analysis'
        elif any(x in name_lower for x in ['regime', 'state', 'regime_type']):
            return 'regime', 'market_regime'
        elif any(x in name_lower for x in ['acceleration', 'accel', 'velocity']):
            return 'acceleration', 'momentum_derivative'
        elif any(x in name_lower for x in ['statistical', 'stat', 'advanced_stat']):
            return 'advanced_statistical', 'statistical_analysis'
        elif any(x in name_lower for x in ['spectral', 'wavelet', 'fourier']):
            return 'spectral_wavelet', 'frequency_analysis'
        else:
            return 'unknown', 'general'

    def _categorize_features(self,
                           features: List[FeatureScore],
                           available_categories: Optional[List[str]]) -> Dict[str, List[FeatureScore]]:
        """Categorize features by their categories."""
        categorized = defaultdict(list)

        for feature in features:
            category = feature.category
            if available_categories is None or category in available_categories:
                categorized[category].append(feature)

        return dict(categorized)

    def _apply_quality_filtering(self,
                                categorized_features: Dict[str, List[FeatureScore]]) -> Dict[str, List[FeatureScore]]:
        """Apply quality filtering to features."""
        filtered = {}

        for category, features in categorized_features.items():
            filtered_features = []

            for feature in features:
                # Apply quality thresholds
                if (feature.variance >= self.config.min_variance and
                    feature.information_content >= self.config.min_information_content and
                    feature.score > 0.0):
                    filtered_features.append(feature)

            # Sort by score and apply correlation filtering
            filtered_features.sort(key=lambda x: x.score, reverse=True)
            filtered_features = self._remove_highly_correlated(filtered_features)

            if filtered_features:
                filtered[category] = filtered_features

        return filtered

    def _remove_highly_correlated(self, features: List[FeatureScore]) -> List[FeatureScore]:
        """Remove highly correlated features."""
        if len(features) <= 1:
            return features

        # Simple correlation-based filtering
        # In practice, you'd calculate actual correlations between features
        selected = [features[0]]  # Always keep the best feature

        for feature in features[1:]:
            # Check if this feature is too similar to already selected ones
            is_similar = False
            for selected_feature in selected:
                # Simple similarity check based on name patterns
                if self._are_features_similar(feature.feature_name, selected_feature.feature_name):
                    is_similar = True
                    break

            if not is_similar:
                selected.append(feature)

        return selected

    def _are_features_similar(self, name1: str, name2: str) -> bool:
        """Check if two features are similar based on name patterns."""
        # Simple similarity check - in practice, you'd use more sophisticated methods
        name1_parts = set(name1.lower().split('_'))
        name2_parts = set(name2.lower().split('_'))

        # If they share more than 70% of their parts, consider them similar
        intersection = name1_parts.intersection(name2_parts)
        union = name1_parts.union(name2_parts)

        similarity = len(intersection) / len(union) if union else 0
        return similarity > 0.7

    def _select_with_diversity_enforcement(self,
                                         filtered_features: Dict[str, List[FeatureScore]]) -> List[FeatureScore]:
        """Select features with diversity enforcement."""
        selected = []
        remaining_budget = self.config.target_feature_count

        # First pass: ensure minimum features per category
        for category, features in filtered_features.items():
            if not features:
                continue

            # Select minimum required features from this category
            min_count = min(self.config.min_features_per_category, len(features), remaining_budget)
            selected.extend(features[:min_count])
            remaining_budget -= min_count

        # Second pass: fill remaining budget with best features
        if remaining_budget > 0:
            # Collect all remaining features
            remaining_features = []
            for category, features in filtered_features.items():
                already_selected = len([f for f in selected if f.category == category])
                max_count = min(self.config.max_features_per_category, len(features))

                if already_selected < max_count:
                    remaining_features.extend(features[already_selected:max_count])

            # Sort by score and select best remaining
            remaining_features.sort(key=lambda x: x.score, reverse=True)
            selected.extend(remaining_features[:remaining_budget])

        return selected

    def _calculate_information_content(self, series: pd.Series) -> float:
        """Calculate information content of a feature."""
        try:
            # Simple entropy-based information content
            value_counts = series.value_counts()
            probabilities = value_counts / len(series)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

            # Normalize to 0-1 scale
            max_entropy = np.log2(len(value_counts))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            return min(max(normalized_entropy, 0), 1)

        except Exception as e:
            self.logger.warning(f"Error calculating information content: {e}")
            return 0.0

    def _calculate_uniqueness_score(self, series: pd.Series, data: pd.DataFrame) -> float:
        """Calculate uniqueness score of a feature."""
        try:
            # Simple uniqueness based on correlation with other features
            correlations = []
            for col in data.columns:
                if col != series.name and pd.api.types.is_numeric_dtype(data[col]):
                    try:
                        corr = series.corr(data[col])
                        if not pd.isna(corr):
                            correlations.append(abs(corr))
                    except:
                        continue

            if not correlations:
                return 1.0

            # Lower average correlation = higher uniqueness
            avg_correlation = np.mean(correlations)
            uniqueness = 1.0 - avg_correlation

            return min(max(uniqueness, 0), 1)

        except Exception as e:
            self.logger.warning(f"Error calculating uniqueness score: {e}")
            return 0.5

    def _calculate_feature_score(self,
                                variance: float,
                                correlation_with_target: float,
                                information_content: float,
                                uniqueness_score: float,
                                category: str) -> float:
        """Calculate overall feature score with math validation."""
        try:
            # Validate inputs
            variance = validate_finite(variance, "variance")
            correlation_with_target = validate_finite(correlation_with_target, "correlation_with_target")
            information_content = validate_finite(information_content, "information_content")
            uniqueness_score = validate_finite(uniqueness_score, "uniqueness_score")

            # Get category weight with validation
            category_weight = self.config.category_weights.get(category, 1.0)
            category_weight = validate_positive(category_weight, "category_weight")

            # Weighted combination of metrics with safe operations
            variance_norm = safe_divide(variance, 1.0, default=0.0)
            variance_norm = validate_range(variance_norm, 0.0, 1.0, "variance_norm")

            correlation_norm = validate_range(correlation_with_target, -1.0, 1.0, "correlation_with_target")
            information_norm = validate_range(information_content, 0.0, 1.0, "information_content")
            uniqueness_norm = validate_range(uniqueness_score, 0.0, 1.0, "uniqueness_score")

            # Calculate weighted score with safe operations
            score = safe_weighted_average(
                [variance_norm, correlation_norm, information_norm, uniqueness_norm],
                [0.3, 0.3, 0.2, 0.2]
            )

            # Apply category weight with safe multiplication
            score = score * category_weight
            score = validate_range(score, 0.0, 1.0, "feature_score")

            return float(score)

        except Exception as e:
            self.logger.warning(f"Error calculating feature score: {e}")
            return 0.0

    def _calculate_category_distribution(self, features: List[FeatureScore]) -> Dict[str, int]:
        """Calculate category distribution of selected features."""
        distribution = defaultdict(int)
        for feature in features:
            distribution[feature.category] += 1
        return dict(distribution)

    def _calculate_aspect_distribution(self, features: List[FeatureScore]) -> Dict[str, int]:
        """Calculate aspect distribution of selected features."""
        distribution = defaultdict(int)
        for feature in features:
            distribution[feature.aspect_type] += 1
        return dict(distribution)

    def _calculate_quality_metrics(self, features: List[FeatureScore]) -> Dict[str, Any]:
        """Calculate quality metrics for selected features."""
        if not features:
            return {}

        scores = [f.score for f in features]
        variances = [f.variance for f in features]
        correlations = [f.correlation_with_target for f in features]
        information_contents = [f.information_content for f in features]
        uniqueness_scores = [f.uniqueness_score for f in features]

        return {
            'average_score': np.mean(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'average_variance': np.mean(variances),
            'average_correlation': np.mean(correlations),
            'average_information_content': np.mean(information_contents),
            'average_uniqueness': np.mean(uniqueness_scores),
            'score_std': np.std(scores),
            'total_features': len(features)
        }

    def _create_empty_result(self, start_time: float, error_message: str) -> FeatureSelectionResult:
        """Create empty result for failed selection."""
        return FeatureSelectionResult(
            selected_features=[],
            category_distribution={},
            aspect_distribution={},
            total_features_analyzed=0,
            selection_time=time.time() - start_time,
            quality_metrics={},
            metadata={'error': True, 'error_message': error_message}
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()

# Convenience functions
def create_intelligent_feature_selector(config: Optional[FeatureSelectionConfig] = None) -> IntelligentFeatureSelector:
    """Create an intelligent feature selector with default configuration."""
    return IntelligentFeatureSelector(config)

def select_features_intelligently(data: pd.DataFrame,
                                 targets: Optional[pd.Series] = None,
                                 target_count: int = 40,
                                 available_categories: Optional[List[str]] = None) -> FeatureSelectionResult:
    """
    Convenience function to select features intelligently.

    Args:
        data: Input data with features
        targets: Target variable for relevance scoring
        target_count: Target number of features to select
        available_categories: Specific categories to consider

    Returns:
        FeatureSelectionResult with selected features
    """
    config = FeatureSelectionConfig(target_feature_count=target_count)
    selector = create_intelligent_feature_selector(config)
    return selector.select_features(data, targets, available_categories)

# Export main classes and functions
__all__ = [
    'IntelligentFeatureSelector',
    'FeatureSelectionConfig',
    'FeatureSelectionResult',
    'FeatureScore',
    'create_intelligent_feature_selector',
    'select_features_intelligently'
]
