"""
Feature Engineering Pipeline for Hybrid NAS-TAS Regime Detection.

Provides comprehensive feature engineering workflow using the existing
feature_generation system with systematic feature selection and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import warnings
from dataclasses import dataclass
import time
from datetime import datetime
from enum import Enum

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
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
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import existing feature generation utilities
try:
    from src.feature_generation.core.feature_generator import FeatureGenerator, FeatureResult, FeatureCategory
    from src.feature_generation.core.factory import get_feature_bank
    from src.feature_generation.core.feature_bank import FeatureBank
    FEATURE_GENERATION_AVAILABLE = True
except ImportError:
    FEATURE_GENERATION_AVAILABLE = False

# Import shared FeatureConfig for compatibility
try:
    from src.training.steps.market_analysis.shared_utils.features import FeatureConfig
    SHARED_FEATURES_AVAILABLE = True
except ImportError:
    SHARED_FEATURES_AVAILABLE = False

# Import existing utilities
try:
    from src.utils.common_operations import (
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
    )
    HARDWARE_UTILS_AVAILABLE = True
except ImportError:
    HARDWARE_UTILS_AVAILABLE = False

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

try:
    from src.utils.ml_common import (
        FeatureSelector, FeatureSelectionConfig,
        UnifiedCrossValidator, perform_cross_validation
    )
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

logger = logging.getLogger(__name__)

class FeatureSelectionMethod(Enum):
    """Feature selection methods."""
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    CHI2 = "chi2"
    F_SCORE = "f_score"
    RECURSIVE = "recursive"
    RFE = "rfe"
    LASSO = "lasso"
    RIDGE = "ridge"
    ELASTIC_NET = "elastic_net"

@dataclass
class FeaturePipelineConfig:
    """Configuration for feature engineering pipeline."""
    # Feature categories to include
    feature_categories: List[FeatureCategory] = None

    # Feature selection
    enable_feature_selection: bool = True
    selection_method: FeatureSelectionMethod = FeatureSelectionMethod.MUTUAL_INFO
    max_features: int = 100
    min_feature_importance: float = 0.01

    # Feature validation
    enable_feature_validation: bool = True
    validation_method: str = "cross_validation"  # "cross_validation", "holdout", "time_series"
    validation_folds: int = 5

    # Feature preprocessing
    enable_normalization: bool = True
    enable_outlier_handling: bool = True
    outlier_threshold: float = 3.0

    # Performance optimization
    use_hardware_acceleration: bool = True
    use_matrix_operations: bool = True
    batch_size: int = 1000
    memory_limit_gb: float = 8.0

    # Feature engineering parameters
    lookback_periods: List[int] = None
    enable_interaction_features: bool = False
    enable_polynomial_features: bool = False
    polynomial_degree: int = 2

    def __post_init__(self):
        if self.feature_categories is None:
            self.feature_categories = [
                FeatureCategory.MOMENTUM,
                FeatureCategory.VOLATILITY,
                FeatureCategory.VOLUME,
                FeatureCategory.TREND
            ]
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 50]

@dataclass
class FeaturePipelineResult:
    """Result from feature engineering pipeline."""
    features: pd.DataFrame
    feature_names: List[str]
    feature_categories: Dict[str, List[str]]
    feature_importance: Dict[str, float]
    selection_info: Dict[str, Any]
    validation_scores: Dict[str, float]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    hardware_optimization_applied: bool = False
    matrix_operations_used: bool = False

class FeatureEngineeringPipeline:
    """Advanced feature engineering pipeline with systematic feature selection and validation."""

    def __init__(self, config: FeaturePipelineConfig):
        """Initialize the feature engineering pipeline.

        Args:
            config: Feature pipeline configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_optimizer = None
        self.cpu_optimizer = None

        if HARDWARE_UTILS_AVAILABLE and config.use_hardware_acceleration:
            try:
                self.hardware_accelerator = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.info("✅ Hardware acceleration initialized for feature engineering")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available: {e}")

        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None

        if MATRIX_OPERATIONS_AVAILABLE and config.use_matrix_operations:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for feature engineering")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available: {e}")

        # Initialize feature calculators
        self.feature_calculators = self._initialize_feature_calculators()

        # Initialize feature selector if available
        self.feature_selector = None
        if ML_COMMON_AVAILABLE and config.enable_feature_selection:
            try:
                selector_config = FeatureSelectionConfig(
                    method=config.selection_method.value,
                    max_features=config.max_features,
                    min_importance=config.min_feature_importance
                )
                self.feature_selector = FeatureSelector(selector_config)
                self.logger.info("✅ Feature selector initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Feature selector not available: {e}")

        self.logger.info("✅ Feature Engineering Pipeline initialized")
        self.logger.info(f"   Feature categories: {[cat.value for cat in config.feature_categories]}")
        self.logger.info(f"   Feature selection: {config.enable_feature_selection}")
        self.logger.info(f"   Feature validation: {config.enable_feature_validation}")

    def _initialize_feature_calculators(self) -> Dict[str, Any]:
        """Initialize feature calculators for different categories."""
        calculators = {}

        if FEATURE_GENERATION_AVAILABLE:
            try:
                # Import the global feature bank and ensure it's properly initialized
                from src.feature_generation.core.feature_bank import get_global_feature_bank, _global_feature_bank

                # Get the global feature bank instance
                self.feature_bank = get_global_feature_bank()

                # Ensure this is the same instance as the global one
                if self.feature_bank is not _global_feature_bank:
                    self.logger.warning("⚠️ Feature bank instance mismatch, using global instance")
                    self.feature_bank = _global_feature_bank

                # Verify the feature bank is properly initialized
                total_generators = len(self.feature_bank.registry.get_all())
                if total_generators == 0:
                    self.logger.error("❌ Feature bank has no generators, this should not happen!")
                    raise ValueError("Feature bank is not properly initialized")

                self.logger.info(f"✅ Feature bank initialized with {total_generators} generators")

                # Map feature categories to their generators
                category_mapping = {
                    'momentum': FeatureCategory.MOMENTUM,
                    'volatility': FeatureCategory.VOLATILITY,
                    'volume': FeatureCategory.VOLUME,
                    'trend': FeatureCategory.TREND,
                    'returns': FeatureCategory.RETURNS,
                    'oscillator': FeatureCategory.OSCILLATOR
                }

                for category_name, category_enum in category_mapping.items():
                    generators = self.feature_bank.get_generators_by_category(category_enum)
                    self.logger.info(f"🔍 {category_name} generators found: {len(generators)}")
                    if generators:
                        calculators[category_name] = generators
                        self.logger.info(f"✅ {category_name} calculators initialized: {len(generators)} generators")
                    else:
                        self.logger.warning(f"⚠️ No generators found for {category_name}")

                self.logger.info(f"✅ Feature calculators initialized: {len(calculators)} categories")
                self.logger.info(f"🔍 Available categories: {list(calculators.keys())}")
            except Exception as e:
                self.logger.warning(f"⚠️ Feature calculator initialization failed: {e}")

        return calculators

    def engineer_features(self, data: pd.DataFrame,
                         target: Optional[pd.Series] = None) -> FeaturePipelineResult:
        """Engineer features using the complete pipeline.

        Args:
            data: Input market data
            target: Optional target variable for feature selection

        Returns:
            FeaturePipelineResult with engineered features
        """
        print(f"🔍 DEBUG: engineer_features called with data shape {data.shape}")
        tprint_info("Starting feature engineering pipeline")
        tprint_debug(f"Input shape: {data.shape}")
        tprint_debug(f"Feature categories: {len(self.config.feature_categories)}")

        with tprint_timer("Feature Engineering Pipeline"):
            start_time = time.time()

            try:
                self.logger.info("🔧 Starting feature engineering pipeline")
                self.logger.info(f"   Input shape: {data.shape}")
                self.logger.info(f"   Feature categories: {len(self.config.feature_categories)}")

                # Step 1: Generate base features
                tprint_info("Step 1: Generating base features")
                tprint_progress(1, 7, "Feature Engineering Pipeline")
                base_features = self._generate_base_features(data)
                tprint_success(f"Base features generated: {base_features.shape}")

                # Step 2: Generate interaction features if enabled
                if self.config.enable_interaction_features:
                    tprint_info("Step 2: Generating interaction features")
                    tprint_progress(2, 7, "Feature Engineering Pipeline")
                    interaction_features = self._generate_interaction_features(base_features)
                    base_features = pd.concat([base_features, interaction_features], axis=1)
                    tprint_success(f"Interaction features added: {base_features.shape}")

                # Step 3: Generate polynomial features if enabled
                if self.config.enable_polynomial_features:
                    tprint_info("Step 3: Generating polynomial features")
                    tprint_progress(3, 7, "Feature Engineering Pipeline")
                    polynomial_features = self._generate_polynomial_features(base_features)
                    base_features = pd.concat([base_features, polynomial_features], axis=1)
                    tprint_success(f"Polynomial features added: {base_features.shape}")

                # Step 4: Handle outliers if enabled
                if self.config.enable_outlier_handling:
                    tprint_info("Step 4: Handling outliers")
                    tprint_progress(4, 7, "Feature Engineering Pipeline")
                    base_features = self._handle_outliers(base_features)
                    tprint_success("Outliers handled")

                # Step 5: Normalize features if enabled
                if self.config.enable_normalization:
                    tprint_info("Step 5: Normalizing features")
                    tprint_progress(5, 7, "Feature Engineering Pipeline")
                    base_features = self._normalize_features(base_features)
                    tprint_success("Features normalized")

                # Step 6: Feature selection if enabled
                selected_features = base_features
                selection_info = {}
                if self.config.enable_feature_selection and target is not None:
                    tprint_info("Step 6: Selecting features")
                    tprint_progress(6, 7, "Feature Engineering Pipeline")
                    selected_features, selection_info = self._select_features(base_features, target)
                    tprint_success(f"Features selected: {selected_features.shape}")

                # Step 7: Feature validation if enabled
                validation_scores = {}
                if self.config.enable_feature_validation and target is not None:
                    tprint_info("Step 7: Validating features")
                    tprint_progress(7, 7, "Feature Engineering Pipeline")
                    validation_scores = self._validate_features(selected_features, target)
                    tprint_success("Features validated")

                # Step 8: Calculate feature importance
                feature_importance = self._calculate_feature_importance(selected_features, target)

                # Step 9: Categorize features
                feature_categories = self._categorize_features(selected_features)

                processing_time = time.time() - start_time

                self.logger.info(f"✅ Feature engineering completed in {processing_time:.2f}s")
                self.logger.info(f"   Generated features: {selected_features.shape[1]}")
                self.logger.info(f"   Selected features: {len(selection_info.get('selected_features', []))}")

                return FeaturePipelineResult(
                features=selected_features,
                feature_names=list(selected_features.columns),
                feature_categories=feature_categories,
                feature_importance=feature_importance,
                selection_info=selection_info,
                validation_scores=validation_scores,
                processing_time=processing_time,
                success=True,
                hardware_optimization_applied=self.hardware_accelerator is not None,
                matrix_operations_used=self.matrix_ops is not None
                )

            except Exception as e:
                processing_time = time.time() - start_time
                self.logger.error(f"❌ Feature engineering failed: {e}")

                return FeaturePipelineResult(
                    features=pd.DataFrame(),
                    feature_names=[],
                    feature_categories={},
                    feature_importance={},
                    selection_info={},
                    validation_scores={},
                    processing_time=processing_time,
                    success=False,
                    error_message=str(e)
                )

    def _generate_base_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate base features for all categories.

        Args:
            data: Input market data

        Returns:
            DataFrame with base features
        """
        print(f"🔍 DEBUG: _generate_base_features called with data shape {data.shape}")
        try:
            all_features = pd.DataFrame(index=data.index)

            self.logger.info(f"🔍 Available feature calculators: {list(self.feature_calculators.keys())}")
            self.logger.info(f"🔍 Config feature categories: {[cat.value for cat in self.config.feature_categories]}")

            for category in self.config.feature_categories:
                self.logger.info(f"🔍 Processing category: {category.value}")
                if category.value in self.feature_calculators:
                    generators = self.feature_calculators[category.value]
                    self.logger.info(f"🔍 Found {len(generators)} generators for {category.value}")
                    category_features = self._generate_category_features(data, category, generators)

                    if category_features is not None and not category_features.empty:
                        all_features = pd.concat([all_features, category_features], axis=1)
                        self.logger.info(f"✅ Generated {category.value} features: {category_features.shape[1]}")
                    else:
                        self.logger.warning(f"⚠️ No features generated for {category.value}")
                else:
                    self.logger.warning(f"⚠️ No calculators found for {category.value}")

            return all_features

        except Exception as e:
            self.logger.error(f"❌ Base feature generation failed: {e}")
            return pd.DataFrame()

    def _generate_category_features(self, data: pd.DataFrame,
                                  category: FeatureCategory,
                                  generators: List[FeatureGenerator]) -> Optional[pd.DataFrame]:
        """Generate features for a specific category.

        Args:
            data: Input market data
            category: Feature category
            generators: List of feature generators for this category

        Returns:
            DataFrame with category features
        """
        try:
            features = pd.DataFrame(index=data.index)

            if not generators:
                self.logger.warning(f"⚠️ No generators available for category: {category}")
                return features

            # Generate features using all generators for this category
            for i, generator in enumerate(generators):
                try:
                    self.logger.info(f"🔍 Generating features with generator: {generator.config.name}")
                    self.logger.info(f"🔍 Data shape: {data.shape}, columns: {list(data.columns)}")
                    result = generator.generate(data)
                    self.logger.info(f"🔍 Generator {generator.config.name} result type: {type(result)}")
                    if result:
                        self.logger.info(f"🔍 Generator {generator.config.name} result has data: {hasattr(result, 'data')}")
                        if hasattr(result, 'data'):
                            self.logger.info(f"🔍 Generator {generator.config.name} data shape: {result.data.shape if not result.data.empty else 'empty'}")
                    if result and hasattr(result, 'data') and not result.data.empty:
                        # Convert Series to DataFrame and add features with category prefix to avoid naming conflicts
                        feature_name = f"{category.value}_{result.name}"
                        category_features = pd.DataFrame({feature_name: result.data})
                        features = pd.concat([features, category_features], axis=1)
                        self.logger.info(f"✅ Generated 1 {category.value} feature: {feature_name}")
                    else:
                        self.logger.warning(f"⚠️ Generator {generator.config.name} returned empty result")
                except Exception as e:
                    self.logger.warning(f"⚠️ Generator {generator.config.name} failed for {category}: {e}")
                    import traceback
                    self.logger.warning(f"⚠️ Generator {generator.config.name} traceback: {traceback.format_exc()}")
                    continue

            return features

        except Exception as e:
            self.logger.error(f"❌ Category feature generation failed for {category}: {e}")
            return None

    def _generate_momentum_features(self, data: pd.DataFrame, calculator: Any) -> pd.DataFrame:
        """Generate momentum features."""
        try:
            features = pd.DataFrame(index=data.index)

            for lookback in self.config.lookback_periods:
                if hasattr(calculator, 'calculate_momentum'):
                    momentum = calculator.calculate_momentum(data, lookback)
                    if momentum is not None:
                        features[f'momentum_{lookback}'] = momentum
                else:
                    # Fallback calculation
                    if 'close' in data.columns:
                        momentum = data['close'].pct_change(lookback)
                        features[f'momentum_{lookback}'] = momentum

                # RSI-like momentum
                if 'close' in data.columns:
                    returns = data['close'].pct_change()
                    gains = returns.where(returns > 0, 0)
                    losses = -returns.where(returns < 0, 0)

                    avg_gain = gains.rolling(lookback).mean()
                    avg_loss = losses.rolling(lookback).mean()

                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    features[f'rsi_{lookback}'] = rsi

            return features

        except Exception as e:
            self.logger.warning(f"⚠️ Momentum feature generation failed: {e}")
            return pd.DataFrame()

    def _generate_volatility_features(self, data: pd.DataFrame, calculator: Any) -> pd.DataFrame:
        """Generate volatility features."""
        try:
            features = pd.DataFrame(index=data.index)

            for lookback in self.config.lookback_periods:
                if hasattr(calculator, 'calculate_volatility'):
                    volatility = calculator.calculate_volatility(data, lookback)
                    if volatility is not None:
                        features[f'volatility_{lookback}'] = volatility
                else:
                    # Fallback calculation
                    if 'close' in data.columns:
                        returns = data['close'].pct_change()
                        volatility = returns.rolling(lookback).std()
                        features[f'volatility_{lookback}'] = volatility

                # GARCH-like volatility
                if 'close' in data.columns:
                    returns = data['close'].pct_change()
                    squared_returns = returns ** 2
                    vol_of_vol = squared_returns.rolling(lookback).std()
                    features[f'vol_of_vol_{lookback}'] = vol_of_vol

            return features

        except Exception as e:
            self.logger.warning(f"⚠️ Volatility feature generation failed: {e}")
            return pd.DataFrame()

    def _generate_volume_features(self, data: pd.DataFrame, calculator: Any) -> pd.DataFrame:
        """Generate volume features."""
        try:
            features = pd.DataFrame(index=data.index)

            if 'volume' in data.columns:
                for lookback in self.config.lookback_periods:
                    # Volume moving average
                    vol_ma = data['volume'].rolling(lookback).mean()
                    features[f'volume_ma_{lookback}'] = vol_ma

                    # Volume ratio
                    vol_ratio = data['volume'] / vol_ma
                    features[f'volume_ratio_{lookback}'] = vol_ratio

                    # Volume volatility
                    vol_vol = data['volume'].rolling(lookback).std()
                    features[f'volume_vol_{lookback}'] = vol_vol

                    # Price-volume relationship
                    if 'close' in data.columns:
                        price_change = data['close'].pct_change()
                        vol_price_corr = data['volume'].rolling(lookback).corr(price_change)
                        features[f'vol_price_corr_{lookback}'] = vol_price_corr

            return features

        except Exception as e:
            self.logger.warning(f"⚠️ Volume feature generation failed: {e}")
            return pd.DataFrame()

    def _generate_trend_features(self, data: pd.DataFrame, calculator: Any) -> pd.DataFrame:
        """Generate trend features."""
        try:
            features = pd.DataFrame(index=data.index)

            if 'close' in data.columns:
                for lookback in self.config.lookback_periods:
                    # Moving averages
                    ma = data['close'].rolling(lookback).mean()
                    features[f'ma_{lookback}'] = ma

                    # Price relative to moving average
                    price_ma_ratio = data['close'] / ma
                    features[f'price_ma_ratio_{lookback}'] = price_ma_ratio

                    # Trend strength (slope)
                    trend_slope = data['close'].rolling(lookback).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                    )
                    features[f'trend_slope_{lookback}'] = trend_slope

                    # Trend direction
                    trend_direction = np.where(data['close'] > ma, 1, -1)
                    features[f'trend_direction_{lookback}'] = trend_direction

            return features

        except Exception as e:
            self.logger.warning(f"⚠️ Trend feature generation failed: {e}")
            return pd.DataFrame()

    def _generate_returns_features(self, data: pd.DataFrame, calculator: Any) -> pd.DataFrame:
        """Generate returns features."""
        try:
            features = pd.DataFrame(index=data.index)

            if 'close' in data.columns:
                # Basic returns
                returns = data['close'].pct_change()
                features['returns'] = returns

                # Log returns
                log_returns = np.log(data['close'] / data['close'].shift(1))
                features['log_returns'] = log_returns

                # Rolling returns for different periods
                for lookback in self.config.lookback_periods:
                    rolling_returns = data['close'].pct_change(lookback)
                    features[f'returns_{lookback}'] = rolling_returns

                    # Rolling return volatility
                    return_vol = returns.rolling(lookback).std()
                    features[f'return_vol_{lookback}'] = return_vol

                    # Rolling return skewness
                    return_skew = returns.rolling(lookback).skew()
                    features[f'return_skew_{lookback}'] = return_skew

                    # Rolling return kurtosis
                    return_kurt = returns.rolling(lookback).kurt()
                    features[f'return_kurt_{lookback}'] = return_kurt

            return features

        except Exception as e:
            self.logger.warning(f"⚠️ Returns feature generation failed: {e}")
            return pd.DataFrame()

    def _generate_oscillator_features(self, data: pd.DataFrame, calculator: Any) -> pd.DataFrame:
        """Generate oscillator features."""
        try:
            features = pd.DataFrame(index=data.index)

            if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
                for lookback in self.config.lookback_periods:
                    # Stochastic oscillator
                    lowest_low = data['low'].rolling(lookback).min()
                    highest_high = data['high'].rolling(lookback).max()
                    stoch = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
                    features[f'stoch_{lookback}'] = stoch

                    # Williams %R
                    williams_r = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
                    features[f'williams_r_{lookback}'] = williams_r

                    # Commodity Channel Index (CCI)
                    typical_price = (data['high'] + data['low'] + data['close']) / 3
                    sma_tp = typical_price.rolling(lookback).mean()
                    mad = typical_price.rolling(lookback).apply(lambda x: np.mean(np.abs(x - x.mean())))
                    cci = (typical_price - sma_tp) / (0.015 * mad)
                    features[f'cci_{lookback}'] = cci

            return features

        except Exception as e:
            self.logger.warning(f"⚠️ Oscillator feature generation failed: {e}")
            return pd.DataFrame()

    def _generate_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between existing features."""
        try:
            interaction_features = pd.DataFrame(index=features.index)

            # Select numeric columns for interactions
            numeric_cols = features.select_dtypes(include=[np.number]).columns

            # Generate pairwise interactions for top features
            top_features = numeric_cols[:10]  # Limit to top 10 features

            for i, col1 in enumerate(top_features):
                for col2 in top_features[i+1:]:
                    # Multiplication interaction
                    interaction_name = f"{col1}_x_{col2}"
                    interaction_features[interaction_name] = features[col1] * features[col2]

                    # Division interaction (with safe division)
                    ratio_name = f"{col1}_div_{col2}"
                    interaction_features[ratio_name] = np.where(
                        np.abs(features[col2]) > 1e-10,
                        features[col1] / features[col2],
                        0
                    )

            return interaction_features

        except Exception as e:
            self.logger.warning(f"⚠️ Interaction feature generation failed: {e}")
            return pd.DataFrame()

    def _generate_polynomial_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial features."""
        try:
            polynomial_features = pd.DataFrame(index=features.index)

            # Select numeric columns for polynomial features
            numeric_cols = features.select_dtypes(include=[np.number]).columns

            # Generate polynomial features for top features
            top_features = numeric_cols[:5]  # Limit to top 5 features

            for col in top_features:
                for degree in range(2, self.config.polynomial_degree + 1):
                    poly_name = f"{col}_poly_{degree}"
                    polynomial_features[poly_name] = features[col] ** degree

            return polynomial_features

        except Exception as e:
            self.logger.warning(f"⚠️ Polynomial feature generation failed: {e}")
            return pd.DataFrame()

    def _handle_outliers(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in features."""
        try:
            processed_features = features.copy()

            for col in processed_features.columns:
                if processed_features[col].dtype in ['float64', 'int64']:
                    # Calculate outlier bounds
                    Q1 = processed_features[col].quantile(0.25)
                    Q3 = processed_features[col].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - self.config.outlier_threshold * IQR
                    upper_bound = Q3 + self.config.outlier_threshold * IQR

                    # Cap outliers
                    processed_features[col] = np.clip(
                        processed_features[col], lower_bound, upper_bound
                    )

            return processed_features

        except Exception as e:
            self.logger.warning(f"⚠️ Outlier handling failed: {e}")
            return features

    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using z-score normalization."""
        try:
            normalized_features = features.copy()

            for col in normalized_features.columns:
                if normalized_features[col].dtype in ['float64', 'int64']:
                    mean_val = normalized_features[col].mean()
                    std_val = normalized_features[col].std()

                    if std_val > 0:
                        normalized_features[col] = (normalized_features[col] - mean_val) / std_val
                    else:
                        normalized_features[col] = 0

            return normalized_features

        except Exception as e:
            self.logger.warning(f"⚠️ Feature normalization failed: {e}")
            return features

    def _select_features(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Select features using the configured method.

        Args:
            features: Input features
            target: Target variable

        Returns:
            Tuple of (selected_features, selection_info)
        """
        try:
            if not self.feature_selector:
                return features, {'method': 'none', 'selected_features': list(features.columns)}

            # Perform feature selection
            selected_features = self.feature_selector.select_features(features, target)

            selection_info = {
                'method': self.config.selection_method.value,
                'selected_features': list(selected_features.columns),
                'n_selected': len(selected_features.columns),
                'n_original': len(features.columns),
                'selection_ratio': len(selected_features.columns) / len(features.columns)
            }

            return selected_features, selection_info

        except Exception as e:
            self.logger.warning(f"⚠️ Feature selection failed: {e}")
            return features, {'method': 'failed', 'error': str(e)}

    def _validate_features(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """Validate features using cross-validation.

        Args:
            features: Input features
            target: Target variable

        Returns:
            Validation scores
        """
        try:
            if not ML_COMMON_AVAILABLE:
                return {}

            # Perform cross-validation
            cv_result = perform_cross_validation(
                features, target,
                n_folds=self.config.validation_folds,
                method=self.config.validation_method
            )

            return {
                'cv_score': cv_result.get('cv_score', 0.0),
                'cv_std': cv_result.get('cv_std', 0.0),
                'validation_method': self.config.validation_method
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Feature validation failed: {e}")
            return {'error': str(e)}

    def _calculate_feature_importance(self, features: pd.DataFrame,
                                    target: Optional[pd.Series]) -> Dict[str, float]:
        """Calculate feature importance.

        Args:
            features: Input features
            target: Optional target variable

        Returns:
            Feature importance scores
        """
        try:
            if target is None:
                # Return uniform importance if no target
                return {col: 1.0 / len(features.columns) for col in features.columns}

            # Calculate correlation-based importance
            importance = {}
            for col in features.columns:
                if features[col].dtype in ['float64', 'int64']:
                    corr = abs(features[col].corr(target))
                    importance[col] = corr if not np.isnan(corr) else 0.0
                else:
                    importance[col] = 0.0

            # Normalize importance scores
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {k: v / total_importance for k, v in importance.items()}

            return importance

        except Exception as e:
            self.logger.warning(f"⚠️ Feature importance calculation failed: {e}")
            return {col: 0.0 for col in features.columns}

    def _categorize_features(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize features by type.

        Args:
            features: Input features

        Returns:
            Dictionary mapping categories to feature names
        """
        try:
            categories = {
                'momentum': [],
                'volatility': [],
                'volume': [],
                'trend': [],
                'returns': [],
                'oscillator': [],
                'interaction': [],
                'polynomial': [],
                'other': []
            }

            for col in features.columns:
                col_lower = col.lower()
                if 'momentum' in col_lower or 'rsi' in col_lower:
                    categories['momentum'].append(col)
                elif 'volatility' in col_lower or 'vol' in col_lower:
                    categories['volatility'].append(col)
                elif 'volume' in col_lower:
                    categories['volume'].append(col)
                elif 'trend' in col_lower or 'ma_' in col_lower:
                    categories['trend'].append(col)
                elif 'return' in col_lower:
                    categories['returns'].append(col)
                elif 'stoch' in col_lower or 'williams' in col_lower or 'cci' in col_lower:
                    categories['oscillator'].append(col)
                elif '_x_' in col_lower or '_div_' in col_lower:
                    categories['interaction'].append(col)
                elif 'poly_' in col_lower:
                    categories['polynomial'].append(col)
                else:
                    categories['other'].append(col)

            return categories

        except Exception as e:
            self.logger.warning(f"⚠️ Feature categorization failed: {e}")
            return {'other': list(features.columns)}

def create_feature_pipeline(config: Optional[FeaturePipelineConfig] = None) -> FeatureEngineeringPipeline:
    """Create a feature engineering pipeline instance.

    Args:
        config: Optional feature pipeline configuration

    Returns:
        FeatureEngineeringPipeline instance
    """
    if config is None:
        config = FeaturePipelineConfig()
    return FeatureEngineeringPipeline(config)

def quick_feature_engineering(data: pd.DataFrame,
                            target: Optional[pd.Series] = None,
                            categories: Optional[List[FeatureCategory]] = None) -> FeaturePipelineResult:
    """Quick feature engineering with default settings.

    Args:
        data: Input market data
        target: Optional target variable
        categories: Optional feature categories

    Returns:
        FeaturePipelineResult
    """
    if categories is None:
        categories = [
            FeatureCategory.MOMENTUM,
            FeatureCategory.VOLATILITY,
            FeatureCategory.VOLUME,
            FeatureCategory.TREND
        ]

    config = FeaturePipelineConfig(
        feature_categories=categories,
        enable_feature_selection=target is not None,
        enable_feature_validation=target is not None
    )

    pipeline = FeatureEngineeringPipeline(config)
    return pipeline.engineer_features(data, target)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)

        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
