"""
Shared Feature Engineering Utilities

This module provides common feature engineering capabilities that can be used by both
NAS and TAS systems. It includes technical indicators, feature selection, and
dimensionality reduction methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Try to import technical analysis libraries
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

try:
    from sklearn.feature_selection import (
        mutual_info_regression, mutual_info_classif,
        f_regression, f_classif, SelectKBest, SelectPercentile,
        RFE, RFECV
    )
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Technical indicators
    enable_technical_indicators: bool = True
    technical_indicators: List[str] = field(default_factory=lambda: [
        'sma', 'ema', 'rsi', 'macd', 'bollinger', 'stoch', 'williams_r', 'cci', 'atr'
    ])

    # Price features
    enable_price_features: bool = True
    price_features: List[str] = field(default_factory=lambda: [
        'returns', 'log_returns', 'price_ratios', 'price_changes'
    ])

    # Volume features
    enable_volume_features: bool = True
    volume_features: List[str] = field(default_factory=lambda: [
        'volume_ratios', 'volume_ma', 'volume_oscillator'
    ])

    # Volatility features
    enable_volatility_features: bool = True
    volatility_features: List[str] = field(default_factory=lambda: [
        'rolling_volatility', 'garch_volatility', 'volatility_ratios'
    ])

    # Momentum features
    enable_momentum_features: bool = True
    momentum_features: List[str] = field(default_factory=lambda: [
        'momentum', 'rate_of_change', 'momentum_oscillator'
    ])

    # Trend features
    enable_trend_features: bool = True
    trend_features: List[str] = field(default_factory=lambda: [
        'trend_direction', 'trend_strength', 'trend_consistency'
    ])

    # Regime features
    enable_regime_features: bool = True
    regime_features: List[str] = field(default_factory=lambda: [
        'regime_labels', 'regime_probabilities', 'regime_transitions'
    ])

    # Interaction features
    enable_interaction_features: bool = True
    interaction_features: List[str] = field(default_factory=lambda: [
        'price_volume_interactions', 'volatility_momentum_interactions'
    ])

    # Polynomial features
    enable_polynomial_features: bool = False
    polynomial_degree: int = 2

    # Cross-timeframe features
    enable_cross_timeframe_features: bool = True
    timeframes: List[str] = field(default_factory=lambda: ['1h', '4h', '1d'])

    # Feature selection
    enable_feature_selection: bool = True
    feature_selection_method: str = "mutual_info"  # "mutual_info", "f_score", "rfe", "embedded"
    max_features: int = 100
    feature_importance_threshold: float = 0.01

    # Dimensionality reduction
    enable_dimensionality_reduction: bool = False
    reduction_method: str = "pca"  # "pca", "ica", "lda"
    n_components: int = 50

    # Scaling
    enable_scaling: bool = True
    scaling_method: str = "standard"  # "standard", "minmax", "robust"

    # Window sizes
    short_window: int = 5
    medium_window: int = 20
    long_window: int = 50

    # Random state
    random_state: int = 42

@dataclass
class FeatureEngineeringResult:
    """Result from feature engineering operations."""

    # Enhanced features
    enhanced_features: np.ndarray
    feature_names: List[str]
    feature_importance: Dict[str, float]
    selected_features: List[str]

    # Feature statistics
    feature_statistics: Dict[str, Any]
    correlation_matrix: np.ndarray

    # Engineering info
    original_feature_count: int
    enhanced_feature_count: int
    selected_feature_count: int

    # Success indicators
    success: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

class BaseFeatureEngineer(ABC):
    """Abstract base class for feature engineers."""

    def __init__(self, config: FeatureConfig):
        """Initialize feature engineer."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def generate_features(self, data: np.ndarray,
                         feature_names: Optional[List[str]] = None) -> FeatureEngineeringResult:
        """Generate enhanced features."""
        pass

class TechnicalIndicatorEngineer(BaseFeatureEngineer):
    """Engineer for technical indicators."""

    def generate_features(self, data: np.ndarray,
                         feature_names: Optional[List[str]] = None) -> FeatureEngineeringResult:
        """Generate technical indicator features."""
        try:
            if not self.config.enable_technical_indicators:
                return self._create_empty_result(data, feature_names)

            self.logger.info("📊 Generating technical indicators...")

            # Convert to DataFrame for easier manipulation
            if len(data.shape) == 1:
                df = pd.DataFrame({'price': data})
            else:
                if feature_names is None:
                    feature_names = [f'feature_{i}' for i in range(data.shape[1])]
                df = pd.DataFrame(data, columns=feature_names)

            enhanced_features = []
            new_feature_names = []

            # Simple Moving Average
            if 'sma' in self.config.technical_indicators:
                for window in [self.config.short_window, self.config.medium_window, self.config.long_window]:
                    sma = df['price'].rolling(window=window).mean() if 'price' in df.columns else df.iloc[:, 0].rolling(window=window).mean()
                    enhanced_features.append(sma.values)
                    new_feature_names.append(f'sma_{window}')

            # Exponential Moving Average
            if 'ema' in self.config.technical_indicators:
                for window in [self.config.short_window, self.config.medium_window]:
                    ema = df['price'].ewm(span=window).mean() if 'price' in df.columns else df.iloc[:, 0].ewm(span=window).mean()
                    enhanced_features.append(ema.values)
                    new_feature_names.append(f'ema_{window}')

            # RSI (Relative Strength Index)
            if 'rsi' in self.config.technical_indicators:
                rsi = self._calculate_rsi(df['price'] if 'price' in df.columns else df.iloc[:, 0])
                enhanced_features.append(rsi)
                new_feature_names.append('rsi')

            # MACD
            if 'macd' in self.config.technical_indicators:
                macd_line, macd_signal, macd_histogram = self._calculate_macd(df['price'] if 'price' in df.columns else df.iloc[:, 0])
                enhanced_features.extend([macd_line, macd_signal, macd_histogram])
                new_feature_names.extend(['macd_line', 'macd_signal', 'macd_histogram'])

            # Bollinger Bands
            if 'bollinger' in self.config.technical_indicators:
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['price'] if 'price' in df.columns else df.iloc[:, 0])
                enhanced_features.extend([bb_upper, bb_middle, bb_lower])
                new_feature_names.extend(['bb_upper', 'bb_middle', 'bb_lower'])

            # Combine with original features
            if enhanced_features:
                enhanced_array = np.column_stack([data] + enhanced_features)
                all_feature_names = (feature_names or [f'feature_{i}' for i in range(data.shape[1])]) + new_feature_names
            else:
                enhanced_array = data
                all_feature_names = feature_names or [f'feature_{i}' for i in range(data.shape[1])]

            return FeatureEngineeringResult(
                enhanced_features=enhanced_array,
                feature_names=all_feature_names,
                feature_importance={},
                selected_features=all_feature_names,
                feature_statistics=self._calculate_feature_statistics(enhanced_array),
                correlation_matrix=np.corrcoef(enhanced_array.T) if enhanced_array.shape[1] > 1 else np.array([[1.0]]),
                original_feature_count=data.shape[1],
                enhanced_feature_count=enhanced_array.shape[1],
                selected_feature_count=enhanced_array.shape[1],
                success=True
            )

        except Exception as e:
            self.logger.error(f"❌ Technical indicator generation failed: {e}")
            return self._create_empty_result(data, feature_names, error_message=str(e))

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate RSI."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50).values
        except Exception:
            return np.full(len(prices), 50.0)

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal).mean()
            macd_histogram = macd_line - macd_signal
            return macd_line.fillna(0).values, macd_signal.fillna(0).values, macd_histogram.fillna(0).values
        except Exception:
            zeros = np.zeros(len(prices))
            return zeros, zeros, zeros

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        try:
            sma = rolling_mean(prices, window=period) if VECTORBT_AVAILABLE and len(prices) > 1000 else prices.rolling(window=period).mean()
            std = rolling_std(prices, window=period) if VECTORBT_AVAILABLE and len(prices) > 1000 else prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper.fillna(prices.mean()).values, sma.fillna(prices.mean()).values, lower.fillna(prices.mean()).values
        except Exception:
            mean_price = prices.mean()
            return np.full(len(prices), mean_price), np.full(len(prices), mean_price), np.full(len(prices), mean_price)

    def _calculate_feature_statistics(self, features: np.ndarray) -> Dict[str, Any]:
        """Calculate feature statistics."""
        try:
            return {
                'mean': np.mean(features, axis=0).tolist(),
                'std': np.std(features, axis=0).tolist(),
                'min': np.min(features, axis=0).tolist(),
                'max': np.max(features, axis=0).tolist(),
                'skewness': self._calculate_skewness(features).tolist(),
                'kurtosis': self._calculate_kurtosis(features).tolist()
            }
        except Exception:
            return {}

    def _calculate_skewness(self, features: np.ndarray) -> np.ndarray:
        """Calculate skewness for each feature."""
        try:
            from scipy.stats import skew
            return np.array([skew(features[:, i]) for i in range(features.shape[1])])
        except Exception:
            return np.zeros(features.shape[1])

    def _calculate_kurtosis(self, features: np.ndarray) -> np.ndarray:
        """Calculate kurtosis for each feature."""
        try:
            from scipy.stats import kurtosis
            return np.array([kurtosis(features[:, i]) for i in range(features.shape[1])])
        except Exception:
            return np.zeros(features.shape[1])

    def _create_empty_result(self, data: np.ndarray, feature_names: Optional[List[str]],
                           error_message: Optional[str] = None) -> FeatureEngineeringResult:
        """Create empty result."""
        return FeatureEngineeringResult(
            enhanced_features=data,
            feature_names=feature_names or [f'feature_{i}' for i in range(data.shape[1])],
            feature_importance={},
            selected_features=feature_names or [f'feature_{i}' for i in range(data.shape[1])],
            feature_statistics={},
            correlation_matrix=np.eye(data.shape[1]),
            original_feature_count=data.shape[1],
            enhanced_feature_count=data.shape[1],
            selected_feature_count=data.shape[1],
            success=error_message is None,
            error_message=error_message
        )

class FeatureSelector:
    """Feature selection utilities."""

    def __init__(self, config: FeatureConfig):
        """Initialize feature selector."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def select_features(self, X: np.ndarray, y: np.ndarray,
                       feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
        """Select features using specified method."""
        try:
            if not self.config.enable_feature_selection:
                return X, feature_names or [f'feature_{i}' for i in range(X.shape[1])], {}

            self.logger.info(f"🔍 Selecting features using {self.config.feature_selection_method}...")

            if self.config.feature_selection_method == "mutual_info":
                return self._mutual_info_selection(X, y, feature_names)
            elif self.config.feature_selection_method == "f_score":
                return self._f_score_selection(X, y, feature_names)
            elif self.config.feature_selection_method == "rfe":
                return self._rfe_selection(X, y, feature_names)
            else:
                self.logger.warning(f"⚠️ Unknown feature selection method: {self.config.feature_selection_method}")
                return X, feature_names or [f'feature_{i}' for i in range(X.shape[1])], {}

        except Exception as e:
            self.logger.error(f"❌ Feature selection failed: {e}")
            return X, feature_names or [f'feature_{i}' for i in range(X.shape[1])], {}

    def _mutual_info_selection(self, X: np.ndarray, y: np.ndarray,
                              feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
        """Select features using mutual information."""
        try:
            if not SKLEARN_AVAILABLE:
                self.logger.warning("⚠️ Scikit-learn not available for mutual information selection")
                return X, feature_names or [f'feature_{i}' for i in range(X.shape[1])], {}

            # Determine if classification or regression
            is_classification = len(np.unique(y)) < 10

            if is_classification:
                scores = mutual_info_classif(X, y)
            else:
                scores = mutual_info_regression(X, y)

            # Select top features
            n_features = min(self.config.max_features, X.shape[1])
            selector = SelectKBest(k=n_features)
            X_selected = selector.fit_transform(X, y)

            # Get selected feature names and importance
            selected_indices = selector.get_support(indices=True)
            selected_names = [feature_names[i] for i in selected_indices] if feature_names else [f'feature_{i}' for i in selected_indices]
            importance = {name: float(scores[i]) for name, i in zip(selected_names, selected_indices)}

            return X_selected, selected_names, importance

        except Exception as e:
            self.logger.error(f"❌ Mutual information selection failed: {e}")
            return X, feature_names or [f'feature_{i}' for i in range(X.shape[1])], {}

    def _f_score_selection(self, X: np.ndarray, y: np.ndarray,
                          feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
        """Select features using F-score."""
        try:
            if not SKLEARN_AVAILABLE:
                self.logger.warning("⚠️ Scikit-learn not available for F-score selection")
                return X, feature_names or [f'feature_{i}' for i in range(X.shape[1])], {}

            # Determine if classification or regression
            is_classification = len(np.unique(y)) < 10

            if is_classification:
                scores, _ = f_classif(X, y)
            else:
                scores, _ = f_regression(X, y)

            # Select top features
            n_features = min(self.config.max_features, X.shape[1])
            selector = SelectKBest(k=n_features)
            X_selected = selector.fit_transform(X, y)

            # Get selected feature names and importance
            selected_indices = selector.get_support(indices=True)
            selected_names = [feature_names[i] for i in selected_indices] if feature_names else [f'feature_{i}' for i in selected_indices]
            importance = {name: float(scores[i]) for name, i in zip(selected_names, selected_indices)}

            return X_selected, selected_names, importance

        except Exception as e:
            self.logger.error(f"❌ F-score selection failed: {e}")
            return X, feature_names or [f'feature_{i}' for i in range(X.shape[1])], {}

    def _rfe_selection(self, X: np.ndarray, y: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
        """Select features using Recursive Feature Elimination."""
        try:
            if not SKLEARN_AVAILABLE:
                self.logger.warning("⚠️ Scikit-learn not available for RFE selection")
                return X, feature_names or [f'feature_{i}' for i in range(X.shape[1])], {}

            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.linear_model import LinearRegression, LogisticRegression

            # Determine if classification or regression
            is_classification = len(np.unique(y)) < 10

            if is_classification:
                estimator = LogisticRegression(random_state=self.config.random_state)
            else:
                estimator = LinearRegression()

            # Use RFE
            n_features = min(self.config.max_features, X.shape[1])
            selector = RFE(estimator, n_features_to_select=n_features)
            X_selected = selector.fit_transform(X, y)

            # Get selected feature names and importance
            selected_indices = selector.get_support(indices=True)
            selected_names = [feature_names[i] for i in selected_indices] if feature_names else [f'feature_{i}' for i in selected_indices]
            importance = {name: float(selector.ranking_[i]) for name, i in zip(selected_names, selected_indices)}

            return X_selected, selected_names, importance

        except Exception as e:
            self.logger.error(f"❌ Error in RFE feature selection: {e}")
            return X, feature_names or [f'feature_{i}' for i in range(X.shape[1])], {}

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

class DimensionalityReducer:
    """Dimensionality reduction utilities."""

    def __init__(self, config: FeatureConfig):
        """Initialize dimensionality reducer."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def reduce_dimensions(self, X: np.ndarray,
                         feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """Reduce dimensions using specified method."""
        try:
            if not self.config.enable_dimensionality_reduction:
                return X, feature_names or [f'feature_{i}' for i in range(X.shape[1])]

            if not SKLEARN_AVAILABLE:
                self.logger.warning("⚠️ Scikit-learn not available for dimensionality reduction")
                return X, feature_names or [f'feature_{i}' for i in range(X.shape[1])]

            self.logger.info(f"📉 Reducing dimensions using {self.config.reduction_method}...")

            if self.config.reduction_method == "pca":
                return self._pca_reduction(X, feature_names)
            else:
                self.logger.warning(f"⚠️ Unknown reduction method: {self.config.reduction_method}")
                return X, feature_names or [f'feature_{i}' for i in range(X.shape[1])]

        except Exception as e:
            self.logger.error(f"❌ Dimensionality reduction failed: {e}")
            return X, feature_names or [f'feature_{i}' for i in range(X.shape[1])]

    def _pca_reduction(self, X: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """Reduce dimensions using PCA."""
        try:
            n_components = min(self.config.n_components, X.shape[1])
            pca = PCA(n_components=n_components, random_state=self.config.random_state)
            X_reduced = pca.fit_transform(X)

            # Create new feature names
            new_feature_names = [f'pca_{i}' for i in range(n_components)]

            self.logger.info(f"✅ PCA reduction completed: {X.shape[1]} -> {X_reduced.shape[1]} features")
            self.logger.info(f"   Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

            return X_reduced, new_feature_names

        except Exception as e:
            self.logger.error(f"❌ PCA reduction failed: {e}")
            return X, feature_names or [f'feature_{i}' for i in range(X.shape[1])]

class FeatureScaler:
    """Feature scaling utilities."""

    def __init__(self, config: FeatureConfig):
        """Initialize feature scaler."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features using specified method."""
        try:
            if not self.config.enable_scaling:
                return X

            if not SKLEARN_AVAILABLE:
                self.logger.warning("⚠️ Scikit-learn not available for feature scaling")
                return X

            self.logger.info(f"📏 Scaling features using {self.config.scaling_method}...")

            if self.config.scaling_method == "standard":
                scaler = StandardScaler()
            elif self.config.scaling_method == "minmax":
                scaler = MinMaxScaler()
            else:
                self.logger.warning(f"⚠️ Unknown scaling method: {self.config.scaling_method}")
                return X

            X_scaled = scaler.fit_transform(X)
            self.logger.info("✅ Feature scaling completed")

            return X_scaled

        except Exception as e:
            self.logger.error(f"❌ Feature scaling failed: {e}")
            return X

class UnifiedFeatureEngineer:
    """Unified feature engineering system."""

    def __init__(self, config: FeatureConfig):
        """Initialize unified feature engineer."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.technical_engineer = TechnicalIndicatorEngineer(config)
        self.feature_selector = FeatureSelector(config)
        self.dimensionality_reducer = DimensionalityReducer(config)
        self.feature_scaler = FeatureScaler(config)

        self.logger.info("✅ Unified Feature Engineer initialized")

    def engineer_features(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                         feature_names: Optional[List[str]] = None) -> FeatureEngineeringResult:
        """Perform comprehensive feature engineering."""
        try:
            self.logger.info("🔧 Starting comprehensive feature engineering...")

            # Step 1: Technical indicators
            if self.config.enable_technical_indicators:
                result = self.technical_engineer.generate_features(X, feature_names)
                X_enhanced = result.enhanced_features
                enhanced_names = result.feature_names
            else:
                X_enhanced = X
                enhanced_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]

            # Step 2: Feature selection
            if self.config.enable_feature_selection and y is not None:
                X_selected, selected_names, importance = self.feature_selector.select_features(
                    X_enhanced, y, enhanced_names
                )
            else:
                X_selected = X_enhanced
                selected_names = enhanced_names
                importance = {}

            # Step 3: Dimensionality reduction
            if self.config.enable_dimensionality_reduction:
                X_reduced, reduced_names = self.dimensionality_reducer.reduce_dimensions(
                    X_selected, selected_names
                )
            else:
                X_reduced = X_selected
                reduced_names = selected_names

            # Step 4: Feature scaling
            if self.config.enable_scaling:
                X_final = self.feature_scaler.scale_features(X_reduced)
            else:
                X_final = X_reduced

            # Calculate final statistics
            final_statistics = self._calculate_final_statistics(X_final)
            correlation_matrix = np.corrcoef(X_final.T) if X_final.shape[1] > 1 else np.array([[1.0]])

            self.logger.info(f"✅ Feature engineering completed")
            self.logger.info(f"   Original features: {X.shape[1]}")
            self.logger.info(f"   Enhanced features: {X_enhanced.shape[1]}")
            self.logger.info(f"   Selected features: {X_selected.shape[1]}")
            self.logger.info(f"   Final features: {X_final.shape[1]}")

            return FeatureEngineeringResult(
                enhanced_features=X_final,
                feature_names=reduced_names,
                feature_importance=importance,
                selected_features=reduced_names,
                feature_statistics=final_statistics,
                correlation_matrix=correlation_matrix,
                original_feature_count=X.shape[1],
                enhanced_feature_count=X_enhanced.shape[1],
                selected_feature_count=X_final.shape[1],
                success=True
            )

        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {e}")
            return FeatureEngineeringResult(
                enhanced_features=X,
                feature_names=feature_names or [f'feature_{i}' for i in range(X.shape[1])],
                feature_importance={},
                selected_features=feature_names or [f'feature_{i}' for i in range(X.shape[1])],
                feature_statistics={},
                correlation_matrix=np.eye(X.shape[1]),
                original_feature_count=X.shape[1],
                enhanced_feature_count=X.shape[1],
                selected_feature_count=X.shape[1],
                success=False,
                error_message=str(e)
            )

    def _calculate_final_statistics(self, features: np.ndarray) -> Dict[str, Any]:
        """Calculate final feature statistics."""
        try:
            return {
                'mean': np.mean(features, axis=0).tolist(),
                'std': np.std(features, axis=0).tolist(),
                'min': np.min(features, axis=0).tolist(),
                'max': np.max(features, axis=0).tolist(),
                'shape': features.shape,
                'memory_usage_mb': features.nbytes / (1024 * 1024)
            }
        except Exception:
            return {}

# Convenience functions
def create_unified_feature_engineer(config: Optional[FeatureConfig] = None) -> UnifiedFeatureEngineer:
    """Create unified feature engineer instance."""
    if config is None:
        config = FeatureConfig()
    return UnifiedFeatureEngineer(config)

def quick_feature_engineering(X: np.ndarray, y: Optional[np.ndarray] = None,
                             feature_names: Optional[List[str]] = None,
                             enable_technical_indicators: bool = True,
                             enable_feature_selection: bool = True,
                             max_features: int = 100) -> FeatureEngineeringResult:
    """Quick feature engineering with default settings."""
    config = FeatureConfig(
        enable_technical_indicators=enable_technical_indicators,
        enable_feature_selection=enable_feature_selection,
        max_features=max_features
    )

    engineer = create_unified_feature_engineer(config)
    return engineer.engineer_features(X, y, feature_names)

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
