"""
Scaling and Normalization for Unified Data-Driven Pipeline.

This module provides comprehensive scaling and normalization capabilities
for consistent feature preprocessing across the pipeline with VectorBT integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler,
    QuantileTransformer, PowerTransformer, Normalizer
)
from sklearn.compose import ColumnTransformer
import logging

# Import VectorBT components
try:
    from src.features_common.vectorbt_extensions.unified_manager import UnifiedVectorizationManager
    from src.features_common.transforms.vectorbt_scaler import VectorBTScaler
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

try:
    from src.utils.tprint import tprint, tprint_error, tprint_warning, tprint_success, tprint_debug
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

class ScalingNormalizer:
    """
    Comprehensive scaling and normalization for the unified pipeline.

    Provides multiple scaling strategies with automatic selection based on
    data characteristics and feature types. Integrates with VectorBT for
    optimized performance when available.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the scaling normalizer."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize VectorBT components if available
        self.vectorbt_manager = None
        self.vectorbt_scaler = None

        if VECTORBT_AVAILABLE:
            try:
                self.vectorbt_manager = UnifiedVectorizationManager()
                self.vectorbt_scaler = VectorBTScaler()
                tprint_success("✅ VectorBT components initialized for scaling")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT initialization failed: {e}")

        # Traditional scaling strategies
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'maxabs': MaxAbsScaler(),
            'quantile': QuantileTransformer(output_distribution='normal'),
            'power': PowerTransformer(method='yeo-johnson'),
            'normalizer': Normalizer()
        }

        # Configuration
        self.default_strategy = self.config.get('default_strategy', 'robust')
        self.auto_select = self.config.get('auto_select', True)
        self.handle_outliers = self.config.get('handle_outliers', True)
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        self.use_vectorbt = self.config.get('use_vectorbt', VECTORBT_AVAILABLE)

        # Fitted scalers and feature mappings
        self.fitted_scalers = {}
        self.feature_mappings = {}
        self.scaling_stats = {}

        tprint_success("✅ ScalingNormalizer initialized")

    def analyze_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data characteristics to recommend scaling strategy.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary with data characteristics and recommendations
        """
        characteristics = {
            'numeric_features': [],
            'categorical_features': [],
            'outlier_features': [],
            'skewed_features': [],
            'recommended_strategy': self.default_strategy,
            'feature_stats': {},
            'vectorbt_suitable': False
        }

        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                characteristics['numeric_features'].append(col)

                # Calculate statistics
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    stats = {
                        'mean': col_data.mean(),
                        'std': col_data.std(),
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'skewness': col_data.skew(),
                        'kurtosis': col_data.kurtosis(),
                        'outlier_count': self._count_outliers(col_data),
                        'total_count': len(col_data)
                    }
                    characteristics['feature_stats'][col] = stats

                    # Check for outliers
                    if stats['outlier_count'] > len(col_data) * 0.05:  # 5% outliers
                        characteristics['outlier_features'].append(col)

                    # Check for skewness
                    if abs(stats['skewness']) > 1.0:
                        characteristics['skewed_features'].append(col)
            else:
                characteristics['categorical_features'].append(col)

        # Check if data is suitable for VectorBT optimization
        if (self.use_vectorbt and VECTORBT_AVAILABLE and
            len(characteristics['numeric_features']) > 0 and
            len(data) > 1000):  # VectorBT is more efficient for larger datasets
            characteristics['vectorbt_suitable'] = True

        # Recommend scaling strategy based on characteristics
        if characteristics['outlier_features']:
            characteristics['recommended_strategy'] = 'robust'
        elif characteristics['skewed_features']:
            characteristics['recommended_strategy'] = 'quantile'
        else:
            characteristics['recommended_strategy'] = 'standard'

        tprint_info(f"📊 Data analysis: {len(characteristics['numeric_features'])} numeric, "
                   f"{len(characteristics['outlier_features'])} with outliers, "
                   f"{len(characteristics['skewed_features'])} skewed, "
                   f"VectorBT suitable: {characteristics['vectorbt_suitable']}")

        return characteristics

    def _count_outliers(self, data: pd.Series, method: str = 'iqr') -> int:
        """Count outliers in a data series."""
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return ((data < lower_bound) | (data > upper_bound)).sum()
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            return (z_scores > self.outlier_threshold).sum()
        else:
            return 0

    def select_scaling_strategy(self, feature_name: str, feature_stats: Dict[str, Any]) -> str:
        """
        Select appropriate scaling strategy for a feature.

        Args:
            feature_name: Name of the feature
            feature_stats: Statistics of the feature

        Returns:
            Selected scaling strategy name
        """
        outlier_ratio = feature_stats['outlier_count'] / feature_stats.get('total_count', 1)
        skewness = abs(feature_stats['skewness'])

        # Strategy selection logic
        if outlier_ratio > 0.1:  # More than 10% outliers
            return 'robust'
        elif skewness > 2.0:  # Highly skewed
            return 'quantile'
        elif skewness > 1.0:  # Moderately skewed
            return 'power'
        elif feature_stats['min'] < 0:  # Has negative values
            return 'standard'
        else:  # Normal distribution, positive values
            return 'minmax'

    def fit_transform(self, data: pd.DataFrame,
                     strategy: Optional[str] = None,
                     feature_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit and transform data using appropriate scaling strategies.

        Args:
            data: Input DataFrame
            strategy: Specific scaling strategy (if None, auto-select)
            feature_list: Specific features to scale (if None, scale all numeric)

        Returns:
            Scaled DataFrame
        """
        tprint_info("🔧 Starting scaling and normalization")

        # Analyze data characteristics
        characteristics = self.analyze_data_characteristics(data)

        # Determine features to scale
        if feature_list is None:
            feature_list = characteristics['numeric_features']

        if not feature_list:
            tprint_info("ℹ️ No numeric features to scale")
            return data

        # Try VectorBT optimization if suitable
        if (characteristics['vectorbt_suitable'] and
            self.vectorbt_scaler is not None and
            strategy in ['standard', 'robust', 'minmax']):

            try:
                tprint_info("🚀 Using VectorBT for optimized scaling")
                scaled_data = self._vectorbt_fit_transform(data, feature_list, strategy)
                if scaled_data is not None:
                    tprint_success("✅ VectorBT scaling completed")
                    return scaled_data
                else:
                    tprint_warning("⚠️ VectorBT scaling failed, falling back to traditional methods")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT scaling error: {e}, using traditional methods")

        # Traditional scaling approach
        scaled_data = data.copy()

        # Process each feature
        for feature in feature_list:
            if feature not in data.columns:
                tprint_warning(f"⚠️ Feature {feature} not found in data")
                continue

            try:
                # Select scaling strategy
                if strategy is None and self.auto_select:
                    feature_stats = characteristics['feature_stats'].get(feature, {})
                    selected_strategy = self.select_scaling_strategy(feature, feature_stats)
                else:
                    selected_strategy = strategy or self.default_strategy

                # Apply scaling
                scaled_feature = self._apply_scaling(
                    data[feature], feature, selected_strategy, fit=True
                )

                if scaled_feature is not None:
                    scaled_data[feature] = scaled_feature
                    tprint_success(f"✅ Scaled {feature} using {selected_strategy}")

            except Exception as e:
                tprint_error(f"❌ Error scaling feature {feature}: {e}")
                continue

        tprint_success(f"✅ Scaling completed: {len(feature_list)} features processed")
        return scaled_data

    def _vectorbt_fit_transform(self, data: pd.DataFrame,
                               feature_list: List[str],
                               strategy: str) -> Optional[pd.DataFrame]:
        """Use VectorBT for optimized scaling."""
        try:
            if not self.vectorbt_scaler:
                return None

            # Prepare data for VectorBT
            numeric_data = data[feature_list].copy()

            # Apply VectorBT scaling
            if strategy == 'standard':
                scaled_data = self.vectorbt_scaler.zscore_normalize(numeric_data)
            elif strategy == 'robust':
                scaled_data = self.vectorbt_scaler.robust_normalize(numeric_data)
            elif strategy == 'minmax':
                scaled_data = self.vectorbt_scaler.minmax_normalize(numeric_data)
            else:
                return None

            # Store fitted scalers for inverse transform
            for feature in feature_list:
                self.fitted_scalers[feature] = {
                    'scaler': self.vectorbt_scaler,
                    'strategy': strategy,
                    'vectorbt': True
                }

            # Create result DataFrame with original structure
            result = data.copy()
            result[feature_list] = scaled_data

            return result

        except Exception as e:
            tprint_error(f"❌ VectorBT scaling failed: {e}")
            return None

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using previously fitted scalers.

        Args:
            data: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        if not self.fitted_scalers:
            tprint_warning("⚠️ No fitted scalers found, returning original data")
            return data

        tprint_info("🔄 Applying fitted scaling transformations")

        # Create a copy to avoid modifying original data
        transformed_data = data.copy()

        for feature, scaler_info in self.fitted_scalers.items():
            if feature not in data.columns:
                tprint_warning(f"⚠️ Feature {feature} not found in data")
                continue

            try:
                scaler = scaler_info['scaler']
                strategy = scaler_info['strategy']
                is_vectorbt = scaler_info.get('vectorbt', False)

                # Apply transformation
                if is_vectorbt and hasattr(scaler, f'{strategy}_normalize'):
                    # Use VectorBT transformation
                    feature_data = data[[feature]].copy()
                    transformed_feature = getattr(scaler, f'{strategy}_normalize')(feature_data)
                    transformed_data[feature] = transformed_feature[feature]
                else:
                    # Use traditional transformation
                    transformed_feature = self._apply_scaling(
                        data[feature], feature, strategy, fit=False
                    )
                    if transformed_feature is not None:
                        transformed_data[feature] = transformed_feature

                tprint_debug(f"✅ Transformed {feature} using {strategy}")

            except Exception as e:
                tprint_error(f"❌ Error transforming feature {feature}: {e}")
                continue

        tprint_success("✅ Transformation completed")
        return transformed_data

    def _apply_scaling(self, data: pd.Series, feature_name: str,
                      strategy: str, fit: bool = True) -> Optional[pd.Series]:
        """Apply specific scaling strategy to a feature."""
        try:
            # Get scaler
            if strategy not in self.scalers:
                tprint_warning(f"⚠️ Unknown scaling strategy: {strategy}")
                return None

            scaler = self.scalers[strategy]

            # Handle missing values
            data_clean = data.dropna()
            if len(data_clean) == 0:
                tprint_warning(f"⚠️ No valid data for feature {feature_name}")
                return None

            # Fit and transform
            if fit:
                scaled_values = scaler.fit_transform(data_clean.values.reshape(-1, 1)).flatten()
                # Store fitted scaler
                self.fitted_scalers[feature_name] = {
                    'scaler': scaler,
                    'strategy': strategy,
                    'vectorbt': False
                }
            else:
                # Use previously fitted scaler
                if feature_name in self.fitted_scalers:
                    scaler = self.fitted_scalers[feature_name]['scaler']
                    scaled_values = scaler.transform(data_clean.values.reshape(-1, 1)).flatten()
                else:
                    tprint_warning(f"⚠️ No fitted scaler found for feature {feature_name}")
                    return None

            # Create Series with original index
            scaled_series = pd.Series(index=data.index, dtype=float)
            scaled_series.loc[data_clean.index] = scaled_values

            return scaled_series

        except Exception as e:
            tprint_error(f"❌ Error applying {strategy} scaling to {feature_name}: {e}")
            return None

    def inverse_transform(self, data: pd.DataFrame, feature_name: str) -> pd.Series:
        """Inverse transform a scaled feature back to original scale."""
        if feature_name not in self.fitted_scalers:
            tprint_warning(f"⚠️ No fitted scaler found for feature {feature_name}")
            return data[feature_name]

        try:
            scaler_info = self.fitted_scalers[feature_name]
            scaler = scaler_info['scaler']
            is_vectorbt = scaler_info.get('vectorbt', False)

            # Handle missing values
            data_clean = data[feature_name].dropna()
            if len(data_clean) == 0:
                return data[feature_name]

            # Inverse transform
            if is_vectorbt and hasattr(scaler, 'inverse_transform'):
                # Use VectorBT inverse transform
                feature_data = data_clean.values.reshape(-1, 1)
                original_values = scaler.inverse_transform(feature_data).flatten()
            else:
                # Use traditional inverse transform
                original_values = scaler.inverse_transform(data_clean.values.reshape(-1, 1)).flatten()

            # Create Series with original index
            original_series = pd.Series(index=data[feature_name].index, dtype=float)
            original_series.loc[data_clean.index] = original_values

            return original_series

        except Exception as e:
            tprint_error(f"❌ Error in inverse transform for {feature_name}: {e}")
            return data[feature_name]

    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get summary of scaling operations."""
        summary = {
            'total_features_scaled': len(self.fitted_scalers),
            'scaling_strategies_used': {},
            'feature_details': {},
            'vectorbt_usage': 0,
            'traditional_usage': 0
        }

        for feature, scaler_info in self.fitted_scalers.items():
            strategy = scaler_info['strategy']
            is_vectorbt = scaler_info.get('vectorbt', False)

            summary['scaling_strategies_used'][strategy] = summary['scaling_strategies_used'].get(strategy, 0) + 1

            if is_vectorbt:
                summary['vectorbt_usage'] += 1
            else:
                summary['traditional_usage'] += 1

            summary['feature_details'][feature] = {
                'strategy': strategy,
                'scaler_type': type(scaler_info['scaler']).__name__,
                'vectorbt': is_vectorbt
            }

        return summary

    def reset(self):
        """Reset all fitted scalers and mappings."""
        self.fitted_scalers = {}
        self.feature_mappings = {}
        self.scaling_stats = {}
        tprint_success("✅ Scaling normalizer reset")
